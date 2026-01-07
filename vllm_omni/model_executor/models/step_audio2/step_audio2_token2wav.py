import io
import os
from collections.abc import Iterable

import numpy as np
import onnxruntime
import s3tokenizer
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi

# Import from step-audio2 pip package (installed with: pip install -e /path/to/Step-Audio2-main)
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    STEP_AUDIO2_DEFAULT_PROMPT_WAV,
)


def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
    """Perform fade_in_out in tensor style"""
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


class StepAudio2Token2WavCore(nn.Module):
    """Core Token2Wav model - handles token to waveform conversion"""

    def __init__(self, model_path: str, float16: bool = False, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.float16 = float16
        self.device = torch.device(device)

        # Lazy loading: models will be loaded on first use
        # This avoids loading models during __init__, which happens before vLLM's
        # GPU memory check, causing memory errors even with low gpu_memory_utilization
        self._models_loaded = False
        self._audio_tokenizer = None
        self._spk_model = None
        self._flow = None
        self._hift = None
        self._speech_window = None

        # Cache for prompt processing
        self.cache = {}

        # Stream configuration
        self.mel_cache_len = 8  # 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)  # 50hz mel -> 24kHz wave

        # Streaming cache
        self.hift_cache_dict = {}
        self.stream_cache = None

    def _ensure_models_loaded(self):
        """Lazy load models on first use"""
        if self._models_loaded:
            return

        from vllm.logger import init_logger

        logger = init_logger(__name__)

        logger.info(f"[Token2Wav] Starting lazy model loading from: {self.model_path}")

        # Load audio tokenizer (ONNX)
        logger.info("[Token2Wav] Loading audio tokenizer (ONNX)...")
        self._audio_tokenizer = (
            s3tokenizer.load_model(f"{self.model_path}/speech_tokenizer_v2_25hz.onnx").to(self.device).eval()
        )
        logger.info("[Token2Wav] Audio tokenizer loaded")

        # Load speaker embedding model (ONNX)
        logger.info("[Token2Wav] Loading speaker embedding model (ONNX)...")
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self._spk_model = onnxruntime.InferenceSession(
            f"{self.model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"]
        )
        logger.info("[Token2Wav] Speaker embedding model loaded")

        # Load flow model (CFM)
        logger.info("[Token2Wav] Loading flow model (CFM)...")
        # Now using step-audio2 pip package, cosyvoice2 is directly available
        with open(f"{self.model_path}/flow.yaml") as f:
            configs = load_hyperpyyaml(f)
        self._flow = configs["flow"]
        if self.float16:
            self._flow.half()
        self._flow.load_state_dict(
            torch.load(f"{self.model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True
        )
        self._flow.to(self.device).eval()
        logger.info("[Token2Wav] Flow model loaded")

        # Load HiFT Generator (vocoder)
        logger.info("[Token2Wav] Loading HiFT Generator (vocoder)...")
        self._hift = HiFTGenerator()
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(f"{self.model_path}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self._hift.load_state_dict(hift_state_dict, strict=True)
        self._hift.to(self.device).eval()
        logger.info("[Token2Wav] HiFT Generator loaded")

        # Load speech window
        logger.info("[Token2Wav] Loading speech window...")
        self._speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(self.device)

        self._models_loaded = True
        logger.info("[Token2Wav] All models loaded successfully")

    @property
    def audio_tokenizer(self):
        """Lazy-loaded audio tokenizer"""
        self._ensure_models_loaded()
        return self._audio_tokenizer

    @property
    def spk_model(self):
        """Lazy-loaded speaker model"""
        self._ensure_models_loaded()
        return self._spk_model

    @property
    def flow(self):
        """Lazy-loaded flow model"""
        self._ensure_models_loaded()
        return self._flow

    @property
    def hift(self):
        """Lazy-loaded HiFT vocoder"""
        self._ensure_models_loaded()
        return self._hift

    @property
    def speech_window(self):
        """Lazy-loaded speech window"""
        self._ensure_models_loaded()
        return self._speech_window

    def _prepare_prompt(self, prompt_wav: str):
        """Prepare prompt audio for conditioning"""
        # Load and process audio for tokenization
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device)
        )

        # Extract speaker embedding
        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(
            self.spk_model.run(None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0],
            device=self.device,
        )

        # Load prompt mel spectrogram
        audio, sample_rate = torchaudio.load(prompt_wav, backend="soundfile")
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).to(self.device)
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=self.device)
        prompt_mels = torch.nn.functional.pad(
            prompt_mels,
            (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]),
            mode="replicate",
        )

        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def forward(
        self, generated_speech_tokens: list, prompt_wav: str, return_bytes: bool = True
    ) -> torch.Tensor | bytes:
        """
        Convert audio tokens to waveform

        Args:
            generated_speech_tokens: List of audio token IDs (0-6560)
            prompt_wav: Path to prompt wav file for speaker conditioning
            return_bytes: If True, return WAV bytes; else return tensor

        Returns:
            Audio waveform as bytes or tensor
        """
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        logger.info(
            f"[Token2WavCore] forward called with {len(generated_speech_tokens)} tokens, prompt_wav: {prompt_wav}"
        )

        # Cache prompt processing
        if prompt_wav not in self.cache:
            logger.info(f"[Token2WavCore] Preparing prompt (not in cache): {prompt_wav}")
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
            logger.info("[Token2WavCore] Prompt prepared and cached")
        else:
            logger.info(f"[Token2WavCore] Using cached prompt: {prompt_wav}")
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        # Convert token list to tensor
        logger.info("[Token2WavCore] Converting tokens to tensor...")
        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        generated_speech_tokens_lens = torch.tensor(
            [generated_speech_tokens.shape[1]], dtype=torch.int32, device=self.device
        )

        # Generate mel spectrogram using flow model
        logger.info("[Token2WavCore] Generating mel spectrogram using flow model...")
        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(
                generated_speech_tokens,
                generated_speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                prompt_mels_lens,
                spk_emb,
                10,
            )
        logger.info(f"[Token2WavCore] Mel spectrogram generated, shape: {mel.shape}")

        # Generate waveform using HiFT vocoder
        logger.info("[Token2WavCore] Generating waveform using HiFT vocoder...")
        wav, _ = self.hift(speech_feat=mel)
        logger.info(f"[Token2WavCore] Waveform generated, shape: {wav.shape}")

        if return_bytes:
            output = io.BytesIO()
            torchaudio.save(output, wav.cpu(), sample_rate=24000, format="wav")
            return output.getvalue()
        else:
            return wav

    def set_stream_cache(self, prompt_wav: str):
        """Initialize streaming cache for a prompt"""
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        self.stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, prompt_speech_tokens[:, :3]], dim=1), prompt_mels, spk_emb, n_timesteps=10
        )

        # Initialize HiFT cache
        self.hift_cache_dict = dict(
            mel=torch.zeros(1, prompt_mels.shape[2], 0, device=self.device),
            source=torch.zeros(1, 1, 0, device=self.device),
            speech=torch.zeros(1, 0, device=self.device),
        )

    def stream(self, generated_speech_tokens: list, prompt_wav: str, last_chunk: bool = False) -> bytes:
        """
        Stream audio generation chunk by chunk

        Args:
            generated_speech_tokens: List of audio tokens for this chunk
            prompt_wav: Path to prompt wav file
            last_chunk: Whether this is the last chunk

        Returns:
            PCM audio bytes for this chunk
        """
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)

        if self.stream_cache is None:
            raise ValueError("stream_cache is not set. Call set_stream_cache() first.")

        # Generate mel chunk
        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens,
                spk=spk_emb,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=10,
            )

        # Manage cache size
        if self.stream_cache["estimator_att_cache"].shape[4] > (prompt_mels.shape[1] + 100):
            self.stream_cache["estimator_att_cache"] = torch.cat(
                [
                    self.stream_cache["estimator_att_cache"][:, :, :, :, : prompt_mels.shape[1]],
                    self.stream_cache["estimator_att_cache"][:, :, :, :, -100:],
                ],
                dim=4,
            )

        # Vocoder processing with cache
        hift_cache_mel = self.hift_cache_dict["mel"]
        hift_cache_source = self.hift_cache_dict["source"]
        hift_cache_speech = self.hift_cache_dict["speech"]
        mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)

        speech, source = self.hift(mel, hift_cache_source)

        # Smooth overlap
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        # Update cache
        self.hift_cache_dict = dict(
            mel=mel[..., -self.mel_cache_len :].clone().detach(),
            source=source[:, :, -self.source_cache_len :].clone().detach(),
            speech=speech[:, -self.source_cache_len :].clone().detach(),
        )

        if not last_chunk:
            speech = speech[:, : -self.source_cache_len]

        # Convert to PCM bytes
        wav_np = speech.cpu().numpy()
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype("<i2")  # 16-bit little-endian PCM
        pcm_bytes = wav_int16.tobytes()
        return pcm_bytes


class StepAudio2Token2WavForConditionalGeneration(nn.Module, SupportsPP):
    """vLLM-compatible wrapper for Step-Audio2 Token2Wav"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "token2wav.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        # Get model path from config
        # Assuming model_path is stored in config or passed via model_config
        model_path = getattr(self.config, "token2wav_path", None)
        if model_path is None:
            # Fallback: try to construct from model name
            model_name_or_path = vllm_config.model_config.model
            model_path = f"{model_name_or_path}/token2wav"

        float16 = getattr(self.config, "token2wav_float16", False)

        # Get device from vllm_config
        device = "cuda"
        if hasattr(vllm_config, "device_config") and vllm_config.device_config:
            device = str(vllm_config.device_config.device)

        # Initialize core Token2Wav model
        self.token2wav = StepAudio2Token2WavCore(model_path=model_path, float16=float16, device=device)

        # Mark that this model has multimodal outputs (required by vLLM-Omni framework)
        # This tells the runner to parse the return value as OmniOutput
        self.have_multimodal_outputs = True

        # vLLM compatibility
        self.make_empty_intermediate_tensors = lambda: None

    def get_language_model(self) -> torch.nn.Module:
        return self.token2wav

    @property
    def sampler(self):
        return Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional: dict | None = None,
        **kwargs,
    ) -> torch.Tensor | bytes:
        """
        Forward pass for Token2Wav

        Args:
            input_ids: Audio token IDs (from stage input processor)
            positions: Position IDs (unused for Token2Wav)
            intermediate_tensors: Intermediate tensors (unused)
            inputs_embeds: Input embeddings (unused)
            additional: Additional information dict (deprecated, use kwargs instead)
            **kwargs: Other kwargs including:
                - request_ids: List of request IDs
                - request_token_spans: List of (start, end) token spans per request
                - additional_information_by_req_id: Dict[request_id, additional_info]
                - runtime_additional_information: Runtime additional info (list or dict)

        Returns:
            Waveform as bytes or tensor

        Raises:
            ValueError: If prompt_wav is not provided
        """
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        logger.info("[Token2Wav] Forward called - starting processing")
        logger.info(f"[Token2Wav] Available kwargs keys: {list(kwargs.keys())}")
        input_ids_shape = input_ids.shape if isinstance(input_ids, torch.Tensor) else type(input_ids)
        logger.info(f"[Token2Wav] Input IDs shape: {input_ids_shape}")

        # Handle profile/dummy run mode
        # Profile mode detection: input_ids is small (< 50 tokens) or all zeros
        if isinstance(input_ids, torch.Tensor):
            is_profile_mode = input_ids.numel() < 50 or input_ids.sum().item() == 0
        else:
            is_profile_mode = len(input_ids) < 50 if hasattr(input_ids, "__len__") else True

        if is_profile_mode:
            logger.info("[Token2Wav] Detected profile/dummy run mode - returning dummy OmniOutput")
            # Determine device
            device = torch.device("cpu")
            if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.numel() > 0:
                device = inputs_embeds.device
            elif isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
                device = input_ids.device

            # Return dummy OmniOutput for profile run (consistent with normal inference)
            dummy_audio = torch.zeros(1000, device=device, dtype=torch.float32)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": dummy_audio,  # Key must match end2end.py access pattern
                },
            )

        logger.info("[Token2Wav] Normal inference mode - processing audio generation")

        # Handle input_ids shape: [batch_size, seq_len] or [seq_len]
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids_list = input_ids.cpu().tolist()
            else:
                input_ids_list = input_ids[0].cpu().tolist()
        else:
            input_ids_list = input_ids if isinstance(input_ids, list) else [input_ids]

        logger.info(f"[Token2Wav] Received {len(input_ids_list)} tokens total")

        audio_tokens = input_ids_list  # All tokens are audio tokens (no encoding)
        prompt_wav = os.environ.get("STEP_AUDIO2_DEFAULT_PROMPT_WAV", STEP_AUDIO2_DEFAULT_PROMPT_WAV)
        logger.info(f"[Token2Wav] Using default prompt_wav: {prompt_wav}")
        logger.info(f"[Token2Wav] Audio tokens: {len(audio_tokens)} tokens")

        # Verify prompt_wav file exists
        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"Token2Wav: prompt_wav file not found: {prompt_wav}")

        # Generate waveform (return tensor, not bytes - framework expects tensor)
        waveform_tensor = self.token2wav(audio_tokens, prompt_wav=prompt_wav, return_bytes=False)

        logger.info(f"[Token2Wav] Generated waveform tensor, shape: {waveform_tensor.shape}")

        # Squeeze to 1D if needed (framework/script expects [samples] not [1, samples])
        if waveform_tensor.dim() == 2 and waveform_tensor.shape[0] == 1:
            waveform_tensor = waveform_tensor.squeeze(0)

        # Move to CPU before returning (critical for cross-process communication)
        waveform_tensor_cpu = waveform_tensor.detach().cpu().contiguous()

        logger.info(f"[Token2Wav] Final output tensor shape: {waveform_tensor_cpu.shape}")
        logger.info(f"[Token2Wav] Output tensor device: {waveform_tensor_cpu.device}")
        logger.info("[Token2Wav] Returning waveform to framework...")

        return OmniOutput(
            text_hidden_states=None,  # No text output
            multimodal_outputs={
                "model_outputs": waveform_tensor_cpu,  # Audio waveform tensor
            },
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # Token2Wav outputs waveform, not logits
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Token2Wav weights are loaded from files in __init__, not from HF checkpoint
        # Return all parameter and buffer names to indicate they're already loaded
        loaded_params = set()
        # Add all parameters
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        # Add all buffers
        for name, _ in self.token2wav.named_buffers():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params

    def load_weights_without_buffers(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Token2Wav weights are loaded from files in __init__, not from HF checkpoint
        # Return only parameter names (no buffers)
        loaded_params = set()
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params
