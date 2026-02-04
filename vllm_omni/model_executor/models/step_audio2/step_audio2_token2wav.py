import io
import os
from collections.abc import Iterable

import onnxruntime
import s3tokenizer
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
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
    DEFAULT_TOKEN2WAV_CONFIG,
    STEP_AUDIO2_DEFAULT_PROMPT_WAV,
)


class StepAudio2Token2WavCore(nn.Module):
    """Core Token2Wav model - handles token to waveform conversion"""

    def __init__(
        self,
        model_path: str,
        float16: bool = False,
        device: str = "cuda",
        n_timesteps: int = DEFAULT_TOKEN2WAV_CONFIG.n_timesteps,
    ):
        super().__init__()
        self.model_path = model_path
        self.float16 = float16
        self.device = torch.device(device)
        self.n_timesteps = n_timesteps

        self._models_loaded = False
        self._audio_tokenizer = None
        self._spk_model = None
        self._flow = None
        self._hift = None

        self.cache = {}

    def _ensure_models_loaded(self):
        """Lazy load models on first use"""
        if self._models_loaded:
            return

        from vllm.logger import init_logger

        logger = init_logger(__name__)
        logger.info(f"Loading Token2Wav models from: {self.model_path}")

        self._audio_tokenizer = (
            s3tokenizer.load_model(f"{self.model_path}/speech_tokenizer_v2_25hz.onnx").to(self.device).eval()
        )

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self._spk_model = onnxruntime.InferenceSession(
            f"{self.model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"]
        )

        with open(f"{self.model_path}/flow.yaml") as f:
            configs = load_hyperpyyaml(f)
        self._flow = configs["flow"]
        if self.float16:
            self._flow.half()
        self._flow.load_state_dict(
            torch.load(f"{self.model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True
        )
        self._flow.to(self.device).eval()

        self._hift = HiFTGenerator()
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(f"{self.model_path}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self._hift.load_state_dict(hift_state_dict, strict=True)
        self._hift.to(self.device).eval()

        self._models_loaded = True
        logger.info("Token2Wav models loaded successfully")

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

    def _prepare_prompt(self, prompt_wav: str):
        """Prepare prompt audio for conditioning"""
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device)
        )

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(
            self.spk_model.run(None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0],
            device=self.device,
        )

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
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        generated_speech_tokens_lens = torch.tensor(
            [generated_speech_tokens.shape[1]], dtype=torch.int32, device=self.device
        )

        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(
                generated_speech_tokens,
                generated_speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                prompt_mels_lens,
                spk_emb,
                self.n_timesteps,
            )

        wav, _ = self.hift(speech_feat=mel)

        if return_bytes:
            output = io.BytesIO()
            torchaudio.save(output, wav.cpu(), sample_rate=DEFAULT_TOKEN2WAV_CONFIG.sample_rate, format="wav")
            return output.getvalue()
        else:
            return wav


class StepAudio2Token2WavForConditionalGeneration(nn.Module, SupportsPP):
    """vLLM-compatible wrapper for Step-Audio2 Token2Wav"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "token2wav.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        model_path = getattr(self.config, "token2wav_path", None)
        if model_path is None:
            model_name_or_path = vllm_config.model_config.model
            model_path = f"{model_name_or_path}/token2wav"

        float16 = getattr(self.config, "token2wav_float16", False)

        device = "cuda"
        if hasattr(vllm_config, "device_config") and vllm_config.device_config:
            device = str(vllm_config.device_config.device)

        self.token2wav = StepAudio2Token2WavCore(model_path=model_path, float16=float16, device=device)

        self.have_multimodal_outputs = True

        self.make_empty_intermediate_tensors = lambda: None

    def get_language_model(self) -> torch.nn.Module:
        return self.token2wav

    @property
    def sampler(self):
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, hidden_size)

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
            positions: Position IDs (unused)
            intermediate_tensors: Intermediate tensors (unused)
            inputs_embeds: Input embeddings (unused)
            additional: Additional information dict (deprecated)
            **kwargs: Other kwargs

        Returns:
            OmniOutput with waveform tensor
        """
        if isinstance(input_ids, torch.Tensor):
            is_profile_mode = input_ids.numel() < 50 or input_ids.sum().item() == 0
        else:
            is_profile_mode = len(input_ids) < 50 if hasattr(input_ids, "__len__") else True

        if is_profile_mode:
            device = torch.device("cpu")
            if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.numel() > 0:
                device = inputs_embeds.device
            elif isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
                device = input_ids.device

            dummy_audio = torch.zeros(1000, device=device, dtype=torch.float32)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": dummy_audio},
            )

        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids_list = input_ids.cpu().tolist()
            else:
                input_ids_list = input_ids[0].cpu().tolist()
        else:
            input_ids_list = input_ids if isinstance(input_ids, list) else [input_ids]

        prompt_wav = os.environ.get("STEP_AUDIO2_DEFAULT_PROMPT_WAV", STEP_AUDIO2_DEFAULT_PROMPT_WAV)

        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"Token2Wav: prompt_wav file not found: {prompt_wav}")

        waveform_tensor = self.token2wav(input_ids_list, prompt_wav=prompt_wav, return_bytes=False)

        if waveform_tensor.dim() == 2 and waveform_tensor.shape[0] == 1:
            waveform_tensor = waveform_tensor.squeeze(0)

        waveform_tensor_cpu = waveform_tensor.detach().cpu().contiguous()

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": waveform_tensor_cpu},
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        for name, _ in self.token2wav.named_buffers():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params

    def load_weights_without_buffers(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params
