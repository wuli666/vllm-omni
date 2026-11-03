# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio stage 0: speech encoder + LFM3 language model.

Each input clip is one ``<|audio|>`` in the prompt, expanded to one
placeholder per 8 mel frames (12.5 Hz) and replaced by the projected encoder
output. The language model is vLLM's hybrid ``Lfm3ForCausalLM`` (short conv,
full and sliding attention); its conv state is managed by vLLM's mamba cache.

Both checkpoint layouts load: LFM3-Audio's training export (``lfm.model.``,
``encoder.``, ``mlp.model.``) and the Transformers layout of LFM3-Omni
(``model.language_model.``, ``model.audio_encoder.``, ...).

On a stage that hands speech on (``engine_output_type != "text"``) every slot
is either text or a speech frame, per request (``lfm3_audio_modes``):

* ``preprocess`` feeds the previous frame back (sum of its 12 code
  embeddings) or advances the mode machine on the sampled text token, and
  decides whether this step's slot is speech;
* ``make_omni_output`` samples the frames of all speech slots of the step
  with one depth decoder pass and publishes each as ``codes.audio`` ``[12, 1]``;
* ``compute_logits`` forces ``<|audio_out|>`` on speech slots, so the sequence
  carries one placeholder per frame, as in the reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import HFMultiModalInputs
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .lfm3_audio_depth import AudioSampling, Lfm3AudioDepthDecoder, to_device
from .lfm3_audio_encoder import Lfm3AudioEncoder, Lfm3AudioProjector
from .lfm3_audio_modes import Lfm3AudioModeState, Lfm3AudioTokens

logger = init_logger(__name__)

AUDIO_PLACEHOLDER = "<|audio|>"
# Request-buffer keys of device tensors, kept on the GPU between steps.
_FRAME_EMBED, _HISTORY = "lfm3_frame_embed", "lfm3_history"
_AUDIO_SAMPLING_KEYS = ("temperature", "top_k", "repetition_penalty", "repetition_window")
# One encoder output per 8 mel frames (three stride-2 convolutions).
ENCODER_STRIDE = 8
# Profiling budget per clip; longer clips still encode, at 12.5 tokens per second.
MAX_PROFILE_AUDIO_SECONDS = 30


def num_audio_tokens(num_frames: int) -> int:
    return -(num_frames // -ENCODER_STRIDE)


class Lfm3AudioProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_feature_extractor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int] | None = None) -> Mapping[str, int]:
        if not mm_counts or mm_counts.get("audio", 0) <= 0:
            return {}
        feature_extractor = self.get_feature_extractor()
        frames = MAX_PROFILE_AUDIO_SECONDS * feature_extractor.sampling_rate // feature_extractor.hop_length
        return {"audio": num_audio_tokens(frames)}


class Lfm3AudioDummyInputsBuilder(BaseDummyInputsBuilder[Lfm3AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, MultiModalDummyOptions],
    ) -> MultiModalDataDict:
        sampling_rate = self.info.get_feature_extractor().sampling_rate
        return {
            "audio": self._get_dummy_audios(
                length=MAX_PROFILE_AUDIO_SECONDS * sampling_rate,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class Lfm3AudioMultiModalProcessor(BaseMultiModalProcessor[Lfm3AudioProcessingInfo]):
    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_hf_mm_inputs(self, mm_items: MultiModalDataItems, hf_kwargs: Mapping[str, object]) -> HFMultiModalInputs:
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)
        sampling_rate = self.info.get_feature_extractor(**hf_kwargs).sampling_rate
        return hf_inputs._replace(hf_kwargs=dict(hf_inputs.hf_kwargs, sampling_rate=sampling_rate))

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audio_token_id = self.info.get_hf_config().audio_token_id
        mask = out_mm_kwargs.get_data().get("feature_attention_mask")
        lengths = [] if mask is None else [num_audio_tokens(int(m.sum())) for m in mask]

        def replacement(item_idx: int) -> list[int]:
            return [audio_token_id] * lengths[item_idx]

        return [PromptReplacement(modality="audio", target=[audio_token_id], replacement=replacement)]


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3AudioMultiModalProcessor,
    info=Lfm3AudioProcessingInfo,
    dummy_inputs=Lfm3AudioDummyInputsBuilder,
)
class Lfm3AudioForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, IsHybrid, HasInnerState):
    merge_by_field_config = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # LFM3-Audio training export.
            "lfm.model.": "language_model.model.",
            "lfm.lm_head.": "language_model.lm_head.",
            "encoder.lfm.lfm.": "audio_encoder.lfm.",
            "encoder.downsample.": "audio_encoder.downsample.",
            "mlp.model.0.": "audio_projector.layer_norm.",
            "mlp.model.1.": "audio_projector.linear_1.",
            "mlp.model.3.": "audio_projector.linear_2.",
            "depthformer.model.": "depth_decoder.model.",
            "depth_embeddings.": "depth_decoder.depth_embeddings.",
            "depth_linear.": "depth_decoder.depth_linear.",
            "mask_embedding": "depth_decoder.mask_embedding",
            # Transformers layout (LFM3-Omni).
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "model.audio_encoder.lfm.": "audio_encoder.lfm.",
            "model.audio_encoder.downsample.": "audio_encoder.downsample.",
            "model.audio_projector.": "audio_projector.",
            "model.codebook_embedding.": "codebook_embedding.",
            "model.depth_decoder.": "depth_decoder.",
        },
        # The encoder stack is LFM2, which names these differently from the LFM3 layers.
        orig_to_new_substr={
            ".self_attn.out_proj.": ".self_attn.o_proj.",
            ".self_attn.q_layernorm.": ".self_attn.q_norm.",
            ".self_attn.k_layernorm.": ".self_attn.k_norm.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER
        raise ValueError("LFM3-Audio accepts audio inputs only")

    # The conv state lives in the language model; its config is the text config.
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int]]:
        text_config = vllm_config.model_config.hf_config.text_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=text_config.hidden_size,
            conv_kernel=text_config.conv_L_cache,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = Lfm3AudioEncoder(config.audio_encoder_config)
            self.audio_projector = Lfm3AudioProjector(
                config.audio_encoder_config.hidden_size, config.text_config.hidden_size
            )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Lfm3ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # Speech output runs on a stage that hands frames on; a text stage keeps the reply in text.
        self.speech_output = getattr(vllm_config.model_config, "engine_output_type", "text") != "text"
        if self.speech_output and vllm_config.speculative_config is not None:
            # Speech slots are decided, sampled and fed back one token per step.
            raise NotImplementedError("LFM3-Audio speech output does not support speculative decoding")
        self.has_preprocess = self.have_multimodal_outputs = self.speech_output
        # Speech frames are sampled for the whole batch at once in make_omni_output, not per request.
        self.has_postprocess = False
        # preprocess only rewrites decode rows: prompt rows keep the encoder's speech embeddings.
        self.preprocess_keeps_mm_inputs = self.speech_output
        self.gpu_resident_buffer_keys = {_FRAME_EMBED, _HISTORY}
        eos = config.text_config.eos_token_id
        self.tokens = Lfm3AudioTokens.from_config(config, tuple(eos) if isinstance(eos, list | tuple) else (eos,))
        self.audio_eos_code = config.audio_eos_code
        if self.speech_output:
            num_codes = config.num_codebooks * config.codebook_size
            self.codebook_embedding = nn.Embedding(num_codes, config.text_config.hidden_size)
            offsets = torch.arange(config.num_codebooks) * config.codebook_size
            self.register_buffer("codebook_offsets", offsets, persistent=False)
            self.depth_decoder = Lfm3AudioDepthDecoder(config)
        # A text slot never samples the placeholders; a text stage also never opens speech.
        never = [config.audio_token_id, config.audio_out_token_id]
        if not self.speech_output:
            never += [config.audio_start_token_id, config.mixed_start_token_id]
        self.register_buffer("never_sampled", torch.tensor(never), persistent=False)
        # Token each row of the current step must sample (-1: free); set by make_omni_output.
        self._forced_tokens: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Speech slots
    # ------------------------------------------------------------------

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        num_tokens = input_ids.shape[0]
        if info.get("_omni_is_prefill", num_tokens > 1):
            if info.get("_omni_num_computed_tokens", 0) + num_tokens < info.get("_omni_prompt_len", num_tokens):
                return input_ids, input_embeds, {}  # an earlier prompt chunk keeps no sample
            state = Lfm3AudioModeState.from_prompt(int(input_ids[-1]), self.tokens)
            updates: dict[str, Any] = {_HISTORY: None}
            seed = info.get("_omni_seed")
            if seed is not None:
                updates["lfm3_generator"] = torch.Generator(device=input_embeds.device).manual_seed(int(seed))
        else:
            state = Lfm3AudioModeState.from_info(info)
            updates = {}
            if info.get("lfm3_frame_pending"):
                # The slot sampled <|audio_out|>; its input is the frame's summed code embeddings.
                frame_embed = info[_FRAME_EMBED].to(device=input_embeds.device, dtype=input_embeds.dtype)
                input_embeds = frame_embed.reshape(1, -1)
            else:
                # ponytail: one host sync per text row per step; batch it in preprocess_decode_batch_mrv2
                # if it shows up in profiles.
                state.after_text(int(input_ids[0]), self.tokens)
        updates |= state.as_updates()
        updates |= {"lfm3_speech_slot": state.speech_slot, "lfm3_frame_pending": False}
        return input_ids, input_embeds, updates

    @staticmethod
    def _audio_sampling(info: dict[str, Any]) -> AudioSampling:
        """Defaults of the checkpoint's generation_config; ``audio_<field>`` in additional information overrides.

        This runs inside the step for the whole batch, so an invalid override falls back to the
        default with a warning instead of raising.
        """
        overrides = {}
        for field in _AUDIO_SAMPLING_KEYS:
            value = info.get(f"audio_{field}")
            value = value[0] if isinstance(value, list | tuple) and value else value
            if value is None:
                continue
            try:
                value = type(getattr(AudioSampling, field))(value)
            except (TypeError, ValueError, OverflowError):
                value = None
            # temperature 0 is greedy, top_k / repetition_window 0 disable them; a penalty must stay positive.
            # ``not value >= 0`` also rejects NaN.
            if value is None or not value >= 0 or (field == "repetition_penalty" and value == 0):
                # warning_once caches its arguments, so they must be hashable.
                logger.warning_once("LFM3-Audio: ignoring invalid audio_%s=%s", field, repr(info.get(f"audio_{field}")))
                continue
            overrides[field] = value
        return AudioSampling(**overrides)

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput:
        """Sample this step's speech frames, all speech slots of the batch in one depth decoder pass.

        ``model_intermediate_buffer`` holds the live request buffers in batch order: the frame state written
        here is what the next step's ``preprocess`` reads.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        empty = model_outputs.new_empty((self.config.num_codebooks, 0), dtype=torch.long)
        frames = [empty] * len(infos)
        speech = [i for i, info in enumerate(infos) if info.get("lfm3_speech_slot")]
        if speech:
            spans = kwargs["request_token_spans"]
            last = to_device([spans[i][1] - 1 for i in speech], model_outputs.device, torch.long)
            samplings = [self._audio_sampling(infos[i]) for i in speech]
            codes = self.depth_decoder.sample_frames(
                model_outputs[last],
                samplings,
                [infos[i].get(_HISTORY) for i in speech],
                [infos[i].get("lfm3_generator") for i in speech],
            )
            end_of_audio = (codes == self.audio_eos_code).any(-1)
            codes = torch.where(end_of_audio[:, None], self.audio_eos_code, codes)
            embeds = self.codebook_embedding(codes + self.codebook_offsets).sum(1)
            for row, (i, sampling, eoa) in enumerate(zip(speech, samplings, end_of_audio.tolist())):
                info, frame = infos[i], codes[row]
                history = info.get(_HISTORY)
                if not eoa:
                    frames[i] = frame.reshape(-1, 1)
                    if sampling.repetition_window:
                        # ponytail: the window is re-snapshotted each frame; keep per-codebook counts if long
                        # replies get slow.
                        history = frame[None] if history is None else torch.cat((history.to(frame.device), frame[None]))
                        history = history[-sampling.repetition_window :]
                state = Lfm3AudioModeState.from_info(info)
                state.after_frame(eoa, self.tokens)
                info.update(
                    state.as_updates(), lfm3_frame_pending=True, **{_FRAME_EMBED: embeds[row], _HISTORY: history}
                )
        forced = [
            self.tokens.audio_out
            if info.get("lfm3_speech_slot")
            else self.tokens.eos_ids[0]
            if info.get("lfm3_finished")
            else -1
            for info in infos
        ]
        self._forced_tokens = (
            to_device(forced, model_outputs.device, torch.long) if any(t >= 0 for t in forced) else None
        )
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={"codes": {"audio": frames}})

    def embed_multimodal(self, **kwargs: Any) -> MultiModalEmbeddings:
        features = kwargs.get("input_features")
        masks = kwargs.get("feature_attention_mask")
        if features is None or masks is None:
            return []
        # Clips encode one at a time on their unpadded frames: the encoder is causal, so this
        # matches the reference's padded batch.
        # ponytail: per-clip loop; pack clips into one causal batch if encoder time shows up in profiles.
        embeddings = []
        for clip, mask in zip(features, masks):
            frames = int(mask.sum())
            embeddings.append(self.audio_projector(self.audio_encoder(clip[:, :frames])))
        return tuple(embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.language_model.compute_logits(hidden_states)
        if logits is None:
            return None
        logits[:, self.never_sampled] = float("-inf")
        forced, self._forced_tokens = self._forced_tokens, None
        if forced is not None and forced.shape[0] == logits.shape[0]:
            rows = forced >= 0
            logits[rows] = float("-inf")
            logits[rows, forced[rows]] = 0.0
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # A text stage has no depth decoder.
        loader = AutoWeightsLoader(
            self, ignore_unexpected_prefixes=[] if self.speech_output else ["depth_decoder.", "codebook_embedding."]
        )
        # The training export also stores untied per-codebook heads that the tied depth decoder never reads.
        weights = ((name, tensor) for name, tensor in weights if not name.endswith(".to_logits.weight"))
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
