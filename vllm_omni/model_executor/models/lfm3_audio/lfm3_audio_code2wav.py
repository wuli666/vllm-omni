# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio stage 1: Liquid codec frames -> 24 kHz audio.

The codec decoder is natively streaming (``LiquidCodecStream`` per request), so
each step decodes only the frames that arrived and returns delta audio; the
overlap-add tail is flushed when the request finishes. Payload: ``codes.audio``
is a ``[12, frames]`` tensor of new frames, ``meta.req_id`` names the stream and
``meta.stream_finished`` (async chunk) or ``meta.finished`` (full payload)
closes it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .liquid_codec import LiquidCodecDecoderModel, LiquidCodecStream

logger = init_logger(__name__)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list | tuple) and value else value


def _flag(value: Any) -> bool:
    value = _first(value)
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item()) if value.numel() else False
    return bool(value)


def _frames(info: dict[str, Any], num_codebooks: int) -> torch.Tensor | None:
    codes = info.get("codes")
    audio = codes.get("audio") if isinstance(codes, dict) else None
    if audio is None:
        return None
    audio = torch.as_tensor(audio)
    if audio.numel() == 0:
        return None
    return audio.reshape(num_codebooks, -1)


class Lfm3AudioCode2Wav(nn.Module):
    """Stage-1 model for LFM3-Audio / LFM3-Omni (GenerationModelRunner)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self.codec = LiquidCodecDecoderModel(vllm_config.model_config.hf_config)
        # ponytail: one decode per request per step; batch the LFM3 stack across streams if stage 1 becomes the
        # bottleneck at high concurrency.
        self._streams: dict[str, LiquidCodecStream] = {}

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage reads codes from the payload, not token embeddings.
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.codec.named_parameters()) | dict(self.codec.named_buffers())
        loaded: set[str] = set()
        for name, tensor in weights:
            target = params.get(name)
            if target is None:
                # The analysis half of the codec (frontend, encoder, quantizer.project_in) is unused here.
                continue
            target.data.copy_(tensor)
            loaded.add(f"codec.{name}")
        # The synthesis window and the ISTFT stay in float32, as in the reference.
        self.codec.decoder.istft.float()
        return loaded

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        for req_id in finished_req_ids:
            self._streams.pop(req_id, None)

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"Lfm3AudioCode2Wav expected OmniOutput, got {type(model_outputs)}")

    def _decode(self, req_id: str, frames: torch.Tensor | None, finished: bool) -> torch.Tensor:
        stream = self._streams.get(req_id)
        if stream is None:
            if frames is None:
                if finished:
                    logger.error("LFM3-Audio code2wav: request %s finished without speech frames", req_id)
                return torch.empty(0, dtype=torch.float32)
            stream = self._streams[req_id] = LiquidCodecStream()
        device = self.codec.from_quantizer.weight.device
        codes = (
            frames.to(device=device, dtype=torch.long).unsqueeze(0)
            if frames is not None
            else torch.empty((1, self.codec.num_codebooks, 0), dtype=torch.long, device=device)
        )
        audio = self.codec.decode(codes, stream, finalize=finished)
        if finished:
            self._streams.pop(req_id, None)
        return audio.reshape(-1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        infos = runtime_additional_information or []
        seq_token_counts = kwargs.get("seq_token_counts")
        num_reqs = len(seq_token_counts) if seq_token_counts is not None else len(infos)
        sample_rate = torch.tensor(self.codec.sample_rate, dtype=torch.int32)
        audios: list[torch.Tensor] = []
        for index in range(num_reqs):
            info = infos[index] if index < len(infos) and isinstance(infos[index], dict) else {}
            meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
            req_id = _first(meta.get("req_id"))
            frames = _frames(info, self.codec.num_codebooks)
            if not isinstance(req_id, str):
                if frames is not None:
                    raise ValueError("LFM3-Audio code2wav payload is missing meta.req_id")
                audios.append(torch.empty(0, dtype=torch.float32))
                continue
            finished = _flag(meta.get("stream_finished")) or _flag(meta.get("finished"))
            audios.append(self._decode(req_id, frames, finished))
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [sample_rate] * num_reqs},
        )
