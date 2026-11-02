# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for LFM3-Audio: speech frames -> Liquid codec.

Stage 0 publishes at most one ``[12, 1]`` frame per request per step
(``codes.audio``). Frames go on in windows: the first ones early
(``codec_chunk_ramp``, default ``2, 4, 8``), then ``codec_chunk_frames``
(default 12, about 1 s) at a time. The codec keeps its own stream state, so a
window carries only new frames. The newest frame is held back until the
request finishes: the window that closes the stream then carries audio and the
codec stage runs to flush its overlap-add tail.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

NUM_CODEBOOKS = 12
DEFAULT_RAMP = (2, 4, 8)
DEFAULT_CHUNK = 12
_STATE_KEY = "_lfm3_audio_stream"


def _config(transfer_manager: Any) -> tuple[tuple[int, ...], int]:
    cached = getattr(transfer_manager, "_lfm3_audio_chunk_config", None)
    if cached is not None:
        return cached
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    cfg = raw.get("extra", raw) if isinstance(raw, dict) else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", DEFAULT_CHUNK))
    ramp_cfg = cfg.get("codec_chunk_ramp", DEFAULT_RAMP)
    ramp = tuple(int(v) for v in (ramp_cfg.split(",") if isinstance(ramp_cfg, str) else ramp_cfg))
    if chunk <= 0 or any(v <= 0 for v in ramp):
        raise ValueError(f"invalid LFM3-Audio codec chunk config chunk={chunk} ramp={ramp}")
    transfer_manager._lfm3_audio_chunk_config = (ramp, chunk)
    return transfer_manager._lfm3_audio_chunk_config


def _frames(output: Any) -> torch.Tensor | None:
    if not isinstance(output, Mapping):
        return None
    codes = output.get("codes", {})
    frames = codes.get("audio") if isinstance(codes, Mapping) else None
    if frames is None:
        frames = output.get("codes.audio")
    if not isinstance(frames, torch.Tensor) or frames.numel() == 0:
        return None
    return frames.detach().to("cpu", dtype=torch.long).reshape(NUM_CODEBOOKS, -1)


def _request_id(request: Any) -> str:
    return str(getattr(request, "external_req_id", None) or getattr(request, "request_id", id(request)))


def lfm3_audio_to_codec_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    finished = bool(is_finished or (hasattr(request, "is_finished") and request.is_finished()))
    # request_payload entries are reclaimed when a request completes or is cancelled.
    state = transfer_manager.request_payload.setdefault(_request_id(request), {}).setdefault(
        _STATE_KEY, {"frames": [], "windows": 0, "done": False}
    )
    if state["done"]:
        return None
    frames = _frames(multimodal_output)
    if frames is not None:
        state["frames"].append(frames)
    if finished:
        state["done"] = True
        pending = state.pop("frames")
        codes = torch.cat(pending, dim=-1) if pending else torch.empty((NUM_CODEBOOKS, 0), dtype=torch.long)
        return OmniPayloadStruct(
            codes=CodesStruct(audio=codes.contiguous()),
            meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
        )
    ramp, chunk = _config(transfer_manager)
    target = ramp[state["windows"]] if state["windows"] < len(ramp) else chunk
    pending = sum(f.shape[-1] for f in state["frames"])
    # The newest frame stays behind for the closing window.
    if pending - 1 < target:
        return None
    window = torch.cat(state["frames"], dim=-1)
    state["frames"] = [window[:, -1:]]
    state["windows"] += 1
    return OmniPayloadStruct(
        codes=CodesStruct(audio=window[:, :-1].contiguous()),
        meta=MetaStruct(finished=torch.tensor(False, dtype=torch.bool)),
    )


__all__ = ["lfm3_audio_to_codec_async_chunk"]
