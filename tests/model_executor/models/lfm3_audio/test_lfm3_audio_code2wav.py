# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio code2wav stage: per-request streams, cleanup, payload validation, weight loading."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.model_executor.models.lfm3_audio.lfm3_audio_code2wav import Lfm3AudioCode2Wav
from vllm_omni.transformers_utils.configs.lfm3_audio import LiquidCodecConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def stage() -> Lfm3AudioCode2Wav:
    torch.manual_seed(0)
    model = Lfm3AudioCode2Wav(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=LiquidCodecConfig())))
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return model.float().eval()


def _codes(num_frames: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 2016, (12, num_frames), generator=torch.Generator().manual_seed(seed))


def _step(stage, request_ids, payloads):
    infos = [{"codes": {"audio": codes}, "meta": {"stream_finished": done}} for codes, done in payloads]
    out = stage(runtime_additional_information=infos, seq_token_counts=[1] * len(infos), request_ids=request_ids)
    return out.multimodal_outputs["model_outputs"]


def test_interleaved_streams_match_whole_decodes(stage):
    a, b = _codes(6, 1), _codes(5, 2)
    first = _step(stage, ["a", "b"], [(a[:, :2], False), (b[:, :3], False)])
    second = _step(stage, ["b", "a"], [(b[:, 3:], True), (a[:, 2:], True)])
    for codes, pieces in ((a, [first[0], second[1]]), (b, [first[1], second[0]])):
        full = stage.codec.decode(codes.unsqueeze(0)).reshape(-1)
        torch.testing.assert_close(torch.cat(pieces), full, atol=1e-5, rtol=1e-4)
    assert not stage._streams


def test_abort_frees_the_stream_by_scheduler_id(stage):
    _step(stage, ["engine-a"], [(_codes(3, 3), False)])
    assert "engine-a" in stage._streams
    stage.on_requests_finished({"engine-a"})
    assert not stage._streams


def test_out_of_range_codes_are_dropped(stage):
    bad = _codes(2, 4)
    bad[0, 0] = 2016
    (audio,) = _step(stage, ["a"], [(bad, False)])
    assert audio.numel() == 0 and not stage._streams


def test_weights_load_and_report_missing(stage):
    state = {k: v.clone() for k, v in stage.codec.state_dict().items()}
    state["quantizer.fsq.project_out.weight"] = state.pop("quantizer.project_out.weight")
    loaded = stage.load_weights(state.items())
    assert "codec.quantizer.project_out.weight" in loaded
    state.pop("decoder.lin.weight")
    with pytest.raises(ValueError, match="missing 1 decoder tensors"):
        stage.load_weights(state.items())


def test_checkpoint_model_type_resolves_a_pipeline():
    pipeline = resolve_pipeline_config("lfm3_audio")
    assert pipeline is not None
    assert pipeline.stages[0].model_arch == "Lfm3AudioForConditionalGeneration"
