# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio speech output: mode machine, depth-decoder sampling, frame streaming to the codec."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.lfm3_audio.lfm3_audio_depth import AudioSampling, Lfm3AudioDepthDecoder
from vllm_omni.model_executor.models.lfm3_audio.lfm3_audio_modes import (
    AUDIO,
    INTERLEAVED,
    TEXT,
    Lfm3AudioModeState,
    Lfm3AudioTokens,
)
from vllm_omni.model_executor.stage_input_processors.lfm3_audio import lfm3_audio_to_codec_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TOKENS = Lfm3AudioTokens(audio_start=10, mixed_start=11, text_end=12, audio_out=13, eos_ids=(14,), n_text=2, n_audio=3)
WORD = 99


def _slots(state: Lfm3AudioModeState, outcomes: list) -> list[str]:
    """Run outcomes ('t', token) / ('f', end_of_audio); record the kind of slot each one filled."""
    kinds = []
    for kind, value in outcomes:
        kinds.append("speech" if state.speech_slot else "text")
        if kind == "t":
            state.after_text(value, TOKENS)
        else:
            state.after_frame(value, TOKENS)
    return kinds


def test_interleaved_blocks_alternate_then_speech_tail():
    state = Lfm3AudioModeState.from_prompt(TOKENS.mixed_start, TOKENS)
    assert (state.mode, state.slots_left, state.speaking) == (INTERLEAVED, 2, False)
    kinds = _slots(
        state,
        [("t", WORD), ("t", WORD), ("f", False), ("f", False), ("f", False), ("t", WORD), ("t", TOKENS.text_end)]
        + [("f", False), ("f", True)],
    )
    assert kinds == ["text"] * 2 + ["speech"] * 3 + ["text"] * 2 + ["speech"] * 2
    assert state.mode == TEXT and state.text_done and not state.finished


def test_markers_inside_a_block_are_text():
    state = Lfm3AudioModeState.from_prompt(TOKENS.mixed_start, TOKENS)
    state.after_text(TOKENS.audio_start, TOKENS)
    assert state.mode == INTERLEAVED and not state.speaking


def test_text_mode_switches_and_end_of_turn():
    state = Lfm3AudioModeState()
    state.after_text(TOKENS.audio_start, TOKENS)
    assert state.mode == AUDIO and state.speech_slot
    state.after_frame(True, TOKENS)
    assert state.mode == TEXT and not state.speech_slot
    state.after_text(TOKENS.mixed_start, TOKENS)
    assert (state.mode, state.slots_left, state.speaking) == (INTERLEAVED, 2, False)
    outside = Lfm3AudioModeState()
    outside.after_text(TOKENS.text_end, TOKENS)
    assert outside.finished


def test_state_round_trips_through_the_request_buffer():
    state = Lfm3AudioModeState(mode=INTERLEAVED, speaking=True, slots_left=1, text_done=False)
    assert Lfm3AudioModeState.from_info(state.as_updates()) == state


def _tiny_depth_config() -> SimpleNamespace:
    depth = SimpleNamespace(
        model_type="lfm3",
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"] * 2,
        norm_eps=1e-5,
        conv_L_cache=3,
        rope_parameters={"full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 1.0}},
    )
    return SimpleNamespace(
        num_codebooks=12,
        codebook_size=17,
        maskgit_schedule=[4, 4, 4],
        depth_config=depth,
        text_config=SimpleNamespace(hidden_size=48),
    )


def test_depth_decoder_reveals_every_codebook():
    torch.manual_seed(0)
    decoder = Lfm3AudioDepthDecoder(_tiny_depth_config()).eval()
    for param in decoder.parameters():
        torch.nn.init.normal_(param, std=0.5)
    hidden = torch.randn(48)
    greedy = AudioSampling(temperature=0.0)
    first = decoder.sample_frame(hidden, greedy, generator=torch.Generator().manual_seed(1))
    again = decoder.sample_frame(hidden, greedy, generator=torch.Generator().manual_seed(1))
    assert first.shape == (12,) and torch.equal(first, again)
    sampled = decoder.sample_frame(hidden, AudioSampling(top_k=4), history=first[None].repeat(5, 1))
    assert sampled.shape == (12,) and int(sampled.min()) >= 0 and int(sampled.max()) < 17


def _manager(**extra) -> SimpleNamespace:
    return SimpleNamespace(request_payload={}, connector=SimpleNamespace(config={"extra": extra}))


def _frame(value: int) -> dict:
    return {"codes": {"audio": torch.full((12, 1), value, dtype=torch.long)}}


def test_windows_ramp_up_and_hold_back_the_newest_frame():
    manager, request = _manager(codec_chunk_ramp="2,3", codec_chunk_frames=4), SimpleNamespace(request_id="r")
    sent = []
    for value in range(12):
        payload = lfm3_audio_to_codec_async_chunk(manager, _frame(value), request)
        if payload is not None:
            assert not bool(payload.meta.finished)
            sent.append(payload.codes.audio[0].tolist())
    assert sent == [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
    last = lfm3_audio_to_codec_async_chunk(manager, {"codes": {"audio": torch.empty(12, 0)}}, request, True)
    assert bool(last.meta.finished) and last.codes.audio[0].tolist() == [9, 10, 11]
    assert lfm3_audio_to_codec_async_chunk(manager, _frame(12), request, True) is None


def test_text_only_reply_still_closes_the_stream():
    manager, request = _manager(), SimpleNamespace(request_id="r")
    assert lfm3_audio_to_codec_async_chunk(manager, {}, request) is None
    last = lfm3_audio_to_codec_async_chunk(manager, {}, request, True)
    assert bool(last.meta.finished) and last.codes.audio.shape == (12, 0)
