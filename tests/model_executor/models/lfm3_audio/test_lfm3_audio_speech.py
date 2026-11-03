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


def test_depth_decoder_samples_each_row_as_if_alone():
    torch.manual_seed(0)
    decoder = Lfm3AudioDepthDecoder(_tiny_depth_config()).double().eval()
    for param in decoder.parameters():
        torch.nn.init.normal_(param, std=0.5)
    hidden = torch.randn(4, 48, dtype=torch.float64)
    greedy = AudioSampling(temperature=0.0)
    first = decoder.sample_frames(hidden[:1], [greedy], [None], [torch.Generator().manual_seed(1)])[0]
    history = first[None].repeat(5, 1)
    samplings = [greedy, AudioSampling(top_k=4), AudioSampling(temperature=0.0, repetition_penalty=1e6), greedy]
    histories = [None, history, history, history]

    def generators():
        return [
            torch.Generator().manual_seed(1),
            torch.Generator().manual_seed(2),
            torch.Generator().manual_seed(1),
            None,
        ]

    batch = decoder.sample_frames(hidden, samplings, histories, generators())
    alone = [
        decoder.sample_frames(hidden[i : i + 1], [samplings[i]], [histories[i]], [generators()[i]])[0] for i in range(3)
    ]
    assert batch.shape == (4, 12) and int(batch.min()) >= 0 and int(batch.max()) < 17
    # Seeded rows sample the same frame whatever else is in the batch.
    assert torch.equal(batch[:3], torch.stack(alone)) and torch.equal(batch[0], first)
    # A code already in the window is pushed down: with a huge penalty greedy picks another one.
    assert not (batch[2] == first).any()


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
    # ``finished`` stays with the receiver's scheduler; the codec reads ``stream_finished``.
    assert bool(last.meta.stream_finished)
    assert lfm3_audio_to_codec_async_chunk(manager, _frame(12), request, True) is None


def test_text_only_reply_still_closes_the_stream():
    manager, request = _manager(), SimpleNamespace(request_id="r")
    assert lfm3_audio_to_codec_async_chunk(manager, {}, request) is None
    last = lfm3_audio_to_codec_async_chunk(manager, {}, request, True)
    assert bool(last.meta.finished) and last.codes.audio.shape == (12, 0)


# ---------------------------------------------------------------------------
# Stage-0 hooks together, on a stand-in for the model (no vLLM engine needed)
# ---------------------------------------------------------------------------

AUDIO_IN, VOCAB, HIDDEN, EOS_CODE = 9, 20, 8, 16


class _ScriptedDepth:
    def __init__(self, frames):
        self.frames = list(frames)

    def sample_frames(self, hidden, samplings, histories, generators):
        self.hidden = hidden
        return torch.stack([self.frames.pop(0) for _ in range(hidden.shape[0])])


def _stage0(frames) -> SimpleNamespace:
    from vllm_omni.model_executor.models.lfm3_audio.lfm3_audio import Lfm3AudioForConditionalGeneration as Model

    torch.manual_seed(0)
    return SimpleNamespace(
        tokens=TOKENS,
        audio_eos_code=EOS_CODE,
        depth_decoder=_ScriptedDepth(frames),
        codebook_embedding=torch.nn.Embedding(12 * 17, HIDDEN),
        codebook_offsets=torch.arange(12) * 17,
        config=SimpleNamespace(num_codebooks=12),
        never_sampled=torch.tensor([AUDIO_IN, TOKENS.audio_out]),
        _forced_tokens=None,
        language_model=SimpleNamespace(compute_logits=lambda h: torch.zeros(h.shape[0], VOCAB)),
        _audio_sampling=Model._audio_sampling,
        hooks=Model,
    )


def _step(model, info, ids, **phase):
    """One step of one request: preprocess -> make_omni_output -> compute_logits."""
    hooks = model.hooks
    embeds = torch.zeros(len(ids), HIDDEN)
    _, embeds, updates = hooks.preprocess(model, torch.tensor(ids), embeds, **info, **phase)
    info.update(updates)
    hidden = torch.ones(len(ids), HIDDEN)
    out = hooks.make_omni_output(model, hidden, model_intermediate_buffer=[info], request_token_spans=[(0, len(ids))])
    logits = hooks.compute_logits(model, torch.ones(1, HIDDEN))
    return embeds, out.multimodal_outputs["codes"]["audio"][0], logits[0]


def test_hooks_run_a_speech_turn_end_to_end():
    frame = torch.arange(12)
    model, info = _stage0([frame, torch.full((12,), EOS_CODE)]), {"req_id": "r"}
    prefill = dict(_omni_is_prefill=True, _omni_prompt_len=3)
    # An earlier prompt chunk keeps no sample and decides nothing.
    _, _, updates = model.hooks.preprocess(model, torch.tensor([5, 6]), torch.zeros(2, HIDDEN), **prefill)
    assert updates == {}
    # A one-token final prompt chunk ending on <|audio_start|> opens a speech slot.
    _, codes, logits = _step(model, info, [TOKENS.audio_start], _omni_num_computed_tokens=2, **prefill)
    assert torch.equal(codes, frame[:, None]) and int(logits.argmax()) == TOKENS.audio_out
    decode = dict(_omni_is_prefill=False, _omni_prompt_len=3)
    # The frame is fed back as its summed code embeddings; the next slot ends the audio.
    expected = model.codebook_embedding(frame + model.codebook_offsets).sum(0)
    embeds, codes, logits = _step(model, info, [TOKENS.audio_out], _omni_num_computed_tokens=3, **decode)
    torch.testing.assert_close(embeds[0], expected)
    assert codes.shape == (12, 0) and info["lfm3_mode"] == TEXT
    # After the end-of-audio frame the slot is text: nothing forced, placeholders never sampled.
    _, codes, logits = _step(model, info, [TOKENS.audio_out], _omni_num_computed_tokens=4, **decode)
    assert codes.shape == (12, 0) and not info["lfm3_speech_slot"]
    assert logits[TOKENS.audio_out] == float("-inf") and logits[AUDIO_IN] == float("-inf")
    assert logits[WORD % VOCAB] == 0


def test_a_mixed_batch_samples_speech_rows_only_from_their_last_token():
    frame = torch.arange(12)
    model = _stage0([frame])
    speech, text = {"lfm3_speech_slot": True}, {"lfm3_speech_slot": False}
    hidden = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)
    # The text request has one token, the speech request two (a final prompt chunk).
    out = model.hooks.make_omni_output(
        model, hidden, model_intermediate_buffer=[text, speech], request_token_spans=[(0, 1), (1, 3)]
    )
    assert torch.equal(model.depth_decoder.hidden, hidden[2:3])
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes[0].shape == (12, 0) and torch.equal(codes[1], frame[:, None])
    assert speech["lfm3_frame_pending"] and torch.equal(speech["lfm3_history"], frame[None])
    assert "lfm3_frame_pending" not in text
    logits = model.hooks.compute_logits(model, torch.ones(2, HIDDEN))
    assert int(logits[1].argmax()) == TOKENS.audio_out and logits[0, TOKENS.audio_out] == float("-inf")
    # Forcing applies to the step it was set for only.
    assert model._forced_tokens is None


def test_invalid_audio_overrides_fall_back_and_zero_window_keeps_no_history():
    model = _stage0([torch.arange(12)])
    sampling = model._audio_sampling(
        {"audio_top_k": [-1], "audio_temperature": 0.5, "audio_repetition_penalty": 0, "audio_repetition_window": "x"}
    )
    assert sampling == AudioSampling(temperature=0.5)
    assert model._audio_sampling({"audio_temperature": float("nan"), "audio_top_k": float("inf")}) == AudioSampling()
    info = {"lfm3_speech_slot": True, "audio_repetition_window": 0}
    out = model.hooks.make_omni_output(
        model, torch.ones(1, HIDDEN), model_intermediate_buffer=[info], request_token_spans=[(0, 1)]
    )
    # The frame still goes to the codec; only the penalty window stays empty.
    assert info["lfm3_history"] is None and out.multimodal_outputs["codes"]["audio"][0].shape == (12, 1)


def test_codec_receives_every_sample_through_the_adapter():
    """Processor windows, as the receiver hands them to the codec (``finished`` is not forwarded)."""
    from vllm_omni.model_executor.models.lfm3_audio.lfm3_audio_code2wav import Lfm3AudioCode2Wav
    from vllm_omni.transformers_utils.configs.lfm3_audio import LiquidCodecConfig

    torch.manual_seed(0)
    codec = Lfm3AudioCode2Wav(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=LiquidCodecConfig())))
    for param in codec.parameters():
        torch.nn.init.normal_(param, std=0.02)
    codec = codec.float().eval()
    manager, request = _manager(), SimpleNamespace(request_id="r")
    num_frames, samples = 9, 0
    for step in range(num_frames + 1):
        done = step == num_frames
        output = {} if done else {"codes": {"audio": torch.randint(0, 2016, (12, 1))}}
        payload = lfm3_audio_to_codec_async_chunk(manager, output, request, done)
        if payload is None:
            continue
        info = {"codes": {"audio": payload.codes.audio}, "meta": {"stream_finished": payload.meta.stream_finished}}
        out = codec(runtime_additional_information=[info], seq_token_counts=[1], request_ids=["r"])
        samples += out.multimodal_outputs["model_outputs"][0].numel()
    assert samples == num_frames * 1920 and not codec._streams
