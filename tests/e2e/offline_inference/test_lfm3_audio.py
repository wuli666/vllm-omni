# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline E2E tests for LFM3-Audio: text to speech and voice chat (speech deployment), ASR (text deployment).

The hard gates are structural: every speech frame reaches the codec (audio samples = frames x 1920), the
speech is not silent, and the transcript is close to the public clip's.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pytest
from vllm.assets.audio import AudioAsset

from tests.helpers.mark import hardware_test
from tests.helpers.media import concat_audio
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

pytest.importorskip("vllm.model_executor.models.lfm3", reason="needs vLLM with Lfm3ForCausalLM (vllm#64959)")

MODEL = os.environ.get("VLLM_OMNI_LFM3_AUDIO_MODEL_DIR") or "LiquidAI/LFM3-Audio-2B"
SPEECH = pytest.param((MODEL, get_deploy_config_path("lfm3_audio.yaml")), id="speech")
TEXT = pytest.param((MODEL, get_deploy_config_path("lfm3_audio_text.yaml")), id="text")
AUDIO_OUT = 125032  # <|audio_out|>: one per speech frame, plus the end-of-audio frame
SAMPLES_PER_FRAME = 1920  # 24 kHz at 12.5 Hz
REFERENCE = (
    "the first words i spoke in the original phonograph a little piece of practical poetry "
    "mary had a little lamb its fleece was white as snow and everywhere that mary went the "
    "lamb was sure to go"
)

pytestmark = [pytest.mark.slow, pytest.mark.tts]


def _prompt(system: str, user: str, prefill: str = "", audio: bool = False) -> dict:
    prompt = {
        "prompt": f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefill}"
    }
    if audio:
        prompt["multi_modal_data"] = {"audio": AudioAsset("mary_had_lamb").audio_and_sample_rate}
    return prompt


def _words(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()


def _wer(hyp: str, ref: str) -> float:
    hyp_words, ref_words = _words(hyp), _words(ref)
    rows = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, 1):
        prev, rows[0] = rows[0], i
        for j, hyp_word in enumerate(hyp_words, 1):
            prev, rows[j] = rows[j], min(rows[j] + 1, rows[j - 1] + 1, prev + (ref_word != hyp_word))
    return rows[-1] / len(ref_words)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", [SPEECH], indirect=True)
def test_speech_reaches_the_codec_frame_for_frame(omni_runner: OmniRunner, run_level: str) -> None:
    prompts = [
        _prompt("Perform TTS in American English using a female voice.", "Cows eat hay.", "<|audio_start|>"),
        _prompt("Respond with interleaved text and audio.", "<|audio|>", "<|mixed_start|>", audio=True),
    ]
    replies = defaultdict(dict)
    for output in omni_runner.omni.generate(prompts):
        reply = replies[output.request_id]
        if output.final_output_type == "text":
            reply["text"], reply["ids"] = output.outputs[0].text, list(output.outputs[0].token_ids)
        else:
            mm = output.multimodal_output or output.outputs[0].multimodal_output
            reply["audio"] = concat_audio(mm["audio"] if mm.get("audio") is not None else mm["model_outputs"])

    assert len(replies) == len(prompts)
    for rid, reply in replies.items():
        frames = reply["ids"].count(AUDIO_OUT) - 1
        assert frames > 0, f"{rid}: no speech frames"
        assert reply["audio"].size == frames * SAMPLES_PER_FRAME, (
            f"{rid}: {reply['audio'].size} samples, {frames} frames"
        )
        assert float(np.sqrt(np.mean(np.square(reply["audio"])))) > 1e-3, f"{rid}: near-silent speech"
    chat = next(r for r in replies.values() if r["text"].strip())
    assert _words(chat["text"]), "voice chat reply has no text"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", [TEXT], indirect=True)
def test_asr(omni_runner: OmniRunner, run_level: str) -> None:
    outputs = omni_runner.omni.generate([_prompt("Perform ASR in English.", "<|audio|>", audio=True)])
    text = outputs[0].outputs[0].text
    wer = _wer(text, REFERENCE)
    print(f"[trend] LFM3-Audio ASR WER: {wer:.3f} ({text!r})")
    assert wer <= 0.5, f"transcript far from the reference (WER={wer:.2f}): {text!r}"
