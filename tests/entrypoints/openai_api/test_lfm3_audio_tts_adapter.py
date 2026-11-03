# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the LFM3-Audio speech adapter: detection, the reject matrix and the prompt."""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.lfm3_audio import Lfm3AudioAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _adapter() -> Lfm3AudioAdapter:
    return Lfm3AudioAdapter(SpeechServingContext(server=SimpleNamespace(), engine_client=SimpleNamespace()))


def test_detection_and_speech_topology():
    assert resolve_adapter("lfm3_audio") is Lfm3AudioAdapter
    assert detect_tts_model_type("lfm3_audio", "Lfm3AudioForConditionalGeneration") == "lfm3_audio"
    # The text-only deployment shares the stage key but has no codec.
    assert Lfm3AudioAdapter.stage_serves_speech("lfm3_audio", frozenset({"lfm3_audio", "lfm3_audio_code2wav"}))
    assert not Lfm3AudioAdapter.stage_serves_speech("lfm3_audio", frozenset({"lfm3_audio"}))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"input": " "}, "empty"),
        ({"voice": "alloy"}, "voices"),
        ({"language": "Klingon"}, "Unsupported language"),
        ({"ref_audio": "data:audio/wav;base64,AAAA"}, "voice cloning"),
    ],
)
def test_rejects(kwargs, error):
    request = OpenAICreateSpeechRequest(**{"input": "Hello.", **kwargs})
    assert error in _adapter().validate(request)


def test_prompt_names_language_and_voice_and_prefills_audio_start():
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="Hello.", voice="Male", language="english")
    assert adapter.validate(request) is None
    prepared = asyncio.run(adapter.build(request, sampling_params_list=[], has_inline_ref_audio=False))
    assert prepared.prompt["prompt"] == (
        "<|startoftext|><|im_start|>system\nPerform TTS in American English using a male voice.<|im_end|>\n"
        "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"
    )
    default = asyncio.run(adapter.build(OpenAICreateSpeechRequest(input="Hi."), [], False))
    assert "Perform TTS in American English using a female voice." in default.prompt["prompt"]
