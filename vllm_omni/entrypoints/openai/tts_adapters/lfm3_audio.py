# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

# Speech languages and locales of the model card; "English" means American English.
LANGUAGES = {
    name.lower(): name
    for name in (
        "Arabic",
        "Danish",
        "German",
        "Greek",
        "British English",
        "American English",
        "Finnish",
        "French",
        "Hebrew",
        "Italian",
        "Japanese",
        "Korean",
        "Dutch",
        "Polish",
        "Portuguese",
        "Spanish",
        "Swedish",
        "Turkish",
        "Chinese",
    )
} | {"english": "American English"}
VOICES = ("female", "male")


@register_tts_adapter
class Lfm3AudioAdapter(ARTTSAdapter):
    """LFM3-Audio speaks in a language and a female or male voice named by the system prompt.

    The assistant turn is prefilled with ``<|audio_start|>``, so the model answers in speech only.
    """

    stage_keys = frozenset({"lfm3_audio"})
    name = "lfm3_audio"

    @classmethod
    def stage_serves_speech(cls, model_stage: str | None, all_stage_keys: frozenset[str]) -> bool:
        # The lfm3_audio_text deployment has no codec and answers in text only.
        return "lfm3_audio_code2wav" in all_stage_keys

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is not None:
            return "LFM3-Audio has no voice cloning; use voice='female' or 'male'."
        if (request.voice or "female").lower() not in VOICES:
            return f"LFM3-Audio voices are {', '.join(VOICES)}; got voice={request.voice!r}."
        if request.language is not None and request.language.lower() not in LANGUAGES:
            return (
                f"Unsupported language {request.language!r}. Supported: {', '.join(sorted(set(LANGUAGES.values())))}."
            )
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        language = LANGUAGES[(request.language or "American English").lower()]
        voice = (request.voice or "female").lower()
        prompt = (
            "<|startoftext|><|im_start|>system\n"
            f"Perform TTS in {language} using a {voice} voice.<|im_end|>\n"
            f"<|im_start|>user\n{request.input}<|im_end|>\n"
            "<|im_start|>assistant\n<|audio_start|>"
        )
        return PreparedRequest(prompt={"prompt": prompt}, tts_params={}, model_type="lfm3_audio")
