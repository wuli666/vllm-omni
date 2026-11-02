# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request text/speech mode machine of LFM3-Audio generation.

Mirrors the reference ``Lfm3AudioForConditionalGeneration._generate``:

* ``text``: the language model samples text. ``<|audio_start|>`` switches to
  ``audio``, ``<|mixed_start|>`` to ``interleaved``.
* ``audio``: every slot is a speech frame until the depth decoder emits an
  end-of-audio frame, which returns to ``text``.
* ``interleaved``: ``interleaved_n_text`` text slots and ``interleaved_n_audio``
  speech slots alternate until ``<|text_end|>``; the remaining speech then runs
  until an end-of-audio frame.

The state lives in the request's model buffer as plain values (``as_updates``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"


@dataclass(frozen=True)
class Lfm3AudioTokens:
    audio_start: int
    mixed_start: int
    text_end: int
    audio_out: int
    eos_ids: tuple[int, ...]
    n_text: int
    n_audio: int

    @classmethod
    def from_config(cls, config: Any, eos_ids: tuple[int, ...]) -> Lfm3AudioTokens:
        return cls(
            audio_start=config.audio_start_token_id,
            mixed_start=config.mixed_start_token_id,
            text_end=config.text_end_token_id,
            audio_out=config.audio_out_token_id,
            eos_ids=eos_ids,
            n_text=config.interleaved_n_text,
            n_audio=config.interleaved_n_audio,
        )


@dataclass
class Lfm3AudioModeState:
    mode: str = TEXT
    speaking: bool = False
    slots_left: int = 0
    text_done: bool = False
    # A text-end outside an interleaved block ends the turn; the next slot is forced to end of sequence.
    finished: bool = False

    @classmethod
    def from_prompt(cls, last_token: int, tokens: Lfm3AudioTokens) -> Lfm3AudioModeState:
        """A prompt ending on a mode marker starts in that mode."""
        if last_token in (tokens.audio_start, tokens.audio_out):
            return cls(mode=AUDIO)
        if last_token == tokens.mixed_start:
            return cls(mode=INTERLEAVED, slots_left=tokens.n_text)
        return cls()

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> Lfm3AudioModeState:
        return cls(
            mode=info.get("lfm3_mode", TEXT),
            speaking=bool(info.get("lfm3_speaking", False)),
            slots_left=int(info.get("lfm3_slots_left", 0)),
            text_done=bool(info.get("lfm3_text_done", False)),
            finished=bool(info.get("lfm3_finished", False)),
        )

    def as_updates(self) -> dict[str, Any]:
        return {
            "lfm3_mode": self.mode,
            "lfm3_speaking": self.speaking,
            "lfm3_slots_left": self.slots_left,
            "lfm3_text_done": self.text_done,
            "lfm3_finished": self.finished,
        }

    @property
    def speech_slot(self) -> bool:
        """Whether the next slot is a speech frame."""
        return self.mode == AUDIO or (self.mode == INTERLEAVED and self.speaking)

    def after_text(self, token: int, tokens: Lfm3AudioTokens) -> None:
        interleaving = self.mode == INTERLEAVED
        if token == tokens.text_end:
            if interleaving:
                self.text_done, self.speaking = True, True
            else:
                self.finished = True
        elif token in tokens.eos_ids or interleaving:
            pass  # end of turn is the sampler's stop; inside a block the markers are ordinary text
        elif token == tokens.audio_start:
            self.mode = AUDIO
        elif token == tokens.mixed_start:
            self.mode, self.slots_left, self.speaking, self.text_done = INTERLEAVED, tokens.n_text, False, False
        self._count_slot(interleaving, tokens)

    def after_frame(self, end_of_audio: bool, tokens: Lfm3AudioTokens) -> None:
        interleaving = self.mode == INTERLEAVED
        if end_of_audio:
            self.mode = TEXT
        self._count_slot(interleaving, tokens)

    def _count_slot(self, interleaving: bool, tokens: Lfm3AudioTokens) -> None:
        if interleaving and not self.text_done:
            self.slots_left -= 1
            if self.slots_left == 0:
                self.speaking = not self.speaking
                self.slots_left = tokens.n_audio if self.speaking else tokens.n_text
