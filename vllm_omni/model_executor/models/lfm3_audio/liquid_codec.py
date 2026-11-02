# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Liquid codec decoder: 12 FSQ codebooks at 12.5 frames/s -> 24 kHz audio.

Decode path of the checkpoint's ``LiquidCodecModel`` (``audio_tokenizer/``):
FSQ codes -> latents -> 6x upsampled LFM3 stack (sliding window 30) -> log
magnitude and phase -> ISTFT overlap-add. Streaming keeps the LFM3 state and
the unfinished overlap-add tail per stream, so chunks decode to the same audio
as the whole sequence. The analysis half of the codec is not needed for
generation and is left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .lfm3_layers import Lfm3Stack, Lfm3StackSpec, Lfm3StackState


def codec_lfm_spec(config: Any, sliding_window: int | None) -> Lfm3StackSpec:
    # Fixed in the reference ``LiquidCodecConfig.lfm_config``; not part of config.json.
    attention = "full_attention" if sliding_window is None else "sliding_attention"
    return Lfm3StackSpec(
        hidden_size=config.encoder_dim,
        intermediate_size=2304,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        layer_types=tuple(attention if i in (2, 4, 6) else "conv" for i in range(8)),
        norm_eps=1e-5,
        conv_width=7,
        sandwich_norm=True,
        sliding_window=sliding_window,
        rope_theta=1_000_000.0,
    )


@dataclass
class LiquidCodecStream:
    """Decoder state carried between the chunks of one stream."""

    lfm: Lfm3StackState = field(default_factory=Lfm3StackState)
    audio: torch.Tensor | None = None
    envelope: torch.Tensor | None = None
    started: bool = False
    finished: bool = False


class LiquidCodecQuantizer(nn.Module):
    """FSQ dequantizer: each code is a mixed-radix number over ``fsq_levels``."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        levels = list(config.fsq_levels)
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int64), persistent=False)
        basis = [math.prod(levels[:i]) for i in range(len(levels))]
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.int64), persistent=False)
        effective_dim = len(levels) * config.n_q
        self.project_out = (
            nn.Identity() if config.quantizer_dim == effective_dim else nn.Linear(effective_dim, config.quantizer_dim)
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """``[batch, codebooks, frames]`` -> ``[batch, quantizer_dim, frames]``."""
        indices = codes.to(self.basis.device).transpose(1, 2).unsqueeze(-1)
        half_width = self.levels // 2
        latents = ((indices // self.basis) % self.levels - half_width) / half_width
        latents = latents.reshape(*latents.shape[:-2], -1)
        if isinstance(self.project_out, nn.Linear):
            latents = latents.to(self.project_out.weight.dtype)
        return self.project_out(latents).movedim(-1, 1)


class LiquidCodecISTFT(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.n_fft = config.stft_n_fft
        self.hop_length = config.stft_hop_length
        self.win_length = config.stft_win_length
        # Loaded from the checkpoint (float32) and kept in float32.
        self.register_buffer("window", torch.hann_window(self.win_length, dtype=torch.float32))

    def forward(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_frames = spec.shape[-1]
        frames = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
        output_size = (num_frames - 1) * self.hop_length + self.win_length
        fold = {"output_size": (1, output_size), "kernel_size": (1, self.win_length), "stride": (1, self.hop_length)}
        audio = F.fold(frames, **fold)[:, 0, 0]
        envelope = F.fold(self.window.square().expand(1, num_frames, -1).transpose(1, 2), **fold)[0, 0, 0]
        return audio, envelope


class LiquidCodecDecoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.frame_stack = config.stft_frame_stack
        self.lfm = Lfm3Stack(codec_lfm_spec(config, config.decoder_window))
        self.lin = nn.Linear(config.encoder_dim, 2 * (config.stft_n_fft // 2 + 1))
        self.istft = LiquidCodecISTFT(config)

    def forward(self, embeddings: torch.Tensor, state: Lfm3StackState | None) -> tuple[torch.Tensor, torch.Tensor]:
        upsampled = F.interpolate(embeddings.mT, embeddings.shape[1] * self.frame_stack, mode="nearest-exact").mT
        hidden = self.lfm(upsampled, state)
        log_magnitude, angle = torch.chunk(self.lin(hidden).mT.contiguous(), 2, 1)
        return self.istft(torch.polar(log_magnitude.float().exp(), angle.float()))


class LiquidCodecDecoderModel(nn.Module):
    """Decode half of ``LiquidCodecModel``; parameter names match ``audio_tokenizer/model.safetensors``."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.num_codebooks = int(config.n_q)
        self.codebook_size = math.prod(config.fsq_levels)
        self.sample_rate = int(config.sampling_rate)
        self.quantizer = LiquidCodecQuantizer(config)
        self.from_quantizer = nn.Linear(config.quantizer_dim, config.encoder_dim)
        self.decoder = LiquidCodecDecoder(config)

    @torch.inference_mode()
    def decode(
        self, codes: torch.Tensor, stream: LiquidCodecStream | None = None, finalize: bool = True
    ) -> torch.Tensor:
        """``codes``: ``[batch, codebooks, frames]`` in ``[0, codebook_size)``; returns float32 ``[batch, samples]``.

        Without ``stream`` the frames are one whole utterance (1920 samples per
        frame). With ``stream`` the trailing samples that later frames still
        overlap are held back until a call with ``finalize=True``.
        """
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(f"expected codes of shape (batch, {self.num_codebooks}, frames), got {tuple(codes.shape)}")
        if stream is not None and stream.finished:
            raise ValueError("this stream is finalized; start a new one")
        config = self.config
        trim = (config.stft_win_length - config.stft_hop_length) // 2
        tail = 2 * trim
        batch = codes.shape[0]
        window = self.decoder.istft.window
        if not codes.shape[-1] and not (finalize and stream is not None and stream.started):
            if stream is not None and finalize:
                stream.finished = True
            return window.new_zeros((batch, 0))
        if codes.shape[-1]:
            latents = self.quantizer.decode(codes).mT.to(self.from_quantizer.weight.dtype)
            audio, envelope = self.decoder(self.from_quantizer(latents), None if stream is None else stream.lfm)
        else:
            audio, envelope = window.new_zeros(batch, tail), window.new_zeros(tail)
        if stream is None:
            return (audio[:, trim:-trim] / envelope[trim:-trim]).float()
        if stream.audio is not None:
            audio = torch.cat([audio[:, :tail] + stream.audio, audio[:, tail:]], dim=1)
            envelope = torch.cat([envelope[:tail] + stream.envelope, envelope[tail:]])
        if not stream.started:
            audio, envelope = audio[:, trim:], envelope[trim:]
            stream.started = True
        if finalize:
            audio, envelope = audio[:, :-trim], envelope[:-trim]
            stream.audio = stream.envelope = None
            stream.finished = True
        else:
            stream.audio, stream.envelope = audio[:, -tail:], envelope[-tail:]
            audio, envelope = audio[:, :-tail], envelope[:-tail]
        return (audio / envelope).float()
