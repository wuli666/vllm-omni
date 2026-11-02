# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Config of the Liquid codec shipped in ``audio_tokenizer/`` of LFM3-Audio / LFM3-Omni."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class LiquidCodecConfig(PretrainedConfig):
    model_type = "liquid_codec"

    def __init__(
        self,
        encoder_dim: int = 512,
        encoder_conv_dim: int = 256,
        stft_n_fft: int = 1280,
        stft_hop_length: int = 320,
        stft_win_length: int = 1280,
        stft_frame_stack: int = 6,
        quantizer_dim: int = 256,
        fsq_levels: list[int] | None = None,
        n_q: int = 12,
        decoder_window: int = 30,
        sampling_rate: int = 24000,
        **kwargs: Any,
    ) -> None:
        self.encoder_dim = encoder_dim
        self.encoder_conv_dim = encoder_conv_dim
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_frame_stack = stft_frame_stack
        self.quantizer_dim = quantizer_dim
        self.fsq_levels = list(fsq_levels or [8, 7, 6, 6])
        self.n_q = n_q
        self.decoder_window = decoder_window
        self.sampling_rate = sampling_rate
        super().__init__(**kwargs)


AutoConfig.register("liquid_codec", LiquidCodecConfig)

__all__ = ["LiquidCodecConfig"]
