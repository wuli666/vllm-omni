# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio speech encoder and projector.

128-bin log-mel frames at 100 Hz -> three stride-2 convolutions (8x in time)
-> an LFM2 stack with causal sliding window attention -> a two-layer MLP into
the language model width. One embedding per 8 mel frames (12.5 Hz).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import nn

from .lfm3_layers import Lfm3Stack, Lfm3StackSpec


class Lfm3AudioDownsampler(nn.Module):
    """Stride-2 convolutions over the (mel, time) plane: 128 mel bins x 8 frames become one vector."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        channels = config.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, channels, **conv)
        self.conv_mid = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_mid = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_end = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_end = nn.Conv2d(channels, channels, kernel_size=1)
        self.lin_out = nn.Conv1d(config.num_mel_bins * channels // 8, config.hidden_size, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[batch, mel, frames]`` (unpadded) -> ``[batch, hidden, ceil(frames / 8)]``."""
        hidden = self.act(self.conv_in(features.unsqueeze(1)))
        hidden = self.act(self.lin_mid(self.conv_mid(hidden)))
        hidden = self.act(self.lin_end(self.conv_end(hidden)))
        return self.lin_out(hidden.flatten(1, 2))


class Lfm3AudioEncoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.downsample = Lfm3AudioDownsampler(config)
        spec = Lfm3StackSpec.from_hf(config.text_config, causal=True)
        # The stack's attention layers are declared full_attention but always run with the encoder's window.
        spec = replace(
            spec,
            layer_types=tuple("conv" if t == "conv" else "sliding_attention" for t in spec.layer_types),
            sliding_window=config.sliding_window,
        )
        self.lfm = Lfm3Stack(spec)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """One clip's ``[mel, frames]`` features -> ``[ceil(frames / 8), hidden]``."""
        features = features.to(self.downsample.conv_in.weight.dtype).unsqueeze(0)
        return self.lfm(self.downsample(features).transpose(1, 2))[0]


class Lfm3AudioProjector(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(out_features, out_features)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(self.layer_norm(audio_features))))
