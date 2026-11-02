# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio depth decoder: one language-model hidden state -> one 12-codebook speech frame.

Each codebook gets a slot of the projected hidden state plus its known code
embedding or a learned mask embedding; a bidirectional LFM3 stack over the 12
slots scores the masked codebooks. A frame is revealed MaskGIT style: the
codebooks in a random order, ``maskgit_schedule`` of them per pass (4, 4, 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .lfm3_layers import Lfm3RMSNorm, Lfm3Stack, Lfm3StackSpec


@dataclass(frozen=True)
class AudioSampling:
    """Speech-code sampling; temperature 0 picks the most likely code."""

    temperature: float = 1.0
    top_k: int = 128
    repetition_penalty: float = 1.1
    repetition_window: int = 4096


class Lfm3AudioCodebookEmbedding(nn.Module):
    """Input embedding and tied output head of one codebook."""

    def __init__(self, codebook_size: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, hidden_size)
        self.embedding_norm = Lfm3RMSNorm(hidden_size, eps)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.embedding_norm(hidden), self.embedding.weight)


class Lfm3AudioDepthDecoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        depth = config.depth_config
        self.num_codebooks = config.num_codebooks
        self.hidden_size = depth.hidden_size
        self.schedule = list(config.maskgit_schedule)
        self.depth_linear = nn.Linear(config.text_config.hidden_size, self.num_codebooks * self.hidden_size)
        self.mask_embedding = nn.Parameter(torch.empty(self.hidden_size))
        self.model = Lfm3Stack(Lfm3StackSpec.from_hf(depth, causal=False))
        self.depth_embeddings = nn.ModuleList(
            Lfm3AudioCodebookEmbedding(config.codebook_size, self.hidden_size, depth.norm_eps)
            for _ in range(self.num_codebooks)
        )

    def forward(self, hidden: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        """``hidden [B, H_lm]``, ``codes``/``masked [B, 12]`` -> logits ``[B, 12, codebook_size]``."""
        depth_in = self.depth_linear(hidden).unflatten(-1, (self.num_codebooks, self.hidden_size))
        code_embeds = torch.stack([emb.embedding(codes[:, k]) for k, emb in enumerate(self.depth_embeddings)], dim=1)
        depth_in = depth_in + torch.where(masked[..., None], self.mask_embedding.to(code_embeds.dtype), code_embeds)
        depth_out = self.model(depth_in)
        return torch.stack([emb.logits(depth_out[:, k]) for k, emb in enumerate(self.depth_embeddings)], dim=1)

    @torch.inference_mode()
    def sample_frame(
        self,
        hidden: torch.Tensor,
        sampling: AudioSampling,
        history: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One frame ``[12]`` from one hidden state ``[H_lm]``.

        ``history [frames, 12]`` holds earlier frames of the request for the
        per-codebook repetition penalty.
        """
        hidden = hidden.reshape(1, -1)
        device = hidden.device
        codes = torch.zeros((1, self.num_codebooks), dtype=torch.long, device=device)
        revealed = torch.zeros_like(codes, dtype=torch.bool)
        order = torch.randperm(self.num_codebooks, device=device, generator=generator)
        for pick in order.split(self.schedule):
            scores = self(hidden, codes, ~revealed)[0, pick].float()  # [len(pick), codebook_size]
            if history is not None and history.numel() and sampling.repetition_penalty != 1.0:
                seen = history[:, pick].T  # [len(pick), frames]
                score = scores.gather(1, seen)
                score = torch.where(score < 0, score * sampling.repetition_penalty, score / sampling.repetition_penalty)
                scores = scores.scatter(1, seen, score)
            if sampling.temperature == 0:
                drawn = scores.argmax(-1)
            else:
                scores = scores / sampling.temperature
                if sampling.top_k:
                    kth = scores.topk(min(sampling.top_k, scores.shape[-1]), dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < kth, float("-inf"))
                drawn = torch.multinomial(scores.softmax(-1), 1, generator=generator)[:, 0]
            codes[0, pick] = drawn
            revealed[0, pick] = True
        return codes[0]
