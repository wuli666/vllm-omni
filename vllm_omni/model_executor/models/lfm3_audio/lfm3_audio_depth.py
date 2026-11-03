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
    def sample_frames(
        self,
        hidden: torch.Tensor,
        samplings: list[AudioSampling],
        histories: list[torch.Tensor | None],
        generators: list[torch.Generator | None],
    ) -> torch.Tensor:
        """One frame ``[B, 12]`` per hidden state ``[B, H_lm]``, all requests in one depth pass per step.

        ``histories[i] [frames, 12]`` holds earlier frames of request ``i`` for the per-codebook repetition
        penalty. Every random draw of a row comes from that row's generator (or the global one), so a seeded
        request samples the same frame whatever else is in the batch.
        """
        rows, k, device = hidden.shape[0], self.num_codebooks, hidden.device
        vocab = self.depth_embeddings[0].embedding.num_embeddings
        # A code seen in the window is penalized once, whatever its count.
        seen = torch.zeros((rows, k, vocab), dtype=torch.bool, device=device)
        for i, history in enumerate(histories):
            if history is not None and history.numel() and samplings[i].repetition_penalty != 1.0:
                seen[i].scatter_(1, history.T.to(device), True)
        penalty = to_device([s.repetition_penalty for s in samplings], device, torch.float32)[:, None, None]
        temperature = to_device([s.temperature for s in samplings], device, torch.float32)[:, None, None]
        greedy = temperature[:, 0, 0] == 0
        top_k = [min(s.top_k, vocab) if s.top_k else vocab for s in samplings]
        kth_index = to_device([k - 1 for k in top_k], device, torch.long)[:, None, None]

        codes = torch.zeros((rows, k), dtype=torch.long, device=device)
        revealed = torch.zeros_like(codes, dtype=torch.bool)
        order = _draw(generators, (k,), device, torch.Tensor.uniform_).argsort(-1)  # a random order per row
        start = 0
        for size in self.schedule:
            pick = order[:, start : start + size]
            start += size
            scores = self(hidden, codes, ~revealed).float().gather(1, pick[..., None].expand(-1, -1, vocab))
            penalized = torch.where(scores < 0, scores * penalty, scores / penalty)
            scores = torch.where(seen.gather(1, pick[..., None].expand(-1, -1, vocab)), penalized, scores)
            scaled = scores / temperature.clamp_min(1e-5)
            kth = scaled.topk(max(top_k), dim=-1).values.gather(-1, kth_index.expand(-1, size, 1))
            probs = scaled.masked_fill(scaled < kth, float("-inf")).softmax(-1)
            # Exponential race: argmax(p / E) with E ~ Exp(1) draws from p.
            drawn = (probs / _draw(generators, (size, vocab), device, torch.Tensor.exponential_)).argmax(-1)
            drawn = torch.where(greedy[:, None], scores.argmax(-1), drawn)
            codes.scatter_(1, pick, drawn)
            revealed.scatter_(1, pick, True)
        return codes


def to_device(values: list, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A small host list on ``device`` without a stream sync (a pageable host copy waits for the GPU)."""
    host = torch.tensor(values, dtype=dtype, pin_memory=device.type == "cuda")
    return host.to(device, non_blocking=True)


def _draw(generators: list[torch.Generator | None], shape: tuple[int, ...], device, fill) -> torch.Tensor:
    """Per-row random tensor ``[rows, *shape]``: seeded rows draw from their own generator."""
    out = torch.empty((len(generators), *shape), device=device)
    if any(generator is None for generator in generators):
        fill(out)
    for i, generator in enumerate(generators):
        if generator is not None:
            fill(out[i], generator=generator)
    return out
