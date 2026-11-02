# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Plain PyTorch LFM3 layers for the small LFM3 stacks of LFM3-Audio.

The language model runs on vLLM's paged ``Lfm3Model``. Three smaller LFM3
stacks run outside the KV cache manager and use these layers instead:

* the Liquid codec decoder: causal, sliding window 30, streamed chunk by chunk;
* the audio encoder: an LFM2 stack (no output gate, RoPE in the model dtype)
  with causal sliding window attention over the downsampled mel frames;
* the depth decoder: bidirectional attention over the 12 codebook slots.

Parameter names match the checkpoint (``layers.N.conv.in_proj`` ...), and the
numerics follow the reference implementation: RMSNorm and RoPE run in float32,
the rotary tables are rounded to the model dtype first, and partial rotation
zeroes the tail frequencies instead of shrinking the rotated width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Lfm3StackSpec:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    norm_eps: float = 1e-5
    conv_width: int = 7
    conv_bias: bool = False
    sandwich_norm: bool = False
    sliding_window: int | None = None
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 1.0
    causal: bool = True
    # LFM3 gates each attention head and rotates in float32; LFM2 (the audio
    # encoder's stack) does neither.
    output_gate: bool = True
    rope_float32: bool = True
    # The rotary tables are rounded to the model dtype, except in the Liquid
    # codec, whose reference keeps them in float32.
    rope_table_float32: bool = False

    @classmethod
    def from_hf(cls, config: Any, *, causal: bool = True) -> Lfm3StackSpec:
        """Build from an HF ``Lfm3Config`` or ``Lfm2Config``."""
        layer_types = tuple(config.layer_types)
        rope = getattr(config, "rope_parameters", None) or {}
        attention_type = next((t for t in layer_types if t != "conv"), None)
        rope_layer = rope.get(attention_type, rope) if attention_type else {}
        lfm3 = getattr(config, "model_type", "lfm3") != "lfm2"
        intermediate_size = config.intermediate_size
        if getattr(config, "block_auto_adjust_ff_dim", False):
            # LFM2 sizes the SwiGLU to 2/3 of intermediate_size, rounded up to block_multiple_of.
            intermediate_size = int(2 * intermediate_size / 3)
            if getattr(config, "block_ffn_dim_multiplier", None) is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
            multiple = config.block_multiple_of
            intermediate_size = multiple * ((intermediate_size + multiple - 1) // multiple)
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
            layer_types=layer_types,
            norm_eps=config.norm_eps,
            conv_width=config.conv_L_cache,
            conv_bias=bool(getattr(config, "conv_bias", False)),
            sandwich_norm=bool(getattr(config, "use_sandwich_norm", False)),
            sliding_window=getattr(config, "sliding_window", None),
            rope_theta=float(rope_layer.get("rope_theta", 1_000_000.0)),
            partial_rotary_factor=float(rope_layer.get("partial_rotary_factor", 1.0)),
            causal=causal,
            output_gate=lfm3,
            rope_float32=lfm3,
        )


@dataclass
class Lfm3StackState:
    """Streaming state of one sequence: conv windows, attention KV and the token count."""

    layers: dict[int, Any] = field(default_factory=dict)
    offset: int = 0


class Lfm3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def lfm3_inv_freq(spec: Lfm3StackSpec) -> torch.Tensor:
    """Default RoPE frequencies; partial rotation keeps the full-width progression and zeroes the tail."""
    dim = spec.head_dim
    angles = int(spec.partial_rotary_factor * dim) // 2
    inv_freq = 1.0 / (spec.rope_theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / dim))
    return torch.cat((inv_freq, torch.zeros(dim // 2 - angles)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Lfm3Attention(nn.Module):
    """Attention with QK RMSNorm, an optional sigmoid output gate per head and an optional sliding window.

    LFM2 checkpoints name the output projection and the QK norms ``out_proj`` /
    ``q_layernorm`` / ``k_layernorm``; the owning model maps them on load.
    """

    def __init__(self, spec: Lfm3StackSpec, sliding_window: int | None) -> None:
        super().__init__()
        h, d = spec.hidden_size, spec.head_dim
        self.num_heads, self.num_kv_heads, self.head_dim = spec.num_heads, spec.num_kv_heads, d
        self.causal = spec.causal
        self.sliding_window = sliding_window
        self.rope_float32 = spec.rope_float32
        self.q_proj = nn.Linear(h, spec.num_heads * d, bias=False)
        self.k_proj = nn.Linear(h, spec.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, spec.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(spec.num_heads * d, h, bias=False)
        self.q_norm = Lfm3RMSNorm(d, spec.norm_eps)
        self.k_norm = Lfm3RMSNorm(d, spec.norm_eps)
        self.gate_proj = nn.Linear(h, spec.num_heads, bias=False) if spec.output_gate else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # cos / sin are [tokens, head_dim] and broadcast over [batch, heads, tokens, head_dim].
        if self.rope_float32:
            cos32, sin32 = cos.float(), sin.float()
            q = (q.float() * cos32 + _rotate_half(q.float()) * sin32).to(x.dtype)
            k = (k.float() * cos32 + _rotate_half(k.float()) * sin32).to(x.dtype)
        else:
            q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

        key_positions = positions
        if cache is not None and "k" in cache:
            k = torch.cat((cache["k"], k), dim=2)
            v = torch.cat((cache["v"], v), dim=2)
            key_positions = torch.cat((cache["positions"], positions))
        if cache is not None:
            # Keep what the next chunk can still attend to: the last window - 1 keys.
            keep = k.shape[2] if self.sliding_window is None else self.sliding_window - 1
            cache["k"], cache["v"], cache["positions"] = k[:, :, -keep:], v[:, :, -keep:], key_positions[-keep:]

        mask = None
        if self.causal or self.sliding_window is not None:
            distance = positions[:, None] - key_positions[None, :]
            mask = torch.ones_like(distance, dtype=torch.bool)
            if self.causal:
                mask &= distance >= 0
            if self.sliding_window is not None:
                mask &= distance.abs() < self.sliding_window
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.head_dim**-0.5, enable_gqa=self.num_kv_heads != self.num_heads
        )
        out = out.transpose(1, 2)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        return self.o_proj(out.reshape(b, t, -1))


class Lfm3ShortConv(nn.Module):
    """Gated short convolution: ``out_proj(C * conv(B * x))`` with a causal depthwise kernel."""

    def __init__(self, spec: Lfm3StackSpec) -> None:
        super().__init__()
        h = spec.hidden_size
        self.width = spec.conv_width
        self.conv = nn.Conv1d(h, h, kernel_size=spec.conv_width, groups=h, bias=spec.conv_bias)
        self.in_proj = nn.Linear(h, 3 * h, bias=spec.conv_bias)
        self.out_proj = nn.Linear(h, h, bias=spec.conv_bias)

    def forward(self, x: torch.Tensor, cache: dict[str, torch.Tensor] | None) -> torch.Tensor:
        b_gate, c_gate, x_in = self.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
        bx = b_gate * x_in
        if cache is not None and "state" in cache:
            padded = torch.cat((cache["state"], bx), dim=-1)
        else:
            padded = F.pad(bx, (self.width - 1, 0))
        if cache is not None:
            cache["state"] = padded[..., -(self.width - 1) :]
        out = self.conv(padded.to(self.conv.weight.dtype)).to(bx.dtype)
        return self.out_proj((c_gate * out).transpose(-1, -2).contiguous())


class Lfm3MLP(nn.Module):
    def __init__(self, spec: Lfm3StackSpec) -> None:
        super().__init__()
        self.w1 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.w3 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.w2 = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm3Layer(nn.Module):
    def __init__(self, spec: Lfm3StackSpec, layer_type: str) -> None:
        super().__init__()
        self.is_attention = layer_type != "conv"
        if self.is_attention:
            window = spec.sliding_window if layer_type == "sliding_attention" else None
            self.self_attn = Lfm3Attention(spec, window)
        else:
            self.conv = Lfm3ShortConv(spec)
        self.feed_forward = Lfm3MLP(spec)
        self.operator_norm = Lfm3RMSNorm(spec.hidden_size, spec.norm_eps)
        self.ffn_norm = Lfm3RMSNorm(spec.hidden_size, spec.norm_eps)
        self.sandwich = spec.sandwich_norm
        if self.sandwich:
            self.post_operator_norm = Lfm3RMSNorm(spec.hidden_size, spec.norm_eps)
            self.post_ffn_norm = Lfm3RMSNorm(spec.hidden_size, spec.norm_eps)

    def forward(self, x, cos, sin, positions, cache):
        h = self.operator_norm(x)
        h = self.self_attn(h, cos, sin, positions, cache) if self.is_attention else self.conv(h, cache)
        if self.sandwich:
            h = self.post_operator_norm(h)
        x = x + h
        h = self.feed_forward(self.ffn_norm(x))
        if self.sandwich:
            h = self.post_ffn_norm(h)
        return x + h


class Lfm3Stack(nn.Module):
    """LFM3 layers + final ``embedding_norm`` over input embeddings (no token embedding)."""

    def __init__(self, spec: Lfm3StackSpec) -> None:
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(Lfm3Layer(spec, t) for t in spec.layer_types)
        self.embedding_norm = Lfm3RMSNorm(spec.hidden_size, spec.norm_eps)
        self.register_buffer("inv_freq", lfm3_inv_freq(spec), persistent=False)

    def forward(self, x: torch.Tensor, state: Lfm3StackState | None = None) -> torch.Tensor:
        """``x``: ``[batch, tokens, hidden]``. With ``state`` the call continues that stream."""
        offset = 0 if state is None else state.offset
        positions = torch.arange(offset, offset + x.shape[1], device=x.device)
        freqs = positions[:, None].float() * self.inv_freq.float()[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        if not self.spec.rope_table_float32:
            cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for index, layer in enumerate(self.layers):
            cache = None if state is None else state.layers.setdefault(index, {})
            x = layer(x, cos, sin, positions, cache)
        if state is not None:
            state.offset += x.shape[1]
        return self.embedding_norm(x)
