# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio checkpoint layouts map onto one module tree; the LFM2 encoder stack sizes its MLP like LFM2."""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.lfm3_audio.lfm3_audio import Lfm3AudioForConditionalGeneration, num_audio_tokens
from vllm_omni.model_executor.models.lfm3_audio.lfm3_layers import Lfm3StackSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# (LFM3-Audio training export, Transformers layout used by LFM3-Omni, vLLM module name)
LAYOUTS = [
    (
        "lfm.model.layers.2.self_attn.o_proj.weight",
        "model.language_model.layers.2.self_attn.o_proj.weight",
        "language_model.model.layers.2.self_attn.o_proj.weight",
    ),
    (
        "encoder.lfm.lfm.layers.2.self_attn.out_proj.weight",
        "model.audio_encoder.lfm.layers.2.self_attn.out_proj.weight",
        "audio_encoder.lfm.layers.2.self_attn.o_proj.weight",
    ),
    (
        "encoder.lfm.lfm.layers.2.self_attn.q_layernorm.weight",
        "model.audio_encoder.lfm.layers.2.self_attn.q_layernorm.weight",
        "audio_encoder.lfm.layers.2.self_attn.q_norm.weight",
    ),
    (
        "encoder.lfm.lfm.layers.0.conv.out_proj.weight",
        "model.audio_encoder.lfm.layers.0.conv.out_proj.weight",
        "audio_encoder.lfm.layers.0.conv.out_proj.weight",
    ),
    (
        "encoder.downsample.lin_out.weight",
        "model.audio_encoder.downsample.lin_out.weight",
        "audio_encoder.downsample.lin_out.weight",
    ),
    ("mlp.model.0.weight", "model.audio_projector.layer_norm.weight", "audio_projector.layer_norm.weight"),
    ("mlp.model.3.bias", "model.audio_projector.linear_2.bias", "audio_projector.linear_2.bias"),
    (
        "depthformer.model.layers.1.self_attn.gate_proj.weight",
        "model.depth_decoder.model.layers.1.self_attn.gate_proj.weight",
        "depth_decoder.model.layers.1.self_attn.gate_proj.weight",
    ),
    (
        "depth_embeddings.7.embedding.weight",
        "model.depth_decoder.depth_embeddings.7.embedding.weight",
        "depth_decoder.depth_embeddings.7.embedding.weight",
    ),
    ("mask_embedding", "model.depth_decoder.mask_embedding", "depth_decoder.mask_embedding"),
]


@pytest.mark.parametrize("export_name,hf_name,expected", LAYOUTS)
def test_both_layouts_map_to_the_same_module(export_name, hf_name, expected):
    mapper = Lfm3AudioForConditionalGeneration.hf_to_vllm_mapper
    assert mapper._map_name(export_name) == expected
    assert mapper._map_name(hf_name) == expected


def test_encoder_stack_is_lfm2():
    # LFM3-Audio-2B's encoder stack: intermediate_size 2560, auto-adjusted to 2/3 and rounded to 256.
    config = SimpleNamespace(
        model_type="lfm2",
        hidden_size=768,
        intermediate_size=2560,
        num_attention_heads=8,
        num_key_value_heads=4,
        layer_types=["conv", "full_attention"],
        norm_eps=1e-5,
        conv_L_cache=3,
        rope_parameters={"rope_theta": 1e6, "rope_type": "default"},
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=256,
    )
    spec = Lfm3StackSpec.from_hf(config)
    assert spec.intermediate_size == 1792
    assert spec.head_dim == 96
    assert not spec.output_gate and not spec.rope_float32


def test_one_audio_token_per_eight_frames():
    assert [num_audio_tokens(n) for n in (1, 8, 9, 480)] == [1, 1, 2, 60]
