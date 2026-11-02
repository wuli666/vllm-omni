# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Liquid codec decoder: streamed chunks decode to the same audio as the whole utterance."""

import pytest
import torch

from vllm_omni.model_executor.models.lfm3_audio.liquid_codec import LiquidCodecDecoderModel, LiquidCodecStream
from vllm_omni.transformers_utils.configs.lfm3_audio import LiquidCodecConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def codec() -> LiquidCodecDecoderModel:
    torch.manual_seed(0)
    model = LiquidCodecDecoderModel(LiquidCodecConfig()).float().eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return model


def _codes(num_frames: int) -> torch.Tensor:
    return torch.randint(0, 2016, (1, 12, num_frames), generator=torch.Generator().manual_seed(1))


def test_frame_length(codec):
    assert codec.codebook_size == 2016
    assert codec.decode(_codes(7)).shape == (1, 7 * 1920)


@pytest.mark.parametrize("chunks", [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 7], [10], [1, 4, 5]])
def test_stream_matches_full(codec, chunks):
    codes = _codes(sum(chunks))
    full = codec.decode(codes)
    stream, pieces, start = LiquidCodecStream(), [], 0
    for index, size in enumerate(chunks):
        pieces.append(codec.decode(codes[..., start : start + size], stream, finalize=index == len(chunks) - 1))
        start += size
    torch.testing.assert_close(torch.cat(pieces, dim=-1), full, atol=1e-5, rtol=1e-4)
    assert stream.finished


def test_empty_final_chunk_flushes_tail(codec):
    codes = _codes(4)
    stream = LiquidCodecStream()
    head = codec.decode(codes, stream, finalize=False)
    tail = codec.decode(codes[..., :0], stream, finalize=True)
    torch.testing.assert_close(torch.cat([head, tail], dim=-1), codec.decode(codes), atol=1e-5, rtol=1e-4)


def test_finished_stream_is_refused(codec):
    stream = LiquidCodecStream()
    codec.decode(_codes(2), stream, finalize=True)
    with pytest.raises(ValueError, match="finalized"):
        codec.decode(_codes(1), stream)
