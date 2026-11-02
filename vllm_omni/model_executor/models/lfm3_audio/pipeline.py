# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio pipeline topologies (frozen).

``lfm3_audio``: speech / text in -> text and speech out. Stage 0 (language
model + depth decoder) streams 12-codebook frames to stage 1 (Liquid codec,
``audio_tokenizer/``), which returns 24 kHz audio.

``lfm3_audio_text``: speech / text in -> text out (ASR, spoken QA) on the
language model stage alone.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.lfm3_audio"
# <|im_end|>. generation_config.json also lists <|text_end|> as an end of sequence, but inside an
# interleaved block it only ends the text and the speech tail follows, so the speech stage stops
# on <|im_end|> alone and the model forces <|im_end|> after a text_end outside a block.
_IM_END = 124900

LFM3_AUDIO_PIPELINE = PipelineConfig(
    model_type="lfm3_audio",
    model_arch="Lfm3AudioForConditionalGeneration",
    default_deploy_config_name="lfm3_audio.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="lfm3_audio",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            model_arch="Lfm3AudioForConditionalGeneration",
            async_chunk_process_next_stage_input_func=f"{_PROC}.lfm3_audio_to_codec_async_chunk",
            sampling_constraints={"detokenize": True, "ignore_eos": True, "stop_token_ids": [_IM_END]},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Lfm3AudioCode2Wav",
            # The codec ships its own config and weights; the tokenizer is the checkpoint's.
            model_subdir="audio_tokenizer",
            tokenizer_subdir="",
            sampling_constraints={"detokenize": False},
        ),
    ),
)

LFM3_AUDIO_TEXT_PIPELINE = PipelineConfig(
    model_type="lfm3_audio_text",
    model_arch="Lfm3AudioForConditionalGeneration",
    default_deploy_config_name="lfm3_audio_text.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="lfm3_audio",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Lfm3AudioForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
