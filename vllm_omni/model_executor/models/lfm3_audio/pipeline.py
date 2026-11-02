# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio pipeline topologies (frozen).

``lfm3_audio_text``: speech / text in -> text out (ASR, spoken QA) on the
language model stage alone.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
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
