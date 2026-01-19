# SPDX-License-Identifier: Apache-2.0
"""Stage input processor for Step-Audio2: Thinker → Token2Wav transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    DEFAULT_TOKEN_CONFIG,
)

logger = init_logger(__name__)


def thinker2token2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create token2wav inputs.

    Workflow:
    1. Extract generated token IDs from thinker output
    2. Separate audio tokens from text tokens
    3. Package audio tokens for token2wav stage

    Step-Audio2 token ranges:
    - Text tokens: 0 - 151688
    - Audio tokens: 151696 - 158257 (vocab size 6562)

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data (unused for Token2Wav in default-voice mode)
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for token2wav stage

    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    thinker_outputs = stage_list[source_stage_id].engine_outputs
    token2wav_inputs = []

    # Get StepAudio2 config from stage (with defaults as fallback)
    def _get_config_value(config, attr_name: str, default_value):
        """Get value from config, fallback to default if not present"""
        if config is None:
            return default_value
        return getattr(config, attr_name, default_value)

    # Try to get config from source stage
    source_stage = stage_list[source_stage_id]
    config = None
    if hasattr(source_stage, "model") and hasattr(source_stage.model, "config"):
        config = source_stage.model.config
    elif hasattr(source_stage, "config"):
        config = source_stage.config

    # Get token configuration from config (with defaults from constants)
    audio_start = _get_config_value(config, "audio_start", DEFAULT_TOKEN_CONFIG.audio_start)
    audio_eos = _get_config_value(config, "audio_eos", DEFAULT_TOKEN_CONFIG.audio_eos)  # Relative to audio start

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]

        # Get all generated tokens (prompt + generated)
        # Device management: Move tensors to CPU and convert to list for cross-process serialization
        prompt_ids = thinker_output.prompt_token_ids
        gen_ids = output.token_ids

        # Convert to CPU list if needed (for multi-process/multi-GPU setup)
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.cpu().tolist()
        if isinstance(gen_ids, torch.Tensor):
            gen_ids = gen_ids.cpu().tolist()

        all_token_ids = prompt_ids + gen_ids

        # Separate audio tokens from text tokens
        # Audio tokens are >= audio_start
        audio_tokens = [
            tid - audio_start  # Convert to 0-based for Token2Wav
            for tid in all_token_ids
            if tid >= audio_start
        ]

        # Remove padding tokens (anything >= EOS)
        audio_tokens = [t for t in audio_tokens if t < audio_eos]

        if not audio_tokens:
            # No audio tokens generated, skip Token2Wav for this request
            logger.info(f"Request {i}: No audio tokens generated, skipping Token2Wav stage")
            continue

        logger.debug("Creating Token2Wav input with %d audio tokens", len(audio_tokens))

        token2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_tokens,  # Pass original tokens (no encoding)
                multi_modal_data=None,
                additional_information=None,  # Not used due to framework limitation
            )
        )

    return token2wav_inputs
