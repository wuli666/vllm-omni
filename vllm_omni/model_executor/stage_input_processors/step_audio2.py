# SPDX-License-Identifier: Apache-2.0
"""Stage input processor for Step-Audio2: Thinker → Token2Wav transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    DEFAULT_TOKEN_CONFIG,
)

logger = init_logger(__name__)


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


# =========================
# Async chunk processor
# =========================


def thinker2token2wav_async_chunk(
    connector: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """
    Async chunk processor: stream audio tokens from Thinker to Token2Wav.

    Unlike Qwen3-Omni which passes hidden states via pooling_output,
    Step Audio2 generates audio tokens as part of the LLM autoregressive
    output (token IDs >= audio_start). This processor extracts those audio
    tokens from the running token stream and sends them in chunks.

    Args:
        connector: Shared connector object with per-request state buffers.
        pooling_output: Pooler output dict (unused — audio tokens are in
            the token ID stream, not in hidden states).
        request: Current engine request with access to all_token_ids.

    Returns:
        dict with "audio_tokens" (list[int]) and "finished" (torch.bool),
        or None if the chunk is not yet full and generation is ongoing.
    """
    audio_start = DEFAULT_TOKEN_CONFIG.audio_start
    audio_eos = DEFAULT_TOKEN_CONFIG.audio_eos

    # Get all token IDs generated so far (prompt + decode)
    all_token_ids = _ensure_list(request.all_token_ids)

    # Extract audio tokens and convert to 0-based IDs for Token2Wav
    audio_tokens = [tid - audio_start for tid in all_token_ids if tid >= audio_start]
    # Remove padding / EOS tokens
    audio_tokens = [t for t in audio_tokens if t < audio_eos]

    # Determine how many audio tokens we've already sent.
    # We use the connector's defaultdict(list) buffer to track sent tokens;
    # len() gives the count of previously sent tokens.
    request_id = request.external_req_id
    already_sent = len(connector.code_prompt_token_ids[request_id])
    new_tokens = audio_tokens[already_sent:]

    chunk_size = 25
    if len(new_tokens) < chunk_size and not request.is_finished():
        # Not enough new tokens for a chunk yet, wait for more
        return None

    if not new_tokens and not request.is_finished():
        return None

    # Record the tokens we're about to send
    connector.code_prompt_token_ids[request_id].extend(new_tokens)

    info = {
        "audio_tokens": new_tokens,
        "finished": torch.tensor(request.is_finished(), dtype=torch.bool),
    }
    return info


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
