# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM-Omni for running offline inference
with Step-Audio2 model for audio understanding and generation.

Step-Audio2 is a two-stage model:
  - Stage 0 (Thinker): Audio understanding → Text + Audio tokens
  - Stage 1 (Token2Wav): Audio tokens → Waveform (24kHz)

Usage examples:
  # Audio to text (ASR)
  python end2end.py --query-type audio_to_text --audio-path input.wav --model-path "$MODEL_PATH"

  # Text to audio (TTS with default voice)
  python end2end.py --query-type text_to_audio --model-path "$MODEL_PATH"

  # Audio to audio (Voice conversion with default voice)
  python end2end.py --query-type audio_to_audio --audio-path input.wav --model-path "$MODEL_PATH"
"""

import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import librosa
import numpy as np
import soundfile as sf
from vllm.assets.audio import AudioAsset
from vllm.sampling_params import SamplingParams
from vllm.utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def get_audio_to_text_query(
    audio_path: Optional[str] = None,
    question: Optional[str] = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """
    Audio Speech Recognition (ASR) - Convert audio to text.

    Args:
        audio_path: Path to audio file
        question: Question about the audio (optional)
        sampling_rate: Target sampling rate (16kHz for Step-Audio2)

    Returns:
        QueryResult with prompt and audio data
    """
    if question is None:
        question = "请记录下你所听到的语音内容。"

    prompt = (
        f"<|im_start|>system\n{question}<|im_end|>\n<|im_start|>user\n<audio_patch><|im_end|>\n<|im_start|>assistant\n"
    )

    # Load audio
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        # Use default test audio
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate
        # Resample to 16kHz if needed
        if audio_data[1] != sampling_rate:
            audio_signal = librosa.resample(audio_data[0], orig_sr=audio_data[1], target_sr=sampling_rate)
            audio_data = (audio_signal.astype(np.float32), sampling_rate)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_text_to_audio_query(
    text: Optional[str] = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """
    Text-to-Speech (TTS) - Convert text to audio.

    Args:
        text: Text to synthesize
        sampling_rate: Target sampling rate (16kHz for Step-Audio2)

    Returns:
        QueryResult with text prompt and speaker wav path

    Note:
        For audio generation, the prompt must end with "<tts_start>"
        to signal the model to generate audio tokens instead of just text.
    """
    if text is None:
        text = "Hello, this is a test of Step Audio 2 text to speech synthesis."

    prompt = (
        "<|im_start|>system\n"
        "请逐字朗读用户提供的文本，不要改写或补充。<|im_end|>\n"
        "<|im_start|>user\n"
        f"{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<tts_start>"
    )

    inputs = {
        "prompt": prompt,
    }

    return QueryResult(
        inputs=inputs,
        limit_mm_per_prompt={},
    )


def get_audio_to_audio_query(
    audio_path: Optional[str] = None,
    question: Optional[str] = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """
    Audio-to-Audio - Voice conversion/cloning.

    Args:
        audio_path: Path to source audio file
        question: Question/instruction about the audio
        sampling_rate: Target sampling rate (16kHz for Step-Audio2)

    Returns:
        QueryResult with audio input

    Note:
        This mode processes input audio and generates output audio.
        Speaker voice is controlled by STEP_AUDIO2_DEFAULT_PROMPT_WAV env var.
    """
    if question is None:
        question = "请仔细聆听这段语音，然后复述其内容。"

    prompt = (
        "<|im_start|>system\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>user\n"
        "<audio_patch><|im_end|>\n"
        "<|im_start|>assistant\n"
        "<tts_start>"
    )

    # Load source audio
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)

    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate
        if audio_data[1] != sampling_rate:
            audio_signal = librosa.resample(audio_data[0], orig_sr=audio_data[1], target_sr=sampling_rate)
            audio_data = (audio_signal.astype(np.float32), sampling_rate)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,  # ← ONLY this should be processed by Thinker
            },
        },
        limit_mm_per_prompt={"audio": 1},  # ← Only 1 audio input allowed
    )


query_map = {
    "audio_to_text": get_audio_to_text_query,
    "text_to_audio": get_text_to_audio_query,
    "audio_to_audio": get_audio_to_audio_query,
}


def main(args):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Use local model path or HuggingFace model name
    if not args.model:
        logger.error("Model path is required!")
        logger.info("Usage:")
        logger.info("  Local model: --model /path/to/your/step-audio-2")
        logger.info("  HuggingFace: --model stepfun-ai/Step-Audio2-mini")
        sys.exit(1)

    model_name = args.model

    # Check if it's a local path
    if os.path.isdir(model_name):
        logger.info(f"Using local model: {model_name}")

        # Verify essential files exist
        config_path = os.path.join(model_name, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Model directory missing config.json: {config_path}")
            logger.info("Expected directory structure:")
            logger.info(f"  {model_name}/")
            logger.info("    ├── config.json        (required)")
            logger.info("    ├── model.safetensors or pytorch_model.bin")
            logger.info("    ├── tokenizer.json")
            logger.info("    └── token2wav/        (Token2Wav models)")
            sys.exit(1)

        # Check token2wav directory
        token2wav_path = os.path.join(model_name, "token2wav")
        if not os.path.isdir(token2wav_path):
            logger.warning(f"Missing token2wav directory: {token2wav_path}")
            logger.warning("Stage 1 (Token2Wav) may not work properly")
        else:
            logger.info("Token2Wav directory exists")
    else:
        logger.info(f"Using HuggingFace model: {model_name}")
        logger.info("Model will be cached in ~/.cache/huggingface/hub/")

    # Get query configuration
    query_func = query_map[args.query_type]

    # Build query based on type
    if args.query_type == "audio_to_text":
        query_result = query_func(
            audio_path=args.audio_path,
            question=args.question,
            sampling_rate=args.sampling_rate,
        )
    elif args.query_type == "text_to_audio":
        query_result = query_func(
            text=args.text,
            sampling_rate=args.sampling_rate,
        )
    elif args.query_type == "audio_to_audio":
        query_result = query_func(
            audio_path=args.audio_path,
            question=args.question,
            sampling_rate=args.sampling_rate,
        )
    else:
        raise ValueError(f"Unknown query type: {args.query_type}")

    # Initialize vLLM-Omni with Step-Audio2
    logger.info(f"Initializing Step-Audio2 with model: {model_name}")
    logger.info(f"Query type: {args.query_type}")

    # Resolve stage config path
    if args.stage_configs_path:
        stage_config_path = args.stage_configs_path
    else:
        # Use default config from the package
        stage_config_path = str(
            Path(__file__).parent.parent.parent.parent / "vllm_omni/model_executor/stage_configs/step_audio_2.yaml"
        )

    logger.info(f"Using stage config: {stage_config_path}")

    omni_llm = Omni(
        model=model_name,
        log_stats=args.enable_stats,
        log_file=("step_audio2_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
        stage_configs_path=stage_config_path,
        worker_backend=args.worker_backend,
        ray_address=args.ray_address,
    )

    # Configure sampling parameters for each stage
    # Stage 0 (Thinker): Generate text and audio tokens
    # Note: For audio generation tasks, we need to allow the model to generate audio tokens
    # Audio tokens are in range [151696, 158257], so we should not stop too early
    # Qwen EOS token is typically 151645 (<|im_end|>), but we want to allow audio generation
    # So we don't set stop_token_ids for audio tasks, but we do set it for text-only tasks
    stop_token_ids = None
    if args.query_type == "audio_to_text":
        # For ASR, stop at EOS token
        stop_token_ids = [151645]  # Qwen EOS token <|im_end|>

    # Adjust sampling parameters based on query type
    # For audio generation tasks, we need higher repetition_penalty to avoid text loops
    # and allow the model to transition to audio token generation
    repetition_penalty = 1.1 if args.query_type in ["text_to_audio", "audio_to_audio"] else 1.05

    thinker_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=SEED,
        detokenize=True,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
    )

    # Stage 1 (Token2Wav): Convert audio tokens to waveform
    # Note: This stage doesn't actually sample, it's deterministic generation
    token2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        seed=SEED,
        detokenize=False,
    )

    sampling_params_list = [
        thinker_sampling_params,
        token2wav_sampling_params,
    ]

    # Prepare prompts
    if args.num_prompts > 1:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        prompts = [query_result.inputs]

    logger.info(f"Running inference on {len(prompts)} prompt(s)...")

    # Run inference
    logger.info("Calling omni_llm.generate()...")
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)
    logger.info("Generation completed! Processing outputs...")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process outputs from each stage
    logger.info(f"Processing {len(omni_outputs)} stage outputs...")
    for stage_idx, stage_outputs in enumerate(omni_outputs):
        logger.info(f"Processing stage {stage_idx}, type: {stage_outputs.final_output_type}")
        if stage_outputs.final_output_type == "text":
            # Stage 0 (Thinker) text output
            logger.info("=" * 50)
            logger.info("Stage 0 (Thinker) - Text Output")

            for output in stage_outputs.request_output:
                request_id = int(output.request_id)
                text_output = output.outputs[0].text

                logger.info(f"[Request {request_id}] Output: {text_output[:100]}...")

                # Save to file
                out_txt = os.path.join(output_dir, f"{request_id:05d}_text.txt")
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(f"Prompt:\n{prompts[request_id]['prompt']}\n\n")
                    f.write(f"Output:\n{text_output}\n")

                logger.info(f"Text saved to {out_txt}")

        elif stage_outputs.final_output_type == "audio":
            # Stage 1 (Token2Wav) audio output
            logger.info("=" * 50)
            logger.info("Stage 1 (Token2Wav) - Audio Output")

            for output in stage_outputs.request_output:
                request_id = int(output.request_id)

                # Get audio tensor (24kHz)
                mm_out = output.multimodal_output or {}
                audio_tensor = None
                for key in ("audio", "wav", "waveform", "audio_pcm", "pcm", "model_outputs"):
                    if key in mm_out and mm_out[key] is not None:
                        audio_tensor = mm_out[key]
                        break

                if audio_tensor is not None:
                    # Save audio file
                    output_wav = os.path.join(output_dir, f"{request_id:05d}_output.wav")
                    audio_numpy = audio_tensor.detach().cpu().numpy()
                    sf.write(output_wav, audio_numpy, samplerate=24000)

                    duration = len(audio_numpy) / 24000
                    logger.info(f"[Request {request_id}] Audio duration: {duration:.2f}s, saved to {output_wav}")
                else:
                    logger.warning(f"[Request {request_id}] No audio output generated")

    logger.info("=" * 50)
    logger.info(f"All outputs saved to: {output_dir}")


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM-Omni for offline inference with Step-Audio2")

    # Model and config
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help=(
            "Model path (required). Can be:\n"
            "  - Local path: /path/to/step-audio-2\n"
            "  - HuggingFace ID: stepfun-ai/Step-Audio2-mini"
        ),
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage config YAML file (default: use built-in step_audio_2.yaml)",
    )

    # Query configuration
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="audio_to_text",
        choices=query_map.keys(),
        help="Query type: audio_to_text (ASR), text_to_audio (TTS), audio_to_audio (voice conversion)",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to input audio file (for audio_to_text and audio_to_audio)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text to synthesize (for text_to_audio mode)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question/instruction about the audio",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Audio sampling rate for input (default: 16000 Hz)",
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate in Thinker stage (default: 1024)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of times to repeat the prompt (for testing)",
    )

    # System configuration
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable detailed statistics logging",
    )
    parser.add_argument(
        "--init-sleep-seconds",
        type=int,
        default=20,
        help="Sleep seconds after starting each stage (default: 20)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Batch timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Initialization timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Shared memory threshold in bytes (default: 65536)",
    )
    parser.add_argument(
        "--worker-backend",
        type=str,
        default="multi_process",
        choices=["multi_process", "ray"],
        help="Worker backend (default: multi_process)",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (for ray backend)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output_step_audio2",
        help="Output directory for generated files (default: output_step_audio2)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
