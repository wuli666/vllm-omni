# Step-Audio2 Offline Inference Examples

This directory contains examples for running offline inference with Step-Audio2 using vLLM-Omni.

## Model Overview

Step-Audio2 is a two-stage audio model:

- **Stage 0 (Thinker)**: Audio understanding → Text + Audio tokens
  - Input: Audio (16kHz)
  - Output: Text transcription + Audio tokens for synthesis

- **Stage 1 (Token2Wav)**: Audio synthesis
  - Input: Audio tokens + Speaker prompt wav
  - Output: Synthesized audio waveform (24kHz)

## Installation

Make sure you have installed vLLM-Omni and all required dependencies:

```bash
# Install vLLM-Omni
pip install -e /path/to/vllm-omni-feature-step-audio2-integration

# Install Step-Audio2 (REQUIRED for Token2Wav stage)
pip install step-audio2
```

## Model Setup

### Option 1: Auto-download from HuggingFace (Recommended)

The script will **automatically download** the model on first run:

```bash
# Just run without specifying --model, it will auto-download stepfun-ai/Step-Audio2-mini
python end2end.py --query-type audio_to_text

# Or explicitly specify the HuggingFace model
python end2end.py --query-type audio_to_text --model stepfun-ai/Step-Audio2-mini
```

Models will be cached in `~/.cache/huggingface/hub/` for future use.

**Available models**:
- `stepfun-ai/Step-Audio2-mini` (smaller, faster)
- `stepfun-ai/Step-Audio2-7B` (larger, better quality)

### Option 2: Manual Download (for offline use)

Download and use locally:

```bash
# Download from HuggingFace
huggingface-cli download stepfun-ai/Step-Audio2-mini --local-dir ./models/Step-Audio2-mini

# Then use the local path
python end2end.py --query-type audio_to_text --model ./models/Step-Audio2-mini
```

Ensure the model directory contains:
```
Step-Audio2-mini/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
└── token2wav/                           # Token2Wav models (REQUIRED)
    ├── speech_tokenizer_v2_25hz.onnx   # Audio tokenizer
    ├── campplus.onnx                    # Speaker encoder
    ├── flow.yaml                        # Flow model config
    ├── flow.pt                          # Flow model weights
    └── hift.pt                          # HiFT vocoder weights
```

## Usage Examples

### 1. Audio to Text (ASR - Speech Recognition)

Transcribe audio to text:

```bash
# Quick start - Using default model and test audio
python end2end.py --query-type audio_to_text

# Using your own audio file (model will auto-download)
python end2end.py --query-type audio_to_text \
    --audio-path /path/to/input.wav

# With specific model
python end2end.py --query-type audio_to_text \
    --audio-path /path/to/input.wav \
    --model stepfun-ai/Step-Audio2-7B

# With custom question
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --question "What is the speaker saying?"
```

**Output**: Text transcription saved to `output_step_audio2/00000_text.txt`

### 2. Text to Audio (TTS - Speech Synthesis)

Convert text to speech with a speaker prompt:

```bash
# Basic TTS (model auto-downloads)
python end2end.py --query-type text_to_audio \
    --text "Hello, this is a test of Step Audio 2 synthesis." \
    --prompt-wav /path/to/speaker_sample.wav

# With specific model
python end2end.py --query-type text_to_audio \
    --text "Hello, this is a test." \
    --prompt-wav speaker.wav \
    --model stepfun-ai/Step-Audio2-7B
```

**Important**: `--prompt-wav` is **REQUIRED** for TTS. This audio file determines the voice characteristics (speaker identity, tone, style) of the generated speech.

**Output**:
- Text: `output_step_audio2/00000_text.txt`
- Audio: `output_step_audio2/00000_output.wav` (24kHz)

### 3. Audio to Audio (Voice Conversion / Cloning)

Transform input audio to match a target speaker's voice:

```bash
# Basic voice conversion (model auto-downloads)
python end2end.py --query-type audio_to_audio \
    --audio-path /path/to/source_audio.wav \
    --prompt-wav /path/to/target_speaker.wav

# With specific model
python end2end.py --query-type audio_to_audio \
    --audio-path source.wav \
    --prompt-wav target_speaker.wav \
    --model stepfun-ai/Step-Audio2-7B
```

This mode:
1. Understands the content in `--audio-path` (source)
2. Generates audio output with the voice style from `--prompt-wav` (target speaker)

**Use cases**: Voice cloning, accent transfer, speaker adaptation

### Advanced Options

```bash
# Use custom stage configuration
python end2end.py --query-type audio_to_text \
    --stage-configs-path /path/to/custom_config.yaml

# Multiple prompts (for batch testing)
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --num-prompts 5

# Custom output directory
python end2end.py --query-type text_to_audio \
    --text "Test synthesis" \
    --prompt-wav speaker.wav \
    --output-dir ./my_outputs

# Enable detailed logging
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --enable-stats

# Adjust generation parameters
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --max-tokens 2048

# Use Ray backend for distributed processing
python end2end.py --query-type text_to_audio \
    --text "Hello world" \
    --prompt-wav speaker.wav \
    --worker-backend ray \
    --ray-address "auto"
```

## Configuration

### Stage Configuration

The default configuration (`step_audio_2.yaml`) uses:

- **Stage 0 (Thinker)**: GPU 0, 80% memory
- **Stage 1 (Token2Wav)**: GPU 1, 30% memory

For **single GPU** setup, edit the config to use `devices: "0"` for both stages.

### Sampling Parameters

- **Thinker (Stage 0)**:
  - Temperature: 0.7 (balanced creativity)
  - Top-p: 0.9
  - Max tokens: 1024 (configurable)

- **Token2Wav (Stage 1)**:
  - Temperature: 0.0 (deterministic)
  - Operates in generation mode (not sampling)

## Common Issues

### 1. ImportError: No module named 's3tokenizer'

**Solution**: Install Step-Audio2 package:
```bash
pip install step-audio2
```

### 2. ValueError: "prompt_wav is required for Token2Wav"

**Solution**: For `text_to_audio` and `audio_to_audio` modes, you **must** provide `--prompt-wav`:
```bash
python end2end.py --query-type text_to_audio \
    --text "Hello" \
    --prompt-wav speaker_sample.wav
```

### 3. FileNotFoundError: token2wav models not found

**Solution**: Ensure your model directory has the complete `token2wav/` subdirectory with all ONNX and PyTorch models.

### 4. CUDA Out of Memory

**Solutions**:
- Use single GPU mode (set both stages to `devices: "0"`)
- Reduce `gpu_memory_utilization` in config
- Reduce `max_num_batched_tokens`
- Process fewer prompts at once

### 5. Model not found in registry

**Solution**: Ensure you're using vLLM-Omni's entry point with `--omni` flag or install vllm-omni properly:
```bash
pip install -e /path/to/vllm-omni-feature-step-audio2-integration
```

## Output Files

The script generates files in the output directory (default: `output_step_audio2/`):

```
output_step_audio2/
├── 00000_text.txt        # Text output from Thinker stage
├── 00000_output.wav      # Audio output from Token2Wav stage (24kHz)
├── 00001_text.txt        # (if multiple prompts)
└── 00001_output.wav
```

## Performance Tips

1. **First run is slow**: Stage initialization takes 20-60 seconds
2. **Single GPU**: Set both stages to `devices: "0"` in config
3. **Multiple prompts**: Use `--num-prompts N` for batch testing
4. **Ray backend**: For multi-node or advanced scheduling
5. **Logging**: Use `--enable-stats` to debug performance issues

## Speaker Prompt Guidelines

For best results with `--prompt-wav`:

- **Duration**: 3-10 seconds recommended
- **Quality**: Clean audio, minimal background noise
- **Format**: WAV, MP3, FLAC (will be resampled to 16kHz)
- **Content**: Clear speech, representative of target voice

## Example Workflow

Complete example from audio to final output:

```bash
# 1. ASR: Transcribe audio
python end2end.py --query-type audio_to_text \
    --audio-path interview.wav \
    --model ./models/Step-Audio2-7B \
    --output-dir ./outputs

# 2. Check the transcription
cat ./outputs/00000_text.txt

# 3. TTS: Synthesize new speech with custom voice
python end2end.py --query-type text_to_audio \
    --text "The quick brown fox jumps over the lazy dog" \
    --prompt-wav ./speaker_samples/female_voice.wav \
    --model ./models/Step-Audio2-7B \
    --output-dir ./outputs

# 4. Listen to the result
# Audio saved to: ./outputs/00000_output.wav
```

## References

- [Step-Audio2 Paper](https://arxiv.org/abs/...)
- [vLLM-Omni Documentation](https://vllm-omni.readthedocs.io)
- [Model on HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio2-7B)
