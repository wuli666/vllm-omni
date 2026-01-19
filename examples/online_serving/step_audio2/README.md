# Step-Audio2 Online Serving

This directory contains examples for running Step-Audio2 with vLLM-Omni's online serving API.

## Installation

Please refer to [README.md](../../../README.md)

## Launch the Server

```bash
vllm serve stepfun-ai/Step-Audio2-mini --omni --port 8092
```

With custom stage configs:
```bash
vllm serve stepfun-ai/Step-Audio2-mini --omni --port 8092 \
    --stage-configs-path /path/to/step_audio_2.yaml
```

With local model:
```bash
vllm serve /path/to/step-audio-2 --omni --port 8092
```

## Send Requests

### Python Client

```bash
cd examples/online_serving/step_audio2

# Audio to Text (ASR)
python openai_chat_completion_client.py --query-type audio_to_text

# Text to Audio (TTS)
python openai_chat_completion_client.py --query-type text_to_audio --text "Hello, this is a test."

# Audio to Audio (Voice Conversion)
python openai_chat_completion_client.py --query-type audio_to_audio --audio-path /path/to/input.wav
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--query-type`, `-q` | Query type: `audio_to_text`, `text_to_audio`, `audio_to_audio` |
| `--audio-path`, `-a` | Path to input audio file (local or URL) |
| `--text`, `-t` | Text to synthesize (for TTS mode) |
| `--prompt`, `-p` | Custom prompt/question |
| `--output-dir`, `-o` | Output directory for audio files (default: `output_online`) |
| `--api-base` | API base URL (default: `http://localhost:8092/v1`) |

**Note**: Speaker voice is controlled by `STEP_AUDIO2_DEFAULT_PROMPT_WAV` env var on the server side.

### Curl

```bash
# Audio to Text
bash run_curl.sh audio_to_text

# Text to Audio
bash run_curl.sh text_to_audio

# Audio to Audio
bash run_curl.sh audio_to_audio
```

## Query Types

### 1. Audio to Text (ASR)

Transcribe audio to text.

```bash
python openai_chat_completion_client.py \
    --query-type audio_to_text \
    --audio-path /path/to/speech.wav \
    --prompt "Transcribe this audio."
```

### 2. Text to Audio (TTS)

Convert text to speech.

```bash
python openai_chat_completion_client.py \
    --query-type text_to_audio \
    --text "Hello, welcome to Step-Audio2 text to speech synthesis."
```

### 3. Audio to Audio (Voice Conversion)

Process input audio and generate output audio.

```bash
python openai_chat_completion_client.py \
    --query-type audio_to_audio \
    --audio-path /path/to/source.wav
```

## Output

- **Text output**: Printed to console
- **Audio output**: Saved to `output_online/audio_0.wav` (24kHz WAV)

## API Format

Step-Audio2 uses the OpenAI-compatible chat completions API:

```json
{
  "model": "stepfun-ai/Step-Audio2-mini",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "Transcribe the audio."}]
    },
    {
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "..."}},
        {"type": "text", "text": "Please transcribe."}
      ]
    }
  ],
  "sampling_params_list": [
    {"temperature": 0.7, "max_tokens": 1024},
    {"temperature": 0.0, "max_tokens": 1}
  ]
}
```

## Troubleshooting

### Server not responding
- Check if the server is running: `curl http://localhost:8092/health`
- Verify the port number matches

### Audio not generated
- Ensure the prompt ends with `<tts_start>` for TTS/audio generation modes
- Check server logs for errors

### Out of memory
- Reduce `gpu_memory_utilization` in stage configs
- Use a smaller batch size
