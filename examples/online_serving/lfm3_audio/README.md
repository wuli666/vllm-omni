# LFM3-Audio: online serving

```bash
vllm serve LiquidAI/LFM3-Audio-2B --omni --trust-remote-code --port 8091
```

The default deployment answers in text and speech. For ASR and spoken
question answering alone, add
`--deploy-config vllm_omni/deploy/lfm3_audio_text.yaml`: it runs the language
model without the codec. See the [recipe](../../../recipes/LiquidAI/LFM3-Audio.md)
for requirements.

## Text to speech: `/v1/audio/speech`

```bash
curl -s http://localhost:8091/v1/audio/speech -H "Content-Type: application/json" \
    -d '{"input": "Rain is expected tomorrow, so move the hay into the barn.", "voice": "male", "language": "British English"}' \
    -o speech.wav

# Streamed 16-bit PCM at 24 kHz
curl -sN http://localhost:8091/v1/audio/speech -H "Content-Type: application/json" \
    -d '{"input": "Rain is expected tomorrow.", "stream": true, "stream_format": "audio", "response_format": "pcm"}' \
    -o speech.pcm
```

`voice` is `female` (default) or `male`; `language` is one of the 19 speech
languages / locales of the model card (default `American English`).

## Voice chat and ASR: `/v1/chat/completions`

The model card prefills the assistant turn with the marker of the reply's
mode. Send it as the last assistant message with `continue_final_message`:
`<|mixed_start|>` asks for interleaved text and speech, and a prompt without a
marker is answered in text.

```python
import base64

import requests

URL = "http://localhost:8091/v1/chat/completions"
audio = {"type": "input_audio", "input_audio": {"data": base64.b64encode(open("question.wav", "rb").read()).decode(), "format": "wav"}}

# Interleaved text and speech
reply = requests.post(URL, json={
    "messages": [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": [audio]},
        {"role": "assistant", "content": "<|mixed_start|>"},
    ],
    "modalities": ["text", "audio"],
    "add_generation_prompt": False,
    "continue_final_message": True,
}).json()
for choice in reply["choices"]:
    message = choice["message"]
    if message.get("content"):
        print(message["content"])
    if message.get("audio"):
        open("reply.wav", "wb").write(base64.b64decode(message["audio"]["data"]))

# Speech recognition
reply = requests.post(URL, json={
    "messages": [{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": [audio]}],
    "modalities": ["text"],
    "temperature": 0.0,
}).json()
print(reply["choices"][0]["message"]["content"])
```
