# LFM3-Audio: offline inference

[LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) takes text or
16 kHz speech and answers in text, 24 kHz speech, or interleaved text and
speech. See the [recipe](../../../recipes/LiquidAI/LFM3-Audio.md) for
requirements and deployments.

```bash
# Text to speech: female or male voice, 19 languages / locales
python end2end.py --task tts --text "Cows on the ranch eat hay in the morning." --voice male --language "British English"

# Voice chat: a spoken question in, interleaved text and speech out
python end2end.py --task chat --audio question.wav

# Speech recognition (text only, runs without the codec stage)
python end2end.py --task asr --audio speech.wav --language English
```

Speech is written to `output_lfm3_audio/<task>.wav`. Without `--audio`, chat
and ASR use a public speech clip.

The prompt follows the model card: a system prompt names the task and the
assistant turn is prefilled with `<|audio_start|>` (speech) or
`<|mixed_start|>` (interleaved text and speech). Text is decoded greedily;
speech frames are sampled with the checkpoint's audio settings.
