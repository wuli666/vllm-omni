# LiquidAI LFM3-Audio

> Speech recognition, text to speech and voice chat with interleaved text and speech on one GPU

## Summary

- Vendor: Liquid AI
- Model: `LiquidAI/LFM3-Audio-2B`
- Task: ASR, spoken question answering, text to speech (19 languages / locales, female or male voice), voice chat answering in interleaved text and 24 kHz speech
- Mode: Online serving (`/v1/audio/speech`, `/v1/chat/completions`) and offline inference
- Maintainer: Community

## When to use this recipe

LFM3-Audio is a 2B model: an LFM3 hybrid backbone (convolution, full and
sliding-window attention), a causal audio encoder at 12.5 Hz, a small
bidirectional depth decoder that samples 12 codebooks per speech frame, and
the LiquidCodec decoder (24 kHz). Pick the deployment by output:

| deploy config (`vllm_omni/deploy/<name>.yaml`) | stages | text out | speech out | endpoint |
| --- | --- | --- | --- | --- |
| `lfm3_audio` (default) | language model + depth decoder → codec | ✅ | ✅ | `/v1/audio/speech`, `/v1/chat/completions` |
| `lfm3_audio_text` | language model | ✅ | ❌ | `/v1/chat/completions` |

The reply's mode follows the last prompt token, as in the model card: a
prompt ending on `<|audio_start|>` is answered in speech, one ending on
`<|mixed_start|>` in interleaved text and speech, anything else in text.

## References

- Model card: [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
- Online example: [`examples/online_serving/lfm3_audio/`](../../examples/online_serving/lfm3_audio/)

## Hardware Support

## GPU

### 1×RTX 4090 48GB

#### Environment

- OS: Linux
- Python: 3.12
- CUDA: 12.x
- vLLM version: 0.32.0 with [vllm#64959](https://github.com/vllm-project/vllm/pull/64959) (`Lfm3ForCausalLM`) and [vllm#64647](https://github.com/vllm-project/vllm/pull/64647) (width-7 causal convolution); both are needed until a vLLM release includes them
- vLLM-Omni version or commit: branch `lfm3-audio`

#### Command

```bash
# Speech and text out (default deployment)
vllm serve LiquidAI/LFM3-Audio-2B --omni --trust-remote-code --port 8091

# Text out only (ASR, spoken QA)
vllm serve LiquidAI/LFM3-Audio-2B --omni --trust-remote-code --port 8091 \
    --deploy-config vllm_omni/deploy/lfm3_audio_text.yaml
```

#### Verification

```bash
curl -s http://localhost:8091/v1/audio/speech -H "Content-Type: application/json" \
    -d '{"input": "Cows on the ranch eat hay in the morning.", "voice": "female"}' -o speech.wav
```

`/v1/audio/speech` accepts `voice` (`female`, the default, or `male`) and
`language` (`American English`, the default, `British English`, `Arabic`,
`Chinese`, `Danish`, `Dutch`, `Finnish`, `French`, `German`, `Greek`,
`Hebrew`, `Italian`, `Japanese`, `Korean`, `Polish`, `Portuguese`, `Spanish`,
`Swedish`, `Turkish`). There is no voice cloning. Voice chat goes through
`/v1/chat/completions`; see the online example.

#### Offline inference

The prompt follows the model card: a system prompt names the task and the
assistant turn is prefilled with the marker of the reply's mode.

| Task | System prompt | User turn | Assistant prefill |
| --- | --- | --- | --- |
| Text to speech | `Perform TTS in <language> using a <female\|male> voice.` | text | `<\|audio_start\|>` |
| Voice chat | `Respond with interleaved text and audio.` | `<\|audio\|>` or text | `<\|mixed_start\|>` |
| ASR | `Perform ASR in <language>.` (or `Perform ASR.`) | `<\|audio\|>` | none |

```python
import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset

from vllm_omni.entrypoints.omni import Omni


def turn(system: str, user: str, prefill: str) -> str:
    return (
        f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefill}"
    )


if __name__ == "__main__":
    omni = Omni(model="LiquidAI/LFM3-Audio-2B", trust_remote_code=True)
    prompts = [
        {"prompt": turn("Perform TTS in American English using a female voice.", "Cows eat hay in the morning.", "<|audio_start|>")},
        {
            # One <|audio|> per clip; the processor expands it to the encoder's 12.5 Hz frames.
            "prompt": turn("Respond with interleaved text and audio.", "<|audio|>", "<|mixed_start|>"),
            "multi_modal_data": {"audio": AudioAsset("mary_had_lamb").audio_and_sample_rate},
        },
    ]
    # Text is decoded greedily; speech frames use the checkpoint's audio sampling settings.
    params = [SamplingParams(temperature=0.0, max_tokens=2048), SamplingParams(temperature=0.0, max_tokens=8192)]
    for output in omni.generate(prompts, params):
        if output.final_output_type == "text":
            print(output.request_id, output.outputs[0].text)
        else:
            audio = output.outputs[0].multimodal_output["audio"]
            audio = torch.cat([a.reshape(-1) for a in audio]) if isinstance(audio, list) else audio.reshape(-1)
            sf.write(f"reply_{output.request_id}.wav", audio.float().cpu().numpy(), 24000)
    omni.close()
```

For ASR alone, pass `deploy_config="vllm_omni/deploy/lfm3_audio_text.yaml"`
to `Omni` and a single `SamplingParams`: the language model runs without the
codec.

#### Notes

- Memory: the default deployment reserves 50% of the GPU for the language
  model stage and 10% for the codec (about 29 GB on a 48 GB card); lower
  `gpu_memory_utilization` in the deploy config for smaller cards.
- Prefix caching is off: the hybrid backbone has several KV cache groups.
- Speech streams to the codec in windows of 2, 4 and 8 frames, then 12
  frames (about 1 s); `codec_chunk_ramp` and `codec_chunk_frames` in the
  deploy config change them.
- Measured on one RTX 4090 48 GB: streamed speech starts after 0.11 s;
  one request speaks at 3.9x real time, 16 concurrent requests at 41x real
  time in total (2.6x each). The speech frames of all requests in a step are
  sampled in one depth decoder pass.
- Speech output requests that are preempted fail instead of being
  recomputed: give the stage enough KV cache for `max_num_seqs` replies.
- Speech frames are sampled with the checkpoint's audio settings
  (temperature 1.0, top-k 128, repetition penalty 1.1 over 4096 frames).
