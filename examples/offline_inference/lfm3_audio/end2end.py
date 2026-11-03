# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio offline inference: text to speech, voice chat and speech recognition.

python end2end.py --task tts --text "Cows eat hay in the morning."
python end2end.py --task chat --audio question.wav
python end2end.py --task asr --audio speech.wav
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

import vllm_omni
from vllm_omni.entrypoints.omni import Omni

# The assistant turn is prefilled with the marker of the reply's mode.
SYSTEM_PROMPTS = {
    "tts": ("Perform TTS in {language} using a {voice} voice.", "<|audio_start|>"),
    "chat": ("Respond with interleaved text and audio.", "<|mixed_start|>"),
    # Without --language the model detects it.
    "asr": ("Perform ASR{language}.", ""),
}


def build_prompt(args) -> dict:
    system, prefill = SYSTEM_PROMPTS[args.task]
    language = args.language or "American English"
    if args.task == "asr":
        language = f" in {args.language}" if args.language else ""
    system = system.format(language=language, voice=args.voice)
    if args.task == "tts":
        user, mm = args.text, None
    else:
        if args.audio:
            audio, sr = load_audio(args.audio, sr=16000)
        else:
            audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
        # The processor expands each <|audio|> into one placeholder per encoder frame (12.5 Hz).
        user, mm = "<|audio|>", {"audio": (np.asarray(audio, dtype=np.float32), sr)}
    prompt = {
        "prompt": f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefill}"
    }
    if mm:
        prompt["multi_modal_data"] = mm
    return prompt


def main(args) -> None:
    deploy = os.path.join(os.path.dirname(vllm_omni.__file__), "deploy", "lfm3_audio_text.yaml")
    # Speech recognition answers in text only: run the language model without the codec stage.
    kwargs = {"deploy_config": deploy} if args.task == "asr" else {}
    omni = Omni(model=args.model, trust_remote_code=True, **kwargs)
    # Text is decoded greedily; speech frames use the checkpoint's audio sampling settings.
    params = [SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)]
    if args.task != "asr":
        params.append(SamplingParams(temperature=0.0, max_tokens=8192, detokenize=False))

    os.makedirs(args.output_dir, exist_ok=True)
    chunks = []
    for output in omni.generate([build_prompt(args)], params):
        if output.final_output_type == "text":
            print(f"text: {output.outputs[0].text!r}")
        elif output.final_output_type == "audio":
            mm = output.outputs[0].multimodal_output
            audio = mm.get("audio") if mm.get("audio") is not None else mm.get("model_outputs")
            chunks += audio if isinstance(audio, list) else [audio]
    if chunks:
        wav = torch.cat([torch.as_tensor(c).reshape(-1).float().cpu() for c in chunks]).numpy()
        path = os.path.join(args.output_dir, f"{args.task}.wav")
        sf.write(path, wav, 24000)
        print(f"audio: {path} ({len(wav) / 24000:.2f} s)")
    omni.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="LiquidAI/LFM3-Audio-2B")
    parser.add_argument("--task", choices=sorted(SYSTEM_PROMPTS), default="tts")
    parser.add_argument("--text", default="Cows on the ranch eat hay in the morning and rest in the shade after lunch.")
    parser.add_argument("--audio", help="Input audio for chat / asr (default: a public speech clip)")
    parser.add_argument("--voice", choices=("female", "male"), default="female")
    parser.add_argument(
        "--language", help="TTS: e.g. American English (default), British English, French; ASR: e.g. English"
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="output_lfm3_audio")
    main(parser.parse_args())
