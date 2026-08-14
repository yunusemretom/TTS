"""End-to-end XTTS v2 benchmark on paragraph-length text.

Compares the stock settings against half precision and batched sentence decoding,
which is the case that matters for reading a document out loud.

Usage:
    python scripts/bench_xtts_paragraph.py --runs 2
"""

import argparse
import os
import time

import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

PARAGRAPH = (
    "Speech synthesis has changed a great deal over the last few years. "
    "Models that once needed a studio recording session for every new voice can now copy "
    "a speaker from a few seconds of audio. "
    "That shift moved the hard problem from data collection to inference cost. "
    "A model is only useful in a product if it can keep up with the person listening to it. "
    "On a laptop graphics card, the autoregressive decoder is the part that decides whether "
    "the system feels responsive or sluggish. "
    "Measuring where the time actually goes is the first step to making it faster."
)


def load_model(model_path, device, half):
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--speaker-wav", default="tests/data/ljspeech/wavs/LJ001-0001.wav")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = ModelManager().download_model(MODEL_NAME)[0]
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"gpu:    {torch.cuda.get_device_name(0)}")

    model = load_model(model_path, device, half=False)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[args.speaker_wav])
    sr = model.config.audio.output_sample_rate

    configs = [
        ("fp32, sequential", False, 1),
        ("fp16, sequential", True, 1),
        ("fp16, batch=4", True, 4),
        ("fp16, batch=8", True, 8),
    ]

    results = {}
    halved = False
    for name, half, batch_size in configs:
        if half and not halved:
            model.use_half_precision()
            halved = True

        times = []
        for i in range(args.runs):
            torch.manual_seed(1234 + i)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.inference(
                PARAGRAPH,
                "en",
                gpt_cond_latent,
                speaker_embedding,
                enable_text_splitting=True,
                batch_size=batch_size,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append((elapsed, len(out["wav"]) / sr))

        steady = times[1:] or times
        rtf = sum(t / a for t, a in steady) / len(steady)
        wall = sum(t for t, _ in steady) / len(steady)
        audio = sum(a for _, a in steady) / len(steady)
        results[name] = rtf
        print(f"{name:20s} {wall:5.2f}s for {audio:5.2f}s audio | RTF {rtf:.3f}")
        if torch.cuda.is_available():
            print(f"{'':20s} peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    print("\n=== summary ===")
    base = results["fp32, sequential"]
    for name, rtf in results.items():
        print(f"{name:20s} RTF {rtf:.3f}  ({base / rtf:.2f}x vs stock)")


if __name__ == "__main__":
    main()
