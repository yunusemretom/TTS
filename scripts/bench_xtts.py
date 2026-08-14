"""Benchmark XTTS v2 inference speed under different runtime settings.

Reports a per-phase breakdown (GPT autoregressive generation, the latent
forward pass, HiFiGAN decoding) so it is clear which part dominates.

Usage:
    python scripts/bench_xtts.py --configs fp32 fp16 --runs 3
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

TEXT = (
    "The quick brown fox jumps over the lazy dog while the engineer measures "
    "how many seconds of audio this model can synthesize in one second of compute."
)


def download_model():
    return ModelManager().download_model(MODEL_NAME)[0]


def load_model(model_path, device, use_deepspeed=False):
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=use_deepspeed, eval=True)
    model.to(device)
    return model


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, *args, **kwargs):
    sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    sync()
    return out, time.perf_counter() - t0


def inference_profiled(model, text, language, gpt_cond_latent, speaker_embedding, half=False):
    """Mirror of Xtts.inference, instrumented per phase."""
    dtype = torch.float16 if half else torch.float32
    gpt_cond_latent = gpt_cond_latent.to(model.device).to(dtype)
    speaker_embedding = speaker_embedding.to(model.device)

    sent = text.strip().lower()
    text_tokens = torch.IntTensor(model.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(model.device)

    with torch.no_grad():
        gpt_codes, t_gen = timeit(
            model.gpt.generate,
            cond_latents=gpt_cond_latent,
            text_inputs=text_tokens,
            input_tokens=None,
            do_sample=True,
            top_p=0.85,
            top_k=50,
            temperature=0.75,
            num_return_sequences=model.gpt_batch_size,
            num_beams=1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            output_attentions=False,
        )

        expected_output_len = torch.tensor(
            [gpt_codes.shape[-1] * model.gpt.code_stride_len], device=text_tokens.device
        )
        text_len = torch.tensor([text_tokens.shape[-1]], device=model.device)

        gpt_latents, t_latent = timeit(
            model.gpt,
            text_tokens,
            text_len,
            gpt_codes,
            expected_output_len,
            cond_latents=gpt_cond_latent,
            return_attentions=False,
            return_latent=True,
        )

        wav, t_vocoder = timeit(model.hifigan_decoder, gpt_latents.float(), g=speaker_embedding)

    return {
        "wav": wav.cpu().squeeze().numpy(),
        "tokens": gpt_codes.shape[-1],
        "gen": t_gen,
        "latent": t_latent,
        "vocoder": t_vocoder,
    }


def run_config(name, model_path, device, speaker_wav, runs, **load_kwargs):
    print(f"\n=== config: {name} ===")
    torch.manual_seed(0)
    half = load_kwargs.pop("half", False)
    model, load_s = timeit(load_model, model_path, device, **load_kwargs)
    print(f"load: {load_s:.2f}s")

    # conditioning runs in fp32: the reference mel is computed in fp32 either way
    (gpt_cond_latent, speaker_embedding), cond_s = timeit(model.get_conditioning_latents, audio_path=[speaker_wav])
    print(f"conditioning: {cond_s:.2f}s")

    if half:
        model.use_half_precision()

    sr = model.config.audio.output_sample_rate
    rows = []
    for i in range(runs):
        torch.manual_seed(1234 + i)
        out, total = timeit(
            inference_profiled, model, TEXT, "en", gpt_cond_latent, speaker_embedding, half=half
        )
        audio_s = len(out["wav"]) / sr
        rows.append((total, audio_s, out))
        print(
            f"run {i + 1}: total {total:.2f}s | gen {out['gen']:.2f}s ({out['tokens']} tok) "
            f"| latent {out['latent']:.2f}s | vocoder {out['vocoder']:.2f}s "
            f"| audio {audio_s:.2f}s | RTF {total / audio_s:.3f}"
        )

    steady = rows[1:] or rows  # drop warmup
    mean_rtf = sum(t / a for t, a, _ in steady) / len(steady)
    share = lambda k: sum(o[k] for _, _, o in steady) / sum(t for t, _, _ in steady) * 100
    print(f"mean RTF: {mean_rtf:.3f}  |  time split: gen {share('gen'):.0f}% "
          f"latent {share('latent'):.0f}% vocoder {share('vocoder'):.0f}%")

    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mean_rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--speaker-wav", default="tests/data/ljspeech/wavs/LJ001-0001.wav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = download_model()
    print(f"model:  {model_path}")
    print(f"device: {args.device}")
    if torch.cuda.is_available():
        print(f"gpu:    {torch.cuda.get_device_name(0)}")

    load_kwargs = {"fp32": {}, "fp16": {"half": True}}

    results = {}
    for cfg in args.configs:
        results[cfg] = run_config(cfg, model_path, args.device, args.speaker_wav, args.runs, **load_kwargs[cfg])

    print("\n=== summary (mean RTF, lower is better) ===")
    base = results.get("fp32")
    for cfg, rtf in results.items():
        speedup = f"  ({base / rtf:.2f}x vs fp32)" if base else ""
        print(f"{cfg:10s} {rtf:.3f}{speedup}")


if __name__ == "__main__":
    main()
