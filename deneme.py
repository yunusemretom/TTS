import shutil
import subprocess
import time

import soundfile as sf
import torch

from TTS.api import TTS

# --- ayarlar ---------------------------------------------------------------
HALF = True  # fp16: GPT'yi yarı hassasiyette çalıştırır (yalnızca GPU'da)
BATCH_SIZE = 4  # birden çok cümleyi tek geçişte çöz; 1 = kapalı (uzun metinde etkili)
WARMUP = True  # ilk çağrı CUDA kernel/bellek kurulumunu da içerir, ölçüme katma
PLAY = True  # üretim biter bitmez sesi çal

LANGUAGE = "tr"
SPEAKER_WAV = "ses2.mp3"
OUTPUT = "output.wav"  # her turda üzerine yazılır
# ---------------------------------------------------------------------------


def play(path):
    """Sesi çalar; bir oynatıcı başarısız olursa sıradakini dener."""
    players = {
        "paplay": [path],
        "pw-play": [path],
        "aplay": ["-q", path],
        "ffplay": ["-nodisp", "-autoexit", "-loglevel", "quiet", path],
    }
    for name, args in players.items():
        exe = shutil.which(name)
        if exe is None:
            continue
        if subprocess.run([exe, *args], check=False).returncode == 0:
            return
        print(f"  {name} çalamadı, sıradaki deneniyor")
    print(f"  ses çalınamadı, dosya kaydedildi: {path}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

# Kullanılabilir modelleri görmek için:
# print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
model = tts.synthesizer.tts_model
sample_rate = model.config.audio.output_sample_rate

if HALF:
    if device == "cuda":
        model.use_half_precision()
        print("fp16: açık")
    else:
        print("fp16: atlandı (yarı hassasiyet yalnızca GPU'da anlamlı)")
else:
    print("fp16: kapalı")
print(f"batch: {BATCH_SIZE}" if BATCH_SIZE > 1 else "batch: kapalı")

# Konuşmacı latent'leri bir kez çıkarılıp her turda yeniden kullanılır. tts.tts_to_file()
# bunları her çağrıda baştan hesaplar; bir döngüde bu tur başına boşa giden zamandır.
print(f"\nkonuşmacı analiz ediliyor: {SPEAKER_WAV}")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_WAV])

if WARMUP:
    model.inference("Isınma turu.", LANGUAGE, gpt_cond_latent, speaker_embedding)

print("\nMetin girin. Boş satır, Ctrl+D veya Ctrl+C çıkar.")
while True:
    try:
        text = input("\nmetin> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not text:
        break

    start = time.perf_counter()
    out = model.inference(
        text,
        LANGUAGE,
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=True,
        batch_size=BATCH_SIZE,
    )
    elapsed = time.perf_counter() - start

    sf.write(OUTPUT, out["wav"], sample_rate)
    audio_seconds = len(out["wav"]) / sample_rate
    print(f"süre: {elapsed:.2f} s | ses: {audio_seconds:.2f} s | RTF: {elapsed / audio_seconds:.3f}")

    if PLAY:
        play(OUTPUT)

print("çıkıldı.")
