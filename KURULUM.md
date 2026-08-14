# Kurulum ve Hızlı Başlangıç

Bu depo (Coqui TTS 0.22.0) artık bakımda değil ve `requirements.txt` bugünkü paket
sürümleriyle çözüldüğünde **import edilebilen ama çalışmayan** bir ortam üretiyor. Aşağıdaki
adımlar, denenmiş ve testleri geçen sürüm kümesini kurar.

Referans donanım: NVIDIA RTX 4050 Laptop (6 GB), sürücü 550.163 (CUDA 12.4), Ubuntu, Python 3.10.

---

## 1. Kurulum

`setup.py` Python **3.9–3.11** arası ister; 3.12 ve üzeri reddedilir.

```bash
cd /path/to/TTS
uv venv --python 3.10
uv pip install -e '.[all,dev,notebooks]'
```

Bu ilk komut tek başına yeterli değil. Ardından üç düzeltme gerekiyor:

### 1a. Sürücünüze uygun torch

Varsayılan çözümleme `cu130` tekerleklerini kurar; bunlar 580+ sürücü ister. CUDA 12.4
sürücüsüyle `torch.cuda.is_available()` sessizce `False` döner ve her şey CPU'da çalışır.

```bash
uv pip install "torch==2.5.1+cu124" "torchaudio==2.5.1+cu124" \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
```

> `--index-url` (extra olmayan) kullanmayın: `nvidia-*` bağımlılıkları çözülemez ve kurulum
> "unsatisfiable" hatasıyla düşer.

Farklı bir sürücünüz varsa `nvidia-smi` çıktısındaki **CUDA Version** alanına bakıp uygun
`cuXXX` derlemesini seçin.

### 1b. Çalışan paket sürümleri

```bash
uv pip install "librosa==0.10.2.post1" "transformers==4.40.2" \
               "numpy==1.26.4" "scipy==1.11.4"
```

Neden bu sürümler:

| Paket | Sorun |
|---|---|
| `librosa` 0.10.0 | `pkg_resources` import ediyor; setuptools 81+ bunu kaldırdı, `import TTS` patlıyor |
| `transformers` 5.x | `SampleOutput` ve `LogitsWarper` kaldırıldı; `stream_generator.py` ve `tortoise/arch_utils.py` bunları kullanıyor |
| `numpy` 2.x | Derlenmiş `monotonic_align` cython uzantısı numpy 1.x ABI'sine göre |

### 1c. espeak ve nltk verisi

Fonemleştirme `espeak` adında bir çalıştırılabilir arıyor. Sistemde çoğunlukla yalnızca
`espeak-ng` bulunur; kodun kendisi bunun symlink olmasını zaten bekliyor, dolayısıyla sudo'ya
gerek yok:

```bash
ln -sf /usr/bin/espeak-ng .venv/bin/espeak
```

`espeak-ng` yoksa: `sudo apt install espeak-ng`.

Korece dışındaki diller için gerekmez, ama testler için nltk verisi:

```bash
.venv/bin/python -c "import nltk; [nltk.download(p, quiet=True) for p in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','punkt','cmudict']]"
```

---

## 2. Komutları çalıştırma

Bu makinede kabuk profili ROS Humble'ı source ediyor ve `PYTHONPATH` venv'in içine sızıyor.
Ayrıca depo testleri çıplak `python` çağırıyor; sistemde yalnızca `python3` var. İkisini de
çözen kalıp:

```bash
env PYTHONPATH= PATH="$PWD/.venv/bin:$PATH" python ...
```

ROS kullanmıyorsanız `PYTHONPATH=` kısmı zararsızdır, bırakabilirsiniz.

### Kurulumu doğrulama

```bash
env PYTHONPATH= .venv/bin/python -c "
import torch, TTS
from TTS.api import TTS as T
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
print('gpu  ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK - CPU modunda')
"
```

`cuda True` ve GPU adı görmelisiniz. `False` görüyorsanız adım 1a atlanmış demektir.

---

## 3. Hızlı test

```bash
env PYTHONPATH= PATH="$PWD/.venv/bin:$PATH" COQUI_TOS_AGREED=1 python deneme.py
```

`COQUI_TOS_AGREED=1`, XTTS'in lisans onayı sorusunu atlar (etkileşimsiz çalıştırmada gerekir).

`deneme.py` temel bir duman testidir: model indirir, ses klonlar, `output.wav` üretir. Tek
eksiği, GPU'ya düşülüp düşülmediğini söylememesi — CUDA çalışmıyorsa sessizce CPU'ya
geçer, sadece çok yavaşlar. Bunu görmek için başına şunu ekleyin:

```python
print("device:", device)
```

---

## 4. Hızlı çıkarım (XTTS)

Bu çatalda XTTS için iki hızlandırma var. Ayrıntılar: `docs/source/models/xtts.md`.

Doğrudan model API'siyle:

```python
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True, half=True)
out = model.inference(text, "tr", gpt_cond_latent, speaker_embedding,
                      enable_text_splitting=True, batch_size=4)
```

`TTS.api` üzerinden — dikkat: API metni modele ulaşmadan cümlelere böldüğü için batch ancak
bölmeyi modele bırakınca devreye girer:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.synthesizer.tts_model.use_half_precision()

tts.tts_to_file(
    text=uzun_metin,
    speaker_wav="Recording.wav",
    language="tr",
    file_path="output.wav",
    split_sentences=False,      # metnin tamamını XTTS'e ver
    enable_text_splitting=True,
    batch_size=4,
)
```

RTX 4050 Laptop'ta paragraf uzunluğunda metin: RTF 0.259 → 0.178 (fp16) → **0.092**
(fp16 + batch=4). VRAM 2.21 → 1.49 GB.

Ölçmek için:

```bash
env PYTHONPATH= PATH="$PWD/.venv/bin:$PATH" COQUI_TOS_AGREED=1 python scripts/bench_xtts_paragraph.py
```

---

## 5. Testler

```bash
# hızlı olanlar (~1 dk)
env PYTHONPATH= PATH="$PWD/.venv/bin:$PATH" python -m pytest tests/text_tests tests/data_tests -q

# ses işleme ve çıkarım (~3 dk)
env PYTHONPATH= PATH="$PWD/.venv/bin:$PATH" python -m pytest tests/aux_tests tests/inference_tests -q

# model ve eğitim testleri (~20 dk)
env PYTHONPATH= PATH="$PWD/.venv/bin:$PATH" python -m pytest tests/tts_tests tests/tts_tests2 tests/vocoder_tests \
  --continue-on-collection-errors -q
```

`pytest` ayrıca kurulmalıdır: `uv pip install pytest`.

### Bilinen ve kod hatası olmayan düşen testler

| Test | Neden |
|---|---|
| `test_phonemizer.py` espeak testleri (4) | espeak-ng 1.50 ile 1.51+ arasındaki fonem farkı (`ᵻ` ↔ `ɪ`) |
| `test_tokenizer.py::...eos_bos_and_blank` | aynı espeak-ng sürüm farkı (`c` ↔ `k`) |
| `test_korean_phonemizer.py` | `mecab` kurulu değil; yalnızca Korece'yi etkiler |
| `test_losses.py::BCELossTest` | `1.4e-45 != 0.0` — torch sürümünden gelen denormal sayı |
| `tests/vc_tests` (GPU'da) | Test CPU tensörü üretip CUDA modeline veriyor; `CUDA_VISIBLE_DEVICES="" ` ile 11/11 geçer |
| `test_xtts_v2-0_gpt_train.py` | XTTS v2 GPT eğitimi 6 GB VRAM'e sığmıyor; `CUDA_VISIBLE_DEVICES=""` ile geçer |

---

## 6. Bu makinede çalışmayanlar

- **DeepSpeed** — `nvcc` yok (inference kernel'lerini JIT derleyemez) ve depo, güncel
  DeepSpeed'in kaldırdığı `replace_method` argümanını geçiyor. Hızlandırma için yukarıdaki
  fp16 + batch yolunu kullanın.
- **XTTS v2 GPT eğitimi** — 6 GB VRAM yetmiyor.
