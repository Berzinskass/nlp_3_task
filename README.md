# Task 3: Work with YourTTS

Code for the lab: zero-shot synthesis with the pretrained YourTTS model
(English and Lithuanian), plus preparing a Lithuanian EMO-corpus dataset for
fine-tuning. Every script writes its numbers and figures to `results/`, so
they can go straight into the report.

| Step | Script | Output |
|---|---|---|
| 0. Supported languages | `step0_check_languages.py` | `results/supported_languages.json` |
| 1–2. Record references | `record_reference.py` (optional) | `audio/reference_en.wav`, `audio/reference_lt.wav` |
| 1–2. Synthesis | `step1_2_synthesize.py` | `audio/output_en.wav`, `audio/output_lt.wav` |
| 1–2. Analysis | `step1_2_analyze.py` | `results/en/`, `results/lt/` |
| 3. EMO dataset | `step3_prepare_dataset.py` | `dataset/`, `results/dataset/` |

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you are on Python 3.12+ (Google Colab, for example), install `coqui-tts`
instead of `TTS==0.22.0`. The scripts work with either one.

## Step 0: supported languages

```bash
python step0_check_languages.py
```

The script prints `tts.languages` and whether Lithuanian is among them. For
YourTTS the list is `['en', 'fr-fr', 'pt-br']`, so Lithuanian is **not**
supported. The model downloads automatically on the first run (~400 MB).

## Steps 1–2: zero-shot synthesis (English and Lithuanian)

1. Record yourself and save the files as `audio/reference_en.wav` and
   `audio/reference_lt.wav`. Any recorder works (phone, Audacity), or run:
   ```bash
   pip install sounddevice
   python record_reference.py --lang en
   python record_reference.py --lang lt
   ```
2. Synthesize both sentences in your voice:
   ```bash
   python step1_2_synthesize.py                  # en + lt (lt uses language="en")
   python step1_2_synthesize.py --ascii-variant  # also writes output_lt_ascii.wav (š→s, ė→e, ...)
   ```
   If you recorded a different unsupported language, pass
   `--text-lt "your sentence"`.
3. Analyze:
   ```bash
   python step1_2_analyze.py
   python step1_2_analyze.py --part lt --synth audio/output_lt_ascii.wav   # optional
   ```

The analysis writes these files to `results/en/` and `results/lt/`:

- `summary.md` / `summary.json`: reference-recording information (duration,
  sampling frequency, format, bit depth, channels) and a table comparing
  natural and synthesized speech. The table covers total duration and speech
  duration (silence trimmed), leading/trailing silence, spectral centroid,
  85% roll-off, effective bandwidth, F0 mean/median/std/min/max/range in
  semitones, and the share of voiced frames.
- `waveforms.png`: both waveforms on a common time axis, with the speech
  region shaded.
- `magnitude_spectra.png`: the long-term magnitude spectrum at full
  resolution and smoothed on a log-frequency axis.
- `f0_contours.png`: F0 contours from pYIN (60–500 Hz), plotted against
  absolute time and against time normalized to the speech region, so
  intonation patterns can be compared even when durations differ.
- `spectrograms.png`: an extra figure that makes the bandwidth difference
  easy to see.

### What to look for in the discussion

- **Duration:** synthesized speech usually has a different speaking rate and
  shorter pauses. Compare *speech duration* rather than file length, because
  your recording probably has silence at the start and end.
- **Spectrum:** YourTTS outputs **16 kHz** audio, so its spectrum stops at
  8 kHz. Your recording (typically 44.1/48 kHz) has energy above that, which
  shows up as fricatives (s, š, f) sounding duller in the synthesized
  version. Compare the formant region (300–3500 Hz) on the smoothed spectra.
- **F0:** check whether the mean F0 matches your voice (speaker similarity)
  and whether the synthesized contour is flatter (smaller std and range),
  which is typical for TTS.
- **Lithuanian with `language="en"`:** the model reads the text with English
  pronunciation rules. Characters missing from its vocabulary (š, ė, ž, č,
  ą, ę, į, ų, ū) are usually **dropped**; check the console for
  `Character ... not found in the vocabulary` warnings and mention them in
  the report. Compare `output_lt.wav` with `output_lt_ascii.wav`.

## Step 3: Lithuanian dataset (EMO corpus)

The full corpus metadata (349,696 transcribed recordings) is in
`data/metadata_of_EMO_dataset.csv.gz`. Each line has the form
`wavs/<emotion>/<speaker>/<session-speaker-emotion>/<file>.wav|<session>|<text>`.
Speakers are F2, F3, F4, M3 and M4; emotions are D (joy), L (sadness),
N (surprise), P (anger) and Z (neutral).

1. Download **one** archive part, e.g. `emo-wavs.zip.009`. A single part of
   a split ZIP cannot be opened by normal unzip tools, so the script reads it
   directly: it recovers every complete WAV in that part and skips the two
   files cut at the part boundaries. A complete `.zip`, or a folder with
   already-extracted WAVs, also works.
2. See which speaker/emotion pairs the part contains:
   ```bash
   python step3_prepare_dataset.py --archive emo-wavs.zip.009 --list
   ```
3. Build the dataset:
   ```bash
   python step3_prepare_dataset.py --archive emo-wavs.zip.009 --speaker F3 --emotion Z
   ```
   Without `--speaker`/`--emotion`, the script picks the pair with the most
   recordings. Add `--target-sr 16000` to resample to YourTTS's rate (the
   original files are copied unchanged by default). The split is 90/10 with
   a fixed `--seed 42`.

Output:

```
dataset/
├── wavs/0001.wav, 0002.wav, ...
├── metadata_train.csv      # 0001|transcription|transcription
├── metadata_val.csv
└── file_mapping.csv        # new id -> original EMO file, split, duration, text
results/dataset/
├── stats.md / stats.json   # speaker, emotion, count, total/min/max/mean duration,
│                           # sampling rate, bit depth, channels, train/val sizes
└── duration_histogram.png
```

The metadata uses the Coqui `ljspeech` format (`<id>|<text>|<normalized
text>`; the id is the file name in `wavs/` without `.wav`). It can be loaded
for fine-tuning with:

```python
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

config = BaseDatasetConfig(formatter="ljspeech", path="dataset/",
                           meta_file_train="metadata_train.csv",
                           meta_file_val="metadata_val.csv", language="lt")
train, val = load_tts_samples(config, eval_split=True)
print(len(train), len(val))
```
