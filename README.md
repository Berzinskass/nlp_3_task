# Task 3: Work with YourTTS

## Setup (Python 3.10)

```bash
pip install -r requirements.txt
```

## Parts 0–2: YourTTS

1. Record yourself and put the recordings in this folder:
   - `reference_en.wav`: "This is a test of the speech synthesis system."
   - `reference_lt.wav`: "Tai yra šnekos sintezės sistemos testas."
2. Run `python part0_2_yourtts.py`. It prints the supported languages and
   creates `output_en.wav` and `output_lt.wav`.
3. Run `python part1_2_plots.py`. It prints the audio info (duration,
   sampling frequency, format) and the mean F0, and saves `plots_en.png` and
   `plots_lt.png` (waveform, magnitude spectrum and F0 contour).

## Part 3: EMO dataset

1. Download one archive part (e.g. `emo-wavs.zip.009`) and extract it into
   a folder called `emo_wavs`. If your unzip program cannot open it, run
   `python extract_zip_part.py emo-wavs.zip.009` instead.
2. Run `python part3_dataset.py`. It prints which speakers and emotions you
   have (e.g. `F3Z` = speaker F3, emotion Z).
3. Set `SPEAKER` and `EMOTION` at the top of `part3_dataset.py` and run it
   again. It creates:

```
dataset/
├── wavs/0001.wav, 0002.wav, ...
├── metadata_train.csv
└── metadata_val.csv
```

It also prints the numbers for the report and saves `duration_histogram.png`.
