
import glob
import gzip
import os
import random
import shutil
from collections import Counter

import matplotlib.pyplot as plt
import soundfile as sf

WAV_FOLDER = "emo_wavs"
SPEAKER = "F3"  # F2, F3, F4, M3 or M4
EMOTION = "Z"   # D joy, L sadness, N surprise, P anger, Z neutral

all_wavs = glob.glob(f"{WAV_FOLDER}/**/*.wav", recursive=True)
print("Speaker+emotion found:", Counter(os.path.basename(w).split("-")[1] for w in all_wavs))

texts = {}
with gzip.open("metadata_of_EMO_dataset.csv.gz", "rt", encoding="utf-8") as f:
    for line in f:
        path, session, text = line.strip().split("|")
        texts[os.path.basename(path)] = text

selected = sorted(w for w in all_wavs if f"-{SPEAKER}{EMOTION}-" in os.path.basename(w)
                  and os.path.basename(w) in texts)  # skip files without transcription
print(f"Selected speaker {SPEAKER}, emotion {EMOTION}: {len(selected)} recordings")
if not selected:
    exit("No files - change SPEAKER and EMOTION to one of the pairs above.")

os.makedirs("dataset/wavs", exist_ok=True)
lines = []
durations = []
for i, wav in enumerate(selected, start=1):
    new_name = f"{i:04d}"
    shutil.copy(wav, f"dataset/wavs/{new_name}.wav")
    text = texts[os.path.basename(wav)]
    lines.append(f"{new_name}|{text}|{text}")  # Coqui "ljspeech" format
    durations.append(sf.info(wav).duration)


random.seed(42)
random.shuffle(lines)
n_val = max(1, len(lines) // 10)
with open("dataset/metadata_val.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(lines[:n_val]) + "\n")
with open("dataset/metadata_train.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(lines[n_val:]) + "\n")

info = sf.info(selected[0])
print(f"Sampling frequency: {info.samplerate} Hz, format: {info.subtype}, "
      f"channels: {info.channels}")
print(f"Total: {len(durations)} recordings, {sum(durations) / 60:.1f} minutes")
print(f"Duration min {min(durations):.2f} s, max {max(durations):.2f} s, "
      f"mean {sum(durations) / len(durations):.2f} s")
print(f"Training set: {len(lines) - n_val}, validation set: {n_val}")

plt.hist(durations, bins=20)
plt.xlabel("Duration (s)")
plt.ylabel("Number of recordings")
plt.title(f"Recording durations, speaker {SPEAKER}, emotion {EMOTION}")
plt.savefig("duration_histogram.png")

