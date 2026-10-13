"""Step 3: build a Coqui TTS dataset (one speaker, one emotion) from the EMO corpus.

    # 1) see which speaker/emotion pairs your archive part contains
    python step3_prepare_dataset.py --archive emo-wavs.zip.009 --list

    # 2) build the dataset (pair with the most recordings if not given)
    python step3_prepare_dataset.py --archive emo-wavs.zip.009 --speaker F3 --emotion Z

--archive may also be a folder that already contains the extracted WAV files.
Output:
    dataset/wavs/0001.wav ...           renamed recordings
    dataset/metadata_train.csv           0001|text|text   (Coqui "ljspeech" format)
    dataset/metadata_val.csv
    dataset/file_mapping.csv            new name -> original EMO file
    results/dataset/stats.json, stats.md, duration_histogram.png
"""

import argparse
import collections
import csv
import gzip
import json
import os
import random
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tts_lab.audio_analysis import BIT_DEPTH  # noqa: E402
from tts_lab.zip_parts import extract_wavs  # noqa: E402

EMOTIONS = {
    "D": "Joy (Džiaugsmas)",
    "L": "Sadness (Liūdesys)",
    "N": "Surprise (Nuostaba)",
    "P": "Anger (Pyktis)",
    "Z": "Neutral (Neutrali)",
}
DEFAULT_METADATA = os.path.join("data", "metadata_of_EMO_dataset.csv.gz")


def load_metadata(path):
    """basename -> {path, speaker, emotion, session, text}.

    Lines look like: wavs/D/F2/0701-F2D/0701-F2D-0001.wav|07|džiugios naujienos ...
    """
    opener = gzip.open if path.endswith(".gz") else open
    meta = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            rel_path, session, text = line.split("|", 2)
            parts = rel_path.split("/")
            meta[parts[-1]] = {"path": rel_path, "emotion": parts[1],
                               "speaker": parts[2], "session": session,
                               "text": text.strip()}
    return meta


def get_wav_dir(archive, extract_dir):
    if os.path.isdir(archive):
        return archive
    if os.path.isdir(extract_dir) and any(
            n.lower().endswith(".wav") for n in os.listdir(extract_dir)):
        print(f"Using WAVs already extracted to {extract_dir}/ "
              "(delete it to re-extract)")
        return extract_dir
    print(f"Extracting WAV files from {archive} to {extract_dir}/ ...")
    files = extract_wavs(archive, extract_dir)
    print(f"Extracted {len(files)} WAV files")
    return extract_dir


def find_wavs(wav_dir):
    found = {}
    for root, _, names in os.walk(wav_dir):
        for name in names:
            if name.lower().endswith(".wav"):
                found[name] = os.path.join(root, name)
    return found


def print_pairs(pairs):
    print("\nAvailable speaker / emotion pairs in this archive:")
    print(f"  {'speaker':8s} {'emotion':24s} {'files':>6s}")
    for (spk, emo), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {spk:8s} {emo + ' ' + EMOTIONS.get(emo, ''):24s} {n:6d}")


def write_metadata(path, items):
    # Coqui "ljspeech" formatter: <id>|<transcription>|<normalized transcription>,
    # the id is the file name in wavs/ without ".wav".
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(f"{item['id']}|{item['text']}|{item['text']}\n")


def duration_stats(durations):
    d = np.asarray(durations)
    return {"count": int(len(d)), "total_s": float(d.sum()),
            "total_hms": hms(d.sum()), "min_s": float(d.min()),
            "max_s": float(d.max()), "mean_s": float(d.mean()),
            "median_s": float(np.median(d)), "std_s": float(d.std())}


def hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def plot_histogram(durations, path, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, np.ceil(max(durations)) + 0.5, 0.5)
    ax.hist(durations, bins=bins, color="#1f77b4", edgecolor="white")
    mean = float(np.mean(durations))
    ax.axvline(mean, color="#d62728", linestyle="--", label=f"mean = {mean:.2f} s")
    ax.set_xlabel("Recording duration (s)")
    ax.set_ylabel("Number of recordings")
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True,
                        help="EMO archive part (e.g. emo-wavs.zip.009) or folder of WAVs")
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--extract-dir", default="emo_raw")
    parser.add_argument("--speaker", help="e.g. F3, M3 (default: pair with most files)")
    parser.add_argument("--emotion", choices=sorted(EMOTIONS),
                        help="D, L, N, P or Z (default: pair with most files)")
    parser.add_argument("--list", action="store_true",
                        help="only list available speaker/emotion pairs")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--report-dir", default=os.path.join("results", "dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sr", type=int, default=None,
                        help="resample to this rate (default: keep original files)")
    args = parser.parse_args()

    meta = load_metadata(args.metadata)
    print(f"Metadata: {len(meta)} transcribed recordings")
    wavs = find_wavs(get_wav_dir(args.archive, args.extract_dir))
    known = {name: path for name, path in wavs.items() if name in meta}
    print(f"WAV files found: {len(wavs)}, with transcription: {len(known)}")
    if not known:
        raise SystemExit("No WAV file matches the metadata - check --archive.")

    pairs = collections.Counter(
        (meta[n]["speaker"], meta[n]["emotion"]) for n in known)
    print_pairs(pairs)
    if args.list:
        return

    candidates = [p for p in pairs
                  if (args.speaker is None or p[0] == args.speaker)
                  and (args.emotion is None or p[1] == args.emotion)]
    if not candidates:
        raise SystemExit(f"No recordings for speaker={args.speaker} "
                         f"emotion={args.emotion} in this archive.")
    speaker, emotion = max(candidates, key=lambda p: pairs[p])
    selected = sorted(n for n in known
                      if meta[n]["speaker"] == speaker and meta[n]["emotion"] == emotion)
    print(f"\nSelected speaker {speaker}, emotion {emotion} "
          f"({EMOTIONS[emotion]}): {len(selected)} recordings")

    # Copy (or resample) into dataset/wavs/0001.wav, ...
    wav_out = os.path.join(args.out, "wavs")
    if os.path.isdir(wav_out):
        shutil.rmtree(wav_out)
    os.makedirs(wav_out)
    width = max(4, len(str(len(selected))))
    items, params = [], collections.Counter()
    for i, name in enumerate(selected, 1):
        src = known[name]
        info = sf.info(src)
        params[(info.samplerate, BIT_DEPTH.get(info.subtype, info.subtype),
                info.channels)] += 1
        item = {"id": f"{i:0{width}d}", "original": name,
                "original_path": meta[name]["path"], "text": meta[name]["text"],
                "duration": info.frames / info.samplerate}
        dst = os.path.join(wav_out, item["id"] + ".wav")
        if args.target_sr and args.target_sr != info.samplerate:
            import librosa

            y, _ = librosa.load(src, sr=args.target_sr, mono=True)
            sf.write(dst, y, args.target_sr, subtype="PCM_16")
        else:
            shutil.copyfile(src, dst)
        items.append(item)

    # 90 / 10 split.
    shuffled = items[:]
    random.Random(args.seed).shuffle(shuffled)
    n_val = max(1, round(len(items) * args.val_ratio)) if len(items) > 1 else 0
    val = sorted(shuffled[:n_val], key=lambda x: x["id"])
    train = sorted(shuffled[n_val:], key=lambda x: x["id"])
    write_metadata(os.path.join(args.out, "metadata_train.csv"), train)
    write_metadata(os.path.join(args.out, "metadata_val.csv"), val)
    with open(os.path.join(args.out, "file_mapping.csv"), "w", encoding="utf-8",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "split", "original_file", "duration_s", "text"])
        val_ids = {x["id"] for x in val}
        for x in items:
            writer.writerow([x["id"], "val" if x["id"] in val_ids else "train",
                             x["original_path"], f"{x['duration']:.3f}", x["text"]])

    # Report.
    os.makedirs(args.report_dir, exist_ok=True)
    durations = [x["duration"] for x in items]
    hist_path = os.path.join(args.report_dir, "duration_histogram.png")
    plot_histogram(durations, hist_path,
                   f"EMO speaker {speaker}, emotion {emotion} - {EMOTIONS[emotion]} "
                   f"(n = {len(items)})")
    out_info = sf.info(os.path.join(wav_out, items[0]["id"] + ".wav"))
    stats = {
        "archive": os.path.basename(os.path.normpath(args.archive)),
        "speaker": speaker,
        "speaker_gender": {"F": "female", "M": "male"}.get(speaker[0], "unknown"),
        "emotion": emotion,
        "emotion_name": EMOTIONS[emotion],
        "sessions": sorted({meta[x["original"]]["session"] for x in items}),
        "audio_parameters_original": [
            {"sample_rate_hz": sr, "bit_depth": bits, "channels": ch, "files": n}
            for (sr, bits, ch), n in params.items()],
        "audio_parameters_dataset": {
            "sample_rate_hz": out_info.samplerate,
            "bit_depth": BIT_DEPTH.get(out_info.subtype, out_info.subtype),
            "channels": out_info.channels, "format": out_info.format},
        "all": duration_stats(durations),
        "train": duration_stats([x["duration"] for x in train]),
        "val": duration_stats([x["duration"] for x in val]) if val else None,
        "split": {"train": len(train), "val": len(val), "seed": args.seed},
        "histogram": hist_path,
    }
    with open(os.path.join(args.report_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    md = stats_markdown(stats)
    with open(os.path.join(args.report_dir, "stats.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"Dataset written to {args.out}/, report to {args.report_dir}/")


def stats_markdown(s):
    a = s["all"]
    lines = [
        "# EMO dataset subset", "",
        f"- Archive: `{s['archive']}`",
        f"- Speaker: **{s['speaker']}** ({s['speaker_gender']})",
        f"- Emotion: **{s['emotion']} - {s['emotion_name']}**",
        f"- Recording sessions: {', '.join(s['sessions'])}", "",
        "## Audio parameters", "",
        "Original EMO recordings:", "",
        "| Sampling frequency (Hz) | Bit depth | Channels | Files |", "|---|---|---|---|",
    ]
    for p in s["audio_parameters_original"]:
        lines.append(f"| {p['sample_rate_hz']} | {p['bit_depth']} | "
                     f"{p['channels']} | {p['files']} |")
    d = s["audio_parameters_dataset"]
    lines += ["", f"Files in dataset/wavs: {d['sample_rate_hz']} Hz, "
                  f"{d['bit_depth']}-bit, {d['channels']} channel(s), {d['format']}."]
    lines += [
        "", "## Durations", "",
        "| Set | Recordings | Total | Min (s) | Max (s) | Mean (s) | Median (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("all", "train", "val"):
        d = s[name]
        if d:
            lines.append(f"| {name} | {d['count']} | {d['total_hms']} "
                         f"({d['total_s']:.1f} s) | {d['min_s']:.2f} | {d['max_s']:.2f} "
                         f"| {d['mean_s']:.2f} | {d['median_s']:.2f} |")
    lines += ["", f"Total: {a['count']} recordings, {a['total_hms']} "
                  f"({a['total_s'] / 60:.1f} min).",
              f"Split: {s['split']['train']} training / {s['split']['val']} validation "
              f"(seed {s['split']['seed']}).", "",
              f"Histogram: `{s['histogram']}`"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
