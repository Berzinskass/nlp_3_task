"""Signal analysis used in the report: audio parameters, waveforms,
magnitude spectra and F0 contours of natural vs. synthesized speech."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

BIT_DEPTH = {
    "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT": 32, "DOUBLE": 64, "ULAW": 8, "ALAW": 8,
}

F0_MIN, F0_MAX = 60.0, 500.0
NATURAL_COLOR, SYNTH_COLOR = "#1f77b4", "#d62728"


def audio_info(path):
    """Container-level parameters of an audio file."""
    info = sf.info(path)
    return {
        "file": os.path.basename(path),
        "format": info.format,
        "format_info": info.format_info,
        "subtype": info.subtype,
        "bit_depth": BIT_DEPTH.get(info.subtype),
        "sample_rate_hz": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_s": info.frames / info.samplerate,
    }


def load_mono(path):
    """Load audio at its native sampling rate, mixed down to mono."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def speech_bounds(y, sr, top_db=35):
    """Start/end time (s) of the non-silent part of the signal."""
    _, (start, end) = librosa.effects.trim(y, top_db=top_db)
    return start / sr, end / sr


def magnitude_spectrum(y, sr):
    """Long-term magnitude spectrum of the whole signal in dB (re. max)."""
    n_fft = int(2 ** np.ceil(np.log2(len(y))))
    mag = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mag_db = 20 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)
    return freqs, mag_db


def smooth_spectrum(freqs, mag_db, bandwidth_hz=50.0):
    """Moving-average smoothing so the spectral envelope is readable."""
    step = freqs[1] - freqs[0]
    width = max(1, int(round(bandwidth_hz / step)))
    return uniform_filter1d(mag_db, width, mode="nearest")


def f0_contour(y, sr, fmin=F0_MIN, fmax=F0_MAX):
    """F0 contour with probabilistic YIN; unvoiced frames are NaN."""
    frame_length = int(2 ** np.ceil(np.log2(0.064 * sr)))
    hop_length = int(round(0.010 * sr))
    f0, voiced, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0 = np.where(voiced, f0, np.nan)
    return times, f0


def bandwidth_limit(freqs, mag_db, threshold_db=-60.0):
    """Highest frequency whose smoothed level is above `threshold_db`."""
    smoothed = smooth_spectrum(freqs, mag_db, 100.0)
    above = np.nonzero(smoothed > threshold_db)[0]
    return float(freqs[above[-1]]) if len(above) else 0.0


def signal_summary(y, sr):
    """Numbers used to compare natural and synthesized speech."""
    start, end = speech_bounds(y, sr)
    freqs, mag_db = magnitude_spectrum(y, sr)
    _, f0 = f0_contour(y, sr)
    voiced = f0[~np.isnan(f0)]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    rms = librosa.feature.rms(y=y)[0]
    active = rms > 0.1 * rms.max()

    def stat(fn, values):
        return float(fn(values)) if len(values) else None

    return {
        "sample_rate_hz": sr,
        "duration_s": len(y) / sr,
        "speech_duration_s": end - start,
        "leading_silence_s": start,
        "trailing_silence_s": len(y) / sr - end,
        "peak_amplitude": float(np.max(np.abs(y))),
        "rms": float(np.sqrt(np.mean(y ** 2))),
        "spectral_centroid_hz": stat(np.mean, centroid[active]),
        "spectral_rolloff85_hz": stat(np.mean, rolloff[active]),
        "bandwidth_limit_60db_hz": bandwidth_limit(freqs, mag_db),
        "f0_mean_hz": stat(np.mean, voiced),
        "f0_median_hz": stat(np.median, voiced),
        "f0_std_hz": stat(np.std, voiced),
        "f0_min_hz": stat(np.min, voiced),
        "f0_max_hz": stat(np.max, voiced),
        "f0_range_semitones": (
            float(12 * np.log2(voiced.max() / voiced.min())) if len(voiced) else None
        ),
        "voiced_fraction": float(np.mean(~np.isnan(f0))),
    }


def _time_axis(y, sr):
    return np.arange(len(y)) / sr


def plot_comparison(natural_path, synth_path, out_dir, title):
    """Save the comparison figures and return the summary of both signals."""
    os.makedirs(out_dir, exist_ok=True)
    signals = {
        "Natural": load_mono(natural_path),
        "Synthesized": load_mono(synth_path),
    }
    colors = {"Natural": NATURAL_COLOR, "Synthesized": SYNTH_COLOR}
    f0s = {name: f0_contour(y, sr) for name, (y, sr) in signals.items()}
    max_t = max(len(y) / sr for y, sr in signals.values())
    figures = {}

    # 1. Waveforms on a common time axis.
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, sharey=True)
    for ax, (name, (y, sr)) in zip(axes, signals.items()):
        ax.plot(_time_axis(y, sr), y, color=colors[name], linewidth=0.6)
        start, end = speech_bounds(y, sr)
        ax.axvspan(start, end, color="grey", alpha=0.12, label="speech region")
        ax.set_title(f"{name} ({sr} Hz, {len(y) / sr:.2f} s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, max_t)
    fig.suptitle(f"{title}: waveforms")
    fig.tight_layout()
    figures["waveforms"] = _save(fig, out_dir, "waveforms.png")

    # 2. Magnitude spectra (raw + smoothed envelope).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (y, sr) in signals.items():
        freqs, mag_db = magnitude_spectrum(y, sr)
        axes[0].plot(freqs, mag_db, color=colors[name], linewidth=0.3, alpha=0.6,
                     label=name)
        axes[1].plot(freqs, smooth_spectrum(freqs, mag_db), color=colors[name],
                     linewidth=1.2, label=f"{name} (Nyquist {sr // 2} Hz)")
    for ax, sub in zip(axes, ["full resolution", "smoothed (50 Hz)"]):
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB re. max)")
        ax.set_ylim(-100, 5)
        ax.set_title(sub)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].set_xscale("log")
    axes[1].set_xlim(50, max(sr for _, sr in signals.values()) / 2)
    axes[1].set_xlabel("Frequency (Hz, log scale)")
    fig.suptitle(f"{title}: magnitude spectra")
    fig.tight_layout()
    figures["spectra"] = _save(fig, out_dir, "magnitude_spectra.png")

    # 3. F0 contours: absolute time and normalized to the speech region.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (y, sr) in signals.items():
        times, f0 = f0s[name]
        start, end = speech_bounds(y, sr)
        axes[0].plot(times, f0, ".-", color=colors[name], markersize=2.5,
                     linewidth=1, label=name)
        rel = (times - start) / max(end - start, 1e-6)
        axes[1].plot(rel, f0, ".-", color=colors[name], markersize=2.5,
                     linewidth=1, label=name)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_xlim(0, max_t)
    axes[0].set_title("absolute time")
    axes[1].set_xlabel("Relative position in utterance (speech start = 0, end = 1)")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_title("time-normalized")
    for ax in axes:
        ax.set_ylabel("F0 (Hz)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{title}: fundamental frequency (pYIN)")
    fig.tight_layout()
    figures["f0"] = _save(fig, out_dir, "f0_contours.png")

    # 4. Spectrograms (extra, helps explain spectral differences).
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    top = max(sr for _, sr in signals.values()) / 2
    for ax, (name, (y, sr)) in zip(axes, signals.items()):
        n_fft = int(2 ** np.ceil(np.log2(0.025 * sr)))
        spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4)), ref=np.max
        )
        img = librosa.display.specshow(spec, sr=sr, hop_length=n_fft // 4,
                                       x_axis="time", y_axis="hz", ax=ax,
                                       vmin=-80, vmax=0)
        ax.set_ylim(0, top)
        ax.set_title(f"{name} (Nyquist {sr // 2} Hz)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    axes[-1].set_xlim(0, max_t)
    fig.suptitle(f"{title}: spectrograms")
    fig.tight_layout()
    figures["spectrograms"] = _save(fig, out_dir, "spectrograms.png")

    summary = {
        "natural": {"file_info": audio_info(natural_path),
                    **signal_summary(*signals["Natural"])},
        "synthesized": {"file_info": audio_info(synth_path),
                        **signal_summary(*signals["Synthesized"])},
        "figures": figures,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_markdown(summary, title))
    return summary


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


ROWS = [
    ("Sampling frequency (Hz)", "sample_rate_hz", "{:.0f}"),
    ("Total duration (s)", "duration_s", "{:.2f}"),
    ("Speech duration, silence trimmed (s)", "speech_duration_s", "{:.2f}"),
    ("Leading silence (s)", "leading_silence_s", "{:.2f}"),
    ("Trailing silence (s)", "trailing_silence_s", "{:.2f}"),
    ("Peak amplitude", "peak_amplitude", "{:.3f}"),
    ("RMS", "rms", "{:.4f}"),
    ("Mean spectral centroid (Hz)", "spectral_centroid_hz", "{:.0f}"),
    ("Mean 85% roll-off (Hz)", "spectral_rolloff85_hz", "{:.0f}"),
    ("Bandwidth, -60 dB (Hz)", "bandwidth_limit_60db_hz", "{:.0f}"),
    ("F0 mean (Hz)", "f0_mean_hz", "{:.1f}"),
    ("F0 median (Hz)", "f0_median_hz", "{:.1f}"),
    ("F0 std (Hz)", "f0_std_hz", "{:.1f}"),
    ("F0 min / max (Hz)", ("f0_min_hz", "f0_max_hz"), "{:.1f}"),
    ("F0 range (semitones)", "f0_range_semitones", "{:.1f}"),
    ("Voiced frames (%)", "voiced_fraction", "{:.1%}"),
]


def _fmt(data, key, fmt):
    if isinstance(key, tuple):
        return " / ".join(_fmt(data, k, fmt) for k in key)
    value = data.get(key)
    return "n/a" if value is None else fmt.format(value)


def summary_markdown(summary, title):
    nat, syn = summary["natural"], summary["synthesized"]
    lines = [f"# {title}", "", "## Files", "",
             "| | Natural | Synthesized |", "|---|---|---|"]
    for label, key in [("File", "file"), ("Format", "format_info"),
                       ("Encoding", "subtype"), ("Bit depth", "bit_depth"),
                       ("Channels", "channels"),
                       ("Sampling frequency (Hz)", "sample_rate_hz")]:
        lines.append(f"| {label} | {nat['file_info'][key]} | {syn['file_info'][key]} |")
    lines += ["", "## Signal comparison", "",
              "| Measure | Natural | Synthesized |", "|---|---|---|"]
    for label, key, fmt in ROWS:
        lines.append(f"| {label} | {_fmt(nat, key, fmt)} | {_fmt(syn, key, fmt)} |")
    lines += ["", "## Figures", ""]
    for name, path in summary["figures"].items():
        lines.append(f"- {name}: `{os.path.basename(path)}`")
    return "\n".join(lines) + "\n"

