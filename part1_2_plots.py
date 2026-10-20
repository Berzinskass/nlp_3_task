# Parts 1-2: compare natural and synthesized speech.
# For each pair it prints the audio info and saves one picture with
# the waveform, magnitude spectrum and F0 contour of both recordings.

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def compare(natural_file, synthesized_file, picture_name):
    fig, axes = plt.subplots(3, 2, figsize=(12, 6.5))

    for col, file in enumerate([natural_file, synthesized_file]):
        # Audio info: duration, sampling frequency, format
        info = sf.info(file)
        print(f"{file}: {info.duration:.2f} s, {info.samplerate} Hz, "
              f"{info.channels} channel(s), {info.format} {info.subtype}")

        y, sr = librosa.load(file, sr=None)  # sr=None keeps the original rate
        time = np.arange(len(y)) / sr

        # 1) Waveform
        axes[0, col].plot(time, y)
        axes[0, col].set_title(f"Waveform: {file}")
        axes[0, col].set_xlabel("Time (s)")

        # 2) Magnitude spectrum (FFT of the whole signal)
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        axes[1, col].plot(freqs, 20 * np.log10(spectrum + 1e-10))
        axes[1, col].set_title(f"Magnitude spectrum: {file}")
        axes[1, col].set_xlabel("Frequency (Hz)")
        axes[1, col].set_ylabel("dB")

        # 3) F0 (fundamental frequency); NaN where there is no voice
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        f0_time = librosa.times_like(f0, sr=sr)
        axes[2, col].plot(f0_time, f0, ".")
        axes[2, col].set_title(f"F0 contour: {file}")
        axes[2, col].set_xlabel("Time (s)")
        axes[2, col].set_ylabel("F0 (Hz)")

        print(f"   mean F0 = {np.nanmean(f0):.0f} Hz, "
              f"F0 range = {np.nanmin(f0):.0f}-{np.nanmax(f0):.0f} Hz")

        # Speech duration without the silence at the start and end
        _, (start, end) = librosa.effects.trim(y, top_db=35)
        print(f"   speech duration (silence removed) = {(end - start) / sr:.2f} s")

        # Share of the 0-8 kHz energy that lies between 4 and 8 kHz
        energy = spectrum ** 2
        share = energy[(freqs >= 4000) & (freqs < 8000)].sum() / energy[freqs < 8000].sum()
        print(f"   energy at 4-8 kHz = {share:.1%} of the 0-8 kHz energy")

    plt.tight_layout()
    plt.savefig(picture_name)
    print("Saved", picture_name, "\n")


compare("reference_en.wav", "output_en.wav", "plots_en.png")
compare("reference_lt.wav", "output_lt.wav", "plots_lt.png")
