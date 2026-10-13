"""Steps 1-2 report material: compare natural and synthesized speech.

For each part this saves to results/<part>/:
  waveforms.png, magnitude_spectra.png, f0_contours.png, spectrograms.png,
  summary.json and summary.md (audio parameters + comparison table).

    python step1_2_analyze.py                  # en and lt
    python step1_2_analyze.py --part lt --synth audio/output_lt_ascii.wav
"""

import argparse
import os

from tts_lab.audio_analysis import plot_comparison

PARTS = {
    "en": ("reference_en.wav", "output_en.wav", "English (zero-shot)"),
    "lt": ("reference_lt.wav", "output_lt.wav", "Lithuanian (zero-shot, language='en')"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["en", "lt", "all"], default="all")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--natural", help="override the natural recording path")
    parser.add_argument("--synth", help="override the synthesized recording path")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    parts = ["en", "lt"] if args.part == "all" else [args.part]
    for part in parts:
        natural, synth, title = PARTS[part]
        natural = args.natural or os.path.join(args.audio_dir, natural)
        synth = args.synth or os.path.join(args.audio_dir, synth)
        out_dir = os.path.join(args.out_dir, part)
        if args.synth:
            out_dir += "_" + os.path.splitext(os.path.basename(args.synth))[0]
        missing = [p for p in (natural, synth) if not os.path.exists(p)]
        if missing:
            print(f"[{part}] skipped, missing: {', '.join(missing)}")
            continue
        summary = plot_comparison(natural, synth, out_dir, title)
        nat, syn = summary["natural"], summary["synthesized"]
        info = nat["file_info"]
        print(f"\n=== {title} ===")
        print(f"Reference: {info['file']} | {info['duration_s']:.2f} s | "
              f"{info['sample_rate_hz']} Hz | {info['format_info']} "
              f"{info['subtype']} | {info['channels']} ch")
        print(f"Synthesized: {syn['duration_s']:.2f} s @ {syn['sample_rate_hz']} Hz")
        print(f"Speech duration: natural {nat['speech_duration_s']:.2f} s, "
              f"synthesized {syn['speech_duration_s']:.2f} s")
        print(f"F0 mean: natural {nat['f0_mean_hz'] or 0:.1f} Hz, "
              f"synthesized {syn['f0_mean_hz'] or 0:.1f} Hz")
        print(f"Results in {out_dir}/ (see summary.md)")


if __name__ == "__main__":
    main()
