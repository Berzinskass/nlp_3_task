"""Optional helper: record a reference sentence from the microphone.

    python record_reference.py --lang en      # -> audio/reference_en.wav
    python record_reference.py --lang lt      # -> audio/reference_lt.wav

Any other recorder (phone, Audacity) is fine too - just save a WAV file
under audio/ with the same name. Requires: pip install sounddevice
"""

import argparse
import os
import time

import soundfile as sf

SENTENCES = {
    "en": "This is a test of the speech synthesis system.",
    "lt": "Tai yra šnekos sintezės sistemos testas.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=sorted(SENTENCES), required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--out-dir", default="audio")
    args = parser.parse_args()

    import sounddevice as sd

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"reference_{args.lang}.wav")
    print(f'Read aloud: "{SENTENCES[args.lang]}"')
    for i in (3, 2, 1):
        print(f"{i}...")
        time.sleep(1)
    print(f"Recording {args.seconds:.0f} s ...")
    audio = sd.rec(int(args.seconds * args.sr), samplerate=args.sr, channels=1,
                   dtype="int16")
    sd.wait()
    sf.write(out, audio, args.sr, subtype="PCM_16")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
