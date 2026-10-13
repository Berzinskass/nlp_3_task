"""Steps 1-2: zero-shot synthesis with YourTTS from your own reference recordings.

    python step1_2_synthesize.py                # both English and Lithuanian
    python step1_2_synthesize.py --part en
    python step1_2_synthesize.py --part lt --ascii-variant

Lithuanian is not supported by YourTTS, so the Lithuanian sentence is
synthesized with the English language setting (as required by the task).
"""

import argparse
import os
import unicodedata

from tts_lab.model import load_yourtts

TEXT_EN = "This is a test of the speech synthesis system."
TEXT_LT = "Tai yra šnekos sintezės sistemos testas."


def strip_diacritics(text):
    """š -> s, ė -> e, ... (characters YourTTS's English vocabulary knows)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def synthesize(tts, text, speaker_wav, language, out_path):
    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(
            f"{speaker_wav} not found - record it first (see record_reference.py)"
        )
    print(f"[{language}] '{text}' with voice of {speaker_wav} -> {out_path}")
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language,
                    file_path=out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["en", "lt", "all"], default="all")
    parser.add_argument("--audio-dir", default="audio",
                        help="folder with reference_en.wav / reference_lt.wav")
    parser.add_argument("--text-en", default=TEXT_EN)
    parser.add_argument("--text-lt", default=TEXT_LT,
                        help="change it if you recorded another unsupported language")
    parser.add_argument("--lt-language", default="en",
                        help="YourTTS language setting used for the Lithuanian text")
    parser.add_argument("--ascii-variant", action="store_true",
                        help="also synthesize the LT text with diacritics removed")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    tts = load_yourtts(args.device)
    print("Supported languages:", tts.languages)
    d = args.audio_dir

    if args.part in ("en", "all"):
        synthesize(tts, args.text_en, os.path.join(d, "reference_en.wav"), "en",
                   os.path.join(d, "output_en.wav"))

    if args.part in ("lt", "all"):
        ref_lt = os.path.join(d, "reference_lt.wav")
        synthesize(tts, args.text_lt, ref_lt, args.lt_language,
                   os.path.join(d, "output_lt.wav"))
        if args.ascii_variant:
            synthesize(tts, strip_diacritics(args.text_lt), ref_lt, args.lt_language,
                       os.path.join(d, "output_lt_ascii.wav"))


if __name__ == "__main__":
    main()
