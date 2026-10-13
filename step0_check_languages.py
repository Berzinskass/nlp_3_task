"""Step 0: load YourTTS and list the languages it supports.

    python step0_check_languages.py
"""

import argparse
import json
import os

from tts_lab.model import YOURTTS_MODEL, load_yourtts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    tts = load_yourtts(args.device)
    languages = list(tts.languages or [])
    speakers = list(tts.speakers or [])
    has_lt = any(lang.lower().split("-")[0] in ("lt", "lit") for lang in languages)

    print("Supported languages:", languages)
    print(f"Built-in speakers: {len(speakers)}")
    print("Lithuanian supported:", "YES" if has_lt else "NO")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "supported_languages.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": YOURTTS_MODEL, "languages": languages,
                   "lithuanian_supported": has_lt,
                   "num_builtin_speakers": len(speakers)}, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
