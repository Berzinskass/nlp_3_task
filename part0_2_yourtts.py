# Parts 0-2: load YourTTS, check its languages and synthesize two sentences.
# Before running, put your recordings reference_en.wav and reference_lt.wav
# in this folder.

import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts").to(device)

# Part 0: supported languages
print("Supported languages:", tts.languages)
print("Is Lithuanian supported?", "lt" in tts.languages)

# Part 1: English sentence in your voice
tts.tts_to_file(
    text="This is a test of the speech synthesis system.",
    speaker_wav="reference_en.wav",
    language="en",
    file_path="output_en.wav",
)

# Part 2: Lithuanian sentence. Lithuanian is not supported,
# so we use the English language setting.
tts.tts_to_file(
    text="Tai yra šnekos sintezės sistemos testas.",
    speaker_wav="reference_lt.wav",
    language="en",
    file_path="output_lt.wav",
)

print("Done: output_en.wav and output_lt.wav")
