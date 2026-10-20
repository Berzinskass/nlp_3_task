

import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts").to(device)

print("Supported languages:", tts.languages)
print("Is Lithuanian supported?", "lt" in tts.languages)

tts.tts_to_file(
    text="This is a test of the speech synthesis system.",
    speaker_wav="reference_en.wav",
    language="en",
    file_path="output_en.wav",
)


tts.tts_to_file(
    text="Tai yra šnekos sintezės sistemos testas.",
    speaker_wav="reference_lt.wav",
    language="en",
    file_path="output_lt.wav",
)

