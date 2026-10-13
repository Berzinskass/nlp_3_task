"""Loading the pretrained YourTTS model."""

import os

YOURTTS_MODEL = "tts_models/multilingual/multi-dataset/your_tts"


def get_device(device=None):
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _allow_full_checkpoint_loading():
    """PyTorch >= 2.6 defaults torch.load(weights_only=True), which breaks
    loading the Coqui checkpoints with TTS 0.22. The checkpoints come from the
    official Coqui model release, so we restore the old default."""
    import torch

    major, minor = (int(p) for p in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 6) or getattr(torch.load, "_tts_lab_patched", False):
        return
    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load._tts_lab_patched = True
    torch.load = load


def load_yourtts(device=None):
    """Return a TTS object with the pretrained YourTTS model on `device`."""
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    _allow_full_checkpoint_loading()
    from TTS.api import TTS

    device = get_device(device)
    print(f"Loading {YOURTTS_MODEL} on {device} ...")
    return TTS(model_name=YOURTTS_MODEL).to(device)
