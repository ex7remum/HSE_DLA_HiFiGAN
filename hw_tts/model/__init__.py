from hw_tts.model.FastSpeech2 import FastSpeech2
from hw_tts.model.waveglow import WaveGlowInfer

from .glow import * # Ensures that all the modules have been loaded in their new locations *first*.
from . import glow  # imports WrapperPackage/packageA
import sys
sys.modules['glow'] = glow  # creates a packageA entry in sys.modules

__all__ = [
    "FastSpeech2",
    "WaveGlowInfer"
]
