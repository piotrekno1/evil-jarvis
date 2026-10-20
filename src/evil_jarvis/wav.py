"""16-bit mono WAV files: microphone clips, recognizer input and voice output."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import BinaryIO

import numpy as np

CHANNELS = 1  # mono
SAMPLE_BYTES = 2  # 16-bit samples
SAMPLE_TYPE = "<i2"  # little-endian 16-bit signed integer, as WAV stores it
SPEECH_SAMPLE_RATE = 16_000  # what Whisper expects

# 16-bit range is -32768..32767. Reading divides by 32768 so the lowest sample
# maps to exactly -1.0. Writing multiplies by 32767 so +1.0 does not overflow.
READ_SCALE = -np.iinfo(np.int16).min
WRITE_SCALE = np.iinfo(np.int16).max


def read_wav(path: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    """Float32 samples in [-1, 1] and the sample rate."""
    source = str(path) if isinstance(path, (str, Path)) else path
    with wave.open(source, "rb") as handle:
        if handle.getnchannels() != CHANNELS or handle.getsampwidth() != SAMPLE_BYTES:
            raise ValueError(f"Expected 16-bit mono WAV: {path}")
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=SAMPLE_TYPE).astype(np.float32) / READ_SCALE, rate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SPEECH_SAMPLE_RATE) -> None:
    """Write float samples in [-1, 1]; anything outside is clipped."""
    pcm = (np.clip(samples, -1.0, 1.0) * WRITE_SCALE).astype(SAMPLE_TYPE)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_BYTES)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
