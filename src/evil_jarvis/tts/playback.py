"""Plays audio through one output stream that stays open for the whole run.

Like the microphone, the stream is never stopped and restarted between
replies: on macOS that can deadlock inside CoreAudio. Between replies it plays
silence. Waiting happens in short slices so Ctrl-C always works.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from evil_jarvis.wav import read_wav

PLAYBACK_RATE = 22_050  # Piper's rate, so voices are never resampled
# Frames per callback. Left to PortAudio, macOS asks for ~15 frames at a time
# (1,500 Python callbacks a second), and any stall in Python is heard as a click.
BLOCK_FRAMES = 1024  # ~46 ms at 22,050 Hz


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target or len(samples) == 0:
        return samples
    positions = np.arange(round(len(samples) * target / rate)) * (rate / target)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


class AudioPlayer:
    WAIT_SLICE_SECONDS = 0.1

    def __init__(self, device: int | str | None = None, rate: int = PLAYBACK_RATE):
        self.device = device
        self.rate = rate
        self._stream = None
        self._lock = threading.Lock()
        self._buffer: np.ndarray | None = None
        self._position = 0
        self._finished = threading.Event()

    def _on_audio(self, outdata, frames, time_info, status) -> None:
        # Runs on PortAudio's audio thread: keep it trivial.
        with self._lock:
            chunk = np.zeros(0, dtype=np.float32)
            if self._buffer is not None:
                chunk = self._buffer[self._position : self._position + frames]
                self._position += len(chunk)
                if self._position >= len(self._buffer):
                    self._buffer = None
                    self._finished.set()
        outdata[: len(chunk), 0] = chunk
        outdata[len(chunk) :, 0] = 0.0

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=BLOCK_FRAMES,
            latency="high",
            callback=self._on_audio,
        )
        self._stream.start()

    def play(self, samples: np.ndarray, rate: int) -> None:
        """Play and return once the audio has left the speaker."""
        samples = resample(np.asarray(samples, dtype=np.float32), rate, self.rate)
        if len(samples) == 0:
            return
        self._ensure_stream()
        with self._lock:
            self._finished.clear()
            self._position = 0
            self._buffer = samples
        try:
            while not self._finished.wait(self.WAIT_SLICE_SECONDS):
                pass
            # The last block is still in the device buffer; let it finish so the
            # microphone does not hear the end of the reply.
            time.sleep(float(self._stream.latency))
        finally:
            with self._lock:
                self._buffer = None

    def play_wav(self, path) -> None:
        samples, rate = read_wav(path)
        self.play(samples, rate)

    def close(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            stream.close()
