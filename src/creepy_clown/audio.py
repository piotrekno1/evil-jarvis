from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import queue
import threading
import wave

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    calibration_seconds: float = 2.0
    # Audio kept from before the first loud frame. Soft onsets ("s", "h", "f")
    # sit below the threshold, so this must cover them in full.
    preroll_ms: int = 500
    speech_start_ms: int = 240
    silence_ms: int = 900
    min_speech_ms: int = 300
    max_utterance_seconds: float = 10.0
    threshold_multiplier: float = 3.0
    minimum_threshold: float = 0.012

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


class VoiceActivityRecorder:
    """Energy-based VAD suited to the prototype's expected quiet room.

    The microphone stream is opened once and kept running. Stopping and
    restarting a PortAudio stream on macOS can deadlock inside CoreAudio
    (seen 2026-10-17: main thread stuck in FinishStoppingStream, Ctrl-C
    ignored), so between utterances incoming audio is dropped instead.
    Frames arrive through a callback and a queue, so the main thread never
    blocks in native code and Ctrl-C always works.
    """

    FRAME_TIMEOUT_SECONDS = 0.25
    STALL_WARNING_SECONDS = 3.0

    def __init__(self, config: DetectorConfig, input_device: int | str | None = None):
        self.config = config
        self.input_device = input_device
        self._threshold: float | None = None
        self._stream = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._listening = threading.Event()

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        values = np.asarray(frame, dtype=np.float32).reshape(-1)
        return float(math.sqrt(float(np.mean(values * values)) + 1e-12))

    def _on_audio(self, indata, frames, time_info, status) -> None:
        # Runs on PortAudio's audio thread: keep it trivial.
        if status:
            LOG.debug("Audio status: %s", status)
        if self._listening.is_set():
            self._frames.put(np.array(indata[:, 0], dtype=np.float32))

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the project dependencies to use the microphone") from exc
        cfg = self.config
        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=cfg.frame_samples,
            device=self.input_device,
            callback=self._on_audio,
        )
        self._stream.start()

    def _drain(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _next_frame(self) -> np.ndarray:
        """Wait for the next frame in short slices so Ctrl-C stays responsive."""
        waited = 0.0
        warned = False
        while True:
            try:
                return self._frames.get(timeout=self.FRAME_TIMEOUT_SECONDS)
            except queue.Empty:
                waited += self.FRAME_TIMEOUT_SECONDS
                if waited >= self.STALL_WARNING_SECONDS and not warned:
                    LOG.warning("No audio from the microphone for %.0f s; still waiting", waited)
                    warned = True

    def close(self) -> None:
        self._listening.clear()
        if self._stream is not None:
            stream, self._stream = self._stream, None
            stream.close()

    def _calibrate(self) -> None:
        frame_count = max(
            1,
            round(self.config.calibration_seconds * 1000 / self.config.frame_ms),
        )
        LOG.info("Calibrating room noise for %.1f seconds...", self.config.calibration_seconds)
        levels = [self.rms(self._next_frame()) for _ in range(frame_count)]
        baseline = float(np.percentile(levels, 80))
        self._threshold = max(
            self.config.minimum_threshold,
            baseline * self.config.threshold_multiplier,
        )
        LOG.info("Listening threshold: %.4f (ambient %.4f)", self._threshold, baseline)

    def capture(self, on_listening=None) -> np.ndarray | None:
        """Wait for and return one utterance, or None for a discarded noise.

        on_listening is called once the stream is running and calibrated, i.e.
        the moment speech will actually be heard. Audio that arrived while not
        capturing (for example the clown's own voice) is discarded.
        """
        cfg = self.config
        start_frames = max(1, cfg.speech_start_ms // cfg.frame_ms)
        silence_frames = max(1, cfg.silence_ms // cfg.frame_ms)
        minimum_frames = max(1, cfg.min_speech_ms // cfg.frame_ms)
        maximum_frames = max(1, round(cfg.max_utterance_seconds * 1000 / cfg.frame_ms))
        # The onset run itself plus the requested lead-in before it.
        preroll = deque(maxlen=start_frames + max(1, cfg.preroll_ms // cfg.frame_ms))

        self._ensure_stream()
        self._drain()
        self._listening.set()
        try:
            if self._threshold is None:
                self._calibrate()
            assert self._threshold is not None

            LOG.info("Listening...")
            if on_listening is not None:
                on_listening()
            consecutive_loud = 0
            while consecutive_loud < start_frames:
                frame = self._next_frame()
                preroll.append(frame)
                level = self.rms(frame)
                LOG.debug("level=%.4f threshold=%.4f", level, self._threshold)
                consecutive_loud = consecutive_loud + 1 if level >= self._threshold else 0

            LOG.info("Voice detected; recording...")
            frames = list(preroll)
            silent_count = 0
            loud_count = consecutive_loud

            while len(frames) < maximum_frames:
                frame = self._next_frame()
                frames.append(frame)
                if self.rms(frame) >= self._threshold:
                    loud_count += 1
                    silent_count = 0
                else:
                    silent_count += 1
                    if silent_count >= silence_frames:
                        break
        finally:
            # Stop collecting; the stream keeps running but frames are dropped.
            self._listening.clear()

        if loud_count < minimum_frames:
            LOG.info("Discarding a very short sound")
            return None
        return np.concatenate(frames)


def write_wav(path: str, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())

