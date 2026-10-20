"""Sparse, persona-owned idle audio, played only while waiting for a visitor."""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np

from evil_jarvis.persona import IdleSettings, Persona
from evil_jarvis.tts.effects import piper_effect_chain
from evil_jarvis.tts.playback import PLAYBACK_RATE, AudioPlayer, resample
from evil_jarvis.tts.vocalization import trim_quiet_edges
from evil_jarvis.wav import read_wav, write_wav

logger = logging.getLogger(__name__)
ROOM_SETTLE_SECONDS = 0.15


def prepare_idle_clip(audio: bytes, effect: str, gain: float) -> np.ndarray:
    """Apply the character's voice processing, then lower idle playback volume."""
    samples, rate = read_wav(BytesIO(audio))
    call_rate = round(rate * 0.86) if effect in ("crow", "crow-harsh") else rate
    samples = trim_quiet_edges(resample(samples, call_rate, PLAYBACK_RATE), PLAYBACK_RATE)
    chain = piper_effect_chain(effect)
    if not shutil.which("ffmpeg"):
        if effect != "clean":
            raise RuntimeError("ffmpeg is required for idle sound effects")
        return samples * gain
    with tempfile.TemporaryDirectory(prefix="evil-jarvis-idle-") as directory:
        raw = Path(directory) / "raw.wav"
        target = Path(directory) / "processed.wav"
        write_wav(raw, samples, PLAYBACK_RATE)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(raw),
                    "-af",
                    chain,
                    str(target),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Could not prepare an idle sound with ffmpeg") from exc
        processed, _ = read_wav(target)
    return processed * gain


class IdleSounds:
    """A main-thread playlist; its action must run while mic detection is paused."""

    def __init__(
        self,
        settings: IdleSettings,
        clips: tuple[np.ndarray, ...],
        player: AudioPlayer,
        *,
        clock=time.monotonic,
        rng=None,
        sleep=time.sleep,
    ):
        if not clips or len(clips) != len(settings.sounds):
            raise ValueError("Idle sound settings and clips must match and be nonempty")
        self.settings, self.clips, self.player = settings, clips, player
        self.clock, self._rng, self._sleep = clock, rng or random.Random(), sleep
        self._last: int | None = None
        self._next_at = 0.0
        self.reset()

    def reset(self) -> None:
        """Start a fresh quiet period when listening resumes after activity."""
        self._next_at = self.clock() + self.settings.after_seconds

    def next_action(self):
        """Return one due playback action, otherwise return immediately."""
        if self.clock() < self._next_at:
            return None
        choices = [i for i in range(len(self.clips)) if i != self._last]
        index = self._rng.choice(choices or [0])
        # Reserve the action so polling cannot enqueue overlapping sounds.
        self._next_at = float("inf")

        def play():
            logger.info("Idle sound: %s (microphone paused)", self.settings.sounds[index])
            try:
                self.player.play(self.clips[index], PLAYBACK_RATE)
                self._sleep(ROOM_SETTLE_SECONDS)
            finally:
                self._last = index
                self._next_at = self.clock() + self._rng.uniform(
                    self.settings.min_interval_seconds, self.settings.max_interval_seconds
                )

        return play


def build_idle_sounds(persona: Persona, player: AudioPlayer, effect: str) -> IdleSounds | None:
    if not persona.idle_audio:
        return None
    clips = tuple(
        prepare_idle_clip(audio, effect, persona.idle.gain) for audio in persona.idle_audio
    )
    return IdleSounds(persona.idle, clips, player)
