"""Whisper on the Apple GPU (Apple Silicon Macs)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from evil_jarvis.stt.guards import is_runaway_transcript
from evil_jarvis.wav import SPEECH_SAMPLE_RATE, read_wav

logger = logging.getLogger(__name__)


class MlxWhisperTranscriber:
    """Whisper large-v3-turbo on the Apple GPU via mlx-whisper.

    No topic hint is passed: in the 2026-10-17 benchmark its words leaked into
    transcripts of short clips (e.g. "duchy, duchy, duchy...").
    """

    SAMPLE_RATE = SPEECH_SAMPLE_RATE

    def __init__(self, model_path: str, language: str | None = None):
        path = Path(model_path).expanduser()
        if not (path / "weights.safetensors").is_file() and not (path / "weights.npz").is_file():
            raise ValueError(
                f"MLX Whisper model not found: {path}. Get it with: evil-jarvis --download"
            )
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to use MLX Whisper") from exc
        self._mlx = mlx_whisper
        self._path = str(path)
        self._language = None if language in (None, "auto") else language
        self.last_language: str | None = self._language
        self.last_discarded = False

    def warm_up(self) -> None:
        """Load the weights now so the first visitor does not wait ~3 s."""

        logger.info("Warming up speech recognition...")
        self._mlx.transcribe(
            np.zeros(self.SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=self._path,
            language=self._language or "pl",
        )

    def transcribe(self, wav_path: str) -> str:
        samples, rate = read_wav(wav_path)
        if rate != self.SAMPLE_RATE:
            raise ValueError(f"Expected {self.SAMPLE_RATE} Hz audio, got {rate} Hz")
        result = self._mlx.transcribe(
            samples,
            path_or_hf_repo=self._path,
            language=self._language,
            condition_on_previous_text=False,
        )
        text = str(result.get("text", "")).strip()
        self.last_language = result.get("language") or self._language
        self.last_discarded = is_runaway_transcript(text, len(samples) / rate, self.last_language)
        if self.last_discarded:
            logger.info("Discarding a runaway transcript: %r", text[:80])
            return ""
        return text
