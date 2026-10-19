"""Whisper via faster-whisper (CTranslate2): NVIDIA GPU on Linux, or CPU."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from evil_jarvis.stt.guards import is_runaway_transcript
from evil_jarvis.wav import SPEECH_SAMPLE_RATE, read_wav

logger = logging.getLogger(__name__)


def cuda_available() -> bool:
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # a broken CUDA install means CPU
        return False


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_path: str,
        language: str | None = None,
        topic_hint: str | None = None,
        device: str | None = None,
    ):
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError(f"faster-whisper model directory does not exist: {path}")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to use faster-whisper") from exc
        self.device = device or ("cuda" if cuda_available() else "cpu")
        compute_type = "float16" if self.device == "cuda" else "int8"
        logger.info("Loading faster-whisper %s on %s (%s)...", path.name, self.device, compute_type)
        self._model = WhisperModel(str(path), device=self.device, compute_type=compute_type)
        self._language = None if language in (None, "auto") else language
        self.last_language: str | None = self._language
        self.last_discarded = False
        # Only used with a fixed language; the persona's hint is in that language.
        self._initial_prompt = topic_hint if self._language else None

    def warm_up(self) -> None:
        """Run once on silence so the first visitor does not wait for CUDA kernels."""

        logger.info("Warming up speech recognition...")
        one_second_of_silence = np.zeros(SPEECH_SAMPLE_RATE, dtype=np.float32)
        segments, _ = self._model.transcribe(one_second_of_silence, language=self._language or "pl")
        list(segments)

    def transcribe(self, wav_path: str) -> str:
        segments, info = self._model.transcribe(
            wav_path,
            language=self._language,
            beam_size=3,
            initial_prompt=self._initial_prompt,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        self.last_language = info.language
        samples, rate = read_wav(wav_path)
        self.last_discarded = is_runaway_transcript(text, len(samples) / rate, self.last_language)
        if self.last_discarded:
            logger.info("Discarding a runaway transcript: %r", text[:80])
            return ""
        return text
