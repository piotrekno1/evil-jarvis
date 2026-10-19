"""Speech to text: turns a recorded utterance (WAV) into words.

Two engines run the same Whisper large-v3-turbo model:
  mlx             Apple GPU, on Apple Silicon Macs
  faster-whisper  NVIDIA GPU (CUDA) on Linux, or CPU anywhere (slow; uses `small`)
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Protocol

from evil_jarvis.paths import MODELS_DIR
from evil_jarvis.stt.faster_whisper import FasterWhisperTranscriber, cuda_available
from evil_jarvis.stt.mlx_whisper import MlxWhisperTranscriber

ENGINES = ("mlx", "faster-whisper")


class Transcriber(Protocol):
    last_language: str | None
    last_discarded: bool

    def transcribe(self, wav_path: str) -> str: ...


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def default_engine() -> str:
    return "mlx" if apple_silicon() else "faster-whisper"


def faster_whisper_device() -> str:

    return "cuda" if cuda_available() else "cpu"


# (model directory, Hugging Face repo) per engine; CPU gets a smaller model.
MODELS = {
    "mlx": (MODELS_DIR / "whisper-large-v3-turbo-mlx", "mlx-community/whisper-large-v3-turbo"),
    "faster-whisper": (
        MODELS_DIR / "whisper-large-v3-turbo-ct2",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    ),
    "faster-whisper-cpu": (MODELS_DIR / "whisper-small-ct2", "Systran/faster-whisper-small"),
}


def model_key(engine: str, device: str | None = None) -> str:
    if engine == "faster-whisper" and (device or faster_whisper_device()) == "cpu":
        return "faster-whisper-cpu"
    return engine


def default_model(engine: str, device: str | None = None) -> Path:
    return MODELS[model_key(engine, device)][0]


def model_present(engine: str, path: Path) -> bool:
    if engine == "mlx":
        return any((path / name).is_file() for name in ("weights.safetensors", "weights.npz"))
    return (path / "model.bin").is_file()


def build_transcriber(
    engine: str, model: str | Path, language: str | None, topic_hint: str | None = None
):
    if engine == "mlx":
        return MlxWhisperTranscriber(str(model), language)

    return FasterWhisperTranscriber(str(model), language, topic_hint=topic_hint)


def telemetry_model(engine: str, model: str | Path) -> tuple[str, str]:
    """WildEdge model id and format."""
    return Path(model).name, "mlx" if engine == "mlx" else "ctranslate2"
