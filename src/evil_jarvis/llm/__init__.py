"""Text to text: writes the character's reply to what the visitor said.

llama.py   LlamaResponder: persona prompt, style examples, reply checks and
           retries and short-term memory
"""

from __future__ import annotations

import os
from typing import Protocol

from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.paths import MODELS_DIR

DEFAULT_MODEL = MODELS_DIR / "Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"
DEFAULT_MODEL_REPO = "speakleash/Bielik-4.5B-v3.0-Instruct-GGUF"


class Responder(Protocol):
    last_reply_info: dict | None

    def reply(self, text: str, language: str | None = None) -> str: ...


def build_responder(model_path: str, persona, memory_seconds: float, telemetry):

    return LlamaResponder(
        model_path,
        persona=persona,
        memory_seconds=memory_seconds,
        telemetry=telemetry,
    )


def gpu_offload_supported() -> bool | None:
    """Whether this llama.cpp build can use a GPU (Metal or CUDA); None if not installed."""

    try:
        import llama_cpp
    except ImportError:
        return None
    # The first call starts the GPU backend, which prints a page of start-up logs.
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        return bool(llama_cpp.llama_supports_gpu_offload())
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
