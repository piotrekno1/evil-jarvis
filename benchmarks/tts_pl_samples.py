"""Render Polish clown lines with several base voices and metallic effects.

    uv run --extra voice python benchmarks/tts_pl_samples.py

Writes WAV files to samples/voices/ for side-by-side listening.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import wave
from pathlib import Path

from evil_jarvis.tts.effects import PIPER_EFFECTS as EFFECTS
from evil_jarvis.tts.effects import piper_effect_chain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "voices"
PIPER = ROOT / "models" / "piper"

TEXT = (
    "Tyle, ile lat ma strach. Ale nie pytaj mnie o PESEL. "
    "Dziękuję, ale to tylko maska. Za nią czai się coś innego."
)

RATE = 22050


def render_say(path: Path) -> None:
    subprocess.run(
        [
            "say",
            "-v",
            "Zosia",
            "-r",
            "160",
            "-o",
            str(path),
            "--file-format=WAVE",
            f"--data-format=LEI16@{RATE}",
            TEXT,
        ],
        check=True,
    )


def render_piper(voice: str):
    from piper import PiperVoice

    model = PiperVoice.load(str(PIPER / f"{voice}.onnx"))

    def render(path: Path) -> None:
        with wave.open(str(path), "wb") as handle:
            model.synthesize_wav(TEXT, handle)

    return render


def apply_effect(source: Path, target: Path, effect: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            piper_effect_chain(effect),  # same chain the clown uses
            str(target),
        ],
        check=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    voices = {
        "zosia": render_say,
        "piper-darkman": render_piper("pl_PL-darkman-medium"),
        "piper-bass": render_piper("pl_PL-bass-high"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, render in voices.items():
            raw = Path(tmp) / f"{name}.wav"
            started = time.perf_counter()
            render(raw)
            synth = time.perf_counter() - started
            with wave.open(str(raw)) as handle:
                seconds = handle.getnframes() / handle.getframerate()
            for effect in EFFECTS:
                started = time.perf_counter()
                apply_effect(raw, OUT / f"{name}--{effect}.wav", effect)
                fx = time.perf_counter() - started
                print(
                    f"{name:14} {effect:15} speech {seconds:4.1f}s  "
                    f"synth {synth:4.2f}s  effect {fx:4.2f}s"
                )
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
