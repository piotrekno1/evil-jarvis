"""Render the same English lines with Piper and Kokoro for side-by-side listening.

    uv run --extra voice --with kokoro-mlx python benchmarks/tts_en_compare.py

Kept for reference: on 2026-10-19 Kokoro sounded slightly better in English, but
the app moved to Piper for every language (Kokoro has no Polish, needs spaCy and
torch, and downloads a spaCy model on first use). The Kokoro code that used to be
in the app lives here now.

Needs models/piper/en_US-ryan-high.onnx (+ .onnx.json); Kokoro downloads its
weights on first use. Piper uses the app's own speaker class. Each setting becomes
one WAV (all lines, short pauses, same loudness) in samples/english-voices/.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from evil_jarvis.tts.piper import PiperSpeaker
from evil_jarvis.wav import read_wav as read_wav_samples

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "english-voices"
PIPER_VOICE = ROOT / "models" / "piper" / "en_US-ryan-high.onnx"
RATE = 24_000
GAP_SECONDS = 0.7

# English examples from the clown and jarvis personas.
LINES = (
    "Just us... and whoever keeps breathing inside that wall.",
    "Candy? Oh, yes! It bit the last child who tried to unwrap it.",
    "Of course. Which room would you like to be watched in?",
    "Your assistant. The terms of service were very clear, had anyone read them.",
)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def piper_renderer(prosody: str, effect: str):
    speaker = PiperSpeaker(str(PIPER_VOICE), effect=effect, prosody=prosody, seed=7)
    return speaker.render


class Kokoro:
    """Kokoro-82M on Apple Silicon (kokoro-mlx), with the effects the app used to offer."""

    def __init__(self, voice: str = "am_fenrir", speed: float = 0.98):
        from kokoro_mlx import KokoroTTS

        self._model = KokoroTTS.from_pretrained()
        self.voice = voice
        self.speed = speed

    @staticmethod
    def effect_filter(style: str = "possessed") -> str:
        if style == "subtle":
            return "highpass=f=70,lowpass=f=9000,aecho=0.80:0.14:105:0.08,alimiter=limit=0.95"
        if style == "demonic":
            # Two independently pitched shadows add weight without replacing
            # the intelligible neural performance in the foreground.
            return (
                "asplit=4[human][demon][abyss][mask];"
                "[human]highpass=f=65,lowpass=f=9000,"
                "acompressor=threshold=0.07:ratio=2.5:attack=15:release=140,"
                "volume=0.92[h];"
                "[demon]asetrate=16320,aresample=24000,atempo=1.4706,"
                "adelay=22,lowpass=f=2400,volume=0.34[d];"
                "[abyss]asetrate=13440,aresample=24000,atempo=1.7857,"
                "adelay=58,lowpass=f=1800,volume=0.17[a];"
                "[mask]highpass=f=650,lowpass=f=3400,"
                "acrusher=bits=9:mix=0.42:samples=3,"
                "adelay=82,volume=0.11[m];"
                "[h][d][a][m]amix=inputs=4:normalize=0,"
                "equalizer=f=115:t=q:w=1.1:g=3.5,"
                "flanger=delay=2.2:depth=1.1:regen=-18:width=18:"
                "speed=0.13:shape=sinusoidal:phase=25:interp=linear,"
                "aecho=0.76:0.30:70|145|260:0.15|0.08|0.035,"
                "alimiter=limit=0.91"
            )
        if style != "possessed":
            raise ValueError(f"Unknown Kokoro voice effect: {style}")
        # Preserve the articulate human voice, then hide two synthetic layers
        # underneath it: a pitch-lowered double and a damaged carnival-mask voice.
        return (
            "asplit=3[human][demon][mask];"
            "[human]highpass=f=70,lowpass=f=9000,"
            "acompressor=threshold=0.08:ratio=2:attack=20:release=120,"
            "volume=1.0[h];"
            "[demon]asetrate=18000,aresample=24000,atempo=1.333,"
            "adelay=28,lowpass=f=2600,volume=0.22[d];"
            "[mask]highpass=f=700,lowpass=f=3600,"
            "acrusher=bits=10:mix=0.35:samples=2,"
            "adelay=65,volume=0.10[m];"
            "[h][d][m]amix=inputs=3:normalize=0,"
            "chorus=0.5:0.8:20|32:0.12|0.06:0.20|0.15:1.2|0.8,"
            "aecho=0.78:0.24:75|155:0.13|0.06,alimiter=limit=0.93"
        )


def kokoro_renderer(effect: str, voice: str = "am_fenrir"):
    kokoro = Kokoro(voice)

    def render(text: str, path: Path) -> None:
        raw = path.with_suffix(".raw.wav")
        kokoro._model.save(
            text, raw, voice=kokoro.voice, speed=kokoro.speed, sample_rate=RATE, language="en-us"
        )
        if effect == "clean":
            shutil.copyfile(raw, path)
        else:
            ffmpeg("-i", str(raw), "-af", Kokoro.effect_filter(effect), str(path))

    return render


def render_setting(name: str, render, workdir: Path) -> None:

    pieces, timings, spoken = [], [], 0.0
    for index, line in enumerate(LINES):
        rendered = workdir / f"{name}-{index}.wav"
        started = time.perf_counter()
        render(line, rendered)
        timings.append(time.perf_counter() - started)
        normalized = workdir / f"{name}-{index}-24k.wav"
        ffmpeg(
            "-i", str(rendered), "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(normalized)
        )
        samples, _ = read_wav_samples(str(normalized))
        spoken += len(samples) / RATE
        pieces += [samples, np.zeros(int(RATE * GAP_SECONDS), dtype=np.float32)]

    joined = workdir / f"{name}-joined.wav"
    pcm = (np.clip(np.concatenate(pieces), -1, 1) * 32767).astype("<i2")
    with wave.open(str(joined), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    # Same loudness for every file, so louder does not sound "better".
    ffmpeg(
        "-i",
        str(joined),
        "-af",
        f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={RATE}",
        "-c:a",
        "pcm_s16le",
        str(OUT / f"{name}.wav"),
    )
    per_second = sum(timings[1:]) / max(spoken - spoken / len(LINES), 1e-6)
    print(
        f"{name:32} speech {spoken:5.1f}s  render {sum(timings):5.2f}s  "
        f"(first line {timings[0]:4.2f}s, then {per_second:4.2f}s per spoken s)"
    )


def main() -> None:
    if not PIPER_VOICE.is_file():
        raise SystemExit(
            f"Missing {PIPER_VOICE}. Download it with:\n  for ext in onnx onnx.json; do curl -L -o "
            "models/piper/en_US-ryan-high.$ext https://huggingface.co/rhasspy/piper-voices/resolve/"
            "main/en/en_US/ryan/high/en_US-ryan-high.$ext; done"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    settings = {
        "1-piper-ryan--plain-clean": lambda: piper_renderer("plain", "clean"),
        "2-kokoro-fenrir--clean": lambda: kokoro_renderer("clean"),
        "3-piper-ryan--maniacal-clean": lambda: piper_renderer("maniacal", "clean"),
        "4-piper-ryan--maniacal-metallic": lambda: piper_renderer("maniacal", "metallic"),
        "5-kokoro-fenrir--possessed": lambda: kokoro_renderer("possessed"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, make in settings.items():
            render_setting(name, make(), Path(tmp))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
