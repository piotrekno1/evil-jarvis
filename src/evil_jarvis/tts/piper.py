"""Piper: local neural voices (Polish, English and about 40 other languages)."""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from evil_jarvis.paths import MODELS_DIR
from evil_jarvis.tts.effects import piper_effect_chain
from evil_jarvis.tts.playback import AudioPlayer
from evil_jarvis.tts.prosody import (
    PROSODY_STYLES,
    fade_edges,
    plan_maniacal_delivery,
    split_phrases,
)
from evil_jarvis.wav import write_wav

logger = logging.getLogger(__name__)
DEFAULT_VOICE = "pl_PL-bass-high"


def voice_path(voice: str) -> Path:
    """A voice name (looked up in models/piper) or a path to an .onnx file."""
    return (
        Path(voice).expanduser()
        if voice.endswith(".onnx")
        else MODELS_DIR / "piper" / f"{voice}.onnx"
    )


def voice_files(path: Path) -> list[tuple[str, Path]] | None:
    """Download URLs from rhasspy/piper-voices for a voice named like en_US-ryan-high, else None."""
    name = path.name.removesuffix(".onnx")
    try:
        locale, speaker, quality = name.split("-", 2)
    except ValueError:
        return None
    base = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{locale.split('_')[0]}/{locale}/{speaker}/{quality}/{name}"
    )
    return [(f"{base}.{ext}", path.parent / f"{name}.{ext}") for ext in ("onnx", "onnx.json")]


class PiperSpeaker:
    """Local neural TTS (Piper, many languages) with erratic rhythm and optional effects."""

    def __init__(
        self,
        voice_path: str,
        *,
        effect: str = "clean",
        speed: float = 1.0,
        prosody: str = "maniacal",
        seed: int | None = None,
        player: AudioPlayer | None = None,
    ):
        path = Path(voice_path).expanduser()
        if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
            raise ValueError(
                f"Piper voice not found: {path} (needs the .onnx and .onnx.json files). "
                "Get it with: evil-jarvis --voice NAME --download"
            )
        piper_effect_chain(effect)  # validate early
        if prosody not in PROSODY_STYLES:
            raise ValueError(
                f"Unknown prosody {prosody!r}; choose from {', '.join(PROSODY_STYLES)}"
            )
        if effect != "clean" and not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for Piper voice effects")
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:
            raise RuntimeError("Install the 'voice' extra to use Piper") from exc
        logger.info("Loading Piper voice %s (%s prosody, %s effect)...", path.stem, prosody, effect)
        self._voice = PiperVoice.load(str(path))
        self._config_type = SynthesisConfig
        self.speed = speed
        self.effect = effect
        self.prosody = prosody
        self._rng = random.Random(seed)
        self._player = player

    def _synthesize(self, text: str, length_scale: float, erratic: bool):

        config = self._config_type(
            length_scale=length_scale,
            # A little more variation in stress and timing than the defaults (0.667/0.8).
            noise_scale=0.8 if erratic else None,
            noise_w_scale=1.0 if erratic else None,
        )
        chunks = [
            chunk.audio_float_array for chunk in self._voice.synthesize(text, syn_config=config)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def render(self, text: str, path: Path) -> None:
        """Write the fully processed reply (prosody + effect + loudness) to a WAV."""

        if self.prosody == "plain":
            audio = self._synthesize(text, 1.0 / self.speed, erratic=False)
        else:
            phrases = split_phrases(text) or [text]
            pieces = []
            for phrase, delivery in zip(
                phrases, plan_maniacal_delivery(phrases, self._rng), strict=True
            ):
                raw = self._synthesize(phrase, 1.0 / (delivery.speed * self.speed), erratic=True)
                gap = int(self._voice.config.sample_rate * delivery.gap_ms / 1000)
                pieces.append(np.zeros(gap, dtype=np.float32))
                pieces.append(fade_edges(raw) * delivery.gain)
                logger.debug("phrase %r: %s", phrase, delivery)
            audio = np.concatenate(pieces)

        with tempfile.TemporaryDirectory(prefix="evil-jarvis-raw-") as directory:
            raw_path = Path(directory) / "raw.wav"
            write_wav(raw_path, audio, self._voice.config.sample_rate)
            if not shutil.which("ffmpeg"):
                shutil.copyfile(raw_path, path)
                return
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(raw_path),
                        "-af",
                        piper_effect_chain(self.effect),
                        str(path),
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError:
                logger.warning("Voice effect failed; playing the unprocessed voice")
                shutil.copyfile(raw_path, path)

    def speak(self, text: str, language: str | None = None) -> None:
        if self._player is None:
            self._player = AudioPlayer()
        with tempfile.TemporaryDirectory(prefix="evil-jarvis-piper-") as directory:
            path = Path(directory) / "voice.wav"
            self.render(text, path)
            self._player.play_wav(path)
