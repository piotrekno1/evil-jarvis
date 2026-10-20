"""Piper: local neural voices (Polish, English and about 40 other languages)."""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
from importlib import resources
from pathlib import Path

import numpy as np

from evil_jarvis.paths import MODELS_DIR
from evil_jarvis.tts.effects import piper_effect_chain
from evil_jarvis.tts.playback import AudioPlayer, resample
from evil_jarvis.tts.pronunciation import for_piper
from evil_jarvis.tts.prosody import (
    PROSODY_STYLES,
    fade_edges,
    plan_maniacal_delivery,
    split_phrases,
)
from evil_jarvis.wav import read_wav, write_wav

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
        vocalization: str | None = None,
        vocalization_probability: float = 0.65,
        vocalization_position: str = "random",
    ):
        if vocalization not in (None, "crow"):
            raise ValueError("Unknown vocalization; choose crow")
        if not 0 <= vocalization_probability <= 1:
            raise ValueError("vocalization_probability must be between 0 and 1")
        if vocalization_position not in ("random", "start", "end", "between"):
            raise ValueError("vocalization_position must be random, start, end or between")
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
        self.vocalization_probability = vocalization_probability
        self.vocalization_position = vocalization_position
        self._call = None
        if vocalization:
            asset = resources.files("evil_jarvis.tts") / "assets" / "crow.wav"
            with resources.as_file(asset) as asset_path:
                samples, rate = read_wav(asset_path)
            self._call = (
                fade_edges(
                    resample(samples, rate, self._voice.config.sample_rate),
                    rate=self._voice.config.sample_rate,
                )
                * 0.65
            )

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

    def _render_speech(self, text: str):
        if self.prosody == "plain":
            return self._synthesize(text, 1.0 / self.speed, erratic=False)
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
        return np.concatenate(pieces)

    def _render_with_call(self, text: str):
        """Insert at most one recorded call, never splitting a spoken sentence."""
        if (
            self._call is None
            or not text.strip()
            or self._rng.random() >= self.vocalization_probability
        ):
            return self._render_speech(text)
        phrases = split_phrases(text)
        position = self.vocalization_position
        if position == "random":
            positions = ["start", "end"]
            if len(phrases) > 1:
                positions.append("between")
            position = self._rng.choice(positions)
        gap = np.zeros(int(self._voice.config.sample_rate * 0.12), dtype=np.float32)
        if position == "between" and len(phrases) > 1:
            boundary = self._rng.randrange(1, len(phrases))
            return np.concatenate(
                [
                    self._render_speech(" ".join(phrases[:boundary])),
                    gap,
                    self._call,
                    gap,
                    self._render_speech(" ".join(phrases[boundary:])),
                ]
            )
        speech = self._render_speech(text)
        if position == "start":
            return np.concatenate([self._call, gap, speech])
        return np.concatenate([speech, gap, self._call])

    def render(self, text: str, path: Path) -> None:
        """Write the reply with optional recorded call, effects and loudness."""
        text = for_piper(text, self._voice.config.espeak_voice)
        audio = self._render_with_call(text)

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
