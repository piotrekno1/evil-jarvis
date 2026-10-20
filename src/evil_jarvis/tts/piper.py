"""Piper: local neural voices (Polish, English and about 40 other languages)."""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
from io import BytesIO
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
from evil_jarvis.tts.vocalization import (
    VocalizationGate,
    compile_triggers,
    join_call,
    join_replacements,
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
        vocalization_audio: bytes | None = None,
        vocalization_probability: float = 0.65,
        vocalization_position: str = "random",
        vocalization_triggers: tuple[str, ...] = (),
        vocalization_min_gap_replies: int = 0,
        vocalization_gate: VocalizationGate | None = None,
    ):
        if not 0 <= vocalization_probability <= 1:
            raise ValueError("vocalization_probability must be between 0 and 1")
        if vocalization_position not in ("random", "start", "end", "between", "trigger", "varied"):
            raise ValueError(
                "vocalization_position must be random, start, end, between, trigger or varied"
            )
        if vocalization_position in ("trigger", "varied") and (
            not vocalization_triggers or vocalization_audio is None
        ):
            raise ValueError("trigger position needs vocalization audio and triggers")
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
        self._vocalization_gate = (
            vocalization_gate
            if vocalization_gate is not None
            else VocalizationGate(vocalization_probability, vocalization_min_gap_replies, seed=seed)
        )
        self.vocalization_position = vocalization_position
        self._trigger_pattern = compile_triggers(vocalization_triggers)
        self._call = None
        if vocalization_audio is not None:
            samples, rate = read_wav(BytesIO(vocalization_audio))
            # Lower the bird recording slightly before the shared crow effect
            # raises the speech. This brings their vocal ranges closer together.
            call_rate = round(rate * 0.86) if effect in ("crow", "crow-harsh") else rate
            self._call = resample(samples, call_rate, self._voice.config.sample_rate)

    @property
    def vocalization_probability(self) -> float:
        return self._vocalization_gate.probability

    @vocalization_probability.setter
    def vocalization_probability(self, probability: float) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("vocalization_probability must be between 0 and 1")
        self._vocalization_gate.probability = probability

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
        """Insert a random call, or replace explicit text cues with recordings."""
        if self.vocalization_position in ("trigger", "varied"):
            return self._render_triggered(text)
        if not self._vocalization_gate.allow(self._call is not None and bool(text.strip())):
            return self._render_speech(text)
        position = self.vocalization_position
        if position == "random":
            positions = ["start", "end"]
            if len(split_phrases(text)) > 1:
                positions.append("between")
            position = self._rng.choice(positions)
        return self._render_positioned(text, position)

    def _render_positioned(self, text: str, position: str):
        """Place a recording only at the opening, ending or a sentence boundary."""
        phrases = split_phrases(text)
        rate = self._voice.config.sample_rate
        if position == "between" and len(phrases) > 1:
            boundary = self._rng.randrange(1, len(phrases))
            return join_call(
                [
                    self._render_speech(" ".join(phrases[:boundary])),
                    self._render_speech(" ".join(phrases[boundary:])),
                ],
                self._call,
                rate,
                boundary=1,
            )
        speech = self._render_speech(text)
        return join_call([speech], self._call, rate, boundary=0 if position == "start" else 1)

    def _render_triggered(self, text: str):
        """Replace configured text cues with audio before pronunciation rewrites."""
        voice = self._voice.config.espeak_voice
        parts = self._trigger_pattern.split(text)
        allowed = self._vocalization_gate.allow(len(parts) > 1)
        if len(parts) == 1:
            return self._render_speech(for_piper(text, voice))
        # Strip every cue even when playback is suppressed. Join the remaining
        # words before TTS so suppressing a sound does not break the intonation.
        cleaned = [part.lstrip(" \t\n,;:.!?…\"'„“”").strip() for part in parts]
        if not allowed:
            spoken = " ".join(part for part in cleaned if any(c.isalnum() for c in part))
            return (
                self._render_speech(for_piper(spoken, voice))
                if spoken
                else np.zeros(0, dtype=np.float32)
            )
        if self.vocalization_position == "varied":
            spoken = " ".join(part for part in cleaned if any(c.isalnum() for c in part))
            if spoken:
                position = self._vocalization_gate.choose_position(
                    can_insert_between=len(split_phrases(spoken)) > 1
                )
                return self._render_positioned(for_piper(spoken, voice), position)
        # Play only the first cue, removing any later cues from this reply.
        parts = [
            cleaned[0],
            " ".join(part for part in cleaned[1:] if any(c.isalnum() for c in part)),
        ]
        speech = []
        for part in parts:
            # Punctuation following a removed cue should not become a spoken
            # fragment or an extra initial pause. Keep ending punctuation so
            # questions and exclamations before a laugh retain their delivery.
            spoken = part.lstrip(" \t\n,;:.!?…\"'„“”")
            if any(c.isalnum() for c in spoken):
                speech.append(self._render_speech(for_piper(spoken, voice)))
            else:
                speech.append(np.zeros(0, dtype=np.float32))
        return join_replacements(speech, self._call, self._voice.config.sample_rate)

    def render(self, text: str, path: Path) -> None:
        """Write the reply with optional recorded call, effects and loudness."""
        if self.vocalization_position not in ("trigger", "varied"):
            text = for_piper(text, self._voice.config.espeak_voice)
        audio = self._render_with_call(text)
        if len(audio) == 0:
            write_wav(path, audio, self._voice.config.sample_rate)
            return

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
