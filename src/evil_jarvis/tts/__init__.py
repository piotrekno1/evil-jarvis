"""Text to speech: speaks a reply with a Piper voice, effect and intonation.

piper.py         the voice itself (synthesis, then effect, then loudness)
effects.py       ffmpeg effect chains (clean, metallic, metallic-demon)
prosody.py       "maniacal" delivery: per-sentence speed, loudness and pauses
per_language.py  a different voice for each reply language (--language auto)
playback.py      one always-open output stream (works on macOS and Linux)
"""

from __future__ import annotations

from typing import Protocol

from evil_jarvis.tts.effects import EFFECT_NAMES
from evil_jarvis.tts.per_language import PerLanguageSpeaker
from evil_jarvis.tts.piper import DEFAULT_VOICE, PiperSpeaker, voice_path
from evil_jarvis.tts.playback import AudioPlayer
from evil_jarvis.tts.prosody import PROSODY_STYLES

__all__ = [
    "DEFAULT_VOICE",
    "EFFECT_NAMES",
    "PROSODY_STYLES",
    "SilentSpeaker",
    "Speaker",
    "build_speaker",
    "voice_path",
]


class Speaker(Protocol):
    def speak(self, text: str, language: str | None = None) -> None: ...


class SilentSpeaker:
    """For --no-speech: replies are only printed."""

    def speak(self, text: str, language: str | None = None) -> None:
        del text, language


def build_speaker(
    voice: str,
    *,
    voices_by_language: dict[str, str] | None = None,
    effect: str = "clean",
    speed: float = 1.0,
    prosody: str = "plain",
    output_device: int | str | None = None,
    vocalization: str | None = None,
    vocalization_probability: float = 0.65,
    vocalization_position: str = "random",
):
    """The main voice, plus one per language when replies may switch language."""

    player = AudioPlayer(output_device)

    def piper(name: str) -> PiperSpeaker:
        return PiperSpeaker(
            str(voice_path(name)),
            effect=effect,
            speed=speed,
            prosody=prosody,
            player=player,
            vocalization=vocalization,
            vocalization_probability=vocalization_probability,
            vocalization_position=vocalization_position,
        )

    default = piper(voice)
    if not voices_by_language:
        return default
    # Load every voice now so a missing file fails at startup, not mid-show.
    return PerLanguageSpeaker(
        {lang: piper(name) for lang, name in voices_by_language.items()}, default
    )
