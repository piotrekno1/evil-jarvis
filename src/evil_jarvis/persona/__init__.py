"""Personas: one character's prompts, style examples, reply checks and voice.

A persona is a folder with persona.toml and optional assets (see builtin/clown/).
Built-in personas live in builtin/; `--persona` also accepts a folder or .toml file.
The reply checks themselves are in checks.py.
"""

from __future__ import annotations

import math
import re
import tomllib
import unicodedata
import wave
from dataclasses import dataclass, field
from importlib import resources
from importlib.resources.abc import Traversable
from io import BytesIO
from pathlib import Path, PurePosixPath

from evil_jarvis.persona.checks import CHECKS
from evil_jarvis.wav import read_wav

DEFAULT_PERSONA = "clown"
LANGUAGE_NAMES = {"pl": "Polish", "en": "English", "de": "German", "fr": "French", "es": "Spanish"}
ANY_LANGUAGE_RULE = "Reply entirely in the language used in the visitor's statement."
GENERIC_LABELS = ("assistant", "reply")
MAX_IDLE_SOUND_SECONDS = 2


@dataclass(frozen=True)
class LanguageProfile:
    """How the persona talks in one language (or in any other, for "default")."""

    system_prompt: str
    examples: tuple[tuple[str, str], ...] = ()
    attempts: int = 1
    temperature: float = 0.8
    top_p: float = 0.92
    repeat_penalty: float = 1.1
    max_tokens: int = 72
    max_words: int | None = None
    checks: tuple[str, ...] = ()
    language_rule: str | None = None
    user_template: str = "{text}"
    asr_hint: str | None = None

    @property
    def generation_params(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
        }

    def reply_problem(self, reply: str, visitor: str = "", previous=()) -> str | None:
        """Return why a generated reply is unusable, or None if it is fine."""
        if not reply:
            return "empty"
        if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in reply):
            return "control characters"
        if not any(c.isalpha() for c in reply):
            return "no words"
        words = re.findall(r"[\w']+", reply.casefold())
        if self.max_words is not None and len(words) > self.max_words:
            return f"too long ({len(words)} words)"
        for name in CHECKS:
            if name in self.checks:
                problem = CHECKS[name](reply, words, visitor, previous, self.examples)
                if problem:
                    return problem
        return None

    def prompt_for(self, text: str, language: str | None) -> tuple[str, str]:
        """System prompt and user message for one visitor line."""
        if not self.language_rule:
            return self.system_prompt, self.user_template.format(text=text, language=language or "")
        name = LANGUAGE_NAMES.get(language or "", language)
        if not name:
            return f"{self.system_prompt}\n{ANY_LANGUAGE_RULE}", text
        rule = self.language_rule.format(language=name)
        return f"{self.system_prompt}\n{rule}", self.user_template.format(text=text, language=name)


@dataclass(frozen=True)
class VoiceSettings:
    voice: str | None = None
    effect: str | None = None
    prosody: str | None = None
    speed: float | None = None
    vocalization: str | None = None  # WAV path relative to persona.toml
    vocalization_probability: float = 0.65
    vocalization_min_gap_replies: int = 0
    vocalization_position: str = "random"
    vocalization_triggers: tuple[str, ...] = ()
    # Voice per language code, e.g. {"en": "en_US-ryan-high"}; `voice` covers the rest.
    by_language: dict[str, str] = field(default_factory=dict)

    def voice_for(self, language: str | None) -> str | None:
        return self.by_language.get(language or "", self.voice)


@dataclass(frozen=True)
class IdleSettings:
    """Persona-owned nonverbal sounds and their quiet-time schedule."""

    sounds: tuple[str, ...] = ()
    after_seconds: float = 30.0
    min_interval_seconds: float = 45.0
    max_interval_seconds: float = 90.0
    gain: float = 0.4


@dataclass(frozen=True)
class Persona:
    name: str
    languages: dict[str, LanguageProfile]
    display_name: str = "Jarvis"
    description: str = ""
    language: str = "pl"
    exit_message: str = "Gone quiet."
    labels: tuple[str, ...] = ()
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    source: str = ""
    vocalization_audio: bytes | None = field(default=None, repr=False)
    idle: IdleSettings = field(default_factory=IdleSettings)
    idle_audio: tuple[bytes, ...] = field(default=(), repr=False)

    def profile(self, language: str | None) -> LanguageProfile:
        return self.languages.get(language or "", self.languages["default"])

    def vocalization_rule(self, language: str | None) -> str:
        """Tell the model about the same literal cues the speech renderer uses."""
        if (
            self.voice.vocalization_position not in ("trigger", "varied")
            or not self.voice.vocalization_triggers
        ):
            return ""
        cues = ", ".join(f'"{cue}"' for cue in self.voice.vocalization_triggers)
        if (language or self.language) == "pl":
            return (
                f"Dźwięk postaci: te dokładne hasła odtwarzają nagranie zamiast mowy: {cues}. "
                "Możesz czasem wstawić jedno pasujące hasło w naturalnym miejscu odpowiedzi, "
                "jako osobne słowo. Najwyżej jedno hasło na odpowiedź; zwykle je pomijaj. "
                "Użyj pisowni z listy pasującej do języka rozmowy. Nie dodawaj nawiasów, "
                "gwiazdek ani opisów dźwięku. Nie wyjaśniaj mechanizmu odtwarzania."
            )
        return (
            f"Character audio: these exact cues play a recording instead of speech: {cues}. "
            "You may occasionally insert one suitable cue at a natural point in your reply "
            "as a separate word. Use at most one cue per reply; omit it in most replies. "
            "Choose a listed spelling appropriate to the conversation language. Do not add "
            "brackets, asterisks or sound descriptions. Do not explain the playback mechanism."
        )

    @property
    def all_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.labels, *GENERIC_LABELS)))


_PERSONA_KEYS = {
    "name",
    "display_name",
    "description",
    "language",
    "exit_message",
    "labels",
    "voice",
    "languages",
    "idle",
}
_PROFILE_KEYS = {
    "system_prompt",
    "examples",
    "attempts",
    "temperature",
    "top_p",
    "repeat_penalty",
    "max_tokens",
    "max_words",
    "checks",
    "language_rule",
    "user_template",
    "asr_hint",
}
_VOICE_KEYS = {
    "voice",
    "effect",
    "prosody",
    "speed",
    "by_language",
    "vocalization",
    "vocalization_probability",
    "vocalization_min_gap_replies",
    "vocalization_position",
    "vocalization_triggers",
}


def reject_unknown(where: str, data: dict, allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {where} setting(s): {', '.join(unknown)}")


def parse_profile(language: str, data: dict) -> LanguageProfile:
    where = f"[languages.{language}]"
    reject_unknown(where, data, _PROFILE_KEYS)
    if not data.get("system_prompt"):
        raise ValueError(f"{where} needs a system_prompt")
    checks = tuple(data.get("checks", ()))
    unknown = sorted(set(checks) - set(CHECKS))
    if unknown:
        raise ValueError(
            f"{where} has unknown check(s) {', '.join(unknown)}; choose from {', '.join(CHECKS)}"
        )
    values = {key: data[key] for key in _PROFILE_KEYS - {"examples", "checks"} if key in data}
    if int(values.get("attempts", 1)) < 1:
        raise ValueError(f"{where} attempts must be at least 1")
    return LanguageProfile(
        **values,
        examples=tuple((e["visitor"], e["reply"]) for e in data.get("examples", ())),
        checks=checks,
    )


def read_audio_asset(name, root: Traversable, label: str) -> bytes:
    """Read a persona-owned WAV, including from an installed package resource."""
    if not isinstance(name, str):
        raise ValueError(f"{label} must be a relative .wav path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".wav":
        raise ValueError(f"{label} must be a relative .wav path inside the persona")
    asset = root.joinpath(*path.parts)
    if not asset.is_file():
        raise ValueError(f"{label} file not found: {name}")
    audio = asset.read_bytes()
    try:
        samples, rate = read_wav(BytesIO(audio))
        if len(samples) == 0 or rate <= 0:
            raise ValueError("WAV contains no audio")
    except (ValueError, wave.Error, EOFError) as exc:
        raise ValueError(f"{label}: invalid vocalization WAV {name}: {exc}") from exc
    return audio


def parse_idle(data: dict, root: Traversable) -> tuple[IdleSettings, tuple[bytes, ...]]:
    allowed = {"sounds", "after_seconds", "min_interval_seconds", "max_interval_seconds", "gain"}
    if not isinstance(data, dict):
        raise ValueError("[idle] must be a table")
    reject_unknown("[idle]", data, allowed)
    sounds = data.get("sounds", [])
    if not isinstance(sounds, list) or not all(isinstance(s, str) and s for s in sounds):
        raise ValueError("[idle] sounds must be a list of WAV paths")
    if len(set(sounds)) != len(sounds):
        raise ValueError("[idle] sounds must not contain duplicate paths")
    values = {key: data[key] for key in allowed - {"sounds"} if key in data}
    for key, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"[idle] {key} must be a finite number")
    settings = IdleSettings(sounds=tuple(sounds), **values)
    if settings.after_seconds < 0 or settings.min_interval_seconds <= 0:
        raise ValueError("[idle] delays must be nonnegative, with a positive minimum interval")
    if settings.max_interval_seconds < settings.min_interval_seconds:
        raise ValueError("[idle] maximum interval must be at least the minimum interval")
    if not 0 < settings.gain <= 1:
        raise ValueError("[idle] gain must be greater than zero and at most one")
    audio = tuple(read_audio_asset(name, root, "[idle] sound") for name in sounds)
    for name, clip in zip(sounds, audio, strict=True):
        samples, rate = read_wav(BytesIO(clip))
        if len(samples) / rate > MAX_IDLE_SOUND_SECONDS:
            raise ValueError(f"[idle] sound must be at most 2 seconds: {name}")
    return settings, audio


def parse_persona(
    data: dict, source: str = "", *, asset_root: Traversable | None = None
) -> Persona:
    reject_unknown("persona", data, _PERSONA_KEYS)
    languages = data.get("languages") or {}
    if "default" not in languages:
        raise ValueError("A persona needs a [languages.default] section")
    voice = dict(data.get("voice", {}))
    reject_unknown("[voice]", voice, _VOICE_KEYS)
    positions = ("random", "start", "end", "between", "trigger", "varied")
    if voice.get("vocalization_position", "random") not in positions:
        raise ValueError(
            "[voice] vocalization_position must be random, start, end, between, trigger or varied"
        )
    triggers = voice.get("vocalization_triggers", [])
    if not isinstance(triggers, list) or not all(
        isinstance(t, str) and t.strip() for t in triggers
    ):
        raise ValueError("[voice] vocalization_triggers must be a list of nonempty strings")
    if voice.get("vocalization_position") in ("trigger", "varied") and (
        not triggers or not voice.get("vocalization")
    ):
        raise ValueError("[voice] trigger position needs vocalization and vocalization_triggers")
    voice["vocalization_triggers"] = tuple(triggers)
    probability = voice.get("vocalization_probability", 0.65)
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError("[voice] vocalization_probability must be between 0 and 1")
    gap = voice.get("vocalization_min_gap_replies", 0)
    if type(gap) is not int or gap < 0:
        raise ValueError("[voice] vocalization_min_gap_replies must be a nonnegative integer")
    root = asset_root if asset_root is not None else Path(source).parent
    audio = (
        read_audio_asset(voice["vocalization"], root, "[voice] vocalization")
        if voice.get("vocalization") is not None
        else None
    )
    idle, idle_audio = parse_idle(data.get("idle", {}), root)
    by_language = voice.get("by_language", {})
    if not isinstance(by_language, dict) or not all(
        isinstance(v, str) for v in by_language.values()
    ):
        raise ValueError(
            "[voice.by_language] must map language codes to voice names, "
            'e.g. en = "en_US-ryan-high"'
        )
    top = {
        k: data[k] for k in ("display_name", "description", "language", "exit_message") if k in data
    }
    return Persona(
        name=data.get("name") or Path(source).stem or "persona",
        languages={code: parse_profile(code, profile) for code, profile in languages.items()},
        labels=tuple(data.get("labels", ())),
        voice=VoiceSettings(**voice),
        source=source,
        vocalization_audio=audio,
        idle=idle,
        idle_audio=idle_audio,
        **top,
    )


def builtin_dir():
    return resources.files("evil_jarvis.persona") / "builtin"


def builtin_personas() -> list[str]:
    names = set()
    for entry in builtin_dir().iterdir():
        if entry.is_dir() and (entry / "persona.toml").is_file():
            names.add(entry.name)
        elif entry.is_file() and entry.name.endswith(".toml"):
            names.add(entry.name.removesuffix(".toml"))
    return sorted(names)


def load_persona(name_or_path: str = DEFAULT_PERSONA) -> Persona:
    """Load a built-in persona, a persona folder, or a standalone TOML file."""
    path = Path(name_or_path).expanduser()
    local = path.is_dir() or name_or_path.endswith(".toml") or path.is_file()
    if path.is_dir():
        path /= "persona.toml"
    if local:
        if not path.is_file():
            raise ValueError(f"Persona file not found: {path}")
        text, source = path.read_text(encoding="utf-8"), str(path)
        root = path.parent
    else:
        root = builtin_dir() / name_or_path
        builtin = root / "persona.toml"
        if not builtin.is_file():
            root = builtin_dir()
            builtin = root / f"{name_or_path}.toml"
        if not builtin.is_file():
            raise ValueError(
                f"Unknown persona {name_or_path!r}; built-in: {', '.join(builtin_personas())} "
                "(or pass a path to a persona folder or .toml file)"
            )
        text, source = builtin.read_text(encoding="utf-8"), str(builtin)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Persona {source} is not valid TOML: {exc}") from exc
    try:
        return parse_persona(data, source, asset_root=root)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Persona {source}: {exc}") from exc
