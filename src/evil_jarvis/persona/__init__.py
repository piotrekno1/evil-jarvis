"""Personas: one character's prompts, style examples, reply checks and voice.

A persona is a TOML file (see builtin/clown.toml for every field). Built-in
personas live in builtin/; `--persona` also accepts a path to any .toml file.
The reply checks themselves are in checks.py.
"""

from __future__ import annotations

import random
import re
import tomllib
import unicodedata
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from evil_jarvis.persona.checks import CHECKS, normalize_text

DEFAULT_PERSONA = "clown"
LANGUAGE_NAMES = {"pl": "Polish", "en": "English", "de": "German", "fr": "French", "es": "Spanish"}
ANY_LANGUAGE_RULE = "Reply entirely in the language used in the visitor's statement."
GENERIC_LABELS = ("assistant", "reply")


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
    # Voice per language code, e.g. {"en": "en_US-ryan-high"}; `voice` covers the rest.
    by_language: dict[str, str] = field(default_factory=dict)

    def voice_for(self, language: str | None) -> str | None:
        return self.by_language.get(language or "", self.voice)


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

    def profile(self, language: str | None) -> LanguageProfile:
        return self.languages.get(language or "", self.languages["default"])

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
_VOICE_KEYS = {"voice", "effect", "prosody", "speed", "by_language"}


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


def parse_persona(data: dict, source: str = "") -> Persona:
    reject_unknown("persona", data, _PERSONA_KEYS)
    languages = data.get("languages") or {}
    if "default" not in languages:
        raise ValueError("A persona needs a [languages.default] section")
    voice = data.get("voice", {})
    reject_unknown("[voice]", voice, _VOICE_KEYS)
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
        **top,
    )


def builtin_dir():
    return resources.files("evil_jarvis.persona") / "builtin"


def builtin_personas() -> list[str]:
    return sorted(
        p.name[: -len(".toml")] for p in builtin_dir().iterdir() if p.name.endswith(".toml")
    )


def load_persona(name_or_path: str = DEFAULT_PERSONA) -> Persona:
    """Load a built-in persona by name, or any persona .toml file by path."""
    path = Path(name_or_path).expanduser()
    if name_or_path.endswith(".toml") or path.is_file():
        if not path.is_file():
            raise ValueError(f"Persona file not found: {path}")
        text, source = path.read_text(encoding="utf-8"), str(path)
    else:
        builtin = builtin_dir() / f"{name_or_path}.toml"
        if not builtin.is_file():
            raise ValueError(
                f"Unknown persona {name_or_path!r}; built-in: {', '.join(builtin_personas())} "
                "(or pass a path to a .toml file)"
            )
        text, source = builtin.read_text(encoding="utf-8"), f"{name_or_path}.toml"
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Persona {source} is not valid TOML: {exc}") from exc
    try:
        return parse_persona(data, source)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Persona {source}: {exc}") from exc
