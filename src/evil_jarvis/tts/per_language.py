"""One Piper voice per language, chosen per reply (for --language auto)."""

from __future__ import annotations


class PerLanguageSpeaker:
    """Speaks each reply with the voice for its language (for --language auto)."""

    def __init__(self, speakers: dict, default):
        self.speakers = speakers
        self.default = default

    def speak(self, text: str, language: str | None = None) -> None:
        self.speakers.get(language or "", self.default).speak(text)
