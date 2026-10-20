"""Rewrites words Piper reads badly into spellings it pronounces well.

The printed reply is unchanged; only the text sent to the voice is rewritten.
"""

from __future__ import annotations

import re

# "hehe", "Hahaha", "he he", "hi-hi": the same laugh syllable twice or more.
LAUGH = re.compile(r"\b(h([aei]))(?:[\s,-]*h\2)+\b", re.IGNORECASE)
LAUGH_BEATS = 3
# Consonant-only fillers are spelled out letter by letter ("hmmm" -> "ha em em em").
FILLER = re.compile(r"\b(?:hm+|mhm+|m{2,})\b[.,…!?]*\s*", re.IGNORECASE)
REPEATED_PAUSE = re.compile(r"(?:\.\.\.|…)(?:\s*(?:\.\.\.|…))+")

# Per espeak voice: words read with the wrong language's rules.
RESPELLINGS = {
    "pl": {"Halloween": "Halołin", "halloween": "halołin"},
}


def laugh(match: re.Match) -> str:
    """Separate beats ("He, he, he"), so each one is its own short laugh."""
    vowel = match.group(2).lower()
    first = match.group(1)[0] + vowel
    return ", ".join([first] + [f"h{vowel}"] * (LAUGH_BEATS - 1))


def for_piper(text: str, espeak_voice: str = "pl") -> str:
    text = LAUGH.sub(laugh, text)
    text = FILLER.sub("", text)
    for word, spelling in RESPELLINGS.get(espeak_voice, {}).items():
        text = re.sub(rf"\b{word}\b", spelling, text)
    text = REPEATED_PAUSE.sub("...", text)
    return re.sub(r"\s{2,}", " ", text).strip()
