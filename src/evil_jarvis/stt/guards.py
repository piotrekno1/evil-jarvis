"""Checks and helpers shared by the speech recognizers."""

from __future__ import annotations

import re
import unicodedata
from itertools import pairwise

MAX_WORDS_PER_SECOND = 5  # nobody speaks faster
WORD_ALLOWANCE = 2  # slack for very short clips
MIN_CLIP_SECONDS = 0.5
LOOP_REPEATS = 4  # the same word 4 times in a row is a Whisper loop
# Whisper learned these from video subtitles and "hears" them in room noise.
KNOWN_HALLUCINATIONS = (
    "dzięki za oglądanie",
    "dziękuję za oglądanie",
    "napisy stworzone przez społeczność",
    "amara org",
    "subskrybuj",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
)


def is_runaway_transcript(text: str, seconds: float, language: str | None = None) -> bool:
    """Detect Whisper hallucinations: subtitle phrases, looping words, impossible speed,
    foreign scripts."""
    words = re.findall(r"\w+", text.casefold())
    if not words:
        return False
    spoken = " ".join(words)
    if any(phrase in spoken for phrase in KNOWN_HALLUCINATIONS):
        return True
    if len(words) > WORD_ALLOWANCE + MAX_WORDS_PER_SECOND * max(seconds, MIN_CLIP_SECONDS):
        return True
    run = 1
    for previous, current in pairwise(words):
        run = run + 1 if current == previous else 1
        if run >= LOOP_REPEATS:
            return True
    if language in ("pl", "en"):
        for character in text:
            if character.isalpha() and not unicodedata.name(character, "").startswith("LATIN"):
                return True
    return False
