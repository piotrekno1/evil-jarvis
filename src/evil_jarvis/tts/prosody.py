"""Delivery: how each sentence of a reply is paced and how loud it is."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

from evil_jarvis.tts.effects import PIPER_RATE

PROSODY_STYLES = ("maniacal", "plain")


@dataclass(frozen=True)
class PhraseDelivery:
    """How one sentence is spoken: speed, loudness and the pause before it."""

    speed: float
    gain: float
    gap_ms: int
    mood: str


def split_phrases(text: str) -> list[str]:
    """Split into whole sentences (keeping punctuation).

    Splitting at commas made Piper restart its intonation mid-sentence, which
    sounded like a second speaker interrupting.
    """
    parts = re.findall(r"[^.!?…]+(?:[.!?…]+|$)", text)
    return [part.strip() for part in parts if part.strip(" .!?…")]


def plan_maniacal_delivery(phrases: list[str], rng: random.Random) -> list[PhraseDelivery]:
    """Erratic but one voice: rushed bursts, quieter asides, a loud ending on "!",
    uneven pauses. Pitch is never shifted, since resampling changes the timbre
    and makes the character sound like a different person."""
    plan: list[PhraseDelivery] = []
    previous = ""
    for index, phrase in enumerate(phrases):
        moods = ["bright", "rush", "aside", "normal"]
        weights = [0.35, 0.30, 0.15, 0.20]
        if previous == "aside" or index == 0:
            weights[2] = 0.0  # never open with, or repeat, an aside
        mood = rng.choices(moods, weights=weights)[0]
        if index == len(phrases) - 1 and len(phrases) > 1 and phrase.endswith("!"):
            mood = "shout"
        # Top speed 1.15: at 1.3 Polish words blurred (2026-10-20 pipeline check).
        speed, gain = {
            "bright": (rng.uniform(1.03, 1.1), 1.0),
            "rush": (rng.uniform(1.1, 1.15), 1.0),
            "aside": (rng.uniform(0.88, 0.95), 0.7),
            "normal": (rng.uniform(0.97, 1.03), 0.95),
            "shout": (rng.uniform(1.03, 1.08), 1.15),
        }[mood]
        gap_ms = 0 if index == 0 else rng.randint(60, 260)
        if mood == "aside":
            gap_ms = rng.randint(300, 450)  # a beat of silence before the aside
        plan.append(PhraseDelivery(speed, gain, gap_ms, mood))
        previous = mood
    return plan


def fade_edges(samples, rate: int = PIPER_RATE, ms: int = 8):
    """Short fade in/out so joins between sentences do not click."""

    n = min(len(samples) // 2, rate * ms // 1000)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        samples = samples.copy()
        samples[:n] *= ramp
        samples[-n:] *= ramp[::-1]
    return samples
