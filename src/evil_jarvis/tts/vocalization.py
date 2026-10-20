"""Join a recorded call to speech without a long gap or a jump in loudness."""

from __future__ import annotations

import random
import re

import numpy as np

from evil_jarvis.tts.prosody import fade_edges


class VocalizationGate:
    """Probability and a reply-based cooldown shared across language voices."""

    MAX_POSITION_REPEATS = 2

    def __init__(self, probability: float, min_gap_replies: int = 0, *, seed: int | None = None):
        if not 0 <= probability <= 1:
            raise ValueError("vocalization_probability must be between 0 and 1")
        if type(min_gap_replies) is not int or min_gap_replies < 0:
            raise ValueError("vocalization_min_gap_replies must be a nonnegative integer")
        self.probability = probability
        self.min_gap_replies = min_gap_replies
        self._remaining = 0
        self._rng = random.Random(seed)
        self._placement_rng = random.Random(seed)
        self.last_position: str | None = None
        self._position_streak = 0

    def allow(self, has_cue: bool) -> bool:
        """Advance once per reply, counting replies without cues toward the gap."""
        if self._remaining:
            self._remaining -= 1
            return False
        if not has_cue or self._rng.random() >= self.probability:
            return False
        self._remaining = self.min_gap_replies
        return True

    def choose_position(self, *, can_insert_between: bool) -> str:
        """Prefer endings slightly, discourage repeats, and never repeat thrice."""
        positions = ["start", "end"]
        weights = [0.3, 0.4]
        if can_insert_between:
            positions.append("between")
            weights.append(0.3)
        if self.last_position in positions:
            index = positions.index(self.last_position)
            weights[index] *= 0 if self._position_streak >= self.MAX_POSITION_REPEATS else 0.2
        chosen = self._placement_rng.choices(positions, weights=weights)[0]
        self._position_streak = self._position_streak + 1 if chosen == self.last_position else 1
        self.last_position = chosen
        return chosen


def compile_triggers(triggers: tuple[str, ...]) -> re.Pattern | None:
    """Match literal persona cues, case-insensitively and outside other words."""
    alternatives = "|".join(re.escape(t) for t in sorted(triggers, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE) if alternatives else None


def trim_quiet_edges(
    samples: np.ndarray, rate: int, *, start: bool = True, end: bool = True
) -> np.ndarray:
    """Remove outer silence, retaining 15 ms of breathing room around the sound."""
    if not len(samples):
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak == 0:
        return samples[:0]
    active = np.flatnonzero(np.abs(samples) > peak * 0.025)
    padding = int(rate * 0.015)
    first = max(0, int(active[0]) - padding) if start else 0
    last = min(len(samples), int(active[-1]) + padding + 1) if end else len(samples)
    return fade_edges(samples[first:last], rate=rate, ms=5)


def active_rms(samples: np.ndarray) -> float:
    """Estimate sound level without counting long pauses as quiet speech."""
    if not len(samples):
        return 0.0
    peak = float(np.max(np.abs(samples)))
    active = samples[np.abs(samples) > peak * 0.1]
    return float(np.sqrt(np.mean(active * active))) if len(active) else 0.0


def join_call(speech: list[np.ndarray], call: np.ndarray, rate: int, boundary: int) -> np.ndarray:
    """Insert a call before, between or after speech pieces at a matched level."""
    reference = np.concatenate(speech)
    call_level = active_rms(call)
    speech_level = active_rms(reference)
    if call_level == 0 or speech_level == 0:
        return reference
    # A caw is intentionally slightly quieter than the surrounding speech.
    gain = min(speech_level * 0.8 / call_level, 2.0)
    call = trim_quiet_edges(call, rate) * gain
    pieces = list(speech)
    # Only shorten silence adjacent to the call; preserve all other phrasing.
    if boundary > 0:
        pieces[boundary - 1] = trim_quiet_edges(pieces[boundary - 1], rate, start=False)
    if boundary < len(pieces):
        pieces[boundary] = trim_quiet_edges(pieces[boundary], rate, end=False)
    gap = np.zeros(int(rate * 0.06), dtype=np.float32)
    pieces[boundary:boundary] = [gap, call, gap]
    return np.concatenate(pieces)


def join_replacements(speech: list[np.ndarray], call: np.ndarray, rate: int) -> np.ndarray:
    """Alternate speech fragments and recorded sounds at explicit text cues."""
    reference = np.concatenate(speech)
    level = active_rms(call)
    if level == 0:
        return reference
    speech_level = active_rms(reference)
    gain = min(speech_level * 0.8 / level, 2.0) if speech_level > 0 else 0.65
    matched_call = trim_quiet_edges(call, rate) * gain
    gap = np.zeros(int(rate * 0.06), dtype=np.float32)
    pieces = []
    for index, samples in enumerate(speech):
        pieces.append(trim_quiet_edges(samples, rate, start=index > 0, end=index < len(speech) - 1))
        if index < len(speech) - 1:
            pieces.extend([gap, matched_call, gap])
    return np.concatenate(pieces)
