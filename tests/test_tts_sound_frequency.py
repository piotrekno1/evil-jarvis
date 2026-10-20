from unittest.mock import patch

import pytest

from evil_jarvis.persona import load_persona
from evil_jarvis.tts import build_speaker
from evil_jarvis.tts.vocalization import VocalizationGate


@pytest.mark.parametrize("probability", [0, 1])
def test_probability_endpoints_are_respected(probability):
    gate = VocalizationGate(probability)
    assert [gate.allow(True) for _ in range(10)] == [bool(probability)] * 10


def test_cue_free_replies_count_toward_cooldown():
    gate = VocalizationGate(1, min_gap_replies=2)
    assert gate.allow(True)
    assert not gate.allow(False)
    assert not gate.allow(False)
    assert gate.allow(True)


@pytest.mark.parametrize(("seed", "allowed"), [(0, False), (1, True)])
def test_quarter_probability_allows_and_suppresses_cues(seed, allowed):
    gate = VocalizationGate(0.25, seed=seed)
    assert gate.allow(True) is allowed


def test_language_switch_does_not_bypass_cooldown():
    decisions = []

    class Voice:
        def __init__(self, _path, **kwargs):
            self.gate = kwargs["vocalization_gate"]

        def speak(self, _text):
            decisions.append(self.gate.allow(True))

    with patch("evil_jarvis.tts.PiperSpeaker", Voice):
        speaker = build_speaker(
            "pl-voice",
            voices_by_language={"en": "en-voice"},
            vocalization_probability=1,
            vocalization_min_gap_replies=2,
        )
    for language in ("pl", "en", "pl", "en"):
        speaker.speak("cue", language)
    assert decisions == [True, False, False, True]


@pytest.mark.parametrize("name", ["clown", "pani-kraka"])
def test_character_defaults_limit_sounds(name):
    voice = load_persona(name).voice
    assert voice.vocalization_probability == 0.25
    assert voice.vocalization_min_gap_replies == 2
    assert voice.vocalization_position == "varied"


def test_variety_uses_all_positions_without_three_identical_placements():
    gate = VocalizationGate(1, seed=7)
    positions = [gate.choose_position(can_insert_between=True) for _ in range(100)]
    assert set(positions) == {"start", "end", "between"}
    assert not any(
        a == b == c for a, b, c in zip(positions, positions[1:], positions[2:], strict=False)
    )


def test_single_sentence_never_gets_an_internal_interruption():
    gate = VocalizationGate(1, seed=7)
    assert {gate.choose_position(can_insert_between=False) for _ in range(50)} == {"start", "end"}


def test_placement_randomness_does_not_change_sound_frequency():
    varied = VocalizationGate(0.25, 2, seed=7)
    fixed = VocalizationGate(0.25, 2, seed=7)
    # Placement should use its own random stream so tuning delivery cannot
    # alter when sounds pass the gate.
    varied_decisions = []
    fixed_decisions = []
    for _ in range(50):
        play = varied.allow(True)
        varied_decisions.append(play)
        fixed_decisions.append(fixed.allow(True))
        if play:
            varied.choose_position(can_insert_between=True)
    assert varied_decisions == fixed_decisions
