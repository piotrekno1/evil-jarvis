import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from evil_jarvis.persona import load_persona, parse_persona
from evil_jarvis.tts.piper import PiperSpeaker
from evil_jarvis.tts.vocalization import VocalizationGate, compile_triggers
from evil_jarvis.wav import read_wav


@pytest.fixture
def clown_speaker():
    instance = PiperSpeaker.__new__(PiperSpeaker)
    instance._voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100, espeak_voice="pl"))
    instance._call = np.array([0.9], dtype=np.float32)
    instance.vocalization_position = "trigger"
    instance._vocalization_gate = VocalizationGate(1, seed=0)
    instance._rng = random.Random(0)
    instance._trigger_pattern = compile_triggers(load_persona("clown").voice.vocalization_triggers)
    instance._render_speech = Mock(return_value=np.array([0.2], dtype=np.float32))
    return instance


@pytest.mark.parametrize(
    ("text", "spoken", "order"),
    [
        ("Hehe! Podejdź bliżej.", ["Podejdź bliżej."], [0.16, 0.2]),
        ("Boisz się? hehe.", ["Boisz się?"], [0.2, 0.16]),
        ("Czekaj... hehe... już idę.", ["Czekaj... ", "już idę."], [0.2, 0.16, 0.2]),
        ("Haha! Raz. hehe. Dwa!", ["Raz. Dwa!"], [0.16, 0.2]),
    ],
)
def test_laugh_cues_become_audio_at_their_exact_position(clown_speaker, text, spoken, order):
    audio = clown_speaker._render_with_call(text)
    # Compare only words/punctuation, not incidental whitespace around a cue.
    actual = [call.args[0].strip() for call in clown_speaker._render_speech.call_args_list]
    assert actual == [part.strip() for part in spoken]
    np.testing.assert_allclose(audio[audio != 0], order)


@pytest.mark.parametrize("cue", ["hehe", "HEHE", "hehehe", "he he he", "hahaha", "hi-hi", "Heh"])
def test_laugh_only_reply_plays_recording_without_synthesizing_words(clown_speaker, cue):
    audio = clown_speaker._render_with_call(cue + "...")
    clown_speaker._render_speech.assert_not_called()
    assert np.any(audio != 0)
    assert np.isfinite(audio).all()


def test_no_cue_means_no_random_laugh(clown_speaker):
    audio = clown_speaker._render_with_call("Zapraszam do cyrku.")
    clown_speaker._render_speech.assert_called_once_with("Zapraszam do cyrku.")
    np.testing.assert_allclose(audio, [0.2])


def test_trigger_does_not_match_inside_another_word(clown_speaker):
    clown_speaker._render_with_call("heheszki")
    clown_speaker._render_speech.assert_called_once_with("heheszki")


def test_disabled_laugh_removes_cue_instead_of_pronouncing_it(clown_speaker):
    clown_speaker.vocalization_probability = 0
    audio = clown_speaker._render_with_call("Hehe! Raz. haha. Dwa!")
    clown_speaker._render_speech.assert_called_once_with("Raz. Dwa!")
    np.testing.assert_allclose(audio, [0.2])


def test_suppressed_laugh_only_reply_is_silent(clown_speaker, tmp_path):
    clown_speaker.vocalization_probability = 0
    path = tmp_path / "quiet.wav"
    clown_speaker.render("Hehe...", path)
    clown_speaker._render_speech.assert_not_called()
    samples, _ = read_wav(path)
    assert len(samples) == 0


def test_cooldown_suppresses_cues_without_speaking_them(clown_speaker):
    clown_speaker._vocalization_gate = VocalizationGate(1, min_gap_replies=2)
    played = []
    for _ in range(4):
        audio = clown_speaker._render_with_call("Hehe! Dobry wieczór.")
        played.append(bool(np.any(np.isclose(audio, 0.16))))
    assert played == [True, False, False, True]
    assert all(
        call.args[0] == "Dobry wieczór." for call in clown_speaker._render_speech.call_args_list
    )


@pytest.mark.parametrize(
    ("position", "order", "spoken"),
    [
        ("start", [0.16, 0.2], ["Pierwsze zdanie. Drugie zdanie!"]),
        ("end", [0.2, 0.16], ["Pierwsze zdanie. Drugie zdanie!"]),
        ("between", [0.2, 0.16, 0.2], ["Pierwsze zdanie.", "Drugie zdanie!"]),
    ],
)
def test_varied_mode_relocates_a_trailing_cue_to_safe_positions(
    clown_speaker, position, order, spoken
):
    clown_speaker.vocalization_position = "varied"
    clown_speaker._vocalization_gate.choose_position = Mock(return_value=position)
    audio = clown_speaker._render_with_call("Pierwsze zdanie. Drugie zdanie! hehe.")
    actual = [call.args[0] for call in clown_speaker._render_speech.call_args_list]
    assert actual == spoken
    np.testing.assert_allclose(audio[audio != 0], order)


def test_varied_mode_still_obeys_frequency_limit(clown_speaker):
    clown_speaker.vocalization_position = "varied"
    clown_speaker.vocalization_probability = 0
    clown_speaker._vocalization_gate.choose_position = Mock()
    audio = clown_speaker._render_with_call("Hehe! Dobry wieczór.")
    clown_speaker._vocalization_gate.choose_position.assert_not_called()
    clown_speaker._render_speech.assert_called_once_with("Dobry wieczór.")
    np.testing.assert_allclose(audio, [0.2])


def test_varied_mode_does_not_add_sound_without_a_cue(clown_speaker):
    clown_speaker.vocalization_position = "varied"
    audio = clown_speaker._render_with_call("Dobry wieczór.")
    np.testing.assert_allclose(audio, [0.2])


def test_varied_mode_keeps_exactly_one_sound_for_multiple_cues(clown_speaker):
    clown_speaker.vocalization_position = "varied"
    clown_speaker._vocalization_gate.choose_position = Mock(return_value="start")
    audio = clown_speaker._render_with_call("Hehe! Raz. Haha! Dwa.")
    clown_speaker._render_speech.assert_called_once_with("Raz. Dwa.")
    assert np.count_nonzero(np.isclose(audio, 0.16)) == 1


@pytest.mark.parametrize(
    ("cue", "language", "text"),
    [("Kra", "pl", "Kra! Oddaj mi łyżeczkę."), ("Caw", "en", "Caw! That spoon is mine.")],
)
def test_crow_uses_same_trigger_renderer(clown_speaker, cue, language, text):
    persona = load_persona("pani-kraka")
    clown_speaker._voice.config.espeak_voice = language
    clown_speaker._trigger_pattern = compile_triggers(persona.voice.vocalization_triggers)
    audio = clown_speaker._render_with_call(text)
    spoken = clown_speaker._render_speech.call_args.args[0]
    assert cue.casefold() not in spoken.casefold()
    np.testing.assert_allclose(audio[audio != 0], [0.16, 0.2])


def test_cue_matching_uses_literal_strings_not_regex():
    pattern = compile_triggers(("[laugh]",))
    assert pattern.split("Hello [laugh].") == ["Hello ", "."]
    assert pattern.search("Hello l.") is None


@pytest.mark.parametrize(
    "settings",
    [
        {"vocalization_triggers": "hehe"},
        {"vocalization_triggers": [""]},
        {"vocalization_triggers": [7]},
        {"vocalization_position": "trigger"},
        {"vocalization_position": "varied"},
        {"vocalization_position": "trigger", "vocalization_triggers": ["hehe"]},
        {"vocalization_min_gap_replies": -1},
        {"vocalization_min_gap_replies": 1.5},
        {"vocalization_min_gap_replies": True},
    ],
)
def test_invalid_trigger_settings_fail_when_persona_loads(settings):
    with pytest.raises(ValueError, match="vocalization"):
        parse_persona({"voice": settings, "languages": {"default": {"system_prompt": "Clown"}}})
