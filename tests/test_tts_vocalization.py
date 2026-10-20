import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from evil_jarvis.persona import load_persona, parse_persona
from evil_jarvis.tts.piper import PiperSpeaker
from evil_jarvis.tts.vocalization import VocalizationGate, active_rms, join_call, trim_quiet_edges


def speaker(position, probability=1.0):
    instance = PiperSpeaker.__new__(PiperSpeaker)
    instance._voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
    instance._rng = random.Random(0)
    instance._call = np.array([0.9], dtype=np.float32)
    instance.vocalization_position = position
    instance._vocalization_gate = VocalizationGate(probability, seed=0)
    instance._render_speech = Mock(return_value=np.array([0.2], dtype=np.float32))
    return instance


@pytest.mark.parametrize(
    ("position", "expected"),
    [("start", [0.16, 0.2]), ("end", [0.2, 0.16]), ("between", [0.2, 0.16, 0.2])],
)
def test_call_is_inserted_at_requested_boundary(position, expected):
    instance = speaker(position)
    audio = instance._render_with_call("First sentence. Second sentence!")
    np.testing.assert_allclose(audio[audio != 0], expected)
    spoken = [call.args[0] for call in instance._render_speech.call_args_list]
    if position == "between":
        assert spoken == ["First sentence.", "Second sentence!"]
    else:
        assert spoken == ["First sentence. Second sentence!"]


def test_between_falls_back_to_end_for_one_sentence():
    instance = speaker("between")
    audio = instance._render_with_call("One sentence, with a comma.")
    np.testing.assert_allclose(audio[audio != 0], [0.2, 0.16])
    instance._render_speech.assert_called_once_with("One sentence, with a comma.")


@pytest.mark.parametrize("disabled", ["probability", "no_call"])
def test_disabled_call_preserves_unsplit_speech(disabled):
    instance = speaker("between", probability=0 if disabled == "probability" else 1)
    if disabled == "no_call":
        instance._call = None
    audio = instance._render_with_call("One. Two.")
    np.testing.assert_allclose(audio, [0.2])
    instance._render_speech.assert_called_once_with("One. Two.")


def test_random_placement_varies_and_inserts_only_one_call():
    instance = speaker("random")
    positions = set()
    for _ in range(30):
        audio = instance._render_with_call("One. Two.")
        nonzero = audio[audio != 0]
        positions.add(tuple(np.round(nonzero, 2)))
        assert np.count_nonzero(np.isclose(audio, 0.16)) == 1
    assert len(positions) == 3


@pytest.mark.parametrize(
    "voice",
    [
        {"vocalization": "typo"},
        {"vocalization_position": "middle"},
        {"vocalization_probability": -0.1},
        {"vocalization_probability": 1.1},
        {"vocalization_probability": "sometimes"},
    ],
)
def test_invalid_vocalization_settings_fail_at_load(voice):
    with pytest.raises(ValueError, match="vocalization"):
        parse_persona({"voice": voice, "languages": {"default": {"system_prompt": "Crow"}}})


def test_personas_own_their_different_recordings():
    persona = load_persona("pani-kraka")
    assert persona.voice.vocalization == "assets/crow.wav"
    assert persona.voice.vocalization_position == "varied"
    assert persona.vocalization_audio.startswith(b"RIFF")
    clown = load_persona("clown")
    assert clown.voice.vocalization == "assets/chuckle.wav"
    assert clown.voice.vocalization_position == "varied"
    assert clown.vocalization_audio != persona.vocalization_audio
    for name in ("jarvis", "madame-nevermore"):
        assert load_persona(name).voice.vocalization is None


def test_call_tracks_surrounding_speech_level():
    call = np.full(50, 0.8, dtype=np.float32)
    for level in (0.1, 0.3):
        speech = np.full(100, level, dtype=np.float32)
        audio = join_call([speech], call, 1000, boundary=1)
        nonzero = audio[audio != 0]
        assert float(np.max(np.abs(nonzero))) <= level + 1e-7
        # A central portion of the call, outside the fades.
        assert np.count_nonzero(np.isclose(audio, level * 0.8)) > 20


def test_join_removes_long_silence_next_to_call_and_preserves_outer_pause():
    speech = np.concatenate([np.zeros(300), np.full(100, 0.2), np.zeros(500)])
    call = np.concatenate([np.zeros(200), np.full(100, 0.4), np.zeros(200)])
    audio = join_call([speech], call, 1000, boundary=1)
    assert np.all(audio[:300] == 0)
    assert len(audio) < len(speech)  # old 500 ms gap is gone, despite adding a call
    assert np.isfinite(audio).all()


def test_silent_audio_does_not_divide_by_zero():
    silence = np.zeros(100, dtype=np.float32)
    assert active_rms(silence) == 0
    assert len(trim_quiet_edges(silence, 1000)) == 0
    np.testing.assert_array_equal(join_call([silence], silence, 1000, 0), silence)
