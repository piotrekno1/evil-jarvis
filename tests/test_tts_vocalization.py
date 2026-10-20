import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from evil_jarvis.persona import load_persona, parse_persona
from evil_jarvis.tts.piper import PiperSpeaker


def speaker(position, probability=1.0):
    instance = PiperSpeaker.__new__(PiperSpeaker)
    instance._voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
    instance._rng = random.Random(0)
    instance._call = np.array([0.9], dtype=np.float32)
    instance.vocalization_position = position
    instance.vocalization_probability = probability
    instance._render_speech = Mock(return_value=np.array([0.2], dtype=np.float32))
    return instance


@pytest.mark.parametrize(
    ("position", "expected"),
    [("start", [0.9, 0.2]), ("end", [0.2, 0.9]), ("between", [0.2, 0.9, 0.2])],
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
    np.testing.assert_allclose(audio[audio != 0], [0.2, 0.9])
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
        positions.add(tuple(np.round(nonzero, 1)))
        assert np.count_nonzero(audio == instance._call[0]) == 1
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


def test_only_pani_kraka_uses_crow_calls():
    assert load_persona("pani-kraka").voice.vocalization == "crow"
    for name in ("clown", "jarvis", "madame-nevermore"):
        assert load_persona(name).voice.vocalization is None
