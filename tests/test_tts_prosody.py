import random
from itertools import pairwise

import pytest

from evil_jarvis.cli import parse_args
from evil_jarvis.tts.prosody import plan_maniacal_delivery, split_phrases

PHRASES = ["Raz.", "Dwa?", "Trzy...", "Cztery.", "Pięć!", "Sześć.", "Siedem!"]


@pytest.mark.parametrize(
    ("text", "phrases"),
    [
        (
            "Boisz się? To tylko gry... HEHE, niespodzianki!",
            ["Boisz się?", "To tylko gry...", "HEHE, niespodzianki!"],
        ),
        ("Bez kropki na końcu", ["Bez kropki na końcu"]),
        ('Ja? Powiedziała: "zejdź mi z oczu!".', ["Ja?", 'Powiedziała: "zejdź mi z oczu!".']),
        ("Mówi: „Dobry wieczór!” Następne zdanie.", ["Mówi: „Dobry wieczór!”", "Następne zdanie."]),
        ('"..."', []),
    ],
)
def test_split_keeps_whole_sentences_and_punctuation(text, phrases):
    assert split_phrases(text) == phrases


@pytest.mark.parametrize("seed", range(50))
def test_plan_never_opens_or_repeats_an_aside(seed):
    moods = [p.mood for p in plan_maniacal_delivery(PHRASES, random.Random(seed))]
    assert moods[0] != "aside"
    assert not any(a == b == "aside" for a, b in pairwise(moods))
    assert moods[-1] == "shout"  # ends on "!"


def test_plan_is_erratic_in_one_voice():
    plan = plan_maniacal_delivery(PHRASES, random.Random(0))
    assert len({p.speed for p in plan}) > 3
    assert not hasattr(plan[0], "semitones")  # one voice: no pitch jumps


def test_same_seed_gives_same_delivery():
    phrases = split_phrases("Mam! Mam, mam... tylko nie pytaj!")
    assert plan_maniacal_delivery(phrases, random.Random(7)) == plan_maniacal_delivery(
        phrases, random.Random(7)
    )


def test_maniacal_is_the_default_prosody():
    assert parse_args([]).prosody == "maniacal"
