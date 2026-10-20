import pytest

from evil_jarvis.tts.pronunciation import for_piper


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Ale nie jestem tu sam... hehe.", "Ale nie jestem tu sam... he, he, he."),
        ("Hahaha! Mam cię.", "Ha, ha, ha! Mam cię."),
        ("He he, zgadnij.", "He, he, he, zgadnij."),
        ("Hi-hi!", "Hi, hi, hi!"),
        ("One są... hmmm... lekko przypalone.", "One są... lekko przypalone."),
        ("Mmm, pyszne.", "pyszne."),
        ("Hm. Ciekawe.", "Ciekawe."),
        ("To nie Halloween.", "To nie Halołin."),
        ("Herbata, hala i hałas zostają.", "Herbata, hala i hałas zostają."),
    ],
)
def test_text_is_respelled_for_the_polish_voice(text, spoken):
    assert for_piper(text, "pl") == spoken


def test_other_languages_keep_their_words():
    assert for_piper("Happy Halloween, hehe.", "en-us") == "Happy Halloween, he, he, he."
