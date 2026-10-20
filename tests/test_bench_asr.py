import pytest

from evil_jarvis.bench.asr import load_prompts, normalize_words, route, word_error_rate


def test_normalize_drops_punctuation_and_keeps_diacritics():
    assert normalize_words("Cześć, klaunie!") == ["cześć", "klaunie"]


@pytest.mark.parametrize(
    ("reference", "heard", "errors"),
    [
        ("Masz dla mnie cukierka?", "masz mnie cukierki", (2, 4)),
        ("Kim jesteś?", "kim jesteś", (0, 2)),
    ],
)
def test_word_error_rate_counts_substitutions_and_deletions(reference, heard, errors):
    assert word_error_rate(reference, heard) == errors


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Lubisz pizzę?", "none"),
        ("Masz słodycze?", "candy"),
        ("Cukierek albo psikus!", "candy"),
        ("Zuchierki albo psikus.", "none"),  # misheard; should not match anything
        ("Daj mi rękę.", "hands"),
        ("Słyszysz mnie?", "presence"),
        ("Już idę, papa.", "goodbye"),
        ("Idę już, pa pa.", "goodbye"),
        ("Odeźw się!", "presence"),
        ("Czy księżyc pamięta cyrk?", "none"),
    ],
)
def test_router_handles_inflections_and_mishearings(text, intent):
    assert route(text) == intent


def test_prompt_file_is_well_formed():
    prompts = load_prompts()
    assert len(prompts) >= 20
    assert len({p.id for p in prompts}) == len(prompts)
    assert all(p.text and p.intent for p in prompts)
