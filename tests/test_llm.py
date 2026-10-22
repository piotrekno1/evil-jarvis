from dataclasses import replace

import pytest

from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.persona import load_persona

PL = load_persona("clown").profile("pl")

BAD_REPLIES = [
    ("", "empty"),
    ("\x17" * 60, "control characters"),
    ("... !!! ...", "no words"),
    ("Yes, and the moon is with you.", "English"),
    ("Kostium 🎃 straszny.", "emoji or symbol"),
    ("Księżyc 沒有 pamięta.", "non-Latin"),
    ("Umiesz żonglować? Oczywiście.", "repeats the visitor"),
    (" ".join(["słowo"] * 30), "too long"),
    ("Tylko ciszę. Ale czasem słyszę... *cichy chichot*", "stage direction"),
    ("Mam! Mam, mam... tylko nie pytaj.", "stutter opening"),
]


def test_good_reply_passes():
    assert PL.reply_problem("Żonglowałem. Teraz żongluje tylko cień.", "Umiesz żonglować?") is None


def test_fitting_style_example_can_be_used_once_but_not_repeated():
    visitor, reply = PL.examples[0]
    assert PL.reply_problem(reply, visitor) is None
    assert PL.reply_problem(reply, visitor, previous=(reply,)) == "repeats an earlier reply"


def test_copy_check_remains_available_to_personas_that_enable_it():
    strict = replace(PL, checks=(*PL.checks, "no_example_copy"))
    visitor, reply = strict.examples[0]
    assert strict.reply_problem(reply, visitor) == "copies a style example"


@pytest.mark.parametrize(
    ("reply", "reason"), BAD_REPLIES, ids=[reason for _, reason in BAD_REPLIES]
)
def test_bad_replies_are_rejected(reply, reason):
    assert reason in (PL.reply_problem(reply, "Umiesz żonglować?") or "")


@pytest.mark.parametrize(
    ("reply", "visitor", "problem"),
    [
        (
            "Rymy? Tylko wtedy, gdy ty rymujesz się z grobem.",
            "Czy ty mówisz do mnie rymami?",
            "opens with the visitor's word",
        ),
        ("Przepis? Mam!", "Znasz przepis?", "opens with the visitor's word"),
        ("Ciemno? Tu zawsze jest jasno.", "Znasz przepis?", None),
        ("Tak? A ja myślę, że nie.", "Mam cukierka.", None),
    ],
)
def test_asking_back_the_visitors_word_is_rejected(reply, visitor, problem):
    assert PL.reply_problem(reply, visitor) == problem


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        ("Mam jeden. Trochę się rusza!", "repeats an earlier reply"),
        ("Mam jeden, trochę gryzie.", "repeats an earlier reply"),
        ("Nie mam. Zjadłem wszystkie.", None),
    ],
)
def test_repeating_an_earlier_reply_is_rejected(reply, problem):
    assert PL.reply_problem(reply, "Masz?", ("Mam jeden. Trochę się rusza.",)) == problem


def test_repeated_slogan_is_rejected_even_at_a_different_position():
    previous = ("Maska ma tylko jeden uśmiech. Tylko dla odważnych.",)
    reply = "Ta kurtyna jest tylko dla odważnych."
    assert PL.reply_problem(reply, "Jaka znowu kurtyna?", previous).startswith("reuses phrase")


def test_visitor_can_ask_for_an_explanation_of_a_previous_phrase():
    previous = ("Bilety są tylko dla odważnych.",)
    assert (
        PL.reply_problem(
            "Hasło tylko dla odważnych to mój kiepski żart.",
            "Co znaczy tylko dla odważnych?",
            previous,
        )
        is None
    )


def test_laugh_syllables_are_not_treated_as_a_recycled_slogan():
    assert PL.reply_problem("Niespodzianka. Hi hi hi.", "Co jest?", ("Witaj. Hi hi hi.",)) is None


def test_polish_goes_to_the_model(responder_with):
    responder = responder_with(["Mam jeden. Trochę się rusza."])
    assert responder.reply("Masz cukierka?", language="pl") == "Mam jeden. Trochę się rusza."
    messages, _ = responder._model.calls[0]
    assert "po polsku" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Masz cukierka?"}


def test_rejected_reply_is_regenerated(responder_with):
    responder = responder_with(["Yes, and the candy is yours.", "Klaun: Mam, ale gryzie."])
    assert responder.reply("Masz cukierka?", language="pl") == "Mam, ale gryzie."
    assert len(responder._model.calls) == 2


def test_markdown_bold_is_kept_without_asterisks(responder_with):
    responder = responder_with(["Jestem w nastroju do... **strasznych psikusów**!"])
    reply = responder.reply("Jak się dzisiaj miewasz?", language="pl")
    assert reply == "Jestem w nastroju do... strasznych psikusów!"
    assert len(responder._model.calls) == 1


def test_clown_stays_silent_when_every_attempt_fails(responder_with):
    responder = responder_with(["the and you"] * PL.attempts)
    assert responder.reply("Masz cukierka?", language="pl") == ""


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def remembering(responder_with):
    """A responder with conversation memory and a clock the test moves."""

    def make(replies, memory_seconds=90.0):
        responder = responder_with(replies, clock=FakeClock())
        responder.memory_seconds = memory_seconds
        return responder

    return make


def test_earlier_exchange_is_sent_with_the_next_line(remembering):
    responder = remembering(["Grinwick. Hehe!", "Sto lat. Albo dwieście..."])
    responder.reply("Jak masz na imię?", language="pl")
    responder.clock.now += 30
    responder.reply("A ile masz lat?", language="pl")
    messages, _ = responder._model.calls[1]
    assert messages[-3:] == [
        {"role": "user", "content": "Jak masz na imię?"},
        {"role": "assistant", "content": "Grinwick. Hehe!"},
        {"role": "user", "content": "A ile masz lat?"},
    ]


def test_memory_is_cleared_after_quiet(remembering):
    responder = remembering(["Grinwick. Hehe!", "Sto lat. Albo dwieście..."])
    responder.reply("Jak masz na imię?", language="pl")
    responder.clock.now += 91
    responder.reply("A ile masz lat?", language="pl")
    messages, _ = responder._model.calls[1]
    assert {"role": "assistant", "content": "Grinwick. Hehe!"} not in messages


def test_ongoing_talk_keeps_only_recent_turns(remembering):
    count = LlamaResponder.MEMORY_TURNS + 2
    responder = remembering([f"Odpowiedź numer {i} hehe." for i in range(count + 1)])
    for i in range(count):
        responder.clock.now += 60  # never 90 s of quiet
        responder.reply(f"Pytanie {i}?", language="pl")
    assert len(responder._history) == LlamaResponder.MEMORY_TURNS
    assert responder._history[-1][0] == f"Pytanie {count - 1}?"


def test_silent_reply_is_not_remembered(remembering):
    responder = remembering(["the and you"] * PL.attempts)
    responder.reply("Masz cukierka?", language="pl")
    assert responder._history == ()


def test_zero_disables_memory(remembering):
    responder = remembering(["Grinwick. Hehe!", "Sto lat."], memory_seconds=0)
    responder.reply("Jak masz na imię?", language="pl")
    responder.reply("A ile masz lat?", language="pl")
    messages, _ = responder._model.calls[1]
    assert {"role": "assistant", "content": "Grinwick. Hehe!"} not in messages


def test_recycled_phrase_gets_a_retry_with_conversation_feedback(remembering):
    responder = remembering(
        [
            "Maska ma jeden uśmiech. Tylko dla odważnych.",
            "Cukierki są tylko dla odważnych.",
            "Częstuj się. Papierek sam wybiera właściciela.",
        ]
    )
    responder.reply("Kim jesteś?", "pl")
    responder.clock.now += 1
    assert (
        responder.reply("Masz cukierki?", "pl") == "Częstuj się. Papierek sam wybiera właściciela."
    )
    assert responder.last_reply_info["attempts"] == 2
    assert responder.last_reply_info["rejections"] == "reuses phrase 'tylko dla odwaznych'"
    retry_messages, _ = responder._model.calls[-1]
    first_messages, _ = responder._model.calls[0]
    assert "Kontynuuj temat ostatniej wypowiedzi" in retry_messages[0]["content"]
    assert "Kontynuuj temat ostatniej wypowiedzi" not in first_messages[0]["content"]


def test_stage_direction_retry_keeps_the_visitors_topic(remembering):
    responder = remembering(
        [
            "Oj tam, oj tam. Znam sztuczkę z jajkiem.",
            "Oj tam, oj tam. Kiedyś jajko wymknęło mi się z rąk.",
            "Oto ona: *unosi jajko w powietrzu* i rzuca je na ziemię.",
            "Podrzucam jajko, a ono znika. Hehe, z kapelusza wyciągam kurę.",
        ]
    )
    responder.reply("Znasz jakąś sztuczkę?", "pl")
    responder.clock.now += 1
    assert responder.reply("Opowiedz mi o tej sztuczce.", "pl") == (
        "Podrzucam jajko, a ono znika. Hehe, z kapelusza wyciągam kurę."
    )
    assert responder.last_reply_info["rejections"] == "repeats an earlier reply | stage direction"
    repeated_messages, _ = responder._model.calls[-2]
    stage_messages, _ = responder._model.calls[-1]
    assert "Kontynuuj temat ostatniej wypowiedzi" in repeated_messages[0]["content"]
    assert "Poprzednia propozycja zawiera didaskalia" in stage_messages[0]["content"]
    assert "Kontynuuj temat ostatniej wypowiedzi" in stage_messages[0]["content"]
    assert "Opowiedz o tym samym" in stage_messages[0]["content"]


def test_conversation_id_changes_after_quiet(responder_with):
    now = [0.0]
    responder = responder_with(["Mam jeden.", "Mam dwa.", "Mam trzy."], clock=lambda: now[0])
    first = responder.upcoming_conversation_id()
    responder.reply("Masz cukierka?", language="pl")
    now[0] = 30.0
    assert responder.upcoming_conversation_id() == first
    responder.reply("A ile masz lat?", language="pl")
    now[0] = 200.0
    second = responder.upcoming_conversation_id()
    assert second != first
    responder.reply("Kim jesteś?", language="pl")
    assert responder._conversation_id == second
