import unittest

from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.persona import load_persona

PL = load_persona("clown").profile("pl")
polish_reply_problem = PL.reply_problem


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create_chat_completion(self, messages, **options):
        self.calls.append(messages)
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


def responder_with(replies):
    responder = object.__new__(LlamaResponder)
    responder._disable_thinking = False
    responder._model = FakeModel(replies)
    return responder


class PolishReplyCheckTests(unittest.TestCase):
    def test_good_reply_passes(self):
        self.assertIsNone(
            polish_reply_problem("Żonglowałem. Teraz żongluje tylko cień.", "Umiesz żonglować?")
        )

    def test_bad_replies_are_rejected(self):
        cases = {
            "": "empty",
            "\x17" * 60: "control characters",
            "... !!! ...": "no words",
            "Yes, and the moon is with you.": "English",
            "Kostium 🎃 straszny.": "emoji or symbol",
            "Księżyc 沒有 pamięta.": "non-Latin",
            "Umiesz żonglować? Oczywiście.": "repeats the visitor",
            " ".join(["słowo"] * 30): "too long",
            PL.examples[0][1]: "copies a style example",
            "Tylko ciszę. Ale czasem słyszę... *cichy chichot*": "stage direction",
            "Mam! Mam, mam... tylko nie pytaj.": "stutter opening",
        }
        for reply, reason in cases.items():
            with self.subTest(reply=reply):
                self.assertIn(reason, polish_reply_problem(reply, "Umiesz żonglować?") or "")


class EchoOpeningTests(unittest.TestCase):
    def test_asking_back_the_visitors_word_is_rejected(self):
        problem = polish_reply_problem(
            "Rymy? Tylko wtedy, gdy ty rymujesz się z grobem.", "Czy ty mówisz do mnie rymami?"
        )
        self.assertEqual(problem, "opens with the visitor's word")
        self.assertIn("visitor's word", polish_reply_problem("Przepis? Mam!", "Znasz przepis?"))

    def test_other_questions_pass(self):
        self.assertIsNone(polish_reply_problem("Ciemno? Tu zawsze jest jasno.", "Znasz przepis?"))
        self.assertIsNone(polish_reply_problem("Tak? A ja myślę, że nie.", "Mam cukierka."))


class PolishGenerationTests(unittest.TestCase):
    def test_polish_goes_to_the_model(self):
        responder = responder_with(["Mam jeden. Trochę się rusza."])
        self.assertEqual(
            responder.reply("Masz cukierka?", language="pl"), "Mam jeden. Trochę się rusza."
        )
        messages = responder._model.calls[0]
        self.assertIn("po polsku", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Masz cukierka?"})

    def test_rejected_reply_is_regenerated(self):
        responder = responder_with(["Yes, and the candy is yours.", "Klaun: Mam, ale gryzie."])
        self.assertEqual(responder.reply("Masz cukierka?", language="pl"), "Mam, ale gryzie.")
        self.assertEqual(len(responder._model.calls), 2)

    def test_markdown_bold_is_kept_without_asterisks(self):
        responder = responder_with(["Jestem w nastroju do... **strasznych psikusów**!"])
        self.assertEqual(
            responder.reply("Jak się dzisiaj miewasz?", language="pl"),
            "Jestem w nastroju do... strasznych psikusów!",
        )
        self.assertEqual(len(responder._model.calls), 1)

    def test_clown_stays_silent_when_every_attempt_fails(self):
        responder = responder_with(["the and you"] * PL.attempts)
        self.assertEqual(responder.reply("Masz cukierka?", language="pl"), "")

    def test_repeating_an_earlier_reply_is_rejected(self):
        earlier = ("Mam jeden. Trochę się rusza.",)
        self.assertIn(
            "repeats an earlier",
            polish_reply_problem("Mam jeden. Trochę się rusza!", "Masz?", earlier),
        )
        self.assertIn(
            "repeats an earlier",
            polish_reply_problem("Mam jeden, trochę gryzie.", "Masz?", earlier),
        )
        self.assertIsNone(polish_reply_problem("Nie mam. Zjadłem wszystkie.", "Masz?", earlier))


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class MemoryTests(unittest.TestCase):
    def make(self, replies, memory_seconds=90.0):
        responder = responder_with(replies)
        responder.memory_seconds = memory_seconds
        responder.clock = FakeClock()
        return responder

    def test_earlier_exchange_is_sent_with_the_next_line(self):
        responder = self.make(["Grinwick. Hehe!", "Sto lat. Albo dwieście..."])
        responder.reply("Jak masz na imię?", language="pl")
        responder.clock.now += 30
        responder.reply("A ile masz lat?", language="pl")
        messages = responder._model.calls[1]
        self.assertEqual(
            messages[-3:],
            [
                {"role": "user", "content": "Jak masz na imię?"},
                {"role": "assistant", "content": "Grinwick. Hehe!"},
                {"role": "user", "content": "A ile masz lat?"},
            ],
        )

    def test_memory_is_cleared_after_quiet(self):
        responder = self.make(["Grinwick. Hehe!", "Sto lat. Albo dwieście..."])
        responder.reply("Jak masz na imię?", language="pl")
        responder.clock.now += 91
        responder.reply("A ile masz lat?", language="pl")
        messages = responder._model.calls[1]
        self.assertNotIn({"role": "assistant", "content": "Grinwick. Hehe!"}, messages)

    def test_ongoing_talk_keeps_only_recent_turns(self):
        count = LlamaResponder.MEMORY_TURNS + 2
        responder = self.make([f"Odpowiedź numer {i} hehe." for i in range(count + 1)])
        for i in range(count):
            responder.clock.now += 60  # never 90 s of quiet
            responder.reply(f"Pytanie {i}?", language="pl")
        self.assertEqual(len(responder._history), LlamaResponder.MEMORY_TURNS)
        self.assertEqual(responder._history[-1][0], f"Pytanie {count - 1}?")

    def test_silent_reply_is_not_remembered(self):
        responder = self.make(["the and you"] * PL.attempts)
        responder.reply("Masz cukierka?", language="pl")
        self.assertEqual(responder._history, ())

    def test_zero_disables_memory(self):
        responder = self.make(["Grinwick. Hehe!", "Sto lat."], memory_seconds=0)
        responder.reply("Jak masz na imię?", language="pl")
        responder.reply("A ile masz lat?", language="pl")
        self.assertNotIn(
            {"role": "assistant", "content": "Grinwick. Hehe!"}, responder._model.calls[1]
        )


if __name__ == "__main__":
    unittest.main()
