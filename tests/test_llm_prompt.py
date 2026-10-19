import unittest
from unittest.mock import patch

from click.testing import CliRunner

from evil_jarvis.cli import main
from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.persona import load_persona

CLOWN = load_persona("clown")
SYSTEM_PROMPT = CLOWN.profile("en").system_prompt
PL = CLOWN.profile("pl")
POLISH_FALLBACK_RESPONSES = PL.curated_fallback


class ClownReplyTests(unittest.TestCase):
    def test_character_prompt_constrains_length_and_safety(self):
        self.assertIn("1–2 short sentences", SYSTEM_PROMPT)
        self.assertIn("same language as the", SYSTEM_PROMPT)
        self.assertIn("unnamed, unhinged Halloween", SYSTEM_PROMPT)
        self.assertIn("sharp conversational", SYSTEM_PROMPT)
        self.assertIn("never\nprefix the response with a name", SYSTEM_PROMPT)
        self.assertIn("Never repeat or paraphrase", SYSTEM_PROMPT)
        self.assertIn("realistic threats", SYSTEM_PROMPT)
        self.assertIn("family Halloween attraction", SYSTEM_PROMPT)

    def test_model_echo_is_removed(self):
        visitor = "Hello, is there anybody out here?"
        reply = visitor + " Only me. The empty tent is laughing."
        self.assertEqual(
            LlamaResponder.strip_transcript_echo(reply, visitor),
            "Only me. The empty tent is laughing.",
        )

    def test_model_character_label_is_removed(self):
        self.assertEqual(
            LlamaResponder.strip_transcript_echo(
                "Grinwick: Candy? It bites back.", "Do you have candy?", CLOWN.all_labels
            ),
            "Candy? It bites back.",
        )

    def test_polish_candy_request_uses_curated_response(self):
        curated = PL.curated_reply("Daj cukierka.")
        self.assertIsNotNone(curated)
        intent, reply = curated
        self.assertEqual(intent, "candy")
        self.assertNotIn("bardzo starze", reply)

    def test_polish_hand_request_uses_safe_curated_response(self):
        curated = PL.curated_reply("Daj mi ręce.")
        self.assertIsNotNone(curated)
        self.assertEqual(curated[0], "hands")

    def test_polish_presence_question_uses_authored_response(self):
        curated = PL.curated_reply("Halo, czy ktoś tu jest?")
        self.assertIsNotNone(curated)
        self.assertEqual(curated[0], "presence")
        self.assertNotIn("kto tam szło", curated[1].lower())

    def test_polish_words_with_l_stroke_match_intents(self):
        self.assertEqual(PL.intent("Masz słodycze?"), "candy")
        self.assertEqual(PL.intent("Słyszysz mnie?"), "presence")

    def test_unusual_polish_input_has_no_curated_intent(self):
        self.assertIsNone(PL.intent("Czy księżyc pamięta cyrk?"))

    def test_curated_mode_uses_generic_line_for_unusual_input(self):
        responder = object.__new__(LlamaResponder)
        responder.curated = True
        with patch(
            "evil_jarvis.llm.llama.random.choice",
            return_value=POLISH_FALLBACK_RESPONSES[0],
        ):
            reply = responder.reply("Czy księżyc pamięta cyrk?", language="pl")
        self.assertEqual(reply, POLISH_FALLBACK_RESPONSES[0])

    def test_missing_model_is_reported_as_a_cli_error(self):
        argv = [
            "--llm-model",
            "/not/a/real/model.gguf",
            "--text-only",
            "--env-file",
            "/nonexistent",
        ]
        result = CliRunner().invoke(main, argv)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("LLM not found: /not/a/real/model.gguf", result.output)


if __name__ == "__main__":
    unittest.main()
