import unittest

from click.testing import CliRunner

from evil_jarvis.cli import main
from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.persona import load_persona

CLOWN = load_persona("clown")
SYSTEM_PROMPT = CLOWN.profile("en").system_prompt


class ClownPromptTests(unittest.TestCase):
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
