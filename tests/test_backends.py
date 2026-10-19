import unittest
from unittest.mock import patch

from creepy_clown.app import main
from creepy_clown.backends import (
    AppleSpeechTranscriber,
    LlamaResponder,
    KokoroSpeaker,
    MacOSSaySpeaker,
    POLISH_FALLBACK_RESPONSES,
    SYSTEM_PROMPT,
    ScriptedResponder,
)


class BackendTests(unittest.TestCase):
    def test_apple_speech_requires_explicit_language(self):
        with self.assertRaisesRegex(ValueError, "explicit language"):
            AppleSpeechTranscriber("auto")

    def test_scripted_responder_returns_nonempty_reply(self):
        self.assertTrue(ScriptedResponder().reply("hello", language="en"))

    def test_character_prompt_constrains_length_and_safety(self):
        self.assertIn("1–2 short sentences", SYSTEM_PROMPT)
        self.assertIn("same language as the", SYSTEM_PROMPT)
        self.assertIn("unnamed, unhinged Halloween", SYSTEM_PROMPT)
        self.assertIn("sharp conversational", SYSTEM_PROMPT)
        self.assertIn("never\nprefix the response with a name", SYSTEM_PROMPT)
        self.assertIn("Never repeat or paraphrase", SYSTEM_PROMPT)
        self.assertIn("realistic threats", SYSTEM_PROMPT)
        self.assertIn("family Halloween attraction", SYSTEM_PROMPT)

    def test_creepy_voice_filter_contains_pitch_and_echo_stages(self):
        effect = MacOSSaySpeaker.effect_filter()
        self.assertIn("asetrate=", effect)
        self.assertIn("asplit=2", effect)
        self.assertIn("adelay=", effect)
        self.assertIn("amix=", effect)
        self.assertIn("aecho=", effect)
        self.assertIn("alimiter=", effect)
        self.assertNotIn("tremolo", effect)

    def test_kokoro_filter_preserves_natural_voice(self):
        effect = KokoroSpeaker.effect_filter("subtle")
        self.assertIn("aecho=", effect)
        self.assertNotIn("asetrate=", effect)
        self.assertNotIn("tremolo", effect)

    def test_possessed_kokoro_filter_layers_three_voices(self):
        effect = KokoroSpeaker.effect_filter("possessed")
        self.assertIn("asplit=3", effect)
        self.assertIn("[demon]asetrate=", effect)
        self.assertIn("acrusher=", effect)
        self.assertIn("chorus=", effect)
        self.assertIn("amix=inputs=3", effect)

    def test_demonic_kokoro_filter_adds_two_low_shadows(self):
        effect = KokoroSpeaker.effect_filter("demonic")
        self.assertIn("asplit=4", effect)
        self.assertIn("[demon]asetrate=16320", effect)
        self.assertIn("[abyss]asetrate=13440", effect)
        self.assertIn("amix=inputs=4", effect)
        self.assertIn("equalizer=f=115", effect)
        self.assertIn("flanger=", effect)

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
                "Grinwick: Candy? It bites back.", "Do you have candy?"
            ),
            "Candy? It bites back.",
        )

    def test_polish_candy_request_uses_curated_response(self):
        curated = LlamaResponder.curated_polish_reply("Daj cukierka.")
        self.assertIsNotNone(curated)
        intent, reply = curated
        self.assertEqual(intent, "candy")
        self.assertNotIn("bardzo starze", reply)

    def test_polish_hand_request_uses_safe_curated_response(self):
        curated = LlamaResponder.curated_polish_reply("Daj mi ręce.")
        self.assertIsNotNone(curated)
        self.assertEqual(curated[0], "hands")

    def test_polish_presence_question_uses_authored_response(self):
        curated = LlamaResponder.curated_polish_reply("Halo, czy ktoś tu jest?")
        self.assertIsNotNone(curated)
        self.assertEqual(curated[0], "presence")
        self.assertNotIn("kto tam szło", curated[1].lower())

    def test_polish_words_with_l_stroke_match_intents(self):
        self.assertEqual(LlamaResponder.polish_intent("Masz słodycze?"), "candy")
        self.assertEqual(LlamaResponder.polish_intent("Słyszysz mnie?"), "presence")

    def test_unusual_polish_input_has_no_curated_intent(self):
        self.assertIsNone(LlamaResponder.curated_polish_reply("Czy księżyc pamięta cyrk?"))

    def test_curated_mode_uses_generic_line_for_unusual_input(self):
        responder = object.__new__(LlamaResponder)
        responder.curated_polish = True
        with patch(
            "creepy_clown.backends.random.choice",
            return_value=POLISH_FALLBACK_RESPONSES[0],
        ):
            reply = responder.reply("Czy księżyc pamięta cyrk?", language="pl")
        self.assertEqual(reply, POLISH_FALLBACK_RESPONSES[0])

    def test_missing_model_is_reported_as_a_cli_error(self):
        argv = ["creepy-clown", "--llm-model", "/not/a/real/model.gguf", "--text-only"]
        with patch("sys.argv", argv), self.assertRaises(SystemExit) as exit_context:
            main()
        self.assertEqual(exit_context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
