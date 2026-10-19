import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import wildedge
from click.testing import CliRunner

from evil_jarvis import conversation, telemetry
from evil_jarvis.cli import main, parse_args
from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.persona import builtin_personas, load_persona
from evil_jarvis.tts.per_language import PerLanguageSpeaker
from evil_jarvis.tts.piper import PiperSpeaker, voice_files

MINIMAL = """
name = "butler"
display_name = "Butler"
language = "en"

[voice]
voice = "en_US-ryan-high"
speed = 1.1

[languages.default]
system_prompt = "You are a butler."
max_words = 5
checks = ["no_symbols"]
"""


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create_chat_completion(self, messages, **options):
        self.calls.append((messages, options))
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


def write_persona(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as handle:
        handle.write(text)
    return handle.name


class LoadingTests(unittest.TestCase):
    def test_every_builtin_persona_loads(self):
        self.assertIn("clown", builtin_personas())
        self.assertIn("jarvis", builtin_personas())
        for name in builtin_personas():
            with self.subTest(persona=name):
                persona = load_persona(name)
                self.assertTrue(persona.profile("pl").system_prompt)
                self.assertTrue(persona.profile("xx").system_prompt)  # falls back to default

    def test_persona_from_a_file(self):
        persona = load_persona(write_persona(MINIMAL))
        self.assertEqual(persona.display_name, "Butler")
        self.assertIs(persona.profile("pl"), persona.languages["default"])
        self.assertEqual(
            persona.profile("en").reply_problem("one two three four five six"), "too long (6 words)"
        )

    def test_typos_are_reported(self):
        cases = {
            MINIMAL.replace('display_name = "Butler"', 'dispaly_name = "Butler"'): "dispaly_name",
            MINIMAL.replace('"no_symbols"', '"no_emoji"'): "no_emoji",
            MINIMAL.replace("[languages.default]", "[languages.en]"): "languages.default",
            MINIMAL.replace('voice = "en_US-ryan-high"', 'voices = "en_US-ryan-high"'): "voices",
            MINIMAL.replace("[voice]\n", '[voice]\ntts = "kokoro"\n'): "tts",
        }
        for text, mention in cases.items():
            with self.subTest(mention=mention), self.assertRaisesRegex(ValueError, mention):
                load_persona(write_persona(text))

    def test_unknown_persona_lists_the_builtins(self):
        with self.assertRaisesRegex(ValueError, "clown"):
            load_persona("nobody")


class ResponderTests(unittest.TestCase):
    def test_persona_drives_prompt_params_and_labels(self):
        responder = object.__new__(LlamaResponder)
        responder._disable_thinking = False
        responder.persona = load_persona("jarvis")
        responder._model = FakeModel(["Jarvis: Otworzę. Ale tylko raz."])
        self.assertEqual(responder.reply("Otwórz okno.", language="pl"), "Otworzę. Ale tylko raz.")
        messages, options = responder._model.calls[0]
        self.assertIn("Jarvisem", messages[0]["content"])
        self.assertEqual(options["max_tokens"], 60)

    def test_default_profile_gets_a_language_rule(self):
        responder = object.__new__(LlamaResponder)
        responder._disable_thinking = False
        responder.persona = load_persona("jarvis")
        responder._model = FakeModel(["Of course. The doors remember you."])
        responder.reply("Hello?", language="en")
        messages, _ = responder._model.calls[0]
        self.assertTrue(messages[0]["content"].endswith("must be in English."))
        self.assertEqual(messages[-1]["content"], "Visitor: Hello?\nReply only in English.")


class Recorder:
    def __init__(self):
        self.spoken = []

    def speak(self, text, language=None):
        self.spoken.append((text, language))


class SpeakerTests(unittest.TestCase):
    def test_each_reply_uses_its_language_voice(self):
        polish, english = Recorder(), Recorder()
        speaker = PerLanguageSpeaker({"en": english}, default=polish)
        speaker.speak("Hello.", language="en")
        speaker.speak("Cześć.", language="pl")
        speaker.speak("Hallo.", language="de")
        self.assertEqual(english.spoken, [("Hello.", None)])
        self.assertEqual([t for t, _ in polish.spoken], ["Cześć.", "Hallo."])

    def test_reply_language_reaches_the_speaker(self):
        class Responder:
            def reply(self, text, language=None):
                return "Hello."

        speaker = Recorder()
        with mock.patch.dict(os.environ, {}, clear=True):
            quiet = telemetry.Telemetry(wildedge.WildEdge())
        with quiet.exchange() as exchange:
            conversation.respond_and_speak(
                "Hi", "en", Responder(), speaker, telemetry=quiet, exchange=exchange, started=0.0
            )
        self.assertEqual(speaker.spoken, [("Hello.", "en")])

    def test_missing_voice_names_its_own_download(self):
        files = voice_files(Path("models/piper/en_GB-alan-medium.onnx"))
        self.assertEqual(
            files[1],
            (
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
                Path("models/piper/en_GB-alan-medium.onnx.json"),
            ),
        )
        self.assertIsNone(voice_files(Path("my-voice.onnx")))
        with self.assertRaisesRegex(ValueError, "--download"):
            PiperSpeaker("/tmp/en_US-joe-medium.onnx")


class CliTests(unittest.TestCase):
    def test_persona_sets_language_and_voice(self):
        args = parse_args(["--persona", "jarvis"])
        self.assertEqual(
            (args.voice, args.voice_effect, args.prosody),
            ("pl_PL-darkman-medium", "metallic", "plain"),
        )

    def test_flags_override_the_persona(self):
        args = parse_args(
            ["--persona", "jarvis", "--voice-effect", "clean", "--prosody", "maniacal"]
        )
        self.assertEqual((args.voice_effect, args.prosody), ("clean", "maniacal"))

    def test_persona_file_sets_language_voice_and_speed(self):
        args = parse_args(["--persona", write_persona(MINIMAL)])
        self.assertEqual(
            (args.language, args.voice, args.piper_speed), ("en", "en_US-ryan-high", 1.1)
        )
        self.assertEqual((args.voice_effect, args.prosody), ("clean", "plain"))  # defaults

    def test_voice_follows_the_language(self):
        self.assertEqual(parse_args([]).voice, "pl_PL-bass-high")
        self.assertEqual(parse_args(["--language", "en"]).voice, "en_US-ryan-high")
        self.assertEqual(
            parse_args(["--persona", "jarvis", "--language", "en"]).voice, "en_GB-alan-medium"
        )
        self.assertEqual(
            parse_args(["--language", "de"]).voice, "pl_PL-bass-high"
        )  # no German voice listed
        self.assertEqual(parse_args(["--language", "en", "--voice", "x"]).voice, "x")

    def test_auto_language_keeps_every_voice(self):
        args = parse_args(["--language", "auto"])
        self.assertEqual(args.voice, "pl_PL-bass-high")
        self.assertEqual(args.voices_by_language, {"en": "en_US-ryan-high"})
        self.assertEqual(parse_args(["--language", "auto", "--voice", "x"]).voices_by_language, {})

    def test_bad_voice_table_is_reported(self):
        text = MINIMAL.replace("speed = 1.1", 'speed = 1.1\nby_language = "en_US-ryan-high"')
        with self.assertRaisesRegex(ValueError, "by_language"):
            load_persona(write_persona(text))

    def test_list_personas(self):
        result = CliRunner().invoke(main, ["--list-personas"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("jarvis", result.output)

    def test_bad_persona_is_a_usage_error(self):
        result = CliRunner().invoke(main, ["--persona", "nobody", "--text-only"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("Unknown persona", result.output)

    def test_help_lists_options(self):
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        for option in ("--persona", "--wildedge-capture", "--voice-effect"):
            self.assertIn(option, result.output)


if __name__ == "__main__":
    unittest.main()
