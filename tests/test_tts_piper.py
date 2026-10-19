import unittest

from click.testing import CliRunner

from evil_jarvis.cli import main, parse_args
from evil_jarvis.tts.effects import PIPER_EFFECTS, piper_effect_chain
from evil_jarvis.tts.piper import PiperSpeaker


class PiperVoiceTests(unittest.TestCase):
    def test_clown_default_voice(self):
        args = parse_args([])
        self.assertEqual(args.voice, "pl_PL-bass-high")
        self.assertEqual(args.voice_effect, "metallic-demon")

    def test_effect_chains_end_with_loudness_normalization(self):
        for effect in PIPER_EFFECTS:
            with self.subTest(effect=effect):
                self.assertIn("loudnorm=", piper_effect_chain(effect))
        self.assertIn("aeval=val(0)*sin", piper_effect_chain("metallic-demon"))

    def test_unknown_effect_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown Piper voice effect"):
            piper_effect_chain("spooky")

    def test_missing_voice_explains_how_to_download(self):
        with self.assertRaisesRegex(ValueError, "Piper voice not found"):
            PiperSpeaker("/not/a/voice.onnx")

    def test_unknown_effect_is_a_cli_error(self):
        result = CliRunner().invoke(main, ["--text-only", "--voice-effect", "possessed"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("possessed", result.output)


if __name__ == "__main__":
    unittest.main()
