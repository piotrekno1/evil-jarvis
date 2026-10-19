import sys
import unittest
from unittest.mock import patch

from creepy_clown.app import build_parser, main
from creepy_clown.backends import PIPER_EFFECTS, PiperSpeaker, piper_effect_chain


class PiperVoiceTests(unittest.TestCase):
    def test_piper_bass_clean_is_the_default_voice(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.tts, "piper")
        self.assertIsNone(args.voice_effect)  # resolved to "clean" for Piper

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

    def test_kokoro_effect_with_piper_is_a_cli_error(self):
        argv = ["creepy-clown", "--text-only", "--voice-effect", "possessed"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as caught:
            main()
        self.assertIn("does not apply", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
