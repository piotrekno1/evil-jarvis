import unittest

from creepy_clown.app import build_parser
from creepy_clown.backends import LlamaResponder, MlxWhisperTranscriber, is_runaway_transcript


class RunawayGuardTests(unittest.TestCase):
    def test_normal_polish_sentence_passes(self):
        self.assertFalse(is_runaway_transcript("Czy księżyc pamięta twój cyrk?", 2.0, "pl"))

    def test_repeated_word_loop_is_caught(self):
        self.assertTrue(is_runaway_transcript("Tanty, chomu, duchy, duchy, duchy, duchy", 2.0, "pl"))

    def test_too_many_words_for_clip_length_is_caught(self):
        self.assertTrue(is_runaway_transcript(" ".join(f"slowo{i}" for i in range(20)), 2.0, "pl"))

    def test_foreign_script_in_polish_is_caught(self):
        self.assertTrue(is_runaway_transcript("Wilczynski intentλλaje pyta沒有 Costume?", 2.0, "pl"))

    def test_empty_transcript_is_not_runaway(self):
        self.assertFalse(is_runaway_transcript("", 2.0, "pl"))


class KeywordGapTests(unittest.TestCase):
    def test_benchmark_router_gaps_are_fixed(self):
        cases = {
            "Cukierek albo psikus!": "candy",
            "Zuchierki albo psikus.": None,  # misheard; should not match anything
            "Daj mi rękę.": "hands",
            "Już idę, papa.": "goodbye",
            "Idę już, pa pa.": "goodbye",
            "Odeźw się!": "presence",
        }
        for text, intent in cases.items():
            with self.subTest(text=text):
                self.assertEqual(LlamaResponder.polish_intent(text), intent)


class MlxTranscriberTests(unittest.TestCase):
    def test_missing_model_explains_how_to_download(self):
        with self.assertRaisesRegex(ValueError, "hf download"):
            MlxWhisperTranscriber("/not/a/model")

    def test_no_speech_flag_is_available(self):
        self.assertTrue(build_parser().parse_args(["--no-speech"]).no_speech)

    def test_polish_is_the_default_language(self):
        self.assertEqual(build_parser().parse_args([]).language, "pl")

    def test_mlx_is_the_default_transcriber(self):
        self.assertEqual(build_parser().parse_args([]).transcriber, "mlx")


if __name__ == "__main__":
    unittest.main()
