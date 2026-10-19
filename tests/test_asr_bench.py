import unittest

from creepy_clown.asr_bench import load_prompts, normalize_words, route, word_error_rate


class AsrBenchTests(unittest.TestCase):
    def test_normalize_drops_punctuation_and_keeps_diacritics(self):
        self.assertEqual(normalize_words("Cześć, klaunie!"), ["cześć", "klaunie"])

    def test_word_error_rate_counts_substitutions_and_deletions(self):
        self.assertEqual(word_error_rate("Masz dla mnie cukierka?", "masz mnie cukierki"), (2, 4))
        self.assertEqual(word_error_rate("Kim jesteś?", "kim jesteś"), (0, 2))

    def test_route_returns_none_label_for_unknown_text(self):
        self.assertEqual(route("Lubisz pizzę?"), "none")
        self.assertEqual(route("Masz słodycze?"), "candy")

    def test_prompt_file_is_well_formed(self):
        prompts = load_prompts()
        self.assertGreaterEqual(len(prompts), 20)
        self.assertEqual(len({p.id for p in prompts}), len(prompts))
        self.assertTrue(all(p.text and p.intent for p in prompts))


if __name__ == "__main__":
    unittest.main()
