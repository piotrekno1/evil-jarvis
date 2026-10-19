import random
import unittest

from creepy_clown.app import build_parser
from creepy_clown.backends import plan_maniacal_delivery, split_phrases


class ProsodyTests(unittest.TestCase):
    def test_split_keeps_whole_sentences_and_punctuation(self):
        self.assertEqual(
            split_phrases("Boisz się? To tylko gry... HEHE, niespodzianki!"),
            ["Boisz się?", "To tylko gry...", "HEHE, niespodzianki!"],
        )
        self.assertEqual(split_phrases("Bez kropki na końcu"), ["Bez kropki na końcu"])

    def test_plan_is_erratic_but_never_opens_or_repeats_an_aside(self):
        phrases = ["Raz.", "Dwa?", "Trzy...", "Cztery.", "Pięć!", "Sześć.", "Siedem!"]
        for seed in range(50):
            plan = plan_maniacal_delivery(phrases, random.Random(seed))
            moods = [p.mood for p in plan]
            self.assertNotEqual(moods[0], "aside")
            self.assertFalse(any(a == b == "aside" for a, b in zip(moods, moods[1:])))
            self.assertEqual(moods[-1], "shout")  # ends on "!"
        self.assertGreater(len({p.speed for p in plan}), 3)
        self.assertFalse(hasattr(plan[0], "semitones"))  # one voice: no pitch jumps

    def test_same_seed_gives_same_delivery(self):
        phrases = split_phrases("Mam! Mam, mam... tylko nie pytaj!")
        self.assertEqual(
            plan_maniacal_delivery(phrases, random.Random(7)),
            plan_maniacal_delivery(phrases, random.Random(7)),
        )

    def test_maniacal_is_the_default_prosody(self):
        self.assertEqual(build_parser().parse_args([]).prosody, "maniacal")


if __name__ == "__main__":
    unittest.main()
