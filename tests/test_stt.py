import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from evil_jarvis import stt
from evil_jarvis.cli import parse_args
from evil_jarvis.stt import faster_whisper as module
from evil_jarvis.stt.guards import is_runaway_transcript
from evil_jarvis.stt.mlx_whisper import MlxWhisperTranscriber
from evil_jarvis.wav import SPEECH_SAMPLE_RATE, write_wav


class RunawayGuardTests(unittest.TestCase):
    def test_normal_polish_sentence_passes(self):
        self.assertFalse(is_runaway_transcript("Czy księżyc pamięta twój cyrk?", 2.0, "pl"))

    def test_repeated_word_loop_is_caught(self):
        self.assertTrue(
            is_runaway_transcript("Tanty, chomu, duchy, duchy, duchy, duchy", 2.0, "pl")
        )

    def test_too_many_words_for_clip_length_is_caught(self):
        self.assertTrue(is_runaway_transcript(" ".join(f"slowo{i}" for i in range(20)), 2.0, "pl"))

    def test_foreign_script_in_polish_is_caught(self):
        self.assertTrue(
            is_runaway_transcript("Wilczynski intentλλaje pyta沒有 Costume?", 2.0, "pl")
        )

    def test_subtitle_phrases_are_caught(self):
        self.assertTrue(is_runaway_transcript("Dzięki za oglądanie!", 1.5, "pl"))
        self.assertTrue(is_runaway_transcript("Thanks for watching.", 1.5, "en"))
        self.assertTrue(
            is_runaway_transcript("Napisy stworzone przez społeczność Amara.org", 3.0, "pl")
        )
        self.assertFalse(is_runaway_transcript("Dzięki za cukierka!", 1.5, "pl"))

    def test_empty_transcript_is_not_runaway(self):
        self.assertFalse(is_runaway_transcript("", 2.0, "pl"))


class MlxTranscriberTests(unittest.TestCase):
    def test_missing_model_explains_how_to_download(self):
        with self.assertRaisesRegex(ValueError, "--download"):
            MlxWhisperTranscriber("/not/a/model")

    def test_no_speech_flag_is_available(self):
        self.assertTrue(parse_args(["--no-speech"]).no_speech)

    def test_polish_is_the_default_language(self):
        self.assertEqual(parse_args([]).language, "pl")

    def test_engine_follows_the_hardware(self):
        with mock.patch.object(stt, "apple_silicon", return_value=True):
            self.assertEqual(stt.default_engine(), "mlx")
        with mock.patch.object(stt, "apple_silicon", return_value=False):
            self.assertEqual(stt.default_engine(), "faster-whisper")
        self.assertEqual(parse_args([]).stt_engine, stt.default_engine())

    def test_cpu_gets_the_small_model(self):
        self.assertEqual(
            stt.default_model("faster-whisper", device="cuda").name, "whisper-large-v3-turbo-ct2"
        )
        self.assertEqual(
            stt.default_model("faster-whisper", device="cpu").name, "whisper-small-ct2"
        )
        self.assertEqual(stt.default_model("mlx").name, "whisper-large-v3-turbo-mlx")


class FasterWhisperTests(unittest.TestCase):
    def make(self, text, device=None):

        created = {}

        class FakeModel:
            def __init__(self, path, device, compute_type):
                created.update(device=device, compute_type=compute_type)

            def transcribe(self, audio, **options):
                return [SimpleNamespace(text=text)], SimpleNamespace(language="pl")

        fake = SimpleNamespace(WhisperModel=FakeModel)
        directory = tempfile.mkdtemp()
        with mock.patch.dict(sys.modules, {"faster_whisper": fake}):
            transcriber = module.FasterWhisperTranscriber(directory, "pl", device=device)
        return transcriber, created

    def wav(self, seconds=1.0):

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        write_wav(path, np.zeros(int(SPEECH_SAMPLE_RATE * seconds), dtype=np.float32))
        return path

    def test_gpu_uses_float16_and_cpu_int8(self):
        _, created = self.make("x", device="cuda")
        self.assertEqual(created, {"device": "cuda", "compute_type": "float16"})
        _, created = self.make("x", device="cpu")
        self.assertEqual(created, {"device": "cpu", "compute_type": "int8"})

    def test_device_is_detected(self):
        with mock.patch("evil_jarvis.stt.faster_whisper.cuda_available", return_value=True):
            transcriber, _ = self.make("x")
        self.assertEqual(transcriber.device, "cuda")

    def test_runaway_transcripts_are_discarded(self):
        transcriber, _ = self.make("duchy duchy duchy duchy duchy", device="cpu")
        self.assertEqual(transcriber.transcribe(self.wav()), "")
        self.assertTrue(transcriber.last_discarded)
        transcriber, _ = self.make("Masz cukierka?", device="cpu")
        self.assertEqual(transcriber.transcribe(self.wav()), "Masz cukierka?")
        self.assertEqual(transcriber.last_language, "pl")


if __name__ == "__main__":
    unittest.main()
