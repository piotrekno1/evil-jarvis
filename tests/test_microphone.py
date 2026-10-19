import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from evil_jarvis.microphone import VoiceActivityRecorder
from evil_jarvis.wav import write_wav


class AudioTests(unittest.TestCase):
    def test_rms_of_silence_is_nearly_zero(self):
        self.assertLess(VoiceActivityRecorder.rms(np.zeros(320, dtype=np.float32)), 0.00001)

    def test_rms_tracks_constant_signal(self):
        frame = np.full(320, 0.25, dtype=np.float32)
        self.assertAlmostEqual(VoiceActivityRecorder.rms(frame), 0.25, places=5)

    def test_write_wav_creates_mono_pcm_file(self):

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.wav"
            write_wav(str(output), np.array([-1.0, 0.0, 1.0], dtype=np.float32))

            with wave.open(str(output), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 16_000)
                self.assertEqual(wav.getnframes(), 3)


if __name__ == "__main__":
    unittest.main()
