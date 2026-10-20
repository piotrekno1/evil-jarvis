import wave

import numpy as np
import pytest

from evil_jarvis.microphone import VoiceActivityRecorder
from evil_jarvis.wav import write_wav


@pytest.mark.parametrize("level", [0.0, 0.25])
def test_rms_of_a_constant_signal_is_its_level(level):
    frame = np.full(320, level, dtype=np.float32)
    assert VoiceActivityRecorder.rms(frame) == pytest.approx(level, abs=1e-5)


def test_write_wav_creates_mono_pcm_file(tmp_path):
    output = tmp_path / "sample.wav"
    write_wav(str(output), np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 3
