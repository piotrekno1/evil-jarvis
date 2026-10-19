import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from evil_jarvis.microphone import DetectorConfig, VoiceActivityRecorder


class FakeStream:
    """Feeds frames through the callback like PortAudio, then silence."""

    instances = 0

    def __init__(self, frames, frame_samples, callback=None, **_):
        FakeStream.instances += 1
        self._frames = list(frames)
        self._silence = np.zeros(frame_samples, dtype=np.float32)
        self._callback = callback
        self._running = False
        self.closed = False
        self.ready = lambda: False  # scripted frames flow only once this is true

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self._running:
            frame = self._frames.pop(0) if self._frames and self.ready() else self._silence
            self._callback(frame.reshape(-1, 1), len(frame), None, None)
            time.sleep(0.0005)

    def push(self, frames):
        self._frames.extend(frames)

    def close(self):
        self._running = False
        self.closed = True


def fake_sounddevice(stream_holder, frames, n):
    def make(**kwargs):
        stream = FakeStream(frames, n, **kwargs)
        stream_holder.append(stream)
        return stream

    return types.SimpleNamespace(InputStream=make)


class RecorderTests(unittest.TestCase):
    def setUp(self):
        FakeStream.instances = 0

    def test_soft_onset_before_loud_speech_is_kept(self):
        cfg = DetectorConfig(calibration_seconds=0.02, preroll_ms=300, speech_start_ms=240)
        n = cfg.frame_samples
        silence = [np.zeros(n, np.float32)] * 20
        soft = [np.full(n, 0.002 + i / 2000, np.float32) for i in range(10)]  # below threshold
        loud = [np.full(n, 0.2, np.float32)] * 30
        frames = [np.zeros(n, np.float32), *silence, *soft, *loud]  # first frame calibrates
        holder = []
        listened = []
        recorder = VoiceActivityRecorder(cfg)
        # Frames only flow while listening; hold them back until capture starts.
        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(holder, [], n)}):
            recorder._ensure_stream()
            holder[0].ready = recorder._listening.is_set
            holder[0].push(frames)
            samples = recorder.capture(on_listening=lambda: listened.append(1))
        recorder.close()

        self.assertEqual(listened, [1])
        lead_in_frames = 300 // cfg.frame_ms
        first_kept = samples[: lead_in_frames * n].reshape(lead_in_frames, n)
        kept_soft = sum(1 for f in first_kept if 0 < f[0] < 0.1)
        self.assertEqual(kept_soft, 10)

    def test_stream_is_opened_once_and_audio_between_captures_is_dropped(self):
        cfg = DetectorConfig(
            calibration_seconds=0.02, speech_start_ms=60, min_speech_ms=60, silence_ms=100
        )
        n = cfg.frame_samples
        loud = [np.full(n, 0.2, np.float32)] * 10
        holder = []
        recorder = VoiceActivityRecorder(cfg)
        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(holder, [], n)}):
            recorder._ensure_stream()
            holder[0].ready = recorder._listening.is_set
            holder[0].push([np.zeros(n, np.float32), *loud])  # quiet frame calibrates
            self.assertIsNotNone(recorder.capture())
            # Loud audio while the clown "speaks" must not leak into the next capture.
            recorder._frames.put(np.full(n, 0.9, np.float32))
            holder[0].push(loud)
            second = recorder.capture()
        recorder.close()

        self.assertEqual(FakeStream.instances, 1)
        self.assertTrue(holder[0].closed)
        self.assertLess(float(np.max(second)), 0.5)


if __name__ == "__main__":
    unittest.main()
