import sys
import threading
import time
import types

import numpy as np
import pytest

from evil_jarvis.microphone import DetectorConfig, VoiceActivityRecorder


class FakeStream:
    """Feeds frames through the callback like PortAudio, then silence."""

    def __init__(self, frame_samples, callback=None, **_):
        self._frames = []
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


@pytest.fixture
def listening(monkeypatch):
    """Opens a recorder on a fake microphone; returns (recorder, every stream opened).

    Frames only flow while the recorder listens, so pushed frames wait for capture().
    """
    recorders = []

    def make(cfg):
        streams = []

        def open_stream(**kwargs):
            streams.append(FakeStream(cfg.frame_samples, **kwargs))
            return streams[-1]

        monkeypatch.setitem(
            sys.modules, "sounddevice", types.SimpleNamespace(InputStream=open_stream)
        )
        recorder = VoiceActivityRecorder(cfg)
        recorders.append(recorder)
        recorder._ensure_stream()
        streams[0].ready = recorder._listening.is_set
        return recorder, streams

    yield make
    for recorder in recorders:
        recorder.close()


def test_soft_onset_before_loud_speech_is_kept(listening):
    cfg = DetectorConfig(calibration_seconds=0.02, preroll_ms=300, speech_start_ms=240)
    n = cfg.frame_samples
    silence = [np.zeros(n, np.float32)] * 20
    soft = [np.full(n, 0.002 + i / 2000, np.float32) for i in range(10)]  # below threshold
    loud = [np.full(n, 0.2, np.float32)] * 30
    recorder, streams = listening(cfg)
    streams[0].push([np.zeros(n, np.float32), *silence, *soft, *loud])  # first frame calibrates
    listened = []
    samples = recorder.capture(on_listening=lambda: listened.append(1))

    assert listened == [1]
    lead_in_frames = 300 // cfg.frame_ms
    first_kept = samples[: lead_in_frames * n].reshape(lead_in_frames, n)
    assert sum(1 for f in first_kept if 0 < f[0] < 0.1) == 10


def test_stream_is_opened_once_and_audio_between_captures_is_dropped(listening):
    cfg = DetectorConfig(
        calibration_seconds=0.02, speech_start_ms=60, min_speech_ms=60, silence_ms=100
    )
    n = cfg.frame_samples
    loud = [np.full(n, 0.2, np.float32)] * 10
    recorder, streams = listening(cfg)
    streams[0].push([np.zeros(n, np.float32), *loud])  # quiet frame calibrates
    assert recorder.capture() is not None
    # Loud audio while the clown "speaks" must not leak into the next capture.
    recorder._frames.put(np.full(n, 0.9, np.float32))
    streams[0].push(loud)
    second = recorder.capture()
    recorder.close()

    assert len(streams) == 1
    assert streams[0].closed
    assert float(np.max(second)) < 0.5
