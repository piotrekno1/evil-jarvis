import wave

import numpy as np
import pytest

from evil_jarvis.microphone import DetectorConfig, VoiceActivityRecorder
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


def fake_recorder(monkeypatch, levels):
    config = DetectorConfig(
        frame_ms=20,
        speech_start_ms=40,
        silence_ms=40,
        min_speech_ms=40,
        preroll_ms=20,
        calibration_seconds=0,
    )
    recorder = VoiceActivityRecorder(config)
    recorder._threshold = 0.1
    frames = iter(np.full(config.frame_samples, level, dtype=np.float32) for level in levels)
    monkeypatch.setattr(recorder, "_ensure_stream", lambda: None)
    monkeypatch.setattr(recorder, "_next_frame", lambda: next(frames))
    return recorder


def test_idle_pauses_detection_and_discards_its_own_audio(monkeypatch):
    recorder = fake_recorder(monkeypatch, [0, 0, 0.3, 0.3, 0, 0])
    calls = []

    def play():
        calls.append("idle")
        assert not recorder._listening.is_set()
        noise = np.full((320, 1), 0.9, dtype=np.float32)
        recorder._on_audio(noise, 320, None, None)
        assert recorder._frames.empty()
        # Also model a frame already queued just before detection was paused.
        recorder._frames.put(noise[:, 0])

    actions = iter([play, None])
    audio = recorder.capture(on_idle=lambda: next(actions))
    assert calls == ["idle"]
    assert recorder._frames.empty()
    assert not recorder._listening.is_set()
    assert np.max(audio) == pytest.approx(0.3)


def test_idle_never_runs_after_speech_onset(monkeypatch):
    recorder = fake_recorder(monkeypatch, [0.3, 0.3, 0, 0])

    def unexpected():
        pytest.fail("idle considered during speech")

    audio = recorder.capture(on_idle=unexpected)
    assert audio is not None


def test_idle_interrupt_leaves_recorder_paused(monkeypatch):
    recorder = fake_recorder(monkeypatch, [0])

    def stop():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        recorder.capture(on_idle=lambda: stop)
    assert not recorder._listening.is_set()
