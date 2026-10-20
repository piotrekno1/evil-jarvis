import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from evil_jarvis import stt
from evil_jarvis.cli import parse_args
from evil_jarvis.stt import faster_whisper as module
from evil_jarvis.stt.guards import is_runaway_transcript
from evil_jarvis.stt.mlx_whisper import MlxWhisperTranscriber
from evil_jarvis.wav import SPEECH_SAMPLE_RATE, write_wav


@pytest.mark.parametrize(
    ("text", "seconds", "language", "runaway"),
    [
        ("Czy księżyc pamięta twój cyrk?", 2.0, "pl", False),
        ("Tanty, chomu, duchy, duchy, duchy, duchy", 2.0, "pl", True),  # repeated word loop
        (" ".join(f"slowo{i}" for i in range(20)), 2.0, "pl", True),  # too many words
        ("Wilczynski intentλλaje pyta沒有 Costume?", 2.0, "pl", True),  # foreign script
        ("Dzięki za oglądanie!", 1.5, "pl", True),  # subtitle phrases
        ("Thanks for watching.", 1.5, "en", True),
        ("Napisy stworzone przez społeczność Amara.org", 3.0, "pl", True),
        ("Dzięki za cukierka!", 1.5, "pl", False),
        ("", 2.0, "pl", False),
    ],
)
def test_runaway_transcripts_are_caught(text, seconds, language, runaway):
    assert is_runaway_transcript(text, seconds, language) == runaway


def test_missing_mlx_model_explains_how_to_download():
    with pytest.raises(ValueError, match="--download"):
        MlxWhisperTranscriber("/not/a/model")


def test_no_speech_flag_is_available():
    assert parse_args(["--no-speech"]).no_speech


def test_polish_is_the_default_language():
    assert parse_args([]).language == "pl"


@pytest.mark.parametrize(("apple", "engine"), [(True, "mlx"), (False, "faster-whisper")])
def test_engine_follows_the_hardware(apple, engine):
    with mock.patch.object(stt, "apple_silicon", return_value=apple):
        assert stt.default_engine() == engine


def test_default_engine_is_used_by_the_cli():
    assert parse_args([]).stt_engine == stt.default_engine()


@pytest.mark.parametrize(
    ("engine", "device", "model"),
    [
        ("faster-whisper", "cuda", "whisper-large-v3-turbo-ct2"),
        ("faster-whisper", "cpu", "whisper-small-ct2"),
        ("mlx", None, "whisper-large-v3-turbo-mlx"),
    ],
)
def test_cpu_gets_the_small_model(engine, device, model):
    options = {"device": device} if device else {}
    assert stt.default_model(engine, **options).name == model


@pytest.fixture
def transcriber(monkeypatch, tmp_path):
    """Builds a FasterWhisperTranscriber over a fake model that always hears `text`.

    Returns (transcriber, the device and compute type the model was created with).
    """

    def make(text, device=None):
        created = {}

        class FakeModel:
            def __init__(self, path, device, compute_type):
                created.update(device=device, compute_type=compute_type)

            def transcribe(self, audio, **options):
                return [SimpleNamespace(text=text)], SimpleNamespace(language="pl")

        monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
        return module.FasterWhisperTranscriber(str(tmp_path), "pl", device=device), created

    return make


@pytest.fixture
def silent_wav(tmp_path):
    path = tmp_path / "clip.wav"
    write_wav(path, np.zeros(SPEECH_SAMPLE_RATE, dtype=np.float32))
    return str(path)


@pytest.mark.parametrize(("device", "compute_type"), [("cuda", "float16"), ("cpu", "int8")])
def test_gpu_uses_float16_and_cpu_int8(transcriber, device, compute_type):
    _, created = transcriber("x", device=device)
    assert created == {"device": device, "compute_type": compute_type}


def test_device_is_detected(transcriber, monkeypatch):
    monkeypatch.setattr(module, "cuda_available", lambda: True)
    whisper, _ = transcriber("x")
    assert whisper.device == "cuda"


def test_runaway_transcript_is_discarded(transcriber, silent_wav):
    whisper, _ = transcriber("duchy duchy duchy duchy duchy", device="cpu")
    assert whisper.transcribe(silent_wav) == ""
    assert whisper.last_discarded


def test_normal_transcript_is_kept(transcriber, silent_wav):
    whisper, _ = transcriber("Masz cukierka?", device="cpu")
    assert whisper.transcribe(silent_wav) == "Masz cukierka?"
    assert whisper.last_language == "pl"
