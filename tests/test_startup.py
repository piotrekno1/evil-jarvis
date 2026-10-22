import os
from pathlib import Path
from unittest import mock

import pytest

from evil_jarvis import download, startup
from evil_jarvis.cli import parse_args
from evil_jarvis.persona import load_persona

MISSING_FILES = ["--llm-model", "/nope.gguf", "--whisper-model", "/nope-whisper"]
MISSING_FILES += ["--voice", "en_US-joe-medium"]


def args_for(*argv):
    return parse_args([*argv, "--env-file", "/nonexistent"])


# .env file


def test_env_file_sets_only_missing_variables(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nexport A_TEST_DSN='https://x@y/z'\nB_TEST=from-file\n\nnot a setting\n"
    )
    with mock.patch.dict(os.environ, {"B_TEST": "already-set"}):
        loaded = startup.load_env_file(env_file)
        assert os.environ["A_TEST_DSN"] == "https://x@y/z"
        assert os.environ["B_TEST"] == "already-set"
    assert loaded == ["A_TEST_DSN"]
    assert "A_TEST_DSN" not in os.environ  # patch.dict restored the environment


def test_missing_env_file_is_fine():
    assert startup.load_env_file(Path("/nonexistent/.env")) == []


# Banner


def test_banner_describes_the_run():
    args = args_for("--persona", "jarvis", "--language", "auto", "--text-only")
    text = startup.banner(args, load_persona("jarvis"), "off")
    assert "Evil Jarvis · Jarvis" in text
    assert "typed text (any language)" in text
    assert (
        "Piper pl_PL-darkman-medium, metallic effect, plain intonation; "
        "English: en_GB-alan-medium" in text
    )
    assert "Bielik-4.5B-v3.0-Instruct.Q8_0" in text


def test_banner_without_speech():
    text = startup.banner(args_for("--no-speech"), load_persona("clown"), "off")
    assert "off (replies printed)" in text


def test_banner_describes_idle_and_flag_disables_it():
    assert "3 sounds after 30 s quiet" in startup.banner(args_for(), load_persona("clown"), "off")
    assert args_for("--no-idle").no_idle


@pytest.mark.parametrize("flag", ["--no-idle", "--no-speech", "--text-only"])
def test_banner_omits_disabled_idle(flag):
    assert "microphone briefly paused" not in startup.banner(
        args_for(flag), load_persona("clown"), "off"
    )


@pytest.mark.parametrize(
    ("module", "function", "returns", "expected"),
    [
        (
            startup.stt,
            "faster_whisper_device",
            "cuda",
            "large-v3-turbo-ct2 (faster-whisper) on the NVIDIA GPU",
        ),
        (startup.stt, "faster_whisper_device", "cpu", "an NVIDIA GPU is recommended"),
        (startup.llm, "gpu_offload_supported", False, "CPU only"),
    ],
)
def test_banner_names_the_hardware(module, function, returns, expected):
    args = args_for(
        "--stt-engine", "faster-whisper", "--whisper-model", "/m/whisper-large-v3-turbo-ct2"
    )
    with mock.patch.object(module, function, return_value=returns):
        assert expected in startup.banner(args, load_persona("clown"), "off")


@pytest.mark.parametrize(
    ("env", "capture", "expected"),
    [
        ({}, False, "set WILDEDGE_DSN"),
        (
            {"WILDEDGE_DSN": "https://s@example.invalid/k"},
            True,
            "timings and text, no audio",
        ),
        ({"WILDEDGE_DSN": "https://s@example.invalid/k"}, False, "timings only, no text or audio"),
    ],
)
def test_telemetry_line(env, capture, expected):
    with mock.patch.dict(os.environ, env, clear=True):
        assert expected in startup.telemetry_summary(capture)


# Requirements


@pytest.mark.parametrize(
    "problem",
    [
        "LLM not found: /nope.gguf (check --llm-model)",
        "Whisper model not found: /nope-whisper (check --whisper-model)",
        "en_US-joe-medium.onnx (run with --download)",
    ],
)
def test_every_missing_file_is_listed(problem):
    assert problem in "\n".join(startup.missing_requirements(args_for(*MISSING_FILES)))


def test_custom_models_are_not_downloadable():
    assert [item.what for item in startup.missing_downloads(args_for(*MISSING_FILES))] == [
        "Piper voice"
    ]


def test_default_models_are_downloadable(tmp_path):
    with (
        mock.patch.object(startup.llm, "DEFAULT_MODEL", tmp_path / "llm.gguf"),
        mock.patch.dict(
            startup.stt.MODELS,
            {key: (tmp_path / key, repo) for key, (_, repo) in startup.stt.MODELS.items()},
        ),
    ):
        args = args_for("--llm-model", str(tmp_path / "llm.gguf"), "--no-speech")
        args.whisper_model = str(startup.stt.MODELS[startup.stt.model_key(args.stt_engine)][0])
        assert [item.what for item in startup.missing_downloads(args)] == ["LLM", "Whisper model"]


def test_url_download_is_atomic(tmp_path):
    source, target = tmp_path / "src.bin", tmp_path / "out" / "voice.onnx"
    source.write_bytes(b"voice")
    download.urls([(source.as_uri(), target)])()
    assert target.read_bytes() == b"voice"
    assert not target.with_name("voice.onnx.part").exists()


def test_missing_packages_name_the_extras():
    with mock.patch.object(startup, "installed", return_value=False):
        problems = startup.missing_requirements(args_for())
    assert problems[-1] == "install the missing packages with: uv sync --extra models --extra voice"


def test_text_run_needs_only_the_llm(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    args = args_for("--text-only", "--no-speech", "--llm-model", str(model))
    with mock.patch.object(startup, "installed", return_value=True):
        assert startup.missing_requirements(args) == []


@pytest.mark.parametrize(
    ("platform", "name", "package", "hint"),
    [
        ("linux", "ffmpeg", "ffmpeg", "sudo apt install ffmpeg"),
        ("darwin", "portaudio", "libportaudio2", "brew install portaudio"),
    ],
)
def test_install_hints_match_the_system(platform, name, package, hint):
    with mock.patch.object(startup.sys, "platform", platform):
        assert startup.system_package(name, package) == hint


def test_mlx_is_refused_off_apple_silicon():
    args = args_for("--stt-engine", "mlx")
    with mock.patch.object(startup.stt, "apple_silicon", return_value=False):
        assert "needs an Apple Silicon Mac" in "\n".join(startup.missing_requirements(args))


def test_every_language_voice_is_checked():
    args = args_for("--language", "auto")
    assert startup.piper_voices(args) == ["pl_PL-bass-high", "en_US-ryan-high"]
