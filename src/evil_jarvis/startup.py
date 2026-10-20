"""Startup: the .env file, the check for anything missing, and the banner."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

from dotenv import dotenv_values

from evil_jarvis import download, llm, stt
from evil_jarvis.persona import LANGUAGE_NAMES, Persona
from evil_jarvis.tts import voice_path
from evil_jarvis.tts.piper import voice_files


def load_env_file(path: Path) -> list[str]:
    """Set variables from a .env file (if it exists) unless already set; returns their names."""

    values = dotenv_values(path) if path.is_file() else {}
    loaded = [key for key, value in values.items() if value is not None and key not in os.environ]
    for key in loaded:
        os.environ[key] = values[key]
    return loaded


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def system_package(mac: str, linux: str) -> str:
    return f"brew install {mac}" if sys.platform == "darwin" else f"sudo apt install {linux}"


def portaudio_works() -> bool:
    try:
        import sounddevice  # noqa: F401 - loading it is the test (needs the PortAudio library)
    except OSError:
        return False
    return True


def piper_voices(args) -> list[str]:
    """Every Piper voice this run loads: the main one, then one per language."""
    if args.no_speech:
        return []
    return list(dict.fromkeys([args.voice, *args.voices_by_language.values()]))


def voice_complete(path: Path) -> bool:
    return path.is_file() and path.with_suffix(".onnx.json").is_file()


def missing_downloads(args) -> list[download.Download]:
    """Models and voices this run needs that are missing and that --download can fetch."""
    found = []
    model = Path(args.llm_model).expanduser()
    if model == llm.DEFAULT_MODEL and not model.is_file():
        found.append(
            download.Download(
                "LLM",
                model,
                download.huggingface_file(llm.DEFAULT_MODEL_REPO, model.name, model.parent),
            )
        )
    if not args.text_only:
        whisper = Path(args.whisper_model).expanduser()
        default, repo = stt.MODELS[stt.model_key(args.stt_engine)]
        if whisper == default and not stt.model_present(args.stt_engine, whisper):
            found.append(
                download.Download(
                    "Whisper model", whisper, download.huggingface_repo(repo, whisper)
                )
            )
    for voice in piper_voices(args):
        path = voice_path(voice)
        files = voice_files(path)
        if files and not voice_complete(path):
            found.append(download.Download("Piper voice", path, download.urls(files)))
    return found


def missing_requirements(args) -> list[str]:
    """Everything this run needs that is not installed or downloaded, with how to get it."""
    problems: list[str] = []
    extras: set[str] = set()
    downloadable = {item.path for item in missing_downloads(args)}

    def need_module(module: str, extra: str, what: str) -> None:
        if not installed(module):
            problems.append(f"{what} is not installed (the '{extra}' extra)")
            extras.add(extra)

    def need_file(what: str, path: Path, present: bool, otherwise: str) -> None:
        if not present:
            hint = "run with --download" if path in downloadable else otherwise
            problems.append(f"{what} not found: {path} ({hint})")

    # Text to text
    need_module("llama_cpp", "models", "llama-cpp-python")
    model = Path(args.llm_model).expanduser()
    need_file("LLM", model, model.is_file(), "check --llm-model")

    # Speech to text
    if not args.text_only:
        if args.stt_engine == "mlx" and not stt.apple_silicon():
            problems.append("--stt-engine mlx needs an Apple Silicon Mac; use faster-whisper")
        need_module(
            "mlx_whisper" if args.stt_engine == "mlx" else "faster_whisper",
            "models",
            args.stt_engine,
        )
        whisper = Path(args.whisper_model).expanduser()
        need_file(
            "Whisper model",
            whisper,
            stt.model_present(args.stt_engine, whisper),
            "check --whisper-model",
        )

    # Text to speech
    if piper_voices(args):
        need_module("piper", "voice", "piper-tts")
        for voice in piper_voices(args):
            path = voice_path(voice)
            need_file(
                "Piper voice",
                path,
                voice_complete(path),
                "see https://huggingface.co/rhasspy/piper-voices",
            )
        if args.voice_effect != "clean" and not shutil.which("ffmpeg"):
            install = system_package("ffmpeg", "ffmpeg")
            problems.append(f"ffmpeg is needed for the {args.voice_effect} voice effect: {install}")

    # Microphone and speaker
    if not (args.text_only and args.no_speech) and not portaudio_works():
        problems.append(
            f"the PortAudio library is missing: {system_package('portaudio', 'libportaudio2')}"
        )

    if extras:
        flags = " ".join(f"--extra {extra}" for extra in sorted(extras))
        problems.append(f"install the missing packages with: uv sync {flags}")
    return problems


def language_name(code: str) -> str:
    return "any language" if code == "auto" else LANGUAGE_NAMES.get(code, code)


def hearing_line(args, language: str) -> str:
    if args.text_only:
        return f"typed text ({language})"
    model = Path(args.whisper_model).name.removeprefix("whisper-")
    if args.stt_engine == "mlx":
        return f"Whisper {model} on the Apple GPU, {language}"
    if stt.faster_whisper_device() == "cuda":
        return f"Whisper {model} (faster-whisper) on the NVIDIA GPU, {language}"
    return (
        f"Whisper {model} (faster-whisper) on the CPU, {language} "
        "(slow: an NVIDIA GPU is recommended)"
    )


def thinking_line(args) -> str:
    thinking = Path(args.llm_model).name.removesuffix(".gguf")
    if args.memory_seconds > 0:
        thinking += f", remembers {args.memory_seconds:g} s"
    if llm.gpu_offload_supported() is False:
        thinking += " (CPU only: this llama.cpp build has no GPU support, see README)"
    return thinking


def voice_line(args) -> str:
    if args.no_speech:
        return "off (replies printed)"
    voice = f"Piper {args.voice}, {args.voice_effect} effect, {args.prosody} intonation"
    if args.voices_by_language:
        voice += "; " + ", ".join(
            f"{language_name(code)}: {name}" for code, name in args.voices_by_language.items()
        )
    return voice


def banner(args, persona: Persona, telemetry_line: str) -> str:
    """What this run listens with, thinks with and speaks with."""
    language = language_name(args.language)
    if args.text_only:
        input_line = "type a line and press Enter (Ctrl-D quits)"
    else:
        device = f" {args.input_device}" if args.input_device else ""
        input_line = (
            f"microphone{device}: stay quiet 2 s for calibration, then speak (Ctrl-C quits)"
        )
    title = f"Evil Jarvis · {persona.display_name}"
    if persona.description:
        title += f": {persona.description}"
    rows = [
        ("hearing", hearing_line(args, language)),
        ("thinking", thinking_line(args)),
        ("voice", voice_line(args)),
        ("metrics", telemetry_line),
        ("input", input_line),
    ]
    if persona.idle.sounds and not (args.no_speech or args.text_only or args.no_idle):
        idle = persona.idle
        rows.append((
            "idle", f"{len(idle.sounds)} sounds after {idle.after_seconds:g} s quiet, then "
            f"every {idle.min_interval_seconds:g}–{idle.max_interval_seconds:g} s; "
            "microphone briefly paused",
        ))
    return "\n".join(["", f"  🎪  {title}", *(f"  {name:8}: {value}" for name, value in rows), ""])


def telemetry_summary(capture: str) -> str:
    if not os.environ.get("WILDEDGE_DSN"):
        return "off (set WILDEDGE_DSN in .env to send timings to WildEdge)"
    if capture == "off":
        return "WildEdge (timings and metadata only)"
    what = (
        "transcripts, replies and voice clips" if capture == "audio" else "transcripts and replies"
    )
    return f"WildEdge, capture on: {what} are uploaded"
