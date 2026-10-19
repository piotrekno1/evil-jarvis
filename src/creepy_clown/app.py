from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import tempfile
import threading
import time

from pathlib import Path

from .audio import DetectorConfig, VoiceActivityRecorder, write_wav
from .backends import (
    AppleSpeechTranscriber,
    MlxWhisperTranscriber,
    PiperSpeaker,
    KokoroSpeaker,
    LlamaResponder,
    MacOSSaySpeaker,
    ScriptedResponder,
    WhisperTranscriber,
)

LOG = logging.getLogger("creepy_clown")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MLX_MODEL = MODELS_DIR / "whisper-large-v3-turbo-mlx"
DEFAULT_LLM_MODEL = MODELS_DIR / "Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"
DEFAULT_PIPER_VOICE = "pl_PL-bass-high"
KOKORO_EFFECTS = ("possessed", "demonic", "subtle")
PIPER_EFFECT_NAMES = ("clean", "metallic", "metallic-demon")
DEFAULT_LLM_DOWNLOAD = (
    "uv run --extra models hf download speakleash/Bielik-4.5B-v3.0-Instruct-GGUF "
    "Bielik-4.5B-v3.0-Instruct.Q8_0.gguf --local-dir models"
)


def install_interrupt_watchdog(grace_seconds: float = 2.0) -> None:
    """Make Ctrl-C always work.

    Python only reacts to Ctrl-C between bytecodes, so a native call that blocks
    (a stalled microphone read, a model call) can swallow it. The OS-level
    signal handler still writes to the wakeup fd, so a watcher thread can see
    the press and force-exit if a clean shutdown has not finished in time.
    A second Ctrl-C exits immediately.
    """
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)

    def watch() -> None:
        presses = 0
        deadline = None
        while True:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready = _wait_readable(read_fd, timeout)
            if ready:
                data = os.read(read_fd, 64)
                presses += data.count(bytes([signal.SIGINT]))
                if presses >= 2:
                    _force_exit("\nForce quitting.\n")
                if presses == 1 and deadline is None:
                    deadline = time.monotonic() + grace_seconds
            elif deadline is not None:
                _force_exit("\nShutdown was stuck; force quitting.\n")

    threading.Thread(target=watch, name="ctrl-c-watchdog", daemon=True).start()


def _wait_readable(fd: int, timeout: float | None) -> bool:
    import select

    ready, _, _ = select.select([fd], [], [], timeout)
    return bool(ready)


def _force_exit(message: str) -> None:
    os.write(sys.stderr.fileno(), message.encode())
    os._exit(130)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline hands-free Halloween clown")
    parser.add_argument(
        "--whisper-model",
        help="model directory: MLX Whisper (default models/whisper-large-v3-turbo-mlx) "
        "or, with --transcriber whisper, a faster-whisper CTranslate2 model",
    )
    parser.add_argument(
        "--transcriber",
        choices=("mlx", "whisper", "apple"),
        default="mlx",
        help="speech-to-text backend: mlx (Whisper turbo on the Apple GPU), "
        "whisper (faster-whisper on CPU), or apple (forced on-device)",
    )
    parser.add_argument(
        "--llm-model",
        help="local GGUF instruct model (default: models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf)",
    )
    parser.add_argument("--scripted", action="store_true", help="Use random canned replies")
    parser.add_argument(
        "--curated-polish",
        action="store_true",
        help="answer Polish with the old keyword-matched handwritten lines instead of the model",
    )
    parser.add_argument(
        "--memory-seconds",
        type=float,
        default=90.0,
        help="remember the conversation until nobody has spoken for this long (0 = no memory)",
    )
    parser.add_argument("--text-only", action="store_true", help="Read visitor lines from stdin")
    parser.add_argument(
        "--language",
        default="pl",
        help="spoken language code (default pl; en for English), or auto to detect it",
    )
    parser.add_argument(
        "--voice",
        help="voice name: Piper voice in models/piper (default pl_PL-bass-high), "
        "Kokoro voice, or installed macOS `say` voice",
    )
    parser.add_argument(
        "--tts",
        choices=("piper", "system", "kokoro"),
        default="piper",
        help="speech synthesis: piper (local neural Polish, default), "
        "system (macOS say), kokoro (local neural English)",
    )
    parser.add_argument("--kokoro-speed", type=float, default=0.98)
    parser.add_argument(
        "--piper-speed", type=float, default=1.0, help="Piper speaking speed (1.2 = 20%% faster)"
    )
    parser.add_argument(
        "--prosody",
        choices=("maniacal", "plain"),
        default="maniacal",
        help="Piper intonation: maniacal (giddy, erratic, default) or plain",
    )
    parser.add_argument(
        "--voice-effect",
        choices=PIPER_EFFECT_NAMES + KOKORO_EFFECTS,
        help="Piper: clean (default), metallic, metallic-demon. "
        "Kokoro: possessed (default), demonic, subtle",
    )
    parser.add_argument("--speech-rate", type=int, default=145)
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="print the clown's replies instead of speaking them",
    )
    parser.add_argument(
        "--plain-voice",
        action="store_true",
        help="disable the pitch, movement, and echo effects",
    )
    parser.add_argument("--input-device", help="sounddevice input device ID or name")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--silence-ms", type=int, default=900)
    parser.add_argument("--max-utterance-seconds", type=float, default=10.0)
    parser.add_argument("--cooldown-seconds", type=float, default=1.0)
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--debug", action="store_true")
    return parser


class SilentSpeaker:
    """Replies are already printed; used while text-to-speech is not ready."""

    def speak(self, text: str) -> None:
        del text


def run_text_loop(responder, speaker, language: str | None = None) -> None:
    while True:
        try:
            text = input("Visitor> ").strip()
        except EOFError:  # Ctrl-D or end of piped input
            print()
            return
        if not text:
            continue
        reply = responder.reply(text, language=language)
        if not reply:
            print("Clown> (silent: no usable reply)")
            continue
        print(f"Clown> {reply}")
        speaker.speak(reply)


def run_voice_loop(args, responder, speaker) -> None:
    if not args.scripted and args.transcriber == "whisper" and not args.whisper_model:
        raise SystemExit("--whisper-model is required unless --scripted is selected")

    if args.scripted:
        transcriber = None
    elif args.transcriber == "mlx":
        transcriber = MlxWhisperTranscriber(args.whisper_model or str(DEFAULT_MLX_MODEL), args.language)
        transcriber.warm_up()
    elif args.transcriber == "apple":
        transcriber = AppleSpeechTranscriber(args.language)
    else:
        transcriber = WhisperTranscriber(args.whisper_model, args.language)
    config = DetectorConfig(
        silence_ms=args.silence_ms,
        max_utterance_seconds=args.max_utterance_seconds,
        threshold_multiplier=args.threshold_multiplier,
    )
    device: int | str | None = args.input_device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    recorder = VoiceActivityRecorder(config, input_device=device)

    while True:
        samples = recorder.capture()
        if samples is None:
            continue

        if transcriber is None:
            text = "Someone spoke nearby."
        else:
            LOG.info("Transcribing...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                wav_path = wav_file.name
            try:
                write_wav(wav_path, samples, config.sample_rate)
                text = transcriber.transcribe(wav_path)
            finally:
                os.unlink(wav_path)

        if not text:
            LOG.info("No words recognized; returning to listening")
            continue
        print(f"Visitor> {text}")
        LOG.info("Generating response...")
        reply_language = transcriber.last_language if transcriber else args.language
        reply = responder.reply(text, language=reply_language)
        if not reply:
            print("Clown> (silent: no usable reply)")
            continue
        print(f"Clown> {reply}")
        if not args.no_speech:
            LOG.info("Speaking (microphone paused)...")
        speaker.speak(reply)
        time.sleep(args.cooldown_seconds)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    install_interrupt_watchdog()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if not args.scripted and not args.llm_model:
        if not DEFAULT_LLM_MODEL.is_file():
            raise SystemExit(
                f"Default model not found: {DEFAULT_LLM_MODEL}\nDownload it with:\n  "
                f"{DEFAULT_LLM_DOWNLOAD}\nor pass --llm-model PATH."
            )
        args.llm_model = str(DEFAULT_LLM_MODEL)
    if args.tts == "kokoro" and args.language != "en" and not args.no_speech:
        raise SystemExit("--tts kokoro currently requires --language en")
    if args.voice_effect:
        allowed = PIPER_EFFECT_NAMES if args.tts == "piper" else KOKORO_EFFECTS
        if args.tts == "system" or args.voice_effect not in allowed:
            raise SystemExit(
                f"--voice-effect {args.voice_effect} does not apply to --tts {args.tts}"
            )

    try:
        responder = (
            ScriptedResponder()
            if args.scripted
            else LlamaResponder(
                args.llm_model,
                curated_polish=args.curated_polish,
                memory_seconds=args.memory_seconds,
            )
        )
        if args.no_speech:
            speaker = SilentSpeaker()
        elif args.tts == "piper":
            voice = args.voice or DEFAULT_PIPER_VOICE
            voice_path = Path(voice) if voice.endswith(".onnx") else MODELS_DIR / "piper" / f"{voice}.onnx"
            speaker = PiperSpeaker(
                str(voice_path),
                effect="clean" if args.plain_voice else (args.voice_effect or "clean"),
                speed=args.piper_speed,
                prosody=args.prosody,
            )
        elif args.tts == "kokoro":
            speaker = KokoroSpeaker(
                voice=args.voice or "am_fenrir",
                speed=args.kokoro_speed,
                creepy=not args.plain_voice,
                effect=args.voice_effect or "possessed",
            )
        else:
            speaker = MacOSSaySpeaker(
                args.voice, args.speech_rate, creepy=not args.plain_voice
            )
        if args.text_only:
            run_text_loop(responder, speaker, language=args.language)
        else:
            run_voice_loop(args, responder, speaker)
    except KeyboardInterrupt:
        print("\nThe clown has gone back to sleep.")
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
