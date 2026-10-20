"""The evil-jarvis command: options, persona defaults, and wiring the pipeline together."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import click
import wildedge

from evil_jarvis import __version__, stt
from evil_jarvis.conversation import run_text_loop, run_voice_loop
from evil_jarvis.idle import build_idle_sounds
from evil_jarvis.llm import DEFAULT_MODEL as DEFAULT_LLM_MODEL
from evil_jarvis.llm import build_responder
from evil_jarvis.microphone import DetectorConfig, VoiceActivityRecorder
from evil_jarvis.paths import ENV_FILE
from evil_jarvis.persona import DEFAULT_PERSONA, Persona, builtin_personas, load_persona
from evil_jarvis.shutdown import install_interrupt_watchdog
from evil_jarvis.startup import (
    banner,
    load_env_file,
    missing_downloads,
    missing_requirements,
    telemetry_summary,
)
from evil_jarvis.telemetry import CAPTURE_LEVELS, Telemetry
from evil_jarvis.tts import (
    DEFAULT_VOICE,
    EFFECT_NAMES,
    PROSODY_STYLES,
    SilentSpeaker,
    build_speaker,
)
from evil_jarvis.tts.playback import AudioPlayer

logger = logging.getLogger(__name__)

EXAMPLES = """\b
Examples:
  evil-jarvis                                talk to the clown
  evil-jarvis --persona jarvis               another character (see --list-personas)
  evil-jarvis --voice-effect metallic-demon  a different voice effect
  evil-jarvis --text-only                    type instead of talking
  evil-jarvis --no-speech                    print replies instead of speaking
  evil-jarvis --language auto                Polish or English, per visitor
  evil-jarvis --check                        show the setup and what is missing
  evil-jarvis --language auto --download     fetch the models and every voice it needs
"""


def list_devices(ctx: click.Context, _param, value: bool) -> None:
    if value and not ctx.resilient_parsing:
        import sounddevice as sd

        click.echo(sd.query_devices())
        ctx.exit()


def list_personas(ctx: click.Context, _param, value: bool) -> None:
    if value and not ctx.resilient_parsing:
        for name in builtin_personas():
            click.echo(f"{name:12} {load_persona(name).description}")
        ctx.exit()


def info_flag(name: str, callback, help: str):
    return click.option(
        name, is_flag=True, expose_value=False, is_eager=True, callback=callback, help=help
    )


def parse_device(value: str | None) -> int | str | None:
    return int(value) if value and value.isdigit() else value


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
    help="Offline hands-free voice character. Voice, language and prompts come from "
    "the persona; the options below override it.",
    epilog=EXAMPLES,
)
@click.option(
    "--persona",
    default=DEFAULT_PERSONA,
    show_default=True,
    help="Built-in persona name, persona folder, or path to a .toml file.",
)
@info_flag("--list-personas", list_personas, "Show built-in personas and exit.")
@click.option(
    "--language",
    help="Spoken language code (e.g. pl, en) or auto to detect it. Default from the persona.",
)
@click.option(
    "--text-only", is_flag=True, help="Read visitor lines from stdin instead of the microphone."
)
# Speech to text
@click.option(
    "--stt-engine",
    type=click.Choice(stt.ENGINES),
    help="Speech recognition: mlx (Apple GPU) or faster-whisper (NVIDIA GPU or CPU). "
    "Default: mlx on Apple Silicon, faster-whisper elsewhere.",
)
@click.option(
    "--whisper-model", help="Whisper model directory (default: large-v3-turbo for the engine)."
)
# Text to text
@click.option(
    "--llm-model",
    help="Local GGUF instruct model (default models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf).",
)
@click.option(
    "--memory-seconds",
    type=float,
    default=90.0,
    show_default=True,
    help="Remember the conversation until nobody has spoken for this long (0 = no memory).",
)
# Text to speech
@click.option(
    "--voice", help="Piper voice in models/piper (e.g. pl_PL-bass-high) or a path to an .onnx file."
)
@click.option(
    "--voice-effect",
    type=click.Choice(EFFECT_NAMES),
    help="Voice effect. Default from the persona.",
)
@click.option(
    "--prosody", type=click.Choice(PROSODY_STYLES), help="Intonation. Default from the persona."
)
@click.option("--piper-speed", type=float, help="Speaking speed (1.2 = 20% faster).")
@click.option("--no-speech", is_flag=True, help="Print replies instead of speaking them.")
@click.option("--no-idle", is_flag=True, help="Disable the persona's idle sounds.")
# Audio devices
@click.option("--input-device", help="Microphone: sounddevice device ID or name.")
@click.option("--output-device", help="Speaker: sounddevice device ID or name.")
@info_flag("--list-devices", list_devices, "Show audio devices and exit.")
@click.option("--silence-ms", type=int, default=900, show_default=True)
@click.option("--max-utterance-seconds", type=float, default=10.0, show_default=True)
@click.option("--cooldown-seconds", type=float, default=1.0, show_default=True)
@click.option("--threshold-multiplier", type=float, default=3.0, show_default=True)
# Telemetry and startup
@click.option(
    "--wildedge-capture",
    type=click.Choice(CAPTURE_LEVELS),
    default="off",
    show_default=True,
    help="WildEdge upload of visitor content (needs WILDEDGE_DSN): off (timings and metadata "
    "only), text (transcripts and replies), audio (also voice clips).",
)
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=ENV_FILE,
    show_default=".env in the project",
    help="Read settings such as WILDEDGE_DSN from this file; variables already set win.",
)
@click.option("--check", is_flag=True, help="Show the setup and anything missing, then exit.")
@click.option(
    "--download", is_flag=True, help="Download the models and voices this run needs, then --check."
)
@click.option("--debug", is_flag=True, help="Verbose logs, including detector levels.")
@click.version_option(__version__, "-V", "--version")
def main(**options) -> None:
    args = SimpleNamespace(**options)
    persona = resolve(args)
    load_env_file(args.env_file)
    if args.wildedge_capture != "off" and not os.environ.get("WILDEDGE_DSN"):
        raise click.UsageError("--wildedge-capture needs WILDEDGE_DSN to be set")
    click.echo(banner(args, persona, telemetry_summary(args.wildedge_capture)))
    if args.download:
        fetch_missing(args)
    problems = missing_requirements(args)
    if problems:
        raise click.ClickException(
            "not ready to start:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    if args.check or args.download:
        click.echo("Everything needed is installed.")
        return
    install_interrupt_watchdog()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Persona: %s (%s)", persona.name, persona.source)

    # Before any model is created: the onnx integration (Piper) patches at load
    # time. attachments_enabled is explicit so WILDEDGE_ATTACHMENTS_ENABLED in
    # the environment can never turn uploads on by itself.
    client = wildedge.init(
        integrations=[] if args.no_speech else ["onnx"],
        app_version=__version__,
        attachments_enabled=args.wildedge_capture != "off",
    )
    telemetry = Telemetry(client, args.wildedge_capture)

    try:
        run(args, persona, telemetry)
    except KeyboardInterrupt:
        click.echo(f"\n{persona.exit_message}")
    except (RuntimeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    finally:
        telemetry.close()


def apply_persona(args, persona: Persona) -> None:
    """Fill unset options from the persona."""
    voice = persona.voice
    args.language = args.language or persona.language
    # With --language auto each reply may be in another language: keep every voice.
    args.voices_by_language = (
        dict(voice.by_language) if not args.voice and args.language == "auto" else {}
    )
    args.voice = args.voice or voice.voice_for(args.language) or DEFAULT_VOICE
    args.voice_effect = args.voice_effect or voice.effect or "clean"
    args.prosody = args.prosody or voice.prosody or "plain"
    args.piper_speed = args.piper_speed or voice.speed or 1.0


def fetch_missing(args) -> None:
    items = missing_downloads(args)
    if not items:
        click.echo("Nothing to download.")
    for item in items:
        click.echo(f"Downloading {item.what} to {item.path}")
        try:
            item.fetch()
        except ImportError as exc:
            raise click.ClickException(
                "downloads need the 'models' extra: uv sync --extra models --extra voice"
            ) from exc
        except Exception as exc:  # network and Hugging Face errors have no common base class
            raise click.ClickException(f"download of {item.path.name} failed: {exc}") from exc


def resolve(args) -> Persona:
    """Load the persona and fill every unset option, so the rest never sees a None."""
    try:
        persona = load_persona(args.persona)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--persona") from exc
    apply_persona(args, persona)
    args.llm_model = args.llm_model or str(DEFAULT_LLM_MODEL)
    args.stt_engine = args.stt_engine or stt.default_engine()
    args.whisper_model = args.whisper_model or str(stt.default_model(args.stt_engine))
    return persona


def parse_args(argv=()) -> SimpleNamespace:
    """Options as main() sees them after the persona is applied (for tests and tools)."""
    with main.make_context("evil-jarvis", list(argv)) as ctx:
        args = SimpleNamespace(**ctx.params)
    resolve(args)
    return args


def run(args, persona: Persona, telemetry) -> None:
    responder = build_responder(args.llm_model, persona, args.memory_seconds, telemetry)
    player = AudioPlayer(parse_device(args.output_device)) if not args.no_speech else None
    speaker = (
        SilentSpeaker()
        if args.no_speech
        else build_speaker(
            args.voice,
            voices_by_language=args.voices_by_language,
            effect=args.voice_effect,
            speed=args.piper_speed,
            prosody=args.prosody,
            output_device=parse_device(args.output_device),
            player=player,
            vocalization_audio=persona.vocalization_audio,
            vocalization_probability=persona.voice.vocalization_probability,
            vocalization_position=persona.voice.vocalization_position,
            vocalization_triggers=persona.voice.vocalization_triggers,
            vocalization_min_gap_replies=persona.voice.vocalization_min_gap_replies,
        )
    )
    if args.text_only:
        try:
            run_text_loop(responder, speaker, telemetry, language=args.language)
        finally:
            if player is not None:
                player.close()
        return

    hint = persona.profile(args.language).asr_hint
    transcriber = stt.build_transcriber(
        args.stt_engine, args.whisper_model, args.language, topic_hint=hint
    )
    transcriber.warm_up()
    recorder = VoiceActivityRecorder(
        DetectorConfig(
            silence_ms=args.silence_ms,
            max_utterance_seconds=args.max_utterance_seconds,
            threshold_multiplier=args.threshold_multiplier,
        ),
        input_device=parse_device(args.input_device),
    )
    asr_handle = telemetry.register_transcriber(
        *stt.telemetry_model(args.stt_engine, args.whisper_model)
    )
    try:
        idle = (
            build_idle_sounds(persona, player, args.voice_effect)
            if not args.no_idle and player is not None
            else None
        )
        run_voice_loop(
            recorder,
            transcriber,
            responder,
            speaker,
            telemetry=telemetry,
            asr_handle=asr_handle,
            cooldown_seconds=args.cooldown_seconds,
            idle=idle,
        )
    finally:
        recorder.close()
        if player is not None:
            player.close()


if __name__ == "__main__":
    main()
