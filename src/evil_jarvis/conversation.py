"""The pipeline: listen → transcribe → reply → speak, one trace per exchange.

Everything here gets ready-made parts (recorder, transcriber, responder,
speaker); cli.py builds them from the command-line options.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from evil_jarvis import telemetry as wildedge_telemetry
from evil_jarvis.tts import SilentSpeaker
from evil_jarvis.wav import write_wav

logger = logging.getLogger(__name__)


def speaker_label(responder) -> str:
    persona = getattr(responder, "persona", None)
    return persona.display_name if persona else "Reply"


def conversation_of(responder) -> str | None:
    upcoming = getattr(responder, "upcoming_conversation_id", None)
    return upcoming() if upcoming else None


def respond_and_speak(text, language, responder, speaker, *, telemetry, exchange, started) -> None:
    """Generate the character's reply to `text` and speak it, inside one exchange trace."""
    with telemetry.span("reply", kind="agent_step") as span:
        reply = responder.reply(text, language=language)
        info = getattr(responder, "last_reply_info", None) or {}
        span.set_attributes(language=language, **wildedge_telemetry.scalar_attributes(info))
        telemetry.summarize(span, input=text, output=reply)
    if not reply:
        print(f"{speaker_label(responder)}> (silent: no usable reply)")
        exchange.set_attributes(outcome="silent")
        return
    print(f"{speaker_label(responder)}> {reply}")
    exchange.set_attributes(
        outcome="spoke", response_latency_ms=round((time.perf_counter() - started) * 1000)
    )
    if not isinstance(speaker, SilentSpeaker):
        logger.info("Speaking (microphone paused)...")
    with telemetry.span("speak") as span:
        span.set_attributes(reply_chars=len(reply), speaker=type(speaker).__name__)
        speaker.speak(reply, language=language)


def run_text_loop(responder, speaker, telemetry, language: str | None = None) -> None:
    """Visitor lines come from stdin (--text-only)."""
    while True:
        try:
            text = input("Visitor> ").strip()
        except EOFError:  # Ctrl-D or end of piped input
            print()
            return
        if not text:
            continue
        started = time.perf_counter()
        with telemetry.exchange(conversation_of(responder)) as exchange:
            exchange.set_attributes(input="text")
            respond_and_speak(
                text,
                language,
                responder,
                speaker,
                telemetry=telemetry,
                exchange=exchange,
                started=started,
            )


def run_voice_loop(
    recorder, transcriber, responder, speaker, *, telemetry, asr_handle=None, cooldown_seconds=1.0
) -> None:
    """Visitor speech comes from the microphone."""
    sample_rate = recorder.config.sample_rate
    while True:
        samples = recorder.capture()
        if samples is None:
            continue
        started = time.perf_counter()  # the visitor just stopped talking
        audio_seconds = len(samples) / sample_rate

        with telemetry.exchange(conversation_of(responder)) as exchange:
            exchange.set_attributes(input="voice", audio_seconds=round(audio_seconds, 2))
            logger.info("Transcribing...")
            text = transcribe(
                transcriber,
                samples,
                sample_rate,
                telemetry=telemetry,
                handle=asr_handle,
                audio_seconds=audio_seconds,
            )
            if not text:
                logger.info("No words recognized; returning to listening")
                outcome = "discarded" if transcriber.last_discarded else "no_words"
                exchange.set_attributes(outcome=outcome)
                continue
            print(f"Visitor> {text}")
            logger.info("Generating response...")
            respond_and_speak(
                text,
                transcriber.last_language,
                responder,
                speaker,
                telemetry=telemetry,
                exchange=exchange,
                started=started,
            )
        time.sleep(cooldown_seconds)


def transcribe(transcriber, samples, sample_rate, *, telemetry, handle, audio_seconds) -> str:
    with telemetry.span("transcribe") as span:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        try:
            write_wav(wav_path, samples, sample_rate)
            began = time.perf_counter()
            try:
                text = transcriber.transcribe(wav_path)
            except Exception as exc:
                telemetry.track_transcription(
                    handle,
                    duration_ms=round((time.perf_counter() - began) * 1000),
                    audio_seconds=audio_seconds,
                    sample_rate=sample_rate,
                    transcript="",
                    error=exc,
                )
                raise
            telemetry.track_transcription(
                handle,
                duration_ms=round((time.perf_counter() - began) * 1000),
                audio_seconds=audio_seconds,
                sample_rate=sample_rate,
                transcript=text,
                language=transcriber.last_language,
                wav_path=wav_path,  # read only when audio capture is on
            )
        finally:
            Path(wav_path).unlink()
        span.set_attributes(
            language=transcriber.last_language,
            words=len(text.split()),
            discarded=transcriber.last_discarded,
        )
        telemetry.summarize(span, output=text)
    return text
