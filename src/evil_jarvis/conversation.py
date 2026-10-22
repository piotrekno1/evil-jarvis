"""The pipeline: listen → transcribe → reply → speak, one trace per exchange.

Everything here gets ready-made parts (recorder, transcriber, responder,
speaker). cli.py builds them from the command-line options. Visitor text goes
on the spans only with `capture` on. Audio is never sent.
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from evil_jarvis.tts import SilentSpeaker
from evil_jarvis.wav import write_wav

logger = logging.getLogger(__name__)


def speaker_label(responder) -> str:
    persona = getattr(responder, "persona", None)
    return persona.display_name if persona else "Reply"


def conversation_of(responder) -> str | None:
    upcoming = getattr(responder, "upcoming_conversation_id", None)
    return upcoming() if upcoming else None


@contextmanager
def exchange_trace(client, responder):
    """One visitor utterance and the character's answer: a trace with one root span."""
    with (
        client.trace(conversation_id=conversation_of(responder)),
        client.span(kind="agent_step", name="exchange") as span,
    ):
        yield span


def respond_and_speak(
    text, language, responder, speaker, *, client, capture, exchange, started
) -> None:
    """Generate the character's reply to `text` and speak it, inside one exchange trace."""
    with client.span(kind="agent_step", name="reply") as span:
        reply = responder.reply(text, language=language)
        info = getattr(responder, "last_reply_info", None) or {}
        span.set_attributes(language=language, **info)
        if capture:
            span.input_summary, span.output_summary = text, reply
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
    with client.span(kind="custom", name="speak") as span:
        span.set_attributes(reply_chars=len(reply), speaker=type(speaker).__name__)
        speaker.speak(reply, language=language)


def run_text_loop(responder, speaker, client, capture, language: str | None = None) -> None:
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
        with exchange_trace(client, responder) as exchange:
            exchange.set_attributes(input="text")
            respond_and_speak(
                text,
                language,
                responder,
                speaker,
                client=client,
                capture=capture,
                exchange=exchange,
                started=started,
            )


def run_voice_loop(
    recorder,
    transcriber,
    responder,
    speaker,
    *,
    client,
    capture,
    cooldown_seconds=1.0,
    idle=None,
) -> None:
    """Visitor speech comes from the microphone."""
    sample_rate = recorder.config.sample_rate
    while True:
        samples = (
            recorder.capture(on_listening=idle.reset, on_idle=idle.next_action)
            if idle is not None
            else recorder.capture()
        )
        if samples is None:
            continue
        started = time.perf_counter()  # the visitor just stopped talking
        audio_seconds = len(samples) / sample_rate

        with exchange_trace(client, responder) as exchange:
            exchange.set_attributes(input="voice", audio_seconds=round(audio_seconds, 2))
            logger.info("Transcribing...")
            text = transcribe(transcriber, samples, sample_rate, client=client, capture=capture)
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
                client=client,
                capture=capture,
                exchange=exchange,
                started=started,
            )
        time.sleep(cooldown_seconds)


def transcribe(transcriber, samples, sample_rate, *, client, capture) -> str:
    with client.span(kind="custom", name="transcribe") as span:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        try:
            write_wav(wav_path, samples, sample_rate)
            text = transcriber.transcribe(wav_path)
        finally:
            Path(wav_path).unlink()
        span.set_attributes(
            language=transcriber.last_language,
            words=len(text.split()),
            discarded=transcriber.last_discarded,
        )
        if capture:
            span.output_summary = text
    return text
