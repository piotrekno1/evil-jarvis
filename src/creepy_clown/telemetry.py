"""Optional WildEdge telemetry.

Off unless the `telemetry` extra is installed and WILDEDGE_DSN is set. By
default only timings, token counts, model metadata and outcomes are sent.
Visitor text and audio leave the machine only with an explicit
`--wildedge-capture text|audio`.

Telemetry must never stop the show: every call is wrapped so a WildEdge
failure is logged and ignored.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
from pathlib import Path
import time
import uuid

LOG = logging.getLogger("creepy_clown")
CAPTURE_LEVELS = ("off", "text", "audio")
SUMMARY_CHARS = 500


class NullSpan:
    """Stands in for a WildEdge span when telemetry is off."""

    input_summary: str | None = None
    output_summary: str | None = None

    def set_attributes(self, **attributes) -> None:
        del attributes

    def fail(self, detail: str | None = None) -> None:
        del detail


def scalar_attributes(attributes: dict) -> dict:
    """WildEdge span attributes must be scalars; lists become a joined string and a count."""
    flat = {}
    for key, value in attributes.items():
        if isinstance(value, (list, tuple)):
            flat[key] = ",".join(str(item) for item in value)
            flat[f"{key}_count"] = len(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def _never_raise(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - telemetry is best effort
            LOG.debug("WildEdge %s failed: %s", method.__name__, exc)
            return None

    return wrapper


class Telemetry:
    """No-op telemetry; `start()` returns a live one when WildEdge is configured."""

    enabled = False
    capture = "off"

    @property
    def capture_text(self) -> bool:
        return self.capture in ("text", "audio")

    @property
    def capture_audio(self) -> bool:
        return self.capture == "audio"

    def exchange(self, conversation_id: str | None = None):
        del conversation_id
        return contextlib.nullcontext(NullSpan())

    def span(self, name: str, kind: str = "custom"):
        del name, kind
        return contextlib.nullcontext(NullSpan())

    def summarize(self, span, *, input: str | None = None, output: str | None = None) -> None:
        """Put visitor or clown text on a span, only when text capture is on."""
        if not self.capture_text:
            return
        if input is not None:
            span.input_summary = input[:SUMMARY_CHARS]
        if output is not None:
            span.output_summary = output[:SUMMARY_CHARS]

    def register_llm(self, llama, load_ms: int):
        del llama, load_ms
        return None

    def register_transcriber(self, model_id: str, model_format: str):
        del model_id, model_format
        return None

    def track_transcription(self, handle, **kwargs) -> None:
        del handle, kwargs

    def track_generation(self, handle, **kwargs) -> None:
        del handle, kwargs

    def close(self) -> None:
        pass


class WildEdgeTelemetry(Telemetry):
    enabled = True

    def __init__(self, client, capture: str = "off"):
        import wildedge

        self._we = wildedge
        self._client = client
        self.capture = capture
        self.run_id = f"clown-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self._exchanges = 0

    @contextlib.contextmanager
    def exchange(self, conversation_id: str | None = None):
        """One visitor utterance and the clown's answer: a trace with one root span."""
        self._exchanges += 1
        with self._client.trace(run_id=self.run_id, conversation_id=conversation_id):
            with self._client.span(
                kind="agent_step", name="exchange", step_index=self._exchanges
            ) as span:
                yield span

    def span(self, name: str, kind: str = "custom"):
        return self._client.span(kind=kind, name=name)

    @_never_raise
    def register_llm(self, llama, load_ms: int):
        from wildedge.integrations.gguf import GGUF_HANDLE_ATTR, detect_accelerator

        if getattr(llama, GGUF_HANDLE_ATTR, None) is not None:
            # Run under `wildedge run --integrations gguf`: the SDK already
            # tracks every call, so manual tracking would count them twice.
            LOG.info("WildEdge gguf integration is active; skipping manual LLM tracking")
            return None
        handle = self._client.register_model(llama, source="local", auto_instrument=False)
        accelerator = detect_accelerator(llama)
        handle.detected_accelerator = accelerator
        handle.track_load(
            duration_ms=load_ms,
            accelerator=accelerator,
            memory_bytes=Path(llama.model_path).stat().st_size,
        )
        return handle

    @_never_raise
    def register_transcriber(self, model_id: str, model_format: str):
        family = "whisper" if "whisper" in model_id else None
        return self._client.register_model(
            None, model_id=model_id, source="local", family=family, model_format=model_format
        )

    @_never_raise
    def track_transcription(
        self,
        handle,
        *,
        duration_ms: int,
        audio_seconds: float,
        sample_rate: int,
        transcript: str,
        language: str | None = None,
        wav_path: str | None = None,
        error: Exception | None = None,
    ) -> None:
        if handle is None:
            return
        if error is not None:
            handle.track_error(
                error_code="UNKNOWN", error_message=str(error)[:500], duration_ms=duration_ms
            )
            return
        attachments = []
        if self.capture_audio and wav_path:
            attachments.append(
                self._we.Attachment(
                    content_type="audio/wav",
                    role="input",
                    data=Path(wav_path).read_bytes(),
                    name="utterance.wav",
                )
            )
        if self.capture_text and transcript:
            attachments.append(
                self._we.Attachment(
                    content_type="text/plain",
                    role="output",
                    data=transcript.encode(),
                    name="transcript.txt",
                )
            )
        handle.track_inference(
            duration_ms=duration_ms,
            input_modality="audio",
            output_modality="text",
            input_meta=self._we.AudioInputMeta(
                duration_ms=round(audio_seconds * 1000),
                sample_rate=sample_rate,
                channels=1,
                format="wav",
                source="microphone",
            ),
            output_meta=None,
            attachments=attachments or None,
            context={"language": language} if language else None,
        )

    @_never_raise
    def track_generation(
        self,
        handle,
        *,
        duration_ms: int,
        messages: list[dict[str, str]],
        params: dict,
        result: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        if handle is None:
            return
        if error is not None:
            handle.track_error(
                error_code="UNKNOWN", error_message=str(error)[:500], duration_ms=duration_ms
            )
            return
        usage = result.get("usage") or {}
        choice = (result.get("choices") or [{}])[0]
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        attachments = None
        if self.capture_text:
            content = str((choice.get("message") or {}).get("content", ""))
            attachments = [
                self._we.Attachment(
                    content_type="application/json",
                    role="input",
                    data=json.dumps(messages, ensure_ascii=False).encode(),
                    name="messages.json",
                ),
                self._we.Attachment(
                    content_type="text/plain",
                    role="output",
                    data=content.encode(),
                    name="reply.txt",
                ),
            ]
        handle.track_inference(
            duration_ms=duration_ms,
            input_modality="text",
            output_modality="generation",
            input_meta=self._we.TextInputMeta(
                token_count=tokens_in,
                prompt_type="chat",
                turn_index=sum(1 for m in messages if m["role"] == "user"),
            ),
            output_meta=self._we.GenerationOutputMeta(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens_per_second=(
                    round(tokens_out / duration_ms * 1000, 1)
                    if tokens_out and duration_ms > 0
                    else None
                ),
                stop_reason=choice.get("finish_reason"),
            ),
            generation_config=self._we.GenerationConfig(
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_tokens"),
                repetition_penalty=params.get("repeat_penalty"),
            ),
            attachments=attachments,
        )

    @_never_raise
    def close(self) -> None:
        # The Ctrl-C watchdog force-quits after 2 s; anything unsent is kept on
        # disk by the SDK's offline persistence and sent on the next start.
        self._client.close(timeout=1.5)


def start(capture: str = "off", integrations=(), app_version: str | None = None) -> Telemetry:
    """Return live telemetry when WILDEDGE_DSN is set, otherwise a no-op."""
    if capture not in CAPTURE_LEVELS:
        raise ValueError(f"Unknown capture level: {capture}")
    if not os.environ.get("WILDEDGE_DSN"):
        if capture != "off":
            raise ValueError("--wildedge-capture needs WILDEDGE_DSN to be set")
        return Telemetry()
    try:
        import wildedge
    except ImportError:
        LOG.warning(
            "WILDEDGE_DSN is set but wildedge-sdk is not installed; running without "
            "telemetry. Install it with: uv sync --extra telemetry"
        )
        return Telemetry()

    # attachments_enabled is passed explicitly so WILDEDGE_ATTACHMENTS_ENABLED
    # in the environment can never turn uploads on by itself.
    client = wildedge.init(
        integrations=list(integrations),
        app_version=app_version,
        attachments_enabled=capture != "off",
    )
    LOG.info("WildEdge telemetry on (capture: %s)", capture)
    if capture != "off":
        what = "transcripts, replies and visitor voice clips" if capture == "audio" else (
            "transcripts and replies"
        )
        LOG.warning(
            "WildEdge capture is ON: %s are uploaded. Make sure the visitor notice says so.",
            what,
        )
    return WildEdgeTelemetry(client, capture)
