"""WildEdge telemetry.

Sends nothing unless WILDEDGE_DSN is set: without it the SDK client drops every
event. By default only timings, token counts, model metadata and outcomes are
sent. Visitor text and audio leave the machine only with an explicit
`--wildedge-capture text|audio`.

The LLM and Whisper calls are tracked by hand until the SDK instruments
llama.cpp chat completions (wildedge-python#64) and Whisper (#65).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import wildedge
from wildedge.integrations.gguf import GGUF_HANDLE_ATTR, detect_accelerator

logger = logging.getLogger(__name__)
CAPTURE_LEVELS = ("off", "text", "audio")
SUMMARY_CHARS = 500
ERROR_CHARS = 500


def scalar_attributes(attributes: dict) -> dict:
    """Lists become a joined string and a count: span attributes must be scalars.

    See wildedge-python#61.
    """
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


def text_attachment(role: str, name: str, text: str) -> wildedge.Attachment:
    return wildedge.Attachment(content_type="text/plain", role=role, data=text.encode(), name=name)


def attachment_filter(capture: str):
    """Enforce the app's capture mode for automatic SDK integrations too."""

    def allowed(attachments):
        if capture == "off":
            return []
        return [
            item
            for item in attachments
            if item.content_type.startswith("text/")
            or item.content_type == "application/json"
            or (capture == "audio" and item.content_type.startswith("audio/"))
        ]

    return allowed


class Telemetry:
    def __init__(
        self, client: wildedge.WildEdge, capture: str = "off", *, auto_transcription: bool = False
    ):
        self.client = client
        self.auto_transcription = auto_transcription
        self.capture = capture
        self.run_id = f"clown-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self._exchanges = 0

    @property
    def capture_text(self) -> bool:
        return self.capture in ("text", "audio")

    @property
    def capture_audio(self) -> bool:
        return self.capture == "audio"

    @contextmanager
    def exchange(self, conversation_id: str | None = None):
        """One visitor utterance and the character's answer: a trace with one root span."""
        self._exchanges += 1
        with (
            self.client.trace(run_id=self.run_id, conversation_id=conversation_id),
            self.client.span(
                kind="agent_step", name="exchange", step_index=self._exchanges
            ) as span,
        ):
            yield span

    def span(self, name: str, kind: str = "custom"):
        return self.client.span(kind=kind, name=name)

    def summarize(self, span, *, input: str | None = None, output: str | None = None) -> None:
        """Put visitor or character text on a span, only when text capture is on."""
        if not self.capture_text:
            return
        if input is not None:
            span.input_summary = input[:SUMMARY_CHARS]
        if output is not None:
            span.output_summary = output[:SUMMARY_CHARS]

    def register_llm(self, llama, load_ms: int):
        if getattr(llama, GGUF_HANDLE_ATTR, None) is not None:
            # Under `wildedge run --integrations gguf` the SDK tracks every call already.
            logger.info("WildEdge gguf integration is active; skipping manual LLM tracking")
            return None
        handle = self.client.register_model(llama, source="local", auto_instrument=False)
        handle.detected_accelerator = detect_accelerator(llama)
        handle.track_load(
            duration_ms=load_ms,
            accelerator=handle.detected_accelerator,
            memory_bytes=Path(llama.model_path).stat().st_size,
        )
        return handle

    def register_transcriber(self, model_id: str, model_format: str):
        if self.auto_transcription:
            return None
        family = "whisper" if "whisper" in model_id else None
        return self.client.register_model(
            None, model_id=model_id, source="local", family=family, model_format=model_format
        )

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
                error_code="UNKNOWN",
                error_message=str(error)[:ERROR_CHARS],
                duration_ms=duration_ms,
            )
            return
        attachments = []
        if self.capture_audio and wav_path:
            attachments.append(
                wildedge.Attachment(
                    content_type="audio/wav",
                    role="input",
                    data=Path(wav_path).read_bytes(),
                    name="utterance.wav",
                )
            )
        if self.capture_text and transcript:
            attachments.append(text_attachment("output", "transcript.txt", transcript))
        handle.track_inference(
            duration_ms=duration_ms,
            input_modality="audio",
            output_modality="text",
            input_meta=wildedge.AudioInputMeta(
                duration_ms=round(audio_seconds * 1000),
                sample_rate=sample_rate,
                channels=1,
                format="wav",
                source="microphone",
            ),
            attachments=attachments or None,
            context={"language": language} if language else None,
        )

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
                error_code="UNKNOWN",
                error_message=str(error)[:ERROR_CHARS],
                duration_ms=duration_ms,
            )
            return
        usage = result.get("usage") or {}
        choice = (result.get("choices") or [{}])[0]
        tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        attachments = None
        if self.capture_text:
            content = str((choice.get("message") or {}).get("content", ""))
            attachments = [
                wildedge.Attachment(
                    content_type="application/json",
                    role="input",
                    data=json.dumps(messages, ensure_ascii=False).encode(),
                    name="messages.json",
                ),
                text_attachment("output", "reply.txt", content),
            ]
        handle.track_inference(
            duration_ms=duration_ms,
            input_modality="text",
            output_modality="generation",
            input_meta=wildedge.TextInputMeta(
                token_count=tokens_in,
                prompt_type="chat",
                turn_index=sum(1 for m in messages if m["role"] == "user"),
            ),
            output_meta=wildedge.GenerationOutputMeta(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens_per_second=round(tokens_out / duration_ms * 1000, 1)
                if tokens_out and duration_ms > 0
                else None,
                stop_reason=choice.get("finish_reason"),
            ),
            generation_config=wildedge.GenerationConfig(
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_tokens"),
                repetition_penalty=params.get("repeat_penalty"),
            ),
            attachments=attachments,
        )

    def close(self) -> None:
        # The Ctrl-C watchdog force-quits after 2 s. The SDK keeps anything
        # unsent on disk and sends it on the next start.
        self.client.close(timeout=1.5)
