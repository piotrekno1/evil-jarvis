import os
from unittest import mock

import pytest
import wildedge
from click.testing import CliRunner

from evil_jarvis import cli, conversation, telemetry

DSN = "https://s@example.invalid/k"
MODEL_RESULT = {
    "choices": [{"message": {"content": "Mam."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 40, "completion_tokens": 12},
}


def off():
    """Telemetry that sends nothing, even if the shell has WILDEDGE_DSN set."""
    with mock.patch.dict(os.environ, {}, clear=True):
        return telemetry.Telemetry(wildedge.WildEdge())


def live(capture):
    return telemetry.Telemetry(mock.MagicMock(), capture)


class FakeHandle:
    def __init__(self):
        self.inferences = []
        self.errors = []

    def track_inference(self, **kwargs):
        self.inferences.append(kwargs)

    def track_error(self, **kwargs):
        self.errors.append(kwargs)


def start(*argv, env=None):
    """Run the CLI up to telemetry setup; returns (result, wildedge.init mock)."""
    with (
        mock.patch.dict(os.environ, env or {}, clear=True),
        mock.patch.object(cli, "missing_requirements", return_value=[]),
        mock.patch.object(cli, "install_interrupt_watchdog"),
        mock.patch.object(cli, "run"),
        mock.patch.object(cli.wildedge, "init") as init,
    ):
        result = CliRunner().invoke(cli.main, ["--text-only", "--env-file", "/nonexistent", *argv])
    return result, init


def generation_event(capture):
    handle = FakeHandle()
    live(capture).track_generation(
        handle,
        duration_ms=500,
        messages=[{"role": "user", "content": "Masz cukierka?"}],
        params={"temperature": 0.7, "max_tokens": 60},
        result=MODEL_RESULT,
    )
    return handle.inferences[0]


@pytest.fixture
def transcription_event(tmp_path):
    wav = tmp_path / "utterance.wav"
    wav.write_bytes(b"RIFF fake")

    def track(capture):
        handle = FakeHandle()
        live(capture).track_transcription(
            handle,
            duration_ms=600,
            audio_seconds=1.5,
            sample_rate=16000,
            transcript="Masz cukierka?",
            wav_path=str(wav),
        )
        return handle.inferences[0]

    return track


def attachment_names(event):
    return None if event["attachments"] is None else [a.name for a in event["attachments"]]


# Starting up


def test_capture_without_dsn_is_refused():
    result, init = start("--wildedge-capture", "text")
    assert result.exit_code == 2
    assert "needs WILDEDGE_DSN" in result.output
    init.assert_not_called()


def test_environment_cannot_turn_on_attachments():
    result, init = start(env={"WILDEDGE_DSN": DSN, "WILDEDGE_ATTACHMENTS_ENABLED": "1"})
    assert result.exit_code == 0, result.output
    assert init.call_args.kwargs["attachments_enabled"] is False


def test_no_dsn_sends_nothing():
    quiet = off()
    assert quiet.client.noop
    with quiet.exchange() as span:
        span.set_attributes(outcome="spoke")


# What gets captured


@pytest.mark.parametrize(
    ("capture", "names"), [("off", None), ("text", ["messages.json", "reply.txt"])]
)
def test_generation_attachments(capture, names):
    assert attachment_names(generation_event(capture)) == names


@pytest.mark.parametrize(
    ("capture", "names"),
    [
        ("off", None),
        ("text", ["transcript.txt"]),
        ("audio", ["utterance.wav", "transcript.txt"]),
    ],
)
def test_transcription_attachments(transcription_event, capture, names):
    assert attachment_names(transcription_event(capture)) == names


def test_generation_metadata():
    event = generation_event("off")
    assert event["output_meta"].tokens_out == 12
    assert event["output_meta"].tokens_per_second == 24.0
    assert event["generation_config"].temperature == 0.7


@pytest.mark.parametrize(("capture", "summary"), [("off", None), ("text", "visitor words")])
def test_summaries_only_with_capture(capture, summary):
    span = mock.Mock(input_summary=None)
    live(capture).summarize(span, input="visitor words")
    assert span.input_summary == summary


def test_lists_are_flattened():
    assert telemetry.scalar_attributes({"rejections": ["English", "empty"], "attempts": 3}) == {
        "rejections": "English,empty",
        "rejections_count": 2,
        "attempts": 3,
    }


# Responder and conversation


def test_each_attempt_is_tracked_and_rejections_recorded(responder_with):
    responder = responder_with(["Yes, and the candy is yours.", "Mam, ale gryzie."])
    handle = FakeHandle()
    responder._llm_handle = handle
    responder._telemetry = mock.Mock()
    responder._telemetry.track_generation = lambda h, **kw: h.track_inference(**kw)
    assert responder.reply("Masz cukierka?", language="pl") == "Mam, ale gryzie."
    assert len(handle.inferences) == 2
    assert responder.last_reply_info["attempts"] == 2
    assert responder.last_reply_info["rejections"] == ["English"]
    assert responder.last_reply_info["source"] == "model"


def test_conversation_id_changes_after_quiet(responder_with):
    now = [0.0]
    responder = responder_with(["Mam jeden.", "Mam dwa.", "Mam trzy."], clock=lambda: now[0])
    first = responder.upcoming_conversation_id()
    responder.reply("Masz cukierka?", language="pl")
    now[0] = 30.0
    assert responder.upcoming_conversation_id() == first
    responder.reply("A ile masz lat?", language="pl")
    now[0] = 200.0
    second = responder.upcoming_conversation_id()
    assert second != first
    responder.reply("Kim jesteś?", language="pl")
    assert responder._conversation_id == second


def test_exchange_outcomes(responder_with):
    spans = []
    quiet = off()

    class Recorder(telemetry.Telemetry):
        def __init__(self):
            super().__init__(quiet.client)

        def exchange(self, conversation_id=None):
            spans.append(("exchange", conversation_id))
            return super().exchange(conversation_id)

        def span(self, name, kind="custom"):
            spans.append((name, kind))
            return super().span(name, kind)

    responder = responder_with(["Mam jeden."])
    with (
        mock.patch("builtins.input", side_effect=["Masz cukierka?", EOFError]),
        mock.patch("builtins.print"),
    ):
        conversation.run_text_loop(responder, conversation.SilentSpeaker(), Recorder(), "pl")
    assert [s[0] for s in spans] == ["exchange", "reply", "speak"]
    assert spans[0][1] == responder._conversation_id


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("off", []),
        ("text", ["text/plain", "application/json"]),
        ("audio", ["text/plain", "application/json", "audio/wav"]),
    ],
)
def test_automatic_attachment_filter(mode, expected):
    attachments = [
        wildedge.Attachment(content_type=kind, role="input", data=b"data")
        for kind in ("text/plain", "application/json", "audio/wav", "image/png")
    ]
    assert [
        item.content_type for item in telemetry.attachment_filter(mode)(attachments)
    ] == expected


def test_automatic_transcription_does_not_register_manual_handle():
    client = mock.MagicMock()
    observed = telemetry.Telemetry(client, auto_transcription=True)
    assert observed.register_transcriber("whisper", "mlx") is None
    client.register_model.assert_not_called()


@pytest.mark.parametrize(
    ("engine", "integration"), [("mlx", "mlx_whisper"), ("faster-whisper", "faster_whisper")]
)
@pytest.mark.parametrize("supported", [False, True])
def test_cli_selects_whisper_with_manual_fallback(engine, integration, supported):
    available = {integration} if supported else set()
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(cli, "missing_requirements", return_value=[]),
        mock.patch.object(cli, "install_interrupt_watchdog"),
        mock.patch.object(cli, "run"),
        mock.patch.object(cli.wildedge.WildEdge, "SUPPORTED_INTEGRATIONS", available),
        mock.patch.object(cli.wildedge, "init") as init,
        mock.patch.object(cli, "Telemetry") as wrapper,
    ):
        result = CliRunner().invoke(
            cli.main, ["--env-file", "/nonexistent", "--stt-engine", engine, "--no-speech"]
        )
    assert result.exit_code == 0, result.output
    assert init.call_args.kwargs["integrations"] == ([integration] if supported else [])
    assert wrapper.call_args.kwargs["auto_transcription"] is supported
