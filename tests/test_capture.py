"""Visitor text is sent only when opted in."""

import os
from unittest import mock

import pytest
from click.testing import CliRunner

from evil_jarvis import cli, conversation

DSN = "https://s@example.invalid/k"


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.input_summary = self.output_summary = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_attributes(self, **attributes):
        self.attributes.update(attributes)


class FakeClient:
    """Records the spans the app opens."""

    def __init__(self):
        self.spans = []

    def span(self, *, kind, name):
        span = FakeSpan(name)
        self.spans.append(span)
        return span

    def named(self, name):
        return next(s for s in self.spans if s.name == name)


def reply_with(responder_with, *, capture):
    client = FakeClient()
    responder = responder_with(["Mam jeden."])
    with client.span(kind="agent_step", name="exchange") as exchange, mock.patch("builtins.print"):
        conversation.respond_and_speak(
            "Masz cukierka?",
            "pl",
            responder,
            conversation.SilentSpeaker(),
            client=client,
            capture=capture,
            exchange=exchange,
            started=0.0,
        )
    return client.named("reply")


def transcribe_with(*, capture):
    client = FakeClient()
    transcriber = mock.Mock(last_language="pl", last_discarded=False)
    transcriber.transcribe.return_value = "Masz cukierka?"
    with mock.patch.object(conversation, "write_wav"):
        conversation.transcribe(transcriber, [0.0], 16000, client=client, capture=capture)
    return client.named("transcribe")


@pytest.mark.parametrize(
    ("env", "capture"),
    [
        ({"WILDEDGE_DSN": DSN}, False),
        ({"WILDEDGE_DSN": DSN, "WILDEDGE_ATTACHMENTS_ENABLED": "0"}, False),
        ({"WILDEDGE_DSN": DSN, "WILDEDGE_ATTACHMENTS_ENABLED": "1"}, True),
    ],
)
def test_capture_follows_the_sdk_attachment_switch(env, capture):
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(cli, "missing_requirements", return_value=[]),
        mock.patch.object(cli, "install_interrupt_watchdog"),
        mock.patch.object(cli, "run") as run,
        mock.patch.object(cli.wildedge, "init"),
    ):
        result = CliRunner().invoke(cli.main, ["--text-only", "--env-file", "/nonexistent"])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[3] is capture  # run(args, persona, client, capture)


@pytest.mark.parametrize("capture", [False, True])
def test_reply_text_only_with_capture(responder_with, capture):
    span = reply_with(responder_with, capture=capture)
    expected = ("Masz cukierka?", "Mam jeden.") if capture else (None, None)
    assert (span.input_summary, span.output_summary) == expected
    assert span.attributes["source"] == "model"


@pytest.mark.parametrize("capture", [False, True])
def test_transcript_only_with_capture(capture):
    span = transcribe_with(capture=capture)
    assert span.output_summary == ("Masz cukierka?" if capture else None)
    assert span.attributes["words"] == 2
