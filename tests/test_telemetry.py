import os
import tempfile
import unittest
from unittest import mock

import wildedge
from click.testing import CliRunner

from evil_jarvis import cli, conversation, telemetry
from evil_jarvis.llm.llama import LlamaResponder


def off():
    """Telemetry that sends nothing, even if the shell has WILDEDGE_DSN set."""
    with mock.patch.dict(os.environ, {}, clear=True):
        return telemetry.Telemetry(wildedge.WildEdge())


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)

    def create_chat_completion(self, messages, **options):
        return {
            "choices": [{"message": {"content": self.replies.pop(0)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        }


def responder_with(replies, clock=None):
    responder = object.__new__(LlamaResponder)
    responder._disable_thinking = False
    responder._model = FakeModel(replies)
    if clock:
        responder.clock = clock
    return responder


class FakeHandle:
    def __init__(self):
        self.inferences = []
        self.errors = []

    def track_inference(self, **kwargs):
        self.inferences.append(kwargs)

    def track_error(self, **kwargs):
        self.errors.append(kwargs)


class StartTests(unittest.TestCase):
    def start(self, *argv, env=None):
        """Run the CLI up to telemetry setup; returns (result, wildedge.init mock)."""
        with (
            mock.patch.dict(os.environ, env or {}, clear=True),
            mock.patch.object(cli, "missing_requirements", return_value=[]),
            mock.patch.object(cli, "install_interrupt_watchdog"),
            mock.patch.object(cli, "run"),
            mock.patch.object(cli.wildedge, "init") as init,
        ):
            result = CliRunner().invoke(
                cli.main, ["--text-only", "--env-file", "/nonexistent", *argv]
            )
        return result, init

    def test_capture_without_dsn_is_refused(self):
        result, init = self.start("--wildedge-capture", "text")
        self.assertEqual(result.exit_code, 2)
        self.assertIn("needs WILDEDGE_DSN", result.output)
        init.assert_not_called()

    def test_environment_cannot_turn_on_attachments(self):
        env = {"WILDEDGE_DSN": "https://s@example.invalid/k", "WILDEDGE_ATTACHMENTS_ENABLED": "1"}
        result, init = self.start(env=env)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(init.call_args.kwargs["attachments_enabled"], False)

    def test_no_dsn_sends_nothing(self):
        quiet = off()
        self.assertTrue(quiet.client.noop)
        with quiet.exchange() as span:
            span.set_attributes(outcome="spoke")


class CaptureTests(unittest.TestCase):
    def live(self, capture):
        return telemetry.Telemetry(mock.MagicMock(), capture)

    def generate(self, capture):
        handle = FakeHandle()
        self.live(capture).track_generation(
            handle,
            duration_ms=500,
            messages=[{"role": "user", "content": "Masz cukierka?"}],
            params={"temperature": 0.7, "max_tokens": 60},
            result=FakeModel(["Mam."]).create_chat_completion([]),
        )
        return handle.inferences[0]

    def transcribe(self, capture):
        handle = FakeHandle()
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            wav.write(b"RIFF fake")
            wav.flush()
            self.live(capture).track_transcription(
                handle,
                duration_ms=600,
                audio_seconds=1.5,
                sample_rate=16000,
                transcript="Masz cukierka?",
                wav_path=wav.name,
            )
        return handle.inferences[0]

    def test_default_sends_no_visitor_content(self):
        self.assertIsNone(self.generate("off")["attachments"])
        self.assertIsNone(self.transcribe("off")["attachments"])

    def test_generation_metadata(self):
        event = self.generate("off")
        self.assertEqual(event["output_meta"].tokens_out, 12)
        self.assertEqual(event["output_meta"].tokens_per_second, 24.0)
        self.assertEqual(event["generation_config"].temperature, 0.7)

    def test_text_capture_sends_text_but_not_audio(self):
        self.assertEqual(
            [a.name for a in self.generate("text")["attachments"]], ["messages.json", "reply.txt"]
        )
        self.assertEqual(
            [a.name for a in self.transcribe("text")["attachments"]], ["transcript.txt"]
        )

    def test_audio_capture_adds_the_voice_clip(self):
        names = [a.name for a in self.transcribe("audio")["attachments"]]
        self.assertEqual(names, ["utterance.wav", "transcript.txt"])

    def test_summaries_only_with_capture(self):
        span = mock.Mock(input_summary=None)
        self.live("off").summarize(span, input="visitor words")
        self.assertIsNone(span.input_summary)
        self.live("text").summarize(span, input="visitor words")
        self.assertEqual(span.input_summary, "visitor words")


class ScalarAttributeTests(unittest.TestCase):
    def test_lists_are_flattened(self):
        self.assertEqual(
            telemetry.scalar_attributes({"rejections": ["English", "empty"], "attempts": 3}),
            {"rejections": "English,empty", "rejections_count": 2, "attempts": 3},
        )


class ResponderTelemetryTests(unittest.TestCase):
    def test_each_attempt_is_tracked_and_rejections_recorded(self):
        responder = responder_with(["Yes, and the candy is yours.", "Mam, ale gryzie."])
        handle = FakeHandle()
        responder._llm_handle = handle
        responder._telemetry = mock.Mock()
        responder._telemetry.track_generation = lambda h, **kw: h.track_inference(**kw)
        self.assertEqual(responder.reply("Masz cukierka?", language="pl"), "Mam, ale gryzie.")
        self.assertEqual(len(handle.inferences), 2)
        self.assertEqual(responder.last_reply_info["attempts"], 2)
        self.assertEqual(responder.last_reply_info["rejections"], ["English"])
        self.assertEqual(responder.last_reply_info["source"], "model")

    def test_conversation_id_changes_after_quiet(self):
        now = [0.0]
        responder = responder_with(["Mam jeden.", "Mam dwa.", "Mam trzy."], clock=lambda: now[0])
        first = responder.upcoming_conversation_id()
        responder.reply("Masz cukierka?", language="pl")
        now[0] = 30.0
        self.assertEqual(responder.upcoming_conversation_id(), first)
        responder.reply("A ile masz lat?", language="pl")
        now[0] = 200.0
        second = responder.upcoming_conversation_id()
        self.assertNotEqual(second, first)
        responder.reply("Kim jesteś?", language="pl")
        self.assertEqual(responder._conversation_id, second)


class TextLoopTests(unittest.TestCase):
    def test_exchange_outcomes(self):
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
        self.assertEqual([s[0] for s in spans], ["exchange", "reply", "speak"])
        self.assertEqual(spans[0][1], responder._conversation_id)


if __name__ == "__main__":
    unittest.main()
