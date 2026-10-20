import os
from pathlib import Path
from unittest import mock

import pytest
import wildedge
from click.testing import CliRunner

from evil_jarvis import conversation, telemetry
from evil_jarvis.cli import main, parse_args
from evil_jarvis.persona import builtin_personas, load_persona
from evil_jarvis.tts.per_language import PerLanguageSpeaker
from evil_jarvis.tts.piper import PiperSpeaker, voice_files

MINIMAL = """
name = "butler"
display_name = "Butler"
language = "en"

[voice]
voice = "en_US-ryan-high"
speed = 1.1

[languages.default]
system_prompt = "You are a butler."
max_words = 5
checks = ["no_symbols"]
"""


@pytest.fixture
def persona_file(tmp_path):
    """Writes persona TOML to a file and returns its path."""

    def write(text):
        path = tmp_path / "persona.toml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    return write


class Recorder:
    def __init__(self):
        self.spoken = []

    def speak(self, text, language=None):
        self.spoken.append((text, language))


# Loading


@pytest.mark.parametrize("name", builtin_personas())
def test_every_builtin_persona_loads(name):
    persona = load_persona(name)
    assert persona.profile("pl").system_prompt
    assert persona.profile("xx").system_prompt  # falls back to default


def test_builtins_include_clown_and_jarvis():
    assert {"clown", "jarvis"} <= set(builtin_personas())


def test_persona_from_a_file(persona_file):
    persona = load_persona(persona_file(MINIMAL))
    assert persona.display_name == "Butler"
    assert persona.profile("pl") is persona.languages["default"]
    assert persona.profile("en").reply_problem("one two three four five six") == (
        "too long (6 words)"
    )


@pytest.mark.parametrize(
    ("text", "mention"),
    [
        (MINIMAL.replace('display_name = "Butler"', 'dispaly_name = "Butler"'), "dispaly_name"),
        (MINIMAL.replace('"no_symbols"', '"no_emoji"'), "no_emoji"),
        (MINIMAL.replace("[languages.default]", "[languages.en]"), "languages.default"),
        (MINIMAL.replace('voice = "en_US-ryan-high"', 'voices = "en_US-ryan-high"'), "voices"),
        (MINIMAL.replace("[voice]\n", '[voice]\ntts = "kokoro"\n'), "tts"),
        (
            MINIMAL.replace("speed = 1.1", 'speed = 1.1\nby_language = "en_US-ryan-high"'),
            "by_language",
        ),
    ],
)
def test_typos_are_reported(persona_file, text, mention):
    with pytest.raises(ValueError, match=mention):
        load_persona(persona_file(text))


def test_unknown_persona_lists_the_builtins():
    with pytest.raises(ValueError, match="clown"):
        load_persona("nobody")


# Responder


def test_persona_drives_prompt_params_and_labels(responder_with):
    responder = responder_with(["Jarvis: Otworzę. Ale tylko raz."], persona=load_persona("jarvis"))
    assert responder.reply("Otwórz okno.", language="pl") == "Otworzę. Ale tylko raz."
    messages, options = responder._model.calls[0]
    assert "Jarvisem" in messages[0]["content"]
    assert options["max_tokens"] == 60


def test_default_profile_gets_a_language_rule(responder_with):
    responder = responder_with(
        ["Of course. The doors remember you."], persona=load_persona("jarvis")
    )
    responder.reply("Hello?", language="en")
    messages, _ = responder._model.calls[0]
    assert messages[0]["content"].endswith("must be in English.")
    assert messages[-1]["content"] == "Visitor: Hello?\nReply only in English."


# Speakers


def test_each_reply_uses_its_language_voice():
    polish, english = Recorder(), Recorder()
    speaker = PerLanguageSpeaker({"en": english}, default=polish)
    speaker.speak("Hello.", language="en")
    speaker.speak("Cześć.", language="pl")
    speaker.speak("Hallo.", language="de")
    assert english.spoken == [("Hello.", None)]
    assert [t for t, _ in polish.spoken] == ["Cześć.", "Hallo."]


def test_reply_language_reaches_the_speaker():
    class Responder:
        def reply(self, text, language=None):
            return "Hello."

    speaker = Recorder()
    with mock.patch.dict(os.environ, {}, clear=True):
        quiet = telemetry.Telemetry(wildedge.WildEdge())
    with quiet.exchange() as exchange:
        conversation.respond_and_speak(
            "Hi", "en", Responder(), speaker, telemetry=quiet, exchange=exchange, started=0.0
        )
    assert speaker.spoken == [("Hello.", "en")]


def test_missing_voice_names_its_own_download():
    files = voice_files(Path("models/piper/en_GB-alan-medium.onnx"))
    assert files[1] == (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        Path("models/piper/en_GB-alan-medium.onnx.json"),
    )
    assert voice_files(Path("my-voice.onnx")) is None
    with pytest.raises(ValueError, match="--download"):
        PiperSpeaker("/tmp/en_US-joe-medium.onnx")


# Command line


def test_persona_sets_language_and_voice():
    args = parse_args(["--persona", "jarvis"])
    assert (args.voice, args.voice_effect, args.prosody) == (
        "pl_PL-darkman-medium",
        "metallic",
        "plain",
    )


def test_flags_override_the_persona():
    args = parse_args(["--persona", "jarvis", "--voice-effect", "clean", "--prosody", "maniacal"])
    assert (args.voice_effect, args.prosody) == ("clean", "maniacal")


def test_persona_file_sets_language_voice_and_speed(persona_file):
    args = parse_args(["--persona", persona_file(MINIMAL)])
    assert (args.language, args.voice, args.piper_speed) == ("en", "en_US-ryan-high", 1.1)
    assert (args.voice_effect, args.prosody) == ("clean", "plain")  # defaults


@pytest.mark.parametrize(
    ("argv", "voice"),
    [
        ([], "pl_PL-bass-high"),
        (["--language", "en"], "en_US-ryan-high"),
        (["--persona", "jarvis", "--language", "en"], "en_GB-alan-medium"),
        (["--language", "de"], "pl_PL-bass-high"),  # no German voice listed
        (["--language", "en", "--voice", "x"], "x"),
    ],
)
def test_voice_follows_the_language(argv, voice):
    assert parse_args(argv).voice == voice


def test_auto_language_keeps_every_voice():
    args = parse_args(["--language", "auto"])
    assert args.voice == "pl_PL-bass-high"
    assert args.voices_by_language == {"en": "en_US-ryan-high"}
    assert parse_args(["--language", "auto", "--voice", "x"]).voices_by_language == {}


def test_list_personas():
    result = CliRunner().invoke(main, ["--list-personas"])
    assert result.exit_code == 0
    assert "jarvis" in result.output


def test_bad_persona_is_a_usage_error():
    result = CliRunner().invoke(main, ["--persona", "nobody", "--text-only"])
    assert result.exit_code == 2
    assert "Unknown persona" in result.output


@pytest.mark.parametrize("option", ["--persona", "--wildedge-capture", "--voice-effect"])
def test_help_lists_options(option):
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert option in result.output
