import os
import wave
import zipfile
from pathlib import Path
from unittest import mock

import pytest
import wildedge
from click.testing import CliRunner

from evil_jarvis import conversation, telemetry
from evil_jarvis.cli import main, parse_args
from evil_jarvis.persona import builtin_dir, builtin_personas, load_persona
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


@pytest.fixture
def persona_bundle(tmp_path):
    folder = tmp_path / "talking-pumpkin"
    assets = folder / "assets"
    assets.mkdir(parents=True)
    with wave.open(str(assets / "laugh.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x01\x00" * 160)
    config = MINIMAL.replace("speed = 1.1", 'speed = 1.1\nvocalization = "assets/laugh.wav"')
    (folder / "persona.toml").write_text(config, encoding="utf-8")
    return folder


@pytest.mark.parametrize("as_folder", [True, False])
def test_persona_asset_is_relative_to_toml_not_working_directory(
    persona_bundle, monkeypatch, as_folder, tmp_path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    path = persona_bundle if as_folder else persona_bundle / "persona.toml"
    persona = load_persona(str(path))
    assert persona.voice.vocalization == "assets/laugh.wav"
    assert persona.vocalization_audio == (persona_bundle / "assets/laugh.wav").read_bytes()


def test_missing_persona_asset_names_the_file(persona_bundle):
    (persona_bundle / "assets/laugh.wav").rename(persona_bundle / "assets/other.wav")
    with pytest.raises(ValueError, match=r"vocalization file not found: assets/laugh.wav"):
        load_persona(str(persona_bundle))


def test_invalid_persona_wav_fails_at_load(persona_bundle):
    (persona_bundle / "assets/laugh.wav").write_bytes(b"not audio")
    with pytest.raises(ValueError, match="invalid vocalization WAV"):
        load_persona(str(persona_bundle))


def test_custom_trigger_is_automatically_explained_to_model(persona_bundle, responder_with):
    config = persona_bundle / "persona.toml"
    text = config.read_text().replace(
        'vocalization = "assets/laugh.wav"',
        'vocalization = "assets/laugh.wav"\n'
        'vocalization_position = "trigger"\n'
        'vocalization_triggers = ["snicker"]',
    )
    config.write_text(text)
    persona = load_persona(str(persona_bundle))
    responder = responder_with(["Snicker. Welcome, visitor."], persona=persona)
    assert responder.reply("Hello?", "en") == "Snicker. Welcome, visitor."
    messages, _ = responder._model.calls[0]
    assert '"snicker"' in messages[0]["content"]
    assert "recording instead of speech" in messages[0]["content"]


@pytest.mark.parametrize("asset", ["/tmp/laugh.wav", "../laugh.wav", "assets/laugh.mp3"])
def test_persona_assets_must_be_local_wavs(persona_bundle, asset):
    config = persona_bundle / "persona.toml"
    config.write_text(config.read_text().replace("assets/laugh.wav", asset))
    with pytest.raises(ValueError, match=r"relative \.wav path inside the persona"):
        load_persona(str(persona_bundle))


def test_persona_folder_needs_a_toml(tmp_path):
    with pytest.raises(ValueError, match="Persona file not found"):
        load_persona(str(tmp_path))


def test_builtin_bundles_and_assets_load_from_zip_resources(tmp_path):
    names = builtin_personas()
    expected = load_persona("pani-kraka").vocalization_audio
    expected_idle = load_persona("pani-kraka").idle_audio
    root = Path(str(builtin_dir()))
    archive_path = tmp_path / "personas.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, "builtin/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        resource_root = zipfile.Path(archive, "builtin/")
        with mock.patch("evil_jarvis.persona.builtin_dir", return_value=resource_root):
            assert builtin_personas() == names
            persona = load_persona("pani-kraka")
    # Audio remains available after the archive/resource context is closed.
    assert persona.vocalization_audio == expected
    assert persona.idle_audio == expected_idle
    assert len(persona.idle_audio) == 3


def test_idle_asset_is_relative_to_persona_folder(persona_bundle, monkeypatch, tmp_path):
    config = persona_bundle / "persona.toml"
    config.write_text(config.read_text() + '\n[idle]\nsounds = ["assets/laugh.wav"]\n')
    monkeypatch.chdir(tmp_path)
    persona = load_persona(str(persona_bundle))
    assert persona.idle_audio == ((persona_bundle / "assets/laugh.wav").read_bytes(),)


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ('sounds = "laugh.wav"', "list of WAV paths"),
        ('sounds = ["x.wav", "x.wav"]', "duplicate"),
        ('sounds = ["../laugh.wav"]', "relative .wav path"),
        ('sounds = ["missing.wav"]', "file not found"),
        ("gain = 0", "gain"),
        ("gain = nan", "finite number"),
        ("after_seconds = true", "finite number"),
        ("after_seconds = -1", "delays"),
        ("min_interval_seconds = 0", "delays"),
        ("max_interval_seconds = 1", "maximum interval"),
        ("gaim = 0.4", "gaim"),
    ],
)
def test_invalid_idle_config_is_reported(persona_file, setting, message):
    with pytest.raises(ValueError, match=message):
        load_persona(persona_file(MINIMAL + "\n[idle]\n" + setting))


def test_idle_clip_must_be_short(persona_bundle):
    config = persona_bundle / "persona.toml"
    config.write_text(config.read_text() + '\n[idle]\nsounds = ["assets/laugh.wav"]\n')
    with wave.open(str(persona_bundle / "assets/laugh.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x01\x00" * 32001)
    with pytest.raises(ValueError, match="at most 2 seconds"):
        load_persona(str(persona_bundle))


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
