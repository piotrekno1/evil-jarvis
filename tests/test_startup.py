import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evil_jarvis import download, startup
from evil_jarvis.cli import parse_args
from evil_jarvis.persona import load_persona


def args_for(*argv):
    return parse_args([*argv, "--env-file", "/nonexistent"])


class EnvFileTests(unittest.TestCase):
    def test_sets_only_missing_variables(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as handle:
            handle.write(
                "# comment\nexport A_TEST_DSN='https://x@y/z'\nB_TEST=from-file\n\nnot a setting\n"
            )
        with mock.patch.dict(os.environ, {"B_TEST": "already-set"}, clear=False):
            loaded = startup.load_env_file(Path(handle.name))
            self.assertEqual(os.environ["A_TEST_DSN"], "https://x@y/z")
            self.assertEqual(os.environ["B_TEST"], "already-set")
        self.assertEqual(loaded, ["A_TEST_DSN"])
        self.assertNotIn("A_TEST_DSN", os.environ)  # patch.dict restored the environment

    def test_missing_file_is_fine(self):
        self.assertEqual(startup.load_env_file(Path("/nonexistent/.env")), [])


class BannerTests(unittest.TestCase):
    def test_banner_describes_the_run(self):
        args = args_for("--persona", "jarvis", "--language", "auto", "--text-only")
        text = startup.banner(args, load_persona("jarvis"), "off")
        self.assertIn("Evil Jarvis · Jarvis", text)
        self.assertIn("typed text (any language)", text)
        self.assertIn(
            "Piper pl_PL-darkman-medium, metallic effect, plain intonation; "
            "English: en_GB-alan-medium",
            text,
        )
        self.assertIn("Bielik-4.5B-v3.0-Instruct.Q8_0", text)

    def test_banner_without_speech(self):
        args = args_for("--no-speech")
        text = startup.banner(args, load_persona("clown"), "off")
        self.assertIn("off (replies printed)", text)

    def test_banner_names_the_hardware(self):
        args = args_for(
            "--stt-engine", "faster-whisper", "--whisper-model", "/m/whisper-large-v3-turbo-ct2"
        )
        with mock.patch.object(startup.stt, "faster_whisper_device", return_value="cuda"):
            self.assertIn(
                "large-v3-turbo-ct2 (faster-whisper) on the NVIDIA GPU",
                startup.banner(args, load_persona("clown"), "off"),
            )
        with mock.patch.object(startup.stt, "faster_whisper_device", return_value="cpu"):
            self.assertIn(
                "an NVIDIA GPU is recommended", startup.banner(args, load_persona("clown"), "off")
            )
        with mock.patch.object(startup.llm, "gpu_offload_supported", return_value=False):
            self.assertIn("CPU only", startup.banner(args, load_persona("clown"), "off"))

    def test_telemetry_line(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIn("set WILDEDGE_DSN", startup.telemetry_summary("off"))
        with mock.patch.dict(
            os.environ, {"WILDEDGE_DSN": "https://s@example.invalid/k"}, clear=True
        ):
            self.assertIn("transcripts and replies are uploaded", startup.telemetry_summary("text"))
            self.assertIn("timings and metadata only", startup.telemetry_summary("off"))


class RequirementTests(unittest.TestCase):
    def test_every_missing_file_is_listed(self):
        args = args_for(
            "--llm-model",
            "/nope.gguf",
            "--whisper-model",
            "/nope-whisper",
            "--voice",
            "en_US-joe-medium",
        )
        problems = "\n".join(startup.missing_requirements(args))
        self.assertIn("LLM not found: /nope.gguf (check --llm-model)", problems)
        self.assertIn("Whisper model not found: /nope-whisper (check --whisper-model)", problems)
        self.assertIn("en_US-joe-medium.onnx (run with --download)", problems)

    def test_only_default_models_and_known_voices_are_downloadable(self):
        args = args_for(
            "--llm-model",
            "/nope.gguf",
            "--whisper-model",
            "/nope-whisper",
            "--voice",
            "en_US-joe-medium",
        )
        self.assertEqual([item.what for item in startup.missing_downloads(args)], ["Piper voice"])
        with tempfile.TemporaryDirectory() as models:
            fake = Path(models)
            with (
                mock.patch.object(startup.llm, "DEFAULT_MODEL", fake / "llm.gguf"),
                mock.patch.dict(
                    startup.stt.MODELS,
                    {key: (fake / key, repo) for key, (_, repo) in startup.stt.MODELS.items()},
                ),
            ):
                args = args_for("--llm-model", str(fake / "llm.gguf"), "--no-speech")
                args.whisper_model = str(
                    startup.stt.MODELS[startup.stt.model_key(args.stt_engine)][0]
                )
                self.assertEqual(
                    [item.what for item in startup.missing_downloads(args)],
                    ["LLM", "Whisper model"],
                )

    def test_url_download_is_atomic(self):

        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "src.bin", Path(directory) / "out" / "voice.onnx"
            source.write_bytes(b"voice")
            download.urls([(source.as_uri(), target)])()
            self.assertEqual(target.read_bytes(), b"voice")
            self.assertFalse(target.with_name("voice.onnx.part").exists())

    def test_missing_packages_name_the_extras(self):
        with mock.patch.object(startup, "installed", return_value=False):
            problems = startup.missing_requirements(args_for())
        self.assertEqual(
            problems[-1], "install the missing packages with: uv sync --extra models --extra voice"
        )

    def test_text_run_needs_only_the_llm(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model:
            args = args_for("--text-only", "--no-speech", "--llm-model", model.name)
            with mock.patch.object(startup, "installed", return_value=True):
                self.assertEqual(startup.missing_requirements(args), [])

    def test_install_hints_match_the_system(self):
        with mock.patch.object(startup.sys, "platform", "linux"):
            self.assertEqual(startup.system_package("ffmpeg", "ffmpeg"), "sudo apt install ffmpeg")
        with mock.patch.object(startup.sys, "platform", "darwin"):
            self.assertEqual(
                startup.system_package("portaudio", "libportaudio2"), "brew install portaudio"
            )

    def test_mlx_is_refused_off_apple_silicon(self):
        args = args_for("--stt-engine", "mlx")
        with mock.patch.object(startup.stt, "apple_silicon", return_value=False):
            self.assertIn(
                "needs an Apple Silicon Mac", "\n".join(startup.missing_requirements(args))
            )

    def test_every_language_voice_is_checked(self):
        args = args_for("--language", "auto")
        self.assertEqual(startup.piper_voices(args), ["pl_PL-bass-high", "en_US-ryan-high"])


if __name__ == "__main__":
    unittest.main()
