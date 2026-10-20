import pytest
from click.testing import CliRunner

from evil_jarvis.cli import main
from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.persona import load_persona

CLOWN = load_persona("clown")


@pytest.mark.parametrize(
    "phrase",
    [
        "1–2 short sentences",
        "same language as the",
        "unnamed, unhinged Halloween",
        "sharp conversational",
        "never\nprefix the response with a name",
        "Never repeat or paraphrase",
        "realistic threats",
        "family Halloween attraction",
    ],
)
def test_character_prompt_constrains_length_and_safety(phrase):
    assert phrase in CLOWN.profile("en").system_prompt


def test_model_echo_is_removed():
    visitor = "Hello, is there anybody out here?"
    reply = visitor + " Only me. The empty tent is laughing."
    assert (
        LlamaResponder.strip_transcript_echo(reply, visitor)
        == "Only me. The empty tent is laughing."
    )


def test_model_character_label_is_removed():
    reply = LlamaResponder.strip_transcript_echo(
        "Grinwick: Candy? It bites back.", "Do you have candy?", CLOWN.all_labels
    )
    assert reply == "Candy? It bites back."


def test_missing_model_is_reported_as_a_cli_error():
    argv = ["--llm-model", "/not/a/real/model.gguf", "--text-only", "--env-file", "/nonexistent"]
    result = CliRunner().invoke(main, argv)
    assert result.exit_code == 1
    assert "LLM not found: /not/a/real/model.gguf" in result.output
