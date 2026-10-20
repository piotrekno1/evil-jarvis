import pytest
from click.testing import CliRunner

from evil_jarvis.cli import main, parse_args
from evil_jarvis.tts.effects import PIPER_EFFECTS, piper_effect_chain
from evil_jarvis.tts.piper import PiperSpeaker


def test_clown_default_voice():
    args = parse_args([])
    assert args.voice == "pl_PL-bass-high"
    assert args.voice_effect == "metallic"


@pytest.mark.parametrize("effect", PIPER_EFFECTS)
def test_effect_chains_end_with_loudness_normalization(effect):
    assert "loudnorm=" in piper_effect_chain(effect)


def test_demon_effect_uses_ring_modulation():
    assert "aeval=val(0)*sin" in piper_effect_chain("metallic-demon")


def test_unknown_effect_is_rejected():
    with pytest.raises(ValueError, match="Unknown Piper voice effect"):
        piper_effect_chain("spooky")


def test_missing_voice_explains_how_to_download():
    with pytest.raises(ValueError, match="Piper voice not found"):
        PiperSpeaker("/not/a/voice.onnx")


def test_unknown_effect_is_a_cli_error():
    result = CliRunner().invoke(main, ["--text-only", "--voice-effect", "possessed"])
    assert result.exit_code == 2
    assert "possessed" in result.output
