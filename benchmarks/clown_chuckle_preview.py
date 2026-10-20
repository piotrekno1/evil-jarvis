"""Render the clown's spoken laugh versus its recorded laugh replacement.

Run: uv run python benchmarks/clown_chuckle_preview.py
"""

from pathlib import Path

from evil_jarvis.persona import load_persona
from evil_jarvis.tts.piper import PiperSpeaker, voice_path


def main():
    persona = load_persona("clown")
    settings = persona.voice
    output = Path("recordings/clown-chuckle")
    output.mkdir(parents=True, exist_ok=True)
    speaker = PiperSpeaker(
        str(voice_path(settings.voice)),
        effect=settings.effect,
        prosody=settings.prosody,
        vocalization_audio=persona.vocalization_audio,
        vocalization_position="trigger",  # demonstrate exact cue positions
        vocalization_triggers=settings.vocalization_triggers,
        vocalization_probability=1.0,
        seed=7,
    )
    cases = {
        "start": "Hehe! Podejdź bliżej. Kurtyna już na ciebie czeka.",
        "middle": "Ale nie jestem tu sam... hehe... ktoś jeszcze liczy twoje kroki.",
        "end": "Oczywiście, że jestem! Ale nie jestem tu sam... hehe.",
    }
    for name, text in cases.items():
        path = output / f"{name}.wav"
        speaker.render(text, path)
        print(path.resolve())
    speaker.vocalization_position = "end"
    speaker.vocalization_probability = 0
    path = output / "before-spoken-hehe.wav"
    speaker.render(cases["end"], path)
    print(path.resolve())


if __name__ == "__main__":
    main()
