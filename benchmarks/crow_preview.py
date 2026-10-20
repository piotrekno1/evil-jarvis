"""Render all three crow-call positions without loading the LLM or microphone.

Run: uv run python benchmarks/crow_preview.py
"""

from pathlib import Path

from evil_jarvis.persona import load_persona
from evil_jarvis.tts.piper import PiperSpeaker, voice_path


def main():
    persona = load_persona("pani-kraka")
    output = Path("recordings/crow-preview")
    output.mkdir(parents=True, exist_ok=True)
    speaker = PiperSpeaker(
        str(voice_path(persona.voice.voice)),
        effect=persona.voice.effect,
        speed=persona.voice.speed,
        prosody=persona.voice.prosody,
        vocalization="crow",
        vocalization_probability=1.0,
        seed=7,
    )
    text = "Pani Kraka, jeśli łaska. Miałam własną wieżę, zanim dostałam ten dziób."
    for position in ("start", "between", "end"):
        speaker.vocalization_position = position
        path = output / f"pani-kraka-{position}.wav"
        speaker.render(text, path)
        print(path.resolve())


if __name__ == "__main__":
    main()
