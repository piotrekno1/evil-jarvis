"""Audition opening, middle and ending sounds with the approved voice effects.

Run: uv run python benchmarks/vocalization_variety_preview.py
Previews force a sound on each line; the live persona frequency is unchanged.
"""

from pathlib import Path

from evil_jarvis.persona import load_persona
from evil_jarvis.tts.piper import PiperSpeaker, voice_path


def main():
    output = Path("recordings/vocalization-variety")
    output.mkdir(parents=True, exist_ok=True)
    lines = {
        "clown": "Masz ochotę na przedstawienie? Kurtyna już wybrała twoje miejsce... hehe.",
        "pani-kraka": "Ta łyżeczka należy do mnie. Nawet klątwa ma swoje zasady. Kra!",
    }
    for name, text in lines.items():
        persona = load_persona(name)
        voice = persona.voice
        speaker = PiperSpeaker(
            str(voice_path(voice.voice)),
            effect=voice.effect,
            prosody=voice.prosody,
            speed=voice.speed or 1,
            vocalization_audio=persona.vocalization_audio,
            vocalization_position="varied",
            vocalization_triggers=voice.vocalization_triggers,
            vocalization_probability=1,
            seed=1,
        )
        for index in range(3):
            path = output / f"{name}-{index + 1}.wav"
            speaker.render(text, path)
            print(f"{path.resolve()} ({speaker._vocalization_gate.last_position})")


if __name__ == "__main__":
    main()
