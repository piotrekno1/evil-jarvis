"""Render each persona's idle sounds with its configured effects and volume.

Run: uv run python benchmarks/idle_preview.py
Does not load speech models or open audio devices.
"""

from pathlib import Path

from evil_jarvis.idle import prepare_idle_clip
from evil_jarvis.persona import load_persona
from evil_jarvis.tts.playback import PLAYBACK_RATE
from evil_jarvis.wav import write_wav


def main():
    output = Path("recordings/idle-preview")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("clown", "pani-kraka"):
        persona = load_persona(name)
        for asset, audio in zip(persona.idle.sounds, persona.idle_audio, strict=True):
            samples = prepare_idle_clip(audio, persona.voice.effect or "clean", persona.idle.gain)
            path = output / f"{name}-{Path(asset).stem}.wav"
            write_wav(path, samples, PLAYBACK_RATE)
            print(f"{path.resolve()} ({len(samples) / PLAYBACK_RATE:.2f} s)")


if __name__ == "__main__":
    main()
