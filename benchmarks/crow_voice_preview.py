"""Compare crow transformations using identical speech audio in each version.

Run: uv run python benchmarks/crow_voice_preview.py
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from evil_jarvis.persona import load_persona
from evil_jarvis.tts.effects import piper_effect_chain
from evil_jarvis.tts.piper import PiperSpeaker, voice_path
from evil_jarvis.tts.playback import resample
from evil_jarvis.tts.pronunciation import for_piper
from evil_jarvis.tts.vocalization import join_call
from evil_jarvis.wav import read_wav, write_wav

TEXT = (
    'Ja? Tylko na "dzień dobry" i "do widzenia". '
    'Ale czasem mam ochotę zakraknąć: "zejdź mi z oczu!".'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--output", type=Path, default=Path("recordings/crow-transform"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    persona = load_persona("pani-kraka")
    speaker = PiperSpeaker(
        str(voice_path(persona.voice.voice)),
        effect="clean",
        prosody="plain",
        speed=persona.voice.speed,
        vocalization_audio=persona.vocalization_audio,
        seed=7,
    )
    text = for_piper(args.text, speaker._voice.config.espeak_voice)
    speech = speaker._render_speech(text)
    rate = speaker._voice.config.sample_rate
    # Capture the old clean reference, then compare both effects on precisely
    # the same new join. The recorded caw is lowered before the shared effect.
    reference = np.concatenate([speech, np.zeros(int(rate * 0.12)), speaker._call * 0.65])
    lowered_call = resample(speaker._call, round(rate * 0.86), rate)
    joined = join_call([speech], lowered_call, rate, boundary=1)
    with tempfile.TemporaryDirectory(prefix="crow-transform-") as directory:
        for effect, audio in (("clean", reference), ("crow", joined), ("crow-harsh", joined)):
            raw = Path(directory) / "raw.wav"
            target = args.output / f"{effect}.wav"
            write_wav(raw, audio, rate)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(raw),
                    "-af",
                    piper_effect_chain(effect),
                    str(target),
                ],
                check=True,
            )
            samples, sample_rate = read_wav(target)
            print(f"{target.resolve()} ({len(samples) / sample_rate:.2f} s)")


if __name__ == "__main__":
    main()
