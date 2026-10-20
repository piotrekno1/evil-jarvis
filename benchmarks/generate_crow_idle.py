"""Generate original, stylized beak-click and feather-rustle idle effects.

Run: uv run python benchmarks/generate_crow_idle.py
No third-party audio is used by this script.
"""

from pathlib import Path

import numpy as np

from evil_jarvis.wav import write_wav

RATE = 22_050
RUSTLE_LOW_HZ = 900
RUSTLE_HIGH_HZ = 7500
OUTPUT = Path(__file__).resolve().parents[1] / "src/evil_jarvis/persona/builtin/pani-kraka/assets"


def main():
    rng = np.random.default_rng(17)
    clicks = np.zeros(round(RATE * 0.6), dtype=np.float32)
    time = np.arange(round(RATE * 0.045)) / RATE
    for offset, gain in ((0.08, 0.75), (0.28, 0.6)):
        transient = 0.65 * np.sin(2 * np.pi * 2100 * time) + 0.35 * rng.normal(size=len(time))
        transient *= np.exp(-time / 0.006) * np.minimum(time / 0.0007, 1) * gain
        start = round(offset * RATE)
        clicks[start : start + len(time)] += transient
    write_wav(OUTPUT / "idle-beak.wav", clicks, RATE)

    time = np.arange(round(RATE * 0.95)) / RATE
    noise = rng.normal(size=len(time))
    frequencies = np.fft.rfftfreq(len(noise), 1 / RATE)
    spectrum = np.fft.rfft(noise)
    spectrum[(frequencies < RUSTLE_LOW_HZ) | (frequencies > RUSTLE_HIGH_HZ)] = 0
    rustle = np.fft.irfft(spectrum, n=len(noise))
    envelope = sum(np.exp(-0.5 * ((time - middle) / 0.055) ** 2) for middle in (0.19, 0.43, 0.69))
    rustle *= envelope
    rustle *= 0.4 / np.max(np.abs(rustle))
    write_wav(OUTPUT / "idle-feathers.wav", rustle.astype(np.float32), RATE)


if __name__ == "__main__":
    main()
