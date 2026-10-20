"""ffmpeg effect chains for Piper voices."""

from __future__ import annotations

PIPER_RATE = 22_050


def pitch(factor: float, rate: int = PIPER_RATE) -> str:
    """Shift pitch and formants by `factor` while keeping the original speed."""
    return f"asetrate={rate * factor:.0f},aresample={rate},atempo={1 / factor:.4f}"


def crow_effect(*, harsh: bool = False) -> str:
    """Nasal resonance and throat rasp, with a clear branch for consonants."""
    factor, drive, depth = (1.24, 4.0, 0.52) if harsh else (1.16, 2.6, 0.32)
    return (
        f"{pitch(factor)},highpass=f=190,lowpass=f=7200,"
        "equalizer=f=1250:t=q:w=1.1:g=5,asplit=2[clear][rasp];"
        "[clear]volume=0.72[c];"
        f"[rasp]highpass=f=450,lowpass=f=3600,volume={drive},"
        "asoftclip=type=tanh:threshold=0.65:oversample=2,"
        f"tremolo=f=47:d={depth},volume=0.28[r];"
        "[c][r]amix=inputs=2:normalize=0:duration=first,"
        "aecho=0.8:0.35:19:0.12,alimiter=limit=0.92"
    )


# Metallic recipe: dry voice + ring-modulated copy (robot buzz) + a comb of
# 3-13 ms echoes with high feedback (the "inside a metal pipe" ring).
PIPER_EFFECTS: dict[str, str] = {
    "clean": "anull",
    # A smaller, nasal voice with a little throat rasp. Keep consonants in the
    # dry branch; saturation and fast, shallow tremolo texture the other branch.
    # Speech and the inserted call go through this same chain and room response.
    "crow": crow_effect(),
    "crow-harsh": crow_effect(harsh=True),
    "metallic": (
        f"{pitch(0.9)},highpass=f=90,asplit=3[dry][ring][comb];"
        "[dry]volume=0.6[d];"
        "[ring]aeval=val(0)*sin(2*PI*60*t):c=same,volume=0.85[r];"
        "[comb]aecho=0.8:0.85:4|7|11:0.55|0.45|0.35,volume=0.7[m];"
        "[d][r][m]amix=inputs=3:normalize=0:duration=first,"
        "equalizer=f=2400:t=q:w=1.2:g=4,"
        "aecho=0.8:0.3:120:0.18,alimiter=limit=0.92"
    ),
    # Demon recipe: pitched down, with ring modulation, a crushed comb and a
    # sub-octave on separate branches. A clean 1.8-6.5 kHz copy of the voice
    # keeps consonants clear; the distortion stays out of that band.
    "metallic-demon": (
        f"{pitch(0.82)},highpass=f=70,asplit=5[dry][ring][comb][sub][clear];"
        "[dry]volume=0.8[d];"
        "[clear]highpass=f=1800,lowpass=f=6500,volume=0.7[c];"
        "[ring]aeval=val(0)*sin(2*PI*38*t):c=same,volume=0.5[r];"
        "[comb]aecho=0.8:0.9:3|5|9|13:0.6|0.5|0.4|0.3,"
        "acrusher=bits=8:mix=0.2:samples=2,volume=0.35[m];"
        f"[sub]{pitch(0.75)},lowpass=f=900,volume=0.35[s];"
        "[r][m][s]amix=inputs=3:normalize=0:duration=first,"
        "flanger=delay=1.5:depth=2:regen=25:speed=0.6[fx];"
        "[d][c][fx]amix=inputs=3:normalize=0:duration=first,"
        "equalizer=f=110:t=q:w=1:g=3,equalizer=f=3000:t=q:w=1:g=3,"
        "acompressor=threshold=0.1:ratio=3:attack=5:release=80,"
        "aecho=0.8:0.3:90:0.08,alimiter=limit=0.92"
    ),
}


def piper_effect_chain(effect: str) -> str:
    """Full ffmpeg -af chain: effect, then loudness normalization."""
    if effect not in PIPER_EFFECTS:
        raise ValueError(
            f"Unknown Piper voice effect {effect!r}; choose from {', '.join(PIPER_EFFECTS)}"
        )
    return (
        f"aresample={PIPER_RATE},aformat=channel_layouts=mono,{PIPER_EFFECTS[effect]},"
        f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={PIPER_RATE}"
    )


EFFECT_NAMES = tuple(PIPER_EFFECTS)
