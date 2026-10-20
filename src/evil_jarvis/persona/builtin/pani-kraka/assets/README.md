## Crow recording

`crow.wav` is converted from the high-quality MP3 preview of
[Crow Caw by qubodup](https://freesound.org/people/qubodup/sounds/813115/),
published under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The source credits Mish7913's “Hooded Crow: Panic Cawing” (CC0).

Converted to 22,050 Hz, mono, 16-bit PCM with ffmpeg. Playback trims quiet edges,
adds short fades and matches the level to the speech. With the `crow` or
`crow-harsh` effect, the recording is lowered in pitch before it passes through
the same voice effect as the speech. The bundled source WAV is unchanged.

## Idle sounds

Idle cawing reuses `crow.wav`. `idle-beak.wav` and `idle-feathers.wav` are
original, procedural effects: two short decaying clicks and three filtered-noise
rustles. They are stylized bird sounds, not field recordings. Regenerate them
with `uv run python benchmarks/generate_crow_idle.py`.
