# Offline Polish clown-model comparison

Date: 2026-10-02

Purpose: a deliberately small comparison for a hands-free Halloween clown,
with generation capped to avoid wasting tokens. All inference was local; the
network was used only to download open model files.

## Setup

- Hardware: Apple M2 Pro, 32 GB RAM
- Prompt: Polish-only creepy clown, one or two short sentences, no question
  repetition, no emoji, darkly funny, child-safe
- Dialogue prompts: three unusual inputs that bypass the curated intent router
- Maximum output: 48 tokens per prompt
- Temperature: 0.5

## Results

### PrismML Bonsai 4B Q1 — 546 MB

Runtime: PrismML's non-mainline llama.cpp fork. Reported generation speed was
60-92 tokens/second, but it requires a custom inference engine.

1. `Czy księżyc pamięta twój cyrk?`
   - `Nie, księżyc nie ma cyrk, ale może mieć dobre pamięć... czyli może wspominać o niezwykłe nocy?`
2. `Co słyszysz, kiedy wszyscy już śpią?`
   - `Słyszysz, że szpią, szpią, szpią – jak szczęśliwy szpiák, który nie zatrzymuje się, nawet jeśli wszyscy już śpią.`
3. `Czy klauni mają sny?`
   - `Tak, klauni mają sny, a ich sny są jak słonek w rzeźbie, nie tylko zielone, ale i zielone...`

Verdict: excellent footprint and speed, but unusable Polish grammar and meaning
for this attraction.

### Qwen3 4B Q4_K_M — 2.3 GB

Runtime: llama-cpp-python. Three generations plus one model load took about 11
seconds on the test Mac.

1. `Czy księżyc pamięta twój cyrk?`
   - `Nie, księżyc nie pamięta mojego cyrku, ale ja pamiętam, że on się z nim rozstawił.`
2. `Co słyszysz, kiedy wszyscy już śpią?`
   - `Ślyszysz ciche kichanie z zaułka i szum wodny z dali. To tylko wiatr i myślenie, jak się zniechęca.`
3. `Czy klauni mają sny?`
   - `Tak, mają, ale są bardzo strachliwe. Sną o tym, jak zniszczać domy...`

Verdict: better than Bonsai, but still grammatically unreliable and occasionally
unsuitable. Keep only as a fallback behind validation or curated routing.

### LFM2.5 2.6B Q4_K_M — 1.6 GB

Compatibility probe: a 24-token request used its entire budget on English
reasoning and never reached the Polish answer. The official model card describes
it as a pure reasoning model that always thinks before answering.

Verdict: wrong latency/token profile for a prop that needs an immediate one-line
reaction. The remaining prompts were intentionally skipped.

## Conclusion

None of these local text models should replace the hybrid response engine for
common Polish interactions. The recommended iPhone design remains:

1. fast deterministic intent matching and rotating authored responses;
2. Apple's on-device Foundation Model only for unusual inputs;
3. output validation with a curated fallback;
4. native Polish speech plus AVAudioEngine effects.

LFM2.5-Audio was not tested because its official model card lists English as
its only supported language. It therefore cannot replace the Polish speech
pipeline in this project.
