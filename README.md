# Creepy Clown

A local Python prototype for a Halloween clown that listens for any nearby
voice, transcribes it offline, creates a short in-character response with a
local model, and speaks it. Microphone audio is ignored while the clown speaks,
so it cannot trigger itself.

This is a desktop interaction prototype, not an iPhone deployment. Once the
timing and character feel right, the same state machine can be implemented in
Swift with Apple's Speech, Foundation Models, and AVFoundation frameworks.

## Demo: Polish speech in, Polish voice out

```bash
brew install portaudio ffmpeg
uv sync --extra models --extra polish-voice
./demo.sh            # talk to the clown (deep Piper voice)
./demo.sh demon      # metallic-demon voice
./demo.sh text       # type instead of talking
./demo.sh silent     # talk, replies printed only
./demo.sh --help     # all options
```

Pipeline: Whisper large-v3-turbo (Apple GPU) → Bielik 4.5B → Piper `pl_PL-bass-high`,
with an optional metallic effect. `demo.sh` checks the model files and prints the
download command for anything missing. The Piper voice is fetched with:

```bash
mkdir -p models/piper
for ext in onnx onnx.json; do
  curl -L -o models/piper/pl_PL-bass-high.$ext \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/bass/high/pl_PL-bass-high.$ext
done
```

Useful knobs: `--voice-effect clean|metallic|metallic-demon`, `--piper-speed 1.15`,
`--voice pl_PL-darkman-medium`, `--prosody maniacal|plain`, `--memory-seconds 90`.

The default `maniacal` intonation splits each reply at punctuation and speaks every
phrase with its own random pitch, speed and loudness (giddy jumps, rushed bursts,
sudden whispers, a shouted ending on "!"), with uneven pauses and extra Piper
timing variation. The Polish prompt also asks Bielik for exclamations, trailing
"..." and the odd "hehe", which the intonation follows. `benchmarks/tts_pl_samples.py` renders every voice
and effect to `samples/polish-voices/` for side-by-side listening.

## Interaction loop

```text
LISTENING -> RECORDING -> TRANSCRIBING -> GENERATING -> SPEAKING -> COOLDOWN
     ^                                                               |
     +---------------------------------------------------------------+
```

- Ambient noise is measured once at startup.
- About 250 ms of sustained sound starts an utterance.
- About 900 ms of silence ends it.
- Utterances shorter than 300 ms are discarded.
- Microphone audio is ignored during speech playback (the stream stays open; restarting it can deadlock CoreAudio).
- Audio and transcripts are not retained.

## Quick start: timing test without AI

The scripted mode only needs NumPy, PortAudio, and `sounddevice`. On macOS:

```bash
brew install portaudio
uv sync
uv run creepy-clown --scripted
```

Say something after the `Listening...` message. Press Control-C to stop.

Use `--list-devices` if Python selects the wrong microphone. A specific input
device can be selected with `--input-device 2`.

## Full offline mode

Install the optional model backends:

```bash
uv sync --extra models
```

Download the two model files once. These commands store them inside the
git-ignored `models/` directory:

```bash
uv run --extra models hf download mlx-community/whisper-large-v3-turbo \
  --local-dir models/whisper-large-v3-turbo-mlx

uv run --extra models hf download speakleash/Bielik-4.5B-v3.0-Instruct-GGUF \
  Bielik-4.5B-v3.0-Instruct.Q8_0.gguf --local-dir models
```

Then start the complete offline pipeline:

```bash
uv run --extra models creepy-clown --voice Zosia
```

Replies come from Bielik 4.5B (a Polish-trained model) by default; pass
`--llm-model PATH` to try another GGUF model. Add `--no-speech` to print replies
instead of speaking them, and `--text-only` to type instead of talking.

Speech recognition defaults to Whisper large-v3-turbo on the Apple GPU
(`--transcriber mlx`). In the 2026-10-17 benchmark it picked the right reply
for 97% of 60 Polish clips at about 0.6 s per clip. Transcripts that loop,
are impossibly long for the clip, or contain non-Latin script are discarded as
Whisper hallucinations. The previous CPU recognizer is still available with
`--transcriber whisper --whisper-model "$PWD/models/whisper-small-ct2"`.

The clown expects Polish by default; fixing the language makes recognition a
little faster and more predictable. Use `--language en` for English or
`--language auto` to let Whisper detect the visitor's language.

### Test Apple's on-device speech recognition

On an Apple Silicon Mac, select the Apple transcriber. The first run builds a
tiny native helper and asks for Speech Recognition permission. This mode sets
`requiresOnDeviceRecognition`, so it fails instead of silently using a network
recognizer when the Polish on-device recognizer is unavailable.

```bash
uv sync --extra models

uv run --extra models creepy-clown \
  --transcriber apple \
  --language pl \
  --llm-model "$PWD/models/Qwen3-4B-Q4_K_M.gguf" \
  --voice Zosia
```

No `--whisper-model` is needed in Apple mode. Switch back to
`--transcriber whisper` to compare it with the existing recognizer.

On the development Mac tested on 2026-10-02, the native API successfully
transcribed the English control sample but reported that forced on-device
recognition was unavailable for `pl_PL`. Keep Whisper selected for offline
Polish on this Mac; availability can differ by OS, device, and installed speech
assets.
The Whisper path must be a local CTranslate2-format model directory accepted by
`faster-whisper`. The LLM must be a local GGUF instruct model accepted by
`llama-cpp-python`. The supplied Mac prototype uses a quantized 4B model; the
eventual iPhone build will use Apple's on-device Foundation Model instead.

Polish replies are generated by the model from a Polish prompt with a few
style examples. Each reply is checked (length, English, foreign characters,
emoji, stage directions, echoing the visitor, copying an example) and
regenerated up to three times; if every attempt fails, the clown stays silent.
The log line `Reply source:` shows where each reply came from. The earlier
keyword-matched handwritten lines remain available with `--curated-polish`.

The clown remembers the last 6 exchanges, so follow-up questions work ("A ile
masz lat?" after "Jak masz na imię?"). The memory is cleared once nobody has
spoken for 90 s, so the next group starts fresh; change this with
`--memory-seconds 300`, or turn memory off with `--memory-seconds 0`. Replies that
repeat an earlier reply (or start with the same three words) are regenerated.

Useful options:

```text
--language pl             default; `en` for English, `auto` to detect it
--voice Zosia             installed Polish macOS voice
--speech-rate 145         words per minute
--plain-voice             disable the creepy pitch and echo effects
--silence-ms 900          silence that ends an utterance
--max-utterance-seconds 10
--cooldown-seconds 1.0
--debug                    print detector levels
```

For English, `Reed` generally sounds less clinical than `Samantha` with the
creepy effect chain:

```text
--language en --voice Reed --speech-rate 135
```

The effect uses a mild pitch reduction, a quiet lower shadow voice, and subtle
room echo. It intentionally avoids strong tremolo, which emphasizes the robotic
quality of macOS speech synthesis.

For substantially more natural English speech, use the optional local Kokoro
neural voice. Its model downloads once and is then cached for offline use:

```bash
uv sync --extra models --extra neural-voice

uv run --extra models --extra neural-voice creepy-clown \
  --transcriber apple \
  --language en \
  --llm-model "$PWD/models/Qwen3-4B-Q4_K_M.gguf" \
  --tts kokoro \
  --voice am_fenrir \
  --kokoro-speed 0.98 \
  --voice-effect possessed
```

Try `am_puck` for a lighter, more chaotic character. Kokoro is English-only in
this prototype; Polish continues to use the system voice. Faster delivery and
punctuation-driven pauses make these voices sound more like a live performer
than the earlier slow audiobook-style preset.

`--voice-effect possessed` keeps the natural voice in front while adding a
pitch-lowered double, a quiet crushed "mask" layer, chorus, and short echoes.
Use `--voice-effect demonic` for two deeper shadow voices, stronger low
resonance, unstable movement, and a longer room tail. Use `--voice-effect
subtle` for a mostly natural voice, or `--plain-voice` for clean Kokoro output.

For a text-only model and voice test, use:

```bash
uv run --extra models creepy-clown --text-only \
  --llm-model "$PWD/models/Qwen3-4B-Q4_K_M.gguf" \
  --language pl --voice Zosia
```

The macOS voice effect is only a prototype. Its pitch, layering, filtering, and
echo stages map directly to AVAudioEngine nodes in the eventual iPhone app;
the final iOS build will not depend on `say`, `ffmpeg`, or `afplay`.

## Development

The project pins Python 3.12 in `.python-version`, and `uv.lock` records the
complete dependency resolution. Run the dependency-free test suite with:

```bash
uv run python -m unittest discover -s tests -v
```

## Practical setup

Place the microphone opening where fabric and foam do not cover it. Keep the
computer powered, disable sleep and notifications, and start the program before
the attraction opens. The threshold adapts to the measured room noise, but the
`--threshold-multiplier` option can make it more or less sensitive.

Even though processing is local and nothing is saved, put up a notice telling
visitors that the clown listens and generates replies.

## Polish speech-to-text benchmark

Compares Whisper (small, large-v3-turbo on CPU and on the Apple GPU) with
Cactus Whistle on your own recordings. Clips stay in the git-ignored
`recordings/` folder.

```bash
uv sync --extra models --extra asr-bench
uv run --extra asr-bench clown-asr-bench record --speaker ania            # 30 scripted lines
uv run --extra asr-bench clown-asr-bench record --speaker kid1 --free     # unscripted; you type what was said
uv run --extra asr-bench clown-asr-bench record --speaker ania --condition noise
uv run --extra asr-bench clown-asr-bench compare
```

`compare` prints and saves a summary to `benchmarks/asr/results/`. "Right
intent" is what matters for this attraction: whether the keyword router picks
the correct reply category from the transcript. Whistle's anonymous usage
telemetry is disabled by the benchmark.
