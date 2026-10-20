<p align="center"><img src="assets/logo.jpg" alt="Evil Jarvis logo" width="200"></p>

# Evil Jarvis

Turn a Halloween prop into a character that talks back. Put a microphone and a
speaker inside it, and visitors can have a conversation with it. Everything runs
on your own machine, so there is no cloud service or account to set up.

Each character has a folder with a TOML definition and optional audio assets.
The TOML sets the personality, example lines, voice and effects. Four come built
in, and all speak Polish and English:

- `clown`: a creepy clown
- `jarvis`: a sinister smart-home assistant
- `pani-kraka`: a quarrelsome witch trapped in a crow, bargaining for shiny things (Polish by default)
- `madame-nevermore`: an elegant witch-crow offering eerie, whimsical fortunes (English by default)

## Requirements

- Apple Silicon Mac, or Linux with an NVIDIA GPU (CPU-only works, slowly)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- PortAudio and ffmpeg: `brew install portaudio ffmpeg` or `sudo apt install libportaudio2 ffmpeg`

## Install

```bash
uv sync --extra models --extra voice
```

On Linux, rebuild llama.cpp with CUDA. faster-whisper uses the GPU when CUDA and
cuDNN are installed (versions in its README), and falls back to Whisper `small`
on the CPU otherwise.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra models --extra voice --reinstall-package llama-cpp-python
```

Then download the models and voices, about 6 GB into `models/`:

```bash
uv run evil-jarvis --download
```

`--download` fetches what the given options need and then checks the setup.
Pass the options you plan to run with, for example
`--persona jarvis --language auto --download`, so their voices come too.
`--check` runs only the check.

## Usage

Start the default character, the clown. It listens on the default microphone
and answers through the default speaker:

```bash
uv run evil-jarvis
```

Pick another character:

```bash
uv run evil-jarvis --persona jarvis
```

Try either witch-crow with `--persona pani-kraka` or `--persona madame-nevermore`.
Pani Kraka uses a nasal, raspy crow voice with plain delivery. Madame Nevermore
uses a clean voice and takes her time. Download the selected persona's voices first:

```bash
uv run evil-jarvis --persona pani-kraka --language auto --download
uv run evil-jarvis --persona madame-nevermore --language auto --download
```

Pani Kraka considers a recorded crow call when the model writes `kra` or `caw`
(with a few common repeated spellings also accepted). Her prompts encourage
an occasional cue in a fitting place; replies without one have no added caw.
The cues work in Polish and English. Her default `"varied"` placement chooses the
opening, ending or a pause between complete sentences, regardless of where the
model wrote the cue. It never interrupts a word or splits a sentence.
The call's loudness follows the speech, with short pauses at the joins. The
`crow` effect raises the speech's pitch, adds nasal resonance and a light rasp,
and applies the same processing to speech and calls. This stylizes the voice;
it cannot correct words the speech model pronounces incorrectly.

For automatic placement instead, set `vocalization_position` in the TOML `[voice]`
section to `"start"`, `"end"`, `"between"` or `"random"`. In those modes,
`vocalization_probability` controls frequency (`0` disables calls, `1` always
plays one). `"between"` falls back to the end when there is only one sentence.
The `vocalization = "assets/crow.wav"` setting
references the persona's [CC0 recording](src/evil_jarvis/persona/builtin/pani-kraka/assets/README.md).
Audio paths resolve relative to the TOML file, so copying a persona folder
includes its sounds. Any persona can reference its own mono, 16-bit PCM WAV.

To compare all three placements without using the microphone or LLM:

```bash
uv run python benchmarks/crow_preview.py
```

This writes three Polish voice previews to `recordings/crow-preview/`.

Compare the original clean voice against two crow transformations on identical
speech audio, using `uv run python benchmarks/crow_voice_preview.py`. This writes
previews to `recordings/crow-transform/`. Pani Kraka defaults to `crow`; try
`--voice-effect crow-harsh` for stronger pitch and rasp, or `--voice-effect clean`
to remove the transformation. All effects require ffmpeg except `clean`.

Answer in whichever language each visitor speaks, instead of only the
persona's language:

```bash
uv run evil-jarvis --language auto
```

Try a persona without a microphone or speaker. You type lines and the replies
are printed:

```bash
uv run evil-jarvis --text-only --no-speech
```

Each run prints which models, devices and voices it uses. Ctrl-C quits.

The most useful options are below. `uv run evil-jarvis --help` lists all of them.

| Option | Purpose |
|---|---|
| `--persona NAME\|FOLDER\|FILE` | Built-in persona, persona folder or `.toml` path. `--list-personas` lists them. |
| `--language CODE` | `pl`, `en`, ... or `auto`. Default from the persona. |
| `--voice`, `--voice-effect`, `--prosody`, `--piper-speed` | Override the persona's voice. Effects: `clean`, `crow`, `crow-harsh`, `metallic`, `metallic-demon`. Prosody: `plain`, `maniacal`. |
| `--llm-model PATH` | Any GGUF instruct model. |
| `--stt-engine` | `mlx` (Apple GPU) or `faster-whisper`. Chosen automatically. |
| `--memory-seconds N` | Conversation memory, cleared after N seconds of silence. Default 90, `0` disables it. |
| `--input-device`, `--output-device` | Audio devices. `--list-devices` lists them. |
| `--silence-ms`, `--threshold-multiplier` | Voice detection tuning. |
| `--env-file PATH` | Environment file. Default `.env` in the project. |

## Personas

Copy `src/evil_jarvis/persona/builtin/clown/`, whose `persona.toml` documents
the settings, and run it with `--persona path/to/folder`. Standalone TOML files
also work with `--persona path/to/file.toml`. Unknown settings and missing or
invalid audio assets are reported when the persona loads.

Personas with sounds have this layout:

```text
pani-kraka/
  persona.toml
  assets/
    crow.wav
    README.md      # recording source and license
```

To give another character a sound, place its WAV inside that character's folder
and add `vocalization = "assets/laugh.wav"` under `[voice]`. The speech engine
loads the referenced audio without needing code specific to the character.

The clown replaces written laugh cues with its own recorded chuckle. The printed
reply keeps `hehe`; the speaker chooses a natural position for the recording
instead of pronouncing the cue. Without a cue, no laugh is added. This works in
either language.

```toml
[voice]
vocalization = "assets/chuckle.wav"
vocalization_position = "varied"
vocalization_triggers = ["hehe", "haha", "heh"]
```

Triggers are literal strings matched case-insensitively as whole words. Both
`"trigger"` and `"varied"` modes only consider replies containing a cue.
`"trigger"` keeps the first cue's exact position; `"varied"` chooses an opening,
ending or sentence boundary, with a slight preference for endings. It lowers
the chance of repeating the previous position and prevents three identical
placements in a row. Placement history is shared across language voices.
`vocalization_probability` applies to these replies too; suppressed cues are
removed silently instead of being spoken. At most one recording plays per reply.
The built-in clown also lists common separated and
longer laugh spellings. The model automatically receives instructions based on
each persona's configured triggers: use a suitable cue occasionally, at most one
per reply. Both the clown and Pani Kraka use a 25% playback chance, followed by
at least two sound-free replies. This cooldown is shared across languages.
Whether and where a cue appears is the model's choice; playback applies the
frequency limit even if the model adds cues to every reply. Change these
settings under `[voice]` to tune it:

```toml
vocalization_probability = 0.25
vocalization_min_gap_replies = 2
```
The clown's [CC0 chuckle](src/evil_jarvis/persona/builtin/clown/assets/README.md)
gets the same metallic effect as its speech. Render a before/after comparison
with `uv run python benchmarks/clown_chuckle_preview.py`.

To hear opening, middle and ending placements for both characters, run
`uv run python benchmarks/vocalization_variety_preview.py`. These audition clips
force a sound to demonstrate placement; they do not change live frequency.

```toml
name = "witch"
display_name = "Witch"
language = "en"

[voice]
voice = "en_US-lessac-high"       # any Piper voice in models/piper
effect = "metallic"
prosody = "plain"

[voice.by_language]               # optional voice per language
pl = "pl_PL-darkman-medium"

[languages.default]               # required, used for languages without their own section
system_prompt = "You are a witch..."
examples = [{ visitor = "Who are you?", reply = "Your next ingredient." }]
attempts = 3                      # regenerate rejected replies, then stay silent
max_words = 25
checks = ["no_symbols", "no_stage_directions", "no_echo", "no_repeat"]
```

Available checks: `latin_only`, `no_symbols`, `no_stage_directions`,
`not_english`, `no_echo`, `no_echo_opening`, `no_example_copy`, `no_repeat`,
`no_reused_phrase`. The clown uses `no_reused_phrase` to reject three-word phrases
recycled from recent replies, unless the visitor is explicitly asking about that
phrase. Repetition retries ask the model for a new idea and different wording.
Empty replies are
always rejected.

## Telemetry

Off by default. To send metrics to [WildEdge](https://wildedge.dev), set
`WILDEDGE_DSN` in `.env`:

```bash
echo 'WILDEDGE_DSN=...' > .env
```

By default it sends only timings, token counts, model metadata and outcomes.
`--wildedge-capture text` also uploads transcripts and replies, and `audio` also
uploads voice clips. If you capture, tell visitors they are being recorded.

## Development

```bash
uv run pytest
uv run ruff format .
uv run ruff check --fix .
```

Tests need no models or audio devices. Ruff settings are in `pyproject.toml`.
Layout:

```text
src/evil_jarvis/
  cli.py           options and wiring
  conversation.py  listen, transcribe, reply, speak
  stt/             speech to text (mlx-whisper, faster-whisper)
  llm/             replies (llama.cpp)
  tts/             Piper voices, effects, prosody, playback
  persona/         persona loading, reply checks, built-in folders and audio assets
  bench/asr.py     speech recognition benchmark (jarvis-asr-bench)
```

`benchmarks/` holds model comparisons and voice sample scripts.
