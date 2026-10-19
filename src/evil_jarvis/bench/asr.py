"""Record Polish test clips and compare speech-to-text backends on them.

    uv run --extra asr-bench jarvis-asr-bench record --speaker ania
    uv run --extra asr-bench jarvis-asr-bench compare

Each clip is scored on word error rate, on whether the keyword router picks
the same intent from the transcript as from the reference text, and on latency.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import click

from evil_jarvis.microphone import DetectorConfig, VoiceActivityRecorder
from evil_jarvis.paths import MODELS_DIR, PROJECT_ROOT
from evil_jarvis.persona import load_persona
from evil_jarvis.persona.checks import normalize_text
from evil_jarvis.stt.faster_whisper import FasterWhisperTranscriber
from evil_jarvis.stt.mlx_whisper import MlxWhisperTranscriber
from evil_jarvis.tts.playback import AudioPlayer
from evil_jarvis.wav import write_wav

PROMPTS = PROJECT_ROOT / "benchmarks" / "asr" / "prompts.tsv"
RECORDINGS = PROJECT_ROOT / "recordings" / "asr"
RESULTS = PROJECT_ROOT / "benchmarks" / "asr" / "results"
MODELS = MODELS_DIR

# Real Polish spellings of the router's keywords, used for Whistle's keyword biasing.
WHISTLE_KEYWORDS = [
    "cukierek",
    "cukierka",
    "słodycze",
    "psikus",
    "ręce",
    "rękę",
    "dłoń",
    "kim jesteś",
    "imię",
    "boję się",
    "strach",
    "strasz",
    "halo",
    "słyszysz",
    "odezwij się",
    "cześć",
    "dzień dobry",
    "dobry wieczór",
    "hej",
    "uśmiech",
    "śmiejesz",
    "żyjesz",
    "prawdziwy",
    "ciemno",
    "do widzenia",
    "żegnaj",
    "klaun",
    "klaunie",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Prompt:
    id: str
    intent: str
    text: str


def load_prompts(path: Path = PROMPTS) -> list[Prompt]:
    with path.open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [Prompt(row["id"], row["intent"], row["text"]) for row in rows]


def manifest_path(root: Path) -> Path:
    return root / "manifest.jsonl"


def load_manifest(root: Path) -> list[dict]:
    path = manifest_path(root)
    if not path.is_file():
        return []
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # A re-recorded clip appends a new entry for the same file; keep the latest.
    latest = {entry["file"]: entry for entry in entries}
    return list(latest.values())


def append_manifest(root: Path, entry: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with manifest_path(root).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------ metrics


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation, keep Polish diacritics."""
    cleaned = "".join(
        " " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in text.casefold()
    )
    return cleaned.split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (word errors, reference word count) so totals can be pooled."""
    ref = normalize_words(reference)
    return edit_distance(ref, normalize_words(hypothesis)), len(ref)


# What a visitor most likely wants, by keyword. The benchmark asks whether a
# recognizer's transcript still gets the visitor's request right. Phrases are
# lowercase without Polish diacritics; the first intent with a match wins.
INTENTS = {
    "candy": ("cukier", "slodycz", "slodkiego"),  # cukier: cukierek, cukierka, cukierki
    "hands": ("reka", "rece", "reke", "raczk", "dlon"),
    "identity": ("kim jestes", "jak masz na imie", "twoje imie", "kto ty"),
    "fear": ("boje", "strach", "strasz"),
    "presence": ("halo", "ktos tu jest", "jest tu ktos", "slyszysz mnie", "jestes tam", "odezw"),
    "greeting": ("czesc", "dzien dobry", "dobry wieczor", "witaj", "hej"),
    "smile": ("usmiech", "smiejesz", "smiech"),
    "alive": ("zyjesz", "prawdziwy", "jestes zywy"),
    "darkness": ("ciemno", "ciemku", "sam tutaj", "co tu robisz", "dlaczego tu"),
    "goodbye": ("do widzenia", "zegnaj", "pa pa", "papa", "ide juz", "juz ide", "odchodze"),
}


def route(text: str) -> str:
    """The intent a visitor's words point to, or "none"."""
    normalized = normalize_text(text)
    for intent, phrases in INTENTS.items():
        if any(phrase in normalized for phrase in phrases):
            return intent
    return "none"


# ----------------------------------------------------------------------- backends


class FasterWhisperBackend:
    def __init__(self, model_dir: Path):

        # CPU on purpose: the labels below say "CPU int8", and results stay comparable.
        self._transcriber = FasterWhisperTranscriber(
            str(model_dir),
            "pl",
            topic_hint=load_persona("clown").profile("pl").asr_hint,
            device="cpu",
        )

    def transcribe(self, wav_path: str) -> str:
        return self._transcriber.transcribe(wav_path)


class MlxWhisperBackend:
    """The app's own MLX transcriber, so the benchmark measures what ships."""

    def __init__(self, model_dir: Path):

        self._transcriber = MlxWhisperTranscriber(str(model_dir), "pl")

    def transcribe(self, wav_path: str) -> str:
        return self._transcriber.transcribe(wav_path)


class WhistleBackend:
    def __init__(self, keywords: list[str] | None = None):
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")  # opt out of usage pings
        import needle

        self._needle = needle
        self._keywords = keywords

    def transcribe(self, wav_path: str) -> str:
        result = self._needle.transcribe(wav_path, language="pl", keywords=self._keywords)
        return str(result["text"]).strip()


def whisper_dir(name: str) -> Path:
    return MODELS / f"whisper-{name}-ct2"


BACKENDS = {
    "whisper-base": (
        lambda: FasterWhisperBackend(whisper_dir("base")),
        "faster-whisper base, CPU int8",
        lambda: whisper_dir("base").is_dir(),
        (
            "uv run --extra models hf download Systran/faster-whisper-base "
            "--local-dir models/whisper-base-ct2"
        ),
    ),
    "whisper-small": (
        lambda: FasterWhisperBackend(whisper_dir("small")),
        "faster-whisper small, CPU int8 (previous app default)",
        lambda: whisper_dir("small").is_dir(),
        (
            "uv run --extra models hf download Systran/faster-whisper-small "
            "--local-dir models/whisper-small-ct2"
        ),
    ),
    "whisper-turbo": (
        lambda: FasterWhisperBackend(whisper_dir("large-v3-turbo")),
        "faster-whisper large-v3-turbo, CPU int8",
        lambda: whisper_dir("large-v3-turbo").is_dir(),
        (
            "uv run --extra models hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo "
            "--local-dir models/whisper-large-v3-turbo-ct2"
        ),
    ),
    "mlx-turbo": (
        lambda: MlxWhisperBackend(MODELS / "whisper-large-v3-turbo-mlx"),
        "mlx-whisper large-v3-turbo, Apple GPU (app default)",
        (MODELS / "whisper-large-v3-turbo-mlx").is_dir,
        (
            "uv run --extra models hf download mlx-community/whisper-large-v3-turbo "
            "--local-dir models/whisper-large-v3-turbo-mlx"
        ),
    ),
    "whistle": (
        WhistleBackend,
        "Cactus Whistle (17 MB), CPU",
        lambda: True,
        None,
    ),
    "whistle-kw": (
        lambda: WhistleBackend(WHISTLE_KEYWORDS),
        "Cactus Whistle with router keyword biasing",
        lambda: True,
        None,
    ),
}
DEFAULT_BACKENDS = "whisper-small,mlx-turbo,whistle,whistle-kw"


# ------------------------------------------------------------------------ record


def play(path: Path) -> None:

    AudioPlayer().play_wav(path)


def record_one(recorder, path: Path) -> bool:

    def ready() -> None:
        print("  >>> speak now (stops after a short pause)", flush=True)

    while True:
        samples = recorder.capture(on_listening=ready)
        if samples is not None:
            break
        print("  ... that was too short, try again")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(str(path), samples, recorder.config.sample_rate)
    seconds = len(samples) / recorder.config.sample_rate
    print(f"  saved {path} ({seconds:.1f}s)")
    return True


def review(path: Path) -> str:
    """Return 'keep', 'redo' or 'quit'."""
    while True:
        choice = input("  [Enter] keep  [p] play  [r] redo  [q] quit > ").strip().lower()
        if choice == "":
            return "keep"
        if choice == "p":
            play(path)
        elif choice in ("r", "q"):
            return {"r": "redo", "q": "quit"}[choice]


def make_recorder(args):

    device = args.input_device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    config = DetectorConfig(
        silence_ms=args.silence_ms,
        threshold_multiplier=args.threshold_multiplier,
    )
    return VoiceActivityRecorder(config, input_device=device)


def cmd_record(args) -> None:
    root = Path(args.out)
    folder = root / args.speaker / args.condition
    recorder = make_recorder(args)
    print(
        f"Speaker: {args.speaker}   Condition: {args.condition}\n"
        "The first clip starts by measuring room noise for 2 seconds; stay quiet\n"
        "until you see >>>.\n"
    )
    if args.free:
        record_free(args, root, folder, recorder)
    else:
        record_prompts(args, root, folder, recorder)
    print("\nDone. Run `jarvis-asr-bench compare` when you have enough clips.")


def record_prompts(args, root: Path, folder: Path, recorder) -> None:
    prompts = load_prompts(Path(args.prompts))
    done = {entry["file"] for entry in load_manifest(root)}
    if args.only:
        wanted = set(args.only.split(","))
        prompts = [p for p in prompts if p.id in wanted]
    for index, prompt in enumerate(prompts, start=1):
        path = folder / f"{prompt.id}.wav"
        if str(path.relative_to(root)) in done and path.exists() and not args.redo:
            continue
        print(f"\n[{index}/{len(prompts)}] Say naturally:  {prompt.text}")
        input("  press Enter, wait for >>> then speak > ")
        while True:
            record_one(recorder, path)
            decision = review(path)
            if decision != "redo":
                break
        if decision == "quit":
            path.unlink(missing_ok=True)
            return
        append_manifest(
            root,
            {
                "file": str(path.relative_to(root)),
                "speaker": args.speaker,
                "condition": args.condition,
                "prompt_id": prompt.id,
                "intent": prompt.intent,
                "reference": prompt.text,
            },
        )


def record_free(args, root: Path, folder: Path, recorder) -> None:
    print(
        "Free mode: the visitor says anything; you type exactly what they said.\n"
        "Leave the transcript empty to discard a clip. Ctrl-C stops."
    )
    existing = len(list(folder.glob("free-*.wav"))) if folder.exists() else 0
    number = existing + 1
    while True:
        path = folder / f"free-{number:03d}.wav"
        input(f"\n[free {number}] press Enter, wait for >>>, then the visitor speaks > ")
        record_one(recorder, path)
        decision = review(path)
        if decision == "quit":
            path.unlink(missing_ok=True)
            return
        if decision == "redo":
            continue
        reference = input("  What exactly was said? > ").strip()
        if not reference:
            path.unlink(missing_ok=True)
            print("  discarded")
            continue
        guess = route(reference)
        intent = input(f"  Intent [{guess}] (Enter accepts) > ").strip() or guess
        append_manifest(
            root,
            {
                "file": str(path.relative_to(root)),
                "speaker": args.speaker,
                "condition": args.condition,
                "prompt_id": path.stem,
                "intent": intent,
                "reference": reference,
            },
        )
        number += 1


# ----------------------------------------------------------------------- compare


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(rows: list[dict]) -> dict:
    errors = sum(r["word_errors"] for r in rows)
    words = sum(r["ref_words"] for r in rows)
    latencies = [r["seconds"] for r in rows]
    return {
        "clips": len(rows),
        "wer": errors / words if words else 0.0,
        "same_route": sum(r["hyp_route"] == r["ref_route"] for r in rows) / len(rows),
        "right_intent": sum(r["hyp_route"] == r["intent"] for r in rows) / len(rows),
        "median_s": statistics.median(latencies),
        "p90_s": percentile(latencies, 0.9),
    }


def cmd_compare(args) -> None:
    root = Path(args.recordings)
    entries = load_manifest(root)
    if args.speaker:
        entries = [e for e in entries if e["speaker"] in args.speaker.split(",")]
    if args.condition:
        entries = [e for e in entries if e["condition"] in args.condition.split(",")]
    entries = [e for e in entries if (root / e["file"]).is_file()]
    if not entries:
        raise SystemExit(f"No recordings found under {root}. Run `jarvis-asr-bench record` first.")

    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        raise SystemExit(f"Unknown backends: {unknown}. Choose from: {', '.join(BACKENDS)}")

    print(f"{len(entries)} clips from {len({e['speaker'] for e in entries})} speaker(s)\n")
    rows: list[dict] = []
    load_seconds: dict[str, float] = {}
    for name in names:
        factory, label, available, download = BACKENDS[name]
        if not available():
            print(f"skip {name}: model missing. Download with:\n  {download}\n")
            continue
        print(f"== {name}: {label}")
        started = time.perf_counter()
        backend = factory()
        backend.transcribe(str(root / entries[0]["file"]))  # warm-up, not timed
        load_seconds[name] = time.perf_counter() - started
        for entry in entries:
            wav = root / entry["file"]
            started = time.perf_counter()
            try:
                text = backend.transcribe(str(wav))
            except Exception as exc:  # keep going; one bad clip shouldn't end the run
                logger.warning("%s failed on %s: %s", name, entry["file"], exc)
                text = ""
            seconds = time.perf_counter() - started
            errors, ref_words = word_error_rate(entry["reference"], text)
            row = {
                "backend": name,
                "file": entry["file"],
                "speaker": entry["speaker"],
                "condition": entry["condition"],
                "intent": entry["intent"],
                "reference": entry["reference"],
                "hypothesis": text,
                "word_errors": errors,
                "ref_words": ref_words,
                "ref_route": route(entry["reference"]),
                "hyp_route": route(text),
                "seconds": round(seconds, 3),
            }
            rows.append(row)
            mark = "ok " if row["hyp_route"] == row["intent"] else "MISS"
            print(f"  {mark} {seconds:5.2f}s  {entry['reference']!r} -> {text!r}")
        print()

    if not rows:
        raise SystemExit("No backend produced results.")
    write_report(rows, load_seconds, entries)


def write_report(rows: list[dict], load_seconds: dict[str, float], entries: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    csv_path = RESULTS / f"{stamp}-clips.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Polish speech-to-text comparison, {stamp}",
        "",
        (
            f"{len(entries)} clips, "
            f"speakers: {', '.join(sorted({e['speaker'] for e in entries}))}, "
            f"conditions: {', '.join(sorted({e['condition'] for e in entries}))}"
        ),
        "",
        "- **Right intent**: the router picked the labelled intent from the transcript.",
        "- **Same route**: the router picked the same intent as from the reference text,",
        "  which isolates transcription quality from router gaps.",
        "- Latency excludes model loading (shown separately).",
        "",
        "| Backend | Right intent | Same route | WER | Median s | p90 s | Load s |",
        "|---|---|---|---|---|---|---|",
    ]
    backends = list(dict.fromkeys(r["backend"] for r in rows))
    for name in backends:
        s = summarize([r for r in rows if r["backend"] == name])
        lines.append(
            f"| {name} | {s['right_intent']:.0%} | {s['same_route']:.0%} | {s['wer']:.0%} "
            f"| {s['median_s']:.2f} | {s['p90_s']:.2f} | {load_seconds.get(name, 0):.1f} |"
        )

    conditions = sorted({r["condition"] for r in rows})
    if len(conditions) > 1:
        lines += [
            "",
            "## Right intent by condition",
            "",
            "| Backend | " + " | ".join(conditions) + " |",
            "|---|" + "---|" * len(conditions),
        ]
        for name in backends:
            cells = []
            for condition in conditions:
                subset = [r for r in rows if r["backend"] == name and r["condition"] == condition]
                cells.append(f"{summarize(subset)['right_intent']:.0%}" if subset else "–")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    gaps = sorted(
        {
            (r["reference"], r["intent"], r["ref_route"])
            for r in rows
            if r["ref_route"] != r["intent"]
        }
    )
    if gaps:
        lines += [
            "",
            "## Router gaps (correct text, wrong intent)",
            "",
            "These fail even with a perfect transcript; fix the keyword lists.",
            "",
        ]
        lines += [f"- {ref!r}: expected `{want}`, router gave `{got}`" for ref, want, got in gaps]

    md_path = RESULTS / f"{stamp}-summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {md_path.relative_to(PROJECT_ROOT)} and {csv_path.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- cli


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Polish speech-to-text benchmark.",
)
def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def run_command(command, **options) -> None:
    try:
        command(SimpleNamespace(**options))
    except KeyboardInterrupt:
        click.echo("\nStopped.")
        sys.exit(130)


@main.command(help="Record test clips.")
@click.option("--speaker", required=True, help="Short label, e.g. ania or kid1.")
@click.option(
    "--condition",
    default="quiet",
    show_default=True,
    help="Recording conditions, e.g. quiet, noise, costume.",
)
@click.option("--free", is_flag=True, help="Unscripted lines; you type what was said.")
@click.option("--prompts", default=str(PROMPTS), show_default=True)
@click.option("--only", help="Comma-separated prompt ids to record.")
@click.option("--redo", is_flag=True, help="Re-record existing clips.")
@click.option("--out", default=str(RECORDINGS), show_default=True)
@click.option("--input-device")
@click.option("--silence-ms", type=int, default=900, show_default=True)
@click.option("--threshold-multiplier", type=float, default=3.0, show_default=True)
def record(**options) -> None:
    run_command(cmd_record, **options)


@main.command(help="Run backends on the recordings.")
@click.option(
    "--backends",
    default=DEFAULT_BACKENDS,
    show_default=True,
    help=f"Comma-separated; available: {', '.join(BACKENDS)}.",
)
@click.option("--recordings", default=str(RECORDINGS), show_default=True)
@click.option("--speaker", help="Comma-separated speakers to include.")
@click.option("--condition", help="Comma-separated conditions to include.")
def compare(**options) -> None:
    run_command(cmd_compare, **options)


if __name__ == "__main__":
    main()
