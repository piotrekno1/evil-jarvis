"""Checks each pipeline stage (voice to text, text to text, text to speech, voice effects).

    uv run --extra models --extra voice python benchmarks/pipeline_check.py OUT_DIR

Uses recordings/asr and the default models; writes OUT_DIR/results.json and the audio.
The 2026-10-20 run is summarized in benchmarks/2026-10-20-pipeline-check.md.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from evil_jarvis.bench.asr import route, word_error_rate
from evil_jarvis.llm import DEFAULT_MODEL
from evil_jarvis.llm.llama import LlamaResponder
from evil_jarvis.persona import load_persona
from evil_jarvis.stt import MODELS, build_transcriber
from evil_jarvis.tts.piper import PiperSpeaker, voice_path
from evil_jarvis.tts.playback import resample
from evil_jarvis.wav import SPEECH_SAMPLE_RATE, read_wav

RECORDINGS = Path("recordings/asr")
WHISPER = str(MODELS["mlx"][0])
EXTRA_VISITOR_LINES = [
    "Czy księżyc pamięta twój cyrk?",
    "Co słyszysz, kiedy wszyscy już śpią?",
    "Czy klauni mają sny?",
    "Gdzie jest moja mama?",
    "Ile masz lat?",
    "Lubisz pizzę?",
    "Ładny kostium.",
    "Mój brat się ciebie boi.",
    "Umiesz żonglować?",
    "Jesteś śmieszny czy straszny?",
    "Jak się dzisiaj miewasz?",
]
# (prosody, effect): plain speech first, then the delivery, then each effect on top.
VARIANTS = {
    "tts-plain": ("plain", "clean"),
    "tts-maniacal": ("maniacal", "clean"),
    "effect-metallic": ("maniacal", "metallic"),
    "effect-metallic-demon": ("maniacal", "metallic-demon"),
}
EASTERN = ("ru", "uk", "be")


def recordings() -> list[dict]:
    entries, seen = [], set()
    for line in (RECORDINGS / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["file"] not in seen and (RECORDINGS / entry["file"]).is_file():
            seen.add(entry["file"])
            entries.append(entry)
    return entries


def check_speech_to_text(entries: list[dict]) -> list[dict]:
    rows = []
    for mode in ("pl", "auto"):
        transcriber = build_transcriber("mlx", WHISPER, mode)
        transcriber.transcribe(str(RECORDINGS / entries[0]["file"]))  # warm-up
        for entry in entries:
            started = time.perf_counter()
            text = transcriber.transcribe(str(RECORDINGS / entry["file"]))
            seconds = time.perf_counter() - started
            errors, words = word_error_rate(entry["reference"], text)
            rows.append(
                {
                    "mode": mode,
                    "file": entry["file"],
                    "speaker": entry["speaker"],
                    "reference": entry["reference"],
                    "heard": text,
                    "errors": errors,
                    "words": words,
                    "language": transcriber.last_language,
                    "discarded": transcriber.last_discarded,
                    "intent_ok": route(text) == entry["intent"],
                    "seconds": round(seconds, 3),
                }
            )
        print(f"voice to text ({mode}): {len(entries)} clips", flush=True)
    return rows


def check_text_to_text(visitor_lines: list[str]) -> list[dict]:
    responder = LlamaResponder(str(DEFAULT_MODEL), persona=load_persona("clown"), memory_seconds=0)
    responder.reply("Cześć!", language="pl")  # warm-up
    rows = []
    for visitor in visitor_lines:
        started = time.perf_counter()
        reply = responder.reply(visitor, language="pl")
        info = responder.last_reply_info or {}
        rows.append(
            {
                "visitor": visitor,
                "reply": reply,
                "attempts": info.get("attempts"),
                "rejections": list(info.get("rejections", [])),
                "seconds": round(time.perf_counter() - started, 2),
                "words": len(reply.split()),
            }
        )
    print(f"text to text: {len(rows)} lines", flush=True)
    return rows


def check_speech(replies: list[str], audio: Path) -> list[dict]:
    import mlx_whisper

    speakers = {
        name: PiperSpeaker(
            str(voice_path("pl_PL-bass-high")), effect=effect, prosody=prosody, seed=1
        )
        for name, (prosody, effect) in VARIANTS.items()
    }
    rows = []
    for index, reply in enumerate(replies):
        for name, speaker in speakers.items():
            wav = audio / f"{index:02d}-{name}.wav"
            started = time.perf_counter()
            speaker.render(reply, wav)
            render_seconds = time.perf_counter() - started
            samples, rate = read_wav(wav)
            clip = resample(samples, rate, SPEECH_SAMPLE_RATE)
            heard = mlx_whisper.transcribe(
                clip, path_or_hf_repo=WHISPER, language="pl", condition_on_previous_text=False
            )
            detected = mlx_whisper.transcribe(
                clip, path_or_hf_repo=WHISPER, language=None, condition_on_previous_text=False
            )
            errors, words = word_error_rate(reply, heard["text"])
            rows.append(
                {
                    "index": index,
                    "variant": name,
                    "wav": str(wav),
                    "reply": reply,
                    "heard": heard["text"].strip(),
                    "errors": errors,
                    "words": words,
                    "detected_language": detected["language"],
                    "render_seconds": round(render_seconds, 2),
                    "audio_seconds": round(len(samples) / rate, 2),
                }
            )
        print(f"speech: {index + 1}/{len(replies)}", flush=True)
    return rows


def add_language_odds(rows: list[dict], path_key: str) -> None:
    """Whisper small's P(Polish) and P(Russian/Ukrainian/Belarusian), a more sensitive check."""
    from faster_whisper import WhisperModel, decode_audio

    small = WhisperModel(str(MODELS["faster-whisper-cpu"][0]), device="cpu", compute_type="int8")
    for row in rows:
        _, _, probabilities = small.detect_language(
            decode_audio(row[path_key], sampling_rate=SPEECH_SAMPLE_RATE)
        )
        odds = dict(probabilities)
        row["p_pl"] = round(odds.get("pl", 0), 3)
        row["p_east"] = round(sum(odds.get(code, 0) for code in EASTERN), 3)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    out = Path(sys.argv[1])
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)

    entries = recordings()
    stt = check_speech_to_text(entries)
    lines = list(dict.fromkeys([e["reference"] for e in entries] + EXTRA_VISITOR_LINES))
    llm = check_text_to_text(lines)
    speech = check_speech([row["reply"] for row in llm if row["reply"]], audio)

    fixed_language = [row for row in stt if row["mode"] == "pl"]
    for row in fixed_language:
        row["path"] = str(RECORDINGS / row["file"])
    add_language_odds(fixed_language, "path")
    add_language_odds(speech, "wav")

    results = {"stt": stt, "llm": llm, "tts": speech}
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"done: {out / 'results.json'}")


if __name__ == "__main__":
    main()
