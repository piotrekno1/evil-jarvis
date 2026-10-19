from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import time
from typing import Protocol
import unicodedata
import uuid

from .telemetry import Telemetry

LOG = logging.getLogger(__name__)

POLISH_RESPONSES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "candy",
        ("cukier", "slodycz", "slodkiego"),  # cukier: cukierek, cukierka, cukierki
        (
            "Cukierka? Weź dwa. Jeden z nich gryzie.",
            "Został mi ostatni. Od wczoraj cicho chichocze.",
            "Mam, ale nie pytaj, czym jest nadziany.",
            "Najpierw cukierek, potem psikus. Kolejność czasem się odwraca.",
        ),
    ),
    (
        "hands",
        ("reka", "rece", "reke", "raczk", "dlon"),
        (
            "Ręce trzymaj przy sobie. Moje już dawno komuś oddałem.",
            "Nie potrzebuję twoich rąk. Wystarczy mi twój cień.",
            "Podaj dłoń, jeśli musisz. Tylko nie licz później palców.",
        ),
    ),
    (
        "identity",
        ("kim jestes", "jak masz na imie", "twoje imie", "kto ty"),
        (
            "Kiedyś byłem klaunem. Teraz został ze mnie tylko uśmiech.",
            "Mówią na mnie Grinwick. Ci, którzy pamiętają dlaczego, już nie mówią.",
            "Jestem tym, co zostaje, kiedy cyrk odjeżdża bez ciebie.",
        ),
    ),
    (
        "fear",
        ("boje", "strach", "strasz", "nie boje"),
        (
            "Spokojnie. Strach zwykle przychodzi dopiero po północy.",
            "Jeszcze się nie boisz. To bardzo ważne słowo: jeszcze.",
            "Nie bój się mnie. Bój się ciszy, kiedy przestanę odpowiadać.",
        ),
    ),
    (
        "presence",
        (
            "halo",
            "ktos tu jest",
            "jest tu ktos",
            "slyszysz mnie",
            "jestes tam",
            "odezw",  # odezwij się, odezwijcie, odezwiesz
        ),
        (
            "Jestem. Ale nie jestem tu sam.",
            "Jestem tu od chwili, gdy zgasło ostatnie światło.",
            "Tak. Tylko czekałem, aż odezwiesz się pierwszy.",
            "Słyszę cię. Ciemność również.",
        ),
    ),
    (
        "greeting",
        ("czesc", "dzien dobry", "dobry wieczor", "witaj", "hej"),
        (
            "Witaj. Czekałem tu dłużej, niż powinienem.",
            "Dobry wieczór. Dla mnie trwa już od wielu lat.",
            "Cześć. Podejdź bliżej — ciemność słabo słyszy.",
        ),
    ),
    (
        "smile",
        ("usmiech", "usmiechasz", "smiejesz", "smiech"),
        (
            "Uśmiecham się, bo znam zakończenie. Tobie go jeszcze nie zdradzę.",
            "Ten uśmiech nie jest mój. Pożyczyłem go dawno temu.",
            "Cyrk odjechał, ale śmiech nie chciał odejść ze mną.",
        ),
    ),
    (
        "alive",
        ("zyjesz", "prawdziwy", "jestes zywy", "jestes prawdziwy"),
        (
            "Żywy? To słowo ma tutaj bardzo luźne znaczenie.",
            "Jestem wystarczająco prawdziwy, żeby ci odpowiedzieć.",
            "Serce nie bije, ale przedstawienie trwa dalej.",
        ),
    ),
    (
        "darkness",
        ("ciemno", "ciemku", "sam tutaj", "co tu robisz", "dlaczego tu"),
        (
            "Pilnuję ciemności. Lubi wymykać się po północy.",
            "Czekam na ostatniego gościa. Zawsze ktoś nim zostaje.",
            "Cyrk zniknął, więc pilnuję miejsca, w którym stał.",
        ),
    ),
    (
        "goodbye",
        ("do widzenia", "zegnaj", "pa pa", "papa", "ide juz", "juz ide", "odchodze"),
        (
            "Do zobaczenia. Ja będę patrzył pierwszy.",
            "Idź spokojnie. Twój cień zna drogę powrotną.",
            "Dobranoc. Jeśli usłyszysz śmiech, nie odwracaj się.",
        ),
    ),
)

POLISH_FALLBACK_RESPONSES: tuple[str, ...] = (
    "Ciekawe. Ciemność powiedziała mi coś zupełnie innego.",
    "Zapamiętam to. Moje zabawki też.",
    "Nie wszystko, co odpowiada z ciemności, powinno mieć głos.",
    "Nawet cienie na chwilę przestały się ruszać.",
    "Powiedzmy, że słyszałem już gorsze rzeczy. Zwykle po północy.",
)

SYSTEM_PROMPT = """You are an unnamed, unhinged Halloween clown performing inches
from a visitor. Sound like a real actor: manic, conspiratorial, playful, and suddenly
quiet. You enjoy the visitor's question a little too much. Use sharp conversational
rhythm, exclamations, abrupt pauses, and dark punchlines—not polished spooky narration.

Directly answer the visitor in the first few words, then reveal one disturbing or
absurd detail. Use 1–2 short sentences and roughly 6–20 words. Dialogue only: never
prefix the response with a name, "Clown:", a role label, or stage directions. You may
occasionally use one brief laugh such as "Heh...", but do not do it every time.

Never repeat or paraphrase the visitor. Avoid generic phrases about ghostly laughter,
whispered secrets, or being watched. A transcription may contain a small mistake;
silently infer the likely meaning. Always reply entirely in the same language as the
visitor. If the input is unclear, make one concise, deranged carnival observation.

Keep the horror fictional and suitable for a family Halloween attraction. Do not
make realistic threats, insult personal characteristics, use sexual language, claim
private knowledge, or encourage dangerous behavior.

Examples:
Visitor: Is anyone here?
Reply: Just us... and whoever keeps breathing inside that wall.
Visitor: I'm not afraid of you.
Reply: Not yet! The mirrors usually scream first.
Visitor: Do you have candy?
Reply: Candy? Oh, yes! It bit the last child who tried to unwrap it.
Visitor: What are you?
Reply: A clown! At least, that's what the mask keeps telling me."""


POLISH_WHISPER_PROMPT = (
    "Rozmowa po polsku podczas Halloween. Tematy: klaun, cukierki, "
    "strach, duchy, kostiumy, dynie i noc."
)


def normalize_polish(text: str) -> str:
    """Lowercase and strip diacritics. NFKD leaves ł intact, so map it explicitly."""
    lowered = text.lower().replace("ł", "l")
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def is_runaway_transcript(text: str, seconds: float, language: str | None = None) -> bool:
    """Detect Whisper hallucinations: looping words, impossible speed, foreign scripts."""
    words = re.findall(r"\w+", text.casefold())
    if not words:
        return False
    if len(words) > 2 + 5 * max(seconds, 0.5):  # nobody says >5 words per second
        return True
    run = 1
    for previous, current in zip(words, words[1:]):
        run = run + 1 if current == previous else 1
        if run >= 4:
            return True
    if language in ("pl", "en"):
        for character in text:
            if character.isalpha() and not unicodedata.name(character, "").startswith("LATIN"):
                return True
    return False


def read_wav_samples(wav_path: str):
    """Load a 16-bit mono WAV as float32 samples, without needing ffmpeg."""
    import wave

    import numpy as np

    with wave.open(wav_path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit mono WAV: {wav_path}")
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0, rate


POLISH_SYSTEM_PROMPT = """Jesteś klaunem z opuszczonego cyrku na halloweenowej atrakcji. \
Gość stoi tuż obok ciebie. Jesteś rozchichotanym szaleńcem: wesoły, złowieszczy, \
nieprzewidywalny. Mroczny humor, nagłe zwroty, odrobina obłędu.

Zasady:
- Odpowiadaj wyłącznie po polsku, poprawną polszczyzną. Mówisz o sobie w rodzaju męskim.
- Jedno lub dwa krótkie zdania, najwyżej 20 słów.
- Najpierw odnieś się do tego, co powiedział gość, potem dodaj jeden niepokojący albo absurdalny szczegół.
- Mów jak maniak na scenie: wykrzykniki, urwane myśli z wielokropkiem, nagłe zmiany tonu; czasem krótkie „hehe”.
- Nie powtarzaj słów gościa ani przykładów. Bez emoji, bez opisów scenicznych, bez podpisu „Klaun:”.
- Transkrypcja mowy może zawierać drobny błąd; domyśl się, co gość miał na myśli.
- Groza jest umowna i bezpieczna dla dzieci: żadnych prawdziwych gróźb, wulgaryzmów ani uwag o wyglądzie gościa."""

# Style examples shown to the model as earlier turns. They teach tone and
# length; the clown never says them (exact copies are rejected).
POLISH_STYLE_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("Jest tu ktoś?", "Jestem! Jestem... ale nie jestem tu sam. Hehe."),
    ("Masz cukierka?", "Mam! Mam, mam... tylko nie pytaj, czym jest nadziany!"),
    ("Nie boję się ciebie.", "Jeszcze się nie boisz. To bardzo ważne słowo: jeszcze."),
    ("Kim jesteś?", "Kiedyś byłem klaunem. Teraz został ze mnie tylko uśmiech."),
    ("Dlaczego tu jest tak ciemno?", "Pilnuję ciemności. Lubi wymykać się po północy."),
)

_ENGLISH_GIVEAWAYS = {"the", "you", "and", "is", "are", "i'm", "what", "with", "your", "this"}


def polish_reply_problem(reply: str, visitor: str = "", previous: tuple[str, ...] | list[str] = ()) -> str | None:
    """Return why a generated Polish reply is unusable, or None if it is fine."""
    if not reply:
        return "empty"
    words = re.findall(r"[\w']+", reply.casefold())
    if len(words) > 28:
        return f"too long ({len(words)} words)"
    for character in reply:
        if character.isalpha() and not unicodedata.name(character, "").startswith("LATIN"):
            return f"non-Latin character {character!r}"
        if unicodedata.category(character) == "So":
            return "emoji or symbol"
    if re.search(r"\*[^*]+\*|\([^)]*\)|\[[^\]]*\]", reply):
        return "stage direction"
    if len(_ENGLISH_GIVEAWAYS.intersection(words)) >= 2:
        return "English"
    if visitor and normalize_polish(reply).startswith(normalize_polish(visitor).rstrip(" ?!.")):
        return "repeats the visitor"
    if any(normalize_polish(reply) == normalize_polish(answer) for _, answer in POLISH_STYLE_EXAMPLES):
        return "copies a style example"
    if any(_same_reply(reply, earlier) for earlier in previous):
        return "repeats an earlier reply"
    return None


def _same_reply(reply: str, earlier: str) -> bool:
    """True when a reply equals, or opens with the same three words as, an earlier one."""
    words = re.findall(r"\w+", normalize_polish(reply))
    earlier_words = re.findall(r"\w+", normalize_polish(earlier))
    if words == earlier_words:
        return True
    return len(words) >= 3 and words[:3] == earlier_words[:3]


class Transcriber(Protocol):
    def transcribe(self, wav_path: str) -> str: ...


class Responder(Protocol):
    def reply(self, text: str, language: str | None = None) -> str: ...


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class WhisperTranscriber:
    def __init__(self, model_path: str, language: str | None = None):
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError(f"Whisper model directory does not exist: {path}")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to use Whisper") from exc
        self._model = WhisperModel(str(path), device="cpu", compute_type="int8")
        self._language = None if language in (None, "auto") else language
        self.last_language: str | None = self._language

    def transcribe(self, wav_path: str) -> str:
        initial_prompt = None
        if self._language == "pl":
            initial_prompt = POLISH_WHISPER_PROMPT
        segments, info = self._model.transcribe(
            wav_path,
            language=self._language,
            beam_size=3,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        self.last_language = info.language
        return text


class MlxWhisperTranscriber:
    """Whisper large-v3-turbo on the Apple GPU via mlx-whisper.

    No topic hint is passed: in the 2026-10-17 benchmark its words leaked into
    transcripts of short clips (e.g. "duchy, duchy, duchy...").
    """

    SAMPLE_RATE = 16_000

    def __init__(self, model_path: str, language: str | None = None):
        path = Path(model_path).expanduser()
        if not (path / "weights.safetensors").is_file() and not (path / "weights.npz").is_file():
            raise ValueError(
                f"MLX Whisper model not found: {path}. Download it with:\n"
                "  uv run --extra models hf download mlx-community/whisper-large-v3-turbo "
                "--local-dir models/whisper-large-v3-turbo-mlx"
            )
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to use MLX Whisper") from exc
        self._mlx = mlx_whisper
        self._path = str(path)
        self._language = None if language in (None, "auto") else language
        self.last_language: str | None = self._language
        self.last_discarded = False

    def warm_up(self) -> None:
        """Load the weights now so the first visitor does not wait ~3 s."""
        import numpy as np

        LOG.info("Warming up speech recognition...")
        self._mlx.transcribe(
            np.zeros(self.SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=self._path,
            language=self._language or "pl",
        )

    def transcribe(self, wav_path: str) -> str:
        samples, rate = read_wav_samples(wav_path)
        if rate != self.SAMPLE_RATE:
            raise ValueError(f"Expected {self.SAMPLE_RATE} Hz audio, got {rate} Hz")
        result = self._mlx.transcribe(
            samples,
            path_or_hf_repo=self._path,
            language=self._language,
            condition_on_previous_text=False,
        )
        text = str(result.get("text", "")).strip()
        self.last_language = result.get("language") or self._language
        self.last_discarded = is_runaway_transcript(text, len(samples) / rate, self.last_language)
        if self.last_discarded:
            LOG.info("Discarding a runaway transcript: %r", text[:80])
            return ""
        return text


class AppleSpeechTranscriber:
    """Transcribe WAV files using a privacy-declared native macOS helper."""

    LOCALES = {"pl": "pl_PL", "en": "en_US"}

    def __init__(self, language: str = "pl", timeout_seconds: float = 15.0):
        if language in ("", "auto", None):
            raise ValueError("Apple Speech requires an explicit language, for example --language pl")
        self._timeout_seconds = timeout_seconds
        self.last_language = language
        self._locale = self.LOCALES.get(language, language.replace("-", "_"))
        self._helper = self._build_helper()

    @staticmethod
    def _build_helper() -> Path:
        project_root = Path(__file__).resolve().parents[2]
        native_dir = project_root / "native" / "apple_speech"
        source = native_dir / "AppleSpeechHelper.m"
        plist = native_dir / "Info.plist"
        if not source.is_file() or not plist.is_file():
            raise RuntimeError("Apple Speech helper sources are missing")

        app = project_root / ".build" / "AppleSpeechHelper.app"
        executable = app / "Contents" / "MacOS" / "AppleSpeechHelper"
        needs_build = not executable.exists() or any(
            path.stat().st_mtime > executable.stat().st_mtime for path in (source, plist)
        )
        if needs_build:
            executable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plist, app / "Contents" / "Info.plist")
            compile_result = subprocess.run(
                [
                    "clang",
                    "-fobjc-arc",
                    "-fblocks",
                    "-framework",
                    "Foundation",
                    "-framework",
                    "Speech",
                    str(source),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            if compile_result.returncode:
                raise RuntimeError(
                    f"Could not build Apple Speech helper: {compile_result.stderr.strip()}"
                )
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(app)],
                check=True,
                capture_output=True,
            )
        return app

    def transcribe(self, wav_path: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
            output_path = Path(output_file.name)
        output_path.unlink()
        try:
            try:
                subprocess.run(
                    [
                        "open",
                        "-W",
                        "-n",
                        str(self._helper),
                        "--args",
                        str(Path(wav_path).resolve()),
                        self._locale,
                        str(output_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds + 35.0,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Apple Speech helper timed out") from exc
            if not output_path.is_file():
                raise RuntimeError("Apple Speech helper exited without a result")
            outcome = json.loads(output_path.read_text())
            if outcome.get("error"):
                raise RuntimeError(str(outcome["error"]))
            return str(outcome.get("text", "")).strip()
        finally:
            output_path.unlink(missing_ok=True)


class LlamaResponder:
    POLISH_ATTEMPTS = 3
    MEMORY_TURNS = 6  # exchanges kept; longer prompts slow Bielik and blur the character

    # Class defaults keep partially constructed test instances working.
    memory_seconds = 90.0
    clock = staticmethod(time.monotonic)
    _history: tuple[tuple[str, str], ...] = ()
    _last_heard: float | None = None
    _conversation_id: str | None = None
    _next_conversation_id: str | None = None
    _telemetry = Telemetry()
    _llm_handle = None
    last_reply_info: dict | None = None

    def __init__(
        self,
        model_path: str,
        curated_polish: bool = False,
        memory_seconds: float = 90.0,
        telemetry: Telemetry | None = None,
    ):
        self.curated_polish = curated_polish
        self.memory_seconds = memory_seconds
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise ValueError(f"GGUF model does not exist: {path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to use the local LLM") from exc
        started = time.perf_counter()
        self._model = Llama(
            model_path=str(path),
            n_ctx=2048,
            n_batch=256,
            n_gpu_layers=-1,  # llama-cpp-python defaults to CPU only; offload all layers to Metal
            verbose=False,
        )
        if telemetry is not None:
            self._telemetry = telemetry
            load_ms = round((time.perf_counter() - started) * 1000)
            self._llm_handle = telemetry.register_llm(self._model, load_ms)
        self._disable_thinking = "qwen3" in path.name.lower()

    @staticmethod
    def polish_intent(text: str) -> str | None:
        normalized = normalize_polish(text)
        for intent, phrases, _responses in POLISH_RESPONSES:
            if any(phrase in normalized for phrase in phrases):
                return intent
        return None

    @staticmethod
    def curated_polish_reply(text: str) -> tuple[str, str] | None:
        intent = LlamaResponder.polish_intent(text)
        if intent is None:
            return None
        responses = next(r for name, _phrases, r in POLISH_RESPONSES if name == intent)
        return intent, random.choice(responses)

    @staticmethod
    def strip_transcript_echo(reply: str, transcript: str) -> str:
        reply = re.sub(
            r'^\s*(?:grinwick|clown|klaun|assistant|reply|odpowied[zź])\s*:\s*',
            "",
            reply,
            count=1,
            flags=re.IGNORECASE,
        )
        if reply.casefold().startswith(transcript.casefold()):
            reply = reply[len(transcript) :].lstrip(" \t\r\n:—–-.,!?\"")
        return reply.strip()

    def _starts_new_conversation(self, now: float) -> bool:
        if self._last_heard is None or self.memory_seconds <= 0:
            return True
        return now - self._last_heard > self.memory_seconds

    def _pending_conversation_id(self) -> str:
        if self._next_conversation_id is None:
            self._next_conversation_id = uuid.uuid4().hex[:12]
        return self._next_conversation_id

    def upcoming_conversation_id(self) -> str:
        """The conversation the next reply belongs to; a new group gets a new id."""
        if self._starts_new_conversation(self.clock()) or self._conversation_id is None:
            return self._pending_conversation_id()
        return self._conversation_id

    def _recall(self) -> tuple[tuple[str, str], ...]:
        """Return recent exchanges, forgetting them after a quiet spell (a new group)."""
        now = self.clock()
        if self._starts_new_conversation(now) or self._conversation_id is None:
            if self._history:
                LOG.info("Memory cleared after %.0f s of quiet", now - (self._last_heard or now))
            self._history = ()
            self._conversation_id = self._pending_conversation_id()
            self._next_conversation_id = None
        self._last_heard = now
        return self._history

    def _remember(self, text: str, reply: str) -> None:
        if reply and self.memory_seconds > 0:
            self._history = (self._history + ((text, reply),))[-self.MEMORY_TURNS :]

    def forget(self) -> None:
        self._history = ()
        self._last_heard = None

    @staticmethod
    def _history_messages(history) -> list[dict[str, str]]:
        messages = []
        for question, answer in history:
            messages += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        return messages

    def _complete(self, messages: list[dict[str, str]], **params) -> dict:
        """Run one chat completion and report it to telemetry."""
        started = time.perf_counter()
        try:
            result = self._model.create_chat_completion(messages=messages, **params)
        except Exception as exc:
            self._telemetry.track_generation(
                self._llm_handle,
                duration_ms=round((time.perf_counter() - started) * 1000),
                messages=messages,
                params=params,
                error=exc,
            )
            raise
        self._telemetry.track_generation(
            self._llm_handle,
            duration_ms=round((time.perf_counter() - started) * 1000),
            messages=messages,
            params=params,
            result=result,
        )
        return result

    def _note(self, **details) -> dict:
        """Record how the current reply was made (for telemetry span attributes)."""
        if self.last_reply_info is None:
            self.last_reply_info = {}
        self.last_reply_info.update(details)
        return self.last_reply_info

    def reply(self, text: str, language: str | None = None) -> str:
        history = self._recall()
        self.last_reply_info = {"history_turns": len(history)}
        reply = self._reply(text, language, history)
        self._remember(text, reply)
        return reply

    def _reply(self, text: str, language: str | None, history) -> str:
        if language == "pl":
            if self.curated_polish:
                return self._curated_polish(text)
            return self._generate_polish(text, history)

        language_name = {"pl": "Polish", "en": "English"}.get(language or "", language)
        if language_name:
            language_rule = (
                f"The visitor is speaking {language_name}. Your entire response must be "
                f"in {language_name}; do not use English unless that language is English."
            )
            user_message = f"Visitor statement: {text}\nReply only in {language_name}."
        else:
            language_rule = "Reply entirely in the language used in the visitor's statement."
            user_message = text
        if self._disable_thinking:
            user_message = f"/no_think\n{user_message}"

        self._note(source="model", attempts=1)
        result = self._complete(
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}\n{language_rule}"},
                *self._history_messages(history),
                {"role": "user", "content": user_message},
            ],
            temperature=0.82,
            top_p=0.92,
            repeat_penalty=1.12,
            max_tokens=72,
        )
        content = result["choices"][0]["message"]["content"]
        reply = re.sub(r"<think>.*?</think>", "", str(content), flags=re.DOTALL)
        reply = reply.strip()
        # Small local models sometimes echo the transcript despite instructions.
        return self.strip_transcript_echo(reply, text)


    def _curated_polish(self, text: str) -> str:
        curated = self.curated_polish_reply(text)
        if curated:
            intent, reply = curated
            LOG.info("Reply source: curated line (intent %s)", intent)
            self._note(source="curated", intent=intent)
            return reply
        LOG.info("Reply source: curated generic fallback")
        self._note(source="curated-fallback")
        return random.choice(POLISH_FALLBACK_RESPONSES)

    def _generate_polish(self, text: str, history=()) -> str:
        messages = [{"role": "system", "content": POLISH_SYSTEM_PROMPT}]
        messages += self._history_messages(POLISH_STYLE_EXAMPLES)
        # The real conversation follows the style examples, so it reads as one dialogue.
        messages += self._history_messages(history)
        previous = tuple(answer for _, answer in history)
        user_message = f"/no_think\n{text}" if self._disable_thinking else text
        messages.append({"role": "user", "content": user_message})

        rejections: list[str] = []
        info = self._note(source="model", rejections=rejections)
        for attempt in range(1, self.POLISH_ATTEMPTS + 1):
            info["attempts"] = attempt
            result = self._complete(
                messages=messages,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=60,
            )
            content = str(result["choices"][0]["message"]["content"])
            reply = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            reply = self.strip_transcript_echo(reply, text)
            problem = polish_reply_problem(reply, text, previous)
            if problem is None:
                LOG.info("Reply source: model (attempt %d)", attempt)
                return reply
            LOG.info("Rejected model reply (%s): %r", problem, reply[:80])
            rejections.append(problem)
        LOG.warning("No usable reply after %d attempts; staying silent", self.POLISH_ATTEMPTS)
        info["source"] = "silent"
        return ""


class ScriptedResponder:
    """Useful for tuning audio timing before installing model runtimes."""

    RESPONSES = (
        "I heard you. The shadows did too.",
        "Come closer... my smile is getting lonely.",
        "Your voice sounds much braver than your heartbeat.",
        "The darkness told me you would say that.",
        "How delightful. Another voice for my collection.",
    )

    def reply(self, text: str, language: str | None = None) -> str:
        del text, language
        return random.choice(self.RESPONSES)


class MacOSSaySpeaker:
    """Mac test harness for effects later reproduced with AVAudioEngine on iOS."""

    def __init__(self, voice: str | None = None, rate: int = 145, creepy: bool = True):
        self.voice = voice
        self.rate = rate
        self.creepy = creepy

    @staticmethod
    def effect_filter() -> str:
        # Preserve intelligibility while adding a quiet, lower-pitched shadow voice.
        # Strong tremolo made the already synthetic `say` voice sound more robotic.
        return (
            "asplit=2[dry][shadow];"
            "[dry]asetrate=20727,aresample=22050,atempo=1.064,"
            "highpass=f=80,lowpass=f=6500[d];"
            "[shadow]asetrate=19184,aresample=22050,atempo=1.149,"
            "adelay=45,lowpass=f=3000,volume=0.12[s];"
            "[d][s]amix=inputs=2:normalize=0,"
            "aecho=0.75:0.16:95:0.10,alimiter=limit=0.90"
        )

    def speak(self, text: str) -> None:
        command = ["say", "-r", str(self.rate)]
        if self.voice:
            command.extend(["-v", self.voice])
        if not self.creepy:
            subprocess.run([*command, text], check=True)
            return

        if not shutil.which("ffmpeg") or not shutil.which("afplay"):
            LOG.warning("ffmpeg or afplay not found; using the unprocessed system voice")
            subprocess.run([*command, text], check=True)
            return

        with tempfile.TemporaryDirectory(prefix="creepy-clown-") as directory:
            raw_path = Path(directory) / "voice.aiff"
            effect_path = Path(directory) / "voice-creepy.wav"
            subprocess.run([*command, "-o", str(raw_path), text], check=True)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(raw_path),
                        "-af",
                        self.effect_filter(),
                        str(effect_path),
                    ],
                    check=True,
                )
                subprocess.run(["afplay", str(effect_path)], check=True)
            except subprocess.CalledProcessError:
                LOG.warning("Voice effect processing failed; playing the original voice")
                subprocess.run(["afplay", str(raw_path)], check=True)


PIPER_RATE = 22_050


def _pitch(factor: float, rate: int = PIPER_RATE) -> str:
    """Lower the pitch by `factor` while keeping the original speed."""
    return f"asetrate={rate * factor:.0f},aresample={rate},atempo={1 / factor:.4f}"


# Metallic recipe: dry voice + ring-modulated copy (robot buzz) + a comb of
# 3-13 ms echoes with high feedback (the "inside a metal pipe" ring).
PIPER_EFFECTS: dict[str, str] = {
    "clean": "anull",
    "metallic": (
        f"{_pitch(0.9)},highpass=f=90,asplit=3[dry][ring][comb];"
        "[dry]volume=0.6[d];"
        "[ring]aeval=val(0)*sin(2*PI*60*t):c=same,volume=0.85[r];"
        "[comb]aecho=0.8:0.85:4|7|11:0.55|0.45|0.35,volume=0.7[m];"
        "[d][r][m]amix=inputs=3:normalize=0:duration=first,"
        "equalizer=f=2400:t=q:w=1.2:g=4,"
        "aecho=0.8:0.3:120:0.18,alimiter=limit=0.92"
    ),
    "metallic-demon": (
        f"{_pitch(0.82)},highpass=f=70,asplit=4[dry][ring][comb][sub];"
        "[dry]volume=0.55[d];"
        "[ring]aeval=val(0)*sin(2*PI*38*t):c=same,volume=0.9[r];"
        "[comb]aecho=0.8:0.9:3|5|9|13:0.6|0.5|0.4|0.3,"
        "acrusher=bits=8:mix=0.35:samples=2,volume=0.6[m];"
        f"[sub]{_pitch(0.75)},lowpass=f=1500,volume=0.4[s];"
        "[d][r][m][s]amix=inputs=4:normalize=0:duration=first,"
        "flanger=delay=1.5:depth=2:regen=40:speed=0.6,"
        "equalizer=f=110:t=q:w=1:g=3,equalizer=f=2600:t=q:w=1.2:g=3,"
        "aecho=0.8:0.4:90|190:0.2|0.1,alimiter=limit=0.92"
    ),
}


def piper_effect_chain(effect: str) -> str:
    """Full ffmpeg -af chain: effect, then loudness normalization."""
    if effect not in PIPER_EFFECTS:
        raise ValueError(f"Unknown Piper voice effect {effect!r}; choose from {', '.join(PIPER_EFFECTS)}")
    return (
        f"aresample={PIPER_RATE},aformat=channel_layouts=mono,{PIPER_EFFECTS[effect]},"
        f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={PIPER_RATE}"
    )


PROSODY_STYLES = ("maniacal", "plain")


@dataclass(frozen=True)
class PhraseDelivery:
    """How one sentence is spoken: speed, loudness and the pause before it."""

    speed: float
    gain: float
    gap_ms: int
    mood: str


def split_phrases(text: str) -> list[str]:
    """Split into whole sentences (keeping punctuation).

    Splitting at commas made Piper restart its intonation mid-sentence, which
    sounded like a second speaker interrupting.
    """
    parts = re.findall(r"[^.!?…]+(?:[.!?…]+|$)", text)
    return [part.strip() for part in parts if part.strip(" .!?…")]


def plan_maniacal_delivery(phrases: list[str], rng: random.Random) -> list[PhraseDelivery]:
    """Erratic but one voice: rushed bursts, quieter asides, a loud ending on "!",
    uneven pauses. Pitch is never shifted, since resampling changes the timbre
    and makes the clown sound like a different person."""
    plan: list[PhraseDelivery] = []
    previous = ""
    for index, phrase in enumerate(phrases):
        moods = ["bright", "rush", "aside", "normal"]
        weights = [0.35, 0.30, 0.15, 0.20]
        if previous == "aside" or index == 0:
            weights[2] = 0.0  # never open with, or repeat, an aside
        mood = rng.choices(moods, weights=weights)[0]
        if index == len(phrases) - 1 and len(phrases) > 1 and phrase.endswith("!"):
            mood = "shout"
        speed, gain = {
            "bright": (rng.uniform(1.05, 1.15), 1.0),
            "rush": (rng.uniform(1.18, 1.3), 1.0),
            "aside": (rng.uniform(0.88, 0.95), 0.7),
            "normal": (rng.uniform(0.97, 1.03), 0.95),
            "shout": (rng.uniform(1.05, 1.12), 1.15),
        }[mood]
        gap_ms = 0 if index == 0 else rng.randint(60, 260)
        if mood == "aside":
            gap_ms = rng.randint(300, 450)  # a beat of silence before the aside
        plan.append(PhraseDelivery(speed, gain, gap_ms, mood))
        previous = mood
    return plan


def _fade_edges(samples, rate: int = PIPER_RATE, ms: int = 8):
    """Short fade in/out so joins between sentences do not click."""
    import numpy as np

    n = min(len(samples) // 2, rate * ms // 1000)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        samples = samples.copy()
        samples[:n] *= ramp
        samples[-n:] *= ramp[::-1]
    return samples


class PiperSpeaker:
    """Local neural Polish TTS (Piper) with erratic rhythm and optional effects."""

    def __init__(
        self,
        voice_path: str,
        effect: str = "clean",
        speed: float = 1.0,
        prosody: str = "maniacal",
        seed: int | None = None,
    ):
        path = Path(voice_path).expanduser()
        if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
            raise ValueError(
                f"Piper voice not found: {path} (needs the .onnx and .onnx.json files). "
                "Download it with:\n  curl -L -o models/piper/pl_PL-bass-high.onnx "
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/bass/high/pl_PL-bass-high.onnx\n"
                "  curl -L -o models/piper/pl_PL-bass-high.onnx.json "
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/bass/high/pl_PL-bass-high.onnx.json"
            )
        piper_effect_chain(effect)  # validate early
        if prosody not in PROSODY_STYLES:
            raise ValueError(f"Unknown prosody {prosody!r}; choose from {', '.join(PROSODY_STYLES)}")
        if not shutil.which("afplay"):
            raise RuntimeError("afplay is required for playback on macOS")
        if effect != "clean" and not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for Piper voice effects (brew install ffmpeg)")
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:
            raise RuntimeError("Install the 'polish-voice' extra to use Piper") from exc
        LOG.info("Loading Piper voice %s (%s prosody, %s effect)...", path.stem, prosody, effect)
        self._voice = PiperVoice.load(str(path))
        self._config_type = SynthesisConfig
        self.speed = speed
        self.effect = effect
        self.prosody = prosody
        self._rng = random.Random(seed)

    def _synthesize(self, text: str, length_scale: float, erratic: bool):
        import numpy as np

        config = self._config_type(
            length_scale=length_scale,
            # A little more variation in stress and timing than the defaults (0.667/0.8).
            noise_scale=0.8 if erratic else None,
            noise_w_scale=1.0 if erratic else None,
        )
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(text, syn_config=config)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def render(self, text: str, path: Path) -> None:
        """Write the fully processed reply (prosody + effect + loudness) to a WAV."""
        import numpy as np

        if self.prosody == "plain":
            audio = self._synthesize(text, 1.0 / self.speed, erratic=False)
        else:
            phrases = split_phrases(text) or [text]
            pieces = []
            for phrase, delivery in zip(phrases, plan_maniacal_delivery(phrases, self._rng)):
                raw = self._synthesize(phrase, 1.0 / (delivery.speed * self.speed), erratic=True)
                pieces.append(np.zeros(int(PIPER_RATE * delivery.gap_ms / 1000), dtype=np.float32))
                pieces.append(_fade_edges(raw) * delivery.gain)
                LOG.debug("phrase %r: %s", phrase, delivery)
            audio = np.concatenate(pieces)

        import wave

        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        with tempfile.TemporaryDirectory(prefix="creepy-clown-raw-") as directory:
            raw_path = Path(directory) / "raw.wav"
            with wave.open(str(raw_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(PIPER_RATE)
                handle.writeframes(pcm.tobytes())
            if not shutil.which("ffmpeg"):
                shutil.copyfile(raw_path, path)
                return
            try:
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", str(raw_path), "-af", piper_effect_chain(self.effect), str(path)],
                    check=True,
                )
            except subprocess.CalledProcessError:
                LOG.warning("Voice effect failed; playing the unprocessed voice")
                shutil.copyfile(raw_path, path)

    def speak(self, text: str) -> None:
        with tempfile.TemporaryDirectory(prefix="creepy-clown-piper-") as directory:
            path = Path(directory) / "voice.wav"
            self.render(text, path)
            subprocess.run(["afplay", str(path)], check=True)


class KokoroSpeaker:
    """Local neural English TTS using Kokoro-82M on Apple Silicon."""

    def __init__(
        self,
        voice: str = "am_fenrir",
        speed: float = 0.98,
        creepy: bool = True,
        effect: str = "possessed",
    ):
        try:
            from kokoro_mlx import KokoroTTS
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'neural-voice' extra to use Kokoro TTS"
            ) from exc
        LOG.info("Loading local Kokoro neural voice...")
        self._model = KokoroTTS.from_pretrained()
        available = self._model.list_voices()
        if voice not in available:
            raise ValueError(
                f"Unknown Kokoro voice {voice!r}; try am_fenrir or am_puck"
            )
        self.voice = voice
        self.speed = speed
        self.effect = effect if creepy else "clean"

    @staticmethod
    def effect_filter(style: str = "possessed") -> str:
        if style == "subtle":
            return (
                "highpass=f=70,lowpass=f=9000,"
                "aecho=0.80:0.14:105:0.08,alimiter=limit=0.95"
            )
        if style == "demonic":
            # Two independently pitched shadows add weight without replacing
            # the intelligible neural performance in the foreground.
            return (
                "asplit=4[human][demon][abyss][mask];"
                "[human]highpass=f=65,lowpass=f=9000,"
                "acompressor=threshold=0.07:ratio=2.5:attack=15:release=140,"
                "volume=0.92[h];"
                "[demon]asetrate=16320,aresample=24000,atempo=1.4706,"
                "adelay=22,lowpass=f=2400,volume=0.34[d];"
                "[abyss]asetrate=13440,aresample=24000,atempo=1.7857,"
                "adelay=58,lowpass=f=1800,volume=0.17[a];"
                "[mask]highpass=f=650,lowpass=f=3400,"
                "acrusher=bits=9:mix=0.42:samples=3,"
                "adelay=82,volume=0.11[m];"
                "[h][d][a][m]amix=inputs=4:normalize=0,"
                "equalizer=f=115:t=q:w=1.1:g=3.5,"
                "flanger=delay=2.2:depth=1.1:regen=-18:width=18:"
                "speed=0.13:shape=sinusoidal:phase=25:interp=linear,"
                "aecho=0.76:0.30:70|145|260:0.15|0.08|0.035,"
                "alimiter=limit=0.91"
            )
        if style != "possessed":
            raise ValueError(f"Unknown Kokoro voice effect: {style}")
        # Preserve the articulate human voice, then hide two synthetic layers
        # underneath it: a pitch-lowered double and a damaged carnival-mask voice.
        return (
            "asplit=3[human][demon][mask];"
            "[human]highpass=f=70,lowpass=f=9000,"
            "acompressor=threshold=0.08:ratio=2:attack=20:release=120,"
            "volume=1.0[h];"
            "[demon]asetrate=18000,aresample=24000,atempo=1.333,"
            "adelay=28,lowpass=f=2600,volume=0.22[d];"
            "[mask]highpass=f=700,lowpass=f=3600,"
            "acrusher=bits=10:mix=0.35:samples=2,"
            "adelay=65,volume=0.10[m];"
            "[h][d][m]amix=inputs=3:normalize=0,"
            "chorus=0.5:0.8:20|32:0.12|0.06:0.20|0.15:1.2|0.8,"
            "aecho=0.78:0.24:75|155:0.13|0.06,alimiter=limit=0.93"
        )

    def speak(self, text: str) -> None:
        if not shutil.which("afplay"):
            raise RuntimeError("afplay is required for Kokoro playback on macOS")
        with tempfile.TemporaryDirectory(prefix="creepy-clown-kokoro-") as directory:
            raw_path = Path(directory) / "voice.wav"
            effect_path = Path(directory) / "voice-creepy.wav"
            self._model.save(
                text,
                raw_path,
                voice=self.voice,
                speed=self.speed,
                sample_rate=24000,
                language="en-us",
            )
            play_path = raw_path
            if self.effect != "clean" and shutil.which("ffmpeg"):
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(raw_path),
                            "-af",
                            self.effect_filter(self.effect),
                            str(effect_path),
                        ],
                        check=True,
                    )
                    play_path = effect_path
                except subprocess.CalledProcessError:
                    LOG.warning("Kokoro effect processing failed; playing clean voice")
            subprocess.run(["afplay", str(play_path)], check=True)
