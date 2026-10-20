"""Replies in character from a local GGUF model (llama.cpp)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path

from evil_jarvis.persona import DEFAULT_PERSONA, Persona, load_persona
from evil_jarvis.telemetry import Telemetry

logger = logging.getLogger(__name__)


class LlamaResponder:
    """Replies in character, using the persona's prompt, examples and checks."""

    MEMORY_TURNS = 6  # exchanges kept; longer prompts slow Bielik and blur the character

    # Class defaults keep partially constructed test instances working.
    memory_seconds = 90.0
    clock = staticmethod(time.monotonic)
    _history: tuple[tuple[str, str], ...] = ()
    _last_heard: float | None = None
    _conversation_id: str | None = None
    _next_conversation_id: str | None = None
    _telemetry: Telemetry | None = None
    _llm_handle = None
    _persona: Persona | None = None
    last_reply_info: dict | None = None

    def __init__(
        self,
        model_path: str,
        persona: Persona | None = None,
        memory_seconds: float = 90.0,
        telemetry: Telemetry | None = None,
    ):
        self._persona = persona
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

    @property
    def persona(self) -> Persona:
        if self._persona is None:
            self._persona = load_persona(DEFAULT_PERSONA)
        return self._persona

    @persona.setter
    def persona(self, persona: Persona) -> None:
        self._persona = persona

    @staticmethod
    def strip_transcript_echo(reply: str, transcript: str, labels=("assistant", "reply")) -> str:
        """Drop a leading "Name:" label and an echo of the visitor's words."""
        names = "|".join(re.escape(label) for label in labels)
        reply = re.sub(rf"^\s*(?:{names})\s*:\s*", "", reply, count=1, flags=re.IGNORECASE)
        if reply.casefold().startswith(transcript.casefold()):
            reply = reply[len(transcript) :].lstrip(' \t\r\n:—–-.,!?"')
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
                logger.info("Memory cleared after %.0f s of quiet", now - (self._last_heard or now))
            self._history = ()
            self._conversation_id = self._pending_conversation_id()
            self._next_conversation_id = None
        self._last_heard = now
        return self._history

    def _remember(self, text: str, reply: str) -> None:
        if reply and self.memory_seconds > 0:
            self._history = (*self._history, (text, reply))[-self.MEMORY_TURNS :]

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
        if self._llm_handle is None:
            return self._model.create_chat_completion(messages=messages, **params)
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
        profile = self.persona.profile(language)
        reply = self._generate(text, language, profile, history)
        self._remember(text, reply)
        return reply

    def _generate(self, text: str, language: str | None, profile, history=()) -> str:
        system_prompt, user_message = profile.prompt_for(text, language)
        sound_rule = self.persona.vocalization_rule(language)
        if sound_rule:
            system_prompt = f"{system_prompt}\n{sound_rule}"
        if self._disable_thinking:
            user_message = f"/no_think\n{user_message}"
        messages = [{"role": "system", "content": system_prompt}]
        # The real conversation follows the style examples, so it reads as one dialogue.
        messages += self._history_messages(profile.examples)
        messages += self._history_messages(history)
        messages.append({"role": "user", "content": user_message})
        previous = tuple(answer for _, answer in history)

        rejections: list[str] = []
        info = self._note(source="model", rejections=rejections)
        for attempt in range(1, profile.attempts + 1):
            info["attempts"] = attempt
            result = self._complete(messages=messages, **profile.generation_params)
            content = str(result["choices"][0]["message"]["content"])
            reply = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            # Markdown bold is emphasis, not a stage direction: keep the words.
            reply = re.sub(r"\*\*([^*]+)\*\*", r"\1", reply)
            # Small local models sometimes echo the transcript despite instructions.
            reply = self.strip_transcript_echo(reply, text, self.persona.all_labels)
            problem = profile.reply_problem(reply, text, previous)
            if problem is None:
                logger.info("Reply source: model (attempt %d)", attempt)
                return reply
            logger.info("Rejected model reply (%s): %r", problem, reply[:80])
            rejections.append(problem)
        logger.warning("No usable reply after %d attempts; staying silent", profile.attempts)
        info["source"] = "silent"
        return ""
