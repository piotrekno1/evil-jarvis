import pytest

from evil_jarvis.llm.llama import LlamaResponder


class FakeModel:
    """Stands in for llama-cpp: returns scripted replies and records each call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # (messages, options) per call

    def create_chat_completion(self, messages, **options):
        self.calls.append((messages, options))
        return {
            "choices": [{"message": {"content": self.replies.pop(0)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        }


@pytest.fixture
def responder_with():
    """Builds a LlamaResponder that answers with scripted replies instead of a model."""

    def make(replies, *, persona=None, clock=None):
        responder = object.__new__(LlamaResponder)
        responder._disable_thinking = False
        responder._model = FakeModel(replies)
        if persona:
            responder.persona = persona
        if clock:
            responder.clock = clock
        return responder

    return make
