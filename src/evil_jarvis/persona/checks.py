"""Reply checks a persona can turn on (the `checks` list in its TOML).

Each check gets the reply, its words, the visitor's line, earlier replies and
the persona's style examples, and returns why the reply is unusable, or None.
"""

from __future__ import annotations

import re
import unicodedata

_ENGLISH_GIVEAWAYS = {"the", "you", "and", "is", "are", "i'm", "what", "with", "your", "this"}


SAME_OPENING_WORDS = 3  # replies starting with the same 3 words count as a repeat
ENGLISH_GIVEAWAYS_NEEDED = 2  # one English-looking word can be a Polish loanword
# "Rymy?" asks back "rymami": Polish endings differ, so compare word beginnings.
ECHO_MIN_WORD = 4  # short answers like "Tak?" are fine
ECHO_SHARED_SHARE = 0.75  # share of the reply's word that must match the visitor's


def normalize_text(text: str) -> str:
    """Lowercase and strip diacritics. NFKD leaves ł intact, so map it explicitly."""
    lowered = text.lower().replace("ł", "l")
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def same_reply(reply: str, earlier: str) -> bool:
    """True when a reply equals, or opens with the same three words as, an earlier one."""
    words = re.findall(r"\w+", normalize_text(reply))
    earlier_words = re.findall(r"\w+", normalize_text(earlier))
    if words == earlier_words:
        return True
    opening = SAME_OPENING_WORDS
    return len(words) >= opening and words[:opening] == earlier_words[:opening]


def latin_only(reply, words, visitor, previous, examples):
    for character in reply:
        if character.isalpha() and not unicodedata.name(character, "").startswith("LATIN"):
            return f"non-Latin character {character!r}"
    return None


def no_symbols(reply, words, visitor, previous, examples):
    if any(unicodedata.category(character) == "So" for character in reply):
        return "emoji or symbol"
    return None


def no_stage_directions(reply, words, visitor, previous, examples):
    if re.search(r"\*[^*]+\*|\([^)]*\)|\[[^\]]*\]", reply):
        return "stage direction"
    return None


def not_english(reply, words, visitor, previous, examples):
    if len(_ENGLISH_GIVEAWAYS.intersection(words)) >= ENGLISH_GIVEAWAYS_NEEDED:
        return "English"
    return None


def no_echo(reply, words, visitor, previous, examples):
    if visitor and normalize_text(reply).startswith(normalize_text(visitor).rstrip(" ?!.")):
        return "repeats the visitor"
    return None


def shared_prefix_length(first: str, second: str) -> int:
    length = 0
    for a, b in zip(first, second, strict=False):
        if a != b:
            break
        length += 1
    return length


def no_echo_opening(reply, words, visitor, previous, examples):
    """Reject a stutter ("Mam! Mam...") or the visitor's word asked back ("Rymy? ...")."""
    if len(words) > 1 and words[0] == words[1]:
        return "stutter opening"
    asked = re.match(r"\W*(\w+)\s*\?", reply)
    if asked:
        word = normalize_text(asked.group(1))
        for visitor_word in re.findall(r"\w+", normalize_text(visitor)):
            shared = shared_prefix_length(word, visitor_word)
            if len(word) >= ECHO_MIN_WORD and shared >= ECHO_SHARED_SHARE * len(word):
                return "opens with the visitor's word"
    return None


def no_example_copy(reply, words, visitor, previous, examples):
    if any(normalize_text(reply) == normalize_text(answer) for _, answer in examples):
        return "copies a style example"
    return None


def no_repeat(reply, words, visitor, previous, examples):
    if any(same_reply(reply, earlier) for earlier in previous):
        return "repeats an earlier reply"
    return None


# Checks a persona can turn on, run in this order. Empty replies are always
# rejected, and `max_words` adds a length check.
CHECKS = {
    "latin_only": latin_only,
    "no_symbols": no_symbols,
    "no_stage_directions": no_stage_directions,
    "not_english": not_english,
    "no_echo": no_echo,
    "no_echo_opening": no_echo_opening,
    "no_example_copy": no_example_copy,
    "no_repeat": no_repeat,
}
