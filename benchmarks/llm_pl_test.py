"""Quick Polish reply test for small local models with a Polish prompt.

    uv run --extra models python benchmarks/llm_pl_test.py MODEL.gguf [MODEL.gguf ...]

Writes a Markdown report next to this file for a native speaker to judge.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys
import time

from creepy_clown.backends import POLISH_STYLE_EXAMPLES as EXAMPLES  # noqa: E402
from creepy_clown.backends import POLISH_SYSTEM_PROMPT as SYSTEM_PROMPT_PL  # noqa: E402
from creepy_clown.backends import polish_reply_problem  # noqa: E402

# The first three are the prompts from benchmarks/2026-10-02-model-comparison.md.
VISITOR_LINES = [
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
]

SAMPLES_PER_LINE = 2


def build_messages(visitor: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_PL}]
    for question, answer in EXAMPLES:
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": visitor})
    return messages


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"^\s*(klaun|clown)\s*:\s*", "", text.strip(), flags=re.IGNORECASE).strip()


def run_model(path: Path) -> tuple[list[tuple[str, str, float, int]], float]:
    from llama_cpp import Llama

    started = time.perf_counter()
    llm = Llama(model_path=str(path), n_ctx=2048, n_gpu_layers=-1, verbose=False, seed=7)
    load_seconds = time.perf_counter() - started
    results = []
    for visitor in VISITOR_LINES:
        for _ in range(SAMPLES_PER_LINE):
            started = time.perf_counter()
            out = llm.create_chat_completion(
                messages=build_messages(visitor),
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=60,
            )
            seconds = time.perf_counter() - started
            reply = clean(out["choices"][0]["message"]["content"])
            problem = polish_reply_problem(reply, visitor)
            if problem:
                reply = f"{reply} ⟨rejected: {problem}⟩"
            tokens = out["usage"]["completion_tokens"]
            results.append((visitor, reply, seconds, tokens))
            print(f"  {seconds:4.2f}s  {visitor}  ->  {reply}", flush=True)
    del llm
    return results, load_seconds


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    report = [
        f"# Polish reply test, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Polish system prompt plus five Polish example turns; temperature 0.7, "
        f"max 60 tokens, {SAMPLES_PER_LINE} replies per line, Apple GPU (Metal).",
        "",
    ]
    for path in paths:
        print(f"== {path.name}")
        results, load_seconds = run_model(path)
        times = sorted(r[2] for r in results)
        tokens = sum(r[3] for r in results)
        report += [
            f"## {path.name} ({path.stat().st_size / 1e9:.1f} GB)",
            "",
            f"Load {load_seconds:.1f}s · median reply {times[len(times) // 2]:.2f}s · "
            f"{tokens / sum(times):.0f} tokens/s overall",
            "",
            "| Visitor | Reply | s |",
            "|---|---|---|",
        ]
        report += [f"| {v} | {r.replace('|', '/')} | {s:.2f} |" for v, r, s, _ in results]
        report.append("")
    out = Path(__file__).with_name(f"{datetime.now():%Y-%m-%d}-polish-reply-test.md")
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
