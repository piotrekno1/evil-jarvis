"""Fetching the models and voices a run needs (--download)."""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Download:
    what: str
    path: Path
    fetch: Callable[[], None]


def huggingface_file(repo: str, filename: str, directory: Path) -> Callable[[], None]:
    def fetch() -> None:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo, filename, local_dir=directory)

    return fetch


def huggingface_repo(repo: str, directory: Path) -> Callable[[], None]:
    def fetch() -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=directory)

    return fetch


def urls(pairs: list[tuple[str, Path]]) -> Callable[[], None]:
    """Download each URL to its path.

    Each file is written to a .part file first, so an interrupted download is
    never mistaken for a finished one.
    """

    def fetch() -> None:
        for url, path in pairs:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            with urllib.request.urlopen(url) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(path)

    return fetch
