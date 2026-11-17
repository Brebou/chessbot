"""Atomic jsonl IO for the ft package (tmp + os.replace, repo convention)."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Iterator


def _opener(path: str | Path):
    return gzip.open if str(path).endswith(".gz") else open


def jsonl_read(path: str | Path) -> Iterator[dict]:
    with _opener(path)(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def jsonl_write(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with _opener(path)(tmp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def jsonl_count(path: str | Path) -> int:
    n = 0
    with _opener(path)(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
