"""Lichess replay sampler: anti-forgetting CE batches for RL training.

Streams games from the training corpus (``{"moves": "uci uci ..."}`` JSONL,
plain or gzip) through a seeded shuffle buffer, and emits supervised batches
``([BOS] m1 ... mk) -> m_{k+1}``. Long games are truncated by dropping an
even number of moves from the front (parity preserved), matching the
inference-time truncation. If the corpus file is missing, the sampler
reports ``available = False`` and the trainer simply skips the replay loss.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


class ReplaySampler:
    """Deterministic streaming shuffle-buffer sampler over a JSONL corpus."""

    def __init__(self, path: str, tokenizer, batch_size: int = 128,
                 buffer_size: int = 50_000, seed: int = 0, max_moves: int = 255):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.buffer_size = max(buffer_size, batch_size)
        self.max_moves = max_moves
        self.available = self.path.exists()
        self._rng = random.Random(f"replay-{seed}")
        self._buffer: List[str] = []
        self._file = None
        self._epoch = 0
        self._empty_epochs = 0

    def _reopen(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = _open_text(self.path)
        self._epoch += 1

    def _extract_moves(self, line: str) -> Optional[str]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None
        moves = row.get("moves")
        return moves if isinstance(moves, str) and moves.strip() else None

    def _fill(self) -> bool:
        """Try to add one game to the buffer; returns False if the corpus is
        exhausted (a few consecutive empty epochs = no valid games at all)."""
        if self._file is None:
            self._reopen()
        line = self._file.readline()
        if not line:
            self._file.close()
            self._file = None
            if self._buffer:
                return True
            self._empty_epochs += 1
            if self._empty_epochs > 2:
                return False
            return self._fill()
        moves = self._extract_moves(line)
        if moves is None:
            return True
        self._empty_epochs = 0
        self._buffer.append(moves)
        if len(self._buffer) > self.buffer_size:
            j = self._rng.randrange(len(self._buffer))
            self._buffer[j] = self._buffer[-1]
            self._buffer.pop()
        return True

    def next_batch(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Next (input_ids (B, T) int64, target_ids (B,) int64) batch, or None
        if the corpus is unavailable or exhausted."""
        if not self.available:
            return None
        while len(self._buffer) < self.batch_size:
            if not self._fill():
                break
        if not self._buffer:
            return None
        batch = []
        while len(batch) < self.batch_size and self._buffer:
            j = self._rng.randrange(len(self._buffer))
            batch.append(self._buffer[j])
            self._buffer[j] = self._buffer[-1]
            self._buffer.pop()
        return self._encode(batch)

    def _encode(self, moves_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tok = self.tokenizer.token_to_id
        histories: List[List[int]] = []
        targets: List[int] = []
        for moves in moves_list:
            ids = [tok[m] for m in moves.split() if m in tok]
            if len(ids) < 2:
                continue
            if len(ids) > self.max_moves:
                excess = len(ids) - self.max_moves
                ids = ids[excess + (excess & 1):]
            histories.append(ids[:-1])
            targets.append(ids[-1])
        n = len(histories)
        if n == 0:
            return (np.empty((0, 0), dtype=np.int64),
                    np.empty((0,), dtype=np.int64))
        lengths = np.fromiter((1 + len(h) for h in histories), dtype=np.int64, count=n)
        arr = np.full((n, int(lengths.max())), self.tokenizer.pad_token_id, dtype=np.int64)
        for i, h in enumerate(histories):
            arr[i, 0] = self.tokenizer.bos_token_id
            arr[i, 1:1 + len(h)] = h
        return arr, np.asarray(targets, dtype=np.int64)

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
