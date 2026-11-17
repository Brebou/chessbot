"""Canonical board/history helpers shared by the ft pipeline.

The model never sees FENs: a position is communicated only through the move
history that led to it. All phases therefore carry full real histories (or
`start_fen` + history-from-synthetic-start for tablebase lines) and the
tokenization window is the model's own context rule: BOS + at most 255 moves,
truncated by dropping an even number of moves from the front (parity).
"""

from __future__ import annotations

from typing import List, Sequence

MAX_MOVES = 255

PIECE_BINS = [(24, 32, "mid"), (16, 23, "late"), (0, 15, "end")]
BIN_NAMES = [name for _, _, name in PIECE_BINS]

PER_GAME_QUOTA = {"mid": 12, "late": 10, "end": 12}
END_BIN_MIN_SPACING = 4
MIN_PLY = 4


def n_pieces_from_fen(fen: str) -> int:
    return sum(1 for c in fen.split()[0] if c.isalpha())


def bin_for_pieces(n: int) -> str:
    for lo, hi, name in PIECE_BINS:
        if lo <= n <= hi:
            return name
    raise ValueError(f"piece count {n} outside bins")


def history_window(moves: Sequence[str], max_moves: int = MAX_MOVES) -> List[str]:
    """Last `max_moves` moves, dropping an even number from the front."""
    if len(moves) <= max_moves:
        return list(moves)
    excess = len(moves) - max_moves
    return list(moves[excess + (excess & 1):])


def truncated_fraction(moves: Sequence[str], max_moves: int = MAX_MOVES) -> float:
    drops = max(0, len(moves) - max_moves)
    if not moves:
        return 0.0
    return drops / len(moves)
