"""Shared utilities for the engine-distillation pipeline.

Data record per game (identical to examples/onpolicy_legality.py so pickles
are interchangeable):
  ids    np.int64 array: [BOS] m1 ... mk (token ids)
  legal  list (len = number of plies) of np.int16 arrays of legal token ids,
         legal[t] = legal moves in the position after t moves
  teach  np.int64 array (len = number of plies), teach[t] = teacher move token
         id in the position after t moves, or -1 (no label)

Alignment used everywhere: x = ids[:-1], target of row t is ids[t+1], and
row t describes the board after t moves. The evaluation never queries the
model before ply 8 (games start from 8-ply book openings), so labels are
only produced for t >= LABEL_FROM.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import chess
import numpy as np

from chess_hw import ChessTokenizer
from chess_hw.engine import Engine
from chess_hw.match import load_openings

TOK = ChessTokenizer()
LABEL_FROM = 8

LEVELS = (1, 2, 3, 4)


def make_engine(level: int) -> Engine:
    return Engine(level=level)


def engine_move(board: chess.Board, level: int, rng: random.Random) -> chess.Move:
    if level == 0:
        return rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
    return Engine(level=level).best_move(board)


def open_from_book(rng: random.Random) -> List[str]:
    return list(load_openings()[rng.randrange(len(load_openings()))])


def replay_targets(
    moves: Sequence[str],
    teacher_level: int,
    label_model_plies_only: Optional[chess.Color] = None,
    label_all: bool = False,
    teacher_plied_levels: Tuple[int, int] = (0, 0),
    label_from: int = LABEL_FROM,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """Replay a game and build legal masks + teacher labels.

    teacher_plied_levels: (white_level, black_level) that played the game.
    Plies played by an engine of level == teacher_level get their label for
    free (the played move IS the teacher move: levels 1-4 are deterministic).
    Other plies are labeled with a fresh teacher engine if label_all, or if
    they are model plies (label_model_plies_only) in DAgger mode.
    """
    teacher = Engine(level=teacher_level) if teacher_level > 0 else None
    b = chess.Board()
    legal: list = []
    teach: List[int] = []
    for t, u in enumerate(moves):
        legal.append(np.array([TOK.token_to_id[b.uci(m)] for m in b.legal_moves], dtype=np.int16))
        lab = -1
        if t >= label_from and teacher is not None:
            free = teacher_plied_levels[0 if b.turn == chess.WHITE else 1] == teacher_level
            model_ply = label_model_plies_only is not None and b.turn == label_model_plies_only
            if free or label_all or model_ply:
                if free:
                    lab = TOK.token_to_id[u]
                else:
                    mv = teacher.best_move(b)
                    lab = TOK.token_to_id[b.uci(mv)] if mv is not None else -1
        teach.append(lab)
        b.push_uci(u)
    ids = np.array(TOK.encode(moves), dtype=np.int64)
    return ids, legal, np.array(teach, dtype=np.int64)


def game_batch(games, dev, label_from: int = LABEL_FROM):
    import torch

    T = max(len(g[0]) for g in games) - 1
    B = len(games)
    x = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, TOK.vocab_size, dtype=torch.bool)
    valid = torch.zeros(B, T, dtype=torch.bool)
    teach = torch.full((B, T), -100, dtype=torch.long)
    for i, (ids, legal, tch) in enumerate(games):
        n = len(ids) - 1
        x[i, :n] = torch.from_numpy(ids[:-1])
        valid[i, :n] = True
        for t in range(n):
            mask[i, t, torch.from_numpy(legal[t].astype(np.int64))] = True
        tt = torch.from_numpy(tch)
        teach[i, :n] = torch.where(tt >= 0, tt, torch.full_like(tt, -100))
    ar = torch.arange(T)
    valid &= ar >= label_from
    return x.to(dev), mask.to(dev), valid.to(dev), teach.to(dev)
