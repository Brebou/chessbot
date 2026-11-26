#!/usr/bin/env python3
"""Step 1: label positions with the provided engine as teacher.

Plays games between engine levels (default: L3 vs L3) from random book
openings and labels every ply with the teacher's move. Labels on plies
played by an engine of the teacher level are free (levels 1-4 are
deterministic, so the played move IS the teacher move).

    python distill/gen_labels.py --n_games 2000 --seed 0 --out work/games_sft.pkl
    python distill/gen_labels.py --n_games 1000 --white 3 --black 2 --label_all --teacher 3 ...

Cost: L3 self-play ~26 s per 100-ply game on one core; scale --workers.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_hw.engine import Engine  # noqa: E402
from chess_hw.match import load_openings  # noqa: E402

from distill.common import engine_move, replay_targets  # noqa: E402

_M = {}


def _init():
    pass


def gen_one(task):
    gi, seed, white_level, black_level, max_plies, teacher_level, label_all = task
    rng = random.Random(f"gen-{seed}-{gi}")
    book = load_openings()
    opening = book[rng.randrange(len(book))]
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        level = white_level if board.turn == chess.WHITE else black_level
        board.push(engine_move(board, level, rng))
    moves = [m.uci() for m in board.move_stack]
    ids, legal, teach = replay_targets(
        moves,
        teacher_level=teacher_level,
        label_all=label_all,
        teacher_plied_levels=(white_level, black_level),
    )
    return ids, legal, teach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--white", type=int, default=3)
    ap.add_argument("--black", type=int, default=3)
    ap.add_argument("--teacher", type=int, default=3)
    ap.add_argument("--label_all", action="store_true",
                    help="also label plies not played by the teacher level (costs one teacher search per ply)")
    ap.add_argument("--max_plies", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, __import__("os").cpu_count() - 1))
    ap.add_argument("--out", default="work/games_sft.pkl")
    args = ap.parse_args()

    tasks = [
        (g, args.seed, args.white, args.black, args.max_plies, args.teacher, args.label_all)
        for g in range(args.n_games)
    ]
    t0 = time.time()
    if args.workers <= 1:
        games = [gen_one(t) for t in tasks]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            games = pool.map(gen_one, tasks, chunksize=2)
    n_plies = sum(len(g[0]) - 1 for g in games)
    n_lab = int(sum((g[2] >= 0).sum() for g in games))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(games, fh)
    print(f"{len(games)} games, {n_plies} plies, {n_lab} labeled positions "
          f"({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
