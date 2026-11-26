#!/usr/bin/env python3
"""Round 2: DAgger — play the current model against the engine, relabel its
positions with the teacher, aggregate with all previous data, retrain from base.

Each round:
  1. play the current model vs engine levels --opps from book openings
     (random-legal fallback on all-3-illegal, exactly like chess_hw.match);
  2. label the model's plies with the L3 teacher (and plies played by an
     engine of the teacher level for free);
  3. retrain from --init on old + new games with train_sft.train_model.

    python distill/dagger.py --model work/distill_r1 --games 2000 --round 2 \
        --prev "work/games_sft.pkl" --out work/distill_r2
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_hw import load_model, save_model  # noqa: E402
from chess_hw.engine import Engine  # noqa: E402
from chess_hw.match import load_openings  # noqa: E402
from chess_hw.play import ModelPlayer  # noqa: E402

from distill.common import replay_targets  # noqa: E402
from distill.train_sft import Lichess, load_games, train_model  # noqa: E402

_M = {}


def _init(path):
    torch.set_num_threads(1)
    _M["p"] = ModelPlayer(load_model(path))


def _dagger_one(task):
    gi, seed, opps, max_plies, teacher_level = task
    rng = random.Random(f"dagger-{seed}-{gi}")
    book = load_openings()
    opening = book[rng.randrange(len(book))]
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    color = rng.random() < 0.5
    opp_level = rng.choice(opps)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if board.turn == color:
            mv = _M["p"].choose(board).move or rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        else:
            mv = Engine(level=opp_level).best_move(board)
        board.push(mv)
    moves = [m.uci() for m in board.move_stack]
    white_lvl = 0 if color == chess.WHITE else opp_level
    black_lvl = opp_level if color == chess.WHITE else 0
    ids, legal, teach = replay_targets(
        moves,
        teacher_level=teacher_level,
        label_model_plies_only=color,
        teacher_plied_levels=(white_lvl, black_lvl),
    )
    return ids, legal, teach, {"opp_level": opp_level, "color": "white" if color else "black"}


def collect(model_path, n_games, seed, opps, max_plies, teacher_level, workers):
    tasks = [(g, seed, opps, max_plies, teacher_level) for g in range(n_games)]
    if workers <= 1:
        return [_dagger_one(t) for t in tasks]
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(str(model_path),)) as pool:
        return pool.map(_dagger_one, tasks, chunksize=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="current model (played on-policy)")
    ap.add_argument("--prev", nargs="*", default=[], help="previous game pickles to aggregate")
    ap.add_argument("--out", default="work/distill_r2", help="output model dir")
    ap.add_argument("--games_out", default=None, help="where to store the new games pickle")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opps", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--teacher", type=int, default=3)
    ap.add_argument("--label_all", action="store_true")
    ap.add_argument("--max_plies", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, __import__("os").cpu_count() - 1))
    ap.add_argument("--init", default=str(ROOT / "base_model"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lam_legal", type=float, default=1.0)
    ap.add_argument("--lam_ce", type=float, default=1.0)
    ap.add_argument("--lam_rep", type=float, default=0.5)
    ap.add_argument("--label_from", type=int, default=8)
    ap.add_argument("--dev_frac", type=float, default=0.02)
    ap.add_argument("--dev_max", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t0 = time.time()
    new = collect(args.model, args.games, args.seed, args.opps, args.max_plies, args.teacher, args.workers)
    games = [(i, l, t) for (i, l, t, _) in new]
    print(f"collected {len(games)} on-policy games ({time.time() - t0:.0f}s)", flush=True)
    games_out = args.games_out or f"{args.out}_games.pkl"
    Path(games_out).parent.mkdir(parents=True, exist_ok=True)
    with open(games_out, "wb") as fh:
        pickle.dump(games, fh)

    all_games = load_games(args.prev) + games
    rnd = random.Random(args.seed)
    idx = rnd.sample(range(len(all_games)), min(len(all_games), max(2, int(len(all_games) * args.dev_frac))))
    dev_set = set(idx)
    dev_games = [all_games[i] for i in idx][: args.dev_max]
    train_games = [all_games[i] for i in range(len(all_games)) if i not in dev_set]
    print(f"train {len(train_games)} / dev {len(dev_games)} games", flush=True)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    lich = Lichess(args.data) if args.data else None
    model = load_model(args.init, device=dev)
    train_model(model, train_games, args, dev, lich, dev_games)
    save_model(model, args.out)
    print(f"saved -> {Path(args.out) / 'model.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
