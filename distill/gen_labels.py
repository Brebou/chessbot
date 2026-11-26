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
import os
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

from distill.common import default_workers, engine_move, progress_iter, replay_targets, wrap_task_errors  # noqa: E402


def _init():
    import os

    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _gen_one_impl(task):
    gi, seed, white_level, black_level, max_plies, teacher_level, label_all, qdepth = task
    rng = random.Random(f"gen-{seed}-{gi}")
    book = load_openings()
    opening = book[rng.randrange(len(book))]
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        level = white_level if board.turn == chess.WHITE else black_level
        board.push(engine_move(board, level, rng, qdepth))
    moves = [m.uci() for m in board.move_stack]
    ids, legal, teach = replay_targets(
        moves,
        teacher_level=teacher_level,
        label_all=label_all,
        teacher_plied_levels=(white_level, black_level),
    )
    return gi, ids, legal, teach


def gen_one(task):
    return wrap_task_errors(_gen_one_impl, task)


def save_games(games, out: Path, done=None):
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(games, fh)
    os.replace(tmp, out)
    if done is not None:
        tmp = out.with_suffix(out.suffix + ".done.tmp")
        with open(tmp, "w") as fh:
            fh.write("\n".join(map(str, sorted(done))))
        os.replace(tmp, out.with_suffix(out.suffix + ".done"))


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
    ap.add_argument("--qdepth", type=int, default=4,
                    help="quiescence depth of the engines (lower = faster, slightly weaker labels)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--out", default="work/games_sft.pkl")
    ap.add_argument("--resume", action="store_true",
                    help="skip games already present in --out (same --seed/--n_games required)")
    ap.add_argument("--log_every", type=int, default=20,
                    help="print a progress line every N games")
    args = ap.parse_args()
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    print(f"gen_labels: {args.n_games} games, white=L{args.white} black=L{args.black} "
          f"teacher=L{args.teacher} label_all={args.label_all} max_plies={args.max_plies} "
          f"workers={args.workers} seed={args.seed}", flush=True)
    print(f"cpu cores available to this process: {affinity}", flush=True)
    if args.workers > affinity:
        print(f"WARNING: workers={args.workers} > {affinity} available cores; "
              f"workers will share cores and run slower. "
              f"Consider --workers {affinity}.", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    games = []
    done_file = out.with_suffix(out.suffix + ".done")
    if args.resume and out.exists() and done_file.exists():
        with open(out, "rb") as fh:
            games = pickle.load(fh)
        with open(done_file) as fh:
            done = {int(x) for x in fh.read().split() if x}
        print(f"resume: {len(games)} games already in {out}, skipping them", flush=True)
    elif args.resume:
        print("resume: no existing checkpoint found, starting fresh", flush=True)
    tasks = [
        (g, args.seed, args.white, args.black, args.max_plies, args.teacher, args.label_all, args.qdepth)
        for g in range(args.n_games) if g not in done
    ]
    t0 = time.time()
    stats = {"plies": 0, "labeled": 0}
    if games:
        stats["plies"] = sum(len(g[0]) - 1 for g in games)
        stats["labeled"] = sum(int((g[2] >= 0).sum()) for g in games)
    since_ckpt = 0

    def status():
        return f"plies={stats['plies']} labeled={stats['labeled']}"

    if args.workers <= 1:
        print("running single-process (workers<=1)", flush=True)
        it = map(gen_one, tasks)
    else:
        print(f"starting pool of {args.workers} workers (spawn, ~5-20s startup, "
              f"first games finish in ~30-60s)", flush=True)
        pool = get_context("spawn").Pool(args.workers, initializer=_init)
        it = pool.imap_unordered(gen_one, tasks, chunksize=1)
    for item in progress_iter(it, len(tasks), "gen", every=args.log_every, extra=status):
        gi, ids, legal, teach = item
        games.append((ids, legal, teach))
        stats["plies"] += len(ids) - 1
        stats["labeled"] += int((teach >= 0).sum())
        since_ckpt += 1
        done.add(gi)
        if since_ckpt >= args.log_every:
            save_games(games, out, done)
            since_ckpt = 0
    if args.workers > 1:
        pool.close()
        pool.join()
    save_games(games, out, done)
    n_plies = stats["plies"]
    n_lab = stats["labeled"]
    print(f"{len(games)} games, {n_plies} plies, {n_lab} labeled positions "
          f"({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
