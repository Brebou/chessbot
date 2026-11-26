#!/usr/bin/env python3
"""Capacity probe: how well does a model imitate the L3 teacher?

Generates held-out teacher positions (L3 self-play from book openings,
different seed from training), then measures:
  top1  fraction of positions where the model's argmax == teacher move
  top3  fraction where the teacher move is in the model's top-3 tokens
        (if legal, the fixed 3-attempt inference then plays the teacher move)
  legal_top3  fraction where at least one of the top-3 tokens is legal

If top1 plateaus at a low value while data scales, the 1M capacity binds.

    python distill/agreement.py --model work/distill_r1 --n_games 64 --workers 8
"""

from __future__ import annotations

import argparse
import os
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

from chess_hw import ChessTokenizer, load_model  # noqa: E402
from chess_hw.engine import Engine  # noqa: E402
from chess_hw.match import load_openings  # noqa: E402

from distill.common import default_workers, progress_iter  # noqa: E402

TOK = ChessTokenizer()


def probe_positions(task):
    gi, seed, max_plies = task
    rng = random.Random(f"probe-{seed}-{gi}")
    book = load_openings()
    board = chess.Board()
    for u in book[rng.randrange(len(book))]:
        board.push_uci(u)
    eng = Engine(level=3)
    positions = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if board.ply() >= 8:
            positions.append(([m.uci() for m in board.move_stack], eng.best_move(board).uci()))
        board.push(eng.best_move(board))
    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base_model")
    ap.add_argument("--n_games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--max_plies", type=int, default=120)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print(f"agreement: model={args.model}, {args.n_games} probe games, seed={args.seed}, "
          f"workers={args.workers}, device will be "
          f"{args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)
    t0 = time.time()
    tasks = [(g, args.seed, args.max_plies) for g in range(args.n_games)]
    if args.workers <= 1:
        it = map(probe_positions, tasks)
    else:
        print(f"starting pool of {args.workers} workers", flush=True)
        pool = get_context("spawn").Pool(args.workers)
        it = pool.imap_unordered(probe_positions, tasks, chunksize=1)
    per_game = []
    for g in progress_iter(it, len(tasks), "probe-gen", every=max(1, args.n_games // 10)):
        per_game.append(g)
    if args.workers > 1:
        pool.close()
        pool.join()
    positions = [p for g in per_game for p in g]
    print(f"{len(positions)} probe positions ({time.time() - t0:.0f}s)", flush=True)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device=dev).eval()
    n = top1 = top3 = legal3 = 0
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(positions), args.bs):
            if s % (args.bs * 10) == 0:
                print(f"  scoring {s}/{len(positions)} positions "
                      f"({time.time() - t0:.0f}s)", flush=True)
            chunk = positions[s:s + args.bs]
            T = max(len(h) for h, _ in chunk) + 1
            x = torch.zeros(len(chunk), T, dtype=torch.long)
            for i, (h, _) in enumerate(chunk):
                ids = TOK.encode(h)
                x[i, :len(ids)] = torch.tensor(ids)
            logits = model(x.to(dev)).float().cpu()
            for i, (_, tmv) in enumerate(chunk):
                t = len(chunk[i][0])
                lg = logits[i, t]
                order = torch.sort(lg, descending=True).indices[:3]
                toks = [TOK.id_to_token[int(j)] for j in order]
                top1 += toks[0] == tmv
                top3 += tmv in toks
                board = chess.Board()
                for u in chunk[i][0]:
                    board.push_uci(u)
                legal = {board.uci(m) for m in board.legal_moves}
                legal3 += any(t in legal for t in toks)
                n += 1
    print(f"model {args.model}: top1 {top1 / n:.3f}  top3 {top3 / n:.3f}  legal_top3 {legal3 / n:.3f}  (n={n})")


if __name__ == "__main__":
    main()
