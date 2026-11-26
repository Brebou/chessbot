#!/usr/bin/env python3
"""GRPO: group-relative policy optimization against the engine.

Each iteration:
  1. roll out G games per "prompt" (one book opening + fixed color + fixed
     opponent level) with the current policy, sampling moves from the
     legal-move-masked softmax at temperature --temp;
  2. reward = game score for the model side (1 / 0.5 / 0);
  3. advantage = (r - mean) / (std + eps) within each group;
  4. one gradient step: loss = -mean(adv * logp(sampled moves))
     + kl_beta * mean(logp - ref_logp) (sample-based KL anchor to --ref).

Sampling is cheap (model forward ~2 ms/move), so opponents should be fast
levels (1, 2). L3 rollouts are ~15x slower and rarely fit a time budget.

    python distill/grpo.py --init work/distill_r2 --out work/grpo \
        --iters 15 --games_per_iter 128 --group 8 --opps 1 2 --workers 15
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_hw import ChessTokenizer, load_model, save_model  # noqa: E402
from chess_hw.engine import Engine  # noqa: E402
from chess_hw.match import load_openings, score_for  # noqa: E402

TOK = ChessTokenizer()
_M = {}


def _init(path):
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _M["model"] = load_model(path).eval()


def _sample_move(board: chess.Board, temp: float, gen: torch.Generator):
    ids = torch.tensor([TOK.encode([m.uci() for m in board.move_stack])])
    logits = _M["model"](ids)[0, -1].float()
    legal = sorted(board.legal_moves, key=lambda m: m.uci())
    idx = torch.tensor([TOK.token_to_id[board.uci(m)] for m in legal])
    masked = torch.full_like(logits, -1e9)
    masked[idx] = logits[idx]
    probs = torch.softmax(masked / temp, -1)
    tok_id = int(torch.multinomial(probs, 1, generator=gen).item())
    mv = board.parse_uci(TOK.id_to_token[tok_id])
    return mv, tok_id


def _episode(task):
    try:
        grp, j, gi, seed, color, opp_level, temp, max_plies = task
        gen = torch.Generator().manual_seed(seed)
        board = chess.Board()
        for u in load_openings()[gi]:
            board.push_uci(u)
        positions, moves_played, token_ids = [], [], []
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if board.turn == color:
                mv, tok_id = _sample_move(board, temp, gen)
                positions.append(len(moves_played))
                token_ids.append(tok_id)
            else:
                mv = Engine(level=opp_level).best_move(board)
            moves_played.append(mv.uci())
            board.push(mv)
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome is not None else "1/2-1/2"
        termination = outcome.termination.name.lower() if outcome is not None else "max_plies"
        return {
            "grp": grp, "j": j,
            "ids": np.array(TOK.encode(moves_played), dtype=np.int64),
            "positions": np.array(positions, dtype=np.int64),
            "token_ids": np.array(token_ids, dtype=np.int64),
            "reward": score_for(result, color),
            "termination": termination,
            "plies": len(moves_played),
            "opp_level": opp_level,
        }
    except Exception as e:
        raise RuntimeError(f"grpo episode failed on task {task}: {type(e).__name__}: {e}") from e


def move_logps(model, episodes, dev):
    """log p of the sampled moves for a batch of episodes (padded, causal-safe)."""
    B = len(episodes)
    T = max(len(e["ids"]) for e in episodes) - 1
    x = torch.zeros(B, T, dtype=torch.long)
    for i, e in enumerate(episodes):
        n = len(e["ids"]) - 1
        x[i, :n] = torch.from_numpy(e["ids"][:-1])
    logits = model(x.to(dev)).float()
    batch = torch.cat([torch.full((len(e["positions"]),), i, dtype=torch.long) for i, e in enumerate(episodes)])
    rows = torch.cat([torch.from_numpy(e["positions"]) for e in episodes])
    cols = torch.cat([torch.from_numpy(e["token_ids"]) for e in episodes])
    lg = logits[batch, rows]
    logp = F.log_softmax(lg, -1)[torch.arange(len(cols)), cols]
    return logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="policy to start from")
    ap.add_argument("--ref", default=str(ROOT / "base_model"), help="KL anchor")
    ap.add_argument("--out", default="work/grpo")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--games_per_iter", type=int, default=128)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--opps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_plies", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_beta", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="default: cgroup CPU quota")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from distill.common import default_workers

    workers = args.workers or default_workers()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"grpo: init={args.init} ref={args.ref} iters={args.iters} "
          f"games/iter={args.games_per_iter} group={args.group} opps={args.opps} "
          f"temp={args.temp} lr={args.lr} kl_beta={args.kl_beta} "
          f"workers={workers} device={dev}", flush=True)

    book = load_openings()
    n_groups = args.games_per_iter // args.group
    rng = random.Random(args.seed)
    model = load_model(args.init, device=dev)
    ref = load_model(args.ref, device=dev).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pool = None
    if workers > 1:
        print(f"starting pool of {workers} workers", flush=True)
        pool = get_context("spawn").Pool(workers, initializer=_init, initargs=(args.init,))

    for it in range(args.iters):
        t0 = time.time()
        groups = []
        for g in range(n_groups):
            gi = rng.randrange(len(book))
            color = rng.random() < 0.5
            opp = rng.choice(args.opps)
            groups.append([(g, j, gi, args.seed * 1_000_003 + it * 100_003 + g * 1009 + j,
                            color, opp, args.temp, args.max_plies)
                           for j in range(args.group)])
        tasks = [t for grp in groups for t in grp]
        if pool is not None:
            results = list(pool.imap_unordered(_episode, tasks, chunksize=1))
        else:
            results = [_episode(t) for t in tasks]
        by_grp = {}
        for e in results:
            by_grp.setdefault(e["grp"], []).append(e)
        t_roll = time.time() - t0

        model.train()
        opt.zero_grad()
        total_moves = 0
        loss_sum = 0.0
        for g, eps in by_grp.items():
            r = np.array([e["reward"] for e in eps])
            adv = (r - r.mean()) / (r.std() + 1e-4)
            counts = [len(e["positions"]) for e in eps]
            adv_move = torch.cat([
                torch.full((c,), float(a), dtype=torch.float32)
                for c, a in zip(counts, adv)
            ]).to(dev)
            logp = move_logps(model, eps, dev)
            with torch.inference_mode():
                ref_logp = move_logps(ref, eps, dev)
            n = len(logp)
            total_moves += n
            loss_g = (-(adv_move * logp).sum() / n
                      + args.kl_beta * (logp - ref_logp).mean()) / n_groups
            loss_g.backward()
            loss_sum += loss_g.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        t_train = time.time() - t0 - t_roll

        rewards = np.array([e["reward"] for e in results])
        per_opp = {}
        for e in results:
            per_opp.setdefault(e["opp_level"], []).append(e["reward"])
        opp_msg = " ".join(f"L{k}: {np.mean(v):.3f}(n={len(v)})" for k, v in sorted(per_opp.items()))
        ends = {}
        for e in results:
            ends[e["termination"]] = ends.get(e["termination"], 0) + 1

        print(f"iter {it}: reward {rewards.mean():.3f} ({opp_msg}) loss {loss_sum:.4f} "
              f"moves {total_moves} ends {ends} | roll {t_roll:.0f}s train {t_train:.0f}s",
              flush=True)

        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            save_model(model, args.out if it == args.iters - 1 else f"{args.out}_it{it + 1}")
            print(f"  saved -> {args.out}" if it == args.iters - 1
                  else f"  snapshot -> {args.out}_it{it + 1}", flush=True)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
