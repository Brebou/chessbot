#!/usr/bin/env python3
"""Step 1 (training): distill the engine teacher into the model with a legality loss.

Loss = lam_legal * (-log P(legal)) at every labeled ply
     + lam_ce    * CE(teacher move) at labeled plies
     + lam_rep   * next-move CE on Lichess games (optional replay vs forgetting)

Trains from --init (default: base_model), keeps the checkpoint with the best
held-out top-1 teacher agreement, and saves with chess_hw.save_model.

    python distill/train_sft.py --games work/games_sft.pkl --out work/distill_r1
"""

from __future__ import annotations

import argparse
import copy
import glob
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_hw import load_model, save_model  # noqa: E402

from distill.common import TOK, game_batch  # noqa: E402


class Lichess:
    def __init__(self, prefix):
        self.ids = np.fromfile(f"{prefix}.bin", dtype=np.uint16)
        self.off = np.fromfile(f"{prefix}.idx", dtype=np.int64)

    def batch(self, bs, rng):
        idx = rng.integers(0, len(self.off) - 1, bs)
        gs = [self.ids[self.off[i]:self.off[i + 1]].astype(np.int64) for i in idx]
        T = max(len(g) for g in gs)
        x = np.zeros((bs, T), dtype=np.int64)
        for r, g in enumerate(gs):
            x[r, :len(g)] = g
        return torch.from_numpy(x)


def load_games(specs):
    files = []
    for s in specs:
        files.extend(sorted(glob.glob(s)))
    games = []
    for f in files:
        with open(f, "rb") as fh:
            part = pickle.load(fh)
        n_lab = int(sum((g[2] >= 0).sum() for g in part))
        print(f"  {f}: {len(part)} games, {sum(len(g[0]) - 1 for g in part)} plies, {n_lab} labeled", flush=True)
        games.extend(part)
    print(f"loaded {len(games)} games total from {len(files)} file(s)", flush=True)
    return games


@torch.inference_mode()
def dev_metrics(model, dev_games, dev, bs=32):
    model.eval()
    n = top1 = top3 = 0
    ce_sum = 0.0
    for s in range(0, len(dev_games), bs):
        x, mask, valid, teach = game_batch(dev_games[s:s + bs], dev)
        logits = model(x)
        sel = teach >= 0
        if not sel.any():
            continue
        lg = logits[sel].float()
        y = teach[sel]
        top1 += (lg.argmax(-1) == y).sum().item()
        top3 += (lg.topk(3, dim=-1).indices == y[:, None]).any(-1).sum().item()
        ce_sum += F.cross_entropy(lg, y, reduction="sum").item()
        n += sel.sum().item()
    model.train()
    return (top1 / max(n, 1), top3 / max(n, 1), ce_sum / max(n, 1))


def train_model(model, games, args, dev, lich=None, dev_games=None):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(len(games) // args.bs, 1))
    model.train()
    best = (-1.0, None)
    order = list(range(len(games)))
    steps_per_ep = (len(order) + args.bs - 1) // args.bs
    total_steps = steps_per_ep * args.epochs
    log_every = getattr(args, "log_every", 20)
    print(f"training: {steps_per_ep} steps/epoch x {args.epochs} epochs = {total_steps} steps, "
          f"bs={args.bs}, lr={args.lr}", flush=True)
    step = 0
    t_start = time.time()
    for ep in range(args.epochs):
        random.Random(args.seed + ep).shuffle(order)
        stats = np.zeros(3)
        nb = 0
        t0 = time.time()
        for s in range(0, len(order), args.bs):
            x, mask, valid, teach = game_batch([games[i] for i in order[s:s + args.bs]], dev, args.label_from)
            logits = model(x)
            logp = F.log_softmax(logits.float(), -1)
            legal_mass = torch.logsumexp(logp.masked_fill(~mask, -1e9), -1)
            l_legal = -(legal_mass * valid).sum() / valid.sum().clamp(min=1)
            l_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), teach.reshape(-1), ignore_index=-100
            ) if (teach >= 0).any() else torch.zeros((), device=dev)
            if lich is not None:
                xr = lich.batch(args.bs, rng).to(dev)
                lr_logits = model(xr[:, :-1])
                yr = xr[:, 1:].clone()
                yr[yr == 0] = -100
                l_rep = F.cross_entropy(
                    lr_logits.reshape(-1, lr_logits.size(-1)).float(), yr.reshape(-1), ignore_index=-100
                )
            else:
                l_rep = torch.zeros((), device=dev)
            loss = args.lam_legal * l_legal + args.lam_ce * l_ce + args.lam_rep * l_rep
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            stats += [l_legal.item(), l_ce.item(), l_rep.item()]
            nb += 1
            step += 1
            if step % log_every == 0 or step == total_steps:
                dt = time.time() - t_start
                loss_val = (args.lam_legal * stats[0] + args.lam_ce * stats[1]
                            + args.lam_rep * stats[2]) / nb
                print(f"  step {step}/{total_steps} (ep {ep}): loss {loss_val:.4f} "
                      f"legal {stats[0] / nb:.4f} ce {stats[1] / nb:.4f} rep {stats[2] / nb:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} ({dt:.0f}s, {step * args.bs / max(dt, 1e-9):.0f} games/s)",
                      flush=True)
        msg = (f"epoch {ep}: legal {stats[0] / nb:.4f} ce {stats[1] / nb:.4f} "
               f"replay {stats[2] / nb:.4f} ({time.time() - t0:.0f}s)")
        if dev_games:
            a1, a3, ce = dev_metrics(model, dev_games, dev)
            msg += f" | dev top1 {a1:.3f} top3 {a3:.3f} ce {ce:.3f}"
            if a1 > best[0]:
                best = (a1, copy.deepcopy(model.state_dict()))
        print(msg, flush=True)
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"restored best dev top1 {best[0]:.3f}", flush=True)
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True, help="pickle files (globs ok)")
    ap.add_argument("--init", default=str(ROOT / "base_model"))
    ap.add_argument("--out", default="work/distill_r1")
    ap.add_argument("--data", default=None, help="prefix of .bin/.idx Lichess token arrays (optional)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lam_legal", type=float, default=1.0)
    ap.add_argument("--lam_ce", type=float, default=1.0)
    ap.add_argument("--lam_rep", type=float, default=0.5)
    ap.add_argument("--label_from", type=int, default=8)
    ap.add_argument("--dev_frac", type=float, default=0.02)
    ap.add_argument("--dev_max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log_every", type=int, default=20, help="print train losses every N steps")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    lich = Lichess(args.data) if args.data else None

    games = load_games(args.games)
    rnd = random.Random(args.seed)
    idx = rnd.sample(range(len(games)), min(len(games), max(2, int(len(games) * args.dev_frac))))
    dev_set = set(idx)
    dev_games = [games[i] for i in idx][: args.dev_max]
    train_games = [games[i] for i in range(len(games)) if i not in dev_set]
    print(f"train {len(train_games)} / dev {len(dev_games)} games", flush=True)

    model = load_model(args.init, device=dev)
    print(f"device={dev}, params={model.num_parameters()}, init={args.init}", flush=True)
    train_model(model, train_games, args, dev, lich, dev_games)
    out = Path(args.out)
    save_model(model, out)
    print(f"saved -> {out / 'model.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
