#!/usr/bin/env python3
"""Paired held-out evaluation: several models on the SAME openings and seeds.

Computes, per model: S_k for each level, P = mean over levels 1..3, and the
delta vs the first model (the baseline). Uses chess_hw.match exactly as the
official evaluation (only the seed differs from the private one).

    python distill/eval_paired.py --models base_model work/distill_r1 \
        --seed 777 --n_openings 100 --levels 1 2 3 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_hw.match import evaluate_vs_engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n_openings", type=int, default=100)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="optional json output")
    args = ap.parse_args()

    results = {}
    for m in args.models:
        print(f"\n=== {m} (seed {args.seed}, {args.n_openings} openings/level) ===", flush=True)
        s = evaluate_vs_engine(m, levels=args.levels, n_openings=args.n_openings,
                               seed=args.seed, workers=args.workers, progress=True)
        results[m] = s
        pl = {k: v["score"] for k, v in s["per_level"].items()}
        P = sum(pl.get(str(k), 0.0) for k in args.levels) / len(args.levels)
        fb = s["random_move_rate"]
        print(f"S = {pl}  P = {P:.3f}  legal@3 = {s['legal_within_3']:.3f}  "
              f"random-move rate = {fb:.3f}", flush=True)

    base = args.models[0]
    P = {m: sum(results[m]["per_level"].get(str(k), {"score": 0.0})["score"] for k in args.levels) / len(args.levels)
         for m in args.models}
    print(f"\nPaired summary (baseline = {base}):")
    for m in args.models:
        print(f"  {m}: P = {P[m]:.3f}  (dP = {P[m] - P[base]:+.3f})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"per_model": results, "P": P}, f, indent=2)


if __name__ == "__main__":
    main()
