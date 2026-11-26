#!/usr/bin/env python3
"""Paired held-out evaluation: several models on the SAME openings and seeds.

Computes, per model: S_k for each level, P = mean over levels 1..3, and the
delta vs the first model (the baseline). Uses chess_hw.match exactly as the
official evaluation (only the seed differs from the private one).

    python distill/eval_paired.py --models base_model work/distill_r1 \
        --seed 777 --n_openings 100 --levels 1 2 3 --workers 8
    # --verbose prints one line per game (result, termination, fallbacks)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_hw.match import (  # noqa: E402
    _engine_task,
    load_openings,
    run_tasks,
    score_for,
    select_openings,
    summarize_vs_engine,
)


def run_eval(name, model_path, args, verbose):
    oids = select_openings(args.seed, args.n_openings)
    tasks = [(name, lvl, args.seed, oid, color)
             for lvl in args.levels for oid in oids for color in ("white", "black")]
    records = [None] * len(tasks)
    agg = {}
    done = [0]
    t0 = time.time()

    def on_result(i, rec):
        records[i] = rec
        lvl = rec["meta"]["level"]
        color = chess.WHITE if rec["meta"]["model_color"] == "white" else chess.BLACK
        s = score_for(rec["result"], color)
        a = agg.setdefault(lvl, {"n": 0, "s": 0.0})
        a["n"] += 1
        a["s"] += s
        done[0] += 1
        if verbose:
            st = rec["white_stats"] if color == chess.WHITE else rec["black_stats"]
            print(f"  [{name}] L{lvl} {rec['meta']['model_color']} oid={rec['opening_id']} "
                  f"{rec['result']} ({rec['termination']}, {len(rec['moves'])} plies, "
                  f"fallbacks={st['fallbacks']}) score={s}", flush=True)
        elif done[0] % args.log_every == 0 or done[0] == len(tasks):
            scores = " ".join(f"L{k}: {v['s'] / v['n']:.3f} ({v['n']}g)" for k, v in sorted(agg.items()))
            print(f"  [{name}] {done[0]}/{len(tasks)} games ({time.time() - t0:.0f}s) {scores}", flush=True)

    records = run_tasks(_engine_task, tasks, {name: str(model_path)}, args.workers,
                        progress=not verbose, on_result=on_result)
    summary = summarize_vs_engine(records)
    summary.update({"model": str(model_path), "seed": args.seed,
                    "n_openings": args.n_openings, "levels": list(args.levels)})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n_openings", type=int, default=100)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--verbose", action="store_true", help="print one line per game")
    ap.add_argument("--out", default=None, help="optional json output")
    args = ap.parse_args()

    results = {}
    for m in args.models:
        print(f"\n=== {m} (seed {args.seed}, {args.n_openings} openings/level, "
              f"levels {args.levels}, workers {args.workers}) ===", flush=True)
        s = run_eval("m", m, args, args.verbose)
        results[m] = s
        pl = {k: v["score"] for k, v in s["per_level"].items()}
        P = sum(pl.get(str(k), 0.0) for k in args.levels) / len(args.levels)
        print(f"S = {pl}  P = {P:.3f}  legal@1 = {s['legal_first_try']:.3f}  "
              f"legal@3 = {s['legal_within_3']:.3f}  random-move rate = {s['random_move_rate']:.3f}",
              flush=True)

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
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
