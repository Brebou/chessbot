"""Iterated REINFORCE-lite: outcome-weighted corrections, fresh games per gen.

Per generation:
  1. play `(gen g-1 model) vs class engine` at levels [1,2] on a FIXED book
     opening set -> fresh outcome-tagged games (never reused)
  2. REINFORCE-lite mix: winning games -> own-move rows; losing games ->
     SF-best corrections where the played move was a blunder; draws -> own
     moves at low weight
  3. fine-tune (kl 0.1, lr 5e-5, 1 epoch) -> gen g model
  4. paired eval vs base AND vs previous gen (fixed fresh-seed eval)

This is d1 in a loop: the d1 recipe was the first to show reproducible
engine gains (L1 +0.071 / L2 +0.054 on seed 777); this tests compounding.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ft.research.arc2 import (_select_oids, collect_outcome_games,
                              build_reinforce_mix)
from ft.research.sweep import EVAL_LEVELS, resolve_engine, train_candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="base_model/model.safetensors")
    ap.add_argument("--anchor-path", default=None)
    ap.add_argument("--out-dir", default="results/it3")
    ap.add_argument("--n-openings", type=int, default=100)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sf-depth", type=int, default=6)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--w-win", type=float, default=1.0)
    ap.add_argument("--w-loss", type=float, default=1.0)
    ap.add_argument("--w-draw", type=float, default=0.3)
    ap.add_argument("--corpus-seed", type=str, default="it3a")
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--selfplay-games", type=int, default=20)
    ap.add_argument("--eval-workers", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=1234)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.anchor_path or args.model_path
    engine = resolve_engine(args.engine)
    oids = _select_oids(args.corpus_seed, args.n_openings)

    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired

    policy = args.model_path
    for gen in range(1, args.generations + 1):
        t0 = time.monotonic()
        gdir = out_dir / f"gen{gen}"
        gdir.mkdir(parents=True, exist_ok=True)
        print(f"=== it3 gen {gen}: outcome games from {policy} ===",
              flush=True)
        games = collect_outcome_games(policy, oids, args.levels,
                                      seed_tag=f"it3a-gen{gen}")
        n_win = sum(1 for g in games if g["out"] == 1.0)
        n_loss = sum(1 for g in games if g["out"] == 0.0)
        print(f"it3 gen{gen}: {len(games)} games, {n_win} wins, {n_loss} "
              f"losses", flush=True)
        mix_path = out_dir / f"mix_gen{gen}.jsonl"
        if not mix_path.exists():
            n_rows, _ = build_reinforce_mix(
                games, mix_path, engine_path=engine, sf_depth=args.sf_depth,
                sf_workers=args.sf_workers, w_win=args.w_win,
                w_loss=args.w_loss, w_draw=args.w_draw, cap=args.cap)
            print(f"it3 gen{gen}: mix {n_rows} rows", flush=True)
        done = train_candidate(
            mix_path, model_path=policy, base_path=base_path,
            out_dir=gdir / "model", epochs=args.epochs, lr=args.lr,
            kl=args.kl, batch=args.batch, seed=gen,
            eval_openings=args.eval_openings,
            selfplay_games=args.selfplay_games,
            eval_workers=args.eval_workers, records_cache={},
            device=args.device)
        mpath = done.get("final_model", str(gdir / "model" / "final" /
                                            "model.safetensors"))
        p = paired_engine_eval(mpath, base_path, levels=EVAL_LEVELS,
                               n_openings=args.eval_openings,
                               seed=args.eval_seed, workers=args.eval_workers,
                               n_bootstrap=1000, records_cache={})
        sl = selfplay_paired(mpath, base_path, n_games=args.selfplay_games,
                             seed=args.eval_seed, workers=args.eval_workers)
        row = {"run": f"it3/gen{gen}", "engine_delta": p["delta"],
               "sigma": p["sigma"], "per_level": p["per_level_delta"],
               "proxy_delta": sl["proxy_delta"],
               "balanced": (p["delta"] + sl["proxy_delta"]) / 2.0,
               "n_win": n_win, "n_loss": n_loss,
               "mix_rows": n_rows,
               "elapsed_s": round(time.monotonic() - t0)}
        if gen > 1:
            prev = out_dir / f"gen{gen-1}" / "model" / "final" / \
                "model.safetensors"
            p2 = paired_engine_eval(mpath, str(prev), levels=EVAL_LEVELS,
                                    n_openings=args.eval_openings,
                                    seed=args.eval_seed,
                                    workers=args.eval_workers,
                                    n_bootstrap=500, records_cache={})
            row["policy_vs_prev"] = p2["delta"]
        with open(Path("ft/research/results.jsonl"), "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"=== it3 gen {gen} done: vs_base={row['engine_delta']:+.4f} "
              f"vs_prev={row['policy_vs_prev']} "
              f"balanced={row['balanced']:+.4f} ({row['elapsed_s']}s) ===",
              flush=True)
        policy = mpath


if __name__ == "__main__":
    main()
