"""Generic paired-eval confirmation CLI: re-score a candidate vs base on
fresh eval seeds. Use it to confirm any near-miss checkpoint before
promoting it to the submission candidate.

Example (GPU box):
  uv run python -m ft.research.confirm \
      --model-path results/run10/gen2/x1/final/model.safetensors \
      --base-path base_model/model.safetensors \
      --seeds 777 888 999 --openings 40 --eval-workers 22 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ft.research.sweep import EVAL_LEVELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-path", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 888])
    ap.add_argument("--openings", type=int, default=40)
    ap.add_argument("--selfplay-games", type=int, default=100)
    ap.add_argument("--eval-workers", type=int, default=22)
    ap.add_argument("--tag", default="confirm")
    args = ap.parse_args()
    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired
    for seed in args.seeds:
        p = paired_engine_eval(args.model_path, args.base_path,
                               levels=EVAL_LEVELS, n_openings=args.openings,
                               seed=seed, workers=args.eval_workers,
                               n_bootstrap=1000, records_cache={})
        s = selfplay_paired(args.model_path, args.base_path,
                            n_games=args.selfplay_games, seed=seed,
                            workers=args.eval_workers)
        row = {"run": f"{args.tag}/seed{seed}",
               "engine_delta": p["delta"], "sigma": p["sigma"],
               "per_level": p["per_level_delta"],
               "proxy_delta": s["proxy_delta"],
               "balanced": (p["delta"] + s["proxy_delta"]) / 2.0,
               "openings": args.openings, "eval_seed": seed,
               "qualifies": p["delta"] - 2 * p["sigma"] > 0}
        print("CONFIRM: " + json.dumps(row, sort_keys=True), flush=True)
        with open(Path("ft/research/results.jsonl"), "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
