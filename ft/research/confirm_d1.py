"""Fresh-seed confirmation of the d1 REINFORCE-lite arm vs base."""
import json
from pathlib import Path

from ft.eval.paired_eval import paired_engine_eval, selfplay_paired
from ft.research.sweep import EVAL_LEVELS


def main() -> None:
    mpath = "results/arc2/d1/final/model.safetensors"
    base = "base_model/model.safetensors"
    for seed, n_open in [(777, 14), (555, 14)]:
        p = paired_engine_eval(mpath, base, levels=EVAL_LEVELS,
                               n_openings=n_open, seed=seed, workers=2,
                               n_bootstrap=1000, records_cache={})
        s = selfplay_paired(mpath, base, n_games=40, seed=seed, workers=2)
        row = {"run": f"confirm/d1-seed{seed}", "engine_delta": p["delta"],
               "sigma": p["sigma"], "per_level": p["per_level_delta"],
               "proxy_delta": s["proxy_delta"],
               "balanced": (p["delta"] + s["proxy_delta"]) / 2.0,
               "openings": n_open, "eval_seed": seed}
        print("CONFIRM: " + json.dumps(row, sort_keys=True), flush=True)
        with open(Path("ft/research/results.jsonl"), "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
