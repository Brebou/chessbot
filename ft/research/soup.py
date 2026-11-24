"""Model soup arm: weight-average closed REINFORCE-arm finals.

All campaign finals share the base anchor (KL 0.1 leash + rehearsal), so
they sit in the same loss basin (Wortsman et al. 2022, "Model soups":
weight-space averaging of same-basin fine-tunes often beats every member).
Never tried in this campaign; kills two birds: variance-reduction on the
near-zero arms and a way to combine L1-heavy and L2-heavy variants.

CLI:
  uv run python -m ft.research.soup \
    --models results/arc2/f2/final/model.safetensors results/arc2g/g1/final/model.safetensors \
    --out results/soup/s1 \
    --base-path base_model/model.safetensors \
    --eval-openings 10 --eval-seed 1234 --eval-workers 22
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file

from chess_hw.model import load_model

EVAL_LEVELS = (1, 2, 3)
RESULTS = Path("ft/research/results.jsonl")


def soup_state_dicts(paths: List[str], *,
                     base_path: Optional[str] = None,
                     base_weight: float = 0.0) -> Dict[str, torch.Tensor]:
    """Uniform-average state dicts (shout: tied embeddings fine, same keys).

    base_weight > 0 blends the soup toward the anchor:[(1-w)*mean(models)
    + w*base], the WiSE-FT variant (Wortsman 2022) that keeps banked play.
    """
    if not paths:
        raise SystemExit("--models needs at least one checkpoint")
    acc: Dict[str, torch.Tensor] = {}
    n = 0
    for p in paths:
        sd = load_file(p)
        if acc and set(sd) != set(acc):
            raise SystemExit(f"key mismatch at {p} — not the same architecture")
        for k, v in sd.items():
            acc[k] = acc.get(k, torch.zeros_like(v, dtype=torch.float32))\
                + v.to(torch.float32)
        n += 1
    soup = {k: v / n for k, v in acc.items()}
    if base_path and base_weight > 0:
        base = load_file(base_path)
        soup = {k: (1.0 - base_weight) * v + base_weight * base[k].to(torch.float32)
                for k, v in soup.items()}
    return soup


def save_soup(soup: Dict[str, torch.Tensor], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model.safetensors"
    save_file(soup, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-path", default="base_model/model.safetensors")
    ap.add_argument("--base-weight", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--selfplay-games", type=int, default=20)
    ap.add_argument("--eval-workers", type=int, default=22)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    t0 = time.monotonic()
    soup = soup_state_dicts(args.models, base_path=args.base_path,
                            base_weight=args.base_weight)
    out = save_soup(soup, Path(args.out))
    print(f"soup: {len(args.models)} members -> {out} "
          f"({time.monotonic()-t0:.0f}s, base_weight={args.base_weight})")

    if args.no_eval:
        return

    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired

    p = paired_engine_eval(str(out), args.base_path, levels=EVAL_LEVELS,
                           n_openings=args.eval_openings, seed=args.eval_seed,
                           workers=args.eval_workers, n_bootstrap=1000)
    sl = selfplay_paired(str(out), args.base_path,
                         n_games=args.selfplay_games, seed=args.eval_seed,
                         workers=args.eval_workers)
    row = {"run": f"soup/{Path(args.out).name}",
           "members": args.models, "base_weight": args.base_weight,
           "engine_delta": p["delta"], "sigma": p["sigma"],
           "per_level": p["per_level_delta"],
           "cand": p["cand_score"], "base_score": p.get("base_score"),
           "proxy_delta": sl["proxy_delta"],
           "balanced": (p["delta"] + sl["proxy_delta"]) / 2.0,
           "eval_seed": args.eval_seed, "openings": args.eval_openings,
           "elapsed_s": round(time.monotonic() - t0)}
    print("SOUP:", json.dumps(row))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
