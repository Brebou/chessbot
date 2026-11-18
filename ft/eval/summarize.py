"""One-glance progress view over a training run's metrics.jsonl.

Usage (any time, mid-run included):

    uv run ft/eval/summarize.py results/ft/run3/metrics.jsonl

Prints the run header (evals so far, span, best qualifying checkpoint,
latest eval) then one aligned row per eval. Malformed/partial trailing
lines (an eval being written while you run this) are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _n(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:+.4f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def _p(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def load_rows(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    rows.sort(key=lambda r: r.get("step", 0))
    return rows


def summarize(path: str) -> dict:
    rows = load_rows(path)
    qualified = [r for r in rows if r.get("significant")]
    best = max(qualified, key=lambda r: r.get("balanced", -9.9), default=None)
    print(f"{path}: {len(rows)} evals, "
          f"steps {rows[0].get('step', '?')}-{rows[-1].get('step', '?')}"
          if rows else f"{path}: 0 evals")
    if best is not None:
        print(f"best qualified: {best.get('eval_tag')} "
              f"(balanced {best['balanced']:+.4f})")
    else:
        print("best qualified: none (delta - 2 sigma > 0 never met)")
    latest = rows[-1] if rows else None
    if latest is not None:
        print(f"latest: {latest.get('eval_tag')} balanced "
              f"{latest.get('balanced', 0):+.4f} mate1 "
              f"{_p((latest.get('fixtures') or {}).get('mate1_top1_rate'))}")
    print(f"{'tag':>10} {'step':>6} {'balanced':>10} {'engine':>10} "
          f"{'proxy':>10} {'mate1':>6} {'wc_late':>8} {'lc_late':>8} "
          f"{'legal_end':>9}")
    for r in rows:
        fx = r.get("fixtures") or {}
        star = "*" if r.get("significant") else " "
        print(f"{str(r.get('eval_tag', '?')):>10} {r.get('step', 0):>6} "
              f"{_n(r.get('balanced')):>9}{star} {_n(r.get('engine_delta')):>10} "
              f"{_n(r.get('proxy_delta')):>10} {_p(fx.get('mate1_top1_rate')):>6} "
              f"{_p(fx.get('winning_capture_rate_late')):>8} "
              f"{_p(fx.get('losing_capture_rate_late')):>8} "
              f"{_p(fx.get('top1_legal_end')):>9}")
    latest = latest or {}
    return {"n_evals": len(rows), "qualified": len(qualified), "latest": latest}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Progress summary over a run's metrics.jsonl")
    parser.add_argument("metrics", help="path to metrics.jsonl")
    args = parser.parse_args()
    if not Path(args.metrics).exists():
        parser.error(f"{args.metrics} does not exist")
    summarize(args.metrics)


if __name__ == "__main__":
    main()
