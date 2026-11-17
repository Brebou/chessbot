"""RL training entry point (Phase 3): policy gradient vs the provided engine.

Usage (GPU machine):
    uv run train_rl.py --model-path base_model --out-dir results/rl/run1 \
        --levels 1 2 --games-per-iter 256 --group-size 4 \
        --lr 1e-5 --kl-coef 0.05 --replay-coef 0.5 \
        --eval-every 10 --eval-seeds 0 1 --eval-workers 16 --device cuda

The run directory receives config.json, metrics.jsonl, periodic checkpoints,
the best-by-multi-seed-eval checkpoint (best/model.safetensors — the
submission candidate) and final/model.safetensors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.config import RLConfig
from rl.train import RLTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="RL training for ChessGPT")
    RLConfig.add_cli_args(parser)
    args = parser.parse_args()
    cfg = RLConfig.from_args(args)
    summary = RLTrainer(cfg).train()
    print("\nDone:", summary)


if __name__ == "__main__":
    main()
