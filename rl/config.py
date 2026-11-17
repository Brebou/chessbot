"""RL hyperparameters, with argparse wiring for train_rl.py."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Tuple


@dataclass
class RLConfig:
    # model / output
    model_path: str = "base_model"
    out_dir: str = "results/rl/run"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    seed: int = 0
    max_iters: int = 100

    # rollouts
    levels: Tuple[int, ...] = (1, 2)          # engine levels to train against
    games_per_iter: int = 256
    group_size: int = 4                        # games per (opening, color, level) group
    max_plies: int = 250
    temperature: float = 1.0
    illegal_penalty: float = -0.25             # reward of a sampled illegal move
    engine_workers: int = 0                    # 0 = in-process engine, >0 = process pool

    # rewards / advantages
    result_reward: float = 1.0
    material_shaping: float = 0.05             # reward per pawn of material delta
    gamma: float = 0.997
    baseline: str = "group"                    # "group" | "global"
    normalize_advantages: bool = True

    # optimization
    lr: float = 1e-5
    grad_clip: float = 1.0
    ppo_clip: float = 0.2                      # 0 disables clipping (plain PG)
    inner_epochs: int = 1
    minibatches: int = 1
    entropy_bonus: float = 0.0
    kl_coef: float = 0.05                      # 0 disables the KL to the reference
    train_batch_size: int = 1024               # transitions per forward chunk

    # replay (anti-forgetting)
    replay_coef: float = 0.5                   # 0 disables replay
    replay_path: str = "data/lichess_train_2025-01_1M.jsonl.gz"
    replay_batch_size: int = 128
    replay_buffer_size: int = 50_000

    # eval / checkpoints
    eval_every: int = 10                       # 0 disables evaluation
    eval_levels: Tuple[int, ...] = (0, 1, 2, 3)
    eval_seeds: Tuple[int, ...] = (0, 1)
    eval_n_openings: int = 25
    eval_workers: int = 8
    save_every: int = 5

    def __post_init__(self) -> None:
        if self.baseline not in ("group", "global"):
            raise ValueError(f"unknown baseline: {self.baseline!r}")
        if self.games_per_iter % self.group_size != 0:
            raise ValueError("games_per_iter must be a multiple of group_size")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if not self.levels:
            raise ValueError("levels must not be empty")

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register every field as an --option; lists as nargs=+."""
        for f in fields(cls):
            opt = "--" + f.name.replace("_", "-")
            default = f.default
            if isinstance(default, bool):
                parser.add_argument(opt, type=lambda s: s.lower() in ("1", "true", "yes"),
                                    default=default)
            elif isinstance(default, tuple):
                if default and isinstance(default[0], int):
                    parser.add_argument(opt, nargs="+", type=int, default=list(default))
                else:
                    parser.add_argument(opt, nargs="+", default=list(default))
            elif isinstance(default, int) and not isinstance(default, bool):
                parser.add_argument(opt, type=int, default=default)
            elif isinstance(default, float):
                parser.add_argument(opt, type=float, default=default)
            else:
                parser.add_argument(opt, type=str, default=default)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RLConfig":
        kwargs = {}
        for f in fields(cls):
            val = getattr(args, f.name)
            if isinstance(f.default, tuple):
                val = tuple(val)
            kwargs[f.name] = val
        return cls(**kwargs)
