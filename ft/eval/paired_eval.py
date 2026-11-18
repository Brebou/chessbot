"""Paired evaluation - the answer to '+-0.03 governs everything'.

The candidate and the base model play the EXACT same engine games (same
levels, same opening seeds, same colors, same deterministic engine seeds),
so opponent variance cancels in the per-cell delta. Cells are
(level, opening_id, model_color). sigma comes from a cell bootstrap
(2000 resamples); 'new best' = delta_balanced - 2*sigma > 0.

The selfplay Elo proxy is paired too: candidate vs reference over fixed
openings with fixed colors; the proxy delta = mean candidate score - 0.5.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import chess

from chess_hw.match import (EnginePlayer, load_openings, play_game, run_tasks,
                            select_openings, score_for, _get_player,
                            _pair_task)
from chess_hw.play import MoveChoice
from diag.fast_engine import FastEngine


class _FastEnginePlayer:
    """EnginePlayer with the move-identical FastEngine (persistent TT, ~4x)."""

    def __init__(self, level: int, seed: str):
        self.name = f"engine-L{level}"
        self.engine = FastEngine(level=level, seed=seed)

    def choose(self, board: chess.Board) -> MoveChoice:
        return MoveChoice(self.engine.best_move(board), 1)


def _fast_engine_task(task):
    name, level, seed, oid, model_color = task
    model = _get_player(name)
    seed_str = f"engine-{seed}-{level}-{oid}-{model_color}"
    if level == 0:
        engine = EnginePlayer(level, seed=seed_str)
    else:
        engine = _FastEnginePlayer(level, seed=seed_str)
    opening = load_openings()[oid]
    white, black = (model, engine) if model_color == "white" else (engine, model)
    rec = play_game(white, black, opening, oid)
    rec.meta = {"level": level, "seed": seed, "model_color": model_color}
    return asdict(rec)


def _engine_records(model_path: str, tasks, workers: int, progress: bool = False):
    return run_tasks(_fast_engine_task, tasks, {"m": str(model_path)},
                     workers, progress)


def paired_engine_eval(candidate_path: str, base_path: str, *, levels=(0, 1, 2, 3, 4),
                       n_openings: int = 40, seed: int = 0, workers: int = 8,
                       n_bootstrap: int = 2000, records_cache: Optional[Dict] = None
                       ) -> dict:
    """`records_cache` (optional) memoizes per-model records dict across calls
    so the BASE model's games are simulated once for a whole training run."""
    tasks: List[Tuple] = []
    for lvl in levels:
        for oid in select_openings(seed, n_openings):
            for color in ("white", "black"):
                tasks.append(("m", lvl, seed, oid, color))

    def records_for(path: str):
        if records_cache is None:
            return _engine_records(path, tasks, workers)
        if path not in records_cache:
            records_cache[path] = _engine_records(path, tasks, workers)
        return records_cache[path]

    base = records_for(base_path)
    cand = records_for(candidate_path)

    cell_keys = [(lvl, oid, color) for lvl, oid, color in
                 [(t[1], t[3], t[4]) for t in tasks]]
    cand_scores: Dict[Tuple, float] = {}
    base_scores: Dict[Tuple, float] = {}
    for key, rc, rb in zip(cell_keys, cand, base):
        color = chess.WHITE if key[2] == "white" else chess.BLACK
        cand_scores[key] = score_for(rc["result"], color)
        base_scores[key] = score_for(rb["result"], color)
    deltas = [cand_scores[k] - base_scores[k] for k in cell_keys]
    mean_delta = sum(deltas) / len(deltas)
    metric_stds = []
    rng = random.Random(f"bootstrap-{seed}")
    n = len(deltas)
    for _ in range(n_bootstrap):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        metric_stds.append(sum(sample) / n)
    mean_of_means = sum(metric_stds) / len(metric_stds)
    sigma = (sum((m - mean_of_means) ** 2 for m in metric_stds) / (n_bootstrap - 1)) ** 0.5
    per_level = {}
    for (lvl, _, _), d in zip(cell_keys, deltas):
        per_level.setdefault(str(lvl), []).append(d)
    return {"delta": mean_delta, "sigma": sigma,
            "significant": mean_delta - 2 * sigma > 0,
            "n_cells": len(cell_keys),
            "per_level_delta": {k: sum(v) / len(v) for k, v in per_level.items()},
            "base_records": base}


def selfplay_paired(candidate_path: str, ref_path: str, *, n_games: int = 200,
                    seed: int = 0, workers: int = 8) -> dict:
    oids = select_openings(seed, n_games)
    tasks = [("a", "b", oid, aw) for oid in oids for aw in (True, False)]
    recs = run_tasks(_pair_task, tasks, {"a": str(candidate_path), "b": str(ref_path)},
                     workers, progress=False)
    scores = [score_for(r["result"], chess.WHITE if r["meta"]["a_white"] else chess.BLACK)
              for r in recs]
    mean = sum(scores) / max(len(scores), 1)
    return {"proxy_score": mean, "proxy_delta": mean - 0.5, "n_games": len(scores)}


def balanced_delta(paired: dict, proxy: dict) -> float:
    return (paired["delta"] + proxy["proxy_delta"]) / 2.0


def paired_stop_criterion(history: Sequence[float], threshold: float = 0.02,
                          patience: int = 2) -> bool:
    """Stop when the balanced paired delta stays below threshold for
    `patience` consecutive checkpoints."""
    if len(history) < patience + 1:
        return False
    return all(d < threshold for d in history[-patience:])
