"""Phase B - position sampling with piece-count binning (replaces ply weights).

One replay per game; decision positions are sampled per piece-count bin with
per-game quotas (mid 12 / late 10 / end 12; deliberately over-represents
finales, the final global subsample lives in Phase E) and end-bin plies
spaced by >= 4 plies to limit correlation.

Positions carry everything downstream phases need:

    {"game_id", "ply", "fen_before", "uci_played", "n_pieces", "bin"}

Exclusion of engine-dependent cases (forced mate in progress, time-scrappy
moves) happens at annotation/bucket time (Phases C/E); sampling only applies
the cheap rules (ply >= MIN_PLY, non-terminal positions).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import chess

from ft.common.board_io import (END_BIN_MIN_SPACING, MIN_PLY, PER_GAME_QUOTA,
                                bin_for_pieces, n_pieces_from_fen)
from ft.common.serialize import jsonl_write


def sample_game_positions(task: Tuple[str, str, dict]) -> List[dict]:
    """(game_id, moves_str, cfg) -> position rows; module-level pool worker."""
    game_id, moves_str, cfg = task
    rng_seed = cfg["seed"]
    quotas = cfg.get("quotas", PER_GAME_QUOTA)
    min_ply = cfg.get("min_ply", MIN_PLY)
    end_spacing = cfg.get("end_spacing", END_BIN_MIN_SPACING)
    rng = random.Random(f"sample-{rng_seed}-{game_id}")
    moves = moves_str.split()
    board = chess.Board()

    candidates: Dict[str, List[Tuple[int, str, str]]] = {"mid": [], "late": [], "end": []}
    for ply, uci in enumerate(moves):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break
        if ply >= min_ply:
            fen_before = board.fen()
            name = bin_for_pieces(n_pieces_from_fen(fen_before))
            candidates[name].append((ply, fen_before, uci))
        try:
            board.push_uci(uci)
        except ValueError:
            break

    out: List[dict] = []
    for name in ("mid", "late", "end"):
        cands = sorted(candidates[name])
        quota = quotas.get(name, len(cands))
        if quota <= 0:
            continue
        if name == "end":
            spaced: List[Tuple[int, str, str]] = []
            last_ply = -end_spacing
            for ply, fen_before, uci in cands:
                if ply - last_ply >= end_spacing:
                    spaced.append((ply, fen_before, uci))
                    last_ply = ply
            if len(spaced) > quota:
                spaced = rng.sample(spaced, quota)
            kept = spaced
        elif len(cands) > quota:
            kept = rng.sample(cands, quota)
        else:
            kept = cands
        for ply, fen_before, uci in sorted(kept):
            out.append({"game_id": game_id, "ply": ply, "fen_before": fen_before,
                        "uci_played": uci, "n_pieces": n_pieces_from_fen(fen_before),
                        "bin": name})
    out.sort(key=lambda r: r["ply"])
    return out


def _sample_task(task: Tuple[str, str, dict]) -> List[dict]:
    return sample_game_positions(task)


def sample_positions(games: List[str], out: str, *, seed: int, workers: int = 8,
                     progress=None) -> int:
    from multiprocessing import get_context

    cfg = {"seed": seed, "quotas": PER_GAME_QUOTA, "min_ply": MIN_PLY,
           "end_spacing": END_BIN_MIN_SPACING}
    tasks = [(f"g{i}", mv, cfg) for i, mv in enumerate(games)]
    rows: List[dict] = []
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for res in pool.imap(_sample_task, tasks, chunksize=16):
            rows.extend(res)
            if progress is not None:
                progress.show(f"sampling: {len(rows)} positions")
    rows.sort(key=lambda r: (r["game_id"], r["ply"]))
    jsonl_write(out, rows)
    return len(rows)
