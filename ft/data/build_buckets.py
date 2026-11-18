"""Phase E - bucket assignment, dedup, and the final mix.

Buckets (example ratios of the final dataset):

    normal      45%  elite positions, margin < 30cp, tranquil move
    mate        20%  mate-in-1 + verified mate-in-2 (mate2 capped at 30%)
    endgame_tb  15%  tablebase-labeled lines (Phase D) + elite finales
                     re-labeled by the tablebase when <= 5 pieces
    contrastive 15%  (good=best_uci, bad=uci_played) pairs, capped 60k
    anchor       5%  played == best, margin < 15cp, calm

A position exists in exactly one bucket: dedup key (fen-before, side), and
priority mate > endgame_tb > contrastive > normal > anchor. Failed-sanity
and invalid-move rows are dropped and counted. `ply_weight` is gone: the
piece binning is the structural equalizer.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import chess

from ft.common.serialize import jsonl_read, jsonl_write
from ft.data.select_criteria import (LOSING_CAPTURE_SEE_CP, best_move_is_quiet,
                                     find_mate_in_2, is_hanging_after,
                                     is_losing_capture)

BUCKET_RATIOS = {"normal": 0.45, "mate": 0.20, "endgame_tb": 0.15,
                 "contrastive": 0.15, "anchor": 0.05}
CONTRASTIVE_CAP = 60_000
MATE2_SHARE_CAP = 0.30
NORMAL_MARGIN_CP = 30
ANCHOR_MARGIN_CP = 15
TRANQUIL_GAP_CP = 50

BUCKET_WEIGHTS = {"normal": 1.0, "mate": 1.0, "endgame_tb": 1.5,
                  "contrastive": 1.0, "anchor": 1.0}


def classify_position(ann: dict) -> Optional[str]:
    """Bucket for one annotated position (priority order); None = drop."""
    if not ann.get("sanity", True):
        return None
    mate_root = ann.get("mate_root")
    if mate_root is not None and mate_root > 0:
        return "mate"
    played = ann["uci_played"]
    losing = is_losing_capture(ann["fen"], played)
    if played != ann["best_uci"] and (
            is_hanging_after(ann["fen"], played)
            or (losing is not None and losing <= LOSING_CAPTURE_SEE_CP)):
        return "contrastive"
    margin = ann["margin"]
    tranquil = ((ann["cp_best"] - ann["cp_second"]) < TRANQUIL_GAP_CP
                or best_move_is_quiet(chess.Board(ann["fen"]), ann["best_uci"]))
    if margin < NORMAL_MARGIN_CP and tranquil:
        return "normal"
    if played == ann["best_uci"] and margin < ANCHOR_MARGIN_CP:
        return "anchor"
    return None


def build_rows(games: List[str], positions_path: str, annotations_path: str,
               *, tb_rows_path: str = "", tb_adapter=None, seed: int = 0
               ) -> "tuple[List[dict], Counter]":
    """Attach real histories to annotated positions, classify, dedup."""
    positions = list(jsonl_read(positions_path))
    anns = {r["fen"]: r for r in jsonl_read(annotations_path)}
    by_game: Dict[str, List[dict]] = defaultdict(list)
    for r in positions:
        by_game[r["game_id"]].append(r)

    classified: List[dict] = []
    drop_stats: Counter = Counter()
    for gi, game in enumerate(games):
        gid = f"g{gi}"
        plies = {r["ply"]: r for r in by_game.get(gid, [])}
        if not plies:
            continue
        board = chess.Board()
        history: List[str] = []
        for ply, uci in enumerate(game.split()):
            if ply in plies:
                r = plies[ply]
                ann = anns.get(r["fen_before"])
                if ann is None:
                    drop_stats["unannotated"] += 1
                elif ann.get("invalid"):
                    drop_stats["invalid"] += 1
                else:
                    bucket = classify_position(ann)
                    if bucket is None:
                        drop_stats["unclassified"] += 1
                    else:
                        row = {"bucket": bucket, "game_id": gid, "ply": ply,
                               "bin": r["bin"], "margin": ann["margin"],
                               "key": r["fen_before"], "history": list(history)}
                        if bucket == "contrastive":
                            row["target"] = ann["best_uci"]
                            row["bad"] = ann["uci_played"]
                        else:
                            row["target"] = ann["uci_played"]
                            if bucket == "mate":
                                row["mate_n"] = ann["mate_root"]
                                if ann["mate_root"] == 2 and not find_mate_in_2(
                                        chess.Board(ann["fen"])):
                                    drop_stats["mate2_unverified"] += 1
                                    continue
                        if tb_adapter is not None and r["n_pieces"] <= 5:
                            tb_row = _tb_relabel(board, tb_adapter)
                            if tb_row is not None:
                                row["bucket"] = "endgame_tb"
                                row["target"] = tb_row
                        classified.append(row)
            try:
                board.push_uci(uci)
            except ValueError:
                break
            history.append(uci)

    deduped: Dict[str, dict] = {}
    for name in ("mate", "endgame_tb", "contrastive", "normal", "anchor"):
        for row in classified:
            if row["bucket"] == name:
                deduped.setdefault(row["key"], row)
    rows = list(deduped.values())

    if tb_rows_path:
        for r in jsonl_read(tb_rows_path):
            rows.append(r)
    for row in rows:
        row["weight"] = BUCKET_WEIGHTS[row["bucket"]]
    return rows, drop_stats


def _tb_relabel(board: chess.Board, adapter) -> Optional[str]:
    from ft.data.tablebase import best_move_for_winner
    probe = adapter.wdl_dtz(board)
    if probe is None or probe[0] <= 0:
        return None
    mv = best_move_for_winner(board, adapter)
    return mv.uci() if mv is not None else None


def subsample_to_ratios(rows: List[dict], *, target_total: Optional[int] = None,
                        seed: int = 0) -> List[dict]:
    """Per-bucket subsample to the target ratios; shortfall is redistributed
    to buckets with spare supply, proportionally."""
    rng = random.Random(f"subsample-{seed}")
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)
    supply = {b: len(by_bucket[b]) for b in BUCKET_RATIOS}
    target_total = target_total or sum(supply.values())
    desired = {b: min(int(round(ratio * target_total)), supply[b])
               for b, ratio in BUCKET_RATIOS.items()}
    allocated = dict(desired)
    shortfall = sum(int(round(ratio * target_total)) - desired[b]
                    for b, ratio in BUCKET_RATIOS.items())
    donors = [b for b in BUCKET_RATIOS if supply[b] > allocated[b]]
    donor_spare = sum(supply[b] - allocated[b] for b in donors)
    if donors and shortfall > 0 and donor_spare:
        for b in donors:
            spare_ratio = (supply[b] - allocated[b]) / donor_spare
            extra = min(supply[b] - allocated[b],
                        int(round(shortfall * spare_ratio)))
            allocated[b] += extra
    out: List[dict] = []
    for b in BUCKET_RATIOS:
        idxs = list(range(supply[b]))
        rng.shuffle(idxs)
        keep = set(idxs[:allocated.get(b, 0)])
        out.extend(r for i, r in enumerate(by_bucket[b]) if i in keep)
    rng.shuffle(out)
    return out


def count_report(rows: List[dict]) -> Dict[str, int]:
    return dict(Counter(r["bucket"] for r in rows))


def _apply_caps(rows: List[dict], seed: int) -> List[dict]:
    mate = [r for r in rows if r["bucket"] == "mate"]
    mate2 = [r for r in mate if r.get("mate_n") == 2]
    if mate and mate2:
        max_mate2 = int(MATE2_SHARE_CAP * len(mate))
        if len(mate2) > max_mate2:
            rng = random.Random(f"cap-mate2-{seed}")
            drop = set(rng.sample(range(len(mate2)), len(mate2) - max_mate2))
            dropped_ids = {id(mate2[i]) for i in drop}
            rows = [r for r in rows if id(r) not in dropped_ids]
    contrastive = [r for r in rows if r["bucket"] == "contrastive"]
    if len(contrastive) > CONTRASTIVE_CAP:
        rng = random.Random(f"cap-contrastive-{seed}")
        drop = set(rng.sample(range(len(contrastive)),
                              len(contrastive) - CONTRASTIVE_CAP))
        dropped_ids = {id(contrastive[i]) for i in drop}
        rows = [r for r in rows if id(r) not in dropped_ids]
    return rows


def build_bucket_dataset(games: List[str], positions_path: str,
                         annotations_path: str, out: str, *,
                         tb_rows_path: str = "", tb_adapter=None,
                         target_total: Optional[int] = None, seed: int = 0,
                         force: bool = False, progress=None) -> Dict[str, int]:
    out_path = Path(out)
    if out_path.exists() and not force:
        counts = count_report(list(jsonl_read(out_path)))
        print(f"✅ {out_path} exists ({sum(counts.values())} rows), skipping")
        return counts
    rows, drop_stats = build_rows(games, positions_path, annotations_path,
                                  tb_rows_path=tb_rows_path, tb_adapter=tb_adapter,
                                  seed=seed)
    if drop_stats:
        print(f"⚠️ dropped: {dict(drop_stats)}")
    rows = _apply_caps(rows, seed)
    rows = subsample_to_ratios(rows, target_total=target_total, seed=seed)
    jsonl_write(out_path, rows)
    counts = count_report(rows)
    print(f"✅ wrote {len(rows)} rows to {out_path}: {counts}")
    return counts


def main() -> None:
    from ft.common.games_io import read_games

    parser = argparse.ArgumentParser(
        description="Phase E - bucket assignment, dedup, final mix")
    parser.add_argument("--games", required=True, help="Phase A output jsonl")
    parser.add_argument("--positions", required=True, help="Phase B output jsonl")
    parser.add_argument("--annotations", required=True, help="Phase C output jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tb-rows", default="", help="Phase D output jsonl")
    parser.add_argument("--tb-dir", default="",
                        help="Syzygy dir; relabels elite rows with <= 5 pieces "
                             "by the tablebase (they keep their real history)")
    parser.add_argument("--target-total", type=int, default=0,
                        help="0 = keep all classified rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    games = read_games(args.games)
    tb_adapter = None
    if args.tb_dir:
        from ft.data.tablebase import SyzygyAdapter
        tb_adapter = SyzygyAdapter(args.tb_dir)
    counts = build_bucket_dataset(games, args.positions, args.annotations,
                                  args.out, tb_rows_path=args.tb_rows,
                                  tb_adapter=tb_adapter,
                                  target_total=args.target_total or None,
                                  seed=args.seed, force=args.force)
    if tb_adapter is not None:
        tb_adapter.close()
    print(f"✅ mix: {sum(counts.values())} rows {counts}")


if __name__ == "__main__":
    main()
