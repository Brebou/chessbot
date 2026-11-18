"""Engine-free mate-in-1 padding from a general (any-Elo) game dump.

Elite play yields almost no mate positions (9,345 of 1,527,462 annotated
positions in the first ft run) so the mate bucket is the bottleneck of the
whole mix. This module scans a general dump for positions with a mate-in-1
and emits mate rows carrying the full real move history - no Stockfish.

Rows go through Phase E as `--extra-rows` and are capped/subsampled with
everything else. Mate-in-2 stays sourced from Phase C annotations (verified
by find_mate_in_2 and capped at 30% of the mate bucket in Phase E).

    uv run ft/data/pad_mates.py --dataset data/lichess_uci_201403.jsonl.gz \
        --out data/ft/mate_pad.jsonl --max-rows 60000 --workers 20
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess

from ft.common.serialize import jsonl_write


def _first_mating_move(board: chess.Board, prefer: str = "") -> str:
    found = ""
    for mv in board.generate_legal_moves():
        if not board.gives_check(mv):
            continue
        if prefer and mv.uci() == prefer:
            return prefer
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate and not found:
            found = mv.uci()
    return found


def _scan_game(task: Tuple[str, str, int, int]) -> List[dict]:
    gid, moves_str, min_ply, max_per_game = task
    moves = moves_str.split()
    board = chess.Board()
    history: List[str] = []
    rows: List[dict] = []
    for ply, uci in enumerate(moves):
        if ply >= min_ply and len(rows) < max_per_game:
            mate_uci = _first_mating_move(board, prefer=uci)
            if mate_uci:
                rows.append({"bucket": "mate", "game_id": gid, "ply": ply,
                             "key": board.fen(), "history": list(history),
                             "target": mate_uci, "mate_n": 1})
        try:
            board.push_uci(uci)
        except ValueError:
            break
        history.append(uci)
    return rows


def mine_mate_rows(dataset: str, out: str, *, n_games: int = 0,
                   max_rows: int = 60_000, max_per_game: int = 1,
                   min_ply: int = 2, workers: int = 8, force: bool = False
                   ) -> Dict[str, int]:
    out_path = Path(out)
    if out_path.exists() and not force:
        print(f"✅ {out_path} exists, skipping the mate padding "
              f"(delete it to force)")
        return {"rows": -1}
    from ft.common.games_io import read_games

    games = read_games(dataset)
    if n_games:
        games = games[:n_games]
    if not games:
        sys.exit(f"⛔ {dataset} contains 0 games")
    print(f"⏳ mate padding: scanning {len(games)} games for mate-in-1 "
          f"positions (no engine, pure python-chess)...")

    tasks = [(f"pad{i}", g, min_ply, max_per_game)
             for i, g in enumerate(games)]
    seen: set = set()
    rows_out: List[dict] = []
    scanned = 0
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for rows in pool.imap_unordered(_scan_game, tasks, chunksize=32):
            scanned += 1
            for row in rows:
                if row["key"] in seen or len(rows_out) >= max_rows:
                    continue
                seen.add(row["key"])
                rows_out.append(row)
            if scanned % 25_000 == 0:
                print(f"  mate padding: games {scanned}/{len(tasks)} | "
                      f"{len(rows_out)} rows", flush=True)
            if len(rows_out) >= max_rows:
                break

    jsonl_write(out_path, rows_out)
    print(f"✅ mate padding: {len(rows_out)} rows from {scanned} games "
          f"-> {out_path}")
    return {"rows": len(rows_out)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine mate-in-1 rows from a general game dump (no engine)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-games", type=int, default=0, help="0 = all")
    parser.add_argument("--max-rows", type=int, default=60_000)
    parser.add_argument("--max-per-game", type=int, default=1)
    parser.add_argument("--min-ply", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    mine_mate_rows(args.dataset, args.out, n_games=args.n_games,
                   max_rows=args.max_rows, max_per_game=args.max_per_game,
                   min_ply=args.min_ply, workers=args.workers,
                   force=args.force)


if __name__ == "__main__":
    main()
