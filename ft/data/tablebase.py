"""Phase D - tablebase finals (Syzygy 3-4-5), perfect labels.

The model is FEN-blind: a position only exists for it as a move history.
So instead of emitting bare positions, each sampled start position is played
out as a tablebase-optimal line - the winning side plays DTZ-minimizing
moves, the losing side maximizes survival - and every winning-side decision
along the line becomes an `endgame_tb` example with the real prefix history
plus `start_fen` (same convention as the old endgame demos).

DTZ of the start position is spread over per-value buckets in
[dtz_lo, dtz_hi], so the model learns progressive marching, not only
execution of the near mate. Winner colors alternate (half the lines won by
Black). Adapter contract: `wdl_dtz(board) -> (wdl, dtz)` from the
side-to-move POV, wdl in {-2..2} (2 = side to move wins), dtz positive when
the side to move is winning, or None when unprobed.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import chess

TB_WEIGHT = 1.5
TB_TYPES = ("KRK", "KQK", "KQKP", "KRKP", "KPK")


class SyzygyAdapter:
    def __init__(self, tb_dir: str):
        self.tablebase = chess.syzygy.open_tablebase(tb_dir)

    def wdl_dtz(self, board: chess.Board):
        try:
            wdl = self.tablebase.probe_wdl(board)
            dtz = self.tablebase.probe_dtz(board)
        except (chess.syzygy.MissingTableError, KeyError):
            return None
        if dtz is None:
            return None
        return int(wdl), int(dtz)

    def close(self):
        self.tablebase.close()


def best_move_for_winner(board: chess.Board, adapter) -> Optional[chess.Move]:
    """Move that keeps the win for the side to move, minimizing DTZ."""
    best, best_dtz = None, None
    for mv in board.legal_moves:
        board.push(mv)
        probe = adapter.wdl_dtz(board)
        board.pop()
        if probe is None:
            continue
        wdl, dtz = probe
        if wdl >= 0:
            continue
        distance = abs(dtz)
        if best_dtz is None or distance < best_dtz:
            best, best_dtz = mv, distance
    return best


def best_move_for_loser(board: chess.Board, adapter) -> Optional[chess.Move]:
    """Survival move: maximize the remaining distance."""
    best, best_dtz = None, None
    for mv in board.legal_moves:
        board.push(mv)
        probe = adapter.wdl_dtz(board)
        board.pop()
        if probe is None:
            continue
        wdl, dtz = probe
        if wdl <= 0:
            continue
        if best_dtz is None or dtz > best_dtz:
            best, best_dtz = mv, dtz
    return best


def tb_line_rows(start_fen: str, adapter, *, max_plies: int = 80) -> List[dict]:
    board = chess.Board(start_fen)
    probe = adapter.wdl_dtz(board)
    if probe is None or probe[0] <= 0:
        return []
    winner = board.turn
    rows: List[dict] = []
    history: List[str] = []
    for i in range(max_plies):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break
        if board.turn == winner:
            target = best_move_for_winner(board, adapter)
            if target is None:
                break
            rows.append({"bucket": "endgame_tb", "start_fen": start_fen,
                         "history": list(history), "target": target.uci(),
                         "weight": TB_WEIGHT})
        else:
            target = best_move_for_loser(board, adapter)
            if target is None:
                break
        history.append(target.uci())
        board.push(target)
    return rows


def _random_placement(rng: random.Random,
                      pieces: List[Tuple[int, bool]]) -> chess.Board:
    board = chess.Board(None)
    squares = rng.sample(range(64), len(pieces))
    for (ptype, color), sq in zip(pieces, squares):
        board.set_piece_at(sq, chess.Piece(ptype, color))
    return board


_START_SHAPES: Dict[str, List[Tuple[int, bool]]] = {
    "KRK": [(chess.KING, True), (chess.ROOK, True), (chess.KING, False)],
    "KQK": [(chess.KING, True), (chess.QUEEN, True), (chess.KING, False)],
    "KQKP": [(chess.KING, True), (chess.QUEEN, True), (chess.KING, False),
             (chess.PAWN, False)],
    "KRKP": [(chess.KING, True), (chess.ROOK, True), (chess.KING, False),
             (chess.PAWN, False)],
    "KPK": [(chess.KING, True), (chess.PAWN, True), (chess.KING, False)],
}


def probe_start(board: chess.Board, adapter) -> Optional[Tuple[str, int]]:
    """Return (fen, dtz) with the winning side to move, else None. The white
    and black-to-move variants of the placement are both probed so winners
    alternate naturally."""
    for turn in (chess.WHITE, chess.BLACK):
        board.turn = turn
        if board.status() != chess.STATUS_VALID:
            continue
        probe = adapter.wdl_dtz(board)
        if probe is None:
            continue
        wdl, dtz = probe
        if wdl > 0 and dtz > 0:
            return board.fen(), dtz
    return None


def sample_tb_starts(adapter, shape: str, n_lines: int, *, dtz_lo: int = 1,
                     dtz_hi: int = 30, seed: int = 0,
                     max_attempts: int = 200_000) -> List[dict]:
    """Winning starts with DTZ spread over per-value buckets (graduated
    difficulty); each start's line must contain >= 1 promotion when the
    shape has a pawn, until the promotion share of the quota is met."""
    rng = random.Random(f"tbstart-{seed}-{shape}")
    pieces = _START_SHAPES[shape]
    has_pawn = any(pt == chess.PAWN for pt, _ in pieces)
    buckets: Dict[int, List[dict]] = {d: [] for d in range(dtz_lo, dtz_hi + 1)}
    promo_lines = 0
    promo_target = n_lines // 4 if has_pawn else 0
    attempts = 0
    while (sum(len(v) for v in buckets.values()) < n_lines
           and attempts < max_attempts):
        attempts += 1
        board = _random_placement(rng, pieces)
        start = probe_start(board, adapter)
        if start is None:
            continue
        fen, dtz = start
        item = {"fen": fen}
        if has_pawn and promo_lines < promo_target:
            promo_move = _line_has_promotion(fen, adapter)
            if promo_move is None:
                continue
            promo_lines += 1
            item["promotion_line"] = True
        buckets[dtz].append(item)
    starts: List[dict] = []
    for d in sorted(buckets):
        for item in buckets[d]:
            starts.append(item)
            if len(starts) >= n_lines:
                break
        if len(starts) >= n_lines:
            break
    return starts


def _line_has_promotion(start_fen: str, adapter) -> bool:
    for row in tb_line_rows(start_fen, adapter, max_plies=40):
        if len(row["target"]) == 5:
            return True
    return False


def build_tb_rows(tb_dir: str, out: str, *, per_type: int = 15000,
                  dtz_lo: int = 1, dtz_hi: int = 30, seed: int = 0,
                  shapes: Tuple[str, ...] = TB_TYPES, progress=None) -> int:
    """Sample starts per shape, play TB-optimal lines, emit winning-side rows."""
    if not tb_dir:
        return 0
    try:
        adapter = SyzygyAdapter(tb_dir)
    except (OSError, ValueError):
        print(f"⚠️ tablebase dir {tb_dir} unusable - skipping Phase D")
        return 0
    rows: List[dict] = []
    for shape in shapes:
        starts = sample_tb_starts(adapter, shape, per_type, dtz_lo=dtz_lo,
                                  dtz_hi=dtz_hi, seed=seed)
        if progress is not None:
            progress.show(f"tb {shape}: {len(starts)} starts")
        promoted = 0
        for item in starts:
            line_rows = tb_line_rows(item["fen"], adapter)
            if any(len(r["target"]) == 5 for r in line_rows):
                promoted += 1
            rows.extend(line_rows)
        print(f"✅ {shape}: {len(starts)} lines, {promoted} with promotions")
    adapter.close()
    from ft.common.serialize import jsonl_write
    jsonl_write(out, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase D - Syzygy tablebase final lines")
    parser.add_argument("--tb-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-type", type=int, default=15_000)
    parser.add_argument("--dtz-lo", type=int, default=1)
    parser.add_argument("--dtz-hi", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shapes", nargs="+", default=list(TB_TYPES))
    args = parser.parse_args()
    n = build_tb_rows(args.tb_dir, args.out, per_type=args.per_type,
                      dtz_lo=args.dtz_lo, dtz_hi=args.dtz_hi, seed=args.seed,
                      shapes=tuple(args.shapes))
    print(f"✅ {n} tb rows -> {args.out}")


if __name__ == "__main__":
    main()
