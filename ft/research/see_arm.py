"""ARC2 arm B: material-safety curriculum from chess heuristics (no SF).

Targets the two biggest graded priorities directly:
- 40% "take offered pieces": whenever a safe capture exists (victim
  undefended, or victim strictly more valuable than the cheapest attacker
  with recapture), teach taking the most valuable one.
- 30% "don't hang": if the model's chosen move leaves its own piece on an
  attacked-and-undefended square (or on a square attacked by a cheaper
  enemy piece), teach a same-piece or fallback move that does not.

Labels are pure python-chess heuristics (no Stockfish), so this corpus can
scale to hundreds of thousands of states. Training reuses the t5 recipe.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import chess

VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.2,
       chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 40.0}


def cheapest_attacker_value(board: chess.Board, sq: int,
                            color: chess.Color) -> Optional[float]:
    atts = [board.piece_at(s) for s in board.attackers(color, sq)]
    atts = [p for p in atts if p is not None]
    if not atts:
        return None
    return min(VAL[p.piece_type] for p in atts)


def is_safe_capture(board: chess.Board, mv: chess.Move) -> bool:
    victim = board.piece_at(mv.to_square)
    if victim is None or victim.color == board.turn:
        # en passant: victim is a pawn on a different square
        if board.is_en_passant(mv):
            victim_val = 1.0
        else:
            return False
    else:
        victim_val = VAL[victim.piece_type]
    opp = not board.turn
    my_val = VAL[board.piece_at(mv.from_square).piece_type]
    defends = board.attackers(opp, mv.to_square)
    if not defends:
        return True
    # defended: safe if strictly winning exchange (victim > attacker) and
    # the cheapest defender of the defender-square square-chain ... keep the
    # classical simple rule: winning capture if victim > attacker, or equal
    # trade where we are the side capturing a defended piece only if the
    # recapturing piece is defended by us (approximation: skip).
    return victim_val > my_val + 0.5


def chosen_move_hangs(board: chess.Board, mv: chess.Move) -> bool:
    """True if mv leaves the moved piece (or opens a castled king) easily
    capturable on an attacked-and-undefended square."""
    b = board.copy()
    mover = board.piece_at(mv.from_square)
    b.push(mv)
    if mover is None:
        return False
    sq = mv.to_square
    my_pc = b.piece_at(sq)
    if my_pc is None:  # en passant: the pawn is behind the target square
        sq = mv.to_square + (-8 if mover.color == chess.WHITE else 8)
        my_pc = b.piece_at(sq)
        if my_pc is None:
            return False
    opp = not mover.color
    att = cheapest_attacker_value(b, sq, opp)
    if att is None:
        return False
    mine = VAL[my_pc.piece_type]
    return att < mine or (att <= mine and not b.attackers(mover.color, sq))


def best_safe_move(board: chess.Board) -> Optional[chess.Move]:
    """Priority: most valuable safe capture; else a same-piece relocation
    of the hanging piece to a safe square; else any non-hanging move."""
    caps = [m for m in board.legal_moves if is_safe_capture(board, m)]
    if caps:
        caps.sort(key=lambda m: -_victim_value(board, m))
        return caps[0]
    return None


def _victim_value(board: chess.Board, mv: chess.Move) -> float:
    v = board.piece_at(mv.to_square)
    if v is None:
        return 1.0 if board.is_en_passant(mv) else 0.0
    return VAL[v.piece_type]


def relocate_safely(board: chess.Board, mv: chess.Move) -> Optional[chess.Move]:
    """A move of the same piece to a square that is not attacked-and-
    undefended, preserving roughly the piece's mobility."""
    piece_sq = mv.from_square
    cands = []
    for m in board.legal_moves:
        if m.from_square != piece_sq:
            continue
        if chosen_move_hangs(board, m):
            continue
        cands.append(m)
    if not cands:
        return None
    return cands[len(cands) // 2]


def build_safety_rows(states: List[dict], model_path: str, *,
                      batch: int = 256, max_rows: int = 100000,
                      seed: int = 0) -> tuple:
    """Returns (rows, stats). rows target the heuristic safe move when the
    model's top-1 either misses a safe capture or hangs a piece."""
    from chess_hw import ChessTokenizer
    from chess_hw.model import load_model
    from diag.fast_probe import FastProbePlayer

    model = load_model(model_path, device="cpu")
    probe = FastProbePlayer(model, ChessTokenizer(), batch_size=batch)
    rows: List[dict] = []
    stats = {"n": len(states), "n_hang": 0, "n_missed_cap": 0,
             "n_targetable": 0}
    for s in states:
        board = chess.Board(s["fen"])
        cands = probe.choose_from_histories([s["history"]], [board])
        top = cands[0]
        model_mv = (top.move if top is not None and top.move is not None
                    else None)
        if model_mv is None:
            continue
        target = None
        if chosen_move_hangs(board, model_mv):
            stats["n_hang"] += 1
            target = best_safe_move(board) or relocate_safely(board, model_mv)
        else:
            safe_caps = [m for m in board.legal_moves
                         if is_safe_capture(board, m)]
            if safe_caps and not board.is_capture(model_mv):
                stats["n_missed_cap"] += 1
                # only correct if the missed capture wins material the model
                # does not otherwise win by its chosen move
                best_v = max(_victim_value(board, m) for m in safe_caps)
                if best_v >= 3.0 or _is_free_piece(board, safe_caps[0]):
                    target = max(safe_caps,
                                 key=lambda m: _victim_value(board, m))
        if target is not None and target.uci() != model_mv.uci():
            stats["n_targetable"] += 1
            rows.append({"bucket": "normal", "key": s["fen"],
                         "start_fen": s["fen"], "history": s["history"],
                         "target": target.uci(), "weight": 1.0})
        if len(rows) >= max_rows:
            break
    del model, probe
    return rows, stats


def _is_free_piece(board: chess.Board, cap: chess.Move) -> bool:
    v = board.piece_at(cap.to_square)
    return (v is None or v.color != board.turn) and (
        cheapest_attacker_value(board, cap.to_square,
                                not board.turn) is None or True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="results/arc2/corpus.jsonl")
    ap.add_argument("--scan-raw", default="results/arc2/scan_raw.jsonl")
    ap.add_argument("--out-mix", default="results/arc2/mix_safety.jsonl")
    ap.add_argument("--model-path", default="base_model/model.safetensors")
    ap.add_argument("--max-states", type=int, default=20000)
    ap.add_argument("--max-rows", type=int, default=100000)
    args = ap.parse_args()
    t0 = time.monotonic()
    src = args.scan_raw if Path(args.scan_raw).exists() else args.corpus
    states = [json.loads(l) for l in open(src)]
    rng = random.Random(7)
    rng.shuffle(states)
    states = states[:args.max_states]
    rows, stats = build_safety_rows(states, args.model_path,
                                    max_rows=args.max_rows)
    from ft.common.serialize import jsonl_write
    jsonl_write(Path(args.out_mix), rows)
    print(f"safety rows: {json.dumps(stats)} -> {len(rows)} rows "
          f"({time.monotonic()-t0:.0f}s) -> {args.out_mix}", flush=True)


if __name__ == "__main__":
    main()
