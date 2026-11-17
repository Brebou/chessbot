"""Bucket criteria primitives (self-contained; unit-tested individually).

Best-move tranquility, mate-in-2 verification, and the SEE-based blunder
tests used by the contrastive bucket. SEE helpers reuse diag.see.
"""

from __future__ import annotations

from typing import Optional

import chess

from diag.see import see

HANG_SEE_CP = 100
LOSING_CAPTURE_SEE_CP = -50


def best_move_is_quiet(board: chess.Board, best_uci: str) -> bool:
    mv = board.parse_uci(best_uci)
    return (not board.is_capture(mv) and not board.gives_check(mv)
            and mv.promotion is None)


def _mate_in_1(board: chess.Board) -> bool:
    for mv in board.legal_moves:
        if not board.gives_check(mv):
            continue
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def find_mate_in_2(board: chess.Board) -> Optional[str]:
    """A move after which EVERY opponent reply allows a mate in 1, or None.
    Checks-first candidate order with early refutation."""
    if board.is_game_over():
        return None
    checks, quiet = [], []
    for mv in board.legal_moves:
        (checks if board.gives_check(mv) else quiet).append(mv)
    for mv in checks + quiet:
        board.push(mv)
        if board.is_game_over():
            board.pop()
            continue
        refuted = False
        for reply in board.legal_moves:
            board.push(reply)
            mate = _mate_in_1(board)
            board.pop()
            if not mate:
                refuted = True
                break
        board.pop()
        if not refuted:
            return mv.uci()
    return None


def _opponent_best_capture_see(board: chess.Board) -> int:
    best = 0
    for mv in board.legal_moves:
        if board.is_capture(mv):
            s = -see(board, mv)
            best = max(best, s)
    return best


def is_hanging_after(fen: str, uci_played: str) -> bool:
    """After the played move, can the opponent win >= HANG_SEE_CP by capture?"""
    board = chess.Board(fen)
    board.push_uci(uci_played)
    return _opponent_best_capture_see(board) >= HANG_SEE_CP


def is_losing_capture(fen: str, uci_played: str) -> Optional[int]:
    """SEE of the played move if it is a capture, else None."""
    board = chess.Board(fen)
    mv = board.parse_uci(uci_played)
    if not board.is_capture(mv):
        return None
    return see(board, mv)
