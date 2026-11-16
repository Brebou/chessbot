"""Static Exchange Evaluation (SEE) for chess positions.

Computes the net material value of the sequence of captures on ``move.to_square``
using the classical swap algorithm: captures are played one by one (least
valuable attacker first, kings only when the target square is not defended
after their capture), then the gain sequence is resolved backwards with the
minimax rule ``gain[d-1] = -max(-gain[d-1], gain[d])``.

X-rays are handled naturally because attackers are recomputed on the updated
board after each capture (a sliding piece recapturing through the square the
capturing piece just left is seen).

Usage:
    from diag.see import see
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    value = see(board, move)  # in centipawns, from the mover's perspective
"""

from __future__ import annotations

from typing import Optional

import chess

# Piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

_MAX_EXCHANGE = 32


def see(board: chess.Board, move: chess.Move, attacking_color: Optional[chess.Color] = None) -> int:
    """Calculate Static Exchange Evaluation for a move.

    Args:
        board: Current board position (``move`` must be legal for ``board.turn``)
        move: The move to evaluate
        attacking_color: Override the moving color (for testing)

    Returns:
        Net material value in centipawns after all recaptures on the target
        square, from the perspective of the side making the move.
        Returns 0 for quiet non-promotion moves.
    """
    if not board.is_capture(move) and not move.promotion:
        return 0

    mover = board.turn if attacking_color is None else attacking_color
    to = move.to_square

    # Value captured by the initial move.
    if board.is_en_passant(move):
        captured_value = PIECE_VALUES[chess.PAWN]
        captured_sq = to - 8 if board.turn == chess.WHITE else to + 8
    elif board.is_capture(move):
        victim = board.piece_type_at(to)
        captured_value = PIECE_VALUES[victim]
        captured_sq = to
    else:
        captured_value = 0
        captured_sq = to

    moving_pt = board.piece_type_at(move.from_square)
    if move.promotion:
        promo_bonus = PIECE_VALUES[move.promotion] - PIECE_VALUES[moving_pt]
    else:
        promo_bonus = 0

    gain = [0] * _MAX_EXCHANGE
    gain[0] = captured_value + promo_bonus

    # Play the initial move on a throwaway board.
    b = board.copy(stack=False)
    b.remove_piece_at(move.from_square)
    if captured_sq != to:
        b.remove_piece_at(captured_sq)
    b.set_piece_at(to, chess.Piece(move.promotion or moving_pt, mover))

    side = not mover
    d = 0
    while d + 1 < _MAX_EXCHANGE:
        attackers = b.attackers(side, to)
        if not attackers:
            break

        # Least valuable attacker first; a king may only capture if the
        # square is not defended after its capture (otherwise the capture
        # would be illegal).
        candidates = sorted(attackers, key=lambda s: PIECE_VALUES[b.piece_type_at(s)])
        chosen = None
        for s in candidates:
            if b.piece_type_at(s) == chess.KING:
                probe = b.copy(stack=False)
                probe.remove_piece_at(s)
                probe.set_piece_at(to, chess.Piece(chess.KING, side))
                if probe.attackers(not side, to):
                    continue
            chosen = s
            break
        if chosen is None:
            break

        d += 1
        gain[d] = PIECE_VALUES[b.piece_type_at(to)] - gain[d - 1]
        pt = b.piece_type_at(chosen)
        b.remove_piece_at(chosen)
        b.set_piece_at(to, chess.Piece(pt, side))
        side = not side

    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1

    return gain[0]
