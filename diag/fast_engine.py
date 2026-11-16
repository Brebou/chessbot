"""Optimized drop-in replacement for the chess_hw Engine (diag-only use).

Same interface and level semantics as ``chess_hw.engine.Engine`` (level 0 =
seeded random, levels 1-4 = alpha-beta + quiescence with material + PST
evaluation), but substantially faster:

- incremental evaluation: the material+PST score is maintained as a delta on
  push/pop instead of re-scanning all pieces at every leaf;
- transposition table (probed only when the stored depth is sufficient, with
  mate scores made ply-relative), skipped when the halfmove clock is high so
  50-move-rule states are never confused;
- cheap pre-filters before the expensive draw checks.

The search is deterministic. Because the transposition table can shortcut a
re-search, the chosen move can very rarely differ from ``chess_hw.Engine`` in
equivalent-score positions; the engine is only used as an opponent during
corpus generation, so this is acceptable.

Usage:
    from diag.fast_engine import FastEngine
    engine = FastEngine(level=3, seed="engine-0-3-12-white")
    move = engine.best_move(board)

    # multiprocessing worker entry point (see gen_corpus.py):
    from diag.fast_engine import engine_best_move_task
    uci = engine_best_move_task((fen, level, seed, qdepth))
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Optional, Tuple

import chess

from chess_hw.engine import MATE, PST, VAL

_KING = chess.KING
_ROOK = chess.ROOK
_PAWN = chess.PAWN

_SQ_NAME = [chess.FILE_NAMES[s % 8] + chess.RANK_NAMES[s // 8] for s in range(64)]
_PROMO_CHAR = {chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q"}

# Castling rook relocations, indexed by the king's destination square.
_CASTLE_ROOK = {
    chess.G1: (chess.H1, chess.F1),
    chess.C1: (chess.A1, chess.D1),
    chess.G8: (chess.H8, chess.F8),
    chess.C8: (chess.A8, chess.D8),
}


def evaluate_white_pov(board: chess.Board) -> int:
    """Material + PST score from White's point of view."""
    score = 0
    pieces_mask = board.pieces_mask
    for pt in range(_PAWN, _KING + 1):
        tbl = PST[chess.WHITE][pt]
        bb = pieces_mask(pt, chess.WHITE)
        while bb:
            score += tbl[(bb & -bb).bit_length() - 1]
            bb &= bb - 1
        tbl = PST[chess.BLACK][pt]
        bb = pieces_mask(pt, chess.BLACK)
        while bb:
            score -= tbl[(bb & -bb).bit_length() - 1]
            bb &= bb - 1
    return score


def ordered_moves(board: chess.Board, captures_only: bool = False) -> list:
    """Legal moves, captures (MVV-LVA) before quiet moves, both uci-sorted.

    Same ordering as ``chess_hw.engine._ordered`` so move choice matches the
    reference engine, but with inlined capture detection and precomputed
    square names (``Move.uci()`` and ``Board.is_capture()`` are surprisingly
    expensive at hundreds of thousands of calls per search).
    """
    caps, quiet = [], []
    if captures_only:
        gen = board.generate_legal_captures()
    else:
        gen = board.legal_moves
        occ_them = board.occupied_co[not board.turn]
        ep = board.ep_square
        pawns = board.pawns
    piece_type_at = board.piece_type_at
    sq_name = _SQ_NAME
    for m in gen:
        to = m.to_square
        promo = m.promotion
        if captures_only:
            is_cap = True
        else:
            is_cap = ((occ_them >> to) & 1) != 0 or (
                to == ep and (pawns >> m.from_square) & 1 != 0
            )
        if is_cap or promo:
            if is_cap:
                v = piece_type_at(to)
                victim = VAL[v] if v else VAL[_PAWN]
            else:
                victim = 0
            score = 10 * victim - VAL[piece_type_at(m.from_square)] // 10
            if promo:
                score += VAL[promo]
            caps.append((-score, sq_name[m.from_square] + sq_name[to] +
                         (_PROMO_CHAR[promo] if promo else ""), m))
        elif not captures_only:
            quiet.append((sq_name[m.from_square] + sq_name[to], m))
    caps.sort(key=lambda t: (t[0], t[1]))
    quiet.sort(key=lambda t: t[0])
    return [t[2] for t in caps] + [t[1] for t in quiet]


class FastEngine:
    """Fast deterministic opponent engine, see module docstring."""

    def __init__(self, level: int = 2, seed: int = 0, qdepth: int = 4,
                 tt_capacity: int = 1 << 21):
        if not 0 <= level <= 4:
            raise ValueError("level must be in 0..4")
        self.level = level
        self.qdepth = qdepth
        self.rng = random.Random(seed)
        self.nodes = 0
        self._tt: dict = {}
        self._tt_capacity = tt_capacity

    # -- evaluation -------------------------------------------------------

    @staticmethod
    def _delta(board: chess.Board, move: chess.Move) -> int:
        """White-POV PST score delta of playing `move` on `board` (pre-push)."""
        frm, to = move.from_square, move.to_square
        color = board.turn
        pt = board.piece_type_at(frm)
        tables = PST[color]
        promo = move.promotion
        d = tables[promo if promo else pt][to] - tables[pt][frm]
        captured_pt = board.piece_type_at(to)
        if captured_pt is not None:
            d += PST[not color][captured_pt][to]
        elif pt == _PAWN and to == board.ep_square:
            d += PST[not color][_PAWN][to - 8 if color == chess.WHITE else to + 8]
        if pt == _KING and abs(to - frm) == 2:
            rf, rt = _CASTLE_ROOK[to]
            d += tables[_ROOK][rt] - tables[_ROOK][rf]
        return d if color == chess.WHITE else -d

    # -- search -----------------------------------------------------------

    def _insufficient(self, board: chess.Board) -> bool:
        if board.pawns or board.rooks or board.queens:
            return False
        return board.is_insufficient_material()

    def _probe_tt(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int):
        entry = self._tt.get(board._transposition_key())
        if entry is None:
            return None, None
        e_depth, e_flag, e_score, e_move = entry
        if e_depth < depth:
            return None, e_move
        s = e_score
        if s > MATE - 1000:
            s -= ply
        elif s < -MATE + 1000:
            s += ply
        if e_flag == 0:
            return s, e_move
        if e_flag == 1 and s <= alpha:
            return alpha, e_move
        if e_flag == 2 and s >= beta:
            return beta, e_move
        return None, e_move

    def _store_tt(self, board: chess.Board, depth: int, flag: int,
                  score: int, ply: int, move: Optional[chess.Move]) -> None:
        if board.halfmove_clock >= 90:
            return
        if score > MATE - 1000:
            score += ply
        elif score < -MATE + 1000:
            score -= ply
        tt = self._tt
        if len(tt) >= self._tt_capacity:
            tt.clear()
        tt[board._transposition_key()] = (depth, flag, score, move)

    def _quiesce(self, board: chess.Board, alpha: int, beta: int, qd: int, score: int) -> int:
        self.nodes += 1
        stand = score if board.turn == chess.WHITE else -score
        if stand >= beta:
            return beta
        if alpha < stand:
            alpha = stand
        if qd == 0:
            return alpha
        for m in ordered_moves(board, captures_only=True):
            d = self._delta(board, m)
            board.push(m)
            s = -self._quiesce(board, -beta, -alpha, qd - 1, score + d)
            board.pop()
            if s >= beta:
                return beta
            if s > alpha:
                alpha = s
        return alpha

    def _search(self, board: chess.Board, depth: int, alpha: int, beta: int,
                ply: int, score: int) -> int:
        self.nodes += 1
        if board.halfmove_clock < 90:
            bound, tt_move = self._probe_tt(board, depth, alpha, beta, ply)
            if bound is not None:
                return bound
        else:
            tt_move = None
        moves = ordered_moves(board)
        if not moves:
            return -MATE + ply if board.is_check() else 0
        if board.halfmove_clock >= 100 or self._insufficient(board):
            return 0
        if depth == 0:
            return self._quiesce(board, alpha, beta, self.qdepth, score)

        if tt_move is not None:
            moves.sort(key=lambda m: m != tt_move)

        alpha_orig = alpha
        best_score = -MATE - 1
        best_move = None
        for m in moves:
            d = self._delta(board, m)
            board.push(m)
            s = -self._search(board, depth - 1, -beta, -alpha, ply + 1, score + d)
            board.pop()
            if s > best_score:
                best_score, best_move = s, m
            if s >= beta:
                self._store_tt(board, depth, 2, beta, ply, m)
                return beta
            if s > alpha:
                alpha = s
        self._store_tt(board, depth, 0 if best_score > alpha_orig else 1,
                       best_score, ply, best_move)
        return alpha

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = ordered_moves(board)
        if not moves:
            return None
        if self.level == 0:
            return self.rng.choice(sorted(moves, key=lambda m: m.uci()))
        score = evaluate_white_pov(board)
        best, best_score = moves[0], -MATE - 1
        alpha, beta = -MATE - 1, MATE + 1
        for m in moves:
            d = self._delta(board, m)
            board.push(m)
            s = -self._search(board, self.level - 1, -beta, -alpha, 1, score + d)
            board.pop()
            if s > best_score:
                best, best_score = m, s
            if s > alpha:
                alpha = s
        return best


# ---------------------------------------------------------------------------
# Multiprocessing worker (used by gen_corpus.py to parallelize engine moves)
# ---------------------------------------------------------------------------

_ENGINE_CACHE: "OrderedDict[Tuple, FastEngine]" = OrderedDict()
_ENGINE_CACHE_MAX = 512


def engine_best_move_task(args: Tuple[str, int, str]) -> Optional[str]:
    """Worker task: (fen, level, seed) -> best move as UCI (or None)."""
    fen, level, seed = args
    key = (level, seed)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        engine = FastEngine(level, seed)
        _ENGINE_CACHE[key] = engine
        if len(_ENGINE_CACHE) > _ENGINE_CACHE_MAX:
            _ENGINE_CACHE.popitem(last=False)
    move = engine.best_move(chess.Board(fen))
    return move.uci() if move else None
