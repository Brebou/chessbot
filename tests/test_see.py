"""Manual SEE test cases with known, hand-verified values (centipawns).

Each case was validated against the classical swap-algorithm definition:
the gain sequence of captures on the target square is resolved backwards
with gain[d-1] = -max(-gain[d-1], gain[d]).
"""

import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diag.see import see


CASES = [
    # (FEN, UCI, expected SEE, label)
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 100,
     "pawn takes undefended pawn"),
    ("4k3/8/1p6/2p5/3P4/8/8/4K3 w - - 0 1", "d4c5", 0,
     "pawn takes pawn defended by pawn (equal trade)"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "f3e5", -220,
     "knight takes pawn, recaptured by knight (handoff claimed +220: wrong)"),
    ("7k/8/8/8/6p1/3Q3r/8/6K1 w - - 0 1", "d3h3", -400,
     "queen takes rook defended by pawn"),
    ("4k3/8/8/8/8/8/4r2R/4K3 w - - 0 1", "h2e2", 500,
     "rook takes free rook"),
    ("4k3/8/8/8/8/8/3n4/2B1K3 w - - 0 1", "c1d2", 320,
     "bishop takes free knight"),
    ("4k3/8/8/8/8/1n6/3n4/2B4K w - - 0 1", "c1d2", -10,
     "bishop takes knight defended by knight (320 vs 330)"),
    ("4k3/8/8/8/8/1n6/3n4/2B1K3 w - - 0 1", "c1d2", 310,
     "same but white king recaptures: 320 - 330 + 320"),
    ("4k3/8/8/8/4pP2/8/8/4K3 b - f3 0 1", "e4f3", 100,
     "en passant capture"),
    ("4k3/P7/8/8/8/8/8/6K1 w - - 0 1", "a7a8q", 800,
     "quiet promotion (900 - 100 pawn)"),
    ("4k3/P7/8/8/8/8/r7/6K1 w - - 0 1", "a7a8q", -100,
     "quiet promotion recaptured by rook (+900 -100 -900)"),
    ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", 1300,
     "capture-promotion: rook + queen - pawn"),
    ("5k2/8/5r2/8/8/8/1B6/6K1 w - - 0 1", "b2f6", 500,
     "x-ray behind attacker: king cannot recapture (b2-f6 diagonal opens)"),
    ("8/5k2/5r2/8/6N1/8/8/6K1 w - - 0 1", "g4f6", 180,
     "knight takes rook, king recaptures: 500 - 320"),
    ("4k3/8/2n5/4p3/8/2Q5/8/4R1K1 w - - 0 1", "c3e5", -480,
     "cascading captures QxP NxQ RxN: 100 - 900 + 320"),
    ("2r1k3/8/8/2p5/3P4/8/8/4K3 w - - 0 1", "d4c5", 0,
     "pawn takes pawn recaptured by rook"),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1e2", 0,
     "quiet non-capture move"),
]


@pytest.mark.parametrize("fen,uci,expected,label", CASES, ids=[c[3] for c in CASES])
def test_see_known_values(fen, uci, expected, label):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} illegal in {fen}"
    assert see(board, move) == expected, f"{label}: {uci} in {fen}"


def test_see_non_capture_is_zero():
    board = chess.Board()
    assert see(board, chess.Move.from_uci("e2e4")) == 0


def test_see_symmetric_exchange_sequence():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert see(board, chess.Move.from_uci("a1a8")) == 500


def test_see_attacking_color_override_matches_turn():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    board.push_uci("e1e2")
    move = chess.Move.from_uci("d5e4")
    assert see(board, move, attacking_color=chess.BLACK) == see(board, move) == 100
