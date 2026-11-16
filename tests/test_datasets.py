"""Tests for the training-dataset annotated-notation parser (diag/common.py)."""

import gzip
import json
import os
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diag.common import parse_annotated_game

TRAIN_FILE = Path(os.environ.get("CHESSBOT_DATA_DIR", "data")) / "lichess_train_2025-01_1M.jsonl.gz"

# Real game samples from dlouapre/lichess_2025-01_1M (shard 0)
CASTLE_GAME = ("WPd2d4 BNg8f6 WNg1f3 BPd7d6 WPh2h3 BPg7g6 WPc2c4 BBf8g7 WNb1c3 BNb8d7 "
               "WPe2e4 BPe7e5 WPd4e5(x) BPd6e5(x) WBc1g5 BPc7c6 WBf1d3 BKe8g8(o) "
               "WKe1g1(o) BQd8c7 WBd3c2 BPa7a5 WQd1e2 BNd7c5 WRa1d1 BBc8e6 WQe2e3 "
               "BRf8d8 WQe3c5(x) BRd8d7")
PROMO_GAME = ("WPe2e4 BPe7e5 WNg1f3 BNg8f6 WPd2d4 BPd7d5 WPe4d5(x) BQd8d5(x) WPd4e5(x) "
              "BQd5e4(+) WQd1e2 BBf8b4(+) WNb1d2 BQe4c2(x) WPe5f6(x+) BKe8d8 WPf6g7(x) "
              "BRh8e8 WPg7g8(Q) BRe8g8(x) WQe2c4 BBb4d2(x+) WBc1d2(x) BQc2c4(x)")
EP_GAME = ("WNg1f3 BPc7c5 WPg2g3 BNb8c6 WBf1g2 BPg7g6 WKe1g1(o) BBf8g7 WPc2c3 BPe7e6 "
           "WPd2d4 BNg8e7 WPd4c5(x) BPd7d5 WPc5d6(xE) BQd8d6(x) WQd1d6(x)")
SHORT_GAME = "WPe2e4 BPd7d5 WNg1f3 BPd5e4(x) WNf3g5 BPf7f5 WNb1c3 BPe7e6 WBf1c4 BQd8g5(x) WPd2d4"
MATE_GAME = ("WPd2d4 BNg8f6 WNb1c3 BPg7g6 WBc1f4 BBf8g7 WPe2e4 BPd7d6 WPe4e5 BPd6e5(x) "
             "WPd4e5(x) BQd8d1(x+) WRa1d1(x) BNf6h5 WNc3d5 BNb8a6 WBf1a6(x) BPb7a6(x) "
             "WNd5c7(x+) BKe8f8 WRd1d8(+*)")


def test_parse_simple_moves():
    assert parse_annotated_game("WPe2e4 BPe7e5") == ["e2e4", "e7e5"]


def test_parse_strips_annotations():
    assert parse_annotated_game("WQh5e5(x+) BKe8f7(*)") == ["h5e5", "e8f7"]


def test_parse_promotion_annotation():
    assert parse_annotated_game("WPg7g8(Q)") == ["g7g8q"]


def test_parse_underpromotion_annotations():
    assert parse_annotated_game("WPb7b8(R) WPg7g8(N) BPf2f1(B)") == \
        ["b7b8r", "g7g8n", "f2f1b"]


def test_parse_en_passant_annotation():
    assert parse_annotated_game("WPe5d6(xE)") == ["e5d6"]


def test_parse_castling_annotations():
    assert parse_annotated_game("WKe1g1(o) BKe8c8(O)") == ["e1g1", "e8c8"]


def test_parse_mixed_annotations():
    assert parse_annotated_game("WQf7f8(x+*) BPe7f6(xQ)") == ["f7f8", "e7f6q"]


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_annotated_game("WXe2e4")
    with pytest.raises(ValueError):
        parse_annotated_game("e2e4")
    with pytest.raises(ValueError):
        parse_annotated_game("WPe2")


def _replay_with_checks(text):
    """Replay an annotated game, asserting every annotation matches the board."""
    board = chess.Board()
    for tok in text.split():
        color, piece, squares, ann = tok[0], tok[1], tok[2:6], tok[6:]
        ann = ann[1:-1] if ann.startswith("(") else ""
        move = chess.Move.from_uci(squares + "".join(
            c.lower() for c in ann if c in "qrbnQRBN"))

        assert board.turn == (chess.WHITE if color == "W" else chess.BLACK), tok
        assert board.piece_type_at(chess.parse_square(squares[:2])) == \
            chess.PIECE_SYMBOLS.index(piece.lower()), tok
        assert move in board.legal_moves, tok

        if "x" in ann:
            assert board.is_capture(move), tok
        else:
            assert not board.is_capture(move) or board.is_en_passant(move), tok
        if "E" in ann:
            assert board.is_en_passant(move), tok
        if "o" in ann or "O" in ann:
            assert board.is_castling(move), tok
            assert ("o" in ann) == (move.to_square in (chess.G1, chess.G8)), tok
        if any(c in ann for c in "qrbnQRBN"):
            assert move.promotion is not None, tok
        if "*" in ann:
            assert board.gives_check(move), tok

        board.push(move)
        if "+" in ann:
            assert board.is_check(), tok
        if "*" in ann:
            assert board.is_checkmate(), tok
    return board


def test_replay_castle_game():
    _replay_with_checks(CASTLE_GAME)


def test_replay_promo_game():
    _replay_with_checks(PROMO_GAME)


def test_replay_ep_game():
    _replay_with_checks(EP_GAME)


def test_replay_short_game():
    _replay_with_checks(SHORT_GAME)


def test_replay_mate_ending():
    _replay_with_checks(MATE_GAME)

    board = _replay_with_checks(MATE_GAME)
    assert board.is_game_over()


@pytest.mark.skipif(not TRAIN_FILE.exists(), reason="training dataset not converted yet")
def test_training_dataset_sample():
    with gzip.open(TRAIN_FILE, "rt") as f:
        lines = [json.loads(next(f)) for _ in range(100)]
    assert len(lines) == 100
    for row in lines:
        moves = row["moves"].split()
        assert len(moves) > 0
        board = chess.Board()
        for uci in moves:
            assert chess.Move.from_uci(uci) in board.legal_moves, uci
            board.push_uci(uci)
