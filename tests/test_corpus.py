"""Corpus validation tests.

Validates a random sample of positions from each generated corpus:
legality (via replay()), label correctness, material balance, and absence
of exact duplicates.

The corpus directory defaults to ``results/corpus`` and can be overridden
with the ``CHESSBOT_CORPUS_DIR`` environment variable. Tests are skipped
when a corpus file is missing (e.g. Lichess-derived corpora without a
dataset).
"""

import os
import random
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diag.common import material_diff, material_value, replay
from diag.see import see

CORPUS_DIR = Path(os.environ.get("CHESSBOT_CORPUS_DIR", "results/corpus"))
SAMPLE_SIZE = 100
SEED = 0


def _load(name: str):
    path = CORPUS_DIR / name
    if not path.exists():
        pytest.skip(f"Corpus {path} not found")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(__import__("json").loads(line))
    return rows


def _sample(rows, size=SAMPLE_SIZE):
    return random.Random(SEED).sample(rows, min(size, len(rows)))


def _assert_no_duplicates(rows, name):
    seen = set()
    for row in rows:
        key = (tuple(row.get("history", row.get("moves", []))), row.get("fen"))
        assert key not in seen, f"Duplicate row in {name}: {key}"
        seen.add(key)


def test_corpus_onpolicy():
    rows = _load("onpolicy_positions.jsonl")
    _assert_no_duplicates(rows, "onpolicy")
    for pos in _sample(rows):
        board = replay(pos["history"])
        assert board.fen() == pos["fen"], f"FEN mismatch at ply {pos['ply']}"
        assert board.ply() == pos["ply"]
        if pos.get("chosen_move"):
            move = chess.Move.from_uci(pos["chosen_move"])
            assert move in board.legal_moves, f"Illegal chosen move {pos['chosen_move']}"
        # Material sanity: both kings present and a finite, integer diff.
        # (No upper bound: a weak model can drop almost everything, and
        # promotions make the diff exceed the classical 39-pawn maximum.)
        assert board.king(chess.WHITE) is not None
        assert board.king(chess.BLACK) is not None
        diff = material_diff(board, chess.WHITE)
        assert diff == diff and abs(diff) != float("inf")


def test_corpus_mate1():
    rows = _load("probe_mate1.jsonl")
    _assert_no_duplicates(rows, "mate1")
    for pos in _sample(rows):
        history = pos.get("history") or pos.get("moves") or []
        board = replay(history)
        assert board.fen() == pos["fen"]
        assert pos["mates"], "Empty mates list"
        for uci in pos["mates"]:
            move = board.parse_uci(uci)
            board.push(move)
            assert board.is_checkmate(), f"{uci} is not mate in {pos['fen']}"
            board.pop()


def test_corpus_captures():
    rows = _load("probe_captures.jsonl")
    _assert_no_duplicates(rows, "captures")
    for pos in _sample(rows):
        board = replay(pos["history"])
        assert board.fen() == pos["fen"]
        move = board.parse_uci(pos["capture_move"])
        assert board.is_capture(move), f"{pos['capture_move']} is not a capture"
        assert isinstance(pos["see"], int)
        expected = see(board, move)
        assert pos["see"] == expected, (
            f"SEE mismatch: corpus={pos['see']} recomputed={expected}")


def test_corpus_endgames():
    rows = _load("probe_endgames.jsonl")
    _assert_no_duplicates(rows, "endgames")
    for pos in _sample(rows):
        assert pos["type"] in ("kqk", "krk")
        board = chess.Board(pos["fen"])
        n_pieces = len(board.piece_map())
        assert n_pieces == 3, f"Endgame {pos['type']} should have 3 pieces, got {n_pieces}"
        if pos.get("history"):
            replayed = replay(pos["history"])
            assert replayed.fen() == pos["fen"], (
                "History does not reach the recorded FEN (only checked when "
                "a history is present; generated endgames may have none)")


def test_corpus_conversion():
    rows = _load("probe_conversion.jsonl")
    if not rows:
        pytest.skip("Conversion corpus is empty (quotas not reached)")
    _assert_no_duplicates(rows, "conversion")
    for pos in _sample(rows):
        assert pos["threshold"] in (2, 3, 5, 9)
        board = replay(pos["history"])
        assert board.fen() == pos["fen"]
        color = chess.WHITE if pos["model_color"] == "white" else chess.BLACK
        diff = material_diff(board, color)
        assert diff >= pos["threshold"], (
            f"Material advantage {diff} below threshold {pos['threshold']}")
        # Model must be to move in the recorded position
        assert (board.turn == chess.WHITE) == (pos["model_color"] == "white")


def test_corpus_promotion():
    rows = _load("probe_promotion.jsonl")
    if not rows:
        pytest.skip("Promotion corpus is empty")
    _assert_no_duplicates(rows, "promotion")
    for pos in _sample(rows):
        board = replay(pos["history"])
        assert board.fen() == pos["fen"]
        legal_promos = {m.uci() for m in board.legal_moves if m.promotion}
        for uci in pos.get("safe_promotions", []):
            assert uci in legal_promos, f"Safe promotion {uci} not legal"
            assert uci[-1] in "qrbn"
        for uci in pos.get("capture_promotions", []):
            assert uci in legal_promos, f"Capture promotion {uci} not legal"
            assert board.is_capture(chess.Move.from_uci(uci))
        # A promotion must actually be available
        assert pos.get("safe_promotions") or pos.get("capture_promotions"), (
            "Promotion record without any promotion")
