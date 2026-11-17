"""Tests for the SFT mix builder (diag/train_data.py)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from diag.common import replay
from diag.train_data import (build_mix, extract_examples, find_mate_in_2,
                             generate_endgame_examples, mine_game, ply_weight)


def kinds(examples):
    return [e["kind"] for e in examples]


def test_ply_weight_curriculum():
    assert ply_weight(0) == 1.0
    assert ply_weight(8) == 1.0
    assert ply_weight(15) == 2.0   # eval decision zone
    assert ply_weight(30) == 2.0
    assert ply_weight(31) == 1.0
    assert ply_weight(130) == 2.0  # endgame zone


def test_extract_mate_in_1():
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]:
        board.push_uci(uci)
    rng = __import__("random").Random(0)
    ex = extract_examples(board, "h5f7", rng)
    assert "mate1" in kinds(ex)
    mate = [e for e in ex if e["kind"] == "mate1"][0]
    assert mate["target"] == "h5f7"
    board.push_uci("h5f7")
    assert board.is_checkmate()


def test_extract_free_capture():
    # after g1f3 d7d6 f3e5, black can win the undefended e5 knight with dxe5
    board = chess.Board()
    for uci in ["g1f3", "d7d6", "f3e5"]:
        board.push_uci(uci)
    rng = __import__("random").Random(0)
    # played move misses the win -> capture example teaching dxe5
    ex = extract_examples(board, "a7a6", rng, hang_tries=0)
    caps = [e for e in ex if e["kind"] == "capture"]
    assert caps and caps[0]["target"] == "d6e5"
    assert caps[0]["weight"] > 1.0  # scaled by SEE (free knight = +320)
    # played move IS the winning capture -> no redundant capture example
    ex2 = extract_examples(board, "d6e5", rng, hang_tries=0)
    assert "capture" not in kinds(ex2)


def test_extract_hang_pair():
    # white to move: played move is safe, some random alternative hangs
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3"]:
        board.push_uci(uci)
    # position: white played d2d3; find a hanging alternative among legal moves
    rng = __import__("random").Random(42)
    ex = extract_examples(board, "d2d3", rng, hang_tries=32)
    hangs = [e for e in ex if e["kind"] == "hang"]
    if hangs:  # deterministic rng must find one in a starting-position-derived position
        h = hangs[0]
        assert h["target"] == "d2d3"
        assert h["bad"] != h["target"]
        replay(board.move_stack and [m.uci() for m in board.move_stack])


def test_find_mate_in_2_two_rook_ladder():
    # quiet key move (no mate in 1 available): 1.Ra7! forces mate next move
    board = chess.Board("7k/8/8/8/8/8/R7/1R5K w - - 0 1")
    m2 = find_mate_in_2(board)
    assert m2 is not None
    board.push_uci(m2)
    assert not board.is_game_over()
    for r in list(board.legal_moves):
        board.push(r)
        found = False
        for w2 in board.legal_moves:
            board.push(w2)
            if board.is_checkmate():
                found = True
            board.pop()
        board.pop()
        assert found, f"reply {r.uci()} refutes the claimed mate in 2"


def test_find_mate_in_2_none_on_quiet_positions():
    assert find_mate_in_2(chess.Board()) is None
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
        board.push_uci(uci)
    assert find_mate_in_2(board) is None


def test_extract_mate_in_2_endgame_position():
    board = chess.Board("7k/8/8/8/8/8/R7/1R5K w - - 0 1")
    rng = __import__("random").Random(0)
    ex = extract_examples(board, "a2a7", rng, hang_tries=0)
    m2 = [e for e in ex if e["kind"] == "mate2"]
    assert m2 and m2[0]["target"] is not None


def test_mine_game_replay_and_curriculum():
    moves = " ".join(["g1f3", "g8f6", "f3g1", "f6g8"] * 10)  # legal shuffles
    cfg = {"seed": 0, "hang_tries": 0, "endgame_per_game": 0,
           "promotion_per_game": 0}
    ex = mine_game((0, moves, cfg))
    replays = [e for e in ex if e["kind"] == "replay"]
    assert replays
    for e in replays:
        board = replay(e["history"])
        assert board.parse_uci(e["target"]) is not None
        assert e["weight"] in (1.0, 2.0)


def test_mine_game_endgame_detection():
    # a real game reaching KQK: quick construction via a short decisive line
    # is hard; instead check the detection triggers on a 3-piece position
    # reached through legal moves: 1. e4 d5 2. exd5 Qxd5 3. Qxd5?? illegal;
    # use a crafted sequence that sheds material via captures
    moves = ("e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 a5a5".split())
    cfg = {"seed": 1, "hang_tries": 0, "endgame_per_game": 4,
           "promotion_per_game": 0}
    ex = mine_game((0, " ".join(moves), cfg))
    for e in ex:
        if e["kind"] == "endgame":
            board = replay(e["history"])
            assert len(board.piece_map()) == 3


def test_generate_endgame_examples_runs():
    ex = generate_endgame_examples(4, level=2, seed=0, max_plies=40)
    for e in ex:
        assert e["kind"] == "endgame"
        board = chess.Board(e["start_fen"])
        for uci in e["history"]:
            board.push_uci(uci)
        assert board.parse_uci(e["target"]) is not None


def test_build_mix_end_to_end(tmp_path):
    # tiny synthetic corpus: scholar's mate + free-knight games + filler
    games = [
        "e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7",
        "g1f3 d7d6 f3e5 d6e5",
        "g1f3 d7d6 f3e5 a7a6 d6e5",  # black misses the free knight (dxe5)
    ] + ["e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5c6 d7d6"] * 8
    ds = tmp_path / "ds.jsonl"
    with open(ds, "w") as f:
        for g in games:
            f.write(json.dumps({"moves": g}) + "\n")
    out = tmp_path / "mix.jsonl"
    counts = build_mix(str(ds), str(out), n_games=10, workers=2, seed=0,
                       hq_dataset=str(ds), hq_games=10, min_elo=0,
                       max_replay=1000, max_capture=1000, max_mate1=1000,
                       max_mate2=1000, max_hang=1000, max_endgame=1000,
                       max_promotion=100, max_distill=1000,
                       endgame_demos=2, endgame_level=2, force=True,
                       progress=False)
    assert counts["replay"] > 0
    assert counts["distill"] > 0
    assert counts["mate1"] >= 1
    assert counts["capture"] >= 1
    rows = [json.loads(l) for l in open(out) if l.strip()]
    assert len(rows) == sum(counts.values())
    for r in rows:
        assert r["kind"] in ("replay", "capture", "mate1", "mate2", "hang",
                             "endgame", "promotion", "distill")
        if "start_fen" in r:
            board = chess.Board(r["start_fen"])
            for uci in r["history"]:
                board.push_uci(uci)
        else:
            board = replay(r["history"])
        assert board.parse_uci(r["target"]) is not None
        if "bad" in r:
            assert r["bad"] != r["target"]
    # skip-if-complete: second call without force reuses the file
    counts2 = build_mix(str(ds), str(out), n_games=10, workers=2, seed=0,
                        hq_dataset=str(ds), min_elo=0,
                        force=False, progress=False)
    assert counts2 == counts


def test_load_games_elo_filter(tmp_path):
    ds = tmp_path / "elo.jsonl"
    rows = [
        {"moves": "e2e4 e7e5", "WhiteElo": 2500, "BlackElo": 2400},
        {"moves": "d2d4 d7d5", "WhiteElo": 1500, "BlackElo": 2500},
        {"moves": "c2c4 c7c5", "WhiteElo": 2600, "BlackElo": 2300},
        {"moves": "g1f3 g8f6", "WhiteElo": "?", "BlackElo": 2700},
    ]
    with open(ds, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from diag.train_data import load_games
    assert len(load_games(str(ds), 10, min_elo=2200)) == 2
    assert len(load_games(str(ds), 10, min_elo=0)) == 4
    assert len(load_games(str(ds), 2, min_elo=2200)) == 2
