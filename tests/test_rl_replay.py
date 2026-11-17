"""Tests for the replay sampler (anti-forgetting batches)."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessTokenizer
from rl.replay import ReplaySampler


@pytest.fixture()
def tok():
    return ChessTokenizer()


def write_corpus(path, games, gz=False):
    opener = gzip.open if gz else open
    mode = "wt" if gz else "w"
    with opener(path, mode, encoding="utf-8") as f:
        for moves in games:
            f.write(json.dumps({"moves": moves}) + "\n")


def test_missing_corpus_reports_unavailable(tmp_path, tok):
    s = ReplaySampler(str(tmp_path / "nope.jsonl.gz"), tok, batch_size=4)
    assert s.available is False
    assert s.next_batch() is None


def test_batch_shapes_and_targets(tmp_path, tok):
    path = tmp_path / "c.jsonl"
    write_corpus(path, ["e2e4 e7e5 g1f3", "d2d4 d7d5 c2c4", "e2e4 c7c5 g1f3 d7d5"])
    s = ReplaySampler(str(path), tok, batch_size=3, seed=1)
    assert s.available
    ids, targets = s.next_batch()
    assert ids.shape[0] == 3
    assert ids.shape[1] == 4  # BOS + 3 moves
    assert targets.shape == (3,)
    assert ids[0, 0] == tok.bos_token_id
    first_moves = {tok.token_to_id[m] for m in ("e2e4", "d2d4")}
    assert all(row[1] in first_moves for row in ids)
    # target is the move right after the last history move
    assert set(targets.tolist()) <= {tok.token_to_id[m] for m in
                                     ("g1f3", "c2c4", "d7d5")}


def test_truncation_preserves_parity(tmp_path, tok):
    path = tmp_path / "c.jsonl"
    long_game = " ".join(["e2e4", "e7e5"] * 200)  # 400 moves > 255
    write_corpus(path, [long_game])
    s = ReplaySampler(str(path), tok, batch_size=1, max_moves=255, seed=0)
    ids, targets = s.next_batch()
    assert ids.shape[1] <= 256
    # even number of moves dropped from the front: history length parity kept
    n_hist = ids.shape[1] - 1
    assert (400 - 1 - n_hist) % 2 == 0


def test_deterministic_given_seed(tmp_path, tok):
    path = tmp_path / "c.jsonl"
    last = ("b8c6", "g8f6", "d7d5", "d7d6", "f8e7", "c7c5", "a7a6", "b7b5")
    games = [f"e2e4 e7e5 g1f3 {m}" for m in last]
    write_corpus(path, games)
    out = []
    for seed in (7, 7, 8):
        s = ReplaySampler(str(path), tok, batch_size=4, seed=seed)
        b = s.next_batch()
        out.append((b[0].tolist(), b[1].tolist()))
    assert out[0] == out[1]
    assert out[0] != out[2]


def test_gzip_corpus(tmp_path, tok):
    path = tmp_path / "c.jsonl.gz"
    write_corpus(path, ["e2e4 e7e5"], gz=True)
    s = ReplaySampler(str(path), tok, batch_size=1, seed=0)
    ids, targets = s.next_batch()
    assert ids.shape[0] == 1
    assert targets[0] == tok.token_to_id["e7e5"]


def test_epoch_wraparound(tmp_path, tok):
    path = tmp_path / "c.jsonl"
    write_corpus(path, ["e2e4 e7e5", "d2d4 d7d5"])
    s = ReplaySampler(str(path), tok, batch_size=2, seed=0)
    batches = [s.next_batch() for _ in range(5)]
    assert all(b is not None and len(b[0]) == 2 for b in batches)
