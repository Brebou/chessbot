"""Tests for RL rollout generation (tiny games vs the engine)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from chess_hw import ChessTokenizer
from chess_hw.match import load_openings
from chess_hw.model import ChessGPT, CONFIG
from diag.common import replay
from rl.rollout import generate_rollouts


TOKENIZER = ChessTokenizer()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ChessGPT(CONFIG)


def test_rollouts_structure(model):
    games = generate_rollouts(
        model, ChessTokenizer(), n_games=4, levels=(0,), group_size=2,
        seed=5, temperature=1.0, batch_size=4, max_plies=12, progress=False)
    assert len(games) == 4
    for g in games:
        assert g.result in ("1-0", "0-1", "1/2-1/2")
        assert g.model_score in (0.0, 0.5, 1.0)
        assert g.level == 0
        assert 0 <= g.opening_idx < len(load_openings())
        for t in g.transitions:
            board = replay(t.history)
            assert (board.turn == chess.WHITE) == (g.model_color == "white")
            assert t.ply == len(t.history)
            if t.legal:
                tok = TOKENIZER.id_to_token[t.token_id]
                assert board.parse_uci(tok) is not None
    # two groups of 2 identical (opening, color) pairs
    keys = [g.group_key for g in games]
    assert len(set(keys)) <= 2


def test_rollouts_reproducible(model):
    kw = dict(n_games=2, levels=(0,), group_size=1, seed=9, temperature=1.0,
              batch_size=2, max_plies=10, progress=False)
    g1 = generate_rollouts(model, ChessTokenizer(), **kw)
    g2 = generate_rollouts(model, ChessTokenizer(), **kw)
    assert [g.result for g in g1] == [g.result for g in g2]
    assert [[t.token_id for t in g.transitions] for g in g1] == \
           [[t.token_id for t in g.transitions] for g in g2]


def test_rollouts_material_recorded(model):
    games = generate_rollouts(
        model, ChessTokenizer(), n_games=2, levels=(0,), group_size=1,
        seed=11, temperature=1.0, batch_size=2, max_plies=16, progress=False)
    for g in games:
        mats = [t.material for t in g.transitions]
        assert all(isinstance(m, float) for m in mats)
        assert all(-40.0 <= m <= 40.0 for m in mats)
