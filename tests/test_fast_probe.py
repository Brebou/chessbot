"""Tests for FastProbePlayer context-overflow truncation and OOM fallback."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessTokenizer
from chess_hw.model import ChessGPT, CONFIG
from diag.fast_probe import FastProbePlayer


@pytest.fixture(scope="module")
def player():
    torch.manual_seed(0)
    model = ChessGPT(CONFIG)
    return FastProbePlayer(model, ChessTokenizer(), batch_size=4, device="cpu")


def test_long_history_truncated_not_raised(player):
    long_history = ["e2e4", "e7e5"] * 209
    out = player.last_logits([long_history, ["e2e4"]])
    assert out.shape == (2, 1971)
    assert np.isfinite(out).all()


def test_truncated_logits_match_manual_truncation(player):
    long_history = ["a2a3", "b7b6"] * 210
    max_moves = player._n_ctx - 1
    excess = len(long_history) - max_moves
    truncated = list(long_history[excess + (excess & 1):])
    assert len(truncated) <= max_moves
    out_long = player.last_logits([long_history])[0]
    out_trunc = player.last_logits([truncated])[0]
    np.testing.assert_allclose(out_long, out_trunc, rtol=1e-5, atol=1e-5)


def test_fit_history_preserves_parity(player):
    history = ["a2a3", "b7b6"] * 130
    fitted = list(player._fit_history(history))
    assert len(fitted) <= player._n_ctx - 1
    dropped = len(history) - len(fitted)
    assert dropped > 0 and dropped % 2 == 0
    assert fitted[-1] == history[-1]


def test_choose_from_histories_accepts_long_history(player):
    import chess

    long_history = ["a2a3", "b7b6"] * 210
    choices = player.choose_from_histories([long_history], [chess.Board()])
    assert len(choices) == 1
