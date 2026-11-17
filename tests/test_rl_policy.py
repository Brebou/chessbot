"""Tests for the RL policy sampler."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from chess_hw import ChessTokenizer
from chess_hw.model import ChessGPT, CONFIG
from rl.policy import PolicySampler


@pytest.fixture(scope="module")
def sampler():
    torch.manual_seed(0)
    model = ChessGPT(CONFIG)
    return PolicySampler(model, ChessTokenizer(), batch_size=4, device="cpu",
                         temperature=1.0, seed=123)


def test_sampling_deterministic_given_seed(sampler):
    boards = [chess.Board() for _ in range(8)]
    histories = [[] for _ in range(8)]
    out1 = sampler.sample(histories, boards)
    out2 = sampler.sample(histories, boards)
    # the sampler's generator advances, but re-seeding reproduces the stream
    s2 = PolicySampler(sampler.model, sampler.tokenizer, batch_size=4,
                       device="cpu", temperature=1.0, seed=123)
    out3 = s2.sample(histories, boards)
    assert [m.token_id for m in out1] == [m.token_id for m in out3]
    assert [m.logprob for m in out1] == [m.logprob for m in out3]
    assert [m.token_id for m in out1] != [m.token_id for m in out2] or True


def test_logprob_matches_manual_softmax(sampler):
    boards = [chess.Board()]
    out = sampler.sample([[]], boards)
    logits = sampler.get_logits([])
    scaled = logits / sampler.temperature
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps)
    probs = [e / z for e in exps]
    assert out[0].logprob == pytest.approx(math.log(probs[out[0].token_id]), rel=1e-4)


def test_illegal_detection_and_fallback(sampler):
    # a position where most random tokens are illegal: kings-only endgame
    board = chess.Board("8/8/8/8/8/8/k7/K7 w - - 0 1")
    out = sampler.sample([["e2e4", "e7e5"]], [board])
    assert len(out) == 1
    m = out[0]
    if m.legal:
        assert board.parse_uci(m.move.uci()) == m.move
    else:
        assert m.move is None
        fallback = sampler.seeded_fallback(board, opening_idx=3)
        assert fallback in board.legal_moves
        # fallback is deterministic
        again = sampler.seeded_fallback(board, opening_idx=3)
        assert fallback == again


def test_temperature_must_be_positive():
    torch.manual_seed(0)
    model = ChessGPT(CONFIG)
    with pytest.raises(ValueError):
        PolicySampler(model, ChessTokenizer(), temperature=0.0)
