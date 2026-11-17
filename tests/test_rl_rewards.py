"""Tests for RL rewards, returns and advantages."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.rewards import (compute_advantages, compute_returns, compute_rewards,
                        rollout_stats)
from rl.rollout import RolloutGame, Transition


def make_game(transitions_spec, score=0.0, final_material=0.0, opening_idx=7,
              color="white", level=1):
    g = RolloutGame(opening_idx=opening_idx, model_color=color, level=level,
                    model_score=score, final_material=final_material,
                    result="1-0" if score == 1.0 else ("1/2-1/2" if score == 0.5 else "0-1"))
    for ply, mat, legal in transitions_spec:
        g.transitions.append(Transition(history=["e2e4"] * ply, token_id=42,
                                        legal=legal, logprob=-1.0, ply=ply,
                                        material=mat))
    return g


def test_returns_discounting():
    rets = compute_returns([1.0, 0.0, 2.0], gamma=0.5)
    assert rets[2] == pytest.approx(2.0)
    assert rets[1] == pytest.approx(0.0 + 0.5 * 2.0)
    assert rets[0] == pytest.approx(1.0 + 0.5 * (0.0 + 0.5 * 2.0))


def test_rewards_material_shaping_and_terminal():
    # material goes 0 -> +1 (capture) -> +1 -> final +1, model wins
    g = make_game([(4, 0.0, True), (6, 1.0, True), (8, 1.0, True)],
                  score=1.0, final_material=1.0)
    rewards = compute_rewards(g, illegal_penalty=-0.25, result_reward=1.0,
                              material_shaping=0.1)
    assert rewards[0] == pytest.approx(0.1 * (1.0 - 0.0))
    assert rewards[1] == pytest.approx(0.1 * (1.0 - 1.0))
    assert rewards[2] == pytest.approx(0.1 * (1.0 - 1.0) + 1.0)


def test_rewards_illegal_transition_penalized_without_shaping():
    g = make_game([(4, 0.0, False), (6, 1.0, True)], score=0.0,
                  final_material=1.0)
    rewards = compute_rewards(g, illegal_penalty=-0.25, result_reward=1.0,
                              material_shaping=0.1)
    assert rewards[0] == pytest.approx(-0.25)
    assert rewards[1] == pytest.approx(0.1 * (1.0 - 1.0) + 0.0)


def test_group_advantages_centered_per_group():
    # two groups with very different returns; each must be centered internally
    games = [
        make_game([(4, 0.0, True)], score=1.0, opening_idx=1, level=1),
        make_game([(4, 0.0, True)], score=0.0, opening_idx=1, level=1),
        make_game([(4, 5.0, True)], score=1.0, opening_idx=2, level=1),
        make_game([(4, 5.0, True)], score=0.0, opening_idx=2, level=1),
    ]
    advs = compute_advantages(games, gamma=1.0, illegal_penalty=-0.25,
                              result_reward=1.0, material_shaping=0.1,
                              baseline="group", normalize=False)
    g1 = [a[0] for a in advs[:2]]
    g2 = [a[0] for a in advs[2:]]
    assert sum(g1) == pytest.approx(0.0, abs=1e-9)
    assert sum(g2) == pytest.approx(0.0, abs=1e-9)
    assert g1[0] > 0 > g1[1]
    assert g2[0] > 0 > g2[1]


def test_global_baseline_uses_all_transitions():
    games = [
        make_game([(4, 0.0, True)], score=1.0, opening_idx=1, level=1),
        make_game([(4, 0.0, True)], score=0.0, opening_idx=2, level=1),
    ]
    advs = compute_advantages(games, gamma=1.0, illegal_penalty=-0.25,
                              result_reward=1.0, material_shaping=0.1,
                              baseline="global", normalize=False)
    flat = [a[0] for a in advs]
    assert sum(flat) == pytest.approx(0.0, abs=1e-9)


def test_normalized_advantages_have_unit_std():
    import random
    rng = random.Random(0)
    games = [make_game([(4, rng.uniform(-3, 3), True)], score=rng.choice([0, 0.5, 1]),
                       opening_idx=rng.randrange(3), level=1) for _ in range(12)]
    advs = compute_advantages(games, gamma=1.0, illegal_penalty=-0.25,
                              result_reward=1.0, material_shaping=0.1,
                              baseline="group", normalize=True)
    flat = [a[0] for a in advs]
    mean = sum(flat) / len(flat)
    var = sum((x - mean) ** 2 for x in flat) / len(flat)
    assert mean == pytest.approx(0.0, abs=1e-6)
    assert var == pytest.approx(1.0, abs=1e-3)


def test_rollout_stats():
    games = [
        make_game([(4, 0.0, True), (6, 1.0, False)], score=1.0),
        make_game([(4, 0.0, True)], score=0.5, level=2),
    ]
    stats = rollout_stats(games)
    assert stats["games"] == 2
    assert stats["decisions"] == 3
    assert stats["illegal_rate"] == pytest.approx(1 / 3)
    assert stats["mean_score"] == pytest.approx(0.75)
    assert stats["by_level"]["1"]["games"] == 1
    assert stats["by_level"]["2"]["games"] == 1
