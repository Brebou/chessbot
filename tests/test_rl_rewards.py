"""Tests for RL rewards, returns and advantages."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.rewards import (compute_advantages, compute_returns, compute_rewards,
                        rollout_stats)
from rl.rollout import RolloutGame, Transition


def make_game(transitions_spec, score=0.0, final_material_white=0.0,
              opening_idx=7, color="white", level=1, selfplay=False,
              result=None):
    if result is None:
        result = "1-0" if score == 1.0 else ("1/2-1/2" if score == 0.5 else "0-1")
    g = RolloutGame(opening_idx=opening_idx, model_color=color, level=level,
                    selfplay=selfplay, model_score=score,
                    final_material_white=final_material_white, result=result)
    for spec in transitions_spec:
        ply, mat_white, legal = spec[0], spec[1], spec[2]
        mover = spec[3] if len(spec) > 3 else color
        g.transitions.append(Transition(history=["e2e4"] * ply, token_id=42,
                                        legal=legal, logprob=-1.0, ply=ply,
                                        mover=mover, material_white=mat_white))
    return g


def test_returns_discounting():
    rets = compute_returns([1.0, 0.0, 2.0], gamma=0.5)
    assert rets[2] == pytest.approx(2.0)
    assert rets[1] == pytest.approx(0.0 + 0.5 * 2.0)
    assert rets[0] == pytest.approx(1.0 + 0.5 * (0.0 + 0.5 * 2.0))


def test_rewards_material_shaping_and_terminal():
    # model is white; White-POV material goes 0 -> +1 (capture) -> +1 -> final +1
    g = make_game([(4, 0.0, True), (6, 1.0, True), (8, 1.0, True)],
                  score=1.0, final_material_white=1.0)
    rewards = compute_rewards(g, illegal_penalty=-0.25, result_reward=1.0,
                              material_shaping=0.1)
    assert rewards[0] == pytest.approx(0.1 * (1.0 - 0.0))
    assert rewards[1] == pytest.approx(0.1 * (1.0 - 1.0))
    assert rewards[2] == pytest.approx(0.1 * (1.0 - 1.0) + 1.0)


def test_rewards_black_mover_sign_flipped():
    # model is black; White-POV material +1 means the model is DOWN a pawn
    g = make_game([(4, 1.0, True), (6, -1.0, True)], score=1.0, color="black",
                  final_material_white=-1.0, result="0-1")
    rewards = compute_rewards(g, illegal_penalty=-0.25, result_reward=1.0,
                              material_shaping=0.1)
    # black captured a piece: White-POV went +1 -> -1, black POV delta = +2
    assert rewards[0] == pytest.approx(0.1 * (-1.0 - 1.0) * -1.0)
    assert rewards[1] == pytest.approx(0.1 * (-1.0 - -1.0) * -1.0 + 1.0)


def test_rewards_illegal_transition_penalized_without_shaping():
    g = make_game([(4, 0.0, False), (6, 1.0, True)], score=0.0,
                  final_material_white=1.0)
    rewards = compute_rewards(g, illegal_penalty=-0.25, result_reward=1.0,
                              material_shaping=0.1)
    assert rewards[0] == pytest.approx(-0.25)
    assert rewards[1] == pytest.approx(0.1 * (1.0 - 1.0) + 0.0)


def test_rewards_selfplay_each_color_credited_own_outcome():
    # self-play: white wins. Last white decision gets +1, last black gets 0.
    g = make_game(
        [(4, 0.0, True, "white"), (5, 0.0, True, "black"),
         (6, 1.0, True, "white"), (7, 1.0, True, "black")],
        score=1.0, final_material_white=1.0, selfplay=True, result="1-0")
    rewards = compute_rewards(g, illegal_penalty=-0.25, result_reward=1.0,
                              material_shaping=0.1)
    # white decisions: shaping deltas +1 then 0, terminal +1
    assert rewards[0] == pytest.approx(0.1 * (1.0 - 0.0))
    assert rewards[2] == pytest.approx(0.1 * (1.0 - 1.0) + 1.0)
    # black decisions: shaping from its own next decision, terminal 0 (lost)
    assert rewards[1] == pytest.approx(0.1 * -1.0 * (1.0 - 0.0))
    assert rewards[3] == pytest.approx(0.1 * -1.0 * (1.0 - 1.0) + 0.0)


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


def test_selfplay_group_key_ignores_color_and_level():
    g1 = make_game([(4, 0.0, True)], selfplay=True, opening_idx=3)
    g2 = make_game([(4, 0.0, True)], selfplay=True, opening_idx=3)
    g3 = make_game([(4, 0.0, True)], selfplay=False, opening_idx=3)
    assert g1.group_key == g2.group_key == (3, "self", -1)
    assert g3.group_key != g1.group_key


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
    games = [make_game([(4, rng.uniform(-3, 3), True)],
                       score=rng.choice([0, 0.5, 1]),
                       opening_idx=rng.randrange(3), level=1) for _ in range(12)]
    advs = compute_advantages(games, gamma=1.0, illegal_penalty=-0.25,
                              result_reward=1.0, material_shaping=0.1,
                              baseline="group", normalize=True)
    flat = [a[0] for a in advs]
    mean = sum(flat) / len(flat)
    var = sum((x - mean) ** 2 for x in flat) / len(flat)
    assert mean == pytest.approx(0.0, abs=1e-6)
    assert var == pytest.approx(1.0, abs=1e-3)


def test_rollout_stats_mixed():
    games = [
        make_game([(4, 0.0, True), (6, 1.0, False)], score=1.0),
        make_game([(4, 0.0, True)], score=0.5, level=2),
        make_game([(4, 0.0, True, "white"), (5, 0.0, True, "black")],
                  selfplay=True, score=1.0, result="1-0"),
    ]
    stats = rollout_stats(games)
    assert stats["games"] == 3
    assert stats["engine_games"] == 2
    assert stats["selfplay_games"] == 1
    assert stats["decisions"] == 5
    assert stats["illegal_rate"] == pytest.approx(1 / 5)
    assert stats["mean_score"] == pytest.approx(0.75)  # engine games only
    assert stats["selfplay_decisive_rate"] == 1.0
    assert stats["by_level"]["1"]["games"] == 1
    assert stats["by_level"]["2"]["games"] == 1
