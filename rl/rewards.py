"""Rewards, discounted returns and group-normalized advantages.

Reward design (per model decision, from the model's point of view):

- **illegal move**: the sampled token was not legal → fixed
  ``illegal_penalty`` (no material shaping; the game continued with the
  seeded random fallback, exactly like eval-time behavior);
- **legal move**: dense material shaping ``material_shaping * (m_{i+1} - m_i)``
  where ``m_i`` is the material diff (model POV) at decision ``i`` and
  ``m_{i+1}`` is the material diff at the next model decision (i.e. after the
  engine replied) — or the final material for the last decision. So a capture
  the engine does not recoup credits positively, and a hung piece credits
  negatively against the decision that hung it;
- **terminal bonus**: the last decision additionally receives
  ``result_reward * model_score``.

Returns are discounted with ``gamma`` from each decision to the end of the
game. Advantages follow GRPO: returns are centered (and optionally scaled)
within a group of games sharing the same (opening, model color, engine
level), so the baseline compares identical starting conditions; with
``baseline="global"`` a single mean over all transitions is subtracted
instead (plain REINFORCE with a global baseline).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from rl.rollout import RolloutGame


def compute_rewards(game: RolloutGame, *, illegal_penalty: float,
                    result_reward: float, material_shaping: float) -> List[float]:
    """Per-transition rewards for one game (aligned with game.transitions)."""
    mats = [t.material for t in game.transitions] + [game.final_material]
    rewards: List[float] = []
    for i, t in enumerate(game.transitions):
        if not t.legal:
            rewards.append(illegal_penalty)
            continue
        r = material_shaping * (mats[i + 1] - mats[i])
        if i == len(game.transitions) - 1:
            r += result_reward * game.model_score
        rewards.append(r)
    return rewards


def compute_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    """Discounted return from each step to the end of the reward sequence."""
    returns = [0.0] * len(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        returns[i] = acc
    return returns


def compute_advantages(
    games: Sequence[RolloutGame], *, gamma: float, illegal_penalty: float,
    result_reward: float, material_shaping: float, baseline: str = "group",
    normalize: bool = True,
) -> List[List[float]]:
    """Per-transition advantages for every game, aligned with
    ``game.transitions`` (empty list for a game with no transitions)."""
    all_returns: List[List[float]] = []
    for game in games:
        rewards = compute_rewards(game, illegal_penalty=illegal_penalty,
                                  result_reward=result_reward,
                                  material_shaping=material_shaping)
        all_returns.append(compute_returns(rewards, gamma))

    if baseline == "group":
        groups: Dict[tuple, List[float]] = {}
        for game, rets in zip(games, all_returns):
            groups.setdefault(game.group_key, []).extend(rets)
        means = {k: sum(v) / max(len(v), 1) for k, v in groups.items()}
        stds = {}
        for k, v in groups.items():
            if normalize and len(v) > 1:
                mu = means[k]
                var = sum((x - mu) ** 2 for x in v) / len(v)
                stds[k] = var ** 0.5
            else:
                stds[k] = 0.0
        advantages = []
        for game, rets in zip(games, all_returns):
            mu, sd = means[game.group_key], stds[game.group_key]
            if normalize and sd > 1e-8:
                advantages.append([(r - mu) / (sd + 1e-8) for r in rets])
            else:
                advantages.append([r - mu for r in rets])
        return advantages

    # global baseline (plain REINFORCE)
    flat = [r for rets in all_returns for r in rets]
    mu = sum(flat) / max(len(flat), 1)
    sd = 0.0
    if normalize and len(flat) > 1:
        var = sum((x - mu) ** 2 for x in flat) / len(flat)
        sd = var ** 0.5
    out = []
    for rets in all_returns:
        if normalize and sd > 1e-8:
            out.append([(r - mu) / (sd + 1e-8) for r in rets])
        else:
            out.append([r - mu for r in rets])
    return out


def rollout_stats(games: Sequence[RolloutGame]) -> dict:
    """Summary statistics of a rollout batch (for logging)."""
    n = len(games)
    if n == 0:
        return {"games": 0}
    decisions = sum(len(g.transitions) for g in games)
    illegal = sum(1 for g in games for t in g.transitions if not t.legal)
    return {
        "games": n,
        "decisions": decisions,
        "illegal_rate": illegal / max(decisions, 1),
        "mean_score": sum(g.model_score for g in games) / n,
        "wins": sum(g.model_score == 1.0 for g in games),
        "draws": sum(g.model_score == 0.5 for g in games),
        "losses": sum(g.model_score == 0.0 for g in games),
        "mean_plies": sum(g.transitions[-1].ply + 1 for g in games if g.transitions) / n,
        "by_level": {
            str(lvl): {
                "games": sum(1 for g in games if g.level == lvl),
                "mean_score": (sum(g.model_score for g in games if g.level == lvl)
                               / max(sum(1 for g in games if g.level == lvl), 1)),
            }
            for lvl in sorted({g.level for g in games})
        },
    }
