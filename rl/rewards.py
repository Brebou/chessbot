"""Rewards, discounted returns and group-normalized advantages.

Reward design (per model decision, from the mover's point of view):

- **illegal move**: the sampled token was not legal → fixed
  ``illegal_penalty`` (no material shaping; the game continued with the
  seeded random fallback, exactly like eval-time behavior);
- **legal move**: dense material shaping ``material_shaping × (m_{i+1} - m_i)``
  where ``m_i`` is the material diff (mover POV) at decision ``i`` and
  ``m_{i+1}`` is the material diff at the mover's *next* decision (i.e. after
  the opponent replied) — or the final material for the mover's last
  decision. A capture the opponent does not recoup credits positively; a
  hung piece charges negatively against the decision that hung it;
- **terminal bonus**: the last decision of each color receives
  ``result_reward × score(color)`` — in self-play games both colors are the
  same policy, so each side is credited with its own outcome.

Returns are discounted with ``gamma`` from each decision to the end of the
game. Advantages follow GRPO: returns are centered (and optionally scaled)
within a group of games sharing identical starting conditions — (opening,
model color, engine level) for engine games, (opening,) for self-play — or
with ``baseline="global"`` a single mean over all transitions is subtracted
(plain REINFORCE with a global baseline).

Mix note: engine games optimize the engine score (grade 1) but over-optimizing
against the alpha-beta opponent is known to distort play style and *lower*
round-robin Elo; self-play games keep the policy distribution tournament-like.
Keep both in the mix and watch the vs-reference eval (see ``rl/train.py``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from rl.rollout import RolloutGame, Transition, score_for_color


def compute_rewards(game: RolloutGame, *, illegal_penalty: float,
                    result_reward: float, material_shaping: float) -> List[float]:
    """Per-transition rewards for one game (aligned with game.transitions)."""
    mat_white = [t.material_white for t in game.transitions]
    final_white = game.final_material_white

    by_color: Dict[str, List[int]] = {}
    for i, t in enumerate(game.transitions):
        by_color.setdefault(t.mover, []).append(i)

    rewards = [0.0] * len(game.transitions)
    for color, idxs in by_color.items():
        sign = 1.0 if color == "white" else -1.0
        for j, i in enumerate(idxs):
            t = game.transitions[i]
            if not t.legal:
                rewards[i] = illegal_penalty
                continue
            next_mat = mat_white[idxs[j + 1]] if j + 1 < len(idxs) else final_white
            r = material_shaping * sign * (next_mat - mat_white[i])
            if j == len(idxs) - 1:
                r += result_reward * score_for_color(game.result, color)
            rewards[i] = r
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
    engine_games = [g for g in games if not g.selfplay]
    selfplay_games = [g for g in games if g.selfplay]
    decisions = sum(len(g.transitions) for g in games)
    illegal = sum(1 for g in games for t in g.transitions if not t.legal)
    stats = {
        "games": n,
        "engine_games": len(engine_games),
        "selfplay_games": len(selfplay_games),
        "decisions": decisions,
        "illegal_rate": illegal / max(decisions, 1),
        "mean_plies": sum(g.transitions[-1].ply + 1 for g in games if g.transitions) / n,
    }
    if engine_games:
        stats["mean_score"] = (sum(g.model_score for g in engine_games)
                               / len(engine_games))
        stats["wins"] = sum(g.model_score == 1.0 for g in engine_games)
        stats["draws"] = sum(g.model_score == 0.5 for g in engine_games)
        stats["losses"] = sum(g.model_score == 0.0 for g in engine_games)
        stats["by_level"] = {
            str(lvl): {
                "games": sum(1 for g in engine_games if g.level == lvl),
                "mean_score": (sum(g.model_score for g in engine_games if g.level == lvl)
                               / max(sum(1 for g in engine_games if g.level == lvl), 1)),
            }
            for lvl in sorted({g.level for g in engine_games})
        }
    if selfplay_games:
        decisive = sum(1 for g in selfplay_games if g.result != "1/2-1/2")
        stats["selfplay_decisive_rate"] = decisive / len(selfplay_games)
    return stats
