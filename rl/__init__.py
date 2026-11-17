"""RL framework for ChessGPT post-training (Phase 3 of the plan d'attaque).

Policy-gradient training (REINFORCE / GRPO-style) of the fixed ChessGPT
architecture against the provided engine, with:

- batched sampling rollouts (GPU) vs ``diag.fast_engine.FastEngine`` (CPU);
- reward = game result + dense material shaping + illegal-move penalty;
- group-normalized advantages (GRPO) keyed by (opening, color, level);
- KL penalty to a frozen reference policy (k3 estimator on sampled tokens);
- Lichess replay CE mixed in against catastrophic forgetting;
- multi-seed evaluation vs the engine and best-checkpoint tracking.

Everything is seeded, atomically written, and never touches ``chess_hw/``.
"""

from rl.config import RLConfig
from rl.policy import PolicySampler, SampledMove
from rl.rewards import compute_advantages, compute_returns, compute_rewards
from rl.rollout import RolloutGame, Transition, generate_rollouts

__all__ = [
    "RLConfig",
    "PolicySampler",
    "SampledMove",
    "Transition",
    "RolloutGame",
    "generate_rollouts",
    "compute_rewards",
    "compute_returns",
    "compute_advantages",
]
