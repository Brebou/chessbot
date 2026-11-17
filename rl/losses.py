"""Losses for the RL update: policy gradient, KL penalty, replay CE.

The update recomputes log-probs from the tokenized histories with a fresh
forward pass (so multiple inner epochs / mini-batches stay valid), using the
same last-position-only trick as ``diag.fast_probe``: hidden states → gather
last non-pad position → tied-head projection → log-softmax over the vocab.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def histories_to_ids(histories: Sequence[Sequence[int]], pad_id: int,
                     bos_id: int) -> np.ndarray:
    """(B, T) int64 padded token matrix: [BOS] m1 ... mk [PAD]..."""
    n = len(histories)
    lengths = np.fromiter((1 + len(h) for h in histories), dtype=np.int64, count=n)
    arr = np.full((n, int(lengths.max())), pad_id, dtype=np.int64)
    for i, h in enumerate(histories):
        arr[i, 0] = bos_id
        arr[i, 1:1 + len(h)] = h
    return arr


def last_logprobs(model, batch_ids: torch.Tensor, lengths: torch.Tensor,
                  action_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-prob of ``action_ids`` at the last non-pad position, plus entropy.

    Returns (logprobs (B,), entropies (B,)) — both differentiable w.r.t. the
    model parameters.
    """
    hidden = model.hidden_states(batch_ids)
    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
    logits = F.linear(last, model.wte.weight)
    logp = torch.log_softmax(logits.float(), dim=-1)
    chosen = logp.gather(1, action_ids.unsqueeze(1))[:, 0]
    entropy = -(logp.exp() * logp).sum(dim=-1)
    return chosen, entropy


def policy_gradient_loss(advantages: torch.Tensor, new_logprobs: torch.Tensor,
                         old_logprobs: torch.Tensor, ppo_clip: float) -> torch.Tensor:
    """Clipped surrogate (PPO) if ``ppo_clip > 0``, else plain REINFORCE."""
    if ppo_clip > 0:
        ratio = torch.exp(new_logprobs - old_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
        return -torch.minimum(unclipped, clipped).mean()
    return -(advantages * new_logprobs).mean()


def kl_penalty_k3(new_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Unbiased k3 KL estimator at the sampled token:
    exp(log_ref - log_new) - (log_ref - log_new) - 1  (>= 0)."""
    log_ratio = ref_logprobs - new_logprobs
    return (torch.exp(log_ratio) - log_ratio - 1.0).mean()


def sft_ce_loss(model, batch_ids: torch.Tensor, lengths: torch.Tensor,
                action_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the target move at the last position (replay loss)."""
    hidden = model.hidden_states(batch_ids)
    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
    logits = F.linear(last, model.wte.weight)
    return F.cross_entropy(logits.float(), action_ids)
