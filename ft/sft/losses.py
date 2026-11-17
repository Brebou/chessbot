"""ft SFT losses: weighted CE + bounded softplus contrastive term.

    loss = weighted_ce + LAMBDA * contrastive

Contrastive pairs (good, bad): d = logp(good) - logp(bad) at the last
position; per-pair loss = clamp(softplus(-BETA * d), max=CAP). With
BETA = 1.0 the term saturates near zero once the good move outranks the bad
by > ~4 nats (vs ~40 nats with the old beta=0.1 logsigmoid), and the clamp
keeps any pathological pair from dominating the batch. Tune LAMBDA so the
contrastive gradient norm contributes ~10-15% of the total (log it).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

BETA = 1.0
LAMBDA = 0.5
PAIR_CAP = 8.0


def last_logprobs(model, batch_ids: torch.Tensor,
                  lengths: torch.Tensor) -> torch.Tensor:
    hidden = model.hidden_states(batch_ids)
    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
    logits = F.linear(last, model.wte.weight).float()
    return torch.log_softmax(logits, dim=-1)


def weighted_ce(logp: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
    ce = -logp.gather(1, targets.unsqueeze(1))[:, 0]
    return (weights * ce).sum() / weights.sum().clamp(min=1e-8)


def contrastive_loss(logp: torch.Tensor, good: torch.Tensor, bad: torch.Tensor,
                     beta: float = BETA, cap: float = PAIR_CAP) -> torch.Tensor:
    d = logp.gather(1, good.unsqueeze(1))[:, 0] - logp.gather(1, bad.unsqueeze(1))[:, 0]
    per_pair = torch.clamp(F.softplus(-beta * d), max=cap)
    return per_pair.mean()


def ft_loss(model, batch_ids: torch.Tensor, lengths: torch.Tensor,
            ce_targets: torch.Tensor, ce_weights: torch.Tensor,
            good: Optional[torch.Tensor] = None, bad: Optional[torch.Tensor] = None,
            bad_mask: Optional[torch.Tensor] = None,
            beta: float = BETA, lambda_contrast: float = LAMBDA,
            ) -> Tuple[torch.Tensor, dict]:
    logp = last_logprobs(model, batch_ids, lengths)
    ce = weighted_ce(logp, ce_targets, ce_weights)
    parts = {"ce": ce.item()}
    loss = ce
    if good is not None and bad is not None and bad_mask is not None and bool(bad_mask.any()):
        contra = contrastive_loss(logp[bad_mask], good[bad_mask], bad[bad_mask], beta)
        parts["contrastive"] = contra.item()
        loss = loss + lambda_contrast * contra
    return loss, parts
