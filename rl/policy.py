"""Batched stochastic sampling from the policy, for RL rollouts.

Extends :class:`diag.fast_probe.FastProbePlayer` (last-position-only forward,
parity-preserving truncation, OOM auto sub-batching) with categorical
sampling instead of greedy top-3:

- the move is drawn from ``softmax(logits / temperature)`` over the full
  vocabulary (the model never sees the legal-move list, exactly like at
  inference time);
- the sampling-time log-prob of the drawn token is returned so the rollout
  can be used as the "old" policy in PPO-style clipped updates;
- a drawn token that is not a legal move in the position is reported as
  ``legal=False`` (the rollout then plays a seeded random legal fallback,
  mirroring ``chess_hw.match.play_game`` semantics, and the transition is
  penalized).

Determinism: a per-sampler ``torch.Generator`` seeded once at construction
makes sampling byte-reproducible on a given device.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import chess
import torch

from diag.fast_probe import FastProbePlayer


@dataclass
class SampledMove:
    """One sampled decision for one position."""

    token_id: int
    logprob: float                 # log softmax(logits / temperature)[token_id]
    move: Optional[chess.Move]     # None if the token is not a legal move
    legal: bool


class PolicySampler(FastProbePlayer):
    """Batched categorical sampler over the policy's next-move distribution."""

    def __init__(self, model, tokenizer, batch_size: int = 1024,
                 device: Optional[str] = None, temperature: float = 1.0,
                 seed: int = 0):
        super().__init__(model, tokenizer, batch_size=batch_size, device=device)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature
        self.seed = seed
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(random.Random(f"sampler-{seed}").getrandbits(63))

    @torch.inference_mode()
    def sample(self, histories: Sequence[Sequence[str]],
               boards: Sequence[chess.Board]) -> List[SampledMove]:
        """Sample one move per (history, board) pair; ``boards[i]`` is the
        position reached by ``histories[i]``, with the model to move."""
        n = len(histories)
        if n != len(boards):
            raise ValueError("histories and boards must have the same length")
        if n == 0:
            return []

        order = sorted(range(n), key=lambda i: -len(histories[i]))
        out: List[Optional[SampledMove]] = [None] * n
        id2tok = self._id2tok
        for start in range(0, n, self.batch_size):
            idxs = order[start:start + self.batch_size]
            logits = self._forward_auto([histories[i] for i in idxs])
            scaled = logits.float() / self.temperature
            probs = torch.softmax(scaled, dim=-1)
            drawn = torch.multinomial(probs, 1, generator=self._generator)[:, 0]
            logprobs = torch.log_softmax(scaled, dim=-1).gather(1, drawn.unsqueeze(1))[:, 0]
            drawn_ids = drawn.cpu().numpy()
            drawn_lps = logprobs.float().cpu().numpy()
            for row, i in enumerate(idxs):
                token_id = int(drawn_ids[row])
                try:
                    move = boards[i].parse_uci(id2tok[token_id])
                    legal = True
                except ValueError:
                    move, legal = None, False
                out[i] = SampledMove(token_id=token_id, logprob=float(drawn_lps[row]),
                                     move=move, legal=legal)
        return [m for m in out if m is not None]

    def seeded_fallback(self, board: chess.Board, opening_idx: int) -> chess.Move:
        """Seeded random legal move, same convention as match.play_game."""
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        return random.Random(f"fallback-{opening_idx}-{board.ply()}").choice(legal)
