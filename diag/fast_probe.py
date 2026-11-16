"""Optimized batched model inference for corpus generation (diag-only use).

Replaces the naive full-logits forward pass with a last-position-only one:

- the reference ``model(input_ids)`` materializes logits of shape
  ``(B, T, vocab)`` (~4 GB fp32 for B=2048, T=256, vocab=1971) although only
  the logits at the last non-pad position are ever used. Here we compute
  ``hidden_states`` (B, T, 128), gather the per-row last position and project
  only those rows through the tied embedding head: (B, vocab).
- one single padded int64 tensor is built in numpy and transferred to the
  device (instead of one small transfer per row);
- top-k is computed on the device and only the (B, k) indices cross back to
  the CPU.

Move selection semantics match ``chess_hw.play`` / ``diag.common.ProbePlayer``:
greedy top-3 tokens, first legal one is played, ``attempt`` is its rank
(0 if none of the top-3 is legal).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_hw.play import MoveChoice


class FastProbePlayer:
    """Batched greedy player with last-position logits only."""

    def __init__(self, model, tokenizer, batch_size: int = 1024,
                 device: Optional[str] = None, topk: int = 3):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.batch_size = batch_size
        self.topk = topk
        self._tok_ids = tokenizer.token_to_id
        self._id2tok = tokenizer.id_to_token
        self._pad = tokenizer.pad_token_id
        self._bos = tokenizer.bos_token_id
        self._n_ctx = model.cfg.n_ctx
        self._wte = model.wte.weight

    # -- core forward ------------------------------------------------------

    @torch.inference_mode()
    def _forward_chunk(self, chunk: List[List[str]]) -> torch.Tensor:
        """Logits at the last move for a chunk of histories, shape (b, vocab)."""
        n = len(chunk)
        lengths = np.fromiter((1 + len(h) for h in chunk), dtype=np.int64, count=n)
        max_len = int(lengths.max())
        if max_len > self._n_ctx:
            raise ValueError(f"history length {max_len} exceeds context {self._n_ctx}")
        arr = np.full((n, max_len), self._pad, dtype=np.int64)
        tok_ids = self._tok_ids
        for i, h in enumerate(chunk):
            arr[i, 0] = self._bos
            pos = 1
            for m in h:
                arr[i, pos] = tok_ids[m]
                pos += 1
        padded = torch.from_numpy(arr).to(self.device, non_blocking=True)
        hidden = self.model.hidden_states(padded)
        last_pos = torch.from_numpy(lengths - 1).to(self.device)
        last_hidden = hidden[torch.arange(n, device=self.device), last_pos]
        return F.linear(last_hidden, self._wte)

    @torch.inference_mode()
    def last_logits(self, histories: Sequence[Sequence[str]]) -> np.ndarray:
        """(B, vocab) float32 logits at the last move of each history."""
        n = len(histories)
        if n == 0:
            return np.empty((0, self.tokenizer.vocab_size), dtype=np.float32)
        order = sorted(range(n), key=lambda i: -len(histories[i]))
        out = np.empty((n, self.tokenizer.vocab_size), dtype=np.float32)
        for start in range(0, n, self.batch_size):
            idxs = order[start:start + self.batch_size]
            chunk = [histories[i] for i in idxs]
            logits = self._forward_chunk(chunk)
            out[idxs] = logits.float().cpu().numpy()
        return out

    # -- move selection ----------------------------------------------------

    @torch.inference_mode()
    def choose_from_histories(self, histories: Sequence[Sequence[str]],
                              boards: Sequence[chess.Board]) -> List[MoveChoice]:
        """Greedy top-k move choice for each (history, board) pair.

        ``boards[i]`` must be the position reached by ``histories[i]`` (the
        position *before* the model's move).
        """
        n = len(histories)
        results: List[Optional[MoveChoice]] = [None] * n
        if n == 0:
            return []
        order = sorted(range(n), key=lambda i: -len(histories[i]))
        id2tok = self._id2tok
        for start in range(0, n, self.batch_size):
            idxs = order[start:start + self.batch_size]
            chunk = [histories[i] for i in idxs]
            logits = self._forward_chunk(chunk)
            k = min(self.topk, logits.shape[-1])
            top = torch.topk(logits, k, dim=-1).indices.cpu().numpy()
            for row, i in enumerate(idxs):
                board = boards[i]
                move, attempt = None, 0
                for j, tok_id in enumerate(top[row], start=1):
                    try:
                        move = board.parse_uci(id2tok[tok_id])
                        attempt = j
                        break
                    except ValueError:
                        continue
                results[i] = MoveChoice(move, attempt)
        return results

    def choose_for_boards(self, boards: Sequence[chess.Board]) -> List[MoveChoice]:
        histories = [[m.uci() for m in b.move_stack] for b in boards]
        return self.choose_from_histories(histories, boards)

    # -- single-position convenience (kept for parity with ProbePlayer) ----

    def get_logits(self, history: Sequence[str]) -> np.ndarray:
        return self.last_logits([list(history)])[0]

    def get_topk(self, history: Sequence[str], k: int = 3):
        logits = self.get_logits(history)
        idx = np.argsort(logits)[-k:][::-1]
        tokens = [self._id2tok[int(i)] for i in idx]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[idx]
        return tokens, probs.tolist()

    def choose(self, board: chess.Board) -> MoveChoice:
        return self.choose_for_boards([board])[0]
