"""Collation, tokenization, game-level splits, and stratified batching.

Batch composition is stratified by bucket with constant ratios
(45/20/15/15/5): each batch draws reserved slots per bucket instead of
i.i.d. sampling, so the mate gradient can never drown. Val is split by
game_id hash (97/3) - never by position - so two plies of the same game
cannot leak across the split. Rows without a game_id (tablebase lines) are
hashed on their start position.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from ft.common.board_io import history_window
from ft.common.serialize import jsonl_read

BUCKET_RATIOS_TRAIN = {"normal": 0.45, "mate": 0.20, "endgame_tb": 0.15,
                       "contrastive": 0.15, "anchor": 0.05}
VAL_FRACTION = 0.03


def game_partition_key(game_id: Optional[str], fen_key: str,
                       fraction: float = VAL_FRACTION) -> str:
    basis = game_id if game_id is not None else fen_key
    h = int(hashlib.md5(basis.encode()).hexdigest(), 16) % 10_000
    return "val" if h < int(fraction * 10_000) else "train"


class SFTItem:
    __slots__ = ("ids", "target_id", "bad_id", "weight", "bucket", "partition")

    def __init__(self, row: dict, tokenizer, partition: str):
        moves = history_window(row["history"])
        self.ids = tokenizer.encode(moves, add_bos=True)
        self.target_id = tokenizer.token_to_id[row["target"]]
        self.bad_id = (tokenizer.token_to_id[row["bad"]]
                       if row.get("bad") else -1)
        self.weight = float(row.get("weight", 1.0))
        self.bucket = row["bucket"]
        self.partition = partition

    def length(self) -> int:
        return len(self.ids)


def load_items(mix_path: str, tokenizer) -> Tuple[List[SFTItem], List[SFTItem]]:
    train_items: List[SFTItem] = []
    val_items: List[SFTItem] = []
    for row in jsonl_read(mix_path):
        fen_key = row.get("key") or row.get("start_fen") or row["history"][0]
        item = SFTItem(row, tokenizer, game_partition_key(row.get("game_id"), fen_key))
        (val_items if item.partition == "val" else train_items).append(item)
    return train_items, val_items


def stratified_batches(items: List["SFTItem"], batch_size: int, seed: int,
                       epoch: int) -> Iterator[List["SFTItem"]]:
    """Epoch = floor(min bucket supply / ratio) batches, each with the exact
    slot allocation; buckets cycle through fresh shuffles when exhausted."""
    rng = random.Random(f"strat-{seed}-{epoch}")
    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_bucket[it.bucket].append(i)
    for v in by_bucket.values():
        rng.shuffle(v)
    scale = min(len(idxs) / BUCKET_RATIOS_TRAIN[b]
                for b, idxs in by_bucket.items() if BUCKET_RATIOS_TRAIN.get(b, 0) > 0)
    n_batches = max(1, int(scale))
    cursor = {b: 0 for b in by_bucket}
    for _ in range(n_batches):
        batch: List["SFTItem"] = []
        for b, ratio in BUCKET_RATIOS_TRAIN.items():
            pool = by_bucket.get(b)
            if not pool:
                continue
            slots = max(1, int(round(ratio * batch_size)))
            for _ in range(slots):
                if cursor[b] >= len(pool):
                    rng.shuffle(pool)
                    cursor[b] = 0
                batch.append(items[pool[cursor[b]]])
                cursor[b] += 1
        rng.shuffle(batch)
        yield batch


def encode_batch(items: List["SFTItem"], tokenizer, device: str):
    lengths = torch.tensor([it.length() for it in items], dtype=torch.long)
    max_len = int(lengths.max().item())
    pad = tokenizer.pad_token_id
    batch_ids = torch.full((len(items), max_len), pad, dtype=torch.long)
    for i, it in enumerate(items):
        batch_ids[i, :it.length()] = torch.tensor(it.ids, dtype=torch.long)
    targets = torch.tensor([it.target_id for it in items], dtype=torch.long)
    weights = torch.tensor([it.weight for it in items], dtype=torch.float)
    has_bad = [it.bad_id >= 0 for it in items]
    good = torch.tensor([it.target_id if h else 0 for it, h in
                         zip(items, has_bad)], dtype=torch.long)
    bad = torch.tensor([it.bad_id if h else 0 for it, h in
                        zip(items, has_bad)], dtype=torch.long)
    bad_mask = torch.tensor(has_bad, dtype=torch.bool)
    return (batch_ids.to(device), lengths.to(device), targets.to(device),
            weights.to(device), good.to(device), bad.to(device), bad_mask.to(device))


def val_loss(model, items: List["SFTItem"], tokenizer, device: str,
             bucket: Optional[str] = None) -> float:
    from ft.sft.losses import weighted_ce
    if bucket is not None:
        items = [it for it in items if it.bucket == bucket]
    if not items:
        return float("nan")
    losses = []
    for start in range(0, len(items), 512):
        chunk = items[start:start + 512]
        ids, lengths, targets, weights, good, bad, bad_mask = encode_batch(
            chunk, tokenizer, device)
        from ft.sft.losses import last_logprobs
        logp = last_logprobs(model, ids, lengths)
        losses.append(weighted_ce(logp, targets, weights).item())
    return sum(losses) / len(losses)
