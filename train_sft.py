"""Phase 1 combined SFT training for ChessGPT.

Consumes the mix built by ``diag/train_data.py`` (kinds: replay, capture,
mate1, mate2, hang, endgame, promotion, distill) and trains the fixed
``ChessGPT`` architecture with additive losses:

- **weighted CE** on the target move at the last history position, with
  per-kind weights (``--w-capture``, ``--w-mate1``, ...) times the example's
  curriculum weight;
- **margin loss** on ``hang`` examples: ``-logsigmoid(beta * (logp(good) -
  logp(bad)))`` pushes probability away from material-losing alternatives.

Only last-position logits are computed (hidden states → gather → tied head),
so long histories stay cheap. Evaluation runs the official multi-seed engine
protocol; the best checkpoint is kept by mean multi-seed score. All writes
are atomic, everything is seeded, and `chess_hw/` is never modified.

Usage:
    uv run train_sft.py --mix data/sft_mix.jsonl --model base_model \
        --out results/sft/run1 --epochs 2 --lr 1e-4 --device cuda \
        --eval-every-steps 2000 --eval-seeds 0 1 --eval-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_hw import ChessTokenizer
from chess_hw.match import evaluate_vs_engine
from chess_hw.model import load_model, save_model


@dataclass
class SFTConfig:
    model_path: str = "base_model"
    mix_path: str = "data/sft_mix.jsonl"
    out_dir: str = "results/sft/run"
    device: str = "auto"
    seed: int = 0
    epochs: int = 2
    batch_size: int = 512
    lr: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.02
    log_every: int = 20
    # per-kind loss weights (tuned on multi-seed eval score, never on loss)
    w_replay: float = 1.0
    w_capture: float = 2.0
    w_mate1: float = 3.0
    w_mate2: float = 3.0
    w_hang: float = 2.0
    w_endgame: float = 2.0
    w_promotion: float = 2.0
    w_distill: float = 1.0
    # anti-hang margin loss
    margin_weight: float = 1.0
    margin_beta: float = 0.1
    # eval / checkpoints (eval at epoch end, plus every eval_every_steps)
    eval_every_steps: int = 0          # 0 = epoch end only
    eval_levels: Tuple[int, ...] = (0, 1, 2, 3)
    eval_seeds: Tuple[int, ...] = (0, 1)
    eval_n_openings: int = 25
    eval_workers: int = 8

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        for f in fields(cls):
            opt = "--" + f.name.replace("_", "-")
            default = f.default
            if isinstance(default, bool):
                parser.add_argument(opt, type=lambda s: s.lower() in ("1", "true", "yes"),
                                    default=default)
            elif isinstance(default, tuple):
                parser.add_argument(opt, nargs="+", type=int, default=list(default))
            elif isinstance(default, int) and not isinstance(default, bool):
                parser.add_argument(opt, type=int, default=default)
            elif isinstance(default, float):
                parser.add_argument(opt, type=float, default=default)
            else:
                parser.add_argument(opt, type=str, default=default)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SFTConfig":
        kwargs = {}
        for f in fields(cls):
            val = getattr(args, f.name)
            if isinstance(f.default, tuple):
                val = tuple(val)
            kwargs[f.name] = val
        return cls(**kwargs)


class SFTDataset:
    """Tokenized in-memory view of the mix, with length-sorted batching."""

    def __init__(self, rows: List[dict], tokenizer: ChessTokenizer, cfg: SFTConfig):
        self.cfg = cfg
        kind_w = {"replay": cfg.w_replay, "capture": cfg.w_capture,
                  "mate1": cfg.w_mate1, "mate2": cfg.w_mate2,
                  "hang": cfg.w_hang, "endgame": cfg.w_endgame,
                  "promotion": cfg.w_promotion, "distill": cfg.w_distill}
        tok = tokenizer.token_to_id
        self.items: List[dict] = []
        for r in rows:
            hist = [tok[m] for m in r["history"] if m in tok]
            if r["target"] not in tok or len(hist) > 255:
                continue
            self.items.append({
                "history": hist,
                "target": tok[r["target"]],
                "bad": tok[r["bad"]] if r.get("bad") in tok else None,
                "weight": kind_w.get(r["kind"], 1.0) * float(r.get("weight", 1.0)),
            })

    def __len__(self) -> int:
        return len(self.items)

    def batches(self, batch_size: int, seed: int, epoch: int) -> List[List[int]]:
        """Length-sorted batches (small padding), shuffled order per epoch."""
        rng = random.Random(f"batches-{seed}-{epoch}")
        order = list(range(len(self.items)))
        rng.shuffle(order)
        order.sort(key=lambda i: len(self.items[i]["history"]) + rng.random())
        return [order[s:s + batch_size] for s in range(0, len(order), batch_size)]


def encode_batch(items: List[dict], idxs: List[int], tokenizer: ChessTokenizer,
                 device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                       torch.Tensor, Optional[torch.Tensor],
                                       torch.Tensor]:
    tok_ids = tokenizer.token_to_id
    pad, bos = tokenizer.pad_token_id, tokenizer.bos_token_id
    sel = [items[i] for i in idxs]
    lengths = np.fromiter((1 + len(it["history"]) for it in sel), dtype=np.int64,
                          count=len(sel))
    arr = np.full((len(sel), int(lengths.max())), pad, dtype=np.int64)
    for r, it in enumerate(sel):
        arr[r, 0] = bos
        arr[r, 1:1 + len(it["history"])] = it["history"]
    targets = np.asarray([it["target"] for it in sel], dtype=np.int64)
    weights = np.asarray([it["weight"] for it in sel], dtype=np.float32)
    has_bad = [it["bad"] is not None for it in sel]
    if any(has_bad):
        bads = np.asarray([it["bad"] if it["bad"] is not None else it["target"]
                           for it in sel], dtype=np.int64)
        bad_t = torch.from_numpy(bads).to(device)
    else:
        bad_t = None
    return (torch.from_numpy(arr).to(device),
            torch.from_numpy(lengths).to(device),
            torch.from_numpy(targets).to(device),
            torch.from_numpy(weights).to(device),
            bad_t,
            torch.tensor(has_bad, device=device))


def sft_loss(model, batch_ids: torch.Tensor, lengths: torch.Tensor,
             targets: torch.Tensor, weights: torch.Tensor,
             bads: Optional[torch.Tensor], bad_mask: torch.Tensor,
             margin_beta: float, margin_weight: float) -> Tuple[torch.Tensor, dict]:
    hidden = model.hidden_states(batch_ids)
    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
    logits = F.linear(last, model.wte.weight).float()
    logp = torch.log_softmax(logits, dim=-1)
    ce = -logp.gather(1, targets.unsqueeze(1))[:, 0]
    ce_loss = (weights * ce).sum() / weights.sum().clamp(min=1e-8)
    parts = {"ce": ce_loss.item()}
    loss = ce_loss
    if bads is not None and bad_mask.any():
        lp_good = -ce
        lp_bad = logp.gather(1, bads.unsqueeze(1))[:, 0]
        margin = -F.logsigmoid(margin_beta * (lp_good - lp_bad))[bad_mask].mean()
        loss = loss + margin_weight * margin
        parts["margin"] = margin.item()
    return loss, parts


class SFTTrainer:
    def __init__(self, cfg: SFTConfig):
        self.cfg = cfg
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if cfg.device == "auto" else cfg.device
        torch.manual_seed(cfg.seed)
        self.run_dir = Path(cfg.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

        self.tokenizer = ChessTokenizer()
        self.model = load_model(cfg.model_path, device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        with open(cfg.mix_path, "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rng = random.Random(f"split-{cfg.seed}")
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * cfg.val_fraction)) if len(rows) > 50 else 0
        self.val_set = SFTDataset(rows[:n_val], self.tokenizer, cfg)
        self.train_set = SFTDataset(rows[n_val:], self.tokenizer, cfg)
        print(f"mix: {len(self.train_set)} train / {len(self.val_set)} val examples")
        self.best_score: Optional[float] = None
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def _write_config(self) -> None:
        tmp = self.run_dir / "config.tmp.json"
        with open(tmp, "w") as f:
            json.dump(asdict(self.cfg), f, indent=2, sort_keys=True)
        os.replace(tmp, self.run_dir / "config.json")

    def _log(self, row: dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _save(self, target: Path) -> Path:
        target.mkdir(parents=True, exist_ok=True)
        tmp = target / "model.tmp.safetensors"
        save_model(self.model, tmp)
        final = target / "model.safetensors"
        os.replace(tmp, final)
        return final

    def _val_loss(self) -> float:
        if len(self.val_set) == 0:
            return float("nan")
        self.model.eval()
        cfg = self.cfg
        total, wsum = 0.0, 0.0
        with torch.no_grad():
            for idxs in self.val_set.batches(cfg.batch_size, cfg.seed, 0):
                arr, lengths, targets, weights, bads, bad_mask = encode_batch(
                    self.val_set.items, idxs, self.tokenizer, self.device)
                loss, _ = sft_loss(self.model, arr, lengths, targets, weights,
                                   bads, bad_mask, cfg.margin_beta, 0.0)
                total += loss.item() * len(idxs)
                wsum += len(idxs)
        self.model.train()
        return total / max(wsum, 1)

    def _evaluate(self, tag: str) -> dict:
        cfg = self.cfg
        eval_dir = self.run_dir / "eval-tmp"
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        self._save(eval_dir)
        scores = []
        per_seed = {}
        for seed in cfg.eval_seeds:
            summary = evaluate_vs_engine(
                model_path=str(eval_dir), levels=list(cfg.eval_levels),
                n_openings=cfg.eval_n_openings, seed=seed,
                workers=cfg.eval_workers, progress=False)
            per_seed[str(seed)] = summary["mean_score"]
            scores.append(summary["mean_score"])
        shutil.rmtree(eval_dir, ignore_errors=True)
        return {"eval_mean_score": sum(scores) / len(scores),
                "eval_per_seed": per_seed, "eval_tag": tag}

    def train(self) -> dict:
        cfg = self.cfg
        started = time.monotonic()
        step = 0
        self.model.train()
        for epoch in range(1, cfg.epochs + 1):
            batches = self.train_set.batches(cfg.batch_size, cfg.seed, epoch)
            for bi, idxs in enumerate(batches):
                arr, lengths, targets, weights, bads, bad_mask = encode_batch(
                    self.train_set.items, idxs, self.tokenizer, self.device)
                loss, parts = sft_loss(self.model, arr, lengths, targets,
                                       weights, bads, bad_mask,
                                       cfg.margin_beta, cfg.margin_weight)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.grad_clip)
                self.optimizer.step()
                step += 1
                if step % cfg.log_every == 0:
                    print(f"\repoch {epoch} step {step}: "
                          f"ce={parts.get('ce', 0):.4f} "
                          f"margin={parts.get('margin', 0):.4f}",
                          end="", flush=True)

                do_eval = (cfg.eval_every_steps > 0
                           and step % cfg.eval_every_steps == 0)
                if do_eval:
                    row = {"step": step, "epoch": epoch,
                           "elapsed": time.monotonic() - started,
                           "ce": parts.get("ce"), "margin": parts.get("margin"),
                           **self._evaluate(f"step-{step}")}
                    self._finish_eval(row)
            val = self._val_loss()
            row = {"epoch": epoch, "step": step, "val_ce": val,
                   "elapsed": time.monotonic() - started,
                   **self._evaluate(f"epoch-{epoch}")}
            self._finish_eval(row)
            print(f"\nepoch {epoch}: val_ce={val:.4f} "
                  f"eval={row['eval_mean_score']:.3f} "
                  f"({row['elapsed']:.0f}s)", flush=True)

        final = self._save(self.run_dir / "final")
        summary = {"best_score": self.best_score, "final_model": str(final),
                   "steps": step}
        tmp = self.run_dir / "summary.tmp.json"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        os.replace(tmp, self.run_dir / "summary.json")
        return summary

    def _finish_eval(self, row: dict) -> None:
        score = row["eval_mean_score"]
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self._save(self.run_dir / "best")
            row["best"] = True
        self._log(row)
        print(f"\neval [{row['eval_tag']}]: {score:.3f} "
              f"(per seed {row['eval_per_seed']})", flush=True)
        self.model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 SFT training")
    SFTConfig.add_cli_args(parser)
    args = parser.parse_args()
    summary = SFTTrainer(SFTConfig.from_args(args)).train()
    print("\nDone:", summary)


if __name__ == "__main__":
    main()
