"""RL trainer: rollout → advantages → policy-gradient update → eval loop.

Per iteration:
1. sample ``games_per_iter`` games vs ``FastEngine`` (lockstep, batched GPU);
2. compute group-normalized advantages (GRPO) or a global baseline;
3. run ``inner_epochs`` over the transitions in ``minibatches``: recomputed
   log-probs, PPO/PG loss + KL penalty to the frozen reference + Lichess
   replay CE (anti-forgetting), gradient clipping, AdamW step;
4. every ``eval_every`` iterations, evaluate vs the engine on ``eval_seeds``
   (multi-seed, never tune on a single seed) and keep the best checkpoint.

All artifacts land in ``cfg.out_dir``: ``config.json``, ``metrics.jsonl``,
``checkpoints/iter-XXXXXX/model.safetensors``, ``best/model.safetensors``,
``final/model.safetensors``, ``summary.json``. Writes are atomic.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from chess_hw import ChessTokenizer
from chess_hw.match import evaluate_vs_engine
from chess_hw.model import load_model, save_model

from rl.config import RLConfig
from rl.losses import (histories_to_ids, kl_penalty_k3, last_logprobs,
                       policy_gradient_loss, sft_ce_loss)
from rl.replay import ReplaySampler
from rl.rewards import compute_advantages, rollout_stats
from rl.rollout import RolloutGame, generate_rollouts


class RLTrainer:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.run_dir = Path(cfg.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

        self.tokenizer = ChessTokenizer()
        self.model = load_model(cfg.model_path, device=self.device)
        self.reference = load_model(cfg.model_path, device=self.device)
        self.reference.requires_grad_(False)
        self.reference.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.replay = ReplaySampler(cfg.replay_path, self.tokenizer,
                                    batch_size=cfg.replay_batch_size,
                                    buffer_size=cfg.replay_buffer_size,
                                    seed=cfg.seed)
        if not self.replay.available and cfg.replay_coef > 0:
            print(f"⚠️ Replay corpus not found at {cfg.replay_path}: "
                  f"replay loss disabled")
        self.best_score: Optional[float] = None
        self.metrics_path = self.run_dir / "metrics.jsonl"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _write_config(self) -> None:
        path = self.run_dir / "config.json"
        tmp = self.run_dir / "config.tmp.json"
        with open(tmp, "w") as f:
            json.dump(asdict(self.cfg), f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def _log(self, row: dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _save(self, target: Path) -> Path:
        target.mkdir(parents=True, exist_ok=True)
        final = target / "model.safetensors"
        tmp = target / "model.tmp.safetensors"
        save_model(self.model, tmp)
        os.replace(tmp, final)
        return final

    # -- update ------------------------------------------------------------

    def _update(self, games: List[RolloutGame], advantages: List[List[float]],
                iteration: int) -> dict:
        cfg = self.cfg
        rng = random.Random(f"update-{cfg.seed}-{iteration}")
        flat: List[tuple] = []
        for game, advs in zip(games, advantages):
            for t, a in zip(game.transitions, advs):
                flat.append(([self.tokenizer.token_to_id[m] for m in t.history],
                             t.token_id, a, t.logprob))
        if not flat:
            return {"pg_loss": 0.0, "kl_loss": 0.0, "replay_loss": 0.0,
                    "entropy": 0.0, "transitions": 0}
        rng.shuffle(flat)

        n_mb = max(cfg.minibatches, 1)
        mb_size = (len(flat) + n_mb - 1) // n_mb
        totals = {"pg": 0.0, "kl": 0.0, "replay": 0.0, "entropy": 0.0}
        steps = 0
        pad, bos = self.tokenizer.pad_token_id, self.tokenizer.bos_token_id

        self.model.train()
        for _ in range(max(cfg.inner_epochs, 1)):
            for start in range(0, len(flat), mb_size):
                chunk = flat[start:start + mb_size]
                histories = [c[0] for c in chunk]
                actions = torch.tensor([c[1] for c in chunk], device=self.device)
                advs = torch.tensor([c[2] for c in chunk], dtype=torch.float32,
                                    device=self.device)
                old_lps = torch.tensor([c[3] for c in chunk], dtype=torch.float32,
                                       device=self.device)
                ids = torch.from_numpy(
                    histories_to_ids(histories, pad, bos)).to(self.device)
                lengths = torch.tensor([1 + len(h) for h in histories],
                                       device=self.device)

                new_lps, entropy = last_logprobs(self.model, ids, lengths, actions)
                loss = policy_gradient_loss(advs, new_lps, old_lps, cfg.ppo_clip)
                pg_val = loss.item()
                kl_val = 0.0
                if cfg.kl_coef > 0:
                    with torch.no_grad():
                        ref_lps, _ = last_logprobs(self.reference, ids, lengths,
                                                   actions)
                    kl = kl_penalty_k3(new_lps, ref_lps)
                    loss = loss + cfg.kl_coef * kl
                    kl_val = kl.item()
                ent_val = 0.0
                if cfg.entropy_bonus > 0:
                    loss = loss - cfg.entropy_bonus * entropy.mean()
                    ent_val = entropy.mean().item()
                rep_val = 0.0
                if cfg.replay_coef > 0 and self.replay.available:
                    batch = self.replay.next_batch()
                    if batch is not None and len(batch[0]):
                        r_ids = torch.from_numpy(batch[0]).to(self.device)
                        r_len = torch.from_numpy(
                            (batch[0] != self.tokenizer.pad_token_id)
                            .sum(axis=1)).to(self.device)
                        r_act = torch.from_numpy(batch[1]).to(self.device)
                        rl = sft_ce_loss(self.model, r_ids, r_len, r_act)
                        loss = loss + cfg.replay_coef * rl
                        rep_val = rl.item()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.grad_clip)
                self.optimizer.step()
                totals["pg"] += pg_val
                totals["kl"] += kl_val
                totals["replay"] += rep_val
                totals["entropy"] += ent_val
                steps += 1
        self.model.eval()
        return {
            "pg_loss": totals["pg"] / steps,
            "kl_loss": totals["kl"] / steps,
            "replay_loss": totals["replay"] / steps,
            "entropy": totals["entropy"] / steps,
            "transitions": len(flat),
        }

    # -- eval --------------------------------------------------------------

    def _evaluate(self, iteration: int) -> dict:
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
        row = {"eval_mean_score": sum(scores) / len(scores),
               "eval_per_seed": per_seed}
        if cfg.eval_vs_ref:
            row.update(self._evaluate_vs_ref(eval_dir))
        shutil.rmtree(eval_dir, ignore_errors=True)
        return row

    def _evaluate_vs_ref(self, eval_dir: Path) -> dict:
        """Elo proxy: score of the current policy against the frozen reference
        model, round-robin style (same openings protocol as the engine eval).
        Guards against engine over-fitting: if the engine score rises while
        this score falls, the mix is distorting play style."""
        import chess
        from chess_hw.match import _pair_task, run_tasks, score_for, select_openings
        cfg = self.cfg
        per_seed = {}
        for seed in cfg.eval_seeds:
            oids = select_openings(seed, cfg.eval_n_openings)
            tasks = [("cur", "ref", oid, a_white)
                     for oid in oids for a_white in (True, False)]
            recs = run_tasks(_pair_task, tasks,
                             {"cur": str(eval_dir), "ref": str(cfg.model_path)},
                             cfg.eval_workers, progress=False)
            pts = [score_for(r["result"],
                             chess.WHITE if r["meta"]["a_white"] else chess.BLACK)
                   for r in recs]
            per_seed[str(seed)] = sum(pts) / max(len(pts), 1)
        return {"eval_vs_ref_score": sum(per_seed.values()) / max(len(per_seed), 1),
                "eval_vs_ref_per_seed": per_seed}

    # -- main loop ---------------------------------------------------------

    def train(self) -> dict:
        cfg = self.cfg
        started = time.monotonic()
        for iteration in range(1, cfg.max_iters + 1):
            iter_started = time.monotonic()
            seed = f"{cfg.seed}-{iteration}"
            games = generate_rollouts(
                self.model, self.tokenizer, n_games=cfg.games_per_iter,
                levels=cfg.levels, group_size=cfg.group_size, seed=seed,
                temperature=cfg.temperature, batch_size=cfg.train_batch_size,
                max_plies=cfg.max_plies, engine_workers=cfg.engine_workers,
                selfplay_ratio=cfg.selfplay_ratio, device=self.device)
            advantages = compute_advantages(
                games, gamma=cfg.gamma, illegal_penalty=cfg.illegal_penalty,
                result_reward=cfg.result_reward,
                material_shaping=cfg.material_shaping, baseline=cfg.baseline,
                normalize=cfg.normalize_advantages)
            losses = self._update(games, advantages, iteration)

            row = {"iteration": iteration, "elapsed": time.monotonic() - started,
                   "iter_seconds": time.monotonic() - iter_started,
                   **rollout_stats(games), **losses}

            if cfg.save_every > 0 and iteration % cfg.save_every == 0:
                self._save(self.run_dir / "checkpoints" / f"iter-{iteration:06d}")

            if cfg.eval_every > 0 and iteration % cfg.eval_every == 0:
                eval_row = self._evaluate(iteration)
                row.update(eval_row)
                if cfg.best_criterion == "engine":
                    score = row["eval_mean_score"]
                elif cfg.best_criterion == "vs_ref":
                    score = row.get("eval_vs_ref_score", row["eval_mean_score"])
                else:  # balanced: mean of engine score and vs-ref score
                    parts = [row["eval_mean_score"]]
                    if "eval_vs_ref_score" in row:
                        parts.append(row["eval_vs_ref_score"])
                    score = sum(parts) / len(parts)
                row["best_criterion_score"] = score
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self._save(self.run_dir / "best")
                    row["best"] = True

            self._log(row)
            print(f"iter {iteration}: score={row.get('mean_score', 0):.3f} "
                  f"illegal={row['illegal_rate']:.3f} "
                  f"pg={row['pg_loss']:.4f} "
                  f"eval={row.get('eval_mean_score', float('nan')):.3f} "
                  f"vs_ref={row.get('eval_vs_ref_score', float('nan')):.3f} "
                  f"({row['iter_seconds']:.0f}s)", flush=True)

        final = self._save(self.run_dir / "final")
        summary = {"best_score": self.best_score, "final_model": str(final),
                   "iterations": cfg.max_iters}
        tmp = self.run_dir / "summary.tmp.json"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        os.replace(tmp, self.run_dir / "summary.json")
        return summary
