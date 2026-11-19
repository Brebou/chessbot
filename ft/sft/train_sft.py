"""FT SFT trainer (spec sections 6-8).

Differences from the Phase-1 root ``train_sft.py``:

- data from the ft pipeline (Phases A-E, ``build_bucket_dataset`` output),
  game-level 97/3 split, stratified 45/20/15/15/5 batches (ft.sft.dataset);
- loss = weighted CE + ``lambda`` * bounded softplus contrastive (ft.sft.losses);
- checkpoint selection is PAIRED: candidate vs base on identical engine games
  (same levels/openings/seeds/colors), plus the paired selfplay proxy; a new
  best needs ``balanced_delta - 2*sigma > 0`` (ft/eval/paired_eval.py);
- stop criteria: (a) balanced plateau below ``plateau_delta`` for
  ``plateau_patience`` consecutive checkpoints; (b) fixture end-bin top1
  legality regressing > 5 pts vs base -> abort; (c) the ``normal``-bucket
  val CE rising > 5% between checkpoints -> rollback to the best weights and
  drop the lr to ``rollback_lr``;
- all diagnostics run on the fixed fixture set (ft.eval.metrics); only the
  best ``max_checkpoints`` dirs survive on disk.

Usage:
    uv run ft/sft/train_sft.py --model-path base_model --mix data/ft/mix.jsonl \
        --out-dir results/ft/sft1 --device cuda --eval-every-steps 2000 \
        --fixtures-dir data/ft_fixtures --eval-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chess_hw import ChessTokenizer
from chess_hw.model import load_model, save_model
from ft.eval.metrics import build_fixtures, fixture_metrics
from ft.eval.paired_eval import (balanced_delta, paired_engine_eval,
                                 paired_stop_criterion, selfplay_paired)
from ft.sft.dataset import encode_batch, load_items, stratified_batches, val_loss
from ft.sft.losses import BETA, LAMBDA, ft_loss

LEGALITY_REGRESSION_FLOOR = 0.05
CE_REGRESSION = 0.05


def _add_dataclass_args(cls, parser: argparse.ArgumentParser) -> None:
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


def _from_args_values(cls, args: argparse.Namespace) -> dict:
    kwargs = {}
    for f in fields(cls):
        val = getattr(args, f.name)
        if isinstance(f.default, tuple):
            val = tuple(val)
        kwargs[f.name] = val
    return kwargs


@dataclass
class FTSFTConfig:
    model_path: str = "base_model"
    base_path: str = ""                # reference model for the paired eval
    mix_path: str = "data/ft/mix.jsonl"
    out_dir: str = "results/ft/sft1"
    device: str = "auto"
    seed: int = 0
    epochs: int = 1
    batch_size: int = 512
    lr: float = 1e-4                   # AdamW, wd 0.01, clip 1.0 (spec section 7)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lambda_contrast: float = LAMBDA
    beta: float = BETA
    kl_lambda: float = 0.0
    log_every: int = 20
    rollback_lr: float = 5e-5          # lr after a `normal`-bucket CE regression
    # paired evaluation
    eval_mode: str = "paired"          # "paired" | "none"
    eval_every_steps: int = 2000       # 0 = epoch end only
    eval_levels: Tuple[int, ...] = (0, 1, 2, 3)
    eval_n_openings: int = 40
    selfplay_games: int = 200
    eval_workers: int = 16
    eval_bootstrap: int = 2000
    # fixtures diagnostics
    fixtures_dir: str = ""
    fixtures_games: str = ""     # Phase A games jsonl; real positions for the
                                 # per-bin fixtures (random playouts can't fill end)
    n_fixture_mate1: int = 500
    n_fixture_per_bin: int = 500
    # stop / retention
    plateau_delta: float = 0.02
    plateau_patience: int = 2
    max_checkpoints: int = 5           # keep only the best-5 checkpoint dirs

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        _add_dataclass_args(cls, parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FTSFTConfig":
        return cls(**_from_args_values(cls, args))


def legality_aborted(base_fixtures: Optional[dict], cand_fixtures: Optional[dict],
                     floor: float = LEGALITY_REGRESSION_FLOOR) -> Optional[str]:
    """Abort reason when the fixture end-bin top1 legality regressed."""
    if not base_fixtures or not cand_fixtures:
        return None
    base = base_fixtures.get("top1_legal_end")
    cand = cand_fixtures.get("top1_legal_end")
    if base is None or cand is None:
        return None
    if cand < base - floor:
        return (f"legality regression: end-bin top1-legal {cand:.3f} vs base "
                f"{base:.3f} (floor -{floor:.2f})")
    return None


def ce_regressed(prev: Optional[float], cur: Optional[float],
                 rel: float = CE_REGRESSION) -> bool:
    return (prev is not None and cur is not None and prev > 0
            and cur > prev * (1 + rel))


class FTSFTTrainer:
    def __init__(self, cfg: FTSFTConfig):
        self.cfg = cfg
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if cfg.device == "auto" else cfg.device
        torch.manual_seed(cfg.seed)
        self.run_dir = Path(cfg.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

        self.tokenizer = ChessTokenizer()
        self.model = load_model(cfg.model_path, device=self.device)
        self.ref_model = None
        if cfg.kl_lambda > 0:
            ref_path = cfg.base_path or cfg.model_path
            self.ref_model = load_model(ref_path, device=self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                           weight_decay=cfg.weight_decay)
        if not Path(cfg.mix_path).exists():
            sys.exit(f"❌ mix not found: {cfg.mix_path} - build it first with "
                     f"`uv run ft/pipeline.py --dataset <lichess dump> ...`")
        self.train_items, self.val_items = load_items(cfg.mix_path, self.tokenizer)
        buckets: Dict[str, int] = {}
        for it in self.train_items:
            buckets[it.bucket] = buckets.get(it.bucket, 0) + 1
        print(f"mix: {len(self.train_items)} train / {len(self.val_items)} val; "
              f"buckets {buckets}")
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict] = None
        self.prev_normal_ce: Optional[float] = None
        self.balanced_history: List[float] = []
        self._records_cache: Dict[str, list] = {}   # base engine records reused
        self._base_fixtures: Optional[dict] = None
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.ckpt_root = self.run_dir / "checkpoints"

    # -- persistence ---------------------------------------------------------

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

    def _snapshot(self) -> Dict:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _metric_rows(self) -> List[dict]:
        if not self.metrics_path.exists():
            return []
        return [json.loads(line) for line in
                self.metrics_path.read_text().splitlines() if line.strip()]

    def _prune_checkpoints(self) -> None:
        """Keep only the best-`max_checkpoints` dirs by balanced delta."""
        cfg = self.cfg
        if not self.ckpt_root.is_dir():
            return
        scored = [(m["balanced"], str(m["eval_tag"]).replace("/", "-"))
                  for m in self._metric_rows()
                  if m.get("balanced") is not None
                  and (self.ckpt_root / str(m["eval_tag"]).replace("/", "-")).is_dir()]
        if scored:
            keep = {tag for _, tag in sorted(scored, reverse=True)[:cfg.max_checkpoints]}
        else:
            keep = {d.name for d in self.ckpt_root.iterdir() if d.is_dir()}
        for d in sorted(self.ckpt_root.iterdir()):
            if d.is_dir() and d.name not in keep:
                shutil.rmtree(d, ignore_errors=True)

    # -- evaluation ----------------------------------------------------------

    def _fixtures(self, model_dir: Path) -> Optional[dict]:
        if not self.cfg.fixtures_dir:
            return None
        build_fixtures(self.cfg.fixtures_dir, n_mate1=self.cfg.n_fixture_mate1,
                       n_per_bin=self.cfg.n_fixture_per_bin, seed=self.cfg.seed,
                       games_path=self.cfg.fixtures_games)
        return fixture_metrics(str(model_dir), self.cfg.fixtures_dir,
                               device=self.device)

    def _base_fixture_metrics(self) -> Optional[dict]:
        if not self.cfg.fixtures_dir:
            return None
        cache = self.run_dir / "base-fixtures.json"
        if cache.exists():
            with open(cache) as f:
                return json.load(f)
        metrics = fixture_metrics(self.cfg.base_path or self.cfg.model_path,
                                  self.cfg.fixtures_dir, device=self.device)
        tmp = self.run_dir / "base-fixtures.tmp.json"
        with open(tmp, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        os.replace(tmp, cache)
        return metrics

    def _checkpoint_eval(self, tag: str) -> dict:
        cfg = self.cfg
        tag = tag.replace("/", "-")
        cand = self._save(self.ckpt_root / tag)
        row: Dict = {"eval_tag": tag}
        if cfg.eval_mode == "paired":
            base_path = cfg.base_path or cfg.model_path
            t0 = time.monotonic()
            paired = paired_engine_eval(
                str(cand), base_path, levels=cfg.eval_levels,
                n_openings=cfg.eval_n_openings, seed=cfg.seed,
                workers=cfg.eval_workers, n_bootstrap=cfg.eval_bootstrap,
                records_cache=self._records_cache)
            row["engine_seconds"] = time.monotonic() - t0
            for key in list(self._records_cache):
                if key != base_path:
                    del self._records_cache[key]
            t0 = time.monotonic()
            proxy = selfplay_paired(str(cand), base_path,
                                    n_games=cfg.selfplay_games, seed=cfg.seed,
                                    workers=cfg.eval_workers)
            row["proxy_seconds"] = time.monotonic() - t0
            row["engine_delta"] = paired["delta"]
            row["engine_sigma"] = paired["sigma"]
            row["engine_per_level_delta"] = paired["per_level_delta"]
            row["engine_cand_score"] = paired["cand_score"]
            row["engine_base_score"] = paired["base_score"]
            row["engine_per_level_cand"] = paired["per_level_cand"]
            row["engine_per_level_base"] = paired["per_level_base"]
            row["proxy_delta"] = proxy["proxy_delta"]
            row["balanced"] = balanced_delta(paired, proxy)
            row["significant"] = row["balanced"] - 2 * paired["sigma"] > 0
        cand_fixtures = self._fixtures(cand)
        if cand_fixtures is not None:
            if self._base_fixtures is None and cfg.eval_mode == "paired":
                self._base_fixtures = self._base_fixture_metrics()
            row["fixtures"] = cand_fixtures
        return row

    def _eval_detail_lines(self, row: dict) -> List[str]:
        lines: List[str] = []
        pl = row.get("engine_per_level_delta") or {}
        if pl:
            delta_txt = "  ".join(
                f"L{lv} {pl[lv]:+.4f}" for lv in sorted(pl, key=int))
            lines.append(f"  delta/level: {delta_txt}")
        cs = row.get("engine_per_level_cand") or {}
        bs = row.get("engine_per_level_base") or {}
        official = [lv for lv in ("1", "2", "3") if lv in cs and lv in bs]
        if official:
            cand = sum(cs[lv] for lv in official) / len(official)
            base = sum(bs[lv] for lv in official) / len(official)
            lines.append(f"  official (S1+S2+S3)/3: cand={cand:.4f} "
                         f"base={base:.4f} delta={cand - base:+.4f}")
        if row.get("engine_sigma") is not None:
            lines.append(f"  engine sigma={row['engine_sigma']:.4f} "
                         f"qualifies (delta - 2*sigma > 0): "
                         f"{row.get('significant')}")
        eng_s = row.get("engine_seconds")
        proxy_s = row.get("proxy_seconds")
        if eng_s is not None or proxy_s is not None:
            lines.append(f"  eval timing: engine {eng_s or 0:.0f}s | "
                         f"proxy {proxy_s or 0:.0f}s")
        fx = row.get("fixtures")
        if fx:
            lines.append(
                "  fixtures: mate1={:.1%} | won-cap end/late={:.1%}/{:.1%} | "
                "lost-cap end/late={:.1%}/{:.1%} | legal end/mid/late="
                "{:.0%}/{:.0%}/{:.0%}".format(
                    fx.get("mate1_top1_rate", 0.0),
                    fx.get("winning_capture_rate_end", 0.0),
                    fx.get("winning_capture_rate_late", 0.0),
                    fx.get("losing_capture_rate_end", 0.0),
                    fx.get("losing_capture_rate_late", 0.0),
                    fx.get("top1_legal_end", 0.0),
                    fx.get("top1_legal_mid", 0.0),
                    fx.get("top1_legal_late", 0.0)))
        return lines

    # -- training ------------------------------------------------------------

    def train(self) -> dict:
        cfg = self.cfg
        started = time.monotonic()
        step = 0
        abort: Optional[str] = None
        self.model.train()
        for epoch in range(1, cfg.epochs + 1):
            for batch in stratified_batches(self.train_items, cfg.batch_size,
                                            cfg.seed, epoch):
                ids, lengths, targets, weights, good, bad, bad_mask = encode_batch(
                    batch, self.tokenizer, self.device)
                loss, parts = ft_loss(self.model, ids, lengths, targets, weights,
                                      good, bad, bad_mask, beta=cfg.beta,
                                      lambda_contrast=cfg.lambda_contrast,
                                      ref_model=self.ref_model,
                                      lambda_kl=cfg.kl_lambda)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                step += 1
                if step % cfg.log_every == 0:
                    print(f"\repoch {epoch} step {step}: ce={parts['ce']:.4f} "
                          f"contra={parts.get('contrastive', 0.0):.4f} "
                          f"kl={parts.get('kl', 0.0):.4f}",
                          end="", flush=True)
                if cfg.eval_every_steps > 0 and step % cfg.eval_every_steps == 0:
                    row = {"step": step, "epoch": epoch,
                           "elapsed": time.monotonic() - started,
                           "ce": parts["ce"]}
                    row.update(self._checkpoint_eval(f"step-{step}"))
                    abort = self._finish_eval(row)
                if abort:
                    break
            if abort:
                break
            row = {"epoch": epoch, "step": step,
                   "elapsed": time.monotonic() - started}
            row["val_ce"] = val_loss(self.model, self.val_items, self.tokenizer,
                                     self.device)
            for bucket in ("normal", "mate", "endgame_tb", "contrastive"):
                row[f"val_ce_{bucket}"] = val_loss(self.model, self.val_items,
                                                   self.tokenizer, self.device,
                                                   bucket=bucket)
            row.update(self._checkpoint_eval(f"epoch-{epoch}"))
            abort = self._finish_eval(row)
            print(f"\nepoch {epoch}: val_ce={row['val_ce']:.4f} "
                  f"({row['elapsed']:.0f}s)", flush=True)

        final = self._save(self.run_dir / "final")
        summary = {"best_score": self.best_score, "final_model": str(final),
                   "steps": step, "abort": abort}
        tmp = self.run_dir / "summary.tmp.json"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        os.replace(tmp, self.run_dir / "summary.json")
        return summary

    def _finish_eval(self, row: dict) -> Optional[str]:
        cfg = self.cfg
        balanced = row.get("balanced")
        abort = legality_aborted(self._base_fixtures, row.get("fixtures"))
        normal_ce = row.get("val_ce_normal")
        if ce_regressed(self.prev_normal_ce, normal_ce) and self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            for g in self.optimizer.param_groups:
                g["lr"] = cfg.rollback_lr
            row["rollback"] = True
            print(f"\n\u26a0\ufe0f normal-bucket CE regression "
                  f"({self.prev_normal_ce:.4f} -> {normal_ce:.4f}): "
                  f"rollback + lr={cfg.rollback_lr}", flush=True)
        self.prev_normal_ce = normal_ce
        if balanced is not None:
            self.balanced_history.append(balanced)
            if row.get("significant") and \
                    (self.best_score is None or balanced > self.best_score):
                self.best_score = balanced
                self.best_state = self._snapshot()
                self._save(self.run_dir / "best")
                row["best"] = True
        self._log(row)
        self._prune_checkpoints()
        if abort is not None:
            print(f"\n\U0001f6d1 {abort}", flush=True)
            return abort
        if balanced is not None and paired_stop_criterion(
                self.balanced_history, cfg.plateau_delta, cfg.plateau_patience):
            print(f"\n\u23f9\ufe0f plateau: balanced < {cfg.plateau_delta} for "
                  f"{cfg.plateau_patience} checkpoints", flush=True)
            return f"plateau below {cfg.plateau_delta}"
        if balanced is not None:
            print(f"\neval [{row['eval_tag']}]: balanced={balanced:.4f} "
                  f"(engine {row['engine_delta']:.4f} / proxy "
                  f"{row['proxy_delta']:.4f})")
            for line in self._eval_detail_lines(row):
                print(line, flush=True)
        else:
            print(f"\neval [{row['eval_tag']}]: no paired eval "
                  f"(eval_mode={cfg.eval_mode})", flush=True)
        self.model.train()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FT SFT training (spec 6-8)")
    FTSFTConfig.add_cli_args(parser)
    args = parser.parse_args()
    summary = FTSFTTrainer(FTSFTConfig.from_args(args)).train()
    print("\nDone:", summary)


if __name__ == "__main__":
    main()
