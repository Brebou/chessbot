"""Autonomous sandbox sweep (run9): corpus-first protocol, CPU-only.

Protocol:
1. GEN: one annotated corpus from run7b-style mixed rollouts (self-play +
   model-vs-FastEngine games), every state scored by Stockfish at low depth
   (before/after, mover-POV) and classified with the run7 rules
   (drop > 150 cp / missed M1 / endgame mate conversion).
2. SWEEP: for each hypothesis the same corpus is re-mixed (corrections +
   optional rehearsal rows), trained, and evaluated with the official
   instruments on a FIXED eval seed: paired engine L1-3 (fixed cells,
   base records cached across the whole sweep) + model-vs-base proxy.
3. Diagnostics: top-1 agreement with base on held-out corpus states.

Every run appends a row to `ft/research/results.jsonl`.

Usage:
    uv run python -m ft.research.sweep \
        --model-path base_model/model.safetensors --out-dir results/auto9 \
        --n-selfplay 150 --n-engine 100 --sf-depth 4 --sf-workers 4 \
        --epochs 2 --eval-openings 10 --selfplay-games 20 --eval-workers 4 \
        --configs r1,r2,r3,r4,r5,r6 --stage all
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import chess

from ft.common.board_io import n_pieces_from_fen
from ft.common.serialize import jsonl_read, jsonl_write
from ft.sft.dagger import rollout_positions
from ft.sft.poc_loop import (build_corrections, engine_opponent_states,
                             poc_classify, poc_score)
from ft.sft.train_sft import FTSFTConfig, FTSFTTrainer

EVAL_LEVELS = (1, 2, 3)


# -- corpus -------------------------------------------------------------------

def build_corpus(*, model_path: str, out_dir: Path, n_selfplay: int,
                 n_engine: int, sf_depth: int, sf_workers: int, seed: int,
                 engine_path: str, batch_size: int,
                 max_plies: int = 250, progress=None) -> List[dict]:
    show = progress.show if progress is not None else (lambda m, force=False: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    show(f"corpus: {n_selfplay} self-play + {n_engine} engine-opponent games...")
    selfplay = rollout_positions(model_path, n_selfplay, stop_pieces=0,
                                 opening_plies=8, seed=seed, device="cpu",
                                 batch_size=batch_size, max_plies=max_plies,
                                 progress=progress)
    eng = engine_opponent_states(model_path, n_engine, opening_plies=8,
                                 seed=str(seed), engine_levels=[1, 2, 3],
                                 level_probs=[0.1, 0.3, 0.6],
                                 workers=sf_workers, max_plies=max_plies,
                                 progress=progress)
    merged: Dict[str, dict] = {}
    for origin, states in (("self", selfplay), ("engine", eng)):
        for s in states:
            if s["fen"] not in merged:
                s = dict(s)
                s["origin"] = origin
                merged[s["fen"]] = s
    states = list(merged.values())
    show(f"corpus: {len(states)} unique states, scoring at SF depth {sf_depth}...")
    scores = poc_score([(s["fen"], s["model_move"]) for s in states],
                       engine_path=engine_path, depth=sf_depth,
                       workers=sf_workers, progress=progress)
    n_mate = n_drop = n_m1miss = 0
    drops: List[float] = []
    for s in states:
        sc = scores.get(s["fen"], {})
        s["score"] = sc
        s["corrected"] = poc_classify(
            sc.get("cp_best"), sc.get("cp_model"), sc.get("mate_best"),
            sc.get("mate_model"), n_pieces_from_fen(s["fen"]))
        if s["corrected"]:
            cb, cm = sc.get("cp_best"), sc.get("cp_model")
            if cb is not None and cm is not None and cb - cm > 150.0:
                n_drop += 1
                drops.append(cb - cm)
            if sc.get("mate_best") is not None and sc.get("mate_best") > 0:
                n_mate += 1
    drops.sort()
    show(f"corpus: {sum(1 for s in states if s['corrected'])}/{len(states)} "
         f"corrected ({n_drop} by cp-drop, median {drops[len(drops)//2] if drops else 0:.0f} cp, "
         f"{n_mate} mate-critical); writing...", force=True)
    jsonl_write(out_dir / "corpus.jsonl", states)
    return states


# -- mixes ---------------------------------------------------------------------

def build_mix(corpus: List[dict], out_path: Path, *, rehearsal: float,
              seed: int) -> tuple:
    """Correction rows (run7 rules) + `rehearsal`-share self-replay rows
    (target = the model's own played move on non-blunder states)."""
    rows = build_corrections(
        corpus, {s["fen"]: s for s in corpus},
        cp_threshold=150.0, mate_piece_gate=10, quick_mate_plies=2,
        mate_slack_plies=1)
    corrected = {s["fen"] for s in corpus if s["corrected"]}
    if rehearsal > 0:
        pool = [s for s in corpus if not s["corrected"]]
        rng = random.Random(f"rehearsal-{seed}")
        rng.shuffle(pool)
        want = int(round(rehearsal * len(rows)))
        for s in pool[:want]:
            rows.append({"bucket": "normal", "key": s["fen"],
                         "start_fen": s["fen"], "history": s["history"],
                         "target": s["model_move"], "weight": 1.0})
    jsonl_write(out_path, rows)
    return len(rows), len(corrected)


# -- training + eval ------------------------------------------------------------

def train_candidate(mix_path: Path, *, model_path: str, base_path: str,
                    out_dir: Path, epochs: int, lr: float, kl: float,
                    batch: int, seed: int, eval_openings: int,
                    selfplay_games: int, eval_workers: int,
                    records_cache: Dict[str, list]) -> dict:
    cfg = FTSFTConfig(
        model_path=model_path, base_path=base_path, mix_path=str(mix_path),
        out_dir=str(out_dir), device="cpu", seed=seed, epochs=epochs,
        batch_size=batch, lr=lr, kl_lambda=kl, eval_mode="paired",
        eval_every_steps=0, eval_levels=EVAL_LEVELS,
        eval_n_openings=eval_openings, selfplay_games=selfplay_games,
        eval_workers=eval_workers, plateau_patience=999)
    trainer = FTSFTTrainer(cfg)
    trainer._records_cache = records_cache
    done = trainer.train()
    done["evals"] = trainer._metric_rows()
    return done


def agreement_metrics(model_path: str, base_path: str, sample: List[dict],
                      batch: int = 128) -> Dict[str, float]:
    """Top-1 agreement of candidate vs base, and vs the played move, on
    held-out states (simple greedy inference, no engine lookalike)."""
    from chess_hw import ChessTokenizer
    from chess_hw.model import load_model
    from diag.fast_probe import FastProbePlayer

    tok = ChessTokenizer()

    def top1(path: str) -> List[Optional[str]]:
        model = load_model(path, device="cpu")
        probe = FastProbePlayer(model, tok, batch_size=batch)
        out = probe.choose_from_histories(
            [s["history"] for s in sample],
            [chess.Board(s["fen"]) for s in sample])
        del model, probe
        return [c.move.uci() if c is not None and c.move is not None
                else None for c in out]

    cand = top1(model_path)
    base = top1(base_path)
    n = max(1, len(sample))
    same = sum(1 for c, b in zip(cand, base) if c is not None and c == b)
    played = sum(1 for s, c in zip(sample, cand)
                 if c is not None and c == s["model_move"])
    base_played = sum(1 for s, b in zip(sample, base)
                      if b is not None and b == s["model_move"])
    return {"agree_cand_base": same / n,
            "cand_repeats_played": played / n,
            "base_repeats_played": base_played / n}


# -- driver ---------------------------------------------------------------------

CONFIGS: Dict[str, dict] = {
    "r1": {"rehearsal": 0.0, "kl": 0.1, "lr": 5e-5},     # run7 recipe control
    "r2": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5},     # + rehearsal share
    "r3": {"rehearsal": 1.0, "kl": 0.0, "lr": 5e-5},     # KL off (isolate)
    "r4": {"rehearsal": 1.0, "kl": 0.1, "lr": 1e-4},     # hotter
    "r5": {"rehearsal": 1.0, "kl": 0.1, "lr": 3e-5},     # cooler
    "r6": {"rehearsal": 1.0, "kl": 0.3, "lr": 5e-5},     # stronger leash
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="base_model/model.safetensors")
    ap.add_argument("--out-dir", default="results/auto9")
    ap.add_argument("--n-selfplay", type=int, default=150)
    ap.add_argument("--n-engine", type=int, default=100)
    ap.add_argument("--sf-depth", type=int, default=4)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--selfplay-games", type=int, default=20)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="r1,r2,r3,r4,r5,r6")
    ap.add_argument("--stage", default="all", choices=["corpus", "sweep", "all"])
    args = ap.parse_args()
    show = (lambda m, force=False: print(m, flush=True))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    if args.stage in ("corpus", "all") and not corpus_path.exists():
        corpus = build_corpus(model_path=args.model_path, out_dir=out_dir,
                              n_selfplay=args.n_selfplay,
                              n_engine=args.n_engine, sf_depth=args.sf_depth,
                              sf_workers=args.sf_workers, seed=args.seed,
                              engine_path=args.engine,
                              batch_size=args.batch, progress=None)
        show = show
    else:
        corpus = list(jsonl_read(corpus_path))
        print(f"corpus: loaded {len(corpus)} states", flush=True)

    if args.stage in ("sweep", "all"):
        base_path = args.model_path
        records_cache: Dict[str, list] = {}
        holdout = random.Random(f"holdout-{args.seed}").sample(
            corpus, min(300, len(corpus)))
        results_path = Path("ft/research/results.jsonl")
        for name in args.configs.split(","):
            cfg = dict(CONFIGS[name])
            t0 = time.monotonic()
            mix_path = out_dir / f"mix_{name}.jsonl"
            n_rows, n_corr = build_mix(corpus, mix_path,
                                       rehearsal=cfg["rehearsal"],
                                       seed=args.seed)
            done = train_candidate(
                mix_path, model_path=args.model_path, base_path=base_path,
                out_dir=out_dir / name, epochs=args.epochs, lr=cfg["lr"],
                kl=cfg["kl"], batch=args.batch, seed=args.seed,
                eval_openings=args.eval_openings,
                selfplay_games=args.selfplay_games,
                eval_workers=args.eval_workers,
                records_cache=records_cache)
            last = done["evals"][-1] if done.get("evals") else {}
            agr = agreement_metrics(done.get("final_model", ""), base_path,
                                    holdout)
            row = {"run": f"auto9/{name}", "config": cfg,
                   "mix_rows": n_rows, "corrections": n_corr,
                   "steps": done.get("steps"), "best_score": done.get("best_score"),
                   "balanced": last.get("balanced"),
                   "engine_delta": last.get("engine_delta"),
                   "engine_per_level": last.get("engine_per_level_delta"),
                   "engine_sigma": last.get("engine_sigma"),
                   "proxy_delta": last.get("proxy_delta"),
                   "agreement": agr,
                   "elapsed_s": round(time.monotonic() - t0)}
            with open(results_path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"✅ {name}: balanced={row['balanced']} "
                  f"engine={row['engine_delta']} proxy={row['proxy_delta']} "
                  f"agree={agr['agree_cand_base']:.3f} "
                  f"({row['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
