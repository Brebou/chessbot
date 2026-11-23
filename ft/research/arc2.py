"""ARC2: book-aligned post-training research (independent from run9/arc1).

The official eval plays paired games from a bundled opening book
(``chess_hw.match.load_openings``, 31,369 entries) against a deterministic
alpha-beta+quiescence engine; every (opening, level, color) cell is a pure
function of the seed. Previous corpora sampled random opening plies — a
distribution mismatch. ARC2 generates on-policy corrections on the REAL
book openings, against the REAL class engine, then applies the surviving
recipe (t5: corrections + 1:1 rehearsal + KL 0.1 + lr 5e-5). Stages:

- ``scan``: deterministic base-vs-engine score over a seeded book subset,
  per level — the offline eval landscape plus per-game state dump.
- ``corpus``: book-aligned state corpus, SF-scored, classified (run7 rules).
- ``sweep``: recipe arms on the book corpus (train + formal paired eval vs
  base on a disjoint eval seed), rows appended to ft/research/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional

import chess

from ft.common.board_io import n_pieces_from_fen
from ft.common.serialize import jsonl_write
from ft.research.sweep import EVAL_LEVELS, base_move_rank, train_candidate
from ft.sft.poc_loop import build_corrections, poc_classify, poc_score

MAX_PLY = 250


def _book_task(task: tuple) -> dict:
    """(model_path, oid, level) -> {'rec': record dict, 'states': [...]}."""
    model_path, oid, level = task
    from chess_hw.match import EnginePlayer, load_openings, play_game
    from diag.fast_probe import FastProbePlayer
    from chess_hw import ChessTokenizer
    from chess_hw.model import load_model

    model = load_model(model_path, device="cpu")
    probe = FastProbePlayer(model, ChessTokenizer(), batch_size=256)
    probe.name = "arc2"
    opening = load_openings()[oid]
    states: List[dict] = []
    results = []
    for white_is_model in (True, False):
        eng = EnginePlayer(level, seed=f"arc2-{oid}-{level}")
        if white_is_model:
            rec = play_game(probe, eng, opening, oid, max_plies=MAX_PLY)
        else:
            rec = play_game(eng, probe, opening, oid, max_plies=MAX_PLY)
        results.append(rec.result)
        board = chess.Board()
        for u in opening:
            board.push_uci(u)
        hist: List[str] = list(opening)
        start = len(opening)
        for i in range(start, len(rec.moves)):
            is_model_ply = ((i - start) % 2 == 0) == white_is_model
            if is_model_ply:
                states.append({"fen": board.fen(), "history": list(hist),
                               "model_move": rec.moves[i],
                               "oid": oid, "level": level})
            board.push_uci(rec.moves[i])
            hist.append(rec.moves[i])
    del model, probe
    return {"oid": oid, "level": level, "results": results,
            "states": states}


def collect_book_corpus(model_path: str, oids: List[int], levels: List[int],
                        workers: int = 1) -> tuple:
    """Sequential book-game collection (one model load, no pools — spawn
    Pool workers deadlocked under this env). Returns (raw states, cells)."""
    from chess_hw.match import EnginePlayer, load_openings, play_game
    from diag.fast_probe import FastProbePlayer
    from chess_hw import ChessTokenizer
    from chess_hw.model import load_model
    import sys

    model = load_model(model_path, device="cpu")
    probe = FastProbePlayer(model, ChessTokenizer(), batch_size=256)
    probe.name = "arc2"
    book = load_openings()
    raw: List[dict] = []
    cells: Dict[tuple, List[str]] = {}
    for n, oid in enumerate(oids):
        for lv in levels:
            res_stats = []
            for white_is_model in (True, False):
                eng = EnginePlayer(lv, seed=f"arc2-{oid}-{lv}")
                if white_is_model:
                    rec = play_game(probe, eng, book[oid], oid, max_plies=MAX_PLY)
                else:
                    rec = play_game(eng, probe, book[oid], oid, max_plies=MAX_PLY)
                res_stats.append(rec.result)
                board = chess.Board()
                for u in book[oid]:
                    board.push_uci(u)
                hist: List[str] = list(book[oid])
                start = len(book[oid])
                for i in range(start, len(rec.moves)):
                    is_model_ply = ((i - start) % 2 == 0) == white_is_model
                    if is_model_ply:
                        raw.append({"fen": board.fen(),
                                    "history": list(hist),
                                    "model_move": rec.moves[i],
                                    "oid": oid, "level": lv})
                    board.push_uci(rec.moves[i])
                    hist.append(rec.moves[i])
            cells[(oid, lv)] = res_stats
        if (n + 1) % 50 == 0:
            print(f"  scan {n+1}/{len(oids)} oids ({len(raw)} states)",
                  flush=True)
    return raw, cells


def scan_base(model_path: str, *, n_openings: int, seed: str, levels: List[int],
              workers: int, out_dir: Path) -> dict:
    oids = _select_oids(seed, n_openings)
    t0 = time.monotonic()
    raw, cells = collect_book_corpus(model_path, oids, levels, workers)
    per_level = {lv: [] for lv in levels}
    from chess_hw.match import score_for
    for (oid, lv), (w_res, b_res) in cells.items():
        per_level[lv].append((score_for(w_res, chess.WHITE)
                              + score_for(b_res, chess.BLACK)) / 2.0)
    summary = {str(lv): round(sum(v) / len(v), 4) for lv, v in per_level.items()}
    print(f"book scan ({len(cells)} cells, {time.monotonic()-t0:.0f}s): "
          f"mean score/level = {summary}", flush=True)
    jsonl_write(out_dir / "scan_raw.jsonl", raw)
    return {"per_level": summary, "n_cells": len(cells),
            "n_states": len(raw)}


def _select_oids(seed: str, n: int) -> List[int]:
    from chess_hw.match import load_openings
    book_n = len(load_openings())
    return random.Random(f"arc2-{seed}").sample(range(book_n),
                                                min(n, book_n))


# -- corpus + sweep ------------------------------------------------------------

def build_book_corpus(*, model_path: str, out_dir: Path, n_openings: int,
                      seed: str, levels: List[int], sf_depth: int,
                      sf_workers: int, workers: int,
                      engine_path: str) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    oids = _select_oids(seed, n_openings)
    print(f"corpus: {len(oids)} book openings x {levels} x both colors "
          f"({len(oids)*len(levels)*2} games)...", flush=True)
    t0 = time.monotonic()
    raw, cells = collect_book_corpus(model_path, oids, levels, workers)
    merged: Dict[str, dict] = {}
    for s in raw:
        if s["fen"] not in merged:
            merged[s["fen"]] = s
    states = list(merged.values())
    print(f"corpus: {len(states)} unique states from {len(cells)} games "
          f"({time.monotonic()-t0:.0f}s); scoring at SF depth {sf_depth}...",
          flush=True)
    from ft.research.sweep import resolve_engine
    scores = poc_score([(s["fen"], s["model_move"]) for s in states],
                       engine_path=resolve_engine(engine_path),
                       depth=sf_depth, workers=sf_workers)
    n_corr = 0
    for s in states:
        sc = scores.get(s["fen"], {})
        s["score"] = sc
        s["corrected"] = poc_classify(
            sc.get("cp_best"), sc.get("cp_model"), sc.get("mate_best"),
            sc.get("mate_model"), n_pieces_from_fen(s["fen"]))
        n_corr += 1 if s["corrected"] else 0
    from chess_hw.match import score_for
    per_level = {lv: [] for lv in levels}
    for (oid, lv), (w_res, b_res) in cells.items():
        per_level[lv].append((score_for(w_res, chess.WHITE)
                              + score_for(b_res, chess.BLACK)) / 2.0)
    print("policy score per level: " + str(
        {lv: round(sum(v) / len(v), 4) for lv, v in per_level.items()}),
        flush=True)
    print(f"corpus: {n_corr}/{len(states)} corrected; writing...", flush=True)
    jsonl_write(out_dir / "corpus.jsonl", states)
    jsonl_write(out_dir / "scan_raw.jsonl", raw)
    return states


def train_and_eval(corpus: List[dict], *, name: str, out_dir: Path,
                   model_path: str, base_path: str, cfg: dict,
                   eval_openings: int, selfplay_games: int,
                   eval_workers: int, eval_seed: int,
                   records_cache: dict) -> dict:
    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired
    t0 = time.monotonic()
    mix_path = out_dir / f"mix_{name}.jsonl"
    n_rows, n_corr = build_mix(corpus, mix_path,
                               rehearsal=cfg["rehearsal"], seed=42,
                               cp_threshold=cfg.get("cp_threshold", 150.0),
                               corr_weight=cfg.get("corr_weight", 1.0),
                               mate_curric=cfg.get("mate_curric", 0.0),
                               distill=cfg.get("distill", 0.0))
    done = train_candidate(
        mix_path, model_path=model_path, base_path=base_path,
        out_dir=out_dir / name, epochs=cfg.get("epochs", 2),
        lr=cfg["lr"], kl=cfg["kl"], batch=cfg.get("batch", 512),
        seed=42, eval_openings=eval_openings,
        selfplay_games=selfplay_games, eval_workers=eval_workers,
        records_cache=records_cache, device=cfg.get("device", "cpu"))
    mpath = done.get("final_model", str(out_dir / name / "final" /
                                        "model.safetensors"))
    p = paired_engine_eval(mpath, base_path, levels=EVAL_LEVELS,
                           n_openings=eval_openings, seed=eval_seed,
                           workers=eval_workers, n_bootstrap=1000,
                           records_cache=records_cache)
    s = selfplay_paired(mpath, base_path, n_games=selfplay_games,
                        seed=eval_seed, workers=eval_workers)
    row = {"run": f"arc2/{name}", "config": cfg, "mix_rows": n_rows,
           "corrections": n_corr, "steps": done.get("steps"),
           "engine_delta": p["delta"], "sigma": p["sigma"],
           "per_level": p["per_level_delta"],
           "cand_score": p.get("cand_score"), "base_score": p.get("base_score"),
           "proxy_delta": s["proxy_delta"],
           "balanced": (p["delta"] + s["proxy_delta"]) / 2.0,
           "eval_seed": eval_seed, "eval_openings": eval_openings,
           "elapsed_s": round(time.monotonic() - t0)}
    with open(Path("ft/research/results.jsonl"), "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"done {name}: engine={row['engine_delta']:+.4f} "
          f"proxy={row['proxy_delta']:+.4f} "
          f"balanced={row['balanced']:+.4f} (sigma {row['sigma']:.4f})",
          flush=True)
    return row


CONFIGS = {
    "a1": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2},      # t5 on book
    "a2": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cp_threshold": 100.0},
    "a3": {"rehearsal": 1.0, "kl": 0.3, "lr": 1e-4, "epochs": 4},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="base_model/model.safetensors")
    ap.add_argument("--anchor-path", default=None)
    ap.add_argument("--out-dir", default="results/arc2")
    ap.add_argument("--n-openings", type=int, default=400)
    ap.add_argument("--scan-openings", type=int, default=200)
    ap.add_argument("--corpus-seed", type=str, default="book1")
    ap.add_argument("--scan-seed", type=str, default="scan1")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--sf-depth", type=int, default=6)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--selfplay-games", type=int, default=20)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--configs", default="a1,a2,a3")
    ap.add_argument("--stage", default="all",
                    choices=["scan", "corpus", "sweep", "all"])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "scan":
        scan_base(args.model_path, n_openings=args.scan_openings,
                  seed=args.scan_seed, levels=args.levels,
                  workers=args.workers, out_dir=out_dir)
    corpus_path = out_dir / "corpus.jsonl"
    if args.stage == "scan":
        return
    if args.stage in ("corpus", "all") and not corpus_path.exists():
        corpus = build_book_corpus(
            model_path=args.model_path, out_dir=out_dir,
            n_openings=args.n_openings, seed=args.corpus_seed,
            levels=args.levels, sf_depth=args.sf_depth,
            sf_workers=args.sf_workers, workers=args.workers,
            engine_path=args.engine)
    else:
        from ft.common.serialize import jsonl_read
        corpus = list(jsonl_read(corpus_path))
        print(f"corpus: loaded {len(corpus)} states", flush=True)
    if args.stage == "corpus":
        return
    base_path = args.anchor_path or args.model_path
    records_cache: dict = {}
    for name in args.configs.split(","):
        cfg = dict(CONFIGS[name])
        cfg["device"] = args.device
        train_and_eval(corpus, name=name, out_dir=out_dir,
                       model_path=args.model_path, base_path=base_path,
                       cfg=cfg, eval_openings=args.eval_openings,
                       selfplay_games=args.selfplay_games,
                       eval_workers=args.eval_workers,
                       eval_seed=args.eval_seed,
                       records_cache=records_cache)


if __name__ == "__main__":
    main()
