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
from ft.research.sweep import (EVAL_LEVELS, base_move_rank, build_mix,
                               train_candidate)
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
    scores = poc_score([(s["fen"], s["model_move"]) for s in states],
                       engine_path=engine_path, depth=sf_depth,
                       workers=sf_workers)
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
    if cfg.get("impact_weight"):
        for s in corpus:
            sc = s.get("score") or {}
            cb, cm = sc.get("cp_best"), sc.get("cp_model")
            if cb is not None and cm is not None and s.get("corrected"):
                s["weight"] = 1.0 + min(1.0, (cb - cm) / 600.0)
    if cfg.get("level_gain"):
        apply_level_gain(corpus, cfg["level_gain"])
    rehearsal = cfg["rehearsal"]
    if cfg.get("rehearsal_levels"):
        # build_mix cannot restrict its rehearsal pool; do it here instead.
        rehearsal = 0.0
        n_rows_c, n_corr = build_mix(  # corrections only
            corpus, mix_path, rehearsal=0.0, seed=42,
            cp_threshold=cfg.get("cp_threshold", 150.0),
            corr_weight=cfg.get("corr_weight", 1.0),
            mate_curric=cfg.get("mate_curric", 0.0),
            distill=cfg.get("distill", 0.0), cap=cfg.get("cap", 0))
        from ft.common.serialize import jsonl_read
        rows = list(jsonl_read(mix_path))
        want = int(round(cfg["rehearsal"] * max(1, len(rows))))
        allowed = set(cfg["rehearsal_levels"].split(","))
        pool = [s for s in corpus if not s.get("corrected")
                and str(s.get("level")) in allowed]
        rng = random.Random(f"reh-lvl-{name}")
        rng.shuffle(pool)
        for s in pool[:want]:
            rows.append({"bucket": "normal", "key": s["fen"],
                         "start_fen": s["fen"], "history": s["history"],
                         "target": s["model_move"], "weight": 1.0})
        if cfg.get("cap") and len(rows) > cfg["cap"]:
            rng2 = random.Random(f"cap-{name}")
            rng2.shuffle(rows)
            rows = rows[:cfg["cap"]]
        jsonl_write(mix_path, rows)
        n_rows = len(rows)
    else:
        n_rows, n_corr = build_mix(  # corrections + rehearsal
            corpus, mix_path, rehearsal=rehearsal, seed=42,
            cp_threshold=cfg.get("cp_threshold", 150.0),
            corr_weight=cfg.get("corr_weight", 1.0),
            mate_curric=cfg.get("mate_curric", 0.0),
            distill=cfg.get("distill", 0.0),
            cap=cfg.get("cap", 0))
    if cfg.get("mirror"):
        from ft.common.serialize import jsonl_read
        rows = list(jsonl_read(mix_path))
        rows = mirror_rows(rows)
        jsonl_write(mix_path, rows)
        n_rows = len(rows)
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
    # -- batch s: fast small-capped arms for the open loop --
    "a1s": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000},
    "a2s": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000, "cp_threshold": 100.0},
    "a4s": {"rehearsal": 0.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000},
    "b1": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000, "impact_weight": True},
    # -- batch c: level targeting + mirror/color-symmetry augmentation --
    # L1 has the most headroom (base 0.167 vs 0.069/0.031): upweight its
    # corrections 3x.
    "c1": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000,
           "level_gain": {"1": 3.0}},
    # Mirror (color-symmetry) augmentation: every correction row is duplicated
    # on the vertically-mirrored board -> 2x gradient signal for free.
    "c2": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000, "mirror": True},
    # L1 corrections 2x + rehearsal restricted to L1 states only.
    "c3": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000,
           "level_gain": {"1": 2.0}, "rehearsal_levels": "1"},
    "c4": {"rehearsal": 1.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 16000,
           "level_gain": {"1": 2.0}, "mirror": True},
    # Extra rehearsal (2:1) — tests whether more non-blunder grounding helps.
    "c5": {"rehearsal": 2.0, "kl": 0.1, "lr": 5e-5, "epochs": 2, "cap": 12000},
}


# -- mirror / color-symmetry augmentation --------------------------------------

def _mirror_square(sq_uci: str) -> str:
    """Mirror one UCI move on the vertical flip (rank r -> 9-r).

    'e2e4' -> 'e7e5', 'a1h1' (castle short as white) -> 'a8h8', promotion
    letters survive: 'e7e8q' -> 'e2e1q'.
    """
    out = []
    for ch in sq_uci:
        if ch.isdigit():
            out.append(str(9 - int(ch)))
        else:
            out.append(ch)
    return "".join(out)


def _mirror_fen(fen: str) -> str:
    # Board.mirror() = vertical flip + color swap + side-to-move flip: the
    # exact color symmetry (same game, model takes the other color).
    return chess.Board(fen).mirror().fen()


def mirror_rows(rows: List[dict]) -> List[dict]:
    """Duplicate mix rows on the vertically-mirrored (color-swapped) board.

    The mirrored position of the same game state is exactly the position the
    corpus reaches when the model takes the OTHER color against the same
    engine — a pure symmetry of chess the tiny model has not learned. Rows
    whose mirrored target is not legal (should not happen) are skipped.
    """
    out = []
    for r in rows:
        out.append(r)
        try:
            board = chess.Board(_mirror_fen(r["start_fen"]))
        except Exception:
            continue
        m_target = _mirror_square(r["target"])
        try:
            mv = chess.Move.from_uci(m_target)
        except Exception:
            continue
        if mv not in board.legal_moves:
            continue
        m_hist = [_mirror_square(u) for u in r["history"]]
        out.append({"bucket": r["bucket"], "key": board.fen(),
                    "start_fen": board.fen(), "history": m_hist,
                    "target": m_target, "weight": r.get("weight", 1.0)})
    return out


def apply_level_gain(corpus: List[dict], gain: Dict[str, float]) -> None:
    """Multiply per-state weight by level gain on corrected states only."""
    for s in corpus:
        if not s.get("corrected"):
            continue
        g = gain.get(str(s.get("level")))
        if g is None:
            g = gain.get("*")
        if g is None:
            g = gain.get("other", 1.0)
        s["weight"] = s.get("weight", 1.0) * g
SAFETY_MIX = "results/arc2/mix_safety.jsonl"


# -- REINFORCE-lite arm: outcome-weighted self-play --------------------------

def collect_outcome_games(model_path: str, oids: List[int], levels: List[int],
                          *, seed_tag: str = "rein", max_plies: int = MAX_PLY,
                          device: str = "cpu",
                          ) -> List[dict]:
    """Play the CURRENT policy vs the class engine (both colors) and keep
    per-model-move states WITH the game outcome, so rows can be weighted by
    winning/losing (outcome-selected supervision). Games are fresh each call:
    never reused across arms/iterations."""
    from chess_hw.match import EnginePlayer, load_openings, play_game
    from diag.fast_probe import FastProbePlayer
    from chess_hw import ChessTokenizer
    from chess_hw.model import load_model

    model = load_model(model_path, device=device)
    probe = FastProbePlayer(model, ChessTokenizer(), batch_size=256)
    probe.name = "arc2-rein"
    book = load_openings()
    games: List[dict] = []
    for n, oid in enumerate(oids):
        for lv in levels:
            for white_is_model in (True, False):
                eng = EnginePlayer(lv, seed=f"{seed_tag}-{oid}-{lv}")
                if white_is_model:
                    rec = play_game(probe, eng, book[oid], oid, max_plies=max_plies)
                else:
                    rec = play_game(eng, probe, book[oid], oid, max_plies=max_plies)
                board = chess.Board()
                for u in book[oid]:
                    board.push_uci(u)
                hist = list(book[oid])
                start = len(book[oid])
                mstates = []
                for i in range(start, len(rec.moves)):
                    is_model_ply = ((i - start) % 2 == 0) == white_is_model
                    if is_model_ply:
                        mstates.append({"fen": board.fen(),
                                        "history": list(hist),
                                        "model_move": rec.moves[i]})
                    board.push_uci(rec.moves[i])
                    hist.append(rec.moves[i])
                # result POV of the model (score 1/0.5/0)
                from chess_hw.match import score_for
                sc = score_for(rec.result, chess.WHITE if white_is_model
                               else chess.BLACK)
                games.append({"oid": oid, "level": lv,
                              "out": sc, "states": mstates})
        if (n + 1) % 50 == 0:
            print(f"  rein games {n+1}/{len(oids)} ({len(games)} games)",
                  flush=True)
    del model, probe
    return games


def build_trajectory_rows(games: List[dict], rows: List[dict], scores: dict,
                          *, w_loss: float, tail_plies: int = 3,
                          drop_cp: float = 100.0,
                          existing_fens: Optional[set] = None) -> int:
    """Trajectory (blunder-path) corrections for losing games.

    Single-position CE only corrects the tipping move; the moves that SET UP
    the losing line still get their probability from the loss. For each
    losing game, find the largest one-ply eval drop among its model states;
    the up-to `tail_plies` model moves BEFORE the drop each get a correction
    row (target = SF best at that state, weight w_loss * 0.25). Nesting the
    corrections on the path gives a denser gradient on games that matter.
    """
    n_added = 0
    for g in games:
        if g["out"] != 0.0 or not g["states"]:
            continue
        states = g["states"]
        drops = []
        for i, s in enumerate(states):
            sc = scores.get(s["fen"], {})
            cb, cm = sc.get("cp_before"), sc.get("cp_after")
            if cb is None or cm is None:
                # derive drop from cp_best vs cp_model at this state
                cb, cm = sc.get("cp_best"), sc.get("cp_model")
            if cb is None or cm is None:
                continue
            drops.append((max(0.0, (cb or 0.0) - (cm or 0.0)), i))
        if not drops:
            continue
        drop, tip = max(drops)
        if drop < drop_cp:
            continue
        for i in range(max(0, tip - tail_plies), tip + 1):
            s = states[i]
            if existing_fens and s["fen"] in existing_fens:
                continue
            sc = scores.get(s["fen"], {})
            best = sc.get("best_uci")
            if not best or best == s["model_move"]:
                continue
            rows.append({"bucket": "normal", "key": s["fen"],
                         "start_fen": s["fen"], "history": s["history"],
                         "target": best, "weight": w_loss * 0.25})
            n_added += 1
    return n_added


def build_reinforce_mix(games: List[dict], out_path: Path, *,
                        engine_path: str, sf_depth: int, sf_workers: int,
                        w_win: float = 1.0, w_loss: float = 1.0,
                        w_draw: float = 0.3, cap: int = 12000,
                        trajectory: bool = False) -> tuple:
    """REINFORCE-lite rows: WINNING games -> CE rows pushing UP the model's
    own played moves (weight w_win); LOSING games -> CE rows to the SF best
    move when the played move was a blunder (weight w_loss), otherwise a
    low-weight own-move row; DRAWS -> own moves at w_draw. Outcome selection
    is the policy-gradient signal."""
    lose_states: List[dict] = []
    rows: List[dict] = []
    for g in games:
        w = {1.0: w_win, 0.5: w_draw, 0.0: w_loss}[g["out"]]
        if w <= 0:
            continue
        for s in g["states"]:
            if g["out"] == 0.0:
                lose_states.append(s)
                continue
            rows.append({"bucket": "normal", "key": s["fen"],
                         "start_fen": s["fen"], "history": s["history"],
                         "target": s["model_move"], "weight": w})
    if lose_states:
        from ft.sft.poc_loop import poc_classify, poc_score
        from ft.common.board_io import n_pieces_from_fen
        uniq = {}
        for s in lose_states:
            uniq.setdefault(s["fen"], s)
        print(f"rein: scoring {len(uniq)} unique losing states...", flush=True)
        scores = poc_score([(s["fen"], s["model_move"]) for s in uniq.values()],
                           engine_path=engine_path, depth=sf_depth,
                           workers=sf_workers)
        n_corr = 0
        for fen, s in uniq.items():
            sc = scores.get(fen, {})
            if not poc_classify(sc.get("cp_best"), sc.get("cp_model"),
                                sc.get("mate_best"), sc.get("mate_model"),
                                n_pieces_from_fen(fen)):
                # not a big blunder per se: low-weight own-move rehearsal
                rows.append({"bucket": "normal", "key": fen,
                             "start_fen": fen, "history": s["history"],
                             "target": s["model_move"],
                             "weight": w_loss * 0.25})
                continue
            best = sc.get("best_uci")
            rows.append({"bucket": "normal", "key": fen,
                         "start_fen": fen, "history": s["history"],
                         "target": best, "weight": w_loss})
            n_corr += 1
        print(f"rein: {n_corr} correction rows from {len(uniq)} losing states",
              flush=True)
        if trajectory:
            seen = {r["key"] for r in rows}
            n_traj = build_trajectory_rows(games, rows, scores,
                                           w_loss=w_loss,
                                           existing_fens=seen)
            print(f"rein: +{n_traj} trajectory (blunder-path) rows",
                  flush=True)
    rng = random.Random("rein-cap")
    rng.shuffle(rows)
    if cap and len(rows) > cap:
        rows = rows[:cap]
    jsonl_write(out_path, rows)
    return len(rows), games


def run_reinforce_arm(out_dir: Path, *, model_path: str, base_path: str,
                      cfg: dict, eval_openings: int, selfplay_games: int,
                      eval_workers: int, eval_seed: int, sf_depth: int,
                      sf_workers: int, engine_path: str,
                      records_cache: dict) -> dict:
    from ft.research.sweep import resolve_engine
    """One REINFORCE-lite iteration: fresh outcome-qualified games from the
    book (vs the class engine), outcome-weighted mix, t5-recipe fine-tune,
    formal paired eval vs base."""
    import json as _json
    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired
    name = cfg.get("name", "d1")
    t0 = time.monotonic()
    engine_path = resolve_engine(engine_path)
    oids = _select_oids(cfg.get("rein_seed", f"rein-{name}"),
                        cfg.get("games", 200))
    games = collect_outcome_games(model_path, oids,
                                  cfg.get("rein_levels", [1, 2]),
                                  device=cfg.get("rein_device", "cpu"))
    n_win = sum(1 for g in games if g["out"] == 1.0)
    n_loss = sum(1 for g in games if g["out"] == 0.0)
    print(f"rein games: {len(games)} total, {n_win} wins, {n_loss} losses, "
          f"{len(games) - n_win - n_loss} draws", flush=True)
    mix_path = out_dir / f"mix_{name}.jsonl"
    n_rows, _ = build_reinforce_mix(
        games, mix_path, engine_path=engine_path, sf_depth=sf_depth,
        sf_workers=sf_workers, w_win=cfg.get("w_win", 1.0),
        w_loss=cfg.get("w_loss", 1.0), w_draw=cfg.get("w_draw", 0.3),
        cap=cfg.get("cap", 12000), trajectory=cfg.get("trajectory", False))
    done = train_candidate(
        mix_path, model_path=model_path, base_path=base_path,
        out_dir=out_dir / name, epochs=cfg.get("epochs", 2), lr=cfg["lr"],
        kl=cfg["kl"], batch=cfg.get("batch", 128), seed=42,
        eval_openings=eval_openings, selfplay_games=selfplay_games,
        eval_workers=eval_workers, records_cache=records_cache,
        device=cfg.get("device", "cpu"))
    mpath = done.get("final_model", str(out_dir / name / "final" /
                                        "model.safetensors"))
    p = paired_engine_eval(mpath, base_path, levels=EVAL_LEVELS,
                           n_openings=eval_openings, seed=eval_seed,
                           workers=eval_workers, n_bootstrap=1000,
                           records_cache=records_cache)
    sl = selfplay_paired(mpath, base_path, n_games=selfplay_games,
                         seed=eval_seed, workers=eval_workers)
    row = {"run": f"arc2/{name}", "config": cfg, "mix_rows": n_rows,
           "n_win": n_win, "n_loss": n_loss,
           "engine_delta": p["delta"], "sigma": p["sigma"],
           "per_level": p["per_level_delta"],
           "proxy_delta": sl["proxy_delta"],
           "balanced": (p["delta"] + sl["proxy_delta"]) / 2.0,
           "eval_seed": eval_seed,
           "elapsed_s": round(time.monotonic() - t0)}
    with open(Path("ft/research/results.jsonl"), "a") as f:
        f.write(_json.dumps(row, sort_keys=True) + "\n")
    print(f"done {name}: engine={row['engine_delta']:+.4f} "
          f"proxy={row['proxy_delta']:+.4f} "
          f"balanced={row['balanced']:+.4f} (sigma {row['sigma']:.4f})",
          flush=True)
    return row


REINFORCE_CONFIGS = {
    "d1": {"games": 200, "rein_levels": [1, 2], "w_win": 1.0,
           "w_loss": 1.0, "w_draw": 0.3, "cap": 12000, "epochs": 2,
           "kl": 0.1, "lr": 5e-5},
    "d2": {"games": 200, "rein_levels": [1], "w_win": 1.0, "w_loss": 1.5,
           "w_draw": 0.0, "cap": 12000, "epochs": 2, "kl": 0.1, "lr": 5e-5},
    # -- batch f: sandbox dose-response on the d1 recipe --
    # f1: same recipe, 2x games (does the d-signal grow with games N?)
    "f1": {"games": 400, "rein_levels": [1, 2], "w_win": 1.0,
           "w_loss": 1.0, "w_draw": 0.3, "cap": 20000, "epochs": 1,
           "kl": 0.1, "lr": 5e-5},
    # f2: heavier loss weight (1.5x corrections on losing games)
    "f2": {"games": 400, "rein_levels": [1, 2], "w_win": 1.0,
           "w_loss": 1.5, "w_draw": 0.3, "cap": 20000, "epochs": 1,
           "kl": 0.1, "lr": 5e-5},
    # f3: pure corrections (no win/draw rows) — isolates the correction leg
    "f3": {"games": 400, "rein_levels": [1, 2], "w_win": 0.0,
           "w_loss": 1.0, "w_draw": 0.0, "cap": 20000, "epochs": 1,
           "kl": 0.1, "lr": 5e-5},
    # -- batch g: trajectory (blunder-path) corrections --
    # g1: f1 recipe + dense corrections along the losing path
    "g1": {"games": 400, "rein_levels": [1, 2], "w_win": 1.0,
           "w_loss": 1.0, "w_draw": 0.3, "cap": 20000, "epochs": 1,
           "kl": 0.1, "lr": 5e-5, "trajectory": True},
    # g2: trajectory-only corrections (even denser focus on lost games)
    "g2": {"games": 400, "rein_levels": [1, 2], "w_win": 0.0,
           "w_loss": 1.0, "w_draw": 0.0, "cap": 20000, "epochs": 1,
           "kl": 0.1, "lr": 5e-5, "trajectory": True},
}


# -- curriculum-ordering arm (batch e) ----------------------------------------

def run_curriculum_arm(corpus: List[dict], out_dir: Path, *, name: str,
                       model_path: str, base_path: str, cfg: dict,
                       eval_openings: int, selfplay_games: int,
                       eval_workers: int, eval_seed: int,
                       records_cache: dict) -> dict:
    """Two-stage ordering hypothesis, untested axis: stage 1 fixes errors
    (corrections only, 1 epoch), stage 2 re-grounds the surviving policy
    (own-move rehearsal from non-blunder states, 1 epoch). Compare against
    the same data mixed together (a1s family)."""
    import json as _json
    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired
    t0 = time.monotonic()
    cap = cfg.get("cap", 12000)
    corrected = {s["fen"] for s in corpus if s.get("corrected")}

    # stage 1: corrections only
    corr_states = [s for s in corpus if s.get("corrected")]
    mix1 = out_dir / f"mix_{name}_s1.jsonl"
    n_rows1, _ = build_mix(corr_states, mix1, rehearsal=0.0, seed=42,
                           cp_threshold=cfg.get("cp_threshold", 150.0),
                           cap=cap // 2)
    m1_dir = out_dir / f"{name}_s1"
    done1 = train_candidate(mix1, model_path=model_path, base_path=base_path,
                            out_dir=m1_dir, epochs=1, lr=cfg["lr"],
                            kl=cfg["kl"], batch=cfg.get("batch", 128),
                            seed=42, eval_openings=eval_openings,
                            selfplay_games=selfplay_games,
                            eval_workers=eval_workers,
                            records_cache={}, device=cfg.get("device", "cpu"))
    m1 = done1.get("final_model", str(m1_dir / "final" / "model.safetensors"))
    print(f"{name} stage1 done: corrections stage at {m1}", flush=True)

    # stage 2: own-move rehearsal on non-corrected states (re-grounding)
    pool = [s for s in corpus if s["fen"] not in corrected]
    rng = random.Random(f"curr-{name}")
    rng.shuffle(pool)
    rows = []
    for s in pool[:cap]:
        rows.append({"bucket": "normal", "key": s["fen"],
                     "start_fen": s["fen"], "history": s["history"],
                     "target": s["model_move"], "weight": 1.0})
    mix2 = out_dir / f"mix_{name}_s2.jsonl"
    jsonl_write(mix2, rows)
    m2_dir = out_dir / f"{name}_s2"
    done2 = train_candidate(mix2, model_path=m1, base_path=base_path,
                            out_dir=m2_dir, epochs=1, lr=cfg["lr"],
                            kl=cfg["kl"], batch=cfg.get("batch", 128),
                            seed=42, eval_openings=eval_openings,
                            selfplay_games=selfplay_games,
                            eval_workers=eval_workers,
                            records_cache=records_cache)
    mpath = done2.get("final_model", str(m2_dir / "final" / "model.safetensors"))
    p = paired_engine_eval(mpath, base_path, levels=EVAL_LEVELS,
                           n_openings=eval_openings, seed=eval_seed,
                           workers=eval_workers, n_bootstrap=1000,
                           records_cache=records_cache)
    sl = selfplay_paired(mpath, base_path, n_games=selfplay_games,
                         seed=eval_seed, workers=eval_workers)
    row = {"run": f"arc2/{name}", "config": cfg, "mix_rows": cap,
           "corrections": n_rows1, "steps": done2.get("steps"),
           "engine_delta": p["delta"], "sigma": p["sigma"],
           "per_level": p["per_level_delta"],
           "proxy_delta": sl["proxy_delta"],
           "balanced": (p["delta"] + sl["proxy_delta"]) / 2.0,
           "eval_seed": eval_seed,
           "elapsed_s": round(time.monotonic() - t0)}
    with open(Path("ft/research/results.jsonl"), "a") as f:
        f.write(_json.dumps(row, sort_keys=True) + chr(10))
    print(f"done {name}: engine={row['engine_delta']:+.4f} "
          f"proxy={row['proxy_delta']:+.4f} "
          f"balanced={row['balanced']:+.4f} (sigma {row['sigma']:.4f})",
          flush=True)
    return row


CURRICULUM_CONFIGS = {
    "e1": {"cap": 12000, "kl": 0.1, "lr": 5e-5},
}


def run_safety_arm(out_dir: Path, *, model_path: str, base_path: str,
                   eval_openings: int, selfplay_games: int,
                   eval_workers: int, eval_seed: int,
                   records_cache: dict) -> dict:
    """Train on the heuristic material-safety rows (+1:1 rehearsal from the
    book corpus) with the t5 recipe; full official eval on disjoint seed."""
    from ft.common.serialize import jsonl_read
    import json
    import time as _t
    rows = list(jsonl_read(Path(SAFETY_MIX)))
    corpus_path = out_dir / "corpus.jsonl"
    corpus = list(jsonl_read(corpus_path))
    n_reh = len(rows)
    pool = [s for s in corpus if not s.get("corrected")]
    rng = random.Random("safety-reh")
    rng.shuffle(pool)
    for s in pool[:n_reh]:
        rows.append({"bucket": "normal", "key": s["fen"],
                     "start_fen": s["fen"], "history": s["history"],
                     "target": s["model_move"], "weight": 1.0})
    mix_path = out_dir / "mix_s1.jsonl"
    jsonl_write(mix_path, rows)
    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired
    t0 = _t.monotonic()
    done = train_candidate(mix_path, model_path=model_path,
                           base_path=base_path, out_dir=out_dir / "s1",
                           epochs=2, lr=5e-5, kl=0.1, batch=128, seed=42,
                           eval_openings=eval_openings,
                           selfplay_games=selfplay_games,
                           eval_workers=eval_workers,
                           records_cache=records_cache, device="cpu")
    mpath = done.get("final_model", "")
    p = paired_engine_eval(mpath, base_path, levels=EVAL_LEVELS,
                           n_openings=eval_openings, seed=eval_seed,
                           workers=eval_workers, n_bootstrap=1000,
                           records_cache=records_cache)
    sl = selfplay_paired(mpath, base_path, n_games=selfplay_games,
                         seed=eval_seed, workers=eval_workers)
    row = {"run": "arc2/s1-safety", "config": {"rows": len(rows),
                                               "safety_rows": n_reh},
           "engine_delta": p["delta"], "sigma": p["sigma"],
           "per_level": p["per_level_delta"],
           "proxy_delta": sl["proxy_delta"],
           "balanced": (p["delta"] + sl["proxy_delta"]) / 2.0,
           "eval_seed": eval_seed, "elapsed_s": round(_t.monotonic() - t0)}
    with open(Path("ft/research/results.jsonl"), "a") as f:
        f.write(json.dumps(row, sort_keys=True) + chr(10))
    print(f"done s1-safety: engine={row['engine_delta']:+.4f} "
          f"proxy={row['proxy_delta']:+.4f} "
          f"balanced={row['balanced']:+.4f}", flush=True)
    return row


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
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--selfplay-games", type=int, default=20)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--configs", default="a1,a2,a3")
    ap.add_argument("--reinforce-games", type=int, default=None,
                    help="override the d-arm game count (d1/d2 default 200)")
    ap.add_argument("--reinforce-device", default="cpu",
                    help="device for the outcome-game collector (cuda on "
                         "the GPU box)")
    ap.add_argument("--reinforce-seed", default=None,
                    help="override the d-arm opening-set seed (vary per "
                         "generation for fresh cells)")
    ap.add_argument("--safety-arm", action="store_true")
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
    from ft.common.serialize import jsonl_read
    import gc

    def load_corpus():
        return list(jsonl_read(corpus_path))

    corpus = None
    if args.stage != "sweep":
        corpus = load_corpus()
        print(f"corpus: loaded {len(corpus)} states", flush=True)
    if args.stage == "corpus":
        return
    base_path = args.anchor_path or args.model_path
    records_cache: dict = {}

    if args.safety_arm:
        run_safety_arm(out_dir, model_path=args.model_path,
                       base_path=base_path, eval_openings=args.eval_openings,
                       selfplay_games=args.selfplay_games,
                       eval_workers=args.eval_workers,
                       eval_seed=args.eval_seed,
                       records_cache=records_cache)
        gc.collect()
    for name in args.configs.split(","):
        if not name:
            continue
        if name in REINFORCE_CONFIGS:
            cfg = dict(REINFORCE_CONFIGS[name])
            cfg["name"] = name
            cfg["device"] = args.device
            cfg.setdefault("batch", args.batch)
            cfg["engine_path"] = args.engine
            cfg["sf_depth"] = args.sf_depth
            cfg["sf_workers"] = args.sf_workers
            if args.reinforce_games is not None:
                cfg["games"] = args.reinforce_games
            if args.reinforce_seed is not None:
                cfg["rein_seed"] = args.reinforce_seed
            cfg["rein_device"] = args.reinforce_device
            run_reinforce_arm(out_dir, model_path=args.model_path,
                              base_path=base_path, cfg=cfg,
                              eval_openings=args.eval_openings,
                              selfplay_games=args.selfplay_games,
                              eval_workers=args.eval_workers,
                              eval_seed=args.eval_seed,
                              sf_depth=cfg["sf_depth"],
                              sf_workers=cfg["sf_workers"],
                              engine_path=cfg["engine_path"],
                              records_cache=records_cache)
            gc.collect()
            continue
        if name in CURRICULUM_CONFIGS:
            if corpus is None:
                corpus = load_corpus()
            cfg = dict(CURRICULUM_CONFIGS[name])
            cfg["device"] = args.device
            cfg.setdefault("batch", args.batch)
            run_curriculum_arm(corpus, out_dir, name=name,
                               model_path=args.model_path,
                               base_path=base_path, cfg=cfg,
                               eval_openings=args.eval_openings,
                               selfplay_games=args.selfplay_games,
                               eval_workers=args.eval_workers,
                               eval_seed=args.eval_seed,
                               records_cache=records_cache)
            corpus = None
            gc.collect()
            continue
        if corpus is None:
            corpus = load_corpus()
        cfg = dict(CONFIGS[name])
        cfg["device"] = args.device
        cfg.setdefault("batch", args.batch)
        train_and_eval(corpus, name=name, out_dir=out_dir,
                       model_path=args.model_path, base_path=base_path,
                       cfg=cfg, eval_openings=args.eval_openings,
                       selfplay_games=args.selfplay_games,
                       eval_workers=args.eval_workers,
                       eval_seed=args.eval_seed,
                       records_cache=records_cache)
        corpus = None
        gc.collect()


if __name__ == "__main__":
    main()
