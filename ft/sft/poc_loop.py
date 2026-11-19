"""Toy continuous on-policy correction loop (proof of concept).

Nothing is reused from previous collections or mixes. Each iteration:

1. plays fresh full games with the current policy: by default 1/3 pure
   self-play and 2/3 vs the FastEngine class-engine lookalike (half with
   the model as white), engine level sampled per game (L1 10%, L2 30%,
   L3 60%); openings from `chess_hw.match`, official greedy decode with
   seeded random fallback;
2. scores every visited state's played move at a LOW engine depth:
   analysis before, push the model move, analysis after (both mover-POV);
3. keeps correction rows only for the three explicit failure classes —
   eval loss > cp_threshold, missed forced mate in <= quick_mate_plies,
   endgame (<= mate_piece_gate pieces) failure to convert a forced mate
   within mate_slack plies. Asymmetric endgames (KRK-style, many mating
   moves) emit one row per equally-fast mating alternative (multipv),
   weight 1/n;
4. briefly fine-tunes on the fresh corrections only, with a forward-KL
   anchor to the frozen base policy (--kl-lambda) and a small lr;
5. evaluates paired vs base at the trainer's epoch ends, then discards the
   corrections and resamples new games.

The point is fast feedback on whether ANY post-training update to this
946k-param model can be made non-negative.

Resume: pass the last policy as ``--model-path`` and keep the frozen
reference (``--base-path base_model``) so the KL anchor and the paired
eval baseline do not silently drift with the policy. The base model's
engine-game records are cached to ``base_engine_records.jsonl`` in the
out dir and reused across resumes (the games are deterministic).

Artifacts at the run level (``--out-dir``): ``base_engine_records.jsonl``
(base games cache), ``metrics.jsonl`` (every eval row, tagged ``iteration``
— the centralized view of all per-iter trainer metrics), and
``summary.jsonl`` (one row per completed iteration: corrections, best
score, timings, GPU line). Per-iteration detail stays under ``iterN/``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess

from chess_hw.play import MoveChoice
from diag.common import ProgressLine
from ft.common.board_io import n_pieces_from_fen
from ft.common.serialize import jsonl_write
from ft.sft.dagger import rollout_positions
from ft.sft.train_sft import FTSFTConfig, FTSFTTrainer

CP_BLUNDER_THRESHOLD = 150.0
MATE_QUICK_PLIES = 2
MATE_PIECE_GATE = 10
MATE_SLACK_PLIES = 1
MULTIPV = 5
MAX_POC_PLIES = 250
BASE_RECORDS_NAME = "base_engine_records.jsonl"
BASE_RECORDS_SCHEMA = 1

_POC_ENGINES: Dict = {}


def base_records_meta(base_path: str, levels, n_openings: int, seed: int) -> dict:
    return {"schema": BASE_RECORDS_SCHEMA,
            "base_path": str(Path(base_path).resolve()),
            "levels": sorted(set(levels)),
            "n_openings": n_openings,
            "seed": seed}


def save_base_records(path: Path, meta: dict, records: list) -> None:
    tmp = path.with_suffix(".tmp.jsonl")
    with open(tmp, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_base_records(path: Path, meta: dict) -> Optional[list]:
    try:
        with open(path) as f:
            header = json.loads(f.readline())
            if header.get("meta") != meta:
                return None
            records = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return None
    return records or None


def _worker_engine(path: str, depth: int):
    import chess.engine

    key = (path, depth)
    if key not in _POC_ENGINES:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        engine.configure({"Threads": 1})
        _POC_ENGINES[key] = engine
    return _POC_ENGINES[key]


def _poc_score_task(task):
    """Module-level spawn worker: (fen, model_move, engine_path, depth).

    Returns mover-POV scores before and after the model's move, plus the
    engine best and all multipv alternatives mating equally fast."""
    fen, model_move, engine_path, depth = task
    import chess.engine

    from ft.data.annotate import score_to_mover

    try:
        engine = _worker_engine(engine_path, depth)
        board = chess.Board(fen)
        infos = engine.analyse(board, chess.engine.Limit(depth=depth),
                               multipv=MULTIPV)
        if isinstance(infos, dict):
            infos = [infos]
        infos = sorted(infos, key=lambda i: i["score"].pov(board.turn)
                       .score(mate_score=10000), reverse=True)
        first = infos[0]
        pov = first["score"].pov(board.turn)
        cp_best = score_to_mover(first["score"])
        mate_best = pov.mate() if pov.is_mate() and pov.mate() != 0 else None
        if mate_best is not None:
            alt_mates = [i["pv"][0].uci() for i in infos
                         if i["score"].pov(board.turn).mate() == mate_best]
        else:
            alt_mates = []
        board.push_uci(model_move)
        after = engine.analyse(board, chess.engine.Limit(depth=depth))
        after_pov = after["score"].pov(board.turn)
        cp_model = -score_to_mover(after["score"])
        mate_model = (-after_pov.mate()
                      if after_pov.is_mate() and after_pov.mate() != 0 else None)
        return {"fen": fen, "cp_best": cp_best, "mate_best": mate_best,
                "cp_model": cp_model, "mate_model": mate_model,
                "best_uci": first["pv"][0].uci(), "alt_mates": alt_mates}
    except Exception:
        return {"fen": fen, "cp_best": None, "mate_best": None,
                "cp_model": None, "mate_model": None, "best_uci": None,
                "alt_mates": []}


def poc_score(tasks: List[tuple], *, engine_path: str, depth: int,
              workers: int, score_fn: Optional[Callable] = None,
              progress=None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if score_fn is not None:
        for fen, model_move in tasks:
            out[fen] = score_fn(fen, model_move)
        if progress is not None:
            progress.show(f"poc score: {len(out)}/{len(tasks)}", force=True)
        return out
    from multiprocessing import get_context

    ctx = get_context("spawn")
    pool_tasks = [(fen, model_move, engine_path, depth)
                  for fen, model_move in tasks]
    with ctx.Pool(workers) as pool:
        for res in pool.imap(_poc_score_task, pool_tasks, chunksize=8):
            out[res["fen"]] = res
            if progress is not None:
                progress.show(f"poc score: {len(out)}/{len(tasks)}")
    return out


def _gpu_memory_line(device: str) -> str:
    """Allocated/reserved/peak MiB line; '' when not on CUDA."""
    import torch

    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return ""
    gc.collect()
    out = (f"gpu: allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB | "
           f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB | "
           f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out + f" -> reserved now {torch.cuda.memory_reserved() / 2**20:.0f} MiB"


class _FastEngineOpponent:
    """FastEngine behind the official Player interface (persistent per worker)."""

    def __init__(self, level: int, seed: str):
        from diag.fast_engine import FastEngine

        self.name = f"engine-L{level}"
        self.engine = FastEngine(level=level, seed=seed)

    def choose(self, board: chess.Board) -> MoveChoice:
        return MoveChoice(self.engine.best_move(board), 1)


def allocate_level_counts(n: int, probs: List[float]) -> List[int]:
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"level probabilities must sum to 1, got {probs}")
    counts = [int(round(p * n)) for p in probs]
    counts[probs.index(max(probs))] += n - sum(counts)
    return counts


def _mix_engine_game_task(task):
    """Spawn worker: (oid, model_white, level, opening_plies, max_plies, seed).

    Plays one full game model-vs-FastEngine with official inference semantics
    (ModelPlayer: first->second->third legal, seeded random fallback) and
    returns the model's decision states: {fen, history, model_move} per
    model turn."""
    from chess_hw.match import _get_player, load_openings, play_game

    oid, model_white, level, opening_plies, max_plies, seed = task
    model = _get_player("m")
    side = "w" if model_white else "b"
    engine = _FastEngineOpponent(
        level, seed=f"mix-{seed}-{level}-{oid}-{side}")
    opening = load_openings()[oid]
    white, black = (model, engine) if model_white else (engine, model)
    rec = play_game(white, black, opening, oid, max_plies=max_plies)
    moves = rec.moves
    board = chess.Board()
    states = []
    for i, uci in enumerate(moves):
        if ((i % 2 == 0) == model_white) and uci:
            states.append({"fen": board.fen(), "history": moves[:i],
                           "model_move": uci})
        board.push_uci(uci)
    return states


def engine_opponent_states(model_path: str, n_games: int, *, opening_plies: int,
                           seed: str, engine_levels: List[int],
                           level_probs: List[float], workers: int,
                           max_plies: int,
                           game_fn: Optional[Callable] = None,
                           progress=None) -> List[dict]:
    """Model-vs-FastEngine games (half model-white, half model-black),
    level sampled per game from engine_levels with level_probs; returns the
    model's decision states in the rollout row format."""
    show = (progress.show if progress is not None
            else (lambda msg, force=False: None))
    if n_games <= 0:
        return []
    from chess_hw.match import load_openings

    show(f"poc rollouts: playing {n_games} engine-opponent games...")
    n_openings = len(load_openings())
    rng = random.Random(f"opp-mix-{seed}")
    levels_seq: List[int] = []
    for level, count in zip(engine_levels,
                            allocate_level_counts(n_games, level_probs)):
        levels_seq += [level] * count
    rng.shuffle(levels_seq)
    tasks = [(rng.randrange(n_openings), g % 2 == 0, level, opening_plies,
              max_plies, seed)
             for g, level in enumerate(levels_seq)]
    if game_fn is not None:
        results = [game_fn(t) for t in tasks]
    else:
        from chess_hw.match import run_tasks

        results = run_tasks(_mix_engine_game_task, tasks,
                            {"m": str(model_path)}, workers, progress=True)
    states = [s for record in results for s in record]
    show(f"poc rollouts: {len(states)} model states from {n_games} "
         f"engine-opponent games", force=True)
    return states


def poc_classify(cp_best: Optional[int], cp_model: Optional[int],
                 mate_best: Optional[int], mate_model: Optional[int],
                 n_pieces: int, *, cp_threshold: float = CP_BLUNDER_THRESHOLD,
                 mate_piece_gate: int = MATE_PIECE_GATE,
                 quick_mate_plies: int = MATE_QUICK_PLIES,
                 mate_slack_plies: int = MATE_SLACK_PLIES) -> bool:
    """True iff the model's played move must be corrected."""
    if mate_model is not None and mate_model < 0:
        return True
    mover_mates = mate_best is not None and mate_best > 0
    model_mates = mate_model is not None and mate_model > 0
    if mover_mates:
        loses = not model_mates or mate_model > mate_best
        still_fast = not model_mates or mate_model > mate_best + mate_slack_plies
        if mate_best <= quick_mate_plies and loses:
            return True
        if n_pieces <= mate_piece_gate and still_fast:
            return True
    if cp_best is not None and cp_model is not None \
            and cp_best - cp_model > cp_threshold:
        return True
    return False


def build_corrections(states: List[dict], scores: Dict[str, dict], *,
                      cp_threshold: float, mate_piece_gate: int,
                      quick_mate_plies: int, mate_slack_plies: int,
                      max_alt_rows: int = 3) -> List[dict]:
    """Correction rows (bucket `normal`, engine-best targets) for the
    classified blunders; endgame mate blunders additionally spread weight
    over equally-fast mating alternatives."""
    rows: List[dict] = []
    for s in states:
        sc = scores.get(s["fen"])
        if sc is None or sc.get("best_uci") is None:
            continue
        if not poc_classify(sc.get("cp_best"), sc.get("cp_model"),
                            sc.get("mate_best"), sc.get("mate_model"),
                            n_pieces_from_fen(s["fen"]),
                            cp_threshold=cp_threshold,
                            mate_piece_gate=mate_piece_gate,
                            quick_mate_plies=quick_mate_plies,
                            mate_slack_plies=mate_slack_plies):
            continue
        targets = [sc["best_uci"]]
        if sc.get("mate_best") is not None and sc.get("mate_best") > 0 \
                and n_pieces_from_fen(s["fen"]) <= mate_piece_gate:
            for alt in sc.get("alt_mates", []):
                if alt != sc["best_uci"] and alt not in targets:
                    targets.append(alt)
        targets = targets[:max_alt_rows]
        weight = 1.0 / len(targets)
        for t in targets:
            rows.append({"bucket": "normal", "key": s["fen"],
                         "start_fen": s["fen"], "history": s["history"],
                         "target": t, "weight": weight})
    return rows


def run_poc_loop(args, score_fn: Optional[Callable] = None,
                 game_fn: Optional[Callable] = None,
                 progress=None) -> List[dict]:
    show = progress.show if progress is not None else (lambda msg: None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = getattr(args, "base_path", "") or args.model_path
    policy_path = args.model_path
    policy_export = out_dir / "policy"
    policy_export.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []
    it = 0
    records_cache: Dict[str, list] = {}
    eval_openings = getattr(args, "eval_openings", 40)
    eval_every = max(1, getattr(args, "eval_every_iterations", 1))
    records_path = out_dir / BASE_RECORDS_NAME
    records_meta = base_records_meta(base_path, args.eval_levels,
                                     eval_openings, args.seed)
    records_loaded = False
    cached_records = load_base_records(records_path, records_meta)
    if cached_records is not None:
        records_loaded = True
        records_cache[base_path] = cached_records
        print(f"\u267b\ufe0f loaded {len(cached_records)} cached base engine "
              f"records from {records_path}", flush=True)
    eval_seed = getattr(args, "eval_seed", None)
    if eval_seed is None:
        eval_seed = args.seed   # anchored: same eval openings every iteration
    try:
        while args.iterations <= 0 or it < args.iterations:
            it += 1
            do_eval = (it % eval_every == 0
                       or (args.iterations > 0 and it == args.iterations))
            show(f"poc iter {it}: playing {args.games} games...")
            t0 = time.monotonic()
            n_engine = int(round(args.games * getattr(args, "engine_share", 0.0)))
            n_self = max(args.games - n_engine, 0)
            states = []
            if n_self > 0:
                states += rollout_positions(policy_path, n_self, stop_pieces=0,
                                            opening_plies=8, seed=args.seed * 1000 + it,
                                            device=args.device,
                                            batch_size=args.batch_size,
                                            max_plies=MAX_POC_PLIES, progress=progress)
            if n_engine > 0:
                states += engine_opponent_states(
                    policy_path, n_engine, opening_plies=8,
                    seed=f"{args.seed}-{it}",
                    engine_levels=list(getattr(args, "opp_levels", [1, 2, 3])),
                    level_probs=list(getattr(args, "opp_level_probs",
                                             [0.1, 0.3, 0.6])),
                    workers=args.workers, max_plies=MAX_POC_PLIES,
                    game_fn=game_fn, progress=progress)
            t1 = time.monotonic()
            tasks = [(s["fen"], s["model_move"]) for s in states]
            scores = poc_score(tasks, engine_path=args.engine, depth=args.depth,
                               workers=args.workers, score_fn=score_fn,
                               progress=progress)
            corrections = build_corrections(
                states, scores, cp_threshold=args.cp_threshold,
                mate_piece_gate=args.mate_piece_gate,
                quick_mate_plies=args.quick_mate_plies,
                mate_slack_plies=args.mate_slack_plies)
            mix_path = out_dir / f"mix_iter{it}.jsonl"
            jsonl_write(mix_path, corrections)
            t2 = time.monotonic()
            n_failed = sum(1 for t in tasks
                           if scores.get(t[0], {}).get("best_uci") is None)
            print(f"✅ poc iter {it}: {len(states)} states, "
                  f"{len(corrections)} correction rows "
                  f"({n_failed} scored failed)", flush=True)
            print(f"\u23f1\ufe0f poc iter {it}: selfplay {t1 - t0:.0f}s | "
                  f"scoring {t2 - t1:.0f}s | train+eval pending...", flush=True)
            if not corrections:
                summary.append({"iteration": it, "corrections": 0})
                with open(out_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"iteration": it, "corrections": 0,
                                        "elapsed": time.monotonic() - t0,
                                        "note": "no corrections"},
                                       sort_keys=True) + "\n")
                continue
            cfg = FTSFTConfig(
                model_path=policy_path, base_path=base_path,
                mix_path=str(mix_path), out_dir=str(out_dir / f"iter{it}"),
                device=args.device, seed=eval_seed, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                kl_lambda=args.kl_lambda,
                eval_mode=args.eval_mode if do_eval else "none",
                eval_every_steps=0, eval_levels=tuple(args.eval_levels),
                eval_workers=args.eval_workers,
                eval_n_openings=getattr(args, "eval_openings", 40),
                selfplay_games=getattr(args, "selfplay_games", 200),
                plateau_patience=args.patience,
                fixtures_dir=args.fixtures_dir,
                fixtures_games=args.fixtures_games)
            trainer = FTSFTTrainer(cfg)
            trainer._records_cache = records_cache
            done = trainer.train()
            done["iteration"] = it
            done["corrections"] = len(corrections)
            summary.append(done)
            done_s = time.monotonic()
            if not records_loaded and records_cache.get(base_path):
                save_base_records(records_path, records_meta,
                                  records_cache[base_path])
                records_loaded = True
            final = done.get("final_model")
            if final:
                policy_path = final
            gpu = _gpu_memory_line(args.device)
            done["gpu"] = gpu
            with open(out_dir / "summary.jsonl", "a") as f:
                f.write(json.dumps(done, sort_keys=True) + "\n")
            run_metrics = out_dir / "metrics.jsonl"
            with open(run_metrics, "a") as f:
                for row in trainer.metric_rows():
                    row["iteration"] = it
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"\u23f1\ufe0f poc iter {it}: selfplay {t1 - t0:.0f}s | "
                  f"scoring {t2 - t1:.0f}s | train+eval {done_s - t2:.0f}s | "
                  f"total {done_s - t0:.0f}s", flush=True)
            if gpu:
                print(f"  {gpu}", flush=True)
    except KeyboardInterrupt:
        print(f"\n⏹️ poc loop interrupted after {len(summary)} completed "
              f"iterations; last policy: {policy_path}", flush=True)
    print(f"✅ poc loop done: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toy loop: fresh self-play games -> correct blunders -> "
                    "KL-anchored micro fine-tune -> repeat")
    parser.add_argument("--model-path", default="base_model")
    parser.add_argument("--base-path", default="",
                        help="frozen reference for the KL anchor and paired "
                             "eval; default: same as --model-path. Pass the "
                             "original base model when resuming so the eval "
                             "baseline does not drift with the policy")
    parser.add_argument("--out-dir", default="results/poc_loop")
    parser.add_argument("--iterations", type=int, default=6,
                        help="number of iterations; 0 = run until Ctrl-C")
    parser.add_argument("--games", type=int, default=500,
                        help="fresh games per iteration (self-play plus "
                             "engine-opponent mix)")
    parser.add_argument("--engine-share", type=float, default=0.66,
                        help="share of games vs the FastEngine opponent "
                             "(split half model-white, half model-black); "
                             "the rest is full self-play. 0.66 = 1/3 "
                             "self-play, 2/3 vs engine")
    parser.add_argument("--opp-levels", type=int, nargs="+", default=[1, 2, 3],
                        help="FastEngine levels sampled as opponents")
    parser.add_argument("--opp-level-probs", type=float, nargs="+",
                        default=[0.1, 0.3, 0.6],
                        help="sampling probabilities for --opp-levels")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl-lambda", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cp-threshold", type=float,
                        default=CP_BLUNDER_THRESHOLD)
    parser.add_argument("--mate-piece-gate", type=int, default=MATE_PIECE_GATE)
    parser.add_argument("--quick-mate-plies", type=int,
                        default=MATE_QUICK_PLIES)
    parser.add_argument("--mate-slack-plies", type=int,
                        default=MATE_SLACK_PLIES)
    parser.add_argument("--eval-workers", type=int, default=16)
    parser.add_argument("--eval-mode", default="paired")
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="fixed eval seed (same openings/cells at every "
                             "iteration); default: --seed")
    parser.add_argument("--eval-every-iterations", type=int, default=1,
                        help="paired qualification eval every N iterations")
    parser.add_argument("--eval-openings", type=int, default=40,
                        help="openings per level/color in paired engine eval")
    parser.add_argument("--selfplay-games", type=int, default=200,
                        help="head-to-head games in the paired proxy eval")
    parser.add_argument("--eval-levels", type=int, nargs="+",
                        default=[1, 2, 3],
                        help="engine levels for the paired eval; the official "
                             "metric is (S1+S2+S3)/3 and level 0 (random "
                             "opponent) only adds noise at the ceiling")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--fixtures-dir", default="")
    parser.add_argument("--fixtures-games", default="")
    args = parser.parse_args()
    from diag.common import ProgressLine
    run_poc_loop(args, progress=ProgressLine())


if __name__ == "__main__":
    main()
