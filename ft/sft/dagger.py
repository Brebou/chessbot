"""Stage 2 - Slice DAgger collection (spec section 9), prepared here, executed
after SFT iteration 1.

Pipeline: greedy self-play from real book openings (same 8 forced plies as
the eval); every episode stops as soon as it reaches ``stop_pieces``. The
visited positions are deduplicated by FEN, then annotated with the same
Phase-C engine settings (fixed depth, MultiPV 2). Every annotated position
becomes a CE row ``{bucket: "dagger", target = engine best move}`` carrying
the real prefix history.

OOD priority: a position whose (piece count, material balance) signature has
no elite-corpus counterpart within +-1 on both components is ``ood=True`` and
is kept first when the iteration-2 share cap trims.

Anti-forgetting: iteration-1 `anchor` rows stay in the mix; ``blend_dagger``
bounds the dagger share (default 15%).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chess

from chess_hw import ChessTokenizer
from chess_hw.model import load_model
from chess_hw.match import load_openings, select_openings
from diag.common import material_value
from diag.fast_probe import FastProbePlayer
from ft.common.board_io import n_pieces_from_fen
from ft.common.serialize import jsonl_read, jsonl_write

MAX_SHARE = 0.15
OOD_TOL = 1


def position_signature(fen: str) -> tuple:
    board = chess.Board(fen)
    return (n_pieces_from_fen(fen),
            material_value(board, chess.WHITE) - material_value(board, chess.BLACK))


def load_elite_signatures(elite_positions_path: str) -> List[tuple]:
    if not elite_positions_path or not Path(elite_positions_path).exists():
        return []
    return [position_signature(r["fen_before"])
            for r in jsonl_read(elite_positions_path) if r.get("fen_before")]


def is_ood(sig: tuple, elite_sigs: List[tuple], tol: int = OOD_TOL) -> bool:
    if not elite_sigs:
        return True
    np_, bal = sig
    for en, eb in elite_sigs:
        if abs(en - np_) <= tol and abs(eb - bal) <= tol:
            return False
    return True


# -- self-play rollouts -------------------------------------------------------

def rollout_positions(model_path: str, n_episodes: int, *, stop_pieces: int = 16,
                      opening_plies: int = 8, seed: int = 0, device: Optional[str] = None,
                      batch_size: int = 512, max_plies: int = 120) -> List[dict]:
    """Greedy model self-play; returns {"fen", "history"} per visited state."""
    tokenizer = ChessTokenizer()
    model = load_model(model_path, device=device)
    probe = FastProbePlayer(model, tokenizer, batch_size=batch_size)
    book = load_openings()
    oids = select_openings(seed, n_episodes)
    active = []
    for e, oid in enumerate(oids):
        board = chess.Board()
        hist = [m for m in book[oid % len(book)][:opening_plies]]
        for uci in hist:
            board.push_uci(uci)
        active.append({"board": board, "hist": hist,
                       "rng": random.Random(f"dagger-ep-{seed}-{e}")})
    rows: Dict[str, dict] = {}
    for _ in range(max_plies):
        if not active:
            break
        choices = probe.choose_from_histories([a["hist"] for a in active],
                                              [a["board"] for a in active])
        still = []
        for a, choice in zip(active, choices):
            board, hist, rng = a["board"], a["hist"], a["rng"]
            if board.outcome(claim_draw=True) is not None:
                continue
            if n_pieces_from_fen(board.fen()) > stop_pieces:
                key = board.fen()
                if key not in rows:
                    rows[key] = {"fen": key, "history": list(hist)}
            if n_pieces_from_fen(board.fen()) <= stop_pieces:
                continue
            legal = list(board.legal_moves)
            if not legal:
                continue
            mv = choice.move if choice.move is not None \
                else rng.choice(legal)
            board.push(mv)
            hist.append(mv.uci())
            if len(hist) < opening_plies + max_plies:
                still.append(a)
        active = still
    return list(rows.values())


# -- engine annotation (same settings as Phase C) ------------------------------

_WORKER_ANNOTATOR: Dict = {}


def _dagger_annotate_task(task):
    """Module-level spawn worker: (fen, engine_path, depth)."""
    fen, engine_path, depth = task
    from ft.data.annotate import StockfishAnnotator

    key = (engine_path, depth)
    if key not in _WORKER_ANNOTATOR:
        _WORKER_ANNOTATOR[key] = StockfishAnnotator(engine_path, depth)
    annotator = _WORKER_ANNOTATOR[key]
    try:
        block = annotator.annotate(fen)
    finally:
        annotator.close()
    return {"fen": fen, "best_uci": block["best_uci"],
            "cp_best": block["cp_best"], "cp_second": block["cp_second"],
            "mate_root": block["mate_root"]}


def _annotate(positions: List[dict], *, engine_path: str, depth: int,
              workers: int, annotate_fn: Optional[Callable], progress=None) -> Dict[str, dict]:
    tasks = [(r["fen"], engine_path, depth) for r in positions]
    out: Dict[str, dict] = {}
    if annotate_fn is not None:
        for i, t in enumerate(tasks):
            res = annotate_fn(t[0])
            out[t[0]] = res
            if progress is not None:
                progress.show(f"dagger annotate: {i + 1}/{len(tasks)}")
        return out
    from multiprocessing import get_context

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for res in pool.imap(_dagger_annotate_task, tasks, chunksize=4):
            out[res["fen"]] = res
            if progress is not None:
                progress.show(f"dagger annotate: {len(out)}/{len(tasks)}")
    return out


# -- collection ---------------------------------------------------------------

def dagger_collect(model_path: str, out: str, *, n_episodes: int = 400,
                   stop_pieces: int = 16, opening_plies: int = 8, seed: int = 0,
                   device: Optional[str] = None, batch_size: int = 512,
                   max_plies: int = 120, engine_path: str = "stockfish",
                   depth: int = 14, workers: int = 8,
                   elite_positions_path: str = "",
                   annotate_fn: Optional[Callable] = None,
                   force: bool = False, progress=None) -> dict:
    out_path = Path(out)
    if out_path.exists() and not force:
        rows = list(jsonl_read(out_path))
        return {"n_rows": len(rows),
                "n_ood": sum(1 for r in rows if r.get("ood"))}
    positions = rollout_positions(model_path, n_episodes, stop_pieces=stop_pieces,
                                  opening_plies=opening_plies, seed=seed,
                                  device=device, batch_size=batch_size,
                                  max_plies=max_plies)
    elite = load_elite_signatures(elite_positions_path)
    anns = _annotate(positions, engine_path=engine_path, depth=depth,
                     workers=workers, annotate_fn=annotate_fn, progress=progress)
    rows: List[dict] = []
    for r in positions:
        ann = anns.get(r["fen"])
        if ann is None or ann.get("best_uci") is None:
            continue
        target = ann["best_uci"]
        try:
            chess.Board(r["fen"]).parse_uci(target)
        except ValueError:
            continue
        rows.append({"bucket": "dagger", "key": r["fen"],
                     "start_fen": r["fen"], "history": r["history"],
                     "target": target, "weight": 1.0,
                     "ood": is_ood(position_signature(r["fen"]), elite),
                     "n_pieces": n_pieces_from_fen(r["fen"])})
    rows.sort(key=lambda r: not r["ood"])
    jsonl_write(out_path, rows)
    stats = {"n_positions": len(positions), "n_rows": len(rows),
             "n_ood": sum(1 for r in rows if r["ood"])}
    print(f"✅ dagger: {stats}")
    return stats


def blend_dagger(rows: List[dict], dagger_rows: List[dict], *,
                 max_share: float = MAX_SHARE, seed: int = 0) -> List[dict]:
    """Bind the dagger share of the iteration-2 mix to `max_share` (share of
    the final dataset), OOD rows first."""
    if not dagger_rows:
        return list(rows)
    rng = random.Random(f"blend-dagger-{seed}")
    held = list(rows)
    rng.shuffle(held)
    order = sorted(dagger_rows, key=lambda r: not r.get("ood", False))
    max_dagger = int(max_share / (1 - max_share) * len(held)) \
        if max_share < 1 else len(order)
    return held + order[:max_dagger]
