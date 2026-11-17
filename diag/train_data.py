"""SFT mix builder for Phase 1 (combined supervised fine-tune).

Produces ``data/sft_mix.jsonl``: one row per training example,

    {"kind": ..., "history": [uci, ...], "target": "uci", "bad": "uci"?, "weight": float}

with kinds (see the plan d'attaque, Phase 1):

- ``replay``    — next-move prediction on the model's own pre-training corpus
                  (first ``--n-games`` games = the seen 940k), anti-forgetting;
- ``capture``   — CE on the best SEE > 0 capture when the played move missed it
                  (weight grows with the SEE, capped at 3);
- ``mate1``     — CE on a mating move (one per position, first found);
- ``hang``      — margin pair: target = the (safe) played move, ``bad`` = an
                  alternative move that hangs >= 1 pawn (opponent's best
                  capture SEE >= 100 after it);
- ``endgame``   — strong-side decisions from real KQK/KRK positions reached in
                  corpus games, plus engine-vs-engine demonstrations from
                  synthetic ``generate_endgame_position`` starts;
- ``promotion`` — safe queen promotions (destination not capturable right after
                  the promotion, or the promotion gives check).

Positions come from replayed games, so every history is real and
parity-correct. Curriculum: plies 9-30 and plies >= 121 get a configurable
weight multiplier (the eval's real decision zone + the conversion zone).

**Data quality split**: tactical examples (capture, mate1, hang, endgame,
promotion) are mined from the *high-quality Elo-filtered* Lichess dumps
(austindavis monthly files with ``WhiteElo``/``BlackElo``, e.g.
``data/lichess_uci_201412.jsonl.gz``, default threshold 2200) so the "good"
moves the model is taught come from strong play, not average blunder-prone
games. The ``replay`` anti-forgetting examples stay on the model's own
pre-training corpus (the seen 940k) to match its training distribution.
If the HQ file is missing, everything falls back to the training corpus.

All computations are seeded, parallelized over a spawn pool of module-level
workers, cached to the output file (skip if it exists, unless ``--force``),
and written atomically.

Usage:
    uv run diag/train_data.py --dataset data/lichess_train_2025-01_1M.jsonl.gz \
        --hq-dataset data/lichess_uci_201412.jsonl.gz --min-elo 2200 \
        --out data/sft_mix.jsonl --n-games 300000 --workers 36 --seed 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw.match import MAX_PLIES

from diag.common import ProgressLine, jsonl_write
from diag.fast_engine import FastEngine
from diag.gen_corpus import generate_endgame_position
from diag.see import see


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def ply_weight(ply: int, early_start: int = 9, early_end: int = 30,
               early_mult: float = 2.0, endgame_ply: int = 121,
               endgame_mult: float = 2.0) -> float:
    """Curriculum multiplier: over-weight the eval decision zone (plies 9-30)
    and endgames (plies >= 121)."""
    if ply >= endgame_ply:
        return endgame_mult
    if early_start <= ply <= early_end:
        return early_mult
    return 1.0


# ---------------------------------------------------------------------------
# Per-position extraction
# ---------------------------------------------------------------------------

def _opponent_best_capture_see(board: chess.Board) -> int:
    """Best SEE value (centipawns, mover POV) over the side-to-move's captures."""
    best = 0
    for mv in board.legal_moves:
        if board.is_capture(mv):
            s = see(board, mv)
            if s > best:
                best = s
    return best


def extract_examples(board: chess.Board, played_uci: str, rng: random.Random,
                     hang_tries: int = 8) -> List[dict]:
    """Tactical examples for one (position, played move) pair."""
    examples: List[dict] = []
    legal = list(board.legal_moves)
    if not legal:
        return examples

    # mate in 1 (first mating move found)
    for m in legal:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            examples.append({"kind": "mate1", "target": m.uci(), "weight": 1.0})
            break

    # best winning capture
    best_mv, best_see = None, 0
    for m in legal:
        if board.is_capture(m):
            s = see(board, m)
            if s > best_see:
                best_mv, best_see = m, s
    try:
        played = board.parse_uci(played_uci)
    except ValueError:
        return examples
    played_winning_cap = board.is_capture(played) and see(board, played) > 0
    if best_mv is not None and not played_winning_cap and played != best_mv:
        w = min(3.0, 1.0 + best_see / 300.0)
        examples.append({"kind": "capture", "target": best_mv.uci(), "weight": w})

    # hang pair: an alternative move that hangs >= 1 pawn
    board.push(played)
    played_safe = _opponent_best_capture_see(board) < 100
    board.pop()
    if played_safe:
        opp = not board.turn
        risky = []
        for m in legal:
            if m == played or board.is_capture(m):
                continue  # losing captures handled by the capture loss
            board.push(m)
            if board.is_attacked_by(opp, m.to_square):
                risky.append(m)
            board.pop()
        rng.shuffle(risky)
        for m in risky[:hang_tries]:
            board.push(m)
            hangs = _opponent_best_capture_see(board) >= 100
            board.pop()
            if hangs:
                examples.append({"kind": "hang", "target": played_uci,
                                 "bad": m.uci(), "weight": 1.0})
                break
    return examples


# ---------------------------------------------------------------------------
# Per-game mining worker (module-level for the spawn pool)
# ---------------------------------------------------------------------------

def mine_game(task: Tuple[int, str, dict]) -> List[dict]:
    """(game_idx, moves_str, cfg) -> list of example dicts for one game."""
    idx, moves_str, cfg = task
    moves = moves_str.split()
    rng = random.Random(f"mine-{cfg['seed']}-{idx}")
    board = chess.Board()
    n = len(moves)
    if n == 0:
        return []

    # decision plies: all plies 9-30 with prob 0.2, plus 8 random plies later;
    # short games (<= 12 moves) are mined at every ply
    decision = set()
    upper = min(n, MAX_PLIES)
    if upper <= 12:
        decision = set(range(upper))
    else:
        for ply in range(8, min(n, 31)):
            if rng.random() < 0.2:
                decision.add(ply)
        if upper > 31:
            for ply in rng.sample(range(31, upper), min(8, upper - 31)):
                decision.add(ply)
        if not decision:
            decision = set(rng.sample(range(upper), min(4, upper)))

    endgame_left = cfg.get("endgame_per_game", 4)
    promo_left = cfg.get("promotion_per_game", 2)
    out: List[dict] = []
    history: List[str] = []

    for ply, uci in enumerate(moves):
        if ply >= MAX_PLIES:
            break
        try:
            move = board.parse_uci(uci)
        except ValueError:
            break

        if ply in decision:
            w = ply_weight(ply)
            out.append({"kind": "replay", "history": list(history),
                        "target": uci, "weight": w})
            for ex in extract_examples(board, uci, rng,
                                       cfg.get("hang_tries", 8)):
                ex.setdefault("history", list(history))
                out.append(ex)

        # cheap endgame detection: exactly 3 pieces, one is Q or R
        if endgame_left > 0 and len(board.piece_map()) == 3:
            pieces = list(board.piece_map().values())
            types = [p.piece_type for p in pieces]
            if chess.QUEEN in types or chess.ROOK in types:
                strong_color = next(p.color for p in pieces
                                    if p.piece_type in (chess.QUEEN, chess.ROOK))
                mover = "white" if board.turn == chess.WHITE else "black"
                if mover == ("white" if strong_color == chess.WHITE else "black"):
                    out.append({"kind": "endgame", "history": list(history),
                                "target": uci, "weight": 1.0})
                    endgame_left -= 1

        # safe promotion available for the side to move
        if promo_left > 0:
            pawn_rank = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
            pawns = board.pawns & board.occupied_co[board.turn] & pawn_rank
            if pawns:
                promo = _safe_promotion(board)
                if promo is not None:
                    out.append({"kind": "promotion", "history": list(history),
                                "target": promo, "weight": 1.0})
                    promo_left -= 1

        board.push(move)
        history.append(uci)
    return out


def _safe_promotion(board: chess.Board) -> Optional[str]:
    """A queen promotion whose destination cannot be captured immediately
    (or that gives check), for the side to move."""
    opp = not board.turn
    for m in board.legal_moves:
        if not m.promotion or m.promotion != chess.QUEEN:
            continue
        board.push(m)
        safe = board.is_check() or not board.is_attacked_by(opp, m.to_square)
        board.pop()
        if safe:
            return m.uci()
    return None


# ---------------------------------------------------------------------------
# Synthetic endgame demonstrations (engine vs engine from generated starts)
# ---------------------------------------------------------------------------

def generate_endgame_examples(n_games: int, level: int, seed: int,
                              max_plies: int = 160) -> List[dict]:
    """Engine-vs-engine games from synthetic KQK/KRK starts; keep the strong
    side's decisions from games the strong side won. Histories start at the
    synthetic position (BOS + moves), so they teach endgame move patterns."""
    rng = random.Random(f"endgen-{seed}")
    out: List[dict] = []
    for i in range(n_games):
        etype = "kqk" if i % 2 == 0 else "krk"
        board = generate_endgame_position(random.Random(f"endpos-{seed}-{i}"), etype)
        start_fen = board.fen()
        white = FastEngine(level=level, seed=f"endw-{seed}-{i}")
        black = FastEngine(level=level, seed=f"endb-{seed}-{i}")
        history: List[str] = []
        winner = None
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.termination.name.lower() == "checkmate":
                    winner = "white" if outcome.winner == chess.WHITE else "black"
                break
            if board.ply() >= max_plies:
                break
            engine = white if board.turn == chess.WHITE else black
            mv = engine.best_move(board)
            if mv is None:
                break
            history.append(mv.uci())
            board.push(mv)
        if winner != "white":
            continue
        # strong side = white (the generator gives White the Q/R)
        replayed = chess.Board(start_fen)
        for j, uci in enumerate(history):
            if replayed.turn == chess.WHITE:
                out.append({"kind": "endgame", "history": history[:j],
                            "target": uci, "weight": 1.0,
                            "start_fen": start_fen})
            replayed.push_uci(uci)
    return out


# ---------------------------------------------------------------------------
# Mix builder
# ---------------------------------------------------------------------------

def _elo(row: dict, key: str) -> Optional[int]:
    """Robust Elo field read: ints, floats, numeric strings; None otherwise."""
    value = row.get(key)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_games(dataset: str, n_games: int, min_elo: int = 0) -> List[str]:
    """First ``n_games`` games (moves strings) from a converted JSONL corpus.

    With ``min_elo > 0``, only games where BOTH players are rated >= min_elo
    are kept (scanning until n_games such games are collected or EOF). Games
    with missing/unparseable Elo fields are skipped when filtering.
    """
    opener = gzip.open if str(dataset).endswith(".gz") else open
    games: List[str] = []
    with opener(dataset, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if min_elo > 0:
                white, black = _elo(row, "WhiteElo"), _elo(row, "BlackElo")
                if white is None or black is None or \
                        white < min_elo or black < min_elo:
                    continue
            games.append(row["moves"])
            if len(games) >= n_games:
                break
    return games


def _mine_pool(games: List[str], cfg: dict, caps: Dict[str, int],
               workers: int, label: str, progress: bool = True) -> Tuple[Counter, List[dict]]:
    """Mine examples from a list of games with per-kind caps (deterministic)."""
    counts: Counter = Counter()
    examples: List[dict] = []
    ctx = get_context("spawn")
    started = time.monotonic()
    prog = ProgressLine(min_interval=2.0) if progress else None
    block = max(workers * 8, 64)
    with ctx.Pool(workers) as pool:
        for start in range(0, len(games), block):
            if all(counts[k] >= c for k, c in caps.items()):
                break
            chunk = games[start:start + block]
            tasks = [(start + i, mv, cfg) for i, mv in enumerate(chunk)]
            for res in pool.imap(mine_game, tasks, chunksize=1):
                for ex in res:
                    k = ex["kind"]
                    if k in caps and counts[k] < caps[k]:
                        counts[k] += 1
                        examples.append(ex)
            if prog is not None:
                done = sum(min(counts[k], c) for k, c in caps.items())
                total = sum(caps.values())
                prog.show(f"{label}: games {start + len(chunk)}/{len(games)} | "
                          f"{done}/{total} examples | {prog.eta(done, total)}")
    if prog is not None:
        prog.close()
    return counts, examples


def build_mix(dataset: str, out: str, *, n_games: int = 300_000,
              workers: int = 8, seed: int = 0,
              hq_dataset: str = "data/lichess_uci_201412.jsonl.gz",
              hq_games: int = 300_000, min_elo: int = 2200,
              max_replay: int = 300_000, max_capture: int = 150_000,
              max_mate1: int = 100_000, max_hang: int = 150_000,
              max_endgame: int = 100_000, max_promotion: int = 20_000,
              endgame_demos: int = 4000, endgame_level: int = 3,
              hang_tries: int = 8, force: bool = False,
              progress: bool = True) -> Dict[str, int]:
    """Build the SFT mix; returns per-kind example counts.

    ``replay`` examples come from the model's own training corpus (first
    ``n_games`` games); tactical examples (capture, mate1, hang, endgame,
    promotion) come from the Elo-filtered HQ dump (``hq_dataset``, both
    players >= ``min_elo``, first ``hq_games`` such games). Falls back to the
    training corpus for everything if the HQ file is missing. Caps are
    applied per kind in game order (deterministic); the output file is
    skipped if it already exists unless ``force``.
    """
    out_path = Path(out)
    if out_path.exists() and not force:
        with open(out_path, "r", encoding="utf-8") as f:
            counts = Counter(json.loads(l)["kind"] for l in f if l.strip())
        print(f"✅ {out_path} already exists ({sum(counts.values())} rows), "
              f"skipping (use --force to rebuild)")
        return dict(counts)

    cfg = {"seed": seed, "hang_tries": hang_tries,
           "endgame_per_game": 4, "promotion_per_game": 2}
    counts: Counter = Counter()
    examples: List[dict] = []
    started = time.monotonic()

    # pass A: replay examples from the model's own (seen) corpus
    games = load_games(dataset, n_games)
    print(f"Loaded {len(games)} games from {dataset} (replay source)")
    c, ex = _mine_pool(games, cfg, {"replay": max_replay}, workers,
                       "replay mining", progress)
    counts.update(c)
    examples.extend(ex)

    # pass B: tactical examples from Elo-filtered high-quality games
    tactical_caps = {"capture": max_capture, "mate1": max_mate1,
                     "hang": max_hang, "endgame": max_endgame,
                     "promotion": max_promotion}
    hq = Path(hq_dataset)
    hq_list: List[str] = []
    if hq.exists():
        hq_list = load_games(str(hq), hq_games, min_elo=min_elo)
        print(f"Loaded {len(hq_list)} games >= {min_elo} Elo from {hq} "
              f"(tactical source)")
        if not hq_list and min_elo > 0:
            print(f"⚠️ 0 games passed the >= {min_elo} Elo filter (missing or "
                  f"unparseable Elo fields?): retrying unfiltered")
            hq_list = load_games(str(hq), hq_games, min_elo=0)
            print(f"Loaded {len(hq_list)} unfiltered games from {hq}")
    else:
        print(f"⚠️ HQ dataset {hq} not found")
    if not hq_list:
        print(f"⚠️ falling back to the training corpus for tactical examples")
        hq_list = games
    c, ex = _mine_pool(hq_list, cfg, tactical_caps, workers,
                       "tactical mining", progress)
    counts.update(c)
    examples.extend(ex)

    if endgame_demos > 0 and counts["endgame"] < max_endgame:
        print(f"Generating {endgame_demos} synthetic endgame demonstrations "
              f"(engine level {endgame_level})...")
        demos = generate_endgame_examples(endgame_demos, endgame_level, seed)
        for ex in demos:
            if counts["endgame"] >= max_endgame:
                break
            counts["endgame"] += 1
            examples.append(ex)

    jsonl_write(out_path, examples)
    print(f"✅ wrote {len(examples)} examples to {out_path} "
          f"({time.monotonic() - started:.0f}s): {dict(counts)}")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 1 SFT mix")
    parser.add_argument("--dataset", default="data/lichess_train_2025-01_1M.jsonl.gz")
    parser.add_argument("--hq-dataset", default="data/lichess_uci_201412.jsonl.gz")
    parser.add_argument("--hq-games", type=int, default=300_000)
    parser.add_argument("--min-elo", type=int, default=2200)
    parser.add_argument("--out", default="data/sft_mix.jsonl")
    parser.add_argument("--n-games", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-replay", type=int, default=300_000)
    parser.add_argument("--max-capture", type=int, default=150_000)
    parser.add_argument("--max-mate1", type=int, default=100_000)
    parser.add_argument("--max-hang", type=int, default=150_000)
    parser.add_argument("--max-endgame", type=int, default=100_000)
    parser.add_argument("--max-promotion", type=int, default=20_000)
    parser.add_argument("--endgame-demos", type=int, default=4000)
    parser.add_argument("--endgame-level", type=int, default=3)
    parser.add_argument("--hang-tries", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_mix(args.dataset, args.out, n_games=args.n_games, workers=args.workers,
              seed=args.seed, hq_dataset=args.hq_dataset, hq_games=args.hq_games,
              min_elo=args.min_elo, max_replay=args.max_replay,
              max_capture=args.max_capture, max_mate1=args.max_mate1,
              max_hang=args.max_hang, max_endgame=args.max_endgame,
              max_promotion=args.max_promotion, endgame_demos=args.endgame_demos,
              endgame_level=args.endgame_level, hang_tries=args.hang_tries,
              force=args.force)


if __name__ == "__main__":
    main()
