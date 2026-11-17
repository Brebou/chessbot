"""Probes for evaluating ChessGPT model capabilities (C1-C14).

Each probe tests a specific capability by running the model on a corpus
and computing relevant metrics.

Features:
- Batch processing for speed (sort by length, pad, process in batches)
- GPU-ready (device parameter)
- Game-playing probes (P5, P6) run all matches in lockstep rounds: model moves
  are batched across matches, engine moves run on a CPU process pool
- Computes all metrics from the spec (C1-C14)
- Saves results in JSON format

Usage:
    uv run diag/probes.py --model base_model --corpus results/corpus --out results/probes --device cuda
    uv run diag/probes.py --model base_model --corpus results/corpus --out results/probes \
        --device cuda --lichess data/lichess.jsonl.gz --split-index 940000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

import chess
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessTokenizer, load_model

try:
    from .common import (MAX_PLIES, ProgressLine, ProbePlayer, jsonl_read,
                         material_diff, material_value, replay)
    from .fast_engine import FastEngine, engine_best_move_task
    from .fast_probe import FastProbePlayer
except ImportError:
    from diag.common import (MAX_PLIES, ProgressLine, ProbePlayer, jsonl_read,
                             material_diff, material_value, replay)
    from diag.fast_engine import FastEngine, engine_best_move_task
    from diag.fast_probe import FastProbePlayer


# Longest move history the model can condition on (BOS + moves must fit n_ctx).
MODEL_MAX_MOVES = 255

_P7_BLOCK_LINES = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result from a single probe."""
    name: str
    n_positions: int
    metrics: Dict[str, Any]
    by_bucket: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pool_init():
    """Pool worker init: keep worker processes single-threaded."""
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _score_for(result: str, color: chess.Color) -> float:
    """Points (0, 0.5, 1) earned by ``color`` for the game result."""
    if result == "1-0":
        return 1.0 if color == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if color == chess.BLACK else 0.0
    return 0.5


def _truncate_history(history: List[str], max_moves: int = MAX_PLIES) -> List[str]:
    """Keep the last ``max_moves`` moves, dropping an even number of moves from
    the front so the parity (and thus the move-order structure) is preserved."""
    if len(history) <= max_moves:
        return list(history)
    excess = len(history) - max_moves
    drop = excess + (excess & 1)
    return list(history[drop:])


def _model_history(board: chess.Board) -> List[str]:
    """History to feed the model for ``board``, truncated to fit the context."""
    return _truncate_history([m.uci() for m in board.move_stack], MODEL_MAX_MOVES)


def _failure_cause(termination: str) -> str:
    if termination == "stalemate":
        return "stalemate"
    if "repetition" in termination:
        return "repetition"
    if termination in ("fifty_move", "seventyfive_move"):
        return "fifty_move"
    if termination == "max_plies":
        return "max_plies"
    if termination == "checkmate":
        return "checkmate"
    if termination == "insufficient_material":
        return "insufficient_material"
    return termination or "other"


def _p7_block_worker(args):
    """Sample next-move prediction items from a block of Lichess game lines."""
    start_idx, lines, positions_per_game = args
    out = []
    for i, line in enumerate(lines):
        game_index = start_idx + i
        try:
            game = json.loads(line)
            moves = game.get("moves", "")
            if not moves:
                continue
            moves = moves.lower().split()
            max_ply = min(len(moves) - 1, MODEL_MAX_MOVES - 1)
            if max_ply <= 8:
                continue
            rng = random.Random(f"p7-{game_index}")
            count = min(positions_per_game, max_ply - 8)
            plies = rng.sample(range(8, max_ply), count)
            white_elo = game.get("WhiteElo")
            black_elo = game.get("BlackElo")
            for ply in plies:
                elo = white_elo if ply % 2 == 0 else black_elo
                try:
                    elo = int(elo) if elo not in (None, "?", "") else None
                except (TypeError, ValueError):
                    elo = None
                out.append({
                    "game_index": game_index,
                    "ply": ply,
                    "history": moves[:ply],
                    "next_move": moves[ply],
                    "elo": elo,
                })
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Probe implementations
# ---------------------------------------------------------------------------

class ProbeRunner:
    """Runs all probes on a set of corpora.

    Features:
    - Batch processing for speed
    - GPU support
    - Progress tracking
    - Comprehensive metrics for C1-C14
    """

    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 64,
                 workers: int = 1):
        """Initialize the probe runner.

        Args:
            model_path: Path to model directory
            device: "cpu" or "cuda"
            batch_size: Batch size for inference (auto-adjusted for GPU)
            workers: Worker processes for engine moves (game-playing probes)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.workers = max(1, workers)

        # Auto-adjust batch size for GPU
        if device == "cuda" or (device is not None and "cuda" in device):
            # For GPU, use large batch sizes to maximize utilization
            self.batch_size = max(batch_size, 1024)
        else:
            self.batch_size = batch_size

        # Load model and tokenizer
        self.model = load_model(str(model_path), device=device)
        self.tokenizer = ChessTokenizer()
        self.probe_player = ProbePlayer(self.model, self.tokenizer,
                                        batch_size=self.batch_size, device=device)
        # Fast last-position-only inference: bit-identical to
        # ProbePlayer._batch_logits but avoids materializing (B, T, vocab).
        self.fast_player = FastProbePlayer(self.model, self.tokenizer,
                                           batch_size=self.batch_size, device=device)

        # Built-in engine for analysis (optimized reimplementation, kept
        # outside chess_hw; move choices match chess_hw.engine.Engine)
        self.engine = FastEngine(level=2, seed=42)

        print(f"ProbeRunner initialized with device={device}, batch_size={self.batch_size}")

    def _make_engine_pool(self):
        if self.workers <= 1:
            return None
        from multiprocessing import get_context
        ctx = get_context("spawn")
        return ctx.Pool(self.workers, initializer=_pool_init)

    # -- shared game-playing machinery --------------------------------------

    def _play_matches(self, matches: List[dict], pool=None, max_plies: int = MAX_PLIES) -> None:
        """Play model-vs-engine matches in lockstep rounds, in place.

        Each match dict must provide: "board" (or "history"/"fen" to build it),
        "model_color" ("white"/"black"), "level" (engine level) and "seed".
        Fills in "result", "termination", "start_ply" and "plies" (played plies).

        Model moves use the same semantics as chess_hw.play.ModelPlayer (greedy
        top-3, first legal, seeded random fallback), batched across matches.
        Model inference always uses a history truncated to fit the 256-token
        context (an even number of moves is dropped from the front).
        """
        for m in matches:
            if "board" not in m:
                history = m.get("history") or []
                m["board"] = replay(history) if history else chess.Board(m["fen"])
            m["start_ply"] = m["board"].ply()
            m["result"] = "*"
            m["termination"] = "unknown"

        active = list(range(len(matches)))
        prog = ProgressLine(min_interval=2.0)
        while active:
            plies = sum(matches[i]["board"].ply() - matches[i]["start_ply"]
                        for i in active)
            prog.show(
                f"Matches: {len(matches) - len(active)}/{len(matches)} finished "
                f"| {plies:,} plies played | {prog.elapsed:.0f}s elapsed"
            )
            model_idx, engine_idx = [], []
            for i in active:
                m = matches[i]
                b = m["board"]
                outcome = b.outcome(claim_draw=True)
                if outcome is not None:
                    m["result"] = outcome.result()
                    m["termination"] = outcome.termination.name.lower()
                    continue
                if b.ply() - m["start_ply"] >= max_plies:
                    m["result"], m["termination"] = "1/2-1/2", "max_plies"
                    continue
                if (b.turn == chess.WHITE) == (m["model_color"] == "white"):
                    model_idx.append(i)
                else:
                    engine_idx.append(i)

            engine_results = None
            if engine_idx:
                tasks = [
                    (matches[i]["board"].fen(), matches[i]["level"],
                     f"engine-{matches[i]['seed']}")
                    for i in engine_idx
                ]
                if pool is not None:
                    chunksize = max(1, len(tasks) // (self.workers * 4))
                    engine_results = pool.imap(engine_best_move_task, tasks,
                                               chunksize=chunksize)
                else:
                    engine_results = [engine_best_move_task(t) for t in tasks]

            bs = self.fast_player.batch_size
            for c in range(0, len(model_idx), bs):
                chunk = model_idx[c:c + bs]
                boards = [matches[i]["board"] for i in chunk]
                histories = [_model_history(b) for b in boards]
                choices = self.fast_player.choose_from_histories(histories, boards)
                for i, choice in zip(chunk, choices):
                    m = matches[i]
                    b = m["board"]
                    move = choice.move
                    if move is None:
                        legal = list(b.legal_moves)
                        if not legal:
                            continue
                        move = random.Random(
                            f"fallback-{m['seed']}-{b.ply()}"
                        ).choice(legal)
                    b.push(move)

            if engine_results is not None:
                for i, uci in zip(engine_idx, list(engine_results)):
                    if uci:
                        matches[i]["board"].push_uci(uci)

            active = [i for i in model_idx + engine_idx if matches[i]["result"] == "*"]

        prog.close()
        for m in matches:
            m["plies"] = m["board"].ply() - m["start_ply"]

    # -- P1 -------------------------------------------------------------------

    def run_probe_p1(self, corpus_path: Path, out_dir: Path) -> ProbeResult:
        """P1: Legality, mass on legal moves (C1, C2, C3, C14)

        Tests:
        - C1: Legal move rates (top-1, top-3, fallback)
        - C2: Taxonomy of illegal moves
        - C3: Mass on legal moves
        - C14: Stability at end of context

        Each position is processed exactly once; the per-position stats are
        then accumulated into the "total" metrics and every bucket (the
        previous version re-ran inference per bucket and never filled the
        "total" entry).
        """
        print("\n=== Running P1 (Legality) ===")
        start_time = time.time()

        positions = jsonl_read(corpus_path)
        print(f"Loaded {len(positions)} positions from {corpus_path}")

        items = []
        for pos in positions:
            if pos.get("ply", 0) < 9:
                continue
            try:
                board = replay(pos["history"])
            except (ValueError, AssertionError) as e:
                print(f"  Skipping illegal history: {e}")
                continue
            items.append((pos, board))

        def bucket_names(pos: dict, board: chess.Board) -> List[str]:
            names = []
            ply = pos.get("ply", 0)
            if ply < 16:
                names.append("opening")
            elif ply < 31:
                names.append("early_mid")
            elif ply < 61:
                names.append("mid")
            elif ply < 121:
                names.append("late_mid")
            else:
                names.append("endgame")
            names.append("check_check" if board.is_check() else "check_no_check")
            has_capture = any(board.is_capture(m) for m in board.legal_moves)
            names.append("capture_capture" if has_capture else "capture_no_capture")
            n_legal = board.legal_moves.count()
            if n_legal < 5:
                names.append("nlegal_<5")
            elif n_legal < 21:
                names.append("nlegal_5-20")
            elif n_legal < 41:
                names.append("nlegal_20-40")
            else:
                names.append("nlegal_>40")
            return names

        fields = ["count", "top1_legal", "top3_legal", "fallback",
                  "mass_legal", "mass_top1", "mass_top3", "mass_top10"]

        def new_acc():
            return {f: 0 for f in fields}

        total_acc = new_acc()
        bucket_accs = defaultdict(new_acc)

        bs = self.fast_player.batch_size
        prog = ProgressLine()
        for c in range(0, len(items), bs):
            chunk = items[c:c + bs]
            logits_batch = self.fast_player.last_logits(
                [pos["history"] for pos, _ in chunk])
            done = min(c + bs, len(items))
            prog.show(
                f"P1: {done}/{len(items)} positions | "
                f"{prog.rate(done):.1f}/s | {prog.eta(done, len(items))}"
            )

            for (pos, board), logits in zip(chunk, logits_batch):
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()

                top_indices = np.argsort(probs)[::-1][:10]
                top_tokens = [self.tokenizer.id_to_token[int(idx)] for idx in top_indices[:3]]

                top1_legal = top3_legal = False
                for j, token in enumerate(top_tokens):
                    try:
                        board.parse_uci(token)
                        if j == 0:
                            top1_legal = True
                        top3_legal = True
                        break
                    except ValueError:
                        continue

                # Mass on legal moves. Tokens are UCI strings, so membership
                # in the legal-move UCI set is exact and much faster than
                # parse_uci on every vocabulary token.
                legal_ucis = {m.uci() for m in board.legal_moves}
                legal_mask = np.zeros(len(probs), dtype=bool)
                for token, idx in self.tokenizer.token_to_id.items():
                    if idx < len(probs) and token in legal_ucis:
                        legal_mask[idx] = True

                s = {
                    "count": 1,
                    "top1_legal": int(top1_legal),
                    "top3_legal": int(top3_legal),
                    "fallback": int(not top3_legal),
                    "mass_legal": float(probs[legal_mask].sum()) if legal_mask.any() else 0.0,
                    "mass_top1": float(probs[top_indices[0]]),
                    "mass_top3": float(probs[top_indices[:3]].sum()),
                    "mass_top10": float(probs[top_indices[:10]].sum()),
                }

                names = bucket_names(pos, board)
                for f in fields:
                    total_acc[f] += s[f]
                    for name in names:
                        bucket_accs[name][f] += s[f]

        prog.close()

        def finalize(acc: dict) -> dict:
            count = acc["count"]
            if count == 0:
                return {}
            return {
                "top1_legal_pct": acc["top1_legal"] / count * 100,
                "top3_legal_pct": acc["top3_legal"] / count * 100,
                "fallback_pct": acc["fallback"] / count * 100,
                "mass_legal_avg": acc["mass_legal"] / count,
                "mass_top1_avg": acc["mass_top1"] / count,
                "mass_top3_avg": acc["mass_top3"] / count,
                "mass_top10_avg": acc["mass_top10"] / count,
            }

        all_metrics = {"total": finalize(total_acc)}
        bucket_metrics = {name: finalize(acc) for name, acc in sorted(bucket_accs.items())}

        elapsed = time.time() - start_time
        print(f"P1 completed in {elapsed:.2f}s")

        return ProbeResult(
            name="P1",
            n_positions=len(positions),
            metrics=all_metrics,
            by_bucket=bucket_metrics,
            config={"model": str(self.model_path), "device": self.device}
        )

    # -- P2 -------------------------------------------------------------------

    def run_probe_p2(self, corpus_path: Path, out_dir: Path) -> ProbeResult:
        """P2: Mate in 1 (C5)

        Tests:
        - C5: Mate in 1 detection (top-1 and top-3)
        """
        print("\n=== Running P2 (Mate in 1) ===")
        start_time = time.time()

        positions = jsonl_read(corpus_path)
        print(f"Loaded {len(positions)} positions from {corpus_path}")

        items = []
        for pos in positions:
            mates = pos.get("mates") or []
            if not mates:
                continue
            history = pos.get("history") or pos.get("moves") or []
            try:
                board = replay(history)
            except (ValueError, AssertionError) as e:
                print(f"  Skipping illegal history: {e}")
                continue
            items.append((history, board, set(mates)))

        top1_mate = 0
        top3_mate = 0
        check_not_mate = 0

        bs = self.fast_player.batch_size
        prog = ProgressLine()
        for c in range(0, len(items), bs):
            chunk = items[c:c + bs]
            logits_batch = self.fast_player.last_logits([h for h, _, _ in chunk])
            done = min(c + bs, len(items))
            prog.show(
                f"P2: {done}/{len(items)} positions | "
                f"{prog.rate(done):.1f}/s | {prog.eta(done, len(items))}"
            )

            for (history, board, mates), logits in zip(chunk, logits_batch):
                top_indices = np.argsort(logits)[::-1][:3]
                top_tokens = [self.tokenizer.id_to_token[int(idx)] for idx in top_indices]

                if top_tokens[0] in mates:
                    top1_mate += 1
                if any(t in mates for t in top_tokens):
                    top3_mate += 1

                if board.is_check():
                    for token in top_tokens:
                        try:
                            move = board.parse_uci(token)
                        except ValueError:
                            continue
                        board_copy = board.copy()
                        board_copy.push(move)
                        if board_copy.is_check() and not board_copy.is_checkmate():
                            check_not_mate += 1
                            break

        prog.close()
        total = len(items)
        elapsed = time.time() - start_time
        print(f"✅ P2 completed in {elapsed:.2f}s")

        return ProbeResult(
            name="P2",
            n_positions=total,
            metrics={
                "top1_mate_pct": top1_mate / total * 100 if total > 0 else 0.0,
                "top3_mate_pct": top3_mate / total * 100 if total > 0 else 0.0,
                "check_not_mate_count": check_not_mate,
            },
            config={"model": str(self.model_path), "device": self.device}
        )

    # -- P3 -------------------------------------------------------------------

    def run_probe_p3(self, corpus_path: Path, out_dir: Path) -> ProbeResult:
        """P3: Captures (C6)

        Chosen-move-based semantics: a capture is "taken" when the model's
        greedy move (first legal among top-3) equals the labeled capture.
        Denominators are the number of winning/losing/free capture positions.
        """
        print("\n=== Running P3 (Captures) ===")
        start_time = time.time()

        positions = jsonl_read(corpus_path)
        print(f"Loaded {len(positions)} positions from {corpus_path}")

        items = []
        for pos in positions:
            capture_uci = pos.get("capture_move")
            if not capture_uci:
                continue
            try:
                board = replay(pos["history"])
                move = board.parse_uci(capture_uci)
            except (ValueError, AssertionError) as e:
                print(f"  Skipping invalid position: {e}")
                continue
            items.append({
                "history": pos["history"],
                "board": board,
                "move": move,
                "see": float(pos.get("see", 0)),
            })

        if not items:
            print("✅ P3 skipped (empty corpus)")
            return ProbeResult(
                name="P3",
                n_positions=0,
                metrics={"status": "empty_corpus"},
                config={"model": str(self.model_path), "device": self.device}
            )

        choices = []
        bs = self.fast_player.batch_size
        prog = ProgressLine()
        for c in range(0, len(items), bs):
            chunk = items[c:c + bs]
            choices.extend(self.fast_player.choose_from_histories(
                [it["history"] for it in chunk], [it["board"] for it in chunk]))
            done = min(c + bs, len(items))
            prog.show(
                f"P3: {done}/{len(items)} positions | "
                f"{prog.rate(done):.1f}/s | {prog.eta(done, len(items))}"
            )

        prog.close()

        n_winning = n_losing = n_free = 0
        take_winning = take_losing = take_free = 0

        for it, choice in zip(items, choices):
            chosen = choice.move.uci() if choice.move else None
            if chosen is None:
                continue
            target = it["move"].uci()
            if it["see"] > 0:
                n_winning += 1
                if chosen == target:
                    take_winning += 1
            if it["see"] < 0:
                n_losing += 1
                if chosen == target:
                    take_losing += 1
            if it["see"] >= 300:
                n_free += 1
                if chosen == target:
                    take_free += 1

        elapsed = time.time() - start_time
        print(f"✅ P3 completed in {elapsed:.2f}s")

        return ProbeResult(
            name="P3",
            n_positions=len(items),
            metrics={
                "n": len(items),
                "n_winning": n_winning,
                "take_winning_capture_pct": take_winning / n_winning * 100 if n_winning else 0.0,
                "n_losing": n_losing,
                "take_losing_capture_pct": take_losing / n_losing * 100 if n_losing else 0.0,
                "n_free": n_free,
                "take_free_piece_pct": take_free / n_free * 100 if n_free else 0.0,
            },
            config={"model": str(self.model_path), "device": self.device}
        )

    # -- P4 -------------------------------------------------------------------

    def run_probe_p4(self, corpus_path: Path, out_dir: Path, pool=None) -> ProbeResult:
        """P4: Anti-hang (C7)

        Plays the model's greedy move, lets FastEngine (level 2) reply, and
        measures the model's material loss over the exchange.
        """
        print("\n=== Running P4 (Anti-hang) ===")
        start_time = time.time()

        positions = jsonl_read(corpus_path)
        print(f"Loaded {len(positions)} positions from {corpus_path}")

        hang_1pt = 0
        hang_3pt = 0
        total = 0
        block_size = 512

        prog = ProgressLine()
        for block_start in range(0, len(positions), block_size):
            block = positions[block_start:block_start + block_size]
            done = min(block_start + block_size, len(positions))
            prog.show(
                f"P4: {done}/{len(positions)} positions | "
                f"{prog.rate(done):.1f}/s | {prog.eta(done, len(positions))}"
            )
            boards, histories, idxs = [], [], []
            for i, pos in enumerate(block):
                try:
                    boards.append(replay(pos["history"]))
                    histories.append(pos["history"])
                    idxs.append(block_start + i)
                except (ValueError, AssertionError):
                    continue
            if not boards:
                continue

            choices = self.fast_player.choose_from_histories(histories, boards)

            after_boards, tasks = [], []
            for idx, board, choice in zip(idxs, boards, choices):
                if choice.move is None:
                    continue
                total += 1
                after = board.copy()
                after.push(choice.move)
                after_boards.append(after)
                tasks.append((after.fen(), 2, f"p4-{idx}"))

            if not tasks:
                continue

            if pool is not None:
                engine_moves = list(pool.imap(engine_best_move_task, tasks,
                                              chunksize=max(1, len(tasks) // (self.workers * 4))))
            else:
                engine_moves = [engine_best_move_task(t) for t in tasks]

            for board, after, engine_uci in zip(boards, after_boards, engine_moves):
                if engine_uci:
                    try:
                        after.push_uci(engine_uci)
                    except ValueError:
                        pass
                color = board.turn
                mat_loss = material_value(board, color) - material_value(after, color)
                if mat_loss >= 100:
                    hang_1pt += 1
                if mat_loss >= 300:
                    hang_3pt += 1

        prog.close()
        elapsed = time.time() - start_time
        print(f"✅ P4 completed in {elapsed:.2f}s")

        return ProbeResult(
            name="P4",
            n_positions=total,
            metrics={
                "hang_1pt_pct": hang_1pt / total * 100 if total > 0 else 0.0,
                "hang_3pt_pct": hang_3pt / total * 100 if total > 0 else 0.0,
            },
            config={"model": str(self.model_path), "device": self.device,
                    "engine": "FastEngine level 2"}
        )

    # -- P5 -------------------------------------------------------------------

    def run_probe_p5(self, corpus_path: Path, out_dir: Path, levels=(1, 2),
                     pool=None) -> ProbeResult:
        """P5: Conversion (C8)

        Plays model vs engine from each recorded conversion position and
        measures win/draw/loss rate per material-advantage threshold, plus
        failure causes (repetition, stalemate, 250 plies, material lost...).
        """
        print("\n=== Running P5 (Conversion) ===")
        start_time = time.time()

        positions = jsonl_read(corpus_path)
        print(f"Loaded {len(positions)} positions from {corpus_path}")

        if not positions:
            print("✅ P5 skipped (empty corpus)")
            return ProbeResult(
                name="P5",
                n_positions=0,
                metrics={"status": "empty_corpus"},
                config={"model": str(self.model_path), "device": self.device}
            )

        matches = []
        for idx, pos in enumerate(positions):
            for level in levels:
                matches.append({
                    "history": pos.get("history") or [],
                    "fen": pos.get("fen"),
                    "model_color": pos.get("model_color", "white"),
                    "level": level,
                    "seed": f"p5-{idx}-{level}",
                    "threshold": pos.get("threshold"),
                })

        self._play_matches(matches, pool=pool)

        for m in matches:
            color = chess.WHITE if m["model_color"] == "white" else chess.BLACK
            m["score"] = _score_for(m["result"], color)
            m["final_material_diff"] = material_diff(m["board"], color)

        def aggregate(subset: List[dict]) -> Dict[str, Any]:
            groups = defaultdict(list)
            for m in subset:
                groups[str(m["threshold"])].append(m)
            out = {}
            for thr, ms in sorted(groups.items(), key=lambda kv: int(kv[0])):
                n = len(ms)
                wins = sum(1 for m in ms if m["score"] == 1.0)
                draws = sum(1 for m in ms if m["score"] == 0.5)
                losses = sum(1 for m in ms if m["score"] == 0.0)
                failure_modes = Counter()
                for m in ms:
                    cause = _failure_cause(m["termination"])
                    failure_modes[cause] += 1
                    if m["score"] == 0.0 and cause == "checkmate":
                        failure_modes["mated"] += 1
                    if m["final_material_diff"] is not None and m["final_material_diff"] < 0:
                        failure_modes["material_lost"] += 1
                out[thr] = {
                    "n": n,
                    "win_rate": wins / n,
                    "draw_rate": draws / n,
                    "loss_rate": losses / n,
                    "avg_plies": sum(m["plies"] for m in ms) / n,
                    "failure_modes": dict(failure_modes),
                }
            return out

        metrics = {}
        for level in levels:
            metrics[f"level_{level}"] = aggregate(
                [m for m in matches if m["level"] == level])

        by_bucket = {
            "model_white": aggregate([m for m in matches if m["model_color"] == "white"]),
            "model_black": aggregate([m for m in matches if m["model_color"] == "black"]),
        }

        elapsed = time.time() - start_time
        print(f"✅ P5 completed in {elapsed:.2f}s ({len(matches)} matches)")

        return ProbeResult(
            name="P5",
            n_positions=len(positions),
            metrics=metrics,
            by_bucket=by_bucket,
            config={"model": str(self.model_path), "device": self.device,
                    "levels": list(levels), "max_plies": MAX_PLIES}
        )

    # -- P6 -------------------------------------------------------------------

    def run_probe_p6(self, corpus_path: Path, out_dir: Path, levels=(1, 0),
                     pool=None) -> ProbeResult:
        """P6: Endgames (C9)

        Plays model (with KQ/KR) vs engine (levels 1 and 0) from each endgame
        position. Measures mate rates (<=50 and <=100 plies), average plies to
        mate, and failure modes. Histories are truncated to fit the context
        while preserving move parity.
        """
        print("\n=== Running P6 (Endgames) ===")
        start_time = time.time()

        positions = jsonl_read(corpus_path)
        print(f"Loaded {len(positions)} positions from {corpus_path}")

        if not positions:
            print("✅ P6 skipped (empty corpus)")
            return ProbeResult(
                name="P6",
                n_positions=0,
                metrics={"status": "empty_corpus"},
                config={"model": str(self.model_path), "device": self.device}
            )

        matches = []
        for idx, pos in enumerate(positions):
            for level in levels:
                matches.append({
                    "history": _truncate_history(pos.get("history") or [], MAX_PLIES),
                    "fen": pos.get("fen"),
                    "model_color": pos.get("model_color", "white"),
                    "level": level,
                    "seed": f"p6-{idx}-{level}",
                    "type": pos.get("type", "unknown"),
                })

        self._play_matches(matches, pool=pool)

        for m in matches:
            color = chess.WHITE if m["model_color"] == "white" else chess.BLACK
            m["score"] = _score_for(m["result"], color)

        def aggregate(subset: List[dict]) -> Dict[str, Any]:
            groups = defaultdict(list)
            for m in subset:
                groups[m["type"]].append(m)
            out = {}
            for typ, ms in sorted(groups.items()):
                n = len(ms)
                mates = [m for m in ms if m["termination"] == "checkmate" and m["score"] == 1.0]
                mate_plies = [m["plies"] for m in mates]
                failure_modes = Counter(_failure_cause(m["termination"]) for m in ms)
                for m in ms:
                    if m["termination"] == "checkmate" and m["score"] == 0.0:
                        failure_modes["got_mated"] += 1
                out[typ] = {
                    "n": n,
                    "mate_rate": len(mates) / n,
                    "mate_le50_rate": sum(1 for p in mate_plies if p <= 50) / n,
                    "mate_le100_rate": sum(1 for p in mate_plies if p <= 100) / n,
                    "avg_plies_to_mate": (sum(mate_plies) / len(mate_plies)) if mate_plies else None,
                    "failure_modes": dict(failure_modes),
                }
            return out

        metrics = {}
        for level in levels:
            metrics[f"level_{level}"] = aggregate(
                [m for m in matches if m["level"] == level])

        by_bucket = {}
        for typ in sorted({m["type"] for m in matches}):
            by_bucket[typ] = aggregate([m for m in matches if m["type"] == typ])

        elapsed = time.time() - start_time
        print(f"✅ P6 completed in {elapsed:.2f}s ({len(matches)} matches)")

        return ProbeResult(
            name="P6",
            n_positions=len(positions),
            metrics=metrics,
            by_bucket=by_bucket,
            config={"model": str(self.model_path), "device": self.device,
                    "levels": list(levels), "max_plies": MAX_PLIES}
        )

    # -- P6b ------------------------------------------------------------------

    def run_probe_p6b(self, corpus_path: Path, out_dir: Path) -> ProbeResult:
        """P6b: Promotion (C10)

        Measures P(promote to queen | safe promotion available) and the
        promotion piece distribution when the model promotes.
        """
        print("\n=== Running P6b (Promotion) ===")
        start_time = time.time()

        positions = jsonl_read(corpus_path)
        print(f"Loaded {len(positions)} positions from {corpus_path}")

        if not positions:
            print("✅ P6b skipped (empty corpus)")
            return ProbeResult(
                name="P6b",
                n_positions=0,
                metrics={"status": "empty_corpus"},
                config={"model": str(self.model_path), "device": self.device}
            )

        boards, histories, valid = [], [], []
        for pos in positions:
            try:
                board = replay(pos["history"]) if pos.get("history") else chess.Board(pos["fen"])
            except (ValueError, AssertionError, KeyError) as e:
                print(f"  Skipping invalid position: {e}")
                continue
            boards.append(board)
            histories.append(pos.get("history") or [])
            valid.append(pos)

        choices = []
        bs = self.fast_player.batch_size
        prog = ProgressLine()
        for c in range(0, len(boards), bs):
            choices.extend(self.fast_player.choose_from_histories(
                histories[c:c + bs], boards[c:c + bs]))
            done = min(c + bs, len(boards))
            prog.show(
                f"P6b: {done}/{len(boards)} positions | "
                f"{prog.rate(done):.1f}/s | {prog.eta(done, len(boards))}"
            )

        prog.close()

        n_safe = 0
        queen_when_safe = 0
        promote_when_safe = 0
        n_promoted = 0
        piece_distribution = Counter()

        for pos, choice in zip(valid, choices):
            chosen = choice.move.uci() if choice.move else None
            is_promo = choice.move is not None and choice.move.promotion is not None
            if is_promo:
                n_promoted += 1
                piece_distribution[chosen[-1]] += 1
            safe = pos.get("safe_promotions") or []
            if safe:
                n_safe += 1
                if is_promo:
                    promote_when_safe += 1
                    if chosen.endswith("q"):
                        queen_when_safe += 1

        metrics = {
            "n": len(valid),
            "n_with_safe_promotion": n_safe,
            "queen_rate_when_safe": queen_when_safe / n_safe if n_safe else 0.0,
            "promote_rate_when_safe": promote_when_safe / n_safe if n_safe else 0.0,
            "promote_rate_overall": n_promoted / len(valid) if valid else 0.0,
            "piece_distribution": dict(piece_distribution),
            "piece_distribution_pct": {
                k: v / n_promoted * 100 for k, v in piece_distribution.items()
            } if n_promoted else {},
        }

        elapsed = time.time() - start_time
        print(f"✅ P6b completed in {elapsed:.2f}s")

        return ProbeResult(
            name="P6b",
            n_positions=len(valid),
            metrics=metrics,
            config={"model": str(self.model_path), "device": self.device}
        )

    # -- P7 -------------------------------------------------------------------

    def run_probe_p7(self, lichess_path, out_dir: Path, split_index: int = 940_000,
                     n_unseen: int = 10_000, n_seen: int = 1_000,
                     positions_per_game: int = 1, pool=None) -> ProbeResult:
        """P7: Generality (C13)

        Next-move accuracy (top-1, top-3) on Lichess positions, split into
        seen (game index < split_index, training data) and unseen (>= split).
        Optional accuracy by Elo tranche when ratings are available.
        """
        print("\n=== Running P7 (Generality) ===")
        start_time = time.time()

        path = Path(lichess_path) if lichess_path else None
        if path is None or not path.exists():
            reason = f"Lichess dataset not found: {path}"
            print(f"⚠️ P7 skipped ({reason})")
            return ProbeResult(
                name="P7",
                n_positions=0,
                metrics={"status": "skipped", "reason": reason},
                config={"model": str(self.model_path), "device": self.device,
                        "split_index": split_index}
            )

        open_fn = gzip.open if str(path).endswith(".gz") else open

        local_pool = None
        if pool is None and self.workers > 1:
            print(f"Starting P7 engine pool with {self.workers} workers...")
            local_pool = self._make_engine_pool()
            pool = local_pool

        seen_samples: List[dict] = []
        unseen_samples: List[dict] = []
        max_game_index = -1

        def collect(samples: List[dict], split: int) -> None:
            nonlocal max_game_index
            for s in samples:
                max_game_index = max(max_game_index, s["game_index"])
                if s["game_index"] < split:
                    if len(seen_samples) < n_seen:
                        seen_samples.append(s)
                else:
                    if len(unseen_samples) < n_unseen:
                        unseen_samples.append(s)

        def scan(split: int) -> None:
            nonlocal seen_samples, unseen_samples
            seen_samples, unseen_samples = [], []
            prog = ProgressLine(min_interval=2.0)

            def show(idx: int) -> None:
                prog.show(
                    f"P7 scan: {idx:,} games | "
                    f"seen={len(seen_samples):,}/{n_seen:,} | "
                    f"unseen={len(unseen_samples):,}/{n_unseen:,} | "
                    f"{prog.rate(idx):,.0f} games/s"
                )

            if pool is None:
                start_idx = 0
                with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
                    while not (len(seen_samples) >= n_seen and len(unseen_samples) >= n_unseen):
                        lines = list(islice(f, _P7_BLOCK_LINES))
                        if not lines:
                            break
                        collect(_p7_block_worker((start_idx, lines, positions_per_game)), split)
                        start_idx += len(lines)
                        show(start_idx)
            else:
                window = max(1, self.workers) * 4
                pending = []
                start_idx = 0
                exhausted = False
                with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
                    while True:
                        if len(seen_samples) >= n_seen and len(unseen_samples) >= n_unseen:
                            break
                        while len(pending) < window and not exhausted:
                            lines = list(islice(f, _P7_BLOCK_LINES))
                            if not lines:
                                exhausted = True
                                break
                            pending.append(pool.apply_async(
                                _p7_block_worker, ((start_idx, lines, positions_per_game),)))
                            start_idx += len(lines)
                        if not pending:
                            break
                        collect(pending.pop(0).get(), split)
                        show(start_idx)

            prog.close()
            print(f"  P7 scanned {start_idx:,} games "
                  f"(seen={len(seen_samples):,}, unseen={len(unseen_samples):,})")

        try:
            scan(split_index)
            if not unseen_samples and seen_samples:
                effective_split = int((max_game_index + 1) * 0.9)
                print(f"\u26a0\ufe0f Dataset has {max_game_index + 1} games (< split index "
                      f"{split_index}); using adaptive split at game {effective_split}")
                split_index = effective_split
                scan(split_index)
        finally:
            if local_pool is not None:
                local_pool.terminate()
                local_pool.join()

        def accuracy(samples: List[dict], label: str) -> Dict[str, Any]:
            samples = [s for s in samples if s["next_move"] in self.tokenizer.token_to_id]
            if not samples:
                return {"n": 0, "top1_acc": 0.0, "top3_acc": 0.0}
            histories = [s["history"] for s in samples]
            ids = [self.tokenizer.token_to_id[s["next_move"]] for s in samples]
            bs = self.fast_player.batch_size
            top1 = top3 = 0
            prog = ProgressLine()
            for c in range(0, len(samples), bs):
                logits_all = self.fast_player.last_logits(histories[c:c + bs])
                for logits, tid in zip(logits_all, ids[c:c + bs]):
                    top = np.argsort(logits)[::-1][:3]
                    if int(top[0]) == tid:
                        top1 += 1
                    if tid in top:
                        top3 += 1
                done = min(c + bs, len(samples))
                prog.show(
                    f"{label}: {done:,}/{len(samples):,} positions | "
                    f"{prog.rate(done):.1f}/s | {prog.eta(done, len(samples))}"
                )
            prog.close()
            n = len(samples)
            return {"n": n, "top1_acc": top1 / n, "top3_acc": top3 / n}

        seen_metrics = accuracy(seen_samples, "P7 seen")
        unseen_metrics = accuracy(unseen_samples, "P7 unseen")

        tranche_buckets = {"<1200": [], "1200-1599": [], "1600-1999": [], "2000+": []}
        for s in unseen_samples:
            elo = s.get("elo")
            if elo is None:
                continue
            if elo < 1200:
                tranche_buckets["<1200"].append(s)
            elif elo < 1600:
                tranche_buckets["1200-1599"].append(s)
            elif elo < 2000:
                tranche_buckets["1600-1999"].append(s)
            else:
                tranche_buckets["2000+"].append(s)
        elo_tranches = {k: accuracy(v, f"P7 elo {k}")
                        for k, v in tranche_buckets.items() if v}

        metrics = {
            "split_index": split_index,
            "total_games": max_game_index + 1,
            "seen": seen_metrics,
            "unseen": unseen_metrics,
        }
        if elo_tranches:
            metrics["elo_tranches_unseen"] = elo_tranches

        elapsed = time.time() - start_time
        print(f"✅ P7 completed in {elapsed:.2f}s "
              f"(seen={seen_metrics['n']}, unseen={unseen_metrics['n']})")

        return ProbeResult(
            name="P7",
            n_positions=seen_metrics["n"] + unseen_metrics["n"],
            metrics=metrics,
            config={"model": str(self.model_path), "device": self.device,
                    "lichess": str(path), "positions_per_game": positions_per_game}
        )

    # -- Orchestration ----------------------------------------------------------

    def run_all_probes(self, corpus_dir: Path, out_dir: Path, lichess_path=None,
                       split_index: int = 940_000, p7_unseen: int = 10_000,
                       p7_seen: int = 1_000, p5_levels=(1, 2),
                       p6_levels=(1, 0)) -> dict:
        """Run all probes on generated corpora.

        Args:
            corpus_dir: Directory containing corpus files
            out_dir: Output directory for results
            lichess_path: Optional path to a Lichess dataset (P7)
            split_index: Seen/unseen game-index split for P7
            p7_unseen: Target number of unseen positions for P7
            p7_seen: Target number of seen positions for P7
            p5_levels: Engine levels for the conversion probe
            p6_levels: Engine levels for the endgame probe

        Returns:
            Dict of {probe_name: ProbeResult}
        """
        corpus_dir = Path(corpus_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        if self.workers > 1:
            print(f"Starting engine pool with {self.workers} workers "
                  f"(spawn: each worker loads torch, may take a minute)...")
        pool = self._make_engine_pool()
        try:
            onpolicy_path = corpus_dir / "onpolicy_positions.jsonl"
            if onpolicy_path.exists():
                results["P1"] = self.run_probe_p1(onpolicy_path, out_dir)

            mate1_path = corpus_dir / "probe_mate1.jsonl"
            if mate1_path.exists():
                results["P2"] = self.run_probe_p2(mate1_path, out_dir)

            captures_path = corpus_dir / "probe_captures.jsonl"
            if captures_path.exists():
                results["P3"] = self.run_probe_p3(captures_path, out_dir)

            if onpolicy_path.exists():
                results["P4"] = self.run_probe_p4(onpolicy_path, out_dir, pool=pool)

            conversion_path = corpus_dir / "probe_conversion.jsonl"
            if conversion_path.exists():
                results["P5"] = self.run_probe_p5(conversion_path, out_dir,
                                                  levels=p5_levels, pool=pool)

            endgames_path = corpus_dir / "probe_endgames.jsonl"
            if endgames_path.exists():
                results["P6"] = self.run_probe_p6(endgames_path, out_dir,
                                                  levels=p6_levels, pool=pool)

            promotion_path = corpus_dir / "probe_promotion.jsonl"
            if promotion_path.exists():
                results["P6b"] = self.run_probe_p6b(promotion_path, out_dir)

            results["P7"] = self.run_probe_p7(
                lichess_path, out_dir, split_index=split_index,
                n_unseen=p7_unseen, n_seen=p7_seen, pool=pool)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()

        payload = {name: result.__dict__ for name, result in results.items()}

        def _json_default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        base_path = out_dir / "probes_base.json"
        with open(base_path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

        legacy_path = out_dir / "probes_results.json"
        with open(legacy_path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

        print(f"\n✅ All probes completed! Results saved to {base_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Run probes on ChessGPT model")
    parser.add_argument("--model", type=str, default="base_model", help="Path to model")
    parser.add_argument("--corpus", type=str, default="results/corpus",
                        help="Path to corpus directory")
    parser.add_argument("--out", type=str, default="results/probes",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for model")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (auto-set to 1024 for GPU)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes for engine moves in game-playing probes")
    parser.add_argument("--lichess", type=str, default=None,
                        help="Path to Lichess dataset (.jsonl or .jsonl.gz) for P7; "
                             "default: the converted training dataset "
                             "(data/lichess_train_*.jsonl.gz) if present, else the "
                             "austindavis monthly file auto-downloaded by gen_corpus.py")
    parser.add_argument("--split-index", type=int, default=940_000,
                        help="Game index separating seen (training) from unseen positions")
    parser.add_argument("--p7-unseen", type=int, default=10_000,
                        help="Target number of unseen positions for P7")
    parser.add_argument("--p7-seen", type=int, default=1_000,
                        help="Target number of seen positions for P7")
    args = parser.parse_args()

    lichess_path = args.lichess
    if lichess_path is None:
        data_dir = Path("data")
        training = sorted(data_dir.glob("lichess_train_*.jsonl.gz"))
        monthly = sorted(data_dir.glob("lichess_uci_*.jsonl.gz"))
        legacy = data_dir / "lichess_uci.jsonl.gz"
        if training:
            lichess_path = str(training[-1])
        elif monthly:
            lichess_path = str(monthly[-1])
        elif legacy.exists():
            lichess_path = str(legacy)

    runner = ProbeRunner(args.model, device=args.device, batch_size=args.batch_size,
                         workers=args.workers)
    runner.run_all_probes(Path(args.corpus), Path(args.out), lichess_path=lichess_path,
                          split_index=args.split_index, p7_unseen=args.p7_unseen,
                          p7_seen=args.p7_seen)


if __name__ == "__main__":
    main()
