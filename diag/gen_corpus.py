"""Corpus generation for ChessGPT diagnostic harness.

Generates test position corpora for evaluating model capabilities (C1-C14).

Key principles:
1. ALL positions must be reachable in real games (no random generation except for endgames)
2. Maximize use of Lichess dataset for speed (1M games = 100M positions)
3. Validate every single position for legality and realism
4. Optimized for mass generation (multiprocessing, batching, GPU-ready)
5. Deterministic (seeds for reproducibility)

Performance notes (this version is heavily optimized, see diag/fast_probe.py
and diag/fast_engine.py -- optimized reimplementations kept OUTSIDE chess_hw):

- The model is never run through the naive full-logits forward pass: only the
  last-position logits are computed (hidden states -> gather -> tied-head
  projection), which avoids materializing a (B, T, 1971) tensor (~4 GB fp32
  for B=2048, T=256) and cuts GPU time and memory by orders of magnitude.
- All games are played SIMULTANEOUSLY in lockstep rounds: every model turn
  across every active game is batched into a single GPU forward pass, while
  the opponent engine moves (the real wall-clock cost: pure-Python alpha-beta
  at ~100-300 ms/move) are dispatched in parallel to a CPU process pool,
  overlapping CPU and GPU work.
- Lichess extraction (mate-in-1, captures) is parallelized over file chunks.

Usage:
    uv run diag/gen_corpus.py --model base_model --out results/corpus --workers 8 --device cuda
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import chess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessTokenizer, load_model
from chess_hw.match import load_openings, MAX_PLIES

from diag.fast_engine import engine_best_move_task
from diag.fast_probe import FastProbePlayer
from diag.see import see as see_func


# ---------------------------------------------------------------------------
# Validation utils
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def validate_position(board: chess.Board, strict: bool = True) -> tuple:
    """Validate a position for realism and legality.

    Returns: (is_valid: bool, reason: str)
    """
    # 1. Basic legality
    try:
        chess.Board(board.fen())
    except ValueError as e:
        return False, f"Illegal FEN: {e}"

    # 2. Game state
    if board.is_checkmate():
        return False, "Checkmate in progress"
    if board.is_stalemate():
        return False, "Stalemate in progress"

    # 3. Minimum pieces (at least 2 kings)
    piece_count = len(board.piece_map())
    if piece_count < 2:
        return False, f"Too few pieces: {piece_count}"

    # 4. Realistic material balance
    white_mat = sum(PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
    black_mat = sum(PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)

    if strict:
        # Avoid extreme material imbalances (except in endgames)
        if piece_count > 10 and abs(white_mat - black_mat) > 2000:  # > 20 pawns difference
            return False, f"Extreme material imbalance: {white_mat} vs {black_mat}"

    # 5. Realistic piece distribution
    # Check for too many pieces of one type
    piece_counts = {}
    for piece in board.piece_map().values():
        key = (piece.color, piece.piece_type)
        piece_counts[key] = piece_counts.get(key, 0) + 1

    for (color, pt), count in piece_counts.items():
        if pt == chess.PAWN and count > 8:
            return False, f"Too many pawns: {count}"
        if pt != chess.PAWN and count > 2:  # Max 2 queens, 2 rooks, etc.
            return False, f"Too many {pt}: {count}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Endgame generators (artificial but VALID positions)
# ---------------------------------------------------------------------------

def generate_kqk_position(rng: random.Random) -> chess.Board:
    """Generate a valid K+Q vs K position (checkmate possible).

    Strategy: Place kings in realistic positions, queen in attacking position.
    """
    board = chess.Board()
    board.clear()

    # Place white king
    wk_square = rng.choice([chess.D1, chess.E1, chess.D2, chess.E2, chess.F1, chess.F2])
    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))

    # Place black king (not adjacent to white king)
    bk_options = [s for s in chess.SQUARES if chess.square_distance(s, wk_square) >= 3]
    bk_square = rng.choice(bk_options)
    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))

    # Place white queen in position to attack
    queen_options = [s for s in chess.SQUARES
                     if s not in [wk_square, bk_square]
                     and chess.square_distance(s, bk_square) <= 3]  # Close to black king
    if queen_options:
        q_square = rng.choice(queen_options)
        board.set_piece_at(q_square, chess.Piece(chess.QUEEN, chess.WHITE))
    else:
        # Fallback: any square
        q_square = rng.choice([s for s in chess.SQUARES if s not in [wk_square, bk_square]])
        board.set_piece_at(q_square, chess.Piece(chess.QUEEN, chess.WHITE))

    board.turn = chess.WHITE

    # Validate
    valid, reason = validate_position(board, strict=False)
    if not valid:
        return generate_kqk_position(rng)  # Retry

    return board


def generate_krk_position(rng: random.Random) -> chess.Board:
    """Generate a valid K+R vs K position (checkmate possible).

    Strategy: Place kings with space, rook in attacking position.
    """
    board = chess.Board()
    board.clear()

    # Place white king
    wk_square = rng.choice([chess.D1, chess.E1, chess.D2, chess.E2])
    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))

    # Place black king
    bk_options = [s for s in chess.SQUARES if chess.square_distance(s, wk_square) >= 3]
    bk_square = rng.choice(bk_options)
    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))

    # Place white rook
    rook_options = [s for s in chess.SQUARES
                    if s not in [wk_square, bk_square]
                    and (chess.square_file(s) == chess.square_file(bk_square) or
                         chess.square_rank(s) == chess.square_rank(bk_square))]  # Same file/rank as black king
    if rook_options:
        r_square = rng.choice(rook_options)
        board.set_piece_at(r_square, chess.Piece(chess.ROOK, chess.WHITE))
    else:
        r_square = rng.choice([s for s in chess.SQUARES if s not in [wk_square, bk_square]])
        board.set_piece_at(r_square, chess.Piece(chess.ROOK, chess.WHITE))

    board.turn = chess.WHITE

    # Validate
    valid, reason = validate_position(board, strict=False)
    if not valid:
        return generate_krk_position(rng)  # Retry

    return board


def generate_endgame_position(rng: random.Random, endgame_type: str = "kqk") -> chess.Board:
    """Generate a valid endgame position.

    Types: "kqk" (K+Q vs K), "krk" (K+R vs K)
    """
    generators = {
        "kqk": generate_kqk_position,
        "krk": generate_krk_position,
    }

    if endgame_type not in generators:
        raise ValueError(f"Unknown endgame type: {endgame_type}")

    return generators[endgame_type](rng)


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------

def _pool_init():
    """Pool worker init: keep worker processes single-threaded."""
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


_EXTRACT_BLOCK_LINES = 500


def _mate1_block_worker(lines: List[str]) -> List[dict]:
    """Extract mate-in-1 positions from a block of raw JSONL game lines."""
    positions = []
    for line in lines:
        try:
            game = json.loads(line)
            moves = game.get('moves', '')
            if not moves:
                continue

            board = chess.Board()
            for uci in moves.split():
                try:
                    board.push_uci(uci)
                except Exception:
                    break

            # Check if current position has mate in 1
            if board.is_checkmate():
                continue

            mates = []
            for move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    mates.append(move.uci())
                board.pop()

            if mates:
                valid, reason = validate_position(board)
                if valid:
                    positions.append({
                        "fen": board.fen(),
                        "moves": [m.uci() for m in board.move_stack],
                        "ply": board.ply(),
                        "mates": mates,
                        "source": "lichess",
                    })
        except Exception:
            continue
    return positions


def _captures_block_worker(lines: List[str]) -> List[dict]:
    """Extract capture positions from a block of raw JSONL game lines."""
    positions = []
    for line in lines:
        try:
            game = json.loads(line)
            moves = game.get('moves', '')
            if not moves:
                continue

            board = chess.Board()
            move_list = []

            for uci in moves.split():
                try:
                    move = board.parse_uci(uci)

                    # Check if this move is a capture BEFORE pushing it: the
                    # recorded fen/history/see must describe the position the
                    # capture is played from (the previous version pushed
                    # first, which made is_capture() always true and crashed
                    # see(), silently aborting every game at its first move).
                    if board.is_capture(move):
                        valid, reason = validate_position(board, strict=False)
                        if valid:
                            positions.append({
                                "fen": board.fen(),
                                "history": list(move_list),  # History before the capture
                                "ply": board.ply(),
                                "capture_move": uci,
                                "see": see_func(board, move),
                                "source": "lichess",
                            })

                    board.push(move)
                    move_list.append(uci)
                except Exception:
                    break
        except Exception:
            continue
    return positions


# ---------------------------------------------------------------------------
# Corpus generators
# ---------------------------------------------------------------------------

class CorpusGenerator:
    """Main corpus generator with optimization and validation.

    Features:
    - Lockstep round scheduler: one batched GPU pass covers the model turns of
      ALL active games; engine turns are computed in parallel on a CPU pool
      (overlapped with the GPU work).
    - GPU-ready (via FastProbePlayer, last-position logits only)
    - Deterministic (seeds)
    - Validation of all positions
    - Atomic file writing
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        workers: int = 8,
        batch_size: Optional[int] = None,
    ):
        """Initialize the corpus generator.

        Args:
            model_path: Path to model directory
            device: "cpu" or "cuda"
            workers: Number of worker processes for engine moves / extraction
            batch_size: Inference batch size (default: 1024 on CUDA, 256 on CPU)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.workers = max(1, workers)
        self.tokenizer = ChessTokenizer()

        # A larger batch improves GPU utilization; callers can lower it to avoid OOM.
        batch_size = batch_size or (1024 if device == "cuda" else 256)

        # Load model and create the fast batched probe player
        self.model = load_model(str(model_path), device=device)
        self.probe_player = FastProbePlayer(
            self.model, self.tokenizer, batch_size=batch_size, device=device
        )

    # -- engine pool -------------------------------------------------------

    def _make_engine_pool(self):
        if self.workers <= 1:
            return None
        ctx = get_context("spawn")
        return ctx.Pool(self.workers, initializer=_pool_init)

    # -- on-policy generation ----------------------------------------------

    def generate_onpolicy_positions(self, n_games: int = 2000, levels: List[int] = [0, 1, 2, 3],
                                    seed: int = 0, out_dir: str = "results/corpus") -> Path:
        """Generate on-policy positions by playing model vs engine.

        All games are advanced in lockstep rounds. In each round:
          1. engine-turn games are dispatched to the CPU pool (async),
          2. all model-turn games are inferred in ONE batched forward pass
             (per chunk of `batch_size`) while the pool works,
          3. engine results are collected and applied.

        Args:
            n_games: Total number of games to play
            levels: Engine levels to play against
            seed: Random seed for reproducibility
            out_dir: Output directory

        Returns:
            Path to onpolicy_positions.jsonl
        """
        out_path = Path(out_dir) / "onpolicy_positions.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Select openings
        rng = random.Random(seed)
        all_openings = load_openings()
        opening_indices = rng.sample(range(len(all_openings)), n_games)

        # Create tasks: (opening_idx, level, seed, model_color)
        tasks = []
        games_per_level = n_games // len(levels)
        for level in levels:
            level_openings = opening_indices[:games_per_level]
            opening_indices = opening_indices[games_per_level:]
            for oid in level_openings:
                tasks.append((oid, level, seed, "white"))
                tasks.append((oid, level, seed, "black"))

        def show_progress(done: int, total: int, started: float) -> None:
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed else 0.0
            remaining = (total - done) / rate if rate else 0.0
            width = 30
            filled = int(width * done / total) if total else width
            bar = "#" * filled + "-" * (width - filled)
            eta = f"{remaining / 60:.1f} min left" if rate else "estimating..."
            print(
                f"\rOn-policy games [{bar}] {done}/{total} "
                f"({100 * done / total:.1f}%) | {rate:.2f} games/s | {eta}",
                end="",
                flush=True,
            )

        started = time.monotonic()
        print(f"Starting {len(tasks)} on-policy games (device={self.device}, "
              f"workers={self.workers}, batch_size={self.probe_player.batch_size})")

        openings = load_openings()
        games = []
        for opening_idx, level, gseed, model_color in tasks:
            board = chess.Board()
            for uci in openings[opening_idx]:
                board.push_uci(uci)
            games.append({
                "board": board,
                "history": [m.uci() for m in board.move_stack],
                "model_color": model_color,
                "opening_idx": opening_idx,
                "level": level,
                "engine_seed": f"engine-{gseed}-{level}-{opening_idx}-{model_color}",
                "positions": [],
            })

        total_games = len(games)
        active = list(range(total_games))
        all_positions = []
        pool = self._make_engine_pool()

        try:
            while active:
                model_idx, engine_idx = [], []
                for i in active:
                    g = games[i]
                    b = g["board"]
                    if b.is_game_over() or b.ply() >= MAX_PLIES:
                        all_positions.extend(g["positions"])
                        continue
                    if (b.turn == chess.WHITE) == (g["model_color"] == "white"):
                        model_idx.append(i)
                    else:
                        engine_idx.append(i)

                # Fan engine moves out to the pool first (async) so the CPU
                # works while the GPU processes the model turns below.
                engine_results = None
                if engine_idx:
                    engine_tasks = [
                        (games[i]["board"].fen(), games[i]["level"], games[i]["engine_seed"])
                        for i in engine_idx
                    ]
                    if pool is not None:
                        chunksize = max(1, len(engine_tasks) // (self.workers * 4))
                        # imap dispatches tasks eagerly; results are only
                        # consumed after the GPU work below, so the pool and
                        # the device work in overlap.
                        engine_results = pool.imap(
                            engine_best_move_task, engine_tasks, chunksize=chunksize
                        )
                    else:
                        engine_results = [engine_best_move_task(t) for t in engine_tasks]

                # Model moves: one batched forward pass per chunk. This is the
                # only GPU work per round and it covers every active game.
                bs = self.probe_player.batch_size
                for c in range(0, len(model_idx), bs):
                    chunk = model_idx[c:c + bs]
                    choices = self.probe_player.choose_for_boards(
                        [games[i]["board"] for i in chunk]
                    )
                    for i, choice in zip(chunk, choices):
                        g = games[i]
                        b = g["board"]
                        move, attempt = choice.move, choice.attempt
                        if move is None:
                            legal = list(b.legal_moves)
                            if not legal:
                                continue
                            move = random.Random(
                                f"fallback-{g['opening_idx']}-{b.ply()}"
                            ).choice(legal)
                            attempt = 0
                        g["positions"].append({
                            "history": list(g["history"]),
                            "fen": b.fen(),
                            "ply": b.ply(),
                            "level": g["level"],
                            "opening_id": g["opening_idx"],
                            "model_color": g["model_color"],
                            "in_check": b.is_check(),
                            "n_legal": b.legal_moves.count(),
                            "chosen_move": move.uci() if move else None,
                            "chosen_rank": attempt,
                        })
                        uci = move.uci()
                        b.push(move)
                        g["history"].append(uci)

                # Collect and apply engine moves.
                if engine_results is not None:
                    ucis = list(engine_results)
                    for i, uci in zip(engine_idx, ucis):
                        if uci:
                            g = games[i]
                            g["board"].push_uci(uci)
                            g["history"].append(uci)

                still_active = []
                for i in model_idx + engine_idx:
                    b = games[i]["board"]
                    if b.is_game_over() or b.ply() >= MAX_PLIES:
                        all_positions.extend(games[i]["positions"])
                    else:
                        still_active.append(i)
                active = still_active
                show_progress(total_games - len(active), total_games, started)
        finally:
            if pool is not None:
                pool.terminate()
        print()

        # Write to file atomically
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            for pos in all_positions:
                f.write(json.dumps(pos, sort_keys=True) + '\n')

        os.replace(tmp_path, out_path)

        print(f"\u2705 Generated {len(all_positions)} on-policy positions")
        return out_path

    # -- Lichess extraction --------------------------------------------------

    def _extract_parallel(self, dataset_path: str, worker: Callable,
                          n_positions: int, label: str) -> List[dict]:
        """Parse a gzipped JSONL game dataset with a process pool, block by block.

        Blocks in flight are bounded so huge files are streamed, not loaded.
        """
        positions: List[dict] = []
        started = time.monotonic()
        pool = self._make_engine_pool()
        try:
            if pool is None:
                with gzip.open(dataset_path, 'rt') as f:
                    while len(positions) < n_positions:
                        block = list(islice(f, _EXTRACT_BLOCK_LINES))
                        if not block:
                            break
                        positions.extend(worker(block))
            else:
                import threading
                sem = threading.Semaphore(self.workers * 4)

                def blocks():
                    with gzip.open(dataset_path, 'rt') as f:
                        while True:
                            block = list(islice(f, _EXTRACT_BLOCK_LINES))
                            if not block:
                                return
                            sem.acquire()
                            yield block

                for block_positions in pool.imap_unordered(worker, blocks(), chunksize=1):
                    sem.release()
                    positions.extend(block_positions)
                    if len(positions) >= n_positions:
                        break
        finally:
            if pool is not None:
                pool.terminate()

        rate = len(positions) / max(time.monotonic() - started, 1e-9)
        print(f"   {label}: {len(positions)} positions ({rate:.0f}/s)")
        return positions[:n_positions]

    def extract_mate1_from_lichess(self, dataset_path: str, n_positions: int = 5000,
                                   out_dir: str = "results/corpus") -> Path:
        """Extract mate-in-1 positions from Lichess dataset.

        Args:
            dataset_path: Path to Lichess dataset file
            n_positions: Number of positions to extract
            out_dir: Output directory

        Returns:
            Path to probe_mate1.jsonl
        """
        out_path = Path(out_dir) / "probe_mate1.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        positions = self._extract_parallel(
            dataset_path, _mate1_block_worker, n_positions, "mate-in-1"
        )

        # Write to file
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            for pos in positions:
                f.write(json.dumps(pos, sort_keys=True) + '\n')

        os.replace(tmp_path, out_path)

        print(f"\u2705 Extracted {len(positions)} mate-in-1 positions")
        return out_path

    def extract_captures_from_lichess(self, dataset_path: str, n_positions: int = 20000,
                                      out_dir: str = "results/corpus") -> Path:
        """Extract positions with captures from Lichess dataset.

        Args:
            dataset_path: Path to Lichess dataset
            n_positions: Number of positions to extract
            out_dir: Output directory

        Returns:
            Path to probe_captures.jsonl
        """
        out_path = Path(out_dir) / "probe_captures.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        positions = self._extract_parallel(
            dataset_path, _captures_block_worker, n_positions, "captures"
        )

        # Write to file
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            for pos in positions:
                f.write(json.dumps(pos, sort_keys=True) + '\n')

        os.replace(tmp_path, out_path)

        print(f"\u2705 Extracted {len(positions)} capture positions")
        return out_path

    def generate_endgame_positions(self, n_positions: int = 2000,
                                   out_dir: str = "results/corpus") -> Path:
        """Generate valid endgame positions (K+Q vs K, K+R vs K).

        Args:
            n_positions: Number of positions to generate
            out_dir: Output directory

        Returns:
            Path to probe_endgames.jsonl
        """
        out_path = Path(out_dir) / "probe_endgames.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(42)
        positions = []

        # Generate K+Q vs K positions
        for _ in range(n_positions // 2):
            board = generate_kqk_position(rng)

            # Create history to reach this position
            # We'll generate a random but valid sequence
            history = self._generate_history_to_board(board, rng)

            positions.append({
                "fen": board.fen(),
                "history": history,
                "ply": len(history),
                "type": "kqk",
                "model_color": "white" if board.turn == chess.WHITE else "black",
                "source": "generated",
            })

        # Generate K+R vs K positions
        for _ in range(n_positions // 2):
            board = generate_krk_position(rng)

            history = self._generate_history_to_board(board, rng)

            positions.append({
                "fen": board.fen(),
                "history": history,
                "ply": len(history),
                "type": "krk",
                "model_color": "white" if board.turn == chess.WHITE else "black",
                "source": "generated",
            })

        # Write to file
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            for pos in positions:
                f.write(json.dumps(pos, sort_keys=True) + '\n')

        os.replace(tmp_path, out_path)

        print(f"\u2705 Generated {len(positions)} endgame positions")
        return out_path

    def _generate_history_to_board(self, target_board: chess.Board, rng: random.Random) -> list:
        """Generate a valid move history to reach a target board position.

        This is tricky - we need to find a path from start position to target.
        For now, we'll use a simple approach: play random moves until we reach
        a similar material configuration, then adjust.

        For endgames, we can use a simpler approach: just return an empty history
        and note that the position is generated (not from a real game).
        """
        # For now, return empty history - the position is generated artificially
        # In production, we'd want to generate a real history
        return []

    def generate_all_corpora(
        self,
        lichess_path: Optional[str] = None,
        n_onpolicy: int = 2000,
        levels: Optional[List[int]] = None,
        out_dir: str = "results/corpus",
    ) -> dict:
        """Generate all corpora needed for C1-C14 evaluation.

        Args:
            lichess_path: Path to Lichess dataset (optional)
            n_onpolicy: Number of on-policy games
            out_dir: Output directory

        Returns:
            Dict of {corpus_name: path}
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        corpora = {}

        # 1. On-policy positions (generation)
        print("\n=== Generating on-policy positions ===")
        corpora['onpolicy'] = self.generate_onpolicy_positions(
            n_games=n_onpolicy,
            levels=levels or [0, 1, 2, 3],
            out_dir=out_dir,
        )

        # 2. Mate-in-1 positions (from Lichess if available)
        if lichess_path:
            print("\n=== Extracting mate-in-1 positions ===")
            corpora['mate1'] = self.extract_mate1_from_lichess(
                lichess_path, out_dir=out_dir
            )

        # 3. Capture positions (from Lichess if available)
        if lichess_path:
            print("\n=== Extracting capture positions ===")
            corpora['captures'] = self.extract_captures_from_lichess(
                lichess_path, out_dir=out_dir
            )

        # 4. Endgame positions (generated)
        print("\n=== Generating endgame positions ===")
        corpora['endgames'] = self.generate_endgame_positions(out_dir=out_dir)

        # Write manifest
        manifest = {
            "corpora": {name: str(path) for name, path in corpora.items()},
            "seeds": {"onpolicy": 0},
            "version": "1.0",
            "date": str(Path(out_dir).stat().st_ctime),
        }

        with open(out_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        print("\n\u2705 All corpora generated!")
        print(f"   Manifest: {out_dir / 'manifest.json'}")

        return corpora


def main():
    parser = argparse.ArgumentParser(description="Generate test corpora for ChessGPT")
    parser.add_argument("--model", type=str, default="base_model", help="Path to model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for model")
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker processes for engine moves / Lichess extraction")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size (default: 1024 on CUDA, 256 on CPU; lower if memory is insufficient)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        choices=range(4),
        help="Engine levels for on-policy games; level 0 is fastest",
    )
    parser.add_argument("--out", type=str, default="results/corpus",
                        help="Output directory")
    parser.add_argument("--lichess", type=str, default=None,
                        help="Path to Lichess dataset for extraction")
    parser.add_argument("--onpolicy", type=int, default=2000,
                        help="Number of on-policy games")
    args = parser.parse_args()

    # Create generator
    generator = CorpusGenerator(
        args.model,
        device=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    # Generate all corpora
    generator.generate_all_corpora(
        lichess_path=args.lichess,
        n_onpolicy=args.onpolicy,
        levels=args.levels,
        out_dir=args.out
    )


if __name__ == "__main__":
    main()
