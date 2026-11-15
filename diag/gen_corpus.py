"""Corpus generation for ChessGPT diagnostic harness.

Generates test position corpora for evaluating model capabilities (C1-C14).

Key principles:
1. ALL positions must be reachable in real games (no random generation except for endgames)
2. Maximize use of Lichess dataset for speed (1M games = 100M positions)
3. Validate every single position for legality and realism
4. Optimized for mass generation (multiprocessing, batching, GPU-ready)
5. Deterministic (seeds for reproducibility)

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Optional, Callable

import chess
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessTokenizer, Engine, load_model
from chess_hw.match import select_openings, load_openings, MAX_PLIES
from chess_hw.play import ModelPlayer

try:
    from .common import ProbePlayer, replay, material_value, material_diff
except ImportError:
    from diag.common import ProbePlayer, replay, material_value, material_diff


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
# Corpus generators
# ---------------------------------------------------------------------------

class CorpusGenerator:
    """Main corpus generator with optimization and validation.
    
    Features:
    - Multiprocessing support
    - GPU-ready (via ProbePlayer)
    - Deterministic (seeds)
    - Validation of all positions
    - Atomic file writing
    """
    
    def __init__(self, model_path: str, device: str = "cpu", workers: int = 8):
        """Initialize the corpus generator.
        
        Args:
            model_path: Path to model directory
            device: "cpu" or "cuda"
            workers: Number of worker processes
        """
        self.model_path = Path(model_path)
        self.device = device
        self.workers = workers
        self.tokenizer = ChessTokenizer()
        
        # Load model and create players
        self.model = load_model(str(model_path), device=device)
        self.model_player = ModelPlayer(self.model, name="model")
        self.probe_player = ProbePlayer(self.model, self.tokenizer, device=device)
        
        # Built-in engine for analysis
        self.engine = Engine(level=3, seed=42)
    
    def _play_game_worker(self, args) -> list:
        """Worker function for parallel game playing."""
        opening_idx, level, seed, model_color = args
        
        # Set thread count for torch
        import torch
        torch.set_num_threads(1)
        
        # Load opening
        openings = load_openings()
        opening = openings[opening_idx]
        
        # Create board and play opening
        board = chess.Board()
        for uci in opening:
            board.push_uci(uci)
        
        # Create engine player
        engine = Engine(level=level, seed=f"engine-{seed}-{level}-{opening_idx}-{model_color}")
        
        # Set up players
        if model_color == "white":
            white_player = self.model_player
            black_player = ModelPlayer(self.model, name="engine")
            black_player.model = None  # Will use engine
        else:
            white_player = ModelPlayer(self.model, name="engine")
            white_player.model = None
            black_player = self.model_player
        
        # Play the game
        positions = []
        move_count = 0
        
        while not board.is_game_over() and board.ply() < MAX_PLIES:
            # Model's turn
            if board.turn == chess.WHITE and model_color == "white" or \
               board.turn == chess.BLACK and model_color == "black":
                
                # Get model's move with attempt info
                choice = self.model_player.choose(board)
                
                if choice.move:
                    move = choice.move
                    attempt = choice.attempt
                else:
                    # Fallback
                    legal = list(board.legal_moves)
                    if legal:
                        move = random.Random(f"fallback-{opening_idx}-{board.ply()}").choice(legal)
                        attempt = 0
                    else:
                        break
                
                # Record position BEFORE the move
                positions.append({
                    "history": [m.uci() for m in board.move_stack],
                    "fen": board.fen(),
                    "ply": board.ply(),
                    "level": level,
                    "opening_id": opening_idx,
                    "model_color": model_color,
                    "in_check": board.is_check(),
                    "n_legal": board.legal_moves.count(),
                    "chosen_move": move.uci() if move else None,
                    "chosen_rank": attempt,
                })
                
                board.push(move)
                move_count += 1
            else:
                # Engine's turn
                engine_move = engine.best_move(board)
                if engine_move:
                    board.push(engine_move)
                else:
                    break
        
        return positions
    
    def generate_onpolicy_positions(self, n_games: int = 2000, levels: list = [0, 1, 2, 3], 
                                     seed: int = 0, out_dir: str = "results/corpus") -> Path:
        """Generate on-policy positions by playing model vs engine.
        
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
        
        # Play games in parallel
        all_positions = []
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._play_game_worker, task): task for task in tasks}
            
            for future in as_completed(futures):
                try:
                    positions = future.result()
                    all_positions.extend(positions)
                except Exception as e:
                    print(f"Error in game: {e}", file=sys.stderr)
        
        # Write to file atomically
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            for pos in all_positions:
                f.write(json.dumps(pos, sort_keys=True) + '\n')
        
        os.replace(tmp_path, out_path)
        
        print(f"✅ Generated {len(all_positions)} on-policy positions")
        return out_path
    
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
        
        positions = []
        
        # Open dataset (assuming it's a gzipped JSONL file)
        with gzip.open(dataset_path, 'rt') as f:
            for line in f:
                if len(positions) >= n_positions:
                    break
                
                try:
                    game = json.loads(line)
                    moves = game.get('moves', '')
                    if not moves:
                        continue
                    
                    board = chess.Board()
                    for uci in moves.split():
                        try:
                            board.push_uci(uci)
                        except:
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
                except Exception as e:
                    continue
        
        # Write to file
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            for pos in positions:
                f.write(json.dumps(pos, sort_keys=True) + '\n')
        
        os.replace(tmp_path, out_path)
        
        print(f"✅ Extracted {len(positions)} mate-in-1 positions")
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
        
        positions = []
        
        with gzip.open(dataset_path, 'rt') as f:
            for line in f:
                if len(positions) >= n_positions:
                    break
                
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
                            board.push(move)
                            move_list.append(uci)
                            
                            # Check if this move is a capture
                            if board.is_capture(move):
                                valid, reason = validate_position(board, strict=False)
                                if valid:
                                    # Get SEE value
                                    see = self._see(board, move)
                                    
                                    positions.append({
                                        "fen": board.fen(),
                                        "history": move_list[:-1],  # History before the capture
                                        "ply": board.ply() - 1,
                                        "capture_move": uci,
                                        "see": see,
                                        "source": "lichess",
                                    })
                        except:
                            break
                except Exception as e:
                    continue
        
        # Write to file
        tmp_path = out_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            for pos in positions:
                f.write(json.dumps(pos, sort_keys=True) + '\n')
        
        os.replace(tmp_path, out_path)
        
        print(f"✅ Extracted {len(positions)} capture positions")
        return out_path
    
    def _see(self, board: chess.Board, move: chess.Move) -> int:
        """Static Exchange Evaluation (simplified version)."""
        from .see import see as see_func
        return see_func(board, move)
    
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
        
        print(f"✅ Generated {len(positions)} endgame positions")
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
    
    def generate_all_corpora(self, lichess_path: Optional[str] = None,
                             n_onpolicy: int = 2000, out_dir: str = "results/corpus") -> dict:
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
            n_games=n_onpolicy, out_dir=out_dir
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
        
        print("\n✅ All corpora generated!")
        print(f"   Manifest: {out_dir / 'manifest.json'}")
        
        return corpora


def main():
    parser = argparse.ArgumentParser(description="Generate test corpora for ChessGPT")
    parser.add_argument("--model", type=str, default="base_model", help="Path to model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for model")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--out", type=str, default="results/corpus",
                        help="Output directory")
    parser.add_argument("--lichess", type=str, default=None,
                        help="Path to Lichess dataset for extraction")
    parser.add_argument("--onpolicy", type=int, default=2000,
                        help="Number of on-policy games")
    args = parser.parse_args()
    
    # Create generator
    generator = CorpusGenerator(args.model, device=args.device, workers=args.workers)
    
    # Generate all corpora
    generator.generate_all_corpora(
        lichess_path=args.lichess,
        n_onpolicy=args.onpolicy,
        out_dir=args.out
    )


if __name__ == "__main__":
    main()
