"""Common utilities for the diagnostic harness.

Provides ProbePlayer for accessing full model logits, board utilities, and I/O helpers.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import chess
import numpy as np
import torch

# Add parent directory to path to import chess_hw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessGPT, ChessTokenizer, load_model
from chess_hw.play import MoveChoice

# Constants
MAX_PLIES = 250
MAX_CONTEXT = 256


def load(model_path: Union[str, Path], device: str = "cpu") -> Tuple[ChessGPT, ChessTokenizer]:
    """Load model and tokenizer, put model in eval mode on the specified device."""
    model = load_model(str(model_path), device=device)
    tokenizer = ChessTokenizer()
    return model, tokenizer


class ProbePlayer:
    """Extended model player with access to full logits.
    
    Unlike ModelPlayer.candidates() which only returns top-3 tokens,
    this class provides access to all logits for analysis.
    
    Args:
        model: ChessGPT model
        tokenizer: ChessTokenizer instance
        batch_size: Batch size for inference (default: 64 for CPU, 512 for GPU)
        device: Device to run inference on
    """
    
    def __init__(self, model: ChessGPT, tokenizer: ChessTokenizer, 
                 batch_size: int = 64, device: Optional[str] = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # Auto-detect GPU and set optimal batch size
        if self.device == 'cuda':
            # For GPU, use much larger batch sizes to maximize utilization
            # Model has ~946K params, each forward pass with batch B uses:
            # - Input: B * 256 * 4 bytes (int64)
            # - Model params: ~3.8 MB (float32)
            # - Output: B * vocab_size * 4 bytes (float32) = B * 1971 * 4
            # - Activations: ~B * 256 * 128 * 4 (hidden states)
            # Total per batch: ~B * (256*4 + 1971*4 + 256*128*4) = B * ~135 KB
            # With 8GB GPU, we can fit batch of ~60K, but practical limit is lower
            # Use 1024 as default for GPU (can be increased based on available memory)
            self.batch_size = batch_size if batch_size > 64 else 1024
        else:
            # For CPU, keep smaller batch size
            self.batch_size = batch_size
        
        print(f"ProbePlayer initialized: device={self.device}, batch_size={self.batch_size}")
        
    def _batch_logits(self, histories: List[List[str]]) -> np.ndarray:
        """Compute logits for a batch of histories.
        
        Optimized for GPU:
        - Keeps all tensors on GPU device
        - Minimizes CPU-GPU transfers
        - Processes in batches for memory efficiency
        
        Args:
            histories: List of move histories (each is a list of UCI strings)
            
        Returns:
            Array of shape (n_histories, vocab_size) with logits at the last position
        """
        if not histories:
            return np.array([]).reshape(0, self.tokenizer.vocab_size)
        
        n_histories = len(histories)
        
        # Process in batches to avoid OOM and maximize GPU utilization
        all_logits = []
        
        for batch_start in range(0, n_histories, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_histories)
            batch_histories = histories[batch_start:batch_end]
            
            # Sort by length for efficient batching (causal attention allows padding)
            sorted_indices = np.argsort([len(h) for h in batch_histories])[::-1]
            sorted_histories = [batch_histories[i] for i in sorted_indices]
            
            # Encode all histories
            all_ids = []
            for h in sorted_histories:
                ids = self.tokenizer.encode(h, add_bos=True)
                all_ids.append(ids)
            
            # Pad to same length
            max_len = max(len(ids) for ids in all_ids)
            assert max_len <= MAX_CONTEXT, f"History length {max_len} exceeds context {MAX_CONTEXT}"
            
            # Create padded tensor directly on the target device
            padded = torch.full((len(all_ids), max_len), self.tokenizer.pad_token_id, 
                              dtype=torch.long, device=self.device)
            for i, ids in enumerate(all_ids):
                padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            
            # Forward pass - keep everything on GPU
            with torch.inference_mode():
                logits = self.model(padded)  # (B, T, vocab_size)
                # Take logits at the last non-padding position for each
                # Compute mask on GPU
                mask = (padded != self.tokenizer.pad_token_id)
                last_pos = mask.sum(dim=1) - 1
                last_logits = logits[torch.arange(len(all_ids), device=self.device), last_pos]
            
            # Convert to numpy only at the end
            logits_np = last_logits.float().cpu().numpy()
            
            # Unsorter indices
            unsort_indices = np.argsort(sorted_indices)
            batch_logits = logits_np[unsort_indices]
            
            all_logits.append(batch_logits)
        
        # Concatenate all batches
        return np.concatenate(all_logits, axis=0)
    

    def get_batch_logits(self, histories: List[List[str]]) -> np.ndarray:
        """Get logits for multiple histories (alias for _batch_logits).
        
        This is the main method for batch inference. For GPU, it will
        automatically use large batches to maximize GPU utilization.
        
        Args:
            histories: List of move histories
            
        Returns:
            Array of shape (n_histories, vocab_size) with logits
        """
        return self._batch_logits(histories)
    
    @staticmethod
    def get_gpu_memory_info() -> dict:
        """Get GPU memory usage information (if CUDA available)."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved': torch.cuda.memory_reserved() / 1024 / 1024,
                'total': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
            }
        return {'allocated': 0, 'reserved': 0, 'total': 0}

    def get_logits(self, history: Sequence[str]) -> np.ndarray:
        """Get logits for a single history.
        
        Args:
            history: List of UCI moves
            
        Returns:
            Array of shape (vocab_size,) with logits
        """
        return self._batch_logits([list(history)])[0]
    
    def get_topk(self, history: Sequence[str], k: int = 3) -> Tuple[List[str], List[float]]:
        """Get top-k tokens and their probabilities.
        
        Args:
            history: List of UCI moves
            k: Number of top tokens to return
            
        Returns:
            Tuple of (tokens, probabilities) sorted by probability descending
        """
        logits = self.get_logits(history)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        topk_indices = np.argsort(probs)[-k:][::-1]
        tokens = [self.tokenizer.id_to_token[int(i)] for i in topk_indices]
        probs = probs[topk_indices].tolist()
        return tokens, probs
    
    def get_legal_mask(self, board: chess.Board) -> np.ndarray:
        """Get a boolean mask of which tokens are legal moves in the current position.
        
        Args:
            board: Current chess board position
            
        Returns:
            Boolean array of shape (vocab_size,) where True indicates a legal move token
        """
        legal_moves = set(m.uci() for m in board.legal_moves)
        mask = np.zeros(self.tokenizer.vocab_size, dtype=bool)
        for token, idx in self.tokenizer.token_to_id.items():
            if token in legal_moves:
                mask[idx] = True
        return mask
    
    def get_legal_probs(self, history: Sequence[str], board: chess.Board) -> dict:
        """Get probabilities for all legal moves.
        
        Args:
            history: List of UCI moves
            board: Current chess board position
            
        Returns:
            Dict mapping UCI move -> probability
        """
        logits = self.get_logits(history)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        legal_mask = self.get_legal_mask(board)
        
        result = {}
        for token, idx in self.tokenizer.token_to_id.items():
            if legal_mask[idx]:
                result[token] = float(probs[idx])
        
        return result
    

    def choose(self, board: chess.Board) -> "MoveChoice":
        """Choose a move like ModelPlayer but using batched inference.
        
        This method mimics chess_hw.play.ModelPlayer.choose() but uses
        the optimized batch inference from ProbePlayer.
        
        Args:
            board: Current chess board position
            
        Returns:
            MoveChoice with move and attempt number (1, 2, 3, or 0)
        """
        history = [m.uci() for m in board.move_stack]
        candidates = self.get_topk(history, k=3)[0]
        
        for attempt, tok in enumerate(candidates, start=1):
            try:
                mv = board.parse_uci(tok)
                return MoveChoice(mv, attempt)
            except ValueError:
                continue
        return MoveChoice(None, 0)

    def choose_with_probs(self, board: chess.Board) -> Tuple[Optional[chess.Move], int, dict]:
        """Choose a move like ModelPlayer but return probabilities for all legal moves.
        
        Args:
            board: Current chess board position
            
        Returns:
            Tuple of (chosen_move, attempt, probabilities_dict)
        """
        history = [m.uci() for m in board.move_stack]
        candidates = self.get_topk(history, k=3)[0]
        
        # Try candidates in order
        for attempt, tok in enumerate(candidates, start=1):
            try:
                mv = board.parse_uci(tok)
                probs = self.get_legal_probs(history, board)
                return mv, attempt, probs
            except ValueError:
                continue
        
        # All top-3 illegal, return None with probabilities
        probs = self.get_legal_probs(history, board)
        return None, 0, probs


def replay(history: Sequence[str]) -> chess.Board:
    """Replay a history of UCI moves and return the resulting board.
    
    Args:
        history: List of UCI moves
        
    Returns:
        chess.Board at the end of the history
        
    Raises:
        AssertionError: If any move is illegal
    """
    board = chess.Board()
    for uci in history:
        try:
            board.push_uci(uci)
        except ValueError as e:
            raise AssertionError(f"Illegal move {uci} in history: {e}")
    return board


def material(board: chess.Board) -> dict:
    """Count material on the board.
    
    Args:
        board: chess.Board
        
    Returns:
        Dict with counts by piece type and color
    """
    counts = {
        chess.WHITE: {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0, 
                      chess.ROOK: 0, chess.QUEEN: 0, chess.KING: 0},
        chess.BLACK: {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0,
                      chess.ROOK: 0, chess.QUEEN: 0, chess.KING: 0},
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            counts[piece.color][piece.piece_type] += 1
    
    return counts


def material_value(board: chess.Board, color: chess.Color) -> float:
    """Calculate material value for a color.
    
    Args:
        board: chess.Board
        color: chess.WHITE or chess.BLACK
        
    Returns:
        Material value in pawns (1=pawn, 3=knight/bishop, 5=rook, 9=queen)
    """
    VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    value = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            value += VAL[piece.piece_type]
    
    return value


def material_diff(board: chess.Board, color: chess.Color) -> float:
    """Calculate material advantage for a color.
    
    Args:
        board: chess.Board
        color: chess.WHITE or chess.BLACK
        
    Returns:
        Positive if color has more material, negative otherwise
    """
    return material_value(board, color) - material_value(board, not color)


def phase(ply: int) -> str:
    """Categorize game phase by ply count.
    
    Args:
        ply: Current ply (half-move) number
        
    Returns:
        Phase string
    """
    if ply < 9:
        return "start"
    elif ply < 16:
        return "opening"
    elif ply < 31:
        return "early_mid"
    elif ply < 61:
        return "mid"
    elif ply < 121:
        return "late_mid"
    else:
        return "endgame"


def jsonl_write(path: Union[str, Path], rows: List[dict]) -> None:
    """Write a list of dicts to a JSONL file atomically.
    
    Args:
        path: Output file path
        rows: List of dictionaries to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    
    # Atomic rename
    os.replace(tmp_path, path)


def jsonl_read(path: Union[str, Path]) -> List[dict]:
    """Read a JSONL file into a list of dicts.
    
    Args:
        path: Input file path
        
    Returns:
        List of dictionaries
    """
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]
