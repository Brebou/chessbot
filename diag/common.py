"""Common utilities for the diagnostic harness.

Provides ProbePlayer for accessing full model logits, board utilities, and I/O helpers.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
import tempfile
import time
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


class ProgressLine:
    """Time-throttled single-line progress display for long-running scans.

    Usage:
        prog = ProgressLine()
        for i, item in enumerate(items):
            ...
            prog.show(f"{i}/{len(items)} done | {prog.rate(i):.1f}/s | {prog.eta(i, len(items))}")
        prog.close()
    """

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.started = time.monotonic()
        self._last = 0.0

    def show(self, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.min_interval:
            return
        self._last = now
        print(f"\r{message}", end="", flush=True)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def rate(self, done: int) -> float:
        return done / max(self.elapsed, 1e-9)

    def eta(self, done: int, total: int) -> str:
        rate = self.rate(done)
        if rate <= 0:
            return "estimating..."
        remaining = (total - done) / rate
        if remaining >= 90:
            return f"{remaining / 60:.1f} min left"
        return f"{remaining:.0f}s left"

    def close(self) -> None:
        if self._last:
            print()


def count_jsonl_rows(path: Union[str, Path]) -> int:
    """Count non-empty lines in a JSONL file (0 if missing/unreadable)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


LICHESS_DATA_DIR = Path("data")
LICHESS_HF_PARQUET_URL = ("https://huggingface.co/datasets/austindavis/lichess-uci/"
                          "resolve/main/data/{month}/train-{shard:05d}-of-{n_shards:05d}.parquet")

TRAIN_DATASET = "dlouapre/lichess_2025-01_1M"
TRAIN_DATASET_HF_PARQUET_URL = (f"https://huggingface.co/datasets/{TRAIN_DATASET}/"
                                "resolve/main/data/train-{shard:05d}-of-{n_shards:05d}.parquet")
TRAIN_DATASET_N_SHARDS = 2
TRAIN_SEEN_GAMES = 940_000
ANNOTATED_TOKEN_RE = re.compile(
    r"^([WB])([PNBRQK])([a-h][1-8][a-h][1-8])(?:\(([a-zA-Z+*]*)\))?$")
_PROMO_ANN_CHARS = set("qrbnQRBN")


def _download_file(url: str, dest: Path) -> None:
    """Stream a URL to disk with progress display, atomically."""
    import urllib.request

    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "chessbot-diagnostics"})
    tmp_path = dest.with_suffix('.tmp')
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, 'wb') as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r   {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB",
                          end="", flush=True)
        print()
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def ensure_lichess_dataset(month: str = "201412",
                           out_path: Union[str, Path, None] = None,
                           force: bool = False) -> Path:
    """Ensure the Lichess UCI dataset is available locally, downloading it if needed.

    Downloads one month of the austindavis/lichess-uci Hugging Face dataset (the
    official Lichess database already converted to UCI moves, with player Elo
    fields) and converts it to the gzipped JSONL format expected by the
    diagnostic pipeline: {"moves": "e2e4 e7e5 ...", "WhiteElo": int, "BlackElo": int}.

    Months are sharded on Hugging Face; all shards of the month are downloaded
    and concatenated. Both the raw parquet shards and the converted file are
    cached, so subsequent calls are no-ops. The default month (2014-12) contains
    ~1.3M games, which keeps P7's default seen/unseen split (index 940,000)
    usable.

    Args:
        month: Dataset month in YYYYMM format (default 2014-12, ~330 MB).
        out_path: Target path for the converted dataset
            (default data/lichess_uci_<month>.jsonl.gz).
        force: Re-convert even if the converted file already exists.

    Returns:
        Path to the converted dataset.
    """
    out_path = (Path(out_path) if out_path
                else LICHESS_DATA_DIR / f"lichess_uci_{month}.jsonl.gz")
    if out_path.exists() and not force:
        return out_path

    LICHESS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    import urllib.error

    shard_paths = []
    n_shards = None
    for candidate in range(1, 9):
        parquet_path = LICHESS_DATA_DIR / f"lichess_uci_{month}_{candidate - 1:05d}.parquet"
        if parquet_path.exists():
            shard_paths.append(parquet_path)
            continue
        url = LICHESS_HF_PARQUET_URL.format(
            month=month, shard=candidate - 1, n_shards=candidate)
        try:
            _download_file(url, parquet_path)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if candidate == 1:
                    continue
                n_shards = candidate - 1
                break
            raise
        shard_paths.append(parquet_path)
    if n_shards is None:
        n_shards = len(shard_paths)
    if not shard_paths:
        raise ValueError(f"No parquet shards found for month {month} "
                         f"(check the month and your connection)")
    if len(shard_paths) < n_shards:
        for idx in range(len(shard_paths), n_shards):
            parquet_path = LICHESS_DATA_DIR / f"lichess_uci_{month}_{idx:05d}.parquet"
            url = LICHESS_HF_PARQUET_URL.format(month=month, shard=idx, n_shards=n_shards)
            _download_file(url, parquet_path)
            shard_paths.append(parquet_path)

    print(f"Converting {len(shard_paths)} parquet shard(s) -> {out_path} ...")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read the downloaded parquet files; "
            "install it with 'uv add pyarrow'") from exc

    tmp_out = out_path.with_suffix('.tmp')
    n_games = 0
    with gzip.open(tmp_out, 'wt') as out:
        for parquet_path in shard_paths:
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(batch_size=50_000,
                                         columns=["Transcript", "WhiteElo", "BlackElo"]):
                cols = batch.to_pydict()
                for moves, white_elo, black_elo in zip(cols["Transcript"],
                                                       cols["WhiteElo"], cols["BlackElo"]):
                    if not moves:
                        continue
                    out.write(json.dumps({"moves": moves,
                                          "WhiteElo": white_elo,
                                          "BlackElo": black_elo}) + '\n')
                    n_games += 1
    os.replace(tmp_out, out_path)
    print(f"   {n_games} games written to {out_path}")
    return out_path


def parse_annotated_game(text: str) -> List[str]:
    """Convert one game from the training dataset notation to UCI moves.

    The training dataset (dlouapre/lichess_2025-01_1M) encodes each move as a
    token like ``WQd1h5(x+)``: side (W/B) + moving piece (P/N/B/R/Q/K) + UCI
    square pair + optional parenthesized annotations. Annotations are
    informational except promotions, which are only encoded there: ``x``
    capture, ``+`` check, ``*`` mate, ``o``/``O`` castling, ``E`` en passant,
    and ``Q``/``R``/``B``/``N`` promotion piece.

    Raises:
        ValueError: on a token that does not match the notation.
    """
    moves = []
    for tok in text.split():
        m = ANNOTATED_TOKEN_RE.match(tok)
        if m is None:
            raise ValueError(f"Unparsable move token: {tok!r}")
        uci = m.group(3)
        for ch in m.group(4) or "":
            if ch in _PROMO_ANN_CHARS:
                uci += ch.lower()
        moves.append(uci)
    return moves


def _valid_uci_game(moves: List[str]) -> bool:
    board = chess.Board()
    try:
        for uci in moves:
            board.push_uci(uci)
    except Exception:
        return False
    return True


def _convert_train_block(texts: List[str]) -> Tuple[int, List[str], int]:
    """Convert a block of raw game texts to UCI JSON lines (pool worker).

    Returns (n_invalid, json_lines, n_valid): games that fail replay validation
    are dropped so downstream corpora and probes only see legal move sequences.
    """
    lines = []
    n_valid = 0
    n_invalid = 0
    for text in texts:
        try:
            moves = parse_annotated_game(text)
        except ValueError:
            n_invalid += 1
            continue
        if not moves or not _valid_uci_game(moves):
            n_invalid += 1
            continue
        lines.append(json.dumps({"moves": " ".join(moves)}, sort_keys=True))
        n_valid += 1
    return n_invalid, lines, n_valid


def ensure_training_dataset(out_path: Union[str, Path, None] = None,
                            workers: int = 4,
                            force: bool = False) -> Path:
    """Ensure the model's training dataset is available as pipeline JSONL.

    Downloads dlouapre/lichess_2025-01_1M (the exact corpus ChessGPT was
    pre-trained on: the first 940,000 games are the model's training data, the
    remaining ~60,000 are unseen) and converts the annotated notation to UCI
    with per-game replay validation. Cached, so subsequent calls are no-ops.

    Args:
        out_path: Target path (default data/lichess_train_2025-01_1M.jsonl.gz).
        workers: Worker processes for conversion/validation.
        force: Re-convert even if the converted file already exists.

    Returns:
        Path to the converted dataset.
    """
    out_path = Path(out_path) if out_path else LICHESS_DATA_DIR / "lichess_train_2025-01_1M.jsonl.gz"
    if out_path.exists() and not force:
        return out_path

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read the downloaded parquet files; "
            "install it with 'uv add pyarrow'") from exc

    LICHESS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard in range(TRAIN_DATASET_N_SHARDS):
        parquet_path = LICHESS_DATA_DIR / f"dlouapre_train_{shard:05d}.parquet"
        if not parquet_path.exists():
            url = TRAIN_DATASET_HF_PARQUET_URL.format(
                shard=shard, n_shards=TRAIN_DATASET_N_SHARDS)
            _download_file(url, parquet_path)
        shard_paths.append(parquet_path)

    print(f"Converting {TRAIN_DATASET} -> {out_path} "
          f"(replay validation, {workers} workers) ...")
    from multiprocessing import get_context

    n_valid = 0
    n_invalid = 0
    tmp_out = out_path.with_suffix('.tmp')
    pool = None
    try:
        if workers > 1:
            pool = get_context("spawn").Pool(workers)

        def blocks():
            block = []
            for parquet_path in shard_paths:
                pf = pq.ParquetFile(parquet_path)
                for batch in pf.iter_batches(batch_size=20_000, columns=["text"]):
                    block.extend(batch.column("text").to_pylist())
                    while len(block) >= 20_000:
                        yield block[:20_000]
                        block = block[20_000:]
            if block:
                yield block

        with gzip.open(tmp_out, 'wt') as out:
            convert = _convert_train_block
            source = pool.imap(convert, blocks(), chunksize=1) if pool else map(convert, blocks())
            for n_inv, lines, n in source:
                n_invalid += n_inv
                n_valid += n
                if lines:
                    out.write("\n".join(lines) + "\n")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    os.replace(tmp_out, out_path)
    print(f"   {n_valid} valid games written ({n_invalid} dropped); "
          f"first {TRAIN_SEEN_GAMES} = seen, rest = unseen for P7")
    return out_path


def jsonl_read(path: Union[str, Path]) -> List[dict]:
    """Read a JSONL file into a list of dicts.
    
    Args:
        path: Input file path
        
    Returns:
        List of dictionaries
    """
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]
