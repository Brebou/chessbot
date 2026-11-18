"""Fixtures and standardized diagnostics.

Fixtures are built ONCE (deterministic, cached under data/ft_fixtures/ by
the caller); every diagnostic is measured on them, never on fresh draws:

- mate1_500.jsonl:  positions with a verified mate-in-1 + the mating uci
- positions_{bin}_{n}.jsonl: real-style positions per bin (fen + history)

Registry targets (base -> iteration 1): top1-legal end > 80%, global
fallback < 5%, mate1 > 70%, winning captures > 85%, losing captures < 12%.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import chess

from chess_hw.model import load_model
from chess_hw.tokenizer import ChessTokenizer
from diag.fast_probe import FastProbePlayer
from diag.see import see
from ft.common.board_io import bin_for_pieces, history_window, n_pieces_from_fen
from ft.common.games_io import read_games
from ft.common.serialize import jsonl_write, jsonl_read


def _random_playout_game(rng: random.Random, max_plies: int = 160) -> List[str]:
    board = chess.Board()
    moves: List[str] = []
    for _ in range(max_plies):
        if board.outcome(claim_draw=True) is not None:
            break
        legal = list(board.legal_moves)
        mv = rng.choice(legal)
        moves.append(mv.uci())
        board.push(mv)
    return moves


def build_mate1_fixtures(n: int = 500, seed: int = 0,
                         max_attempts: int = 200_000, max_plies: int = 120) -> List[dict]:
    """Mate-in-1 positions induced from seeded random playouts (positions the
    model actually risks reaching), verified with the python-chess oracle."""
    rng = random.Random(f"mate1-fixture-{seed}")
    out: List[dict] = []
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board()
        moves: List[str] = []
        target_len = rng.randrange(10, max_plies)
        for _ in range(target_len):
            legal = list(board.legal_moves)
            if not legal:
                break
            mv = rng.choice(legal)
            moves.append(mv.uci())
            board.push(mv)
        if board.is_game_over():
            continue
        for mv in board.legal_moves:
            board.push(mv)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                out.append({"fen": board.fen(), "history": moves, "target": mv.uci()})
                break
    return out


def build_position_fixtures(n_per_bin: int, seed: int = 0,
                            games: Optional[List[str]] = None
                            ) -> Dict[str, List[dict]]:
    rng = random.Random(f"pos-fixture-{seed}")
    by_bin: Dict[str, List[dict]] = {"mid": [], "late": [], "end": []}
    need = n_per_bin * 3
    if games:
        for game in games:
            if sum(len(v) for v in by_bin.values()) >= need:
                break
            per_game = Counter()
            board = chess.Board()
            for i, uci in enumerate(game.split()):
                if all(per_game[b] >= 2 for b in by_bin):
                    break
                name = bin_for_pieces(n_pieces_from_fen(board.fen()))
                if per_game[name] < 2 and len(by_bin[name]) < n_per_bin:
                    by_bin[name].append({"fen": board.fen(),
                                         "history": game.split()[:i]})
                    per_game[name] += 1
                try:
                    board.push_uci(uci)
                except ValueError:
                    break
        return by_bin
    tries = 0
    while (sum(len(v) for v in by_bin.values()) < n_per_bin * 3
           and tries < n_per_bin * 200):
        tries += 1
        moves = _random_playout_game(rng)
        board = chess.Board()
        for i, uci in enumerate(moves):
            name = bin_for_pieces(n_pieces_from_fen(board.fen()))
            if len(by_bin[name]) < n_per_bin:
                by_bin[name].append({"fen": board.fen(), "history": moves[:i]})
            board.push_uci(uci)
    return by_bin


def build_fixtures(out_dir: str, *, n_mate1: int = 500, n_per_bin: int = 500,
                   seed: int = 0, games_path: str = "") -> dict:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = {}
    names = ("mid", "late", "end")
    pos_paths = {name: d / f"positions_{name}_{n_per_bin}.jsonl"
                 for name in names}
    if not all(q.exists() for q in pos_paths.values()):
        games = list(read_games(games_path)) if games_path else None
        for name, rows in build_position_fixtures(n_per_bin, seed,
                                                  games=games).items():
            if not pos_paths[name].exists():
                jsonl_write(pos_paths[name], rows)
    for name in names:
        written[f"positions_{name}"] = str(pos_paths[name])
    p = d / f"mate1_{n_mate1}.jsonl"
    if not p.exists():
        jsonl_write(p, build_mate1_fixtures(n_mate1, seed))
    written["mate1"] = str(p)
    return written


def fixture_metrics(model_path: str, fixtures_dir: str, *, device: str = "cpu",
                    batch_size: int = 1024) -> dict:
    model = load_model(model_path, device=device)
    tokenizer = ChessTokenizer()
    probe = FastProbePlayer(model, tokenizer, batch_size=batch_size)
    d = Path(fixtures_dir)
    metrics: Dict[str, object] = {}
    total_legal = total_decisions = 0

    mate_path = next(iter(sorted(d.glob("mate1_*.jsonl"))), None)
    if mate_path is not None:
        rows = list(jsonl_read(mate_path))
        boards = [chess.Board(r["fen"]) for r in rows]
        choices = probe.choose_from_histories(
            [history_window(r["history"]) for r in rows], boards)
        hits = sum(1 for r, c in zip(rows, choices)
                   if c.move is not None and c.move.uci() == r["target"])
        metrics["mate1_top1_rate"] = hits / max(len(rows), 1)

    for name in ("mid", "late", "end"):
        path = next(iter(sorted(d.glob(f"positions_{name}_*.jsonl"))), None)
        if path is None:
            continue
        rows = list(jsonl_read(path))
        boards = [chess.Board(r["fen"]) for r in rows]
        choices = probe.choose_from_histories(
            [history_window(r["history"]) for r in rows], boards)
        legal = sum(1 for c in choices if c.move is not None)
        metrics[f"top1_legal_{name}"] = legal / max(len(rows), 1)
        total_legal += legal
        total_decisions += len(rows)

        winning = losing = captures = 0
        for r, c in zip(rows, choices):
            if c.move is None:
                continue
            board = chess.Board(r["fen"])
            if not board.is_capture(c.move):
                continue
            captures += 1
            s = see(board, c.move)
            winning += s > 0
            losing += s < 0
        if captures:
            metrics[f"winning_capture_rate_{name}"] = winning / captures
            metrics[f"losing_capture_rate_{name}"] = losing / captures

    if total_decisions:
        metrics["top1_legal_global"] = total_legal / total_decisions
    return metrics
