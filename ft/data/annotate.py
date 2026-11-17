"""Phase C - one engine annotation pass, the backbone of every bucket.

Per annotated position (jsonl; parquet/zstd can be swapped in later, the
schema is what matters):

    fen, uci_played, cp_root, cp_best, best_uci, second_uci, cp_second,
    cp_played_after, mate_root, is_check_before, margin, sanity

Score convention: centipawns from the mover's POV *before* the played move;
Mate(n) maps to 10000 - n (and -10000 + n when the mover is being mated).
`cp_played_after` = eval of the position after the played move, still from
the mover's POV. `margin = cp_best - cp_played_after >= 0` is the loss the
human move caused - the single criterion used by buckets. `mate_root` is
None, or +N (mover mates in N) / -N (mover is mated in N).

Engine settings are non-negotiable for reproducibility: fixed depth (no time
limits), MultiPV 2 (the second line powers the tranquility criterion),
Threads 1 per worker, parallelism per FEN. A join index from Lichess
`chess-position-evaluations` (optional jsonl of fen->eval) is consulted
first; uncovered FENs go to Stockfish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ft.common.serialize import jsonl_read

MATE_CP = 10000


def score_to_mover(score) -> int:
    """chess.engine.PovScore -> centipawns from the side-to-move POV."""
    relative = score.relative
    if relative.is_mate():
        n = relative.mate() or 1
        return (MATE_CP - abs(n)) if n > 0 else -(MATE_CP - abs(n))
    return int(max(-MATE_CP, min(MATE_CP, relative.score(mate_score=MATE_CP))))


class StockfishAnnotator:
    """In-process annotator; each pool worker owns one engine exclusively."""

    def __init__(self, path: str = "stockfish", depth: int = 14, multipv: int = 2):
        self.path = path
        self.depth = depth
        self.multipv = multipv
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            import chess.engine
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Threads": 1})
        return self._engine

    def close(self):
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def _analyse(self, board: chess.Board):
        limit = chess.engine.Limit(depth=self.depth)
        infos = self.engine.analyse(board, limit, multipv=self.multipv)
        if isinstance(infos, dict):
            infos = [infos]
        return sorted(infos,
                      key=lambda i: i["score"].pov(board.turn).score(mate_score=MATE_CP),
                      reverse=True)

    def annotate(self, fen: str) -> dict:
        board = chess.Board(fen)
        infos = self._analyse(board)
        first = infos[0]
        score = first["score"].pov(board.turn)
        mate_root = score.mate() if score.is_mate() and score.mate() != 0 else None
        return {"cp_best": score_to_mover(first["score"]),
                "best_uci": first["pv"][0].uci(),
                "cp_second": score_to_mover(infos[1]["score"]) if len(infos) > 1
                else score_to_mover(first["score"]),
                "second_uci": infos[1]["pv"][0].uci() if len(infos) > 1 else None,
                "mate_root": mate_root}

    def eval_after(self, fen: str, uci: str) -> dict:
        board = chess.Board(fen)
        board.push_uci(uci)
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        return {"cp_played_after": -score_to_mover(info["score"])}


class JoinAnnotator:
    """Precomputed Lichess `chess-position-evaluations` rows joined by FEN.

    Expected jsonl keys: fen, cp_best, best_uci, cp_second?, second_uci?,
    mate_root?, cp_played_after?, uci_played? (mover-POV, same units).
    """

    def __init__(self, index_path: str):
        self.index = {r["fen"]: r for r in jsonl_read(index_path)}

    def lookup(self, fen: str):
        return self.index.get(fen)


class ChainedAnnotator:
    """Lichess-evals join first, Stockfish for uncovered FENs."""

    def __init__(self, join: JoinAnnotator, stockfish: StockfishAnnotator):
        self.join = join
        self.stockfish = stockfish

    def annotate(self, fen: str) -> dict:
        row = self.join.lookup(fen)
        if row is None:
            return self.stockfish.annotate(fen)
        if row.get("cp_played_after") is None:
            after_uci = row.get("uci_played", row["best_uci"])
            row["cp_played_after"] = self.stockfish.eval_after(
                fen, after_uci)["cp_played_after"]
        return row


class _SFOnly:
    """Uniform wrapper so the worker need not know whether a join exists."""

    def __init__(self, sf: StockfishAnnotator):
        self.stockfish = sf

    def annotate(self, fen: str) -> dict:
        return self.stockfish.annotate(fen)


def _build_annotator(engine_path: str, depth: int, join_path: str):
    sf = StockfishAnnotator(engine_path, depth)
    if join_path:
        return ChainedAnnotator(JoinAnnotator(join_path), sf)
    return _SFOnly(sf)


_WORKER_ANNOTATOR = {}


def annotate_task(task):
    """Module-level spawn worker: (fen, uci_played, engine_path, depth, join_path)."""
    fen, uci_played, engine_path, depth, join_path = task
    import chess.engine as _ce

    key = (engine_path, depth, join_path)
    if key not in _WORKER_ANNOTATOR:
        _WORKER_ANNOTATOR[key] = _build_annotator(engine_path, depth, join_path)
    annotator = _WORKER_ANNOTATOR[key]
    try:
        board = chess.Board(fen)
        played = board.parse_uci(uci_played)
    except ValueError:
        annotator.stockfish.close()
        return {"fen": fen, "uci_played": uci_played, "invalid": True}
    block = annotator.annotate(fen)
    after = chess.Board(fen)
    after.push(played)
    block["cp_played_after"] = -score_to_mover(
        annotator.stockfish.engine.analyse(after, _ce.Limit(depth=depth))["score"])
    annotator.stockfish.close()
    row = {"fen": fen, "uci_played": uci_played,
           "cp_root": block["cp_best"], **block,
           "is_check_before": chess.Board(fen).is_check()}
    row["margin"] = row["cp_best"] - row["cp_played_after"]
    row["sanity"] = row["cp_best"] >= row["cp_second"]
    return row


def _count_valid_lines(path: Path) -> int:
    """Number of complete jsonl rows; a torn last line (crash mid-write) does
    not count and will be overwritten on resume."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                break
            n += 1
    return n


def annotate_positions(positions_path: str, out: str, *, engine_path: str = "stockfish",
                       depth: int = 14, join_path: str = "", workers: int = 8,
                       progress=None, annotate_fn: Optional[Callable] = None) -> int:
    """`annotate_fn` (optional) replaces the Stockfish pools (used by tests and
    the orchestrator); it maps fen -> partial annotation dict {best_uci,
    cp_best, cp_second, second_uci?, mate_root?} and the rest of the row is
    filled here with the same schema as the pool worker.

    Rows are written incrementally to `<out>.tmp` (renamed at the end), so a
    multi-hour run stays observable and restartable: with the positions file
    unchanged, a rerun parses the valid prefix of the tmp file and only
    annotates what is missing. A partial tmp file is tolerated (torn last line
    is ignored); delete <out>.tmp to start over.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    resumed = _count_valid_lines(tmp) if tmp.exists() else 0
    if tmp.exists() and not resumed:
        tmp.unlink()
    if resumed:
        print(f"↩️ Phase C resume: {resumed} rows already in {tmp}, "
              f"skipping those positions")
    rows_iter = jsonl_read(positions_path)
    if resumed:
        import itertools

        rows_iter = itertools.islice(rows_iter, resumed, None)
    n = 0
    with open(tmp, "a", encoding="utf-8") as f:
        def _emit(row: dict) -> None:
            nonlocal n
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
            if n % 10_000 == 0:
                f.flush()

        if annotate_fn is not None:
            for r in rows_iter:
                _emit(_annotate_row(r, annotate_fn(r["fen_before"])))
                if progress is not None:
                    progress.show(f"annotate: {resumed + n}")
        else:
            from multiprocessing import get_context

            tasks = [(r["fen_before"], r["uci_played"], engine_path, depth, join_path)
                     for r in rows_iter]
            ctx = get_context("spawn")
            with ctx.Pool(workers) as pool:
                for res in pool.imap(annotate_task, tasks, chunksize=4):
                    _emit(res)
                    if progress is not None:
                        progress.show(f"annotate: {resumed + n}/{resumed + len(tasks)}")
    os.replace(tmp, out_path)
    return resumed + n


def _annotate_row(position_row: dict, block: Dict) -> dict:
    fen = position_row["fen_before"]
    try:
        played = chess.Board(fen).parse_uci(position_row["uci_played"]).uci()
    except ValueError:
        return {"fen": fen, "uci_played": position_row["uci_played"],
                "invalid": True}
    row = {block_key: block_value for block_key, block_value in block.items()}
    row.update({"fen": fen, "uci_played": played,
                "cp_root": block["cp_best"],
                "cp_played_after": block.get("cp_played_after", block["cp_best"]),
                "is_check_before": chess.Board(fen).is_check(),
                "sanity": True})
    if "cp_played_after" in block:
        row["margin"] = row["cp_best"] - row["cp_played_after"]
    else:
        row["margin"] = 0
        row["cp_played_after"] = row["cp_best"]
    row.setdefault("second_uci", None)
    row.setdefault("mate_root", None)
    return row


def main() -> None:
    from diag.common import ProgressLine

    parser = argparse.ArgumentParser(
        description="Phase C - Stockfish annotation pass (depth, MultiPV 2)")
    parser.add_argument("--positions", required=True, help="Phase B output jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--join", default="",
                        help="optional Lichess chess-position-evaluations index")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    n = annotate_positions(args.positions, args.out, engine_path=args.engine,
                           depth=args.depth, join_path=args.join,
                           workers=args.workers, progress=ProgressLine())
    print(f"\n✅ {n} annotations -> {args.out}")


if __name__ == "__main__":
    main()
