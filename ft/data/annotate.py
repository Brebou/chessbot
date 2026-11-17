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

import chess
from ft.common.serialize import jsonl_read, jsonl_write

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


def annotate_positions(positions_path: str, out: str, *, engine_path: str = "stockfish",
                       depth: int = 14, join_path: str = "", workers: int = 8,
                       progress=None) -> int:
    from multiprocessing import get_context

    rows = list(jsonl_read(positions_path))
    tasks = [(r["fen_before"], r["uci_played"], engine_path, depth, join_path)
             for r in rows]
    out_rows = []
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for res in pool.imap(annotate_task, tasks, chunksize=4):
            out_rows.append(res)
            if progress is not None:
                progress.show(f"annotate: {len(out_rows)}/{len(tasks)}")
    jsonl_write(out, out_rows)
    return len(out_rows)
