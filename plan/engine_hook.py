"""OFFICIAL engine wired to chess_hw (the judge's own implementation).

`chess_hw.engine.Engine(level, seed, qdepth)` is exactly the engine used by the
official/paired eval (see chess_hw/match.py::_engine_task, which seeds it
f"engine-{seed}-{level}-{oid}-{model_color}"). Per-level single instances with a
stable seed keep plan/play.py deterministic like the paired eval.
"""

from chess_hw.engine import Engine

_E = {}


def engine_move(board, level, seed=None):
    key = (int(level), seed if seed is not None else f"engine-{int(level)}")
    eng = _E.get(key)
    if eng is None:
        eng = _E[key] = Engine(level=key[0], seed=key[1])
    return eng.best_move(board)  # chess.Move during a live position; None only if no legal move
