"""Plug the OFFICIAL engine in here (strongly recommended: step 2 trains against it).

Return a chess.Move for `board` at engine `level` (1,2,3), or None to use the built-in approximation.
Example (adapt names to chess_hw):
    from chess_hw.engine import Engine
    _E = {}
    def engine_move(board, level):
        e = _E.setdefault(level, Engine(depth=level))
        return chess.Move.from_uci(e.best_move(board.fen()))
"""
def engine_move(board, level):
    return None
