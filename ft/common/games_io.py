"""Games-file IO for the ft pipeline (Phase A output consumed by B and E)."""

from __future__ import annotations

from typing import List

from ft.common.serialize import jsonl_read, jsonl_write


def write_games(path: str, games: List[str]) -> int:
    jsonl_write(path, ({"moves": g} for g in games))
    return len(games)


def read_games(path: str) -> List[str]:
    games = []
    for row in jsonl_read(path):
        if "moves" in row and isinstance(row["moves"], str):
            games.append(row["moves"])
        else:
            raise ValueError(f"{path}: expected rows with a 'moves' string; "
                             f"got keys {sorted(row)} - rerun Phase A "
                             f"(ft/data/select_games.py)")
    return games
