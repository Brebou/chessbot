"""Phase A - game selection with a STRICT Elite Elo filter.

The Elite corpus (high-quality source of cloning labels and `normal` rows)
rejects any game where either Elo is missing or unparseable - never a silent
fallback. If the rejection rate exceeds 20% the dataset format is the
problem, and we fail fast instead of producing a biased corpus.

The general dump (blunder mine for contrastive pairs) does not need Elo at
all: use `min_elo=0` there.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_elo(value: object) -> Optional[int]:
    """Strict: ints, floats, integer strings; '?'/'None'/garbage -> None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    i = int(f)
    return i if i == f else None


@dataclass
class SelectionStats:
    read: int = 0
    kept: int = 0
    rejected_elo: int = 0
    rejected_malformed: int = 0

    @property
    def rejection_rate(self) -> float:
        considered = self.rejected_elo + self.kept
        return self.rejected_elo / considered if considered else 0.0


def _iter_rows(dataset: str):
    opener = gzip.open if str(dataset).endswith(".gz") else open
    with opener(dataset, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def _elo_pair(row: dict) -> Tuple[Optional[int], Optional[int]]:
    white = None
    black = None
    for key in ("WhiteElo", "White Elo"):
        if key in row:
            white = parse_elo(row[key])
            break
    for key in ("BlackElo", "Black Elo"):
        if key in row:
            black = parse_elo(row[key])
            break
    return white, black


def select_games(dataset: str, n_games: int, *, strict_elo: bool = False,
                 min_elo: int = 0, parser_reject_limit: float = 0.2
                 ) -> Tuple[List[str], SelectionStats]:
    """First n_games passing the filter; returns (moves strings, stats).

    strict_elo=True (Elite corpus): a game with a missing/unparseable Elo on
    either side is REJECTED. If the resulting rejection rate exceeds
    `parser_reject_limit`, raise - that is a parser/format bug, not data.
    """
    stats = SelectionStats()
    games: List[str] = []
    stopped = False
    for _, row in _iter_rows(dataset):
        if stopped:
            break
        moves = row.get("moves")
        if not isinstance(moves, str) or not moves.split():
            stats.read += 1
            stats.rejected_malformed += 1
            continue
        stats.read += 1
        if strict_elo or min_elo > 0:
            white, black = _elo_pair(row)
            if white is None or black is None:
                stats.rejected_elo += 1
                if strict_elo:
                    continue
            elif min(white, black) < min_elo:
                stats.rejected_elo += 1
                continue
        games.append(moves)
        stats.kept += 1
        if len(games) >= n_games:
            stopped = True
    if strict_elo and stats.kept > 0 and stats.rejection_rate > parser_reject_limit:
        raise ValueError(
            f"elite Elo rejection rate {stats.rejection_rate:.0%} exceeds "
            f"{parser_reject_limit:.0%}: the Elo parser does not match this "
            f"dump format ({stats.rejected_elo} rejected over "
            f"{stats.rejected_elo + stats.kept} games with moves)")
    return games, stats


def main() -> None:
    from ft.common.games_io import write_games

    parser = argparse.ArgumentParser(
        description="Phase A - game selection with the Elite Elo filter")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-games", type=int, default=150_000)
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--strict-elo", action="store_true",
                        help="reject any game with a missing/unparseable Elo "
                             "(Elite corpus only)")
    args = parser.parse_args()
    games, stats = select_games(args.dataset, args.n_games,
                                strict_elo=args.strict_elo,
                                min_elo=args.min_elo)
    n = write_games(args.out, games)
    print(f"\u2705 {n} games -> {args.out} (read {stats.read}, rejected_elo "
          f"{stats.rejected_elo}, malformed {stats.rejected_malformed})")


if __name__ == "__main__":
    main()
