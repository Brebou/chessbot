"""One command to build the ft training mix (Phases A-E of the spec).

Everything is written under ``--work-dir`` (default ``data/ft/``), each phase
is skipped when its output already exists (delete the file to force it):

    uv run ft/pipeline.py --dataset data/lichess_uci_201412.jsonl.gz \
        --n-games 200000 --min-elo 2400 --workers 36 --stockfish stockfish

Then train:
    uv run ft/sft/train_sft.py --model-path base_model \
        --mix data/ft/mix.jsonl --device cuda --workers 16
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ft.common.games_io import read_games, write_games
from ft.data.select_games import select_games


@dataclass
class PipelineConfig:
    dataset: str = "data/lichess_uci_201412.jsonl.gz"   # source with Elo headers
    work_dir: str = "data/ft"
    n_games: int = 150_000
    min_elo: int = 2400
    strict_elo: bool = False    # True = the Elite corpus (fail-fast parser)
    stockfish: str = "stockfish"
    depth: int = 14
    join_path: str = ""         # optional Lichess chess-position-evaluations index
    tb_dir: str = ""            # Syzygy 3-4-5 directory; Phase D skipped if empty
    tb_per_type: int = 15_000
    workers: int = 8
    seed: int = 0
    target_total: int = 0       # 0 = keep all classified rows (Phase E)
    force: bool = False

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        from ft.sft.train_sft import _add_dataclass_args
        _add_dataclass_args(cls, parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        from ft.sft.train_sft import _from_args_values
        return cls(**_from_args_values(cls, args))


def run_pipeline(cfg: PipelineConfig, annotate_fn=None) -> dict:
    from ft.data.annotate import annotate_positions
    from ft.data.build_buckets import build_bucket_dataset
    from ft.data.sample_positions import sample_positions
    from ft.data.tablebase import build_tb_rows

    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        "games": str(work / "games.jsonl"),
        "positions": str(work / "positions.jsonl"),
        "annotations": str(work / "annotations.jsonl"),
        "tb_rows": str(work / "tb_rows.jsonl"),
        "mix": str(work / "mix.jsonl"),
    }

    games_path = Path(paths["games"])
    if games_path.exists() and not cfg.force:
        print(f"✅ {games_path} exists, skipping Phase A")
    else:
        games, stats = select_games(cfg.dataset, cfg.n_games,
                                    strict_elo=cfg.strict_elo,
                                    min_elo=cfg.min_elo)
        n = write_games(paths["games"], games)
        print(f"✅ Phase A: {n} games -> {paths['games']} "
              f"(rejected_elo {stats.rejected_elo}, malformed "
              f"{stats.rejected_malformed})")

    positions_path = Path(paths["positions"])
    if positions_path.exists() and not cfg.force:
        print(f"✅ {positions_path} exists, skipping Phase B")
    else:
        n = sample_positions(read_games(paths["games"]), paths["positions"],
                             seed=cfg.seed, workers=cfg.workers)
        print(f"✅ Phase B: {n} positions -> {paths['positions']}")

    annotations_path = Path(paths["annotations"])
    if annotations_path.exists() and not cfg.force:
        print(f"✅ {annotations_path} exists, skipping Phase C")
    else:
        n = annotate_positions(paths["positions"], paths["annotations"],
                               engine_path=cfg.stockfish, depth=cfg.depth,
                               join_path=cfg.join_path, workers=cfg.workers,
                               annotate_fn=annotate_fn)
        print(f"✅ Phase C: {n} annotations -> {paths['annotations']}")

    tb_rows_path = None
    if cfg.tb_dir:
        tb_rows = Path(paths["tb_rows"])
        tb_rows_path = paths["tb_rows"]
        if tb_rows.exists() and not cfg.force:
            print(f"✅ {tb_rows} exists, skipping Phase D")
        else:
            n = build_tb_rows(cfg.tb_dir, tb_rows_path,
                              per_type=cfg.tb_per_type, seed=cfg.seed)
            print(f"✅ Phase D: {n} tb rows -> {tb_rows}")

    counts = build_bucket_dataset(read_games(paths["games"]),
                                  paths["positions"], paths["annotations"],
                                  paths["mix"], tb_rows_path=tb_rows_path or "",
                                  target_total=cfg.target_total or None,
                                  seed=cfg.seed, force=cfg.force)
    print(f"✅ Phase E: {sum(counts.values())} rows -> {paths['mix']} {counts}")
    return {**paths, "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ft SFT mix (Phases A-E)")
    PipelineConfig.add_cli_args(parser)
    args = parser.parse_args()
    run_pipeline(PipelineConfig.from_args(args))


if __name__ == "__main__":
    main()
