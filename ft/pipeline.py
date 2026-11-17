"""One command to build the ft training mix (Phases A-E of the spec).

Everything is written under ``--work-dir`` (default ``data/ft/``), each phase
is skipped when its output already exists (delete the file to force it). By
default Phase A auto-downloads the Lichess Elite UCI corpus (2400+/2200+,
bullet excluded); pass ``--dataset`` to use a local jsonl.gz instead:

    uv run ft/pipeline.py --n-games 200000 --workers 36 --stockfish stockfish

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
from ft.common.serialize import jsonl_count
from ft.data.download import ensure_elite_dataset
from ft.data.select_games import select_games


@dataclass
class PipelineConfig:
    dataset: str = ""           # empty = auto-download the Lichess Elite UCI corpus
    work_dir: str = "data/ft"
    n_games: int = 150_000
    min_elo: int = 0            # only applies to an explicit dataset with Elo columns
    strict_elo: bool = False    # True = reject instead of falling back on unusable Elo
    auto_download: bool = True  # fall back to the elite corpus when the dataset keeps 0 games
    elite_shards: int = 1       # elite parquet shards (1 shard ~ 980k games)
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


def _phase_a_games(cfg: PipelineConfig):
    """Phase A source resolution + selection. Returns (games, stats, source).

    Priority: an explicitly configured dataset; then the already-downloaded
    elite corpus; then an auto-download of the elite corpus (elite BY
    CONSTRUCTION - min_elo does not apply, so few Elo-0 rejections). A
    configured month dump whose Elo columns are all unusable (e.g. the 2014
    exports where every Elo is 0) triggers the elite fallback too.
    """
    if cfg.dataset:
        if not Path(cfg.dataset).exists():
            print(f"⚠️ dataset {cfg.dataset} does not exist - "
                  f"falling back to the Lichess Elite corpus")
        else:
            games, stats = select_games(cfg.dataset, cfg.n_games,
                                        strict_elo=cfg.strict_elo,
                                        min_elo=cfg.min_elo)
            if games or not cfg.auto_download:
                return games, stats, cfg.dataset
            print(f"⚠️ {cfg.dataset} kept 0/{stats.rejected_elo + stats.kept} games "
                  f"(Elo fields missing or all 0 = unrated in this export) - "
                  f"falling back to the Lichess Elite corpus")
    elite, n = ensure_elite_dataset(str(Path(cfg.work_dir) / "elite_games.jsonl.gz"),
                                    n_shards=cfg.elite_shards,
                                    n_games=cfg.n_games)
    games, stats = select_games(elite, cfg.n_games, strict_elo=False, min_elo=0)
    return games, stats, elite


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
        games, stats, games_source = _phase_a_games(cfg)
        if not games:
            raise SystemExit(
                "⛔ Phase A kept 0 games - no usable dataset. Delete "
                f"{paths['games']} and rerun so the pipeline downloads the "
                "Lichess Elite corpus (see `uv run ft/data/download.py --elite 1`).")
        n = write_games(paths["games"], games)
        print(f"✅ Phase A: {n} games -> {paths['games']} (source "
              f"{games_source}, rejected_elo {stats.rejected_elo}, malformed "
              f"{stats.rejected_malformed})")

    if not read_games(paths["games"]):
        raise SystemExit(
            f"⛔ {paths['games']} contains 0 games - the mix would be empty. "
            "Delete it and rerun (see `uv run ft/data/download.py --elite 1`).")

    positions_path = Path(paths["positions"])
    if positions_path.exists() and not cfg.force:
        print(f"✅ {positions_path} exists, skipping Phase B")
    else:
        from diag.common import ProgressLine

        print(f"⏳ Phase B: replaying games and sampling positions "
              f"(minutes)...")
        n = sample_positions(read_games(paths["games"]), paths["positions"],
                             seed=cfg.seed, workers=cfg.workers,
                             progress=ProgressLine())
        print(f"\n✅ Phase B: {n} positions -> {paths['positions']}")

    annotations_path = Path(paths["annotations"])
    if annotations_path.exists() and not cfg.force:
        print(f"✅ {annotations_path} exists, skipping Phase C")
    else:
        from diag.common import ProgressLine

        n_positions = jsonl_count(paths["positions"])
        print(f"⏳ Phase C: annotating {n_positions} positions, depth {cfg.depth}, "
              f"engine {cfg.stockfish}, {cfg.workers} workers - THE LONG STEP "
              f"(~2 searches/position; live progress on the console and rows "
              f"stream into {annotations_path}.tmp as they complete)")
        n = annotate_positions(paths["positions"], paths["annotations"],
                               engine_path=cfg.stockfish, depth=cfg.depth,
                               join_path=cfg.join_path, workers=cfg.workers,
                               progress=ProgressLine(), annotate_fn=annotate_fn)
        print(f"\n✅ Phase C: {n} annotations -> {paths['annotations']}")

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
