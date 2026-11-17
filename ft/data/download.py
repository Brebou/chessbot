"""Dataset fetchers for Phase A (the "good datasets" of the ft mix).

Two sources:

- **Elite** (`austindavis/lichess-elite-uci`): the Lichess Elite Database
  (2400+ vs 2200+, 2500+ vs 2300+ from 2021 on, bullet excluded) already
  converted to UCI, single `text` column. Elite BY CONSTRUCTION - no Elo
  parsing, `min_elo` does not apply. Each default-config shard holds
  ~980k games.
- **General** (`austindavis/lichess-uci` monthly exports, via the existing
  `diag.common.ensure_lichess_dataset`): all levels, WhiteElo/BlackElo
  columns (0 = unrated in the 2014 exports).

Everything is cached under data/ (gitignored) and written atomically; a
re-run is a no-op. Downloads use plain urllib (no new dependency).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ft.common.serialize import jsonl_count

ELITE_REPO_BASE = ("https://huggingface.co/datasets/austindavis/lichess-elite-uci"
                   "/resolve/main/data")
ELITE_DEFAULT_SHARDS = 6
RAW_DIR = Path("data") / "raw_parquet"


def download_file(url: str, target: Path, *, chunk: int = 1 << 20) -> Path:
    """Plain urllib download to an atomic tmp file (repo convention)."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as src, \
            open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=chunk)
    os.replace(tmp, target)
    return target


def convert_parquet_to_games(parquet_paths, out: str, *, n_games: int = 0) -> int:
    """Read {text: "uci uci ..."} (or {Transcript: ...}) parquet shards ->
    {"moves": ...} jsonl.gz, the ft pipeline games format. n_games=0 keeps
    everything."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required ('uv add pyarrow')") from exc

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for parquet_path in parquet_paths:
            pf = pq.ParquetFile(parquet_path)
            cols = ("text",) if "text" in pf.schema_arrow.names else ("Transcript",)
            for batch in pf.iter_batches(batch_size=50_000, columns=list(cols)):
                for moves in batch.to_pydict()[cols[0]]:
                    if not moves or not moves.strip():
                        continue
                    f.write(json.dumps({"moves": moves.strip()}) + "\n")
                    n += 1
                    if n_games and n >= n_games:
                        os.replace(tmp, out_path)
                        return n
    os.replace(tmp, out_path)
    return n


def ensure_elite_dataset(out: str = "data/ft/elite_games.jsonl.gz", *,
                         n_shards: int = 1, n_games: int = 0,
                         force: bool = False) -> tuple:
    """Download `n_shards` default-config parquet shards of the Lichess Elite
    Database and convert to games jsonl.gz. Returns (path, n_games)."""
    out_path = Path(out)
    if out_path.exists() and not force:
        return out, jsonl_count(out_path)
    shard_paths = []
    for i in range(n_shards):
        name = f"train-{i:05d}-of-{ELITE_DEFAULT_SHARDS:05d}.parquet"
        shard = RAW_DIR / f"elite_uci_{name}"
        url = f"{ELITE_REPO_BASE}/{name}"
        print(f"fetching {url} -> {shard} (skipped when already cached)")
        download_file(url, shard)
        shard_paths.append(shard)
    n = convert_parquet_to_games(shard_paths, out, n_games=n_games)
    print(f"✅ elite dataset: {n} games -> {out_path}")
    return out, n


def ensure_general_dataset(month: str = "201412", out: str | None = None) -> Path:
    """Existing month dump with Elo columns (see diag.common)."""
    from diag.common import ensure_lichess_dataset
    return ensure_lichess_dataset(month, out_path=out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the ft Phase-A sources (elite / general month)")
    parser.add_argument("--elite", type=int, metavar="N_SHARDS", default=None,
                        help="download the Lichess Elite UCI corpus "
                             "(arg = number of shards, 1 shard ~ 980k games)")
    parser.add_argument("--elite-out", default="data/ft/elite_games.jsonl.gz")
    parser.add_argument("--elite-games", type=int, default=0,
                        help="keep only the first N games (0 = all)")
    parser.add_argument("--general", metavar="YYYYMM", default=None,
                        help="download one austindavis/lichess-uci month "
                             "with Elo columns")
    args = parser.parse_args()
    if args.elite is None and args.general is None:
        parser.error("nothing to do: pass --elite N or --general YYYYMM")
    if args.elite is not None:
        ensure_elite_dataset(args.elite_out, n_shards=args.elite,
                             n_games=args.elite_games)
    if args.general is not None:
        ensure_general_dataset(args.general)


if __name__ == "__main__":
    main()
