"""Tests for the ft pipeline orchestration and script entrypoints."""

import json
import sys
from pathlib import Path

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ft.common.games_io import read_games, write_games
from ft.data.download import convert_parquet_to_games
from ft.pipeline import PipelineConfig, _phase_a_games, run_pipeline


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from chess_hw.model import CONFIG, ChessGPT, save_model

    path = tmp_path_factory.mktemp("ft_cli_model")
    torch.manual_seed(0)
    save_model(ChessGPT(CONFIG), path / "model.safetensors")
    return path


def _random_games(n: int, rng) -> list:
    games = []
    for i in range(n):
        board = chess.Board()
        moves = []
        for _ in range(10 + i % 15):
            legal = list(board.legal_moves)
            if not legal or board.outcome(claim_draw=True) is not None:
                break
            mv = rng.choice(legal)
            moves.append(mv.uci())
            board.push(mv)
        games.append(" ".join(moves))
    return games


def _stub_annotate(fen: str) -> dict:
    board = chess.Board(fen)
    mv = sorted(board.legal_moves, key=lambda m: m.uci())[0]
    return {"best_uci": mv.uci(), "cp_best": 0, "cp_second": 0}


def test_games_io_roundtrip(tmp_path):
    games = ["e2e4 e7e5", "d2d4"]
    p = tmp_path / "games.jsonl"
    assert write_games(str(p), games) == 2
    assert read_games(str(p)) == games


@pytest.fixture(scope="module")
def mini_pipeline(tmp_path_factory):
    import random

    root = tmp_path_factory.mktemp("ft_pipe")
    raw = root / "raw_games.jsonl"
    write_games(str(raw), _random_games(40, random.Random(0)))
    rows = [{"moves": json.loads(line)["moves"],
             "WhiteElo": 2500, "BlackElo": 2450}
            for line in raw.read_text().splitlines()]
    src = root / "src.jsonl"
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cfg = PipelineConfig(dataset=str(src), work_dir=str(root / "ft"),
                         n_games=40, min_elo=0, stockfish="", tb_dir="",
                         workers=2, seed=0)
    paths = run_pipeline(cfg, annotate_fn=_stub_annotate)
    return cfg, paths


def test_pipeline_builds_all_phase_outputs(mini_pipeline):
    cfg, paths = mini_pipeline
    for key in ("games", "positions", "annotations", "mix"):
        assert Path(paths[key]).exists(), key
    counts = paths["counts"]
    assert sum(counts.values()) > 0
    rows = [json.loads(l) for l in open(paths["mix"]) if l.strip()]
    assert all(r["bucket"] in {"normal", "mate", "endgame_tb",
                               "contrastive", "anchor"} for r in rows)
    for r in rows:
        chess.Board(r["key"]).parse_uci(r["target"])


def test_pipeline_second_run_uses_cache(mini_pipeline):
    cfg, paths = mini_pipeline
    before = Path(paths["mix"]).read_text()
    again = run_pipeline(cfg, annotate_fn=_stub_annotate)
    assert Path(again["mix"]).read_text() == before
    assert sum(again["counts"].values()) == sum(paths["counts"].values())


def test_pipeline_skips_phase_d_without_tb(mini_pipeline):
    _, paths = mini_pipeline
    assert not Path(paths["tb_rows"]).exists()


def test_trainer_missing_mix_hint(tiny_model, tmp_path):
    from ft.sft.train_sft import FTSFTConfig, FTSFTTrainer

    cfg = FTSFTConfig(model_path=str(tiny_model),
                      mix_path=str(tmp_path / "nope.jsonl"),
                      out_dir=str(tmp_path / "run"), device="cpu",
                      eval_mode="none", eval_every_steps=0, fixtures_dir="")
    with pytest.raises(SystemExit, match="ft/pipeline.py"):
        FTSFTTrainer(cfg)


def test_pipeline_config_cli_roundtrip():
    import argparse

    parser = argparse.ArgumentParser()
    PipelineConfig.add_cli_args(parser)
    args = parser.parse_args(["--dataset", "d.jsonl.gz", "--n-games", "10",
                              "--min-elo", "2400", "--force", "true"])
    cfg = PipelineConfig.from_args(args)
    assert cfg.n_games == 10 and cfg.min_elo == 2400 and cfg.force is True


def test_select_games_treats_elo_zero_as_unrated(tmp_path):
    from ft.data.select_games import select_games

    src = tmp_path / "src.jsonl"
    rows = [{"moves": "e2e4 e7e5", "WhiteElo": 0, "BlackElo": 2450},
            {"moves": "d2d4 d7d5", "WhiteElo": 2500, "BlackElo": 0},
            {"moves": "c2c4 c7c5", "WhiteElo": 2600, "BlackElo": 2500}]
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    games, _ = select_games(str(src), 10, strict_elo=True, min_elo=2400)
    assert games == ["c2c4 c7c5"]
    games0, _ = select_games(str(src), 10, min_elo=0)
    assert len(games0) == 3


def test_convert_parquet_to_games(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard = tmp_path / "s.parquet"
    pq.write_table(pa.table({"text": pa.array([
        "d2d4 d7d5  ", "", "b1c3 b8c6"])}), shard)
    out = tmp_path / "games.jsonl.gz"
    n = convert_parquet_to_games([shard], str(out))
    assert n == 2
    assert read_games(str(out)) == ["d2d4 d7d5", "b1c3 b8c6"]


def test_phase_a_fallback_to_elite(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    fake_shard = tmp_path / "fake.parquet"
    pq.write_table(pa.table({"text": pa.array(["e2e4 e7e5", "d2d4 d7d5"])}),
                   fake_shard)
    monkeypatch.setattr("ft.pipeline.ensure_elite_dataset",
                        _fake_ensure(tmp_path, fake_shard))
    cfg = PipelineConfig(dataset=str(tmp_path / "absent.jsonl"), work_dir=str(tmp_path),
                         n_games=2, min_elo=2400, auto_download=True)
    games, stats, source = _phase_a_games(cfg)
    assert games == ["e2e4 e7e5", "d2d4 d7d5"]
    assert "elite" in str(source) or "fake" in str(source)


def _fake_ensure(tmp_path, shard):
    from ft.data.download import convert_parquet_to_games

    def _ensure(out, *, n_shards=1, n_games=0, force=False):
        convert_parquet_to_games([shard], out, n_games=n_games)
        return out, 2
    return _ensure


def test_pipeline_aborts_with_empty_source(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    empty_shard = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), empty_shard)
    monkeypatch.setattr("ft.pipeline.ensure_elite_dataset",
                        _fake_ensure(tmp_path, empty_shard))
    cfg = PipelineConfig(dataset="", work_dir=str(tmp_path), n_games=10,
                         auto_download=True, stockfish="")
    with pytest.raises(SystemExit, match="Phase A kept 0 games"):
        run_pipeline(cfg, annotate_fn=_stub_annotate)


def test_pipeline_aborts_on_empty_games_file(tmp_path):
    write_games(str(tmp_path / "games.jsonl"), [])  # empty output from a failed run
    cfg = PipelineConfig(dataset="", work_dir=str(tmp_path), auto_download=False,
                         stockfish="")
    with pytest.raises(SystemExit, match="0 games"):
        run_pipeline(cfg, annotate_fn=_stub_annotate)


def test_download_module_importable():
    import ft.data.download as d
    assert d.ELITE_DEFAULT_SHARDS == 6


class _StubTB:
    """wdl is given from White's POV and re-signed per side-to-move, mirroring
    the Syzygy contract (positive = side to move wins)."""

    def __init__(self, wdl: int, dtz: int):
        self.wdl, self.dtz = wdl, dtz

    def wdl_dtz(self, board):
        try:
            board.legal_moves
        except Exception:
            return None
        sign = 1 if board.turn == chess.WHITE else -1
        return (sign * self.wdl, sign * self.dtz)


def test_sample_tb_starts_skips_out_of_range_dtz(tmp_path):
    from ft.data.tablebase import sample_tb_starts

    starts = sample_tb_starts(_StubTB(2, 45), "KRK", 5, seed=0,
                              max_attempts=200)
    assert starts == []


def test_tb_relabel_returns_tb_target_for_winning_position():
    from ft.data.build_buckets import _tb_relabel

    board = chess.Board()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    target = _tb_relabel(board, _StubTB(2, 12))
    assert target is not None and chess.Move.from_uci(target)

    board2 = chess.Board()
    board2.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board2.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert _tb_relabel(board2, _StubTB(0, 5)) is None

_FOOLS_MATE = "f2f3 e7e5 g2g4 d8h4"
_SCHOLARS_MATE = "e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7"


def test_pad_mates_mines_mate_in_1(tmp_path):
    from ft.data.pad_mates import mine_mate_rows
    src = tmp_path / "games.jsonl"
    write_games(str(src), [_FOOLS_MATE, "e2e4 e7e5", "d2d4 d7d5"])
    out = tmp_path / "mate_pad.jsonl"
    stats = mine_mate_rows(str(src), str(out), workers=1, force=True)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert stats["rows"] == len(rows) == 1
    row = rows[0]
    assert row["bucket"] == "mate" and row["target"] == "d8h4"
    assert row["mate_n"] == 1 and row["game_id"] == "pad0"
    assert row["history"] == ["f2f3", "e7e5", "g2g4"]
    board = chess.Board()
    for uci in row["history"]:
        board.push_uci(uci)
    board.parse_uci(row["target"])
    board.push_uci(row["target"])
    assert board.is_checkmate()


def test_pad_mates_dedups_keys_across_games(tmp_path):
    from ft.data.pad_mates import mine_mate_rows
    src = tmp_path / "games.jsonl"
    write_games(str(src), [_FOOLS_MATE, _FOOLS_MATE])
    out = tmp_path / "mate_pad.jsonl"
    mine_mate_rows(str(src), str(out), workers=1, force=True)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    again = mine_mate_rows(str(src), str(out), workers=1, force=False)
    assert again["rows"] == -1
    assert len(out.read_text().splitlines()) == 1


def test_pad_mates_stops_at_max_rows(tmp_path):
    from ft.data.pad_mates import mine_mate_rows
    src = tmp_path / "games.jsonl"
    write_games(str(src), [_FOOLS_MATE, _SCHOLARS_MATE])
    out = tmp_path / "mate_pad.jsonl"
    mine_mate_rows(str(src), str(out), max_rows=1, workers=1, force=True)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["target"] in ("d8h4", "h5f7")


def _fools_mate_row() -> dict:
    board = chess.Board()
    history = []
    for uci in _FOOLS_MATE.split():
        if uci == "d8h4":
            break
        board.push_uci(uci)
        history.append(uci)
    return {"bucket": "mate", "game_id": "padx", "ply": 3,
            "key": board.fen(), "history": history, "target": "d8h4",
            "mate_n": 1}


def test_parse_bucket_weight_args():
    from ft.data.build_buckets import parse_bucket_weight_args
    assert parse_bucket_weight_args(["mate=2.0", "endgame_tb = 1.5"]) == {
        "mate": 2.0, "endgame_tb": 1.5}
    assert parse_bucket_weight_args([]) == {}
    with pytest.raises(ValueError):
        parse_bucket_weight_args(["bogus"])
    with pytest.raises(ValueError):
        parse_bucket_weight_args(["mate=x"])
    from ft.data.build_buckets import merged_bucket_weights
    with pytest.raises(ValueError):
        merged_bucket_weights(parse_bucket_weight_args(["nope=1.0"]))
    with pytest.raises(ValueError):
        merged_bucket_weights(parse_bucket_weight_args(["mate=-1"]))
    with pytest.raises(ValueError):
        merged_bucket_weights(parse_bucket_weight_args(["mate=0"]))


def test_bucket_weight_override_scales_rows(mini_pipeline, tmp_path):
    from ft.data.build_buckets import build_bucket_dataset
    _, paths = mini_pipeline
    extra = tmp_path / "pad.jsonl"
    with open(extra, "w") as f:
        f.write(json.dumps(_fools_mate_row()) + "\n")
    counts = build_bucket_dataset(read_games(paths["games"]),
                                  paths["positions"], paths["annotations"],
                                  str(tmp_path / "mix_weighted.jsonl"),
                                  extra_rows_paths=(str(extra),),
                                  bucket_weight_overrides={"mate": 2.0,
                                                           "normal": 0.5},
                                  force=True)
    assert counts["mate"] >= 1
    weights = {json.loads(l)["bucket"]: json.loads(l)["weight"]
               for l in (tmp_path / "mix_weighted.jsonl").read_text().splitlines()
               if l.strip()}
    assert weights["mate"] == 2.0
    assert weights["normal"] == 0.5
    assert weights["contrastive"] == 1.0


def test_annotator_import_binds_chess_engine_submodule():
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, '.')\n"
            "from ft.data.annotate import StockfishAnnotator\n"
            "import chess\nassert hasattr(chess, 'engine')\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr


def test_extra_rows_add_mate_volume_to_mix(mini_pipeline, tmp_path):
    from ft.data.build_buckets import build_bucket_dataset
    _, paths = mini_pipeline
    extra = tmp_path / "pad.jsonl"
    rows = [_fools_mate_row(),
            {"bucket": "nope", "history": [], "target": "e2e4"}]
    with open(extra, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = build_bucket_dataset(read_games(paths["games"]),
                                  paths["positions"], paths["annotations"],
                                  str(tmp_path / "mix_padded.jsonl"),
                                  extra_rows_paths=(str(extra),), force=True)
    assert counts["mate"] == 1
    mate_rows = [json.loads(l) for l
                 in (tmp_path / "mix_padded.jsonl").read_text().splitlines()
                 if json.loads(l)["bucket"] == "mate"]
    chess.Board(mate_rows[0]["key"]).parse_uci(mate_rows[0]["target"])
    assert mate_rows[0]["weight"] == 1.0
