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
