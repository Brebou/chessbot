"""Tests for the ft pipeline orchestration and script entrypoints."""

import json
import sys
from pathlib import Path

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ft.common.games_io import read_games, write_games
from ft.pipeline import PipelineConfig, run_pipeline


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
