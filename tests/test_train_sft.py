"""Tests for the Phase 1 SFT trainer (train_sft.py)."""

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw.model import CONFIG, ChessGPT, save_model
from train_sft import (SFTConfig, SFTDataset, SFTTrainer, encode_batch, sft_loss)


MIX_ROWS = [
    {"kind": "replay", "history": ["e2e4"], "target": "e7e5", "weight": 1.0},
    {"kind": "replay", "history": ["d2d4"], "target": "d7d5", "weight": 1.0},
    {"kind": "mate1", "history": ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"],
     "target": "h5f7", "weight": 1.0},
    {"kind": "capture", "history": ["g1f3", "d7d6", "f3e5"],
     "target": "d6e5", "weight": 2.0},
    {"kind": "hang", "history": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d3"],
     "target": "f1c4", "bad": "f1b5", "weight": 1.0},
    {"kind": "endgame", "history": ["e2e4", "e7e5"], "target": "g1f3", "weight": 2.0},
] * 10


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("sft_model")
    torch.manual_seed(0)
    save_model(ChessGPT(CONFIG), path / "model.safetensors")
    return path


def _make_cfg(tiny_model, tmp_path, **kw):
    defaults = dict(
        model_path=str(tiny_model), mix_path=str(tmp_path / "mix.jsonl"),
        out_dir=str(tmp_path / "run"), device="cpu", seed=0, epochs=1,
        batch_size=8, lr=1e-4, val_fraction=0.1, eval_every_steps=0,
        eval_levels=(0,), eval_seeds=(0,), eval_n_openings=2, eval_workers=2,
    )
    defaults.update(kw)
    return SFTConfig(**defaults)


def test_sft_loss_margin_and_ce():
    from chess_hw import ChessTokenizer
    tok = ChessTokenizer()
    torch.manual_seed(1)
    model = ChessGPT(CONFIG)
    items = [{"history": [tok.token_to_id["e2e4"]], "target": tok.token_to_id["e7e5"],
              "bad": tok.token_to_id["d7d5"], "weight": 2.0}]
    arr, lengths, targets, weights, bads, bad_mask = encode_batch(
        items, [0], tok, "cpu")
    assert bads is not None and bad_mask.all()
    loss, parts = sft_loss(model, arr, lengths, targets, weights, bads,
                           bad_mask, margin_beta=0.1, margin_weight=1.0)
    assert torch.isfinite(loss)
    assert "margin" in parts and parts["margin"] >= 0.0
    # without bad ids: pure weighted CE
    loss2, parts2 = sft_loss(model, arr, lengths, targets, weights, None,
                             bad_mask, 0.1, 1.0)
    assert torch.isfinite(loss2) and "margin" not in parts2
    # weighted CE: doubling the weight must double the raw CE contribution
    lp = torch.log_softmax(torch.nn.functional.linear(
        model.hidden_states(arr)[0, lengths[0] - 1], model.wte.weight).float(), -1)
    raw_ce = -lp[targets[0]].item()
    assert abs(parts2["ce"] - raw_ce) < 1e-5


def test_dataset_batching_length_sorted():
    from chess_hw import ChessTokenizer
    tok = ChessTokenizer()
    cfg = SFTConfig()
    ds = SFTDataset(MIX_ROWS, tok, cfg)
    assert len(ds) == len(MIX_ROWS)
    batches = ds.batches(8, seed=0, epoch=1)
    assert sum(len(b) for b in batches) == len(ds)
    lens = [len(ds.items[i]["history"]) for b in batches for i in b]
    assert lens == sorted(lens)


def test_trainer_tiny_run(tiny_model, tmp_path):
    mix = tmp_path / "mix.jsonl"
    with open(mix, "w") as f:
        for r in MIX_ROWS:
            f.write(json.dumps(r) + "\n")
    cfg = _make_cfg(tiny_model, tmp_path)
    trainer = SFTTrainer(cfg)
    assert (tmp_path / "run" / "config.json").exists()
    summary = trainer.train()
    run = tmp_path / "run"
    assert (run / "summary.json").exists()
    assert (run / "final" / "model.safetensors").exists()
    assert (run / "best" / "model.safetensors").exists()
    metrics = [json.loads(l) for l in open(run / "metrics.jsonl") if l.strip()]
    assert metrics and "eval_mean_score" in metrics[-1]
    assert summary["best_score"] is not None


def test_config_cli_roundtrip():
    parser = argparse.ArgumentParser()
    SFTConfig.add_cli_args(parser)
    args = parser.parse_args(["--mix", "data/sft_mix.jsonl", "--epochs", "3",
                              "--w-mate1", "5.0", "--eval-seeds", "0", "1", "2"])
    cfg = SFTConfig.from_args(args)
    assert cfg.epochs == 3
    assert cfg.w_mate1 == 5.0
    assert cfg.eval_seeds == (0, 1, 2)
    assert cfg.w_capture == 2.0  # default preserved
