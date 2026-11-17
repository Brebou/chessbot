"""Smoke test for the RL trainer: one tiny run on a random model (CPU)."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw.model import CONFIG, ChessGPT, save_model
from rl.config import RLConfig
from rl.losses import (histories_to_ids, kl_penalty_k3, last_logprobs,
                       policy_gradient_loss, sft_ce_loss)
from rl.train import RLTrainer


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    torch.manual_seed(0)
    save_model(ChessGPT(CONFIG), path / "model.safetensors")
    return path


def test_losses_shapes_and_finiteness(tiny_model):
    import chess
    from chess_hw import ChessTokenizer
    tok = ChessTokenizer()
    torch.manual_seed(1)
    model = ChessGPT(CONFIG)
    histories = [[tok.token_to_id[m] for m in ["e2e4", "e7e5"]],
                 [tok.token_to_id[m] for m in ["d2d4"]]]
    ids = torch.from_numpy(histories_to_ids(histories, tok.pad_token_id,
                                            tok.bos_token_id))
    lengths = torch.tensor([1 + len(h) for h in histories])
    actions = torch.tensor([tok.token_to_id["g1f3"], tok.token_to_id["d7d5"]])
    new_lps, entropy = last_logprobs(model, ids, lengths, actions)
    assert new_lps.shape == (2,)
    assert torch.isfinite(new_lps).all()
    assert torch.isfinite(entropy).all()
    advs = torch.tensor([1.0, -1.0])
    old_lps = new_lps.detach()
    loss = policy_gradient_loss(advs, new_lps, old_lps, ppo_clip=0.2)
    assert torch.isfinite(loss)
    with torch.no_grad():
        ref_lps, _ = last_logprobs(model, ids, lengths, actions)
    kl = kl_penalty_k3(new_lps, ref_lps)
    assert kl.item() >= -1e-6  # k3 estimator is non-negative
    ce = sft_ce_loss(model, ids, lengths, actions)
    assert torch.isfinite(ce) and ce.item() >= 0


def test_trainer_tiny_run(tiny_model, tmp_path):
    cfg = RLConfig(
        model_path=str(tiny_model), out_dir=str(tmp_path / "run"),
        device="cpu", seed=3, max_iters=2, levels=(0,), games_per_iter=4,
        group_size=2, max_plies=10, temperature=1.0, engine_workers=0,
        lr=1e-4, ppo_clip=0.2, inner_epochs=1, minibatches=2,
        kl_coef=0.05, replay_coef=0.0, eval_every=2, eval_levels=(0,),
        eval_seeds=(0,), eval_n_openings=2, eval_workers=2, save_every=1,
    )
    trainer = RLTrainer(cfg)
    assert (tmp_path / "run" / "config.json").exists()
    summary = trainer.train()

    run = tmp_path / "run"
    assert (run / "summary.json").exists()
    assert summary["best_score"] is not None
    assert (run / "best" / "model.safetensors").exists()
    assert (run / "final" / "model.safetensors").exists()
    assert (run / "checkpoints" / "iter-000001" / "model.safetensors").exists()

    metrics = [json.loads(l) for l in
               open(run / "metrics.jsonl") if l.strip()]
    assert len(metrics) == 2
    for row in metrics:
        assert row["games"] == 4
        assert 0.0 <= row["illegal_rate"] <= 1.0
        assert row["transitions"] > 0
        for key in ("pg_loss", "kl_loss", "entropy"):
            assert row[key] == row[key]  # finite (not NaN)
    assert "eval_mean_score" in metrics[-1]


def test_config_cli_roundtrip():
    import argparse
    parser = argparse.ArgumentParser()
    RLConfig.add_cli_args(parser)
    args = parser.parse_args([
        "--model-path", "base_model", "--levels", "1", "2",
        "--games-per-iter", "16", "--lr", "2e-5", "--ppo-clip", "0",
        "--normalize-advantages", "false",
    ])
    cfg = RLConfig.from_args(args)
    assert cfg.levels == (1, 2)
    assert cfg.games_per_iter == 16
    assert cfg.lr == 2e-5
    assert cfg.ppo_clip == 0.0
    assert cfg.normalize_advantages is False
    assert cfg.group_size == 4  # default preserved
