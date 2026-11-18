"""Tests for the ft SFT package: losses, dataset, trainer, paired eval, dagger."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessTokenizer
from chess_hw.model import CONFIG, ChessGPT, save_model

from ft.eval.paired_eval import paired_engine_eval, paired_stop_criterion
from ft.sft.dataset import (BUCKET_RATIOS_TRAIN, game_partition_key,
                            load_items, stratified_batches)
from ft.sft.dagger import blend_dagger, dagger_collect, rollout_positions
from ft.sft.losses import PAIR_CAP, contrastive_loss, ft_loss
from ft.sft.train_sft import (FTSFTConfig, FTSFTTrainer, ce_regressed,
                              legality_aborted)

TOKENIZER = ChessTokenizer()
TOK = TOKENIZER.token_to_id


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("ft_model")
    torch.manual_seed(0)
    save_model(ChessGPT(CONFIG), path / "model.safetensors")
    return path


# -- losses -------------------------------------------------------------------

def test_contrastive_softplus_saturation_and_cap():
    good, bad = torch.tensor([3]), torch.tensor([7])
    logp = torch.full((1, 100), -5.0)
    assert abs(contrastive_loss(logp, good, bad).item() - 0.693) < 1e-3
    logp[0, 3], logp[0, 7] = -1.0, -9.0          # d = 8 nats -> saturated
    assert contrastive_loss(logp, good, bad).item() < 1e-3
    logp[0, 3], logp[0, 7] = -20.0, -1.0         # d = -19 -> clamped at cap
    assert abs(contrastive_loss(logp, good, bad).item() - PAIR_CAP) < 1e-5


def test_ft_loss_lambda_scales_contrastive(tiny_model):
    torch.manual_seed(1)
    model = ChessGPT(CONFIG)
    ids = torch.tensor([[TOKENIZER.bos_token_id, TOK["e2e4"]]])
    lengths = torch.tensor([2])
    targets = torch.tensor([TOK["e7e5"]])
    weights = torch.tensor([1.0])
    good = torch.tensor([TOK["e7e5"]])
    bad = torch.tensor([TOK["d7d5"]])
    bad_mask = torch.tensor([True])
    base, _ = ft_loss(model, ids, lengths, targets, weights,
                      good, bad, bad_mask, lambda_contrast=0.0)
    full, parts_full = ft_loss(model, ids, lengths, targets, weights,
                               good, bad, bad_mask, lambda_contrast=1.0)
    assert "contrastive" in parts_full
    assert full.item() == pytest.approx(base.item() + parts_full["contrastive"], rel=1e-5)
    no_bad, _ = ft_loss(model, ids, lengths, targets, weights,
                        good, bad, torch.tensor([False]), lambda_contrast=1.0)
    assert no_bad.item() == pytest.approx(base.item(), rel=1e-5)


# -- dataset ------------------------------------------------------------------

def test_game_level_partition_key():
    assert game_partition_key("g42", "fen") == game_partition_key("g42", "other")
    vals = sum(game_partition_key(f"game{i}", "fen") == "val"
               for i in range(10_000))
    assert 100 < vals < 700          # ~3% expected


def _mix_rows():
    buckets = list(BUCKET_RATIOS_TRAIN)
    rows = []
    for i in range(120):
        rows.append({"bucket": buckets[i % len(buckets)],
                     "game_id": f"g{i}", "history": ["e2e4", "e7e5"],
                     "target": "g1f3", "bad": None, "weight": 1.0})
    return rows


def test_load_items_and_stratified_batches(tmp_path):
    mix = tmp_path / "mix.jsonl"
    with open(mix, "w") as f:
        for r in _mix_rows():
            f.write(json.dumps(r) + "\n")
    train, val = load_items(str(mix), TOKENIZER)
    assert len(train) + len(val) == 120
    batch_size = 20
    batches = list(stratified_batches(train, batch_size, seed=0, epoch=0))
    assert batches
    first = batches[0]
    per_bucket = {}
    for it in first:
        per_bucket[it.bucket] = per_bucket.get(it.bucket, 0) + 1
    assert per_bucket["normal"] == int(0.45 * batch_size)
    assert per_bucket["mate"] == max(1, int(round(0.20 * batch_size)))
    assert sum(per_bucket.values()) == batch_size


# -- stop criteria ------------------------------------------------------------

def test_stop_criteria_units():
    base = {"top1_legal_end": 0.60}
    assert legality_aborted(base, {"top1_legal_end": 0.52}) is not None
    assert legality_aborted(base, {"top1_legal_end": 0.58}) is None
    assert legality_aborted(None, {"top1_legal_end": 0.0}) is None
    assert ce_regressed(1.0, 1.06)
    assert not ce_regressed(1.0, 1.03) and not ce_regressed(None, 2.0)
    assert paired_stop_criterion([0.05, 0.01, 0.01], 0.02, 2)
    assert not paired_stop_criterion([0.05, 0.03, 0.01], 0.02, 2)
    assert not paired_stop_criterion([0.01], 0.02, 2)


# -- trainer ------------------------------------------------------------------

def test_ft_trainer_tiny_run(tiny_model, tmp_path):
    mix = tmp_path / "mix.jsonl"
    with open(mix, "w") as f:
        for r in _mix_rows():
            f.write(json.dumps(r) + "\n")
    cfg = FTSFTConfig(model_path=str(tiny_model), mix_path=str(mix),
                      out_dir=str(tmp_path / "run"), device="cpu", seed=0,
                      epochs=1, batch_size=8, lr=1e-4, log_every=5,
                      eval_mode="none", eval_every_steps=0, fixtures_dir="")
    trainer = FTSFTTrainer(cfg)
    summary = trainer.train()
    run = tmp_path / "run"
    assert (run / "config.json").exists()
    assert (run / "summary.json").exists()
    assert (run / "final" / "model.safetensors").exists()
    assert (run / "checkpoints" / "epoch-1" / "model.safetensors").exists()
    metrics = [json.loads(l) for l in open(run / "metrics.jsonl") if l.strip()]
    assert metrics and metrics[-1]["eval_tag"] == "epoch-1"
    assert "val_ce_normal" in metrics[-1]
    assert summary["steps"] > 0 and summary["best_score"] is None


def test_ft_config_cli_roundtrip():
    parser = argparse.ArgumentParser()
    FTSFTConfig.add_cli_args(parser)
    args = parser.parse_args(["--mix", "data/ft/mix.jsonl", "--epochs", "2",
                              "--lambda-contrast", "0.25",
                              "--eval-levels", "0", "1"])
    cfg = FTSFTConfig.from_args(args)
    assert cfg.epochs == 2
    assert cfg.lambda_contrast == 0.25
    assert cfg.batch_size == 512
    assert cfg.eval_levels == (0, 1)


# -- paired eval ---------------------------------------------------------------

def test_paired_eval_with_fake_records(monkeypatch):
    import ft.eval.paired_eval as pe

    def fake_records(model_path, tasks, workers, progress=False):
        out = []
        for (_, lvl, seed, oid, model_color) in tasks:
            if "cand" in str(model_path):
                result = "1-0" if model_color == "white" else "0-1"
            else:
                result = "0-1" if model_color == "white" else "1-0"
            out.append({"result": result, "meta": {}})
        return out

    monkeypatch.setattr(pe, "_engine_records", fake_records)
    res = paired_engine_eval("cand", "base", levels=(0, 1), n_openings=2,
                             seed=0, workers=1, n_bootstrap=200)
    assert res["delta"] == pytest.approx(1.0)
    assert res["sigma"] == pytest.approx(0.0, abs=1e-9)
    assert res["significant"] is True
    assert res["n_cells"] == 8


# -- dagger --------------------------------------------------------------------

def _stub_annotate(fen: str) -> dict:
    board = chess.Board(fen)
    return {"best_uci": next(iter(board.legal_moves)).uci(),
            "cp_best": 10, "cp_second": 5, "mate_root": None}


def test_dagger_collect_and_rollout(tiny_model, tmp_path):
    out = tmp_path / "dagger.jsonl"
    stats = dagger_collect(str(tiny_model), str(out), n_episodes=2,
                           stop_pieces=30, max_plies=4, seed=0, device="cpu",
                           annotate_fn=_stub_annotate)
    assert stats["n_rows"] == stats["n_positions"] > 0
    rows = list(json.loads(l) for l in open(out) if l.strip())
    assert all(r["bucket"] == "dagger" for r in rows)
    assert all(r["ood"] for r in rows)          # no elite corpus -> all OOD
    fens = [r["key"] for r in rows]
    assert len(fens) == len(set(fens))
    for r in rows:
        chess.Board(r["key"]).parse_uci(r["target"])
        replay = chess.Board()
        for uci in r["history"]:               # history replays to the FEN
            replay.push_uci(uci)
        assert replay.fen() == r["key"]


def test_dagger_cached_output(tiny_model, tmp_path):
    out = tmp_path / "dagger.jsonl"
    dagger_collect(str(tiny_model), str(out), n_episodes=1, stop_pieces=30,
                   max_plies=2, seed=1, device="cpu", annotate_fn=_stub_annotate)
    again = dagger_collect(str(tiny_model), str(out), n_episodes=1,
                           stop_pieces=30, max_plies=2, seed=1, device="cpu",
                           annotate_fn=_stub_annotate)
    assert again["n_rows"] == 2 and again["n_ood"] == 2


def test_is_ood_index_matches_neighborhood():
    from ft.sft.dagger import is_ood
    assert is_ood((30, 0), set()) is True
    elite = {(30, 0), (32, -2)}
    assert is_ood((31, 1), elite) is False
    assert is_ood((29, -1), elite) is False
    assert is_ood((30, 2), elite) is True
    assert is_ood((34, -2), elite) is True


def test_build_iteration2_mix_folds_and_relabels(tmp_path):
    from ft.sft.dagger import build_iteration2_mix
    base = tmp_path / "mix.jsonl"
    with open(base, "w") as f:
        for i in range(100):
            f.write(json.dumps({"bucket": "normal", "id": i}) + "\n")
    dagger = tmp_path / "dagger.jsonl"
    with open(dagger, "w") as f:
        for i in range(40):
            f.write(json.dumps({"bucket": "dagger", "key": f"fen{i}",
                                "ood": i < 10}) + "\n")
    out = tmp_path / "mix_it2.jsonl"
    build_iteration2_mix(str(base), str(dagger), str(out), max_share=0.15)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 117
    folded = [r for r in rows if r.get("folded") or "key" in r]
    assert len(folded) == 17
    assert all(r["bucket"] == "normal" for r in folded)
    assert all(r["ood"] for r in folded[:10])


def test_blend_dagger_share_cap():
    rows = [{"bucket": "normal", "id": i} for i in range(100)]
    dagger = [{"bucket": "dagger", "ood": False} for _ in range(40)]
    dagger[0]["ood"] = True
    blended = blend_dagger(rows, dagger, max_share=0.15, seed=0)
    n_dagger = sum(1 for r in blended if r["bucket"] == "dagger")
    assert n_dagger == 17                     # floor(0.15/0.85 * 100)
    assert blended[100]["ood"] is True        # OOD rows kept first


def test_rollout_positions_model_moves(tiny_model):
    rows = rollout_positions(str(tiny_model), n_episodes=1, stop_pieces=30,
                             max_plies=3, seed=3, device="cpu")
    assert rows
    assert all("history" in r and "fen" in r for r in rows)

# -- fixtures -------------------------------------------------------------------

def _sheds_material_game(rng) -> str:
    board = chess.Board()
    moves = []
    while len(board.piece_map()) > 10:
        legal = list(board.legal_moves)
        caps = [m for m in legal if board.is_capture(m)]
        if not caps:
            caps = legal
        if not caps or board.outcome(claim_draw=True) is not None:
            break
        mv = caps[rng.randrange(len(caps))]
        moves.append(mv.uci())
        board.push(mv)
    return " ".join(moves)


def test_build_position_fixtures_from_games_fill_end_bin():
    from ft.data.sample_positions import bin_for_pieces
    from ft.eval.metrics import build_position_fixtures
    rng = random.Random(0)
    games = [_sheds_material_game(rng) for _ in range(60)]
    bins = build_position_fixtures(20, seed=0, games=games)
    assert len(bins["end"]) == 20
    assert len(bins["mid"]) == 20
    for row in bins["end"]:
        assert bin_for_pieces(len(chess.Board(row["fen"]).piece_map())) == "end"
    board = chess.Board()
    for uci in bins["end"][0]["history"]:
        board.push_uci(uci)
    assert board.fen() == bins["end"][0]["fen"]


def test_build_fixtures_caches_deterministic_and_metrics_run(tiny_model, tmp_path):
    from ft.eval.metrics import build_fixtures, fixture_metrics
    d = tmp_path / "fx"
    games = tmp_path / "games.jsonl"
    games.write_text(json.dumps({"moves": "e2e4 e7e5 g1f3 g8f6"}) + "\n")
    build_fixtures(str(d), n_mate1=6, n_per_bin=4, seed=1,
                   games_path=str(games))
    first = {p.name: p.read_text() for p in sorted(d.glob("*.jsonl"))}
    assert len(first) == 4
    mtimes = {p.name: p.stat().st_mtime_ns for p in sorted(d.glob("*.jsonl"))}
    build_fixtures(str(d), n_mate1=6, n_per_bin=4, seed=1, games_path="")
    assert {p.name: p.read_text() for p in sorted(d.glob("*.jsonl"))} == first
    assert {p.name: p.stat().st_mtime_ns for p in sorted(d.glob("*.jsonl"))} == mtimes
    metrics = fixture_metrics(str(tiny_model / "model.safetensors"), str(d),
                              device="cpu")
    assert 0.0 <= metrics["mate1_top1_rate"] <= 1.0
    assert 0.0 <= metrics["top1_legal_end"] <= 1.0
    assert "top1_legal_global" in metrics

def test_fast_engine_move_identical_levels_1_to_3():
    from chess_hw.engine import Engine
    from diag.fast_engine import FastEngine

    def playout(rng, plies):
        board = chess.Board()
        moves = []
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal or board.outcome(claim_draw=True) is not None:
                break
            mv = legal[rng.randrange(len(legal))]
            moves.append(mv.uci())
            board.push(mv)
        return moves

    rng = random.Random(7)
    boards = []
    for plies in (16, 60):
        moves = playout(rng, plies)
        board = chess.Board()
        for i, uci in enumerate(moves):
            if i % 4 == 0 and len(boards) < 14:
                boards.append(board.copy())
            board.push_uci(uci)
    seed = "engine-0-0-3-white"
    for level in (1, 2):
        for board in boards:
            a = Engine(level=level, seed=seed).best_move(board)
            b = FastEngine(level=level, seed=seed).best_move(board)
            assert (a.uci() if a else None) == (b.uci() if b else None)
    for board in boards[-2:]:
        a = Engine(level=3, seed=seed).best_move(board)
        b = FastEngine(level=3, seed=seed).best_move(board)
        assert (a.uci() if a else None) == (b.uci() if b else None)

class _BucketOnly:
    def __init__(self, bucket: str):
        self.bucket = bucket


def test_stratified_batches_epoch_is_a_full_pass_over_the_scarcest_bucket():
    supplies = {"mate": 20, "normal": 46, "contrastive": 15,
                "endgame_tb": 15, "anchor": 5}
    items = [b for b, n in supplies.items() for _ in range(n)]
    items = [_BucketOnly(b) for b in items]
    batches = list(stratified_batches(items, batch_size=10, seed=0, epoch=0))
    assert len(batches) == 10
    assert all(len(b) == 11 for b in batches)  # round(45/20/15/15/5 * 10) = 4/2/2/2/1
    for batch in batches:
        counts = Counter(it.bucket for it in batch)
        assert counts["mate"] == 2
        assert counts["normal"] == 4
