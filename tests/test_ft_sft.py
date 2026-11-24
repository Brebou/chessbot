"""Tests for the ft SFT package: losses, dataset, trainer, paired eval, dagger."""

import argparse
import copy
import json
import math
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
from ft.sft.dagger import (blend_dagger, build_poc_mix, classify_model_move,
                           dagger_collect, rollout_positions)
from ft.sft.losses import PAIR_CAP, anchor_kl, contrastive_loss, ft_loss
from ft.sft.poc_loop import build_corrections, poc_classify, run_poc_loop
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
    assert res["cand_score"] == pytest.approx(1.0)
    assert res["base_score"] == pytest.approx(0.0)
    assert res["per_level_cand"] == {"0": pytest.approx(1.0),
                                     "1": pytest.approx(1.0)}
    assert res["per_level_base"] == {"0": pytest.approx(0.0),
                                     "1": pytest.approx(0.0)}


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


def test_rollout_records_model_move(tiny_model):
    rows = rollout_positions(str(tiny_model), n_episodes=1, stop_pieces=10,
                             max_plies=4, seed=5, device="cpu")
    assert rows and all("model_move" in r for r in rows)
    for r in rows:
        chess.Board(r["fen"]).parse_uci(r["model_move"])


def test_anchor_kl_math_and_zero_case():
    logp = torch.log(torch.tensor([[0.5, 0.5]]))
    assert anchor_kl(logp, logp).item() == pytest.approx(0.0, abs=1e-6)
    ref_logp = torch.log(torch.tensor([[0.7, 0.3]]))
    kl = anchor_kl(logp, ref_logp)
    expected = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
    assert kl.item() == pytest.approx(expected, rel=1e-4)
    assert kl.item() > 0


def test_ft_sft_config_eval_seed_default_none():
    from ft.sft.train_sft import FTSFTConfig

    cfg = FTSFTConfig(model_path="m", mix_path="x", out_dir="o")
    assert cfg.eval_seed is None          # None = fall back to cfg.seed
    cfg2 = FTSFTConfig(model_path="m", mix_path="x", out_dir="o",
                       eval_seed=1234)
    assert cfg2.eval_seed == 1234


def test_ft_loss_kl_anchor_term():
    torch.manual_seed(1)
    model = ChessGPT(CONFIG)
    ref = copy.deepcopy(model)
    ids = torch.tensor([[TOKENIZER.bos_token_id, TOK["e2e4"]]])
    lengths = torch.tensor([2])
    targets = torch.tensor([TOK["e7e5"]])
    weights = torch.tensor([1.0])
    loss0, parts0 = ft_loss(model, ids, lengths, targets, weights,
                            ref_model=ref, lambda_kl=0.5)
    assert parts0["kl"] == pytest.approx(0.0, abs=1e-5)
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.05 * torch.randn_like(p))
    loss1, parts1 = ft_loss(model, ids, lengths, targets, weights,
                            ref_model=ref, lambda_kl=0.5)
    assert parts1["kl"] > 1e-4
    assert loss1.item() > loss0.item()


def test_classify_model_move_rules():
    assert classify_model_move(0, -300, None, None, 30) == "blunder"
    assert classify_model_move(0, -100, None, None, 30) == "ok"
    assert classify_model_move(0, -9950, None, -4, 30) == "blunder"
    assert classify_model_move(None, None, 1, None, 40) == "blunder"
    assert classify_model_move(None, None, 1, 2, 8) == "blunder"
    assert classify_model_move(None, None, 5, 7, 8) == "blunder"
    assert classify_model_move(None, None, 5, 7, 20) == "ok"
    assert classify_model_move(None, None, 2, 2, 30) == "ok"


def test_build_poc_mix_keeps_only_blunder_corrections(tmp_path):
    dagger = tmp_path / "dagger.jsonl"
    ce_a = {"bucket": "dagger", "key": "FEN_A", "start_fen": "FEN_A",
            "history": ["e2e4"], "target": "e4d5", "weight": 1.0,
            "ood": False, "n_pieces": 30}
    ce_b = {"bucket": "dagger", "key": "FEN_B", "start_fen": "FEN_B",
            "history": ["d2d4"], "target": "g1f3", "weight": 1.0,
            "ood": False, "n_pieces": 8}
    ce_c = {"bucket": "dagger", "key": "FEN_C", "start_fen": "FEN_C",
            "history": ["c2c4"], "target": "d2d4", "weight": 1.0,
            "ood": False, "n_pieces": 20}
    pair_a = {**ce_a, "bucket": "contrastive", "bad": "a2a3"}
    pair_b = {**ce_b, "bucket": "contrastive", "bad": "b1c3"}
    with open(dagger, "w") as f:
        for r in (ce_a, pair_a, ce_b, pair_b, ce_c):
            f.write(json.dumps(r) + "\n")

    def score_fn(fen, model_move):
        if fen == "FEN_A":
            return {"fen": fen, "cp_best": 0, "cp_model": -300,
                    "mate_best": None, "mate_model": None}
        return {"fen": fen, "cp_best": 0, "cp_model": 0,
                "mate_best": None, "mate_model": None}

    out = tmp_path / "poc.jsonl"
    stats = build_poc_mix(str(dagger), str(out), score_fn=score_fn, force=True)
    rows = [json.loads(l) for l in open(out) if l.strip()]
    assert stats["n_rows"] == len(rows) == 1
    assert rows[0]["key"] == "FEN_A"
    assert rows[0]["bucket"] == "normal" and rows[0]["target"] == "e4d5"
    assert stats["n_within_baseline"] == 1 and stats["n_scored"] == 2


def test_poc_classify_rules():
    assert poc_classify(0, -300, None, None, 30)
    assert not poc_classify(0, -100, None, None, 30)
    assert poc_classify(None, -9950, None, -4, 30)
    assert poc_classify(None, None, 1, None, 40)
    assert poc_classify(None, None, 1, 2, 40)
    assert poc_classify(None, None, 5, 8, 8)
    assert not poc_classify(None, None, 5, 6, 8)
    assert not poc_classify(None, None, 5, 6, 40)
    assert not poc_classify(None, None, 2, 2, 30)
    assert not poc_classify(None, None, None, None, 30)


def test_build_corrections_rows(tmp_path):
    states = [
        {"fen": "FEN_A", "history": ["e2e4"], "model_move": "a2a3"},
        {"fen": "FEN_B", "history": ["d2d4"], "model_move": "b1c3"},
        {"fen": "FEN_C", "history": ["c2c4"], "model_move": "g1f3"},
    ]
    scores = {
        "FEN_A": {"fen": "FEN_A", "cp_best": 0, "cp_model": -300,
                  "mate_best": None, "mate_model": None,
                  "best_uci": "d1h5", "alt_mates": []},
        "FEN_B": {"fen": "FEN_B", "cp_best": 9000, "cp_model": 300,
                  "mate_best": 6, "mate_model": None,
                  "best_uci": "a1a8", "alt_mates": ["a1a8", "a1a7"]},
        "FEN_C": {"fen": "FEN_C", "cp_best": 0, "cp_model": 0,
                  "mate_best": None, "mate_model": None,
                  "best_uci": "g1f3", "alt_mates": []},
    }
    rows = build_corrections(states, scores, cp_threshold=150.0,
                             mate_piece_gate=10, quick_mate_plies=2,
                             mate_slack_plies=1)
    assert [r["key"] for r in rows] == ["FEN_A", "FEN_B", "FEN_B"]
    assert rows[0]["target"] == "d1h5" and rows[0]["weight"] == 1.0
    assert rows[0]["bucket"] == "normal"
    alt_rows = [r for r in rows if r["key"] == "FEN_B"]
    assert sorted(r["target"] for r in alt_rows) == ["a1a7", "a1a8"]
    assert all(r["weight"] == 0.5 for r in alt_rows)


def test_gpu_memory_line_noop_on_cpu():
    from ft.sft.poc_loop import _gpu_memory_line

    assert _gpu_memory_line("cpu") == ""


def test_base_records_cache_roundtrip(tmp_path):
    from ft.sft.poc_loop import base_records_meta, load_base_records, save_base_records

    meta = base_records_meta("base_model", (0, 1, 2, 3), 40, 0)
    records = [{"result": "1-0", "meta": {"level": 1}}]
    path = tmp_path / "base_engine_records.jsonl"
    save_base_records(path, meta, records)
    assert load_base_records(path, meta) == records
    other = base_records_meta("other_model", (0, 1, 2, 3), 40, 0)
    assert load_base_records(path, other) is None
    other = base_records_meta("base_model", (0, 1, 2, 3), 20, 0)
    assert load_base_records(path, other) is None
    path.write_text("{broken\n")
    assert load_base_records(path, meta) is None


def test_allocate_level_counts_rounding_and_validation():
    from ft.sft.poc_loop import allocate_level_counts

    with pytest.raises(ValueError):
        allocate_level_counts(10, [0.5, 0.3])
    assert allocate_level_counts(0, [0.1, 0.3, 0.6]) == [0, 0, 0]
    assert allocate_level_counts(10, [0.1, 0.3, 0.6]) == [1, 3, 6]
    counts = allocate_level_counts(7, [0.1, 0.3, 0.6])
    assert counts == [1, 2, 4]
    assert sum(counts) == 7
    counts = allocate_level_counts(8, [0.1, 0.3, 0.6])
    assert counts == [1, 2, 5] and sum(counts) == 8


def test_engine_opponent_states_mix_and_row_shape():
    from ft.sft.poc_loop import engine_opponent_states

    tasks: list = []

    def game_fn(task):
        tasks.append(task)
        oid, model_white, level, opening_plies, max_plies, seed = task
        return [{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                 "history": [], "model_move": "e2e4"}]

    states = engine_opponent_states("_unused", 10, opening_plies=8,
                                    seed="0-1", engine_levels=[1, 2, 3],
                                    level_probs=[0.1, 0.3, 0.6], workers=1,
                                    max_plies=250, game_fn=game_fn)
    assert len(tasks) == 10
    assert len(states) == 10
    for state in states:
        assert set(state) == {"fen", "history", "model_move"}
    assert [bool(t[1]) for t in tasks] == [g % 2 == 0 for g in range(10)]
    levels = [t[2] for t in tasks]
    for level, expected in zip((1, 2, 3), (1, 3, 6)):
        assert levels.count(level) == expected
    assert engine_opponent_states("_unused", 0, opening_plies=8, seed="0-1",
                                  engine_levels=[1], level_probs=[1.0],
                                  workers=1, max_plies=250,
                                  game_fn=game_fn) == []


def test_use_fixtures_gating():
    from types import SimpleNamespace

    from ft.sft.poc_loop import _use_fixtures

    on = SimpleNamespace(fixtures_dir="data/ft_fixtures", fixtures_every=3)
    assert _use_fixtures(on, 1, 0)
    assert _use_fixtures(on, 3, 5)
    assert not _use_fixtures(on, 4, 5)
    assert _use_fixtures(on, 6, 5)
    every = SimpleNamespace(fixtures_dir="d")
    assert _use_fixtures(every, 7, 12)
    assert not _use_fixtures(SimpleNamespace(fixtures_dir=""), 1, 0)
    missing = SimpleNamespace(fixtures_dir="data/ft_fixtures", fixtures_every=5)
    assert _use_fixtures(missing, 5, 4)
    assert not _use_fixtures(missing, 4, 4)


def test_softmax_weights_stable_and_ordered():
    from ft.sft.poc_loop import _softmax_weights

    w = _softmax_weights([0.0, -50.0, -150.0], temp=100.0)
    assert abs(sum(w) - 1.0) < 1e-9
    assert w[0] > w[1] > w[2] > 0
    assert abs(w[1] - math.exp(-0.5) / (1 + math.exp(-0.5) + math.exp(-1.5))) < 1e-9
    one = _softmax_weights([0.0], temp=100.0)
    assert one == [1.0]
    mate_scale = _softmax_weights([10000.0, 9850.0, -10000.0], temp=100.0)
    assert abs(sum(mate_scale) - 1.0) < 1e-9
    assert 0 < mate_scale[1] < 0.5
    assert mate_scale[2] < 1e-10


def test_dense_lambda_normalization():
    from ft.sft.poc_loop import _dense_lambda

    assert _dense_lambda(0, 0, 8) == 0.0
    assert _dense_lambda(16, 8, 8) == 0.25
    assert _dense_lambda(7, 700, 8) == pytest.approx(7 / 700 / 8)


def test_build_dense_rows_band_weights_and_played_move():
    from ft.sft.poc_loop import build_dense_rows

    def state(fen, move):
        return {"fen": fen, "history": [], "model_move": move}

    def scores(cp_best, cp_model, table, mate_best=None, mate_model=None):
        return {"cp_best": cp_best, "cp_model": cp_model,
                "mate_best": mate_best, "mate_model": mate_model,
                "best_uci": table[0][0] if table else None,
                "multipv": [{"uci": u, "cp": cp} for u, cp in table]}

    states = [
        state("FEN_X", "b1b2"),
        state("FEN_BLUNDER", "b1b2"),
        state("FEN_TINY", "b1b2"),
        state("FEN_MATE", "b1b2"),
        state("FEN_OUTSIDE", "e2e4"),
    ]
    sc = {
        "FEN_X": scores(0, -100, [("b1b2", -50), ("c1c2", -60), ("d1d2", -150)]),
        "FEN_BLUNDER": scores(0, -300, [("b1b2", 0), ("c1c2", -50)]),
        "FEN_TINY": scores(0, -20, [("b1b2", -5), ("c1c2", -10)]),
        "FEN_MATE": scores(0, -100, [("b1b2", 0), ("c1c2", -50)],
                           mate_best=2),
        "FEN_OUTSIDE": scores(0, -100, [("c1c2", -50), ("d1d2", -60)]),
    }
    rows, n = build_dense_rows(states, sc, cp_threshold=150.0,
                               dense_cp=35.0, dense_temp=100.0)
    assert n == 2
    keys = {r["key"] for r in rows}
    assert keys == {"FEN_X", "FEN_OUTSIDE"}
    assert all(r["bucket"] == "dense" for r in rows)
    by_key: dict = {}
    for r in rows:
        by_key.setdefault(r["key"], []).append(r)
    x = by_key["FEN_X"]
    assert [r["target"] for r in x] == ["b1b2", "c1c2", "d1d2"]
    assert abs(sum(r["weight"] for r in x) - 1.0) < 1e-9
    assert x[0]["weight"] > x[1]["weight"] > x[2]["weight"]
    outside = by_key["FEN_OUTSIDE"]
    assert [r["target"] for r in outside] == ["c1c2", "d1d2", "e2e4"]
    assert abs(sum(r["weight"] for r in outside) - 1.0) < 1e-9
    assert outside[2]["weight"] < outside[0]["weight"]
    empty, n_empty = build_dense_rows([], {}, cp_threshold=150.0,
                                      dense_cp=35.0, dense_temp=100.0)
    assert empty == [] and n_empty == 0


def test_poc_loop_single_iteration_smoke(tiny_model, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        model_path=str(tiny_model / "model.safetensors"),
        out_dir=str(tmp_path / "poc"), iterations=1, games=1, epochs=1,
        batch_size=8, lr=5e-5, kl_lambda=0.1, seed=0, device="cpu",
        engine="stockfish", depth=6, workers=1, cp_threshold=150.0,
        mate_piece_gate=10, quick_mate_plies=2, mate_slack_plies=1,
        eval_workers=1, eval_mode="none", eval_levels=(0,), patience=3,
        fixtures_dir="", fixtures_games="")
    seen: dict = {}

    def score_fn(fen, model_move):
        seen[fen] = True
        if len(seen) == 1:
            return {"fen": fen, "cp_best": 0, "cp_model": -300,
                    "mate_best": None, "mate_model": None,
                    "best_uci": "e2e4", "alt_mates": []}
        return {"fen": fen, "cp_best": 0, "cp_model": 0,
                "mate_best": None, "mate_model": None,
                "best_uci": model_move, "alt_mates": []}

    summary = run_poc_loop(args, score_fn=score_fn)
    assert summary[0]["corrections"] >= 1
    final = summary[0].get("final_model")
    assert final and Path(final).exists()

    poc_dir = tmp_path / "poc"
    metrics_lines = [json.loads(l) for l in
                     (poc_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert metrics_lines and all("iteration" in row for row in metrics_lines)
    summary_lines = [json.loads(l) for l in
                     (poc_dir / "summary.jsonl").read_text().splitlines() if l.strip()]
    assert len(summary_lines) == 1
    assert summary_lines[0]["iteration"] == 1
    assert summary_lines[0]["corrections"] == summary[0]["corrections"]


def test_rollout_and_collect_accept_progress_show_object(tiny_model, tmp_path):
    class StubProgress:
        def __init__(self):
            self.messages = []

        def show(self, message, force=False):
            self.messages.append(message)

    prog = StubProgress()
    rows = rollout_positions(str(tiny_model), n_episodes=1, stop_pieces=10,
                             max_plies=2, seed=5, device="cpu", progress=prog)
    assert prog.messages
    prog2 = StubProgress()
    out = tmp_path / "dagger_prog.jsonl"
    dagger_collect(str(tiny_model), str(out), n_episodes=1, stop_pieces=30,
                   max_plies=2, seed=0, device="cpu",
                   annotate_fn=_stub_annotate, emit_pairs=False,
                   force=True, progress=prog2)
    assert prog2.messages


def test_dagger_emit_pairs_makes_contrastive_rows(tiny_model, tmp_path):
    out = tmp_path / "dagger.jsonl"
    stats = dagger_collect(str(tiny_model), str(out), n_episodes=2,
                           stop_pieces=30, max_plies=4, seed=0, device="cpu",
                           annotate_fn=_stub_annotate, emit_pairs=True,
                           force=True)
    rows = list(json.loads(l) for l in open(out) if l.strip())
    pairs = [r for r in rows if r["bucket"] == "contrastive"]
    assert stats["n_rows"] == len(rows) > len(pairs) >= 0
    for r in pairs:
        assert r["bad"] != r["target"]
        chess.Board(r["key"]).parse_uci(r["bad"])


def test_arc2_mirror_symmetry():
    from ft.research.arc2 import _mirror_fen, _mirror_square, mirror_rows

    # start position is its own mirror
    assert _mirror_fen(chess.STARTING_FEN).split()[0] == \
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    assert _mirror_square('e2e4') == 'e7e5'
    assert _mirror_square('a1h1') == 'a8h8'
    assert _mirror_square('e7e8q') == 'e2e1q'
    # color symmetry: a real position + legal move mirrors to a mirrored
    # position where the mirrored move is legal
    fen = "r1bqk2r/ppppbppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 6 5"
    board = chess.Board(fen)
    legals = list(board.legal_moves)
    target = next(str(m) for m in legals if board.piece_type_at(m.from_square)
                  == chess.KNIGHT)
    rows = [{"bucket": "normal", "key": fen, "start_fen": fen,
             "history": ["e2e4", "e7e5", "g1f3", "b8c6"], "target": target,
             "weight": 1.0}]
    out = mirror_rows(rows)
    assert len(out) == 2
    mf = chess.Board(out[1]["start_fen"])
    mv = chess.Move.from_uci(out[1]["target"])
    assert mv in mf.legal_moves
    # side to move flipped to the other color
    assert out[0]["start_fen"].split()[1] != out[1]["start_fen"].split()[1]
    assert out[1]["weight"] == 1.0


def test_arc2_level_gain():
    from ft.research.arc2 import apply_level_gain
    corpus = [
        {"fen": "A", "corrected": True, "level": 1},
        {"fen": "B", "corrected": True, "level": 3},
        {"fen": "C", "corrected": False, "level": 1},
        {"fen": "D", "corrected": True, "level": 2},
    ]
    apply_level_gain(corpus, {"1": 3.0})
    # corrected states on level 1 get the gain; everything else untouched
    assert corpus[0].get("weight") == 3.0
    assert corpus[1].get("weight", 1.0) == 1.0
    assert corpus[3].get("weight", 1.0) == 1.0
    assert "weight" not in corpus[2]

    apply_level_gain(corpus, {"*": 2.0})
    assert corpus[0]["weight"] == 6.0  # multiplies existing weight
    assert corpus[1]["weight"] == 2.0
    assert corpus[3]["weight"] == 2.0


def test_build_reinforce_mix_outcomes(monkeypatch, tmp_path):
    from ft.research import arc2
    from ft.research.arc2 import build_reinforce_mix
    games = [
        {"out": 1.0, "level": 1, "oid": 1,
         "states": [{"fen": "WIN", "history": [], "model_move": "e2e4"}]},
        {"out": 0.0, "level": 1, "oid": 2,
         "states": [{"fen": "BLUN", "history": [], "model_move": "e2e3"}]},
        {"out": 0.5, "level": 2, "oid": 3,
         "states": [{"fen": "DRAW", "history": [], "model_move": "d2d4"}]},
    ]

    def fake_score(tasks, **kw):
        return {"BLUN": {"cp_best": 500.0, "cp_model": 0.0, "mate_best": None,
                         "mate_model": None, "best_uci": "a1a2"}}

    monkeypatch.setattr("ft.sft.poc_loop.poc_score", fake_score)
    out = tmp_path / "mix.jsonl"
    rows, _ = build_reinforce_mix(games, out, engine_path="sf", sf_depth=6,
                                  sf_workers=1, cap=0)
    by_fen = {}
    from ft.common.serialize import jsonl_read
    for r in jsonl_read(out):
        by_fen.setdefault(r["key"], []).append(r)
    win = by_fen["WIN"][0]
    draw = by_fen["DRAW"][0]
    blun = [r for r in by_fen["BLUN"]]
    assert win["target"] == "e2e4" and win["weight"] == 1.0
    assert draw["target"] == "d2d4" and draw["weight"] == 0.3
    assert len(blun) == 1 and blun[0]["target"] == "a1a2" and blun[0]["weight"] == 1.0
    assert rows == 3


def test_build_trajectory_rows():
    from ft.research.arc2 import build_trajectory_rows
    games = [{"out": 0.0, "level": 1, "oid": 9, "states": [
        {"fen": "B0", "history": [], "model_move": "e2e4"},
        {"fen": "B1", "history": [], "model_move": "d2d4"},
        {"fen": "B2", "history": [], "model_move": "e2e5"},
    ]}]
    scores = {
        "B0": {"cp_best": 100.0, "cp_model": 80.0, "best_uci": "d2d4"},
        "B1": {"cp_best": 100.0, "cp_model": 90.0, "best_uci": "e2e4"},
        "B2": {"cp_best": 100.0, "cp_model": -400.0, "best_uci": "e2e4"},
    }
    rows = []
    n = build_trajectory_rows(games, rows, scores, w_loss=1.0)
    # tip is B2 (drop 500); tail = B2..B0 (3 plies): all differ from played
    assert n == 3
    by = {r["key"]: r for r in rows}
    assert by["B2"]["target"] == "e2e4" and by["B2"]["weight"] == 0.25
    assert by["B0"]["target"] == "d2d4" and by["B0"]["weight"] == 0.25
    # dedupe: existing fens are skipped
    rows2 = []
    n2 = build_trajectory_rows(games, rows2, scores, w_loss=1.0,
                               existing_fens={"B1", "B2"})
    assert n2 == 1 and rows2[0]["key"] == "B0"
    # small drop (< drop_cp) -> no rows
    scores_low = {k: {"cp_best": 100.0, "cp_model": 80.0, "best_uci": "d2d4"}
                  for k in ("B0", "B1", "B2")}
    rows3 = []
    assert build_trajectory_rows(games, rows3, scores_low, w_loss=1.0) == 0


def test_reinforce_configs_registered():
    from ft.research.arc2 import REINFORCE_CONFIGS
    for name in ("d1", "d2", "f1", "f2", "f3", "g1", "g2"):
        assert name in REINFORCE_CONFIGS
    assert REINFORCE_CONFIGS["g1"].get("trajectory") is True
    assert REINFORCE_CONFIGS["f3"]["w_win"] == 0.0
