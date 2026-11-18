"""Tests for the ft eval package: fixtures, paired eval, summarizer."""

import json

import chess
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_metrics(tmp_path, rows):
    path = tmp_path / "metrics.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_summarize_reports_best_and_latest(tmp_path, capsys):
    from ft.eval.summarize import summarize
    path = _write_metrics(tmp_path, [
        {"eval_tag": "step-288", "step": 288, "balanced": -0.05,
         "engine_delta": -0.04, "proxy_delta": -0.06,
         "fixtures": {"mate1_top1_rate": 0.09,
                      "winning_capture_rate_late": 0.58,
                      "losing_capture_rate_late": 0.25,
                      "top1_legal_end": 0.95}},
        {"eval_tag": "step-576", "step": 576, "balanced": 0.05,
         "engine_delta": 0.06, "proxy_delta": 0.04, "significant": True,
         "fixtures": {"mate1_top1_rate": 0.12}},
    ])
    stats = summarize(path)
    out = capsys.readouterr().out
    assert stats["n_evals"] == 2 and stats["qualified"] == 1
    assert "best qualified: step-576" in out
    assert "steps 288-576" in out
    assert "step-576" in out and "*" in out


def test_summarize_handles_nan_and_partial_lines(tmp_path, capsys):
    from ft.eval.summarize import summarize
    path = tmp_path / "metrics.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"eval_tag": "epoch-1", "step": 576,
                            "balanced": -0.04, "val_ce_endgame_tb": float("nan"),
                            "fixtures": {"mate1_top1_rate": 0.084}}) + "\n")
        f.write('{"eval_tag": "trunc')          # partially written line
    stats = summarize(str(path))
    out = capsys.readouterr().out
    assert stats == {"n_evals": 1, "qualified": 0, "latest": stats["latest"]}
    assert "-" in out                            # NaN rendered as "-"
    assert "best qualified: none" in out
