"""Tests for progress display and corpus cache-skip helpers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diag.common import ProgressLine, count_jsonl_rows
from diag.gen_corpus import reuse_if_complete


def test_count_jsonl_rows_missing_file(tmp_path):
    assert count_jsonl_rows(tmp_path / "missing.jsonl") == 0


def test_count_jsonl_rows_counts_non_empty_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = "\n".join(json.dumps({"i": i}) for i in range(5)) + "\n\n"
    path.write_text(rows)
    assert count_jsonl_rows(path) == 5


def test_reuse_if_complete(tmp_path):
    path = tmp_path / "probe_x.jsonl"
    assert not reuse_if_complete("x", path, 3)
    path.write_text("\n".join('{"i": %d}' % i for i in range(3)) + "\n")
    assert reuse_if_complete("x", path, 3)
    assert reuse_if_complete("x", path, 1)
    assert not reuse_if_complete("x", path, 4)


def test_progress_line_throttles_and_closes(capsys):
    prog = ProgressLine(min_interval=60.0)
    prog.show("first")
    prog.show("second")
    prog.close()
    out = capsys.readouterr().out
    assert "first" in out
    assert "second" not in out
    assert out.endswith("\n")


def test_progress_line_force_updates(capsys):
    prog = ProgressLine(min_interval=60.0)
    prog.show("first")
    prog.show("second", force=True)
    prog.close()
    out = capsys.readouterr().out
    assert "first" in out
    assert "second" in out
