"""Tournament results matrix generator.

Reads a competition ``results.json`` round-robin file and renders the
team-vs-team winrate matrix with a red (0%) -> green (100%) coloration.

Inputs (from the kit's results.json):
- ``round_robin.matrix[a][b]``: a's score vs b (fraction in [0, 1], draws = 0.5).
- ``round_robin.standings``: ranked list with display names.
- ``round_robin.pairs``: per-pairing matches, used for W/D/L tooltips.

Outputs:
- an HTML file (default: ``<input stem>-matrix.html``) with the colored
  matrix, and
- an ANSI truecolor rendering of the same matrix on the terminal.

Usage:
    uv run python diag/tournament_matrix.py results.json [--out out.html]
        [--sort rank|name|avg] [--no-ansi] [--csv path]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

# Piecewise-linear gradient: red (0%) -> light neutral (50%) -> green (100%).
_STOPS = [
    (0.00, (215, 48, 39)),    # red
    (0.50, (247, 247, 247)),  # near-white
    (1.00, (26, 152, 80)),    # green
]

# Cells are tinted with dark text unless the background is dark enough to
# switch to white (checked per color via relative luminance).


def _interp(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))  # type: ignore[return-value]


def color_for(value: float) -> Tuple[int, int, int]:
    """Map a score in [0, 1] to an (r, g, b) red->green color."""
    v = min(max(value, 0.0), 1.0)
    for (v0, c0), (v1, c1) in zip(_STOPS, _STOPS[1:]):
        if v <= v1:
            return _interp(c0, c1, (v - v0) / (v1 - v0))
    return _STOPS[-1][1]


def _text_color(rgb: Tuple[int, int, int]) -> str:
    """Dark text on light backgrounds, white text on saturated dark ones."""
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 96 else "#1b1b1b"


def _ansi_bg(rgb: Tuple[int, int, int], text: str, width: int) -> str:
    r, g, b = rgb
    txt = text.rjust(width) if _is_number(text) else text.ljust(width)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    fg = 97 if luminance < 96 else 30
    return f"\x1b[48;2;{r};{g};{b};{fg}m{txt}\x1b[0m"


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TournamentResults:
    """Parsed view of a round-robin results.json."""

    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        rr = self.data.get("round_robin", {})
        self.matrix: Dict[str, Dict[str, float]] = rr.get("matrix", {})
        self.standings: List[dict] = rr.get("standings", [])
        self.pairs: List[dict] = rr.get("pairs", [])

        self.names: Dict[str, str] = {}
        self.ranks: Dict[str, int] = {}
        self.scores: Dict[str, Optional[float]] = {}
        for row in self.standings:
            sid = row.get("id")
            if sid is None:
                continue
            self.names[sid] = row.get("team") or sid
            self.ranks[sid] = row.get("rank", 0)
            self.scores[sid] = row.get("score")
        # Fallback names from the teams block (e.g. empty standings).
        for sid, meta in (self.data.get("teams") or {}).items():
            self.names.setdefault(sid, (meta or {}).get("team") or sid)
        # Matrix ids missing from standings (should not happen, be safe).
        for sid in self.matrix:
            self.names.setdefault(sid, sid)
            self.ranks.setdefault(sid, 1 + len(self.ranks))
            self.scores.setdefault(sid, None)

        # Wins/draws/losses per (a, b) for tooltips.
        self.wdl: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
        for pair in self.pairs:
            a, b = pair.get("a"), pair.get("b")
            if not a or not b:
                continue
            w = d = l = 0
            for m in pair.get("matches", []):
                pts = m.get("points", (0, 1))
                if len(pts) == 2 and pts[0] == pts[1]:
                    d += 1
                elif pts[0] > pts[1]:
                    w += 1
                else:
                    l += 1
            self.wdl[(a, b)] = (w, d, l)

    @property
    def team_ids(self) -> List[str]:
        return list(self.matrix.keys())

    def cell(self, a: str, b: str) -> Optional[float]:
        v = self.matrix.get(a, {}).get(b)
        if v is None:  # tolerate one-sided entries
            v2 = self.matrix.get(b, {}).get(a)
            v = 1.0 - v2 if v2 is not None else None
        return v

    def label(self, sid: str, maxlen: int) -> str:
        name = self.names.get(sid, sid)
        rank = self.ranks.get(sid)
        short = name if len(name) <= maxlen else name[: maxlen - 1] + "\u2026"
        return f"{short} [{rank}]" if rank else short


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — round-robin matrix</title>
<style>
  body {{ font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: #fafafa; color: #1b1b1b; margin: 24px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 14px; }}
  table {{ border-collapse: separate; border-spacing: 2px; font-size: 12px; }}
  td, th {{ width: {cellw}px; min-width: {cellw}px; height: 26px;
        text-align: center; vertical-align: middle; }}
  th.rowhdr {{ position: sticky; left: 0; width: {roww}px; min-width: {roww}px;
        max-width: {roww}px; text-align: left; background: #ffffff;
        font-weight: 600; font-size: 12px; padding: 0 6px; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; z-index: 2; }}
  thead th {{ position: sticky; top: 0; background: #ffffff; z-index: 3;
        font-weight: 600; }}
  thead th.rowhdr {{ z-index: 4; }}
  .cell {{ border-radius: 3px; cursor: default; font-variant-numeric: tabular-nums; }}
  .cell:hover {{ outline: 2px solid #333; }}
  .diag {{ background: #e0e0e0; color: #888; }}
  .avg {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
  .avgcell {{ background: #ffffff; }}
  .legend td {{ width: 26px; height: 14px; }}
  .legend-label {{ font-size: 12px; color: #444; margin-top: 6px; }}
  .legend-wrap {{ margin-top: 18px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{meta}</div>
<table>
<thead>
<tr><th class="rowhdr">{colhdr}</th>{colhdrs}</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<div class="legend-wrap">
  <table class="legend"><tr>{legend}</tr></table>
  <div class="legend-label">{leglo} → &nbsp;points &nbsp;→ {leghi}</div>
</div>
</body>
</html>
"""


def build_html(title: str, meta: str, res: TournamentResults, ids: List[str],
               colw: int = 52, roww: int = 210) -> str:
    labels = [res.label(sid, 28) for sid in ids]
    colhdrs = "".join(f"<th>{i + 1}</th>" for i in range(len(ids)))

    rows = []
    for ri, a in enumerate(ids):
        cells = [f'<th class="rowhdr" title="{a}">{labels[ri]}</th>']
        for ci, b in enumerate(ids):
            if a == b:
                cells.append('<td class="diag">—</td>')
                continue
            v = res.cell(a, b)
            if v is None:
                cells.append('<td class="diag">?</td>')
                continue
            rgb = color_for(v)
            wdl = res.wdl.get((a, b))
            tip = f"{res.names.get(a, a)} vs {res.names.get(b, b)}"
            if wdl is not None:
                tip += f": {wdl[0]}W {wdl[1]}D {wdl[2]}L"
            else:
                tip += f": points {v * 100:.1f}%"
            cells.append(
                f'<td class="cell" style="background:rgb{rgb};color:{_text_color(rgb)}" '
                f'title="{tip}">{v * 100:.0f}</td>'
            )
        # Row average vs the other teams.
        opps = [v for b2 in ids if b2 != a and (v := res.cell(a, b2)) is not None]
        avg = sum(opps) / len(opps) if opps else float("nan")
        rgb = color_for(avg)
        rows.append(
            "<tr>" + "".join(cells)
            + f'<td class="avg avgcell" style="color:{_text_color(rgb)}"'
            + f' title="average score vs the field">{avg * 100:.1f}</td>'
            + "</tr>"
        )

    legend = "".join(
        f'<td style="background:rgb{color_for(i / 20)}"></td>' for i in range(21)
    )
    return _HTML_TEMPLATE.format(
        title=title,
        meta=meta,
        cellw=colw,
        roww=roww,
        colhdr="Team \\ vs",
        colhdrs=colhdrs,
        rows="\n".join(rows),
        legend=legend,
        leglo="0% (red)",
        leghi="100% (green)",
    )


# ---------------------------------------------------------------------------
# ANSI terminal output
# ---------------------------------------------------------------------------


def build_ansi(res: TournamentResults, ids: List[str]) -> str:
    names = {sid: res.label(sid, 22) for sid in ids}
    roww = max(len(n) for n in names.values()) + 1
    numw = 5

    lines: List[str] = []
    header1 = "".ljust(roww)
    idxs = []
    for i, _ in enumerate(ids):
        idxs.append(str(i + 1).rjust(numw))
    lines.append(header1 + "\x1b[1m" + "".join(idxs) + "\x1b[0m")

    for a in ids:
        cells = [names[a].ljust(roww)]
        for b in ids:
            if a == b:
                cells.append("—".rjust(numw))
                continue
            v = res.cell(a, b)
            if v is None:
                cells.append(".".rjust(numw))
            else:
                cells.append(_ansi_bg(color_for(v), f"{v * 100:.0f}", numw))
        opps = [v for b2 in ids if b2 != a and (v := res.cell(a, b2)) is not None]
        avg = sum(opps) / len(opps) if opps else float("nan")
        cells.append("\x1b[1m avg " + f"{avg * 100:.0f}%".rjust(4) + "\x1b[0m")
        lines.append("".join(cells))

    lines.append("")
    lines.append("Columns: " + ", ".join(f"{i + 1}={res.label(sid, 30)}" for i, sid in enumerate(ids)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV output (optional)
# ---------------------------------------------------------------------------


def build_csv(res: TournamentResults, ids: List[str]) -> str:
    def esc(s: str) -> str:
        return '"' + s.replace('"', '""') + '"'

    lines = [",".join(["team"] + [esc(res.names.get(b, b)) for b in ids])]
    for a in ids:
        row = [esc(res.names.get(a, a))]
        for b in ids:
            v = res.cell(a, b)
            row.append("" if v is None else f"{v:.4f}")
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def order_ids(res: TournamentResults, sort_mode: str) -> List[str]:
    ids = res.team_ids
    rank_i, name_i = res.ranks, res.names
    if sort_mode == "name":
        return sorted(ids, key=lambda s: name_i.get(s, s).lower())
    if sort_mode == "avg":
        def avg(s: str) -> float:
            vals = [v for b in ids if b != s and (v := res.cell(s, b)) is not None]
            return sum(vals) / len(vals) if vals else -1.0
        return sorted(ids, key=avg, reverse=True)
    # rank (default): standings order, unknown ranks last
    return sorted(ids, key=lambda s: (rank_i.get(s, 10**9), name_i.get(s, s)))


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the team-vs-team winrate matrix from a round-robin results.json."
    )
    parser.add_argument("results_json", type=Path, help="path to the tournament results.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="output HTML path (default: <input stem>-matrix.html)")
    parser.add_argument("--sort", choices=["rank", "name", "avg"], default="rank",
                        help="row ordering (default: rank)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write the raw winrate matrix as CSV at this path")
    parser.add_argument("--no-ansi", action="store_true",
                        help="skip the terminal rendering")
    args = parser.parse_args(argv)

    try:
        res = TournamentResults(args.results_json)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot load {args.results_json}: {exc}", file=sys.stderr)
        return 1

    ids = order_ids(res, args.sort)
    if not ids:
        print("error: no round_robin matrix found in the file", file=sys.stderr)
        return 1

    kits = res.data.get("kit", "?")
    run = res.data.get("run", "?")
    started = res.data.get("started", "?")
    title = res.data.get("mode", "round-robin").capitalize()

    out = args.out or args.results_json.with_name(args.results_json.stem + "-matrix.html")
    meta = (
        f"{len(ids)} teams &middot; kit {kits} &middot; run {run} &middot; "
        f"started {started} &middot; cell = points vs opponent (%) &middot; "
        f"ordered by {args.sort}"
    )
    html = build_html(f"{title} results ({run})", meta, res, ids)
    try:
        write_atomic(out, html)
    except OSError as exc:
        print(f"error: cannot write {out}: {exc}", file=sys.stderr)
        return 1

    if args.csv:
        try:
            write_atomic(args.csv, build_csv(res, ids))
        except OSError as exc:
            print(f"error: cannot write {args.csv}: {exc}", file=sys.stderr)
            return 1

    if not args.no_ansi:
        print(build_ansi(res, ids))
        print()

    print(f"HTML matrix written to {out}")
    if args.csv:
        print(f"CSV matrix written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
