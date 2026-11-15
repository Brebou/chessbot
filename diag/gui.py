"""Single-page graphical interface for testing the ChessGPT model.

Features
- Drag-and-drop (or click-to-move) board with an evaluation bar.
- Automatic position analysis (toggle) using Stockfish; falls back to the built-in
  engine if no Stockfish executable can be found.
- For every move proposed by the model, the rank shown is the *Stockfish* rank:
  its position among all legal moves sorted by how good the resulting position is
  for the side to move (1 = Stockfish's best move).
- Everything lives on one page; secondary tools are in collapsible accordions.

Usage:
    uv run run_gui.py --model base_model --device cpu [--stockfish /path/to/stockfish]
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import random
import shutil
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import gradio as gr
import numpy as np

# Add parent directory to path to import chess_hw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import ChessTokenizer, Engine, load_model
from chess_hw.engine import MATE as _BUILTIN_MATE
from chess_hw.play import ModelPlayer

from . import board_component
from .common import ProbePlayer, material_diff, material_value, phase

MATE_CP = 100_000  # centipawn value used to order mate scores


# --------------------------------------------------------------------------- #
# Engines / analysis
# --------------------------------------------------------------------------- #
def find_stockfish(explicit: Optional[str] = None) -> Optional[str]:
    """Locate a Stockfish executable (explicit path, $STOCKFISH_PATH, PATH, common dirs)."""
    candidates = [explicit, os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish")]
    candidates += [
        "/usr/games/stockfish",  # Debian / Ubuntu apt
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",  # macOS (Apple silicon, brew)
        "/usr/bin/stockfish",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def fmt_cp(cp: int, mate: Optional[int] = None, signed: bool = True) -> str:
    """Format a score in pawns ('+0.35') or as a mate ('#3', '#-2')."""
    if mate is not None:
        return f"#{mate}"
    return f"{cp / 100:+.2f}" if signed else f"{cp / 100:.2f}"


@dataclass
class MoveEval:
    uci: str
    san: str
    cp: int  # from the mover's point of view (mate mapped to +-MATE_CP)
    mate: Optional[int]  # mover's point of view, None if not a mate score
    rank: int = 0  # 1 = best (ties share a rank)


@dataclass
class Analysis:
    fen: str
    source: str
    depth: int
    turn: bool  # chess.WHITE / chess.BLACK = side to move
    moves: dict = field(default_factory=dict)  # uci -> MoveEval
    ordered: list = field(default_factory=list)  # MoveEval sorted best -> worst

    @property
    def n(self) -> int:
        return len(self.moves)

    @property
    def best(self) -> Optional[MoveEval]:
        return self.ordered[0] if self.ordered else None

    def white_pov(self) -> tuple:
        """(cp, mate) of the position from White's point of view."""
        b = self.best
        if b is None:
            return 0, None
        sign = 1 if self.turn == chess.WHITE else -1
        return sign * b.cp, (sign * b.mate if b.mate is not None else None)

    def white_text(self) -> str:
        cp, mate = self.white_pov()
        return fmt_cp(cp, mate)

    def white_pct(self) -> float:
        cp, mate = self.white_pov()
        if mate is not None:
            return 100.0 if mate > 0 else 0.0
        return 100.0 / (1.0 + 10 ** (-cp / 400.0))


def _finalize(analysis: Analysis) -> Analysis:
    analysis.ordered = sorted(analysis.moves.values(), key=lambda m: -m.cp)
    rank, prev = 0, None
    for i, m in enumerate(analysis.ordered, start=1):
        if m.cp != prev:  # competition ranking: ties share the best rank
            rank, prev = i, m.cp
        m.rank = rank
    return analysis


class StockfishAnalyzer:
    """Ranks *all* legal moves with a single MultiPV search."""

    name = "Stockfish"

    def __init__(self, path: str, threads: Optional[int] = None, hash_mb: int = 128):
        self.path = path
        self._closed = False
        threads = threads or max(1, min(8, (os.cpu_count() or 2) - 1))
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            self.engine.configure({"Threads": threads, "Hash": hash_mb})
        except chess.engine.EngineError:
            pass
        atexit.register(self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.engine.quit()
        except Exception:
            pass

    def analyse(self, board: chess.Board, depth: int) -> Analysis:
        a = Analysis(board.fen(), self.name, depth, board.turn)
        legal = list(board.legal_moves)
        if not legal:
            return a
        infos = self.engine.analyse(board, chess.engine.Limit(depth=depth), multipv=len(legal))
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            mv = pv[0]
            rel = info["score"].relative
            a.moves[mv.uci()] = MoveEval(mv.uci(), board.san(mv), rel.score(mate_score=MATE_CP), rel.mate())
        return _finalize(a)


class BuiltinAnalyzer:
    """Fallback: full-window search of every root move with the project's Engine."""

    name = "Built-in engine"

    def analyse(self, board: chess.Board, depth: int) -> Analysis:
        level = max(1, min(3, depth // 4 or 1))  # keep it responsive: 1..3 plies
        eng = Engine(level=level)
        a = Analysis(board.fen(), self.name, level, board.turn)
        b = board.copy(stack=False)
        for mv in list(b.legal_moves):
            san = b.san(mv)
            b.push(mv)
            s = -eng._search(b, level - 1, -_BUILTIN_MATE - 1, _BUILTIN_MATE + 1, 1)
            b.pop()
            mate = None
            if abs(s) > _BUILTIN_MATE - 1000:
                mate = (_BUILTIN_MATE - abs(s) + 1) // 2 * (1 if s > 0 else -1)
                s = MATE_CP if s > 0 else -MATE_CP
            a.moves[mv.uci()] = MoveEval(mv.uci(), san, s, mate)
        return _finalize(a)


# --------------------------------------------------------------------------- #
# Main application state
# --------------------------------------------------------------------------- #
def resolve(board: chess.Board, token: str) -> Optional[chess.Move]:
    """Token -> legal move (normalises castling like the official harness), else None."""
    try:
        return board.parse_uci(token)
    except ValueError:
        return None


class ChessGUI:
    def __init__(self, model_path: str, device: str = "cpu", stockfish_path: Optional[str] = None,
                 depth: int = 10):
        self.model_path = Path(model_path)
        self.device = device
        self.default_depth = depth
        self.lock = threading.RLock()

        self.model = load_model(str(model_path), device=device)
        self.tokenizer = ChessTokenizer()
        self.model_player = ModelPlayer(self.model, name="model")
        self.probe_player = ProbePlayer(self.model, self.tokenizer, device=device)

        self.analyzer = None
        sf = find_stockfish(stockfish_path)
        if sf:
            try:
                self.analyzer = StockfishAnalyzer(sf)
            except Exception as exc:  # engine failed to start
                print(f"Stockfish failed to start ({exc}); falling back to built-in engine.", file=sys.stderr)
        else:
            print("Stockfish executable not found (use --stockfish or $STOCKFISH_PATH); "
                  "falling back to the built-in engine.", file=sys.stderr)
        if self.analyzer is None:
            self.analyzer = BuiltinAnalyzer()

        self._cache: "OrderedDict[tuple, Analysis]" = OrderedDict()
        self.board = chess.Board()
        self.notes: dict = {}  # ply index -> annotation for model-played moves
        self.orientation = "white"
        self.msg = ""
        self.rev = 0

    # ---- analysis ---------------------------------------------------------
    def _key(self, board: chess.Board, depth: int) -> tuple:
        return (board.fen(), self.analyzer.name, depth)

    def cached(self, board: Optional[chess.Board] = None) -> Optional[Analysis]:
        board = board or self.board
        with self.lock:
            for (fen, name, _d), a in reversed(self._cache.items()):
                if fen == board.fen() and name == self.analyzer.name:
                    return a
        return None

    def get_analysis(self, board: Optional[chess.Board] = None, depth: Optional[int] = None) -> Analysis:
        board = board or self.board
        depth = int(depth or self.default_depth)
        key = self._key(board, depth)
        with self.lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            a = self.analyzer.analyse(board.copy(stack=False), depth)
            self._cache[key] = a
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
            return a

    # ---- model ------------------------------------------------------------
    def model_candidates(self, analysis: Optional[Analysis], k: int = 10) -> tuple:
        """Top-k model tokens with their Stockfish rank. Returns (rows, summary)."""
        board = self.board
        history = [m.uci() for m in board.move_stack]
        try:
            logits = self.probe_player.get_logits(history)
        except AssertionError as exc:  # history longer than the model context
            return [], f"Model unavailable: {exc}"
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = np.argsort(probs)[::-1][:k]
        rows, n_legal, first_rank = [], 0, None
        for i, idx in enumerate(top, start=1):
            tok = self.tokenizer.id_to_token[int(idx)]
            mv = resolve(board, tok)
            p = f"{probs[idx] * 100:.2f}%"
            if mv is None:
                rows.append(["—", f"{tok} (illegal)", p, i, "", ""])
                continue
            n_legal += 1
            me = analysis.moves.get(mv.uci()) if analysis else None
            if me is None:
                rows.append(["…", f"{board.san(mv)} ({mv.uci()})", p, i, "", ""])
                continue
            if first_rank is None and i == 1:
                first_rank = me.rank
            best = analysis.best
            if me.mate is not None or best.mate is not None:
                delta = "0.00" if me.cp == best.cp else "—"
            else:
                delta = f"{(me.cp - best.cp) / 100:+.2f}" if me.cp != best.cp else "0.00"
            rows.append([f"{me.rank}/{analysis.n}", f"{me.san} ({me.uci})", p, i,
                         fmt_cp(me.cp, me.mate), delta])
        summary = f"{n_legal}/{len(top)} of the model's top-{len(top)} tokens are legal"
        return rows, summary

    def _rank_text(self, analysis: Analysis, move: chess.Move) -> str:
        me = analysis.moves.get(move.uci())
        if me is None:
            return ""
        best = analysis.best
        loss = "" if (me.mate is not None or best.mate is not None) else f", {(me.cp - best.cp) / 100:+.2f} vs best"
        return f"Stockfish rank {me.rank}/{analysis.n}{loss}" if analysis.source == "Stockfish" \
            else f"{analysis.source} rank {me.rank}/{analysis.n}{loss}"

    def _short_note(self, analysis: Optional[Analysis], move: chess.Move) -> str:
        if not analysis:
            return ""
        me = analysis.moves.get(move.uci())
        return f"#{me.rank}/{analysis.n}" if me else ""

    # ---- moves ------------------------------------------------------------
    def _push(self, move: chess.Move, note: str = "") -> None:
        self.board.push(move)
        if note:
            self.notes[len(self.board.move_stack) - 1] = note

    def play_move(self, text: str) -> None:
        text = (text or "").strip()
        with self.lock:
            if not text:
                return
            try:
                try:
                    move = self.board.parse_uci(text)
                except ValueError:
                    move = self.board.parse_san(text)
            except ValueError:
                self.msg = f"❌ Illegal or unreadable move: `{text}`"
                return
            self._push(move)
            self.msg = ""

    def undo_move(self) -> None:
        with self.lock:
            if self.board.move_stack:
                self.notes.pop(len(self.board.move_stack) - 1, None)
                self.board.pop()
            self.msg = ""

    def redo_move(self) -> None:
        with self.lock:
            self.msg = "Redo not fully supported by python-chess"
            # Note: python-chess doesn't support redo natively
            # We'd need to maintain our own move history for full redo support

    def reset_board(self) -> None:
        with self.lock:
            self.board = chess.Board()
            self.notes.clear()
            self.msg = ""

    def flip(self) -> None:
        self.orientation = "black" if self.orientation == "white" else "white"

    def get_top_3_moves(self, depth: int = 10) -> list:
        """Get the top 3 engine moves for the current position."""
        analysis = self.get_analysis(self.board, depth)
        if analysis and analysis.ordered:
            return [m.uci() for m in analysis.ordered[:3]]
        return []

    def play_model_move(self, depth: int) -> None:
        with self.lock:
            if self.board.is_game_over():
                self.msg = "Game is over."
                return
            try:
                choice = self.model_player.choose(self.board)
            except AssertionError as exc:
                self.msg = f"Model unavailable: {exc}"
                return
            move, attempt = choice.move, choice.attempt
            fallback = move is None
            if fallback:
                move = random.choice(list(self.board.legal_moves))
            analysis = self.get_analysis(self.board, depth)  # rank is computed *before* the move
            san = self.board.san(move)
            self._push(move, self._short_note(analysis, move))
            how = "all top-3 illegal, random move" if fallback else f"attempt {attempt}"
            self.msg = f"🤖 Model played **{san}** ({how}) — {self._rank_text(analysis, move)}"

    def auto_play_to_ply(self, target_ply: int, engine_depth: int) -> None:
        with self.lock:
            engine = Engine(level=int(engine_depth), seed=42)
            # Negative target_ply means add |target_ply| moves from current position
            if int(target_ply) < 0:
                n_moves = -int(target_ply)
                for _ in range(n_moves):
                    if self.board.is_game_over():
                        break
                    mv = engine.best_move(self.board)
                    if mv is None:
                        break
                    self._push(mv)
                self.msg = f"Engine auto-played {n_moves} moves from current position."
            else:
                while self.board.ply() < int(target_ply) and not self.board.is_game_over():
                    mv = engine.best_move(self.board)
                    if mv is None:
                        break
                    self._push(mv)
                self.msg = f"Engine auto-played to ply {self.board.ply()}."

    def model_self_play(self, max_moves: int, annotate: bool, depth: int) -> None:
        with self.lock:
            played = 0
            for _ in range(int(max_moves)):
                if self.board.is_game_over():
                    break
                try:
                    choice = self.model_player.choose(self.board)
                except AssertionError:
                    break
                move = choice.move or random.choice(list(self.board.legal_moves))
                note = ""
                if annotate:
                    note = self._short_note(self.get_analysis(self.board, depth), move)
                self._push(move, note)
                played += 1
            outcome = self.board.outcome(claim_draw=True)
            self.msg = f"Self-play: {played} plies" + (f" — result {outcome.result()}" if outcome else "")

    # ---- views ------------------------------------------------------------
    def history_text(self) -> str:
        if not self.board.move_stack:
            return "No moves yet"
        tmp, items = chess.Board(), []
        for i, mv in enumerate(self.board.move_stack):
            san = tmp.san(mv)
            tmp.push(mv)
            note = self.notes.get(i)
            items.append(f"{san} [SF {note}]" if note else san)
        lines = []
        for i in range(0, len(items), 2):  # one full move per line: "3. Bc4 Nf6"
            lines.append(f"{i // 2 + 1}. " + " ".join(items[i:i + 2]))
        return "\n".join(lines)

    def game_info(self) -> dict:
        b = self.board
        return {
            "fen": b.fen(), "ply": b.ply(), "turn": "white" if b.turn else "black",
            "is_check": b.is_check(), "is_checkmate": b.is_checkmate(), "is_stalemate": b.is_stalemate(),
            "is_insufficient_material": b.is_insufficient_material(), "can_claim_draw": b.can_claim_draw(),
            "halfmove_clock": b.halfmove_clock, "fullmove_number": b.fullmove_number,
            "legal_moves_count": b.legal_moves.count(),
            "material_white": material_value(b, chess.WHITE), "material_black": material_value(b, chess.BLACK),
            "material_diff": material_diff(b, chess.WHITE), "phase": phase(b.ply()),
        }

    def board_value(self) -> dict:
        b = self.board
        self.rev += 1
        legal: dict = {}
        for mv in b.legal_moves:
            legal.setdefault(chess.square_name(mv.from_square), []).append(mv.uci())
        # Let the user also drag the king onto g/c file for castling: already in legal (e1g1 etc.)
        pieces = {chess.square_name(s): p.symbol() for s, p in b.piece_map().items()}
        last = None
        if b.move_stack:
            m = b.peek()
            last = [chess.square_name(m.from_square), chess.square_name(m.to_square)]
        check = chess.square_name(b.king(b.turn)) if b.is_check() else None
        a = self.cached()
        outcome = b.outcome()
        over = ""
        if outcome:
            over = {"1-0": "White wins", "0-1": "Black wins"}.get(outcome.result(), "Draw") + \
                   f" ({outcome.termination.name.lower().replace('_', ' ')})"
        return {
            "rev": self.rev, "pieces": pieces, "turn": "w" if b.turn else "b",
            "orientation": self.orientation, "legal": legal, "last": last, "check": check, "over": over,
            "eval": {"white_pct": round(a.white_pct(), 1), "text": a.white_text()} if a and a.n else None,
        }

    def status_md(self, analyzing: bool) -> str:
        b = self.board
        outcome = b.outcome()
        head = f"**Game over: {outcome.result()}**" if outcome else \
            f"**{'White' if b.turn else 'Black'} to move**" + (" (check)" if b.is_check() else "")
        a = self.cached()
        if a and a.n:
            top = ", ".join(f"{m.san} ({fmt_cp(m.cp, m.mate)})" for m in a.ordered[:3])
            head += f" · Eval **{a.white_text()}** ({a.source}, depth {a.depth}) · best: {top}"
        elif a is not None:
            head += f" · Eval **{a.white_text()}**"
        elif analyzing:
            head += " · analyzing…"
        else:
            head += " · _not analyzed_"
        return head + (f"  \n{self.msg}" if self.msg else "")

    def snapshot(self, keep_table: bool) -> tuple:
        """(board, status, table, history, fen, info) – table is left untouched when keep_table."""
        with self.lock:
            table = gr.skip() if keep_table else []
            return (self.board_value(), self.status_md(analyzing=keep_table), table,
                    self.history_text(), self.board.fen(), self.game_info())

    def analyze(self, depth: int) -> tuple:
        """Run the analysis for the current position. Returns (board, status, table)."""
        with self.lock:
            if self.board.is_game_over():
                rows, summary = [], ""
                self.get_analysis(self.board, depth)  # trivial: no legal moves
            else:
                a = self.get_analysis(self.board, depth)
                rows, summary = self.model_candidates(a)
            status = self.status_md(analyzing=False)
            if summary:
                status += f"  \n_Model: {summary}._"
            return self.board_value(), status, rows


# --------------------------------------------------------------------------- #
# Gradio interface
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container { max-width: 1280px !important; padding-top: 6px !important; }
#cb-title { margin: 0 !important; } #cb-title h3 { margin: 4px 0 !important; }
.cb-compact { gap: 6px !important; }
.cb-compact button { min-width: 0 !important; }
footer { display: none !important; }
"""

TABLE_HEADERS = ["SF rank", "Move", "Model prob", "Model #", "Eval (mover)", "Δ vs best"]


def create_interface(model_path: str, device: str = "cpu", stockfish_path: Optional[str] = None,
                     depth: int = 10):
    gui = ChessGUI(model_path, device, stockfish_path, depth)
    engine_name = gui.analyzer.name
    max_depth = 24 if engine_name == "Stockfish" else 12

    with gr.Blocks(title="ChessGPT Tester") as demo:
        gr.Markdown("### ♟ ChessGPT Tester — drag pieces to play", elem_id="cb-title")

        with gr.Row(equal_height=False):
            # ---------------- left: board ----------------
            with gr.Column(scale=5, min_width=380):
                board = gr.HTML(
                    gui.board_value(),
                    html_template=board_component.HTML_TEMPLATE,
                    css_template=board_component.CSS_TEMPLATE,
                    js_on_load=board_component.js_on_load(),
                    padding=False,
                )
                with gr.Row(elem_classes="cb-compact"):
                    model_btn = gr.Button("🤖 Model move", variant="primary", size="sm")
                    undo_btn = gr.Button("⬅️ Undo", size="sm", shortcut="left")
                    redo_btn = gr.Button("➡️ Redo", size="sm", shortcut="right")
                    flip_btn = gr.Button("⇅ Flip", size="sm")
                    reset_btn = gr.Button("⟲ Reset", variant="stop", size="sm")
                with gr.Row(elem_classes="cb-compact"):
                    best_move_1_btn = gr.Button("➡️ Best 1", size="sm")
                    best_move_2_btn = gr.Button("➡️ Best 2", size="sm")
                    best_move_3_btn = gr.Button("➡️ Best 3", size="sm")
                status = gr.Markdown(gui.status_md(False))

            # ---------------- right: analysis + accordions ----------------
            with gr.Column(scale=6, min_width=420):
                with gr.Row(elem_classes="cb-compact"):
                    auto = gr.Checkbox(label=f"Auto-analyze ({engine_name})", value=True, scale=3)
                    analyze_btn = gr.Button("Analyze now", size="sm", scale=1)

                with gr.Accordion("Model candidates — ranked by Stockfish", open=True):
                    gr.Markdown(
                        "*SF rank* = rank of the move among **all legal moves**, sorted by the evaluation "
                        "of the resulting position for the side to move (1 = engine's best). "
                        "*Model #* is the model's own order.",
                    )
                    table = gr.Dataframe(
                        headers=TABLE_HEADERS, value=[], interactive=False, wrap=False,
                        datatype=["str", "str", "str", "number", "str", "str"],
                        column_widths=["12%", "30%", "14%", "10%", "17%", "17%"], max_height=330,
                    )

                with gr.Accordion("Moves & position", open=False):
                    history = gr.Textbox(gui.history_text(), label="Move history  ([SF #rank/total] = model moves)",
                                         lines=5, max_lines=8, interactive=False)
                    fen = gr.Textbox(gui.board.fen(), label="FEN", interactive=False)
                    with gr.Row(elem_classes="cb-compact"):
                        uci_in = gr.Textbox(label="Type a move (e2e4 or Nf3)", placeholder="e2e4", scale=3)
                        play_btn = gr.Button("Play", scale=1)
                    info = gr.JSON(gui.game_info(), label="Game info", open=False)

                with gr.Accordion("Auto-play & self-play", open=False):
                    with gr.Row():
                        target_ply = gr.Slider(0, 120, value=10, step=1, label="Auto-play to ply")
                        eng_level = gr.Slider(1, 4, value=2, step=1, label="Built-in engine level")
                        auto_play_btn = gr.Button("Auto-play")
                    with gr.Row():
                        n_moves = gr.Slider(1, 20, value=5, step=1, label="Add moves from current")
                        add_moves_btn = gr.Button("Add moves")
                    with gr.Row():
                        max_moves = gr.Slider(1, 60, value=10, step=1, label="Model self-play: plies")
                        self_btn = gr.Button("Model vs Model")

                with gr.Accordion("Engine settings", open=False):
                    depth_sl = gr.Slider(
                        4, max_depth, value=min(depth, max_depth), step=1,
                        label="Analysis depth" + ("" if engine_name == "Stockfish"
                                                  else " (built-in engine: depth//4 plies, max 3)"),
                    )
                    gr.Markdown(f"Engine: **{engine_name}**" +
                                (f" (`{gui.analyzer.path}`)" if engine_name == "Stockfish" else
                                 " — Stockfish not found; pass `--stockfish PATH` for real rankings."))

        outs = [board, status, table, history, fen, info]
        fast = dict(show_progress="hidden", trigger_mode="multiple", api_name=False)

        def after(ev):
            """Chain the (slower) analysis after a fast state change."""
            return ev.then(run_analysis, [auto, depth_sl], [board, status, table],
                           show_progress="hidden", trigger_mode="always_last", concurrency_limit=1, api_name=False)

        def run_analysis(auto_on, d):
            if not auto_on:
                return gr.skip(), gr.skip(), gr.skip()
            return gui.analyze(d)

        # --- board drag & drop ---
        def on_board(auto_on, evt: gr.EventData):
            def field(name):  # EventData raises when the call carries no payload (e.g. API clients)
                try:
                    return getattr(evt, name, None)
                except (TypeError, AttributeError):
                    return None

            if field("action") == "move":
                gui.play_move(field("uci") or "")
            return gui.snapshot(keep_table=auto_on)

        after(board.click(on_board, [auto], outs, **fast))

        # --- buttons ---
        def do_undo(a):
            gui.undo_move()
            return gui.snapshot(a)

        def do_redo(a):
            gui.redo_move()
            return gui.snapshot(a)

        def do_reset(a):
            gui.reset_board()
            return gui.snapshot(a)

        def do_flip(a):
            gui.flip()
            return gui.snapshot(True)  # keep table: flipping changes nothing about the position

        def do_model(a, d):
            gui.play_model_move(d)
            return gui.snapshot(a)

        def do_type(a, text):
            gui.play_move(text)
            return (*gui.snapshot(a), "")

        def do_auto_play(a, ply, lvl):
            gui.auto_play_to_ply(ply, lvl)
            return gui.snapshot(a)

        def do_self(a, n, d):
            gui.model_self_play(n, annotate=a, depth=d)
            return gui.snapshot(a)

        def do_add_moves(a, n, lvl):
            gui.auto_play_to_ply(-n, lvl)
            return gui.snapshot(a)

        def do_play_best_move(a, depth, move_idx):
            top_moves = gui.get_top_3_moves(depth)
            if move_idx < len(top_moves):
                try:
                    move = gui.board.parse_uci(top_moves[move_idx])
                    gui._push(move)
                except ValueError:
                    gui.msg = f"Could not play move {top_moves[move_idx]}"
            else:
                gui.msg = "Not enough engine moves available"
            return gui.snapshot(a)

        after(undo_btn.click(do_undo, [auto], outs, **fast))
        after(redo_btn.click(do_redo, [auto], outs, **fast))
        after(reset_btn.click(do_reset, [auto], outs, **fast))
        flip_btn.click(do_flip, [auto], outs, **fast)
        after(model_btn.click(do_model, [auto, depth_sl], outs, **fast))
        after(play_btn.click(do_type, [auto, uci_in], outs + [uci_in], **fast))
        after(uci_in.submit(do_type, [auto, uci_in], outs + [uci_in], **fast))
        after(auto_play_btn.click(do_auto_play, [auto, target_ply, eng_level], outs, **fast))
        after(self_btn.click(do_self, [auto, max_moves, depth_sl], outs, **fast))
        after(add_moves_btn.click(do_add_moves, [auto, n_moves, eng_level], outs, **fast))
        after(best_move_1_btn.click(lambda a, d: do_play_best_move(a, d, 0), [auto, depth_sl], outs, **fast))
        after(best_move_2_btn.click(lambda a, d: do_play_best_move(a, d, 1), [auto, depth_sl], outs, **fast))
        after(best_move_3_btn.click(lambda a, d: do_play_best_move(a, d, 2), [auto, depth_sl], outs, **fast))

        # --- analysis triggers ---
        analyze_btn.click(lambda d: gui.analyze(d), [depth_sl], [board, status, table],
                          show_progress="hidden", concurrency_limit=1, api_name=False)
        auto.change(run_analysis, [auto, depth_sl], [board, status, table], show_progress="hidden",
                    concurrency_limit=1, api_name=False)
        depth_sl.release(run_analysis, [auto, depth_sl], [board, status, table], show_progress="hidden",
                         concurrency_limit=1, api_name=False)

        # --- initial state (also refreshes after a page reload) ---
        demo.load(lambda: gui.snapshot(keep_table=False), None, outs, api_name=False).then(
            run_analysis, [auto, depth_sl], [board, status, table],
            show_progress="hidden", concurrency_limit=1, api_name=False)

    demo._cb_gui = gui  # handy for tests
    return demo


def main():
    parser = argparse.ArgumentParser(description="ChessGPT Testing GUI")
    parser.add_argument("--model", type=str, default="base_model", help="Path to model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--stockfish", type=str, default=None, help="Path to the Stockfish executable")
    parser.add_argument("--depth", type=int, default=10, help="Default Stockfish analysis depth")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / args.model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {args.model}")

    demo = create_interface(str(model_path), args.device, args.stockfish, args.depth)
    try:
        demo.launch(share=args.share, server_name=args.host, server_port=args.port,
                    theme=gr.themes.Soft(), css=CSS)
    finally:
        # python-chess runs the engine in a non-daemon thread: quit it explicitly, otherwise
        # the interpreter can hang on exit after Ctrl+C.
        close = getattr(demo._cb_gui.analyzer, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
