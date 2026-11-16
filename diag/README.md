# ChessGPT Diagnostic Tools

This directory contains diagnostic and testing tools for the ChessGPT model.

## Quick Start

### GUI Interface

For visual testing and exploration:

```bash
# Launch the graphical interface
uv run run_gui.py --model base_model --device cpu

# Or using the module directly
uv run -m diag --model base_model --device cpu
```

The GUI is a single page and provides:
- A drag-and-drop board (or click a piece, then its destination) with an evaluation bar
- Automatic position analysis after every move (toggle: *Auto-analyze*)
- Model candidate moves with probabilities, each annotated with its **Stockfish rank** (rank among all
  legal moves, sorted by the evaluation of the resulting position for the side to move; 1 = best)
- Model move / undo / reset / flip, auto-play to a position, and model self-play (collapsible sections)

**Stockfish:** the engine *executable* is needed for the rankings (`--stockfish PATH`, `$STOCKFISH_PATH`, `PATH`,
or a common install dir such as `/usr/games/stockfish`). Without it the GUI falls back to the built-in
engine (shallow search; rankings are only indicative).

### Command Line Tools

The `diag/` package includes:

- `common.py`: Core utilities including `ProbePlayer` for accessing full model logits
- `gui.py`: Single-page graphical interface for testing
- `board_component.py`: Drag-and-drop board widget (HTML/JS used by the GUI)
- `__init__.py`: Package initialization

## Features

### ProbePlayer

Extended version of `ModelPlayer` that provides access to full logits:

```python
from diag.common import ProbePlayer, load

model, tokenizer = load("base_model", device="cpu")
probe = ProbePlayer(model, tokenizer)

# Get logits for a position
history = ["e2e4", "e7e5"]
logits = probe.get_logits(history)

# Get top-k predictions with probabilities
tokens, probs = probe.get_topk(history, k=5)

# Get probabilities for all legal moves
import chess
board = chess.Board()
board.push_uci("e2e4")
board.push_uci("e7e5")
legal_probs = probe.get_legal_probs(history, board)
```

### GUI Capabilities

1. **Play**: drag & drop, click-to-move, or type a move (UCI/SAN); undo, reset, flip
2. **Auto-analysis**: eval bar, best moves and the model's candidates re-computed after each move (can be switched off)
3. **Model candidates**: top-10 tokens with probability, model order, and Stockfish rank / eval / loss vs best
4. **Model's move**: played with its Stockfish rank shown and stored in the move history (`[SF #rank/total]`)
5. **Auto-play / self-play**: engine plays to a target ply; model plays against itself
6. **Position info**: FEN, move history and full game state (collapsed by default)

### Command Line Tools

The `diag/` package includes:

- `common.py`: Core utilities including `ProbePlayer` for accessing full model logits
- `gui.py`: Single-page graphical interface for testing
- `board_component.py`: Drag-and-drop board widget (HTML/JS used by the GUI)
- `__init__.py`: Package initialization

## Diagnostic Pipeline (C1-C14)

The full capability profile is produced by five CLI stages (all support `--help`):

```bash
# 1. Corpora (on-policy games, mate1, captures, endgames, conversion, promotion)
uv run diag/gen_corpus.py --model base_model --out results/corpus --workers 4
# 2. Probes P1-P7 (P7 needs a Lichess dataset: --lichess data.jsonl.gz)
uv run diag/probes.py --model base_model --corpus results/corpus --out results/probes --workers 4
# 3. Multi-seed evaluation vs the engine (C4)
uv run diag/run_eval.py --model base_model --seeds 0 1 --levels 0 1 2 3 --n_openings 50 --workers 4 --out results/eval_base --games_out
# 4. Game analysis (C4 by level, C8 fallback, C11 repetitions, C12 variance)
uv run diag/analyze_games.py --games results/eval_base --out results/analysis
# 5. Capability profile report
uv run diag/report.py --eval results/eval_base --analysis results/analysis --probes results/probes --out results/profile
```

Outputs: `results/corpus/*.jsonl` + `manifest.json`, `results/probes/probes_base.json`,
`results/eval_base/summary.json` + per-seed game files, `results/analysis/analysis_base.json`,
and `results/profile/profile.md` + `profile.json`.

Notes:

- **Lichess dataset:** by default `gen_corpus.py` auto-downloads the model's actual
  training corpus, [dlouapre/lichess_2025-01_1M](https://huggingface.co/datasets/dlouapre/lichess_2025-01_1M)
  (~280 MB, 1M games in the annotated `WQd1h5(x+)` notation), parses it to UCI with
  per-game replay validation and caches it as `data/lichess_train_2025-01_1M.jsonl.gz`.
  This makes P7's seen/unseen split (index 940,000) **exact**: the first 940,000 games
  are the model's training data, the rest are unseen. Use `--lichess-source uci` to
  download a month of [austindavis/lichess-uci](https://huggingface.co/datasets/austindavis/lichess-uci)
  instead (UCI notation with Elo fields, `--lichess-month YYYYMM`, default 2014-12),
  `--lichess PATH` for a local file, or `--no-lichess` to disable (then `mate1`/
  `captures` fall back to scanning the on-policy games, and P7 (C13) is recorded as
  `SKIPPED` in `probes_base.json`). `probes.py` reuses the downloaded file
  automatically for P7 (training dataset preferred). If the dataset has fewer games
  than the split index, P7 falls back to an adaptive 90/10 split.
- `fast_engine.py` (optimized engine, move-identical to `chess_hw.engine.Engine`) and
  `fast_probe.py` (last-position batched logits, bit-identical to `ProbePlayer._batch_logits`)
  are used by the pipeline for speed; `chess_hw/` itself is never modified.
- All stages are seeded and deterministic: running the probes twice yields identical output.
- `tests/` contains SEE unit tests (`test_see.py`) and corpus validation (`test_corpus.py`,
  override the corpus directory with `CHESSBOT_CORPUS_DIR`).

## Requirements

- Python >= 3.11
- Dependencies: `torch`, `python-chess`, `numpy`, `gradio`

All dependencies are included in the project's `uv.lock` file.

## Notes

- The GUI is for testing and development only, not part of the official evaluation
- All model inference uses the official `ModelPlayer` logic (greedy, top-3 attempts)
- Stockfish (if found) is used for evaluation and move ranking; otherwise the built-in engine is used
- Stockfish integration is optional and not required for the project
