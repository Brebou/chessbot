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

## Requirements

- Python >= 3.11
- Dependencies: `torch`, `python-chess`, `numpy`, `gradio`

All dependencies are included in the project's `uv.lock` file.

## Notes

- The GUI is for testing and development only, not part of the official evaluation
- All model inference uses the official `ModelPlayer` logic (greedy, top-3 attempts)
- Stockfish (if found) is used for evaluation and move ranking; otherwise the built-in engine is used
- Stockfish integration is optional and not required for the project
