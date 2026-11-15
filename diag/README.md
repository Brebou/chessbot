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

The GUI provides:
- Interactive chess board manipulation
- Model move predictions with probabilities
- Position evaluation using the built-in engine
- Auto-play to specific positions
- Model self-play functionality

**Note:** Stockfish is optional. To enable it, install the Python wrapper (`uv add stockfish`) and the Stockfish engine executable, then make sure the executable is on `PATH`. The built-in engine (levels 0-4) is always available.

### Command Line Tools

The `diag/` package includes:

- `common.py`: Core utilities including `ProbePlayer` for accessing full model logits
- `gui.py`: Graphical interface for testing
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

1. **Manual Play**: Enter UCI moves directly, undo/redo
2. **Model Predictions**: See top-10 moves with probabilities
3. **Model's Move**: Play the model's chosen move (with attempt info)
4. **Evaluation**: Get position evaluation from built-in engine or Stockfish
5. **Auto-Play**: Automatically play to a target ply using the engine
6. **Self-Play**: Have the model play against itself
7. **Position Analysis**: Full game state information

## Requirements

- Python >= 3.11
- Dependencies: `torch`, `python-chess`, `numpy`, `gradio`

All dependencies are included in the project's `uv.lock` file.

## Notes

- The GUI is for testing and development only, not part of the official evaluation
- All model inference uses the official `ModelPlayer` logic (greedy, top-3 attempts)
- The built-in engine is used for evaluation by default
- Stockfish integration is optional and not required for the project
