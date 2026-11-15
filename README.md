# ChessBot

A chess AI project using the ChessGPT model for testing and evaluation.

## Quick Start

### Prerequisites

- Python >= 3.11
- uv (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Brebou/chessbot.git
cd chessbot

# Install dependencies
uv sync
```

### Running the GUI

```bash
# Launch the testing interface (optionally point to a Stockfish executable)
uv run run_gui.py --model base_model --device cpu [--stockfish /path/to/stockfish] [--depth 10]
```

This starts a single-page Gradio interface at http://localhost:7860 where you can:
- Play by **dragging and dropping** pieces (or click a piece, then a destination)
- Get an **automatic evaluation** (eval bar + best moves) after every move — toggle it with *Auto-analyze*
- See the model's top candidate moves, each with its **Stockfish rank**: the move's position among *all* legal
  moves sorted by the evaluation of the resulting position for the side to move (1 = Stockfish's best)
- Let the model play a move (its Stockfish rank is shown and kept in the move history), undo, reset, flip the board
- Auto-play to a given ply, or have the model play against itself (collapsible sections)

Stockfish is looked up via `--stockfish`, `$STOCKFISH_PATH`, `PATH`, and a few common install locations
(e.g. `/usr/games/stockfish` on Debian/Ubuntu). Without it the GUI falls back to the built-in engine
(shallow search, so rankings are only indicative).

## Project Structure

- `diag/`: Diagnostic tools and GUI
  - `common.py`: Core utilities and ProbePlayer
  - `gui.py`: Graphical interface (single page)
  - `board_component.py`: Drag-and-drop board widget (HTML/JS for Gradio)
  - `run_eval.py`: CLI evaluation tool
- `run_gui.py`: Convenience script to launch the GUI

## License

MIT
