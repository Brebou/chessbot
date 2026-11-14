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
# Launch the testing interface
uv run run_gui.py --model base_model --device cpu
```

This will start a Gradio interface at http://localhost:7860 where you can:
- Play moves manually
- See model predictions
- Evaluate positions
- Auto-play to specific positions
- Have the model play against itself

## Project Structure

- `diag/`: Diagnostic tools and GUI
  - `common.py`: Core utilities and ProbePlayer
  - `gui.py`: Graphical interface
  - `run_eval.py`: CLI evaluation tool
- `run_gui.py`: Convenience script to launch the GUI

## License

MIT
