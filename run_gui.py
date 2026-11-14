#!/usr/bin/env python
"""Convenience script to launch the ChessGPT testing GUI.

Usage:
    uv run run_gui.py [--model base_model] [--device cpu] [--share]
"""

import sys
from pathlib import Path

# Add diag directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag.gui import main

if __name__ == "__main__":
    main()
