"""Run the GUI module."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diag.gui import main

if __name__ == "__main__":
    main()
