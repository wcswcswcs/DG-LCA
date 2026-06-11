#!/usr/bin/env python3
"""v21 D-CHE/D-FOU source-channel writer runner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_mlp_source_dynamics import main_with_scope


if __name__ == "__main__":
    main_with_scope("kan_writer")
