#!/usr/bin/env python3
"""v22 C3 D-CHE/D-FOU source-channel writer wrapper."""

from __future__ import annotations

import sys

from experiments.run_v22_source_chain_dynamics import main


if __name__ == "__main__":
    if "--v21-scope" not in sys.argv and "--merge-only" not in sys.argv:
        sys.argv.extend(["--v21-scope", "kan", "--carriers", "D-CHE,D-FOU", "--run-label", "v2200_c3_kan_source_writer"])
    main()
