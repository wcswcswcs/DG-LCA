#!/usr/bin/env python3
"""Entry point for v22.16-M MLP+FU finalizer patch tests and recap."""

from __future__ import annotations

import sys

from experiments.run_v22_16_mlp_fu_mainline import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "finalize"])
    main()
