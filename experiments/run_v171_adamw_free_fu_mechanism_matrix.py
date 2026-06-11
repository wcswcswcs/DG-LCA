#!/usr/bin/env python3
"""v17 AdamW-free FU mechanism matrix shard runner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v17_common import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "mechanism-shard"])
    main()
