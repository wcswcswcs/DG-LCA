#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_01_s06_truth_gate import check_update_semantics
from experiments.run_v21_01_common import ensure_out, int_flag


def main() -> None:
    out = ensure_out(os.environ.get("V21_01_OUT_DIR"))
    rows = check_update_semantics(out, os.environ.get("V21_01_DEVICE", "cuda:0"))
    assert rows and all(int_flag(r.get("pass")) for r in rows), rows


if __name__ == "__main__":
    main()
