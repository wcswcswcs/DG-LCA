#!/usr/bin/env python3
"""Refresh v22.77 Part D route from existing family summaries."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_77_conditional_edge_signal_metric_kan_mpfu import OUT_ROOT, read_rows, route_from_part_d, write_json


def main() -> None:
    preflight_path = OUT_ROOT / "v22_77_part_d_preflight_route.json"
    family_path = OUT_ROOT / "v22_77_part_d_family_summaries.csv"
    family = read_rows(family_path)
    obj = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {}
    route, reason = route_from_part_d(family)  # type: ignore[arg-type]
    obj.update(
        {
            "preflight_route": route,
            "route_reason": reason,
            "route_refreshed_at_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime()),
            "route_refresh_source": str(family_path),
        }
    )
    write_json(preflight_path, obj)
    print(preflight_path)


if __name__ == "__main__":
    main()
