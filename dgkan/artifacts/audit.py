"""Audit helpers for no-fake/no-proxy v9 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

from .writer import read_csv_rows


def _intish(value: object) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return 0


def audit_no_fake(paths: Sequence[Path]) -> Dict[str, object]:
    rows_checked = 0
    fake_data = 0
    proxy = 0
    offload = 0
    for path in paths:
        for row in read_csv_rows(path):
            rows_checked += 1
            fake_data += int(_intish(row.get("fake_data_used", 0)) != 0)
            proxy += int(_intish(row.get("proxy_row_used", 0)) != 0)
            offload += int(_intish(row.get("cpu_offload_used", 0)) != 0)
    return {
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_data + proxy,
        "fake_data_used": fake_data,
        "proxy_row_used": proxy,
        "cpu_offload_used": offload,
        "no_fake": fake_data == 0,
        "no_proxy": proxy == 0,
    }
