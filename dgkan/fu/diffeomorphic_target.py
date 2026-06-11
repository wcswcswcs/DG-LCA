"""Audit helpers for v22.03 diffeomorphic source-channel target claims.

The metrics here are intentionally lightweight and artifact-facing. They are
used to audit train-stream target geometry; they do not generate labels or
post-hoc training directions.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_ratio(num: Any, den: Any) -> float | str:
    n = finite_float(num)
    d = finite_float(den)
    if not math.isfinite(n) or not math.isfinite(d) or abs(d) <= 1.0e-12:
        return ""
    return n / d


def source_channel_info_volume(row: dict[str, Any], *, step: int = 4000) -> dict[str, Any]:
    """Return source-channel audit ratios from a row of trace-enriched metrics."""

    hidden = finite_float(row.get(f"hidden_source_energy_h{step}", row.get("hidden_source_energy")))
    readout = finite_float(row.get(f"readout_source_energy_h{step}", row.get("readout_source_energy")))
    source_norm = finite_float(row.get(f"source_state_norm_h{step}", row.get("source_state_norm")))
    whitened = finite_float(row.get(f"source_state_whitened_norm_h{step}", row.get("source_state_whitened_norm")))
    rank = finite_float(row.get(f"source_subspace_rank_h{step}", row.get("source_subspace_rank")))
    projection = finite_float(row.get(f"source_channel_projection_h{step}", row.get("source_channel_projection")))
    total = sum(v for v in [hidden, readout] if math.isfinite(v))
    hidden_fraction = hidden / total if math.isfinite(hidden) and total > 0 else ""
    readout_fraction = readout / total if math.isfinite(readout) and total > 0 else ""
    condition_proxy = safe_ratio(source_norm, whitened)
    fold_proxy = 1.0 - max(-1.0, min(1.0, projection)) if math.isfinite(projection) else ""
    return {
        "info_volume_step": step,
        "source_hidden_fraction": hidden_fraction,
        "source_readout_fraction": readout_fraction,
        "source_norm_over_whitened_norm": condition_proxy,
        "source_subspace_rank": rank if math.isfinite(rank) else "",
        "source_projection_fold_proxy": fold_proxy,
    }


def summarize_info_volume(rows: Iterable[dict[str, Any]], *, step: int = 4000) -> dict[str, Any]:
    audits = [source_channel_info_volume(r, step=step) for r in rows]
    keys = [
        "source_hidden_fraction",
        "source_readout_fraction",
        "source_norm_over_whitened_norm",
        "source_subspace_rank",
        "source_projection_fold_proxy",
    ]
    out: dict[str, Any] = {"rows": len(audits), "info_volume_step": step}
    for key in keys:
        vals = [finite_float(a.get(key)) for a in audits]
        vals = [v for v in vals if math.isfinite(v)]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else ""
        out[f"{key}_min"] = min(vals) if vals else ""
        out[f"{key}_max"] = max(vals) if vals else ""
    fold = finite_float(out.get("source_projection_fold_proxy_mean"))
    condition = finite_float(out.get("source_norm_over_whitened_norm_mean"))
    out["diffeomorphic_audit_pass"] = int(
        (not math.isfinite(fold) or fold <= 1.25)
        and (not math.isfinite(condition) or condition <= 50.0)
    )
    return out


def diffeomorphic_target_unit_tests() -> list[dict[str, Any]]:
    row = {
        "hidden_source_energy_h4000": "3.0",
        "readout_source_energy_h4000": "1.0",
        "source_state_norm_h4000": "4.0",
        "source_state_whitened_norm_h4000": "2.0",
        "source_subspace_rank_h4000": "8",
        "source_channel_projection_h4000": "0.5",
    }
    audit = source_channel_info_volume(row, step=4000)
    return [
        {
            "test": "source_channel_info_volume",
            "hidden_fraction": audit["source_hidden_fraction"],
            "readout_fraction": audit["source_readout_fraction"],
            "condition_proxy": audit["source_norm_over_whitened_norm"],
            "fold_proxy": audit["source_projection_fold_proxy"],
            "pass": int(
                abs(float(audit["source_hidden_fraction"]) - 0.75) < 1.0e-9
                and abs(float(audit["source_projection_fold_proxy"]) - 0.5) < 1.0e-9
            ),
        }
    ]


__all__ = [
    "diffeomorphic_target_unit_tests",
    "finite_float",
    "safe_ratio",
    "source_channel_info_volume",
    "summarize_info_volume",
]

