#!/usr/bin/env python3
"""Summarize v12.23 actuator rows for continuation audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def release_score(row: Mapping[str, Any]) -> float:
    return -safe_float(row.get("NoiseSignalLeak_delta_audit"), 0.0) - safe_float(row.get("RealSignalReservoirRatio_delta_audit"), 0.0)


def control_gap(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("control_gap"), float("nan"))


def summarize(rows: Sequence[Mapping[str, Any]], actuator_id: str) -> dict[str, Any]:
    group = [r for r in rows if str(r.get("actuator_id", "")) == actuator_id]
    gaps = [control_gap(r) for r in group if math.isfinite(control_gap(r))]
    drift = [safe_float(r.get("logit_max_abs_drift")) for r in group]
    drift = [v for v in drift if math.isfinite(v)]
    safe_group = [r for r in group if safe_float(r.get("logit_max_abs_drift"), 999.0) <= 0.05]
    return {
        "actuator_id": actuator_id,
        "rows": len(group),
        "safe_rows": sum(safe_int(r.get("role_safe_movement_pass")) for r in group),
        "release_rows": sum(safe_int(r.get("release_audit_pass")) for r in group),
        "safe_release_rows": sum(safe_int(r.get("role_safe_movement_pass")) and safe_int(r.get("release_audit_pass")) for r in group),
        "exploratory_rows": sum(safe_int(r.get("exploratory_release_gate")) for r in group),
        "best_control_gap": max(gaps) if gaps else float("nan"),
        "min_drift": min(drift) if drift else float("nan"),
        "best_by_release_score": max(group, key=release_score, default={}),
        "best_safe_by_release_score": max(safe_group, key=release_score, default={}),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            flat = {}
            for key in fields:
                value = row.get(key, "")
                flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            writer.writerow(flat)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    rows = read_rows(out_dir / "v1223_actuator_safety_roleaware.csv")
    aids = [x.strip() for x in str(args.actuator_ids).split(",") if x.strip()]
    summaries = [summarize(rows, aid) for aid in aids]
    payload = {
        "stage": "V1223_ACTUATOR_CONTINUATION_AUDIT",
        "source_csv": str(out_dir / "v1223_actuator_safety_roleaware.csv"),
        "actuator_ids": aids,
        "summaries": summaries,
    }
    if args.artifact_prefix:
        (out_dir / f"{args.artifact_prefix}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        write_csv(out_dir / f"{args.artifact_prefix}.csv", summaries)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--actuator-ids", required=True)
    parser.add_argument("--artifact-prefix", default="")
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
