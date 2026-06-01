#!/usr/bin/env python3
"""Summarize v12.15 Line D focused repair artifacts.

The raw Line D runner is the older v1283 classic-family runner.  This script
does not invent measurements; it only joins the measured v1283 CSV/JSON files
into v12.15-named summary artifacts for audit and replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["stage", "status"]
        materialized = [{"stage": path.stem.upper(), "status": "no_rows"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def by_candidate(rows: Sequence[Mapping[str, str]], stage: str | None = None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if stage is not None and row.get("stage") != stage:
            continue
        cid = str(row.get("candidate_id", "")).strip()
        if cid:
            out[cid] = dict(row)
    return out


def family_from_candidate(candidate_id: str) -> str:
    prefix = candidate_id.split("-", 1)[0]
    if prefix.startswith("B7"):
        return "Rational"
    if prefix.startswith("B2"):
        return "RBF"
    if prefix.startswith("B3"):
        return "Chebyshev"
    if prefix.startswith("B4"):
        return "Fourier"
    if prefix.startswith("B5"):
        return "Wavelet"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v12.15 Line D focused repair")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    failures = read_csv(raw_dir / "v1283_family_failure_table.csv")
    efficiency_l3 = {
        row.get("candidate_id", ""): row
        for row in read_csv(raw_dir / "v1283_family_efficiency.csv")
        if row.get("implementation_level") == "L3-analytic-manual-ce-torch-reduction"
    }
    expression = by_candidate(read_csv(raw_dir / "v1283_family_expression.csv"), "V1283_FAMILY_EXPRESSION_SUMMARY")
    for row in read_csv(raw_dir / "v1283_family_expression.csv"):
        cid = row.get("candidate_id", "")
        if row.get("stage") == "V1283_FAMILY_EXPRESSION" and cid and cid not in expression:
            expression[cid] = row
    task = by_candidate(read_csv(raw_dir / "v1283_family_task_triage.csv"), "V1283_FAMILY_TASK_SUMMARY")
    for row in read_csv(raw_dir / "v1283_family_task_triage.csv"):
        cid = row.get("candidate_id", "")
        if row.get("stage") == "V1283_FAMILY_TASK_TRIAGE" and cid and cid not in task:
            task[cid] = row
    linec = by_candidate(read_csv(raw_dir / "v1283_family_linec_summary.csv"))
    route = read_json(raw_dir / "v1283_route_decision.json")
    status = read_json(raw_dir / "v1283_family_status.json").get("families", {})
    provenance_rows = read_csv(raw_dir / "v1283_provenance_audit.csv")
    no_fake = int(all(str(row.get("no_fake_pass", "0")) == "1" for row in provenance_rows)) if provenance_rows else 0
    no_proxy = int(all(str(row.get("no_proxy_pass", "0")) == "1" for row in provenance_rows)) if provenance_rows else 0
    no_cpu = int(all(str(row.get("no_cpu_offload_pass", "0")) == "1" for row in provenance_rows)) if provenance_rows else 0

    candidate_rows: list[dict[str, Any]] = []
    for failure in failures:
        cid = str(failure.get("candidate_id", "")).strip()
        if not cid:
            continue
        eff = efficiency_l3.get(cid, {})
        expr = expression.get(cid, {})
        task_row = task.get(cid, {})
        linec_row = linec.get(cid, {})
        family = failure.get("family") or expr.get("family") or eff.get("family") or family_from_candidate(cid)
        candidate_rows.append(
            {
                "stage": "V1215_LINED_FOCUSED_REPAIR_CANDIDATE",
                "family": family,
                "candidate_id": cid,
                "l3_official_efficiency_pass": eff.get("official_efficiency_pass", ""),
                "l3_step_ratio_q90": eff.get("step_ratio_q90", ""),
                "l3_memory_ratio_q90": eff.get("memory_ratio_q90", ""),
                "A4_expression_pass": expr.get("A4_expression_pass", ""),
                "A5_task_pass": task_row.get("A5_task_pass", ""),
                "A5_mean_delta": task_row.get("mean_delta", ""),
                "A5_worst_delta": task_row.get("worst_delta", ""),
                "A5_near_pass_rate": task_row.get("near_pass_rate", ""),
                "LineC_CouplingR2": linec_row.get("candidate_CouplingR2", ""),
                "LineC_NoiseSignalLeak": linec_row.get("candidate_NoiseSignalLeak", ""),
                "LineC_RealSignalReservoirRatio": linec_row.get("candidate_RealSignalReservoirRatio", ""),
                "LineC_CouplingR2_delta_vs_mlp": linec_row.get("CouplingR2_delta_vs_mlp", ""),
                "LineC_NoiseSignalLeak_delta_vs_mlp": linec_row.get("NoiseSignalLeak_delta_vs_mlp", ""),
                "LineC_RealSignalReservoirRatio_delta_vs_mlp": linec_row.get("RealSignalReservoirRatio_delta_vs_mlp", ""),
                "LineC_interpretation": linec_row.get("linec_interpretation", ""),
                "failure_code": failure.get("failure_code", ""),
                "failure_reason": failure.get("reason", ""),
                "raw_dir": str(raw_dir),
                "no_fake_pass": no_fake,
                "no_proxy_pass": no_proxy,
                "no_cpu_offload_pass": no_cpu,
            }
        )

    family_rows: list[dict[str, Any]] = []
    for family, payload in sorted(status.items()):
        if family == "BSpline":
            focused_count = 0
        else:
            focused_count = sum(1 for row in candidate_rows if row.get("family") == family)
        family_rows.append(
            {
                "stage": "V1215_LINED_FOCUSED_REPAIR_FAMILY_STATUS",
                "family": family,
                "status": payload.get("status", ""),
                "focused_candidate_count": focused_count,
                "best_L3_manual_step_ratio": payload.get("best_L3_manual_step_ratio", ""),
                "A4_expression_pass_candidates": ";".join(payload.get("A4_expression_pass_candidates", []) or []),
                "A5_task_pass_candidates": ";".join(payload.get("A5_task_pass_candidates", []) or []),
                "blocker": payload.get("blocker", ""),
                "raw_dir": str(raw_dir),
            }
        )

    family_pass_count = sum(1 for row in family_rows if row.get("status") == "FamilyPass")
    route_payload = {
        "stage": "V1215_LINED_FOCUSED_REPAIR_ROUTE",
        "raw_runner_route": route.get("route", ""),
        "lineD_completed": 1,
        "lineD_focused_candidate_count": len(candidate_rows),
        "lineD_family_pass_count": family_pass_count,
        "lineD_family_linec_measured_count": route.get("family_linec_measured_count", ""),
        "lineD_family_linec_failed_count": route.get("family_linec_failed_count", ""),
        "lineD_all_family_linec_interpretation": route.get("family_linec_interpretations", ""),
        "lineD_no_fake_pass": no_fake,
        "lineD_no_proxy_pass": no_proxy,
        "lineD_no_cpu_offload_pass": no_cpu,
        "lineD_next_recommended_action": (
            "stop claiming current focused Line D candidates as functional/base repair; "
            "Rational/RBF/Fourier remain expression blocked, Chebyshev/Wavelet remain task blocked, "
            "and all measured family Line C summaries are coupling_collapse"
        ),
        "raw_dir": str(raw_dir),
    }

    candidate_path = out_dir / "v1215_lineD_focused_repair_candidates.csv"
    family_path = out_dir / "v1215_lineD_focused_repair_family_status.csv"
    route_path = out_dir / "v1215_lineD_focused_repair_route.json"
    write_csv(candidate_path, candidate_rows)
    write_csv(family_path, family_rows)
    write_json(route_path, route_payload)
    hash_payload = {
        path.name: sha256_file(path)
        for path in [candidate_path, family_path, route_path]
    }
    write_json(out_dir / "v1215_lineD_focused_repair_hash_manifest.json", hash_payload)


if __name__ == "__main__":
    main()
