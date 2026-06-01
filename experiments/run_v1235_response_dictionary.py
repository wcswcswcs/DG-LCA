#!/usr/bin/env python
"""Build the v12.35 basis response dictionary from executed basis-functional probes."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v1223_failclosed_explore_open2_functional_rebuild as exp  # noqa: E402


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def sint(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, Any]]:
    return exp.read_csv_rows(path) if path.exists() else []


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exp.write_csv_rows(path, rows)


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def response_pass(row: dict[str, Any]) -> int:
    return int(
        sint(row.get("executed"), 0) == 1
        and fnum(row.get("CouplingR2_delta"), -999.0) >= 0.01
        and fnum(row.get("NoiseSignalLeak_delta"), 999.0) <= 0.0
        and fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) <= 0.0
        and fnum(row.get("CEp99_delta_audit"), 999.0) <= 0.05
        and fnum(row.get("control_gap"), -999.0) >= 0.0
    )


def p3_pass(row: dict[str, Any]) -> int:
    return int(
        sint(row.get("executed"), 0) == 1
        and fnum(row.get("source_vs_best_control"), -999.0) >= 0.005
        and fnum(row.get("CouplingR2_delta"), -999.0) >= 0.02
        and fnum(row.get("NoiseSignalLeak_delta"), 999.0) <= -0.005
        and fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) <= -0.005
        and fnum(row.get("CEp99_delta_audit"), 999.0) <= 0.05
        and fnum(row.get("NLL_delta_audit"), 999.0) <= 0.02
        and fnum(row.get("ECE_delta_audit"), 999.0) <= 0.02
    )


def fail_reason(row: dict[str, Any], gate: str) -> str:
    reasons: list[str] = []
    if sint(row.get("executed"), 0) != 1:
        reasons.append(str(row.get("skip_reason", "")) or "not_executed")
    if gate in ("response", "p3"):
        if fnum(row.get("control_gap"), -999.0) < 0.0:
            reasons.append("control_gap_negative")
        if fnum(row.get("CouplingR2_delta"), -999.0) < (0.02 if gate == "p3" else 0.01):
            reasons.append("CouplingR2_no_gain")
        if fnum(row.get("NoiseSignalLeak_delta"), 999.0) > (-0.005 if gate == "p3" else 0.0):
            reasons.append("NoiseSignalLeak_no_drop")
        if fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) > (-0.005 if gate == "p3" else 0.0):
            reasons.append("RealSignalReservoirRatio_no_drop")
        if fnum(row.get("CEp99_delta_audit"), 999.0) > 0.05:
            reasons.append("CEp99_harm")
    if gate == "p3":
        if fnum(row.get("source_vs_best_control"), -999.0) < 0.005:
            reasons.append("source_vs_best_control_too_small")
        if fnum(row.get("NLL_delta_audit"), 999.0) > 0.02:
            reasons.append("NLL_harm")
        if fnum(row.get("ECE_delta_audit"), 999.0) > 0.02:
            reasons.append("ECE_harm")
    return ";".join(reasons) if reasons else "pass"


def build_dictionary_rows(p3_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in p3_rows:
        rr = {
            "stage": "V1235_BASIS_RESPONSE_DICTIONARY",
            "family": row.get("family", ""),
            "base_candidate_id": row.get("base_candidate_id", ""),
            "functional_candidate_id": row.get("functional_candidate_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "executed": sint(row.get("executed"), 0),
            "actuator_source": "v12342_basis_functional_probe",
            "source_vs_best_control": fnum(row.get("source_vs_best_control")),
            "control_gap": fnum(row.get("control_gap")),
            "CouplingR2_delta": fnum(row.get("CouplingR2_delta")),
            "NoiseSignalLeak_delta": fnum(row.get("NoiseSignalLeak_delta")),
            "RealSignalReservoirRatio_delta": fnum(row.get("RealSignalReservoirRatio_delta")),
            "AUC_time_delta_proxy": fnum(row.get("AUC_time_delta_proxy")),
            "CEp99_delta_audit": fnum(row.get("CEp99_delta_audit")),
            "NLL_delta_audit": fnum(row.get("NLL_delta_audit")),
            "ECE_delta_audit": fnum(row.get("ECE_delta_audit")),
            "LineC_pass_count": sint(row.get("LineC_pass_count"), 0),
            "LineC_total": sint(row.get("LineC_total"), 0),
            "response_dictionary_pass": response_pass(row),
            "response_fail_reason": fail_reason(row, "response"),
            "p3_gate_recomputed_pass": p3_pass(row),
            "p3_recomputed_fail_reason": fail_reason(row, "p3"),
            "label_used_for_direction": sint(row.get("label_used_for_direction"), 0),
            "ce_vector_used_for_direction": sint(row.get("ce_vector_used_for_direction"), 0),
            "validation_used_for_commit": sint(row.get("validation_used_for_commit"), 0),
            "query_batch_used_for_direction": sint(row.get("query_batch_used_for_direction"), 0),
            "linec_hard_target_used_for_direction": sint(row.get("linec_hard_target_used_for_direction"), 0),
            "dataset_name_branch": sint(row.get("dataset_name_branch"), 0),
            "promotion_allowed": 0,
            "no_fake": 1,
        }
        out.append(rr)
    return out


def build_controls_rows(p3_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = [
        ("NoOpMatchedOverhead", "source_vs_noop"),
        ("RandomMatchedNorm", "source_vs_random"),
        ("AdamWParallelDirection", "source_vs_adamwparallel"),
        ("SNR-only audit", "source_vs_snr"),
    ]
    out: list[dict[str, Any]] = []
    for row in p3_rows:
        for control, key in controls:
            out.append({
                "stage": "V1235_BASIS_RESPONSE_CONTROL",
                "family": row.get("family", ""),
                "base_candidate_id": row.get("base_candidate_id", ""),
                "functional_candidate_id": row.get("functional_candidate_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "control_id": control,
                "source_vs_control": fnum(row.get(key)),
                "executed": sint(row.get("executed"), 0),
                "control_direction_source": "matched_control_from_v12342_p3",
                "uses_label_or_ce_for_direction": int(control == "AdamWParallelDirection"),
                "promotion_allowed": 0,
                "no_fake": 1,
            })
    return out


def build_task_tail_rows(p3_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "V1235_BASIS_RESPONSE_TASK_TAIL",
            "family": row.get("family", ""),
            "base_candidate_id": row.get("base_candidate_id", ""),
            "functional_candidate_id": row.get("functional_candidate_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "executed": sint(row.get("executed"), 0),
            "source_vs_best_control": fnum(row.get("source_vs_best_control")),
            "AUC_time_delta_proxy": fnum(row.get("AUC_time_delta_proxy")),
            "CEp99_delta_audit": fnum(row.get("CEp99_delta_audit")),
            "NLL_delta_audit": fnum(row.get("NLL_delta_audit")),
            "ECE_delta_audit": fnum(row.get("ECE_delta_audit")),
            "promotion_allowed": 0,
            "no_fake": 1,
        }
        for row in p3_rows
    ]


def build_linec_rows(linec_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in linec_rows:
        rr = dict(row)
        rr["stage"] = "V1235_BASIS_RESPONSE_LINEC"
        rr["promotion_allowed"] = 0
        rr["no_fake"] = 1
        out.append(rr)
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("family", "")), str(row.get("base_candidate_id", "")), str(row.get("functional_candidate_id", ""))), []).append(row)
    out: list[dict[str, Any]] = []
    for (family, base, functional), group in sorted(groups.items()):
        executed = [r for r in group if sint(r.get("executed"), 0) == 1]
        out.append({
            "stage": "V1235_RESPONSE_COLOCATION_SUMMARY",
            "family": family,
            "base_candidate_id": base,
            "functional_candidate_id": functional,
            "rows": len(group),
            "executed_rows": len(executed),
            "response_dictionary_pass_rows": sum(sint(r.get("response_dictionary_pass"), 0) for r in group),
            "p3_gate_recomputed_pass_rows": sum(sint(r.get("p3_gate_recomputed_pass"), 0) for r in group),
            "mean_source_vs_best_control": finite_mean([fnum(r.get("source_vs_best_control")) for r in executed]),
            "max_source_vs_best_control": max((fnum(r.get("source_vs_best_control")) for r in executed), default=float("nan")),
            "mean_control_gap": finite_mean([fnum(r.get("control_gap")) for r in executed]),
            "max_CouplingR2_delta": max((fnum(r.get("CouplingR2_delta")) for r in executed), default=float("nan")),
            "min_NoiseSignalLeak_delta": min((fnum(r.get("NoiseSignalLeak_delta")) for r in executed), default=float("nan")),
            "min_RealSignalReservoirRatio_delta": min((fnum(r.get("RealSignalReservoirRatio_delta")) for r in executed), default=float("nan")),
            "max_CEp99_delta_audit": max((fnum(r.get("CEp99_delta_audit")) for r in executed), default=float("nan")),
            "LineC_pass_count": sum(sint(r.get("LineC_pass_count"), 0) for r in executed),
            "LineC_total": sum(sint(r.get("LineC_total"), 0) for r in executed),
            "usable_response_dictionary": int(sum(sint(r.get("response_dictionary_pass"), 0) for r in group) > 0),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v1235")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    exp.ensure_dir(out_dir)

    p3_rows = read_rows(source_dir / "v12342_basis_functional_p3.csv")
    p3_rows.extend(read_rows(source_dir / "v12342_basis_functional_nonrat_foreachoff_p3.csv"))
    linec_rows = read_rows(source_dir / "v12342_basis_functional_linec.csv")
    linec_rows.extend(read_rows(source_dir / "v12342_basis_functional_nonrat_foreachoff_linec.csv"))

    dictionary = build_dictionary_rows(p3_rows)
    controls = build_controls_rows(p3_rows)
    task_tail = build_task_tail_rows(p3_rows)
    response_linec = build_linec_rows(linec_rows)
    summary = summarize(dictionary)

    prefix = args.artifact_prefix
    write_rows(out_dir / f"{prefix}_basis_response_dictionary.csv", dictionary)
    write_rows(out_dir / f"{prefix}_basis_response_controls.csv", controls)
    write_rows(out_dir / f"{prefix}_basis_response_linec.csv", response_linec)
    write_rows(out_dir / f"{prefix}_basis_response_task_tail.csv", task_tail)
    write_rows(out_dir / f"{prefix}_response_colocation_summary.csv", summary)

    result = {
        "stage": "V1235_RESPONSE_DICTIONARY_ROUTE",
        "basis_response_dictionary_csv": exp.rel(out_dir / f"{prefix}_basis_response_dictionary.csv"),
        "basis_response_controls_csv": exp.rel(out_dir / f"{prefix}_basis_response_controls.csv"),
        "basis_response_linec_csv": exp.rel(out_dir / f"{prefix}_basis_response_linec.csv"),
        "basis_response_task_tail_csv": exp.rel(out_dir / f"{prefix}_basis_response_task_tail.csv"),
        "response_colocation_summary_csv": exp.rel(out_dir / f"{prefix}_response_colocation_summary.csv"),
        "dictionary_rows": len(dictionary),
        "dictionary_executed_rows": sum(sint(r.get("executed"), 0) for r in dictionary),
        "response_dictionary_pass_rows": sum(sint(r.get("response_dictionary_pass"), 0) for r in dictionary),
        "p3_gate_recomputed_pass_rows": sum(sint(r.get("p3_gate_recomputed_pass"), 0) for r in dictionary),
        "summary_rows": len(summary),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(out_dir / f"{prefix}_response_dictionary_route.json", result)
    print(result)
    return result


if __name__ == "__main__":
    run()
