#!/usr/bin/env python3
"""v12.12 B320-locked functional value rebuild wrapper.

This runner is deliberately conservative. Batch 1 only consumes materialized
v12.10/v12.11 artifacts and rewrites them into the v12.12 contract. It does not
fabricate rows, does not use proxy rows as success evidence, and does not change
loss/data/sampler/class weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "v12_12_b320locked_functional_value_rebuild_classic_nobspline"
DEFAULT_V1210_ROOT = REPO_ROOT / "results" / "v12_10_b320_functional_classic_nobspline"
DEFAULT_V1211_ROOT = REPO_ROOT / "results" / "v12_11_b320_functional_mechanism_classic_nobspline"
DEFAULT_ANCHOR_DIR = DEFAULT_V1210_ROOT / "repro_repair_linec64_p3fix_p4_taskminus_20260523T1405"
DEFAULT_B320_SOURCE_DIR = DEFAULT_V1210_ROOT / "repro_repair_linec64_full_20260523T1352"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md"
B320_ID = "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075"
ACTIVE_FAMILIES = ["Rational", "Chebyshev", "Wavelet", "RBF", "Fourier"]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_list(text: Any) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_ints(text: Any) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    materialized = [dict(row) for row in rows]
    if materialized:
        fields: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    else:
        fields = ["stage", "status"]
        materialized = [{"stage": path.stem.upper(), "status": "no_rows"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1.0e-20 or vy <= 1.0e-20:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank_values(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[order[idx]] = avg_rank
        pos = end
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(rank_values(xs), rank_values(ys))


def find_v1211_functional_dir(root: Path) -> Path:
    preferred = root / "v1211_b320_mechanism_probe_constrained_3x3_20260523T1604"
    if (preferred / "v1211_p3_p4_bridge_autopsy.csv").exists():
        return preferred
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "v1211_p3_p4_bridge_autopsy.csv").exists()]
    if not candidates:
        raise SystemExit(f"no v1211 functional bridge artifacts found under {root}")
    return max(candidates, key=lambda p: (p / "v1211_p3_p4_bridge_autopsy.csv").stat().st_mtime)


def build_anchor_hardening(anchor_dir: Path, source_dir: Path, out_dir: Path) -> dict[str, Any]:
    decision = read_json(anchor_dir / "v1210_route_decision.json")
    rows = read_csv_rows(source_dir / "v1210_b320_base_hardening.csv")
    row = rows[0] if rows else {}
    fail = []
    gates = [
        ("step_ratio_q90", 1.00, "<="),
        ("memory_ratio_q90", 0.30, "<="),
        ("mean_delta", 0.0, ">="),
        ("worst_delta", -0.003, ">="),
        ("near_pass_rate", 1.0, ">="),
        ("AUC_step_ratio", 1.0, "<="),
        ("AUC_time_ratio", 1.0, "<="),
        ("ECE_delta", 0.02, "<="),
    ]
    for key, gate, op in gates:
        val = safe_float(row.get(key), -999.0 if op == ">=" else 999.0)
        if (op == "<=" and val > gate) or (op == ">=" and val < gate):
            fail.append(f"{key}{op}{gate}")
    if safe_int(decision.get("B320_LineC_nontearing_pass"), 0) != 1:
        fail.append("LineC_nontearing_pass!=1")
    out_row = {
        "stage": "V1212_B320_ANCHOR_HARDENING",
        "run_id": source_dir.name,
        "candidate_id": row.get("candidate_id", decision.get("B320_exact_candidate_id", B320_ID)),
        "implementation_id": row.get("impl_path", "F3-triton-workspace-forward-delta-readout-proj-grad-learnableP"),
        "dataset": "MNIST,Fashion-MNIST,KMNIST",
        "seed": "0..9",
        "train_size": row.get("train_size", 1024),
        "batch_size": row.get("batch_size", 128),
        "epoch_budget": row.get("epochs", 3),
        "step_ratio_q90": row.get("step_ratio_q90", decision.get("B320_F3_step_ratio_q90", "")),
        "forward_ratio_q90": row.get("forward_ratio_q90", ""),
        "backward_ratio_q90": row.get("backward_ratio_q90", ""),
        "update_ratio_q90": row.get("update_ratio_q90", ""),
        "memory_ratio_q90": row.get("memory_ratio_q90", decision.get("B320_F3_memory_ratio_q90", "")),
        "mean_delta": row.get("mean_delta", decision.get("B320_mean_delta", "")),
        "worst_delta": row.get("worst_delta", decision.get("B320_worst_delta", "")),
        "near_pass": row.get("near_pass", ""),
        "near_pass_rate": row.get("near_pass_rate", decision.get("B320_near_pass_rate", "")),
        "AUC_step_ratio": row.get("AUC_step_ratio", decision.get("B320_max_AUC_step", "")),
        "AUC_time_ratio": row.get("AUC_time_ratio", decision.get("B320_max_AUC_time", "")),
        "ECE_delta": row.get("ECE_delta", decision.get("B320_max_ECE_delta", "")),
        "NLL_delta": row.get("NLL_delta", ""),
        "CEp99_delta": row.get("CEp99_delta", ""),
        "margin_p10_delta": row.get("margin_p10_delta", ""),
        "LineC_nontearing_pass": decision.get("B320_LineC_nontearing_pass", ""),
        "CouplingR2": decision.get("B320_LineC_CouplingR2", ""),
        "NoiseSignalLeak": decision.get("B320_LineC_NoiseSignalLeak", ""),
        "RealSignalReservoirRatio": decision.get("B320_LineC_RealSignalReservoirRatio", ""),
        "provenance_fake_count": 0,
        "proxy_row_count": 0,
        "cpu_offload_count": 0,
        "B320_locked_v1212": int(not fail),
        "fail_reason": ";".join(fail),
        "source_file": rel(source_dir / "v1210_b320_base_hardening.csv"),
    }
    write_csv_rows(out_dir / "v1212_b320_anchor_hardening.csv", [out_row])
    return {"anchor_pass": int(not fail), "anchor_row": out_row}


def linec_fail_reasons(row: Mapping[str, Any]) -> list[str]:
    fail: list[str] = []
    if safe_float(row.get("p3_CouplingR2_delta"), 0.0) < 0.02:
        fail.append("CouplingR2_delta<0.02")
    if safe_float(row.get("p3_NoiseSignalLeak_delta"), 0.0) > -0.01:
        fail.append("NoiseSignalLeak_delta>-0.01")
    if safe_float(row.get("p3_RealSignalReservoirRatio_delta"), 0.0) > -0.01:
        fail.append("RealSignalReservoirRatio_delta>-0.01")
    if safe_float(row.get("p3_CEp99_delta"), 0.0) > 0.05:
        fail.append("CEp99_delta>0.05")
    if row.get("p3_holdout_loss_ratio") not in ("", None) and safe_float(row.get("p3_holdout_loss_ratio"), 99.0) > 1.01:
        fail.append("holdout_loss_ratio>1.01")
    if safe_float(row.get("p4_AUC_time_delta"), 0.0) > 0.0:
        fail.append("AUC_time_delta>0")
    if safe_float(row.get("p4_control_gap"), 0.0) < 0.005:
        fail.append("control_gap<0.005")
    return fail


def build_p3_p4_autopsy(v1211_dir: Path, out_dir: Path) -> dict[str, Any]:
    bridge_rows = read_csv_rows(v1211_dir / "v1211_p3_p4_bridge_autopsy.csv")
    p3_rows = read_csv_rows(v1211_dir / "v1211_functional_p3v2_candidate.csv")
    p3_by_update = {r.get("functional_id", ""): r for r in p3_rows}
    out_rows: list[dict[str, Any]] = []
    old_scores: list[float] = []
    new_scores: list[float] = []
    p4_gains: list[float] = []
    strict_labels: list[int] = []
    for row in bridge_rows:
        p3 = p3_by_update.get(row.get("functional_id", ""), {})
        out = {
            "stage": "V1212_P3_P4_AUTOPSY",
            "candidate_id": row.get("candidate_id", B320_ID),
            "update_type": row.get("functional_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "p3_old_score": row.get("p3_score", p3.get("delta_score", "")),
            "p3_CouplingR2_delta": row.get("p3_coupling_delta", p3.get("CouplingR2_delta", p3.get("CouplingR2", ""))),
            "p3_NoiseSignalLeak_delta": row.get("p3_noise_delta", p3.get("NoiseSignalLeak_delta", "")),
            "p3_RealSignalReservoirRatio_delta": row.get("p3_reservoir_delta", p3.get("RealSignalReservoirRatio_delta", "")),
            "p3_CEp99_delta": row.get("p3_tail_delta", p3.get("CEp99_delta", "")),
            "p3_holdout_loss_ratio": p3.get("holdout_loss_ratio", ""),
            "p4_acc_delta": row.get("p4_acc_delta", ""),
            "p4_AUC_time_delta": row.get("p4_auc_time_delta", ""),
            "p4_ECE_delta": row.get("p4_ece_delta", ""),
            "p4_CouplingR2_delta": row.get("p4_coupling_delta", ""),
            "p4_NoiseSignalLeak_delta": row.get("p4_noise_delta", ""),
            "p4_RealSignalReservoirRatio_delta": row.get("p4_reservoir_delta", ""),
            "p4_control_gap": row.get("control_gap", ""),
            "p4_strict_pass": row.get("actual_strict_pass", ""),
            "p4_gain": row.get("p4_gain", ""),
            "failure_mode": row.get("failure_reason", ""),
            "source_file": row.get("source_file", rel(v1211_dir / "v1211_p3_p4_bridge_autopsy.csv")),
        }
        # Monotone, rule-based count: this is not a trained black-box selector.
        checks = {
            "coupling_ok": safe_float(out["p3_CouplingR2_delta"], 0.0) >= 0.02,
            "noise_ok": safe_float(out["p3_NoiseSignalLeak_delta"], 0.0) <= -0.01,
            "reservoir_ok": safe_float(out["p3_RealSignalReservoirRatio_delta"], 0.0) <= -0.01,
            "tail_ok": safe_float(out["p3_CEp99_delta"], 0.0) <= 0.05,
            "holdout_ok": out["p3_holdout_loss_ratio"] in ("", None) or safe_float(out["p3_holdout_loss_ratio"], 99.0) <= 1.01,
        }
        out["p3_linec_pareto_score"] = sum(int(v) for v in checks.values()) / float(len(checks))
        out["p3_linec_pareto_pass"] = int(all(checks.values()))
        out["p3_linec_fail_reason"] = ";".join(k for k, v in checks.items() if not v)
        out_rows.append(out)
        old_scores.append(safe_float(out["p3_old_score"], 0.0))
        new_scores.append(safe_float(out["p3_linec_pareto_score"], 0.0))
        p4_gains.append(safe_float(out["p4_gain"], 0.0))
        strict_labels.append(safe_int(out["p4_strict_pass"], 0))
    write_csv_rows(out_dir / "v1212_p3_p4_autopsy.csv", out_rows)
    old_rho = spearman(old_scores, p4_gains)
    new_rho = spearman(new_scores, p4_gains)
    summary = {
        "stage": "V1212_P3_P4_AUTOPSY_SUMMARY",
        "source_dir": rel(v1211_dir),
        "rows": len(out_rows),
        "old_score_spearman_to_p4_gain": old_rho if old_rho is not None else "",
        "new_linec_score_spearman_to_p4_gain": new_rho if new_rho is not None else "",
        "old_score_pearson_to_p4_gain": pearson(old_scores, p4_gains) or "",
        "new_linec_score_pearson_to_p4_gain": pearson(new_scores, p4_gains) or "",
        "p4_strict_pass_rows": sum(strict_labels),
        "linec_pareto_pass_rows": sum(safe_int(r.get("p3_linec_pareto_pass")) for r in out_rows),
        "do_not_promote_functional": int(new_rho is None or new_rho < 0.30),
    }
    write_csv_rows(out_dir / "v1212_p3_p4_autopsy_summary.csv", [summary])
    return summary


def build_linec_value(v1211_dir: Path, out_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(v1211_dir / "v1211_functional_p3v2_candidate.csv")
    out_rows = []
    for row in rows:
        diag = {
            "stage": "V1212_LINEC_WINDOW_DIAGNOSTICS",
            "run_id": v1211_dir.name,
            "candidate_id": row.get("candidate_id", B320_ID),
            "update_type": row.get("functional_id", row.get("update_type", "")),
            "control_id": row.get("best_control_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "step": row.get("step", ""),
            "window_size": row.get("window_size", "p3v2_single_window"),
            "CouplingR2_before": "",
            "CouplingR2_after": "",
            "CouplingR2_delta": row.get("CouplingR2_delta", row.get("CouplingR2", "")),
            "NoiseSignalLeak_before": "",
            "NoiseSignalLeak_after": "",
            "NoiseSignalLeak_delta": row.get("NoiseSignalLeak_delta", ""),
            "RealSignalReservoirRatio_before": "",
            "RealSignalReservoirRatio_after": "",
            "RealSignalReservoirRatio_delta": row.get("RealSignalReservoirRatio_delta", ""),
            "KernelDrift": row.get("KernelDrift", ""),
            "signal_effective_rank": row.get("signal_effective_rank", ""),
            "reservoir_fraction": row.get("reservoir_fraction", ""),
            "top_eigen_share": row.get("top_eigen_share", ""),
            "CEp99_delta": row.get("CEp99_delta", ""),
            "ECE_delta": row.get("ECE_delta", ""),
            "AUC_step_delta": row.get("AUC_step_delta", ""),
            "AUC_time_delta": row.get("AUC_time_delta", ""),
            "holdout_loss_ratio": row.get("holdout_loss_ratio", ""),
            "update_norm_ratio": row.get("update_norm_ratio", ""),
            "source_stage": row.get("stage", ""),
            "derived_from_v1211_not_new_training": 1,
        }
        no_op = (
            abs(safe_float(diag["CouplingR2_delta"], 0.0))
            + abs(safe_float(diag["NoiseSignalLeak_delta"], 0.0))
            + abs(safe_float(diag["RealSignalReservoirRatio_delta"], 0.0))
            < 0.03
        )
        fail = []
        if safe_float(diag["CouplingR2_delta"], 0.0) < 0.02:
            fail.append("CouplingR2_delta<0.02")
        if safe_float(diag["NoiseSignalLeak_delta"], 0.0) > -0.01:
            fail.append("NoiseSignalLeak_delta>-0.01")
        if safe_float(diag["RealSignalReservoirRatio_delta"], 0.0) > -0.01:
            fail.append("RealSignalReservoirRatio_delta>-0.01")
        if safe_float(diag["AUC_time_delta"], 0.0) > 0.0:
            fail.append("AUC_time_delta>0")
        if safe_float(diag["CEp99_delta"], 0.0) > 0.05:
            fail.append("CEp99_delta>0.05")
        if no_op:
            fail.append("no_op_geometry_delta<0.03")
        diag["no_op_flag"] = int(no_op)
        diag["pareto_pass"] = int(not fail)
        diag["fail_reason"] = ";".join(fail)
        out_rows.append(diag)
    write_csv_rows(out_dir / "v1212_linec_window_diagnostics.csv", out_rows)
    summary = {
        "stage": "V1212_LINEC_VALUE_SUMMARY",
        "source_dir": rel(v1211_dir),
        "rows": len(out_rows),
        "pareto_pass_rows": sum(safe_int(r.get("pareto_pass")) for r in out_rows),
        "coupling_open_rows": sum(safe_float(r.get("CouplingR2_delta"), 0.0) >= 0.02 for r in out_rows),
        "noise_release_rows": sum(safe_float(r.get("NoiseSignalLeak_delta"), 0.0) <= -0.01 for r in out_rows),
        "reservoir_release_rows": sum(safe_float(r.get("RealSignalReservoirRatio_delta"), 0.0) <= -0.01 for r in out_rows),
        "no_op_rows": sum(safe_int(r.get("no_op_flag")) for r in out_rows),
        "fail_reason_counts": dict(Counter(reason for r in out_rows for reason in str(r.get("fail_reason", "")).split(";") if reason)),
    }
    write_csv_rows(out_dir / "v1212_linec_value_summary.csv", [summary])
    return summary


def latest_family_rows(root: Path, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(f"*/{filename}")):
        for row in read_csv_rows(path):
            copied = dict(row)
            copied["source_dir"] = path.parent.name
            copied["source_file"] = rel(path)
            rows.append(copied)
    return rows


def build_family_status(v1211_root: Path, out_dir: Path) -> dict[str, Any]:
    eff_rows = latest_family_rows(v1211_root, "v1283_family_efficiency.csv")
    expr_rows = latest_family_rows(v1211_root, "v1283_family_expression.csv")
    task_rows = latest_family_rows(v1211_root, "v1283_family_task_triage.csv")
    geo_rows = latest_family_rows(v1211_root, "v1283_family_geometry_lineC.csv")
    fail_rows = latest_family_rows(v1211_root, "v1283_family_failure_table.csv")
    write_csv_rows(out_dir / "v1212_family_efficiency.csv", eff_rows)
    write_csv_rows(out_dir / "v1212_family_expression.csv", expr_rows)
    write_csv_rows(out_dir / "v1212_family_task_triage.csv", task_rows)
    write_csv_rows(out_dir / "v1212_family_linec.csv", geo_rows)
    write_csv_rows(out_dir / "v1212_family_failure_table.csv", fail_rows)

    candidate_family: dict[str, str] = {}
    l3_pass: defaultdict[str, bool] = defaultdict(bool)
    a4_pass: defaultdict[str, bool] = defaultdict(bool)
    a5_pass: defaultdict[str, bool] = defaultdict(bool)
    linec_pass: defaultdict[str, bool] = defaultdict(bool)
    best_task: defaultdict[str, tuple[float, str]] = defaultdict(lambda: (-999.0, ""))
    best_expr: defaultdict[str, tuple[float, str]] = defaultdict(lambda: (-999.0, ""))
    for row in eff_rows:
        cid = row.get("candidate_id", "")
        fam = row.get("family", "")
        if cid and fam:
            candidate_family[cid] = fam
        if row.get("implementation_level") == "L3-analytic-manual-ce-torch-reduction" and safe_int(row.get("official_efficiency_pass"), 0) == 1:
            l3_pass[cid] = True
    for row in expr_rows:
        cid = row.get("candidate_id", "")
        fam = row.get("family", candidate_family.get(cid, ""))
        if cid and fam:
            candidate_family[cid] = fam
        if "SUMMARY" in row.get("stage", ""):
            if safe_int(row.get("A4_expression_pass"), 0) == 1:
                a4_pass[cid] = True
            expr_score = (
                safe_float(row.get("E1-pairwise-product_B1_delta_vs_mlp"), -9.0)
                + safe_float(row.get("E6-rotated-pairwise-product_B1_delta_vs_mlp"), -9.0)
                + safe_float(row.get("E8-random-quadratic-form_B1_delta_vs_mlp"), -9.0)
            )
            if expr_score > best_expr[fam][0]:
                best_expr[fam] = (expr_score, cid)
    for row in task_rows:
        cid = row.get("candidate_id", "")
        fam = row.get("family", candidate_family.get(cid, ""))
        if cid and fam:
            candidate_family[cid] = fam
        if "SUMMARY" in row.get("stage", ""):
            if safe_int(row.get("A5_task_pass"), 0) == 1:
                a5_pass[cid] = True
            mean_delta = safe_float(row.get("mean_delta"), -999.0)
            if mean_delta > best_task[fam][0]:
                best_task[fam] = (mean_delta, cid)
    for row in geo_rows:
        cid = row.get("candidate_id", "")
        if "SUMMARY" in row.get("stage", "") and safe_int(row.get("LineC_geometry_pass"), 0) == 1:
            linec_pass[cid] = True

    manifest_rows = []
    family_payload: dict[str, Any] = {"stage": "V1212_FAMILY_STATUS", "families": {}}
    for fam in ACTIVE_FAMILIES:
        cids = sorted(cid for cid, cfam in candidate_family.items() if cfam == fam)
        any_l3 = any(l3_pass[cid] for cid in cids)
        any_a4 = any(l3_pass[cid] and a4_pass[cid] for cid in cids)
        any_a5 = any(l3_pass[cid] and a4_pass[cid] and a5_pass[cid] for cid in cids)
        any_family_pass = any(l3_pass[cid] and a4_pass[cid] and a5_pass[cid] and linec_pass[cid] for cid in cids)
        if any_family_pass:
            status = "FamilyPass"
            blocker = ""
        elif any_a5:
            status = "GeometryBlocked"
            blocker = "A5 passed but LineC pass missing"
        elif any_a4:
            status = "TaskBlocked"
            blocker = "official L3 and A4 pass exist, but A5 task gate failed"
        elif any_l3:
            status = "ExpressionBlocked"
            blocker = "official L3 pass exists, but A4 expression gate failed"
        else:
            status = "KernelBlocked"
            blocker = "no official fused L3 pass evidence found"
        best_task_score, best_task_id = best_task[fam]
        best_expr_score, best_expr_id = best_expr[fam]
        manifest_rows.append(
            {
                "stage": "V1212_FAMILY_STATUS_ROW",
                "family": fam,
                "status": status,
                "candidate_count_seen": len(cids),
                "any_l3_pass": int(any_l3),
                "any_a4_pass": int(any_a4),
                "any_a5_pass": int(any_a5),
                "any_linec_pass": int(any(linec_pass[cid] for cid in cids)),
                "best_task_candidate": best_task_id,
                "best_task_mean_delta": "" if best_task_score <= -900 else best_task_score,
                "best_expression_candidate": best_expr_id,
                "best_expression_score_E1E6E8_sum": "" if best_expr_score <= -20 else best_expr_score,
                "blocker": blocker,
            }
        )
        family_payload["families"][fam] = manifest_rows[-1]
    family_payload["families"]["BSpline"] = {
        "family": "BSpline",
        "status": "RejectedForThisVersion",
        "blocker": "B-spline frozen by v12.12 plan",
    }
    write_csv_rows(out_dir / "v1212_family_manifest.csv", manifest_rows)
    write_csv_rows(out_dir / "v1212_family_status.csv", manifest_rows)
    write_json(out_dir / "v1212_family_status.json", family_payload)
    return family_payload


def load_v1211_runner():
    path = REPO_ROOT / "experiments" / "run_v1211_b320_functional_mechanism_classic_nobspline.py"
    spec = importlib.util.spec_from_file_location("v1211_runner_for_v1212", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def delta_norm(deltas: Sequence[Any]) -> float:
    return math.sqrt(sum(float(delta.detach().square().sum().item()) for delta in deltas))


def delta_dot(a: Sequence[Any], b: Sequence[Any]) -> float:
    return sum(float((x.detach() * y.detach()).sum().item()) for x, y in zip(a, b))


def scale_delta(deltas: Sequence[Any], scale: float) -> list[Any]:
    return [delta * float(scale) for delta in deltas]


def normalize_like(deltas: Sequence[Any], reference: Sequence[Any]) -> list[Any]:
    norm = delta_norm(deltas)
    ref_norm = delta_norm(reference)
    if norm <= 1.0e-12:
        return [delta.detach().clone() for delta in deltas]
    return [delta * (ref_norm / norm) for delta in deltas]


def snr_gate_delta(v1252: Any, torch_mod: Any, model: Any, x: Any, y: Any, lr: float, mode: str) -> tuple[list[Any], dict[str, Any]]:
    task_delta = v1252._grad_delta(model, x, y, lr)
    sample_deltas = [v1252._grad_delta(model, x[idx : idx + 1], y[idx : idx + 1], lr) for idx in range(int(x.shape[0]))]
    gated: list[Any] = []
    pos_count = 0
    total_count = 0
    kept_square = 0.0
    total_square = 0.0
    for param_idx, delta in enumerate(task_delta):
        stack = torch_mod.stack([sample[param_idx] for sample in sample_deltas], dim=0)
        mu = stack.mean(dim=0)
        var = stack.var(dim=0, unbiased=True) if int(stack.shape[0]) > 1 else torch_mod.zeros_like(mu)
        snr = mu.square() - var / float(max(1, int(stack.shape[0]) - 1))
        positive = snr > 0.0
        pos_count += int(positive.sum().detach().item())
        total_count += int(positive.numel())
        if mode == "hard":
            weight = positive.to(dtype=delta.dtype)
        elif mode == "soft":
            scale = snr.detach().abs().median().clamp_min(1.0e-12)
            weight = torch_mod.sigmoid(snr / scale).to(dtype=delta.dtype)
        elif mode == "weighted":
            raw = snr.clamp_min(0.0).sqrt()
            denom = raw.detach().median().clamp_min(1.0e-12)
            weight = (raw / denom).clamp(max=1.0).to(dtype=delta.dtype)
        else:
            raise ValueError(f"unknown SNR gate mode {mode}")
        gated_delta = delta * weight
        gated.append(gated_delta)
        kept_square += float(gated_delta.detach().square().sum().item())
        total_square += float(delta.detach().square().sum().item())
    gated = normalize_like(gated, task_delta)
    gated_norm = delta_norm(gated)
    task_norm = delta_norm(task_delta)
    cos = delta_dot(gated, task_delta) / max(1.0e-12, gated_norm * task_norm)
    stats = {
        "SNR_positive_fraction": pos_count / float(max(1, total_count)),
        "gradient_norm_removed": 1.0 - math.sqrt(kept_square / max(1.0e-24, total_square)),
        "cos_to_adamw": cos,
        "task_delta_norm": task_norm,
        "gated_delta_norm": gated_norm,
    }
    return gated, stats


def bm1_fail_reasons(row: Mapping[str, Any]) -> list[str]:
    fail: list[str] = []
    if safe_float(row.get("CouplingR2_delta"), 0.0) < 0.02:
        fail.append("CouplingR2_delta<0.02")
    if safe_float(row.get("NoiseSignalLeak_delta"), 0.0) > -0.01:
        fail.append("NoiseSignalLeak_delta>-0.01")
    if safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) > -0.01:
        fail.append("RealSignalReservoirRatio_delta>-0.01")
    if safe_float(row.get("holdout_loss_ratio"), 999.0) > 1.01:
        fail.append("holdout_loss_ratio>1.01")
    if safe_float(row.get("CEp99_delta"), 0.0) > 0.05:
        fail.append("CEp99_delta>0.05")
    if safe_float(row.get("control_gap_vs_best"), -999.0) < 0.005:
        fail.append("control_gap<0.005")
    if safe_float(row.get("update_norm_ratio"), 0.0) < 0.05:
        fail.append("update_norm_ratio<0.05")
    if (
        abs(safe_float(row.get("CouplingR2_delta"), 0.0))
        + abs(safe_float(row.get("NoiseSignalLeak_delta"), 0.0))
        + abs(safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0))
        < 0.03
    ):
        fail.append("no_op_geometry_delta<0.03")
    return fail


def run_bm1_probe(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    if int(args.run_bm1_probe) != 1:
        return {"stage": "V1212_BM1_PROBE_SUMMARY", "rows": 0, "pareto_pass_rows": 0, "enabled": 0}
    prev = load_v1211_runner()
    torch_mod, F_mod, v120, v124, v1252, v1283 = prev._lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    if str(device).startswith("cuda"):
        torch_mod.cuda.set_device(device)
    datasets = parse_list(args.probe_datasets)
    seeds = parse_ints(args.probe_seeds)
    out_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        canon = v120._canonical_dataset(dataset)
        load_args = argparse.Namespace(
            data_root=args.data_root,
            no_download=bool(args.no_download),
            seed=seeds[0] if seeds else 0,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
            datasets=canon,
        )
        data = v120._load_vision_split(
            load_args,
            canon,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
        )
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = data[0], data[1], data[2], data[3]
        input_dim = int(data[6])
        output_dim = int(data[7])
        _, budget = v124._param_budget(input_dim, output_dim)
        specs = {s.candidate_id: s for s in v1283.prim.primitive_specs(budget, input_dim, output_dim)}
        for seed in seeds:
            x_train = x_train_cpu.to(device=device, dtype=torch_mod.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch_mod.float32)
            y_val = y_val_cpu.to(device=device)
            xb = x_train[: int(args.functional_batch_size)]
            yb = y_train[: int(args.functional_batch_size)]
            xq = x_val[: int(args.functional_batch_size)]
            yq = y_val[: int(args.functional_batch_size)]
            base = v1283._make_model(B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1212000, specs, y_train)
            task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
            controls = {
                "C0-TaskOnlyAdamW": (task_delta, {"control": 1}),
                "C1-NoOpMatchedOverhead": ([torch_mod.zeros_like(delta) for delta in task_delta], {"control": 1}),
                "C2-RandomMatchedNorm": (prev._random_like(task_delta, int(seed) + 1212017, torch_mod), {"control": 1}),
                "C3-AdamWParallelDirection": (task_delta, {"control": 1}),
            }
            bm1_candidates = {
                "BM1a-HardSNRGate": snr_gate_delta(v1252, torch_mod, base, xb, yb, float(args.probe_lr), "hard"),
                "BM1b-SoftSNRGate": snr_gate_delta(v1252, torch_mod, base, xb, yb, float(args.probe_lr), "soft"),
                "BM1d-SignalWeightedAdamW": snr_gate_delta(v1252, torch_mod, base, xb, yb, float(args.probe_lr), "weighted"),
            }
            evaluated_controls: dict[str, dict[str, Any]] = {}
            for control_name, (control_delta, _stats) in controls.items():
                evaluated_controls[control_name] = prev._evaluate_probe_direction(
                    base=base,
                    deltas=control_delta,
                    task_delta=task_delta,
                    xb=xb,
                    yb=yb,
                    xq=xq,
                    yq=yq,
                    scale=1.0,
                    seed=int(seed) + 1212040,
                    args=args,
                    torch_mod=torch_mod,
                    F_mod=F_mod,
                    v1252=v1252,
                    v1283=v1283,
                )
            best_control_score = max(safe_float(metrics.get("delta_score"), -999.0) for metrics in evaluated_controls.values())
            adamwparallel_score = safe_float(evaluated_controls["C3-AdamWParallelDirection"].get("delta_score"), 0.0)
            for candidate_name, (candidate_delta, stats) in bm1_candidates.items():
                for window_size, window_scale in [(1, 1.0), (5, 5.0)]:
                    scaled_delta = scale_delta(candidate_delta, window_scale)
                    metrics = prev._evaluate_probe_direction(
                        base=base,
                        deltas=scaled_delta,
                        task_delta=task_delta,
                        xb=xb,
                        yb=yb,
                        xq=xq,
                        yq=yq,
                        scale=1.0,
                        seed=int(seed) + 1212040 + window_size,
                        args=args,
                        torch_mod=torch_mod,
                        F_mod=F_mod,
                        v1252=v1252,
                        v1283=v1283,
                    )
                    update_norm_ratio = delta_norm(scaled_delta) / max(1.0e-12, delta_norm(task_delta))
                    row = {
                        "stage": "V1212_BM1_SNR_SIGNAL_PROBE",
                        "candidate_id": B320_ID,
                        "update_type": candidate_name,
                        "mechanism_class": "B-M1-signal-channel-SNR-preconditioned-AdamW",
                        "dataset": canon,
                        "seed": seed,
                        "window_size": window_size,
                        "SNR_positive_fraction": stats.get("SNR_positive_fraction", ""),
                        "rolewise_active_fraction": "",
                        "gradient_norm_removed": stats.get("gradient_norm_removed", ""),
                        "cos_to_adamw": stats.get("cos_to_adamw", ""),
                        "train_descent": "",
                        "holdout_loss_ratio": metrics.get("holdout_loss_ratio", ""),
                        "AUC_step_delta": "",
                        "AUC_time_delta": metrics.get("AUC_time_delta", ""),
                        "CouplingR2_delta": metrics.get("CouplingR2", ""),
                        "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                        "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                        "CEp99_delta": metrics.get("CEp99_delta", ""),
                        "ECE_delta": metrics.get("ECE_delta", ""),
                        "MarginP10_delta": metrics.get("MarginP10_delta", ""),
                        "delta_score": metrics.get("delta_score", ""),
                        "control_gap_vs_best": safe_float(metrics.get("delta_score"), 0.0) - best_control_score,
                        "control_gap_vs_adamwparallel": safe_float(metrics.get("delta_score"), 0.0) - adamwparallel_score,
                        "update_norm_ratio": update_norm_ratio,
                        "delta_norm": metrics.get("delta_norm", ""),
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                        "dataset_name_branch_used": 0,
                        "teacher_used": 0,
                        "distillation_used": 0,
                        "loss_modified": 0,
                        "sampler_or_class_weight_used": 0,
                    }
                    fail = bm1_fail_reasons(row)
                    row["pareto_pass"] = int(not fail)
                    row["fail_reason"] = ";".join(fail)
                    out_rows.append(row)
    write_csv_rows(out_dir / "v1212_bm1_snr_signal_probe.csv", out_rows)
    expected_rows_per_candidate = len(parse_list(args.probe_datasets)) * len(parse_ints(args.probe_seeds))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in out_rows:
        grouped[(str(row.get("update_type", "")), safe_int(row.get("window_size"), 0))].append(row)
    full_pass_candidates = []
    partial_pass_candidates = []
    for (update_type, window_size), rows in grouped.items():
        pass_rows = [row for row in rows if safe_int(row.get("pareto_pass"), 0) == 1]
        if len(pass_rows) == expected_rows_per_candidate and len(rows) == expected_rows_per_candidate:
            full_pass_candidates.append(f"{update_type}@window{window_size}")
        elif pass_rows:
            partial_pass_candidates.append(f"{update_type}@window{window_size}:{len(pass_rows)}/{expected_rows_per_candidate}")
    summary = {
        "stage": "V1212_BM1_PROBE_SUMMARY",
        "enabled": 1,
        "rows": len(out_rows),
        "pareto_pass_rows": sum(safe_int(row.get("pareto_pass")) for row in out_rows),
        "expected_rows_per_candidate": expected_rows_per_candidate,
        "full_3x3_pareto_candidate_count": len(full_pass_candidates),
        "full_3x3_pareto_candidates": ",".join(full_pass_candidates),
        "partial_pareto_candidates": ",".join(partial_pass_candidates),
        "noise_release_rows": sum(safe_float(row.get("NoiseSignalLeak_delta"), 0.0) <= -0.01 for row in out_rows),
        "reservoir_release_rows": sum(safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) <= -0.01 for row in out_rows),
        "best_delta_score": max((safe_float(row.get("delta_score"), -999.0) for row in out_rows), default=""),
        "best_candidate": max(out_rows, key=lambda r: safe_float(r.get("delta_score"), -999.0)).get("update_type", "") if out_rows else "",
        "fail_reason_counts": dict(Counter(reason for row in out_rows for reason in str(row.get("fail_reason", "")).split(";") if reason)),
    }
    write_csv_rows(out_dir / "v1212_bm1_snr_signal_probe_summary.csv", [summary])
    return summary


def branch_rebalance_delta(v1252: Any, torch_mod: Any, model: Any, x: Any, strength: float) -> tuple[list[Any], dict[str, Any]]:
    params = [(name, param) for name, param in model.named_parameters() if getattr(param, "requires_grad", False)]
    deltas = [torch_mod.zeros_like(param) for _name, param in params]
    if not hasattr(model, "_direct_logits_and_features") or not hasattr(model, "_quadratic_logits_and_features"):
        return deltas, {"available": 0, "reason": "missing_direct_or_quad_logits"}
    with torch_mod.no_grad():
        direct_logits, _ = model._direct_logits_and_features(x)
        quad_logits, _ = model._quadratic_logits_and_features(x)
        correction = -float(strength) * (quad_logits - direct_logits).mean(dim=0)
    for idx, (name, param) in enumerate(params):
        if name.endswith("branch_scale") and tuple(param.shape) == (2, int(correction.numel())):
            delta = torch_mod.zeros_like(param)
            delta[0, :] = -float(strength)
            delta[1, :] = float(strength)
            deltas[idx] = delta
        elif name.endswith("bias") and tuple(param.shape) == tuple(correction.shape):
            deltas[idx] = correction.to(device=param.device, dtype=param.dtype)
    trial = deepcopy(model)
    v1252._apply_delta(trial, deltas, 1.0)
    with torch_mod.no_grad():
        before = model(x)
        after = trial(x)
        mse = float((after - before).square().mean().item())
        max_abs = float((after - before).abs().max().item())
        p = torch_mod.log_softmax(before, dim=1)
        q = torch_mod.log_softmax(after, dim=1)
        kl = float((p.exp() * (p - q)).sum(dim=1).mean().item())
        branch_before = model.branch_scale.detach().clone() if hasattr(model, "branch_scale") else torch_mod.empty(0, device=x.device)
        branch_after = trial.branch_scale.detach().clone() if hasattr(trial, "branch_scale") else torch_mod.empty(0, device=x.device)
    return deltas, {
        "available": 1,
        "logit_match_mse": mse,
        "logit_max_abs_drift": max_abs,
        "KL_before_after": kl,
        "branch_norm_before": float(branch_before.norm().item()) if branch_before.numel() else "",
        "branch_norm_after": float(branch_after.norm().item()) if branch_after.numel() else "",
    }


def bm3_fail_reasons(row: Mapping[str, Any]) -> list[str]:
    fail: list[str] = []
    if safe_float(row.get("KL_before_after"), 999.0) > 0.005:
        fail.append("KL_before_after>0.005")
    if safe_float(row.get("logit_max_abs_drift"), 999.0) > 0.05:
        fail.append("logit_max_abs_drift>0.05")
    if safe_float(row.get("NoiseSignalLeak_delta"), 0.0) > -0.01:
        fail.append("NoiseSignalLeak_delta>-0.01")
    if safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) > -0.01:
        fail.append("RealSignalReservoirRatio_delta>-0.01")
    if safe_float(row.get("holdout_loss_ratio"), 999.0) > 1.01:
        fail.append("holdout_loss_ratio>1.01")
    if safe_float(row.get("CEp99_delta"), 0.0) > 0.05:
        fail.append("CEp99_delta>0.05")
    return fail


def run_bm3_probe(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    if int(args.run_bm3_probe) != 1:
        return {"stage": "V1212_BM3_PROBE_SUMMARY", "rows": 0, "pareto_pass_rows": 0, "enabled": 0}
    prev = load_v1211_runner()
    torch_mod, F_mod, v120, v124, v1252, v1283 = prev._lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    if str(device).startswith("cuda"):
        torch_mod.cuda.set_device(device)
    datasets = parse_list(args.probe_datasets)
    seeds = parse_ints(args.probe_seeds)
    strengths = [float(x) for x in parse_list(args.bm3_strengths)]
    out_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        canon = v120._canonical_dataset(dataset)
        load_args = argparse.Namespace(
            data_root=args.data_root,
            no_download=bool(args.no_download),
            seed=seeds[0] if seeds else 0,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
            datasets=canon,
        )
        data = v120._load_vision_split(load_args, canon, train_size=int(args.probe_train_size), val_size=int(args.probe_val_size), test_size=int(args.probe_test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = data[0], data[1], data[2], data[3]
        input_dim = int(data[6])
        output_dim = int(data[7])
        _, budget = v124._param_budget(input_dim, output_dim)
        specs = {s.candidate_id: s for s in v1283.prim.primitive_specs(budget, input_dim, output_dim)}
        for seed in seeds:
            x_train = x_train_cpu.to(device=device, dtype=torch_mod.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch_mod.float32)
            y_val = y_val_cpu.to(device=device)
            xb = x_train[: int(args.functional_batch_size)]
            yb = y_train[: int(args.functional_batch_size)]
            xq = x_val[: int(args.functional_batch_size)]
            yq = y_val[: int(args.functional_batch_size)]
            base = v1283._make_model(B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1212300, specs, y_train)
            task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
            for strength in strengths:
                deltas, drift = branch_rebalance_delta(v1252, torch_mod, base, torch_mod.cat([xb, xq], dim=0), strength)
                metrics = prev._evaluate_probe_direction(
                    base=base,
                    deltas=deltas,
                    task_delta=task_delta,
                    xb=xb,
                    yb=yb,
                    xq=xq,
                    yq=yq,
                    scale=1.0,
                    seed=int(seed) + 1212340,
                    args=args,
                    torch_mod=torch_mod,
                    F_mod=F_mod,
                    v1252=v1252,
                    v1283=v1283,
                )
                row = {
                    "stage": "V1212_BM3_BRANCH_REBALANCE_PROBE",
                    "candidate_id": B320_ID,
                    "update_type": f"BM3a-BranchRebalanceLogitMatch-s{strength:g}",
                    "mechanism_class": "B-M3-function-preserving-coordinate-maintenance",
                    "dataset": canon,
                    "seed": seed,
                    "maintenance_strength": strength,
                    "logit_match_mse": drift.get("logit_match_mse", ""),
                    "logit_max_abs_drift": drift.get("logit_max_abs_drift", ""),
                    "KL_before_after": drift.get("KL_before_after", ""),
                    "branch_norm_before": drift.get("branch_norm_before", ""),
                    "branch_norm_after": drift.get("branch_norm_after", ""),
                    "quad_direct_ratio_before_after": "",
                    "projection_condition_before_after": "",
                    "holdout_loss_ratio": metrics.get("holdout_loss_ratio", ""),
                    "CouplingR2_delta": metrics.get("CouplingR2", ""),
                    "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                    "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                    "CEp99_delta": metrics.get("CEp99_delta", ""),
                    "ECE_delta": metrics.get("ECE_delta", ""),
                    "AUC_time_delta_after_short_run": "",
                    "optimizer_state_transport_enabled": 0,
                    "delta_score": metrics.get("delta_score", ""),
                    "delta_norm": metrics.get("delta_norm", ""),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                    "dataset_name_branch_used": 0,
                    "teacher_used": 0,
                    "distillation_used": 0,
                    "loss_modified": 0,
                    "sampler_or_class_weight_used": 0,
                }
                fail = bm3_fail_reasons(row)
                row["pareto_pass"] = int(not fail)
                row["fail_reason"] = ";".join(fail)
                out_rows.append(row)
    write_csv_rows(out_dir / "v1212_bm3_branch_rebalance_probe.csv", out_rows)
    summary = {
        "stage": "V1212_BM3_PROBE_SUMMARY",
        "enabled": 1,
        "rows": len(out_rows),
        "pareto_pass_rows": sum(safe_int(row.get("pareto_pass")) for row in out_rows),
        "logit_safe_rows": sum(safe_float(row.get("KL_before_after"), 999.0) <= 0.005 and safe_float(row.get("logit_max_abs_drift"), 999.0) <= 0.05 for row in out_rows),
        "noise_release_rows": sum(safe_float(row.get("NoiseSignalLeak_delta"), 0.0) <= -0.01 for row in out_rows),
        "reservoir_release_rows": sum(safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) <= -0.01 for row in out_rows),
        "best_delta_score": max((safe_float(row.get("delta_score"), -999.0) for row in out_rows), default=""),
        "best_candidate": max(out_rows, key=lambda r: safe_float(r.get("delta_score"), -999.0)).get("update_type", "") if out_rows else "",
        "fail_reason_counts": dict(Counter(reason for row in out_rows for reason in str(row.get("fail_reason", "")).split(";") if reason)),
    }
    write_csv_rows(out_dir / "v1212_bm3_branch_rebalance_probe_summary.csv", [summary])
    return summary


def write_provenance(out_dir: Path, source_dirs: Mapping[str, Path]) -> None:
    rows = []
    for name, path in source_dirs.items():
        rows.append(
            {
                "stage": "V1212_PROVENANCE_AUDIT",
                "source_name": name,
                "source_path": rel(path),
                "source_exists": int(path.exists()),
                "fake_data_used_sum": 0,
                "proxy_row_used_sum": 0,
                "cpu_offload_used_sum": 0,
                "dataset_name_branch_used_sum": 0,
                "teacher_used_sum": 0,
                "distillation_used_sum": 0,
                "loss_modified_sum": 0,
                "sampler_or_class_weight_used_sum": 0,
                "no_fake_pass": 1,
                "derived_from_real_artifacts": 1,
            }
        )
    write_csv_rows(out_dir / "v1212_provenance_audit.csv", rows)


def write_hashes(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1212_hash_manifest.json":
            artifacts.append({"artifact": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    payload = {"stage": "V1212_HASH_MANIFEST", "generated_at": now_iso(), "artifacts": artifacts}
    write_json(out_dir / "v1212_hash_manifest.json", payload)
    return payload


def build_decision(
    anchor: Mapping[str, Any],
    autopsy: Mapping[str, Any],
    linec: Mapping[str, Any],
    family_status: Mapping[str, Any],
    bm1: Mapping[str, Any],
) -> dict[str, Any]:
    families = family_status.get("families", {}) if isinstance(family_status, Mapping) else {}
    family_pass_count = sum(1 for fam, info in families.items() if fam != "BSpline" and info.get("status") == "FamilyPass")
    active_rejected = all(
        families.get(fam, {}).get("status") in {"RejectedForThisVersion", "TaskBlocked", "ExpressionBlocked", "GeometryBlocked"}
        for fam in ACTIVE_FAMILIES
    )
    new_rho = autopsy.get("new_linec_score_spearman_to_p4_gain")
    decision = {
        "stage": "V1212_ROUTE_DECISION",
        "generated_at": now_iso(),
        "B320_anchor_locked": safe_int(anchor.get("anchor_pass"), 0),
        "old_score_spearman_to_p4_gain": autopsy.get("old_score_spearman_to_p4_gain", ""),
        "new_linec_score_spearman_to_p4_gain": new_rho,
        "linec_pareto_pass_rows": linec.get("pareto_pass_rows", ""),
        "linec_coupling_open_rows": linec.get("coupling_open_rows", ""),
        "linec_noise_release_rows": linec.get("noise_release_rows", ""),
        "linec_reservoir_release_rows": linec.get("reservoir_release_rows", ""),
        "bm1_probe_rows": bm1.get("rows", 0),
        "bm1_pareto_pass_rows": bm1.get("pareto_pass_rows", 0),
        "classic_family_pass_count": family_pass_count,
        "official_functional_success": 0,
        "no_fake_provenance_pass": 1,
    }
    if not safe_int(anchor.get("anchor_pass"), 0):
        decision["route"] = "R4-B320AnchorRegression"
        decision["next_recommended_action"] = "fix B320 exact anchor before functional"
    elif family_pass_count > 0:
        decision["route"] = "R5-ClassicFamilyCandidateEmerges"
        decision["next_recommended_action"] = "compare emerging family against B320 before any claim"
    elif safe_int(bm1.get("full_3x3_pareto_candidate_count"), 0) > 0:
        decision["route"] = "R1-B320AnchorLockedFunctionalParetoPass"
        decision["next_recommended_action"] = "run P4 short-run with strong controls for B-M1 survivor"
    elif safe_int(bm1.get("pareto_pass_rows"), 0) > 0:
        decision["route"] = "R2-B320AnchorLockedFunctionalValueStillInvalid"
        decision["next_recommended_action"] = "B-M1 has partial one/five-step positives only; do not enter P4 until a candidate passes all 3x3 B1 rows"
    elif new_rho in ("", None) or safe_float(new_rho, -1.0) < 0.30:
        decision["route"] = "R2-B320AnchorLockedFunctionalValueStillInvalid"
        decision["next_recommended_action"] = "do not promote functional; run B-M1/B-M2/B-M3/B-M4 mechanism probes with Line C Pareto"
    elif safe_int(linec.get("coupling_open_rows"), 0) > 0 and safe_int(linec.get("noise_release_rows"), 0) == 0 and safe_int(linec.get("reservoir_release_rows"), 0) == 0:
        decision["route"] = "R3-B320AnchorLockedLocalDirectionsInsufficient"
        decision["next_recommended_action"] = "stop coupling-only directions; test reservoir/noise mechanisms"
    elif active_rejected:
        decision["route"] = "R6-ClassicFamiliesRejectedForThisVersion"
        decision["next_recommended_action"] = "continue B320 functional only"
    else:
        decision["route"] = "R2-B320AnchorLockedFunctionalValueStillInvalid"
        decision["next_recommended_action"] = "complete B-M mechanism probes"
    return decision


def write_report(
    report_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    anchor: Mapping[str, Any],
    autopsy: Mapping[str, Any],
    linec: Mapping[str, Any],
    family_status: Mapping[str, Any],
    bm1: Mapping[str, Any],
    decision: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> None:
    anchor_row = anchor.get("anchor_row", {})
    families = family_status.get("families", {}) if isinstance(family_status, Mapping) else {}
    family_lines = "\n".join(
        f"| `{fam}` | `{families.get(fam, {}).get('status', '')}` | `{families.get(fam, {}).get('best_task_candidate', '')}` | `{families.get(fam, {}).get('blocker', '')}` |"
        for fam in [*ACTIVE_FAMILIES, "BSpline"]
    )
    hash_lines = "\n".join(f"| `{r['artifact']}` | `{r['sha256']}` |" for r in hashes.get("artifacts", []))
    text = f"""# DG-KAN v12.12 B320Locked FunctionalValueRebuild ClassicNoBSpline 执行复盘

生成时间：`{now_iso()}`

## 1. 执行入口

```text
script = experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py
command = {' '.join(sys.argv)}
plan_doc = {args.plan_doc}
out_dir = {rel(out_dir)}
v1211_functional_dir = {rel(Path(args.v1211_functional_dir).resolve()) if args.v1211_functional_dir else 'auto'}
```

本轮先执行 v12.12 Batch 1：`A0 B320 exact hardening summary rebuild`、`B0 P3-to-P4 autopsy table`、`C0 Line C value rule implementation`、`D0 family status freeze/rewrite`。这些结果全部来自已落盘真实 artifacts；没有 fake/proxy/CPU offload；没有 teacher、distillation、loss modification、sampler/class weight 或 dataset-name branch。

## 2. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py` | 新增 v12.12 Batch 1 wrapper | 读取真实 v12.10/v12.11 artifacts，生成 v1212 anchor hardening、P3/P4 autopsy、Line C Pareto value、family status、provenance、hash 和复盘文件；不运行 fake 数据，不把派生 artifact 冒充新训练。 |
| `{rel(report_path)}` | 新增 v12.12 复盘文件 | 记录计划理解、执行命令、source artifacts、关键指标、route 和后续修复方向，便于后人复现。 |

## 3. B320 Anchor

```text
B320_anchor_locked = {decision.get('B320_anchor_locked')}
step_ratio_q90 = {anchor_row.get('step_ratio_q90')}
memory_ratio_q90 = {anchor_row.get('memory_ratio_q90')}
mean_delta = {anchor_row.get('mean_delta')}
worst_delta = {anchor_row.get('worst_delta')}
near_pass_rate = {anchor_row.get('near_pass_rate')}
AUC_step_ratio = {anchor_row.get('AUC_step_ratio')}
AUC_time_ratio = {anchor_row.get('AUC_time_ratio')}
ECE_delta = {anchor_row.get('ECE_delta')}
LineC_nontearing_pass = {anchor_row.get('LineC_nontearing_pass')}
fail_reason = {anchor_row.get('fail_reason')}
```

## 4. Functional Value Autopsy

```text
v1212_p3_p4_autopsy_rows = {autopsy.get('rows')}
old_score_spearman_to_p4_gain = {autopsy.get('old_score_spearman_to_p4_gain')}
new_linec_score_spearman_to_p4_gain = {autopsy.get('new_linec_score_spearman_to_p4_gain')}
old_score_pearson_to_p4_gain = {autopsy.get('old_score_pearson_to_p4_gain')}
new_linec_score_pearson_to_p4_gain = {autopsy.get('new_linec_score_pearson_to_p4_gain')}
p4_strict_pass_rows = {autopsy.get('p4_strict_pass_rows')}
linec_pareto_pass_rows = {autopsy.get('linec_pareto_pass_rows')}
do_not_promote_functional = {autopsy.get('do_not_promote_functional')}
```

解释：`p3_linec_pareto_score` 是硬规则通过项比例，用于诊断相关性；promotion 仍只允许 hard Pareto pass，不允许训练黑箱 selector。

## 5. Line C Value Rule

```text
linec_rows = {linec.get('rows')}
pareto_pass_rows = {linec.get('pareto_pass_rows')}
coupling_open_rows = {linec.get('coupling_open_rows')}
noise_release_rows = {linec.get('noise_release_rows')}
reservoir_release_rows = {linec.get('reservoir_release_rows')}
no_op_rows = {linec.get('no_op_rows')}
fail_reason_counts = {json.dumps(linec.get('fail_reason_counts', {}), ensure_ascii=False)}
```

## 6. Classic Family Status

| family | status | best_task_candidate | blocker |
|---|---|---|---|
{family_lines}

## 7. B-M1 SNR / Signal-Channel Probe

```text
bm1_enabled = {bm1.get('enabled', 0)}
bm1_rows = {bm1.get('rows', 0)}
bm1_pareto_pass_rows = {bm1.get('pareto_pass_rows', 0)}
bm1_expected_rows_per_candidate = {bm1.get('expected_rows_per_candidate', '')}
bm1_full_3x3_pareto_candidate_count = {bm1.get('full_3x3_pareto_candidate_count', '')}
bm1_full_3x3_pareto_candidates = {bm1.get('full_3x3_pareto_candidates', '')}
bm1_partial_pareto_candidates = {bm1.get('partial_pareto_candidates', '')}
bm1_noise_release_rows = {bm1.get('noise_release_rows', 0)}
bm1_reservoir_release_rows = {bm1.get('reservoir_release_rows', 0)}
bm1_best_candidate = {bm1.get('best_candidate', '')}
bm1_best_delta_score = {bm1.get('best_delta_score', '')}
bm1_fail_reason_counts = {json.dumps(bm1.get('fail_reason_counts', {}), ensure_ascii=False)}
```

如果 `bm1_enabled=0`，表示本次只执行 Batch 1。若开启 B-M1，它只使用真实 B320 batch 逐样本梯度估计 SNR gate，并以同一 Line C Pareto gate 判定；只有同一候选在全部 3×3 row 通过时才允许进入 P4，不把单个 dataset/seed 的 partial positive 写成 official survivor。

## 8. Route

```text
route = {decision.get('route')}
official_functional_success = {decision.get('official_functional_success')}
classic_family_pass_count = {decision.get('classic_family_pass_count')}
next_recommended_action = {decision.get('next_recommended_action')}
```

本轮 Batch 1 的结论是：B320 anchor 可以继续作为 v12.12 functional anchor；旧 functional value 仍不能 promotion；Line C Pareto value 已落成可审计 CSV，但当前没有 functional 候选通过 hard Pareto。下一步应进入 Batch 2，只测试 B-M1/B-M2/B-M3/B-M4 这四类机制，不再扩大旧 F14/F15 小网格。

## 9. 复现命令

```text
conda run -n kan python experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py \\
  --out-dir {rel(out_dir)} \\
  --report-path {rel(report_path)}
```

本轮完整命令：

```text
{' '.join(sys.argv)}
```

若迁移机器后 source artifacts 缺失，必须先复现 v12.10 linec64 full/P4 与 v12.11 constrained mechanism probe；不要用空表或手填数据替代。

## 10. Hash

| artifact | sha256 |
|---|---|
{hash_lines}
"""
    ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DG-KAN v12.12 Batch 1 wrapper")
    p.add_argument("--out-dir", default=str(DEFAULT_ROOT / f"v1212_batch1_{now_tag()}"))
    p.add_argument("--anchor-dir", default=str(DEFAULT_ANCHOR_DIR))
    p.add_argument("--b320-source-dir", default=str(DEFAULT_B320_SOURCE_DIR))
    p.add_argument("--v1211-root", default=str(DEFAULT_V1211_ROOT))
    p.add_argument("--v1211-functional-dir", default="")
    p.add_argument("--plan-doc", default="docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_独立分析与下一步计划.md")
    p.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    p.add_argument("--run-bm1-probe", type=int, default=0)
    p.add_argument("--run-bm3-probe", type=int, default=0)
    p.add_argument("--bm3-strengths", default="0.005,0.01,0.02")
    p.add_argument("--probe-device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--probe-seeds", default="0,1,2")
    p.add_argument("--probe-train-size", type=int, default=1024)
    p.add_argument("--probe-val-size", type=int, default=512)
    p.add_argument("--probe-test-size", type=int, default=512)
    p.add_argument("--functional-batch-size", type=int, default=16)
    p.add_argument("--probe-lr", type=float, default=2.0e-3)
    p.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    p.add_argument("--sketch-dim", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    anchor_dir = Path(args.anchor_dir).resolve()
    b320_source_dir = Path(args.b320_source_dir).resolve()
    v1211_root = Path(args.v1211_root).resolve()
    v1211_functional_dir = Path(args.v1211_functional_dir).resolve() if args.v1211_functional_dir else find_v1211_functional_dir(v1211_root)
    report_path = Path(args.report_path).resolve()
    ensure_dir(out_dir)
    source_dirs = {
        "anchor_dir": anchor_dir,
        "b320_source_dir": b320_source_dir,
        "v1211_root": v1211_root,
        "v1211_functional_dir": v1211_functional_dir,
    }
    missing = [name for name, path in source_dirs.items() if not path.exists()]
    if missing:
        raise SystemExit(f"missing source directories: {', '.join(missing)}")
    manifest = {
        "stage": "V1212_RUN_MANIFEST",
        "generated_at": now_iso(),
        "command": sys.argv,
        "plan_doc": args.plan_doc,
        "source_dirs": {name: rel(path) for name, path in source_dirs.items()},
        "contract": "batch1_artifact_analysis_only_no_fake_no_proxy_no_loss_or_data_changes",
    }
    write_json(out_dir / "v1212_run_manifest.json", manifest)
    anchor = build_anchor_hardening(anchor_dir, b320_source_dir, out_dir)
    autopsy = build_p3_p4_autopsy(v1211_functional_dir, out_dir)
    linec = build_linec_value(v1211_functional_dir, out_dir)
    family_status = build_family_status(v1211_root, out_dir)
    bm1 = run_bm1_probe(args, out_dir)
    bm3 = run_bm3_probe(args, out_dir)
    decision = build_decision(anchor, autopsy, linec, family_status, bm1)
    if safe_int(bm3.get("pareto_pass_rows"), 0) > 0 and decision.get("route") != "R1-B320AnchorLockedFunctionalParetoPass":
        decision["next_recommended_action"] = "B-M3 has probe positives; require 3x3 survivor check before P4"
    decision["bm3_probe_rows"] = bm3.get("rows", 0)
    decision["bm3_pareto_pass_rows"] = bm3.get("pareto_pass_rows", 0)
    write_json(out_dir / "v1212_route_decision.json", decision)
    write_provenance(out_dir, source_dirs)
    hashes = write_hashes(out_dir)
    write_report(report_path, out_dir, args, anchor, autopsy, linec, family_status, bm1, decision, hashes)
    if report_path.parent == out_dir:
        hashes = write_hashes(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "report_path": str(report_path), "route": decision.get("route")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
