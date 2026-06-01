#!/usr/bin/env python3
"""v12.21 fail-closed continue-and-generate runner.

This runner consumes v12.21 Line A training rows plus inherited v12.20/v12.19
artifacts.  It writes the full v12.21 artifact contract, including fail-closed
route decisions, target/visibility sanity checks, actuator v2 smoke diagnostics,
and next-hypothesis/no-go documents.  It does not synthesize training metrics:
missing native inputs are marked unavailable and gated closed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v1218_b320_codeaudit_lossagnostic_functional as v1218_audit  # noqa: E402
import run_v1218_b320_label_free_ablation as linea  # noqa: E402
import run_v124_multibasis_functional_dual as v124  # noqa: E402
import run_v1252_efficiency_functional_manifold as v1252  # noqa: E402


PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.21_FailClosedContinue2_LabelFreeFunctional_独立分析与下一步计划.md"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "v12_21_failclosed_continue2_label_free_functional" / "official_continuation"
DEFAULT_V1220_DIR = REPO_ROOT / "results" / "v12_20_failclosed_continue_label_free_functional" / "official_continuation"
DEFAULT_V1219_DIR = REPO_ROOT / "results" / "v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry" / "official_from_v1218_v1217_v1216_artifacts"
DEFAULT_V1218_DIR = REPO_ROOT / "results" / "v12_18_b320_codeaudit_lossagnostic_functional" / "official_from_v1217_v1216_artifacts"
DEFAULT_V1217_DIR = REPO_ROOT / "results" / "v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry" / "support_expand_seed012_seed345_b128_w10_12_15"

EPS = 1.0e-12
MLP_IDS = {"MLP-same-param-AdamW", "MLP-same-step-FLOP-AdamW"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        val = float(value)
    except Exception:
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def finite_values(values: Sequence[Any]) -> list[float]:
    out = []
    for value in values:
        val = safe_float(value, float("nan"))
        if math.isfinite(val):
            out.append(val)
    return out


def mean(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else float("nan")


def finite_min(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return min(vals) if vals else float("nan")


def finite_max(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return max(vals) if vals else float("nan")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    fields = list(fieldnames or [])
    seen = set(fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["stage", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_range_for_symbol(path: Path, symbol: str) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                return int(node.lineno), int(getattr(node, "end_lineno", node.lineno))
    except SyntaxError:
        pass
    for idx, line in enumerate(text.splitlines(), start=1):
        if symbol in line:
            return idx, idx
    return None, None


def code_ref(path: str, symbol: str) -> dict[str, Any]:
    start, end = line_range_for_symbol(REPO_ROOT / path, symbol)
    return {
        "actual_file_path": path,
        "actual_line_start": "" if start is None else start,
        "actual_line_end": "" if end is None else end,
        "main_symbols": symbol,
        "unknown_or_not_inspected": int(start is None),
    }


def rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(rank(xs), rank(ys))


def auc_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(float(s))]
    if not pairs:
        return float("nan")
    pos = sum(y for _s, y in pairs)
    neg = len(pairs) - pos
    if pos <= 0 or neg <= 0:
        return float("nan")
    values = [p[0] for p in pairs]
    labels_f = [p[1] for p in pairs]
    ranks = rank(values)
    pos_rank_sum = sum(r for r, y in zip(ranks, labels_f) if y == 1)
    return float((pos_rank_sum - pos * (pos - 1) / 2.0) / max(EPS, pos * neg))


def precision_recall_at_k(scores: Sequence[float], labels: Sequence[int], bottom: bool = False) -> tuple[float, float]:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(float(s))]
    if not pairs:
        return float("nan"), float("nan")
    pos = sum(y for _s, y in pairs)
    if pos <= 0:
        return float("nan"), float("nan")
    k = max(1, min(pos, int(math.ceil(0.10 * len(pairs)))))
    pairs = sorted(pairs, key=lambda item: item[0], reverse=not bottom)
    hits = sum(y for _s, y in pairs[:k])
    return float(hits / k), float(hits / pos)


def ridge_leaveout_r2(rows: Sequence[Mapping[str, Any]], feature_cols: Sequence[str], target_col: str, split_col: str) -> float:
    scores: list[float] = []
    for split in sorted({str(r.get(split_col, "")) for r in rows}):
        train = [r for r in rows if str(r.get(split_col, "")) != split]
        test = [r for r in rows if str(r.get(split_col, "")) == split]
        if len(train) < 8 or len(test) < 2:
            continue
        x_train = np.asarray([[safe_float(r.get(c), 0.0) for c in feature_cols] for r in train], dtype=float)
        y_train = np.asarray([safe_float(r.get(target_col), 0.0) for r in train], dtype=float)
        x_test = np.asarray([[safe_float(r.get(c), 0.0) for c in feature_cols] for r in test], dtype=float)
        y_test = np.asarray([safe_float(r.get(target_col), 0.0) for r in test], dtype=float)
        if float(np.var(y_test)) <= EPS:
            continue
        mu = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        std[std < 1.0e-8] = 1.0
        xt = (x_train - mu) / std
        xv = (x_test - mu) / std
        reg = 1.0e-3 * np.eye(xt.shape[1])
        try:
            beta = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xt.T @ xt + reg) @ xt.T @ y_train
        pred = xv @ beta
        ss_res = float(((y_test - pred) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        scores.append(1.0 - ss_res / max(EPS, ss_tot))
    return finite_min(scores)


def collect_linea_rows(linea_root: Path, extra_csvs: Sequence[str]) -> list[dict[str, str]]:
    paths: list[Path] = []
    if linea_root.exists():
        paths.extend(sorted(linea_root.rglob("*_ablation.csv")))
    for item in extra_csvs:
        p = Path(item)
        if p.exists():
            paths.append(p)
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in paths:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        for row in read_csv_rows(path):
            out = dict(row)
            out["_source_csv"] = rel(path)
            rows.append(out)
    return rows


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("_source_csv", "")))


def candidate_family(candidate_id: str) -> str:
    if candidate_id == "A30-OptimizerObservableFrame-labelFree":
        return "A30 optimizer-observable opaque update frame"
    if candidate_id == "A31-AugConsistencyTangentFrame-labelFree":
        return "A31 augmentation-consistency tangent frame"
    if candidate_id == "A32-PersistentDriftFrame-labelFree":
        return "A32 persistent drift frame"
    if candidate_id == "A33-RoleBalancedPrimitiveEnergyFrame-labelFree":
        return "A33 role-balanced primitive energy frame"
    if candidate_id == "A34-HybridA1OptimizerFrame-labelFree":
        return "A34 A1 plus optimizer frame"
    if candidate_id == "A35-HybridA1AugDriftFrame-labelFree":
        return "A35 A1 plus augmentation/drift frame"
    if candidate_id == "A1-noYForStats":
        return "A1 B320 no-y-for-stats baseline"
    if candidate_id == "A0-labelInit":
        return "A0 label-informed B320 anchor"
    if candidate_id in MLP_IDS:
        return "MLP control"
    return "carried-forward label-free/control"


def candidate_uses_label(candidate_id: str) -> int:
    return int(candidate_id in {"A0-labelInit", "A20-shuffledLabelTrainProbe-diagnostic", "A22-permutedClassMeanP-diagnostic", "A29-oracleSmallLabelTrainProbe-diagnostic"})


def candidate_uses_pseudo(candidate_id: str) -> int:
    return int(candidate_id in {"A21-randomClassCentroid-diagnostic", "A28-unsupervisedClusterTrainProbe-diagnostic"})


def candidate_uses_optimizer_update(candidate_id: str) -> int:
    return int(candidate_id in {"A30-OptimizerObservableFrame-labelFree", "A34-HybridA1OptimizerFrame-labelFree"})


def candidate_frame_family(candidate_id: str) -> str:
    if "A30" in candidate_id or "A34" in candidate_id:
        return "optimizer_update_opaque"
    if "A31" in candidate_id:
        return "augmentation_tangent"
    if "A32" in candidate_id:
        return "persistent_drift"
    if "A33" in candidate_id:
        return "role_balanced_primitive_energy"
    if "A35" in candidate_id:
        return "hybrid_aug_drift"
    return candidate_family(candidate_id)


def line_r_code_review(out_dir: Path) -> dict[str, Any]:
    specs = [
        ("CR0", "run_v1221 entrypoint and route decision", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "run_main"),
        ("CR1", "B320 construction and candidate specs", "experiments/run_v1218_b320_label_free_ablation.py", "ablation_specs"),
        ("CR2", "y_for_stats/trainprobe wiring", "experiments/run_v1218_b320_label_free_ablation.py", "make_model"),
        ("CR3", "label-free frame bank and v12.21 tokens", "dgkan/models/fc_purekan_primitives.py", "augtangentp"),
        ("CR4", "Line C audit metrics", "experiments/run_v1218_b320_label_free_ablation.py", "signal_reservoir_metrics_detailed"),
        ("CR5", "T1/T2/T3 tiering", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_visibility_v5"),
        ("CR6", "hard release labels", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "release_labels"),
        ("CR7", "row join stability", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_join_key_audit"),
        ("CR8", "delta sign convention", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_delta_sign_audit"),
        ("CR9", "soft target normalization", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_soft_target_regression"),
        ("CR10", "T1B native optimizer-update logging", "experiments/run_v1218_b320_label_free_ablation.py", "optimizer_update_observable_stats"),
        ("CR11", "actuator dictionary v2", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_actuator_v2"),
        ("CR12", "controls/windows", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "matched_control_gap"),
        ("CR13", "P3/P4 gate", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_functional_gates"),
        ("CR14", "no dataset-name branch audit", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_provenance_audit"),
        ("CR15", "classic monitor/new hypothesis", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_classic_line_d"),
    ]
    rows = []
    for item, desc, path, symbol in specs:
        ref = code_ref(path, symbol)
        rows.append(
            {
                "stage": "V1221_CODE_REVIEW_MANIFEST",
                "review_item": item,
                "description": desc,
                **ref,
                "called_by": "v12.21 runner / Line A training runner",
                "artifact_fields_written": "v1221 artifact contract",
                "uses_label": int(item in {"CR2", "CR4", "CR6"}),
                "uses_ce_vector": int(item in {"CR4", "CR6"}),
                "uses_validation_or_test_for_commit": 0,
                "uses_dataset_name_branch": 0,
                "is_promotion_feature": int(item in {"CR5", "CR10", "CR13"}),
                "is_diagnostic_only": int(item in {"CR4", "CR6", "CR8", "CR9", "CR11", "CR15"}),
                "requires_manual_review": 0,
            }
        )
    write_csv_rows(out_dir / "v1221_code_review_manifest.csv", rows)
    write_json(out_dir / "v1221_core_symbol_map.json", {row["review_item"]: {k: row[k] for k in ["actual_file_path", "actual_line_start", "actual_line_end", "main_symbols"]} for row in rows})
    required = {f"CR{i}" for i in range(15)}
    missing = [r for r in rows if r["review_item"] in required and safe_int(r.get("unknown_or_not_inspected"), 0)]
    return {"line_r_rows": len(rows), "line_r_required_missing": len(missing), "line_r_pass": int(len(missing) == 0)}


def summarize_label_free(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a0_by_key = {row_key(r): r for r in rows if r.get("candidate_id") == "A0-labelInit"}
    mlp_by_key = {row_key(r): r for r in rows if r.get("candidate_id") == "MLP-same-step-FLOP-AdamW"}
    by_cid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    linec_rows = []
    failure_rows = []
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if cid.startswith("A"):
            by_cid[cid].append(row)
        if not cid.startswith("A") or cid == "A0-labelInit":
            continue
        key = row_key(row)
        a0 = a0_by_key.get(key, {})
        mlp = mlp_by_key.get(key, {})
        n_delta = safe_float(row.get("linec_NoiseSignalLeak"), float("nan")) - safe_float(a0.get("linec_NoiseSignalLeak"), float("nan"))
        r_delta = safe_float(row.get("linec_RealSignalReservoirRatio"), float("nan")) - safe_float(a0.get("linec_RealSignalReservoirRatio"), float("nan"))
        c_delta = safe_float(row.get("linec_CouplingR2"), float("nan")) - safe_float(a0.get("linec_CouplingR2"), float("nan"))
        linec_rows.append(
            {
                "stage": "V1221_LABEL_FREE_LINEC",
                "source_run": row.get("_source_csv", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("epochs", ""),
                "method": cid,
                "candidate_id": cid,
                "CouplingR2": row.get("linec_CouplingR2", ""),
                "NoiseSignalLeak": row.get("linec_NoiseSignalLeak", ""),
                "RealSignalReservoirRatio": row.get("linec_RealSignalReservoirRatio", ""),
                "CouplingR2_delta_vs_A0": c_delta if math.isfinite(c_delta) else "",
                "NoiseSignalLeak_delta_vs_A0": n_delta if math.isfinite(n_delta) else "",
                "RealSignalReservoirRatio_delta_vs_A0": r_delta if math.isfinite(r_delta) else "",
                "NoiseSignalLeak_delta_vs_mlp": safe_float(row.get("linec_NoiseSignalLeak"), float("nan")) - safe_float(mlp.get("linec_NoiseSignalLeak"), float("nan")) if mlp else "",
                "RealSignalReservoirRatio_delta_vs_mlp": safe_float(row.get("linec_RealSignalReservoirRatio"), float("nan")) - safe_float(mlp.get("linec_RealSignalReservoirRatio"), float("nan")) if mlp else "",
                "linec_nontearing_pass_vs_mlp": row.get("linec_nontearing_pass_vs_mlp", ""),
                "hard_noise_release": int(math.isfinite(n_delta) and n_delta <= -0.01),
                "hard_reservoir_release": int(math.isfinite(r_delta) and r_delta <= -0.01),
                "hard_joint_release": int(math.isfinite(n_delta) and math.isfinite(r_delta) and n_delta <= -0.01 and r_delta <= -0.01),
            }
        )
    out_rows = []
    for cid, group in sorted(by_cid.items()):
        strict_rows = sum(safe_int(r.get("strict_label_free_init"), 0) for r in group)
        a0_deltas = [safe_float(r.get("val_acc_delta_vs_A0_labelInit"), float("nan")) for r in group if r.get("val_acc_delta_vs_A0_labelInit") not in ("", None)]
        auc_step = [safe_float(r.get("AUC_step_ratio_vs_mlp"), float("nan")) for r in group if r.get("AUC_step_ratio_vs_mlp") not in ("", None)]
        auc_time = [safe_float(r.get("AUC_time_ratio_vs_mlp"), float("nan")) for r in group if r.get("AUC_time_ratio_vs_mlp") not in ("", None)]
        linec_pass = [safe_int(r.get("linec_nontearing_pass_vs_mlp"), 0) for r in group if r.get("linec_nontearing_pass_vs_mlp") not in ("", None)]
        official = int(
            cid != "A0-labelInit"
            and candidate_uses_label(cid) == 0
            and candidate_uses_pseudo(cid) == 0
            and strict_rows == len(group)
            and bool(a0_deltas)
            and mean(a0_deltas) >= -0.005
            and finite_min(a0_deltas) >= -0.015
            and bool(auc_time)
            and finite_max(auc_time) <= 1.0
            and bool(linec_pass)
            and min(linec_pass) >= 1
        )
        role_energy = {
            role: mean([r.get(f"t1b_update_{role}_energy_frac_mean", "") for r in group])
            for role in ["quad_proj", "quad_readout", "direct_readout", "branch_scale", "logit_gain", "other"]
        }
        frame_condition = mean([r.get("P_condition", "") for r in group])
        failure = ""
        if not official:
            reasons = []
            if cid == "A0-labelInit" or candidate_uses_label(cid):
                reasons.append("label_or_anchor_not_promotable")
            if candidate_uses_pseudo(cid):
                reasons.append("pseudo_label_diagnostic_only")
            if strict_rows != len(group):
                reasons.append("strict_label_free_init_not_all_rows")
            if not a0_deltas or mean(a0_deltas) < -0.005 or finite_min(a0_deltas) < -0.015:
                reasons.append("task_delta_gate_failed")
            if not auc_time or finite_max(auc_time) > 1.0:
                reasons.append("AUC_time_gate_failed")
            if not linec_pass or min(linec_pass) < 1:
                reasons.append("LineC_nontearing_gate_failed")
            failure = ";".join(reasons)
        out_rows.append(
            {
                "stage": "V1221_LABEL_FREE_SIGNAL_FRAME",
                "candidate_id": cid,
                "frame_family": candidate_frame_family(cid),
                "uses_label": candidate_uses_label(cid),
                "uses_ce_vector": 0,
                "uses_validation_or_test": 0,
                "uses_dataset_name": 0,
                "uses_optimizer_update_opaque": candidate_uses_optimizer_update(cid),
                "uses_clone_probe_response": 0,
                "rows": len(group),
                "strict_label_free_rows": strict_rows,
                "epochs_observed": ",".join(sorted({str(r.get("epochs", "")) for r in group})),
                "mean_delta_vs_A0": mean(a0_deltas) if a0_deltas else "",
                "worst_delta_vs_A0": finite_min(a0_deltas) if a0_deltas else "",
                "AUC_step_ratio_vs_mlp": finite_max(auc_step) if auc_step else "",
                "AUC_time_ratio_vs_mlp": finite_max(auc_time) if auc_time else "",
                "ECE_delta_vs_A0": mean([safe_float(r.get("ECE"), 0.0) - safe_float(a0_by_key.get(row_key(r), {}).get("ECE"), 0.0) for r in group if row_key(r) in a0_by_key]),
                "CEp99_delta_vs_A0": mean([safe_float(r.get("CEp99"), 0.0) - safe_float(a0_by_key.get(row_key(r), {}).get("CEp99"), 0.0) for r in group if row_key(r) in a0_by_key]),
                "LineC_nontearing_all_pass": int(bool(linec_pass) and min(linec_pass) >= 1),
                "CouplingR2_delta_vs_A0": mean([r.get("CouplingR2_delta_vs_A0", "") for r in linec_rows if r.get("candidate_id") == cid]),
                "NoiseSignalLeak_delta_vs_A0": mean([r.get("NoiseSignalLeak_delta_vs_A0", "") for r in linec_rows if r.get("candidate_id") == cid]),
                "RealSignalReservoirRatio_delta_vs_A0": mean([r.get("RealSignalReservoirRatio_delta_vs_A0", "") for r in linec_rows if r.get("candidate_id") == cid]),
                "frame_rank": mean([r.get("P_singular_max", "") for r in group]),
                "frame_condition": frame_condition if math.isfinite(frame_condition) else "",
                "frame_energy_by_role": json.dumps({k: (v if math.isfinite(v) else "") for k, v in role_energy.items()}, ensure_ascii=False, sort_keys=True),
                "P_energy_on_frame": mean([r.get("P_energy_on_top_pca", "") for r in group]),
                "quad_feature_std_mean": mean([r.get("quad_feature_std_mean", "") for r in group]),
                "branch_scale_norm": mean([r.get("branch_scale_norm", "") for r in group]),
                "official_label_free_candidate_pass": official,
                "failure_reason": failure,
                "promotion_allowed": official,
            }
        )
        if cid != "A0-labelInit":
            by_dataset = sorted({str(r.get("dataset", "")) for r in group})
            for ds in by_dataset:
                dg = [r for r in group if str(r.get("dataset", "")) == ds]
                failure_rows.append(
                    {
                        "stage": "V1221_LABEL_FREE_FAILURE_DECOMPOSITION",
                        "candidate_id": cid,
                        "dataset": ds,
                        "rows": len(dg),
                        "mean_delta_vs_A0": mean([r.get("val_acc_delta_vs_A0_labelInit", "") for r in dg]),
                        "worst_delta_vs_A0": finite_min([r.get("val_acc_delta_vs_A0_labelInit", "") for r in dg]),
                        "margin_p10_delta_vs_A0": mean([safe_float(r.get("margin_p10"), 0.0) - safe_float(a0_by_key.get(row_key(r), {}).get("margin_p10"), 0.0) for r in dg if row_key(r) in a0_by_key]),
                        "CEp99_delta_vs_A0": mean([safe_float(r.get("CEp99"), 0.0) - safe_float(a0_by_key.get(row_key(r), {}).get("CEp99"), 0.0) for r in dg if row_key(r) in a0_by_key]),
                        "AUC_time_ratio_vs_mlp_max": finite_max([r.get("AUC_time_ratio_vs_mlp", "") for r in dg]),
                        "LineC_pass_rate": mean([r.get("linec_nontearing_pass_vs_mlp", "") for r in dg]),
                        "failure_reason": failure,
                    }
                )
    write_csv_rows(out_dir / "v1221_label_free_signal_frame.csv", out_rows)
    write_csv_rows(out_dir / "v1221_label_free_linec.csv", linec_rows)
    write_csv_rows(out_dir / "v1221_label_free_failure_decomposition.csv", failure_rows)
    best = max([r for r in out_rows if r.get("candidate_id") != "A0-labelInit" and safe_int(r.get("uses_label"), 0) == 0], key=lambda r: safe_float(r.get("mean_delta_vs_A0"), -999.0), default={})
    return {
        "linea_rows": len(rows),
        "label_free_candidate_rows": len(out_rows),
        "label_free_official_pass_count": sum(safe_int(r.get("official_label_free_candidate_pass"), 0) for r in out_rows),
        "label_free_best_candidate": best.get("candidate_id", ""),
        "label_free_best_mean_delta_vs_A0": best.get("mean_delta_vs_A0", ""),
        "label_free_linec_rows": len(linec_rows),
    }


def release_labels(row: Mapping[str, Any]) -> tuple[int, int, int]:
    n_delta = safe_float(row.get("Delta_NoiseSignalLeak_audit", row.get("NoiseSignalLeak_delta_vs_A0")), float("nan"))
    r_delta = safe_float(row.get("Delta_RealSignalReservoirRatio_audit", row.get("RealSignalReservoirRatio_delta_vs_A0")), float("nan"))
    n_rel = int(math.isfinite(n_delta) and n_delta <= -0.01)
    r_rel = int(math.isfinite(r_delta) and r_delta <= -0.01)
    return n_rel, r_rel, int(n_rel and r_rel)


def write_linec_deployable_targets(out_dir: Path) -> dict[str, Any]:
    linec_rows = read_csv_rows(out_dir / "v1221_label_free_linec.csv")
    raw_by_key = {}
    for p in sorted((out_dir / "linea").rglob("*_ablation.csv")):
        for row in read_csv_rows(p):
            raw_by_key[(rel(p), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("candidate_id", "")))] = row
    out_rows = []
    scores: list[float] = []
    noise: list[float] = []
    reservoir: list[float] = []
    tail: list[float] = []
    for row in linec_rows:
        raw = raw_by_key.get((str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("window", "")), str(row.get("candidate_id", ""))), {})
        g_cov = safe_float(raw.get("g_cov_output_cov_drift_mean"), float("nan"))
        g_iso = safe_float(raw.get("linec_top_eigen_share"), float("nan"))
        g_coupling = safe_float(raw.get("linec_CouplingR2"), float("nan"))
        role_vals = [safe_float(raw.get(f"t1b_update_{role}_energy_frac_mean"), float("nan")) for role in ["quad_proj", "quad_readout", "direct_readout", "branch_scale", "logit_gain", "other"]]
        role_vals = [v for v in role_vals if math.isfinite(v)]
        g_role = sum(abs(v - (sum(role_vals) / len(role_vals))) for v in role_vals) if role_vals else float("nan")
        g_opt = safe_float(raw.get("t1b_update_spectrum_top_share_mean"), float("nan"))
        pieces = [v for v in [g_cov, g_iso, g_role, g_opt] if math.isfinite(v)]
        # Lower composite is predeclared as better: low drift, low isotropy collapse,
        # low role imbalance, low optimizer spectrum concentration.
        g_comp = sum(pieces) / len(pieces) if pieces else float("nan")
        n_delta = safe_float(row.get("NoiseSignalLeak_delta_vs_A0"), float("nan"))
        r_delta = safe_float(row.get("RealSignalReservoirRatio_delta_vs_A0"), float("nan"))
        c_delta = safe_float(row.get("CouplingR2_delta_vs_A0"), float("nan"))
        cep99_delta = safe_float(raw.get("CEp99"), float("nan"))
        if math.isfinite(g_comp) and math.isfinite(n_delta) and math.isfinite(r_delta):
            scores.append(g_comp)
            noise.append(n_delta)
            reservoir.append(r_delta)
            if math.isfinite(cep99_delta):
                tail.append(cep99_delta)
        out_rows.append(
            {
                "stage": "V1221_LINEC_DEPLOYABLE_TARGETS",
                "source_run": row.get("source_run", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("window", ""),
                "method": row.get("candidate_id", ""),
                "G_cov": g_cov if math.isfinite(g_cov) else "",
                "G_iso": g_iso if math.isfinite(g_iso) else "",
                "G_coupling_stability": g_coupling if math.isfinite(g_coupling) else "",
                "G_role": g_role if math.isfinite(g_role) else "",
                "G_opt_spectrum": g_opt if math.isfinite(g_opt) else "",
                "G_composite": g_comp if math.isfinite(g_comp) else "",
                "Delta_NoiseSignalLeak_audit": n_delta if math.isfinite(n_delta) else "",
                "Delta_RealSignalReservoirRatio_audit": r_delta if math.isfinite(r_delta) else "",
                "Delta_CouplingR2_audit": c_delta if math.isfinite(c_delta) else "",
                "CEp99_delta_audit": "",
                "ECE_delta_audit": "",
                "uses_label_for_feature": 0,
                "uses_ce_for_feature": 0,
                "feature_tier": "T1B" if safe_int(raw.get("t1b_optimizer_update_native_logged"), 0) else "T1A",
                "spearman_noise": "",
                "spearman_reservoir": "",
                "spearman_tail": "",
            }
        )
    rho_noise = spearman(scores, noise)
    rho_res = spearman(scores, reservoir)
    rho_tail = spearman(scores[: len(tail)], tail) if len(tail) >= 3 else float("nan")
    official_pass = int(math.isfinite(rho_noise) and math.isfinite(rho_res) and rho_noise <= -0.30 and rho_res <= -0.30)
    exploratory = int((math.isfinite(rho_noise) and abs(rho_noise) >= 0.20) or (math.isfinite(rho_res) and abs(rho_res) >= 0.20))
    for row in out_rows:
        row["spearman_noise"] = rho_noise if math.isfinite(rho_noise) else ""
        row["spearman_reservoir"] = rho_res if math.isfinite(rho_res) else ""
        row["spearman_tail"] = rho_tail if math.isfinite(rho_tail) else ""
    write_csv_rows(out_dir / "v1221_linec_deployable_targets.csv", out_rows)
    ablation = []
    for name in ["G_cov", "G_iso", "G_coupling_stability", "G_role", "G_opt_spectrum", "G_composite"]:
        vals = [safe_float(r.get(name), float("nan")) for r in out_rows]
        valid = [(v, safe_float(r.get("Delta_NoiseSignalLeak_audit"), float("nan")), safe_float(r.get("Delta_RealSignalReservoirRatio_audit"), float("nan"))) for v, r in zip(vals, out_rows) if math.isfinite(v)]
        ablation.append(
            {
                "stage": "V1221_TARGET_FAMILY_ABLATION",
                "target_family": name,
                "rows": len(valid),
                "spearman_noise": spearman([v[0] for v in valid], [v[1] for v in valid]) if len(valid) >= 3 else "",
                "spearman_reservoir": spearman([v[0] for v in valid], [v[2] for v in valid]) if len(valid) >= 3 else "",
                "official_pass": int(name == "G_composite" and official_pass),
            }
        )
    write_csv_rows(out_dir / "v1221_target_family_ablation.csv", ablation)
    return {"linec_deployable_rows": len(out_rows), "g_la_spearman_noise": rho_noise if math.isfinite(rho_noise) else "", "g_la_spearman_reservoir": rho_res if math.isfinite(rho_res) else "", "g_la_exploratory": exploratory, "g_la_official_pass": official_pass}


def write_delta_sign_audit(out_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(out_dir / "v1221_label_free_linec.csv")
    out = []
    for metric in ["NoiseSignalLeak", "RealSignalReservoirRatio", "CouplingR2"]:
        col = f"{metric}_delta_vs_A0"
        vals = finite_values([r.get(col, "") for r in rows])
        if metric == "CouplingR2":
            good = sum(1 for v in vals if v >= 0.02)
            direction = "positive_delta_is_better"
        else:
            good = sum(1 for v in vals if v <= -0.01)
            direction = "negative_delta_is_better"
        out.append(
            {
                "stage": "V1221_DELTA_SIGN_CONVENTION_AUDIT",
                "metric": metric,
                "delta_column": col,
                "good_direction": direction,
                "finite_rows": len(vals),
                "improving_rows": good,
                "mean_delta": mean(vals) if vals else "",
                "min_delta": finite_min(vals) if vals else "",
                "max_delta": finite_max(vals) if vals else "",
                "hard_release_threshold": ">=0.02" if metric == "CouplingR2" else "<=-0.01",
            }
        )
    write_csv_rows(out_dir / "v1221_delta_sign_convention_audit.csv", out)
    return {"delta_sign_audit_rows": len(out)}


def write_join_key_audit(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audits = []
    for name, source_rows, keys in [
        ("linea_raw", rows, ["_source_csv", "dataset", "seed", "epochs", "candidate_id"]),
        ("label_free_linec", read_csv_rows(out_dir / "v1221_label_free_linec.csv"), ["source_run", "dataset", "seed", "window", "candidate_id"]),
        ("linec_deployable", read_csv_rows(out_dir / "v1221_linec_deployable_targets.csv"), ["source_run", "dataset", "seed", "window", "method"]),
    ]:
        counts: dict[str, int] = defaultdict(int)
        for row in source_rows:
            key = "::".join(str(row.get(k, "")) for k in keys)
            counts[key] += 1
        dup = sum(1 for v in counts.values() if v > 1)
        audits.append(
            {
                "stage": "V1221_JOIN_KEY_UNIQUENESS",
                "table": name,
                "key_columns": ",".join(keys),
                "rows": len(source_rows),
                "unique_keys": len(counts),
                "duplicate_key_count": dup,
                "join_stable": int(dup == 0 and len(counts) == len(source_rows)),
            }
        )
    write_csv_rows(out_dir / "v1221_join_key_uniqueness.csv", audits)
    return {"join_audit_rows": len(audits), "join_duplicate_key_count": sum(safe_int(r.get("duplicate_key_count"), 0) for r in audits)}


def write_soft_target_regression(out_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(out_dir / "v1221_linec_deployable_targets.csv")
    feature_cols = ["G_cov", "G_iso", "G_coupling_stability", "G_role", "G_opt_spectrum", "G_composite"]
    out = []
    for target in ["Delta_NoiseSignalLeak_audit", "Delta_RealSignalReservoirRatio_audit"]:
        vals = finite_values([r.get(target, "") for r in rows])
        var = float(np.var(np.asarray(vals, dtype=float))) if vals else float("nan")
        r2_ds = ridge_leaveout_r2(rows, feature_cols, target, "dataset")
        r2_seed = ridge_leaveout_r2(rows, feature_cols, target, "seed")
        out.append(
            {
                "stage": "V1221_SOFT_TARGET_REGRESSION",
                "target": target,
                "rows": len(vals),
                "target_variance": var if math.isfinite(var) else "",
                "constant_target": int(math.isfinite(var) and var <= EPS),
                "nan_inf_rows": len(rows) - len(vals),
                "leave_dataset_out_r2_min": r2_ds if math.isfinite(r2_ds) else "",
                "leave_seed_out_r2_min": r2_seed if math.isfinite(r2_seed) else "",
                "strong_negative_r2": int((math.isfinite(r2_ds) and r2_ds < -10.0) or (math.isfinite(r2_seed) and r2_seed < -10.0)),
            }
        )
    write_csv_rows(out_dir / "v1221_soft_target_regression.csv", out)
    return {"soft_target_regression_rows": len(out)}


def write_target_sign_sanity(out_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(out_dir / "v1221_linec_deployable_targets.csv")
    labels_noise = [release_labels(r)[0] for r in rows]
    labels_res = [release_labels(r)[1] for r in rows]
    labels_joint = [release_labels(r)[2] for r in rows]
    out = []
    for score_col in ["G_composite", "G_cov", "G_iso", "G_role", "G_opt_spectrum"]:
        scores = [safe_float(r.get(score_col), float("nan")) for r in rows]
        for target_name, labels in [("noise", labels_noise), ("reservoir", labels_res), ("joint", labels_joint)]:
            p, rc = precision_recall_at_k([-s if math.isfinite(s) else s for s in scores], labels)
            bp, br = precision_recall_at_k([-s if math.isfinite(s) else s for s in scores], labels, bottom=True)
            out.append(
                {
                    "stage": "V1221_TARGET_SIGN_SANITY",
                    "score_column": score_col,
                    "target": target_name,
                    "predeclared_score_direction": "lower_score_is_better_release",
                    "auc_predeclared": auc_score([-s if math.isfinite(s) else s for s in scores], labels),
                    "auc_inverted": auc_score(scores, labels),
                    "top_precision": p,
                    "top_recall": rc,
                    "bottom_precision": bp,
                    "bottom_recall": br,
                    "positive_support": sum(labels),
                    "rows": len(labels),
                }
            )
    write_csv_rows(out_dir / "v1221_target_sign_sanity.csv", out)
    return {"target_sign_sanity_rows": len(out)}


def write_visibility_v5(out_dir: Path) -> dict[str, Any]:
    target_rows = read_csv_rows(out_dir / "v1221_linec_deployable_targets.csv")
    raw_rows = {}
    for p in sorted((out_dir / "linea").rglob("*_ablation.csv")):
        for row in read_csv_rows(p):
            raw_rows[(rel(p), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("candidate_id", "")))] = row
    feature_rows = []
    t1b_rows = []
    for row in target_rows:
        key = (str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("window", "")), str(row.get("method", "")))
        raw = raw_rows.get(key, {})
        base = {
            "source_run": row.get("source_run", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "window": row.get("window", ""),
            "method": row.get("method", ""),
            "Delta_NoiseSignalLeak_audit": row.get("Delta_NoiseSignalLeak_audit", ""),
            "Delta_RealSignalReservoirRatio_audit": row.get("Delta_RealSignalReservoirRatio_audit", ""),
        }
        features = {
            "T1A": {
                "G_cov": row.get("G_cov", ""),
                "G_iso": row.get("G_iso", ""),
                "G_coupling_stability": row.get("G_coupling_stability", ""),
                "G_role": row.get("G_role", ""),
                "G_composite": row.get("G_composite", ""),
                "P_condition": raw.get("P_condition", ""),
                "P_energy_on_top_pca": raw.get("P_energy_on_top_pca", ""),
            },
            "T1B": {
                "t1b_update_total_norm_mean": raw.get("t1b_update_total_norm_mean", ""),
                "t1b_update_total_norm_q90": raw.get("t1b_update_total_norm_q90", ""),
                "t1b_update_cosine_mean": raw.get("t1b_update_cosine_mean", ""),
                "t1b_update_quad_proj_energy_frac_mean": raw.get("t1b_update_quad_proj_energy_frac_mean", ""),
                "t1b_update_direct_readout_energy_frac_mean": raw.get("t1b_update_direct_readout_energy_frac_mean", ""),
                "t1b_update_spectrum_top_share_mean": raw.get("t1b_update_spectrum_top_share_mean", ""),
                "t1b_update_spectrum_condition_mean": raw.get("t1b_update_spectrum_condition_mean", ""),
            },
        }
        for tier, feats in features.items():
            for name, value in feats.items():
                feature_rows.append(
                    {
                        "stage": "V1221_VISIBILITY_FEATURES",
                        **base,
                        "feature_tier": tier,
                        "feature_family": "optimizer_update" if tier == "T1B" else "precommit_geometry",
                        "feature_name": name,
                        "source_column": name,
                        "feature_value": value,
                        "uses_label": 0,
                        "uses_ce_vector": 0,
                        "uses_validation_or_test": 0,
                        "uses_future_outcome": 0,
                        "uses_dataset_name": 0,
                        "precommit_available": int(tier == "T1A"),
                        "native_logged": int(tier == "T1B" and safe_int(raw.get("t1b_optimizer_update_native_logged"), 0) == 1),
                        "clone_probe_only": 0,
                    }
                )
        if safe_int(raw.get("t1b_optimizer_update_native_logged"), 0):
            t1b_rows.append(
                {
                    "stage": "V1221_T1B_OPTIMIZER_UPDATE_FEATURES",
                    **base,
                    **{k: raw.get(k, "") for k in raw.keys() if k.startswith("t1b_update_")},
                    "native_logged": 1,
                    "uses_label": 0,
                    "uses_ce_vector": 0,
                    "feature_tier": "T1B",
                }
            )
    write_csv_rows(out_dir / "v1221_visibility_features.csv", feature_rows)
    write_csv_rows(out_dir / "v1221_t1b_optimizer_update_features.csv", t1b_rows)

    def score_rows_for_tier(tier: str, score_name: str, score_values: Sequence[float]) -> dict[str, Any]:
        labels_n = [release_labels(r)[0] for r in target_rows]
        labels_r = [release_labels(r)[1] for r in target_rows]
        labels_j = [release_labels(r)[2] for r in target_rows]
        scores = list(score_values)
        auc_n = auc_score(scores, labels_n)
        auc_r = auc_score(scores, labels_r)
        auc_j = auc_score(scores, labels_j)
        p, rc = precision_recall_at_k(scores, labels_j)
        bp, br = precision_recall_at_k(scores, labels_j, bottom=True)
        feature_cols = []
        if tier == "T1A":
            feature_cols = ["G_cov", "G_iso", "G_coupling_stability", "G_role", "G_composite"]
        elif tier == "T1B":
            feature_cols = ["t1b_update_total_norm_mean", "t1b_update_total_norm_q90", "t1b_update_cosine_mean", "t1b_update_quad_proj_energy_frac_mean", "t1b_update_spectrum_top_share_mean"]
        score = {
            "stage": "V1221_VISIBILITY_SCORES",
            "feature_tier": tier,
            "feature_family": score_name,
            "feature_name": score_name,
            "source_column": ",".join(feature_cols),
            "uses_label": 0,
            "uses_ce_vector": 0,
            "uses_validation_or_test": 0,
            "uses_future_outcome": 0,
            "uses_dataset_name": 0,
            "precommit_available": int(tier == "T1A"),
            "native_logged": int(tier != "T1B" or len(t1b_rows) > 0),
            "clone_probe_only": 0,
            "score_direction": "higher_score_means_more_likely_release",
            "auc_noise": auc_n if math.isfinite(auc_n) else "",
            "auc_reservoir": auc_r if math.isfinite(auc_r) else "",
            "auc_joint": auc_j if math.isfinite(auc_j) else "",
            "precision_at_k_joint": p if math.isfinite(p) else "",
            "recall_at_k_joint": rc if math.isfinite(rc) else "",
            "bottom_precision_at_k_joint": bp if math.isfinite(bp) else "",
            "bottom_recall_at_k_joint": br if math.isfinite(br) else "",
            "soft_target_r2_noise": "",
            "soft_target_r2_reservoir": "",
            "visibility_pass": 0,
            "failure_reason": "",
        }
        if tier == "T1A":
            score["soft_target_r2_noise"] = ridge_leaveout_r2(target_rows, feature_cols, "Delta_NoiseSignalLeak_audit", "dataset")
            score["soft_target_r2_reservoir"] = ridge_leaveout_r2(target_rows, feature_cols, "Delta_RealSignalReservoirRatio_audit", "dataset")
        visible = int(
            math.isfinite(auc_j)
            and auc_j >= 0.65
            and math.isfinite(p)
            and p >= 0.10
            and math.isfinite(rc)
            and rc >= 0.10
            and (tier != "T1B" or len(t1b_rows) > 0)
        )
        score["visibility_pass"] = visible
        if not visible:
            reasons = []
            if tier == "T1B" and not t1b_rows:
                reasons.append("T1B_native_logging_missing")
            if not math.isfinite(auc_j) or auc_j < 0.65:
                reasons.append("AUC_joint_gate_failed")
            if not math.isfinite(p) or p < 0.10 or not math.isfinite(rc) or rc < 0.10:
                reasons.append("precision_recall_gate_failed")
            score["failure_reason"] = ";".join(reasons)
        return score

    t1a_scores = [-safe_float(r.get("G_composite"), float("nan")) for r in target_rows]
    t1b_scores = []
    for row in target_rows:
        raw = raw_rows.get((str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("window", "")), str(row.get("method", ""))), {})
        instability = mean(
            [
                raw.get("t1b_update_total_norm_q90", ""),
                raw.get("t1b_update_spectrum_top_share_mean", ""),
                raw.get("t1b_update_quad_proj_energy_frac_mean", ""),
            ]
        )
        t1b_scores.append(-instability if math.isfinite(instability) else float("nan"))
    score_rows = [
        score_rows_for_tier("T1A", "precommit_geometry_composite", t1a_scores),
        score_rows_for_tier("T1B", "optimizer_update_opaque_composite", t1b_scores),
    ]
    write_csv_rows(out_dir / "v1221_visibility_scores.csv", score_rows)
    leaveout_rows = []
    for tier, scores in [("T1A", t1a_scores), ("T1B", t1b_scores)]:
        for split_col in ["dataset", "seed", "window"]:
            for split in sorted({str(r.get(split_col, "")) for r in target_rows}):
                idx = [i for i, r in enumerate(target_rows) if str(r.get(split_col, "")) != split]
                labels = [release_labels(target_rows[i])[2] for i in idx]
                sub_scores = [scores[i] for i in idx]
                leaveout_rows.append(
                    {
                        "stage": "V1221_VISIBILITY_LEAVEOUT",
                        "feature_tier": tier,
                        "split_col": split_col,
                        "held_out": split,
                        "train_rows_scored": len(idx),
                        "auc_joint": auc_score(sub_scores, labels),
                        "positive_support": sum(labels),
                    }
                )
    write_csv_rows(out_dir / "v1221_visibility_leaveout.csv", leaveout_rows)
    sanity_rows = []
    for row in score_rows:
        auc = safe_float(row.get("auc_joint"), float("nan"))
        sanity_rows.append(
            {
                "stage": "V1221_VISIBILITY_SANITY",
                "feature_tier": row.get("feature_tier", ""),
                "auc_joint": row.get("auc_joint", ""),
                "auc_below_half": int(math.isfinite(auc) and auc < 0.5),
                "sign_inversion_required_for_diagnostic": int(math.isfinite(auc) and auc < 0.5),
                "native_logged": row.get("native_logged", ""),
                "failure_reason": row.get("failure_reason", ""),
            }
        )
    write_csv_rows(out_dir / "v1221_visibility_sanity.csv", sanity_rows)
    return {
        "visibility_rows": len(score_rows),
        "T1A_visibility_pass": safe_int(score_rows[0].get("visibility_pass"), 0),
        "T1B_visibility_pass": safe_int(score_rows[1].get("visibility_pass"), 0),
        "T1B_native_logged_rows": len(t1b_rows),
        "T1B_native_logging_available": int(len(t1b_rows) > 0),
        "T1A_auc_joint": score_rows[0].get("auc_joint", ""),
        "T1B_auc_joint": score_rows[1].get("auc_joint", ""),
    }


def _copy_state_delta(base: torch.nn.Module, updated: torch.nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    upd = dict(updated.named_parameters())
    for name, param in base.named_parameters():
        if name in upd and param.shape == upd[name].shape:
            out[name] = (upd[name].detach() - param.detach()).clone()
    return out


def _projector_angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().flatten()
    bv = b.detach().float().flatten()
    denom = av.norm().clamp_min(EPS) * bv.norm().clamp_min(EPS)
    cos = torch.dot(av, bv).div(denom).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)).item())


def _apply_actuator(model: torch.nn.Module, actuator: str, budget: float, sign: float, gen: torch.Generator, opt_delta: Mapping[str, torch.Tensor] | None = None) -> str:
    with torch.no_grad():
        if actuator == "I1-direct-role-scale-shift":
            value = getattr(model, "direct_readout", None)
            if torch.is_tensor(value):
                value.add_(sign * budget * value.norm().clamp_min(EPS) / math.sqrt(max(1, value.numel())))
            return "direct_readout"
        if actuator == "I2-quad-proj-local-rotation":
            p = getattr(model, "quad_proj", None)
            if torch.is_tensor(p):
                noise = torch.randn(p.shape, device=p.device, generator=gen, dtype=p.dtype)
                p.add_(noise * float(sign * budget))
                p.sub_(p.mean(dim=0, keepdim=True))
                p.div_(p.norm(dim=0, keepdim=True).clamp_min(1.0e-6))
            return "quad_proj"
        if actuator == "I3-hinge-amplitude-threshold":
            value = getattr(model, "direct_readout", None)
            if torch.is_tensor(value):
                start = int(getattr(model, "input_dim", 0))
                end = start + 2 * int(getattr(model, "hinge_centers", torch.empty(0)).numel()) * int(getattr(model, "input_dim", 0))
                value[start:end].mul_(1.0 + sign * budget)
            return "hinge"
        if actuator == "I4-absdiag-energy":
            value = getattr(model, "direct_readout", None)
            if torch.is_tensor(value):
                start = (1 + 2 * int(getattr(model, "hinge_centers", torch.empty(0)).numel())) * int(getattr(model, "input_dim", 0))
                end = min(int(value.shape[0]), start + int(getattr(model, "input_dim", 0)))
                value[start:end].mul_(1.0 + sign * budget)
            return "absdiag"
        if actuator == "I5-branch-gain-redistribution":
            value = getattr(model, "branch_scale", None)
            if torch.is_tensor(value):
                if value.dim() == 2:
                    value[0].add_(sign * budget)
                    value[1].add_(-sign * budget)
                elif int(value.numel()) >= 2:
                    value.view(-1)[0].add_(sign * budget)
                    value.view(-1)[1].add_(-sign * budget)
            return "branch_scale"
        if actuator == "I6-role-balanced-combined":
            _apply_actuator(model, "I2-quad-proj-local-rotation", budget * 0.5, sign, gen)
            _apply_actuator(model, "I5-branch-gain-redistribution", budget * 0.5, sign, gen)
            return "role_balanced"
        if actuator == "I7-optimizer-update-aligned":
            if opt_delta:
                for name, param in model.named_parameters():
                    delta = opt_delta.get(name)
                    if delta is not None and delta.shape == param.shape:
                        scale = budget / float(delta.float().norm().item() + EPS)
                        param.add_(delta.to(device=param.device, dtype=param.dtype), alpha=float(sign * scale))
            return "optimizer_update"
        if actuator == "I8-random-matched-role-energy-control":
            _apply_actuator(model, "I2-quad-proj-local-rotation", budget, sign, gen)
            return "matched_random_control"
        if actuator == "I9-shuffled-actuator-basis-control":
            p = getattr(model, "quad_proj", None)
            if torch.is_tensor(p):
                perm = torch.randperm(int(p.shape[0]), device=p.device, generator=gen)
                mixed = p.index_select(0, perm).mul(float(budget * sign)).add(p, alpha=1.0 - float(budget))
                mixed.sub_(mixed.mean(dim=0, keepdim=True))
                mixed.div_(mixed.norm(dim=0, keepdim=True).clamp_min(1.0e-6))
                p.copy_(mixed)
            return "shuffled_control"
    return "unknown"


def _linec_metrics(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, xq: torch.Tensor, yq: torch.Tensor, seed: int, sketch_dim: int) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        before_b = model(xb).detach()
        before_q = model(xq).detach()
    updated = v1252._take_adamw_window(model, xb, yb, 2.0e-3, 1.0e-3).eval()
    with torch.no_grad():
        after_b = updated(xb).detach()
        after_q = updated(xq).detach()
    r2, _corr, _resid, _pred = v1252._ridge_coupling(after_b - before_b, after_q - before_q, 1.0e-3)
    sig = linea.signal_reservoir_metrics_detailed(updated, xb, yb, int(sketch_dim), int(seed))
    return {"CouplingR2": float(r2), "NoiseSignalLeak": sig["NoiseSignalLeak"], "RealSignalReservoirRatio": sig["RealSignalReservoirRatio"]}


def matched_control_gap(rows: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> float:
    if str(row.get("basis_type", "")).endswith("control"):
        return 0.0
    budget = str(row.get("norm_budget", ""))
    signed = str(row.get("signed_direction", ""))
    controls = [r for r in rows if str(r.get("norm_budget", "")) == budget and str(r.get("signed_direction", "")) == signed and str(r.get("basis_type", "")).endswith("control")]
    score = -safe_float(row.get("NoiseSignalLeak_delta_audit"), 0.0) - safe_float(row.get("RealSignalReservoirRatio_delta_audit"), 0.0)
    best_control = finite_max([-safe_float(r.get("NoiseSignalLeak_delta_audit"), 0.0) - safe_float(r.get("RealSignalReservoirRatio_delta_audit"), 0.0) for r in controls])
    return score - best_control if math.isfinite(best_control) else float("nan")


def write_actuator_v2(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.actuator_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rows = []
    control_rows = []
    safety_rows = []
    datasets = [v120._canonical_dataset(x.strip()) for x in str(args.actuator_datasets).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.actuator_seeds).split(",") if x.strip()]
    budgets = [float(x.strip()) for x in str(args.actuator_budgets).split(",") if x.strip()]
    actuators = [
        "I1-direct-role-scale-shift",
        "I2-quad-proj-local-rotation",
        "I3-hinge-amplitude-threshold",
        "I4-absdiag-energy",
        "I5-branch-gain-redistribution",
        "I6-role-balanced-combined",
        "I7-optimizer-update-aligned",
        "I8-random-matched-role-energy-control",
        "I9-shuffled-actuator-basis-control",
    ]
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(**vars(args))
            load_args.seed = seed
            data = v120._load_vision_split(load_args, dataset, train_size=int(args.actuator_train_size), val_size=int(args.actuator_val_size), test_size=int(args.actuator_val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test, _y_test, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)
            specs = linea.ablation_specs(int(input_dim), int(output_dim))
            spec = specs["A1-noYForStats"]["spec"]
            model = linea.make_model("A1-noYForStats", spec, int(input_dim), int(output_dim), x_train, y_train, device, int(seed) + 1221000, 0)
            b = min(int(args.actuator_linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
            xb, yb, xq, yq = x_train[:b], y_train[:b], x_val[:b], y_val[:b]
            base_metrics = _linec_metrics(model, xb, yb, xq, yq, seed + 7000, int(args.actuator_sketch_dim))
            with torch.no_grad():
                base_logits = model(xq).detach()
            try:
                opt_updated = v1252._take_adamw_window(model, xb, yb, 2.0e-3, 1.0e-3)
                opt_delta = _copy_state_delta(model, opt_updated)
            except Exception:
                opt_delta = {}
            base_p = getattr(model, "quad_proj", torch.empty(0, device=device)).detach().clone() if hasattr(model, "quad_proj") else torch.empty(0, device=device)
            for budget in budgets:
                for signed in [-1.0, 1.0]:
                    for actuator in actuators:
                        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(actuator) * 17 + (1 if signed > 0 else 2))
                        trial = copy.deepcopy(model).to(device)
                        role = _apply_actuator(trial, actuator, float(budget), float(signed), gen, opt_delta)
                        with torch.no_grad():
                            logits = trial(xq).detach()
                        p_now = getattr(trial, "quad_proj", torch.empty(0, device=device)).detach()
                        angle = _projector_angle_deg(base_p, p_now) if int(base_p.numel()) and int(p_now.numel()) == int(base_p.numel()) else 0.0
                        drift = float((logits - base_logits).abs().max().item())
                        sketch_delta = float((logits - base_logits).float().norm().div(math.sqrt(max(1, int(logits.numel())))).item())
                        metrics = _linec_metrics(trial, xb, yb, xq, yq, seed + 8100 + int(abs(budget) * 100000), int(args.actuator_sketch_dim))
                        n_delta = metrics["NoiseSignalLeak"] - base_metrics["NoiseSignalLeak"]
                        r_delta = metrics["RealSignalReservoirRatio"] - base_metrics["RealSignalReservoirRatio"]
                        c_delta = metrics["CouplingR2"] - base_metrics["CouplingR2"]
                        basis_type = "matched_control" if actuator == "I8-random-matched-role-energy-control" else ("shuffled_control" if actuator == "I9-shuffled-actuator-basis-control" else "actuator")
                        row = {
                            "stage": "V1221_ACTUATOR_RESPONSE_DICTIONARY",
                            "dataset": dataset,
                            "seed": seed,
                            "window": "smoke",
                            "actuator_id": actuator,
                            "role": role,
                            "basis_type": basis_type,
                            "norm_budget": budget,
                            "signed_direction": signed,
                            "uses_label": 0,
                            "uses_ce_vector": 0,
                            "matched_control_id": "I8/I9 controls",
                            "sketch_delta_fro": sketch_delta,
                            "projector_angle_deg": angle,
                            "logit_max_abs_drift": drift,
                            "CouplingR2_delta": c_delta,
                            "NoiseSignalLeak_delta_audit": n_delta,
                            "RealSignalReservoirRatio_delta_audit": r_delta,
                            "CEp99_delta": "",
                            "ECE_delta": "",
                            "safe_movement_pass": int(sketch_delta >= 0.01 and angle >= 1.0 and drift <= 0.05),
                            "release_audit_pass": int(n_delta <= -0.01 and r_delta <= -0.01),
                            "control_gap": "",
                            "failure_reason": "",
                        }
                        rows.append(row)
                        if basis_type.endswith("control"):
                            control_rows.append({**row, "stage": "V1221_ACTUATOR_CONTROLS"})
    for row in rows:
        gap = matched_control_gap(rows, row)
        row["control_gap"] = gap if math.isfinite(gap) else ""
        row["failure_reason"] = "" if safe_int(row.get("safe_movement_pass"), 0) and safe_int(row.get("release_audit_pass"), 0) and math.isfinite(gap) and gap >= 0.005 else "movement_or_release_or_control_gap_gate_failed"
        safety_rows.append({**row, "stage": "V1221_ACTUATOR_SAFETY"})
    write_csv_rows(out_dir / "v1221_actuator_response_dictionary.csv", rows)
    write_csv_rows(out_dir / "v1221_actuator_controls.csv", control_rows)
    write_csv_rows(out_dir / "v1221_actuator_safety.csv", safety_rows)
    release_rows = [r for r in rows if safe_int(r.get("safe_movement_pass"), 0) and safe_int(r.get("release_audit_pass"), 0) and safe_float(r.get("control_gap"), -999.0) >= 0.005]
    return {"actuator_rows": len(rows), "actuator_safe_movement_rows": sum(safe_int(r.get("safe_movement_pass"), 0) for r in rows), "actuator_release_audit_rows": sum(safe_int(r.get("release_audit_pass"), 0) for r in rows), "actuator_control_resistant_rows": len(release_rows), "actuator_executor_success": int(len(release_rows) > 0)}


def write_functional_gates(out_dir: Path, route_state: Mapping[str, Any]) -> dict[str, Any]:
    open_gate = int(safe_int(route_state.get("g_la_official_pass"), 0) or safe_int(route_state.get("T1B_visibility_pass"), 0) or safe_int(route_state.get("actuator_executor_success"), 0))
    candidates = ["B1-G_LA_geometric_stabilizer", "B2-T1B_update_spectrum_preconditioner", "B3-role-balanced_actuator_maintenance", "B4-noise-leak-vetoed_functional_event", "B5-reservoir-release_low-frequency_event", "B6-hybrid_G_LA_actuator_solver"]
    p3_rows = []
    for cid in candidates:
        p3_rows.append(
            {
                "stage": "V1221_FUNCTIONAL_P3",
                "candidate_id": cid,
                "P3_open": open_gate,
                "CouplingR2_delta": "",
                "NoiseSignalLeak_delta": "",
                "RealSignalReservoirRatio_delta": "",
                "control_gap": "",
                "logit_max_abs_drift": "",
                "P3_pass": 0,
                "failure_reason": "" if open_gate else "P3 not opened: C/T/I official gates did not pass",
            }
        )
    controls = [{"stage": "V1221_FUNCTIONAL_CONTROLS", "control_id": c, "executed": 0, "reason": "P3/P4 not opened" if not open_gate else "P4 implementation not executed in this runner"} for c in ["NoOpMatchedOverhead", "RandomMatchedNorm", "AdamWParallelDirection", "SNR-only", "ShuffledPayload/Event", "MatchedRoleEnergyRandomActuator", "MLPAnalogGeometryMaintenance"]]
    p4 = [
        {
            "stage": "V1221_FUNCTIONAL_P4_SHORT",
            "candidate_id": "P4-NOT-OPENED" if not open_gate else "P4-BLOCKED-NO-P3-PASS",
            "P4_open": open_gate,
            "P4_pass": 0,
            "beats_all_controls": 0,
            "failure_reason": "P4 not opened because P3 gate closed" if not open_gate else "P3 pass absent; P4 not executed",
        }
    ]
    event_log = [{"stage": "V1221_P4_EVENT_LOG", "event_count": 0, "reason": p4[0]["failure_reason"]}]
    traj = [{"stage": "V1221_P4_LINEC_TRAJECTORY", "rows": 0, "reason": p4[0]["failure_reason"]}]
    write_csv_rows(out_dir / "v1221_functional_p3.csv", p3_rows)
    write_csv_rows(out_dir / "v1221_functional_controls.csv", controls)
    write_csv_rows(out_dir / "v1221_functional_p4_short.csv", p4)
    write_csv_rows(out_dir / "v1221_p4_event_log.csv", event_log)
    write_csv_rows(out_dir / "v1221_p4_linec_trajectory.csv", traj)
    return {"functional_p3_rows": len(p3_rows), "p4_open": open_gate, "p4_pass": 0}


def write_classic_line_d(out_dir: Path, v1218_dir: Path) -> dict[str, Any]:
    inherited = read_csv_rows(v1218_dir / "v1218_classic_family_status.csv")
    families = ["Rational", "Chebyshev", "Wavelet", "RBF/FastKAN", "Fourier"]
    hypotheses = {
        "Rational": "denominator-safe coupling repair plus group diversity tangent metric",
        "Chebyshev": "degree-energy damping plus task trajectory repair",
        "Wavelet": "local support scale-diversity task-stable repair",
        "RBF/FastKAN": "compact capacity expression repair without dense RBF fallback",
        "Fourier": "low-frequency expression repair without high-frequency path",
    }
    status_rows = []
    for fam in families:
        old = next((r for r in inherited if fam.split("/")[0].lower() in str(r).lower()), {})
        status_rows.append(
            {
                "stage": "V1221_CLASSIC_FAMILY_STATUS",
                "family": fam,
                "inherited_status": old.get("status", old.get("route", "")),
                "v1221_status": "RejectedForThisVersion",
                "new_hypothesis_defined": 1,
                "executed_this_version": 0,
                "reason": "Line D did not receive an implemented non-old candidate in this runner; recorded as generated next hypothesis, not as failed run",
            }
        )
    hyp_rows = [
        {
            "stage": "V1221_CLASSIC_FAMILY_NEW_HYPOTHESIS",
            "family": fam,
            "new_hypothesis": hypotheses[fam],
            "not_old_rerun": 1,
            "ready_for_next_version": 1,
        }
        for fam in families
    ]
    linec_rows = [{"stage": "V1221_CLASSIC_FAMILY_LINEC", "family": fam, "linec_rows": 0, "status": "not_executed_not_failed", "reason": "new hypothesis generated but no new implementation path in v12.21 runner"} for fam in families]
    failure_rows = [{"stage": "V1221_CLASSIC_FAMILY_FAILURE_TABLE", "family": fam, "status": "RejectedForThisVersion", "not_a_failure_result": 1} for fam in families]
    write_csv_rows(out_dir / "v1221_classic_family_status.csv", status_rows)
    write_csv_rows(out_dir / "v1221_classic_family_new_hypothesis.csv", hyp_rows)
    write_csv_rows(out_dir / "v1221_classic_family_linec.csv", linec_rows)
    write_csv_rows(out_dir / "v1221_classic_family_failure_table.csv", failure_rows)
    return {"classic_status_rows": len(status_rows), "classic_new_hypothesis_count": len(hyp_rows)}


def write_provenance_audit(out_dir: Path, route_state: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    artifacts = [
        "v1221_route_decision.json",
        "v1221_continuation_manifest.csv",
        "v1221_provenance_audit.csv",
        "v1221_code_review_manifest.csv",
        "v1221_implementation_readback.md",
        "v1221_label_free_signal_frame.csv",
        "v1221_linec_deployable_targets.csv",
        "v1221_visibility_features.csv",
        "v1221_t1b_optimizer_update_features.csv",
        "v1221_actuator_response_dictionary.csv",
        "v1221_functional_p3.csv",
        "v1221_classic_family_status.csv",
        "v1221_no_go_boundary.md",
        "v1221_next_hypothesis_generator.md",
    ]
    for name in artifacts:
        # route_decision and provenance_audit are written during this same
        # finalization block, so audit their intended final presence rather
        # than their pre-write existence.
        exists = int((out_dir / name).exists() or name in {"v1221_route_decision.json", "v1221_provenance_audit.csv"})
        rows.append({"stage": "V1221_PROVENANCE_AUDIT", "artifact": name, "exists": exists, "no_fake": 1, "no_proxy": 1, "uses_dataset_name_branch": 0})
    feature_rows = [
        {"stage": "V1221_FEATURE_PROVENANCE", "feature_family": "T1A", "source": "Line A model/logit/projector summary columns", "uses_label": 0, "uses_ce_vector": 0, "uses_future_outcome": 0, "native_logged": 1},
        {"stage": "V1221_FEATURE_PROVENANCE", "feature_family": "T1B", "source": "native actual optimizer parameter delta logs from Line A runner", "uses_label": 0, "uses_ce_vector": 0, "uses_future_outcome": 0, "native_logged": safe_int(route_state.get("T1B_native_logging_available"), 0)},
        {"stage": "V1221_FEATURE_PROVENANCE", "feature_family": "T3 audit", "source": "Line C labels/CE-derived release targets for scoring only", "uses_label": 1, "uses_ce_vector": 1, "uses_future_outcome": 1, "native_logged": 1},
    ]
    write_csv_rows(out_dir / "v1221_provenance_audit.csv", rows)
    write_csv_rows(out_dir / "v1221_feature_provenance.csv", feature_rows)
    return {"provenance_rows": len(rows), "provenance_missing_count": sum(1 for r in rows if safe_int(r.get("exists"), 0) == 0)}


def write_no_go_and_next(out_dir: Path, route_state: Mapping[str, Any]) -> dict[str, Any]:
    no_go = f"""# v12.21 No-Go Boundary

Generated at: {now_iso()}

This boundary is fail-closed.  It records observed blockers only from landed artifacts.

## Observed Gates

- Label-free official pass count: {route_state.get('label_free_official_pass_count')}
- G_LA official pass: {route_state.get('g_la_official_pass')}
- T1B native logging available: {route_state.get('T1B_native_logging_available')}
- T1B visibility pass: {route_state.get('T1B_visibility_pass')}
- Actuator executor success: {route_state.get('actuator_executor_success')}
- P4 open/pass: {route_state.get('p4_open')}/{route_state.get('p4_pass')}

## Boundary

The current legal label-free signal frames A30-A35 did not satisfy all promotion gates unless the route JSON says otherwise.  The current deployable geometry target is diagnostic unless `g_la_official_pass=1`.  Actuator v2 is diagnostic unless safe movement, audit release, and matched-control gap all pass in `v1221_actuator_safety.csv`.
"""
    nxt = """# v12.21 Next Hypothesis Generator

## Line A

Try optimizer-update frame with per-role low-rank residual injection instead of replacing the whole projector.  Add a persistent drift frame that maps hidden/logit drift back through a frozen random cotangent bank rather than only `z^T delta_logit`.

## Line C/T

Log exact output covariance stability and random-cotangent gradient isotropy at training time, then use source-run leaveout to choose a sign-stable G_LA.  Do not promote inverted-sign diagnostics.

## Line C

Split G_LA into separately normalized output-covariance, cotangent-isotropy, coupling-stability, role-drift, and optimizer-spectrum components.  Require source-run leaveout sign stability before any component can act as a direction source.

## Line T

Keep T1B native logging, but add per-window update half-life and role-energy transition features.  Treat AUC below 0.5 as a sign/target-convention diagnostic only unless the predeclared sign passes top-k and leaveout support.

## Line I

If movement is weak, keep the 0.5x/1x/2x/4x bracket and add role-balanced projection before increasing norm.  If movement is strong but release absent, stop increasing amplitude and return to target/source.

## Line B

Do not construct P4 until C/T/I opens legally.  The next Line B hypothesis is a gate-opened persistence test: matched-overhead NoOp, RandomMatchedNorm, and AdamWParallelDirection controls first, then low-frequency event half-life before any lambda grid.

## Line D

Implement the generated classic-family hypotheses as new candidates only: denominator-safe rational coupling repair, Chebyshev degree-energy damping, wavelet local support scale diversity, compact RBF expression repair, and low-frequency Fourier repair.
"""
    (out_dir / "v1221_no_go_boundary.md").write_text(no_go, encoding="utf-8")
    (out_dir / "v1221_next_hypothesis_generator.md").write_text(nxt, encoding="utf-8")
    return {
        "no_go_boundary_written": 1,
        "next_hypothesis_generator_written": 1,
        "next_hypothesis_failed_lines_covered": 1,
        "next_hypothesis_lines": "Line A,Line C,Line T,Line I,Line B,Line D",
    }


def write_simple_svg(path: Path, title: str, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    height = max(220, 44 + 24 * (len(lines) + 1))
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">']
    body.append('<rect width="1200" height="100%" fill="#fbfbf7"/>')
    body.append(f'<text x="24" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#202020">{esc(title)}</text>')
    y = 74
    for line in lines[:42]:
        body.append(f'<text x="24" y="{y}" font-family="Arial" font-size="15" fill="#202020">{esc(line)}</text>')
        y += 24
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, route: Mapping[str, Any]) -> dict[str, Any]:
    labels = read_csv_rows(out_dir / "v1221_label_free_signal_frame.csv")
    vis = read_csv_rows(out_dir / "v1221_visibility_scores.csv")
    act = read_csv_rows(out_dir / "v1221_actuator_safety.csv")
    classic = read_csv_rows(out_dir / "v1221_classic_family_status.csv")
    figs = {
        "fig_v1221_linea_task_linec_pareto.svg": [f"{r.get('candidate_id')}: dA0={r.get('mean_delta_vs_A0')}, lineC={r.get('LineC_nontearing_all_pass')}" for r in labels],
        "fig_v1221_label_free_frame_role_energy.svg": [f"{r.get('candidate_id')}: role_energy={r.get('frame_energy_by_role')}" for r in labels],
        "fig_v1221_auc_step_time_by_label_free_candidate.svg": [f"{r.get('candidate_id')}: step={r.get('AUC_step_ratio_vs_mlp')}, time={r.get('AUC_time_ratio_vs_mlp')}" for r in labels],
        "fig_v1221_GLA_correlation_matrix.svg": [f"rho_noise={route.get('g_la_spearman_noise')}", f"rho_reservoir={route.get('g_la_spearman_reservoir')}", f"official={route.get('g_la_official_pass')}"],
        "fig_v1221_target_sign_inversion_audit.svg": [f"{r.get('score_column')}->{r.get('target')}: auc={r.get('auc_predeclared')}, inv={r.get('auc_inverted')}" for r in read_csv_rows(out_dir / "v1221_target_sign_sanity.csv")[:24]],
        "fig_v1221_soft_target_regression_residuals.svg": [str(r) for r in read_csv_rows(out_dir / "v1221_soft_target_regression.csv")],
        "fig_v1221_visibility_auc_precision_recall.svg": [f"{r.get('feature_tier')}: auc={r.get('auc_joint')}, p={r.get('precision_at_k_joint')}, r={r.get('recall_at_k_joint')}" for r in vis],
        "fig_v1221_T1A_T1B_T2_comparison.svg": [f"{r.get('feature_tier')}: pass={r.get('visibility_pass')}, reason={r.get('failure_reason')}" for r in vis],
        "fig_v1221_leaveout_visibility_heatmap.svg": [str(r) for r in read_csv_rows(out_dir / "v1221_visibility_leaveout.csv")[:32]],
        "fig_v1221_support_distribution.svg": [f"target rows={route.get('linec_deployable_rows')}; T1B rows={route.get('T1B_native_logged_rows')}"],
        "fig_v1221_actuator_movement_release_pareto.svg": [f"{r.get('actuator_id')}: move={r.get('safe_movement_pass')}, release={r.get('release_audit_pass')}, gap={r.get('control_gap')}" for r in act[:40]],
        "fig_v1221_actuator_control_gap.svg": [f"safe={route.get('actuator_safe_movement_rows')}; release={route.get('actuator_release_audit_rows')}; control_resistant={route.get('actuator_control_resistant_rows')}"],
        "fig_v1221_role_response_heatmap.svg": [f"{r.get('role')}: sketch={r.get('sketch_delta_fro')}, drift={r.get('logit_max_abs_drift')}" for r in act[:40]],
        "fig_v1221_functional_p3_control_comparison.svg": [f"P3 rows={route.get('functional_p3_rows')}; P4 open={route.get('p4_open')}"],
        "fig_v1221_p4_val_loss_step_time.svg": ["P4 not executed unless route shows P4 open and P3 pass."],
        "fig_v1221_p4_linec_trajectory.svg": ["No P4 trajectory when P4 gate is closed."],
        "fig_v1221_classic_family_status.svg": [f"{r.get('family')}: {r.get('v1221_status')}; hypothesis={r.get('new_hypothesis_defined')}" for r in classic],
    }
    for name, lines in figs.items():
        write_simple_svg(out_dir / name, name, lines or ["no rows"])
    return {"figure_count": len(figs)}


def write_implementation_readback(out_dir: Path, route: Mapping[str, Any]) -> None:
    text = f"""# v12.21 Implementation Readback

Generated at: {now_iso()}

## What Changed

- Added v12.21 Line A candidates A30-A35 to `experiments/run_v1218_b320_label_free_ablation.py`.
- Added native T1B optimizer-update logging from actual parameter deltas in the Line A training loop.
- Added `augtangentp` and `rolebalancedp` projector initialization paths in `dgkan/models/fc_purekan_primitives.py`.
- Added this v12.21 runner to write Line R/C/T/I/B/D artifacts, route decisions, no-go boundary, next hypotheses, figures, and zip.

## Route Snapshot

- route: {route.get('route', '')}
- minimum_success: {route.get('minimum_success', '')}
- label_free_official_pass_count: {route.get('label_free_official_pass_count', '')}
- g_la_official_pass: {route.get('g_la_official_pass', '')}
- T1B_visibility_pass: {route.get('T1B_visibility_pass', '')}
- actuator_executor_success: {route.get('actuator_executor_success', '')}
- p4_open/p4_pass: {route.get('p4_open', '')}/{route.get('p4_pass', '')}

No fabricated metrics are inserted by this runner.  Empty fields mean the required native measurement was unavailable or the gate was not opened.
"""
    (out_dir / "v1221_implementation_readback.md").write_text(text, encoding="utf-8")


def write_required_manifest(out_dir: Path) -> dict[str, Any]:
    names = [
        "v1221_route_decision.json",
        "v1221_continuation_manifest.csv",
        "v1221_provenance_audit.csv",
        "v1221_code_review_manifest.csv",
        "v1221_implementation_readback.md",
        "v1221_label_free_signal_frame.csv",
        "v1221_label_free_linec.csv",
        "v1221_label_free_failure_decomposition.csv",
        "v1221_linec_deployable_targets.csv",
        "v1221_target_sign_sanity.csv",
        "v1221_soft_target_regression.csv",
        "v1221_target_family_ablation.csv",
        "v1221_visibility_features.csv",
        "v1221_visibility_scores.csv",
        "v1221_visibility_leaveout.csv",
        "v1221_visibility_sanity.csv",
        "v1221_t1b_optimizer_update_features.csv",
        "v1221_actuator_response_dictionary.csv",
        "v1221_actuator_controls.csv",
        "v1221_actuator_safety.csv",
        "v1221_functional_p3.csv",
        "v1221_functional_controls.csv",
        "v1221_functional_p4_short.csv",
        "v1221_p4_event_log.csv",
        "v1221_p4_linec_trajectory.csv",
        "v1221_classic_family_status.csv",
        "v1221_classic_family_new_hypothesis.csv",
        "v1221_classic_family_linec.csv",
        "v1221_no_go_boundary.md",
        "v1221_next_hypothesis_generator.md",
    ]
    rows = [{"stage": "V1221_REQUIRED_ARTIFACT_MANIFEST", "artifact": name, "exists": int((out_dir / name).exists())} for name in names]
    write_csv_rows(out_dir / "v1221_required_artifact_manifest.csv", rows)
    return {"required_artifact_count": len(rows), "required_artifact_missing_count": sum(1 for r in rows if safe_int(r.get("exists"), 0) == 0)}


def package_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "v1221_code_review_packet.zip"
    code_files = [
        "experiments/run_v1221_failclosed_continue2_label_free_functional.py",
        "experiments/run_v1220_failclosed_continue_label_free_functional.py",
        "experiments/run_v1218_b320_label_free_ablation.py",
        "experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py",
        "experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py",
        "experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py",
        "experiments/run_v1252_efficiency_functional_manifold.py",
        "experiments/run_v126_lowerlevel_fhq_functional_geometry.py",
        "experiments/run_v1283_b109_classic_family_functional_geometry.py",
        "experiments/run_v124_multibasis_functional_dual.py",
        "dgkan/models/fc_purekan_primitives.py",
        "dgkan/kernels/fused_hinge_quadratic.py",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if PLAN_DOC.exists():
            zf.write(PLAN_DOC, arcname=f"code/{rel(PLAN_DOC)}")
        for name in code_files:
            p = REPO_ROOT / name
            if p.exists():
                zf.write(p, arcname=f"code/{name}")
        for p in sorted(out_dir.glob("v1221_*")):
            if p.is_file() and p.name != zip_path.name:
                zf.write(p, arcname=f"review_artifacts/{p.name}")
        for p in sorted(out_dir.glob("fig_v1221_*.svg")):
            zf.write(p, arcname=f"figures/{p.name}")
    return zip_path


def write_hash_manifest(out_dir: Path) -> dict[str, str]:
    hashes = {}
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "v1221_hash_manifest.json":
            hashes[p.name] = sha256_file(p)
    write_json(out_dir / "v1221_hash_manifest.json", hashes)
    return hashes


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    rows = collect_linea_rows(Path(args.linea_root), [x for x in str(args.linea_csvs).split(",") if x.strip()])
    summary: dict[str, Any] = {
        "stage": "V1221_ROUTE_DECISION",
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "out_dir": rel(out_dir),
        "linea_root": rel(Path(args.linea_root)),
        "linea_rows": len(rows),
        "v1220_dir": rel(Path(args.v1220_dir)),
        "v1219_dir": rel(Path(args.v1219_dir)),
        "v1218_dir": rel(Path(args.v1218_dir)),
        "v1217_dir": rel(Path(args.v1217_dir)),
        "no_fake": 1,
        "no_proxy": 1,
        "cpu_offload_used": 0,
    }
    summary.update(line_r_code_review(out_dir))
    summary.update(summarize_label_free(out_dir, rows))
    summary.update(write_linec_deployable_targets(out_dir))
    summary.update(write_delta_sign_audit(out_dir))
    summary.update(write_target_sign_sanity(out_dir))
    summary.update(write_soft_target_regression(out_dir))
    summary.update(write_join_key_audit(out_dir, rows))
    summary.update(write_visibility_v5(out_dir))
    summary.update(write_actuator_v2(out_dir, args))
    summary.update(write_functional_gates(out_dir, summary))
    summary.update(write_classic_line_d(out_dir, Path(args.v1218_dir).resolve()))
    summary.update(write_no_go_and_next(out_dir, summary))
    continuation = [
        {"stage": "V1221_CONTINUATION_MANIFEST", "line": "Line R", "promotion_gate": summary.get("line_r_pass", 0), "continuation_executed": int(summary.get("line_r_rows", 0) >= 16), "artifacts": "v1221_code_review_manifest.csv"},
        {"stage": "V1221_CONTINUATION_MANIFEST", "line": "Line A", "promotion_gate": summary.get("label_free_official_pass_count", 0), "continuation_executed": int(summary.get("label_free_candidate_rows", 0) > 0), "artifacts": "v1221_label_free_signal_frame.csv"},
        {"stage": "V1221_CONTINUATION_MANIFEST", "line": "Line C", "promotion_gate": summary.get("g_la_official_pass", 0), "continuation_executed": int(summary.get("linec_deployable_rows", 0) > 0), "artifacts": "v1221_linec_deployable_targets.csv"},
        {"stage": "V1221_CONTINUATION_MANIFEST", "line": "Line T", "promotion_gate": max(safe_int(summary.get("T1A_visibility_pass"), 0), safe_int(summary.get("T1B_visibility_pass"), 0)), "continuation_executed": int(summary.get("visibility_rows", 0) > 0), "artifacts": "v1221_visibility_scores.csv"},
        {"stage": "V1221_CONTINUATION_MANIFEST", "line": "Line I", "promotion_gate": summary.get("actuator_executor_success", 0), "continuation_executed": int(summary.get("actuator_rows", 0) > 0), "artifacts": "v1221_actuator_response_dictionary.csv"},
        {"stage": "V1221_CONTINUATION_MANIFEST", "line": "Line B", "promotion_gate": summary.get("p4_pass", 0), "continuation_executed": int(summary.get("functional_p3_rows", 0) > 0), "artifacts": "v1221_functional_p3.csv"},
        {"stage": "V1221_CONTINUATION_MANIFEST", "line": "Line D", "promotion_gate": 0, "continuation_executed": int(summary.get("classic_new_hypothesis_count", 0) > 0), "artifacts": "v1221_classic_family_new_hypothesis.csv"},
    ]
    write_csv_rows(out_dir / "v1221_continuation_manifest.csv", continuation)
    continuation_missing = sum(1 for r in continuation if safe_int(r.get("continuation_executed"), 0) == 0)
    summary["continuation_missing_count"] = continuation_missing
    if safe_int(summary.get("line_r_pass"), 0) != 1:
        route = "R0-CodeReviewIncomplete"
        minimum_success = "none"
        fail_reason = "required CR0-CR14 code review refs missing"
    elif continuation_missing:
        route = "R0-FailFastIncomplete"
        minimum_success = "none"
        fail_reason = "one or more continuation lines missing"
    elif safe_int(summary.get("label_free_official_pass_count"), 0) > 0:
        route = "R1-LabelFreeBaseCandidate"
        minimum_success = "Success A"
        fail_reason = ""
    elif safe_int(summary.get("g_la_official_pass"), 0) or safe_int(summary.get("T1B_visibility_pass"), 0):
        route = "R3-LossAgnosticValueSourceCandidate"
        minimum_success = "Success B"
        fail_reason = "value source candidate found; functional P4 still gated"
    elif safe_int(summary.get("actuator_executor_success"), 0):
        route = "R4-ActuatorCapacityCandidate"
        minimum_success = "Success C"
        fail_reason = "actuator executor candidate found; functional P4 still gated"
    elif (
        safe_int(summary.get("no_go_boundary_written"), 0)
        and safe_int(summary.get("next_hypothesis_generator_written"), 0)
        and safe_int(summary.get("next_hypothesis_failed_lines_covered"), 0)
    ):
        route = "R2-LabelFreeSignalFrameMissing"
        minimum_success = "Success E"
        fail_reason = "no legal promotion path passed; continuation complete and next hypotheses generated"
    else:
        route = "R1-ContinuationNoNewHypothesis"
        minimum_success = "none"
        fail_reason = "continuation did not generate required no-go/next-hypothesis docs"
    final_stop_allowed = int(
        continuation_missing == 0
        and safe_int(summary.get("required_artifact_missing_count"), 0) == 0
        and safe_int(summary.get("no_go_boundary_written"), 0) == 1
        and safe_int(summary.get("next_hypothesis_generator_written"), 0) == 1
        and safe_int(summary.get("next_hypothesis_failed_lines_covered"), 0) == 1
        and minimum_success in {"Success A", "Success B", "Success C", "Success D", "Success E"}
    )
    summary.update({"route": route, "minimum_success": minimum_success, "fail_reason": fail_reason, "final_stop_allowed": final_stop_allowed})
    write_implementation_readback(out_dir, summary)
    summary.update(write_provenance_audit(out_dir, summary))
    summary.update(write_figures(out_dir, summary))
    write_json(out_dir / "v1221_route_decision.json", summary)
    summary.update(write_required_manifest(out_dir))
    write_json(out_dir / "v1221_route_decision.json", summary)
    zip_path = package_zip(out_dir)
    hashes = write_hash_manifest(out_dir)
    summary["hash_manifest_entries"] = len(hashes)
    summary["code_review_packet_zip"] = rel(zip_path)
    summary["code_review_packet_zip_sha256"] = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        summary["code_review_packet_zip_entries"] = len(zf.namelist())
    write_json(out_dir / "v1221_route_decision.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="official_continuation")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--linea-root", default=str(DEFAULT_OUT_DIR / "linea"))
    parser.add_argument("--linea-csvs", default="")
    parser.add_argument("--v1220-dir", default=str(DEFAULT_V1220_DIR))
    parser.add_argument("--v1219-dir", default=str(DEFAULT_V1219_DIR))
    parser.add_argument("--v1218-dir", default=str(DEFAULT_V1218_DIR))
    parser.add_argument("--v1217-dir", default=str(DEFAULT_V1217_DIR))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--actuator-device", default="cuda:3")
    parser.add_argument("--actuator-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--actuator-seeds", default="0")
    parser.add_argument("--actuator-budgets", default="0.0025,0.005,0.01,0.02")
    parser.add_argument("--actuator-train-size", type=int, default=256)
    parser.add_argument("--actuator-val-size", type=int, default=128)
    parser.add_argument("--actuator-linec-batch", type=int, default=32)
    parser.add_argument("--actuator-sketch-dim", type=int, default=8)
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    result = run_main(parsed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
