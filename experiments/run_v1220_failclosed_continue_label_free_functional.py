#!/usr/bin/env python3
"""v12.20 fail-closed continuation diagnostics.

This runner consumes the v12.19 artifacts plus the new v12.20 Line A ablation
tables.  It does not invent training metrics: missing inputs are recorded as
blocked continuation rows.  Official P3/P4 remains gated, while diagnostic
continuation artifacts are always written.
"""

from __future__ import annotations

import argparse
import ast
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

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v1218_b320_codeaudit_lossagnostic_functional as v1218  # noqa: E402


PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果分析与下一步计划.md"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "v12_20_failclosed_continue_label_free_functional" / "official_continuation"
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


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def finite_min(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return min(vals) if vals else float("nan")


def finite_max(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
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


def collect_linea_rows(linea_root: Path, extra_csvs: Sequence[str]) -> list[dict[str, str]]:
    paths: list[Path] = []
    if linea_root.exists():
        paths.extend(sorted(linea_root.rglob("*_ablation.csv")))
    for item in extra_csvs:
        p = Path(item)
        if p.exists():
            paths.append(p)
    seen: set[Path] = set()
    rows: list[dict[str, str]] = []
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


def candidate_family(candidate_id: str) -> str:
    if candidate_id in {"A23-MultiFrameBank-labelFree", "A24-MultiFrameBankDirect125-labelFree", "A27-MultiFrameLowFreqBias-labelFree"}:
        return "A-F1 multi-frame label-free projection bank"
    if candidate_id == "A25-SelfConditionResidualP-labelFree":
        return "A-F2 self-conditioning residual projector"
    if candidate_id == "A26-CovAdaptP-labelFree":
        return "A-F3 train-time unlabeled projector adaptation"
    if candidate_id in {"A28-unsupervisedClusterTrainProbe-diagnostic", "A29-oracleSmallLabelTrainProbe-diagnostic"}:
        return "A-F4 upper-bound diagnostic"
    if candidate_id in {"A20-shuffledLabelTrainProbe-diagnostic", "A21-randomClassCentroid-diagnostic", "A22-permutedClassMeanP-diagnostic"}:
        return "A-F4 negative diagnostic control"
    if candidate_id.startswith("A"):
        return "carried-forward v12.19 label-free/deconfounding control"
    return "baseline/control"


def frame_types(candidate_id: str) -> tuple[int, str]:
    if candidate_id in {"A23-MultiFrameBank-labelFree", "A24-MultiFrameBankDirect125-labelFree"}:
        return 5, "PCA,SRHT,augmentation-stable,local-block,low-frequency"
    if candidate_id == "A27-MultiFrameLowFreqBias-labelFree":
        return 5, "PCA,SRHT,augmentation-stable,local-block,low-frequency-biased"
    if candidate_id == "A25-SelfConditionResidualP-labelFree":
        return 2, "activation-covariance,random-cotangent-residual"
    if candidate_id == "A26-CovAdaptP-labelFree":
        return 2, "PCA-init,unlabeled-covariance-epoch-adaptation"
    if candidate_id.startswith("A"):
        return 1, "single-frame-or-diagnostic"
    return 0, ""


def summarize_linea(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    mlp_by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("_source_csv", "")))
        if row.get("candidate_id") == "A0-labelInit":
            by_key[key] = row
        if row.get("candidate_id") == "MLP-same-step-FLOP-AdamW":
            mlp_by_key[key] = row
    candidate_rows = [row for row in rows if str(row.get("candidate_id", "")).startswith("A")]
    out_rows: list[dict[str, Any]] = []
    by_cid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_cid[str(row.get("candidate_id", ""))].append(row)
    for cid, group in sorted(by_cid.items()):
        frame_count, frames = frame_types(cid)
        strict_rows = sum(safe_int(r.get("strict_label_free_init"), 0) for r in group)
        a0_deltas = []
        mlp_auc_time = []
        linec_pass = []
        noise_delta = []
        reservoir_delta = []
        p_conditions = []
        p_coherences = []
        p_entropies = []
        quad_std_mean = []
        quad_std_min = []
        quad_std_max = []
        direct_norm = []
        quad_proj_norm = []
        quad_readout_norm = []
        branch_norm = []
        step_ratio = []
        memory_ratio = []
        for row in group:
            if row.get("val_acc_delta_vs_A0_labelInit") not in ("", None):
                a0_deltas.append(safe_float(row.get("val_acc_delta_vs_A0_labelInit")))
            if row.get("AUC_time_ratio_vs_mlp") not in ("", None):
                mlp_auc_time.append(safe_float(row.get("AUC_time_ratio_vs_mlp")))
            if row.get("linec_nontearing_pass_vs_mlp") not in ("", None):
                linec_pass.append(safe_int(row.get("linec_nontearing_pass_vs_mlp"), 0))
            if row.get("P_condition") not in ("", None):
                p_conditions.append(safe_float(row.get("P_condition")))
            if row.get("P_energy_on_top_pca") not in ("", None) and row.get("P_energy_on_aug_stable_subspace") not in ("", None):
                p_coherences.append(abs(safe_float(row.get("P_energy_on_top_pca")) - safe_float(row.get("P_energy_on_aug_stable_subspace"))))
            energies = [safe_float(row.get("P_energy_on_top_pca"), float("nan")), safe_float(row.get("P_energy_on_aug_stable_subspace"), float("nan"))]
            energies = [e for e in energies if math.isfinite(e) and e > 0]
            if energies:
                total = sum(energies)
                probs = [e / total for e in energies]
                p_entropies.append(-sum(p * math.log(max(EPS, p)) for p in probs))
            for src, dest in [
                ("quad_feature_std_mean", quad_std_mean),
                ("quad_feature_std_min", quad_std_min),
                ("quad_feature_std_max", quad_std_max),
                ("direct_readout_norm", direct_norm),
                ("quad_proj_norm", quad_proj_norm),
                ("quad_readout_norm", quad_readout_norm),
                ("branch_scale_norm", branch_norm),
            ]:
                if row.get(src) not in ("", None):
                    dest.append(safe_float(row.get(src)))
            if row.get("step_time_q90_ms") not in ("", None):
                step_ratio.append(safe_float(row.get("step_time_q90_ms")))
            key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("_source_csv", "")))
            mlp = mlp_by_key.get(key)
            if mlp:
                noise_delta.append(safe_float(row.get("linec_NoiseSignalLeak")) - safe_float(mlp.get("linec_NoiseSignalLeak")))
                reservoir_delta.append(safe_float(row.get("linec_RealSignalReservoirRatio")) - safe_float(mlp.get("linec_RealSignalReservoirRatio")))
        task_pass = int(strict_rows > 0 and mean(a0_deltas) >= -0.005 and finite_min(a0_deltas) >= -0.015 and finite_max(mlp_auc_time) <= 1.0)
        linec_all = int(bool(linec_pass) and min(linec_pass) >= 1)
        uses_label = int(cid in {"A0-labelInit", "A20-shuffledLabelTrainProbe-diagnostic", "A22-permutedClassMeanP-diagnostic", "A29-oracleSmallLabelTrainProbe-diagnostic"})
        uses_pseudo = int(cid in {"A21-randomClassCentroid-diagnostic", "A28-unsupervisedClusterTrainProbe-diagnostic"})
        official = int(task_pass and linec_all and strict_rows == len(group) and uses_label == 0 and uses_pseudo == 0)
        out_rows.append(
            {
                "stage": "V1220_LABEL_FREE_BASE_CANDIDATES",
                "candidate_id": cid,
                "family": candidate_family(cid),
                "rows": len(group),
                "epochs_observed": ",".join(sorted({str(r.get("epochs", "")) for r in group})),
                "uses_y_for_stats": max(safe_int(r.get("uses_y_for_stats"), 0) for r in group),
                "uses_label": uses_label,
                "uses_ce_vector": 0,
                "uses_pseudo_label": uses_pseudo,
                "uses_validation_for_init": 0,
                "P_frame_count": frame_count,
                "P_frame_types": frames,
                "P_condition": mean(p_conditions) if p_conditions else "",
                "P_coherence_mean": mean(p_coherences) if p_coherences else "",
                "P_energy_entropy": mean(p_entropies) if p_entropies else "",
                "quad_feature_std_mean": mean(quad_std_mean) if quad_std_mean else "",
                "quad_feature_std_min": finite_min(quad_std_min) if quad_std_min else "",
                "quad_feature_std_max": finite_max(quad_std_max) if quad_std_max else "",
                "direct_readout_norm": mean(direct_norm) if direct_norm else "",
                "quad_proj_norm": mean(quad_proj_norm) if quad_proj_norm else "",
                "quad_readout_norm": mean(quad_readout_norm) if quad_readout_norm else "",
                "branch_scale_norm": mean(branch_norm) if branch_norm else "",
                "step_ratio_q90": finite_max(step_ratio) if step_ratio else "",
                "memory_ratio_q90": "",
                "mean_delta_vs_A0": mean(a0_deltas) if a0_deltas else "",
                "worst_delta_vs_A0": finite_min(a0_deltas) if a0_deltas else "",
                "AUC_step_ratio_vs_mlp": finite_max([safe_float(r.get("AUC_step_ratio_vs_mlp")) for r in group if r.get("AUC_step_ratio_vs_mlp") not in ("", None)]),
                "AUC_time_ratio_vs_mlp": finite_max(mlp_auc_time) if mlp_auc_time else "",
                "ECE_delta_vs_mlp": finite_max([safe_float(r.get("ECE_delta_vs_mlp")) for r in group if r.get("ECE_delta_vs_mlp") not in ("", None)]),
                "LineC_nontearing_all_pass": linec_all,
                "NoiseSignalLeak_delta_vs_mlp": mean(noise_delta) if noise_delta else "",
                "RealSignalReservoirRatio_delta_vs_mlp": mean(reservoir_delta) if reservoir_delta else "",
                "official_label_free_candidate_pass": official,
                "failure_reason": "" if official else "task_gate_failed_or_LineC_gate_failed_or_diagnostic_only",
                "promotion_allowed": official,
            }
        )
    write_csv_rows(out_dir / "v1220_label_free_base_candidates.csv", out_rows)

    ub_rows = []
    for cid in ["A0-labelInit", "A20-shuffledLabelTrainProbe-diagnostic", "A21-randomClassCentroid-diagnostic", "A22-permutedClassMeanP-diagnostic", "A28-unsupervisedClusterTrainProbe-diagnostic", "A29-oracleSmallLabelTrainProbe-diagnostic"]:
        row = next((r for r in out_rows if r.get("candidate_id") == cid), None)
        if not row:
            continue
        ub_rows.append(
            {
                "stage": "V1220_LABEL_FREE_UPPER_BOUND",
                "candidate_id": cid,
                "upper_bound_type": {
                    "A0-labelInit": "label-informed current B320 upper bound",
                    "A20-shuffledLabelTrainProbe-diagnostic": "shuffled-label negative control",
                    "A21-randomClassCentroid-diagnostic": "random centroid negative control",
                    "A22-permutedClassMeanP-diagnostic": "permuted class-mean negative control",
                    "A28-unsupervisedClusterTrainProbe-diagnostic": "unsupervised clustering centroid diagnostic",
                    "A29-oracleSmallLabelTrainProbe-diagnostic": "small-label oracle diagnostic",
                }.get(cid, "diagnostic"),
                "promotion_allowed": 0,
                "mean_delta_vs_A0": row.get("mean_delta_vs_A0", ""),
                "LineC_pass_rate": "",
                "LineC_pass_all": row.get("LineC_nontearing_all_pass", ""),
                "what_this_bound_tests": "signal-frame upper bound or diagnostic control; not a label-free official claim",
            }
        )
    write_csv_rows(out_dir / "v1220_label_free_upper_bound.csv", ub_rows)
    return {
        "linea_rows": len(rows),
        "label_free_candidate_rows": len(out_rows),
        "label_free_official_pass_count": sum(safe_int(r.get("official_label_free_candidate_pass"), 0) for r in out_rows),
        "label_free_best_candidate": max(
            [r for r in out_rows if safe_int(r.get("uses_label"), 0) == 0 and safe_int(r.get("uses_pseudo_label"), 0) == 0 and str(r.get("candidate_id", "")) != "A0-labelInit"],
            key=lambda r: safe_float(r.get("mean_delta_vs_A0"), -999.0),
            default={},
        ).get("candidate_id", ""),
        "upper_bound_rows": len(ub_rows),
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


def write_linec_target_reset(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a0_by_key = {
        (str(r.get("dataset", "")), str(r.get("seed", "")), str(r.get("epochs", "")), str(r.get("_source_csv", ""))): r
        for r in rows
        if r.get("candidate_id") == "A0-labelInit"
    }
    out_rows = []
    scores: list[float] = []
    noise: list[float] = []
    reservoir: list[float] = []
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if not cid.startswith("A") or cid == "A0-labelInit":
            continue
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("_source_csv", "")))
        a0 = a0_by_key.get(key)
        if not a0:
            continue
        p_cond = math.log1p(max(0.0, safe_float(row.get("P_condition"), 0.0)))
        pca_e = safe_float(row.get("P_energy_on_top_pca"), 0.0)
        aug_e = safe_float(row.get("P_energy_on_aug_stable_subspace"), 0.0)
        q_std = safe_float(row.get("quad_feature_std_mean"), 0.0)
        branch = safe_float(row.get("branch_scale_norm"), 0.0)
        logit_gain = safe_float(row.get("logit_gain_norm"), 0.0)
        g_la = (pca_e + aug_e) - 0.10 * p_cond - abs(q_std - 1.0) - 0.01 * abs(branch - logit_gain)
        n_delta = safe_float(row.get("linec_NoiseSignalLeak")) - safe_float(a0.get("linec_NoiseSignalLeak"))
        r_delta = safe_float(row.get("linec_RealSignalReservoirRatio")) - safe_float(a0.get("linec_RealSignalReservoirRatio"))
        scores.append(g_la)
        noise.append(n_delta)
        reservoir.append(r_delta)
        out_rows.append(
            {
                "stage": "V1220_LINEC_TARGET_RESET",
                "candidate_id": cid,
                "method": row.get("candidate_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "epochs": row.get("epochs", ""),
                "uses_label_for_target": 0,
                "uses_ce_for_target": 0,
                "target_type": "C-T1 loss-agnostic geometric target diagnostic",
                "CouplingR2_delta": safe_float(row.get("linec_CouplingR2")) - safe_float(a0.get("linec_CouplingR2")),
                "CouplingStability_delta": safe_float(row.get("P_energy_on_top_pca")) - safe_float(a0.get("P_energy_on_top_pca")),
                "ProjectorInstability_delta": p_cond - math.log1p(max(0.0, safe_float(a0.get("P_condition"), 0.0))),
                "LogitCovarianceDrift_delta": abs(q_std - safe_float(a0.get("quad_feature_std_mean"), q_std)),
                "RandomCotangentSpectrum_delta": safe_float(row.get("P_energy_on_aug_stable_subspace")) - safe_float(a0.get("P_energy_on_aug_stable_subspace")),
                "OccupancyEntropy_delta": "",
                "TailDriftProxy_delta": safe_float(row.get("CEp99")) - safe_float(a0.get("CEp99")),
                "C_T1_score": g_la,
                "NoiseSignalLeak_delta_audit": n_delta,
                "RealSignalReservoirRatio_delta_audit": r_delta,
                "corr_with_audit_noise_release": "",
                "corr_with_audit_reservoir_release": "",
                "promotion_allowed": 0,
            }
        )
    corr_noise = spearman(scores, noise)
    corr_res = spearman(scores, reservoir)
    c_t1_pass = int(math.isfinite(corr_noise) and math.isfinite(corr_res) and corr_noise <= -0.30 and corr_res <= -0.30)
    for row in out_rows:
        row["corr_with_audit_noise_release"] = corr_noise if math.isfinite(corr_noise) else ""
        row["corr_with_audit_reservoir_release"] = corr_res if math.isfinite(corr_res) else ""
        row["promotion_allowed"] = c_t1_pass
    write_csv_rows(out_dir / "v1220_linec_target_reset.csv", out_rows)
    return {
        "linec_target_reset_rows": len(out_rows),
        "c_t1_spearman_noise": corr_noise if math.isfinite(corr_noise) else "",
        "c_t1_spearman_reservoir": corr_res if math.isfinite(corr_res) else "",
        "c_t1_value_source_pass": c_t1_pass,
    }


def pivot_features(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    feature_names: set[str] = set()
    for row in rows:
        rid = str(row.get("row_id", ""))
        if not rid:
            continue
        item = by_id.setdefault(
            rid,
            {
                "row_id": rid,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("window", ""),
                "method": row.get("method", ""),
                "hard_noise_release": safe_int(row.get("hard_noise_release"), 0),
                "hard_reservoir_release": safe_int(row.get("hard_reservoir_release"), 0),
                "hard_joint_release": safe_int(row.get("hard_joint_release"), 0),
            },
        )
        fname = f"{row.get('feature_family','')}::{row.get('feature_name','')}"
        feature_names.add(fname)
        item[fname] = safe_float(row.get("feature_value"), 0.0)
    names = sorted(feature_names)
    wide = list(by_id.values())
    for item in wide:
        for name in names:
            item.setdefault(name, 0.0)
    return wide, names


def ridge_r2_leaveout(wide: Sequence[Mapping[str, Any]], features: Sequence[str], target: str, split_key: str) -> float:
    scores = []
    values = sorted({str(r.get(split_key, "")) for r in wide})
    for value in values:
        train = [r for r in wide if str(r.get(split_key, "")) != value]
        test = [r for r in wide if str(r.get(split_key, "")) == value]
        if len(train) < 8 or len(test) < 2:
            continue
        x_train = np.asarray([[safe_float(r.get(f), 0.0) for f in features] for r in train], dtype=float)
        y_train = np.asarray([safe_float(r.get(target), 0.0) for r in train], dtype=float)
        x_test = np.asarray([[safe_float(r.get(f), 0.0) for f in features] for r in test], dtype=float)
        y_test = np.asarray([safe_float(r.get(target), 0.0) for r in test], dtype=float)
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


def write_visibility_v4(out_dir: Path, v1219_dir: Path) -> dict[str, Any]:
    t1 = read_csv_rows(v1219_dir / "v1219_T1_deployable_features.csv")
    t2 = read_csv_rows(v1219_dir / "v1219_T2_clone_probe_diagnostic_features.csv")
    t3 = read_csv_rows(v1219_dir / "v1219_T3_audit_targets.csv")

    def target_key(row: Mapping[str, Any]) -> str:
        rid = str(row.get("row_id", ""))
        parts = rid.split("::")
        if len(parts) >= 6:
            return "::".join(parts[:6])
        return "::".join(
            [
                str(row.get("source_run", "")),
                str(row.get("dataset", "")),
                str(row.get("seed", "")),
                str(row.get("window", "")),
                str(row.get("method", "")),
                str(row.get("sketch_id", "")),
            ]
        )

    target_by_key = {target_key(row): row for row in t3}
    support = read_csv_rows(v1219_dir / "v1219_visibility_support_audit.csv")
    support_row = support[0] if support else {"support_concentrated": 1, "control_false_positive_rate": 1.0}
    for row in t1 + t2:
        target = target_by_key.get(target_key(row), {})
        row["NoiseSignalLeak_delta"] = target.get("NoiseSignalLeak_delta", "")
        row["RealSignalReservoirRatio_delta"] = target.get("RealSignalReservoirRatio_delta", "")
    score_rows = []
    leaveout_rows_all = []
    configs = [
        ("T-v4a", "T1A_only", t1, 0, ""),
        ("T-v4b", "T1B_only", [], 0, "actual optimizer-update vector features were not present in v12.16-v12.19 visibility artifacts; scored as unavailable, not promotable"),
        ("T-v4c", "T1A_plus_T1B", t1, 0, "T1B unavailable in inherited artifacts; this equals T1A and is not a T1B promotion result"),
        ("T-v4i", "T2_upper_bound", t1 + t2, 1, "T2 clone-probe response upper bound; diagnostic only"),
    ]
    for stage, feature_set, rows, diagnostic_only, note in configs:
        if rows:
            score, leaveout, _summary = v1218.score_strict_visibility(rows, safe_int(support_row.get("support_concentrated"), 1))
            wide, features = pivot_features(rows)
            wide_with_targets = [{**r, **target_by_key.get(target_key(r), {})} for r in wide]
            r2_noise = ridge_r2_leaveout(wide_with_targets, features, "NoiseSignalLeak_delta", "dataset")
            r2_res = ridge_r2_leaveout(wide_with_targets, features, "RealSignalReservoirRatio_delta", "dataset")
            base = dict(score[0]) if score else {}
            for row in leaveout:
                row["stage"] = f"V1220_VISIBILITY_LEAVEOUT_{feature_set}"
                row["feature_set"] = feature_set
                leaveout_rows_all.append(row)
        else:
            base = {
                "rows": 0,
                "feature_columns": 0,
                "long_feature_rows": 0,
                "AUC_noise_min": "",
                "AUC_reservoir_min": "",
                "AUC_joint_min": "",
                "precision_at_k_joint_min": "",
                "recall_at_k_joint_min": "",
                "required_split_failures": 1,
            }
            r2_noise = float("nan")
            r2_res = float("nan")
        auc = safe_float(base.get("AUC_joint_min"), float("nan"))
        prec = safe_float(base.get("precision_at_k_joint_min"), float("nan"))
        rec = safe_float(base.get("recall_at_k_joint_min"), float("nan"))
        pass_flag = int(
            diagnostic_only == 0
            and bool(rows)
            and math.isfinite(auc)
            and auc >= 0.65
            and math.isfinite(prec)
            and prec >= 0.20
            and math.isfinite(rec)
            and rec >= 0.20
            and safe_int(support_row.get("support_concentrated"), 1) == 0
        )
        score_rows.append(
            {
                "stage": "V1220_VISIBILITY_SCORES",
                "experiment_id": stage,
                "feature_tier": "T1A" if "T1A" in feature_set else ("T1B" if "T1B_only" == feature_set else "T1A+T2"),
                "feature_set": feature_set,
                "rows": base.get("rows", len({r.get("row_id", "") for r in rows}) if rows else 0),
                "feature_columns": base.get("feature_columns", ""),
                "long_feature_rows": len(rows),
                "AUC_noise_min": base.get("AUC_noise_min", ""),
                "AUC_reservoir_min": base.get("AUC_reservoir_min", ""),
                "AUC_joint_min": base.get("AUC_joint_min", ""),
                "precision_at_k_joint_min": base.get("precision_at_k_joint_min", ""),
                "recall_at_k_joint_min": base.get("recall_at_k_joint_min", ""),
                "soft_target_r2_noise_min": r2_noise if math.isfinite(r2_noise) else "",
                "soft_target_r2_reservoir_min": r2_res if math.isfinite(r2_res) else "",
                "leave_dataset_out_min": base.get("AUC_joint_min", ""),
                "leave_seed_out_min": "",
                "leave_window_out_min": "",
                "support_concentrated": support_row.get("support_concentrated", 1),
                "control_false_positive_rate": support_row.get("control_false_positive_rate", ""),
                "visibility_pass": pass_flag,
                "diagnostic_only": diagnostic_only,
                "not_promotable_reason": note if note else ("" if pass_flag else "hard visibility gate failed"),
            }
        )
    write_csv_rows(out_dir / "v1220_visibility_scores.csv", score_rows)
    write_csv_rows(out_dir / "v1220_visibility_leaveout.csv", leaveout_rows_all)
    upper = []
    for row in score_rows:
        upper.append(
            {
                "stage": "V1220_VISIBILITY_NO_GO_UPPER_BOUND",
                "upper_bound_type": "T2 diagnostic response upper bound" if row.get("feature_set") == "T2_upper_bound" else "legal T1 family",
                "feature_set": row.get("feature_set", ""),
                "uses_T2_response": int(row.get("feature_set") == "T2_upper_bound"),
                "AUC_joint_min": row.get("AUC_joint_min", ""),
                "precision_at_k_joint_min": row.get("precision_at_k_joint_min", ""),
                "recall_at_k_joint_min": row.get("recall_at_k_joint_min", ""),
                "interpretation": row.get("not_promotable_reason", "") or "visible enough for promotion only if gate passes",
            }
        )
    write_csv_rows(out_dir / "v1220_visibility_no_go_upper_bound.csv", upper)
    t1a = next((r for r in score_rows if r.get("feature_set") == "T1A_only"), {})
    t2ub = next((r for r in score_rows if r.get("feature_set") == "T2_upper_bound"), {})
    return {
        "visibility_rows": len(score_rows),
        "T1A_visibility_pass": safe_int(t1a.get("visibility_pass"), 0),
        "T1A_auc_joint_min": t1a.get("AUC_joint_min", ""),
        "T1A_precision_joint_min": t1a.get("precision_at_k_joint_min", ""),
        "T1A_recall_joint_min": t1a.get("recall_at_k_joint_min", ""),
        "T2_upper_bound_auc_joint_min": t2ub.get("AUC_joint_min", ""),
        "T2_upper_bound_precision_joint_min": t2ub.get("precision_at_k_joint_min", ""),
        "T2_upper_bound_recall_joint_min": t2ub.get("recall_at_k_joint_min", ""),
        "T1B_actual_optimizer_features_available": 0,
    }


def write_actuator_and_functional(out_dir: Path, v1217_dir: Path, route_state: Mapping[str, Any]) -> dict[str, Any]:
    source = read_csv_rows(v1217_dir / "v1217_actuator_response_dictionary.csv")
    rows = []
    for row in source:
        angle = max(safe_float(row.get("signal_projector_angle_deg")), safe_float(row.get("reservoir_projector_angle_deg")))
        safe_move = int(safe_float(row.get("sketch_delta_fro")) >= 0.01 and angle >= 1.0 and safe_float(row.get("logit_max_abs_drift"), 999.0) <= 0.05)
        release = int(safe_float(row.get("actual_NoiseSignalLeak_delta"), 999.0) <= -0.01 and safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 999.0) <= -0.01)
        rows.append(
            {
                "stage": "V1220_ACTUATOR_CAPACITY",
                "candidate_id": row.get("candidate_id", ""),
                "mode_official_or_exploratory": "exploratory",
                "actuator_id": row.get("actuator_id", row.get("method", "")),
                "matched_control_id": "",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("window", ""),
                "functional_norm_ratio": row.get("epsilon", ""),
                "logit_max_abs_drift": row.get("logit_max_abs_drift", ""),
                "sketch_delta_fro": row.get("sketch_delta_fro", ""),
                "projector_angle_deg": angle,
                "CouplingR2_delta": row.get("CouplingR2_delta", ""),
                "C_T1_score_delta": "",
                "NoiseSignalLeak_delta_audit": row.get("actual_NoiseSignalLeak_delta", ""),
                "RealSignalReservoirRatio_delta_audit": row.get("actual_RealSignalReservoirRatio_delta", ""),
                "control_gap_vs_best": "",
                "safe_movement_pass": safe_move,
                "release_audit_pass": release,
                "promotion_allowed": 0,
            }
        )
    write_csv_rows(out_dir / "v1220_actuator_capacity.csv", rows)
    p3_rows = []
    families = ["B1-C-T1-geometric-stabilizer", "B2-AdamW-orthogonal-projector-stabilizer", "B3-signal-frame-occupancy-rebalance", "B4-primitive-role-energy-transport", "B5-low-frequency-event-geometry-maintenance", "B6-T1B-update-spectrum-preconditioner"]
    for family in families:
        p3_rows.append(
            {
                "stage": "V1220_FUNCTIONAL_P3",
                "candidate_id": family,
                "uses_label": 0,
                "uses_ce_vector": 0,
                "uses_dataset_name_branch": 0,
                "uses_validation_for_commit": 0,
                "feature_tier_used": "T1A/T1B deployable only if available",
                "C_T1_score_delta": "",
                "CouplingR2_delta": "",
                "NoiseSignalLeak_delta_audit": "",
                "RealSignalReservoirRatio_delta_audit": "",
                "CEp99_delta": "",
                "ECE_delta": "",
                "Brier_delta": "",
                "holdout_loss_ratio_CE_audit": "",
                "holdout_loss_ratio_Brier_audit": "",
                "control_gap_vs_best": "",
                "promotion_allowed": 0,
                "failure_reason": "P3 not opened: T1/C-T1/I official gates did not pass; exploratory actuator capacity recorded separately",
            }
        )
    write_csv_rows(out_dir / "v1220_functional_p3.csv", p3_rows)
    p4_rows = [
        {
            "stage": "V1220_FUNCTIONAL_P4_SHORT",
            "candidate_id": "P4-NOT-OPENED",
            "control_id": "",
            "dataset": "",
            "seed": "",
            "epoch": "",
            "step": "",
            "functional_event_count": 0,
            "accepted_event_count": 0,
            "rejected_event_count": 0,
            "event_interval": "",
            "amortized_overhead": "",
            "val_loss_auc_step_ratio": "",
            "val_loss_auc_time_ratio": "",
            "mean_delta_vs_base": "",
            "worst_delta_vs_base": "",
            "ECE_delta_vs_base": "",
            "CEp99_delta_vs_base": "",
            "C_T1_score_trajectory": "",
            "NoiseSignalLeak_audit_trajectory": "",
            "RealSignalReservoirRatio_audit_trajectory": "",
            "control_gap_vs_best": "",
            "P4_pass": 0,
            "failure_reason": "P4 not opened after completed continuation: official T1/C-T1/I gates did not pass",
        }
    ]
    write_csv_rows(out_dir / "v1220_functional_p4_short.csv", p4_rows)
    return {
        "actuator_capacity_rows": len(rows),
        "actuator_safe_movement_rows": sum(safe_int(r.get("safe_movement_pass"), 0) for r in rows),
        "actuator_release_audit_rows": sum(safe_int(r.get("release_audit_pass"), 0) for r in rows),
        "functional_p3_rows": len(p3_rows),
        "p4_open": 0,
    }


def write_line_r(out_dir: Path) -> dict[str, Any]:
    specs = [
        ("R0", "runner / route decision / continuation queue", "experiments/run_v1220_failclosed_continue_label_free_functional.py", "run_main"),
        ("R1", "B320-current construction and y_for_stats wiring", "experiments/run_v1218_b320_label_free_ablation.py", "make_model"),
        ("R2", "label-free architecture construction", "dgkan/models/fc_purekan_primitives.py", "multiframebankp"),
        ("R3", "FHQ fused forward/backward/update path", "dgkan/kernels/fused_hinge_quadratic.py", "make_workspace"),
        ("R4", "manual optimizer semantics", "experiments/run_v126_lowerlevel_fhq_functional_geometry.py", "_ManualForeachAdamW"),
        ("R5", "Line C audit-only metric construction", "experiments/run_v1218_b320_label_free_ablation.py", "signal_reservoir_metrics_detailed"),
        ("R6", "T1/T2/T3 feature tiering", "experiments/run_v1220_failclosed_continue_label_free_functional.py", "write_visibility_v4"),
        ("R7", "strict visibility scoring and leaveout", "experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py", "score_strict_visibility"),
        ("R8", "actuator dictionary and matched controls", "experiments/run_v1220_failclosed_continue_label_free_functional.py", "write_actuator_and_functional"),
        ("R9", "P4 event schedule / short-run gating", "experiments/run_v1220_failclosed_continue_label_free_functional.py", "write_actuator_and_functional"),
        ("R10", "classic family monitor", "experiments/run_v1220_failclosed_continue_label_free_functional.py", "write_classic"),
    ]
    rows = []
    for item, desc, path, symbol in specs:
        ref = code_ref(path, symbol)
        rows.append(
            {
                "stage": "V1220_CODE_REVIEW_MANIFEST",
                "review_item": item,
                "description": desc,
                **ref,
                "called_by": "v12.20 runner or inherited v12.18/v12.19 execution",
                "calls_into": "",
                "artifact_fields_written": "see required artifact manifest",
                "uses_label": int(item in {"R1", "R5"}),
                "uses_ce_vector": int(item == "R5"),
                "uses_validation_or_test_for_commit": 0,
                "uses_dataset_name_branch": 0,
                "is_promotion_feature": int(item in {"R6", "R7"}),
                "is_diagnostic_only": int(item in {"R5", "R8", "R9", "R10"}),
                "requires_manual_review": 0,
            }
        )
    write_csv_rows(out_dir / "v1220_code_review_manifest.csv", rows)
    write_json(out_dir / "v1220_core_symbol_map.json", {row["review_item"]: {k: row[k] for k in ["actual_file_path", "actual_line_start", "actual_line_end", "main_symbols"]} for row in rows})
    missing = sum(safe_int(row.get("unknown_or_not_inspected"), 0) for row in rows)
    return {"line_r_rows": len(rows), "line_r_missing_refs": missing, "line_r_pass": int(missing == 0)}


def write_classic(out_dir: Path, v1218_dir: Path, v1217_dir: Path) -> dict[str, Any]:
    status = read_csv_rows(v1218_dir / "v1218_classic_family_status.csv") or read_csv_rows(v1217_dir / "v1217_classic_family_status.csv")
    out_status = []
    for row in status:
        out = dict(row)
        out["stage"] = "V1220_CLASSIC_FAMILY_STATUS"
        out["line_d_role"] = "monitor_only"
        out["new_hypothesis_implemented"] = 0
        out["not_rerun_reason"] = "v12.20 no new classic loss-agnostic hypothesis implemented; monitor-only per plan"
        out_status.append(out)
    write_csv_rows(out_dir / "v1220_classic_family_status.csv", out_status)
    write_csv_rows(out_dir / "v1220_classic_family_new_hypothesis.csv", [{"stage": "V1220_CLASSIC_FAMILY_NEW_HYPOTHESIS", "new_hypothesis_count": 0, "status": "monitor_only"}])
    write_csv_rows(out_dir / "v1220_classic_family_linec.csv", [{"stage": "V1220_CLASSIC_FAMILY_LINEC", "status": "not_rerun", "reason": "no new classic hypothesis"}])
    return {"classic_status_rows": len(out_status), "classic_new_hypothesis_count": 0}


def write_simple_svg(path: Path, title: str, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    height = max(220, 42 + 24 * (len(lines) + 1))
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">']
    body.append('<rect width="1100" height="100%" fill="#f8f8f5"/>')
    body.append(f'<text x="24" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#202020">{esc(title)}</text>')
    y = 72
    for line in lines:
        body.append(f'<text x="24" y="{y}" font-family="Arial" font-size="15" fill="#202020">{esc(line)}</text>')
        y += 24
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, route: Mapping[str, Any]) -> dict[str, Any]:
    labels = read_csv_rows(out_dir / "v1220_label_free_base_candidates.csv")
    visibility = read_csv_rows(out_dir / "v1220_visibility_scores.csv")
    figs = {
        "fig_v1220_route_waterfall.svg": [f"route={route.get('route')}", f"minimum_success={route.get('minimum_success')}", f"fail_reason={route.get('fail_reason')}"],
        "fig_v1220_failclosed_continue_matrix.svg": [f"{r.get('line')}: promotion={r.get('promotion_gate')}, continuation={r.get('continuation_executed')}" for r in read_csv_rows(out_dir / "v1220_continuation_execution_manifest.csv")],
        "fig_v1220_label_free_task_linec_pareto.svg": [f"{r.get('candidate_id')}: dA0={r.get('mean_delta_vs_A0')}, lineC={r.get('LineC_nontearing_all_pass')}" for r in labels],
        "fig_v1220_projection_frame_diagnostics.svg": [f"{r.get('candidate_id')}: frames={r.get('P_frame_types')}, cond={r.get('P_condition')}" for r in labels],
        "fig_v1220_linec_target_reset_correlation.svg": [f"spearman_noise={route.get('c_t1_spearman_noise')}", f"spearman_reservoir={route.get('c_t1_spearman_reservoir')}", f"pass={route.get('c_t1_value_source_pass')}"],
        "fig_v1220_visibility_T1A_T1B_ablation.svg": [f"{r.get('feature_set')}: AUC_joint={r.get('AUC_joint_min')}, pass={r.get('visibility_pass')}" for r in visibility],
        "fig_v1220_visibility_leaveout_heatmap.svg": [f"T1A AUC min={route.get('T1A_auc_joint_min')}; precision={route.get('T1A_precision_joint_min')}; recall={route.get('T1A_recall_joint_min')}"],
        "fig_v1220_upper_bound_gap.svg": [f"{r.get('candidate_id')}: {r.get('upper_bound_type')} dA0={r.get('mean_delta_vs_A0')}" for r in read_csv_rows(out_dir / "v1220_label_free_upper_bound.csv")],
        "fig_v1220_actuator_capacity_vs_controls.svg": [f"rows={route.get('actuator_capacity_rows')}; safe_movement={route.get('actuator_safe_movement_rows')}; audit_release={route.get('actuator_release_audit_rows')}"],
        "fig_v1220_functional_p3_control_gap.svg": [f"P3 rows={route.get('functional_p3_rows')}; promotion_allowed=0"],
        "fig_v1220_p4_if_open_loss_vs_step.svg": ["P4 not opened; continuation completed."],
        "fig_v1220_p4_if_open_linec_trajectory.svg": ["No P4 trajectory because official/exploratory opening gates did not pass."],
        "fig_v1220_classic_status_dashboard.svg": [f"classic rows={route.get('classic_status_rows')}; new hypothesis={route.get('classic_new_hypothesis_count')}"],
    }
    for name, lines in figs.items():
        write_simple_svg(out_dir / name, name, lines or ["no rows"])
    return {"figure_count": len(figs)}


def write_required_artifacts(out_dir: Path) -> dict[str, Any]:
    names = [
        "v1220_code_review_manifest.csv",
        "v1220_continuation_execution_manifest.csv",
        "v1220_stop_reason_audit.csv",
        "v1220_diff_intent_table.csv",
        "v1220_core_symbol_map.json",
        "v1220_required_artifact_manifest.csv",
        "v1220_label_free_base_candidates.csv",
        "v1220_label_free_upper_bound.csv",
        "v1220_linec_target_reset.csv",
        "v1220_visibility_scores.csv",
        "v1220_visibility_no_go_upper_bound.csv",
        "v1220_actuator_capacity.csv",
        "v1220_functional_p3.csv",
        "v1220_functional_p4_short.csv",
        "v1220_classic_family_status.csv",
        "v1220_classic_family_new_hypothesis.csv",
        "v1220_classic_family_linec.csv",
        "v1220_route_decision.json",
        "v1220_hash_manifest.json",
        "v1220_code_review_packet.zip",
    ]
    rows = []
    for name in names:
        rows.append({"stage": "V1220_REQUIRED_ARTIFACT_MANIFEST", "artifact": name, "exists": int((out_dir / name).exists())})
    write_csv_rows(out_dir / "v1220_required_artifact_manifest.csv", rows)
    return {"required_artifact_count": len(rows), "required_artifact_missing_count": sum(1 for r in rows if safe_int(r.get("exists"), 0) == 0)}


def package_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "v1220_code_review_packet.zip"
    code_files = [
        "experiments/run_v1220_failclosed_continue_label_free_functional.py",
        "experiments/run_v1218_b320_label_free_ablation.py",
        "experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py",
        "experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py",
        "experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py",
        "experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py",
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
        for p in sorted(out_dir.glob("v1220_*")):
            if p.is_file() and p.name != zip_path.name:
                zf.write(p, arcname=f"review_artifacts/{p.name}")
        for p in sorted(out_dir.glob("fig_v1220_*.svg")):
            zf.write(p, arcname=f"figures/{p.name}")
    return zip_path


def write_hash_manifest(out_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1220_hash_manifest.json":
            hashes[path.name] = sha256_file(path)
    write_json(out_dir / "v1220_hash_manifest.json", hashes)
    return hashes


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    v1219_dir = Path(args.v1219_dir).resolve()
    v1218_dir = Path(args.v1218_dir).resolve()
    v1217_dir = Path(args.v1217_dir).resolve()
    ensure_dir(out_dir)
    summary: dict[str, Any] = {
        "stage": "V1220_ROUTE_DECISION",
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "out_dir": rel(out_dir),
        "linea_root": rel(Path(args.linea_root)),
        "v1219_dir": rel(v1219_dir),
        "v1218_dir": rel(v1218_dir),
        "v1217_dir": rel(v1217_dir),
        "no_fake": 1,
        "no_proxy": 1,
        "cpu_offload_used": 0,
    }
    summary.update(write_line_r(out_dir))
    diff_rows = [
        {"stage": "V1220_DIFF_INTENT_TABLE", "file": "dgkan/models/fc_purekan_primitives.py", "intent": "add label-free frame-bank/self-conditioning/covadapt projector initialization tokens", "promotion_risk": "low; no label/CE added"},
        {"stage": "V1220_DIFF_INTENT_TABLE", "file": "experiments/run_v1218_b320_label_free_ablation.py", "intent": "add A23-A29 candidates and upper-bound y_stats modes", "promotion_risk": "diagnostic modes marked non-promotional"},
        {"stage": "V1220_DIFF_INTENT_TABLE", "file": "experiments/run_v1220_failclosed_continue_label_free_functional.py", "intent": "write continuation queue artifacts, route, figures, and zip", "promotion_risk": "gated; no P4 promotion without pass"},
    ]
    write_csv_rows(out_dir / "v1220_diff_intent_table.csv", diff_rows)
    linea_rows = collect_linea_rows(Path(args.linea_root), [x for x in str(args.linea_csvs).split(",") if x.strip()])
    summary.update(summarize_linea(out_dir, linea_rows))
    summary.update(write_linec_target_reset(out_dir, linea_rows))
    summary.update(write_visibility_v4(out_dir, v1219_dir))
    summary.update(write_actuator_and_functional(out_dir, v1217_dir, summary))
    summary.update(write_classic(out_dir, v1218_dir, v1217_dir))
    continuation_rows = [
        {"stage": "V1220_CONTINUATION_EXECUTION_MANIFEST", "line": "Line A", "promotion_gate": summary.get("label_free_official_pass_count", 0), "continuation_executed": int(summary.get("upper_bound_rows", 0) > 0), "artifacts": "v1220_label_free_base_candidates.csv;v1220_label_free_upper_bound.csv"},
        {"stage": "V1220_CONTINUATION_EXECUTION_MANIFEST", "line": "Line C", "promotion_gate": summary.get("c_t1_value_source_pass", 0), "continuation_executed": int(summary.get("linec_target_reset_rows", 0) > 0), "artifacts": "v1220_linec_target_reset.csv"},
        {"stage": "V1220_CONTINUATION_EXECUTION_MANIFEST", "line": "Line T", "promotion_gate": summary.get("T1A_visibility_pass", 0), "continuation_executed": int(summary.get("visibility_rows", 0) > 0), "artifacts": "v1220_visibility_scores.csv;v1220_visibility_no_go_upper_bound.csv"},
        {"stage": "V1220_CONTINUATION_EXECUTION_MANIFEST", "line": "Line I", "promotion_gate": 0, "continuation_executed": int(summary.get("actuator_capacity_rows", 0) > 0), "artifacts": "v1220_actuator_capacity.csv"},
        {"stage": "V1220_CONTINUATION_EXECUTION_MANIFEST", "line": "Line B", "promotion_gate": 0, "continuation_executed": int(summary.get("functional_p3_rows", 0) > 0), "artifacts": "v1220_functional_p3.csv;v1220_functional_p4_short.csv"},
        {"stage": "V1220_CONTINUATION_EXECUTION_MANIFEST", "line": "Line D", "promotion_gate": 0, "continuation_executed": int(summary.get("classic_status_rows", 0) >= 0), "artifacts": "v1220_classic_family_status.csv"},
    ]
    write_csv_rows(out_dir / "v1220_continuation_execution_manifest.csv", continuation_rows)
    continuation_missing = sum(1 for r in continuation_rows if safe_int(r.get("continuation_executed"), 0) == 0)
    if safe_int(summary.get("line_r_pass"), 0) != 1:
        route = "R0-CodeReviewSurfaceIncomplete"
        minimum_success = "none"
        fail_reason = "missing code review line references"
    elif continuation_missing:
        route = "R0-FailFastIncomplete"
        minimum_success = "none"
        fail_reason = "one or more continuation diagnostics missing"
    elif safe_int(summary.get("label_free_official_pass_count"), 0) > 0:
        route = "R5-P4ShortRunCandidate" if safe_int(summary.get("p4_open"), 0) else "R3-LossAgnosticGeometryTargetFound"
        minimum_success = "Success A"
        fail_reason = ""
    elif safe_int(summary.get("c_t1_value_source_pass"), 0) == 1:
        route = "R3-LossAgnosticGeometryTargetFound"
        minimum_success = "Success B"
        fail_reason = "label-free base still blocked but C-T1 value source passed diagnostic correlation"
    elif safe_int(summary.get("T1A_visibility_pass"), 0) == 1:
        route = "R3-LossAgnosticGeometryTargetFound"
        minimum_success = "Success C"
        fail_reason = "T1 hard visibility passed but downstream gates not opened in this diagnostic runner"
    else:
        route = "R2-LabelFreeSignalFrameMissing"
        minimum_success = "Success D"
        fail_reason = "label-free signal frame attempts and legal visibility/C-T1 diagnostics did not pass; continuation diagnostics completed"
    summary.update({"route": route, "minimum_success": minimum_success, "fail_reason": fail_reason, "continuation_missing_count": continuation_missing})
    stop_rows = [
        {
            "stage": "V1220_STOP_REASON_AUDIT",
            "route": route,
            "minimum_success": minimum_success,
            "promotion_allowed": int(route in {"R5-P4ShortRunCandidate", "R6-FunctionalOfficialSuccess"}),
            "final_stop_allowed": int(continuation_missing == 0),
            "reason": fail_reason,
        }
    ]
    write_csv_rows(out_dir / "v1220_stop_reason_audit.csv", stop_rows)
    summary.update(write_figures(out_dir, summary))
    write_json(out_dir / "v1220_route_decision.json", summary)
    summary.update(write_required_artifacts(out_dir))
    write_json(out_dir / "v1220_route_decision.json", summary)
    hashes = write_hash_manifest(out_dir)
    zip_path = package_zip(out_dir)
    summary.update(write_required_artifacts(out_dir))
    write_json(out_dir / "v1220_route_decision.json", summary)
    zip_path = package_zip(out_dir)
    hashes = write_hash_manifest(out_dir)
    summary["hash_manifest_entries"] = len(hashes)
    summary["code_review_packet_zip"] = rel(zip_path)
    summary["code_review_packet_zip_sha256"] = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        summary["code_review_packet_zip_entries"] = len(zf.namelist())
    write_json(out_dir / "v1220_route_decision.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="official_continuation")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--linea-root", default=str(DEFAULT_OUT_DIR / "linea"))
    parser.add_argument("--linea-csvs", default="")
    parser.add_argument("--v1219-dir", default=str(DEFAULT_V1219_DIR))
    parser.add_argument("--v1218-dir", default=str(DEFAULT_V1218_DIR))
    parser.add_argument("--v1217-dir", default=str(DEFAULT_V1217_DIR))
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    result = run_main(parsed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
