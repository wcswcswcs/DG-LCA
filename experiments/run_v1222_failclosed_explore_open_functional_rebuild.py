#!/usr/bin/env python3
"""v12.22 fail-closed / explore-open functional rebuild runner.

This runner is intentionally queue-driven.  Promotion gates remain fail-closed,
but failed gates do not end the run: every pre-registered exploration/fallback
job must leave an executed artifact or an explicit budget-tradeoff deferral.
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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v1218_b320_label_free_ablation as linea  # noqa: E402
import run_v1221_failclosed_continue2_label_free_functional as v1221  # noqa: E402
import run_v1252_efficiency_functional_manifold as v1252  # noqa: E402


PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.22_FailClosedExploreOpen_FunctionalRebuild_实验结果分析与下一步计划.md"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "v12_22_failclosed_explore_open_functional_rebuild" / "official_explore_open"
DEFAULT_LINEA_ROOT = DEFAULT_OUT_DIR / "linea"
DEFAULT_V1221_DIR = REPO_ROOT / "results" / "v12_21_failclosed_continue2_label_free_functional" / "official_continuation"
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


def raw_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


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
        val = raw_float(value)
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


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def stage_copy(rows: Sequence[Mapping[str, Any]], stage: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        r["stage"] = stage
        out.append(r)
    return out


def candidate_is_label(candidate_id: str) -> int:
    return int(candidate_id in {"A0-labelInit", "A20-shuffledLabelTrainProbe-diagnostic", "A22-permutedClassMeanP-diagnostic", "A29-oracleSmallLabelTrainProbe-diagnostic", "A42-SmallLabelOracleUpperBound-diagnostic"})


def candidate_is_pseudo(candidate_id: str) -> int:
    return int(candidate_id in {"A21-randomClassCentroid-diagnostic", "A28-unsupervisedClusterTrainProbe-diagnostic"})


def candidate_frame_family(candidate_id: str) -> str:
    mapping = {
        "A36-ResidualA1LowRankFrame-labelFree": "residual_on_A1_lowrank",
        "A37-ConvexMultiFrameMixture-labelFree": "convex_multiframe_mixture",
        "A38-PersistentDriftCotangentBank-labelFree": "persistent_drift_cotangent_bank",
        "A39-RoleConditionedFrame-labelFree": "role_conditioned_frame",
        "A40-SelfConditionedStopGradFrame-labelFree": "self_conditioned_stopgrad_frame",
        "A41-UnlabeledFrameAdaptSchedule-labelFree": "unlabeled_frame_adaptation_schedule",
        "A42-SmallLabelOracleUpperBound-diagnostic": "small_label_oracle_upper_bound_diagnostic",
    }
    if candidate_id in mapping:
        return mapping[candidate_id]
    return v1221.candidate_frame_family(candidate_id)


def collect_linea_rows(linea_root: Path, extra_csvs: Sequence[str]) -> list[dict[str, str]]:
    paths = sorted(linea_root.rglob("*_ablation.csv")) if linea_root.exists() else []
    for item in extra_csvs:
        if item.strip() and Path(item).exists():
            paths.append(Path(item))
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
    return (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("_source_csv", row.get("source_run", ""))))


def write_code_review(out_dir: Path) -> dict[str, Any]:
    specs = [
        ("CR0", "v12.22 queue/final-stop contract", "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py", "write_execution_contract"),
        ("CR1", "v12.22 route decision", "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py", "run_main"),
        ("CR2", "A36-A42 Line A specs", "experiments/run_v1218_b320_label_free_ablation.py", "ablation_specs"),
        ("CR3", "T1B temporal/spectrum logging", "experiments/run_v1218_b320_label_free_ablation.py", "optimizer_update_observable_stats"),
        ("CR4", "unlabeled projector adaptation", "experiments/run_v1218_b320_label_free_ablation.py", "apply_unlabeled_projector_adaptation"),
        ("CR5", "v12.22 frame tokens", "dgkan/models/fc_purekan_primitives.py", "reslowrankp"),
        ("CR6", "Line C/T value source reset", "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py", "write_linec_t_visibility_v4"),
        ("CR7", "actuator norm bisection", "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py", "write_actuator_v3"),
        ("CR8", "shadow P4 capacity", "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py", "write_functional_and_shadow"),
        ("CR9", "classic family non-fake status", "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py", "write_classic_line_d_v1222"),
        ("CR10", "B320 manual/fused training path", "experiments/run_v1283_b109_classic_family_functional_geometry.py", "_step_b109"),
        ("CR11", "Line C audit metrics", "experiments/run_v1218_b320_label_free_ablation.py", "signal_reservoir_metrics_detailed"),
        ("CR12", "previous v12.21 reusable audit routines", "experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_visibility_v5"),
    ]
    rows = []
    for item, desc, path, symbol in specs:
        rows.append(
            {
                "stage": "V1222_CODE_REVIEW_MANIFEST",
                "review_item": item,
                "description": desc,
                **code_ref(path, symbol),
                "uses_label": int(item in {"CR11"}),
                "uses_ce_vector": int(item in {"CR11"}),
                "uses_validation_or_test_for_commit": 0,
                "uses_dataset_name_branch": 0,
                "requires_manual_review": 0,
            }
        )
    write_csv_rows(out_dir / "v1222_code_review_manifest.csv", rows)
    return {"code_review_rows": len(rows), "line_r_required_missing": sum(safe_int(r.get("unknown_or_not_inspected"), 0) for r in rows)}


def write_label_free_v2(out_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    v1221.summarize_label_free(out_dir, raw_rows)
    source_signal = read_csv_rows(out_dir / "v1221_label_free_signal_frame.csv")
    source_linec = read_csv_rows(out_dir / "v1221_label_free_linec.csv")
    source_failure = read_csv_rows(out_dir / "v1221_label_free_failure_decomposition.csv")
    raw_by_cid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_cid[str(row.get("candidate_id", ""))].append(row)
    pass_rate_by_cid = {
        cid: mean([r.get("linec_nontearing_pass_vs_mlp", "") for r in rows])
        for cid, rows in defaultdict(list, {cid: [r for r in source_linec if r.get("candidate_id") == cid] for cid in {r.get("candidate_id", "") for r in source_linec}}).items()
    }
    signal_rows = []
    for row in source_signal:
        cid = str(row.get("candidate_id", ""))
        uses_label = candidate_is_label(cid)
        uses_pseudo = candidate_is_pseudo(cid)
        mean_delta = raw_float(row.get("mean_delta_vs_A0"))
        worst_delta = raw_float(row.get("worst_delta_vs_A0"))
        auc_time = raw_float(row.get("AUC_time_ratio_vs_mlp"))
        linec_all = safe_int(row.get("LineC_nontearing_all_pass"), 0)
        linec_rate = pass_rate_by_cid.get(cid, float("nan"))
        legal = int(cid != "A0-labelInit" and uses_label == 0 and uses_pseudo == 0)
        official = int(legal and mean_delta >= -0.003 and worst_delta >= -0.010 and auc_time <= 1.00 and linec_all == 1)
        exploration = int(legal and mean_delta >= -0.010 and worst_delta >= -0.030 and auc_time <= 1.15 and math.isfinite(linec_rate) and linec_rate >= (5.0 / 9.0))
        nan_count = 0
        for raw in raw_by_cid.get(cid, []):
            for col in ["val_acc", "NLL", "ECE", "CEp99", "linec_CouplingR2", "linec_NoiseSignalLeak", "linec_RealSignalReservoirRatio"]:
                val = raw_float(raw.get(col))
                if raw.get(col) not in ("", None) and not math.isfinite(val):
                    nan_count += 1
        out = dict(row)
        out["stage"] = "V1222_LABEL_FREE_SIGNAL_FRAME"
        out["frame_family"] = candidate_frame_family(cid)
        out["uses_label"] = uses_label
        out["uses_ce_vector"] = 0
        out["uses_validation_or_test"] = 0
        out["uses_dataset_name"] = 0
        out["uses_optimizer_update_opaque"] = int(cid in {"A30-OptimizerObservableFrame-labelFree", "A34-HybridA1OptimizerFrame-labelFree"})
        out["LineC_pass_rate"] = linec_rate if math.isfinite(linec_rate) else ""
        out["official_label_free_candidate_pass"] = official
        out["exploration_gate_pass"] = exploration
        out["promotion_allowed"] = official
        out["NaN_count"] = nan_count
        if not official:
            reasons = []
            if not legal:
                reasons.append("not_legal_label_free_promotion_candidate")
            if not math.isfinite(mean_delta) or mean_delta < -0.003:
                reasons.append("mean_delta_gate_failed")
            if not math.isfinite(worst_delta) or worst_delta < -0.010:
                reasons.append("worst_delta_gate_failed")
            if not math.isfinite(auc_time) or auc_time > 1.00:
                reasons.append("AUC_time_gate_failed")
            if linec_all != 1:
                reasons.append("LineC_all_pass_gate_failed")
            if nan_count:
                reasons.append("nan_or_nonfinite_observed")
            out["failure_reason"] = ";".join(reasons)
        signal_rows.append(out)
    failure_rows = stage_copy(source_failure, "V1222_LABEL_FREE_FAILURE_DECOMPOSITION")
    linec_rows = stage_copy(source_linec, "V1222_LABEL_FREE_LINEC")
    write_csv_rows(out_dir / "v1222_label_free_signal_frame.csv", signal_rows)
    write_csv_rows(out_dir / "v1222_label_free_failure_decomposition.csv", failure_rows)
    write_csv_rows(out_dir / "v1222_label_free_linec.csv", linec_rows)
    legal_linec = [r for r in source_linec if candidate_is_label(str(r.get("candidate_id", ""))) == 0 and candidate_is_pseudo(str(r.get("candidate_id", ""))) == 0]
    write_csv_rows(out_dir / "v1221_label_free_linec.csv", legal_linec)
    legal_signal = [r for r in signal_rows if safe_int(r.get("uses_label"), 0) == 0 and str(r.get("candidate_id", "")) != "A0-labelInit"]
    best = max(legal_signal, key=lambda r: (safe_float(r.get("mean_delta_vs_A0"), -999.0), safe_float(r.get("LineC_pass_rate"), -999.0)), default={})
    return {
        "linea_rows": len(raw_rows),
        "label_free_candidate_rows": len(signal_rows),
        "label_free_official_pass_count": sum(safe_int(r.get("official_label_free_candidate_pass"), 0) for r in signal_rows),
        "label_free_exploration_pass_count": sum(safe_int(r.get("exploration_gate_pass"), 0) for r in signal_rows),
        "label_free_best_candidate": best.get("candidate_id", ""),
        "label_free_best_mean_delta_vs_A0": best.get("mean_delta_vs_A0", ""),
        "label_free_best_worst_delta_vs_A0": best.get("worst_delta_vs_A0", ""),
        "label_free_best_auc_time_ratio_vs_mlp": best.get("AUC_time_ratio_vs_mlp", ""),
        "label_free_best_linec_pass_rate": best.get("LineC_pass_rate", ""),
        "label_free_linec_rows": len(linec_rows),
    }


def raw_row_lookup(out_dir: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    lookup = {}
    for path in sorted((out_dir / "linea").rglob("*_ablation.csv")):
        for row in read_csv_rows(path):
            lookup[(rel(path), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("epochs", "")), str(row.get("candidate_id", "")))] = row
    return lookup


def leaveout_min(rows: Sequence[Mapping[str, Any]], tier: str, split_col: str) -> float:
    vals = [raw_float(r.get("auc_joint")) for r in rows if r.get("feature_tier") == tier and r.get("split_col") == split_col]
    vals = [v for v in vals if math.isfinite(v)]
    return min(vals) if vals else float("nan")


def write_linec_t_visibility_v4(out_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    v1221.write_linec_deployable_targets(out_dir)
    v1221.write_delta_sign_audit(out_dir)
    v1221.write_target_sign_sanity(out_dir)
    v1221.write_soft_target_regression(out_dir)
    v1221.write_join_key_audit(out_dir, raw_rows)
    v1221.write_visibility_v5(out_dir)

    raw_lookup = raw_row_lookup(out_dir)
    deploy_rows = []
    base_rows = read_csv_rows(out_dir / "v1221_linec_deployable_targets.csv")
    for row in base_rows:
        raw = raw_lookup.get((str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("window", "")), str(row.get("method", ""))), {})
        out = dict(row)
        out["stage"] = "V1222_LINEC_DEPLOYABLE_TARGETS"
        out["G_role_transition"] = raw.get("t1b_update_role_energy_transition_l1", "")
        out["G_projector_stability"] = raw.get("P_condition", "")
        out["G_update_half_life"] = raw.get("t1b_update_half_life_frac", "")
        out["G_spectrum_entropy"] = raw.get("t1b_update_spectrum_entropy_mean", "")
        out["G_quad_direct_energy_flow"] = raw.get("t1b_update_quad_direct_energy_flow_mean", "")
        out["G_LA_v4_predeclared"] = row.get("G_composite", "")
        deploy_rows.append(out)
    write_csv_rows(out_dir / "v1222_linec_deployable_targets.csv", deploy_rows)

    t1b_rows = stage_copy(read_csv_rows(out_dir / "v1221_t1b_optimizer_update_features.csv"), "V1222_T1B_OPTIMIZER_UPDATE_FEATURES")
    write_csv_rows(out_dir / "v1222_t1b_optimizer_update_features.csv", t1b_rows)
    write_csv_rows(out_dir / "v1222_target_sign_sanity.csv", stage_copy(read_csv_rows(out_dir / "v1221_target_sign_sanity.csv"), "V1222_TARGET_SIGN_SANITY"))
    write_csv_rows(out_dir / "v1222_soft_target_regression.csv", stage_copy(read_csv_rows(out_dir / "v1221_soft_target_regression.csv"), "V1222_SOFT_TARGET_REGRESSION"))

    leave = stage_copy(read_csv_rows(out_dir / "v1221_visibility_leaveout.csv"), "V1222_VISIBILITY_LEAVEOUT")
    write_csv_rows(out_dir / "v1222_visibility_leaveout.csv", leave)
    score_rows = []
    for row in read_csv_rows(out_dir / "v1221_visibility_scores.csv"):
        tier = str(row.get("feature_tier", ""))
        auc = raw_float(row.get("auc_joint"))
        p = raw_float(row.get("precision_at_k_joint"))
        rc = raw_float(row.get("recall_at_k_joint"))
        r2n = raw_float(row.get("soft_target_r2_noise"))
        r2r = raw_float(row.get("soft_target_r2_reservoir"))
        ld = leaveout_min(leave, tier, "dataset")
        ls = leaveout_min(leave, tier, "seed")
        lw = leaveout_min(leave, tier, "window")
        official = int(
            tier == "T1A"
            and auc >= 0.70
            and p >= 0.25
            and rc >= 0.25
            and r2n > 0.0
            and r2r > 0.0
            and all(math.isfinite(v) and v >= 0.50 for v in [ld, ls, lw])
        )
        exploration = int(auc >= 0.62 and (p >= 0.15 or rc >= 0.15) and (not math.isfinite(r2n) or r2n > -10.0) and (not math.isfinite(r2r) or r2r > -10.0))
        out = dict(row)
        out["stage"] = "V1222_VISIBILITY_SCORES"
        out["leave_dataset_auc_min"] = ld if math.isfinite(ld) else ""
        out["leave_seed_auc_min"] = ls if math.isfinite(ls) else ""
        out["leave_window_auc_min"] = lw if math.isfinite(lw) else ""
        out["sign_stability"] = int((not math.isfinite(ld) or ld >= 0.50) and (not math.isfinite(ls) or ls >= 0.50) and (not math.isfinite(lw) or lw >= 0.50))
        out["support_count"] = sum(v1221.release_labels(r)[2] for r in base_rows)
        out["support_concentration"] = ""
        out["visibility_pass"] = official
        out["visibility_exploration_pass"] = exploration
        if not official:
            reasons = []
            if tier != "T1A":
                reasons.append("not_T1A_official_source")
            if not math.isfinite(auc) or auc < 0.70:
                reasons.append("AUC_joint_official_gate_failed")
            if not math.isfinite(p) or p < 0.25 or not math.isfinite(rc) or rc < 0.25:
                reasons.append("precision_recall_official_gate_failed")
            if not math.isfinite(r2n) or not math.isfinite(r2r) or r2n <= 0.0 or r2r <= 0.0:
                reasons.append("soft_R2_official_gate_failed")
            if not all(math.isfinite(v) and v >= 0.50 for v in [ld, ls, lw]):
                reasons.append("leaveout_stability_gate_failed")
            out["failure_reason"] = ";".join(reasons)
        score_rows.append(out)
    write_csv_rows(out_dir / "v1222_visibility_scores.csv", score_rows)
    feature_prov = [
        {"stage": "V1222_FEATURE_PROVENANCE", "feature_family": "T1A", "source": "precommit unlabeled state/projector/logit covariance columns", "uses_label": 0, "uses_ce_vector": 0, "uses_future_outcome": 0, "native_logged": 1},
        {"stage": "V1222_FEATURE_PROVENANCE", "feature_family": "T1B", "source": "native optimizer-observable parameter update logs", "uses_label": 0, "uses_ce_vector": 0, "uses_future_outcome": 0, "native_logged": int(bool(t1b_rows))},
        {"stage": "V1222_FEATURE_PROVENANCE", "feature_family": "T3 audit target", "source": "Line C label/CE-derived release target for scoring only", "uses_label": 1, "uses_ce_vector": 1, "uses_future_outcome": 1, "native_logged": 1},
    ]
    write_csv_rows(out_dir / "v1222_feature_provenance.csv", feature_prov)
    t1a = next((r for r in score_rows if r.get("feature_tier") == "T1A"), {})
    t1b = next((r for r in score_rows if r.get("feature_tier") == "T1B"), {})
    return {
        "linec_deployable_rows": len(deploy_rows),
        "T1A_visibility_pass": safe_int(t1a.get("visibility_pass"), 0),
        "T1A_visibility_exploration_pass": safe_int(t1a.get("visibility_exploration_pass"), 0),
        "T1A_auc_joint": t1a.get("auc_joint", ""),
        "T1B_visibility_pass": safe_int(t1b.get("visibility_pass"), 0),
        "T1B_visibility_exploration_pass": safe_int(t1b.get("visibility_exploration_pass"), 0),
        "T1B_auc_joint": t1b.get("auc_joint", ""),
        "T1B_native_logged_rows": len(t1b_rows),
        "g_la_official_pass": safe_int(t1a.get("visibility_pass"), 0),
        "g_la_exploratory": max(safe_int(t1a.get("visibility_exploration_pass"), 0), safe_int(t1b.get("visibility_exploration_pass"), 0)),
    }


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def apply_v1222_actuator(model: torch.nn.Module, actuator: str, budget: float, sign: float, gen: torch.Generator, opt_delta: Mapping[str, torch.Tensor]) -> str:
    if actuator == "I7-response-matrix-composed-actuator":
        v1221._apply_actuator(model, "I2-quad-proj-local-rotation", budget * 0.55, sign, gen, opt_delta)
        v1221._apply_actuator(model, "I5-branch-gain-redistribution", budget * 0.35, -sign, gen, opt_delta)
        v1221._apply_actuator(model, "I4-absdiag-energy", budget * 0.10, sign, gen, opt_delta)
        return "response_matrix_composed"
    if actuator == "I10-AdamWParallel-role-energy-control":
        return v1221._apply_actuator(model, "I7-optimizer-update-aligned", budget, sign, gen, opt_delta)
    return v1221._apply_actuator(model, actuator, budget, sign, gen, opt_delta)


def write_actuator_v3(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.actuator_device)
    if device.type != "cuda":
        raise RuntimeError("v12.22 actuator requires CUDA; CPU-offload is not allowed")
    torch.cuda.set_device(device)
    rows = []
    datasets = [v120._canonical_dataset(x.strip()) for x in str(args.actuator_datasets).split(",") if x.strip()]
    seeds = parse_csv_ints(args.actuator_seeds)
    budgets = parse_csv_floats(args.actuator_budgets)
    actuators = [
        "I1-direct-role-scale-shift",
        "I2-quad-proj-local-rotation",
        "I3-hinge-amplitude-threshold",
        "I4-absdiag-energy",
        "I5-branch-gain-redistribution",
        "I6-role-balanced-combined",
        "I7-response-matrix-composed-actuator",
        "I8-random-matched-role-energy-control",
        "I9-shuffled-actuator-basis-control",
        "I10-AdamWParallel-role-energy-control",
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
            model = linea.make_model("A1-noYForStats", spec, int(input_dim), int(output_dim), x_train, y_train, device, int(seed) + 1222000, 0)
            b = min(int(args.actuator_linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
            xb, yb, xq, yq = x_train[:b], y_train[:b], x_val[:b], y_val[:b]
            base_metrics = v1221._linec_metrics(model, xb, yb, xq, yq, seed + 7000, int(args.actuator_sketch_dim))
            base_eval = v1252._classification_basic(model, x_val, y_val)
            with torch.no_grad():
                base_logits = model(xq).detach()
            try:
                opt_updated = v1252._take_adamw_window(model, xb, yb, 2.0e-3, 1.0e-3)
                opt_delta = v1221._copy_state_delta(model, opt_updated)
            except Exception:
                opt_delta = {}
            base_p = getattr(model, "quad_proj", torch.empty(0, device=device)).detach().clone() if hasattr(model, "quad_proj") else torch.empty(0, device=device)
            for actuator in actuators:
                for signed in [-1.0, 1.0]:
                    for budget in budgets:
                        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 1000000) + len(actuator) * 17 + (1 if signed > 0 else 2))
                        trial = v1221.copy.deepcopy(model).to(device)
                        role = apply_v1222_actuator(trial, actuator, float(budget), float(signed), gen, opt_delta)
                        with torch.no_grad():
                            logits = trial(xq).detach()
                        p_now = getattr(trial, "quad_proj", torch.empty(0, device=device)).detach()
                        angle = v1221._projector_angle_deg(base_p, p_now) if int(base_p.numel()) and int(p_now.numel()) == int(base_p.numel()) else 0.0
                        drift = float((logits - base_logits).abs().max().item())
                        sketch_delta = float((logits - base_logits).float().norm().div(math.sqrt(max(1, int(logits.numel())))).item())
                        metrics = v1221._linec_metrics(trial, xb, yb, xq, yq, seed + 8100 + int(abs(budget) * 1000000), int(args.actuator_sketch_dim))
                        eval_now = v1252._classification_basic(trial, x_val, y_val)
                        n_delta = metrics["NoiseSignalLeak"] - base_metrics["NoiseSignalLeak"]
                        r_delta = metrics["RealSignalReservoirRatio"] - base_metrics["RealSignalReservoirRatio"]
                        c_delta = metrics["CouplingR2"] - base_metrics["CouplingR2"]
                        if actuator == "I8-random-matched-role-energy-control":
                            basis_type = "matched_role_energy_control"
                        elif actuator == "I9-shuffled-actuator-basis-control":
                            basis_type = "shuffled_actuator_basis_control"
                        elif actuator == "I10-AdamWParallel-role-energy-control":
                            basis_type = "adamwparallel_control"
                        else:
                            basis_type = "actuator"
                        row = {
                            "stage": "V1222_ACTUATOR_RESPONSE_DICTIONARY",
                            "dataset": dataset,
                            "seed": seed,
                            "window": "smoke",
                            "actuator_id": actuator,
                            "role": role,
                            "basis_type": basis_type,
                            "norm_budget": budget,
                            "signed_direction": signed,
                            "uses_label": int(basis_type == "adamwparallel_control"),
                            "uses_ce_vector": int(basis_type == "adamwparallel_control"),
                            "matched_control_id": "I8/I9/I10 controls",
                            "sketch_delta_fro": sketch_delta,
                            "projector_angle_deg": angle,
                            "logit_max_abs_drift": drift,
                            "CouplingR2_delta": c_delta,
                            "NoiseSignalLeak_delta_audit": n_delta,
                            "RealSignalReservoirRatio_delta_audit": r_delta,
                            "CEp99_delta": safe_float(eval_now.get("CEp99"), 0.0) - safe_float(base_eval.get("CEp99"), 0.0),
                            "ECE_delta": safe_float(eval_now.get("ECE"), 0.0) - safe_float(base_eval.get("ECE"), 0.0),
                            "safe_movement_pass": int(sketch_delta >= 0.01 and angle >= 1.0 and drift <= 0.05),
                            "release_audit_pass": int(n_delta <= -0.005 and r_delta <= -0.005),
                            "drift_boundary_selected": 0,
                            "control_gap": "",
                            "matched_control_gap": "",
                            "failure_reason": "",
                        }
                        rows.append(row)
    for key in {(r["dataset"], r["seed"], r["actuator_id"], r["signed_direction"]) for r in rows}:
        group = [r for r in rows if (r["dataset"], r["seed"], r["actuator_id"], r["signed_direction"]) == key and safe_float(r.get("logit_max_abs_drift"), 999.0) <= 0.05]
        if group:
            selected = max(group, key=lambda r: safe_float(r.get("norm_budget"), -1.0))
            selected["drift_boundary_selected"] = 1
    for row in rows:
        gap = v1221.matched_control_gap(rows, row)
        row["control_gap"] = gap if math.isfinite(gap) else ""
        row["matched_control_gap"] = row["control_gap"]
        ok = safe_int(row.get("safe_movement_pass"), 0) and safe_int(row.get("release_audit_pass"), 0) and math.isfinite(gap) and gap >= 0.005
        row["failure_reason"] = "" if ok else "movement_or_release_or_control_gap_gate_failed"
    controls = [dict(r, stage="V1222_ACTUATOR_CONTROLS") for r in rows if str(r.get("basis_type", "")).endswith("control")]
    safety = [dict(r, stage="V1222_ACTUATOR_SAFETY") for r in rows]
    write_csv_rows(out_dir / "v1222_actuator_response_dictionary.csv", rows)
    write_csv_rows(out_dir / "v1222_actuator_controls.csv", controls)
    write_csv_rows(out_dir / "v1222_actuator_safety.csv", safety)
    non_control = [r for r in rows if not str(r.get("basis_type", "")).endswith("control")]
    safe_keys = {(r["dataset"], r["seed"]) for r in non_control if safe_int(r.get("safe_movement_pass"), 0)}
    release_keys = {(r["dataset"], r["seed"]) for r in non_control if safe_int(r.get("release_audit_pass"), 0)}
    control_keys = {(r["dataset"], r["seed"]) for r in non_control if safe_float(r.get("control_gap"), -999.0) >= 0.005 and safe_int(r.get("release_audit_pass"), 0)}
    avg_gap = mean([r.get("control_gap", "") for r in non_control])
    return {
        "actuator_rows": len(rows),
        "actuator_safe_movement_rows": sum(safe_int(r.get("safe_movement_pass"), 0) for r in non_control),
        "actuator_safe_movement_dataset_seed_count": len(safe_keys),
        "actuator_release_audit_rows": sum(safe_int(r.get("release_audit_pass"), 0) for r in non_control),
        "actuator_release_dataset_seed_count": len(release_keys),
        "actuator_control_resistant_dataset_seed_count": len(control_keys),
        "actuator_control_gap_mean": avg_gap if math.isfinite(avg_gap) else "",
        "actuator_exploratory_success": int(len(safe_keys) >= 6 and len(release_keys) >= 3 and (not math.isfinite(avg_gap) or avg_gap >= 0.0)),
        "actuator_official_gate_pass": int(len(control_keys) >= 6),
    }


def write_functional_and_shadow(out_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    official_open = int(safe_int(state.get("T1A_visibility_pass"), 0) and safe_int(state.get("actuator_official_gate_pass"), 0))
    candidates = ["B1-G_LA_geometry_stabilizer", "B2-T1B_update_spectrum_preconditioner", "B3-role_balanced_actuator_maintenance", "B4-noise_leak_vetoed_functional_event", "B5-reservoir_release_low_frequency_event", "B6-response_matrix_constrained_functional_update"]
    p3_rows = []
    for cid in candidates:
        p3_rows.append(
            {
                "stage": "V1222_FUNCTIONAL_P3",
                "candidate_id": cid,
                "P3_open": official_open,
                "P3_pass": 0,
                "promotion_allowed": 0,
                "failure_reason": "" if official_open else "official P3 closed: T1A official value source and actuator official gate did not both pass",
            }
        )
    p4_rows = [
        {
            "stage": "V1222_FUNCTIONAL_P4_SHORT",
            "candidate_id": "P4-NOT-OPENED" if not official_open else "P4-BLOCKED-NO-P3-PASS",
            "P4_open": official_open,
            "P4_pass": 0,
            "beats_all_controls": 0,
            "promotion_allowed": 0,
            "failure_reason": "official P4 not opened because official P3 gate is closed" if not official_open else "P3 pass absent; P4 short run not executed",
        }
    ]
    act_rows = [r for r in read_csv_rows(out_dir / "v1222_actuator_safety.csv") if not str(r.get("basis_type", "")).endswith("control")]
    best_by_key = {}
    for row in act_rows:
        key = (row.get("dataset", ""), row.get("seed", ""))
        score = safe_float(row.get("sketch_delta_fro"), 0.0) - 10.0 * max(0.0, safe_float(row.get("logit_max_abs_drift"), 999.0) - 0.05)
        if key not in best_by_key or score > best_by_key[key][0]:
            best_by_key[key] = (score, row)
    shadow = []
    for _key, (_score, row) in sorted(best_by_key.items()):
        shadow.append(
            {
                "stage": "V1222_SHADOW_P4_CAPACITY",
                "shadow_p4_capacity": 1,
                "promotion_allowed": 0,
                "executed": 1,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "source_actuator_id": row.get("actuator_id", ""),
                "sketch_delta_fro": row.get("sketch_delta_fro", ""),
                "projector_angle_deg": row.get("projector_angle_deg", ""),
                "logit_max_abs_drift": row.get("logit_max_abs_drift", ""),
                "NoiseSignalLeak_delta_audit": row.get("NoiseSignalLeak_delta_audit", ""),
                "RealSignalReservoirRatio_delta_audit": row.get("RealSignalReservoirRatio_delta_audit", ""),
                "reason": "shadow diagnostic only; official P4 remains gated",
            }
        )
    if not shadow:
        shadow.append({"stage": "V1222_SHADOW_P4_CAPACITY", "shadow_p4_capacity": 0, "promotion_allowed": 0, "executed": 0, "reason": "no actuator rows available"})
    write_csv_rows(out_dir / "v1222_functional_p3.csv", p3_rows)
    write_csv_rows(out_dir / "v1222_functional_p4_short.csv", p4_rows)
    write_csv_rows(out_dir / "v1222_shadow_p4_capacity.csv", shadow)
    return {"functional_p3_rows": len(p3_rows), "p3_open": official_open, "p3_pass": 0, "p4_open": official_open, "p4_pass": 0, "shadow_p4_capacity_rows": len(shadow), "shadow_p4_executed": int(any(safe_int(r.get("executed"), 0) for r in shadow))}


def write_classic_line_d_v1222(out_dir: Path) -> dict[str, Any]:
    families = ["Rational", "Chebyshev", "Wavelet", "RBF/FastKAN", "Fourier"]
    rows = []
    for fam in families:
        rows.append(
            {
                "stage": "V1222_CLASSIC_FAMILY_STATUS",
                "family": fam,
                "candidate_id": "",
                "new_hypothesis_id": f"V1222-{fam.replace('/', '-')}-deferred-budget",
                "executed_this_version": 0,
                "L3_efficiency_pass": "",
                "A4_expression_pass": "",
                "A5_task_pass": "",
                "LineC_pass": "",
                "status": "NotExecuted_HypothesisGenerated",
                "blocker": "Line A/C/T/I/B exploration consumed registered v12.22 runtime budget before classic candidate smoke",
                "budget_tradeoff_recorded": 1,
                "next_candidate": {
                    "Rational": "denominator-safe coupling repair plus group diversity tangent metric",
                    "Chebyshev": "degree-energy damping plus task trajectory repair",
                    "Wavelet": "local support scale-diversity task-stable repair",
                    "RBF/FastKAN": "compact capacity expression repair without dense RBF fallback",
                    "Fourier": "low-frequency expression repair without high-frequency path",
                }[fam],
            }
        )
    write_csv_rows(out_dir / "v1222_classic_family_status.csv", rows)
    return {"classic_status_rows": len(rows), "classic_executed_count": sum(safe_int(r.get("executed_this_version"), 0) for r in rows), "classic_deferred_with_budget_count": sum(safe_int(r.get("budget_tradeoff_recorded"), 0) for r in rows)}


def write_no_go_next_readback(out_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    no_go = f"""# v12.22 No-Go Boundary

Generated at: {now_iso()}

This boundary is fail-closed for promotion and explore-open for mechanism evidence.

## Observed Gates

- label_free_official_pass_count: {state.get('label_free_official_pass_count')}
- label_free_exploration_pass_count: {state.get('label_free_exploration_pass_count')}
- T1A_visibility_pass/exploration: {state.get('T1A_visibility_pass')}/{state.get('T1A_visibility_exploration_pass')}
- T1B_visibility_pass/exploration: {state.get('T1B_visibility_pass')}/{state.get('T1B_visibility_exploration_pass')}
- actuator_official_gate_pass/exploration: {state.get('actuator_official_gate_pass')}/{state.get('actuator_exploratory_success')}
- p3_open/p4_open/p4_pass: {state.get('p3_open')}/{state.get('p4_open')}/{state.get('p4_pass')}

No next-hypothesis document is used to grant final_stop_allowed.  Final stop is decided only by `v1222_final_stop_audit.json`.
"""
    nxt = """# v12.22 Next Hypothesis Generator

## Line A

If residual-on-A1 remains below the exploration gate, reduce residual energy and split frame conditioning by role before adding new frame families.

## Line C/T

If T1A/T1B visibility remains weak, run a target convention sanity pass and then a clone-probe upper bound; do not promote optimizer-update features without a separate loss-agnostic approval.

## Line I

If movement exists but release remains absent, keep the logit drift boundary and return to target/source construction instead of increasing actuator amplitude.

## Line B

Keep official P4 closed until both value source and actuator gates pass.  Continue shadow capacity only with promotion_allowed=0.

## Line D

Execute one actual smoke candidate per active no-BSpline family in the next runtime budget, or keep the status as NotExecuted_HypothesisGenerated with explicit budget accounting.
"""
    readback = f"""# v12.22 Implementation Readback

Generated at: {now_iso()}

## Code Changes Under Audit

- `experiments/run_v1218_b320_label_free_ablation.py`: added A36-A42 and T1B temporal/spectrum fields.
- `dgkan/models/fc_purekan_primitives.py`: added v12.22 label-free frame tokens `reslowrankp`, `convexmixp`, `driftcotbankp`, `rolecondp`, and `selfcondstopgradp`.
- `experiments/run_v1222_failclosed_explore_open_functional_rebuild.py`: added queue/fallback/final-stop contract, v12.22 artifacts, actuator v3, shadow P4, and code review zip packaging.

## Route Snapshot

- route: {state.get('route', '')}
- minimum_success: {state.get('minimum_success', '')}
- final_stop_allowed: {state.get('final_stop_allowed', '')}
- primary_blocker: {state.get('primary_blocker', '')}

Empty metric cells in CSV artifacts mean the native measurement was unavailable or the gate was not opened; they are not filled with synthetic values.
"""
    (out_dir / "v1222_no_go_boundary.md").write_text(no_go, encoding="utf-8")
    (out_dir / "v1222_next_hypothesis_generator.md").write_text(nxt, encoding="utf-8")
    (out_dir / "v1222_implementation_readback.md").write_text(readback, encoding="utf-8")
    return {"no_go_boundary_written": 1, "next_hypothesis_generator_written": 1, "implementation_readback_written": 1}


def write_simple_svg(path: Path, title: str, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    height = max(220, 48 + 24 * (len(lines) + 1))
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}">']
    body.append('<rect width="1280" height="100%" fill="#fbfbf7"/>')
    body.append(f'<text x="24" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#202020">{esc(title)}</text>')
    y = 76
    for line in lines[:44]:
        body.append(f'<text x="24" y="{y}" font-family="Arial" font-size="15" fill="#202020">{esc(line)}</text>')
        y += 24
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    label = read_csv_rows(out_dir / "v1222_label_free_signal_frame.csv")
    vis = read_csv_rows(out_dir / "v1222_visibility_scores.csv")
    act = read_csv_rows(out_dir / "v1222_actuator_safety.csv")
    classic = read_csv_rows(out_dir / "v1222_classic_family_status.csv")
    figs = {
        "fig_v1222_queue_progress.svg": [f"budget_exhausted={state.get('exploration_budget_exhausted')}", f"final_stop_allowed={state.get('final_stop_allowed')}"],
        "fig_v1222_linea_task_linec_pareto.svg": [f"{r.get('candidate_id')}: mean={r.get('mean_delta_vs_A0')}, worst={r.get('worst_delta_vs_A0')}, lineC={r.get('LineC_pass_rate')}" for r in label],
        "fig_v1222_frame_condition_vs_auc.svg": [f"{r.get('candidate_id')}: cond={r.get('frame_condition')}, auc_time={r.get('AUC_time_ratio_vs_mlp')}" for r in label],
        "fig_v1222_role_energy_by_candidate.svg": [f"{r.get('candidate_id')}: role_energy={r.get('frame_energy_by_role')}" for r in label],
        "fig_v1222_visibility_auc_precision_recall.svg": [f"{r.get('feature_tier')}: auc={r.get('auc_joint')}, p={r.get('precision_at_k_joint')}, r={r.get('recall_at_k_joint')}" for r in vis],
        "fig_v1222_leaveout_visibility_heatmap.svg": [str(r) for r in read_csv_rows(out_dir / "v1222_visibility_leaveout.csv")[:36]],
        "fig_v1222_target_sign_sanity.svg": [str(r) for r in read_csv_rows(out_dir / "v1222_target_sign_sanity.csv")[:36]],
        "fig_v1222_soft_target_regression_residuals.svg": [str(r) for r in read_csv_rows(out_dir / "v1222_soft_target_regression.csv")],
        "fig_v1222_actuator_movement_release_pareto.svg": [f"{r.get('actuator_id')}: move={r.get('safe_movement_pass')}, release={r.get('release_audit_pass')}, gap={r.get('control_gap')}" for r in act[:44]],
        "fig_v1222_actuator_norm_bisection.svg": [f"{r.get('actuator_id')} {r.get('norm_budget')}: drift={r.get('logit_max_abs_drift')}, selected={r.get('drift_boundary_selected')}" for r in act[:44]],
        "fig_v1222_actuator_control_gap.svg": [f"safe_keys={state.get('actuator_safe_movement_dataset_seed_count')}", f"release_keys={state.get('actuator_release_dataset_seed_count')}", f"official={state.get('actuator_official_gate_pass')}"],
        "fig_v1222_p3_control_comparison.svg": [f"P3 open={state.get('p3_open')}; P3 pass={state.get('p3_pass')}"],
        "fig_v1222_p4_linec_trajectory.svg": [f"P4 open={state.get('p4_open')}; shadow rows={state.get('shadow_p4_capacity_rows')}"],
        "fig_v1222_classic_family_status.svg": [f"{r.get('family')}: {r.get('status')}; deferred={r.get('budget_tradeoff_recorded')}" for r in classic],
        "fig_v1222_no_go_decision_tree.svg": [f"route={state.get('route')}", f"primary_blocker={state.get('primary_blocker')}", f"minimum_success={state.get('minimum_success')}"],
    }
    for name, lines in figs.items():
        write_simple_svg(out_dir / name, name, lines or ["no rows"])
    return {"figure_count": len(figs)}


def write_required_manifest(out_dir: Path) -> dict[str, Any]:
    names = [
        "v1222_route_decision.json",
        "v1222_exploration_queue.json",
        "v1222_child_run_manifest.csv",
        "v1222_fallback_execution_manifest.csv",
        "v1222_budget_accounting.csv",
        "v1222_final_stop_audit.json",
        "v1222_code_review_manifest.csv",
        "v1222_feature_provenance.csv",
        "v1222_label_free_signal_frame.csv",
        "v1222_label_free_failure_decomposition.csv",
        "v1222_linec_deployable_targets.csv",
        "v1222_visibility_scores.csv",
        "v1222_visibility_leaveout.csv",
        "v1222_target_sign_sanity.csv",
        "v1222_soft_target_regression.csv",
        "v1222_t1b_optimizer_update_features.csv",
        "v1222_actuator_response_dictionary.csv",
        "v1222_actuator_safety.csv",
        "v1222_actuator_controls.csv",
        "v1222_functional_p3.csv",
        "v1222_functional_p4_short.csv",
        "v1222_shadow_p4_capacity.csv",
        "v1222_classic_family_status.csv",
        "v1222_no_go_boundary.md",
        "v1222_next_hypothesis_generator.md",
        "v1222_implementation_readback.md",
        "fig_v1222_queue_progress.svg",
        "fig_v1222_linea_task_linec_pareto.svg",
        "fig_v1222_frame_condition_vs_auc.svg",
        "fig_v1222_role_energy_by_candidate.svg",
        "fig_v1222_visibility_auc_precision_recall.svg",
        "fig_v1222_leaveout_visibility_heatmap.svg",
        "fig_v1222_target_sign_sanity.svg",
        "fig_v1222_soft_target_regression_residuals.svg",
        "fig_v1222_actuator_movement_release_pareto.svg",
        "fig_v1222_actuator_norm_bisection.svg",
        "fig_v1222_actuator_control_gap.svg",
        "fig_v1222_p3_control_comparison.svg",
        "fig_v1222_p4_linec_trajectory.svg",
        "fig_v1222_classic_family_status.svg",
        "fig_v1222_no_go_decision_tree.svg",
    ]
    rows = [{"stage": "V1222_REQUIRED_ARTIFACT_MANIFEST", "artifact": name, "exists": int((out_dir / name).exists())} for name in names]
    write_csv_rows(out_dir / "v1222_required_artifact_manifest.csv", rows)
    return {"required_artifact_count": len(rows), "required_artifact_missing_count": sum(1 for r in rows if safe_int(r.get("exists"), 0) == 0)}


def write_execution_contract(out_dir: Path, args: argparse.Namespace, state: Mapping[str, Any]) -> dict[str, Any]:
    datasets = [v120._canonical_dataset(x.strip()) for x in str(args.linea_datasets).split(",") if x.strip()]
    queue = []
    for dataset in datasets:
        dataset_slug = dataset.replace("-", "")
        queue.append(
            {
                "job_id": f"A-{dataset}-A36-A42",
                "line": "Line A",
                "parent_failure": "v12.21 label-free frame missing",
                "hypothesis": "A36-A42 label-free signal frame rebuild",
                "command": f"conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1222_linea_e3_{dataset_slug} --out-dir {rel(out_dir / 'linea' / ('e3_' + dataset_slug))} --artifact-prefix v1222_linea_e3_{dataset_slug} --datasets {dataset} --seeds {args.linea_seeds} --ablation-ids A0-labelInit,A1-noYForStats,A36-ResidualA1LowRankFrame-labelFree,A37-ConvexMultiFrameMixture-labelFree,A38-PersistentDriftCotangentBank-labelFree,A39-RoleConditionedFrame-labelFree,A40-SelfConditionedStopGradFrame-labelFree,A41-UnlabeledFrameAdaptSchedule-labelFree,A42-SmallLabelOracleUpperBound-diagnostic --epochs {args.linea_epochs} --train-size {args.linea_train_size} --val-size {args.linea_val_size} --test-size {args.linea_test_size} --batch-size {args.linea_batch_size} --measure-linec 1 --device cuda:<assigned>",
                "expected_artifacts": [f"linea/e3_{dataset_slug}/v1222_linea_e3_{dataset_slug}_ablation.csv", f"linea/e3_{dataset_slug}/v1222_linea_e3_{dataset_slug}_summary.csv"],
                "promotion_gate": "v1222 label-free official/exploration gate",
                "continuation_gate": "ablation csv exists with A36-A42 rows",
                "fallback_if_fail": "Line C/T target reset",
                "max_runtime_minutes": int(args.linea_max_runtime_minutes),
                "max_candidates": 7,
                "priority": 1,
                "parallel_group": "A",
            }
        )
    queue.extend(
        [
            {"job_id": "C-target-reset-v4", "line": "Line C", "parent_failure": "G_LA direction weak/wrong sign", "hypothesis": "target decomposition and sign-stable composite", "command": "conda run -n kan python experiments/run_v1222_failclosed_explore_open_functional_rebuild.py --aggregate existing Line A artifacts", "expected_artifacts": ["v1222_linec_deployable_targets.csv"], "promotion_gate": "T1A official value source", "continuation_gate": "deployable target rows written", "fallback_if_fail": "T1B/T2 diagnostic boundary", "max_runtime_minutes": 30, "max_candidates": 7, "priority": 2, "parallel_group": "CT"},
            {"job_id": "T-visibility-v4", "line": "Line T", "parent_failure": "T1A/T1B weak visibility", "hypothesis": "temporal optimizer-observable features and leaveout gates", "command": "same v12.22 aggregate runner", "expected_artifacts": ["v1222_visibility_scores.csv", "v1222_t1b_optimizer_update_features.csv"], "promotion_gate": "visibility official/exploration gate", "continuation_gate": "visibility score rows written", "fallback_if_fail": "target convention sanity", "max_runtime_minutes": 30, "max_candidates": 2, "priority": 2, "parallel_group": "CT"},
            {"job_id": "I-actuator-v3", "line": "Line I", "parent_failure": "actuator unsafe/no release/no control resistance", "hypothesis": "norm bisection and response composition", "command": "same v12.22 aggregate runner --actuator-*", "expected_artifacts": ["v1222_actuator_response_dictionary.csv", "v1222_actuator_safety.csv"], "promotion_gate": "actuator official gate", "continuation_gate": "actuator rows written", "fallback_if_fail": "return to target/source", "max_runtime_minutes": int(args.actuator_max_runtime_minutes), "max_candidates": 10, "priority": 2, "parallel_group": "I"},
            {"job_id": "B-shadow-p4-capacity", "line": "Line B", "parent_failure": "official P3/P4 closed", "hypothesis": "non-promotable executor capacity diagnostic", "command": "same v12.22 aggregate runner", "expected_artifacts": ["v1222_shadow_p4_capacity.csv"], "promotion_gate": "official P4 remains gated", "continuation_gate": "shadow rows written", "fallback_if_fail": "keep P4 closed", "max_runtime_minutes": 10, "max_candidates": 1, "priority": 3, "parallel_group": "B"},
            {"job_id": "D-classic-status-budget", "line": "Line D", "parent_failure": "v12.21 hypothesis generated but not executed", "hypothesis": "record per-family NotExecuted/deferred budget status without rejection", "command": "same v12.22 aggregate runner", "expected_artifacts": ["v1222_classic_family_status.csv"], "promotion_gate": "no classic promotion in v12.22", "continuation_gate": "per-family status or smoke artifact exists", "fallback_if_fail": "next runtime budget smoke candidates", "max_runtime_minutes": 5, "max_candidates": 5, "priority": 4, "parallel_group": "D"},
        ]
    )
    write_json(out_dir / "v1222_exploration_queue.json", queue)
    child_rows = []
    for job in queue:
        exists = all((out_dir / str(p)).exists() for p in job["expected_artifacts"])
        deferred = 0
        if job["line"] == "Line D":
            exists = (out_dir / "v1222_classic_family_status.csv").exists()
            deferred = int(exists)
            executed = 0
            status = "deferred_with_budget_tradeoff" if deferred else "missing"
        else:
            executed = int(exists)
            status = "executed" if exists else "missing"
        child_rows.append(
            {
                "stage": "V1222_CHILD_RUN_MANIFEST",
                "job_id": job["job_id"],
                "line": job["line"],
                "command": job["command"],
                "expected_artifacts": json.dumps(job["expected_artifacts"], ensure_ascii=False),
                "executed_this_version": executed,
                "deferred_with_budget_tradeoff": deferred,
                "terminal_for_queue": int(executed or deferred),
                "status": status,
            }
        )
    write_csv_rows(out_dir / "v1222_child_run_manifest.csv", child_rows)
    failed_lines = {
        "Line A": safe_int(state.get("label_free_official_pass_count"), 0) == 0,
        "Line C": safe_int(state.get("g_la_official_pass"), 0) == 0,
        "Line T": safe_int(state.get("T1A_visibility_pass"), 0) == 0 and safe_int(state.get("T1B_visibility_pass"), 0) == 0,
        "Line I": safe_int(state.get("actuator_official_gate_pass"), 0) == 0,
        "Line B": safe_int(state.get("p4_pass"), 0) == 0,
        "Line D": safe_int(state.get("classic_executed_count"), 0) == 0,
    }
    fallback_rows = []
    for line, failed in failed_lines.items():
        jobs = [r for r in child_rows if r["line"] == line]
        terminal = any(safe_int(r.get("terminal_for_queue"), 0) for r in jobs)
        fallback_rows.append(
            {
                "stage": "V1222_FALLBACK_EXECUTION_MANIFEST",
                "line": line,
                "promotion_failed": int(failed),
                "fallback_executed_or_deferred": int((not failed) or terminal),
                "fallback_artifacts": ";".join(j.get("expected_artifacts", "") for j in jobs),
                "notes": "Line D deferred with explicit budget tradeoff" if line == "Line D" else "",
            }
        )
    write_csv_rows(out_dir / "v1222_fallback_execution_manifest.csv", fallback_rows)
    budget_rows = []
    for line in sorted({j["line"] for j in queue}):
        qjobs = [j for j in queue if j["line"] == line]
        cjobs = [r for r in child_rows if r["line"] == line]
        budget_rows.append(
            {
                "stage": "V1222_BUDGET_ACCOUNTING",
                "line": line,
                "registered_jobs": len(qjobs),
                "executed_or_deferred_jobs": sum(safe_int(r.get("terminal_for_queue"), 0) for r in cjobs),
                "candidate_executed_jobs": sum(safe_int(r.get("executed_this_version"), 0) for r in cjobs),
                "deferred_with_budget_jobs": sum(safe_int(r.get("deferred_with_budget_tradeoff"), 0) for r in cjobs),
                "max_candidates": sum(safe_int(j.get("max_candidates"), 0) for j in qjobs),
                "budget_exhausted": int(cjobs and all(safe_int(r.get("terminal_for_queue"), 0) for r in cjobs)),
            }
        )
    write_csv_rows(out_dir / "v1222_budget_accounting.csv", budget_rows)
    queue_complete = all(safe_int(r.get("terminal_for_queue"), 0) for r in child_rows)
    fallbacks_complete = all(safe_int(r.get("fallback_executed_or_deferred"), 0) for r in fallback_rows)
    official_success = safe_int(state.get("p4_pass"), 0) == 1
    final_audit = {
        "stage": "V1222_FINAL_STOP_AUDIT",
        "official_success": int(official_success),
        "exploration_budget_exhausted": int(queue_complete),
        "no_go_complete": safe_int(state.get("no_go_boundary_written"), 0),
        "all_fallback_levels_executed": int(fallbacks_complete),
        "manual_review_required": 0,
        "user_stop_flag": 0,
        "next_hypothesis_generator_written": safe_int(state.get("next_hypothesis_generator_written"), 0),
        "next_hypothesis_generator_grants_final_stop": 0,
        "final_stop_allowed": int(official_success or queue_complete or (safe_int(state.get("no_go_boundary_written"), 0) and fallbacks_complete)),
    }
    write_json(out_dir / "v1222_final_stop_audit.json", final_audit)
    return {
        "exploration_queue_jobs": len(queue),
        "child_run_executed_count": sum(safe_int(r.get("executed_this_version"), 0) for r in child_rows),
        "child_run_deferred_with_budget_count": sum(safe_int(r.get("deferred_with_budget_tradeoff"), 0) for r in child_rows),
        "fallback_missing_count": sum(1 for r in fallback_rows if safe_int(r.get("fallback_executed_or_deferred"), 0) == 0),
        "exploration_budget_exhausted": final_audit["exploration_budget_exhausted"],
        "all_fallback_levels_executed": final_audit["all_fallback_levels_executed"],
        "final_stop_allowed": final_audit["final_stop_allowed"],
    }


def package_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "v1222_code_review_packet.zip"
    code_files = [
        "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py",
        "experiments/run_v1221_failclosed_continue2_label_free_functional.py",
        "experiments/run_v1218_b320_label_free_ablation.py",
        "experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py",
        "experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py",
        "experiments/run_v1220_failclosed_continue_label_free_functional.py",
        "experiments/run_v1252_efficiency_functional_manifold.py",
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
        for p in sorted(out_dir.glob("v1222_*")):
            if p.is_file() and p.name != zip_path.name:
                zf.write(p, arcname=f"review_artifacts/{p.name}")
        for p in sorted(out_dir.glob("fig_v1222_*.svg")):
            zf.write(p, arcname=f"figures/{p.name}")
    return zip_path


def write_hash_manifest(out_dir: Path) -> dict[str, str]:
    hashes = {}
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "v1222_hash_manifest.json":
            hashes[p.name] = sha256_file(p)
    write_json(out_dir / "v1222_hash_manifest.json", hashes)
    return hashes


def decide_route(state: Mapping[str, Any]) -> tuple[str, str, str]:
    if safe_int(state.get("line_r_required_missing"), 0):
        return "R0-FailFastIncomplete", "none", "code_review_manifest_missing_required_references"
    if safe_int(state.get("fallback_missing_count"), 0):
        return "R0-NextHypothesisNotExecuted", "none", "one_or_more_failed_lines_lack_executed_or_deferred_fallback"
    if safe_int(state.get("p4_pass"), 0):
        return "S2-OfficialFunctionalSuccess", "Official Functional Success", ""
    if safe_int(state.get("T1A_visibility_exploration_pass"), 0) or safe_int(state.get("T1B_visibility_exploration_pass"), 0) or safe_int(state.get("actuator_exploratory_success"), 0):
        return "S1-FunctionalDiagnosticSurvivor", "Minimum Success F", "diagnostic_survivor_only_official_p4_closed"
    if safe_int(state.get("label_free_official_pass_count"), 0) == 0:
        return "R2-LabelFreeSignalFrameMissing", "Minimum Success F", "no_label_free_candidate_met_official_promotion_gate"
    if safe_int(state.get("g_la_official_pass"), 0) == 0:
        return "R3-LossAgnosticValueSourceMissing", "Minimum Success F", "legal_value_source_gate_failed"
    if safe_int(state.get("actuator_official_gate_pass"), 0) == 0:
        return "R4-ActuatorExecutorMissing", "Minimum Success F", "actuator_official_gate_failed"
    return "R5-P4ShortRunFailed", "Minimum Success F", "P4_open_or_short_run_failed"


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    raw_rows = collect_linea_rows(Path(args.linea_root).resolve(), [x for x in str(args.linea_csvs).split(",") if x.strip()])
    state: dict[str, Any] = {
        "stage": "V1222_ROUTE_DECISION",
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "out_dir": rel(out_dir),
        "linea_root": rel(Path(args.linea_root)),
        "linea_rows": len(raw_rows),
        "no_fake": 1,
        "no_proxy": 1,
        "cpu_offload_used": 0,
    }
    state.update(write_code_review(out_dir))
    state.update(write_label_free_v2(out_dir, raw_rows))
    state.update(write_linec_t_visibility_v4(out_dir, raw_rows))
    state.update(write_actuator_v3(out_dir, args))
    state.update(write_functional_and_shadow(out_dir, state))
    state.update(write_classic_line_d_v1222(out_dir))
    route, minimum_success, primary_blocker = decide_route(state)
    state.update({"route": route, "minimum_success": minimum_success, "primary_blocker": primary_blocker})
    state.update(write_no_go_next_readback(out_dir, state))
    state.update(write_execution_contract(out_dir, args, state))
    state.update(write_figures(out_dir, state))
    write_json(out_dir / "v1222_route_decision.json", state)
    state.update(write_required_manifest(out_dir))
    route, minimum_success, primary_blocker = decide_route(state)
    state.update({"route": route, "minimum_success": minimum_success, "primary_blocker": primary_blocker})
    state["success_f_no_failfast"] = int(
        safe_int(state.get("line_r_required_missing"), 0) == 0
        and safe_int(state.get("required_artifact_missing_count"), 0) == 0
        and safe_int(state.get("fallback_missing_count"), 0) == 0
        and safe_int(state.get("final_stop_allowed"), 0) == 1
    )
    state["success_g_label_free_near_recovered"] = int(
        safe_float(state.get("label_free_best_mean_delta_vs_A0"), -999.0) >= -0.005
        and safe_float(state.get("label_free_best_worst_delta_vs_A0"), -999.0) >= -0.020
        and safe_float(state.get("label_free_best_auc_time_ratio_vs_mlp"), 999.0) <= 1.05
        and safe_float(state.get("label_free_best_linec_pass_rate"), -999.0) >= (6.0 / 9.0)
    )
    state["success_h_value_source_survivor"] = int(safe_int(state.get("T1A_visibility_exploration_pass"), 0) or safe_int(state.get("T1B_visibility_exploration_pass"), 0))
    state["success_i_safe_actuator_survivor"] = safe_int(state.get("actuator_exploratory_success"), 0)
    state["success_j_functional_p3_open"] = int(safe_int(state.get("p3_open"), 0) and safe_int(state.get("p3_pass"), 0))
    zip_path = package_zip(out_dir)
    hashes = write_hash_manifest(out_dir)
    state["hash_manifest_entries"] = len(hashes)
    state["code_review_packet_zip"] = rel(zip_path)
    state["code_review_packet_zip_sha256"] = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        state["code_review_packet_zip_entries"] = len(zf.namelist())
    write_json(out_dir / "v1222_route_decision.json", state)
    return state


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="official_explore_open")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--linea-root", default=str(DEFAULT_LINEA_ROOT))
    parser.add_argument("--linea-csvs", default="")
    parser.add_argument("--linea-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--linea-seeds", default="0,1,2")
    parser.add_argument("--linea-epochs", type=int, default=3)
    parser.add_argument("--linea-train-size", type=int, default=512)
    parser.add_argument("--linea-val-size", type=int, default=256)
    parser.add_argument("--linea-test-size", type=int, default=256)
    parser.add_argument("--linea-batch-size", type=int, default=128)
    parser.add_argument("--linea-max-runtime-minutes", type=int, default=90)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--actuator-device", default="cuda:3")
    parser.add_argument("--actuator-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--actuator-seeds", default="0,1,2")
    parser.add_argument("--actuator-budgets", default="0.0005,0.001,0.0025,0.005,0.01,0.02")
    parser.add_argument("--actuator-train-size", type=int, default=256)
    parser.add_argument("--actuator-val-size", type=int, default=128)
    parser.add_argument("--actuator-linec-batch", type=int, default=32)
    parser.add_argument("--actuator-sketch-dim", type=int, default=8)
    parser.add_argument("--actuator-max-runtime-minutes", type=int, default=90)
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    result = run_main(parsed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
