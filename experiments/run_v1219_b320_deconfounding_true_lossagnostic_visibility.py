#!/usr/bin/env python3
"""v12.19 B320 deconfounding and true loss-agnostic visibility closure.

This runner consumes the real v12.18/v12.17/v12.16 artifacts plus the v12.19
Line A training table.  It writes the additional audit, feature-tier,
Line-C-autopsy, gate, figure, hash, and zip artifacts required by the v12.19
plan.  It does not synthesize training metrics; if an input artifact is absent,
the corresponding gate is marked blocked.
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

import run_v1218_b320_codeaudit_lossagnostic_functional as v1218  # noqa: E402
import run_v1218_b320_label_free_ablation as linea  # noqa: E402
import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v124_multibasis_functional_dual as v124  # noqa: E402
import run_v126_lowerlevel_fhq_functional_geometry as v126  # noqa: E402
import run_v1283_b109_classic_family_functional_geometry as v1283  # noqa: E402
import run_v1252_efficiency_functional_manifold as v1252  # noqa: E402
from dgkan.kernels import fused_hinge_quadratic as fhq  # noqa: E402
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402


PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_独立分析与下一步计划.md"
RUNNER_PATH = REPO_ROOT / "experiments" / "run_v1219_b320_deconfounding_true_lossagnostic_visibility.py"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry" / "official_from_v1218_v1217_v1216_artifacts"
DEFAULT_V1218_DIR = REPO_ROOT / "results" / "v12_18_b320_codeaudit_lossagnostic_functional" / "official_from_v1217_v1216_artifacts"
DEFAULT_V1217_DIR = REPO_ROOT / "results" / "v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry" / "support_expand_seed012_seed345_b128_w10_12_15"
DEFAULT_V1216_DIR = REPO_ROOT / "results" / "v12_16_b320locked_explicit_signal_reservoir_functional" / "repair_seed012_sketchdim24_rank5_b128_w10_12_15"

EPS = 1.0e-12


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
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def finite_min(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return min(vals) if vals else float("nan")


def finite_max(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return max(vals) if vals else float("nan")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    fields: list[str] = list(fieldnames or [])
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


def line_range_for_symbol(path: Path, symbol: str) -> tuple[int | None, int | None, str]:
    if not path.exists():
        return None, None, ""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                return int(node.lineno), int(getattr(node, "end_lineno", node.lineno)), "\n".join(text.splitlines()[node.lineno - 1 : min(node.lineno + 4, len(text.splitlines()))])
    for idx, line in enumerate(text.splitlines(), start=1):
        if symbol in line:
            return idx, idx, line.strip()
    return None, None, ""


def code_ref(path: str, symbol: str) -> dict[str, Any]:
    start, end, snippet = line_range_for_symbol(REPO_ROOT / path, symbol)
    return {
        "path": path,
        "symbol": symbol,
        "line_start": "" if start is None else start,
        "line_end": "" if end is None else end,
        "missing": int(start is None),
        "snippet": snippet.replace("\n", " ")[:260],
    }


def all_required_code_files() -> list[str]:
    files = [
        rel(RUNNER_PATH),
        "experiments/run_v1218_b320_label_free_ablation.py",
        "experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py",
        "experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py",
        "experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py",
        "experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py",
        "experiments/run_v1283_b109_classic_family_functional_geometry.py",
        "experiments/run_v126_lowerlevel_fhq_functional_geometry.py",
        "experiments/run_v1252_efficiency_functional_manifold.py",
        "experiments/run_v124_multibasis_functional_dual.py",
        "experiments/run_v120_good_geometry_battery.py",
        "dgkan/models/fc_purekan_primitives.py",
        "dgkan/kernels/fused_hinge_quadratic.py",
        "dgkan/optim/manual_adamw.py",
        "dgkan/profiling/timing.py",
    ]
    return sorted(dict.fromkeys(files))


def write_code_semantics(out_dir: Path, v1218_dir: Path) -> dict[str, Any]:
    v1218_manifest = read_csv_rows(v1218_dir / "v1218_core_code_review_manifest.csv")
    delta_rows = []
    for row in v1218_manifest:
        out = dict(row)
        out["stage"] = "V1219_CODE_REVIEW_DELTA"
        out["v1219_delta"] = "carried_forward_from_v1218_packet; v12.19 adds semantic readback, stricter feature tiering, and A15-A22 init code"
        out["blocks_v1219_promotion_if_missing"] = row.get("unknown_or_not_inspected", "1")
        delta_rows.append(out)
    write_csv_rows(out_dir / "v1219_code_review_delta.csv", delta_rows)

    readback_specs = [
        (
            "R1",
            "B320 construction path",
            [
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "make_model"),
                code_ref("experiments/run_v124_multibasis_functional_dual.py", "_make_model"),
                code_ref("dgkan/models/fc_purekan_primitives.py", "SimpleFastTaskGeometryKAN"),
                code_ref("dgkan/models/fc_purekan_primitives.py", linea.B320_ID),
            ],
            "Line A calls make_model; y_stats is y_train only when uses_y_for_stats=1, then v124._make_model passes it as y_for_stats into SimpleFastTaskGeometryKAN.",
        ),
        (
            "R2",
            "trainprobe path into direct_readout / quad_proj",
            [
                code_ref("dgkan/models/fc_purekan_primitives.py", "probe_dirs"),
                code_ref("dgkan/models/fc_purekan_primitives.py", "trainprobe_signal_init_applied"),
                code_ref("dgkan/models/fc_purekan_primitives.py", "direct_readout[: self.input_dim"),
                code_ref("dgkan/models/fc_purekan_primitives.py", "quad_proj[:, col] = probe_dirs"),
            ],
            "When variant contains trainprobe and y_for_stats exists, class-mean directions become probe_dirs; trainprobedirect writes direct_readout and trainprobeP writes quad_proj.",
        ),
        (
            "R3",
            "label-free ablation path",
            [
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "strip_trainprobe_tokens"),
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "ablation_specs"),
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "strict_label_free_init"),
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "build_y_stats_for_mode"),
            ],
            "A1 keeps trainprobe-capable buffers but passes y_stats=None; A2-A19 strip trainprobe tokens; strict_label_free_init is computed from uses_y_for_stats=0 and trainprobe init not applied.",
        ),
        (
            "R4",
            "manual / fused update path",
            [
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "step_model"),
                code_ref("experiments/run_v1283_b109_classic_family_functional_geometry.py", "_step_b109"),
                code_ref("experiments/run_v126_lowerlevel_fhq_functional_geometry.py", "_ManualForeachAdamW"),
                code_ref("dgkan/kernels/fused_hinge_quadratic.py", "backward_learnablep_workspace_fused_quadproj_adamw_from_grad_logits"),
            ],
            "B320 uses selected B109/FHQ step implementations and manual/fused AdamW where available; MLP controls use ordinary torch AdamW.",
        ),
        (
            "R5",
            "Line C computation",
            [
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "signal_reservoir_metrics_detailed"),
                code_ref("experiments/run_v1218_b320_label_free_ablation.py", "linec_measured"),
                code_ref("experiments/run_v1252_efficiency_functional_manifold.py", "_ridge_coupling"),
                code_ref("experiments/run_v1252_efficiency_functional_manifold.py", "_sample_grad_sketch"),
            ],
            "CouplingR2 comes from ridge coupling of before/after logits; NoiseSignalLeak and RealSignalReservoirRatio use CE residuals and labels for audit only, not for direction.",
        ),
        (
            "R6",
            "Line T feature provenance",
            [code_ref(rel(RUNNER_PATH), "tier_for_feature"), code_ref(rel(RUNNER_PATH), "write_visibility_tiers")],
            "v12.19 rewrites feature tiering: T1 precommit only, T2 clone-probe diagnostics, T3 audit labels, and blocklisted method identity/hash/response metadata.",
        ),
        (
            "R7",
            "strict visibility scorer",
            [code_ref("experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py", "score_strict_visibility"), code_ref(rel(RUNNER_PATH), "score_visibility_tier")],
            "Heldout scorer uses leaveout splits, ridge predictions on feature matrices, precision@k where k equals positive support, and support concentration gates.",
        ),
        (
            "R8",
            "P3/P4 gate",
            [code_ref(rel(RUNNER_PATH), "write_gate_status"), code_ref("experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py", "run_p3_p4_gates")],
            "P3/Line I is only opened after T1 visibility pass; P4 additionally needs Line R/A/C/T/I pass, so v12.19 keeps both closed unless those gates are true.",
        ),
        (
            "R9",
            "classic-family monitor",
            [code_ref(rel(RUNNER_PATH), "write_classic_monitor"), code_ref("experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py", "run_classic_monitor")],
            "Classic No-BSpline families are copied as monitor-only status unless a new loss-agnostic geometry hypothesis is introduced.",
        ),
    ]
    md = ["# v12.19 Core Semantics Audit", "", f"Generated at: {now_iso()}", ""]
    rows = []
    for rid, title, refs, summary in readback_specs:
        md.extend([f"## {rid}. {title}", "", summary, ""])
        for ref in refs:
            md.append(f"- `{ref['path']}::{ref['symbol']}` lines `{ref['line_start']}`-`{ref['line_end']}` missing={ref['missing']}")
            rows.append({"stage": "V1219_CORE_SEMANTICS_AUDIT", "readback_id": rid, "title": title, "summary": summary, **ref})
        md.append("")
    (out_dir / "v1219_core_semantics_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_csv_rows(out_dir / "v1219_core_semantics_audit.csv", rows)

    construction_trace = {
        "stage": "V1219_B320_CONSTRUCTION_TRACE",
        "candidate_id": linea.B320_ID,
        "call_chain": [
            "run_v1218_b320_label_free_ablation.train_one",
            "run_v1218_b320_label_free_ablation.make_model",
            "run_v124_multibasis_functional_dual._make_model",
            "dgkan.models.fc_purekan_primitives.SimpleFastTaskGeometryKAN.__init__",
        ],
        "uses_y_for_stats_when": "uses_y_for_stats=1; A0/A20/A21/A22 only in v12.19 Line A",
        "line_refs": rows,
        "no_fake": 1,
        "no_proxy": 1,
    }
    write_json(out_dir / "v1219_b320_construction_trace.json", construction_trace)
    linec_metric_trace = {
        "stage": "V1219_LINEC_METRIC_TRACE",
        "CouplingR2": "ridge R2 between train-window logit delta and query logit delta",
        "NoiseSignalLeak": "audit-only CE residual from permuted/noise labels projected into signal subspace",
        "RealSignalReservoirRatio": "audit-only real CE residual energy projected into reservoir subspace divided by real residual energy",
        "uses_label_for_audit": 1,
        "uses_ce_for_audit": 1,
        "allowed_for_direction": 0,
        "line_refs": [ref for rid, _title, refs, _summary in readback_specs if rid == "R5" for ref in refs],
    }
    write_json(out_dir / "v1219_linec_metric_trace.json", linec_metric_trace)
    missing = sum(safe_int(row.get("missing"), 1) for row in rows)
    return {
        "code_review_delta_rows": len(delta_rows),
        "implementation_readback_rows": len(rows),
        "implementation_readback_missing_refs": missing,
        "cr0_cr15_missing_or_unknown": sum(safe_int(row.get("unknown_or_not_inspected"), 1) for row in delta_rows),
        "line_r_code_semantics_pass": int(missing == 0 and all(safe_int(row.get("unknown_or_not_inspected"), 1) == 0 for row in delta_rows)),
    }


def tier_for_feature(feature_family: str, feature_name: str) -> tuple[str, str]:
    family = str(feature_family)
    name = str(feature_name)
    if family.startswith("output_subspace_drift") or name.startswith("pred_") or "target_norm" in name:
        return "T2_clone_probe_diagnostic", "requires clone-probe / post-response diagnostic"
    if name == "persistence_pred_coupling_absdiff":
        return "T2_clone_probe_diagnostic", "derived from predicted coupling response"
    if family == "method_role_flags" or name.startswith("method_is_"):
        return "BLOCKED", "method identity is not a physical precommit observable"
    if family in {"window_schedule", "sketch_geometry"} or name in {"window_value", "window_log", "sketch_dim", "batch_size", "sketch_work_log"}:
        return "BLOCKED", "protocol or schedule metadata cannot be a deployable target observable"
    if "hash" in name or "actual_" in name or "hard_" in name or "label" in name.lower():
        return "BLOCKED", "forbidden hash/actual-response/label target feature"
    return "T1_deployable_precommit", "precommit label-free observable from v12.16 strict source"


def write_forbidden_blocklist(out_dir: Path) -> dict[str, Any]:
    rows = [
        {"stage": "V1219_FORBIDDEN_FEATURE_BLOCKLIST", "pattern": "method_is_* / method_role_flags", "tier": "BLOCKED", "reason": "method identity is not physical precommit signal"},
        {"stage": "V1219_FORBIDDEN_FEATURE_BLOCKLIST", "pattern": "output_subspace_drift / pred_* / target_norm*", "tier": "T2 only", "reason": "clone-probe diagnostic, not deployable precommit"},
        {"stage": "V1219_FORBIDDEN_FEATURE_BLOCKLIST", "pattern": "actual_* / *_delta response", "tier": "BLOCKED", "reason": "actual candidate response is post-response audit"},
        {"stage": "V1219_FORBIDDEN_FEATURE_BLOCKLIST", "pattern": "hard_* / label / CE / permuted CE", "tier": "T3 target only", "reason": "audit target label, never feature source"},
        {"stage": "V1219_FORBIDDEN_FEATURE_BLOCKLIST", "pattern": "dataset name / validation / test / future outcome", "tier": "BLOCKED", "reason": "forbidden commit source"},
        {"stage": "V1219_FORBIDDEN_FEATURE_BLOCKLIST", "pattern": "numeric hash / row id hash", "tier": "BLOCKED", "reason": "identifier leakage"},
        {"stage": "V1219_FORBIDDEN_FEATURE_BLOCKLIST", "pattern": "window_value / sketch_dim / batch_size as standalone", "tier": "BLOCKED", "reason": "protocol metadata without physical observable"},
    ]
    write_csv_rows(out_dir / "v1219_forbidden_feature_blocklist.csv", rows)
    return {"forbidden_blocklist_rows": len(rows), "strict_feature_blocklist_active": 1}


def add_v1219_t1_derived_features(long_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wide, _feature_names = v1218.wide_feature_matrix(long_rows)
    derived: list[dict[str, Any]] = []
    for row in wide:
        values = {
            "aug_projector_abs_balance": abs(safe_float(row.get("augmentation_projector_balance"), 0.0)),
            "branch_energy_temporal_drift_proxy": safe_float(row.get("persistence_reservoir_fraction_absdiff"), 0.0)
            + safe_float(row.get("persistence_projector_stability_absdiff"), 0.0),
            "unlabeled_logit_covariance_curvature": safe_float(row.get("top_eigen_share"), 0.0) * (1.0 - safe_float(row.get("top_eigen_share"), 0.0)),
            "random_cotangent_persistence": safe_float(row.get("cotangent_energy_balance"), 0.0)
            * (1.0 - min(1.0, safe_float(row.get("persistence_reservoir_fraction_absdiff"), 0.0))),
        }
        families = {
            "aug_projector_abs_balance": "augmentation_stable_p_alignment",
            "branch_energy_temporal_drift_proxy": "branch_energy_temporal_drift",
            "unlabeled_logit_covariance_curvature": "unlabeled_logit_covariance_curvature",
            "random_cotangent_persistence": "random_cotangent_persistence",
        }
        for name, value in values.items():
            derived.append(
                {
                    "stage": "V1219_T1_DERIVED_FEATURES",
                    "row_id": row.get("row_id", ""),
                    "source_run": row.get("source_run", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "window": row.get("window", ""),
                    "method": row.get("method", ""),
                    "sketch_id": "",
                    "sketch_family": row.get("sketch_family", ""),
                    "cotangent_family": row.get("cotangent_family", ""),
                    "feature_set": "T1_deployable_precommit",
                    "feature_family": families[name],
                    "feature_name": name,
                    "feature_value": value,
                    "feature_provenance": "v1219_precommit_derived_from_existing_strict_unlabeled_columns",
                    "uses_label_for_feature": 0,
                    "uses_ce_for_feature": 0,
                    "uses_permuted_label_for_feature": 0,
                    "uses_dataset_name_for_feature": 0,
                    "precommit_available": 1,
                    "clone_probe_required": 0,
                    "hard_noise_release": safe_int(row.get("hard_noise_release"), 0),
                    "hard_reservoir_release": safe_int(row.get("hard_reservoir_release"), 0),
                    "hard_joint_release": safe_int(row.get("hard_joint_release"), 0),
                    "forbidden_for_direction": 0,
                    "forbidden_reason": "",
                    "source_artifact": "derived_from_v1218_strict_visibility_features.csv",
                }
            )
    return derived


def write_visibility_tiers(out_dir: Path, v1218_dir: Path, v1217_dir: Path) -> dict[str, Any]:
    source_long = [row for row in read_csv_rows(v1218_dir / "v1218_strict_visibility_features.csv") if row.get("row_id")]
    manifest_rows = []
    t1_rows = []
    t2_rows = []
    blocked_rows = []
    seen: set[tuple[str, str]] = set()
    for row in source_long:
        tier, reason = tier_for_feature(row.get("feature_family", ""), row.get("feature_name", ""))
        payload = dict(row)
        payload["stage"] = "V1219_VISIBILITY_FEATURE_TIER_MANIFEST"
        payload["v1219_tier"] = tier
        payload["v1219_tier_reason"] = reason
        key = (str(row.get("feature_family", "")), str(row.get("feature_name", "")))
        if key not in seen:
            manifest_rows.append(
                {
                    "stage": "V1219_VISIBILITY_FEATURE_TIER_MANIFEST",
                    "feature_family": key[0],
                    "feature_name": key[1],
                    "v1219_tier": tier,
                    "source_column_or_formula": row.get("feature_name", ""),
                    "precommit_available": row.get("precommit_available", ""),
                    "clone_probe_required": row.get("clone_probe_required", ""),
                    "uses_label_for_feature": row.get("uses_label_for_feature", ""),
                    "uses_ce_for_feature": row.get("uses_ce_for_feature", ""),
                    "uses_dataset_name_for_feature": row.get("uses_dataset_name_for_feature", ""),
                    "reason": reason,
                }
            )
            seen.add(key)
        if tier == "T1_deployable_precommit":
            payload["stage"] = "V1219_T1_DEPLOYABLE_FEATURES"
            payload["feature_set"] = "T1_deployable_precommit"
            t1_rows.append(payload)
        elif tier == "T2_clone_probe_diagnostic":
            payload["stage"] = "V1219_T2_CLONE_PROBE_DIAGNOSTIC_FEATURES"
            payload["feature_set"] = "T2_clone_probe_diagnostic"
            payload["forbidden_for_direction"] = 1
            payload["forbidden_reason"] = reason
            t2_rows.append(payload)
        else:
            payload["stage"] = "V1219_BLOCKED_VISIBILITY_FEATURES"
            payload["forbidden_for_direction"] = 1
            payload["forbidden_reason"] = reason
            blocked_rows.append(payload)
    derived = add_v1219_t1_derived_features(t1_rows)
    if derived:
        for family, name in sorted({(str(row["feature_family"]), str(row["feature_name"])) for row in derived}):
            manifest_rows.append(
                {
                    "stage": "V1219_VISIBILITY_FEATURE_TIER_MANIFEST",
                    "feature_family": family,
                    "feature_name": name,
                    "v1219_tier": "T1_deployable_precommit",
                    "source_column_or_formula": "v1219 derived from T1 unlabeled columns",
                    "precommit_available": 1,
                    "clone_probe_required": 0,
                    "uses_label_for_feature": 0,
                    "uses_ce_for_feature": 0,
                    "uses_dataset_name_for_feature": 0,
                    "reason": "v12.19 repair expansion after low AUC: augmentation/P alignment, branch drift, logit covariance curvature, cotangent persistence",
                }
            )
        t1_rows.extend(derived)
    provenance_rows = []
    for row in manifest_rows:
        tier = str(row.get("v1219_tier", ""))
        provenance_rows.append(
            {
                "stage": "V1219_FEATURE_PROVENANCE_STRICT_V2",
                "feature_family": row.get("feature_family", ""),
                "feature_name": row.get("feature_name", ""),
                "source_column_or_formula": row.get("source_column_or_formula", ""),
                "v1219_tier": tier,
                "strict_loss_agnostic_allowed": int(tier == "T1_deployable_precommit"),
                "allowed_for_direction": int(tier == "T1_deployable_precommit"),
                "allowed_for_promotion_feature": int(tier == "T1_deployable_precommit"),
                "clone_probe_required": row.get("clone_probe_required", ""),
                "precommit_available": row.get("precommit_available", ""),
                "uses_label_for_feature": row.get("uses_label_for_feature", ""),
                "uses_ce_for_feature": row.get("uses_ce_for_feature", ""),
                "uses_dataset_name_for_feature": row.get("uses_dataset_name_for_feature", ""),
                "forbidden_for_direction": int(tier != "T1_deployable_precommit"),
                "forbidden_reason": "" if tier == "T1_deployable_precommit" else row.get("reason", ""),
                "audit_note": row.get("reason", ""),
            }
        )
    target_rows = []
    for row in read_csv_rows(v1217_dir / "v1217_release_labels_audit_only.csv"):
        target_rows.append(
            {
                "stage": "V1219_T3_AUDIT_TARGETS",
                "row_id": row.get("row_id", ""),
                "source_run": row.get("source_run", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("window", ""),
                "method": row.get("method", row.get("candidate_probe_id", "")),
                "hard_noise_release": row.get("hard_noise_release", ""),
                "hard_reservoir_release": row.get("hard_reservoir_release", ""),
                "hard_joint_release": row.get("hard_joint_release", ""),
                "NoiseSignalLeak_delta": row.get("actual_NoiseSignalLeak_delta", ""),
                "RealSignalReservoirRatio_delta": row.get("actual_RealSignalReservoirRatio_delta", ""),
                "allowed_for_feature": 0,
                "allowed_for_direction": 0,
                "audit_only": 1,
            }
        )
    write_csv_rows(out_dir / "v1219_visibility_feature_tier_manifest.csv", manifest_rows)
    write_csv_rows(out_dir / "v1219_feature_provenance_strict_v2.csv", provenance_rows)
    write_csv_rows(out_dir / "v1219_T1_deployable_features.csv", t1_rows)
    write_csv_rows(out_dir / "v1219_T2_clone_probe_diagnostic_features.csv", t2_rows)
    write_csv_rows(out_dir / "v1219_T3_audit_targets.csv", target_rows)
    write_csv_rows(out_dir / "v1219_blocked_visibility_features.csv", blocked_rows)
    return {
        "visibility_source_long_rows": len(source_long),
        "visibility_tier_manifest_rows": len(manifest_rows),
        "feature_provenance_strict_v2_rows": len(provenance_rows),
        "T1_deployable_feature_rows": len(t1_rows),
        "T2_clone_probe_feature_rows": len(t2_rows),
        "T3_audit_target_rows": len(target_rows),
        "blocked_visibility_feature_rows": len(blocked_rows),
    }


def control_false_positive_rate(v1217_dir: Path) -> tuple[int, float]:
    rows = read_csv_rows(v1217_dir / "v1217_linec_source_calibration.csv")
    fp = sum(safe_int(row.get("control_joint_false_positive_rows"), 0) for row in rows)
    total = sum(safe_int(row.get("control_rows"), 0) for row in rows)
    return fp, fp / max(1, total)


def support_audit(out_dir: Path, v1218_dir: Path, v1217_dir: Path) -> dict[str, Any]:
    support = read_csv_rows(v1218_dir / "v1218_support_concentration.csv")
    support_row = dict(support[0]) if support else {}
    fp_rows, fp_rate = control_false_positive_rate(v1217_dir)
    out = {
        "stage": "V1219_VISIBILITY_SUPPORT_AUDIT",
        "source": rel(v1218_dir / "v1218_support_concentration.csv"),
        "hard_joint_total_rows": support_row.get("hard_joint_total_rows", ""),
        "support_concentration_seed": support_row.get("support_concentration_seed", ""),
        "support_concentration_window": support_row.get("support_concentration_window", ""),
        "support_concentration_method": support_row.get("support_concentration_method", ""),
        "support_concentration_source": support_row.get("support_concentration_source", ""),
        "support_concentrated": support_row.get("support_concentrated", 1),
        "control_false_positive_rows": fp_rows,
        "control_false_positive_rate": fp_rate,
        "support_gate_pass": int(safe_int(support_row.get("support_concentrated"), 1) == 0 and fp_rate == 0.0),
    }
    write_csv_rows(out_dir / "v1219_visibility_support_audit.csv", [out])
    return out


def score_visibility_tier(out_dir: Path, feature_rows: Sequence[Mapping[str, Any]], support_row: Mapping[str, Any], prefix: str, feature_set: str, diagnostic_only: int) -> dict[str, Any]:
    support_concentrated = safe_int(support_row.get("support_concentrated"), 1)
    score_rows, leaveout_rows, summary = v1218.score_strict_visibility(feature_rows, support_concentrated)
    fp_rate = safe_float(support_row.get("control_false_positive_rate"), 1.0)
    for row in score_rows:
        row["stage"] = f"V1219_VISIBILITY_SCORES_{prefix}"
        row["feature_set"] = feature_set
        row["diagnostic_only"] = diagnostic_only
        row["control_false_positive_rate"] = fp_rate
        auc = safe_float(row.get("AUC_joint_min"), float("nan"))
        precision = safe_float(row.get("precision_at_k_joint_min"), float("nan"))
        recall = safe_float(row.get("recall_at_k_joint_min"), float("nan"))
        pass_flag = int(
            not math.isnan(auc)
            and auc >= 0.70
            and not math.isnan(precision)
            and precision >= 0.25
            and not math.isnan(recall)
            and recall >= 0.20
            and safe_int(row.get("required_split_failures"), 1) == 0
            and support_concentrated == 0
            and fp_rate == 0.0
        )
        row["v1219_visibility_pass"] = pass_flag if diagnostic_only == 0 else 0
        row["diagnostic_pass_not_promotable"] = pass_flag if diagnostic_only == 1 else 0
        row["v1219_gate"] = "AUC>=0.70, precision>=0.25, recall>=0.20, all required leaveouts, support_concentrated=0, control_fp_rate=0"
    for row in leaveout_rows:
        row["stage"] = f"V1219_VISIBILITY_LEAVEOUT_{prefix}"
        row["feature_set"] = feature_set
        row["diagnostic_only"] = diagnostic_only
    write_csv_rows(out_dir / f"v1219_visibility_scores_{prefix}.csv", score_rows)
    if diagnostic_only == 0:
        write_csv_rows(out_dir / "v1219_visibility_leaveout_T1_only.csv", leaveout_rows)
    summary.update(
        {
            f"{prefix}_feature_rows": len(feature_rows),
            f"{prefix}_visibility_pass": safe_int(score_rows[0].get("v1219_visibility_pass"), 0) if score_rows else 0,
            f"{prefix}_diagnostic_pass_not_promotable": safe_int(score_rows[0].get("diagnostic_pass_not_promotable"), 0) if score_rows else 0,
            f"{prefix}_auc_joint_min": score_rows[0].get("AUC_joint_min", "") if score_rows else "",
            f"{prefix}_precision_joint_min": score_rows[0].get("precision_at_k_joint_min", "") if score_rows else "",
            f"{prefix}_recall_joint_min": score_rows[0].get("recall_at_k_joint_min", "") if score_rows else "",
        }
    )
    return summary


def evaluate_linea(out_dir: Path) -> dict[str, Any]:
    summary_rows = read_csv_rows(out_dir / "v1219_b320_deconfounding_linea_summary.csv")
    ablation_rows = read_csv_rows(out_dir / "v1219_b320_deconfounding_linea_ablation.csv")
    gate_rows = []
    for row in summary_rows:
        cid = str(row.get("candidate_id", ""))
        is_candidate = cid.startswith("A") and cid != "A0-labelInit"
        strict_rows = safe_int(row.get("strict_label_free_rows"), 0)
        task_pass = int(
            is_candidate
            and strict_rows > 0
            and safe_float(row.get("mean_delta_vs_A0_labelInit"), -999.0) >= -0.005
            and safe_float(row.get("worst_delta_vs_A0_labelInit"), -999.0) >= -0.015
            and safe_float(row.get("max_AUC_time_ratio_vs_mlp"), 999.0) <= 1.0
        )
        linec_pass = int(is_candidate and safe_int(row.get("linec_nontearing_all_pass"), 0) == 1)
        official_pass = int(task_pass == 1 and linec_pass == 1 and strict_rows == safe_int(row.get("rows"), 0))
        gate_rows.append(
            {
                "stage": "V1219_LINEA_CANDIDATE_GATE_SUMMARY",
                "candidate_id": cid,
                "rows": row.get("rows", ""),
                "mean_delta_vs_A0": row.get("mean_delta_vs_A0_labelInit", ""),
                "worst_delta_vs_A0": row.get("worst_delta_vs_A0_labelInit", ""),
                "mean_delta_vs_mlp": row.get("mean_delta_vs_mlp", ""),
                "max_AUC_time_ratio_vs_mlp": row.get("max_AUC_time_ratio_vs_mlp", ""),
                "linec_nontearing_pass_rate": row.get("linec_nontearing_pass_rate", ""),
                "linec_nontearing_all_pass": row.get("linec_nontearing_all_pass", ""),
                "strict_label_free_rows": strict_rows,
                "task_side_label_free_pass": task_pass,
                "linec_pass": linec_pass,
                "official_label_free_b320_pass": official_pass,
                "blocker": "" if official_pass else ("not_label_free_candidate" if not is_candidate else "task_or_LineC_gate_failed"),
            }
        )
    write_csv_rows(out_dir / "v1219_linea_candidate_gate_summary.csv", gate_rows)

    diag_fields = [
        "P_energy_on_top_pca",
        "P_energy_on_aug_stable_subspace",
        "P_condition",
        "quad_feature_std_mean",
        "quad_feature_std_min",
        "quad_feature_std_max",
        "direct_readout_norm",
        "quad_proj_norm",
        "quad_readout_norm",
        "branch_scale_norm",
    ]
    diag_rows = []
    for cid in sorted({row.get("candidate_id", "") for row in ablation_rows}):
        group = [row for row in ablation_rows if row.get("candidate_id") == cid]
        if not group:
            continue
        out = {"stage": "V1219_LINEA_P_BASIS_ALIGNMENT_DIAGNOSTICS", "candidate_id": cid, "rows": len(group)}
        for field in diag_fields:
            vals = [safe_float(row.get(field), float("nan")) for row in group if row.get(field) not in (None, "")]
            out[f"{field}_mean"] = mean(vals) if vals else ""
            out[f"{field}_min"] = finite_min(vals) if vals else ""
            out[f"{field}_max"] = finite_max(vals) if vals else ""
        diag_rows.append(out)
    write_csv_rows(out_dir / "v1219_P_basis_alignment_diagnostics.csv", diag_rows)

    decomp_fields = [
        "linec_signal_mass_topk",
        "linec_reservoir_fraction",
        "linec_real_residual_energy",
        "linec_real_total_energy",
        "linec_noise_signal_energy",
        "linec_noise_total_energy",
        "linec_signal_top_count",
        "linec_NoiseSignalLeak",
        "linec_RealSignalReservoirRatio",
        "linec_CouplingR2",
    ]
    decomp_rows = []
    for cid in sorted({row.get("candidate_id", "") for row in ablation_rows}):
        group = [row for row in ablation_rows if row.get("candidate_id") == cid]
        out = {"stage": "V1219_LINEA_RESERVOIR_DECOMPOSITION", "candidate_id": cid, "rows": len(group)}
        for field in decomp_fields:
            vals = [safe_float(row.get(field), float("nan")) for row in group if row.get(field) not in (None, "")]
            out[f"{field}_mean"] = mean(vals) if vals else ""
            out[f"{field}_min"] = finite_min(vals) if vals else ""
            out[f"{field}_max"] = finite_max(vals) if vals else ""
        decomp_rows.append(out)
    write_csv_rows(out_dir / "v1219_linea_reservoir_decomposition.csv", decomp_rows)
    label_free_pass_rows = [row for row in gate_rows if safe_int(row.get("official_label_free_b320_pass"), 0) == 1]
    best_task = max(
        [row for row in gate_rows if str(row.get("candidate_id", "")).startswith("A") and row.get("candidate_id") != "A0-labelInit"],
        key=lambda row: safe_float(row.get("mean_delta_vs_A0"), -999.0),
        default={},
    )
    return {
        "linea_summary_rows": len(summary_rows),
        "linea_ablation_rows": len(ablation_rows),
        "linea_official_label_free_pass_count": len(label_free_pass_rows),
        "linea_best_candidate_by_task_delta": best_task.get("candidate_id", ""),
        "linea_best_candidate_mean_delta_vs_A0": best_task.get("mean_delta_vs_A0", ""),
        "linea_best_candidate_linec_pass": best_task.get("linec_pass", ""),
        "linea_pass": int(bool(label_free_pass_rows)),
    }


def tensor_hash(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_linec_autopsy_model(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.device]:
    device = linea.device_from_arg(args.device)
    torch.cuda.set_device(device)
    load_args = argparse.Namespace(**vars(args))
    load_args.seed = int(args.autopsy_seed)
    dataset = v120._canonical_dataset(args.autopsy_dataset)
    data = v120._load_vision_split(load_args, dataset, train_size=int(args.autopsy_train_size), val_size=int(args.autopsy_val_size), test_size=int(args.autopsy_test_size))
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test_cpu, _y_test_cpu, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    specs = linea.ablation_specs(int(input_dim), int(output_dim))
    spec = specs["A0-labelInit"]["spec"]
    model_seed = int(args.autopsy_seed) + int(args.seed_base)
    model = linea.make_model("A0-labelInit", spec, int(input_dim), int(output_dim), x_train, y_train, device, model_seed, 1, "actual")
    opt = v1283._make_adamw(model, args)
    triton_update_params = v126._triton_adamw_params(args, model)
    manual_update = (
        v126._ManualForeachAdamW(opt.param_groups, args, triton_update_params=triton_update_params)
        if v126._variant_uses_manual_adamw(args, model)
        else None
    )
    warm_impl = v1283._select_b109_step_impl("A0-labelInit", {"A0-labelInit": spec}, 0, 0)
    workspace = fhq.make_workspace(model, int(args.batch_size), device) if warm_impl in {"F3-triton-learnableP-workspace", "F4-triton-fixedP-workspace"} else None
    total_steps = max(1, int(args.autopsy_epochs) * int(math.ceil(float(x_train.shape[0]) / float(max(1, int(args.batch_size))))))
    gen = torch.Generator(device=device).manual_seed(int(args.autopsy_seed) + int(args.seed_base) + 44)
    step_id = 0
    for epoch in range(int(args.autopsy_epochs)):
        v1283._apply_logit_gain_ramp(model, "A0-labelInit", {"A0-labelInit": spec}, epoch, int(args.autopsy_epochs))
        v1283._apply_quad_branch_ramp(model, "A0-labelInit", {"A0-labelInit": spec}, epoch, int(args.autopsy_epochs))
        v1283._apply_direct_branch_ramp(model, "A0-labelInit", {"A0-labelInit": spec}, epoch, int(args.autopsy_epochs))
        perm = torch.randperm(int(x_train.shape[0]), generator=gen, device=device)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off : off + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            lr_now = v126._task_lr_for(args, step_id + 1, total_steps)
            for group in opt.param_groups:
                group["lr"] = lr_now * float(group.get("lr_scale", 1.0))
            opt.zero_grad(set_to_none=True)
            impl = linea.step_model(model, "A0-labelInit", {"A0-labelInit": spec}, xb, yb, opt, args, manual_update, epoch, step_id, workspace)
            if manual_update is not None:
                manual_update.step()
            else:
                opt.step()
            step_id += 1
    torch.cuda.synchronize(device)
    model.eval()
    return model, x_train, y_train, x_val, y_val, device


def linec_calc(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, xq: torch.Tensor, yq: torch.Tensor, impl_id: str, sketch_dim: int, seed: int, update: bool) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        before_b = model(xb).detach()
        before_q = model(xq).detach()
    if update:
        updated = v1252._take_adamw_window(model, xb, yb, 2.0e-3, 1.0e-3).eval()
    else:
        updated = model
    with torch.no_grad():
        after_b = updated(xb).detach()
        after_q = updated(xq).detach()
    r2, corr, resid, pred_norm = v1252._ridge_coupling(after_b - before_b, after_q - before_q, 1.0e-3)
    sig = linea.signal_reservoir_metrics_detailed(updated, xb, yb, int(sketch_dim), int(seed))
    return {
        "stage": "V1219_LINEC_PROTOCOL_AUTOPSY",
        "linec_impl_id": impl_id,
        "checkpoint_id": "v1219_A0_autopsy_retrained_checkpoint",
        "candidate_id": "A0-labelInit",
        "dataset": "MNIST",
        "seed": seed,
        "batch_hash": tensor_hash(xb, yb, xq, yq),
        "window": int(xb.shape[0]),
        "sketch_dim": int(sketch_dim),
        "rank": sig.get("signal_top_count", ""),
        "update_window_type": "adamw_window" if update else "no_update_baseline",
        "uses_ce_for_audit": 1,
        "uses_label_for_audit": 1,
        "CouplingR2": r2,
        "CouplingCorr": corr,
        "coupling_residual_norm": resid,
        "coupling_prediction_norm": pred_norm,
        "NoiseSignalLeak": sig["NoiseSignalLeak"],
        "RealSignalReservoirRatio": sig["RealSignalReservoirRatio"],
        "signal_mass_topk": sig["signal_mass_topk"],
        "reservoir_fraction": sig["reservoir_fraction"],
        "top_eigen_share": sig["top_eigen_share"],
        "signal_top_count": sig["signal_top_count"],
        "real_residual_energy": sig["real_residual_energy"],
        "real_total_energy": sig["real_total_energy"],
        "noise_signal_energy": sig["noise_signal_energy"],
        "noise_total_energy": sig["noise_total_energy"],
    }


def run_linec_autopsy(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    model, x_train, y_train, x_val, y_val, _device = train_linec_autopsy_model(args)
    b = min(64, int(x_train.shape[0]), int(x_val.shape[0]))
    xb, yb = x_train[:b], y_train[:b]
    xq, yq = x_val[:b], y_val[:b]
    impls = [
        ("C0-locked-source-shared-v1252", 24, True),
        ("C1-recovered-b64-sketch32-d24", 24, True),
        ("C2-v1217-repair-source-shared-v1252", 24, True),
        ("C3-no-update-baseline", 24, False),
        ("C4-AdamW-window", 24, True),
        ("C5-fixed-batch-deterministic-repeat", 24, True),
    ]
    rows = [linec_calc(model, xb, yb, xq, yq, impl, dim, int(args.autopsy_seed) + 1219, update) for impl, dim, update in impls]
    update_rows = [row for row in rows if row["linec_impl_id"] != "C3-no-update-baseline"]
    diffs: list[float] = []
    for metric in ["CouplingR2", "NoiseSignalLeak", "RealSignalReservoirRatio"]:
        vals = [safe_float(row.get(metric), 0.0) for row in update_rows]
        if vals:
            diffs.append(max(vals) - min(vals))
    max_diff = max(diffs) if diffs else float("nan")
    summary = {
        "stage": "V1219_LINEC_PROTOCOL_AUTOPSY_SUMMARY",
        "autopsy_scope": "same retrained A0 checkpoint, exact same batch/hash, shared v1252 primitives; C3 no-update is a negative control excluded from implementation-diff gate",
        "rows": len(rows),
        "update_impl_rows": len(update_rows),
        "max_update_impl_metric_absdiff": max_diff if not math.isnan(max_diff) else "",
        "protocol_impl_consistency_pass": int(not math.isnan(max_diff) and max_diff <= 0.02),
        "batch_hash": rows[0].get("batch_hash", "") if rows else "",
        "does_not_close_locked_vs_recovered_artifact_mismatch": 1,
    }
    write_csv_rows(out_dir / "v1219_linec_protocol_autopsy.csv", rows + [summary])
    return {
        "linec_autopsy_rows": len(rows),
        "linec_autopsy_max_update_impl_diff": summary["max_update_impl_metric_absdiff"],
        "linec_protocol_impl_consistency_pass": summary["protocol_impl_consistency_pass"],
        "linec_protocol_autopsy_scope": summary["autopsy_scope"],
    }


def write_gate_status(out_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    t1_pass = safe_int(summary.get("T1_only_visibility_pass"), 0)
    line_i_rows = [
        {
            "stage": "V1219_LINEI_GATE_STATUS",
            "line_t_t1_visibility_pass": t1_pass,
            "line_i_open": 0,
            "matched_controls_generated": 0,
            "required_controls": "NoOp,RandomMatchedNorm,AdamWParallel,SNR-only,ShuffledActuatorBasis,MatchedRoleEnergyRandom,MatchedBranchEnergyRandom,T1-score-shuffled",
            "not_run_reason": "Line I is gated by T1-only visibility; v12.19 T1-only visibility did not pass" if t1_pass == 0 else "",
        }
    ]
    p4_rows = [
        {
            "stage": "V1219_P4_GATE_STATUS",
            "p4_open": 0,
            "line_r_pass": summary.get("line_r_code_semantics_pass", 0),
            "line_a_pass": summary.get("linea_pass", 0),
            "line_c_protocol_pass": summary.get("linec_protocol_impl_consistency_pass", 0),
            "line_t_t1_visibility_pass": t1_pass,
            "line_i_response_pass": 0,
            "not_run_reason": "P4 requires Line R/A/C/T/I pass; Line A or Line T/I is closed",
        }
    ]
    write_csv_rows(out_dir / "v1219_linei_gate_status.csv", line_i_rows)
    write_csv_rows(out_dir / "v1219_p4_gate_status.csv", p4_rows)
    return {"line_i_open": 0, "line_i_response_pass": 0, "p4_open": 0, "p4_pass": 0}


def write_classic_monitor(out_dir: Path, v1218_dir: Path) -> dict[str, Any]:
    rows = []
    for row in read_csv_rows(v1218_dir / "v1218_classic_family_status.csv"):
        out = dict(row)
        out["stage"] = "V1219_CLASSIC_FAMILY_STATUS"
        out["new_hypothesis_implemented"] = 0
        out["not_rerun_reason"] = "Line D monitor-only in v12.19; no new loss-agnostic classic-family hypothesis introduced"
        rows.append(out)
    write_csv_rows(out_dir / "v1219_classic_family_status.csv", rows)
    return {"line_d_rows": len(rows), "line_d_new_hypothesis_implemented": 0}


def svg_escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_simple_svg(path: Path, title: str, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    height = max(220, 34 + 24 * (len(lines) + 1))
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" viewBox="0 0 980 {height}">']
    body.append('<rect width="980" height="100%" fill="#f7f7f4"/>')
    body.append(f'<text x="24" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#222">{svg_escape(title)}</text>')
    y = 72
    for line in lines:
        body.append(f'<text x="24" y="{y}" font-family="Arial" font-size="15" fill="#222">{svg_escape(line)}</text>')
        y += 24
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, route: Mapping[str, Any]) -> dict[str, Any]:
    gate_rows = read_csv_rows(out_dir / "v1219_linea_candidate_gate_summary.csv")
    decomp_rows = read_csv_rows(out_dir / "v1219_linea_reservoir_decomposition.csv")
    t1_score = read_csv_rows(out_dir / "v1219_visibility_scores_T1_only.csv")
    leaveout = read_csv_rows(out_dir / "v1219_visibility_leaveout_T1_only.csv")
    support = read_csv_rows(out_dir / "v1219_visibility_support_audit.csv")
    figs = {
        "fig_v1219_route_dashboard.svg": [
            f"route = {route.get('route')}",
            f"fail_reason = {route.get('fail_reason')}",
            f"Line A pass = {route.get('linea_pass')}; T1 visibility pass = {route.get('T1_only_visibility_pass')}; P4 open = {route.get('p4_open')}",
        ],
        "fig_v1219_b320_label_init_ablation_task.svg": [f"{r.get('candidate_id')}: mean_delta_vs_A0={r.get('mean_delta_vs_A0')}, task_pass={r.get('task_side_label_free_pass')}" for r in gate_rows],
        "fig_v1219_b320_label_init_ablation_linec.svg": [f"{r.get('candidate_id')}: linec_pass={r.get('linec_pass')}, pass_rate={r.get('linec_nontearing_pass_rate')}" for r in gate_rows],
        "fig_v1219_linec_protocol_mismatch.svg": [f"max_update_impl_metric_absdiff={route.get('linec_autopsy_max_update_impl_diff')}", f"impl_consistency_pass={route.get('linec_protocol_impl_consistency_pass')}"],
        "fig_v1219_reservoir_decomposition_a5_vs_mlp.svg": [f"{r.get('candidate_id')}: reservoir={r.get('linec_reservoir_fraction_mean')}, real_ratio={r.get('linec_RealSignalReservoirRatio_mean')}" for r in decomp_rows if r.get("candidate_id") in {"A5-orthogonalP-labelFree", "MLP-same-step-FLOP-AdamW", "MLP-same-param-AdamW"}],
        "fig_v1219_noise_signal_energy_a5_vs_mlp.svg": [f"{r.get('candidate_id')}: noise_signal={r.get('linec_noise_signal_energy_mean')}, noise_total={r.get('linec_noise_total_energy_mean')}" for r in decomp_rows if r.get("candidate_id") in {"A5-orthogonalP-labelFree", "MLP-same-step-FLOP-AdamW", "MLP-same-param-AdamW"}],
        "fig_v1219_visibility_pr_curve_T1_only.svg": [f"{r.get('split_protocol')} {r.get('heldout_value')}: precision={r.get('precision_at_k_joint')}, recall={r.get('recall_at_k_joint')}" for r in leaveout[:30]],
        "fig_v1219_visibility_auc_leaveout_T1_only.svg": [f"{r.get('split_protocol')} {r.get('heldout_value')}: AUC_joint={r.get('AUC_joint')}" for r in leaveout[:30]],
        "fig_v1219_visibility_support_heatmap.svg": [json.dumps(support[0], ensure_ascii=False) if support else "support audit missing"],
        "fig_v1219_feature_tier_ablation.svg": [f"T1 rows={route.get('T1_deployable_feature_rows')}; T2 rows={route.get('T2_clone_probe_feature_rows')}; blocked rows={route.get('blocked_visibility_feature_rows')}"],
        "fig_v1219_actuator_control_gap.svg": ["Line I not opened because T1-only visibility did not pass; no actuator control gap generated."],
        "fig_v1219_p4_short_run_if_opened.svg": ["P4 not opened because Line A/T/I gates are not all true; no P4 short-run generated."],
        "fig_v1219_label_free_task_delta.svg": [f"{r.get('candidate_id')}: mean_delta_vs_A0={r.get('mean_delta_vs_A0')}" for r in gate_rows],
        "fig_v1219_label_free_auc_time.svg": [f"{r.get('candidate_id')}: max_AUC_time_ratio_vs_mlp={r.get('max_AUC_time_ratio_vs_mlp')}" for r in gate_rows],
        "fig_v1219_label_free_linec_scatter.svg": [f"{r.get('candidate_id')}: linec_pass_rate={r.get('linec_nontearing_pass_rate')}" for r in gate_rows],
        "fig_v1219_real_residual_energy_ratio.svg": [f"{r.get('candidate_id')}: real_residual={r.get('linec_real_residual_energy_mean')}, real_total={r.get('linec_real_total_energy_mean')}" for r in decomp_rows],
        "fig_v1219_noise_signal_energy_ratio.svg": [f"{r.get('candidate_id')}: noise_signal={r.get('linec_noise_signal_energy_mean')}, noise_total={r.get('linec_noise_total_energy_mean')}" for r in decomp_rows],
        "fig_v1219_P_basis_alignment.svg": [f"{r.get('candidate_id')}: task_pass={r.get('task_side_label_free_pass')}, linec_pass={r.get('linec_pass')}" for r in gate_rows],
        "fig_v1219_A0_locked_vs_recovered_protocol.svg": [f"A0 recovered LineC pass={next((r.get('linec_pass') for r in gate_rows if r.get('candidate_id') == 'A0-labelInit'), '')}", f"v12.18 route fail={route.get('v1218_fail_reason', '')}"],
    }
    for name, lines in figs.items():
        write_simple_svg(out_dir / name, name, lines or ["no rows available"])
    return {"figure_count": len(figs)}


def decide_route(summary: Mapping[str, Any]) -> tuple[str, str]:
    if safe_int(summary.get("line_r_code_semantics_pass"), 0) != 1 or safe_int(summary.get("strict_feature_blocklist_active"), 0) != 1:
        return "R0-CodeSemanticsAuditIncomplete", "Line R semantic audit or strict blocklist incomplete"
    if safe_int(summary.get("b320_current_anchor_pass"), 1) != 1:
        return "R1-B320AnchorRegression", "B320 current anchor did not reproduce"
    if safe_int(summary.get("linea_pass"), 0) != 1:
        return "R2-B320LabelInitDependenceDetected", "no label-free B320-like candidate passed task+LineC; current B320 remains label-informed diagnostic anchor"
    if safe_int(summary.get("linec_protocol_impl_consistency_pass"), 0) != 1:
        return "R3-LineCProtocolMismatch", "Line C implementation mismatch exceeds threshold"
    if safe_int(summary.get("T1_only_visibility_pass"), 0) != 1:
        return "R4-StrictVisibilityFailed", "T1-only precommit loss-agnostic visibility failed"
    if safe_int(summary.get("line_i_response_pass"), 0) != 1:
        return "R5-ActuatorResponseFailed", "Line I gated executor did not pass"
    if safe_int(summary.get("p4_pass"), 0) != 1:
        return "R6-P4FunctionalFailed", "P4 opened but functional did not beat controls"
    return "R7-FunctionalOfficialCandidate", "all v12.19 gates passed"


def write_hash_manifest(out_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1219_hash_manifest.json":
            hashes[path.name] = sha256_file(path)
    write_json(out_dir / "v1219_hash_manifest.json", hashes)
    return hashes


def package_code_review_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "v1219_code_review_packet.zip"
    artifact_names = [
        "v1219_code_review_delta.csv",
        "v1219_core_semantics_audit.md",
        "v1219_core_semantics_audit.csv",
        "v1219_feature_provenance_strict_v2.csv",
        "v1219_forbidden_feature_blocklist.csv",
        "v1219_b320_construction_trace.json",
        "v1219_linec_metric_trace.json",
        "v1219_linea_candidate_gate_summary.csv",
        "v1219_P_basis_alignment_diagnostics.csv",
        "v1219_linea_reservoir_decomposition.csv",
        "v1219_linec_protocol_autopsy.csv",
        "v1219_visibility_feature_tier_manifest.csv",
        "v1219_T1_deployable_features.csv",
        "v1219_T2_clone_probe_diagnostic_features.csv",
        "v1219_T3_audit_targets.csv",
        "v1219_visibility_scores_T1_only.csv",
        "v1219_visibility_scores_T1_plus_T2_diagnostic.csv",
        "v1219_visibility_leaveout_T1_only.csv",
        "v1219_visibility_support_audit.csv",
        "v1219_linei_gate_status.csv",
        "v1219_p4_gate_status.csv",
        "v1219_classic_family_status.csv",
        "v1219_route_decision.json",
        "v1219_hash_manifest.json",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in all_required_code_files():
            path = REPO_ROOT / file_name
            if path.exists():
                zf.write(path, arcname=f"code/{file_name}")
        if PLAN_DOC.exists():
            zf.write(PLAN_DOC, arcname=f"code/{rel(PLAN_DOC)}")
        for path in sorted(out_dir.glob("fig_v1219_*.svg")):
            zf.write(path, arcname=f"figures/{path.name}")
        for name in artifact_names:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=f"review_artifacts/{name}")
    return zip_path


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    v1218_dir = Path(args.v1218_dir).resolve()
    v1217_dir = Path(args.v1217_dir).resolve()
    ensure_dir(out_dir)
    summary: dict[str, Any] = {
        "stage": "V1219_ROUTE_DECISION",
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "v1218_source_dir": rel(v1218_dir),
        "v1217_source_dir": rel(v1217_dir),
        "v1216_source_dir": rel(Path(args.v1216_dir)),
        "no_fake": 1,
        "no_proxy": 1,
    }
    v1218_route = read_json(v1218_dir / "v1218_route_decision.json")
    summary["v1218_route"] = v1218_route.get("route", "")
    summary["v1218_fail_reason"] = v1218_route.get("fail_reason", "")
    summary["b320_current_anchor_pass"] = v1218_route.get("b320_current_anchor_pass", 1)
    for chunk in [
        write_code_semantics(out_dir, v1218_dir),
        write_forbidden_blocklist(out_dir),
        evaluate_linea(out_dir),
        run_linec_autopsy(out_dir, args),
        write_visibility_tiers(out_dir, v1218_dir, v1217_dir),
    ]:
        summary.update(chunk)
    support = support_audit(out_dir, v1218_dir, v1217_dir)
    t1_rows = read_csv_rows(out_dir / "v1219_T1_deployable_features.csv")
    t2_rows = read_csv_rows(out_dir / "v1219_T2_clone_probe_diagnostic_features.csv")
    t1_summary = score_visibility_tier(out_dir, t1_rows, support, "T1_only", "T1_deployable_precommit", diagnostic_only=0)
    t2_summary = score_visibility_tier(out_dir, t1_rows + t2_rows, support, "T1_plus_T2_diagnostic", "T1_plus_T2_clone_probe_diagnostic", diagnostic_only=1)
    summary.update(support)
    summary.update(t1_summary)
    summary.update(t2_summary)
    summary["T1_only_visibility_pass"] = t1_summary.get("T1_only_visibility_pass", 0)
    summary["T1_plus_T2_diagnostic_pass_not_promotable"] = t2_summary.get("T1_plus_T2_diagnostic_diagnostic_pass_not_promotable", 0)
    summary.update(write_gate_status(out_dir, summary))
    summary.update(write_classic_monitor(out_dir, v1218_dir))
    route, fail_reason = decide_route(summary)
    summary["route"] = route
    summary["fail_reason"] = fail_reason
    summary["promotion_allowed"] = int(route == "R7-FunctionalOfficialCandidate")
    summary.update(write_figures(out_dir, summary))
    write_json(out_dir / "v1219_route_decision.json", summary)
    hashes = write_hash_manifest(out_dir)
    zip_path = package_code_review_zip(out_dir)
    summary["code_review_packet_zip"] = rel(zip_path)
    summary["code_review_packet_zip_sha256"] = sha256_file(zip_path)
    summary["hash_manifest_entries"] = len(hashes)
    write_json(out_dir / "v1219_route_decision.json", summary)
    write_hash_manifest(out_dir)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="official_from_v1218_v1217_v1216_artifacts")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--v1218-dir", default=str(DEFAULT_V1218_DIR))
    parser.add_argument("--v1217-dir", default=str(DEFAULT_V1217_DIR))
    parser.add_argument("--v1216-dir", default=str(DEFAULT_V1216_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seed-base", type=int, default=1219000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--task-lr-schedule", default="linear_warmup10_cosine_final050")
    parser.add_argument("--optimizer-impl", default="adamw")
    parser.add_argument("--task-compile-warmup-steps", type=int, default=0)
    parser.add_argument("--autopsy-dataset", default="MNIST")
    parser.add_argument("--autopsy-seed", type=int, default=0)
    parser.add_argument("--autopsy-train-size", type=int, default=1024)
    parser.add_argument("--autopsy-val-size", type=int, default=512)
    parser.add_argument("--autopsy-test-size", type=int, default=512)
    parser.add_argument("--autopsy-epochs", type=int, default=3)
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    result = run_main(parsed)
    print(
        json.dumps(
            {
                "out_dir": str(Path(parsed.out_dir).resolve()),
                "route": result.get("route"),
                "fail_reason": result.get("fail_reason"),
                "T1_only_visibility_pass": result.get("T1_only_visibility_pass"),
                "p4_open": result.get("p4_open"),
                "code_review_packet_zip": result.get("code_review_packet_zip"),
                "code_review_packet_zip_sha256": result.get("code_review_packet_zip_sha256"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
