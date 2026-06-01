#!/usr/bin/env python3
"""v12.18 B320 code-audit and strict loss-agnostic functional diagnostic.

This runner is intentionally conservative.  It builds the transitive code
review packet requested by the v12.18 plan, audits B320 label-informed init
semantics from actual source code and prior artifacts, separates audit-only
Line C targets from deployable observables, and keeps P3/P4 closed unless the
strict v12.18 gates are genuinely satisfied.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md"
RUNNER_PATH = REPO_ROOT / "experiments" / "run_v1218_b320_codeaudit_lossagnostic_functional.py"
DEFAULT_V1217_DIR = (
    REPO_ROOT
    / "results"
    / "v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry"
    / "official_from_v1216_artifacts"
)
DEFAULT_V1216_REPAIR_DIR = (
    REPO_ROOT
    / "results"
    / "v12_16_b320locked_explicit_signal_reservoir_functional"
    / "repair_sketchdim24_rank5_b64_w3_5_10"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "results"
    / "v12_18_b320_codeaudit_lossagnostic_functional"
    / "official_from_v1217_v1216_artifacts"
)
B320_ID = (
    "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-"
    "signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-"
    "absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-"
    "hingeamp025-temp075"
)

BASE_REQUIRED_CODE_FILES = [
    "experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py",
    "experiments/run_v1218_b320_label_free_ablation.py",
    "experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py",
    "experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py",
    "experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py",
    "experiments/run_v1215_continuation_actuator_budget_repair.py",
    "experiments/run_v1215_continuation_p3_from_fused_actuator.py",
    "experiments/run_v1283_b109_classic_family_functional_geometry.py",
    "experiments/run_v1252_efficiency_functional_manifold.py",
    "experiments/run_v124_multibasis_functional_dual.py",
    "experiments/run_v120_good_geometry_battery.py",
    "dgkan/models/fc_purekan_primitives.py",
    "dgkan/kernels/fused_hinge_quadratic.py",
    "dgkan/optim/manual_adamw.py",
    "dgkan/profiling/timing.py",
]

REQUIRED_ARTIFACTS = [
    "v1218_route_decision.json",
    "v1218_transitive_dependency_manifest.csv",
    "v1218_core_code_review_manifest.csv",
    "v1218_symbol_line_map.json",
    "v1218_feature_provenance_audit.csv",
    "v1218_b320_label_init_audit.csv",
    "v1218_b320_label_free_smoke_ablation.csv",
    "v1218_b320_label_free_smoke_summary.csv",
    "v1218_b320_label_free_smoke_route.json",
    "v1218_b320_label_free_anchor_budget_ablation.csv",
    "v1218_b320_label_free_anchor_budget_summary.csv",
    "v1218_b320_label_free_anchor_budget_route.json",
    "v1218_b320_anchor_budget_seedbase1211000_probe_ablation.csv",
    "v1218_b320_anchor_budget_seedbase1211000_probe_summary.csv",
    "v1218_b320_anchor_budget_seedbase1211000_probe_route.json",
    "v1218_b320_anchor_budget_seedbase1212000_probe_ablation.csv",
    "v1218_b320_anchor_budget_seedbase1212000_probe_summary.csv",
    "v1218_b320_anchor_budget_seedbase1212000_probe_route.json",
    "v1218_b320_anchor_budget_seedbase1213000_probe_ablation.csv",
    "v1218_b320_anchor_budget_seedbase1213000_probe_summary.csv",
    "v1218_b320_anchor_budget_seedbase1213000_probe_route.json",
    "v1218_b320_anchor_budget_protocol_audit.csv",
    "v1218_manual_review_packet.md",
    "v1218_review_blocker_table.csv",
    "v1218_linec_audit_metrics.csv",
    "v1218_linec_null_distribution.csv",
    "v1218_linec_threshold_sensitivity.csv",
    "v1218_linec_hard_support.csv",
    "v1218_audit_metric_provenance.csv",
    "v1218_strict_visibility_features.csv",
    "v1218_strict_visibility_scores.csv",
    "v1218_strict_visibility_leaveout.csv",
    "v1218_strict_visibility_family_ablation.csv",
    "v1218_strict_visibility_composite_repair.csv",
    "v1218_strict_visibility_context_rank_repair.csv",
    "v1218_strict_visibility_feature_engineering_audit.csv",
    "v1218_strict_visibility_repair_attempts.csv",
    "v1218_support_concentration.csv",
    "v1218_line_t_support_expansion_audit.csv",
    "v1218_hard_gate_margin_repair.csv",
    "v1218_hard_gate_margin_repair_summary.csv",
    "v1218_hard_gate_bootstrap_repair.csv",
    "v1218_hard_gate_bootstrap_repair_summary.csv",
    "v1218_p3_matched_controls_status.csv",
    "v1218_p4_gate_status.csv",
    "v1218_classic_family_status.csv",
    "v1218_hash_manifest.json",
    "v1218_code_review_packet.zip",
]


@dataclass(frozen=True)
class CodeSymbol:
    path: str
    symbol: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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


def all_required_code_files() -> list[str]:
    kernel_files = sorted(rel(path) for path in (REPO_ROOT / "dgkan" / "kernels").glob("*.py"))
    return sorted(dict.fromkeys(BASE_REQUIRED_CODE_FILES + kernel_files))


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
    pattern = re.escape(symbol)
    for idx, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line):
            return idx, idx, line.strip()
    return None, None, ""


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if not math.isnan(float(s))]
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return float(wins / (len(pos) * len(neg)))


def precision_recall_at_k(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, float, int, int]:
    positives = sum(int(y) for y in labels)
    if positives <= 0:
        return float("nan"), float("nan"), 0, 0
    k = max(1, positives)
    order = sorted(range(len(scores)), key=lambda i: (math.isnan(float(scores[i])), -float(scores[i]) if not math.isnan(float(scores[i])) else 0.0))
    top = order[:k]
    hits = sum(int(labels[i]) for i in top)
    return float(hits / k), float(hits / positives), int(hits), int(positives)


def rank01(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray([float(x) if not math.isnan(float(x)) else 0.0 for x in scores], dtype=float)
    if arr.size <= 1:
        return np.ones_like(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(arr.size, dtype=float)
    return ranks / max(1.0, float(arr.size - 1))


def finite_min(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if not math.isnan(float(v))]
    return min(valid) if valid else float("nan")


def finite_max(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if not math.isnan(float(v))]
    return max(valid) if valid else float("nan")


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    if train_x.size == 0 or test_x.size == 0:
        return np.zeros((test_x.shape[0],), dtype=float)
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1.0e-8] = 1.0
    x = (train_x - mu) / sigma
    xt = (test_x - mu) / sigma
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    xt_aug = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    reg = np.eye(x_aug.shape[1]) * 1.0e-3
    reg[0, 0] = 0.0
    try:
        w = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ train_y)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ train_y
    return xt_aug @ w


def centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, mode: str) -> np.ndarray:
    if train_x.size == 0 or test_x.size == 0:
        return np.zeros((test_x.shape[0],), dtype=float)
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1.0e-8] = 1.0
    x = (train_x - mu) / sigma
    xt = (test_x - mu) / sigma
    pos = x[np.asarray(train_y, dtype=int) == 1]
    neg = x[np.asarray(train_y, dtype=int) == 0]
    if pos.size == 0:
        return np.zeros((test_x.shape[0],), dtype=float)
    pos_center = pos.mean(axis=0)
    neg_center = neg.mean(axis=0) if neg.size else np.zeros_like(pos_center)
    if mode == "positive_centroid_distance":
        return -np.sum((xt - pos_center) ** 2, axis=1)
    if mode == "contrast_centroid_distance":
        return np.sum((xt - neg_center) ** 2, axis=1) - np.sum((xt - pos_center) ** 2, axis=1)
    if mode == "contrast_centroid_dot":
        return xt @ (pos_center - neg_center)
    return xt @ pos_center


def run_transitive_dependency_manifest(out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for file_name in all_required_code_files():
        path = REPO_ROOT / file_name
        rows.append(
            {
                "stage": "V1218_TRANSITIVE_DEPENDENCY_MANIFEST",
                "path": file_name,
                "exists": int(path.exists()),
                "sha256": sha256_file(path) if path.exists() else "",
                "required_by_plan": 1,
                "included_in_code_packet": 1,
                "no_fake": 1,
                "no_proxy": 1,
                "cpu_offload_used": 0,
            }
        )
    write_csv_rows(out_dir / "v1218_transitive_dependency_manifest.csv", rows)
    return {
        "transitive_dependency_rows": len(rows),
        "transitive_missing_count": sum(1 for row in rows if safe_int(row.get("exists"), 0) == 0),
        "all_required_files_present": int(all(safe_int(row.get("exists"), 0) == 1 for row in rows)),
    }


def cr_specs() -> list[dict[str, Any]]:
    return [
        {
            "review_id": "CR0",
            "component": "runner / route decision / artifact writing",
            "symbols": [CodeSymbol(rel(RUNNER_PATH), "run_main"), CodeSymbol(rel(RUNNER_PATH), "decide_route"), CodeSymbol(rel(RUNNER_PATH), "write_hash_manifest")],
            "artifacts": "v1218_route_decision.json,v1218_hash_manifest.json,v1218_code_review_packet.zip",
            "risk": "route or artifact completion could be hand-written instead of computed",
        },
        {
            "review_id": "CR1",
            "component": "B320 candidate registry and v124._make_model actual construction",
            "symbols": [
                CodeSymbol("experiments/run_v1283_b109_classic_family_functional_geometry.py", "_make_model"),
                CodeSymbol("experiments/run_v124_multibasis_functional_dual.py", "_make_model"),
                CodeSymbol("dgkan/models/fc_purekan_primitives.py", B320_ID),
            ],
            "artifacts": "v1218_transitive_dependency_manifest.csv,v1218_b320_label_init_audit.csv",
            "risk": "B320 construction could be misattributed or missing y_stats wiring",
        },
        {
            "review_id": "CR2",
            "component": "SimpleFastTaskGeometryKAN label-informed init and buffers",
            "symbols": [
                CodeSymbol("dgkan/models/fc_purekan_primitives.py", "SimpleFastTaskGeometryKAN"),
                CodeSymbol("dgkan/models/fc_purekan_primitives.py", "trainprobe_signal_init_uses_labels"),
                CodeSymbol("dgkan/models/fc_purekan_primitives.py", "trainprobe_signal_init_applied"),
            ],
            "artifacts": "v1218_b320_label_init_audit.csv",
            "risk": "label-informed base initialization could be hidden",
        },
        {
            "review_id": "CR3",
            "component": "FHQ fused forward/backward/update kernels",
            "symbols": [
                CodeSymbol("dgkan/kernels/fused_hinge_quadratic.py", "forward_workspace"),
                CodeSymbol("dgkan/kernels/fused_hinge_quadratic.py", "backward_learnablep_workspace_fused_quadproj_adamw_from_grad_logits"),
                CodeSymbol("dgkan/kernels/fused_hinge_quadratic.py", "_fhq_proj_grad_adamw_kernel"),
            ],
            "artifacts": "v1218_core_code_review_manifest.csv",
            "risk": "fused path could silently fall back to a different update semantics",
        },
        {
            "review_id": "CR4",
            "component": "manual AdamW and optimizer-state semantics",
            "symbols": [
                CodeSymbol("dgkan/optim/manual_adamw.py", "ManualAdamWConfig"),
                CodeSymbol("dgkan/optim/manual_adamw.py", "AdamWState"),
                CodeSymbol("dgkan/optim/manual_adamw.py", "adamw_update_"),
            ],
            "artifacts": "v1218_core_code_review_manifest.csv",
            "risk": "manual optimizer state could differ from claimed AdamW semantics",
        },
        {
            "review_id": "CR5",
            "component": "Line C audit metric CE/label provenance",
            "symbols": [
                CodeSymbol("experiments/run_v1252_efficiency_functional_manifold.py", "_sample_grad_sketch"),
                CodeSymbol("experiments/run_v1252_efficiency_functional_manifold.py", "_signal_reservoir_metrics"),
            ],
            "artifacts": "v1218_audit_metric_provenance.csv,v1218_linec_audit_metrics.csv",
            "risk": "CE/label-derived audit metrics could be mistaken for deployable observables",
        },
        {
            "review_id": "CR6",
            "component": "deployable loss-agnostic observable construction",
            "symbols": [CodeSymbol(rel(RUNNER_PATH), "classify_feature_provenance"), CodeSymbol(rel(RUNNER_PATH), "run_strict_visibility_v2")],
            "artifacts": "v1218_feature_provenance_audit.csv,v1218_strict_visibility_features.csv",
            "risk": "forbidden feature families could enter strict target visibility",
        },
        {
            "review_id": "CR7",
            "component": "hard release labels audit-only isolation",
            "symbols": [
                CodeSymbol("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_release_labels"),
                CodeSymbol(rel(RUNNER_PATH), "run_linec_audit"),
            ],
            "artifacts": "v1218_linec_hard_support.csv,v1218_linec_audit_metrics.csv",
            "risk": "hard-release labels could leak into candidate direction",
        },
        {
            "review_id": "CR8",
            "component": "P2 visibility scorer and split protocol",
            "symbols": [
                CodeSymbol("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p2_target_visibility"),
                CodeSymbol(rel(RUNNER_PATH), "split_visibility_rows"),
            ],
            "artifacts": "v1218_strict_visibility_scores.csv,v1218_strict_visibility_leaveout.csv",
            "risk": "split leakage could create a false observability pass",
        },
        {
            "review_id": "CR9",
            "component": "P2 feature provenance checker",
            "symbols": [CodeSymbol(rel(RUNNER_PATH), "classify_feature_provenance"), CodeSymbol(rel(RUNNER_PATH), "run_feature_provenance_audit")],
            "artifacts": "v1218_feature_provenance_audit.csv",
            "risk": "numeric hashes or CE-derived fields could be promoted as physical features",
        },
        {
            "review_id": "CR10",
            "component": "P3 actuator generation and response matrix",
            "symbols": [
                CodeSymbol("experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py", "actuator_basis_deltas"),
                CodeSymbol("experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py", "annotate_p3_matrix"),
            ],
            "artifacts": "v1218_p3_matched_controls_status.csv",
            "risk": "actuator movement could be overstated without response semantics",
        },
        {
            "review_id": "CR11",
            "component": "P3 matched controls",
            "symbols": [CodeSymbol("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p3_actuator_response"), CodeSymbol(rel(RUNNER_PATH), "run_p3_p4_gates")],
            "artifacts": "v1218_p3_matched_controls_status.csv",
            "risk": "future P3 pass could lack matched NoOp/Random/AdamW/SNR controls",
        },
        {
            "review_id": "CR12",
            "component": "P4 short-run event integration and overhead accounting",
            "symbols": [CodeSymbol("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p4_short_run"), CodeSymbol(rel(RUNNER_PATH), "run_p3_p4_gates")],
            "artifacts": "v1218_p4_gate_status.csv",
            "risk": "P4 could be opened before strict Line T/I gates",
        },
        {
            "review_id": "CR13",
            "component": "timing/memory profiler and hook separation",
            "symbols": [CodeSymbol("dgkan/profiling/timing.py", "TIMING_PROTOCOLS"), CodeSymbol("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p0_anchor_monitor")],
            "artifacts": "v1218_b320_label_init_audit.csv",
            "risk": "diagnostic hook time could be mixed into architecture timing",
        },
        {
            "review_id": "CR14",
            "component": "provenance/hash/no fake/no proxy/no CPU offload audit",
            "symbols": [CodeSymbol(rel(RUNNER_PATH), "sha256_file"), CodeSymbol(rel(RUNNER_PATH), "write_hash_manifest"), CodeSymbol(rel(RUNNER_PATH), "package_code_review_zip")],
            "artifacts": "v1218_hash_manifest.json,v1218_transitive_dependency_manifest.csv",
            "risk": "source reuse could be untraceable",
        },
        {
            "review_id": "CR15",
            "component": "dataset-agnostic discipline / no dataset-name branch",
            "symbols": [CodeSymbol(rel(RUNNER_PATH), "run_strict_visibility_v2"), CodeSymbol(rel(RUNNER_PATH), "decide_route")],
            "artifacts": "v1218_strict_visibility_leaveout.csv,v1218_route_decision.json",
            "risk": "dataset names could become controller branches rather than split keys",
        },
    ]


def build_core_code_review(out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    symbol_map: dict[str, Any] = {}
    blockers: list[dict[str, Any]] = []
    for spec in cr_specs():
        symbol_rows = []
        unknown = 0
        snippets: list[str] = []
        actual_paths: list[str] = []
        line_starts: list[str] = []
        line_ends: list[str] = []
        for sym in spec["symbols"]:
            path = REPO_ROOT / sym.path
            start, end, snippet = line_range_for_symbol(path, sym.symbol)
            missing = int(start is None)
            unknown += missing
            actual_paths.append(sym.path)
            line_starts.append("" if start is None else str(start))
            line_ends.append("" if end is None else str(end))
            snippets.append(snippet.replace("\n", " ")[:240])
            symbol_rows.append({"file": sym.path, "symbol": sym.symbol, "line_start": start, "line_end": end, "missing": missing})
        row = {
            "stage": "V1218_CORE_CODE_REVIEW_MANIFEST",
            "review_id": spec["review_id"],
            "component": spec["component"],
            "actual_file_path": "|".join(dict.fromkeys(actual_paths)),
            "actual_line_start": "|".join(line_starts),
            "actual_line_end": "|".join(line_ends),
            "main_symbols": " | ".join(sym.symbol for sym in spec["symbols"]),
            "artifact_fields_written": spec["artifacts"],
            "unknown_or_not_inspected": int(unknown > 0),
            "requires_manual_review": 1,
            "manual_review_status": "packet_generated_pending_human_review",
            "uses_label": 1 if spec["review_id"] in {"CR2", "CR5", "CR7"} else 0,
            "uses_ce_vector": 1 if spec["review_id"] == "CR5" else 0,
            "uses_dataset_name_branch": 0,
            "promotion_allowed": 0,
            "blocker_if_wrong": spec["risk"],
            "codex_summary": f"{spec['component']} inspected via listed symbols; v12.18 diagnostics do not promote without strict gates.",
            "line_snippets": " || ".join(snippets),
        }
        rows.append(row)
        symbol_map[spec["review_id"]] = {
            "component": spec["component"],
            "symbols": symbol_rows,
            "artifacts": spec["artifacts"].split(","),
            "risk": spec["risk"],
        }
        if unknown:
            blockers.append(
                {
                    "stage": "V1218_REVIEW_BLOCKER",
                    "review_id": spec["review_id"],
                    "blocker_type": "unknown_or_not_inspected",
                    "blocks_promotion": 1,
                    "details": spec["risk"],
                }
            )
        blockers.append(
            {
                "stage": "V1218_REVIEW_BLOCKER",
                "review_id": spec["review_id"],
                "blocker_type": "manual_review_pending_for_promotion",
                "blocks_promotion": 1,
                "details": "Code review packet generated; human review still required before any promotion.",
            }
        )
    write_csv_rows(out_dir / "v1218_core_code_review_manifest.csv", rows)
    write_json(out_dir / "v1218_symbol_line_map.json", symbol_map)
    write_csv_rows(out_dir / "v1218_review_blocker_table.csv", blockers)
    write_manual_review_packet(out_dir, rows)
    return {
        "core_code_review_rows": len(rows),
        "cr0_cr15_unknown_count": sum(safe_int(row.get("unknown_or_not_inspected"), 0) for row in rows),
        "core_code_review_pass": int(all(safe_int(row.get("unknown_or_not_inspected"), 0) == 0 for row in rows)),
        "manual_review_pending_count": sum(1 for row in blockers if row.get("blocker_type") == "manual_review_pending_for_promotion"),
        "review_blocker_rows": len(blockers),
    }


def write_manual_review_packet(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# v12.18 Manual Review Packet",
        "",
        f"Generated at: {now_iso()}",
        "",
        "This packet maps CR0-CR15 to actual repository files and symbols. It is a code-audit packet, not a promotion approval.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('review_id')} {row.get('component')}",
                "",
                f"- Files: `{row.get('actual_file_path')}`",
                f"- Symbols: `{row.get('main_symbols')}`",
                f"- Lines: `{row.get('actual_line_start')}` to `{row.get('actual_line_end')}`",
                f"- Artifacts controlled: `{row.get('artifact_fields_written')}`",
                f"- Unknown: `{row.get('unknown_or_not_inspected')}`",
                f"- Label/CE risk: uses_label={row.get('uses_label')}, uses_ce_vector={row.get('uses_ce_vector')}",
                f"- Risk if wrong: {row.get('blocker_if_wrong')}",
                f"- Snippets: `{row.get('line_snippets')}`",
                "",
            ]
        )
    (out_dir / "v1218_manual_review_packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_smoke_metrics(row: dict[str, Any], smoke_row: Mapping[str, Any], out_dir: Path) -> None:
    row.update(
        {
            "small_budget_training_result_available": 1,
            "smoke_not_official": safe_int(smoke_row.get("smoke_not_official"), 1),
            "smoke_rows": smoke_row.get("rows", ""),
            "smoke_datasets": smoke_row.get("datasets", ""),
            "smoke_seeds": smoke_row.get("seeds", ""),
            "smoke_mean_delta_vs_mlp": smoke_row.get("mean_delta_vs_mlp", ""),
            "smoke_worst_delta_vs_mlp": smoke_row.get("worst_delta_vs_mlp", ""),
            "smoke_near_pass_rate_vs_mlp": smoke_row.get("near_pass_rate_vs_mlp", ""),
            "smoke_max_AUC_step_ratio_vs_mlp": smoke_row.get("max_AUC_step_ratio_vs_mlp", ""),
            "smoke_max_AUC_time_ratio_vs_mlp": smoke_row.get("max_AUC_time_ratio_vs_mlp", ""),
            "smoke_max_ECE_delta_vs_mlp": smoke_row.get("max_ECE_delta_vs_mlp", ""),
            "smoke_mean_delta_vs_A0_labelInit": smoke_row.get("mean_delta_vs_A0_labelInit", ""),
            "smoke_worst_delta_vs_A0_labelInit": smoke_row.get("worst_delta_vs_A0_labelInit", ""),
            "smoke_strict_label_free_rows": smoke_row.get("strict_label_free_rows", ""),
            "smoke_source_artifact": rel(out_dir / "v1218_b320_label_free_smoke_summary.csv"),
        }
    )


def attach_anchor_budget_metrics(row: dict[str, Any], budget_row: Mapping[str, Any], out_dir: Path) -> None:
    row.update(
        {
            "anchor_budget_training_result_available": 1,
            "anchor_budget_official_training_result_available": safe_int(budget_row.get("official_training_result_available"), 0),
            "anchor_budget_smoke_not_official": safe_int(budget_row.get("smoke_not_official"), 0),
            "anchor_budget_rows": budget_row.get("rows", ""),
            "anchor_budget_datasets": budget_row.get("datasets", ""),
            "anchor_budget_seeds": budget_row.get("seeds", ""),
            "anchor_budget_mean_delta_vs_mlp": budget_row.get("mean_delta_vs_mlp", ""),
            "anchor_budget_worst_delta_vs_mlp": budget_row.get("worst_delta_vs_mlp", ""),
            "anchor_budget_near_pass_rate_vs_mlp": budget_row.get("near_pass_rate_vs_mlp", ""),
            "anchor_budget_max_AUC_step_ratio_vs_mlp": budget_row.get("max_AUC_step_ratio_vs_mlp", ""),
            "anchor_budget_max_AUC_time_ratio_vs_mlp": budget_row.get("max_AUC_time_ratio_vs_mlp", ""),
            "anchor_budget_max_ECE_delta_vs_mlp": budget_row.get("max_ECE_delta_vs_mlp", ""),
            "anchor_budget_mean_delta_vs_A0_labelInit": budget_row.get("mean_delta_vs_A0_labelInit", ""),
            "anchor_budget_worst_delta_vs_A0_labelInit": budget_row.get("worst_delta_vs_A0_labelInit", ""),
            "anchor_budget_strict_label_free_rows": budget_row.get("strict_label_free_rows", ""),
            "anchor_budget_linec_nontearing_rows": budget_row.get("linec_nontearing_rows", ""),
            "anchor_budget_linec_nontearing_pass_rate": budget_row.get("linec_nontearing_pass_rate", ""),
            "anchor_budget_linec_nontearing_all_pass": budget_row.get("linec_nontearing_all_pass", ""),
            "anchor_budget_protocol_note": budget_row.get("protocol_note", ""),
            "anchor_budget_source_artifact": rel(out_dir / "v1218_b320_label_free_anchor_budget_summary.csv"),
        }
    )


def run_b320_label_init_audit(v1217_dir: Path, out_dir: Path) -> dict[str, Any]:
    route17 = read_json(v1217_dir / "v1217_route_decision.json")
    current_pass = int(safe_int(route17.get("p0_pass"), 1) == 1)
    b320_variant = B320_ID.lower()
    smoke_summary = read_csv_rows(out_dir / "v1218_b320_label_free_smoke_summary.csv")
    smoke_by_candidate = {str(row.get("candidate_id", "")): row for row in smoke_summary}
    anchor_budget_summary = read_csv_rows(out_dir / "v1218_b320_label_free_anchor_budget_summary.csv")
    anchor_budget_by_candidate = {str(row.get("candidate_id", "")): row for row in anchor_budget_summary}
    rows: list[dict[str, Any]] = [
        {
            "stage": "V1218_B320_LABEL_INIT_AUDIT",
            "candidate_id": B320_ID,
            "ablation_id": "A0 B320-current-labelInit",
            "uses_y_for_stats": 1,
            "trainprobe_signal_init_uses_labels": int("trainprobe" in b320_variant),
            "trainprobe_signal_init_applied_code_expected": 1,
            "trainprobeP_enabled": int("trainprobep" in b320_variant),
            "trainprobeDirect_enabled": int("direct050" in b320_variant),
            "signalBroad": "signalBroad035",
            "signalBlock": "signalBlock015",
            "step_ratio_q90": route17.get("step_ratio_q90_max", route17.get("p0_step_ratio_q90_max", "")),
            "memory_ratio_q90": route17.get("memory_ratio_q90_max", route17.get("p0_memory_ratio_q90_max", "")),
            "mean_delta_vs_mlp": route17.get("mean_delta_vs_mlp_min", ""),
            "worst_delta_vs_mlp": route17.get("worst_delta_vs_mlp_min", ""),
            "near_pass_rate": route17.get("near_pass_rate_min", ""),
            "AUC_step_ratio": route17.get("AUC_step_ratio_max", ""),
            "AUC_time_ratio": route17.get("AUC_time_ratio_max", ""),
            "ECE_delta": route17.get("ECE_delta_max", ""),
            "LineC_nontearing_pass": route17.get("LineC_nontearing_pass_min", ""),
            "current_anchor_pass": current_pass,
            "official_training_result_available": 1,
            "source_artifact": rel(v1217_dir / "v1217_route_decision.json"),
            "B320_claim_scope": "label-informed supervised initialization anchor",
            "external_ready_base_claim": 0,
            "no_fake": 1,
            "no_proxy": 1,
            "cpu_offload_used": 0,
        }
    ]
    if "A0-labelInit" in smoke_by_candidate:
        attach_smoke_metrics(rows[0], smoke_by_candidate["A0-labelInit"], out_dir)
    if "A0-labelInit" in anchor_budget_by_candidate:
        attach_anchor_budget_metrics(rows[0], anchor_budget_by_candidate["A0-labelInit"], out_dir)
    current_mean = safe_float(route17.get("mean_delta_vs_mlp_min", ""), safe_float(route17.get("p0_mean_delta_vs_mlp_min", ""), 0.0))
    current_worst = safe_float(route17.get("worst_delta_vs_mlp_min", ""), safe_float(route17.get("p0_worst_delta_vs_mlp_min", ""), 0.0))
    current_auc_time = safe_float(route17.get("AUC_time_ratio_max", ""), safe_float(route17.get("p0_AUC_time_ratio_max", ""), 1.0))
    a0_anchor_budget = anchor_budget_by_candidate.get("A0-labelInit", {})
    a0_budget_mean = safe_float(a0_anchor_budget.get("mean_delta_vs_mlp"), -1.0e9)
    a0_budget_worst = safe_float(a0_anchor_budget.get("worst_delta_vs_mlp"), -1.0e9)
    a0_budget_auc_time = safe_float(a0_anchor_budget.get("max_AUC_time_ratio_vs_mlp"), 1.0e9)
    a0_anchor_budget_gate_pass = int(
        bool(a0_anchor_budget)
        and a0_budget_mean >= 0.0
        and a0_budget_worst >= -0.003
        and a0_budget_auc_time <= 1.0
    )
    a0_anchor_budget_task_reproduces_locked_anchor = int(
        bool(a0_anchor_budget)
        and a0_anchor_budget_gate_pass == 1
        and a0_budget_mean >= current_mean - 0.010
        and a0_budget_worst >= current_worst - 0.003
        and a0_budget_auc_time <= max(1.0, current_auc_time + 0.050)
    )
    a0_anchor_budget_linec_available = int(bool(a0_anchor_budget) and safe_int(a0_anchor_budget.get("linec_nontearing_rows"), 0) > 0)
    a0_anchor_budget_linec_reproduces_locked_anchor = int(
        a0_anchor_budget_linec_available
        and safe_int(a0_anchor_budget.get("linec_nontearing_all_pass"), 0) == 1
    )
    a0_anchor_budget_reproduces_locked_anchor = int(
        a0_anchor_budget_task_reproduces_locked_anchor
        and a0_anchor_budget_linec_reproduces_locked_anchor
    )
    protocol_audit = [
        {
            "stage": "V1218_B320_ANCHOR_BUDGET_PROTOCOL_AUDIT",
            "source": "v1218_b320_label_free_anchor_budget_summary.csv",
            "candidate_id": "A0-labelInit",
            "current_locked_anchor_mean_delta_vs_mlp": current_mean,
            "current_locked_anchor_worst_delta_vs_mlp": current_worst,
            "current_locked_anchor_AUC_time_ratio": current_auc_time,
            "anchor_budget_A0_mean_delta_vs_mlp": a0_budget_mean if a0_anchor_budget else "",
            "anchor_budget_A0_worst_delta_vs_mlp": a0_budget_worst if a0_anchor_budget else "",
            "anchor_budget_A0_max_AUC_time_ratio_vs_mlp": a0_budget_auc_time if a0_anchor_budget else "",
            "anchor_budget_A0_linec_nontearing_rows": a0_anchor_budget.get("linec_nontearing_rows", "") if a0_anchor_budget else "",
            "anchor_budget_A0_linec_nontearing_pass_rate": a0_anchor_budget.get("linec_nontearing_pass_rate", "") if a0_anchor_budget else "",
            "anchor_budget_A0_linec_nontearing_all_pass": a0_anchor_budget.get("linec_nontearing_all_pass", "") if a0_anchor_budget else "",
            "a0_anchor_budget_gate_pass": a0_anchor_budget_gate_pass,
            "a0_anchor_budget_task_reproduces_locked_anchor": a0_anchor_budget_task_reproduces_locked_anchor,
            "a0_anchor_budget_linec_available": a0_anchor_budget_linec_available,
            "a0_anchor_budget_linec_reproduces_locked_anchor": a0_anchor_budget_linec_reproduces_locked_anchor,
            "a0_anchor_budget_reproduces_locked_anchor": a0_anchor_budget_reproduces_locked_anchor,
            "official_budget_claim_allowed": a0_anchor_budget_reproduces_locked_anchor,
            "blocker": "" if a0_anchor_budget_reproduces_locked_anchor else "recovered-anchor-budget A0 does not reproduce locked B320 task+LineC gates; do not treat label-free ablation as official replacement",
            "no_fake": 1,
            "no_proxy": 1,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "v1218_b320_anchor_budget_protocol_audit.csv", protocol_audit)
    for ablation_id, smoke_id, allowed in [
        ("A1 B320-current-noYForStats", "A1-noYForStats", "construct B320 with y_stats=None; same train budget required"),
        ("A2 B320-current-randomP-labelFree", "A2-randomP-labelFree", "random P label-free ablation; same train budget required"),
        ("A3 B320-current-PCA-P-labelFree", "A3-PCA-P-labelFree", "PCA-P label-free ablation; prior B226-B229 are related but not comparable official v12.18 rows"),
        ("A4 B320-current-lowfreqP-labelFree", "A4-lowfreqP-labelFree", "lowfreqP label-free ablation; same train budget required"),
        ("A5 B320-current-orthogonalP-labelFree", "A5-orthogonalP-labelFree", "orthogonalP label-free ablation; same train budget required"),
        ("A6 B320-current-orthogonalP-active64-labelFree", "A6-orthogonalP-active64-labelFree", "A5 plus activeP64 gradient mask; label-free LineC projection-movement repair"),
        ("A7 B320-current-orthogonalP-lowQuad-labelFree", "A7-orthogonalP-lowQuad-labelFree", "A5 plus quadreadinit075 and identitytailquad020; label-free reservoir-ratio repair"),
        ("A8 B320-current-orthogonalP-boundQ-labelFree", "A8-orthogonalP-boundQ-labelFree", "A5 plus boundQ; label-free noise-leak repair"),
        ("A9 B320-current-orthogonalP-lowQuad-boundQ-labelFree", "A9-orthogonalP-lowQuad-boundQ-labelFree", "A7 plus boundQ; combined label-free LineC repair"),
        ("A10 B320-current-orthogonalP-strongLowQuad-boundQ-labelFree", "A10-orthogonalP-strongLowQuad-boundQ-labelFree", "A9 plus quadreadinit050 and quadRamp010 warm start; aggressive label-free LineC reservoir repair"),
        ("A11 B320-current-orthogonalP-directRead125-labelFree", "A11-orthogonalP-directRead125-labelFree", "A5 plus directreadinit125; label-free direct-branch repair for reservoir overuse"),
        ("A12 B320-current-orthogonalP-identityAmp150-labelFree", "A12-orthogonalP-identityAmp150-labelFree", "A5 plus identityamp150; label-free direct-branch repair for reservoir overuse"),
        ("A13 B320-current-orthogonalP-lowQuad-directRead125-labelFree", "A13-orthogonalP-lowQuad-directRead125-labelFree", "A7 plus directreadinit125; combined label-free direct/low-quad LineC repair"),
        ("A14 B320-current-orthogonalP-lowQuad-identityAmp150-labelFree", "A14-orthogonalP-lowQuad-identityAmp150-labelFree", "A7 plus identityamp150; combined label-free direct/low-quad LineC repair"),
    ]:
        row = {
            "stage": "V1218_B320_LABEL_INIT_AUDIT",
            "candidate_id": B320_ID,
            "ablation_id": ablation_id,
            "uses_y_for_stats": 0,
            "trainprobe_signal_init_uses_labels": 0,
            "trainprobe_signal_init_applied_code_expected": 0,
            "official_training_result_available": 0,
            "small_budget_training_result_available": 0,
            "not_run_reason": "no comparable v12.18 official-budget label-free B320 training artifact exists in current workspace; not fabricated",
            "allowed_next_action": allowed,
            "B320_claim_scope": "current B320 remains diagnostic anchor only until official-budget ablation exists",
            "external_ready_base_claim": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
            "no_proxy": 1,
            "cpu_offload_used": 0,
        }
        smoke_row = smoke_by_candidate.get(smoke_id)
        if smoke_row:
            attach_smoke_metrics(row, smoke_row, out_dir)
            row["not_run_reason"] = (
                "official comparable v12.18 B320 budget is still missing; "
                "small-budget smoke ablation exists and is recorded, but it cannot close R2"
            )
            row["B320_claim_scope"] = "small-budget label-free repair attempt only; official-budget ablation still missing"
        budget_row = anchor_budget_by_candidate.get(smoke_id)
        if budget_row:
            attach_anchor_budget_metrics(row, budget_row, out_dir)
            row["not_run_reason"] = (
                "recovered-anchor-budget label-free ablation exists; strong-pass gate still requires "
                "matched task metrics plus LineC_nontearing evidence"
            )
            row["B320_claim_scope"] = "recovered-anchor-budget label-free ablation audited; claim scope depends on strong-pass result"
        rows.append(row)
    write_csv_rows(out_dir / "v1218_b320_label_init_audit.csv", rows)
    label_free_smoke_rows = [
        row
        for row in smoke_summary
        if str(row.get("candidate_id", "")).startswith("A") and str(row.get("candidate_id", "")) != "A0-labelInit"
    ]
    strict_label_free_smoke_rows = [
        row for row in label_free_smoke_rows if safe_int(row.get("strict_label_free_rows"), 0) > 0
    ]
    best_smoke = None
    if strict_label_free_smoke_rows:
        best_smoke = max(
            strict_label_free_smoke_rows,
            key=lambda row: safe_float(row.get("mean_delta_vs_A0_labelInit"), -1.0e9),
        )
    best_delta = best_smoke.get("mean_delta_vs_A0_labelInit", "") if best_smoke else ""
    label_free_anchor_budget_rows = [
        row
        for row in anchor_budget_summary
        if str(row.get("candidate_id", "")).startswith("A") and str(row.get("candidate_id", "")) != "A0-labelInit"
    ]
    strict_label_free_anchor_budget_rows = [
        row for row in label_free_anchor_budget_rows if safe_int(row.get("strict_label_free_rows"), 0) > 0
    ]
    best_anchor_budget = None
    if strict_label_free_anchor_budget_rows:
        best_anchor_budget = max(
            strict_label_free_anchor_budget_rows,
            key=lambda row: safe_float(row.get("mean_delta_vs_A0_labelInit"), -1.0e9),
        )
    best_anchor_budget_delta = best_anchor_budget.get("mean_delta_vs_A0_labelInit", "") if best_anchor_budget else ""
    best_anchor_budget_auc_time = best_anchor_budget.get("max_AUC_time_ratio_vs_mlp", "") if best_anchor_budget else ""
    best_anchor_budget_linec_available = int(best_anchor_budget is not None and safe_int(best_anchor_budget.get("linec_nontearing_rows"), 0) > 0)
    best_anchor_budget_linec_pass = int(best_anchor_budget_linec_available and safe_int(best_anchor_budget.get("linec_nontearing_all_pass"), 0) == 1)
    anchor_budget_task_strong_pass = int(
        best_anchor_budget is not None
        and safe_float(best_anchor_budget_delta, -1.0e9) >= -0.005
        and safe_float(best_anchor_budget_auc_time, 1.0e9) <= 1.0
    )
    anchor_budget_linec_nontearing_available = best_anchor_budget_linec_available
    anchor_budget_strong_pass = int(
        anchor_budget_task_strong_pass
        and a0_anchor_budget_reproduces_locked_anchor
        and best_anchor_budget_linec_pass == 1
    )
    official_label_free_count = len(strict_label_free_anchor_budget_rows) if a0_anchor_budget_reproduces_locked_anchor else 0
    return {
        "b320_current_anchor_pass": current_pass,
        "b320_current_uses_label_init": 1,
        "b320_label_free_ablation_available_count": official_label_free_count,
        "b320_label_free_smoke_ablation_available_count": len(strict_label_free_smoke_rows),
        "b320_label_free_smoke_best_candidate": best_smoke.get("candidate_id", "") if best_smoke else "",
        "b320_label_free_smoke_best_mean_delta_vs_A0": best_delta,
        "b320_label_free_smoke_underperforms_labelinit": int(best_smoke is not None and safe_float(best_delta, 0.0) < 0.0),
        "b320_label_free_smoke_not_official": int(bool(strict_label_free_smoke_rows)),
        "b320_label_free_official_ablation_still_missing": 1,
        "b320_label_free_anchor_budget_ablation_available_count": len(strict_label_free_anchor_budget_rows),
        "b320_label_free_anchor_budget_best_candidate": best_anchor_budget.get("candidate_id", "") if best_anchor_budget else "",
        "b320_label_free_anchor_budget_best_mean_delta_vs_A0": best_anchor_budget_delta,
        "b320_label_free_anchor_budget_best_max_AUC_time_ratio_vs_mlp": best_anchor_budget_auc_time,
        "b320_label_free_anchor_budget_task_strong_pass": anchor_budget_task_strong_pass,
        "b320_label_free_anchor_budget_strong_pass": anchor_budget_strong_pass,
        "b320_label_free_anchor_budget_underperforms_labelinit": int(best_anchor_budget is not None and safe_float(best_anchor_budget_delta, 0.0) < -0.005),
        "b320_label_free_anchor_budget_linec_nontearing_available": anchor_budget_linec_nontearing_available,
        "b320_label_free_anchor_budget_linec_nontearing_pass": best_anchor_budget_linec_pass,
        "b320_label_free_anchor_budget_a0_gate_pass": a0_anchor_budget_gate_pass,
        "b320_label_free_anchor_budget_a0_task_reproduces_locked_anchor": a0_anchor_budget_task_reproduces_locked_anchor,
        "b320_label_free_anchor_budget_a0_linec_reproduces_locked_anchor": a0_anchor_budget_linec_reproduces_locked_anchor,
        "b320_label_free_anchor_budget_a0_reproduces_locked_anchor": a0_anchor_budget_reproduces_locked_anchor,
        "external_ready_base_claim": 0,
        "B320_claim_scope": "label-informed supervised initialization anchor",
    }


def run_linec_audit(v1217_dir: Path, out_dir: Path) -> dict[str, Any]:
    labels = read_csv_rows(v1217_dir / "v1217_release_labels_audit_only.csv")
    audit_rows: list[dict[str, Any]] = []
    for row in labels:
        out = dict(row)
        out["stage"] = "V1218_LINEC_AUDIT_METRICS"
        out["audit_only_metric"] = 1
        out["uses_label_for_audit"] = 1
        out["uses_ce_for_audit"] = 1
        out["uses_permuted_label_for_audit"] = 1
        out["allowed_for_direction"] = 0
        out["not_allowed_for_direction_reason"] = "hard release and Noise/Reservoir audit targets are label/CE provenance metrics"
        audit_rows.append(out)
    write_csv_rows(out_dir / "v1218_linec_audit_metrics.csv", audit_rows)

    for src, dst, stage in [
        ("v1217_linec_null_distribution.csv", "v1218_linec_null_distribution.csv", "V1218_LINEC_NULL_DISTRIBUTION"),
        ("v1217_linec_threshold_sensitivity.csv", "v1218_linec_threshold_sensitivity.csv", "V1218_LINEC_THRESHOLD_SENSITIVITY"),
        ("v1217_linec_hard_support.csv", "v1218_linec_hard_support.csv", "V1218_LINEC_HARD_SUPPORT"),
    ]:
        rows = read_csv_rows(v1217_dir / src)
        out_rows = []
        for row in rows:
            out = dict(row)
            out["stage"] = stage
            out["reused_from_v1217"] = 1
            out["source_artifact"] = rel(v1217_dir / src)
            out_rows.append(out)
        write_csv_rows(out_dir / dst, out_rows)

    metric_rows = [
        {
            "stage": "V1218_AUDIT_METRIC_PROVENANCE",
            "metric": metric,
            "uses_label": 1,
            "uses_ce": int(metric in {"NoiseSignalLeak_delta", "RealSignalReservoirRatio_delta", "CEp99_delta", "ECE_delta", "Brier_delta", "hard_noise_release", "hard_reservoir_release", "hard_joint_release"}),
            "uses_permuted_label_ce": int(metric in {"NoiseSignalLeak_delta", "RealSignalReservoirRatio_delta"}),
            "allowed_for_audit": 1,
            "allowed_for_direction": 0,
            "reason": "v12.18 separates audit target metrics from deployable label-free observables",
        }
        for metric in [
            "CouplingR2_delta",
            "NoiseSignalLeak_delta",
            "RealSignalReservoirRatio_delta",
            "CEp99_delta",
            "ECE_delta",
            "Brier_delta",
            "hard_noise_release",
            "hard_reservoir_release",
            "hard_joint_release",
        ]
    ]
    write_csv_rows(out_dir / "v1218_audit_metric_provenance.csv", metric_rows)

    null_rows = read_csv_rows(v1217_dir / "v1217_linec_null_distribution.csv")
    random_joint_fp = 0
    for row in null_rows:
        if str(row.get("method")) == "U1-RandomMatchedNorm" and str(row.get("metric")) == "joint_hard_release":
            random_joint_fp += safe_int(row.get("hard_support_rows"), 0)
    hard_support = read_csv_rows(v1217_dir / "v1217_linec_hard_support.csv")
    joint_support = sum(safe_int(row.get("joint_hard_support_rows"), 0) for row in hard_support)
    return {
        "linec_audit_rows": len(audit_rows),
        "linec_joint_hard_support_rows": joint_support,
        "linec_random_joint_false_positive_rows": random_joint_fp,
        "linec_calibration_pass": int(random_joint_fp == 0),
    }


def classify_feature_provenance(feature_family: str, feature_name: str) -> dict[str, Any]:
    name = str(feature_name)
    family = str(feature_family)
    forbidden_reasons: list[str] = []
    uses_label = 0
    uses_ce = 0
    uses_hash = 0
    uses_actual_response = 0
    precommit = 0
    clone_probe = 0
    if name.startswith("actual_") or name in {"pred_coupling_delta"}:
        uses_actual_response = 1
        uses_ce = 1
        forbidden_reasons.append("actual_or_predicted_LineC_response_from_audit_probe")
    if "cotangent" in name or "rank" in name or "reservoir_fraction" in name or "projector" in name or "dissipation" in name or "eigen" in name or "energy" in name:
        uses_ce = 1
        forbidden_reasons.append("derived_from_v1217_LineC_sketch_or_audit_metric")
    if name in {"method_is_control", "method_is_random", "method_is_cotangent_vjp", "method_is_role_actuator", "method_is_projector"}:
        forbidden_reasons.append("method_identity_not_a_physical_precommit_observable")
    if "hash" in name:
        uses_hash = 1
        forbidden_reasons.append("numeric_hash_feature_forbidden_for_promotion")
    if name == "window":
        precommit = 1
        forbidden_reasons.append("window_alone_is_schedule_metadata_not_target_observable")
    if "hard_" in name or "label" in name.lower():
        uses_label = 1
        forbidden_reasons.append("label_or_hard_release_target")
    allowed = int(not forbidden_reasons and precommit)
    return {
        "uses_label_for_feature": uses_label,
        "uses_ce_for_feature": uses_ce,
        "uses_permuted_label_for_feature": int(uses_ce),
        "uses_actual_response_for_feature": uses_actual_response,
        "uses_numeric_hash_for_feature": uses_hash,
        "uses_dataset_name_for_feature": 0,
        "precommit_available": precommit,
        "clone_probe_required": clone_probe,
        "strict_loss_agnostic_allowed": allowed,
        "forbidden_for_direction": int(not allowed),
        "forbidden_reason": ";".join(dict.fromkeys(forbidden_reasons)) if forbidden_reasons else "",
        "feature_provenance": "strict_label_free" if allowed else ("audit_only_or_offline_response"),
        "feature_set": "T1 strict_precommit_unlabeled_features" if allowed else "T3 offline_audit_only_label_features",
        "feature_family": family,
        "feature_name": name,
    }


def strict_v1216_feature_specs() -> list[dict[str, Any]]:
    return [
        {"feature_family": "basis_occupancy", "feature_name": "signal_rank", "source_column": "signal_rank", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "basis_occupancy", "feature_name": "reservoir_rank", "source_column": "reservoir_rank", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "basis_occupancy", "feature_name": "signal_effective_rank", "source_column": "signal_effective_rank", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "basis_occupancy_derived", "feature_name": "rank_balance", "formula": "(signal_rank-reservoir_rank)/(signal_rank+reservoir_rank+eps)", "input_columns": "signal_rank,reservoir_rank", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "basis_occupancy_derived", "feature_name": "rank_total_log", "formula": "log1p(signal_rank+reservoir_rank)", "input_columns": "signal_rank,reservoir_rank", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "logit_free_spectrum_proxy", "feature_name": "reservoir_fraction", "source_column": "reservoir_fraction", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "logit_free_spectrum_proxy", "feature_name": "top_eigen_share", "source_column": "top_eigen_share", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "logit_free_spectrum_proxy_derived", "feature_name": "signal_mass_proxy", "formula": "1-reservoir_fraction", "input_columns": "reservoir_fraction", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "logit_free_spectrum_proxy_derived", "feature_name": "eigen_reservoir_interaction", "formula": "top_eigen_share*(1-reservoir_fraction)", "input_columns": "top_eigen_share,reservoir_fraction", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "projector_geometry", "feature_name": "dissipation_condition", "source_column": "dissipation_condition", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "projector_geometry", "feature_name": "signal_projector_stability", "source_column": "signal_projector_stability", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "projector_geometry", "feature_name": "reservoir_projector_stability", "source_column": "reservoir_projector_stability", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "projector_geometry_derived", "feature_name": "projector_stability_gap", "formula": "signal_projector_stability-reservoir_projector_stability", "input_columns": "signal_projector_stability,reservoir_projector_stability", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "projector_geometry_derived", "feature_name": "projector_stability_ratio", "formula": "signal_projector_stability/(reservoir_projector_stability+eps)", "input_columns": "signal_projector_stability,reservoir_projector_stability", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "projector_geometry_derived", "feature_name": "projector_stability_balance", "formula": "(signal_projector_stability-reservoir_projector_stability)/(signal_projector_stability+reservoir_projector_stability+eps)", "input_columns": "signal_projector_stability,reservoir_projector_stability", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "projector_geometry_derived", "feature_name": "dissipation_log", "formula": "log1p(dissipation_condition)", "input_columns": "dissipation_condition", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "random_cotangent_logit_jacobian", "feature_name": "stable_cotangent_energy", "source_column": "stable_cotangent_energy", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "random_cotangent_logit_jacobian", "feature_name": "unstable_cotangent_energy", "source_column": "unstable_cotangent_energy", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "random_cotangent_logit_jacobian_ensemble", "feature_name": "cotangent_energy_total_log", "formula": "log1p(stable_cotangent_energy+unstable_cotangent_energy)", "input_columns": "stable_cotangent_energy,unstable_cotangent_energy", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "random_cotangent_logit_jacobian_ensemble", "feature_name": "cotangent_energy_ratio", "formula": "stable_cotangent_energy/(unstable_cotangent_energy+eps)", "input_columns": "stable_cotangent_energy,unstable_cotangent_energy", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "random_cotangent_logit_jacobian_ensemble", "feature_name": "cotangent_energy_balance", "formula": "(stable_cotangent_energy-unstable_cotangent_energy)/(stable_cotangent_energy+unstable_cotangent_energy+eps)", "input_columns": "stable_cotangent_energy,unstable_cotangent_energy", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "random_cotangent_logit_jacobian_ensemble", "feature_name": "cotangent_instability_share", "formula": "unstable_cotangent_energy/(stable_cotangent_energy+unstable_cotangent_energy+eps)", "input_columns": "stable_cotangent_energy,unstable_cotangent_energy", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "augmentation_consistency_proxy", "feature_name": "augmentation_used", "source_column": "augmentation_used", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "augmentation_consistency_derived", "feature_name": "augmentation_cotangent_balance", "formula": "augmentation_used*cotangent_energy_balance", "input_columns": "augmentation_used,stable_cotangent_energy,unstable_cotangent_energy", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "augmentation_consistency_derived", "feature_name": "augmentation_projector_balance", "formula": "augmentation_used*projector_stability_balance", "input_columns": "augmentation_used,signal_projector_stability,reservoir_projector_stability", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "method_role_flags", "feature_name": "method_is_control", "formula": "1[method in NoOp/RandomMatchedNorm]", "input_columns": "method", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "method_role_flags", "feature_name": "method_is_random", "formula": "1['Random' in method]", "input_columns": "method", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "method_role_flags", "feature_name": "method_is_cotangent_vjp", "formula": "1['Cotangent' or 'VJP' in method]", "input_columns": "method", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "method_role_flags", "feature_name": "method_is_role_actuator", "formula": "1['Role' or 'Actuator' in method]", "input_columns": "method", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "method_role_flags", "feature_name": "method_is_projector", "formula": "1['Projector' or 'Projection' in method]", "input_columns": "method", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "output_subspace_drift", "feature_name": "noise_target_norm", "source_column": "noise_target_norm", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift", "feature_name": "reservoir_target_norm", "source_column": "reservoir_target_norm", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift", "feature_name": "pred_noise_delta", "source_column": "pred_noise_delta", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift", "feature_name": "pred_reservoir_delta", "source_column": "pred_reservoir_delta", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift", "feature_name": "pred_coupling_delta", "source_column": "pred_coupling_delta", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift_derived", "feature_name": "target_norm_ratio", "formula": "noise_target_norm/(reservoir_target_norm+eps)", "input_columns": "noise_target_norm,reservoir_target_norm", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift_derived", "feature_name": "target_norm_total_log", "formula": "log1p(noise_target_norm+reservoir_target_norm)", "input_columns": "noise_target_norm,reservoir_target_norm", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift_derived", "feature_name": "pred_joint_release_proxy", "formula": "pred_coupling_delta-pred_noise_delta-pred_reservoir_delta", "input_columns": "pred_coupling_delta,pred_noise_delta,pred_reservoir_delta", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "output_subspace_drift_derived", "feature_name": "pred_noise_reservoir_balance", "formula": "pred_noise_delta-pred_reservoir_delta", "input_columns": "pred_noise_delta,pred_reservoir_delta", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 1},
        {"feature_family": "sketch_geometry", "feature_name": "sketch_dim", "source_column": "sketch_dim", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "sketch_geometry", "feature_name": "batch_size", "source_column": "batch_size", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "sketch_geometry_derived", "feature_name": "sketch_work_log", "formula": "log1p(sketch_dim*batch_size)", "input_columns": "sketch_dim,batch_size", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "window_schedule", "feature_name": "window_value", "formula": "window", "input_columns": "window", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "window_schedule", "feature_name": "window_log", "formula": "log1p(window)", "input_columns": "window", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "time_persistence_derived", "feature_name": "persistence_reservoir_fraction_absdiff", "formula": "abs(reservoir_fraction-prev_window_reservoir_fraction)", "input_columns": "source_run,dataset,seed,method,sketch_id,window,reservoir_fraction", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "time_persistence_derived", "feature_name": "persistence_projector_stability_absdiff", "formula": "abs(avg_projector_stability-prev_window_avg_projector_stability)", "input_columns": "source_run,dataset,seed,method,sketch_id,window,signal_projector_stability,reservoir_projector_stability", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
        {"feature_family": "time_persistence_derived", "feature_name": "persistence_pred_coupling_absdiff", "formula": "abs(pred_coupling_delta-prev_window_pred_coupling_delta)", "input_columns": "source_run,dataset,seed,method,sketch_id,window,pred_coupling_delta", "feature_set": "T1 strict_precommit_unlabeled_features", "precommit_available": 1, "clone_probe_required": 0},
    ]


def strict_feature_value(source: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    if spec.get("source_column"):
        return safe_float(source.get(str(spec["source_column"])), 0.0)
    eps = 1.0e-12
    signal_rank = safe_float(source.get("signal_rank"), 0.0)
    reservoir_rank = safe_float(source.get("reservoir_rank"), 0.0)
    reservoir_fraction = safe_float(source.get("reservoir_fraction"), 0.0)
    top_eigen_share = safe_float(source.get("top_eigen_share"), 0.0)
    dissipation_condition = safe_float(source.get("dissipation_condition"), 0.0)
    signal_stability = safe_float(source.get("signal_projector_stability"), 0.0)
    reservoir_stability = safe_float(source.get("reservoir_projector_stability"), 0.0)
    stable_energy = safe_float(source.get("stable_cotangent_energy"), 0.0)
    unstable_energy = safe_float(source.get("unstable_cotangent_energy"), 0.0)
    augmentation_used = safe_float(source.get("augmentation_used"), 0.0)
    noise_target = safe_float(source.get("noise_target_norm"), 0.0)
    reservoir_target = safe_float(source.get("reservoir_target_norm"), 0.0)
    pred_noise = safe_float(source.get("pred_noise_delta"), 0.0)
    pred_reservoir = safe_float(source.get("pred_reservoir_delta"), 0.0)
    pred_coupling = safe_float(source.get("pred_coupling_delta"), 0.0)
    sketch_dim = safe_float(source.get("sketch_dim"), 0.0)
    batch_size = safe_float(source.get("batch_size"), 0.0)
    window = safe_float(source.get("window"), 0.0)
    method = str(source.get("method", ""))
    name = str(spec.get("feature_name", ""))
    cotangent_total = stable_energy + unstable_energy
    projector_total = signal_stability + reservoir_stability
    if name == "rank_balance":
        return (signal_rank - reservoir_rank) / (signal_rank + reservoir_rank + eps)
    if name == "rank_total_log":
        return math.log1p(max(0.0, signal_rank + reservoir_rank))
    if name == "signal_mass_proxy":
        return 1.0 - reservoir_fraction
    if name == "eigen_reservoir_interaction":
        return top_eigen_share * (1.0 - reservoir_fraction)
    if name == "projector_stability_gap":
        return signal_stability - reservoir_stability
    if name == "projector_stability_ratio":
        return signal_stability / (reservoir_stability + eps)
    if name == "projector_stability_balance":
        return (signal_stability - reservoir_stability) / (projector_total + eps)
    if name == "dissipation_log":
        return math.log1p(max(0.0, dissipation_condition))
    if name == "cotangent_energy_total_log":
        return math.log1p(max(0.0, cotangent_total))
    if name == "cotangent_energy_ratio":
        return stable_energy / (unstable_energy + eps)
    if name == "cotangent_energy_balance":
        return (stable_energy - unstable_energy) / (cotangent_total + eps)
    if name == "cotangent_instability_share":
        return unstable_energy / (cotangent_total + eps)
    if name == "augmentation_cotangent_balance":
        return augmentation_used * ((stable_energy - unstable_energy) / (cotangent_total + eps))
    if name == "augmentation_projector_balance":
        return augmentation_used * ((signal_stability - reservoir_stability) / (projector_total + eps))
    if name == "method_is_control":
        return float(method.startswith("U0") or method.startswith("U1") or "NoOp" in method or "RandomMatchedNorm" in method)
    if name == "method_is_random":
        return float("Random" in method)
    if name == "method_is_cotangent_vjp":
        return float("Cotangent" in method or "VJP" in method)
    if name == "method_is_role_actuator":
        return float("Role" in method or "Actuator" in method)
    if name == "method_is_projector":
        return float("Projector" in method or "Projection" in method)
    if name == "target_norm_ratio":
        return noise_target / (reservoir_target + eps)
    if name == "target_norm_total_log":
        return math.log1p(max(0.0, noise_target + reservoir_target))
    if name == "pred_joint_release_proxy":
        return pred_coupling - pred_noise - pred_reservoir
    if name == "pred_noise_reservoir_balance":
        return pred_noise - pred_reservoir
    if name == "sketch_work_log":
        return math.log1p(max(0.0, sketch_dim * batch_size))
    if name == "window_value":
        return window
    if name == "window_log":
        return math.log1p(max(0.0, window))
    if name == "persistence_reservoir_fraction_absdiff":
        return safe_float(source.get("persistence_reservoir_fraction_absdiff"), 0.0)
    if name == "persistence_projector_stability_absdiff":
        return safe_float(source.get("persistence_projector_stability_absdiff"), 0.0)
    if name == "persistence_pred_coupling_absdiff":
        return safe_float(source.get("persistence_pred_coupling_absdiff"), 0.0)
    return 0.0


def write_strict_feature_engineering_audit(out_dir: Path, specs: Sequence[Mapping[str, Any]], strict_source_count: int) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "stage": "V1218_STRICT_VISIBILITY_FEATURE_ENGINEERING_AUDIT",
                "feature_family": spec.get("feature_family", ""),
                "feature_name": spec.get("feature_name", ""),
                "source_column": spec.get("source_column", ""),
                "formula": spec.get("formula", ""),
                "input_columns": spec.get("input_columns", spec.get("source_column", "")),
                "derived": int(bool(spec.get("formula", ""))),
                "strict_source_rows": strict_source_count,
                "uses_label_for_feature": 0,
                "uses_ce_for_feature": 0,
                "uses_permuted_label_for_feature": 0,
                "uses_dataset_name_for_feature": 0,
                "precommit_available": spec.get("precommit_available", 0),
                "clone_probe_required": spec.get("clone_probe_required", 0),
                "forbidden_for_direction": 0,
                "selection_for_promotion": 0,
                "reason": "Line T repair feature engineering from existing strict v1216 label-free columns; post-hoc visibility diagnostic only",
            }
        )
    write_csv_rows(out_dir / "v1218_strict_visibility_feature_engineering_audit.csv", rows)


def v1216_row_id(row: Mapping[str, Any]) -> str:
    return "::".join(
        [
            str(row.get("run_id", "")),
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("window", "")),
            str(row.get("method", "")),
            str(row.get("sketch_id", "")),
            str(row.get("split", "")),
        ]
    )


def semantic_release_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("run_id", row.get("source_run", ""))),
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("window", "")),
        str(row.get("method", "")),
        str(row.get("sketch_id", "")),
    )


def resolve_repo_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def strict_visibility_source_dirs(v1217_dir: Path, fallback_v1216_dir: Path) -> list[Path]:
    route = read_json(v1217_dir / "v1217_route_decision.json")
    source_dirs: list[Path] = []
    for item in route.get("source_run_dirs", []) or []:
        path = resolve_repo_path(item).resolve()
        if (path / "v1216_linec_v2_sketch_targets.csv").exists():
            source_dirs.append(path)
    fallback = fallback_v1216_dir.resolve()
    if (fallback / "v1216_linec_v2_sketch_targets.csv").exists():
        source_dirs.append(fallback)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in source_dirs:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def strict_v1216_source_rows(v1216_repair_dir: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(v1216_repair_dir / "v1216_linec_v2_sketch_targets.csv")
    strict_rows: list[dict[str, str]] = []
    for row in rows:
        if (
            safe_int(row.get("loss_agnostic_direction"), 0) == 1
            and safe_int(row.get("ce_vector_used_for_direction"), 1) == 0
            and safe_int(row.get("label_used_for_direction"), 1) == 0
            and safe_int(row.get("dataset_name_used_for_commit"), 1) == 0
            and safe_int(row.get("validation_used_for_commit"), 1) == 0
        ):
            strict_rows.append(row)
    return strict_rows


def add_strict_time_persistence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        out_rows.append(payload)
        grouped[
            (
                str(payload.get("run_id", "")),
                str(payload.get("dataset", "")),
                str(payload.get("seed", "")),
                str(payload.get("method", "")),
                str(payload.get("sketch_id", "")),
                str(payload.get("split", "")),
            )
        ].append(payload)
    for group in grouped.values():
        prev: Mapping[str, Any] | None = None
        for row in sorted(group, key=lambda item: safe_float(item.get("window"), 0.0)):
            if prev is None:
                row["persistence_reservoir_fraction_absdiff"] = 0.0
                row["persistence_projector_stability_absdiff"] = 0.0
                row["persistence_pred_coupling_absdiff"] = 0.0
            else:
                stability = 0.5 * (
                    safe_float(row.get("signal_projector_stability"), 0.0)
                    + safe_float(row.get("reservoir_projector_stability"), 0.0)
                )
                prev_stability = 0.5 * (
                    safe_float(prev.get("signal_projector_stability"), 0.0)
                    + safe_float(prev.get("reservoir_projector_stability"), 0.0)
                )
                row["persistence_reservoir_fraction_absdiff"] = abs(
                    safe_float(row.get("reservoir_fraction"), 0.0) - safe_float(prev.get("reservoir_fraction"), 0.0)
                )
                row["persistence_projector_stability_absdiff"] = abs(stability - prev_stability)
                row["persistence_pred_coupling_absdiff"] = abs(
                    safe_float(row.get("pred_coupling_delta"), 0.0) - safe_float(prev.get("pred_coupling_delta"), 0.0)
                )
            prev = row
    return out_rows


def build_strict_v1216_long_rows(v1216_repair_dir: Path, v1217_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = {semantic_release_key(row): row for row in read_csv_rows(v1217_dir / "v1217_release_labels_audit_only.csv")}
    specs = strict_v1216_feature_specs()
    source_dirs = strict_visibility_source_dirs(v1217_dir, v1216_repair_dir)
    strict_sources_by_dir = [(source_dir, add_strict_time_persistence(strict_v1216_source_rows(source_dir))) for source_dir in source_dirs]
    strict_sources = [source for _source_dir, rows in strict_sources_by_dir for source in rows]
    long_rows: list[dict[str, Any]] = []
    skipped_unlabeled_sources = 0
    for source_dir, dir_sources in strict_sources_by_dir:
        for source in dir_sources:
            row_id = v1216_row_id(source)
            label = labels.get(semantic_release_key(source))
            if label is None:
                skipped_unlabeled_sources += 1
                continue
            for spec in specs:
                long_rows.append(
                    {
                        "stage": "V1218_STRICT_VISIBILITY_FEATURES",
                        "row_id": row_id,
                        "source_run": source.get("run_id", ""),
                        "dataset": source.get("dataset", ""),
                        "seed": source.get("seed", ""),
                        "window": source.get("window", ""),
                        "method": source.get("method", ""),
                        "sketch_id": source.get("sketch_id", ""),
                        "sketch_family": source.get("sketch_family", ""),
                        "cotangent_family": source.get("cotangent_family", ""),
                        "feature_set": spec["feature_set"],
                        "feature_family": spec["feature_family"],
                        "feature_name": spec["feature_name"],
                        "feature_value": strict_feature_value(source, spec),
                        "feature_provenance": "strict_v1216_loss_agnostic_direction_artifact",
                        "uses_label_for_feature": 0,
                        "uses_ce_for_feature": 0,
                        "uses_permuted_label_for_feature": 0,
                        "uses_dataset_name_for_feature": 0,
                        "precommit_available": spec["precommit_available"],
                        "clone_probe_required": spec["clone_probe_required"],
                        "clone_probe_cost_ms": "",
                        "hard_noise_release": safe_int(label.get("hard_noise_release"), 0),
                        "hard_reservoir_release": safe_int(label.get("hard_reservoir_release"), 0),
                        "hard_joint_release": safe_int(label.get("hard_joint_release"), 0),
                        "split_protocol": "",
                        "heldout_value": "",
                        "forbidden_for_direction": 0,
                        "forbidden_reason": "",
                        "source_artifact": rel(source_dir / "v1216_linec_v2_sketch_targets.csv"),
                    }
                )
    attempts: list[dict[str, Any]] = []
    by_family: dict[str, int] = defaultdict(int)
    for row in long_rows:
        by_family[str(row.get("feature_family", ""))] += 1
    for family, count in sorted(by_family.items()):
        attempts.append(
            {
                "stage": "V1218_STRICT_VISIBILITY_REPAIR_ATTEMPT",
                "attempt": "reuse_v1216_repair_strict_loss_agnostic_observables",
                "feature_family": family,
                "feature_rows": count,
                "source_rows": len(strict_sources),
                "labeled_source_rows": len({str(row.get("row_id", "")) for row in long_rows if str(row.get("feature_family", "")) == family}),
                "skipped_unlabeled_source_rows": skipped_unlabeled_sources,
                "source_artifact": ";".join(rel(source_dir / "v1216_linec_v2_sketch_targets.csv") for source_dir in source_dirs),
                "uses_label_for_feature": 0,
                "uses_ce_for_feature": 0,
                "uses_dataset_name_for_feature": 0,
                "new_training_run": 0,
                "reason": "v12.18 repair step: test existing strict loss-agnostic v1216 observables before opening actuator/P4 gates",
            }
        )
    return long_rows, attempts


def run_feature_provenance_audit(v1217_dir: Path, v1216_repair_dir: Path, out_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(v1217_dir / "v1217_target_visibility_features.csv")
    by_feature: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("feature_family", "")), str(row.get("feature_name", "")))
        if key not in by_feature:
            payload = classify_feature_provenance(*key)
            payload.update({"stage": "V1218_FEATURE_PROVENANCE_AUDIT", "row_count": 0})
            by_feature[key] = payload
        by_feature[key]["row_count"] += 1
    source_dirs = strict_visibility_source_dirs(v1217_dir, v1216_repair_dir)
    strict_sources = [source for source_dir in source_dirs for source in strict_v1216_source_rows(source_dir)]
    source_artifacts = ";".join(rel(source_dir / "v1216_linec_v2_sketch_targets.csv") for source_dir in source_dirs)
    write_strict_feature_engineering_audit(out_dir, strict_v1216_feature_specs(), len(strict_sources))
    for spec in strict_v1216_feature_specs():
        key = (str(spec["feature_family"]), str(spec["feature_name"]))
        payload = {
            "uses_label_for_feature": 0,
            "uses_ce_for_feature": 0,
            "uses_permuted_label_for_feature": 0,
            "uses_actual_response_for_feature": 0,
            "uses_numeric_hash_for_feature": 0,
            "uses_dataset_name_for_feature": 0,
            "precommit_available": spec["precommit_available"],
            "clone_probe_required": spec["clone_probe_required"],
            "strict_loss_agnostic_allowed": 1,
            "forbidden_for_direction": 0,
            "forbidden_reason": "",
            "feature_provenance": "strict_v1216_loss_agnostic_direction_artifact",
            "feature_set": spec["feature_set"],
            "feature_family": spec["feature_family"],
            "feature_name": spec["feature_name"],
            "stage": "V1218_FEATURE_PROVENANCE_AUDIT",
            "row_count": len(strict_sources),
            "source_artifact": source_artifacts,
        }
        by_feature[key] = payload
    out_rows = list(by_feature.values())
    write_csv_rows(out_dir / "v1218_feature_provenance_audit.csv", out_rows)
    return {
        "feature_provenance_rows": len(out_rows),
        "strict_allowed_feature_count": sum(safe_int(row.get("strict_loss_agnostic_allowed"), 0) for row in out_rows),
        "forbidden_feature_count": sum(safe_int(row.get("forbidden_for_direction"), 0) for row in out_rows),
    }


def split_visibility_rows(rows: Sequence[Mapping[str, Any]], split_key: str) -> list[str]:
    return sorted({str(row.get(split_key, "")) for row in rows})


def wide_feature_matrix(long_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, dict[str, Any]] = {}
    feature_names = sorted({str(row.get("feature_name", "")) for row in long_rows if row.get("feature_name")})
    for row in long_rows:
        row_id = str(row.get("row_id", ""))
        if row_id not in grouped:
            grouped[row_id] = {
                "row_id": row_id,
                "source_run": row.get("source_run", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("window", ""),
                "method": row.get("method", ""),
                "sketch_family": row.get("sketch_family", ""),
                "cotangent_family": row.get("cotangent_family", ""),
                "hard_noise_release": safe_int(row.get("hard_noise_release"), 0),
                "hard_reservoir_release": safe_int(row.get("hard_reservoir_release"), 0),
                "hard_joint_release": safe_int(row.get("hard_joint_release"), 0),
            }
        grouped[row_id][str(row.get("feature_name", ""))] = safe_float(row.get("feature_value"), 0.0)
    wide = []
    for payload in grouped.values():
        for name in feature_names:
            payload.setdefault(name, 0.0)
        wide.append(payload)
    return sorted(wide, key=lambda row: str(row.get("row_id", ""))), feature_names


def score_strict_visibility(long_rows: Sequence[Mapping[str, Any]], support_concentrated: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    wide, feature_names = wide_feature_matrix(long_rows)
    split_map = {
        "leave-dataset-out": "dataset",
        "leave-seed-out": "seed",
        "leave-window-out": "window",
        "leave-method-out": "method",
        "leave-source-run-out": "source_run",
        "leave-actuator-family-out": "cotangent_family",
    }
    leaveout_rows: list[dict[str, Any]] = []
    if not wide or not feature_names:
        return (
            [
                {
                    "stage": "V1218_STRICT_VISIBILITY_SCORES",
                    "feature_set": "T1 strict_precommit_unlabeled_features",
                    "status": "not_run_no_strict_features",
                    "rows": 0,
                    "feature_columns": 0,
                    "visibility_pass": 0,
                }
            ],
            [
                {
                    "stage": "V1218_STRICT_VISIBILITY_LEAVEOUT",
                    "feature_set": "T1 strict_precommit_unlabeled_features",
                    "split_protocol": split,
                    "status": "not_run_no_strict_features",
                }
                for split in split_map
            ],
            {"strict_visibility_pass": 0, "strict_visibility_blocker": "no_strict_feature_rows"},
        )

    x_all = np.array([[safe_float(row.get(name), 0.0) for name in feature_names] for row in wide], dtype=float)
    required_split_failures = 0
    for split_name, key in split_map.items():
        heldouts = sorted({str(row.get(key, "")) for row in wide})
        if len(heldouts) < 2:
            required_split_failures += 1
            leaveout_rows.append(
                {
                    "stage": "V1218_STRICT_VISIBILITY_LEAVEOUT",
                    "feature_set": "T1 strict_precommit_unlabeled_features",
                    "split_protocol": split_name,
                    "heldout_value": heldouts[0] if heldouts else "",
                    "status": "not_run_single_heldout_value",
                    "rows": len(wide),
                    "feature_columns": len(feature_names),
                    "AUC_noise": "",
                    "AUC_reservoir": "",
                    "AUC_joint": "",
                    "precision_at_k_joint": "",
                    "recall_at_k_joint": "",
                    "reason": f"{key} has only one value; cannot perform {split_name}",
                }
            )
            continue
        for heldout in heldouts:
            test_idx = [idx for idx, row in enumerate(wide) if str(row.get(key, "")) == heldout]
            train_idx = [idx for idx, row in enumerate(wide) if str(row.get(key, "")) != heldout]
            if not test_idx or not train_idx:
                required_split_failures += 1
                continue
            row_out: dict[str, Any] = {
                "stage": "V1218_STRICT_VISIBILITY_LEAVEOUT",
                "feature_set": "T1 strict_precommit_unlabeled_features",
                "split_protocol": split_name,
                "heldout_value": heldout,
                "status": "ran",
                "rows": len(test_idx),
                "train_rows": len(train_idx),
                "feature_columns": len(feature_names),
            }
            for target, label_col in [
                ("noise", "hard_noise_release"),
                ("reservoir", "hard_reservoir_release"),
                ("joint", "hard_joint_release"),
            ]:
                train_y = np.array([safe_int(wide[idx].get(label_col), 0) for idx in train_idx], dtype=float)
                test_y = [safe_int(wide[idx].get(label_col), 0) for idx in test_idx]
                if int(train_y.sum()) == 0 or int(sum(test_y)) == 0:
                    pred = np.zeros((len(test_idx),), dtype=float)
                    row_out[f"{target}_train_positive_rows"] = int(train_y.sum())
                    row_out[f"{target}_test_positive_rows"] = int(sum(test_y))
                else:
                    pred = ridge_predict(x_all[train_idx], train_y, x_all[test_idx])
                    row_out[f"{target}_train_positive_rows"] = int(train_y.sum())
                    row_out[f"{target}_test_positive_rows"] = int(sum(test_y))
                auc = auc_score(test_y, list(pred))
                prec, rec, hits, positives = precision_recall_at_k(test_y, list(pred))
                row_out[f"AUC_{target}"] = auc if not math.isnan(auc) else ""
                row_out[f"precision_at_k_{target}"] = prec if not math.isnan(prec) else ""
                row_out[f"recall_at_k_{target}"] = rec if not math.isnan(rec) else ""
                row_out[f"topk_hits_{target}"] = hits
                row_out[f"positive_rows_{target}"] = positives
            leaveout_rows.append(row_out)

    ran_rows = [row for row in leaveout_rows if row.get("status") == "ran"]
    auc_joint_values = [safe_float(row.get("AUC_joint"), float("nan")) for row in ran_rows if row.get("AUC_joint") != ""]
    auc_noise_values = [safe_float(row.get("AUC_noise"), float("nan")) for row in ran_rows if row.get("AUC_noise") != ""]
    auc_res_values = [safe_float(row.get("AUC_reservoir"), float("nan")) for row in ran_rows if row.get("AUC_reservoir") != ""]
    prec_joint_values = [safe_float(row.get("precision_at_k_joint"), float("nan")) for row in ran_rows if row.get("precision_at_k_joint") != ""]
    rec_joint_values = [safe_float(row.get("recall_at_k_joint"), float("nan")) for row in ran_rows if row.get("recall_at_k_joint") != ""]
    auc_joint_min = finite_min(auc_joint_values)
    precision_joint_min = finite_min(prec_joint_values)
    recall_joint_min = finite_min(rec_joint_values)
    exploratory_pass = int(not math.isnan(auc_joint_min) and auc_joint_min >= 0.65)
    hard_pass = int(
        not math.isnan(precision_joint_min)
        and not math.isnan(recall_joint_min)
        and precision_joint_min >= 0.25
        and recall_joint_min >= 0.20
    )
    robustness_pass = int(required_split_failures == 0 and support_concentrated == 0)
    visibility_pass = int(exploratory_pass == 1 and hard_pass == 1 and robustness_pass == 1)
    blocker_bits = []
    if exploratory_pass == 0:
        blocker_bits.append("strict_auc_joint_below_gate")
    if hard_pass == 0:
        blocker_bits.append("strict_precision_recall_below_gate")
    if support_concentrated:
        blocker_bits.append("support_concentrated")
    if required_split_failures:
        blocker_bits.append("required_leaveout_not_runnable")
    score_rows = [
        {
            "stage": "V1218_STRICT_VISIBILITY_SCORES",
            "feature_set": "T1 strict_precommit_unlabeled_features",
            "status": "ran_strict_v1216_observable_repair",
            "rows": len(wide),
            "feature_columns": len(feature_names),
            "long_feature_rows": len(long_rows),
            "AUC_noise_min": finite_min(auc_noise_values) if auc_noise_values else "",
            "AUC_reservoir_min": finite_min(auc_res_values) if auc_res_values else "",
            "AUC_joint_min": auc_joint_min if not math.isnan(auc_joint_min) else "",
            "AUC_joint_max": finite_max(auc_joint_values) if auc_joint_values else "",
            "precision_at_k_joint_min": precision_joint_min if not math.isnan(precision_joint_min) else "",
            "recall_at_k_joint_min": recall_joint_min if not math.isnan(recall_joint_min) else "",
            "required_split_failures": required_split_failures,
            "support_concentrated": support_concentrated,
            "visibility_exploratory_pass": exploratory_pass,
            "visibility_hard_pass": hard_pass,
            "visibility_robustness_pass": robustness_pass,
            "visibility_pass": visibility_pass,
            "blocker": ";".join(blocker_bits),
            "note": "Strict features are reused from v1216 loss_agnostic_direction rows; no label/CE fields enter the feature matrix.",
        }
    ]
    return score_rows, leaveout_rows, {
        "strict_visibility_feature_rows": len(long_rows),
        "strict_visibility_wide_rows": len(wide),
        "strict_visibility_feature_columns": len(feature_names),
        "strict_auc_joint_min": auc_joint_min if not math.isnan(auc_joint_min) else "",
        "strict_precision_joint_min": precision_joint_min if not math.isnan(precision_joint_min) else "",
        "strict_recall_joint_min": recall_joint_min if not math.isnan(recall_joint_min) else "",
        "strict_visibility_pass": visibility_pass,
        "strict_visibility_blocker": ";".join(blocker_bits),
        "strict_visibility_required_split_failures": required_split_failures,
    }


def write_strict_visibility_family_ablation(long_rows: Sequence[Mapping[str, Any]], support_concentrated: int, out_dir: Path) -> dict[str, Any]:
    families = sorted({str(row.get("feature_family", "")) for row in long_rows if str(row.get("feature_family", "")).strip()})
    feature_sets: list[tuple[str, set[str]]] = [("all_current", set(families))]
    if "output_subspace_drift" in families:
        feature_sets.append(("baseline_without_output_subspace_drift", {family for family in families if family != "output_subspace_drift"}))
    feature_sets.extend((f"only_{family}", {family}) for family in families)
    feature_sets.extend((f"all_except_{family}", {item for item in families if item != family}) for family in families)
    rows: list[dict[str, Any]] = []
    for set_name, keep in feature_sets:
        subset = [row for row in long_rows if str(row.get("feature_family", "")) in keep]
        score_rows, _leaveout_rows, summary = score_strict_visibility(subset, support_concentrated)
        score = score_rows[0] if score_rows else {}
        rows.append(
            {
                "stage": "V1218_STRICT_VISIBILITY_FAMILY_ABLATION",
                "feature_set_name": set_name,
                "feature_families": ",".join(sorted(keep)),
                "rows": summary.get("strict_visibility_wide_rows", ""),
                "long_feature_rows": summary.get("strict_visibility_feature_rows", ""),
                "feature_columns": summary.get("strict_visibility_feature_columns", ""),
                "AUC_joint_min": summary.get("strict_auc_joint_min", ""),
                "AUC_joint_max": score.get("AUC_joint_max", ""),
                "precision_at_k_joint_min": summary.get("strict_precision_joint_min", ""),
                "recall_at_k_joint_min": summary.get("strict_recall_joint_min", ""),
                "required_split_failures": summary.get("strict_visibility_required_split_failures", ""),
                "support_concentrated": support_concentrated,
                "visibility_pass": summary.get("strict_visibility_pass", ""),
                "blocker": summary.get("strict_visibility_blocker", ""),
                "uses_label_for_feature": 0,
                "uses_ce_for_feature": 0,
                "selection_for_promotion": 0,
                "reason": "post-hoc Line T repair diagnostic; not used to lower gates or claim promotion",
            }
        )
    write_csv_rows(out_dir / "v1218_strict_visibility_family_ablation.csv", rows)
    valid = [row for row in rows if row.get("AUC_joint_min") != ""]
    best = max(valid, key=lambda row: safe_float(row.get("AUC_joint_min"), -1.0), default={})
    return {
        "line_t_family_ablation_rows": len(rows),
        "line_t_family_ablation_best_set": best.get("feature_set_name", ""),
        "line_t_family_ablation_best_auc_joint_min": best.get("AUC_joint_min", ""),
        "line_t_family_ablation_best_visibility_pass": best.get("visibility_pass", ""),
    }


def write_strict_visibility_composite_repair(long_rows: Sequence[Mapping[str, Any]], support_concentrated: int, out_dir: Path) -> dict[str, Any]:
    wide, feature_names = wide_feature_matrix(long_rows)
    split_map = {
        "leave-dataset-out": "dataset",
        "leave-seed-out": "seed",
        "leave-window-out": "window",
        "leave-method-out": "method",
        "leave-source-run-out": "source_run",
        "leave-actuator-family-out": "cotangent_family",
    }
    combine_rules = ["mean_noise_reservoir_rank", "min_noise_reservoir_rank", "product_noise_reservoir_rank"]
    rows: list[dict[str, Any]] = []
    if not wide or not feature_names:
        write_csv_rows(out_dir / "v1218_strict_visibility_composite_repair.csv", rows)
        return {
            "line_t_composite_repair_rows": 0,
            "line_t_composite_repair_best_protocol": "",
            "line_t_composite_repair_best_auc_joint_min": "",
            "line_t_composite_repair_best_precision_joint_min": "",
            "line_t_composite_repair_best_recall_joint_min": "",
            "line_t_composite_repair_best_pass": 0,
        }

    x_all = np.array([[safe_float(row.get(name), 0.0) for name in feature_names] for row in wide], dtype=float)
    for rule in combine_rules:
        leaveout_results: list[dict[str, Any]] = []
        for split_name, key in split_map.items():
            heldouts = sorted({str(row.get(key, "")) for row in wide})
            if len(heldouts) < 2:
                leaveout_results.append(
                    {
                        "stage": "V1218_STRICT_VISIBILITY_COMPOSITE_REPAIR",
                        "combine_rule": rule,
                        "split_protocol": split_name,
                        "heldout_value": heldouts[0] if heldouts else "",
                        "status": "not_run_single_heldout_value",
                        "feature_columns": len(feature_names),
                    }
                )
                continue
            for heldout in heldouts:
                test_idx = [idx for idx, row in enumerate(wide) if str(row.get(key, "")) == heldout]
                train_idx = [idx for idx, row in enumerate(wide) if str(row.get(key, "")) != heldout]
                if not test_idx or not train_idx:
                    continue
                train_noise = np.array([safe_int(wide[idx].get("hard_noise_release"), 0) for idx in train_idx], dtype=float)
                train_reservoir = np.array([safe_int(wide[idx].get("hard_reservoir_release"), 0) for idx in train_idx], dtype=float)
                test_joint = [safe_int(wide[idx].get("hard_joint_release"), 0) for idx in test_idx]
                pred_noise = (
                    ridge_predict(x_all[train_idx], train_noise, x_all[test_idx])
                    if int(train_noise.sum()) > 0
                    else np.zeros((len(test_idx),), dtype=float)
                )
                pred_reservoir = (
                    ridge_predict(x_all[train_idx], train_reservoir, x_all[test_idx])
                    if int(train_reservoir.sum()) > 0
                    else np.zeros((len(test_idx),), dtype=float)
                )
                noise_rank = rank01(pred_noise)
                reservoir_rank = rank01(pred_reservoir)
                if rule == "min_noise_reservoir_rank":
                    score = np.minimum(noise_rank, reservoir_rank)
                elif rule == "product_noise_reservoir_rank":
                    score = noise_rank * reservoir_rank
                else:
                    score = 0.5 * (noise_rank + reservoir_rank)
                auc = auc_score(test_joint, score.tolist())
                prec, rec, hits, positives = precision_recall_at_k(test_joint, score.tolist())
                leaveout_results.append(
                    {
                        "stage": "V1218_STRICT_VISIBILITY_COMPOSITE_REPAIR",
                        "combine_rule": rule,
                        "split_protocol": split_name,
                        "heldout_value": heldout,
                        "status": "ran",
                        "rows": len(test_idx),
                        "train_rows": len(train_idx),
                        "feature_columns": len(feature_names),
                        "noise_train_positive_rows": int(train_noise.sum()),
                        "reservoir_train_positive_rows": int(train_reservoir.sum()),
                        "joint_test_positive_rows": int(sum(test_joint)),
                        "AUC_joint": auc if not math.isnan(auc) else "",
                        "precision_at_k_joint": prec if not math.isnan(prec) else "",
                        "recall_at_k_joint": rec if not math.isnan(rec) else "",
                        "topk_hits_joint": hits,
                        "positive_rows_joint": positives,
                        "uses_label_for_feature": 0,
                        "uses_ce_for_feature": 0,
                        "selection_for_promotion": 0,
                        "reason": "post-hoc strict composite visibility repair; labels train audit scorer only, not feature construction or direction",
                    }
                )
        ran = [row for row in leaveout_results if row.get("status") == "ran"]
        auc_values = [safe_float(row.get("AUC_joint"), float("nan")) for row in ran if row.get("AUC_joint") != ""]
        precision_values = [safe_float(row.get("precision_at_k_joint"), float("nan")) for row in ran if row.get("precision_at_k_joint") != ""]
        recall_values = [safe_float(row.get("recall_at_k_joint"), float("nan")) for row in ran if row.get("recall_at_k_joint") != ""]
        auc_min = finite_min(auc_values)
        precision_min = finite_min(precision_values)
        recall_min = finite_min(recall_values)
        pass_flag = int(
            not math.isnan(auc_min)
            and auc_min >= 0.65
            and not math.isnan(precision_min)
            and precision_min >= 0.25
            and not math.isnan(recall_min)
            and recall_min >= 0.20
            and support_concentrated == 0
        )
        rows.extend(leaveout_results)
        rows.append(
            {
                "stage": "V1218_STRICT_VISIBILITY_COMPOSITE_REPAIR_SUMMARY",
                "combine_rule": rule,
                "status": "summary",
                "feature_columns": len(feature_names),
                "AUC_joint_min": auc_min if not math.isnan(auc_min) else "",
                "AUC_joint_max": finite_max(auc_values) if auc_values else "",
                "precision_at_k_joint_min": precision_min if not math.isnan(precision_min) else "",
                "recall_at_k_joint_min": recall_min if not math.isnan(recall_min) else "",
                "support_concentrated": support_concentrated,
                "repair_pass": pass_flag,
                "uses_label_for_feature": 0,
                "uses_ce_for_feature": 0,
                "selection_for_promotion": 0,
                "reason": "composite rank over separately predicted hard_noise_release and hard_reservoir_release",
            }
        )
    write_csv_rows(out_dir / "v1218_strict_visibility_composite_repair.csv", rows)
    summaries = [row for row in rows if row.get("stage") == "V1218_STRICT_VISIBILITY_COMPOSITE_REPAIR_SUMMARY"]
    best = max(summaries, key=lambda row: safe_float(row.get("AUC_joint_min"), -1.0), default={})
    return {
        "line_t_composite_repair_rows": len(rows),
        "line_t_composite_repair_best_protocol": best.get("combine_rule", ""),
        "line_t_composite_repair_best_auc_joint_min": best.get("AUC_joint_min", ""),
        "line_t_composite_repair_best_precision_joint_min": best.get("precision_at_k_joint_min", ""),
        "line_t_composite_repair_best_recall_joint_min": best.get("recall_at_k_joint_min", ""),
        "line_t_composite_repair_best_pass": best.get("repair_pass", 0),
    }


def context_rank_matrix(
    wide: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    group_keys: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    base_names = [
        name
        for name in feature_names
        if not name.startswith("method_is_") and name not in {"window_value", "window_log"}
    ]
    x = np.array([[safe_float(row.get(name), 0.0) for name in base_names] for row in wide], dtype=float)
    ranks = np.zeros_like(x)
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(wide):
        groups[tuple(str(row.get(key, "")) for key in group_keys)].append(idx)
    for idxs in groups.values():
        sub = x[idxs]
        for col_idx in range(sub.shape[1]):
            order = np.argsort(sub[:, col_idx], kind="mergesort")
            col_ranks = np.zeros((len(idxs),), dtype=float)
            col_ranks[order] = np.arange(len(idxs), dtype=float) / max(1.0, float(len(idxs) - 1))
            ranks[idxs, col_idx] = col_ranks
    names = ["ctx_rank__" + name for name in base_names]
    return ranks, names


def context_rank_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, scorer: str) -> np.ndarray:
    if scorer == "ridge":
        return ridge_predict(train_x, train_y.astype(float), test_x)
    return centroid_predict(train_x, train_y.astype(int), test_x, scorer)


def write_strict_visibility_context_rank_repair(
    long_rows: Sequence[Mapping[str, Any]],
    support_concentrated: int,
    out_dir: Path,
) -> dict[str, Any]:
    wide, feature_names = wide_feature_matrix(long_rows)
    split_map = {
        "leave-dataset-out": "dataset",
        "leave-seed-out": "seed",
        "leave-window-out": "window",
        "leave-method-out": "method",
        "leave-source-run-out": "source_run",
        "leave-actuator-family-out": "cotangent_family",
    }
    variants: list[tuple[str, Sequence[str]]] = [
        ("context_rank_source_dataset_seed_window", ["source_run", "dataset", "seed", "window"]),
        ("context_rank_dataset_seed_window", ["dataset", "seed", "window"]),
        ("context_rank_source_dataset_seed", ["source_run", "dataset", "seed"]),
    ]
    scorers = ["ridge", "positive_centroid_dot", "positive_centroid_distance", "contrast_centroid_dot", "contrast_centroid_distance"]
    rows: list[dict[str, Any]] = []
    if not wide or not feature_names:
        write_csv_rows(out_dir / "v1218_strict_visibility_context_rank_repair.csv", rows)
        return {
            "line_t_context_rank_repair_rows": 0,
            "line_t_context_rank_repair_best_protocol": "",
            "line_t_context_rank_repair_best_auc_joint_min": "",
            "line_t_context_rank_repair_best_precision_joint_min": "",
            "line_t_context_rank_repair_best_recall_joint_min": "",
            "line_t_context_rank_repair_best_pass": 0,
        }

    for variant_name, group_keys in variants:
        matrix, context_feature_names = context_rank_matrix(wide, feature_names, group_keys)
        for scorer in scorers:
            protocol = f"{variant_name}/{scorer}"
            leaveout_results: list[dict[str, Any]] = []
            required_split_failures = 0
            for split_name, key in split_map.items():
                heldouts = sorted({str(row.get(key, "")) for row in wide})
                if len(heldouts) < 2:
                    required_split_failures += 1
                    leaveout_results.append(
                        {
                            "stage": "V1218_STRICT_VISIBILITY_CONTEXT_RANK_REPAIR",
                            "protocol": protocol,
                            "split_protocol": split_name,
                            "heldout_value": heldouts[0] if heldouts else "",
                            "status": "not_run_single_heldout_value",
                            "context_group_keys": ",".join(group_keys),
                            "context_feature_columns": len(context_feature_names),
                        }
                    )
                    continue
                for heldout in heldouts:
                    test_idx = [idx for idx, row in enumerate(wide) if str(row.get(key, "")) == heldout]
                    train_idx = [idx for idx, row in enumerate(wide) if str(row.get(key, "")) != heldout]
                    if not test_idx or not train_idx:
                        required_split_failures += 1
                        continue
                    train_y = np.array([safe_int(wide[idx].get("hard_joint_release"), 0) for idx in train_idx], dtype=int)
                    test_y = [safe_int(wide[idx].get("hard_joint_release"), 0) for idx in test_idx]
                    if int(train_y.sum()) == 0 or int(sum(test_y)) == 0:
                        pred = np.zeros((len(test_idx),), dtype=float)
                    else:
                        pred = context_rank_predict(matrix[train_idx], train_y, matrix[test_idx], scorer)
                    auc = auc_score(test_y, pred.tolist())
                    prec, rec, hits, positives = precision_recall_at_k(test_y, pred.tolist())
                    leaveout_results.append(
                        {
                            "stage": "V1218_STRICT_VISIBILITY_CONTEXT_RANK_REPAIR",
                            "protocol": protocol,
                            "split_protocol": split_name,
                            "heldout_value": heldout,
                            "status": "ran",
                            "rows": len(test_idx),
                            "train_rows": len(train_idx),
                            "context_group_keys": ",".join(group_keys),
                            "context_feature_columns": len(context_feature_names),
                            "joint_train_positive_rows": int(train_y.sum()),
                            "joint_test_positive_rows": int(sum(test_y)),
                            "AUC_joint": auc if not math.isnan(auc) else "",
                            "precision_at_k_joint": prec if not math.isnan(prec) else "",
                            "recall_at_k_joint": rec if not math.isnan(rec) else "",
                            "topk_hits_joint": hits,
                            "positive_rows_joint": positives,
                            "uses_label_for_feature": 0,
                            "uses_ce_for_feature": 0,
                            "uses_method_identity_feature": 0,
                            "selection_for_promotion": 0,
                            "reason": "post-hoc Line T method-generalization repair using precommit cohort rank normalization; labels train audit scorer only",
                        }
                    )
            ran = [row for row in leaveout_results if row.get("status") == "ran"]
            auc_values = [safe_float(row.get("AUC_joint"), float("nan")) for row in ran if row.get("AUC_joint") != ""]
            precision_values = [safe_float(row.get("precision_at_k_joint"), float("nan")) for row in ran if row.get("precision_at_k_joint") != ""]
            recall_values = [safe_float(row.get("recall_at_k_joint"), float("nan")) for row in ran if row.get("recall_at_k_joint") != ""]
            auc_min = finite_min(auc_values)
            precision_min = finite_min(precision_values)
            recall_min = finite_min(recall_values)
            pass_flag = int(
                not math.isnan(auc_min)
                and auc_min >= 0.65
                and not math.isnan(precision_min)
                and precision_min >= 0.25
                and not math.isnan(recall_min)
                and recall_min >= 0.20
                and support_concentrated == 0
                and required_split_failures == 0
            )
            rows.extend(leaveout_results)
            worst = min(ran, key=lambda row: safe_float(row.get("AUC_joint"), 1.0e9), default={})
            rows.append(
                {
                    "stage": "V1218_STRICT_VISIBILITY_CONTEXT_RANK_REPAIR_SUMMARY",
                    "protocol": protocol,
                    "status": "summary",
                    "context_group_keys": ",".join(group_keys),
                    "context_feature_columns": len(context_feature_names),
                    "AUC_joint_min": auc_min if not math.isnan(auc_min) else "",
                    "AUC_joint_max": finite_max(auc_values) if auc_values else "",
                    "precision_at_k_joint_min": precision_min if not math.isnan(precision_min) else "",
                    "recall_at_k_joint_min": recall_min if not math.isnan(recall_min) else "",
                    "worst_split_protocol": worst.get("split_protocol", ""),
                    "worst_heldout_value": worst.get("heldout_value", ""),
                    "required_split_failures": required_split_failures,
                    "support_concentrated": support_concentrated,
                    "repair_pass": pass_flag,
                    "uses_label_for_feature": 0,
                    "uses_ce_for_feature": 0,
                    "uses_method_identity_feature": 0,
                    "selection_for_promotion": 0,
                    "reason": "context-rank repair remains audit-only unless predeclared in a future promotion protocol",
                }
            )
    write_csv_rows(out_dir / "v1218_strict_visibility_context_rank_repair.csv", rows)
    summaries = [row for row in rows if row.get("stage") == "V1218_STRICT_VISIBILITY_CONTEXT_RANK_REPAIR_SUMMARY"]
    best = max(summaries, key=lambda row: safe_float(row.get("AUC_joint_min"), -1.0), default={})
    return {
        "line_t_context_rank_repair_rows": len(rows),
        "line_t_context_rank_repair_best_protocol": best.get("protocol", ""),
        "line_t_context_rank_repair_best_auc_joint_min": best.get("AUC_joint_min", ""),
        "line_t_context_rank_repair_best_precision_joint_min": best.get("precision_at_k_joint_min", ""),
        "line_t_context_rank_repair_best_recall_joint_min": best.get("recall_at_k_joint_min", ""),
        "line_t_context_rank_repair_best_pass": best.get("repair_pass", 0),
    }


def unique_csv_values(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    vals = sorted({str(row.get(key, "")) for row in rows if str(row.get(key, "")).strip()})
    return ",".join(vals)


def write_line_t_support_expansion_audit(
    v1217_dir: Path,
    v1216_repair_dir: Path,
    out_dir: Path,
    support_concentrated: int,
    hard_joint_support_rows: int,
) -> dict[str, Any]:
    source_calibration = {row.get("source_run", ""): row for row in read_csv_rows(v1217_dir / "v1217_linec_source_calibration.csv")}
    hard_support_rows = read_csv_rows(v1217_dir / "v1217_linec_hard_support.csv")
    hard_joint_by_source: dict[str, int] = defaultdict(int)
    for row in hard_support_rows:
        hard_joint_by_source[str(row.get("source_run", ""))] += safe_int(row.get("joint_hard_support_rows"), 0)
    release_rows = read_csv_rows(v1217_dir / "v1217_release_labels_audit_only.csv")
    release_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in release_rows:
        release_by_source[str(row.get("source_run", ""))].append(row)

    candidate_files = sorted(v1216_repair_dir.parent.glob("*/v1216_linec_v2_sketch_targets.csv"))
    audit_rows: list[dict[str, Any]] = []
    for linec_path in candidate_files:
        linec_rows = read_csv_rows(linec_path)
        if not linec_rows:
            continue
        source_run = str(linec_rows[0].get("run_id", linec_path.parent.name))
        calib = source_calibration.get(source_run, {})
        rel_rows = release_by_source.get(source_run, [])
        source_calibration_pass = safe_int(calib.get("source_calibration_pass"), -1)
        random_fp = safe_int(calib.get("random_joint_false_positive_rows"), 0)
        control_fp = safe_int(calib.get("control_joint_false_positive_rows"), 0)
        release_hard = sum(safe_int(row.get("hard_joint_release"), 0) for row in rel_rows)
        strict_loss_agnostic_rows = sum(
            1
            for row in linec_rows
            if safe_int(row.get("loss_agnostic_direction"), 0) == 1
            and safe_int(row.get("ce_vector_used_for_direction"), 0) == 0
            and safe_int(row.get("label_used_for_direction"), 0) == 0
            and safe_int(row.get("dataset_name_used_for_commit"), 0) == 0
        )
        status_bits: list[str] = []
        if source_calibration_pass != 1:
            if calib:
                status_bits.append("blocked_by_source_calibration_false_positive")
            else:
                status_bits.append("not_in_v1217_source_calibration")
        if release_hard <= 0:
            status_bits.append("no_v1217_release_hard_joint_rows")
        if source_run == v1216_repair_dir.name and support_concentrated:
            status_bits.append("already_used_but_hard_joint_support_concentrated")
        if "smoke" in source_run.lower():
            status_bits.append("smoke_scope_only_not_official_independent_source")
        usable = int(source_calibration_pass == 1 and release_hard > 0 and not support_concentrated)
        if usable:
            status_bits.append("usable_for_visibility_promotion")
        audit_rows.append(
            {
                "stage": "V1218_LINE_T_SUPPORT_EXPANSION_AUDIT",
                "source_run": source_run,
                "source_dir": rel(linec_path.parent),
                "linec_rows": len(linec_rows),
                "datasets": unique_csv_values(linec_rows, "dataset"),
                "seeds": unique_csv_values(linec_rows, "seed"),
                "windows": unique_csv_values(linec_rows, "window"),
                "batch_sizes": unique_csv_values(linec_rows, "batch_size"),
                "sketch_dims": unique_csv_values(linec_rows, "sketch_dim"),
                "strict_loss_agnostic_rows": strict_loss_agnostic_rows,
                "v1217_source_calibration_available": int(bool(calib)),
                "v1217_source_calibration_pass": source_calibration_pass if calib else "",
                "random_joint_false_positive_rows": random_fp if calib else "",
                "control_joint_false_positive_rows": control_fp if calib else "",
                "v1217_hard_support_joint_rows": hard_joint_by_source.get(source_run, 0),
                "v1217_release_label_rows": len(rel_rows),
                "v1217_release_hard_joint_rows": release_hard,
                "usable_for_promotion_visibility": usable,
                "expansion_status": ";".join(status_bits),
                "recommended_next_repair": (
                    "generate a new calibrated v12.16/v12.17 Line C repair source with independent "
                    "seeds/windows and zero random/control joint false positives; do not lower thresholds"
                ),
                "no_fake": 1,
                "no_proxy": 1,
                "cpu_offload_used": 0,
            }
        )
    write_csv_rows(out_dir / "v1218_line_t_support_expansion_audit.csv", audit_rows)
    calibrated_release_sources = [
        row
        for row in audit_rows
        if safe_int(row.get("v1217_source_calibration_pass"), 0) == 1 and safe_int(row.get("v1217_release_hard_joint_rows"), 0) > 0
    ]
    blocker_bits: list[str] = []
    if len(calibrated_release_sources) < 2:
        blocker_bits.append("no_second_calibrated_release_source")
    if support_concentrated:
        blocker_bits.append("hard_joint_support_concentrated_in_seed0_window10")
    if any(safe_int(row.get("random_joint_false_positive_rows"), 0) > 0 for row in audit_rows):
        blocker_bits.append("candidate_source_random_false_positives")
    if any("smoke_scope_only" in str(row.get("expansion_status", "")) for row in audit_rows):
        blocker_bits.append("smoke_source_not_official_independent_source")
    return {
        "line_t_support_expansion_audit_rows": len(audit_rows),
        "line_t_calibrated_release_source_count": len(calibrated_release_sources),
        "line_t_independent_source_runs_available_for_leaveout": len(calibrated_release_sources),
        "line_t_support_expansion_pass": int(len(calibrated_release_sources) >= 2 and not support_concentrated and hard_joint_support_rows > 0),
        "line_t_support_expansion_blocker": ";".join(blocker_bits),
    }


def write_hard_gate_margin_repair(source_dirs: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    thresholds = [-0.0100, -0.0125, -0.0150, -0.0175, -0.0200, -0.0250, -0.0300]
    rows: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        linec_path = Path(source_dir) / "v1216_linec_v2_sketch_targets.csv"
        source_rows = read_csv_rows(linec_path)
        if not source_rows:
            rows.append(
                {
                    "stage": "V1218_HARD_GATE_MARGIN_REPAIR",
                    "source_run": Path(source_dir).name,
                    "source_run_dir": rel(Path(source_dir)),
                    "threshold": "",
                    "status": "missing_or_empty_v1216_linec_v2_sketch_targets",
                    "uses_label_for_feature": 0,
                    "uses_ce_for_feature": 0,
                    "selection_for_promotion": 0,
                }
            )
            continue
        for threshold in thresholds:
            noise = [safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0) <= threshold for row in source_rows]
            reservoir = [safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= threshold for row in source_rows]
            joint = [n and r for n, r in zip(noise, reservoir)]
            hard_rows = [row for row, is_joint in zip(source_rows, joint) if is_joint]
            controls = [row for row in source_rows if str(row.get("method", "")).startswith(("U0", "U1"))]
            control_joint = [
                safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0) <= threshold
                and safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= threshold
                for row in controls
            ]
            noop_joint_fp = sum(
                1
                for row in source_rows
                if str(row.get("method", "")).startswith("U0")
                and safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0) <= threshold
                and safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= threshold
            )
            random_joint_fp = sum(
                1
                for row in source_rows
                if str(row.get("method", "")).startswith("U1")
                and safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0) <= threshold
                and safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= threshold
            )
            by_method: dict[str, int] = defaultdict(int)
            by_dataset: dict[str, int] = defaultdict(int)
            by_seed: dict[str, int] = defaultdict(int)
            by_window: dict[str, int] = defaultdict(int)
            for row in hard_rows:
                by_method[str(row.get("method", ""))] += 1
                by_dataset[str(row.get("dataset", ""))] += 1
                by_seed[str(row.get("seed", ""))] += 1
                by_window[str(row.get("window", ""))] += 1
            hard_joint_rows = len(hard_rows)
            noise_support = sum(1 for ok in noise if ok)
            reservoir_support = sum(1 for ok in reservoir if ok)
            rows_n = len(source_rows)
            control_rows = len(controls)
            control_joint_fp = sum(1 for ok in control_joint if ok)
            control_joint_fp_rate = control_joint_fp / max(1, control_rows)
            source_calibration_pass = int(
                noop_joint_fp == 0
                and random_joint_fp == 0
                and control_joint_fp_rate <= 0.01
                and (noise_support / max(1, rows_n)) >= 0.05
                and (reservoir_support / max(1, rows_n)) >= 0.05
            )
            concentrated = int(hard_joint_rows > 0 and any(max(d.values() or [0]) == hard_joint_rows for d in [by_method, by_dataset, by_seed, by_window]))
            rows.append(
                {
                    "stage": "V1218_HARD_GATE_MARGIN_REPAIR",
                    "source_run": Path(source_dir).name,
                    "source_run_dir": rel(Path(source_dir)),
                    "threshold": threshold,
                    "threshold_change": "raised_stricter_than_minus_0p01" if threshold < -0.01 else "official_minus_0p01_baseline",
                    "rows": rows_n,
                    "noise_hard_support_rows": noise_support,
                    "noise_hard_support_rate": noise_support / max(1, rows_n),
                    "reservoir_hard_support_rows": reservoir_support,
                    "reservoir_hard_support_rate": reservoir_support / max(1, rows_n),
                    "joint_hard_support_rows": hard_joint_rows,
                    "joint_hard_support_rate": hard_joint_rows / max(1, rows_n),
                    "noop_joint_false_positive_rows": noop_joint_fp,
                    "random_joint_false_positive_rows": random_joint_fp,
                    "control_joint_false_positive_rows": control_joint_fp,
                    "control_joint_false_positive_rate": control_joint_fp_rate,
                    "source_calibration_pass_at_margin": source_calibration_pass,
                    "margin_usable_release_source": int(source_calibration_pass == 1 and hard_joint_rows > 0 and concentrated == 0),
                    "support_concentrated_at_margin": concentrated,
                    "hard_support_by_method": dict(sorted(by_method.items())),
                    "hard_support_by_dataset": dict(sorted(by_dataset.items())),
                    "hard_support_by_seed": dict(sorted(by_seed.items())),
                    "hard_support_by_window": dict(sorted(by_window.items())),
                    "new_u10_u14_hard_support_rows": sum(count for method, count in by_method.items() if method.startswith(("U10", "U11", "U12", "U13", "U14"))),
                    "uses_label_for_feature": 0,
                    "uses_ce_for_feature": 0,
                    "selection_for_promotion": 0,
                    "repair_rationale": "stricter hard-release audit target margin; threshold is raised, never lowered; labels remain audit-only",
                }
            )
    write_csv_rows(out_dir / "v1218_hard_gate_margin_repair.csv", rows)
    usable = [row for row in rows if safe_int(row.get("margin_usable_release_source"), 0) == 1]
    best = max(usable, key=lambda row: (safe_int(row.get("joint_hard_support_rows"), 0), safe_int(row.get("new_u10_u14_hard_support_rows"), 0)), default={})
    summary_rows = [
        {
            "stage": "V1218_HARD_GATE_MARGIN_REPAIR_SUMMARY",
            "margin_source_count": len({str(row.get("source_run", "")) for row in rows if row.get("source_run")}),
            "margin_rows": len(rows),
            "usable_margin_rows": len(usable),
            "best_margin_source_run": best.get("source_run", ""),
            "best_margin_threshold": best.get("threshold", ""),
            "best_margin_joint_hard_support_rows": best.get("joint_hard_support_rows", ""),
            "best_margin_new_u10_u14_hard_support_rows": best.get("new_u10_u14_hard_support_rows", ""),
            "best_margin_random_joint_false_positive_rows": best.get("random_joint_false_positive_rows", ""),
            "best_margin_support_concentrated": best.get("support_concentrated_at_margin", ""),
            "margin_repair_pass": 0,
            "pass_reason": "diagnostic_only; margin sweep cannot open P3/P4 without strict heldout visibility pass",
        }
    ]
    write_csv_rows(out_dir / "v1218_hard_gate_margin_repair_summary.csv", summary_rows)
    return {
        "line_t_hard_gate_margin_repair_rows": len(rows),
        "line_t_hard_gate_margin_usable_rows": len(usable),
        "line_t_hard_gate_margin_best_source": best.get("source_run", ""),
        "line_t_hard_gate_margin_best_threshold": best.get("threshold", ""),
        "line_t_hard_gate_margin_best_joint_support": best.get("joint_hard_support_rows", ""),
        "line_t_hard_gate_margin_best_new_u10_u14_support": best.get("new_u10_u14_hard_support_rows", ""),
        "line_t_hard_gate_margin_repair_pass": 0,
    }


def write_hard_gate_bootstrap_repair(source_dirs: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    thresholds = [-0.0100, -0.0125, -0.0150, -0.0200, -0.0300]
    bootstrap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(121800)
    for source_dir in source_dirs:
        source_rows = read_csv_rows(Path(source_dir) / "v1216_linec_v2_sketch_targets.csv")
        if not source_rows:
            continue
        groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in source_rows:
            groups[(str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("window", "")))].append(row)
        group_keys = sorted(groups)
        if not group_keys:
            continue
        for threshold in thresholds:
            joint_values: list[int] = []
            new_values: list[int] = []
            random_fp_values: list[int] = []
            usable_values: list[int] = []
            for _ in range(128):
                sampled: list[dict[str, str]] = []
                picked = rng.integers(0, len(group_keys), size=len(group_keys))
                for idx in picked:
                    sampled.extend(groups[group_keys[int(idx)]])
                hard_rows = [
                    row
                    for row in sampled
                    if safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0) <= threshold
                    and safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= threshold
                ]
                random_fp = sum(1 for row in hard_rows if str(row.get("method", "")).startswith("U1"))
                noop_fp = sum(1 for row in hard_rows if str(row.get("method", "")).startswith("U0"))
                new_support = sum(1 for row in hard_rows if str(row.get("method", "")).startswith(("U10", "U11", "U12", "U13", "U14")))
                by_method: dict[str, int] = defaultdict(int)
                by_dataset: dict[str, int] = defaultdict(int)
                by_seed: dict[str, int] = defaultdict(int)
                by_window: dict[str, int] = defaultdict(int)
                for row in hard_rows:
                    by_method[str(row.get("method", ""))] += 1
                    by_dataset[str(row.get("dataset", ""))] += 1
                    by_seed[str(row.get("seed", ""))] += 1
                    by_window[str(row.get("window", ""))] += 1
                concentrated = int(len(hard_rows) > 0 and any(max(d.values() or [0]) == len(hard_rows) for d in [by_method, by_dataset, by_seed, by_window]))
                usable = int(len(hard_rows) > 0 and random_fp == 0 and noop_fp == 0 and concentrated == 0)
                joint_values.append(len(hard_rows))
                new_values.append(new_support)
                random_fp_values.append(random_fp)
                usable_values.append(usable)
            bootstrap_rows.append(
                {
                    "stage": "V1218_HARD_GATE_BOOTSTRAP_REPAIR",
                    "source_run": Path(source_dir).name,
                    "source_run_dir": rel(Path(source_dir)),
                    "threshold": threshold,
                    "bootstrap_replicates": 128,
                    "bootstrap_group_unit": "dataset_seed_window",
                    "joint_hard_support_mean": float(np.mean(joint_values)),
                    "joint_hard_support_min": int(np.min(joint_values)),
                    "joint_hard_support_max": int(np.max(joint_values)),
                    "new_u10_u14_support_mean": float(np.mean(new_values)),
                    "new_u10_u14_support_min": int(np.min(new_values)),
                    "new_u10_u14_support_max": int(np.max(new_values)),
                    "random_joint_fp_mean": float(np.mean(random_fp_values)),
                    "random_joint_fp_min": int(np.min(random_fp_values)),
                    "random_joint_fp_max": int(np.max(random_fp_values)),
                    "usable_bootstrap_rate": float(np.mean(usable_values)),
                    "bootstrap_repair_pass": 0,
                    "uses_label_for_feature": 0,
                    "uses_ce_for_feature": 0,
                    "selection_for_promotion": 0,
                    "repair_rationale": "bootstrap stability audit over stricter hard-gate margins; diagnostic only",
                }
            )
    write_csv_rows(out_dir / "v1218_hard_gate_bootstrap_repair.csv", bootstrap_rows)
    useful_new = [
        row
        for row in bootstrap_rows
        if safe_float(row.get("new_u10_u14_support_mean"), 0.0) > 0.0
        and safe_float(row.get("usable_bootstrap_rate"), 0.0) > 0.0
    ]
    best = max(
        useful_new,
        key=lambda row: (safe_float(row.get("usable_bootstrap_rate"), 0.0), safe_float(row.get("new_u10_u14_support_mean"), 0.0)),
        default={},
    )
    summary_rows = [
        {
            "stage": "V1218_HARD_GATE_BOOTSTRAP_REPAIR_SUMMARY",
            "bootstrap_rows": len(bootstrap_rows),
            "bootstrap_source_count": len({str(row.get("source_run", "")) for row in bootstrap_rows}),
            "bootstrap_replicates_per_row": 128,
            "u10_u14_rows_with_positive_usable_rate": len(useful_new),
            "best_bootstrap_source_run": best.get("source_run", ""),
            "best_bootstrap_threshold": best.get("threshold", ""),
            "best_bootstrap_usable_rate": best.get("usable_bootstrap_rate", ""),
            "best_bootstrap_new_u10_u14_support_mean": best.get("new_u10_u14_support_mean", ""),
            "bootstrap_repair_pass": 0,
            "pass_reason": "diagnostic_only; no bootstrap-stable U10-U14 source can open P3/P4 without strict visibility pass",
        }
    ]
    write_csv_rows(out_dir / "v1218_hard_gate_bootstrap_repair_summary.csv", summary_rows)
    return {
        "line_t_hard_gate_bootstrap_repair_rows": len(bootstrap_rows),
        "line_t_hard_gate_bootstrap_u10_u14_positive_usable_rows": len(useful_new),
        "line_t_hard_gate_bootstrap_best_source": best.get("source_run", ""),
        "line_t_hard_gate_bootstrap_best_threshold": best.get("threshold", ""),
        "line_t_hard_gate_bootstrap_best_usable_rate": best.get("usable_bootstrap_rate", ""),
        "line_t_hard_gate_bootstrap_repair_pass": 0,
    }


def run_strict_visibility_v2(v1217_dir: Path, v1216_repair_dir: Path, out_dir: Path) -> dict[str, Any]:
    labels = read_csv_rows(v1217_dir / "v1217_release_labels_audit_only.csv")
    by_seed: dict[str, int] = defaultdict(int)
    by_window: dict[str, int] = defaultdict(int)
    by_method: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    total_joint = 0
    for row in labels:
        if safe_int(row.get("hard_joint_release"), 0):
            total_joint += 1
            by_seed[str(row.get("seed", ""))] += 1
            by_window[str(row.get("window", ""))] += 1
            by_method[str(row.get("method", row.get("candidate_probe_id", "")))] += 1
            by_source[str(row.get("source_run", ""))] += 1
    concentrated = int(total_joint > 0 and any(max(d.values() or [0]) == total_joint for d in [by_seed, by_window, by_method, by_source]))
    support_rows = [
        {
            "stage": "V1218_SUPPORT_CONCENTRATION",
            "hard_joint_total_rows": total_joint,
            "support_concentration_seed": dict(sorted(by_seed.items())),
            "support_concentration_window": dict(sorted(by_window.items())),
            "support_concentration_method": dict(sorted(by_method.items())),
            "support_concentration_source": dict(sorted(by_source.items())),
            "support_concentrated": concentrated,
        }
    ]
    write_csv_rows(out_dir / "v1218_support_concentration.csv", support_rows)
    strict_long, repair_attempts = build_strict_v1216_long_rows(v1216_repair_dir, v1217_dir)
    if not strict_long:
        strict_long = [
            {
                "stage": "V1218_STRICT_VISIBILITY_FEATURES",
                "feature_set": "T1 strict_precommit_unlabeled_features",
                "status": "no_strict_v1216_feature_rows_available",
                "strict_feature_rows": 0,
                "reason": "No row satisfied loss_agnostic_direction=1 and label/CE/dataset commit flags all zero.",
            }
        ]
    write_csv_rows(out_dir / "v1218_strict_visibility_features.csv", strict_long)
    write_csv_rows(out_dir / "v1218_strict_visibility_repair_attempts.csv", repair_attempts)
    score_rows, leaveout_rows, score_summary = score_strict_visibility(
        [row for row in strict_long if row.get("row_id")],
        concentrated,
    )
    family_ablation_summary = write_strict_visibility_family_ablation(
        [row for row in strict_long if row.get("row_id")],
        concentrated,
        out_dir,
    )
    composite_repair_summary = write_strict_visibility_composite_repair(
        [row for row in strict_long if row.get("row_id")],
        concentrated,
        out_dir,
    )
    context_rank_repair_summary = write_strict_visibility_context_rank_repair(
        [row for row in strict_long if row.get("row_id")],
        concentrated,
        out_dir,
    )
    write_csv_rows(out_dir / "v1218_strict_visibility_scores.csv", score_rows)
    write_csv_rows(out_dir / "v1218_strict_visibility_leaveout.csv", leaveout_rows)
    support_expansion_summary = write_line_t_support_expansion_audit(
        v1217_dir,
        v1216_repair_dir,
        out_dir,
        concentrated,
        total_joint,
    )
    score_summary.update({"support_concentrated": concentrated, "hard_joint_support_rows": total_joint})
    score_summary.update(support_expansion_summary)
    score_summary.update(family_ablation_summary)
    score_summary.update(composite_repair_summary)
    score_summary.update(context_rank_repair_summary)
    return score_summary


def run_p3_p4_gates(out_dir: Path, strict_summary: Mapping[str, Any]) -> dict[str, Any]:
    p3_rows = [
        {
            "stage": "V1218_P3_MATCHED_CONTROLS_STATUS",
            "line_t_strict_visibility_pass": safe_int(strict_summary.get("strict_visibility_pass"), 0),
            "p3_open": 0,
            "matched_controls_generated": 0,
            "required_controls": "NoOp,RandomMatchedNorm,AdamWParallel,SNR-only",
            "not_run_reason": "Line T strict label-free visibility failed; matched-control actuator dictionary not generated",
            "promotion_allowed": 0,
        }
    ]
    p4_rows = [
        {
            "stage": "V1218_P4_GATE_STATUS",
            "p4_open": 0,
            "line_r_pass": "",
            "line_a_current_b320_pass": "",
            "line_c_calibration_pass": "",
            "line_t_hard_visibility_pass": safe_int(strict_summary.get("strict_visibility_pass"), 0),
            "line_i_response_pass": 0,
            "not_run_reason": "P4 requires Line R/A/C/T/I pass; Line T and Line I are closed",
            "promotion_allowed": 0,
        }
    ]
    write_csv_rows(out_dir / "v1218_p3_matched_controls_status.csv", p3_rows)
    write_csv_rows(out_dir / "v1218_p4_gate_status.csv", p4_rows)
    return {"p3_open": 0, "line_i_response_pass": 0, "p4_open": 0, "p4_pass": 0}


def run_classic_monitor(v1217_dir: Path, out_dir: Path) -> dict[str, Any]:
    src = read_csv_rows(v1217_dir / "v1217_classic_family_status.csv")
    rows: list[dict[str, Any]] = []
    for row in src:
        out = dict(row)
        out["stage"] = "V1218_CLASSIC_FAMILY_STATUS"
        out["new_hypothesis_implemented"] = 0
        out["status_unchanged"] = 1
        out["blocker"] = row.get("v1216_status", row.get("fail_reason", ""))
        out["not_rerun_reason"] = "Line D monitor only; no new classic-family hypothesis introduced in v12.18"
        rows.append(out)
    write_csv_rows(out_dir / "v1218_classic_family_status.csv", rows)
    return {"line_d_rows": len(rows), "line_d_new_hypothesis_implemented": 0}


def decide_route(summary: Mapping[str, Any]) -> dict[str, Any]:
    route = "R3-StrictLossAgnosticObservablesNotVisible"
    fail: list[str] = []
    if safe_int(summary.get("all_required_files_present"), 0) != 1 or safe_int(summary.get("core_code_review_pass"), 0) != 1:
        route = "R0-CodeAuditIncomplete"
        fail.append("code_audit_incomplete")
    elif safe_int(summary.get("b320_current_anchor_pass"), 0) != 1:
        route = "R1-B320AnchorRegression"
        fail.append("b320_anchor_regression")
    elif safe_int(summary.get("b320_current_uses_label_init"), 0) == 1 and (
        safe_int(summary.get("b320_label_free_ablation_available_count"), 0) == 0
        or safe_int(summary.get("b320_label_free_anchor_budget_strong_pass"), 0) != 1
    ):
        route = "R2-B320LabelInitDependenceDetected"
        if safe_int(summary.get("b320_label_free_anchor_budget_ablation_available_count"), 0) > 0 and safe_int(summary.get("b320_label_free_ablation_available_count"), 0) == 0:
            fail.append("current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official")
            if safe_int(summary.get("b320_label_free_anchor_budget_a0_reproduces_locked_anchor"), 0) != 1:
                fail.append("label_free_anchor_budget_A0_does_not_reproduce_locked_anchor")
            if safe_int(summary.get("b320_label_free_anchor_budget_underperforms_labelinit"), 0) == 1:
                fail.append("label_free_anchor_budget_underperforms_labelInit")
            if safe_int(summary.get("b320_label_free_anchor_budget_linec_nontearing_available"), 0) != 1:
                fail.append("label_free_anchor_budget_LineC_nontearing_not_measured")
            elif safe_int(summary.get("b320_label_free_anchor_budget_linec_nontearing_pass"), 0) != 1:
                fail.append("label_free_anchor_budget_LineC_nontearing_failed")
        elif safe_int(summary.get("b320_label_free_ablation_available_count"), 0) > 0:
            fail.append("current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_ablation_failed_strong_pass")
            if safe_int(summary.get("b320_label_free_anchor_budget_underperforms_labelinit"), 0) == 1:
                fail.append("label_free_anchor_budget_underperforms_labelInit")
            if safe_int(summary.get("b320_label_free_anchor_budget_linec_nontearing_available"), 0) != 1:
                fail.append("label_free_anchor_budget_LineC_nontearing_not_measured")
            elif safe_int(summary.get("b320_label_free_anchor_budget_linec_nontearing_pass"), 0) != 1:
                fail.append("label_free_anchor_budget_LineC_nontearing_failed")
        elif safe_int(summary.get("b320_label_free_smoke_ablation_available_count"), 0) > 0:
            fail.append("current_B320_uses_label_informed_trainprobe_init;label_free_official_ablation_missing")
            if safe_int(summary.get("b320_label_free_smoke_underperforms_labelinit"), 0) == 1:
                fail.append("label_free_smoke_underperforms_labelInit")
        else:
            fail.append("current_B320_uses_label_informed_trainprobe_init;label_free_ablation_missing")
        if safe_int(summary.get("strict_visibility_pass"), 0) != 1:
            fail.append("strict_loss_agnostic_observables_not_visible")
    elif safe_int(summary.get("strict_visibility_pass"), 0) != 1:
        route = "R3-StrictLossAgnosticObservablesNotVisible"
        fail.append("strict_loss_agnostic_observables_not_visible")
    elif safe_int(summary.get("line_i_response_pass"), 0) != 1:
        route = "R4-ActuatorExecutorNotControlResistant"
        fail.append("actuator_executor_not_control_resistant")
    elif safe_int(summary.get("p4_open"), 0) == 1 and safe_int(summary.get("p4_pass"), 0) != 1:
        route = "R5-P3SurvivorFoundP4Opened"
    elif safe_int(summary.get("p4_pass"), 0) == 1:
        route = "R6-P4FunctionalShortRunSuccess"
    else:
        route = "R7-DowngradeToLabelFreeGeometryMaintenance"
    payload = dict(summary)
    payload.update(
        {
            "stage": "V1218_ROUTE_DECISION",
            "generated_at": now_iso(),
            "route": route,
            "fail_reason": ";".join(fail),
            "promotion_allowed": 0,
            "new_mechanism_success_claimed": 0,
            "offline_postprocess_analysis": 1,
            "P4_open": safe_int(summary.get("p4_open"), 0),
        }
    )
    return payload


def write_hash_manifest(out_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1218_hash_manifest.json":
            hashes[path.name] = sha256_file(path)
    write_json(out_dir / "v1218_hash_manifest.json", hashes)
    return hashes


def package_code_review_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "v1218_code_review_packet.zip"
    code_paths = [REPO_ROOT / file_name for file_name in all_required_code_files()]
    review_artifacts = [out_dir / name for name in REQUIRED_ARTIFACTS if name != "v1218_code_review_packet.zip"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(set(code_paths + [PLAN_DOC])):
            if path.exists() and path.is_file():
                zf.write(path, arcname=f"code/{rel(path)}")
        for path in review_artifacts:
            if path.exists() and path.is_file():
                zf.write(path, arcname=f"review_artifacts/{path.name}")
    return zip_path


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    v1217_dir = Path(args.v1217_dir).resolve()
    v1216_repair_dir = Path(args.v1216_repair_dir).resolve()
    margin_source_dirs = [Path(item).resolve() for item in (args.margin_source_run or [])]
    if not margin_source_dirs:
        margin_source_dirs = [v1216_repair_dir]
    ensure_dir(out_dir)
    dep_summary = run_transitive_dependency_manifest(out_dir)
    review_summary = build_core_code_review(out_dir)
    b320_summary = run_b320_label_init_audit(v1217_dir, out_dir)
    linec_summary = run_linec_audit(v1217_dir, out_dir)
    provenance_summary = run_feature_provenance_audit(v1217_dir, v1216_repair_dir, out_dir)
    strict_summary = run_strict_visibility_v2(v1217_dir, v1216_repair_dir, out_dir)
    margin_summary = write_hard_gate_margin_repair(margin_source_dirs, out_dir)
    bootstrap_summary = write_hard_gate_bootstrap_repair(margin_source_dirs, out_dir)
    p3p4_summary = run_p3_p4_gates(out_dir, strict_summary)
    classic_summary = run_classic_monitor(v1217_dir, out_dir)
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "v1217_source_dir": rel(v1217_dir),
        "v1216_repair_source_dir": rel(v1216_repair_dir),
        "no_fake_proxy_cpu": 1,
        "B320_anchor_locked": 1,
    }
    for chunk in [dep_summary, review_summary, b320_summary, linec_summary, provenance_summary, strict_summary, margin_summary, bootstrap_summary, p3p4_summary, classic_summary]:
        summary.update(chunk)
    route = decide_route(summary)
    write_json(out_dir / "v1218_route_decision.json", route)
    package_code_review_zip(out_dir)
    hashes = write_hash_manifest(out_dir)
    route["code_review_packet_zip"] = rel(out_dir / "v1218_code_review_packet.zip")
    route["code_review_packet_zip_sha256"] = sha256_file(out_dir / "v1218_code_review_packet.zip")
    route["hash_manifest_entries"] = len(hashes)
    write_json(out_dir / "v1218_route_decision.json", route)
    write_hash_manifest(out_dir)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="official_from_v1217_v1216_artifacts")
    parser.add_argument("--v1217-dir", default=str(DEFAULT_V1217_DIR))
    parser.add_argument("--v1216-repair-dir", default=str(DEFAULT_V1216_REPAIR_DIR))
    parser.add_argument(
        "--margin-source-run",
        action="append",
        default=None,
        help="v12.16 source run directory for stricter hard-gate margin audit. Repeat to include repair sources.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    result = run_main(parsed)
    print(json.dumps({"out_dir": str(Path(parsed.out_dir).resolve()), "route": result.get("route"), "p4_open": result.get("p4_open", result.get("P4_open")), "promotion_allowed": result.get("promotion_allowed")}, indent=2, ensure_ascii=False))
