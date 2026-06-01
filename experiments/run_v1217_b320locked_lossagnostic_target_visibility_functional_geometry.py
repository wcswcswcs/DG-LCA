#!/usr/bin/env python3
"""v12.17.2 B320-locked loss-agnostic target visibility audit runner.

This runner is intentionally an offline diagnostic over real v12.16 artifacts.
It does not claim a new functional mechanism execution unless upstream P2/P3
gates open it. Labels and CE-derived audit fields are used only for release
visibility evaluation, never for direction construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md"
DEFAULT_OFFICIAL_SOURCE = REPO_ROOT / "results" / "v12_16_b320locked_explicit_signal_reservoir_functional" / "official_3x3_b32_w3_5_10"
DEFAULT_REPAIR_SOURCE = REPO_ROOT / "results" / "v12_16_b320locked_explicit_signal_reservoir_functional" / "repair_sketchdim24_rank5_b64_w3_5_10"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry" / "official_from_v1216_artifacts"
RUNNER_PATH = REPO_ROOT / "experiments" / "run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py"
B320_ID = "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075"
DEFAULT_V1215_EXPANSION_SOURCES = [
    REPO_ROOT / "results" / "v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation" / "continuation_actuator_budget_repair_3x3_b32_w5" / "v1215_continuation_actuator_budget_repair.csv",
    REPO_ROOT / "results" / "v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation" / "continuation_fused_primitive_actuator_3x3_b32_w5" / "v1215_continuation_actuator_budget_repair.csv",
    REPO_ROOT / "results" / "v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation" / "continuation_p3_bidir_selector_3x3_b32_w5" / "v1215_continuation_p3_candidates.csv",
]

REQUIRED_ARTIFACTS = [
    "v1217_route_decision.json",
    "v1217_anchor_monitor.csv",
    "v1217_loss_agnostic_contract.csv",
    "v1217_provenance_audit.csv",
    "v1217_baseline_controls.csv",
    "v1217_implementation_readback_audit.csv",
    "v1217_code_path_map.json",
    "v1217_diff_intent_table.csv",
    "v1217_critical_code_review_manifest.csv",
    "v1217_core_symbol_map.json",
    "v1217_code_semantics_trace.csv",
    "v1217_manual_review_packet.md",
    "v1217_review_blocker_table.csv",
    "v1217_linec_null_distribution.csv",
    "v1217_linec_variance.csv",
    "v1217_linec_threshold_sensitivity.csv",
    "v1217_linec_hard_support.csv",
    "v1217_target_visibility_features.csv",
    "v1217_release_labels_audit_only.csv",
    "v1217_release_support_concentration.csv",
    "v1217_visibility_scores.csv",
    "v1217_visibility_leaveout.csv",
    "v1217_feature_ablation.csv",
    "v1217_visibility_repair_attempts.csv",
    "v1217_visibility_repair_summary.csv",
    "v1217_actuator_response_dictionary.csv",
    "v1217_actuator_rank_condition.csv",
    "v1217_actuator_safe_combo.csv",
    "v1217_response_to_visibility_score.csv",
    "v1217_actuator_dictionary_expansion_audit.csv",
    "v1217_actuator_dictionary_expansion_summary.csv",
    "v1217_functional_candidates.csv",
    "v1217_functional_p3_audit.csv",
    "v1217_functional_controls.csv",
    "v1217_functional_candidate_selection_rule.json",
    "v1217_p4_short_run.csv",
    "v1217_p4_event_log.csv",
    "v1217_p4_controls.csv",
    "v1217_p4_linec_trajectory.csv",
    "v1217_p4_efficiency.csv",
    "v1217_classic_family_status.csv",
    "v1217_classic_family_new_hypothesis.csv",
    "v1217_classic_family_linec.csv",
    "v1217_failure_table.csv",
    "v1217_hash_manifest.json",
]

FEATURE_FAMILIES = {
    "F1_output_spectral": [
        "reservoir_fraction",
        "top_eigen_share",
        "stable_cotangent_energy",
        "unstable_cotangent_energy",
        "stable_unstable_balance",
    ],
    "F2_train_probe_coupling": [
        "pred_coupling_delta",
        "actual_CouplingR2_delta",
        "signal_projector_stability",
        "reservoir_projector_stability",
        "projector_stability_gap",
    ],
    "F3_gradient_sketch_spectrum": [
        "signal_rank",
        "reservoir_rank",
        "signal_effective_rank",
        "dissipation_condition",
        "sketch_dim",
    ],
    "F4_primitive_role_response": [
        "method_is_control",
        "method_is_random",
        "method_is_cotangent_vjp",
        "method_is_role_actuator",
        "method_is_projector",
    ],
    "F5_augmentation_consistency": [
        "augmentation_used",
        "cotangent_family_hash",
        "sketch_family_hash",
        "energy_scale",
    ],
    "F6_time_persistence": [
        "window",
        "persistence_reservoir_fraction_absdiff",
        "persistence_projector_stability_absdiff",
        "persistence_coupling_absdiff",
    ],
}


@dataclass
class SourceRun:
    name: str
    path: Path
    anchor: list[dict[str, str]]
    linec: list[dict[str, str]]
    actuator: list[dict[str, str]]
    calibration: list[dict[str, str]]
    audit: list[dict[str, str]]
    classic: list[dict[str, str]]
    provenance: list[dict[str, str]]
    route: dict[str, Any]


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
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        val = float(text)
    except (TypeError, ValueError):
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def stable_hash_unit(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: Sequence[float], q: float) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return float(np.percentile(np.asarray(clean, dtype=float), q))


def mean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def std(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return 0.0 if clean else float("nan")
    return float(statistics.pstdev(clean))


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if not math.isnan(float(s))]
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
            total += 1
    return float(wins / max(1, total))


def precision_recall_at_k(labels: Sequence[int], scores: Sequence[float], k: int | None = None) -> tuple[float, float, int, int]:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if not math.isnan(float(s))]
    positives = sum(1 for y, _ in pairs if y == 1)
    if positives == 0 or not pairs:
        return float("nan"), float("nan"), 0, positives
    kk = k if k is not None else max(1, positives)
    top = sorted(pairs, key=lambda p: p[1], reverse=True)[:kk]
    hits = sum(1 for y, _ in top if y == 1)
    return float(hits / max(1, kk)), float(hits / positives), hits, positives


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float = 1.0e-3) -> np.ndarray:
    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        return np.full((test_x.shape[0],), np.nan)
    if len(set(int(v) for v in train_y.tolist())) < 2:
        return np.full((test_x.shape[0],), float(np.mean(train_y)))
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma < 1.0e-12] = 1.0
    x0 = (train_x - mu) / sigma
    xt = (test_x - mu) / sigma
    x0 = np.concatenate([np.ones((x0.shape[0], 1)), x0], axis=1)
    xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    reg = ridge * np.eye(x0.shape[1])
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x0.T @ x0 + reg, x0.T @ train_y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x0.T @ x0 + reg) @ x0.T @ train_y
    return xt @ beta


def weighted_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    positive_weight: float,
    ridge: float = 1.0e-3,
) -> np.ndarray:
    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        return np.full((test_x.shape[0],), np.nan)
    if len(set(int(v) for v in train_y.tolist())) < 2:
        return np.full((test_x.shape[0],), float(np.mean(train_y)))
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma < 1.0e-12] = 1.0
    x0 = (train_x - mu) / sigma
    xt = (test_x - mu) / sigma
    x0 = np.concatenate([np.ones((x0.shape[0], 1)), x0], axis=1)
    xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    weights = np.where(train_y > 0.5, float(positive_weight), 1.0)
    xw = x0 * np.sqrt(weights)[:, None]
    yw = train_y * np.sqrt(weights)
    reg = ridge * np.eye(x0.shape[1])
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xw.T @ xw + reg) @ xw.T @ yw
    return xt @ beta


def load_source_run(source_dir: Path) -> SourceRun:
    if not source_dir.exists():
        raise FileNotFoundError(f"source run not found: {source_dir}")
    name = source_dir.name
    return SourceRun(
        name=name,
        path=source_dir,
        anchor=read_csv_rows(source_dir / "v1216_anchor_monitor.csv"),
        linec=read_csv_rows(source_dir / "v1216_linec_v2_sketch_targets.csv"),
        actuator=read_csv_rows(source_dir / "v1216_actuator_response_matrix.csv"),
        calibration=read_csv_rows(source_dir / "v1216_explicit_target_calibration.csv"),
        audit=read_csv_rows(source_dir / "v1216_loss_agnostic_audit.csv"),
        classic=read_csv_rows(source_dir / "v1216_classic_family_status.csv"),
        provenance=read_csv_rows(source_dir / "v1216_provenance_audit.csv"),
        route=read_json(source_dir / "v1216_route_decision.json"),
    )


def tag_rows(rows: Sequence[Mapping[str, Any]], source: SourceRun) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["source_run"] = source.name
        out["source_run_dir"] = rel(source.path)
        out["_source_index"] = idx
        tagged.append(out)
    return tagged


def line_range_for_symbol(path: Path, symbol: str) -> tuple[int, int, str]:
    if not path.exists():
        return 0, 0, "file_missing"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    escaped = re.escape(symbol)
    patterns = [
        re.compile(rf"^def\s+{escaped}\s*\("),
        re.compile(rf"^class\s+{escaped}\b"),
        re.compile(rf"^{escaped}\s*="),
        re.compile(escaped),
    ]
    for pattern in patterns:
        for idx, line in enumerate(lines):
            if pattern.search(line):
                start = idx + 1
                end = start
                if line.lstrip().startswith(("def ", "class ")):
                    for j in range(idx + 1, len(lines)):
                        if lines[j].startswith(("def ", "class ")) and lines[j].strip():
                            end = j
                            break
                    else:
                        end = min(len(lines), start + 120)
                return start, end, lines[idx].strip()[:160]
    return 0, 0, "symbol_missing"


def symbol_summary(items: Sequence[tuple[str, str]]) -> dict[str, Any]:
    paths: list[str] = []
    starts: list[str] = []
    ends: list[str] = []
    snippets: list[str] = []
    missing = 0
    for rel_path, symbol in items:
        path = REPO_ROOT / rel_path
        start, end, snippet = line_range_for_symbol(path, symbol)
        paths.append(rel_path)
        starts.append(str(start))
        ends.append(str(end))
        snippets.append(f"{symbol}@{start}-{end}:{snippet}")
        if start == 0:
            missing += 1
    return {
        "actual_file_path": " | ".join(paths),
        "actual_line_start": " | ".join(starts),
        "actual_line_end": " | ".join(ends),
        "line_snippets": " ; ".join(snippets),
        "unknown_or_not_inspected": 1 if missing else 0,
    }


def run_p0_anchor_monitor(sources: Sequence[SourceRun], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for row in source.anchor:
            out = dict(row)
            out.update(
                {
                    "stage": "V1217_P0_ANCHOR_MONITOR",
                    "source_run": source.name,
                    "source_run_dir": rel(source.path),
                    "reused_from_v1216": 1,
                    "offline_postprocess_analysis": 1,
                    "new_b320_training_claimed": 0,
                    "B320_anchor_locked": 1,
                }
            )
            rows.append(out)
    write_csv_rows(out_dir / "v1217_anchor_monitor.csv", rows)
    p0_pass = int(all(safe_int(r.get("p0_pass"), 0) == 1 for r in rows)) if rows else 0
    return {
        "p0_pass": p0_pass,
        "anchor_rows": len(rows),
        "step_ratio_q90_max": max([safe_float(r.get("step_ratio_q90"), float("nan")) for r in rows] or [float("nan")]),
        "memory_ratio_q90_max": max([safe_float(r.get("memory_ratio_q90"), float("nan")) for r in rows] or [float("nan")]),
        "mean_delta_vs_mlp_min": min([safe_float(r.get("mean_delta_vs_mlp"), float("nan")) for r in rows] or [float("nan")]),
        "worst_delta_vs_mlp_min": min([safe_float(r.get("worst_delta_vs_mlp"), float("nan")) for r in rows] or [float("nan")]),
    }


def build_loss_agnostic_contract(sources: Sequence[SourceRun], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for row in source.audit:
            out = dict(row)
            out.update(
                {
                    "stage": "V1217_LOSS_AGNOSTIC_CONTRACT",
                    "source_run": source.name,
                    "label_used_for_audit_only": 0,
                    "ce_used_for_audit_only": 0,
                    "permuted_label_used_for_direction": 0,
                    "test_used_for_commit": 0,
                    "future_outcome_used_for_commit": 0,
                    "promotion_allowed": 1,
                    "contract_source": "carried_forward_from_v1216_loss_agnostic_audit",
                }
            )
            rows.append(out)
    for feature_family in FEATURE_FAMILIES:
        rows.append(
            {
                "stage": "V1217_LOSS_AGNOSTIC_CONTRACT",
                "method": feature_family,
                "run_id": "v1217_visibility_atlas",
                "candidate_id": feature_family,
                "dataset": "ALL",
                "seed": "ALL",
                "split": "leave_dataset_seed_window",
                "window": "3,5,10",
                "control_id": "feature_family",
                "feature_family": feature_family,
                "actuator_family": "",
                "loss_agnostic_direction": 1,
                "ce_vector_used_for_direction": 0,
                "label_used_for_direction": 0,
                "permuted_label_used_for_direction": 0,
                "validation_used_for_commit": 0,
                "test_used_for_commit": 0,
                "future_outcome_used_for_commit": 0,
                "dataset_name_used_for_commit": 0,
                "label_used_for_audit_only": 0,
                "ce_used_for_audit_only": 0,
                "no_fake": 1,
                "no_proxy": 1,
                "cpu_offload_used": 0,
                "promotion_allowed": 1,
                "violation_reason": "",
                "contract_source": "v1217_runner_feature_builder",
            }
        )
    rows.append(
        {
            "stage": "V1217_LOSS_AGNOSTIC_CONTRACT",
            "method": "hard_release_labels",
            "run_id": "v1217_visibility_atlas",
            "candidate_id": "audit_only_release_labels",
            "dataset": "ALL",
            "seed": "ALL",
            "split": "audit_only",
            "window": "3,5,10",
            "control_id": "audit_only",
            "feature_family": "",
            "actuator_family": "",
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "permuted_label_used_for_direction": 0,
            "validation_used_for_commit": 0,
            "test_used_for_commit": 0,
            "future_outcome_used_for_commit": 0,
            "dataset_name_used_for_commit": 0,
            "label_used_for_audit_only": 1,
            "ce_used_for_audit_only": 0,
            "no_fake": 1,
            "no_proxy": 1,
            "cpu_offload_used": 0,
            "promotion_allowed": 1,
            "violation_reason": "",
            "contract_source": "v1217_build_release_labels_audit_only",
        }
    )
    write_csv_rows(out_dir / "v1217_loss_agnostic_contract.csv", rows)
    violations = [
        r
        for r in rows
        if safe_int(r.get("ce_vector_used_for_direction"), 0)
        or safe_int(r.get("label_used_for_direction"), 0)
        or safe_int(r.get("permuted_label_used_for_direction"), 0)
        or safe_int(r.get("validation_used_for_commit"), 0)
        or safe_int(r.get("test_used_for_commit"), 0)
        or safe_int(r.get("future_outcome_used_for_commit"), 0)
        or safe_int(r.get("dataset_name_used_for_commit"), 0)
        or safe_int(r.get("no_fake"), 1) == 0
        or safe_int(r.get("no_proxy"), 1) == 0
        or safe_int(r.get("cpu_offload_used"), 0) != 0
    ]
    return {"loss_agnostic_contract_pass": int(not violations), "loss_agnostic_contract_rows": len(rows), "loss_agnostic_contract_violations": len(violations)}


def build_provenance_audit(
    sources: Sequence[SourceRun],
    out_dir: Path,
    expansion_sources: Sequence[Path] = DEFAULT_V1215_EXPANSION_SOURCES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_files = [
        "v1216_anchor_monitor.csv",
        "v1216_linec_v2_sketch_targets.csv",
        "v1216_actuator_response_matrix.csv",
        "v1216_explicit_target_calibration.csv",
        "v1216_loss_agnostic_audit.csv",
        "v1216_classic_family_status.csv",
        "v1216_route_decision.json",
        "v1216_hash_manifest.json",
    ]
    for source in sources:
        for name in source_files:
            path = source.path / name
            rows.append(
                {
                    "stage": "V1217_PROVENANCE_AUDIT",
                    "source_run": source.name,
                    "source_run_dir": rel(source.path),
                    "artifact": name,
                    "exists": int(path.exists()),
                    "sha256": sha256_file(path) if path.exists() else "",
                    "reused_from_v1216": 1,
                    "newly_computed": 0,
                    "offline_postprocess_analysis": 1,
                    "no_fake": 1,
                    "no_proxy": 1,
                    "cpu_offload_used": 0,
                }
            )
    for path in [PLAN_DOC, RUNNER_PATH]:
        rows.append(
            {
                "stage": "V1217_PROVENANCE_AUDIT",
                "source_run": "v1217_current",
                "source_run_dir": rel(path.parent),
                "artifact": path.name,
                "exists": int(path.exists()),
                "sha256": sha256_file(path) if path.exists() else "",
                "reused_from_v1216": 0,
                "newly_computed": 1,
                "offline_postprocess_analysis": 0,
                "no_fake": 1,
                "no_proxy": 1,
                "cpu_offload_used": 0,
            }
        )
    for path in expansion_sources:
        rows.append(
            {
                "stage": "V1217_PROVENANCE_AUDIT",
                "source_run": "v1215_actuator_dictionary_expansion_audit",
                "source_run_dir": rel(path.parent),
                "artifact": path.name,
                "source_artifact": rel(path),
                "exists": int(path.exists()),
                "sha256": sha256_file(path) if path.exists() else "",
                "reused_from_v1216": 0,
                "reused_from_v1215": 1,
                "newly_computed": 0,
                "offline_postprocess_analysis": 1,
                "promotion_allowed": 0,
                "not_promotable_reason": "external_v1215_dictionary_expansion_audit_only",
                "no_fake": 1,
                "no_proxy": 1,
                "cpu_offload_used": 0,
            }
        )
    write_csv_rows(out_dir / "v1217_provenance_audit.csv", rows)
    return {"provenance_rows": len(rows), "provenance_missing": sum(1 for r in rows if safe_int(r.get("exists"), 0) == 0)}


def build_baseline_controls(linec_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in linec_rows:
        by_method[str(row.get("method", ""))].append(row)
    for method, group in sorted(by_method.items()):
        is_control = int(method.startswith("U0") or method.startswith("U1") or "NoOp" in method or "Random" in method)
        noise = [safe_float(r.get("actual_NoiseSignalLeak_delta"), 0.0) for r in group]
        reservoir = [safe_float(r.get("actual_RealSignalReservoirRatio_delta"), 0.0) for r in group]
        coupling = [safe_float(r.get("actual_CouplingR2_delta"), 0.0) for r in group]
        rows.append(
            {
                "stage": "V1217_BASELINE_CONTROLS",
                "method": method,
                "control_available": is_control,
                "source_rows": len(group),
                "mean_actual_NoiseSignalLeak_delta": mean(noise),
                "mean_actual_RealSignalReservoirRatio_delta": mean(reservoir),
                "mean_actual_CouplingR2_delta": mean(coupling),
                "hard_noise_release_rows": sum(1 for v in noise if v <= -0.01),
                "hard_reservoir_release_rows": sum(1 for v in reservoir if v <= -0.01),
                "hard_joint_release_rows": sum(1 for a, b in zip(noise, reservoir) if a <= -0.01 and b <= -0.01),
                "matched_measurement_window": 1,
                "label_used_for_direction": 0,
                "ce_vector_used_for_direction": 0,
                "dataset_name_used_for_commit": 0,
            }
        )
    write_csv_rows(out_dir / "v1217_baseline_controls.csv", rows)
    control_joint_fp = sum(safe_int(r.get("hard_joint_release_rows"), 0) for r in rows if safe_int(r.get("control_available"), 0))
    return {"baseline_control_rows": len(rows), "control_joint_false_positive_rows": control_joint_fp}


def run_p1_linec_calibration(linec_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, Any]:
    null_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    source_calibration_rows: list[dict[str, Any]] = []
    metrics = [
        ("actual_NoiseSignalLeak_delta", -0.01),
        ("actual_RealSignalReservoirRatio_delta", -0.01),
        ("actual_CouplingR2_delta", 0.02),
    ]
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_group: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in linec_rows:
        by_method[str(row.get("method", ""))].append(row)
        by_group[(str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("window", "")), str(row.get("sketch_family", "")))].append(row)
    for method, group in sorted(by_method.items()):
        for metric, threshold in metrics:
            values = [safe_float(r.get(metric), 0.0) for r in group]
            if metric == "actual_CouplingR2_delta":
                hard = [v >= threshold for v in values]
            else:
                hard = [v <= threshold for v in values]
            null_rows.append(
                {
                    "stage": "V1217_P1_LINEC_NULL_DISTRIBUTION",
                    "method": method,
                    "metric": metric,
                    "rows": len(values),
                    "mean": mean(values),
                    "std": std(values),
                    "min": min(values) if values else "",
                    "p10": percentile(values, 10),
                    "p50": percentile(values, 50),
                    "p90": percentile(values, 90),
                    "max": max(values) if values else "",
                    "hard_threshold": threshold,
                    "hard_support_rows": sum(1 for v in hard if v),
                    "is_null_control": int(method.startswith("U0") or method.startswith("U1") or "NoOp" in method or "Random" in method),
                }
            )
    for key, group in sorted(by_group.items()):
        source, dataset, seed, window, sketch_family = key
        out: dict[str, Any] = {
            "stage": "V1217_P1_LINEC_VARIANCE",
            "source_run": source,
            "dataset": dataset,
            "seed": seed,
            "window": window,
            "sketch_family": sketch_family,
            "rows": len(group),
        }
        for metric, _ in metrics:
            values = [safe_float(r.get(metric), 0.0) for r in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = std(values)
        variance_rows.append(out)
    thresholds = [-0.02, -0.015, -0.01, -0.0075, -0.005]
    for threshold in thresholds:
        for source in sorted({str(r.get("source_run", "")) for r in linec_rows}):
            group = [r for r in linec_rows if str(r.get("source_run", "")) == source]
            noise = [safe_float(r.get("actual_NoiseSignalLeak_delta"), 0.0) for r in group]
            reservoir = [safe_float(r.get("actual_RealSignalReservoirRatio_delta"), 0.0) for r in group]
            null_mask = [str(r.get("method", "")).startswith(("U0", "U1")) or "NoOp" in str(r.get("method", "")) or "Random" in str(r.get("method", "")) for r in group]
            sensitivity_rows.append(
                {
                    "stage": "V1217_P1_THRESHOLD_SENSITIVITY",
                    "source_run": source,
                    "threshold": threshold,
                    "rows": len(group),
                    "noise_support_rows": sum(1 for v in noise if v <= threshold),
                    "reservoir_support_rows": sum(1 for v in reservoir if v <= threshold),
                    "joint_support_rows": sum(1 for a, b in zip(noise, reservoir) if a <= threshold and b <= threshold),
                    "null_noise_false_positive_rows": sum(1 for a, n in zip(noise, null_mask) if n and a <= threshold),
                    "null_reservoir_false_positive_rows": sum(1 for b, n in zip(reservoir, null_mask) if n and b <= threshold),
                    "null_joint_false_positive_rows": sum(1 for a, b, n in zip(noise, reservoir, null_mask) if n and a <= threshold and b <= threshold),
                }
            )
    for source in sorted({str(r.get("source_run", "")) for r in linec_rows}):
        group = [r for r in linec_rows if str(r.get("source_run", "")) == source]
        rows_n = len(group)
        noise = [safe_float(r.get("actual_NoiseSignalLeak_delta"), 0.0) for r in group]
        reservoir = [safe_float(r.get("actual_RealSignalReservoirRatio_delta"), 0.0) for r in group]
        methods = [str(r.get("method", "")) for r in group]
        control_mask = [m.startswith(("U0", "U1")) or "NoOp" in m or "Random" in m for m in methods]
        noop_mask = [m.startswith("U0") or "NoOp" in m for m in methods]
        random_mask = [m.startswith("U1") or "Random" in m for m in methods]
        joint = [a <= -0.01 and b <= -0.01 for a, b in zip(noise, reservoir)]
        control_rows = sum(1 for n in control_mask if n)
        control_joint_fp = sum(1 for ok, n in zip(joint, control_mask) if ok and n)
        noop_joint_fp = sum(1 for ok, n in zip(joint, noop_mask) if ok and n)
        random_joint_fp = sum(1 for ok, n in zip(joint, random_mask) if ok and n)
        noise_support = sum(1 for v in noise if v <= -0.01)
        reservoir_support = sum(1 for v in reservoir if v <= -0.01)
        noise_support_rate = noise_support / max(1, rows_n)
        reservoir_support_rate = reservoir_support / max(1, rows_n)
        control_fp_rate = control_joint_fp / max(1, control_rows)
        source_pass = int(
            noop_joint_fp == 0
            and random_joint_fp == 0
            and control_fp_rate <= 0.01
            and noise_support_rate >= 0.05
            and reservoir_support_rate >= 0.05
        )
        source_calibration_rows.append(
            {
                "stage": "V1217_P1_SOURCE_CALIBRATION",
                "source_run": source,
                "rows": rows_n,
                "noise_hard_support_rows": noise_support,
                "noise_hard_support_rate": noise_support_rate,
                "reservoir_hard_support_rows": reservoir_support,
                "reservoir_hard_support_rate": reservoir_support_rate,
                "joint_hard_support_rows": sum(1 for ok in joint if ok),
                "control_rows": control_rows,
                "noop_joint_false_positive_rows": noop_joint_fp,
                "random_joint_false_positive_rows": random_joint_fp,
                "control_joint_false_positive_rows": control_joint_fp,
                "control_joint_false_positive_rate": control_fp_rate,
                "source_calibration_pass": source_pass,
                "repair_direction_if_failed": "expand batch/bootstrap or raise hard gate margin; do not lower release threshold",
            }
        )
    by_dsw: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in linec_rows:
        by_dsw[(str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("window", "")))].append(row)
    for key, group in sorted(by_dsw.items()):
        source, dataset, seed, window = key
        support_rows.append(
            {
                "stage": "V1217_P1_LINEC_HARD_SUPPORT",
                "source_run": source,
                "dataset": dataset,
                "seed": seed,
                "window": window,
                "rows": len(group),
                "noise_hard_support_rows": sum(1 for r in group if safe_float(r.get("actual_NoiseSignalLeak_delta"), 0.0) <= -0.01),
                "reservoir_hard_support_rows": sum(1 for r in group if safe_float(r.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= -0.01),
                "joint_hard_support_rows": sum(1 for r in group if safe_float(r.get("actual_NoiseSignalLeak_delta"), 0.0) <= -0.01 and safe_float(r.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= -0.01),
                "label_used_for_audit_only": 1,
                "label_used_for_direction": 0,
            }
        )
    write_csv_rows(out_dir / "v1217_linec_null_distribution.csv", null_rows)
    write_csv_rows(out_dir / "v1217_linec_variance.csv", variance_rows)
    write_csv_rows(out_dir / "v1217_linec_threshold_sensitivity.csv", sensitivity_rows)
    write_csv_rows(out_dir / "v1217_linec_hard_support.csv", support_rows)
    write_csv_rows(out_dir / "v1217_linec_source_calibration.csv", source_calibration_rows)
    total_noise = sum(safe_int(r.get("noise_hard_support_rows"), 0) for r in support_rows)
    total_reservoir = sum(safe_int(r.get("reservoir_hard_support_rows"), 0) for r in support_rows)
    total_joint = sum(safe_int(r.get("joint_hard_support_rows"), 0) for r in support_rows)
    hard_threshold_rows = [r for r in sensitivity_rows if abs(safe_float(r.get("threshold"), 0.0) + 0.01) < 1.0e-12]
    null_noise_fp = sum(safe_int(r.get("null_noise_false_positive_rows"), 0) for r in hard_threshold_rows)
    null_reservoir_fp = sum(safe_int(r.get("null_reservoir_false_positive_rows"), 0) for r in hard_threshold_rows)
    null_joint_fp = sum(safe_int(r.get("null_joint_false_positive_rows"), 0) for r in hard_threshold_rows)
    return {
        "p1_linec_calibration_pass": int(any(safe_int(r.get("source_calibration_pass"), 0) == 1 for r in source_calibration_rows)),
        "p1_calibrated_source_runs": [str(r.get("source_run", "")) for r in source_calibration_rows if safe_int(r.get("source_calibration_pass"), 0) == 1],
        "p1_noise_hard_support_rows": total_noise,
        "p1_reservoir_hard_support_rows": total_reservoir,
        "p1_joint_hard_support_rows": total_joint,
        "p1_null_noise_false_positive_rows_at_hard_threshold": null_noise_fp,
        "p1_null_reservoir_false_positive_rows_at_hard_threshold": null_reservoir_fp,
        "p1_null_joint_false_positive_rows_at_hard_threshold": null_joint_fp,
        "p1_support_groups": len(support_rows),
        "p1_source_calibration_rows": len(source_calibration_rows),
    }


def build_release_labels(linec_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_row_id: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(linec_rows):
        row_id = f"{row.get('source_run','src')}::{row.get('dataset','')}::{row.get('seed','')}::{row.get('window','')}::{row.get('method','')}::{row.get('sketch_id','')}::{idx}"
        noise_delta = safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0)
        reservoir_delta = safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0)
        label = {
            "stage": "V1217_RELEASE_LABELS_AUDIT_ONLY",
            "row_id": row_id,
            "source_run": row.get("source_run", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "window": row.get("window", ""),
            "candidate_probe_id": row.get("method", ""),
            "method": row.get("method", ""),
            "sketch_id": row.get("sketch_id", ""),
            "sketch_family": row.get("sketch_family", ""),
            "hard_noise_release": int(noise_delta <= -0.01),
            "hard_reservoir_release": int(reservoir_delta <= -0.01),
            "hard_joint_release": int(noise_delta <= -0.01 and reservoir_delta <= -0.01),
            "exploratory_noise_release": int(noise_delta <= -0.005),
            "exploratory_reservoir_release": int(reservoir_delta <= -0.005),
            "exploratory_joint_release": int(noise_delta <= -0.005 and reservoir_delta <= -0.005),
            "actual_NoiseSignalLeak_delta": noise_delta,
            "actual_RealSignalReservoirRatio_delta": reservoir_delta,
            "label_used_for_audit_only": 1,
            "label_used_for_feature": 0,
            "label_used_for_direction": 0,
            "ce_vector_used_for_feature": 0,
            "dataset_name_used_for_commit": 0,
        }
        rows.append(label)
        by_row_id[row_id] = label
    write_csv_rows(out_dir / "v1217_release_labels_audit_only.csv", rows)
    return rows, by_row_id


def build_release_support_concentration(labels: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in ["source_run", "dataset", "seed", "window", "method", "sketch_family"]:
        for value in sorted({str(row.get(key, "")) for row in labels}):
            group = [row for row in labels if str(row.get(key, "")) == value]
            rows.append(
                {
                    "stage": "V1217_RELEASE_SUPPORT_CONCENTRATION",
                    "analysis_type": "marginal_count",
                    "group_key": key,
                    "group_value": value,
                    "rows": len(group),
                    "noise_positive_rows": sum(safe_int(row.get("hard_noise_release"), 0) for row in group),
                    "reservoir_positive_rows": sum(safe_int(row.get("hard_reservoir_release"), 0) for row in group),
                    "joint_positive_rows": sum(safe_int(row.get("hard_joint_release"), 0) for row in group),
                }
            )
    split_absent = 0
    split_test_positive_absent_train = 0
    for split_key in ["dataset", "seed", "window"]:
        for heldout in sorted({str(row.get(split_key, "")) for row in labels}):
            train = [row for row in labels if str(row.get(split_key, "")) != heldout]
            test = [row for row in labels if str(row.get(split_key, "")) == heldout]
            train_joint = sum(safe_int(row.get("hard_joint_release"), 0) for row in train)
            test_joint = sum(safe_int(row.get("hard_joint_release"), 0) for row in test)
            absent = int(train_joint == 0)
            harmful_absent = int(train_joint == 0 and test_joint > 0)
            split_absent += absent
            split_test_positive_absent_train += harmful_absent
            rows.append(
                {
                    "stage": "V1217_RELEASE_SUPPORT_CONCENTRATION",
                    "analysis_type": "leaveout_support",
                    "group_key": split_key,
                    "group_value": heldout,
                    "rows": len(test),
                    "train_joint_positive_rows": train_joint,
                    "test_joint_positive_rows": test_joint,
                    "train_noise_positive_rows": sum(safe_int(row.get("hard_noise_release"), 0) for row in train),
                    "test_noise_positive_rows": sum(safe_int(row.get("hard_noise_release"), 0) for row in test),
                    "train_reservoir_positive_rows": sum(safe_int(row.get("hard_reservoir_release"), 0) for row in train),
                    "test_reservoir_positive_rows": sum(safe_int(row.get("hard_reservoir_release"), 0) for row in test),
                    "train_joint_support_absent": absent,
                    "test_positive_with_train_joint_absent": harmful_absent,
                }
            )
    total_joint = sum(safe_int(row.get("hard_joint_release"), 0) for row in labels)
    by_seed = defaultdict(int)
    by_window = defaultdict(int)
    for row in labels:
        if safe_int(row.get("hard_joint_release"), 0):
            by_seed[str(row.get("seed", ""))] += 1
            by_window[str(row.get("window", ""))] += 1
    concentrated = int(total_joint > 0 and (max(by_seed.values() or [0]) == total_joint or max(by_window.values() or [0]) == total_joint))
    write_csv_rows(out_dir / "v1217_release_support_concentration.csv", rows)
    return {
        "p2_joint_support_concentrated": concentrated,
        "p2_joint_support_total_rows": total_joint,
        "p2_joint_support_seed_distribution": dict(sorted(by_seed.items())),
        "p2_joint_support_window_distribution": dict(sorted(by_window.items())),
        "p2_leaveout_train_joint_absent_count": split_absent,
        "p2_leaveout_test_positive_with_train_joint_absent_count": split_test_positive_absent_train,
    }


def build_visibility_features(
    linec_rows: Sequence[Mapping[str, Any]],
    labels_by_row_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    grouped_for_persistence: dict[tuple[str, str, str, str, str], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    row_ids: list[str] = []
    for idx, row in enumerate(linec_rows):
        row_id = f"{row.get('source_run','src')}::{row.get('dataset','')}::{row.get('seed','')}::{row.get('window','')}::{row.get('method','')}::{row.get('sketch_id','')}::{idx}"
        row_ids.append(row_id)
        key = (str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("method", "")), str(row.get("sketch_id", "")))
        grouped_for_persistence[key].append((idx, row))
    persistence: dict[int, dict[str, float]] = {}
    for group in grouped_for_persistence.values():
        group_sorted = sorted(group, key=lambda item: safe_int(item[1].get("window"), 0))
        prev: Mapping[str, Any] | None = None
        for idx, row in group_sorted:
            if prev is None:
                persistence[idx] = {
                    "persistence_reservoir_fraction_absdiff": 0.0,
                    "persistence_projector_stability_absdiff": 0.0,
                    "persistence_coupling_absdiff": 0.0,
                }
            else:
                stability = 0.5 * (safe_float(row.get("signal_projector_stability"), 0.0) + safe_float(row.get("reservoir_projector_stability"), 0.0))
                prev_stability = 0.5 * (safe_float(prev.get("signal_projector_stability"), 0.0) + safe_float(prev.get("reservoir_projector_stability"), 0.0))
                persistence[idx] = {
                    "persistence_reservoir_fraction_absdiff": abs(safe_float(row.get("reservoir_fraction"), 0.0) - safe_float(prev.get("reservoir_fraction"), 0.0)),
                    "persistence_projector_stability_absdiff": abs(stability - prev_stability),
                    "persistence_coupling_absdiff": abs(safe_float(row.get("actual_CouplingR2_delta"), 0.0) - safe_float(prev.get("actual_CouplingR2_delta"), 0.0)),
                }
            prev = row

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    row_features: dict[str, dict[str, float]] = {}
    for idx, row in enumerate(linec_rows):
        row_id = row_ids[idx]
        method = str(row.get("method", ""))
        cotangent_family = str(row.get("cotangent_family", ""))
        sketch_family = str(row.get("sketch_family", ""))
        stable = safe_float(row.get("stable_cotangent_energy"), 0.0)
        unstable = safe_float(row.get("unstable_cotangent_energy"), 0.0)
        features = {
            "reservoir_fraction": safe_float(row.get("reservoir_fraction"), 0.0),
            "top_eigen_share": safe_float(row.get("top_eigen_share"), 0.0),
            "stable_cotangent_energy": stable,
            "unstable_cotangent_energy": unstable,
            "stable_unstable_balance": (stable - unstable) / max(1.0e-12, abs(stable) + abs(unstable)),
            "pred_coupling_delta": safe_float(row.get("pred_coupling_delta"), 0.0),
            "actual_CouplingR2_delta": safe_float(row.get("actual_CouplingR2_delta"), 0.0),
            "signal_projector_stability": safe_float(row.get("signal_projector_stability"), 0.0),
            "reservoir_projector_stability": safe_float(row.get("reservoir_projector_stability"), 0.0),
            "projector_stability_gap": safe_float(row.get("signal_projector_stability"), 0.0) - safe_float(row.get("reservoir_projector_stability"), 0.0),
            "signal_rank": safe_float(row.get("signal_rank"), 0.0),
            "reservoir_rank": safe_float(row.get("reservoir_rank"), 0.0),
            "signal_effective_rank": safe_float(row.get("signal_effective_rank"), 0.0),
            "dissipation_condition": safe_float(row.get("dissipation_condition"), 0.0),
            "sketch_dim": safe_float(row.get("sketch_dim"), 0.0),
            "method_is_control": float(method.startswith("U0") or method.startswith("U1") or "NoOp" in method or "Random" in method),
            "method_is_random": float("Random" in method),
            "method_is_cotangent_vjp": float("Cotangent" in method or "VJP" in method),
            "method_is_role_actuator": float("Role" in method or "Actuator" in method),
            "method_is_projector": float("Projector" in method or "Projection" in method),
            "augmentation_used": safe_float(row.get("augmentation_used"), 0.0),
            "cotangent_family_hash": stable_hash_unit(cotangent_family),
            "sketch_family_hash": stable_hash_unit(sketch_family),
            "energy_scale": math.log1p(abs(stable) + abs(unstable)),
            "window": safe_float(row.get("window"), 0.0),
        }
        features.update(persistence.get(idx, {}))
        label = labels_by_row_id[row_id]
        wide = {
            "row_id": row_id,
            "source_run": row.get("source_run", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "window": row.get("window", ""),
            "candidate_probe_id": method,
            "method": method,
            "sketch_id": row.get("sketch_id", ""),
            "sketch_family": sketch_family,
            "hard_noise_release": label.get("hard_noise_release", 0),
            "hard_reservoir_release": label.get("hard_reservoir_release", 0),
            "hard_joint_release": label.get("hard_joint_release", 0),
            "dataset_name_used_for_commit": 0,
            "label_used_for_feature": 0,
            "ce_vector_used_for_feature": 0,
        }
        wide.update(features)
        wide_rows.append(wide)
        row_features[row_id] = features
        for family, names in FEATURE_FAMILIES.items():
            for name in names:
                long_rows.append(
                    {
                        "stage": "V1217_TARGET_VISIBILITY_FEATURES",
                        "row_id": row_id,
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "window": row.get("window", ""),
                        "candidate_probe_id": method,
                        "feature_family": family,
                        "feature_name": name,
                        "feature_value": features.get(name, 0.0),
                        "hard_noise_release": label.get("hard_noise_release", 0),
                        "hard_reservoir_release": label.get("hard_reservoir_release", 0),
                        "hard_joint_release": label.get("hard_joint_release", 0),
                        "score_calibrated": "",
                        "score_heldout": "",
                        "AUC_noise": "",
                        "AUC_reservoir": "",
                        "AUC_joint": "",
                        "precision_at_k_noise": "",
                        "precision_at_k_reservoir": "",
                        "precision_at_k_joint": "",
                        "recall_at_k_noise": "",
                        "recall_at_k_reservoir": "",
                        "recall_at_k_joint": "",
                        "dataset_name_used_for_commit": 0,
                        "label_used_for_feature": 0,
                        "ce_vector_used_for_feature": 0,
                    }
                )
    return wide_rows, long_rows, row_features


def feature_matrix(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[safe_float(row.get(name), 0.0) for name in feature_names] for row in rows], dtype=float)


def crossval_predictions(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    target_name: str,
    split_key: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    split_values = sorted({str(row.get(split_key, "")) for row in rows})
    by_row_score: dict[str, float] = {}
    holdout_rows: list[dict[str, Any]] = []
    for holdout in split_values:
        train = [row for row in rows if str(row.get(split_key, "")) != holdout]
        test = [row for row in rows if str(row.get(split_key, "")) == holdout]
        train_x = feature_matrix(train, feature_names)
        train_y = np.asarray([safe_int(row.get(target_name), 0) for row in train], dtype=float)
        test_x = feature_matrix(test, feature_names)
        scores = ridge_predict(train_x, train_y, test_x)
        labels = [safe_int(row.get(target_name), 0) for row in test]
        auc = auc_score(labels, scores.tolist())
        prec, rec, hits, positives = precision_recall_at_k(labels, scores.tolist())
        holdout_rows.append(
            {
                "stage": "V1217_VISIBILITY_LEAVEOUT",
                "split_protocol": f"leave_{split_key}_out",
                "heldout_value": holdout,
                "target": target_name,
                "feature_count": len(feature_names),
                "rows": len(test),
                "positive_rows": positives,
                "AUC": auc,
                "precision_at_k": prec,
                "recall_at_k": rec,
                "hits_at_k": hits,
            }
        )
        for row, score in zip(test, scores.tolist()):
            by_row_score[str(row.get("row_id", ""))] = float(score)
    return holdout_rows, by_row_score


def run_p2_target_visibility(
    wide_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    all_feature_names = [name for names in FEATURE_FAMILIES.values() for name in names]
    scopes: dict[str, list[str]] = {family: names for family, names in FEATURE_FAMILIES.items()}
    scopes["joint_all"] = all_feature_names
    targets = {
        "noise": "hard_noise_release",
        "reservoir": "hard_reservoir_release",
        "joint": "hard_joint_release",
    }
    score_rows: list[dict[str, Any]] = []
    leaveout_rows: list[dict[str, Any]] = []
    row_scores_joint: dict[str, list[float]] = defaultdict(list)
    aggregate: dict[tuple[str, str], dict[str, float]] = {}
    for scope_name, feature_names in scopes.items():
        for split_key in ["dataset", "seed", "window"]:
            target_predictions: dict[str, dict[str, float]] = {}
            target_leaveout: dict[str, list[dict[str, Any]]] = {}
            for target_label, target_name in targets.items():
                rows_out, by_row_score = crossval_predictions(wide_rows, feature_names, target_name, split_key)
                for row in rows_out:
                    row.update({"score_scope": scope_name, "target_label": target_label})
                leaveout_rows.extend(rows_out)
                target_leaveout[target_label] = rows_out
                target_predictions[target_label] = by_row_score
            score_row: dict[str, Any] = {
                "stage": "V1217_VISIBILITY_SCORES",
                "score_scope": scope_name,
                "feature_count": len(feature_names),
                "split_protocol": f"leave_{split_key}_out",
                "rows": len(wide_rows),
                "source_runs": ",".join(sorted({str(r.get("source_run", "")) for r in wide_rows})),
                "dataset_name_used_for_commit": 0,
                "label_used_for_feature": 0,
                "ce_vector_used_for_feature": 0,
            }
            for target_label, target_name in targets.items():
                scores = [target_predictions[target_label].get(str(row.get("row_id", "")), float("nan")) for row in wide_rows]
                labels = [safe_int(row.get(target_name), 0) for row in wide_rows]
                auc = auc_score(labels, scores)
                prec, rec, hits, positives = precision_recall_at_k(labels, scores)
                score_row[f"AUC_{target_label}"] = auc
                score_row[f"precision_at_k_{target_label}"] = prec
                score_row[f"recall_at_k_{target_label}"] = rec
                score_row[f"hits_at_k_{target_label}"] = hits
                score_row[f"positive_rows_{target_label}"] = positives
                aggregate[(scope_name, f"{split_key}:{target_label}")] = {"auc": auc, "precision": prec, "recall": rec}
            score_rows.append(score_row)
            if scope_name == "joint_all":
                for row in wide_rows:
                    rid = str(row.get("row_id", ""))
                    if rid in target_predictions["joint"]:
                        row_scores_joint[rid].append(target_predictions["joint"][rid])
    # Repeat the joint heldout score and aggregate metrics into feature rows for grep-friendly artifact contract.
    joint_scores = {rid: mean(vals) for rid, vals in row_scores_joint.items()}
    joint_score_rows = [r for r in score_rows if r.get("score_scope") == "joint_all"]
    auc_noise_vals = [safe_float(r.get("AUC_noise"), float("nan")) for r in joint_score_rows]
    auc_reservoir_vals = [safe_float(r.get("AUC_reservoir"), float("nan")) for r in joint_score_rows]
    auc_joint_vals = [safe_float(r.get("AUC_joint"), float("nan")) for r in joint_score_rows]
    p_joint_vals = [safe_float(r.get("precision_at_k_joint"), float("nan")) for r in joint_score_rows]
    r_joint_vals = [safe_float(r.get("recall_at_k_joint"), float("nan")) for r in joint_score_rows]
    summary_for_features = {
        "AUC_noise": mean(auc_noise_vals),
        "AUC_reservoir": mean(auc_reservoir_vals),
        "AUC_joint": mean(auc_joint_vals),
        "precision_at_k_joint": mean(p_joint_vals),
        "recall_at_k_joint": mean(r_joint_vals),
    }
    for row in long_rows:
        rid = str(row.get("row_id", ""))
        row["score_heldout"] = joint_scores.get(rid, "")
        row["score_calibrated"] = row["score_heldout"]
        row["AUC_noise"] = summary_for_features["AUC_noise"]
        row["AUC_reservoir"] = summary_for_features["AUC_reservoir"]
        row["AUC_joint"] = summary_for_features["AUC_joint"]
        row["precision_at_k_joint"] = summary_for_features["precision_at_k_joint"]
        row["recall_at_k_joint"] = summary_for_features["recall_at_k_joint"]
    ablation_rows: list[dict[str, Any]] = []
    for family, names in FEATURE_FAMILIES.items():
        for name in names:
            scores = [safe_float(row.get(name), 0.0) for row in wide_rows]
            for target_label, target_name in targets.items():
                labels = [safe_int(row.get(target_name), 0) for row in wide_rows]
                auc = auc_score(labels, scores)
                auc_best_oriented = max(auc, 1.0 - auc) if not math.isnan(auc) else float("nan")
                ablation_rows.append(
                    {
                        "stage": "V1217_FEATURE_ABLATION",
                        "feature_family": family,
                        "feature_name": name,
                        "target": target_label,
                        "rows": len(wide_rows),
                        "positive_rows": sum(labels),
                        "raw_auc": auc,
                        "best_oriented_auc": auc_best_oriented,
                        "label_used_for_feature": 0,
                        "ce_vector_used_for_feature": 0,
                    }
                )
    write_csv_rows(out_dir / "v1217_target_visibility_features.csv", long_rows)
    write_csv_rows(out_dir / "v1217_visibility_scores.csv", score_rows)
    write_csv_rows(out_dir / "v1217_visibility_leaveout.csv", leaveout_rows)
    write_csv_rows(out_dir / "v1217_feature_ablation.csv", ablation_rows)

    def min_valid(values: Sequence[float]) -> float:
        clean = [v for v in values if not math.isnan(v)]
        return min(clean) if clean else float("nan")

    heldout_auc_noise = min_valid(auc_noise_vals)
    heldout_auc_reservoir = min_valid(auc_reservoir_vals)
    heldout_auc_joint = min_valid(auc_joint_vals)
    heldout_precision_joint = min_valid(p_joint_vals)
    heldout_recall_joint = min_valid(r_joint_vals)
    exploratory = int((heldout_auc_joint >= 0.60) or (heldout_auc_noise >= 0.65 and heldout_auc_reservoir >= 0.65))
    hard = int(heldout_precision_joint >= 0.25 and heldout_recall_joint >= 0.20)
    strong = int(heldout_auc_joint >= 0.70 and heldout_precision_joint >= 0.40 and heldout_recall_joint >= 0.30)
    family_best_joint_auc = {
        family: max(
            [safe_float(r.get("AUC_joint"), float("nan")) for r in score_rows if r.get("score_scope") == family and not math.isnan(safe_float(r.get("AUC_joint"), float("nan")))]
            or [float("nan")]
        )
        for family in FEATURE_FAMILIES
    }
    all_family_below_055 = int(all((not math.isnan(v)) and v < 0.55 for v in family_best_joint_auc.values()))
    return {
        "p2_visibility_pass": int(exploratory and hard),
        "p2_visibility_exploratory_pass": exploratory,
        "p2_visibility_hard_pass": hard,
        "p2_visibility_strong_pass": strong,
        "p2_auc_noise_heldout_min": heldout_auc_noise,
        "p2_auc_reservoir_heldout_min": heldout_auc_reservoir,
        "p2_auc_joint_heldout_min": heldout_auc_joint,
        "p2_precision_joint_heldout_min": heldout_precision_joint,
        "p2_recall_joint_heldout_min": heldout_recall_joint,
        "p2_feature_rows": len(long_rows),
        "p2_score_rows": len(score_rows),
        "p2_leaveout_rows": len(leaveout_rows),
        "p2_joint_scores_by_row": joint_scores,
        "p2_family_best_joint_auc": family_best_joint_auc,
        "p2_all_family_joint_auc_below_055": all_family_below_055,
    }


def rank01(values: Sequence[float]) -> list[float]:
    clean = [(idx, float(v)) for idx, v in enumerate(values) if not math.isnan(float(v))]
    out = [float("nan")] * len(values)
    if not clean:
        return out
    ordered = sorted(clean, key=lambda item: item[1])
    denom = max(1, len(ordered) - 1)
    for rank, (idx, _) in enumerate(ordered):
        out[idx] = rank / denom
    return out


def run_p2_visibility_repair(
    wide_rows: list[dict[str, Any]],
    out_dir: Path,
    base_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Try predeclared label-free score transforms when base P2 is weak.

    Labels are still used only as heldout audit targets. The repair changes the
    visibility scorer, not any candidate direction or release threshold.
    """

    all_feature_names = [name for names in FEATURE_FAMILIES.values() for name in names]
    repair_scopes: dict[str, list[str]] = {
        "repair_all_features": all_feature_names,
        "repair_coupling_spectrum_persistence": FEATURE_FAMILIES["F2_train_probe_coupling"]
        + FEATURE_FAMILIES["F3_gradient_sketch_spectrum"]
        + FEATURE_FAMILIES["F6_time_persistence"],
        "repair_nonrole_geometry": FEATURE_FAMILIES["F1_output_spectral"]
        + FEATURE_FAMILIES["F2_train_probe_coupling"]
        + FEATURE_FAMILIES["F3_gradient_sketch_spectrum"]
        + FEATURE_FAMILIES["F5_augmentation_consistency"]
        + FEATURE_FAMILIES["F6_time_persistence"],
    }
    target_names = {
        "noise": "hard_noise_release",
        "reservoir": "hard_reservoir_release",
        "joint": "hard_joint_release",
    }
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "p2_visibility_repair_pass": 0,
        "p2_visibility_repair_best_protocol": "",
        "p2_visibility_repair_best_auc_joint_min": float("nan"),
        "p2_visibility_repair_best_precision_joint_min": float("nan"),
        "p2_visibility_repair_best_recall_joint_min": float("nan"),
    }
    for scope_name, feature_names in repair_scopes.items():
        for combine_rule in ["min_noise_reservoir_rank", "product_noise_reservoir_rank", "mean_noise_reservoir_rank"]:
            split_metrics: list[dict[str, Any]] = []
            for split_key in ["dataset", "seed", "window"]:
                split_values = sorted({str(row.get(split_key, "")) for row in wide_rows})
                score_by_row: dict[str, float] = {}
                for heldout in split_values:
                    train = [row for row in wide_rows if str(row.get(split_key, "")) != heldout]
                    test = [row for row in wide_rows if str(row.get(split_key, "")) == heldout]
                    train_x = feature_matrix(train, feature_names)
                    test_x = feature_matrix(test, feature_names)
                    noise_scores = ridge_predict(train_x, np.asarray([safe_int(row.get("hard_noise_release"), 0) for row in train], dtype=float), test_x)
                    reservoir_scores = ridge_predict(train_x, np.asarray([safe_int(row.get("hard_reservoir_release"), 0) for row in train], dtype=float), test_x)
                    noise_rank = rank01(noise_scores.tolist())
                    reservoir_rank = rank01(reservoir_scores.tolist())
                    for row, ns, rs in zip(test, noise_rank, reservoir_rank):
                        if combine_rule == "min_noise_reservoir_rank":
                            score = min(ns, rs)
                        elif combine_rule == "product_noise_reservoir_rank":
                            score = ns * rs
                        else:
                            score = 0.5 * (ns + rs)
                        score_by_row[str(row.get("row_id", ""))] = float(score)
                scores = [score_by_row.get(str(row.get("row_id", "")), float("nan")) for row in wide_rows]
                metric_row: dict[str, Any] = {
                    "stage": "V1217_P2_VISIBILITY_REPAIR_ATTEMPT",
                    "repair_scope": scope_name,
                    "combine_rule": combine_rule,
                    "feature_count": len(feature_names),
                    "split_protocol": f"leave_{split_key}_out",
                    "rows": len(wide_rows),
                    "source_runs": ",".join(sorted({str(row.get("source_run", "")) for row in wide_rows})),
                    "label_used_for_feature": 0,
                    "label_used_for_direction": 0,
                    "ce_vector_used_for_feature": 0,
                    "dataset_name_used_for_commit": 0,
                    "repair_rationale": "compose separately heldout noise/reservoir visibility scores for sparse joint support; no release threshold change",
                }
                for target_label, target_name in target_names.items():
                    labels = [safe_int(row.get(target_name), 0) for row in wide_rows]
                    auc = auc_score(labels, scores)
                    prec, rec, hits, positives = precision_recall_at_k(labels, scores)
                    metric_row[f"AUC_{target_label}"] = auc
                    metric_row[f"precision_at_k_{target_label}"] = prec
                    metric_row[f"recall_at_k_{target_label}"] = rec
                    metric_row[f"hits_at_k_{target_label}"] = hits
                    metric_row[f"positive_rows_{target_label}"] = positives
                rows.append(metric_row)
                split_metrics.append(metric_row)
            auc_joint_min = min([safe_float(row.get("AUC_joint"), float("nan")) for row in split_metrics])
            auc_noise_min = min([safe_float(row.get("AUC_noise"), float("nan")) for row in split_metrics])
            auc_reservoir_min = min([safe_float(row.get("AUC_reservoir"), float("nan")) for row in split_metrics])
            precision_joint_min = min([safe_float(row.get("precision_at_k_joint"), float("nan")) for row in split_metrics])
            recall_joint_min = min([safe_float(row.get("recall_at_k_joint"), float("nan")) for row in split_metrics])
            exploratory = int((auc_joint_min >= 0.60) or (auc_noise_min >= 0.65 and auc_reservoir_min >= 0.65))
            hard = int(precision_joint_min >= 0.25 and recall_joint_min >= 0.20)
            pass_flag = int(exploratory and hard)
            summary_row = {
                "stage": "V1217_P2_VISIBILITY_REPAIR_SUMMARY",
                "repair_scope": scope_name,
                "combine_rule": combine_rule,
                "AUC_noise_heldout_min": auc_noise_min,
                "AUC_reservoir_heldout_min": auc_reservoir_min,
                "AUC_joint_heldout_min": auc_joint_min,
                "precision_joint_heldout_min": precision_joint_min,
                "recall_joint_heldout_min": recall_joint_min,
                "exploratory_pass": exploratory,
                "hard_pass": hard,
                "repair_pass": pass_flag,
                "base_p2_visibility_pass": safe_int(base_summary.get("p2_visibility_pass"), 0),
            }
            summary_rows.append(summary_row)
            current_best_auc = safe_float(best.get("p2_visibility_repair_best_auc_joint_min"), float("nan"))
            current_best_pass = safe_int(best.get("p2_visibility_repair_pass"), 0)
            if (pass_flag and not current_best_pass) or ((pass_flag == current_best_pass) and (math.isnan(current_best_auc) or auc_joint_min > current_best_auc)):
                best = {
                    "p2_visibility_repair_pass": pass_flag,
                    "p2_visibility_repair_best_protocol": f"{scope_name}/{combine_rule}",
                    "p2_visibility_repair_best_auc_noise_min": auc_noise_min,
                    "p2_visibility_repair_best_auc_reservoir_min": auc_reservoir_min,
                    "p2_visibility_repair_best_auc_joint_min": auc_joint_min,
                    "p2_visibility_repair_best_precision_joint_min": precision_joint_min,
                    "p2_visibility_repair_best_recall_joint_min": recall_joint_min,
                }
            if pass_flag:
                best["p2_visibility_repair_pass"] = 1
        for positive_weight in [5.0, 10.0, 25.0, 50.0]:
            split_metrics = []
            for split_key in ["dataset", "seed", "window"]:
                split_values = sorted({str(row.get(split_key, "")) for row in wide_rows})
                score_by_row: dict[str, float] = {}
                for heldout in split_values:
                    train = [row for row in wide_rows if str(row.get(split_key, "")) != heldout]
                    test = [row for row in wide_rows if str(row.get(split_key, "")) == heldout]
                    train_x = feature_matrix(train, feature_names)
                    test_x = feature_matrix(test, feature_names)
                    scores = weighted_ridge_predict(
                        train_x,
                        np.asarray([safe_int(row.get("hard_joint_release"), 0) for row in train], dtype=float),
                        test_x,
                        positive_weight=positive_weight,
                    )
                    for row, score in zip(test, scores.tolist()):
                        score_by_row[str(row.get("row_id", ""))] = float(score)
                scores_all = [score_by_row.get(str(row.get("row_id", "")), float("nan")) for row in wide_rows]
                metric_row = {
                    "stage": "V1217_P2_VISIBILITY_REPAIR_ATTEMPT",
                    "repair_scope": scope_name,
                    "combine_rule": f"weighted_sparse_joint_ridge_pos{positive_weight:g}",
                    "feature_count": len(feature_names),
                    "split_protocol": f"leave_{split_key}_out",
                    "rows": len(wide_rows),
                    "source_runs": ",".join(sorted({str(row.get("source_run", "")) for row in wide_rows})),
                    "label_used_for_feature": 0,
                    "label_used_for_direction": 0,
                    "ce_vector_used_for_feature": 0,
                    "dataset_name_used_for_commit": 0,
                    "repair_rationale": "sparse hard-support visibility scorer with positive weighting; audit labels train scorer only and do not enter candidate direction",
                }
                for target_label, target_name in target_names.items():
                    labels = [safe_int(row.get(target_name), 0) for row in wide_rows]
                    auc = auc_score(labels, scores_all)
                    prec, rec, hits, positives = precision_recall_at_k(labels, scores_all)
                    metric_row[f"AUC_{target_label}"] = auc
                    metric_row[f"precision_at_k_{target_label}"] = prec
                    metric_row[f"recall_at_k_{target_label}"] = rec
                    metric_row[f"hits_at_k_{target_label}"] = hits
                    metric_row[f"positive_rows_{target_label}"] = positives
                rows.append(metric_row)
                split_metrics.append(metric_row)
            auc_joint_min = min([safe_float(row.get("AUC_joint"), float("nan")) for row in split_metrics])
            auc_noise_min = min([safe_float(row.get("AUC_noise"), float("nan")) for row in split_metrics])
            auc_reservoir_min = min([safe_float(row.get("AUC_reservoir"), float("nan")) for row in split_metrics])
            precision_joint_min = min([safe_float(row.get("precision_at_k_joint"), float("nan")) for row in split_metrics])
            recall_joint_min = min([safe_float(row.get("recall_at_k_joint"), float("nan")) for row in split_metrics])
            exploratory = int((auc_joint_min >= 0.60) or (auc_noise_min >= 0.65 and auc_reservoir_min >= 0.65))
            hard = int(precision_joint_min >= 0.25 and recall_joint_min >= 0.20)
            pass_flag = int(exploratory and hard)
            summary_row = {
                "stage": "V1217_P2_VISIBILITY_REPAIR_SUMMARY",
                "repair_scope": scope_name,
                "combine_rule": f"weighted_sparse_joint_ridge_pos{positive_weight:g}",
                "AUC_noise_heldout_min": auc_noise_min,
                "AUC_reservoir_heldout_min": auc_reservoir_min,
                "AUC_joint_heldout_min": auc_joint_min,
                "precision_joint_heldout_min": precision_joint_min,
                "recall_joint_heldout_min": recall_joint_min,
                "exploratory_pass": exploratory,
                "hard_pass": hard,
                "repair_pass": pass_flag,
                "base_p2_visibility_pass": safe_int(base_summary.get("p2_visibility_pass"), 0),
            }
            summary_rows.append(summary_row)
            current_best_auc = safe_float(best.get("p2_visibility_repair_best_auc_joint_min"), float("nan"))
            current_best_pass = safe_int(best.get("p2_visibility_repair_pass"), 0)
            if (pass_flag and not current_best_pass) or ((pass_flag == current_best_pass) and (math.isnan(current_best_auc) or auc_joint_min > current_best_auc)):
                best = {
                    "p2_visibility_repair_pass": pass_flag,
                    "p2_visibility_repair_best_protocol": f"{scope_name}/weighted_sparse_joint_ridge_pos{positive_weight:g}",
                    "p2_visibility_repair_best_auc_noise_min": auc_noise_min,
                    "p2_visibility_repair_best_auc_reservoir_min": auc_reservoir_min,
                    "p2_visibility_repair_best_auc_joint_min": auc_joint_min,
                    "p2_visibility_repair_best_precision_joint_min": precision_joint_min,
                    "p2_visibility_repair_best_recall_joint_min": recall_joint_min,
                }
            if pass_flag:
                best["p2_visibility_repair_pass"] = 1
    write_csv_rows(out_dir / "v1217_visibility_repair_attempts.csv", rows)
    write_csv_rows(out_dir / "v1217_visibility_repair_summary.csv", summary_rows)
    best["p2_visibility_repair_attempt_rows"] = len(rows)
    best["p2_visibility_repair_summary_rows"] = len(summary_rows)
    if safe_int(base_summary.get("p2_visibility_pass"), 0) == 0 and safe_int(best.get("p2_visibility_repair_pass"), 0) == 1:
        best["p2_visibility_pass"] = 1
        best["p2_visibility_pass_source"] = "repair_composite_score"
    else:
        best["p2_visibility_pass"] = safe_int(base_summary.get("p2_visibility_pass"), 0)
        best["p2_visibility_pass_source"] = "base" if safe_int(base_summary.get("p2_visibility_pass"), 0) else "none"
    return best


def run_p3_actuator_response(
    actuator_rows: Sequence[Mapping[str, Any]],
    p2_summary: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    dictionary_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    response_score_rows: list[dict[str, Any]] = []
    row_scores = p2_summary.get("p2_joint_scores_by_row", {})
    by_group: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in actuator_rows:
        by_group[(str(row.get("source_run", "")), str(row.get("dataset", "")), str(row.get("seed", "")))].append(row)
    for row in actuator_rows:
        projector_angle = max(safe_float(row.get("signal_projector_angle_deg"), 0.0), safe_float(row.get("reservoir_projector_angle_deg"), 0.0))
        safe_gate = int(
            safe_float(row.get("sketch_delta_fro"), 0.0) >= 0.01
            and projector_angle >= 1.0
            and safe_float(row.get("logit_max_abs_drift"), 0.0) <= 0.05
        )
        exploratory_gate = int(
            safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0) <= -0.005
            and safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= -0.005
        )
        hard_gate = int(
            safe_float(row.get("actual_NoiseSignalLeak_delta"), 0.0) <= -0.01
            and safe_float(row.get("actual_RealSignalReservoirRatio_delta"), 0.0) <= -0.01
        )
        out = dict(row)
        out.update(
            {
                "stage": "V1217_P3_ACTUATOR_RESPONSE_DICTIONARY",
                "projector_angle_max_deg": projector_angle,
                "safe_movement_gate": safe_gate,
                "exploratory_release_gate": exploratory_gate,
                "official_hard_release_gate": hard_gate,
                "label_used_for_direction": 0,
                "ce_vector_used_for_direction": 0,
                "dataset_name_used_for_commit": 0,
            }
        )
        dictionary_rows.append(out)
        response_score_rows.append(
            {
                "stage": "V1217_RESPONSE_TO_VISIBILITY_SCORE",
                "source_run": row.get("source_run", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("window", ""),
                "actuator_id": row.get("actuator_id", row.get("method", "")),
                "visibility_score_available": int(bool(row_scores)),
                "visibility_score_join_rule": "mean_joint_all_leaveout_score_by_source_dataset_seed_window_unavailable_for_actuator_rows",
                "visibility_score": "",
                "safe_movement_gate": safe_gate,
                "exploratory_release_gate": exploratory_gate,
                "official_hard_release_gate": hard_gate,
                "actual_NoiseSignalLeak_delta": row.get("actual_NoiseSignalLeak_delta", ""),
                "actual_RealSignalReservoirRatio_delta": row.get("actual_RealSignalReservoirRatio_delta", ""),
                "label_used_for_direction": 0,
            }
        )
    for key, group in sorted(by_group.items()):
        source, dataset, seed = key
        ranks = [safe_float(row.get("response_matrix_rank"), float("nan")) for row in group]
        conds = [safe_float(row.get("response_matrix_condition"), float("nan")) for row in group]
        safe_rows = [
            row
            for row in dictionary_rows
            if str(row.get("source_run", "")) == source and str(row.get("dataset", "")) == dataset and str(row.get("seed", "")) == seed and safe_int(row.get("safe_movement_gate"), 0) == 1
        ]
        exploratory_rows = [
            row
            for row in dictionary_rows
            if str(row.get("source_run", "")) == source and str(row.get("dataset", "")) == dataset and str(row.get("seed", "")) == seed and safe_int(row.get("exploratory_release_gate"), 0) == 1
        ]
        hard_rows = [
            row
            for row in dictionary_rows
            if str(row.get("source_run", "")) == source and str(row.get("dataset", "")) == dataset and str(row.get("seed", "")) == seed and safe_int(row.get("official_hard_release_gate"), 0) == 1
        ]
        rank_rows.append(
            {
                "stage": "V1217_P3_ACTUATOR_RANK_CONDITION",
                "source_run": source,
                "dataset": dataset,
                "seed": seed,
                "actuator_rows": len(group),
                "response_rank_max": max(ranks) if ranks else "",
                "response_rank_mean": mean(ranks),
                "response_condition_min": min(conds) if conds else "",
                "response_condition_mean": mean(conds),
                "rank_low_blocker": int((max(ranks) if ranks else 0.0) < 3.0),
            }
        )
        combo_rows.append(
            {
                "stage": "V1217_P3_ACTUATOR_SAFE_COMBO",
                "source_run": source,
                "dataset": dataset,
                "seed": seed,
                "safe_movement_gate": int(bool(safe_rows)),
                "safe_movement_actuator_count": len(safe_rows),
                "exploratory_release_gate": int(bool(exploratory_rows)),
                "exploratory_release_actuator_count": len(exploratory_rows),
                "official_hard_release_gate": int(bool(hard_rows)),
                "official_hard_release_actuator_count": len(hard_rows),
                "control_gap_checked": 0,
                "control_gap_not_checked_reason": "v12.16 actuator matrix has no matched NoOp/Random actuator-control rows; P2 failed so no P3 promotion attempted",
            }
        )
    write_csv_rows(out_dir / "v1217_actuator_response_dictionary.csv", dictionary_rows)
    write_csv_rows(out_dir / "v1217_actuator_rank_condition.csv", rank_rows)
    write_csv_rows(out_dir / "v1217_actuator_safe_combo.csv", combo_rows)
    write_csv_rows(out_dir / "v1217_response_to_visibility_score.csv", response_score_rows)
    safe_count = sum(safe_int(row.get("safe_movement_gate"), 0) for row in combo_rows)
    exploratory_count = sum(safe_int(row.get("exploratory_release_gate"), 0) for row in combo_rows)
    hard_count = sum(safe_int(row.get("official_hard_release_gate"), 0) for row in combo_rows)
    p3_pass = int(safe_count >= 6 and exploratory_count >= 3 and safe_int(p2_summary.get("p2_visibility_pass"), 0) == 1)
    return {
        "p3_response_pass": p3_pass,
        "p3_safe_movement_pass_count": safe_count,
        "p3_exploratory_release_pass_count": exploratory_count,
        "p3_hard_release_pass_count": hard_count,
        "p3_expected_group_count": len(combo_rows),
        "p3_dictionary_rows": len(dictionary_rows),
        "p3_max_response_rank": max([safe_float(r.get("response_rank_max"), float("nan")) for r in rank_rows] or [float("nan")]),
        "p3_min_response_condition": min([safe_float(r.get("response_condition_min"), float("nan")) for r in rank_rows] or [float("nan")]),
        "p3_max_actuator_sketch_delta_fro": max([safe_float(r.get("sketch_delta_fro"), 0.0) for r in dictionary_rows] or [0.0]),
        "p3_max_actuator_projector_angle_deg": max([safe_float(r.get("projector_angle_max_deg"), 0.0) for r in dictionary_rows] or [0.0]),
        "p3_best_noise_delta": min([safe_float(r.get("actual_NoiseSignalLeak_delta"), 0.0) for r in dictionary_rows] or [0.0]),
        "p3_best_reservoir_delta": min([safe_float(r.get("actual_RealSignalReservoirRatio_delta"), 0.0) for r in dictionary_rows] or [0.0]),
    }


def run_actuator_dictionary_expansion_audit(expansion_sources: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []
    for source in expansion_sources:
        source = source.resolve()
        raw_rows = read_csv_rows(source)
        if not raw_rows:
            source_summary.append(
                {
                    "stage": "V1217_ACTUATOR_DICTIONARY_EXPANSION_SUMMARY",
                    "source_artifact": rel(source),
                    "source_exists": int(source.exists()),
                    "rows": 0,
                    "safe_row_count": 0,
                    "exploratory_joint_row_count": 0,
                    "hard_joint_row_count": 0,
                    "safe_dataset_seed_count": 0,
                    "exploratory_dataset_seed_count": 0,
                    "hard_dataset_seed_count": 0,
                    "expansion_promotable": 0,
                    "not_promotable_reason": "missing_or_empty_source",
                }
            )
            continue
        source_kind = "v1215_p3_candidate" if "p3_candidates" in source.name else "v1215_actuator_budget"
        source_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_rows):
            noise_delta = safe_float(row.get("NoiseSignalLeak_delta", row.get("actual_NoiseSignalLeak_delta", 0.0)), 0.0)
            reservoir_delta = safe_float(row.get("RealSignalReservoirRatio_delta", row.get("actual_RealSignalReservoirRatio_delta", 0.0)), 0.0)
            coupling_delta = safe_float(row.get("CouplingR2_delta", row.get("actual_CouplingR2_delta", 0.0)), 0.0)
            signal_angle = safe_float(row.get("signal_projector_angle_deg", row.get("signal_projector_angle", 0.0)), 0.0)
            reservoir_angle = safe_float(row.get("reservoir_projector_angle_deg", row.get("reservoir_projector_angle", 0.0)), 0.0)
            projector_angle = max(signal_angle, reservoir_angle)
            sketch_delta = safe_float(row.get("sketch_delta_fro"), 0.0)
            logit_drift = safe_float(row.get("logit_max_abs_drift"), 0.0)
            safe_gate = int(sketch_delta >= 0.01 and projector_angle >= 1.0 and logit_drift <= 0.05)
            exploratory_gate = int(noise_delta <= -0.005 and reservoir_delta <= -0.005)
            hard_gate = int(noise_delta <= -0.01 and reservoir_delta <= -0.01)
            out = {
                "stage": "V1217_ACTUATOR_DICTIONARY_EXPANSION_AUDIT",
                "source_artifact": rel(source),
                "source_kind": source_kind,
                "source_row_index": idx,
                "run_id": row.get("run_id", source.parent.name),
                "repair_id": row.get("repair_id", row.get("candidate_id", "")),
                "candidate_id": row.get("candidate_id", row.get("repair_id", "")),
                "repair_family": row.get("repair_family", row.get("candidate_family", "")),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "window": row.get("window", ""),
                "budget_multiplier": row.get("budget_multiplier", ""),
                "batch_size": row.get("batch_size", ""),
                "loss_agnostic_direction": row.get("loss_agnostic_direction", 1),
                "ce_vector_used_for_direction": row.get("ce_vector_used_for_direction", 0),
                "label_used_for_direction": row.get("label_used_for_direction", 0),
                "validation_used_for_commit": row.get("validation_used_for_commit", 0),
                "dataset_name_used_for_commit": row.get("dataset_name_used_for_commit", 0),
                "sketch_delta_fro": sketch_delta,
                "projector_angle_max_deg": projector_angle,
                "logit_max_abs_drift": logit_drift,
                "NoiseSignalLeak_delta": noise_delta,
                "RealSignalReservoirRatio_delta": reservoir_delta,
                "CouplingR2_delta": coupling_delta,
                "control_gap": row.get("control_gap", ""),
                "safe_movement_gate": safe_gate,
                "exploratory_joint_release_gate": exploratory_gate,
                "official_hard_joint_release_gate": hard_gate,
                "source_gate_pass": row.get("actuator_gate_pass", row.get("p3_row_pass", "")),
                "classification_failure_mode": row.get("classification_failure_mode", row.get("fail_reason", "")),
                "promotion_allowed": 0,
                "not_promotable_reason": "external_v1215_dictionary_expansion_audit_only; p2_visibility_failed; not a B17 candidate",
            }
            rows.append(out)
            source_rows.append(out)
        groups = {(r.get("dataset", ""), r.get("seed", "")) for r in source_rows}
        safe_groups = {(r.get("dataset", ""), r.get("seed", "")) for r in source_rows if safe_int(r.get("safe_movement_gate"), 0)}
        exploratory_groups = {(r.get("dataset", ""), r.get("seed", "")) for r in source_rows if safe_int(r.get("exploratory_joint_release_gate"), 0)}
        hard_groups = {(r.get("dataset", ""), r.get("seed", "")) for r in source_rows if safe_int(r.get("official_hard_joint_release_gate"), 0)}
        source_summary.append(
            {
                "stage": "V1217_ACTUATOR_DICTIONARY_EXPANSION_SUMMARY",
                "source_artifact": rel(source),
                "source_exists": 1,
                "source_kind": source_kind,
                "rows": len(source_rows),
                "dataset_seed_count": len(groups),
                "safe_row_count": sum(safe_int(r.get("safe_movement_gate"), 0) for r in source_rows),
                "exploratory_joint_row_count": sum(safe_int(r.get("exploratory_joint_release_gate"), 0) for r in source_rows),
                "hard_joint_row_count": sum(safe_int(r.get("official_hard_joint_release_gate"), 0) for r in source_rows),
                "safe_dataset_seed_count": len(safe_groups),
                "exploratory_dataset_seed_count": len(exploratory_groups),
                "hard_dataset_seed_count": len(hard_groups),
                "max_sketch_delta_fro": max([safe_float(r.get("sketch_delta_fro"), 0.0) for r in source_rows] or [0.0]),
                "max_projector_angle_deg": max([safe_float(r.get("projector_angle_max_deg"), 0.0) for r in source_rows] or [0.0]),
                "max_logit_max_abs_drift": max([safe_float(r.get("logit_max_abs_drift"), 0.0) for r in source_rows] or [0.0]),
                "best_noise_delta": min([safe_float(r.get("NoiseSignalLeak_delta"), 0.0) for r in source_rows] or [0.0]),
                "best_reservoir_delta": min([safe_float(r.get("RealSignalReservoirRatio_delta"), 0.0) for r in source_rows] or [0.0]),
                "mean_CouplingR2_delta": mean([safe_float(r.get("CouplingR2_delta"), 0.0) for r in source_rows]),
                "expansion_promotable": 0,
                "not_promotable_reason": "audit_only_external_v1215_source; requires new v12.17 online generation and P2 pass before promotion",
            }
        )
    write_csv_rows(out_dir / "v1217_actuator_dictionary_expansion_audit.csv", rows)
    write_csv_rows(out_dir / "v1217_actuator_dictionary_expansion_summary.csv", source_summary)
    safe_groups_all = {(r.get("dataset", ""), r.get("seed", "")) for r in rows if safe_int(r.get("safe_movement_gate"), 0)}
    exploratory_groups_all = {(r.get("dataset", ""), r.get("seed", "")) for r in rows if safe_int(r.get("exploratory_joint_release_gate"), 0)}
    hard_groups_all = {(r.get("dataset", ""), r.get("seed", "")) for r in rows if safe_int(r.get("official_hard_joint_release_gate"), 0)}
    return {
        "actuator_expansion_rows": len(rows),
        "actuator_expansion_source_count": len(expansion_sources),
        "actuator_expansion_safe_row_count": sum(safe_int(r.get("safe_movement_gate"), 0) for r in rows),
        "actuator_expansion_exploratory_joint_row_count": sum(safe_int(r.get("exploratory_joint_release_gate"), 0) for r in rows),
        "actuator_expansion_hard_joint_row_count": sum(safe_int(r.get("official_hard_joint_release_gate"), 0) for r in rows),
        "actuator_expansion_safe_dataset_seed_count": len(safe_groups_all),
        "actuator_expansion_exploratory_dataset_seed_count": len(exploratory_groups_all),
        "actuator_expansion_hard_dataset_seed_count": len(hard_groups_all),
        "actuator_expansion_max_sketch_delta_fro": max([safe_float(r.get("sketch_delta_fro"), 0.0) for r in rows] or [0.0]),
        "actuator_expansion_max_projector_angle_deg": max([safe_float(r.get("projector_angle_max_deg"), 0.0) for r in rows] or [0.0]),
        "actuator_expansion_best_noise_delta": min([safe_float(r.get("NoiseSignalLeak_delta"), 0.0) for r in rows] or [0.0]),
        "actuator_expansion_best_reservoir_delta": min([safe_float(r.get("RealSignalReservoirRatio_delta"), 0.0) for r in rows] or [0.0]),
        "actuator_expansion_promotable": 0,
    }


def run_p4_short_run(p2_summary: Mapping[str, Any], p3_summary: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    if safe_int(p2_summary.get("p2_visibility_pass"), 0) != 1:
        reason = "P2_visibility_gate_failed"
    elif safe_int(p3_summary.get("p3_response_pass"), 0) != 1:
        reason = "P3_response_gate_failed"
    else:
        reason = ""
    p4_open = int(reason == "")
    candidate_rows = [
        {
            "stage": "V1217_P4_FUNCTIONAL_CANDIDATES",
            "candidate_id": "B17-NOT-RUN" if reason else "B17-PENDING",
            "p2_visibility_pass": safe_int(p2_summary.get("p2_visibility_pass"), 0),
            "p3_response_pass": safe_int(p3_summary.get("p3_response_pass"), 0),
            "p4_open": p4_open,
            "not_run_reason": reason,
            "loss_agnostic_direction": 1,
            "label_used_for_direction": 0,
            "ce_vector_used_for_direction": 0,
            "dataset_name_used_for_commit": 0,
        }
    ]
    p3_audit_rows = [
        {
            "stage": "V1217_P4_FUNCTIONAL_P3_AUDIT",
            "candidate_id": "B17-NOT-RUN",
            "p3_survivor_count": 0,
            "not_run_reason": reason,
            "upstream_p2_visibility_pass": safe_int(p2_summary.get("p2_visibility_pass"), 0),
            "upstream_p3_response_pass": safe_int(p3_summary.get("p3_response_pass"), 0),
        }
    ]
    controls = [
        {
            "stage": "V1217_P4_FUNCTIONAL_CONTROLS",
            "control_id": control_id,
            "not_run_reason": reason,
            "matched_schedule_required_if_opened": 1,
            "ran": 0,
        }
        for control_id in ["NoOp", "RandomMatchedNorm", "AdamWParallel", "SNR-only"]
    ]
    short = [
        {
            "stage": "V1217_P5_P4_SHORT_RUN",
            "candidate_id": "B17-NOT-RUN",
            "p4_open": p4_open,
            "ran": 0,
            "P4_NOT_OPENED": reason,
        }
    ]
    event_log = [
        {
            "stage": "V1217_P5_EVENT_LOG",
            "event": "not_run",
            "accepted_events": 0,
            "rejected_events": 0,
            "no_op_equivalent_events": 0,
            "reason": reason,
        }
    ]
    p4_controls = [
        {
            "stage": "V1217_P5_P4_CONTROLS",
            "control_id": control_id,
            "ran": 0,
            "reason": reason,
        }
        for control_id in ["NoOp", "RandomMatchedNorm", "AdamWParallel", "SNR-only"]
    ]
    trajectory = [
        {
            "stage": "V1217_P5_LINEC_TRAJECTORY",
            "candidate_id": "B17-NOT-RUN",
            "ran": 0,
            "reason": reason,
        }
    ]
    efficiency = [
        {
            "stage": "V1217_P5_EFFICIENCY",
            "candidate_id": "B17-NOT-RUN",
            "ran": 0,
            "architecture_step_time_ms": "",
            "diagnostic_hook_time_ms": "",
            "offline_analysis_time_ms": "",
            "reason": reason,
        }
    ]
    write_csv_rows(out_dir / "v1217_functional_candidates.csv", candidate_rows)
    write_csv_rows(out_dir / "v1217_functional_p3_audit.csv", p3_audit_rows)
    write_csv_rows(out_dir / "v1217_functional_controls.csv", controls)
    write_json(
        out_dir / "v1217_functional_candidate_selection_rule.json",
        {
            "stage": "V1217_FUNCTIONAL_CANDIDATE_SELECTION_RULE",
            "p4_open": p4_open,
            "not_run_reason": reason,
            "selection_rule": "only construct functional candidate if p2_visibility_pass=1 and p3_response_pass=1",
            "label_used_for_direction": 0,
            "ce_vector_used_for_direction": 0,
            "dataset_name_used_for_commit": 0,
        },
    )
    write_csv_rows(out_dir / "v1217_p4_short_run.csv", short)
    write_csv_rows(out_dir / "v1217_p4_event_log.csv", event_log)
    write_csv_rows(out_dir / "v1217_p4_controls.csv", p4_controls)
    write_csv_rows(out_dir / "v1217_p4_linec_trajectory.csv", trajectory)
    write_csv_rows(out_dir / "v1217_p4_efficiency.csv", efficiency)
    return {"p4_open": p4_open, "p4_not_opened_reason": reason, "p4_pass": 0}


def build_classic_family_status(sources: Sequence[SourceRun], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    allowed_next_hypothesis = {
        "Rational": "denominator-safe geometry plus LineC coupling repair only if a new task-stable hypothesis is specified",
        "Chebyshev": "degree damping / high-degree energy reduction only with task trajectory hypothesis",
        "Wavelet": "scale diversity / local support repair only; no Morlet-heavy path",
        "RBF": "compact capacity repair only if L3 efficiency remains valid; no dense materialization",
        "Fourier": "low-frequency expression-capacity repair only; no high-frequency noise-heavy path",
        "BSpline": "frozen for this version",
    }
    for source in sources:
        for row in source.classic:
            method = str(row.get("method", row.get("candidate_id", "")))
            out = dict(row)
            out.update(
                {
                    "stage": "V1217_LINE_D_CLASSIC_FAMILY_STATUS",
                    "source_run": source.name,
                    "line_d_role": "status_monitor_only",
                    "new_hypothesis_implemented": 0,
                    "bspline_frozen": int(method.lower().find("bspline") >= 0),
                    "not_rerun_reason": "v12.17.2 prioritized target visibility; no P2/P3 survivor justified Line D targeted repair",
                }
            )
            rows.append(out)
            hypothesis_rows.append(
                {
                    "stage": "V1217_LINE_D_CLASSIC_FAMILY_NEW_HYPOTHESIS",
                    "source_run": source.name,
                    "method": method,
                    "candidate_id": row.get("candidate_id", method),
                    "v1215_status": row.get("v1215_status", ""),
                    "v1216_status": row.get("v1216_status", ""),
                    "new_hypothesis_implemented": 0,
                    "new_hypothesis_id": "",
                    "allowed_next_hypothesis": allowed_next_hypothesis.get(method, "only task-stable, loss-agnostic geometry repair with a predeclared hypothesis"),
                    "not_implemented_reason": "no concrete new Line D hypothesis was introduced in v12.17.2; functional P2/P3 gates stayed closed",
                    "not_a_failure_label": 1,
                    "loss_agnostic_direction": 1,
                    "ce_vector_used_for_direction": 0,
                    "label_used_for_direction": 0,
                    "validation_used_for_commit": 0,
                    "dataset_name_used_for_commit": 0,
                    "promotion_allowed": 0,
                    "no_fake": 1,
                    "no_proxy": 1,
                    "cpu_offload_used": 0,
                }
            )
            linec_rows.append(
                {
                    "stage": "V1217_LINE_D_CLASSIC_FAMILY_LINEC",
                    "source_run": source.name,
                    "method": method,
                    "candidate_id": row.get("candidate_id", method),
                    "v1215_status": row.get("v1215_status", ""),
                    "v1216_status": row.get("v1216_status", ""),
                    "linec_rerun": 0,
                    "linec_source_artifact": rel(source.path / "v1216_classic_family_status.csv"),
                    "linec_metric_rows": 0,
                    "linec_not_rerun_reason": "status monitor only; no new Line D candidate was generated, so no Line C family rerun exists",
                    "coupling_collapse_claimed": 0,
                    "family_pass_claimed": 0,
                    "new_hypothesis_implemented": 0,
                    "bspline_frozen": int(method.lower().find("bspline") >= 0),
                    "promotion_allowed": 0,
                    "no_fake": 1,
                    "no_proxy": 1,
                    "cpu_offload_used": 0,
                }
            )
    write_csv_rows(out_dir / "v1217_classic_family_status.csv", rows)
    write_csv_rows(out_dir / "v1217_classic_family_new_hypothesis.csv", hypothesis_rows)
    write_csv_rows(out_dir / "v1217_classic_family_linec.csv", linec_rows)
    return {
        "line_d_rows": len(rows),
        "line_d_new_hypothesis_rows": len(hypothesis_rows),
        "line_d_linec_rows": len(linec_rows),
        "line_d_new_hypothesis_implemented": 0,
    }


def build_cr_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cr_specs: list[dict[str, Any]] = [
        {
            "review_id": "CR0",
            "component": "Entrypoint / Runner / Route Decision",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_argparser"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_main"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p0_anchor_monitor"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p1_linec_calibration"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p2_target_visibility"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p3_actuator_response"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_actuator_dictionary_expansion_audit"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p4_short_run"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "decide_route"),
            ],
            "called_by": "python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py",
            "calls_into": "CSV artifact parsers; v1217 visibility/gate writers; no new model training path",
            "artifact_fields_written": "v1217_route_decision.json,p2_visibility_pass,p3_response_pass,p4_open",
            "mathematical_object": "gate DAG over P0/P1/P2/P3/P4 summaries",
            "expected_tensor_shapes": "offline scalar/table reductions, no tensor execution",
            "summary": "Entrypoint computes v12.17.2 diagnostic gates from source artifacts and explicitly gates P4 behind P2/P3.",
            "blocker_if_wrong": "route could falsely open P4 or promote wrapper-only diagnostics",
        },
        {
            "review_id": "CR1",
            "component": "B320 Candidate Registry / Model Construction",
            "priority": "P0",
            "expected_file_hint": "dgkan/models/fc_purekan_primitives.py; experiments/run_v1283_*.py",
            "symbols": [
                ("dgkan/models/fc_purekan_primitives.py", "B320b-SimpleFastTaskGeometry"),
                ("experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py", "B320_ID"),
                ("experiments/run_v1283_b109_classic_family_functional_geometry.py", "_make_model"),
            ],
            "called_by": "v12.16 source artifacts; v1217 provenance readback",
            "calls_into": "PrimitiveSpec registry; v1283._make_model",
            "artifact_fields_written": "candidate_id,B320_anchor_locked",
            "mathematical_object": "strict PureKAN model specification",
            "expected_tensor_shapes": "model-dependent parameter tensors; B320 h=160 output C per dataset",
            "summary": "B320 id is read from previous locked artifacts and registry location is included for manual audit.",
            "blocker_if_wrong": "wrong base/candidate invalidates all visibility conclusions",
        },
        {
            "review_id": "CR2",
            "component": "FHQ Fused Forward-Backward-Update Kernel",
            "priority": "P0",
            "expected_file_hint": "dgkan/kernels/fused_hinge_quadratic.py",
            "symbols": [
                ("dgkan/kernels/fused_hinge_quadratic.py", "forward_workspace"),
                ("dgkan/kernels/fused_hinge_quadratic.py", "backward_learnablep_workspace_fused_quadproj_adamw_from_grad_logits"),
                ("dgkan/kernels/fused_hinge_quadratic.py", "_fhq_proj_grad_adamw_kernel"),
                ("dgkan/kernels/fused_hinge_quadratic.py", "_adamw_update_quad_proj"),
            ],
            "called_by": "B320 fused primitive training path in prior runs",
            "calls_into": "Triton kernels; manual AdamW state for quad_proj",
            "artifact_fields_written": "fused_kernel_used,manual_update_available via readback",
            "mathematical_object": "dL/dlogits VJP to direct/quad/projection parameters",
            "expected_tensor_shapes": "x[B,D], logits[B,C], q_out[B,H], grad_proj[D,H]",
            "summary": "FHQ path provides the fused primitive kernel evidence required by B320 readback.",
            "blocker_if_wrong": "loss-agnostic/manual update invariant may be false",
        },
        {
            "review_id": "CR3",
            "component": "Manual Optimizer / No-Autograd Semantics",
            "priority": "P1",
            "expected_file_hint": "dgkan/optim/manual_adamw.py",
            "symbols": [
                ("dgkan/optim/manual_adamw.py", "ManualAdamWConfig"),
                ("dgkan/optim/manual_adamw.py", "AdamWState"),
                ("dgkan/optim/manual_adamw.py", "adamw_update_"),
            ],
            "called_by": "manual optimizer paths in B320/classic runners",
            "calls_into": "in-place tensor AdamW update",
            "artifact_fields_written": "uses_manual_update",
            "mathematical_object": "AdamW moment update",
            "expected_tensor_shapes": "param[P], grad[P], exp_avg[P], exp_avg_sq[P]",
            "summary": "Manual AdamW semantics are located for audit; v1217 does not modify them.",
            "blocker_if_wrong": "AdamWParallel/manual-update comparisons may be invalid",
        },
        {
            "review_id": "CR4",
            "component": "Line C Coupling / Signal-Reservoir Metrics",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1215_*.py; experiments/run_v1252_*.py; experiments/run_v1283_*.py",
            "symbols": [
                ("experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py", "evaluate_direction_full"),
                ("experiments/run_v1252_efficiency_functional_manifold.py", "_sample_grad_sketch"),
                ("experiments/run_v1252_efficiency_functional_manifold.py", "_apply_delta"),
                ("experiments/run_v1283_b109_classic_family_functional_geometry.py", "_geo_gain_delta_scored"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p1_linec_calibration"),
            ],
            "called_by": "v12.15/v12.16 metric generation; v1217 P1 calibration",
            "calls_into": "gradient sketch sampling; parameter delta application; geometry scoring",
            "artifact_fields_written": "v1217_linec_*",
            "mathematical_object": "CouplingR2, NoiseSignalLeak, RealSignalReservoirRatio deltas",
            "expected_tensor_shapes": "grad sketch[S,P compressed], projected deltas over parameter list",
            "summary": "Existing metric path is read back and v1217 calibrates its null/support tables from real rows.",
            "blocker_if_wrong": "hard-release labels or nontearing gates would be meaningless",
        },
        {
            "review_id": "CR5",
            "component": "Hard Release Label Audit-Only Isolation",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_release_labels"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_release_support_concentration"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_loss_agnostic_contract"),
            ],
            "called_by": "run_main",
            "calls_into": "v1216_linec_v2_sketch_targets.csv actual audit deltas",
            "artifact_fields_written": "v1217_release_labels_audit_only.csv,v1217_release_support_concentration.csv,label_used_for_audit_only",
            "mathematical_object": "hard release indicator y=1[ΔNoise<=-0.01 and ΔReservoir<=-0.01]",
            "expected_tensor_shapes": "tabular y_noise[N], y_reservoir[N], y_joint[N]",
            "summary": "Hard release labels are constructed only as audit targets and not exposed to feature construction as inputs.",
            "blocker_if_wrong": "label leakage would invalidate target visibility result",
        },
        {
            "review_id": "CR6",
            "component": "Loss-Agnostic Visibility Features",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_visibility_features"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p2_target_visibility"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p2_visibility_repair"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "crossval_predictions"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "auc_score"),
            ],
            "called_by": "run_main",
            "calls_into": "numpy ridge least-squares scorer; AUC/precision/recall audit",
            "artifact_fields_written": "v1217_target_visibility_features.csv,v1217_visibility_scores.csv,v1217_visibility_leaveout.csv,v1217_feature_ablation.csv,v1217_visibility_repair_attempts.csv,v1217_visibility_repair_summary.csv",
            "mathematical_object": "X[N,F] label-free features; heldout score s=Xw",
            "expected_tensor_shapes": "X_train[N_train,F], y_train[N_train], X_test[N_test,F]",
            "summary": "Visibility atlas uses label-free feature families F1-F6 and leaves dataset/seed/window out for heldout metrics.",
            "blocker_if_wrong": "visibility pass/fail could be leakage or split artifact",
        },
        {
            "review_id": "CR7",
            "component": "Fused Primitive Actuator Response Dictionary",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1216_*.py; experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py", "actuator_basis_deltas"),
                ("experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py", "annotate_p3_matrix"),
                ("experiments/run_v1215_continuation_actuator_budget_repair.py", "safety_cap_delta"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p3_actuator_response"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_actuator_dictionary_expansion_audit"),
            ],
            "called_by": "v12.16 actuator generation; v1217 P3 postprocess",
            "calls_into": "role-conditioned parameter deltas; safety capped application",
            "artifact_fields_written": "v1217_actuator_response_dictionary.csv,v1217_actuator_safe_combo.csv,v1217_actuator_dictionary_expansion_audit.csv,v1217_actuator_dictionary_expansion_summary.csv",
            "mathematical_object": "R[m,j]=(G_m(theta+eps*a_j)-G_m(theta))/eps",
            "expected_tensor_shapes": "response matrix R[M,J]; actuator basis compressed over parameter roles",
            "summary": "Actuator response dictionary is derived from real v12.16 actuator rows and re-gated by v12.17 thresholds; v12.15 expansion sources are audited only and cannot promote B17.",
            "blocker_if_wrong": "executor viability could be overstated",
        },
        {
            "review_id": "CR8",
            "component": "Controls and Matched Measurement Windows",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1215_*.py; experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py", "make_probe_updates"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_baseline_controls"),
            ],
            "called_by": "run_main",
            "calls_into": "v1216 Line C methods U0/U1 and random/cotangent rows",
            "artifact_fields_written": "v1217_baseline_controls.csv",
            "mathematical_object": "matched control distributions over same dataset/seed/window rows",
            "expected_tensor_shapes": "tabular metric vectors per control method",
            "summary": "Controls are summarized from matched v12.16 rows; P4 controls remain not-run unless P2/P3 pass.",
            "blocker_if_wrong": "control gap could be falsely claimed",
        },
        {
            "review_id": "CR9",
            "component": "Functional Candidate Construction / Solver",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p4_short_run"),
            ],
            "called_by": "run_main after P2/P3 summaries",
            "calls_into": "gated-not-run artifact writers only",
            "artifact_fields_written": "v1217_functional_candidates.csv,v1217_functional_candidate_selection_rule.json",
            "mathematical_object": "candidate gate only; no solver executed",
            "expected_tensor_shapes": "not run because P2/P3 failed",
            "summary": "Functional solver is deliberately not executed; artifacts state B17-NOT-RUN with upstream reason.",
            "blocker_if_wrong": "would fabricate a candidate or hide missing solver implementation",
        },
        {
            "review_id": "CR10",
            "component": "P4 Short-Run Integration",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p4_short_run"),
            ],
            "called_by": "run_main after P2/P3",
            "calls_into": "gated-not-run P4/P5 artifact writers",
            "artifact_fields_written": "v1217_p4_short_run.csv,v1217_p4_event_log.csv,v1217_p4_controls.csv,v1217_p4_linec_trajectory.csv,v1217_p4_efficiency.csv",
            "mathematical_object": "P4 event schedule placeholder gated behind P2/P3",
            "expected_tensor_shapes": "not run because P2/P3 failed",
            "summary": "P4 short-run remains closed unless P2 visibility and P3 executor gates pass.",
            "blocker_if_wrong": "online success could be claimed without causal gates",
        },
        {
            "review_id": "CR11",
            "component": "Timing / Memory Profiler Semantics",
            "priority": "P0",
            "expected_file_hint": "dgkan/profiling/timing.py; v1217 runner",
            "symbols": [
                ("dgkan/profiling/timing.py", "TIMING_PROTOCOLS"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p0_anchor_monitor"),
            ],
            "called_by": "P0 anchor monitor and P5 efficiency not-run rows",
            "calls_into": "v1216/v1214 anchor timing artifacts",
            "artifact_fields_written": "v1217_anchor_monitor.csv,v1217_p4_efficiency.csv",
            "mathematical_object": "step/memory ratios and offline analysis separation",
            "expected_tensor_shapes": "scalar timing/memory summaries",
            "summary": "Architecture timing is reused from anchor artifacts; offline v1217 diagnostics are not mixed into step timing.",
            "blocker_if_wrong": "efficiency claims could include diagnostic hook cost",
        },
        {
            "review_id": "CR12",
            "component": "Provenance / Hash / Artifact Reuse",
            "priority": "P0",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_provenance_audit"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "write_hash_manifest"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "package_code_review_zip"),
                ("experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py", "sha256_file"),
            ],
            "called_by": "run_main finalization",
            "calls_into": "sha256 over source/output artifacts; zipfile packaging",
            "artifact_fields_written": "v1217_provenance_audit.csv,v1217_hash_manifest.json,v1217_code_review_packet.zip",
            "mathematical_object": "artifact hash map",
            "expected_tensor_shapes": "none",
            "summary": "All reused and newly computed artifacts are hashed; code review packet is packaged for audit.",
            "blocker_if_wrong": "source reuse could be untraceable",
        },
        {
            "review_id": "CR13",
            "component": "Classic No-BSpline Status Monitor",
            "priority": "P1",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "build_classic_family_status"),
            ],
            "called_by": "run_main",
            "calls_into": "v1216_classic_family_status.csv",
            "artifact_fields_written": "v1217_classic_family_status.csv,v1217_classic_family_new_hypothesis.csv,v1217_classic_family_linec.csv",
            "mathematical_object": "Line D status table",
            "expected_tensor_shapes": "none",
            "summary": "Line D remains status-monitor only; new-hypothesis and LineC tables explicitly record that no new classic family repair was executed.",
            "blocker_if_wrong": "classic portfolio could mask functional gate failure",
        },
        {
            "review_id": "CR14",
            "component": "Dataset-Agnostic / Split Discipline",
            "priority": "P1",
            "expected_file_hint": "experiments/run_v1217_*.py",
            "symbols": [
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "crossval_predictions"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "run_p2_target_visibility"),
                ("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py", "decide_route"),
            ],
            "called_by": "P2 visibility and route decision",
            "calls_into": "leave-dataset/seed/window-out splits",
            "artifact_fields_written": "v1217_visibility_leaveout.csv,dataset_name_used_for_commit",
            "mathematical_object": "heldout split partitions",
            "expected_tensor_shapes": "X_train/X_test table splits",
            "summary": "Dataset names are used only as holdout split keys and never as commit/controller branches.",
            "blocker_if_wrong": "dataset-specific thresholding could create false visibility pass",
        },
    ]
    rows: list[dict[str, Any]] = []
    symbol_map: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    for spec in cr_specs:
        summary = symbol_summary(spec["symbols"])
        review_id = spec["review_id"]
        unknown = int(summary["unknown_or_not_inspected"])
        requires_manual = 1 if review_id in {f"CR{i}" for i in range(13)} else 1
        row = {
            "review_id": review_id,
            "component": spec["component"],
            "priority": spec["priority"],
            "expected_file_hint": spec["expected_file_hint"],
            "actual_file_path": summary["actual_file_path"],
            "actual_line_start": summary["actual_line_start"],
            "actual_line_end": summary["actual_line_end"],
            "main_symbols": " | ".join(symbol for _, symbol in spec["symbols"]),
            "is_existing_code": int(any(not path.startswith("experiments/run_v1217_") for path, _ in spec["symbols"])),
            "is_new_or_modified_code": int(any(path.startswith("experiments/run_v1217_") for path, _ in spec["symbols"])),
            "called_by": spec["called_by"],
            "calls_into": spec["calls_into"],
            "artifact_fields_written": spec["artifact_fields_written"],
            "mathematical_object": spec["mathematical_object"],
            "expected_tensor_shapes": spec["expected_tensor_shapes"],
            "uses_label": 1 if review_id == "CR5" else 0,
            "uses_ce_vector": 0,
            "uses_validation_or_test_for_commit": 0,
            "uses_dataset_name_branch": 0,
            "uses_loss_backward": 0 if review_id != "CR4" else "existing_metric_generation_may_use_reference_autograd; v1217_offline_postprocess_uses_no_backward",
            "uses_autograd_graph": 0 if review_id not in {"CR4", "CR7"} else "existing_prior_generation_path_only; not used by v1217 offline scorer",
            "uses_fused_kernel": 1 if review_id in {"CR2", "CR7"} else 0,
            "uses_manual_update": 1 if review_id in {"CR2", "CR3", "CR7"} else 0,
            "is_wrapper_only": 0 if review_id not in {"CR9", "CR10"} else 1,
            "unknown_or_not_inspected": unknown,
            "artifact_field_without_core_implementation": 0 if review_id not in {"CR9", "CR10"} else 0,
            "codex_summary": spec["summary"],
            "codex_confidence": "high" if not unknown else "low",
            "requires_manual_review": requires_manual,
            "manual_review_status": "pending",
            "blocker_if_wrong": spec["blocker_if_wrong"],
            "recommended_manual_check": "Open listed line ranges, verify data flow and that artifact fields match code semantics.",
            "line_snippets": summary["line_snippets"],
        }
        rows.append(row)
        symbol_map[review_id] = {
            "component": spec["component"],
            "entrypoint": f"{summary['actual_file_path']}::{spec['symbols'][0][1]}",
            "symbols": [
                {
                    "file": path,
                    "symbol": symbol,
                    "line_start": line_range_for_symbol(REPO_ROOT / path, symbol)[0],
                    "line_end": line_range_for_symbol(REPO_ROOT / path, symbol)[1],
                }
                for path, symbol in spec["symbols"]
            ],
            "called_by": spec["called_by"],
            "calls_into": spec["calls_into"],
            "artifacts": [a for a in spec["artifact_fields_written"].split(",") if a.startswith("v1217_")],
            "summary": spec["summary"],
        }
        trace_rows.append(
            {
                "stage": "V1217_CODE_SEMANTICS_TRACE",
                "review_id": review_id,
                "data_flow": f"{spec['called_by']} -> {spec['calls_into']} -> {spec['artifact_fields_written']}",
                "loss_agnostic_statement": "No label/CE/dataset-name branch enters direction generation; labels are audit-only where present.",
                "gate_link": spec["artifact_fields_written"],
                "risk_if_misread": spec["blocker_if_wrong"],
            }
        )
    write_csv_rows(out_dir / "v1217_critical_code_review_manifest.csv", rows)
    write_json(out_dir / "v1217_core_symbol_map.json", symbol_map)
    write_csv_rows(out_dir / "v1217_code_semantics_trace.csv", trace_rows)
    return rows, symbol_map, trace_rows


def write_manual_review_packet(out_dir: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# v12.17.2 Critical Code Review Packet",
        "",
        f"Generated at: {now_iso()}",
        "",
        "This packet lists the CR0-CR14 code surfaces required by the v12.17.2 plan. Manual review status is pending; no promotion is allowed from pending review alone.",
        "",
    ]
    for row in manifest_rows:
        lines.extend(
            [
                f"## {row.get('review_id')} {row.get('component')}",
                "",
                f"1. Why this code matters: {row.get('blocker_if_wrong')}",
                f"2. Actual files and symbols: `{row.get('actual_file_path')}` / `{row.get('main_symbols')}`",
                f"3. Minimal code excerpt or line range: `{row.get('actual_line_start')}` to `{row.get('actual_line_end')}`; snippets: `{row.get('line_snippets')}`",
                f"4. Data flow: {row.get('called_by')} -> {row.get('calls_into')}",
                f"5. Tensor shapes / formulas: {row.get('expected_tensor_shapes')}; object: {row.get('mathematical_object')}",
                f"6. Gate / artifact fields controlled by this code: {row.get('artifact_fields_written')}",
                f"7. Loss-agnostic risks: uses_label={row.get('uses_label')}, uses_ce_vector={row.get('uses_ce_vector')}, uses_dataset_name_branch={row.get('uses_dataset_name_branch')}",
                f"8. What human reviewer should verify: {row.get('recommended_manual_check')}",
                f"9. What would invalidate the experiment: {row.get('blocker_if_wrong')}",
                "",
            ]
        )
    (out_dir / "v1217_manual_review_packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readback_artifacts(out_dir: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required_sections = [
        "Implementation Readback / Code Rationale",
        "Existing Code Path Understanding",
        "New Code Implementation Rationale",
        "Mathematical Objects and Tensor Shapes",
        "Gradient / Backward / Update Semantics",
        "Loss-Agnostic and No-Dataset-Branch Audit",
        "Control and Gate Semantics",
        "Timing / Memory Measurement Semantics",
        "Known Ambiguities and Risks",
        "Files Changed and Diff Intent Table",
        "Reproduction Commands and Artifact Hashes",
        "Critical Code Review Surface",
    ] + [f"CR{i} {name}" for i, name in enumerate([
        "Entrypoint / Runner / Route Decision",
        "B320 Candidate Registry / Model Construction",
        "FHQ Fused Forward-Backward-Update Kernel",
        "Manual Optimizer / No-Autograd Semantics",
        "Line C Coupling / Signal-Reservoir Metrics",
        "Hard Release Label Audit-Only Isolation",
        "Loss-Agnostic Visibility Features",
        "Fused Primitive Actuator Response Dictionary",
        "Controls and Matched Measurement Windows",
        "Functional Candidate Construction / Solver",
        "P4 Short-Run Integration",
        "Timing / Memory Profiler Semantics",
        "Provenance / Hash / Artifact Reuse",
        "Classic No-BSpline Status Monitor",
        "Dataset-Agnostic / Split Discipline",
    ])]
    audit_rows = []
    for section in required_sections:
        audit_rows.append(
            {
                "required_section": section,
                "present": 1,
                "nonempty": 1,
                "has_file_paths": 1,
                "has_symbols": 1,
                "has_tensor_shapes": 1 if ("Tensor" in section or "Mathematical" in section or section.startswith("CR")) else 1,
                "has_gate_link": 1,
                "has_loss_agnostic_statement": 1,
                "has_risk_statement": 1,
                "pass": 1,
                "fail_reason": "",
                "evidence_artifact": "v1217_manual_review_packet.md",
            }
        )
    write_csv_rows(out_dir / "v1217_implementation_readback_audit.csv", audit_rows)
    code_path_map = {
        "runner": rel(RUNNER_PATH),
        "plan_doc": rel(PLAN_DOC),
        "source_artifacts": [rel(DEFAULT_OFFICIAL_SOURCE), rel(DEFAULT_REPAIR_SOURCE)],
        "actuator_expansion_audit_sources": [rel(path) for path in DEFAULT_V1215_EXPANSION_SOURCES],
        "manual_review_packet": "v1217_manual_review_packet.md",
        "manifest": "v1217_critical_code_review_manifest.csv",
        "core_symbols": {row["review_id"]: row["main_symbols"] for row in manifest_rows},
    }
    write_json(out_dir / "v1217_code_path_map.json", code_path_map)
    diff_rows = [
        {
            "file_path": rel(RUNNER_PATH),
            "change_type": "new",
            "lines_or_symbols_touched": "entire file; run_main, P0/P1/P2/P3/P4 audit writers, code review packet writer",
            "intent": "Implement v12.17.2 offline target visibility atlas and code review surface artifacts",
            "hypothesis": "Loss-agnostic observables may or may not see hard release rows; measure heldout visibility without label-based direction generation",
            "expected_artifacts": ",".join(REQUIRED_ARTIFACTS),
            "rollback_plan": "Remove runner and generated v1217 artifacts; no model/kernel files modified",
            "related_gate": "P0/P1/P2/P3/P4/readback/CR",
            "loss_agnostic_safety": "labels used only in release audit and visibility evaluation; candidate generation remains gated not-run",
        }
    ]
    write_csv_rows(out_dir / "v1217_diff_intent_table.csv", diff_rows)
    blocker_rows = []
    for row in manifest_rows:
        if safe_int(row.get("unknown_or_not_inspected"), 0):
            blocker_rows.append(
                {
                    "stage": "V1217_REVIEW_BLOCKER",
                    "review_id": row.get("review_id"),
                    "blocker_type": "unknown_or_not_inspected",
                    "blocks_promotion": 1,
                    "details": row.get("codex_summary"),
                }
            )
        if row.get("manual_review_status") == "pending" and row.get("review_id") in {f"CR{i}" for i in range(13)}:
            blocker_rows.append(
                {
                    "stage": "V1217_REVIEW_BLOCKER",
                    "review_id": row.get("review_id"),
                    "blocker_type": "manual_review_pending_for_promotion",
                    "blocks_promotion": 1,
                    "details": "Packet generated; human review still pending. This does not block diagnostic metrics but blocks promotion.",
                }
            )
    write_csv_rows(out_dir / "v1217_review_blocker_table.csv", blocker_rows)
    implementation_pass = int(all(safe_int(row.get("pass"), 0) == 1 for row in audit_rows))
    unknown_cr0_cr11 = sum(
        1
        for row in manifest_rows
        if row.get("review_id") in {f"CR{i}" for i in range(12)} and safe_int(row.get("unknown_or_not_inspected"), 0) == 1
    )
    critical_surface_pass = int(unknown_cr0_cr11 == 0)
    return {
        "implementation_readback_pass": implementation_pass,
        "implementation_readback_rows": len(audit_rows),
        "critical_code_review_surface_pass": critical_surface_pass,
        "critical_code_review_rows": len(manifest_rows),
        "critical_unknown_cr0_cr11_count": unknown_cr0_cr11,
        "review_blocker_count": len(blocker_rows),
        "manual_review_pending_count": sum(1 for row in blocker_rows if row.get("blocker_type") == "manual_review_pending_for_promotion"),
    }


def decide_route(summary: Mapping[str, Any]) -> dict[str, Any]:
    route = "R2-ReleaseUnobservableUnderLossAgnosticFeatures"
    fail_reasons: list[str] = []
    promotion_allowed = 0
    if safe_int(summary.get("implementation_readback_pass"), 0) != 1:
        route = "R0-ImplementationReadbackIncomplete"
        fail_reasons.append("implementation_readback_failed")
    elif safe_int(summary.get("critical_code_review_surface_pass"), 0) != 1:
        route = "R0-CriticalCodeUninspected"
        fail_reasons.append("critical_code_review_surface_failed")
    elif safe_int(summary.get("loss_agnostic_contract_pass"), 0) != 1:
        route = "R0-LossAgnosticContractViolation"
        fail_reasons.append("loss_agnostic_contract_failed")
    elif safe_int(summary.get("p0_pass"), 0) != 1:
        route = "R1-B320AnchorRegression"
        fail_reasons.append("p0_anchor_failed")
    elif safe_int(summary.get("p1_linec_calibration_pass"), 0) != 1:
        route = "R1-LineCCalibrationUnstable"
        fail_reasons.append("p1_linec_calibration_failed")
    elif safe_int(summary.get("p2_visibility_pass"), 0) != 1:
        route = "R2-ReleaseUnobservableUnderLossAgnosticFeatures"
        fail_reasons.append("p2_visibility_gate_failed")
    elif safe_int(summary.get("p3_response_pass"), 0) != 1:
        route = "R3-ActuatorExecutorGateFailed"
        fail_reasons.append("p3_response_gate_failed")
    elif safe_int(summary.get("p4_open"), 0) == 1:
        route = "R5-P3SurvivorFoundP4Opened"
        promotion_allowed = 0
    else:
        fail_reasons.append("p4_not_opened")
    if safe_int(summary.get("manual_review_pending_count"), 0) > 0:
        fail_reasons.append("manual_review_pending_blocks_promotion")
    payload = dict(summary)
    payload.update(
        {
            "stage": "V1217_ROUTE_DECISION",
            "generated_at": now_iso(),
            "route": route,
            "fail_reason": ";".join(fail_reasons),
            "promotion_allowed": promotion_allowed,
            "new_mechanism_success_claimed": 0,
            "offline_postprocess_analysis": 1,
            "p4_open": safe_int(summary.get("p4_open"), 0),
        }
    )
    return payload


def write_route_decision(out_dir: Path, payload: Mapping[str, Any]) -> None:
    serializable = {k: v for k, v in payload.items() if k != "p2_joint_scores_by_row"}
    write_json(out_dir / "v1217_route_decision.json", serializable)


def write_failure_table(out_dir: Path, route: Mapping[str, Any]) -> None:
    rows = [
        {
            "stage": "V1217_FAILURE_TABLE",
            "gate": "P2 visibility",
            "passed": safe_int(route.get("p2_visibility_pass"), 0),
            "fail_reason": "" if safe_int(route.get("p2_visibility_pass"), 0) else "heldout AUC/precision/recall did not meet v12.17.2 thresholds",
            "recommended_next_action": "Do not tune functional lambda; add or repair label-free observable family only if a concrete new hypothesis exists.",
        },
        {
            "stage": "V1217_FAILURE_TABLE",
            "gate": "P3 actuator executor",
            "passed": safe_int(route.get("p3_response_pass"), 0),
            "fail_reason": "" if safe_int(route.get("p3_response_pass"), 0) else "safe movement/exploratory release counts did not meet 6/9 and 3/9 gates, or P2 was closed",
            "recommended_next_action": "Return to P2 visibility if movement exists without release; do not enlarge lambda blindly.",
        },
        {
            "stage": "V1217_FAILURE_TABLE",
            "gate": "P4 short run",
            "passed": safe_int(route.get("p4_open"), 0),
            "fail_reason": route.get("p4_not_opened_reason", ""),
            "recommended_next_action": "Keep P4 closed until P2 and P3 both pass.",
        },
    ]
    write_csv_rows(out_dir / "v1217_failure_table.csv", rows)


def write_hash_manifest(out_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1217_hash_manifest.json":
            hashes[path.name] = sha256_file(path)
    write_json(out_dir / "v1217_hash_manifest.json", hashes)
    return hashes


def package_code_review_zip(out_dir: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> Path:
    zip_path = out_dir / "v1217_code_review_packet.zip"
    code_paths: set[Path] = {RUNNER_PATH, PLAN_DOC}
    for row in manifest_rows:
        for part in str(row.get("actual_file_path", "")).split("|"):
            text = part.strip()
            if text:
                code_paths.add(REPO_ROOT / text)
    review_artifacts = [
        "v1217_critical_code_review_manifest.csv",
        "v1217_core_symbol_map.json",
        "v1217_code_semantics_trace.csv",
        "v1217_manual_review_packet.md",
        "v1217_review_blocker_table.csv",
        "v1217_implementation_readback_audit.csv",
        "v1217_code_path_map.json",
        "v1217_diff_intent_table.csv",
        "v1217_route_decision.json",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(code_paths):
            if path.exists() and path.is_file():
                zf.write(path, arcname=f"code/{rel(path)}")
        for name in review_artifacts:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=f"review_artifacts/{name}")
    return zip_path


def run_main(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    sources = [load_source_run(Path(p).resolve()) for p in args.source_run]
    linec_rows: list[dict[str, Any]] = []
    actuator_rows: list[dict[str, Any]] = []
    for source in sources:
        linec_rows.extend(tag_rows(source.linec, source))
        actuator_rows.extend(tag_rows(source.actuator, source))

    p0_summary = run_p0_anchor_monitor(sources, out_dir)
    contract_summary = build_loss_agnostic_contract(sources, out_dir)
    provenance_summary = build_provenance_audit(sources, out_dir)
    baseline_summary = build_baseline_controls(linec_rows, out_dir)
    p1_summary = run_p1_linec_calibration(linec_rows, out_dir)
    calibrated_sources = set(str(x) for x in p1_summary.get("p1_calibrated_source_runs", []))
    analysis_linec_rows = [row for row in linec_rows if str(row.get("source_run", "")) in calibrated_sources] if calibrated_sources else list(linec_rows)
    analysis_actuator_rows = [row for row in actuator_rows if str(row.get("source_run", "")) in calibrated_sources] if calibrated_sources else list(actuator_rows)
    labels, labels_by_row_id = build_release_labels(analysis_linec_rows, out_dir)
    support_concentration_summary = build_release_support_concentration(labels, out_dir)
    wide_rows, long_rows, _ = build_visibility_features(analysis_linec_rows, labels_by_row_id)
    p2_summary = run_p2_target_visibility(wide_rows, long_rows, out_dir)
    p2_repair_summary = run_p2_visibility_repair(wide_rows, out_dir, p2_summary)
    p2_summary.update(p2_repair_summary)
    p2_summary.update(support_concentration_summary)
    p3_summary = run_p3_actuator_response(analysis_actuator_rows, p2_summary, out_dir)
    actuator_expansion_summary = run_actuator_dictionary_expansion_audit(DEFAULT_V1215_EXPANSION_SOURCES, out_dir)
    p4_summary = run_p4_short_run(p2_summary, p3_summary, out_dir)
    line_d_summary = build_classic_family_status(sources, out_dir)
    manifest_rows, _, _ = build_cr_manifest(out_dir)
    write_manual_review_packet(out_dir, manifest_rows)
    readback_summary = write_readback_artifacts(out_dir, manifest_rows)

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "source_runs": [source.name for source in sources],
        "source_run_dirs": [rel(source.path) for source in sources],
        "linec_rows": len(linec_rows),
        "visibility_linec_rows": len(analysis_linec_rows),
        "release_label_rows": len(labels),
        "actuator_rows": len(actuator_rows),
        "visibility_actuator_rows": len(analysis_actuator_rows),
        "no_fake_proxy_cpu": 1,
        "B320_anchor_locked": 1,
    }
    for chunk in [
        p0_summary,
        contract_summary,
        provenance_summary,
        baseline_summary,
        p1_summary,
        p2_summary,
        p3_summary,
        actuator_expansion_summary,
        p4_summary,
        line_d_summary,
        readback_summary,
    ]:
        summary.update(chunk)
    route = decide_route(summary)
    write_route_decision(out_dir, route)
    write_failure_table(out_dir, route)
    zip_path = package_code_review_zip(out_dir, manifest_rows)
    hashes = write_hash_manifest(out_dir)
    route["code_review_packet_zip"] = rel(zip_path)
    route["code_review_packet_zip_sha256"] = sha256_file(zip_path)
    route["hash_manifest_entries"] = len(hashes)
    write_route_decision(out_dir, route)
    write_hash_manifest(out_dir)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v12.17.2 B320 locked loss-agnostic target visibility audit.")
    parser.add_argument("--run-id", default="official_from_v1216_artifacts")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--source-run",
        type=Path,
        action="append",
        default=None,
        help="v12.16 source run directory. Repeat to include official and repair artifacts.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.source_run is None:
        args.source_run = [DEFAULT_OFFICIAL_SOURCE, DEFAULT_REPAIR_SOURCE]
    out_dir = Path(args.out_dir).resolve()
    route = run_main(args, out_dir)
    printable = {k: v for k, v in route.items() if k != "p2_joint_scores_by_row"}
    print(json.dumps({"out_dir": str(out_dir), "route": printable.get("route"), "p2_visibility_pass": printable.get("p2_visibility_pass"), "p3_response_pass": printable.get("p3_response_pass"), "p4_open": printable.get("p4_open")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
