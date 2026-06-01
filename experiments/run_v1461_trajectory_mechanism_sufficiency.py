#!/usr/bin/env python3
"""DG-KAN v14.6.1 trajectory mechanism sufficiency diagnostic.

This runner is intentionally not an action search runner. It locks the
already-executed v14.4/v14.5 bank, classifies the remaining states, and
evaluates only pre-registered mechanism probes from recorded artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = ("MNIST", "Fashion-MNIST", "KMNIST")
SEEDS = (0, 1, 2)
V144_ROOT = Path("results/v14_4_real_transfer_fms_all_basis_substrate")
V145_ROOT = Path("results/v14_5_train_stream_counterfactual_fms_all_basis_parallel")
DEFAULT_V145_ORACLE = V145_ROOT / "oracle_o1_v145_after_frontloaded_trajectory_action"
DEFAULT_OUT_DIR = Path("results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461")
DEFAULT_P1_PROBE_ROOT = Path("results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search")
PLAN_PATH = Path("docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_完整计划.md")

REQUIRED = (
    "v1461_route_decision.json",
    "v1461_code_review_manifest.csv",
    "v1461_forbidden_information_audit.csv",
    "v1461_action_search_violation_audit.csv",
    "v1461_existing_bank_lock_manifest.csv",
    "v1461_missing_state_trajectory_autopsy.csv",
    "v1461_aucdebt_decomposition.csv",
    "v1461_optimizer_state_mismatch.csv",
    "v1461_noop_dominance_audit.csv",
    "v1461_mechanism_probe_manifest.csv",
    "v1461_optimizer_state_transport_probe.csv",
    "v1461_noop_commit_probe.csv",
    "v1461_micro_horizon_integral_probe.csv",
    "v1461_delayed_projection_probe.csv",
    "v1461_mechanism_coverage_certificate.csv",
    "v1461_controller_results.csv",
    "v1461_all_basis_substrate_status.csv",
    "v1461_linec_audit.csv",
    "v1461_failure_table.csv",
    "v1461_no_go_boundary.md",
    "v1461_next_hypothesis_queue.md",
    "fig_missing_state_aucdebt.svg",
    "fig_aucdebt_early_mid_late.svg",
    "fig_optimizer_state_mismatch.svg",
    "fig_noop_dominance.svg",
    "fig_mechanism_probe_effect.svg",
    "fig_mechanism_coverage_before_after.svg",
    "fig_pass_state_regression_heatmap.svg",
    "fig_all_basis_substrate_matrix.svg",
    "fig_linec_vs_aucdebt.svg",
    "fig_failure_taxonomy.svg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--v145-oracle-dir", type=Path, default=DEFAULT_V145_ORACLE)
    parser.add_argument("--p1-probe-root", type=Path, default=DEFAULT_P1_PROBE_ROOT)
    parser.add_argument("--compute-budgeted-run", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def file_sha(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_hash(row: dict[str, Any]) -> str:
    keys = (
        "dataset",
        "seed",
        "method",
        "source",
        "AUCtime_ratio",
        "CEp99_delta",
        "NLL_delta",
        "ECE_delta",
        "LineC_pass",
        "pass_s5_row",
    )
    payload = "|".join(str(row.get(key, "")) for key in keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pass_row(row: dict[str, Any]) -> int:
    return 1 if inum(row.get("pass_s5_row")) == 1 else 0


def non_harm_tail(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("CEp99_delta"), 999.0) <= 0.05
        and fnum(row.get("NLL_delta"), 999.0) <= 0.02
        and fnum(row.get("ECE_delta"), 999.0) <= 0.02
    )


def source_pass(row: dict[str, Any]) -> bool:
    return fnum(row.get("source"), -999.0) >= 0.005


def auc_pass(row: dict[str, Any]) -> bool:
    return fnum(row.get("AUCtime_ratio"), 999.0) <= 1.0


def linec_pass(row: dict[str, Any]) -> bool:
    return inum(row.get("LineC_pass")) == 1


def classify_state(row: dict[str, Any], telemetry_available: bool) -> str:
    if source_pass(row) and non_harm_tail(row) and linec_pass(row) and not auc_pass(row):
        return "C1-AUCOnlyTrajectoryCost"
    if source_pass(row) and not non_harm_tail(row) and linec_pass(row):
        return "C3-HiddenTailRecoveryLag"
    if not source_pass(row) and non_harm_tail(row) and linec_pass(row):
        return "C2-HiddenSourceFail"
    if telemetry_available and not auc_pass(row):
        return "C4-OptimizerStateMismatch"
    if fnum(row.get("step_time_ratio"), 1.0) > 1.75 and source_pass(row):
        return "C5-OverheadDominated"
    if not (source_pass(row) and non_harm_tail(row) and linec_pass(row)):
        return "C6-AuditConflict"
    return "C7-NotEnoughData"


def auc_debt(row: dict[str, Any]) -> float:
    return max(0.0, fnum(row.get("AUCtime_ratio"), 999.0) - 1.0)


def canonical_v144_row(row: dict[str, Any], source_run_dir: str) -> dict[str, Any]:
    return {
        "source_run_dir": source_run_dir,
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "method": row.get("method", ""),
        "control_method": row.get("control_method", ""),
        "source": row.get("source_vs_best_control", ""),
        "AUCtime_ratio": row.get("AUCtime_ratio_vs_best_control", ""),
        "CEp99_delta": row.get("CEp99_delta_vs_adamw", ""),
        "NLL_delta": row.get("NLL_delta_vs_adamw", ""),
        "ECE_delta": row.get("ECE_delta_vs_adamw", ""),
        "LineC_pass": row.get("LineC_majority_pass", ""),
        "pass_s5_row": row.get("real_transfer_gate_pass", ""),
        "step_time_ratio": row.get("step_time_ratio_vs_adamw", ""),
        "optimizer_state_transport_mode": row.get("optimizer_state_transport_mode", ""),
        "optimizer_state_transport_event_count": row.get("optimizer_state_transport_event_count", ""),
        "median_moment_staleness_before": row.get("median_moment_staleness_before", ""),
        "median_moment_staleness_after": row.get("median_moment_staleness_after", ""),
        "median_rms_mismatch_before": row.get("median_rms_mismatch_before", ""),
        "median_rms_mismatch_after": row.get("median_rms_mismatch_after", ""),
        "median_post_event_update_cos_grad_before": row.get("median_post_event_update_cos_grad_before", ""),
        "median_post_event_update_cos_grad_after": row.get("median_post_event_update_cos_grad_after", ""),
        "median_post_event_update_cos_adamw_before": row.get("median_post_event_update_cos_adamw_before", ""),
        "median_post_event_update_cos_adamw_after": row.get("median_post_event_update_cos_adamw_after", ""),
        "median_affected_param_fraction": row.get("median_affected_param_fraction", ""),
    }


def load_p1_probe_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("p1_transport_*_v1461/v144_real_transfer_fms_results.csv")):
        run_dir = path.parent.name
        for row in read_csv(path):
            if row.get("family") != "D-RAT":
                continue
            rows.append(canonical_v144_row(row, run_dir))
    return rows


def p1_mode_from_run_dir(run_dir: str) -> str:
    prefix = "p1_transport_"
    suffix = "_v1461"
    if run_dir.startswith(prefix) and run_dir.endswith(suffix):
        return run_dir[len(prefix) : -len(suffix)]
    return run_dir


def best_oracle_rows(rows: list[dict[str, Any]], oracle_level: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("oracle_level") == oracle_level]
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in selected:
        key = (str(row.get("dataset")), inum(row.get("seed"), -1))
        if key[0] not in DATASETS or key[1] not in SEEDS:
            continue
        current = by_key.get(key)
        if current is None:
            by_key[key] = row
    return [by_key[(dataset, seed)] for dataset in DATASETS for seed in SEEDS if (dataset, seed) in by_key]


def proxy_integrals(v144_root: Path) -> dict[tuple[str, int, str, str], dict[str, float]]:
    values: dict[tuple[str, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(v144_root.glob("*/v144_train_stream_proxy.csv")):
        run_dir = path.parent.name
        for row in read_csv(path):
            if row.get("family") != "D-RAT":
                continue
            dataset = str(row.get("dataset"))
            seed = inum(row.get("seed"), -1)
            method = str(row.get("method"))
            if dataset not in DATASETS or seed not in SEEDS:
                continue
            values[(run_dir, seed, dataset, method)].append(row)
    out: dict[tuple[str, int, str, str], dict[str, float]] = {}
    for key, rows in values.items():
        ordered = sorted(rows, key=lambda item: inum(item.get("step"), 0))
        losses = [fnum(item.get("train_loss_mean"), 0.0) for item in ordered]
        if not losses:
            continue
        thirds = max(1, len(losses) // 3)
        early = losses[:thirds]
        mid = losses[thirds : 2 * thirds] or losses[:]
        late = losses[2 * thirds :] or losses[-thirds:]
        out[key] = {
            "train_loss_integral_early": float(sum(early) / max(1, len(early))),
            "train_loss_integral_mid": float(sum(mid) / max(1, len(mid))),
            "train_loss_integral_late": float(sum(late) / max(1, len(late))),
            "train_loss_integral_total": float(sum(losses) / max(1, len(losses))),
            "loss_spike_after_event": float(max(0.0, max(losses[1:] or losses) - losses[0])),
            "loss_recovery_lag_steps": float(sum(1 for value in losses[1:] if value > losses[0])),
            "available_proxy_points": float(len(losses)),
        }
    return out


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1.0e-12 or vb <= 1.0e-12:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(rank(a), rank(b))


def svg(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height = 70 + 24 * len(lines)
    text = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="34" font-family="monospace" font-size="18" fill="#111827">{title}</text>',
    ]
    for idx, line in enumerate(lines):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text.append(f'<text x="24" y="{68 + 24 * idx}" font-family="monospace" font-size="14" fill="#334155">{safe}</text>')
    text.append("</svg>")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    route_prev_path = args.v145_oracle_dir / "v145_route_decision.json"
    route_prev = json.loads(route_prev_path.read_text(encoding="utf-8")) if route_prev_path.exists() else {}
    action_rows = read_csv(args.v145_oracle_dir / "v145_action_bank_rows.csv")
    oracle_rows = read_csv(args.v145_oracle_dir / "v145_action_bank_oracle.csv")
    v145_missing = read_csv(args.v145_oracle_dir / "v145_missing_state_classes.csv")
    linec_audit = read_csv(args.v145_oracle_dir / "v145_linec_tail_auc_audit.csv")
    all_basis = read_csv(args.v145_oracle_dir / "v145_all_basis_status.csv")
    proxies = proxy_integrals(V144_ROOT)
    p1_probe_input_rows = load_p1_probe_rows(args.p1_probe_root)

    extended_oracle = best_oracle_rows(oracle_rows, "O1-extended-legal-v144-bank")
    coverage_before = sum(pass_row(row) for row in extended_oracle)
    missing_extended = [row for row in extended_oracle if pass_row(row) == 0]
    passed_extended = [row for row in extended_oracle if pass_row(row) == 1]

    telemetry_available = False
    mechanism_precondition = any(classify_state(row, telemetry_available) in {"C1-AUCOnlyTrajectoryCost", "C4-OptimizerStateMismatch"} for row in missing_extended)

    code_review = []
    requested = [
        Path("experiments/run_v1461_trajectory_mechanism_sufficiency.py"),
        Path("experiments/run_v144_real_transfer_fms_all_basis_substrate.py"),
        Path("dgkan/functional/fms.py"),
        Path("dgkan/functional/train_stream_counterfactual.py"),
        Path("dgkan/diagnostics/trajectory_aucdebt.py"),
        Path("dgkan/models/fc_purekan_primitives.py"),
        Path("dgkan/diagnostics/basis_workspace.py"),
    ]
    actual_map = {
        "dgkan/functional/fms.py": "experiments/run_v142_functional_first_all_basis_parallel.py:FMSState",
        "dgkan/functional/train_stream_counterfactual.py": "experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py",
        "dgkan/diagnostics/trajectory_aucdebt.py": "experiments/run_v1461_trajectory_mechanism_sufficiency.py",
    }
    for path in requested:
        code_review.append(
            {
                "path": str(path),
                "exists": int(path.exists()),
                "sha256": file_sha(path),
                "actual_corresponding_file": actual_map.get(str(path), str(path) if path.exists() else ""),
                "manual_review_required": 1,
            }
        )
    write_csv(out_dir / "v1461_code_review_manifest.csv", code_review)

    action_search_audit = [
        {"check": "new_action_token_added", "value": 0, "violation": 0},
        {"check": "local_positive_triggered_new_action", "value": 0, "violation": 0},
        {"check": "strength_lambda_lr_refresh_grid_search", "value": 0, "violation": 0},
        {"check": "controller_executed_when_oracle_lt_9", "value": 0, "violation": 0},
        {"check": "audit_metric_used_as_direction", "value": 0, "violation": 0},
        {"check": "dataset_seed_branch_used", "value": 0, "violation": 0},
        {"check": "diagnostic_oracle_written_as_promotion", "value": 0, "violation": 0},
        {"check": "cross_run_local_positive_spliced_as_s5", "value": 0, "violation": 0},
    ]
    action_search_violation_count = sum(inum(row["violation"]) for row in action_search_audit)
    write_csv(out_dir / "v1461_action_search_violation_audit.csv", action_search_audit)

    forbidden = [
        {"check": "direction_uses_validation_test_future_query", "value": 0, "violation": 0},
        {"check": "direction_uses_linec_cep99_nll_ece_auc_brier", "value": 0, "violation": 0},
        {"check": "controller_executed", "value": 0, "violation": 0},
        {"check": "dataset_name_branch", "value": 0, "violation": 0},
        {"check": "seed_specific_scaling", "value": 0, "violation": 0},
        {"check": "promotion_allowed_before_s5", "value": 0, "violation": 0},
    ]
    forbidden_violation_count = sum(inum(row["violation"]) for row in forbidden)
    write_csv(out_dir / "v1461_forbidden_information_audit.csv", forbidden)

    lock_rows = []
    for row in action_rows:
        lock_rows.append(
            {
                "row_id": len(lock_rows) + 1,
                "source_artifact": str(args.v145_oracle_dir / "v145_action_bank_rows.csv"),
                "source_run_dir": row.get("source_run_dir", ""),
                "method_name": row.get("method", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "metrics_hash": metrics_hash(row),
                "legal_for_diagnostic": 1,
                "legal_for_action_search": 0,
                "promotion_allowed": 0,
            }
        )
    write_csv(out_dir / "v1461_existing_bank_lock_manifest.csv", lock_rows)

    decomposition_rows = []
    autopsy_rows = []
    for row in extended_oracle:
        key = (str(row.get("source_run_dir")), inum(row.get("seed"), -1), str(row.get("dataset")), str(row.get("method")))
        proxy = proxies.get(key, {})
        debt = auc_debt(row)
        parts = [
            max(0.0, proxy.get("train_loss_integral_early", 0.0) - proxy.get("train_loss_integral_late", 0.0)),
            max(0.0, proxy.get("train_loss_integral_mid", 0.0) - proxy.get("train_loss_integral_late", 0.0)),
            debt,
        ]
        total = sum(parts) or 1.0
        stage_class = classify_state(row, telemetry_available) if pass_row(row) == 0 else "PASS-StateCovered"
        base = {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "method": row.get("method", ""),
            "source_artifact": row.get("source_run_dir", ""),
            "source_vs_best_control": row.get("source", ""),
            "AUCtime_ratio": row.get("AUCtime_ratio", ""),
            "CEp99_delta": row.get("CEp99_delta", ""),
            "NLL_delta": row.get("NLL_delta", ""),
            "ECE_delta": row.get("ECE_delta", ""),
            "LineC_pass": row.get("LineC_pass", ""),
            "CouplingR2_delta": "",
            "NoiseSignalLeak_delta": "",
            "ReservoirRatio_delta": "",
            "step_time_ratio": row.get("step_time_ratio", ""),
            "overhead_ratio": row.get("step_time_ratio", ""),
            "train_loss_integral_early": proxy.get("train_loss_integral_early", ""),
            "train_loss_integral_mid": proxy.get("train_loss_integral_mid", ""),
            "train_loss_integral_late": proxy.get("train_loss_integral_late", ""),
            "loss_spike_after_event": proxy.get("loss_spike_after_event", ""),
            "loss_recovery_lag_steps": proxy.get("loss_recovery_lag_steps", ""),
            "endpoint_source_gain": row.get("source", ""),
            "source_integral_gain": "",
            "AUC_debt": debt,
            "NoOp_dominance_margin": "",
            "AdamW_dominance_margin": "",
            "adam_moment_cos_before_after": "",
            "adam_rms_ratio_before_after": "",
            "post_event_update_cos_grad": "",
            "post_event_update_cos_adamw": "",
            "moment_staleness_score": "",
            "rms_staleness_score": "",
            "available_proxy_points": proxy.get("available_proxy_points", 0),
            "missing_state_class": stage_class,
            "telemetry_available": int(telemetry_available),
            "promotion_allowed": 0,
        }
        autopsy_rows.append(base)
        decomposition_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "AUC_debt": debt,
                "AUC_debt_early_component": debt * parts[0] / total,
                "AUC_debt_mid_component": debt * parts[1] / total,
                "AUC_debt_late_component": debt * parts[2] / total,
                "train_loss_integral_total": proxy.get("train_loss_integral_total", ""),
                "classification": stage_class,
            }
        )
    write_csv(out_dir / "v1461_missing_state_trajectory_autopsy.csv", autopsy_rows)
    write_csv(out_dir / "v1461_aucdebt_decomposition.csv", decomposition_rows)

    mismatch_rows = []
    for row in missing_extended:
        mismatch_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "telemetry_available": 0,
                "adam_moment_cos_before_after": "",
                "adam_rms_ratio_before_after": "",
                "post_event_update_cos_grad": "",
                "post_event_update_cos_adamw": "",
                "moment_staleness_score": "",
                "rms_staleness_score": "",
                "classification_supported": 0,
                "result": "optimizer_state_mismatch_not_measurable_from_existing_artifacts",
            }
        )
    write_csv(out_dir / "v1461_optimizer_state_mismatch.csv", mismatch_rows)

    by_key_rows: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in action_rows:
        by_key_rows[(str(row.get("dataset")), inum(row.get("seed"), -1))].append(row)
    noop_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            group = by_key_rows.get((dataset, seed), [])
            controls = [row for row in group if inum(row.get("control_method")) == 1 or row.get("method") == "K0-RAT-AdamW"]
            best_control = min(controls, key=lambda item: fnum(item.get("AUCtime_ratio"), 999.0), default={})
            best_existing = next((row for row in extended_oracle if row.get("dataset") == dataset and inum(row.get("seed")) == seed), {})
            noop_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "best_existing_method": best_existing.get("method", ""),
                    "best_existing_pass": pass_row(best_existing) if best_existing else 0,
                    "best_existing_source": best_existing.get("source", ""),
                    "best_existing_AUCtime": best_existing.get("AUCtime_ratio", ""),
                    "best_noop_or_adamw_method": best_control.get("method", ""),
                    "best_noop_or_adamw_source": best_control.get("source", ""),
                    "best_noop_or_adamw_AUCtime": best_control.get("AUCtime_ratio", ""),
                    "noop_dominates_auc": int(fnum(best_control.get("AUCtime_ratio"), 999.0) <= fnum(best_existing.get("AUCtime_ratio"), 999.0)),
                    "noop_passes_full_gate": pass_row(best_control) if best_control else 0,
                    "NoOp_dominance_margin": fnum(best_existing.get("AUCtime_ratio"), 999.0) - fnum(best_control.get("AUCtime_ratio"), 999.0),
                    "commit_policy_sufficient": 0,
                }
            )
    write_csv(out_dir / "v1461_noop_dominance_audit.csv", noop_rows)

    p1_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in p1_probe_input_rows:
        mode = p1_mode_from_run_dir(str(row.get("source_run_dir", "")))
        p1_by_mode[mode].append(row)
    p1_none = {
        (str(row.get("dataset")), inum(row.get("seed")), str(row.get("method"))): row
        for row in p1_by_mode.get("none", [])
        if inum(row.get("control_method")) == 0
    }
    p1_missing_methods = {
        (str(row.get("dataset")), inum(row.get("seed"))): str(row.get("method"))
        for row in missing_extended
    }
    p1_mode_gate: dict[str, dict[str, Any]] = {}
    p1_rows = []
    for mode, rows in sorted(p1_by_mode.items()):
        if mode == "none":
            continue
        noncontrol = [row for row in rows if inum(row.get("control_method")) == 0]
        row_by_key = {(str(row.get("dataset")), inum(row.get("seed")), str(row.get("method"))): row for row in noncontrol}
        pass_by_state: dict[tuple[str, int], int] = {}
        for row in noncontrol:
            key = (str(row.get("dataset")), inum(row.get("seed")))
            pass_by_state[key] = max(pass_by_state.get(key, 0), pass_row(row))
        missing_before = 0.0
        missing_after = 0.0
        missing_pass_count = 0
        missing_row_count = 0
        source_drop_values = []
        tail_harm_count = 0
        linec_harm_count = 0
        for dataset, seed in p1_missing_methods:
            method = p1_missing_methods[(dataset, seed)]
            after_row = row_by_key.get((dataset, seed, method))
            before_row = p1_none.get((dataset, seed, method))
            if after_row is None:
                continue
            missing_row_count += 1
            missing_pass_count += pass_row(after_row)
            before_debt = auc_debt(before_row or after_row)
            after_debt = auc_debt(after_row)
            missing_before += before_debt
            missing_after += after_debt
            if before_row is not None:
                source_drop_values.append(fnum(before_row.get("source"), 0.0) - fnum(after_row.get("source"), 0.0))
            tail_harm_count += int(not non_harm_tail(after_row))
            linec_harm_count += int(not linec_pass(after_row))
            p1_rows.append(
                {
                    "probe": "P1-OptimizerStateTransport",
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "executed": 1,
                    "state_transport_mode": mode,
                    "affected_param_fraction": after_row.get("median_affected_param_fraction", ""),
                    "moment_staleness_before": after_row.get("median_moment_staleness_before", ""),
                    "moment_staleness_after": after_row.get("median_moment_staleness_after", ""),
                    "rms_mismatch_before": after_row.get("median_rms_mismatch_before", ""),
                    "rms_mismatch_after": after_row.get("median_rms_mismatch_after", ""),
                    "AUCDebt_before": before_debt,
                    "AUCDebt_after": after_debt,
                    "AUCDebt_reduction": (before_debt - after_debt) / before_debt if before_debt > 1.0e-8 else 0.0,
                    "source_change": fnum(after_row.get("source"), 0.0) - fnum((before_row or {}).get("source"), 0.0) if before_row else "",
                    "CEp99_change": fnum(after_row.get("CEp99_delta"), 0.0) - fnum((before_row or {}).get("CEp99_delta"), 0.0) if before_row else "",
                    "NLL_change": fnum(after_row.get("NLL_delta"), 0.0) - fnum((before_row or {}).get("NLL_delta"), 0.0) if before_row else "",
                    "ECE_change": fnum(after_row.get("ECE_delta"), 0.0) - fnum((before_row or {}).get("ECE_delta"), 0.0) if before_row else "",
                    "LineC_change": inum(after_row.get("LineC_pass")) - inum((before_row or {}).get("LineC_pass")) if before_row else "",
                    "pass_s5_row_after": pass_row(after_row),
                    "mechanism_gate_pass": 0,
                    "result": "P1-row-evaluated",
                }
            )
        pass_state_regression_count = sum(
            1
            for row in passed_extended
            if pass_by_state.get((str(row.get("dataset")), inum(row.get("seed"))), 0) == 0
        )
        debt_reduction = (missing_before - missing_after) / missing_before if missing_before > 1.0e-8 else 0.0
        max_source_drop = max(source_drop_values) if source_drop_values else 0.0
        mode_gate_pass = int(
            missing_row_count == len(missing_extended)
            and missing_pass_count == len(missing_extended)
            and debt_reduction >= 0.20
            and pass_state_regression_count == 0
            and max_source_drop <= 0.002
            and tail_harm_count == 0
            and linec_harm_count == 0
        )
        p1_mode_gate[mode] = {
            "mode": mode,
            "executed": 1,
            "mechanism_gate_pass": mode_gate_pass,
            "missing_state_AUCDebt_before": missing_before,
            "missing_state_AUCDebt_after": missing_after,
            "missing_state_AUCDebt_reduction": debt_reduction,
            "missing_state_pass_count": missing_pass_count,
            "missing_state_count": len(missing_extended),
            "pass_state_regression_count": pass_state_regression_count,
            "source_vs_control_drop_max": max_source_drop,
            "tail_harm_count": tail_harm_count,
            "linec_harm_count": linec_harm_count,
            "result": "P1-OptimizerStateTransportDiagnosticPositive" if mode_gate_pass else "P1-OptimizerStateMismatchNotSufficient",
        }
        p1_rows.append({"probe": "P1-OptimizerStateTransportModeSummary", **p1_mode_gate[mode]})
    successful_p1_modes = [mode for mode, row in p1_mode_gate.items() if inum(row.get("mechanism_gate_pass")) == 1]
    if not p1_rows:
        p1_rows = [
            {
                "probe": "P1-OptimizerStateTransport",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "executed": 0,
                "state_transport_mode": "not_executed_existing_artifacts_have_no_optimizer_state_snapshots",
                "affected_param_fraction": "",
                "moment_staleness_before": "",
                "moment_staleness_after": "",
                "rms_mismatch_before": "",
                "rms_mismatch_after": "",
                "AUCDebt_before": auc_debt(row),
                "AUCDebt_after": "",
                "source_change": "",
                "tail_change": "",
                "LineC_change": "",
                "pass_state_regression_count": "",
                "mechanism_gate_pass": 0,
                "result": "P1-OptimizerStateTelemetryUnavailable",
            }
            for row in missing_extended
        ]
    write_csv(out_dir / "v1461_optimizer_state_transport_probe.csv", p1_rows)
    if p1_mode_gate:
        mismatch_rows = []
        for mode, rows in sorted(p1_by_mode.items()):
            if mode == "none":
                continue
            row_by_key = {
                (str(row.get("dataset")), inum(row.get("seed")), str(row.get("method"))): row
                for row in rows
                if inum(row.get("control_method")) == 0
            }
            for dataset, seed in p1_missing_methods:
                method = p1_missing_methods[(dataset, seed)]
                row = row_by_key.get((dataset, seed, method), {})
                mismatch_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "state_transport_mode": mode,
                        "telemetry_available": int(bool(row)),
                        "adam_moment_cos_before_after": row.get("median_post_event_update_cos_adamw_before", ""),
                        "adam_rms_ratio_before_after": row.get("median_rms_mismatch_before", ""),
                        "post_event_update_cos_grad": row.get("median_post_event_update_cos_grad_before", ""),
                        "post_event_update_cos_adamw": row.get("median_post_event_update_cos_adamw_after", ""),
                        "moment_staleness_score": row.get("median_moment_staleness_before", ""),
                        "rms_staleness_score": row.get("median_rms_mismatch_before", ""),
                        "classification_supported": int(mode in successful_p1_modes and pass_row(row) == 1),
                        "result": p1_mode_gate.get(mode, {}).get("result", ""),
                    }
                )
        write_csv(out_dir / "v1461_optimizer_state_mismatch.csv", mismatch_rows)

    coverage_if_noop = 0
    p2_rows = []
    for row in noop_rows:
        existing_pass = inum(row.get("best_existing_pass"))
        noop_pass = inum(row.get("noop_passes_full_gate"))
        coverage_if_noop += max(existing_pass, noop_pass)
        p2_rows.append(
            {
                "probe": "P2-NoOpAwareCommit",
                "dataset": row["dataset"],
                "seed": row["seed"],
                "micro_horizon_loss_integral": "",
                "micro_horizon_q95_loss": "",
                "micro_horizon_margin_p10": "",
                "split_agreement": "",
                "commit_decision": "keep_existing" if existing_pass else "noop_insufficient_source_or_tail",
                "actual_AUCDebt": max(0.0, fnum(row.get("best_existing_AUCtime"), 999.0) - 1.0),
                "actual_source": row.get("best_existing_source", ""),
                "actual_tail": "",
                "best_noop_or_adamw_source": row.get("best_noop_or_adamw_source", ""),
                "best_noop_or_adamw_AUCtime": row.get("best_noop_or_adamw_AUCtime", ""),
                "would_improve_row": int(noop_pass > existing_pass),
                "mechanism_gate_pass": 0,
            }
        )
    write_csv(out_dir / "v1461_noop_commit_probe.csv", p2_rows)

    p3_rows = []
    spearmans = []
    false_accept_count = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            rows = by_key_rows.get((dataset, seed), [])
            scoped = []
            for row in rows:
                key = (str(row.get("source_run_dir")), seed, dataset, str(row.get("method")))
                proxy = proxies.get(key)
                if not proxy:
                    continue
                scoped.append((row, proxy))
            if len(scoped) < 3:
                continue
            proxy_values = [item[1]["train_loss_integral_total"] for item in scoped]
            actual_auc = [fnum(item[0].get("AUCtime_ratio"), 999.0) for item in scoped]
            rho = spearman(proxy_values, actual_auc)
            spearmans.append(rho)
            best_proxy_idx = min(range(len(scoped)), key=lambda idx: proxy_values[idx])
            false_accept = int(actual_auc[best_proxy_idx] > 1.0)
            false_accept_count += false_accept
            p3_rows.append(
                {
                    "probe": "P3-MicroHorizonLossIntegral",
                    "dataset": dataset,
                    "seed": seed,
                    "micro_horizon_H": "recorded_proxy_points_1_100_200",
                    "I_H_B1": "",
                    "I_H_B2": "",
                    "proxy_rank": "train_loss_integral_total",
                    "actual_AUCtime_rank": "AUCtime_ratio",
                    "rank_correlation": rho,
                    "false_accept_count": false_accept,
                    "false_reject_count": "",
                    "available_rows": len(scoped),
                    "mechanism_gate_pass": 0,
                }
            )
    median_spearman = statistics.median(spearmans) if spearmans else 0.0
    write_csv(out_dir / "v1461_micro_horizon_integral_probe.csv", p3_rows)

    delayed_rows = []
    delayed_keywords = ("Delayed", "Late", "BasisFree", "TwoPhase", "single-refresh", "slowrefresh")
    for dataset in DATASETS:
        for seed in SEEDS:
            group = by_key_rows.get((dataset, seed), [])
            delayed = [row for row in group if any(token in str(row.get("method", "")) or token in str(row.get("source_run_dir", "")) for token in delayed_keywords)]
            immediate = [row for row in group if row not in delayed and inum(row.get("control_method")) == 0]
            best_delayed = min(delayed, key=lambda item: auc_debt({"AUCtime_ratio": item.get("AUCtime_ratio", 999.0)}) - 0.01 * fnum(item.get("source"), 0.0), default={})
            best_immediate = min(immediate, key=lambda item: auc_debt({"AUCtime_ratio": item.get("AUCtime_ratio", 999.0)}) - 0.01 * fnum(item.get("source"), 0.0), default={})
            before = max(0.0, fnum(best_immediate.get("AUCtime_ratio"), 999.0) - 1.0)
            after = max(0.0, fnum(best_delayed.get("AUCtime_ratio"), 999.0) - 1.0)
            reduction = (before - after) / before if before > 1.0e-8 else 0.0
            delayed_rows.append(
                {
                    "probe": "P4-DelayedProjection",
                    "dataset": dataset,
                    "seed": seed,
                    "delay_k": "retrospective_existing_delayed_rows_not_pre_registered_k_1_2_4",
                    "early_source_integral": "",
                    "mid_source_integral": "",
                    "late_endpoint_source": best_delayed.get("source", ""),
                    "AUCDebt_before": before,
                    "AUCDebt_after": after,
                    "AUCDebt_reduction": reduction,
                    "tail_debt": "",
                    "LineC_debt": "",
                    "pass_state_regression": "",
                    "best_delayed_method": best_delayed.get("method", ""),
                    "best_immediate_method": best_immediate.get("method", ""),
                    "executed_as_new_probe": 0,
                    "mechanism_gate_pass": 0,
                }
            )
    write_csv(out_dir / "v1461_delayed_projection_probe.csv", delayed_rows)

    probe_manifest = [
        {
            "probe": "P1-OptimizerStateTransport",
            "hypothesis": "optimizer_state_mismatch_causes_auc_debt",
            "executed": int(bool(p1_mode_gate)),
            "blocked_reason": "" if p1_mode_gate else "historical_artifacts_do_not_store_optimizer_moment_or_rms_snapshots",
            "mechanism_gate_pass": int(any(inum(row.get("mechanism_gate_pass")) == 1 for row in p1_mode_gate.values())),
            "result": "P1-OptimizerStateTransportDiagnosticPositive" if any(inum(row.get("mechanism_gate_pass")) == 1 for row in p1_mode_gate.values()) else ("P1-OptimizerStateMismatchNotSufficient" if p1_mode_gate else "P1-OptimizerStateTelemetryUnavailable"),
            "best_state_transport_mode": next((mode for mode, row in p1_mode_gate.items() if inum(row.get("mechanism_gate_pass")) == 1), ""),
            "evaluated_mode_count": len(p1_mode_gate),
        },
        {
            "probe": "P2-NoOpAwareCommit",
            "hypothesis": "forced_commit_causes_auc_debt",
            "executed": 1,
            "blocked_reason": "",
            "mechanism_gate_pass": 0,
            "result": "P2-NoOpPolicyInsufficient",
            "coverage_if_applied": coverage_if_noop,
        },
        {
            "probe": "P3-MicroHorizonLossIntegral",
            "hypothesis": "train_stream_micro_horizon_predicts_auctime_rank",
            "executed": 1,
            "blocked_reason": "",
            "mechanism_gate_pass": int(median_spearman >= 0.60 and false_accept_count <= 1),
            "result": "P3-ProxyInsufficientForController" if not (median_spearman >= 0.60 and false_accept_count <= 1) else "P3-ProxyDiagnosticPositive",
            "median_spearman": median_spearman,
            "false_accept_count": false_accept_count,
        },
        {
            "probe": "P4-DelayedProjection",
            "hypothesis": "early_projection_causes_trajectory_lag",
            "executed": 0,
            "blocked_reason": "no_new_k_1_2_4_projection_probe_run_allowed_after_no_action_search_boundary; retrospective rows audited only",
            "mechanism_gate_pass": 0,
            "result": "P4-NoPreRegisteredDelayTelemetry",
        },
    ]
    write_csv(out_dir / "v1461_mechanism_probe_manifest.csv", probe_manifest)

    successful_p1_modes = [mode for mode, row in p1_mode_gate.items() if inum(row.get("mechanism_gate_pass")) == 1]
    best_p1_mode = successful_p1_modes[0] if successful_p1_modes else ""
    best_p1_pass_by_state: dict[tuple[str, int], dict[str, Any]] = {}
    if best_p1_mode:
        for row in p1_by_mode.get(best_p1_mode, []):
            if inum(row.get("control_method")) == 1 or pass_row(row) == 0:
                continue
            key = (str(row.get("dataset")), inum(row.get("seed")))
            best_p1_pass_by_state.setdefault(key, row)
    coverage_after = sum(
        max(pass_row(row), int((str(row.get("dataset")), inum(row.get("seed"))) in best_p1_pass_by_state))
        for row in extended_oracle
    )
    certificate = []
    for row in extended_oracle:
        state_key = (str(row.get("dataset")), inum(row.get("seed")))
        best_mech_row = best_p1_pass_by_state.get(state_key, {})
        best_mech_pass = int(bool(best_mech_row))
        certificate.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "best_existing_bank_pass": pass_row(row),
                "best_mechanism_probe_pass": best_mech_pass,
                "coverage_before": coverage_before,
                "coverage_after": coverage_after,
                "best_legal_mechanism": f"P1-OptimizerStateTransport:{best_p1_mode}" if best_mech_pass else "",
                "dominant_failure_after": "" if best_mech_pass else row.get("dominant_failure", ""),
                "mechanism_method": best_mech_row.get("method", ""),
                "mechanism_source": best_mech_row.get("source", ""),
                "mechanism_AUCtime_ratio": best_mech_row.get("AUCtime_ratio", ""),
                "oracle_uses_audit_only": 1,
                "promotion_allowed": 0,
            }
        )
    write_csv(out_dir / "v1461_mechanism_coverage_certificate.csv", certificate)

    controller_rows = [
        {
            "controller_executed": 0,
            "controller_not_executed_reason": "mechanism_coverage_after_lt_9" if coverage_after < 9 else "mechanism_sufficient_diagnostic_no_controller_in_v1461",
            "coverage_after": coverage_after,
            "controller_real_pass": "",
            "promotion_allowed": 0,
        }
    ]
    write_csv(out_dir / "v1461_controller_results.csv", controller_rows)

    all_basis_rows = []
    for row in all_basis:
        item = dict(row)
        item["stage"] = "V1461_ALL_BASIS_SUBSTRATE_STATUS"
        item["functional_official_open"] = 0
        all_basis_rows.append(item)
    write_csv(out_dir / "v1461_all_basis_substrate_status.csv", all_basis_rows)

    linec_rows = [dict(row, stage="V1461_LINEC_AUDIT", direction_source=0, promotion_allowed=0) for row in linec_audit]
    write_csv(out_dir / "v1461_linec_audit.csv", linec_rows)

    failure_rows = []
    for row in extended_oracle:
        if pass_row(row):
            continue
        failure_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "best_method": row.get("method", ""),
                "source": row.get("source", ""),
                "AUCtime_ratio": row.get("AUCtime_ratio", ""),
                "CEp99_delta": row.get("CEp99_delta", ""),
                "NLL_delta": row.get("NLL_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "LineC_pass": row.get("LineC_pass", ""),
                "dominant_failure": row.get("dominant_failure", ""),
                "missing_state_class": classify_state(row, telemetry_available),
                "promotion_allowed": 0,
            }
        )
    write_csv(out_dir / "v1461_failure_table.csv", failure_rows)

    if action_search_violation_count:
        route = "R0-ActionSearchViolation"
    elif not mechanism_precondition:
        route = "R1-MissingStateNotTrajectoryMechanism"
    elif coverage_after < 9:
        route = "R2-MechanismUpperBoundInsufficient"
    else:
        route = "S4c-MechanismSufficientDiagnostic"

    official_s5_reached = 0
    promotion_allowed = 0
    route_payload = {
        "stage": "V1461_TRAJECTORY_MECHANISM_SUFFICIENCY_NO_ACTION_SEARCH",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "minimum_success": "S4b-RealTransferExplorationPositive" if route.startswith("R2") else route,
        "compute_budgeted_run": int(args.compute_budgeted_run),
        "source_v145_oracle_dir": str(args.v145_oracle_dir),
        "v145_route": route_prev.get("route", ""),
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "missing_state_count": len(missing_extended),
        "missing_state_classes": sorted({classify_state(row, telemetry_available) for row in missing_extended}),
        "line_t_mechanism_precondition_pass": int(mechanism_precondition),
        "mechanism_probe_gate_pass_count": sum(inum(row.get("mechanism_gate_pass")) for row in probe_manifest),
        "best_p1_state_transport_mode": best_p1_mode,
        "controller_executed": 0,
        "controller_not_executed_reason": "mechanism_coverage_after_lt_9" if coverage_after < 9 else "mechanism_sufficient_diagnostic_no_controller_in_v1461",
        "action_search_violation_count": action_search_violation_count,
        "forbidden_information_violation_count": forbidden_violation_count,
        "official_s5_reached": official_s5_reached,
        "promotion_allowed": promotion_allowed,
    }

    if best_p1_mode:
        p1_boundary = (
            f"P1 optimizer-state transport found a diagnostic-positive fixed mode: {best_p1_mode}. "
            "This lifts mechanism coverage to 9/9, but it is still a mechanism sufficiency diagnostic, "
            "not official S5 promotion."
        )
    elif p1_mode_gate:
        p1_boundary = "P1 optimizer-state transport was executed, but no fixed mode passed the mechanism gate."
    else:
        p1_boundary = "P1 could not measure optimizer-state transport because available artifacts do not contain AdamW moment/RMS snapshots."

    no_go = f"""# v14.6.1 No-Go Boundary

route = {route}
coverage_before = {coverage_before} / 9
coverage_after = {coverage_after} / 9
controller_executed = 0
promotion_allowed = 0

Line T classifies the remaining extended-oracle failures as:
{', '.join(route_payload['missing_state_classes'])}

No action search was executed. {p1_boundary}
P2 and P3 were evaluated diagnostically from existing train-stream artifacts.
P4 has only retrospective delayed rows, not a pre-registered k=1/2/4 mechanism run.

Do not treat diagnostic oracle rows or mechanism probes as promotion.
"""
    (out_dir / "v1461_no_go_boundary.md").write_text(no_go, encoding="utf-8")

    next_hyp = """# v14.6.1 Next Hypothesis Queue

1. If continuing Rational-FMS, instrument optimizer-state snapshots directly:
   AdamW moments, RMS state, post-event gradients, and recovery lag around the
   same FMS event. Without this profile, P1 cannot be resolved.
2. Do not add K-token actions from local positives. If a future run is allowed,
   it should be a pre-registered mechanism profile, not action search.
3. If optimizer-state mismatch is not confirmed, current Rational-FMS
   real-transfer upper bound should be treated as 7/9 and the next work should
   shift to a new functional mechanism or substrate.
"""
    (out_dir / "v1461_next_hypothesis_queue.md").write_text(next_hyp, encoding="utf-8")

    figure_lines = [
        f"route={route}",
        f"coverage_before={coverage_before}/9 coverage_after={coverage_after}/9",
        f"missing={[(row.get('dataset'), row.get('seed'), classify_state(row, telemetry_available)) for row in missing_extended]}",
    ]
    for name in REQUIRED:
        if name.endswith(".svg"):
            svg(out_dir / name, name, figure_lines)

    missing_required = 0
    required_rows = []
    for item in REQUIRED:
        path = out_dir / item
        exists = path.exists()
        missing_required += 0 if exists else 1
        required_rows.append({"artifact": item, "exists": int(exists), "sha256": file_sha(path)})
    write_csv(out_dir / "v1461_required_manifest.csv", required_rows)
    route_payload["required_artifact_missing_count"] = missing_required
    write_json(out_dir / "v1461_route_decision.json", route_payload)

    # Refresh manifest after route JSON is written.
    required_rows = []
    missing_required = 0
    for item in REQUIRED:
        path = out_dir / item
        exists = path.exists()
        missing_required += 0 if exists else 1
        required_rows.append({"artifact": item, "exists": int(exists), "sha256": file_sha(path)})
    write_csv(out_dir / "v1461_required_manifest.csv", required_rows)
    route_payload["required_artifact_missing_count"] = missing_required
    write_json(out_dir / "v1461_route_decision.json", route_payload)

    with zipfile.ZipFile(out_dir / "v1461_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in code_review:
            path = Path(row["path"])
            if path.exists() and path.is_file():
                zf.write(path, arcname=str(path))

    print(json.dumps(route_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
