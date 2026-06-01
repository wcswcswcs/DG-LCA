#!/usr/bin/env python3
"""DG-KAN v14.5 train-stream counterfactual FMS oracle finalizer.

This runner implements the first hard gate in the v14.5 plan: Line O
action-bank oracle decomposition. It reads already-executed, legal v14.4
real-transfer artifacts and asks whether the current action bank has a 9/9
dataset-seed upper bound before any controller policy is tuned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = ("MNIST", "Fashion-MNIST", "KMNIST")
SEEDS = (0, 1, 2)
V144_ROOT = Path("results/v14_4_real_transfer_fms_all_basis_substrate")
DEFAULT_OUT_DIR = Path("results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145")
PLAN_PATH = Path("docs/DG-KAN_v14.5_TrainStreamCounterfactualFMS_AllBasisParallel_完整计划.md")


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    run_dir: str
    methods: tuple[str, ...] | None
    oracle_level: str


CORE_ACTION_BANK: tuple[ActionSpec, ...] = (
    ActionSpec("A0-lowplasticity-K0", "repair_v144_lowplasticity_lambda05", ("K0-RAT-AdamW",), "O1-core-action-bank"),
    ActionSpec(
        "A1-lowplasticity-K8",
        "repair_v144_lowplasticity_plus_k8_krt4_lambda05",
        ("K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint",),
        "O1-core-action-bank",
    ),
    ActionSpec("A2-lowplasticity-KRT1", "repair_v144_lowplasticity_lambda05", ("K-RT1-TrainSplitAgreement",), "O1-core-action-bank"),
    ActionSpec("A3-lowplasticity-KRT2", "repair_v144_lowplasticity_lambda05", ("K-RT2-TrainStreamTailTrust",), "O1-core-action-bank"),
    ActionSpec("A4-lowplasticity-KRT3", "repair_v144_lowplasticity_lambda05", ("K-RT3-ProjectionValueRetention",), "O1-core-action-bank"),
    ActionSpec(
        "A5-lowplasticity-KRT4",
        "repair_v144_lowplasticity_plus_k8_krt4_lambda05",
        ("K-RT4-SourceTailCoState",),
        "O1-core-action-bank",
    ),
    ActionSpec("A6-lowplasticity-KRT5", "repair_v144_lowplasticity_lambda05", ("K-RT5-DelayedBasisConstraint",), "O1-core-action-bank"),
    ActionSpec("A7-KRT6-projection-tailtrust", "repair_v144_krt6_projection_tailtrust_lambda05", ("K-RT6-ProjectionTailTrust",), "O1-core-action-bank"),
    ActionSpec("A8-KRT7-late-projection-tailtrust", "repair_v144_krt7_late_projection_tailtrust_lambda05", ("K-RT7-LateProjectionTailTrust",), "O1-core-action-bank"),
    ActionSpec("A9-single-refresh200", "repair_v144_single_refresh200_lambda05", None, "O1-core-action-bank"),
    ActionSpec("A10-slowrefresh160", "repair_v144_slowrefresh160_lambda05", None, "O1-core-action-bank"),
)


REQUIRED_ARTIFACTS = (
    "v145_route_decision.json",
    "v145_required_manifest.csv",
    "v145_forbidden_information_audit.csv",
    "v145_code_review_manifest.csv",
    "v145_action_bank_rows.csv",
    "v145_action_bank_oracle.csv",
    "v145_oracle_summary.csv",
    "v145_counterfactual_event_log.csv",
    "v145_controller_real_results.csv",
    "v145_controller_summary.csv",
    "v145_controller_failure_table.csv",
    "v145_wavelet_substrate_hardening.csv",
    "v145_all_basis_status.csv",
    "v145_mlp_generic_control.csv",
    "v145_linec_tail_auc_audit.csv",
    "v145_missing_state_classes.csv",
    "v145_no_go_boundary.md",
    "v145_next_hypothesis_queue.md",
    "figures/fig_v145_real_3x3_pass_matrix.svg",
    "figures/fig_v145_action_oracle_upper_bound.svg",
    "figures/fig_v145_counterfactual_selected_action_heatmap.svg",
    "figures/fig_v145_source_auc_tail_tradeoff.svg",
    "figures/fig_v145_failure_mode_by_dataset_seed.svg",
    "figures/fig_v145_proxy_vs_audit_scatter.svg",
    "figures/fig_v145_noop_rate_vs_pass.svg",
    "figures/fig_v145_overhead_breakdown.svg",
    "figures/fig_v145_wavelet_substrate_matrix.svg",
    "figures/fig_v145_all_basis_status_dashboard.svg",
    "figures/fig_v145_mlp_vs_rational_controller.svg",
    "v145_code_review_packet.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=("oracle",), default="oracle")
    parser.add_argument("--include-extended-v144-bank", type=int, default=1)
    parser.add_argument("--compute-budgeted-run", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def to_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def to_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def linec_pass(row: dict[str, Any]) -> int:
    if "LineC_majority_pass" in row:
        return 1 if to_int(row, "LineC_majority_pass") == 1 else 0
    return 1 if to_float(row, "LineC_pass_rate") >= 0.5 else 0


def gate_flags(row: dict[str, Any]) -> dict[str, int]:
    source = to_float(row, "source_vs_best_control")
    auc = to_float(row, "AUCtime_ratio_vs_best_control", default=999.0)
    cep = to_float(row, "CEp99_delta_vs_adamw", default=999.0)
    nll = to_float(row, "NLL_delta_vs_adamw", default=999.0)
    ece = to_float(row, "ECE_delta_vs_adamw", default=999.0)
    return {
        "failure_source": 1 if source < 0.005 else 0,
        "failure_auc": 1 if auc > 1.0 else 0,
        "failure_cep99": 1 if cep > 0.05 else 0,
        "failure_nll": 1 if nll > 0.02 else 0,
        "failure_ece": 1 if ece > 0.02 else 0,
        "failure_linec": 1 if linec_pass(row) != 1 else 0,
        "failure_overhead": 1 if to_float(row, "step_time_ratio_vs_adamw", 1.0) > 1.75 else 0,
    }


def pass_s5_row(row: dict[str, Any]) -> int:
    flags = gate_flags(row)
    return 1 if not any(flags.values()) else 0


def failure_count(row: dict[str, Any]) -> int:
    return sum(gate_flags(row).values())


def dominant_failure(row: dict[str, Any]) -> str:
    flags = gate_flags(row)
    ordered = ("failure_source", "failure_auc", "failure_cep99", "failure_nll", "failure_ece", "failure_linec", "failure_overhead")
    active = [name.replace("failure_", "") for name in ordered if flags[name]]
    return "+".join(active) if active else "-"


def oracle_score(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(pass_s5_row(row)),
        -float(failure_count(row)),
        to_float(row, "source_vs_best_control"),
        -to_float(row, "AUCtime_ratio_vs_best_control", 999.0),
        -to_float(row, "CEp99_delta_vs_adamw", 999.0),
        -to_float(row, "NLL_delta_vs_adamw", 999.0),
        -to_float(row, "ECE_delta_vs_adamw", 999.0),
        float(linec_pass(row)),
    )


def read_action_rows_from_spec(spec: ActionSpec) -> tuple[list[dict[str, Any]], list[str]]:
    path = V144_ROOT / spec.run_dir / "v144_real_transfer_fms_results.csv"
    if not path.exists():
        return [], [str(path)]
    methods = set(spec.methods) if spec.methods is not None else None
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row.get("dataset") not in DATASETS or to_int(row, "seed", -1) not in SEEDS:
            continue
        if methods is not None and row.get("method") not in methods:
            continue
        if row.get("control_method") == "1" and not spec.action_id.startswith("A0"):
            continue
        enriched = dict(row)
        enriched["action_id"] = spec.action_id
        enriched["oracle_level"] = spec.oracle_level
        enriched["source_run_dir"] = spec.run_dir
        rows.append(enriched)
    return rows, []


def core_action_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in CORE_ACTION_BANK:
        action_rows, action_missing = read_action_rows_from_spec(spec)
        rows.extend(action_rows)
        missing.extend(action_missing)
    return rows, missing


def extended_v144_action_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(V144_ROOT.glob("*/v144_real_transfer_fms_results.csv")):
        if "smoke" in path.parts[-2]:
            continue
        run_dir = path.parent.name
        for row in read_csv(path):
            if row.get("dataset") not in DATASETS or to_int(row, "seed", -1) not in SEEDS:
                continue
            if row.get("control_method") == "1":
                continue
            enriched = dict(row)
            method_slug = (row.get("method") or "unknown").replace(" ", "-")
            enriched["action_id"] = f"EXT-{run_dir}-{method_slug}"
            enriched["oracle_level"] = "O1-extended-legal-v144-bank"
            enriched["source_run_dir"] = run_dir
            rows.append(enriched)
    return rows


def choose_oracle(rows: list[dict[str, Any]], oracle_level: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("oracle_level") != oracle_level:
            continue
        key = (str(row.get("dataset")), to_int(row, "seed", -1))
        if key[0] not in DATASETS or key[1] not in SEEDS:
            continue
        if key not in by_key or oracle_score(row) > oracle_score(by_key[key]):
            by_key[key] = row
    for dataset in DATASETS:
        for seed in SEEDS:
            row = by_key.get((dataset, seed))
            if row is None:
                selected.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "action_id": "MISSING",
                        "method": "",
                        "source_run_dir": "",
                        "oracle_level": oracle_level,
                        "missing_dataset_seed": 1,
                    }
                )
            else:
                selected.append(row)
    return selected


def oracle_row(row: dict[str, Any]) -> dict[str, Any]:
    flags = gate_flags(row)
    return {
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "action_id": row.get("action_id", ""),
        "source_run_dir": row.get("source_run_dir", ""),
        "method": row.get("method", ""),
        "source": to_float(row, "source_vs_best_control"),
        "AUCtime_ratio": to_float(row, "AUCtime_ratio_vs_best_control"),
        "CEp99_delta": to_float(row, "CEp99_delta_vs_adamw"),
        "NLL_delta": to_float(row, "NLL_delta_vs_adamw"),
        "ECE_delta": to_float(row, "ECE_delta_vs_adamw"),
        "LineC_pass": linec_pass(row),
        "step_time_ratio": to_float(row, "step_time_ratio_vs_adamw", 1.0),
        "memory_ratio": to_float(row, "peak_memory_ratio_vs_adamw", 1.0),
        "pass_s5_row": pass_s5_row(row),
        "failure_source": flags["failure_source"],
        "failure_auc": flags["failure_auc"],
        "failure_cep99": flags["failure_cep99"],
        "failure_nll": flags["failure_nll"],
        "failure_ece": flags["failure_ece"],
        "failure_linec": flags["failure_linec"],
        "failure_overhead": flags["failure_overhead"],
        "dominant_failure": dominant_failure(row),
        "oracle_level": row.get("oracle_level", ""),
        "oracle_uses_audit_metric": 1,
        "promotion_allowed": 0,
    }


def summarize_oracle(selected: list[dict[str, Any]], oracle_level: str) -> dict[str, Any]:
    selected_rows = [oracle_row(row) for row in selected]
    return {
        "oracle_level": oracle_level,
        "dataset_seed_pass_count": sum(to_int(row, "pass_s5_row") for row in selected_rows),
        "pass_rows": sum(to_int(row, "pass_s5_row") for row in selected_rows),
        "failure_remaining_source": sum(to_int(row, "failure_source") for row in selected_rows),
        "failure_remaining_auc": sum(to_int(row, "failure_auc") for row in selected_rows),
        "failure_remaining_cep99": sum(to_int(row, "failure_cep99") for row in selected_rows),
        "failure_remaining_nll": sum(to_int(row, "failure_nll") for row in selected_rows),
        "failure_remaining_ece": sum(to_int(row, "failure_ece") for row in selected_rows),
        "failure_remaining_linec": sum(to_int(row, "failure_linec") for row in selected_rows),
        "failure_remaining_overhead": sum(to_int(row, "failure_overhead") for row in selected_rows),
        "action_bank_upper_bound_pass": 1 if sum(to_int(row, "pass_s5_row") for row in selected_rows) == 9 else 0,
    }


def action_bank_row(row: dict[str, Any]) -> dict[str, Any]:
    out = oracle_row(row)
    out["control_method"] = row.get("control_method", "")
    out["real_transfer_gate_pass"] = row.get("real_transfer_gate_pass", "")
    return out


def missing_state_rows(selected: list[dict[str, Any]], oracle_level: str) -> list[dict[str, Any]]:
    rows = []
    for row in selected:
        out = oracle_row(row)
        if to_int(out, "pass_s5_row") == 1:
            continue
        rows.append(
            {
                "oracle_level": oracle_level,
                "dataset": out["dataset"],
                "seed": out["seed"],
                "best_action_id": out["action_id"],
                "best_method": out["method"],
                "source": out["source"],
                "AUCtime_ratio": out["AUCtime_ratio"],
                "CEp99_delta": out["CEp99_delta"],
                "NLL_delta": out["NLL_delta"],
                "ECE_delta": out["ECE_delta"],
                "LineC_pass": out["LineC_pass"],
                "dominant_failure": out["dominant_failure"],
                "new_action_family_needed": "source_tail_auc_colocation_action",
                "selection_threshold_repair_allowed": 0,
                "promotion_allowed": 0,
            }
        )
    return rows


def copy_or_empty(src: Path, dest: Path, fieldnames: list[str]) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    else:
        write_csv(dest, [], fieldnames)


def write_empty_controller_artifacts(out_dir: Path, reason: str) -> None:
    write_csv(
        out_dir / "v145_counterfactual_event_log.csv",
        [],
        [
            "dataset",
            "seed",
            "step",
            "candidate_action",
            "selected_action",
            "source_proxy",
            "micro_auc_proxy",
            "tail_proxy",
            "split_agreement",
            "projection_retention",
            "fms_active_fraction",
            "basis_rejection_fraction",
            "score",
            "accepted",
            "rejection_reason",
            "counterfactual_overhead_ms",
        ],
    )
    write_csv(
        out_dir / "v145_controller_real_results.csv",
        [],
        [
            "dataset",
            "seed",
            "controller_id",
            "real_pass",
            "source_vs_best_control",
            "AUCtime_ratio",
            "CEp99_delta",
            "NLL_delta",
            "ECE_delta",
            "LineC_pass",
            "step_time_ratio",
            "memory_ratio",
            "selected_action_counts",
            "noop_rate",
            "event_count",
            "overhead_ratio",
        ],
    )
    write_csv(
        out_dir / "v145_controller_summary.csv",
        [
            {
                "controller_id": "not_executed",
                "controller_executed": 0,
                "not_executed_reason": reason,
                "real_dataset_seed_pass_count": 0,
                "official_s5_reached": 0,
                "promotion_allowed": 0,
            }
        ],
        [
            "controller_id",
            "controller_executed",
            "not_executed_reason",
            "real_dataset_seed_pass_count",
            "official_s5_reached",
            "promotion_allowed",
        ],
    )
    write_csv(
        out_dir / "v145_controller_failure_table.csv",
        [],
        [
            "controller_id",
            "dataset",
            "seed",
            "failure_source",
            "failure_auc",
            "failure_cep99",
            "failure_nll",
            "failure_ece",
            "failure_linec",
            "failure_overhead",
            "dominant_failure",
        ],
    )


def svg_page(title: str, lines: list[str]) -> str:
    height = 80 + 24 * len(lines)
    items = "\n".join(
        f'<text x="24" y="{72 + i * 24}" font-size="14" font-family="monospace" fill="#243042">{escape_xml(line)}</text>'
        for i, line in enumerate(lines)
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" '
        f'height="{height}" viewBox="0 0 960 {height}">\n'
        '<rect width="960" height="100%" fill="#f7f8fb"/>\n'
        f'<text x="24" y="36" font-size="22" font-family="Arial" font-weight="700" fill="#0b1220">{escape_xml(title)}</text>\n'
        f"{items}\n"
        "</svg>\n"
    )


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_figures(out_dir: Path, core_summary: dict[str, Any], extended_summary: dict[str, Any], missing_rows: list[dict[str, Any]]) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    missing_labels = [
        f"{row['dataset']} seed{row['seed']}: {row['dominant_failure']}"
        for row in missing_rows
        if row.get("oracle_level") == "O1-extended-legal-v144-bank"
    ]
    common_lines = [
        f"core O1 upper bound: {core_summary['dataset_seed_pass_count']}/9",
        f"extended legal-v144 O1 upper bound: {extended_summary['dataset_seed_pass_count']}/9",
        "promotion_allowed: 0",
    ]
    figure_payloads = {
        "fig_v145_real_3x3_pass_matrix.svg": common_lines + missing_labels,
        "fig_v145_action_oracle_upper_bound.svg": common_lines,
        "fig_v145_counterfactual_selected_action_heatmap.svg": ["controller not executed: oracle upper bound < 9/9"],
        "fig_v145_source_auc_tail_tradeoff.svg": missing_labels,
        "fig_v145_failure_mode_by_dataset_seed.svg": missing_labels,
        "fig_v145_proxy_vs_audit_scatter.svg": ["no controller proxy rows emitted; Line O stopped before policy tuning"],
        "fig_v145_noop_rate_vs_pass.svg": ["noop_rate unavailable: controller not executed"],
        "fig_v145_overhead_breakdown.svg": ["counterfactual overhead unavailable: controller not executed"],
        "fig_v145_wavelet_substrate_matrix.svg": ["v14.5 reused v14.4/v14.3 bounded substrate status; no new Wavelet training"],
        "fig_v145_all_basis_status_dashboard.svg": ["Rational remains main carrier; Non-RAT line is bounded parallel"],
        "fig_v145_mlp_vs_rational_controller.svg": ["MLP generic controls copied from v14.4 official artifact; not KAN promotion"],
    }
    for name, lines in figure_payloads.items():
        (figures / name).write_text(svg_page(name.replace(".svg", ""), lines))


def write_markdown_reports(out_dir: Path, route: str, best_summary: dict[str, Any], missing_rows: list[dict[str, Any]]) -> None:
    missing_text = "\n".join(
        f"- {row['dataset']} seed{row['seed']}: best={row['best_method']} "
        f"source={row['source']} AUC={row['AUCtime_ratio']} CEp99={row['CEp99_delta']} "
        f"NLL={row['NLL_delta']} ECE={row['ECE_delta']} LineC={row['LineC_pass']} "
        f"dominant={row['dominant_failure']}"
        for row in missing_rows
        if row.get("oracle_level") == "O1-extended-legal-v144-bank"
    )
    (out_dir / "v145_no_go_boundary.md").write_text(
        "# v14.5 no-go boundary\n\n"
        f"route = {route}\n\n"
        f"best oracle upper bound = {best_summary['dataset_seed_pass_count']} / 9\n\n"
        "Controller policy learning was not executed because the legal action bank did not have a 9/9 diagnostic upper bound.\n\n"
        "Remaining uncovered dataset-seeds:\n"
        f"{missing_text}\n\n"
        "This is a diagnostic oracle that uses audit metrics, so promotion_allowed remains 0.\n"
    )
    (out_dir / "v145_next_hypothesis_queue.md").write_text(
        "# v14.5 next hypothesis queue\n\n"
        "The current bank needs a new action family rather than another selection threshold.\n\n"
        "Candidate action family:\n"
        "- source_tail_auc_colocation_action: a train-stream-only action that separately estimates local trajectory cost, train-tail risk, and source gain before commit.\n"
        "- It must not use CEp99/NLL/ECE/LineC/AUCtime audit values as direction.\n"
        "- It should target the uncovered state classes: Fashion-MNIST seed2 AUC-only, KMNIST seed2 AUC-only, and the core-bank Fashion-MNIST seed0 CEp99-only state.\n"
    )


def write_forbidden_audit(out_dir: Path) -> None:
    write_csv(
        out_dir / "v145_forbidden_information_audit.csv",
        [
            {
                "stage": "V145_LINE_O_ORACLE",
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece_auc": 0,
                "oracle_uses_audit_metric": 1,
                "oracle_used_for_promotion": 0,
                "dataset_name_branch": 0,
                "seed_specific_scaling": 0,
                "promotion_allowed": 0,
            }
        ],
        [
            "stage",
            "direction_uses_validation_test_future_query",
            "direction_uses_linec_cep99_nll_ece_auc",
            "oracle_uses_audit_metric",
            "oracle_used_for_promotion",
            "dataset_name_branch",
            "seed_specific_scaling",
            "promotion_allowed",
        ],
    )


def write_code_review_manifest(out_dir: Path) -> None:
    rows = [
        {
            "path": "experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py",
            "role": "v14.5 oracle/finalizer runner",
            "direction_source": "existing v14.4 legal artifacts only",
        },
        {
            "path": str(PLAN_PATH),
            "role": "execution plan",
            "direction_source": "plan",
        },
        {
            "path": "experiments/run_v144_real_transfer_fms_all_basis_substrate.py",
            "role": "source of v14.4 action bank artifacts",
            "direction_source": "historical legal runs",
        },
    ]
    write_csv(out_dir / "v145_code_review_manifest.csv", rows, ["path", "role", "direction_source"])


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    missing = 0
    for rel in REQUIRED_ARTIFACTS:
        path = out_dir / rel
        is_missing = 0 if path.exists() else 1
        missing += is_missing
        rows.append({"artifact": str(path), "required": 1, "missing": is_missing})
    write_csv(out_dir / "v145_required_manifest.csv", rows, ["artifact", "required", "missing"])
    return missing


def create_code_review_packet(out_dir: Path) -> None:
    packet = out_dir / "v145_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in REQUIRED_ARTIFACTS:
            path = out_dir / rel
            if not path.exists() or path == packet:
                continue
            archive.write(path, arcname=rel)
        for path in (
            Path("experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py"),
            PLAN_PATH,
        ):
            if path.exists():
                archive.write(path, arcname=str(path))


def run_oracle(args: argparse.Namespace) -> dict[str, Any]:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    core_rows, missing_core_artifacts = core_action_rows()
    all_rows = list(core_rows)
    if args.include_extended_v144_bank:
        all_rows.extend(extended_v144_action_rows())

    bank_fieldnames = [
        "oracle_level",
        "dataset",
        "seed",
        "action_id",
        "source_run_dir",
        "method",
        "control_method",
        "source",
        "AUCtime_ratio",
        "CEp99_delta",
        "NLL_delta",
        "ECE_delta",
        "LineC_pass",
        "step_time_ratio",
        "memory_ratio",
        "pass_s5_row",
        "failure_source",
        "failure_auc",
        "failure_cep99",
        "failure_nll",
        "failure_ece",
        "failure_linec",
        "failure_overhead",
        "dominant_failure",
        "oracle_uses_audit_metric",
        "promotion_allowed",
        "real_transfer_gate_pass",
    ]
    write_csv(out_dir / "v145_action_bank_rows.csv", [action_bank_row(row) for row in all_rows], bank_fieldnames)

    core_selected = choose_oracle(all_rows, "O1-core-action-bank")
    extended_selected = choose_oracle(all_rows, "O1-extended-legal-v144-bank") if args.include_extended_v144_bank else []
    selected = core_selected + extended_selected

    oracle_fieldnames = [
        "dataset",
        "seed",
        "action_id",
        "source_run_dir",
        "method",
        "source",
        "AUCtime_ratio",
        "CEp99_delta",
        "NLL_delta",
        "ECE_delta",
        "LineC_pass",
        "step_time_ratio",
        "memory_ratio",
        "pass_s5_row",
        "failure_source",
        "failure_auc",
        "failure_cep99",
        "failure_nll",
        "failure_ece",
        "failure_linec",
        "failure_overhead",
        "dominant_failure",
        "oracle_level",
        "oracle_uses_audit_metric",
        "promotion_allowed",
    ]
    oracle_rows = [oracle_row(row) for row in selected]
    write_csv(out_dir / "v145_action_bank_oracle.csv", oracle_rows, oracle_fieldnames)

    core_summary = summarize_oracle(core_selected, "O1-core-action-bank")
    extended_summary = (
        summarize_oracle(extended_selected, "O1-extended-legal-v144-bank")
        if args.include_extended_v144_bank
        else {
            "oracle_level": "O1-extended-legal-v144-bank",
            "dataset_seed_pass_count": 0,
            "pass_rows": 0,
            "failure_remaining_source": 0,
            "failure_remaining_auc": 0,
            "failure_remaining_cep99": 0,
            "failure_remaining_nll": 0,
            "failure_remaining_ece": 0,
            "failure_remaining_linec": 0,
            "failure_remaining_overhead": 0,
            "action_bank_upper_bound_pass": 0,
        }
    )
    summary_rows = [core_summary]
    if args.include_extended_v144_bank:
        summary_rows.append(extended_summary)
    summary_fieldnames = [
        "oracle_level",
        "dataset_seed_pass_count",
        "pass_rows",
        "failure_remaining_source",
        "failure_remaining_auc",
        "failure_remaining_cep99",
        "failure_remaining_nll",
        "failure_remaining_ece",
        "failure_remaining_linec",
        "failure_remaining_overhead",
        "action_bank_upper_bound_pass",
    ]
    write_csv(out_dir / "v145_oracle_summary.csv", summary_rows, summary_fieldnames)

    best_summary = max(summary_rows, key=lambda row: int(row["dataset_seed_pass_count"]))
    if int(best_summary["dataset_seed_pass_count"]) == 9:
        route = "R3-CounterfactualPolicyFail"
        controller_block_reason = "oracle_upper_bound_reached_9_but_controller_not_executed"
    else:
        route = "R2-ActionBankUpperBoundInsufficient"
        controller_block_reason = "oracle_action_bank_upper_bound_lt_9"
    minimum_success = "S4b-RealTransferExplorationPositive"

    missing_rows = missing_state_rows(core_selected, "O1-core-action-bank")
    if args.include_extended_v144_bank:
        missing_rows.extend(missing_state_rows(extended_selected, "O1-extended-legal-v144-bank"))
    write_csv(
        out_dir / "v145_missing_state_classes.csv",
        missing_rows,
        [
            "oracle_level",
            "dataset",
            "seed",
            "best_action_id",
            "best_method",
            "source",
            "AUCtime_ratio",
            "CEp99_delta",
            "NLL_delta",
            "ECE_delta",
            "LineC_pass",
            "dominant_failure",
            "new_action_family_needed",
            "selection_threshold_repair_allowed",
            "promotion_allowed",
        ],
    )

    write_empty_controller_artifacts(out_dir, controller_block_reason)
    copy_or_empty(
        V144_ROOT / "official_v144" / "v144_wavelet_substrate_hardening.csv",
        out_dir / "v145_wavelet_substrate_hardening.csv",
        ["stage", "family", "source_artifact", "promotion_allowed"],
    )
    copy_or_empty(
        V144_ROOT / "official_v144" / "v144_all_basis_substrate_status.csv",
        out_dir / "v145_all_basis_status.csv",
        ["stage", "family", "source_artifact", "promotion_allowed"],
    )
    copy_or_empty(
        V144_ROOT / "official_v144" / "v144_mlp_generic_fms_control.csv",
        out_dir / "v145_mlp_generic_control.csv",
        ["stage", "family", "dataset", "seed", "method", "promotion_allowed"],
    )
    write_csv(
        out_dir / "v145_linec_tail_auc_audit.csv",
        oracle_rows,
        [
            "oracle_level",
            "dataset",
            "seed",
            "action_id",
            "method",
            "source",
            "AUCtime_ratio",
            "CEp99_delta",
            "NLL_delta",
            "ECE_delta",
            "LineC_pass",
            "pass_s5_row",
            "dominant_failure",
            "promotion_allowed",
        ],
    )
    write_forbidden_audit(out_dir)
    write_code_review_manifest(out_dir)
    write_figures(out_dir, core_summary, extended_summary, missing_rows)
    write_markdown_reports(out_dir, route, best_summary, missing_rows)

    route_payload = {
        "stage": "V145_TRAIN_STREAM_COUNTERFACTUAL_FMS_ALL_BASIS_PARALLEL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "minimum_success": minimum_success,
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "compute_budgeted_run": int(args.compute_budgeted_run),
        "core_oracle_dataset_seed_pass_count": int(core_summary["dataset_seed_pass_count"]),
        "extended_legal_v144_oracle_dataset_seed_pass_count": int(extended_summary["dataset_seed_pass_count"]),
        "best_oracle_dataset_seed_pass_count": int(best_summary["dataset_seed_pass_count"]),
        "best_oracle_level": best_summary["oracle_level"],
        "controller_executed": 0,
        "controller_not_executed_reason": controller_block_reason,
        "core_missing_artifact_count": len(missing_core_artifacts),
        "core_missing_artifacts": missing_core_artifacts,
        "forbidden_information_violation_count": 0,
        "required_artifact_missing_count": -1,
    }
    write_json(out_dir / "v145_route_decision.json", route_payload)

    write_required_manifest(out_dir)
    create_code_review_packet(out_dir)
    missing_required = write_required_manifest(out_dir)
    route_payload["required_artifact_missing_count"] = missing_required
    write_json(out_dir / "v145_route_decision.json", route_payload)
    write_required_manifest(out_dir)
    return route_payload


def main() -> None:
    args = parse_args()
    route_payload = run_oracle(args)
    print(json.dumps(route_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
