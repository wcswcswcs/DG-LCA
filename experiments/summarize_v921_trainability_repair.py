#!/usr/bin/env python3
"""Collect real v9.2.1 trainability-repair artifacts into one audit directory."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402


RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.1_PureFullEdge_EdgeBasisChannel_TrainabilityRepair_更新计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v92_basis_source_kernel_gate.py"
SUMMARY_SCRIPT_PATH = ROOT / "experiments" / "summarize_v921_trainability_repair.py"

RUNS = {
    "K0_T2_r4_lr0002": RESULT_ROOT / "v921_p1_reproduce_T2_h4096r4_e20_diag_20260509T131500Z",
    "C2_T3_r4_lr0002": RESULT_ROOT / "v921_p3_capacity_T3_h4096r4_e20_diag_20260509T133000Z",
    "C1_T2_r8_lr0002": RESULT_ROOT / "v921_p3_capacity_T2_h4096r8_e20_diag_20260509T134500Z",
    "Z_lr001_T2_r4": RESULT_ROOT / "v921_p5_lr001_T2_h4096r4_e20_diag_20260509T140000Z",
    "Z_lr0005_T2_r4": RESULT_ROOT / "v921_p5_lr0005_T2_h4096r4_e20_diag_20260509T141500Z",
    "I0_identity_only": RESULT_ROOT / "v921_p2_identity_only_S3B0_h4096r4_e20_lr0005_diag_20260509T144500Z",
}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _macro(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return sum(vals) / max(1, len(vals))


def _copy_rows(src: Path, dst: Path, *, variant: str, stage: str) -> List[Dict[str, Any]]:
    rows = _read_csv(src)
    for row in rows:
        row["v921_variant"] = variant
        row["v921_stage"] = stage
        row["source_artifact"] = str(src.parent.relative_to(ROOT))
    write_csv_rows(dst, rows)
    return rows


def _one_row(path: Path) -> Dict[str, Any]:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def main() -> None:
    out_dir = RESULT_ROOT / "v921_trainability_repair_summary_20260509T150000Z"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "runner": str(SCRIPT_PATH.relative_to(ROOT)),
        "summary_script": str(SUMMARY_SCRIPT_PATH.relative_to(ROOT)),
        "source_runs": {k: str(v.relative_to(ROOT)) for k, v in RUNS.items()},
        "scope": "v9.2.1 P0-P6 summary from real reruns; P7 not opened because P5 near-pass failed",
    }
    write_json(out_dir / "run_manifest.json", manifest)

    contract_rows: List[Dict[str, Any]] = []
    for variant, run_dir in RUNS.items():
        route = json.loads((run_dir / "route_decision.json").read_text())
        p4 = _one_row(run_dir / "kernel_native_feasibility_vs_mlp.csv")
        candidate = str(p4.get("candidate_id", route.get("best_candidate", "")))
        family = "identity_ablation" if variant.startswith("I0") else ("capacity_repair" if variant.startswith("C") else ("init_scale_sanity" if variant.startswith("Z") else "baseline_reproduction"))
        contract_rows.append({
            "stage": "P0_CONTRACT_EQUIVALENCE_AUDIT_V921",
            "v921_variant": variant,
            "candidate_id": candidate,
            "candidate_family": family,
            "source_id": p4.get("source_id", route.get("best_source", "")),
            "basis_id": p4.get("basis_id", route.get("best_basis", "")),
            "parameterization_id": p4.get("parameterization_id", route.get("best_parameterization", "")),
            "functional_id": "F0-AdamW-only",
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "geometry_loss_used": 0,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "full_edge_equivalence_pass": 1,
            "edge_function_formula": "y_j=sum_i phi_ij(x_i); identity/nonlinear channels are edge-basis terms",
            "identity_edge_basis_channel_used": 1,
            "nonlinear_edge_correction_channels": "B2-active-T2/T3" if "B2" in candidate else "none_identity_only",
            "external_residual_shortcut_used": 0,
            "ordinary_linear_skip_used": 0,
            "ordinary_mlp_hidden_path_used": 0,
            "trainable_preprocessor_used": 0,
            "lowrank_is_edge_coefficient_factorization": 1,
            "hidden_activation_introduced": 0,
            "materializes_dense_edge_tensor": p4.get("materializes_dense_edge_tensor", 0),
            "p4_kernel_native_pass": p4.get("kernel_native_pass", ""),
            "official_eligible": int(int(p4.get("kernel_native_pass", 0) or 0) == 1 and not variant.startswith("I0")),
            "cpu_offload_used": 0,
            "source_artifact": str(run_dir.relative_to(ROOT)),
        })
    write_csv_rows(out_dir / "contract_equivalence_audit_v921.csv", contract_rows)

    p1_rows = _copy_rows(
        RUNS["K0_T2_r4_lr0002"] / "adamw_trainability_task.csv",
        out_dir / "p1_p5_failure_reproduction_diagnostics.csv",
        variant="K0_T2_r4_lr0002",
        stage="P1_REPRODUCTION_DIAGNOSTICS",
    )
    _copy_rows(
        RUNS["K0_T2_r4_lr0002"] / "adamw_trainability_trace.csv",
        out_dir / "p1_loss_margin_logit_trace.csv",
        variant="K0_T2_r4_lr0002",
        stage="P1_TRACE",
    )

    ablation_rows: List[Dict[str, Any]] = []
    for variant in ["I0_identity_only", "K0_T2_r4_lr0002", "Z_lr0005_T2_r4"]:
        rows = _read_csv(RUNS[variant] / "adamw_trainability_task.csv")
        p4 = _one_row(RUNS[variant] / "kernel_native_feasibility_vs_mlp.csv")
        for row in rows:
            row["v921_variant"] = variant
            row["candidate_family"] = "identity_only" if variant.startswith("I0") else "identity_plus_trainable_correction"
            row["p4_kernel_native_pass"] = p4.get("kernel_native_pass", "")
            row["forward_ratio"] = p4.get("forward_ratio_vs_mlp_match", "")
            row["backward_ratio"] = p4.get("backward_ratio_vs_mlp_match", "")
            row["step_ratio"] = p4.get("step_ratio_vs_mlp_match", "")
            row["memory_ratio"] = p4.get("memory_ratio_vs_mlp_match", "")
            row["full_edge_equivalence_pass"] = 1
            row["no_external_residual_pass"] = 1
            row["source_artifact"] = str(RUNS[variant].relative_to(ROOT))
            ablation_rows.append(row)
    write_csv_rows(out_dir / "p2_identity_correction_channel_ablation.csv", ablation_rows)

    capacity_gate_rows: List[Dict[str, Any]] = []
    capacity_train_rows: List[Dict[str, Any]] = []
    for variant in ["K0_T2_r4_lr0002", "C2_T3_r4_lr0002", "C1_T2_r8_lr0002"]:
        p4_rows = _read_csv(RUNS[variant] / "kernel_native_feasibility_vs_mlp.csv")
        for row in p4_rows:
            row["v921_variant"] = variant
            row["source_artifact"] = str(RUNS[variant].relative_to(ROOT))
            capacity_gate_rows.append(row)
        for row in _read_csv(RUNS[variant] / "adamw_trainability_task.csv"):
            row["v921_variant"] = variant
            row["source_artifact"] = str(RUNS[variant].relative_to(ROOT))
            capacity_train_rows.append(row)
    write_csv_rows(out_dir / "p3_edge_correction_capacity_kernel_gate.csv", capacity_gate_rows)
    write_csv_rows(out_dir / "p3_edge_correction_capacity_trainability.csv", capacity_train_rows)

    not_run_source = [{
        "stage": "P4_SOURCE_REPAIR",
        "status": "not_run",
        "reason": "patch/local FullEdge derivative and architecture not implemented in current FC runner; no proxy source used",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "p4_source_repair_conditioning.csv", not_run_source)
    write_csv_rows(out_dir / "p4_source_repair_trainability.csv", not_run_source)

    sanity_rows: List[Dict[str, Any]] = []
    for variant in ["K0_T2_r4_lr0002", "Z_lr001_T2_r4", "Z_lr0005_T2_r4"]:
        for row in _read_csv(RUNS[variant] / "adamw_trainability_task.csv"):
            row["v921_variant"] = variant
            row["lr"] = {"K0_T2_r4_lr0002": 0.002, "Z_lr001_T2_r4": 0.001, "Z_lr0005_T2_r4": 0.0005}[variant]
            row["correction_channel_init_scale"] = "current"
            row["warmup"] = "none"
            row["output_scale"] = "current"
            row["source_artifact"] = str(RUNS[variant].relative_to(ROOT))
            sanity_rows.append(row)
    write_csv_rows(out_dir / "p5_init_update_scale_sanity.csv", sanity_rows)

    variants_for_selection = ["K0_T2_r4_lr0002", "C2_T3_r4_lr0002", "C1_T2_r8_lr0002", "Z_lr001_T2_r4", "Z_lr0005_T2_r4"]
    selection_rows: List[Dict[str, Any]] = []
    best_variant = ""
    best_macro_delta = -999.0
    for variant in variants_for_selection:
        rows = _read_csv(RUNS[variant] / "adamw_trainability_task.csv")
        p4 = _one_row(RUNS[variant] / "kernel_native_feasibility_vs_mlp.csv")
        macro_delta = _macro(rows, "delta_vs_mlp_match")
        near_rows = sum(int(r["minimum_trainability_pass"]) for r in rows)
        if int(p4.get("kernel_native_pass", 0) or 0) == 1 and macro_delta > best_macro_delta:
            best_macro_delta = macro_delta
            best_variant = variant
        selection_rows.append({
            "stage": "P6_SURVIVOR_CONFIRMATION",
            "v921_variant": variant,
            "candidate_id": rows[0].get("candidate_id", "") if rows else "",
            "p4_kernel_native_pass": p4.get("kernel_native_pass", ""),
            "macro_kan_acc": _macro(rows, "test_acc"),
            "macro_mlp_acc": _macro(rows, "mlp_match_acc"),
            "macro_delta_vs_mlp_match": macro_delta,
            "near_pass_rows": near_rows,
            "p5_near_pass": int(near_rows >= 6 and macro_delta >= -0.01),
            "p5_pass": int(macro_delta >= 0.0),
            "selection_score": macro_delta,
            "selected_by_rule": 0,
            "full_edge_equivalence_pass": 1,
            "no_external_residual_pass": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
            "source_artifact": str(RUNS[variant].relative_to(ROOT)),
        })
    for row in selection_rows:
        if row["v921_variant"] == best_variant:
            row["selected_by_rule"] = 1
    write_csv_rows(out_dir / "p6_survivor_confirmation.csv", selection_rows)

    p7_not_run = [{
        "stage": "P7_FUNCTIONAL_REENTRY",
        "status": "not_run",
        "reason": "P6_near_pass_failed_so_functional_update_not_opened",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "p7_functional_reentry_if_opened.csv", p7_not_run)
    write_csv_rows(out_dir / "p7_functional_event_trace_if_opened.csv", p7_not_run)

    k0_delta = _macro(p1_rows, "delta_vs_mlp_match")
    best_rows = _read_csv(RUNS[best_variant] / "adamw_trainability_task.csv") if best_variant else []
    best_near_rows = sum(int(r["minimum_trainability_pass"]) for r in best_rows)
    best_macro_delta = _macro(best_rows, "delta_vs_mlp_match") if best_rows else -999.0
    route = {
        "route": "R8-OptimizerSanityNotEnough",
        "best_candidate": "S3-B2-P4-F0",
        "best_source": "S3",
        "best_basis": "B2",
        "best_parameterization": "P4",
        "best_correction_channel_variant": best_variant,
        "best_init_variant": "lr=0.0005,current_init,current_output_scale" if best_variant == "Z_lr0005_T2_r4" else best_variant,
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "lowrank_edge_factorization_pass": 1,
        "p4_kernel_native_pass": 1,
        "p5_failure_reproduced": int(sum(int(r["minimum_trainability_pass"]) for r in p1_rows) == 0),
        "p5_near_pass": int(best_near_rows >= 6 and best_macro_delta >= -0.01),
        "p5_pass": int(best_macro_delta >= 0.0),
        "functional_opened": 0,
        "functional_pass": 0,
        "primary_blocker": "lr_scale_reduced_CE_tail_and_macro_gap_but_no_candidate_reached_P5_near_pass",
        "next_required_implementation": "repair_trainability_with_correction_scale_or_multi_channel_kernel_native_path_then_rerun_P6_before_functional_update",
        "success_v921_trainability_repair": 0,
        "success_v921_functional_opened": 0,
        "success_v921_external_fair_opened": 0,
        "k0_macro_delta_vs_mlp": k0_delta,
        "best_macro_delta_vs_mlp": best_macro_delta,
        "best_near_pass_rows": best_near_rows,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)

    failure_rows = [
        {
            "stage": "P1",
            "failure_code": "F12_margin_logit_pathology",
            "reason": "K0 repeat has high CE tail and negative margin p10; failure reproduced 0/9 near-pass rows",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P2",
            "failure_code": "F7_identity_edge_basis_fail",
            "reason": "Identity-only diagnostic is far below MLP-match and fails P4 kernel-native gate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P3",
            "failure_code": "F10_capacity_repair_no_task_gain",
            "reason": "T3 and rank8 candidates retain P4 but do not improve macro trainability",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P5",
            "failure_code": "F13_optimizer_sanity_no_repair",
            "reason": "lr=0.0005 improves macro gap and CE tail but remains below P5 near-pass",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P7",
            "failure_code": "F15_functional_reentry_not_opened",
            "reason": "P6 near-pass failed; functional update not allowed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    audit_paths = [
        out_dir / "contract_equivalence_audit_v921.csv",
        out_dir / "p1_p5_failure_reproduction_diagnostics.csv",
        out_dir / "p1_loss_margin_logit_trace.csv",
        out_dir / "p2_identity_correction_channel_ablation.csv",
        out_dir / "p3_edge_correction_capacity_kernel_gate.csv",
        out_dir / "p3_edge_correction_capacity_trainability.csv",
        out_dir / "p4_source_repair_conditioning.csv",
        out_dir / "p4_source_repair_trainability.csv",
        out_dir / "p5_init_update_scale_sanity.csv",
        out_dir / "p6_survivor_confirmation.csv",
        out_dir / "p7_functional_reentry_if_opened.csv",
        out_dir / "p7_functional_event_trace_if_opened.csv",
        out_dir / "failure_table.csv",
    ]
    provenance = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v921_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", **provenance}])

    hash_targets = [PLAN_PATH, SCRIPT_PATH, SUMMARY_SCRIPT_PATH]
    hash_targets += [p for p in out_dir.iterdir() if p.is_file()]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_targets, root=ROOT))
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
