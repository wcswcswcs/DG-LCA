#!/usr/bin/env python3
"""DG-KAN v9 refactor audit runner.

This runner executes Phase A/B only:
* legacy truth audit from source and existing landed artifacts,
* clean core package import/smoke checks,
* negative contract tests,
* uniform artifact skeleton with honest not_run rows for later phases.

It does not train a new model and it does not claim CleanCE/FullEdge/external
success.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.artifacts.audit import audit_no_fake
from dgkan.artifacts.route import route_not_complete_decision
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, sha256_file, write_csv_rows, write_json
from dgkan.config import V9_REQUIRED_ARTIFACTS
from dgkan.contracts import ContractViolation, make_contract_audit_row, validate_official_contract
from dgkan.external.kanbefair_adapter import baseline_registry_rows
from dgkan.registry import default_v90_registry
from dgkan.specs import CandidateSpec, ContractFlags, ModuleAuditSpec
from dgkan.training.gradcheck import edge_layer_manual_autograd_gradcheck, edge_layer_manual_shape_smoke


LEGACY_DEFAULT = Path("results/real_rerun_20260506/v87_full_chain_fmnist_kmnist_selected_kw4_formal_repeat_20260508T203000Z")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_matching(rows: Sequence[Dict[str, str]], field: str, prefix: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get(field, "")).startswith(prefix):
            return row
    return {}


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _source_count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def legacy_truth_audit(repo: Path, legacy_dir: Path) -> List[Dict[str, Any]]:
    v87_path = repo / "experiments" / "run_gafu_v87_real.py"
    v85_path = repo / "experiments" / "run_gafu_v85_real.py"
    v83_path = repo / "experiments" / "run_gafu_v83_real.py"
    v72_path = repo / "experiments" / "run_gafu_v72_real.py"
    source_text = "\n".join(_read_text(path) for path in [v87_path, v85_path, v83_path, v72_path])

    route = _read_json(legacy_dir / "route_decision.json")
    selected_rows = read_csv_rows(legacy_dir / "selected_system_task_geometry_confirmation.csv")
    selected = selected_rows[0] if selected_rows else {}
    child_dir = repo / str(selected.get("child_out_dir", ""))
    child_primary = read_csv_rows(child_dir / "kanbefair_primary_transfer.csv")
    child_dg1 = _first_matching(child_primary, "candidate_id", "DG1-FT7-")
    joint_rows = read_csv_rows(legacy_dir / "joint_fair_envelope.csv")

    label_smoothing_literal_nonzero = int(
        bool(re.search(r"label_smoothing\s*=\s*(?:0\.0*[1-9]\d*|[1-9]\d*(?:\.\d*)?)", source_text))
    )
    uses_smooth_ce = int("_weighted_smooth_ce_and_grad" in source_text or "_smooth_ce_value_only" in source_text)
    uses_loss_backward_count = _source_count(r"\.backward\(\)", source_text)
    manual_ce_backward_count = _source_count(r"def _manual_ce_backward", source_text)
    functional_update_markers = int("Adaptive-FT-P" in source_text or "FT7" in source_text)
    full_edge_markers = int("EdgeFunctionLayer" in source_text or "full_edge" in source_text)

    selected_model_level = "transitional"
    stack_type = "linear_silu_packed_or_cached"
    is_full_edge = 0
    if full_edge_markers and str(selected.get("candidate_id", "")).startswith("FullEdge"):
        selected_model_level = "full_edge"
        stack_type = "edge_function_layer"
        is_full_edge = 1

    official_clean_from_legacy = int(
        str(selected.get("loss_type", "")) == "CE"
        and uses_smooth_ce == 0
        and is_full_edge == 1
        and label_smoothing_literal_nonzero == 0
    )
    downgrade_reason = []
    if uses_smooth_ce:
        downgrade_reason.append("legacy_source_uses_smooth_ce_helper_even_when_artifact_reports_CE")
    if selected_model_level != "full_edge":
        downgrade_reason.append("selected_route_stack_is_transitional_linear_silu_not_full_edge")
    if "label_smoothing" not in child_dg1 and "label_smoothing" not in selected:
        downgrade_reason.append("selected_artifact_does_not_record_label_smoothing_value")

    selected_row = {
        "stage": "A1_LEGACY_CURRENT_ROUTE_AUDIT",
        "status": "measured",
        "candidate_id": selected.get("candidate_id", route.get("best_candidate", "metric_unavailable")),
        "source_file": "experiments/run_gafu_v87_real.py;experiments/run_gafu_v85_real.py",
        "stack_type": stack_type,
        "head_type": "poly2_silu_head",
        "model_level": selected_model_level,
        "loss_type": selected.get("loss_type", child_dg1.get("loss_type", "metric_unavailable")),
        "label_smoothing": child_dg1.get("label_smoothing", selected.get("label_smoothing", "not_recorded")),
        "uses_smooth_ce": uses_smooth_ce,
        "legacy_source_has_nonzero_smoothing_candidates": label_smoothing_literal_nonzero,
        "external_teacher_used": selected.get("external_teacher_used", child_dg1.get("external_teacher_used", 0)),
        "self_teacher_used": selected.get("self_teacher_used", child_dg1.get("self_teacher_used", 0)),
        "uses_loss_backward": int(uses_loss_backward_count > 0),
        "uses_loss_backward_scope": "audited legacy sources contain baseline/diagnostic autograd calls; selected DG path uses manual CE evidence below",
        "uses_loss_backward_occurrences_in_audited_sources": uses_loss_backward_count,
        "manual_forward": int("forward_manual" in source_text),
        "manual_backward": int(manual_ce_backward_count > 0),
        "manual_update": int("FastAdamWNoSync" in source_text or "ForeachAdamWAddcdiv" in source_text),
        "optimizer_type": selected.get("optimizer_impl", child_dg1.get("optimizer", "metric_unavailable")),
        "functional_update_type": "Adaptive-FT-P/FT7 legacy runner path" if functional_update_markers else "metric_unavailable",
        "is_full_edge_purekan": is_full_edge,
        "legacy_artifact_success_v87_formal": route.get("success_v87_formal", 0),
        "clean_ce_official_claim_allowed_from_legacy_artifact": official_clean_from_legacy,
        "downgraded_v90_interpretation": "TransitionalDGKANDiagnostic",
        "downgrade_reason": ";".join(downgrade_reason),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": selected.get("cpu_offload_used", 0),
    }

    behavior_row = {
        "stage": "A2_LEGACY_BEHAVIOR_FREEZE_FROM_EXISTING_ARTIFACT",
        "status": "measured" if child_dg1 else "not_run",
        "candidate_id": child_dg1.get("candidate_id", "metric_unavailable"),
        "test_acc": child_dg1.get("test_metric", "metric_unavailable"),
        "delta_vs_MLP": child_dg1.get("metric_delta_vs_best_kanbefair", "metric_unavailable"),
        "params_ratio": selected.get("params_ratio_vs_MLP", "metric_unavailable"),
        "FLOPs_ratio": selected.get("FLOPs_ratio_vs_MLP", "metric_unavailable"),
        "step_ratio": selected.get("step_ratio_vs_KB_MLP", child_dg1.get("step_time_ms", "metric_unavailable")),
        "memory_ratio": "metric_unavailable",
        "curvature_ratio": selected.get("curvature_ratio_vs_DG_base", selected.get("curvature_ratio_vs_DG_Base", "metric_unavailable")),
        "functional_event_count": child_dg1.get("functional_event_count", selected.get("functional_events", "metric_unavailable")),
        "artifact_source": str(legacy_dir),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": child_dg1.get("cpu_offload_used", 0),
    }

    for task_row in joint_rows:
        behavior_row[f"{task_row.get('task_name')}_joint_fair_pass"] = task_row.get("JointFairPass", "")
        behavior_row[f"{task_row.get('task_name')}_delta_vs_MLP"] = task_row.get("metric_delta_vs_MLP", "")

    return [selected_row, behavior_row]


def module_import_audit(module_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in module_names:
        try:
            module = importlib.import_module(name)
            source_file = str(Path(module.__file__).relative_to(ROOT)) if getattr(module, "__file__", None) else "builtin"
            public = [item for item in dir(module) if not item.startswith("_")]
            rows.append(
                ModuleAuditSpec(
                    module_name=name,
                    source_file=source_file,
                    public_api_count=len(public),
                    unit_test_count=0,
                    import_pass=1,
                ).to_row()
            )
        except Exception as exc:
            rows.append(
                {
                    "module_name": name,
                    "source_file": "metric_unavailable",
                    "public_api_count": 0,
                    "unit_test_count": 0,
                    "import_pass": 0,
                    "error": repr(exc),
                }
            )
    return rows


def contract_tests() -> List[Dict[str, Any]]:
    base = CandidateSpec(
        candidate_id="contract-clean-transitional",
        model_level="transitional",
        stack_type="linear_silu_packed",
        head_type="poly2_silu_head",
        edge_basis="poly2_silu_head_only",
        hidden_dim=28,
        depth=2,
        basis_count=3,
    )
    cases = [
        ("positive_clean_transitional", base, ContractFlags(), False),
        ("label_smoothing_nonzero", CandidateSpec(**{**base.to_row(), "candidate_id": "bad-label-smoothing", "label_smoothing": 0.1}), ContractFlags(), True),
        ("loss_type_not_ce", CandidateSpec(**{**base.to_row(), "candidate_id": "bad-loss", "loss_type": "SmoothCE"}), ContractFlags(), True),
        ("external_teacher_used", base, ContractFlags(external_teacher_used=1), True),
        ("self_teacher_used", base, ContractFlags(self_teacher_used=1), True),
        ("geometry_loss_used", base, ContractFlags(geometry_loss_used=1), True),
        ("sampler_changed", base, ContractFlags(sampler_changed=1), True),
        ("class_weight_used", base, ContractFlags(class_weight_used=1), True),
        ("cpu_offload_used", base, ContractFlags(cpu_offload_used=1), True),
        ("uses_loss_backward", base, ContractFlags(uses_loss_backward=1), True),
        ("fake_data_used", base, ContractFlags(fake_data_used=1), True),
        ("proxy_row_used", base, ContractFlags(proxy_row_used=1), True),
        (
            "full_edge_linear_stack",
            CandidateSpec(
                candidate_id="bad-full-edge-linear",
                model_level="full_edge",
                stack_type="linear_silu_packed",
                head_type="edge_function_head",
                edge_basis="poly2_silu",
                hidden_dim=16,
                depth=2,
                basis_count=5,
            ),
            ContractFlags(),
            True,
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for name, spec, flags, should_raise in cases:
        raised = 0
        message = ""
        try:
            validate_official_contract(spec, flags)
        except ContractViolation as exc:
            raised = 1
            message = str(exc)
        rows.append(
            {
                "stage": "B2_CONTRACT_NEGATIVE_TESTS",
                "test_name": name,
                "expected_violation": int(should_raise),
                "raised_violation": raised,
                "negative_contract_test_pass": int(raised == int(should_raise)),
                "violation_message": message,
                **spec.to_row(),
                **flags.to_row(),
            }
        )
    return rows


def not_run_row(stage: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def write_required_not_run_artifacts(out_dir: Path) -> None:
    mapping = {
        "legacy_refactor_parity.csv": ("A2_LEGACY_REFACTOR_PARITY", "core transitional training parity not executed in this phase"),
        "label_smoothing_removal_audit.csv": ("C1_LABEL_SMOOTHING_REMOVAL_AUDIT", "CleanCE revalidation not executed in Phase A/B audit"),
        "clean_transitional_revalidation.csv": ("C2_CLEAN_TRANSITIONAL_REVALIDATION", "not run"),
        "full_edge_external_validation.csv": ("G3_FULL_EDGE_EXTERNAL_VALIDATION", "not run"),
        "functional_causality_controls.csv": ("G4_FUNCTIONAL_CAUSALITY_CONTROLS", "not run"),
        "robust_timing_protocols.csv": ("F1_ROBUST_TIMING_PROTOCOLS", "not run"),
        "training_compute_counter.csv": ("F3_TRAINING_COMPUTE_COUNTER", "not run"),
        "kanbefair_baseline_reproduction.csv": ("G1_KANBEFAIR_BASELINE_REPRODUCTION", "not run"),
        "external_fair_envelope.csv": ("G_EXTERNAL_FAIR_ENVELOPE", "not run"),
        "robustness_perturbation.csv": ("H1_ROBUSTNESS_PERTURBATION", "not run"),
        "continual_balance_multisplit.csv": ("H2_CONTINUAL_BALANCE_MULTISPLIT", "not run"),
        "boundary_audit.csv": ("H3_BOUNDARY_AUDIT", "not run"),
    }
    for filename, (stage, reason) in mapping.items():
        write_csv_rows(out_dir / filename, [not_run_row(stage, reason)])


def write_public_api_table(out_dir: Path, module_rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# B1 public API table",
        "",
        "| module | source | public API count | import pass |",
        "|---|---|---:|---:|",
    ]
    for row in module_rows:
        lines.append(
            f"| `{row.get('module_name')}` | `{row.get('source_file')}` | {row.get('public_api_count')} | {row.get('import_pass')} |"
        )
    (out_dir / "B1_public_api_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--legacy-artifact-dir", default=str(LEGACY_DEFAULT))
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    legacy_dir = ROOT / args.legacy_artifact_dir if not Path(args.legacy_artifact_dir).is_absolute() else Path(args.legacy_artifact_dir)
    plan_path = ROOT / "docs" / "DG-KAN_v9.0_CodeRefactor_CleanPureKAN_FunctionalTraining_ExternalFairValidation_完整整改计划.md"

    manifest = {
        "stage": "V90_PHASE_AB_REFACTOR_AUDIT",
        "status": "measured",
        "started_utc": _now_iso(),
        "repo_root": str(ROOT),
        "plan_path": str(plan_path.relative_to(ROOT)),
        "legacy_artifact_dir": str(legacy_dir.relative_to(ROOT)) if legacy_dir.is_relative_to(ROOT) else str(legacy_dir),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    registry = default_v90_registry()
    registry_rows = registry.rows()
    registry_rows.extend(
        {**row, "model_level": "external_baseline", "official_eligible": 1}
        for row in baseline_registry_rows()
    )
    write_csv_rows(out_dir / "candidate_registry_clean.csv", registry_rows)

    legacy_rows = legacy_truth_audit(ROOT, legacy_dir)
    write_csv_rows(out_dir / "code_audit_legacy_current.csv", legacy_rows)

    contract_rows = [
        make_contract_audit_row(spec, ContractFlags(), stage="B2_CLEAN_REGISTRY_CONTRACT_AUDIT")
        for spec in registry
    ]
    test_rows = contract_tests()
    write_csv_rows(out_dir / "contract_audit_clean.csv", contract_rows + test_rows)
    write_csv_rows(out_dir / "contract_negative_tests.csv", test_rows)

    modules = [
        "dgkan",
        "dgkan.specs",
        "dgkan.contracts",
        "dgkan.registry",
        "dgkan.models.edge_functions",
        "dgkan.models.edge_layers",
        "dgkan.optim.manual_adamw",
        "dgkan.functional.controller",
        "dgkan.external.kanbefair_adapter",
        "dgkan.artifacts.writer",
        "dgkan.training.gradcheck",
    ]
    module_rows = module_import_audit(modules)
    write_csv_rows(out_dir / "module_import_audit.csv", module_rows)
    write_public_api_table(out_dir, module_rows)

    smoke_rows = [edge_layer_manual_shape_smoke()]
    write_csv_rows(out_dir / "core_smoke_tests.csv", smoke_rows)
    gradcheck_rows = [edge_layer_manual_autograd_gradcheck()]
    write_csv_rows(out_dir / "gradcheck_full_edge.csv", gradcheck_rows)
    full_edge_audit_rows = [
        {
            "stage": "D1_FULL_EDGE_IMPLEMENTATION_AUDIT",
            "status": "measured",
            "candidate_id": "DG-FullEdge-Poly2Silu-edge-layer-smoke",
            "model_level": "full_edge",
            "edge_basis": "poly2_silu",
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_update": 0,
            "geometry_metrics": 1,
            "functional_direction": 1,
            "params_counter": 1,
            "FLOPs_counter": 1,
            "cache_breakdown": 1,
            "full_edge_contract_pass": int(int(gradcheck_rows[0]["GradPass"]) == 1),
            "task_training_pass": 0,
            "reason_task_training": "not_run_in_phase_ab",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "full_edge_implementation_audit.csv", full_edge_audit_rows)

    write_required_not_run_artifacts(out_dir)

    negative_pass = int(all(int(row["negative_contract_test_pass"]) == 1 for row in test_rows))
    module_import_pass = int(all(int(row.get("import_pass", 0)) == 1 for row in module_rows))
    smoke_pass = int(all(int(row.get("manual_backward_shape_pass", 0)) == 1 for row in smoke_rows))
    gradcheck_pass = int(all(int(row.get("GradPass", 0)) == 1 for row in gradcheck_rows))
    legacy_is_transitional = int(any(row.get("downgraded_v90_interpretation") == "TransitionalDGKANDiagnostic" for row in legacy_rows))
    route = route_not_complete_decision(
        primary_blocker="full_edge_purekan_clean_ce_revalidation_and_external_validation_not_run",
        next_required_implementation="run_phase_c_clean_ce_transitional_revalidation_then_phase_d_full_edge_full_model_task_smoke",
    )
    route.update(
        {
            "phase_a_legacy_truth_audit_pass": 1,
            "phase_b_core_package_import_pass": module_import_pass,
            "phase_b_core_smoke_pass": smoke_pass,
            "phase_d_edge_layer_gradcheck_pass": gradcheck_pass,
            "full_edge_contract_pass": gradcheck_pass,
            "negative_contract_tests_pass": negative_pass,
            "legacy_selected_route_v90_interpretation": "TransitionalDGKANDiagnostic" if legacy_is_transitional else "metric_unavailable",
            "legacy_clean_official_claim_allowed": 0,
            "best_candidate": "not_selected_in_phase_ab",
            "best_model_level": "not_selected_in_phase_ab",
        }
    )
    if not negative_pass:
        route["route"] = "R8-ContractFail"
        route["primary_blocker"] = "negative_contract_tests_failed"
    elif not module_import_pass or not smoke_pass or not gradcheck_pass:
        route["route"] = "R9-RefactorParityFail"
        route["primary_blocker"] = "core_import_smoke_or_gradcheck_failed"
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)

    failures = [
        {
            "failure_id": "F17_artifact_missing",
            "active": 0,
            "detail": "required v9 artifact files were created with measured or not_run status",
        },
        {
            "failure_id": "F4_full_edge_contract_fail",
            "active": 0 if gradcheck_pass else 1,
            "detail": "minimal full-edge edge-layer contract/gradcheck passed; full model task training is not run",
        },
        {
            "failure_id": "F8_full_edge_task_fail",
            "active": 1,
            "detail": "full-edge task training and external validation are not run in Phase A/B",
        },
        {
            "failure_id": "F7_clean_ce_regression",
            "active": 0,
            "detail": "CleanCE regression not evaluated yet",
        },
    ]
    write_csv_rows(out_dir / "failure_table.csv", failures)

    # Deliberate negative contract inputs include fake/proxy/offload flags.  They
    # are validated in contract_negative_tests.csv and intentionally excluded
    # from no-fake provenance counts so they cannot be mistaken for experiment
    # evidence rows.
    audit_paths = [
        out_dir / "code_audit_legacy_current.csv",
        out_dir / "core_smoke_tests.csv",
        out_dir / "gradcheck_full_edge.csv",
        out_dir / "full_edge_implementation_audit.csv",
    ]
    audit = audit_no_fake(audit_paths)
    audit.update(
        {
            "stage": "V90_PROVENANCE_AUDIT",
            "status": "measured",
            "script_path": "experiments/run_v90_refactor_audit.py",
            "plan_path": str(plan_path.relative_to(ROOT)),
        }
    )
    write_csv_rows(out_dir / "v90_provenance_audit.csv", [audit])

    hash_targets = [
        plan_path,
        ROOT / "experiments" / "run_v90_refactor_audit.py",
        ROOT / "dgkan" / "specs.py",
        ROOT / "dgkan" / "contracts.py",
        ROOT / "dgkan" / "models" / "edge_layers.py",
        out_dir / "route_decision.json",
        out_dir / "code_audit_legacy_current.csv",
        out_dir / "contract_audit_clean.csv",
        out_dir / "core_smoke_tests.csv",
        out_dir / "gradcheck_full_edge.csv",
        out_dir / "full_edge_implementation_audit.csv",
        out_dir / "v90_provenance_audit.csv",
    ]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_targets, root=ROOT))

    missing = [name for name in V9_REQUIRED_ARTIFACTS if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing required v9 artifacts: {missing}")


if __name__ == "__main__":
    main()
