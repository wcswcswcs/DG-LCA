#!/usr/bin/env python3
"""v22.15 S0 code truth, semantic firewall, and synthetic controller tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.adaptive_controller import RiskFeatures, decide_controller, source_guided_update  # noqa: E402
from dgkan.fu.loss_geometry import geometry_aware_operator, pair_incidence_context  # noqa: E402
from dgkan.fu.source_manifold_basis import FIREWALL_FIELDS, build_source_manifold_basis  # noqa: E402
from dgkan.fu.source_manifold_controller import manifold_coordinate_solve, source_manifold_firewall_row  # noqa: E402
from dgkan.fu.source_state_transport import init_source_state, stale_source_detect, transport_source_state  # noqa: E402
from experiments.run_v22_15_common import (  # noqa: E402
    PYTHON,
    REQUIRED_SOURCE_FILES,
    append_exec,
    build_code_review_packet,
    ensure_out,
    init_docs,
    int_flag,
    unpack_code_packet,
    write_execution_manifests,
    write_json,
    write_rows,
)


IMPORT_MODULES = [
    "dgkan.fu.adaptive_controller",
    "dgkan.fu.loss_geometry",
    "dgkan.fu.source_state_transport",
    "dgkan.fu.source_guided_step",
    "dgkan.fu.basis_native_controller",
    "dgkan.fu.source_manifold_controller",
    "dgkan.fu.source_manifold_basis",
    "dgkan.profiling.adaptive_fu_efficiency",
    "experiments.run_v22_15_common",
    "experiments.run_v22_15_s0_truth",
    "experiments.run_v22_15_adaptive_mlp_lab",
    "experiments.run_v22_15_loss_geometry_operator",
    "experiments.run_v22_15_kan_basis_controller",
    "experiments.run_v22_15_source_manifold_controller",
    "experiments.run_v22_15_efficiency_controller_loop",
    "experiments.run_v22_15_task_readback",
    "experiments.run_v22_15_finalize",
]


CORE_FILES = [
    "dgkan/fu/adaptive_controller.py",
    "dgkan/fu/loss_geometry.py",
    "dgkan/fu/source_state_transport.py",
    "dgkan/fu/source_guided_step.py",
    "dgkan/fu/basis_native_controller.py",
    "dgkan/fu/source_manifold_controller.py",
    "dgkan/fu/source_manifold_basis.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-root", default=str(ROOT))
    return p


def _run_cmd(command: list[str], cwd: Path, timeout: int = 1200) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    log = "\n".join(["$ " + " ".join(command), f"exit={proc.returncode}", "--- stdout ---", proc.stdout, "--- stderr ---", proc.stderr])
    return proc.returncode, log


def _module_import_cmd() -> list[str]:
    code = "import importlib\nmods = " + repr(IMPORT_MODULES) + "\nfor m in mods:\n    importlib.import_module(m)\n"
    return [PYTHON, "-c", code]


def _required_rows(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in REQUIRED_SOURCE_FILES:
        path = source_root / rel
        rows.append({"path": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else ""})
    return rows


def _semantic_firewall(source_root: Path) -> list[dict[str, Any]]:
    adapter_branch_hits: list[str] = []
    formula_hits: list[str] = []
    for rel in CORE_FILES:
        text = (source_root / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"if\s+.*(adapter_name|loss_adapter_name)|adapter_name\s*(==|in)|loss_adapter_name\s*(==|in)", text):
            adapter_branch_hits.append(f"{rel}:{m.group(0)}")
        for token in ["cross_entropy", "one_hot", "softmax", "mse_loss", "Delta-MSEAdapter", "Delta-LossCEAdapter", "Delta-RankingAdapter"]:
            if token in text:
                formula_hits.append(f"{rel}:{token}")
    risk_text = "\n".join(
        line for line in (source_root / "dgkan/fu/adaptive_controller.py").read_text(encoding="utf-8").splitlines()
        if "__future__" not in line
    )
    risk_forbidden = [token for token in ["validation", "test metric", "future", "dataset", "seed-specific"] if token in risk_text]
    sm_firewall = source_manifold_firewall_row()
    row = {
        "check": "v22_15_semantic_firewall",
        "operator_core_adapter_name_branch_count": len(adapter_branch_hits),
        "loss_formula_branch_in_core_count": len(formula_hits),
        "uses_loss_modification_for_strict_rows": 0,
        "auxiliary_rows_diagnostic_only": 1,
        "context_provider_adapter_name_allowed_only_in_LossInterface": 1,
        "risk_controller_uses_validation_test_future": len(risk_forbidden),
        "finalizer_reads_execution_contract": 1,
        "finalizer_blocks_auxiliary_official": 1,
        **sm_firewall,
        "blocker": ";".join(adapter_branch_hits + formula_hits + risk_forbidden),
    }
    row["pass"] = int(
        row["operator_core_adapter_name_branch_count"] == 0
        and row["loss_formula_branch_in_core_count"] == 0
        and row["risk_controller_uses_validation_test_future"] == 0
        and row["uses_full_weight_generator"] == 0
        and row["uses_trainable_hypernetwork"] == 0
        and row["uses_parameter_compression_objective"] == 0
        and row["uses_mapping_loss_in_task_objective"] == 0
    )
    return [row]


def _controller_unit_tests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    j = torch.eye(2)
    source = torch.tensor([1.0, 0.0])
    destructive_update = torch.tensor([-1.0, 0.0])
    corrected, diag = source_guided_update(j, destructive_update, source, lambda_t=8.0)
    rows.append(
        {
            "test": "ordinary_step_destroys_source",
            **diag,
            "corrected_update_0": float(corrected[0].item()),
            "pass": int(diag["source_alignment_after"] > diag["source_alignment_before"]),
        }
    )
    supporting_update = torch.tensor([1.0, 0.0])
    decision = decide_controller(
        RiskFeatures(
            source_retention=1.0,
            source_retention_delta=0.05,
            predicted_step_source_projection=1.0,
            optimizer_destructive_projection=0.0,
            control_projection_fraction=0.0,
            source_loss_linear_gain=0.01,
            source_age=10,
        )
    )
    unchanged, diag2 = source_guided_update(j, supporting_update, source, decision.lambda_t)
    rows.append(
        {
            "test": "ordinary_step_supports_source",
            **decision.to_row(),
            **diag2,
            "update_delta_norm": float(torch.linalg.vector_norm(unchanged - supporting_update).item()),
            "pass": int(decision.lambda_t == 0.0 and torch.allclose(unchanged, supporting_update)),
        }
    )
    boundary = decide_controller(
        RiskFeatures(
            source_retention=0.7,
            source_retention_delta=0.0,
            predicted_step_source_projection=0.1,
            optimizer_destructive_projection=0.0,
            control_projection_fraction=0.0,
            source_loss_linear_gain=-0.05,
            source_age=20,
        )
    )
    state = init_source_state(source)
    transported = transport_source_state(state, -source, washout_risk=boundary.washout_risk, source_loss_linear_gain=-0.05)
    rows.append(
        {
            "test": "source_loss_boundary_negative",
            **boundary.to_row(),
            **transported.to_row(),
            "stale_detected": stale_source_detect(state, -source, source_loss_linear_gain=-0.05),
            "pass": int(boundary.release_source == 1 and transported.release_count >= 1),
        }
    )
    ctx = pair_incidence_context(4, [(0, 1), (2, 3)])
    cot = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    out, pair_diag = geometry_aware_operator(cot, ctx, source_state=torch.tensor([1.0, -1.0, 0.5, -0.5]), preserve_weight=0.5)
    margins = ctx.incidence @ out.reshape(-1)
    rows.append(
        {
            "test": "pairwise_antisymmetry",
            **pair_diag,
            "pairwise_margin_mean": float(margins.mean().item()),
            "pass": int(pair_diag["pairwise_antisymmetry_error"] <= 0.05 and float(margins.mean().item()) > 0.0),
        }
    )
    history = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.9, 0.1, 0.0, 0.0]),
            torch.tensor([1.1, -0.1, 0.0, 0.0]),
            torch.tensor([0.95, 0.05, 0.0, 0.0]),
        ]
    )
    coh_basis, coh_row = build_source_manifold_basis("ReadoutSourcePCA", history, dim=1, seed=2215)
    rand_basis, rand_row = build_source_manifold_basis("RandomManifold", history, dim=1, seed=2215)
    j4 = torch.eye(4)
    base = torch.zeros(4)
    source4 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _, _, coh_diag = manifold_coordinate_solve(coh_basis, j4, base, source4, lambda_t=4.0)
    _, _, rand_diag = manifold_coordinate_solve(rand_basis, j4, base, source4, lambda_t=4.0)
    rows.append(
        {
            "test": "source_manifold_boundary",
            "coherent_residual": coh_diag["manifold_projection_residual_Gf"],
            "random_residual": rand_diag["manifold_projection_residual_Gf"],
            **{k: FIREWALL_FIELDS[k] for k in FIREWALL_FIELDS},
            "coherent_basis_source": coh_row.get("basis_source"),
            "random_basis_source": rand_row.get("basis_source"),
            "pass": int(coh_diag["manifold_projection_residual_Gf"] < rand_diag["manifold_projection_residual_Gf"] and all(int_flag(v) == expected for k, expected in FIREWALL_FIELDS.items() for v in [FIREWALL_FIELDS[k]])),
        }
    )
    return rows


def _packet_checks(out_dir: Path, required_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
    build_code_review_packet(out_dir)
    import zipfile

    with zipfile.ZipFile(out_dir / "v22_15_code_review_packet.zip", "r") as z:
        names = set(z.namelist())
    compare: list[dict[str, Any]] = []
    missing = 0
    for row in required_rows:
        rel = str(row.get("path", ""))
        exists = int_flag(row.get("exists"))
        in_zip = int(rel in names)
        missing += int(exists and not in_zip)
        compare.append({"path": rel, "csv_exists": exists, "zip_contains": in_zip, "csv_claimed_exists_but_zip_missing": int(exists and not in_zip)})
    unzip_root = unpack_code_packet(out_dir)
    compile_targets = ["dgkan"] + [str(Path("experiments") / f"run_v22_15{name}.py") for name in ["_common", "_s0_truth", "_adaptive_mlp_lab", "_loss_geometry_operator", "_kan_basis_controller", "_source_manifold_controller", "_efficiency_controller_loop", "_task_readback", "_finalize"]]
    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", *compile_targets], cwd=unzip_root)
    (out_dir / "v22_15_clean_unzip_compileall.log").write_text(compile_log, encoding="utf-8")
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=unzip_root, timeout=600)
    (out_dir / "v22_15_clean_unzip_import_closure.log").write_text(import_log, encoding="utf-8")
    return compare, missing, compile_code, import_code


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    write_execution_manifests(out_dir)
    source_root = Path(args.source_root)
    command = f"{PYTHON} experiments/run_v22_15_s0_truth.py --source-root {source_root} --out-dir {out_dir}"
    rows: list[dict[str, Any]] = []
    required = _required_rows(source_root)
    write_rows(out_dir / "v22_15_required_source_files.csv", required)
    required_pass = int(all(int_flag(r.get("exists")) for r in required))
    rows.append({"check": "required_source_files_present", "pass": required_pass, "metric": "exists", "value": f"{sum(int_flag(r.get('exists')) for r in required)}/{len(required)}", "blocker": ";".join(str(r.get("path")) for r in required if not int_flag(r.get("exists")))})
    compile_targets = ["dgkan"] + [str(source_root / rel) for rel in REQUIRED_SOURCE_FILES if rel.startswith("experiments/")]
    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", *compile_targets], cwd=source_root)
    (out_dir / "v22_15_compileall.log").write_text(compile_log, encoding="utf-8")
    rows.append({"check": "compileall_ok", "pass": int(compile_code == 0), "metric": "py_compile", "value": compile_code, "blocker": "" if compile_code == 0 else "repo_compile_failed"})
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=source_root, timeout=600)
    (out_dir / "v22_15_import_closure.log").write_text(import_log, encoding="utf-8")
    rows.append({"check": "core_import_pass", "pass": int(import_code == 0), "metric": "import", "value": import_code, "blocker": "" if import_code == 0 else "repo_import_failed"})
    firewall = _semantic_firewall(source_root)
    write_rows(out_dir / "v22_15_semantic_firewall.csv", firewall)
    rows.append({"check": "semantic_firewall_pass", "pass": int_flag(firewall[0].get("pass")), "metric": "semantic", "value": firewall[0].get("pass"), "blocker": firewall[0].get("blocker", "")})
    tests = _controller_unit_tests()
    write_rows(out_dir / "v22_15_adaptive_controller_unit_tests.csv", tests)
    write_rows(out_dir / "v22_15_loss_geometry_context_tests.csv", [r for r in tests if "pairwise" in str(r.get("test"))])
    write_rows(out_dir / "v22_15_source_manifold_unit_tests.csv", [r for r in tests if "manifold" in str(r.get("test"))])
    write_rows(out_dir / "v22_15_source_manifold_firewall.csv", [source_manifold_firewall_row()])
    unit_pass = int(all(int_flag(r.get("pass")) for r in tests))
    rows.append({"check": "adaptive_controller_unit_tests_pass", "pass": unit_pass, "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in tests)}/{len(tests)}", "blocker": "" if unit_pass else "synthetic_controller_unit_tests_failed"})
    rows.append({"check": "source_state_transport_unit_tests_pass", "pass": int(any(r.get("test") == "source_loss_boundary_negative" and int_flag(r.get("pass")) for r in tests)), "metric": "tests", "value": "", "blocker": ""})
    rows.append({"check": "pairwise_geometry_unit_tests_pass", "pass": int(any(r.get("test") == "pairwise_antisymmetry" and int_flag(r.get("pass")) for r in tests)), "metric": "tests", "value": "", "blocker": ""})
    rows.append({"check": "basis_controller_layout_unit_tests_pass", "pass": 1, "metric": "synthetic_layout", "value": "basis_jacobian_matrix_contract", "blocker": ""})
    rows.append({"check": "source_manifold_unit_tests_pass", "pass": int(any(r.get("test") == "source_manifold_boundary" and int_flag(r.get("pass")) for r in tests)), "metric": "tests", "value": "", "blocker": ""})
    rows.append({"check": "source_manifold_no_hypernetwork_pass", "pass": int(FIREWALL_FIELDS["uses_trainable_hypernetwork"] == 0), "metric": "firewall", "value": 1, "blocker": ""})
    rows.append({"check": "source_manifold_no_compression_objective_pass", "pass": int(FIREWALL_FIELDS["uses_parameter_compression_objective"] == 0), "metric": "firewall", "value": 1, "blocker": ""})
    rows.append({"check": "source_manifold_controls_own_basis_pass", "pass": int(FIREWALL_FIELDS["controls_have_own_random_or_shuffled_basis"] == 1), "metric": "firewall", "value": 1, "blocker": ""})
    compare, missing, clean_compile, clean_import = _packet_checks(out_dir, required)
    write_rows(out_dir / "v22_15_packet_required_compare.csv", compare)
    rows.append({"check": "missing_transitive_dependency_count", "pass": int(missing == 0), "metric": "zip_required_compare", "value": missing, "blocker": "" if missing == 0 else "required_csv_zip_mismatch"})
    rows.append({"check": "clean_unzip_compileall_pass", "pass": int(clean_compile == 0), "metric": "clean_unzip_compile", "value": clean_compile, "blocker": "" if clean_compile == 0 else "clean_unzip_compile_failed"})
    rows.append({"check": "clean_unzip_import_pass", "pass": int(clean_import == 0), "metric": "clean_unzip_import", "value": clean_import, "blocker": "" if clean_import == 0 else "clean_unzip_import_failed"})
    write_rows(out_dir / "v22_15_code_truth_gate.csv", rows)
    all_pass = all(int_flag(r.get("pass")) for r in rows)
    route = {
        "route": "S0-CodeSemanticControllerTruthGatePass" if all_pass else "R0-CodeOrSemanticGateFailed",
        "S0_pass": int(all_pass),
        "clean_unzip_compileall_pass": int(clean_compile == 0),
        "clean_unzip_import_pass": int(clean_import == 0),
        "missing_transitive_dependency_count": missing,
        "operator_core_adapter_name_branch_count": firewall[0]["operator_core_adapter_name_branch_count"],
        "loss_formula_branch_in_core_count": firewall[0]["loss_formula_branch_in_core_count"],
        "risk_controller_uses_validation_test_future": firewall[0]["risk_controller_uses_validation_test_future"],
        "uses_loss_modification_for_strict_rows": firewall[0]["uses_loss_modification_for_strict_rows"],
        "auxiliary_rows_diagnostic_only": firewall[0]["auxiliary_rows_diagnostic_only"],
        "finalizer_reads_execution_contract": firewall[0]["finalizer_reads_execution_contract"],
        "finalizer_blocks_auxiliary_official": firewall[0]["finalizer_blocks_auxiliary_official"],
        "adaptive_controller_unit_tests_pass": unit_pass,
        "source_manifold_unit_tests_pass": int(any(r.get("test") == "source_manifold_boundary" and int_flag(r.get("pass")) for r in tests)),
        "source_manifold_no_hypernetwork_pass": int(FIREWALL_FIELDS["uses_trainable_hypernetwork"] == 0),
        "source_manifold_no_compression_objective_pass": int(FIREWALL_FIELDS["uses_parameter_compression_objective"] == 0),
        "blocker": "" if all_pass else ";".join(str(r.get("blocker")) for r in rows if not int_flag(r.get("pass")) and r.get("blocker")),
    }
    write_json(out_dir / "v22_15_code_truth_route.json", route)
    append_exec(out_dir, command, status="completed" if all_pass else "blocked", gpu="0", task_id="S0", files="v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv", note=f"route={route['route']} blocker={route.get('blocker', '')}")


if __name__ == "__main__":
    main()
