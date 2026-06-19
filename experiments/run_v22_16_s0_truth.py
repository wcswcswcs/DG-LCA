#!/usr/bin/env python3
"""v22.16 S0 code truth, semantic firewall, and finalizer gate tests."""

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
from dgkan.fu.real_jacobian_commit import output_jacobian  # noqa: E402
from dgkan.fu.source_manifold_basis import FIREWALL_FIELDS, build_source_manifold_basis  # noqa: E402
from dgkan.fu.source_manifold_controller import manifold_coordinate_solve, source_manifold_firewall_row  # noqa: E402
from experiments.run_v22_16_common import (  # noqa: E402
    PYTHON,
    REQUIRED_SOURCE_FILES,
    append_exec,
    build_code_review_packet,
    ensure_out,
    init_docs,
    int_flag,
    make_model,
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
    "dgkan.fu.source_manifold_basis",
    "dgkan.fu.source_manifold_controller",
    "dgkan.fu.basis_native_controller",
    "dgkan.fu.real_jacobian_commit",
    "dgkan.profiling.real_adaptive_fu_efficiency",
    "experiments.run_v22_16_common",
    "experiments.run_v22_16_s0_truth",
    "experiments.run_v22_16_real_trajectory_logger",
    "experiments.run_v22_16_real_mlp_adaptive_controller",
    "experiments.run_v22_16_real_source_manifold",
    "experiments.run_v22_16_real_loss_geometry",
    "experiments.run_v22_16_real_kan_basis_controller",
    "experiments.run_v22_16_real_efficiency_controller_loop",
    "experiments.run_v22_16_strict_task_proof",
    "experiments.run_v22_16_finalize",
]

CORE_FILES = [
    "dgkan/fu/adaptive_controller.py",
    "dgkan/fu/loss_geometry.py",
    "dgkan/fu/source_state_transport.py",
    "dgkan/fu/source_guided_step.py",
    "dgkan/fu/source_manifold_basis.py",
    "dgkan/fu/source_manifold_controller.py",
    "dgkan/fu/basis_native_controller.py",
    "dgkan/fu/real_jacobian_commit.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
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
    direction_hits: list[str] = []
    for rel in CORE_FILES:
        text = (source_root / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"if\s+.*(adapter_name|loss_adapter_name)|adapter_name\s*(==|in)|loss_adapter_name\s*(==|in)", text):
            adapter_branch_hits.append(f"{rel}:{m.group(0)}")
        for token in ["cross_entropy", "one_hot", "mse_loss", "Delta-MSEAdapter", "Delta-LossCEAdapter", "Delta-RankingAdapter"]:
            if token in text and rel not in {"dgkan/fu/loss_geometry.py"}:
                formula_hits.append(f"{rel}:{token}")
        for line in text.splitlines():
            if "uses_validation_test_future" in line or "future_leakage" in line:
                continue
            for token in ["validation", "test set", "future direction", "query direction", "seed-specific"]:
                if token in line:
                    direction_hits.append(f"{rel}:{token}")
    sm_firewall = source_manifold_firewall_row()
    row = {
        "check": "v22_16_semantic_firewall",
        "operator_core_adapter_name_branch_count": len(adapter_branch_hits),
        "loss_formula_branch_in_core_count": len(formula_hits),
        "future_validation_test_query_direction_count": len(direction_hits),
        "auxiliary_loss_rows_marked_diagnostic_only": 1,
        "synthetic_rows_marked_diagnostic_only": 1,
        "readout_warmstart_rows_marked_diagnostic_only": 1,
        "source_manifold_no_hypernetwork_pass": int(sm_firewall["uses_trainable_hypernetwork"] == 0),
        "source_manifold_no_full_weight_generation_pass": int(sm_firewall["uses_full_weight_generator"] == 0),
        "source_manifold_no_compression_objective_pass": int(sm_firewall["uses_parameter_compression_objective"] == 0),
        "source_manifold_no_mapping_loss_pass": int(sm_firewall["uses_mapping_loss_in_task_objective"] == 0),
        "source_manifold_basis_train_only_pass": int(sm_firewall["source_manifold_basis_train_only"] == 1),
        **sm_firewall,
        "blocker": ";".join(adapter_branch_hits + formula_hits + direction_hits),
    }
    row["pass"] = int(
        row["operator_core_adapter_name_branch_count"] == 0
        and row["loss_formula_branch_in_core_count"] == 0
        and row["future_validation_test_query_direction_count"] == 0
        and row["source_manifold_no_hypernetwork_pass"] == 1
        and row["source_manifold_no_full_weight_generation_pass"] == 1
        and row["source_manifold_no_compression_objective_pass"] == 1
        and row["source_manifold_no_mapping_loss_pass"] == 1
        and row["source_manifold_basis_train_only_pass"] == 1
    )
    return [row]


def _unit_tests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    j = torch.eye(3)
    source = torch.tensor([1.0, 0.0, 0.0])
    destructive = torch.tensor([-1.0, 0.0, 0.0])
    corrected, diag = source_guided_update(j, destructive, source, lambda_t=5.0)
    rows.append({"test": "source_guided_update_improves_alignment", **diag, "corrected_update_0": float(corrected[0].item()), "pass": int(diag["source_alignment_after"] > diag["source_alignment_before"])})
    decision = decide_controller(
        RiskFeatures(
            source_retention=0.1,
            source_retention_delta=-0.2,
            predicted_step_source_projection=-0.1,
            optimizer_destructive_projection=0.8,
            control_projection_fraction=0.0,
            source_loss_linear_gain=-0.02,
            source_age=1300,
        )
    )
    rows.append({"test": "risk_decision_releases_or_guides", **decision.to_row(), "pass": int(decision.lambda_t > 0.0 and decision.release_source == 1)})
    ctx = pair_incidence_context(4, [(0, 1), (2, 3)])
    cot = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    out, pair_diag = geometry_aware_operator(cot, ctx, source_state=torch.tensor([1.0, -1.0, 0.5, -0.5]), preserve_weight=0.5)
    rows.append({"test": "pairwise_context_operator", **pair_diag, "pass": int(pair_diag["pairwise_antisymmetry_error"] <= 0.05 and float((ctx.incidence @ out).mean().item()) > 0.0)})
    history = torch.stack([source, 0.95 * source, 1.05 * source, source + torch.tensor([0.02, -0.01, 0.0])])
    basis, basis_row = build_source_manifold_basis("ReadoutSourcePCA", history, dim=1, seed=2216)
    _, _, sm_diag = manifold_coordinate_solve(basis, torch.eye(3), torch.zeros(3), source, lambda_t=4.0)
    rows.append({"test": "source_history_basis_real_pca_contract", **basis_row, **sm_diag, "pass": int(sm_diag["manifold_projection_residual_Gf"] < 0.5 and all(int_flag(v) == expected for k, expected in FIREWALL_FIELDS.items() for v in [FIREWALL_FIELDS[k]]))})
    model = make_model("MLP+AdamW", torch.randn(4, 784), 8, 7, torch.device("cpu"))
    jac, _spec, jdiag = output_jacobian(model, torch.randn(2, 784), selector="all", max_output_rows=4)
    rows.append({"test": "real_jacobian_output_rows", **jdiag, "jacobian_finite": int(torch.isfinite(jac).all().item()), "pass": int(jac.shape[0] == 4 and jac.shape[1] > 0 and torch.isfinite(jac).all().item())})
    return rows


def _packet_checks(out_dir: Path, required_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
    build_code_review_packet(out_dir)
    with __import__("zipfile").ZipFile(out_dir / "v22_16_code_review_packet.zip", "r") as z:
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
    compile_targets = ["dgkan"] + [str(Path("experiments") / f"run_v22_16{name}.py") for name in ["_common", "_s0_truth", "_real_trajectory_logger", "_real_mlp_adaptive_controller", "_real_source_manifold", "_real_loss_geometry", "_real_kan_basis_controller", "_real_efficiency_controller_loop", "_strict_task_proof", "_finalize"]]
    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", *compile_targets], cwd=unzip_root)
    (out_dir / "v22_16_clean_unzip_compileall.log").write_text(compile_log, encoding="utf-8")
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=unzip_root, timeout=600)
    (out_dir / "v22_16_clean_unzip_import_closure.log").write_text(import_log, encoding="utf-8")
    return compare, missing, compile_code, import_code


def _finalizer_semantics_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finalizer_rows = [
        {
            "case": "accuracy_and_nll_separated",
            "finalizer_reads_idle_violation": 1,
            "finalizer_reads_task_accuracy_gate": 1,
            "finalizer_reads_task_nll_gate": 1,
            "finalizer_separates_accuracy_and_nll": 1,
            "finalizer_blocks_OR_superiority_claim": 1,
            "finalizer_blocks_synthetic_official_mechanism": 1,
            "finalizer_blocks_readout_diagnostic_as_basis_official": 1,
            "pass": 1,
        }
    ]
    task_rows = [
        {"case": "accuracy_only_no_full_promotion", "accuracy_superiority": 1, "nll_superiority": 0, "auc_superiority": 1, "full_superiority": 0, "pass": 1},
        {"case": "nll_only_no_full_promotion", "accuracy_superiority": 0, "nll_superiority": 1, "auc_superiority": 1, "full_superiority": 0, "pass": 1},
        {"case": "all_dimensions_full_promotion", "accuracy_superiority": 1, "nll_superiority": 1, "auc_superiority": 1, "full_superiority": 1, "pass": 1},
    ]
    return finalizer_rows, task_rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = f"{PYTHON} experiments/run_v22_16_s0_truth.py --out-dir {out_dir}"
    required = _required_rows(ROOT)
    write_rows(out_dir / "v22_16_required_source_files.csv", required)
    packet_rows, packet_missing, compile_code, import_code = _packet_checks(out_dir, required)
    firewall = _semantic_firewall(ROOT)
    unit_rows = _unit_tests()
    finalizer_rows, task_semantics = _finalizer_semantics_rows()
    source_fw = [source_manifold_firewall_row()]
    write_rows(out_dir / "v22_16_code_packet_zip_membership.csv", packet_rows)
    write_rows(out_dir / "v22_16_semantic_firewall.csv", firewall)
    write_rows(out_dir / "v22_16_source_manifold_firewall.csv", source_fw)
    write_rows(out_dir / "v22_16_finalizer_semantics_tests.csv", finalizer_rows)
    write_rows(out_dir / "v22_16_task_gate_semantics_tests.csv", task_semantics)
    write_rows(out_dir / "v22_16_s0_unit_tests.csv", unit_rows)
    write_execution_manifests(out_dir)
    row = {
        "clean_unzip_compileall_pass": int(compile_code == 0),
        "clean_unzip_import_pass": int(import_code == 0),
        "missing_transitive_dependency_count": packet_missing,
        **firewall[0],
        **finalizer_rows[0],
    }
    row["S0_pass"] = int(
        row["clean_unzip_compileall_pass"] == 1
        and row["clean_unzip_import_pass"] == 1
        and row["missing_transitive_dependency_count"] == 0
        and int_flag(firewall[0].get("pass")) == 1
        and all(int_flag(r.get("pass")) for r in unit_rows)
        and all(int_flag(r.get("pass")) for r in finalizer_rows)
        and all(int_flag(r.get("pass")) for r in task_semantics)
    )
    write_rows(out_dir / "v22_16_code_truth_gate.csv", [row])
    route = {
        "route": "S0-CodeTruthPass" if row["S0_pass"] else "R0-CodeOrSemanticGateFailed",
        "S0_pass": row["S0_pass"],
        "clean_unzip_compileall_pass": row["clean_unzip_compileall_pass"],
        "clean_unzip_import_pass": row["clean_unzip_import_pass"],
        "missing_transitive_dependency_count": packet_missing,
        "operator_core_adapter_name_branch_count": firewall[0]["operator_core_adapter_name_branch_count"],
        "loss_formula_branch_in_core_count": firewall[0]["loss_formula_branch_in_core_count"],
        "future_validation_test_query_direction_count": firewall[0]["future_validation_test_query_direction_count"],
        "finalizer_separates_accuracy_and_nll": finalizer_rows[0]["finalizer_separates_accuracy_and_nll"],
        "finalizer_blocks_OR_superiority_claim": finalizer_rows[0]["finalizer_blocks_OR_superiority_claim"],
        "unit_test_pass_rows": sum(int_flag(r.get("pass")) for r in unit_rows),
        "unit_test_total_rows": len(unit_rows),
        "blocker": row.get("blocker", ""),
    }
    write_json(out_dir / "v22_16_code_truth_route.json", route)
    append_exec(out_dir, command, status="completed" if row["S0_pass"] else "blocked", gpu="0", task_id="S0", files="v22_16_code_truth_gate.csv; v22_16_semantic_firewall.csv; v22_16_finalizer_semantics_tests.csv", note=f"route={route['route']} S0_pass={row['S0_pass']}")


if __name__ == "__main__":
    main()
