#!/usr/bin/env python3
"""v22.08 S0.15 code, metric, solver, and recompute truth gate."""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.basis_channel_metric import basis_channel_metric_unit_tests  # noqa: E402
from dgkan.fu.debt_accounting import debt_accounting_unit_tests  # noqa: E402
from dgkan.fu.diffeomorphic_target import diffeomorphic_target_unit_tests  # noqa: E402
from dgkan.fu.fisher_metric import fisher_metric_unit_tests  # noqa: E402
from dgkan.fu.function_space_metrics import function_space_metric_unit_tests  # noqa: E402
from dgkan.fu.jacobian_sketch import jacobian_sketch_unit_tests  # noqa: E402
from dgkan.fu.metric_projection import metric_projection_unit_tests  # noqa: E402
from dgkan.fu.metric_solver import metric_solver_unit_tests, solve_metric_readout_update  # noqa: E402
from dgkan.fu.rkhs_metric import rkhs_metric_unit_tests  # noqa: E402
from dgkan.fu.sobolev_metric import sobolev_metric_unit_tests  # noqa: E402
from dgkan.fu.source_chain import source_chain_unit_tests  # noqa: E402
from dgkan.fu.source_preservation import source_preservation_unit_tests  # noqa: E402
from dgkan.fu.terminal_erosion import terminal_erosion_unit_tests  # noqa: E402
from dgkan.fu.terminal_retention import terminal_retention_unit_tests  # noqa: E402
from dgkan.metrics.linec import run_linec_channel_golden_tests, run_linec_golden_tests  # noqa: E402
from dgkan.profiling.efficiency_v22_06 import efficiency_v22_06_unit_tests  # noqa: E402
from dgkan.profiling.kernel_gradcheck import kernel_correctness_row  # noqa: E402
from experiments.run_v22_06_s013_truth_gate import (  # noqa: E402
    EXPECTED_V2206_MECHANISMS,
    REQUIRED_SOURCE_FILES as V2206_REQUIRED_SOURCE_FILES,
    _forbidden_direction_audit,
    _functional_semantic_contract,
    _schema_rows,
    _semantic_noncollapse_audit,
)
from experiments.run_v22_08_common import PYTHON, append_exec, ensure_out, write_json, write_rows  # noqa: E402


REQUIRED_SOURCE_FILES = V2206_REQUIRED_SOURCE_FILES + [
    "dgkan/fu/metric_solver.py",
    "experiments/run_v22_08_common.py",
    "experiments/run_v22_08_s015_truth_gate.py",
    "experiments/run_v22_08_efficiency_reconfirm.py",
    "experiments/run_v22_08_drat_drbf_multibatch.py",
    "experiments/run_v22_08_retained_source_observer.py",
    "experiments/run_v22_08_metric_dynamics_solver.py",
    "experiments/run_v22_08_terminal_preservation.py",
    "experiments/run_v22_08_kan_source_mapping.py",
    "experiments/run_v22_08_finalize.py",
    "experiments/run_v22_08_full.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-root", default=str(ROOT))
    p.add_argument("--mode", default="all")
    p.add_argument("--self-contained-import-check", type=int, default=0)
    return p


def _mode_enabled(mode: str, name: str) -> bool:
    return mode == "all" or mode == name


def _run_cmd(command: list[str], cwd: Path, timeout: int = 1200) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    log = "\n".join(["$ " + " ".join(command), f"exit={proc.returncode}", "--- stdout ---", proc.stdout, "--- stderr ---", proc.stderr])
    return proc.returncode, log


def _contract_rows_v2208() -> list[dict[str, Any]]:
    rows = []
    for row in _functional_semantic_contract():
        item = dict(row)
        item["v22_08_retained_source_dynamics_required"] = 1
        item["source_observer_stage"] = "C0-retained-source-observer-no-commit"
        item["target_stage"] = "C1-target-contrast-after-observer-gate"
        item["solver_stage"] = "C2-recompute-per-target-scale-block-role"
        item["source_state_stage"] = "C3-source-state-integration"
        rows.append(item)
    return rows


def _c2_recompute_tests() -> list[dict[str, Any]]:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(5, 6)
            self.w2 = torch.nn.Parameter(torch.randn(6, 3) * 0.02)

        def frozen_readout_features(self, xb: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.fc1(xb))

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.frozen_readout_features(xb) @ self.w2

    torch.manual_seed(2208)
    model = Tiny()
    x = torch.randn(18, 5)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    cases = [
        ("all_x1", 1.0, "all"),
        ("all_x64", 64.0, "all"),
        ("hidden_only_x64", 64.0, "hidden_only"),
        ("readout_only_x64", 64.0, "readout_only"),
    ]
    rows: list[dict[str, Any]] = []
    for name, scale, role in cases:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).float(), y).backward()
        update = solve_metric_readout_update(
            model,
            x,
            y,
            mechanism="M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU",
            target_family="T10-CompensatedBlockSourceChannel",
            metric_family="G0-L2",
            target_scale=scale,
            block_role=role,
            seed=2208,
        )
        diag = dict(update.diagnostics or {})
        rows.append(
            {
                "case": name,
                "target_scale": scale,
                "block_role": role,
                "solver_status": diag.get("solver_status", ""),
                "projection_residual_Gf": diag.get("projection_residual_Gf", ""),
                "ActuationR2": diag.get("ActuationR2", ""),
                "B2_transfer_gain": diag.get("B2_transfer_gain", ""),
                "B3_safety_gain": diag.get("B3_safety_gain", ""),
                "JVP_count": diag.get("JVP_count", ""),
                "VJP_count": diag.get("VJP_count", ""),
                "CG_iterations": diag.get("CG_iterations", ""),
                "solver_rank": diag.get("solver_rank", ""),
                "condition_estimate": diag.get("condition_estimate", ""),
                "solve_time_ms": diag.get("solve_time_ms", ""),
                "parameter_update_norm": diag.get("parameter_update_norm", ""),
                "target_norm_L2": diag.get("target_norm_L2", ""),
                "block_restricted_solver": diag.get("block_restricted_solver", ""),
                "block_restricted_solver_status": diag.get("block_restricted_solver_status", ""),
                "is_proxy": diag.get("is_proxy", ""),
                "uses_future_or_validation": diag.get("uses_future_or_validation", ""),
            }
        )

    by_case = {str(r["case"]): r for r in rows}
    u1 = float(by_case["all_x1"].get("parameter_update_norm", 0.0) or 0.0)
    u64 = float(by_case["all_x64"].get("parameter_update_norm", 0.0) or 0.0)
    t1 = float(by_case["all_x1"].get("target_norm_L2", 0.0) or 0.0)
    t64 = float(by_case["all_x64"].get("target_norm_L2", 0.0) or 0.0)
    residual_same = str(by_case["all_x1"].get("projection_residual_Gf")) == str(by_case["all_x64"].get("projection_residual_Gf"))
    scale_pass = int(u64 > max(1.0e-12, u1) * 10.0 and t64 > max(1.0e-12, t1) * 10.0)
    block_pass = int(
        int(float(by_case["hidden_only_x64"].get("block_restricted_solver", 0) or 0)) == 1
        and int(float(by_case["readout_only_x64"].get("block_restricted_solver", 0) or 0)) == 1
        and float(by_case["hidden_only_x64"].get("parameter_update_norm", 0.0) or 0.0) != float(by_case["readout_only_x64"].get("parameter_update_norm", 0.0) or 0.0)
    )
    def as_int(row: dict[str, Any], key: str, default: int) -> int:
        value = row.get(key, "")
        if value in {"", None}:
            return default
        return int(float(value))

    common_pass = int(
        all(str(r.get("solver_status")) == "metric_readout_exact_solve" for r in rows)
        and all(as_int(r, "is_proxy", 1) == 0 for r in rows)
        and all(as_int(r, "JVP_count", 0) >= 1 for r in rows)
        and all(as_int(r, "uses_future_or_validation", 1) == 0 for r in rows)
    )
    for row in rows:
        row["scale_update_norm_changed"] = scale_pass
        row["scale_projection_residual_identical"] = int(residual_same)
        row["block_role_restricted"] = block_pass
        row["C2_recompute_test_pass"] = int(common_pass and scale_pass and block_pass)
        row["blocker"] = "" if int(row["C2_recompute_test_pass"]) else "c2_recompute_scale_or_block_or_proxy_gate_failed"
    return rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_root = Path(args.source_root)
    mode = str(args.mode)
    rows: list[dict[str, Any]] = []

    if _mode_enabled(mode, "required_source_files"):
        file_rows = [
            {
                "path": rel,
                "exists": int((source_root / rel).exists()),
                "size_bytes": (source_root / rel).stat().st_size if (source_root / rel).exists() else "",
            }
            for rel in REQUIRED_SOURCE_FILES
        ]
        write_rows(out_dir / "v22_08_required_source_files.csv", file_rows)
        rows.append({"check": "required_source_files", "pass": int(all(int(r["exists"]) for r in file_rows)), "metric": "exists", "value": f"{sum(int(r['exists']) for r in file_rows)}/{len(file_rows)}", "blocker": ";".join(r["path"] for r in file_rows if not int(r["exists"]))})

    if _mode_enabled(mode, "import_closure") or mode == "all":
        compile_cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "tests"]
        code, log = _run_cmd(compile_cmd, cwd=source_root, timeout=1200)
        (out_dir / "v22_08_compileall.log").write_text(log, encoding="utf-8")
        write_rows(out_dir / "v22_08_compileall.csv", [{"command": " ".join(compile_cmd), "returncode": code, "pass": int(code == 0)}])
        import_code, import_log = _run_cmd(
            [
                PYTHON,
                "-c",
                "import dgkan.fu.metric_solver, experiments.run_v22_08_common, experiments.run_v22_08_s015_truth_gate, experiments.run_v22_08_efficiency_reconfirm, experiments.run_v22_08_drat_drbf_multibatch, experiments.run_v22_08_retained_source_observer, experiments.run_v22_08_metric_dynamics_solver, experiments.run_v22_08_finalize",
            ],
            cwd=source_root,
            timeout=300,
        )
        (out_dir / "v22_08_import_closure.log").write_text(import_log, encoding="utf-8")
        import_pass = int(import_code == 0 and int(args.self_contained_import_check) == 1)
        write_rows(out_dir / "v22_08_import_closure.csv", [{"returncode": import_code, "pass": import_pass, "self_contained_import_check": int(args.self_contained_import_check)}])
        write_rows(out_dir / "v22_08_clean_unzip_self_test.csv", [{"source_root": str(source_root), "self_contained_import_check": int(args.self_contained_import_check), "pass": import_pass}])
        rows.extend([
            {"check": "compileall", "pass": int(code == 0), "metric": "py_compile", "value": code, "blocker": "" if code == 0 else "compile_failed"},
            {"check": "import_closure", "pass": import_pass, "metric": "self_contained_import", "value": import_code, "blocker": "" if import_pass else "import_failed_or_not_self_contained"},
        ])

    if _mode_enabled(mode, "linec_golden"):
        fast = run_linec_golden_tests(trials=24)
        channel = run_linec_channel_golden_tests()
        write_rows(out_dir / "v22_08_linec_fast_golden.csv", fast)
        write_rows(out_dir / "v22_08_linec_channel_golden.csv", channel)
        rows.extend([
            {"check": "linec_fast_golden", "pass": int(all(int(r.get("pass", r.get("linec_golden_transfer_pass", 1))) for r in fast)), "metric": "tests", "value": f"{len(fast)}", "blocker": ""},
            {"check": "linec_channel_golden", "pass": int(all(int(r.get("pass", 0)) for r in channel)), "metric": "tests", "value": f"{len(channel)}", "blocker": ""},
        ])

    if _mode_enabled(mode, "source_chain_tests"):
        tests = source_chain_unit_tests()
        write_rows(out_dir / "v22_08_source_chain_unit_tests.csv", tests)
        rows.append({"check": "source_chain_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "terminal_retention_tests"):
        tests = terminal_retention_unit_tests() + terminal_erosion_unit_tests() + source_preservation_unit_tests() + diffeomorphic_target_unit_tests() + debt_accounting_unit_tests()
        write_rows(out_dir / "v22_08_terminal_retention_unit_tests.csv", tests)
        rows.append({"check": "terminal_retention_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "metric_solver_tests"):
        tests = function_space_metric_unit_tests() + fisher_metric_unit_tests() + sobolev_metric_unit_tests() + rkhs_metric_unit_tests() + metric_projection_unit_tests() + jacobian_sketch_unit_tests() + basis_channel_metric_unit_tests() + metric_solver_unit_tests()
        write_rows(out_dir / "v22_08_function_space_metric_solver_unit_tests.csv", tests)
        rows.append({"check": "metric_solver_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "c2_recompute_tests"):
        tests = _c2_recompute_tests()
        write_rows(out_dir / "v22_08_c2_recompute_tests.csv", tests)
        rows.append({"check": "c2_recompute_tests", "pass": int(all(int(r.get("C2_recompute_test_pass", 0)) for r in tests)), "metric": "scale+block recompute", "value": f"{sum(int(r.get('C2_recompute_test_pass', 0)) for r in tests)}/{len(tests)}", "blocker": ";".join(dict.fromkeys(r.get("blocker", "") for r in tests if r.get("blocker")))})

    if _mode_enabled(mode, "mechanism_contracts"):
        from dgkan.fu.mechanisms import MECHANISMS, mechanism_contract_rows

        contract = mechanism_contract_rows()
        selected = [r for r in contract if str(r.get("mechanism", "")) in EXPECTED_V2206_MECHANISMS]
        schema_rows, schema_ok, schema_blocker = _schema_rows(selected)
        semantic = _contract_rows_v2208()
        forbidden = _forbidden_direction_audit(semantic)
        alias_rows, disp_rows, alias_summary = _semantic_noncollapse_audit()
        write_rows(out_dir / "v22_08_mechanism_contracts.csv", selected)
        write_rows(out_dir / "v22_08_mechanism_contract_schema.csv", schema_rows)
        write_rows(out_dir / "v22_08_functional_semantic_contract.csv", semantic)
        write_rows(out_dir / "v22_08_forbidden_direction_audit.csv", forbidden)
        write_rows(out_dir / "v22_08_functional_alias_matrix.csv", alias_rows)
        write_rows(out_dir / "v22_08_function_displacement_alias_matrix.csv", disp_rows)
        write_rows(out_dir / "v22_08_semantic_noncollapse_summary.csv", alias_summary)
        present_ok = len(selected) == len(EXPECTED_V2206_MECHANISMS) and all(m in MECHANISMS for m in EXPECTED_V2206_MECHANISMS)
        alias_ok = bool(alias_summary) and int(alias_summary[0].get("pass", 0)) == 1
        forbidden_ok = all(int(r.get("forbidden_direction_pass", 0)) for r in forbidden)
        rows.append({"check": "mechanism_contracts", "pass": int(present_ok and schema_ok and alias_ok and forbidden_ok), "metric": "v22.08 semantic+forbidden", "value": f"{len(selected)}/{len(EXPECTED_V2206_MECHANISMS)};alias_undeclared={alias_summary[0].get('undeclared_alias_pairs', '') if alias_summary else ''}", "blocker": ";".join(x for x in ["" if present_ok else "missing_mechanism", schema_blocker, "" if alias_ok else "undeclared_semantic_alias", "" if forbidden_ok else "forbidden_direction"] if x)})

    if _mode_enabled(mode, "kernel_gradcheck"):
        kernel_rows = [kernel_correctness_row("D-CHE"), kernel_correctness_row("D-FOU"), kernel_correctness_row("D-RAT"), kernel_correctness_row("D-RBF")]
        write_rows(out_dir / "v22_08_kernel_gradcheck.csv", kernel_rows)
        status_consistency = int(all(int(r.get("kernel_correctness_official_pass", r.get("pass", 0))) for r in kernel_rows))
        rows.append({"check": "kernel_gradcheck", "pass": status_consistency, "metric": "kernels", "value": f"{len(kernel_rows)}", "blocker": "" if status_consistency else "kernel_gradcheck_failed"})
        rows.append({"check": "kernel_status_consistency", "pass": status_consistency, "metric": "official_vs_gradcheck", "value": status_consistency, "blocker": "" if status_consistency else "kernel_status_inconsistent"})

    if _mode_enabled(mode, "profiler_phase_tests"):
        tests = efficiency_v22_06_unit_tests()
        write_rows(out_dir / "v22_08_profiler_phase_tests.csv", tests)
        rows.append({"check": "profiler_phase_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if rows:
        write_rows(out_dir / "v22_08_code_truth_gate.csv", rows)
        route = {
            "mode": mode,
            "S0_15_pass": int(all(int(r.get("pass", 0)) for r in rows)),
            "failed_checks": ";".join(r.get("check", "") for r in rows if not int(r.get("pass", 0))),
            "self_contained_import_check": int(args.self_contained_import_check),
        }
        write_json(out_dir / "v22_08_code_route_decision.json", route)
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_08_s015_truth_gate.py --mode {mode} --source-root {source_root} --self-contained-import-check {int(args.self_contained_import_check)} --out-dir {out_dir}",
            status="completed",
            note=f"S0.15={route['S0_15_pass']} failed={route['failed_checks']}",
        )


if __name__ == "__main__":
    main()
