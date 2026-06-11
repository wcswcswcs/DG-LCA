#!/usr/bin/env python3
"""v22.07 S0.14 code, solver, semantic, and packet truth gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.basis_channel_metric import basis_channel_metric_unit_tests  # noqa: E402
from dgkan.fu.debt_accounting import debt_accounting_unit_tests  # noqa: E402
from dgkan.fu.diffeomorphic_target import diffeomorphic_target_unit_tests  # noqa: E402
from dgkan.fu.fisher_metric import fisher_metric_unit_tests  # noqa: E402
from dgkan.fu.function_space_metrics import function_space_metric_unit_tests  # noqa: E402
from dgkan.fu.jacobian_sketch import jacobian_sketch_unit_tests  # noqa: E402
from dgkan.fu.metric_projection import metric_projection_unit_tests  # noqa: E402
from dgkan.fu.metric_solver import metric_solver_unit_tests  # noqa: E402
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
from experiments.run_v22_07_common import PYTHON, append_exec, ensure_out, run_cmd, write_json, write_rows  # noqa: E402


REQUIRED_SOURCE_FILES = V2206_REQUIRED_SOURCE_FILES + [
    "experiments/run_v22_07_common.py",
    "experiments/run_v22_07_s014_truth_gate.py",
    "experiments/run_v22_07_efficiency_reconfirm.py",
    "experiments/run_v22_07_drat_drbf_multibatch.py",
    "experiments/run_v22_07_metric_dynamics_fu.py",
    "experiments/run_v22_07_terminal_preservation.py",
    "experiments/run_v22_07_kan_source_mapping.py",
    "experiments/run_v22_07_finalize.py",
    "experiments/run_v22_07_full.py",
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


def _contract_rows_v2207() -> list[dict[str, Any]]:
    rows = []
    for row in _functional_semantic_contract():
        item = dict(row)
        item["v22_07_metric_dynamics_readback_required"] = 1
        item["source_estimator_stage"] = "C0-no-commit-audit"
        item["function_target_stage"] = "C1-target-contrast"
        item["metric_solver_stage"] = "C2-true-solver-readback"
        item["source_formation_stage"] = "C3-horizon-matrix"
        rows.append(item)
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
        write_rows(out_dir / "v22_07_required_source_files.csv", file_rows)
        rows.append(
            {
                "check": "required_source_files",
                "pass": int(all(int(r["exists"]) for r in file_rows)),
                "metric": "exists",
                "value": f"{sum(int(r['exists']) for r in file_rows)}/{len(file_rows)}",
                "blocker": ";".join(r["path"] for r in file_rows if not int(r["exists"])),
            }
        )

    if _mode_enabled(mode, "import_closure") or mode == "all":
        compile_cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "tests"]
        code, log = run_cmd(compile_cmd, cwd=source_root, timeout=1200)
        (out_dir / "v22_07_compileall.log").write_text(log, encoding="utf-8")
        write_rows(out_dir / "v22_07_compileall.csv", [{"command": " ".join(compile_cmd), "returncode": code, "pass": int(code == 0)}])
        import_code, import_log = run_cmd(
            [
                PYTHON,
                "-c",
                "import dgkan.fu.metric_solver, dgkan.fu.jacobian_sketch, dgkan.fu.basis_channel_metric, dgkan.profiling.efficiency_v22_06, experiments.run_v22_07_common, experiments.run_v22_07_s014_truth_gate",
            ],
            cwd=source_root,
            timeout=300,
        )
        (out_dir / "v22_07_import_closure.log").write_text(import_log, encoding="utf-8")
        import_pass = int(import_code == 0 and int(args.self_contained_import_check) == 1)
        write_rows(out_dir / "v22_07_import_closure.csv", [{"returncode": import_code, "pass": import_pass, "self_contained_import_check": int(args.self_contained_import_check)}])
        write_rows(out_dir / "v22_07_clean_unzip_self_test.csv", [{"source_root": str(source_root), "self_contained_import_check": int(args.self_contained_import_check), "pass": import_pass}])
        rows.extend(
            [
                {"check": "compileall", "pass": int(code == 0), "metric": "py_compile", "value": code, "blocker": "" if code == 0 else "compile_failed"},
                {"check": "import_closure", "pass": import_pass, "metric": "self_contained_import", "value": import_code, "blocker": "" if import_pass else "import_failed_or_not_self_contained"},
            ]
        )

    if _mode_enabled(mode, "linec_golden"):
        fast = run_linec_golden_tests(trials=24)
        channel = run_linec_channel_golden_tests()
        write_rows(out_dir / "v22_07_linec_fast_golden.csv", fast)
        write_rows(out_dir / "v22_07_linec_channel_golden.csv", channel)
        rows.extend(
            [
                {"check": "linec_fast_golden", "pass": int(all(int(r.get("pass", r.get("linec_golden_transfer_pass", 1))) for r in fast)), "metric": "tests", "value": f"{len(fast)}", "blocker": ""},
                {"check": "linec_channel_golden", "pass": int(all(int(r.get("pass", 0)) for r in channel)), "metric": "tests", "value": f"{len(channel)}", "blocker": ""},
            ]
        )

    if _mode_enabled(mode, "source_chain_tests"):
        tests = source_chain_unit_tests()
        write_rows(out_dir / "v22_07_source_chain_unit_tests.csv", tests)
        rows.append({"check": "source_chain_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "terminal_retention_tests"):
        tests = terminal_retention_unit_tests() + terminal_erosion_unit_tests() + source_preservation_unit_tests() + diffeomorphic_target_unit_tests() + debt_accounting_unit_tests()
        write_rows(out_dir / "v22_07_terminal_retention_unit_tests.csv", tests)
        rows.append({"check": "terminal_retention_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "metric_solver_tests"):
        tests = function_space_metric_unit_tests() + fisher_metric_unit_tests() + sobolev_metric_unit_tests() + rkhs_metric_unit_tests() + metric_projection_unit_tests() + jacobian_sketch_unit_tests() + basis_channel_metric_unit_tests() + metric_solver_unit_tests()
        write_rows(out_dir / "v22_07_function_space_metric_solver_unit_tests.csv", tests)
        rows.append({"check": "metric_solver_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "mechanism_contracts"):
        from dgkan.fu.mechanisms import MECHANISMS, mechanism_contract_rows

        contract = mechanism_contract_rows()
        selected = [r for r in contract if str(r.get("mechanism", "")) in EXPECTED_V2206_MECHANISMS]
        schema_rows, schema_ok, schema_blocker = _schema_rows(selected)
        semantic = _contract_rows_v2207()
        forbidden = _forbidden_direction_audit(semantic)
        alias_rows, disp_rows, alias_summary = _semantic_noncollapse_audit()
        write_rows(out_dir / "v22_07_mechanism_contracts.csv", selected)
        write_rows(out_dir / "v22_07_mechanism_contract_schema.csv", schema_rows)
        write_rows(out_dir / "v22_07_functional_semantic_contract.csv", semantic)
        write_rows(out_dir / "v22_07_forbidden_direction_audit.csv", forbidden)
        write_rows(out_dir / "v22_07_functional_alias_matrix.csv", alias_rows)
        write_rows(out_dir / "v22_07_function_displacement_alias_matrix.csv", disp_rows)
        write_rows(out_dir / "v22_07_semantic_noncollapse_summary.csv", alias_summary)
        present_ok = len(selected) == len(EXPECTED_V2206_MECHANISMS) and all(m in MECHANISMS for m in EXPECTED_V2206_MECHANISMS)
        alias_ok = bool(alias_summary) and int(alias_summary[0].get("pass", 0)) == 1
        forbidden_ok = all(int(r.get("forbidden_direction_pass", 0)) for r in forbidden)
        rows.append(
            {
                "check": "mechanism_contracts",
                "pass": int(present_ok and schema_ok and alias_ok and forbidden_ok),
                "metric": "v22.07 semantic+solver",
                "value": f"{len(selected)}/{len(EXPECTED_V2206_MECHANISMS)};alias_undeclared={alias_summary[0].get('undeclared_alias_pairs', '') if alias_summary else ''}",
                "blocker": ";".join(x for x in ["" if present_ok else "missing_mechanism", schema_blocker, "" if alias_ok else "undeclared_semantic_alias", "" if forbidden_ok else "forbidden_direction"] if x),
            }
        )

    if _mode_enabled(mode, "kernel_gradcheck"):
        kernel_rows = [kernel_correctness_row("D-CHE"), kernel_correctness_row("D-FOU"), kernel_correctness_row("D-RAT"), kernel_correctness_row("D-RBF")]
        write_rows(out_dir / "v22_07_kernel_gradcheck.csv", kernel_rows)
        rows.append({"check": "kernel_gradcheck", "pass": int(all(int(r.get("kernel_correctness_official_pass", r.get("pass", 0))) for r in kernel_rows)), "metric": "kernels", "value": f"{len(kernel_rows)}", "blocker": ""})

    if _mode_enabled(mode, "profiler_phase_tests"):
        tests = efficiency_v22_06_unit_tests()
        write_rows(out_dir / "v22_07_profiler_phase_tests.csv", tests)
        rows.append({"check": "profiler_phase_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if rows:
        write_rows(out_dir / "v22_07_code_truth_gate.csv", rows)
        route = {
            "mode": mode,
            "pass": int(all(int(r.get("pass", 0)) for r in rows)),
            "failed_checks": ";".join(r.get("check", "") for r in rows if not int(r.get("pass", 0))),
            "self_contained_import_check": int(args.self_contained_import_check),
        }
        write_json(out_dir / "v22_07_code_route_decision.json", route)
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_07_s014_truth_gate.py --mode {mode} --source-root {source_root} --self-contained-import-check {int(args.self_contained_import_check)} --out-dir {out_dir}",
            status="completed",
            note=f"pass={route['pass']} failed={route['failed_checks']}",
        )


if __name__ == "__main__":
    main()

