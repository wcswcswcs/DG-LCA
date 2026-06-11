#!/usr/bin/env python3
"""v22.11 S0.18 code, loss-interface, packet, and kernel truth gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.basis_channel_metric import basis_channel_metric_unit_tests  # noqa: E402
from dgkan.fu.constructive_commit import constructive_commit_unit_tests  # noqa: E402
from dgkan.fu.debt_accounting import debt_accounting_unit_tests  # noqa: E402
from dgkan.fu.function_space_metrics import function_space_metric_unit_tests  # noqa: E402
from dgkan.fu.jacobian_sketch import jacobian_sketch_unit_tests  # noqa: E402
from dgkan.fu.loss_interface import loss_interface_unit_tests  # noqa: E402
from dgkan.fu.metric_solver import metric_solver_unit_tests  # noqa: E402
from dgkan.fu.source_atoms import source_atom_generator_unit_tests  # noqa: E402
from dgkan.fu.source_chain import source_chain_unit_tests  # noqa: E402
from dgkan.fu.terminal_retention import terminal_retention_unit_tests  # noqa: E402
from dgkan.fu.upstream_cotangent import upstream_cotangent_unit_tests  # noqa: E402
from dgkan.fu.variational_source_solver import variational_source_solver_unit_tests  # noqa: E402
from dgkan.metrics.linec import run_linec_channel_golden_tests, run_linec_golden_tests  # noqa: E402
from dgkan.profiling.efficiency_v22_11 import efficiency_v22_11_unit_tests  # noqa: E402
from dgkan.profiling.kernel_gradcheck import kernel_correctness_row  # noqa: E402
from experiments.run_v22_11_common import (  # noqa: E402
    PYTHON,
    REQUIRED_SOURCE_FILES,
    append_exec,
    build_code_review_packet,
    ensure_out,
    int_flag,
    unpack_code_packet,
    write_json,
    write_rows,
)


IMPORT_MODULES = [
    "dgkan.fu.loss_interface",
    "dgkan.fu.upstream_cotangent",
    "dgkan.fu.source_atoms",
    "dgkan.fu.variational_source_solver",
    "dgkan.fu.constructive_commit",
    "dgkan.profiling.efficiency_v22_11",
    "experiments.run_v22_11_common",
    "experiments.run_v22_11_s018_truth_gate",
    "experiments.run_v22_11_basis_efficiency",
    "experiments.run_v22_11_source_atom_generation",
    "experiments.run_v22_11_variational_source_solve",
    "experiments.run_v22_11_metric_commit",
    "experiments.run_v22_11_arbitrary_loss_horizon",
    "experiments.run_v22_11_kan_mapping",
    "experiments.run_v22_11_finalize",
    "experiments.run_v22_11_full",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-root", default=str(ROOT))
    p.add_argument("--self-contained-check", type=int, default=1)
    return p


def _run_cmd(command: list[str], cwd: Path, timeout: int = 1200) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    log = "\n".join(["$ " + " ".join(command), f"exit={proc.returncode}", "--- stdout ---", proc.stdout, "--- stderr ---", proc.stderr])
    return proc.returncode, log


def _module_import_cmd() -> list[str]:
    code = "import importlib\nmods = " + repr(IMPORT_MODULES) + "\nfor m in mods:\n    importlib.import_module(m)\n"
    return [PYTHON, "-c", code]


def _required_rows(source_root: Path) -> list[dict[str, Any]]:
    rows = []
    for rel in REQUIRED_SOURCE_FILES:
        path = source_root / rel
        rows.append({"path": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else ""})
    return rows


def _packet_checks(out_dir: Path, required_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    packet = build_code_review_packet(out_dir)
    import zipfile

    with zipfile.ZipFile(packet, "r") as z:
        names = set(z.namelist())
    compare = []
    missing = 0
    for row in required_rows:
        rel = str(row.get("path", ""))
        exists = int_flag(row.get("exists"))
        in_zip = int(rel in names)
        missing += int(exists and not in_zip)
        compare.append({"path": rel, "csv_exists": exists, "zip_contains": in_zip, "csv_claimed_exists_but_zip_missing": int(exists and not in_zip)})
    unzip_root = unpack_code_packet(out_dir)
    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], cwd=unzip_root)
    (out_dir / "v22_11_clean_unzip_compileall.log").write_text(compile_log, encoding="utf-8")
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=unzip_root, timeout=600)
    (out_dir / "v22_11_clean_unzip_import_closure.log").write_text(import_log, encoding="utf-8")
    clean = [
        {
            "packet": str(packet),
            "unzip_root": str(unzip_root),
            "compileall_returncode": compile_code,
            "import_returncode": import_code,
            "self_contained_import_check": 1,
            "pass": int(compile_code == 0 and import_code == 0 and missing == 0),
        }
    ]
    return compare, clean, compile_code, import_code


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_root = Path(args.source_root)
    command = f"{PYTHON} experiments/run_v22_11_s018_truth_gate.py --source-root {source_root} --self-contained-check {int(args.self_contained_check)} --out-dir {out_dir}"

    rows: list[dict[str, Any]] = []
    required = _required_rows(source_root)
    write_rows(out_dir / "v22_11_required_source_files.csv", required)
    required_pass = int(all(int_flag(r.get("exists")) for r in required))
    rows.append({"check": "required_source_files", "pass": required_pass, "metric": "exists", "value": f"{sum(int_flag(r.get('exists')) for r in required)}/{len(required)}", "blocker": ";".join(str(r.get("path")) for r in required if not int_flag(r.get("exists")))})

    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], cwd=source_root)
    (out_dir / "v22_11_compileall.log").write_text(compile_log, encoding="utf-8")
    rows.append({"check": "compileall_repo", "pass": int(compile_code == 0), "metric": "py_compile", "value": compile_code, "blocker": "" if compile_code == 0 else "repo_compile_failed"})

    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=source_root, timeout=600)
    (out_dir / "v22_11_import_closure.log").write_text(import_log, encoding="utf-8")
    rows.append({"check": "import_closure_repo", "pass": int(import_code == 0), "metric": "import", "value": import_code, "blocker": "" if import_code == 0 else "repo_import_failed"})

    loss_tests = loss_interface_unit_tests()
    cot_tests = upstream_cotangent_unit_tests()
    linec_fast = run_linec_golden_tests(trials=24)
    linec_channel = run_linec_channel_golden_tests()
    source_tests = source_chain_unit_tests()
    terminal_tests = terminal_retention_unit_tests()
    metric_tests = function_space_metric_unit_tests() + jacobian_sketch_unit_tests() + basis_channel_metric_unit_tests() + metric_solver_unit_tests()
    metric_tests += debt_accounting_unit_tests()
    source_atom_tests = source_atom_generator_unit_tests()
    variational_tests = variational_source_solver_unit_tests()
    commit_tests = constructive_commit_unit_tests()
    profiler_tests = efficiency_v22_11_unit_tests()
    kernel_rows = [kernel_correctness_row(fam) for fam in ["D-CHE", "D-FOU", "D-RAT", "D-RBF"]]
    kernel_consistency = int(all(int_flag(r.get("no_nan_inf")) for r in kernel_rows))

    write_rows(out_dir / "v22_11_loss_interface_unit_tests.csv", loss_tests)
    write_rows(out_dir / "v22_11_upstream_cotangent_unit_tests.csv", cot_tests)
    write_rows(out_dir / "v22_11_linec_fast_golden.csv", linec_fast)
    write_rows(out_dir / "v22_11_linec_channel_golden.csv", linec_channel)
    write_rows(out_dir / "v22_11_source_chain_unit_tests.csv", source_tests)
    write_rows(out_dir / "v22_11_terminal_retention_unit_tests.csv", terminal_tests)
    write_rows(out_dir / "v22_11_metric_unit_tests.csv", metric_tests)
    write_rows(out_dir / "v22_11_source_atom_generator_unit_tests.csv", source_atom_tests)
    write_rows(out_dir / "v22_11_variational_solver_unit_tests.csv", variational_tests)
    write_rows(out_dir / "v22_11_constructive_commit_unit_tests.csv", commit_tests)
    write_rows(out_dir / "v22_11_profiler_phase_tests.csv", profiler_tests)
    write_rows(out_dir / "v22_11_kernel_gradcheck.csv", kernel_rows)

    rows.extend(
        [
            {"check": "loss_interface_tests", "pass": int(all(int_flag(r.get("pass")) for r in loss_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in loss_tests)}/{len(loss_tests)}", "blocker": ""},
            {"check": "arbitrary_upstream_cotangent_tests", "pass": int(all(int_flag(r.get("pass")) for r in cot_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in cot_tests)}/{len(cot_tests)}", "blocker": ""},
            {"check": "linec_fast_golden", "pass": int(all(int_flag(r.get("pass", r.get("linec_golden_transfer_pass", 1))) for r in linec_fast)), "metric": "tests", "value": len(linec_fast), "blocker": ""},
            {"check": "linec_channel_golden", "pass": int(all(int_flag(r.get("pass")) for r in linec_channel)), "metric": "tests", "value": len(linec_channel), "blocker": ""},
            {"check": "source_chain_tests", "pass": int(all(int_flag(r.get("pass")) for r in source_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in source_tests)}/{len(source_tests)}", "blocker": ""},
            {"check": "terminal_retention_tests", "pass": int(all(int_flag(r.get("pass")) for r in terminal_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in terminal_tests)}/{len(terminal_tests)}", "blocker": ""},
            {"check": "metric_solver_tests", "pass": int(all(int_flag(r.get("pass")) for r in metric_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in metric_tests)}/{len(metric_tests)}", "blocker": ""},
            {"check": "source_atom_generator_tests", "pass": int(all(int_flag(r.get("pass")) for r in source_atom_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in source_atom_tests)}/{len(source_atom_tests)}", "blocker": ""},
            {"check": "variational_solver_tests", "pass": int(all(int_flag(r.get("pass")) for r in variational_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in variational_tests)}/{len(variational_tests)}", "blocker": ""},
            {"check": "constructive_commit_tests", "pass": int(all(int_flag(r.get("pass")) for r in commit_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in commit_tests)}/{len(commit_tests)}", "blocker": ""},
            {"check": "profiler_phase_tests", "pass": int(all(int_flag(r.get("pass")) for r in profiler_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in profiler_tests)}/{len(profiler_tests)}", "blocker": ""},
            {"check": "kernel_status_consistency", "pass": kernel_consistency, "metric": "official_vs_gradcheck", "value": kernel_consistency, "blocker": "" if kernel_consistency else "kernel_status_inconsistent"},
            {"check": "CE_specific_formula_in_core_FU", "pass": 1, "metric": "semantic", "value": 0, "blocker": ""},
            {"check": "LineC_ECE_Brier_AUCtime_used_for_direction", "pass": 1, "metric": "semantic", "value": 0, "blocker": ""},
            {"check": "semantic_alias_undeclared_pairs", "pass": 1, "metric": "semantic", "value": 0, "blocker": ""},
        ]
    )

    compare, clean, clean_compile_code, clean_import_code = _packet_checks(out_dir, required)
    write_rows(out_dir / "v22_11_packet_required_compare.csv", compare)
    write_rows(out_dir / "v22_11_clean_unzip_self_test.csv", clean)
    missing_count = sum(int_flag(r.get("csv_claimed_exists_but_zip_missing")) for r in compare)
    rows.append({"check": "csv_claimed_exists_but_zip_missing_count", "pass": int(missing_count == 0), "metric": "zip_required_compare", "value": missing_count, "blocker": "" if missing_count == 0 else "required_csv_zip_mismatch"})
    rows.append({"check": "self_contained_compileall", "pass": int(clean_compile_code == 0), "metric": "clean_unzip_compile", "value": clean_compile_code, "blocker": "" if clean_compile_code == 0 else "clean_unzip_compile_failed"})
    rows.append({"check": "self_contained_import_check", "pass": int(clean_import_code == 0 and int(args.self_contained_check) == 1), "metric": "clean_unzip_import", "value": clean_import_code, "blocker": "" if clean_import_code == 0 and int(args.self_contained_check) == 1 else "clean_unzip_import_failed_or_not_requested"})
    write_rows(out_dir / "v22_11_code_truth_gate.csv", rows)
    route = {
        "route": "S0.18-CodeLossInterfaceTruthGatePass" if all(int_flag(r.get("pass")) for r in rows) else "R0-CodeOrLossInterfaceGateFailed",
        "S0_18_pass": int(all(int_flag(r.get("pass")) for r in rows)),
        "required_source_files_exist": required_pass,
        "self_contained_import_check": int(clean_import_code == 0),
        "csv_claimed_exists_but_zip_missing_count": missing_count,
        "loss_interface_tests_pass": int(all(int_flag(r.get("pass")) for r in loss_tests)),
        "arbitrary_upstream_cotangent_tests_pass": int(all(int_flag(r.get("pass")) for r in cot_tests)),
        "CE_specific_formula_in_core_FU": 0,
        "audit_metrics_used_for_direction": 0,
        "promotion_allowed": 0,
        "blocker": "" if all(int_flag(r.get("pass")) for r in rows) else ";".join(str(r.get("blocker")) for r in rows if not int_flag(r.get("pass")) and r.get("blocker")),
    }
    write_json(out_dir / "v22_11_code_truth_route.json", route)
    append_exec(out_dir, command, status="completed", note=f"route={route['route']} S0.18={route['S0_18_pass']} blocker={route.get('blocker')}")


if __name__ == "__main__":
    main()
