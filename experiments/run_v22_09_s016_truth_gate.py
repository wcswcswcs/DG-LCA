#!/usr/bin/env python3
"""v22.09 code, solver, semantic, and self-contained packet truth gate."""

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
from dgkan.fu.debt_accounting import debt_accounting_unit_tests  # noqa: E402
from dgkan.fu.fisher_metric import fisher_metric_unit_tests  # noqa: E402
from dgkan.fu.function_space_metrics import function_space_metric_unit_tests  # noqa: E402
from dgkan.fu.jacobian_sketch import jacobian_sketch_unit_tests  # noqa: E402
from dgkan.fu.metric_projection import metric_projection_unit_tests  # noqa: E402
from dgkan.fu.metric_solver import metric_solver_unit_tests  # noqa: E402
from dgkan.fu.optimizer_state_integration import optimizer_state_integration_unit_tests  # noqa: E402
from dgkan.fu.retained_source_certificate import retained_source_certificate_unit_tests  # noqa: E402
from dgkan.fu.rkhs_metric import rkhs_metric_unit_tests  # noqa: E402
from dgkan.fu.sobolev_metric import sobolev_metric_unit_tests  # noqa: E402
from dgkan.fu.source_chain import source_chain_unit_tests  # noqa: E402
from dgkan.fu.source_state_dynamics import source_state_dynamics_unit_tests  # noqa: E402
from dgkan.fu.terminal_retention import terminal_retention_unit_tests  # noqa: E402
from dgkan.fu.train_flow_commutator import train_flow_commutator_unit_tests  # noqa: E402
from dgkan.metrics.linec import run_linec_channel_golden_tests, run_linec_golden_tests  # noqa: E402
from dgkan.profiling.efficiency_v22_09 import efficiency_v22_09_unit_tests  # noqa: E402
from experiments.run_v22_09_common import (  # noqa: E402
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
    "dgkan.fu.core",
    "dgkan.fu.source_chain",
    "dgkan.fu.terminal_retention",
    "dgkan.fu.debt_accounting",
    "dgkan.fu.function_space_metrics",
    "dgkan.fu.metric_projection",
    "dgkan.fu.metric_solver",
    "dgkan.fu.jacobian_sketch",
    "dgkan.fu.sobolev_metric",
    "dgkan.fu.rkhs_metric",
    "dgkan.fu.fisher_metric",
    "dgkan.fu.basis_channel_metric",
    "dgkan.fu.retained_source_certificate",
    "dgkan.fu.train_flow_commutator",
    "dgkan.fu.source_state_dynamics",
    "dgkan.fu.optimizer_state_integration",
    "dgkan.metrics.linec",
    "dgkan.profiling.efficiency_v22_06",
    "dgkan.profiling.efficiency_v22_09",
    "experiments.run_v22_09_common",
    "experiments.run_v22_09_s016_truth_gate",
    "experiments.run_v22_09_basis_efficiency_closure",
    "experiments.run_v22_09_drat_drbf_shape_localk_repair_scan",
    "experiments.run_v22_09_retained_source_observer",
    "experiments.run_v22_09_post_boundary_flow_replay_certificate",
    "experiments.run_v22_09_post_optimizer_state_holonomy_certificate",
    "experiments.run_v22_09_post_row_orthogonal_source_transport_certificate",
    "experiments.run_v22_09_post_info_volume_signal_channel_certificate",
    "experiments.run_v22_09_post_fast_slow_memory_resonance_certificate",
    "experiments.run_v22_09_post_fast_slow_memory_damped_repair",
    "experiments.run_v22_09_post_time_reversal_adjoint_certificate",
    "experiments.run_v22_09_post_time_reversal_adjoint_damped_repair",
    "experiments.run_v22_09_metric_dynamics_solver",
    "experiments.run_v22_09_terminal_preservation",
    "experiments.run_v22_09_kan_mapping",
    "experiments.run_v22_09_finalize",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-root", default=str(ROOT))
    p.add_argument("--mode", default="all")
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


def _semantic_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from experiments.run_v22_06_s013_truth_gate import _forbidden_direction_audit, _functional_semantic_contract, _semantic_noncollapse_audit

        semantic = [dict(r) for r in _functional_semantic_contract()]
        for row in semantic:
            row["v22_09_retained_source_observability_required"] = 1
            row["source_observer_stage"] = "C0-retained-source-certificate-no-commit"
            row["target_stage"] = "C1-target-contrast-after-C0"
            row["solver_stage"] = "C2-block-recompute"
            row["source_state_stage"] = "C3-optimizer-state-integrated-source-formation"
        forbidden = _forbidden_direction_audit(semantic)
        alias, displacement_alias, summary = _semantic_noncollapse_audit()
        return semantic, forbidden, alias, displacement_alias, summary
    except Exception:
        from dgkan.fu.mechanisms import mechanism_contract_rows

        contracts = mechanism_contract_rows()
    semantic = []
    forbidden = []
    alias = []
    displacement_alias = []
    seen_semantic: dict[tuple[str, str, str], str] = {}
    seen_disp: dict[tuple[str, str], str] = {}
    forbidden_tokens = ("future", "validation", "val_", "test_", "query", "LineC", "ECE", "Brier", "AUCtime", "CEp99")
    for row in contracts:
        mechanism = str(row.get("mechanism", ""))
        source = str(row.get("source", ""))
        kind = str(row.get("kind", ""))
        space = str(row.get("space", ""))
        sign_rule = str(row.get("sign_rule", ""))
        key = (kind, space, sign_rule)
        disp_key = (space, source)
        semantic.append(
            {
                "mechanism": mechanism,
                "kind": kind,
                "space": space,
                "sign_rule": sign_rule,
                "source": source,
                "v22_09_retained_source_observability_required": 1,
            }
        )
        hit = [tok for tok in forbidden_tokens if tok.lower() in source.lower()]
        forbidden.append(
            {
                "mechanism": mechanism,
                "source": source,
                "forbidden_tokens": ";".join(hit),
                "forbidden_direction_pass": int(not hit),
            }
        )
        if key in seen_semantic:
            alias.append({"mechanism_a": seen_semantic[key], "mechanism_b": mechanism, "semantic_key": "|".join(key), "declared_alias": 0})
        else:
            seen_semantic[key] = mechanism
        if disp_key in seen_disp:
            displacement_alias.append({"mechanism_a": seen_disp[disp_key], "mechanism_b": mechanism, "function_displacement_key": "|".join(disp_key), "declared_alias": 0})
        else:
            seen_disp[disp_key] = mechanism
    summary = [{"pass": int(len(alias) == 0), "undeclared_alias_pairs": len(alias), "fallback_semantic_audit": 1}]
    return semantic, forbidden, alias, displacement_alias, summary


def _packet_checks(out_dir: Path, required_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, Path]:
    packet = build_code_review_packet(out_dir)
    with __import__("zipfile").ZipFile(packet, "r") as z:
        names = set(z.namelist())
    compare = []
    missing = 0
    for row in required_rows:
        rel = str(row.get("path", ""))
        exists_in_csv = int_flag(row.get("exists"))
        in_zip = int(rel in names)
        if exists_in_csv and not in_zip:
            missing += 1
        compare.append({"path": rel, "csv_exists": exists_in_csv, "zip_contains": in_zip, "csv_claimed_exists_but_zip_missing": int(exists_in_csv and not in_zip)})
    unzip_root = unpack_code_packet(out_dir)
    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "tests"], cwd=unzip_root, timeout=1200)
    (out_dir / "v22_09_clean_unzip_compileall.log").write_text(compile_log, encoding="utf-8")
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=unzip_root, timeout=600)
    (out_dir / "v22_09_clean_unzip_import_closure.log").write_text(import_log, encoding="utf-8")
    clean_rows = [
        {
            "packet": str(packet),
            "unzip_root": str(unzip_root),
            "compileall_returncode": compile_code,
            "import_returncode": import_code,
            "self_contained_import_check": 1,
            "pass": int(compile_code == 0 and import_code == 0 and missing == 0),
        }
    ]
    return compare, clean_rows, compile_code, import_code, unzip_root


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_root = Path(args.source_root)
    rows: list[dict[str, Any]] = []

    required = _required_rows(source_root)
    write_rows(out_dir / "v22_09_required_source_files.csv", required)
    required_pass = int(all(int_flag(r.get("exists")) for r in required))
    rows.append(
        {
            "check": "required_source_files",
            "pass": required_pass,
            "metric": "exists",
            "value": f"{sum(int_flag(r.get('exists')) for r in required)}/{len(required)}",
            "blocker": ";".join(str(r.get("path")) for r in required if not int_flag(r.get("exists"))),
        }
    )

    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "tests"], cwd=source_root, timeout=1200)
    (out_dir / "v22_09_compileall.log").write_text(compile_log, encoding="utf-8")
    write_rows(out_dir / "v22_09_compileall.csv", [{"command": f"{PYTHON} -m compileall -q dgkan experiments tests", "returncode": compile_code, "pass": int(compile_code == 0)}])
    rows.append({"check": "compileall_repo", "pass": int(compile_code == 0), "metric": "py_compile", "value": compile_code, "blocker": "" if compile_code == 0 else "repo_compile_failed"})

    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=source_root, timeout=600)
    (out_dir / "v22_09_import_closure.log").write_text(import_log, encoding="utf-8")
    write_rows(out_dir / "v22_09_import_closure.csv", [{"returncode": import_code, "pass": int(import_code == 0), "self_contained_import_check": 0}])
    rows.append({"check": "import_closure_repo", "pass": int(import_code == 0), "metric": "import", "value": import_code, "blocker": "" if import_code == 0 else "repo_import_failed"})

    linec_fast = run_linec_golden_tests(trials=24)
    linec_channel = run_linec_channel_golden_tests()
    source_tests = source_chain_unit_tests()
    terminal_tests = terminal_retention_unit_tests()
    metric_tests = (
        function_space_metric_unit_tests()
        + fisher_metric_unit_tests()
        + sobolev_metric_unit_tests()
        + rkhs_metric_unit_tests()
        + metric_projection_unit_tests()
        + jacobian_sketch_unit_tests()
        + basis_channel_metric_unit_tests()
        + metric_solver_unit_tests()
    )
    new_tests = (
        retained_source_certificate_unit_tests()
        + train_flow_commutator_unit_tests()
        + source_state_dynamics_unit_tests()
        + optimizer_state_integration_unit_tests()
    )
    profiler_tests = efficiency_v22_09_unit_tests()
    write_rows(out_dir / "v22_09_linec_fast_golden.csv", linec_fast)
    write_rows(out_dir / "v22_09_linec_channel_golden.csv", linec_channel)
    write_rows(out_dir / "v22_09_source_chain_unit_tests.csv", source_tests)
    write_rows(out_dir / "v22_09_terminal_retention_unit_tests.csv", terminal_tests)
    write_rows(out_dir / "v22_09_function_space_metric_solver_unit_tests.csv", metric_tests)
    write_rows(out_dir / "v22_09_retained_source_observability_unit_tests.csv", new_tests)
    write_rows(out_dir / "v22_09_profiler_phase_tests.csv", profiler_tests)
    rows.extend(
        [
            {"check": "linec_fast_golden", "pass": int(all(int_flag(r.get("pass", r.get("linec_golden_transfer_pass", 1))) for r in linec_fast)), "metric": "tests", "value": len(linec_fast), "blocker": ""},
            {"check": "linec_channel_golden", "pass": int(all(int_flag(r.get("pass")) for r in linec_channel)), "metric": "tests", "value": len(linec_channel), "blocker": ""},
            {"check": "source_chain_tests", "pass": int(all(int_flag(r.get("pass")) for r in source_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in source_tests)}/{len(source_tests)}", "blocker": ""},
            {"check": "terminal_retention_tests", "pass": int(all(int_flag(r.get("pass")) for r in terminal_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in terminal_tests)}/{len(terminal_tests)}", "blocker": ""},
            {"check": "metric_solver_tests", "pass": int(all(int_flag(r.get("pass")) for r in metric_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in metric_tests)}/{len(metric_tests)}", "blocker": ""},
            {"check": "retained_source_observability_tests", "pass": int(all(int_flag(r.get("pass")) for r in new_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in new_tests)}/{len(new_tests)}", "blocker": ""},
            {"check": "profiler_phase_tests", "pass": int(all(int_flag(r.get("pass")) for r in profiler_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in profiler_tests)}/{len(profiler_tests)}", "blocker": ""},
        ]
    )

    semantic, forbidden, alias, displacement_alias, alias_summary = _semantic_rows()
    write_rows(out_dir / "v22_09_semantic_contract.csv", semantic)
    write_rows(out_dir / "v22_09_semantic_alias_matrix.csv", alias)
    write_rows(out_dir / "v22_09_function_displacement_alias_matrix.csv", displacement_alias)
    write_rows(out_dir / "v22_09_semantic_noncollapse_summary.csv", alias_summary)
    write_rows(out_dir / "v22_09_forbidden_direction_audit.csv", forbidden)
    forbidden_pass = int(all(int_flag(r.get("forbidden_direction_pass")) for r in forbidden))
    alias_undeclared = int_flag(alias_summary[0].get("undeclared_alias_pairs")) if alias_summary else len(alias)
    rows.append(
        {
            "check": "semantic_contract",
            "pass": int(forbidden_pass and alias_undeclared == 0),
            "metric": "forbidden+alias",
            "value": f"forbidden_pass={forbidden_pass};undeclared_alias_pairs={alias_undeclared}",
            "blocker": ";".join(x for x in ["" if forbidden_pass else "forbidden_direction", "" if alias_undeclared == 0 else "undeclared_semantic_alias_pairs"] if x),
        }
    )

    compare, clean_rows, clean_compile_code, clean_import_code, unzip_root = _packet_checks(out_dir, required)
    write_rows(out_dir / "v22_09_packet_required_compare.csv", compare)
    write_rows(out_dir / "v22_09_clean_unzip_self_test.csv", clean_rows)
    missing_count = sum(int_flag(r.get("csv_claimed_exists_but_zip_missing")) for r in compare)
    rows.append({"check": "csv_claimed_exists_but_zip_missing_count", "pass": int(missing_count == 0), "metric": "zip_required_compare", "value": missing_count, "blocker": "" if missing_count == 0 else "required_csv_zip_mismatch"})
    rows.append({"check": "self_contained_compileall", "pass": int(clean_compile_code == 0), "metric": "clean_unzip_compile", "value": clean_compile_code, "blocker": "" if clean_compile_code == 0 else "clean_unzip_compile_failed"})
    rows.append({"check": "self_contained_import_check", "pass": int(clean_import_code == 0 and int(args.self_contained_check) == 1), "metric": "clean_unzip_import", "value": clean_import_code, "blocker": "" if clean_import_code == 0 and int(args.self_contained_check) == 1 else "clean_unzip_import_failed_or_not_requested"})

    write_rows(out_dir / "v22_09_code_truth_gate.csv", rows)
    route = {
        "mode": args.mode,
        "S0_16_pass": int(all(int_flag(r.get("pass")) for r in rows)),
        "required_source_files_exist": required_pass,
        "missing_required_files": ";".join(str(r.get("path")) for r in required if not int_flag(r.get("exists"))),
        "csv_claimed_exists_but_zip_missing_count": missing_count,
        "compileall_pass": int(compile_code == 0 and clean_compile_code == 0),
        "required_import_errors": int(import_code != 0 or clean_import_code != 0),
        "self_contained_import_check": int(args.self_contained_check),
        "forbidden_direction_violation": sum(1 for r in forbidden if not int_flag(r.get("forbidden_direction_pass"))),
        "undeclared_semantic_alias_pairs": alias_undeclared,
        "clean_unzip_root": str(unzip_root),
        "CodeRoute": "R0-CodePacketNotSelfContained" if not int(clean_rows[0].get("pass", 0)) else "R0-CodePacketSelfContained",
    }
    write_json(out_dir / "v22_09_code_route_decision.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_s016_truth_gate.py --mode {args.mode} --source-root {source_root} --self-contained-check {int(args.self_contained_check)} --out-dir {out_dir}",
        status="completed",
        note=f"S0.16={route['S0_16_pass']} code_route={route['CodeRoute']} missing_zip={missing_count} clean_import={clean_import_code}",
    )


if __name__ == "__main__":
    main()
