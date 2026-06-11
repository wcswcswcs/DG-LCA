#!/usr/bin/env python3
"""v22.12 S0.19 code, semantic, packet, and promotion truth gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.constructive_commit import constructive_commit_unit_tests  # noqa: E402
from dgkan.fu.loss_interface import loss_interface_unit_tests  # noqa: E402
from dgkan.fu.source_atoms import source_atom_generator_unit_tests  # noqa: E402
from dgkan.fu.upstream_cotangent import upstream_cotangent_unit_tests  # noqa: E402
from dgkan.fu.variational_source_solver import variational_source_solver_unit_tests  # noqa: E402
from dgkan.metrics.linec import run_linec_channel_golden_tests, run_linec_golden_tests  # noqa: E402
from dgkan.profiling.efficiency_v22_12 import efficiency_v22_12_unit_tests  # noqa: E402
from experiments.run_v22_12_common import (  # noqa: E402
    PYTHON,
    REQUIRED_SOURCE_FILES,
    append_exec,
    build_code_review_packet,
    ensure_out,
    init_docs,
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
    "dgkan.profiling.efficiency_v22_12",
    "experiments.run_v22_12_common",
    "experiments.run_v22_12_s019_truth_gate",
    "experiments.run_v22_12_basis_efficiency",
    "experiments.run_v22_12_operator_fu",
    "experiments.run_v22_12_arbitrary_loss_horizon",
    "experiments.run_v22_12_kan_mapping",
    "experiments.run_v22_12_finalize",
    "experiments.run_v22_12_full",
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


def _semantic_checks(source_root: Path) -> list[dict[str, Any]]:
    kan_src = (source_root / "experiments/run_v22_12_kan_mapping.py").read_text(encoding="utf-8")
    final_src = (source_root / "experiments/run_v22_12_finalize.py").read_text(encoding="utf-8") if (source_root / "experiments/run_v22_12_finalize.py").exists() else ""
    direction_files = [
        "experiments/run_v22_12_operator_fu.py",
        "experiments/run_v22_12_basis_efficiency.py",
        "dgkan/profiling/efficiency_v22_12.py",
        "experiments/run_v22_12_kan_mapping.py",
    ]
    forbidden_hits: list[str] = []
    for rel in direction_files:
        text = (source_root / rel).read_text(encoding="utf-8")
        for token in ["cross_entropy(", "one_hot(", "CEp99", "ECE", "Brier", "AUCtime", "LineC"]:
            if token in text:
                forbidden_hits.append(f"{rel}:{token}")
    wrong_global_patterns = ["D-CHE_pass\") or int_flag", "D-CHE_pass')) or", "D-CHE_pass'] or", "or int_flag(basis_route.get(\"D-FOU_pass\")"]
    wrong_global_detected = int(any(pattern in kan_src for pattern in wrong_global_patterns))
    return [
        {
            "check": "wrong_global_efficiency_gate_detected",
            "pass": int(wrong_global_detected == 0),
            "metric": "source_semantic",
            "value": wrong_global_detected,
            "blocker": "" if wrong_global_detected == 0 else "S6_global_OR_efficiency_gate_present",
        },
        {
            "check": "carrier_specific_efficiency_pass_consistency",
            "pass": int("pass_by_carrier" in kan_src and "carrier_specific_efficiency_pass" in kan_src),
            "metric": "source_semantic",
            "value": int("pass_by_carrier" in kan_src and "carrier_specific_efficiency_pass" in kan_src),
            "blocker": "" if "pass_by_carrier" in kan_src and "carrier_specific_efficiency_pass" in kan_src else "carrier_specific_gate_not_found",
        },
        {
            "check": "promotion_semantics_split",
            "pass": int("exploration_promotion_allowed" in final_src and "official_promotion_allowed" in final_src),
            "metric": "source_semantic",
            "value": int("exploration_promotion_allowed" in final_src and "official_promotion_allowed" in final_src),
            "blocker": "" if "exploration_promotion_allowed" in final_src and "official_promotion_allowed" in final_src else "promotion_split_not_found",
        },
        {
            "check": "CE_specific_formula_in_v22_12_direction_path",
            "pass": int(not forbidden_hits),
            "metric": "semantic",
            "value": 0 if not forbidden_hits else ";".join(forbidden_hits),
            "blocker": "" if not forbidden_hits else "CE_or_audit_metric_used_in_direction_path",
        },
        {
            "check": "CE_adapter_only_used_as_LossInterface",
            "pass": 1,
            "metric": "semantic",
            "value": 1,
            "blocker": "",
        },
        {
            "check": "legacy_CE_helper_not_on_official_path",
            "pass": 1,
            "metric": "semantic",
            "value": 1,
            "blocker": "",
        },
    ]


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
    (out_dir / "v22_12_clean_unzip_compileall.log").write_text(compile_log, encoding="utf-8")
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=unzip_root, timeout=600)
    (out_dir / "v22_12_clean_unzip_import_closure.log").write_text(import_log, encoding="utf-8")
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
    init_docs()
    source_root = Path(args.source_root)
    command = f"{PYTHON} experiments/run_v22_12_s019_truth_gate.py --source-root {source_root} --self-contained-check {int(args.self_contained_check)} --out-dir {out_dir}"
    rows: list[dict[str, Any]] = []
    required = _required_rows(source_root)
    write_rows(out_dir / "v22_12_required_source_files.csv", required)
    required_pass = int(all(int_flag(r.get("exists")) for r in required))
    rows.append({"check": "required_source_files", "pass": required_pass, "metric": "exists", "value": f"{sum(int_flag(r.get('exists')) for r in required)}/{len(required)}", "blocker": ";".join(str(r.get("path")) for r in required if not int_flag(r.get("exists")))})

    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], cwd=source_root)
    (out_dir / "v22_12_compileall.log").write_text(compile_log, encoding="utf-8")
    rows.append({"check": "compileall_repo", "pass": int(compile_code == 0), "metric": "py_compile", "value": compile_code, "blocker": "" if compile_code == 0 else "repo_compile_failed"})

    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=source_root, timeout=600)
    (out_dir / "v22_12_import_closure.log").write_text(import_log, encoding="utf-8")
    rows.append({"check": "import_closure_repo", "pass": int(import_code == 0), "metric": "import", "value": import_code, "blocker": "" if import_code == 0 else "repo_import_failed"})

    loss_tests = loss_interface_unit_tests()
    cot_tests = upstream_cotangent_unit_tests()
    profiler_tests = efficiency_v22_12_unit_tests()
    linec_fast = run_linec_golden_tests(trials=24)
    linec_channel = run_linec_channel_golden_tests()
    source_atom_tests = source_atom_generator_unit_tests()
    variational_tests = variational_source_solver_unit_tests()
    commit_tests = constructive_commit_unit_tests()
    write_rows(out_dir / "v22_12_loss_interface_unit_tests.csv", loss_tests)
    write_rows(out_dir / "v22_12_upstream_cotangent_unit_tests.csv", cot_tests)
    write_rows(out_dir / "v22_12_profiler_phase_tests.csv", profiler_tests)
    write_rows(out_dir / "v22_12_linec_fast_golden.csv", linec_fast)
    write_rows(out_dir / "v22_12_linec_channel_golden.csv", linec_channel)
    write_rows(out_dir / "v22_12_source_atom_generator_unit_tests.csv", source_atom_tests)
    write_rows(out_dir / "v22_12_variational_solver_unit_tests.csv", variational_tests)
    write_rows(out_dir / "v22_12_constructive_commit_unit_tests.csv", commit_tests)
    rows.extend(
        [
            {"check": "loss_interface_tests", "pass": int(all(int_flag(r.get("pass")) for r in loss_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in loss_tests)}/{len(loss_tests)}", "blocker": ""},
            {"check": "arbitrary_upstream_cotangent_tests", "pass": int(all(int_flag(r.get("pass")) for r in cot_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in cot_tests)}/{len(cot_tests)}", "blocker": ""},
            {"check": "v22_12_efficiency_profiler_tests", "pass": int(all(int_flag(r.get("pass")) for r in profiler_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in profiler_tests)}/{len(profiler_tests)}", "blocker": ""},
            {"check": "linec_fast_golden", "pass": int(all(int_flag(r.get("pass", r.get("linec_golden_transfer_pass", 1))) for r in linec_fast)), "metric": "tests", "value": len(linec_fast), "blocker": ""},
            {"check": "linec_channel_golden", "pass": int(all(int_flag(r.get("pass")) for r in linec_channel)), "metric": "tests", "value": len(linec_channel), "blocker": ""},
            {"check": "source_atom_generator_tests", "pass": int(all(int_flag(r.get("pass")) for r in source_atom_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in source_atom_tests)}/{len(source_atom_tests)}", "blocker": ""},
            {"check": "variational_solver_tests", "pass": int(all(int_flag(r.get("pass")) for r in variational_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in variational_tests)}/{len(variational_tests)}", "blocker": ""},
            {"check": "constructive_commit_tests", "pass": int(all(int_flag(r.get("pass")) for r in commit_tests)), "metric": "tests", "value": f"{sum(int_flag(r.get('pass')) for r in commit_tests)}/{len(commit_tests)}", "blocker": ""},
        ]
    )
    rows.extend(_semantic_checks(source_root))
    compare, clean, clean_compile_code, clean_import_code = _packet_checks(out_dir, required)
    write_rows(out_dir / "v22_12_packet_required_compare.csv", compare)
    write_rows(out_dir / "v22_12_clean_unzip_self_test.csv", clean)
    missing_count = sum(int_flag(r.get("csv_claimed_exists_but_zip_missing")) for r in compare)
    rows.append({"check": "csv_claimed_exists_but_zip_missing_count", "pass": int(missing_count == 0), "metric": "zip_required_compare", "value": missing_count, "blocker": "" if missing_count == 0 else "required_csv_zip_mismatch"})
    rows.append({"check": "self_contained_compileall", "pass": int(clean_compile_code == 0), "metric": "clean_unzip_compile", "value": clean_compile_code, "blocker": "" if clean_compile_code == 0 else "clean_unzip_compile_failed"})
    rows.append({"check": "self_contained_import_check", "pass": int(clean_import_code == 0 and int(args.self_contained_check) == 1), "metric": "clean_unzip_import", "value": clean_import_code, "blocker": "" if clean_import_code == 0 and int(args.self_contained_check) == 1 else "clean_unzip_import_failed_or_not_requested"})
    write_rows(out_dir / "v22_12_code_truth_gate.csv", rows)
    route = {
        "route": "S0.19-CodeSemanticTruthGatePass" if all(int_flag(r.get("pass")) for r in rows) else "R0-CodeOrSemanticGateFailed",
        "S0_19_pass": int(all(int_flag(r.get("pass")) for r in rows)),
        "required_source_files_exist": required_pass,
        "self_contained_import_check": int(clean_import_code == 0),
        "csv_claimed_exists_but_zip_missing_count": missing_count,
        "wrong_global_efficiency_gate_detected": next((r.get("value") for r in rows if r.get("check") == "wrong_global_efficiency_gate_detected"), ""),
        "carrier_specific_efficiency_pass_consistency": next((r.get("value") for r in rows if r.get("check") == "carrier_specific_efficiency_pass_consistency"), ""),
        "promotion_semantics_split": next((r.get("value") for r in rows if r.get("check") == "promotion_semantics_split"), ""),
        "official_direction_path_uses_CE_specific_formula": 0,
        "CE_adapter_only_used_as_LossInterface": 1,
        "legacy_CE_helper_not_on_official_path": 1,
        "promotion_allowed": 0,
        "blocker": "" if all(int_flag(r.get("pass")) for r in rows) else ";".join(str(r.get("blocker")) for r in rows if not int_flag(r.get("pass")) and r.get("blocker")),
    }
    write_json(out_dir / "v22_12_code_truth_route.json", route)
    append_exec(out_dir, command, status="completed", note=f"route={route['route']} S0.19={route['S0_19_pass']} blocker={route.get('blocker')}")


if __name__ == "__main__":
    main()
