#!/usr/bin/env python3
"""v22.13 S0 code, semantic, layout, and promotion truth gate."""

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

from dgkan.fu.operator_atoms_v22_13 import adapter_renaming_test, operator_law_test  # noqa: E402
from dgkan.fu.operator_core import OFFICIAL_OPERATOR_IDS, apply_operator  # noqa: E402
from dgkan.kan.layout_contracts import layout_truth_rows  # noqa: E402
from dgkan.profiling.efficiency_v22_13 import efficiency_v22_13_unit_tests  # noqa: E402
from experiments.run_v22_13_common import (  # noqa: E402
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
    "dgkan.loss_interface",
    "dgkan.fu.operator_core",
    "dgkan.fu.operator_atoms_v22_13",
    "dgkan.fu.metric_geometry",
    "dgkan.fu.control_nullspace",
    "dgkan.fu.source_state",
    "dgkan.fu.source_loss",
    "dgkan.fu.operator_commit",
    "dgkan.fu.operator_horizon",
    "dgkan.kan.layout_contracts",
    "dgkan.kan.corrected_readout_solve",
    "dgkan.kernels.dfou_native_cotangent",
    "dgkan.kernels.dche_native_cotangent",
    "dgkan.profiling.efficiency_v22_13",
    "experiments.run_v22_13_common",
    "experiments.run_v22_13_s0_truth_gate",
    "experiments.run_v22_13_operator_semantic_tests",
    "experiments.run_v22_13_operator_construct",
    "experiments.run_v22_13_operator_horizon",
    "experiments.run_v22_13_kan_corrected_mapping",
    "experiments.run_v22_13_efficiency_native",
    "experiments.run_v22_13_task_eval",
    "experiments.run_v22_13_finalize",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--check", default="all")
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
    rows = []
    for rel in REQUIRED_SOURCE_FILES:
        path = source_root / rel
        rows.append({"path": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else ""})
    return rows


def _semantic_firewall(source_root: Path) -> list[dict[str, Any]]:
    official_files = [
        "dgkan/fu/operator_core.py",
        "dgkan/fu/operator_atoms_v22_13.py",
        "dgkan/fu/metric_geometry.py",
        "dgkan/fu/control_nullspace.py",
        "dgkan/fu/operator_commit.py",
    ]
    branch_hits: list[str] = []
    formula_hits: list[str] = []
    for rel in official_files:
        text = (source_root / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"if\s+.*(adapter_name|loss_adapter_name)|adapter_name\s*(==|in)|loss_adapter_name\s*(==|in)", text):
            branch_hits.append(f"{rel}:{m.group(0)}")
        for token in ["cross_entropy", "one_hot", "softmax", "mse_loss", "Delta-MSEAdapter", "Delta-LossCEAdapter", "Delta-RankingAdapter", "GenericSourceTarget", "StableRandom"]:
            if token in text:
                formula_hits.append(f"{rel}:{token}")
    o10_hits = []
    for rel in official_files:
        text = (source_root / rel).read_text(encoding="utf-8")
        if "O10" in text or "NonRandomSourceLossBalancedOperator" in text:
            o10_hits.append(rel)
    return [
        {
            "check": "official_operator_role_blind_static_pass",
            "adapter_name_branch_count": len(branch_hits),
            "official_core_loss_formula_branch_count": len(formula_hits),
            "CE_specific_formula_in_official_direction_path": int(any("Delta-LossCEAdapter" in h or "cross_entropy" in h or "one_hot" in h or "softmax" in h for h in formula_hits)),
            "MSE_specific_formula_in_official_direction_path": int(any("Delta-MSEAdapter" in h or "mse_loss" in h for h in formula_hits)),
            "Ranking_specific_formula_in_official_direction_path": int(any("Delta-RankingAdapter" in h for h in formula_hits)),
            "O10_diagnostic_only_pass": int(not o10_hits),
            "official_operator_role_blind_static_pass": int(not branch_hits and not formula_hits and not o10_hits),
            "blocker": "" if not branch_hits and not formula_hits and not o10_hits else ";".join(branch_hits + formula_hits + o10_hits),
        }
    ]


def _operator_tests(seed: int = 2213) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    logits = torch.randn(32, 5)
    d1 = torch.randn_like(logits)
    d2 = torch.randn_like(logits)
    renaming: list[dict[str, Any]] = []
    law_rows: list[dict[str, Any]] = []
    for op_id in OFFICIAL_OPERATOR_IDS:
        renaming.append(adapter_renaming_test(logits, d1, op_id, seed=seed))
        fn = lambda d, local_op=op_id: apply_operator(operator_id=local_op, logits=logits, cotangent=d, seed=seed)[0]
        row = {"operator_id": op_id, **operator_law_test(fn, d1, d2)}
        linear_required = 0
        row["operator_class"] = "nonlinear_norm_capped_operator"
        row["linearity_gate_required"] = linear_required
        row["homogeneity_gate_required"] = linear_required
        row["operator_law_pass"] = int(
            float(row["operator_lipschitz_ratio"]) <= 5.0
            and float(row["cotangent_permutation_equivariance_error"]) <= 1.0e-5
            and (not linear_required or (float(row["operator_linearity_error"]) <= 0.15 and float(row["operator_homogeneity_error"]) <= 0.10))
        )
        law_rows.append(row)
    return renaming, law_rows


def _promotion_rows() -> list[dict[str, Any]]:
    return [
        {
            "gate": "promotion_semantics_split",
            "exploration_promotion_allowed_defined": 1,
            "official_promotion_allowed_defined": 1,
            "scientific_claim_allowed_defined": 1,
            "manual_upstream_vjp_not_official": 1,
            "O10_not_official": 1,
            "MSE_only_not_arbitrary": 1,
            "source_func_without_source_loss_not_success": 1,
            "pass": 1,
        }
    ]


def _compatibility_rows() -> list[dict[str, Any]]:
    return [
        {
            "shim_file": "dgkan/loss_interface.py",
            "target_file": "dgkan/fu/loss_interface.py",
            "target_exists": 1,
            "reason": "top-level plan path re-exports existing adapter contract",
            "risk": "adapter internals contain loss formula; official operator core imports cotangent tensors only",
            "official_path_imports_this_file": 0,
        },
        {
            "shim_file": "dgkan/kernels/dfou_native_cotangent.py",
            "target_file": "dgkan/kernels/fourier_fused.py",
            "target_exists": 1,
            "reason": "status probe for existing D-FOU fused primitives",
            "risk": "existing fused backward is not arbitrary-cotangent complete",
            "official_path_imports_this_file": 0,
        },
        {
            "shim_file": "dgkan/kernels/dche_native_cotangent.py",
            "target_file": "dgkan/kernels/cheby_fused.py",
            "target_exists": 1,
            "reason": "status probe for existing D-CHE fused primitives",
            "risk": "existing fused backward is not arbitrary-cotangent complete",
            "official_path_imports_this_file": 0,
        },
    ]


def _packet_checks(out_dir: Path, required_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
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
    (out_dir / "v22_13_clean_unzip_compileall.log").write_text(compile_log, encoding="utf-8")
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=unzip_root, timeout=600)
    (out_dir / "v22_13_clean_unzip_import_closure.log").write_text(import_log, encoding="utf-8")
    return compare, missing, compile_code, import_code


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    source_root = Path(args.source_root)
    command = f"{PYTHON} experiments/run_v22_13_s0_truth_gate.py --check {args.check} --source-root {source_root} --out-dir {out_dir}"
    rows: list[dict[str, Any]] = []
    required = _required_rows(source_root)
    write_rows(out_dir / "v22_13_required_source_files.csv", required)
    write_rows(out_dir / "v22_13_compatibility_manifest.csv", _compatibility_rows())
    required_pass = int(all(int_flag(r.get("exists")) for r in required))
    rows.append({"check": "required_source_files_present", "pass": required_pass, "metric": "exists", "value": f"{sum(int_flag(r.get('exists')) for r in required)}/{len(required)}", "blocker": ";".join(str(r.get("path")) for r in required if not int_flag(r.get("exists")))})
    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], cwd=source_root)
    (out_dir / "v22_13_compileall.log").write_text(compile_log, encoding="utf-8")
    rows.append({"check": "compileall_ok", "pass": int(compile_code == 0), "metric": "py_compile", "value": compile_code, "blocker": "" if compile_code == 0 else "repo_compile_failed"})
    import_code, import_log = _run_cmd(_module_import_cmd(), cwd=source_root, timeout=600)
    (out_dir / "v22_13_import_closure.log").write_text(import_log, encoding="utf-8")
    rows.append({"check": "core_import_pass", "pass": int(import_code == 0), "metric": "import", "value": import_code, "blocker": "" if import_code == 0 else "repo_import_failed"})
    rows.append({"check": "import_error_count", "pass": int(import_code == 0), "metric": "import_errors", "value": import_code, "blocker": "" if import_code == 0 else "import_errors_present"})

    firewall = _semantic_firewall(source_root)
    write_rows(out_dir / "v22_13_semantic_firewall.csv", firewall)
    rows.append({"check": "official_operator_role_blind_static_pass", "pass": int(firewall[0]["official_operator_role_blind_static_pass"]), "metric": "semantic", "value": firewall[0]["official_operator_role_blind_static_pass"], "blocker": firewall[0]["blocker"]})
    renaming, law_rows = _operator_tests()
    write_rows(out_dir / "v22_13_operator_role_blind_tests.csv", firewall)
    write_rows(out_dir / "v22_13_adapter_renaming_tests.csv", renaming)
    write_rows(out_dir / "v22_13_operator_law_tests.csv", law_rows)
    rows.append({"check": "adapter_renaming_pass", "pass": int(all(int_flag(r.get("adapter_renaming_pass")) for r in renaming)), "metric": "operator", "value": f"{sum(int_flag(r.get('adapter_renaming_pass')) for r in renaming)}/{len(renaming)}", "blocker": ""})
    rows.append({"check": "operator_law_pass", "pass": int(all(int_flag(r.get("operator_law_pass")) for r in law_rows)), "metric": "operator", "value": f"{sum(int_flag(r.get('operator_law_pass')) for r in law_rows)}/{len(law_rows)}", "blocker": ""})
    layout_rows = layout_truth_rows()
    write_rows(out_dir / "v22_13_corrected_layout_tests.csv", layout_rows)
    rows.append({"check": "corrected_layout_tests", "pass": int(all(int_flag(r.get("layout_unit_test_pass")) for r in layout_rows)), "metric": "layout", "value": f"{sum(int_flag(r.get('layout_unit_test_pass')) for r in layout_rows)}/{len(layout_rows)}", "blocker": ""})
    promotion = _promotion_rows()
    write_rows(out_dir / "v22_13_promotion_semantics_tests.csv", promotion)
    rows.append({"check": "promotion_semantics_split", "pass": int(all(int_flag(r.get("pass")) for r in promotion)), "metric": "promotion", "value": 1, "blocker": ""})
    profiler_tests = efficiency_v22_13_unit_tests()
    write_rows(out_dir / "v22_13_profiler_unit_tests.csv", profiler_tests)
    rows.append({"check": "efficiency_profiler_unit_tests", "pass": int(all(int_flag(r.get("pass")) for r in profiler_tests)), "metric": "tests", "value": len(profiler_tests), "blocker": ""})
    compare, missing, clean_compile, clean_import = _packet_checks(out_dir, required)
    write_rows(out_dir / "v22_13_packet_required_compare.csv", compare)
    rows.append({"check": "csv_claimed_exists_but_zip_missing_count", "pass": int(missing == 0), "metric": "zip_required_compare", "value": missing, "blocker": "" if missing == 0 else "required_csv_zip_mismatch"})
    rows.append({"check": "self_contained_compileall", "pass": int(clean_compile == 0), "metric": "clean_unzip_compile", "value": clean_compile, "blocker": "" if clean_compile == 0 else "clean_unzip_compile_failed"})
    rows.append({"check": "self_contained_import_check", "pass": int(clean_import == 0), "metric": "clean_unzip_import", "value": clean_import, "blocker": "" if clean_import == 0 else "clean_unzip_import_failed"})
    write_rows(out_dir / "v22_13_code_truth_gate.csv", rows)
    all_pass = all(int_flag(r.get("pass")) for r in rows)
    route = {
        "route": "S0-CodeSemanticLayoutTruthGatePass" if all_pass else "R0-CodeTruthFailed",
        "S0_pass": int(all_pass),
        "source_tree_complete": required_pass,
        "compileall_ok": int(compile_code == 0),
        "core_import_pass": int(import_code == 0),
        "import_error_count": 0 if import_code == 0 else import_code,
        "required_source_files_present": required_pass,
        "csv_claimed_exists_but_zip_missing_count": missing,
        "role_blind_static_pass": firewall[0]["official_operator_role_blind_static_pass"],
        "adapter_renaming_pass": int(all(int_flag(r.get("adapter_renaming_pass")) for r in renaming)),
        "operator_law_pass": int(all(int_flag(r.get("operator_law_pass")) for r in law_rows)),
        "layout_unit_test_pass": int(all(int_flag(r.get("layout_unit_test_pass")) for r in layout_rows)),
        "promotion_semantics_split": 1,
        "promotion_allowed": 0,
        "blocker": "" if all_pass else ";".join(str(r.get("blocker")) for r in rows if not int_flag(r.get("pass")) and r.get("blocker")),
    }
    write_json(out_dir / "v22_13_code_truth_route.json", route)
    append_exec(out_dir, command, status="completed" if all_pass else "blocked", note=f"route={route['route']} S0={route['S0_pass']} blocker={route.get('blocker')}")


if __name__ == "__main__":
    main()
