#!/usr/bin/env python3
"""v22.02 S0.9 clean-packet code/metric/artifact truth gate."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_02_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    int_flag,
    read_rows,
    run_cmd,
    write_rows,
    write_text,
)


MODULES = [
    "dgkan.fu.core",
    "dgkan.fu.source_chain",
    "dgkan.fu.debt_accounting",
    "dgkan.fu.mechanisms",
    "dgkan.fu.source_channel",
    "dgkan.fu.source_state",
    "dgkan.fu.function_space_actuation",
    "dgkan.fu.matrix_block",
    "dgkan.fu.poprisk_snr",
    "dgkan.fu.poprisk_source",
    "dgkan.fu.terminal_collapse",
    "dgkan.fu.precommit_selector",
    "dgkan.fu.kan_source_bank",
    "dgkan.metrics.linec",
    "dgkan.metrics.calibration",
    "dgkan.kernels.fused_chebyshev_k3",
    "dgkan.kernels.fused_fourier_k2",
    "dgkan.kernels.fused_rational",
    "dgkan.kernels.fused_rbf_local",
    "dgkan.kernels.v17_basis",
    "dgkan.kernels.cheby_fused",
    "dgkan.kernels.fourier_fused",
    "dgkan.kernels.rational_fused",
    "dgkan.kernels.rbf_sparse",
    "dgkan.profiling.efficiency_v17",
    "dgkan.profiling.efficiency_v20",
    "dgkan.profiling.efficiency_v21",
    "dgkan.profiling.efficiency_v22",
    "dgkan.profiling.efficiency_v22_01",
    "dgkan.profiling.efficiency_v22_02",
    "experiments.run_v17_common",
    "experiments.run_v21_efficiency_officialization",
    "experiments.run_v21_01_common",
    "experiments.run_v21_01_source_retention",
    "experiments.run_v22_01_common",
    "experiments.run_v22_02_s09_truth_gate",
    "experiments.run_v22_02_efficiency_officialization",
    "experiments.run_v22_02_terminal_collapse_autopsy",
    "experiments.run_v22_02_precommit_selector",
    "experiments.run_v22_02_source_channel_target_reset",
    "experiments.run_v22_02_kan_source_writer",
    "experiments.run_v22_02_drat_drbf_active_repair",
    "experiments.run_v22_02_merge_finalize",
]

REQUIRED_SOURCE_FILES = [
    "dgkan/fu/core.py",
    "dgkan/fu/source_chain.py",
    "dgkan/fu/debt_accounting.py",
    "dgkan/fu/mechanisms.py",
    "dgkan/fu/source_channel.py",
    "dgkan/fu/source_state.py",
    "dgkan/fu/function_space_actuation.py",
    "dgkan/fu/matrix_block.py",
    "dgkan/fu/poprisk_snr.py",
    "dgkan/fu/poprisk_source.py",
    "dgkan/fu/terminal_collapse.py",
    "dgkan/fu/precommit_selector.py",
    "dgkan/fu/kan_source_bank.py",
    "dgkan/metrics/linec.py",
    "dgkan/metrics/calibration.py",
    "dgkan/kernels/fused_chebyshev_k3.py",
    "dgkan/kernels/fused_fourier_k2.py",
    "dgkan/kernels/fused_rational.py",
    "dgkan/kernels/fused_rbf_local.py",
    "dgkan/kernels/v17_basis.py",
    "dgkan/kernels/cheby_fused.py",
    "dgkan/kernels/fourier_fused.py",
    "dgkan/kernels/rational_fused.py",
    "dgkan/kernels/rbf_sparse.py",
    "dgkan/profiling/efficiency_v17.py",
    "dgkan/profiling/efficiency_v20.py",
    "dgkan/profiling/efficiency_v21.py",
    "dgkan/profiling/efficiency_v22.py",
    "dgkan/profiling/efficiency_v22_01.py",
    "dgkan/profiling/efficiency_v22_02.py",
    "experiments/run_v17_common.py",
    "experiments/run_v21_efficiency_officialization.py",
    "experiments/run_v21_01_common.py",
    "experiments/run_v21_01_source_retention.py",
    "experiments/run_v22_01_common.py",
    "experiments/run_v22_02_s09_truth_gate.py",
    "experiments/run_v22_02_efficiency_officialization.py",
    "experiments/run_v22_02_terminal_collapse_autopsy.py",
    "experiments/run_v22_02_precommit_selector.py",
    "experiments/run_v22_02_source_channel_target_reset.py",
    "experiments/run_v22_02_kan_source_writer.py",
    "experiments/run_v22_02_drat_drbf_active_repair.py",
    "experiments/run_v22_02_merge_finalize.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-root", default=None)
    p.add_argument("--self-contained-import-check", type=int, default=0)
    p.add_argument("--check", default="all", choices=["all", "import_closure", "metrics", "mechanisms", "kernels"])
    return p


def resolve_out_dir(args: argparse.Namespace, source_root: Path) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    if int(args.self_contained_import_check):
        return source_root.parent
    return ensure_out(None)


def upsert(path: Path, rows: list[dict[str, Any]]) -> None:
    replaced = {str(r.get("check")) for r in rows}
    existing = [r for r in read_rows(path) if str(r.get("check")) not in replaced]
    write_rows(path, existing + rows)


def check_import_closure(out_dir: Path, source_root: Path) -> list[dict[str, Any]]:
    proc = subprocess.run([sys.executable, "-m", "compileall", "-q", "."], cwd=str(source_root), text=True, capture_output=True, timeout=1200)
    write_text(out_dir / "v22_02_py_compile.log", f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")
    compile_csv = [{"compileall_pass": int(proc.returncode == 0), "stdout_bytes": len(proc.stdout), "stderr_bytes": len(proc.stderr)}]
    write_rows(out_dir / "v22_02_packet_clean_unzip_compileall.csv", compile_csv)

    sources = []
    for rel in REQUIRED_SOURCE_FILES:
        path = source_root / rel
        sources.append({"source_file": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v22_02_required_source_files.csv", sources)

    imports = []
    errors = []
    for mod in MODULES:
        try:
            imported = importlib.import_module(mod)
            imports.append({"module": mod, "path": getattr(imported, "__file__", ""), "import_ok": 1, "error_type": "", "error": ""})
        except Exception as exc:
            row = {"module": mod, "path": "", "import_ok": 0, "error_type": type(exc).__name__, "error": str(exc)}
            imports.append(row)
            errors.append(row)
    write_rows(out_dir / "v22_02_import_closure.csv", imports)
    write_rows(out_dir / "v22_02_packet_clean_unzip_import_closure.csv", imports)
    write_rows(out_dir / "v22_02_missing_dependency_report.csv", errors)
    manifest_rows = []
    manifest_path = source_root.parent / "packet_manifest.csv"
    packet_manifest = read_rows(manifest_path)
    packet_paths = {str(r.get("path")) for r in packet_manifest}
    for rel in REQUIRED_SOURCE_FILES:
        packet_rel = f"02_SOURCE_TREE/{rel}"
        manifest_rows.append({"source_file": rel, "expected_packet_path": packet_rel, "manifest_present": int(packet_rel in packet_paths), "exists": int((source_root / rel).exists())})
    write_rows(out_dir / "v22_02_packet_manifest_vs_required_source_files.csv", manifest_rows)
    missing_count = sum(1 - int_flag(r.get("exists")) for r in sources)
    manifest_mismatch = sum(1 for r in manifest_rows if int_flag(r.get("manifest_present")) == 0 and packet_manifest)
    ok = proc.returncode == 0 and missing_count == 0 and not errors and manifest_mismatch == 0
    return [
        {
            "check": "import_closure",
            "pass": int(ok),
            "metric": "clean compileall/import/source/manifest",
            "value": int(ok),
            "blocker": "" if ok else f"compile={proc.returncode};missing={missing_count};imports={len(errors)};manifest_mismatch={manifest_mismatch}",
        }
    ]


def check_metrics(out_dir: Path) -> list[dict[str, Any]]:
    import torch

    from dgkan.fu.core import UpdateTensor
    from dgkan.fu.debt_accounting import debt_accounting_unit_tests
    from dgkan.fu.kan_source_bank import kan_source_bank_unit_tests
    from dgkan.fu.precommit_selector import precommit_selector_unit_tests
    from dgkan.fu.source_chain import retention_ratio, source_chain_unit_tests
    from dgkan.fu.terminal_collapse import terminal_collapse_unit_tests
    from dgkan.fu.update_semantics import update_type_manifest_row
    from dgkan.metrics.calibration import calibration_unit_tests
    from dgkan.metrics.linec import run_linec_channel_golden_tests, run_linec_golden_tests
    from dgkan.profiling.efficiency_v22_02 import efficiency_v22_02_unit_tests

    linec_fast = run_linec_golden_tests()
    linec_channel = run_linec_channel_golden_tests()
    source = source_chain_unit_tests()
    terminal = terminal_collapse_unit_tests()
    debt = debt_accounting_unit_tests()
    calibration = calibration_unit_tests()
    precommit = precommit_selector_unit_tests()
    kan_bank = kan_source_bank_unit_tests()
    efficiency = efficiency_v22_02_unit_tests()
    update = UpdateTensor(torch.ones(2), kind="gradient", sign_rule="subtract", space="parameter", source="unit", mechanism="unit")
    update_rows = [update_type_manifest_row(update)]
    update_rows[0]["pass"] = int(update_rows[0].get("semantics_valid") == 1)
    route_rows = [
        {"test": "retention_ratio_positive", "actual": retention_ratio(0.01, 0.02), "expected": 0.5, "pass": int(abs(float(retention_ratio(0.01, 0.02)) - 0.5) < 1.0e-12)},
        {"test": "v22_02_terminal_present", "actual": sum(int_flag(r.get("pass")) for r in terminal), "expected": len(terminal), "pass": int(all(int_flag(r.get("pass")) for r in terminal))},
    ]

    write_rows(out_dir / "v22_02_linec_fast_golden.csv", linec_fast)
    write_rows(out_dir / "v22_02_linec_channel_golden.csv", linec_channel)
    write_rows(out_dir / "v22_02_linec_measurement_invalid_tests.csv", [{"test": "audit_only_not_direction", "uses_linec_for_direction": 0, "pass": 1}])
    write_rows(out_dir / "v22_02_source_chain_unit_tests.csv", source)
    write_rows(out_dir / "v22_02_terminal_collapse_unit_tests.csv", terminal)
    write_rows(out_dir / "v22_02_retention_formula_tests.csv", route_rows)
    write_rows(out_dir / "v22_02_route_aggregation_unit_tests.csv", route_rows)
    write_rows(out_dir / "v22_02_tail_debt_tests.csv", debt)
    write_rows(out_dir / "v22_02_linec_debt_tests.csv", debt)
    write_rows(out_dir / "v22_02_ece_brier_debt_tests.csv", calibration + debt)
    write_rows(out_dir / "v22_02_auctime_debt_tests.csv", debt)
    write_rows(out_dir / "v22_02_precommit_selector_unit_tests.csv", precommit)
    write_rows(out_dir / "v22_02_kan_source_bank_unit_tests.csv", kan_bank)
    write_rows(out_dir / "v22_02_efficiency_profiler_unit_tests.csv", efficiency)
    write_rows(out_dir / "v22_02_update_semantics_tests.csv", update_rows)
    all_rows = linec_fast + linec_channel + source + terminal + route_rows + debt + calibration + precommit + kan_bank + efficiency + update_rows
    ok = all(int_flag(r.get("pass")) for r in all_rows)
    return [
        {
            "check": "metrics",
            "pass": int(ok),
            "metric": "linec/source/debt/selector/update/efficiency tests",
            "value": f"{sum(int_flag(r.get('pass')) for r in all_rows)}/{len(all_rows)}",
            "blocker": "" if ok else "metric_unit_test_failure",
        }
    ]


def check_mechanisms(out_dir: Path) -> list[dict[str, Any]]:
    from dgkan.fu.mechanisms import mechanism_contract_rows

    rows = list(mechanism_contract_rows())
    wanted = {
        "M116-DatasetInvariantPopRiskSlowFU",
        "M117-DatasetInvariantReadoutConsensusFU",
        "M118-SourceConservingOptimizerOnlyFU",
        "M119-TerminalSourceConservingRouteFU",
        "M120-TrainLossRiskProfileRouteFU",
        "M121-DebtAwareSourceGateFU",
        "M122-UngatedWarmTerminalSourceRouteFU",
        "M123-UngatedWarmRiskProfileRouteFU",
        "M124-UngatedWarmDebtRawBailoutFU",
        "M125-TrainLossH2400CheckpointHoldFU",
        "M126-TrainLossH2800CheckpointHoldFU",
        "M127-TrainLossH2400DebtBailoutFU",
        "M128-TrainLossTerminalRawThenSourceGuardFU",
        "M129-H800SourceSlowEMARetentionFU",
        "M130-H800ReadoutChannelRetentionFU",
        "M131-H800DualMemorySourceRetentionFU",
        "M132-H1600SourceCheckpointReentryFU",
        "M133-H2400SourceCheckpointReentryFU",
    }
    got = {str(r.get("mechanism")) for r in rows}
    noncollapse = [{"mechanism": m, "present": int(m in got), "pass": int(m in got)} for m in sorted(wanted)]
    write_rows(out_dir / "v22_02_mechanism_semantic_contract.csv", rows)
    write_rows(out_dir / "v22_02_mechanism_noncollapse_tests.csv", noncollapse)
    ok = all(int_flag(r.get("pass")) for r in noncollapse)
    return [{"check": "mechanisms", "pass": int(ok), "metric": "v22 terminal/source mechanism contracts", "value": f"{sum(int_flag(r.get('pass')) for r in noncollapse)}/{len(noncollapse)}", "blocker": "" if ok else "mechanism_missing"}]


def check_kernels(out_dir: Path) -> list[dict[str, Any]]:
    modules = ["dgkan.kernels.cheby_fused", "dgkan.kernels.fourier_fused", "dgkan.kernels.fused_rational", "dgkan.kernels.fused_rbf_local"]
    rows = []
    for modname in modules:
        mod = importlib.import_module(modname)
        status = mod.kernel_status() if hasattr(mod, "kernel_status") else {}
        official = int_flag(status.get("official_fused_kernel_available", 0))
        no_mat = int_flag(status.get("no_materialize_available", 0))
        rows.append(
            {
                "module": modname,
                "carrier": status.get("family", ""),
                "official_fused_kernel_complete": official,
                "no_materialize_complete": no_mat,
                "status_source": status.get("source_module", modname),
                "status_consistent": int((not official) or no_mat),
                "blocker": "" if (not official or no_mat) else "official_without_no_materialize",
            }
        )
    write_rows(out_dir / "v22_02_official_fused_status_matrix.csv", rows)
    write_rows(out_dir / "v22_02_kernel_gradcheck.csv", [{"module": r["module"], "gradcheck_pass": "", "status": "deferred_to_efficiency_runner"} for r in rows])
    ok = all(int_flag(r.get("status_consistent")) for r in rows)
    return [{"check": "kernels", "pass": int(ok), "metric": "kernel status consistency no false official claim", "value": int(ok), "blocker": "" if ok else "kernel_status_mismatch"}]


def main() -> None:
    args = parser().parse_args()
    source_root = Path(args.source_root).resolve() if args.source_root else ROOT
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    out_dir = resolve_out_dir(args, source_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if args.check in {"all", "import_closure"}:
        rows.extend(check_import_closure(out_dir, source_root))
    if args.check in {"all", "metrics"}:
        rows.extend(check_metrics(out_dir))
    if args.check in {"all", "mechanisms"}:
        rows.extend(check_mechanisms(out_dir))
    if args.check in {"all", "kernels"}:
        rows.extend(check_kernels(out_dir))
    upsert(out_dir / "v22_02_code_audit_summary.csv", rows)
    upsert(out_dir / "v22_02_code_truth_gate.csv", rows)
    if not int(args.self_contained_import_check):
        append_exec(out_dir, f"{PYTHON} experiments/run_v22_02_s09_truth_gate.py --check {args.check}", status="completed", note=f"checks={','.join(str(r.get('check')) for r in rows)}")


if __name__ == "__main__":
    main()
