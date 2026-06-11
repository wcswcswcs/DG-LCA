#!/usr/bin/env python3
"""v22.04 S0.11 code truth gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.debt_accounting import debt_accounting_unit_tests  # noqa: E402
from dgkan.fu.diffeomorphic_target import diffeomorphic_target_unit_tests  # noqa: E402
from dgkan.fu.mechanisms import MECHANISMS, mechanism_contract_rows  # noqa: E402
from dgkan.fu.source_chain import source_chain_unit_tests  # noqa: E402
from dgkan.fu.source_preservation import source_preservation_unit_tests  # noqa: E402
from dgkan.fu.terminal_erosion import terminal_erosion_unit_tests  # noqa: E402
from dgkan.fu.terminal_retention import terminal_retention_unit_tests  # noqa: E402
from dgkan.metrics.linec import run_linec_channel_golden_tests, run_linec_golden_tests  # noqa: E402
from dgkan.profiling.efficiency_v22_04 import efficiency_v22_04_unit_tests  # noqa: E402
from dgkan.profiling.kernel_gradcheck import kernel_correctness_row  # noqa: E402
from experiments.run_v22_04_common import PYTHON, append_exec, ensure_out, run_cmd, write_json, write_rows  # noqa: E402


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/core.py",
    "dgkan/fu/source_chain.py",
    "dgkan/fu/terminal_retention.py",
    "dgkan/fu/debt_accounting.py",
    "dgkan/fu/terminal_erosion.py",
    "dgkan/fu/source_preservation.py",
    "dgkan/fu/diffeomorphic_target.py",
    "dgkan/fu/matrix_block.py",
    "dgkan/fu/poprisk_snr.py",
    "dgkan/fu/mechanisms.py",
    "dgkan/metrics/linec.py",
    "dgkan/kernels/fused_chebyshev_k3.py",
    "dgkan/kernels/fused_fourier_k2.py",
    "dgkan/kernels/rational_fused.py",
    "dgkan/kernels/rbf_sparse.py",
    "dgkan/kernels/v17_basis.py",
    "dgkan/profiling/efficiency_v20.py",
    "dgkan/profiling/efficiency_v21.py",
    "dgkan/profiling/efficiency_v22.py",
    "dgkan/profiling/efficiency_v22_03.py",
    "dgkan/profiling/efficiency_v22_04.py",
    "experiments/run_v22_04_common.py",
    "experiments/run_v22_04_s011_truth_gate.py",
    "experiments/run_v22_04_efficiency_officialization.py",
    "experiments/run_v22_04_drat_drbf_officialization.py",
    "experiments/run_v22_04_terminal_erosion_autopsy.py",
    "experiments/run_v22_04_source_preserving_fu.py",
    "experiments/run_v22_04_kan_source_mapping.py",
    "experiments/run_v22_04_merge_finalize.py",
]


EXPECTED_MECHANISMS = [
    "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
    "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
    "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
    "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
    "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
    "M215-EarlySourceSlowEMATerminalAcceptMemoryFU",
    "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
    "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
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


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_root = Path(args.source_root)
    mode = str(args.mode)
    rows: list[dict[str, Any]] = []
    files = [source_root / rel for rel in REQUIRED_SOURCE_FILES]

    if _mode_enabled(mode, "required_source_files"):
        file_rows = [
            {"path": rel, "exists": int((source_root / rel).exists()), "size_bytes": (source_root / rel).stat().st_size if (source_root / rel).exists() else ""}
            for rel in REQUIRED_SOURCE_FILES
        ]
        write_rows(out_dir / "v22_04_required_source_files.csv", file_rows)
        rows.append(
            {
                "check": "required_source_files",
                "pass": int(all((source_root / rel).exists() for rel in REQUIRED_SOURCE_FILES)),
                "metric": "exists",
                "value": f"{sum(int(r['exists']) for r in file_rows)}/{len(file_rows)}",
                "blocker": ";".join(r["path"] for r in file_rows if not r["exists"]),
            }
        )

    if _mode_enabled(mode, "import_closure") or mode == "all":
        compile_cmd = [PYTHON, "-m", "compileall", "-q", *[str(p) for p in files if p.exists()]]
        code, log = run_cmd(compile_cmd, cwd=source_root, timeout=300)
        (out_dir / "v22_04_compileall.log").write_text(log, encoding="utf-8")
        write_rows(out_dir / "v22_04_compileall.csv", [{"command": " ".join(compile_cmd), "returncode": code, "pass": int(code == 0)}])
        import_code, import_log = run_cmd(
            [
                PYTHON,
                "-c",
                "import dgkan.fu.terminal_erosion, dgkan.fu.source_preservation, dgkan.fu.diffeomorphic_target, dgkan.metrics.linec, dgkan.profiling.efficiency_v22_04, experiments.run_v22_04_common, experiments.run_v22_04_source_preserving_fu",
            ],
            cwd=source_root,
            timeout=120,
        )
        (out_dir / "v22_04_import_closure.log").write_text(import_log, encoding="utf-8")
        write_rows(out_dir / "v22_04_import_closure.csv", [{"returncode": import_code, "pass": int(import_code == 0), "self_contained": int(args.self_contained_import_check)}])
        rows.extend(
            [
                {"check": "compileall", "pass": int(code == 0), "metric": "py_compile", "value": code, "blocker": "" if code == 0 else "compile_failed"},
                {"check": "import_closure", "pass": int(import_code == 0), "metric": "import", "value": import_code, "blocker": "" if import_code == 0 else "import_failed"},
            ]
        )

    if _mode_enabled(mode, "linec_golden"):
        fast = run_linec_golden_tests(trials=24)
        channel = run_linec_channel_golden_tests()
        write_rows(out_dir / "v22_04_linec_fast_golden.csv", fast)
        write_rows(out_dir / "v22_04_linec_channel_golden.csv", channel)
        rows.extend(
            [
                {"check": "linec_fast_golden", "pass": int(all(int(r.get("pass", r.get("linec_golden_transfer_pass", 1))) for r in fast)), "metric": "tests", "value": f"{len(fast)}", "blocker": ""},
                {"check": "linec_channel_golden", "pass": int(all(int(r.get("pass", 0)) for r in channel)), "metric": "tests", "value": f"{len(channel)}", "blocker": ""},
            ]
        )

    if _mode_enabled(mode, "source_chain_tests"):
        tests = source_chain_unit_tests()
        write_rows(out_dir / "v22_04_source_chain_unit_tests.csv", tests)
        rows.append({"check": "source_chain_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "terminal_retention_tests"):
        tests = terminal_retention_unit_tests() + terminal_erosion_unit_tests() + source_preservation_unit_tests() + diffeomorphic_target_unit_tests() + debt_accounting_unit_tests()
        write_rows(out_dir / "v22_04_terminal_retention_unit_tests.csv", tests)
        rows.append({"check": "terminal_retention_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "mechanism_contracts"):
        contract = mechanism_contract_rows()
        selected = [r for r in contract if str(r.get("mechanism", "")) in EXPECTED_MECHANISMS]
        write_rows(out_dir / "v22_04_mechanism_contracts.csv", selected)
        rows.append(
            {
                "check": "mechanism_contracts",
                "pass": int(len(selected) == len(EXPECTED_MECHANISMS) and all(m in MECHANISMS for m in EXPECTED_MECHANISMS)),
                "metric": "v22.04 mechanisms",
                "value": f"{len(selected)}/{len(EXPECTED_MECHANISMS)}",
                "blocker": ";".join(m for m in EXPECTED_MECHANISMS if m not in MECHANISMS),
            }
        )

    if _mode_enabled(mode, "kernel_gradcheck"):
        kernel_rows = [kernel_correctness_row("D-CHE"), kernel_correctness_row("D-FOU"), kernel_correctness_row("D-RAT"), kernel_correctness_row("D-RBF")]
        write_rows(out_dir / "v22_04_kernel_gradcheck.csv", kernel_rows)
        rows.append({"check": "kernel_gradcheck", "pass": int(all(int(r.get("kernel_correctness_official_pass", r.get("pass", 0))) for r in kernel_rows)), "metric": "kernels", "value": f"{len(kernel_rows)}", "blocker": ""})

    if _mode_enabled(mode, "profiler_phase_tests"):
        tests = efficiency_v22_04_unit_tests()
        write_rows(out_dir / "v22_04_profiler_phase_tests.csv", tests)
        rows.append({"check": "profiler_phase_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if rows:
        existing = []
        if (out_dir / "v22_04_code_truth_gate.csv").exists() and mode != "all":
            import csv

            with (out_dir / "v22_04_code_truth_gate.csv").open(newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        write_rows(out_dir / "v22_04_code_truth_gate.csv", existing + rows)
        route = {
            "mode": mode,
            "pass": int(all(int(r.get("pass", 0)) for r in existing + rows)),
            "failed_checks": ";".join(r.get("check", "") for r in existing + rows if not int(r.get("pass", 0))),
        }
        write_json(out_dir / "v22_04_code_route_decision.json", route)
        append_exec(out_dir, f"{PYTHON} experiments/run_v22_04_s011_truth_gate.py --mode {mode} --source-root {source_root}", status="completed", note=f"pass={route['pass']} failed={route['failed_checks']}")


if __name__ == "__main__":
    main()
