#!/usr/bin/env python3
"""v22.03 S0.10 code truth gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.diffeomorphic_target import diffeomorphic_target_unit_tests  # noqa: E402
from dgkan.fu.mechanisms import MECHANISMS, mechanism_contract_rows  # noqa: E402
from dgkan.fu.terminal_retention import terminal_retention_unit_tests  # noqa: E402
from experiments.run_v22_03_common import PYTHON, append_exec, ensure_out, run_cmd, write_rows  # noqa: E402


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/terminal_retention.py",
    "dgkan/fu/diffeomorphic_target.py",
    "dgkan/fu/mechanisms.py",
    "experiments/run_v17_common.py",
    "experiments/run_v21_common.py",
    "experiments/run_v21_01_source_retention.py",
    "experiments/run_v22_03_common.py",
    "experiments/run_v22_03_code_truth_gate.py",
    "experiments/run_v22_03_efficiency_full_loop.py",
    "experiments/run_v22_03_terminal_erosion_autopsy.py",
    "experiments/run_v22_03_source_preservation.py",
    "experiments/run_v22_03_kan_source_channel_writer.py",
    "experiments/run_v22_03_drat_drbf_repair.py",
    "experiments/run_v22_03_drat_drbf_runner_integration.py",
    "experiments/run_v22_03_drat_drbf_limited_smoke_summary.py",
    "experiments/run_v22_03_finalize.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-root", default=str(ROOT))
    p.add_argument("--self-contained-import-check", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_root = Path(args.source_root)
    files = [source_root / rel for rel in REQUIRED_SOURCE_FILES]
    compile_cmd = [PYTHON, "-m", "compileall", "-q", *[str(p) for p in files if p.exists()]]
    code, log = run_cmd(compile_cmd, cwd=source_root, timeout=300)
    (out_dir / "v22_03_compileall.log").write_text(log, encoding="utf-8")
    import_code, import_log = run_cmd(
        [
            PYTHON,
            "-c",
            "import dgkan.fu.terminal_retention, dgkan.fu.diffeomorphic_target, experiments.run_v22_03_common, experiments.run_v22_03_source_preservation, experiments.run_v22_03_drat_drbf_repair, experiments.run_v22_03_drat_drbf_runner_integration, experiments.run_v22_03_drat_drbf_limited_smoke_summary",
        ],
        cwd=source_root,
        timeout=120,
    )
    (out_dir / "v22_03_import_closure.log").write_text(import_log, encoding="utf-8")
    retention_tests = terminal_retention_unit_tests()
    diffeo_tests = diffeomorphic_target_unit_tests()
    contract = mechanism_contract_rows()
    expected_v2203_mechs = [
        "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
        "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
        "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
        "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU",
        "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU",
        "M175-EarlySourceSlowEMALowRankReadoutTransportFU",
        "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
        "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
        "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
        "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
        "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
        "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
        "M185-EarlySourceSlowEMASNRTerminalPredictorFU",
        "M186-EarlySourceSlowEMASplitConsensusEstimatorFU",
        "M187-EarlySourceSlowEMASignalReservoirTransportFU",
        "M188-EarlySourceSlowEMATerminalSourceFloorFU",
        "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
        "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
        "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
        "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
        "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
        "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
        "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
        "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
        "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
        "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
        "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
        "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
        "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
        "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
        "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
        "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
        "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
        "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
        "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
        "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
        "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU",
        "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU",
        "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU",
        "M212-EarlySourceSlowEMATerminalAntiSourceClipFU",
        "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU",
        "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU",
        "M215-EarlySourceSlowEMATerminalAcceptMemoryFU",
        "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU",
        "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU",
        "M176-LowNDSDiffeomorphicTargetOnlyFU",
        "M177-InfoVolumeDiffeomorphicTargetOnlyFU",
        "M178-LowRankReadoutTransportTargetOnlyFU",
    ]
    v2203_mechs = [m for m in expected_v2203_mechs if m in MECHANISMS]
    rows = [
        {
            "check": "required_source_files",
            "pass": int(all(p.exists() for p in files)),
            "metric": "exists",
            "value": f"{sum(1 for p in files if p.exists())}/{len(files)}",
            "blocker": ";".join(str(p.relative_to(source_root)) for p in files if not p.exists()),
        },
        {
            "check": "compileall",
            "pass": int(code == 0),
            "metric": "py_compile",
            "value": code,
            "blocker": "" if code == 0 else "compile_failed",
        },
        {
            "check": "import_closure",
            "pass": int(import_code == 0),
            "metric": "import",
            "value": import_code,
            "blocker": "" if import_code == 0 else "import_failed",
        },
        {
            "check": "terminal_retention_tests",
            "pass": int(all(int(r.get("pass", 0)) for r in retention_tests)),
            "metric": "tests",
            "value": f"{sum(int(r.get('pass', 0)) for r in retention_tests)}/{len(retention_tests)}",
            "blocker": "",
        },
        {
            "check": "diffeomorphic_target_tests",
            "pass": int(all(int(r.get("pass", 0)) for r in diffeo_tests)),
            "metric": "tests",
            "value": f"{sum(int(r.get('pass', 0)) for r in diffeo_tests)}/{len(diffeo_tests)}",
            "blocker": "",
        },
        {
            "check": "mechanism_contracts",
            "pass": int(
                len(v2203_mechs) == len(expected_v2203_mechs)
                and all(any(r.get("mechanism") == m for r in contract) for m in expected_v2203_mechs)
            ),
            "metric": "v22.03 mechanisms",
            "value": f"{len(v2203_mechs)}/{len(expected_v2203_mechs)}",
            "blocker": "",
        },
    ]
    write_rows(out_dir / "v22_03_code_truth_gate.csv", rows)
    write_rows(out_dir / "v22_03_terminal_retention_unit_tests.csv", retention_tests)
    write_rows(out_dir / "v22_03_diffeomorphic_target_unit_tests.csv", diffeo_tests)
    write_rows(out_dir / "v22_03_mechanism_contracts.csv", [r for r in contract if str(r.get("mechanism", "")) in expected_v2203_mechs])
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_03_code_truth_gate.py --source-root {source_root}", status="completed", note=f"pass={int(all(int(r['pass']) for r in rows))}")


if __name__ == "__main__":
    main()
