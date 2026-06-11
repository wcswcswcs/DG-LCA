#!/usr/bin/env python3
"""v22 S0.7 truth gate."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.core import small_step_sanity  # noqa: E402
from dgkan.fu.mechanisms import mechanism_contract_rows  # noqa: E402
from dgkan.fu.source_chain import source_chain_unit_tests  # noqa: E402
from experiments.run_v17_common import (  # noqa: E402
    carrier_model,
    load_dataset,
    make_update,
    resolve_device,
    run_linec_channel_golden_tests,
    run_linec_golden_tests,
)
from experiments.run_v22_common import (  # noqa: E402
    PYTHON,
    ROOT,
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
    "dgkan.fu.mechanisms",
    "dgkan.fu.source_chain",
    "dgkan.fu.poprisk_snr",
    "dgkan.fu.slow_state",
    "dgkan.profiling.efficiency_v22",
    "experiments.run_v22_common",
    "experiments.run_v22_source_chain_dynamics",
    "experiments.run_v22_efficiency_officialization",
    "experiments.run_v22_merge_finalize",
    "experiments.run_v22_mlp_source_lab",
    "experiments.run_v22_kan_source_writer",
    "experiments.run_v22_function_space_target_reset",
    "experiments.run_v22_source_observability_audit",
]

REQUIRED_SOURCE_FILES = [
    "dgkan/fu/core.py",
    "dgkan/fu/mechanisms.py",
    "dgkan/fu/source_chain.py",
    "dgkan/fu/function_space_actuation.py",
    "dgkan/fu/linec_readback.py",
    "dgkan/fu/poprisk_snr.py",
    "dgkan/fu/slow_state.py",
    "dgkan/profiling/efficiency_v22.py",
    "experiments/run_v22_common.py",
    "experiments/run_v22_s07_truth_gate.py",
    "experiments/run_v22_efficiency_officialization.py",
    "experiments/run_v22_source_chain_dynamics.py",
    "experiments/run_v22_mlp_source_lab.py",
    "experiments/run_v22_kan_source_writer.py",
    "experiments/run_v22_function_space_target_reset.py",
    "experiments/run_v22_source_observability_audit.py",
    "experiments/run_v22_merge_finalize.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--check",
        default="all",
        choices=["all", "import_closure", "source_chain", "linec", "update_semantics", "mechanism_contracts", "efficiency_profiler"],
    )
    return p


def upsert_truth(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    replaced = {str(r.get("check")) for r in rows}
    existing = [r for r in read_rows(out_dir / "v22_code_truth_gate.csv") if str(r.get("check")) not in replaced]
    write_rows(out_dir / "v22_code_truth_gate.csv", existing + rows)


def check_import_closure(out_dir: Path) -> list[dict[str, Any]]:
    code, report = run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "tests"], timeout=1200)
    write_text(out_dir / "v22_compileall.log", report)
    source_rows = []
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        source_rows.append({"source_file": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v22_required_source_files.csv", source_rows)
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
    write_rows(out_dir / "v22_import_closure.csv", imports)
    write_rows(out_dir / "v22_import_error_details.csv", errors)
    ok = code == 0 and all(int_flag(r.get("import_ok")) for r in imports) and all(int_flag(r.get("exists")) for r in source_rows)
    return [{"check": "import_closure", "pass": int(ok), "metric": "compileall/import/source self-contained", "value": int(ok), "blocker": "" if ok else "compile_import_or_source_missing"}]


def check_source_chain(out_dir: Path) -> list[dict[str, Any]]:
    rows = source_chain_unit_tests()
    write_rows(out_dir / "v22_source_chain_formula_tests.csv", rows)
    ok = all(int_flag(r.get("pass")) for r in rows)
    return [{"check": "source_chain", "pass": int(ok), "metric": "early/continuous/late formulas", "value": f"{sum(int_flag(r.get('pass')) for r in rows)}/{len(rows)}", "blocker": "" if ok else "source_chain_formula_failure"}]


def check_linec(out_dir: Path) -> list[dict[str, Any]]:
    fast = run_linec_golden_tests()
    channel = run_linec_channel_golden_tests()
    write_rows(out_dir / "v22_linec_fast_golden.csv", fast)
    write_rows(out_dir / "v22_linec_channel_golden.csv", channel)
    ok = all(int_flag(r.get("pass")) for r in fast + channel)
    return [{"check": "linec", "pass": int(ok), "metric": "linec_fast/channel golden", "value": f"{sum(int_flag(r.get('pass')) for r in fast)}/{len(fast)};{sum(int_flag(r.get('pass')) for r in channel)}/{len(channel)}", "blocker": "" if ok else "linec_golden_failure"}]


def check_update_semantics(out_dir: Path, device_name: str) -> list[dict[str, Any]]:
    device = resolve_device(device_name)

    class Args:
        input_size = 8
        classes = 10
        hidden = 16
        param_budget = 4000
        basis_repair_variant = "R0-current"
        fu_lr = 1.0e-3
        sanity_eps = 1.0e-3

    x_train, y_train, _x_val, _y_val = load_dataset("MNIST", ROOT / "data", 32, 16, 6220, device, 8)
    mechanisms = [
        "M17-ReadoutCarrierTwoPhaseLineCFU",
        "M49-LossCotangentTargetFU",
        "M38-AdamWSplitFisherAgreementResidualFU",
        "M60-B1CrossSplitConsensusTransferFU",
        "M72-LossWarmToB1ConsensusMigrationFU",
        "M81-ViewConsistentLossTargetFU",
        "M83-LowBankLossB3NullFU",
        "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU",
        "M109-AdamWBoundaryToGainGatedLowBankB3NullFU",
        "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU",
        "M111-TrainLossTerminalPositiveLookaheadFloorFU",
        "M112-TrainLossTerminalCheckpointReentryFU",
        "M113-TrainLossTerminalHardSplitSourceFU",
        "M114-TrainLossTerminalAdamWLookaheadFU",
        "M115-TrainLossTerminalOptimizerSelectorFU",
    ]
    rows = []
    for idx, mech in enumerate(mechanisms):
        try:
            model = carrier_model("D-FOU", x_train, 6220 + idx, Args, device)
            model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x_train[:16]).float(), y_train[:16])
            loss.backward()
            update = make_update(model, mech, x_train[:16], y_train[:16], seed=6220 + idx)
            sign = small_step_sanity(model, x_train[:16], y_train[:16], update, eps=1.0e-3)
            finite = int(torch.isfinite(update.tensor).all().item())
            norm = float(torch.linalg.vector_norm(update.tensor.detach()).item())
            rows.append(
                {
                    "mechanism": mech,
                    "finite": finite,
                    "update_norm": norm,
                    "space": update.space,
                    "kind": update.kind,
                    "source": update.source,
                    "small_step_loss_sanity": sign.get("small_step_loss_sanity", ""),
                    "pass": int(finite and norm > 0.0),
                }
            )
        except Exception as exc:
            rows.append({"mechanism": mech, "finite": 0, "update_norm": "", "pass": 0, "blocker": f"{type(exc).__name__}:{exc}"})
    write_rows(out_dir / "v22_update_semantics_results.csv", rows)
    ok = all(int_flag(r.get("pass")) for r in rows)
    return [{"check": "update_semantics", "pass": int(ok), "metric": "finite nonzero update semantics", "value": f"{sum(int_flag(r.get('pass')) for r in rows)}/{len(rows)}", "blocker": "" if ok else "update_semantics_failure"}]


def check_mechanism_contracts(out_dir: Path) -> list[dict[str, Any]]:
    rows = list(mechanism_contract_rows())
    write_rows(out_dir / "v22_mechanism_semantic_contracts.csv", rows)
    wanted = {
        "M17-ReadoutCarrierTwoPhaseLineCFU",
        "M38-AdamWSplitFisherAgreementResidualFU",
        "M49-LossCotangentTargetFU",
        "M60-B1CrossSplitConsensusTransferFU",
        "M72-LossWarmToB1ConsensusMigrationFU",
        "M77-GainGatedLossWarmB1ConsensusMigrationFU",
        "M80-LossWarmToB1ConsensusB3NullMigrationFU",
        "M81-ViewConsistentLossTargetFU",
        "M82-LossWarmToViewConsistentLossMigrationFU",
        "M83-LowBankLossB3NullFU",
        "M84-LossWarmToLowBankLossB3NullMigrationFU",
        "M85-GainGatedLowBankLossB3NullFU",
        "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU",
        "M109-AdamWBoundaryToGainGatedLowBankB3NullFU",
        "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU",
        "M97-TrainLossLateHoldRecoveryFU",
        "M98-TrainLossLateLookaheadFloorFU",
        "M99-TrainLossTerminalLookaheadFloorFU",
        "M100-TrainLossEarlyTerminalLookaheadFloorFU",
        "M101-AdamWBoundaryDualTimescaleAntiWashoutFU",
        "M102-AdamWBoundaryDualTimescaleSourceAnchorFU",
        "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU",
        "M104-AdamWBoundaryDualTimescaleReadoutChannelFU",
        "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU",
        "M106-TrainLossTerminalProjectedLookaheadFloorFU",
        "M107-TrainLossTerminalConsensusLookaheadFloorFU",
        "M108-TrainLossTerminalSelectorLookaheadFloorFU",
        "M111-TrainLossTerminalPositiveLookaheadFloorFU",
        "M112-TrainLossTerminalCheckpointReentryFU",
        "M113-TrainLossTerminalHardSplitSourceFU",
        "M114-TrainLossTerminalAdamWLookaheadFU",
        "M115-TrainLossTerminalOptimizerSelectorFU",
    }
    got = {str(r.get("mechanism", "")) for r in rows}
    missing = sorted(wanted - got)
    write_rows(out_dir / "v22_mechanism_contract_missing.csv", [{"mechanism": m} for m in missing])
    ok = not missing
    return [{"check": "mechanism_contracts", "pass": int(ok), "metric": "planned mechanism contracts", "value": f"{len(wanted)-len(missing)}/{len(wanted)}", "blocker": "" if ok else "missing_mechanism_contracts:" + ",".join(missing)}]


def check_efficiency_profiler(out_dir: Path) -> list[dict[str, Any]]:
    from dgkan.profiling.efficiency_v22 import v22_efficiency_gate

    row = {
        "forward_ratio_vs_mlp": 1.05,
        "backward_ratio_vs_mlp": 1.05,
        "step_ratio_vs_mlp": 1.05,
        "memory_ratio_vs_mlp": 1.0,
        "full_loop_timing_pass": 1,
        "manual_grad_relerr_max": 1.0e-6,
        "no_materialize_complete": 1,
        "official_fused_kernel_complete": 1,
    }
    gate = v22_efficiency_gate(row)
    write_rows(out_dir / "v22_efficiency_profiler_gate_smoke.csv", [{**row, **gate}])
    ok = int_flag(gate.get("v22_E1_exploration_gate")) and int_flag(gate.get("v22_S1_official_like_gate"))
    return [{"check": "efficiency_profiler", "pass": int(ok), "metric": "v22 E1/S1 gate smoke", "value": int(ok), "blocker": "" if ok else "efficiency_gate_smoke_failure"}]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_s07_truth_gate.py --check {args.check} --device {args.device}", status="started")
    if args.check in {"all", "import_closure"}:
        upsert_truth(out_dir, check_import_closure(out_dir))
    if args.check in {"all", "source_chain"}:
        upsert_truth(out_dir, check_source_chain(out_dir))
    if args.check in {"all", "linec"}:
        upsert_truth(out_dir, check_linec(out_dir))
    if args.check in {"all", "update_semantics"}:
        upsert_truth(out_dir, check_update_semantics(out_dir, args.device))
    if args.check in {"all", "mechanism_contracts"}:
        upsert_truth(out_dir, check_mechanism_contracts(out_dir))
    if args.check in {"all", "efficiency_profiler"}:
        upsert_truth(out_dir, check_efficiency_profiler(out_dir))
    rows = read_rows(out_dir / "v22_code_truth_gate.csv")
    ok = all(int_flag(r.get("pass")) for r in rows)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_s07_truth_gate.py --check {args.check}", status="completed", note=f"S0.7_pass={int(ok)} checks={len(rows)}")


if __name__ == "__main__":
    main()
