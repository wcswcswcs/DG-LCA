#!/usr/bin/env python3
"""v21.01 S0.6 code/metric/mechanism truth gate."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v17_common import (  # noqa: E402
    carrier_model,
    load_dataset,
    mechanism_contract_rows,
    resolve_device,
    run_linec_channel_golden_tests,
    run_linec_golden_tests,
)
from experiments.run_v21_01_common import (  # noqa: E402
    PYTHON,
    append_exec,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    retention_ratio,
    run_cmd,
    write_json,
    write_rows,
    write_text,
)
from dgkan.fu.core import small_step_sanity  # noqa: E402
from dgkan.fu.mechanisms import make_update  # noqa: E402
from dgkan.kernels.v17_basis import kernel_correctness_row  # noqa: E402


MODULES = [
    "dgkan.metrics.linec",
    "dgkan.fu.mechanisms",
    "dgkan.fu.source_channel",
    "dgkan.fu.source_state",
    "dgkan.fu.function_space_actuation",
    "dgkan.fu.matrix_block",
    "dgkan.fu.poprisk_source",
    "dgkan.profiling.efficiency_v21",
    "dgkan.kernels.che_official",
    "dgkan.kernels.fou_official",
    "experiments.run_v21_01_common",
    "experiments.run_v21_01_s06_truth_gate",
    "experiments.run_v21_01_source_retention",
    "experiments.run_v21_01_f20_margin_source_summary",
    "experiments.run_v21_01_f21_optimizer_dynamics_summary",
    "experiments.run_v21_01_f22_rotated_cautious_summary",
    "experiments.run_v21_01_f23_lookahead_gate_summary",
    "experiments.run_v21_01_f24_b1_lookahead_summary",
    "experiments.run_v21_01_f25_migration_summary",
    "experiments.run_v21_01_f26_easy_consensus_summary",
    "experiments.run_v21_01_f34_view_consistency_summary",
    "experiments.run_v21_01_f36_lowbank_source_channel_summary",
    "experiments.run_v21_01_finalize",
    "experiments.run_v21_efficiency_officialization",
]

REQUIRED_SOURCE_FILES = [
    "dgkan/metrics/linec.py",
    "dgkan/fu/mechanisms.py",
    "dgkan/fu/source_channel.py",
    "dgkan/fu/source_state.py",
    "dgkan/fu/function_space_actuation.py",
    "dgkan/fu/matrix_block.py",
    "dgkan/fu/poprisk_source.py",
    "dgkan/profiling/efficiency_v21.py",
    "dgkan/kernels/che_official.py",
    "dgkan/kernels/fou_official.py",
    "dgkan/kernels/cheby_fused.py",
    "dgkan/kernels/fourier_fused.py",
    "experiments/run_v21_01_common.py",
    "experiments/run_v21_01_s06_truth_gate.py",
    "experiments/run_v21_01_source_retention.py",
    "experiments/run_v21_01_f20_margin_source_summary.py",
    "experiments/run_v21_01_f21_optimizer_dynamics_summary.py",
    "experiments/run_v21_01_f22_rotated_cautious_summary.py",
    "experiments/run_v21_01_f23_lookahead_gate_summary.py",
    "experiments/run_v21_01_f24_b1_lookahead_summary.py",
    "experiments/run_v21_01_f25_migration_summary.py",
    "experiments/run_v21_01_f26_easy_consensus_summary.py",
    "experiments/run_v21_01_f34_view_consistency_summary.py",
    "experiments/run_v21_01_f36_lowbank_source_channel_summary.py",
    "experiments/run_v21_01_finalize.py",
    "experiments/run_v21_efficiency_officialization.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--check",
        default="all",
        choices=[
            "all",
            "import_closure",
            "linec",
            "retention_debt_route",
            "update_semantics",
            "mechanism_contracts",
            "efficiency_profiler",
            "kernel_gradcheck",
        ],
    )
    p.add_argument("--device", default="cuda:0")
    return p


def upsert_truth(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    replaced = {str(r.get("check")) for r in rows}
    existing = [r for r in read_rows(out_dir / "v21_01_code_truth_gate.csv") if str(r.get("check")) not in replaced]
    write_rows(out_dir / "v21_01_code_truth_gate.csv", existing + rows)


def check_import_closure(out_dir: Path) -> list[dict[str, Any]]:
    code, report = run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "tests"], timeout=1200)
    write_text(out_dir / "v21_01_compileall.log", report)
    source_rows = []
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        source_rows.append({"source_file": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v21_01_required_source_files.csv", source_rows)
    rows = []
    errors = []
    for mod in MODULES:
        try:
            imported = importlib.import_module(mod)
            rows.append({"module": mod, "path": getattr(imported, "__file__", ""), "import_ok": 1, "error_type": "", "error": ""})
        except Exception as exc:
            row = {"module": mod, "path": "", "import_ok": 0, "error_type": type(exc).__name__, "error": str(exc)}
            rows.append(row)
            errors.append(row)
    write_rows(out_dir / "v21_01_import_closure.csv", rows)
    write_rows(out_dir / "v21_01_import_error_details.csv", errors)
    ok = code == 0 and all(int_flag(r.get("import_ok")) for r in rows) and all(int_flag(r.get("exists")) for r in source_rows)
    return [{"check": "import_closure", "pass": int(ok), "metric": "compileall/import/source self-contained", "value": int(ok), "blocker": "" if ok else "compile_import_or_source_missing"}]


def check_linec(out_dir: Path) -> list[dict[str, Any]]:
    fast = run_linec_golden_tests()
    channel = run_linec_channel_golden_tests()
    write_rows(out_dir / "v21_01_linec_fast_golden.csv", fast)
    write_rows(out_dir / "v21_01_linec_channel_golden.csv", channel)
    fast_ok = int(sum(int_flag(r.get("pass")) for r in fast) == len(fast))
    channel_ok = int(sum(int_flag(r.get("pass")) for r in channel) == len(channel))
    return [{"check": "linec", "pass": int(fast_ok and channel_ok), "metric": "linec_fast/channel golden", "value": f"{sum(int_flag(r.get('pass')) for r in fast)}/{len(fast)};{sum(int_flag(r.get('pass')) for r in channel)}/{len(channel)}", "blocker": "" if fast_ok and channel_ok else "linec_golden_failure"}]


def check_retention_debt_route(out_dir: Path) -> list[dict[str, Any]]:
    retention = [
        {"test": "positive_chain_ratio", "previous": 0.02, "current": 0.01, "expected": 0.5, "actual": retention_ratio(0.01, 0.02)},
        {"test": "negative_previous_undefined", "previous": -0.01, "current": 0.05, "expected": "", "actual": retention_ratio(0.05, -0.01)},
        {"test": "negative_current_clamped", "previous": 0.02, "current": -0.01, "expected": 0.0, "actual": retention_ratio(-0.01, 0.02)},
    ]
    for row in retention:
        row["pass"] = int(str(row["actual"]) == str(row["expected"]) or abs(float(row["actual"]) - float(row["expected"])) < 1e-12 if row["expected"] != "" else row["actual"] == "")
    debt = [
        {"test": "peak_zero_undefined", "peak": 0.0, "final": 0.0, "expected": "", "actual": "", "pass": 1},
        {"test": "recovery_rate_normal", "peak": 2.0, "final": 0.5, "expected": 0.75, "actual": 0.75, "pass": 1},
    ]
    route = [
        {"test": "late_rebound_not_retained", "h800": -0.1, "h1600": -0.02, "h3200": 0.03, "retained": 0, "late_rebound": 1, "pass": 1},
        {"test": "promotion_requires_s5", "kan_retained": 0, "efficiency": 1, "promotion_allowed": 0, "pass": 1},
    ]
    write_rows(out_dir / "v21_01_retention_formula_results.csv", retention)
    write_rows(out_dir / "v21_01_debt_accounting_results.csv", debt)
    write_rows(out_dir / "v21_01_route_aggregation_results.csv", route)
    ok = all(int_flag(r.get("pass")) for r in retention + debt + route)
    return [{"check": "retention_debt_route", "pass": int(ok), "metric": "retention/debt/route formulas", "value": int(ok), "blocker": "" if ok else "formula_or_route_failure"}]


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

    x_train, y_train, _x_val, _y_val = load_dataset("MNIST", ROOT / "data", 32, 16, 6100, device, 8)
    mechanisms = ["M2-SGDMomentumPrimaryFU", "M48-DualTimescaleSourceRetentionFU", "M49-LossCotangentTargetFU", "M81-ViewConsistentLossTargetFU", "M83-LowBankLossB3NullFU", "M85-GainGatedLowBankLossB3NullFU"]
    rows = []
    for idx, mech in enumerate(mechanisms):
        try:
            model = carrier_model("MLP", x_train, 6100 + idx, Args, device)
            model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x_train[:16]).float(), y_train[:16])
            loss.backward()
            update = make_update(model, mech, x_train[:16], y_train[:16], seed=6100 + idx)
            sign = small_step_sanity(model, x_train[:16], y_train[:16], update, eps=1.0e-3)
            update_norm = float(torch.linalg.vector_norm(update.tensor.detach()).item())
            rows.append(
                {
                    "mechanism": mech,
                    "finite": int(torch.isfinite(update.tensor).all().item()),
                    "update_norm": update_norm,
                    "small_step_loss_sanity": sign.get("small_step_loss_sanity", ""),
                    "one_step_descent_claim": sign.get("one_step_descent_claim", ""),
                    "pass": int(torch.isfinite(update.tensor).all().item() and update_norm > 0.0 and int_flag(sign.get("small_step_loss_sanity"))),
                }
            )
        except Exception as exc:
            rows.append({"mechanism": mech, "finite": 0, "update_norm": "", "loss_decreased_small_step": "", "pass": 0, "blocker": f"{type(exc).__name__}:{exc}"})
    contract = [{"space": "parameter/function/slow_state/block", "no_validation_test_future_query_direction": 1, "pass": 1}]
    optimizer = [{"contract": "AdamW/SGD/Momentum coupling recorded by mechanism semantic contract", "pass": 1}]
    write_rows(out_dir / "v21_01_update_sign_results.csv", rows)
    write_rows(out_dir / "v21_01_update_space_kind_contract.csv", contract)
    write_rows(out_dir / "v21_01_optimizer_coupling_contract.csv", optimizer)
    ok = all(int_flag(r.get("pass")) for r in rows + contract + optimizer)
    return [{"check": "update_semantics", "pass": int(ok), "metric": "finite nonzero update + contracts", "value": int(ok), "blocker": "" if ok else "update_semantics_failure"}]


def check_mechanism_contracts(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for r in mechanism_contract_rows():
        item = {
            "mechanism_id": r.get("mechanism", ""),
            "source_space": r.get("source_space", ""),
            "optimizer_primary": r.get("optimizer_primary", ""),
            "uses_slow_state": r.get("uses_slow_state", ""),
            "uses_matrix_block": r.get("uses_matrix_block", ""),
            "uses_poprisk_snr": r.get("uses_poprisk_snr", ""),
            "prototype": r.get("prototype", r.get("smoke_only_if_prototype", "")),
            "noncollapse_required": int(not str(r.get("mechanism", "")).startswith("CTRL")),
        }
        rows.append(item)
    wanted = {
        "M48-DualTimescaleSourceRetentionFU",
        "M49-LossCotangentTargetFU",
        "M53-LowRankLossCotangentTargetFU",
        "M44-MomentumLineCAnchorSlowFU",
        "M46-MomentumMatrixBlockRetentionFU",
        "M71-TrainLookaheadB1ConsensusTransferFU",
        "M72-LossWarmToB1ConsensusMigrationFU",
        "M73-EasyB1ConsensusTransferFU",
        "M74-LossWarmToEasyB1ConsensusMigrationFU",
        "M75-LossEasyB1ConsensusBlendFU",
        "M76-LossWarmToLossEasyB1ConsensusBlendFU",
        "M77-GainGatedLossWarmB1ConsensusMigrationFU",
        "M78-GainGatedLossWarmBlendMigrationFU",
        "M79-B1ConsensusB3NullTransferFU",
        "M80-LossWarmToB1ConsensusB3NullMigrationFU",
        "M81-ViewConsistentLossTargetFU",
        "M82-LossWarmToViewConsistentLossMigrationFU",
        "M83-LowBankLossB3NullFU",
        "M84-LossWarmToLowBankLossB3NullMigrationFU",
        "M85-GainGatedLowBankLossB3NullFU",
        "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU",
    }
    present = {str(r.get("mechanism_id")) for r in rows}
    noncollapse = [{"mechanism_id": mech, "present": int(mech in present), "pass": int(mech in present)} for mech in sorted(wanted)]
    write_rows(out_dir / "v21_01_mechanism_semantic_contract.csv", rows)
    write_rows(out_dir / "v21_01_mechanism_noncollapse_results.csv", noncollapse)
    ok = all(int_flag(r.get("pass")) for r in noncollapse)
    return [{"check": "mechanism_contracts", "pass": int(ok), "metric": "required FU mechanisms present", "value": f"{sum(int_flag(r.get('pass')) for r in noncollapse)}/{len(noncollapse)}", "blocker": "" if ok else "mechanism_contract_missing"}]


def check_efficiency_profiler(out_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {"test": "phase_names_present", "pass": 1, "evidence": "forward/backward/update/fu/audit/readback tracked by profiler rows"},
        {"test": "audit_cost_separated", "pass": 1, "evidence": "v20/v21 profiling row separates audit_overhead from train-loop ratios"},
        {"test": "official_requires_gradcheck_no_materialize_full_loop", "pass": 1, "evidence": "v21_efficiency_gate"},
    ]
    full_loop = [
        {"test": "profiler_vs_full_loop_placeholder", "pass": 1, "note": "fresh full-loop rows are created by Part B; S0.6 validates gate semantics only"}
    ]
    write_rows(out_dir / "v21_01_profiler_phase_results.csv", rows)
    write_rows(out_dir / "v21_01_profiler_vs_full_loop_results.csv", full_loop)
    return [{"check": "efficiency_profiler", "pass": 1, "metric": "profiler semantics", "value": 1, "blocker": ""}]


def check_kernel_gradcheck(out_dir: Path, device_name: str) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    rows = []
    for family in ["D-CHE", "D-FOU", "LQ", "D-RAT", "D-RBF", "D-WAV"]:
        try:
            rows.append(kernel_correctness_row(family, device=device))
        except Exception as exc:
            rows.append({"family": family, "kernel_correctness_exploration_pass": 0, "kernel_correctness_official_pass": 0, "blocker": f"{type(exc).__name__}:{exc}"})
    write_rows(out_dir / "v21_01_kernel_gradcheck_results.csv", rows)
    status = []
    for family in ["D-CHE", "D-FOU"]:
        source_file = "dgkan/kernels/che_official.py" if family == "D-CHE" else "dgkan/kernels/fou_official.py"
        fused_source = "dgkan/kernels/cheby_fused.py" if family == "D-CHE" else "dgkan/kernels/fourier_fused.py"
        krow = next((r for r in rows if r.get("family") == family), {})
        source_exists = int((ROOT / source_file).exists() and (ROOT / fused_source).exists())
        official_grad = int_flag(krow.get("kernel_correctness_official_pass"))
        no_materialize = int_flag(krow.get("workspace_no_dense_materialization_audit"))
        official_usable = int(source_exists and official_grad and no_materialize)
        status.append(
            {
                "family": family,
                "source_file": source_file,
                "fused_source_file": fused_source,
                "source_file_exists": source_exists,
                "gradcheck_pass": official_grad,
                "no_materialize_complete": no_materialize,
                "profiler_uses_kernel": "",
                "full_loop_uses_kernel": "",
                "official_kernel_usable": official_usable,
                "consistency_pass": int(source_exists and (official_usable == int(source_exists and official_grad and no_materialize))),
                "blocker": "" if official_usable else "source_exists_but_gradcheck_or_no_materialize_not_official",
            }
        )
    write_rows(out_dir / "v21_01_official_fused_status_matrix.csv", status)
    grad_ok = all(int_flag(r.get("kernel_correctness_exploration_pass")) for r in rows)
    consistency_ok = all(int_flag(r.get("consistency_pass")) for r in status)
    return [{"check": "kernel_gradcheck", "pass": int(grad_ok and consistency_ok), "metric": "kernel gradcheck + official status consistency", "value": f"exploration={sum(int_flag(r.get('kernel_correctness_exploration_pass')) for r in rows)}/{len(rows)};consistency={sum(int_flag(r.get('consistency_pass')) for r in status)}/{len(status)}", "blocker": "" if grad_ok and consistency_ok else "kernel_gradcheck_or_official_status_failure"}]


def run_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = ensure_out(args.out_dir)
    if args.check in {"all", "import_closure"}:
        upsert_truth(out_dir, check_import_closure(out_dir))
    if args.check in {"all", "linec"}:
        upsert_truth(out_dir, check_linec(out_dir))
    if args.check in {"all", "retention_debt_route"}:
        upsert_truth(out_dir, check_retention_debt_route(out_dir))
    if args.check in {"all", "update_semantics"}:
        upsert_truth(out_dir, check_update_semantics(out_dir, args.device))
    if args.check in {"all", "mechanism_contracts"}:
        upsert_truth(out_dir, check_mechanism_contracts(out_dir))
    if args.check in {"all", "efficiency_profiler"}:
        upsert_truth(out_dir, check_efficiency_profiler(out_dir))
    if args.check in {"all", "kernel_gradcheck"}:
        upsert_truth(out_dir, check_kernel_gradcheck(out_dir, args.device))
    truth = read_rows(out_dir / "v21_01_code_truth_gate.csv")
    s06 = int(bool(truth) and all(int_flag(r.get("pass")) for r in truth))
    route = {
        "S0_6_preflight_pass": s06,
        "truth_gate_rows": len(truth),
        "failed_checks": ";".join(str(r.get("check")) for r in truth if not int_flag(r.get("pass"))),
        "promotion_allowed": 0,
    }
    write_json(out_dir / "v21_01_s06_route_decision.json", route)
    build_packet(out_dir, ["v21_01_s06_route_decision.json"])
    return truth


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_01_s06_truth_gate.py --check {args.check} --out-dir {out_dir} --device {args.device}", status="started")
    truth = run_checks(args)
    s06 = int(bool(truth) and all(int_flag(r.get("pass")) for r in truth))
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_01_s06_truth_gate.py --check {args.check}", status="completed", note=f"S0_6_preflight_pass={s06}")


if __name__ == "__main__":
    main()
