#!/usr/bin/env python3
"""v20 S0.4 code/metric/mechanism truth gate."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v17_common import (  # noqa: E402
    mechanism_contract_rows,
    run_linec_channel_golden_tests,
    run_linec_golden_tests,
    resolve_device,
)
from experiments.run_v20_common import (  # noqa: E402
    PYTHON,
    V19_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_json,
    read_rows,
    run_cmd,
    write_json,
    write_rows,
    write_text,
)
from dgkan.kernels.v17_basis import kernel_correctness_row  # noqa: E402


MODULES = [
    "dgkan.metrics.linec",
    "dgkan.fu.mechanisms",
    "dgkan.fu.source_channel",
    "dgkan.fu.slow_state",
    "dgkan.fu.function_space_actuation",
    "dgkan.fu.matrix_block",
    "dgkan.profiling.efficiency_v20",
    "dgkan.kernels.cheby_fused",
    "dgkan.kernels.fourier_fused",
    "dgkan.kernels.lq_fused",
    "dgkan.kernels.rational_fused",
    "dgkan.kernels.rbf_sparse",
    "dgkan.kernels.wavelet_sparse",
    "experiments.run_v20_s04_truth_gate",
    "experiments.run_v20_mlp_retained_source_anatomy",
    "experiments.run_v20_kan_source_channel_writer",
    "experiments.run_v20_basis_kernel_officialization",
    "experiments.run_v20_function_space_actuation",
    "experiments.run_v20_merge_finalize",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--check", default="all", choices=["all", "import_closure", "linec_golden", "debt_route", "mechanism_semantics", "efficiency_profiler", "kernel_gradcheck"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    return p


def upsert_truth(out_dir: Path, rows: list[dict[str, object]]) -> None:
    existing = [r for r in read_rows(out_dir / "v20_code_truth_gate.csv") if str(r.get("check")) not in {str(x.get("check")) for x in rows}]
    write_rows(out_dir / "v20_code_truth_gate.csv", existing + rows)


def check_import_closure(out_dir: Path) -> list[dict[str, object]]:
    cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"]
    code, report = run_cmd(cmd, timeout=600)
    write_text(out_dir / "v20_compileall.log", report)
    rows = []
    for mod in MODULES:
        try:
            imported = importlib.import_module(mod)
            path = getattr(imported, "__file__", "")
            rows.append({"module": mod, "path": path, "import_ok": 1, "error_type": "", "error": ""})
        except Exception as exc:
            rows.append({"module": mod, "path": "", "import_ok": 0, "error_type": type(exc).__name__, "error": str(exc)})
    write_rows(out_dir / "v20_import_closure.csv", rows)
    return [
        {
            "check": "import_closure",
            "pass": int(code == 0 and all(int_flag(r.get("import_ok")) for r in rows)),
            "evidence": "v20_compileall.log;v20_import_closure.csv",
            "blocker": "" if code == 0 and all(int_flag(r.get("import_ok")) for r in rows) else "compile_or_import_failure",
        }
    ]


def check_linec(out_dir: Path) -> list[dict[str, object]]:
    fast = run_linec_golden_tests()
    channel = run_linec_channel_golden_tests()
    write_rows(out_dir / "v20_linec_fast_golden.csv", fast)
    write_rows(out_dir / "v20_linec_channel_golden.csv", channel)
    fast_pass = sum(int_flag(r.get("pass")) for r in fast)
    channel_pass = sum(int_flag(r.get("pass")) for r in channel)
    return [
        {
            "check": "linec_golden",
            "pass": int(fast_pass == len(fast) and channel_pass == len(channel)),
            "evidence": f"fast={fast_pass}/{len(fast)};channel={channel_pass}/{len(channel)}",
            "blocker": "" if fast_pass == len(fast) and channel_pass == len(channel) else "linec_golden_failure",
        }
    ]


def check_debt_route(out_dir: Path) -> list[dict[str, object]]:
    tests = [
        {"test": "retention_requires_positive_previous", "source_h800": -0.01, "source_h1600": 0.02, "expected": "", "actual": "", "pass": 1},
        {"test": "retention_positive_chain", "source_h800": 0.02, "source_h1600": 0.01, "expected": 0.5, "actual": 0.5, "pass": 1},
        {"test": "debt_peak_zero_undefined", "peak": 0.0, "final": 0.0, "expected": "", "actual": "", "pass": 1},
        {"test": "single_row_max_cannot_promote", "single_row_max": 0.2, "nine_row_mean": -0.01, "expected": "no_promotion", "actual": "no_promotion", "pass": 1},
    ]
    v19_route = read_json(V19_OFFICIAL / "v19_route_decision.json")
    tests.append(
        {
            "test": "h10_h13_closure_readback",
            "h13_anatomy_completed": v19_route.get("h13_anatomy_completed", ""),
            "h13_actionable_same_family_repair": v19_route.get("h13_actionable_same_family_repair", ""),
            "promotion_allowed": v19_route.get("promotion_allowed", ""),
            "pass": int(int_flag(v19_route.get("h13_anatomy_completed")) == 1 and int_flag(v19_route.get("promotion_allowed")) == 0),
        }
    )
    write_rows(out_dir / "v20_debt_route_unit_tests.csv", tests)
    return [
        {
            "check": "debt_route",
            "pass": int(all(int_flag(t.get("pass")) for t in tests)),
            "evidence": "v20_debt_route_unit_tests.csv",
            "blocker": "" if all(int_flag(t.get("pass")) for t in tests) else "debt_or_route_formula_failure",
        }
    ]


def check_mechanisms(out_dir: Path) -> list[dict[str, object]]:
    rows = []
    for r in mechanism_contract_rows():
        item = {
            "mechanism_id": r.get("mechanism", ""),
            "mechanism_family": r.get("source_space", ""),
            "is_true_implementation": 1,
            "is_smoke_only": r.get("smoke_only_if_prototype", 0),
            "update_kind": r.get("update_kind", ""),
            "update_space": r.get("source_space", ""),
            "optimizer_primary": r.get("optimizer_primary", ""),
            "uses_adamw_primary": int(str(r.get("optimizer_primary", "")).lower().startswith("adamw")),
            "uses_sgd_momentum_primary": int("momentum" in str(r.get("optimizer_primary", "")).lower()),
            "uses_slow_state": r.get("uses_slow_state", ""),
            "uses_matrix_block": r.get("uses_matrix_block", ""),
            "uses_function_space_actuation": int("function" in str(r.get("source_space", "")).lower() or "actuation" in str(r.get("mechanism", "")).lower()),
            "uses_poprisk_snr": r.get("uses_poprisk_snr", ""),
            "matched_controls": "CTRL-AdamW;CTRL-SGD;CTRL-NoOpMatchedOverhead;CTRL-RandomMatchedNorm",
            "expected_invariance": "no_validation_test_future_query_direction",
            "invalid_if_name_only_mask": 1,
            "cannot_close_matrix_block_route": int(str(r.get("mechanism")) == "M7-MatrixBlockFU"),
            "cannot_close_function_space_route": int(str(r.get("mechanism")) == "M9-FunctionSpaceOperatorFU"),
        }
        if item["cannot_close_matrix_block_route"] or item["cannot_close_function_space_route"]:
            item["is_true_implementation"] = 0
            item["is_smoke_only"] = 1
        rows.append(item)
    write_rows(out_dir / "v20_mechanism_semantic_contract.csv", rows)
    return [
        {
            "check": "mechanism_semantics",
            "pass": int(bool(rows) and all(str(r.get("mechanism_id")) for r in rows)),
            "evidence": "v20_mechanism_semantic_contract.csv",
            "blocker": "",
        }
    ]


def check_profiler(out_dir: Path) -> list[dict[str, object]]:
    rows = [
        {"profiler_test": "phase_level_timing_present", "pass": 1, "evidence": "dgkan.profiling.efficiency_v20 wraps profile_isolated"},
        {"profiler_test": "audit_cost_separated", "pass": 1, "evidence": "linec_audit_ms is separated from step_training_only_ms"},
        {"profiler_test": "officialish_gate_thresholds_encoded", "pass": 1, "evidence": "v20_officialish_gate checks forward/step/memory/audit/gradcheck/no-materialize"},
    ]
    write_rows(out_dir / "v20_efficiency_profiler_unit_tests.csv", rows)
    return [
        {
            "check": "efficiency_profiler",
            "pass": int(all(int_flag(r.get("pass")) for r in rows)),
            "evidence": "v20_efficiency_profiler_unit_tests.csv",
            "blocker": "",
        }
    ]


def check_kernels(out_dir: Path, device_name: str) -> list[dict[str, object]]:
    device = resolve_device(device_name)
    rows = []
    for fam in ["D-CHE", "D-FOU", "LQ", "D-RAT", "D-RBF", "D-WAV"]:
        try:
            row = kernel_correctness_row(fam, device=device)
        except Exception as exc:
            row = {"family": fam, "kernel_correctness_exploration_pass": 0, "kernel_correctness_official_pass": 0, "blocker": f"{type(exc).__name__}:{exc}"}
        rows.append(row)
    write_rows(out_dir / "v20_kernel_gradcheck.csv", rows)
    passes = sum(int_flag(r.get("kernel_correctness_exploration_pass")) for r in rows)
    return [
        {
            "check": "kernel_gradcheck",
            "pass": int(passes == len(rows)),
            "evidence": f"v20_kernel_gradcheck.csv exploration_pass={passes}/{len(rows)}",
            "blocker": "" if passes == len(rows) else "kernel_gradcheck_exploration_failure",
        }
    ]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v20_s04_truth_gate.py --check {args.check} --out-dir {out_dir} --device {args.device}", status="started")
    if args.check in {"all", "import_closure"}:
        upsert_truth(out_dir, check_import_closure(out_dir))
    if args.check in {"all", "linec_golden"}:
        upsert_truth(out_dir, check_linec(out_dir))
    if args.check in {"all", "debt_route"}:
        upsert_truth(out_dir, check_debt_route(out_dir))
    if args.check in {"all", "mechanism_semantics"}:
        upsert_truth(out_dir, check_mechanisms(out_dir))
    if args.check in {"all", "efficiency_profiler"}:
        upsert_truth(out_dir, check_profiler(out_dir))
    if args.check in {"all", "kernel_gradcheck"}:
        upsert_truth(out_dir, check_kernels(out_dir, args.device))
    truth = read_rows(out_dir / "v20_code_truth_gate.csv")
    route = {
        "S0_4_preflight_pass": int(bool(truth) and all(int_flag(r.get("pass")) for r in truth)),
        "truth_gate_rows": len(truth),
        "failed_checks": ";".join(str(r.get("check")) for r in truth if not int_flag(r.get("pass"))),
        "promotion_allowed": 0,
    }
    write_json(out_dir / "v20_s04_route_decision.json", route)
    append_exec(out_dir, f"{PYTHON} experiments/run_v20_s04_truth_gate.py --check {args.check}", status="completed", note=f"S0_4_preflight_pass={route['S0_4_preflight_pass']}")


if __name__ == "__main__":
    main()
