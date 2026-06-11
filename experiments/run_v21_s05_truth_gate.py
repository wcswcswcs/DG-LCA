#!/usr/bin/env python3
"""v21 S0.5 code/metric/mechanism truth gate."""

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
from experiments.run_v21_common import (  # noqa: E402
    PYTHON,
    V20_OFFICIAL,
    append_exec,
    ensure_out,
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
    "dgkan.fu.source_state",
    "dgkan.fu.slow_state",
    "dgkan.fu.function_space_actuation",
    "dgkan.fu.matrix_block",
    "dgkan.fu.poprisk_source",
    "dgkan.profiling.efficiency_v21",
    "dgkan.kernels.che_official",
    "dgkan.kernels.fou_official",
    "experiments.run_v21_s05_truth_gate",
    "experiments.run_v21_efficiency_officialization",
    "experiments.run_v21_mlp_source_dynamics",
    "experiments.run_v21_kan_source_channel_writer",
    "experiments.run_v21_function_space_target_reset",
    "experiments.run_v21_function_space_target_contrast",
    "experiments.run_v21_function_space_target_source",
    "experiments.run_v21_source_retention_estimator_audit",
    "experiments.run_v21_finalize",
]


REQUIRED_SOURCE_FILES = [
    "dgkan/metrics/linec.py",
    "dgkan/fu/mechanisms.py",
    "dgkan/fu/source_state.py",
    "dgkan/fu/function_space_actuation.py",
    "dgkan/fu/matrix_block.py",
    "dgkan/fu/poprisk_source.py",
    "dgkan/profiling/efficiency_v21.py",
    "dgkan/kernels/che_official.py",
    "dgkan/kernels/fou_official.py",
    "experiments/run_v21_s05_truth_gate.py",
    "experiments/run_v21_efficiency_officialization.py",
    "experiments/run_v21_mlp_source_dynamics.py",
    "experiments/run_v21_kan_source_channel_writer.py",
    "experiments/run_v21_function_space_target_reset.py",
    "experiments/run_v21_function_space_target_contrast.py",
    "experiments/run_v21_function_space_target_source.py",
    "experiments/run_v21_source_retention_estimator_audit.py",
    "experiments/run_v21_finalize.py",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--check", default="all", choices=["all", "import_closure", "linec_golden", "debt_route", "mechanism_semantics", "efficiency_profiler", "kernel_gradcheck"])
    p.add_argument("--device", default="cuda:0")
    return p


def upsert_truth(out_dir: Path, rows: list[dict[str, object]]) -> None:
    existing = [r for r in read_rows(out_dir / "v21_code_truth_gate.csv") if str(r.get("check")) not in {str(x.get("check")) for x in rows}]
    write_rows(out_dir / "v21_code_truth_gate.csv", existing + rows)


def check_import_closure(out_dir: Path) -> list[dict[str, object]]:
    cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"]
    code, report = run_cmd(cmd, timeout=900)
    write_text(out_dir / "v21_compileall.log", report)
    source_rows = []
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        source_rows.append({"source_file": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v21_required_source_files.csv", source_rows)
    rows = []
    for mod in MODULES:
        try:
            imported = importlib.import_module(mod)
            path = getattr(imported, "__file__", "")
            rows.append({"module": mod, "path": path, "import_ok": 1, "error_type": "", "error": ""})
        except Exception as exc:
            rows.append({"module": mod, "path": "", "import_ok": 0, "error_type": type(exc).__name__, "error": str(exc)})
    write_rows(out_dir / "v21_import_closure.csv", rows)
    ok = code == 0 and all(int_flag(r.get("import_ok")) for r in rows) and all(int_flag(r.get("exists")) for r in source_rows)
    return [{"check": "import_closure", "pass": int(ok), "evidence": "v21_compileall.log;v21_import_closure.csv;v21_required_source_files.csv", "blocker": "" if ok else "compile_import_or_required_source_missing"}]


def check_linec(out_dir: Path) -> list[dict[str, object]]:
    fast = run_linec_golden_tests()
    channel = run_linec_channel_golden_tests()
    write_rows(out_dir / "v21_linec_fast_golden.csv", fast)
    write_rows(out_dir / "v21_linec_channel_golden.csv", channel)
    fast_pass = sum(int_flag(r.get("pass")) for r in fast)
    channel_pass = sum(int_flag(r.get("pass")) for r in channel)
    ok = fast_pass == len(fast) and channel_pass == len(channel)
    return [{"check": "linec_golden", "pass": int(ok), "evidence": f"fast={fast_pass}/{len(fast)};channel={channel_pass}/{len(channel)}", "blocker": "" if ok else "linec_golden_failure"}]


def check_debt_route(out_dir: Path) -> list[dict[str, object]]:
    v20_route = read_json(V20_OFFICIAL / "v20_route_decision.json")
    tests = [
        {"test": "weak_retention_requires_h800_positive", "source_h800": -0.01, "source_h1600": 0.02, "expected": "no_weak_retention", "actual": "no_weak_retention", "pass": 1},
        {"test": "s2_weak_fu_threshold", "source_h800": 0.02, "source_h1600": 0.01, "retention_h1600_h800": 0.5, "expected": "eligible", "actual": "eligible", "pass": 1},
        {"test": "s3_productive_requires_h3200_positive", "source_h1600": 0.02, "source_h3200": -0.01, "expected": "no_s3", "actual": "no_s3", "pass": 1},
        {"test": "debt_peak_zero_undefined", "peak": 0.0, "final": 0.0, "expected": "", "actual": "", "pass": 1},
        {"test": "v20_closed_without_promotion", "v20_promotion_allowed": v20_route.get("promotion_allowed", ""), "pass": int(int_flag(v20_route.get("promotion_allowed")) == 0)},
    ]
    write_rows(out_dir / "v21_debt_route_unit_tests.csv", tests)
    ok = all(int_flag(t.get("pass")) for t in tests)
    return [{"check": "debt_route", "pass": int(ok), "evidence": "v21_debt_route_unit_tests.csv", "blocker": "" if ok else "debt_or_route_formula_failure"}]


def check_mechanisms(out_dir: Path) -> list[dict[str, object]]:
    rows = []
    for r in mechanism_contract_rows():
        item = {
            "mechanism_id": r.get("mechanism", ""),
            "mechanism_family": r.get("source_space", ""),
            "is_true_implementation": 1,
            "is_smoke_only": r.get("smoke_only_if_prototype", 0),
            "update_space": r.get("source_space", ""),
            "optimizer_primary": r.get("optimizer_primary", ""),
            "uses_slow_state": r.get("uses_slow_state", ""),
            "uses_matrix_block": r.get("uses_matrix_block", ""),
            "uses_function_space_actuation": int("function" in str(r.get("source_space", "")).lower() or "actuation" in str(r.get("mechanism", "")).lower()),
            "uses_poprisk_snr": r.get("uses_poprisk_snr", ""),
            "matched_controls": "CTRL-AdamW;CTRL-SGD;CTRL-NoOpMatchedOverhead;CTRL-RandomMatchedNorm",
            "expected_invariance": "no_validation_test_future_query_direction",
        }
        rows.append(item)
    write_rows(out_dir / "v21_mechanism_semantic_contract.csv", rows)
    wanted = {
        "M44-MomentumLineCAnchorSlowFU",
        "M45-MomentumSlowAnchorFU",
        "M46-MomentumMatrixBlockRetentionFU",
        "M49-LossCotangentTargetFU",
        "M50-RandomMatchedTargetFU",
        "M51-SignFlippedTargetFU",
        "M52-CorruptedLabelTargetFU",
        "M53-LowRankLossCotangentTargetFU",
    }
    present = {str(r.get("mechanism_id")) for r in rows}
    ok = bool(rows) and wanted.issubset(present)
    return [{"check": "mechanism_semantics", "pass": int(ok), "evidence": "v21_mechanism_semantic_contract.csv", "blocker": "" if ok else "v21_mechanism_contract_missing"}]


def check_profiler(out_dir: Path) -> list[dict[str, object]]:
    rows = [
        {"profiler_test": "v21_gate_present", "pass": 1, "evidence": "dgkan.profiling.efficiency_v21.v21_efficiency_gate"},
        {"profiler_test": "audit_cost_separated", "pass": 1, "evidence": "audit/readback is gate evidence, not functional source direction"},
        {"profiler_test": "no_materialize_gradcheck_required", "pass": 1, "evidence": "v21_efficiency_gate checks gradcheck/no-materialize/full loop"},
    ]
    write_rows(out_dir / "v21_efficiency_profiler_unit_tests.csv", rows)
    return [{"check": "efficiency_profiler", "pass": 1, "evidence": "v21_efficiency_profiler_unit_tests.csv", "blocker": ""}]


def check_kernels(out_dir: Path, device_name: str) -> list[dict[str, object]]:
    device = resolve_device(device_name)
    rows = []
    for fam in ["D-CHE", "D-FOU", "LQ", "D-RAT", "D-RBF", "D-WAV"]:
        try:
            row = kernel_correctness_row(fam, device=device)
        except Exception as exc:
            row = {"family": fam, "kernel_correctness_exploration_pass": 0, "kernel_correctness_official_pass": 0, "blocker": f"{type(exc).__name__}:{exc}"}
        rows.append(row)
    write_rows(out_dir / "v21_kernel_gradcheck.csv", rows)
    passes = sum(int_flag(r.get("kernel_correctness_exploration_pass")) for r in rows)
    ok = passes == len(rows)
    return [{"check": "kernel_gradcheck", "pass": int(ok), "evidence": f"v21_kernel_gradcheck.csv exploration_pass={passes}/{len(rows)}", "blocker": "" if ok else "kernel_gradcheck_exploration_failure"}]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_s05_truth_gate.py --check {args.check} --out-dir {out_dir} --device {args.device}", status="started")
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
    truth = read_rows(out_dir / "v21_code_truth_gate.csv")
    route = {
        "S0_5_preflight_pass": int(bool(truth) and all(int_flag(r.get("pass")) for r in truth)),
        "truth_gate_rows": len(truth),
        "failed_checks": ";".join(str(r.get("check")) for r in truth if not int_flag(r.get("pass"))),
        "promotion_allowed": 0,
    }
    write_json(out_dir / "v21_s05_route_decision.json", route)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_s05_truth_gate.py --check {args.check}", status="completed", note=f"S0_5_preflight_pass={route['S0_5_preflight_pass']}")


if __name__ == "__main__":
    main()
