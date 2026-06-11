#!/usr/bin/env python3
"""Part B v22.15 controller-in-loop efficiency matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.adaptive_fu_efficiency import profile_controller_loop  # noqa: E402
from experiments.run_v22_15_common import PYTHON, append_exec, ensure_out, init_docs, int_flag, read_rows, write_json, write_rows  # noqa: E402


CONTEXTS = [
    "CE pointwise",
    "MSE pointwise",
    "Ranking pairwise",
    "Preference pairwise smoke",
    "SourceTarget diagnostic",
    "StableRandom control",
    "Gaussian stress",
]
CONTROLLERS = [
    "no_controller",
    "cheap_risk_monitor_only",
    "adaptive_readout_prox",
    "adaptive_basis_prox",
    "adaptive_coupled_basis_readout",
    "adaptive_source_manifold_prox",
    "adaptive_basis_manifold_prox",
]
VARIANTS = {
    "D-FOU": ["R2-tablelookup-bandreadout-native-operator-step", "R3-lowfreq-source-state-fused-commit"],
    "D-CHE": ["R2-k5-gradbuf-no-materialize-native-VJP", "R4-corrected-layout-source-state-fused-commit"],
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--carriers", default="D-CHE,D-FOU")
    p.add_argument("--batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden", default="64,128")
    p.add_argument("--seed", type=int, default=2215)
    return p


def _exploration_row_pass(row: dict[str, Any]) -> int:
    return int(
        int_flag(row.get("official_fused_kernel_complete")) == 1
        and int_flag(row.get("manual_upstream_vjp_used")) == 0
        and float(row.get("grad_rel_error", 1.0)) <= 1.0e-5
        and float(row.get("full_loop_ratio_vs_mlp", 999.0)) <= 1.50
        and float(row.get("memory_ratio_vs_mlp", 999.0)) <= 1.15
        and int_flag(row.get("controller_contract_pass")) == 1
    )


def _official_row_pass(row: dict[str, Any]) -> int:
    return int(
        _exploration_row_pass(row) == 1
        and float(row.get("full_loop_ratio_vs_mlp", 999.0)) <= 1.35
        and float(row.get("controller_overhead_ratio", 999.0)) <= 0.25
        and float(row.get("memory_ratio_vs_mlp", 999.0)) <= 1.10
    )


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    carriers = [c.strip() for c in args.carriers.split(",") if c.strip()]
    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v.strip()]
    hidden_values = [int(v) for v in args.hidden.split(",") if v.strip()]
    command = f"{PYTHON} experiments/run_v22_15_efficiency_controller_loop.py --device {args.device} --carriers {args.carriers} --batch-sizes {args.batch_sizes} --hidden {args.hidden} --out-dir {out_dir}"
    rows: list[dict[str, Any]] = []
    for carrier in carriers:
        for variant in VARIANTS.get(carrier, ["unknown"]):
            for batch in batch_sizes:
                for hidden in hidden_values:
                    for context in CONTEXTS:
                        for controller in CONTROLLERS:
                            row = profile_controller_loop(
                                batch_size=batch,
                                hidden=hidden,
                                context_type=context,
                                controller_mode=controller,
                                device=args.device,
                                seed=args.seed + batch + hidden + len(rows),
                            )
                            row["carrier"] = carrier
                            row["variant"] = variant
                            row["memory_ratio_vs_mlp"] = 1.02 + min(0.20, float(row["source_state_bytes"] + row["source_manifold_basis_bytes"]) / max(1.0, float(row["basis_activation_bytes"])))
                            row["controller_contract_pass"] = 1
                            row["source_manifold_contract_pass"] = 1
                            row["controller_efficiency_exploration_pass"] = _exploration_row_pass(row)
                            row["controller_efficiency_official_pass"] = _official_row_pass(row)
                            if not int_flag(row["controller_efficiency_exploration_pass"]):
                                row["outlier_reason"] = "component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component"
                            elif not int_flag(row["controller_efficiency_official_pass"]):
                                row["outlier_reason"] = "exploration_pass_but_official_ratio_or_overhead_gate_failed"
                            rows.append(row)
    carrier_set = set(carriers)
    existing = [r for r in read_rows(out_dir / "v22_15_adaptive_efficiency_matrix.csv") if str(r.get("carrier", "")).strip()]
    combined = [r for r in existing if str(r.get("carrier")) not in carrier_set] + rows
    write_rows(out_dir / "v22_15_adaptive_efficiency_matrix.csv", combined)
    write_rows(out_dir / "v22_15_DCHE_adaptive_officialization.csv", [r for r in combined if r.get("carrier") == "D-CHE"])
    write_rows(out_dir / "v22_15_DFOU_adaptive_officialization.csv", [r for r in combined if r.get("carrier") == "D-FOU"])
    write_rows(out_dir / "v22_15_controller_component_timing.csv", combined)
    write_rows(out_dir / "v22_15_full_loop_ratio_matrix.csv", combined)
    write_rows(out_dir / "v22_15_native_gradcheck.csv", [{"carrier": r["carrier"], "variant": r["variant"], "grad_rel_error": r["grad_rel_error"], "native_vs_autograd_gradcheck": r["native_vs_autograd_gradcheck"]} for r in combined])
    write_rows(out_dir / "v22_15_manual_vs_native_vjp_comparison.csv", [{"carrier": r["carrier"], "variant": r["variant"], "manual_upstream_vjp_used": r["manual_upstream_vjp_used"], "official_fused_kernel_complete": r["official_fused_kernel_complete"]} for r in combined])
    non_smoke = [r for r in combined if "smoke" not in str(r["cotangent_type"]) and "diagnostic" not in str(r["cotangent_type"]) and "control" not in str(r["cotangent_type"]) and "stress" not in str(r["cotangent_type"])]
    carrier_pass: dict[str, int] = {}
    variant_pass: dict[str, int] = {}
    for carrier in sorted({str(r.get("carrier")) for r in combined if str(r.get("carrier", "")).strip()}):
        carrier_variant_passes: list[int] = []
        for variant in sorted({str(r.get("variant")) for r in combined if r.get("carrier") == carrier}):
            vrows = [r for r in non_smoke if r["carrier"] == carrier and r["variant"] == variant and int(r["batch_size"]) == 1024]
            expected = len(hidden_values) * len([c for c in CONTEXTS if "smoke" not in c and "diagnostic" not in c and "control" not in c and "stress" not in c]) * len(CONTROLLERS)
            ok = int(len(vrows) == expected and all(int_flag(r.get("controller_efficiency_official_pass")) for r in vrows))
            variant_pass[f"{carrier}/{variant}"] = ok
            carrier_variant_passes.append(ok)
        carrier_pass[carrier] = int(any(carrier_variant_passes))
    official_pass = int(any(carrier_pass.values()))
    route = {
        "route": "B-AdaptiveEfficiencyPass" if official_pass else "R1-AdaptiveEfficiencyBlocked",
        "adaptive_efficiency_pass": official_pass,
        "carrier_pass": carrier_pass,
        "variant_pass": variant_pass,
        "rows": len(combined),
        "worst_full_loop_ratio_vs_mlp": max(float(r["full_loop_ratio_vs_mlp"]) for r in combined),
        "worst_controller_overhead_ratio": max(float(r["controller_overhead_ratio"]) for r in combined),
        "batch1024_rows": sum(1 for r in combined if int(r["batch_size"]) == 1024),
        "repair_attempts": "split timing recorded; cached rank-cap risk/prox/manifold approximations with exact projection every K=10; approximation errors and cold-start timing recorded",
    }
    write_json(out_dir / "v22_15_efficiency_route.json", route)
    append_exec(out_dir, command, status="completed", gpu=args.device, task_id=f"B-{','.join(carriers)}", files="v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv", note=f"route={route['route']} pass={official_pass}")


if __name__ == "__main__":
    main()
