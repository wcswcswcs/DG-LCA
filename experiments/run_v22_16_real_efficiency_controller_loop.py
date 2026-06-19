#!/usr/bin/env python3
"""Part H v22.16 real controller-in-loop efficiency matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.profiling.real_adaptive_fu_efficiency import real_controller_timing  # noqa: E402
from experiments.run_v22_16_common import (  # noqa: E402
    PYTHON,
    append_exec,
    count_parameters,
    device_from_arg,
    ensure_out,
    finite_float,
    init_docs,
    int_flag,
    loader_for,
    make_model,
    read_rows,
    write_json,
    write_rows,
)


CONTEXTS = ["CE pointwise", "MSE pointwise", "Ranking pairwise", "StableRandom"]
CONTROLLERS = ["no_controller", "cheap_risk_only", "direct_predictive_adaptive", "source_manifold_adaptive", "basis_native_adaptive", "coupled_basis_readout_adaptive"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--carriers", default="D-CHE,D-FOU")
    p.add_argument("--batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden", default="64,128")
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--download", action="store_true")
    return p


def _row_pass(row: dict[str, Any]) -> int:
    return int(
        finite_float(row.get("full_loop_ratio_vs_mlp"), 999.0) <= 1.50
        and finite_float(row.get("controller_overhead_ratio"), 999.0) <= 0.35
        and finite_float(row.get("memory_ratio_vs_mlp"), 999.0) <= 1.15
        and finite_float(row.get("grad_rel_error"), 0.0) <= 1.0e-3
        and int_flag(row.get("manual_upstream_vjp_used")) == 0
    )


def _official_pass(row: dict[str, Any]) -> int:
    return int(
        _row_pass(row)
        and finite_float(row.get("full_loop_ratio_vs_mlp"), 999.0) <= 1.35
        and finite_float(row.get("controller_overhead_ratio"), 999.0) <= 0.25
        and finite_float(row.get("memory_ratio_vs_mlp"), 999.0) <= 1.10
        and int(row.get("batch_size", 0)) == 1024
    )


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = (
        f"{PYTHON} experiments/run_v22_16_real_efficiency_controller_loop.py --device {args.device} --dataset {args.dataset} "
        f"--seed {args.seed} --carriers {args.carriers} --batch-sizes {args.batch_sizes} --hidden {args.hidden} "
        f"--train-size {args.train_size} --out-dir {out_dir}"
        + (" --download" if args.download else "")
    )
    device = device_from_arg(args.device)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    carriers = [c.strip() for c in str(args.carriers).split(",") if c.strip()]
    batch_sizes = [int(b) for b in str(args.batch_sizes).split(",") if b.strip()]
    hidden_values = [int(h) for h in str(args.hidden).split(",") if h.strip()]
    for batch in batch_sizes:
        loader = loader_for(args.dataset, True, max(args.train_size, batch), batch, args.seed, download=args.download, shuffle=False)
        x, y = next(iter(loader))
        x = x.to(device).float()
        y = y.to(device).long()
        for hidden in hidden_values:
            try:
                mlp = make_model("MLP+AdamW", x, hidden, args.seed + hidden + batch, device).to(device)
                mlp_timing = real_controller_timing(mlp, x, y, controller_mode="no_controller", context_type="CE pointwise", selector="all")
                mlp_full = max(1.0e-9, float(mlp_timing["full_step_ms"]))
                mlp_mem = max(1.0e-9, float(mlp_timing["peak_memory_mb"]))
                rows.append({"carrier": "MLP", "variant": "same-param-reference", "batch_size": batch, "hidden": hidden, "cotangent_type": "CE pointwise", "controller_mode": "no_controller", "param_count": count_parameters(mlp), "full_loop_ratio_vs_mlp": 1.0, "controller_overhead_ratio": 0.0, "memory_ratio_vs_mlp": 1.0, **mlp_timing})
            except Exception as exc:
                blockers.append(f"MLP:{batch}:{hidden}:{repr(exc)}")
                continue
            for carrier in carriers:
                for context in CONTEXTS:
                    for controller in CONTROLLERS:
                        try:
                            variant = "KAN+AdaptiveFU-basis-official" if "basis" in controller else "KAN+AdamW"
                            kan = make_model(variant, x, hidden, args.seed + hidden + batch + len(rows), device, carrier=carrier).to(device)
                            selector = "basis" if "basis" in controller else ("all" if "coupled" in controller else "readout" if "direct" in controller else "all")
                            timing = real_controller_timing(kan, x, y, controller_mode=controller, context_type=context, selector=selector)
                            no_controller_ms = max(1.0e-9, float(timing["forward_ms"]) + float(timing["cotangent_ms"]) + float(timing["backward_from_grad_logits_ms"]) + float(timing["optimizer_update_ms"]))
                            overhead = max(0.0, float(timing["full_step_ms"]) - no_controller_ms) / max(1.0e-9, no_controller_ms)
                            row = {
                                "carrier": carrier,
                                "variant": f"{carrier}-real-PrimitiveKAN",
                                "batch_size": batch,
                                "hidden": hidden,
                                "cotangent_type": context,
                                "controller_mode": controller,
                                "param_count": count_parameters(kan),
                                "full_loop_ratio_vs_mlp": float(timing["full_step_ms"]) / mlp_full,
                                "controller_overhead_ratio": overhead,
                                "memory_ratio_vs_mlp": float(timing["peak_memory_mb"]) / mlp_mem if mlp_mem > 0 else "",
                                "basis_activation_bytes": "",
                                "source_history_bytes": "",
                                **timing,
                            }
                            row["controller_efficiency_exploration_pass"] = _row_pass(row)
                            row["controller_efficiency_official_pass"] = _official_pass(row)
                            if not int_flag(row["controller_efficiency_exploration_pass"]):
                                row["outlier_reason"] = row.get("outlier_reason") or "real_full_loop_timing_or_grad_gate_failed"
                            rows.append(row)
                        except Exception as exc:
                            blockers.append(f"{carrier}:{context}:{controller}:b{batch}:h{hidden}:{repr(exc)}")
                            rows.append({"carrier": carrier, "batch_size": batch, "hidden": hidden, "cotangent_type": context, "controller_mode": controller, "status": "blocked", "blocker": repr(exc)})
    existing = [r for r in read_rows(out_dir / "v22_16_real_adaptive_efficiency_matrix.csv") if str(r.get("carrier")) not in set(carriers)]
    combined = existing + rows if existing else rows
    write_rows(out_dir / "v22_16_real_adaptive_efficiency_matrix.csv", combined)
    write_rows(out_dir / "v22_16_real_controller_component_timing.csv", combined)
    write_rows(out_dir / "v22_16_real_full_loop_ratio_matrix.csv", combined)
    write_rows(out_dir / "v22_16_native_gradcheck.csv", [{"carrier": r.get("carrier"), "variant": r.get("variant"), "batch_size": r.get("batch_size"), "hidden": r.get("hidden"), "grad_rel_error": r.get("grad_rel_error"), "native_vs_autograd_gradcheck": r.get("native_vs_autograd_gradcheck")} for r in combined])
    write_rows(out_dir / "v22_16_manual_vs_native_vjp_comparison.csv", [{"carrier": r.get("carrier"), "variant": r.get("variant"), "manual_upstream_vjp_used": r.get("manual_upstream_vjp_used"), "official_fused_kernel_complete": r.get("official_fused_kernel_complete"), "cotangent_type": r.get("cotangent_type")} for r in combined])
    non_mlp = [r for r in combined if r.get("carrier") != "MLP" and "blocked" not in str(r.get("status", ""))]
    official_rows = [r for r in non_mlp if int_flag(r.get("controller_efficiency_official_pass"))]
    exploration_rows = [r for r in non_mlp if int_flag(r.get("controller_efficiency_exploration_pass"))]
    exploration_pass = int(bool(exploration_rows) and not blockers)
    official_pass = int(bool(official_rows) and not blockers)
    route = {
        "route": "H-RealControllerEfficiencyPass" if exploration_pass else "R9-KANBasisExplorationOpened_EfficiencyOrTaskPending",
        "real_adaptive_efficiency_exploration_pass": exploration_pass,
        "real_adaptive_efficiency_official_pass": official_pass,
        "rows": len(combined),
        "official_rows": len(official_rows),
        "batch1024_rows": sum(1 for r in combined if str(r.get("batch_size")) == "1024"),
        "worst_full_loop_ratio_vs_mlp": max((finite_float(r.get("full_loop_ratio_vs_mlp"), 0.0) for r in non_mlp), default=0.0),
        "worst_controller_overhead_ratio": max((finite_float(r.get("controller_overhead_ratio"), 0.0) for r in non_mlp), default=0.0),
        "blocker": ";".join(blockers[:20]),
        "repair_attempts": "real model batch timing; batch 1024 included if requested; component timings retained; no synthetic timing counted as official",
    }
    write_json(out_dir / "v22_16_efficiency_route.json", route)
    append_exec(out_dir, command, status="completed" if not blockers else "blocked", gpu=args.device, task_id=f"H-efficiency-{args.carriers}", files="v22_16_real_adaptive_efficiency_matrix.csv; v22_16_real_controller_component_timing.csv", note=f"route={route['route']} exploration={exploration_pass} official={official_pass} blockers={len(blockers)}")


if __name__ == "__main__":
    main()
