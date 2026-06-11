#!/usr/bin/env python3
"""v22.09 basis efficiency closure and D-RAT/D-RBF robust officialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_07_drat_drbf_multibatch as v2207  # noqa: E402
from experiments.run_v22_09_common import (  # noqa: E402
    PYTHON,
    V2204_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--v2204-dir", default=str(V2204_OFFICIAL))
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--official-transition-batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--data-root", default="data")
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--profiler-repeats", type=int, default=2)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--repair-iters", type=int, default=12)
    p.add_argument("--repair-warmup", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.003)
    return p


def _eff_pass(row: dict[str, Any]) -> tuple[int, str]:
    blockers = []
    if finite_float(row.get("forward_ratio_vs_mlp"), 999.0) > 1.25:
        blockers.append("forward_ratio")
    if finite_float(row.get("step_ratio_vs_mlp"), 999.0) > 1.25:
        blockers.append("step_ratio")
    if finite_float(row.get("memory_ratio_vs_mlp"), 999.0) > 1.05:
        blockers.append("memory_ratio")
    if not int_flag(row.get("same_kernel_functional_runner_proof")):
        blockers.append("same_kernel_functional_runner_proof_missing")
    if int_flag(row.get("fallback_kernel_used")):
        blockers.append("fallback_kernel_used")
    if not int_flag(row.get("official_fused_kernel_complete")):
        blockers.append("official_fused_kernel_incomplete")
    return int(not blockers), ";".join(blockers)


def _normalize_dche_dfou(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["variant_family"] = "CHE21-R2/R4" if row.get("carrier") == "D-CHE" else "FOU21-R3"
    out["forward_ratio_vs_mlp"] = row.get("best_forward_ratio", row.get("forward_ratio_vs_mlp", ""))
    out["backward_ratio_vs_mlp"] = row.get("best_backward_ratio", row.get("backward_ratio_vs_mlp", ""))
    out["step_ratio_vs_mlp"] = row.get("best_step_ratio", row.get("step_ratio_vs_mlp", ""))
    out["memory_ratio_vs_mlp"] = row.get("best_memory_ratio", row.get("memory_ratio_vs_mlp", ""))
    out["functional_direction_ms"] = row.get("functional_direction_ms", "")
    out["metric_solver_ms"] = row.get("metric_solver_ms", "")
    out["LineC_audit_ms"] = row.get("mean_linec_audit_ms", "")
    out["horizon_readback_ms"] = row.get("mean_horizon_readback_ms", "")
    out["same_kernel_functional_runner_proof"] = int_flag(row.get("v22_04_same_kernel_runner_proof")) or int_flag(row.get("v22_03_same_kernel_functional_runner_proof"))
    out["official_fused_kernel_complete"] = int_flag(row.get("full_loop_official_closure"))
    out["fallback_kernel_used"] = int_flag(row.get("fallback_kernel_used"))
    out["functional_runner_same_kernel"] = out["same_kernel_functional_runner_proof"]
    out["readback_source"] = "v22_04_measured_full_loop_summary"
    passed, blocker = _eff_pass(out)
    out["v22_09_S1_pass"] = passed
    out["v22_09_decision"] = "OfficialEfficientCarrier" if passed else "EfficiencyBlocked"
    out["v22_09_blocker"] = blocker
    return out


def _normalize_components(row: dict[str, Any]) -> None:
    if str(row.get("carrier")) == "D-RAT":
        row["numden_fused_ms"] = row.get("numden_fused_ms", row.get("numden_backward_ms", ""))
        row["safety_clamp_ms"] = row.get("safety_clamp_ms", row.get("safety_guard_ms", row.get("denominator_safety_ms", "")))
        row["derivative_telemetry_ms"] = row.get("derivative_telemetry_ms", row.get("telemetry_ms", ""))
        row["train_path_without_telemetry_ms"] = row.get("train_path_without_telemetry_ms", row.get("train_forward_without_telemetry_ms", ""))
        row.setdefault("component_sum_vs_total_error", "")
        keys = [
            "numerator_eval_ms",
            "denominator_eval_ms",
            "reciprocal_ms",
            "numden_fused_ms",
            "safety_clamp_ms",
            "derivative_telemetry_ms",
            "train_path_without_telemetry_ms",
        ]
    else:
        row["dense_materialization_bytes"] = row.get("dense_materialization_bytes", row.get("dense_materialized_bytes", row.get("basis_materialized_bytes", "")))
        row["no_dense_materialization_proof"] = row.get("no_dense_materialization_proof", int(not int_flag(row.get("dense_basis_materialized"))))
        row.setdefault("component_sum_vs_total_error", "")
        keys = [
            "active_center_fraction",
            "mean_local_K",
            "local_gather_ms",
            "exp_eval_ms",
            "local_backward_ms",
            "dense_materialization_bytes",
            "no_dense_materialization_proof",
        ]
    row["component_telemetry_complete"] = int(all(str(row.get(k, "")).strip() for k in keys))


def _classify_v2209(row: dict[str, Any]) -> dict[str, Any]:
    forward = finite_float(row.get("forward_ratio_vs_mlp"), 999.0)
    step = finite_float(row.get("step_ratio_vs_mlp"), 999.0)
    memory = finite_float(row.get("memory_ratio_vs_mlp"), 999.0)
    grad = int_flag(row.get("gradcheck_pass"))
    official = int_flag(row.get("official_fused_kernel_complete"))
    same_kernel = int_flag(row.get("functional_runner_kernel_match"))
    telemetry = int_flag(row.get("component_telemetry_complete"))
    near = int(forward <= 3.0 and step <= 2.0 and memory <= 1.2 and grad and same_kernel)
    robust = int(forward <= 1.25 and step <= 1.25 and memory <= 1.05 and grad and official and same_kernel and telemetry)
    blockers = []
    if forward > 1.25:
        blockers.append("forward_ratio")
    if step > 1.25:
        blockers.append("step_ratio")
    if memory > 1.05:
        blockers.append("memory_ratio")
    if not grad:
        blockers.append("gradcheck")
    if not official:
        blockers.append("official_fused_kernel_complete")
    if not same_kernel:
        blockers.append("functional_runner_kernel_match")
    if not telemetry:
        blockers.append("component_telemetry_incomplete")
    return {
        "near_E1_v22_09": near,
        "robust_production_pass": robust,
        "v22_09_official_blocker": ";".join(dict.fromkeys(blockers)),
    }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for carrier in sorted({str(r.get("carrier", "")) for r in rows}):
        group = [r for r in rows if str(r.get("carrier", "")) == carrier]
        robust_rows = sum(int_flag(r.get("robust_production_pass")) for r in group)
        near_rows = sum(int_flag(r.get("near_E1_v22_09")) for r in group)
        telemetry_rows = sum(int_flag(r.get("component_telemetry_complete")) for r in group)
        decision = "RobustProductionPass" if robust_rows >= 3 else ("NearE1LimitedSmokeOnly" if near_rows >= 3 else "OfficialFusedBlocked")
        blocker_parts: list[str] = []
        for item in [str(r.get("v22_09_official_blocker", "")) for r in group]:
            for part in item.split(";"):
                part = part.strip()
                if part and part not in blocker_parts:
                    blocker_parts.append(part)
        out.append(
            {
                "carrier": carrier,
                "profile_rows": len(group),
                "robust_production_pass_rows": robust_rows,
                "near_E1_rows": near_rows,
                "component_telemetry_complete_rows": telemetry_rows,
                "best_forward_ratio": min(finite_float(r.get("forward_ratio_vs_mlp"), 999.0) for r in group),
                "best_backward_ratio": min(finite_float(r.get("backward_ratio_vs_mlp"), 999.0) for r in group),
                "best_step_ratio": min(finite_float(r.get("step_ratio_vs_mlp"), 999.0) for r in group),
                "best_memory_ratio": min(finite_float(r.get("memory_ratio_vs_mlp"), 999.0) for r in group),
                "decision": decision,
                "blocker": ";".join(blocker_parts),
                "repair_direction_attempted_or_next": "D-RAT: telemetry-free train path and fused num/den reciprocal; D-RBF: active mask/no dense materialization/local backward fused",
            }
        )
    return out


def _limited_smoke_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in summary:
        if str(row.get("decision")) != "NearE1LimitedSmokeOnly":
            continue
        rows.append(
            {
                "carrier": row.get("carrier", ""),
                "scope": "MNIST seed0 h800/h1600 only",
                "promotion_allowed": 0,
                "limited_smoke_executed": 0,
                "status": "deferred_no_official_fused_functional_runner",
                "reason": "near_E1 allows smoke, but official fused kernel/full FU runner gate remains closed; no success is claimed",
            }
        )
    return rows


def _bridge_telemetry_value(carrier: str, key: str, repair: dict[str, Any]) -> Any:
    if carrier == "D-RAT":
        aliases = {
            "reciprocal_ms": "reciprocal_or_division_ms",
            "numden_fused_ms": "numden_backward_ms",
            "safety_clamp_ms": "denominator_safety_ms",
            "derivative_telemetry_ms": "derivative_telemetry_ms",
            "train_path_without_telemetry_ms": "train_forward_without_telemetry_ms",
        }
    else:
        aliases = {
            "mean_local_K": "mean_local_k",
            "dense_materialization_bytes": "basis_materialized_bytes",
            "no_dense_materialization_proof": "no_dense_materialization_proof",
        }
    return repair.get(key, repair.get(aliases.get(key, ""), ""))


def _telemetry_bridge_rows(official_rows: list[dict[str, Any]], repair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bridge: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for repair in repair_rows:
        if not int_flag(repair.get("component_telemetry_complete")):
            continue
        key = (str(repair.get("carrier", "")), str(repair.get("batch_size", "")))
        current = by_key.get(key)
        if current is None or finite_float(repair.get("forward_ratio_vs_mlp"), 999.0) < finite_float(current.get("forward_ratio_vs_mlp"), 999.0):
            by_key[key] = repair
    for row in official_rows:
        carrier = str(row.get("carrier", ""))
        item = dict(row)
        repair = by_key.get((carrier, str(row.get("batch_size", "")))) or by_key.get((carrier, "512"))
        if repair:
            if carrier == "D-RAT":
                for key in [
                    "numerator_eval_ms",
                    "denominator_eval_ms",
                    "reciprocal_ms",
                    "numden_fused_ms",
                    "safety_clamp_ms",
                    "derivative_telemetry_ms",
                    "train_path_without_telemetry_ms",
                ]:
                    item[key] = _bridge_telemetry_value(carrier, key, repair)
            if carrier == "D-RBF":
                for key in [
                    "active_center_fraction",
                    "mean_local_K",
                    "local_gather_ms",
                    "exp_eval_ms",
                    "local_backward_ms",
                    "dense_materialization_bytes",
                    "no_dense_materialization_proof",
                ]:
                    item[key] = _bridge_telemetry_value(carrier, key, repair)
            item["telemetry_bridge_repair_variant"] = repair.get("component_variant", "")
            item["telemetry_bridge_batch_size"] = repair.get("batch_size", "")
            item["telemetry_bridge_scope"] = "audit_only_official_runner_timing_plus_repair_component_telemetry"
        else:
            item["telemetry_bridge_scope"] = "no_matching_repair_telemetry"
        _normalize_components(item)
        item.update(_classify_v2209(item))
        item["official_closure_claimed"] = 0
        item["telemetry_bridge_claimed_official"] = 0
        bridge.append(item)
    return bridge


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)

    source = Path(args.v2204_dir)
    full_loop = read_rows(source / "v22_04_efficiency_full_loop_summary.csv")
    dche_dfou = [_normalize_dche_dfou(r) for r in full_loop if str(r.get("carrier", "")) in {"D-CHE", "D-FOU"}]

    device = v2207.v2205_repair.repair._device(args.device)
    drat_rows, drat_waterfall = v2207.v2205_repair._drat_official_transition_rows(args, device)
    drbf_rows, drbf_waterfall = v2207.v2205_repair._drbf_official_transition_rows(args, device)
    rows = drat_rows + drbf_rows
    for row in rows:
        v2207._augment_component_fields(row)
        _normalize_components(row)
        row.update(_classify_v2209(row))
        row["v22_09_multibatch_probe"] = 1
    summary = _summary(rows)
    batch_sizes = [int(x) for x in str(args.official_transition_batch_sizes).split(",") if x.strip()]
    needs_repair = any(str(r.get("decision")) != "RobustProductionPass" for r in summary)
    repair_rows: list[dict[str, Any]] = []
    repair_summary: list[dict[str, Any]] = []
    if needs_repair:
        repair_rows, repair_summary = v2207._attempt_repair(args, device, batch_sizes)
    v2207._merge_repair_summary(summary, repair_summary)
    smoke = _limited_smoke_rows(summary)
    bridge_rows = _telemetry_bridge_rows(rows, repair_rows)
    bridge_summary = _summary(bridge_rows)

    route = {
        "D-CHE_S1_pass": int(any(r.get("carrier") == "D-CHE" and int_flag(r.get("v22_09_S1_pass")) for r in dche_dfou)),
        "D-FOU_S1_pass": int(any(r.get("carrier") == "D-FOU" and int_flag(r.get("v22_09_S1_pass")) for r in dche_dfou)),
        "D-RAT_robust_pass": int(any(r.get("carrier") == "D-RAT" and str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "D-RBF_robust_pass": int(any(r.get("carrier") == "D-RBF" and str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "D-RAT_D-RBF_multibatch_closed": int(all(str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "limited_smoke_allowed_rows": len(smoke),
        "repair_attempted": int(bool(repair_summary)),
        "repair_attempt_rows": sum(int_flag(r.get("repair_attempt_rows")) for r in repair_summary),
        "repair_official_closure_claimed": 0,
    }

    write_rows(out_dir / "v22_09_efficiency_full_loop_reconfirm.csv", dche_dfou)
    write_rows(out_dir / "v22_09_drat_drbf_multibatch_officialization.csv", rows)
    write_rows(out_dir / "v22_09_drat_component_waterfall.csv", drat_waterfall)
    write_rows(out_dir / "v22_09_drbf_component_waterfall.csv", drbf_waterfall)
    write_rows(out_dir / "v22_09_drat_drbf_repair_attempts.csv", repair_rows)
    write_rows(out_dir / "v22_09_drat_drbf_repair_attempts_summary.csv", repair_summary)
    write_rows(out_dir / "v22_09_drat_drbf_limited_functional_smoke.csv", smoke)
    write_rows(out_dir / "v22_09_drat_drbf_telemetry_bridge.csv", bridge_rows)
    write_rows(out_dir / "v22_09_drat_drbf_telemetry_bridge_summary.csv", bridge_summary)
    write_rows(out_dir / "v22_09_drat_drbf_multibatch_summary.csv", summary)
    write_json(out_dir / "v22_09_basis_efficiency_route.json", route)
    simple_svg(out_dir / "figures/v22_09_basis_efficiency_dashboard.svg", "v22.09 basis efficiency", dche_dfou + rows, "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_09_D-RAT_component_waterfall.svg", "v22.09 D-RAT component waterfall", [r for r in rows if r.get("carrier") == "D-RAT"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/v22_09_D-RBF_component_waterfall.svg", "v22.09 D-RBF component waterfall", [r for r in rows if r.get("carrier") == "D-RBF"], "forward_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_basis_efficiency_closure.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --out-dir {out_dir}",
        status="completed",
        note=f"D-CHE={route['D-CHE_S1_pass']} D-FOU={route['D-FOU_S1_pass']} DRAT_DRBF_closed={route['D-RAT_D-RBF_multibatch_closed']} repair_rows={len(repair_rows)} smoke_deferred_rows={len(smoke)}",
    )


if __name__ == "__main__":
    main()
