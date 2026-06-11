#!/usr/bin/env python3
"""v22.07 D-RAT/D-RBF multibatch officialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_03_drat_drbf_repair as active_repair  # noqa: E402
from experiments import run_v22_05_drat_drbf_repair as v2205_repair  # noqa: E402
from experiments.run_v22_07_common import PYTHON, append_exec, ensure_out, finite_float, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
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


def _f(value: Any, default: float = 999.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def _classify(row: dict[str, Any]) -> dict[str, Any]:
    forward = _f(row.get("forward_ratio_vs_mlp"))
    step = _f(row.get("step_ratio_vs_mlp"))
    memory = _f(row.get("memory_ratio_vs_mlp"), 99.0)
    grad = int(_f(row.get("gradcheck_pass"), 0.0))
    official = int(_f(row.get("official_fused_kernel_complete"), 0.0))
    same_kernel = int(_f(row.get("functional_runner_kernel_match"), 0.0))
    den_safe = int(_f(row.get("den_safety_pass", 1.0), 1.0))
    near = int(forward <= 3.0 and step <= 2.0 and memory <= 1.2 and grad and same_kernel)
    robust = int(forward <= 1.50 and step <= 1.50 and memory <= 1.10 and grad and official and same_kernel and den_safe)
    blockers = []
    if forward > 1.50:
        blockers.append("forward_ratio")
    if step > 1.50:
        blockers.append("step_ratio")
    if memory > 1.10:
        blockers.append("memory_ratio")
    if not grad:
        blockers.append("gradcheck")
    if not official:
        blockers.append("official_fused_kernel_complete")
    if not same_kernel:
        blockers.append("same_kernel_runner_proof")
    if not den_safe:
        blockers.append("denominator_safety")
    return {
        "near_E1_v22_07": near,
        "robust_production_pass": robust,
        "v22_07_official_blocker": ";".join(blockers),
    }


def _augment_component_fields(row: dict[str, Any]) -> None:
    if str(row.get("carrier")) == "D-RAT":
        row.setdefault("numerator_eval_ms", "")
        row.setdefault("denominator_eval_ms", "")
        row.setdefault("reciprocal_ms", "")
        row.setdefault("safety_guard_ms", "")
        row.setdefault("telemetry_ms", "")
        row.setdefault("rational_den_min", "")
        row.setdefault("rational_den_p01", "")
        den_min = finite_float(row.get("rational_den_min"), float("nan"))
        den_p01 = finite_float(row.get("rational_den_p01"), float("nan"))
        row["den_safety_pass"] = int((den_min != den_min or den_min > 1.0e-6) and (den_p01 != den_p01 or den_p01 > 1.0e-6))
        row["component_telemetry_complete"] = int(all(str(row.get(k, "")).strip() for k in ["numerator_eval_ms", "denominator_eval_ms", "reciprocal_ms", "safety_guard_ms"]))
    else:
        row.setdefault("active_center_fraction", "")
        row.setdefault("mean_local_K", "")
        row.setdefault("local_gather_ms", "")
        row.setdefault("exp_eval_ms", "")
        row.setdefault("readout_matmul_ms", "")
        row.setdefault("local_backward_ms", "")
        row.setdefault("dense_materialized_bytes", "")
        row["den_safety_pass"] = 1
        row["component_telemetry_complete"] = int(all(str(row.get(k, "")).strip() for k in ["active_center_fraction", "mean_local_K", "local_gather_ms", "exp_eval_ms", "readout_matmul_ms", "local_backward_ms", "dense_materialized_bytes"]))


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for carrier in sorted({str(r.get("carrier", "")) for r in rows}):
        group = [r for r in rows if str(r.get("carrier", "")) == carrier]
        robust_rows = sum(int(r.get("robust_production_pass", 0)) for r in group)
        near_rows = sum(int(r.get("near_E1_v22_07", 0)) for r in group)
        telemetry_rows = sum(int(r.get("component_telemetry_complete", 0)) for r in group)
        decision = "RobustProductionPass" if robust_rows >= 3 else ("NearE1LimitedSmokeAllowed" if near_rows >= 3 else "OfficialFusedBlocked")
        blockers = sorted({str(r.get("v22_07_official_blocker", "")) for r in group if str(r.get("v22_07_official_blocker", "")).strip()})
        if telemetry_rows < len(group):
            blockers.append("component_telemetry_incomplete")
        repair_direction = ""
        if carrier == "D-RAT" and decision != "RobustProductionPass":
            repair_direction = "separate numerator/denominator/reciprocal waterfall; telemetry-free training path; reciprocal approximation; denominator safety separated from timing"
        if carrier == "D-RBF" and decision != "RobustProductionPass":
            repair_direction = "active-center threshold repair; local K reduction; exp approximation; no dense materialization proof; local backward fused proof"
        out.append(
            {
                "carrier": carrier,
                "profile_rows": len(group),
                "robust_production_pass_rows": robust_rows,
                "near_E1_rows": near_rows,
                "component_telemetry_complete_rows": telemetry_rows,
                "best_forward_ratio": min(_f(r.get("forward_ratio_vs_mlp")) for r in group),
                "best_backward_ratio": min(_f(r.get("backward_ratio_vs_mlp")) for r in group),
                "best_step_ratio": min(_f(r.get("step_ratio_vs_mlp")) for r in group),
                "best_memory_ratio": min(_f(r.get("memory_ratio_vs_mlp")) for r in group),
                "decision": decision,
                "blocker": ";".join(blockers),
                "repair_direction_attempted_or_next": repair_direction,
            }
        )
    return out


def _repair_direction(carrier: str) -> str:
    if carrier == "D-RAT":
        return "separate numerator/denominator/reciprocal micro waterfall; telemetry-free training path; reciprocal approximation; denominator safety separated from training timing"
    if carrier == "D-RBF":
        return "active-center threshold repair; local K reduction; exp approximation; no dense materialization proof; local backward fused proof"
    return ""


def _repair_component_complete(row: dict[str, Any]) -> int:
    if str(row.get("carrier")) == "D-RAT":
        keys = [
            "numerator_eval_ms",
            "denominator_eval_ms",
            "reciprocal_or_division_ms",
            "denominator_safety_ms",
            "derivative_telemetry_ms",
            "train_forward_without_telemetry_ms",
            "numden_backward_ms",
            "readout_contraction_ms",
        ]
    else:
        keys = [
            "active_center_fraction",
            "mean_local_k",
            "local_gather_ms",
            "exp_eval_ms",
            "local_backward_ms",
            "readout_contraction_ms",
            "basis_materialized_bytes",
        ]
    return int(all(str(row.get(k, "")).strip() for k in keys))


def _normalize_repair_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    carrier = str(out.get("carrier", ""))
    out["v22_07_repair_attempt"] = 1
    out["repair_attempt_scope"] = "component_microkernel_not_official_fused_training_loop"
    out["repair_direction_attempted"] = _repair_direction(carrier)
    out["official_closure_claimed"] = 0
    out["same_kernel_runner_proof"] = out.get("functional_runner_kernel_match", 0)
    if carrier == "D-RAT":
        out.setdefault("reciprocal_ms", out.get("reciprocal_or_division_ms", ""))
        out.setdefault("safety_guard_ms", out.get("denominator_safety_ms", ""))
        out.setdefault("telemetry_ms", out.get("derivative_telemetry_ms", ""))
        out.setdefault("rational_den_min", out.get("denominator_min", ""))
        out.setdefault("rational_den_p01", out.get("denominator_p01", ""))
    if carrier == "D-RBF":
        out.setdefault("mean_local_K", out.get("mean_local_k", ""))
        out.setdefault("readout_matmul_ms", out.get("readout_contraction_ms", ""))
        out.setdefault("dense_materialized_bytes", out.get("basis_materialized_bytes", ""))
        out["no_dense_materialization_proof"] = int(not int(_f(out.get("dense_basis_materialized"), 0)))
    out["component_telemetry_complete"] = _repair_component_complete(out)
    return out


def _is_reference_variant(row: dict[str, Any]) -> bool:
    return "-R0-current-reference" in str(row.get("component_variant", ""))


def _best_repair_row(group: list[dict[str, Any]]) -> dict[str, Any]:
    if not group:
        return {}
    candidates = [r for r in group if not _is_reference_variant(r)] or group
    return sorted(
        candidates,
        key=lambda r: (
            int(_f(r.get("micro_near_E1"), 0)),
            -_f(r.get("forward_ratio_vs_mlp")),
            -_f(r.get("step_ratio_vs_mlp")),
            -_f(r.get("memory_ratio_vs_mlp")),
        ),
        reverse=True,
    )[0]


def _repair_summary(rows: list[dict[str, Any]], error: str = "") -> list[dict[str, Any]]:
    if error:
        return [
            {
                "carrier": "D-RAT/D-RBF",
                "repair_attempt_rows": 0,
                "repair_micro_near_E1_rows": 0,
                "repair_nonreference_micro_near_E1_rows": 0,
                "repair_component_telemetry_complete_rows": 0,
                "repair_best_forward_ratio": "",
                "repair_best_step_ratio": "",
                "repair_best_memory_ratio": "",
                "repair_best_variant": "",
                "repair_attempt_status": "RepairAttemptErrored",
                "repair_blocker": error,
                "official_closure_claimed": 0,
            }
        ]
    out: list[dict[str, Any]] = []
    for carrier in sorted({str(r.get("carrier", "")) for r in rows}):
        group = [r for r in rows if str(r.get("carrier", "")) == carrier]
        best = _best_repair_row(group)
        micro_near = sum(int(_f(r.get("micro_near_E1"), 0)) for r in group)
        nonref_micro_near = sum(int(_f(r.get("micro_near_E1"), 0)) for r in group if not _is_reference_variant(r))
        telemetry = sum(int(_f(r.get("component_telemetry_complete"), 0)) for r in group)
        blockers = ["official_fused_missing", "functional_runner_kernel_mismatch"]
        if not micro_near:
            blockers.insert(0, "micro_near_E1_not_reached")
        if telemetry < len(group):
            blockers.append("component_telemetry_incomplete")
        status = "MicroNearE1RunnerBlocked" if micro_near else "RepairAttemptDidNotReachMicroNearE1"
        out.append(
            {
                "carrier": carrier,
                "repair_attempt_rows": len(group),
                "repair_micro_near_E1_rows": micro_near,
                "repair_nonreference_micro_near_E1_rows": nonref_micro_near,
                "repair_component_telemetry_complete_rows": telemetry,
                "repair_best_forward_ratio": best.get("forward_ratio_vs_mlp", ""),
                "repair_best_backward_ratio": best.get("backward_ratio_vs_mlp", ""),
                "repair_best_step_ratio": best.get("step_ratio_vs_mlp", ""),
                "repair_best_memory_ratio": best.get("memory_ratio_vs_mlp", ""),
                "repair_best_variant": best.get("component_variant", ""),
                "repair_best_batch_size": best.get("batch_size", ""),
                "repair_direction_attempted": _repair_direction(carrier),
                "repair_attempt_status": status,
                "repair_blocker": ";".join(dict.fromkeys(blockers)),
                "official_closure_claimed": 0,
            }
        )
    return out


def _attempt_repair(args: argparse.Namespace, device: Any, batch_sizes: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        rows = active_repair._rat_rows(device, batch_sizes, args.hidden, args.repair_warmup, args.repair_iters)
        rows.extend(active_repair._rbf_rows(device, batch_sizes, args.hidden, args.repair_warmup, args.repair_iters))
    except Exception as exc:  # pragma: no cover - artifact records runtime blocker.
        return [], _repair_summary([], error=repr(exc))
    normalized = [_normalize_repair_row(r) for r in rows]
    return normalized, _repair_summary(normalized)


def _merge_repair_summary(summary: list[dict[str, Any]], repair_summary: list[dict[str, Any]]) -> None:
    by_carrier = {str(r.get("carrier", "")): r for r in repair_summary}
    for row in summary:
        repair = by_carrier.get(str(row.get("carrier", "")), {})
        for key in [
            "repair_attempt_rows",
            "repair_micro_near_E1_rows",
            "repair_nonreference_micro_near_E1_rows",
            "repair_component_telemetry_complete_rows",
            "repair_best_forward_ratio",
            "repair_best_backward_ratio",
            "repair_best_step_ratio",
            "repair_best_memory_ratio",
            "repair_best_variant",
            "repair_best_batch_size",
            "repair_attempt_status",
            "repair_blocker",
            "official_closure_claimed",
        ]:
            row[key] = repair.get(key, "")
        if repair:
            row["repair_direction_attempted_or_next"] = repair.get("repair_direction_attempted", row.get("repair_direction_attempted_or_next", ""))


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = v2205_repair.repair._device(args.device)
    drat_rows, drat_waterfall = v2205_repair._drat_official_transition_rows(args, device)
    drbf_rows, drbf_waterfall = v2205_repair._drbf_official_transition_rows(args, device)
    rows = drat_rows + drbf_rows
    for row in rows:
        _augment_component_fields(row)
        row.update(_classify(row))
        row["v22_07_multibatch_probe"] = 1
    summary = _summary(rows)
    batch_sizes = [int(x) for x in str(args.official_transition_batch_sizes).split(",") if x.strip()]
    needs_repair = any(
        str(r.get("decision")) != "RobustProductionPass"
        or int(r.get("component_telemetry_complete_rows", 0)) < int(r.get("profile_rows", 0))
        for r in summary
    )
    repair_rows: list[dict[str, Any]] = []
    repair_summary: list[dict[str, Any]] = []
    if needs_repair:
        repair_rows, repair_summary = _attempt_repair(args, device, batch_sizes)
    _merge_repair_summary(summary, repair_summary)
    route = {
        "D-RAT_robust_pass": int(any(r.get("carrier") == "D-RAT" and str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "D-RBF_robust_pass": int(any(r.get("carrier") == "D-RBF" and str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "D-RAT_near_E1_rows": next((r.get("near_E1_rows") for r in summary if r.get("carrier") == "D-RAT"), 0),
        "D-RBF_near_E1_rows": next((r.get("near_E1_rows") for r in summary if r.get("carrier") == "D-RBF"), 0),
        "multibatch_status_closed": int(all(str(r.get("decision")) == "RobustProductionPass" for r in summary)),
        "limited_smoke_allowed": int(any(int(r.get("near_E1_rows", 0)) >= 3 for r in summary)),
        "repair_attempted": int(bool(repair_summary)),
        "repair_attempt_rows": sum(int(_f(r.get("repair_attempt_rows"), 0)) for r in repair_summary),
        "D-RAT_repair_micro_near_E1_rows": next((r.get("repair_micro_near_E1_rows") for r in repair_summary if r.get("carrier") == "D-RAT"), 0),
        "D-RBF_repair_micro_near_E1_rows": next((r.get("repair_micro_near_E1_rows") for r in repair_summary if r.get("carrier") == "D-RBF"), 0),
        "repair_official_closure_claimed": 0,
    }
    write_rows(out_dir / "v22_07_drat_drbf_multibatch_officialization.csv", rows)
    write_rows(out_dir / "v22_07_drat_component_waterfall.csv", drat_waterfall)
    write_rows(out_dir / "v22_07_drbf_component_waterfall.csv", drbf_waterfall)
    write_rows(out_dir / "v22_07_drat_drbf_repair_attempts.csv", repair_rows)
    write_rows(out_dir / "v22_07_drat_drbf_repair_attempts_summary.csv", repair_summary)
    write_rows(out_dir / "v22_07_drat_drbf_multibatch_summary.csv", summary)
    write_json(out_dir / "v22_07_drat_drbf_multibatch_route.json", route)
    simple_svg(out_dir / "figures/D-RAT_D-RBF_multibatch_efficiency_dashboard.svg", "v22.07 D-RAT/D-RBF multibatch", rows, "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/D-RAT_component_waterfall.svg", "v22.07 D-RAT component waterfall", [r for r in rows if r.get("carrier") == "D-RAT"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/D-RBF_component_waterfall.svg", "v22.07 D-RBF component waterfall", [r for r in rows if r.get("carrier") == "D-RBF"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/D-RAT_D-RBF_repair_attempt_dashboard.svg", "v22.07 D-RAT/D-RBF repair attempts", repair_rows, "forward_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_drat_drbf_multibatch.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(rows)} robust={sum(int(r.get('robust_production_pass',0)) for r in rows)} near={sum(int(r.get('near_E1_v22_07',0)) for r in rows)} telemetry_complete={sum(int(r.get('component_telemetry_complete',0)) for r in rows)} repair_rows={len(repair_rows)} repair_micro_near={sum(int(_f(r.get('micro_near_E1'),0)) for r in repair_rows)}",
    )


if __name__ == "__main__":
    main()
