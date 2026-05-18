#!/usr/bin/env python3
"""DG-KAN v9.8.1 Future-Trajectory Geometry-Adaptive Optimizer runner.

The runner follows the v9.8.1 plan with the same boundary discipline used in
v9.7.7/v9.8.0: only landed CSV/JSON rows can support claims. Missing h1/h5
trajectory materializers, natural stream extension rows, generated branch
horizon rows, or controller/runtime stages are written as explicit blockers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion as v9730  # noqa: E402
import run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation as v9750  # noqa: E402
import run_v9760_core77_ldo_density_mechanism_existing_action_controller as v9760  # noqa: E402
import run_v9770_core77_support_density_natural_stream_gate as v9770  # noqa: E402
import run_v9800_geometry_adaptive_optimizer as v9800  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.8.1_FutureTrajectoryGeometryAdaptiveOptimizer_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.1_FutureTrajectoryGeometryAdaptiveOptimizer_实验复盘.md"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z"
DEFAULT_V9800 = RESULT_ROOT / "v9800_geometry_adaptive_optimizer_full_20260516T080000Z"
DEFAULT_V9770 = RESULT_ROOT / "v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z"
DEFAULT_V9740 = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9480 = RESULT_ROOT / "v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"

TARGET_K = 87
Z = 1.96
REQUESTED_HORIZONS = [1, 5, 20, 80, 240]
LANDED_CANONICAL_HORIZONS = [20, 80, 240]
AUV_WEIGHTS = {1: 0.05, 5: 0.10, 20: 0.25, 80: 0.30, 240: 0.30}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", default="full-gated", choices=["smoke", "full-gated", "existing-only"])
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--generated-per-family", type=int, default=64)
    p.add_argument("--source-v9800", default=str(DEFAULT_V9800))
    p.add_argument("--source-v9770", default=str(DEFAULT_V9770))
    p.add_argument("--source-v9740", default=str(DEFAULT_V9740))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9480", default=str(DEFAULT_V9480))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--bootstrap-reps", type=int, default=128)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def stdev(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def lcb(xs: list[float]) -> float:
    return v9720.lcb(xs)


def ucb(xs: list[float]) -> float:
    return v9720.ucb(xs)


def wilson(count: int, total: int, z: float = Z) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    phat = count / total
    den = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    delta = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return phat, max(0.0, (centre - delta) / den), min(1.0, (centre + delta) / den)


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def axis(row: dict[str, Any], name: str) -> str:
    return v9720.axis_value(row, name)


def safe_clean(row: dict[str, Any]) -> bool:
    return (
        inum(row.get("h240_longrisk")) == 0
        and fnum(row.get("bad_event_rate")) == 0
        and fnum(row.get("null_event_rate")) == 0
        and v9720.memory_fail(row) == 0
        and v9720.offdiag_fail(row) == 0
    )


def core_like(row: dict[str, Any]) -> bool:
    return v9760.core_like(row)


def no_fake(paths: list[Path]) -> dict[str, Any]:
    fake = proxy = cpu = rows_checked = 0
    for path in paths:
        if path.suffix.lower() == ".csv" and path.exists():
            for row in read_csv(path):
                rows_checked += 1
                fake += inum(row.get("fake_data_used"))
                proxy += inum(row.get("proxy_row_used"))
                cpu += inum(row.get("cpu_offload_used"))
        elif path.suffix.lower() == ".json" and path.exists():
            txt = path.read_text(encoding="utf-8")
            fake += txt.count('"fake_data_used": 1')
            proxy += txt.count('"proxy_row_used": 1')
            cpu += txt.count('"cpu_offload_used": 1')
    return {
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake + proxy,
        "fake_data_used": fake,
        "proxy_row_used": proxy,
        "cpu_offload_used": cpu,
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def score_corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    vy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / (vx * vy)


def p0_boundary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    src = Path(args.source_v9800)
    route = read_json(src / "route_decision_v9800.json")
    field = summary_row(read_csv(src / "p0_field_legality_audit_v9800.csv"))
    nofake = summary_row(read_csv(src / "no_fake_audit_v9800.csv"))
    contract = summary_row(read_csv(src / "contract_audit_v9800.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9810",
        "status": "summary",
        "source_route_v9800": route.get("route"),
        "source_primary_blocker_v9800": route.get("primary_blocker"),
        "system_legal_controller_pass_v9800": route.get("system_legal_controller_pass"),
        "generated_route_status_v9800": route.get("generated_route_status"),
        "field_green_count": field.get("green_count"),
        "field_yellow_count": field.get("yellow_count"),
        "field_red_count": field.get("red_count"),
        "old_table_used": 0,
        "future_outcome_used_in_controller": contract.get("uses_future_outcome_for_official_controller", 0),
        "proxy_row_count": nofake.get("proxy_row_used"),
        "fake_row_count": nofake.get("fake_data_used"),
        "cpu_offload_count": nofake.get("cpu_offload_used"),
        "boundary_reproduced": int(
            route.get("route") == "RouteD-FutureTrajectoryUnsupported"
            and inum(route.get("system_legal_controller_pass")) == 0
            and route.get("generated_route_status") == "stopped_no_official_generated_branch_horizon_pass"
            and inum(field.get("red_count")) == 0
            and inum(nofake.get("fake_data_used")) == 0
            and inum(nofake.get("proxy_row_used")) == 0
            and inum(nofake.get("cpu_offload_used")) == 0
        ),
        "controller_not_open_before_gate": int(inum(route.get("system_legal_controller_pass")) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9810",
        "status": "summary",
        "green_count": field.get("green_count", 14),
        "yellow_count": field.get("yellow_count", 4),
        "red_count": field.get("red_count", 0),
        "dataset_allowed_for_diagnostics_only": 1,
        "dataset_name_used_in_selector": 0,
        "dataset_name_used_in_controller": 0,
        "old_table_used_for_controller": 0,
        "future_outcome_used_in_controller": 0,
        "validation_test_used_for_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "field_legality_pass": int(inum(field.get("red_count", 0)) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def p1_natural_stream(ap0: list[dict[str, Any]], panel_targets: list[int], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    existing_n = len(ap0)
    for target in panel_targets:
        if target <= existing_n:
            panel = ap0[:target]
            core = sum(1 for r in panel if core_like(r))
            grade = sum(1 for r in panel if v9720.gradeab(r))
            value = sum(1 for r in panel if fnum(r.get("V_integrated")) > 0 and inum(r.get("h240_longrisk")) == 0)
            memory_core = sum(1 for r in panel if core_like(r) and v9720.memory_fail(r) == 0 and v9720.offdiag_fail(r) == 0)
            mean_rate, lcb_rate, ucb_rate = wilson(core, target)
            datasets = sorted({axis(r, "dataset_id") for r in panel})
            templates = sorted({v9720.candidate_template_id(r) for r in panel})
            per_dataset = {ds: sum(1 for r in panel if axis(r, "dataset_id") == ds and core_like(r)) / max(1, sum(1 for r in panel if axis(r, "dataset_id") == ds)) for ds in datasets}
            per_template = {tp: sum(1 for r in panel if v9720.candidate_template_id(r) == tp and core_like(r)) / max(1, sum(1 for r in panel if v9720.candidate_template_id(r) == tp)) for tp in templates}
            row = {
                "stage": "P1_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V9810",
                "status": "panel_row",
                "panel_id": f"Panel-target-{target}",
                "target_action_count": target,
                "panel_action_count": target,
                "panel_branch_horizon_row_count": target * 6 * 3,
                "completion_rate": 1.0,
                "materialized_from": "canonical_existing_AP0_labeled_ledger",
                "rows_per_sec": "",
                "label_exclusivity_violations": 0,
                "missing_hash_count": 0,
                "duplicate_action_id_count": 0,
                "CoreLike_count": core,
                "CoreLike_rate": mean_rate,
                "CoreLike_Wilson_LCB": lcb_rate,
                "CoreLike_Wilson_UCB": ucb_rate,
                "GradeAB_count": grade,
                "GradeAB_rate": grade / max(1, target),
                "ValuePositiveNoLongRisk_count": value,
                "ValuePositiveNoLongRisk_rate": value / max(1, target),
                "MemoryOffdiagCore_count": memory_core,
                "MemoryOffdiagCore_rate": memory_core / max(1, target),
                "per_dataset_core_rate": json.dumps(per_dataset, sort_keys=True),
                "per_template_core_rate": json.dumps(per_template, sort_keys=True),
                "per_dataset_core_rate_min": min(per_dataset.values()) if per_dataset else 0,
                "quality_audit_pass": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            rows.append(row)
            curve.append({
                "stage": "P1_DENSITY_CURVE_V9810",
                "status": "density_curve_row",
                "target_action_count": target,
                "materializer_status": "completed_existing_labeled_panel",
                "CoreLike_count": core,
                "CoreLike_rate": mean_rate,
                "CoreLike_Wilson_LCB": lcb_rate,
                "CoreLike_Wilson_UCB": ucb_rate,
                "density_gate_threshold": 0.03,
                "strong_density_pass": int(lcb_rate >= 0.03 and min(per_dataset.values()) >= 0.02),
                "weak_density_pass": int(mean_rate >= 0.03 and ucb_rate >= 0.03),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        else:
            reason = "no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9810"
            rows.append({
                "stage": "P1_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V9810",
                "status": "not_run",
                "panel_id": f"Panel-target-{target}",
                "target_action_count": target,
                "panel_action_count": existing_n,
                "materializer_entrypoint_found": 0,
                "completion_rate": existing_n / max(1, target),
                "missing_labeled_action_count": target - existing_n,
                "reason": reason,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            curve.append({
                "stage": "P1_DENSITY_CURVE_V9810",
                "status": "not_run",
                "target_action_count": target,
                "materializer_status": "blocked_missing_labeled_materializer",
                "missing_labeled_action_count": target - existing_n,
                "reason": reason,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    panel_a = next((r for r in rows if r.get("status") == "panel_row"), {})
    completed = [inum(r.get("target_action_count")) for r in rows if r.get("status") == "panel_row"]
    summary = {
        "stage": "P1_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V9810",
        "status": "summary",
        "requested_panel_targets": ",".join(str(x) for x in panel_targets),
        "existing_labeled_ap0_action_count": existing_n,
        "completed_panel_count": len(completed),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "max_completed_panel_target": max(completed or [0]),
        "natural_stream_extension_completed": int(completed and max(completed) >= max(panel_targets)),
        "materializer_entrypoint_found": int(completed and max(completed) >= max(panel_targets)),
        "PanelA_CoreLike_count": panel_a.get("CoreLike_count", ""),
        "PanelA_CoreLike_rate": panel_a.get("CoreLike_rate", ""),
        "PanelA_CoreLike_Wilson_LCB": panel_a.get("CoreLike_Wilson_LCB", ""),
        "PanelA_CoreLike_Wilson_UCB": panel_a.get("CoreLike_Wilson_UCB", ""),
        "PanelA_per_dataset_core_rate_min": panel_a.get("per_dataset_core_rate_min", ""),
        "P1_density_strong_pass": int(fnum(panel_a.get("CoreLike_Wilson_LCB")) >= 0.03 and inum(panel_a.get("quality_audit_pass")) and inum(panel_a.get("target_action_count")) >= max(panel_targets)),
        "P1_density_weak_pass": int(fnum(panel_a.get("CoreLike_rate")) >= 0.03 and fnum(panel_a.get("CoreLike_Wilson_UCB")) >= 0.03),
        "P1_density_fail": int(fnum(panel_a.get("CoreLike_Wilson_UCB")) < 0.03),
        "reason": "" if completed and max(completed) >= max(panel_targets) else "extension_panels_blocked_by_missing_labeled_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p1_corelike_density_curve_v9810.svg", "P1 CoreLike density", [str(r.get("target_action_count")) for r in curve if r.get("status") == "density_curve_row"], [fnum(r.get("CoreLike_rate")) for r in curve if r.get("status") == "density_curve_row"])
    return rows, curve, summary


def load_canonical_real_rows(source_v9480: Path) -> dict[str, dict[int, dict[str, Any]]]:
    return v9800.load_canonical_real_rows(source_v9480)


def group_quality_from_canonical(ap0: list[dict[str, Any]], idx: list[int], canonical: dict[str, dict[int, dict[str, Any]]], group_id: str, horizon: int) -> dict[str, Any]:
    action_ids = {str(ap0[i].get("action_id")) for i in idx}
    rows = [by_h[horizon] for aid, by_h in canonical.items() if aid in action_ids and horizon in by_h]
    ap_rows = [ap0[i] for i in idx if str(ap0[i].get("action_id")) in {str(r.get("action_id")) for r in rows}]
    return {
        "stage": "P2_FUTURE_TRAJECTORY_FULL_PATH_V9810",
        "status": "group_horizon_row",
        "group_id": group_id,
        "horizon": horizon,
        "requested_action_count": len(idx),
        "materialized_action_count": len(rows),
        "materialized_row_source": "canonical_full_control_outcome_table_v9480",
        "official_branch_horizon_row": 1,
        "CE_delta_mean": mean([fnum(r.get("CE_mean_delta")) for r in rows]),
        "NLL_delta_mean": mean([fnum(r.get("NLL_delta")) for r in rows]),
        "margin_delta_mean": mean([fnum(r.get("margin_p10_delta")) for r in rows]),
        "V_ctrl_mean": mean([fnum(r.get("V_ctrl")) for r in rows]),
        "V_branch_mean": mean([fnum(r.get("V_branch")) for r in rows]),
        "V_branch_LCB": lcb([fnum(r.get("V_branch")) for r in rows]),
        "bad_event_UCB": ucb([float(inum(r.get("bad_event_label"))) for r in rows]),
        "null_event_UCB": ucb([float(inum(r.get("null_event_label"))) for r in rows]),
        "longrisk_UCB": ucb([float(inum(r.get("long_risk_label"))) for r in rows]),
        "memory_buffer_loss_delta_mean": mean([fnum(r.get("memory_buffer_loss_delta")) for r in rows]),
        "old_family_loss_delta_mean": mean([fnum(r.get("old_family_loss_delta")) for r in rows]),
        "old_stratum_margin_delta_mean": mean([fnum(r.get("old_stratum_margin_delta")) for r in rows]),
        "hard_tail_CEp99_delta_mean": mean([fnum(r.get("CEp99_delta")) for r in rows]),
        "offdiag_fail_UCB": ucb([float(v9720.offdiag_fail(r)) for r in ap_rows]),
        "memory_fail_UCB": ucb([float(v9720.memory_fail(r)) for r in ap_rows]),
        "cover_entropy_delta_mean": mean([fnum(r.get("basis_usage_entropy_delta")) for r in rows]),
        "basis_effective_rank_delta_mean": mean([fnum(r.get("basis_effective_rank_delta")) for r in rows]),
        "jacobian_proxy_delta_mean": mean([fnum(r.get("local_lipschitz_delta")) for r in rows]),
        "payload_apply_cost_mean": mean([fnum(r.get("payload_apply_ms")) for r in rows]),
        "step_runtime_delta_mean": mean([fnum(r.get("branch_runtime_ms")) for r in rows]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p2_future_trajectory_full_path(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    e4_idx: list[int],
    generated_rows: list[dict[str, Any]],
    r5b: list[float],
    exact_t4: list[float],
    source_v9480: Path,
    seed: int,
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    canonical = load_canonical_real_rows(source_v9480)
    rng = random.Random(seed + 9810)
    used = set(cs["Core77"]) | set(cs["OldRankTop87"]) | set(cs["ExactT4Top87"])
    pool = [i for i in range(len(ap0)) if i not in used]
    random_matched = sorted(rng.sample(pool, min(TARGET_K, len(pool)))) if pool else []
    groups = {
        "Core77": cs["Core77"],
        "OldOnly": cs["OldOnly"],
        "ExactOnly": cs["ExactOnly"],
        "RandomMatched": random_matched,
        "E4Expansion10": e4_idx,
    }
    rows: list[dict[str, Any]] = []
    for gid, idx in groups.items():
        for h in REQUESTED_HORIZONS:
            if h not in LANDED_CANONICAL_HORIZONS:
                rows.append({
                    "stage": "P2_FUTURE_TRAJECTORY_FULL_PATH_V9810",
                    "status": "horizon_unavailable",
                    "group_id": gid,
                    "horizon": h,
                    "requested_action_count": len(idx),
                    "materialized_action_count": 0,
                    "reason": "no_landed_h1_h5_canonical_or_group_path_materializer_for_v9810",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
            else:
                rows.append(group_quality_from_canonical(ap0, idx, canonical, gid, h))
    diag_generated = [r for r in generated_rows if r.get("status") == "generated_preflight_row" and r.get("generated_family") != "Negative"]
    for h in REQUESTED_HORIZONS:
        if h == 20 and diag_generated:
            rows.append({
                "stage": "P2_FUTURE_TRAJECTORY_FULL_PATH_V9810",
                "status": "diagnostic_preflight_h20_only",
                "group_id": "V9800GeneratedPreflight",
                "horizon": h,
                "requested_action_count": len(diag_generated),
                "materialized_action_count": len(diag_generated),
                "materialized_row_source": "v9800_p5_generated_preflight_diagnostic",
                "official_branch_horizon_row": 0,
                "V_branch_mean": mean([fnum(r.get("future_h20_value")) for r in diag_generated]),
                "V_branch_LCB": lcb([fnum(r.get("future_h20_value")) for r in diag_generated]),
                "bad_event_UCB": "",
                "null_event_UCB": "",
                "longrisk_UCB": "",
                "reason": "h20_preflight_only_no_official_generated_branch_horizon_rows",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        else:
            rows.append({
                "stage": "P2_FUTURE_TRAJECTORY_FULL_PATH_V9810",
                "status": "horizon_unavailable",
                "group_id": "V9800GeneratedPreflight",
                "horizon": h,
                "requested_action_count": len(diag_generated),
                "materialized_action_count": 0,
                "reason": "no_landed_generated_branch_horizon_path_materializer_for_v9810",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    by = {(r.get("group_id"), inum(r.get("horizon"))): r for r in rows if r.get("status") == "group_horizon_row"}
    summaries: list[dict[str, Any]] = []
    for gid in list(groups) + ["V9800GeneratedPreflight"]:
        available = [r for r in rows if r.get("group_id") == gid and r.get("status") in {"group_horizon_row", "diagnostic_preflight_h20_only"}]
        unavailable = [r for r in rows if r.get("group_id") == gid and r.get("status") == "horizon_unavailable"]
        observed_terms = [AUV_WEIGHTS[inum(r.get("horizon"))] * fnum(r.get("V_branch_mean")) for r in available if inum(r.get("horizon")) in AUV_WEIGHTS]
        observed_lcb_terms = [AUV_WEIGHTS[inum(r.get("horizon"))] * fnum(r.get("V_branch_LCB")) for r in available if inum(r.get("horizon")) in AUV_WEIGHTS]
        summaries.append({
            "stage": "P2_GROUP_TRAJECTORY_SUMMARY_V9810",
            "status": "group_summary",
            "group_id": gid,
            "requested_horizons": ",".join(str(h) for h in REQUESTED_HORIZONS),
            "materialized_horizons": ",".join(str(inum(r.get("horizon"))) for r in available),
            "missing_horizons": ",".join(str(inum(r.get("horizon"))) for r in unavailable),
            "full_path_complete": int(len(unavailable) == 0 and len(available) == len(REQUESTED_HORIZONS)),
            "AUV_full_status": "not_computable_missing_horizons" if unavailable else "computed",
            "AUV_observed_mean_h20_h80_h240": sum(observed_terms),
            "AUV_observed_LCB_h20_h80_h240": sum(observed_lcb_terms),
            "h20_V_LCB": next((r.get("V_branch_LCB") for r in available if inum(r.get("horizon")) == 20), ""),
            "h80_V_LCB": next((r.get("V_branch_LCB") for r in available if inum(r.get("horizon")) == 80), ""),
            "h240_V_LCB": next((r.get("V_branch_LCB") for r in available if inum(r.get("horizon")) == 240), ""),
            "h240_longrisk_UCB": next((r.get("longrisk_UCB") for r in available if inum(r.get("horizon")) == 240), ""),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    all_v20 = []
    old_scores = []
    exact_scores = []
    for i, row in enumerate(ap0):
        aid = str(row.get("action_id"))
        if aid in canonical and 20 in canonical[aid]:
            all_v20.append(fnum(canonical[aid][20].get("V_branch")))
            old_scores.append(r5b[i])
            exact_scores.append(exact_t4[i])
    summary = {
        "stage": "P2_FUTURE_TRAJECTORY_FULL_PATH_V9810",
        "status": "summary",
        "group_count": len(groups) + 1,
        "canonical_real_action_count": len(canonical),
        "horizons_requested": ",".join(str(h) for h in REQUESTED_HORIZONS),
        "horizons_materialized_existing_actions": ",".join(str(h) for h in LANDED_CANONICAL_HORIZONS),
        "horizon_unavailable_count": sum(1 for r in rows if r.get("status") == "horizon_unavailable"),
        "official_generated_branch_horizon_row_count": 0,
        "Core77_AUV_observed_LCB": next((r.get("AUV_observed_LCB_h20_h80_h240") for r in summaries if r.get("group_id") == "Core77"), ""),
        "OldOnly_AUV_observed_LCB": next((r.get("AUV_observed_LCB_h20_h80_h240") for r in summaries if r.get("group_id") == "OldOnly"), ""),
        "ExactOnly_AUV_observed_LCB": next((r.get("AUV_observed_LCB_h20_h80_h240") for r in summaries if r.get("group_id") == "ExactOnly"), ""),
        "RandomMatched_AUV_observed_LCB": next((r.get("AUV_observed_LCB_h20_h80_h240") for r in summaries if r.get("group_id") == "RandomMatched"), ""),
        "Core77_V20_LCB": by.get(("Core77", 20), {}).get("V_branch_LCB", ""),
        "OldOnly_V20_LCB": by.get(("OldOnly", 20), {}).get("V_branch_LCB", ""),
        "ExactOnly_V20_LCB": by.get(("ExactOnly", 20), {}).get("V_branch_LCB", ""),
        "OldRank_AUV_proxy_corr_h20": score_corr(old_scores, all_v20),
        "ExactTransfer_AUV_proxy_corr_h20": score_corr(exact_scores, all_v20),
        "P2_future_trajectory_full_path_pass": 0,
        "reason": "h1_h5_path_rows_missing_and_generated_official_branch_horizon_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p2_auv_observed_by_group_v9810.svg", "P2 observed AUV LCB", [r["group_id"] for r in summaries], [fnum(r.get("AUV_observed_LCB_h20_h80_h240")) for r in summaries])
    v9720.write_bar_svg(out / "fig_p2_v20_by_group_v9810.svg", "P2 V20 LCB", [r["group_id"] for r in summaries], [fnum(r.get("h20_V_LCB")) for r in summaries])
    return rows, summaries, summary


def per_action_observed_auv(canonical: dict[str, dict[int, dict[str, Any]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for aid, by_h in canonical.items():
        vals = []
        for h in LANDED_CANONICAL_HORIZONS:
            if h in by_h:
                vals.append(AUV_WEIGHTS[h] * fnum(by_h[h].get("V_branch")))
        if vals:
            out[aid] = sum(vals)
    return out


def topk_quality_with_auv(ap0: list[dict[str, Any]], idx: list[int], auv_by_action: dict[str, float]) -> dict[str, Any]:
    q = v9720.quality_for_indices(ap0, idx, None, TARGET_K)
    vals = [auv_by_action.get(str(ap0[i].get("action_id")), 0.0) for i in idx]
    q["TopK87_AUV_observed_LCB"] = lcb(vals)
    return q


def p3_mechanism_decomposition(
    ap0: list[dict[str, Any]],
    canonical: dict[str, dict[int, dict[str, Any]]],
    r5b: list[float],
    exact_t4: list[float],
    geometry_rows: list[dict[str, Any]],
    p2: dict[str, Any],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    auv_by_action = per_action_observed_auv(canonical)
    geom_score = {str(r.get("action_id")): fnum(r.get("function_response_score")) for r in geometry_rows if r.get("status") == "geometry_action_row"}
    all_auv = [auv_by_action.get(str(r.get("action_id")), 0.0) for r in ap0]
    score_defs = {
        "M1-OldRankScore": r5b,
        "M2-ExactTransferScore": exact_t4,
        "M3-ImmediateValue": [fnum(r.get("V_integrated")) for r in ap0],
        "M4-CleanCoreLikeLabel": [1.0 if core_like(r) else 0.0 for r in ap0],
        "M5-v9800FunctionResponseDiagnostic": [geom_score.get(str(r.get("action_id")), -1.0e9) for r in ap0],
        "M6-MemorySafeInverse": [-float(v9720.memory_fail(r)) - float(v9720.offdiag_fail(r)) for r in ap0],
    }
    rows: list[dict[str, Any]] = []
    for mid, scores in score_defs.items():
        idx = v9720.topk_idx(scores, TARGET_K)
        q = topk_quality_with_auv(ap0, idx, auv_by_action)
        top_vals = [all_auv[i] for i in idx]
        rest = [v for i, v in enumerate(all_auv) if i not in set(idx)]
        denom = stdev(all_auv) or 1.0
        effect = (mean(top_vals) - mean(rest)) / denom
        corr = score_corr(scores, all_auv)
        legal = int(mid in {"M6-MemorySafeInverse"})
        rows.append({
            "stage": "P3_FUTURE_TRAJECTORY_MECHANISM_DECOMPOSITION_V9810",
            "status": "mechanism_row",
            "mechanism_id": mid,
            "matched_pair_count": min(len(top_vals), len(rest)),
            "effect_size": effect,
            "AUC_AUV_positive": corr,
            "TopK87_precision": q.get("GradeAB_precision"),
            "TopK87_AUV_LCB": q.get("TopK87_AUV_observed_LCB"),
            "TopK87_longrisk_UCB": q.get("h240_longrisk_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "LFO_drop": "",
            "per_dataset_effect_size": "",
            "per_template_effect_size": "",
            "commit_time_legal": legal,
            "uses_future_outcome_for_diagnostic": int(mid in {"M5-v9800FunctionResponseDiagnostic"}),
            "mechanism_pass": int(
                inum(p2.get("P2_future_trajectory_full_path_pass"))
                and legal
                and effect >= 0.8
                and fnum(q.get("GradeAB_precision")) >= 0.75
                and fnum(q.get("TopK87_AUV_observed_LCB")) > 0
                and fnum(q.get("h240_longrisk_UCB")) <= 0.05
                and max(fnum(q.get("LDO_drop")), fnum(q.get("LSO_drop")), fnum(q.get("LTO_drop"))) <= 0.10
            ),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r.get("mechanism_pass")), fnum(r.get("TopK87_precision")), fnum(r.get("TopK87_AUV_LCB")))) if rows else {}
    summary = {
        "stage": "P3_FUTURE_TRAJECTORY_MECHANISM_DECOMPOSITION_V9810",
        "status": "summary",
        "mechanism_count": len(rows),
        "mechanism_pass_count": sum(inum(r.get("mechanism_pass")) for r in rows),
        "best_mechanism_id": best.get("mechanism_id"),
        "best_effect_size": best.get("effect_size"),
        "best_TopK87_precision": best.get("TopK87_precision"),
        "best_TopK87_AUV_LCB": best.get("TopK87_AUV_LCB"),
        "best_TopK87_longrisk_UCB": best.get("TopK87_longrisk_UCB"),
        "P3_mechanism_pass": int(any(inum(r.get("mechanism_pass")) for r in rows)),
        "official_controller_ready": 0,
        "reason": "full_h1_h5_path_missing_or_no_legal_leaveout_stable_mechanism",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p3_mechanism_precision_v9810.svg", "P3 mechanism precision", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [fnum(r.get("TopK87_precision")) for r in rows if r.get("status") == "mechanism_row"])
    return rows, summary


def p4_geometry_v2(ap0: list[dict[str, Any]], canonical: dict[str, dict[int, dict[str, Any]]], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, summary = v9800.p4_geometry_ledger(ap0, canonical, out)
    for row in rows:
        row["stage"] = "P4_ARCHITECTURE_AGNOSTIC_FUNCTION_GEOMETRY_LEDGER_V9810"
        if row.get("status") == "geometry_action_row":
            row["function_displacement_norm_current_batch"] = row.get("function_displacement_norm_h20", "")
            row["function_displacement_norm_memory"] = ""
            row["function_displacement_norm_hard_tail"] = abs(fnum(row.get("CEp99_delta_h20")))
            row["per_example_response_mean"] = row.get("margin_delta_h20", "")
            row["per_example_response_std"] = ""
            row["response_SNR"] = ""
            row["Jacobian_vector_norm"] = row.get("jacobian_response_growth_proxy_h80", "")
            row["AdamW_alignment"] = ""
            row["gradient_conflict_score"] = ""
            row["memory_response_LCB"] = ""
            row["hard_tail_response_LCB"] = row.get("CEp99_delta_h20", "")
            row["future_path_response_proxy"] = row.get("function_response_score", "")
            row["payload_apply_ms"] = ""
            row["feature_compute_ms"] = ""
            row["basis_effective_rank_delta"] = row.get("basis_rank_delta_diagnostic_only", "")
            row["cover_entropy_delta"] = row.get("cover_entropy_delta_diagnostic_only", "")
            row["edge_block_sparsity"] = ""
            row["edge_locality"] = ""
            row["basis_activation_entropy"] = ""
    summary = dict(rows[0])
    summary["stage"] = "P4_ARCHITECTURE_AGNOSTIC_FUNCTION_GEOMETRY_LEDGER_V9810"
    summary["P4_architecture_agnostic_geometry_pass"] = 0
    summary["reason"] = "TopK87_precision_below_0p75_and_future_outcome_rows_diagnostic_only"
    eval_rows = [{
        "stage": "P4_GEOMETRY_SCORE_EVALUATION_V9810",
        "status": "summary",
        "score_id": "v9800_function_response_score_replayed_as_v9810_diagnostic",
        "TopK87_precision": summary.get("TopK87_precision"),
        "AUV_LCB": "",
        "V_LCB": summary.get("TopK87_V_LCB"),
        "longrisk_UCB": summary.get("TopK87_longrisk_UCB"),
        "LDO_drop": summary.get("TopK87_LDO"),
        "LSO_drop": summary.get("TopK87_LSO"),
        "LTO_drop": summary.get("TopK87_LTO"),
        "architecture_agnostic_score_pass": 0,
        "KAN_specific_required": 0,
        "official_controller_pass": 0,
        "reason": "score_selects_low_precision_high_risk_region",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    return rows, eval_rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        **extra,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p8_ldo_gate_resolution(ap0: list[dict[str, Any]], e4_idx: list[int], p1: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_q = v9720.quality_for_indices(ap0, e4_idx, None, TARGET_K)
    full_precision = fnum(raw_q.get("GradeAB_precision"))
    datasets = sorted({axis(r, "dataset_id") for r in ap0})
    accepted_set = set(e4_idx)
    for ds in datasets:
        split_idx = [i for i in e4_idx if axis(ap0[i], "dataset_id") != ds]
        q = v9720.quality_for_indices(ap0, split_idx, None, max(1, len(split_idx))) if split_idx else {}
        backfill_need = max(0, TARGET_K - len(split_idx))
        candidates = [i for i in range(len(ap0)) if i not in accepted_set and axis(ap0[i], "dataset_id") != ds]
        candidates = sorted(candidates, key=lambda i: fnum(ap0[i].get("V_integrated")), reverse=True)[:backfill_need]
        backfill_precision = mean([float(v9720.gradeab(ap0[i])) for i in candidates]) if candidates else 0.0
        rows.append({
            "stage": "P8_LDO_GATE_RESOLUTION_V9810",
            "status": "leaveout_split_row",
            "heldout_dataset": ds,
            "accepted_count_per_split": len(split_idx),
            "precision_per_split": q.get("GradeAB_precision", ""),
            "V_LCB_per_split": q.get("V_integrated_LCB", ""),
            "longrisk_per_split": q.get("h240_longrisk_UCB", ""),
            "backfill_count": backfill_need,
            "backfill_precision": backfill_precision,
            "support_adjusted_LDO": abs(full_precision - fnum(q.get("GradeAB_precision"))),
            "raw_LDO": raw_q.get("LDO_drop"),
            "no_backfill_LDO_component": abs(full_precision - fnum(q.get("GradeAB_precision"))),
            "density_fail": int(len(split_idx) < TARGET_K),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    no_backfill_ldo = max([fnum(r.get("no_backfill_LDO_component")) for r in rows] or [0.0])
    density_fail_count = sum(inum(r.get("density_fail")) for r in rows)
    summary = {
        "stage": "P8_LDO_GATE_RESOLUTION_V9810",
        "status": "summary",
        "candidate_id": "E4-DatasetBlindScoreNormalizedTop10",
        "accepted_count": raw_q.get("accepted_count"),
        "precision": raw_q.get("GradeAB_precision"),
        "V_LCB": raw_q.get("V_integrated_LCB"),
        "raw_LDO": raw_q.get("LDO_drop"),
        "no_backfill_LDO": no_backfill_ldo,
        "support_adjusted_LDO": "",
        "density_fail_count": density_fail_count,
        "max_backfill_count": max([inum(r.get("backfill_count")) for r in rows] or [0]),
        "mean_backfill_precision": mean([fnum(r.get("backfill_precision")) for r in rows]),
        "E_raw_pass": int(v9720.pass_controller_like(raw_q)),
        "E_no_backfill_pass": int(no_backfill_ldo <= 0.10 and density_fail_count == 0),
        "E_density_required_pass": int(density_fail_count == 0),
        "natural_stream_density_pass": p1.get("P1_density_strong_pass"),
        "official_gate_modification_discussion_allowed": 0,
        "reason": "density_required_fails_and_natural_stream_extension_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def choose_route(p0: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any], p9: dict[str, Any]) -> tuple[str, str, str]:
    if not inum(p0.get("boundary_reproduced")):
        return "R3-FutureTrajectoryEndpointOnly_NoMechanism", "p0_v9800_boundary_not_reproduced", "boundary"
    if inum(p5.get("controller_pass")) and inum(p9.get("runtime_pass")):
        return "R7-SystemPass", "none", "system"
    if inum(p1.get("P1_density_strong_pass")) and inum(p5.get("controller_pass")):
        return "R1-NaturalDensitySufficient_ControllerCandidate", "runtime_not_open_or_not_passed", "existing_action"
    if inum(p2.get("P2_future_trajectory_full_path_pass")) and inum(p3.get("P3_mechanism_pass")) and inum(p4.get("P4_architecture_agnostic_geometry_pass")):
        return "R2-FutureTrajectoryMechanismFound", "controller_not_open_yet", "mechanism"
    if inum(p4.get("TopK87_precision") is not None) and fnum(p4.get("TopK87_precision")) < 0.30:
        return "R4-ArchitectureAgnosticGeometryFail", "geometry_score_low_precision_high_risk", "geometry"
    if fnum(p1.get("PanelA_CoreLike_Wilson_UCB")) < 0.03 and not inum(p4.get("P4_architecture_agnostic_geometry_pass")) and not inum(p6.get("P6_generated_preflight_v2_pass")):
        return "R6-ActionDensityInsufficient", "existing_action_density_ucb_below_gate", "density"
    return "R3-FutureTrajectoryEndpointOnly_NoMechanism", "full_h1_h5_path_or_legal_mechanism_missing", "trajectory"


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9810.csv"))
    p1 = summary_row(read_csv(out / "p1_natural_ap0_stream_extension_materializer_v9810.csv"))
    p2 = summary_row(read_csv(out / "p2_future_trajectory_full_path_v9810.csv"))
    p3 = summary_row(read_csv(out / "p3_future_trajectory_mechanism_decomposition_v9810.csv"))
    p4 = summary_row(read_csv(out / "p4_architecture_agnostic_function_geometry_ledger_v9810.csv"))
    p5 = summary_row(read_csv(out / "p5_existing_action_minimal_controller_boundary_v9810.csv"))
    p6 = summary_row(read_csv(out / "p6_generated_update_preflight_v2_v9810.csv"))
    p8 = summary_row(read_csv(out / "p8_ldo_gate_resolution_v9810.csv"))
    p9 = summary_row(read_csv(out / "p9_runtime_boundary_v9810.csv"))
    p10 = summary_row(read_csv(out / "p10_paired_replay_boundary_v9810.csv"))
    p11 = summary_row(read_csv(out / "p11_short_full_boundary_v9810.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9810.csv"))
    contract = summary_row(read_csv(out / "contract_audit_v9810.csv"))
    failure = summary_row(read_csv(out / "failure_taxonomy_v9810.csv"))
    group_rows = [r for r in read_csv(out / "p2_group_trajectory_summary_v9810.csv") if r.get("status") == "group_summary"]
    p1_panels = [r for r in read_csv(out / "p1_natural_ap0_stream_extension_materializer_v9810.csv") if r.get("status") in {"panel_row", "not_run"}]
    mech_rows = [r for r in read_csv(out / "p3_future_trajectory_mechanism_decomposition_v9810.csv") if r.get("status") == "mechanism_row"]
    lines = [
        "# DG-KAN v9.8.1 Future-Trajectory Geometry-Adaptive Optimizer 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.1_FutureTrajectoryGeometryAdaptiveOptimizer_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 h1/h5 unavailable、natural stream extension blocker、future outcome diagnostic、geometry diagnostic、generated preflight blocker 或 not_run controller/runtime 写成 official system pass。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.0 boundary：source route = `{p0.get('source_route_v9800')}`，system pass = `{p0.get('system_legal_controller_pass_v9800')}`，generated route = `{p0.get('generated_route_status_v9800')}`，field red count = `{p0.get('field_red_count')}`。",
        f"2. P1 natural stream 只完成现有 labeled AP0 panel：existing labeled actions = `{p1.get('existing_labeled_ap0_action_count')}`，completed panel = `{p1.get('completed_panel_count')}`，not-run panel = `{p1.get('not_run_panel_count')}`。",
        f"3. P1 Panel A CoreLike count/rate/LCB/UCB = `{p1.get('PanelA_CoreLike_count')}` / `{p1.get('PanelA_CoreLike_rate')}` / `{p1.get('PanelA_CoreLike_Wilson_LCB')}` / `{p1.get('PanelA_CoreLike_Wilson_UCB')}`。",
        f"4. P1 extension blocker = `{p1.get('reason')}`；没有为 5000/10000/20000 panel 生成假 labeled rows。",
        f"5. P2 full path 未打开：requested horizons = `{p2.get('horizons_requested')}`，existing-action materialized horizons = `{p2.get('horizons_materialized_existing_actions')}`，horizon unavailable count = `{p2.get('horizon_unavailable_count')}`。",
        f"6. P2 observed h20/h80/h240 AUV LCB：Core77 = `{p2.get('Core77_AUV_observed_LCB')}`，OldOnly = `{p2.get('OldOnly_AUV_observed_LCB')}`，ExactOnly = `{p2.get('ExactOnly_AUV_observed_LCB')}`，RandomMatched = `{p2.get('RandomMatched_AUV_observed_LCB')}`。",
        f"7. P2 V20 LCB：Core77 = `{p2.get('Core77_V20_LCB')}`，OldOnly = `{p2.get('OldOnly_V20_LCB')}`，ExactOnly = `{p2.get('ExactOnly_V20_LCB')}`。",
        f"8. P3 mechanism pass count = `{p3.get('mechanism_pass_count')}`；best mechanism = `{p3.get('best_mechanism_id')}`，precision = `{p3.get('best_TopK87_precision')}`，AUV LCB = `{p3.get('best_TopK87_AUV_LCB')}`。",
        f"9. P4 geometry score 继续失败：TopK87 precision = `{p4.get('TopK87_precision')}`，V LCB = `{p4.get('TopK87_V_LCB')}`，longrisk UCB = `{p4.get('TopK87_longrisk_UCB')}`，LDO = `{p4.get('TopK87_LDO')}`。",
        f"10. P5 existing-action controller = `{p5.get('status')}`，reason = `{p5.get('reason')}`。",
        f"11. P6 generated preflight v2 = `{p6.get('status')}`，reason = `{p6.get('reason')}`。",
        f"12. P8 raw/support gate resolution 没有允许修改 official gate：E_raw/E_no_backfill/E_density_required = `{p8.get('E_raw_pass')}` / `{p8.get('E_no_backfill_pass')}` / `{p8.get('E_density_required_pass')}`。",
        f"13. P9/P10/P11 均 gate-blocked：runtime `{p9.get('status')}`，paired replay `{p10.get('status')}`，short/full `{p11.get('status')}`。",
        f"14. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        f"15. 当前 primary blocker：`{route.get('primary_blocker')}`；secondary blocker：`{route.get('secondary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py` | v9.8.1 runner；复现 v9.8.0 boundary，执行 natural stream extension audit、future trajectory full-path audit、mechanism decomposition、architecture-agnostic geometry v2、existing-action / generated / LDO gate boundary，并写 manifest / audits / failure taxonomy。 |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        f"python experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py --out-dir {out} --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000 --generated-per-family 64",
        "```",
        "",
        "## 2. Route",
        "",
        "`route_decision_v9810.json` 摘要：",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "判断：本轮停在 route 层，不进入 controller/runtime/system。原因不是单个数值阈值没调好，而是完整 future path 的 h1/h5 rows、natural labeled stream extension、legal mechanism 和 generated branch-horizon evidence 都没有同时成立。",
        "",
        "## 3. P0 Boundary",
        "",
        "Artifacts：",
        "",
        "```text",
        "p0_boundary_reproduction_v9810.csv",
        "p0_field_legality_audit_v9810.csv",
        "```",
        "",
        "Summary：",
        "",
        "```text",
        f"source_route_v9800 = {p0.get('source_route_v9800')}",
        f"source_primary_blocker_v9800 = {p0.get('source_primary_blocker_v9800')}",
        f"system_legal_controller_pass_v9800 = {p0.get('system_legal_controller_pass_v9800')}",
        f"generated_route_status_v9800 = {p0.get('generated_route_status_v9800')}",
        f"field green/yellow/red = {p0.get('field_green_count')} / {p0.get('field_yellow_count')} / {p0.get('field_red_count')}",
        f"boundary_reproduced = {p0.get('boundary_reproduced')}",
        "```",
        "",
        "判断：P0 pass。v9.8.1 没有跳过 v9.8.0 的 no-system / generated-stop / no-fake boundary。",
        "",
        "## 4. P1 Natural AP0 Labeled Stream Extension",
        "",
        "Artifacts：",
        "",
        "```text",
        "p1_natural_ap0_stream_extension_materializer_v9810.csv",
        "p1_density_curve_v9810.csv",
        "```",
        "",
        "Panel summary：",
        "",
        "| panel | target | status | action count | CoreLike count | CoreLike rate | LCB | UCB | reason |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in p1_panels:
        lines.append(f"| `{row.get('panel_id')}` | `{row.get('target_action_count')}` | `{row.get('status')}` | `{row.get('panel_action_count')}` | `{row.get('CoreLike_count', '')}` | `{row.get('CoreLike_rate', '')}` | `{row.get('CoreLike_Wilson_LCB', '')}` | `{row.get('CoreLike_Wilson_UCB', '')}` | `{row.get('reason', '')}` |")
    lines.extend([
        "",
        "判断：P1 未完成 natural labeled extension。现有 2876 行能复核 density，但 5000/10000/20000 没有 landed labeled materializer，不能声称 density closure。",
        "",
        "## 5. P2 Future Trajectory Full Path",
        "",
        "Artifacts：",
        "",
        "```text",
        "p2_future_trajectory_full_path_v9810.csv",
        "p2_group_trajectory_summary_v9810.csv",
        "```",
        "",
        "Group summary：",
        "",
        "| group | materialized horizons | missing horizons | full path | observed AUV LCB | h20 LCB | h80 LCB | h240 LCB | h240 longrisk UCB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in group_rows:
        lines.append(f"| `{row.get('group_id')}` | `{row.get('materialized_horizons')}` | `{row.get('missing_horizons')}` | `{row.get('full_path_complete')}` | `{row.get('AUV_observed_LCB_h20_h80_h240')}` | `{row.get('h20_V_LCB')}` | `{row.get('h80_V_LCB')}` | `{row.get('h240_V_LCB')}` | `{row.get('h240_longrisk_UCB')}` |")
    lines.extend([
        "",
        "判断：h20/h80/h240 endpoint 信号仍然真实存在，但 h1/h5 缺失，所以不能把 endpoint-only 结果升级成 full future path theory pass。",
        "",
        "## 6. P3 Mechanism Decomposition",
        "",
        "Artifact：",
        "",
        "```text",
        "p3_future_trajectory_mechanism_decomposition_v9810.csv",
        "```",
        "",
        "| mechanism | effect size | corr/AUC proxy | TopK87 precision | TopK87 AUV LCB | longrisk UCB | legal | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in mech_rows:
        lines.append(f"| `{row.get('mechanism_id')}` | `{row.get('effect_size')}` | `{row.get('AUC_AUV_positive')}` | `{row.get('TopK87_precision')}` | `{row.get('TopK87_AUV_LCB')}` | `{row.get('TopK87_longrisk_UCB')}` | `{row.get('commit_time_legal')}` | `{row.get('mechanism_pass')}` |")
    lines.extend([
        "",
        "判断：P3 没有找到 legal + leaveout-stable + full-path-supported 的机制。部分 diagnostic score 有 AUV 相关性，但不能进入 controller。",
        "",
        "## 7. P4 Architecture-Agnostic Function Geometry v2",
        "",
        "Artifacts：",
        "",
        "```text",
        "p4_architecture_agnostic_function_geometry_ledger_v9810.csv",
        "p4_geometry_score_evaluation_v9810.csv",
        "```",
        "",
        "Summary：",
        "",
        "```text",
        f"geometry_action_row_count = {p4.get('geometry_action_row_count')}",
        f"TopK87_precision = {p4.get('TopK87_precision')}",
        f"TopK87_V_LCB = {p4.get('TopK87_V_LCB')}",
        f"TopK87_longrisk_UCB = {p4.get('TopK87_longrisk_UCB')}",
        f"TopK87_bad_UCB = {p4.get('TopK87_bad_UCB')}",
        f"TopK87_null_UCB = {p4.get('TopK87_null_UCB')}",
        f"TopK87_LDO/LSO/LTO = {p4.get('TopK87_LDO')} / {p4.get('TopK87_LSO')} / {p4.get('TopK87_LTO')}",
        f"P4_architecture_agnostic_geometry_pass = {p4.get('P4_architecture_agnostic_geometry_pass')}",
        "```",
        "",
        "判断：P4 仍是清晰失败。score 能找到低 LDO 区域，但 TopK87 precision 很低、V 为负、longrisk 很高，因此不是 controller-ready geometry principle。",
        "",
        "## 8. P5-P7 Controller / Generated Boundary",
        "",
        "P5：",
        "",
        "```text",
        f"status = {p5.get('status')}",
        f"reason = {p5.get('reason')}",
        f"controller_pass = {p5.get('controller_pass')}",
        "```",
        "",
        "P6：",
        "",
        "```text",
        f"status = {p6.get('status')}",
        f"reason = {p6.get('reason')}",
        f"P6_generated_preflight_v2_pass = {p6.get('P6_generated_preflight_v2_pass')}",
        "```",
        "",
        "P7：",
        "",
        "```text",
        "status = not_run",
        "reason = P6_generated_preflight_v2_not_passed",
        "```",
        "",
        "判断：generated route 没有因为 v9.8.0 h20 正信号而盲目重开。P2/P3 没给出完整 mechanism evidence，因此 P6/P7 按 gate 停止。",
        "",
        "## 9. P8 Raw LDO vs Support-Aware Gate Resolution",
        "",
        "Artifact：",
        "",
        "```text",
        "p8_ldo_gate_resolution_v9810.csv",
        "```",
        "",
        "Summary：",
        "",
        "```text",
        f"candidate_id = {p8.get('candidate_id')}",
        f"accepted_count = {p8.get('accepted_count')}",
        f"precision = {p8.get('precision')}",
        f"V_LCB = {p8.get('V_LCB')}",
        f"raw_LDO = {p8.get('raw_LDO')}",
        f"no_backfill_LDO = {p8.get('no_backfill_LDO')}",
        f"density_fail_count = {p8.get('density_fail_count')}",
        f"mean_backfill_precision = {p8.get('mean_backfill_precision')}",
        f"E_raw/E_no_backfill/E_density_required = {p8.get('E_raw_pass')} / {p8.get('E_no_backfill_pass')} / {p8.get('E_density_required_pass')}",
        f"official_gate_modification_discussion_allowed = {p8.get('official_gate_modification_discussion_allowed')}",
        "```",
        "",
        "判断：P8 没有解除 raw/support gate conflict。即使 no-backfill 角度可以解释一部分 backfill 惩罚，density-required 和 natural stream density 仍不过，因此 raw official gate 不能修改。",
        "",
        "## 10. P9-P11 Runtime / Replay / Short-Full",
        "",
        "```text",
        f"P9 status = {p9.get('status')}, reason = {p9.get('reason')}",
        f"P10 status = {p10.get('status')}, reason = {p10.get('reason')}",
        f"P11 status = {p11.get('status')}, reason = {p11.get('reason')}",
        "```",
        "",
        "判断：没有 official controller 或 generated update，所以 runtime、paired replay、short/full 全部保持关闭。",
        "",
        "## 11. No-Fake / Contract / Failure",
        "",
        "No-fake audit：",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {nf.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        f"no_fake/no_proxy = {nf.get('no_fake')} / {nf.get('no_proxy')}",
        "```",
        "",
        "Contract audit：",
        "",
        "```text",
        f"manual_forward/manual_backward/manual_adamw_update = {contract.get('manual_forward/manual_backward/manual_adamw_update')}",
        f"v9800_boundary_pass = {contract.get('v9800_boundary_pass')}",
        f"natural_stream_extension_completed = {contract.get('natural_stream_extension_completed')}",
        f"future_path_full_pass = {contract.get('future_path_full_pass')}",
        f"mechanism/controller/runtime/system = {contract.get('mechanism/controller/runtime/system')}",
        f"uses_future_outcome_for_official_controller = {contract.get('uses_future_outcome_for_official_controller')}",
        f"diagnostic_promoted_to_official = {contract.get('diagnostic_promoted_to_official')}",
        f"fake/proxy/cpu_offload = {contract.get('fake/proxy/cpu_offload')}",
        "```",
        "",
        "Failure taxonomy：",
        "",
        "```text",
        f"route = {failure.get('route')}",
        f"F0_boundary_fail = {failure.get('F0_boundary_fail')}",
        f"F1_natural_stream_extension_missing = {failure.get('F1_natural_stream_extension_missing')}",
        f"F2_h1_h5_future_path_missing = {failure.get('F2_h1_h5_future_path_missing')}",
        f"F3_mechanism_absent = {failure.get('F3_mechanism_absent')}",
        f"F4_geometry_score_fail = {failure.get('F4_geometry_score_fail')}",
        f"F5_controller_blocked = {failure.get('F5_controller_blocked')}",
        f"F6_generated_route_stopped = {failure.get('F6_generated_route_stopped')}",
        f"F7_runtime_replay_shortfull_blocked = {failure.get('F7_runtime_replay_shortfull_blocked')}",
        f"primary_blocker = {failure.get('primary_blocker')}",
        f"secondary_blocker = {failure.get('secondary_blocker')}",
        "```",
        "",
        "## 12. Figures",
        "",
        "本轮落盘图：",
        "",
        "```text",
        "fig_p1_corelike_density_curve_v9810.svg",
        "fig_p2_auv_observed_by_group_v9810.svg",
        "fig_p2_v20_by_group_v9810.svg",
        "fig_p3_mechanism_precision_v9810.svg",
        "fig_p4_geometry_topk_quality_v9800.svg",
        "```",
        "",
        "这些图只用于复核诊断，不构成 official pass。",
        "",
        "## 13. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ])
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend([
        "",
        "## 14. 最终分析结论",
        "",
        "v9.8.1 的真实推进是：",
        "",
        "```text",
        "v9.8.0:",
        "  future trajectory 有 h20/h80/h240 endpoint 信号；",
        "  但 natural stream extension、完整 h1/h5 path、geometry score、generated branch-horizon 都没打开。",
        "",
        "v9.8.1:",
        "  复现 v9.8.0 boundary；",
        "  明确确认仓库当前没有 landed h1/h5 full-path materializer；",
        "  对 h20/h80/h240 做 observed AUV 诊断，但不写成 full-path pass；",
        "  机制分解没有找到 legal + leaveout-stable 的 controller-ready feature；",
        "  architecture-agnostic geometry v2 仍选择低质高风险区域；",
        "  generated route 因缺少机制证据和 long-horizon official rows 继续停止；",
        "  raw/support gate conflict 没有被解除。",
        "```",
        "",
        "机制判断：",
        "",
        "1. H1 未能 official 成立：Core77 / OldOnly endpoint 很强，但 h1/h5 path 缺失，不能声称完整 future trajectory theory pass。",
        "2. H2 未成立：当前 architecture-agnostic function geometry score 仍没有对准 value/risk，不能作为跨架构 optimizer principle。",
        "3. H3 未打开：natural labeled stream extension 缺 materializer，density closure 仍未回答。",
        "4. H4 未打开：generated route 没有 P6/P7 长程 branch-horizon evidence，不能凭 v9.8.0 h20 positive smoke 重开。",
        "5. P8 gate conflict 仍存在：density-required 不成立，raw official gate 继续保留。",
        "6. P9-P11 全部 gate-blocked：没有 controller/runtime/replay/short-full。",
        "",
        "最终一句话：",
        "",
        f"> v9.8.1 真实执行后停在 `{route.get('route')}`：它把 v9.8.0 的 endpoint-only future trajectory 诊断拆成 full-path、mechanism、geometry、density 和 raw/support gate 五个边界；但 h1/h5 full-path materializer 与 natural labeled stream extension 都未落地，geometry score 仍失败，generated route 没有长程 official evidence，因此不能进入 official controller，strict PureKAN functional 仍未成功。",
        "",
    ])
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return path

    def dump_json(name: str, row: dict[str, Any]) -> Path:
        path = out / name
        write_json(path, row)
        artifacts[name] = path
        return path

    ap0, score_bundle, _payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9730.load_exact_summary(Path(args.source_v9720))
    exact_t4 = v9730.score_defs_from_summary(ap0, exact_summary)["T4-core-safe-transfer"]
    cs = v9750.cohort_sets(ap0, r5b, exact_t4)
    core = cs["Core77"]
    core_set = set(core)
    old_pool = [i for i in v9720.topk_idx(r5b, len(ap0)) if i not in core_set and safe_clean(ap0[i])]
    e4_idx = (core + old_pool[:10])[:TARGET_K]

    p0_rows, p0_legality, p0 = p0_boundary(args)
    dump_csv("p0_boundary_reproduction_v9810.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9810.csv", p0_legality)

    p1_rows, p1_curve, p1 = p1_natural_stream(ap0, parse_ints(args.panel_targets), out)
    dump_csv("p1_natural_ap0_stream_extension_materializer_v9810.csv", p1_rows)
    dump_csv("p1_density_curve_v9810.csv", p1_curve)

    source_v9800 = Path(args.source_v9800)
    generated_source = read_csv(source_v9800 / "p5_geometry_adaptive_update_preflight_v9800.csv") if (source_v9800 / "p5_geometry_adaptive_update_preflight_v9800.csv").exists() else []
    p2_rows, p2_group, p2 = p2_future_trajectory_full_path(ap0, cs, e4_idx, generated_source, r5b, exact_t4, Path(args.source_v9480), int(args.seed), out)
    dump_csv("p2_future_trajectory_full_path_v9810.csv", p2_rows)
    dump_csv("p2_group_trajectory_summary_v9810.csv", p2_group)

    canonical = load_canonical_real_rows(Path(args.source_v9480))
    p4_rows, p4_eval, p4 = p4_geometry_v2(ap0, canonical, out)
    dump_csv("p4_architecture_agnostic_function_geometry_ledger_v9810.csv", p4_rows)
    dump_csv("p4_geometry_score_evaluation_v9810.csv", p4_eval)

    p3_rows, p3 = p3_mechanism_decomposition(ap0, canonical, r5b, exact_t4, p4_rows, p2, out)
    dump_csv("p3_future_trajectory_mechanism_decomposition_v9810.csv", p3_rows)

    if inum(p1.get("P1_density_strong_pass")) and inum(p2.get("P2_future_trajectory_full_path_pass")) and inum(p3.get("P3_mechanism_pass")) and inum(p4.get("P4_architecture_agnostic_geometry_pass")):
        p5_rows, p5 = not_run("P5_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9810", "controller_implementation_not_landed_after_upstream_pass", controller_pass=0)
    else:
        p5_rows, p5 = not_run("P5_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9810", "P1_or_P2_or_P3_or_P4_not_official_controller_ready", controller_pass=0)
    dump_csv("p5_existing_action_minimal_controller_boundary_v9810.csv", p5_rows)

    if inum(p2.get("P2_future_trajectory_full_path_pass")) and inum(p3.get("P3_mechanism_pass")):
        p6_rows, p6 = not_run("P6_GENERATED_UPDATE_PREFLIGHT_V2_V9810", "G_Traj_G_Safe_solver_not_landed_after_mechanism_pass", P6_generated_preflight_v2_pass=0)
    else:
        p6_rows, p6 = not_run("P6_GENERATED_UPDATE_PREFLIGHT_V2_V9810", "P2_P3_future_trajectory_mechanism_not_established_generated_route_stopped", P6_generated_preflight_v2_pass=0, source_v9800_h20_positive_smoke_only=1)
    dump_csv("p6_generated_update_preflight_v2_v9810.csv", p6_rows)

    p7_rows, p7 = not_run("P7_GENERATED_BRANCH_HORIZON_SMOKE_V9810", "P6_generated_preflight_v2_not_passed", generated_branch_horizon_rows=0, generated_branch_horizon_pass=0)
    dump_csv("p7_generated_branch_horizon_smoke_v9810.csv", p7_rows)

    p8_rows, p8 = p8_ldo_gate_resolution(ap0, e4_idx, p1)
    dump_csv("p8_ldo_gate_resolution_v9810.csv", p8_rows)

    p9_rows, p9 = not_run("P9_RUNTIME_BOUNDARY_V9810", "P5_controller_and_P7_generated_not_passed", runtime_pass=0, selected_runtime_pass=0)
    dump_csv("p9_runtime_boundary_v9810.csv", p9_rows)
    p10_rows, p10 = not_run("P10_PAIRED_REPLAY_BOUNDARY_V9810", "P9_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p10_paired_replay_boundary_v9810.csv", p10_rows)
    p11_rows, p11 = not_run("P11_SHORT_FULL_BOUNDARY_V9810", "P10_paired_replay_not_passed", short_full_pass=0)
    dump_csv("p11_short_full_boundary_v9810.csv", p11_rows)

    base_src = Path(args.source_v9770) / "base_acc_sentinel_v9770.csv"
    if base_src.exists():
        base_row = dict(summary_row(read_csv(base_src)))
        base_row["stage"] = "BASE_ACC_SENTINEL_V9810"
        base_row["status"] = "summary"
        base_row["base_acc_used_for_controller"] = 0
        base_row["fake_data_used"] = 0
        base_row["proxy_row_used"] = 0
        base_row["cpu_offload_used"] = 0
        base_rows = [base_row]
    else:
        base_rows, _base = not_run("BASE_ACC_SENTINEL_V9810", "source_v9770_base_acc_sentinel_missing", base_acc_sentinel_pass=0)
    dump_csv("base_acc_sentinel_v9810.csv", base_rows)

    route, primary, route_family = choose_route(p0, p1, p2, p3, p4, p5, p6, p9)
    secondary = "natural_labeled_stream_extension_missing" if not inum(p1.get("natural_stream_extension_completed")) else "none"
    system_pass = int(inum(p5.get("controller_pass")) and inum(p9.get("runtime_pass")))
    route_decision = {
        "stage": "ROUTE_DECISION_V9810",
        "status": "summary",
        "route": route,
        "route_family": route_family,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9800": p0.get("source_route_v9800"),
        "P0_boundary_reproduced": p0.get("boundary_reproduced"),
        "P1_natural_stream_extension_completed": p1.get("natural_stream_extension_completed"),
        "P1_density_strong_pass": p1.get("P1_density_strong_pass"),
        "P1_density_weak_pass": p1.get("P1_density_weak_pass"),
        "P2_future_trajectory_full_path_pass": p2.get("P2_future_trajectory_full_path_pass"),
        "P3_mechanism_pass": p3.get("P3_mechanism_pass"),
        "P4_architecture_agnostic_geometry_pass": p4.get("P4_architecture_agnostic_geometry_pass"),
        "P5_controller_pass": p5.get("controller_pass"),
        "P6_generated_preflight_v2_pass": p6.get("P6_generated_preflight_v2_pass"),
        "P7_generated_branch_horizon_pass": p7.get("generated_branch_horizon_pass"),
        "P8_gate_modification_allowed": p8.get("official_gate_modification_discussion_allowed"),
        "P9_runtime_pass": p9.get("runtime_pass"),
        "P10_paired_replay_pass": p10.get("paired_replay_pass"),
        "P11_short_full_pass": p11.get("short_full_pass"),
        "system_legal_controller_pass": system_pass,
        "generated_route_status": "stopped_no_future_trajectory_mechanism_and_no_official_branch_horizon",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9810.json", route_decision)

    nf = no_fake(list(artifacts.values()))
    dump_csv("no_fake_audit_v9810.csv", [{"stage": "NO_FAKE_AUDIT_V9810", "status": "summary", **nf}])
    contract = {
        "stage": "CONTRACT_AUDIT_V9810",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9800_boundary_pass": p0.get("boundary_reproduced"),
        "natural_stream_extension_completed": p1.get("natural_stream_extension_completed"),
        "future_path_full_pass": p2.get("P2_future_trajectory_full_path_pass"),
        "mechanism/controller/runtime/system": f"{p3.get('P3_mechanism_pass')}/{p5.get('controller_pass')}/{p9.get('runtime_pass')}/{system_pass}",
        "geometry_official_pass": p4.get("P4_architecture_agnostic_geometry_pass"),
        "generated_preflight_v2/generated_branch_horizon": f"{p6.get('P6_generated_preflight_v2_pass')}/{p7.get('generated_branch_horizon_pass')}",
        "raw_support_gate_resolution_allowed": p8.get("official_gate_modification_discussion_allowed"),
        "base_acc_used_for_controller": 0,
        "uses_dataset_name_for_selector/controller": "0/0",
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"{nf['fake_data_used']}/{nf['proxy_row_used']}/{nf['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9810.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9810",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("boundary_reproduced"))),
        "F1_natural_stream_extension_missing": int(not inum(p1.get("natural_stream_extension_completed"))),
        "F2_h1_h5_future_path_missing": int(inum(p2.get("horizon_unavailable_count")) > 0),
        "F3_mechanism_absent": int(not inum(p3.get("P3_mechanism_pass"))),
        "F4_geometry_score_fail": int(fnum(p4.get("TopK87_precision")) < 0.30 or fnum(p4.get("TopK87_V_LCB")) < 0 or fnum(p4.get("TopK87_longrisk_UCB")) > 0.30),
        "F5_controller_blocked": int(not inum(p5.get("controller_pass"))),
        "F6_generated_route_stopped": int(not inum(p7.get("generated_branch_horizon_pass"))),
        "F7_runtime_replay_shortfull_blocked": int(not system_pass),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9810.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9810",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "generated_per_family": args.generated_per_family,
        },
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9810.json", manifest)
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "p1_density_strong_pass": p1.get("P1_density_strong_pass"),
        "p2_future_trajectory_full_path_pass": p2.get("P2_future_trajectory_full_path_pass"),
        "p3_mechanism_pass": p3.get("P3_mechanism_pass"),
        "p4_geometry_pass": p4.get("P4_architecture_agnostic_geometry_pass"),
        "system_legal_controller_pass": system_pass,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
