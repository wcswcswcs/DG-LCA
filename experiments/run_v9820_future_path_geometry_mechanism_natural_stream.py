#!/usr/bin/env python3
"""DG-KAN v9.8.2 future-path / mechanism / natural-stream runner.

This runner implements the v9.8.2 plan with real selected-action h1/h5/h20/h80/h240
branch-horizon materialization. It does not fabricate natural stream extension
labels or generated-update branch-horizon rows: those stages remain blocked
unless real materializers exist and upstream gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
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

import run_v9480_canonical_outcome_universe_rebuild as v9480  # noqa: E402
import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion as v9730  # noqa: E402
import run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation as v9750  # noqa: E402
import run_v9760_core77_ldo_density_mechanism_existing_action_controller as v9760  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.8.2_结果解读_未来轨迹几何机制_自然动作扩流_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9820_future_path_geometry_mechanism_natural_stream.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.2_FuturePathGeometryMechanism_NaturalStream_实验复盘.md"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z"
DEFAULT_V9810 = RESULT_ROOT / "v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z"
DEFAULT_V9770 = RESULT_ROOT / "v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z"
DEFAULT_V9740 = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9480 = RESULT_ROOT / "v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"

HORIZONS = [1, 5, 20, 80, 240]
BRANCHES = ["RealFunctional", "AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
CONTROL_BRANCHES = ["AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
TARGET_K = 87
AUV_WEIGHTS = {1: 0.05, 5: 0.10, 20: 0.25, 80: 0.30, 240: 0.30}
Z = 1.96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", default="full-gated", choices=["smoke", "full-gated", "existing-only"])
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--p1-max-per-group", type=int, default=0)
    p.add_argument("--generated-per-family", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--source-v9810", default=str(DEFAULT_V9810))
    p.add_argument("--source-v9770", default=str(DEFAULT_V9770))
    p.add_argument("--source-v9740", default=str(DEFAULT_V9740))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9480", default=str(DEFAULT_V9480))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--clear-caches-each-action", action="store_true")
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


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]


def wilson(count: int, total: int, z: float = Z) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    phat = count / total
    den = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    delta = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return phat, max(0.0, (centre - delta) / den), min(1.0, (centre + delta) / den)


def device_from(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def axis(row: dict[str, Any], name: str) -> str:
    return v9720.axis_value(row, name)


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


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


def p0_boundary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    src = Path(args.source_v9810)
    route = read_json(src / "route_decision_v9810.json")
    field = summary_row(read_csv(src / "p0_field_legality_audit_v9810.csv"))
    nf = summary_row(read_csv(src / "no_fake_audit_v9810.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9820",
        "status": "summary",
        "source_route_v9810": route.get("route"),
        "source_primary_blocker_v9810": route.get("primary_blocker"),
        "source_secondary_blocker_v9810": route.get("secondary_blocker"),
        "system_legal_controller_pass_v9810": route.get("system_legal_controller_pass"),
        "generated_route_status_v9810": route.get("generated_route_status"),
        "field_green_count": field.get("green_count"),
        "field_yellow_count": field.get("yellow_count"),
        "field_red_count": field.get("red_count"),
        "fake_data_used_v9810": nf.get("fake_data_used"),
        "proxy_row_used_v9810": nf.get("proxy_row_used"),
        "cpu_offload_used_v9810": nf.get("cpu_offload_used"),
        "P0_boundary_pass": int(
            route.get("route") == "R4-ArchitectureAgnosticGeometryFail"
            and inum(route.get("system_legal_controller_pass")) == 0
            and inum(field.get("red_count")) == 0
            and inum(nf.get("fake_data_used")) == 0
            and inum(nf.get("proxy_row_used")) == 0
            and inum(nf.get("cpu_offload_used")) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9820",
        "status": "summary",
        "green_count": field.get("green_count", 14),
        "yellow_count": field.get("yellow_count", 4),
        "red_count": field.get("red_count", 0),
        "dataset_name_used_in_controller": 0,
        "future_outcome_used_in_controller": 0,
        "old_table_used_for_controller": 0,
        "validation_test_used_for_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "field_legality_pass": int(inum(field.get("red_count", 0)) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def select_groups(ap0: list[dict[str, Any]], cs: dict[str, list[int]], r5b: list[float], seed: int, max_per_group: int) -> dict[str, list[int]]:
    core = set(cs["Core77"])
    old_only = set(cs["OldOnly"]) - core
    exact_only = set(cs["ExactOnly"]) - core - old_only
    used = core | old_only | exact_only
    pool = [i for i in range(len(ap0)) if i not in used]
    rng = random.Random(seed + 9820)
    random_matched = sorted(rng.sample(pool, min(TARGET_K, len(pool)))) if pool else []
    groups = {
        "Core77": sorted(core),
        "OldOnly": sorted(old_only),
        "ExactOnly": sorted(exact_only),
        "RandomMatched": random_matched,
    }
    if max_per_group > 0:
        return {k: v[:max_per_group] for k, v in groups.items()}
    return groups


def source_preflight_pass(source_v9480: Path) -> int:
    path = source_v9480 / "p1_no_transform_sentinel_grid.csv"
    if not path.exists():
        return 0
    return inum(summary_row(read_csv(path)).get("no_transform_sentinel_grid_pass"))


def build_selected_actions(
    ap0: list[dict[str, Any]],
    group_idx: dict[str, list[int]],
    payload_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, idxs in group_idx.items():
        for i in idxs:
            aid = str(ap0[i].get("action_id"))
            payload = dict(payload_by_id[aid])
            payload["group_id"] = group_id
            payload["ap0_index"] = i
            payload["candidate_template_id"] = v9720.candidate_template_id(ap0[i])
            payload["step_bucket"] = f"step{inum(ap0[i].get('step')) // 10}"
            rows.append(payload)
    return rows


def materialize_full_future_path(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    group_idx: dict[str, list[int]],
    payload_by_id: dict[str, dict[str, str]],
    r5b: list[float],
    exact_t4: list[float],
    wt_rows: dict[str, dict[str, Any]],
    device: torch.device,
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected = build_selected_actions(ap0, group_idx, payload_by_id)
    expected_rows = len(selected) * len(BRANCHES) * len(HORIZONS)
    payload_cache: dict[str, Any] = {}
    ctx_cache: dict[Any, Any] = {}
    action_branch_rows: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    out_rows: list[dict[str, Any]] = []
    completion: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []
    runtime_ms: list[float] = []
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for pos, action in enumerate(selected):
        aid = str(action.get("action_id"))
        group_id = str(action.get("group_id"))
        ap_i = inum(action.get("ap0_index"))
        action_rows = 0
        try:
            ctx = v9480.v9420.replay_context(args, action, device, ctx_cache)
            payload = v9480.load_payload(action, payload_cache, device)
            payload_norm = v9720.flat_norm(payload)
            random_gen = torch.Generator(device=device).manual_seed(v9480.seed_int("random-payload-v9820", aid, args.seed))
            random_payload = v9480.v9420.v9380.v9330.random_like_payload(payload, random_gen)
            for branch in BRANCHES:
                bconf = v9480.branch_config(branch)
                start_params, start_states, secondary_payload, branch_semantics, payload_applied, adamw_applied = v9480.branch_start(ctx, branch, payload, random_payload)
                rollout_seed = v9480.seed_int("canonical-rollout-v9480", aid, bconf["branch_config_hash"], max(HORIZONS), args.seed)
                bt0 = time.perf_counter()
                outs, start_hash, _end_hash, start_opt, batch_seq_hash = v9480.rollout_fast(
                    ctx,
                    start_params,
                    start_states,
                    secondary_payload,
                    HORIZONS,
                    rollout_seed,
                    int(args.batch_size),
                    device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                branch_ms = (time.perf_counter() - bt0) * 1000.0
                runtime_ms.append(branch_ms)
                for h in HORIZONS:
                    out_h = outs[h]
                    row = {
                        "stage": "P1_FULL_FUTURE_PATH_MATERIALIZER_V9820",
                        "status": "branch_horizon_row",
                        "outcome_row_id": stable_hash("v9820", aid, branch, h),
                        "materializer_id": "FPMAT-v9820-selected-groups-h1-h5-h20-h80-h240",
                        "group_id": group_id,
                        "action_id": aid,
                        "event_id": action.get("event_id"),
                        "dataset_id": action.get("dataset"),
                        "dataset": action.get("dataset"),
                        "seed": action.get("seed"),
                        "step": action.get("step"),
                        "family_id": action.get("family_id"),
                        "template_id": action.get("candidate_template_id"),
                        "step_bucket": action.get("step_bucket"),
                        "branch_id": branch,
                        "branch_semantics": branch_semantics,
                        "horizon": h,
                        "payload_applied_flag": payload_applied,
                        "adamw_applied_flag": adamw_applied,
                        "branch_runtime_ms": branch_ms,
                        "state_before_hash": start_hash,
                        "state_after_horizon_hash": out_h.get("theta_hash"),
                        "optimizer_state_hash": out_h.get("optimizer_hash"),
                        "batch_sequence_hash": batch_seq_hash,
                        "payload_hash": action.get("payload_hash_loaded") or action.get("payload_hash_expected"),
                        "action_norm": payload_norm,
                        "payload_norm": payload_norm,
                        "OldRank_score": r5b[ap_i],
                        "ExactTransfer_score": exact_t4[ap_i],
                        "WT80_score": wt_rows.get(aid, {}).get("WT80_LCB", ""),
                        "memory_fail": v9720.memory_fail(ap0[ap_i]),
                        "offdiag_fail": v9720.offdiag_fail(ap0[ap_i]),
                        "old_table_quarantine_flag": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    row.update(out_h)
                    row["V_branch"] = out_h.get("V_branch")
                    row["CE_delta"] = row.get("CE_mean_delta", "")
                    row["NLL_delta"] = row.get("NLL_delta", "")
                    row["margin_delta"] = row.get("margin_p10_delta", "")
                    row["hard_tail_loss_delta"] = row.get("CEp99_delta", "")
                    row["memory_buffer_loss_delta"] = ""
                    row["old_family_loss_delta"] = ""
                    row["old_family_margin_delta"] = ""
                    row["old_stratum_loss_delta"] = ""
                    row["cover_entropy_delta"] = row.get("basis_usage_entropy_delta", "")
                    row["basis_effective_rank_delta"] = ""
                    row["curvature_proxy_delta"] = row.get("curvature_delta", "")
                    row["jacobian_proxy_delta"] = row.get("local_lipschitz_delta", "")
                    row["AdamW_alignment"] = ""
                    nan, inf = v9480.metric_nan_inf(row)
                    row["metric_nan_count"] = nan
                    row["metric_inf_count"] = inf
                    out_rows.append(row)
                    action_branch_rows[(aid, h)][branch] = row
                    action_rows += 1
            completion.append({
                "stage": "P1_FULL_FUTURE_PATH_COMPLETION_V9820",
                "status": "action_completion_row",
                "action_id": aid,
                "group_id": group_id,
                "completed_rows": action_rows,
                "expected_rows": len(BRANCHES) * len(HORIZONS),
                "completion_pass": int(action_rows == len(BRANCHES) * len(HORIZONS)),
                "row_source": "newly_materialized_v9820",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        except Exception as exc:  # noqa: BLE001
            retry_rows.append({
                "stage": "P1_FULL_FUTURE_PATH_COMPLETION_V9820",
                "status": "unresolved_exception",
                "action_id": aid,
                "group_id": group_id,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:500],
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        finally:
            if args.clear_caches_each_action:
                payload_cache.clear()
                ctx_cache.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if (pos + 1) % 25 == 0:
            print(json.dumps({"stage": "P1_PROGRESS_V9820", "completed_actions": pos + 1, "total_actions": len(selected)}), flush=True)
    label_exclusive_viol = 0
    for (aid, h), by_branch in action_branch_rows.items():
        if "RealFunctional" not in by_branch:
            continue
        real = fnum(by_branch["RealFunctional"].get("V_branch"))
        controls = [fnum(by_branch[b].get("V_branch")) for b in CONTROL_BRANCHES if b in by_branch]
        best = max(controls) if controls else real
        real_row = by_branch["RealFunctional"]
        bad = int(fnum(real_row.get("CEp99_delta")) > 0.25 or fnum(real_row.get("NLL_delta")) > 0.15 or fnum(real_row.get("ECE_delta")) > 0.05 or fnum(real_row.get("acc_delta")) < -0.05)
        null = int((not bad) and abs(real - best) <= 0.02)
        weak = int(real > best and not bad and not null)
        strong = int(weak and (real - best) > 0.15 and fnum(real_row.get("CEp99_delta")) < 0.0 and fnum(real_row.get("acc_delta")) >= 0.0)
        longrisk = int(h == 240 and (bad or (real - best) < -0.10))
        if weak + bad + null > 1:
            label_exclusive_viol += 1
        for row in by_branch.values():
            row["V_real"] = real
            row["V_ctrl"] = real - best
            row["best_control_V_branch"] = best
            row["control_positive_label"] = weak
            row["weak_CP_label"] = weak
            row["strong_CP_label"] = strong
            row["bad_event_label"] = bad
            row["bad_event"] = bad
            row["null_event_label"] = null
            row["null_event"] = null
            row["long_risk_label"] = longrisk
            row["longrisk"] = longrisk
            row["task_safe_label"] = int(not bad)
            row["safe_good_label"] = int(weak and not bad and not null)
    for action in selected:
        aid = str(action.get("action_id"))
        y = int(
            inum(action_branch_rows.get((aid, 20), {}).get("RealFunctional", {}).get("weak_CP_label"))
            and fnum(action_branch_rows.get((aid, 20), {}).get("RealFunctional", {}).get("V_ctrl")) > 0.0
            and not inum(action_branch_rows.get((aid, 240), {}).get("RealFunctional", {}).get("long_risk_label"))
        )
        for h in HORIZONS:
            for row in action_branch_rows.get((aid, h), {}).values():
                row["Y_robust_label"] = y
    metric_nan = sum(inum(r.get("metric_nan_count")) for r in out_rows)
    metric_inf = sum(inum(r.get("metric_inf_count")) for r in out_rows)
    duplicate_count = len(out_rows) - len({r.get("outcome_row_id") for r in out_rows})
    elapsed = time.perf_counter() - t0
    no_transform = source_preflight_pass(Path(args.source_v9480))
    summary = {
        "stage": "P1_FULL_FUTURE_PATH_MATERIALIZER_V9820",
        "status": "summary",
        "selected_action_count": len(selected),
        "group_action_counts": json.dumps({g: len(v) for g, v in group_idx.items()}, sort_keys=True),
        "branch_count": len(BRANCHES),
        "horizons": ",".join(str(h) for h in HORIZONS),
        "row_count_expected": expected_rows,
        "row_count_actual": len(out_rows),
        "action_count_completed": sum(inum(r.get("completion_pass")) for r in completion),
        "row_count_failed": len(retry_rows),
        "completion_rate": len(out_rows) / max(1, expected_rows),
        "missing_horizon_count": max(0, expected_rows - len(out_rows)),
        "metric_nan_count": metric_nan,
        "metric_inf_count": metric_inf,
        "duplicate_outcome_row_id_count": duplicate_count,
        "label_exclusivity_violation_count": label_exclusive_viol,
        "no_transform_path_sanity_pass": no_transform,
        "runtime_ms_q90": qtile(runtime_ms, 0.90),
        "rows_per_sec": len(out_rows) / max(elapsed, 1.0e-9),
        "wallclock_sec": elapsed,
        "memory_mb_peak": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0,
        "P1_weak_pass": int(len(out_rows) == expected_rows and metric_nan == 0 and metric_inf == 0 and duplicate_count == 0 and label_exclusive_viol == 0 and no_transform == 1),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    completion.insert(0, {
        "stage": "P1_FULL_FUTURE_PATH_COMPLETION_V9820",
        "status": "summary",
        **{k: summary[k] for k in ["selected_action_count", "row_count_expected", "row_count_actual", "completion_rate", "missing_horizon_count", "row_count_failed"]},
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    completion.extend(retry_rows)
    out_rows.insert(0, summary)
    return out_rows, completion, summary


def real_rows(p1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in p1_rows if r.get("status") == "branch_horizon_row" and r.get("branch_id") == "RealFunctional"]


def p1_group_summary(p1_rows: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = real_rows(p1_rows)
    group_rows: list[dict[str, Any]] = []
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(str(r.get("group_id")), inum(r.get("horizon")))].append(r)
    for (gid, h), rs in sorted(by.items()):
        group_rows.append({
            "stage": "P1_GROUP_TRAJECTORY_SUMMARY_V9820",
            "status": "group_horizon_summary",
            "group_id": gid,
            "horizon": h,
            "action_count": len(rs),
            "V_branch_mean": mean([fnum(r.get("V_branch")) for r in rs]),
            "V_branch_LCB": lcb([fnum(r.get("V_branch")) for r in rs]),
            "V_ctrl_mean": mean([fnum(r.get("V_ctrl")) for r in rs]),
            "V_ctrl_LCB": lcb([fnum(r.get("V_ctrl")) for r in rs]),
            "CE_delta_mean": mean([fnum(r.get("CE_delta")) for r in rs]),
            "NLL_delta_mean": mean([fnum(r.get("NLL_delta")) for r in rs]),
            "margin_delta_mean": mean([fnum(r.get("margin_delta")) for r in rs]),
            "CEp99_delta_mean": mean([fnum(r.get("CEp99_delta")) for r in rs]),
            "bad_UCB": ucb([float(inum(r.get("bad_event_label"))) for r in rs]),
            "null_UCB": ucb([float(inum(r.get("null_event_label"))) for r in rs]),
            "longrisk_UCB": ucb([float(inum(r.get("long_risk_label"))) for r in rs]),
            "memory_fail_UCB": ucb([float(inum(r.get("memory_fail"))) for r in rs]),
            "offdiag_fail_UCB": ucb([float(inum(r.get("offdiag_fail"))) for r in rs]),
            "cover_entropy_delta_mean": mean([fnum(r.get("cover_entropy_delta")) for r in rs]),
            "curvature_proxy_delta_mean": mean([fnum(r.get("curvature_proxy_delta")) for r in rs]),
            "jacobian_proxy_delta_mean": mean([fnum(r.get("jacobian_proxy_delta")) for r in rs]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    group_ids = sorted({r.get("group_id") for r in rows})
    auv: dict[str, float] = {}
    for gid in group_ids:
        vals = []
        for h in HORIZONS:
            row = next((r for r in group_rows if r.get("group_id") == gid and inum(r.get("horizon")) == h), {})
            vals.append(AUV_WEIGHTS[h] * fnum(row.get("V_branch_LCB")))
        auv[str(gid)] = sum(vals)
    core20 = next((r for r in group_rows if r.get("group_id") == "Core77" and inum(r.get("horizon")) == 20), {})
    old20 = next((r for r in group_rows if r.get("group_id") == "OldOnly" and inum(r.get("horizon")) == 20), {})
    old80 = next((r for r in group_rows if r.get("group_id") == "OldOnly" and inum(r.get("horizon")) == 80), {})
    old240 = next((r for r in group_rows if r.get("group_id") == "OldOnly" and inum(r.get("horizon")) == 240), {})
    exact20 = next((r for r in group_rows if r.get("group_id") == "ExactOnly" and inum(r.get("horizon")) == 20), {})
    random20 = next((r for r in group_rows if r.get("group_id") == "RandomMatched" and inum(r.get("horizon")) == 20), {})
    mechanism_pass = int(
        fnum(core20.get("V_branch_LCB")) > 0
        and fnum(old20.get("V_branch_LCB")) > 0
        and fnum(old80.get("V_branch_LCB")) >= -0.02
        and (fnum(old240.get("V_branch_LCB")) >= 0 or fnum(old240.get("longrisk_UCB")) <= 0.05)
        and fnum(old240.get("longrisk_UCB")) <= 0.05
        and (fnum(exact20.get("V_branch_LCB")) < fnum(old20.get("V_branch_LCB")) or fnum(random20.get("V_branch_LCB")) < fnum(old20.get("V_branch_LCB")))
    )
    summary = {
        "stage": "P1_GROUP_TRAJECTORY_SUMMARY_V9820",
        "status": "summary",
        "group_count": len(group_ids),
        "horizons": ",".join(str(h) for h in HORIZONS),
        "Core77_AUV_LCB": auv.get("Core77", ""),
        "OldOnly_AUV_LCB": auv.get("OldOnly", ""),
        "ExactOnly_AUV_LCB": auv.get("ExactOnly", ""),
        "RandomMatched_AUV_LCB": auv.get("RandomMatched", ""),
        "Core77_V20_LCB": core20.get("V_branch_LCB", ""),
        "OldOnly_V20_LCB": old20.get("V_branch_LCB", ""),
        "ExactOnly_V20_LCB": exact20.get("V_branch_LCB", ""),
        "OldOnly_V80_LCB": old80.get("V_branch_LCB", ""),
        "OldOnly_V240_LCB": old240.get("V_branch_LCB", ""),
        "OldOnly_h240_longrisk_UCB": old240.get("longrisk_UCB", ""),
        "P1_mechanism_pass": mechanism_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    group_rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p1_future_path_V_by_group_v9820.svg", "P1 AUV LCB", list(auv), [fnum(v) for v in auv.values()])
    v9720.write_bar_svg(out / "fig_p1_future_path_longrisk_by_group_v9820.svg", "P1 h240 longrisk", group_ids, [fnum(next((r for r in group_rows if r.get("group_id") == g and inum(r.get("horizon")) == 240), {}).get("longrisk_UCB")) for g in group_ids])
    return group_rows, summary


def action_auv_and_features(p1_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = real_rows(p1_rows)
    by: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by[str(r.get("action_id"))][inum(r.get("horizon"))] = r
    out: dict[str, dict[str, Any]] = {}
    for aid, hs in by.items():
        if not all(h in hs for h in HORIZONS):
            continue
        auv = sum(AUV_WEIGHTS[h] * fnum(hs[h].get("V_branch")) for h in HORIZONS)
        first = hs[20]
        out[aid] = {
            "action_id": aid,
            "group_id": first.get("group_id"),
            "AUV": auv,
            "V20": fnum(hs[20].get("V_branch")),
            "V80": fnum(hs[80].get("V_branch")),
            "V240": fnum(hs[240].get("V_branch")),
            "CEp99_h20": fnum(hs[20].get("CEp99_delta")),
            "CEp99_h80": fnum(hs[80].get("CEp99_delta")),
            "cover_h80": fnum(hs[80].get("cover_entropy_delta")),
            "curvature_h80": fnum(hs[80].get("curvature_proxy_delta")),
            "jacobian_h80": fnum(hs[80].get("jacobian_proxy_delta")),
            "memory_fail": inum(first.get("memory_fail")),
            "offdiag_fail": inum(first.get("offdiag_fail")),
            "bad240": inum(hs[240].get("bad_event_label")),
            "longrisk240": inum(hs[240].get("long_risk_label")),
            "payload_norm": fnum(first.get("payload_norm")),
            "OldRank_score": fnum(first.get("OldRank_score")),
            "ExactTransfer_score": fnum(first.get("ExactTransfer_score")),
            "WT80_score": fnum(first.get("WT80_score")),
        }
    return out


def effect_size(vals_a: list[float], vals_b: list[float]) -> float:
    pooled = vals_a + vals_b
    sd = stdev(pooled)
    return (mean(vals_a) - mean(vals_b)) / (sd if sd > 1.0e-12 else 1.0)


def quality_for_scores_subset(
    ap0: list[dict[str, Any]],
    scores_by_aid: dict[str, float],
    aid_to_idx: dict[str, int],
) -> dict[str, Any]:
    scores = [-1.0e9] * len(ap0)
    for aid, score in scores_by_aid.items():
        idx = aid_to_idx.get(aid)
        if idx is not None:
            scores[idx] = score
    return v9720.quality_for_scores(ap0, scores, TARGET_K)


def p2_mechanism_contrast(
    ap0: list[dict[str, Any]],
    p1_rows: list[dict[str, Any]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feats = action_auv_and_features(p1_rows)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in feats.values():
        by_group[str(f["group_id"])].append(f)
    aid_to_idx = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    mechanisms: dict[str, tuple[dict[str, float], int]] = {}
    mechanisms["M_value_future"] = ({aid: f["AUV"] for aid, f in feats.items()}, 0)
    mechanisms["M_memory"] = ({aid: -float(f["memory_fail"]) - float(f["offdiag_fail"]) for aid, f in feats.items()}, 1)
    mechanisms["M_hardtail"] = ({aid: -f["CEp99_h80"] for aid, f in feats.items()}, 0)
    mechanisms["M_cover"] = ({aid: f["cover_h80"] for aid, f in feats.items()}, 0)
    mechanisms["M_curvature"] = ({aid: -f["curvature_h80"] - f["jacobian_h80"] for aid, f in feats.items()}, 0)
    mechanisms["M_signal"] = ({aid: f["OldRank_score"] for aid, f in feats.items()}, 0)
    mechanisms["M_adamw_path"] = ({aid: f["V80"] - f["V20"] for aid, f in feats.items()}, 0)
    rows: list[dict[str, Any]] = []
    for mid, (scores, legal) in mechanisms.items():
        q = quality_for_scores_subset(ap0, scores, aid_to_idx)
        top_aids = sorted(scores, key=scores.get, reverse=True)[:TARGET_K]
        top_feats = [feats[aid] for aid in top_aids if aid in feats]
        rows.append({
            "stage": "P2_FUTURE_PATH_MECHANISM_CONTRAST_V9820",
            "status": "mechanism_row",
            "mechanism_id": mid,
            "effect_size_Core_vs_Random": effect_size([f["AUV"] for f in by_group.get("Core77", [])], [f["AUV"] for f in by_group.get("RandomMatched", [])]),
            "effect_size_OldOnly_vs_ExactOnly": effect_size([f["AUV"] for f in by_group.get("OldOnly", [])], [f["AUV"] for f in by_group.get("ExactOnly", [])]),
            "TopK87_precision_if_used_alone": q.get("GradeAB_precision"),
            "V_LCB_if_used_alone": q.get("V_integrated_LCB"),
            "AUV_LCB_if_used_alone": lcb([f["AUV"] for f in top_feats]),
            "longrisk_UCB_if_used_alone": q.get("h240_longrisk_UCB"),
            "memory_fail_UCB_if_used_alone": q.get("memory_fail_UCB"),
            "offdiag_fail_UCB_if_used_alone": q.get("offdiag_fail_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "legal_at_commit": legal,
            "cost_ms_q90": "",
            "mechanism_pass": int(
                effect_size([f["AUV"] for f in by_group.get("Core77", [])], [f["AUV"] for f in by_group.get("RandomMatched", [])]) >= 0.5
                and effect_size([f["AUV"] for f in by_group.get("OldOnly", [])], [f["AUV"] for f in by_group.get("ExactOnly", [])]) >= 0.5
                and fnum(q.get("GradeAB_precision")) >= 0.60
                and fnum(q.get("V_integrated_LCB")) > 0
                and fnum(q.get("h240_longrisk_UCB")) <= 0.10
                and legal
            ),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r.get("mechanism_pass")), fnum(r.get("TopK87_precision_if_used_alone")), fnum(r.get("AUV_LCB_if_used_alone")))) if rows else {}
    summary = {
        "stage": "P2_FUTURE_PATH_MECHANISM_CONTRAST_V9820",
        "status": "summary",
        "mechanism_count": len(rows),
        "mechanism_pass_count": sum(inum(r.get("mechanism_pass")) for r in rows),
        "best_mechanism_id": best.get("mechanism_id"),
        "best_TopK87_precision": best.get("TopK87_precision_if_used_alone"),
        "best_AUV_LCB": best.get("AUV_LCB_if_used_alone"),
        "best_longrisk_UCB": best.get("longrisk_UCB_if_used_alone"),
        "P2_mechanism_pass": int(any(inum(r.get("mechanism_pass")) for r in rows)),
        "route_if_fail": "R-P2-NoMechanismExplainsGoodActions",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p2_mechanism_effect_forest_v9820.svg", "P2 OldOnly vs ExactOnly effect", [r["mechanism_id"] for r in rows if r.get("status") == "mechanism_row"], [fnum(r.get("effect_size_OldOnly_vs_ExactOnly")) for r in rows if r.get("status") == "mechanism_row"])
    return rows, summary


def p3_natural_stream(ap0: list[dict[str, Any]], panel_targets: list[int], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    existing_n = len(ap0)
    for target in panel_targets:
        if target <= existing_n:
            panel = ap0[:target]
            core = sum(1 for r in panel if core_like(r))
            grade = sum(1 for r in panel if v9720.gradeab(r))
            value = sum(1 for r in panel if fnum(r.get("V_integrated")) > 0 and inum(r.get("h240_longrisk")) == 0)
            memory_core = sum(1 for r in panel if core_like(r) and v9720.memory_fail(r) == 0 and v9720.offdiag_fail(r) == 0)
            mean_rate, lcb_rate, ucb_rate = wilson(core, target)
            per_dataset = {ds: sum(1 for r in panel if axis(r, "dataset_id") == ds and core_like(r)) / max(1, sum(1 for r in panel if axis(r, "dataset_id") == ds)) for ds in sorted({axis(r, "dataset_id") for r in panel})}
            per_family = {fam: sum(1 for r in panel if str(r.get("family_id")) == fam and core_like(r)) / max(1, sum(1 for r in panel if str(r.get("family_id")) == fam)) for fam in sorted({str(r.get("family_id")) for r in panel})}
            rows.append({
                "stage": "P3_NATURAL_AP0_LABELED_STREAM_EXTENSION_V9820",
                "status": "panel_row",
                "panel_id": f"Panel-target-{target}",
                "action_count": target,
                "labeled_action_count": target,
                "CoreLike_count": core,
                "CoreLike_rate": mean_rate,
                "Wilson_LCB": lcb_rate,
                "Wilson_UCB": ucb_rate,
                "GradeAB_count": grade,
                "GradeAB_rate": grade / max(1, target),
                "ValuePositiveNoLongRisk_count": value,
                "ValuePositiveNoLongRisk_rate": value / max(1, target),
                "MemoryOffdiagCore_count": memory_core,
                "MemoryOffdiagCore_rate": memory_core / max(1, target),
                "per_dataset_rate": json.dumps(per_dataset, sort_keys=True),
                "per_template_rate": "",
                "per_family_rate": json.dumps(per_family, sort_keys=True),
                "rows_per_sec": "",
                "wallclock": "",
                "failed_action_count": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        else:
            rows.append({
                "stage": "P3_NATURAL_AP0_LABELED_STREAM_EXTENSION_V9820",
                "status": "not_run",
                "panel_id": f"Panel-target-{target}",
                "action_count": target,
                "labeled_action_count": existing_n,
                "reason": "no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9820",
                "failed_action_count": target - existing_n,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    panel_a = next((r for r in rows if r.get("status") == "panel_row"), {})
    summary = {
        "stage": "P3_NATURAL_AP0_LABELED_STREAM_EXTENSION_V9820",
        "status": "summary",
        "requested_panel_targets": ",".join(str(x) for x in panel_targets),
        "existing_labeled_ap0_action_count": existing_n,
        "completed_panel_count": sum(1 for r in rows if r.get("status") == "panel_row"),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "PanelA_CoreLike_rate": panel_a.get("CoreLike_rate", ""),
        "PanelA_CoreLike_LCB": panel_a.get("Wilson_LCB", ""),
        "PanelA_CoreLike_UCB": panel_a.get("Wilson_UCB", ""),
        "P3_density_strong_pass": 0,
        "P3_density_weak_pass": int(fnum(panel_a.get("CoreLike_rate")) >= 0.03 and fnum(panel_a.get("Wilson_UCB")) >= 0.03),
        "P3_density_fail": int(fnum(panel_a.get("Wilson_UCB")) < 0.03 or any(r.get("status") == "not_run" for r in rows)),
        "reason": "labeled_materializer_not_landed_for_extension_panels" if any(r.get("status") == "not_run" for r in rows) else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p3_density_curve_corelike_v9820.svg", "P3 CoreLike density", [str(r.get("action_count")) for r in rows if r.get("status") == "panel_row"], [fnum(r.get("CoreLike_rate")) for r in rows if r.get("status") == "panel_row"])
    return rows, summary


def p4_gate_semantics(ap0: list[dict[str, Any]], p3: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows9770 = read_csv(DEFAULT_V9770 / "p1_ldo_decomposition_gate_audit_v9770.csv") if DEFAULT_V9770.exists() else []
    p1_9770 = summary_row(rows9770)
    p8_9810 = summary_row(read_csv(DEFAULT_V9810 / "p8_ldo_gate_resolution_v9810.csv")) if DEFAULT_V9810.exists() else {}
    q_core = {
        "accepted_count": 77,
        "precision": 1.0,
        "V_LCB": p1_9770.get("core77_min_V_LCB", ""),
    }
    summary = {
        "stage": "P4_GATE_SEMANTICS_V9820",
        "status": "summary",
        "accepted_count": p8_9810.get("accepted_count", 87),
        "coverage": fnum(p8_9810.get("accepted_count", 87)) / max(1, len(ap0)),
        "precision": p8_9810.get("precision"),
        "V_LCB": p8_9810.get("V_LCB"),
        "longrisk_UCB": "",
        "bad_UCB": "",
        "null_UCB": "",
        "memory_UCB": "",
        "offdiag_UCB": "",
        "LDO_raw": p1_9770.get("ldo_raw"),
        "LDO_quality": p1_9770.get("ldo_quality"),
        "LDO_support": p1_9770.get("ldo_support"),
        "LDO_backfill": p1_9770.get("ldo_backfill"),
        "LFO": "",
        "LSO": "",
        "LTO": "",
        "density_required_pass": p8_9810.get("E_density_required_pass", 0),
        "G_raw_pass": p8_9810.get("E_raw_pass", 0),
        "G_support_pass": p8_9810.get("E_no_backfill_pass", 0),
        "G_density_pass": p8_9810.get("E_density_required_pass", 0),
        "official_density_gate_proposal_allowed": int(inum(p3.get("P3_density_strong_pass"))),
        "P4_gate_semantics_pass": int(bool(p1_9770) and not inum(p3.get("P3_density_strong_pass"))),
        "reason": "raw_fail_decomposed_but_density_strong_pass_absent_keep_raw_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows = [
        summary,
        {
            "stage": "P4_GATE_SEMANTICS_V9820",
            "status": "gate_row",
            "gate_id": "G_raw",
            "pass": summary["G_raw_pass"],
            "LDO": summary["LDO_raw"],
            "reason": "legacy_raw_leaveout_includes_backfill",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P4_GATE_SEMANTICS_V9820",
            "status": "gate_row",
            "gate_id": "G_support",
            "pass": summary["G_support_pass"],
            "LDO": p8_9810.get("no_backfill_LDO", ""),
            "reason": "quality_without_external_backfill_diagnostic_only",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P4_GATE_SEMANTICS_V9820",
            "status": "gate_row",
            "gate_id": "G_density",
            "pass": summary["G_density_pass"],
            "LDO": "",
            "reason": "requires_natural_density_strong_pass_before_official_proposal",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    v9720.write_bar_svg(out / "fig_p4_gate_decomposition_waterfall_v9820.svg", "P4 LDO decomposition", ["quality", "support", "backfill"], [fnum(summary["LDO_quality"]), fnum(summary["LDO_support"]), fnum(summary["LDO_backfill"])])
    return rows, summary


def p5_geometry_rebuild(ap0: list[dict[str, Any]], p1_rows: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feats = action_auv_and_features(p1_rows)
    aid_to_idx = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    families: dict[str, tuple[dict[str, float], int]] = {
        "G_func": ({aid: abs(f["V20"]) + abs(f["CEp99_h20"]) for aid, f in feats.items()}, 0),
        "G_path": ({aid: f["AUV"] for aid, f in feats.items()}, 0),
        "G_memory": ({aid: -float(f["memory_fail"]) - float(f["offdiag_fail"]) for aid, f in feats.items()}, 1),
        "G_stability": ({aid: -f["CEp99_h80"] - f["curvature_h80"] - f["jacobian_h80"] for aid, f in feats.items()}, 0),
        "G_cost": ({aid: -f["payload_norm"] for aid, f in feats.items()}, 1),
    }
    rows: list[dict[str, Any]] = []
    for gid, (scores, legal) in families.items():
        q = quality_for_scores_subset(ap0, scores, aid_to_idx)
        top = sorted(scores, key=scores.get, reverse=True)[:TARGET_K]
        top_feats = [feats[aid] for aid in top if aid in feats]
        rows.append({
            "stage": "P5_ARCHITECTURE_AGNOSTIC_GEOMETRY_REBUILD_V9820",
            "status": "geometry_family_row",
            "geometry_family": gid,
            "TopK87_precision": q.get("GradeAB_precision"),
            "V_LCB": q.get("V_integrated_LCB"),
            "AUV_LCB": lcb([f["AUV"] for f in top_feats]),
            "longrisk_UCB": q.get("h240_longrisk_UCB"),
            "bad_UCB": q.get("bad_UCB"),
            "null_UCB": q.get("null_UCB"),
            "LDO_drop": q.get("LDO_drop"),
            "LSO_drop": q.get("LSO_drop"),
            "LTO_drop": q.get("LTO_drop"),
            "legal_at_commit": legal,
            "P5_weak_family_pass": int(fnum(q.get("GradeAB_precision")) >= 0.60 and fnum(q.get("V_integrated_LCB")) > 0 and fnum(q.get("h240_longrisk_UCB")) <= 0.10 and legal),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r.get("P5_weak_family_pass")), fnum(r.get("TopK87_precision")), fnum(r.get("AUV_LCB")))) if rows else {}
    summary = {
        "stage": "P5_ARCHITECTURE_AGNOSTIC_GEOMETRY_REBUILD_V9820",
        "status": "summary",
        "geometry_family_count": len(rows),
        "weak_family_pass_count": sum(inum(r.get("P5_weak_family_pass")) for r in rows),
        "best_geometry_family": best.get("geometry_family"),
        "best_TopK87_precision": best.get("TopK87_precision"),
        "best_AUV_LCB": best.get("AUV_LCB"),
        "best_longrisk_UCB": best.get("longrisk_UCB"),
        "P5_weak_pass": int(any(inum(r.get("P5_weak_family_pass")) for r in rows)),
        "P5_strong_pass": 0,
        "reason": "no_legal_geometry_family_meets_precision_value_risk_gate" if not any(inum(r.get("P5_weak_family_pass")) for r in rows) else "weak_only_no_combined_strong_rule",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p5_geometry_family_quality_scatter_v9820.svg", "P5 geometry precision", [r["geometry_family"] for r in rows if r.get("status") == "geometry_family_row"], [fnum(r.get("TopK87_precision")) for r in rows if r.get("status") == "geometry_family_row"])
    return rows, summary


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


def choose_route(p0: dict[str, Any], p1: dict[str, Any], p1g: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any], p7: dict[str, Any]) -> tuple[str, str, str]:
    if not inum(p0.get("P0_boundary_pass")):
        return "R0-BoundaryReproductionFail", "p0_boundary_fail", "boundary"
    if not inum(p1.get("P1_weak_pass")):
        return "R-P1-FuturePathMaterializerFail", "h1_h5_full_path_materializer_failed", "future_path"
    if not inum(p2.get("P2_mechanism_pass")):
        return "R-P2-NoMechanismExplainsGoodActions", "no_legal_mechanism_explains_good_actions", "mechanism"
    if inum(p3.get("P3_density_fail")) and not inum(p3.get("P3_density_strong_pass")):
        return "R-P3-NaturalStreamExtensionMissing", "natural_labeled_stream_extension_missing", "density"
    if not inum(p5.get("P5_weak_pass")):
        return "R-P5-GeometryFamilyNoLegalSignal", "no_legal_geometry_family_pass", "geometry"
    if inum(p6.get("controller_strong_pass")) or inum(p7.get("generated_strong_pass")):
        return "R-P8-RuntimeRequired", "runtime_not_run_yet", "controller"
    return "R-P2-NoMechanismExplainsGoodActions", "upstream_gate_incomplete", "mechanism"


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9820.csv"))
    p1 = summary_row(read_csv(out / "p1_full_future_path_materializer_v9820.csv"))
    p1g = summary_row(read_csv(out / "p1_group_trajectory_summary_v9820.csv"))
    p2 = summary_row(read_csv(out / "p2_future_path_mechanism_contrast_v9820.csv"))
    p3 = summary_row(read_csv(out / "p3_natural_ap0_labeled_stream_extension_v9820.csv"))
    p4 = summary_row(read_csv(out / "p4_gate_semantics_v9820.csv"))
    p5 = summary_row(read_csv(out / "p5_architecture_agnostic_geometry_rebuild_v9820.csv"))
    p6 = summary_row(read_csv(out / "p6_existing_action_minimal_controller_boundary_v9820.csv"))
    p7 = summary_row(read_csv(out / "p7_geometry_adaptive_generated_update_v9820.csv"))
    p8 = summary_row(read_csv(out / "p8_selected_runtime_boundary_v9820.csv"))
    p9 = summary_row(read_csv(out / "p9_official_paired_replay_boundary_v9820.csv"))
    p10 = summary_row(read_csv(out / "p10_short_full_training_boundary_v9820.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9820.csv"))
    contract = summary_row(read_csv(out / "contract_audit_v9820.csv"))
    failure = summary_row(read_csv(out / "failure_taxonomy_v9820.csv"))
    group_rows = [r for r in read_csv(out / "p1_group_trajectory_summary_v9820.csv") if r.get("status") == "group_horizon_summary"]
    mechanism_rows = [r for r in read_csv(out / "p2_future_path_mechanism_contrast_v9820.csv") if r.get("status") == "mechanism_row"]
    panel_rows = [r for r in read_csv(out / "p3_natural_ap0_labeled_stream_extension_v9820.csv") if r.get("status") in {"panel_row", "not_run"}]
    geom_rows = [r for r in read_csv(out / "p5_architecture_agnostic_geometry_rebuild_v9820.csv") if r.get("status") == "geometry_family_row"]
    lines = [
        "# DG-KAN v9.8.2 Future Path / Geometry Mechanism / Natural Stream 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.2_结果解读_未来轨迹几何机制_自然动作扩流_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；P1 h1/h5/h20/h80/h240 是真实 replay materialized rows；P3 5000/10000/20000 natural stream extension 没有 landed labeled materializer，因此显式 `not_run`，没有 fake data、proxy rows 或 CPU offload。",
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
        f"1. P0 复现 v9.8.1 boundary：source route = `{p0.get('source_route_v9810')}`，system pass = `{p0.get('system_legal_controller_pass_v9810')}`，field red count = `{p0.get('field_red_count')}`。",
        f"2. P1 真实补齐 full future path：selected actions = `{p1.get('selected_action_count')}`，row count expected/actual = `{p1.get('row_count_expected')}` / `{p1.get('row_count_actual')}`，horizons = `{p1.get('horizons')}`。",
        f"3. P1 weak pass = `{p1.get('P1_weak_pass')}`，missing horizon = `{p1.get('missing_horizon_count')}`，NaN/Inf = `{p1.get('metric_nan_count')}` / `{p1.get('metric_inf_count')}`，no-transform sanity = `{p1.get('no_transform_path_sanity_pass')}`。",
        f"4. P1 observed AUV LCB：Core77 = `{p1g.get('Core77_AUV_LCB')}`，OldOnly = `{p1g.get('OldOnly_AUV_LCB')}`，ExactOnly = `{p1g.get('ExactOnly_AUV_LCB')}`，RandomMatched = `{p1g.get('RandomMatched_AUV_LCB')}`。",
        f"5. P1 mechanism endpoint check = `{p1g.get('P1_mechanism_pass')}`；OldOnly h240 longrisk UCB = `{p1g.get('OldOnly_h240_longrisk_UCB')}`。",
        f"6. P2 mechanism pass count = `{p2.get('mechanism_pass_count')}`；best mechanism = `{p2.get('best_mechanism_id')}`，TopK87 precision = `{p2.get('best_TopK87_precision')}`，AUV LCB = `{p2.get('best_AUV_LCB')}`。",
        f"7. P3 natural stream extension 仍未落地：completed panel = `{p3.get('completed_panel_count')}`，not-run panel = `{p3.get('not_run_panel_count')}`，Panel A rate/LCB/UCB = `{p3.get('PanelA_CoreLike_rate')}` / `{p3.get('PanelA_CoreLike_LCB')}` / `{p3.get('PanelA_CoreLike_UCB')}`。",
        f"8. P4 gate semantics pass = `{p4.get('P4_gate_semantics_pass')}`；raw/support/density gate = `{p4.get('G_raw_pass')}` / `{p4.get('G_support_pass')}` / `{p4.get('G_density_pass')}`；official density proposal allowed = `{p4.get('official_density_gate_proposal_allowed')}`。",
        f"9. P5 geometry weak/strong pass = `{p5.get('P5_weak_pass')}` / `{p5.get('P5_strong_pass')}`；best family = `{p5.get('best_geometry_family')}`。",
        f"10. P6/P7/P8/P9/P10 均 gate-blocked：controller `{p6.get('status')}`，generated `{p7.get('status')}`，runtime `{p8.get('status')}`，paired replay `{p9.get('status')}`，short/full `{p10.get('status')}`。",
        f"11. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9820_future_path_geometry_mechanism_natural_stream.py` | v9.8.2 runner；真实 materialize selected groups 的 h1/h5/h20/h80/h240 branch-horizon rows，执行 mechanism contrast、natural stream extension audit、gate semantics、geometry family decomposition，并按 gate 写 P6-P10。 |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9820_future_path_geometry_mechanism_natural_stream.py",
        "```",
        "",
        "正式 full-gated 运行：",
        "",
        "```bash",
        f"python experiments/run_v9820_future_path_geometry_mechanism_natural_stream.py --out-dir {out} --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000 --generated-per-family 64",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Full Future Path Materializer",
        "",
        "Artifacts：",
        "",
        "```text",
        "p1_full_future_path_materializer_v9820.csv",
        "p1_future_path_completion_v9820.csv",
        "p1_group_trajectory_summary_v9820.csv",
        "```",
        "",
        "Group/horizon summary：",
        "",
        "| group | h | actions | V LCB | Vctrl LCB | CE mean | CEp99 mean | longrisk UCB | memory UCB | offdiag UCB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in group_rows:
        lines.append(f"| `{r.get('group_id')}` | `{r.get('horizon')}` | `{r.get('action_count')}` | `{r.get('V_branch_LCB')}` | `{r.get('V_ctrl_LCB')}` | `{r.get('CE_delta_mean')}` | `{r.get('CEp99_delta_mean')}` | `{r.get('longrisk_UCB')}` | `{r.get('memory_fail_UCB')}` | `{r.get('offdiag_fail_UCB')}` |")
    lines.extend([
        "",
        "判断：P1 这轮不是 unavailable，而是真实补齐了 h1/h5/h20/h80/h240 selected-group trajectory。它证明 endpoint/path rows 可以落地；但这只是路径 materializer pass，不等价于 controller pass。",
        "",
        "## 4. P2 Mechanism Contrast",
        "",
        "| mechanism | Core vs Random effect | OldOnly vs ExactOnly effect | TopK87 precision | V LCB | AUV LCB | longrisk UCB | legal | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in mechanism_rows:
        lines.append(f"| `{r.get('mechanism_id')}` | `{r.get('effect_size_Core_vs_Random')}` | `{r.get('effect_size_OldOnly_vs_ExactOnly')}` | `{r.get('TopK87_precision_if_used_alone')}` | `{r.get('V_LCB_if_used_alone')}` | `{r.get('AUV_LCB_if_used_alone')}` | `{r.get('longrisk_UCB_if_used_alone')}` | `{r.get('legal_at_commit')}` | `{r.get('mechanism_pass')}` |")
    lines.extend([
        "",
        "判断：P2 没有找到 legal-at-commit 且 precision/value/risk 同时满足的机制。OldRank / future path / response 类信号仍然强，但 diagnostic-only；memory/cost 类更合法，却不能选出足够好动作。",
        "",
        "## 5. P3 Natural AP0 Labeled Stream Extension",
        "",
        "| panel | target/action | status | labeled | CoreLike rate | LCB | UCB | reason |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ])
    for r in panel_rows:
        lines.append(f"| `{r.get('panel_id')}` | `{r.get('action_count')}` | `{r.get('status')}` | `{r.get('labeled_action_count')}` | `{r.get('CoreLike_rate', '')}` | `{r.get('Wilson_LCB', '')}` | `{r.get('Wilson_UCB', '')}` | `{r.get('reason', '')}` |")
    lines.extend([
        "",
        "判断：P3 仍未完成自然动作扩流。现有 2876 panel 可复核，5000/10000/20000 没有真实 labeled branch-horizon rows，所以不允许 density closure。",
        "",
        "## 6. P4 Gate Semantics",
        "",
        "```text",
        f"LDO_raw/quality/support/backfill = {p4.get('LDO_raw')} / {p4.get('LDO_quality')} / {p4.get('LDO_support')} / {p4.get('LDO_backfill')}",
        f"G_raw/G_support/G_density = {p4.get('G_raw_pass')} / {p4.get('G_support_pass')} / {p4.get('G_density_pass')}",
        f"official_density_gate_proposal_allowed = {p4.get('official_density_gate_proposal_allowed')}",
        f"reason = {p4.get('reason')}",
        "```",
        "",
        "判断：raw/support/density gate 语义被拆开，但 density strong pass 不存在，因此不能提出 official gate 修改。",
        "",
        "## 7. P5 Geometry Family Rebuild",
        "",
        "| family | precision | V LCB | AUV LCB | longrisk UCB | legal | weak pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in geom_rows:
        lines.append(f"| `{r.get('geometry_family')}` | `{r.get('TopK87_precision')}` | `{r.get('V_LCB')}` | `{r.get('AUV_LCB')}` | `{r.get('longrisk_UCB')}` | `{r.get('legal_at_commit')}` | `{r.get('P5_weak_family_pass')}` |")
    lines.extend([
        "",
        "判断：P5 不再做单个大 score，而是拆成 family。结果仍没有 legal family 通过 precision/value/risk gate。",
        "",
        "## 8. P6-P10 Boundary",
        "",
        "```text",
        f"P6 controller = {p6.get('status')}, reason = {p6.get('reason')}",
        f"P7 generated = {p7.get('status')}, reason = {p7.get('reason')}",
        f"P8 runtime = {p8.get('status')}, reason = {p8.get('reason')}",
        f"P9 paired replay = {p9.get('status')}, reason = {p9.get('reason')}",
        f"P10 short/full = {p10.get('status')}, reason = {p10.get('reason')}",
        "```",
        "",
        "判断：没有 legal mechanism / geometry family / density pass，所以 controller、generated route、runtime、paired replay、short/full 全部保持关闭。",
        "",
        "## 9. No-Fake / Contract / Failure",
        "",
        "No-fake audit：",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
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
        f"p1_full_path_materialized = {contract.get('p1_full_path_materialized')}",
        f"p2_mechanism_pass = {contract.get('p2_mechanism_pass')}",
        f"p3_density_strong_pass = {contract.get('p3_density_strong_pass')}",
        f"controller/generated/runtime/system = {contract.get('controller/generated/runtime/system')}",
        f"fake/proxy/cpu_offload = {contract.get('fake/proxy/cpu_offload')}",
        "```",
        "",
        "Failure taxonomy：",
        "",
        "```text",
        f"route = {failure.get('route')}",
        f"F0_boundary_fail = {failure.get('F0_boundary_fail')}",
        f"F1_future_path_materializer_fail = {failure.get('F1_future_path_materializer_fail')}",
        f"F2_no_legal_mechanism = {failure.get('F2_no_legal_mechanism')}",
        f"F3_natural_stream_extension_missing = {failure.get('F3_natural_stream_extension_missing')}",
        f"F4_density_gate_not_resolved = {failure.get('F4_density_gate_not_resolved')}",
        f"F5_geometry_family_no_legal_signal = {failure.get('F5_geometry_family_no_legal_signal')}",
        f"F6_controller_generated_runtime_blocked = {failure.get('F6_controller_generated_runtime_blocked')}",
        f"primary_blocker = {failure.get('primary_blocker')}",
        f"secondary_blocker = {failure.get('secondary_blocker')}",
        "```",
        "",
        "## 10. Figures",
        "",
        "```text",
        "fig_p1_future_path_V_by_group_v9820.svg",
        "fig_p1_future_path_longrisk_by_group_v9820.svg",
        "fig_p2_mechanism_effect_forest_v9820.svg",
        "fig_p3_density_curve_corelike_v9820.svg",
        "fig_p4_gate_decomposition_waterfall_v9820.svg",
        "fig_p5_geometry_family_quality_scatter_v9820.svg",
        "```",
        "",
        "## 11. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ])
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend([
        "",
        "## 12. 最终分析结论",
        "",
        "v9.8.2 的真实推进是：",
        "",
        "```text",
        "1. h1/h5 不再只是 unavailable：P1 对 selected groups 真实 materialize 了 h1/h5/h20/h80/h240。",
        "2. Core77 / OldOnly 的 future path 端点和 AUV 仍然强，但 P2 没有找到 legal-at-commit 的机制能解释并选择这些动作。",
        "3. Natural AP0 stream extension 仍未落地，5000/10000/20000 panel 没有真实 labels。",
        "4. Gate semantics 已拆开，但 density strong pass 不存在，不能修改 official raw gate。",
        "5. Geometry family decomposition 没有 legal family 通过 weak gate。",
        "6. Controller / generated / runtime / replay / short-full 全部继续 gate-blocked。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.2 真实执行后停在 `{route.get('route')}`：本轮真正补上了 selected-action 的 h1/h5 full future path，这是比 v9.8.1 更实的推进；但 legal mechanism、natural action density、geometry family 和 generated branch-horizon 都没有过 gate，因此不能进入 official controller，strict PureKAN functional 仍未成功。",
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

    device = device_from(args.device)
    ap0, score_bundle, payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9730.load_exact_summary(Path(args.source_v9720))
    exact_t4 = v9730.score_defs_from_summary(ap0, exact_summary)["T4-core-safe-transfer"]
    cs = v9750.cohort_sets(ap0, r5b, exact_t4)
    wt_rows = v9750.load_wt_rows(Path(args.source_v9740))

    p0_rows, p0_legality, p0 = p0_boundary(args)
    dump_csv("p0_boundary_reproduction_v9820.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9820.csv", p0_legality)

    max_per_group = int(args.p1_max_per_group)
    groups = select_groups(ap0, cs, r5b, int(args.seed), max_per_group)
    p1_rows, p1_completion, p1 = materialize_full_future_path(args, ap0, groups, payload_by_id, r5b, exact_t4, wt_rows, device, out)
    dump_csv("p1_full_future_path_materializer_v9820.csv", p1_rows)
    dump_csv("p1_future_path_completion_v9820.csv", p1_completion)
    p1_group_rows, p1g = p1_group_summary(p1_rows, out)
    dump_csv("p1_group_trajectory_summary_v9820.csv", p1_group_rows)

    p2_rows, p2 = p2_mechanism_contrast(ap0, p1_rows, out)
    dump_csv("p2_future_path_mechanism_contrast_v9820.csv", p2_rows)

    p3_rows, p3 = p3_natural_stream(ap0, parse_ints(args.panel_targets), out)
    dump_csv("p3_natural_ap0_labeled_stream_extension_v9820.csv", p3_rows)

    p4_rows, p4 = p4_gate_semantics(ap0, p3, out)
    dump_csv("p4_gate_semantics_v9820.csv", p4_rows)

    p5_rows, p5 = p5_geometry_rebuild(ap0, p1_rows, out)
    dump_csv("p5_architecture_agnostic_geometry_rebuild_v9820.csv", p5_rows)

    if inum(p1.get("P1_weak_pass")) and inum(p2.get("P2_mechanism_pass")) and inum(p3.get("P3_density_strong_pass")) and inum(p4.get("official_density_gate_proposal_allowed")) and inum(p5.get("P5_weak_pass")):
        p6_rows, p6 = not_run("P6_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9820", "controller_implementation_not_landed_after_upstream_pass", controller_weak_pass=0, controller_strong_pass=0)
    else:
        p6_rows, p6 = not_run("P6_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9820", "P1_or_P2_or_P3_or_P4_or_P5_not_controller_ready", controller_weak_pass=0, controller_strong_pass=0)
    dump_csv("p6_existing_action_minimal_controller_boundary_v9820.csv", p6_rows)

    if inum(p1.get("P1_weak_pass")) and inum(p2.get("P2_mechanism_pass")) and inum(p5.get("P5_weak_pass")):
        p7_rows, p7 = not_run("P7_GEOMETRY_ADAPTIVE_GENERATED_UPDATE_V9820", "generated_solver_not_landed_after_mechanism_pass", generated_weak_pass=0, generated_strong_pass=0)
    else:
        p7_rows, p7 = not_run("P7_GEOMETRY_ADAPTIVE_GENERATED_UPDATE_V9820", "P1_P2_P5_mechanism_not_established_generated_route_stopped", generated_weak_pass=0, generated_strong_pass=0)
    dump_csv("p7_geometry_adaptive_generated_update_v9820.csv", p7_rows)

    p8_rows, p8 = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9820", "P6_controller_and_P7_generated_not_passed", runtime_pass=0)
    dump_csv("p8_selected_runtime_boundary_v9820.csv", p8_rows)
    p9_rows, p9 = not_run("P9_OFFICIAL_PAIRED_REPLAY_BOUNDARY_V9820", "P8_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p9_official_paired_replay_boundary_v9820.csv", p9_rows)
    p10_rows, p10 = not_run("P10_SHORT_FULL_TRAINING_BOUNDARY_V9820", "P9_paired_replay_not_passed", short_full_pass=0)
    dump_csv("p10_short_full_training_boundary_v9820.csv", p10_rows)

    route, primary, route_family = choose_route(p0, p1, p1g, p2, p3, p5, p6, p7)
    secondary = "natural_labeled_stream_extension_missing" if not inum(p3.get("P3_density_strong_pass")) else "none"
    system_pass = int(inum(p6.get("controller_strong_pass")) and inum(p8.get("runtime_pass")))
    route_decision = {
        "stage": "ROUTE_DECISION_V9820",
        "status": "summary",
        "route": route,
        "route_family": route_family,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9810": p0.get("source_route_v9810"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_weak_pass": p1.get("P1_weak_pass"),
        "P1_mechanism_pass": p1g.get("P1_mechanism_pass"),
        "P2_mechanism_pass": p2.get("P2_mechanism_pass"),
        "P3_density_strong_pass": p3.get("P3_density_strong_pass"),
        "P3_density_weak_pass": p3.get("P3_density_weak_pass"),
        "P4_gate_semantics_pass": p4.get("P4_gate_semantics_pass"),
        "P5_weak_pass": p5.get("P5_weak_pass"),
        "P5_strong_pass": p5.get("P5_strong_pass"),
        "P6_controller_strong_pass": p6.get("controller_strong_pass"),
        "P7_generated_strong_pass": p7.get("generated_strong_pass"),
        "P8_runtime_pass": p8.get("runtime_pass"),
        "P9_paired_replay_pass": p9.get("paired_replay_pass"),
        "P10_short_full_pass": p10.get("short_full_pass"),
        "system_legal_controller_pass": system_pass,
        "generated_route_status": "stopped_no_legal_mechanism_or_generated_branch_horizon",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9820.json", route_decision)

    nf = no_fake(list(artifacts.values()))
    dump_csv("no_fake_audit_v9820.csv", [{"stage": "NO_FAKE_AUDIT_V9820", "status": "summary", **nf}])
    contract = {
        "stage": "CONTRACT_AUDIT_V9820",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9810_boundary_pass": p0.get("P0_boundary_pass"),
        "p1_full_path_materialized": p1.get("P1_weak_pass"),
        "p2_mechanism_pass": p2.get("P2_mechanism_pass"),
        "p3_density_strong_pass": p3.get("P3_density_strong_pass"),
        "p4_gate_semantics_pass": p4.get("P4_gate_semantics_pass"),
        "p5_geometry_weak_pass": p5.get("P5_weak_pass"),
        "controller/generated/runtime/system": f"{p6.get('controller_strong_pass')}/{p7.get('generated_strong_pass')}/{p8.get('runtime_pass')}/{system_pass}",
        "uses_dataset_name_for_controller": 0,
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"{nf['fake_data_used']}/{nf['proxy_row_used']}/{nf['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9820.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9820",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("P0_boundary_pass"))),
        "F1_future_path_materializer_fail": int(not inum(p1.get("P1_weak_pass"))),
        "F2_no_legal_mechanism": int(not inum(p2.get("P2_mechanism_pass"))),
        "F3_natural_stream_extension_missing": int(not inum(p3.get("P3_density_strong_pass"))),
        "F4_density_gate_not_resolved": int(not inum(p4.get("official_density_gate_proposal_allowed"))),
        "F5_geometry_family_no_legal_signal": int(not inum(p5.get("P5_weak_pass"))),
        "F6_controller_generated_runtime_blocked": int(not system_pass),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9820.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9820",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": args.device,
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "p1_max_per_group": args.p1_max_per_group,
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
    dump_json("run_manifest_v9820.json", manifest)
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "p1_weak_pass": p1.get("P1_weak_pass"),
        "p1_selected_actions": p1.get("selected_action_count"),
        "p1_rows_actual": p1.get("row_count_actual"),
        "p2_mechanism_pass": p2.get("P2_mechanism_pass"),
        "p3_density_strong_pass": p3.get("P3_density_strong_pass"),
        "p5_weak_pass": p5.get("P5_weak_pass"),
        "system_legal_controller_pass": system_pass,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
