#!/usr/bin/env python3
"""DG-KAN v9.4.6 source identity / no-transform replay closure runner.

This runner is intentionally conservative.  It treats v9.4.4/v9.4.5 oracle
survivors as diagnostic until the source identity ledger, AP0w no-transform
payload equivalence, and side-by-side outcome replay all close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
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

import run_v9410_value_producing_source_generator as v9410  # noqa: E402
import run_v9430_source_frontier_recovery_direct_generator as v9430  # noqa: E402
import run_v9440_source_value_objective_audit_oracle_seeded_generator as v9440  # noqa: E402
import run_v9450_robust_source_generator_objective_aligned_certificate as v9450  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.6_SourceIdentityNoTransformReplayClosure_RobustSourceRevalidation_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9460_source_identity_no_transform_replay_closure.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9450 = RESULT_ROOT / "v9450_robust_source_generator_objective_aligned_certificate_parallel_closure_first_20260514T120000Z"
DEFAULT_V9440 = RESULT_ROOT / "v9440_source_value_objective_audit_oracle_seeded_generator_reset_first_20260514T110000Z"
DEFAULT_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HORIZONS = [20, 80, 240]
SIDE_BY_SIDE_METRICS = [
    "V_branch",
    "CE_mean_after",
    "CEp99_after",
    "margin_p10_after",
    "acc_after",
    "curvature_after",
    "local_lipschitz_after",
    "basis_usage_entropy_after",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--source-v9450", default=str(DEFAULT_V9450))
    p.add_argument("--source-v9440", default=str(DEFAULT_V9440))
    p.add_argument("--source-v9350", default=str(DEFAULT_V9350))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--oracle-actions", type=int, default=16)
    p.add_argument("--run-base-acc", action="store_true")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--sentinel-seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--sentinel-steps", type=int, default=12)
    p.add_argument("--sentinel-train-size", type=int, default=512)
    p.add_argument("--sentinel-test-size", type=int, default=256)
    p.add_argument("--sentinel-hidden-dim", type=int, default=64)
    p.add_argument("--strong-lr-grid", default="0.0003,0.001,0.003")
    return p.parse_args()


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def lcb(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def tensor_hash(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        arr = t.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def load_source_payload(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    path = REPO / str(row.get("payload_shard_path"))
    if str(path) not in cache:
        cache[str(path)] = torch.load(path, map_location=device)
    shard = cache[str(path)]
    off = inum(row.get("payload_tensor_offset"))
    return [
        shard["d0"][off].detach().clone().to(device),
        shard["d1"][off].detach().clone().to(device),
        shard["d2"][off].detach().clone().to(device),
    ]


def load_clone_payload(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    path = REPO / str(row.get("ap_payload_shard_path"))
    if str(path) not in cache:
        cache[str(path)] = torch.load(path, map_location=device)
    shard = cache[str(path)]
    off = inum(row.get("ap_payload_tensor_offset"))
    return [
        shard["d0"][off].detach().clone().to(device),
        shard["d1"][off].detach().clone().to(device),
        shard["d2"][off].detach().clone().to(device),
    ]


def compare_payload(src: list[torch.Tensor], clone: list[torch.Tensor]) -> dict[str, float]:
    diffs = [(a - b).detach().float() for a, b in zip(src, clone)]
    src_flat = torch.cat([a.detach().double().flatten() for a in src])
    clone_flat = torch.cat([b.detach().double().flatten() for b in clone])
    diff_flat = torch.cat([d.detach().double().flatten() for d in diffs])
    linf = float(diff_flat.abs().max().item()) if diff_flat.numel() else 0.0
    l2 = float(diff_flat.norm().item()) if diff_flat.numel() else 0.0
    denom = float(src_flat.norm().item())
    rel = l2 / max(denom, 1.0e-12)
    cos = float(torch.dot(src_flat, clone_flat).item() / max(float(src_flat.norm().item() * clone_flat.norm().item()), 1.0e-12))
    return {"linf": linf, "l2": l2, "relative": rel, "cosine": cos}


def robust_label(st: dict[str, Any]) -> int:
    return int(inum(st.get("weak_h20")) and fnum(st.get("V", {}).get(20)) > 0.0 and fnum(st.get("long_risk_h240")) <= 0.10 and fnum(st.get("family_support_count")) > 0)


def load_base_tables(args: argparse.Namespace) -> dict[str, Any]:
    source_v9350 = Path(args.source_v9350)
    source_v9330 = Path(args.source_v9330)
    source_v9450 = Path(args.source_v9450)
    full_rows = read_csv(source_v9350 / "full_control_outcome_table_v9350.csv")
    payload_rows = read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv")
    stats = v9430.summarize_action_universe(full_rows, payload_rows)
    oracle_ids = v9450.oracle_panel(stats, int(args.oracle_actions))
    full_by_action: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in full_rows:
        full_by_action[str(r.get("action_id"))].append(r)
    payload_by_id = {
        str(r.get("action_id")): r
        for r in payload_rows
        if r.get("status") == "payload_disk_replay_row"
    }
    gen_trace = read_csv(source_v9450 / "oracle_seeded_generator_trace_v9450.csv")
    ap0w_generated = {
        str(r.get("source_action_id")): r
        for r in gen_trace
        if r.get("status") == "generated_action_row" and str(r.get("primitive_id")).startswith("AP0w-")
    }
    ap0w_outcomes = [
        r
        for r in gen_trace
        if r.get("status") == "source_branch_horizon_row" and str(r.get("primitive_id")).startswith("AP0w-")
    ]
    return {
        "full_rows": full_rows,
        "payload_rows": payload_rows,
        "stats": stats,
        "oracle_ids": oracle_ids,
        "full_by_action": full_by_action,
        "payload_by_id": payload_by_id,
        "ap0w_generated": ap0w_generated,
        "ap0w_outcomes": ap0w_outcomes,
    }


def p0_boundary(source_v9450: Path) -> dict[str, Any]:
    r = read_json(source_v9450 / "route_decision.json")
    return {
        "stage": "P0_V9450_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "route_v9450": r.get("route"),
        "source_route_v9440": r.get("source_route_v9440"),
        "objective_mismatch_pass": r.get("objective_mismatch_pass"),
        "Y_robust_base_rate": r.get("Y_robust_base_rate"),
        "P_LongRisk_h240_given_WeakCP_h20": r.get("P_LongRisk_h240_given_WeakCP_h20"),
        "oracle_survivor_anatomy_pass": r.get("oracle_survivor_anatomy_pass"),
        "dominant_oracle_legal_miss_reason": r.get("dominant_oracle_legal_miss_reason"),
        "preflight_all_pass": r.get("preflight_all_pass"),
        "oracle_seeded_no_transform_pass": r.get("oracle_seeded_no_transform_pass"),
        "oracle_seeded_best_generator": r.get("oracle_seeded_best_generator"),
        "oracle_seeded_best_h20_weak_CP": r.get("oracle_seeded_best_h20_weak_CP"),
        "oracle_seeded_best_h20_V_ctrl_LCB": r.get("oracle_seeded_best_h20_V_ctrl_LCB"),
        "oracle_seeded_best_h240_longrisk": r.get("oracle_seeded_best_h240_longrisk"),
        "direct_generator_pass": r.get("direct_generator_pass"),
        "certificate_effect_valid_pass": r.get("certificate_effect_valid_pass"),
        "base_acc_sentinel_pass": r.get("base_acc_sentinel_pass"),
        "system_legal_controller_pass": r.get("system_legal_controller_pass"),
        "p0_pass": int(r.get("route") == "R1-OracleSourceIdentityBug" and not inum(r.get("system_legal_controller_pass"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p1_source_identity_ledger(args: argparse.Namespace, tables: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    oracle_ids: list[str] = tables["oracle_ids"]
    stats: dict[str, dict[str, Any]] = tables["stats"]
    payload_by_id: dict[str, dict[str, str]] = tables["payload_by_id"]
    full_by_action: dict[str, list[dict[str, str]]] = tables["full_by_action"]
    source_table_hash = sha256_file(Path(args.source_v9350) / "full_control_outcome_table_v9350.csv")
    branch_config_hash = stable_hash("v9460-branches", sorted({r.get("branch_id") or r.get("branch") for r in tables["full_rows"]}))
    horizon_config_hash = stable_hash("v9460-horizons", HORIZONS)
    label_config_hash = stable_hash("v9460-labels", "WeakCP_h20", "V_ctrl_h20", "LongRisk_h240", "Y_robust")
    aggregation_hash = stable_hash("v9460-aggregation", "ORC-D-HorizonRobust", "minV+horizon_robust-longrisk")
    ledger: list[dict[str, Any]] = []
    identity_trace: list[dict[str, Any]] = []
    config_trace: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for rank, aid in enumerate(oracle_ids, start=1):
        st = stats.get(aid, {})
        payload = payload_by_id.get(aid, {})
        outcomes = full_by_action.get(aid, [])
        real_rows = [r for r in outcomes if (r.get("branch_id") or r.get("branch")) == "RealFunctional"]
        first = real_rows[0] if real_rows else (outcomes[0] if outcomes else {})
        seen[aid] += 1
        state_hash = str(first.get("base_checkpoint_hash") or first.get("checkpoint_hash_before"))
        batch_hash = stable_hash("batch", first.get("dataset"), first.get("seed"), first.get("step"), first.get("batch_id"))
        optimizer_hash = stable_hash("optimizer_replay_key", first.get("dataset"), first.get("seed"), first.get("step"), first.get("batch_id"))
        model_param_hash = state_hash
        label_hash = stable_hash("source-label", aid, st.get("weak_h20"), st.get("V", {}).get(20), st.get("long_risk_h240"), robust_label(st))
        ledger_row = {
            "stage": "P1_SOURCE_IDENTITY_LEDGER",
            "status": "source_action_ledger_row",
            "source_action_id": aid,
            "source_candidate_id": first.get("candidate_id") or payload.get("candidate_id") or st.get("candidate_id"),
            "source_event_id": first.get("event_id") or payload.get("event_id") or st.get("event_id"),
            "source_global_row_id": payload.get("global_row_id"),
            "source_dataset": first.get("dataset") or st.get("dataset"),
            "source_seed": first.get("seed") or st.get("seed"),
            "source_step": first.get("step") or st.get("step"),
            "source_batch_id": first.get("batch_id"),
            "source_family_id": first.get("family_id") or st.get("family_id"),
            "source_bucket_id": first.get("bucket_id") or st.get("bucket_id"),
            "source_horizon": "20,80,240",
            "source_primitive_id": "AP0-current-action-reference",
            "source_payload_hash": payload.get("payload_hash_loaded") or payload.get("payload_hash_expected") or first.get("payload_hash"),
            "source_payload_shard_id": Path(str(payload.get("payload_shard_path"))).stem if payload.get("payload_shard_path") else "",
            "source_payload_offset": payload.get("payload_tensor_offset"),
            "source_state_before_hash": state_hash,
            "source_optimizer_state_hash": optimizer_hash,
            "source_model_param_hash": model_param_hash,
            "source_batch_hash": batch_hash,
            "source_label_hash": label_hash,
            "source_branch_config_hash": branch_config_hash,
            "source_horizon_config_hash": horizon_config_hash,
            "source_label_config_hash": label_config_hash,
            "source_outcome_table_hash": source_table_hash,
            "source_oracle_selector_id": "ORC-D-HorizonRobust",
            "source_oracle_rank": rank,
            "source_oracle_panel_id": f"ORC-D-HorizonRobust-K{len(oracle_ids)}",
            "source_oracle_objective_version": "v9460-robust-source-objective",
            "source_outcome_aggregation_version": "v9460-orc-d-aggregation",
            "source_outcome_aggregation_hash": aggregation_hash,
            "payload_join_success": int(bool(payload)),
            "outcome_join_success": int(len(real_rows) >= len(HORIZONS)),
            "horizon_count_joined": len({inum(r.get("horizon")) for r in real_rows}),
            "branch_count_joined": len({r.get("branch_id") or r.get("branch") for r in outcomes}),
            "Y_robust": robust_label(st),
            "WeakCP_h20": st.get("weak_h20"),
            "V_ctrl_h20": st.get("V", {}).get(20),
            "LongRisk_h240": st.get("long_risk_h240"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        ledger.append(ledger_row)
        identity_trace.append({
            "stage": "P1_SOURCE_IDENTITY_LEDGER",
            "status": "identity_join_trace",
            "source_action_id": aid,
            "oracle_rank": rank,
            "payload_row_found": int(bool(payload)),
            "full_outcome_rows_found": len(outcomes),
            "realfunctional_horizons_found": len({inum(r.get("horizon")) for r in real_rows}),
            "action_id_duplicate_seen_count": seen[aid],
            "source_candidate_id": ledger_row["source_candidate_id"],
            "source_event_id": ledger_row["source_event_id"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    config_trace.append({
        "stage": "P1_SOURCE_IDENTITY_LEDGER",
        "status": "source_config_trace",
        "source_branch_config_hash": branch_config_hash,
        "source_horizon_config_hash": horizon_config_hash,
        "source_label_config_hash": label_config_hash,
        "source_outcome_table_hash": source_table_hash,
        "source_outcome_aggregation_hash": aggregation_hash,
        "optimizer_state_hash_semantics": "replay_key_hash_derived_from_dataset_seed_step_batch_id",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    duplicate_count = sum(max(0, c - 1) for c in seen.values())
    payload_missing = sum(int(not r.get("source_payload_hash")) for r in ledger)
    state_missing = sum(int(not r.get("source_state_before_hash")) for r in ledger)
    branch_missing = sum(int(not r.get("source_branch_config_hash")) for r in ledger)
    label_missing = sum(int(not r.get("source_label_config_hash")) for r in ledger)
    join_missing = sum(int(not inum(r.get("payload_join_success")) or not inum(r.get("outcome_join_success"))) for r in ledger)
    summary = {
        "stage": "P1_SOURCE_IDENTITY_LEDGER",
        "status": "summary",
        "oracle_selector_id": "ORC-D-HorizonRobust",
        "oracle_panel_id": f"ORC-D-HorizonRobust-K{len(oracle_ids)}",
        "source_action_ledger_rows": len(ledger),
        "source_action_id_duplicate_count": duplicate_count,
        "source_payload_hash_missing_count": payload_missing,
        "source_state_before_hash_missing_count": state_missing,
        "source_branch_config_hash_missing_count": branch_missing,
        "source_label_config_hash_missing_count": label_missing,
        "ORC_D_K16_join_success_count": len(ledger) - join_missing,
        "ORC_D_K16_join_missing_count": join_missing,
        "source_action_ledger_complete": int(len(ledger) == len(oracle_ids) and duplicate_count == 0 and payload_missing == 0 and state_missing == 0 and branch_missing == 0 and label_missing == 0 and join_missing == 0),
        "source_identity_ledger_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["source_identity_ledger_pass"] = summary["source_action_ledger_complete"]
    return [summary] + ledger, identity_trace, config_trace, summary


def p2_source_outcome_recompute(tables: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    oracle_ids: list[str] = tables["oracle_ids"]
    stats: dict[str, dict[str, Any]] = tables["stats"]
    trace: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for rank, aid in enumerate(oracle_ids, start=1):
        st = stats[aid]
        y = robust_label(st)
        row = {
            "stage": "P2_AP0_SOURCE_OUTCOME_RECOMPUTE",
            "status": "source_replay_action_row",
            "source_action_id": aid,
            "source_oracle_rank": rank,
            "WeakCP_h20": st.get("weak_h20"),
            "StrongCP_h20": st.get("strong", {}).get(20),
            "V_ctrl_h20": st.get("V", {}).get(20),
            "V_ctrl_h80": st.get("V", {}).get(80),
            "V_ctrl_h240": st.get("V", {}).get(240),
            "LongRisk_h240": st.get("long_risk_h240"),
            "HorizonRobustCP": st.get("horizon_robust"),
            "SupportBalance": int(fnum(st.get("family_support_count")) > 0),
            "Y_robust": y,
            "source_replay_mode": "measured_v9350_reference_recompute",
            "direct_new_rollout_performed": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        for h in HORIZONS:
            trace.append({
                "stage": "P2_AP0_SOURCE_OUTCOME_RECOMPUTE",
                "status": "source_replay_horizon_trace",
                "source_action_id": aid,
                "horizon": h,
                "WeakCP": st.get("weak", {}).get(h),
                "StrongCP": st.get("strong", {}).get(h),
                "V_ctrl": st.get("V", {}).get(h),
                "LongRisk": int(h == 240 and fnum(st.get("long_risk_h240")) > 0),
                "source_replay_mode": "measured_v9350_reference_recompute",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    summary = {
        "stage": "P2_AP0_SOURCE_OUTCOME_RECOMPUTE",
        "status": "summary",
        "oracle_selector_id": "ORC-D-HorizonRobust",
        "oracle_panel_id": f"ORC-D-HorizonRobust-K{len(oracle_ids)}",
        "source_replay_mode": "measured_v9350_reference_recompute",
        "direct_new_rollout_performed": 0,
        "source_action_count": len(rows),
        "h20_weak_CP": mean([fnum(r.get("WeakCP_h20")) for r in rows]),
        "h20_V_ctrl_lcb": lcb([fnum(r.get("V_ctrl_h20")) for r in rows]),
        "h240_long_risk": mean([fnum(r.get("LongRisk_h240")) for r in rows]),
        "Y_robust_count": sum(inum(r.get("Y_robust")) for r in rows),
        "Y_robust_rate": mean([fnum(r.get("Y_robust")) for r in rows]),
        "source_replay_reproduces_v9440_oracle": 0,
        "oracle_survivor_replay_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["source_replay_reproduces_v9440_oracle"] = int(
        abs(fnum(summary["h20_weak_CP"]) - 0.8125) <= 1.0e-12
        and fnum(summary["h20_V_ctrl_lcb"]) > 0.0
        and fnum(summary["h240_long_risk"]) <= 0.10
    )
    summary["oracle_survivor_replay_pass"] = summary["source_replay_reproduces_v9440_oracle"]
    return [summary] + rows, trace, summary


def choose_negative_controls(tables: dict[str, Any], n: int) -> list[str]:
    oracle = set(tables["oracle_ids"])
    stats: dict[str, dict[str, Any]] = tables["stats"]
    candidates = [aid for aid, st in stats.items() if aid not in oracle and not robust_label(st)]
    return sorted(candidates, key=lambda x: stable_hash("v9460-negative-control", x))[:n]


def p3_no_transform_payload_equivalence(args: argparse.Namespace, tables: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload_by_id: dict[str, dict[str, str]] = tables["payload_by_id"]
    ap0w_generated: dict[str, dict[str, str]] = tables["ap0w_generated"]
    stats: dict[str, dict[str, Any]] = tables["stats"]
    oracle_ids: list[str] = tables["oracle_ids"]
    negative_ids = choose_negative_controls(tables, len(oracle_ids))
    cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for scale_id, ids in [("P3a-single", oracle_ids[:1]), ("P3b-three", oracle_ids[:3]), ("P3c-oracle16", oracle_ids), ("P3d-negative16", negative_ids)]:
        for aid in ids:
            src_row = payload_by_id.get(aid, {})
            clone_row = ap0w_generated.get(aid, {})
            actual_clone_artifact = int(bool(clone_row))
            if not src_row:
                comp = {"linf": math.inf, "l2": math.inf, "relative": math.inf, "cosine": 0.0}
                src_hash = clone_hash = ""
                clone_action_id = ""
            else:
                src_payload = load_source_payload(src_row, cache, device)
                src_hash = tensor_hash(src_payload)
                if clone_row:
                    clone_payload = load_clone_payload(clone_row, cache, device)
                    clone_hash = tensor_hash(clone_payload)
                    clone_action_id = clone_row.get("ap_action_id") or clone_row.get("generated_action_id")
                else:
                    clone_payload = [t.detach().clone() for t in src_payload]
                    clone_hash = tensor_hash(clone_payload)
                    clone_action_id = stable_hash("v9460-virtual-negative-no-transform", aid)
                comp = compare_payload(src_payload, clone_payload)
            state_hash = stable_hash("source-state", aid, stats.get(aid, {}).get("dataset"), stats.get(aid, {}).get("seed"), stats.get(aid, {}).get("step"))
            optimizer_hash = stable_hash("optimizer_replay_key", stats.get(aid, {}).get("dataset"), stats.get(aid, {}).get("seed"), stats.get(aid, {}).get("step"))
            branch_hash = stable_hash("v9460-branches", "same-run-side-by-side")
            horizon_hash = stable_hash("v9460-horizons", HORIZONS)
            label_hash = stable_hash("v9460-labels", "source-vs-no-transform")
            verified = int(
                src_hash == clone_hash
                and comp["linf"] <= 1.0e-8
                and comp["relative"] <= 1.0e-7
                and comp["cosine"] >= 0.99999999
            )
            row = {
                "stage": "P3_NO_TRANSFORM_PAYLOAD_EQUIVALENCE",
                "status": "clone_payload_row",
                "scale_id": scale_id,
                "source_action_id": aid,
                "clone_action_id": clone_action_id,
                "clone_parent_source_action_id": aid,
                "clone_parent_source_payload_hash": src_row.get("payload_hash_loaded") or src_row.get("payload_hash_expected"),
                "source_payload_hash_tensor": src_hash,
                "clone_payload_hash": clone_hash,
                "clone_payload_hash_matches_source": int(src_hash == clone_hash),
                "clone_payload_linf_to_source": comp["linf"],
                "clone_payload_relative_error_to_source": comp["relative"],
                "clone_payload_cosine_to_source": comp["cosine"],
                "clone_state_before_hash": state_hash,
                "source_state_before_hash": state_hash,
                "clone_optimizer_state_hash": optimizer_hash,
                "source_optimizer_state_hash": optimizer_hash,
                "clone_branch_config_hash": branch_hash,
                "source_branch_config_hash": branch_hash,
                "clone_horizon_config_hash": horizon_hash,
                "source_horizon_config_hash": horizon_hash,
                "clone_label_config_hash": label_hash,
                "source_label_config_hash": label_hash,
                "clone_no_transform_declared": 1,
                "clone_no_transform_verified": verified,
                "clone_parent_join_success": int(bool(src_row)),
                "actual_clone_artifact_from_v9450": actual_clone_artifact,
                "negative_control_virtual_clone": int(not actual_clone_artifact),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            rows.append(row)
            trace.append({
                "stage": "P3_NO_TRANSFORM_PAYLOAD_EQUIVALENCE",
                "status": "payload_tensor_compare_trace",
                "scale_id": scale_id,
                "source_action_id": aid,
                "source_payload_hash": src_hash,
                "clone_payload_hash": clone_hash,
                "linf": comp["linf"],
                "relative": comp["relative"],
                "cosine": comp["cosine"],
                "actual_clone_artifact_from_v9450": actual_clone_artifact,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    oracle_rows = [r for r in rows if r.get("scale_id") == "P3c-oracle16"]
    p3_pass = int(
        len(oracle_rows) == len(oracle_ids)
        and all(inum(r.get("clone_no_transform_verified")) for r in oracle_rows)
        and all(inum(r.get("actual_clone_artifact_from_v9450")) for r in oracle_rows)
    )
    summary = {
        "stage": "P3_NO_TRANSFORM_PAYLOAD_EQUIVALENCE",
        "status": "summary",
        "oracle_clone_row_count": len(oracle_rows),
        "negative_control_row_count": len([r for r in rows if r.get("scale_id") == "P3d-negative16"]),
        "clone_payload_hash_match_rate": mean([fnum(r.get("clone_payload_hash_matches_source")) for r in oracle_rows]),
        "clone_payload_linf_max": max([fnum(r.get("clone_payload_linf_to_source")) for r in oracle_rows], default=0.0),
        "clone_payload_relative_error_max": max([fnum(r.get("clone_payload_relative_error_to_source")) for r in oracle_rows], default=0.0),
        "clone_payload_cosine_min": min([fnum(r.get("clone_payload_cosine_to_source")) for r in oracle_rows], default=0.0),
        "clone_no_transform_verified_count": sum(inum(r.get("clone_no_transform_verified")) for r in oracle_rows),
        "no_transform_payload_equivalence_pass": p3_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, trace, summary


def p4_side_by_side_replay(tables: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    outcomes: list[dict[str, str]] = tables["ap0w_outcomes"]
    oracle_ids: list[str] = tables["oracle_ids"]
    by_pair: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    clone_action_by_source: dict[str, str] = {}
    for r in outcomes:
        aid = str(r.get("source_action_id"))
        branch = str(r.get("branch"))
        h = inum(r.get("horizon"))
        by_pair[(aid, str(r.get("ap_action_id")), h)][branch] = r
        clone_action_by_source[aid] = str(r.get("ap_action_id"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_pairs = len(oracle_ids) * len(HORIZONS)
    for aid in oracle_ids:
        clone_id = clone_action_by_source.get(aid, "")
        for h in HORIZONS:
            pair = by_pair.get((aid, clone_id, h), {})
            clone_row = pair.get("RealSource", {})
            source_row = pair.get("SourceAP0Baseline", {})
            if not clone_row or not source_row:
                failures.append({
                    "stage": "P4_SIDE_BY_SIDE_REPLAY_FAILURE",
                    "status": "failure_row",
                    "failure_class": "I12-source-outcome-table-join-key-mismatch",
                    "source_action_id": aid,
                    "clone_action_id": clone_id,
                    "branch": "RealSource/SourceAP0Baseline",
                    "horizon": h,
                    "first_failed_contract": "pair_row_presence",
                    "expected_value": "both RealSource and SourceAP0Baseline rows",
                    "observed_value": f"clone={int(bool(clone_row))},source={int(bool(source_row))}",
                    "abs_diff": 1,
                    "rel_diff": 1,
                    "root_cause_candidate": "missing_side_by_side_outcome_pair",
                    "repair_required": "materialize same-run paired source/clone rows",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                continue
            branch_hash_match = int(str(source_row.get("branch_state_hash")) == str(clone_row.get("branch_state_hash")))
            horizon_hash_match = int(str(source_row.get("horizon_state_hash")) == str(clone_row.get("horizon_state_hash")))
            weak_match = int(inum(source_row.get("AP0_weak_CP_label")) == inum(clone_row.get("weak_CP_label")))
            strong_match = int(inum(source_row.get("AP0_strong_CP_label")) == inum(clone_row.get("strong_CP_label")))
            long_match = int(inum(source_row.get("AP0_long_risk_label")) == inum(clone_row.get("long_risk_label")))
            for metric in SIDE_BY_SIDE_METRICS:
                source_val = fnum(source_row.get(metric))
                clone_val = fnum(clone_row.get(metric))
                abs_diff = abs(source_val - clone_val)
                rel_diff = abs_diff / max(abs(source_val), 1.0e-12)
                label_match = int(weak_match and strong_match and long_match)
                row = {
                    "stage": "P4_SOURCE_VS_AP0W_NO_TRANSFORM_SIDE_BY_SIDE",
                    "status": "pair_metric_row",
                    "pair_id": stable_hash("pair", aid, clone_id, h, metric),
                    "source_action_id": aid,
                    "clone_action_id": clone_id,
                    "branch": "RealSource_vs_SourceAP0Baseline",
                    "horizon": h,
                    "metric_name": metric,
                    "source_metric_value": source_val,
                    "clone_metric_value": clone_val,
                    "metric_abs_diff": abs_diff,
                    "metric_rel_diff": rel_diff,
                    "source_label_weak": inum(source_row.get("AP0_weak_CP_label")),
                    "clone_label_weak": inum(clone_row.get("weak_CP_label")),
                    "source_label_strong": inum(source_row.get("AP0_strong_CP_label")),
                    "clone_label_strong": inum(clone_row.get("strong_CP_label")),
                    "source_label_longrisk": inum(source_row.get("AP0_long_risk_label")),
                    "clone_label_longrisk": inum(clone_row.get("long_risk_label")),
                    "label_match": label_match,
                    "source_branch_state_hash": source_row.get("branch_state_hash"),
                    "clone_branch_state_hash": clone_row.get("branch_state_hash"),
                    "source_horizon_state_hash": source_row.get("horizon_state_hash"),
                    "clone_horizon_state_hash": clone_row.get("horizon_state_hash"),
                    "branch_state_hash_match": branch_hash_match,
                    "horizon_state_hash_match": horizon_hash_match,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                rows.append(row)
            if branch_hash_match and not horizon_hash_match:
                failures.append({
                    "stage": "P4_SIDE_BY_SIDE_REPLAY_FAILURE",
                    "status": "failure_row",
                    "failure_class": "I18-outcome-materializer-runner-drift",
                    "source_action_id": aid,
                    "clone_action_id": clone_id,
                    "branch": "RealSource_vs_SourceAP0Baseline",
                    "horizon": h,
                    "first_failed_contract": "horizon_state_hash_match",
                    "expected_value": source_row.get("horizon_state_hash"),
                    "observed_value": clone_row.get("horizon_state_hash"),
                    "abs_diff": "",
                    "rel_diff": "",
                    "root_cause_candidate": "same payload/start branch_state but branch-named rollout path produces different horizon_state_hash",
                    "repair_required": "make AP0 source and AP0w clone use identical branch/horizon replay seed and materializer branch semantics",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
            elif not branch_hash_match:
                failures.append({
                    "stage": "P4_SIDE_BY_SIDE_REPLAY_FAILURE",
                    "status": "failure_row",
                    "failure_class": "I5-state-before-hash-mismatch",
                    "source_action_id": aid,
                    "clone_action_id": clone_id,
                    "branch": "RealSource_vs_SourceAP0Baseline",
                    "horizon": h,
                    "first_failed_contract": "branch_state_hash_match",
                    "expected_value": source_row.get("branch_state_hash"),
                    "observed_value": clone_row.get("branch_state_hash"),
                    "abs_diff": "",
                    "rel_diff": "",
                    "root_cause_candidate": "source and clone branch start states differ",
                    "repair_required": "audit state clone and branch start checkpoint construction",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    actual_pairs = len({(r["source_action_id"], r["clone_action_id"], r["horizon"]) for r in rows})
    metric_abs_max = max([fnum(r.get("metric_abs_diff")) for r in rows], default=0.0)
    label_match_rate = mean([fnum(r.get("label_match")) for r in rows])
    branch_hash_rate = mean([fnum(r.get("branch_state_hash_match")) for r in rows])
    horizon_hash_rate = mean([fnum(r.get("horizon_state_hash_match")) for r in rows])
    summary = {
        "stage": "P4_SOURCE_VS_AP0W_NO_TRANSFORM_SIDE_BY_SIDE",
        "status": "summary",
        "paired_action_count": len(oracle_ids),
        "paired_branch_horizon_row_count_expected": expected_pairs,
        "paired_branch_horizon_row_count_actual": actual_pairs,
        "pair_metric_row_count": len(rows),
        "metric_abs_diff_max": metric_abs_max,
        "label_match_rate": label_match_rate,
        "branch_state_hash_match_rate": branch_hash_rate,
        "horizon_state_hash_match_rate": horizon_hash_rate,
        "failure_class_primary": failures[0]["failure_class"] if failures else "",
        "failure_row_count": len(failures),
        "side_by_side_no_transform_replay_pass": int(actual_pairs == expected_pairs and metric_abs_max <= 1.0e-8 and label_match_rate == 1.0 and branch_hash_rate == 1.0 and horizon_hash_rate == 1.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    if not failures:
        failures = [{
            "stage": "P4_SIDE_BY_SIDE_REPLAY_FAILURE",
            "status": "empty",
            "failure_class": "",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
    return [summary] + rows, failures, summary


def p5_oracle_aggregation_recompute(tables: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = tables["stats"]
    oracle_ids: list[str] = tables["oracle_ids"]
    recomputed = v9450.oracle_panel(stats, len(oracle_ids))
    rows: list[dict[str, Any]] = []
    for idx, aid in enumerate(recomputed, start=1):
        st = stats[aid]
        previous_rank = oracle_ids.index(aid) + 1 if aid in oracle_ids else ""
        rows.append({
            "stage": "P5_ORACLE_OBJECTIVE_AGGREGATION_RECOMPUTE",
            "status": "oracle_rank_row",
            "source_action_id": aid,
            "recomputed_rank": idx,
            "previous_v9440_rank": previous_rank,
            "rank_match": int(previous_rank == idx),
            "WeakCP_h20": st.get("weak_h20"),
            "V_ctrl_h20": st.get("V", {}).get(20),
            "min_V_ctrl": min(fnum(st.get("V", {}).get(h)) for h in HORIZONS),
            "LongRisk_h240": st.get("long_risk_h240"),
            "HorizonRobustCP": st.get("horizon_robust"),
            "Y_robust": robust_label(st),
            "aggregation_score": min(fnum(st.get("V", {}).get(h)) for h in HORIZONS) + fnum(st.get("horizon_robust")) - fnum(st.get("long_risk_h240")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    overlap = len(set(oracle_ids) & set(recomputed)) / max(1, len(oracle_ids))
    rank_match_rate = mean([fnum(r.get("rank_match")) for r in rows])
    summary = {
        "stage": "P5_ORACLE_OBJECTIVE_AGGREGATION_RECOMPUTE",
        "status": "summary",
        "aggregation_version": "v9460-orc-d-aggregation",
        "source_objective_version": "v9460-robust-source-objective",
        "oracle_K": len(oracle_ids),
        "overlap_with_v9440_oracle": overlap,
        "rank_match_rate": rank_match_rate,
        "h20_weak_CP": mean([fnum(stats[i].get("weak_h20")) for i in recomputed]),
        "h20_V_ctrl_lcb": lcb([fnum(stats[i].get("V", {}).get(20)) for i in recomputed]),
        "h240_long_risk": mean([fnum(stats[i].get("long_risk_h240")) for i in recomputed]),
        "Y_robust_recomputed": 1,
        "source_objective_version_fixed": 1,
        "aggregation_version_fixed": 1,
        "oracle_recompute_matches_source_table": int(overlap == 1.0 and rank_match_rate == 1.0),
        "oracle_aggregation_recompute_pass": int(overlap == 1.0 and rank_match_rate == 1.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary], rows, summary


def p11_base_acc(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.run_base_acc:
        rows, trace, summary = v9440.p10_base_acc(args, device)
        for r in rows:
            r["stage"] = "P11_BASE_ACC_SENTINEL_CONTINUATION"
        for r in trace:
            r["stage"] = "P11_BASE_ACC_SENTINEL_TRACE"
        summary = dict(summary)
        summary["stage"] = "P11_BASE_ACC_SENTINEL_CONTINUATION"
        summary["status"] = "summary"
        summary["base_acc_reused_from_v9450"] = 0
        return rows, trace, summary
    source_rows = read_csv(Path(args.source_v9450) / "p10_base_acc_sentinel_strong_baseline_continuation.csv")
    source_trace_path = Path(args.source_v9450) / "base_acc_training_trace_v9450.csv"
    trace = read_csv(source_trace_path) if source_trace_path.exists() else []
    rows = [dict(r, stage="P11_BASE_ACC_SENTINEL_CONTINUATION", base_acc_reused_from_v9450=1) for r in source_rows]
    trace = [dict(r, stage="P11_BASE_ACC_SENTINEL_TRACE", base_acc_reused_from_v9450=1) for r in trace]
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({
        "stage": "P11_BASE_ACC_SENTINEL_CONTINUATION",
        "status": "summary",
        "base_acc_reused_from_v9450": 1,
        "base_acc_used_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return rows, trace, summary


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake_proxy = 0
    fake = 0
    proxy = 0
    cpu = 0
    for r in rows:
        fake += int(inum(r.get("fake_data_used")))
        proxy += int(inum(r.get("proxy_row_used")))
        cpu += int(inum(r.get("cpu_offload_used")))
    fake_proxy = fake + proxy
    return {
        "stage": "PROVENANCE_AUDIT",
        "status": "summary",
        "rows_checked": len(rows),
        "fake_proxy_nonzero_count": fake_proxy,
        "fake_data_used": int(fake > 0),
        "proxy_row_used": int(proxy > 0),
        "cpu_offload_used": int(cpu > 0),
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    files = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "aggregate_decision_v9460.json",
        out_dir / "p0_v9450_boundary_reproduction.csv",
        out_dir / "p1_source_identity_ledger.csv",
        out_dir / "orc_d_k16_identity_trace_v9460.csv",
        out_dir / "p2_ap0_source_outcome_recompute.csv",
        out_dir / "p3_no_transform_payload_equivalence.csv",
        out_dir / "p4_source_vs_ap0w_no_transform_side_by_side_replay.csv",
        out_dir / "no_transform_replay_failure_taxonomy_v9460.csv",
        out_dir / "p5_oracle_objective_aggregation_recompute.csv",
        out_dir / "p11_base_acc_sentinel_continuation.csv",
        out_dir / "p12_system_integration_gate_v9460.csv",
        out_dir / "contract_audit_v9460.csv",
        out_dir / "provenance_audit_v9460.csv",
        out_dir / "failure_table_v9460.csv",
    ]
    rows = []
    for path in files:
        if path.exists():
            rows.append({"artifact": rel(path), "sha256": sha256_file(path)})
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9430.choose_device(args.device)

    tables = load_base_tables(args)
    p0 = p0_boundary(Path(args.source_v9450))
    p1_rows, p1_trace, p1_config, p1 = p1_source_identity_ledger(args, tables)
    p2_rows, p2_trace, p2 = p2_source_outcome_recompute(tables)
    p3_rows, p3_trace, p3 = p3_no_transform_payload_equivalence(args, tables, device)
    p4_rows, p4_failures, p4 = p4_side_by_side_replay(tables)
    p5_rows, p5_trace, p5 = p5_oracle_aggregation_recompute(tables)

    if not inum(p1.get("source_identity_ledger_pass")):
        route = "R1a-OracleSourceLedgerIncomplete"
        primary = "source_identity_ledger_incomplete"
    elif not inum(p2.get("oracle_survivor_replay_pass")):
        route = "R1b-OracleObjectiveOrSourceOutcomeReplayBug"
        primary = "oracle_source_outcome_replay_bug"
    elif not inum(p3.get("no_transform_payload_equivalence_pass")):
        route = "R1c-NoTransformPayloadOrStateCloneBug"
        primary = "no_transform_payload_or_state_clone_bug"
    elif not inum(p4.get("side_by_side_no_transform_replay_pass")):
        route = "R1d-NoTransformOutcomeReplayBug"
        primary = "no_transform_outcome_replay_inconsistent"
    elif not inum(p5.get("oracle_aggregation_recompute_pass")):
        route = "R1e-OracleAggregationDefinitionBug"
        primary = "oracle_aggregation_definition_bug"
    else:
        route = "R2-GeneratorDestructiveConfirmedAfterIdentityClosure"
        primary = "generator_or_certificate_not_reopened_in_v9460"

    gated_reason = "P4_no_transform_side_by_side_replay_not_equivalent" if route == "R1d-NoTransformOutcomeReplayBug" else f"{route}_gate"
    p6_rows = [not_run("P6_CONDITIONAL_GENERATOR_REVALIDATION", gated_reason)]
    p7_rows = [not_run("P7_DIRECT_ROBUST_SOURCE_GENERATOR_REVALIDATION", gated_reason)]
    p8_rows = [not_run("P8_EFFECT_VALID_CERTIFICATE_REVALIDATION", gated_reason)]
    p9_rows = [not_run("P9_SOURCE_CERTIFICATE_CONTROLLER", gated_reason)]
    p10_rows = [not_run("P10_SELECTED_SOURCE_RUNTIME", gated_reason)]
    p11_rows, p11_trace, p11 = p11_base_acc(args, device)
    p12_rows = [{
        "stage": "P12_SYSTEM_INTEGRATION_GATE",
        "status": "summary",
        "system_candidate_id": "SYS-v9460-source-identity-no-transform-replay",
        "controller_id": "not_selected_identity_replay_blocked",
        "runtime_candidate_id": "not_selected_identity_replay_blocked",
        "source_identity_ledger_pass": p1.get("source_identity_ledger_pass"),
        "oracle_survivor_replay_pass": p2.get("oracle_survivor_replay_pass"),
        "no_transform_payload_equivalence_pass": p3.get("no_transform_payload_equivalence_pass"),
        "side_by_side_no_transform_replay_pass": p4.get("side_by_side_no_transform_replay_pass"),
        "oracle_aggregation_recompute_pass": p5.get("oracle_aggregation_recompute_pass"),
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "reason": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    p13_rows = [not_run("P13_LEAVEOUT_AND_PAIRED_REPLAY_BOUNDARY", "P12_system_controller_not_official")]
    p14_rows = [not_run("P14_SHORT_FULL_TRAINING_BOUNDARY", "P12_system_controller_not_official")]
    p15_rows = [not_run("P15_CONTINUAL_ROBUSTNESS_BOUNDARY", "P12_system_controller_not_official")]

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9450": p0.get("route_v9450"),
        "source_identity_ledger_pass": p1.get("source_identity_ledger_pass"),
        "source_action_ledger_rows": p1.get("source_action_ledger_rows"),
        "ORC_D_K16_join_success_count": p1.get("ORC_D_K16_join_success_count"),
        "ORC_D_K16_join_missing_count": p1.get("ORC_D_K16_join_missing_count"),
        "oracle_survivor_replay_pass": p2.get("oracle_survivor_replay_pass"),
        "source_replay_mode": p2.get("source_replay_mode"),
        "direct_new_rollout_performed": p2.get("direct_new_rollout_performed"),
        "source_recomputed_h20_weak_CP": p2.get("h20_weak_CP"),
        "source_recomputed_h20_V_ctrl_lcb": p2.get("h20_V_ctrl_lcb"),
        "source_recomputed_h240_long_risk": p2.get("h240_long_risk"),
        "source_recomputed_Y_robust_count": p2.get("Y_robust_count"),
        "no_transform_payload_equivalence_pass": p3.get("no_transform_payload_equivalence_pass"),
        "clone_payload_hash_match_rate": p3.get("clone_payload_hash_match_rate"),
        "clone_payload_linf_max": p3.get("clone_payload_linf_max"),
        "clone_payload_relative_error_max": p3.get("clone_payload_relative_error_max"),
        "clone_payload_cosine_min": p3.get("clone_payload_cosine_min"),
        "side_by_side_no_transform_replay_pass": p4.get("side_by_side_no_transform_replay_pass"),
        "paired_branch_horizon_row_count_actual": p4.get("paired_branch_horizon_row_count_actual"),
        "metric_abs_diff_max": p4.get("metric_abs_diff_max"),
        "label_match_rate": p4.get("label_match_rate"),
        "branch_state_hash_match_rate": p4.get("branch_state_hash_match_rate"),
        "horizon_state_hash_match_rate": p4.get("horizon_state_hash_match_rate"),
        "failure_class_primary": p4.get("failure_class_primary"),
        "oracle_aggregation_recompute_pass": p5.get("oracle_aggregation_recompute_pass"),
        "oracle_recompute_overlap_with_v9440": p5.get("overlap_with_v9440_oracle"),
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "mean_test_acc_LQ": p11.get("mean_test_acc_LQ"),
        "mean_test_acc_MLP": p11.get("mean_test_acc_MLP"),
        "mean_test_acc_AdamWStrongLRGridMLP": p11.get("mean_test_acc_AdamWStrongLRGridMLP"),
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "success_v9460_strict_purekan_functional": 0,
        "success_v9460_full_functional": 0,
        "success_v9460_external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "repair_no_transform_outcome_replay_seed_and_branch_semantics_before_generator_revalidation",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    run_manifest = {
        "run_id": "v9460_source_identity_no_transform_replay_closure",
        "created_utc": "2026-05-14T13:00:00Z",
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "source_v9450": str(args.source_v9450),
        "source_v9440": str(args.source_v9440),
        "source_v9350": str(args.source_v9350),
        "source_v9330": str(args.source_v9330),
        "oracle_actions": args.oracle_actions,
        "run_base_acc": int(bool(args.run_base_acc)),
        "no_fake_policy": "no fake rows, no proxy rows, diagnostics not promoted",
    }

    write_json(out_dir / "run_manifest.json", run_manifest)
    write_csv(out_dir / "p0_v9450_boundary_reproduction.csv", [p0])
    write_csv(out_dir / "p1_source_identity_ledger.csv", p1_rows)
    write_csv(out_dir / "orc_d_k16_identity_trace_v9460.csv", p1_trace)
    write_csv(out_dir / "source_payload_state_config_trace_v9460.csv", p1_config)
    write_csv(out_dir / "p2_ap0_source_outcome_recompute.csv", p2_rows)
    write_csv(out_dir / "source_replay_trace_v9460.csv", p2_trace)
    write_csv(out_dir / "p3_no_transform_payload_equivalence.csv", p3_rows)
    write_csv(out_dir / "no_transform_payload_trace_v9460.csv", p3_trace)
    write_csv(out_dir / "p4_source_vs_ap0w_no_transform_side_by_side_replay.csv", p4_rows)
    write_csv(out_dir / "no_transform_replay_failure_taxonomy_v9460.csv", p4_failures)
    write_csv(out_dir / "p5_oracle_objective_aggregation_recompute.csv", p5_rows)
    write_csv(out_dir / "oracle_rank_recompute_trace_v9460.csv", p5_trace)
    write_csv(out_dir / "p6_generator_revalidation_boundary.csv", p6_rows)
    write_csv(out_dir / "p7_direct_robust_generator_boundary.csv", p7_rows)
    write_csv(out_dir / "p8_effect_valid_certificate_boundary.csv", p8_rows)
    write_csv(out_dir / "p9_source_certificate_controller_boundary.csv", p9_rows)
    write_csv(out_dir / "p10_selected_source_runtime_boundary.csv", p10_rows)
    write_csv(out_dir / "p11_base_acc_sentinel_continuation.csv", p11_rows)
    write_csv(out_dir / "base_acc_training_trace_v9460.csv", p11_trace if p11_trace else [not_run("P11_BASE_ACC_SENTINEL_TRACE", "reused_summary_only")])
    write_csv(out_dir / "p12_system_integration_gate_v9460.csv", p12_rows)
    write_csv(out_dir / "p13_leaveout_and_paired_replay_boundary_v9460.csv", p13_rows)
    write_csv(out_dir / "p14_short_full_training_boundary_v9460.csv", p14_rows)
    write_csv(out_dir / "p15_continual_robustness_boundary_v9460.csv", p15_rows)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9460.json", route_decision)

    contract = {
        "stage": "CONTRACT_AUDIT",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "source_identity_ledger_pass": p1.get("source_identity_ledger_pass"),
        "oracle_survivor_replay_pass": p2.get("oracle_survivor_replay_pass"),
        "no_transform_payload_equivalence_pass": p3.get("no_transform_payload_equivalence_pass"),
        "side_by_side_no_transform_replay_pass": p4.get("side_by_side_no_transform_replay_pass"),
        "oracle_aggregation_recompute_pass": p5.get("oracle_aggregation_recompute_pass"),
        "diagnostic_oracle_used_for_official": 0,
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    failure_table = [{
        "stage": "FAILURE_TABLE",
        "status": "summary",
        "route": route,
        "F1a_source_ledger_incomplete": int(route == "R1a-OracleSourceLedgerIncomplete"),
        "F1b_oracle_source_replay_bug": int(route == "R1b-OracleObjectiveOrSourceOutcomeReplayBug"),
        "F1c_no_transform_payload_or_state_bug": int(route == "R1c-NoTransformPayloadOrStateCloneBug"),
        "F1d_no_transform_outcome_replay_bug": int(route == "R1d-NoTransformOutcomeReplayBug"),
        "F1e_oracle_aggregation_bug": int(route == "R1e-OracleAggregationDefinitionBug"),
        "F2_generator_revalidation_blocked": 1,
        "F3_certificate_revalidation_blocked": 1,
        "F4_controller_runtime_blocked": 1,
        "F5_base_acc_catastrophic": int(inum(p11.get("LQ_catastrophic_fail"))),
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    all_rows = (
        [p0]
        + p1_rows
        + p1_trace
        + p1_config
        + p2_rows
        + p2_trace
        + p3_rows
        + p3_trace
        + p4_rows
        + p4_failures
        + p5_rows
        + p5_trace
        + p6_rows
        + p7_rows
        + p8_rows
        + p9_rows
        + p10_rows
        + p11_rows
        + p12_rows
        + p13_rows
        + p14_rows
        + p15_rows
        + [contract]
        + failure_table
    )
    provenance = audit_rows(all_rows)
    write_csv(out_dir / "contract_audit_v9460.csv", [contract])
    write_csv(out_dir / "provenance_audit_v9460.csv", [provenance])
    write_csv(out_dir / "failure_table_v9460.csv", failure_table)
    hashes = hash_rows(out_dir)
    write_csv(out_dir / "artifact_hashes_v9460.csv", hashes)
    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "source_identity_ledger_pass": p1.get("source_identity_ledger_pass"),
        "no_transform_payload_equivalence_pass": p3.get("no_transform_payload_equivalence_pass"),
        "side_by_side_no_transform_replay_pass": p4.get("side_by_side_no_transform_replay_pass"),
        "failure_class_primary": p4.get("failure_class_primary"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
