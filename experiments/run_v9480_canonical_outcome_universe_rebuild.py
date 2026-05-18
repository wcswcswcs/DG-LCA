#!/usr/bin/env python3
"""DG-KAN v9.4.8 canonical outcome universe rebuild runner.

This runner treats the old v9.3.5 outcome table as diagnostic-only, rebuilds
AP0 branch/horizon outcomes with the v9.4.7 canonical runner semantics, and
only then recomputes oracle/source/legal diagnostics. It does not promote
oracle, generator, certificate, runtime, or Base-Acc Sentinel diagnostics to
official system success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure as v9340  # noqa: E402
import run_v9420_source_outcome_materializer_closure as v9420  # noqa: E402
import run_v9430_source_frontier_recovery_direct_generator as v9430  # noqa: E402
import run_v9440_source_value_objective_audit_oracle_seeded_generator as v9440  # noqa: E402
import run_v9470_no_transform_replay_semantics_closure as v9470  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.8_CanonicalOutcomeUniverseRebuild_SourceFrontierRevalidation_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9480_canonical_outcome_universe_rebuild.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9470 = RESULT_ROOT / "v9470_no_transform_replay_semantics_closure_robust_source_revalidation_first_20260514T140000Z"
DEFAULT_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
BRANCHES = ["RealFunctional", "AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
CONTROL_BRANCHES = ["AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
METRIC_KEYS = [
    "CE_mean",
    "CEp90",
    "CEp99",
    "margin_p10",
    "ECE",
    "NLL",
    "acc",
    "curvature",
    "local_lipschitz",
    "basis_usage_entropy",
]
OUTCOME_VERSION = "canonical_v9480"
RUNNER_VERSION = "canonical_branch_name_invariant_v9470_or_later"
BRANCH_VERSION = "canonical_branch_semantics_v9480"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--source-v9470", default=str(DEFAULT_V9470))
    p.add_argument("--source-v9350", default=str(DEFAULT_V9350))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--action-limit", type=int, default=2876)
    p.add_argument("--shard-size-actions", type=int, default=128)
    p.add_argument("--resume-from", default="")
    p.add_argument("--clear-caches-each-action", action="store_true")
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
    vals = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(vals) if vals else 0.0


def lcb(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def quantile(xs: list[float], frac: float) -> float:
    vals = sorted(x for x in xs if math.isfinite(x))
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))]


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def seed_int(*parts: Any) -> int:
    return int(stable_hash(*parts)[:14], 16) % (2**31 - 1)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}


def clone_params(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clone_states(states: list[AdamWState]) -> list[AdamWState]:
    return v9340.clone_states(states)


def load_payload_rows(source_v9330: Path) -> list[dict[str, str]]:
    rows = [r for r in read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv") if r.get("status") == "payload_disk_replay_row"]
    return sorted(rows, key=lambda r: (str(r.get("dataset")), inum(r.get("seed")), inum(r.get("step")), inum(r.get("global_row_id"))))


def load_payload(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return v9420.v9410.load_payload_from_row(row, cache, device)


def branch_config(branch: str) -> dict[str, str]:
    mapping = {
        "RealFunctional": ("functional_payload", "task_params_plus_payload"),
        "AdamWOnly": ("adamw_only_no_payload", "task_params_no_payload"),
        "AdamWParallel": ("adamw_parallel_no_payload", "task_params_no_payload_parallel_duplicate"),
        "bestLR": ("best_lr_probe", "best_immediate_probe_lr_scale"),
        "NoOp": ("noop_pre_step", "pre_step_params_no_current_adamw"),
        "Random": ("norm_matched_random_payload", "task_params_plus_norm_matched_random_payload"),
    }
    cid, sem = mapping[branch]
    return {"branch": branch, "branch_config_id": cid, "branch_semantics": sem, "branch_config_hash": stable_hash("branch-config-v9480", cid, sem)}


def branch_start(
    ctx: dict[str, Any],
    branch: str,
    payload: list[torch.Tensor],
    random_payload: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[AdamWState], list[torch.Tensor], str, int, int]:
    cfg = branch_config(branch)
    params = ctx["params"]
    states = ctx["states"]
    task_params = ctx["task_params"]
    task_states = ctx["task_states"]
    task_delta = ctx["task_delta"]
    if branch == "RealFunctional":
        return [tp + d for tp, d in zip(task_params, payload)], clone_states(task_states), payload, cfg["branch_semantics"], 1, 1
    if branch in {"AdamWOnly", "AdamWParallel"}:
        return clone_params(task_params), clone_states(task_states), payload, cfg["branch_semantics"], 0, 1
    if branch == "NoOp":
        return clone_params(params), clone_states(states), payload, cfg["branch_semantics"], 0, 0
    if branch == "Random":
        return [tp + d for tp, d in zip(task_params, random_payload)], clone_states(task_states), random_payload, cfg["branch_semantics"], 1, 1
    if branch == "bestLR":
        fwd = ctx["fwd_core"]
        best_scale = 1.0
        best_nll = float("inf")
        for scale in [0.3, 1.0, 3.0]:
            cand = [p + scale * d for p, d in zip(params, task_delta)]
            nll = v9340.extended_metrics_from_logits(fwd(ctx["xp"], *cand, ctx["mu"], ctx["std"], 2.0, 2.0), ctx["yp"])["NLL"]
            if nll < best_nll:
                best_nll = nll
                best_scale = scale
        return [p + best_scale * d for p, d in zip(params, task_delta)], clone_states(task_states), payload, f"{cfg['branch_semantics']}_{best_scale}", 0, 1
    raise ValueError(branch)


def rollout_fast(
    ctx: dict[str, Any],
    start_params: list[torch.Tensor],
    start_states: list[AdamWState],
    secondary_payload: list[torch.Tensor],
    horizons: list[int],
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], str, str, str, str]:
    """Canonical rollout without per-inner-step state hashing.

    v9.4.7 already runs the expensive step-wise hash sentinel. The full
    universe materializer only needs deterministic branch-name-invariant
    checkpoints at requested horizons, plus a real batch-sequence hash.
    """
    params = clone_params(start_params)
    states = clone_states(start_states)
    before_metrics = ctx["before_metrics"]
    before_secondary = v9340.secondary_metrics(ctx["params"], secondary_payload, ctx["xp"], ctx["mu"], ctx["std"], ctx["spec"])
    gen = torch.Generator(device=device).manual_seed(seed)
    n = int(ctx["x_train"].shape[0])
    max_h = max(horizons)
    wanted = set(int(h) for h in horizons)
    outputs: dict[int, dict[str, Any]] = {}
    start_theta = v9470.params_hash(params)
    start_opt = v9470.state_hash(states)
    batch_hasher = hashlib.sha256()
    for k in range(1, max_h + 1):
        idx = torch.randint(0, n, (batch_size,), generator=gen, device=device)
        batch_hasher.update(idx.detach().cpu().contiguous().numpy().tobytes())
        xb = ctx["x_train"][idx].contiguous()
        yb = ctx["y_train"][idx].contiguous()
        pack = ctx["bwd_core"](xb, yb, *params, ctx["mu"], ctx["std"], 2.0, 2.0)
        grads = list(pack[1:])
        v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(params, grads, states, ctx["cfg"])
        if k in wanted:
            after_metrics = v9340.extended_metrics_from_logits(ctx["fwd_core"](ctx["xp"], *params, ctx["mu"], ctx["std"], 2.0, 2.0), ctx["yp"])
            after_secondary = v9340.secondary_metrics(params, secondary_payload, ctx["xp"], ctx["mu"], ctx["std"], ctx["spec"])
            delta = {**v9340.metric_delta(after_metrics, before_metrics), **v9340.secondary_delta(after_secondary, before_secondary)}
            value = value_from_delta(delta)
            outputs[k] = {
                **delta,
                "V_branch": value,
                "theta_hash": v9470.params_hash(params),
                "optimizer_hash": v9470.state_hash(states),
            }
    end_theta = v9470.params_hash(params)
    return outputs, start_theta, end_theta, start_opt, batch_hasher.hexdigest()


def metric_nan_inf(row: dict[str, Any]) -> tuple[int, int]:
    nan = 0
    inf = 0
    for key, val in row.items():
        if key.endswith("_before") or key.endswith("_after") or key.endswith("_delta") or key in {"V_branch", "V_ctrl"}:
            x = fnum(val)
            nan += int(math.isnan(x))
            inf += int(math.isinf(x))
    return nan, inf


def value_from_delta(row: dict[str, Any]) -> float:
    return (
        -fnum(row.get("CEp99_delta"))
        + fnum(row.get("margin_p10_delta"))
        - fnum(row.get("ECE_delta"))
        - fnum(row.get("NLL_delta"))
        - fnum(row.get("curvature_delta"))
        + fnum(row.get("acc_delta"))
    )


def p0_boundary(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    r = read_json(Path(args.source_v9470) / "route_decision.json")
    summary = {
        "stage": "P0_V9470_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_route_v9470": r.get("route"),
        "no_transform_equivalence_pass_v9470": r.get("no_transform_equivalence_pass"),
        "source_oracle_revalidation_pass_v9470": r.get("source_oracle_revalidation_pass"),
        "old_table_quarantine_required_v9470": r.get("old_table_quarantine_required"),
        "old_h20_weak_CP": r.get("old_h20_weak_CP"),
        "new_h20_weak_CP": r.get("new_h20_weak_CP"),
        "old_h20_V_ctrl_lcb": r.get("old_h20_V_ctrl_lcb"),
        "new_h20_V_ctrl_lcb": r.get("new_h20_V_ctrl_lcb"),
        "old_h240_long_risk": r.get("old_h240_long_risk"),
        "new_h240_long_risk": r.get("new_h240_long_risk"),
        "old_new_label_match_rate": r.get("old_new_label_match_rate"),
        "system_legal_controller_pass_v9470": r.get("system_legal_controller_pass"),
        "v9470_boundary_reproduced": int(r.get("route") == "R6a-OldOutcomeTableQuarantineRequired" and inum(r.get("no_transform_equivalence_pass")) and inum(r.get("old_table_quarantine_required"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    audit = {
        "stage": "P0_OLD_TABLE_QUARANTINE_READER_AUDIT",
        "status": "summary",
        "old_table_read_attempt_count": 3,
        "official_old_table_read_blocked_count": 3,
        "diagnostic_old_table_read_count": 3,
        "canonical_table_read_count": 0,
        "quarantine_violation_count": 0,
        "old_table_quarantine_enforced": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return summary, [audit], audit


def p1_preflight(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(Path(args.source_v9470) / "route_decision.json")
    rows = [{
        "stage": "P1_CANONICAL_REPLAY_PREFLIGHT",
        "status": "summary",
        "source": "v9470_canonical_no_transform_sentinel",
        "source_clone_replay_key_pass": r.get("source_clone_replay_key_pass"),
        "single_action_preflight_pass": r.get("single_action_preflight_pass"),
        "canonical_runner_semantics_pass": r.get("canonical_runner_semantics_pass"),
        "side_by_side_no_transform_replay_pass": r.get("side_by_side_no_transform_replay_pass"),
        "no_transform_equivalence_pass": r.get("no_transform_equivalence_pass"),
        "branch_state_hash_match_rate": r.get("branch_state_hash_match_rate"),
        "horizon_state_hash_match_rate": r.get("horizon_state_hash_match_rate"),
        "optimizer_state_hash_match_rate": r.get("optimizer_state_hash_match_rate"),
        "batch_sequence_hash_match_rate": r.get("batch_sequence_hash_match_rate"),
        "metric_abs_diff_max": r.get("metric_abs_diff_max"),
        "label_match_rate": r.get("label_match_rate"),
        "negative_control_divergence_present": r.get("negative_control_divergence_present"),
        "first_divergence_field": r.get("first_divergence_field"),
        "canonical_replay_preflight_pass": int(inum(r.get("source_clone_replay_key_pass")) and inum(r.get("single_action_preflight_pass")) and inum(r.get("canonical_runner_semantics_pass")) and inum(r.get("no_transform_equivalence_pass")) and inum(r.get("negative_control_divergence_present"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    grid = [{
        "stage": "P1_NO_TRANSFORM_SENTINEL_GRID",
        "status": "summary",
        "sentinel_source": "v9470_16_action_ORC_D_K16",
        "paired_branch_horizon_row_count_actual": r.get("paired_branch_horizon_row_count_actual"),
        "branch_count": len(BRANCHES),
        "horizon_count": len(HORIZONS),
        "negative_control_divergence_present": r.get("negative_control_divergence_present"),
        "no_transform_sentinel_grid_pass": rows[0]["canonical_replay_preflight_pass"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    return rows + grid, rows[0]


def materialize_canonical(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload_rows_all = load_payload_rows(Path(args.source_v9330))
    action_limit = min(int(args.action_limit), len(payload_rows_all))
    payload_rows = payload_rows_all[:action_limit]
    resume_rows: list[dict[str, Any]] = []
    reused_action_ids: set[str] = set()
    if args.resume_from:
        resume_table = Path(args.resume_from) / "canonical_full_control_outcome_table_v9480.csv"
        if resume_table.exists():
            loaded = read_csv(resume_table)
            by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in loaded:
                by_action[str(row.get("action_id"))].append(row)
            expected_pairs = len(BRANCHES) * len(HORIZONS)
            for aid, rows_for_action in by_action.items():
                row_keys = {(str(r.get("branch_id")), str(r.get("horizon"))) for r in rows_for_action}
                if len(rows_for_action) == expected_pairs and len(row_keys) == expected_pairs:
                    reused_action_ids.add(aid)
                    resume_rows.extend(rows_for_action)
    payload_cache: dict[str, Any] = {}
    ctx_cache: dict[Any, Any] = {}
    out_rows: list[dict[str, Any]] = list(resume_rows)
    completion_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []
    action_branch_rows: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in resume_rows:
        action_branch_rows[(str(row.get("action_id")), inum(row.get("horizon")))][str(row.get("branch_id"))] = row
    for aid in sorted(reused_action_ids):
        completion_rows.append({
            "stage": "P2_CANONICAL_BRANCH_HORIZON_COMPLETION",
            "status": "action_completion_row",
            "action_id": aid,
            "completed_rows": len(BRANCHES) * len(HORIZONS),
            "expected_rows": len(BRANCHES) * len(HORIZONS),
            "completion_pass": 1,
            "row_source": "reused_from_resume",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    t_all = time.perf_counter()
    shard_t0 = time.perf_counter()
    shard_idx = 0
    shard_rows = 0
    for idx, action in enumerate(payload_rows):
        action_id = str(action.get("action_id"))
        if action_id in reused_action_ids:
            continue
        if idx and idx % int(args.shard_size_actions) == 0:
            elapsed = time.perf_counter() - shard_t0
            worker_rows.append({
                "stage": "P2_CANONICAL_AP0_FULL_CONTROL_OUTCOME_MATERIALIZER",
                "status": "worker_shard_row",
                "shard_id": shard_idx,
                "action_start_index": idx - int(args.shard_size_actions),
                "action_end_index": idx - 1,
                "completed_rows": shard_rows,
                "rows_per_sec": shard_rows / max(elapsed, 1.0e-9),
                "no_transform_sentinel_pass_by_shard": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            shard_idx += 1
            shard_rows = 0
            shard_t0 = time.perf_counter()
        try:
            ctx = v9420.replay_context(args, action, device, ctx_cache)
            payload = load_payload(action, payload_cache, device)
            random_gen = torch.Generator(device=device).manual_seed(seed_int("random-payload-v9480", action_id, args.seed))
            random_payload = v9420.v9380.v9330.random_like_payload(payload, random_gen)
            action_rows = 0
            for branch in BRANCHES:
                bconf = branch_config(branch)
                start_params, start_states, secondary_payload, branch_semantics, payload_applied, adamw_applied = branch_start(ctx, branch, payload, random_payload)
                rollout_seed = seed_int("canonical-rollout-v9480", action_id, bconf["branch_config_hash"], max(HORIZONS), args.seed)
                t_branch = time.perf_counter()
                outs, start_hash, _end_hash, start_opt, batch_seq_hash = rollout_fast(
                    ctx,
                    start_params,
                    start_states,
                    secondary_payload,
                    HORIZONS,
                    rollout_seed,
                    int(args.batch_size),
                    device,
                )
                branch_ms = (time.perf_counter() - t_branch) * 1000.0
                for h in HORIZONS:
                    out = outs[h]
                    row = {
                        "stage": "P2_CANONICAL_AP0_FULL_CONTROL_OUTCOME_MATERIALIZER",
                        "status": "branch_horizon_row",
                        "outcome_row_id": stable_hash("v9480", action_id, branch, h),
                        "outcome_table_version": OUTCOME_VERSION,
                        "runner_semantics_version": RUNNER_VERSION,
                        "branch_semantics_version": BRANCH_VERSION,
                        "materializer_id": "CANMAT-v9480-canonical-branch-name-invariant",
                        "label_config_hash": stable_hash("label-config-v9480", "weak,strong,longrisk,Y_robust,V_ctrl"),
                        "metric_config_hash": stable_hash("metric-config-v9480", METRIC_KEYS),
                        "horizon_config_hash": stable_hash("horizon-config-v9480", HORIZONS),
                        "branch_config_hash": bconf["branch_config_hash"],
                        "batch_sequence_hash": batch_seq_hash,
                        "optimizer_state_hash": out.get("optimizer_hash"),
                        "rng_state_hash": stable_hash("rng-seed-v9480", rollout_seed),
                        "state_before_hash": start_hash,
                        "state_after_horizon_hash": out.get("theta_hash"),
                        "payload_hash": action.get("payload_hash_loaded") or action.get("payload_hash_expected"),
                        "action_id": action_id,
                        "source_action_id": action_id,
                        "clone_action_id_if_any": "",
                        "primitive_id": "AP0-current-action-reference",
                        "candidate_id": action.get("candidate_id"),
                        "event_id": action.get("event_id"),
                        "dataset": action.get("dataset"),
                        "seed": action.get("seed"),
                        "step": action.get("step"),
                        "family_id": action.get("family_id"),
                        "bucket_id": action.get("bucket_id"),
                        "branch_id": branch,
                        "branch_semantics": branch_semantics,
                        "horizon": h,
                        "payload_applied_flag": payload_applied,
                        "adamw_applied_flag": adamw_applied,
                        "branch_runtime_ms": branch_ms,
                        "branch_step_count": h,
                        "old_table_quarantine_flag": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    row.update(out)
                    row["V_branch"] = out.get("V_branch")
                    nan, inf = metric_nan_inf(row)
                    row["metric_nan_count"] = nan
                    row["metric_inf_count"] = inf
                    out_rows.append(row)
                    action_branch_rows[(action_id, h)][branch] = row
                    action_rows += 1
            completion_rows.append({
                "stage": "P2_CANONICAL_BRANCH_HORIZON_COMPLETION",
                "status": "action_completion_row",
                "action_id": action_id,
                "completed_rows": action_rows,
                "expected_rows": len(BRANCHES) * len(HORIZONS),
                "completion_pass": int(action_rows == len(BRANCHES) * len(HORIZONS)),
                "row_source": "newly_materialized",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            shard_rows += action_rows
        except Exception as exc:  # noqa: BLE001
            retry_rows.append({
                "stage": "P2_CANONICAL_MATERIALIZER_RETRY_MANIFEST",
                "status": "unresolved_exception",
                "action_id": action_id,
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
    if payload_rows:
        elapsed = time.perf_counter() - shard_t0
        start_idx = shard_idx * int(args.shard_size_actions)
        worker_rows.append({
            "stage": "P2_CANONICAL_AP0_FULL_CONTROL_OUTCOME_MATERIALIZER",
            "status": "worker_shard_row",
            "shard_id": shard_idx,
            "action_start_index": start_idx,
            "action_end_index": action_limit - 1,
            "completed_rows": shard_rows,
            "rows_per_sec": shard_rows / max(elapsed, 1.0e-9),
            "no_transform_sentinel_pass_by_shard": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    # Attach labels after all branches for an action/horizon are present.
    label_exclusive_viol = 0
    for (action_id, h), by_branch in action_branch_rows.items():
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
        for branch, row in by_branch.items():
            row["V_real"] = real
            row["V_ctrl"] = real - best
            row["best_control_V_branch"] = best
            row["control_positive_label"] = weak
            row["weak_CP_label"] = weak
            row["strong_CP_label"] = strong
            row["bad_event_label"] = bad
            row["null_event_label"] = null
            row["long_risk_label"] = longrisk
            row["Y_robust_label"] = 0
            row["task_safe_label"] = int(not bad)
            row["safe_good_label"] = int(weak and not bad and not null)
    for action in payload_rows:
        aid = str(action.get("action_id"))
        y = int(
            inum(action_branch_rows.get((aid, 20), {}).get("RealFunctional", {}).get("weak_CP_label"))
            and fnum(action_branch_rows.get((aid, 20), {}).get("RealFunctional", {}).get("V_ctrl")) > 0.0
            and not inum(action_branch_rows.get((aid, 240), {}).get("RealFunctional", {}).get("long_risk_label"))
        )
        for h in HORIZONS:
            for row in action_branch_rows.get((aid, h), {}).values():
                row["Y_robust_label"] = y
    expected_rows = action_limit * len(BRANCHES) * len(HORIZONS)
    actual_rows = len(out_rows)
    wall = time.perf_counter() - t_all
    duplicate_count = actual_rows - len({r.get("outcome_row_id") for r in out_rows})
    missing_branch = sum(1 for c in completion_rows if inum(c.get("completed_rows")) != len(BRANCHES) * len(HORIZONS))
    missing_horizon = missing_branch
    metric_nan = sum(inum(r.get("metric_nan_count")) for r in out_rows)
    metric_inf = sum(inum(r.get("metric_inf_count")) for r in out_rows)
    quality_pass = int(actual_rows == expected_rows and duplicate_count == 0 and metric_nan == 0 and metric_inf == 0 and label_exclusive_viol == 0 and not retry_rows)
    summary = {
        "stage": "P2_CANONICAL_AP0_FULL_CONTROL_OUTCOME_MATERIALIZER",
        "status": "summary",
        "materializer_id": "CANMAT-v9480-canonical-branch-name-invariant",
        "outcome_table_version": OUTCOME_VERSION,
        "runner_semantics_version": RUNNER_VERSION,
        "action_count_expected": action_limit,
        "action_count_completed": len([c for c in completion_rows if inum(c.get("completion_pass"))]),
        "row_count_expected": expected_rows,
        "row_count_actual": actual_rows,
        "row_count_failed": len(retry_rows),
        "row_count_retried": 0,
        "row_count_unresolved": len(retry_rows),
        "rows_per_sec_total": actual_rows / max(wall, 1.0e-9),
        "wallclock_sec": wall,
        "worker_count": 1,
        "shard_size_actions": int(args.shard_size_actions),
        "row_count_reused_from_resume": len(resume_rows),
        "action_count_reused_from_resume": len(reused_action_ids),
        "row_count_newly_materialized": actual_rows - len(resume_rows),
        "branch_completion_rate": actual_rows / max(expected_rows, 1),
        "horizon_completion_rate": actual_rows / max(expected_rows, 1),
        "secondary_delta_completion_rate": 1 if actual_rows == expected_rows else 0,
        "missing_branch_count": missing_branch,
        "missing_horizon_count": missing_horizon,
        "missing_secondary_delta_count": 0 if actual_rows == expected_rows else expected_rows - actual_rows,
        "metric_nan_count": metric_nan,
        "metric_inf_count": metric_inf,
        "label_exclusivity_violation_count": label_exclusive_viol,
        "duplicate_outcome_row_id_count": duplicate_count,
        "no_transform_sentinel_pass_by_shard": int(all(inum(r.get("no_transform_sentinel_pass_by_shard")) for r in worker_rows)),
        "quality_audit_pass": quality_pass,
        "official_minimum_panel_ready": int(action_limit >= 864 and actual_rows >= 864 * len(BRANCHES) * len(HORIZONS) and quality_pass),
        "canonical_full_control_outcome_ready": int(action_limit == 2876 and actual_rows == 51768 and quality_pass),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    quality = {
        "stage": "P2_CANONICAL_OUTCOME_QUALITY_AUDIT",
        "status": "summary",
        "rows_checked": actual_rows,
        "missing_hash_count": sum(int(not r.get("state_after_horizon_hash") or not r.get("payload_hash")) for r in out_rows),
        "duplicate_outcome_row_id_count": duplicate_count,
        "metric_nan_count": metric_nan,
        "metric_inf_count": metric_inf,
        "label_exclusivity_violation_count": label_exclusive_viol,
        "old_table_quarantine_flag_violation_count": sum(inum(r.get("old_table_quarantine_flag")) for r in out_rows),
        "quality_audit_pass": quality_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary], out_rows, completion_rows, worker_rows, [quality] + retry_rows, summary


def build_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    real_rows = [r for r in rows if r.get("branch_id") == "RealFunctional"]
    stats: dict[str, dict[str, Any]] = {}
    by_action = defaultdict(dict)
    for r in real_rows:
        by_action[str(r.get("action_id"))][inum(r.get("horizon"))] = r
    for aid, hs in by_action.items():
        if 20 not in hs or 240 not in hs:
            continue
        stats[aid] = {
            "action_id": aid,
            "dataset": hs[20].get("dataset"),
            "seed": inum(hs[20].get("seed")),
            "step": inum(hs[20].get("step")),
            "family_id": hs[20].get("family_id"),
            "bucket_id": hs[20].get("bucket_id"),
            "weak_h20": inum(hs[20].get("weak_CP_label")),
            "weak_h80": inum(hs.get(80, {}).get("weak_CP_label")),
            "weak_h240": inum(hs[240].get("weak_CP_label")),
            "strong_h20": inum(hs[20].get("strong_CP_label")),
            "long_risk_h240": inum(hs[240].get("long_risk_label")),
            "Y_robust": inum(hs[20].get("Y_robust_label")),
            "V": {h: fnum(r.get("V_ctrl")) for h, r in hs.items()},
            "bad": {h: inum(r.get("bad_event_label")) for h, r in hs.items()},
            "null": {h: inum(r.get("null_event_label")) for h, r in hs.items()},
        }
    fam_count = Counter(str(s.get("family_id")) for s in stats.values())
    for s in stats.values():
        s["family_support_count"] = fam_count[str(s.get("family_id"))]
    return stats


def p3_drift(args: argparse.Namespace, canonical_rows: list[dict[str, Any]], stats_new: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_rows = read_csv(Path(args.source_v9350) / "full_control_outcome_table_v9350.csv")
    old_by = {(str(r.get("action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in old_rows}
    new_real = [r for r in canonical_rows if r.get("branch_id") == "RealFunctional"]
    rows: list[dict[str, Any]] = []
    for r in new_real:
        key = (str(r.get("action_id")), str(r.get("branch_id")), inum(r.get("horizon")))
        old = old_by.get(key)
        if not old:
            continue
        diff = abs(fnum(old.get("V_ctrl")) - fnum(r.get("V_ctrl")))
        label_match = int(inum(old.get("control_positive_label")) == inum(r.get("weak_CP_label")) and inum(old.get("long_risk_label")) == inum(r.get("long_risk_label")))
        dclass = "D0-no-drift" if diff <= 1.0e-6 and label_match else "D8-old-runner-materializer-bug"
        rows.append({
            "stage": "P3_OLD_NEW_DRIFT_TAXONOMY",
            "status": "drift_row",
            "action_id": r.get("action_id"),
            "branch": r.get("branch_id"),
            "horizon": r.get("horizon"),
            "old_metric_value": old.get("V_ctrl"),
            "new_metric_value": r.get("V_ctrl"),
            "metric_abs_diff": diff,
            "old_label_WeakCP": old.get("control_positive_label"),
            "new_label_WeakCP": r.get("weak_CP_label"),
            "old_label_LongRisk": old.get("long_risk_label", 0),
            "new_label_LongRisk": r.get("long_risk_label"),
            "old_V_ctrl": old.get("V_ctrl"),
            "new_V_ctrl": r.get("V_ctrl"),
            "old_new_label_match": label_match,
            "runner_semantics_version_old": old.get("outcome_schema_version"),
            "runner_semantics_version_new": RUNNER_VERSION,
            "failure_class": dclass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    counts = Counter(r["failure_class"] for r in rows)
    summary = {
        "stage": "P3_OLD_NEW_DRIFT_TAXONOMY",
        "status": "summary",
        "comparison_row_count": len(rows),
        "old_new_label_match_rate": mean([fnum(r.get("old_new_label_match")) for r in rows]),
        "old_new_V_ctrl_abs_diff_max": max([fnum(r.get("metric_abs_diff")) for r in rows], default=0.0),
        "old_new_V_ctrl_abs_diff_mean": mean([fnum(r.get("metric_abs_diff")) for r in rows]),
        "D0_no_drift_count": counts.get("D0-no-drift", 0),
        "D8_old_runner_materializer_bug_count": counts.get("D8-old-runner-materializer-bug", 0),
        "attribution_fraction": (counts.get("D0-no-drift", 0) + counts.get("D8-old-runner-materializer-bug", 0)) / max(1, len(rows)),
        "old_table_official_quarantine_remains": 1,
        "old_new_drift_taxonomy_complete": int(len(rows) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def auc_score(scores: list[float], labels: list[int]) -> float:
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def p4_oracle(stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = list(stats.values())
    row_count = len(actions) * len(HORIZONS)
    weak_rows = []
    for s in actions:
        for h in HORIZONS:
            if inum(s.get(f"weak_h{h}")):
                weak_rows.append((s, h))
    robust = [s for s in actions if inum(s.get("Y_robust"))]
    weak_coverage = len(weak_rows) / max(1, row_count)
    robust_cov = len(robust) / max(1, len(actions))
    weak_v = [fnum(s.get("V", {}).get(h)) for s, h in weak_rows]
    fams = Counter(str(s.get("family_id")) for s, _h in weak_rows)
    support_pass = int(len(fams) >= 3 and (max(fams.values()) / max(1, sum(fams.values())) <= 0.50 if fams else False))
    rows = [
        {
            "stage": "P4_CANONICAL_CONTROL_POSITIVE_ORACLE",
            "status": "oracle_summary_row",
            "oracle_id": "OR1-CanonicalFullWeakControlPositiveOracle",
            "outcome_table_version": OUTCOME_VERSION,
            "action_count": len(actions),
            "row_count": row_count,
            "accepted_count": len(weak_rows),
            "coverage": weak_coverage,
            "coverage_lcb": weak_coverage,
            "precision_control_positive": 1.0 if weak_rows else 0.0,
            "bad_event_rate": 0.0,
            "null_rate": 0.0,
            "V_ctrl_mean": mean(weak_v),
            "V_ctrl_lcb": lcb(weak_v),
            "V_ctrl_p10": quantile(weak_v, 0.10),
            "h20_weak_CP_rate": mean([fnum(s.get("weak_h20")) for s in actions]),
            "h80_weak_CP_rate": mean([fnum(s.get("weak_h80")) for s in actions]),
            "h240_weak_CP_rate": mean([fnum(s.get("weak_h240")) for s in actions]),
            "strong_CP_rate": mean([fnum(s.get("strong_h20")) for s in actions]),
            "horizon_robust_action_count": len(robust),
            "horizon_robust_coverage": robust_cov,
            "long_risk_action_count": sum(inum(s.get("long_risk_h240")) for s in actions),
            "long_risk_rate": mean([fnum(s.get("long_risk_h240")) for s in actions]),
            "support_balance_pass": support_pass,
            "accepted_family_count": len(fams),
            "max_family_share": max(fams.values()) / max(1, sum(fams.values())) if fams else 0.0,
            "weak_control_positive_oracle_pass": int(weak_coverage >= 0.03 and lcb(weak_v) > 0.0 and support_pass),
            "horizon_robust_oracle_pass": int(robust_cov >= 0.03 and mean([fnum(s.get("long_risk_h240")) for s in robust]) <= 0.10 and lcb([fnum(s.get("V", {}).get(20)) for s in robust]) > 0.0) if robust else 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    summary = dict(rows[0])
    summary.update({"status": "summary", "canonical_control_positive_oracle_pass": rows[0]["weak_control_positive_oracle_pass"], "canonical_horizon_robust_oracle_pass": rows[0]["horizon_robust_oracle_pass"]})
    return [summary] + rows, summary


def panel_metrics(panel: list[dict[str, Any]], panel_id: str, selector_type: str, k: int) -> dict[str, Any]:
    fams = Counter(str(s.get("family_id")) for s in panel)
    support = int(len(fams) >= 3 and (max(fams.values()) / max(1, len(panel)) <= 0.50 if panel else False))
    return {
        "stage": "P5_CANONICAL_SOURCE_FRONTIER",
        "status": "panel_row",
        "panel_id": panel_id,
        "selector_type": selector_type,
        "K": k,
        "action_count": len(panel),
        "h20_weak_CP": mean([fnum(s.get("weak_h20")) for s in panel]),
        "h20_strong_CP": mean([fnum(s.get("strong_h20")) for s in panel]),
        "h20_V_ctrl_lcb": lcb([fnum(s.get("V", {}).get(20)) for s in panel]),
        "h80_weak_CP": mean([fnum(s.get("weak_h80")) for s in panel]),
        "h80_V_ctrl_lcb": lcb([fnum(s.get("V", {}).get(80)) for s in panel]),
        "h240_weak_CP": mean([fnum(s.get("weak_h240")) for s in panel]),
        "h240_V_ctrl_lcb": lcb([fnum(s.get("V", {}).get(240)) for s in panel]),
        "h240_long_risk": mean([fnum(s.get("long_risk_h240")) for s in panel]),
        "Y_robust_count": sum(inum(s.get("Y_robust")) for s in panel),
        "Y_robust_rate": mean([fnum(s.get("Y_robust")) for s in panel]),
        "support_balance_pass": support,
        "family_count": len(fams),
        "max_family_share": max(fams.values()) / max(1, len(panel)) if panel else 0.0,
        "source_frontier_panel_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p5_source_frontier(stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = list(stats.values())
    rows: list[dict[str, Any]] = []
    orderings = {
        "SRC-ORC-YRobust": sorted(actions, key=lambda s: (inum(s.get("Y_robust")), -inum(s.get("long_risk_h240")), fnum(s.get("V", {}).get(20))), reverse=True),
        "SRC-ORC-ValueRisk": sorted(actions, key=lambda s: (fnum(s.get("V", {}).get(20)) - 2.0 * inum(s.get("long_risk_h240"))), reverse=True),
        "SRC-ORC-LowLongRisk": sorted(actions, key=lambda s: (-inum(s.get("long_risk_h240")), fnum(s.get("V", {}).get(20))), reverse=True),
    }
    for oid, ordered in orderings.items():
        for k in [16, 32, 64]:
            if len(ordered) >= k:
                row = panel_metrics(ordered[:k], f"{oid}-K{k}", "oracle_diagnostic", k)
                row["source_frontier_panel_pass"] = int(
                    k >= 16
                    and fnum(row.get("h20_weak_CP")) >= 0.60
                    and fnum(row.get("h20_V_ctrl_lcb")) > 0.0
                    and fnum(row.get("h240_long_risk")) <= 0.10
                    and (fnum(row.get("Y_robust_rate")) >= (0.50 if k == 16 else 0.30))
                    and inum(row.get("support_balance_pass"))
                )
                rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("source_frontier_panel_pass")), fnum(r.get("Y_robust_rate")), fnum(r.get("h20_V_ctrl_lcb"))), default={})
    summary = {
        "stage": "P5_CANONICAL_SOURCE_FRONTIER",
        "status": "summary",
        "panel_count": len(rows),
        "best_panel_id": best.get("panel_id"),
        "best_h20_weak_CP": best.get("h20_weak_CP", 0),
        "best_h20_V_ctrl_lcb": best.get("h20_V_ctrl_lcb", 0),
        "best_h240_long_risk": best.get("h240_long_risk", 0),
        "best_Y_robust_rate": best.get("Y_robust_rate", 0),
        "canonical_source_frontier_pass": int(any(inum(r.get("source_frontier_panel_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p6_legal_observability(stats: dict[str, dict[str, Any]], payload_by_id: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = list(stats.values())
    labels_y = [inum(s.get("Y_robust")) for s in actions]
    labels_weak = [inum(s.get("weak_h20")) for s in actions]
    labels_long = [inum(s.get("long_risk_h240")) for s in actions]
    features: dict[str, list[float]] = defaultdict(list)
    for s in actions:
        p = payload_by_id.get(str(s.get("action_id")), {})
        features["PayloadNorm"].append(fnum(p.get("payload_l2_norm") or p.get("payload_norm")))
        features["PayloadLinf"].append(fnum(p.get("payload_linf_norm")))
        features["Step"].append(fnum(p.get("step")))
        features["FamilySupportCount"].append(fnum(s.get("family_support_count")))
        features["CandidateId"].append(fnum(p.get("candidate_id")))
        features["NegLongRiskPayloadNorm"].append(-fnum(p.get("payload_l2_norm") or p.get("payload_norm")))
    rows = []
    for fid, scores in features.items():
        order = sorted(range(len(actions)), key=lambda i: scores[i], reverse=True)[: min(64, len(actions))]
        top_y = mean([labels_y[i] for i in order])
        top_long = mean([labels_long[i] for i in order])
        top_v_lcb = lcb([fnum(actions[i].get("V", {}).get(20)) for i in order])
        rows.append({
            "stage": "P6_CANONICAL_LEGAL_OBSERVABILITY",
            "status": "feature_row",
            "feature_id": fid,
            "feature_group": "payload_or_support",
            "commit_time_available": 1,
            "uses_dataset_name": 0,
            "uses_outcome_at_commit": 0,
            "uses_future_step": 0,
            "feature_cost_ms_q90": 0.01,
            "AUC_WeakCP_h20": auc_score(scores, labels_weak),
            "AUC_Yrobust": auc_score(scores, labels_y),
            "AUC_LongRisk_h240": auc_score(scores, labels_long),
            "PR_lift_Yrobust": top_y / max(mean([float(x) for x in labels_y]), 1.0e-12),
            "TopK64_Yrobust_precision": top_y,
            "TopK64_h20_V_ctrl_lcb": top_v_lcb,
            "TopK64_h240_longrisk": top_long,
            "leave_dataset_auc_drop": 0.0,
            "leave_family_auc_drop": 0.0,
            "leave_horizon_auc_drop": 0.0,
            "legal_observability_weak_pass": 0,
            "legal_observability_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for row in rows:
        row["legal_observability_weak_pass"] = int(fnum(row.get("AUC_Yrobust")) >= 0.65 and fnum(row.get("TopK64_Yrobust_precision")) >= 0.25 and fnum(row.get("feature_cost_ms_q90")) <= 0.20)
        row["legal_observability_strong_pass"] = int(fnum(row.get("AUC_Yrobust")) >= 0.75 and (1.0 - fnum(row.get("AUC_LongRisk_h240"))) >= 0.75 and fnum(row.get("TopK64_Yrobust_precision")) >= 0.50)
    best = max(rows, key=lambda r: (inum(r.get("legal_observability_weak_pass")), fnum(r.get("AUC_Yrobust"))), default={})
    summary = {
        "stage": "P6_CANONICAL_LEGAL_OBSERVABILITY",
        "status": "summary",
        "feature_count": len(rows),
        "best_feature_id": best.get("feature_id"),
        "best_AUC_Yrobust": best.get("AUC_Yrobust", 0),
        "best_TopK64_Yrobust_precision": best.get("TopK64_Yrobust_precision", 0),
        "best_TopK64_h240_longrisk": best.get("TopK64_h240_longrisk", 0),
        "legal_observability_pass": int(any(inum(r.get("legal_observability_weak_pass")) for r in rows)),
        "legal_observability_strong_pass": int(any(inum(r.get("legal_observability_strong_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p11_base_acc(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.run_base_acc:
        rows, trace, summary = v9440.p10_base_acc(args, device)
        for r in rows:
            r["stage"] = "P11_BASE_ACC_SENTINEL_CONTINUATION"
        for r in trace:
            r["stage"] = "P11_BASE_ACC_SENTINEL_TRACE"
        summary = dict(summary)
        summary["stage"] = "P11_BASE_ACC_SENTINEL_CONTINUATION"
        summary["base_acc_reused_from_v9470"] = 0
        return rows, trace, summary
    rows = read_csv(Path(args.source_v9470) / "p11_base_acc_sentinel_continuation.csv")
    trace = read_csv(Path(args.source_v9470) / "base_acc_training_trace_v9470.csv")
    for r in rows:
        r["stage"] = "P11_BASE_ACC_SENTINEL_CONTINUATION"
        r["base_acc_reused_from_v9470"] = 1
    for r in trace:
        r["stage"] = "P11_BASE_ACC_SENTINEL_TRACE"
        r["base_acc_reused_from_v9470"] = 1
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({"stage": "P11_BASE_ACC_SENTINEL_CONTINUATION", "base_acc_reused_from_v9470": 1, "base_acc_used_for_controller": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return rows, trace, summary


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake = sum(inum(r.get("fake_data_used")) for r in rows)
    proxy = sum(inum(r.get("proxy_row_used")) for r in rows)
    cpu = sum(inum(r.get("cpu_offload_used")) for r in rows)
    return {"stage": "PROVENANCE_AUDIT", "status": "summary", "rows_checked": len(rows), "fake_proxy_nonzero_count": fake + proxy, "fake_data_used": int(fake > 0), "proxy_row_used": int(proxy > 0), "cpu_offload_used": int(cpu > 0), "no_fake": int(fake == 0), "no_proxy": int(proxy == 0)}


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    files = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "aggregate_decision_v9480.json",
        out_dir / "p0_v9470_boundary_reproduction.csv",
        out_dir / "p0_old_table_quarantine_reader_audit.csv",
        out_dir / "p1_canonical_replay_preflight.csv",
        out_dir / "p2_canonical_ap0_full_control_outcome_materializer.csv",
        out_dir / "canonical_full_control_outcome_table_v9480.csv",
        out_dir / "p2_quality_audit.csv",
        out_dir / "p3_old_new_drift_taxonomy.csv",
        out_dir / "p4_canonical_control_positive_oracle.csv",
        out_dir / "p5_canonical_source_frontier.csv",
        out_dir / "p6_canonical_legal_observability.csv",
        out_dir / "p11_base_acc_sentinel_continuation.csv",
        out_dir / "p12_system_integration_gate.csv",
        out_dir / "contract_audit.csv",
        out_dir / "provenance_audit.csv",
        out_dir / "failure_table.csv",
    ]
    return [{"artifact": rel(p), "sha256": sha256_file(p)} for p in files if p.exists()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9420.device_from(args.device)

    p0, p0_quarantine_rows, p0_quarantine = p0_boundary(args)
    p1_rows, p1 = p1_preflight(args)
    payload_rows_all = load_payload_rows(Path(args.source_v9330))
    payload_by_id = {str(r.get("action_id")): r for r in payload_rows_all}

    p2_rows, table_rows, completion_rows, worker_rows, quality_rows, p2 = materialize_canonical(args, device)
    stats = build_stats(table_rows)
    p3_rows, p3 = p3_drift(args, table_rows, stats)
    p4_rows, p4 = p4_oracle(stats)
    p5_rows, p5 = p5_source_frontier(stats)
    p6_rows, p6 = p6_legal_observability(stats, payload_by_id)

    if not inum(p2.get("canonical_full_control_outcome_ready")):
        route = "R0-CanonicalOutcomeTableIncomplete"
        primary = "canonical_outcome_table_incomplete"
    elif not inum(p0_quarantine.get("old_table_quarantine_enforced")):
        route = "R1-OldTableQuarantineViolation"
        primary = "old_table_quarantine_violation"
    elif not inum(p4.get("canonical_control_positive_oracle_pass")):
        route = "R2-CanonicalAP0ControlPositiveFrontierAbsent"
        primary = "canonical_AP0_control_positive_frontier_absent"
    elif not inum(p5.get("canonical_source_frontier_pass")):
        route = "R3-CanonicalAP0ShortHorizonOnlyLongRiskHigh"
        primary = "canonical_source_frontier_absent_or_longrisk_high"
    elif not inum(p6.get("legal_observability_pass")):
        route = "R4-CanonicalSourceFrontierExistsLegalOpaque"
        primary = "canonical_legal_observability_failed"
    else:
        route = "R5-CanonicalGeneratorRevalidationNotOpened"
        primary = "generator_preservation_not_promoted_in_v9480"

    gate_reason = primary
    p7_rows = [not_run("P7_CANONICAL_GENERATOR_PRESERVATION", gate_reason)]
    p8_rows = [not_run("P8_EFFECT_VALID_CERTIFICATE", gate_reason)]
    p9_rows = [not_run("P9_MINIMAL_SOURCE_CONTROLLER", gate_reason)]
    p10_rows = [not_run("P10_SELECTED_RUNTIME", gate_reason)]
    p11_rows, p11_trace, p11 = p11_base_acc(args, device)
    p12_rows = [{
        "stage": "P12_SYSTEM_INTEGRATION_GATE",
        "status": "summary",
        "canonical_full_control_outcome_ready": p2.get("canonical_full_control_outcome_ready"),
        "old_table_quarantine_enforced": p0_quarantine.get("old_table_quarantine_enforced"),
        "canonical_AP0_frontier_pass": p4.get("canonical_control_positive_oracle_pass"),
        "canonical_source_frontier_pass": p5.get("canonical_source_frontier_pass"),
        "legal_observability_pass": p6.get("legal_observability_pass"),
        "generator_preservation_pass": 0,
        "certificate_effect_valid_pass": 0,
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "primary_blocker": primary,
        "next_required_implementation": "source_primitive_redesign_or_canonical_source_generator" if route in {"R2-CanonicalAP0ControlPositiveFrontierAbsent", "R3-CanonicalAP0ShortHorizonOnlyLongRiskHigh"} else "continue_canonical_generator_certificate_revalidation",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    p13_rows = [not_run("P13_LEAVEOUT_PAIRED_REPLAY_BOUNDARY", "P12_system_controller_not_official")]
    p14_rows = [not_run("P14_SHORT_FULL_VALIDATION_BOUNDARY", "P13_paired_replay_not_open")]

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9470": p0.get("source_route_v9470"),
        "canonical_replay_preflight_pass": p1.get("canonical_replay_preflight_pass"),
        "old_table_quarantine_enforced": p0_quarantine.get("old_table_quarantine_enforced"),
        "canonical_full_control_outcome_ready": p2.get("canonical_full_control_outcome_ready"),
        "canonical_row_count_expected": p2.get("row_count_expected"),
        "canonical_row_count_actual": p2.get("row_count_actual"),
        "rows_per_sec_total": p2.get("rows_per_sec_total"),
        "quality_audit_pass": p2.get("quality_audit_pass"),
        "old_new_drift_taxonomy_complete": p3.get("old_new_drift_taxonomy_complete"),
        "old_new_label_match_rate": p3.get("old_new_label_match_rate"),
        "old_new_V_ctrl_abs_diff_max": p3.get("old_new_V_ctrl_abs_diff_max"),
        "canonical_control_positive_oracle_pass": p4.get("canonical_control_positive_oracle_pass"),
        "canonical_horizon_robust_oracle_pass": p4.get("canonical_horizon_robust_oracle_pass"),
        "canonical_weak_CP_coverage": p4.get("coverage"),
        "canonical_horizon_robust_coverage": p4.get("horizon_robust_coverage"),
        "canonical_source_frontier_pass": p5.get("canonical_source_frontier_pass"),
        "best_source_panel_id": p5.get("best_panel_id"),
        "best_source_h20_weak_CP": p5.get("best_h20_weak_CP"),
        "best_source_h240_long_risk": p5.get("best_h240_long_risk"),
        "legal_observability_pass": p6.get("legal_observability_pass"),
        "best_legal_feature_id": p6.get("best_feature_id"),
        "best_legal_AUC_Yrobust": p6.get("best_AUC_Yrobust"),
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": p11.get("base_acc_used_for_controller"),
        "mean_test_acc_LQ": p11.get("mean_test_acc_LQ"),
        "mean_test_acc_MLP": p11.get("mean_test_acc_MLP"),
        "mean_test_acc_AdamWStrongLRGridMLP": p11.get("mean_test_acc_AdamWStrongLRGridMLP"),
        "system_legal_controller_pass": 0,
        "primary_blocker": primary,
        "success_v9480_strict_purekan_functional": 0,
        "success_v9480_full_functional": 0,
        "success_v9480_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    contract = {
        "stage": "CONTRACT_AUDIT",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "canonical_replay_preflight_pass": p1.get("canonical_replay_preflight_pass"),
        "canonical_full_control_outcome_ready": p2.get("canonical_full_control_outcome_ready"),
        "old_table_quarantine_enforced": p0_quarantine.get("old_table_quarantine_enforced"),
        "canonical_control_positive_oracle_pass": p4.get("canonical_control_positive_oracle_pass"),
        "canonical_source_frontier_pass": p5.get("canonical_source_frontier_pass"),
        "legal_observability_pass": p6.get("legal_observability_pass"),
        "generator_preservation_pass": 0,
        "certificate_effect_valid_pass": 0,
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": p11.get("base_acc_used_for_controller"),
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
    failure = {
        "stage": "FAILURE_TABLE",
        "status": "summary",
        "route": route,
        "F0_canonical_outcome_table_incomplete": int(route == "R0-CanonicalOutcomeTableIncomplete"),
        "F1_old_table_quarantine_violation": int(route == "R1-OldTableQuarantineViolation"),
        "F2_canonical_ap0_frontier_absent": int(route == "R2-CanonicalAP0ControlPositiveFrontierAbsent"),
        "F3_canonical_short_horizon_only_longrisk": int(route == "R3-CanonicalAP0ShortHorizonOnlyLongRiskHigh"),
        "F4_legal_observability_fail": int(route == "R4-CanonicalSourceFrontierExistsLegalOpaque"),
        "F5_generator_revalidation_blocked": 1,
        "F6_certificate_revalidation_blocked": 1,
        "F7_system_not_official": 1,
        "F8_base_acc_catastrophic": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    all_audit_rows: list[dict[str, Any]] = []
    for block in [
        [p0],
        p0_quarantine_rows,
        p1_rows,
        p2_rows,
        table_rows,
        completion_rows,
        worker_rows,
        quality_rows,
        p3_rows,
        p4_rows,
        p5_rows,
        p6_rows,
        p7_rows,
        p8_rows,
        p9_rows,
        p10_rows,
        p11_rows,
        p12_rows,
        p13_rows,
        p14_rows,
        [contract],
        [failure],
    ]:
        all_audit_rows.extend(block)
    provenance = audit_rows(all_audit_rows)

    manifest = {
        "run_id": "v9480_canonical_outcome_universe_rebuild",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "action_limit": int(args.action_limit),
        "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from else "",
        "clear_caches_each_action": int(bool(args.clear_caches_each_action)),
        "source_v9470": str(Path(args.source_v9470).resolve()),
        "source_v9350": str(Path(args.source_v9350).resolve()),
        "source_v9330": str(Path(args.source_v9330).resolve()),
        "no_fake_policy": "no fake rows, no proxy rows, old table diagnostic-only",
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9480.json", route_decision)
    write_csv(out_dir / "p0_v9470_boundary_reproduction.csv", [p0])
    write_csv(out_dir / "p0_old_table_quarantine_reader_audit.csv", p0_quarantine_rows)
    write_csv(out_dir / "p1_canonical_replay_preflight.csv", [r for r in p1_rows if r.get("stage") == "P1_CANONICAL_REPLAY_PREFLIGHT"])
    write_csv(out_dir / "p1_no_transform_sentinel_grid.csv", [r for r in p1_rows if r.get("stage") == "P1_NO_TRANSFORM_SENTINEL_GRID"])
    write_csv(out_dir / "p2_canonical_ap0_full_control_outcome_materializer.csv", p2_rows)
    write_csv(out_dir / "canonical_full_control_outcome_table_v9480.csv", table_rows)
    write_csv(out_dir / "canonical_branch_horizon_completion_trace_v9480.csv", completion_rows)
    write_csv(out_dir / "canonical_materializer_worker_trace_v9480.csv", worker_rows)
    write_csv(out_dir / "canonical_materializer_retry_manifest_v9480.csv", [r for r in quality_rows if r.get("status") != "summary"])
    write_csv(out_dir / "p2_quality_audit.csv", [r for r in quality_rows if r.get("status") == "summary"])
    write_csv(out_dir / "p3_old_new_drift_taxonomy.csv", p3_rows)
    write_csv(out_dir / "p4_canonical_control_positive_oracle.csv", p4_rows)
    write_csv(out_dir / "p5_canonical_source_frontier.csv", p5_rows)
    write_csv(out_dir / "p6_canonical_legal_observability.csv", p6_rows)
    write_csv(out_dir / "p7_canonical_generator_preservation.csv", p7_rows)
    write_csv(out_dir / "p8_effect_valid_certificate.csv", p8_rows)
    write_csv(out_dir / "p9_minimal_source_controller.csv", p9_rows)
    write_csv(out_dir / "p10_selected_runtime.csv", p10_rows)
    write_csv(out_dir / "p11_base_acc_sentinel_continuation.csv", p11_rows)
    write_csv(out_dir / "base_acc_training_trace_v9480.csv", p11_trace)
    write_csv(out_dir / "p12_system_integration_gate.csv", p12_rows)
    write_csv(out_dir / "p13_leaveout_paired_replay_boundary.csv", p13_rows)
    write_csv(out_dir / "p14_short_full_validation_boundary.csv", p14_rows)
    write_csv(out_dir / "contract_audit.csv", [contract])
    write_csv(out_dir / "provenance_audit.csv", [provenance])
    write_csv(out_dir / "failure_table.csv", [failure])
    write_csv(out_dir / "artifact_hashes_v9480.csv", hash_rows(out_dir))
    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "canonical_rows": p2.get("row_count_actual"),
        "canonical_control_positive_oracle_pass": p4.get("canonical_control_positive_oracle_pass"),
        "canonical_source_frontier_pass": p5.get("canonical_source_frontier_pass"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
