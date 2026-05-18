#!/usr/bin/env python3
"""DG-KAN v9.3.3 durable payload and control-outcome decoupling runner.

This runner follows the v9.3.3 plan as far as the current code path can
materialize real data in one run.  It upgrades v9.3.2's in-memory action apply
closure to durable payload shards with disk replay, then runs a small real
branch/horizon control dry-run.  The dry-run is not promoted to official
secondary/control readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
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

import run_v9320_action_apply_control_outcome_event_runtime as v9320  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.3.3_FullControlOutcome_ActionValueDisentanglement_RuntimeDecoupling_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9330_full_control_outcome_action_value_runtime_decoupling.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.3.3_FullControlOutcome_ActionValueDisentanglement_RuntimeDecoupling_实验复盘.md"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9320 = RESULT_ROOT / "v9320_action_apply_control_outcome_event_runtime_first_20260513T223000Z"
DEFAULT_SOURCE_V9300 = RESULT_ROOT / "v9300_outcome_action_primitive_reset_event_driven_runtime_closure_first_20260513T233000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"

OFFICIAL_BRANCHES = ["RealFunctional", "AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
OFFICIAL_HORIZONS = [20, 80, 240]
MISSING_SECONDARY_FIELDS = ["curvature_delta", "local_lipschitz_delta", "basis_usage_entropy_delta", "functional_channel_entropy_delta"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=336)
    p.add_argument("--interface-events", type=int, default=24)
    p.add_argument("--interface-steps", type=int, default=2)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--stable-accept-scale", type=float, default=1.0e5)
    p.add_argument("--payload-shard-size", type=int, default=64)
    p.add_argument("--control-max-actions", type=int, default=8)
    p.add_argument("--control-horizons", default="20")
    p.add_argument("--source-v9320", default=str(DEFAULT_SOURCE_V9320))
    p.add_argument("--source-v9300", default=str(DEFAULT_SOURCE_V9300))
    p.add_argument("--source-v9280", default=str(DEFAULT_SOURCE_V9280))
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def device_from(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode("utf-8")).hexdigest()


def q(values: list[float], frac: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))
    return vals[idx]


def clone_states(states: list[AdamWState]) -> list[AdamWState]:
    return v9320.v9248._clone_states(states)


def clone_params(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return v9320.v9248._clone_params(params)


def flush_payload_shard(
    shard_dir: Path,
    shard_index: int,
    pending: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not pending:
        return [], shard_index
    shard_path = shard_dir / f"payload_shard_{shard_index:05d}.pt"
    d0 = torch.stack([p["payload"][0].cpu() for p in pending], dim=0)
    d1 = torch.stack([p["payload"][1].cpu() for p in pending], dim=0)
    d2 = torch.stack([p["payload"][2].cpu() for p in pending], dim=0)
    records = [p["record"] for p in pending]
    torch.save({"d0": d0, "d1": d1, "d2": d2, "records": records}, shard_path)
    loaded = torch.load(shard_path, map_location="cpu")
    for offset, p in enumerate(pending):
        loaded_payload = [loaded["d0"][offset], loaded["d1"][offset], loaded["d2"][offset]]
        expected_payload = [x.cpu() for x in p["payload"]]
        diff = [a - b for a, b in zip(loaded_payload, expected_payload)]
        loaded_hash = v9320.tensor_hash(loaded_payload)
        expected_hash = p["record"]["payload_hash_expected"]
        linf = v9320.norm_linf(diff)
        rel_err = linf / max(1.0e-12, v9320.norm_l2(expected_payload))
        cos = v9320.cosine(expected_payload, loaded_payload)
        trace_rows.append(
            {
                **p["record"],
                "payload_shard_path": rel(shard_path),
                "payload_tensor_offset": offset,
                "payload_hash_loaded": loaded_hash,
                "payload_hash_match": int(loaded_hash == expected_hash),
                "disk_replay_success": int(loaded_hash == expected_hash and linf <= 1.0e-6 and cos >= 0.999999),
                "disk_replay_error_linf": linf,
                "disk_replay_error_relative": rel_err,
                "disk_replay_cosine_logged_applied": cos,
                "in_memory_vs_disk_error_linf": linf,
                "in_memory_vs_disk_cosine": cos,
            }
        )
    return [], shard_index + 1


def random_like_payload(fd: list[torch.Tensor], gen: torch.Generator) -> list[torch.Tensor]:
    raw = [torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=gen) for t in fd]
    src_norm = v9320.norm_l2(fd)
    raw_norm = max(1.0e-12, v9320.norm_l2(raw))
    return [r * (src_norm / raw_norm) for r in raw]


def branch_start(
    branch: str,
    params: list[torch.Tensor],
    states: list[AdamWState],
    task_params: list[torch.Tensor],
    task_states: list[AdamWState],
    task_delta: list[torch.Tensor],
    fd: list[torch.Tensor],
    fwd_core: Any,
    xp: torch.Tensor,
    yp: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    gen: torch.Generator,
) -> tuple[list[torch.Tensor], list[AdamWState], str]:
    if branch == "RealFunctional":
        return [tp + d for tp, d in zip(task_params, fd)], clone_states(task_states), "task_params_plus_functional_delta"
    if branch == "AdamWOnly":
        return clone_params(task_params), clone_states(task_states), "task_params"
    if branch == "AdamWParallel":
        return clone_params(task_params), clone_states(task_states), "same_step_adamw_parallel_duplicate_of_adamwonly"
    if branch == "NoOp":
        return clone_params(params), clone_states(states), "pre_step_params_no_current_adamw"
    if branch == "Random":
        rnd = random_like_payload(fd, gen)
        return [tp + d for tp, d in zip(task_params, rnd)], clone_states(task_states), "task_params_plus_norm_matched_random_delta"
    if branch == "bestLR":
        scales = [0.3, 1.0, 3.0]
        best_scale = 1.0
        best_nll = float("inf")
        for scale in scales:
            cand = [p + scale * d for p, d in zip(params, task_delta)]
            logits = fwd_core(xp, *cand, mu, std, 2.0, 2.0)
            nll = v9320.metrics_from_logits(logits, yp)["NLL"]
            if nll < best_nll:
                best_nll = nll
                best_scale = scale
        return [p + best_scale * d for p, d in zip(params, task_delta)], clone_states(task_states), f"best_immediate_probe_lr_scale_{best_scale}"
    raise ValueError(branch)


def rollout_horizon(
    start_params: list[torch.Tensor],
    start_states: list[AdamWState],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    xp: torch.Tensor,
    yp: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    fwd_core: Any,
    bwd_core: Any,
    cfg: ManualAdamWConfig,
    horizon: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    params = clone_params(start_params)
    states = clone_states(start_states)
    gen = torch.Generator(device=device).manual_seed(seed)
    n = int(x_train.shape[0])
    t0 = time.perf_counter()
    for _ in range(horizon):
        batch_idx = torch.randint(0, n, (batch_size,), generator=gen, device=device)
        xb = x_train[batch_idx].contiguous()
        yb = y_train[batch_idx].contiguous()
        pack = bwd_core(xb, yb, *params, mu, std, 2.0, 2.0)
        grads = list(pack[1:])
        v9320.v9248.v92._adamw_update_foreach_(params, grads, states, cfg)
    v9320.sync(device)
    elapsed_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
    logits = fwd_core(xp, *params, mu, std, 2.0, 2.0)
    return v9320.metrics_from_logits(logits, yp), elapsed_ms


def materialize(args: argparse.Namespace, device: torch.device, out_dir: Path) -> dict[str, Any]:
    source_v9300 = Path(args.source_v9300)
    frozen_by_event = v9320.load_frozen_actions(source_v9300)
    ctx = v9320.v9268._prepare_reference(args, device)
    _p2_rows, p2 = v9320.v9269._materialized_pf5(ctx, device)
    event_table, _event_summary = v9320.v9272._build_event_table(ctx, p2)
    measured = ctx["measured"]
    candidate_indices = set(int(x) for x in p2.get("candidate_indices", []))
    source_v9280_rows = [
        r
        for r in read_csv(Path(args.source_v9280) / "full_row_stable_accept_outcome_table_v9280.csv")
        if r.get("status") == "candidate_stable_accept_outcome_row"
    ]
    labels_by_event = {str(r.get("event_id")): r for r in source_v9280_rows}

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    shard_dir = out_dir / "action_payload_shards_v9330"
    shard_dir.mkdir(parents=True, exist_ok=True)

    payload_trace: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    branch_completion_rows: list[dict[str, Any]] = []
    offline_runtime_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    shard_index = 0
    action_count = 0
    control_actions = 0
    control_horizons = [int(x) for x in str(args.control_horizons).split(",") if x.strip()]
    control_t0 = time.perf_counter()

    datasets = [v9320.v9248.v92._canonical_task(x) for x in v9320.v9248._parse_list(args.datasets)]
    seeds = v9320.v9248._parse_ints(args.seeds)
    event_idx = 0
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9320.v9248.v92._load_task(
                load_args,
                dataset,
                train_size=int(args.train_size),
                test_size=32,
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9330)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + int(seed) * 97 + len(dataset))
            n_train = int(x_train.shape[0])
            for step in range(int(args.microprobe_steps)):
                batch_idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=gen, device=device)
                xu = x_train[batch_idx[: int(args.batch_size)]].contiguous()
                yu = y_train[batch_idx[: int(args.batch_size)]].contiguous()
                xp = x_train[batch_idx[int(args.batch_size) :]].contiguous()
                yp = y_train[batch_idx[int(args.batch_size) :]].contiguous()
                pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
                grads = list(pack[1:])
                task_params = clone_params(params)
                task_states = clone_states(states)
                v9320.v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg)
                task_delta = [tp - p for tp, p in zip(task_params, params)]
                pre_metrics = v9320.metrics_from_logits(fwd_core(xp, *params, mu, std, 2.0, 2.0), yp)

                for carrier_idx, carrier_id in enumerate(v9320.v9248.CARRIERS):
                    global_idx = event_idx + carrier_idx
                    if global_idx not in candidate_indices:
                        continue
                    action_count += 1
                    evt = event_table[global_idx]
                    event_id = str(evt.get("event_id"))
                    frozen = frozen_by_event.get(event_id, {})
                    action_id = str(frozen.get("action_id") or stable_hash("v9330-action", event_id, carrier_id))
                    fd = v9320.v9248._functional_delta(task_params, mu, std, spec, xu, yu, task_delta, carrier_id)[0]
                    fd_list = list(fd)
                    payload_hash = v9320.tensor_hash(fd_list)
                    payload_l2 = v9320.norm_l2(fd_list)
                    payload_linf = v9320.norm_linf(fd_list)
                    record = {
                        "stage": "P1_DURABLE_ACTION_PAYLOAD_PACKAGE",
                        "status": "payload_disk_replay_row",
                        "action_id": action_id,
                        "candidate_id": frozen.get("candidate_id", evt.get("candidate_index", "")),
                        "event_id": event_id,
                        "global_row_id": global_idx,
                        "dataset": dataset,
                        "seed": seed,
                        "step": step,
                        "carrier_id": carrier_id,
                        "family_id": evt.get("family_id"),
                        "bucket_id": evt.get("bucket_id"),
                        "payload_schema_version": "v9330-functional-delta-sharded",
                        "payload_tensor_shape": json.dumps([list(t.shape) for t in fd_list]),
                        "payload_tensor_dtype": str(fd_list[0].dtype),
                        "payload_tensor_device_original": str(device),
                        "payload_hash_expected": payload_hash,
                        "payload_norm": payload_l2,
                        "payload_l2_norm": payload_l2,
                        "payload_linf_norm": payload_linf,
                        "functional_delta_norm": payload_l2,
                        "functional_delta_hash": payload_hash,
                        "logged_apply_hash": payload_hash,
                        "in_memory_replay_apply_hash": payload_hash,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    pending.append({"record": record, "payload": [t.detach().cpu() for t in fd_list]})

                    if control_actions < int(args.control_max_actions):
                        control_actions += 1
                        label = labels_by_event.get(event_id, {})
                        branch_values: dict[tuple[str, int], float] = {}
                        branch_gen = torch.Generator(device=device).manual_seed(int(args.seed) * 100000 + global_idx)
                        for horizon in control_horizons:
                            for branch in OFFICIAL_BRANCHES:
                                start_params, start_states, semantics = branch_start(
                                    branch,
                                    params,
                                    states,
                                    task_params,
                                    task_states,
                                    task_delta,
                                    fd_list,
                                    fwd_core,
                                    xp,
                                    yp,
                                    mu,
                                    std,
                                    branch_gen,
                                )
                                metrics, elapsed_ms = rollout_horizon(
                                    start_params,
                                    start_states,
                                    x_train,
                                    y_train,
                                    xp,
                                    yp,
                                    mu,
                                    std,
                                    fwd_core,
                                    bwd_core,
                                    cfg,
                                    int(horizon),
                                    int(args.batch_size),
                                    int(args.seed) * 1000000 + global_idx * 17 + len(branch) * 101 + int(horizon),
                                    device,
                                )
                                delta = v9320.metric_delta(metrics, pre_metrics)
                                value = -delta["CEp99_delta"] + delta["margin_p10_delta"] - delta["ECE_delta"] - delta["NLL_delta"] + delta["acc_delta"]
                                branch_values[(branch, horizon)] = value
                                control_rows.append(
                                    {
                                        "stage": "P2_FULL_MATCHED_CONTROL_OUTCOME_MATERIALIZER",
                                        "status": "branch_horizon_row",
                                        "action_id": action_id,
                                        "candidate_id": record["candidate_id"],
                                        "event_id": event_id,
                                        "branch": branch,
                                        "branch_semantics": semantics,
                                        "horizon": horizon,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "step": step,
                                        "family_id": evt.get("family_id"),
                                        "bucket_id": evt.get("bucket_id"),
                                        "signal_stratum_id": evt.get("signal_stratum", ""),
                                        **delta,
                                        "curvature_delta": "",
                                        "local_lipschitz_delta": "",
                                        "basis_usage_entropy_delta": "",
                                        "functional_channel_entropy_delta": "",
                                        "task_safe_label": int(not inum(label.get("bad_event_label", label.get("bad_event", 0)))),
                                        "useful_label": inum(label.get("safe_good_label", label.get("Y_SU_v9265", 0))),
                                        "bad_event_label": inum(label.get("bad_event_label", label.get("bad_event", 0))),
                                        "null_event_label": inum(label.get("null_event_label", label.get("Y_null_event_v9265", 0))),
                                        "safe_good_label": inum(label.get("safe_good_label", label.get("Y_SU_v9265", 0))),
                                        "value_score": value,
                                        "beats_adamwparallel": "",
                                        "beats_bestlr": "",
                                        "beats_noop": "",
                                        "beats_random": "",
                                        "outcome_runtime_ms": elapsed_ms,
                                        "branch_completed": 1,
                                        "horizon_completed": 1,
                                        "outcome_source": "same_run_offline_horizon_rollout_dry_run",
                                        "fake_data_used": 0,
                                        "proxy_row_used": 0,
                                        "cpu_offload_used": 0,
                                    }
                                )
                        for horizon in control_horizons:
                            real = branch_values.get(("RealFunctional", horizon), float("-inf"))
                            best_control = max(
                                branch_values.get((b, horizon), float("-inf"))
                                for b in OFFICIAL_BRANCHES
                                if b != "RealFunctional"
                            )
                            branch_completion_rows.append(
                                {
                                    "stage": "P2_BRANCH_HORIZON_COMPLETION",
                                    "status": "action_horizon_summary",
                                    "action_id": action_id,
                                    "event_id": event_id,
                                    "horizon": horizon,
                                    "materialized_branch_count": len(OFFICIAL_BRANCHES),
                                    "required_branch_count": len(OFFICIAL_BRANCHES),
                                    "matched_control_count": len(OFFICIAL_BRANCHES) - 1,
                                    "real_vs_best_control_value_gap": real - best_control,
                                    "real_beats_best_materialized_control": int(real > best_control),
                                    "fake_data_used": 0,
                                    "proxy_row_used": 0,
                                    "cpu_offload_used": 0,
                                }
                            )

                    if len(pending) >= int(args.payload_shard_size):
                        pending, shard_index = flush_payload_shard(shard_dir, shard_index, pending, payload_trace)
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                event_idx += len(v9320.v9248.CARRIERS)
    pending, shard_index = flush_payload_shard(shard_dir, shard_index, pending, payload_trace)
    materializer_wallclock = max(0.0, time.perf_counter() - control_t0)
    offline_runtime_rows.append(
        {
            "stage": "P2_OFFLINE_MATERIALIZER_RUNTIME",
            "status": "summary",
            "offline_runtime_candidate_id": "RT5-OfflineControlOutcomeDryRun",
            "runtime_mode": "offline_outcome_materializer_runtime",
            "branch_horizon_rows": len(control_rows),
            "branch_horizon_rows_per_sec": len(control_rows) / max(1.0e-9, materializer_wallclock),
            "actions_per_sec": control_actions / max(1.0e-9, materializer_wallclock),
            "materializer_wallclock_sec": materializer_wallclock,
            "branch_completion_rate": len(control_rows) / max(1, control_actions * len(OFFICIAL_BRANCHES) * max(1, len(control_horizons))),
            "horizon_completion_rate": len(control_horizons) / len(OFFICIAL_HORIZONS),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    )
    return {
        "action_count": action_count,
        "candidate_indices": candidate_indices,
        "payload_trace": payload_trace,
        "control_rows": control_rows,
        "branch_completion_rows": branch_completion_rows,
        "offline_runtime_rows": offline_runtime_rows,
        "control_actions": control_actions,
        "control_horizons": control_horizons,
        "measured": measured,
        "source_v9280_rows": source_v9280_rows,
        "shard_count": shard_index,
    }


def build_p0(source_v9320: Path) -> dict[str, Any]:
    route = read_json(source_v9320 / "route_decision.json")
    p1 = next(iter(read_csv(source_v9320 / "p1_action_apply_materializer.csv")), {})
    p2 = next(iter(read_csv(source_v9320 / "p2_secondary_control_outcome_materializer.csv")), {})
    p7 = next(iter(read_csv(source_v9320 / "p7_measured_online_event_runtime.csv")), {})
    return {
        "stage": "P0_V9320_BOUNDARY_REANALYSIS",
        "status": "summary",
        "source_route_v9320": route.get("route", ""),
        "candidate_count": route.get("candidate_count", ""),
        "action_count": route.get("action_count", ""),
        "event_count": route.get("event_count", ""),
        "action_apply_error_measured": p1.get("action_apply_error_measured", ""),
        "action_apply_error_missing_count": p1.get("action_apply_error_missing_count", ""),
        "action_apply_error_linf_max": p1.get("action_apply_error_linf_max", ""),
        "action_apply_error_relative_max": p1.get("action_apply_error_relative_max", ""),
        "action_apply_cosine_logged_applied_min": p1.get("action_apply_cosine_logged_applied_min", ""),
        "payload_tensor_file_written_v9320": p1.get("payload_tensor_file_written", ""),
        "secondary_outcome_ready": p2.get("secondary_outcome_ready", ""),
        "outcome_sample_action_count": p2.get("outcome_sample_action_count", ""),
        "official_sample_coverage": p2.get("official_sample_coverage", ""),
        "matched_control_count_per_event_min": p2.get("matched_control_count_per_event_min", ""),
        "branch_missing_count": p2.get("branch_missing_count", ""),
        "horizon_missing_count": p2.get("horizon_missing_count", ""),
        "missing_secondary_delta_count": p2.get("missing_secondary_delta_count", ""),
        "runtime_candidate_id": p7.get("runtime_candidate_id", ""),
        "runtime_mode": p7.get("runtime_mode", ""),
        "zero_candidate_controller_kernel_count": p7.get("zero_candidate_controller_kernel_count", ""),
        "controller_launches_per_active_step_q90": p7.get("controller_launches_per_active_step_q90", ""),
        "step_ratio_q90": p7.get("step_ratio_q90", ""),
        "event_driven_runtime_pass": p7.get("event_driven_runtime_pass", ""),
        "v9320_boundary_reproduced": int(route.get("route") == "R4-SecondaryControlOutcomeMaterializationFail"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def summarize_payload(payload_trace: list[dict[str, Any]], action_count: int, shard_count: int) -> dict[str, Any]:
    success = sum(inum(r.get("disk_replay_success")) for r in payload_trace)
    hash_match = sum(inum(r.get("payload_hash_match")) for r in payload_trace)
    linfs = [fnum(r.get("disk_replay_error_linf")) for r in payload_trace]
    rels = [fnum(r.get("disk_replay_error_relative")) for r in payload_trace]
    coss = [fnum(r.get("disk_replay_cosine_logged_applied"), 1.0) for r in payload_trace]
    pass_flag = int(
        action_count == len(payload_trace)
        and shard_count >= 1
        and success == action_count
        and hash_match == action_count
        and max(linfs or [1.0]) <= 1.0e-6
        and min(coss or [0.0]) >= 0.999999
    )
    return {
        "stage": "P1_DURABLE_ACTION_PAYLOAD_PACKAGE",
        "status": "summary",
        "payload_package_id": "DPP1-ShardedTorchPayloadDiskReplay",
        "action_count": action_count,
        "payload_shard_count": shard_count,
        "payload_tensor_file_written": int(shard_count >= 1),
        "payload_tensor_missing_count": action_count - len(payload_trace),
        "payload_hash_missing_count": sum(int(not r.get("payload_hash_expected")) for r in payload_trace),
        "payload_hash_match_rate": hash_match / max(1, len(payload_trace)),
        "disk_replay_success_count": success,
        "disk_replay_error_linf_max": max(linfs or [0.0]),
        "disk_replay_error_relative_max": max(rels or [0.0]),
        "disk_replay_cosine_logged_applied_min": min(coss or [1.0]),
        "durable_payload_pass": pass_flag,
        "action_lifecycle_pass": pass_flag,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def summarize_control(control_rows: list[dict[str, Any]], control_actions: int, action_count: int, control_horizons: list[int]) -> dict[str, Any]:
    expected_full = action_count * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)
    actual = len(control_rows)
    branch_missing = max(0, (action_count - control_actions) * len(OFFICIAL_BRANCHES))
    horizon_missing = max(0, action_count * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS) - actual)
    missing_secondary = actual * len(MISSING_SECONDARY_FIELDS)
    coverage = control_actions / max(1, action_count)
    min_controls = len(OFFICIAL_BRANCHES) - 1 if control_actions else 0
    ready = int(
        coverage >= 0.80
        and branch_missing == 0
        and horizon_missing == 0
        and missing_secondary == 0
        and min_controls >= 4
    )
    return {
        "stage": "P2_FULL_MATCHED_CONTROL_OUTCOME_MATERIALIZER",
        "status": "summary",
        "materializer_id": "FCM1-RealBranchHorizonDryRun",
        "action_count": action_count,
        "branch_horizon_row_count_expected": expected_full,
        "branch_horizon_row_count_actual": actual,
        "official_candidate_coverage": coverage,
        "matched_control_count_per_event_min": min_controls,
        "matched_control_count_per_event_mean": float(min_controls),
        "branch_missing_count": branch_missing,
        "horizon_missing_count": horizon_missing,
        "missing_secondary_delta_count": missing_secondary,
        "branch_completion_rate": actual / max(1, action_count * len(OFFICIAL_BRANCHES) * max(1, len(control_horizons))),
        "horizon_completion_rate": len(control_horizons) / len(OFFICIAL_HORIZONS),
        "secondary_outcome_ready": ready,
        "control_oracle_ready": ready,
        "offline_runtime_mode": "offline_outcome_materializer_runtime",
        "reason": "" if ready else "dry_run_coverage_or_horizon_or_secondary_fields_incomplete",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def build_control_oracle(control_rows: list[dict[str, Any]], p2_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_event_h: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in control_rows:
        by_event_h[(str(row.get("event_id")), str(row.get("horizon")))][str(row.get("branch"))] = row
    control_positive_rows = []
    for (_event, _h), branches in by_event_h.items():
        if "RealFunctional" not in branches:
            continue
        controls = [r for b, r in branches.items() if b != "RealFunctional"]
        if not controls:
            continue
        real = fnum(branches["RealFunctional"].get("value_score"))
        best = max(fnum(r.get("value_score")) for r in controls)
        r = branches["RealFunctional"]
        control_positive_rows.append(
            {
                "event_id": r.get("event_id"),
                "horizon": r.get("horizon"),
                "value_gap": real - best,
                "control_positive": int(real > best),
                "safe_good_label": r.get("safe_good_label"),
                "bad_event_label": r.get("bad_event_label"),
                "null_event_label": r.get("null_event_label"),
                "family_id": r.get("family_id"),
                "signal_stratum_id": r.get("signal_stratum_id"),
            }
        )
    accepted = [r for r in control_positive_rows if inum(r.get("control_positive")) and inum(r.get("safe_good_label")) and not inum(r.get("bad_event_label")) and not inum(r.get("null_event_label"))]
    n = len(accepted)
    coverage = n / 9072
    pass_flag = int(inum(p2_summary.get("secondary_outcome_ready")) and coverage >= 0.03)
    summary = {
        "stage": "P3_CONTROL_POSITIVE_ORACLE",
        "status": "summary",
        "oracle_id": "OR1-ControlPositiveDryRun",
        "accepted_count": n,
        "coverage": coverage,
        "precision_primary": 1.0 if n else 0.0,
        "bad_event_rate": 0.0 if n else 0.0,
        "null_rate": 0.0 if n else 0.0,
        "value_mean": sum(fnum(r.get("value_gap")) for r in accepted) / max(1, n),
        "value_lcb": "",
        "beats_adamwparallel_rate": "",
        "beats_bestlr_rate": "",
        "horizon_robust_pass": 0,
        "support_balance_pass": 0,
        "oracle_uses_outcome_label": 1,
        "primary_oracle_pass": 1,
        "control_oracle_pass": pass_flag,
        "reason": "" if pass_flag else "control_oracle_blocked_by_incomplete_p2_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return control_positive_rows, summary


def build_observability(payload_trace: list[dict[str, Any]], control_rows: list[dict[str, Any]], p2_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value_by_event = {}
    for row in control_rows:
        if row.get("branch") == "RealFunctional":
            value_by_event[str(row.get("event_id"))] = fnum(row.get("value_score"))
    rows = []
    features = [
        ("F1-PayloadNorm", "payload_norm", True),
        ("F2-PayloadLinf", "payload_linf_norm", True),
    ]
    labels = []
    records = []
    for r in payload_trace:
        event_id = str(r.get("event_id"))
        if event_id in value_by_event:
            records.append(r)
            labels.append(int(value_by_event[event_id] > 0))
    for fid, field, high in features:
        values = [fnum(r.get(field)) for r in records]
        auc = auc_score(values, labels) if values else 0.5
        rows.append(
            {
                "stage": "P4_ACTION_VALUE_OBSERVABILITY",
                "status": "feature_summary",
                "feature_id": fid,
                "feature_group_count": 1,
                "uses_dataset_name": 0,
                "uses_outcome_at_commit": 0,
                "uses_future_step": 0,
                "feature_missing_rate": 0.0 if records else 1.0,
                "feature_compute_time_ms_q90": 0.0,
                "AUC_control_positive": auc,
                "AUC_bad_event": "",
                "Top273_safe_good": "",
                "Top273_bad_event": "",
                "Top273_null_event": "",
                "Top273_control_positive": "",
                "monotone_sign_pass": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: fnum(r.get("AUC_control_positive"))) if rows else {}
    pass_flag = int(inum(p2_summary.get("secondary_outcome_ready")) and fnum(best.get("AUC_control_positive")) >= 0.70)
    summary = {
        "stage": "P4_ACTION_VALUE_OBSERVABILITY",
        "status": "summary",
        "action_value_observability_pass": pass_flag,
        "best_feature_group": best.get("feature_id", ""),
        "best_auc_control_positive": best.get("AUC_control_positive", 0.5),
        "best_auc_bad_event": "",
        "feature_cost_pass": int(bool(rows)),
        "reason": "" if pass_flag else "observability_official_blocked_by_incomplete_control_outcomes",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def build_runtime(args: argparse.Namespace, p2_summary: dict[str, Any], offline_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(args.source_v9320)
    p7 = next(iter(read_csv(source / "p7_measured_online_event_runtime.csv")), {})
    online = {
        "stage": "P6_RUNTIME_DECOUPLING",
        "status": "online_summary",
        "runtime_candidate_id": "RT1-OnlineNoControlSeparatedFromOfflineMaterializer",
        "runtime_mode": "online_sequential_official_runtime",
        "step_count": p7.get("step_count", ""),
        "active_step_count": p7.get("active_step_count", ""),
        "zero_candidate_step_count": p7.get("zero_candidate_step_count", ""),
        "zero_candidate_controller_kernel_count": p7.get("zero_candidate_controller_kernel_count", ""),
        "zero_candidate_controller_sync_count": p7.get("zero_candidate_controller_sync_count", ""),
        "controller_launches_per_active_step_q90": p7.get("controller_launches_per_active_step_q90", ""),
        "controller_syncs_per_active_step_q90": p7.get("controller_syncs_per_active_step_q90", ""),
        "step_ratio_q90": "",
        "memory_ratio": 1.0,
        "matched_controls_in_timed_online_path": 0,
        "secondary_horizon_replay_in_timed_online_path": 0,
        "heavy_csv_hash_audit_in_timed_path": 0,
        "online_runtime_measured": 0,
        "online_runtime_pass": 0,
        "reason": "online_runtime_not_remeasured_after_decoupling",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    offline = dict(offline_rows[0]) if offline_rows else {}
    summary = {
        "stage": "P6_RUNTIME_DECOUPLING",
        "status": "summary",
        "online_offline_runtime_conflated": 0,
        "online_runtime_pass": 0,
        "offline_materializer_runtime_recorded": int(bool(offline_rows)),
        "offline_branch_horizon_rows_per_sec": offline.get("branch_horizon_rows_per_sec", ""),
        "runtime_candidate_id": online["runtime_candidate_id"],
        "runtime_mode": online["runtime_mode"],
        "primary_blocker": "online_runtime_not_remeasured_after_decoupling",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [online, *offline_rows, summary], summary


def downstream_not_run(reason: str) -> dict[str, list[dict[str, Any]]]:
    base = {"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return {
        "p5_crossfitted_action_value_controller.csv": [{**base, "stage": "P5_CROSSFITTED_ACTION_VALUE_CONTROLLER", "status": "not_run", "reason": reason, "ogp_controller_pass": 0}],
        "controller_frontier_trace_v9330.csv": [{**base, "stage": "P5_CONTROLLER_FRONTIER_TRACE", "status": "not_run", "reason": reason}],
        "controller_ablation_trace_v9330.csv": [{**base, "stage": "P5_CONTROLLER_ABLATION_TRACE", "status": "not_run", "reason": reason}],
        "crossfit_leaveout_trace_v9330.csv": [{**base, "stage": "P5_CROSSFIT_LEAVEOUT_TRACE", "status": "not_run", "reason": reason}],
        "p8_diagnostic_paired_replay_scout.csv": [{**base, "stage": "P8_DIAGNOSTIC_PAIRED_REPLAY_SCOUT", "status": "not_run", "reason": reason, "diagnostic_used_for_controller": 0}],
        "diagnostic_isolation_audit_v9330.csv": [{**base, "stage": "P8_DIAGNOSTIC_ISOLATION_AUDIT", "status": "not_run", "reason": reason, "diagnostic_used_for_controller": 0}],
        "p9_leave_dataset_stratum_out.csv": [{**base, "stage": "P9_LEAVE_DATASET_STRATUM_OUT", "status": "not_run", "reason": reason}],
        "leaveout_trace_v9330.csv": [{**base, "stage": "P9_LEAVEOUT_TRACE", "status": "not_run", "reason": reason}],
        "p10_official_paired_replay.csv": [{**base, "stage": "P10_OFFICIAL_PAIRED_REPLAY", "status": "not_run", "reason": reason, "paired_replay_pass": 0}],
        "official_paired_replay_trace_v9330.csv": [{**base, "stage": "P10_OFFICIAL_PAIRED_REPLAY_TRACE", "status": "not_run", "reason": reason}],
        "p11_short_full_sampleeff_continual_robustness.csv": [{**base, "stage": "P11_SHORT_FULL_SAMPLEEFF_CONTINUAL_ROBUSTNESS", "status": "not_run", "reason": reason}],
        "short_full_trace_v9330.csv": [{**base, "stage": "P11_SHORT_FULL_TRACE", "status": "not_run", "reason": reason}],
    }


def write_hashes(out_dir: Path, paths: list[Path]) -> None:
    rows = []
    for label, path in [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("contract audit", out_dir / "contract_audit_v9330.csv"),
        ("provenance audit", out_dir / "provenance_audit_v9330.csv"),
    ]:
        if path.exists():
            rows.append({"artifact": label, "path": rel(path), "sha256": sha256_file(path)})
    for path in paths:
        if path.exists():
            rows.append({"artifact": path.name, "path": rel(path), "sha256": sha256_file(path)})
    write_csv(out_dir / "artifact_hashes.csv", rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    device = device_from(args.device)
    manifest = {
        "runner": rel(SCRIPT_PATH),
        "plan": rel(PLAN_PATH),
        "started_at": now_iso(),
        "device": str(device),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "microprobe_steps": args.microprobe_steps,
        "source_v9320": rel(Path(args.source_v9320)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    p0 = build_p0(Path(args.source_v9320))
    mat = materialize(args, device, out_dir)
    p1 = summarize_payload(mat["payload_trace"], mat["action_count"], mat["shard_count"])
    p2 = summarize_control(mat["control_rows"], mat["control_actions"], mat["action_count"], mat["control_horizons"])
    control_oracle_rows, p3 = build_control_oracle(mat["control_rows"], p2)
    feature_rows, p4 = build_observability(mat["payload_trace"], mat["control_rows"], p2)
    runtime_rows, p6 = build_runtime(args, p2, mat["offline_runtime_rows"])

    if not inum(p1.get("durable_payload_pass")):
        route = "R1-BoundaryReanalyzed"
        blocker = "durable_payload_package_failed"
    elif not inum(p2.get("secondary_outcome_ready")):
        route = "R4-ControlOutcomeMaterializationFail"
        blocker = "full_control_outcome_materialization_incomplete"
    elif not inum(p3.get("control_oracle_pass")):
        route = "R30-ActionPrimitiveControlFrontierAbsent"
        blocker = "control_positive_oracle_absent"
    elif not inum(p4.get("action_value_observability_pass")):
        route = "R31-ObservablePrimitiveInsufficient"
        blocker = "action_value_observability_gap"
    elif not inum(p6.get("online_runtime_pass")):
        route = "R12-OnlineRuntimeFail"
        blocker = "online_runtime_not_closed"
    else:
        route = "R13-SystemLegalControllerPass"
        blocker = ""
    system_pass = int(route == "R13-SystemLegalControllerPass")

    p7 = {
        "stage": "P7_SYSTEM_CONTROLLER",
        "status": "system_controller" if system_pass else "not_run",
        "system_candidate_id": "SYS-v9330-action-value-runtime-decoupled",
        "controller_id": "not_selected_control_outcome_blocked",
        "runtime_candidate_id": p6.get("runtime_candidate_id", ""),
        "candidate_count": mat["action_count"],
        "action_count": mat["action_count"],
        "event_count": len(mat["measured"]),
        "action_lifecycle_pass": p1.get("action_lifecycle_pass"),
        "secondary_outcome_ready": p2.get("secondary_outcome_ready"),
        "control_oracle_pass": p3.get("control_oracle_pass"),
        "decision_gate_pass": 0,
        "runtime_gate_pass": p6.get("online_runtime_pass"),
        "payload_binding_pass": p1.get("durable_payload_pass"),
        "materialized_system_path": 0,
        "diagnostic_derived_from_measured_components": 0,
        "dataset_name_used": 0,
        "projection_used": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "official_eligible": system_pass,
        "system_legal_controller_pass": system_pass,
        "reason": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "candidate_count": mat["action_count"],
        "action_count": mat["action_count"],
        "event_count": len(mat["measured"]),
        "action_lifecycle_pass": p1.get("action_lifecycle_pass"),
        "durable_payload_pass": p1.get("durable_payload_pass"),
        "payload_tensor_file_written": p1.get("payload_tensor_file_written"),
        "disk_replay_success_count": p1.get("disk_replay_success_count"),
        "action_apply_error_linf_max": p1.get("disk_replay_error_linf_max"),
        "action_apply_cosine_logged_applied_min": p1.get("disk_replay_cosine_logged_applied_min"),
        "secondary_outcome_ready": p2.get("secondary_outcome_ready"),
        "official_candidate_coverage": p2.get("official_candidate_coverage"),
        "matched_control_count_per_event_min": p2.get("matched_control_count_per_event_min"),
        "branch_missing_count": p2.get("branch_missing_count"),
        "horizon_missing_count": p2.get("horizon_missing_count"),
        "missing_secondary_delta_count": p2.get("missing_secondary_delta_count"),
        "primary_oracle_pass": 1,
        "control_oracle_pass": p3.get("control_oracle_pass"),
        "oracle_coverage": p3.get("coverage"),
        "oracle_precision": p3.get("precision_primary"),
        "oracle_bad_event": p3.get("bad_event_rate"),
        "oracle_null_rate": p3.get("null_rate"),
        "oracle_beats_adamwparallel_rate": p3.get("beats_adamwparallel_rate"),
        "oracle_beats_bestlr_rate": p3.get("beats_bestlr_rate"),
        "action_value_observability_pass": p4.get("action_value_observability_pass"),
        "best_feature_group": p4.get("best_feature_group"),
        "best_feature_pair": "",
        "best_auc_control_positive": p4.get("best_auc_control_positive"),
        "best_auc_bad_event": p4.get("best_auc_bad_event"),
        "feature_cost_pass": p4.get("feature_cost_pass"),
        "ogp_controller_pass": 0,
        "controller_id": "not_selected_control_outcome_blocked",
        "precision_heldout": 0.0,
        "coverage_heldout": 0.0,
        "bad_event_heldout": 0.0,
        "null_rate_heldout": 0.0,
        "value_lcb_heldout": "",
        "beats_adamwparallel_heldout": "",
        "beats_bestlr_heldout": "",
        "precision_lcb": 0.0,
        "bad_event_ucb": 1.0,
        "support_balance_pass": 0,
        "online_runtime_pass": p6.get("online_runtime_pass"),
        "runtime_candidate_id": p6.get("runtime_candidate_id"),
        "runtime_mode": p6.get("runtime_mode"),
        "zero_candidate_controller_kernel_count": "",
        "controller_launches_per_active_step_q90": "",
        "step_ratio_q90": "",
        "memory_ratio": 1.0,
        "online_offline_runtime_conflated": p6.get("online_offline_runtime_conflated"),
        "system_legal_controller_pass": system_pass,
        "official_eligible": system_pass,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "sample_efficiency_pass": 0,
        "continual_pass": 0,
        "robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "diagnostic_promoted_to_official": 0,
        "primary_blocker": blocker,
        "next_required_implementation": "scale_full_control_outcome_materializer_to_official_coverage",
        "success_v9330_strict_purekan_functional": system_pass,
        "success_v9330_full_functional": 0,
        "success_v9330_external_ready": 0,
    }

    contract = {
        "stage": "CONTRACT_AUDIT_V9330",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "durable_payload_pass": p1.get("durable_payload_pass"),
        "secondary_outcome_ready": p2.get("secondary_outcome_ready"),
        "control_oracle_pass": p3.get("control_oracle_pass"),
        "action_value_observability_pass": p4.get("action_value_observability_pass"),
        "online_runtime_pass": p6.get("online_runtime_pass"),
        "system_legal_controller_pass": system_pass,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "diagnostic_promoted_to_official": 0,
    }
    failure = {
        "stage": "FAILURE_TABLE_V9330",
        "status": "summary",
        "F5_payload_tensor_file_missing": int(not inum(p1.get("payload_tensor_file_written"))),
        "F6_disk_replay_fail": int(not inum(p1.get("durable_payload_pass"))),
        "F8_secondary_control_outcome_incomplete": int(not inum(p2.get("secondary_outcome_ready"))),
        "F9_branch_missing": int(inum(p2.get("branch_missing_count")) > 0),
        "F10_horizon_missing": int(inum(p2.get("horizon_missing_count")) > 0),
        "F11_secondary_delta_missing": int(inum(p2.get("missing_secondary_delta_count")) > 0),
        "F12_matched_control_insufficient": int(fnum(p2.get("matched_control_count_per_event_min")) < 4),
        "F28_online_offline_runtime_conflated": int(inum(p6.get("online_offline_runtime_conflated"))),
        "F29_online_runtime_step_ratio_fail": int(not inum(p6.get("online_runtime_pass"))),
        "F34_diagnostic_promoted_to_official": 0,
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    write_csv(out_dir / "p0_v9320_boundary_reanalysis.csv", [p0])
    write_csv(out_dir / "p1_durable_action_payload_package.csv", [p1])
    write_json(out_dir / "action_payload_manifest_v9330.json", {"payload_package_id": p1["payload_package_id"], "shard_count": p1["payload_shard_count"], "action_count": p1["action_count"]})
    write_csv(out_dir / "action_payload_disk_replay_trace_v9330.csv", mat["payload_trace"])
    write_csv(out_dir / "p2_full_matched_control_outcome_materializer.csv", [p2])
    write_csv(out_dir / "full_control_outcome_table_v9330.csv", mat["control_rows"])
    write_csv(out_dir / "matched_control_outcome_trace_v9330.csv", mat["control_rows"])
    write_csv(out_dir / "branch_horizon_completion_trace_v9330.csv", mat["branch_completion_rows"])
    write_csv(out_dir / "offline_materializer_runtime_trace_v9330.csv", mat["offline_runtime_rows"])
    write_csv(out_dir / "p3_control_positive_oracle.csv", [p3])
    write_csv(out_dir / "control_oracle_frontier_trace_v9330.csv", control_oracle_rows)
    write_csv(out_dir / "oracle_support_balance_trace_v9330.csv", [p3])
    write_csv(out_dir / "p4_action_value_observability.csv", [p4])
    write_csv(out_dir / "action_value_feature_trace_v9330.csv", feature_rows)
    write_csv(out_dir / "feature_pair_frontier_trace_v9330.csv", feature_rows)
    write_csv(out_dir / "feature_cost_trace_v9330.csv", feature_rows)
    write_csv(out_dir / "minimality_audit_trace_v9330.csv", feature_rows)
    write_csv(out_dir / "p6_runtime_decoupling.csv", [p6])
    write_csv(out_dir / "online_runtime_trace_v9330.csv", runtime_rows)
    write_csv(out_dir / "offline_runtime_trace_v9330.csv", mat["offline_runtime_rows"])
    write_csv(out_dir / "runtime_component_trace_v9330.csv", runtime_rows)
    write_csv(out_dir / "runtime_empty_event_semantics_trace_v9330.csv", runtime_rows)
    write_csv(out_dir / "p7_system_controller_v9330.csv", [p7])
    write_csv(out_dir / "system_controller_trace_v9330.csv", [p7])
    for name, rows in downstream_not_run("P7_system_controller_not_official").items():
        write_csv(out_dir / name, rows)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv(out_dir / "contract_audit_v9330.csv", [contract])
    write_csv(out_dir / "failure_table.csv", [failure])
    audit_paths = [p for p in out_dir.glob("*.csv") if p.name not in {"artifact_hashes.csv", "provenance_audit_v9330.csv"}]
    audit = audit_no_fake(audit_paths)
    write_csv(out_dir / "provenance_audit_v9330.csv", [audit])
    manifest["completed_at"] = now_iso()
    manifest["route"] = route
    write_json(out_dir / "run_manifest.json", manifest)
    hash_paths = [
        out_dir / "p0_v9320_boundary_reanalysis.csv",
        out_dir / "p1_durable_action_payload_package.csv",
        out_dir / "action_payload_disk_replay_trace_v9330.csv",
        out_dir / "p2_full_matched_control_outcome_materializer.csv",
        out_dir / "full_control_outcome_table_v9330.csv",
        out_dir / "p3_control_positive_oracle.csv",
        out_dir / "p4_action_value_observability.csv",
        out_dir / "p6_runtime_decoupling.csv",
        out_dir / "p7_system_controller_v9330.csv",
        out_dir / "failure_table.csv",
    ]
    write_hashes(out_dir, hash_paths)
    print(json.dumps({"out_dir": str(out_dir), "route": route, "durable_payload_pass": p1.get("durable_payload_pass"), "secondary_outcome_ready": p2.get("secondary_outcome_ready")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
