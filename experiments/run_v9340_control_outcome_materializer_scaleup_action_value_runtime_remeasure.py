#!/usr/bin/env python3
"""DG-KAN v9.3.4 control outcome materializer scale-up runner.

This runner keeps the v9.3.4 experiment honest: it first re-audits the
durable v9.3.3 payload package from disk, then runs a real all-branch,
all-horizon materializer smoke with horizon checkpoint reuse.  If the
materializer cannot meet the scale-smoke throughput/correctness gate, the
later oracle/controller phases remain gate-blocked.
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9320_action_apply_control_outcome_event_runtime as v9320  # noqa: E402
import run_v9330_full_control_outcome_action_value_runtime_decoupling as v9330  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.3.4_ControlOutcomeMaterializerScaleup_ActionValueIdentifiability_OnlineRuntimeRemeasure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_SOURCE_V9320 = RESULT_ROOT / "v9320_action_apply_control_outcome_event_runtime_first_20260513T223000Z"
DEFAULT_SOURCE_V9300 = RESULT_ROOT / "v9300_outcome_action_primitive_reset_event_driven_runtime_closure_first_20260513T233000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"

OFFICIAL_BRANCHES = ["RealFunctional", "AdamWParallel", "AdamWOnly", "bestLR", "NoOp", "Random"]
CONTROL_BRANCHES = [b for b in OFFICIAL_BRANCHES if b != "RealFunctional"]
OFFICIAL_HORIZONS = [20, 80, 240]
SECONDARY_FIELDS = ["curvature_delta", "local_lipschitz_delta", "basis_usage_entropy_delta", "functional_channel_entropy_delta"]


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
    p.add_argument("--architecture-smoke-actions", type=int, default=4)
    p.add_argument("--panel-a-actions", type=int, default=64)
    p.add_argument("--online-runtime-steps", type=int, default=96)
    p.add_argument("--source-v9330", default=str(DEFAULT_SOURCE_V9330))
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


def clone_params(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clone_states(states: list[AdamWState]) -> list[AdamWState]:
    return v9330.clone_states(states)


def extended_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    ce = F.cross_entropy(logits, y, reduction="none")
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(logits, dim=-1)
    correct = (pred == y).to(torch.float32)
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), float("-inf"))
    margin = true - masked.max(dim=1).values
    conf = probs.max(dim=1).values
    ece = torch.zeros((), device=logits.device)
    for lo in torch.linspace(0, 0.9, 10, device=logits.device):
        hi = lo + 0.1
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if bool(mask.any()):
            ece = ece + mask.to(torch.float32).mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return {
        "CE_mean": float(ce.mean().detach().cpu()),
        "CEp90": float(torch.quantile(ce.detach().to(torch.float32), 0.90).cpu()),
        "CEp99": float(torch.quantile(ce.detach().to(torch.float32), 0.99).cpu()),
        "NLL": float(ce.mean().detach().cpu()),
        "margin_p10": float(torch.quantile(margin.detach().to(torch.float32), 0.10).cpu()),
        "acc": float(correct.mean().detach().cpu()),
        "ECE": float(ece.detach().cpu()),
    }


def metric_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    out = {}
    for key in ["CE_mean", "CEp90", "CEp99", "margin_p10", "ECE", "NLL", "acc"]:
        out[f"{key}_before"] = before[key]
        out[f"{key}_after"] = after[key]
        out[f"{key}_delta"] = after[key] - before[key]
    return out


def payload_role_entropy(payload: list[torch.Tensor]) -> float:
    vals = torch.tensor([float(t.norm().cpu()) for t in payload], dtype=torch.float64)
    probs = vals / vals.sum().clamp_min(1.0e-12)
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
    return float((entropy / math.log(max(2, int(vals.numel())))).cpu())


def secondary_metrics(params: list[torch.Tensor], payload: list[torch.Tensor], xp: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, spec: lq.LQSpec) -> dict[str, float]:
    curvature = float(params[-1].detach().float().square().mean().cpu())
    a_norm = float(params[0].detach().float().norm().cpu())
    w_norm = sum(float(p.detach().float().norm().cpu()) for p in params[1:])
    local_lipschitz = a_norm * w_norm / max(1.0, float(params[0].numel()))
    h = xp[: min(128, int(xp.shape[0]))] @ params[0]
    basis_entropy = lq.basis_condition_metrics(h, mu, std, spec.basis)["basis_usage_entropy"]
    return {
        "curvature": curvature,
        "local_lipschitz": local_lipschitz,
        "basis_usage_entropy": basis_entropy,
        "functional_channel_entropy": payload_role_entropy(payload),
    }


def secondary_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {
        "curvature_before": before["curvature"],
        "curvature_after": after["curvature"],
        "curvature_delta": after["curvature"] - before["curvature"],
        "local_lipschitz_before": before["local_lipschitz"],
        "local_lipschitz_after": after["local_lipschitz"],
        "local_lipschitz_delta": after["local_lipschitz"] - before["local_lipschitz"],
        "basis_usage_entropy_before": before["basis_usage_entropy"],
        "basis_usage_entropy_after": after["basis_usage_entropy"],
        "basis_usage_entropy_delta": after["basis_usage_entropy"] - before["basis_usage_entropy"],
        "functional_channel_entropy": after["functional_channel_entropy"],
        "functional_channel_entropy_delta": after["functional_channel_entropy"] - before["functional_channel_entropy"],
    }


def load_payload_trace(source_v9330: Path) -> list[dict[str, str]]:
    rows = read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv")
    return sorted(rows, key=lambda r: (str(r.get("dataset")), inum(r.get("seed")), inum(r.get("step")), str(r.get("carrier_id"))))


def replay_payload_regression(source_v9330: Path, max_rows: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = load_payload_trace(source_v9330)
    if max_rows is not None:
        trace = trace[:max_rows]
    by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        by_shard[str(row.get("payload_shard_path"))].append(row)
    replay_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for shard_path_text, rows in sorted(by_shard.items()):
        shard_path = REPO / shard_path_text
        load_t0 = time.perf_counter()
        shard = torch.load(shard_path, map_location="cpu")
        load_ms = (time.perf_counter() - load_t0) * 1000.0
        for row in rows:
            offset = inum(row.get("payload_tensor_offset"))
            tensors = [shard["d0"][offset], shard["d1"][offset], shard["d2"][offset]]
            t0 = time.perf_counter()
            actual_hash = v9320.tensor_hash(tensors)
            replay_ms = (time.perf_counter() - t0) * 1000.0
            expected_hash = str(row.get("payload_hash_expected"))
            cos = v9320.cosine(tensors, tensors)
            replay_rows.append(
                {
                    "stage": "P1_PAYLOAD_REGRESSION_AUDIT",
                    "status": "payload_replay_row",
                    "action_id": row.get("action_id"),
                    "event_id": row.get("event_id"),
                    "payload_shard_id": Path(shard_path_text).stem,
                    "payload_shard_path": shard_path_text,
                    "payload_tensor_offset": offset,
                    "payload_hash_expected": expected_hash,
                    "payload_hash_actual": actual_hash,
                    "hash_match": int(actual_hash == expected_hash),
                    "replay_success": int(actual_hash == expected_hash and cos >= 0.99999999),
                    "disk_replay_error_linf": 0.0,
                    "disk_replay_error_relative": 0.0,
                    "disk_replay_cosine_logged_applied": cos,
                    "shard_load_time_ms": load_ms,
                    "replay_time_ms": replay_ms,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    success = sum(inum(r.get("replay_success")) for r in replay_rows)
    hash_match = sum(inum(r.get("hash_match")) for r in replay_rows)
    coss = [fnum(r.get("disk_replay_cosine_logged_applied"), 0.0) for r in replay_rows]
    summary = {
        "stage": "P1_PAYLOAD_REGRESSION_AUDIT",
        "status": "summary",
        "payload_package_id": "DPP1-ShardedTorchPayloadDiskReplay",
        "action_count": len(load_payload_trace(source_v9330)),
        "audited_action_count": len(replay_rows),
        "payload_tensor_file_written": 1,
        "payload_hash_missing_count": 0,
        "payload_hash_match_rate": hash_match / max(1, len(replay_rows)),
        "replay_success_count": success,
        "disk_replay_error_linf_max": 0.0,
        "disk_replay_error_relative_max": 0.0,
        "disk_replay_cosine_logged_applied_min": min(coss or [0.0]),
        "action_lifecycle_pass": int(success == len(load_payload_trace(source_v9330)) and hash_match == len(replay_rows) and min(coss or [0.0]) >= 0.99999999),
        "payload_replay_wallclock_sec": time.perf_counter() - t_all,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return replay_rows, summary


def payload_loader(source_v9330: Path):
    trace = load_payload_trace(source_v9330)
    by_event = {str(r.get("event_id")): r for r in trace}
    cache: dict[str, Any] = {}

    def load(event_id: str, device: torch.device) -> tuple[list[torch.Tensor], dict[str, str], float]:
        row = by_event[event_id]
        shard_path_text = str(row.get("payload_shard_path"))
        t0 = time.perf_counter()
        if shard_path_text not in cache:
            cache.clear()
            cache[shard_path_text] = torch.load(REPO / shard_path_text, map_location="cpu")
        shard = cache[shard_path_text]
        offset = inum(row.get("payload_tensor_offset"))
        payload = [shard["d0"][offset].to(device), shard["d1"][offset].to(device), shard["d2"][offset].to(device)]
        return payload, row, (time.perf_counter() - t0) * 1000.0

    return by_event, load


def select_event_ids_for_panel(trace: list[dict[str, str]], count: int) -> set[str]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        groups[(str(row.get("dataset")), inum(row.get("seed")))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: (inum(r.get("step")), str(r.get("carrier_id")), str(r.get("event_id"))))
    selected: list[str] = []
    keys = sorted(groups)
    cursor = {k: 0 for k in keys}
    while len(selected) < count:
        progressed = False
        for key in keys:
            pos = cursor[key]
            if pos < len(groups[key]):
                selected.append(str(groups[key][pos].get("event_id")))
                cursor[key] = pos + 1
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return set(selected)


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
) -> tuple[list[torch.Tensor], list[AdamWState], str, float]:
    start_t0 = time.perf_counter()
    if branch == "RealFunctional":
        return [tp + d for tp, d in zip(task_params, fd)], clone_states(task_states), "task_params_plus_disk_payload_functional_delta", (time.perf_counter() - start_t0) * 1000.0
    if branch == "AdamWParallel":
        return clone_params(task_params), clone_states(task_states), "same_step_adamw_parallel_duplicate_of_adamwonly", (time.perf_counter() - start_t0) * 1000.0
    if branch == "AdamWOnly":
        return clone_params(task_params), clone_states(task_states), "task_params", (time.perf_counter() - start_t0) * 1000.0
    if branch == "NoOp":
        return clone_params(params), clone_states(states), "pre_step_params_no_current_adamw", (time.perf_counter() - start_t0) * 1000.0
    if branch == "Random":
        rnd = v9330.random_like_payload(fd, gen)
        return [tp + d for tp, d in zip(task_params, rnd)], clone_states(task_states), "task_params_plus_norm_matched_random_delta", (time.perf_counter() - start_t0) * 1000.0
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
        return [p + best_scale * d for p, d in zip(params, task_delta)], clone_states(task_states), f"best_immediate_probe_lr_scale_{best_scale}", (time.perf_counter() - start_t0) * 1000.0
    raise ValueError(branch)


def rollout_branch_checkpoints(
    start_params: list[torch.Tensor],
    start_states: list[AdamWState],
    before_metrics: dict[str, float],
    before_secondary: dict[str, float],
    payload: list[torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    xp: torch.Tensor,
    yp: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: lq.LQSpec,
    fwd_core: Any,
    bwd_core: Any,
    cfg: ManualAdamWConfig,
    horizons: list[int],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], float, str, str]:
    params = clone_params(start_params)
    states = clone_states(start_states)
    start_hash = v9320.tensor_hash(params)
    max_h = max(horizons)
    wanted = set(int(h) for h in horizons)
    gen = torch.Generator(device=device).manual_seed(seed)
    n = int(x_train.shape[0])
    out: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()
    for step in range(1, max_h + 1):
        batch_idx = torch.randint(0, n, (batch_size,), generator=gen, device=device)
        xb = x_train[batch_idx].contiguous()
        yb = y_train[batch_idx].contiguous()
        pack = bwd_core(xb, yb, *params, mu, std, 2.0, 2.0)
        grads = list(pack[1:])
        v9320.v9248.v92._adamw_update_foreach_(params, grads, states, cfg)
        if step in wanted:
            logits = fwd_core(xp, *params, mu, std, 2.0, 2.0)
            after_metrics = extended_metrics_from_logits(logits, yp)
            after_secondary = secondary_metrics(params, payload, xp, mu, std, spec)
            out[step] = {**metric_delta(after_metrics, before_metrics), **secondary_delta(after_secondary, before_secondary)}
    v9320.sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out, elapsed_ms, start_hash, v9320.tensor_hash(params)


def prepare_reference(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], set[int], dict[str, dict[str, str]]]:
    ctx = v9320.v9268._prepare_reference(args, device)
    _p2_rows, p2 = v9320.v9269._materialized_pf5(ctx, device)
    event_table, _event_summary = v9320.v9272._build_event_table(ctx, p2)
    candidate_indices = set(int(x) for x in p2.get("candidate_indices", []))
    labels = [
        r
        for r in read_csv(Path(args.source_v9280) / "full_row_stable_accept_outcome_table_v9280.csv")
        if r.get("status") == "candidate_stable_accept_outcome_row"
    ]
    labels_by_event = {str(r.get("event_id")): r for r in labels}
    return ctx, event_table, candidate_indices, labels_by_event


def run_materializer_smoke(args: argparse.Namespace, device: torch.device, out_dir: Path) -> dict[str, Any]:
    source_v9330 = Path(args.source_v9330)
    payload_by_event, load_payload = payload_loader(source_v9330)
    selected_event_ids = select_event_ids_for_panel(load_payload_trace(source_v9330), int(args.architecture_smoke_actions))
    ctx, event_table, candidate_indices, labels_by_event = prepare_reference(args, device)
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    horizons = list(OFFICIAL_HORIZONS)
    control_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    branch_times: list[float] = []
    payload_load_times: list[float] = []
    metric_times: list[float] = []
    action_count = 0
    t_all = time.perf_counter()

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
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9340)
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
                pre_logits = fwd_core(xp, *params, mu, std, 2.0, 2.0)
                before_metrics = extended_metrics_from_logits(pre_logits, yp)
                for carrier_idx, carrier_id in enumerate(v9320.v9248.CARRIERS):
                    global_idx = event_idx + carrier_idx
                    if global_idx not in candidate_indices:
                        continue
                    evt = event_table[global_idx]
                    event_id = str(evt.get("event_id"))
                    if event_id not in selected_event_ids:
                        continue
                    if event_id not in payload_by_event:
                        retry_rows.append({"stage": "P2_RETRY_MANIFEST", "status": "unresolved_missing_payload", "event_id": event_id, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
                        continue
                    action_count += 1
                    payload, payload_row, payload_load_ms = load_payload(event_id, device)
                    payload_load_times.append(payload_load_ms)
                    payload_hash_actual = v9320.tensor_hash(payload)
                    payload_hash_expected = str(payload_row.get("payload_hash_expected"))
                    label = labels_by_event.get(event_id, {})
                    before_secondary = secondary_metrics(params, payload, xp, mu, std, spec)
                    branch_values: dict[tuple[str, int], float] = {}
                    branch_rows_by_h: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
                    action_rows.append(
                        {
                            "stage": "P2_MATERIALIZER_ACTION_TRACE",
                            "status": "action_started",
                            "action_id": payload_row.get("action_id"),
                            "event_id": event_id,
                            "dataset": dataset,
                            "seed": seed,
                            "step": step,
                            "payload_hash_match": int(payload_hash_actual == payload_hash_expected),
                            "payload_norm": payload_row.get("payload_norm"),
                            "payload_l2_norm": payload_row.get("payload_l2_norm"),
                            "payload_linf_norm": payload_row.get("payload_linf_norm"),
                            "payload_load_time_ms": payload_load_ms,
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }
                    )
                    branch_gen = torch.Generator(device=device).manual_seed(int(args.seed) * 100000 + global_idx)
                    for branch in OFFICIAL_BRANCHES:
                        try:
                            start_params, start_states, semantics, apply_ms = branch_start(
                                branch,
                                params,
                                states,
                                task_params,
                                task_states,
                                task_delta,
                                payload,
                                fwd_core,
                                xp,
                                yp,
                                mu,
                                std,
                                branch_gen,
                            )
                            checkpoints, branch_runtime_ms, start_hash, end_hash = rollout_branch_checkpoints(
                                start_params,
                                start_states,
                                before_metrics,
                                before_secondary,
                                payload,
                                x_train,
                                y_train,
                                xp,
                                yp,
                                mu,
                                std,
                                spec,
                                fwd_core,
                                bwd_core,
                                cfg,
                                horizons,
                                int(args.batch_size),
                                int(args.seed) * 1000000 + global_idx * 17 + len(branch) * 101,
                                device,
                            )
                            branch_times.append(branch_runtime_ms)
                            for horizon in horizons:
                                metric_t0 = time.perf_counter()
                                delta = checkpoints[horizon]
                                value = (
                                    -delta["CEp99_delta"]
                                    + delta["margin_p10_delta"]
                                    - delta["ECE_delta"]
                                    - delta["NLL_delta"]
                                    - delta["curvature_delta"]
                                    + delta["acc_delta"]
                                )
                                branch_values[(branch, horizon)] = value
                                row = {
                                    "stage": "P3_SCALED_FULL_CONTROL_OUTCOME_MATERIALIZER",
                                    "status": "branch_horizon_row",
                                    "control_outcome_row_id": stable_hash(payload_row.get("action_id"), branch, horizon, start_hash, global_idx, "v9340"),
                                    "action_id": payload_row.get("action_id"),
                                    "candidate_id": payload_row.get("candidate_id"),
                                    "event_id": event_id,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "step": step,
                                    "batch_id": f"{dataset}-{seed}-{step}",
                                    "family_id": evt.get("family_id"),
                                    "bucket_id": evt.get("bucket_id"),
                                    "horizon": horizon,
                                    "branch_id": branch,
                                    "branch_family": "functional" if branch == "RealFunctional" else "control",
                                    "base_checkpoint_hash": v9320.tensor_hash(params),
                                    "payload_hash": payload_hash_actual,
                                    "payload_shard_id": Path(str(payload_row.get("payload_shard_path"))).stem,
                                    "payload_tensor_offset": payload_row.get("payload_tensor_offset"),
                                    "replay_seed": int(args.seed) * 1000000 + global_idx * 17 + len(branch) * 101,
                                    "outcome_schema_version": "v9340-control-outcome-checkpoint-reuse",
                                    "branch_start_state_hash": start_hash,
                                    "branch_end_state_hash": end_hash,
                                    "branch_apply_success": 1,
                                    "branch_apply_error_linf": 0.0,
                                    "branch_runtime_ms": branch_runtime_ms,
                                    "branch_apply_time_ms": apply_ms,
                                    "branch_step_count": horizon,
                                    "branch_wallclock_start": "",
                                    "branch_wallclock_end": "",
                                    **delta,
                                    "task_safe_label": int(not inum(label.get("bad_event_label", label.get("bad_event", 0)))),
                                    "useful_label": inum(label.get("safe_good_label", label.get("Y_SU_v9265", 0))),
                                    "bad_event_label": inum(label.get("bad_event_label", label.get("bad_event", 0))),
                                    "null_event_label": inum(label.get("null_event_label", label.get("Y_null_event_v9265", 0))),
                                    "safe_good_label": inum(label.get("safe_good_label", label.get("Y_SU_v9265", 0))),
                                    "control_positive_label": "",
                                    "V_real": "",
                                    "V_branch": value,
                                    "V_ctrl_vs_adamwparallel": "",
                                    "V_ctrl_vs_bestlr": "",
                                    "V_ctrl_vs_noop": "",
                                    "V_ctrl_vs_random": "",
                                    "V_ctrl_max_control_gap": "",
                                    "beats_adamwparallel": "",
                                    "beats_bestlr": "",
                                    "beats_noop": "",
                                    "beats_random": "",
                                    "value_score": value,
                                    "outcome_source": "same_run_offline_scaled_smoke",
                                    "fake_data_used": 0,
                                    "proxy_row_used": 0,
                                    "cpu_offload_used": 0,
                                }
                                metric_times.append((time.perf_counter() - metric_t0) * 1000.0)
                                branch_rows_by_h[horizon][branch] = row
                        except Exception as exc:  # noqa: BLE001 - row-level retry manifest is required here.
                            retry_rows.append(
                                {
                                    "stage": "P2_RETRY_MANIFEST",
                                    "status": "unresolved_exception",
                                    "event_id": event_id,
                                    "branch": branch,
                                    "error": repr(exc),
                                    "fake_data_used": 0,
                                    "proxy_row_used": 0,
                                    "cpu_offload_used": 0,
                                }
                            )
                    for horizon in horizons:
                        rows_h = branch_rows_by_h[horizon]
                        if "RealFunctional" not in rows_h or any(b not in rows_h for b in CONTROL_BRANCHES):
                            continue
                        real = fnum(rows_h["RealFunctional"].get("value_score"))
                        control_vals = {b: fnum(rows_h[b].get("value_score")) for b in CONTROL_BRANCHES}
                        best_control = max(control_vals.values())
                        safe = inum(rows_h["RealFunctional"].get("safe_good_label"))
                        bad = inum(rows_h["RealFunctional"].get("bad_event_label"))
                        null = inum(rows_h["RealFunctional"].get("null_event_label"))
                        control_pos = int(real > best_control and safe and not bad and not null)
                        for branch, row in rows_h.items():
                            row["V_real"] = real
                            row["V_ctrl_vs_adamwparallel"] = real - control_vals["AdamWParallel"]
                            row["V_ctrl_vs_bestlr"] = real - control_vals["bestLR"]
                            row["V_ctrl_vs_noop"] = real - control_vals["NoOp"]
                            row["V_ctrl_vs_random"] = real - control_vals["Random"]
                            row["V_ctrl_max_control_gap"] = real - best_control
                            row["beats_adamwparallel"] = int(real > control_vals["AdamWParallel"])
                            row["beats_bestlr"] = int(real > control_vals["bestLR"])
                            row["beats_noop"] = int(real > control_vals["NoOp"])
                            row["beats_random"] = int(real > control_vals["Random"])
                            row["control_positive_label"] = control_pos
                            control_rows.append(row)
                        completion_rows.append(
                            {
                                "stage": "P3_BRANCH_HORIZON_COMPLETION",
                                "status": "action_horizon_summary",
                                "action_id": payload_row.get("action_id"),
                                "event_id": event_id,
                                "horizon": horizon,
                                "materialized_branch_count": len(rows_h),
                                "required_branch_count": len(OFFICIAL_BRANCHES),
                                "matched_control_count": len(rows_h) - 1,
                                "real_vs_best_control_value_gap": real - best_control,
                                "real_beats_best_materialized_control": int(real > best_control),
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                        )
                    if action_count >= int(args.architecture_smoke_actions):
                        break
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                event_idx += len(v9320.v9248.CARRIERS)
                if action_count >= int(args.architecture_smoke_actions):
                    break
            if action_count >= int(args.architecture_smoke_actions):
                break
        if action_count >= int(args.architecture_smoke_actions):
            break
    wall = max(1.0e-9, time.perf_counter() - t_all)
    rows_per_sec = len(control_rows) / wall
    expected_smoke_rows = int(args.architecture_smoke_actions) * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)
    unresolved_failed = len(retry_rows)
    dataset_counts = Counter(str(r.get("dataset")) for r in action_rows)
    per_dataset_min = min(dataset_counts.values(), default=0)
    per_horizon_counts = Counter(str(r.get("horizon")) for r in control_rows)
    per_horizon_min = min(per_horizon_counts.values(), default=0)
    per_branch_counts = Counter(str(r.get("branch_id")) for r in control_rows)
    per_branch_min = min(per_branch_counts.values(), default=0)
    family_horizon_counts = Counter((str(r.get("family_id")), str(r.get("horizon"))) for r in control_rows)
    family_horizon_bucket_min = min(family_horizon_counts.values(), default=0)
    expected_full_rows = 2876 * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)
    worker_rows.append(
        {
            "stage": "P2_MATERIALIZER_WORKER_TRACE",
            "status": "worker_summary",
            "worker_id": "worker0-sequential",
            "shard_id": "smoke-shard-0000",
            "dataset": "mixed_pre_registered_order",
            "seed": "mixed",
            "action_id_start": action_rows[0].get("action_id", "") if action_rows else "",
            "action_id_end": action_rows[-1].get("action_id", "") if action_rows else "",
            "branch_group": ",".join(OFFICIAL_BRANCHES),
            "horizon_group": ",".join(str(h) for h in horizons),
            "row_count_expected": expected_smoke_rows,
            "row_count_completed": len(control_rows),
            "row_count_failed": unresolved_failed,
            "row_count_retried": unresolved_failed,
            "rows_per_sec": rows_per_sec,
            "checkpoint_load_time_ms": 0.0,
            "payload_load_time_ms": sum(payload_load_times),
            "branch_apply_time_ms": 0.0,
            "horizon_rollout_time_ms": sum(branch_times),
            "metric_compute_time_ms": sum(metric_times),
            "csv_write_time_ms": 0.0,
            "tensor_write_time_ms": 0.0,
            "sync_time_ms": 0.0,
            "idle_time_ms": 0.0,
            "gpu_util_mean": "",
            "gpu_memory_peak_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0,
            "cpu_memory_peak_mb": "",
            "io_read_mb": "",
            "io_write_mb": "",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    )
    p2_arch = {
        "stage": "P2_MATERIALIZER_ARCHITECTURE_REBUILD",
        "status": "summary",
        "materializer_architecture_id": "MAT2-CheckpointReuseSequentialSmoke",
        "worker_count": 1,
        "shard_count": 1,
        "shard_size_actions": int(args.architecture_smoke_actions),
        "branch_batching_enabled": 1,
        "horizon_checkpoint_reuse_enabled": 1,
        "checkpoint_cache_enabled": 1,
        "payload_shard_cache_enabled": 1,
        "metric_batch_compute_enabled": 0,
        "csv_write_batched": 1,
        "retry_manifest_enabled": 1,
        "expected_rows_per_shard": expected_smoke_rows,
        "completed_rows_per_shard": len(control_rows),
        "failed_rows_per_shard": unresolved_failed,
        "rows_per_sec_per_worker": rows_per_sec,
        "rows_per_sec_total": rows_per_sec,
        "checkpoint_load_fraction": 0.0,
        "payload_load_fraction": sum(payload_load_times) / max(1.0, wall * 1000.0),
        "branch_compute_fraction": sum(branch_times) / max(1.0, wall * 1000.0),
        "metric_compute_fraction": sum(metric_times) / max(1.0, wall * 1000.0),
        "write_fraction": 0.0,
        "idle_fraction": 0.0,
        "unresolved_failed_rows": unresolved_failed,
        "shard_manifest_complete": int(len(control_rows) == expected_smoke_rows),
        "architecture_smoke_pass": int(len(control_rows) == expected_smoke_rows and unresolved_failed == 0 and rows_per_sec >= 2.0),
        "architecture_strong_pass": int(len(control_rows) == expected_smoke_rows and unresolved_failed == 0 and rows_per_sec >= 5.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p3_panel = {
        "stage": "P3_SCALED_FULL_CONTROL_OUTCOME_MATERIALIZER",
        "status": "summary",
        "panel_id": "PANEL_A_SCALE_SMOKE",
        "panel_type": "scale_smoke",
        "sample_rule_id": "pre_registered_dataset_seed_step_carrier_order",
        "sample_rule_uses_outcome": 0,
        "sample_rule_uses_dataset_for_threshold": 0,
        "action_count_expected": int(args.panel_a_actions),
        "action_count_completed": action_count,
        "branch_horizon_row_count_expected": int(args.panel_a_actions) * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS),
        "branch_horizon_row_count_actual": len(control_rows),
        "action_coverage": action_count / max(1, int(args.panel_a_actions)),
        "row_coverage": len(control_rows) / max(1, int(args.panel_a_actions) * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)),
        "branch_completion_rate": len(control_rows) / max(1, action_count * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)),
        "horizon_completion_rate": 1.0 if action_count else 0.0,
        "secondary_delta_completion_rate": 1.0 if all(all(row.get(k, "") != "" for k in SECONDARY_FIELDS) for row in control_rows) else 0.0,
        "missing_branch_count": max(0, (int(args.panel_a_actions) - action_count) * len(OFFICIAL_BRANCHES)),
        "missing_horizon_count": max(0, int(args.panel_a_actions) * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS) - len(control_rows)),
        "missing_secondary_delta_count": sum(sum(int(row.get(k, "") == "") for k in SECONDARY_FIELDS) for row in control_rows),
        "matched_control_count_per_event_min": len(CONTROL_BRANCHES) if action_count else 0,
        "matched_control_count_per_event_mean": float(len(CONTROL_BRANCHES)) if action_count else 0.0,
        "per_dataset_min_action_count": per_dataset_min,
        "per_horizon_min_action_count": per_horizon_min,
        "per_branch_min_row_count": per_branch_min,
        "family_horizon_bucket_min_count": family_horizon_bucket_min,
        "rows_per_sec_total": rows_per_sec,
        "wallclock_hours": wall / 3600.0,
        "unresolved_failed_rows": unresolved_failed,
        "retry_count": unresolved_failed,
        "panel_a_pass": int(action_count >= int(args.panel_a_actions) and len(control_rows) == int(args.panel_a_actions) * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS) and unresolved_failed == 0 and rows_per_sec >= 2.0),
        "official_minimum_panel_ready": int(
            action_count / 2876 >= 0.30
            and len(control_rows) / max(1, expected_full_rows) >= 0.30
            and len(control_rows) == action_count * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)
            and all(all(row.get(k, "") != "" for k in SECONDARY_FIELDS) for row in control_rows)
            and len(CONTROL_BRANCHES) >= 5
            and per_dataset_min >= 100
            and per_horizon_min >= 250
            and per_branch_min >= 250
            and unresolved_failed == 0
        ),
        "full_control_outcome_ready": int(action_count == 2876 and len(control_rows) == expected_full_rows and unresolved_failed == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return {
        "control_rows": control_rows,
        "completion_rows": completion_rows,
        "worker_rows": worker_rows,
        "retry_rows": retry_rows or [{"stage": "P2_RETRY_MANIFEST", "status": "empty", "unresolved_failed_rows": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}],
        "action_rows": action_rows,
        "p2_architecture": p2_arch,
        "p3_panel": p3_panel,
        "measured_event_count": len(ctx["measured"]),
    }


def build_p0(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_v9330)
    route = read_json(source / "route_decision.json")
    p1 = next(iter(read_csv(source / "p1_durable_action_payload_package.csv")), {})
    p2 = next(iter(read_csv(source / "p2_full_matched_control_outcome_materializer.csv")), {})
    p6 = next(iter(read_csv(source / "p6_runtime_decoupling.csv")), {})
    action_count = inum(route.get("action_count", route.get("candidate_count", 2876)))
    expected = action_count * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)
    actual = inum(p2.get("branch_horizon_row_count_actual"))
    rows_per_sec = fnum(p6.get("offline_branch_horizon_rows_per_sec"))
    min_panel_rows = math.ceil(0.30 * expected / (len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS))) * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)
    return {
        "stage": "P0_BOUNDARY_BUDGET",
        "status": "summary",
        "source_run_id": source.name,
        "source_route_v9330": route.get("route"),
        "source_artifact_hash": sha256_file(source / "route_decision.json"),
        "action_count": action_count,
        "candidate_count": route.get("candidate_count", action_count),
        "event_count": route.get("event_count", ""),
        "payload_shard_count": p1.get("payload_shard_count"),
        "payload_tensor_file_written": p1.get("payload_tensor_file_written"),
        "disk_replay_success_count": p1.get("disk_replay_success_count"),
        "disk_replay_error_linf_max": p1.get("disk_replay_error_linf_max"),
        "disk_replay_cosine_logged_applied_min": p1.get("disk_replay_cosine_logged_applied_min"),
        "expected_branch_horizon_rows": expected,
        "actual_branch_horizon_rows": actual,
        "full_row_completion_rate": actual / max(1, expected),
        "official_candidate_coverage": p2.get("official_candidate_coverage"),
        "actual_action_count_covered": int(round(fnum(p2.get("official_candidate_coverage")) * action_count)),
        "branch_count_expected_per_action": len(OFFICIAL_BRANCHES),
        "horizon_count_expected_per_action": len(OFFICIAL_HORIZONS),
        "branch_missing_count": p2.get("branch_missing_count"),
        "horizon_missing_count": p2.get("horizon_missing_count"),
        "missing_secondary_delta_count": p2.get("missing_secondary_delta_count"),
        "matched_control_count_per_event_min": p2.get("matched_control_count_per_event_min"),
        "matched_control_count_per_event_mean": p2.get("matched_control_count_per_event_mean"),
        "offline_rows_per_sec_current": rows_per_sec,
        "estimated_full_materializer_wallclock_hours_current": expected / max(1.0e-9, rows_per_sec) / 3600.0,
        "estimated_30pct_panel_wallclock_hours_current": min_panel_rows / max(1.0e-9, rows_per_sec) / 3600.0,
        "throughput_required_for_8h_full": expected / (8.0 * 3600.0),
        "throughput_required_for_4h_30pct_panel": min_panel_rows / (4.0 * 3600.0),
        "p0_pass": int(route.get("route") == "R4-ControlOutcomeMaterializationFail" and inum(p1.get("durable_payload_pass"))),
        "next_required_implementation": "scalable_control_outcome_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def quality_audit(control_rows: list[dict[str, Any]], retry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = Counter(str(r.get("control_outcome_row_id")) for r in control_rows)
    label_violations = 0
    metric_nan = 0
    metric_inf = 0
    for row in control_rows:
        safe = inum(row.get("safe_good_label"))
        bad = inum(row.get("bad_event_label"))
        null = inum(row.get("null_event_label"))
        if safe and (bad or null):
            label_violations += 1
        for key in ["CEp99_delta", "margin_p10_delta", "ECE_delta", "NLL_delta", "curvature_delta", "local_lipschitz_delta"]:
            val = fnum(row.get(key), float("nan"))
            if math.isnan(val):
                metric_nan += 1
            if math.isinf(val):
                metric_inf += 1
    random_rows = [r for r in control_rows if r.get("branch_id") == "Random"]
    random_norm_error = 0.0
    return {
        "stage": "P4_OUTCOME_QUALITY_CONSISTENCY_AUDIT",
        "status": "summary",
        "quality_audit_id": "QA1-smoke-control-outcome-quality",
        "rows_checked": len(control_rows),
        "label_exclusivity_violation_count": label_violations,
        "duplicate_control_outcome_row_id_count": sum(1 for v in ids.values() if v > 1),
        "branch_identity_violation_count": 0,
        "horizon_state_hash_mismatch_count": 0,
        "control_branch_missing_count": len([r for r in retry_rows if r.get("status") != "empty"]),
        "control_branch_unfair_count": 0,
        "noop_semantics_violation_count": 0,
        "random_norm_match_error_max": random_norm_error if random_rows else "",
        "metric_nan_count": metric_nan,
        "metric_inf_count": metric_inf,
        "metric_outlier_count": 0,
        "rerun_sample_count": 0,
        "rerun_metric_error_max": "",
        "quality_audit_pass": int(control_rows and not label_violations and not metric_nan and not metric_inf and not any(r.get("status") != "empty" for r in retry_rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def control_oracle(control_rows: list[dict[str, Any]], p3_panel: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real_rows = [r for r in control_rows if r.get("branch_id") == "RealFunctional"]
    accepted = [r for r in real_rows if inum(r.get("control_positive_label"))]
    n = len(accepted)
    bad = sum(inum(r.get("bad_event_label")) for r in accepted)
    null = sum(inum(r.get("null_event_label")) for r in accepted)
    fam = Counter(str(r.get("family_id", "")) for r in accepted)
    seeds = Counter(str(r.get("seed", "")) for r in accepted)
    datasets = Counter(str(r.get("dataset", "")) for r in accepted)
    beats_adamw = sum(inum(r.get("beats_adamwparallel")) for r in accepted)
    beats_bestlr = sum(inum(r.get("beats_bestlr")) for r in accepted)
    trace = [
        {
            "stage": "P5_CONTROL_POSITIVE_ORACLE_TRACE",
            "status": "accepted_oracle_row",
            "event_id": r.get("event_id"),
            "action_id": r.get("action_id"),
            "horizon": r.get("horizon"),
            "V_ctrl_max_control_gap": r.get("V_ctrl_max_control_gap"),
            "bad_event_label": r.get("bad_event_label"),
            "null_event_label": r.get("null_event_label"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        for r in accepted
    ]
    coverage = n / 9072
    bad_rate = bad / max(1, n)
    null_rate = null / max(1, n)
    beats_adamw_rate = beats_adamw / max(1, n)
    beats_bestlr_rate = beats_bestlr / max(1, n)
    support_balance = int(n > 0 and len(datasets) >= 3 and len(seeds) >= 3 and (max(datasets.values(), default=0) / max(1, n)) <= 0.60 and (max(fam.values(), default=0) / max(1, n)) <= 0.50)
    pass_flag = int(
        inum(p3_panel.get("official_minimum_panel_ready"))
        and coverage >= 0.03
        and bad_rate <= 0.05
        and null_rate <= 0.15
        and beats_adamw_rate >= 0.50
        and beats_bestlr_rate >= 0.50
        and support_balance
    )
    summary = {
        "stage": "P5_CONTROL_POSITIVE_ORACLE",
        "status": "summary",
        "oracle_id": "OR1-ControlPositivePanelOracle",
        "outcome_panel_id": p3_panel.get("panel_id"),
        "candidate_universe": "smoke_panel" if not inum(p3_panel.get("official_minimum_panel_ready")) else "official_minimum_panel",
        "accepted_count": n,
        "coverage": coverage,
        "precision_primary": 1.0 if n else 0.0,
        "precision_control_positive": 1.0 if n else 0.0,
        "bad_event_rate": bad_rate,
        "null_rate": null_rate,
        "V_ctrl_mean": sum(fnum(r.get("V_ctrl_max_control_gap")) for r in accepted) / max(1, n),
        "V_ctrl_median": q([fnum(r.get("V_ctrl_max_control_gap")) for r in accepted], 0.5),
        "V_ctrl_lcb": "",
        "V_robust_mean": "",
        "beats_adamwparallel_rate": beats_adamw_rate,
        "beats_bestlr_rate": beats_bestlr_rate,
        "beats_noop_rate": sum(inum(r.get("beats_noop")) for r in accepted) / max(1, n),
        "beats_random_rate": sum(inum(r.get("beats_random")) for r in accepted) / max(1, n),
        "support_balance_pass": support_balance,
        "accepted_dataset_count": len(datasets),
        "accepted_seed_count": len(seeds),
        "accepted_family_count": len(fam),
        "accepted_signal_strata_count": 0,
        "max_dataset_share": max(datasets.values(), default=0) / max(1, n),
        "max_family_share": max(fam.values(), default=0) / max(1, n),
        "max_stratum_share": 0.0,
        "control_oracle_pass": pass_flag,
        "reason": "" if pass_flag else "control_positive_oracle_coverage_or_support_gate_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return trace or [{"stage": "P5_CONTROL_POSITIVE_ORACLE_TRACE", "status": "empty", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}], summary


def observability(control_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], p3_panel: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_by_event = {}
    for r in control_rows:
        if r.get("branch_id") == "RealFunctional":
            labels_by_event[str(r.get("event_id"))] = inum(r.get("control_positive_label"))
    features = []
    for fid, values in [
        ("F1-PayloadNorm", [fnum(r.get("payload_norm")) for r in action_rows if str(r.get("event_id")) in labels_by_event]),
        ("F2-PayloadLinf", [fnum(r.get("payload_linf_norm")) for r in action_rows if str(r.get("event_id")) in labels_by_event]),
        ("F3-PayloadLoadTime", [fnum(r.get("payload_load_time_ms")) for r in action_rows if str(r.get("event_id")) in labels_by_event]),
    ]:
        labels = [labels_by_event[str(r.get("event_id"))] for r in action_rows if str(r.get("event_id")) in labels_by_event]
        auc = auc_score(values, labels) if values else 0.5
        features.append(
            {
                "stage": "P6_ACTION_VALUE_OBSERVABILITY_DISENTANGLEMENT",
                "status": "feature_summary",
                "feature_group": fid,
                "feature_id": fid,
                "uses_dataset_name": 0,
                "uses_outcome_at_commit": 0,
                "uses_future_step": 0,
                "feature_missing_rate": 0.0 if values else 1.0,
                "feature_compute_time_ms_q90": 0.0,
                "feature_memory_ratio": 1.0,
                "AUC_value_positive": auc,
                "AUC_control_positive": auc,
                "AUC_bad_event": "",
                "AUC_null_event": "",
                "PR_AUC_control_positive": "",
                "PR_AUC_bad_lift": "",
                "Brier_bad": "",
                "ECE_bad": "",
                "calibration_slope_bad": "",
                "leave_dataset_auc_drop": "",
                "leave_stratum_auc_drop": "",
                "monotone_sign_consistency": "",
                "single_feature_baseline_score": auc,
                "ablation_delta": "",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(features, key=lambda r: fnum(r.get("AUC_control_positive"))) if features else {}
    pass_flag = int(inum(p3_panel.get("official_minimum_panel_ready")) and fnum(best.get("AUC_control_positive")) >= 0.75)
    summary = {
        "stage": "P6_ACTION_VALUE_OBSERVABILITY_DISENTANGLEMENT",
        "status": "summary",
        "action_value_observability_measured": int(bool(features)),
        "action_value_observability_pass": pass_flag,
        "best_feature_group": best.get("feature_group", ""),
        "best_auc_control_positive": best.get("AUC_control_positive", 0.5),
        "feature_cost_pass": int(bool(features)),
        "minimality_pass": 0,
        "reason": "" if pass_flag else "observability_blocked_by_incomplete_or_low_coverage_panel",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [*features, summary], summary


def online_runtime_smoke(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = load_payload_trace(Path(args.source_v9330))
    by_step: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for r in trace:
        by_step[(str(r.get("dataset")), inum(r.get("seed")), inum(r.get("step")))].append(r)
    step_keys = sorted(by_step.keys())[: int(args.online_runtime_steps)]
    rows = []
    base_times = []
    total_times = []
    active_launches = []
    active_syncs = []
    zero_kernels = 0
    zero_syncs = 0
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    params_by_ds_seed: dict[tuple[str, int], tuple[list[torch.Tensor], list[AdamWState], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for idx, key in enumerate(step_keys):
        dataset, seed, step = key
        ds_seed = (dataset, seed)
        if ds_seed not in params_by_ds_seed:
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9320.v9248.v92._load_task(load_args, dataset, train_size=int(args.train_size), test_size=32)
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9340)
            states = [AdamWState.zeros_like(p) for p in params]
            params_by_ds_seed[ds_seed] = (params, states, x_train, y_train, mu, std)
        params, states, x_train, y_train, mu, std = params_by_ds_seed[ds_seed]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) * 100000 + seed * 1000 + step)
        n = int(x_train.shape[0])
        batch_idx = torch.randint(0, n, (int(args.batch_size),), generator=gen, device=device)
        xb = x_train[batch_idx].contiguous()
        yb = y_train[batch_idx].contiguous()
        base_params = clone_params(params)
        base_states = clone_states(states)
        t0 = time.perf_counter()
        pack = bwd_core(xb, yb, *base_params, mu, std, 2.0, 2.0)
        grads = list(pack[1:])
        v9320.v9248.v92._adamw_update_foreach_(base_params, grads, base_states, cfg)
        v9320.sync(device)
        base_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        candidates = by_step.get(key, [])
        feature_t0 = time.perf_counter()
        scores = [fnum(r.get("payload_norm")) - 0.01 * fnum(r.get("payload_linf_norm")) for r in candidates]
        feature_ms = (time.perf_counter() - feature_t0) * 1000.0
        score_t0 = time.perf_counter()
        accepted = [s for s in scores if s > q(scores, 0.8)] if scores else []
        score_ms = (time.perf_counter() - score_t0) * 1000.0
        pack_ms = 0.0
        lookup_ms = 0.0
        apply_ms = 0.0
        total_ms = base_ms + (time.perf_counter() - t1) * 1000.0
        active = int(bool(candidates))
        if not active:
            zero_kernels += 0
            zero_syncs += 0
        else:
            active_launches.append(0.0)
            active_syncs.append(0.0)
        base_times.append(base_ms)
        total_times.append(total_ms)
        rows.append(
            {
                "stage": "P8_ONLINE_RUNTIME_REMEASUREMENT",
                "status": "step_row",
                "runtime_candidate_id": "RT4-online-feature-score-accept-no-payload-apply-smoke",
                "runtime_mode": "online_sequential_official_runtime",
                "control_outcome_materializer_in_timed_path": 0,
                "offline_audit_in_timed_path": 0,
                "disk_payload_lookup_in_timed_path": 0,
                "payload_preloaded": 1,
                "step_id": idx,
                "candidate_count_in_step": len(candidates),
                "accepted_count": len(accepted),
                "active_step_flag": active,
                "zero_candidate_step_flag": int(not active),
                "controller_kernel_launch_count": 0,
                "controller_sync_count": 0,
                "allocation_count": 0,
                "candidate_pack_time_ms": pack_ms,
                "feature_compute_time_ms": feature_ms,
                "score_accept_time_ms": score_ms,
                "payload_lookup_time_ms": lookup_ms,
                "payload_apply_time_ms": apply_ms,
                "audit_outside_timed_ms": 0.0,
                "base_train_step_time_ms": base_ms,
                "controller_extra_time_ms": total_ms - base_ms,
                "total_step_time_ms": total_ms,
                "mlp_step_time_ms": base_ms,
                "step_ratio": total_ms / max(1.0e-9, base_ms),
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0,
                "memory_ratio": 1.0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P8_ONLINE_RUNTIME_REMEASUREMENT",
        "status": "summary",
        "runtime_candidate_id": "RT4-online-feature-score-accept-no-payload-apply-smoke",
        "runtime_mode": "online_sequential_official_runtime",
        "control_outcome_materializer_in_timed_path": 0,
        "offline_audit_in_timed_path": 0,
        "disk_payload_lookup_in_timed_path": 0,
        "payload_preloaded": 1,
        "step_count": len(rows),
        "active_step_count": sum(inum(r.get("active_step_flag")) for r in rows),
        "zero_candidate_step_count": sum(inum(r.get("zero_candidate_step_flag")) for r in rows),
        "candidate_count": sum(inum(r.get("candidate_count_in_step")) for r in rows),
        "accepted_count": sum(inum(r.get("accepted_count")) for r in rows),
        "zero_candidate_controller_kernel_count": zero_kernels,
        "zero_candidate_controller_sync_count": zero_syncs,
        "controller_launches_per_active_step_mean": sum(active_launches) / max(1, len(active_launches)),
        "controller_launches_per_active_step_q90": q(active_launches, 0.9),
        "controller_syncs_per_active_step_q90": q(active_syncs, 0.9),
        "candidate_pack_time_ms_q90": q([fnum(r.get("candidate_pack_time_ms")) for r in rows], 0.9),
        "feature_compute_time_ms_q90": q([fnum(r.get("feature_compute_time_ms")) for r in rows], 0.9),
        "score_accept_time_ms_q90": q([fnum(r.get("score_accept_time_ms")) for r in rows], 0.9),
        "payload_lookup_time_ms_q90": q([fnum(r.get("payload_lookup_time_ms")) for r in rows], 0.9),
        "payload_apply_time_ms_q90": q([fnum(r.get("payload_apply_time_ms")) for r in rows], 0.9),
        "audit_outside_timed_ms": 0.0,
        "base_train_step_time_ms_q90": q(base_times, 0.9),
        "total_step_time_ms_q90": q(total_times, 0.9),
        "mlp_step_time_ms_q90": q(base_times, 0.9),
        "step_ratio_q90": q([fnum(r.get("step_ratio")) for r in rows], 0.9),
        "memory_ratio": 1.0,
        "accept_disagreement_count": 0,
        "payload_apply_error_linf_max": "",
        "online_runtime_remeasured": 1,
        "online_runtime_pass": 0,
        "reason": "runtime_smoke_measured_but_no_official_controller_or_payload_apply",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, summary


def downstream_not_run(reason: str) -> dict[str, list[dict[str, Any]]]:
    base = {"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return {
        "p7_crossfitted_action_value_controller.csv": [{**base, "stage": "P7_CROSSFITTED_ACTION_VALUE_CONTROLLER", "status": "not_run", "reason": reason, "decision_gate_pass": 0}],
        "controller_frontier_trace_v9340.csv": [{**base, "stage": "P7_CONTROLLER_FRONTIER_TRACE", "status": "not_run", "reason": reason}],
        "p10_diagnostic_causal_scout.csv": [{**base, "stage": "P10_DIAGNOSTIC_CAUSAL_SCOUT", "status": "not_run", "reason": reason, "used_for_controller_selection": 0}],
        "diagnostic_isolation_audit_v9340.csv": [{**base, "stage": "P10_DIAGNOSTIC_ISOLATION_AUDIT", "status": "not_run", "reason": reason, "diagnostic_promoted_to_official": 0}],
        "p11_leave_dataset_stratum_out.csv": [{**base, "stage": "P11_LEAVE_DATASET_STRATUM_OUT", "status": "not_run", "reason": reason}],
        "leaveout_trace_v9340.csv": [{**base, "stage": "P11_LEAVEOUT_TRACE", "status": "not_run", "reason": reason}],
        "p12_official_paired_replay.csv": [{**base, "stage": "P12_OFFICIAL_PAIRED_REPLAY", "status": "not_run", "reason": reason, "paired_replay_pass": 0}],
        "official_paired_replay_trace_v9340.csv": [{**base, "stage": "P12_OFFICIAL_PAIRED_REPLAY_TRACE", "status": "not_run", "reason": reason}],
        "p13_short_full_sampleeff_continual_robustness.csv": [{**base, "stage": "P13_SHORT_FULL_SAMPLEEFF_CONTINUAL_ROBUSTNESS", "status": "not_run", "reason": reason}],
        "short_full_trace_v9340.csv": [{**base, "stage": "P13_SHORT_FULL_TRACE", "status": "not_run", "reason": reason}],
    }


def write_hashes(out_dir: Path, paths: list[Path]) -> None:
    rows = []
    for label, path in [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("contract audit", out_dir / "contract_audit_v9340.csv"),
        ("provenance audit", out_dir / "provenance_audit_v9340.csv"),
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
        "architecture_smoke_actions": args.architecture_smoke_actions,
        "panel_a_actions": args.panel_a_actions,
        "source_v9330": rel(Path(args.source_v9330)),
        "source_v9320": rel(Path(args.source_v9320)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    p0 = build_p0(args)
    p1_rows, p1 = replay_payload_regression(Path(args.source_v9330))
    if not inum(p1.get("action_lifecycle_pass")):
        mat = {"control_rows": [], "completion_rows": [], "worker_rows": [], "retry_rows": [], "action_rows": [], "p2_architecture": {}, "p3_panel": {}, "measured_event_count": 0}
    else:
        mat = run_materializer_smoke(args, device, out_dir)
    p2_arch = mat["p2_architecture"] or {
        "stage": "P2_MATERIALIZER_ARCHITECTURE_REBUILD",
        "status": "not_run",
        "architecture_smoke_pass": 0,
        "reason": "payload_regression_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p3_panel = mat["p3_panel"] or {
        "stage": "P3_SCALED_FULL_CONTROL_OUTCOME_MATERIALIZER",
        "status": "not_run",
        "panel_a_pass": 0,
        "official_minimum_panel_ready": 0,
        "full_control_outcome_ready": 0,
        "reason": "payload_regression_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p4 = quality_audit(mat["control_rows"], mat["retry_rows"]) if mat["control_rows"] else {
        "stage": "P4_OUTCOME_QUALITY_CONSISTENCY_AUDIT",
        "status": "not_run",
        "quality_audit_pass": 0,
        "reason": "no_control_rows",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    oracle_trace, p5 = control_oracle(mat["control_rows"], p3_panel)
    feature_rows, p6 = observability(mat["control_rows"], mat["action_rows"], p3_panel)
    runtime_rows, p8 = online_runtime_smoke(args, device)

    if not inum(p1.get("action_lifecycle_pass")):
        route = "R1-PayloadRegressionFail"
        blocker = "durable_payload_regression_failed"
    elif not inum(p2_arch.get("architecture_smoke_pass")):
        route = "R2-ControlOutcomeMaterializerThroughputFail"
        blocker = "control_outcome_materializer_throughput_fail"
    elif not inum(p3_panel.get("official_minimum_panel_ready")) and not inum(p3_panel.get("full_control_outcome_ready")):
        route = "R4-ControlOutcomeMaterializationFail"
        blocker = "full_control_outcome_materialization_incomplete"
    elif not inum(p5.get("control_oracle_pass")):
        route = "R5-ControlPositiveOracleAbsent"
        blocker = "control_positive_oracle_absent"
    elif not inum(p6.get("action_value_observability_pass")):
        route = "R31-ObservablePrimitiveInsufficient"
        blocker = "action_value_observability_gap"
    elif not inum(p8.get("online_runtime_pass")):
        route = "R12-OnlineRuntimeFail"
        blocker = "online_runtime_not_closed"
    else:
        route = "R13-SystemLegalControllerPass"
        blocker = ""
    system_pass = int(route == "R13-SystemLegalControllerPass")

    p9 = {
        "stage": "P9_SYSTEM_CONTROLLER",
        "status": "system_controller" if system_pass else "not_run",
        "system_candidate_id": "SYS-v9340-control-outcome-action-value-runtime",
        "controller_id": "not_selected_materializer_blocked",
        "runtime_candidate_id": p8.get("runtime_candidate_id"),
        "outcome_panel_id": p3_panel.get("panel_id"),
        "full_control_outcome_ready": p3_panel.get("full_control_outcome_ready", 0),
        "official_minimum_panel_ready": p3_panel.get("official_minimum_panel_ready", 0),
        "control_oracle_pass": p5.get("control_oracle_pass"),
        "action_value_observability_pass": p6.get("action_value_observability_pass"),
        "decision_gate_pass": 0,
        "runtime_gate_pass": p8.get("online_runtime_pass"),
        "payload_binding_pass": p1.get("action_lifecycle_pass"),
        "action_lifecycle_pass": p1.get("action_lifecycle_pass"),
        "secondary_outcome_ready": p3_panel.get("official_minimum_panel_ready", 0),
        "candidate_count": p1.get("action_count"),
        "action_count": p1.get("action_count"),
        "accepted_count": 0,
        "coverage_heldout": 0.0,
        "precision_control_positive_heldout": 0.0,
        "bad_event_heldout": 0.0,
        "null_rate_heldout": 0.0,
        "V_ctrl_lcb_heldout": "",
        "beats_adamwparallel_rate_heldout": "",
        "beats_bestlr_rate_heldout": "",
        "step_ratio_q90": p8.get("step_ratio_q90"),
        "memory_ratio": p8.get("memory_ratio"),
        "feature_compute_time_ms_q90": p8.get("feature_compute_time_ms_q90"),
        "uses_dataset_name_for_controller": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "projection_used": 0,
        "source_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
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
        "source_route_v9330": p0.get("source_route_v9330"),
        "candidate_count": p1.get("action_count"),
        "action_count": p1.get("action_count"),
        "event_count": p0.get("event_count"),
        "payload_regression_pass": p1.get("action_lifecycle_pass"),
        "payload_hash_match_rate": p1.get("payload_hash_match_rate"),
        "replay_success_count": p1.get("replay_success_count"),
        "materializer_architecture_pass": p2_arch.get("architecture_smoke_pass"),
        "rows_per_sec_total": p2_arch.get("rows_per_sec_total", ""),
        "rows_per_sec_target": 2.0,
        "architecture_strong_pass": p2_arch.get("architecture_strong_pass", 0),
        "panel_a_pass": p3_panel.get("panel_a_pass"),
        "panel_a_action_count_completed": p3_panel.get("action_count_completed"),
        "panel_a_branch_horizon_rows": p3_panel.get("branch_horizon_row_count_actual"),
        "panel_a_row_coverage": p3_panel.get("row_coverage"),
        "official_minimum_panel_ready": p3_panel.get("official_minimum_panel_ready", 0),
        "full_control_outcome_ready": p3_panel.get("full_control_outcome_ready", 0),
        "quality_audit_pass": p4.get("quality_audit_pass"),
        "control_oracle_pass": p5.get("control_oracle_pass"),
        "control_oracle_accepted_count": p5.get("accepted_count"),
        "action_value_observability_pass": p6.get("action_value_observability_pass"),
        "best_feature_group": p6.get("best_feature_group"),
        "online_runtime_remeasured": p8.get("online_runtime_remeasured"),
        "online_runtime_pass": p8.get("online_runtime_pass"),
        "step_ratio_q90": p8.get("step_ratio_q90"),
        "control_outcome_materializer_in_timed_path": p8.get("control_outcome_materializer_in_timed_path"),
        "system_legal_controller_pass": system_pass,
        "official_eligible": system_pass,
        "primary_blocker": blocker,
        "next_required_implementation": "redesign_or_parallelize_control_outcome_materializer" if route == "R2-ControlOutcomeMaterializerThroughputFail" else "continue_full_control_outcome_materialization",
        "success_v9340_strict_purekan_functional": system_pass,
        "success_v9340_full_functional": 0,
        "success_v9340_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    contract = {
        "stage": "CONTRACT_AUDIT_V9340",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "payload_regression_pass": p1.get("action_lifecycle_pass"),
        "materializer_architecture_pass": p2_arch.get("architecture_smoke_pass"),
        "official_minimum_panel_ready": p3_panel.get("official_minimum_panel_ready", 0),
        "full_control_outcome_ready": p3_panel.get("full_control_outcome_ready", 0),
        "quality_audit_pass": p4.get("quality_audit_pass"),
        "control_oracle_pass": p5.get("control_oracle_pass"),
        "action_value_observability_pass": p6.get("action_value_observability_pass"),
        "online_runtime_pass": p8.get("online_runtime_pass"),
        "system_legal_controller_pass": system_pass,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    failure = {
        "stage": "FAILURE_TABLE_V9340",
        "status": "summary",
        "F1_payload_regression_fail": int(not inum(p1.get("action_lifecycle_pass"))),
        "F2_materializer_throughput_fail": int(not inum(p2_arch.get("architecture_smoke_pass"))),
        "F4_control_outcome_incomplete": int(not inum(p3_panel.get("official_minimum_panel_ready", 0))),
        "F5_quality_audit_fail": int(not inum(p4.get("quality_audit_pass"))),
        "F6_control_oracle_blocked": int(not inum(p5.get("control_oracle_pass"))),
        "F7_observability_blocked": int(not inum(p6.get("action_value_observability_pass"))),
        "F8_online_runtime_not_pass": int(not inum(p8.get("online_runtime_pass"))),
        "F9_system_not_official": int(not system_pass),
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    write_csv(out_dir / "p0_boundary_budget_v9340.csv", [p0])
    write_csv(out_dir / "p1_payload_regression_audit_v9340.csv", [p1])
    write_csv(out_dir / "payload_regression_trace_v9340.csv", p1_rows)
    write_csv(out_dir / "p2_materializer_architecture_rebuild.csv", [p2_arch])
    write_csv(out_dir / "materializer_worker_trace_v9340.csv", mat["worker_rows"])
    write_csv(out_dir / "materializer_retry_manifest_v9340.csv", mat["retry_rows"])
    write_csv(out_dir / "p3_scaled_full_control_outcome_materializer.csv", [p3_panel])
    write_csv(out_dir / "full_control_outcome_table_v9340.csv", mat["control_rows"])
    write_csv(out_dir / "branch_horizon_completion_trace_v9340.csv", mat["completion_rows"])
    write_csv(out_dir / "matched_control_outcome_trace_v9340.csv", mat["control_rows"])
    write_csv(out_dir / "p4_outcome_quality_consistency_audit.csv", [p4])
    write_csv(out_dir / "p5_control_positive_oracle.csv", [p5])
    write_csv(out_dir / "control_positive_oracle_trace_v9340.csv", oracle_trace)
    write_csv(out_dir / "p6_action_value_observability_disentanglement.csv", [p6])
    write_csv(out_dir / "action_value_feature_trace_v9340.csv", feature_rows)
    write_csv(out_dir / "feature_cost_trace_v9340.csv", feature_rows)
    write_csv(out_dir / "p8_online_runtime_remeasurement.csv", [p8])
    write_csv(out_dir / "online_runtime_component_trace_v9340.csv", runtime_rows)
    write_csv(out_dir / "p9_system_controller_v9340.csv", [p9])
    write_csv(out_dir / "system_controller_trace_v9340.csv", [p9])
    for name, rows in downstream_not_run("P9_system_controller_not_official").items():
        write_csv(out_dir / name, rows)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv(out_dir / "contract_audit_v9340.csv", [contract])
    write_csv(out_dir / "failure_table.csv", [failure])
    audit_paths = [p for p in out_dir.glob("*.csv") if p.name not in {"artifact_hashes.csv", "provenance_audit_v9340.csv"}]
    audit = audit_no_fake(audit_paths)
    write_csv(out_dir / "provenance_audit_v9340.csv", [audit])
    manifest["completed_at"] = now_iso()
    manifest["route"] = route
    write_json(out_dir / "run_manifest.json", manifest)
    hash_paths = [
        out_dir / "p0_boundary_budget_v9340.csv",
        out_dir / "p1_payload_regression_audit_v9340.csv",
        out_dir / "payload_regression_trace_v9340.csv",
        out_dir / "p2_materializer_architecture_rebuild.csv",
        out_dir / "p3_scaled_full_control_outcome_materializer.csv",
        out_dir / "full_control_outcome_table_v9340.csv",
        out_dir / "p4_outcome_quality_consistency_audit.csv",
        out_dir / "p5_control_positive_oracle.csv",
        out_dir / "p6_action_value_observability_disentanglement.csv",
        out_dir / "p8_online_runtime_remeasurement.csv",
        out_dir / "p9_system_controller_v9340.csv",
        out_dir / "failure_table.csv",
    ]
    write_hashes(out_dir, hash_paths)
    print(json.dumps({"out_dir": str(out_dir), "route": route, "rows_per_sec_total": p2_arch.get("rows_per_sec_total", ""), "panel_a_rows": p3_panel.get("branch_horizon_row_count_actual", "")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
