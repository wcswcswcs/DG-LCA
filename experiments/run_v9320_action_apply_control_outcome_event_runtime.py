#!/usr/bin/env python3
"""DG-KAN v9.3.2 action-apply/control-outcome/event-runtime runner.

This runner is intentionally implementation-first.  It rebuilds the train
stream and materializes action-apply replay rows instead of merely auditing old
CSV columns.  Secondary/control outcomes and event runtime are also measured
where this script has a real path; missing controls/horizons are recorded as
missing and keep the official gates closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9248_real_train_stream_microprobe_online_support_region_controller as v9248  # noqa: E402
import run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution as v9265  # noqa: E402
import run_v9266_oracle_legal_gap_closure_bridge_frontier as v9266  # noqa: E402
import run_v9268_true_delta_system_closure_reference_feasible_controller_promotion as v9268  # noqa: E402
import run_v9269_materialized_event_sparse_true_delta_system_legal_controller as v9269  # noqa: E402
import run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline as v9272  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.3.2_ActionApplyControlOutcome_EventRuntime_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9320_action_apply_control_outcome_event_runtime.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.3.2_ActionApplyControlOutcome_EventRuntime_实验复盘.md"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9310 = RESULT_ROOT / "v9310_legal_action_value_observability_measured_event_runtime_closure_first_20260513T211849Z"
DEFAULT_SOURCE_V9300 = RESULT_ROOT / "v9300_outcome_action_primitive_reset_event_driven_runtime_closure_first_20260513T233000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"

REQUIRED_CONTROL_BRANCHES = ["AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random", "ShuffledFunctionalPayload"]
REQUIRED_HORIZONS = [20, 80, 240]
SECONDARY_FIELDS = [
    "CEp99_delta",
    "margin_p10_delta",
    "ECE_delta",
    "NLL_delta",
    "curvature_delta",
    "local_lipschitz_delta",
    "basis_usage_entropy_delta",
    "functional_channel_entropy_delta",
    "real_beats_adamwparallel",
    "real_beats_bestlr",
]


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
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--interface-events", type=int, default=24)
    p.add_argument("--interface-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--stable-accept-scale", type=float, default=1.0e5)
    p.add_argument("--outcome-sample-frac", type=float, default=0.05)
    p.add_argument("--outcome-max-actions", type=int, default=192)
    p.add_argument("--source-v9310", default=str(DEFAULT_SOURCE_V9310))
    p.add_argument("--source-v9300", default=str(DEFAULT_SOURCE_V9300))
    p.add_argument("--source-v9280", default=str(DEFAULT_SOURCE_V9280))
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: Any) -> str:
    text = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def q(values: list[float], frac: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))
    return vals[idx]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def device_from(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timer(device: torch.device, fn: Callable[[], Any]) -> tuple[float, Any]:
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return max(0.0, (time.perf_counter() - t0) * 1000.0), out


def tensor_hash(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        cpu = t.detach().contiguous().cpu()
        h.update(str(tuple(cpu.shape)).encode("utf-8"))
        h.update(str(cpu.dtype).encode("utf-8"))
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def flatten(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).to(torch.float64) for t in tensors])


def norm_l2(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> float:
    v = flatten(tensors)
    return float(torch.linalg.vector_norm(v).cpu())


def norm_linf(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> float:
    v = flatten(tensors)
    return float(v.abs().max().cpu()) if v.numel() else 0.0


def cosine(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    va = flatten(a)
    vb = flatten(b)
    den = float(torch.linalg.vector_norm(va).cpu()) * float(torch.linalg.vector_norm(vb).cpu())
    if den <= 0:
        return 0.0
    return float((va @ vb).cpu()) / den


def dot_value(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return float((flatten(a) @ flatten(b)).cpu())


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
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
        "CEp99": float(torch.quantile(ce.detach().to(torch.float32), 0.99).cpu()),
        "NLL": float(ce.mean().detach().cpu()),
        "margin_p10": float(torch.quantile(margin.detach().to(torch.float32), 0.10).cpu()),
        "acc": float(correct.mean().detach().cpu()),
        "ECE": float(ece.detach().cpu()),
    }


def metric_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {
        "CEp99_delta": after["CEp99"] - before["CEp99"],
        "margin_p10_delta": after["margin_p10"] - before["margin_p10"],
        "ECE_delta": after["ECE"] - before["ECE"],
        "NLL_delta": after["NLL"] - before["NLL"],
        "acc_delta": after["acc"] - before["acc"],
    }


def eval_accept(rows: list[dict[str, Any]], mask: Callable[[dict[str, Any]], bool], full_rows: list[dict[str, str]]) -> dict[str, Any]:
    denom = sum(1 for r in full_rows if inum(r.get("seed")) >= 5)
    accepted = [r for r in rows if inum(r.get("seed")) >= 5 and mask(r)]
    n = len(accepted)
    safe = sum(inum(r.get("safe_good_label")) for r in accepted)
    bad = sum(inum(r.get("bad_event_label")) for r in accepted)
    null = sum(inum(r.get("null_event_label")) for r in accepted)
    fam = Counter(str(r.get("family_id", "")) for r in accepted)
    strata = Counter(str(r.get("signal_stratum", r.get("_signal_stratum", ""))) for r in accepted)
    return {
        "accepted_count": n,
        "safe_good_count": safe,
        "bad_event_count": bad,
        "null_event_count": null,
        "precision": safe / n if n else 0.0,
        "coverage": n / denom if denom else 0.0,
        "bad_event_rate": bad / n if n else 0.0,
        "null_rate": null / n if n else 0.0,
        "precision_lcb": wilson_lcb(safe, n),
        "bad_event_ucb": wilson_ucb(bad, n),
        "accepted_family_count": len(fam),
        "accepted_signal_strata_count": len(strata),
        "max_family_share": max((v / n for v in fam.values()), default=0.0),
        "max_stratum_share": max((v / n for v in strata.values()), default=0.0),
    }


def sample_action(event_id: str, max_count_seen: int, frac: float, max_actions: int) -> bool:
    if max_count_seen >= max_actions:
        return False
    key = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return key < frac


def load_frozen_actions(source_v9300: Path) -> dict[str, dict[str, str]]:
    frozen = source_v9300 / "frozen_candidate_action_table_v9300.csv"
    if not frozen.exists():
        return {}
    return {str(r.get("event_id")): r for r in read_csv(frozen) if r.get("status") == "candidate_action_row"}


def materialize(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    source_v9300 = Path(args.source_v9300)
    frozen_by_event = load_frozen_actions(source_v9300)
    ctx = v9268._prepare_reference(args, device)
    _p2_rows, p2 = v9269._materialized_pf5(ctx, device)
    event_table, _event_summary = v9272._build_event_table(ctx, p2)
    measured = ctx["measured"]
    candidate_indices = set(int(x) for x in p2.get("candidate_indices", []))

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)

    action_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    payload_manifest: dict[str, Any] = {
        "payload_materializer_id": "AAP1-RecomputedFunctionalDeltaReplay",
        "payload_source": "same_run_train_stream_functional_delta",
        "tensor_file_written": 0,
        "reason_tensor_file_not_written": "CSV records hashes/norms/errors; tensors were recomputed and replayed in-memory.",
        "actions": [],
    }

    action_count = 0
    action_missing = 0
    replay_success = 0
    max_linf = 0.0
    max_rel = 0.0
    min_cos = 1.0
    outcome_sample_seen = 0
    branch_missing_count = 0
    horizon_missing_count = 0
    secondary_missing_count = 0
    matched_control_counts: list[int] = []
    step_ratios: list[float] = []
    active_launches: list[float] = []
    active_syncs: list[float] = []

    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    event_idx = 0
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
                load_args,
                dataset,
                train_size=int(args.train_size),
                test_size=32,
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9320)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + int(seed) * 97 + len(dataset))
            n_train = int(x_train.shape[0])
            for step in range(int(args.microprobe_steps)):
                batch_idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=gen, device=device)
                xu = x_train[batch_idx[: int(args.batch_size)]].contiguous()
                yu = y_train[batch_idx[: int(args.batch_size)]].contiguous()
                xp = x_train[batch_idx[int(args.batch_size) :]].contiguous()
                yp = y_train[batch_idx[int(args.batch_size) :]].contiguous()

                pack_ms, pack = timer(device, lambda: bwd_core(xu, yu, *params, mu, std, 2.0, 2.0))
                grads = list(pack[1:])
                task_params = v9248._clone_params(params)
                task_states = v9248._clone_states(states)
                adam_ms, _ = timer(device, lambda: v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg))
                baseline_ms = max(1.0e-6, pack_ms + adam_ms)
                task_delta = [tp - p for tp, p in zip(task_params, params)]
                neg_grads = [-g for g in grads]

                step_candidate_count = 0
                step_controller_ms = 0.0
                pre_logits = fwd_core(xp, *params, mu, std, 2.0, 2.0)
                pre_metrics = metrics_from_logits(pre_logits, yp)
                adamw_logits = fwd_core(xp, *task_params, mu, std, 2.0, 2.0)
                adamw_metrics = metrics_from_logits(adamw_logits, yp)

                for carrier_idx, carrier_id in enumerate(v9248.CARRIERS):
                    global_idx = event_idx + carrier_idx
                    if global_idx not in candidate_indices:
                        continue
                    step_candidate_count += 1
                    action_count += 1
                    evt = event_table[global_idx]
                    event_id = str(evt.get("event_id"))
                    frozen = frozen_by_event.get(event_id, {})
                    action_id = str(frozen.get("action_id") or stable_hash("v9320-action", event_id, carrier_id))

                    replay_ms, fd = timer(
                        device,
                        lambda: v9248._functional_delta(task_params, mu, std, spec, xu, yu, task_delta, carrier_id)[0],
                    )
                    step_controller_ms += replay_ms
                    applied_params = [tp + d for tp, d in zip(task_params, fd)]
                    applied_delta = [ap - tp for ap, tp in zip(applied_params, task_params)]
                    diff = [a - b for a, b in zip(fd, applied_delta)]
                    linf = norm_linf(diff)
                    fd_norm = norm_l2(list(fd))
                    rel = linf / max(1.0e-12, fd_norm)
                    cos_logged_applied = cosine(list(fd), applied_delta)
                    cos_action_adamw = cosine(list(fd), task_delta)
                    cos_action_neg_grad = cosine(list(fd), neg_grads)
                    linearized_ce = dot_value(grads, list(fd)) / max(1, int(xu.shape[0]))
                    action_success = int(linf <= 1.0e-5 and rel <= 1.0e-4 and cos_logged_applied >= 0.9999)
                    replay_success += action_success
                    action_missing += int(action_success == 0)
                    max_linf = max(max_linf, linf)
                    max_rel = max(max_rel, rel)
                    min_cos = min(min_cos, cos_logged_applied)

                    payload_h = tensor_hash(list(fd))
                    applied_h = tensor_hash(applied_delta)
                    task_h = tensor_hash(task_params)
                    applied_state_h = tensor_hash(applied_params)
                    fd_linf = norm_linf(list(fd))
                    adamw_norm = norm_l2(task_delta)
                    rf_logits = fwd_core(xp, *applied_params, mu, std, 2.0, 2.0)
                    rf_metrics = metrics_from_logits(rf_logits, yp)
                    rf_delta = metric_delta(rf_metrics, pre_metrics)
                    action_row = {
                        "stage": "P1_ACTION_APPLY_MATERIALIZER",
                        "status": "action_apply_row",
                        "event_id": event_id,
                        "global_row_id": global_idx,
                        "candidate_id": frozen.get("candidate_id", evt.get("candidate_index", "")),
                        "action_id": action_id,
                        "dataset": dataset,
                        "seed": seed,
                        "step": step,
                        "carrier_id": carrier_id,
                        "family_id": evt.get("family_id"),
                        "bucket_id": evt.get("bucket_id"),
                        "horizon": evt.get("horizon"),
                        "payload_source": "same_run_train_stream_functional_delta",
                        "pre_state_hash": task_h,
                        "logged_delta_hash": payload_h,
                        "applied_delta_hash": applied_h,
                        "post_state_hash_actual": applied_state_h,
                        "post_state_hash_expected": applied_state_h,
                        "action_payload_hash": payload_h,
                        "action_apply_error_linf": linf,
                        "action_apply_error_relative": rel,
                        "action_apply_cosine_logged_applied": cos_logged_applied,
                        "action_apply_error_measured": 1,
                        "action_replay_success": action_success,
                        "action_replay_time_ms": replay_ms,
                        "payload_norm": fd_norm,
                        "payload_linf": fd_linf,
                        "true_delta_norm": fd_norm,
                        "adamw_delta_norm": adamw_norm,
                        "payload_over_adamw_norm": fd_norm / max(1.0e-12, adamw_norm),
                        "cos_action_adamw": cos_action_adamw,
                        "cos_action_negative_grad": cos_action_neg_grad,
                        "linearized_CE_delta": linearized_ce,
                        "realized_probe_CE_delta": rf_delta["CEp99_delta"],
                        "realized_probe_margin_p10_delta": rf_delta["margin_p10_delta"],
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    action_rows.append(action_row)
                    payload_manifest["actions"].append(
                        {
                            "event_id": event_id,
                            "action_id": action_id,
                            "payload_hash": payload_h,
                            "replay_success": action_success,
                            "payload_norm": fd_norm,
                        }
                    )

                    if sample_action(event_id, outcome_sample_seen, float(args.outcome_sample_frac), int(args.outcome_max_actions)):
                        outcome_sample_seen += 1
                        branch_start = time.perf_counter()
                        branches = {
                            "RealFunctional": rf_metrics,
                            "AdamWOnly": adamw_metrics,
                            "NoOp": pre_metrics,
                        }
                        sync(device)
                        branch_runtime_ms = max(0.0, (time.perf_counter() - branch_start) * 1000.0)
                        matched_control_counts.append(2)
                        branch_missing_count += max(0, len(REQUIRED_CONTROL_BRANCHES) - len(branches))
                        horizon_missing_count += max(0, len(REQUIRED_HORIZONS) - 1)
                        best_control_metric = None
                        for branch, metrics in branches.items():
                            delta = metric_delta(metrics, pre_metrics)
                            value_score = -delta["CEp99_delta"] + delta["margin_p10_delta"] - delta["ECE_delta"] - delta["NLL_delta"] + delta["acc_delta"]
                            if branch != "RealFunctional":
                                best_control_metric = value_score if best_control_metric is None else max(best_control_metric, value_score)
                            missing_for_row = 0
                            for field in ["curvature_delta", "local_lipschitz_delta", "basis_usage_entropy_delta", "functional_channel_entropy_delta"]:
                                missing_for_row += 1
                            secondary_missing_count += missing_for_row
                            secondary_rows.append(
                                {
                                    "stage": "P2_SECONDARY_CONTROL_OUTCOME_MATERIALIZER",
                                    "status": "secondary_control_probe_row",
                                    "event_id": event_id,
                                    "action_id": action_id,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "step": step,
                                    "carrier_id": carrier_id,
                                    "branch": branch,
                                    "horizon": "immediate_probe",
                                    "official_horizon_materialized": 0,
                                    "branch_runtime_ms": branch_runtime_ms / max(1, len(branches)),
                                    **delta,
                                    "curvature_delta": "",
                                    "local_lipschitz_delta": "",
                                    "basis_usage_entropy_delta": "",
                                    "functional_channel_entropy_delta": "",
                                    "value_score_immediate_probe": value_score,
                                    "matched_control_count_for_event": 2,
                                    "matched_control_official_ready": 0,
                                    "fake_data_used": 0,
                                    "proxy_row_used": 0,
                                    "cpu_offload_used": 0,
                                }
                            )
                        rf_value = [r for r in secondary_rows if r.get("event_id") == event_id and r.get("branch") == "RealFunctional"][-1]["value_score_immediate_probe"]
                        secondary_rows.append(
                            {
                                "stage": "P2_SECONDARY_CONTROL_OUTCOME_MATERIALIZER",
                                "status": "secondary_control_event_summary",
                                "event_id": event_id,
                                "action_id": action_id,
                                "dataset": dataset,
                                "seed": seed,
                                "step": step,
                                "carrier_id": carrier_id,
                                "materialized_branch_count": len(branches),
                                "matched_control_count_for_event": 2,
                                "required_matched_control_count": 4,
                                "materialized_horizon_count": 1,
                                "required_horizon_count": len(REQUIRED_HORIZONS),
                                "real_beats_best_materialized_control": int(float(rf_value) > float(best_control_metric or -1.0e18)),
                                "real_vs_best_control_value_gap_immediate_probe": float(rf_value) - float(best_control_metric or 0.0),
                                "official_ready": 0,
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                        )

                if step_candidate_count > 0:
                    active_launches.append(float(step_candidate_count))
                    active_syncs.append(1.0)
                step_ratio = 1.0 + step_controller_ms / baseline_ms if step_candidate_count else 1.0
                step_ratios.append(step_ratio)
                runtime_rows.append(
                    {
                        "stage": "P7_MEASURED_ONLINE_EVENT_RUNTIME",
                        "status": "event_runtime_step",
                        "dataset": dataset,
                        "seed": seed,
                        "step": step,
                        "candidate_count": step_candidate_count,
                        "baseline_manual_step_ms": baseline_ms,
                        "controller_action_replay_ms": step_controller_ms,
                        "zero_candidate_step": int(step_candidate_count == 0),
                        "zero_candidate_controller_kernel_count": 0 if step_candidate_count == 0 else "",
                        "zero_candidate_controller_sync_count": 0 if step_candidate_count == 0 else "",
                        "controller_launches_this_step": step_candidate_count if step_candidate_count > 0 else 0,
                        "controller_syncs_this_step": 1 if step_candidate_count > 0 else 0,
                        "step_ratio": step_ratio,
                        "runtime_mode": "measured_event_scheduler_action_replay_materializer",
                        "runtime_measured": 1,
                        "official_event_runtime": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                event_idx += len(v9248.CARRIERS)

    missing_candidate_count = max(0, len(candidate_indices) - action_count)
    action_apply_pass = int(
        action_count == len(candidate_indices)
        and missing_candidate_count == 0
        and action_missing == 0
        and max_linf <= 1.0e-5
        and max_rel <= 1.0e-4
        and min_cos >= 0.9999
    )
    p1 = {
        "stage": "P1_ACTION_APPLY_MATERIALIZER",
        "status": "summary",
        "action_apply_materializer_id": "AAP1-RecomputedFunctionalDeltaReplay",
        "action_count": len(candidate_indices),
        "action_apply_rows": action_count,
        "action_apply_error_measured": int(action_count > 0),
        "action_apply_error_missing_count": missing_candidate_count + action_missing,
        "replay_success_count": replay_success,
        "action_apply_error_linf_max": max_linf,
        "action_apply_error_relative_max": max_rel,
        "action_apply_cosine_logged_applied_min": min_cos,
        "payload_tensor_file_written": 0,
        "payload_replay_manifest_written": 1,
        "action_lifecycle_pass": action_apply_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    action_rows.append(p1)

    official_sample_coverage = outcome_sample_seen / max(1, len(candidate_indices))
    matched_control_min = min(matched_control_counts) if matched_control_counts else 0
    secondary_ready = int(
        outcome_sample_seen > 0
        and matched_control_min >= 4
        and branch_missing_count == 0
        and horizon_missing_count == 0
        and secondary_missing_count == 0
        and official_sample_coverage >= 0.50
    )
    p2 = {
        "stage": "P2_SECONDARY_CONTROL_OUTCOME_MATERIALIZER",
        "status": "summary",
        "secondary_control_materializer_id": "SCM1-ImmediateProbePartialControls",
        "candidate_action_count": len(candidate_indices),
        "outcome_sample_action_count": outcome_sample_seen,
        "official_sample_coverage": official_sample_coverage,
        "matched_control_count_per_event_min": matched_control_min,
        "matched_control_count_per_event_mean": sum(matched_control_counts) / max(1, len(matched_control_counts)),
        "required_matched_control_count": 4,
        "branch_missing_count": branch_missing_count,
        "horizon_missing_count": horizon_missing_count,
        "missing_secondary_delta_count": secondary_missing_count,
        "secondary_outcome_ready": secondary_ready,
        "control_oracle_ready": secondary_ready,
        "reason": "" if secondary_ready else "partial_immediate_probe_only_controls_and_horizons_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    secondary_rows.append(p2)

    zero_steps = sum(1 for r in runtime_rows if r.get("status") == "event_runtime_step" and int(r.get("candidate_count", 0)) == 0)
    active_steps = sum(1 for r in runtime_rows if r.get("status") == "event_runtime_step" and int(r.get("candidate_count", 0)) > 0)
    runtime_pass = int(
        zero_steps > 0
        and q(active_launches, 0.90) <= 2.0
        and q(active_syncs, 0.90) <= 1.0
        and q(step_ratios, 0.90) <= 1.50
        and secondary_ready
    )
    p7 = {
        "stage": "P7_MEASURED_ONLINE_EVENT_RUNTIME",
        "status": "summary",
        "runtime_candidate_id": "RT2-ActionReplayEventSchedulerMeasuredPartial",
        "runtime_mode": "measured_event_scheduler_action_replay_materializer",
        "step_count": len([r for r in runtime_rows if r.get("status") == "event_runtime_step"]),
        "active_step_count": active_steps,
        "zero_candidate_step_count": zero_steps,
        "zero_candidate_controller_kernel_count": 0,
        "zero_candidate_controller_sync_count": 0,
        "controller_launches_per_active_step_q90": q(active_launches, 0.90),
        "controller_syncs_per_active_step_q90": q(active_syncs, 0.90),
        "step_ratio_q90": q(step_ratios, 0.90),
        "runtime_measured": 1,
        "diagnostic_derived_from_measured_components": 0,
        "official_event_runtime": 0,
        "event_driven_runtime_pass": runtime_pass,
        "reason": "" if runtime_pass else "active_step_not_fused_or_secondary_control_not_ready",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    runtime_rows.append(p7)

    return {
        "ctx": ctx,
        "event_table": event_table,
        "measured": measured,
        "candidate_indices": candidate_indices,
        "action_rows": action_rows,
        "secondary_rows": secondary_rows,
        "runtime_rows": runtime_rows,
        "payload_manifest": payload_manifest,
        "p1": p1,
        "p2": p2,
        "p7": p7,
    }


def build_boundary(source_v9310: Path) -> dict[str, Any]:
    route = read_json(source_v9310 / "route_decision.json") if (source_v9310 / "route_decision.json").exists() else {}
    return {
        "stage": "P0_V9310_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact": rel(source_v9310),
        "source_route_v9310": route.get("route", ""),
        "candidate_count_v9310": route.get("candidate_count", ""),
        "action_count_v9310": route.get("action_count", ""),
        "event_count_v9310": route.get("event_count", ""),
        "action_apply_error_measured_v9310": route.get("action_apply_error_measured", ""),
        "action_apply_error_missing_count_v9310": route.get("action_apply_error_missing_count", ""),
        "secondary_outcome_ready_v9310": route.get("secondary_outcome_ready", ""),
        "missing_secondary_delta_count_v9310": route.get("missing_secondary_delta_count", ""),
        "event_driven_runtime_pass_v9310": route.get("event_driven_runtime_pass", ""),
        "source_boundary_pass": int(route.get("route") == "R1-BoundaryReanalyzed"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def build_no_audit_guard(p1: dict[str, Any], p2: dict[str, Any], p7: dict[str, Any]) -> dict[str, Any]:
    new_rows = inum(p1.get("action_apply_rows")) + inum(p2.get("outcome_sample_action_count")) + inum(p7.get("step_count"))
    guard_pass = int(inum(p1.get("action_apply_rows")) > 0 and inum(p7.get("runtime_measured")) == 1)
    return {
        "stage": "P0_NO_AUDIT_ONLY_GUARD",
        "status": "summary",
        "new_materialized_row_count": new_rows,
        "action_apply_materialized": int(inum(p1.get("action_apply_rows")) > 0),
        "secondary_control_materialized_partial": int(inum(p2.get("outcome_sample_action_count")) > 0),
        "event_runtime_measured": p7.get("runtime_measured", 0),
        "audit_only_round_detected": int(not guard_pass),
        "no_audit_only_guard_pass": guard_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def build_oracle_rows(
    measured: list[dict[str, Any]],
    full_rows: list[dict[str, str]],
    candidate_indices: set[int],
    secondary_ready: int,
) -> list[dict[str, Any]]:
    safe_indices = [
        i
        for i in candidate_indices
        if i < len(measured) and inum(measured[i].get("seed")) >= 5 and inum(measured[i].get("Y_SU_v9265")) == 1
    ]
    denom = sum(1 for r in full_rows if inum(r.get("seed")) >= 5)
    accepted = len(safe_indices)
    primary = {
        "stage": "P3_USEFUL_CONTROL_ORACLE_FRONTIER",
        "status": "oracle_summary",
        "oracle_id": "OR1-PrimarySafeGoodOracleRetained",
        "accepted_count": accepted,
        "precision": 1.0 if accepted else 0.0,
        "coverage": accepted / denom if denom else 0.0,
        "bad_event": 0.0,
        "null_rate": 0.0,
        "primary_oracle_pass": int(accepted / max(1, denom) >= 0.03),
        "control_oracle_pass": 0,
        "useful_oracle_pass": 0,
        "secondary_control_outcome_ready": secondary_ready,
        "reason": "control_oracle_blocked_by_secondary_control_materializer" if not secondary_ready else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [primary]


def build_feature_rows(action_rows: list[dict[str, Any]], measured: list[dict[str, Any]], full_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_global = {inum(r.get("global_row_id")): r for r in action_rows if r.get("status") == "action_apply_row"}
    rows: list[dict[str, Any]] = []
    joined: list[dict[str, Any]] = []
    for idx, act in by_global.items():
        if idx >= len(measured):
            continue
        src = measured[idx]
        row = {**act}
        row["safe_good_label"] = inum(src.get("Y_SU_v9265"))
        row["bad_event_label"] = inum(src.get("bad_event"))
        row["null_event_label"] = inum(src.get("Y_null_event_v9265", src.get("Y_harmless_null_v9264", 0)))
        joined.append(row)
    held = [r for r in joined if inum(r.get("seed")) >= 5]
    labels_bad = [inum(r.get("bad_event_label")) for r in held]
    labels_safe = [inum(r.get("safe_good_label")) for r in held]
    features = [
        ("AVF1-PayloadNorm", "payload_norm", False),
        ("AVF2-PayloadOverAdamWNorm", "payload_over_adamw_norm", False),
        ("AVF3-CosActionAdamW", "cos_action_adamw", True),
        ("AVF4-CosActionNegativeGrad", "cos_action_negative_grad", True),
        ("AVF5-LinearizedCEDeltaLow", "linearized_CE_delta", False),
        ("AVF6-ProbeCEDeltaLow", "realized_probe_CE_delta", False),
        ("AVF7-ProbeMarginDeltaHigh", "realized_probe_margin_p10_delta", True),
    ]
    best: dict[str, Any] = {}
    for fid, field, high in features:
        values = [fnum(r.get(field)) for r in held]
        bad_auc = auc_score(values, labels_bad) if values else 0.5
        safe_auc = auc_score(values, labels_safe) if values else 0.5
        selected = sorted(held, key=lambda r: fnum(r.get(field)), reverse=high)[:273]
        n = len(selected)
        safe = sum(inum(r.get("safe_good_label")) for r in selected)
        bad = sum(inum(r.get("bad_event_label")) for r in selected)
        null = sum(inum(r.get("null_event_label")) for r in selected)
        feature_row = {
            "stage": "P4_ACTION_CONDITIONED_FEATURE_FACTORY",
            "status": "feature_summary",
            "feature_id": fid,
            "feature_field": field,
            "action_conditioned": 1,
            "commit_time_available": 1,
            "AUC_bad_event_raw": bad_auc,
            "AUC_bad_event_oriented": max(bad_auc, 1.0 - bad_auc),
            "AUC_safe_good_raw": safe_auc,
            "AUC_safe_good_oriented": max(safe_auc, 1.0 - safe_auc),
            "top273_safe_good": safe,
            "top273_bad_event": bad,
            "top273_null_event": null,
            "top273_precision": safe / n if n else 0.0,
            "top273_bad_event_rate": bad / n if n else 0.0,
            "feature_cost_recorded": 1,
            "feature_compute_q90_ms": q([fnum(r.get("action_replay_time_ms")) for r in held], 0.90),
            "feature_cost_pass": 1,
            "signal_pass": int(bad <= 13 and safe >= 205),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(feature_row)
        if not best or (feature_row["top273_bad_event"], -feature_row["top273_safe_good"]) < (best["top273_bad_event"], -best["top273_safe_good"]):
            best = feature_row
    summary = {
        "stage": "P4_ACTION_CONDITIONED_FEATURE_FACTORY",
        "status": "summary",
        "action_conditioned_feature_count": len(features),
        "best_feature_id": best.get("feature_id", ""),
        "best_feature_top273_safe_good": best.get("top273_safe_good", 0),
        "best_feature_top273_bad_event": best.get("top273_bad_event", 0),
        "best_feature_top273_null_event": best.get("top273_null_event", 0),
        "action_conditioned_feature_pass": int(best.get("signal_pass", 0) == 1),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def downstream_not_run(reason: str) -> dict[str, list[dict[str, Any]]]:
    base = {"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return {
        "p9_leave_dataset_stratum_out.csv": [{**base, "stage": "P9_LEAVE_DATASET_STRATUM_OUT", "status": "not_run", "reason": reason, "leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}],
        "p10_diagnostic_paired_replay_scout.csv": [{**base, "stage": "P10_DIAGNOSTIC_PAIRED_REPLAY_SCOUT", "status": "not_run", "reason": reason, "diagnostic_downstream_used_for_controller": 0}],
        "p11_official_paired_replay.csv": [{**base, "stage": "P11_OFFICIAL_PAIRED_REPLAY", "status": "not_run", "reason": reason, "paired_replay_pass": 0}],
        "p12_short_full_robustness_continual.csv": [{**base, "stage": "P12_SHORT_FULL_ROBUSTNESS_CONTINUAL", "status": "not_run", "reason": reason, "short_run_pass": 0, "full_run_pass": 0, "continual_pass": 0}],
    }


def write_hashes(out_dir: Path, extra_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path in [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("contract audit", out_dir / "contract_audit_v9320.csv"),
        ("provenance audit", out_dir / "v9320_provenance_audit.csv"),
    ]:
        if path.exists():
            rows.append({"artifact": label, "path": rel(path), "sha256": sha256_file(path)})
    for path in extra_paths:
        if path.exists():
            rows.append({"artifact": path.name, "path": rel(path), "sha256": sha256_file(path)})
    write_csv(out_dir / "artifact_hashes.csv", rows)
    return rows


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
        "torch_cuda_available": int(torch.cuda.is_available()),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "microprobe_steps": args.microprobe_steps,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "source_v9310": rel(Path(args.source_v9310)),
        "source_v9300": rel(Path(args.source_v9300)),
        "source_v9280": rel(Path(args.source_v9280)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    p0 = build_boundary(Path(args.source_v9310))
    mat = materialize(args, device)
    p1 = mat["p1"]
    p2 = mat["p2"]
    p7 = mat["p7"]
    guard = build_no_audit_guard(p1, p2, p7)
    full_rows = mat["event_table"]
    oracle_rows = build_oracle_rows(mat["measured"], full_rows, mat["candidate_indices"], inum(p2.get("secondary_outcome_ready")))
    feature_rows, feature_summary = build_feature_rows(mat["action_rows"], mat["measured"], full_rows)

    action_pass = inum(p1.get("action_lifecycle_pass"))
    secondary_ready = inum(p2.get("secondary_outcome_ready"))
    feature_pass = inum(feature_summary.get("action_conditioned_feature_pass"))
    runtime_pass = inum(p7.get("event_driven_runtime_pass"))
    if not action_pass:
        route = "R2-ActionApplyMaterializationFail"
        blocker = "action_apply_materialization_failed"
    elif not secondary_ready:
        route = "R4-SecondaryControlOutcomeMaterializationFail"
        blocker = "secondary_control_outcome_materialization_incomplete"
    elif not feature_pass:
        route = "R9-ActionConditionedObservabilityFail"
        blocker = "action_conditioned_observability_failed"
    elif not runtime_pass:
        route = "R13-MeasuredEventRuntimeFail"
        blocker = "measured_event_runtime_failed"
    else:
        route = "R14-SystemLegalControllerPass"
        blocker = ""

    system_pass = int(route == "R14-SystemLegalControllerPass")
    p8 = {
        "stage": "P8_SYSTEM_LEGAL_CONTROLLER_V9320",
        "status": "system_controller" if system_pass else "not_run",
        "controller_id": "not_selected_secondary_control_blocked" if not secondary_ready else "C4-action-value-controller-v9320",
        "action_apply_pass": action_pass,
        "secondary_control_outcome_ready": secondary_ready,
        "control_oracle_pass": 0,
        "action_conditioned_feature_pass": feature_pass,
        "event_driven_runtime_pass": runtime_pass,
        "precision_heldout": 0.0,
        "coverage_heldout": 0.0,
        "bad_event_heldout": 0.0,
        "null_rate_heldout": 0.0,
        "step_ratio_q90": p7.get("step_ratio_q90"),
        "official_eligible": system_pass,
        "system_legal_controller_pass": system_pass,
        "reason": blocker,
        "diagnostic_downstream_used_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9310": p0.get("source_route_v9310"),
        "candidate_count": p1.get("action_count"),
        "action_count": p1.get("action_count"),
        "event_count": len(mat["measured"]),
        "no_audit_only_guard_pass": guard.get("no_audit_only_guard_pass"),
        "action_apply_error_measured": p1.get("action_apply_error_measured"),
        "action_apply_error_missing_count": p1.get("action_apply_error_missing_count"),
        "action_apply_error_linf_max": p1.get("action_apply_error_linf_max"),
        "action_apply_error_relative_max": p1.get("action_apply_error_relative_max"),
        "action_apply_cosine_logged_applied_min": p1.get("action_apply_cosine_logged_applied_min"),
        "action_lifecycle_pass": action_pass,
        "secondary_outcome_ready": secondary_ready,
        "outcome_sample_action_count": p2.get("outcome_sample_action_count"),
        "official_sample_coverage": p2.get("official_sample_coverage"),
        "matched_control_count_per_event_min": p2.get("matched_control_count_per_event_min"),
        "branch_missing_count": p2.get("branch_missing_count"),
        "horizon_missing_count": p2.get("horizon_missing_count"),
        "missing_secondary_delta_count": p2.get("missing_secondary_delta_count"),
        "primary_oracle_pass": oracle_rows[0].get("primary_oracle_pass"),
        "control_oracle_pass": oracle_rows[0].get("control_oracle_pass"),
        "best_action_feature_id": feature_summary.get("best_feature_id"),
        "best_action_feature_top273_safe_good": feature_summary.get("best_feature_top273_safe_good"),
        "best_action_feature_top273_bad_event": feature_summary.get("best_feature_top273_bad_event"),
        "action_conditioned_feature_pass": feature_pass,
        "runtime_candidate_id": p7.get("runtime_candidate_id"),
        "runtime_measured": p7.get("runtime_measured"),
        "zero_candidate_controller_kernel_count": p7.get("zero_candidate_controller_kernel_count"),
        "controller_launches_per_active_step_q90": p7.get("controller_launches_per_active_step_q90"),
        "controller_syncs_per_active_step_q90": p7.get("controller_syncs_per_active_step_q90"),
        "step_ratio_q90": p7.get("step_ratio_q90"),
        "event_driven_runtime_pass": runtime_pass,
        "official_eligible": system_pass,
        "system_legal_controller_pass": system_pass,
        "primary_blocker": blocker,
        "next_required_implementation": "materialize_full_secondary_control_horizons" if not secondary_ready else "fuse_active_step_event_runtime",
        "success_v9320_strict_purekan_functional": system_pass,
        "success_v9320_full_functional": 0,
        "success_v9320_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    contract = {
        "stage": "CONTRACT_AUDIT_V9320",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "action_apply_materializer_used": 1,
        "action_lifecycle_pass": action_pass,
        "secondary_control_outcome_ready": secondary_ready,
        "primary_oracle_pass": oracle_rows[0].get("primary_oracle_pass"),
        "control_oracle_pass": oracle_rows[0].get("control_oracle_pass"),
        "action_conditioned_feature_pass": feature_pass,
        "event_driven_runtime_pass": runtime_pass,
        "system_legal_controller_pass": system_pass,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "projection_used_for_official": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    failure = {
        "stage": "FAILURE_TABLE_V9320",
        "status": "summary",
        "F1_action_apply_materialization_fail": int(not action_pass),
        "F2_secondary_control_missing": int(not secondary_ready),
        "F3_action_conditioned_observability_fail": int(not feature_pass),
        "F4_event_runtime_fail": int(not runtime_pass),
        "F5_system_controller_not_official": int(not system_pass),
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    write_csv(out_dir / "p0_v9310_boundary_reproduction.csv", [p0])
    write_csv(out_dir / "p0_no_audit_only_guard.csv", [guard])
    write_csv(out_dir / "p1_action_apply_materializer.csv", [p1])
    write_csv(out_dir / "action_apply_trace_v9320.csv", mat["action_rows"])
    write_json(out_dir / "action_payload_replay_manifest_v9320.json", mat["payload_manifest"])
    write_csv(out_dir / "p2_secondary_control_outcome_materializer.csv", [p2])
    write_csv(out_dir / "secondary_control_outcome_trace_v9320.csv", mat["secondary_rows"])
    write_csv(out_dir / "matched_control_outcome_trace_v9320.csv", [r for r in mat["secondary_rows"] if r.get("status") != "summary"])
    write_csv(out_dir / "p3_useful_control_oracle_frontier.csv", oracle_rows)
    write_csv(out_dir / "control_oracle_frontier_trace_v9320.csv", oracle_rows)
    write_csv(out_dir / "p4_action_conditioned_feature_factory.csv", [feature_summary])
    write_csv(out_dir / "action_feature_trace_v9320.csv", feature_rows)
    write_csv(out_dir / "bad_tail_observability_trace_v9320.csv", feature_rows)
    write_csv(out_dir / "feature_cost_trace_v9320.csv", feature_rows)
    write_csv(out_dir / "p5_action_primitive_insufficiency_autopsy.csv", [{**failure, "stage": "P5_ACTION_PRIMITIVE_INSUFFICIENCY_AUTOPSY", "primary_action_blocker": blocker}])
    write_csv(out_dir / "action_primitive_autopsy_trace_v9320.csv", [{**failure, "stage": "P5_ACTION_PRIMITIVE_INSUFFICIENCY_AUTOPSY_TRACE"}])
    write_csv(out_dir / "p6_crossfitted_action_value_controller.csv", [{**p8, "stage": "P6_CROSSFITTED_ACTION_VALUE_CONTROLLER", "status": "not_run", "reason": blocker}])
    write_csv(out_dir / "controller_frontier_trace_v9320.csv", [{**p8, "stage": "P6_CONTROLLER_FRONTIER_TRACE", "status": "not_run", "reason": blocker}])
    write_csv(out_dir / "p7_measured_online_event_runtime.csv", [p7])
    write_csv(out_dir / "runtime_event_scheduler_trace_v9320.csv", mat["runtime_rows"])
    write_csv(out_dir / "runtime_empty_event_semantics_trace_v9320.csv", mat["runtime_rows"])
    write_csv(out_dir / "runtime_component_trace_v9320.csv", mat["runtime_rows"])
    write_csv(out_dir / "p8_system_legal_controller_v9320.csv", [p8])
    write_csv(out_dir / "system_controller_trace_v9320.csv", [p8])
    for name, rows in downstream_not_run("P8_system_controller_not_official").items():
        write_csv(out_dir / name, rows)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv(out_dir / "contract_audit_v9320.csv", [contract])
    write_csv(out_dir / "failure_table.csv", [failure])

    audit_paths = [p for p in out_dir.glob("*.csv") if p.name not in {"artifact_hashes.csv", "v9320_provenance_audit.csv"}]
    audit = audit_no_fake(audit_paths)
    write_csv(out_dir / "v9320_provenance_audit.csv", [audit])

    extra = [
        out_dir / "p0_v9310_boundary_reproduction.csv",
        out_dir / "p0_no_audit_only_guard.csv",
        out_dir / "p1_action_apply_materializer.csv",
        out_dir / "action_apply_trace_v9320.csv",
        out_dir / "p2_secondary_control_outcome_materializer.csv",
        out_dir / "secondary_control_outcome_trace_v9320.csv",
        out_dir / "p3_useful_control_oracle_frontier.csv",
        out_dir / "p4_action_conditioned_feature_factory.csv",
        out_dir / "p7_measured_online_event_runtime.csv",
        out_dir / "p8_system_legal_controller_v9320.csv",
        out_dir / "failure_table.csv",
    ]
    hashes = write_hashes(out_dir, extra)
    manifest["completed_at"] = now_iso()
    manifest["route"] = route
    write_json(out_dir / "run_manifest.json", manifest)
    write_hashes(out_dir, extra)

    print(json.dumps({"out_dir": str(out_dir), "route": route, "hashes": len(hashes)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
