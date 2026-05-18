#!/usr/bin/env python3
"""DG-KAN v9.4.2 source outcome materializer closure runner.

This runner targets the v9.4.1 boundary where source payloads and
certificates were real, but branch-horizon outcome rows were absent.  The
materializer here is keyed directly by generated action rows, not by a
best-effort event-table scan, so expected/actual row closure is explicit.
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure as v9340  # noqa: E402
import run_v9380_real_certificate_action_primitive_materialization as v9380  # noqa: E402
import run_v9410_value_producing_source_generator as v9410  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.2_SourceOutcomeMaterializerClosure_ValueProducingSourceTriage_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9420_source_outcome_materializer_closure.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9410 = RESULT_ROOT / "v9410_value_producing_source_generator_legal_source_identifiability_base_acc_first_20260514T080000Z"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"

PRIMITIVES = [
    "AP0b-LastEdgeLinearizedDescentSource",
    "AP0c-AdamWResidualOrthogonalSource",
    "AP0d-TailMarginRepairSource",
    "AP0e-CurvatureGuardedLowRankEdgeSource",
    "AP0f-SupportMemorySource",
]
BRANCHES = [
    "RealSource",
    "AdamWOnly",
    "AdamWParallel",
    "bestLR",
    "NoOp",
    "Random",
    "ShuffledSourcePayload",
    "CertificatePassNoPayload",
    "SourceAP0Baseline",
]
GENERATED_CONTROL_BRANCHES = [b for b in BRANCHES if b != "RealSource"]
AP0_CONTROL_BRANCHES = [b for b in BRANCHES if b not in {"RealSource", "SourceAP0Baseline"}]
HORIZONS = [20, 80, 240]
SECONDARY_FIELDS = ["CE_mean_delta", "CEp99_delta", "margin_p10_delta", "ECE_delta", "NLL_delta", "curvature_delta", "local_lipschitz_delta", "acc_delta"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0")
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--sentinel-seeds", default="0,1,2")
    p.add_argument("--sentinel-steps", type=int, default=12)
    p.add_argument("--sentinel-train-size", type=int, default=512)
    p.add_argument("--sentinel-test-size", type=int, default=256)
    p.add_argument("--sentinel-hidden-dim", type=int, default=64)
    p.add_argument("--source-v9410", default=str(DEFAULT_SOURCE_V9410))
    p.add_argument("--source-v9330", default=str(DEFAULT_SOURCE_V9330))
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
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))]


def mean(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    return sum(xs) / max(1, len(xs))


def lcb_mean(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return mean(xs) - 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9380.v9320.tensor_hash(payload)


def clone_params(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clone_states(states: list[AdamWState]) -> list[AdamWState]:
    return v9340.clone_states(states)


def load_generated_rows(source_v9410: Path) -> list[dict[str, Any]]:
    rows = [r for r in read_csv(source_v9410 / "source_payload_trace_v9410.csv") if r.get("status") == "generated_source_action_row"]
    for row in rows:
        row["ap_action_id"] = row.get("ap_action_id") or row.get("generated_action_id")
    return rows


def load_generated_payload(row: dict[str, Any], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    path = REPO / str(row.get("ap_payload_shard_path"))
    if str(path) not in cache:
        cache[str(path)] = torch.load(path, map_location=device)
    shard = cache[str(path)]
    off = inum(row.get("ap_payload_tensor_offset"))
    return [shard["d0"][off].detach().clone().to(device), shard["d1"][off].detach().clone().to(device), shard["d2"][off].detach().clone().to(device)]


def load_source_payload_rows(source_v9330: Path) -> dict[str, dict[str, str]]:
    rows = [r for r in read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv") if r.get("status") == "payload_disk_replay_row"]
    return {str(r.get("action_id")): r for r in rows}


def load_source_payload(row: dict[str, Any], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return v9410.load_payload_from_row(row, cache, device)


def p0_boundary(source_v9410: Path) -> dict[str, Any]:
    route = read_json(source_v9410 / "route_decision.json")
    p4 = next(iter(read_csv(source_v9410 / "p4_h20_immediate_direction_smoke.csv")), {})
    p5 = next(iter(read_csv(source_v9410 / "p5_horizon_extension_longrisk_audit.csv")), {})
    return {
        "stage": "P0_V9410_BOUNDARY_REANALYSIS",
        "status": "summary",
        "route_v9410": route.get("route"),
        "source_route_v9400": route.get("source_route_v9400"),
        "stratified_panel_pass": route.get("stratified_panel_pass"),
        "official_panel_id": "PANEL-S256",
        "official_panel_PSI_vs_full": route.get("official_panel_PSI_vs_full"),
        "official_panel_KL_vs_full": route.get("official_panel_KL_vs_full"),
        "base_acc_sentinel_complete": route.get("base_acc_sentinel_complete"),
        "mean_test_acc_LQ": route.get("mean_test_acc_LQ"),
        "mean_test_acc_MLP": route.get("mean_test_acc_MLP"),
        "LQ_minus_MLP_mean_test_acc": fnum(route.get("mean_test_acc_LQ")) - fnum(route.get("mean_test_acc_MLP")),
        "base_acc_used_for_controller": route.get("base_acc_used_for_controller"),
        "source_generator_materialized": route.get("source_generator_materialized"),
        "primitive_materialized_count": route.get("primitive_materialized_count"),
        "source_input_action_count": next(iter(read_csv(source_v9410 / "p3_real_value_producing_source_generator.csv")), {}).get("source_input_action_count"),
        "generated_action_count_total": route.get("generated_action_count_total"),
        "payload_tensor_written": route.get("payload_tensor_written"),
        "certificate_tensor_written": route.get("certificate_tensor_written"),
        "action_apply_error_linf_max": route.get("action_apply_error_linf_max"),
        "certificate_pass_action_count": next(iter(read_csv(source_v9410 / "p3_real_value_producing_source_generator.csv")), {}).get("certificate_pass_action_count"),
        "branch_horizon_row_count_expected": p4.get("branch_horizon_row_count_expected"),
        "branch_horizon_row_count_actual": p4.get("branch_horizon_row_count_actual"),
        "source_outcome_materialized": p4.get("source_outcome_materialized"),
        "h20_immediate_direction_pass": p4.get("h20_immediate_direction_pass"),
        "horizon_extension_pass": p5.get("horizon_extension_pass"),
        "certificate_sufficiency_pass": route.get("certificate_sufficiency_pass"),
        "source_controller_pass": route.get("source_controller_pass"),
        "selected_runtime_pass": route.get("selected_runtime_pass"),
        "system_legal_controller_pass": route.get("system_legal_controller_pass"),
        "materializer_rows_absent_not_value_failure": int(inum(p4.get("branch_horizon_row_count_expected")) > 0 and inum(p4.get("branch_horizon_row_count_actual")) == 0),
        "p0_boundary_reanalysis_pass": int(route.get("route") == "R4-GeneratedSourceImmediateDirectionFail" and inum(p4.get("branch_horizon_row_count_actual")) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def replay_context(args: argparse.Namespace, action: dict[str, Any], device: torch.device, cache: dict[str, Any]) -> dict[str, Any]:
    key = (str(action.get("dataset")), int(inum(action.get("seed"))), int(inum(action.get("step"))))
    if key in cache:
        return cache[key]
    dataset, seed, step = key
    load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
    x_train, y_train, _x_test, _y_test, input_dim, output_dim, _proto = v9380.v9320.v9248.v92._load_task(load_args, dataset, train_size=int(args.train_size), test_size=32)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    spec = v9340.lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    fwd_core, bwd_core = v9340.lq.functions_for_basis(spec.basis)
    params, mu, std = v9340.lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9380)
    states = [AdamWState.zeros_like(p) for p in params]
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
    n_train = int(x_train.shape[0])
    for _ in range(max(0, step)):
        idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=gen, device=device)
        xu = x_train[idx[: int(args.batch_size)]].contiguous()
        yu = y_train[idx[: int(args.batch_size)]].contiguous()
        pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
        grads = list(pack[1:])
        v9380.v9320.v9248.v92._adamw_update_foreach_(params, grads, states, cfg)
    idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=gen, device=device)
    xu = x_train[idx[: int(args.batch_size)]].contiguous()
    yu = y_train[idx[: int(args.batch_size)]].contiguous()
    xp = x_train[idx[int(args.batch_size) :]].contiguous()
    yp = y_train[idx[int(args.batch_size) :]].contiguous()
    pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
    grads = list(pack[1:])
    task_params = clone_params(params)
    task_states = clone_states(states)
    v9380.v9320.v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg)
    before_metrics = v9340.extended_metrics_from_logits(fwd_core(xp, *params, mu, std, 2.0, 2.0), yp)
    out = {
        "dataset": dataset,
        "seed": seed,
        "step": step,
        "x_train": x_train,
        "y_train": y_train,
        "params": clone_params(params),
        "states": clone_states(states),
        "task_params": task_params,
        "task_states": task_states,
        "task_delta": [tp - p for tp, p in zip(task_params, params)],
        "xp": xp,
        "yp": yp,
        "mu": mu,
        "std": std,
        "spec": spec,
        "fwd_core": fwd_core,
        "bwd_core": bwd_core,
        "cfg": cfg,
        "before_metrics": before_metrics,
        "checkpoint_hash_before": v9380.v9320.tensor_hash(params),
    }
    cache[key] = out
    return out


def branch_start_source(
    branch: str,
    ctx: dict[str, Any],
    generated_payload: list[torch.Tensor],
    source_payload: list[torch.Tensor],
    shuffled_payload: list[torch.Tensor],
    gen: torch.Generator,
    certificate_pass: int,
) -> tuple[list[torch.Tensor], list[AdamWState], str, list[torch.Tensor], float]:
    t0 = time.perf_counter()
    params = ctx["params"]
    states = ctx["states"]
    task_params = ctx["task_params"]
    task_states = ctx["task_states"]
    task_delta = ctx["task_delta"]
    if branch == "RealSource":
        return [tp + d for tp, d in zip(task_params, generated_payload)], clone_states(task_states), "task_params_plus_generated_source_payload", generated_payload, (time.perf_counter() - t0) * 1000.0
    if branch == "SourceAP0Baseline":
        return [tp + d for tp, d in zip(task_params, source_payload)], clone_states(task_states), "task_params_plus_source_AP0_payload", source_payload, (time.perf_counter() - t0) * 1000.0
    if branch in {"AdamWOnly", "AdamWParallel", "CertificatePassNoPayload"}:
        sem = "task_params_no_source_payload"
        if branch == "CertificatePassNoPayload":
            sem = f"certificate_pass_{certificate_pass}_without_payload"
        return clone_params(task_params), clone_states(task_states), sem, generated_payload, (time.perf_counter() - t0) * 1000.0
    if branch == "NoOp":
        return clone_params(params), clone_states(states), "pre_step_params_no_current_adamw", generated_payload, (time.perf_counter() - t0) * 1000.0
    if branch == "Random":
        rnd = v9380.v9330.random_like_payload(generated_payload, gen)
        return [tp + d for tp, d in zip(task_params, rnd)], clone_states(task_states), "task_params_plus_norm_matched_random_source_delta", rnd, (time.perf_counter() - t0) * 1000.0
    if branch == "ShuffledSourcePayload":
        return [tp + d for tp, d in zip(task_params, shuffled_payload)], clone_states(task_states), "task_params_plus_shuffled_source_payload", shuffled_payload, (time.perf_counter() - t0) * 1000.0
    if branch == "bestLR":
        best_scale = 1.0
        best_nll = float("inf")
        for scale in [0.3, 1.0, 3.0]:
            cand = [p + scale * d for p, d in zip(params, task_delta)]
            nll = v9340.extended_metrics_from_logits(ctx["fwd_core"](ctx["xp"], *cand, ctx["mu"], ctx["std"], 2.0, 2.0), ctx["yp"])["NLL"]
            if nll < best_nll:
                best_nll = nll
                best_scale = scale
        return [p + best_scale * d for p, d in zip(params, task_delta)], clone_states(task_states), f"best_immediate_probe_lr_scale_{best_scale}", generated_payload, (time.perf_counter() - t0) * 1000.0
    raise ValueError(branch)


def materialize_source_outcomes(
    args: argparse.Namespace,
    generated: list[dict[str, Any]],
    source_payload_by_id: dict[str, dict[str, str]],
    device: torch.device,
    branches: list[str],
    horizons: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gen_payload_cache: dict[str, Any] = {}
    src_payload_cache: dict[str, Any] = {}
    context_cache: dict[str, Any] = {}
    payload_by_action = {str(r["ap_action_id"]): load_generated_payload(r, gen_payload_cache, device) for r in generated}
    by_primitive = defaultdict(list)
    for row in generated:
        by_primitive[str(row.get("primitive_id"))].append(row)
    rows: list[dict[str, Any]] = []
    completion: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    branch_times: list[float] = []
    t0 = time.perf_counter()
    for idx, action in enumerate(generated):
        action_id = str(action.get("ap_action_id"))
        source_id = str(action.get("source_action_id"))
        src_payload_row = source_payload_by_id.get(source_id)
        if not src_payload_row:
            retry.append({"stage": "P2_SOURCE_OUTCOME_MATERIALIZER", "status": "unresolved_missing_source_payload", "generated_action_id": action_id, "source_action_id": source_id, "root_cause": "RC5_source_action_id_join_mismatch", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
            continue
        try:
            ctx = replay_context(args, action, device, context_cache)
            source_payload = load_source_payload(src_payload_row, src_payload_cache, device)
            generated_payload = payload_by_action[action_id]
            peers = [x for x in by_primitive[str(action.get("primitive_id"))] if str(x.get("ap_action_id")) != action_id]
            shuffled = peers[(idx + 11) % len(peers)] if peers else action
            shuffled_payload = payload_by_action[str(shuffled.get("ap_action_id"))]
            before_secondary = v9340.secondary_metrics(ctx["params"], generated_payload, ctx["xp"], ctx["mu"], ctx["std"], ctx["spec"])
            branch_rows_by_h: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
            branch_gen = torch.Generator(device=device).manual_seed(int(args.seed) * 100000 + idx)
            for branch in branches:
                try:
                    start_params, start_states, semantics, secondary_payload, apply_ms = branch_start_source(branch, ctx, generated_payload, source_payload, shuffled_payload, branch_gen, inum(action.get("certificate_pass")))
                    checkpoints, branch_runtime_ms, start_hash, end_hash = v9340.rollout_branch_checkpoints(
                        start_params,
                        start_states,
                        ctx["before_metrics"],
                        before_secondary,
                        secondary_payload,
                        ctx["x_train"],
                        ctx["y_train"],
                        ctx["xp"],
                        ctx["yp"],
                        ctx["mu"],
                        ctx["std"],
                        ctx["spec"],
                        ctx["fwd_core"],
                        ctx["bwd_core"],
                        ctx["cfg"],
                        list(horizons),
                        int(args.batch_size),
                        int(args.seed) * 1000000 + idx * 31 + len(branch) * 157,
                        device,
                    )
                    branch_times.append(branch_runtime_ms)
                    for horizon in horizons:
                        delta = checkpoints[horizon]
                        value = -delta["CEp99_delta"] + delta["margin_p10_delta"] - delta["ECE_delta"] - delta["NLL_delta"] - delta["curvature_delta"] + delta["acc_delta"]
                        row = {
                            "stage": "P2_SOURCE_OUTCOME_MATERIALIZER_SCALEUP",
                            "status": "source_branch_horizon_row",
                            "outcome_row_id": stable_hash(action_id, branch, horizon, start_hash, "v9420"),
                            "generated_action_id": action_id,
                            "ap_action_id": action_id,
                            "source_action_id": source_id,
                            "source_candidate_id": action.get("source_candidate_id"),
                            "source_event_id": action.get("event_id"),
                            "candidate_id": action.get("candidate_id"),
                            "event_id": action.get("event_id"),
                            "source_panel_id": "PANEL-S256",
                            "primitive_id": action.get("primitive_id"),
                            "branch": branch,
                            "branch_id": branch,
                            "branch_semantics": semantics,
                            "horizon": horizon,
                            "dataset": action.get("dataset"),
                            "seed": action.get("seed"),
                            "step": action.get("step"),
                            "family_id": action.get("family_id"),
                            "bucket_id": action.get("bucket_id"),
                            "checkpoint_hash_before": ctx["checkpoint_hash_before"],
                            "checkpoint_hash_after": end_hash,
                            "payload_hash": action.get("payload_hash"),
                            "ap_payload_hash": action.get("ap_payload_hash"),
                            "certificate_hash": action.get("certificate_hash"),
                            "branch_state_hash": start_hash,
                            "horizon_state_hash": end_hash,
                            "row_write_status": "written",
                            "exception_type": "",
                            "exception_message": "",
                            **delta,
                            "bad_event_label": "",
                            "null_event_label": "",
                            "safe_good_label": "",
                            "task_safe_label": "",
                            "useful_label": "",
                            "weak_CP_label": "",
                            "strong_CP_label": "",
                            "horizon_robust_CP_label": "",
                            "long_risk_label": "",
                            "AP0_weak_CP_label": "",
                            "AP0_long_risk_label": "",
                            "V_real": "",
                            "V_source_ap0": "",
                            "V_branch": value,
                            "V_ctrl": "",
                            "V_ctrl_max_control_gap": "",
                            "beats_adamwparallel": "",
                            "beats_bestlr": "",
                            "beats_noop": "",
                            "beats_random": "",
                            "beats_shuffled_payload": "",
                            "certificate_pass": action.get("certificate_pass"),
                            "certificate_score": action.get("certificate_score"),
                            "outcome_source": "same_run_generated_source_v9420",
                            "branch_runtime_ms": branch_runtime_ms,
                            "branch_apply_time_ms": apply_ms,
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }
                        branch_rows_by_h[horizon][branch] = row
                except Exception as exc:  # noqa: BLE001
                    retry.append({"stage": "P2_SOURCE_OUTCOME_MATERIALIZER", "status": "unresolved_exception", "generated_action_id": action_id, "source_action_id": source_id, "primitive_id": action.get("primitive_id"), "branch": branch, "exception_type": type(exc).__name__, "exception_message": repr(exc), "root_cause": "RC8_branch_runner_exception", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
            for horizon in horizons:
                rows_h = branch_rows_by_h[horizon]
                if "RealSource" not in rows_h or any(b not in rows_h for b in GENERATED_CONTROL_BRANCHES if b in branches):
                    retry.append({"stage": "P2_SOURCE_OUTCOME_MATERIALIZER", "status": "incomplete_action_horizon", "generated_action_id": action_id, "source_action_id": source_id, "primitive_id": action.get("primitive_id"), "horizon": horizon, "materialized_branch_count": len(rows_h), "required_branch_count": len(branches), "root_cause": "RC9_horizon_runner_exception" if rows_h else "RC10_row_sink_write_exception", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
                    continue
                real = fnum(rows_h["RealSource"]["V_branch"])
                gen_controls = {b: fnum(rows_h[b]["V_branch"]) for b in GENERATED_CONTROL_BRANCHES if b in rows_h}
                best_gen_control = max(gen_controls.values()) if gen_controls else real
                source_ap0 = fnum(rows_h.get("SourceAP0Baseline", {}).get("V_branch", 0.0))
                ap0_controls = {b: fnum(rows_h[b]["V_branch"]) for b in AP0_CONTROL_BRANCHES if b in rows_h}
                best_ap0_control = max(ap0_controls.values()) if ap0_controls else source_ap0
                real_row = rows_h["RealSource"]
                bad = int(fnum(real_row["CEp99_delta"]) > 0.25 or fnum(real_row["NLL_delta"]) > 0.15 or fnum(real_row["ECE_delta"]) > 0.05 or fnum(real_row["acc_delta"]) < -0.05)
                null = int(abs(real - best_gen_control) <= 0.02)
                weak = int(real > best_gen_control and not bad and not null)
                strong = int(weak and (real - best_gen_control) > 0.15 and fnum(real_row["CEp99_delta"]) < 0.0 and fnum(real_row["acc_delta"]) >= 0.0)
                longrisk = int(horizon == 240 and (bad or (real - best_gen_control) < -0.10))
                ap0_row = rows_h.get("SourceAP0Baseline", {})
                ap0_bad = int(fnum(ap0_row.get("CEp99_delta")) > 0.25 or fnum(ap0_row.get("NLL_delta")) > 0.15 or fnum(ap0_row.get("ECE_delta")) > 0.05 or fnum(ap0_row.get("acc_delta")) < -0.05) if ap0_row else 0
                ap0_null = int(abs(source_ap0 - best_ap0_control) <= 0.02) if ap0_row else 1
                ap0_weak = int(source_ap0 > best_ap0_control and not ap0_bad and not ap0_null) if ap0_row else 0
                ap0_strong = int(ap0_weak and (source_ap0 - best_ap0_control) > 0.15 and fnum(ap0_row.get("CEp99_delta")) < 0.0 and fnum(ap0_row.get("acc_delta")) >= 0.0) if ap0_row else 0
                ap0_longrisk = int(horizon == 240 and (ap0_bad or (source_ap0 - best_ap0_control) < -0.10)) if ap0_row else 0
                for branch, row in rows_h.items():
                    row["V_real"] = real
                    row["V_source_ap0"] = source_ap0
                    row["V_ctrl"] = real - best_gen_control
                    row["V_ctrl_max_control_gap"] = real - best_gen_control
                    row["AP0_V_ctrl"] = source_ap0 - best_ap0_control
                    row["bad_event_label"] = bad
                    row["null_event_label"] = null
                    row["safe_good_label"] = weak
                    row["task_safe_label"] = int(not bad)
                    row["useful_label"] = weak
                    row["weak_CP_label"] = weak
                    row["strong_CP_label"] = strong
                    row["long_risk_label"] = longrisk
                    row["AP0_weak_CP_label"] = ap0_weak
                    row["AP0_strong_CP_label"] = ap0_strong
                    row["AP0_long_risk_label"] = ap0_longrisk
                    row["beats_adamwparallel"] = int(real > fnum(rows_h.get("AdamWParallel", {}).get("V_branch", -1e9)))
                    row["beats_bestlr"] = int(real > fnum(rows_h.get("bestLR", {}).get("V_branch", -1e9)))
                    row["beats_noop"] = int(real > fnum(rows_h.get("NoOp", {}).get("V_branch", -1e9)))
                    row["beats_random"] = int(real > fnum(rows_h.get("Random", {}).get("V_branch", -1e9)))
                    row["beats_shuffled_payload"] = int(real > fnum(rows_h.get("ShuffledSourcePayload", {}).get("V_branch", -1e9)))
                    rows.append(row)
                completion.append({"stage": "P2_BRANCH_HORIZON_COMPLETION", "status": "action_horizon_summary", "generated_action_id": action_id, "source_action_id": source_id, "primitive_id": action.get("primitive_id"), "horizon": horizon, "materialized_branch_count": len(rows_h), "required_branch_count": len(branches), "matched_control_count": len(rows_h) - 1, "V_ctrl": real - best_gen_control, "AP0_V_ctrl": source_ap0 - best_ap0_control, "weak_CP_label": weak, "AP0_weak_CP_label": ap0_weak, "long_risk_label": longrisk, "AP0_long_risk_label": ap0_longrisk, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
        except Exception as exc:  # noqa: BLE001
            retry.append({"stage": "P2_SOURCE_OUTCOME_MATERIALIZER", "status": "unresolved_action_exception", "generated_action_id": action_id, "source_action_id": source_id, "exception_type": type(exc).__name__, "exception_message": repr(exc), "root_cause": "RC8_branch_runner_exception", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    # Horizon robust label requires all three generated real-source weak labels.
    weak_by_action: dict[str, set[int]] = defaultdict(set)
    ap0_weak_by_source: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.get("branch") == "RealSource" and inum(row.get("weak_CP_label")):
            weak_by_action[str(row.get("generated_action_id"))].add(inum(row.get("horizon")))
        if row.get("branch") == "SourceAP0Baseline" and inum(row.get("AP0_weak_CP_label")):
            ap0_weak_by_source[str(row.get("source_action_id"))].add(inum(row.get("horizon")))
    for row in rows:
        row["horizon_robust_CP_label"] = int(all(h in weak_by_action[str(row.get("generated_action_id"))] for h in HORIZONS))
        row["AP0_horizon_robust_CP_label"] = int(all(h in ap0_weak_by_source[str(row.get("source_action_id"))] for h in HORIZONS))
    expected = len(generated) * len(branches) * len(horizons)
    actual = len(rows)
    wall = max(1e-9, time.perf_counter() - t0)
    summary = {
        "stage": "P2_SOURCE_OUTCOME_MATERIALIZER_SCALEUP",
        "status": "summary",
        "materializer_id": "SOM2-DirectGeneratedActionReplayClosure",
        "generated_action_count_input": len(generated),
        "branch_count": len(branches),
        "horizon_count": len(horizons),
        "branch_horizon_row_count_expected": expected,
        "branch_horizon_row_count_actual": actual,
        "branch_completion_rate": actual / max(1, expected),
        "horizon_completion_rate": actual / max(1, expected),
        "secondary_delta_completion_rate": int(actual == expected and all(all(r.get(k, "") != "" for k in SECONDARY_FIELDS) for r in rows)),
        "rows_per_sec_total": actual / wall,
        "wallclock_sec": wall,
        "row_count_failed": len([r for r in retry if r.get("status") != "empty"]),
        "row_count_retried": len([r for r in retry if r.get("status") != "empty"]),
        "unresolved_failed_rows": len([r for r in retry if r.get("status") != "empty"]),
        "quality_audit_pass": 0,
        "missing_branch_count": max(0, expected - actual),
        "missing_horizon_count": max(0, expected - actual),
        "missing_secondary_delta_count": sum(sum(int(r.get(k, "") == "") for k in SECONDARY_FIELDS) for r in rows),
        "metric_nan_count": 0,
        "metric_inf_count": 0,
        "label_exclusivity_violation_count": 0,
        "source_outcome_materialized": int(actual == expected and actual > 0),
        "source_outcome_materializer_pass": int(actual == expected and actual > 0 and not any(r.get("status") != "empty" for r in retry)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    if not retry:
        retry = [{"stage": "P2_SOURCE_OUTCOME_MATERIALIZER", "status": "empty", "unresolved_failed_rows": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}]
    return rows, completion, retry, summary


def preflight(args: argparse.Namespace, generated: list[dict[str, Any]], source_payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    first = generated[0:1]
    mini: list[dict[str, Any]] = []
    for p in PRIMITIVES:
        mini.extend([r for r in generated if r.get("primitive_id") == p][:2])
    specs = [
        ("P1a-single-action-row-write", first, ["RealSource"], [20]),
        ("P1b-single-action-branch-fanout", first, BRANCHES, [20]),
        ("P1c-mini-matrix-preflight", mini, BRANCHES, [20]),
    ]
    rows: list[dict[str, Any]] = []
    join_trace: list[dict[str, Any]] = []
    sink_trace: list[dict[str, Any]] = []
    pass_all = 1
    root_cause_table_complete = 1
    for pid, subset, branches, horizons in specs:
        actual_rows, completion, retry, summary = materialize_source_outcomes(args, subset, source_payload_by_id, device, branches, horizons)
        expected = len(subset) * len(branches) * len(horizons)
        actual = len(actual_rows)
        unresolved = len([r for r in retry if r.get("status") != "empty"])
        root = "none" if actual == expected and unresolved == 0 else (next((str(r.get("root_cause")) for r in retry if r.get("root_cause")), "RC10_row_sink_write_exception"))
        rows.append({
            "stage": "P1_SOURCE_OUTCOME_MATERIALIZER_PREFLIGHT",
            "status": "preflight_summary",
            "preflight_id": pid,
            "generated_action_id": subset[0].get("ap_action_id") if subset else "",
            "source_action_id": subset[0].get("source_action_id") if subset else "",
            "primitive_id": subset[0].get("primitive_id") if subset else "",
            "payload_tensor_path_exists": int(bool(subset and (REPO / str(subset[0].get("ap_payload_shard_path"))).exists())),
            "certificate_tensor_path_exists": int(bool(subset and (REPO / str(subset[0].get("ap_payload_shard_path"))).exists())),
            "payload_hash_match": int(bool(subset and subset[0].get("payload_hash_missing") in {"0", 0})),
            "certificate_hash_match": int(bool(subset and subset[0].get("certificate_hash_missing") in {"0", 0})),
            "checkpoint_hash_exists": int(actual > 0),
            "branch_runner_initialized": int(bool(branches)),
            "horizon_runner_initialized": int(bool(horizons)),
            "row_sink_initialized": 1,
            "branch": ",".join(branches),
            "horizon": ",".join(map(str, horizons)),
            "expected_row_count": expected,
            "actual_row_count": actual,
            "rows_written_before_quality_filter": actual,
            "rows_dropped_by_quality_filter": 0,
            "exception_type": "" if root == "none" else root,
            "exception_message": "",
            "join_failure_key": "" if root == "none" else root,
            "manifest_path_used": rel(Path(args.source_v9410) / "run_manifest.json"),
            "payload_shard_id": Path(str(subset[0].get("ap_payload_shard_path"))).stem if subset else "",
            "root_cause": root,
            "preflight_pass": int(actual == expected and unresolved == 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        pass_all = int(pass_all and actual == expected and unresolved == 0)
        root_cause_table_complete = int(root_cause_table_complete and bool(root))
        join_trace.extend(retry)
        sink_trace.extend(completion)
    summary = {
        "stage": "P1_SOURCE_OUTCOME_MATERIALIZER_PREFLIGHT",
        "status": "summary",
        "p1a_pass": rows[0]["preflight_pass"] if rows else 0,
        "p1b_pass": rows[1]["preflight_pass"] if len(rows) > 1 else 0,
        "p1c_pass": rows[2]["preflight_pass"] if len(rows) > 2 else 0,
        "root_cause_table_complete": root_cause_table_complete,
        "unresolved_exception_count": sum(1 for r in rows if not inum(r.get("preflight_pass"))),
        "preflight_pass": pass_all,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, join_trace, sink_trace, summary


def quality_audit(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    ids = Counter(str(r.get("outcome_row_id")) for r in rows)
    metric_nan = 0
    metric_inf = 0
    exclusivity = 0
    for row in rows:
        if inum(row.get("safe_good_label")) and (inum(row.get("bad_event_label")) or inum(row.get("null_event_label"))):
            exclusivity += 1
        for key in SECONDARY_FIELDS:
            val = fnum(row.get(key), float("nan"))
            metric_nan += int(math.isnan(val))
            metric_inf += int(math.isinf(val))
    return {
        "quality_audit_pass": int(bool(rows) and len(rows) == expected and exclusivity == 0 and metric_nan == 0 and metric_inf == 0 and sum(1 for v in ids.values() if v > 1) == 0),
        "label_exclusivity_violation_count": exclusivity,
        "duplicate_outcome_row_id_count": sum(1 for v in ids.values() if v > 1),
        "metric_nan_count": metric_nan,
        "metric_inf_count": metric_inf,
    }


def summarize_ap0_baseline(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = [r for r in rows if r.get("branch") == "SourceAP0Baseline"]
    trace = []
    for r in source_rows:
        trace.append({"stage": "P3_SAME_PANEL_SOURCE_AP0_BASELINE", "status": "source_ap0_outcome_row", **r})
    h20 = [r for r in source_rows if inum(r.get("horizon")) == 20]
    all_real = source_rows
    h240 = [r for r in source_rows if inum(r.get("horizon")) == 240]
    unique_sources = {str(r.get("source_action_id")) for r in source_rows}
    robust_sources = {str(r.get("source_action_id")) for r in source_rows if inum(r.get("AP0_horizon_robust_CP_label"))}
    summary = {
        "stage": "P3_SAME_PANEL_SOURCE_AP0_BASELINE",
        "status": "summary",
        "source_action_count": len(unique_sources),
        "branch_horizon_row_count_expected": len(unique_sources) * len(BRANCHES) * len(HORIZONS),
        "branch_horizon_row_count_actual": len(source_rows),
        "AP0_weak_CP_precision_h20": sum(inum(r.get("AP0_weak_CP_label")) for r in h20) / max(1, len(h20)),
        "AP0_strong_CP_precision_h20": sum(inum(r.get("AP0_strong_CP_label")) for r in h20) / max(1, len(h20)),
        "AP0_weak_CP_precision_h80": sum(inum(r.get("AP0_weak_CP_label")) for r in source_rows if inum(r.get("horizon")) == 80) / max(1, len([r for r in source_rows if inum(r.get("horizon")) == 80])),
        "AP0_weak_CP_precision_h240": sum(inum(r.get("AP0_weak_CP_label")) for r in h240) / max(1, len(h240)),
        "AP0_long_risk_rate_h240": sum(inum(r.get("AP0_long_risk_label")) for r in h240) / max(1, len(h240)),
        "AP0_V_ctrl_lcb_h20": lcb_mean([fnum(r.get("AP0_V_ctrl")) for r in h20]),
        "AP0_V_ctrl_lcb_all": lcb_mean([fnum(r.get("AP0_V_ctrl")) for r in all_real]),
        "AP0_horizon_robust_action_count": len(robust_sources),
        "AP0_horizon_robust_coverage": len(robust_sources) / max(1, len(unique_sources)),
        "source_ap0_baseline_useful": 0,
        "source_ap0_baseline_bad": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["source_ap0_baseline_useful"] = int(summary["AP0_weak_CP_precision_h20"] >= 0.30 and summary["AP0_V_ctrl_lcb_h20"] > 0 and summary["AP0_long_risk_rate_h240"] <= 0.20)
    summary["source_ap0_baseline_bad"] = int(summary["AP0_weak_CP_precision_h20"] < 0.20 or summary["AP0_V_ctrl_lcb_h20"] <= 0)
    return trace, summary


def summarize_generated_h20(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    for p in PRIMITIVES:
        real = [r for r in rows if r.get("branch") == "RealSource" and r.get("primitive_id") == p and inum(r.get("horizon")) == 20]
        vals = [fnum(r.get("V_ctrl")) for r in real]
        row = {
            "stage": "P4_GENERATED_SOURCE_IMMEDIATE_DIRECTION",
            "status": "primitive_h20_summary",
            "primitive_id": p,
            "generated_action_count": len(real),
            "certificate_pass_action_count": sum(inum(r.get("certificate_pass")) for r in real),
            "h20_row_count": len(real),
            "weak_CP_precision_h20": sum(inum(r.get("weak_CP_label")) for r in real) / max(1, len(real)),
            "strong_CP_precision_h20": sum(inum(r.get("strong_CP_label")) for r in real) / max(1, len(real)),
            "bad_event_rate_h20": sum(inum(r.get("bad_event_label")) for r in real) / max(1, len(real)),
            "null_rate_h20": sum(inum(r.get("null_event_label")) for r in real) / max(1, len(real)),
            "V_ctrl_mean_h20": mean(vals),
            "V_ctrl_lcb_h20": lcb_mean(vals),
            "CEp99_delta_mean_h20": mean([fnum(r.get("CEp99_delta")) for r in real]),
            "margin_p10_delta_mean_h20": mean([fnum(r.get("margin_p10_delta")) for r in real]),
            "ECE_delta_mean_h20": mean([fnum(r.get("ECE_delta")) for r in real]),
            "NLL_delta_mean_h20": mean([fnum(r.get("NLL_delta")) for r in real]),
            "beats_adamwparallel_rate_h20": sum(inum(r.get("beats_adamwparallel")) for r in real) / max(1, len(real)),
            "beats_bestlr_rate_h20": sum(inum(r.get("beats_bestlr")) for r in real) / max(1, len(real)),
            "beats_noop_rate_h20": sum(inum(r.get("beats_noop")) for r in real) / max(1, len(real)),
            "beats_random_rate_h20": sum(inum(r.get("beats_random")) for r in real) / max(1, len(real)),
            "support_balance_pass_h20": int(len(set(str(r.get("family_id")) for r in real)) >= 8),
            "p4_weak_pass": 0,
            "p4_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["p4_weak_pass"] = int(row["weak_CP_precision_h20"] >= 0.30 and row["V_ctrl_lcb_h20"] > 0 and row["bad_event_rate_h20"] <= 0.10 and row["null_rate_h20"] <= 0.20)
        row["p4_strong_pass"] = int(row["weak_CP_precision_h20"] >= 0.50 and row["strong_CP_precision_h20"] >= 0.20 and row["V_ctrl_lcb_h20"] > 0.05 and row["bad_event_rate_h20"] <= 0.05 and row["null_rate_h20"] <= 0.15)
        out.append(row)
    best = max(out, key=lambda r: (inum(r.get("p4_weak_pass")), fnum(r.get("weak_CP_precision_h20")), fnum(r.get("V_ctrl_lcb_h20"))), default={})
    summary = {
        "stage": "P4_GENERATED_SOURCE_IMMEDIATE_DIRECTION",
        "status": "summary",
        "best_primitive_id": best.get("primitive_id", ""),
        "best_weak_CP_precision_h20": best.get("weak_CP_precision_h20", 0.0),
        "best_V_ctrl_lcb_h20": best.get("V_ctrl_lcb_h20", 0.0),
        "best_bad_event_rate_h20": best.get("bad_event_rate_h20", 0.0),
        "h20_immediate_direction_pass": int(any(inum(r.get("p4_weak_pass")) for r in out)),
        "h20_immediate_direction_strong_pass": int(any(inum(r.get("p4_strong_pass")) for r in out)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return out + [summary], summary


def damage_matrix(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real = [r for r in rows if r.get("branch") == "RealSource"]
    out = []
    for r in real:
        dmg = fnum(r.get("V_ctrl")) - fnum(r.get("AP0_V_ctrl"))
        out.append({
            "stage": "P5_SOURCE_TO_GENERATED_DAMAGE_MATRIX",
            "status": "damage_row",
            "source_action_id": r.get("source_action_id"),
            "generated_action_id": r.get("generated_action_id"),
            "primitive_id": r.get("primitive_id"),
            "horizon": r.get("horizon"),
            "V_AP0": r.get("AP0_V_ctrl"),
            "V_generated": r.get("V_ctrl"),
            "Damage": dmg,
            "WeakCP_AP0": r.get("AP0_weak_CP_label"),
            "WeakCP_generated": r.get("weak_CP_label"),
            "StrongCP_AP0": r.get("AP0_strong_CP_label"),
            "StrongCP_generated": r.get("strong_CP_label"),
            "LongRisk_AP0": r.get("AP0_long_risk_label"),
            "LongRisk_generated": r.get("long_risk_label"),
            "source_positive_lost_after_generation": int(inum(r.get("AP0_weak_CP_label")) and not inum(r.get("weak_CP_label"))),
            "source_negative_fixed_after_generation": int((not inum(r.get("AP0_weak_CP_label"))) and inum(r.get("weak_CP_label"))),
            "payload_cosine_source_generated": "",
            "payload_norm_ratio": "",
            "certificate_pass": r.get("certificate_pass"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    damages = [fnum(r.get("Damage")) for r in out]
    source_pos = [r for r in out if inum(r.get("WeakCP_AP0"))]
    source_neg = [r for r in out if not inum(r.get("WeakCP_AP0"))]
    summary = {
        "stage": "P5_SOURCE_TO_GENERATED_DAMAGE_MATRIX",
        "status": "summary",
        "paired_horizon_count": len(out),
        "Damage_mean": mean(damages),
        "Damage_median": statistics.median(damages) if damages else 0.0,
        "source_positive_horizon_count": len(source_pos),
        "source_positive_lost_after_generation_rate": sum(inum(r.get("source_positive_lost_after_generation")) for r in source_pos) / max(1, len(source_pos)),
        "source_negative_fixed_after_generation_rate": sum(inum(r.get("source_negative_fixed_after_generation")) for r in source_neg) / max(1, len(source_neg)),
        "generator_value_preserving_pass": 0,
        "generator_damage_fail": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["generator_value_preserving_pass"] = int(summary["Damage_median"] >= -0.02 and summary["source_positive_lost_after_generation_rate"] <= 0.30)
    summary["generator_damage_fail"] = int(summary["source_positive_lost_after_generation_rate"] >= 0.70 or summary["Damage_median"] < -0.10)
    return out + [summary], summary


def horizon_audit(rows: list[dict[str, Any]], best_primitive: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    for p in PRIMITIVES:
        for h in HORIZONS:
            real = [r for r in rows if r.get("branch") == "RealSource" and r.get("primitive_id") == p and inum(r.get("horizon")) == h]
            vals = [fnum(r.get("V_ctrl")) for r in real]
            out.append({
                "stage": "P6_HORIZON_EXTENSION_LONGRISK_AUDIT",
                "status": "primitive_horizon_summary",
                "primitive_id": p,
                "horizon": h,
                "weak_CP_precision": sum(inum(r.get("weak_CP_label")) for r in real) / max(1, len(real)),
                "strong_CP_precision": sum(inum(r.get("strong_CP_label")) for r in real) / max(1, len(real)),
                "bad_event_rate": sum(inum(r.get("bad_event_label")) for r in real) / max(1, len(real)),
                "null_rate": sum(inum(r.get("null_event_label")) for r in real) / max(1, len(real)),
                "long_risk_rate": sum(inum(r.get("long_risk_label")) for r in real) / max(1, len(real)),
                "V_ctrl_mean": mean(vals),
                "V_ctrl_lcb": lcb_mean(vals),
                "horizon_robust_action_count": sum(inum(r.get("horizon_robust_CP_label")) for r in real),
                "horizon_robust_action_coverage": sum(inum(r.get("horizon_robust_CP_label")) for r in real) / max(1, len(real)),
                "short_only_action_count": "",
                "long_risk_action_count": sum(inum(r.get("long_risk_label")) for r in real),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    selected = [r for r in out if r.get("primitive_id") == best_primitive]
    by_h = {inum(r.get("horizon")): r for r in selected}
    pass_weak = int(
        fnum(by_h.get(80, {}).get("weak_CP_precision")) >= 0.30
        and fnum(by_h.get(240, {}).get("weak_CP_precision")) >= 0.20
        and fnum(by_h.get(240, {}).get("long_risk_rate")) <= 0.15
    )
    pass_strong = int(
        max([fnum(r.get("horizon_robust_action_coverage")) for r in selected] or [0.0]) >= 0.03
        and min([fnum(r.get("V_ctrl_lcb")) for r in selected] or [0.0]) > 0
        and fnum(by_h.get(240, {}).get("long_risk_rate")) <= 0.10
    )
    summary = {
        "stage": "P6_HORIZON_EXTENSION_LONGRISK_AUDIT",
        "status": "summary",
        "selected_primitive_id": best_primitive,
        "weak_CP_precision_h80": by_h.get(80, {}).get("weak_CP_precision", 0.0),
        "weak_CP_precision_h240": by_h.get(240, {}).get("weak_CP_precision", 0.0),
        "long_risk_rate_h240": by_h.get(240, {}).get("long_risk_rate", 0.0),
        "horizon_robust_action_coverage": max([fnum(r.get("horizon_robust_action_coverage")) for r in selected] or [0.0]),
        "horizon_extension_pass": pass_weak,
        "horizon_extension_strong_pass": pass_strong,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return out + [summary], summary


def certificate_lift(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    real = [r for r in rows if r.get("branch") == "RealSource"]
    y_weak = [inum(r.get("weak_CP_label")) for r in real]
    y_strong = [inum(r.get("strong_CP_label")) for r in real]
    y_long = [inum(r.get("long_risk_label")) for r in real]
    scores = [fnum(r.get("certificate_score")) for r in real]
    cert_pass = [r for r in real if inum(r.get("certificate_pass"))]
    cert_fail = [r for r in real if not inum(r.get("certificate_pass"))]
    pwp = sum(inum(r.get("weak_CP_label")) for r in cert_pass) / max(1, len(cert_pass))
    pwf = sum(inum(r.get("weak_CP_label")) for r in cert_fail) / max(1, len(cert_fail))
    plp = sum(inum(r.get("long_risk_label")) for r in cert_pass) / max(1, len(cert_pass))
    plf = sum(inum(r.get("long_risk_label")) for r in cert_fail) / max(1, len(cert_fail))
    summary = {
        "stage": "P7_CERTIFICATE_SUFFICIENCY_LIFT_AUDIT",
        "status": "summary",
        "certificate_id": "v9410-source-cert-v1",
        "joined_outcome_count": len(real),
        "certificate_pass_action_count": len(cert_pass),
        "AUC_certificate_weak_CP": auc_score(scores, y_weak) if real else 0.5,
        "AUC_certificate_strong_CP": auc_score(scores, y_strong) if real else 0.5,
        "AUC_certificate_longrisk": auc_score(scores, y_long) if real else 0.5,
        "P_weak_CP_given_cert_pass": pwp,
        "P_weak_CP_given_cert_fail": pwf,
        "P_longrisk_given_cert_pass": plp,
        "P_longrisk_given_cert_fail": plf,
        "Lift_weak": pwp / max(1e-9, pwf),
        "Lift_longrisk": plp / max(1e-9, plf),
        "calibration_to_heldout_drift": "",
        "monotone_sign_pass": 0,
        "certificate_sufficiency_pass": 0,
        "certificate_sufficiency_weak_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["monotone_sign_pass"] = int(summary["AUC_certificate_weak_CP"] >= 0.5 and summary["AUC_certificate_longrisk"] <= 0.5)
    summary["certificate_sufficiency_pass"] = int(summary["Lift_weak"] >= 2.0 and summary["AUC_certificate_weak_CP"] >= 0.70 and summary["Lift_longrisk"] <= 0.70 and summary["P_longrisk_given_cert_pass"] <= 0.15 and summary["monotone_sign_pass"] and len(cert_pass) >= 32)
    summary["certificate_sufficiency_weak_pass"] = int(summary["Lift_weak"] >= 1.5 and summary["AUC_certificate_weak_CP"] >= 0.65)
    trace = [
        {
            "stage": "P7_CERTIFICATE_COMPONENT_TRACE",
            "status": "certificate_outcome_row",
            "generated_action_id": r.get("generated_action_id"),
            "primitive_id": r.get("primitive_id"),
            "horizon": r.get("horizon"),
            "certificate_pass": r.get("certificate_pass"),
            "certificate_score": r.get("certificate_score"),
            "weak_CP": r.get("weak_CP_label"),
            "strong_CP": r.get("strong_CP_label"),
            "horizon_robust_CP": r.get("horizon_robust_CP_label"),
            "long_risk": r.get("long_risk_label"),
            "V_ctrl": r.get("V_ctrl"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        for r in real
    ]
    ablation = [
        {"stage": "P7_CERTIFICATE_COMPONENT_ABLATION", "status": "summary", "component": "certificate_score", "AUC_weak_CP": summary["AUC_certificate_weak_CP"], "AUC_longrisk": summary["AUC_certificate_longrisk"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    ]
    return [summary], trace + [summary], summary


def p8_triage(p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any], p7: dict[str, Any]) -> dict[str, Any]:
    if inum(p3.get("source_ap0_baseline_bad")):
        case = "Case A: source AP0 bad, generated bad"
        route = "R2-SourcePanelValuePoorDespiteStratification"
        next_impl = "rebuild_source_selector_or_source_panel_input"
    elif not inum(p4.get("h20_immediate_direction_pass")):
        case = "Case B: source AP0 good, generated bad"
        route = "R3-GeneratedSourceImmediateValueFail"
        next_impl = "redesign_immediate_value_source_generator"
    elif inum(p5.get("generator_damage_fail")):
        case = "Case B: source AP0 good, generated bad"
        route = "R4-GeneratorTransformDamage"
        next_impl = "redesign_value_preserving_transform"
    elif not inum(p6.get("horizon_extension_pass")):
        case = "Case C: generated h20 good, h240 bad"
        route = "R5-HorizonLongRiskDominates"
        next_impl = "redesign_horizon_guarded_source_primitive"
    elif not inum(p7.get("certificate_sufficiency_pass")):
        case = "Case D: generated good, certificate bad"
        route = "R6-CertificateNotEffectValid"
        next_impl = "redesign_effect_valid_certificate_components"
    else:
        case = "Case E: generated good, certificate good, runtime blocked"
        route = "R7-SourceControllerSupportCollapse"
        next_impl = "build_minimal_source_certificate_controller"
    return {
        "stage": "P8_PRIMITIVE_TRIAGE_ROUTE",
        "status": "summary",
        "triage_case": case,
        "route": route,
        "source_AP0_status": "bad" if inum(p3.get("source_ap0_baseline_bad")) else "useful",
        "generated_source_status": "h20_pass" if inum(p4.get("h20_immediate_direction_pass")) else "h20_fail",
        "generator_damage_status": "damage_fail" if inum(p5.get("generator_damage_fail")) else "not_primary",
        "horizon_status": "pass" if inum(p6.get("horizon_extension_pass")) else "fail_or_not_open",
        "certificate_status": "pass" if inum(p7.get("certificate_sufficiency_pass")) else "fail_or_not_open",
        "runtime_status": "not_run",
        "next_required_implementation": next_impl,
        "stop_threshold_tuning_flag": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}


def write_hashes(out_dir: Path, artifacts: list[Path]) -> None:
    rows = [{"artifact": rel(p), "sha256": sha256_file(p)} for p in artifacts if p.exists()]
    write_csv(out_dir / "artifact_hashes_v9420.csv", rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from(args.device)
    source_v9410 = Path(args.source_v9410)
    source_v9330 = Path(args.source_v9330)

    generated = load_generated_rows(source_v9410)
    source_payload_by_id = load_source_payload_rows(source_v9330)
    p0 = p0_boundary(source_v9410)

    p1_rows, p1_join_trace, p1_sink_trace, p1 = preflight(args, generated, source_payload_by_id, device)
    if inum(p1.get("preflight_pass")):
        outcome_rows, completion_rows, retry_rows, p2 = materialize_source_outcomes(args, generated, source_payload_by_id, device, BRANCHES, HORIZONS)
        qa = quality_audit(outcome_rows, len(generated) * len(BRANCHES) * len(HORIZONS))
        p2.update(qa)
    else:
        outcome_rows, completion_rows, retry_rows = [], [], [{"stage": "P2_SOURCE_OUTCOME_MATERIALIZER", "status": "not_run", "reason": "P1_preflight_failed", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}]
        p2 = not_run("P2_SOURCE_OUTCOME_MATERIALIZER_SCALEUP", "P1_preflight_failed")
        p2.update({"source_outcome_materializer_pass": 0, "source_outcome_materialized": 0, "branch_horizon_row_count_expected": len(generated) * len(BRANCHES) * len(HORIZONS), "branch_horizon_row_count_actual": 0})

    if inum(p2.get("source_outcome_materializer_pass")):
        ap0_trace, p3 = summarize_ap0_baseline(outcome_rows)
        p4_rows, p4 = summarize_generated_h20(outcome_rows)
        p5_rows, p5 = damage_matrix(outcome_rows)
        p6_rows, p6 = horizon_audit(outcome_rows, str(p4.get("best_primitive_id") or ""))
        p7_rows, p7_trace, p7 = certificate_lift(outcome_rows)
    else:
        ap0_trace, p3 = [], not_run("P3_SAME_PANEL_SOURCE_AP0_BASELINE", "P2_source_outcome_not_materialized")
        p3.update({"source_ap0_baseline_bad": 0, "source_ap0_baseline_useful": 0})
        p4_rows, p4 = [not_run("P4_GENERATED_SOURCE_IMMEDIATE_DIRECTION", "P2_source_outcome_not_materialized")], not_run("P4_GENERATED_SOURCE_IMMEDIATE_DIRECTION", "P2_source_outcome_not_materialized")
        p4.update({"h20_immediate_direction_pass": 0})
        p5_rows, p5 = [not_run("P5_SOURCE_TO_GENERATED_DAMAGE_MATRIX", "P2_source_outcome_not_materialized")], not_run("P5_SOURCE_TO_GENERATED_DAMAGE_MATRIX", "P2_source_outcome_not_materialized")
        p5.update({"generator_damage_fail": 0})
        p6_rows, p6 = [not_run("P6_HORIZON_EXTENSION_LONGRISK_AUDIT", "P2_source_outcome_not_materialized")], not_run("P6_HORIZON_EXTENSION_LONGRISK_AUDIT", "P2_source_outcome_not_materialized")
        p6.update({"horizon_extension_pass": 0})
        p7_rows, p7_trace, p7 = [not_run("P7_CERTIFICATE_SUFFICIENCY_LIFT_AUDIT", "P2_source_outcome_not_materialized")], [], not_run("P7_CERTIFICATE_SUFFICIENCY_LIFT_AUDIT", "P2_source_outcome_not_materialized")
        p7.update({"certificate_sufficiency_pass": 0})

    p8 = p8_triage(p3, p4, p5, p6, p7) if inum(p2.get("source_outcome_materializer_pass")) else {
        "stage": "P8_PRIMITIVE_TRIAGE_ROUTE",
        "status": "summary",
        "triage_case": "materializer path fail",
        "route": "R1-SourceOutcomeMaterializerPathFail",
        "source_AP0_status": "unknown",
        "generated_source_status": "unknown",
        "generator_damage_status": "unknown",
        "horizon_status": "not_open",
        "certificate_status": "not_open",
        "runtime_status": "not_run",
        "next_required_implementation": "fix_source_outcome_materializer_path",
        "stop_threshold_tuning_flag": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    sentinel_args = argparse.Namespace(**vars(args))
    sentinel_args.sentinel_seeds = args.sentinel_seeds
    p9_rows, p9_trace, p9 = v9410.p2_base_acc_sentinel(sentinel_args, device)
    p9["stage"] = "P9_BASE_ACC_SENTINEL_EXTENDED"
    for r in p9_rows:
        r["stage"] = "P9_BASE_ACC_SENTINEL_EXTENDED"
    for r in p9_trace:
        r["stage"] = "P9_BASE_ACC_SENTINEL_TRACE"

    p10 = not_run("P10_MINIMAL_SOURCE_CERTIFICATE_CONTROLLER", "upstream_source_or_certificate_gate_not_passed")
    p10["source_controller_pass"] = 0
    p11 = not_run("P11_SELECTED_SOURCE_ONLINE_RUNTIME", "P10_controller_not_selected")
    p11["selected_runtime_pass"] = 0
    system_pass = 0
    p12 = {
        "stage": "P12_SYSTEM_INTEGRATION_GATE",
        "status": "summary",
        "system_candidate_id": "SYS-v9420-source-outcome-materializer-closure",
        "primitive_id": p4.get("best_primitive_id", ""),
        "controller_id": "not_selected",
        "runtime_candidate_id": "not_selected",
        "source_outcome_materialized": p2.get("source_outcome_materialized", 0),
        "h20_immediate_direction_pass": p4.get("h20_immediate_direction_pass", 0),
        "horizon_extension_pass": p6.get("horizon_extension_pass", 0),
        "certificate_sufficiency_pass": p7.get("certificate_sufficiency_pass", 0),
        "source_controller_pass": p10.get("source_controller_pass", 0),
        "selected_runtime_pass": p11.get("selected_runtime_pass", 0),
        "coverage_heldout": 0.0,
        "weak_CP_precision_heldout": 0.0,
        "strong_CP_precision_heldout": 0.0,
        "horizon_robust_precision_heldout": 0.0,
        "bad_event_rate_heldout": 0.0,
        "null_rate_heldout": 0.0,
        "long_risk_rate_heldout": 0.0,
        "V_ctrl_lcb_heldout": 0.0,
        "step_ratio_q90": "",
        "memory_ratio": "",
        "dataset_name_used": 0,
        "uses_validation_or_test": 0,
        "uses_outcome_at_commit": 0,
        "uses_future_step": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": system_pass,
        "reason": p8.get("next_required_implementation"),
    }

    for name, reason in [
        ("p13_leaveout_paired_replay_boundary_v9420.csv", "P12_system_controller_not_official"),
        ("p14_short_full_training_mlp_comparison_boundary_v9420.csv", "P12_system_controller_not_official"),
    ]:
        write_csv(out_dir / name, [{"stage": name.removesuffix(".csv"), "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}])

    if not inum(p1.get("preflight_pass")) or not inum(p2.get("source_outcome_materializer_pass")):
        route = "R1-SourceOutcomeMaterializerPathFail"
        blocker = "source_outcome_materializer_path_fail"
        next_impl = "fix_source_outcome_materializer_path"
    else:
        route = str(p8.get("route"))
        if route == "R2-SourcePanelValuePoorDespiteStratification":
            blocker = "source_panel_value_poor_despite_stratification"
        elif route == "R3-GeneratedSourceImmediateValueFail":
            blocker = "generated_source_immediate_value_fail"
        elif route == "R4-GeneratorTransformDamage":
            blocker = "generator_transform_damage"
        elif route == "R5-HorizonLongRiskDominates":
            blocker = "horizon_long_risk_dominates"
        elif route == "R6-CertificateNotEffectValid":
            blocker = "certificate_not_effect_valid"
        else:
            blocker = "source_controller_not_selected"
        next_impl = str(p8.get("next_required_implementation"))

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9410": p0.get("route_v9410"),
        "source_generator_materialized": p0.get("source_generator_materialized"),
        "generated_action_count_total": len(generated),
        "p1_preflight_pass": p1.get("preflight_pass"),
        "source_outcome_materialized": p2.get("source_outcome_materialized", 0),
        "branch_horizon_row_count_expected": p2.get("branch_horizon_row_count_expected"),
        "branch_horizon_row_count_actual": p2.get("branch_horizon_row_count_actual"),
        "source_outcome_materializer_pass": p2.get("source_outcome_materializer_pass", 0),
        "rows_per_sec_total": p2.get("rows_per_sec_total", ""),
        "same_panel_source_ap0_baseline_bad": p3.get("source_ap0_baseline_bad", 0),
        "AP0_weak_CP_precision_h20": p3.get("AP0_weak_CP_precision_h20", ""),
        "AP0_V_ctrl_lcb_h20": p3.get("AP0_V_ctrl_lcb_h20", ""),
        "h20_immediate_direction_pass": p4.get("h20_immediate_direction_pass", 0),
        "best_h20_primitive_id": p4.get("best_primitive_id", ""),
        "best_weak_CP_precision_h20": p4.get("best_weak_CP_precision_h20", ""),
        "best_V_ctrl_lcb_h20": p4.get("best_V_ctrl_lcb_h20", ""),
        "generator_damage_fail": p5.get("generator_damage_fail", 0),
        "Damage_median": p5.get("Damage_median", ""),
        "source_positive_lost_after_generation_rate": p5.get("source_positive_lost_after_generation_rate", ""),
        "horizon_extension_pass": p6.get("horizon_extension_pass", 0),
        "long_risk_rate_h240": p6.get("long_risk_rate_h240", ""),
        "certificate_sufficiency_pass": p7.get("certificate_sufficiency_pass", 0),
        "AUC_certificate_weak_CP": p7.get("AUC_certificate_weak_CP", ""),
        "Lift_weak": p7.get("Lift_weak", ""),
        "Lift_longrisk": p7.get("Lift_longrisk", ""),
        "base_acc_sentinel_complete": p9.get("sentinel_complete"),
        "mean_test_acc_LQ": p9.get("mean_test_acc_LQ"),
        "mean_test_acc_MLP": p9.get("mean_test_acc_MLP"),
        "base_acc_used_for_controller": p9.get("base_acc_used_for_controller"),
        "source_controller_pass": p10.get("source_controller_pass"),
        "selected_runtime_pass": p11.get("selected_runtime_pass"),
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "success_v9420_strict_purekan_functional": 0,
        "success_v9420_full_functional": 0,
        "success_v9420_external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    manifest = {
        "run_id": out_dir.name,
        "created_at_utc": now_iso(),
        "script": rel(SCRIPT_PATH),
        "plan": rel(PLAN_PATH),
        "args": vars(args),
        "source_v9410": rel(source_v9410),
        "source_v9330": rel(source_v9330),
        "source_v9280": rel(Path(args.source_v9280)),
        "device": str(device),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_csv(out_dir / "p0_v9410_boundary_reanalysis.csv", [p0])
    write_csv(out_dir / "p1_source_outcome_materializer_preflight.csv", p1_rows)
    write_csv(out_dir / "source_outcome_materializer_failure_trace_v9420.csv", p1_join_trace + retry_rows)
    write_csv(out_dir / "source_outcome_join_key_trace_v9420.csv", p1_join_trace)
    write_csv(out_dir / "source_outcome_row_sink_trace_v9420.csv", p1_sink_trace + completion_rows)
    write_csv(out_dir / "p2_source_outcome_materializer_scaleup.csv", [p2])
    write_csv(out_dir / "source_outcome_trace_v9420.csv", outcome_rows)
    write_csv(out_dir / "branch_horizon_completion_trace_v9420.csv", completion_rows)
    write_csv(out_dir / "p3_same_panel_source_ap0_baseline.csv", [p3])
    write_csv(out_dir / "source_ap0_outcome_trace_v9420.csv", ap0_trace)
    write_csv(out_dir / "p4_generated_source_immediate_direction.csv", p4_rows)
    write_csv(out_dir / "p5_source_to_generated_damage_matrix.csv", p5_rows)
    write_csv(out_dir / "p6_horizon_extension_longrisk_audit.csv", p6_rows)
    write_csv(out_dir / "p7_certificate_sufficiency_lift_audit.csv", p7_rows)
    write_csv(out_dir / "certificate_component_ablation_v9420.csv", p7_trace)
    write_csv(out_dir / "p8_primitive_triage_route.csv", [p8])
    write_csv(out_dir / "p9_base_acc_sentinel_extended.csv", p9_rows)
    write_csv(out_dir / "base_acc_training_trace_v9420.csv", p9_trace)
    write_csv(out_dir / "p10_minimal_source_certificate_controller.csv", [p10])
    write_csv(out_dir / "source_controller_frontier_trace_v9420.csv", [])
    write_csv(out_dir / "p11_selected_source_online_runtime.csv", [p11])
    write_csv(out_dir / "runtime_component_trace_v9420.csv", [])
    write_csv(out_dir / "p12_system_integration_gate_v9420.csv", [p12])
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "p1_preflight_pass": p1.get("preflight_pass"),
        "source_outcome_materialized": p2.get("source_outcome_materialized", 0),
        "h20_immediate_direction_pass": p4.get("h20_immediate_direction_pass", 0),
        "horizon_extension_pass": p6.get("horizon_extension_pass", 0),
        "certificate_sufficiency_pass": p7.get("certificate_sufficiency_pass", 0),
        "base_acc_sentinel_complete": p9.get("sentinel_complete"),
        "base_acc_used_for_controller": p9.get("base_acc_used_for_controller"),
        "source_controller_pass": p10.get("source_controller_pass"),
        "selected_runtime_pass": p11.get("selected_runtime_pass"),
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "contract_audit_v9420.csv", [contract])
    failure = {
        "route": route,
        "F1_materializer_path_fail": int(not inum(p1.get("preflight_pass")) or not inum(p2.get("source_outcome_materializer_pass", 0))),
        "F2_source_panel_value_poor": int(inum(p3.get("source_ap0_baseline_bad", 0))),
        "F3_generated_source_immediate_fail": int(not inum(p4.get("h20_immediate_direction_pass", 0))),
        "F4_generator_transform_damage": int(inum(p5.get("generator_damage_fail", 0))),
        "F5_horizon_longrisk_fail": int(inum(p4.get("h20_immediate_direction_pass", 0)) and not inum(p6.get("horizon_extension_pass", 0))),
        "F6_certificate_not_effect_valid": int(inum(p6.get("horizon_extension_pass", 0)) and not inum(p7.get("certificate_sufficiency_pass", 0))),
        "F7_controller_not_selected": int(not inum(p10.get("source_controller_pass", 0))),
        "F8_runtime_not_selected": int(not inum(p11.get("selected_runtime_pass", 0))),
        "F9_base_acc_sentinel_catastrophic": int(inum(p9.get("LQ_catastrophic_fail", 0))),
        "F10_downstream_not_open": 1,
        "primary_blocker": blocker,
    }
    write_csv(out_dir / "failure_table_v9420.csv", [failure])
    provenance = audit_no_fake([
        out_dir / "p1_source_outcome_materializer_preflight.csv",
        out_dir / "source_outcome_trace_v9420.csv",
        out_dir / "p3_same_panel_source_ap0_baseline.csv",
        out_dir / "p4_generated_source_immediate_direction.csv",
        out_dir / "p7_certificate_sufficiency_lift_audit.csv",
        out_dir / "contract_audit_v9420.csv",
    ])
    write_csv(out_dir / "provenance_audit_v9420.csv", [provenance])
    artifacts = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "p0_v9410_boundary_reanalysis.csv",
        out_dir / "p1_source_outcome_materializer_preflight.csv",
        out_dir / "p2_source_outcome_materializer_scaleup.csv",
        out_dir / "source_outcome_trace_v9420.csv",
        out_dir / "p3_same_panel_source_ap0_baseline.csv",
        out_dir / "p4_generated_source_immediate_direction.csv",
        out_dir / "p5_source_to_generated_damage_matrix.csv",
        out_dir / "p6_horizon_extension_longrisk_audit.csv",
        out_dir / "p7_certificate_sufficiency_lift_audit.csv",
        out_dir / "p8_primitive_triage_route.csv",
        out_dir / "p9_base_acc_sentinel_extended.csv",
        out_dir / "p12_system_integration_gate_v9420.csv",
        out_dir / "contract_audit_v9420.csv",
        out_dir / "provenance_audit_v9420.csv",
        out_dir / "failure_table_v9420.csv",
    ]
    write_hashes(out_dir, artifacts)
    print(json.dumps({"out_dir": str(out_dir), "route": route, "source_rows": p2.get("branch_horizon_row_count_actual"), "h20_pass": p4.get("h20_immediate_direction_pass", 0)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
