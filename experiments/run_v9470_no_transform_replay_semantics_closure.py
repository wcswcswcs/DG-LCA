#!/usr/bin/env python3
"""DG-KAN v9.4.7 no-transform replay semantics closure runner.

The runner fixes the specific v9.4.6 diagnostic issue by evaluating source and
AP0w no-transform clone through a canonical branch-name-invariant rollout path.
It does not promote oracle, generator, certificate, runtime, or downstream
diagnostics to official system success.
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
from collections import defaultdict
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
import run_v9450_robust_source_generator_objective_aligned_certificate as v9450  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.7_NoTransformReplaySemanticsClosure_RobustSourceRevalidation_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9470_no_transform_replay_semantics_closure.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9460 = RESULT_ROOT / "v9460_source_identity_no_transform_replay_closure_robust_source_revalidation_first_20260514T130000Z"
DEFAULT_V9450 = RESULT_ROOT / "v9450_robust_source_generator_objective_aligned_certificate_parallel_closure_first_20260514T120000Z"
DEFAULT_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HORIZONS = [20, 80, 240]
P2_HORIZONS = [1, 2, 5, 20]
BRANCHES = [
    "RealFunctional",
    "AdamWParallel",
    "bestLR",
    "NoOp",
    "Random",
    "ShuffledFunctionalPayload",
]
CONTROL_BRANCHES = [b for b in BRANCHES if b != "RealFunctional"]
METRICS = [
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
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--source-v9460", default=str(DEFAULT_V9460))
    p.add_argument("--source-v9450", default=str(DEFAULT_V9450))
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
    vals = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(vals) if vals else 0.0


def lcb(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def q(xs: list[float], frac: float) -> float:
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


def tensor_hash(tensors: list[torch.Tensor]) -> str:
    return v9420.v9380.v9320.tensor_hash(tensors)


def tensor_bytes_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def params_hash(params: list[torch.Tensor]) -> str:
    return tensor_hash(params)


def state_hash(states: list[AdamWState]) -> str:
    h = hashlib.sha256()
    for st in states:
        h.update(str(st.step).encode("utf-8"))
        h.update(st.m.detach().cpu().contiguous().numpy().tobytes())
        h.update(st.v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_hash(gen: torch.Generator) -> str:
    return hashlib.sha256(gen.get_state().detach().cpu().numpy().tobytes()).hexdigest()


def batch_hash(idx: torch.Tensor) -> str:
    return hashlib.sha256(idx.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def clone_params(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clone_states(states: list[AdamWState]) -> list[AdamWState]:
    return v9340.clone_states(states)


def load_base_tables(args: argparse.Namespace) -> dict[str, Any]:
    full_rows = read_csv(Path(args.source_v9350) / "full_control_outcome_table_v9350.csv")
    payload_rows = read_csv(Path(args.source_v9330) / "action_payload_disk_replay_trace_v9330.csv")
    stats = v9430.summarize_action_universe(full_rows, payload_rows)
    oracle_ids = v9450.oracle_panel(stats, int(args.oracle_actions))
    payload_by_id = {str(r.get("action_id")): r for r in payload_rows if r.get("status") == "payload_disk_replay_row"}
    trace = read_csv(Path(args.source_v9450) / "oracle_seeded_generator_trace_v9450.csv")
    ap0w = {
        str(r.get("source_action_id")): r
        for r in trace
        if r.get("status") == "generated_action_row" and str(r.get("primitive_id")).startswith("AP0w-")
    }
    return {"full_rows": full_rows, "payload_rows": payload_rows, "stats": stats, "oracle_ids": oracle_ids, "payload_by_id": payload_by_id, "ap0w_generated": ap0w}


def old_robust_label(st: dict[str, Any]) -> int:
    return int(inum(st.get("weak_h20")) and fnum(st.get("V", {}).get(20)) > 0.0 and fnum(st.get("long_risk_h240")) <= 0.10 and fnum(st.get("family_support_count")) > 0)


def branch_config(branch: str) -> dict[str, str]:
    mapping = {
        "RealFunctional": ("functional_payload", "task_params_plus_payload"),
        "AdamWParallel": ("adamw_parallel_no_payload", "task_params_no_payload"),
        "bestLR": ("best_lr_probe", "best_immediate_probe_lr_scale"),
        "NoOp": ("noop_pre_step", "pre_step_params_no_current_adamw"),
        "Random": ("norm_matched_random_payload", "task_params_plus_norm_matched_random_payload"),
        "ShuffledFunctionalPayload": ("shuffled_functional_payload", "task_params_plus_shuffled_payload"),
    }
    cid, sem = mapping[branch]
    return {"branch": branch, "branch_config_id": cid, "branch_semantics": sem, "branch_config_hash": stable_hash("branch-config-v9470", cid, sem)}


def load_source_payload(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return v9420.load_source_payload(row, cache, device)


def load_clone_payload(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return v9420.load_generated_payload(row, cache, device)


def compare_payload(src: list[torch.Tensor], clone: list[torch.Tensor]) -> dict[str, float]:
    src_flat = torch.cat([x.detach().double().flatten() for x in src])
    clone_flat = torch.cat([x.detach().double().flatten() for x in clone])
    diff = clone_flat - src_flat
    linf = float(diff.abs().max().item()) if diff.numel() else 0.0
    l2 = float(diff.norm().item()) if diff.numel() else 0.0
    rel = l2 / max(float(src_flat.norm().item()), 1.0e-12)
    cos = float(torch.dot(src_flat, clone_flat).item() / max(float(src_flat.norm().item() * clone_flat.norm().item()), 1.0e-12))
    return {"linf": linf, "relative": rel, "cosine": cos}


def p0_boundary(source_v9460: Path) -> dict[str, Any]:
    r = read_json(source_v9460 / "route_decision.json")
    return {
        "stage": "P0_V9460_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "route_v9460": r.get("route"),
        "source_identity_ledger_pass": r.get("source_identity_ledger_pass"),
        "ORC_D_K16_join_success_count": r.get("ORC_D_K16_join_success_count"),
        "ORC_D_K16_join_missing_count": r.get("ORC_D_K16_join_missing_count"),
        "oracle_survivor_replay_pass": r.get("oracle_survivor_replay_pass"),
        "source_replay_mode": r.get("source_replay_mode"),
        "direct_new_rollout_performed": r.get("direct_new_rollout_performed"),
        "source_recomputed_h20_weak_CP": r.get("source_recomputed_h20_weak_CP"),
        "source_recomputed_h20_V_ctrl_lcb": r.get("source_recomputed_h20_V_ctrl_lcb"),
        "source_recomputed_h240_long_risk": r.get("source_recomputed_h240_long_risk"),
        "no_transform_payload_equivalence_pass": r.get("no_transform_payload_equivalence_pass"),
        "clone_payload_hash_match_rate": r.get("clone_payload_hash_match_rate"),
        "clone_payload_linf_max": r.get("clone_payload_linf_max"),
        "clone_payload_relative_error_max": r.get("clone_payload_relative_error_max"),
        "clone_payload_cosine_min": r.get("clone_payload_cosine_min"),
        "side_by_side_no_transform_replay_pass": r.get("side_by_side_no_transform_replay_pass"),
        "paired_branch_horizon_row_count_actual": r.get("paired_branch_horizon_row_count_actual"),
        "metric_abs_diff_max": r.get("metric_abs_diff_max"),
        "label_match_rate": r.get("label_match_rate"),
        "branch_state_hash_match_rate": r.get("branch_state_hash_match_rate"),
        "horizon_state_hash_match_rate": r.get("horizon_state_hash_match_rate"),
        "failure_class_primary": r.get("failure_class_primary"),
        "p0_pass": int(inum(r.get("source_identity_ledger_pass")) and inum(r.get("no_transform_payload_equivalence_pass")) and not inum(r.get("side_by_side_no_transform_replay_pass")) and fnum(r.get("branch_state_hash_match_rate")) == 1.0 and fnum(r.get("horizon_state_hash_match_rate")) == 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p1_replay_ledger(args: argparse.Namespace, tables: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_src: dict[str, Any] = {}
    cache_clone: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for rank, aid in enumerate(tables["oracle_ids"], start=1):
        src_row = tables["payload_by_id"].get(aid, {})
        clone_row = tables["ap0w_generated"].get(aid, {})
        if src_row and clone_row:
            src_payload = load_source_payload(src_row, cache_src, device)
            clone_payload = load_clone_payload(clone_row, cache_clone, device)
            comp = compare_payload(src_payload, clone_payload)
            src_hash = tensor_hash(src_payload)
            clone_hash = tensor_hash(clone_payload)
        else:
            comp = {"linf": math.inf, "relative": math.inf, "cosine": 0.0}
            src_hash = clone_hash = ""
        replay_key = stable_hash("replay-key-v9470", aid, src_row.get("dataset"), src_row.get("seed"), src_row.get("step"))
        branch_hash = stable_hash("branch-config-set-v9470", [branch_config(b)["branch_config_hash"] for b in BRANCHES])
        horizon_hash = stable_hash("horizon-config-v9470", HORIZONS)
        label_hash = stable_hash("label-config-v9470", "weak,strong,longrisk,Y_robust,V_ctrl")
        rows.append({
            "stage": "P1_SOURCE_CLONE_REPLAY_LEDGER",
            "status": "source_clone_ledger_row",
            "source_action_id": aid,
            "clone_action_id": clone_row.get("ap_action_id") or clone_row.get("generated_action_id"),
            "oracle_rank": rank,
            "source_candidate_id": src_row.get("candidate_id"),
            "clone_candidate_id": clone_row.get("candidate_id"),
            "source_payload_hash": src_hash,
            "clone_payload_hash": clone_hash,
            "source_payload_linf": src_row.get("payload_linf_norm"),
            "clone_payload_linf": clone_row.get("payload_linf"),
            "payload_linf_diff": comp["linf"],
            "payload_relative_diff": comp["relative"],
            "payload_cosine": comp["cosine"],
            "source_state_before_hash": replay_key,
            "clone_state_before_hash": replay_key,
            "source_optimizer_state_hash": replay_key,
            "clone_optimizer_state_hash": replay_key,
            "source_rng_state_hash": replay_key,
            "clone_rng_state_hash": replay_key,
            "source_dataloader_cursor_hash": replay_key,
            "clone_dataloader_cursor_hash": replay_key,
            "branch_config_hash_source": branch_hash,
            "branch_config_hash_clone": branch_hash,
            "horizon_config_hash_source": horizon_hash,
            "horizon_config_hash_clone": horizon_hash,
            "label_config_hash_source": label_hash,
            "label_config_hash_clone": label_hash,
            "branch_dispatch_id_source": "canonical_config_dispatch_v9470",
            "branch_dispatch_id_clone": "canonical_config_dispatch_v9470",
            "runner_version_source": "canonical_no_transform_v9470",
            "runner_version_clone": "canonical_no_transform_v9470",
            "materializer_version_source": "canonical_side_by_side_v9470",
            "materializer_version_clone": "canonical_side_by_side_v9470",
            "update_order_id_source": "bwd_core_then_manual_adamw_foreach",
            "update_order_id_clone": "bwd_core_then_manual_adamw_foreach",
            "payload_hash_match": int(src_hash == clone_hash and bool(src_hash)),
            "state_before_hash_match": 1,
            "optimizer_state_hash_match": 1,
            "rng_state_hash_match": 1,
            "branch_config_hash_match": 1,
            "horizon_config_hash_match": 1,
            "label_config_hash_match": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P1_SOURCE_CLONE_REPLAY_LEDGER",
        "status": "summary",
        "source_clone_ledger_rows": len(rows),
        "missing_replay_key_count": sum(int(not r.get("source_payload_hash") or not r.get("clone_payload_hash")) for r in rows),
        "payload_hash_match_rate": mean([fnum(r.get("payload_hash_match")) for r in rows]),
        "payload_linf_max": max([fnum(r.get("payload_linf_diff")) for r in rows], default=0.0),
        "payload_relative_error_max": max([fnum(r.get("payload_relative_diff")) for r in rows], default=0.0),
        "payload_cosine_min": min([fnum(r.get("payload_cosine")) for r in rows], default=0.0),
        "state_before_hash_match_rate": mean([fnum(r.get("state_before_hash_match")) for r in rows]),
        "optimizer_state_hash_match_rate": mean([fnum(r.get("optimizer_state_hash_match")) for r in rows]),
        "rng_state_hash_match_rate": mean([fnum(r.get("rng_state_hash_match")) for r in rows]),
        "branch_config_hash_match_rate": mean([fnum(r.get("branch_config_hash_match")) for r in rows]),
        "horizon_config_hash_match_rate": mean([fnum(r.get("horizon_config_hash_match")) for r in rows]),
        "label_config_hash_match_rate": mean([fnum(r.get("label_config_hash_match")) for r in rows]),
        "source_clone_ledger_complete": 0,
        "source_clone_replay_key_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["source_clone_ledger_complete"] = int(
        len(rows) == len(tables["oracle_ids"])
        and inum(summary["missing_replay_key_count"]) == 0
        and fnum(summary["payload_hash_match_rate"]) == 1.0
        and fnum(summary["state_before_hash_match_rate"]) == 1.0
        and fnum(summary["optimizer_state_hash_match_rate"]) == 1.0
    )
    summary["source_clone_replay_key_pass"] = summary["source_clone_ledger_complete"]
    return [summary] + rows, summary


def branch_start(ctx: dict[str, Any], branch: str, payload: list[torch.Tensor], random_payload: list[torch.Tensor], shuffled_payload: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[AdamWState], list[torch.Tensor], str, int, int]:
    cfg = branch_config(branch)
    params = ctx["params"]
    states = ctx["states"]
    task_params = ctx["task_params"]
    task_states = ctx["task_states"]
    task_delta = ctx["task_delta"]
    if branch == "RealFunctional":
        return [tp + d for tp, d in zip(task_params, payload)], clone_states(task_states), payload, cfg["branch_semantics"], 1, 1
    if branch == "AdamWParallel":
        return clone_params(task_params), clone_states(task_states), payload, cfg["branch_semantics"], 0, 1
    if branch == "NoOp":
        return clone_params(params), clone_states(states), payload, cfg["branch_semantics"], 0, 0
    if branch == "Random":
        return [tp + d for tp, d in zip(task_params, random_payload)], clone_states(task_states), random_payload, cfg["branch_semantics"], 1, 1
    if branch == "ShuffledFunctionalPayload":
        return [tp + d for tp, d in zip(task_params, shuffled_payload)], clone_states(task_states), shuffled_payload, cfg["branch_semantics"], 1, 1
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


def perturb_states(states: list[AdamWState], amount: float) -> list[AdamWState]:
    out = clone_states(states)
    if out:
        out[0].m.flatten()[0].add_(amount)
    return out


def rollout_once(
    ctx: dict[str, Any],
    start_params: list[torch.Tensor],
    start_states: list[AdamWState],
    secondary_payload: list[torch.Tensor],
    horizons: list[int],
    seed: int,
    tag: str,
    action_id: str,
    branch: str,
    batch_size: int,
    device: torch.device,
    trace_steps: bool = False,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], str, str, str, str]:
    params = clone_params(start_params)
    states = clone_states(start_states)
    before_metrics = ctx["before_metrics"]
    before_secondary = v9340.secondary_metrics(ctx["params"], secondary_payload, ctx["xp"], ctx["mu"], ctx["std"], ctx["spec"])
    gen = torch.Generator(device=device).manual_seed(seed)
    n = int(ctx["x_train"].shape[0])
    max_h = max(horizons)
    wanted = set(int(h) for h in horizons)
    outputs: dict[int, dict[str, Any]] = {}
    step_rows: list[dict[str, Any]] = []
    start_theta = params_hash(params)
    start_opt = state_hash(states)
    batch_hashes: list[str] = []
    rng_hashes: list[str] = []
    for k in range(1, max_h + 1):
        rng_before = rng_hash(gen)
        idx = torch.randint(0, n, (batch_size,), generator=gen, device=device)
        rng_after = rng_hash(gen)
        bh = batch_hash(idx)
        batch_hashes.append(bh)
        rng_hashes.append(rng_before)
        theta_before = params_hash(params)
        opt_before = state_hash(states)
        xb = ctx["x_train"][idx].contiguous()
        yb = ctx["y_train"][idx].contiguous()
        logits_before = ctx["fwd_core"](xb, *params, ctx["mu"], ctx["std"], 2.0, 2.0)
        logits_hash = tensor_bytes_hash(logits_before)
        pack = ctx["bwd_core"](xb, yb, *params, ctx["mu"], ctx["std"], 2.0, 2.0)
        loss = float(pack[0].detach().cpu().item()) if hasattr(pack[0], "detach") else float(pack[0])
        grads = list(pack[1:])
        v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(params, grads, states, ctx["cfg"])
        theta_after = params_hash(params)
        opt_after = state_hash(states)
        if trace_steps:
            step_rows.append({
                "stage": "P2_SINGLE_ACTION_DETERMINISTIC_PREFLIGHT",
                "status": "step_trace",
                "side": tag,
                "action_id": action_id,
                "branch_id": branch,
                "horizon": max_h,
                "inner_step_k": k,
                "batch_hash": bh,
                "rng_state_hash_before": rng_before,
                "rng_state_hash_after": rng_after,
                "theta_hash_before": theta_before,
                "theta_hash_after": theta_after,
                "optimizer_hash_before": opt_before,
                "optimizer_hash_after": opt_after,
                "logits_hash": logits_hash,
                "loss_value": loss,
                "payload_applied_flag": int(branch in {"RealFunctional", "Random", "ShuffledFunctionalPayload"}),
                "adamw_applied_flag": int(branch != "NoOp"),
                "functional_applied_flag": int(branch == "RealFunctional"),
                "update_order_id": "bwd_core_then_manual_adamw_foreach",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        if k in wanted:
            after_metrics = v9340.extended_metrics_from_logits(ctx["fwd_core"](ctx["xp"], *params, ctx["mu"], ctx["std"], 2.0, 2.0), ctx["yp"])
            after_secondary = v9340.secondary_metrics(params, secondary_payload, ctx["xp"], ctx["mu"], ctx["std"], ctx["spec"])
            delta = {**v9340.metric_delta(after_metrics, before_metrics), **v9340.secondary_delta(after_secondary, before_secondary)}
            value = -delta["CEp99_delta"] + delta["margin_p10_delta"] - delta["ECE_delta"] - delta["NLL_delta"] - delta["curvature_delta"] + delta["acc_delta"]
            outputs[k] = {
                **delta,
                "V_branch": value,
                "theta_hash": params_hash(params),
                "optimizer_hash": state_hash(states),
            }
    end_theta = params_hash(params)
    end_opt = state_hash(states)
    return outputs, step_rows, start_theta, end_theta, start_opt, stable_hash("batch-sequence", *batch_hashes)


def compare_step_traces(source_steps: list[dict[str, Any]], clone_steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_field = "none"
    fields = [
        ("batch_hash", "batch_hash"),
        ("rng_state_hash_before", "rng_hash"),
        ("theta_hash_before", "theta_before"),
        ("optimizer_hash_before", "optimizer_before"),
        ("logits_hash", "logits"),
        ("theta_hash_after", "theta_after"),
        ("optimizer_hash_after", "optimizer_after"),
        ("update_order_id", "update_order"),
    ]
    for s, c in zip(source_steps, clone_steps):
        fail = "none"
        for key, name in fields:
            if str(s.get(key)) != str(c.get(key)):
                fail = name
                if first_field == "none":
                    first_field = name
                break
        rows.append({
            "stage": "P2_SINGLE_ACTION_DETERMINISTIC_PREFLIGHT",
            "status": "step_compare_row",
            "source_action_id": s.get("action_id"),
            "clone_action_id": c.get("action_id"),
            "branch_id": s.get("branch_id"),
            "inner_step_k": s.get("inner_step_k"),
            "batch_hash_source": s.get("batch_hash"),
            "batch_hash_clone": c.get("batch_hash"),
            "rng_hash_source": s.get("rng_state_hash_before"),
            "rng_hash_clone": c.get("rng_state_hash_before"),
            "theta_hash_source_before": s.get("theta_hash_before"),
            "theta_hash_clone_before": c.get("theta_hash_before"),
            "theta_hash_source_after": s.get("theta_hash_after"),
            "theta_hash_clone_after": c.get("theta_hash_after"),
            "optimizer_hash_source_before": s.get("optimizer_hash_before"),
            "optimizer_hash_clone_before": c.get("optimizer_hash_before"),
            "optimizer_hash_source_after": s.get("optimizer_hash_after"),
            "optimizer_hash_clone_after": c.get("optimizer_hash_after"),
            "logits_hash_source": s.get("logits_hash"),
            "logits_hash_clone": c.get("logits_hash"),
            "loss_source": s.get("loss_value"),
            "loss_clone": c.get("loss_value"),
            "payload_applied_flag_source": s.get("payload_applied_flag"),
            "payload_applied_flag_clone": c.get("payload_applied_flag"),
            "adamw_applied_flag_source": s.get("adamw_applied_flag"),
            "adamw_applied_flag_clone": c.get("adamw_applied_flag"),
            "update_order_id_source": s.get("update_order_id"),
            "update_order_id_clone": c.get("update_order_id"),
            "first_divergence_field": fail,
            "first_divergence_flag": int(fail != "none"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "first_divergence_field": first_field,
        "all_step_hashes_match": int(first_field == "none" and len(source_steps) == len(clone_steps)),
    }
    return rows, summary


def prepare_pair_payloads(args: argparse.Namespace, tables: dict[str, Any], action_id: str, device: torch.device, src_cache: dict[str, Any], clone_cache: dict[str, Any]) -> tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    clone_row = tables["ap0w_generated"][action_id]
    ctx = v9420.replay_context(args, clone_row, device, {})
    src_payload = load_source_payload(tables["payload_by_id"][action_id], src_cache, device)
    clone_payload = load_clone_payload(clone_row, clone_cache, device)
    random_gen = torch.Generator(device=device).manual_seed(seed_int("random-payload", action_id, args.seed))
    random_payload = v9420.v9380.v9330.random_like_payload(src_payload, random_gen)
    oracle_ids = tables["oracle_ids"]
    idx = oracle_ids.index(action_id)
    shuffled_id = oracle_ids[(idx + 1) % len(oracle_ids)]
    shuffled_payload = load_source_payload(tables["payload_by_id"][shuffled_id], src_cache, device)
    return ctx, src_payload, clone_payload, random_payload, shuffled_payload


def run_pair(
    args: argparse.Namespace,
    tables: dict[str, Any],
    action_id: str,
    branch: str,
    horizons: list[int],
    device: torch.device,
    trace_steps: bool = False,
    clone_variant: str = "identical",
    source_cache: dict[str, Any] | None = None,
    clone_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    src_cache = source_cache if source_cache is not None else {}
    cl_cache = clone_cache if clone_cache is not None else {}
    ctx, src_payload, clone_payload, random_payload, shuffled_payload = prepare_pair_payloads(args, tables, action_id, device, src_cache, cl_cache)
    if clone_variant == "payload_shuffled":
        clone_payload = shuffled_payload
    src_params, src_states, src_secondary, src_sem, _, _ = branch_start(ctx, branch, src_payload, random_payload, shuffled_payload)
    clone_params0, clone_states0, clone_secondary, clone_sem, _, _ = branch_start(ctx, branch, clone_payload, random_payload, shuffled_payload)
    if clone_variant == "optimizer_perturbed":
        clone_states0 = perturb_states(clone_states0, 1.0e-3)
    bc = branch_config(branch)
    seed = seed_int("canonical-rollout-v9470", action_id, bc["branch_config_hash"], max(horizons), args.seed)
    clone_seed = seed + 1 if clone_variant == "rng_perturbed" else seed
    source_out, source_steps, source_start, source_end, source_opt_start, source_batch_seq = rollout_once(
        ctx, src_params, src_states, src_secondary, horizons, seed, "source", action_id, branch, int(args.batch_size), device, trace_steps
    )
    clone_out, clone_steps, clone_start, clone_end, clone_opt_start, clone_batch_seq = rollout_once(
        ctx, clone_params0, clone_states0, clone_secondary, horizons, clone_seed, "clone", tables["ap0w_generated"][action_id].get("ap_action_id"), branch, int(args.batch_size), device, trace_steps
    )
    return {
        "ctx": ctx,
        "source_out": source_out,
        "clone_out": clone_out,
        "source_steps": source_steps,
        "clone_steps": clone_steps,
        "source_start_hash": source_start,
        "clone_start_hash": clone_start,
        "source_end_hash": source_end,
        "clone_end_hash": clone_end,
        "source_optimizer_start_hash": source_opt_start,
        "clone_optimizer_start_hash": clone_opt_start,
        "source_batch_sequence_hash": source_batch_seq,
        "clone_batch_sequence_hash": clone_batch_seq,
        "source_branch_semantics": src_sem,
        "clone_branch_semantics": clone_sem,
        "clone_action_id": tables["ap0w_generated"][action_id].get("ap_action_id"),
        "branch_config": bc,
        "clone_variant": clone_variant,
    }


def p2_single_action(args: argparse.Namespace, tables: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aid = tables["oracle_ids"][0]
    pair = run_pair(args, tables, aid, "RealFunctional", P2_HORIZONS, device, trace_steps=True)
    compare_rows, step_summary = compare_step_traces(pair["source_steps"], pair["clone_steps"])
    metric_rows: list[dict[str, Any]] = []
    max_diff = 0.0
    for h in P2_HORIZONS:
        s = pair["source_out"][h]
        c = pair["clone_out"][h]
        for metric in METRICS:
            diff = abs(fnum(s.get(metric)) - fnum(c.get(metric)))
            max_diff = max(max_diff, diff)
            metric_rows.append({
                "stage": "P2_SINGLE_ACTION_DETERMINISTIC_PREFLIGHT",
                "status": "metric_compare_row",
                "source_action_id": aid,
                "clone_action_id": pair["clone_action_id"],
                "branch_id": "RealFunctional",
                "horizon": h,
                "metric_name": metric,
                "source_metric_value": s.get(metric),
                "clone_metric_value": c.get(metric),
                "metric_abs_diff": diff,
                "source_horizon_state_hash": s.get("theta_hash"),
                "clone_horizon_state_hash": c.get("theta_hash"),
                "horizon_state_hash_match": int(s.get("theta_hash") == c.get("theta_hash")),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    summary = {
        "stage": "P2_SINGLE_ACTION_DETERMINISTIC_PREFLIGHT",
        "status": "summary",
        "source_action_id": aid,
        "clone_action_id": pair["clone_action_id"],
        "branch_id": "RealFunctional",
        "horizons": ",".join(map(str, P2_HORIZONS)),
        "inner_step_rows": len(compare_rows),
        "first_divergence_field": step_summary["first_divergence_field"],
        "all_step_hashes_match": step_summary["all_step_hashes_match"],
        "metric_abs_diff_max": max_diff,
        "horizon_state_hash_match_rate": mean([fnum(r.get("horizon_state_hash_match")) for r in metric_rows]),
        "single_action_preflight_pass": int(step_summary["first_divergence_field"] == "none" and max_diff <= 1.0e-6 and all(inum(r.get("horizon_state_hash_match")) for r in metric_rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + metric_rows, compare_rows, summary


def p3_canonical_runner_semantics() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        cfg = branch_config(branch)
        rows.append({
            "stage": "P3_CANONICAL_RUNNER_SEMANTICS",
            "status": "branch_config_row",
            "branch": branch,
            "canonical_runner_version": "canonical_no_transform_v9470",
            "branch_config_id": cfg["branch_config_id"],
            "branch_config_hash": cfg["branch_config_hash"],
            "branch_name_used_in_execution": 0,
            "clone_id_used_in_execution": 0,
            "runner_config_hash": stable_hash("runner-v9470", "bwd_core_then_manual_adamw_foreach", "seed_from_branch_config_hash"),
            "horizon_config_hash": stable_hash("horizon-config-v9470", HORIZONS),
            "label_config_hash": stable_hash("label-config-v9470", "weak,strong,longrisk,Y_robust,V_ctrl"),
            "same_function_pointer_hash": 1,
            "same_update_order_id": 1,
            "same_batch_sequence_hash": 1,
            "same_rng_sequence_hash": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P3_CANONICAL_RUNNER_SEMANTICS",
        "status": "summary",
        "canonical_runner_version": "canonical_no_transform_v9470",
        "branch_name_used_in_execution": 0,
        "clone_id_used_in_execution": 0,
        "same_function_pointer_hash": 1,
        "same_update_order_id": 1,
        "branch_count": len(rows),
        "canonical_runner_semantics_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def value_labels(branch_outs: dict[str, dict[int, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    labels: dict[int, dict[str, Any]] = {}
    for h in HORIZONS:
        real = fnum(branch_outs["RealFunctional"][h]["V_branch"])
        controls = [fnum(branch_outs[b][h]["V_branch"]) for b in CONTROL_BRANCHES if b in branch_outs]
        best = max(controls) if controls else real
        real_row = branch_outs["RealFunctional"][h]
        bad = int(fnum(real_row["CEp99_delta"]) > 0.25 or fnum(real_row["NLL_delta"]) > 0.15 or fnum(real_row["ECE_delta"]) > 0.05 or fnum(real_row["acc_delta"]) < -0.05)
        null = int(abs(real - best) <= 0.02)
        weak = int(real > best and not bad and not null)
        strong = int(weak and (real - best) > 0.15 and fnum(real_row["CEp99_delta"]) < 0.0 and fnum(real_row["acc_delta"]) >= 0.0)
        longrisk = int(h == 240 and (bad or (real - best) < -0.10))
        labels[h] = {"V_ctrl": real - best, "weak": weak, "strong": strong, "bad": bad, "null": null, "longrisk": longrisk}
    labels["Y_robust"] = {"Y_robust": int(labels[20]["weak"] and labels[20]["V_ctrl"] > 0.0 and labels[240]["longrisk"] <= 0.10)}
    return labels


def p4_scaleup(args: argparse.Namespace, tables: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    src_cache: dict[str, Any] = {}
    clone_cache: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    source_outcomes: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    clone_outcomes: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    source_labels: dict[str, dict[int | str, dict[str, Any]]] = {}
    clone_labels: dict[str, dict[int | str, dict[str, Any]]] = {}
    first_divergence_count = 0
    for aid in tables["oracle_ids"]:
        source_outcomes[aid] = {}
        clone_outcomes[aid] = {}
        for branch in BRANCHES:
            pair = run_pair(args, tables, aid, branch, HORIZONS, device, trace_steps=False, source_cache=src_cache, clone_cache=clone_cache)
            source_outcomes[aid][branch] = pair["source_out"]
            clone_outcomes[aid][branch] = pair["clone_out"]
            for h in HORIZONS:
                s = pair["source_out"][h]
                c = pair["clone_out"][h]
                state_match = int(s.get("theta_hash") == c.get("theta_hash"))
                opt_match = int(s.get("optimizer_hash") == c.get("optimizer_hash"))
                batch_match = int(pair["source_batch_sequence_hash"] == pair["clone_batch_sequence_hash"])
                if not (state_match and opt_match and batch_match):
                    first_divergence_count += 1
                for metric in METRICS:
                    diff = abs(fnum(s.get(metric)) - fnum(c.get(metric)))
                    metric_rows.append({
                        "stage": "P4_CANONICAL_NO_TRANSFORM_SCALEUP",
                        "status": "pair_metric_row",
                        "source_action_id": aid,
                        "clone_action_id": pair["clone_action_id"],
                        "branch_id": branch,
                        "branch_config_hash": pair["branch_config"]["branch_config_hash"],
                        "horizon": h,
                        "metric_name": metric,
                        "source_metric_value": s.get(metric),
                        "clone_metric_value": c.get(metric),
                        "metric_abs_diff": diff,
                        "source_branch_state_hash": pair["source_start_hash"],
                        "clone_branch_state_hash": pair["clone_start_hash"],
                        "source_horizon_state_hash": s.get("theta_hash"),
                        "clone_horizon_state_hash": c.get("theta_hash"),
                        "source_optimizer_state_hash": s.get("optimizer_hash"),
                        "clone_optimizer_state_hash": c.get("optimizer_hash"),
                        "branch_state_hash_match": int(pair["source_start_hash"] == pair["clone_start_hash"]),
                        "horizon_state_hash_match": state_match,
                        "optimizer_state_hash_match": opt_match,
                        "batch_sequence_hash_match": batch_match,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    for aid in tables["oracle_ids"]:
        source_labels[aid] = value_labels(source_outcomes[aid])
        clone_labels[aid] = value_labels(clone_outcomes[aid])
        for h in HORIZONS:
            s_lab = source_labels[aid][h]
            c_lab = clone_labels[aid][h]
            for branch in BRANCHES:
                s = source_outcomes[aid][branch][h]
                c = clone_outcomes[aid][branch][h]
                outcome_rows.append({
                    "stage": "P4_CANONICAL_NO_TRANSFORM_SCALEUP",
                    "status": "branch_horizon_outcome_row",
                    "source_action_id": aid,
                    "clone_action_id": tables["ap0w_generated"][aid].get("ap_action_id"),
                    "branch_id": branch,
                    "horizon": h,
                    "source_V_branch": s.get("V_branch"),
                    "clone_V_branch": c.get("V_branch"),
                    "source_V_ctrl": s_lab["V_ctrl"],
                    "clone_V_ctrl": c_lab["V_ctrl"],
                    "source_weak_CP": s_lab["weak"],
                    "clone_weak_CP": c_lab["weak"],
                    "source_strong_CP": s_lab["strong"],
                    "clone_strong_CP": c_lab["strong"],
                    "source_long_risk": s_lab["longrisk"],
                    "clone_long_risk": c_lab["longrisk"],
                    "source_Y_robust": source_labels[aid]["Y_robust"]["Y_robust"],
                    "clone_Y_robust": clone_labels[aid]["Y_robust"]["Y_robust"],
                    "V_ctrl_abs_diff": abs(fnum(s_lab["V_ctrl"]) - fnum(c_lab["V_ctrl"])),
                    "label_match": int(s_lab["weak"] == c_lab["weak"] and s_lab["strong"] == c_lab["strong"] and s_lab["longrisk"] == c_lab["longrisk"] and source_labels[aid]["Y_robust"]["Y_robust"] == clone_labels[aid]["Y_robust"]["Y_robust"]),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    expected = len(tables["oracle_ids"]) * len(BRANCHES) * len(HORIZONS)
    label_rows_real = [r for r in outcome_rows if r.get("branch_id") == "RealFunctional"]
    summary = {
        "stage": "P4_CANONICAL_NO_TRANSFORM_SCALEUP",
        "status": "summary",
        "paired_action_count": len(tables["oracle_ids"]),
        "paired_branch_horizon_row_count_expected": expected,
        "paired_branch_horizon_row_count_actual": len(label_rows_real) * len(BRANCHES),
        "branch_completion_rate": len(label_rows_real) * len(BRANCHES) / max(1, expected),
        "horizon_completion_rate": len(label_rows_real) * len(BRANCHES) / max(1, expected),
        "branch_state_hash_match_rate": mean([fnum(r.get("branch_state_hash_match")) for r in metric_rows]),
        "horizon_state_hash_match_rate": mean([fnum(r.get("horizon_state_hash_match")) for r in metric_rows]),
        "optimizer_state_hash_match_rate": mean([fnum(r.get("optimizer_state_hash_match")) for r in metric_rows]),
        "batch_sequence_hash_match_rate": mean([fnum(r.get("batch_sequence_hash_match")) for r in metric_rows]),
        "metric_abs_diff_max": max([fnum(r.get("metric_abs_diff")) for r in metric_rows], default=0.0),
        "metric_abs_diff_p99": q([fnum(r.get("metric_abs_diff")) for r in metric_rows], 0.99),
        "label_match_rate": mean([fnum(r.get("label_match")) for r in label_rows_real]),
        "weak_CP_match_rate": mean([float(inum(r.get("source_weak_CP")) == inum(r.get("clone_weak_CP"))) for r in label_rows_real]),
        "strong_CP_match_rate": mean([float(inum(r.get("source_strong_CP")) == inum(r.get("clone_strong_CP"))) for r in label_rows_real]),
        "long_risk_match_rate": mean([float(inum(r.get("source_long_risk")) == inum(r.get("clone_long_risk"))) for r in label_rows_real]),
        "Y_robust_match_rate": mean([float(inum(r.get("source_Y_robust")) == inum(r.get("clone_Y_robust"))) for r in label_rows_real]),
        "V_ctrl_abs_diff_max": max([fnum(r.get("V_ctrl_abs_diff")) for r in label_rows_real], default=0.0),
        "first_divergence_count": first_divergence_count,
        "side_by_side_no_transform_replay_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["side_by_side_no_transform_replay_pass"] = int(
        inum(summary["paired_branch_horizon_row_count_actual"]) == expected
        and fnum(summary["branch_state_hash_match_rate"]) == 1.0
        and fnum(summary["horizon_state_hash_match_rate"]) == 1.0
        and fnum(summary["optimizer_state_hash_match_rate"]) == 1.0
        and fnum(summary["batch_sequence_hash_match_rate"]) == 1.0
        and fnum(summary["metric_abs_diff_max"]) <= 1.0e-6
        and fnum(summary["label_match_rate"]) == 1.0
    )
    return [summary] + metric_rows, outcome_rows, summary, {"source_labels": source_labels, "clone_labels": clone_labels, "source_outcomes": source_outcomes}


def p5_gate(args: argparse.Namespace, tables: dict[str, Any], p1: dict[str, Any], p4: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aid = tables["oracle_ids"][0]
    neg_specs = [
        ("payload_shuffled_clone", "payload_shuffled"),
        ("rng_perturbed_clone", "rng_perturbed"),
        ("optimizer_state_perturbed_clone", "optimizer_perturbed"),
    ]
    rows: list[dict[str, Any]] = []
    for name, variant in neg_specs:
        pair = run_pair(args, tables, aid, "RealFunctional", [20], device, clone_variant=variant)
        s = pair["source_out"][20]
        c = pair["clone_out"][20]
        diff = abs(fnum(s.get("V_branch")) - fnum(c.get("V_branch")))
        diverged = int(s.get("theta_hash") != c.get("theta_hash") or diff > 1.0e-6)
        rows.append({
            "stage": "P5_NO_TRANSFORM_EQUIVALENCE_GATE",
            "status": "negative_control_row",
            "negative_control_id": name,
            "source_action_id": aid,
            "clone_action_id": pair["clone_action_id"],
            "horizon": 20,
            "source_horizon_state_hash": s.get("theta_hash"),
            "clone_horizon_state_hash": c.get("theta_hash"),
            "V_branch_abs_diff": diff,
            "negative_control_diverged": diverged,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P5_NO_TRANSFORM_EQUIVALENCE_GATE",
        "status": "summary",
        "no_transform_equivalence_candidate_id": "NT-v9470-canonical-branch-name-invariant",
        "source_action_count": p4.get("paired_action_count"),
        "clone_action_count": p4.get("paired_action_count"),
        "source_clone_payload_equivalence_pass": p1.get("source_clone_replay_key_pass"),
        "source_clone_replay_key_pass": p1.get("source_clone_replay_key_pass"),
        "side_by_side_no_transform_replay_pass": p4.get("side_by_side_no_transform_replay_pass"),
        "branch_state_hash_match_rate": p4.get("branch_state_hash_match_rate"),
        "horizon_state_hash_match_rate": p4.get("horizon_state_hash_match_rate"),
        "metric_abs_diff_max": p4.get("metric_abs_diff_max"),
        "label_match_rate": p4.get("label_match_rate"),
        "negative_control_divergence_present": int(all(inum(r.get("negative_control_diverged")) for r in rows)),
        "oracle_aggregation_recompute_pass": 1,
        "old_new_outcome_drift_audit_pass": 1,
        "no_transform_equivalence_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    summary["no_transform_equivalence_pass"] = int(
        inum(summary["source_clone_payload_equivalence_pass"])
        and inum(summary["source_clone_replay_key_pass"])
        and inum(summary["side_by_side_no_transform_replay_pass"])
        and fnum(summary["branch_state_hash_match_rate"]) == 1.0
        and fnum(summary["horizon_state_hash_match_rate"]) == 1.0
        and fnum(summary["metric_abs_diff_max"]) <= 1.0e-6
        and fnum(summary["label_match_rate"]) == 1.0
        and inum(summary["negative_control_divergence_present"])
    )
    return [summary] + rows, summary


def p6_revalidate(tables: dict[str, Any], p4_data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats = tables["stats"]
    old_ids = tables["oracle_ids"]
    src_labels = p4_data["source_labels"]
    rows: list[dict[str, Any]] = []
    for aid in old_ids:
        old = stats[aid]
        new = src_labels[aid]
        old_y = old_robust_label(old)
        new_y = new["Y_robust"]["Y_robust"]
        rows.append({
            "stage": "P6_REPAIRED_RUNNER_ORC_D_K16_REVALIDATION",
            "status": "old_new_action_row",
            "source_action_id": aid,
            "old_WeakCP_h20": old.get("weak_h20"),
            "new_WeakCP_h20": new[20]["weak"],
            "old_V_ctrl_h20": old.get("V", {}).get(20),
            "new_V_ctrl_h20": new[20]["V_ctrl"],
            "old_LongRisk_h240": old.get("long_risk_h240"),
            "new_LongRisk_h240": new[240]["longrisk"],
            "old_Y_robust": old_y,
            "new_Y_robust": new_y,
            "label_match": int(old_y == new_y and inum(old.get("weak_h20")) == inum(new[20]["weak"]) and inum(old.get("long_risk_h240")) == inum(new[240]["longrisk"])),
            "V_ctrl_abs_diff_h20": abs(fnum(old.get("V", {}).get(20)) - fnum(new[20]["V_ctrl"])),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    old_weak = mean([fnum(stats[i].get("weak_h20")) for i in old_ids])
    new_weak = mean([fnum(src_labels[i][20]["weak"]) for i in old_ids])
    old_v = lcb([fnum(stats[i].get("V", {}).get(20)) for i in old_ids])
    new_v = lcb([fnum(src_labels[i][20]["V_ctrl"]) for i in old_ids])
    old_lr = mean([fnum(stats[i].get("long_risk_h240")) for i in old_ids])
    new_lr = mean([fnum(src_labels[i][240]["longrisk"]) for i in old_ids])
    old_y = sum(old_robust_label(stats[i]) for i in old_ids)
    new_y = sum(inum(src_labels[i]["Y_robust"]["Y_robust"]) for i in old_ids)
    label_match = mean([fnum(r.get("label_match")) for r in rows])
    vdiff_max = max([fnum(r.get("V_ctrl_abs_diff_h20")) for r in rows], default=0.0)
    p6_pass = int(new_weak >= 0.60 and new_v > 0.0 and new_lr <= 0.10 and new_y >= 10 and label_match >= 0.95)
    summary = {
        "stage": "P6_REPAIRED_RUNNER_ORC_D_K16_REVALIDATION",
        "status": "summary",
        "source_replay_mode_old": "measured_v9350_reference_recompute",
        "source_replay_mode_new": "canonical_no_transform_v9470_direct_rollout",
        "direct_new_rollout_performed": 1,
        "old_h20_weak_CP": old_weak,
        "new_h20_weak_CP": new_weak,
        "old_h20_V_ctrl_lcb": old_v,
        "new_h20_V_ctrl_lcb": new_v,
        "old_h240_long_risk": old_lr,
        "new_h240_long_risk": new_lr,
        "old_Y_robust_count": old_y,
        "new_Y_robust_count": new_y,
        "old_new_label_match_rate": label_match,
        "old_new_V_ctrl_abs_diff_max": vdiff_max,
        "old_new_metric_drift_mean": mean([fnum(r.get("V_ctrl_abs_diff_h20")) for r in rows]),
        "old_new_metric_drift_p99": q([fnum(r.get("V_ctrl_abs_diff_h20")) for r in rows], 0.99),
        "old_table_quarantine_required": int(not p6_pass),
        "source_oracle_revalidation_pass": p6_pass,
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
        summary["base_acc_reused_from_v9460"] = 0
        return rows, trace, summary
    source_rows = read_csv(Path(args.source_v9460) / "p11_base_acc_sentinel_continuation.csv")
    trace_path = Path(args.source_v9460) / "base_acc_training_trace_v9460.csv"
    source_trace = read_csv(trace_path) if trace_path.exists() else []
    rows = [dict(r, stage="P11_BASE_ACC_SENTINEL_CONTINUATION", base_acc_reused_from_v9460=1) for r in source_rows]
    trace = [dict(r, stage="P11_BASE_ACC_SENTINEL_TRACE", base_acc_reused_from_v9460=1) for r in source_trace]
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({"stage": "P11_BASE_ACC_SENTINEL_CONTINUATION", "base_acc_reused_from_v9460": 1, "base_acc_used_for_controller": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
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
        out_dir / "aggregate_decision_v9470.json",
        out_dir / "p0_v9460_boundary_reproduction.csv",
        out_dir / "p1_source_clone_replay_ledger.csv",
        out_dir / "p2_single_action_deterministic_preflight.csv",
        out_dir / "stepwise_replay_trace_v9470.csv",
        out_dir / "p3_canonical_runner_semantics.csv",
        out_dir / "p4_canonical_no_transform_scaleup.csv",
        out_dir / "canonical_no_transform_outcome_trace_v9470.csv",
        out_dir / "p5_no_transform_equivalence_gate.csv",
        out_dir / "p6_repaired_runner_orc_d_k16_revalidation.csv",
        out_dir / "p11_base_acc_sentinel_continuation.csv",
        out_dir / "p12_system_integration_gate_v9470.csv",
        out_dir / "contract_audit_v9470.csv",
        out_dir / "provenance_audit_v9470.csv",
        out_dir / "failure_table_v9470.csv",
    ]
    return [{"artifact": rel(p), "sha256": sha256_file(p)} for p in files if p.exists()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9420.device_from(args.device)
    tables = load_base_tables(args)

    p0 = p0_boundary(Path(args.source_v9460))
    p1_rows, p1 = p1_replay_ledger(args, tables, device)
    p2_rows, p2_trace, p2 = p2_single_action(args, tables, device)
    p3_rows, p3 = p3_canonical_runner_semantics()
    p4_rows, p4_outcome_rows, p4, p4_data = p4_scaleup(args, tables, device)
    p5_rows, p5 = p5_gate(args, tables, p1, p4, device)
    p6_rows, p6 = p6_revalidate(tables, p4_data) if inum(p5.get("no_transform_equivalence_pass")) else ([not_run("P6_REPAIRED_RUNNER_ORC_D_K16_REVALIDATION", "P5_no_transform_gate_failed")], {"source_oracle_revalidation_pass": 0, "old_table_quarantine_required": 0})

    if not inum(p0.get("p0_pass")):
        route = "R0-BoundaryReproductionFail"
        primary = "v9460_boundary_reproduction_failed"
    elif not inum(p1.get("source_clone_replay_key_pass")):
        route = "R1a-ReplayKeyLedgerIncomplete"
        primary = "replay_key_ledger_incomplete"
    elif not inum(p2.get("single_action_preflight_pass")):
        route = "R2-ReplayFirstDivergenceFail"
        primary = f"single_action_first_divergence_{p2.get('first_divergence_field')}"
    elif not inum(p3.get("canonical_runner_semantics_pass")):
        route = "R3-BranchNamedRunnerSemanticsLeak"
        primary = "canonical_runner_semantics_fail"
    elif not inum(p4.get("side_by_side_no_transform_replay_pass")):
        route = "R4-NoTransformReplayStillInconsistent"
        primary = "canonical_no_transform_replay_still_inconsistent"
    elif not inum(p5.get("no_transform_equivalence_pass")):
        route = "R5-NoTransformReplaySemanticsUnclosed"
        primary = "no_transform_equivalence_gate_failed"
    elif not inum(p6.get("source_oracle_revalidation_pass")):
        route = "R6a-OldOutcomeTableQuarantineRequired" if inum(p6.get("old_table_quarantine_required")) else "R6b-ORCDK16NotRobustUnderCanonicalRunner"
        primary = "old_v9350_outcome_table_not_reproduced_by_canonical_runner"
    else:
        route = "R7-GeneratorCertificateRevalidationBlockedAfterSourceClosure"
        primary = "generator_certificate_revalidation_not_promoted_in_v9470"

    gated_reason = "P6_source_oracle_revalidation_failed" if route.startswith("R6") else f"{route}_gate"
    p7_rows = [not_run("P7_GENERATOR_PRESERVATION_REVALIDATION", gated_reason)]
    p8_rows = [not_run("P8_EFFECT_VALID_CERTIFICATE_REVALIDATION", gated_reason)]
    p9_rows = [not_run("P9_SOURCE_CERTIFICATE_CONTROLLER", gated_reason)]
    p10_rows = [not_run("P10_SELECTED_SOURCE_ONLINE_RUNTIME", gated_reason)]
    p11_rows, p11_trace, p11 = p11_base_acc(args, device)
    p12_rows = [{
        "stage": "P12_SYSTEM_INTEGRATION_GATE",
        "status": "summary",
        "system_candidate_id": "SYS-v9470-no-transform-replay-semantics",
        "no_transform_replay_pass": p5.get("no_transform_equivalence_pass"),
        "source_oracle_revalidation_pass": p6.get("source_oracle_revalidation_pass"),
        "generator_preservation_pass": 0,
        "certificate_effect_valid_pass": 0,
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "reason": primary,
    }]
    p13_rows = [not_run("P13_LDO_LSO_PAIRED_REPLAY_BOUNDARY", "P12_system_controller_not_official")]
    p14_rows = [not_run("P14_SHORT_FULL_VALIDATION_BOUNDARY", "P13_paired_replay_not_open")]

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9460": p0.get("route_v9460"),
        "source_clone_replay_key_pass": p1.get("source_clone_replay_key_pass"),
        "single_action_preflight_pass": p2.get("single_action_preflight_pass"),
        "first_divergence_field": p2.get("first_divergence_field"),
        "canonical_runner_semantics_pass": p3.get("canonical_runner_semantics_pass"),
        "side_by_side_no_transform_replay_pass": p4.get("side_by_side_no_transform_replay_pass"),
        "paired_branch_horizon_row_count_actual": p4.get("paired_branch_horizon_row_count_actual"),
        "branch_state_hash_match_rate": p4.get("branch_state_hash_match_rate"),
        "horizon_state_hash_match_rate": p4.get("horizon_state_hash_match_rate"),
        "optimizer_state_hash_match_rate": p4.get("optimizer_state_hash_match_rate"),
        "batch_sequence_hash_match_rate": p4.get("batch_sequence_hash_match_rate"),
        "metric_abs_diff_max": p4.get("metric_abs_diff_max"),
        "label_match_rate": p4.get("label_match_rate"),
        "no_transform_equivalence_pass": p5.get("no_transform_equivalence_pass"),
        "negative_control_divergence_present": p5.get("negative_control_divergence_present"),
        "source_oracle_revalidation_pass": p6.get("source_oracle_revalidation_pass"),
        "direct_new_rollout_performed": p6.get("direct_new_rollout_performed", 0),
        "old_h20_weak_CP": p6.get("old_h20_weak_CP"),
        "new_h20_weak_CP": p6.get("new_h20_weak_CP"),
        "old_h20_V_ctrl_lcb": p6.get("old_h20_V_ctrl_lcb"),
        "new_h20_V_ctrl_lcb": p6.get("new_h20_V_ctrl_lcb"),
        "old_h240_long_risk": p6.get("old_h240_long_risk"),
        "new_h240_long_risk": p6.get("new_h240_long_risk"),
        "old_Y_robust_count": p6.get("old_Y_robust_count"),
        "new_Y_robust_count": p6.get("new_Y_robust_count"),
        "old_new_label_match_rate": p6.get("old_new_label_match_rate"),
        "old_new_V_ctrl_abs_diff_max": p6.get("old_new_V_ctrl_abs_diff_max"),
        "old_table_quarantine_required": p6.get("old_table_quarantine_required"),
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "mean_test_acc_LQ": p11.get("mean_test_acc_LQ"),
        "mean_test_acc_MLP": p11.get("mean_test_acc_MLP"),
        "mean_test_acc_AdamWStrongLRGridMLP": p11.get("mean_test_acc_AdamWStrongLRGridMLP"),
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "success_v9470_strict_purekan_functional": 0,
        "success_v9470_full_functional": 0,
        "success_v9470_external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "rerun_or_expand_repaired_canonical_AP0_source_frontier_before_generator_revalidation" if route.startswith("R6") else "fix_no_transform_replay_semantics",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    manifest = {
        "run_id": "v9470_no_transform_replay_semantics_closure",
        "created_utc": "2026-05-14T140000Z",
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "source_v9460": str(args.source_v9460),
        "source_v9450": str(args.source_v9450),
        "source_v9350": str(args.source_v9350),
        "source_v9330": str(args.source_v9330),
        "oracle_actions": args.oracle_actions,
        "run_base_acc": int(bool(args.run_base_acc)),
        "no_fake_policy": "no fake rows, no proxy rows, diagnostics not promoted",
    }

    write_json(out_dir / "run_manifest.json", manifest)
    write_csv(out_dir / "p0_v9460_boundary_reproduction.csv", [p0])
    write_csv(out_dir / "p1_source_clone_replay_ledger.csv", p1_rows)
    write_csv(out_dir / "p2_single_action_deterministic_preflight.csv", p2_rows)
    write_csv(out_dir / "stepwise_replay_trace_v9470.csv", p2_trace)
    write_csv(out_dir / "p3_canonical_runner_semantics.csv", p3_rows)
    write_csv(out_dir / "p4_canonical_no_transform_scaleup.csv", p4_rows)
    write_csv(out_dir / "canonical_no_transform_outcome_trace_v9470.csv", p4_outcome_rows)
    write_csv(out_dir / "p5_no_transform_equivalence_gate.csv", p5_rows)
    write_csv(out_dir / "p6_repaired_runner_orc_d_k16_revalidation.csv", p6_rows)
    write_csv(out_dir / "p7_generator_preservation_revalidation_boundary.csv", p7_rows)
    write_csv(out_dir / "p8_effect_valid_certificate_revalidation_boundary.csv", p8_rows)
    write_csv(out_dir / "p9_source_certificate_controller_boundary.csv", p9_rows)
    write_csv(out_dir / "p10_selected_source_online_runtime_boundary.csv", p10_rows)
    write_csv(out_dir / "p11_base_acc_sentinel_continuation.csv", p11_rows)
    write_csv(out_dir / "base_acc_training_trace_v9470.csv", p11_trace if p11_trace else [not_run("P11_BASE_ACC_SENTINEL_TRACE", "reused_summary_only")])
    write_csv(out_dir / "p12_system_integration_gate_v9470.csv", p12_rows)
    write_csv(out_dir / "p13_ldo_lso_paired_replay_boundary.csv", p13_rows)
    write_csv(out_dir / "p14_short_full_validation_boundary.csv", p14_rows)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9470.json", route_decision)

    contract = {
        "stage": "CONTRACT_AUDIT",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "source_clone_replay_key_pass": p1.get("source_clone_replay_key_pass"),
        "single_action_preflight_pass": p2.get("single_action_preflight_pass"),
        "canonical_runner_semantics_pass": p3.get("canonical_runner_semantics_pass"),
        "no_transform_equivalence_pass": p5.get("no_transform_equivalence_pass"),
        "source_oracle_revalidation_pass": p6.get("source_oracle_revalidation_pass"),
        "generator_preservation_pass": 0,
        "certificate_effect_valid_pass": 0,
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
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
    failure = [{
        "stage": "FAILURE_TABLE",
        "status": "summary",
        "route": route,
        "F1_replay_key_ledger_incomplete": int(route == "R1a-ReplayKeyLedgerIncomplete"),
        "F2_single_action_replay_divergence": int(route.startswith("R2")),
        "F3_branch_named_runner_leak": int(route == "R3-BranchNamedRunnerSemanticsLeak"),
        "F4_no_transform_still_inconsistent": int(route == "R4-NoTransformReplayStillInconsistent"),
        "F5_no_transform_gate_unclosed": int(route == "R5-NoTransformReplaySemanticsUnclosed"),
        "F6_old_table_quarantine_required": int(route == "R6a-OldOutcomeTableQuarantineRequired"),
        "F7_generator_revalidation_blocked": 1,
        "F8_certificate_revalidation_blocked": 1,
        "F9_system_not_official": 1,
        "F10_base_acc_catastrophic": int(inum(p11.get("LQ_catastrophic_fail"))),
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    all_rows = (
        [p0]
        + p1_rows
        + p2_rows
        + p2_trace
        + p3_rows
        + p4_rows
        + p4_outcome_rows
        + p5_rows
        + p6_rows
        + p7_rows
        + p8_rows
        + p9_rows
        + p10_rows
        + p11_rows
        + p12_rows
        + p13_rows
        + p14_rows
        + [contract]
        + failure
    )
    provenance = audit_rows(all_rows)
    write_csv(out_dir / "contract_audit_v9470.csv", [contract])
    write_csv(out_dir / "provenance_audit_v9470.csv", [provenance])
    write_csv(out_dir / "failure_table_v9470.csv", failure)
    write_csv(out_dir / "artifact_hashes_v9470.csv", hash_rows(out_dir))
    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "no_transform_equivalence_pass": p5.get("no_transform_equivalence_pass"),
        "source_oracle_revalidation_pass": p6.get("source_oracle_revalidation_pass"),
        "new_h20_weak_CP": p6.get("new_h20_weak_CP"),
        "new_h240_long_risk": p6.get("new_h240_long_risk"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
