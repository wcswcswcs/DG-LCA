#!/usr/bin/env python3
"""DG-KAN v9.7.2 exact transfer materializer runner.

The runner builds the missing exact-transfer object from landed AP0 payloads:
for each canonical AP0 action it replays the train state, computes manual
per-example gradients on held-out check samples, and records
response_linear = -g_i^T delta.  Proxy transfer is only used as a diagnostic
baseline.  No fake rows, proxy official rows, CPU offload, or loss.backward
graph are used.
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

import run_v9420_source_outcome_materializer_closure as v9420  # noqa: E402
import run_v9490_canonical_legal_observability_source_generator_certificate as v9490  # noqa: E402
import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
import run_v9700_dual_validation_existing_action_transfer_principle as v9700  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.2_ExactTransferMaterializer_CoreExpansion_DirectUpdate_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9720_exact_transfer_materializer_core_expansion_direct_update.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9710 = RESULT_ROOT / "v9710_exact_transfer_gate_core_expansion_direct_update_first_20260516T000000Z"
DEFAULT_V9700 = RESULT_ROOT / "v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

TARGET_K = 87
HORIZON_KS = [64, 77, 87, 97]
Z = 1.96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9710", default=str(DEFAULT_V9710))
    p.add_argument("--source-v9700", default=str(DEFAULT_V9700))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--preflight-samples", type=int, default=8)
    p.add_argument("--p1-sample-count", type=int, default=64)
    p.add_argument("--p1-action-limit", type=int, default=2876)
    p.add_argument("--p2-action-count", type=int, default=64)
    p.add_argument("--direct-actions-per-subspace", type=int, default=32)
    return p.parse_args()


def device_from(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def repo_rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO / p).resolve()
    else:
        p = p.resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def pstdev(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def lcb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - Z * statistics.pstdev(vals) / math.sqrt(len(vals))


def ucb(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) + Z * statistics.pstdev(vals) / math.sqrt(len(vals))


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    pos = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[pos]


def pearson(xs: list[float], ys: list[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return 0.0
    xvals = [x for x, _ in vals]
    yvals = [y for _, y in vals]
    mx, my = mean(xvals), mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in vals) / math.sqrt(vx * vy)


def rank_values(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return 0.0
    return pearson(rank_values([x for x, _ in vals]), rank_values([y for _, y in vals]))


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 820
    height = 340
    margin = 64
    maxv = max([abs(v) for v in values] + [1.0])
    zero_y = height - margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{margin}" y="32" font-family="Arial" font-size="18" fill="#111">{title}</text>',
        f'<line x1="{margin}" y1="{zero_y}" x2="{width-margin}" y2="{zero_y}" stroke="#333"/>',
    ]
    slot = (width - 2 * margin) / max(1, len(values))
    bar_w = max(14, int(slot) - 10)
    for i, (label, val) in enumerate(zip(labels, values)):
        h = (height - 2 * margin - 28) * (abs(float(val)) / maxv)
        x = margin + i * slot + 5
        y = zero_y - h if val >= 0 else zero_y
        fill = "#4b78a8" if val >= 0 else "#b45f5f"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{fill}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-margin+18}" font-family="Arial" font-size="10" fill="#333">{label[:18]}</text>')
        parts.append(f'<text x="{x:.1f}" y="{max(44, y-6):.1f}" font-family="Arial" font-size="10" fill="#333">{val:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def flat_dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return sum(float((x.detach().float() * y.detach().float()).sum().item()) for x, y in zip(a, b))


def flat_norm(a: list[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, flat_dot(a, a)))


def clone_params(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clone_states(states: list[AdamWState]) -> list[AdamWState]:
    return v9420.clone_states(states)


def response_stats(values: list[float]) -> dict[str, float]:
    mu = mean(values)
    sd = pstdev(values)
    return {
        "response_linear_mean": mu,
        "response_linear_std": sd,
        "response_linear_p10": qtile(values, 0.10),
        "response_linear_p50": qtile(values, 0.50),
        "response_linear_p90": qtile(values, 0.90),
        "transfer_lcb": mu - Z * sd / math.sqrt(max(1, len(values))),
        "transfer_snr": (mu * mu) / (sd * sd + 1.0e-12),
        "transfer_sign_frac": mean([1.0 if v > 0 else 0.0 for v in values]),
    }


def gradeab(row: dict[str, Any]) -> int:
    return v9660.gradeab(row)


def memory_fail(row: dict[str, Any]) -> int:
    return v9660.memory_fail(row)


def offdiag_fail(row: dict[str, Any]) -> int:
    return v9660.offdiag_fail(row)


def candidate_template_id(row: dict[str, Any]) -> str:
    return v9660.candidate_template_id(row)


def axis_value(row: dict[str, Any], axis: str) -> str:
    return v9660.axis_value(row, axis)


def topk_idx(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(k, len(scores))]


def quality_for_scores(ap0: list[dict[str, Any]], scores: list[float], k: int) -> dict[str, Any]:
    idx = topk_idx(scores, k)
    return quality_for_indices(ap0, idx, scores, k)


def quality_for_indices(ap0: list[dict[str, Any]], idx: list[int], scores: list[float] | None = None, k: int | None = None) -> dict[str, Any]:
    rows = [ap0[i] for i in idx]
    n = len(rows)
    full_n = len(ap0)
    member_scores = [0.0] * len(ap0)
    for i in idx:
        member_scores[i] = 1.0
    if scores is None:
        scores = member_scores
    ldo, _held, _detail = v9700.leaveout_precision_drop(scores, ap0, [axis_value(r, "dataset_id") for r in ap0], k or max(1, n))
    lso, _hs, _sd = v9700.leaveout_precision_drop(scores, ap0, [axis_value(r, "stratum_id") for r in ap0], k or max(1, n))
    lto, _ht, _td = v9700.leaveout_precision_drop(scores, ap0, [candidate_template_id(r) for r in ap0], k or max(1, n))
    ds = Counter(axis_value(r, "dataset_id") for r in rows)
    templates = Counter(candidate_template_id(r) for r in rows)
    return {
        "accepted_count": n,
        "coverage": n / max(1, full_n),
        "GradeAB_precision": mean([float(gradeab(r)) for r in rows]),
        "V_integrated_LCB": lcb([fnum(r.get("V_integrated")) for r in rows]),
        "h240_longrisk_UCB": ucb([float(inum(r.get("h240_longrisk"))) for r in rows]),
        "bad_UCB": ucb([fnum(r.get("bad_event_rate")) for r in rows]),
        "null_UCB": ucb([fnum(r.get("null_event_rate")) for r in rows]),
        "memory_fail_UCB": ucb([float(memory_fail(r)) for r in rows]),
        "offdiag_fail_UCB": ucb([float(offdiag_fail(r)) for r in rows]),
        "LDO_drop": ldo,
        "LSO_drop": lso,
        "LTO_drop": lto,
        "max_dataset_share": max((v / max(1, n) for v in ds.values()), default=0.0),
        "max_template_share": max((v / max(1, n) for v in templates.values()), default=0.0),
        "support_by_dataset": json.dumps(dict(ds), sort_keys=True),
        "support_by_template_count": len(templates),
    }


def pass_controller_like(q: dict[str, Any]) -> int:
    return int(
        inum(q.get("accepted_count")) >= TARGET_K
        and fnum(q.get("GradeAB_precision")) >= 0.75
        and fnum(q.get("V_integrated_LCB")) > 0
        and fnum(q.get("h240_longrisk_UCB")) <= 0.05
        and fnum(q.get("bad_UCB")) <= 0.05
        and fnum(q.get("null_UCB")) <= 0.15
        and fnum(q.get("LDO_drop")) <= 0.10
        and fnum(q.get("LSO_drop")) <= 0.10
        and fnum(q.get("LTO_drop")) <= 0.10
    )


def pass_weak(q: dict[str, Any]) -> int:
    return int(
        inum(q.get("accepted_count")) >= TARGET_K
        and fnum(q.get("GradeAB_precision")) >= 0.75
        and fnum(q.get("V_integrated_LCB")) > 0
        and fnum(q.get("h240_longrisk_UCB")) <= 0.10
        and fnum(q.get("LDO_drop")) <= 0.20
    )


def load_ap0_and_payloads(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, str]]]:
    ap0, score_bundle = v9660.load_ap0(args)
    payload_rows = v9490.load_payload_rows(Path(args.source_v9330))
    payload_by_id = {str(r.get("action_id")): dict(r) for r in payload_rows}
    return ap0, score_bundle, payload_by_id


def p0_boundary(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source / "route_decision_v9710.json")
    field = summary_row(read_csv(source / "p0_field_legality_audit_v9710.csv"))
    nofake = summary_row(read_csv(source / "no_fake_audit_v9710.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9720",
        "status": "summary",
        "source_route_v9710": route.get("route"),
        "exact_linear_transfer_row_count_v9710": route.get("exact_linear_transfer_row_count"),
        "exact_per_sample_gradient_available_v9710": route.get("exact_per_sample_gradient_available"),
        "exact_apply_checkpoint_available_v9710": route.get("exact_apply_checkpoint_available"),
        "generated_route_status_v9710": route.get("generated_route_status"),
        "system_legal_controller_pass_v9710": route.get("system_legal_controller_pass"),
        "field_green_count": field.get("green_count", 14),
        "field_yellow_count": field.get("yellow_count", 4),
        "field_red_count": field.get("red_count", 0),
        "fake_data_used_v9710": nofake.get("fake_data_used"),
        "proxy_row_used_v9710": nofake.get("proxy_row_used"),
        "cpu_offload_used_v9710": nofake.get("cpu_offload_used"),
        "boundary_reproduction_pass": int(
            route.get("route") == "R1-ExactTransferArtifactMissing"
            and inum(route.get("exact_linear_transfer_row_count")) == 0
            and inum(route.get("exact_per_sample_gradient_available")) == 0
            and route.get("generated_route_status") == "stopped_no_new_objective"
            and inum(nofake.get("fake_data_used")) == 0
            and inum(nofake.get("proxy_row_used")) == 0
            and inum(nofake.get("cpu_offload_used")) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9720",
        "status": "summary",
        "green_count": field.get("green_count", 14),
        "yellow_count": field.get("yellow_count", 4),
        "red_count": field.get("red_count", 0),
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_outcome_derived_field": 0,
        "uses_validation_or_test": 0,
        "field_legality_pass": int(inum(field.get("red_count", 0)) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def per_seed_task_init(args: argparse.Namespace, dataset: str, seed: int, device: torch.device) -> dict[str, Any]:
    load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
    x_train, y_train, _x_test, _y_test, input_dim, output_dim, _proto = v9420.v9380.v9320.v9248.v92._load_task(load_args, dataset, train_size=int(args.train_size), test_size=32)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    spec = v9420.v9340.lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    fwd_core, bwd_core = v9420.v9340.lq.functions_for_basis(spec.basis)
    params, mu, std = v9420.v9340.lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9380)
    states = [AdamWState.zeros_like(p) for p in params]
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
    return {
        "dataset": dataset,
        "seed": seed,
        "x_train": x_train,
        "y_train": y_train,
        "params": params,
        "states": states,
        "mu": mu,
        "std": std,
        "spec": spec,
        "fwd_core": fwd_core,
        "bwd_core": bwd_core,
        "cfg": cfg,
        "gen": gen,
    }


def dot_payload_with_grads(grads: list[torch.Tensor], payload: list[torch.Tensor]) -> float:
    return flat_dot(grads, payload)


def exact_response_materialize(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    device: torch.device,
    action_limit: int,
    sample_count: int,
    out: Path,
    stage_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, list[float]]]:
    selected = [r for r in ap0[:action_limit] if str(r.get("action_id")) in payload_by_id]
    expected = min(action_limit, len(ap0)) * sample_count
    missing_payload = min(action_limit, len(ap0)) - len(selected)
    by_ds_seed: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_ds_seed[(str(row.get("dataset")), inum(row.get("seed")))][inum(row.get("step"))].append(row)
    rows: list[dict[str, Any]] = []
    action_values: dict[str, list[float]] = defaultdict(list)
    grad_times: list[float] = []
    payload_cache: dict[str, Any] = {}
    nan_count = 0
    inf_count = 0
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for (dataset, seed), steps_map in sorted(by_ds_seed.items()):
        env = per_seed_task_init(args, dataset, seed, device)
        params: list[torch.Tensor] = env["params"]
        states: list[AdamWState] = env["states"]
        x_train: torch.Tensor = env["x_train"]
        y_train: torch.Tensor = env["y_train"]
        n_train = int(x_train.shape[0])
        max_step = max(steps_map)
        for step in range(max_step + 1):
            idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=env["gen"], device=device)
            xu = x_train[idx[: int(args.batch_size)]].contiguous()
            yu = y_train[idx[: int(args.batch_size)]].contiguous()
            xp = x_train[idx[int(args.batch_size) :]].contiguous()
            yp = y_train[idx[int(args.batch_size) :]].contiguous()
            pack = env["bwd_core"](xu, yu, *params, env["mu"], env["std"], 2.0, 2.0)
            update_grads = list(pack[1:])
            if step in steps_map:
                task_params = clone_params(params)
                task_states = clone_states(states)
                v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(task_params, update_grads, task_states, env["cfg"])
                take = min(sample_count, int(xp.shape[0]))
                grad_infos: list[dict[str, Any]] = []
                for sample_i in range(take):
                    x = xp[sample_i : sample_i + 1].contiguous()
                    y = yp[sample_i : sample_i + 1].contiguous()
                    gt0 = time.perf_counter()
                    gpack = env["bwd_core"](x, y, *task_params, env["mu"], env["std"], 2.0, 2.0)
                    grads = [g.detach() for g in list(gpack[1:])]
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    compute_ms = (time.perf_counter() - gt0) * 1000.0
                    grad_times.append(compute_ms)
                    gnorm = flat_norm(grads)
                    sample_id = stable_hash("check-sample", dataset, seed, step, sample_i, int(yp[sample_i].item()))
                    grad_hash = stable_hash("grad-signature", dataset, seed, step, sample_i, f"{gnorm:.12g}", [tuple(g.shape) for g in grads])
                    grad_infos.append(
                        {
                            "sample_i": sample_i,
                            "check_sample_id": sample_id,
                            "check_sample_group": f"{dataset}:seed{seed}:step{step}:check",
                            "check_label": int(yp[sample_i].item()),
                            "grads": grads,
                            "grad_hash": grad_hash,
                            "grad_norm": gnorm,
                            "compute_ms": compute_ms,
                        }
                    )
                for action in steps_map[step]:
                    aid = str(action.get("action_id"))
                    payload = v9490.load_payload(payload_by_id[aid], payload_cache, device)
                    payload_norm = flat_norm(payload)
                    for info in grad_infos:
                        response = -dot_payload_with_grads(info["grads"], payload)
                        if math.isnan(response):
                            nan_count += 1
                        if math.isinf(response):
                            inf_count += 1
                        action_values[aid].append(float(response))
                        rows.append(
                            {
                                "stage": stage_prefix,
                                "status": "exact_linear_response_row",
                                "action_id": aid,
                                "event_id": action.get("event_id"),
                                "dataset": action.get("dataset"),
                                "seed": action.get("seed"),
                                "step": action.get("step"),
                                "candidate_template_id": candidate_template_id(action),
                                "family_id": action.get("family_id"),
                                "step_bucket": f"step{inum(action.get('step')) // 10}",
                                "payload_hash": payload_by_id[aid].get("payload_hash_loaded") or payload_by_id[aid].get("payload_hash_expected") or action.get("payload_hash"),
                                "payload_norm": payload_norm,
                                "check_sample_id": info["check_sample_id"],
                                "check_sample_group": info["check_sample_group"],
                                "check_sample_index": info["sample_i"],
                                "check_label": info["check_label"],
                                "grad_hash": info["grad_hash"],
                                "grad_norm": info["grad_norm"],
                                "response_linear": response,
                                "response_sign": 1 if response > 0 else -1 if response < 0 else 0,
                                "response_abs": abs(response),
                                "compute_ms": info["compute_ms"],
                                "memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0,
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                        )
            v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(params, update_grads, states, env["cfg"])
    duplicate_count = len(rows) - len({(r["action_id"], r["check_sample_id"]) for r in rows})
    action_summary: dict[str, dict[str, Any]] = {}
    for action in selected:
        aid = str(action.get("action_id"))
        vals = action_values.get(aid, [])
        action_summary[aid] = {**response_stats(vals), "response_count": len(vals)}
    completion = len(rows) / max(1, expected)
    action_completion = sum(1 for vals in action_values.values() if len(vals) >= sample_count) / max(1, min(action_limit, len(ap0)))
    summary = {
        "stage": stage_prefix,
        "status": "summary",
        "action_count_requested": min(action_limit, len(ap0)),
        "action_count_materialized": len(action_values),
        "sample_count_per_action": sample_count,
        "exact_linear_transfer_row_count": len(rows),
        "expected_row_count": expected,
        "completion_rate": completion,
        "action_completion_rate": action_completion,
        "sample_completion_rate": completion,
        "missing_gradient_count": max(0, expected - len(rows)),
        "missing_payload_count": missing_payload,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_row_count": duplicate_count,
        "response_linear_mean": mean([fnum(r.get("response_linear")) for r in rows]),
        "response_linear_std": pstdev([fnum(r.get("response_linear")) for r in rows]),
        "response_linear_p10": qtile([fnum(r.get("response_linear")) for r in rows], 0.10),
        "response_linear_p50": qtile([fnum(r.get("response_linear")) for r in rows], 0.50),
        "response_linear_p90": qtile([fnum(r.get("response_linear")) for r in rows], 0.90),
        "compute_ms_q50": qtile(grad_times, 0.50),
        "compute_ms_q90": qtile(grad_times, 0.90),
        "memory_mb_peak": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0,
        "wallclock_sec": time.perf_counter() - t0,
        "materializer_strong_pass": int(completion == 1.0 and action_completion == 1.0 and missing_payload == 0 and nan_count == 0 and inf_count == 0 and duplicate_count == 0),
        "materializer_weak_pass": int(completion >= 0.95 and nan_count == 0 and inf_count == 0 and missing_payload == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, action_summary, action_values


def p0_exact_preflight(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    device: torch.device,
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if device.type != "cuda":
        row = {
            "stage": "P0_EXACT_TRANSFER_PREFLIGHT_V9720",
            "status": "summary",
            "reason": "cuda_unavailable_cpu_offload_disallowed",
            "preflight_action_count": 0,
            "preflight_sample_count": 0,
            "per_example_gradient_available": 0,
            "r_i_count": 0,
            "missing_gradient_count": 8,
            "nan_count": 0,
            "inf_count": 0,
            "exact_transfer_preflight_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [row], row
    rows, summary, _action_summary, _values = exact_response_materialize(
        args,
        ap0,
        payload_by_id,
        device,
        action_limit=1,
        sample_count=int(args.preflight_samples),
        out=out,
        stage_prefix="P0_EXACT_TRANSFER_PREFLIGHT_V9720",
    )
    summary.update(
        {
            "preflight_action_count": summary.get("action_count_materialized"),
            "preflight_sample_count": args.preflight_samples,
            "per_example_gradient_available": int(inum(summary.get("exact_linear_transfer_row_count")) == int(args.preflight_samples)),
            "r_i_count": summary.get("exact_linear_transfer_row_count"),
            "gradient_compute_ms_q50": summary.get("compute_ms_q50"),
            "gradient_compute_ms_q90": summary.get("compute_ms_q90"),
            "exact_transfer_preflight_pass": int(
                inum(summary.get("materializer_strong_pass"))
                and fnum(summary.get("compute_ms_q90")) <= 1000.0
                and fnum(summary.get("memory_mb_peak")) > 0
            ),
        }
    )
    write_bar_svg(out / "fig_p0_response_histogram.svg", "P0 response linear", [str(i) for i in range(min(8, len(rows) - 1))], [fnum(r.get("response_linear")) for r in rows[1:9]])
    write_bar_svg(out / "fig_p0_grad_compute_time.svg", "P0 gradient compute ms", ["q50", "q90"], [fnum(summary.get("compute_ms_q50")), fnum(summary.get("compute_ms_q90"))])
    return rows, summary


def p1_materializer(
    args: argparse.Namespace,
    ap0: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    device: torch.device,
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, list[float]]]:
    rows, summary, action_summary, action_values = exact_response_materialize(
        args,
        ap0,
        payload_by_id,
        device,
        action_limit=int(args.p1_action_limit),
        sample_count=int(args.p1_sample_count),
        out=out,
        stage_prefix="P1_EXACT_LINEAR_TRANSFER_MATERIALIZER_V9720",
    )
    full_summary = dict(summary)
    a8_rows = min(8, inum(summary.get("action_count_materialized"))) * min(32, int(args.p1_sample_count))
    a128_rows = min(128, inum(summary.get("action_count_materialized"))) * int(args.p1_sample_count)
    full_summary.update(
        {
            "P1a_expected_rows": 8 * 32,
            "P1a_materialized_rows": a8_rows,
            "P1a_pass": int(a8_rows == 8 * 32),
            "P1b_expected_rows": 128 * int(args.p1_sample_count),
            "P1b_materialized_rows": a128_rows,
            "P1b_pass": int(a128_rows == 128 * int(args.p1_sample_count)),
            "P1c_expected_rows": min(int(args.p1_action_limit), len(ap0)) * int(args.p1_sample_count),
            "P1c_materialized_rows": summary.get("exact_linear_transfer_row_count"),
            "P1_exact_materializer_strong_pass": summary.get("materializer_strong_pass"),
            "P1_exact_materializer_weak_pass": summary.get("materializer_weak_pass"),
        }
    )
    rows[0] = full_summary
    write_bar_svg(out / "fig_p1_response_linear_histogram.svg", "P1 response quantiles", ["p10", "p50", "p90"], [fnum(summary.get("response_linear_p10")), fnum(summary.get("response_linear_p50")), fnum(summary.get("response_linear_p90"))])
    write_bar_svg(out / "fig_p1_compute_time_by_action_size.svg", "P1 compute ms", ["q50", "q90"], [fnum(summary.get("compute_ms_q50")), fnum(summary.get("compute_ms_q90"))])
    return rows, full_summary, action_summary, action_values


def select_apply_audit_actions(ap0: list[dict[str, Any]], r5b: list[float], proxy_scores: list[float], action_summary: dict[str, dict[str, Any]], count: int) -> list[dict[str, Any]]:
    core = [r for r in ap0 if is_core_action(r)]
    by_id = {str(r.get("action_id")): r for r in ap0}
    selected: list[dict[str, Any]] = []

    def add(rows: list[dict[str, Any]], n: int) -> None:
        seen = {str(r.get("action_id")) for r in selected}
        for r in rows:
            if str(r.get("action_id")) not in seen and str(r.get("action_id")) in action_summary:
                selected.append(r)
                seen.add(str(r.get("action_id")))
            if len(seen) >= len(selected) and len(selected) >= count:
                break

    add(core[:16], 16)
    add([ap0[i] for i in topk_idx(r5b, len(ap0))], 16)
    add([ap0[i] for i in topk_idx(proxy_scores, len(ap0))], 16)
    add([ap0[i] for i in topk_idx([-s for s in r5b], len(ap0))], 16)
    if len(selected) < count:
        add([by_id[aid] for aid in sorted(action_summary) if aid in by_id], count - len(selected))
    return selected[:count]


def exact_apply_audit(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    linear_lookup: dict[str, list[float]],
    device: torch.device,
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_ds_seed: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_ds_seed[(str(row.get("dataset")), inum(row.get("seed")))][inum(row.get("step"))].append(row)
    rows: list[dict[str, Any]] = []
    payload_cache: dict[str, Any] = {}
    all_linear: list[float] = []
    all_apply: list[float] = []
    all_abs_err: list[float] = []
    t0 = time.perf_counter()
    for (dataset, seed), steps_map in sorted(by_ds_seed.items()):
        env = per_seed_task_init(args, dataset, seed, device)
        params: list[torch.Tensor] = env["params"]
        states: list[AdamWState] = env["states"]
        x_train: torch.Tensor = env["x_train"]
        y_train: torch.Tensor = env["y_train"]
        n_train = int(x_train.shape[0])
        max_step = max(steps_map)
        for step in range(max_step + 1):
            idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=env["gen"], device=device)
            xu = x_train[idx[: int(args.batch_size)]].contiguous()
            yu = y_train[idx[: int(args.batch_size)]].contiguous()
            xp = x_train[idx[int(args.batch_size) :]].contiguous()
            yp = y_train[idx[int(args.batch_size) :]].contiguous()
            pack = env["bwd_core"](xu, yu, *params, env["mu"], env["std"], 2.0, 2.0)
            update_grads = list(pack[1:])
            if step in steps_map:
                task_params = clone_params(params)
                task_states = clone_states(states)
                v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(task_params, update_grads, task_states, env["cfg"])
                take = min(int(args.p1_sample_count), int(xp.shape[0]))
                xb = xp[:take].contiguous()
                yb = yp[:take].contiguous()
                with torch.no_grad():
                    ce_before = F.cross_entropy(env["fwd_core"](xb, *task_params, env["mu"], env["std"], 2.0, 2.0), yb, reduction="none")
                for action in steps_map[step]:
                    aid = str(action.get("action_id"))
                    payload = v9490.load_payload(payload_by_id[aid], payload_cache, device)
                    after_params = [p + d for p, d in zip(task_params, payload)]
                    at0 = time.perf_counter()
                    with torch.no_grad():
                        ce_after = F.cross_entropy(env["fwd_core"](xb, *after_params, env["mu"], env["std"], 2.0, 2.0), yb, reduction="none")
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    apply_ms = (time.perf_counter() - at0) * 1000.0
                    lin_vals = linear_lookup.get(aid, [])
                    for sample_i in range(take):
                        apply_resp = float((ce_before[sample_i] - ce_after[sample_i]).item())
                        lin_resp = float(lin_vals[sample_i]) if sample_i < len(lin_vals) else 0.0
                        all_linear.append(lin_resp)
                        all_apply.append(apply_resp)
                        all_abs_err.append(abs(apply_resp - lin_resp))
                        rows.append(
                            {
                                "stage": "P2_EXACT_APPLY_AUDIT_SUBSET_V9720",
                                "status": "exact_apply_row",
                                "action_id": aid,
                                "dataset": action.get("dataset"),
                                "seed": action.get("seed"),
                                "step": action.get("step"),
                                "candidate_template_id": candidate_template_id(action),
                                "check_sample_index": sample_i,
                                "CE_before": float(ce_before[sample_i].item()),
                                "CE_after_exact_apply": float(ce_after[sample_i].item()),
                                "response_apply": apply_resp,
                                "response_linear": lin_resp,
                                "linear_apply_error": apply_resp - lin_resp,
                                "apply_compute_ms": apply_ms / max(1, take),
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                        )
            v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(params, update_grads, states, env["cfg"])
    corr = pearson(all_linear, all_apply)
    sp = spearman(all_linear, all_apply)
    sign_match = mean([1.0 if ((x > 0) == (y > 0)) else 0.0 for x, y in zip(all_linear, all_apply)])
    per_ds_corr = {}
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        per_ds_corr[ds] = pearson([fnum(r.get("response_linear")) for r in sub], [fnum(r.get("response_apply")) for r in sub])
    summary = {
        "stage": "P2_EXACT_APPLY_AUDIT_SUBSET_V9720",
        "status": "summary",
        "exact_apply_action_count": len(selected),
        "exact_apply_sample_count": len(rows),
        "linear_apply_correlation": corr,
        "linear_apply_spearman": sp,
        "linear_apply_mae": mean(all_abs_err),
        "linear_apply_sign_match_rate": sign_match,
        "apply_response_mean": mean(all_apply),
        "apply_response_std": pstdev(all_apply),
        "per_dataset_correlation": json.dumps(per_ds_corr, sort_keys=True),
        "exact_apply_compute_ms_q90": qtile([fnum(r.get("apply_compute_ms")) for r in rows], 0.90),
        "P2_apply_strong_pass": int(corr >= 0.70 and sign_match >= 0.70 and all(v >= 0.50 for v in per_ds_corr.values())),
        "P2_apply_weak_pass": int(corr >= 0.50 and sign_match >= 0.60 and sum(1 for v in per_ds_corr.values() if v >= 0.50) >= 2),
        "wallclock_sec": time.perf_counter() - t0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_linear_apply_corr_by_dataset.svg", "P2 correlation by dataset", list(per_ds_corr), list(per_ds_corr.values()))
    return [summary] + rows, summary


def is_core_action(r: dict[str, Any]) -> bool:
    return (
        gradeab(r) == 1
        and fnum(r.get("V_integrated")) > 0
        and inum(r.get("h240_longrisk")) == 0
        and fnum(r.get("bad_event_rate")) == 0
        and fnum(r.get("null_event_rate")) == 0
        and memory_fail(r) == 0
        and offdiag_fail(r) == 0
    )


def p3_scores(ap0: list[dict[str, Any]], action_summary: dict[str, dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    score_defs: dict[str, list[float]] = {
        "T0-mean-transfer": [],
        "T1-LCB-transfer": [],
        "T2-SNR-transfer": [],
        "T3-sign-agreement-transfer": [],
        "T4-core-safe-transfer": [],
    }
    for r in ap0:
        s = action_summary.get(str(r.get("action_id")), {})
        mean_s = fnum(s.get("response_linear_mean"))
        lcb_s = fnum(s.get("transfer_lcb"))
        snr_s = fnum(s.get("transfer_snr"))
        sign_s = fnum(s.get("transfer_sign_frac"))
        hard_ok = int(inum(r.get("h240_longrisk")) == 0 and memory_fail(r) == 0 and offdiag_fail(r) == 0 and fnum(r.get("bad_event_rate")) == 0 and fnum(r.get("null_event_rate")) <= 0.10)
        score_defs["T0-mean-transfer"].append(mean_s)
        score_defs["T1-LCB-transfer"].append(lcb_s)
        score_defs["T2-SNR-transfer"].append(snr_s)
        score_defs["T3-sign-agreement-transfer"].append(sign_s)
        score_defs["T4-core-safe-transfer"].append(lcb_s if hard_ok else -1.0e9)
    rows: list[dict[str, Any]] = []
    best = None
    for sid, scores in score_defs.items():
        for k in HORIZON_KS:
            q = quality_for_scores(ap0, scores, k)
            row = {
                "stage": "P3_EXACT_TRANSFER_SCORE_DEFINITIONS_V9720",
                "status": "score_topk_row",
                "score_id": sid,
                "TopK": k,
                **q,
                "score_mean": mean(scores),
                "score_std": pstdev(scores),
                "strong_pass": pass_controller_like(q) if k == TARGET_K else 0,
                "weak_pass": pass_weak(q) if k == TARGET_K else 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            rows.append(row)
            if k == TARGET_K and (best is None or (fnum(row.get("GradeAB_precision")), fnum(row.get("V_integrated_LCB")), -fnum(row.get("LDO_drop"))) > (fnum(best.get("GradeAB_precision")), fnum(best.get("V_integrated_LCB")), -fnum(best.get("LDO_drop")))):
                best = row
    strong = sum(inum(r.get("strong_pass")) for r in rows)
    weak = sum(inum(r.get("weak_pass")) for r in rows)
    best = best or rows[0]
    summary = {
        "stage": "P3_EXACT_TRANSFER_SCORE_DEFINITIONS_V9720",
        "status": "summary",
        "score_count": len(score_defs),
        "evaluation_row_count": len(rows),
        "score_strong_pass_count": strong,
        "score_weak_pass_count": weak,
        "best_score_id": best.get("score_id"),
        "best_TopK87_precision": best.get("GradeAB_precision"),
        "best_TopK87_V_LCB": best.get("V_integrated_LCB"),
        "best_TopK87_longrisk_UCB": best.get("h240_longrisk_UCB"),
        "best_TopK87_LDO_drop": best.get("LDO_drop"),
        "P3_score_strong_pass": int(strong > 0),
        "P3_score_weak_pass": int(weak > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p3_score_precision_topk87.svg", "P3 TopK87 precision", [r["score_id"] for r in rows if inum(r.get("TopK")) == 87], [fnum(r.get("GradeAB_precision")) for r in rows if inum(r.get("TopK")) == 87])
    return [summary] + rows, summary, score_defs


def p4_core_expansion(ap0: list[dict[str, Any]], score_defs: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core_idx = [i for i, r in enumerate(ap0) if is_core_action(r)]
    rows: list[dict[str, Any]] = []
    best = None
    for sid in ["T1-LCB-transfer", "T2-SNR-transfer", "T3-sign-agreement-transfer", "T4-core-safe-transfer"]:
        scores = score_defs[sid]
        pool = [
            i
            for i, r in enumerate(ap0)
            if i not in set(core_idx)
            and inum(r.get("h240_longrisk")) == 0
            and memory_fail(r) == 0
            and offdiag_fail(r) == 0
            and fnum(r.get("bad_event_rate")) == 0
            and fnum(r.get("null_event_rate")) <= 0.10
        ]
        expansion = sorted(pool, key=lambda i: scores[i], reverse=True)[: max(0, TARGET_K - len(core_idx))]
        accepted = core_idx + expansion
        q = quality_for_indices(ap0, accepted, [1.0 if i in set(accepted) else 0.0 for i in range(len(ap0))], len(accepted))
        exp_rows = [ap0[i] for i in expansion]
        row = {
            "stage": "P4_CORE77_EXACT_TRANSFER_EXPANSION_V9720",
            "status": "candidate_row",
            "score_id": sid,
            "core_count": len(core_idx),
            "expansion_count": len(expansion),
            "accepted_count": len(accepted),
            "expansion_action_ids": ",".join(str(ap0[i].get("action_id")) for i in expansion),
            "expansion_dataset_distribution": json.dumps(dict(Counter(axis_value(r, "dataset_id") for r in exp_rows)), sort_keys=True),
            "expansion_template_distribution_count": len(Counter(candidate_template_id(r) for r in exp_rows)),
            "expansion_transfer_score_mean": mean([scores[i] for i in expansion]),
            **q,
            "strong_pass": pass_controller_like(q),
            "weak_pass": pass_weak(q),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if best is None or (inum(row.get("strong_pass")), inum(row.get("weak_pass")), fnum(row.get("GradeAB_precision")), fnum(row.get("V_integrated_LCB")), -fnum(row.get("LDO_drop"))) > (inum(best.get("strong_pass")), inum(best.get("weak_pass")), fnum(best.get("GradeAB_precision")), fnum(best.get("V_integrated_LCB")), -fnum(best.get("LDO_drop"))):
            best = row
    best = best or rows[0]
    summary = {
        "stage": "P4_CORE77_EXACT_TRANSFER_EXPANSION_V9720",
        "status": "summary",
        "candidate_count": len(rows),
        "strong_pass_count": sum(inum(r.get("strong_pass")) for r in rows),
        "weak_pass_count": sum(inum(r.get("weak_pass")) for r in rows),
        "core_count": len(core_idx),
        "needed_expansion_to_87": max(0, TARGET_K - len(core_idx)),
        "best_score_id": best.get("score_id"),
        "best_accepted_count": best.get("accepted_count"),
        "best_GradeAB_precision": best.get("GradeAB_precision"),
        "best_V_integrated_LCB": best.get("V_integrated_LCB"),
        "best_longrisk_UCB": best.get("h240_longrisk_UCB"),
        "best_LDO_drop": best.get("LDO_drop"),
        "P4_core_expansion_pass": int(sum(inum(r.get("strong_pass")) + inum(r.get("weak_pass")) for r in rows) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p4_core_expansion_precision.svg", "P4 precision", [r["score_id"] for r in rows], [fnum(r.get("GradeAB_precision")) for r in rows])
    return [summary] + rows, summary


def p5_compare(ap0: list[dict[str, Any]], r5b: list[float], proxy_scores: list[float], score_defs: dict[str, list[float]], p4: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = {
        "old-R8A/R5B": r5b,
        "proxy-TransferLCB": proxy_scores,
        "exact-T1-LCB": score_defs["T1-LCB-transfer"],
        "exact-T2-SNR": score_defs["T2-SNR-transfer"],
        "exact-T3-sign": score_defs["T3-sign-agreement-transfer"],
        "exact-T4-core-safe": score_defs["T4-core-safe-transfer"],
    }
    rows: list[dict[str, Any]] = []
    best_exact = None
    for mid, scores in methods.items():
        for k in [64, 87, 97]:
            q = quality_for_scores(ap0, scores, k)
            row = {
                "stage": "P5_EXACT_TRANSFER_VS_OLD_PROXY_V9720",
                "status": "method_topk_row",
                "method_id": mid,
                "TopK": k,
                **q,
                "PSI_mean": mean([v9700.psi_kl_wasserstein([scores[i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds], scores)[0] for ds in sorted({axis_value(r, "dataset_id") for r in ap0})]),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            rows.append(row)
            if mid.startswith("exact") and k == 87 and (best_exact is None or (fnum(row.get("GradeAB_precision")), fnum(row.get("V_integrated_LCB")), -fnum(row.get("LDO_drop"))) > (fnum(best_exact.get("GradeAB_precision")), fnum(best_exact.get("V_integrated_LCB")), -fnum(best_exact.get("LDO_drop")))):
                best_exact = row
    old = next(r for r in rows if r.get("method_id") == "old-R8A/R5B" and inum(r.get("TopK")) == 87)
    proxy = next(r for r in rows if r.get("method_id") == "proxy-TransferLCB" and inum(r.get("TopK")) == 87)
    best_exact = best_exact or rows[0]
    strong = int(fnum(best_exact.get("GradeAB_precision")) >= 0.75 and fnum(best_exact.get("V_integrated_LCB")) > 0 and fnum(best_exact.get("h240_longrisk_UCB")) <= 0.05 and fnum(best_exact.get("LDO_drop")) <= 0.10)
    useful = int(fnum(best_exact.get("GradeAB_precision")) >= 0.60 and fnum(best_exact.get("V_integrated_LCB")) > 0 and fnum(best_exact.get("LDO_drop")) <= fnum(old.get("LDO_drop")) - 0.15)
    exact_fail = int(fnum(best_exact.get("GradeAB_precision")) <= fnum(proxy.get("GradeAB_precision")) + 0.10 or fnum(best_exact.get("V_integrated_LCB")) < 0 or fnum(best_exact.get("LDO_drop")) >= fnum(old.get("LDO_drop")) - 0.02)
    summary = {
        "stage": "P5_EXACT_TRANSFER_VS_OLD_PROXY_V9720",
        "status": "summary",
        "method_count": len(methods),
        "best_exact_method": best_exact.get("method_id"),
        "best_exact_precision": best_exact.get("GradeAB_precision"),
        "best_exact_V_LCB": best_exact.get("V_integrated_LCB"),
        "best_exact_longrisk_UCB": best_exact.get("h240_longrisk_UCB"),
        "best_exact_LDO_drop": best_exact.get("LDO_drop"),
        "old_rank_precision": old.get("GradeAB_precision"),
        "old_rank_LDO_drop": old.get("LDO_drop"),
        "proxy_transfer_precision": proxy.get("GradeAB_precision"),
        "proxy_transfer_LDO_drop": proxy.get("LDO_drop"),
        "exact_transfer_strong_pass": strong,
        "exact_transfer_useful_diagnostic": useful,
        "exact_transfer_fail": exact_fail,
        "P5_exact_better_than_proxy": int(not exact_fail and (strong or useful)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p5_method_precision_ldo.svg", "P5 TopK87 precision", [r["method_id"] for r in rows if inum(r.get("TopK")) == 87], [fnum(r.get("GradeAB_precision")) for r in rows if inum(r.get("TopK")) == 87])
    return [summary] + rows, summary


def p6_dataset_shift(ap0: list[dict[str, Any]], score_defs: dict[str, list[float]], best_score_id: str, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = score_defs.get(best_score_id, score_defs["T1-LCB-transfer"])
    rows: list[dict[str, Any]] = []
    all_psi = []
    precisions = []
    v_lcbs = []
    for ds in sorted({axis_value(r, "dataset_id") for r in ap0}):
        idx = [i for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]
        local_scores = [scores[i] for i in idx]
        local_rows = [ap0[i] for i in idx]
        local_top = topk_idx(local_scores, min(TARGET_K, len(idx)))
        selected = [local_rows[i] for i in local_top]
        psi = v9700.psi_kl_wasserstein(local_scores, scores)[0]
        prec = mean([float(gradeab(r)) for r in selected])
        vl = lcb([fnum(r.get("V_integrated")) for r in selected])
        long = ucb([float(inum(r.get("h240_longrisk"))) for r in selected])
        core_count = sum(1 for r in local_rows if is_core_action(r))
        rows.append(
            {
                "stage": "P6_EXACT_TRANSFER_DATASET_SHIFT_AUDIT_V9720",
                "status": "dataset_row",
                "dataset": ds,
                "score_id": best_score_id,
                "score_mean": mean(local_scores),
                "score_std": pstdev(local_scores),
                "TopK_accepted_count": len(selected),
                "TopK_precision": prec,
                "V_LCB": vl,
                "longrisk_UCB": long,
                "target_density": sum(gradeab(r) for r in local_rows) / max(1, len(local_rows)),
                "core_coverage": core_count / max(1, len(local_rows)),
                "expansion_coverage": max(0, len(selected) - core_count) / max(1, len(local_rows)),
                "PSI": psi,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
        all_psi.append(psi)
        precisions.append(prec)
        v_lcbs.append(vl)
    macro_v = mean(v_lcbs)
    psi_mean = mean(all_psi)
    solved = int(min(precisions or [0]) >= 0.70 and (min(v_lcbs or [0]) >= 0 or macro_v > 0) and psi_mean <= 0.15)
    unresolved = int(max(all_psi or [0]) > 0.30 or min(precisions or [1]) < 0.55)
    summary = {
        "stage": "P6_EXACT_TRANSFER_DATASET_SHIFT_AUDIT_V9720",
        "status": "summary",
        "score_id": best_score_id,
        "dataset_count": len(rows),
        "PSI_mean": psi_mean,
        "per_dataset_precision_min": min(precisions or [0]),
        "per_dataset_V_LCB_min": min(v_lcbs or [0]),
        "macro_V_LCB": macro_v,
        "dataset_shift_solved": solved,
        "dataset_shift_unresolved": unresolved,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p6_per_dataset_precision.svg", "P6 per-dataset precision", [r["dataset"] for r in rows], [fnum(r.get("TopK_precision")) for r in rows])
    return [summary] + rows, summary


def p7_direct_update(p2: dict[str, Any], p3: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p2.get("P2_apply_weak_pass")) and inum(p3.get("P3_score_weak_pass"))):
        row = {
            "stage": "P7_DIRECT_TRANSFER_SOLVED_SMALL_UPDATE_V9720",
            "status": "not_run",
            "reason": "P2_or_P3_weak_pass_failed",
            "subspace_count": 6,
            "generated_action_count": 0,
            "branch_horizon_rows": 0,
            "direct_update_weak_pass": 0,
            "direct_update_strong_pass": 0,
            "new_objective_evidence_present": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [row], row
    row = {
        "stage": "P7_DIRECT_TRANSFER_SOLVED_SMALL_UPDATE_V9720",
        "status": "not_run",
        "reason": "direct_update_materializer_requires_separate_branch_horizon_runner",
        "subspace_count": 6,
        "generated_action_count": 0,
        "branch_horizon_rows": 0,
        "direct_update_weak_pass": 0,
        "direct_update_strong_pass": 0,
        "new_objective_evidence_present": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def boundary_not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, **extra, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return [row], row


def base_acc(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source / "base_acc_sentinel_v9710.csv")]
    for row in rows:
        row["stage"] = "BASE_ACC_SENTINEL_V9720"
        row["base_acc_reused_from_v9710"] = 1
        row["base_acc_used_for_controller"] = 0
    return rows, summary_row(rows)


def count_artifact_rows(paths: list[Path]) -> dict[str, Any]:
    rows_checked = 0
    fake_proxy = 0
    fake = 0
    proxy = 0
    cpu = 0
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        for row in read_csv(path):
            rows_checked += 1
            fake += inum(row.get("fake_data_used"))
            proxy += inum(row.get("proxy_row_used"))
            cpu += inum(row.get("cpu_offload_used"))
            fake_proxy += int(inum(row.get("fake_data_used")) or inum(row.get("proxy_row_used")) or inum(row.get("cpu_offload_used")))
    return {
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_proxy,
        "fake_data_used": fake,
        "proxy_row_used": proxy,
        "cpu_offload_used": cpu,
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


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

    def dump_json(name: str, obj: dict[str, Any]) -> Path:
        path = out / name
        write_json(path, obj)
        artifacts[name] = path
        return path

    source_v9710 = Path(args.source_v9710)
    device = device_from(args.device)
    ap0, score_bundle, payload_by_id = load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    p1_proxy_rows = read_csv(Path(args.source_v9700) / "p1_cross_sample_transfer_ledger_v9700.csv")
    proxy_by_action = {r.get("action_id"): fnum(r.get("transfer_lcb_batch")) for r in p1_proxy_rows if r.get("status") == "transfer_action_row"}
    proxy_scores = [proxy_by_action.get(str(r.get("action_id")), -1.0e9) for r in ap0]

    p0_rows, p0_field_rows, p0 = p0_boundary(source_v9710)
    dump_csv("p0_boundary_reproduction_v9720.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9720.csv", p0_field_rows)
    pre_rows, pre = p0_exact_preflight(args, ap0, payload_by_id, device, out)
    dump_csv("p0_exact_transfer_preflight_1action_v9720.csv", pre_rows)

    if not inum(pre.get("exact_transfer_preflight_pass")):
        p1_rows = [{"stage": "P1_EXACT_LINEAR_TRANSFER_MATERIALIZER_V9720", "status": "not_run", "reason": "P0_exact_preflight_failed", "P1_exact_materializer_weak_pass": 0, "P1_exact_materializer_strong_pass": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}]
        p1 = p1_rows[0]
        action_summary: dict[str, dict[str, Any]] = {}
        action_values: dict[str, list[float]] = {}
    else:
        p1_rows, p1, action_summary, action_values = p1_materializer(args, ap0, payload_by_id, device, out)
    dump_csv("p1_exact_linear_transfer_materializer_v9720.csv", p1_rows)

    if inum(p1.get("P1_exact_materializer_weak_pass")):
        audit_actions = select_apply_audit_actions(ap0, r5b, proxy_scores, action_summary, int(args.p2_action_count))
        p2_rows, p2 = exact_apply_audit(args, audit_actions, payload_by_id, action_values, device, out)
    else:
        p2_rows, p2 = boundary_not_run("P2_EXACT_APPLY_AUDIT_SUBSET_V9720", "P1_exact_materializer_weak_pass_failed", P2_apply_weak_pass=0, P2_apply_strong_pass=0)
    dump_csv("p2_exact_apply_audit_subset_v9720.csv", p2_rows)

    if inum(p1.get("P1_exact_materializer_weak_pass")):
        p3_rows, p3, score_defs = p3_scores(ap0, action_summary, out)
        p4_rows, p4 = p4_core_expansion(ap0, score_defs, out)
        p5_rows, p5 = p5_compare(ap0, r5b, proxy_scores, score_defs, p4, out)
        best_score = str(p3.get("best_score_id") or "T1-LCB-transfer")
        p6_rows, p6 = p6_dataset_shift(ap0, score_defs, best_score, out)
    else:
        score_defs = {"T1-LCB-transfer": [0.0 for _ in ap0]}
        p3_rows, p3 = boundary_not_run("P3_EXACT_TRANSFER_SCORE_DEFINITIONS_V9720", "P1_exact_materializer_weak_pass_failed", P3_score_weak_pass=0, P3_score_strong_pass=0)
        p4_rows, p4 = boundary_not_run("P4_CORE77_EXACT_TRANSFER_EXPANSION_V9720", "P1_exact_materializer_weak_pass_failed", P4_core_expansion_pass=0)
        p5_rows, p5 = boundary_not_run("P5_EXACT_TRANSFER_VS_OLD_PROXY_V9720", "P1_exact_materializer_weak_pass_failed", P5_exact_better_than_proxy=0, exact_transfer_fail=1)
        p6_rows, p6 = boundary_not_run("P6_EXACT_TRANSFER_DATASET_SHIFT_AUDIT_V9720", "P1_exact_materializer_weak_pass_failed", dataset_shift_solved=0)
    dump_csv("p3_exact_transfer_score_definitions_v9720.csv", p3_rows)
    dump_csv("p4_core77_exact_transfer_expansion_v9720.csv", p4_rows)
    dump_csv("p5_exact_transfer_vs_old_proxy_v9720.csv", p5_rows)
    dump_csv("p6_exact_transfer_dataset_shift_audit_v9720.csv", p6_rows)

    p7_rows, p7 = p7_direct_update(p2, p3)
    dump_csv("p7_direct_transfer_solved_small_update_v9720.csv", p7_rows)

    if inum(p4.get("P4_core_expansion_pass")) or inum(p5.get("exact_transfer_strong_pass")):
        p8_rows, p8 = boundary_not_run("P8_MINIMAL_EXISTING_ACTION_CONTROLLER_V9720", "controller_freeze_not_implemented_without_certificate_runner", controller_pass=0)
    else:
        p8_rows, p8 = boundary_not_run("P8_MINIMAL_EXISTING_ACTION_CONTROLLER_V9720", "P4_or_P5_weak_pass_failed", controller_pass=0)
    dump_csv("p8_minimal_existing_action_controller_v9720.csv", p8_rows)
    p9_rows, p9 = boundary_not_run("P9_SELECTED_RUNTIME_PREFLIGHT_V9720", "P8_controller_not_selected", selected_runtime_pass=0)
    dump_csv("p9_selected_runtime_preflight_v9720.csv", p9_rows)
    p10_rows, p10 = boundary_not_run("p10_SYSTEM_BOUNDARY_V9720", "P8_or_P9_not_passed", system_legal_controller_pass=0, paired_replay_opened=0)
    dump_csv("p10_system_boundary_v9720.csv", p10_rows)
    p11_rows, p11 = boundary_not_run("P11_PAIRED_REPLAY_BOUNDARY_V9720", "P10_system_not_official", paired_replay_pass=0)
    dump_csv("p11_paired_replay_boundary_v9720.csv", p11_rows)
    p12_rows, p12 = boundary_not_run("P12_SHORT_FULL_BOUNDARY_V9720", "P11_paired_replay_not_open", short_run_boundary_open=0, full_run_boundary_open=0)
    dump_csv("p12_short_full_boundary_v9720.csv", p12_rows)
    base_rows, base_summary = base_acc(source_v9710)
    dump_csv("base_acc_sentinel_v9720.csv", base_rows)

    if not inum(pre.get("exact_transfer_preflight_pass")):
        route = "R0-ExactTransferPreflightMissing"
        primary = "exact_transfer_preflight_missing"
    elif fnum(p1.get("completion_rate")) < 0.95:
        route = "R1-ExactTransferMaterializerIncomplete"
        primary = "exact_transfer_materializer_incomplete"
    elif fnum(p2.get("linear_apply_correlation")) < 0.50 or fnum(p2.get("linear_apply_sign_match_rate")) < 0.60:
        route = "R2-LinearTransferNotPredictive"
        primary = "linear_transfer_not_predictive"
    elif inum(p4.get("P4_core_expansion_pass")):
        route = "R5-CoreExpansionSolvedControllerCandidateReady"
        primary = "controller_candidate_ready"
    elif inum(p7.get("direct_update_weak_pass")) or inum(p7.get("direct_update_strong_pass")):
        route = "R6-DirectTransferSolvedUpdatePromising"
        primary = "direct_transfer_solved_update_promising"
    elif inum(p5.get("exact_transfer_fail")):
        route = "R3-ExactTransferNotBetterThanProxy"
        primary = "exact_transfer_not_better_than_proxy"
    else:
        route = "R4-ExactTransferImprovesStabilityButQualityLow"
        primary = "exact_transfer_quality_or_stability_insufficient"
    if inum(p8.get("controller_pass")) and inum(p9.get("selected_runtime_pass")) and inum(p10.get("system_legal_controller_pass")):
        route = "R7-SystemPass"
        primary = "none"
    secondary = "generated_route_stopped_no_new_objective"

    route_row = {
        "stage": "P13_ROUTE_DECISION_V9720",
        "status": "summary",
        "route": route,
        "source_route_v9710": p0.get("source_route_v9710"),
        "p0_boundary_pass": p0.get("boundary_reproduction_pass"),
        "exact_transfer_preflight_pass": pre.get("exact_transfer_preflight_pass"),
        "P1_exact_materializer_strong_pass": p1.get("P1_exact_materializer_strong_pass"),
        "P1_exact_materializer_weak_pass": p1.get("P1_exact_materializer_weak_pass"),
        "exact_linear_transfer_row_count": p1.get("exact_linear_transfer_row_count", 0),
        "P2_apply_strong_pass": p2.get("P2_apply_strong_pass", 0),
        "P2_apply_weak_pass": p2.get("P2_apply_weak_pass", 0),
        "linear_apply_correlation": p2.get("linear_apply_correlation", ""),
        "linear_apply_sign_match_rate": p2.get("linear_apply_sign_match_rate", ""),
        "P3_score_strong_pass": p3.get("P3_score_strong_pass", 0),
        "P3_score_weak_pass": p3.get("P3_score_weak_pass", 0),
        "best_score_id": p3.get("best_score_id", ""),
        "best_TopK87_precision": p3.get("best_TopK87_precision", ""),
        "best_TopK87_LDO_drop": p3.get("best_TopK87_LDO_drop", ""),
        "P4_core_expansion_pass": p4.get("P4_core_expansion_pass", 0),
        "best_core_expansion_score_id": p4.get("best_score_id", ""),
        "best_core_expansion_precision": p4.get("best_GradeAB_precision", ""),
        "best_core_expansion_LDO_drop": p4.get("best_LDO_drop", ""),
        "P5_exact_better_than_proxy": p5.get("P5_exact_better_than_proxy", 0),
        "exact_transfer_fail": p5.get("exact_transfer_fail", 0),
        "dataset_shift_solved": p6.get("dataset_shift_solved", 0),
        "direct_update_weak_pass": p7.get("direct_update_weak_pass", 0),
        "direct_update_strong_pass": p7.get("direct_update_strong_pass", 0),
        "controller_pass": p8.get("controller_pass", 0),
        "selected_runtime_pass": p9.get("selected_runtime_pass", 0),
        "system_legal_controller_pass": p10.get("system_legal_controller_pass", 0),
        "generated_route_status": "stopped_no_new_objective",
        "APGU_APGV_APGW_allowed": 0,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9720.json", route_row)
    allowed = {
        "stage": "ALLOWED_NEXT_GATES_V9720",
        "status": "summary",
        "selected_runtime_allowed": int(inum(p8.get("controller_pass"))),
        "paired_replay_allowed": int(inum(p10.get("system_legal_controller_pass"))),
        "short_full_allowed": int(inum(p11.get("paired_replay_pass"))),
        "generated_route_allowed": int(inum(p7.get("direct_update_weak_pass")) or inum(p7.get("direct_update_strong_pass"))),
        "APGU_APGV_APGW_allowed": 0,
    }
    stop = {
        "stage": "STOP_CONDITIONS_V9720",
        "status": "summary",
        "stop_generated_blind_variants": 1,
        "stop_transfer_route": int(route in {"R2-LinearTransferNotPredictive", "R3-ExactTransferNotBetterThanProxy"}),
        "fix_materializer": int(route in {"R0-ExactTransferPreflightMissing", "R1-ExactTransferMaterializerIncomplete"}),
        "enter_system": int(route == "R7-SystemPass"),
    }
    dump_json("allowed_next_gates_v9720.json", allowed)
    dump_json("stop_conditions_v9720.json", stop)

    nofake = count_artifact_rows(list(artifacts.values()))
    nofake_row = {"stage": "NO_FAKE_AUDIT_V9720", "status": "summary", **nofake}
    dump_csv("no_fake_audit_v9720.csv", [nofake_row])
    contract = {
        "stage": "CONTRACT_AUDIT_V9720",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "v9710_boundary_pass": p0.get("boundary_reproduction_pass"),
        "exact_preflight_pass": pre.get("exact_transfer_preflight_pass"),
        "exact_materializer_pass": p1.get("P1_exact_materializer_weak_pass", 0),
        "exact_apply_pass": int(inum(p2.get("P2_apply_weak_pass", 0)) or inum(p2.get("P2_apply_strong_pass", 0))),
        "core_expansion_pass": p4.get("P4_core_expansion_pass", 0),
        "direct_update_pass": int(inum(p7.get("direct_update_weak_pass", 0)) or inum(p7.get("direct_update_strong_pass", 0))),
        "controller_runtime_system": f"{p8.get('controller_pass',0)}/{p9.get('selected_runtime_pass',0)}/{p10.get('system_legal_controller_pass',0)}",
        "generated_route_stop": 1,
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": base_summary.get("base_acc_used_for_controller"),
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "proxy_transfer_promoted_to_official": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
    }
    dump_csv("contract_audit_v9720.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9720",
        "status": "summary",
        "route": route,
        "F0_exact_transfer_preflight_missing": int(route == "R0-ExactTransferPreflightMissing"),
        "F1_exact_transfer_materializer_incomplete": int(route == "R1-ExactTransferMaterializerIncomplete"),
        "F2_linear_transfer_not_predictive": int(route == "R2-LinearTransferNotPredictive"),
        "F3_exact_transfer_not_better_than_proxy": int(route == "R3-ExactTransferNotBetterThanProxy"),
        "F4_exact_transfer_quality_low": int(route == "R4-ExactTransferImprovesStabilityButQualityLow"),
        "F5_core_expansion_solved": int(route == "R5-CoreExpansionSolvedControllerCandidateReady"),
        "F6_direct_update_promising": int(route == "R6-DirectTransferSolvedUpdatePromising"),
        "F7_system_pass": int(route == "R7-SystemPass"),
        "F8_system_not_official": int(not inum(p10.get("system_legal_controller_pass", 0))),
        "F9_base_acc_catastrophic": inum(base_summary.get("LQ_catastrophic_fail")),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
    }
    dump_csv("failure_taxonomy_v9720.csv", [failure])

    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifact_hashes = {name: sha256_file(path) for name, path in artifacts.items() if path.exists()}
    manifest = {
        "version": "v9720",
        "created_utc": "2026-05-16T010000Z",
        "out_dir": repo_rel(out),
        "seed": args.seed,
        "device": str(device),
        "data_root": args.data_root,
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
        "wallclock_sec": time.perf_counter() - t0,
        "sources": {
            "v9710": repo_rel(args.source_v9710),
            "v9700": repo_rel(args.source_v9700),
            "v9550": repo_rel(args.source_v9550),
            "v9560": repo_rel(args.source_v9560),
            "v9570": repo_rel(args.source_v9570),
            "v9580": repo_rel(args.source_v9580),
            "v9330": repo_rel(args.source_v9330),
        },
        "artifact_sha256": artifact_hashes,
    }
    dump_json("run_manifest_v9720.json", manifest)

    print(
        json.dumps(
            {
                "route": route,
                "primary_blocker": primary,
                "exact_linear_transfer_row_count": p1.get("exact_linear_transfer_row_count", 0),
                "linear_apply_correlation": p2.get("linear_apply_correlation", ""),
                "best_score_id": p3.get("best_score_id", ""),
                "best_TopK87_precision": p3.get("best_TopK87_precision", ""),
                "best_core_expansion_precision": p4.get("best_GradeAB_precision", ""),
                "direct_update_generated_actions": p7.get("generated_action_count", 0),
                "system_legal_controller_pass": p10.get("system_legal_controller_pass", 0),
                "out_dir": repo_rel(out),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
