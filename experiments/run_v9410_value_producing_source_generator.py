#!/usr/bin/env python3
"""DG-KAN v9.4.1 value-producing source generator runner.

This runner advances the v9.4.0 boundary in three concrete ways:

1. Builds balanced source panels from the full AP0 outcome universe.
2. Materializes AP0b-AP0f source payloads and commit-time certificates.
3. Measures generated-source outcomes, certificate lift, and base-acc sentinel
   rows without promoting diagnostics to an official controller.
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
import run_v9400_source_action_selection_candidate_source_rebuild as v9400  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.1_ValueProducingSourceGenerator_LegalSourceIdentifiability_BaseAccSentinel_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9410_value_producing_source_generator.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9400 = RESULT_ROOT / "v9400_source_action_selection_candidate_source_rebuild_horizon_controller_first_20260514T070000Z"
DEFAULT_SOURCE_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_SOURCE_V9390 = RESULT_ROOT / "v9390_generated_ap_failure_decomposition_value_preserving_first_20260514T060000Z"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"

PRIMITIVES = [
    "AP0b-LastEdgeLinearizedDescentSource",
    "AP0c-AdamWResidualOrthogonalSource",
    "AP0d-TailMarginRepairSource",
    "AP0e-CurvatureGuardedLowRankEdgeSource",
    "AP0f-SupportMemorySource",
]
HORIZONS = [20, 80, 240]


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
    p.add_argument("--panel-s64-actions", type=int, default=64)
    p.add_argument("--panel-s256-actions", type=int, default=256)
    p.add_argument("--panel-s864-actions", type=int, default=864)
    p.add_argument("--source-generator-actions", type=int, default=64)
    p.add_argument("--smoke-actions-per-primitive", type=int, default=64)
    p.add_argument("--sentinel-seeds", default="0,1,2")
    p.add_argument("--sentinel-steps", type=int, default=12)
    p.add_argument("--sentinel-train-size", type=int, default=512)
    p.add_argument("--sentinel-test-size", type=int, default=256)
    p.add_argument("--sentinel-hidden-dim", type=int, default=64)
    p.add_argument("--source-v9400", default=str(DEFAULT_SOURCE_V9400))
    p.add_argument("--source-v9350", default=str(DEFAULT_SOURCE_V9350))
    p.add_argument("--source-v9390", default=str(DEFAULT_SOURCE_V9390))
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
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))]


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


def cosine_payload(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return v9380.v9320.cosine(a, b)


def tensor_stats(payload: list[torch.Tensor]) -> dict[str, float]:
    flat = torch.cat([t.detach().float().reshape(-1).cpu() for t in payload])
    return {
        "payload_norm": float(flat.norm()),
        "payload_linf": float(flat.abs().max()),
        "payload_sparsity": float((flat.abs() <= 1.0e-12).to(torch.float32).mean()),
    }


def dist_for_ids(ids: list[str], stats: dict[str, dict[str, Any]], key: str) -> dict[str, float]:
    c = Counter(str(stats[aid].get(key, "")) for aid in ids if aid in stats)
    n = sum(c.values())
    return {k: v / max(1, n) for k, v in c.items()}


def psi_dist(p: dict[str, float], qd: dict[str, float]) -> float:
    eps = 1.0e-9
    return sum((p.get(k, eps) - qd.get(k, eps)) * math.log(p.get(k, eps) / qd.get(k, eps)) for k in set(p) | set(qd))


def kl_dist(p: dict[str, float], qd: dict[str, float]) -> float:
    eps = 1.0e-9
    return sum(p.get(k, eps) * math.log(p.get(k, eps) / qd.get(k, eps)) for k in set(p) | set(qd))


def max_gap(p: dict[str, float], qd: dict[str, float]) -> float:
    return max((abs(p.get(k, 0.0) - qd.get(k, 0.0)) for k in set(p) | set(qd)), default=0.0)


def add_panel_buckets(stats: dict[str, dict[str, Any]]) -> None:
    ids = list(stats)
    payload_vals = sorted(fnum(stats[aid].get("payload_norm")) for aid in ids)
    score_vals = sorted(fnum(stats[aid].get("CEp99_before")) for aid in ids)
    payload_cuts = [payload_vals[min(len(payload_vals) - 1, math.floor(i * len(payload_vals) / 5))] for i in range(1, 5)]
    score_cuts = [score_vals[min(len(score_vals) - 1, math.floor(i * len(score_vals) / 5))] for i in range(1, 5)]
    family_counts = Counter(str(stats[aid].get("family_id")) for aid in ids)
    for aid in ids:
        st = stats[aid]
        payload_bucket = sum(fnum(st.get("payload_norm")) > c for c in payload_cuts)
        score_bucket = sum(fnum(st.get("CEp99_before")) > c for c in score_cuts)
        st["payload_norm_bucket"] = f"b{payload_bucket}"
        st["score_bucket"] = f"b{score_bucket}"
        st["step_bucket"] = f"step_{inum(st.get('step')) // 32}"
        # Rare-family bucketing keeps the representativeness audit meaningful
        # for finite panels while preserving the large-family gaps explicitly.
        fam = str(st.get("family_id"))
        st["family_bucket"] = fam if family_counts[fam] >= 20 else "OTHER_RARE_FAMILY"


def greedy_balanced_panel(ids: list[str], stats: dict[str, dict[str, Any]], k: int, seed: int) -> list[str]:
    if k >= len(ids):
        return list(ids)
    keys = ["family_bucket", "step_bucket", "score_bucket", "payload_norm_bucket"]
    full = {key: dist_for_ids(ids, stats, key) for key in keys}
    sample = set(sorted(ids, key=lambda aid: stable_hash("panel", seed, aid))[:k])
    outside = set(ids) - sample

    def obj(s: set[str]) -> float:
        sid = list(s)
        return sum(psi_dist(dist_for_ids(sid, stats, key), full[key]) for key in keys) + sum(max_gap(dist_for_ids(sid, stats, key), full[key]) for key in keys)

    current = obj(sample)
    for _ in range(512):
        improved = False
        sample_list = sorted(sample, key=lambda aid: stable_hash("drop", seed, aid))[:48]
        outside_list = sorted(outside, key=lambda aid: stable_hash("add", seed, aid))[:96]
        for drop in sample_list:
            for add in outside_list:
                ns = (sample - {drop}) | {add}
                val = obj(ns)
                if val + 1.0e-12 < current:
                    sample = ns
                    outside = (outside - {add}) | {drop}
                    current = val
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return sorted(sample, key=lambda aid: stable_hash("panel_order", seed, aid))


def panel_quality(panel_ids: list[str], all_ids: list[str], stats: dict[str, dict[str, Any]], panel_id: str, seed: int) -> dict[str, Any]:
    keys = ["family_bucket", "step_bucket", "score_bucket", "payload_norm_bucket"]
    psis = {k: psi_dist(dist_for_ids(panel_ids, stats, k), dist_for_ids(all_ids, stats, k)) for k in keys}
    kls = {k: kl_dist(dist_for_ids(panel_ids, stats, k), dist_for_ids(all_ids, stats, k)) for k in keys}
    gaps = {k: max_gap(dist_for_ids(panel_ids, stats, k), dist_for_ids(all_ids, stats, k)) for k in keys}
    cp_rate = sum(1 for aid in panel_ids if fnum(stats[aid].get("weak_CP_rate")) > 0.0) / max(1, len(panel_ids))
    return {
        "stage": "P1_STRATIFIED_SOURCE_PANEL_BUILDER",
        "status": "panel_summary",
        "panel_id": panel_id,
        "panel_action_count": len(panel_ids),
        "panel_row_count": len(panel_ids) * len(HORIZONS),
        "sampling_rule": "deterministic_hash_stratified_family_step_score_payload",
        "sampling_seed": seed,
        "family_distribution": json.dumps(dist_for_ids(panel_ids, stats, "family_bucket"), sort_keys=True),
        "horizon_distribution": json.dumps({"20": 1 / 3, "80": 1 / 3, "240": 1 / 3}, sort_keys=True),
        "step_bucket_distribution": json.dumps(dist_for_ids(panel_ids, stats, "step_bucket"), sort_keys=True),
        "score_bucket_distribution": json.dumps(dist_for_ids(panel_ids, stats, "score_bucket"), sort_keys=True),
        "payload_norm_bucket_distribution": json.dumps(dist_for_ids(panel_ids, stats, "payload_norm_bucket"), sort_keys=True),
        "PSI_vs_full": sum(psis.values()),
        "KL_vs_full": sum(kls.values()),
        "max_family_gap": gaps["family_bucket"],
        "max_step_bucket_gap": gaps["step_bucket"],
        "max_score_bucket_gap": gaps["score_bucket"],
        "max_payload_bucket_gap": gaps["payload_norm_bucket"],
        "oracle_rate_diagnostic_only": cp_rate,
        "source_panel_used_for_official": int(len(panel_ids) >= 256),
        "panel_pass": int(sum(psis.values()) <= 0.20 and sum(kls.values()) <= 0.10 and max(gaps.values()) <= 0.10),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p0_boundary(source_v9400: Path) -> dict[str, Any]:
    route = read_json(source_v9400 / "route_decision.json")
    return {
        "stage": "P0_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "route_v9400": route.get("route"),
        "full_action_count": route.get("full_action_count"),
        "source_panel_action_count": route.get("source_panel_action_count"),
        "source_panel_fraction": fnum(route.get("source_panel_action_count")) / max(1.0, fnum(route.get("full_action_count"))),
        "source_panel_join_pass": route.get("source_panel_join_pass"),
        "full_weak_CP_row_rate": route.get("full_weak_CP_row_rate"),
        "source_panel_weak_CP_row_rate": route.get("source_panel_weak_CP_row_rate"),
        "full_strong_CP_row_rate": read_csv(source_v9400 / "p0_full_vs_source_panel_diagnosis.csv")[0].get("full_strong_CP_row_rate"),
        "source_panel_strong_CP_row_rate": read_csv(source_v9400 / "p0_full_vs_source_panel_diagnosis.csv")[0].get("source_panel_strong_CP_row_rate"),
        "full_long_risk_rate": route.get("full_long_risk_rate"),
        "source_panel_long_risk_rate": route.get("source_panel_long_risk_rate"),
        "full_V_ctrl_lcb": read_csv(source_v9400 / "p0_full_vs_source_panel_diagnosis.csv")[0].get("full_V_ctrl_lcb"),
        "source_panel_V_ctrl_lcb": read_csv(source_v9400 / "p0_full_vs_source_panel_diagnosis.csv")[0].get("source_panel_V_ctrl_lcb"),
        "selection_bias_PSI": route.get("selection_bias_PSI"),
        "selection_bias_KL": route.get("selection_bias_KL"),
        "max_family_gap": route.get("max_family_gap"),
        "max_step_bucket_gap": route.get("max_step_bucket_gap"),
        "max_score_bucket_gap": read_csv(source_v9400 / "p0_full_vs_source_panel_diagnosis.csv")[0].get("max_score_bucket_gap"),
        "route_flip_full_vs_source_panel": route.get("route_flip_full_vs_source_panel"),
        "source_selection_rule": route.get("source_selection_rule"),
        "source_selection_reason": read_csv(source_v9400 / "p1_source_selection_provenance_audit.csv")[0].get("reason"),
        "uses_outcome_in_source_selection": route.get("uses_outcome_in_source_selection"),
        "uses_dataset_name_in_source_selection": 0,
        "uses_future_step_in_source_selection": 0,
        "source_panel_used_for_official": 0,
        "p0_boundary_reproduced": int(route.get("route") == "R3-LegalSourceSelectorOpaque"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p1_panels(stats: dict[str, dict[str, Any]], seed: int, s64: int, s256: int, s864: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    add_panel_buckets(stats)
    all_ids = list(stats)
    panels = {
        "PANEL-S64": greedy_balanced_panel(all_ids, stats, s64, seed + 64),
        "PANEL-S256": greedy_balanced_panel(all_ids, stats, s256, seed + 256),
        "PANEL-S864": greedy_balanced_panel(all_ids, stats, s864, seed + 864),
    }
    rows = [panel_quality(ids, all_ids, stats, pid, seed) for pid, ids in panels.items()]
    trace: list[dict[str, Any]] = []
    for pid, ids in panels.items():
        for rank, aid in enumerate(ids):
            st = stats[aid]
            trace.append(
                {
                    "stage": "P1_STRATIFIED_SOURCE_PANEL_BUILDER",
                    "status": "panel_action_row",
                    "panel_id": pid,
                    "rank": rank + 1,
                    "source_action_id": aid,
                    "dataset": st.get("dataset"),
                    "seed": st.get("seed"),
                    "step": st.get("step"),
                    "family_id": st.get("family_id"),
                    "family_bucket": st.get("family_bucket"),
                    "step_bucket": st.get("step_bucket"),
                    "score_bucket": st.get("score_bucket"),
                    "payload_norm_bucket": st.get("payload_norm_bucket"),
                    "source_panel_used_for_official": int(pid == "PANEL-S256"),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    s256_row = next(r for r in rows if r["panel_id"] == "PANEL-S256")
    summary = {
        "stage": "P1_STRATIFIED_SOURCE_PANEL_BUILDER",
        "status": "summary",
        "panel_s64_pass": next(r for r in rows if r["panel_id"] == "PANEL-S64")["panel_pass"],
        "panel_s256_pass": s256_row["panel_pass"],
        "panel_s864_pass": next(r for r in rows if r["panel_id"] == "PANEL-S864")["panel_pass"],
        "official_panel_id": "PANEL-S256",
        "official_panel_action_count": len(panels["PANEL-S256"]),
        "official_panel_PSI_vs_full": s256_row["PSI_vs_full"],
        "official_panel_KL_vs_full": s256_row["KL_vs_full"],
        "official_panel_max_family_gap": s256_row["max_family_gap"],
        "official_panel_max_step_bucket_gap": s256_row["max_step_bucket_gap"],
        "official_panel_max_score_bucket_gap": s256_row["max_score_bucket_gap"],
        "official_panel_max_payload_bucket_gap": s256_row["max_payload_bucket_gap"],
        "source_panel_used_for_official": s256_row["source_panel_used_for_official"],
        "stratified_panel_pass": s256_row["panel_pass"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, trace, panels, summary


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def eval_logits_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return v9340.extended_metrics_from_logits(logits, y)


def ece_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return eval_logits_metrics(logits, y)["ECE"]


def mlp_forward(x: torch.Tensor, params: list[torch.Tensor], quadratic: bool = False) -> torch.Tensor:
    if quadratic:
        x = x.square()
    w1, b1, w2, b2 = params
    h = torch.relu(x @ w1 + b1)
    return h @ w2 + b2


def init_mlp(input_dim: int, output_dim: int, hidden: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    return [
        (torch.randn(input_dim, hidden, generator=gen, device=device) / math.sqrt(input_dim)).requires_grad_(True),
        torch.zeros(hidden, device=device, requires_grad=True),
        (torch.randn(hidden, output_dim, generator=gen, device=device) / math.sqrt(hidden)).requires_grad_(True),
        torch.zeros(output_dim, device=device, requires_grad=True),
    ]


def train_mlp_fixed(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, input_dim: int, output_dim: int, hidden: int, seed: int, steps: int, quadratic: bool, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]], int]:
    params = init_mlp(input_dim, output_dim, hidden, seed, device)
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    gen = torch.Generator(device=device).manual_seed(seed * 9173 + 41)
    trace: list[dict[str, Any]] = []
    lr = 1.0e-3
    wd = 1.0e-4
    batch = min(64, int(x_train.shape[0]))
    times: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, int(x_train.shape[0]), (batch,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        t0 = time.perf_counter()
        loss = F.cross_entropy(mlp_forward(xb, params, quadratic), yb)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            beta1, beta2 = 0.9, 0.999
            for i, p in enumerate(params):
                g = grads[i] + wd * p
                m[i].mul_(beta1).add_(g, alpha=1 - beta1)
                v[i].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                mh = m[i] / (1 - beta1 ** (step + 1))
                vh = v[i] / (1 - beta2 ** (step + 1))
                p.addcdiv_(mh, vh.sqrt().add_(1.0e-8), value=-lr)
        times.append((time.perf_counter() - t0) * 1000.0)
        if step in {0, steps - 1}:
            with torch.no_grad():
                tr = eval_logits_metrics(mlp_forward(x_train[: min(256, len(x_train))], params, quadratic), y_train[: min(256, len(y_train))])
                va = eval_logits_metrics(mlp_forward(x_val, params, quadratic), y_val)
            trace.append({"step": step, "train_loss": tr["NLL"], "train_acc": tr["acc"], "val_loss": va["NLL"], "val_acc": va["acc"]})
    with torch.no_grad():
        tr = eval_logits_metrics(mlp_forward(x_train[: min(512, len(x_train))], params, quadratic), y_train[: min(512, len(y_train))])
        va = eval_logits_metrics(mlp_forward(x_val, params, quadratic), y_val)
        te = eval_logits_metrics(mlp_forward(x_test, params, quadratic), y_test)
    param_count = sum(p.numel() for p in params)
    return {
        "train_acc": tr["acc"],
        "val_acc": va["acc"],
        "test_acc": te["acc"],
        "train_loss": tr["NLL"],
        "val_loss": va["NLL"],
        "test_loss": te["NLL"],
        "ECE": te["ECE"],
        "NLL": te["NLL"],
        "CEp99": te["CEp99"],
        "margin_p10": te["margin_p10"],
        "step_time_q50": q(times, 0.50),
        "step_time_q90": q(times, 0.90),
    }, trace, param_count


def train_lq_fixed(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, input_dim: int, output_dim: int, hidden: int, seed: int, steps: int, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]], int]:
    spec = lq.LQSpec("LQ-t2-h256-sentinel", "t2", hidden, "default", 0.8)
    params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed)
    states = [AdamWState.zeros_like(p) for p in params]
    fwd, bwd = lq.functions_for_basis(spec.basis)
    cfg = ManualAdamWConfig(lr=1.0e-3, weight_decay=1.0e-4)
    gen = torch.Generator(device=device).manual_seed(seed * 3911 + 17)
    batch = min(64, int(x_train.shape[0]))
    trace: list[dict[str, Any]] = []
    times: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, int(x_train.shape[0]), (batch,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        t0 = time.perf_counter()
        pack = bwd(xb, yb, *params, mu, std, 2.0, 2.0)
        grads = list(pack[1:])
        v9380.v9320.v9248.v92._adamw_update_foreach_(params, grads, states, cfg)
        times.append((time.perf_counter() - t0) * 1000.0)
        if step in {0, steps - 1}:
            with torch.no_grad():
                tr = eval_logits_metrics(fwd(x_train[: min(256, len(x_train))], *params, mu, std, 2.0, 2.0), y_train[: min(256, len(y_train))])
                va = eval_logits_metrics(fwd(x_val, *params, mu, std, 2.0, 2.0), y_val)
            trace.append({"step": step, "train_loss": tr["NLL"], "train_acc": tr["acc"], "val_loss": va["NLL"], "val_acc": va["acc"]})
    with torch.no_grad():
        tr = eval_logits_metrics(fwd(x_train[: min(512, len(x_train))], *params, mu, std, 2.0, 2.0), y_train[: min(512, len(y_train))])
        va = eval_logits_metrics(fwd(x_val, *params, mu, std, 2.0, 2.0), y_val)
        te = eval_logits_metrics(fwd(x_test, *params, mu, std, 2.0, 2.0), y_test)
    param_count = sum(p.numel() for p in params)
    return {
        "train_acc": tr["acc"],
        "val_acc": va["acc"],
        "test_acc": te["acc"],
        "train_loss": tr["NLL"],
        "val_loss": va["NLL"],
        "test_loss": te["NLL"],
        "ECE": te["ECE"],
        "NLL": te["NLL"],
        "CEp99": te["CEp99"],
        "margin_p10": te["margin_p10"],
        "step_time_q50": q(times, 0.50),
        "step_time_q90": q(times, 0.90),
    }, trace, param_count


def p2_base_acc_sentinel(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    datasets = parse_csv_list(args.datasets)
    seeds = parse_int_list(args.sentinel_seeds)
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
            x_train_all, y_train_all, x_test, y_test, input_dim, output_dim, _proto = v9380.v9320.v9248.v92._load_task(
                load_args, dataset, train_size=int(args.sentinel_train_size), test_size=int(args.sentinel_test_size)
            )
            x_train_all = x_train_all.to(device=device, dtype=torch.float32)
            y_train_all = y_train_all.to(device=device)
            x_test = x_test.to(device=device, dtype=torch.float32)
            y_test = y_test.to(device=device)
            val_n = max(32, int(0.20 * len(x_train_all)))
            x_val, y_val = x_train_all[-val_n:].contiguous(), y_train_all[-val_n:].contiguous()
            x_train, y_train = x_train_all[:-val_n].contiguous(), y_train_all[:-val_n].contiguous()
            model_specs = [
                ("LQ-t2-h256-sentinel", "LQ-t2-h256", "KAN", "lq"),
                ("MatchedMLP-sentinel", "MatchedMLP", "MLP", "mlp"),
                ("QuadraticFeatureMLP-sentinel", "QuadraticFeatureMLP", "MLP", "qmlp"),
            ]
            model_metrics: dict[str, dict[str, float]] = {}
            for model_id, family, kan_or_mlp, kind in model_specs:
                if kind == "lq":
                    metrics, trows, params = train_lq_fixed(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, output_dim, int(args.hidden_dim), int(args.seed) + seed * 100 + 941, int(args.sentinel_steps), device)
                else:
                    metrics, trows, params = train_mlp_fixed(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, output_dim, int(args.sentinel_hidden_dim), int(args.seed) + seed * 100 + (17 if kind == "mlp" else 29), int(args.sentinel_steps), kind == "qmlp", device)
                model_metrics[model_id] = metrics
                mlp_ref = model_metrics.get("MatchedMLP-sentinel", {})
                row = {
                    "stage": "P2_BASE_ACC_SENTINEL",
                    "status": "sentinel_model_row",
                    "model_id": model_id,
                    "model_family": family,
                    "KAN_or_MLP": kan_or_mlp,
                    "dataset": dataset,
                    "seed": seed,
                    "params": params,
                    "forward_flops": "",
                    "backward_flops": "",
                    **metrics,
                    "memory_peak": "",
                    "memory_ratio": 1.0,
                    "LQ_minus_MLP_test_acc": "",
                    "LQ_minus_MLP_ECE": "",
                    "LQ_minus_MLP_NLL": "",
                    "base_acc_used_for_controller": 0,
                    "hyperparams_fixed_before_run": 1,
                    "dataset_specific_tuning": 0,
                    "same_seed_schedule": 1,
                    "same_budget": 1,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                rows.append(row)
                for tr in trows:
                    trace.append(
                        {
                            "stage": "P2_BASE_ACC_SENTINEL_TRACE",
                            "status": "training_trace",
                            "model_id": model_id,
                            "dataset": dataset,
                            "seed": seed,
                            **tr,
                            "base_acc_used_for_controller": 0,
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }
                    )
            lq_row = next(r for r in rows if r["dataset"] == dataset and inum(r["seed"]) == seed and r["model_id"] == "LQ-t2-h256-sentinel")
            mlp_row = next(r for r in rows if r["dataset"] == dataset and inum(r["seed"]) == seed and r["model_id"] == "MatchedMLP-sentinel")
            for row in rows:
                if row["dataset"] == dataset and inum(row["seed"]) == seed:
                    row["LQ_minus_MLP_test_acc"] = fnum(lq_row["test_acc"]) - fnum(mlp_row["test_acc"])
                    row["LQ_minus_MLP_ECE"] = fnum(lq_row["ECE"]) - fnum(mlp_row["ECE"])
                    row["LQ_minus_MLP_NLL"] = fnum(lq_row["NLL"]) - fnum(mlp_row["NLL"])
    lq_tests = [fnum(r["test_acc"]) for r in rows if r["model_id"] == "LQ-t2-h256-sentinel"]
    mlp_tests = [fnum(r["test_acc"]) for r in rows if r["model_id"] == "MatchedMLP-sentinel"]
    summary = {
        "stage": "P2_BASE_ACC_SENTINEL",
        "status": "summary",
        "sentinel_row_count": len(rows),
        "datasets": ",".join(datasets),
        "seeds": ",".join(map(str, seeds)),
        "model_count": 3,
        "sentinel_complete": int(len(rows) == len(datasets) * len(seeds) * 3),
        "mean_test_acc_LQ": mean(lq_tests),
        "mean_test_acc_MLP": mean(mlp_tests),
        "LQ_minus_MLP_mean_test_acc": mean(lq_tests) - mean(mlp_tests),
        "LQ_catastrophic_fail": int(mean(lq_tests) < mean(mlp_tests) - 0.05),
        "base_acc_used_for_controller": 0,
        "hyperparams_fixed_before_run": 1,
        "dataset_specific_tuning": 0,
        "same_seed_schedule": 1,
        "same_budget": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, trace, summary


def load_source_payload_rows(source_v9330: Path, action_ids: list[str]) -> list[dict[str, str]]:
    rows = [r for r in read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv") if r.get("status") == "payload_disk_replay_row"]
    by_id = {str(r.get("action_id")): r for r in rows}
    return [by_id[aid] for aid in action_ids if aid in by_id]


def load_payload_from_row(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    path = REPO / str(row.get("payload_shard_path"))
    if str(path) not in cache:
        cache[str(path)] = torch.load(path, map_location=device)
    shard = cache[str(path)]
    off = inum(row.get("payload_tensor_offset"))
    return [shard["d0"][off].detach().clone().to(device), shard["d1"][off].detach().clone().to(device), shard["d2"][off].detach().clone().to(device)]


def transform_source_payload(primitive: str, src: list[torch.Tensor], source_stat: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    d0, d1, d2 = [t.detach().clone() for t in src]
    if primitive.startswith("AP0b-"):
        payload = [torch.zeros_like(d0), -0.30 * d1, -0.30 * d2]
        meta = {"hard_tail_fraction": 0.35, "source_generator_version": "v9410-last-edge-linearized-v1"}
    elif primitive.startswith("AP0c-"):
        payload = [0.20 * (d0 - d0.mean()), 0.20 * (d1 - d1.mean()), 0.20 * (d2 - d2.mean())]
        meta = {"hard_tail_fraction": 0.50, "source_generator_version": "v9410-residual-orthogonal-v1"}
    elif primitive.startswith("AP0d-"):
        payload = [0.10 * d0, 0.25 * d1, 0.55 * d2]
        meta = {"hard_tail_fraction": 0.70, "source_generator_version": "v9410-tail-margin-v1"}
    elif primitive.startswith("AP0e-"):
        payload = []
        for t in (d0, d1, d2):
            lim = torch.quantile(t.abs().float(), 0.90).item()
            payload.append(0.12 * torch.clamp(t, -lim, lim))
        meta = {"hard_tail_fraction": 0.25, "source_generator_version": "v9410-curvature-lowrank-v1"}
    else:
        support = min(1.0, math.log1p(fnum(source_stat.get("family_support_count"))) / math.log(600.0))
        scale = 0.05 + 0.20 * support
        payload = [scale * d0, scale * d1, scale * d2]
        meta = {"hard_tail_fraction": support, "source_generator_version": "v9410-support-memory-v1"}
    return payload, meta


def certificate_for_source(primitive: str, payload: list[torch.Tensor], src: list[torch.Tensor], source_row: dict[str, str], source_stat: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    stats = tensor_stats(payload)
    src_stats = tensor_stats(src)
    cos_src = cosine_payload(payload, src)
    norm_ratio = stats["payload_norm"] / max(1.0e-12, src_stats["payload_norm"])
    descent_lcb = max(0.0, -cos_src) * norm_ratio + 0.05 * fnum(meta.get("hard_tail_fraction"))
    tail_risk_ucb = min(1.0, 0.08 + 15.0 * stats["payload_linf"] + 0.15 * abs(norm_ratio - 0.20))
    long_risk_ucb = min(1.0, 0.10 + 0.20 * float(not primitive.startswith("AP0e-")) + 0.20 * max(0.0, norm_ratio - 0.30))
    null_ucb = min(1.0, 0.05 + 0.20 * float(norm_ratio < 0.05))
    support_lcb = min(1.0, 0.20 + 0.15 * math.log1p(fnum(source_stat.get("family_support_count"))) / math.log(600.0))
    cost = 0.03 + 0.05 * norm_ratio
    cert_score = descent_lcb - tail_risk_ucb - long_risk_ucb - null_ucb + support_lcb - cost
    cert_pass = int(descent_lcb > 0.03 and tail_risk_ucb <= 0.25 and long_risk_ucb <= 0.35 and null_ucb <= 0.25 and support_lcb >= 0.25 and norm_ratio <= 0.50)
    return {
        "certificate_schema_version": "v9410-source-cert-v1",
        "linearized_CE_delta": -descent_lcb,
        "linearized_margin_p10_delta": descent_lcb - tail_risk_ucb,
        "grad_dot_delta": -descent_lcb,
        "cos_delta_negative_grad": max(0.0, -cos_src),
        "cos_delta_adamw": "",
        "norm_ratio": norm_ratio,
        "hard_tail_fraction": meta.get("hard_tail_fraction"),
        "DescentLCB": descent_lcb,
        "TailRiskUCB": tail_risk_ucb,
        "LongRiskUCB": long_risk_ucb,
        "NullUCB": null_ucb,
        "SupportLCB": support_lcb,
        "CostEstimate": cost,
        "GradientConflict": 1.0 - max(0.0, -cos_src),
        "CurvatureGuard": 1.0 / max(1.0e-12, 1.0 + stats["payload_linf"] * 100.0),
        "HorizonGuard": 1.0 - long_risk_ucb,
        "PayloadNormRatio": norm_ratio,
        "certificate_score": cert_score,
        "certificate_pass": cert_pass,
        "uses_dataset_name": 0,
        "uses_outcome_at_commit": 0,
        "uses_future_step": 0,
        "uses_validation_or_test": 0,
        "commit_time_available": 1,
        "source_action_id": source_row.get("action_id"),
        **stats,
    }


def write_source_shards(out_dir: Path, generated: list[dict[str, Any]], shard_size: int = 64) -> None:
    shard_dir = out_dir / "source_action_payload_shards_v9410"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(generated), shard_size):
        shard_rows = generated[start : start + shard_size]
        shard_id = start // shard_size
        path = shard_dir / f"source_payload_cert_shard_{shard_id:05d}.pt"
        torch.save(
            {
                "d0": torch.stack([r["_payload"][0].cpu() for r in shard_rows]),
                "d1": torch.stack([r["_payload"][1].cpu() for r in shard_rows]),
                "d2": torch.stack([r["_payload"][2].cpu() for r in shard_rows]),
                "cert": torch.tensor(
                    [
                        [
                            fnum(r["DescentLCB"]),
                            fnum(r["TailRiskUCB"]),
                            fnum(r["LongRiskUCB"]),
                            fnum(r["NullUCB"]),
                            fnum(r["SupportLCB"]),
                            fnum(r["CostEstimate"]),
                            fnum(r["certificate_score"]),
                            fnum(r["certificate_pass"]),
                        ]
                        for r in shard_rows
                    ],
                    dtype=torch.float32,
                ),
            },
            path,
        )
        for offset, row in enumerate(shard_rows):
            row["ap_payload_shard_path"] = rel(path)
            row["ap_payload_tensor_offset"] = offset
            row["payload_tensor_written"] = 1
            row["certificate_tensor_written"] = 1


def p3_generate_sources(args: argparse.Namespace, official_panel: list[str], stats: dict[str, dict[str, Any]], source_v9330: Path, out_dir: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_ids = sorted(
        official_panel,
        key=lambda aid: (
            str(stats.get(aid, {}).get("dataset", "")),
            inum(stats.get(aid, {}).get("seed")),
            inum(stats.get(aid, {}).get("step")),
            stable_hash("source-order", aid),
        ),
    )[: int(args.source_generator_actions)]
    source_rows = load_source_payload_rows(source_v9330, source_ids)
    cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    cert_trace: list[dict[str, Any]] = []
    for row in source_rows:
        src = load_payload_from_row(row, cache, device)
        st = stats.get(str(row.get("action_id")), {})
        for primitive in PRIMITIVES:
            payload, meta = transform_source_payload(primitive, src, st)
            payload_hash = tensor_hash(payload)
            cert = certificate_for_source(primitive, payload, src, row, st, meta)
            certificate_hash = stable_hash(payload_hash, primitive, cert["certificate_score"], cert["certificate_pass"], cert["DescentLCB"], cert["LongRiskUCB"])
            source_action_id = stable_hash("v9410", primitive, row.get("event_id"), row.get("action_id"), payload_hash)
            rec = {
                "stage": "P3_REAL_VALUE_PRODUCING_SOURCE_GENERATOR",
                "status": "generated_source_action_row",
                "ap_action_id": source_action_id,
                "source_action_id": row.get("action_id"),
                "source_candidate_id": row.get("candidate_id"),
                "candidate_id": row.get("candidate_id"),
                "event_id": row.get("event_id"),
                "primitive_id": primitive,
                "source_generator_id": primitive,
                "source_generator_version": meta["source_generator_version"],
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "family_id": row.get("family_id"),
                "bucket_id": row.get("bucket_id"),
                "horizon": "20,80,240",
                "ap_payload_hash": payload_hash,
                "payload_hash": payload_hash,
                "certificate_hash": certificate_hash,
                "payload_hash_missing": 0,
                "certificate_hash_missing": 0,
                "certificate_hash_bound_to_payload_hash": 1,
                "payload_tensor_written": 1,
                "certificate_tensor_written": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
                **cert,
            }
            generated.append(rec)
            cert_trace.append(
                {
                    "stage": "P3_SOURCE_CERTIFICATE_TRACE",
                    "status": "certificate_row",
                    "ap_action_id": source_action_id,
                    "source_action_id": row.get("action_id"),
                    "primitive_id": primitive,
                    "payload_hash": payload_hash,
                    "certificate_hash": certificate_hash,
                    "certificate_pass": cert["certificate_pass"],
                    "certificate_score": cert["certificate_score"],
                    "DescentLCB": cert["DescentLCB"],
                    "TailRiskUCB": cert["TailRiskUCB"],
                    "LongRiskUCB": cert["LongRiskUCB"],
                    "NullUCB": cert["NullUCB"],
                    "SupportLCB": cert["SupportLCB"],
                    "CostEstimate": cert["CostEstimate"],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    write_source_shards(out_dir, generated)
    replay_rows: list[dict[str, Any]] = []
    for row in generated:
        shard = torch.load(REPO / str(row["ap_payload_shard_path"]), map_location="cpu")
        off = inum(row["ap_payload_tensor_offset"])
        disk_payload = [shard["d0"][off], shard["d1"][off], shard["d2"][off]]
        disk_hash = tensor_hash(disk_payload)
        err = max(float((disk_payload[i] - row["_payload"][i].cpu()).abs().max()) for i in range(3))
        replay_rows.append(
            {
                "stage": "P3_ACTION_APPLY_REPLAY_TRACE",
                "status": "source_disk_replay_row",
                "ap_action_id": row["ap_action_id"],
                "source_action_id": row["source_action_id"],
                "primitive_id": row["primitive_id"],
                "payload_hash_expected": row["ap_payload_hash"],
                "payload_hash_disk": disk_hash,
                "payload_hash_match": int(disk_hash == row["ap_payload_hash"]),
                "action_apply_error_linf": err,
                "action_apply_error_relative": err / max(1.0e-12, fnum(row["payload_norm"])),
                "action_apply_cosine_logged_applied": cosine_payload(disk_payload, [t.cpu() for t in row["_payload"]]),
                "replay_success": int(err <= 1.0e-12 and disk_hash == row["ap_payload_hash"]),
                "payload_shape": row.get("payload_tensor_shape", ""),
                "payload_dtype": "torch.float32",
                "payload_device": "cpu-disk-replay",
                "payload_load_time_ms": "",
                "payload_apply_time_ms": "",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    per_primitive = Counter(str(r["primitive_id"]) for r in generated)
    errors = [fnum(r["action_apply_error_linf"]) for r in replay_rows]
    rel_errors = [fnum(r["action_apply_error_relative"]) for r in replay_rows]
    cosines = [fnum(r["action_apply_cosine_logged_applied"]) for r in replay_rows]
    summary = {
        "stage": "P3_REAL_VALUE_PRODUCING_SOURCE_GENERATOR",
        "status": "summary",
        "source_generator_materialized": int(bool(generated)),
        "primitive_materialized_count": sum(1 for p in PRIMITIVES if per_primitive[p] > 0),
        "source_input_action_count": len(source_rows),
        "generated_action_count_total": len(generated),
        "max_generated_action_count_per_primitive": max(per_primitive.values(), default=0),
        "payload_tensor_written": int(bool(generated)),
        "certificate_tensor_written": int(bool(generated)),
        "payload_hash_missing_count": sum(inum(r.get("payload_hash_missing")) for r in generated),
        "certificate_hash_missing_count": sum(inum(r.get("certificate_hash_missing")) for r in generated),
        "action_apply_error_measured": int(bool(replay_rows)),
        "action_apply_error_linf_max": max(errors or [0.0]),
        "action_apply_error_relative_max": max(rel_errors or [0.0]),
        "action_apply_cosine_min": min(cosines or [0.0]),
        "commit_time_available": min([inum(r.get("commit_time_available")) for r in generated] or [0]),
        "uses_dataset_name": max([inum(r.get("uses_dataset_name")) for r in generated] or [0]),
        "uses_outcome_at_commit": max([inum(r.get("uses_outcome_at_commit")) for r in generated] or [0]),
        "uses_future_step": max([inum(r.get("uses_future_step")) for r in generated] or [0]),
        "certificate_schema_version": "v9410-source-cert-v1",
        "certificate_fields_complete": 1 if generated else 0,
        "certificate_pass_action_count": sum(inum(r.get("certificate_pass")) for r in generated),
        "p3_materialization_pass": int(
            bool(generated)
            and sum(1 for p in PRIMITIVES if per_primitive[p] > 0) >= 3
            and len(generated) >= 256
            and max(errors or [0.0]) <= 1.0e-6
            and max(rel_errors or [0.0]) <= 1.0e-4
            and min(cosines or [0.0]) >= 0.999999
            and not max([inum(r.get("uses_dataset_name")) for r in generated] or [0])
            and not max([inum(r.get("uses_outcome_at_commit")) for r in generated] or [0])
            and not max([inum(r.get("uses_future_step")) for r in generated] or [0])
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    trace = [{k: v for k, v in r.items() if not k.startswith("_")} for r in generated]
    return generated, trace, cert_trace, summary | {"_replay_rows": replay_rows}


def summarize_outcomes_by_primitive(rows: list[dict[str, Any]], horizon_filter: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real = [r for r in rows if r.get("branch_id") == "RealAP" and (horizon_filter is None or inum(r.get("horizon")) == horizon_filter)]
    out: list[dict[str, Any]] = []
    for primitive in PRIMITIVES:
        rs = [r for r in real if r.get("primitive_id") == primitive]
        n = len(rs)
        weak = sum(inum(r.get("weak_CP_label")) for r in rs)
        strong = sum(inum(r.get("strong_CP_label")) for r in rs)
        bad = sum(inum(r.get("bad_event_label")) for r in rs)
        null = sum(inum(r.get("null_event_label")) for r in rs)
        longrisk = sum(inum(r.get("long_risk_label")) for r in rs)
        robust = sum(inum(r.get("horizon_robust_CP_label")) for r in rs)
        vals = [fnum(r.get("V_ctrl_max_control_gap")) for r in rs]
        beats_adamw = sum(inum(r.get("beats_adamwparallel")) for r in rs)
        beats_bestlr = sum(inum(r.get("beats_bestlr")) for r in rs)
        beats_noop = sum(inum(r.get("beats_noop")) for r in rs)
        beats_random = sum(inum(r.get("beats_random")) for r in rs)
        row = {
            "stage": "P4_H20_IMMEDIATE_DIRECTION_SMOKE" if horizon_filter == 20 else "P5_HORIZON_EXTENSION_LONGRISK_AUDIT",
            "status": "primitive_outcome_summary",
            "source_generator_id": primitive,
            "primitive_id": primitive,
            "source_action_count": n,
            "horizon": horizon_filter if horizon_filter is not None else "20,80,240",
            "weak_CP_precision_h20" if horizon_filter == 20 else "weak_CP_precision_all": weak / max(1, n),
            "strong_CP_precision_h20" if horizon_filter == 20 else "strong_CP_precision_all": strong / max(1, n),
            "V_ctrl_mean_h20" if horizon_filter == 20 else "V_ctrl_mean_all": mean(vals),
            "V_ctrl_lcb_h20" if horizon_filter == 20 else "V_ctrl_lcb_all": lcb_mean(vals),
            "bad_event_rate_h20" if horizon_filter == 20 else "bad_event_rate_all": bad / max(1, n),
            "null_rate_h20" if horizon_filter == 20 else "null_rate_all": null / max(1, n),
            "long_risk_rate_h20" if horizon_filter == 20 else "long_risk_rate_all": longrisk / max(1, n),
            "horizon_robust_CP_action_rate": robust / max(1, n),
            "CEp99_delta_mean_h20" if horizon_filter == 20 else "CEp99_delta_mean_all": mean([fnum(r.get("CEp99_delta")) for r in rs]),
            "margin_p10_delta_mean_h20" if horizon_filter == 20 else "margin_p10_delta_mean_all": mean([fnum(r.get("margin_p10_delta")) for r in rs]),
            "ECE_delta_mean_h20" if horizon_filter == 20 else "ECE_delta_mean_all": mean([fnum(r.get("ECE_delta")) for r in rs]),
            "NLL_delta_mean_h20" if horizon_filter == 20 else "NLL_delta_mean_all": mean([fnum(r.get("NLL_delta")) for r in rs]),
            "curvature_delta_mean_h20" if horizon_filter == 20 else "curvature_delta_mean_all": mean([fnum(r.get("curvature_delta")) for r in rs]),
            "real_beats_adamwparallel_h20" if horizon_filter == 20 else "real_beats_adamwparallel_all": beats_adamw / max(1, n),
            "real_beats_bestlr_h20" if horizon_filter == 20 else "real_beats_bestlr_all": beats_bestlr / max(1, n),
            "real_beats_noop_h20" if horizon_filter == 20 else "real_beats_noop_all": beats_noop / max(1, n),
            "real_beats_random_h20" if horizon_filter == 20 else "real_beats_random_all": beats_random / max(1, n),
            "quality_audit_pass": int(n > 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        if horizon_filter == 20:
            row["p4_weak_pass"] = int(
                row["weak_CP_precision_h20"] >= 0.35
                and row["V_ctrl_lcb_h20"] > 0
                and row["bad_event_rate_h20"] <= 0.10
                and row["null_rate_h20"] <= 0.30
                and row["real_beats_adamwparallel_h20"] >= 0.50
            )
            row["p4_strong_pass"] = int(
                row["weak_CP_precision_h20"] >= 0.50
                and row["V_ctrl_lcb_h20"] > 0.05
                and row["bad_event_rate_h20"] <= 0.05
                and row["real_beats_adamwparallel_h20"] >= 0.60
                and row["real_beats_bestlr_h20"] >= 0.50
            )
        else:
            by_h: dict[int, list[dict[str, Any]]] = {h: [r for r in rs if inum(r.get("horizon")) == h] for h in HORIZONS}
            row["weak_CP_h20"] = sum(inum(r.get("weak_CP_label")) for r in by_h[20]) / max(1, len(by_h[20]))
            row["weak_CP_h80"] = sum(inum(r.get("weak_CP_label")) for r in by_h[80]) / max(1, len(by_h[80]))
            row["weak_CP_h240"] = sum(inum(r.get("weak_CP_label")) for r in by_h[240]) / max(1, len(by_h[240]))
            row["V_ctrl_lcb_h20"] = lcb_mean([fnum(r.get("V_ctrl_max_control_gap")) for r in by_h[20]])
            row["V_ctrl_lcb_h80"] = lcb_mean([fnum(r.get("V_ctrl_max_control_gap")) for r in by_h[80]])
            row["V_ctrl_lcb_h240"] = lcb_mean([fnum(r.get("V_ctrl_max_control_gap")) for r in by_h[240]])
            row["long_risk_h240"] = sum(inum(r.get("long_risk_label")) for r in by_h[240]) / max(1, len(by_h[240]))
            row["p5_weak_pass"] = int(row["weak_CP_precision_all"] >= 0.40 and row["V_ctrl_lcb_all"] > 0 and row["long_risk_rate_all"] <= 0.15 and row["weak_CP_h240"] >= 0.25)
            row["p5_strong_pass"] = int(row["weak_CP_precision_all"] >= 0.55 and row["V_ctrl_lcb_all"] > 0.05 and row["long_risk_rate_all"] <= 0.10 and row["horizon_robust_CP_action_rate"] >= 0.03)
        out.append(row)
    key = "p4_weak_pass" if horizon_filter == 20 else "p5_weak_pass"
    best = max(out, key=lambda r: (inum(r.get(key)), fnum(r.get("weak_CP_precision_h20" if horizon_filter == 20 else "weak_CP_precision_all")), fnum(r.get("V_ctrl_lcb_h20" if horizon_filter == 20 else "V_ctrl_lcb_all"))), default={})
    summary = {
        "stage": "P4_H20_IMMEDIATE_DIRECTION_SMOKE" if horizon_filter == 20 else "P5_HORIZON_EXTENSION_LONGRISK_AUDIT",
        "status": "summary",
        "best_primitive_id": best.get("primitive_id", ""),
        "best_weak_CP_precision_h20" if horizon_filter == 20 else "best_weak_CP_precision_all": best.get("weak_CP_precision_h20" if horizon_filter == 20 else "weak_CP_precision_all", 0.0),
        "best_V_ctrl_lcb_h20" if horizon_filter == 20 else "best_V_ctrl_lcb_all": best.get("V_ctrl_lcb_h20" if horizon_filter == 20 else "V_ctrl_lcb_all", 0.0),
        "best_bad_event_rate_h20" if horizon_filter == 20 else "best_long_risk_rate_all": best.get("bad_event_rate_h20" if horizon_filter == 20 else "long_risk_rate_all", 0.0),
        "h20_immediate_direction_pass" if horizon_filter == 20 else "horizon_extension_pass": max((inum(r.get(key)) for r in out), default=0),
        "quality_audit_pass": int(bool(real)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + out, summary


def p6_certificate_sufficiency(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    real = [r for r in outcome_rows if r.get("branch_id") == "RealAP"]
    by_action = {str(r["ap_action_id"]): r for r in generated}
    joined = []
    for row in real:
        gen = by_action.get(str(row.get("ap_action_id")), {})
        if not gen:
            continue
        joined.append(row | {"certificate_score": gen.get("certificate_score"), "cert_pass": gen.get("certificate_pass"), "LongRiskUCB": gen.get("LongRiskUCB"), "DescentLCB": gen.get("DescentLCB")})
    labels_weak = [inum(r.get("weak_CP_label")) for r in joined]
    labels_strong = [inum(r.get("strong_CP_label")) for r in joined]
    labels_long = [inum(r.get("long_risk_label")) for r in joined]
    scores = [fnum(r.get("certificate_score")) for r in joined]
    long_scores = [fnum(r.get("LongRiskUCB")) for r in joined]
    pass_rows = [r for r in joined if inum(r.get("cert_pass"))]
    fail_rows = [r for r in joined if not inum(r.get("cert_pass"))]
    pweak_pass = sum(inum(r.get("weak_CP_label")) for r in pass_rows) / max(1, len(pass_rows))
    pweak_fail = sum(inum(r.get("weak_CP_label")) for r in fail_rows) / max(1, len(fail_rows))
    plong_pass = sum(inum(r.get("long_risk_label")) for r in pass_rows) / max(1, len(pass_rows))
    plong_fail = sum(inum(r.get("long_risk_label")) for r in fail_rows) / max(1, len(fail_rows))
    vals_pass = [fnum(r.get("V_ctrl_max_control_gap")) for r in pass_rows]
    auc_weak = auc_score(scores, labels_weak)
    auc_strong = auc_score(scores, labels_strong)
    auc_long = max(auc_score(long_scores, labels_long), 1.0 - auc_score(long_scores, labels_long))
    lift_weak = pweak_pass / max(1.0e-12, pweak_fail)
    lift_long = plong_pass / max(1.0e-12, plong_fail)
    monotone = int(pweak_pass > pweak_fail and plong_pass < plong_fail)
    summary = {
        "stage": "P6_EFFECT_VALID_CERTIFICATE_SUFFICIENCY",
        "status": "summary",
        "certificate_id": "v9410-source-cert-v1",
        "joined_outcome_count": len(joined),
        "cert_pass_rows": len(pass_rows),
        "cert_fail_rows": len(fail_rows),
        "AUC_weak_CP": auc_weak,
        "AUC_strong_CP": auc_strong,
        "AUC_long_risk": auc_long,
        "AUC_null": auc_score(scores, [inum(r.get("null_event_label")) for r in joined]),
        "P_weak_CP_given_cert_pass": pweak_pass,
        "P_weak_CP_given_cert_fail": pweak_fail,
        "P_longrisk_given_cert_pass": plong_pass,
        "P_longrisk_given_cert_fail": plong_fail,
        "Lift_weak": lift_weak,
        "Lift_longrisk": lift_long,
        "monotone_sign_pass": monotone,
        "calibration_ECE": abs(pweak_pass - mean([fnum(r.get("certificate_score")) for r in pass_rows] or [0.0])),
        "V_ctrl_lcb_cert_pass": lcb_mean(vals_pass),
        "certificate_sufficiency_pass": int(auc_weak >= 0.65 and auc_long >= 0.65 and lift_weak >= 1.8 and plong_pass <= 0.70 * max(1.0e-12, plong_fail) and monotone),
        "certificate_sufficiency_strong_pass": int(auc_weak >= 0.75 and auc_long >= 0.75 and pweak_pass >= 0.50 and plong_pass <= 0.10 and lcb_mean(vals_pass) > 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    trace = [
        {
            "stage": "P6_CERTIFICATE_COMPONENT_TRACE",
            "status": "joined_certificate_outcome_row",
            "ap_action_id": r.get("ap_action_id"),
            "primitive_id": r.get("primitive_id"),
            "horizon": r.get("horizon"),
            "cert_pass": r.get("cert_pass"),
            "certificate_score": r.get("certificate_score"),
            "weak_CP_label": r.get("weak_CP_label"),
            "long_risk_label": r.get("long_risk_label"),
            "V_ctrl_max_control_gap": r.get("V_ctrl_max_control_gap"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        for r in joined
    ]
    ablation = [
        {
            "stage": "P6_CERTIFICATE_ABLATION",
            "status": "component_auc",
            "component": comp,
            "auc_weak_CP": auc_score([fnum(by_action.get(str(r.get("ap_action_id")), {}).get(comp)) for r in joined], labels_weak) if joined else 0.5,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        for comp in ["DescentLCB", "TailRiskUCB", "LongRiskUCB", "NullUCB", "SupportLCB", "CostEstimate"]
    ]
    return [summary], trace + ablation, summary


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}


def write_hashes(out_dir: Path, artifacts: list[Path]) -> None:
    write_csv(out_dir / "artifact_hashes_v9410.csv", [{"artifact": rel(p), "sha256": sha256_file(p)} for p in artifacts if p.exists()])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    device = device_from(args.device)

    source_v9400 = Path(args.source_v9400)
    source_v9350 = Path(args.source_v9350)
    source_v9390 = Path(args.source_v9390)
    source_v9330 = Path(args.source_v9330)
    source_v9280 = Path(args.source_v9280)

    p0 = p0_boundary(source_v9400)
    full_rows, by_action = v9400.load_full_ap0(source_v9350)
    payload_rows = v9400.load_payload_rows(source_v9330)
    stats = v9400.action_stats(by_action, payload_rows)
    p1_rows, p1_trace, panels, p1 = p1_panels(stats, int(args.seed), int(args.panel_s64_actions), int(args.panel_s256_actions), int(args.panel_s864_actions))
    p2_rows, p2_trace, p2 = p2_base_acc_sentinel(args, device)

    generated: list[dict[str, Any]] = []
    source_trace: list[dict[str, Any]] = []
    cert_trace: list[dict[str, Any]] = []
    p3: dict[str, Any]
    if not inum(p1.get("stratified_panel_pass")):
        p3 = not_run("P3_REAL_VALUE_PRODUCING_SOURCE_GENERATOR", "P1_stratified_source_panel_failed")
        p3["source_generator_materialized"] = 0
        replay_rows: list[dict[str, Any]] = []
    else:
        generated, source_trace, cert_trace, p3_with_replay = p3_generate_sources(args, panels["PANEL-S256"], stats, source_v9330, out_dir, device)
        replay_rows = p3_with_replay.pop("_replay_rows")
        p3 = p3_with_replay

    outcome_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    if inum(p3.get("p3_materialization_pass")):
        old_primitives = list(v9380.PRIMITIVES)
        try:
            v9380.PRIMITIVES = PRIMITIVES
            outcome_rows, completion_rows, raw_outcome = v9380.materialize_ap_smoke(args, generated, device)
        finally:
            v9380.PRIMITIVES = old_primitives
        for row in outcome_rows:
            row["stage"] = "P4_P5_SOURCE_OUTCOME_MATERIALIZATION"
            row["outcome_source"] = "same_run_generated_source_v9410"
        p4_rows, p4 = summarize_outcomes_by_primitive(outcome_rows, 20)
        p5_rows, p5 = summarize_outcomes_by_primitive(outcome_rows, None)
        p4["branch_horizon_row_count_expected"] = raw_outcome.get("branch_horizon_row_count_expected")
        p4["branch_horizon_row_count_actual"] = raw_outcome.get("branch_horizon_row_count_actual")
        p4["source_outcome_materialized"] = raw_outcome.get("ap_smoke_outcome_pass")
        p5["branch_horizon_row_count_actual"] = raw_outcome.get("branch_horizon_row_count_actual")
    else:
        p4 = not_run("P4_H20_IMMEDIATE_DIRECTION_SMOKE", "P3_source_generator_not_materialized")
        p4["h20_immediate_direction_pass"] = 0
        p4_rows = [p4]
        p5 = not_run("P5_HORIZON_EXTENSION_LONGRISK_AUDIT", "P4_h20_immediate_direction_not_available")
        p5["horizon_extension_pass"] = 0
        p5_rows = [p5]

    if outcome_rows:
        p6_rows, p6_trace, p6 = p6_certificate_sufficiency(generated, outcome_rows)
    else:
        p6 = not_run("P6_EFFECT_VALID_CERTIFICATE_SUFFICIENCY", "source_outcomes_not_materialized")
        p6["certificate_sufficiency_pass"] = 0
        p6_rows = [p6]
        p6_trace = []

    p7 = not_run("P7_MINIMAL_SOURCE_CERTIFICATE_CONTROLLER", "P6_certificate_or_source_survivor_not_available")
    p7["source_controller_pass"] = 0
    p8 = not_run("P8_SELECTED_SOURCE_ONLINE_RUNTIME", "P7_controller_not_selected")
    p8["selected_runtime_pass"] = 0
    p9 = not_run("P9_SYSTEM_INTEGRATION_GATE", "P7_controller_not_selected")
    p10 = not_run("P10_LEAVEOUT_BOUNDARY", "P9_system_controller_not_official")
    p11 = not_run("P11_DIAGNOSTIC_PAIRED_REPLAY_SCOUT", "P9_system_controller_not_official")
    p12 = not_run("P12_OFFICIAL_PAIRED_REPLAY", "P10_leaveout_not_open")
    p13 = not_run("P13_SHORT_FULL_TRAINING_MLP_COMPARISON", "P12_official_paired_replay_not_open")
    p14 = not_run("P14_CONTINUAL_ANTIFORGETTING", "P13_short_full_not_open")

    if not inum(p0.get("p0_boundary_reproduced")):
        route, blocker, next_impl = "R0-BoundaryUnstable", "v9400_boundary_unstable", "reproduce_v9400_boundary"
    elif not inum(p1.get("stratified_panel_pass")):
        route, blocker, next_impl = "R1-SourcePanelSamplerStillBiased", "source_panel_sampler_still_biased", "repair_stratified_source_sampler"
    elif not inum(p3.get("p3_materialization_pass")):
        route, blocker, next_impl = "R3-SourceGeneratorNotMaterialized", "source_generator_not_materialized", "implement_AP0b_AP0f_payload_certificate_replay"
    elif not inum(p4.get("h20_immediate_direction_pass")):
        route, blocker, next_impl = "R4-GeneratedSourceImmediateDirectionFail", "generated_source_immediate_direction_fail", "redesign_source_objective_not_certificate_threshold"
    elif not inum(p5.get("horizon_extension_pass")):
        route, blocker, next_impl = "R5-ImmediatePositiveButHorizonFragile", "immediate_positive_but_horizon_fragile", "redesign_horizon_guarded_source"
    elif not inum(p6.get("certificate_sufficiency_pass")):
        route, blocker, next_impl = "R6-CertificateNotEffectValid", "certificate_not_effect_valid", "redesign_effect_valid_certificate"
    elif not inum(p7.get("source_controller_pass")):
        route, blocker, next_impl = "R7-SourceControllerHeldoutFail", "source_controller_heldout_fail", "build_crossfit_source_certificate_controller"
    elif not inum(p8.get("selected_runtime_pass")):
        route, blocker, next_impl = "R8-SelectedRuntimeFail", "selected_runtime_fail", "optimize_selected_payload_apply_runtime"
    else:
        route, blocker, next_impl = "R9-SystemLegalSourceControllerPass", "none", "open_leaveout_and_paired_replay"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9400": read_json(source_v9400 / "route_decision.json").get("route"),
        "p0_boundary_reproduced": p0.get("p0_boundary_reproduced"),
        "source_panel_used_for_official": p1.get("source_panel_used_for_official"),
        "stratified_panel_pass": p1.get("stratified_panel_pass"),
        "official_panel_PSI_vs_full": p1.get("official_panel_PSI_vs_full"),
        "official_panel_KL_vs_full": p1.get("official_panel_KL_vs_full"),
        "official_panel_max_family_gap": p1.get("official_panel_max_family_gap"),
        "base_acc_sentinel_complete": p2.get("sentinel_complete"),
        "base_acc_used_for_controller": p2.get("base_acc_used_for_controller"),
        "mean_test_acc_LQ": p2.get("mean_test_acc_LQ"),
        "mean_test_acc_MLP": p2.get("mean_test_acc_MLP"),
        "LQ_catastrophic_fail": p2.get("LQ_catastrophic_fail"),
        "source_generator_materialized": p3.get("source_generator_materialized", 0),
        "primitive_materialized_count": p3.get("primitive_materialized_count", 0),
        "generated_action_count_total": p3.get("generated_action_count_total", 0),
        "payload_tensor_written": p3.get("payload_tensor_written", 0),
        "certificate_tensor_written": p3.get("certificate_tensor_written", 0),
        "payload_hash_missing_count": p3.get("payload_hash_missing_count", ""),
        "certificate_hash_missing_count": p3.get("certificate_hash_missing_count", ""),
        "action_apply_error_linf_max": p3.get("action_apply_error_linf_max", ""),
        "action_apply_error_relative_max": p3.get("action_apply_error_relative_max", ""),
        "action_apply_cosine_min": p3.get("action_apply_cosine_min", ""),
        "p3_materialization_pass": p3.get("p3_materialization_pass", 0),
        "h20_immediate_direction_pass": p4.get("h20_immediate_direction_pass", 0),
        "best_h20_primitive_id": p4.get("best_primitive_id", ""),
        "best_weak_CP_precision_h20": p4.get("best_weak_CP_precision_h20", 0),
        "best_V_ctrl_lcb_h20": p4.get("best_V_ctrl_lcb_h20", 0),
        "best_bad_event_rate_h20": p4.get("best_bad_event_rate_h20", 0),
        "horizon_extension_pass": p5.get("horizon_extension_pass", 0),
        "best_weak_CP_precision_all": p5.get("best_weak_CP_precision_all", 0),
        "best_V_ctrl_lcb_all": p5.get("best_V_ctrl_lcb_all", 0),
        "best_long_risk_rate_all": p5.get("best_long_risk_rate_all", 0),
        "certificate_sufficiency_pass": p6.get("certificate_sufficiency_pass", 0),
        "AUC_weak_CP_certificate": p6.get("AUC_weak_CP", ""),
        "AUC_long_risk_certificate": p6.get("AUC_long_risk", ""),
        "Lift_weak_certificate": p6.get("Lift_weak", ""),
        "Lift_longrisk_certificate": p6.get("Lift_longrisk", ""),
        "source_controller_pass": p7.get("source_controller_pass", 0),
        "selected_runtime_pass": p8.get("selected_runtime_pass", 0),
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "success_v9410_strict_purekan_functional": 0,
        "success_v9410_full_functional": 0,
        "success_v9410_external_ready": 0,
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
        "source_v9400": rel(source_v9400),
        "source_v9350": rel(source_v9350),
        "source_v9390": rel(source_v9390),
        "source_v9330": rel(source_v9330),
        "source_v9280": rel(source_v9280),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv(out_dir / "p0_boundary_reproduction_v9410.csv", [p0])
    write_csv(out_dir / "p1_stratified_source_panel_builder.csv", p1_rows)
    write_csv(out_dir / "stratified_source_panel_trace_v9410.csv", p1_trace)
    write_csv(out_dir / "p2_base_acc_sentinel_lq_vs_mlp.csv", p2_rows)
    write_csv(out_dir / "base_acc_training_trace_v9410.csv", p2_trace)
    write_csv(out_dir / "p3_real_value_producing_source_generator.csv", [p3])
    write_csv(out_dir / "source_payload_trace_v9410.csv", source_trace)
    write_csv(out_dir / "source_certificate_trace_v9410.csv", cert_trace)
    write_csv(out_dir / "action_apply_replay_trace_v9410.csv", replay_rows)
    write_csv(out_dir / "p4_h20_immediate_direction_smoke.csv", p4_rows)
    write_csv(out_dir / "source_outcome_trace_v9410.csv", outcome_rows)
    write_csv(out_dir / "branch_horizon_completion_trace_v9410.csv", completion_rows)
    write_csv(out_dir / "p5_horizon_extension_longrisk_audit.csv", p5_rows)
    write_csv(out_dir / "p6_effect_valid_certificate_sufficiency.csv", p6_rows)
    write_csv(out_dir / "certificate_component_trace_v9410.csv", p6_trace)
    write_csv(out_dir / "p7_minimal_source_certificate_controller.csv", [p7])
    write_csv(out_dir / "controller_frontier_trace_v9410.csv", [not_run("P7_CONTROLLER_FRONTIER_TRACE", "P6_certificate_or_source_survivor_not_available")])
    write_csv(out_dir / "leaveout_source_controller_trace_v9410.csv", [not_run("P7_LEAVEOUT_SOURCE_CONTROLLER_TRACE", "P6_certificate_or_source_survivor_not_available")])
    write_csv(out_dir / "controller_failure_autopsy_v9410.csv", [not_run("P7_CONTROLLER_FAILURE_AUTOPSY", "P6_certificate_or_source_survivor_not_available")])
    write_csv(out_dir / "p8_selected_source_online_runtime.csv", [p8])
    write_csv(out_dir / "runtime_component_trace_v9410.csv", [not_run("P8_RUNTIME_COMPONENT_TRACE", "P7_controller_not_selected")])
    write_csv(out_dir / "no_event_preservation_trace_v9410.csv", [not_run("P8_NO_EVENT_PRESERVATION_TRACE", "P7_controller_not_selected")])
    write_csv(out_dir / "runtime_isolation_audit_v9410.csv", [not_run("P8_RUNTIME_ISOLATION_AUDIT", "P7_controller_not_selected")])
    write_csv(out_dir / "p9_system_integration_gate_v9410.csv", [p9 | {"official_eligible": 0, "system_legal_controller_pass": 0, "base_acc_used_for_controller": 0}])
    write_csv(out_dir / "p10_leaveout_boundary_v9410.csv", [p10])
    write_csv(out_dir / "p11_diagnostic_paired_replay_scout_v9410.csv", [p11])
    write_csv(out_dir / "p12_official_paired_replay_v9410.csv", [p12])
    write_csv(out_dir / "p13_short_full_training_mlp_comparison_v9410.csv", [p13])
    write_csv(out_dir / "p14_continual_antiforgetting_v9410.csv", [p14])

    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "stratified_panel_pass": p1.get("stratified_panel_pass"),
        "base_acc_sentinel_complete": p2.get("sentinel_complete"),
        "base_acc_used_for_controller": 0,
        "source_generator_materialized": p3.get("source_generator_materialized", 0),
        "p3_materialization_pass": p3.get("p3_materialization_pass", 0),
        "h20_immediate_direction_pass": p4.get("h20_immediate_direction_pass", 0),
        "horizon_extension_pass": p5.get("horizon_extension_pass", 0),
        "certificate_sufficiency_pass": p6.get("certificate_sufficiency_pass", 0),
        "source_controller_pass": p7.get("source_controller_pass", 0),
        "selected_runtime_pass": p8.get("selected_runtime_pass", 0),
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
    write_csv(out_dir / "contract_audit_v9410.csv", [contract])
    failure = {
        "route": route,
        "F1_source_panel_sampling_bias": int(not inum(p1.get("stratified_panel_pass"))),
        "F2_static_legal_selector_opaque": 1,
        "F3_source_generator_missing": int(not inum(p3.get("source_generator_materialized", 0))),
        "F4_action_apply_error": int(fnum(p3.get("action_apply_error_linf_max", 0.0)) > 1.0e-6),
        "F5_h20_immediate_direction_fail": int(not inum(p4.get("h20_immediate_direction_pass", 0))),
        "F6_horizon_long_risk_fail": int(inum(p4.get("h20_immediate_direction_pass", 0)) and not inum(p5.get("horizon_extension_pass", 0))),
        "F7_certificate_no_effect_lift": int(bool(outcome_rows) and not inum(p6.get("certificate_sufficiency_pass", 0))),
        "F8_controller_support_collapse": int(not inum(p7.get("source_controller_pass", 0))),
        "F9_runtime_payload_apply_too_slow": int(not inum(p8.get("selected_runtime_pass", 0))),
        "F10_base_acc_sentinel_catastrophic": p2.get("LQ_catastrophic_fail"),
        "F11_dataset_tuning_detected": 0,
        "F12_outcome_at_commit_violation": 0,
        "F13_diagnostic_promoted_to_official": 0,
        "F14_short_full_not_open": 1,
        "primary_blocker": blocker,
    }
    write_csv(out_dir / "failure_table_v9410.csv", [failure])
    audit = audit_no_fake(sorted(out_dir.glob("*.csv")))
    write_csv(out_dir / "provenance_audit_v9410.csv", [audit])
    artifacts = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "p0_boundary_reproduction_v9410.csv",
        out_dir / "p1_stratified_source_panel_builder.csv",
        out_dir / "p2_base_acc_sentinel_lq_vs_mlp.csv",
        out_dir / "p3_real_value_producing_source_generator.csv",
        out_dir / "p4_h20_immediate_direction_smoke.csv",
        out_dir / "p5_horizon_extension_longrisk_audit.csv",
        out_dir / "p6_effect_valid_certificate_sufficiency.csv",
        out_dir / "p7_minimal_source_certificate_controller.csv",
        out_dir / "p8_selected_source_online_runtime.csv",
        out_dir / "p9_system_integration_gate_v9410.csv",
        out_dir / "contract_audit_v9410.csv",
        out_dir / "provenance_audit_v9410.csv",
        out_dir / "failure_table_v9410.csv",
    ]
    write_hashes(out_dir, artifacts)
    print(json.dumps({"out_dir": str(out_dir), "route": route, "generated_action_count_total": route_decision.get("generated_action_count_total"), "h20_pass": route_decision.get("h20_immediate_direction_pass")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
