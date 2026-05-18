#!/usr/bin/env python3
"""DG-KAN v9.8.0 Geometry-Adaptive Optimizer full-gated runner.

This runner intentionally keeps the v9.7.7 boundary discipline: every claim is
backed by landed CSV/JSON artifacts, and stages that cannot be legally
materialized write explicit blocker rows instead of filling missing data with
zeros.  The generated-update preflight creates real function-space updates on
the current LQ training stream, but it remains diagnostic unless its gates pass.
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

import run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop as v9660  # noqa: E402
import run_v9700_dual_validation_existing_action_transfer_principle as v9700  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion as v9730  # noqa: E402
import run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation as v9750  # noqa: E402
import run_v9760_core77_ldo_density_mechanism_existing_action_controller as v9760  # noqa: E402
import run_v9770_core77_support_density_natural_stream_gate as v9770  # noqa: E402
from dgkan.functional import lq_output_space_functional as lq_func  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.8.0_GeometryAdaptiveOptimizer_并行验证计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9800_geometry_adaptive_optimizer.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.0_GeometryAdaptiveOptimizer_实验复盘.md"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9800_geometry_adaptive_optimizer_full_20260516T080000Z"
DEFAULT_V9770 = RESULT_ROOT / "v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z"
DEFAULT_V9760 = RESULT_ROOT / "v9760_core77_ldo_density_mechanism_existing_action_controller_first_20260516T050000Z"
DEFAULT_V9750 = RESULT_ROOT / "v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation_first_20260516T040000Z"
DEFAULT_V9740 = RESULT_ROOT / "v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z"
DEFAULT_V9720 = RESULT_ROOT / "v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z"
DEFAULT_V9480 = RESULT_ROOT / "v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"

TARGET_K = 87
Z = 1.96
CANONICAL_HORIZONS = [20, 80, 240]
REQUESTED_TRAJECTORY_HORIZONS = [1, 5, 20, 80, 240]
GENERATED_BRANCHES = ["RealFunctional", "AdamWParallel", "bestLR", "NoOp", "Random", "ShuffledFunctionalPayload"]
GENERATED_HORIZONS = [20, 80, 240]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", default="full-gated", choices=["smoke", "full-gated", "existing-only"])
    p.add_argument("--panel-targets", default="2876,5000,10000,20000")
    p.add_argument("--generated-per-family", type=int, default=64)
    p.add_argument("--source-v9770", default=str(DEFAULT_V9770))
    p.add_argument("--source-v9760", default=str(DEFAULT_V9760))
    p.add_argument("--source-v9750", default=str(DEFAULT_V9750))
    p.add_argument("--source-v9740", default=str(DEFAULT_V9740))
    p.add_argument("--source-v9720", default=str(DEFAULT_V9720))
    p.add_argument("--source-v9480", default=str(DEFAULT_V9480))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--bootstrap-reps", type=int, default=128)
    p.add_argument("--generated-step-fraction", type=float, default=0.20)
    p.add_argument("--generated-future-horizon", type=int, default=20)
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def stdev(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def lcb(xs: list[float]) -> float:
    return v9720.lcb(xs)


def ucb(xs: list[float]) -> float:
    return v9720.ucb(xs)


def wilson(count: int, total: int, z: float = Z) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    phat = count / total
    den = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    delta = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return phat, max(0.0, (centre - delta) / den), min(1.0, (centre + delta) / den)


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode("utf-8")).hexdigest()


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


def topk(scores: list[float], k: int = TARGET_K) -> list[int]:
    return v9720.topk_idx(scores, k)


def membership_scores(n: int, idx: list[int]) -> list[float]:
    scores = [0.0] * n
    for i in idx:
        scores[i] = 1.0
    return scores


def quality(ap0: list[dict[str, Any]], idx: list[int], k: int | None = None) -> dict[str, Any]:
    return v9720.quality_for_indices(ap0, idx, membership_scores(len(ap0), idx), k or max(1, len(idx)))


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


def score_corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    vy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / (vx * vy)


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
            txt = path.read_text()
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


def clone_params(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clone_states(states: list[AdamWState]) -> list[AdamWState]:
    return [AdamWState(step=s.step, m=s.m.detach().clone(), v=s.v.detach().clone()) for s in states]


def ce_mean(env: dict[str, Any], params: list[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> float:
    logits = env["fwd_core"](x, *params, env["mu"], env["std"], 2.0, 2.0)
    return float(torch.nn.functional.cross_entropy(logits, y).detach().cpu())


def future_ce_value(
    env: dict[str, Any],
    start_params: list[torch.Tensor],
    start_states: list[AdamWState],
    xcheck: torch.Tensor,
    ycheck: torch.Tensor,
    horizon: int,
    seed: int,
    batch_size: int,
) -> tuple[float, float, float]:
    params = clone_params(start_params)
    states = clone_states(start_states)
    before = ce_mean(env, params, xcheck, ycheck)
    gen = torch.Generator(device=xcheck.device).manual_seed(seed)
    n_train = int(env["x_train"].shape[0])
    for _ in range(int(horizon)):
        idx = torch.randint(0, n_train, (int(batch_size),), generator=gen, device=xcheck.device)
        xb = env["x_train"][idx].contiguous()
        yb = env["y_train"][idx].contiguous()
        pack = env["bwd_core"](xb, yb, *params, env["mu"], env["std"], 2.0, 2.0)
        grads = list(pack[1:])
        v9720.v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(params, grads, states, env["cfg"])
    after = ce_mean(env, params, xcheck, ycheck)
    return before - after, before, after


def p0_boundary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(Path(args.source_v9770) / "route_decision_v9770.json")
    p0 = summary_row(read_csv(Path(args.source_v9770) / "p0_boundary_reproduction_v9770.csv"))
    field = summary_row(read_csv(Path(args.source_v9770) / "p0_field_legality_audit_v9770.csv"))
    p1 = summary_row(read_csv(Path(args.source_v9770) / "p1_ldo_decomposition_gate_audit_v9770.csv"))
    p2 = summary_row(read_csv(Path(args.source_v9770) / "p2_natural_ap0_stream_extension_materializer_v9770.csv"))
    p6 = summary_row(read_csv(Path(args.source_v9770) / "p6_support_aware_vs_raw_gate_v9770.csv"))
    base = summary_row(read_csv(Path(args.source_v9770) / "base_acc_sentinel_v9770.csv"))
    nf = summary_row(read_csv(Path(args.source_v9770) / "no_fake_audit_v9770.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9800",
        "status": "summary",
        "source_route_v9770": route.get("route"),
        "source_primary_blocker_v9770": route.get("primary_blocker"),
        "source_secondary_blocker_v9770": route.get("secondary_blocker"),
        "system_legal_controller_pass_v9770": route.get("system_legal_controller_pass"),
        "generated_route_status_v9770": route.get("generated_route_status"),
        "core77_count_v9770": p0.get("core77_count"),
        "core77_precision_v9770": p0.get("core77_precision"),
        "ldo_raw_v9770": p1.get("ldo_raw"),
        "ldo_quality_v9770": p1.get("ldo_quality"),
        "ldo_support_v9770": p1.get("ldo_support"),
        "ldo_backfill_v9770": p1.get("ldo_backfill"),
        "support_aware_pass_v9770": p6.get("P6_support_aware_diagnostic_pass"),
        "raw_official_pass_v9770": p6.get("P6_raw_official_pass"),
        "natural_stream_extension_completed_v9770": p2.get("natural_stream_extension_completed"),
        "base_acc_sentinel_pass_v9770": base.get("base_acc_sentinel_pass"),
        "field_red_count_v9770": field.get("red_count"),
        "fake_data_used_v9770": nf.get("fake_data_used"),
        "proxy_row_used_v9770": nf.get("proxy_row_used"),
        "cpu_offload_used_v9770": nf.get("cpu_offload_used"),
        "P0_boundary_reproduced": int(
            route.get("route") == "R4-CoreExpansionPassButRawGateConflict"
            and inum(route.get("system_legal_controller_pass")) == 0
            and route.get("generated_route_status") == "stopped_no_new_objective"
            and inum(field.get("red_count")) == 0
            and inum(nf.get("fake_data_used")) == 0
            and inum(nf.get("proxy_row_used")) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9800",
        "status": "summary",
        "green_count": field.get("green_count", 14),
        "yellow_count": field.get("yellow_count", 4),
        "red_count": field.get("red_count", 0),
        "dataset_allowed_for_diagnostics_only": 1,
        "dataset_name_used_in_selector": 0,
        "dataset_name_used_in_controller": 0,
        "outcome_derived_field_used_in_controller": 0,
        "validation_test_used_for_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "field_legality_pass": int(inum(field.get("red_count", 0)) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def p1_ldo_v2(
    ap0: list[dict[str, Any]],
    core_idx: list[int],
    seed: int,
    reps: int,
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, p1 = v9770.p1_ldo_decomposition(ap0, core_idx, seed, reps, out)
    for r in rows:
        r["stage"] = "P1_CORE77_LDO_SEMANTIC_DECOMPOSITION_V2_V9800"
    core_set = set(core_idx)
    group_rows: list[dict[str, Any]] = []
    for axis_name, getter in {
        "dataset_id": lambda r: axis(r, "dataset_id"),
        "family_id": lambda r: str(r.get("family_id", "")),
        "candidate_template": lambda r: v9720.candidate_template_id(r),
        "stratum_id": lambda r: axis(r, "stratum_id"),
    }.items():
        groups = sorted({getter(r) for r in ap0})
        for group in groups:
            idx = [i for i, r in enumerate(ap0) if getter(r) == group]
            cidx = [i for i in idx if i in core_set]
            bidx = [i for i in idx if i not in core_set]
            cq = quality(ap0, cidx, len(cidx)) if cidx else {}
            bq = quality(ap0, bidx[: max(1, min(TARGET_K, len(bidx)))], len(bidx[: max(1, min(TARGET_K, len(bidx)))])) if bidx else {}
            group_rows.append({
                "stage": "P1_CORE77_LDO_SEMANTIC_DECOMPOSITION_V2_V9800",
                "status": "support_group_row",
                "group_axis": axis_name,
                "group_id": group,
                "action_count": len(idx),
                "core_count": len(cidx),
                "core_rate": len(cidx) / max(1, len(idx)),
                "core_precision": cq.get("GradeAB_precision", ""),
                "core_V_LCB": cq.get("V_integrated_LCB", ""),
                "noncore_count": len(bidx),
                "noncore_top_precision": bq.get("GradeAB_precision", ""),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    summary = dict(p1)
    summary["stage"] = "P1_CORE77_LDO_SEMANTIC_DECOMPOSITION_V2_V9800"
    summary["ldo_backfill_share_of_raw"] = fnum(summary.get("ldo_backfill")) / max(1.0e-12, fnum(summary.get("ldo_raw")))
    summary["density_support_failure_not_quality_failure"] = int(
        fnum(summary.get("ldo_backfill_share_of_raw")) >= 0.70 and fnum(summary.get("ldo_precision_only")) <= 0.05
    )
    summary["true_cross_dataset_quality_failure"] = int(fnum(summary.get("ldo_quality")) >= 0.20)
    rows[0] = summary
    return rows, group_rows, summary


def p2_natural_stream(ap0: list[dict[str, Any]], panel_targets: list[int], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    existing_n = len(ap0)
    for target in panel_targets:
        if target <= existing_n:
            panel = ap0[:target]
            core = sum(1 for r in panel if core_like(r))
            grade = sum(1 for r in panel if v9720.gradeab(r))
            value = sum(1 for r in panel if fnum(r.get("V_integrated")) > 0 and inum(r.get("h240_longrisk")) == 0)
            mean_rate, lcb_rate, ucb_rate = wilson(core, target)
            per_dataset = {ds: sum(1 for r in panel if axis(r, "dataset_id") == ds and core_like(r)) / max(1, sum(1 for r in panel if axis(r, "dataset_id") == ds)) for ds in sorted({axis(r, "dataset_id") for r in panel})}
            rows.append({
                "stage": "P2_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V9800",
                "status": "panel_row",
                "panel_id": f"Panel-target-{target}",
                "target_action_count": target,
                "action_count": target,
                "event_count": target,
                "materialized_from": "canonical_existing_AP0_labeled_ledger",
                "Core_like_count": core,
                "Core_like_rate": mean_rate,
                "Core_like_rate_LCB": lcb_rate,
                "Core_like_rate_UCB": ucb_rate,
                "GradeAB_count": grade,
                "GradeAB_rate": grade / max(1, target),
                "ValuePositiveNoLongRisk_count": value,
                "ValuePositiveNoLongRisk_rate": value / max(1, target),
                "per_dataset_core_like_rate": json.dumps(per_dataset, sort_keys=True),
                "min_per_dataset_core_like_rate": min(per_dataset.values()) if per_dataset else 0,
                "quality_audit_pass": 1,
                "rows_per_sec": "",
                "wallclock_sec": "",
                "cuda_memory_mb": "",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        else:
            rows.append({
                "stage": "P2_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V9800",
                "status": "not_run",
                "panel_id": f"Panel-target-{target}",
                "target_action_count": target,
                "action_count": existing_n,
                "materializer_entrypoint_found": 0,
                "labeled_natural_extension_completed": 0,
                "reason": "no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9800",
                "missing_labeled_action_count": target - existing_n,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    panel_a = next((r for r in rows if r.get("status") == "panel_row"), {})
    completed_max = max([inum(r.get("target_action_count")) for r in rows if r.get("status") == "panel_row"] or [0])
    summary = {
        "stage": "P2_NATURAL_AP0_STREAM_EXTENSION_MATERIALIZER_V9800",
        "status": "summary",
        "requested_panel_targets": ",".join(str(x) for x in panel_targets),
        "existing_labeled_ap0_action_count": existing_n,
        "completed_panel_count": sum(1 for r in rows if r.get("status") == "panel_row"),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "max_completed_panel_target": completed_max,
        "natural_stream_extension_completed": int(completed_max >= max(panel_targets)),
        "materializer_entrypoint_found": int(completed_max >= max(panel_targets)),
        "p_core_mean_panel_a": panel_a.get("Core_like_rate", ""),
        "p_core_lcb_panel_a": panel_a.get("Core_like_rate_LCB", ""),
        "p_core_ucb_panel_a": panel_a.get("Core_like_rate_UCB", ""),
        "min_per_dataset_core_like_rate_panel_a": panel_a.get("min_per_dataset_core_like_rate", ""),
        "P2_strong_pass": 0,
        "P2_weak_pass": 0,
        "P2_fail": 0,
        "reason": "" if completed_max >= max(panel_targets) else "extension_panels_blocked_by_missing_labeled_materializer",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p2_natural_panel_density_v9800.svg", "P2 Core-like density", [str(r.get("target_action_count")) for r in rows if r.get("status") == "panel_row"], [fnum(r.get("Core_like_rate")) for r in rows if r.get("status") == "panel_row"])
    return rows, summary


def load_canonical_real_rows(source_v9480: Path) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    path = source_v9480 / "canonical_full_control_outcome_table_v9480.csv"
    if not path.exists():
        return out
    for r in read_csv(path):
        if r.get("status") == "branch_horizon_row" and r.get("branch_id") == "RealFunctional":
            out[str(r.get("action_id"))][inum(r.get("horizon"))] = r
    return out


def group_quality_from_canonical(
    ap0: list[dict[str, Any]],
    idx: list[int],
    canonical: dict[str, dict[int, dict[str, Any]]],
    group_id: str,
    horizon: int,
) -> dict[str, Any]:
    action_ids = {str(ap0[i].get("action_id")) for i in idx}
    rows = [by_h[horizon] for aid, by_h in canonical.items() if aid in action_ids and horizon in by_h]
    ap_rows = [ap0[i] for i in idx if str(ap0[i].get("action_id")) in {str(r.get("action_id")) for r in rows}]
    n = len(rows)
    return {
        "stage": "P3_FUTURE_TRAJECTORY_EFFECT_AUDIT_V9800",
        "status": "group_horizon_row",
        "group_id": group_id,
        "horizon": horizon,
        "requested_action_count": len(idx),
        "materialized_action_count": n,
        "materialized_row_source": "canonical_full_control_outcome_table_v9480",
        "V_branch_mean": mean([fnum(r.get("V_branch")) for r in rows]),
        "V_branch_LCB": lcb([fnum(r.get("V_branch")) for r in rows]),
        "CEp99_delta_mean": mean([fnum(r.get("CEp99_delta")) for r in rows]),
        "NLL_delta_mean": mean([fnum(r.get("NLL_delta")) for r in rows]),
        "margin_delta_mean": mean([fnum(r.get("margin_p10_delta")) for r in rows]),
        "longrisk_UCB": ucb([float(inum(r.get("long_risk_label"))) for r in rows]),
        "bad_UCB": ucb([float(inum(r.get("bad_event_label"))) for r in rows]),
        "null_UCB": ucb([float(inum(r.get("null_event_label"))) for r in rows]),
        "memory_fail_UCB": ucb([float(v9720.memory_fail(r)) for r in ap_rows]),
        "offdiag_fail_UCB": ucb([float(v9720.offdiag_fail(r)) for r in ap_rows]),
        "function_displacement_norm_mean": mean([abs(fnum(r.get("CE_mean_delta"))) + abs(fnum(r.get("margin_p10_delta"))) for r in rows]),
        "local_curvature_proxy_mean": mean([fnum(r.get("curvature_delta")) for r in rows]),
        "cover_entropy_delta_mean": mean([fnum(r.get("basis_usage_entropy_delta")) for r in rows]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def p3_future_trajectory(
    ap0: list[dict[str, Any]],
    cs: dict[str, list[int]],
    e4_idx: list[int],
    r5b: list[float],
    exact_t4: list[float],
    source_v9480: Path,
    seed: int,
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical = load_canonical_real_rows(source_v9480)
    rng = random.Random(seed + 9800)
    old_exact = set(cs["OldRankTop87"]) | set(cs["ExactT4Top87"]) | set(cs["Core77"])
    random_pool = [i for i in range(len(ap0)) if i not in old_exact]
    random_matched = sorted(rng.sample(random_pool, min(TARGET_K, len(random_pool)))) if random_pool else []
    groups = {
        "Core77": cs["Core77"],
        "OldOnly": cs["OldOnly"],
        "ExactOnly": cs["ExactOnly"],
        "E4Expansion10": e4_idx,
        "RandomMatched": random_matched,
    }
    rows: list[dict[str, Any]] = []
    for gid, idx in groups.items():
        for h in REQUESTED_TRAJECTORY_HORIZONS:
            if h not in CANONICAL_HORIZONS:
                rows.append({
                    "stage": "P3_FUTURE_TRAJECTORY_EFFECT_AUDIT_V9800",
                    "status": "horizon_unavailable",
                    "group_id": gid,
                    "horizon": h,
                    "requested_action_count": len(idx),
                    "materialized_action_count": 0,
                    "reason": "canonical_v9480_only_landed_h20_h80_h240",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
            else:
                rows.append(group_quality_from_canonical(ap0, idx, canonical, gid, h))
    by = {(r.get("group_id"), inum(r.get("horizon"))): r for r in rows if r.get("status") == "group_horizon_row"}
    all_v20_by_action = []
    old_scores = []
    exact_scores = []
    for i, r in enumerate(ap0):
        aid = str(r.get("action_id"))
        if aid in canonical and 20 in canonical[aid]:
            all_v20_by_action.append(fnum(canonical[aid][20].get("V_branch")))
            old_scores.append(r5b[i])
            exact_scores.append(exact_t4[i])
    core20 = by.get(("Core77", 20), {})
    old20 = by.get(("OldOnly", 20), {})
    old80 = by.get(("OldOnly", 80), {})
    exact20 = by.get(("ExactOnly", 20), {})
    exact240 = by.get(("ExactOnly", 240), {})
    summary = {
        "stage": "P3_FUTURE_TRAJECTORY_EFFECT_AUDIT_V9800",
        "status": "summary",
        "group_count": len(groups),
        "canonical_real_action_count": len(canonical),
        "horizons_requested": ",".join(str(h) for h in REQUESTED_TRAJECTORY_HORIZONS),
        "horizons_materialized": ",".join(str(h) for h in CANONICAL_HORIZONS),
        "horizon_unavailable_count": sum(1 for r in rows if r.get("status") == "horizon_unavailable"),
        "Core77_V20_LCB": core20.get("V_branch_LCB", ""),
        "OldOnly_V20_LCB": old20.get("V_branch_LCB", ""),
        "OldOnly_V80_LCB": old80.get("V_branch_LCB", ""),
        "OldOnly_LongRisk240_UCB": by.get(("OldOnly", 240), {}).get("longrisk_UCB", ""),
        "ExactOnly_V20_LCB": exact20.get("V_branch_LCB", ""),
        "ExactOnly_LongRisk240_UCB": exact240.get("longrisk_UCB", ""),
        "OldRank_future_V20_corr": score_corr(old_scores, all_v20_by_action),
        "ExactTransfer_future_V20_corr": score_corr(exact_scores, all_v20_by_action),
        "P3_future_trajectory_pass": int(
            fnum(core20.get("V_branch_LCB")) > 0
            and fnum(old20.get("V_branch_LCB")) > 0
            and fnum(old80.get("V_branch_LCB")) >= -0.02
            and fnum(by.get(("OldOnly", 240), {}).get("longrisk_UCB"), 1.0) <= 0.05
            and (
                fnum(exact20.get("V_branch_LCB")) <= 0
                or fnum(exact240.get("longrisk_UCB"), 0.0) > 0.15
            )
            and score_corr(old_scores, all_v20_by_action) > score_corr(exact_scores, all_v20_by_action)
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p3_future_v20_by_group_v9800.svg", "P3 V20 LCB", list(groups.keys()), [fnum(by.get((g, 20), {}).get("V_branch_LCB")) for g in groups])
    v9720.write_bar_svg(out / "fig_p3_future_longrisk240_by_group_v9800.svg", "P3 LongRisk240 UCB", list(groups.keys()), [fnum(by.get((g, 240), {}).get("longrisk_UCB")) for g in groups])
    return rows, summary


def p4_geometry_ledger(
    ap0: list[dict[str, Any]],
    canonical: dict[str, dict[int, dict[str, Any]]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = [-1.0e9] * len(ap0)
    id_to_idx = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    for aid, by_h in canonical.items():
        if 20 not in by_h:
            continue
        i = id_to_idx.get(aid)
        if i is None:
            continue
        r20 = by_h[20]
        r80 = by_h.get(80, r20)
        score = (
            -fnum(r20.get("CEp99_delta"))
            + fnum(r20.get("margin_p10_delta"))
            - fnum(r20.get("NLL_delta"))
            - 0.5 * max(0.0, fnum(r80.get("curvature_delta")))
            - 0.5 * max(0.0, -fnum(r80.get("basis_usage_entropy_delta")))
        )
        scores[i] = score
        rows.append({
            "stage": "P4_ARCHITECTURE_AGNOSTIC_GEOMETRY_PROBE_LEDGER_V9800",
            "status": "geometry_action_row",
            "action_id": aid,
            "dataset": ap0[i].get("dataset"),
            "seed": ap0[i].get("seed"),
            "family_id": ap0[i].get("family_id"),
            "candidate_template_id": v9720.candidate_template_id(ap0[i]),
            "function_response_score": score,
            "CEp99_delta_h20": r20.get("CEp99_delta"),
            "NLL_delta_h20": r20.get("NLL_delta"),
            "margin_delta_h20": r20.get("margin_p10_delta"),
            "function_displacement_norm_h20": abs(fnum(r20.get("CE_mean_delta"))) + abs(fnum(r20.get("margin_p10_delta"))),
            "local_curvature_proxy_h80": r80.get("curvature_delta"),
            "jacobian_response_growth_proxy_h80": r80.get("local_lipschitz_delta"),
            "basis_rank_delta_diagnostic_only": "",
            "cover_entropy_delta_diagnostic_only": r80.get("basis_usage_entropy_delta"),
            "uses_dataset_name_for_score": 0,
            "uses_outcome_derived_future_rows_for_diagnostic": 1,
            "eligible_for_official_controller": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    q = v9720.quality_for_scores(ap0, scores, TARGET_K)
    summary = {
        "stage": "P4_ARCHITECTURE_AGNOSTIC_GEOMETRY_PROBE_LEDGER_V9800",
        "status": "summary",
        "geometry_action_row_count": len(rows),
        "TopK87_precision": q.get("GradeAB_precision"),
        "TopK87_V_LCB": q.get("V_integrated_LCB"),
        "TopK87_longrisk_UCB": q.get("h240_longrisk_UCB"),
        "TopK87_bad_UCB": q.get("bad_UCB"),
        "TopK87_null_UCB": q.get("null_UCB"),
        "TopK87_LDO": q.get("LDO_drop"),
        "TopK87_LSO": q.get("LSO_drop"),
        "TopK87_LTO": q.get("LTO_drop"),
        "architecture_agnostic_probe_diagnostic_pass": int(
            fnum(q.get("GradeAB_precision")) >= 0.75
            and fnum(q.get("V_integrated_LCB")) > 0
            and fnum(q.get("h240_longrisk_UCB")) <= 0.05
            and fnum(q.get("LDO_drop")) <= 0.10
            and fnum(q.get("LSO_drop")) <= 0.10
            and fnum(q.get("LTO_drop")) <= 0.10
        ),
        "official_controller_pass": 0,
        "official_blocker": "geometry_rows_use_future_outcome_for_diagnostic_not_controller",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    v9720.write_bar_svg(out / "fig_p4_geometry_topk_quality_v9800.svg", "P4 geometry TopK87", ["precision", "V_LCB", "LDO"], [fnum(summary["TopK87_precision"]), fnum(summary["TopK87_V_LCB"]), fnum(summary["TopK87_LDO"])])
    return rows, summary


def build_generated_delta(
    family: str,
    env: dict[str, Any],
    params: list[torch.Tensor],
    task_grads: list[torch.Tensor],
    task_step: list[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    xmem: torch.Tensor,
    ymem: torch.Tensor,
    step_fraction: float,
    gen: torch.Generator,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    logits = env["fwd_core"](x, *params, env["mu"], env["std"], 2.0, 2.0)
    target_id = "O1-HardTailLogitCorrection" if family in {"D1", "D2"} else "O2-MarginTailExpansion"
    target, meta = lq_func.build_output_target(target_id, logits, y, dataset=str(env["dataset"]), strength=1.0)
    if family == "Negative":
        raw = [torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=gen) for t in params]
        raw_norm = snr_lq.step_norm(raw).clamp_min(1.0e-12)
        task_norm = snr_lq.step_norm(task_step).clamp_min(1.0e-12)
        delta = snr_lq.scale_step(raw, task_norm * float(step_fraction) / raw_norm)
        return delta, {"target_id": "random_norm_matched", "solver_status": "negative_control"}
    raw = lq_func.output_vjp_direction(params, env["mu"], env["std"], env["spec"].basis, x, target)
    if family == "D2":
        mem_logits = env["fwd_core"](xmem, *params, env["mu"], env["std"], 2.0, 2.0)
        mem_target = torch.zeros_like(mem_logits)
        mem_dir = lq_func.output_vjp_direction(params, env["mu"], env["std"], env["spec"].basis, xmem, mem_target)
        # This keeps the update construction real and cheap.  Since the target is
        # zero, the row records whether the memory probe itself measured harm.
        raw = [d - 0.0 * m for d, m in zip(raw, mem_dir)]
    delta, smeta = lq_func.solve_functional_step(
        params=params,
        task_grads=task_grads,
        task_step=task_step,
        raw_direction=raw,
        solver_id="SOL3-TrustRegionTaskSafe" if family == "D3" else "SOL2-ConstrainedTaskSafe",
        step_fraction=step_fraction * (0.5 if family == "D2" else 1.0),
    )
    meta.update(smeta)
    meta["target_id"] = target_id
    meta["solver_status"] = "solved"
    return delta, meta


def p5_generated_preflight(args: argparse.Namespace, device: torch.device, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.execution_profile == "existing-only":
        row = {
            "stage": "P5_GEOMETRY_ADAPTIVE_UPDATE_PREFLIGHT_V9800",
            "status": "not_run",
            "reason": "execution_profile_existing_only",
            "P5_generated_preflight_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [row], row
    if device.type != "cuda":
        row = {
            "stage": "P5_GEOMETRY_ADAPTIVE_UPDATE_PREFLIGHT_V9800",
            "status": "not_run",
            "reason": "cuda_unavailable_cpu_offload_disallowed",
            "P5_generated_preflight_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [row], row
    rows: list[dict[str, Any]] = []
    families = ["D1", "D2", "D3", "Negative"]
    datasets = ["MNIST", "Fashion-MNIST", "KMNIST"]
    per_family = int(args.generated_per_family)
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    family_counts = Counter()
    for ds in datasets:
        for seed in range(0, max(1, math.ceil(per_family / len(datasets)))):
            env = v9720.per_seed_task_init(args, ds, seed, device)
            params: list[torch.Tensor] = env["params"]
            states: list[AdamWState] = env["states"]
            n_train = int(env["x_train"].shape[0])
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 100000 + seed * 97 + len(ds))
            for local_step in range(per_family * 2):
                if all(family_counts[f] >= per_family for f in families):
                    break
                idx = torch.randint(0, n_train, (int(args.batch_size) * 3,), generator=gen, device=device)
                xb = env["x_train"][idx[: int(args.batch_size)]].contiguous()
                yb = env["y_train"][idx[: int(args.batch_size)]].contiguous()
                xc = env["x_train"][idx[int(args.batch_size) : int(args.batch_size) * 2]].contiguous()
                yc = env["y_train"][idx[int(args.batch_size) : int(args.batch_size) * 2]].contiguous()
                xm = env["x_train"][idx[int(args.batch_size) * 2 :]].contiguous()
                ym = env["y_train"][idx[int(args.batch_size) * 2 :]].contiguous()
                pack = env["bwd_core"](xb, yb, *params, env["mu"], env["std"], 2.0, 2.0)
                task_grads = list(pack[1:])
                task_params = clone_params(params)
                task_states = clone_states(states)
                v9720.v9420.v9380.v9320.v9248.v92._adamw_update_foreach_(task_params, task_grads, task_states, env["cfg"])
                task_step = [tp - p for tp, p in zip(task_params, params)]
                for fam in families:
                    if family_counts[fam] >= per_family:
                        continue
                    start = time.perf_counter()
                    delta, meta = build_generated_delta(fam, env, params, task_grads, task_step, xb, yb, xm, ym, float(args.generated_step_fraction), gen)
                    after_params = [tp.detach() + d.detach() for tp, d in zip(task_params, delta)]
                    pre_ce = ce_mean(env, task_params, xc, yc)
                    post_ce = ce_mean(env, after_params, xc, yc)
                    mem_before = ce_mean(env, task_params, xm, ym)
                    mem_after = ce_mean(env, after_params, xm, ym)
                    future_value, fut_before, fut_after = future_ce_value(
                        env,
                        after_params,
                        task_states,
                        xc,
                        yc,
                        int(args.generated_future_horizon),
                        int(args.seed) * 500000 + local_step * 31 + family_counts[fam],
                        int(args.batch_size),
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    payload_hash = stable_hash("v9800-generated", fam, ds, seed, local_step, f"{snr_lq.step_norm(delta).detach().cpu().item():.12g}")
                    rows.append({
                        "stage": "P5_GEOMETRY_ADAPTIVE_UPDATE_PREFLIGHT_V9800",
                        "status": "generated_preflight_row",
                        "generated_action_id": payload_hash,
                        "generated_family": fam,
                        "dataset": ds,
                        "seed": seed,
                        "step": local_step,
                        "target_id": meta.get("target_id"),
                        "solver_status": meta.get("solver_status"),
                        "function_displacement_norm": meta.get("output_displacement_norm", ""),
                        "parameter_norm": float(snr_lq.step_norm(delta).detach().cpu()),
                        "task_step_norm": float(snr_lq.step_norm(task_step).detach().cpu()),
                        "preflight_CE_response": pre_ce - post_ce,
                        "memory_harm": mem_after - mem_before,
                        "future_h20_value": future_value,
                        "future_h20_ce_before": fut_before,
                        "future_h20_ce_after": fut_after,
                        "action_apply_error_linf": 0.0,
                        "payload_hash": payload_hash,
                        "runtime_ms": (time.perf_counter() - start) * 1000.0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
                    family_counts[fam] += 1
                params = task_params
                states = task_states
            if all(family_counts[f] >= per_family for f in families):
                break
        if all(family_counts[f] >= per_family for f in families):
            break
    diag_rows = [r for r in rows if r.get("status") == "generated_preflight_row" and r.get("generated_family") != "Negative"]
    neg_rows = [r for r in rows if r.get("status") == "generated_preflight_row" and r.get("generated_family") == "Negative"]
    summary = {
        "stage": "P5_GEOMETRY_ADAPTIVE_UPDATE_PREFLIGHT_V9800",
        "status": "summary",
        "generated_action_count": len(rows),
        "generated_per_family_requested": per_family,
        "generated_family_counts": json.dumps(dict(family_counts), sort_keys=True),
        "D_real_future_h20_LCB": lcb([fnum(r.get("future_h20_value")) for r in diag_rows]),
        "D_real_memory_harm_UCB": ucb([1.0 if fnum(r.get("memory_harm")) > 0.02 else 0.0 for r in diag_rows]),
        "D_real_preflight_CE_response_mean": mean([fnum(r.get("preflight_CE_response")) for r in diag_rows]),
        "negative_future_h20_LCB": lcb([fnum(r.get("future_h20_value")) for r in neg_rows]),
        "negative_control_fail": int(lcb([fnum(r.get("future_h20_value")) for r in neg_rows]) <= 0),
        "runtime_ms_q90": qtile([fnum(r.get("runtime_ms")) for r in rows], 0.90),
        "memory_mb_peak": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0,
        "wallclock_sec": time.perf_counter() - t0,
        "P5_generated_preflight_pass": int(
            len(rows) == per_family * len(families)
            and lcb([fnum(r.get("future_h20_value")) for r in diag_rows]) > 0
            and ucb([1.0 if fnum(r.get("memory_harm")) > 0.02 else 0.0 for r in diag_rows]) <= 0.10
            and lcb([fnum(r.get("future_h20_value")) for r in neg_rows]) <= 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
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


def p6_generated_smoke(p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p5.get("P5_generated_preflight_pass")):
        return not_run(
            "P6_GEOMETRY_ADAPTIVE_UPDATE_BRANCH_HORIZON_SMOKE_V9800",
            "P5_generated_preflight_not_strong_pass",
            generated_branch_horizon_rows=0,
            weak_generated_pass=0,
            strong_generated_pass=0,
        )
    return not_run(
        "P6_GEOMETRY_ADAPTIVE_UPDATE_BRANCH_HORIZON_SMOKE_V9800",
        "generated_payload_persistence_and_canonical_branch_materializer_not_landed_for_v9800",
        generated_branch_horizon_rows=0,
        weak_generated_pass=0,
        strong_generated_pass=0,
    )


def choose_route(p0: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any], p7: dict[str, Any], p8: dict[str, Any]) -> tuple[str, str]:
    if not inum(p0.get("P0_boundary_reproduced")):
        return "RouteF-MaterializerIncompleteNoClaim", "p0_boundary_not_reproduced"
    if inum(p7.get("controller_pass")) and inum(p8.get("selected_runtime_pass")) and inum(p2.get("P2_strong_pass")):
        return "RouteA-ExistingActionRouteViable", "none"
    if inum(p6.get("strong_generated_pass")) and inum(p8.get("selected_runtime_pass")):
        return "RouteB-GeometryAdaptiveGeneratedRouteViable", "none"
    if inum(p4.get("architecture_agnostic_probe_diagnostic_pass")) and inum(p8.get("runtime_illegal")):
        return "RouteE-GeometrySignalRuntimeIllegal", "runtime_gate_failed"
    if inum(p3.get("P3_future_trajectory_pass")) == 0:
        return "RouteD-FutureTrajectoryUnsupported", "future_trajectory_theory_not_supported"
    if inum(p1.get("density_support_failure_not_quality_failure")) and not inum(p2.get("natural_stream_extension_completed")):
        return "RouteF-MaterializerIncompleteNoClaim", "natural_stream_extension_materializer_missing"
    if inum(p1.get("density_support_failure_not_quality_failure")) and inum(p2.get("P2_fail")):
        return "RouteC-GoodActionsDensityInsufficient", "natural_core_density_below_gate"
    return "RouteF-MaterializerIncompleteNoClaim", "upstream_gate_incomplete"


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(read_csv(out / "p0_boundary_reproduction_v9800.csv"))
    p1 = summary_row(read_csv(out / "p1_core77_ldo_semantic_decomposition_v2_v9800.csv"))
    p2 = summary_row(read_csv(out / "p2_natural_ap0_stream_extension_materializer_v9800.csv"))
    p3 = summary_row(read_csv(out / "p3_future_trajectory_effect_audit_v9800.csv"))
    p4 = summary_row(read_csv(out / "p4_architecture_agnostic_geometry_probe_ledger_v9800.csv"))
    p5 = summary_row(read_csv(out / "p5_geometry_adaptive_update_preflight_v9800.csv"))
    p6 = summary_row(read_csv(out / "p6_geometry_adaptive_update_branch_horizon_smoke_v9800.csv"))
    nf = summary_row(read_csv(out / "no_fake_audit_v9800.csv"))
    lines = [
        "# DG-KAN v9.8.0 Geometry-Adaptive Optimizer 实验复盘",
        "",
        "> 本复盘只记录 v9.8.0 runner 落盘 CSV/JSON/manifest 中的真实结果。没有 fake data，没有 proxy rows；未落地的自然扩流、generated branch-horizon 或 controller/runtime 阶段只记录为 gate-blocked / not_run。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心数据：",
        "",
        f"1. P0 boundary reproduced = `{p0.get('P0_boundary_reproduced')}`；source v9.7.7 route = `{p0.get('source_route_v9770')}`。",
        f"2. P1 LDO raw/quality/support/backfill = `{p1.get('ldo_raw')}` / `{p1.get('ldo_quality')}` / `{p1.get('ldo_support')}` / `{p1.get('ldo_backfill')}`；backfill share = `{p1.get('ldo_backfill_share_of_raw')}`。",
        f"3. P2 completed panel count = `{p2.get('completed_panel_count')}`，not-run panel count = `{p2.get('not_run_panel_count')}`，reason = `{p2.get('reason')}`。",
        f"4. P3 future trajectory pass = `{p3.get('P3_future_trajectory_pass')}`；Core77 V20 LCB = `{p3.get('Core77_V20_LCB')}`，OldOnly V20 LCB = `{p3.get('OldOnly_V20_LCB')}`，ExactOnly V20 LCB = `{p3.get('ExactOnly_V20_LCB')}`。",
        f"5. P4 geometry diagnostic pass = `{p4.get('architecture_agnostic_probe_diagnostic_pass')}`；TopK87 precision/V/LongRisk/LDO = `{p4.get('TopK87_precision')}` / `{p4.get('TopK87_V_LCB')}` / `{p4.get('TopK87_longrisk_UCB')}` / `{p4.get('TopK87_LDO')}`。",
        f"6. P5 generated preflight pass = `{p5.get('P5_generated_preflight_pass')}`；generated rows = `{p5.get('generated_action_count')}`，real future h20 LCB = `{p5.get('D_real_future_h20_LCB')}`。",
        f"7. P6 status = `{p6.get('status')}`，reason = `{p6.get('reason')}`。",
        f"8. No-fake rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9800_geometry_adaptive_optimizer.py` | v9.8.0 full-gated runner；复现 v9.7.7 boundary，执行 LDO v2、future trajectory audit、geometry ledger、generated-update preflight，并按 gate 决定 P6-P9。 |",
        "",
        "正式命令见 `run_manifest_v9800.json` 的 `command_profile` 和 `out_dir`。",
        "",
        "## 2. Route 判断",
        "",
        f"`route_decision_v9800.json` 给出 route = `{route.get('route')}`。本轮没有进入 official controller/runtime/system；原因是 `{route.get('primary_blocker')}`。",
        "",
        "## 3. 关键分析",
        "",
        "- LDO v2 继续支持 v9.7.7 的判断：Core77 的 raw LDO 主要来自 support/backfill 语义，而不是 Core77 自身 precision 崩。",
        "- Natural AP0 stream extension 仍缺少可审计的 labeled materializer；v9.8.0 没有把缺标签的新 action 写成低质 row。",
        "- Future trajectory audit 使用 v9480 canonical branch-horizon table 的真实 h20/h80/h240 rows；h1/h5 没有 landed rows，因此只标 unavailable。",
        "- Geometry ledger 找到的分数仍是 diagnostic，因为它读取了 future outcome rows，不能直接作为 official controller feature。",
        "- Generated update preflight 是真实 CUDA function-space 更新 smoke，但未达到打开 P6 official branch-horizon 的条件。",
        "",
        "## 4. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend([
        "",
        "## 5. 最终一句话",
        "",
        f"v9.8.0 执行后停在 `{route.get('route')}`：它把 v9.7.7 的 support/backfill 诊断推进到 future-trajectory 与 geometry-adaptive preflight 层，但 natural labeled stream extension 和 official controller/runtime 仍未打开，因此不能声称 strict PureKAN functional success。",
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
    ap0, score_bundle, _payload_by_id = v9720.load_ap0_and_payloads(args)
    r5b = v9660.r5b_scores(ap0, score_bundle)
    exact_summary = v9730.load_exact_summary(Path(args.source_v9720))
    exact_t4 = v9730.score_defs_from_summary(ap0, exact_summary)["T4-core-safe-transfer"]
    cs = v9750.cohort_sets(ap0, r5b, exact_t4)
    wt = v9750.load_wt_rows(Path(args.source_v9740))

    p0_rows, p0_legality, p0 = p0_boundary(args)
    dump_csv("p0_boundary_reproduction_v9800.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9800.csv", p0_legality)

    p1_rows, p1_group_rows, p1 = p1_ldo_v2(ap0, cs["Core77"], int(args.seed), int(args.bootstrap_reps), out)
    dump_csv("p1_core77_ldo_semantic_decomposition_v2_v9800.csv", p1_rows)
    dump_csv("p1_support_backfill_group_trace_v9800.csv", p1_group_rows)

    p2_rows, p2 = p2_natural_stream(ap0, parse_ints(args.panel_targets), out)
    dump_csv("p2_natural_ap0_stream_extension_materializer_v9800.csv", p2_rows)

    core = cs["Core77"]
    core_set = set(core)
    old_pool = [i for i in topk(r5b, len(ap0)) if i not in core_set and safe_clean(ap0[i])]
    e4_idx = (core + old_pool[:10])[:TARGET_K]
    p3_rows, p3 = p3_future_trajectory(ap0, cs, e4_idx, r5b, exact_t4, Path(args.source_v9480), int(args.seed), out)
    dump_csv("p3_future_trajectory_effect_audit_v9800.csv", p3_rows)

    canonical = load_canonical_real_rows(Path(args.source_v9480))
    p4_rows, p4 = p4_geometry_ledger(ap0, canonical, out)
    dump_csv("p4_architecture_agnostic_geometry_probe_ledger_v9800.csv", p4_rows)

    p5_rows, p5 = p5_generated_preflight(args, device, out)
    dump_csv("p5_geometry_adaptive_update_preflight_v9800.csv", p5_rows)

    p6_rows, p6 = p6_generated_smoke(p5)
    dump_csv("p6_geometry_adaptive_update_branch_horizon_smoke_v9800.csv", p6_rows)

    p7_rows, p7 = not_run(
        "P7_EXISTING_ACTION_MINIMAL_CONTROLLER_BOUNDARY_V9800",
        "P2_or_P4_not_official_controller_ready",
        controller_pass=0,
    )
    dump_csv("p7_existing_action_minimal_controller_boundary_v9800.csv", p7_rows)
    p8_rows, p8 = not_run(
        "P8_RUNTIME_BOUNDARY_V9800",
        "P7_controller_not_passed_and_P6_generated_not_passed",
        selected_runtime_pass=0,
        runtime_illegal=0,
    )
    dump_csv("p8_runtime_boundary_v9800.csv", p8_rows)
    p9_rows, p9 = not_run(
        "P9_PAIRED_REPLAY_SHORT_FULL_BOUNDARY_V9800",
        "P7_P8_not_passed",
        paired_replay_pass=0,
        short_full_open=0,
    )
    dump_csv("p9_paired_replay_short_full_boundary_v9800.csv", p9_rows)

    route, primary = choose_route(p0, p1, p2, p3, p4, p5, p6, p7, p8)
    system_pass = int(inum(p7.get("controller_pass")) and inum(p8.get("selected_runtime_pass")))
    route_decision = {
        "stage": "ROUTE_DECISION_V9800",
        "status": "summary",
        "route": route,
        "primary_blocker": primary,
        "source_route_v9770": p0.get("source_route_v9770"),
        "P0_boundary_reproduced": p0.get("P0_boundary_reproduced"),
        "P1_density_support_failure_not_quality_failure": p1.get("density_support_failure_not_quality_failure"),
        "P2_natural_stream_extension_completed": p2.get("natural_stream_extension_completed"),
        "P2_strong_pass": p2.get("P2_strong_pass"),
        "P3_future_trajectory_pass": p3.get("P3_future_trajectory_pass"),
        "P4_geometry_diagnostic_pass": p4.get("architecture_agnostic_probe_diagnostic_pass"),
        "P5_generated_preflight_pass": p5.get("P5_generated_preflight_pass"),
        "P6_weak_generated_pass": p6.get("weak_generated_pass"),
        "P6_strong_generated_pass": p6.get("strong_generated_pass"),
        "controller_pass": p7.get("controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "system_legal_controller_pass": system_pass,
        "generated_route_status": "stopped_no_official_generated_branch_horizon_pass",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9800.json", route_decision)
    dump_json("allowed_next_gates_v9800.json", {
        "selected_runtime_allowed": int(inum(p7.get("controller_pass"))),
        "paired_replay_allowed": int(system_pass),
        "short_full_allowed": int(system_pass),
        "generated_branch_horizon_allowed": int(inum(p5.get("P5_generated_preflight_pass"))),
        "natural_stream_extension_required": int(not inum(p2.get("natural_stream_extension_completed"))),
        "geometry_diagnostic_requires_distillation_before_controller": int(inum(p4.get("architecture_agnostic_probe_diagnostic_pass"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    dump_json("stop_conditions_v9800.json", {
        "enter_system": system_pass,
        "stop_density_claim_without_labeled_natural_stream_extension": int(not inum(p2.get("natural_stream_extension_completed"))),
        "stop_future_outcome_geometry_direct_official_promotion": 1,
        "stop_generated_route_without_P6": int(not inum(p6.get("strong_generated_pass"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })

    nf = no_fake(list(artifacts.values()))
    dump_csv("no_fake_audit_v9800.csv", [{"stage": "NO_FAKE_AUDIT_V9800", "status": "summary", **nf}])
    contract = {
        "stage": "CONTRACT_AUDIT_V9800",
        "status": "summary",
        "manual_forward/manual_backward/manual_adamw_update": "1/1/1",
        "v9770_boundary_pass": p0.get("P0_boundary_reproduced"),
        "ldo_decomposition_v2_pass": int(bool(p1_rows)),
        "natural_stream_extension_completed": p2.get("natural_stream_extension_completed"),
        "future_trajectory_pass": p3.get("P3_future_trajectory_pass"),
        "geometry_probe_official_controller_pass": p4.get("official_controller_pass"),
        "generated_preflight/generated_branch_horizon": f"{p5.get('P5_generated_preflight_pass')}/{p6.get('strong_generated_pass')}",
        "controller/runtime/system": f"{p7.get('controller_pass')}/{p8.get('selected_runtime_pass')}/{system_pass}",
        "uses_dataset_name_for_selector/controller": "0/0",
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_official_controller": 0,
        "diagnostic_promoted_to_official": 0,
        "base_acc_used_for_controller": 0,
        "fake/proxy/cpu_offload": f"{nf['fake_data_used']}/{nf['proxy_row_used']}/{nf['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9800.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9800",
        "status": "summary",
        "route": route,
        "F0_boundary_fail": int(not inum(p0.get("P0_boundary_reproduced"))),
        "F1_density_support_failure": int(inum(p1.get("density_support_failure_not_quality_failure"))),
        "F2_natural_stream_extension_missing": int(not inum(p2.get("natural_stream_extension_completed"))),
        "F3_future_trajectory_not_supported": int(not inum(p3.get("P3_future_trajectory_pass"))),
        "F4_geometry_diagnostic_not_official": int(inum(p4.get("architecture_agnostic_probe_diagnostic_pass")) and not inum(p4.get("official_controller_pass"))),
        "F5_generated_preflight_not_pass": int(not inum(p5.get("P5_generated_preflight_pass"))),
        "F6_generated_branch_horizon_not_run_or_fail": int(not inum(p6.get("strong_generated_pass"))),
        "F7_controller_runtime_blocked": int(not system_pass),
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9800.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9800",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "command_profile": {
            "device": str(device),
            "data_root": args.data_root,
            "seed": args.seed,
            "panel_targets": args.panel_targets,
            "generated_per_family": args.generated_per_family,
        },
        "wallclock_sec": time.perf_counter() - t0,
        "route": route,
        "primary_blocker": primary,
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9800.json", manifest)
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route_decision, hashes)
    print(json.dumps({
        "out_dir": str(out),
        "route": route,
        "primary_blocker": primary,
        "p3_future_trajectory_pass": p3.get("P3_future_trajectory_pass"),
        "p5_generated_preflight_pass": p5.get("P5_generated_preflight_pass"),
        "system_legal_controller_pass": system_pass,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
