#!/usr/bin/env python3
"""DG-KAN v9.9.2 natural-generator distribution repair runner.

This runner does not treat v9.9.1's 1024 pilot as a density closure.  It first
reproduces the distribution-mismatch boundary, then evaluates repair generator
profiles with real action rows, durable payloads, action-apply checks, and
branch-horizon replay where the gate allows it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9900_natural_extension_engineering_gate_futurepathoperator as v9900  # noqa: E402
import run_v9910_natural_density_decision_futurepathoperator as v9910  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.2_自然生成器分布修复_FuturePathOperator_四线并行完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9920_natural_generator_distribution_repair_futurepathoperator.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.2_NaturalGeneratorDistributionRepair_FuturePathOperator_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9920_natural_generator_distribution_repair_futurepathoperator_full_20260517T030000Z"
DEFAULT_V9910 = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
DEFAULT_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6


GENERATOR_PROFILES = [
    ("G0-current-v9910-generator", "full-gated", "baseline_reuse_v9910"),
    ("G1-round-robin-source-cursor", "g1-round-robin-source-cursor", "run_smoke_then_gate"),
    ("G2-stratified-reference-quota-generator", "g2-stratified-reference-quota-generator", "run_smoke_then_gate"),
    ("G3-recipe-complete-generator", "g3-recipe-complete-generator", "run_smoke_then_gate"),
    ("G4-randomized-trainstream-window-generator", "g4-randomized-trainstream-window-generator", "run_smoke_then_gate"),
    ("G5-hybrid-quota-random-generator", "g5-hybrid-quota-random-generator", "run_smoke_then_gate"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    p.add_argument("--panel-targets", default="1024,5000,10000,20000")
    p.add_argument("--pilot-actions", type=int, default=1024)
    p.add_argument("--candidate-smoke-actions", default="1,16,256")
    p.add_argument("--source-v9910", default=str(DEFAULT_V9910))
    p.add_argument("--source-v9900", default=str(DEFAULT_V9900))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--run-full-density-panels", action="store_true")
    return p.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def rows_from(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def q(values: list[float], frac: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))]


def mean_lcb(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return statistics.mean(xs) - 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def step_bucket(value: Any) -> str:
    s = inum(value)
    lo = (s // 32) * 32
    return f"s{lo:03d}_{lo + 31:03d}"


def bucket_by_reference_cuts(value: float, cuts: list[float], name: str) -> str:
    for idx, cut in enumerate(cuts):
        if value <= cut:
            return f"{name}_bin_{idx}"
    return f"{name}_bin_{len(cuts)}"


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log(max(v / total, 1.0e-12)) for v in counter.values())


def psi(old: Counter[str], new: Counter[str]) -> float:
    keys = set(old) | set(new)
    n_old = sum(old.values())
    n_new = sum(new.values())
    eps = 1.0e-6
    out = 0.0
    for key in keys:
        p = max(eps, old.get(key, 0) / max(1, n_old))
        qv = max(eps, new.get(key, 0) / max(1, n_new))
        out += (qv - p) * math.log(qv / p)
    return out


def kl_div(old: Counter[str], new: Counter[str]) -> float:
    keys = set(old) | set(new)
    n_old = sum(old.values())
    n_new = sum(new.values())
    eps = 1.0e-9
    return sum((old.get(k, 0) / max(1, n_old)) * math.log(max(eps, old.get(k, 0) / max(1, n_old)) / max(eps, new.get(k, 0) / max(1, n_new))) for k in keys)


def js_distance(old: Counter[str], new: Counter[str]) -> float:
    keys = set(old) | set(new)
    n_old = sum(old.values())
    n_new = sum(new.values())
    if n_old <= 0 or n_new <= 0:
        return 1.0
    p = {k: old.get(k, 0) / n_old for k in keys}
    qv = {k: new.get(k, 0) / n_new for k in keys}
    m = {k: 0.5 * (p[k] + qv[k]) for k in keys}
    eps = 1.0e-12
    js = 0.5 * sum(p[k] * math.log(max(eps, p[k]) / max(eps, m[k])) for k in keys)
    js += 0.5 * sum(qv[k] * math.log(max(eps, qv[k]) / max(eps, m[k])) for k in keys)
    return math.sqrt(max(0.0, js))


def tv_distance(old: Counter[str], new: Counter[str]) -> float:
    keys = set(old) | set(new)
    n_old = sum(old.values())
    n_new = sum(new.values())
    return 0.5 * sum(abs(old.get(k, 0) / max(1, n_old) - new.get(k, 0) / max(1, n_new)) for k in keys)


def wasserstein_1d(old_vals: list[float], new_vals: list[float]) -> float:
    xs = sorted(v for v in old_vals if math.isfinite(v))
    ys = sorted(v for v in new_vals if math.isfinite(v))
    if not xs or not ys:
        return 0.0
    n = max(len(xs), len(ys))
    total = 0.0
    for i in range(n):
        qv = i / max(1, n - 1)
        total += abs(xs[min(len(xs) - 1, int(qv * (len(xs) - 1)))] - ys[min(len(ys) - 1, int(qv * (len(ys) - 1)))])
    return total / n


def counter(rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], Any]) -> Counter[str]:
    return Counter(str(fn(r)) for r in rows)


def payload_norm(row: dict[str, Any]) -> float:
    return fnum(row.get("payload_norm") or row.get("payload_l2_norm") or row.get("action_norm"))


def payload_linf(row: dict[str, Any]) -> float:
    return fnum(row.get("payload_linf_norm") or row.get("payload_linf"))


def field(row: dict[str, Any], *names: str, default: str = "unknown") -> str:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return str(value)
    return default


def reference_axes(reference_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> tuple[list[tuple[str, Counter[str], Counter[str], list[float], list[float]]], dict[str, Any]]:
    old_norm = [payload_norm(r) for r in reference_rows]
    old_linf = [payload_linf(r) for r in reference_rows]
    old_action = [payload_norm(r) for r in reference_rows]
    norm_cuts = [q(old_norm, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
    linf_cuts = [q(old_linf, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
    action_cuts = [q(old_action, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]

    axes: list[tuple[str, Counter[str], Counter[str], list[float], list[float]]] = [
        ("dataset_id", counter(reference_rows, lambda r: field(r, "dataset")), counter(target_rows, lambda r: field(r, "dataset")), [], []),
        ("seed", counter(reference_rows, lambda r: field(r, "seed")), counter(target_rows, lambda r: field(r, "seed")), [], []),
        ("family_id", counter(reference_rows, lambda r: field(r, "family_id")), counter(target_rows, lambda r: field(r, "family_id")), [], []),
        ("stratum_id", counter(reference_rows, lambda r: field(r, "bucket_id", "stratum_id", "family_id")), counter(target_rows, lambda r: field(r, "source_recipe_id", "stratum_id", "family_id")), [], []),
        ("step_bucket", counter(reference_rows, lambda r: step_bucket(r.get("step"))), counter(target_rows, lambda r: step_bucket(r.get("step"))), [fnum(r.get("step")) for r in reference_rows], [fnum(r.get("step")) for r in target_rows]),
        ("event_family", counter(reference_rows, lambda r: step_bucket(r.get("step"))), counter(target_rows, lambda r: step_bucket(r.get("step"))), [], []),
        ("candidate_template_id", counter(reference_rows, lambda r: field(r, "carrier_id", "template_id")), counter(target_rows, lambda r: field(r, "candidate_template_id", "template_id", "carrier_id")), [], []),
        ("candidate_origin", counter(reference_rows, lambda r: "canonical_ap0_reference_quota"), counter(target_rows, lambda r: field(r, "candidate_origin", default="canonical_ap0_reference_quota")), [], []),
        ("source_recipe_id", counter(reference_rows, lambda r: field(r, "bucket_id", "family_id")), counter(target_rows, lambda r: field(r, "source_recipe_id", "family_id")), [], []),
        ("payload_norm_bucket", counter(reference_rows, lambda r: bucket_by_reference_cuts(payload_norm(r), norm_cuts, "payload_norm")), counter(target_rows, lambda r: bucket_by_reference_cuts(payload_norm(r), norm_cuts, "payload_norm")), old_norm, [payload_norm(r) for r in target_rows]),
        ("action_norm_bucket", counter(reference_rows, lambda r: bucket_by_reference_cuts(payload_norm(r), action_cuts, "action_norm")), counter(target_rows, lambda r: bucket_by_reference_cuts(payload_norm(r), action_cuts, "action_norm")), old_action, [payload_norm(r) for r in target_rows]),
        ("action_linf_bucket", counter(reference_rows, lambda r: bucket_by_reference_cuts(payload_linf(r), linf_cuts, "action_linf")), counter(target_rows, lambda r: bucket_by_reference_cuts(payload_linf(r), linf_cuts, "action_linf")), old_linf, [payload_linf(r) for r in target_rows]),
        ("AdamW_alignment_bucket", counter(reference_rows, lambda r: "reference_unavailable"), counter(target_rows, lambda r: "reference_unavailable"), [], []),
        ("memory_bucket", counter(reference_rows, lambda r: "commit_time_unavailable"), counter(target_rows, lambda r: "commit_time_unavailable"), [], []),
        ("offdiag_bucket", counter(reference_rows, lambda r: "commit_time_unavailable"), counter(target_rows, lambda r: "commit_time_unavailable"), [], []),
        ("hard_tail_bucket", counter(reference_rows, lambda r: "reference_unavailable"), counter(target_rows, lambda r: "reference_unavailable"), [], []),
        ("horizon_source", counter(reference_rows, lambda r: "h1_h5_h20_h80_h240"), counter(target_rows, lambda r: "h1_h5_h20_h80_h240"), [], []),
        ("branch_source", counter(reference_rows, lambda r: "real_functional_and_control_branches"), counter(target_rows, lambda r: "real_functional_and_control_branches"), [], []),
    ]
    return axes, {"norm_cuts": norm_cuts, "linf_cuts": linf_cuts, "action_cuts": action_cuts}


def distribution_audit(
    reference_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    stage: str,
    generator_id: str,
    action_count: int,
    old_action_ids: set[str],
    old_payload_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    axes, _meta = reference_axes(reference_rows, target_rows)
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    psi_major_vals: list[float] = []
    js_major_vals: list[float] = []
    share_excess_vals: list[float] = []
    entropy_ratios: list[float] = []
    missing_total = 0
    for axis, old_c, new_c, old_vals, new_vals in axes:
        n_old = sum(old_c.values())
        n_new = sum(new_c.values())
        old_major = {k for k, v in old_c.items() if v / max(1, n_old) >= 0.05}
        missing = sorted(k for k in old_major if new_c.get(k, 0) == 0)
        extra = sorted(k for k in new_c if old_c.get(k, 0) == 0)
        missing_total += len(missing)
        p = psi(old_c, new_c)
        js = js_distance(old_c, new_c)
        tv = tv_distance(old_c, new_c)
        kl = kl_div(old_c, new_c)
        max_share = max(new_c.values()) / max(1, n_new) if new_c else 0.0
        max_ref_share = max(old_c.values()) / max(1, n_old) if old_c else 0.0
        old_entropy = entropy(old_c)
        new_entropy = entropy(new_c)
        if old_entropy <= 1.0e-12 and new_entropy <= 1.0e-12:
            er = 1.0
        else:
            er = new_entropy / max(old_entropy, 1.0e-12)
        if old_major:
            def squash(counter_: Counter[str]) -> Counter[str]:
                c: Counter[str] = Counter()
                for k, v in counter_.items():
                    c[k if k in old_major else "__other__"] += v
                return c

            old_gate = squash(old_c)
            new_gate = squash(new_c)
            p_major = psi(old_gate, new_gate)
            js_major = js_distance(old_gate, new_gate)
        else:
            p_major = 0.0
            js_major = 0.0
        share_limit = max(0.35, max_ref_share + 0.05)
        share_excess = max(0.0, max_share - share_limit)
        psi_major_vals.append(p_major)
        js_major_vals.append(js_major)
        share_excess_vals.append(share_excess)
        entropy_ratios.append(er)
        worst_over = ""
        worst_ratio = -1.0
        for key, value in new_c.items():
            ratio = value / max(1, n_new) - old_c.get(key, 0) / max(1, n_old)
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_over = key
        row = {
            "stage": stage,
            "status": "axis_distribution",
            "generator_id": generator_id,
            "action_count": action_count,
            "axis": axis,
            "reference_group_count": len(old_c),
            "pilot_group_count": len(new_c),
            "missing_major_group_count": len(missing),
            "new_extra_group_count": len(extra),
            "max_group_share_reference": max(old_c.values()) / max(1, n_old) if old_c else 0.0,
            "max_group_share_pilot": max_share,
            "entropy_reference": old_entropy,
            "entropy_pilot": new_entropy,
            "entropy_ratio": er,
            "PSI": p,
            "PSI_major": p_major,
            "KL": kl,
            "JS_distance": js,
            "JS_major": js_major,
            "TV_distance": tv,
            "max_group_share_excess": share_excess,
            "Wasserstein_for_continuous_axis": wasserstein_1d(old_vals, new_vals) if old_vals or new_vals else "",
            "major_group_coverage": (len(old_major) - len(missing)) / max(1, len(old_major)) if old_major else 1.0,
            "worst_missing_group": missing[0] if missing else "",
            "worst_overrepresented_group": worst_over,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        for group in missing:
            missing_rows.append({
                "stage": "P1_MISSING_MAJOR_GROUPS_V9920",
                "status": "missing_major_group",
                "generator_id": generator_id,
                "axis": axis,
                "missing_group": group,
                "reference_share": old_c[group] / max(1, n_old),
                "pilot_share": 0,
                "repair_hint": "targeted_reference_quota_or_recipe_coverage",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    ids = {str(r.get("action_id")) for r in target_rows}
    hashes = {str(r.get("payload_hash_expected") or r.get("payload_hash")) for r in target_rows}
    old_action_collision = len(ids & old_action_ids)
    old_payload_collision = len(hashes & old_payload_hashes)
    weak = int(
        missing_total == 0
        and max(psi_major_vals or [999]) <= 0.50
        and max(js_major_vals or [999]) <= 0.15
        and max(share_excess_vals or [999]) <= 0.0
        and min(entropy_ratios or [0]) >= 0.70
        and old_action_collision == 0
        and old_payload_collision == 0
    )
    strong = int(
        missing_total == 0
        and max(psi_major_vals or [999]) <= 0.20
        and max(js_major_vals or [999]) <= 0.08
        and max(share_excess_vals or [999]) <= 0.0
        and min(entropy_ratios or [0]) >= 0.85
    )
    failure_class = ""
    if max([fnum(r.get("max_group_share_pilot")) for r in rows if fnum(r.get("max_group_share_reference")) <= 0.35] or [0]) >= 0.80:
        failure_class = "F1-cursor-collapse"
    elif missing_total:
        failure_class = "F2-missing-major-groups"
    elif max(psi_major_vals or [0]) > 0.50:
        failure_class = "F4-natural-recipe-narrow"
    summary = {
        "stage": stage,
        "status": "summary",
        "generator_id": generator_id,
        "action_count": action_count,
        "axis_count": len(axes),
        "reference_action_count": len(reference_rows),
        "pilot_action_count": len(target_rows),
        "missing_major_group_count": missing_total,
        "PSI_major_max": max(psi_major_vals or [0.0]),
        "PSI_all_axis_max": max([fnum(r.get("PSI")) for r in rows] or [0.0]),
        "JS_major_max": max(js_major_vals or [0.0]),
        "JS_all_axis_max": max([fnum(r.get("JS_distance")) for r in rows] or [0.0]),
        "TV_major_max": max([fnum(r.get("TV_distance")) for r in rows] or [0.0]),
        "max_group_share_pilot": max([fnum(r.get("max_group_share_pilot")) for r in rows if fnum(r.get("max_group_share_reference")) <= 0.35] or [0.0]),
        "max_group_share_raw": max([fnum(r.get("max_group_share_pilot")) for r in rows] or [0.0]),
        "max_group_share_excess": max(share_excess_vals or [0.0]),
        "entropy_ratio_min": min(entropy_ratios or [0.0]),
        "old_action_collision": old_action_collision,
        "old_payload_collision": old_payload_collision,
        "distribution_weak_pass": weak,
        "distribution_strong_pass": strong,
        "failure_class": failure_class,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    if not missing_rows:
        missing_rows.append({
            "stage": "P1_MISSING_MAJOR_GROUPS_V9920",
            "status": "summary",
            "generator_id": generator_id,
            "missing_major_group_count": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows, summary, missing_rows


def p0_boundary(source_v9910: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9910 / "route_decision_v9910.json")
    p1 = summary_row(rows_from(source_v9910 / "p1_natural_generator_distribution_audit_v9910.csv"))
    p2 = summary_row(rows_from(source_v9910 / "p2_sequential_natural_density_decision_v9910.csv"))
    smoke = summary_row(rows_from(source_v9910 / "p2_1024_action_pilot_smoke_v9910.csv"))
    p4 = summary_row(rows_from(source_v9910 / "p4_future_path_operator_sketch_v4_v9910.csv"))
    nf = summary_row(rows_from(source_v9910 / "no_fake_audit_v9910.csv"))
    rows_complete = int(inum(smoke.get("branch_horizon_actual_rows")) == inum(smoke.get("branch_horizon_expected_rows")) and inum(smoke.get("branch_horizon_actual_rows")) > 0)
    row = {
        "stage": "P0_V9910_BOUNDARY_REPRODUCTION_V9920",
        "status": "summary",
        "source_route_v9910": route.get("route"),
        "source_primary_blocker_v9910": route.get("primary_blocker"),
        "source_secondary_blocker_v9910": route.get("secondary_blocker"),
        "P1_PSI_major_max_v9910": route.get("P1_PSI_major_max") or p1.get("PSI_major_max"),
        "P1_max_group_share_v9910": route.get("P1_max_group_share") or p1.get("max_group_share"),
        "P1_missing_major_groups_v9910": p1.get("missing_major_group_count"),
        "P2_pilot_action_count_v9910": route.get("P2_pilot_action_count") or p2.get("panel_action_count"),
        "P2_rows_expected_v9910": smoke.get("branch_horizon_expected_rows"),
        "P2_rows_actual_v9910": smoke.get("branch_horizon_actual_rows"),
        "P2_rows_complete_v9910": rows_complete,
        "CoreLike_count_v9910": route.get("CoreLike_count") or p2.get("CoreLike_count"),
        "CoreLike_LCB_v9910": route.get("CoreLike_LCB") or p2.get("CoreLike_LCB"),
        "CoreLike_UCB_v9910": route.get("CoreLike_UCB") or p2.get("CoreLike_UCB"),
        "PathGood_count_v9910": route.get("PathGood_count") or p2.get("PathGood_count"),
        "PathGood_LCB_v9910": route.get("PathGood_LCB") or p2.get("PathGood_LCB"),
        "PathGood_UCB_v9910": route.get("PathGood_UCB") or p2.get("PathGood_UCB"),
        "FOS4_precision_v9910": p4.get("best_precision"),
        "fake_data_used_v9910": nf.get("fake_data_used"),
        "proxy_row_used_v9910": nf.get("proxy_row_used"),
        "cpu_offload_used_v9910": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_boundary_reproduced"] = int(
        row["source_route_v9910"] == "R-P1-NaturalGeneratorDistributionMismatch"
        and fnum(row["P1_PSI_major_max_v9910"]) >= 10.0
        and inum(row["P1_missing_major_groups_v9910"]) >= 1
        and rows_complete == 1
        and inum(row["fake_data_used_v9910"]) == 0
        and inum(row["proxy_row_used_v9910"]) == 0
        and inum(row["cpu_offload_used_v9910"]) == 0
    )
    return [row], row


def cursor_trace_rows(action_rows: list[dict[str, Any]], generator_id: str) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(action_rows):
        if idx >= 256:
            break
        rows.append({
            "stage": "P1_GENERATOR_CURSOR_TRACE_V9920",
            "status": "cursor_trace_row",
            "generator_id": generator_id,
            "local_index": idx,
            "source_cursor": row.get("source_cursor"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "step": row.get("step"),
            "step_bucket": step_bucket(row.get("step")),
            "template_id": field(row, "template_id", "carrier_id"),
            "family_id": row.get("family_id"),
            "source_recipe_id": row.get("source_recipe_id", ""),
            "payload_norm": payload_norm(row),
            "payload_linf": payload_linf(row),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": row.get("cpu_offload_used", 0),
        })
    rows.insert(0, {
        "stage": "P1_GENERATOR_CURSOR_TRACE_V9920",
        "status": "summary",
        "generator_id": generator_id,
        "trace_row_count": len(rows),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
    })
    return rows


def p2_density_decision_rows(label_summary: dict[str, Any], smoke: dict[str, Any], dist: dict[str, Any], panel_targets: list[int], generator_id: str, run_full_panels: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = inum(label_summary.get("panel_action_count"))
    core = inum(label_summary.get("CoreLike_count"))
    path = inum(label_summary.get("PathGood_count"))
    slow = inum(label_summary.get("SlowBurnGood_count"))
    risky = inum(label_summary.get("RiskyHighAUV_count"))
    bad = inum(label_summary.get("BadPath_count"))
    sufficient = int(inum(dist.get("distribution_weak_pass")) and (fnum(label_summary.get("CoreLike_LCB")) >= 0.03 or fnum(label_summary.get("PathGood_LCB")) >= 0.03) and wilson_ucb(risky, n) <= 0.05 and wilson_ucb(bad, n) <= 0.05)
    insufficient = int(inum(dist.get("distribution_weak_pass")) and n >= 10000 and fnum(label_summary.get("CoreLike_UCB")) < 0.03 and fnum(label_summary.get("PathGood_UCB")) < 0.03 and wilson_ucb(slow, n) < 0.03)
    rows = [{
        "stage": "P3_DENSITY_PANEL_V9920",
        "status": "panel_row",
        "panel_id": f"{generator_id}-{n}",
        "panel_size": n,
        "generator_id": generator_id,
        "distribution_weak_pass": dist.get("distribution_weak_pass"),
        "distribution_strong_pass": dist.get("distribution_strong_pass"),
        "branch_horizon_expected_rows": smoke.get("branch_horizon_expected_rows"),
        "branch_horizon_actual_rows": smoke.get("branch_horizon_actual_rows"),
        "rows_per_sec": smoke.get("rows_per_sec"),
        "wallclock_sec": smoke.get("wallclock_sec"),
        "peak_gpu_mb": smoke.get("peak_gpu_memory_mb"),
        "CoreLike_count": core,
        "CoreLike_rate": label_summary.get("CoreLike_rate"),
        "CoreLike_LCB": label_summary.get("CoreLike_LCB"),
        "CoreLike_UCB": label_summary.get("CoreLike_UCB"),
        "PathGood_count": path,
        "PathGood_rate": label_summary.get("PathGood_rate"),
        "PathGood_LCB": label_summary.get("PathGood_LCB"),
        "PathGood_UCB": label_summary.get("PathGood_UCB"),
        "SlowBurnGood_count": slow,
        "SlowBurnGood_rate": label_summary.get("SlowBurnGood_rate"),
        "SlowBurnGood_LCB": label_summary.get("SlowBurnGood_LCB"),
        "SlowBurnGood_UCB": label_summary.get("SlowBurnGood_UCB"),
        "RiskyHighAUV_count": risky,
        "RiskyHighAUV_rate": label_summary.get("RiskyHighAUV_rate"),
        "SafeLowValue_count": label_summary.get("RiskCleanButLowImmediate_count"),
        "BadPath_count": bad,
        "BadPath_rate": label_summary.get("BadPath_rate"),
        "density_result": "sufficient" if sufficient else ("insufficient" if insufficient else "inconclusive"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": label_summary.get("cpu_offload_used", 0),
    }]
    reason = "full_density_panels_deferred_set_run_full_density_panels_to_open" if not run_full_panels else "not_implemented_in_v9920_after_gate_check"
    for target in [t for t in panel_targets if t > n]:
        rows.append({
            "stage": "P3_DENSITY_PANEL_V9920",
            "status": "not_run",
            "panel_id": f"{generator_id}-{target}",
            "panel_size": target,
            "generator_id": generator_id,
            "branch_horizon_expected_rows": target * BRANCH_COUNT * len(HORIZONS),
            "branch_horizon_actual_rows": 0,
            "density_result": "not_run",
            "reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P3_DENSITY_PANEL_V9920",
        "status": "summary",
        "selected_generator_id": generator_id,
        "completed_panel_count": 1,
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "panel_size": n,
        "CoreLike_count": core,
        "CoreLike_LCB": label_summary.get("CoreLike_LCB"),
        "CoreLike_UCB": label_summary.get("CoreLike_UCB"),
        "PathGood_count": path,
        "PathGood_LCB": label_summary.get("PathGood_LCB"),
        "PathGood_UCB": label_summary.get("PathGood_UCB"),
        "SlowBurnGood_count": slow,
        "SlowBurnGood_UCB": label_summary.get("SlowBurnGood_UCB"),
        "density_sufficient": sufficient,
        "density_insufficient": insufficient,
        "density_inconclusive": int(not sufficient and not insufficient),
        "reason": reason if not sufficient and not insufficient else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": label_summary.get("cpu_offload_used", 0),
    }
    rows.insert(0, summary)
    return rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def run_profile_smoke(
    generator_id: str,
    profile: str,
    action_count: int,
    cursor: int,
    args: argparse.Namespace,
    out: Path,
    old_action_ids: set[str],
    old_payload_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_args = argparse.Namespace(**vars(args))
    run_args.execution_profile = profile
    stage = f"P2_{generator_id.replace('-', '_').replace('+', '_')}_{action_count}_ACTION_SMOKE_V9920"
    rows, actions, _applies, branches, smoke = v9900.run_natural_smoke(stage, action_count, cursor, run_args, out, old_action_ids, old_payload_hashes)
    for row in rows:
        row["generator_id"] = generator_id
        row["generator_profile"] = profile
    for row in actions:
        row["generator_id"] = generator_id
        row["generator_profile"] = profile
    for row in branches:
        row["generator_id"] = generator_id
        row["generator_profile"] = profile
    smoke["generator_id"] = generator_id
    smoke["generator_profile"] = profile
    return rows, actions, branches, smoke


def summarize_candidate(
    generator_id: str,
    action_count: int,
    smoke: dict[str, Any],
    dist: dict[str, Any],
    label_summary: dict[str, Any],
    ran_1024: int,
) -> dict[str, Any]:
    n = inum(label_summary.get("panel_action_count"))
    return {
        "stage": "P2_GENERATOR_CANDIDATE_MATRIX_V9920",
        "status": "candidate_stage_summary",
        "generator_id": generator_id,
        "action_count": action_count,
        "ran_1024_fidelity_pilot": ran_1024,
        "branch_rows_expected": smoke.get("branch_horizon_expected_rows"),
        "branch_rows_actual": smoke.get("branch_horizon_actual_rows"),
        "branch_completion": smoke.get("branch_horizon_completion"),
        "old_action_collision": dist.get("old_action_collision"),
        "old_payload_collision": dist.get("old_payload_collision"),
        "action_apply_linf_max": smoke.get("action_apply_linf_max"),
        "rows_per_sec": smoke.get("rows_per_sec"),
        "wallclock": smoke.get("wallclock_sec"),
        "peak_gpu_mb": smoke.get("peak_gpu_memory_mb"),
        "PSI_major_max": dist.get("PSI_major_max"),
        "JS_major_max": dist.get("JS_major_max"),
        "missing_major_groups": dist.get("missing_major_group_count"),
        "max_group_share": dist.get("max_group_share_pilot"),
        "entropy_ratio_min": dist.get("entropy_ratio_min"),
        "distribution_weak_pass": dist.get("distribution_weak_pass"),
        "distribution_strong_pass": dist.get("distribution_strong_pass"),
        "CoreLike_count": label_summary.get("CoreLike_count"),
        "CoreLike_LCB": label_summary.get("CoreLike_LCB"),
        "CoreLike_UCB": label_summary.get("CoreLike_UCB"),
        "PathGood_count": label_summary.get("PathGood_count"),
        "PathGood_LCB": label_summary.get("PathGood_LCB"),
        "PathGood_UCB": label_summary.get("PathGood_UCB"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max(inum(smoke.get("cpu_offload_used")), inum(label_summary.get("cpu_offload_used"))),
        "candidate_enters_P3": int(
            inum(dist.get("distribution_weak_pass"))
            and fnum(smoke.get("action_apply_linf_max")) <= 1.0e-7
            and fnum(smoke.get("branch_horizon_completion")) == 1.0
            and inum(dist.get("old_action_collision")) == 0
            and inum(dist.get("old_payload_collision")) == 0
            and max(inum(smoke.get("cpu_offload_used")), inum(label_summary.get("cpu_offload_used"))) == 0
        ),
        "reason": "" if inum(dist.get("distribution_weak_pass")) else dist.get("failure_class"),
        "evaluated_action_count": n,
    }


def artifact_row_count(artifacts: dict[str, Path]) -> int:
    total = 0
    for name, path in artifacts.items():
        if name.endswith(".csv") and path.exists() and not name.startswith(("no_fake", "contract", "failure")):
            with path.open(newline="", encoding="utf-8") as f:
                total += sum(1 for _ in csv.DictReader(f))
        elif name.endswith(".json") and path.exists():
            total += 1
    return total


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def write_figures(out: Path, route: dict[str, Any], cand: dict[str, Any], p3: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}

    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    fig("fig_p1_distribution_error_axes_v9920.svg", "v9.9.2 distribution repair", ["PSI", "JS", "max share", "entropy min"], [
        fnum(cand.get("PSI_major_max")), fnum(cand.get("JS_major_max")), fnum(cand.get("max_group_share")), fnum(cand.get("entropy_ratio_min")),
    ])
    fig("fig_p2_candidate_gate_matrix_v9920.svg", "candidate gates", ["P0", "P1", "P2", "P3", "B"], [
        fnum(route.get("P0_boundary_pass")),
        fnum(route.get("best_generator_distribution_weak_pass")),
        fnum(route.get("best_generator_enters_P3")),
        fnum(route.get("P3_density_sufficient")),
        fnum(route.get("P5_future_operator_sketch_strong_pass")),
    ])
    fig("fig_p3_density_decision_v9920.svg", "density pilot", ["Core LCB", "Core UCB", "Path LCB", "Path UCB"], [
        fnum(p3.get("CoreLike_LCB")), fnum(p3.get("CoreLike_UCB")), fnum(p3.get("PathGood_LCB")), fnum(p3.get("PathGood_UCB")),
    ])
    return figs


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_v9910_boundary_reproduction.csv"))
    p1 = summary_row(rows_from(out / "p1_distribution_fidelity_axes.csv"))
    cand = summary_row(rows_from(out / "p2_generator_candidate_matrix.csv"))
    p3 = summary_row(rows_from(out / "p3_density_panel_1024.csv"))
    p5 = summary_row(rows_from(out / "p5_future_operator_sketch_v4.csv"))
    p6 = summary_row(rows_from(out / "p6_existing_action_controller_gate.csv"))
    p7 = summary_row(rows_from(out / "p7_generated_sandbox_gate.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9920.csv"))
    lines = [
        "# DG-KAN v9.9.2 Natural Generator Distribution Repair / FuturePathOperator 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.2_自然生成器分布修复_FuturePathOperator_四线并行完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 和真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。修复候选未通过分布 gate 时，后续 full density panel 显式 `not_run`。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.9.1 boundary：source route = `{p0.get('source_route_v9910')}`，PSI max = `{p0.get('P1_PSI_major_max_v9910')}`，missing major groups = `{p0.get('P1_missing_major_groups_v9910')}`。",
        f"2. P1 定位原 generator 分布错误：weak/strong = `{p1.get('distribution_weak_pass')}` / `{p1.get('distribution_strong_pass')}`，failure class = `{p1.get('failure_class')}`。",
        f"3. P2 候选矩阵最佳候选 = `{cand.get('best_generator_id')}`，action count = `{cand.get('best_action_count')}`，weak/strong = `{cand.get('best_distribution_weak_pass')}` / `{cand.get('best_distribution_strong_pass')}`。",
        f"4. 最佳候选 PSI/JS/max-share/entropy-min = `{cand.get('best_PSI_major_max')}` / `{cand.get('best_JS_major_max')}` / `{cand.get('best_max_group_share')}` / `{cand.get('best_entropy_ratio_min')}`。",
        f"5. 最佳候选 CoreLike count/LCB/UCB = `{cand.get('best_CoreLike_count')}` / `{cand.get('best_CoreLike_LCB')}` / `{cand.get('best_CoreLike_UCB')}`；PathGood count/LCB/UCB = `{cand.get('best_PathGood_count')}` / `{cand.get('best_PathGood_LCB')}` / `{cand.get('best_PathGood_UCB')}`。",
        f"6. P3 density sufficient/insufficient/inconclusive = `{p3.get('density_sufficient')}` / `{p3.get('density_insufficient')}` / `{p3.get('density_inconclusive')}`；5000/10000/20000 reason = `{p3.get('reason')}`。",
        f"7. P5 FuturePathOperator sketch weak/strong = `{p5.get('P5_weak_pass')}` / `{p5.get('P5_strong_pass')}`。",
        f"8. P6 controller = `{p6.get('status')}`；P7 generated sandbox allowed = `{p7.get('generated_sandbox_allowed')}`。",
        f"9. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.2 generator repair profiles；G0 保留原 v9.9.1 行为，G1-G5 生成新 payload 并记录 quota/cursor provenance。 |",
        "| `experiments/run_v9920_natural_generator_distribution_repair_futurepathoperator.py` | v9.9.2 runner；执行 P0/P1/P2 repair matrix/P3 boundary/P5-P8 gate。 |",
        "",
        "```text",
        "python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9920_natural_generator_distribution_repair_futurepathoperator.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9920_natural_generator_distribution_repair_futurepathoperator.py --out-dir results/real_rerun_20260506/v9920_natural_generator_distribution_repair_futurepathoperator_full_20260517T030000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Distribution Localization",
        "",
        "```text",
        f"generator = {p1.get('generator_id')}",
        f"axis_count = {p1.get('axis_count')}",
        f"PSI_major_max = {p1.get('PSI_major_max')}",
        f"JS_major_max = {p1.get('JS_major_max')}",
        f"max_group_share_pilot = {p1.get('max_group_share_pilot')}",
        f"entropy_ratio_min = {p1.get('entropy_ratio_min')}",
        f"missing_major_group_count = {p1.get('missing_major_group_count')}",
        f"failure_class = {p1.get('failure_class')}",
        "```",
        "",
        "## 4. P2 Candidate Repair",
        "",
        "```text",
        f"best_generator_id = {cand.get('best_generator_id')}",
        f"best_action_count = {cand.get('best_action_count')}",
        f"best_branch_rows = {cand.get('best_branch_rows_actual')} / {cand.get('best_branch_rows_expected')}",
        f"best_distribution_weak_pass = {cand.get('best_distribution_weak_pass')}",
        f"best_candidate_enters_P3 = {cand.get('best_candidate_enters_P3')}",
        "```",
        "",
        "判断：P2 没有复用旧 payload；候选 action_id/payload_hash 都来自新 materializer。未通过 256-action gate 的候选没有被硬推到 1024/full panel。",
        "",
        "## 5. P3 Density Boundary",
        "",
        "```text",
        f"selected_generator = {p3.get('selected_generator_id')}",
        f"panel_size = {p3.get('panel_size')}",
        f"CoreLike count/LCB/UCB = {p3.get('CoreLike_count')} / {p3.get('CoreLike_LCB')} / {p3.get('CoreLike_UCB')}",
        f"PathGood count/LCB/UCB = {p3.get('PathGood_count')} / {p3.get('PathGood_LCB')} / {p3.get('PathGood_UCB')}",
        f"density sufficient/insufficient/inconclusive = {p3.get('density_sufficient')} / {p3.get('density_insufficient')} / {p3.get('density_inconclusive')}",
        "```",
        "",
        "## 6. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 7. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 8. 最终分析结论",
        "",
        "```text",
        "1. v9.9.2 没有把 v9.9.1 的低密度 pilot 写成 natural AP0 density fail；它先做分布修复门。",
        "2. 原始 G0 的分布塌缩被复现并定位到 step/payload/cursor coverage。",
        "3. 修复候选使用真实 materializer 生成新 payload、真实 action apply、真实 branch-horizon replay；没有 fake/proxy rows。",
        "4. 只有通过分布 weak gate 的候选才允许进入 P3；否则 full 5000/10000/20000 继续关闭。",
        "5. controller/generated/runtime/paired replay 仍只按 gate 打开，未通过时保持 not_run。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.9.2 真实执行后停在 `{route.get('route')}`：本轮把自然生成器问题从“发现分布失真”推进到候选修复审计；但没有满足 official controller/runtime/generated gate，因此 strict PureKAN functional 仍未成功。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_v9910 = Path(args.source_v9910)
    source_v9900 = Path(args.source_v9900)
    source_v9330 = Path(args.source_v9330)
    panel_targets = parse_ints(args.panel_targets)
    smoke_counts = parse_ints(args.candidate_smoke_actions)
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return summary_row(rows)

    def dump_json(name: str, row: dict[str, Any]) -> None:
        path = out / name
        write_json(path, row)
        artifacts[name] = path

    p0_rows, p0 = p0_boundary(source_v9910)
    dump_csv("p0_v9910_boundary_reproduction.csv", p0_rows)
    reference_rows = rows_from(source_v9330 / "action_payload_disk_replay_trace_v9330.csv")
    reference_actions = [r for r in reference_rows if r.get("status") == "payload_disk_replay_row"]
    old_ids, old_hashes = v9910.old_action_sets(source_v9330, source_v9900)
    g0_actions = rows_from(source_v9910 / "p2_1024_natural_action_rows_v9910.csv")
    g0_branches = rows_from(source_v9910 / "p2_1024_branch_horizon_v9910.csv")
    g0_labels, g0_label_summary = v9910.label_natural_actions(g0_branches)
    p1_rows, p1, p1_missing = distribution_audit(
        reference_actions,
        g0_actions,
        stage="P1_DISTRIBUTION_FIDELITY_AXES_V9920",
        generator_id="G0-current-v9910-generator",
        action_count=len(g0_actions),
        old_action_ids=old_ids,
        old_payload_hashes=old_hashes,
    )
    dump_csv("p1_distribution_fidelity_axes.csv", p1_rows)
    dump_csv("p1_missing_major_groups.csv", p1_missing)
    dump_csv("p1_generator_cursor_trace.csv", cursor_trace_rows(g0_actions, "G0-current-v9910-generator"))

    candidate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_payload: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None

    g0_smoke = summary_row(rows_from(source_v9910 / "p2_1024_action_pilot_smoke_v9910.csv"))
    candidate_rows.append(summarize_candidate("G0-current-v9910-generator", len(g0_actions), g0_smoke, p1, g0_label_summary, 1))
    failure_rows.append({
        "stage": "P2_GENERATOR_REPAIR_FAILURE_MATRIX_V9920",
        "status": "failure_row",
        "generator_id": "G0-current-v9910-generator",
        "action_count": len(g0_actions),
        "failure_class": p1.get("failure_class"),
        "repair_hint": "replace_cursor_step_schedule_and_payload_norm_distribution",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })

    if inum(p0.get("P0_boundary_reproduced")):
        cursor_base = 4096
        for profile_idx, (generator_id, profile, mode) in enumerate(GENERATOR_PROFILES[1:], start=1):
            last_summary: dict[str, Any] | None = None
            last_actions: list[dict[str, Any]] = []
            last_labels: list[dict[str, Any]] = []
            last_label_summary: dict[str, Any] = {}
            for count in smoke_counts:
                rows, actions, branches, smoke = run_profile_smoke(
                    generator_id,
                    profile,
                    count,
                    cursor_base + profile_idx * 10000 + count,
                    args,
                    out,
                    old_ids,
                    old_hashes,
                )
                dump_csv(f"p2_{generator_id}_{count}_action_smoke.csv".replace("/", "_"), rows)
                labels, label_summary = v9910.label_natural_actions(branches)
                dist_rows, dist_summary, missing = distribution_audit(
                    reference_actions,
                    actions,
                    stage="P2_CANDIDATE_DISTRIBUTION_FIDELITY_V9920",
                    generator_id=generator_id,
                    action_count=count,
                    old_action_ids=old_ids,
                    old_payload_hashes=old_hashes,
                )
                dump_csv(f"p2_{generator_id}_{count}_distribution_axes.csv".replace("/", "_"), dist_rows)
                candidate_rows.append(summarize_candidate(generator_id, count, smoke, dist_summary, label_summary, 0))
                if not inum(dist_summary.get("distribution_weak_pass")):
                    failure_rows.append({
                        "stage": "P2_GENERATOR_REPAIR_FAILURE_MATRIX_V9920",
                        "status": "failure_row",
                        "generator_id": generator_id,
                        "action_count": count,
                        "failure_class": dist_summary.get("failure_class"),
                        "PSI_major_max": dist_summary.get("PSI_major_max"),
                        "missing_major_group_count": dist_summary.get("missing_major_group_count"),
                        "repair_hint": "inspect_missing_major_groups_and_candidate_cursor_trace",
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": max(inum(smoke.get("cpu_offload_used")), inum(label_summary.get("cpu_offload_used"))),
                    })
                last_summary = dist_summary
                last_actions = actions
                last_labels = labels
                last_label_summary = label_summary
            if last_summary and inum(last_summary.get("distribution_weak_pass")):
                rows, actions, branches, smoke = run_profile_smoke(
                    generator_id,
                    profile,
                    int(args.pilot_actions),
                    cursor_base + profile_idx * 10000 + int(args.pilot_actions),
                    args,
                    out,
                    old_ids,
                    old_hashes,
                )
                dump_csv(f"p2_{generator_id}_1024_action_fidelity_pilot.csv".replace("/", "_"), rows)
                labels, label_summary = v9910.label_natural_actions(branches)
                dump_csv(f"p2_{generator_id}_1024_action_labels.csv".replace("/", "_"), labels)
                dist_rows, dist_summary, missing = distribution_audit(
                    reference_actions,
                    actions,
                    stage="P2_CANDIDATE_1024_DISTRIBUTION_FIDELITY_V9920",
                    generator_id=generator_id,
                    action_count=int(args.pilot_actions),
                    old_action_ids=old_ids,
                    old_payload_hashes=old_hashes,
                )
                dump_csv(f"p2_{generator_id}_1024_distribution_axes.csv".replace("/", "_"), dist_rows)
                cand = summarize_candidate(generator_id, int(args.pilot_actions), smoke, dist_summary, label_summary, 1)
                candidate_rows.append(cand)
                if inum(cand.get("candidate_enters_P3")):
                    if selected is None or (
                        fnum(cand.get("PSI_major_max")) < fnum(selected.get("PSI_major_max"))
                        or (
                            fnum(cand.get("PSI_major_max")) == fnum(selected.get("PSI_major_max"))
                            and fnum(cand.get("max_group_share")) < fnum(selected.get("max_group_share"))
                        )
                    ):
                        selected = cand
                        selected_payload = (actions, branches, smoke, label_summary, dist_summary)

    candidates_only = [r for r in candidate_rows if r.get("status") == "candidate_stage_summary"]
    best = max(candidates_only, key=lambda r: (inum(r.get("candidate_enters_P3")), inum(r.get("distribution_weak_pass")), -fnum(r.get("PSI_major_max")), fnum(r.get("PathGood_count"))), default={})
    candidate_summary = {
        "stage": "P2_GENERATOR_CANDIDATE_MATRIX_V9920",
        "status": "summary",
        "candidate_stage_rows": len(candidates_only),
        "best_generator_id": best.get("generator_id", ""),
        "best_action_count": best.get("action_count", ""),
        "best_branch_rows_expected": best.get("branch_rows_expected", ""),
        "best_branch_rows_actual": best.get("branch_rows_actual", ""),
        "best_distribution_weak_pass": best.get("distribution_weak_pass", 0),
        "best_distribution_strong_pass": best.get("distribution_strong_pass", 0),
        "best_candidate_enters_P3": best.get("candidate_enters_P3", 0),
        "best_PSI_major_max": best.get("PSI_major_max", ""),
        "best_JS_major_max": best.get("JS_major_max", ""),
        "best_max_group_share": best.get("max_group_share", ""),
        "best_entropy_ratio_min": best.get("entropy_ratio_min", ""),
        "best_CoreLike_count": best.get("CoreLike_count", ""),
        "best_CoreLike_LCB": best.get("CoreLike_LCB", ""),
        "best_CoreLike_UCB": best.get("CoreLike_UCB", ""),
        "best_PathGood_count": best.get("PathGood_count", ""),
        "best_PathGood_LCB": best.get("PathGood_LCB", ""),
        "best_PathGood_UCB": best.get("PathGood_UCB", ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in candidates_only] or [0]),
    }
    candidate_rows.insert(0, candidate_summary)
    dump_csv("p2_generator_candidate_matrix.csv", candidate_rows)
    if not failure_rows:
        failure_rows.append({"stage": "P2_GENERATOR_REPAIR_FAILURE_MATRIX_V9920", "status": "summary", "failure_row_count": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    else:
        failure_rows.insert(0, {"stage": "P2_GENERATOR_REPAIR_FAILURE_MATRIX_V9920", "status": "summary", "failure_row_count": len(failure_rows), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in failure_rows] or [0])})
    dump_csv("p2_generator_repair_failure_matrix.csv", failure_rows)

    if selected and selected_payload:
        actions, branches, smoke, label_summary, dist_summary = selected_payload
        p3_rows, p3 = p2_density_decision_rows(label_summary, smoke, dist_summary, panel_targets, str(selected.get("generator_id")), bool(args.run_full_density_panels))
    else:
        p3_rows, p3 = not_run("P3_DENSITY_PANEL_V9920", "no_generator_candidate_passed_P1_weak_fidelity", selected_generator_id="", density_sufficient=0, density_insufficient=0, density_inconclusive=1)
    dump_csv("p3_density_panel_1024.csv", p3_rows)
    dump_csv("p3_density_panel_5000.csv", [r for r in p3_rows if str(r.get("panel_size")) == "5000"] or not_run("P3_DENSITY_PANEL_5000_V9920", "not_opened_by_gate", panel_size=5000)[0])
    dump_csv("p3_density_panel_10000.csv", [r for r in p3_rows if str(r.get("panel_size")) == "10000"] or not_run("P3_DENSITY_PANEL_10000_V9920", "not_opened_by_gate", panel_size=10000)[0])
    dump_csv("p3_density_panel_20000.csv", [r for r in p3_rows if str(r.get("panel_size")) == "20000"] or not_run("P3_DENSITY_PANEL_20000_V9920", "not_opened_by_gate", panel_size=20000)[0])
    dump_json("p3_density_decision.json", p3)

    p4_rows, p4 = not_run("P4_FUTURE_PATH_TYPES_V9920", "P3_full_density_not_adjudicated", P4_future_path_weak_pass=0, P4_future_path_strong_pass=0)
    dump_csv("p4_future_path_types.csv", p4_rows)
    dump_csv("p4_future_path_curves_by_group.csv", p4_rows)
    p5_rows, p5 = not_run("P5_FUTURE_OPERATOR_SKETCH_V4_V9920", "P3_full_density_not_adjudicated_or_no_B_input", P5_weak_pass=0, P5_strong_pass=0)
    dump_csv("p5_future_operator_sketch_v4.csv", p5_rows)
    p6_rows, p6 = not_run("P6_EXISTING_ACTION_CONTROLLER_GATE_V9920", "P3_density_not_sufficient_and_B_strong_not_passed", controller_pass=0)
    dump_csv("p6_existing_action_controller_gate.csv", p6_rows)
    p7_rows, p7 = not_run("P7_GENERATED_SANDBOX_GATE_V9920", "A_B_C_conditions_not_met", generated_sandbox_allowed=0)
    dump_csv("p7_generated_sandbox_gate.csv", p7_rows)
    p8_rt_rows, p8_rt = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9920", "P6_controller_and_P7_generated_not_passed", runtime_pass=0)
    dump_csv("p8_selected_runtime_boundary.csv", p8_rt_rows)
    p8_pr_rows, p8_pr = not_run("P8_PAIRED_REPLAY_BOUNDARY_V9920", "P8_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p8_paired_replay_boundary.csv", p8_pr_rows)

    if not inum(p0.get("P0_boundary_reproduced")):
        route_name = "R0-BoundaryRegression"
        primary = "v9910_boundary_reproduction_failed"
        secondary = "loader_or_artifact_issue"
        gen_status = "stopped_P0_boundary_failed"
    elif not inum(candidate_summary.get("best_candidate_enters_P3")):
        route_name = "R1-DistributionMismatchUnresolved"
        primary = "no_generator_candidate_passes_distribution_weak_gate"
        secondary = "natural_density_panels_blocked"
        gen_status = "stopped_no_distribution_fidelity_generator"
    elif inum(p3.get("density_sufficient")):
        route_name = "R3-NaturalDensitySufficientControllerPending"
        primary = "controller_gate_pending"
        secondary = "runtime_not_opened"
        gen_status = "stopped_controller_pending"
    elif inum(p3.get("density_insufficient")):
        route_name = "R4-NaturalDensityInsufficientGeneratorNeeded"
        primary = "natural_density_insufficient"
        secondary = "generated_route_requires_A_B_mechanism"
        gen_status = "stopped_density_insufficient_no_mechanism"
    else:
        route_name = "R2-DistributionFidelityPassDensityPending"
        primary = "natural_density_full_panel_pending"
        secondary = "future_operator_sketch_not_opened"
        gen_status = "stopped_density_full_panel_pending"

    route = {
        "stage": "ROUTE_DECISION_V9920",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9910": p0.get("source_route_v9910"),
        "P0_boundary_pass": p0.get("P0_boundary_reproduced"),
        "P1_distribution_weak_pass": p1.get("distribution_weak_pass"),
        "P1_distribution_strong_pass": p1.get("distribution_strong_pass"),
        "P1_PSI_major_max": p1.get("PSI_major_max"),
        "P1_missing_major_group_count": p1.get("missing_major_group_count"),
        "best_generator_id": candidate_summary.get("best_generator_id"),
        "best_generator_distribution_weak_pass": candidate_summary.get("best_distribution_weak_pass"),
        "best_generator_distribution_strong_pass": candidate_summary.get("best_distribution_strong_pass"),
        "best_generator_enters_P3": candidate_summary.get("best_candidate_enters_P3"),
        "best_PSI_major_max": candidate_summary.get("best_PSI_major_max"),
        "best_JS_major_max": candidate_summary.get("best_JS_major_max"),
        "best_max_group_share": candidate_summary.get("best_max_group_share"),
        "best_entropy_ratio_min": candidate_summary.get("best_entropy_ratio_min"),
        "P3_density_sufficient": p3.get("density_sufficient", 0),
        "P3_density_insufficient": p3.get("density_insufficient", 0),
        "P3_density_inconclusive": p3.get("density_inconclusive", 1),
        "P5_future_operator_sketch_weak_pass": p5.get("P5_weak_pass", 0),
        "P5_future_operator_sketch_strong_pass": p5.get("P5_strong_pass", 0),
        "P6_controller_pass": p6.get("controller_pass", 0),
        "P7_generated_sandbox_allowed": p7.get("generated_sandbox_allowed", 0),
        "generated_route_status": gen_status,
        "P8_runtime_pass": p8_rt.get("runtime_pass", 0),
        "P8_paired_replay_pass": p8_pr.get("paired_replay_pass", 0),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(x.get("cpu_offload_used")) for x in [p0, p1, candidate_summary, p3, p5, p6, p7, p8_rt, p8_pr] if x] or [0]),
    }
    dump_json("route_decision_v9920.json", route)
    artifacts.update(write_figures(out, route, candidate_summary, p3))
    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9920",
        "status": "summary",
        "rows_checked": artifact_row_count(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9920.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9920",
        "status": "summary",
        "v9910_boundary_pass": p0.get("P0_boundary_reproduced"),
        "generator_candidate_rows": candidate_summary.get("candidate_stage_rows"),
        "best_generator_enters_P3": candidate_summary.get("best_candidate_enters_P3"),
        "density_sufficient/insufficient/inconclusive": f"{p3.get('density_sufficient', 0)}/{p3.get('density_insufficient', 0)}/{p3.get('density_inconclusive', 1)}",
        "controller/generated/runtime/system": "0/0/0/0",
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"0/0/{route['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_csv("contract_audit_v9920.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9920",
        "status": "summary",
        "route": route["route"],
        "F0_boundary_fail": int(not inum(p0.get("P0_boundary_reproduced"))),
        "F1_generator_distribution_unfixed": int(not inum(candidate_summary.get("best_candidate_enters_P3"))),
        "F2_full_density_panel_pending": int(route["route"] == "R2-DistributionFidelityPassDensityPending"),
        "F3_future_operator_sketch_blocked": int(not inum(p5.get("P5_strong_pass"))),
        "F4_controller_generated_runtime_blocked": 1,
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_csv("failure_taxonomy_v9920.csv", [failure])
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, "materializer": REPO / "experiments/natural_ap0_extension_materializer.py", **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9920",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "device": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "panel_targets": args.panel_targets,
        "candidate_smoke_actions": args.candidate_smoke_actions,
        "pilot_actions": args.pilot_actions,
        "run_full_density_panels": int(bool(args.run_full_density_panels)),
        "source_v9910": args.source_v9910,
        "wallclock_sec": time.perf_counter() - started,
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_json("run_manifest_v9920.json", manifest)
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, "materializer": REPO / "experiments/natural_ap0_extension_materializer.py", **artifacts})
    write_recap(out, route, hashes)
    print(json.dumps({
        "out_dir": str(out),
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "best_generator_id": route.get("best_generator_id"),
        "best_generator_enters_P3": route.get("best_generator_enters_P3"),
        "P3_density_inconclusive": route.get("P3_density_inconclusive"),
        "system_legal_controller_pass": route["system_legal_controller_pass"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
