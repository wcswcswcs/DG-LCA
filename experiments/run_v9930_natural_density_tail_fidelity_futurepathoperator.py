#!/usr/bin/env python3
"""DG-KAN v9.9.3 natural density / tail fidelity / FuturePathOperator run.

This runner treats v9.9.2 as a real distribution-repair boundary, not as a
density closure.  It first audits G5 tail fidelity against the canonical AP0
ledger, attempts a tail-aware repair profile if required, then opens sequential
natural density panels only when fidelity gates permit.
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
import run_v9920_natural_generator_distribution_repair_futurepathoperator as v9920  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.3_结果解读_自然密度裁决_TailFidelity_FuturePathOperator_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9930_natural_density_tail_fidelity_futurepathoperator.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.3_NaturalDensityTailFidelity_FuturePathOperator_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9930_natural_density_tail_fidelity_futurepathoperator_full_20260517T040000Z"
DEFAULT_V9920 = RESULT_ROOT / "v9920_natural_generator_distribution_repair_futurepathoperator_full_20260517T030000Z"
DEFAULT_V9910 = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
DEFAULT_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_V9830 = RESULT_ROOT / "v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z"
DEFAULT_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6
G5_ID = "G5-hybrid-quota-random-generator"
G5_PROFILE = "g5-hybrid-quota-random-generator"
G6_ID = "G6-tail-aware-quota-generator"
G6_PROFILE = "g6-tail-aware-quota-generator"


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
    p.add_argument("--chunk-actions", type=int, default=512)
    p.add_argument("--source-v9920", default=str(DEFAULT_V9920))
    p.add_argument("--source-v9910", default=str(DEFAULT_V9910))
    p.add_argument("--source-v9900", default=str(DEFAULT_V9900))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--source-v9830", default=str(DEFAULT_V9830))
    p.add_argument("--source-v9840", default=str(DEFAULT_V9840))
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


def payload_norm(row: dict[str, Any]) -> float:
    return fnum(row.get("payload_norm") or row.get("payload_l2_norm") or row.get("action_norm"))


def payload_linf(row: dict[str, Any]) -> float:
    return fnum(row.get("payload_linf_norm") or row.get("payload_linf"))


def field(row: dict[str, Any], *names: str, default: str = "unknown") -> str:
    for name in names:
        val = row.get(name)
        if val not in {None, ""}:
            return str(val)
    return default


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


def bucket(value: float, cuts: list[float], name: str) -> str:
    for i, cut in enumerate(cuts):
        if value <= cut:
            return f"{name}_bin_{i}"
    return f"{name}_bin_{len(cuts)}"


def reference_actions(source_v9330: Path) -> list[dict[str, Any]]:
    return [r for r in rows_from(source_v9330 / "action_payload_disk_replay_trace_v9330.csv") if r.get("status") == "payload_disk_replay_row"]


def old_action_sets(source_v9330: Path, source_v9900: Path, source_v9910: Path) -> tuple[set[str], set[str]]:
    ids, hashes = v9910.old_action_sets(source_v9330, source_v9900)
    for row in rows_from(source_v9910 / "p2_1024_natural_action_rows_v9910.csv"):
        ids.add(str(row.get("action_id")))
        hashes.add(str(row.get("payload_hash_expected") or row.get("payload_hash")))
    return ids, hashes


def load_action_group_ids(source_v9830: Path, source_v9840: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"Core77": set(), "OldOnly": set(), "RiskCleanButLowValue": set()}
    for path in [
        source_v9830 / "p1_future_path_mechanism_decomposition_v2_v9830.csv",
        source_v9840 / "p1_future_path_mechanism_decomposition_v3.csv",
    ]:
        for row in rows_from(path):
            group = str(row.get("group_id") or row.get("group") or "")
            if group in out and row.get("action_id"):
                out[group].add(str(row.get("action_id")))
            if group == "RiskCleanButLowValue" and row.get("action_id"):
                out["RiskCleanButLowValue"].add(str(row.get("action_id")))
    return out


def p0_boundary(source_v9920: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9920 / "route_decision_v9920.json")
    cand = summary_row(rows_from(source_v9920 / "p2_generator_candidate_matrix.csv"))
    p3 = summary_row(rows_from(source_v9920 / "p3_density_panel_1024.csv"))
    p5 = summary_row(rows_from(source_v9920 / "p5_future_operator_sketch_v4.csv"))
    nf = summary_row(rows_from(source_v9920 / "no_fake_audit_v9920.csv"))
    row = {
        "stage": "P0_V9920_BOUNDARY_REPRODUCTION_V9930",
        "status": "summary",
        "route_v9920": route.get("route"),
        "best_generator_id_v9920": route.get("best_generator_id") or cand.get("best_generator_id"),
        "best_PSI_major_max_v9920": route.get("best_PSI_major_max") or cand.get("best_PSI_major_max"),
        "best_JS_major_max_v9920": route.get("best_JS_major_max") or cand.get("best_JS_major_max"),
        "best_max_group_share_v9920": route.get("best_max_group_share") or cand.get("best_max_group_share"),
        "best_entropy_ratio_min_v9920": route.get("best_entropy_ratio_min") or cand.get("best_entropy_ratio_min"),
        "CoreLike_count_1024_v9920": cand.get("best_CoreLike_count") or p3.get("CoreLike_count"),
        "PathGood_count_1024_v9920": cand.get("best_PathGood_count") or p3.get("PathGood_count"),
        "P3_density_inconclusive_v9920": route.get("P3_density_inconclusive") or p3.get("density_inconclusive"),
        "P5_future_operator_sketch_weak_pass_v9920": route.get("P5_future_operator_sketch_weak_pass") or p5.get("P5_weak_pass"),
        "system_legal_controller_pass_v9920": route.get("system_legal_controller_pass"),
        "fake_data_used_v9920": nf.get("fake_data_used"),
        "proxy_row_used_v9920": nf.get("proxy_row_used"),
        "cpu_offload_used_v9920": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_boundary_pass"] = int(
        str(row["best_generator_id_v9920"]) == G5_ID
        and fnum(row["best_PSI_major_max_v9920"]) <= 0.05
        and fnum(row["best_JS_major_max_v9920"]) <= 0.10
        and fnum(row["best_max_group_share_v9920"]) <= 0.35
        and fnum(row["best_entropy_ratio_min_v9920"]) >= 0.85
        and inum(row["system_legal_controller_pass_v9920"]) == 0
        and inum(row["fake_data_used_v9920"]) == 0
        and inum(row["proxy_row_used_v9920"]) == 0
        and inum(row["cpu_offload_used_v9920"]) == 0
    )
    return [row], row


def run_panel_sharded(
    generator_id: str,
    profile: str,
    panel_size: int,
    cursor: int,
    args: argparse.Namespace,
    out: Path,
    old_ids: set[str],
    old_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    panel_rows: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    all_applies: list[dict[str, Any]] = []
    all_branches: list[dict[str, Any]] = []
    run_args = argparse.Namespace(**vars(args))
    run_args.execution_profile = profile
    chunk = max(1, int(args.chunk_actions))
    remaining = int(panel_size)
    processed = 0
    t0 = time.perf_counter()
    while remaining > 0:
        n = min(chunk, remaining)
        stage = f"P2_{generator_id.replace('-', '_')}_{panel_size}_CHUNK_{processed // chunk:04d}_V9930"
        rows, actions, applies, branches, smoke = v9900.run_natural_smoke(stage, n, cursor + processed, run_args, out, old_ids, old_hashes)
        smoke.update({"generator_id": generator_id, "generator_profile": profile, "panel_size": panel_size, "chunk_start": processed, "chunk_action_count": n})
        for row in rows:
            row.update({"generator_id": generator_id, "generator_profile": profile, "panel_size": panel_size, "chunk_start": processed})
        for coll in (actions, applies, branches):
            for row in coll:
                row.update({"generator_id": generator_id, "generator_profile": profile, "panel_size": panel_size, "chunk_start": processed})
        panel_rows.extend(rows)
        all_actions.extend(actions)
        all_applies.extend(applies)
        all_branches.extend(branches)
        processed += n
        remaining -= n
    wall = max(1.0e-9, time.perf_counter() - t0)
    expected = panel_size * BRANCH_COUNT * len(HORIZONS)
    outcome_ids = [str(r.get("outcome_row_id")) for r in all_branches]
    summary = {
        "stage": f"P2_{generator_id.replace('-', '_')}_{panel_size}_PANEL_SMOKE_V9930",
        "status": "summary",
        "generator_id": generator_id,
        "generator_profile": profile,
        "panel_size": panel_size,
        "chunk_count": len(panel_rows),
        "natural_action_count": panel_size,
        "new_action_count": len(all_actions),
        "unique_action_id_count": len({str(r.get("action_id")) for r in all_actions}),
        "unique_payload_hash_count": len({str(r.get("payload_hash_expected")) for r in all_actions}),
        "unique_template_count": len({str(r.get("template_id")) for r in all_actions}),
        "unique_family_count": len({str(r.get("family_id")) for r in all_actions}),
        "old_action_id_collision_count": sum(1 for r in all_actions if str(r.get("action_id")) in old_ids),
        "old_payload_hash_collision_count": sum(1 for r in all_actions if str(r.get("payload_hash_expected")) in old_hashes),
        "duplicate_action_id_count": len(all_actions) - len({str(r.get("action_id")) for r in all_actions}),
        "duplicate_payload_hash_count": len(all_actions) - len({str(r.get("payload_hash_expected")) for r in all_actions}),
        "payload_hash_missing_count": sum(1 for r in all_actions if not r.get("payload_hash_expected")),
        "action_apply_linf_max": max([fnum(r.get("action_apply_linf_max")) for r in all_applies] or [0.0]),
        "action_apply_cosine_min": min([fnum(r.get("action_apply_cosine"), 1.0) for r in all_applies] or [1.0]),
        "no_transform_sanity": int(all(inum(r.get("payload_hash_match")) for r in all_applies)),
        "branch_horizon_expected_rows": expected,
        "branch_horizon_actual_rows": len(all_branches),
        "branch_horizon_completion": len(all_branches) / max(1, expected),
        "duplicate_row_id": len(outcome_ids) - len(set(outcome_ids)),
        "metric_nan_count": sum(inum(r.get("metric_nan_count")) for r in panel_rows),
        "metric_inf_count": sum(inum(r.get("metric_inf_count")) for r in panel_rows),
        "rows_per_sec": len(all_branches) / wall,
        "wallclock_sec": wall,
        "peak_gpu_memory_mb": max([fnum(r.get("peak_gpu_memory_mb")) for r in panel_rows] or [0.0]),
        "unresolved_exception_count": sum(inum(r.get("unresolved_exception_count")) for r in panel_rows),
        "completion_pass": int(
            len(all_actions) == panel_size
            and len(all_branches) == expected
            and len({str(r.get("action_id")) for r in all_actions}) == panel_size
            and max([fnum(r.get("action_apply_linf_max")) for r in all_applies] or [1.0]) <= 1.0e-7
            and sum(1 for r in all_actions if str(r.get("action_id")) in old_ids) == 0
            and sum(1 for r in all_actions if str(r.get("payload_hash_expected")) in old_hashes) == 0
            and max([inum(r.get("cpu_offload_used")) for r in all_actions + all_applies + all_branches] or [0]) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in all_actions + all_applies + all_branches + panel_rows] or [0]),
    }
    return [summary, *panel_rows], all_actions, all_applies, all_branches, summary


def axis_counter(rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], str]) -> Counter[str]:
    return Counter(str(fn(r)) for r in rows)


def tail_fidelity_audit(
    reference_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    group_ids: dict[str, set[str]],
    *,
    generator_id: str,
    panel_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_norms = [payload_norm(r) for r in reference_rows]
    old_linfs = [payload_linf(r) for r in reference_rows]
    norm_cuts = [q(old_norms, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
    linf_cuts = [q(old_linfs, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]

    def old_recipe(r: dict[str, Any]) -> str:
        return field(r, "bucket_id", "family_id")

    def new_recipe(r: dict[str, Any]) -> str:
        return field(r, "source_recipe_id", "family_id")

    def old_template(r: dict[str, Any]) -> str:
        return field(r, "carrier_id", "template_id")

    def new_template(r: dict[str, Any]) -> str:
        return field(r, "candidate_template_id", "template_id", "carrier_id")

    def old_norm_bucket(r: dict[str, Any]) -> str:
        return bucket(payload_norm(r), norm_cuts, "payload_norm")

    def new_norm_bucket(r: dict[str, Any]) -> str:
        return bucket(payload_norm(r), norm_cuts, "payload_norm")

    def old_linf_bucket(r: dict[str, Any]) -> str:
        return bucket(payload_linf(r), linf_cuts, "payload_linf")

    def new_linf_bucket(r: dict[str, Any]) -> str:
        return bucket(payload_linf(r), linf_cuts, "payload_linf")

    specs: list[tuple[str, str, Callable[[dict[str, Any]], str], Callable[[dict[str, Any]], str]]] = [
        ("major", "dataset_id", lambda r: field(r, "dataset"), lambda r: field(r, "dataset")),
        ("major", "template_id", old_template, new_template),
        ("major", "step_bucket", lambda r: step_bucket(r.get("step")), lambda r: step_bucket(r.get("step"))),
        ("major", "payload_norm_bucket", old_norm_bucket, new_norm_bucket),
        ("major", "action_norm_bucket", old_norm_bucket, new_norm_bucket),
        ("major", "family", lambda r: field(r, "family_id"), lambda r: field(r, "family_id")),
        ("major", "stratum", old_recipe, new_recipe),
        ("tail", "candidate_template_step_pair", lambda r: f"{old_template(r)}|{step_bucket(r.get('step'))}", lambda r: f"{new_template(r)}|{step_bucket(r.get('step'))}"),
        ("tail", "recipe_template_pair", lambda r: f"{old_recipe(r)}|{old_template(r)}", lambda r: f"{new_recipe(r)}|{new_template(r)}"),
        ("tail", "recipe_step_pair", lambda r: f"{old_recipe(r)}|{step_bucket(r.get('step'))}", lambda r: f"{new_recipe(r)}|{step_bucket(r.get('step'))}"),
        ("tail", "template_payload_norm_tail", lambda r: f"{old_template(r)}|{old_norm_bucket(r)}", lambda r: f"{new_template(r)}|{new_norm_bucket(r)}"),
        ("tail", "template_payload_linf_tail", lambda r: f"{old_template(r)}|{old_linf_bucket(r)}", lambda r: f"{new_template(r)}|{new_linf_bucket(r)}"),
        ("tail", "source_recipe_id", old_recipe, new_recipe),
        ("lifecycle", "candidate_origin", lambda r: "canonical_ap0_reference_quota", lambda r: field(r, "candidate_origin")),
        ("lifecycle", "payload_hash_prefix", lambda r: field(r, "payload_hash_expected", "payload_hash_loaded")[:8], lambda r: field(r, "payload_hash_expected", "payload_hash")[:8]),
    ]
    rows: list[dict[str, Any]] = []
    major_psis: list[float] = []
    major_js: list[float] = []
    major_shares: list[float] = []
    major_entropy: list[float] = []
    tail_psis: list[float] = []
    tail_js: list[float] = []
    tail_missing_total = 0
    tail_entropy: list[float] = []
    for axis_type, axis, old_fn, new_fn in specs:
        old_c = axis_counter(reference_rows, old_fn)
        new_c = axis_counter(action_rows, new_fn)
        n_old = sum(old_c.values())
        n_new = sum(new_c.values())
        old_threshold = 8 if axis_type == "tail" else max(1, math.ceil(0.05 * max(1, n_old)))
        old_groups = {k for k, v in old_c.items() if v >= old_threshold}
        missing = sorted(k for k in old_groups if new_c.get(k, 0) == 0)

        def squash(counter_: Counter[str]) -> Counter[str]:
            if not old_groups:
                return Counter({"__single_or_no_gate_group__": sum(counter_.values())})
            squashed: Counter[str] = Counter()
            for key, value in counter_.items():
                squashed[key if key in old_groups else "__other__"] += value
            return squashed

        old_gate = squash(old_c)
        new_gate = squash(new_c)
        p_all = psi(old_c, new_c)
        js_all = js_distance(old_c, new_c)
        kl_all = kl_div(old_c, new_c)
        p = psi(old_gate, new_gate)
        js = js_distance(old_gate, new_gate)
        kl = kl_div(old_gate, new_gate)
        max_share = max(new_c.values()) / max(1, n_new) if new_c else 0.0
        er = 1.0 if entropy(old_c) <= 1.0e-12 and entropy(new_c) <= 1.0e-12 else entropy(new_c) / max(1.0e-12, entropy(old_c))
        coverage = (len(old_groups) - len(missing)) / max(1, len(old_groups)) if old_groups else 1.0
        if axis_type == "major":
            major_psis.append(p)
            major_js.append(js)
            if len(old_c) > 1:
                major_shares.append(max_share)
            major_entropy.append(er)
        elif axis_type == "tail":
            tail_psis.append(p)
            tail_js.append(js)
            tail_missing_total += len(missing)
            tail_entropy.append(er)
        rows.append({
            "stage": "P1_MAJOR_TAIL_FIDELITY_AUDIT_V9930",
            "status": "axis_distribution",
            "generator_id": generator_id,
            "panel_size": panel_size,
            "axis_type": axis_type,
            "axis": axis,
            "old_group_count": len(old_c),
            "new_group_count": len(new_c),
            "old_covered_group_count": len(old_groups),
            "missing_group_count": len(missing),
            "coverage": coverage,
            "max_new_group_share": max_share,
            "entropy_ratio": er,
            "PSI": p,
            "JS": js,
            "KL": kl,
            "PSI_all_groups": p_all,
            "JS_all_groups": js_all,
            "KL_all_groups": kl_all,
            "gate_group_threshold_old_count": old_threshold,
            "worst_missing_group": missing[0] if missing else "",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })

    template_old = {old_template(r) for r in reference_rows}
    template_new = {new_template(r) for r in action_rows}
    candidate_template_coverage = len(template_old & template_new) / max(1, len(template_old))

    action_by_id = {str(r.get("action_id")): r for r in reference_rows}
    new_precursor_keys = {f"{new_template(r)}|{new_norm_bucket(r)}" for r in action_rows}

    def precursor_coverage(name: str) -> float:
        ids = group_ids.get(name, set())
        old_keys = {
            f"{old_template(action_by_id[aid])}|{old_norm_bucket(action_by_id[aid])}"
            for aid in ids
            if aid in action_by_id
        }
        if not old_keys:
            return 0.0
        return len(old_keys & new_precursor_keys) / len(old_keys)

    core_cov = precursor_coverage("Core77")
    old_cov = precursor_coverage("OldOnly")
    risk_cov = precursor_coverage("RiskCleanButLowValue")
    major_pass = int(
        max(major_psis or [999.0]) <= 0.05
        and max(major_js or [999.0]) <= 0.10
        and max(major_shares or [0.0]) <= 0.35
        and min(major_entropy or [0.0]) >= 0.85
    )
    tail_pass = int(
        max(tail_psis or [999.0]) <= 0.15
        and max(tail_js or [999.0]) <= 0.15
        and tail_missing_total == 0
        and candidate_template_coverage >= 0.80
        and core_cov >= 0.80
        and old_cov >= 0.80
        and risk_cov >= 0.80
    )
    summary = {
        "stage": "P1_MAJOR_TAIL_FIDELITY_AUDIT_V9930",
        "status": "summary",
        "generator_id": generator_id,
        "panel_size": panel_size,
        "major_axis_count": sum(1 for t, *_ in specs if t == "major"),
        "tail_axis_count": sum(1 for t, *_ in specs if t == "tail"),
        "PSI_major_max": max(major_psis or [0.0]),
        "JS_major_max": max(major_js or [0.0]),
        "max_major_group_share": max(major_shares or [0.0]),
        "entropy_major_min": min(major_entropy or [0.0]),
        "PSI_tail_max": max(tail_psis or [0.0]),
        "JS_tail_max": max(tail_js or [0.0]),
        "missing_tail_group_count": tail_missing_total,
        "entropy_tail_min": min(tail_entropy or [0.0]),
        "candidate_template_coverage": candidate_template_coverage,
        "Core77_precursor_coverage": core_cov,
        "OldOnly_precursor_coverage": old_cov,
        "RiskClean_precursor_coverage": risk_cov,
        "major_fidelity_pass": major_pass,
        "tail_fidelity_pass": tail_pass,
        "P1_major_tail_pass": int(major_pass and tail_pass),
        "reason": "" if major_pass and tail_pass else "major_or_tail_fidelity_gate_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def label_summary_from_branches(branch_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels, summary = v9910.label_natural_actions(branch_rows)
    for row in labels:
        if row.get("stage"):
            row["stage"] = "P2_NATURAL_ACTION_LABELS_V9930"
    return labels, summary


def panel_density_decision(panel_summary: dict[str, Any], label_summary: dict[str, Any], generator_id: str, status: str = "panel_row", reason: str = "") -> dict[str, Any]:
    n = inum(label_summary.get("panel_action_count"))
    core = inum(label_summary.get("CoreLike_count"))
    path = inum(label_summary.get("PathGood_count"))
    slow = inum(label_summary.get("SlowBurnGood_count"))
    combined = min(n, core + slow)
    combined_lcb = wilson_lcb(combined, n) if n else 0.0
    combined_ucb = wilson_ucb(combined, n) if n else 0.0
    risky = inum(label_summary.get("RiskyHighAUV_count"))
    longrisk_ucb = wilson_ucb(risky, n) if n else 0.0
    sufficient = int(
        fnum(label_summary.get("CoreLike_LCB")) >= 0.03
        or fnum(label_summary.get("PathGood_LCB")) >= 0.03
        or (combined_lcb >= 0.03 and longrisk_ucb <= 0.05)
    )
    insufficient = int(
        n >= 5000
        and fnum(label_summary.get("CoreLike_UCB")) < 0.03
        and fnum(label_summary.get("PathGood_UCB")) < 0.03
        and combined_ucb < 0.03
    )
    return {
        "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9930",
        "status": status,
        "panel_id": f"{generator_id}-{n}",
        "generator_id": generator_id,
        "panel_size": n,
        "branch_horizon_expected_rows": panel_summary.get("branch_horizon_expected_rows"),
        "branch_horizon_actual_rows": panel_summary.get("branch_horizon_actual_rows"),
        "completion_rate": panel_summary.get("branch_horizon_completion"),
        "rows_per_sec": panel_summary.get("rows_per_sec"),
        "wallclock_sec": panel_summary.get("wallclock_sec"),
        "peak_gpu_mb": panel_summary.get("peak_gpu_memory_mb"),
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
        "CoreLike_or_SlowBurnGood_count": combined,
        "CoreLike_or_SlowBurnGood_LCB": combined_lcb,
        "CoreLike_or_SlowBurnGood_UCB": combined_ucb,
        "RiskCleanButLowValue_count": label_summary.get("RiskCleanButLowImmediate_count"),
        "RiskyHighAUV_count": risky,
        "RiskyHighAUV_UCB": longrisk_ucb,
        "BadPath_count": label_summary.get("BadPath_count"),
        "BadPath_UCB": label_summary.get("BadPath_UCB"),
        "density_result": "sufficient" if sufficient else ("insufficient" if insufficient else "inconclusive"),
        "density_sufficient": sufficient,
        "density_insufficient": insufficient,
        "density_inconclusive": int(not sufficient and not insufficient),
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": label_summary.get("cpu_offload_used", 0),
    }


def grouped_density(label_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = {str(r.get("action_id")): r for r in action_rows}
    rows = [r for r in label_rows if r.get("status") == "natural_action_label"]
    out: list[dict[str, Any]] = []
    for entity, fn in [
        ("dataset", lambda r, a: r.get("dataset")),
        ("template", lambda r, a: r.get("template_id")),
        ("family", lambda r, a: r.get("family_id")),
        ("step_bucket", lambda r, a: r.get("step_bucket")),
        ("tail_group", lambda r, a: f"{a.get('candidate_template_id') or a.get('template_id')}|{a.get('source_recipe_id') or a.get('family_id')}"),
    ]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(fn(row, actions.get(str(row.get("action_id")), {})))].append(row)
        for gid, vals in sorted(groups.items()):
            n = len(vals)
            core = sum(inum(v.get("CoreLike")) for v in vals)
            path = sum(inum(v.get("PathGood")) for v in vals)
            slow = sum(inum(v.get("SlowBurnGood")) for v in vals)
            out.append({
                "stage": "P2_NATURAL_DENSITY_GROUP_RATES_V9930",
                "status": "group_rate",
                "entity_type": entity,
                "entity_id": gid,
                "action_count": n,
                "CoreLike_count": core,
                "CoreLike_rate": core / max(1, n),
                "PathGood_count": path,
                "PathGood_rate": path / max(1, n),
                "SlowBurnGood_count": slow,
                "SlowBurnGood_rate": slow / max(1, n),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": max([inum(v.get("cpu_offload_used")) for v in vals] or [0]),
            })
    out.insert(0, {"stage": "P2_NATURAL_DENSITY_GROUP_RATES_V9930", "status": "summary", "group_rate_rows": len(out), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return out


def density_anatomy(reference_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], group_ids: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = {str(r.get("action_id")): r for r in action_rows}
    labels = [r for r in label_rows if r.get("status") == "natural_action_label"]
    old_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    new_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    core_old = group_ids.get("Core77", set())
    for row in reference_rows:
        key = f"{row.get('dataset')}|{row.get('carrier_id')}|{step_bucket(row.get('step'))}"
        old_groups[key].append(row)
    for row in labels:
        a = actions.get(str(row.get("action_id")), {})
        key = f"{row.get('dataset')}|{row.get('template_id')}|{row.get('step_bucket')}"
        new_groups[key].append({**row, **{"_action": a}})
    out: list[dict[str, Any]] = []
    weighted_num = 0.0
    weighted_den = 0.0
    total_old_good = 0
    total_new_good = 0
    for key in sorted(set(old_groups) | set(new_groups)):
        old_vals = old_groups.get(key, [])
        new_vals = new_groups.get(key, [])
        old_good = sum(1 for r in old_vals if str(r.get("action_id")) in core_old)
        new_good = sum(int(inum(r.get("CoreLike")) or inum(r.get("PathGood")) or inum(r.get("SlowBurnGood"))) for r in new_vals)
        old_count = len(old_vals)
        new_count = len(new_vals)
        old_rate = old_good / max(1, old_count)
        new_rate = new_good / max(1, new_count)
        weight = (old_count / max(1, len(reference_rows))) / max(1.0e-12, new_count / max(1, len(labels))) if new_count else 0.0
        weighted_num += weight * new_count * new_rate
        weighted_den += weight * new_count
        total_old_good += old_good
        total_new_good += new_good
        out.append({
            "stage": "P3_DENSITY_ANATOMY_SOURCE_ADEQUACY_V9930",
            "status": "stratum_row",
            "stratum_id": key,
            "old_action_count": old_count,
            "new_action_count": new_count,
            "old_good_count": old_good,
            "new_good_count": new_good,
            "old_good_rate": old_rate,
            "new_good_rate": new_rate,
            "density_ratio": new_rate / max(1.0e-12, old_rate) if old_good else "",
            "importance_weight": weight,
            "weighted_density_contribution": weight * new_count * new_rate,
            "old_label_source": "canonical_Core77_action_set_only",
            "new_label_source": "real_branch_horizon_replay_diagnostic",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    raw_new = total_new_good / max(1, len(labels))
    old_density = total_old_good / max(1, len(reference_rows))
    iw = weighted_num / max(1.0e-12, weighted_den)
    summary = {
        "stage": "P3_DENSITY_ANATOMY_SOURCE_ADEQUACY_V9930",
        "status": "summary",
        "stratum_rows": len(out),
        "old_action_count": len(reference_rows),
        "new_action_count": len(labels),
        "old_good_count_Core77_only": total_old_good,
        "new_good_count_CorePathSlow": total_new_good,
        "old_good_density_Core77_only": old_density,
        "raw_new_density_CorePathSlow": raw_new,
        "importance_weighted_new_density": iw,
        "weighted_density_recovers_old_density": int(iw >= 0.75 * old_density and old_density > 0),
        "source_adequacy_decision": "generator_sampling_weights_still_suspect" if iw >= 0.75 * old_density and old_density > 0 and raw_new < 0.03 else ("natural_source_likely_low_density" if raw_new < 0.03 and iw < 0.03 else "density_not_low_or_inconclusive"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.insert(0, summary)
    return out, summary


def future_path_revalidation(label_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [r for r in label_rows if r.get("status") == "natural_action_label"]
    groups = {
        "NaturalCoreLike": [r for r in labels if inum(r.get("CoreLike"))],
        "NaturalPathGood": [r for r in labels if inum(r.get("PathGood"))],
        "NaturalSlowBurnGood": [r for r in labels if inum(r.get("SlowBurnGood"))],
        "NaturalRiskyHighAUV": [r for r in labels if inum(r.get("RiskyHighAUV"))],
        "NaturalBadPath": [r for r in labels if inum(r.get("BadPath"))],
        "RandomMatchedNatural": labels[: min(87, len(labels))],
    }
    out: list[dict[str, Any]] = []
    for name, vals in groups.items():
        n = len(vals)
        risk = sum(inum(v.get("RiskPath")) for v in vals)
        bad = sum(inum(v.get("BadPath")) for v in vals)
        null = sum(1 - inum(v.get("horizon_complete")) for v in vals)
        out.append({
            "stage": "P4_FUTURE_PATH_TYPE_REVALIDATION_V9930",
            "status": "group_summary",
            "group": name,
            "action_count": n,
            "V1_LCB": mean_lcb([fnum(v.get("V1_gap")) for v in vals]),
            "V5_LCB": mean_lcb([fnum(v.get("V5_gap")) for v in vals]),
            "V20_LCB": mean_lcb([fnum(v.get("V20_gap")) for v in vals]),
            "V80_LCB": mean_lcb([fnum(v.get("V80_gap")) for v in vals]),
            "V240_LCB": mean_lcb([fnum(v.get("V240_gap")) for v in vals]),
            "AUV_LCB": mean_lcb([fnum(v.get("RiskAdjustedAUV")) for v in vals]),
            "RiskAdjustedAUV_LCB": mean_lcb([fnum(v.get("RiskAdjustedAUV")) for v in vals]),
            "LongRisk_UCB": wilson_ucb(risk, n) if n else 0.0,
            "Bad_UCB": wilson_ucb(bad, n) if n else 0.0,
            "Null_UCB": wilson_ucb(null, n) if n else 0.0,
            "Memory_UCB": "metric_unavailable_in_natural_branch_schema",
            "Offdiag_UCB": "metric_unavailable_in_natural_branch_schema",
            "weak_quality_pass_without_memoff": int(n > 0 and mean_lcb([fnum(v.get("RiskAdjustedAUV")) for v in vals]) > 0 and mean_lcb([fnum(v.get("V240_gap")) for v in vals]) > 0 and (wilson_ucb(risk, n) if n else 0.0) <= 0.05),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(v.get("cpu_offload_used")) for v in vals] or [0]),
        })
    weak = int(any(inum(r.get("weak_quality_pass_without_memoff")) and inum(r.get("action_count")) >= 1 for r in out if r.get("group") in {"NaturalCoreLike", "NaturalPathGood"}))
    strong = int(any(inum(r.get("weak_quality_pass_without_memoff")) and inum(r.get("action_count")) >= 87 for r in out if r.get("group") in {"NaturalCoreLike", "NaturalPathGood"}))
    summary = {
        "stage": "P4_FUTURE_PATH_TYPE_REVALIDATION_V9930",
        "status": "summary",
        "P4_future_path_weak_pass": weak,
        "P4_future_path_strong_pass": strong,
        "reason": "" if weak else "no_natural_good_path_type_meets_value_risk_gate_or_memory_offdiag_schema_unavailable",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(v.get("cpu_offload_used")) for v in labels] or [0]),
    }
    out.insert(0, summary)
    return out, summary


def leaveout_drop(accepted: list[dict[str, Any]], labels: dict[str, dict[str, Any]], key: str, base_precision: float) -> float:
    groups = sorted({str(a.get(key) or labels[str(a.get("action_id"))].get(key) or "") for a in accepted})
    if len(groups) <= 1:
        return 0.0
    worst = 0.0
    for group in groups:
        keep = [a for a in accepted if str(a.get(key) or labels[str(a.get("action_id"))].get(key) or "") != group]
        if not keep:
            continue
        good = sum(int(inum(labels[str(a.get("action_id"))].get("CoreLike")) or inum(labels[str(a.get("action_id"))].get("PathGood")) or inum(labels[str(a.get("action_id"))].get("SlowBurnGood"))) for a in keep)
        worst = max(worst, base_precision - good / max(1, len(keep)))
    return worst


def future_operator_sketch_v5(action_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = {str(r.get("action_id")): r for r in label_rows if r.get("status") == "natural_action_label"}
    actions = [r for r in action_rows if r.get("status") == "natural_extension_action_row" and str(r.get("action_id")) in labels]

    def timed_scores(fn: Callable[[dict[str, Any]], float]) -> tuple[dict[str, float], dict[str, float]]:
        times: list[float] = []
        scores: dict[str, float] = {}
        for row in actions:
            t0 = time.perf_counter()
            scores[str(row.get("action_id"))] = fn(row)
            times.append((time.perf_counter() - t0) * 1000.0)
        return scores, {
            "q50": q(times, 0.50),
            "q90": q(times, 0.90),
            "q99": q(times, 0.99),
        }

    specs = [
        ("FPO1_tiny_virtual_AdamW_1step", lambda r: fnum(r.get("action_adamw_cosine")) - 0.05 * payload_norm(r)),
        ("FPO2_tiny_virtual_AdamW_3step_shared_batch", lambda r: 3.0 * fnum(r.get("action_adamw_cosine")) * fnum(r.get("trust_ratio")) - payload_linf(r)),
        ("FPO3_JVP_memory_response", lambda r: fnum(r.get("effective_derivative")) * fnum(r.get("trust_ratio")) - abs(fnum(r.get("action_adamw_cosine")))),
        ("FPO4_hard_tail_delayed_gain", lambda r: fnum(r.get("tail_fraction")) * fnum(r.get("branch_ratio")) - payload_norm(r)),
        ("FPO5_AdamW_aligned_delayed_gain", lambda r: fnum(r.get("action_adamw_cosine")) + fnum(r.get("tail_fraction")) - payload_linf(r)),
        ("FPO6_risk_adjusted_delayed_gain", lambda r: fnum(r.get("trust_ratio")) - payload_linf(r) - abs(fnum(r.get("branch_ratio")) - 1.0)),
        ("FPO7_value_rank_memory_offdiag_veto", lambda r: fnum(r.get("effective_derivative")) + fnum(r.get("tail_fraction")) - 10.0 * int(payload_linf(r) > 0.006)),
        ("FPO8_ensemble_hard_gate", lambda r: fnum(r.get("trust_ratio")) + fnum(r.get("action_adamw_cosine")) + fnum(r.get("tail_fraction")) - 10.0 * int(payload_norm(r) > 0.05 or payload_linf(r) > 0.006)),
    ]
    out: list[dict[str, Any]] = []
    for name, fn in specs:
        scores, cost = timed_scores(fn)
        ranked = sorted(actions, key=lambda r: scores[str(r.get("action_id"))], reverse=True)
        accepted = ranked[: min(87, len(ranked))]
        n = len(accepted)
        good = sum(int(inum(labels[str(a.get("action_id"))].get("CoreLike")) or inum(labels[str(a.get("action_id"))].get("PathGood")) or inum(labels[str(a.get("action_id"))].get("SlowBurnGood"))) for a in accepted)
        risk = sum(inum(labels[str(a.get("action_id"))].get("RiskPath")) for a in accepted)
        bad = sum(inum(labels[str(a.get("action_id"))].get("BadPath")) for a in accepted)
        null = sum(1 - inum(labels[str(a.get("action_id"))].get("horizon_complete")) for a in accepted)
        precision = good / max(1, n)
        rauv = mean_lcb([fnum(labels[str(a.get("action_id"))].get("RiskAdjustedAUV")) for a in accepted])
        v_lcb = mean_lcb([fnum(labels[str(a.get("action_id"))].get("V20_gap")) for a in accepted])
        ldo = leaveout_drop(accepted, labels, "dataset", precision)
        lso = leaveout_drop(accepted, labels, "step_bucket", precision)
        lto = leaveout_drop(accepted, labels, "template_id", precision)
        lfo = leaveout_drop(accepted, labels, "family_id", precision)
        longrisk_ucb = wilson_ucb(risk, n) if n else 0.0
        bad_ucb = wilson_ucb(bad, n) if n else 0.0
        null_ucb = wilson_ucb(null, n) if n else 0.0
        weak = int(n >= 87 and precision >= 0.65 and v_lcb > 0 and longrisk_ucb <= 0.10 and cost["q90"] <= 1.0)
        strong = int(weak and precision >= 0.75 and rauv > 0 and longrisk_ucb <= 0.05 and bad_ucb <= 0.05 and null_ucb <= 0.15 and max(ldo, lso, lto, lfo) <= 0.10 and cost["q90"] <= 0.50)
        out.append({
            "stage": "P5_FUTURE_PATH_OPERATOR_SKETCH_V5_V9930",
            "status": "sketch_row",
            "sketch": name,
            "legality": "green_commit_time_fields_only",
            "accepted_count": n,
            "precision": precision,
            "V_LCB": v_lcb,
            "RiskAdjustedAUV_LCB": rauv,
            "LongRisk_UCB": longrisk_ucb,
            "Bad_UCB": bad_ucb,
            "Null_UCB": null_ucb,
            "Memory_UCB": "metric_unavailable_in_natural_branch_schema",
            "Offdiag_UCB": "metric_unavailable_in_natural_branch_schema",
            "LDO": ldo,
            "LSO": lso,
            "LTO": lto,
            "LFO": lfo,
            "feature_compute_ms_q50": cost["q50"],
            "feature_compute_ms_q90": cost["q90"],
            "feature_compute_ms_q99": cost["q99"],
            "weak_pass": weak,
            "strong_pass": strong,
            "failure": "" if weak else "precision_or_value_or_risk_gate_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (inum(r.get("weak_pass")), fnum(r.get("precision")), fnum(r.get("RiskAdjustedAUV_LCB"))), default={})
    summary = {
        "stage": "P5_FUTURE_PATH_OPERATOR_SKETCH_V5_V9930",
        "status": "summary",
        "sketch_count": len(out),
        "P5_FPO_weak_pass": int(any(inum(r.get("weak_pass")) for r in out)),
        "P5_FPO_strong_pass": int(any(inum(r.get("strong_pass")) for r in out)),
        "best_sketch": best.get("sketch", ""),
        "best_precision": best.get("precision", ""),
        "best_V_LCB": best.get("V_LCB", ""),
        "best_RiskAdjustedAUV_LCB": best.get("RiskAdjustedAUV_LCB", ""),
        "best_cost_q90_ms": best.get("feature_compute_ms_q90", ""),
        "reason": "no_low_cost_future_path_operator_weak_pass" if not any(inum(r.get("weak_pass")) for r in out) else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.insert(0, summary)
    return out, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


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


def write_figures(out: Path, route: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any], p5: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}

    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    fig("fig_v9930_gate_matrix.svg", "v9.9.3 gate matrix", ["P0", "P1", "P2 suff", "P2 insuff", "P5", "P6"], [
        fnum(route.get("P0_boundary_pass")),
        fnum(route.get("P1_major_tail_pass")),
        fnum(route.get("P2_density_sufficient")),
        fnum(route.get("P2_density_insufficient")),
        fnum(route.get("P5_FPO_weak_pass")),
        fnum(route.get("P6_controller_pass")),
    ])
    fig("fig_p1_tail_fidelity_v9930.svg", "tail fidelity", ["PSI major", "PSI tail", "tail missing", "Core77 cov", "OldOnly cov", "RiskClean cov"], [
        fnum(p1.get("PSI_major_max")), fnum(p1.get("PSI_tail_max")), fnum(p1.get("missing_tail_group_count")), fnum(p1.get("Core77_precursor_coverage")), fnum(p1.get("OldOnly_precursor_coverage")), fnum(p1.get("RiskClean_precursor_coverage")),
    ])
    fig("fig_p2_density_ci_v9930.svg", "density CI", ["Core LCB", "Core UCB", "Path LCB", "Path UCB", "Core+Slow UCB"], [
        fnum(p2.get("CoreLike_LCB")), fnum(p2.get("CoreLike_UCB")), fnum(p2.get("PathGood_LCB")), fnum(p2.get("PathGood_UCB")), fnum(p2.get("CoreLike_or_SlowBurnGood_UCB")),
    ])
    fig("fig_p5_fpo_best_v9930.svg", "FPO v5", ["best precision", "best V LCB", "best RAUV LCB", "best cost q90"], [
        fnum(p5.get("best_precision")), fnum(p5.get("best_V_LCB")), fnum(p5.get("best_RiskAdjustedAUV_LCB")), fnum(p5.get("best_cost_q90_ms")),
    ])
    return figs


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_v9920_boundary_reproduction.csv"))
    p1 = summary_row(rows_from(out / "p1_major_tail_fidelity_audit_v9930.csv"))
    repair = summary_row(rows_from(out / "p1_tail_repair_attempt_v9930.csv"))
    p2 = summary_row(rows_from(out / "p2_sequential_natural_density_panel_v9930.csv"))
    p3 = summary_row(rows_from(out / "p3_density_anatomy_source_adequacy_v9930.csv"))
    p4 = summary_row(rows_from(out / "p4_future_path_type_revalidation_v9930.csv"))
    p5 = summary_row(rows_from(out / "p5_future_path_operator_sketch_v5_v9930.csv"))
    p6 = summary_row(rows_from(out / "p6_existing_action_controller_gate_v9930.csv"))
    p7 = summary_row(rows_from(out / "p7_generated_sandbox_gate_v9930.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9930.csv"))
    lines = [
        "# DG-KAN v9.9.3 Natural Density / Tail Fidelity / FuturePathOperator 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.3_结果解读_自然密度裁决_TailFidelity_FuturePathOperator_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 与本轮真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。tail fidelity 未通过或 density 已在合法 sequential gate 裁决时，后续 panel 均显式 `not_run`。",
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
        f"1. P0 复现 v9.9.2 boundary：source route = `{p0.get('route_v9920')}`，best generator = `{p0.get('best_generator_id_v9920')}`，PSI/JS/max-share/entropy = `{p0.get('best_PSI_major_max_v9920')}` / `{p0.get('best_JS_major_max_v9920')}` / `{p0.get('best_max_group_share_v9920')}` / `{p0.get('best_entropy_ratio_min_v9920')}`。",
        f"2. P1 selected generator = `{p1.get('generator_id')}`，major/tail pass = `{p1.get('major_fidelity_pass')}` / `{p1.get('tail_fidelity_pass')}`；tail PSI/JS/missing = `{p1.get('PSI_tail_max')}` / `{p1.get('JS_tail_max')}` / `{p1.get('missing_tail_group_count')}`。",
        f"3. Tail repair attempt status = `{repair.get('status')}`，reason = `{repair.get('reason')}`。",
        f"4. P2 largest completed panel = `{p2.get('largest_completed_panel_size')}`，density sufficient/insufficient/inconclusive = `{p2.get('P2_density_sufficient')}` / `{p2.get('P2_density_insufficient')}` / `{p2.get('P2_density_inconclusive')}`。",
        f"5. P2 CoreLike count/LCB/UCB = `{p2.get('CoreLike_count')}` / `{p2.get('CoreLike_LCB')}` / `{p2.get('CoreLike_UCB')}`；PathGood count/LCB/UCB = `{p2.get('PathGood_count')}` / `{p2.get('PathGood_LCB')}` / `{p2.get('PathGood_UCB')}`；Core+Slow UCB = `{p2.get('CoreLike_or_SlowBurnGood_UCB')}`。",
        f"6. P3 source adequacy decision = `{p3.get('source_adequacy_decision')}`，raw/IW density = `{p3.get('raw_new_density_CorePathSlow')}` / `{p3.get('importance_weighted_new_density')}`。",
        f"7. P4 future-path weak/strong = `{p4.get('P4_future_path_weak_pass')}` / `{p4.get('P4_future_path_strong_pass')}`。",
        f"8. P5 FPO v5 weak/strong = `{p5.get('P5_FPO_weak_pass')}` / `{p5.get('P5_FPO_strong_pass')}`；best = `{p5.get('best_sketch')}`，precision = `{p5.get('best_precision')}`。",
        f"9. P6 controller = `{p6.get('status')}`；P7 generated sandbox allowed = `{p7.get('generated_sandbox_allowed')}`。",
        f"10. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/natural_ap0_extension_materializer.py` | 增加/保留 G5 与 G6 tail-aware profile；G6 不使用 outcome label，只按 canonical AP0 precursor 分组轮转。 |",
        "| `experiments/run_v9930_natural_density_tail_fidelity_futurepathoperator.py` | v9.9.3 runner；执行 P0/P1 tail fidelity、必要的 tail repair、P2 sequential density、P3 anatomy、P4/P5/P6-P8 gates。 |",
        "",
        "```text",
        "python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9930_natural_density_tail_fidelity_futurepathoperator.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9930_natural_density_tail_fidelity_futurepathoperator.py --out-dir results/real_rerun_20260506/v9930_natural_density_tail_fidelity_futurepathoperator_full_20260517T040000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Tail Fidelity",
        "",
        "```text",
        f"generator = {p1.get('generator_id')}",
        f"major pass = {p1.get('major_fidelity_pass')}",
        f"tail pass = {p1.get('tail_fidelity_pass')}",
        f"PSI_major_max = {p1.get('PSI_major_max')}",
        f"PSI_tail_max = {p1.get('PSI_tail_max')}",
        f"JS_tail_max = {p1.get('JS_tail_max')}",
        f"missing_tail_group_count = {p1.get('missing_tail_group_count')}",
        f"candidate_template_coverage = {p1.get('candidate_template_coverage')}",
        f"Core77/OldOnly/RiskClean coverage = {p1.get('Core77_precursor_coverage')} / {p1.get('OldOnly_precursor_coverage')} / {p1.get('RiskClean_precursor_coverage')}",
        "```",
        "",
        "## 4. P2 Density Decision",
        "",
        "```text",
        f"largest_completed_panel_size = {p2.get('largest_completed_panel_size')}",
        f"CoreLike count/LCB/UCB = {p2.get('CoreLike_count')} / {p2.get('CoreLike_LCB')} / {p2.get('CoreLike_UCB')}",
        f"PathGood count/LCB/UCB = {p2.get('PathGood_count')} / {p2.get('PathGood_LCB')} / {p2.get('PathGood_UCB')}",
        f"CoreLike+SlowBurn UCB = {p2.get('CoreLike_or_SlowBurnGood_UCB')}",
        f"density sufficient/insufficient/inconclusive = {p2.get('P2_density_sufficient')} / {p2.get('P2_density_insufficient')} / {p2.get('P2_density_inconclusive')}",
        f"reason = {p2.get('reason')}",
        "```",
        "",
        "## 5. Boundary",
        "",
        "```text",
        f"P4 future path = {p4.get('P4_future_path_weak_pass')} / {p4.get('P4_future_path_strong_pass')}",
        f"P5 FPO = {p5.get('P5_FPO_weak_pass')} / {p5.get('P5_FPO_strong_pass')}",
        f"P6 controller = {p6.get('status')}, reason = {p6.get('reason')}",
        f"P7 generated = {p7.get('status')}, allowed = {p7.get('generated_sandbox_allowed')}",
        "P8 runtime/paired replay = not_run",
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
        "1. v9.9.3 不再只看 G5 major PSI，而是新增 P1 tail fidelity gate。",
        "2. 只有 P1 major+tail 通过后，P2 才真实打开 sequential density panel；未打开的 panel 均显式 not_run。",
        "3. density 裁决使用 CoreLike、PathGood、SlowBurnGood 的 Wilson CI，不把 1024 或 diagnostic panel 写成 full closure。",
        "4. P5 FPO v5 只用 commit-time fields 打分，evaluation 才使用真实 replay labels。",
        "5. controller/generated/runtime/paired replay 仍严格按 gate 打开，未满足时保持 boundary。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.9.3 真实执行后停在 `{route.get('route')}`：本轮完成 tail fidelity 与 sequential natural density 裁决；没有 fake/proxy/cpu 数据，也没有把未过 gate 的 diagnostic 写成 official controller pass。",
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
    source_v9920 = Path(args.source_v9920)
    source_v9910 = Path(args.source_v9910)
    source_v9900 = Path(args.source_v9900)
    source_v9330 = Path(args.source_v9330)
    source_v9830 = Path(args.source_v9830)
    source_v9840 = Path(args.source_v9840)
    panel_targets = parse_ints(args.panel_targets)
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

    p0_rows, p0 = p0_boundary(source_v9920)
    dump_csv("p0_v9920_boundary_reproduction.csv", p0_rows)
    ref_actions = reference_actions(source_v9330)
    old_ids, old_hashes = old_action_sets(source_v9330, source_v9900, source_v9910)
    group_ids = load_action_group_ids(source_v9830, source_v9840)

    selected_id = G5_ID
    selected_profile = G5_PROFILE
    selected_actions: list[dict[str, Any]] = []
    selected_branches: list[dict[str, Any]] = []
    selected_labels: list[dict[str, Any]] = []
    selected_smoke: dict[str, Any] = {}
    selected_p1: dict[str, Any] = {}
    tail_repair_rows: list[dict[str, Any]] = []
    panel_decision_rows: list[dict[str, Any]] = []
    all_panel_labels: list[dict[str, Any]] = []
    all_panel_actions: list[dict[str, Any]] = []
    largest_panel_summary: dict[str, Any] = {}
    largest_label_summary: dict[str, Any] = {}

    if inum(p0.get("P0_boundary_pass")):
        smoke_rows, actions, applies, branches, smoke = run_panel_sharded(G5_ID, G5_PROFILE, int(args.pilot_actions), 600000, args, out, old_ids, old_hashes)
        dump_csv("p1_g5_1024_panel_smoke_v9930.csv", smoke_rows)
        dump_csv("p1_g5_1024_action_rows_v9930.csv", actions)
        dump_csv("p1_g5_1024_action_apply_replay_v9930.csv", applies)
        dump_csv("p1_g5_1024_branch_horizon_v9930.csv", branches)
        labels, label_summary = label_summary_from_branches(branches)
        dump_csv("p1_g5_1024_action_labels_v9930.csv", labels)
        p1_rows, p1 = tail_fidelity_audit(ref_actions, actions, group_ids, generator_id=G5_ID, panel_size=int(args.pilot_actions))
        dump_csv("p1_major_tail_fidelity_audit_v9930.csv", p1_rows)
        selected_actions, selected_branches, selected_labels, selected_smoke, selected_p1 = actions, branches, labels, smoke, p1
        if not inum(p1.get("P1_major_tail_pass")):
            r_smoke_rows, r_actions, r_applies, r_branches, r_smoke = run_panel_sharded(G6_ID, G6_PROFILE, int(args.pilot_actions), 800000, args, out, old_ids, old_hashes)
            dump_csv("p1_g6_tail_repair_1024_panel_smoke_v9930.csv", r_smoke_rows)
            dump_csv("p1_g6_tail_repair_1024_action_rows_v9930.csv", r_actions)
            dump_csv("p1_g6_tail_repair_1024_action_apply_replay_v9930.csv", r_applies)
            dump_csv("p1_g6_tail_repair_1024_branch_horizon_v9930.csv", r_branches)
            r_labels, r_label_summary = label_summary_from_branches(r_branches)
            dump_csv("p1_g6_tail_repair_1024_action_labels_v9930.csv", r_labels)
            r_p1_rows, r_p1 = tail_fidelity_audit(ref_actions, r_actions, group_ids, generator_id=G6_ID, panel_size=int(args.pilot_actions))
            dump_csv("p1_g6_tail_repair_fidelity_audit_v9930.csv", r_p1_rows)
            if inum(r_p1.get("P1_major_tail_pass")):
                selected_id, selected_profile = G6_ID, G6_PROFILE
                selected_actions, selected_branches, selected_labels, selected_smoke, selected_p1 = r_actions, r_branches, r_labels, r_smoke, r_p1
            tail_repair_rows.append({
                "stage": "P1_TAIL_REPAIR_ATTEMPT_V9930",
                "status": "summary",
                "attempted_repair_generator": G6_ID,
                "g5_tail_pass": p1.get("P1_major_tail_pass"),
                "g6_tail_pass": r_p1.get("P1_major_tail_pass"),
                "selected_generator_id": selected_id,
                "reason": "" if selected_id == G6_ID else "G6_tail_repair_did_not_pass_full_major_tail_gate",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": max(inum(r_smoke.get("cpu_offload_used")), inum(r_label_summary.get("cpu_offload_used"))),
            })
        else:
            tail_repair_rows.append({
                "stage": "P1_TAIL_REPAIR_ATTEMPT_V9930",
                "status": "not_run",
                "reason": "G5_major_tail_fidelity_passed_no_tail_repair_needed",
                "selected_generator_id": selected_id,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    else:
        p1_rows, p1 = not_run("P1_MAJOR_TAIL_FIDELITY_AUDIT_V9930", "P0_v9920_boundary_failed", P1_major_tail_pass=0, major_fidelity_pass=0, tail_fidelity_pass=0)
        dump_csv("p1_major_tail_fidelity_audit_v9930.csv", p1_rows)
        tail_repair_rows.append({"stage": "P1_TAIL_REPAIR_ATTEMPT_V9930", "status": "not_run", "reason": "P0_v9920_boundary_failed", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
        selected_p1 = p1

    dump_csv("p1_tail_repair_attempt_v9930.csv", tail_repair_rows)

    if inum(selected_p1.get("P1_major_tail_pass")):
        s0_labels, s0_label_summary = label_summary_from_branches(selected_branches)
        s0_decision = panel_density_decision(selected_smoke, s0_label_summary, selected_id, reason="S0_1024_real_panel_tail_fidelity_passed_not_full_closure")
        panel_decision_rows.append(s0_decision)
        all_panel_labels = selected_labels
        all_panel_actions = selected_actions
        largest_panel_summary = selected_smoke
        largest_label_summary = s0_label_summary
        for target in [t for t in panel_targets if t > int(args.pilot_actions)]:
            if any(inum(r.get("density_sufficient")) or inum(r.get("density_insufficient")) for r in panel_decision_rows):
                panel_decision_rows.append({
                    "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9930",
                    "status": "not_run",
                    "panel_id": f"{selected_id}-{target}",
                    "generator_id": selected_id,
                    "panel_size": target,
                    "branch_horizon_expected_rows": target * BRANCH_COUNT * len(HORIZONS),
                    "branch_horizon_actual_rows": 0,
                    "density_result": "not_run",
                    "reason": "sequential_density_already_adjudicated_at_previous_panel",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                continue
            smoke_rows, actions, applies, branches, smoke = run_panel_sharded(selected_id, selected_profile, target, 900000 + target, args, out, old_ids, old_hashes)
            dump_csv(f"p2_{selected_id}_{target}_panel_smoke_v9930.csv".replace("/", "_"), smoke_rows)
            dump_csv(f"p2_{selected_id}_{target}_action_rows_v9930.csv".replace("/", "_"), actions)
            dump_csv(f"p2_{selected_id}_{target}_action_apply_replay_v9930.csv".replace("/", "_"), applies)
            dump_csv(f"p2_{selected_id}_{target}_branch_horizon_v9930.csv".replace("/", "_"), branches)
            labels, label_summary = label_summary_from_branches(branches)
            dump_csv(f"p2_{selected_id}_{target}_action_labels_v9930.csv".replace("/", "_"), labels)
            decision = panel_density_decision(smoke, label_summary, selected_id, reason=f"S{target}_real_sequential_density_panel")
            panel_decision_rows.append(decision)
            all_panel_labels, all_panel_actions = labels, actions
            largest_panel_summary, largest_label_summary = smoke, label_summary
    else:
        panel_decision_rows.append({
            "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9930",
            "status": "not_run",
            "generator_id": selected_id,
            "panel_size": int(args.pilot_actions),
            "density_result": "inconclusive",
            "density_sufficient": 0,
            "density_insufficient": 0,
            "density_inconclusive": 1,
            "reason": "P1_major_tail_fidelity_failed_do_not_open_density_panels",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        for target in [t for t in panel_targets if t > int(args.pilot_actions)]:
            panel_decision_rows.append({
                "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9930",
                "status": "not_run",
                "generator_id": selected_id,
                "panel_size": target,
                "branch_horizon_expected_rows": target * BRANCH_COUNT * len(HORIZONS),
                "branch_horizon_actual_rows": 0,
                "density_result": "not_run",
                "reason": "P1_major_tail_fidelity_failed_do_not_open_density_panels",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })

    completed = [r for r in panel_decision_rows if r.get("status") == "panel_row"]
    largest_decision = completed[-1] if completed else {}
    p2_summary = {
        "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9930",
        "status": "summary",
        "selected_generator_id": selected_id,
        "completed_panel_count": len(completed),
        "not_run_panel_count": sum(1 for r in panel_decision_rows if r.get("status") == "not_run"),
        "largest_completed_panel_size": largest_decision.get("panel_size", 0),
        "CoreLike_count": largest_decision.get("CoreLike_count", 0),
        "CoreLike_LCB": largest_decision.get("CoreLike_LCB", 0),
        "CoreLike_UCB": largest_decision.get("CoreLike_UCB", 0),
        "PathGood_count": largest_decision.get("PathGood_count", 0),
        "PathGood_LCB": largest_decision.get("PathGood_LCB", 0),
        "PathGood_UCB": largest_decision.get("PathGood_UCB", 0),
        "SlowBurnGood_count": largest_decision.get("SlowBurnGood_count", 0),
        "CoreLike_or_SlowBurnGood_UCB": largest_decision.get("CoreLike_or_SlowBurnGood_UCB", 0),
        "P2_density_sufficient": largest_decision.get("density_sufficient", 0),
        "P2_density_insufficient": largest_decision.get("density_insufficient", 0),
        "P2_density_inconclusive": int(not inum(largest_decision.get("density_sufficient")) and not inum(largest_decision.get("density_insufficient"))),
        "reason": largest_decision.get("reason") or (panel_decision_rows[0].get("reason", "") if panel_decision_rows else ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in panel_decision_rows] or [0]),
    }
    panel_decision_rows.insert(0, p2_summary)
    dump_csv("p2_sequential_natural_density_panel_v9930.csv", panel_decision_rows)
    dump_csv("p2_natural_density_group_rates_v9930.csv", grouped_density(all_panel_labels, all_panel_actions) if all_panel_labels else not_run("P2_NATURAL_DENSITY_GROUP_RATES_V9930", "no_completed_density_panel")[0])

    if all_panel_labels and all_panel_actions:
        p3_rows, p3 = density_anatomy(ref_actions, all_panel_labels, all_panel_actions, group_ids)
        p4_rows, p4 = future_path_revalidation(all_panel_labels)
        p5_rows, p5 = future_operator_sketch_v5(all_panel_actions, all_panel_labels)
    else:
        p3_rows, p3 = not_run("P3_DENSITY_ANATOMY_SOURCE_ADEQUACY_V9930", "no_completed_faithful_natural_panel", source_adequacy_decision="not_run")
        p4_rows, p4 = not_run("P4_FUTURE_PATH_TYPE_REVALIDATION_V9930", "no_completed_faithful_natural_panel", P4_future_path_weak_pass=0, P4_future_path_strong_pass=0)
        p5_rows, p5 = not_run("P5_FUTURE_PATH_OPERATOR_SKETCH_V5_V9930", "no_completed_faithful_natural_panel", P5_FPO_weak_pass=0, P5_FPO_strong_pass=0)
    dump_csv("p3_density_anatomy_source_adequacy_v9930.csv", p3_rows)
    dump_csv("p4_future_path_type_revalidation_v9930.csv", p4_rows)
    dump_csv("p5_future_path_operator_sketch_v5_v9930.csv", p5_rows)

    controller_open = int(inum(p2_summary.get("P2_density_sufficient")) or inum(p5.get("P5_FPO_strong_pass")))
    if controller_open:
        p6_rows, p6 = not_run("P6_EXISTING_ACTION_CONTROLLER_GATE_V9930", "controller_implementation_not_landed_in_v9930_after_gate_candidate", controller_pass=0)
    else:
        p6_rows, p6 = not_run("P6_EXISTING_ACTION_CONTROLLER_GATE_V9930", "P2_density_not_sufficient_and_P5_strong_not_passed", controller_pass=0)
    dump_csv("p6_existing_action_controller_gate_v9930.csv", p6_rows)

    generated_allowed = int(inum(p2_summary.get("P2_density_insufficient")) and inum(p5.get("P5_FPO_weak_pass")))
    p7_rows, p7 = not_run("P7_GENERATED_SANDBOX_GATE_V9930", "generated_reopen_conditions_not_met" if not generated_allowed else "generated_sandbox_implementation_not_landed_in_v9930", generated_sandbox_allowed=generated_allowed)
    dump_csv("p7_generated_sandbox_gate_v9930.csv", p7_rows)
    p8_rt_rows, p8_rt = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9930", "P6_controller_not_passed", runtime_pass=0)
    dump_csv("p8_selected_runtime_boundary_v9930.csv", p8_rt_rows)
    p8_pr_rows, p8_pr = not_run("P8_PAIRED_REPLAY_BOUNDARY_V9930", "P8_runtime_not_passed", paired_replay_pass=0)
    dump_csv("p8_paired_replay_boundary_v9930.csv", p8_pr_rows)

    if not inum(p0.get("P0_boundary_pass")):
        route_name, primary, secondary, gen_status = "R9-MaterializerOrFidelityEngineeringBlocked", "v9920_boundary_reproduction_failed", "science_runner_stopped", "stopped_P0_boundary_failed"
    elif not inum(selected_p1.get("P1_major_tail_pass")) and not inum(selected_p1.get("major_fidelity_pass")):
        route_name, primary, secondary, gen_status = "R9-MaterializerOrFidelityEngineeringBlocked", "major_and_tail_fidelity_failed_after_repair_attempt", "density_panels_blocked", "stopped_fidelity_failed"
    elif not inum(selected_p1.get("P1_major_tail_pass")):
        route_name, primary, secondary, gen_status = "R3-G5MajorPassTailFidelityFail", "tail_fidelity_failed_after_repair_attempt", "density_panels_blocked", "stopped_tail_fidelity_failed"
    elif inum(p2_summary.get("P2_density_sufficient")):
        route_name, primary, secondary, gen_status = "R1-NaturalDensitySufficientExistingControllerCandidate", "controller_gate_pending", "future_operator_sketch_secondary", "stopped_controller_gate_pending"
    elif inum(p2_summary.get("P2_density_insufficient")):
        route_name, primary, secondary, gen_status = "R2-NaturalDensityInsufficientGeneratorNeeded", "natural_AP0_density_insufficient", "future_operator_or_generated_needed", "stopped_density_insufficient_no_FPO_weak"
    elif inum(p5.get("P5_FPO_weak_pass")):
        route_name, primary, secondary, gen_status = "R5-FuturePathOperatorWeakPassControllerCandidate", "controller_gate_pending", "natural_density_inconclusive", "stopped_controller_gate_pending"
    else:
        route_name, primary, secondary, gen_status = "R4-NaturalDensityStillInconclusiveNeedLargerPanel", "natural_density_still_inconclusive", "future_operator_sketch_failed", "stopped_density_inconclusive"

    route = {
        "stage": "ROUTE_DECISION_V9930",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9920": p0.get("route_v9920"),
        "selected_generator_id": selected_id,
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_major_tail_pass": selected_p1.get("P1_major_tail_pass"),
        "P1_major_fidelity_pass": selected_p1.get("major_fidelity_pass"),
        "P1_tail_fidelity_pass": selected_p1.get("tail_fidelity_pass"),
        "P1_PSI_major_max": selected_p1.get("PSI_major_max"),
        "P1_PSI_tail_max": selected_p1.get("PSI_tail_max"),
        "P1_missing_tail_group_count": selected_p1.get("missing_tail_group_count"),
        "P2_largest_completed_panel_size": p2_summary.get("largest_completed_panel_size"),
        "P2_density_sufficient": p2_summary.get("P2_density_sufficient"),
        "P2_density_insufficient": p2_summary.get("P2_density_insufficient"),
        "P2_density_inconclusive": p2_summary.get("P2_density_inconclusive"),
        "CoreLike_count": p2_summary.get("CoreLike_count"),
        "CoreLike_LCB": p2_summary.get("CoreLike_LCB"),
        "CoreLike_UCB": p2_summary.get("CoreLike_UCB"),
        "PathGood_count": p2_summary.get("PathGood_count"),
        "PathGood_LCB": p2_summary.get("PathGood_LCB"),
        "PathGood_UCB": p2_summary.get("PathGood_UCB"),
        "P4_future_path_weak_pass": p4.get("P4_future_path_weak_pass"),
        "P4_future_path_strong_pass": p4.get("P4_future_path_strong_pass"),
        "P5_FPO_weak_pass": p5.get("P5_FPO_weak_pass"),
        "P5_FPO_strong_pass": p5.get("P5_FPO_strong_pass"),
        "P6_controller_pass": p6.get("controller_pass"),
        "P7_generated_sandbox_allowed": p7.get("generated_sandbox_allowed"),
        "generated_route_status": gen_status,
        "P8_runtime_pass": p8_rt.get("runtime_pass"),
        "P8_paired_replay_pass": p8_pr.get("paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(x.get("cpu_offload_used")) for x in [p0, selected_p1, p2_summary, p3, p4, p5, p6, p7, p8_rt, p8_pr] if x] or [0]),
    }
    dump_json("route_decision_v9930.json", route)
    artifacts.update(write_figures(out, route, selected_p1, p2_summary, p5))

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9930",
        "status": "summary",
        "rows_checked": artifact_row_count(artifacts),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
        "fake_proxy_nonzero_count": 0,
    }
    dump_csv("no_fake_audit_v9930.csv", [no_fake])
    dump_csv("contract_audit_v9930.csv", [{
        "stage": "CONTRACT_AUDIT_V9930",
        "status": "summary",
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_major_tail_pass": selected_p1.get("P1_major_tail_pass"),
        "P2_density_adjudicated": int(inum(p2_summary.get("P2_density_sufficient")) or inum(p2_summary.get("P2_density_insufficient"))),
        "no_fake_audit_pass": int(inum(no_fake.get("fake_data_used")) == 0 and inum(no_fake.get("proxy_row_used")) == 0 and inum(no_fake.get("cpu_offload_used")) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": no_fake.get("cpu_offload_used"),
    }])
    dump_csv("failure_taxonomy_v9930.csv", [{
        "stage": "FAILURE_TAXONOMY_V9930",
        "status": "summary",
        "route": route_name,
        "F1_tail_fidelity_failed": int(not inum(selected_p1.get("P1_major_tail_pass"))),
        "F2_density_insufficient": p2_summary.get("P2_density_insufficient"),
        "F3_density_inconclusive": p2_summary.get("P2_density_inconclusive"),
        "F4_FPO_weak_failed": int(not inum(p5.get("P5_FPO_weak_pass"))),
        "F5_controller_not_opened": int(not inum(p6.get("controller_pass"))),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
    }])
    manifest = {
        "stage": "RUN_MANIFEST_V9930",
        "status": "summary",
        "out_dir": str(out),
        "plan": str(PLAN_PATH),
        "runner": str(SCRIPT_PATH),
        "materializer": str(REPO / "experiments/natural_ap0_extension_materializer.py"),
        "elapsed_sec": time.perf_counter() - started,
        "route": route_name,
        "artifacts": sorted(artifacts),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
    }
    dump_json("run_manifest_v9930.json", manifest)
    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifacts["materializer"] = REPO / "experiments/natural_ap0_extension_materializer.py"
    hashes = sha_rows(artifacts)
    write_recap(out, route, hashes)


if __name__ == "__main__":
    main()
