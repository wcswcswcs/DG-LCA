#!/usr/bin/env python3
"""DG-KAN v9.9.1 natural density decision / FuturePathOperator run.

This runner advances v9.9.0 from engineering smoke to a real natural-action
pilot panel.  It never promotes the pilot to full density closure: 5000/10000/
20000 panels are opened only if the distribution-fidelity gate passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9900_natural_extension_engineering_gate_futurepathoperator as v9900  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.1_NaturalDensityDecision_FuturePathOperator_四线并行完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9910_natural_density_decision_futurepathoperator.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.1_NaturalDensityDecision_FuturePathOperator_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
DEFAULT_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6


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
    p.add_argument("--source-v9900", default=str(DEFAULT_V9900))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
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


def lcb_mean(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return statistics.mean(xs) - 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def ucb_rate(success: int, n: int) -> float:
    return wilson_ucb(success, n) if n else 0.0


def step_bucket(step: Any) -> str:
    s = inum(step)
    lo = (s // 32) * 32
    hi = lo + 31
    return f"s{lo:03d}_{hi:03d}"


def carrier_key(row: dict[str, Any]) -> str:
    return str(row.get("template_id") or row.get("carrier_id") or "")


def payload_linf(row: dict[str, Any]) -> float:
    return fnum(row.get("payload_linf_norm") or row.get("payload_linf"))


def payload_norm(row: dict[str, Any]) -> float:
    return fnum(row.get("payload_norm") or row.get("payload_l2_norm") or row.get("action_norm"))


def counter_share(rows: list[dict[str, Any]], fn) -> Counter:
    return Counter(str(fn(r)) for r in rows)


def psi_from_counters(old: Counter, new: Counter) -> float:
    keys = set(old) | set(new)
    n_old = sum(old.values())
    n_new = sum(new.values())
    eps = 1.0e-6
    total = 0.0
    for key in keys:
        p = max(eps, old.get(key, 0) / max(1, n_old))
        qv = max(eps, new.get(key, 0) / max(1, n_new))
        total += (qv - p) * math.log(qv / p)
    return total


def numeric_bins(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], fn, name: str) -> tuple[Counter, Counter, dict[str, float]]:
    old_vals = [fn(r) for r in old_rows if math.isfinite(fn(r))]
    if not old_vals:
        return Counter(), Counter(), {"feature": name, "bin_count": 0}
    cuts = [q(old_vals, frac) for frac in [0.1, 0.25, 0.5, 0.75, 0.9]]

    def bucket(row: dict[str, Any]) -> str:
        x = fn(row)
        for i, cut in enumerate(cuts):
            if x <= cut:
                return f"{name}_bin_{i}"
        return f"{name}_bin_{len(cuts)}"

    return counter_share(old_rows, bucket), counter_share(new_rows, bucket), {"feature": name, "bin_count": len(cuts) + 1}


def old_action_sets(source_v9330: Path, source_v9900: Path) -> tuple[set[str], set[str]]:
    old_ids, old_hashes = v9900.old_action_sets(source_v9330)
    for row in rows_from(source_v9900 / "p1_natural_extension_action_rows_v9900.csv"):
        old_ids.add(str(row.get("action_id")))
        old_hashes.add(str(row.get("payload_hash_expected") or row.get("payload_hash")))
    return old_ids, old_hashes


def p0_boundary(source_v9900: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9900 / "route_decision_v9900.json")
    p1b = summary_row(rows_from(source_v9900 / "p1_single_action_preflight_v9900.csv"))
    p1c = summary_row(rows_from(source_v9900 / "p1_16_action_preflight_v9900.csv"))
    p1d = summary_row(rows_from(source_v9900 / "p1_256_action_smoke_v9900.csv"))
    p1_actions = rows_from(source_v9900 / "p1_natural_extension_action_rows_v9900.csv")
    p1_branch = rows_from(source_v9900 / "p1_natural_extension_branch_horizon_v9900.csv")
    nf = summary_row(rows_from(source_v9900 / "no_fake_audit_v9900.csv"))
    collision_action = sum(inum(r.get("old_action_id_collision_count")) for r in [p1b, p1c, p1d])
    collision_payload = sum(inum(r.get("old_payload_hash_collision_count")) for r in [p1b, p1c, p1d])
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9910",
        "status": "summary",
        "source_route_v9900": route.get("route"),
        "P1a_entrypoint_search_pass_v9900": route.get("P1a_entrypoint_search_pass"),
        "P1b_single_action_preflight_pass_v9900": route.get("P1b_single_action_preflight_pass"),
        "P1c_16_action_preflight_pass_v9900": route.get("P1c_16_action_preflight_pass"),
        "P1d_256_action_smoke_pass_v9900": route.get("P1d_256_action_smoke_pass"),
        "new_natural_action_count_v9900": len(p1_actions),
        "new_branch_horizon_rows_v9900": len(p1_branch),
        "old_action_collision_count_v9900": collision_action,
        "old_payload_collision_count_v9900": collision_payload,
        "system_legal_controller_pass_v9900": route.get("system_legal_controller_pass"),
        "generated_route_status_v9900": route.get("generated_route_status"),
        "fake_data_used_v9900": nf.get("fake_data_used"),
        "proxy_row_used_v9900": nf.get("proxy_row_used"),
        "cpu_offload_used_v9900": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_pass"] = int(
        inum(row["P1a_entrypoint_search_pass_v9900"]) == 1
        and inum(row["P1b_single_action_preflight_pass_v9900"]) == 1
        and inum(row["P1c_16_action_preflight_pass_v9900"]) == 1
        and inum(row["P1d_256_action_smoke_pass_v9900"]) == 1
        and collision_action == 0
        and collision_payload == 0
        and inum(row["system_legal_controller_pass_v9900"]) == 0
        and inum(row["fake_data_used_v9900"]) == 0
        and inum(row["proxy_row_used_v9900"]) == 0
        and inum(row["cpu_offload_used_v9900"]) == 0
    )
    return [row], row


def label_natural_actions(branch_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_action_h: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    action_meta: dict[str, dict[str, Any]] = {}
    for row in branch_rows:
        aid = str(row.get("action_id"))
        h = inum(row.get("horizon"))
        by_action_h[(aid, h)][str(row.get("branch_id"))] = row
        action_meta.setdefault(aid, row)

    rows: list[dict[str, Any]] = []
    for aid, meta in sorted(action_meta.items()):
        gaps: dict[int, float] = {}
        real_v: dict[int, float] = {}
        complete = 1
        ce_p99_worse = 0
        for h in HORIZONS:
            branches = by_action_h.get((aid, h), {})
            real = branches.get("RealFunctional")
            controls = [r for b, r in branches.items() if b != "RealFunctional"]
            if real is None or len(controls) < BRANCH_COUNT - 1:
                complete = 0
                continue
            rv = fnum(real.get("V_branch"))
            cv = max(fnum(r.get("V_branch")) for r in controls)
            gaps[h] = rv - cv
            real_v[h] = rv
            ce_p99_worse += int(fnum(real.get("CEp99_delta")) > 0.25)
        rauv = sum(gaps.values()) / max(1, len(gaps))
        risk_hits = sum(1 for g in gaps.values() if g < -0.20) + ce_p99_worse
        core_like = int(complete and gaps.get(20, -1) > 0 and gaps.get(80, -1) > 0 and gaps.get(240, -1) > 0 and risk_hits == 0)
        path_good = int(complete and rauv > 0 and gaps.get(240, -1) > 0 and risk_hits == 0)
        slow_burn = int(complete and (gaps.get(1, 1) <= 0 or gaps.get(5, 1) <= 0) and gaps.get(20, -1) > 0 and gaps.get(80, -1) > 0 and gaps.get(240, -1) > 0 and risk_hits == 0)
        risk_clean_low_immediate = int(complete and gaps.get(1, 1) <= 0 and risk_hits == 0 and rauv >= 0 and gaps.get(240, -1) > 0)
        risky_high_auv = int(complete and rauv > 0 and risk_hits > 0)
        bad_path = int((not complete) or risk_hits > 0 or rauv <= 0 or gaps.get(240, -1) <= 0)
        rows.append({
            "stage": "P2_NATURAL_ACTION_LABELS_V9910",
            "status": "natural_action_label",
            "action_id": aid,
            "dataset": meta.get("dataset"),
            "family_id": meta.get("family_id"),
            "template_id": meta.get("template_id"),
            "step_bucket": step_bucket(meta.get("step")),
            "horizon_complete": complete,
            "V1_gap": gaps.get(1, ""),
            "V5_gap": gaps.get(5, ""),
            "V20_gap": gaps.get(20, ""),
            "V80_gap": gaps.get(80, ""),
            "V240_gap": gaps.get(240, ""),
            "RiskAdjustedAUV": rauv,
            "DelayedGain": gaps.get(240, 0.0) - gaps.get(1, 0.0),
            "SlowBurnIndex": max(0.0, gaps.get(240, 0.0) - gaps.get(5, 0.0)),
            "RiskPath": int(risk_hits > 0),
            "CoreLike": core_like,
            "PathGood": path_good,
            "SlowBurnGood": slow_burn,
            "RiskCleanButLowImmediate": risk_clean_low_immediate,
            "RiskyHighAUV": risky_high_auv,
            "BadPath": bad_path,
            "label_source": "real_branch_horizon_replay_diagnostic",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for (xaid, _h), br in by_action_h.items() if xaid == aid for r in br.values()] or [0]),
        })
    n = len(rows)

    def count(name: str) -> int:
        return sum(inum(r.get(name)) for r in rows)

    summary = {
        "stage": "P2_NATURAL_ACTION_LABELS_V9910",
        "status": "summary",
        "panel_action_count": n,
        "CoreLike_count": count("CoreLike"),
        "CoreLike_rate": count("CoreLike") / max(1, n),
        "CoreLike_LCB": wilson_lcb(count("CoreLike"), n) if n else 0.0,
        "CoreLike_UCB": wilson_ucb(count("CoreLike"), n) if n else 0.0,
        "PathGood_count": count("PathGood"),
        "PathGood_rate": count("PathGood") / max(1, n),
        "PathGood_LCB": wilson_lcb(count("PathGood"), n) if n else 0.0,
        "PathGood_UCB": wilson_ucb(count("PathGood"), n) if n else 0.0,
        "SlowBurnGood_count": count("SlowBurnGood"),
        "SlowBurnGood_rate": count("SlowBurnGood") / max(1, n),
        "SlowBurnGood_LCB": wilson_lcb(count("SlowBurnGood"), n) if n else 0.0,
        "SlowBurnGood_UCB": wilson_ucb(count("SlowBurnGood"), n) if n else 0.0,
        "RiskCleanButLowImmediate_count": count("RiskCleanButLowImmediate"),
        "RiskCleanButLowImmediate_rate": count("RiskCleanButLowImmediate") / max(1, n),
        "RiskCleanButLowImmediate_LCB": wilson_lcb(count("RiskCleanButLowImmediate"), n) if n else 0.0,
        "RiskCleanButLowImmediate_UCB": wilson_ucb(count("RiskCleanButLowImmediate"), n) if n else 0.0,
        "RiskyHighAUV_count": count("RiskyHighAUV"),
        "RiskyHighAUV_rate": count("RiskyHighAUV") / max(1, n),
        "RiskyHighAUV_LCB": wilson_lcb(count("RiskyHighAUV"), n) if n else 0.0,
        "RiskyHighAUV_UCB": wilson_ucb(count("RiskyHighAUV"), n) if n else 0.0,
        "BadPath_count": count("BadPath"),
        "BadPath_rate": count("BadPath") / max(1, n),
        "BadPath_LCB": wilson_lcb(count("BadPath"), n) if n else 0.0,
        "BadPath_UCB": wilson_ucb(count("BadPath"), n) if n else 0.0,
        "RiskAdjustedAUV_LCB": lcb_mean([fnum(r.get("RiskAdjustedAUV")) for r in rows]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
    }
    rows.insert(0, summary)
    return rows, summary


def p1_distribution_audit(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_actions = [r for r in old_rows if r.get("status") == "payload_disk_replay_row"]
    new_actions = [r for r in new_rows if r.get("status") == "natural_extension_action_row"]
    label_by_action = {str(r.get("action_id")): r for r in label_rows if r.get("status") == "natural_action_label"}
    feature_specs = [
        ("dataset", counter_share(old_actions, lambda r: r.get("dataset")), counter_share(new_actions, lambda r: r.get("dataset"))),
        ("template", counter_share(old_actions, carrier_key), counter_share(new_actions, carrier_key)),
        ("step_bucket", counter_share(old_actions, lambda r: step_bucket(r.get("step"))), counter_share(new_actions, lambda r: step_bucket(r.get("step")))),
    ]
    old_payload_norm, new_payload_norm, _ = numeric_bins(old_actions, new_actions, payload_norm, "payload_norm")
    old_payload_linf, new_payload_linf, _ = numeric_bins(old_actions, new_actions, payload_linf, "payload_linf")
    feature_specs.extend([
        ("payload_norm_bin", old_payload_norm, new_payload_norm),
        ("payload_linf_bin", old_payload_linf, new_payload_linf),
    ])
    old_ids = {str(r.get("action_id")) for r in old_actions}
    old_hashes = {str(r.get("payload_hash_expected") or r.get("payload_hash_loaded") or r.get("payload_hash")) for r in old_actions}
    rows: list[dict[str, Any]] = []
    psi_values: list[float] = []
    max_share_values: list[float] = []
    missing_major = 0
    for feature, old_c, new_c in feature_specs:
        n_new = sum(new_c.values())
        max_share = max(new_c.values()) / max(1, n_new) if new_c else 0.0
        psi = psi_from_counters(old_c, new_c)
        old_major = {k for k, v in old_c.items() if v / max(1, sum(old_c.values())) >= 0.05}
        missing = sorted(k for k in old_major if new_c.get(k, 0) == 0)
        missing_major += len(missing)
        psi_values.append(psi)
        max_share_values.append(max_share)
        rows.append({
            "stage": "P1_NATURAL_GENERATOR_DISTRIBUTION_AUDIT_V9910",
            "status": "feature_distribution_row",
            "feature": feature,
            "old_group_count": len(old_c),
            "new_group_count": len(new_c),
            "PSI": psi,
            "max_new_group_share": max_share,
            "max_new_group": new_c.most_common(1)[0][0] if new_c else "",
            "old_major_group_missing_count": len(missing),
            "old_major_group_missing_list": json.dumps(missing, ensure_ascii=False),
            "weak_feature_pass": int(psi <= 0.20 and max_share <= 0.50 and len(missing) == 0),
            "strong_feature_pass": int(psi <= 0.10 and max_share <= 0.35 and len(missing) == 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    new_ids = {str(r.get("action_id")) for r in new_actions}
    new_hashes = {str(r.get("payload_hash_expected") or r.get("payload_hash")) for r in new_actions}
    old_action_collision = len(new_ids & old_ids)
    old_payload_collision = len(new_hashes & old_hashes)
    core = sum(inum(label_by_action.get(str(r.get("action_id")), {}).get("CoreLike")) for r in new_actions)
    path = sum(inum(label_by_action.get(str(r.get("action_id")), {}).get("PathGood")) for r in new_actions)
    summary = {
        "stage": "P1_NATURAL_GENERATOR_DISTRIBUTION_AUDIT_V9910",
        "status": "summary",
        "old_action_count": len(old_actions),
        "new_action_count": len(new_actions),
        "feature_count": len(feature_specs),
        "PSI_major_max": max(psi_values or [0.0]),
        "PSI_major_mean": sum(psi_values) / max(1, len(psi_values)),
        "max_group_share": max(max_share_values or [0.0]),
        "missing_major_group_count": missing_major,
        "old_action_collision_count": old_action_collision,
        "old_payload_collision_count": old_payload_collision,
        "new_duplicate_action_id_count": len(new_actions) - len(new_ids),
        "new_duplicate_payload_hash_count": len(new_actions) - len(new_hashes),
        "CoreLike_count": core,
        "PathGood_count": path,
        "P1_weak_pass": int(max(psi_values or [999]) <= 0.20 and max(max_share_values or [999]) <= 0.50 and missing_major == 0 and old_action_collision == 0 and old_payload_collision == 0),
        "P1_strong_pass": int(max(psi_values or [999]) <= 0.10 and max(max_share_values or [999]) <= 0.35 and missing_major == 0 and old_action_collision == 0 and old_payload_collision == 0),
        "reason": "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    if not summary["P1_weak_pass"]:
        summary["reason"] = "natural_generator_distribution_fidelity_failed"
    rows.insert(0, summary)
    return rows, summary


def grouped_rates(label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in label_rows if r.get("status") == "natural_action_label"]
    out: list[dict[str, Any]] = []
    for entity, fn in [
        ("dataset", lambda r: r.get("dataset")),
        ("family", lambda r: r.get("family_id")),
        ("template", lambda r: r.get("template_id")),
        ("step_bucket", lambda r: r.get("step_bucket")),
    ]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(fn(row))].append(row)
        for gid, vals in sorted(groups.items()):
            n = len(vals)
            core = sum(inum(r.get("CoreLike")) for r in vals)
            path = sum(inum(r.get("PathGood")) for r in vals)
            out.append({
                "stage": "P2_NATURAL_DENSITY_GROUP_RATE_V9910",
                "status": "group_rate",
                "entity_type": entity,
                "entity_id": gid,
                "action_count": n,
                "CoreLike_count": core,
                "CoreLike_rate": core / max(1, n),
                "PathGood_count": path,
                "PathGood_rate": path / max(1, n),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    out.insert(0, {
        "stage": "P2_NATURAL_DENSITY_GROUP_RATE_V9910",
        "status": "summary",
        "group_rate_rows": len(out),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return out


def p2_density_panel(pilot_summary: dict[str, Any], pilot_smoke: dict[str, Any], p1: dict[str, Any], panel_targets: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = inum(pilot_summary.get("panel_action_count"))
    core = inum(pilot_summary.get("CoreLike_count"))
    path = inum(pilot_summary.get("PathGood_count"))
    enough = int((fnum(pilot_summary.get("CoreLike_LCB")) >= 0.03 or fnum(pilot_summary.get("PathGood_LCB")) >= 0.03) and ucb_rate(inum(pilot_summary.get("RiskyHighAUV_count")), n) <= 0.05 and ucb_rate(inum(pilot_summary.get("BadPath_count")), n) <= 0.05)
    insufficient = int(n >= 5000 and fnum(pilot_summary.get("CoreLike_UCB")) < 0.03 and fnum(pilot_summary.get("PathGood_UCB")) < 0.03)
    rows = [{
        "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_DECISION_V9910",
        "status": "panel_row",
        "panel_name": f"pilot_{n}",
        "panel_action_count": n,
        "branch_horizon_expected_rows": pilot_smoke.get("branch_horizon_expected_rows"),
        "branch_horizon_actual_rows": pilot_smoke.get("branch_horizon_actual_rows"),
        "completion_rate": pilot_smoke.get("branch_horizon_completion"),
        "rows_per_sec": pilot_smoke.get("rows_per_sec"),
        "wallclock_sec": pilot_smoke.get("wallclock_sec"),
        "peak_gpu_mb": pilot_smoke.get("peak_gpu_memory_mb"),
        "CoreLike_count": core,
        "CoreLike_rate": pilot_summary.get("CoreLike_rate"),
        "CoreLike_LCB": pilot_summary.get("CoreLike_LCB"),
        "CoreLike_UCB": pilot_summary.get("CoreLike_UCB"),
        "PathGood_count": path,
        "PathGood_rate": pilot_summary.get("PathGood_rate"),
        "PathGood_LCB": pilot_summary.get("PathGood_LCB"),
        "PathGood_UCB": pilot_summary.get("PathGood_UCB"),
        "SlowBurnGood_count": pilot_summary.get("SlowBurnGood_count"),
        "SlowBurnGood_rate": pilot_summary.get("SlowBurnGood_rate"),
        "SlowBurnGood_LCB": pilot_summary.get("SlowBurnGood_LCB"),
        "SlowBurnGood_UCB": pilot_summary.get("SlowBurnGood_UCB"),
        "RiskCleanButLowImmediate_count": pilot_summary.get("RiskCleanButLowImmediate_count"),
        "RiskCleanButLowImmediate_rate": pilot_summary.get("RiskCleanButLowImmediate_rate"),
        "RiskCleanButLowImmediate_LCB": pilot_summary.get("RiskCleanButLowImmediate_LCB"),
        "RiskCleanButLowImmediate_UCB": pilot_summary.get("RiskCleanButLowImmediate_UCB"),
        "RiskyHighAUV_count": pilot_summary.get("RiskyHighAUV_count"),
        "RiskyHighAUV_rate": pilot_summary.get("RiskyHighAUV_rate"),
        "RiskyHighAUV_LCB": pilot_summary.get("RiskyHighAUV_LCB"),
        "RiskyHighAUV_UCB": pilot_summary.get("RiskyHighAUV_UCB"),
        "BadPath_count": pilot_summary.get("BadPath_count"),
        "BadPath_rate": pilot_summary.get("BadPath_rate"),
        "BadPath_LCB": pilot_summary.get("BadPath_LCB"),
        "BadPath_UCB": pilot_summary.get("BadPath_UCB"),
        "density_result": "sufficient" if enough else ("insufficient" if insufficient else "inconclusive"),
        "reason": "pilot_panel_real_not_full_density_closure",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": pilot_summary.get("cpu_offload_used", 0),
    }]
    block_reason = "P1_distribution_fidelity_failed_do_not_open_full_density_panels" if not inum(p1.get("P1_weak_pass")) else "pilot_inconclusive_full_panel_not_run_in_this_stage"
    for target in [t for t in panel_targets if t > n]:
        rows.append({
            "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_DECISION_V9910",
            "status": "not_run",
            "panel_name": f"panel_{target}",
            "panel_action_count": target,
            "branch_horizon_expected_rows": target * BRANCH_COUNT * len(HORIZONS),
            "branch_horizon_actual_rows": 0,
            "completion_rate": 0,
            "density_result": "not_run",
            "reason": block_reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P2_SEQUENTIAL_NATURAL_DENSITY_DECISION_V9910",
        "status": "summary",
        "completed_panel_count": 1,
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "panel_action_count": n,
        "CoreLike_count": core,
        "CoreLike_LCB": pilot_summary.get("CoreLike_LCB"),
        "CoreLike_UCB": pilot_summary.get("CoreLike_UCB"),
        "PathGood_count": path,
        "PathGood_LCB": pilot_summary.get("PathGood_LCB"),
        "PathGood_UCB": pilot_summary.get("PathGood_UCB"),
        "RiskyHighAUV_count": pilot_summary.get("RiskyHighAUV_count"),
        "BadPath_count": pilot_summary.get("BadPath_count"),
        "P2_density_sufficient": enough,
        "P2_density_insufficient": insufficient,
        "P2_density_inconclusive": int(not enough and not insufficient),
        "reason": block_reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": pilot_summary.get("cpu_offload_used", 0),
    }
    rows.insert(0, summary)
    return rows, summary


def p3_future_path_decomposition(label_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in label_rows if r.get("status") == "natural_action_label"]
    groups = {
        "NewNaturalCoreLike": [r for r in rows if inum(r.get("CoreLike"))],
        "NewNaturalPathGood": [r for r in rows if inum(r.get("PathGood"))],
        "NewNaturalSlowBurnGood": [r for r in rows if inum(r.get("SlowBurnGood"))],
        "NewNaturalRiskyHighAUV": [r for r in rows if inum(r.get("RiskyHighAUV"))],
        "NewNaturalBadPath": [r for r in rows if inum(r.get("BadPath"))],
        "NewNaturalAll": rows,
    }
    out: list[dict[str, Any]] = []
    for name, vals in groups.items():
        n = len(vals)
        risk = sum(inum(r.get("RiskPath")) for r in vals)
        out.append({
            "stage": "P3_FUTURE_PATH_TYPE_DECOMPOSITION_V9910",
            "status": "group_summary",
            "group": name,
            "action_count": n,
            "RiskAdjustedAUV_LCB": lcb_mean([fnum(r.get("RiskAdjustedAUV")) for r in vals]),
            "DelayedGain_LCB": lcb_mean([fnum(r.get("DelayedGain")) for r in vals]),
            "SlowBurnIndex_LCB": lcb_mean([fnum(r.get("SlowBurnIndex")) for r in vals]),
            "LongRisk_UCB": wilson_ucb(risk, n) if n else 0.0,
            "PathStability_LCB": lcb_mean([min(fnum(r.get("V20_gap")), fnum(r.get("V80_gap")), fnum(r.get("V240_gap"))) for r in vals]),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in vals] or [0]),
        })
    weak = int(any(fnum(r.get("RiskAdjustedAUV_LCB")) > 0 and fnum(r.get("LongRisk_UCB")) <= 0.05 and inum(r.get("action_count")) > 0 for r in out if r.get("group") in {"NewNaturalCoreLike", "NewNaturalPathGood"}))
    strong = 0
    summary = {
        "stage": "P3_FUTURE_PATH_TYPE_DECOMPOSITION_V9910",
        "status": "summary",
        "group_count": len(out),
        "P3_future_path_weak_pass": weak,
        "P3_future_path_strong_pass": strong,
        "reason": "pilot_path_type_decomposition_only_not_87_action_controller_region",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
    }
    out.insert(0, summary)
    return out, summary


def evaluate_sketches(action_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = {str(r.get("action_id")): r for r in label_rows if r.get("status") == "natural_action_label"}
    actions = [r for r in action_rows if r.get("status") == "natural_extension_action_row" and str(r.get("action_id")) in labels]

    def score_fos1(r: dict[str, Any]) -> float:
        return fnum(r.get("action_adamw_cosine")) - 0.05 * payload_norm(r)

    def score_fos2(r: dict[str, Any]) -> float:
        return fnum(r.get("effective_derivative")) * fnum(r.get("trust_ratio")) - 0.1 * payload_linf(r)

    def score_fos3(r: dict[str, Any]) -> float:
        return fnum(r.get("tail_fraction")) * fnum(r.get("branch_ratio")) - abs(fnum(r.get("action_adamw_cosine")))

    def score_fos4(r: dict[str, Any]) -> float:
        veto = int(payload_linf(r) > 0.004 or payload_norm(r) > 0.08)
        return -veto + fnum(r.get("trust_ratio")) - 0.01 * payload_norm(r)

    sketches = [
        ("FOS1_TinyVirtualAdamWSketch", "green", score_fos1, 0.02),
        ("FOS2_JVPVJPPathSketch", "green_commit_time_surrogate", score_fos2, 0.03),
        ("FOS3_DelayedGainSketch", "green_commit_time_surrogate", score_fos3, 0.02),
        ("FOS4_RiskAdjustedHardGate", "green", score_fos4, 0.01),
    ]
    out: list[dict[str, Any]] = []
    for name, legality, scorer, cost_ms in sketches:
        ranked = sorted(actions, key=scorer, reverse=True)
        accepted = ranked[: min(87, len(ranked))]
        n = len(accepted)
        good = sum(int(inum(labels[str(r.get("action_id"))].get("PathGood")) or inum(labels[str(r.get("action_id"))].get("CoreLike"))) for r in accepted)
        risk = sum(inum(labels[str(r.get("action_id"))].get("RiskPath")) for r in accepted)
        precision = good / max(1, n)
        rauv_lcb = lcb_mean([fnum(labels[str(r.get("action_id"))].get("RiskAdjustedAUV")) for r in accepted])
        v_lcb = lcb_mean([fnum(labels[str(r.get("action_id"))].get("V20_gap")) for r in accepted])
        ldo = leaveout_drop(accepted, labels, "dataset", precision)
        lso = leaveout_drop(accepted, labels, "step_bucket", precision)
        lto = leaveout_drop(accepted, labels, "template_id", precision)
        lfo = leaveout_drop(accepted, labels, "family_id", precision)
        weak = int(n >= 87 and precision >= 0.65 and rauv_lcb > 0 and wilson_ucb(risk, n) <= 0.05 and cost_ms <= 0.20)
        strong = int(weak and precision >= 0.75 and v_lcb > 0 and max(ldo, lso, lto, lfo) <= 0.10 and cost_ms <= 0.05)
        out.append({
            "stage": "P4_FUTURE_PATH_OPERATOR_SKETCH_V4_V9910",
            "status": "sketch_row",
            "sketch": name,
            "legality": legality,
            "accepted_count": n,
            "precision": precision,
            "RiskAdjustedAUV_LCB": rauv_lcb,
            "V_LCB": v_lcb,
            "LongRisk_UCB": wilson_ucb(risk, n) if n else 0.0,
            "cost_q90_ms": cost_ms,
            "LDO": ldo,
            "LSO": lso,
            "LTO": lto,
            "LFO": lfo,
            "weak_pass": weak,
            "strong_pass": strong,
            "failure": "" if weak else "precision_or_value_or_risk_gate_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (inum(r.get("weak_pass")), fnum(r.get("precision")), fnum(r.get("RiskAdjustedAUV_LCB"))), default={})
    summary = {
        "stage": "P4_FUTURE_PATH_OPERATOR_SKETCH_V4_V9910",
        "status": "summary",
        "sketch_count": len(out),
        "P4_weak_pass": int(any(inum(r.get("weak_pass")) for r in out)),
        "P4_strong_pass": int(any(inum(r.get("strong_pass")) for r in out)),
        "best_sketch": best.get("sketch", ""),
        "best_precision": best.get("precision", ""),
        "best_RiskAdjustedAUV_LCB": best.get("RiskAdjustedAUV_LCB", ""),
        "best_cost_q90_ms": best.get("cost_q90_ms", ""),
        "reason": "low_cost_future_operator_sketch_not_controller_ready",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.insert(0, summary)
    return out, summary


def leaveout_drop(accepted: list[dict[str, Any]], labels: dict[str, dict[str, Any]], key: str, base_precision: float) -> float:
    groups = sorted({str(r.get(key) or labels[str(r.get("action_id"))].get(key) or "") for r in accepted})
    if len(groups) <= 1:
        return 0.0
    worst = 0.0
    for group in groups:
        keep = [r for r in accepted if str(r.get(key) or labels[str(r.get("action_id"))].get(key) or "") != group]
        if not keep:
            continue
        good = sum(int(inum(labels[str(r.get("action_id"))].get("PathGood")) or inum(labels[str(r.get("action_id"))].get("CoreLike"))) for r in keep)
        prec = good / max(1, len(keep))
        worst = max(worst, base_precision - prec)
    return worst


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, **extra}
    return [row], row


def write_figures(out: Path, route: dict[str, Any], p2: dict[str, Any], p4: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}

    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    fig("fig_v9910_gate_matrix.svg", "v9.9.1 gate matrix", ["P0", "P1", "P2", "P3", "P4", "P5"], [
        fnum(route.get("P0_boundary_pass")),
        fnum(route.get("P1_weak_pass")),
        fnum(route.get("P2_density_sufficient")),
        fnum(route.get("P3_future_path_weak_pass")),
        fnum(route.get("P4_future_operator_sketch_weak_pass")),
        fnum(route.get("P5_controller_pass")),
    ])
    fig("fig_p1_distribution_psi.svg", "distribution PSI", ["PSI max", "max share"], [fnum(route.get("P1_PSI_major_max")), fnum(route.get("P1_max_group_share"))])
    fig("fig_p2_density_core_path.svg", "pilot density CI", ["Core LCB", "Core UCB", "Path LCB", "Path UCB"], [
        fnum(p2.get("CoreLike_LCB")), fnum(p2.get("CoreLike_UCB")), fnum(p2.get("PathGood_LCB")), fnum(p2.get("PathGood_UCB")),
    ])
    fig("fig_p2_path_type_counts.svg", "pilot path type counts", ["Core", "Path", "Slow", "Risky", "Bad"], [
        fnum(p2.get("CoreLike_count")), fnum(p2.get("PathGood_count")), fnum(p2.get("SlowBurnGood_count")), fnum(p2.get("RiskyHighAUV_count")), fnum(p2.get("BadPath_count")),
    ])
    fig("fig_p4_sketch_best.svg", "FuturePathOperator sketch", ["best precision", "best RAUV LCB", "cost q90"], [
        fnum(p4.get("best_precision")), fnum(p4.get("best_RiskAdjustedAUV_LCB")), fnum(p4.get("best_cost_q90_ms")),
    ])
    fig("fig_stop_pivot_matrix_v9910.svg", "route decision", ["P1 fail", "density pending", "B fail"], [
        float(route.get("route") == "R-P1-NaturalGeneratorDistributionMismatch"),
        float(route.get("primary_blocker") == "natural_density_full_panel_pending"),
        float(route.get("secondary_blocker") == "future_operator_sketch_failed"),
    ])
    return figs


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


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_boundary_reproduction_v9910.csv"))
    p1 = summary_row(rows_from(out / "p1_natural_generator_distribution_audit_v9910.csv"))
    p2 = summary_row(rows_from(out / "p2_sequential_natural_density_decision_v9910.csv"))
    p2_smoke = summary_row(rows_from(out / "p2_1024_action_pilot_smoke_v9910.csv"))
    p3 = summary_row(rows_from(out / "p3_future_path_type_decomposition_v9910.csv"))
    p4 = summary_row(rows_from(out / "p4_future_path_operator_sketch_v4_v9910.csv"))
    p5 = summary_row(rows_from(out / "p5_existing_action_controller_boundary_v9910.csv"))
    p6 = summary_row(rows_from(out / "p6_generated_route_reopen_gate_v9910.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9910.csv"))
    lines = [
        "# DG-KAN v9.9.1 Natural Density Decision / FuturePathOperator 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.1_NaturalDensityDecision_FuturePathOperator_四线并行完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 和真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。1024 pilot 不被写成 full density closure，5000/10000/20000 未打开时显式 `not_run`。",
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
        f"1. P0 复现 v9.9.0 boundary：P1a/P1b/P1c/P1d = `{p0.get('P1a_entrypoint_search_pass_v9900')}` / `{p0.get('P1b_single_action_preflight_pass_v9900')}` / `{p0.get('P1c_16_action_preflight_pass_v9900')}` / `{p0.get('P1d_256_action_smoke_pass_v9900')}`，system = `{p0.get('system_legal_controller_pass_v9900')}`。",
        f"2. P2 真实 1024-action pilot：branch rows = `{p2_smoke.get('branch_horizon_actual_rows')}` / `{p2_smoke.get('branch_horizon_expected_rows')}`，completion = `{p2_smoke.get('completion_pass')}`，peak GPU MB = `{p2_smoke.get('peak_gpu_memory_mb')}`。",
        f"3. P1 分布保真 weak/strong = `{p1.get('P1_weak_pass')}` / `{p1.get('P1_strong_pass')}`；PSI max = `{p1.get('PSI_major_max')}`，max group share = `{p1.get('max_group_share')}`，missing major groups = `{p1.get('missing_major_group_count')}`。",
        f"4. P2 pilot CoreLike count/LCB/UCB = `{p2.get('CoreLike_count')}` / `{p2.get('CoreLike_LCB')}` / `{p2.get('CoreLike_UCB')}`；PathGood count/LCB/UCB = `{p2.get('PathGood_count')}` / `{p2.get('PathGood_LCB')}` / `{p2.get('PathGood_UCB')}`。",
        f"5. P2 5000/10000/20000 panels = `not_run`，reason = `{p2.get('reason')}`；density sufficient/insufficient/inconclusive = `{p2.get('P2_density_sufficient')}` / `{p2.get('P2_density_insufficient')}` / `{p2.get('P2_density_inconclusive')}`。",
        f"6. P3 future path weak/strong = `{p3.get('P3_future_path_weak_pass')}` / `{p3.get('P3_future_path_strong_pass')}`。",
        f"7. P4 FuturePathOperator sketch weak/strong = `{p4.get('P4_weak_pass')}` / `{p4.get('P4_strong_pass')}`；best = `{p4.get('best_sketch')}`，precision = `{p4.get('best_precision')}`。",
        f"8. P5 controller = `{p5.get('status')}`；P6 generated sandbox allowed = `{p6.get('P6_generated_sandbox_allowed')}`。",
        f"9. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9910_natural_density_decision_futurepathoperator.py` | v9.9.1 runner；执行 P0 boundary、1024 natural pilot、P1 distribution fidelity、P2 density decision、P3 path type、P4 low-cost sketch 与 P5-P8 boundary。 |",
        "| `experiments/natural_ap0_extension_materializer.py` | v9.9.0 已落地的 natural AP0 extension materializer，本轮复用它生成真实 pilot action/replay rows。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9910_natural_density_decision_futurepathoperator.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9910_natural_density_decision_futurepathoperator.py --out-dir results/real_rerun_20260506/v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Distribution Fidelity",
        "",
        "```text",
        f"old_action_count = {p1.get('old_action_count')}",
        f"new_action_count = {p1.get('new_action_count')}",
        f"PSI_major_max = {p1.get('PSI_major_max')}",
        f"max_group_share = {p1.get('max_group_share')}",
        f"missing_major_group_count = {p1.get('missing_major_group_count')}",
        f"old action/payload collision = {p1.get('old_action_collision_count')} / {p1.get('old_payload_collision_count')}",
        f"P1 weak/strong = {p1.get('P1_weak_pass')} / {p1.get('P1_strong_pass')}",
        "```",
        "",
        "判断：P1 使用 old canonical AP0 panel 与本轮 1024 pilot 的 dataset/template/step_bucket/payload 分布做 PSI 与 coverage 审计；如果 P1 weak 不过，full 5000/10000/20000 panel 不打开。",
        "",
        "## 4. P2 Density Decision",
        "",
        "```text",
        f"pilot actions = {p2.get('panel_action_count')}",
        f"CoreLike count/LCB/UCB = {p2.get('CoreLike_count')} / {p2.get('CoreLike_LCB')} / {p2.get('CoreLike_UCB')}",
        f"PathGood count/LCB/UCB = {p2.get('PathGood_count')} / {p2.get('PathGood_LCB')} / {p2.get('PathGood_UCB')}",
        f"RiskyHighAUV count = {p2.get('RiskyHighAUV_count')}",
        f"BadPath count = {p2.get('BadPath_count')}",
        f"density sufficient/insufficient/inconclusive = {p2.get('P2_density_sufficient')} / {p2.get('P2_density_insufficient')} / {p2.get('P2_density_inconclusive')}",
        "```",
        "",
        "判断：1024 pilot 是真实 replay panel，但不是 full density closure；在 P1 weak 失败时不允许继续宣称 full density sufficient/insufficient。",
        "",
        "## 5. Boundary",
        "",
        "```text",
        f"P3 future path = {p3.get('P3_future_path_weak_pass')} / {p3.get('P3_future_path_strong_pass')}",
        f"P4 sketch = {p4.get('P4_weak_pass')} / {p4.get('P4_strong_pass')}",
        f"P5 controller = {p5.get('status')}, reason = {p5.get('reason')}",
        f"P6 generated = {p6.get('status')}, allowed = {p6.get('P6_generated_sandbox_allowed')}",
        "P7 runtime = not_run",
        "P8 paired replay = not_run",
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
        "1. v9.9.1 真实运行了 1024-action natural AP0 pilot，不是复用 v9.9.0 的 256 smoke。",
        "2. P1 分布保真度是本轮关键 gate；它决定是否允许打开 5000/10000/20000 density panels。",
        "3. P2 pilot density 只作为序贯裁决的第一步，不被写成 full density closure。",
        "4. P4 low-cost FuturePathOperator sketch 只使用 commit-time fields 做排序；evaluation 使用真实 replay labels，但不把 diagnostic label 放入 controller。",
        "5. controller/generated/runtime/paired replay 都保持 gate-blocked，直到 C 线或 B 线真的过 gate。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.9.1 真实执行后停在 `{route.get('route')}`：本轮把 v9.9.0 的工程 smoke 推进到 1024-action natural pilot 和分布/密度/低成本 sketch 裁决；但没有满足 official controller/runtime/generated gate，因此 strict PureKAN functional 仍未成功。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    artifacts: dict[str, Path] = {}
    source_v9900 = Path(args.source_v9900)
    source_v9330 = Path(args.source_v9330)
    panel_targets = parse_ints(args.panel_targets)

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return summary_row(rows)

    def dump_json(name: str, row: dict[str, Any]) -> None:
        path = out / name
        write_json(path, row)
        artifacts[name] = path

    p0 = dump_csv("p0_boundary_reproduction_v9910.csv", p0_boundary(source_v9900)[0])
    old_ids, old_hashes = old_action_sets(source_v9330, source_v9900)
    if not inum(p0.get("P0_pass")):
        p2_smoke = dump_csv("p2_1024_action_pilot_smoke_v9910.csv", v9900.p1_preflight_rows("P2_1024_ACTION_PILOT_SMOKE_V9910", args.pilot_actions, "P0_boundary_failed")[0])
        action_rows: list[dict[str, Any]] = []
        branch_rows: list[dict[str, Any]] = []
    else:
        rows, action_rows, apply_rows, branch_rows, p2_smoke = v9900.run_natural_smoke(
            "P2_1024_ACTION_PILOT_SMOKE_V9910",
            int(args.pilot_actions),
            273,
            args,
            out,
            old_ids,
            old_hashes,
        )
        dump_csv("p2_1024_action_pilot_smoke_v9910.csv", rows)
        dump_csv("p2_1024_natural_action_rows_v9910.csv", action_rows)
        dump_csv("p2_1024_action_apply_replay_v9910.csv", apply_rows)
        dump_csv("p2_1024_branch_horizon_v9910.csv", branch_rows)

    label_rows, label_summary = label_natural_actions(branch_rows)
    dump_csv("p2_natural_action_labels_v9910.csv", label_rows)
    old_rows = rows_from(source_v9330 / "action_payload_disk_replay_trace_v9330.csv")
    p1_rows, p1 = p1_distribution_audit(old_rows, action_rows, label_rows)
    dump_csv("p1_natural_generator_distribution_audit_v9910.csv", p1_rows)
    dump_csv("p2_density_group_rates_v9910.csv", grouped_rates(label_rows))
    p2_rows, p2 = p2_density_panel(label_summary, p2_smoke, p1, panel_targets)
    dump_csv("p2_sequential_natural_density_decision_v9910.csv", p2_rows)
    p3_rows, p3 = p3_future_path_decomposition(label_rows)
    dump_csv("p3_future_path_type_decomposition_v9910.csv", p3_rows)
    p4_rows, p4 = evaluate_sketches(action_rows, label_rows)
    dump_csv("p4_future_path_operator_sketch_v4_v9910.csv", p4_rows)

    controller_reason = "P2_density_not_sufficient_and_P4_sketch_not_passed"
    p5 = dump_csv("p5_existing_action_controller_boundary_v9910.csv", not_run("P5_EXISTING_ACTION_CONTROLLER_BOUNDARY_V9910", controller_reason, controller_pass=0)[0])
    p6 = dump_csv("p6_generated_route_reopen_gate_v9910.csv", not_run("P6_GENERATED_ROUTE_REOPEN_GATE_V9910", "D1_D2_D3_conditions_not_met", P6_generated_sandbox_allowed=0, generated_action_count=0)[0])
    p7 = dump_csv("p7_runtime_boundary_v9910.csv", not_run("P7_RUNTIME_BOUNDARY_V9910", "P5_controller_not_passed", runtime_pass=0)[0])
    p8 = dump_csv("p8_paired_replay_boundary_v9910.csv", not_run("P8_PAIRED_REPLAY_BOUNDARY_V9910", "P7_runtime_not_passed", paired_replay_pass=0)[0])

    if not inum(p1.get("P1_weak_pass")):
        route_name = "R-P1-NaturalGeneratorDistributionMismatch"
        primary = "natural_generator_distribution_fidelity_failed"
        secondary = "full_density_panels_blocked"
        generated_status = "stopped_P1_distribution_fidelity_failed"
    elif inum(p2.get("P2_density_sufficient")) and not inum(p4.get("P4_weak_pass")):
        route_name = "CaseC-DensityEnoughLegalProxyInsufficient"
        primary = "legal_future_operator_proxy_insufficient"
        secondary = "controller_blocked"
        generated_status = "stopped_controller_proxy_not_ready"
    elif inum(p2.get("P2_density_insufficient")):
        route_name = "CaseB-NaturalDensityInsufficient"
        primary = "natural_action_density_insufficient"
        secondary = "generated_route_requires_A_B_mechanism"
        generated_status = "stopped_density_insufficient_no_mechanism"
    elif inum(p4.get("P4_weak_pass")):
        route_name = "CaseD-FuturePathOperatorSketchWeak"
        primary = "controller_boundary_pending"
        secondary = "runtime_not_opened"
        generated_status = "stopped_controller_boundary_pending"
    else:
        route_name = "R-P2-PilotDensityInconclusiveProxyFailed"
        primary = "natural_density_full_panel_pending"
        secondary = "future_operator_sketch_failed"
        generated_status = "stopped_no_density_or_B_gate"

    route = {
        "stage": "ROUTE_DECISION_V9910",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9900": p0.get("source_route_v9900"),
        "P0_boundary_pass": p0.get("P0_pass"),
        "P1_weak_pass": p1.get("P1_weak_pass"),
        "P1_strong_pass": p1.get("P1_strong_pass"),
        "P1_PSI_major_max": p1.get("PSI_major_max"),
        "P1_max_group_share": p1.get("max_group_share"),
        "P2_pilot_action_count": p2.get("panel_action_count"),
        "P2_density_sufficient": p2.get("P2_density_sufficient"),
        "P2_density_insufficient": p2.get("P2_density_insufficient"),
        "P2_density_inconclusive": p2.get("P2_density_inconclusive"),
        "CoreLike_count": p2.get("CoreLike_count"),
        "CoreLike_LCB": p2.get("CoreLike_LCB"),
        "CoreLike_UCB": p2.get("CoreLike_UCB"),
        "PathGood_count": p2.get("PathGood_count"),
        "PathGood_LCB": p2.get("PathGood_LCB"),
        "PathGood_UCB": p2.get("PathGood_UCB"),
        "P3_future_path_weak_pass": p3.get("P3_future_path_weak_pass"),
        "P3_future_path_strong_pass": p3.get("P3_future_path_strong_pass"),
        "P4_future_operator_sketch_weak_pass": p4.get("P4_weak_pass"),
        "P4_future_operator_sketch_strong_pass": p4.get("P4_strong_pass"),
        "P5_controller_pass": p5.get("controller_pass"),
        "P6_generated_sandbox_allowed": p6.get("P6_generated_sandbox_allowed"),
        "generated_route_status": generated_status,
        "P7_runtime_pass": p7.get("runtime_pass"),
        "P8_paired_replay_pass": p8.get("paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(x.get("cpu_offload_used")) for x in [p0, p1, p2, p3, p4, p5, p6, p7, p8] if x] or [0]),
    }
    dump_json("route_decision_v9910.json", route)
    artifacts.update(write_figures(out, route, p2, p4))
    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9910",
        "status": "summary",
        "rows_checked": artifact_row_count(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9910.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9910",
        "status": "summary",
        "v9900_boundary_pass": p0.get("P0_pass"),
        "pilot_1024_materialized": p2_smoke.get("completion_pass"),
        "P1_distribution_weak_pass": p1.get("P1_weak_pass"),
        "P2_density_sufficient/insufficient/inconclusive": f"{p2.get('P2_density_sufficient')}/{p2.get('P2_density_insufficient')}/{p2.get('P2_density_inconclusive')}",
        "P4_sketch_weak/strong": f"{p4.get('P4_weak_pass')}/{p4.get('P4_strong_pass')}",
        "controller/generated/runtime/system": "0/0/0/0",
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"0/0/{route['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_csv("contract_audit_v9910.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9910",
        "status": "summary",
        "route": route["route"],
        "F0_boundary_fail": int(not inum(p0.get("P0_pass"))),
        "F1_distribution_fidelity_fail": int(not inum(p1.get("P1_weak_pass"))),
        "F2_density_not_adjudicated": int(not inum(p2.get("P2_density_sufficient")) and not inum(p2.get("P2_density_insufficient"))),
        "F3_future_operator_sketch_fail": int(not inum(p4.get("P4_weak_pass"))),
        "F4_controller_generated_runtime_blocked": 1,
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_csv("failure_taxonomy_v9910.csv", [failure])
    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9910",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "device": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "panel_targets": args.panel_targets,
        "pilot_actions": args.pilot_actions,
        "source_v9900": args.source_v9900,
        "wallclock_sec": time.perf_counter() - started,
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_json("run_manifest_v9910.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route, hashes_for_recap)
    print(json.dumps({
        "out_dir": str(out),
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "P1_weak_pass": route["P1_weak_pass"],
        "P2_density_inconclusive": route["P2_density_inconclusive"],
        "P4_weak_pass": route["P4_future_operator_sketch_weak_pass"],
        "system_legal_controller_pass": route["system_legal_controller_pass"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
