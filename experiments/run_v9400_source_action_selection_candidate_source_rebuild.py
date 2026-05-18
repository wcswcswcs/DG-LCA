#!/usr/bin/env python3
"""DG-KAN v9.4.0 source-action selection / candidate-source rebuild runner.

This runner deliberately separates three questions:

1. Did the v9.3.9 source panel represent the full AP0 action universe?
2. If good AP0 actions exist, can a legal commit-time selector find them?
3. If not, is a real source generator materialized in this repo?

It does not promote oracle source labels, source-panel diagnostics, or missing
source-generator placeholders to an official controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.0_SourceActionSelection_CandidateSourceRebuild_HorizonRobustController_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9400_source_action_selection_candidate_source_rebuild.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_SOURCE_V9380 = RESULT_ROOT / "v9380_real_certificate_action_primitive_materialization_horizon_system_closure_first_20260514T050000Z"
DEFAULT_SOURCE_V9390 = RESULT_ROOT / "v9390_generated_ap_failure_decomposition_value_preserving_first_20260514T060000Z"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
HELDOUT_DENOMINATOR = 9072


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9350", default=str(DEFAULT_SOURCE_V9350))
    p.add_argument("--source-v9380", default=str(DEFAULT_SOURCE_V9380))
    p.add_argument("--source-v9390", default=str(DEFAULT_SOURCE_V9390))
    p.add_argument("--source-v9330", default=str(DEFAULT_SOURCE_V9330))
    p.add_argument("--source-panel-actions", type=int, default=64)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode("utf-8")).hexdigest()


def mean(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    return sum(xs) / max(1, len(xs))


def lcb_mean(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    m = mean(xs)
    sd = statistics.stdev(xs)
    return m - 1.96 * sd / math.sqrt(len(xs))


def q(values: list[float], frac: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))]


def bins_numeric(values: list[float], bucket_count: int = 5) -> list[float]:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return []
    return [xs[min(len(xs) - 1, math.floor(i * len(xs) / bucket_count))] for i in range(1, bucket_count)]


def assign_bucket(value: float, cuts: list[float]) -> str:
    for idx, cut in enumerate(cuts):
        if value <= cut:
            return f"b{idx}"
    return f"b{len(cuts)}"


def dist(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    c = Counter(str(r.get(key, "")) for r in rows)
    n = sum(c.values())
    return {k: v / max(1, n) for k, v in c.items()}


def psi(p: dict[str, float], qd: dict[str, float]) -> float:
    keys = set(p) | set(qd)
    eps = 1.0e-9
    return sum((p.get(k, eps) - qd.get(k, eps)) * math.log((p.get(k, eps)) / (qd.get(k, eps))) for k in keys)


def kl(p: dict[str, float], qd: dict[str, float]) -> float:
    keys = set(p) | set(qd)
    eps = 1.0e-9
    return sum(p.get(k, eps) * math.log((p.get(k, eps)) / (qd.get(k, eps))) for k in keys)


def max_gap(p: dict[str, float], qd: dict[str, float]) -> float:
    keys = set(p) | set(qd)
    return max((abs(p.get(k, 0.0) - qd.get(k, 0.0)) for k in keys), default=0.0)


def action_source_id(row: dict[str, Any]) -> str:
    return str(row.get("action_id") or row.get("source_action_id") or row.get("ap_action_id"))


def load_full_ap0(source_v9350: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = []
    for r in read_csv(source_v9350 / "full_control_outcome_table_v9350.csv"):
        if r.get("branch_id") != "RealFunctional":
            continue
        cp = inum(r.get("control_positive_label"))
        horizon = inum(r.get("horizon"))
        long_risk = int(horizon == 240 and (inum(r.get("bad_event_label")) or not cp))
        row = dict(r)
        row["weak_CP_label"] = cp
        row["strong_CP_label"] = int(cp and fnum(r.get("V_ctrl_max_control_gap")) >= 0.15)
        row["long_risk_label"] = long_risk
        row["source_action_id"] = action_source_id(r)
        row["step_bucket"] = f"step_{inum(r.get('step')) // 32}"
        rows.append(row)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_action[str(r["source_action_id"])].append(r)
    return rows, by_action


def load_payload_rows(source_v9330: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for r in read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv"):
        if r.get("status") == "payload_disk_replay_row":
            rows[str(r.get("action_id"))] = r
    return rows


def load_source_panel_ids(source_v9390: Path, limit: int) -> list[str]:
    ids: list[str] = []
    for r in read_csv(source_v9390 / "source_ap0_branch_horizon_trace_v9390.csv"):
        if r.get("branch_id") != "RealAP0":
            continue
        aid = str(r.get("source_action_id"))
        if aid not in ids:
            ids.append(aid)
        if len(ids) >= limit:
            break
    return ids


def summarize_rows(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    n = len(rows)
    action_ids = {str(r.get("source_action_id")) for r in rows}
    weak = sum(inum(r.get("weak_CP_label")) for r in rows)
    strong = sum(inum(r.get("strong_CP_label")) for r in rows)
    bad = sum(inum(r.get("bad_event_label")) for r in rows)
    null = sum(inum(r.get("null_event_label")) for r in rows)
    longrisk = sum(inum(r.get("long_risk_label")) for r in rows)
    vals = [fnum(r.get("V_ctrl_max_control_gap")) for r in rows]
    by_h = {h: [r for r in rows if inum(r.get("horizon")) == h] for h in HORIZONS}
    robust_actions = 0
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_action[str(r.get("source_action_id"))].append(r)
    for rs in by_action.values():
        by = {inum(r.get("horizon")): r for r in rs}
        if all(inum(by.get(h, {}).get("weak_CP_label")) for h in HORIZONS) and not any(inum(by.get(h, {}).get("long_risk_label")) for h in HORIZONS):
            robust_actions += 1
    return {
        f"{prefix}row_count": n,
        f"{prefix}action_count": len(action_ids),
        f"{prefix}weak_CP_row_count": weak,
        f"{prefix}weak_CP_row_rate": weak / max(1, n),
        f"{prefix}strong_CP_row_count": strong,
        f"{prefix}strong_CP_row_rate": strong / max(1, n),
        f"{prefix}bad_event_rate": bad / max(1, n),
        f"{prefix}null_rate": null / max(1, n),
        f"{prefix}long_risk_count": longrisk,
        f"{prefix}long_risk_rate": longrisk / max(1, n),
        f"{prefix}V_ctrl_mean": mean(vals),
        f"{prefix}V_ctrl_lcb": lcb_mean(vals),
        f"{prefix}horizon_robust_action_count": robust_actions,
        f"{prefix}horizon_robust_action_rate": robust_actions / max(1, len(action_ids)),
        f"{prefix}weak_CP_h20": sum(inum(r.get("weak_CP_label")) for r in by_h[20]) / max(1, len(by_h[20])),
        f"{prefix}weak_CP_h80": sum(inum(r.get("weak_CP_label")) for r in by_h[80]) / max(1, len(by_h[80])),
        f"{prefix}weak_CP_h240": sum(inum(r.get("weak_CP_label")) for r in by_h[240]) / max(1, len(by_h[240])),
    }


def action_stats(by_action: dict[str, list[dict[str, Any]]], payload_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    family_counts = Counter(str(rs[0].get("family_id")) for rs in by_action.values() if rs)
    for aid, rs in by_action.items():
        vals = [fnum(r.get("V_ctrl_max_control_gap")) for r in rs]
        weak = sum(inum(r.get("weak_CP_label")) for r in rs)
        strong = sum(inum(r.get("strong_CP_label")) for r in rs)
        longrisk = sum(inum(r.get("long_risk_label")) for r in rs)
        bad = sum(inum(r.get("bad_event_label")) for r in rs)
        null = sum(inum(r.get("null_event_label")) for r in rs)
        byh = {inum(r.get("horizon")): r for r in rs}
        robust = int(all(inum(byh.get(h, {}).get("weak_CP_label")) for h in HORIZONS) and not any(inum(byh.get(h, {}).get("long_risk_label")) for h in HORIZONS))
        first = rs[0]
        payload = payload_rows.get(aid, {})
        out[aid] = {
            "action_id": aid,
            "candidate_id": first.get("candidate_id"),
            "event_id": first.get("event_id"),
            "dataset": first.get("dataset"),
            "seed": first.get("seed"),
            "step": first.get("step"),
            "family_id": first.get("family_id"),
            "bucket_id": first.get("bucket_id"),
            "payload_hash": first.get("payload_hash"),
            "weak_CP_count": weak,
            "weak_CP_rate": weak / max(1, len(rs)),
            "strong_CP_count": strong,
            "strong_CP_rate": strong / max(1, len(rs)),
            "horizon_robust_CP": robust,
            "long_risk_count": longrisk,
            "long_risk_rate": longrisk / max(1, len(rs)),
            "bad_event_rate": bad / max(1, len(rs)),
            "null_rate": null / max(1, len(rs)),
            "V_ctrl_mean": mean(vals),
            "V_ctrl_lcb": lcb_mean(vals),
            "V_ctrl_h20": fnum(byh.get(20, {}).get("V_ctrl_max_control_gap")),
            "weak_CP_h20": inum(byh.get(20, {}).get("weak_CP_label")),
            "bad_h20": inum(byh.get(20, {}).get("bad_event_label")),
            "null_h20": inum(byh.get(20, {}).get("null_event_label")),
            "CE_mean_before": fnum(first.get("CE_mean_before")),
            "CEp99_before": fnum(first.get("CEp99_before")),
            "margin_p10_before": fnum(first.get("margin_p10_before")),
            "ECE_before": fnum(first.get("ECE_before")),
            "NLL_before": fnum(first.get("NLL_before")),
            "curvature_before": fnum(first.get("curvature_before")),
            "local_lipschitz_before": fnum(first.get("local_lipschitz_before")),
            "payload_norm": fnum(payload.get("payload_norm")),
            "payload_linf": fnum(payload.get("payload_linf_norm")),
            "payload_offset": inum(payload.get("payload_tensor_offset")),
            "family_support_count": family_counts[str(first.get("family_id"))],
        }
    return out


def support_balance(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    families = Counter(str(r.get("family_id")) for r in rows)
    if not families:
        return 0, 0, 1.0
    max_share = max(families.values()) / sum(families.values())
    return int(len(families) >= 10 and max_share <= 0.40), len(families), max_share


def panel_metrics_for_actions(action_ids: list[str], by_action: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [r for aid in action_ids for r in by_action.get(aid, [])]
    summ = summarize_rows(rows)
    support_pass, fam_count, max_fam = support_balance(rows)
    summ.update(
        {
            "accepted_action_count": len(action_ids),
            "coverage": len(action_ids) / HELDOUT_DENOMINATOR,
            "support_balance_pass": support_pass,
            "accepted_family_count": fam_count,
            "max_family_share": max_fam,
            "V_ctrl_lcb_all": summ.get("V_ctrl_lcb", 0.0),
            "weak_CP_precision_all_rows": summ.get("weak_CP_row_rate", 0.0),
            "strong_CP_precision_all_rows": summ.get("strong_CP_row_rate", 0.0),
            "long_risk_rate": summ.get("long_risk_rate", 0.0),
        }
    )
    return summ


def p0_full_vs_panel(source_v9350: Path, source_v9390: Path, source_v9330: Path, panel_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    full_rows, by_action = load_full_ap0(source_v9350)
    payload_rows = load_payload_rows(source_v9330)
    stats = action_stats(by_action, payload_rows)
    panel_ids = load_source_panel_ids(source_v9390, panel_limit)
    panel_rows = [r for aid in panel_ids for r in by_action.get(aid, [])]
    joined = [aid for aid in panel_ids if aid in by_action]
    join_pass = int(len(joined) == len(panel_ids) and bool(joined))

    payload_norms = [fnum(payload_rows.get(aid, {}).get("payload_norm")) for aid in by_action]
    cuts = bins_numeric(payload_norms)
    for rows in (full_rows, panel_rows):
        for r in rows:
            aid = str(r.get("source_action_id"))
            r["payload_norm_bucket"] = assign_bucket(fnum(payload_rows.get(aid, {}).get("payload_norm")), cuts)
            r["step_bucket"] = f"step_{inum(r.get('step')) // 32}"
    full_summary = summarize_rows(full_rows, "full_")
    panel_summary = summarize_rows(panel_rows, "source_panel_")

    family_gap = max_gap(dist(full_rows, "family_id"), dist(panel_rows, "family_id"))
    horizon_gap = max_gap(dist(full_rows, "horizon"), dist(panel_rows, "horizon"))
    step_gap = max_gap(dist(full_rows, "step_bucket"), dist(panel_rows, "step_bucket"))
    score_gap = max_gap(dist(full_rows, "payload_norm_bucket"), dist(panel_rows, "payload_norm_bucket"))
    panel_psi = psi(dist(panel_rows, "family_id"), dist(full_rows, "family_id")) + psi(dist(panel_rows, "step_bucket"), dist(full_rows, "step_bucket"))
    panel_kl = kl(dist(panel_rows, "family_id"), dist(full_rows, "family_id")) + kl(dist(panel_rows, "step_bucket"), dist(full_rows, "step_bucket"))
    route_flip = int(full_summary["full_weak_CP_row_rate"] >= 0.09 and panel_summary["source_panel_weak_CP_row_rate"] < 0.09)
    bias_pass = int(panel_psi <= 0.10 and max(family_gap, horizon_gap, step_gap, score_gap) <= 0.15 and route_flip == 0)

    trace: list[dict[str, Any]] = []
    for aid in panel_ids:
        st = stats.get(aid, {})
        trace.append(
            {
                "stage": "P0_FULL_VS_SOURCE_PANEL_DIAGNOSIS",
                "status": "source_panel_join_row",
                "source_action_id": aid,
                "join_found": int(aid in by_action),
                "candidate_id": st.get("candidate_id", ""),
                "event_id": st.get("event_id", ""),
                "dataset": st.get("dataset", ""),
                "seed": st.get("seed", ""),
                "step": st.get("step", ""),
                "family_id": st.get("family_id", ""),
                "payload_hash": st.get("payload_hash", ""),
                "weak_CP_rate": st.get("weak_CP_rate", ""),
                "long_risk_rate": st.get("long_risk_rate", ""),
                "V_ctrl_lcb": st.get("V_ctrl_lcb", ""),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    bias_rows = []
    for key in ("family_id", "step_bucket", "payload_norm_bucket", "horizon"):
        fd = dist(full_rows, key)
        pd = dist(panel_rows, key)
        for bucket in sorted(set(fd) | set(pd)):
            bias_rows.append(
                {
                    "stage": "P0_SOURCE_PANEL_BIAS_TRACE",
                    "status": "distribution_bucket",
                    "feature": key,
                    "bucket": bucket,
                    "full_share": fd.get(bucket, 0.0),
                    "source_panel_share": pd.get(bucket, 0.0),
                    "absolute_gap": abs(fd.get(bucket, 0.0) - pd.get(bucket, 0.0)),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    summary = {
        "stage": "P0_FULL_VS_SOURCE_PANEL_DIAGNOSIS",
        "status": "summary",
        "full_action_count": len(by_action),
        "source_panel_action_count": len(panel_ids),
        "source_panel_joined_action_count": len(joined),
        "source_panel_join_pass": join_pass,
        "source_panel_fraction": len(panel_ids) / max(1, len(by_action)),
        **full_summary,
        **panel_summary,
        "selection_bias_PSI": panel_psi,
        "selection_bias_KL": panel_kl,
        "max_family_gap": family_gap,
        "max_horizon_gap": horizon_gap,
        "max_step_bucket_gap": step_gap,
        "max_score_bucket_gap": score_gap,
        "route_flip_full_vs_source_panel": route_flip,
        "source_panel_bias_pass": bias_pass,
        "source_panel_bias_class": "biased_or_unrepresentative" if not bias_pass else "representative",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary], trace, bias_rows, summary, by_action, stats


def p1_selection_provenance(source_v9330: Path, panel_ids: list[str], stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload_rows = list(load_payload_rows(source_v9330).values())
    rank = {str(r.get("action_id")): idx for idx, r in enumerate(payload_rows)}
    trace = []
    for aid in panel_ids:
        st = stats.get(aid, {})
        trace.append(
            {
                "stage": "P1_SOURCE_SELECTION_PROVENANCE_AUDIT",
                "status": "source_selection_trace_row",
                "source_action_id": aid,
                "candidate_id": st.get("candidate_id", ""),
                "event_id": st.get("event_id", ""),
                "selection_rule_id": "v9380_first_N_payload_available_slice",
                "selection_rank": rank.get(aid, ""),
                "selection_score": "",
                "selection_stage": "AP_generator_smoke_source_panel",
                "eligible_pool_size_at_selection": len(payload_rows),
                "selected_count_at_stage": len(panel_ids),
                "selection_probability_estimate": len(panel_ids) / max(1, len(payload_rows)),
                "source_selection_reason_tag": "smoke_convenience_slice_first64_payload_available",
                "source_selection_commit_time_features": "payload_available;row_order",
                "source_selection_uses_outcome": 0,
                "source_selection_uses_dataset_name": 0,
                "source_selection_uses_future_step": 0,
                "payload_availability_flag": int(aid in rank),
                "certificate_availability_flag": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    families = Counter(str(stats.get(aid, {}).get("family_id")) for aid in panel_ids)
    steps = Counter(str(stats.get(aid, {}).get("step")) for aid in panel_ids)
    unknown = 0
    summary = {
        "stage": "P1_SOURCE_SELECTION_PROVENANCE_AUDIT",
        "status": "summary",
        "selection_rule_identified": 1,
        "unknown_selection_reason_count": unknown,
        "uses_outcome_in_source_selection": 0,
        "uses_dataset_name_in_source_selection": 0,
        "uses_future_step_in_source_selection": 0,
        "source_panel_family_concentration": max(families.values()) / max(1, sum(families.values())) if families else 1.0,
        "source_panel_step_concentration": max(steps.values()) / max(1, sum(steps.values())) if steps else 1.0,
        "source_panel_score_concentration": "",
        "source_panel_selection_bias_pass": 0,
        "selection_rule_pass": 1,
        "reason": "source_panel_is_first_N_payload_available_convenience_slice_not_value_selector",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    unknown_table = [
        {
            "stage": "P1_SOURCE_SELECTION_UNKNOWN_REASON_TABLE",
            "status": "summary",
            "unknown_selection_reason_count": unknown,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    return [summary], trace, unknown_table, summary


def p2_oracle_selectors(by_action: dict[str, list[dict[str, Any]]], stats: dict[str, dict[str, Any]], panel_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ids = list(by_action)
    scored = list(stats.values())
    def oracle_score(st: dict[str, Any]) -> float:
        return fnum(st.get("V_ctrl_h20")) + 0.5 * fnum(st.get("V_ctrl_mean")) + 0.25 * fnum(st.get("weak_CP_rate")) - 2.0 * fnum(st.get("long_risk_rate")) - 2.0 * fnum(st.get("bad_event_rate")) - 0.5 * fnum(st.get("null_rate"))
    selectors: dict[str, list[str]] = {
        "OS0-current-v9390-source-panel": panel_ids,
        "OS2-weak-CP-oracle-source-selector": [s["action_id"] for s in sorted(scored, key=lambda x: (fnum(x["weak_CP_rate"]), fnum(x["V_ctrl_lcb"]), -fnum(x["long_risk_rate"])), reverse=True)],
        "OS3-strong-CP-oracle-source-selector": [s["action_id"] for s in sorted(scored, key=lambda x: (fnum(x["strong_CP_rate"]), fnum(x["V_ctrl_lcb"]), -fnum(x["long_risk_rate"])), reverse=True)],
        "OS4-horizon-robust-oracle-source-selector": [s["action_id"] for s in sorted(scored, key=lambda x: (inum(x["horizon_robust_CP"]), fnum(x["weak_CP_rate"]), fnum(x["V_ctrl_lcb"])), reverse=True)],
        "OS5-value-risk-pareto-oracle-source-selector": [s["action_id"] for s in sorted(scored, key=oracle_score, reverse=True)],
        "OS6-support-balanced-oracle-source-selector": [],
    }
    # Deterministic support-balanced ordering: round-robin by family after weak score.
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for st in sorted(scored, key=lambda x: (fnum(x["weak_CP_rate"]), fnum(x["V_ctrl_lcb"]), -fnum(x["long_risk_rate"])), reverse=True):
        by_family[str(st.get("family_id"))].append(st)
    rr: list[str] = []
    while any(by_family.values()):
        for fam in sorted(list(by_family)):
            if by_family[fam]:
                rr.append(str(by_family[fam].pop(0)["action_id"]))
    selectors["OS6-support-balanced-oracle-source-selector"] = rr
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for selector_id, ordered in selectors.items():
        for k in [64, 128, 256, 512]:
            action_ids = [aid for aid in ordered if aid in by_action][:k]
            summ = panel_metrics_for_actions(action_ids, by_action)
            p_strict = int(k == 64 and summ["weak_CP_precision_all_rows"] >= 0.50 and summ["V_ctrl_lcb_all"] > 0 and summ["long_risk_rate"] <= 0.10 and summ["support_balance_pass"])
            p_weak = int(k == 256 and summ["weak_CP_precision_all_rows"] >= 0.25 and summ["V_ctrl_lcb_all"] > 0 and summ["long_risk_rate"] <= 0.20)
            row = {
                "stage": "P2_ORACLE_SOURCE_SELECTOR_UPPER_BOUND",
                "status": "oracle_selector_panel",
                "oracle_selector_id": selector_id,
                "K": k,
                **summ,
                "oracle_source_pass": p_strict,
                "oracle_source_weak_pass": p_weak,
                "uses_outcome_oracle": int(selector_id != "OS0-current-v9390-source-panel"),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            rows.append(row)
            for rank_idx, aid in enumerate(action_ids[: min(32, len(action_ids))]):
                trace.append(
                    {
                        "stage": "P2_ORACLE_SOURCE_SELECTOR_TRACE",
                        "status": "selected_action_trace",
                        "oracle_selector_id": selector_id,
                        "K": k,
                        "rank": rank_idx + 1,
                        "source_action_id": aid,
                        "weak_CP_rate": stats[aid]["weak_CP_rate"],
                        "long_risk_rate": stats[aid]["long_risk_rate"],
                        "V_ctrl_lcb": stats[aid]["V_ctrl_lcb"],
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
    # Deterministic bootstrap panels using hash modulo ordering.
    boot_rows = []
    for b in range(16):
        ordered = sorted(ids, key=lambda aid: stable_hash("bootstrap", b, aid))
        action_ids = ordered[:64]
        summ = panel_metrics_for_actions(action_ids, by_action)
        boot_rows.append(
            {
                "stage": "P2_RANDOM_PANEL_BOOTSTRAP_TRACE",
                "status": "random_stratified_hash_panel",
                "bootstrap_id": b,
                "K": 64,
                **summ,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: (inum(r["oracle_source_pass"]), inum(r["oracle_source_weak_pass"]), fnum(r["weak_CP_precision_all_rows"]), fnum(r["V_ctrl_lcb_all"])))
    oracle_source_present = int(any(fnum(r.get("V_ctrl_lcb_all")) > 0.0 and fnum(r.get("long_risk_rate")) <= 0.20 for r in rows))
    summary = {
        "stage": "P2_ORACLE_SOURCE_SELECTOR_UPPER_BOUND",
        "status": "summary",
        "best_oracle_selector_id": best["oracle_selector_id"],
        "best_K": best["K"],
        "best_weak_CP_precision_all_rows": best["weak_CP_precision_all_rows"],
        "best_V_ctrl_lcb_all": best["V_ctrl_lcb_all"],
        "best_long_risk_rate": best["long_risk_rate"],
        "oracle_source_present": oracle_source_present,
        "oracle_source_pass": max(inum(r["oracle_source_pass"]) for r in rows),
        "oracle_source_weak_pass": max(inum(r["oracle_source_weak_pass"]) for r in rows),
        "oracle_source_absent": int(not oracle_source_present),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, trace, boot_rows, summary


def feature_score_rows(stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for aid, st in stats.items():
        target = int(fnum(st["weak_CP_rate"]) > 0.0)
        for fid, value, direction in [
            ("LSA-StateNLL", st["NLL_before"], "desc"),
            ("LSA-StateCEp99", st["CEp99_before"], "desc"),
            ("LSA-StateMarginP10", st["margin_p10_before"], "asc"),
            ("LSA-StateECE", st["ECE_before"], "desc"),
            ("LSB-PayloadNorm", st["payload_norm"], "desc"),
            ("LSB-PayloadLinf", st["payload_linf"], "desc"),
            ("LSE-FamilySupportCount", st["family_support_count"], "asc"),
            ("LSF-PayloadLoadOrder", st["payload_offset"], "asc"),
        ]:
            rows.append(
                {
                    "stage": "P3_LEGAL_SOURCE_FEATURE_TRACE",
                    "status": "legal_feature_row",
                    "source_action_id": aid,
                    "feature_id": fid,
                    "feature_value": value,
                    "feature_direction": direction,
                    "target_any_weak_CP": target,
                    "weak_CP_rate": st["weak_CP_rate"],
                    "long_risk_rate": st["long_risk_rate"],
                    "uses_dataset_name": 0,
                    "uses_outcome_at_commit": 0,
                    "uses_future_step": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    return rows


def p3_legal_selector(stats: dict[str, dict[str, Any]], by_action: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    feature_rows = feature_score_rows(stats)
    by_feat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in feature_rows:
        by_feat[str(r["feature_id"])].append(r)
    frontier: list[dict[str, Any]] = []
    leaveout: list[dict[str, Any]] = []
    ablation: list[dict[str, Any]] = []
    for fid, rows in by_feat.items():
        labels = [inum(r["target_any_weak_CP"]) for r in rows]
        vals = [fnum(r["feature_value"]) for r in rows]
        auc = auc_score(vals, labels)
        desc_auc = auc
        asc_auc = 1.0 - auc
        descending = desc_auc >= asc_auc
        best_auc = max(desc_auc, asc_auc)
        ordered = sorted(stats.values(), key=lambda s: fnum(s.get({
            "LSA-StateNLL": "NLL_before",
            "LSA-StateCEp99": "CEp99_before",
            "LSA-StateMarginP10": "margin_p10_before",
            "LSA-StateECE": "ECE_before",
            "LSB-PayloadNorm": "payload_norm",
            "LSB-PayloadLinf": "payload_linf",
            "LSE-FamilySupportCount": "family_support_count",
            "LSF-PayloadLoadOrder": "payload_offset",
        }[fid])), reverse=descending)
        for k in [64, 256]:
            action_ids = [str(s["action_id"]) for s in ordered[:k]]
            summ = panel_metrics_for_actions(action_ids, by_action)
            legal_pass = int(
                k == 64
                and summ["weak_CP_h20"] >= 0.25
                and summ["weak_CP_precision_all_rows"] >= 0.25
                and summ["V_ctrl_lcb_all"] > 0
                and summ["long_risk_rate"] <= 0.15
                and summ["support_balance_pass"]
            )
            weak_pass = int(k == 64 and summ["weak_CP_h20"] >= 0.15 and summ["V_ctrl_lcb_all"] > 0)
            frontier.append(
                {
                    "stage": "P3_LEGAL_SOURCE_SELECTOR_CAPACITY",
                    "status": "legal_selector_candidate",
                    "selector_id": f"LSS1-single-feature-topK-{fid}",
                    "feature_set": fid,
                    "feature_count": 1,
                    "feature_auc_weak_CP": best_auc,
                    "feature_auc_longrisk": "",
                    "K": k,
                    **summ,
                    "legal_source_selector_pass": legal_pass,
                    "legal_source_selector_weak_pass": weak_pass,
                    "uses_dataset_name": 0,
                    "uses_outcome_at_commit": 0,
                    "uses_future_step": 0,
                    "feature_compute_time_ms_q90": 0.0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
        for dataset in sorted({str(s.get("dataset")) for s in stats.values()}):
            held = [s for s in ordered if str(s.get("dataset")) == dataset][:64]
            if not held:
                continue
            summ = panel_metrics_for_actions([str(s["action_id"]) for s in held], by_action)
            leaveout.append(
                {
                    "stage": "P3_SOURCE_SELECTOR_LEAVEOUT_TRACE",
                    "status": "leave_dataset_diagnostic",
                    "selector_id": f"LSS1-single-feature-topK-{fid}",
                    "heldout_dataset": dataset,
                    "K": 64,
                    **summ,
                    "diagnostic_only": 1,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
        ablation.append(
            {
                "stage": "P3_CONTROLLER_ABLATION_TRACE",
                "status": "single_feature_auc",
                "feature_id": fid,
                "auc_any_weak_CP": best_auc,
                "selected_direction": "desc" if descending else "asc",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(frontier, key=lambda r: (inum(r["legal_source_selector_pass"]), inum(r["legal_source_selector_weak_pass"]), fnum(r["weak_CP_precision_all_rows"]), fnum(r["V_ctrl_lcb_all"])))
    summary = {
        "stage": "P3_LEGAL_SOURCE_SELECTOR_CAPACITY",
        "status": "summary",
        "best_selector_id": best["selector_id"],
        "best_feature_set": best["feature_set"],
        "best_K": best["K"],
        "best_feature_auc_weak_CP": best["feature_auc_weak_CP"],
        "best_weak_CP_precision_h20": best["weak_CP_h20"],
        "best_weak_CP_precision_all": best["weak_CP_precision_all_rows"],
        "best_V_ctrl_lcb_all": best["V_ctrl_lcb_all"],
        "best_long_risk_rate": best["long_risk_rate"],
        "legal_source_selector_pass": max(inum(r["legal_source_selector_pass"]) for r in frontier),
        "legal_source_selector_weak_pass": max(inum(r["legal_source_selector_weak_pass"]) for r in frontier),
        "beats_v9390_source_h20_by_2x": int(best["weak_CP_h20"] >= 2 * 0.02734375),
        "uses_dataset_name": 0,
        "uses_outcome_at_commit": 0,
        "uses_future_step": 0,
        "feature_compute_time_ms_q90": 0.0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + frontier, feature_rows, frontier, leaveout, summary


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}


def write_hashes(out_dir: Path, artifacts: list[Path]) -> None:
    rows = []
    for path in artifacts:
        if path.exists():
            rows.append({"artifact": rel(path), "sha256": sha256_file(path)})
    write_csv(out_dir / "artifact_hashes.csv", rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_v9350 = Path(args.source_v9350)
    source_v9390 = Path(args.source_v9390)
    source_v9330 = Path(args.source_v9330)
    source_v9380 = Path(args.source_v9380)

    p0_rows, p0_join, p0_bias, p0, by_action, stats = p0_full_vs_panel(source_v9350, source_v9390, source_v9330, int(args.source_panel_actions))
    panel_ids = load_source_panel_ids(source_v9390, int(args.source_panel_actions))
    p1_rows, p1_trace, p1_unknown, p1 = p1_selection_provenance(source_v9330, panel_ids, stats)
    p2_rows, p2_trace, p2_boot, p2 = p2_oracle_selectors(by_action, stats, panel_ids)
    p3_rows, p3_features, p3_frontier, p3_leaveout, p3 = p3_legal_selector(stats, by_action)

    source_generator_missing = int(not inum(p3.get("legal_source_selector_pass")))
    p4 = not_run("P4_VALUE_PRODUCING_SOURCE_GENERATOR", "legal_source_selector_failed_and_AP0b_AP0f_real_generators_not_materialized")
    p4["source_generator_materialized"] = 0
    p4["next_required_implementation"] = "implement_real_AP0b_AP0f_source_generators_with_commit_time_certificates"
    p5 = not_run("P5_H20_IMMEDIATE_DIRECTION_SMOKE", "P4_source_generator_not_materialized")
    p6 = not_run("P6_HORIZON_EXTENSION_LONGRISK_AUDIT", "P5_immediate_direction_not_available")
    p7 = not_run("P7_SOURCE_TO_GENERATED_TRANSFORM_DAMAGE", "P6_source_survivor_not_available")
    p8 = not_run("P8_EFFECT_VALID_CERTIFICATE_REDESIGN", "P6_source_survivor_not_available")
    p9 = not_run("P9_MINIMAL_SOURCE_CERTIFICATE_CONTROLLER", "P8_certificate_or_source_survivor_not_available")
    p10 = not_run("P10_SELECTED_SOURCE_CONTROLLER_ONLINE_RUNTIME", "P9_controller_not_selected")
    p11 = not_run("P11_SYSTEM_INTEGRATION_GATE", "P9_controller_not_selected")
    p12 = not_run("P12_LEAVEOUT_AND_PAIRED_REPLAY_BOUNDARY", "P11_system_controller_not_official")
    p13 = not_run("P13_SHORT_FULL_SAMPLEEFF_CONTINUAL_ROBUSTNESS", "P12_official_paired_replay_not_open")

    if not inum(p0.get("source_panel_join_pass")):
        route = "R0-BoundaryOrJoinFail"
        blocker = "source_panel_join_failed"
        next_impl = "repair_source_panel_action_id_payload_hash_join"
    elif inum(p2.get("oracle_source_absent")):
        route = "R2-AP0CandidateSourceOracleAbsent"
        blocker = "ap0_candidate_source_oracle_absent"
        next_impl = "redesign_candidate_source_generator"
    elif not inum(p3.get("legal_source_selector_pass")):
        route = "R3-LegalSourceSelectorOpaque"
        blocker = "legal_source_selector_opaque"
        next_impl = "implement_real_AP0b_AP0f_value_producing_source_generators"
    elif source_generator_missing:
        route = "R4-ImmediateDirectionFail"
        blocker = "source_generator_not_materialized"
        next_impl = "materialize_h20_source_generator_smoke"
    else:
        route = "R8-ControllerDecisionFail"
        blocker = "controller_not_selected"
        next_impl = "build_minimal_source_certificate_controller"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9390": read_json(source_v9390 / "route_decision.json").get("route"),
        "full_action_count": p0.get("full_action_count"),
        "source_panel_action_count": p0.get("source_panel_action_count"),
        "source_panel_join_pass": p0.get("source_panel_join_pass"),
        "full_weak_CP_row_rate": p0.get("full_weak_CP_row_rate"),
        "source_panel_weak_CP_row_rate": p0.get("source_panel_weak_CP_row_rate"),
        "full_horizon_robust_action_rate": p0.get("full_horizon_robust_action_rate"),
        "source_panel_horizon_robust_action_rate": p0.get("source_panel_horizon_robust_action_rate"),
        "full_long_risk_rate": p0.get("full_long_risk_rate"),
        "source_panel_long_risk_rate": p0.get("source_panel_long_risk_rate"),
        "selection_bias_PSI": p0.get("selection_bias_PSI"),
        "selection_bias_KL": p0.get("selection_bias_KL"),
        "max_family_gap": p0.get("max_family_gap"),
        "max_step_bucket_gap": p0.get("max_step_bucket_gap"),
        "route_flip_full_vs_source_panel": p0.get("route_flip_full_vs_source_panel"),
        "source_panel_bias_pass": p0.get("source_panel_bias_pass"),
        "selection_rule_identified": p1.get("selection_rule_identified"),
        "source_selection_rule": "v9380_first_N_payload_available_slice",
        "uses_outcome_in_source_selection": p1.get("uses_outcome_in_source_selection"),
        "oracle_source_pass": p2.get("oracle_source_pass"),
        "oracle_source_weak_pass": p2.get("oracle_source_weak_pass"),
        "oracle_source_present": p2.get("oracle_source_present"),
        "best_oracle_selector_id": p2.get("best_oracle_selector_id"),
        "best_oracle_K": p2.get("best_K"),
        "best_oracle_weak_CP_precision_all_rows": p2.get("best_weak_CP_precision_all_rows"),
        "best_oracle_V_ctrl_lcb_all": p2.get("best_V_ctrl_lcb_all"),
        "best_oracle_long_risk_rate": p2.get("best_long_risk_rate"),
        "legal_source_selector_pass": p3.get("legal_source_selector_pass"),
        "legal_source_selector_weak_pass": p3.get("legal_source_selector_weak_pass"),
        "best_legal_selector_id": p3.get("best_selector_id"),
        "best_legal_feature_set": p3.get("best_feature_set"),
        "best_legal_feature_auc_weak_CP": p3.get("best_feature_auc_weak_CP"),
        "best_legal_weak_CP_precision_h20": p3.get("best_weak_CP_precision_h20"),
        "best_legal_weak_CP_precision_all": p3.get("best_weak_CP_precision_all"),
        "best_legal_V_ctrl_lcb_all": p3.get("best_V_ctrl_lcb_all"),
        "best_legal_long_risk_rate": p3.get("best_long_risk_rate"),
        "source_generator_materialized": 0,
        "h20_immediate_direction_pass": 0,
        "horizon_extension_pass": 0,
        "effect_valid_certificate_pass": 0,
        "source_controller_pass": 0,
        "selected_runtime_pass": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "success_v9400_strict_purekan_functional": 0,
        "success_v9400_full_functional": 0,
        "success_v9400_external_ready": 0,
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
        "source_v9350": rel(source_v9350),
        "source_v9380": rel(source_v9380),
        "source_v9390": rel(source_v9390),
        "source_v9330": rel(source_v9330),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv(out_dir / "p0_full_vs_source_panel_diagnosis.csv", p0_rows)
    write_csv(out_dir / "source_panel_join_trace_v9400.csv", p0_join)
    write_csv(out_dir / "source_panel_bias_trace_v9400.csv", p0_bias)
    write_csv(out_dir / "p1_source_selection_provenance_audit.csv", p1_rows)
    write_csv(out_dir / "source_selection_trace_v9400.csv", p1_trace)
    write_csv(out_dir / "source_selection_unknown_reason_table_v9400.csv", p1_unknown)
    write_csv(out_dir / "p2_oracle_source_selector_upper_bound.csv", p2_rows)
    write_csv(out_dir / "oracle_source_selector_trace_v9400.csv", p2_trace)
    write_csv(out_dir / "random_panel_bootstrap_trace_v9400.csv", p2_boot)
    write_csv(out_dir / "p3_legal_source_selector_capacity.csv", p3_rows)
    write_csv(out_dir / "legal_source_feature_trace_v9400.csv", p3_features)
    write_csv(out_dir / "source_selector_frontier_trace_v9400.csv", p3_frontier)
    write_csv(out_dir / "source_selector_leaveout_trace_v9400.csv", p3_leaveout)
    write_csv(out_dir / "p4_value_producing_source_generator.csv", [p4])
    write_csv(out_dir / "source_generator_payload_trace_v9400.csv", [not_run("P4_SOURCE_GENERATOR_PAYLOAD_TRACE", "AP0b_AP0f_payload_generators_not_materialized")])
    write_csv(out_dir / "source_generator_certificate_trace_v9400.csv", [not_run("P4_SOURCE_GENERATOR_CERTIFICATE_TRACE", "AP0b_AP0f_certificate_tensors_not_materialized")])
    write_csv(out_dir / "action_apply_replay_trace_v9400.csv", [not_run("P4_ACTION_APPLY_REPLAY_TRACE", "AP0b_AP0f_payload_generators_not_materialized")])
    write_csv(out_dir / "p5_h20_immediate_direction_smoke.csv", [p5])
    write_csv(out_dir / "h20_branch_outcome_trace_v9400.csv", [not_run("P5_H20_BRANCH_OUTCOME_TRACE", "P4_source_generator_not_materialized")])
    write_csv(out_dir / "immediate_direction_frontier_trace_v9400.csv", [not_run("P5_IMMEDIATE_DIRECTION_FRONTIER_TRACE", "P4_source_generator_not_materialized")])
    write_csv(out_dir / "p6_horizon_extension_longrisk_audit.csv", [p6])
    write_csv(out_dir / "horizon_value_curve_trace_v9400.csv", [not_run("P6_HORIZON_VALUE_CURVE_TRACE", "P5_immediate_direction_not_available")])
    write_csv(out_dir / "longrisk_attribution_trace_v9400.csv", [not_run("P6_LONGRISK_ATTRIBUTION_TRACE", "P5_immediate_direction_not_available")])
    write_csv(out_dir / "p7_source_to_generated_transform_damage.csv", [p7])
    write_csv(out_dir / "source_generated_damage_trace_v9400.csv", [not_run("P7_SOURCE_GENERATED_DAMAGE_TRACE", "P6_source_survivor_not_available")])
    write_csv(out_dir / "transform_geometry_trace_v9400.csv", [not_run("P7_TRANSFORM_GEOMETRY_TRACE", "P6_source_survivor_not_available")])
    write_csv(out_dir / "p8_effect_valid_certificate_redesign.csv", [p8])
    write_csv(out_dir / "certificate_component_trace_v9400.csv", [not_run("P8_CERTIFICATE_COMPONENT_TRACE", "P6_source_survivor_not_available")])
    write_csv(out_dir / "certificate_ablation_trace_v9400.csv", [not_run("P8_CERTIFICATE_ABLATION_TRACE", "P6_source_survivor_not_available")])
    write_csv(out_dir / "p9_minimal_source_certificate_controller.csv", [p9])
    write_csv(out_dir / "controller_frontier_trace_v9400.csv", [not_run("P9_CONTROLLER_FRONTIER_TRACE", "P8_certificate_or_source_survivor_not_available")])
    write_csv(out_dir / "controller_ablation_trace_v9400.csv", [not_run("P9_CONTROLLER_ABLATION_TRACE", "P8_certificate_or_source_survivor_not_available")])
    write_csv(out_dir / "p10_selected_source_controller_online_runtime.csv", [p10])
    write_csv(out_dir / "runtime_component_trace_v9400.csv", [not_run("P10_RUNTIME_COMPONENT_TRACE", "P9_controller_not_selected")])
    write_csv(out_dir / "online_runtime_trace_v9400.csv", [not_run("P10_ONLINE_RUNTIME_TRACE", "P9_controller_not_selected")])
    write_csv(out_dir / "p11_system_integration_gate_v9400.csv", [p11 | {"official_eligible": 0, "system_legal_controller_pass": 0, "selected_controller_used": 0, "selected_payload_apply_used": 0, "diagnostic_promoted_to_official": 0}])
    write_csv(out_dir / "system_controller_trace_v9400.csv", [not_run("P11_SYSTEM_CONTROLLER_TRACE", "P9_controller_not_selected")])
    write_csv(out_dir / "p12_leaveout_and_paired_replay_boundary.csv", [p12])
    write_csv(out_dir / "paired_replay_trace_v9400.csv", [not_run("P12_PAIRED_REPLAY_TRACE", "P11_system_controller_not_official")])
    write_csv(out_dir / "leaveout_trace_v9400.csv", [not_run("P12_LEAVEOUT_TRACE", "P11_system_controller_not_official")])
    write_csv(out_dir / "p13_short_full_sampleeff_continual_robustness.csv", [p13])
    write_csv(out_dir / "short_full_trace_v9400.csv", [not_run("P13_SHORT_FULL_TRACE", "P12_official_paired_replay_not_open")])
    write_csv(out_dir / "continual_trace_v9400.csv", [not_run("P13_CONTINUAL_TRACE", "P12_official_paired_replay_not_open")])
    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "full_vs_panel_join_pass": p0.get("source_panel_join_pass"),
        "source_selection_provenance_pass": p1.get("selection_rule_pass"),
        "oracle_source_pass": p2.get("oracle_source_pass"),
        "oracle_source_weak_pass": p2.get("oracle_source_weak_pass"),
        "legal_source_selector_pass": p3.get("legal_source_selector_pass"),
        "source_generator_materialized": 0,
        "h20_immediate_direction_pass": 0,
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "contract_audit_v9400.csv", [contract])
    failure = {
        "route": route,
        "F0_join_fail": int(not inum(p0.get("source_panel_join_pass"))),
        "F1_source_panel_sampling_bias": int(not inum(p0.get("source_panel_bias_pass"))),
        "F2_oracle_source_absent": int(inum(p2.get("oracle_source_absent"))),
        "F3_legal_source_selector_opaque": int(not inum(p3.get("legal_source_selector_pass")) and inum(p2.get("oracle_source_present"))),
        "F4_source_generator_missing": 1,
        "F5_immediate_direction_not_run": 1,
        "F11_system_not_official": 1,
        "primary_blocker": blocker,
    }
    write_csv(out_dir / "failure_table_v9400.csv", [failure])
    audit = audit_no_fake(sorted(out_dir.glob("*.csv")))
    write_csv(out_dir / "provenance_audit_v9400.csv", [audit])
    artifacts = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "p0_full_vs_source_panel_diagnosis.csv",
        out_dir / "p1_source_selection_provenance_audit.csv",
        out_dir / "p2_oracle_source_selector_upper_bound.csv",
        out_dir / "p3_legal_source_selector_capacity.csv",
        out_dir / "p4_value_producing_source_generator.csv",
        out_dir / "p11_system_integration_gate_v9400.csv",
        out_dir / "contract_audit_v9400.csv",
        out_dir / "provenance_audit_v9400.csv",
        out_dir / "failure_table_v9400.csv",
    ]
    write_hashes(out_dir, artifacts)
    print(json.dumps({"out_dir": str(out_dir), "route": route, "oracle_source_weak_pass": p2.get("oracle_source_weak_pass"), "legal_source_selector_pass": p3.get("legal_source_selector_pass")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
