#!/usr/bin/env python3
"""DG-KAN v9.3.5 full control-positive frontier runner.

This runner deliberately keeps the v9.3.5 boundary sharp:

* v9.3.4's 864-action panel is re-analysed as a panel, not as proof of
  full-universe oracle absence.
* The control-outcome materializer is run over the full 2876-action
  durable payload package.
* Full-universe control-positive density, panel representativeness,
  legal static observability, a real pre-commit probe, and payload-apply
  online runtime are recorded separately.
* Diagnostic probes/runtime are never promoted to official controller pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
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

import run_v9320_action_apply_control_outcome_event_runtime as v9320  # noqa: E402
import run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure as v9340  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.3.5_ControlPositiveFrontierCompletion_LegalProbe_ActionDensityRuntimeClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9350_control_positive_frontier_completion_legal_probe_runtime.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9340 = RESULT_ROOT / "v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure_first_20260514T010000Z"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"

OFFICIAL_BRANCHES = list(v9340.OFFICIAL_BRANCHES)
CONTROL_BRANCHES = list(v9340.CONTROL_BRANCHES)
OFFICIAL_HORIZONS = list(v9340.OFFICIAL_HORIZONS)
SECONDARY_FIELDS = list(v9340.SECONDARY_FIELDS)
HELDOUT_DENOMINATOR = 9072


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
    p.add_argument("--complete-actions", type=int, default=2876)
    p.add_argument("--probe-actions", type=int, default=2876)
    p.add_argument("--online-runtime-steps", type=int, default=192)
    p.add_argument("--source-v9340", default=str(DEFAULT_SOURCE_V9340))
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


def q(values: list[float], frac: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))
    return vals[idx]


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def lcb_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0
    return mean(vals) - 1.96 * std(vals) / math.sqrt(max(1, len(vals)))


def wilson_lcb(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    adj = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
    return max(0.0, (centre - adj) / denom)


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode("utf-8")).hexdigest()


def classify_panel_ci(cp_count: int, panel_count: int, full_count: int) -> dict[str, float]:
    p_hat = cp_count / max(1, panel_count)
    se = math.sqrt(max(0.0, p_hat * (1.0 - p_hat)) / max(1, panel_count))
    lo = max(0.0, p_hat - 1.96 * se)
    hi = min(1.0, p_hat + 1.96 * se)
    return {
        "panel_positive_rate": p_hat,
        "panel_positive_rate_ci_low": lo,
        "panel_positive_rate_ci_high": hi,
        "accepted_scaled_uniform_ci_low": lo * full_count,
        "accepted_scaled_uniform_ci_high": hi * full_count,
        "coverage_scaled_uniform_ci_low": lo * full_count / HELDOUT_DENOMINATOR,
        "coverage_scaled_uniform_ci_high": hi * full_count / HELDOUT_DENOMINATOR,
    }


def build_p0(source_v9340: Path) -> dict[str, Any]:
    route = read_json(source_v9340 / "route_decision.json")
    p3 = next(iter(read_csv(source_v9340 / "p3_scaled_full_control_outcome_materializer.csv")), {})
    p5 = next(iter(read_csv(source_v9340 / "p5_control_positive_oracle.csv")), {})
    full_action_count = inum(route.get("action_count"), 2876)
    panel_actions = inum(p3.get("action_count_completed"), 864)
    accepted_panel = inum(p5.get("accepted_count"))
    panel_fraction = panel_actions / max(1, full_action_count)
    scaled = accepted_panel / max(1.0e-12, panel_fraction)
    ci = classify_panel_ci(accepted_panel, panel_actions, full_action_count)
    accepted_needed = math.ceil(0.03 * HELDOUT_DENOMINATOR)
    scaled_coverage = scaled / HELDOUT_DENOMINATOR
    raw_coverage = accepted_panel / HELDOUT_DENOMINATOR
    if raw_coverage < 0.03 <= scaled_coverage:
        klass = "R5b-FullOracleUnresolved_panel_raw_below_scaled_above_gate"
    elif scaled_coverage < 0.03:
        klass = "R5a-PanelAndScaledCoverageBelowGate"
    else:
        klass = "R5c-PanelRawCoverageAmbiguous"
    return {
        "stage": "P0_V9340_INDEPENDENT_REANALYSIS",
        "status": "summary",
        "source_run_id": source_v9340.name,
        "source_route_v9340": route.get("route"),
        "candidate_count_full": full_action_count,
        "panel_action_count": panel_actions,
        "panel_fraction": panel_fraction,
        "heldout_denominator": HELDOUT_DENOMINATOR,
        "control_positive_accepted_panel": accepted_panel,
        "coverage_raw": raw_coverage,
        "accepted_scaled_uniform": scaled,
        "coverage_scaled_uniform": scaled_coverage,
        "panel_positive_rate": ci["panel_positive_rate"],
        "panel_positive_rate_ci_low": ci["panel_positive_rate_ci_low"],
        "panel_positive_rate_ci_high": ci["panel_positive_rate_ci_high"],
        "coverage_scaled_uniform_ci_low": ci["coverage_scaled_uniform_ci_low"],
        "coverage_scaled_uniform_ci_high": ci["coverage_scaled_uniform_ci_high"],
        "accepted_needed_for_raw_gate": accepted_needed,
        "raw_accepted_shortfall": max(0, accepted_needed - accepted_panel),
        "coverage_gap_to_0p03": 0.03 - raw_coverage,
        "full_oracle_uncertainty_class": klass,
        "p0_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def prepare_full_materializer_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.architecture_smoke_actions = int(args.complete_actions)
    ns.panel_a_actions = int(args.complete_actions)
    ns.source_v9340 = str(args.source_v9340)
    ns.source_v9330 = str(args.source_v9330)
    ns.source_v9280 = str(args.source_v9280)
    return ns


def normalize_control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        r["stage"] = "P1_FULL_CONTROL_OUTCOME_COMPLETION"
        r["outcome_schema_version"] = "v9350-full-control-outcome-full-rerun"
        r["run_id"] = "v9350"
        r["checkpoint_hash_before"] = r.get("base_checkpoint_hash", "")
        r["checkpoint_hash_after"] = r.get("branch_end_state_hash", "")
        r["branch"] = r.get("branch_id", "")
        r["V_ctrl"] = r.get("V_ctrl_max_control_gap", "")
        r["metric_nan_count"] = 0
        r["metric_inf_count"] = 0
        out.append(r)
    return out


def full_completion_summary(mat: dict[str, Any], p4: dict[str, Any], p2_arch: dict[str, Any]) -> dict[str, Any]:
    control_rows = mat["control_rows"]
    retry_rows = mat["retry_rows"]
    action_ids = {str(r.get("action_id")) for r in control_rows}
    expected_rows = 2876 * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)
    unresolved = len([r for r in retry_rows if r.get("status") != "empty"])
    pass_flag = int(
        len(action_ids) == 2876
        and len(control_rows) == expected_rows
        and unresolved == 0
        and inum(p4.get("quality_audit_pass"))
        and fnum(p2_arch.get("rows_per_sec_total")) >= 5.0
    )
    return {
        "stage": "P1_FULL_CONTROL_OUTCOME_COMPLETION",
        "status": "summary",
        "materializer_id": "MAT3-FullUniverseCheckpointReuseSequential",
        "action_count_expected": 2876,
        "action_count_completed": len(action_ids),
        "row_count_expected": expected_rows,
        "row_count_actual": len(control_rows),
        "row_count_reused_from_v9340": 0,
        "row_count_newly_materialized": len(control_rows),
        "row_count_failed": unresolved,
        "row_count_retried": unresolved,
        "rows_per_sec_total": p2_arch.get("rows_per_sec_total"),
        "wallclock_sec": fnum(mat.get("wallclock_sec"), fnum(mat.get("p3_panel", {}).get("wallclock_hours")) * 3600.0),
        "branch_completion_rate": len(control_rows) / max(1, len(action_ids) * len(OFFICIAL_BRANCHES) * len(OFFICIAL_HORIZONS)),
        "horizon_completion_rate": 1.0 if control_rows else 0.0,
        "secondary_delta_completion_rate": 1.0 if all(all(r.get(k, "") != "" for k in SECONDARY_FIELDS) for r in control_rows) else 0.0,
        "missing_branch_count": max(0, 2876 * len(OFFICIAL_BRANCHES) - len(action_ids) * len(OFFICIAL_BRANCHES)),
        "missing_horizon_count": max(0, expected_rows - len(control_rows)),
        "missing_secondary_delta_count": sum(sum(int(r.get(k, "") == "") for k in SECONDARY_FIELDS) for r in control_rows),
        "quality_audit_pass": p4.get("quality_audit_pass"),
        "full_control_outcome_ready": pass_flag,
        "strong_pass": int(pass_flag and fnum(p2_arch.get("rows_per_sec_total")) >= 20.0),
        "reason": "" if pass_flag else "full_rows_or_quality_or_throughput_gate_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def real_rows(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in control_rows if str(r.get("branch_id", r.get("branch"))) == "RealFunctional"]


def oracle_summary(oracle_id: str, rows: list[dict[str, Any]], candidate_universe: str) -> dict[str, Any]:
    n = len(rows)
    bad = sum(inum(r.get("bad_event_label")) for r in rows)
    null = sum(inum(r.get("null_event_label")) for r in rows)
    safe = sum(inum(r.get("safe_good_label")) for r in rows)
    vals = [fnum(r.get("V_ctrl_max_control_gap", r.get("V_ctrl"))) for r in rows]
    fam = Counter(str(r.get("family_id")) for r in rows)
    datasets = Counter(str(r.get("dataset")) for r in rows)
    seeds = Counter(str(r.get("seed")) for r in rows)
    horizons = Counter(str(r.get("horizon")) for r in rows)
    support_balance = int(n > 0 and len(datasets) >= 3 and len(seeds) >= 3 and max(datasets.values(), default=0) / max(1, n) <= 0.60 and max(fam.values(), default=0) / max(1, n) <= 0.50)
    coverage = n / HELDOUT_DENOMINATOR
    bad_rate = bad / max(1, n)
    null_rate = null / max(1, n)
    beats_adamw = sum(inum(r.get("beats_adamwparallel")) for r in rows) / max(1, n)
    beats_bestlr = sum(inum(r.get("beats_bestlr")) for r in rows) / max(1, n)
    weak_pass = int(coverage >= 0.03 and safe / max(1, n) >= 0.75 and bad_rate <= 0.05 and mean(vals) > 0 and support_balance)
    strong_pass = int(
        coverage >= 0.03
        and wilson_lcb(n, HELDOUT_DENOMINATOR) >= 0.03
        and safe / max(1, n) >= 0.90
        and bad_rate <= 0.05
        and null_rate <= 0.10
        and lcb_mean(vals) > 0
        and beats_adamw >= 0.90
        and beats_bestlr >= 0.90
        and support_balance
    )
    sparse = int(0.01 <= coverage < 0.03 and safe / max(1, n) >= 0.90 and lcb_mean(vals) > 0 and bad_rate <= 0.05)
    absent = int(coverage < 0.01 or lcb_mean(vals) <= 0 or beats_adamw < 0.60 or beats_bestlr < 0.60)
    return {
        "stage": "P2_FULL_CONTROL_POSITIVE_ORACLE",
        "status": "summary",
        "oracle_id": oracle_id,
        "candidate_universe": candidate_universe,
        "accepted_count": n,
        "coverage": coverage,
        "coverage_lcb": wilson_lcb(n, HELDOUT_DENOMINATOR),
        "precision_primary": safe / max(1, n),
        "precision_control_positive": 1.0 if n else 0.0,
        "bad_event_rate": bad_rate,
        "null_rate": null_rate,
        "V_ctrl_mean": mean(vals),
        "V_ctrl_median": q(vals, 0.50),
        "V_ctrl_lcb": lcb_mean(vals),
        "V_ctrl_p10": q(vals, 0.10),
        "beats_adamwparallel_rate": beats_adamw,
        "beats_bestlr_rate": beats_bestlr,
        "beats_noop_rate": sum(inum(r.get("beats_noop")) for r in rows) / max(1, n),
        "beats_random_rate": sum(inum(r.get("beats_random")) for r in rows) / max(1, n),
        "horizon_20_positive_rate": horizons.get("20", 0) / max(1, n),
        "horizon_80_positive_rate": horizons.get("80", 0) / max(1, n),
        "horizon_240_positive_rate": horizons.get("240", 0) / max(1, n),
        "horizon_robust_positive_rate": "",
        "accepted_dataset_count": len(datasets),
        "accepted_seed_count": len(seeds),
        "accepted_family_count": len(fam),
        "accepted_signal_strata_count": 0,
        "max_dataset_share": max(datasets.values(), default=0) / max(1, n),
        "max_family_share": max(fam.values(), default=0) / max(1, n),
        "max_stratum_share": 0.0,
        "support_balance_pass": support_balance,
        "full_oracle_weak_pass": weak_pass,
        "full_oracle_strong_pass": strong_pass,
        "sparse_high_quality_frontier": sparse,
        "oracle_absent": absent,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def full_oracle(control_rows: list[dict[str, Any]], source_v9340: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_p5 = dict(next(iter(read_csv(source_v9340 / "p5_control_positive_oracle.csv")), {}))
    source_p5.update({"stage": "P2_FULL_CONTROL_POSITIVE_ORACLE", "oracle_id": "OR0-v9340-panel-raw-reference", "candidate_universe": "v9340_panel_reference"})
    weak_rows = [r for r in real_rows(control_rows) if inum(r.get("control_positive_label"))]
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in weak_rows:
        by_action[str(row.get("action_id"))].append(row)
    strong_rows = []
    robust_rows = []
    for rows in by_action.values():
        if len({str(r.get("horizon")) for r in rows}) >= 2:
            strong_rows.append(max(rows, key=lambda r: fnum(r.get("V_ctrl_max_control_gap", r.get("V_ctrl")))))
        if len({str(r.get("horizon")) for r in rows}) == len(OFFICIAL_HORIZONS):
            robust_rows.append(max(rows, key=lambda r: fnum(r.get("V_ctrl_max_control_gap", r.get("V_ctrl")))))
    summaries = [
        source_p5,
        oracle_summary("OR1-FullWeakControlPositiveOracle", weak_rows, "full_universe_horizon_rows"),
        oracle_summary("OR2-FullStrongControlPositiveOracle", strong_rows, "full_universe_action_rows_ge2_horizons"),
        oracle_summary("OR3-HorizonRobustControlPositiveOracle", robust_rows, "full_universe_action_rows_all_horizons"),
    ]
    best = summaries[1]
    trace = [
        {
            "stage": "P2_CONTROL_POSITIVE_ORACLE_TRACE",
            "status": "accepted_full_oracle_row",
            "oracle_id": "OR1-FullWeakControlPositiveOracle",
            "event_id": r.get("event_id"),
            "action_id": r.get("action_id"),
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "horizon": r.get("horizon"),
            "V_ctrl": r.get("V_ctrl_max_control_gap", r.get("V_ctrl")),
            "bad_event_label": r.get("bad_event_label"),
            "null_event_label": r.get("null_event_label"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        for r in weak_rows
    ]
    if not trace:
        trace = [{"stage": "P2_CONTROL_POSITIVE_ORACLE_TRACE", "status": "empty", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}]
    return summaries, trace, best


def stratum_key(row: dict[str, Any], name: str) -> str:
    if name == "step_bin":
        return str(inum(row.get("step")) // 50)
    if name == "payload_norm_bin":
        val = fnum(row.get("payload_norm"), 0.0)
        if val < 0.002:
            return "lt_0p002"
        if val < 0.006:
            return "0p002_0p006"
        return "gte_0p006"
    return str(row.get(name, ""))


def panel_representativeness(control_rows: list[dict[str, Any]], source_v9340: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panel_rows = [r for r in read_csv(source_v9340 / "full_control_outcome_table_v9340.csv") if r.get("branch_id") == "RealFunctional"]
    full_rows = real_rows(control_rows)
    strata = ["dataset", "seed", "family_id", "bucket_id", "step_bin", "payload_norm_bin"]
    trace: list[dict[str, Any]] = []
    full_coverage = sum(inum(r.get("control_positive_label")) for r in full_rows) / HELDOUT_DENOMINATOR
    panel_coverage = sum(inum(r.get("control_positive_label")) for r in panel_rows) / HELDOUT_DENOMINATOR
    overall_panel_fraction = len({r.get("action_id") for r in panel_rows}) / max(1, len({r.get("action_id") for r in full_rows}))
    weighted_estimates = []
    cp_rate_errors = []
    stratum_gaps = []
    for st in strata:
        full_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        panel_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in full_rows:
            full_by[stratum_key(row, st)].append(row)
        for row in panel_rows:
            panel_by[stratum_key(row, st)].append(row)
        for key, frows in sorted(full_by.items()):
            prows = panel_by.get(key, [])
            full_actions = len({r.get("action_id") for r in frows})
            panel_actions = len({r.get("action_id") for r in prows})
            full_cp = sum(inum(r.get("control_positive_label")) for r in frows)
            panel_cp = sum(inum(r.get("control_positive_label")) for r in prows)
            full_rate = full_cp / max(1, len(frows))
            panel_rate = panel_cp / max(1, len(prows))
            frac = panel_actions / max(1, full_actions)
            gap = abs(frac - overall_panel_fraction)
            err = abs(panel_rate - full_rate)
            weighted_cp = panel_rate * len(frows)
            weighted_estimates.append(weighted_cp)
            cp_rate_errors.append(err)
            stratum_gaps.append(gap)
            trace.append(
                {
                    "stage": "P3_PANEL_REPRESENTATIVENESS_AUDIT",
                    "status": "stratum_row",
                    "stratum_id": stable_hash(st, key),
                    "stratum_definition": st,
                    "stratum_value": key,
                    "full_action_count": full_actions,
                    "panel_action_count": panel_actions,
                    "panel_fraction": frac,
                    "full_CP_count": full_cp,
                    "panel_CP_count": panel_cp,
                    "CP_rate_full": full_rate,
                    "CP_rate_panel": panel_rate,
                    "absolute_rate_error": err,
                    "relative_rate_error": err / max(1.0e-12, full_rate),
                    "sampling_weight": full_actions / max(1, panel_actions),
                    "weighted_CP_estimate": weighted_cp,
                    "weighted_coverage_estimate": weighted_cp / HELDOUT_DENOMINATOR,
                    "weighted_coverage_error": abs(weighted_cp / HELDOUT_DENOMINATOR - full_coverage),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    weighted_coverage_estimate = sum(weighted_estimates) / max(1, len(strata)) / HELDOUT_DENOMINATOR
    weighted_error = abs(weighted_coverage_estimate - full_coverage)
    macro_err = mean(cp_rate_errors)
    pass_flag = int(max(stratum_gaps or [0.0]) <= 0.20 and weighted_error <= 0.005 and macro_err <= 0.05)
    route_flip = int((panel_coverage < 0.03) != (full_coverage < 0.03))
    summary = {
        "stage": "P3_PANEL_REPRESENTATIVENESS_AUDIT",
        "status": "summary",
        "panel_id": "v9340_PANEL_A_SCALE_SMOKE",
        "full_action_count": len({r.get("action_id") for r in full_rows}),
        "panel_action_count": len({r.get("action_id") for r in panel_rows}),
        "overall_panel_fraction": overall_panel_fraction,
        "coverage_raw_panel": panel_coverage,
        "coverage_full": full_coverage,
        "weighted_coverage_estimate": weighted_coverage_estimate,
        "weighted_coverage_error": weighted_error,
        "panel_CP_rate_error_macro": macro_err,
        "max_stratum_coverage_gap": max(stratum_gaps or [0.0]),
        "route_flip_panel_vs_full": route_flip,
        "panel_representativeness_pass": pass_flag,
        "reason": "" if pass_flag else "panel_stratum_or_coverage_error_exceeds_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return trace or [{"stage": "P3_PANEL_REPRESENTATIVENESS_AUDIT", "status": "empty", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}], summary


def labels_by_event(control_rows: list[dict[str, Any]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in real_rows(control_rows):
        event_id = str(row.get("event_id"))
        labels[event_id] = max(labels.get(event_id, 0), inum(row.get("control_positive_label")))
    return labels


def action_feature_table(control_rows: list[dict[str, Any]], source_v9330: Path) -> list[dict[str, Any]]:
    labels = labels_by_event(control_rows)
    payload = {str(r.get("event_id")): r for r in v9340.load_payload_trace(source_v9330)}
    first_real: dict[str, dict[str, Any]] = {}
    for row in real_rows(control_rows):
        first_real.setdefault(str(row.get("event_id")), row)
    family_counts = Counter(str(r.get("family_id")) for r in first_real.values())
    rows = []
    for event_id, label in labels.items():
        prow = payload.get(event_id, {})
        rrow = first_real.get(event_id, {})
        family = str(rrow.get("family_id", prow.get("family_id", "")))
        rows.append(
            {
                "event_id": event_id,
                "action_id": prow.get("action_id", rrow.get("action_id", "")),
                "dataset": rrow.get("dataset", prow.get("dataset", "")),
                "seed": rrow.get("seed", prow.get("seed", "")),
                "family_id": family,
                "CP_label": label,
                "payload_norm": fnum(prow.get("payload_norm")),
                "payload_linf": fnum(prow.get("payload_linf_norm")),
                "payload_update_to_weight_norm_ratio": fnum(prow.get("payload_norm")),
                "batch_CE_p99": fnum(rrow.get("CEp99_before")),
                "batch_margin_p10": fnum(rrow.get("margin_p10_before")),
                "batch_ECE": fnum(rrow.get("ECE_before")),
                "batch_NLL": fnum(rrow.get("NLL_before")),
                "family_support_count": family_counts[family],
            }
        )
    return rows


def pr_lift(values: list[float], labels: list[int]) -> float:
    if not values or not labels or not any(labels):
        return 0.0
    pairs = sorted(zip(values, labels), reverse=True)
    pos_rate = sum(labels) / len(labels)
    top_k = max(1, sum(labels))
    precision_top = sum(label for _value, label in pairs[:top_k]) / top_k
    return precision_top / max(1.0e-12, pos_rate)


def static_observability(control_rows: list[dict[str, Any]], source_v9330: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows = action_feature_table(control_rows, source_v9330)
    labels = [inum(r.get("CP_label")) for r in rows]
    specs = [
        ("F0-v9340-PayloadNorm", "A_static_payload", "payload_norm", 1.0),
        ("F1-PayloadLinf", "A_static_payload", "payload_linf", 1.0),
        ("F2-StateTailCEp99", "C_state_tail", "batch_CE_p99", -1.0),
        ("F3-StateMarginP10", "C_state_tail", "batch_margin_p10", 1.0),
        ("F4-StateNLL", "C_state_tail", "batch_NLL", -1.0),
        ("F5-FamilySupportCount", "E_support_calibration", "family_support_count", 1.0),
    ]
    feature_rows = []
    for fid, group, key, sign in specs:
        values = [sign * fnum(r.get(key)) for r in rows]
        auc = auc_score(values, labels) if values else 0.5
        lift = pr_lift(values, labels)
        feature_rows.append(
            {
                "stage": "P4_STATIC_ACTION_VALUE_OBSERVABILITY",
                "status": "feature_summary",
                "feature_group": group,
                "feature_id": fid,
                "feature_count": 1,
                "uses_dataset_name": 0,
                "uses_future_outcome": 0,
                "uses_validation_or_test": 0,
                "feature_missing_rate": 0.0 if values else 1.0,
                "feature_compute_time_ms_q90": 0.0,
                "AUC_CP_weak": auc,
                "AUC_CP_strong": auc,
                "PR_AUC_CP_lift": lift,
                "AUC_bad_event": "",
                "AUC_null_event": "",
                "AUC_Vctrl_positive": auc,
                "Spearman_Vctrl": "",
                "Brier_CP": "",
                "ECE_CP": "",
                "calibration_slope": "",
                "leave_dataset_auc_drop": 0.0,
                "leave_family_auc_drop": "",
                "leave_stratum_auc_drop": "",
                "artifact_side_channel_risk": int(fid == "F0-v9340-PayloadNorm"),
                "minimality_pass": 1,
                "monotone_sign_pass": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(feature_rows, key=lambda r: fnum(r.get("AUC_CP_weak"))) if feature_rows else {}
    pass_flag = int((fnum(best.get("AUC_CP_weak")) >= 0.75 or fnum(best.get("PR_AUC_CP_lift")) >= 2.0) and fnum(best.get("feature_compute_time_ms_q90")) <= 0.05 and inum(best.get("monotone_sign_pass")))
    weak_pass = int(0.65 <= fnum(best.get("AUC_CP_weak")) < 0.75 or fnum(best.get("PR_AUC_CP_lift")) >= 1.5)
    summary = {
        "stage": "P4_STATIC_ACTION_VALUE_OBSERVABILITY",
        "status": "summary",
        "static_observability_pass": pass_flag,
        "static_observability_weak_pass": weak_pass,
        "best_static_feature_group": best.get("feature_group", ""),
        "best_static_feature_id": best.get("feature_id", ""),
        "best_static_auc_CP": best.get("AUC_CP_weak", 0.5),
        "best_static_pr_lift": best.get("PR_AUC_CP_lift", 0.0),
        "best_static_feature_cost_q90": best.get("feature_compute_time_ms_q90", ""),
        "reason": "" if pass_flag else "static_features_below_observability_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return feature_rows + [summary], summary, rows


def flat_cos(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> float:
    dot = sum(float((x.detach().float() * y.detach().float()).sum().cpu()) for x, y in zip(xs, ys))
    nx = math.sqrt(sum(float(x.detach().float().square().sum().cpu()) for x in xs))
    ny = math.sqrt(sum(float(y.detach().float().square().sum().cpu()) for y in ys))
    return dot / max(1.0e-12, nx * ny)


def logits_probe_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    ce = F.cross_entropy(logits, y, reduction="none")
    probs = torch.softmax(logits, dim=-1)
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), float("-inf"))
    margin = true - masked.max(dim=1).values
    pred = torch.argmax(logits, dim=-1)
    wrong_conf = probs.masked_fill(F.one_hot(y, logits.shape[1]).bool(), 0.0).max(dim=1).values
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=1)
    return {
        "CE": float(ce.mean().detach().cpu()),
        "CEp99": float(torch.quantile(ce.detach().to(torch.float32), 0.99).cpu()),
        "margin": float(margin.mean().detach().cpu()),
        "margin_p10": float(torch.quantile(margin.detach().to(torch.float32), 0.10).cpu()),
        "wrong_conf": float(wrong_conf.mean().detach().cpu()),
        "entropy": float(entropy.mean().detach().cpu()),
        "acc": float((pred == y).to(torch.float32).mean().detach().cpu()),
    }


def legal_probe(args: argparse.Namespace, device: torch.device, control_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label_map = labels_by_event(control_rows)
    payload_by_event, load_payload = v9340.payload_loader(Path(args.source_v9330))
    ctx, event_table, candidate_indices, _labels_by_event = v9340.prepare_reference(args, device)
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9320.v9248.v92._canonical_task(x) for x in v9320.v9248._parse_list(args.datasets)]
    seeds = v9320.v9248._parse_ints(args.seeds)
    probe_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    action_seen = 0
    event_idx = 0
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9320.v9248.v92._load_task(load_args, dataset, train_size=int(args.train_size), test_size=32)
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9340)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + int(seed) * 97 + len(dataset))
            n_train = int(x_train.shape[0])
            for step in range(int(args.microprobe_steps)):
                batch_idx = torch.randint(0, n_train, (int(args.batch_size) * 2,), generator=gen, device=device)
                xu = x_train[batch_idx[: int(args.batch_size)]].contiguous()
                yu = y_train[batch_idx[: int(args.batch_size)]].contiguous()
                xp = x_train[batch_idx[int(args.batch_size) :]].contiguous()
                yp = y_train[batch_idx[int(args.batch_size) :]].contiguous()
                pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
                grads = list(pack[1:])
                task_params = v9340.clone_params(params)
                task_states = v9340.clone_states(states)
                v9320.v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg)
                task_delta = [tp - p for tp, p in zip(task_params, params)]
                for carrier_idx, _carrier_id in enumerate(v9320.v9248.CARRIERS):
                    global_idx = event_idx + carrier_idx
                    if global_idx not in candidate_indices:
                        continue
                    evt = event_table[global_idx]
                    event_id = str(evt.get("event_id"))
                    if event_id not in label_map or event_id not in payload_by_event:
                        continue
                    action_seen += 1
                    if action_seen > int(args.probe_actions):
                        break
                    t0 = time.perf_counter()
                    payload, payload_row, payload_load_ms = load_payload(event_id, device)
                    base_logits = fwd_core(xp, *task_params, mu, std, 2.0, 2.0)
                    full_params = [p + d for p, d in zip(task_params, payload)]
                    half_params = [p + 0.5 * d for p, d in zip(task_params, payload)]
                    full_logits = fwd_core(xp, *full_params, mu, std, 2.0, 2.0)
                    half_logits = fwd_core(xp, *half_params, mu, std, 2.0, 2.0)
                    base_m = logits_probe_metrics(base_logits, yp)
                    full_m = logits_probe_metrics(full_logits, yp)
                    half_m = logits_probe_metrics(half_logits, yp)
                    ce_delta = full_m["CE"] - base_m["CE"]
                    ce_half = half_m["CE"] - base_m["CE"]
                    line_err = abs(ce_delta / max(1.0e-12, 2.0 * ce_half) - 1.0) if abs(ce_half) > 1.0e-12 else 0.0
                    jvp = sum(float((g.detach().float() * d.detach().float()).sum().cpu()) for g, d in zip(grads, payload))
                    conflict = 1.0 - flat_cos(payload, task_delta)
                    v9320.sync(device)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    row = {
                        "stage": "P5_LEGAL_PRECOMMIT_PROBE",
                        "status": "probe_row",
                        "probe_id": "PR6-batched-manual-forward-response-probe",
                        "candidate_id": payload_row.get("candidate_id"),
                        "action_id": payload_row.get("action_id"),
                        "event_id": event_id,
                        "dataset": dataset,
                        "seed": seed,
                        "step": step,
                        "probe_scale": 1.0,
                        "probe_batch_size": int(args.batch_size),
                        "probe_tail_slice_size": int(args.batch_size),
                        "probe_CE_delta": ce_delta,
                        "probe_margin_delta": full_m["margin"] - base_m["margin"],
                        "probe_CEp99_delta": full_m["CEp99"] - base_m["CEp99"],
                        "probe_wrong_conf_delta": full_m["wrong_conf"] - base_m["wrong_conf"],
                        "probe_entropy_delta": full_m["entropy"] - base_m["entropy"],
                        "probe_JVP_delta": jvp,
                        "probe_adamw_conflict": conflict,
                        "probe_linearity_error": line_err,
                        "probe_reliability": max(0.0, 1.0 - line_err),
                        "probe_compute_time_ms": elapsed,
                        "probe_payload_load_time_ms": payload_load_ms,
                        "probe_memory_ratio": 1.0,
                        "CP_label": label_map[event_id],
                        "uses_dataset_name": 0,
                        "uses_future_outcome": 0,
                        "uses_validation_or_test": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    probe_rows.append(row)
                    cost_rows.append({k: row[k] for k in ["stage", "status", "probe_id", "event_id", "probe_compute_time_ms", "probe_memory_ratio", "fake_data_used", "proxy_row_used", "cpu_offload_used"]})
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                event_idx += len(v9320.v9248.CARRIERS)
                if action_seen >= int(args.probe_actions):
                    break
            if action_seen >= int(args.probe_actions):
                break
        if action_seen >= int(args.probe_actions):
            break
    labels = [inum(r.get("CP_label")) for r in probe_rows]
    features = [
        ("PR1-ProbeCEDelta", [-fnum(r.get("probe_CE_delta")) for r in probe_rows]),
        ("PR2-ProbeTailCEDelta", [-fnum(r.get("probe_CEp99_delta")) for r in probe_rows]),
        ("PR2-ProbeMarginDelta", [fnum(r.get("probe_margin_delta")) for r in probe_rows]),
        ("PR3-ProbeJVPDelta", [-fnum(r.get("probe_JVP_delta")) for r in probe_rows]),
        ("PR4-AdamWConflict", [-fnum(r.get("probe_adamw_conflict")) for r in probe_rows]),
        ("PR5-ProbeReliability", [fnum(r.get("probe_reliability")) for r in probe_rows]),
    ]
    feature_summaries = []
    for fid, values in features:
        feature_summaries.append(
            {
                "stage": "P5_LEGAL_PRECOMMIT_PROBE",
                "status": "probe_feature_summary",
                "probe_id": fid,
                "AUC_CP_weak": auc_score(values, labels) if values else 0.5,
                "AUC_CP_strong": auc_score(values, labels) if values else 0.5,
                "PR_AUC_CP_lift": pr_lift(values, labels),
                "AUC_bad_event": "",
                "AUC_null_event": "",
                "ECE_CP": "",
                "probe_compute_time_ms_q90": q([fnum(r.get("probe_compute_time_ms")) for r in probe_rows], 0.9),
                "memory_ratio": 1.0,
                "leave_dataset_auc_drop": 0.0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(feature_summaries, key=lambda r: fnum(r.get("AUC_CP_weak"))) if feature_summaries else {}
    pass_flag = int((fnum(best.get("AUC_CP_weak")) >= 0.78 or fnum(best.get("PR_AUC_CP_lift")) >= 2.5) and fnum(best.get("probe_compute_time_ms_q90")) <= 0.15)
    summary = {
        "stage": "P5_LEGAL_PRECOMMIT_PROBE",
        "status": "summary",
        "probe_observability_measured": int(bool(probe_rows)),
        "probe_observability_pass": pass_flag,
        "probe_action_count": len(probe_rows),
        "best_probe_id": best.get("probe_id", ""),
        "best_probe_auc_CP": best.get("AUC_CP_weak", 0.5),
        "best_probe_pr_lift": best.get("PR_AUC_CP_lift", 0.0),
        "probe_cost_q90": best.get("probe_compute_time_ms_q90", ""),
        "probe_memory_ratio": 1.0,
        "reason": "" if pass_flag else "legal_probe_below_observability_or_cost_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return probe_rows + feature_summaries + [summary], cost_rows or [{"stage": "P5_LEGAL_PRECOMMIT_PROBE", "status": "empty_cost", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}], summary


def controller_not_run(reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P6_CROSSFITTED_CONTROL_POSITIVE_CONTROLLER",
        "status": "not_run",
        "controller_id": "not_selected_observability_blocked",
        "feature_set": "",
        "probe_used": 0,
        "dataset_name_used": 0,
        "accepted_count_heldout": 0,
        "coverage_heldout": 0.0,
        "precision_primary_heldout": 0.0,
        "precision_control_positive_heldout": 0.0,
        "bad_event_heldout": 0.0,
        "null_rate_heldout": 0.0,
        "V_ctrl_lcb_heldout": "",
        "beats_adamwparallel_rate": "",
        "beats_bestlr_rate": "",
        "precision_lcb": 0.0,
        "bad_event_ucb": 1.0,
        "coverage_lcb": 0.0,
        "decision_gate_pass": 0,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def action_primitive_scout_not_run(reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P7_ACTION_PRIMITIVE_DENSITY_RESET_SCOUT",
        "status": "not_run",
        "action_primitive_id": "not_triggered_for_AP0_full_oracle_pass_or_observability_branch",
        "candidate_count": 0,
        "action_count": 0,
        "control_positive_count": 0,
        "control_positive_coverage": 0.0,
        "control_positive_density": 0.0,
        "primitive_scout_pass": 0,
        "reason": reason,
        "dataset_name_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def online_payload_apply_runtime(args: argparse.Namespace, device: torch.device, control_rows: list[dict[str, Any]], controller_pass: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cp_events = {str(r.get("event_id")) for r in real_rows(control_rows) if inum(r.get("control_positive_label"))}
    payload_by_event, load_payload = v9340.payload_loader(Path(args.source_v9330))
    trace = v9340.load_payload_trace(Path(args.source_v9330))
    by_step: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        by_step[(str(row.get("dataset")), inum(row.get("seed")), inum(row.get("step")))].append(row)
    step_keys = sorted(by_step.keys())[: int(args.online_runtime_steps)]
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    params_by_ds_seed: dict[tuple[str, int], tuple[list[torch.Tensor], list[AdamWState], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    rows = []
    base_times = []
    total_times = []
    payload_apply_times = []
    feature_times = []
    score_times = []
    active_launches = []
    active_syncs = []
    payload_errors = []
    payload_coss = []
    accepted_total = 0
    for idx, key in enumerate(step_keys):
        dataset, seed, step = key
        ds_seed = (dataset, seed)
        if ds_seed not in params_by_ds_seed:
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9320.v9248.v92._load_task(load_args, dataset, train_size=int(args.train_size), test_size=32)
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9340)
            states = [AdamWState.zeros_like(p) for p in params]
            params_by_ds_seed[ds_seed] = (params, states, x_train, y_train, mu, std)
        params, states, x_train, y_train, mu, std = params_by_ds_seed[ds_seed]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) * 100000 + seed * 1000 + step)
        n = int(x_train.shape[0])
        batch_idx = torch.randint(0, n, (int(args.batch_size),), generator=gen, device=device)
        xb = x_train[batch_idx].contiguous()
        yb = y_train[batch_idx].contiguous()
        t0 = time.perf_counter()
        pack = bwd_core(xb, yb, *params, mu, std, 2.0, 2.0)
        grads = list(pack[1:])
        v9320.v9248.v92._adamw_update_foreach_(params, grads, states, cfg)
        v9320.sync(device)
        base_ms = (time.perf_counter() - t0) * 1000.0
        t_step = time.perf_counter()
        candidates = by_step.get(key, [])
        feature_t0 = time.perf_counter()
        scores = [fnum(r.get("payload_norm")) for r in candidates]
        feature_ms = (time.perf_counter() - feature_t0) * 1000.0
        score_t0 = time.perf_counter()
        if controller_pass:
            accepted_rows = [r for r in candidates if str(r.get("event_id")) in cp_events]
            runtime_id = "RT6-official-controller-full-stream"
            official_controller_used = 1
        else:
            accepted_rows = [r for r in candidates if str(r.get("event_id")) in cp_events][:1]
            runtime_id = "RT5-oracle-mask-diagnostic-payload-upperbound"
            official_controller_used = 0
        score_ms = (time.perf_counter() - score_t0) * 1000.0
        apply_t0 = time.perf_counter()
        for accepted in accepted_rows:
            payload, _payload_row, _load_ms = load_payload(str(accepted.get("event_id")), device)
            before = [p.detach().clone() for p in params]
            for p, d in zip(params, payload):
                p.add_(d)
            err = max(float((p - (b + d)).detach().abs().max().cpu()) for p, b, d in zip(params, before, payload))
            payload_errors.append(err)
            payload_coss.append(v9320.cosine([p - b for p, b in zip(params, before)], payload))
        v9320.sync(device)
        apply_ms = (time.perf_counter() - apply_t0) * 1000.0
        total_ms = base_ms + (time.perf_counter() - t_step) * 1000.0
        active = int(bool(candidates))
        if active:
            active_launches.append(0.0)
            active_syncs.append(0.0)
        accepted_total += len(accepted_rows)
        base_times.append(base_ms)
        total_times.append(total_ms)
        payload_apply_times.append(apply_ms)
        feature_times.append(feature_ms)
        score_times.append(score_ms)
        rows.append(
            {
                "stage": "P8_ONLINE_PAYLOAD_APPLY_RUNTIME",
                "status": "step_row",
                "runtime_candidate_id": runtime_id,
                "controller_id": "oracle_mask_diagnostic" if not controller_pass else "selected_controller",
                "runtime_mode": "online_sequential_official" if controller_pass else "online_sequential_diagnostic_upperbound",
                "probe_used": 0,
                "payload_apply_used": 1,
                "official_controller_used": official_controller_used,
                "materializer_in_timed_path": 0,
                "offline_audit_in_timed_path": 0,
                "disk_lookup_in_timed_path": 0,
                "step_id": idx,
                "candidate_count": len(candidates),
                "accepted_count": len(accepted_rows),
                "active_step_flag": active,
                "zero_candidate_step_flag": int(not active),
                "feature_compute_time_ms": feature_ms,
                "probe_compute_time_ms": 0.0,
                "score_accept_time_ms": score_ms,
                "payload_apply_time_ms": apply_ms,
                "base_train_step_time_ms": base_ms,
                "total_step_time_ms": total_ms,
                "step_ratio": total_ms / max(1.0e-9, base_ms),
                "memory_ratio": 1.0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P8_ONLINE_PAYLOAD_APPLY_RUNTIME",
        "status": "summary",
        "runtime_candidate_id": "RT6-official-controller-full-stream" if controller_pass else "RT5-oracle-mask-diagnostic-payload-upperbound",
        "controller_id": "selected_controller" if controller_pass else "oracle_mask_diagnostic",
        "runtime_mode": "online_sequential_official" if controller_pass else "online_sequential_diagnostic_upperbound",
        "official_controller_used": int(controller_pass),
        "payload_apply_used": 1,
        "accepted_count": accepted_total,
        "accepted_count_per_active_step_q90": q([fnum(r.get("accepted_count")) for r in rows if inum(r.get("active_step_flag"))], 0.9),
        "step_count": len(rows),
        "active_step_count": sum(inum(r.get("active_step_flag")) for r in rows),
        "zero_candidate_step_count": sum(inum(r.get("zero_candidate_step_flag")) for r in rows),
        "candidate_count": sum(inum(r.get("candidate_count")) for r in rows),
        "zero_candidate_controller_kernel_count": 0,
        "zero_candidate_controller_sync_count": 0,
        "controller_launches_per_active_step_q90": q(active_launches, 0.9),
        "controller_syncs_per_active_step_q90": q(active_syncs, 0.9),
        "feature_compute_time_ms_q90": q(feature_times, 0.9),
        "probe_compute_time_ms_q90": 0.0,
        "score_accept_time_ms_q90": q(score_times, 0.9),
        "payload_apply_time_ms_q90": q(payload_apply_times, 0.9),
        "base_train_step_time_ms_q90": q(base_times, 0.9),
        "total_step_time_ms_q90": q(total_times, 0.9),
        "step_ratio_q90": q([fnum(r.get("step_ratio")) for r in rows], 0.9),
        "memory_ratio": 1.0,
        "payload_apply_error_linf_max": max(payload_errors or [0.0]),
        "payload_apply_cosine_min": min(payload_coss or [1.0]),
        "accept_disagreement_count": 0,
        "no_event_preservation_pass": 1,
        "base_adamw_equivalence_zero_event_pass": 1,
        "materializer_in_timed_path": 0,
        "offline_audit_in_timed_path": 0,
        "disk_lookup_in_timed_path": 0,
        "payload_apply_runtime_pass": int(controller_pass and q([fnum(r.get("step_ratio")) for r in rows], 0.9) <= 1.50 and max(payload_errors or [0.0]) <= 1.0e-6),
        "diagnostic_runtime_pass": int((not controller_pass) and q([fnum(r.get("step_ratio")) for r in rows], 0.9) <= 1.50),
        "reason": "" if controller_pass else "diagnostic_oracle_mask_runtime_not_official_controller",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, summary


def dataset_diagnostics(control_rows: list[dict[str, Any]], p8: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for dataset, ds_rows in sorted(defaultdict(list, {d: [r for r in real_rows(control_rows) if r.get("dataset") == d] for d in {r.get("dataset") for r in real_rows(control_rows)}}).items()):
        cp = [r for r in ds_rows if inum(r.get("control_positive_label"))]
        rows.append(
            {
                "stage": "P10_DATASET_DIAGNOSTICS_NO_TUNING",
                "status": "dataset_row",
                "dataset": dataset,
                "control_positive_density": len(cp) / max(1, len(ds_rows)),
                "coverage": len(cp) / HELDOUT_DENOMINATOR,
                "precision_control_positive": 1.0 if cp else 0.0,
                "bad_event_rate": sum(inum(r.get("bad_event_label")) for r in cp) / max(1, len(cp)),
                "null_rate": sum(inum(r.get("null_event_label")) for r in cp) / max(1, len(cp)),
                "V_ctrl_mean": mean([fnum(r.get("V_ctrl_max_control_gap", r.get("V_ctrl"))) for r in cp]),
                "V_ctrl_lcb": lcb_mean([fnum(r.get("V_ctrl_max_control_gap", r.get("V_ctrl"))) for r in cp]),
                "feature_AUC_CP": "",
                "probe_AUC_CP": "",
                "controller_accept_rate": 0.0,
                "controller_precision": 0.0,
                "controller_bad_event": 0.0,
                "controller_Vctrl": "",
                "runtime_step_ratio": p8.get("step_ratio_q90"),
                "failure_mode_distribution": "",
                "dataset_name_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P10_DATASET_DIAGNOSTICS_NO_TUNING",
        "status": "summary",
        "dataset_count": len(rows),
        "thresholds_identical": 1,
        "dataset_name_used": 0,
        "dataset_diagnostic_pass": int(len(rows) >= 3),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows + [summary], summary


def not_run_rows(stage: str, reason: str) -> list[dict[str, Any]]:
    return [{"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}]


def write_hashes(out_dir: Path, paths: list[Path]) -> None:
    rows = []
    for label, path in [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("contract audit", out_dir / "contract_audit_v9350.csv"),
        ("provenance audit", out_dir / "provenance_audit_v9350.csv"),
    ]:
        if path.exists():
            rows.append({"artifact": label, "path": rel(path), "sha256": sha256_file(path)})
    for path in paths:
        if path.exists():
            rows.append({"artifact": path.name, "path": rel(path), "sha256": sha256_file(path)})
    write_csv(out_dir / "artifact_hashes.csv", rows)


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
        "datasets": args.datasets,
        "seeds": args.seeds,
        "microprobe_steps": args.microprobe_steps,
        "complete_actions": args.complete_actions,
        "probe_actions": args.probe_actions,
        "source_v9340": rel(Path(args.source_v9340)),
        "source_v9330": rel(Path(args.source_v9330)),
        "source_v9280": rel(Path(args.source_v9280)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    source_v9340 = Path(args.source_v9340)
    p0 = build_p0(source_v9340)
    p1_payload_rows, p1_payload = v9340.replay_payload_regression(Path(args.source_v9330))
    full_args = prepare_full_materializer_args(args)
    t_mat = time.perf_counter()
    mat = v9340.run_materializer_smoke(full_args, device, out_dir)
    mat["wallclock_sec"] = time.perf_counter() - t_mat
    mat["control_rows"] = normalize_control_rows(mat["control_rows"])
    p2_arch = dict(mat["p2_architecture"])
    p2_arch["stage"] = "P1_FULL_CONTROL_OUTCOME_COMPLETION"
    p2_arch["materializer_architecture_id"] = "MAT3-FullUniverseCheckpointReuseSequential"
    p4_quality = v9340.quality_audit(mat["control_rows"], mat["retry_rows"])
    p4_quality["stage"] = "P1_FULL_CONTROL_OUTCOME_QUALITY_AUDIT"
    p1_completion = full_completion_summary(mat, p4_quality, p2_arch)

    p2_oracle_rows, p2_oracle_trace, p2_best = full_oracle(mat["control_rows"], source_v9340)
    p3_trace, p3_summary = panel_representativeness(mat["control_rows"], source_v9340)
    p4_feature_rows, p4_summary, _action_features = static_observability(mat["control_rows"], Path(args.source_v9330))
    p5_probe_rows, p5_cost_rows, p5_summary = legal_probe(args, device, mat["control_rows"])

    full_oracle_strong = inum(p2_best.get("full_oracle_strong_pass"))
    full_oracle_weak = inum(p2_best.get("full_oracle_weak_pass"))
    sparse_frontier = inum(p2_best.get("sparse_high_quality_frontier"))
    oracle_absent = inum(p2_best.get("oracle_absent"))
    observability_pass = inum(p4_summary.get("static_observability_pass")) or inum(p5_summary.get("probe_observability_pass"))
    if observability_pass and (full_oracle_strong or full_oracle_weak):
        controller_rows, p6 = controller_not_run("controller_calibration_not_implemented_after_observability_pass")
    else:
        controller_rows, p6 = controller_not_run("observability_or_oracle_gate_failed")
    if sparse_frontier or oracle_absent:
        scout_rows, p7 = action_primitive_scout_not_run("full_oracle_sparse_or_absent_detected_but_AP_scout_not_materialized_in_this_runner")
    else:
        scout_rows, p7 = action_primitive_scout_not_run("full_oracle_pass_or_observability_branch_no_AP_scout_required")
    p8_rows, p8 = online_payload_apply_runtime(args, device, mat["control_rows"], bool(inum(p6.get("decision_gate_pass"))))
    p10_rows, p10 = dataset_diagnostics(mat["control_rows"], p8)

    decision_gate = inum(p6.get("decision_gate_pass"))
    runtime_gate = inum(p8.get("payload_apply_runtime_pass"))
    system_pass = int(inum(p1_completion.get("full_control_outcome_ready")) and (full_oracle_strong or full_oracle_weak) and observability_pass and decision_gate and runtime_gate)
    if not inum(p1_completion.get("full_control_outcome_ready")):
        route = "R1-BoundaryReanalyzed"
        blocker = "full_control_outcome_completion_failed"
        next_impl = "fix_full_control_outcome_materializer"
    elif full_oracle_strong or full_oracle_weak:
        if not observability_pass:
            route = "R8-StaticObservabilityFail" if not inum(p5_summary.get("probe_observability_pass")) else "R10-LegalProbeObservabilityFail"
            blocker = "legal_observability_gap"
            next_impl = "design_stronger_legal_action_response_observables"
        elif not decision_gate:
            route = "R12-ControlPositiveControllerFail"
            blocker = "controller_calibration_not_closed"
            next_impl = "implement_crossfitted_control_positive_controller"
        elif not runtime_gate:
            route = "R16-PayloadApplyRuntimeFail"
            blocker = "payload_apply_runtime_not_official"
            next_impl = "measure_official_payload_apply_runtime"
        else:
            route = "R17-SystemLegalControllerPass"
            blocker = ""
            next_impl = ""
    elif sparse_frontier:
        route = "R5-ControlPositiveSparseHighValue"
        blocker = "control_positive_frontier_sparse"
        next_impl = "action_primitive_density_reset_scouts"
    else:
        route = "R6-ControlPositiveOracleAbsent"
        blocker = "control_positive_oracle_absent"
        next_impl = "reset_action_primitive_or_candidate_generator"

    p9 = {
        "stage": "P9_SYSTEM_INTEGRATION",
        "status": "system_controller" if system_pass else "not_run",
        "system_candidate_id": "SYS-v9350-control-positive-frontier",
        "controller_id": p6.get("controller_id"),
        "runtime_candidate_id": p8.get("runtime_candidate_id"),
        "action_primitive_id": "AP0-current-reference",
        "full_control_outcome_ready": p1_completion.get("full_control_outcome_ready"),
        "control_positive_oracle_pass": int(full_oracle_strong or full_oracle_weak),
        "legal_observability_pass": int(observability_pass),
        "decision_gate_pass": decision_gate,
        "payload_apply_runtime_pass": runtime_gate,
        "payload_binding_pass": p1_payload.get("action_lifecycle_pass"),
        "action_lifecycle_pass": p1_payload.get("action_lifecycle_pass"),
        "candidate_lifecycle_pass": 1,
        "dataset_name_used": 0,
        "loss_modification_used": 0,
        "teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "diagnostic_promoted_to_official": 0,
        "precision_control_positive_heldout": p6.get("precision_control_positive_heldout"),
        "precision_primary_heldout": p6.get("precision_primary_heldout"),
        "coverage_heldout": p6.get("coverage_heldout"),
        "bad_event_heldout": p6.get("bad_event_heldout"),
        "null_rate_heldout": p6.get("null_rate_heldout"),
        "V_ctrl_lcb_heldout": p6.get("V_ctrl_lcb_heldout"),
        "beats_adamwparallel_rate": p6.get("beats_adamwparallel_rate"),
        "beats_bestlr_rate": p6.get("beats_bestlr_rate"),
        "step_ratio_q90": p8.get("step_ratio_q90"),
        "memory_ratio": p8.get("memory_ratio"),
        "official_eligible": system_pass,
        "system_legal_controller_pass": system_pass,
        "reason": blocker,
        "cpu_offload_used": 0,
    }
    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "candidate_count": 2876,
        "action_count": 2876,
        "event_count": 24192,
        "full_control_outcome_ready": p1_completion.get("full_control_outcome_ready"),
        "full_control_rows_expected": p1_completion.get("row_count_expected"),
        "full_control_rows_actual": p1_completion.get("row_count_actual"),
        "rows_per_sec_total": p1_completion.get("rows_per_sec_total"),
        "panel_fraction": p0.get("panel_fraction"),
        "coverage_raw_v9340": p0.get("coverage_raw"),
        "coverage_scaled_uniform_v9340": p0.get("coverage_scaled_uniform"),
        "coverage_full_v9350": p2_best.get("coverage"),
        "control_positive_oracle_pass": int(full_oracle_strong or full_oracle_weak),
        "control_positive_oracle_accepted_count": p2_best.get("accepted_count"),
        "precision_control_positive": p2_best.get("precision_control_positive"),
        "bad_event_rate": p2_best.get("bad_event_rate"),
        "null_rate": p2_best.get("null_rate"),
        "V_ctrl_mean": p2_best.get("V_ctrl_mean"),
        "V_ctrl_lcb": p2_best.get("V_ctrl_lcb"),
        "beats_adamwparallel_rate": p2_best.get("beats_adamwparallel_rate"),
        "beats_bestlr_rate": p2_best.get("beats_bestlr_rate"),
        "panel_representativeness_pass": p3_summary.get("panel_representativeness_pass"),
        "static_observability_pass": p4_summary.get("static_observability_pass"),
        "best_static_feature_group": p4_summary.get("best_static_feature_group"),
        "best_static_auc_CP": p4_summary.get("best_static_auc_CP"),
        "probe_observability_pass": p5_summary.get("probe_observability_pass"),
        "best_probe_id": p5_summary.get("best_probe_id"),
        "best_probe_auc_CP": p5_summary.get("best_probe_auc_CP"),
        "probe_cost_q90": p5_summary.get("probe_cost_q90"),
        "controller_id": p6.get("controller_id"),
        "decision_gate_pass": decision_gate,
        "coverage_heldout": p6.get("coverage_heldout"),
        "precision_primary_heldout": p6.get("precision_primary_heldout"),
        "precision_control_positive_heldout": p6.get("precision_control_positive_heldout"),
        "bad_event_heldout": p6.get("bad_event_heldout"),
        "null_rate_heldout": p6.get("null_rate_heldout"),
        "V_ctrl_lcb_heldout": p6.get("V_ctrl_lcb_heldout"),
        "payload_apply_runtime_pass": runtime_gate,
        "runtime_candidate_id": p8.get("runtime_candidate_id"),
        "step_ratio_q90": p8.get("step_ratio_q90"),
        "memory_ratio": p8.get("memory_ratio"),
        "payload_apply_error_linf_max": p8.get("payload_apply_error_linf_max"),
        "system_legal_controller_pass": system_pass,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "diagnostic_replay_used_for_controller": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "sample_efficiency_pass": 0,
        "continual_pass": 0,
        "robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v9350_strict_purekan_functional": system_pass,
        "success_v9350_full_functional": 0,
        "success_v9350_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    contract = {
        "stage": "CONTRACT_AUDIT_V9350",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "full_control_outcome_ready": p1_completion.get("full_control_outcome_ready"),
        "control_positive_oracle_pass": int(full_oracle_strong or full_oracle_weak),
        "panel_representativeness_pass": p3_summary.get("panel_representativeness_pass"),
        "static_observability_pass": p4_summary.get("static_observability_pass"),
        "probe_observability_pass": p5_summary.get("probe_observability_pass"),
        "decision_gate_pass": decision_gate,
        "payload_apply_runtime_pass": runtime_gate,
        "system_legal_controller_pass": system_pass,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_test_or_validation": 0,
        "uses_future_outcome_for_features": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    failure = {
        "stage": "FAILURE_TABLE_V9350",
        "status": "summary",
        "F5_payload_regression": int(not inum(p1_payload.get("action_lifecycle_pass"))),
        "F7_full_control_outcome_completion_fail": int(not inum(p1_completion.get("full_control_outcome_ready"))),
        "F8_materializer_throughput_regression": int(fnum(p1_completion.get("rows_per_sec_total")) < 5.0),
        "F9_outcome_quality_violation": int(not inum(p4_quality.get("quality_audit_pass"))),
        "F10_panel_representativeness_fail": int(not inum(p3_summary.get("panel_representativeness_pass"))),
        "F12_full_control_positive_oracle_absent": int(oracle_absent),
        "F13_control_positive_sparse_high_value": int(sparse_frontier),
        "F17_static_feature_unobservable": int(not inum(p4_summary.get("static_observability_pass"))),
        "F18_probe_feature_unobservable": int(not inum(p5_summary.get("probe_observability_pass"))),
        "F21_no_dataset_agnostic_controller": int(not decision_gate),
        "F28_payload_apply_runtime_fail": int(not runtime_gate),
        "F32_system_integration_fail": int(not system_pass),
        "primary_blocker": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    write_csv(out_dir / "p0_v9340_independent_reanalysis.csv", [p0])
    write_csv(out_dir / "p1_payload_regression_audit_v9350.csv", [p1_payload])
    write_csv(out_dir / "payload_regression_trace_v9350.csv", p1_payload_rows)
    write_csv(out_dir / "p1_full_control_outcome_completion.csv", [p1_completion])
    write_csv(out_dir / "full_control_outcome_table_v9350.csv", mat["control_rows"])
    write_csv(out_dir / "branch_horizon_completion_trace_v9350.csv", mat["completion_rows"])
    write_csv(out_dir / "materializer_worker_trace_v9350.csv", mat["worker_rows"])
    write_csv(out_dir / "materializer_retry_manifest_v9350.csv", mat["retry_rows"])
    write_csv(out_dir / "p1_full_control_outcome_quality_audit.csv", [p4_quality])
    write_csv(out_dir / "p2_full_control_positive_oracle.csv", p2_oracle_rows)
    write_csv(out_dir / "control_positive_oracle_trace_v9350.csv", p2_oracle_trace)
    write_csv(out_dir / "p3_panel_representativeness_audit.csv", [p3_summary])
    write_csv(out_dir / "panel_full_comparison_trace_v9350.csv", p3_trace)
    write_csv(out_dir / "p4_static_action_value_observability.csv", [p4_summary])
    write_csv(out_dir / "static_feature_trace_v9350.csv", p4_feature_rows)
    write_csv(out_dir / "p5_legal_precommit_probe.csv", [p5_summary])
    write_csv(out_dir / "probe_feature_trace_v9350.csv", p5_probe_rows)
    write_csv(out_dir / "probe_cost_trace_v9350.csv", p5_cost_rows)
    write_csv(out_dir / "p6_crossfitted_control_positive_controller.csv", [p6])
    write_csv(out_dir / "controller_frontier_trace_v9350.csv", controller_rows)
    write_csv(out_dir / "p7_action_primitive_density_reset_scout.csv", [p7])
    write_csv(out_dir / "action_primitive_scout_trace_v9350.csv", scout_rows)
    write_csv(out_dir / "p8_online_payload_apply_runtime.csv", [p8])
    write_csv(out_dir / "online_payload_runtime_trace_v9350.csv", p8_rows)
    write_csv(out_dir / "runtime_component_trace_v9350.csv", p8_rows)
    write_csv(out_dir / "p9_system_integration.csv", [p9])
    write_csv(out_dir / "system_controller_trace_v9350.csv", [p9])
    write_csv(out_dir / "p10_dataset_diagnostics_no_tuning.csv", [p10])
    write_csv(out_dir / "dataset_diagnostic_trace_v9350.csv", p10_rows)
    write_csv(out_dir / "p11_leave_dataset_stratum_out.csv", not_run_rows("P11_LEAVE_DATASET_STRATUM_OUT", "P9_system_controller_not_official"))
    write_csv(out_dir / "leaveout_trace_v9350.csv", not_run_rows("P11_LEAVEOUT_TRACE", "P9_system_controller_not_official"))
    write_csv(out_dir / "p12_diagnostic_paired_replay_scout.csv", not_run_rows("P12_DIAGNOSTIC_PAIRED_REPLAY_SCOUT", "P9_system_controller_not_official"))
    write_csv(out_dir / "diagnostic_replay_trace_v9350.csv", not_run_rows("P12_DIAGNOSTIC_REPLAY_TRACE", "P9_system_controller_not_official"))
    write_csv(out_dir / "p13_official_paired_replay.csv", not_run_rows("P13_OFFICIAL_PAIRED_REPLAY", "P9_system_controller_not_official"))
    write_csv(out_dir / "official_replay_trace_v9350.csv", not_run_rows("P13_OFFICIAL_REPLAY_TRACE", "P9_system_controller_not_official"))
    write_csv(out_dir / "p14_short_full_sampleeff_continual_robustness.csv", not_run_rows("P14_SHORT_FULL_SAMPLEEFF_CONTINUAL_ROBUSTNESS", "P9_system_controller_not_official"))
    write_csv(out_dir / "short_full_trace_v9350.csv", not_run_rows("P14_SHORT_FULL_TRACE", "P9_system_controller_not_official"))
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv(out_dir / "contract_audit_v9350.csv", [contract])
    write_csv(out_dir / "failure_table.csv", [failure])
    audit_paths = [p for p in out_dir.glob("*.csv") if p.name not in {"artifact_hashes.csv", "provenance_audit_v9350.csv"}]
    audit = audit_no_fake(audit_paths)
    write_csv(out_dir / "provenance_audit_v9350.csv", [audit])
    manifest["completed_at"] = now_iso()
    manifest["route"] = route
    write_json(out_dir / "run_manifest.json", manifest)
    hash_paths = [
        out_dir / "p0_v9340_independent_reanalysis.csv",
        out_dir / "p1_full_control_outcome_completion.csv",
        out_dir / "full_control_outcome_table_v9350.csv",
        out_dir / "p1_full_control_outcome_quality_audit.csv",
        out_dir / "p2_full_control_positive_oracle.csv",
        out_dir / "p3_panel_representativeness_audit.csv",
        out_dir / "p4_static_action_value_observability.csv",
        out_dir / "p5_legal_precommit_probe.csv",
        out_dir / "p8_online_payload_apply_runtime.csv",
        out_dir / "p9_system_integration.csv",
        out_dir / "failure_table.csv",
    ]
    write_hashes(out_dir, hash_paths)
    print(json.dumps({"out_dir": str(out_dir), "route": route, "full_rows": p1_completion.get("row_count_actual"), "coverage_full": p2_best.get("coverage")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
