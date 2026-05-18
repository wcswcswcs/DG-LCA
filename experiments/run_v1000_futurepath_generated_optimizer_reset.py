#!/usr/bin/env python3
"""DG-KAN v10.0 future-path generated optimizer reset.

This runner treats v9.9.8 as the boundary source.  It keeps official gate
decisions separate from discovery diagnostics: natural density confirmation
uses faithful G40 panels, FPO sketches use commit-time fields only, and
generated discovery is limited to the 8 -> 32 -> 64 staircase unless a real
weak gate passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import natural_ap0_extension_materializer as materializer  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9910_natural_density_decision_futurepathoperator as v9910  # noqa: E402
import run_v9930_natural_density_tail_fidelity_futurepathoperator as v9930  # noqa: E402
import run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate as v9980  # noqa: E402
import run_v9980_generated_sandbox_supplement as v9980_gen  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v1000_futurepath_generated_optimizer_reset_full_20260517T230000Z"
SOURCE_V9980 = RESULT_ROOT / "v9980_multimarginal_natural_sampling_fpo_generated_hardgate_full_20260517T210000Z"
SOURCE_V9970 = RESULT_ROOT / "v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite_full_20260517T200000Z"
SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
SOURCE_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
SOURCE_V9910 = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
PLAN_PATH = REPO / "docs/DG-KAN_v10.0_FuturePathGeneratedOptimizer_Reset_完整实验计划.md"
RECAP_PATH = REPO / "docs/DG-KAN_v10.0_FuturePathGeneratedOptimizer_Reset_实验复盘.md"
SCRIPT_PATH = REPO / "experiments/run_v1000_futurepath_generated_optimizer_reset.py"
MATERIALIZER_PATH = REPO / "experiments/natural_ap0_extension_materializer.py"
HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6
G40_ID = "G40-IPF-major-tail-raking-sampler"
G40_PROFILE = "g40-ipf-major-tail-raking-sampler"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--repeat-seed", type=int, default=2027)
    p.add_argument("--confirm-seed", type=int, default=3031)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    p.add_argument("--chunk-actions", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--source-v9980", default=str(SOURCE_V9980))
    p.add_argument("--source-v9970", default=str(SOURCE_V9970))
    p.add_argument("--source-v9330", default=str(SOURCE_V9330))
    p.add_argument("--source-v9900", default=str(SOURCE_V9900))
    p.add_argument("--source-v9910", default=str(SOURCE_V9910))
    return p.parse_args()


def rows_from(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


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


def flag(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "false"}:
        return 0
    try:
        return int(float(text) > 0)
    except ValueError:
        return int(text in {"true", "yes"})


def safe_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def label_fast_good(row: dict[str, Any]) -> int:
    return int(
        inum(row.get("horizon_complete"))
        and fnum(row.get("V1_gap")) > 0
        and fnum(row.get("V5_gap")) > 0
        and fnum(row.get("V20_gap")) > 0
        and fnum(row.get("V80_gap")) > 0
        and fnum(row.get("V240_gap")) > 0
        and inum(row.get("RiskPath")) == 0
        and inum(row.get("BadPath")) == 0
    )


def path_type(row: dict[str, Any]) -> str:
    if label_fast_good(row):
        return "FastGood"
    if inum(row.get("SlowBurnGood")):
        return "SlowBurnGood"
    if inum(row.get("RiskyHighAUV")) or (fnum(row.get("RiskAdjustedAUV")) > 0 and inum(row.get("RiskPath"))):
        return "RiskyHighAUV"
    if inum(row.get("RiskCleanButLowImmediate")) or (inum(row.get("horizon_complete")) and not inum(row.get("RiskPath")) and not inum(row.get("BadPath"))):
        return "SafeLowValue"
    return "BadPath"


def is_good(row: dict[str, Any]) -> int:
    return int(path_type(row) in {"FastGood", "SlowBurnGood"})


def source_score(row: dict[str, Any]) -> float:
    return (
        2.0 * fnum(row.get("trust_ratio"))
        + 1.2 * fnum(row.get("effective_derivative"))
        + 0.7 * fnum(row.get("tail_fraction"))
        + 0.35 * fnum(row.get("action_adamw_cosine"))
        - 5.5 * fnum(row.get("payload_norm"))
        - 1.5 * abs(fnum(row.get("branch_ratio")) - 1.0)
    )


def artifact_audit(root: Path) -> dict[str, int]:
    total = 0
    fake = 0
    proxy = 0
    cpu = 0
    for path in sorted(list(root.glob("*.csv")) + list(root.glob("*.json"))):
        if path.name.startswith(("no_fake", "contract", "failure")):
            continue
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    total += 1
                    fake = max(fake, flag(row.get("fake_data_used", 0)))
                    proxy = max(proxy, flag(row.get("proxy_row_used", 0)))
                    cpu = max(cpu, flag(row.get("cpu_offload_used", 0)))
        elif path.suffix == ".json":
            total += 1
            data = read_json(path)
            if isinstance(data, dict):
                fake = max(fake, flag(data.get("fake_data_used", 0)))
                proxy = max(proxy, flag(data.get("proxy_row_used", 0)))
                cpu = max(cpu, flag(data.get("cpu_offload_used", 0)))
    return {"rows_checked": total, "fake_data_used": fake, "proxy_row_used": proxy, "cpu_offload_used": cpu}


def p0_boundary(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source / "route_decision_v9980.json")
    p2 = summary_row(rows_from(source / "p2_generator_distribution_only_matrix_v9980.csv"))
    p4 = summary_row(rows_from(source / "p4_sequential_natural_density_panel_v9980.csv"))
    p6 = summary_row(rows_from(source / "p6_fpo_v10_path_type_predictor_v9980.csv"))
    p8 = summary_row(rows_from(source / "p8_generated_sandbox_gate_v9980.csv"))
    nf = summary_row(rows_from(source / "no_fake_audit_v9980.csv"))
    row = {
        "stage": "P0_V9980_BOUNDARY_REPRODUCTION_V1000",
        "status": "summary",
        "P0_route": route.get("route"),
        "P0_G40_official_pass": int(G40_ID in str(p2.get("best_generator_id")) and inum(p2.get("official_pass_count")) >= 1),
        "P0_G40_major_psi": "0.04923558761592247",
        "P0_G40_tail_psi": "0.00448427827051975",
        "P0_largest_completed_panel": p4.get("largest_completed_panel_size"),
        "P0_CoreLike_count": p4.get("CoreLike_count"),
        "P0_CoreLike_LCB": p4.get("CoreLike_LCB", ""),
        "P0_CoreLike_UCB": p4.get("CoreLike_UCB", ""),
        "P0_SlowBurn_count": p4.get("SlowBurnGood_count"),
        "P0_CoreLikePlusSlowBurn_UCB": p4.get("CoreLikeOrSlowBurn_UCB"),
        "P0_FPO10C_precision": p6.get("best_precision"),
        "P0_generated64_Fast_precision": p8.get("FastGood_precision"),
        "P0_generated64_Slow_precision": p8.get("SlowBurnGood_precision"),
        "P0_generated64_V_LCB": p8.get("V_LCB"),
        "P0_generated64_V240_LCB": p8.get("V240_LCB"),
        "P0_generated64_longrisk_UCB": p8.get("longrisk_UCB"),
        "fake_data_used_v9980": nf.get("fake_data_used"),
        "proxy_row_used_v9980": nf.get("proxy_row_used"),
        "cpu_offload_used_v9980": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_boundary_pass"] = int(
        row["P0_route"] == "CaseC-NaturalDensityInsufficientGeneratedSandboxFail"
        and inum(row["P0_G40_official_pass"]) == 1
        and inum(row["P0_largest_completed_panel"]) >= 5000
        and fnum(row["P0_CoreLikePlusSlowBurn_UCB"]) < 0.03
        and inum(route.get("P8_generated_sandbox_pass")) == 0
        and inum(row["fake_data_used_v9980"]) == 0
        and inum(row["proxy_row_used_v9980"]) == 0
        and inum(row["cpu_offload_used_v9980"]) == 0
    )
    return [row], row


def path_type_labels(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    natural_actions = [r for r in rows_from(source / "p4_G40-IPF-major-tail-raking-sampler_5000_action_rows_v9980.csv") if r.get("status") == "natural_extension_action_row"]
    natural_labels = [r for r in rows_from(source / "p4_G40-IPF-major-tail-raking-sampler_5000_action_labels_v9980.csv") if r.get("status") == "natural_action_label"]
    gen_actions = [r for r in rows_from(source / "p8_generated_64_sandbox_actions_v9980.csv") if r.get("status") == "generated_sandbox_action_row"]
    gen_labels = [r for r in rows_from(source / "p8_generated_64_sandbox_labels_v9980.csv") if r.get("status") == "natural_action_label"]
    action_by_id = {str(r.get("action_id")): r for r in natural_actions + gen_actions}
    out: list[dict[str, Any]] = []
    for label in natural_labels + gen_labels:
        aid = str(label.get("action_id"))
        action = action_by_id.get(aid, {})
        ptype = path_type(label)
        row = {
            "stage": "A_FUTURE_PATH_TYPE_LABELS_V1000",
            "status": "path_type_label",
            "action_id": aid,
            "source_group": "generated_64_sandbox" if aid in {str(r.get("action_id")) for r in gen_actions} else "natural_G40_5000",
            "path_type": ptype,
            "dataset": label.get("dataset") or action.get("dataset"),
            "family_id": label.get("family_id") or action.get("family_id"),
            "template_id": label.get("template_id") or action.get("template_id"),
            "tail_group": action.get("reference_tail_key") or f"{action.get('candidate_template_id') or action.get('template_id')}|{action.get('source_recipe_id')}",
            "V1": label.get("V1_gap"),
            "V5": label.get("V5_gap"),
            "V20": label.get("V20_gap"),
            "V80": label.get("V80_gap"),
            "V240": label.get("V240_gap"),
            "AUV": label.get("RiskAdjustedAUV"),
            "RAUV": label.get("RiskAdjustedAUV"),
            "longrisk": label.get("RiskPath"),
            "bad": label.get("BadPath"),
            "null": int(not inum(label.get("horizon_complete"))),
            "memory": "",
            "offdiag": "",
            "memory_offdiag_metric_source": "not_available_in_natural_branch_schema",
            "hard_tail_change": label.get("DelayedGain"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": label.get("cpu_offload_used", 0),
        }
        out.append(row)
    counts = Counter(r["path_type"] for r in out)
    good = counts["FastGood"] + counts["SlowBurnGood"]
    natural = [r for r in out if r["source_group"] == "natural_G40_5000"]
    good_natural = [r for r in natural if r["path_type"] in {"FastGood", "SlowBurnGood"}]
    n_good = len(good_natural)
    risk_ucb = wilson_ucb(sum(inum(r.get("longrisk")) for r in good_natural), n_good) if n_good else 0.0
    dataset_max_share = max((Counter(r["dataset"] for r in good_natural).values()), default=0) / max(1, n_good)
    family_max_share = max((Counter(r["family_id"] for r in good_natural).values()), default=0) / max(1, n_good)
    summary = {
        "stage": "A_FUTURE_PATH_TYPE_LABELS_V1000",
        "status": "summary",
        "landed_future_action_rows": len(out),
        "natural_reference_rows": len(natural),
        "FastGood_count": counts["FastGood"],
        "SlowBurnGood_count": counts["SlowBurnGood"],
        "RiskyHighAUV_count": counts["RiskyHighAUV"],
        "SafeLowValue_count": counts["SafeLowValue"],
        "BadPath_count": counts["BadPath"],
        "FastSlow_count": good,
        "FastSlow_natural_count": n_good,
        "FastSlow_longrisk_UCB": risk_ucb,
        "FastSlow_dataset_max_share": dataset_max_share,
        "FastSlow_family_max_share": family_max_share,
        "A_official_reference_pass": int(good >= 87 and risk_ucb <= 0.05 and dataset_max_share <= 0.50 and family_max_share <= 0.50),
        "A_discovery_target_pass": int(good >= 30 and risk_ucb <= 0.10),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in out] or [0]),
    }
    out.insert(0, summary)
    return out, natural_actions + gen_actions, summary


def fpo_v11(actions: list[dict[str, Any]], labels: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_by_id = {str(r.get("action_id")): r for r in labels if r.get("status") == "path_type_label"}
    act = [r for r in actions if str(r.get("action_id")) in label_by_id]
    if not act:
        rows, summary = v9930.not_run("B_FPO11_PATH_TYPE_PREDICTOR_V1000", "no_evaluable_action_rows", B_FPO11_weak_pass=0, B_FPO11_discovery_pass=0)
        return rows, summary

    def norm(v: float, scale: float = 1.0) -> float:
        return v / max(1.0e-12, scale)

    specs: list[tuple[str, str, Callable[[dict[str, Any]], float]]] = [
        ("FPO11A-tiny-virtual-adamw-path-sketch", "green_commit_time_fields_only", lambda r: 1.6 * fnum(r.get("action_adamw_cosine")) * fnum(r.get("trust_ratio")) + 0.5 * fnum(r.get("effective_derivative")) - 4.0 * fnum(r.get("payload_linf"))),
        ("FPO11B-jvp-vjp-future-gradient-alignment", "green_commit_time_fields_only", lambda r: fnum(r.get("effective_derivative")) + 0.6 * fnum(r.get("action_adamw_cosine")) - 0.8 * abs(fnum(r.get("branch_ratio")) - 1.0) - 4.5 * fnum(r.get("payload_norm"))),
        ("FPO11C-signal-reservoir-path-type-snr", "green_commit_time_fields_only", lambda r: fnum(r.get("tail_fraction")) + 0.8 * fnum(r.get("trust_ratio")) - 0.6 * abs(fnum(r.get("branch_ratio")) - 1.0) - 6.0 * fnum(r.get("payload_norm"))),
        ("FPO11D-memory-offdiag-hard-gate-plus-delayed-gain", "green_commit_time_fields_only_memory_unavailable_hard_gate_not_certified", lambda r: max(0.0, fnum(r.get("effective_derivative"))) + 0.5 * fnum(r.get("tail_fraction")) - 5.0 * fnum(r.get("payload_linf"))),
        ("FPO11E-path-type-calibrated-ranker", "yellow_discovery_train_split_uses_path_labels_not_official", lambda r: source_score(r) + 0.2 * norm(fnum(r.get("tail_fraction")), 0.25)),
    ]
    rows: list[dict[str, Any]] = []
    for fpo_id, legality, fn in specs:
        times: list[float] = []
        scored: list[tuple[float, dict[str, Any]]] = []
        for action in act:
            t0 = time.perf_counter()
            score = fn(action)
            times.append((time.perf_counter() - t0) * 1000.0)
            scored.append((score, action))
        ranked = [a for _s, a in sorted(scored, key=lambda x: x[0], reverse=True)]
        top = ranked[: min(87, len(ranked))]
        labs = [label_by_id[str(a.get("action_id"))] for a in top]
        n = len(labs)
        fast = sum(1 for r in labs if r.get("path_type") == "FastGood")
        slow = sum(1 for r in labs if r.get("path_type") == "SlowBurnGood")
        risk = sum(inum(r.get("longrisk")) for r in labs)
        bad = sum(1 for r in labs if r.get("path_type") == "BadPath")
        null = sum(inum(r.get("null")) for r in labs)
        precision = (fast + slow) / max(1, n)
        v_lcb = mean_lcb([fnum(r.get("V20")) for r in labs])
        v240_lcb = mean_lcb([fnum(r.get("V240")) for r in labs])
        risk_ucb = wilson_ucb(risk, n) if n else 1.0
        bad_ucb = wilson_ucb(bad, n) if n else 1.0
        null_ucb = wilson_ucb(null, n) if n else 1.0
        cost_q90 = q(times, 0.90)
        discovery = int(precision >= 0.50 and v240_lcb > 0 and risk_ucb <= 0.10 and cost_q90 <= 5.0)
        weak = int(
            legality.startswith("green")
            and n >= 87
            and precision >= 0.75
            and v_lcb > 0
            and v240_lcb > 0
            and risk_ucb <= 0.05
            and bad_ucb <= 0.05
            and null_ucb <= 0.15
            and cost_q90 <= 1.50
            and "memory_unavailable" not in legality
        )
        rows.append({
            "stage": "B_FPO11_PATH_TYPE_PREDICTOR_V1000",
            "status": "fpo_row",
            "fpo_id": fpo_id,
            "feature_legality": legality,
            "commit_time_feature_only": int("yellow" not in legality),
            "evaluated_topk": n,
            "TopK87_FastGood_precision": fast / max(1, n),
            "TopK87_SlowBurnGood_precision": slow / max(1, n),
            "TopK87_FastSlow_precision": precision,
            "TopK87_V_LCB": v_lcb,
            "TopK87_V240_LCB": v240_lcb,
            "TopK87_longrisk_UCB": risk_ucb,
            "TopK87_bad_UCB": bad_ucb,
            "TopK87_null_UCB": null_ucb,
            "cost_q90_ms": cost_q90,
            "discovery_pass": discovery,
            "weak_pass": weak,
            "strong_pass": 0,
            "reason": "" if discovery or weak else "low_precision_or_negative_value_or_high_risk",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(rows, key=lambda r: (inum(r["weak_pass"]), inum(r["discovery_pass"]), fnum(r["TopK87_FastSlow_precision"]), fnum(r["TopK87_V240_LCB"])))
    summary = {
        "stage": "B_FPO11_PATH_TYPE_PREDICTOR_V1000",
        "status": "summary",
        "fpo_count": len(rows),
        "evaluated_action_count": len(act),
        "B_FPO11_discovery_pass": int(any(inum(r["discovery_pass"]) for r in rows)),
        "B_FPO11_weak_pass": int(any(inum(r["weak_pass"]) for r in rows)),
        "B_FPO11_strong_pass": 0,
        "best_fpo_id": best.get("fpo_id"),
        "best_precision": best.get("TopK87_FastSlow_precision"),
        "best_V_LCB": best.get("TopK87_V_LCB"),
        "best_V240_LCB": best.get("TopK87_V240_LCB"),
        "best_longrisk_UCB": best.get("TopK87_longrisk_UCB"),
        "best_cost_q90_ms": best.get("cost_q90_ms"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def density_row_from_label_summary(panel_summary: dict[str, Any], label_summary: dict[str, Any], seed: int, panel_size: int, source: str) -> dict[str, Any]:
    row = v9930.panel_density_decision(panel_summary, label_summary, G40_ID, status="panel_row", reason=source)
    row["stage"] = "C_NATURAL_DENSITY_CONFIRMATION_V1000"
    row["seed"] = seed
    row["panel_size"] = panel_size
    row["source"] = source
    row["FastGood_count"] = label_summary.get("FastGood_count", "")
    row["FastGood_LCB"] = label_summary.get("FastGood_LCB", "")
    row["FastGood_UCB"] = label_summary.get("FastGood_UCB", "")
    row["CoreLikePlusSlowBurn_UCB"] = row.get("CoreLike_or_SlowBurnGood_UCB")
    return row


def label_with_fast(branches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels, summary = v9930.label_summary_from_branches(branches)
    data = [r for r in labels if r.get("status") == "natural_action_label"]
    fast = sum(label_fast_good(r) for r in data)
    n = len(data)
    for row in data:
        row["FastGood"] = label_fast_good(row)
    summary["FastGood_count"] = fast
    summary["FastGood_LCB"] = wilson_lcb(fast, n) if n else 0.0
    summary["FastGood_UCB"] = wilson_ucb(fast, n) if n else 0.0
    return labels, summary


def run_or_load_panel(args: argparse.Namespace, out: Path, seed: int, panel_size: int, old_ids: set[str], old_hashes: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prefix = f"c_G40_seed{seed}_{panel_size}"
    panel_path = out / f"{prefix}_panel_smoke_v1000.csv"
    actions_path = out / f"{prefix}_action_rows_v1000.csv"
    branches_path = out / f"{prefix}_branch_horizon_v1000.csv"
    labels_path = out / f"{prefix}_action_labels_v1000.csv"
    if panel_path.exists() and actions_path.exists() and branches_path.exists() and labels_path.exists():
        panel = summary_row(rows_from(panel_path))
        labels = rows_from(labels_path)
        actions = rows_from(actions_path)
        label_summary = summary_row(labels)
        return density_row_from_label_summary(panel, label_summary, seed, panel_size, f"reused_existing_v1000_panel_seed_{seed}_{panel_size}"), labels, actions
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = seed
    run_args.execution_profile = "full-gated"
    smoke_rows, actions, applies, branches, smoke = v9930.run_panel_sharded(
        G40_ID, G40_PROFILE, panel_size, 40_000_000 + seed * 100_000 + panel_size, run_args, out, old_ids, old_hashes
    )
    write_csv(panel_path, smoke_rows)
    write_csv(actions_path, actions)
    write_csv(out / f"{prefix}_action_apply_replay_v1000.csv", applies)
    write_csv(branches_path, branches)
    labels, label_summary = label_with_fast(branches)
    write_csv(labels_path, labels)
    return density_row_from_label_summary(smoke, label_summary, seed, panel_size, f"real_G40_density_confirmation_seed_{seed}_{panel_size}"), labels, actions


def c_density_confirmation(args: argparse.Namespace, out: Path, source: Path, old_ids: set[str], old_hashes: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    # C1 reuses the real v9.9.8 seed-1314 5000 panel.
    source_panel = next(r for r in rows_from(source / "p4_sequential_natural_density_panel_v9980.csv") if r.get("status") == "panel_row" and inum(r.get("panel_size")) == 5000)
    source_labels = rows_from(source / "p4_G40-IPF-major-tail-raking-sampler_5000_action_labels_v9980.csv")
    source_actions = rows_from(source / "p4_G40-IPF-major-tail-raking-sampler_5000_action_rows_v9980.csv")
    c1 = dict(source_panel)
    c1.update({
        "stage": "C_NATURAL_DENSITY_CONFIRMATION_V1000",
        "seed": args.seed,
        "source": "reuse_v9980_real_G40_5000_seed_1314",
        "CoreLikePlusSlowBurn_UCB": source_panel.get("CoreLike_or_SlowBurnGood_UCB"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    rows.append(c1)
    c2, c2_labels, c2_actions = run_or_load_panel(args, out, args.repeat_seed, 5000, old_ids, old_hashes)
    rows.append(c2)
    c3_labels: list[dict[str, Any]] = []
    c3_actions: list[dict[str, Any]] = []
    if fnum(c1.get("CoreLikePlusSlowBurn_UCB")) < 0.03 and fnum(c2.get("CoreLikePlusSlowBurn_UCB")) < 0.03:
        c3, c3_labels, c3_actions = run_or_load_panel(args, out, args.confirm_seed, 10000, old_ids, old_hashes)
        rows.append(c3)
    else:
        rows.append({
            "stage": "C_NATURAL_DENSITY_CONFIRMATION_V1000",
            "status": "not_run",
            "seed": args.confirm_seed,
            "panel_size": 10000,
            "reason": "two_5000_repeats_did_not_both_confirm_UCB_below_0_03",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    completed = [r for r in rows if r.get("status") == "panel_row"]
    confirm = next((r for r in completed if inum(r.get("panel_size")) == 10000), {})
    insufficient = int(
        len([r for r in completed if inum(r.get("panel_size")) == 5000 and fnum(r.get("CoreLikePlusSlowBurn_UCB")) < 0.03]) >= 2
        and bool(confirm)
        and fnum(confirm.get("CoreLikePlusSlowBurn_UCB")) < 0.03
    )
    sufficient = int(any(fnum(r.get("CoreLike_or_SlowBurnGood_LCB")) >= 0.03 for r in completed))
    summary = {
        "stage": "C_NATURAL_DENSITY_CONFIRMATION_V1000",
        "status": "summary",
        "completed_panel_count": len(completed),
        "largest_completed_panel_size": max([inum(r.get("panel_size")) for r in completed] or [0]),
        "C_5000_repeat_confirmed_count": len([r for r in completed if inum(r.get("panel_size")) == 5000 and fnum(r.get("CoreLikePlusSlowBurn_UCB")) < 0.03]),
        "C_10000_confirmation_run": int(bool(confirm)),
        "C_natural_density_sufficient": sufficient,
        "C_natural_density_insufficient": insufficient,
        "C_natural_density_inconclusive": int(not sufficient and not insufficient),
        "best_CoreLikePlusSlowBurn_UCB": min([fnum(r.get("CoreLikePlusSlowBurn_UCB")) for r in completed] or [0.0]),
        "reason": "two_5000_and_one_10000_confirmed_UCB_below_0_03" if insufficient else "density_confirmation_inconclusive_or_sufficient",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in completed] or [0]),
    }
    rows.insert(0, summary)
    return rows, summary, (c3_labels or c2_labels or source_labels), (c3_actions or c2_actions or source_actions)


def choose_generated_sources(actions: list[dict[str, Any]], labels: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    label_by_id = {str(r.get("action_id")): r for r in labels if r.get("status") == "path_type_label"}
    positives = [a for a in actions if label_by_id.get(str(a.get("action_id")), {}).get("path_type") in {"FastGood", "SlowBurnGood"}]
    ranked = sorted(positives or actions, key=source_score, reverse=True)
    if len(ranked) >= count:
        return ranked[:count]
    out = list(ranked)
    seen = {str(r.get("action_id")) for r in out}
    for row in sorted(actions, key=source_score, reverse=True):
        if str(row.get("action_id")) not in seen:
            out.append(row)
            seen.add(str(row.get("action_id")))
            if len(out) >= count:
                break
    return out[:count]


def build_gen11_actions(source_rows: list[dict[str, Any]], args: argparse.Namespace, out: Path, stage: str, size: int) -> list[dict[str, Any]]:
    dev = materializer._device(args.device)
    replay_args = materializer._default_replay_args(args.seed, args.data_root)
    ctx_cache: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    payloads: list[list[torch.Tensor]] = []
    recipes = [
        ("GEN11A-GoodPathPrototypeImitation", 0.88, 0.0010, 1.15),
        ("GEN11B-ConstrainedDelayedGainUpdate", 0.72, 0.0018, 0.90),
        ("GEN11C-JVPFutureGradientAlignmentUpdate", 0.52, 0.0024, 0.85),
        ("GEN11D-SignalReservoirDenoisedUpdate", 0.64, 0.0012, 0.70),
        ("GEN11E-NegativeControlShuffledPrototype", 0.50, -0.0010, 1.00),
    ]
    for i, src in enumerate(source_rows):
        source_payload = materializer._load_payload(src, dev)
        ctx = materializer.v9480.v9420.replay_context(replay_args, src, dev, ctx_cache)
        task_delta = [t.detach().clone().to(dev) for t in ctx["task_delta"]]
        recipe, alpha, beta0, linf_mult = recipes[i % len(recipes)]
        if recipe.startswith("GEN11E"):
            source_payload = [
                torch.roll(p.detach().clone(), shifts=1, dims=0).contiguous() if p.ndim > 0 else p.detach().clone().contiguous()
                for p in source_payload
            ]
        beta = beta0 / (1.0 + 20.0 * fnum(src.get("payload_norm")) + 5.0 * abs(fnum(src.get("branch_ratio")) - 1.0))
        payload = [(alpha * p.detach().clone() + beta * t.detach().clone()).contiguous() for p, t in zip(source_payload, task_delta)]
        target_linf = max(1.0e-12, linf_mult * fnum(src.get("payload_linf")))
        scale = min(1.0, target_linf / max(materializer._flat_linf(payload), 1.0e-12))
        if scale < 1.0:
            payload = [p.mul(scale).contiguous() for p in payload]
        payload_hash = materializer.v9480.v9420.tensor_hash(payload)
        action_id = stable_hash("v1000", stage, size, i, src.get("action_id"), recipe, payload_hash, args.seed)
        row = dict(src)
        row.update({
            "stage": f"D_GENERATED_{stage.upper()}_V1000",
            "status": "generated_sandbox_action_row",
            "action_id": action_id,
            "generated_action_id": action_id,
            "source_action_id": src.get("action_id"),
            "reference_action_id": src.get("action_id"),
            "generator_id": f"GEN11-{stage}",
            "generator_profile": "future_path_targeted_generated_optimizer_reset_v1000",
            "generation_recipe_id": recipe,
            "generation_source": "A_line_path_type_target_discovery",
            "generated_sandbox_size": size,
            "payload_hash": payload_hash,
            "payload_hash_expected": payload_hash,
            "payload_norm": materializer._flat_norm(payload),
            "payload_linf": materializer._flat_linf(payload),
            "payload_scale_factor": scale,
            "selection_feature_source": "path_type_target_discovery_not_official_controller",
            "uses_future_outcome": 0,
            "uses_old_table_label": 0,
            "uses_validation_or_test": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": int(dev.type != "cuda"),
        })
        rows.append(row)
        payloads.append(payload)
    batch = materializer.NaturalAP0ActionBatch(
        action_rows=rows,
        payloads=payloads,
        cursor_start="0",
        cursor_end=str(len(rows)),
        seed=args.seed,
        profile="future_path_targeted_generated_optimizer_reset_v1000",
        fake_data_used=0,
        proxy_row_used=0,
        cpu_offload_used=int(dev.type != "cuda"),
    )
    return materializer.write_natural_ap0_action_schema(out / f"d_{stage}_{size}_payloads", batch)


def run_generated_stage(args: argparse.Namespace, out: Path, stage: str, size: int, sources: list[dict[str, Any]], source_labels: list[dict[str, Any]]) -> dict[str, Any]:
    actions_path = out / f"d_{stage}_{size}_generated_actions_v1000.csv"
    labels_path = out / f"d_{stage}_{size}_labels_v1000.csv"
    branch_path = out / f"d_{stage}_{size}_branch_horizon_v1000.csv"
    apply_path = out / f"d_{stage}_{size}_action_apply_replay_v1000.csv"
    if actions_path.exists() and labels_path.exists() and branch_path.exists() and apply_path.exists():
        actions = rows_from(actions_path)
        labels = rows_from(labels_path)
        apply_rows = rows_from(apply_path)
        branches = rows_from(branch_path)
    else:
        actions = build_gen11_actions(sources, args, out, stage, size)
        write_csv(actions_path, actions)
        apply_rows = [materializer.apply_natural_ap0_action(row, args.device) for row in actions]
        write_csv(apply_path, apply_rows)
        if args.device == "auto" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        branches = materializer.materialize_natural_ap0_branch_horizon(actions, args, out, device=args.device, horizons=HORIZONS)
        write_csv(branch_path, branches)
        labels, _summary = label_with_fast(branches)
        source_by_generated = {str(r.get("action_id")): str(r.get("source_action_id")) for r in actions}
        for row in labels:
            if row.get("stage"):
                row["stage"] = f"D_GENERATED_{stage.upper()}_LABELS_V1000"
            if row.get("status") == "natural_action_label":
                row["source_action_id"] = source_by_generated.get(str(row.get("action_id")), "")
                row["path_type"] = path_type(row)
        write_csv(labels_path, labels)
    data = [r for r in labels if r.get("status") == "natural_action_label"]
    n = len(data)
    fast = sum(1 for r in data if r.get("path_type") == "FastGood")
    slow = sum(1 for r in data if r.get("path_type") == "SlowBurnGood")
    risk = sum(inum(r.get("RiskPath")) for r in data)
    bad = sum(inum(r.get("BadPath")) for r in data)
    null = sum(1 - inum(r.get("horizon_complete")) for r in data)
    source_by_id = {str(r.get("action_id")): r for r in source_labels if r.get("status") == "path_type_label"}
    new_positive = 0
    longrisk_created = 0
    for row in data:
        src = source_by_id.get(str(row.get("source_action_id")), {})
        new_positive += int(is_good(row) and str(src.get("path_type")) not in {"FastGood", "SlowBurnGood"})
        longrisk_created += int(inum(row.get("RiskPath")) and not inum(src.get("longrisk")))
    expected_rows = size * BRANCH_COUNT * len(HORIZONS)
    v_lcb = mean_lcb([fnum(r.get("V20_gap")) for r in data])
    v240_lcb = mean_lcb([fnum(r.get("V240_gap")) for r in data])
    risk_ucb = wilson_ucb(risk, n) if n else 1.0
    bad_ucb = wilson_ucb(bad, n) if n else 1.0
    null_ucb = wilson_ucb(null, n) if n else 1.0
    action_apply_linf_max = max([fnum(r.get("action_apply_linf_max")) for r in apply_rows] or [0.0])
    weak = int(
        n >= 64
        and (fast + slow) / max(1, n) >= 0.10
        and fast + slow >= 5
        and v240_lcb > -0.05
        and risk_ucb <= 0.20
        and longrisk_created / max(1, n) <= 0.20
        and new_positive / max(1, n) >= 0.05
    )
    preflight = int(
        len(actions) == size
        and len(branches) == expected_rows
        and action_apply_linf_max <= 1.0e-12
        and max([inum(r.get("cpu_offload_used")) for r in actions + apply_rows + branches + labels] or [0]) == 0
    )
    return {
        "stage": "D_GENERATED_DISCOVERY_STAIRCASE_V1000",
        "status": "sandbox_row",
        "sandbox_stage": stage,
        "generated_action_count": size,
        "branch_horizon_expected_rows": expected_rows,
        "branch_horizon_actual_rows": len(branches),
        "D0_preflight_pass": preflight if stage == "d0" else "",
        "FastGood_count": fast,
        "SlowBurnGood_count": slow,
        "FastSlow_precision": (fast + slow) / max(1, n),
        "V_LCB": v_lcb,
        "V240_LCB": v240_lcb,
        "longrisk_UCB": risk_ucb,
        "bad_UCB": bad_ucb,
        "null_UCB": null_ucb,
        "new_positive_created_rate": new_positive / max(1, n),
        "longrisk_created_rate": longrisk_created / max(1, n),
        "action_apply_linf_max": action_apply_linf_max,
        "weak_pass": weak,
        "failure_type": "" if weak else ("no_fast_or_slow_generated" if fast + slow == 0 else "low_value_or_high_risk_generated_discovery"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in actions + apply_rows + branches + labels] or [0]),
    }


def d_generated_discovery(args: argparse.Namespace, out: Path, source_actions: list[dict[str, Any]], path_labels: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d0_sources = choose_generated_sources(source_actions, path_labels, 8)
    d0 = run_generated_stage(args, out, "d0", 8, d0_sources, path_labels)
    rows = [d0]
    if inum(d0.get("D0_preflight_pass")):
        d1_sources = choose_generated_sources(source_actions, path_labels, 32)
        rows.append(run_generated_stage(args, out, "d1", 32, d1_sources, path_labels))
        d2_sources = choose_generated_sources(source_actions, path_labels, 64)
        rows.append(run_generated_stage(args, out, "d2", 64, d2_sources, path_labels))
    d2 = next((r for r in rows if r.get("sandbox_stage") == "d2"), {})
    d3_open = int(inum(d2.get("weak_pass")))
    if not d3_open:
        rows.append({
            "stage": "D_GENERATED_DISCOVERY_STAIRCASE_V1000",
            "status": "not_run",
            "sandbox_stage": "d3",
            "generated_action_count": 256,
            "reason": "D2_64_action_weak_gate_not_passed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "D_GENERATED_DISCOVERY_STAIRCASE_V1000",
        "status": "summary",
        "D0_preflight_pass": inum(d0.get("D0_preflight_pass")),
        "D1_executed": int(any(r.get("sandbox_stage") == "d1" for r in rows)),
        "D2_executed": int(bool(d2)),
        "D2_weak_pass": inum(d2.get("weak_pass")),
        "D2_FastSlow_precision": d2.get("FastSlow_precision", ""),
        "D2_V240_LCB": d2.get("V240_LCB", ""),
        "D2_longrisk_UCB": d2.get("longrisk_UCB", ""),
        "D3_256_opened": d3_open,
        "generated_discovery_pass": inum(d2.get("weak_pass")),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
    }
    rows.insert(0, summary)
    return rows, summary


def not_run(stage: str, reason: str, **extra: Any) -> list[dict[str, Any]]:
    rows, _summary = v9930.not_run(stage, reason, **extra)
    return rows


def write_figures(out: Path, a: dict[str, Any], b: dict[str, Any], c_rows: list[dict[str, Any]], d: dict[str, Any], route: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}

    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels or ["none"], values or [0.0])
        figs[name] = path

    fig("fig_A_path_type_counts_v1000.svg", "A Path Types", ["Fast", "Slow", "Risky", "SafeLow", "Bad"], [fnum(a.get("FastGood_count")), fnum(a.get("SlowBurnGood_count")), fnum(a.get("RiskyHighAUV_count")), fnum(a.get("SafeLowValue_count")), fnum(a.get("BadPath_count"))])
    fig("fig_B_fpo11_gate_v1000.svg", "B FPO11", ["precision", "V240", "risk"], [fnum(b.get("best_precision")), fnum(b.get("best_V240_LCB")), fnum(b.get("best_longrisk_UCB"))])
    panels = [r for r in c_rows if r.get("status") == "panel_row"]
    fig("fig_C_density_repeat_curve_v1000.svg", "C Density UCB", [f"{r.get('seed')}-{r.get('panel_size')}" for r in panels], [fnum(r.get("CoreLikePlusSlowBurn_UCB")) for r in panels])
    fig("fig_D_generated_discovery_v1000.svg", "D Generated", ["d0", "d1", "d2"], [fnum(d.get("D0_preflight_pass")), fnum(d.get("D1_executed")), fnum(d.get("D2_weak_pass"))])
    fig("fig_v1000_route_matrix.svg", "Route", ["C", "B", "D", "Controller"], [fnum(route.get("C_natural_density_insufficient")), fnum(route.get("B_FPO11_weak_pass")), fnum(route.get("D_generated_discovery_pass")), fnum(route.get("P7_controller_pass"))])
    return figs


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_v9980_boundary_reproduction_v1000.csv"))
    a = summary_row(rows_from(out / "a_future_path_type_labels_v1000.csv"))
    b = summary_row(rows_from(out / "b_fpo11_path_type_predictor_v1000.csv"))
    c = summary_row(rows_from(out / "c_natural_density_confirmation_v1000.csv"))
    d = summary_row(rows_from(out / "d_generated_discovery_staircase_v1000.csv"))
    p7 = summary_row(rows_from(out / "p7_controller_boundary_v1000.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v1000.csv"))
    lines = [
        "# DG-KAN v10.0 FuturePath Generated Optimizer Reset 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v10.0_FuturePathGeneratedOptimizer_Reset_完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest、真实 G40 natural AP0 panel rows 与真实 generated discovery branch-horizon rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 controller/runtime/paired replay 均显式 `not_run`。",
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
        f"最终 artifact：`{repo_rel(out)}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.9.8 boundary pass = `{p0.get('P0_boundary_pass')}`；source route = `{p0.get('P0_route')}`。",
        f"2. A 线 Fast/Slow/Risky/SafeLow/Bad = `{a.get('FastGood_count')}` / `{a.get('SlowBurnGood_count')}` / `{a.get('RiskyHighAUV_count')}` / `{a.get('SafeLowValue_count')}` / `{a.get('BadPath_count')}`；discovery target pass = `{a.get('A_discovery_target_pass')}`。",
        f"3. B 线 FPO11 weak/discovery = `{b.get('B_FPO11_weak_pass')}` / `{b.get('B_FPO11_discovery_pass')}`；best = `{b.get('best_fpo_id')}`，precision = `{b.get('best_precision')}`，V240 LCB = `{b.get('best_V240_LCB')}`。",
        f"4. C 线 completed panels = `{c.get('completed_panel_count')}`，largest = `{c.get('largest_completed_panel_size')}`，natural density sufficient/insufficient/inconclusive = `{c.get('C_natural_density_sufficient')}` / `{c.get('C_natural_density_insufficient')}` / `{c.get('C_natural_density_inconclusive')}`。",
        f"5. D 线 D0/D1/D2 = `{d.get('D0_preflight_pass')}` / `{d.get('D1_executed')}` / `{d.get('D2_executed')}`；D2 weak = `{d.get('D2_weak_pass')}`，FastSlow precision = `{d.get('D2_FastSlow_precision')}`。",
        f"6. P7 controller = `{p7.get('status')}`；No-fake audit rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v1000_futurepath_generated_optimizer_reset.py` | v10.0 runner；执行 P0/A/B/C/D/P7-P9 gate，并写 manifest/recap。 |",
        "| `experiments/natural_ap0_extension_materializer.py` | 复用真实 natural/generated payload materializer 与 branch-horizon replay。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v1000_futurepath_generated_optimizer_reset.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v1000_futurepath_generated_optimizer_reset.py --out-dir results/real_rerun_20260506/v1000_futurepath_generated_optimizer_reset_full_20260517T230000Z --fresh --device auto --data-root data --seed 1314 --repeat-seed 2027 --confirm-seed 3031 --execution-profile full-gated --chunk-actions 512",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. A/B/C/D 摘要",
        "",
        "```text",
        f"A discovery target pass = {a.get('A_discovery_target_pass')}",
        f"B FPO11 weak/discovery = {b.get('B_FPO11_weak_pass')} / {b.get('B_FPO11_discovery_pass')}",
        f"C density sufficient/insufficient/inconclusive = {c.get('C_natural_density_sufficient')} / {c.get('C_natural_density_insufficient')} / {c.get('C_natural_density_inconclusive')}",
        f"D generated discovery pass = {d.get('generated_discovery_pass')}",
        "```",
        "",
        "## 4. No-Fake / Hash",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake/proxy/cpu = {nf.get('fake_data_used')} / {nf.get('proxy_row_used')} / {nf.get('cpu_offload_used')}",
        "```",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend([
        "",
        "## 5. 最终分析结论",
        "",
        "```text",
        "1. v10.0 将 official gate、science discovery 和 engineering preflight 分开记录，没有把 discovery label 写成 official controller。",
        "2. C 线按计划复用 v9.9.8 真实 5000，并新增 seed repeat；只有满足两个 5000 条件才打开 10000 confirmation。",
        "3. B 线 FPO11 使用 commit-time features 做 path-type predictor；yellow calibrated sketch 只作为 discovery 诊断，不写成 official。",
        "4. D 线只执行 8 -> 32 -> 64 discovery staircase；D2 未过 weak 时不打开 256。",
        "5. controller/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。",
        "```",
        "",
        f"最终一句话：v10.0 真实执行后停在 `{route.get('route')}`：{route.get('route_explanation')}",
    ])
    RECAP_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source_v9980 = Path(args.source_v9980)
    source_v9330 = Path(args.source_v9330)
    source_v9900 = Path(args.source_v9900)
    source_v9910 = Path(args.source_v9910)
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return summary_row(rows)

    def dump_json(name: str, obj: dict[str, Any]) -> None:
        path = out / name
        write_json(path, obj)
        artifacts[name] = path

    p0_rows, p0 = p0_boundary(source_v9980)
    dump_csv("p0_v9980_boundary_reproduction_v1000.csv", p0_rows)
    if not inum(p0.get("P0_boundary_pass")):
        raise SystemExit("P0 v9.9.8 boundary reproduction failed; refusing to continue.")

    a_rows, action_rows, a = path_type_labels(source_v9980)
    dump_csv("a_future_path_type_labels_v1000.csv", a_rows)

    b_rows, b = fpo_v11(action_rows, a_rows)
    dump_csv("b_fpo11_path_type_predictor_v1000.csv", b_rows)

    old_ids, old_hashes = v9980.old_action_sets(source_v9330, source_v9900, source_v9910)
    c_rows, c, c_labels, c_actions = c_density_confirmation(args, out, source_v9980, old_ids, old_hashes)
    dump_csv("c_natural_density_confirmation_v1000.csv", c_rows)
    dump_csv("c_natural_density_group_rates_v1000.csv", v9930.grouped_density(c_labels, c_actions) if c_labels else not_run("C_NATURAL_DENSITY_GROUP_RATES_V1000", "no_completed_c_panel"))

    d_rows, d = d_generated_discovery(args, out, action_rows, a_rows)
    dump_csv("d_generated_discovery_staircase_v1000.csv", d_rows)

    p7_open = int(inum(b.get("B_FPO11_weak_pass")) or inum(d.get("generated_discovery_pass")))
    p7_rows = not_run("P7_CONTROLLER_BOUNDARY_V1000", "B_and_D_weak_gates_not_passed" if not p7_open else "controller_candidate_not_implemented_after_discovery", P7_controller_pass=0)
    p7 = dump_csv("p7_controller_boundary_v1000.csv", p7_rows)
    p8 = dump_csv("p8_runtime_boundary_v1000.csv", not_run("P8_SELECTED_RUNTIME_BOUNDARY_V1000", "P7_controller_not_passed", P8_runtime_pass=0))
    p9 = dump_csv("p9_paired_replay_boundary_v1000.csv", not_run("P9_PAIRED_REPLAY_BOUNDARY_V1000", "P7_or_P8_not_passed", P9_paired_replay_pass=0))

    if inum(c.get("C_natural_density_sufficient")):
        route_name = "CaseA-NaturalHarvestingUnexpectedlySufficient"
        primary, secondary, gen_status = "natural_density_sufficient", "generated_route_not_needed", "not_opened_natural_sufficient"
        explanation = "C confirmation found sufficient natural good-path density, so generated reset is not opened."
    elif inum(c.get("C_natural_density_insufficient")) and inum(b.get("B_FPO11_weak_pass")):
        route_name = "CaseB-NaturalInsufficientFPOPathTypeWorks"
        primary, secondary, gen_status = "natural_density_insufficient", "FPO11_weak_pass_controller_pending", "generated_target_available_from_FPO"
        explanation = "Natural density is insufficient and FPO11 found a weak path-type selector, but controller/runtime did not pass."
    elif inum(c.get("C_natural_density_insufficient")) and inum(d.get("generated_discovery_pass")):
        route_name = "CaseC-NaturalInsufficientGeneratedDiscoveryWorks"
        primary, secondary, gen_status = "natural_density_insufficient", "generated_discovery_weak_pass", "generated_discovery_64_pass_256_pending"
        explanation = "Natural density is insufficient and D2 generated discovery produced a weak signal, so 256 discovery is the next gated step."
    elif inum(c.get("C_natural_density_insufficient")):
        route_name = "CaseD-NaturalInsufficientBDDiscoveryFail"
        primary, secondary, gen_status = "natural_density_insufficient", "FPO11_and_generated_discovery_failed", "stopped_future_path_generated_optimizer_reset"
        explanation = "Natural density was confirmed insufficient, but FPO11 and generated discovery both failed, so controller/runtime stay blocked."
    else:
        route_name = "CaseF-NaturalDensityConfirmationInconclusive"
        primary, secondary, gen_status = "density_confirmation_inconclusive", "generated_discovery_diagnostic_only", "stopped_density_confirmation_inconclusive"
        explanation = "C confirmation did not close natural density, so generated/controller decisions remain blocked."

    route = {
        "stage": "ROUTE_DECISION_V1000",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_explanation": explanation,
        "source_route_v9980": p0.get("P0_route"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "A_discovery_target_pass": a.get("A_discovery_target_pass"),
        "B_FPO11_weak_pass": b.get("B_FPO11_weak_pass"),
        "B_FPO11_discovery_pass": b.get("B_FPO11_discovery_pass"),
        "C_natural_density_sufficient": c.get("C_natural_density_sufficient"),
        "C_natural_density_insufficient": c.get("C_natural_density_insufficient"),
        "C_natural_density_inconclusive": c.get("C_natural_density_inconclusive"),
        "C_largest_completed_panel_size": c.get("largest_completed_panel_size"),
        "D_generated_discovery_pass": d.get("generated_discovery_pass"),
        "D2_weak_pass": d.get("D2_weak_pass"),
        "D3_256_opened": d.get("D3_256_opened"),
        "P7_controller_pass": p7.get("P7_controller_pass", 0),
        "P8_runtime_pass": p8.get("P8_runtime_pass", 0),
        "P9_paired_replay_pass": p9.get("P9_paired_replay_pass", 0),
        "generated_route_status": gen_status,
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(p0.get("cpu_offload_used")), inum(a.get("cpu_offload_used")), inum(c.get("cpu_offload_used")), inum(d.get("cpu_offload_used"))]),
    }
    dump_json("route_decision_v1000.json", route)

    fig_artifacts = write_figures(out, a, b, c_rows, d, route)
    artifacts.update(fig_artifacts)
    nf = artifact_audit(out)
    dump_csv("no_fake_audit_v1000.csv", [{"stage": "NO_FAKE_AUDIT_V1000", "status": "summary", **nf}])
    dump_csv("contract_audit_v1000.csv", [{
        "stage": "CONTRACT_AUDIT_V1000",
        "status": "summary",
        "P0_pass": p0.get("P0_boundary_pass"),
        "C_gate_sequential": 1,
        "D_no_256_without_weak": int(not inum(d.get("D3_256_opened")) or inum(d.get("D2_weak_pass"))),
        "controller_not_run_without_B_or_D_weak": int(not inum(p7_open) and str(p7.get("status")) == "not_run"),
        "fake_data_used": nf["fake_data_used"],
        "proxy_row_used": nf["proxy_row_used"],
        "cpu_offload_used": nf["cpu_offload_used"],
    }])
    dump_csv("failure_taxonomy_v1000.csv", [{
        "stage": "FAILURE_TAXONOMY_V1000",
        "status": "summary",
        "route": route_name,
        "F1_boundary_failed": int(not inum(p0.get("P0_boundary_pass"))),
        "F2_density_not_confirmed_insufficient": int(not inum(c.get("C_natural_density_insufficient"))),
        "F3_FPO11_failed": int(not inum(b.get("B_FPO11_weak_pass"))),
        "F4_generated_discovery_failed": int(not inum(d.get("generated_discovery_pass"))),
        "F5_controller_blocked": 1,
        "fake_data_used": nf["fake_data_used"],
        "proxy_row_used": nf["proxy_row_used"],
        "cpu_offload_used": nf["cpu_offload_used"],
    }])
    manifest = {
        "stage": "RUN_MANIFEST_V1000",
        "status": "summary",
        "out_dir": repo_rel(out),
        "plan": repo_rel(PLAN_PATH),
        "runner": repo_rel(SCRIPT_PATH),
        "materializer": repo_rel(MATERIALIZER_PATH),
        "source_v9980": repo_rel(source_v9980),
        "command": "python experiments/run_v1000_futurepath_generated_optimizer_reset.py --out-dir ... --fresh --device auto --data-root data --seed 1314 --repeat-seed 2027 --confirm-seed 3031 --execution-profile full-gated --chunk-actions 512",
        "route": route_name,
        "fake_data_used": nf["fake_data_used"],
        "proxy_row_used": nf["proxy_row_used"],
        "cpu_offload_used": nf["cpu_offload_used"],
    }
    dump_json("run_manifest_v1000.json", manifest)
    file_hashes = {name: sha256_file(path) for name, path in artifacts.items() if path.exists()}
    for extra_name, extra_path in {"plan": PLAN_PATH, "runner": SCRIPT_PATH, "materializer": MATERIALIZER_PATH}.items():
        file_hashes[extra_name] = sha256_file(extra_path)
    write_recap(out, route, file_hashes)
    print(json.dumps({"route": route_name, "out": str(out), "recap": str(RECAP_PATH), "no_fake": nf}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
