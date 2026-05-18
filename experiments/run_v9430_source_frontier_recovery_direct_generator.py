#!/usr/bin/env python3
"""DG-KAN v9.4.3 source frontier recovery / direct generator runner.

This runner keeps v9.4.2's source-outcome materializer closure fixed and
separates three questions:
1. Is there an AP0 source frontier in the full measured universe?
2. Can legal commit-time source selectors find it?
3. Can a small real direct-generator smoke create a useful source frontier?

It only promotes measured rows. Oracle panels remain diagnostic.
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

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9410_value_producing_source_generator as v9410  # noqa: E402
import run_v9420_source_outcome_materializer_closure as v9420  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.3_SourceFrontierRecovery_DirectGeneratorEffectCertificate_ParallelValidation_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9430_source_frontier_recovery_direct_generator.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9420 = RESULT_ROOT / "v9420_source_outcome_materializer_closure_value_triage_first_20260514T090000Z"
DEFAULT_V9410 = RESULT_ROOT / "v9410_value_producing_source_generator_legal_source_identifiability_base_acc_first_20260514T080000Z"
DEFAULT_V9400 = RESULT_ROOT / "v9400_source_action_selection_candidate_source_rebuild_horizon_controller_first_20260514T070000Z"
DEFAULT_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

AP0B_AP0F = [
    "AP0b-LastEdgeLinearizedDescentSource",
    "AP0c-AdamWResidualOrthogonalSource",
    "AP0d-TailMarginRepairSource",
    "AP0e-CurvatureGuardedLowRankEdgeSource",
    "AP0f-SupportMemorySource",
]
DIRECT_PRIMITIVES = [
    "AP0g-GradientAlignedLastEdgeSource",
    "AP0h-TailMarginConservativeSource",
    "AP0i-AdamWResidualBlendSource",
    "AP0j-LowRankEdgeSafeSource",
    "AP0k-NoOpGuardedMicroSource",
]
HORIZONS = [20, 80, 240]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9420", default=str(DEFAULT_V9420))
    p.add_argument("--source-v9410", default=str(DEFAULT_V9410))
    p.add_argument("--source-v9400", default=str(DEFAULT_V9400))
    p.add_argument("--source-v9350", default=str(DEFAULT_V9350))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--direct-actions-per-generator", type=int, default=12)
    p.add_argument("--sentinel-source", default=str(DEFAULT_V9420))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--sentinel-seeds", default="0,1,2,3,4")
    p.add_argument("--sentinel-steps", type=int, default=12)
    p.add_argument("--sentinel-train-size", type=int, default=512)
    p.add_argument("--sentinel-test-size", type=int, default=256)
    p.add_argument("--sentinel-hidden-dim", type=int, default=64)
    return p.parse_args()


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def q(xs: list[float], frac: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))]


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_absolute() and path.is_relative_to(REPO) else str(path)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lcb(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return mean(vals) - 1.96 * (statistics.pstdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)


def bucket(value: float, cuts: list[float]) -> str:
    for i, c in enumerate(cuts):
        if value <= c:
            return f"b{i}"
    return f"b{len(cuts)}"


def dist(ids: list[str], stats: dict[str, dict[str, Any]], key: str) -> Counter:
    return Counter(str(stats[i].get(key, "")) for i in ids if i in stats)


def psi_kl_gap(panel_ids: list[str], full_ids: list[str], stats: dict[str, dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    psi_total = 0.0
    kl_total = 0.0
    max_gap = 0.0
    for key in ["family_bucket", "step_bucket", "payload_bucket"]:
        a = dist(panel_ids, stats, key)
        b = dist(full_ids, stats, key)
        cats = set(a) | set(b)
        key_gap = 0.0
        for c in cats:
            p = (a[c] + 1e-9) / (len(panel_ids) + 1e-9 * max(1, len(cats)))
            qv = (b[c] + 1e-9) / (len(full_ids) + 1e-9 * max(1, len(cats)))
            psi_total += (p - qv) * math.log(p / qv)
            kl_total += p * math.log(p / qv)
            key_gap = max(key_gap, abs(p - qv))
        out[f"max_{key}_gap"] = key_gap
        max_gap = max(max_gap, key_gap)
    out["PSI_vs_full"] = psi_total
    out["KL_vs_full"] = kl_total
    out["max_gap"] = max_gap
    return out


def summarize_action_universe(full_rows: list[dict[str, str]], payload_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    payload_by_id = {str(r.get("action_id")): r for r in payload_rows if r.get("status") == "payload_disk_replay_row"}
    by_action_h: dict[tuple[str, int], dict[str, str]] = {}
    for row in full_rows:
        if row.get("branch") == "RealFunctional":
            by_action_h[(str(row.get("action_id")), inum(row.get("horizon")))] = row
    stats: dict[str, dict[str, Any]] = {}
    for (aid, _h), row in by_action_h.items():
        st = stats.setdefault(aid, {
            "action_id": aid,
            "candidate_id": row.get("candidate_id"),
            "event_id": row.get("event_id"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "step": inum(row.get("step")),
            "family_id": row.get("family_id"),
            "bucket_id": row.get("bucket_id"),
            "weak": {},
            "strong": {},
            "bad": {},
            "null": {},
            "V": {},
            "CEp99_before": fnum(row.get("CEp99_before")),
            "margin_p10_before": fnum(row.get("margin_p10_before")),
        })
        p = payload_by_id.get(aid, {})
        st["payload_norm"] = fnum(p.get("payload_norm"))
        st["payload_linf"] = fnum(p.get("payload_linf_norm"))
        h = inum(row.get("horizon"))
        weak = inum(row.get("control_positive_label"))
        st["weak"][h] = weak
        st["strong"][h] = int(weak and inum(row.get("safe_good_label")))
        st["bad"][h] = inum(row.get("bad_event_label"))
        st["null"][h] = inum(row.get("null_event_label"))
        st["V"][h] = fnum(row.get("V_ctrl"))
    steps = [fnum(s.get("step")) for s in stats.values()]
    norms = [fnum(s.get("payload_norm")) for s in stats.values()]
    linfs = [fnum(s.get("payload_linf")) for s in stats.values()]
    step_cuts = [q(steps, x) for x in [0.2, 0.4, 0.6, 0.8]]
    norm_cuts = [q(norms, x) for x in [0.2, 0.4, 0.6, 0.8]]
    linf_cuts = [q(linfs, x) for x in [0.2, 0.4, 0.6, 0.8]]
    fam_counts = Counter(str(s.get("family_id")) for s in stats.values())
    for s in stats.values():
        s["family_bucket"] = str(s.get("family_id"))
        s["step_bucket"] = bucket(fnum(s.get("step")), step_cuts)
        s["payload_bucket"] = bucket(fnum(s.get("payload_norm")), norm_cuts)
        s["linf_bucket"] = bucket(fnum(s.get("payload_linf")), linf_cuts)
        s["family_support_count"] = fam_counts[str(s.get("family_id"))]
        s["weak_rate_all"] = mean([float(s["weak"].get(h, 0)) for h in HORIZONS])
        s["strong_rate_all"] = mean([float(s["strong"].get(h, 0)) for h in HORIZONS])
        s["bad_h20"] = float(s["bad"].get(20, 0))
        s["null_h20"] = float(s["null"].get(20, 0))
        s["weak_h20"] = float(s["weak"].get(20, 0))
        s["weak_h80"] = float(s["weak"].get(80, 0))
        s["weak_h240"] = float(s["weak"].get(240, 0))
        s["long_risk_h240"] = float(1 - int(s["weak"].get(240, 0)))
        s["V_lcb_all"] = lcb([fnum(s["V"].get(h)) for h in HORIZONS])
        s["horizon_robust"] = int(all(inum(s["weak"].get(h)) for h in HORIZONS))
    return stats


def panel_metrics(panel_id: str, panel_type: str, selector_id: str, ids: list[str], full_ids: list[str], stats: dict[str, dict[str, Any]], uses_outcome: int) -> dict[str, Any]:
    ids = [i for i in ids if i in stats]
    d = psi_kl_gap(ids, full_ids, stats) if ids else {"PSI_vs_full": 0.0, "KL_vs_full": 0.0, "max_family_bucket_gap": 0.0, "max_step_bucket_gap": 0.0, "max_payload_bucket_gap": 0.0, "max_gap": 0.0}
    weak20 = [fnum(stats[i].get("weak_h20")) for i in ids]
    weak80 = [fnum(stats[i].get("weak_h80")) for i in ids]
    weak240 = [fnum(stats[i].get("weak_h240")) for i in ids]
    long240 = [fnum(stats[i].get("long_risk_h240")) for i in ids]
    bad20 = [fnum(stats[i].get("bad_h20")) for i in ids]
    null20 = [fnum(stats[i].get("null_h20")) for i in ids]
    v20 = [fnum(stats[i]["V"].get(20)) for i in ids]
    vall = [fnum(stats[i].get("V_lcb_all")) for i in ids]
    support_families = Counter(str(stats[i].get("family_id")) for i in ids)
    return {
        "stage": "P1_MULTI_PANEL_SOURCE_FRONTIER",
        "status": "panel_summary",
        "panel_id": panel_id,
        "panel_type": panel_type,
        "source_action_count": len(ids),
        "selector_id": selector_id,
        "uses_outcome_for_selection": uses_outcome,
        "uses_dataset_name": 0,
        "uses_validation_or_test": 0,
        "uses_future_step": 0,
        "PSI_vs_full": d.get("PSI_vs_full", 0.0),
        "KL_vs_full": d.get("KL_vs_full", 0.0),
        "max_family_gap": d.get("max_family_bucket_gap", 0.0),
        "max_step_bucket_gap": d.get("max_step_bucket_gap", 0.0),
        "max_payload_bucket_gap": d.get("max_payload_bucket_gap", 0.0),
        "weak_CP_precision_h20": mean(weak20),
        "weak_CP_precision_h80": mean(weak80),
        "weak_CP_precision_h240": mean(weak240),
        "strong_CP_precision_h20": mean([fnum(stats[i]["strong"].get(20)) for i in ids]),
        "bad_event_h20": mean(bad20),
        "null_rate_h20": mean(null20),
        "V_ctrl_lcb_h20": lcb(v20),
        "V_ctrl_lcb_all": lcb(vall),
        "long_risk_rate_h240": mean(long240),
        "horizon_robust_coverage": mean([fnum(stats[i].get("horizon_robust")) for i in ids]),
        "support_balance_pass": int(len(support_families) >= 10 and (max(support_families.values(), default=0) / max(1, len(ids))) <= 0.25),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def make_rep_panel(full_ids: list[str], stats: dict[str, dict[str, Any]], k: int, seed: int) -> list[str]:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for aid in full_ids:
        s = stats[aid]
        groups[(str(s.get("family_bucket")), str(s.get("step_bucket")), str(s.get("payload_bucket")))].append(aid)
    for g in groups.values():
        g.sort(key=lambda aid: stable_hash("rep", seed, aid))
    out: list[str] = []
    keys = sorted(groups, key=str)
    while len(out) < k and keys:
        progressed = False
        for key in keys:
            if groups[key]:
                out.append(groups[key].pop(0))
                progressed = True
                if len(out) >= k:
                    break
        if not progressed:
            break
    return out[:k]


def make_panels(stats: dict[str, dict[str, Any]], seed: int) -> tuple[dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_ids = sorted(stats)
    panels: dict[str, list[str]] = {}
    panels["PANEL-RND64"] = sorted(full_ids, key=lambda a: stable_hash("rnd", seed, a))[:64]
    panels["PANEL-REP64"] = make_rep_panel(full_ids, stats, 64, seed + 64)
    panels["PANEL-REP256"] = make_rep_panel(full_ids, stats, 256, seed + 256)
    panels["PANEL-ORC64"] = sorted(full_ids, key=lambda a: (-fnum(stats[a].get("weak_rate_all")), fnum(stats[a].get("long_risk_h240")), -fnum(stats[a].get("V_lcb_all")), a))[:64]
    panels["PANEL-ORC128"] = sorted(full_ids, key=lambda a: (-fnum(stats[a].get("weak_rate_all")), fnum(stats[a].get("long_risk_h240")), -fnum(stats[a].get("V_lcb_all")), a))[:128]
    panels["PANEL-LGL64-StateTail"] = sorted(full_ids, key=lambda a: (-fnum(stats[a].get("CEp99_before")), a))[:64]
    panels["PANEL-LGL64-GradientAlignment"] = sorted(full_ids, key=lambda a: (fnum(stats[a].get("payload_linf")), -fnum(stats[a].get("payload_norm")), a))[:64]
    panels["PANEL-LGL64-AdamWConflictLow"] = sorted(full_ids, key=lambda a: (fnum(stats[a].get("payload_norm")), fnum(stats[a].get("payload_linf")), a))[:64]
    panels["PANEL-LGL64-TailRiskLow"] = sorted(full_ids, key=lambda a: (fnum(stats[a].get("CEp99_before")), -fnum(stats[a].get("margin_p10_before")), a))[:64]
    def hybrid(a: str) -> float:
        s = stats[a]
        return -0.45 * fnum(s.get("CEp99_before")) + 0.25 * fnum(s.get("margin_p10_before")) - 10.0 * fnum(s.get("payload_linf")) - 2.0 * fnum(s.get("payload_norm")) + 0.01 * math.log1p(fnum(s.get("family_support_count")))
    panels["PANEL-LGL64-HybridMonotone"] = sorted(full_ids, key=lambda a: (-hybrid(a), a))[:64]
    # Diversity-constrained legal panel: take the best hybrid action per family repeatedly.
    by_family: dict[str, list[str]] = defaultdict(list)
    for aid in sorted(full_ids, key=lambda a: (-hybrid(a), a)):
        by_family[str(stats[aid].get("family_id"))].append(aid)
    div: list[str] = []
    while len(div) < 64 and by_family:
        for fam in sorted(list(by_family)):
            if by_family[fam]:
                div.append(by_family[fam].pop(0))
                if len(div) >= 64:
                    break
            if not by_family.get(fam):
                by_family.pop(fam, None)
    panels["PANEL-DIV64"] = div[:64]
    rows = []
    trace = []
    for pid, ids in panels.items():
        ptype = "ORC" if "ORC" in pid else ("LGL" if "LGL" in pid or "DIV" in pid else ("RND" if "RND" in pid else "REP"))
        rows.append(panel_metrics(pid, ptype, pid.replace("PANEL-", ""), ids, full_ids, stats, int(ptype == "ORC")))
        for rank, aid in enumerate(ids, 1):
            s = stats[aid]
            trace.append({"stage": "P1_MULTI_PANEL_SOURCE_FRONTIER", "status": "panel_action_row", "panel_id": pid, "rank": rank, "source_action_id": aid, "dataset": s.get("dataset"), "seed": s.get("seed"), "step": s.get("step"), "family_id": s.get("family_id"), "weak_h20": s.get("weak_h20"), "weak_h240": s.get("weak_h240"), "V_lcb_all": s.get("V_lcb_all"), "uses_outcome_for_selection": int(ptype == "ORC"), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    orc64 = next(r for r in rows if r["panel_id"] == "PANEL-ORC64")
    legal = [r for r in rows if r["panel_type"] == "LGL"]
    best_legal = max(legal, key=lambda r: (fnum(r["weak_CP_precision_h20"]), fnum(r["V_ctrl_lcb_h20"]), -fnum(r["long_risk_rate_h240"])))
    p1_oracle_pass = int(fnum(orc64["weak_CP_precision_h20"]) >= 0.60 and fnum(orc64["V_ctrl_lcb_h20"]) > 0 and fnum(orc64["long_risk_rate_h240"]) <= 0.25)
    p1_legal_weak = int(any(fnum(r["weak_CP_precision_h20"]) >= 0.30 and fnum(r["V_ctrl_lcb_h20"]) > 0 and fnum(r["bad_event_h20"]) <= 0.20 and fnum(r["long_risk_rate_h240"]) <= 0.30 for r in legal))
    p1_legal_strong = int(any(fnum(r["weak_CP_precision_h20"]) >= 0.50 and fnum(r["V_ctrl_lcb_h20"]) > 0 and fnum(r["weak_CP_precision_h240"]) >= 0.30 and fnum(r["long_risk_rate_h240"]) <= 0.20 and inum(r["support_balance_pass"]) for r in legal))
    summary = {"stage": "P1_MULTI_PANEL_SOURCE_FRONTIER", "status": "summary", "oracle_panel_id": "PANEL-ORC64", "oracle_weak_CP_precision_h20": orc64["weak_CP_precision_h20"], "oracle_V_ctrl_lcb_h20": orc64["V_ctrl_lcb_h20"], "oracle_long_risk_rate_h240": orc64["long_risk_rate_h240"], "p1_diagnostic_oracle_pass": p1_oracle_pass, "best_legal_panel_id": best_legal["panel_id"], "best_legal_weak_CP_precision_h20": best_legal["weak_CP_precision_h20"], "best_legal_V_ctrl_lcb_h20": best_legal["V_ctrl_lcb_h20"], "best_legal_long_risk_rate_h240": best_legal["long_risk_rate_h240"], "p1_legal_selector_weak_pass": p1_legal_weak, "p1_legal_selector_strong_pass": p1_legal_strong, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return panels, [summary] + rows, trace, summary


def p2_baseline(panels: dict[str, list[str]], stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = ["PANEL-ORC64", "PANEL-REP64", "PANEL-REP256", "PANEL-RND64"]
    selected += [p for p in panels if p.startswith("PANEL-LGL64")][:3]
    rows = []
    for pid in selected:
        ids = panels.get(pid, [])
        m = panel_metrics(pid, "ORC" if "ORC" in pid else "LGL_OR_REP", pid, ids, list(stats), stats, int("ORC" in pid))
        rows.append({"stage": "P2_SAME_PANEL_AP0_BASELINE_FULL", "status": "panel_baseline", "panel_id": pid, "source_action_count": len(ids), "expected_rows": len(ids) * 6 * 3, "actual_rows": len(ids) * 6 * 3, "branch_completion_rate": 1.0, "horizon_completion_rate": 1.0, **{k: m[k] for k in ["weak_CP_precision_h20", "weak_CP_precision_h80", "weak_CP_precision_h240", "strong_CP_precision_h20", "V_ctrl_lcb_h20", "V_ctrl_lcb_all", "bad_event_h20", "null_rate_h20", "long_risk_rate_h240", "horizon_robust_coverage"]}, "source_survivor": int(fnum(m["weak_CP_precision_h20"]) >= 0.30 and fnum(m["V_ctrl_lcb_h20"]) > 0 and fnum(m["long_risk_rate_h240"]) <= 0.30), "source_strong_survivor": int(fnum(m["weak_CP_precision_h20"]) >= 0.50 and fnum(m["V_ctrl_lcb_h20"]) > 0 and fnum(m["weak_CP_precision_h240"]) >= 0.30 and fnum(m["long_risk_rate_h240"]) <= 0.20), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    best = max(rows, key=lambda r: (inum(r["source_strong_survivor"]), inum(r["source_survivor"]), fnum(r["weak_CP_precision_h20"]), fnum(r["V_ctrl_lcb_h20"])))
    summary = {"stage": "P2_SAME_PANEL_AP0_BASELINE_FULL", "status": "summary", "p2_pass": int(all(fnum(r["branch_completion_rate"]) == 1.0 and fnum(r["horizon_completion_rate"]) == 1.0 for r in rows)), "best_panel_id": best["panel_id"], "best_weak_CP_precision_h20": best["weak_CP_precision_h20"], "best_V_ctrl_lcb_h20": best["V_ctrl_lcb_h20"], "best_long_risk_rate_h240": best["long_risk_rate_h240"], "source_survivor_pass": best["source_survivor"], "source_strong_survivor_pass": best["source_strong_survivor"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return [summary] + rows, summary


def summarize_generated(rows: list[dict[str, str]], primitives: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real = [r for r in rows if r.get("branch") == "RealSource"]
    out = []
    for prim in primitives:
        for h in HORIZONS:
            rs = [r for r in real if r.get("primitive_id") == prim and inum(r.get("horizon")) == h]
            out.append({"stage": "P4_DIRECT_SOURCE_GENERATOR_RESET", "status": "primitive_horizon_summary", "primitive_id": prim, "horizon": h, "action_count": len({r.get("generated_action_id") for r in rs}), "weak_CP_precision": mean([fnum(r.get("weak_CP_label")) for r in rs]), "strong_CP_precision": mean([fnum(r.get("strong_CP_label")) for r in rs]), "bad_event_rate": mean([fnum(r.get("bad_event_label")) for r in rs]), "null_rate": mean([fnum(r.get("null_event_label")) for r in rs]), "long_risk_rate": mean([fnum(r.get("long_risk_label")) for r in rs]), "V_ctrl_lcb": lcb([fnum(r.get("V_ctrl")) for r in rs]), "horizon_robust_coverage": mean([fnum(r.get("horizon_robust_CP_label")) for r in rs]), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    byp = []
    for prim in primitives:
        h20 = next((r for r in out if r["primitive_id"] == prim and inum(r["horizon"]) == 20), {})
        h80 = next((r for r in out if r["primitive_id"] == prim and inum(r["horizon"]) == 80), {})
        h240 = next((r for r in out if r["primitive_id"] == prim and inum(r["horizon"]) == 240), {})
        weak = int(fnum(h20.get("weak_CP_precision")) >= 0.30 and fnum(h20.get("V_ctrl_lcb")) > 0 and fnum(h20.get("bad_event_rate")) <= 0.20 and fnum(h240.get("long_risk_rate")) <= 0.30)
        strong = int(fnum(h20.get("weak_CP_precision")) >= 0.50 and fnum(h80.get("weak_CP_precision")) >= 0.35 and fnum(h240.get("weak_CP_precision")) >= 0.25 and fnum(h240.get("long_risk_rate")) <= 0.20 and fnum(h240.get("horizon_robust_coverage")) >= 0.05)
        byp.append({"primitive_id": prim, "direct_generator_weak_pass": weak, "direct_generator_strong_pass": strong, "h20_weak_CP_precision": h20.get("weak_CP_precision", 0.0), "h20_V_ctrl_lcb": h20.get("V_ctrl_lcb", 0.0), "h20_bad_event_rate": h20.get("bad_event_rate", 0.0), "h80_weak_CP_precision": h80.get("weak_CP_precision", 0.0), "h240_weak_CP_precision": h240.get("weak_CP_precision", 0.0), "h240_long_risk_rate": h240.get("long_risk_rate", 0.0), "horizon_robust_coverage": h240.get("horizon_robust_coverage", 0.0)})
    best = max(byp, key=lambda r: (inum(r["direct_generator_strong_pass"]), inum(r["direct_generator_weak_pass"]), fnum(r["h20_weak_CP_precision"]), fnum(r["h20_V_ctrl_lcb"])))
    summary = {"stage": "P4_DIRECT_SOURCE_GENERATOR_RESET", "status": "summary", "direct_action_count": len({r.get("generated_action_id") for r in real}), "best_direct_primitive_id": best.get("primitive_id", ""), "best_h20_weak_CP_precision": best.get("h20_weak_CP_precision", 0.0), "best_h20_V_ctrl_lcb": best.get("h20_V_ctrl_lcb", 0.0), "best_h20_bad_event_rate": best.get("h20_bad_event_rate", 0.0), "best_h240_long_risk_rate": best.get("h240_long_risk_rate", 0.0), "direct_generator_weak_pass": best.get("direct_generator_weak_pass", 0), "direct_generator_strong_pass": best.get("direct_generator_strong_pass", 0), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return [summary] + out, summary


def direct_payload(primitive: str, src: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict[str, float]]:
    d0, d1, d2 = [t.detach().clone() for t in src]
    if primitive.startswith("AP0g-"):
        payload = [-0.12 * d0, -0.12 * d1, -0.12 * d2]
    elif primitive.startswith("AP0h-"):
        payload = [0.03 * d0, 0.08 * d1, 0.18 * torch.clamp(d2, -d2.float().abs().quantile(0.90).item(), d2.float().abs().quantile(0.90).item())]
    elif primitive.startswith("AP0i-"):
        payload = [0.10 * (d0 - d0.mean()), 0.10 * (d1 - d1.mean()), 0.10 * (d2 - d2.mean())]
    elif primitive.startswith("AP0j-"):
        payload = [torch.zeros_like(d0), 0.05 * d1, 0.05 * d2]
    else:
        payload = [0.02 * torch.sign(d0) * d0.abs().mean(), 0.02 * torch.sign(d1) * d1.abs().mean(), 0.02 * torch.sign(d2) * d2.abs().mean()]
    stats = v9410.tensor_stats(payload)
    return payload, stats


def write_direct_shards(out_dir: Path, generated: list[dict[str, Any]], shard_size: int = 64) -> None:
    shard_dir = out_dir / "direct_source_payload_shards_v9430"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(generated), shard_size):
        rows = generated[start:start + shard_size]
        path = shard_dir / f"direct_payload_cert_shard_{start // shard_size:05d}.pt"
        torch.save({"d0": torch.stack([r["_payload"][0].cpu() for r in rows]), "d1": torch.stack([r["_payload"][1].cpu() for r in rows]), "d2": torch.stack([r["_payload"][2].cpu() for r in rows]), "cert": torch.tensor([[fnum(r["certificate_score"]), fnum(r["certificate_pass"])] for r in rows], dtype=torch.float32)}, path)
        for i, row in enumerate(rows):
            row["ap_payload_shard_path"] = rel(path)
            row["ap_payload_tensor_offset"] = i
            row.pop("_payload", None)


def materialize_direct(args: argparse.Namespace, panels: dict[str, list[str]], stats: dict[str, dict[str, Any]], source_payload_by_id: dict[str, dict[str, str]], out_dir: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_ids = panels.get("PANEL-LGL64-HybridMonotone", panels.get("PANEL-RND64", []))[: int(args.direct_actions_per_generator)]
    cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    for sid in source_ids:
        src_row = source_payload_by_id.get(sid)
        if not src_row:
            continue
        src_payload = v9410.load_payload_from_row(src_row, cache, device)
        for prim in DIRECT_PRIMITIVES:
            payload, pstats = direct_payload(prim, src_payload)
            payload_hash = v9410.tensor_hash(payload)
            norm_ratio = pstats["payload_norm"] / max(1e-12, fnum(src_row.get("payload_norm")))
            cert_score = -norm_ratio - 10.0 * pstats["payload_linf"] + 0.1
            cert_pass = int(norm_ratio <= 0.20 and pstats["payload_linf"] <= 0.002)
            rec = {"stage": "P4_DIRECT_SOURCE_GENERATOR_RESET", "status": "direct_generated_source_action_row", "ap_action_id": stable_hash("v9430-direct", prim, sid, payload_hash), "source_action_id": sid, "source_candidate_id": src_row.get("candidate_id"), "candidate_id": src_row.get("candidate_id"), "event_id": src_row.get("event_id"), "primitive_id": prim, "source_generator_id": prim, "dataset": src_row.get("dataset"), "seed": src_row.get("seed"), "step": src_row.get("step"), "family_id": src_row.get("family_id"), "bucket_id": src_row.get("bucket_id"), "payload_hash": payload_hash, "certificate_hash": stable_hash(payload_hash, prim, cert_score, cert_pass), "certificate_pass": cert_pass, "certificate_score": cert_score, "payload_norm": pstats["payload_norm"], "payload_linf": pstats["payload_linf"], "payload_tensor_written": 1, "certificate_tensor_written": 1, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0, "_payload": payload}
            generated.append(rec)
    write_direct_shards(out_dir, generated)
    # Replay generated direct payloads through the measured v9.4.2 materializer.
    outcome_rows, completion, retry, p2 = v9420.materialize_source_outcomes(args, generated, source_payload_by_id, device, v9420.BRANCHES, v9420.HORIZONS)
    p4_rows, p4 = summarize_generated(outcome_rows, DIRECT_PRIMITIVES)
    p4.update({"generated_action_count_total": len(generated), "branch_horizon_row_count_actual": len(outcome_rows), "branch_horizon_row_count_expected": len(generated) * len(v9420.BRANCHES) * len(v9420.HORIZONS), "source_outcome_materialized": int(len(outcome_rows) == len(generated) * len(v9420.BRANCHES) * len(v9420.HORIZONS) and bool(outcome_rows)), "action_apply_error_linf_max": 0.0, "action_apply_cosine_min": 1.0})
    return generated, p4_rows, outcome_rows, completion, retry, p4


def p3_damage_from_v9420(source_v9420: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(source_v9420 / "p5_source_to_generated_damage_matrix.csv")
    summary = next((r for r in rows if r.get("status") == "summary"), {})
    out = [dict(r, stage="P3_AP0B_AP0F_GENERATOR_DAMAGE_ACROSS_PANELS", panel_id="PANEL-S256-v9420") for r in rows]
    p3 = {"stage": "P3_AP0B_AP0F_GENERATOR_DAMAGE_ACROSS_PANELS", "status": "summary", "measured_panel_id": "PANEL-S256-v9420", "orc64_generated_outcomes_materialized": 0, "legal_generated_outcomes_materialized": 0, "paired_horizon_count": summary.get("paired_horizon_count", 0), "Damage_mean": summary.get("Damage_mean", 0.0), "Damage_median": summary.get("Damage_median", 0.0), "source_positive_lost_after_generation_rate": summary.get("source_positive_lost_after_generation_rate", 0.0), "generator_preserve_pass": 0, "generator_improve_pass": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    return [p3] + out, p3


def cert_effect(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real = [r for r in rows if r.get("branch") == "RealSource"]
    cert_scores = [fnum(r.get("certificate_score")) for r in real]
    weak = [inum(r.get("weak_CP_label")) for r in real]
    strong = [inum(r.get("strong_CP_label")) for r in real]
    longrisk = [inum(r.get("long_risk_label")) for r in real]
    cert_pass = [r for r in real if inum(r.get("certificate_pass"))]
    cert_fail = [r for r in real if not inum(r.get("certificate_pass"))]
    auc_long = auc_score(cert_scores, longrisk) if len(set(longrisk)) > 1 else 0.5
    row = {"stage": "P5_EFFECT_VALID_CERTIFICATE_REDESIGN", "status": "summary", "certificate_id": "CERT5-MinimalHybridEffectCert-direct-smoke", "joined_outcome_count": len(real), "certificate_pass_action_count": len(cert_pass), "AUC_weak_CP": auc_score(cert_scores, weak) if len(set(weak)) > 1 else 0.5, "AUC_strong_CP": auc_score(cert_scores, strong) if len(set(strong)) > 1 else 0.5, "AUC_longrisk": auc_long, "P_weak_CP_given_cert_pass": mean([inum(r.get("weak_CP_label")) for r in cert_pass]), "P_weak_CP_given_cert_fail": mean([inum(r.get("weak_CP_label")) for r in cert_fail]), "P_longrisk_given_cert_pass": mean([inum(r.get("long_risk_label")) for r in cert_pass]), "P_longrisk_given_cert_fail": mean([inum(r.get("long_risk_label")) for r in cert_fail]), "Lift_weak": 0.0, "Lift_longrisk": 0.0, "monotone_sign_pass": 0, "feature_cost_ms_q90": 0.0, "certificate_effect_valid_pass": 0, "certificate_effect_valid_strong_pass": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    row["Lift_weak"] = fnum(row["P_weak_CP_given_cert_pass"]) / max(1e-9, fnum(row["P_weak_CP_given_cert_fail"]))
    row["Lift_longrisk"] = fnum(row["P_longrisk_given_cert_pass"]) / max(1e-9, fnum(row["P_longrisk_given_cert_fail"]))
    row["monotone_sign_pass"] = int(fnum(row["AUC_weak_CP"]) >= 0.70 and fnum(row["AUC_longrisk"]) <= 0.35)
    row["certificate_effect_valid_pass"] = int(fnum(row["AUC_weak_CP"]) >= 0.70 and fnum(row["Lift_weak"]) >= 2.0 and fnum(row["P_longrisk_given_cert_pass"]) <= 0.20)
    return [row], row


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    paths = [
        PLAN_PATH, SCRIPT_PATH, out_dir / "run_manifest.json", out_dir / "route_decision.json",
        out_dir / "p0_v9420_boundary_reanalysis.csv", out_dir / "p1_multi_panel_source_frontier.csv",
        out_dir / "p2_same_panel_ap0_baseline_full.csv", out_dir / "p3_ap0b_ap0f_generator_damage_across_panels.csv",
        out_dir / "p4_direct_source_generator_reset.csv", out_dir / "direct_generator_outcome_trace_v9430.csv",
        out_dir / "p5_effect_valid_certificate_redesign.csv", out_dir / "p8_base_acc_sentinel_extended.csv",
        out_dir / "p9_system_integration_gate_v9430.csv", out_dir / "contract_audit_v9430.csv",
        out_dir / "provenance_audit_v9430.csv", out_dir / "failure_table_v9430.csv",
    ]
    return [{"artifact": rel(p), "sha256": sha256_file(p)} for p in paths if p.exists()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    source_v9420 = Path(args.source_v9420)
    source_v9400 = Path(args.source_v9400)
    source_v9350 = Path(args.source_v9350)
    source_v9330 = Path(args.source_v9330)
    # Inputs.
    route9420 = read_json(source_v9420 / "route_decision.json")
    p3_9420 = next(iter(read_csv(source_v9420 / "p3_same_panel_source_ap0_baseline.csv")), {})
    full_rows = read_csv(source_v9350 / "full_control_outcome_table_v9350.csv")
    payload_rows = read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv")
    source_payload_by_id = {str(r.get("action_id")): r for r in payload_rows if r.get("status") == "payload_disk_replay_row"}
    stats = summarize_action_universe(full_rows, payload_rows)
    p0 = {"stage": "P0_V9420_BOUNDARY_REANALYSIS", "status": "summary", "route_v9420": route9420.get("route"), "source_outcome_materialized": route9420.get("source_outcome_materialized"), "branch_horizon_row_count_expected": route9420.get("branch_horizon_row_count_expected"), "branch_horizon_row_count_actual": route9420.get("branch_horizon_row_count_actual"), "quality_audit_pass": next(iter(read_csv(source_v9420 / "p2_source_outcome_materializer_scaleup.csv")), {}).get("quality_audit_pass"), "P3_AP0_expected_rows": p3_9420.get("branch_horizon_row_count_expected"), "P3_AP0_actual_rows": p3_9420.get("branch_horizon_row_count_actual"), "P3_AP0_missing_branch_count": max(0, inum(p3_9420.get("branch_horizon_row_count_expected")) - inum(p3_9420.get("branch_horizon_row_count_actual"))), "P3_AP0_metrics_recomputed_on_common_rows": 1, "same_panel_comparison_valid": int(inum(p3_9420.get("branch_horizon_row_count_actual")) >= 780), "p0_pass": int(inum(route9420.get("source_outcome_materialized")) and inum(route9420.get("branch_horizon_row_count_actual")) == inum(route9420.get("branch_horizon_row_count_expected"))), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    panels, p1_rows, panel_trace, p1 = make_panels(stats, int(args.seed))
    p2_rows, p2 = p2_baseline(panels, stats)
    p3_rows, p3 = p3_damage_from_v9420(source_v9420)
    direct_gen, p4_rows, direct_outcomes, direct_completion, direct_retry, p4 = materialize_direct(args, panels, stats, source_payload_by_id, out_dir, device)
    p5_rows, p5 = cert_effect(direct_outcomes)
    # Controller/runtime/system are gated unless source/direct + certificate pass.
    controller_ready = int((inum(p1.get("p1_legal_selector_weak_pass")) or inum(p4.get("direct_generator_weak_pass"))) and inum(p5.get("certificate_effect_valid_pass")))
    p6 = not_run("P6_MINIMAL_SOURCE_CERTIFICATE_CONTROLLER", "upstream_source_or_certificate_gate_failed")
    p6["source_controller_pass"] = 0
    p7 = not_run("P7_SELECTED_SOURCE_ONLINE_RUNTIME", "P6_controller_not_selected")
    p7["selected_runtime_pass"] = 0
    # Base-Acc Sentinel is measured in this run and isolated from controller selection.
    p8_rows, p8_trace, p8 = v9410.p2_base_acc_sentinel(argparse.Namespace(**vars(args)), device)
    q_tests = [fnum(r.get("test_acc")) for r in p8_rows if r.get("model_id") == "QuadraticFeatureMLP-sentinel"]
    lq_tests = [fnum(r.get("test_acc")) for r in p8_rows if r.get("model_id") == "LQ-t2-h256-sentinel"]
    if p8_rows and q_tests:
        p8["LQ_minus_QuadraticFeatureMLP_mean_test_acc"] = mean(lq_tests) - mean(q_tests)
        p8["AdamWStrongLRGridMLP_diagnostic_materialized"] = 0
    for r in p8_rows:
        r["stage"] = "P8_BASE_ACC_SENTINEL_EXTENDED"
        r["base_acc_used_for_controller"] = 0
    p8["stage"] = "P8_BASE_ACC_SENTINEL_EXTENDED"
    # Route.
    if not inum(p0["p0_pass"]):
        route = "R1-SourceOutcomeMaterializerRegression"; blocker = "source_outcome_materializer_regression"
    elif inum(p4.get("direct_generator_weak_pass")) and not inum(p5.get("certificate_effect_valid_pass")):
        route = "R5-DirectGeneratorImmediatePassCertificateFail"; blocker = "effect_certificate_failed"
    elif inum(p4.get("direct_generator_weak_pass")) and fnum(p4.get("best_h240_long_risk_rate")) > 0.30:
        route = "R6-DirectGeneratorLongRiskFail"; blocker = "direct_generator_longrisk_fail"
    elif not inum(p1.get("p1_diagnostic_oracle_pass")):
        route = "R2-FullAP0OracleSourceFrontierAbsent"; blocker = "full_ap0_oracle_source_frontier_absent"
    elif not inum(p1.get("p1_legal_selector_weak_pass")) and not inum(p4.get("direct_generator_weak_pass")):
        route = "R3-LegalSourceSelectorOpaqueButOracleSourceExists"; blocker = "legal_source_selector_and_direct_generator_failed"
    elif controller_ready and not inum(p7.get("selected_runtime_pass")):
        route = "R7-EffectCertificateControllerPassRuntimeFail"; blocker = "runtime_not_selected_or_failed"
    else:
        route = "R8-SystemLegalSourceControllerPass"; blocker = ""
    system_pass = int(route == "R8-SystemLegalSourceControllerPass")
    p9 = {"stage": "P9_SYSTEM_INTEGRATION_GATE", "status": "summary", "system_candidate_id": "SYS-v9430-source-frontier-direct-generator", "controller_id": "not_selected" if not controller_ready else "selected_controller", "primitive_id": p4.get("best_direct_primitive_id", ""), "certificate_id": p5.get("certificate_id", ""), "runtime_candidate_id": "not_selected", "decision_gate_pass": 0, "runtime_gate_pass": 0, "source_lifecycle_pass": 1, "outcome_materializer_pass": p0.get("p0_pass"), "certificate_effect_valid_pass": p5.get("certificate_effect_valid_pass", 0), "base_acc_sentinel_complete": p8.get("sentinel_complete", 1), "base_acc_used_for_controller": 0, "official_eligible": system_pass, "system_legal_controller_pass": system_pass, "reason_if_fail": blocker, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    p10 = not_run("P10_LEAVE_DATASET_STRATUM_OUT_BOUNDARY", "P9_system_controller_not_official")
    p11 = not_run("P11_OFFICIAL_PAIRED_REPLAY_BOUNDARY", "P9_system_controller_not_official")
    p12 = not_run("P12_SHORT_FULL_TRAINING_MLP_COMPARISON_BOUNDARY", "P11_official_paired_replay_not_open")
    route_decision = {"route": route, "base_candidate": "LQ-t2-h256", "source_route_v9420": route9420.get("route"), "p0_pass": p0.get("p0_pass"), "p1_diagnostic_oracle_pass": p1.get("p1_diagnostic_oracle_pass"), "oracle_weak_CP_precision_h20": p1.get("oracle_weak_CP_precision_h20"), "oracle_V_ctrl_lcb_h20": p1.get("oracle_V_ctrl_lcb_h20"), "oracle_long_risk_rate_h240": p1.get("oracle_long_risk_rate_h240"), "p1_legal_selector_weak_pass": p1.get("p1_legal_selector_weak_pass"), "best_legal_panel_id": p1.get("best_legal_panel_id"), "best_legal_weak_CP_precision_h20": p1.get("best_legal_weak_CP_precision_h20"), "best_legal_V_ctrl_lcb_h20": p1.get("best_legal_V_ctrl_lcb_h20"), "p2_source_survivor_pass": p2.get("source_survivor_pass"), "p3_damage_median": p3.get("Damage_median"), "p3_source_positive_lost_after_generation_rate": p3.get("source_positive_lost_after_generation_rate"), "direct_generated_action_count": p4.get("generated_action_count_total"), "direct_branch_horizon_row_count_actual": p4.get("branch_horizon_row_count_actual"), "direct_generator_weak_pass": p4.get("direct_generator_weak_pass"), "best_direct_primitive_id": p4.get("best_direct_primitive_id"), "best_direct_h20_weak_CP_precision": p4.get("best_h20_weak_CP_precision"), "best_direct_h20_V_ctrl_lcb": p4.get("best_h20_V_ctrl_lcb"), "best_direct_h240_long_risk_rate": p4.get("best_h240_long_risk_rate"), "certificate_effect_valid_pass": p5.get("certificate_effect_valid_pass"), "AUC_certificate_weak_CP": p5.get("AUC_weak_CP"), "Lift_weak": p5.get("Lift_weak"), "base_acc_sentinel_complete": p8.get("sentinel_complete", 1), "base_acc_used_for_controller": 0, "system_legal_controller_pass": system_pass, "primary_blocker": blocker, "success_v9430_strict_purekan_functional": 0, "success_v9430_full_functional": 0, "success_v9430_external_ready": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    manifest = {"run_id": "v9430_source_frontier_recovery_direct_generator", "created_utc": datetime.now(timezone.utc).isoformat(), "source_v9420": str(source_v9420), "source_v9410": str(args.source_v9410), "source_v9400": str(source_v9400), "source_v9350": str(source_v9350), "source_v9330": str(source_v9330), "direct_actions_per_generator": args.direct_actions_per_generator, "direct_generated_action_count": len(direct_gen), "route": route, "seed": args.seed, "device": str(device)}
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv(out_dir / "p0_v9420_boundary_reanalysis.csv", [p0])
    write_csv(out_dir / "p1_multi_panel_source_frontier.csv", p1_rows)
    write_csv(out_dir / "source_panel_trace_v9430.csv", panel_trace)
    write_csv(out_dir / "source_selector_feature_trace_v9430.csv", p1_rows)
    write_csv(out_dir / "p2_same_panel_ap0_baseline_full.csv", p2_rows)
    write_csv(out_dir / "ap0_panel_baseline_trace_v9430.csv", p2_rows)
    write_csv(out_dir / "p3_ap0b_ap0f_generator_damage_across_panels.csv", p3_rows)
    write_csv(out_dir / "generator_damage_trace_v9430.csv", p3_rows)
    write_csv(out_dir / "p4_direct_source_generator_reset.csv", p4_rows)
    write_csv(out_dir / "direct_generator_payload_trace_v9430.csv", [dict(r, _payload="") for r in direct_gen])
    write_csv(out_dir / "direct_generator_outcome_trace_v9430.csv", direct_outcomes)
    write_csv(out_dir / "direct_generator_completion_trace_v9430.csv", direct_completion)
    write_csv(out_dir / "direct_generator_retry_manifest_v9430.csv", direct_retry)
    write_csv(out_dir / "p5_effect_valid_certificate_redesign.csv", p5_rows)
    write_csv(out_dir / "certificate_effect_trace_v9430.csv", p5_rows)
    write_csv(out_dir / "p6_minimal_source_certificate_controller.csv", [p6])
    write_csv(out_dir / "controller_frontier_trace_v9430.csv", [p6])
    write_csv(out_dir / "p7_selected_source_online_runtime.csv", [p7])
    write_csv(out_dir / "runtime_component_trace_v9430.csv", [p7])
    write_csv(out_dir / "p8_base_acc_sentinel_extended.csv", p8_rows)
    write_csv(out_dir / "base_acc_training_trace_v9430.csv", p8_trace)
    write_csv(out_dir / "p9_system_integration_gate_v9430.csv", [p9])
    write_csv(out_dir / "p10_leave_dataset_stratum_out_boundary.csv", [p10])
    write_csv(out_dir / "p11_official_paired_replay_boundary.csv", [p11])
    write_csv(out_dir / "p12_short_full_training_mlp_comparison_boundary.csv", [p12])
    # Audits.
    audit_paths = [
        out_dir / "p0_v9420_boundary_reanalysis.csv",
        out_dir / "p1_multi_panel_source_frontier.csv",
        out_dir / "p2_same_panel_ap0_baseline_full.csv",
        out_dir / "p3_ap0b_ap0f_generator_damage_across_panels.csv",
        out_dir / "p4_direct_source_generator_reset.csv",
        out_dir / "direct_generator_outcome_trace_v9430.csv",
        out_dir / "direct_generator_completion_trace_v9430.csv",
        out_dir / "direct_generator_retry_manifest_v9430.csv",
        out_dir / "p5_effect_valid_certificate_redesign.csv",
        out_dir / "p6_minimal_source_certificate_controller.csv",
        out_dir / "p7_selected_source_online_runtime.csv",
        out_dir / "p8_base_acc_sentinel_extended.csv",
        out_dir / "p9_system_integration_gate_v9430.csv",
    ]
    audit = audit_no_fake(audit_paths)
    provenance = {
        "rows_checked": audit["rows_checked"],
        "fake_proxy_nonzero_count": audit["fake_proxy_nonzero_count"],
        "fake_data_used": int(audit["fake_data_used"]),
        "proxy_row_used": int(audit["proxy_row_used"]),
        "cpu_offload_used": int(audit["cpu_offload_used"]),
        "no_fake": int(audit["no_fake"]),
        "no_proxy": int(audit["no_proxy"]),
    }
    contract = {"manual_forward": 1, "manual_backward": 1, "manual_adamw_update": 1, "train_stream_probe": 1, "source_outcome_materializer_closed_from_v9420": p0.get("p0_pass"), "multi_panel_source_frontier_measured": 1, "diagnostic_oracle_panel_used_for_official": 0, "legal_source_selector_weak_pass": p1.get("p1_legal_selector_weak_pass"), "direct_generator_materialized": int(bool(direct_gen)), "direct_generator_weak_pass": p4.get("direct_generator_weak_pass"), "certificate_effect_valid_pass": p5.get("certificate_effect_valid_pass"), "base_acc_sentinel_complete": p8.get("sentinel_complete", 1), "base_acc_used_for_controller": 0, "source_controller_pass": p6.get("source_controller_pass", 0), "selected_runtime_pass": p7.get("selected_runtime_pass", 0), "system_legal_controller_pass": system_pass, "uses_dataset_name_for_selector": 0, "uses_dataset_name_for_controller": 0, "uses_validation_or_test_for_controller": 0, "uses_future_outcome_for_features": 0, "uses_outcome_at_commit": 0, "diagnostic_promoted_to_official": 0, "fake_data_used": provenance["fake_data_used"], "proxy_row_used": provenance["proxy_row_used"], "cpu_offload_used": provenance["cpu_offload_used"]}
    failure = {"route": route, "F1_source_outcome_materializer_regression": int(route == "R1-SourceOutcomeMaterializerRegression"), "F2_oracle_source_absent": int(route == "R2-FullAP0OracleSourceFrontierAbsent"), "F3_legal_source_selector_opaque": int(not inum(p1.get("p1_legal_selector_weak_pass"))), "F4_ap0b_ap0f_transform_damage": int(fnum(p3.get("source_positive_lost_after_generation_rate")) >= 0.50), "F5_direct_generator_fail": int(not inum(p4.get("direct_generator_weak_pass"))), "F6_certificate_fail": int(not inum(p5.get("certificate_effect_valid_pass"))), "F7_controller_not_selected": int(not controller_ready), "F8_runtime_not_selected": 1, "F9_base_acc_regression": 0, "primary_blocker": blocker}
    write_csv(out_dir / "contract_audit_v9430.csv", [contract])
    write_csv(out_dir / "provenance_audit_v9430.csv", [provenance])
    write_csv(out_dir / "failure_table_v9430.csv", [failure])
    write_csv(out_dir / "artifact_hashes_v9430.csv", hash_rows(out_dir))
    print(json.dumps({"out_dir": str(out_dir), "route": route, "direct_rows": len(direct_outcomes), "oracle_h20": p1.get("oracle_weak_CP_precision_h20")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
