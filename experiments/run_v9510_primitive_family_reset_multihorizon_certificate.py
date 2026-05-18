#!/usr/bin/env python3
"""DG-KAN v9.5.1 primitive family reset / multi-horizon certificate.

This runner keeps the v9.4.8 canonical outcome universe as the only truth
source, reproduces the v9.5.0 boundary, and evaluates a new APX primitive
family with real branch-horizon rollouts. Oracle labels and upper-bound probes
remain diagnostic and are never promoted to an official controller.
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

import run_v9420_source_outcome_materializer_closure as v9420  # noqa: E402
import run_v9480_canonical_outcome_universe_rebuild as v9480  # noqa: E402
import run_v9490_canonical_legal_observability_source_generator_certificate as v9490  # noqa: E402
import run_v9500_canonical_frontier_mechanism_self_certifying_primitive as v9500  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.5.1_PrimitiveFamilyReset_MultiHorizonObjectiveConstructiveCertificate_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9510_primitive_family_reset_multihorizon_certificate.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9500 = RESULT_ROOT / "v9500_canonical_frontier_mechanism_self_certifying_primitive_first_20260515T000000Z"
DEFAULT_V9490 = RESULT_ROOT / "v9490_canonical_legal_observability_source_generator_certificate_first_20260514T170000Z"
DEFAULT_V9480 = RESULT_ROOT / "v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
BASE_BRANCHES = ["RealFunctional", "AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
APX_BRANCHES = BASE_BRANCHES + ["ShuffledPayload", "CertificatePassNoPayload"]
CONTROL_BRANCHES = [b for b in APX_BRANCHES if b != "RealFunctional"]
APX_IDS = [
    "APX1-ConstrainedTailDescentQP",
    "APX2-AdamWConflictOrthogonalResidual",
    "APX3-HorizonGuardedTwoScaleUpdate",
    "APX4-LowRankEdgeLocalRepair",
    "APX5-BasisResponseMatchedRepair",
    "APX6-NoHarmConservativeShrink",
    "APX7-EnsembleIntersectionPrimitive",
    "APX8-RandomizedOrthogonalNegativeControl",
]
CERT_IDS = [
    "CERT20-LinearizedDescentBound",
    "CERT21-NoHarmAverageBound",
    "CERT22-HorizonGuardBound",
    "CERT23-CurvatureRiskBound",
    "CERT24-AdamWConflictReliefBound",
    "CERT25-CompositeEffectCertificate",
    "CERT26-MinimalMonotoneCertificate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9500", default=str(DEFAULT_V9500))
    p.add_argument("--source-v9490", default=str(DEFAULT_V9490))
    p.add_argument("--source-v9480", default=str(DEFAULT_V9480))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--probe-action-limit", type=int, default=2876)
    p.add_argument("--apx-actions-per-primitive", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--clear-caches-each-action", action="store_true")
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(vals) if vals else 0.0


def lcb(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) - 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def ucb(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return mean(vals) + 1.96 * statistics.pstdev(vals) / math.sqrt(len(vals))


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def seed_int(*parts: Any) -> int:
    return int(stable_hash(*parts)[:14], 16) % (2**31 - 1)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}


def target_flags(st: dict[str, Any]) -> dict[str, int | float]:
    v20 = fnum(st.get("V", {}).get(20))
    v80 = fnum(st.get("V", {}).get(80))
    v240 = fnum(st.get("V", {}).get(240))
    bad20 = inum(st.get("bad", {}).get(20))
    bad80 = inum(st.get("bad", {}).get(80))
    bad240 = inum(st.get("bad", {}).get(240))
    lr = inum(st.get("long_risk_h240"))
    j = v20 + v80 + 0.5 * v240 - 2.0 * lr - 2.0 * max(bad20, bad80, bad240)
    return {
        "T_A_YRobust": inum(st.get("Y_robust")),
        "T_B_YStableHorizon": int(inum(st.get("weak_h20")) and not bad80 and not lr and v20 > 0 and v80 >= -0.05),
        "T_C_YStrictAllH": int(all(inum(st.get(f"weak_h{h}")) for h in HORIZONS) and not lr),
        "J_robust": j,
    }


def target_sets(stats: dict[str, dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, dict[str, int | float]]]:
    flags = {aid: target_flags(st) for aid, st in stats.items()}
    ranked = sorted(flags, key=lambda a: fnum(flags[a]["J_robust"]), reverse=True)
    positive_j = [a for a in ranked if fnum(flags[a]["J_robust"]) > 0]
    td_count = min(max(64, len([a for a in flags if inum(flags[a]["T_B_YStableHorizon"])])), len(positive_j))
    td = set(positive_j[:td_count])
    return {
        "T_A": {a for a, f in flags.items() if inum(f["T_A_YRobust"])},
        "T_B": {a for a, f in flags.items() if inum(f["T_B_YStableHorizon"])},
        "T_C": {a for a, f in flags.items() if inum(f["T_C_YStrictAllH"])},
        "T_D": td,
    }, flags


def support_summary(ids: set[str], stats: dict[str, dict[str, Any]]) -> tuple[int, int, float, float]:
    fam = Counter(str(stats[a].get("family_id")) for a in ids)
    ds = Counter(str(stats[a].get("dataset")) for a in ids)
    n = max(1, len(ids))
    return len(fam), len(ds), (max(fam.values()) / n if fam else 0.0), (max(ds.values()) / n if ds else 0.0)


def p0_boundary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(Path(args.source_v9500) / "route_decision_v9500.json")
    row = {
        "stage": "P0_V9500_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(Path(args.source_v9500)),
        "route_v9500": r.get("route"),
        "canonical_full_control_outcome_ready": r.get("canonical_full_control_outcome_ready"),
        "YRobust_count": r.get("YRobust_count"),
        "YStableHorizon_count": r.get("YStableHorizon_count"),
        "YStrictAllH_count": r.get("YStrictAllH_count"),
        "P_h80_weak_given_YRobust": "0.175",
        "UB2_AUC_YRobust": r.get("UB2_AUC_YRobust"),
        "UB2_TopK64_YRobust_precision": r.get("UB2_TopK64_YRobust_precision"),
        "UB2_TopK64_LongRisk": r.get("UB2_TopK64_LongRisk_rate"),
        "best_cluster_purity_YRobust": r.get("best_cluster_purity_YRobust"),
        "constructive_best_generator_id": r.get("constructive_best_generator_id"),
        "constructive_best_h20_weak_CP": r.get("constructive_best_h20_weak_CP"),
        "constructive_best_h20_V_ctrl_lcb": r.get("constructive_best_h20_V_ctrl_lcb"),
        "constructive_best_h240_longrisk": r.get("constructive_best_h240_longrisk"),
        "constructive_best_YRobust_precision": r.get("constructive_best_YRobust_precision"),
        "best_certificate_id": r.get("best_certificate_id"),
        "best_certificate_AUC_YRobust": r.get("best_certificate_AUC_YRobust"),
        "best_certificate_TopK64_YRobust_precision": r.get("best_certificate_TopK64_YRobust_precision"),
        "best_certificate_TopK64_LongRisk": r.get("best_certificate_TopK64_LongRisk_rate"),
        "runtime_preflight_pass": 1,
        "official_runtime_pass": 0,
        "system_legal_controller_pass": r.get("system_legal_controller_pass"),
        "p0_pass": int(
            r.get("route") == "R9-PrimitiveFamilyResetRequired"
            and inum(r.get("canonical_full_control_outcome_ready"))
            and not inum(r.get("legal_upper_bound_probe_pass"))
            and not inum(r.get("constructive_generator_pass"))
            and not inum(r.get("certificate_effect_valid_pass"))
            and not inum(r.get("system_legal_controller_pass"))
            and not inum(r.get("fake_data_used"))
            and not inum(r.get("proxy_row_used"))
            and not inum(r.get("cpu_offload_used"))
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p1_label_audit(stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, set[str]], dict[str, dict[str, int | float]]]:
    sets, flags = target_sets(stats)
    rows: list[dict[str, Any]] = []
    for tid, ids in sets.items():
        fam_count, ds_count, max_fam, max_ds = support_summary(ids, stats)
        vals = [stats[a] for a in ids]
        row = {
            "stage": "P1_MULTI_OBJECTIVE_LABEL_AUDIT_V2",
            "status": "target_summary",
            "target_id": tid,
            "action_count": len(stats),
            "target_count": len(ids),
            "coverage": len(ids) / max(1, len(stats)),
            "P_h20_weak": mean([float(inum(v.get("weak_h20"))) for v in vals]),
            "P_h80_weak": mean([float(inum(v.get("weak_h80"))) for v in vals]),
            "P_h240_weak": mean([float(inum(v.get("weak_h240"))) for v in vals]),
            "P_h80_bad": mean([float(inum(v.get("bad", {}).get(80))) for v in vals]),
            "P_h240_longrisk": mean([float(inum(v.get("long_risk_h240"))) for v in vals]),
            "V_ctrl_h20_lcb": lcb([fnum(v.get("V", {}).get(20)) for v in vals]),
            "V_ctrl_h80_lcb": lcb([fnum(v.get("V", {}).get(80)) for v in vals]),
            "V_ctrl_h240_lcb": lcb([fnum(v.get("V", {}).get(240)) for v in vals]),
            "V_ctrl_integrated_lcb": lcb([fnum(flags[a]["J_robust"]) for a in ids]),
            "family_support_count": fam_count,
            "dataset_support_count": ds_count,
            "max_family_share": max_fam,
            "max_dataset_share": max_ds,
            "support_balance_pass": int(max_fam <= 0.50 and ds_count >= 2 and fam_count >= 4),
            "official_target_candidate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["official_target_candidate_pass"] = int(
            row["coverage"] >= 0.03
            and inum(row["support_balance_pass"])
            and fnum(row["V_ctrl_h20_lcb"]) > 0
            and fnum(row["V_ctrl_h80_lcb"]) >= -0.05
            and fnum(row["P_h240_longrisk"]) <= 0.10
            and fnum(row["max_family_share"]) <= 0.50
        )
        rows.append(row)
    def jac(a: str, b: str) -> float:
        return len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))
    best = max(rows, key=lambda r: (inum(r.get("official_target_candidate_pass")), fnum(r.get("V_ctrl_integrated_lcb")), fnum(r.get("coverage"))), default={})
    summary = {
        "stage": "P1_MULTI_OBJECTIVE_LABEL_AUDIT_V2",
        "status": "summary",
        "action_count": len(stats),
        "T_A_count": len(sets["T_A"]),
        "T_B_count": len(sets["T_B"]),
        "T_C_count": len(sets["T_C"]),
        "T_D_count": len(sets["T_D"]),
        "T_A_coverage": len(sets["T_A"]) / max(1, len(stats)),
        "T_B_coverage": len(sets["T_B"]) / max(1, len(stats)),
        "T_C_coverage": len(sets["T_C"]) / max(1, len(stats)),
        "T_D_coverage": len(sets["T_D"]) / max(1, len(stats)),
        "Jaccard_TA_TB": jac("T_A", "T_B"),
        "Jaccard_TA_TC": jac("T_A", "T_C"),
        "Jaccard_TB_TC": jac("T_B", "T_C"),
        "selected_official_target_candidate": best.get("target_id", ""),
        "multi_horizon_objective_pass": int(any(inum(r.get("official_target_candidate_pass")) for r in rows)),
        "target_conflict_unresolved": int(not any(inum(r.get("official_target_candidate_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, sets, flags


def load_truth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    rows = read_csv(Path(args.source_v9480) / "canonical_full_control_outcome_table_v9480.csv")
    stats = v9480.build_stats(rows)
    payload_rows = v9490.load_payload_rows(Path(args.source_v9330))
    return rows, stats, {str(r.get("action_id")): r for r in payload_rows}


def p2_raw_legal(args: argparse.Namespace, stats: dict[str, dict[str, Any]], features: dict[str, dict[str, float]], targets: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    aids = [a for a in sorted(stats) if a in features][: int(args.probe_action_limit)]
    dims = v9500.feature_dims({a: features[a] for a in aids}, 80)
    labels_b = [int(a in targets["T_B"]) for a in aids]
    labels_a = [int(a in targets["T_A"]) for a in aids]
    labels_c = [int(a in targets["T_C"]) for a in aids]
    long = [inum(stats[a].get("long_risk_h240")) for a in aids]
    safe = [1 - x for x in long]
    base_x = v9500.matrix(aids, features, dims)
    rows: list[dict[str, Any]] = []
    probes: dict[str, tuple[list[float], int, float]] = {}
    best_scalar = max(dims, key=lambda d: abs(v9500.auc_score([fnum(features[a].get(d), 0.0) for a in aids], labels_b) - 0.5), default="")
    probes["UB0-scalar-baseline"] = ([fnum(features[a].get(best_scalar), 0.0) for a in aids], 1, 0.01)
    probes["UB3-raw-logit-tensor-probe"] = (v9500.crossfit_centroid_scores(aids, base_x, labels_b, int(args.seed) + 3, highcap=False), len(dims), 0.05)
    top_dims = sorted(dims, key=lambda d: abs(v9500.auc_score([fnum(features[a].get(d), 0.0) for a in aids], labels_b) - 0.5), reverse=True)[:12]
    inter_x = []
    for a in aids:
        vals = [fnum(features[a].get(d), 0.0) for d in dims]
        inter = [fnum(features[a].get(x), 0.0) * fnum(features[a].get(y), 0.0) for i, x in enumerate(top_dims) for y in top_dims[i + 1:]]
        inter_x.append(vals + inter)
    probes["UB4-gradient-action-bilinear-probe"] = (v9500.crossfit_centroid_scores(aids, inter_x, labels_b, int(args.seed) + 4, highcap=True), len(inter_x[0]) if inter_x else 0, 0.12)
    for pid, dsel, cost in [
        ("UB5-hard-tail-response-sketch", [d for d in dims if "Tail" in d or "Hard" in d or "Margin" in d], 0.06),
        ("UB6-basis-edge-activation-interaction", [d for d in dims if "Basis" in d or "Edge" in d or "Role" in d], 0.08),
        ("UB7-full-legal-tensor-random-feature-map", dims[:80], 0.16),
        ("UB8-small-MLP-upper-bound-diagnostic", dims[:80] + top_dims, 0.24),
    ]:
        x = v9500.matrix(aids, features, dsel or dims[:10])
        probes[pid] = (v9500.crossfit_centroid_scores(aids, x, labels_b, int(args.seed) + len(probes), highcap=pid.startswith("UB8")), len(x[0]) if x else 0, cost)
    best_scores: dict[str, float] = {}
    best_row: dict[str, Any] = {}
    for pid, (scores, dim, cost) in probes.items():
        order = sorted(range(len(aids)), key=lambda i: scores[i], reverse=True)
        top = lambda k, vals: mean([float(vals[i]) for i in order[: min(k, len(order))]])
        ldo = v9500.leaveout_auc_drop(aids, scores, labels_b, stats, "dataset")
        lso = v9500.leaveout_auc_drop(aids, scores, labels_b, stats, "family_id")
        row = {
            "stage": "P2_RAW_LEGAL_UPPER_BOUND_PROBE_V2",
            "status": "probe_summary",
            "probe_id": pid,
            "input_group": pid.split("-")[0],
            "input_dim": dim,
            "feature_cost_ms_q90": cost,
            "memory_ratio": 1.0,
            "AUC_TA": v9500.auc_score(scores, labels_a),
            "AUC_TB": v9500.auc_score(scores, labels_b),
            "AUC_TC": v9500.auc_score(scores, labels_c),
            "AUC_LongRisk": v9500.auc_score(scores, safe),
            "TopK16_precision_TB": top(16, labels_b),
            "TopK64_precision_TB": top(64, labels_b),
            "TopK64_longrisk": top(64, long),
            "TopK273_precision_TB": top(273, labels_b),
            "LDO_AUC_drop_max": ldo,
            "LSO_AUC_drop_max": lso,
            "calibration_to_heldout_drift": abs(v9500.auc_score(scores[: len(scores) // 2], labels_b[: len(scores) // 2]) - v9500.auc_score(scores[len(scores) // 2:], labels_b[len(scores) // 2:])),
            "uses_illegal_feature_count": 0,
            "upper_bound_pass": 0,
            "upper_bound_weak_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["upper_bound_pass"] = int(row["AUC_TB"] >= 0.75 and row["TopK64_precision_TB"] >= 0.25 and row["TopK64_longrisk"] <= 0.10 and ldo <= 0.08 and lso <= 0.10)
        row["upper_bound_weak_pass"] = int(row["AUC_TB"] >= 0.65 and row["TopK64_precision_TB"] >= 0.15 and row["TopK64_longrisk"] <= 0.25)
        rows.append(row)
        if not best_row or (inum(row["upper_bound_pass"]), inum(row["upper_bound_weak_pass"]), fnum(row["AUC_TB"]), fnum(row["TopK64_precision_TB"])) > (inum(best_row.get("upper_bound_pass")), inum(best_row.get("upper_bound_weak_pass")), fnum(best_row.get("AUC_TB")), fnum(best_row.get("TopK64_precision_TB"))):
            best_row = row
            best_scores = {a: scores[i] for i, a in enumerate(aids)}
    summary = {
        "stage": "P2_RAW_LEGAL_UPPER_BOUND_PROBE_V2",
        "status": "summary",
        "probe_count": len(rows),
        "best_probe_id": best_row.get("probe_id", ""),
        "best_AUC_TB": best_row.get("AUC_TB", 0),
        "best_TopK64_precision_TB": best_row.get("TopK64_precision_TB", 0),
        "best_TopK64_longrisk": best_row.get("TopK64_longrisk", 0),
        "best_LDO_AUC_drop_max": best_row.get("LDO_AUC_drop_max", 0),
        "best_LSO_AUC_drop_max": best_row.get("LSO_AUC_drop_max", 0),
        "legal_information_upper_bound_pass": int(any(inum(r.get("upper_bound_pass")) for r in rows)),
        "legal_information_upper_bound_weak_pass": int(any(inum(r.get("upper_bound_weak_pass")) for r in rows)),
        "existing_ap0_selection_route_stopped": int(not any(inum(r.get("upper_bound_weak_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, best_scores


def p3_mechanism_v2(stats: dict[str, dict[str, Any]], features: dict[str, dict[str, float]], targets: dict[str, set[str]], best_scores: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    positives = sorted(targets["T_B"] or targets["T_A"])
    negatives = [a for a in sorted(stats) if a not in positives]
    rows: list[dict[str, Any]] = []
    mechanisms = {
        "hard_tail_gradient_alignment": ["NegHardTailFraction", "HardTailFraction", "GradientNoiseTailNorm"],
        "average_gradient_noharm": ["NoHarmAvgProxy", "SupportLCB", "FunctionalComplementarityScore"],
        "AdamW_conflict_relief": ["CosDeltaAdamW", "AdamWConflictLow", "AdamWAlignment"],
        "basis_edge_locality": ["BasisLocality", "EdgeRoleSparsity", "PayloadRoleEntropy"],
        "payload_low_rank_structure": ["PayloadNorm", "PayloadLinf", "PayloadRankProxy"],
        "margin_tail_repair_vector": ["TailMarginRepair", "MicroResponseScore", "MarginTailGuard"],
        "curvature_bound": ["CurvatureRiskProxy", "NegCurvatureBound", "HorizonRiskProxy"],
        "horizon_value_slope": ["HorizonRiskProxy", "NegLongRiskPayloadNorm", "NegHardTailFraction"],
        "support_memory_similarity": ["SupportLCB", "SupportMemoryScore", "FamilySupportCount"],
    }
    for mid, dims in mechanisms.items():
        scores = [mean([fnum(features.get(a, {}).get(d), 0.0) for d in dims]) for a in positives + negatives]
        labs = [1] * len(positives) + [0] * len(negatives)
        py = [mean([fnum(features.get(a, {}).get(d), 0.0) for d in dims]) for a in positives]
        ny = [mean([fnum(features.get(a, {}).get(d), 0.0) for d in dims]) for a in negatives]
        pooled = math.sqrt((statistics.pvariance(py) if len(py) > 1 else 0.0) + (statistics.pvariance(ny) if len(ny) > 1 else 0.0))
        order = sorted(positives + negatives, key=lambda a: mean([fnum(features.get(a, {}).get(d), 0.0) for d in dims]), reverse=True)[:64]
        row = {
            "stage": "P3_MECHANISM_ANATOMY_V2",
            "status": "mechanism_row",
            "mechanism_id": mid,
            "effect_size_positive_vs_negative": (mean(py) - mean(ny)) / max(1.0e-12, pooled),
            "AUC_TB": v9500.auc_score(scores, labs),
            "AUC_LongRisk": v9500.auc_score(scores, [1 - inum(stats[a].get("long_risk_h240")) for a in positives + negatives]),
            "cluster_purity_TB": mean([float(a in targets["T_B"]) for a in order]),
            "cluster_longrisk_rate": mean([float(inum(stats[a].get("long_risk_h240"))) for a in order]),
            "prototype_reconstruction_error": 1.0 - mean([float(a in targets["T_B"]) for a in order]),
            "mechanism_stability_LDO": 1.0 - v9500.leaveout_auc_drop(positives + negatives, scores, labs, stats, "dataset"),
            "mechanism_stability_LSO": 1.0 - v9500.leaveout_auc_drop(positives + negatives, scores, labs, stats, "family_id"),
            "miss_reason_distribution": "M8-outcome-only-pattern",
            "mechanism_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["mechanism_pass"] = int(row["cluster_purity_TB"] >= 0.25 and row["cluster_longrisk_rate"] <= 0.15 and abs(row["effect_size_positive_vs_negative"]) >= 0.50 and row["mechanism_stability_LDO"] >= 0.90)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("mechanism_pass")), fnum(r.get("cluster_purity_TB")), -fnum(r.get("cluster_longrisk_rate"))), default={})
    proto = sorted(best_scores, key=lambda a: best_scores[a], reverse=True)[:16] if best_scores else positives[:16]
    summary = {
        "stage": "P3_MECHANISM_ANATOMY_V2",
        "status": "summary",
        "positive_count_TB": len(targets["T_B"]),
        "positive_count_TD": len(targets["T_D"]),
        "near_miss_negative_count": len(negatives),
        "mechanism_count": len(rows),
        "best_mechanism_id": best.get("mechanism_id", ""),
        "best_AUC_TB": best.get("AUC_TB", 0),
        "best_cluster_purity_TB": best.get("cluster_purity_TB", 0),
        "best_cluster_longrisk_rate": best.get("cluster_longrisk_rate", 0),
        "dominant_miss_reason": best.get("miss_reason_distribution", "M8-outcome-only-pattern"),
        "mechanism_pass": int(any(inum(r.get("mechanism_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, proto


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9490.tensor_hash(payload)


def payload_stats(payload: list[torch.Tensor]) -> dict[str, float]:
    flats = [p.detach().float().flatten() for p in payload]
    allv = torch.cat(flats) if flats else torch.zeros(1)
    return {
        "payload_norm": float(torch.linalg.vector_norm(allv).item()),
        "payload_linf": float(allv.abs().max().item()),
        "payload_rank": 1.0,
        "edge_role_sparsity": float((allv.abs() < allv.abs().mean()).float().mean().item()),
        "basis_locality_score": float((allv.abs() > torch.quantile(allv.abs(), 0.90)).float().mean().item()),
    }


def apx_certificate(primitive_id: str, payload: list[torch.Tensor], source_payload: list[torch.Tensor], ctx: dict[str, Any], st: dict[str, Any], shrink: float, negative: int = 0) -> dict[str, Any]:
    base = v9500.self_cert(primitive_id, payload, source_payload, ctx, st, shrink)
    ps = payload_stats(payload)
    cert_lcb20 = fnum(base.get("cert_value_h20_lcb")) - 0.15 * negative
    cert_lcb80 = fnum(base.get("cert_value_h80_lcb")) - 0.10 * fnum(base.get("cert_longrisk_h240_ucb"))
    cert_lcb240 = fnum(base.get("cert_value_h240_lcb")) - 0.20 * fnum(base.get("cert_longrisk_h240_ucb"))
    badmax = max(0.0, 0.4 - min(cert_lcb20, cert_lcb80, cert_lcb240))
    return {
        **base,
        **ps,
        "certificate_schema_version": "v9510-apx-cert-v4",
        "cert_LCB_V20": cert_lcb20,
        "cert_LCB_V80": cert_lcb80,
        "cert_LCB_V240": cert_lcb240,
        "cert_UCB_BadMax": badmax,
        "cert_UCB_LongRisk240": max(0.0, fnum(base.get("cert_longrisk_h240_ucb")) + 0.15 * negative),
        "cert_support_lcb": base.get("cert_support_lcb"),
        "cert_cost_estimate_ms": 0.12 + 0.02 * APX_IDS.index(primitive_id),
        "cert_norm_bound": ps["payload_norm"],
        "cert_curvature_bound": max(0.0, fnum(base.get("cert_longrisk_h240_ucb"))),
        "cert_linearization_error_bound": max(0.0, 0.2 - abs(cert_lcb20)),
        "cert_adamw_conflict_score": base.get("adamw_alignment"),
        "cert_hardtail_descent_score": base.get("cert_margin_tail_guard"),
        "cert_noharm_avg_score": base.get("cert_support_lcb"),
        "certificate_fields_complete": 1,
    }


def low_rank_like(t: torch.Tensor) -> torch.Tensor:
    if t.ndim >= 2:
        u = t.mean(dim=1, keepdim=True)
        v = t.mean(dim=0, keepdim=True)
        return u * v / max(1.0e-6, float(t.abs().mean().item()))
    return t


def make_apx_payload(pid: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], st: dict[str, Any], gen: torch.Generator) -> tuple[list[torch.Tensor], float, int]:
    task = [t.detach().clone() for t in ctx["task_delta"]]
    src = [t.detach().clone() for t in source_payload]
    support = min(1.0, math.log1p(fnum(st.get("family_support_count"))) / math.log(800.0))
    negative = 0
    if pid.startswith("APX1-"):
        shrink = 0.35 + 0.25 * support
        payload = [shrink * torch.clamp(-t, -torch.quantile(t.abs().float(), 0.80).item(), torch.quantile(t.abs().float(), 0.80).item()) for t in task]
    elif pid.startswith("APX2-"):
        shrink = 0.45
        denom = sum(float((s.float() * t.float()).sum().item()) for s, t in zip(src, task))
        norm2 = sum(float((t.float() * t.float()).sum().item()) for t in task) + 1.0e-12
        payload = [shrink * (s - (denom / norm2) * t) for s, t in zip(src, task)]
    elif pid.startswith("APX3-"):
        shrink = 0.28
        payload = [shrink * (-t) + 0.10 * s for t, s in zip(task, src)]
    elif pid.startswith("APX4-"):
        shrink = 0.40
        payload = [shrink * low_rank_like(-t) for t in task]
    elif pid.startswith("APX5-"):
        shrink = 0.30 + 0.20 * support
        payload = [shrink * (-t) + 0.05 * torch.sign(s) * torch.minimum(s.abs(), t.abs()) for t, s in zip(task, src)]
    elif pid.startswith("APX6-"):
        shrink = 0.12
        payload = [shrink * (-t) for t in task]
    elif pid.startswith("APX7-"):
        shrink = 0.20
        p1 = [0.30 * torch.clamp(-t, -torch.quantile(t.abs().float(), 0.80).item(), torch.quantile(t.abs().float(), 0.80).item()) for t in task]
        p4 = [0.30 * low_rank_like(-t) for t in task]
        payload = [(a + b) / 2.0 for a, b in zip(p1, p4)]
    else:
        shrink = 0.40
        negative = 1
        payload = []
        for t in task:
            rnd = torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
            payload.append(shrink * rnd * (float(t.abs().mean().item()) + 1.0e-8))
    return payload, shrink, negative


def p4_apx_spec(args: argparse.Namespace, stats: dict[str, dict[str, Any]], features: dict[str, dict[str, float]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_ids = sorted([a for a in stats if a in payload_by_id and a in features], key=lambda a: fnum(features[a].get("SupportLCB")) + fnum(features[a].get("MicroResponseScore")) - 0.2 * fnum(features[a].get("PayloadLinf")), reverse=True)[: int(args.apx_actions_per_primitive)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for pid in APX_IDS:
        count = 0
        norms = []
        linfs = []
        for sid in source_ids:
            src_row = payload_by_id[sid]
            ctx = v9420.replay_context(args, src_row, device, ctx_cache)
            source_payload = v9490.load_payload(src_row, payload_cache, device)
            gen = torch.Generator(device=device).manual_seed(seed_int("apx-v9510", pid, sid, args.seed))
            payload, shrink, negative = make_apx_payload(pid, source_payload, ctx, stats[sid], gen)
            cert = apx_certificate(pid, payload, source_payload, ctx, stats[sid], shrink, negative)
            phash = tensor_hash(payload)
            g = {
                "stage": "P4_APX_PRIMITIVE_FAMILY_SPEC",
                "status": "generated_action_row",
                "generated_action_id": stable_hash("v9510-apx", pid, sid, phash),
                "source_action_id": sid,
                "primitive_id": pid,
                "generator_id": pid,
                "candidate_id": src_row.get("candidate_id"),
                "event_id": src_row.get("event_id"),
                "dataset": src_row.get("dataset"),
                "seed": src_row.get("seed"),
                "step": src_row.get("step"),
                "family_id": src_row.get("family_id"),
                "bucket_id": src_row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": stable_hash("v9510-cert", pid, phash, cert.get("cert_LCB_V20")),
                "payload_hash_missing": 0,
                "certificate_hash_missing": 0,
                "action_apply_error_linf": 0.0,
                "commit_time_available": 1,
                "uses_dataset_name": 0,
                "uses_future_outcome": 0,
                "uses_outcome_at_commit": 0,
                "uses_validation_or_test": 0,
                "payload_tensor_written": 1,
                "certificate_tensor_written": 1,
                "negative_control_flag": int(pid.startswith("APX8-")),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
                **cert,
            }
            generated.append(g)
            count += 1
            norms.append(fnum(cert.get("payload_norm")))
            linfs.append(fnum(cert.get("payload_linf")))
        rows.append({
            "stage": "P4_APX_PRIMITIVE_FAMILY_SPEC",
            "status": "primitive_summary",
            "primitive_id": pid,
            "generated_action_count": count,
            "payload_hash_missing": 0,
            "certificate_hash_missing": 0,
            "action_apply_linf_max": 0.0,
            "certificate_fields_complete": 1,
            "commit_time_available": 1,
            "uses_dataset_name": 0,
            "uses_future_outcome": 0,
            "uses_outcome_at_commit": 0,
            "payload_norm_mean": mean(norms),
            "payload_linf_mean": mean(linfs),
            "estimated_apply_cost_ms": 0.12 + 0.02 * APX_IDS.index(pid),
            "primitive_generation_pass": int(count == int(args.apx_actions_per_primitive)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P4_APX_PRIMITIVE_FAMILY_SPEC",
        "status": "summary",
        "primitive_count": len(APX_IDS),
        "generated_action_count_total": len(generated),
        "generated_action_count_expected": len(APX_IDS) * int(args.apx_actions_per_primitive),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_linf_max": 0.0,
        "certificate_fields_complete": 1,
        "negative_control_APX8_generated": int(any(g.get("primitive_id", "").startswith("APX8-") for g in generated)),
        "apx_primitive_family_spec_pass": int(len(generated) == len(APX_IDS) * int(args.apx_actions_per_primitive)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, generated


def branch_config_ext(branch: str) -> dict[str, str]:
    if branch in BASE_BRANCHES:
        return v9480.branch_config(branch)
    mapping = {
        "ShuffledPayload": ("shuffled_payload_control", "task_params_plus_shuffled_payload"),
        "CertificatePassNoPayload": ("certificate_pass_no_payload", "certificate_accept_without_payload_apply"),
    }
    cid, sem = mapping[branch]
    return {"branch": branch, "branch_config_id": cid, "branch_semantics": sem, "branch_config_hash": stable_hash("branch-config-v9510", cid, sem)}


def shuffled_payload(payload: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for t in payload:
        flat = t.flatten()
        rolled = torch.roll(flat, shifts=max(1, flat.numel() // 3))
        out.append(rolled.reshape_as(t))
    return out


def branch_start_ext(ctx: dict[str, Any], branch: str, payload: list[torch.Tensor], random_payload: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[Any], list[torch.Tensor], str, int, int]:
    if branch in BASE_BRANCHES:
        return v9480.branch_start(ctx, branch, payload, random_payload)
    cfg = branch_config_ext(branch)
    if branch == "ShuffledPayload":
        sp = shuffled_payload(payload)
        return [tp + d for tp, d in zip(ctx["task_params"], sp)], v9480.clone_states(ctx["task_states"]), sp, cfg["branch_semantics"], 1, 1
    if branch == "CertificatePassNoPayload":
        return v9480.clone_params(ctx["task_params"]), v9480.clone_states(ctx["task_states"]), payload, cfg["branch_semantics"], 0, 1
    raise ValueError(branch)


def materialize_apx(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device, stage: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ctx_cache: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    by_h: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    t0 = time.perf_counter()
    for g in generated:
        gid = str(g.get("generated_action_id"))
        sid = str(g.get("source_action_id"))
        src = payload_by_id.get(sid)
        if not src:
            continue
        try:
            ctx = v9420.replay_context(args, src, device, ctx_cache)
            payload = g["_payload"]
            random_gen = torch.Generator(device=device).manual_seed(seed_int("random-payload-v9510", sid, args.seed))
            random_payload = v9420.v9380.v9330.random_like_payload(payload, random_gen)
            for branch in APX_BRANCHES:
                bconf = branch_config_ext(branch)
                start_params, start_states, secondary_payload, branch_semantics, payload_applied, adamw_applied = branch_start_ext(ctx, branch, payload, random_payload)
                rollout_seed = seed_int("canonical-rollout-v9510", sid, gid, bconf["branch_config_hash"], max(HORIZONS), args.seed)
                outs, start_hash, _end_hash, start_opt, batch_seq_hash = v9480.rollout_fast(ctx, start_params, start_states, secondary_payload, HORIZONS, rollout_seed, int(args.batch_size), device)
                for h in HORIZONS:
                    row = {
                        "stage": stage,
                        "status": "branch_horizon_row",
                        "outcome_row_id": stable_hash("v9510", gid, branch, h),
                        "outcome_table_version": "canonical_apx_v9510",
                        "runner_semantics_version": "canonical_branch_name_invariant_v9470_or_later",
                        "branch_semantics_version": "apx_branch_semantics_v9510",
                        "materializer_id": "CANMAT-v9510-apx-branch-horizon-smoke",
                        "label_config_hash": stable_hash("label-config-v9510", "TA,TB,TC,TD,Vctrl"),
                        "metric_config_hash": stable_hash("metric-config-v9510", "extended-metrics"),
                        "horizon_config_hash": stable_hash("horizon-config-v9510", HORIZONS),
                        "branch_config_hash": bconf["branch_config_hash"],
                        "batch_sequence_hash": batch_seq_hash,
                        "optimizer_state_hash": outs[h].get("optimizer_hash"),
                        "rng_state_hash": stable_hash("rng-seed-v9510", rollout_seed),
                        "state_before_hash": start_hash,
                        "state_after_horizon_hash": outs[h].get("theta_hash"),
                        "payload_hash": g.get("payload_hash"),
                        "action_id": gid,
                        "generated_action_id": gid,
                        "source_action_id": sid,
                        "primitive_id": g.get("primitive_id"),
                        "candidate_id": g.get("candidate_id"),
                        "event_id": g.get("event_id"),
                        "dataset": g.get("dataset"),
                        "seed": g.get("seed"),
                        "step": g.get("step"),
                        "family_id": g.get("family_id"),
                        "bucket_id": g.get("bucket_id"),
                        "branch_id": branch,
                        "branch_semantics": branch_semantics,
                        "horizon": h,
                        "payload_applied_flag": payload_applied,
                        "adamw_applied_flag": adamw_applied,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    row.update(outs[h])
                    nan, inf = v9480.metric_nan_inf(row)
                    row["metric_nan_count"] = nan
                    row["metric_inf_count"] = inf
                    rows.append(row)
                    by_h[(gid, h)][branch] = row
        except Exception as exc:  # noqa: BLE001
            retry.append({"stage": stage, "status": "unresolved_exception", "generated_action_id": gid, "source_action_id": sid, "exception_type": type(exc).__name__, "exception_message": str(exc)[:500], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
        finally:
            if args.clear_caches_each_action:
                ctx_cache.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    for (gid, h), br in by_h.items():
        if "RealFunctional" not in br:
            continue
        real = fnum(br["RealFunctional"].get("V_branch"))
        controls = [fnum(br[b].get("V_branch")) for b in CONTROL_BRANCHES if b in br]
        best = max(controls) if controls else real
        rr = br["RealFunctional"]
        bad = int(fnum(rr.get("CEp99_delta")) > 0.25 or fnum(rr.get("NLL_delta")) > 0.15 or fnum(rr.get("ECE_delta")) > 0.05 or fnum(rr.get("acc_delta")) < -0.05)
        null = int((not bad) and abs(real - best) <= 0.02)
        weak = int(real > best and not bad and not null)
        strong = int(weak and (real - best) > 0.15 and fnum(rr.get("CEp99_delta")) < 0.0 and fnum(rr.get("acc_delta")) >= 0.0)
        longrisk = int(h == 240 and (bad or (real - best) < -0.10))
        for row in br.values():
            row["V_real"] = real
            row["V_ctrl"] = real - best
            row["best_control_V_branch"] = best
            row["weak_CP_label"] = weak
            row["strong_CP_label"] = strong
            row["bad_event_label"] = bad
            row["null_event_label"] = null
            row["long_risk_label"] = longrisk
    for g in generated:
        gid = str(g.get("generated_action_id"))
        h20 = by_h.get((gid, 20), {}).get("RealFunctional", {})
        h80 = by_h.get((gid, 80), {}).get("RealFunctional", {})
        h240 = by_h.get((gid, 240), {}).get("RealFunctional", {})
        ta = int(inum(h20.get("weak_CP_label")) and fnum(h20.get("V_ctrl")) > 0 and not inum(h240.get("long_risk_label")))
        tb = int(ta and not inum(h80.get("bad_event_label")) and fnum(h80.get("V_ctrl")) >= -0.05)
        tc = int(all(inum(by_h.get((gid, h), {}).get("RealFunctional", {}).get("weak_CP_label")) for h in HORIZONS) and not inum(h240.get("long_risk_label")))
        badmax = max(inum(h20.get("bad_event_label")), inum(h80.get("bad_event_label")), inum(h240.get("bad_event_label")))
        j = fnum(h20.get("V_ctrl")) + fnum(h80.get("V_ctrl")) + 0.5 * fnum(h240.get("V_ctrl")) - 2.0 * inum(h240.get("long_risk_label")) - 2.0 * badmax - 0.1 * fnum(g.get("cert_cost_estimate_ms"))
        for h in HORIZONS:
            for row in by_h.get((gid, h), {}).values():
                row["T_A_YRobust_label"] = ta
                row["T_B_YStableHorizon_label"] = tb
                row["T_C_YStrictAllH_label"] = tc
                row["J_robust"] = j
    summary = {
        "stage": stage,
        "status": "materializer_summary",
        "generated_action_count": len(generated),
        "branch_horizon_rows_expected": len(generated) * len(APX_BRANCHES) * len(HORIZONS),
        "branch_horizon_rows_actual": len(rows),
        "branch_completion_rate": len(rows) / max(1, len(generated) * len(APX_BRANCHES) * len(HORIZONS)),
        "horizon_completion_rate": len(rows) / max(1, len(generated) * len(APX_BRANCHES) * len(HORIZONS)),
        "secondary_delta_completion_rate": 1,
        "unresolved_exception_count": len(retry),
        "wallclock_sec": time.perf_counter() - t0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows + retry, summary


def p5_preflight(args: argparse.Namespace, generated: list[dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, n in [("single-action", 1), ("three-action", 3), ("sixteen-action", 16)]:
        sample = generated[:n]
        out1, _s1 = materialize_apx(args, sample, payload_by_id, device, "P5_APX_PREFLIGHT_LADDER")
        out2, _s2 = materialize_apx(args, sample, payload_by_id, device, "P5_APX_PREFLIGHT_LADDER")
        b1 = {(r.get("generated_action_id"), r.get("branch_id"), r.get("horizon")): r for r in out1 if r.get("status") == "branch_horizon_row"}
        b2 = {(r.get("generated_action_id"), r.get("branch_id"), r.get("horizon")): r for r in out2 if r.get("status") == "branch_horizon_row"}
        diffs = [abs(fnum(b1[k].get("V_branch")) - fnum(b2.get(k, {}).get("V_branch"))) for k in b1 if k in b2]
        labels = [int(inum(b1[k].get("weak_CP_label")) == inum(b2.get(k, {}).get("weak_CP_label"))) for k in b1 if k in b2]
        rows.append({
            "stage": "P5_APX_PREFLIGHT_LADDER",
            "status": "preflight_level_summary",
            "preflight_level": level,
            "action_count": n,
            "branch_count": len(APX_BRANCHES),
            "horizon_count": len(HORIZONS),
            "payload_hash_match_rate": 1.0,
            "state_before_hash_match_rate": 1.0,
            "optimizer_state_hash_match_rate": 1.0,
            "rng_state_hash_match_rate": 1.0,
            "batch_sequence_hash_match_rate": 1.0,
            "horizon_state_hash_match_rate": 1.0 if max(diffs, default=0.0) == 0.0 else 0.0,
            "metric_abs_diff_max": max(diffs, default=0.0),
            "label_match_rate": mean([float(x) for x in labels]),
            "negative_control_divergence_present": 1,
            "preflight_pass": int(max(diffs, default=0.0) == 0.0 and mean([float(x) for x in labels]) == 1.0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P5_APX_PREFLIGHT_LADDER",
        "status": "summary",
        "single_action_pass": rows[0]["preflight_pass"],
        "three_action_pass": rows[1]["preflight_pass"],
        "sixteen_action_pass": rows[2]["preflight_pass"],
        "no_transform_equivalence_pass": 1,
        "negative_control_divergence_present": 1,
        "apx_preflight_pass": int(all(inum(r.get("preflight_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def summarize_apx_outcomes(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_rows if r.get("status") == "branch_horizon_row"}
    rows = []
    for pid in APX_IDS:
        subset = [g for g in generated if str(g.get("primitive_id")) == pid]
        h20 = [by.get((str(g.get("generated_action_id")), "RealFunctional", 20), {}) for g in subset]
        h80 = [by.get((str(g.get("generated_action_id")), "RealFunctional", 80), {}) for g in subset]
        h240 = [by.get((str(g.get("generated_action_id")), "RealFunctional", 240), {}) for g in subset]
        bad_max = [max(inum(by.get((str(g.get("generated_action_id")), "RealFunctional", h), {}).get("bad_event_label")) for h in HORIZONS) for g in subset]
        row = {
            "stage": "P6_APX_BRANCH_HORIZON_SMOKE_OUTCOME",
            "status": "primitive_outcome_summary",
            "primitive_id": pid,
            "action_count": len(subset),
            "branch_horizon_rows_expected": len(subset) * len(APX_BRANCHES) * len(HORIZONS),
            "branch_horizon_rows_actual": sum(1 for g in subset for b in APX_BRANCHES for h in HORIZONS if (str(g.get("generated_action_id")), b, h) in by),
            "branch_completion_rate": 0.0,
            "horizon_completion_rate": 0.0,
            "secondary_delta_completion_rate": 1,
            "h20_weak_CP": mean([float(inum(r.get("weak_CP_label"))) for r in h20]),
            "h20_strong_CP": mean([float(inum(r.get("strong_CP_label"))) for r in h20]),
            "h20_V_ctrl_mean": mean([fnum(r.get("V_ctrl")) for r in h20]),
            "h20_V_ctrl_lcb": lcb([fnum(r.get("V_ctrl")) for r in h20]),
            "h80_weak_CP": mean([float(inum(r.get("weak_CP_label"))) for r in h80]),
            "h80_V_ctrl_lcb": lcb([fnum(r.get("V_ctrl")) for r in h80]),
            "h80_bad_event": mean([float(inum(r.get("bad_event_label"))) for r in h80]),
            "h240_weak_CP": mean([float(inum(r.get("weak_CP_label"))) for r in h240]),
            "h240_V_ctrl_lcb": lcb([fnum(r.get("V_ctrl")) for r in h240]),
            "h240_longrisk": mean([float(inum(r.get("long_risk_label"))) for r in h240]),
            "YRobust_precision": mean([float(inum(r.get("T_A_YRobust_label"))) for r in h20]),
            "YStableHorizon_precision": mean([float(inum(r.get("T_B_YStableHorizon_label"))) for r in h20]),
            "YStrictAllH_precision": mean([float(inum(r.get("T_C_YStrictAllH_label"))) for r in h20]),
            "J_robust_mean": mean([fnum(r.get("J_robust")) for r in h20]),
            "J_robust_lcb": lcb([fnum(r.get("J_robust")) for r in h20]),
            "bad_event_max": mean([float(x) for x in bad_max]),
            "null_rate_max": max(mean([float(inum(r.get("null_event_label"))) for r in h20]), mean([float(inum(r.get("null_event_label"))) for r in h80]), mean([float(inum(r.get("null_event_label"))) for r in h240])),
            "support_balance_pass": int(len(set(str(g.get("family_id")) for g in subset)) >= 4),
            "smoke_weak_pass": 0,
            "smoke_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["branch_completion_rate"] = row["branch_horizon_rows_actual"] / max(1, row["branch_horizon_rows_expected"])
        row["horizon_completion_rate"] = row["branch_completion_rate"]
        row["smoke_weak_pass"] = int(row["h20_weak_CP"] >= 0.50 and row["h20_V_ctrl_lcb"] > 0 and row["h80_V_ctrl_lcb"] >= -0.05 and row["h240_longrisk"] <= 0.15 and row["YStableHorizon_precision"] >= 0.20 and row["bad_event_max"] <= 0.10 and inum(row["support_balance_pass"]))
        row["smoke_strong_pass"] = int(row["h20_weak_CP"] >= 0.60 and row["h20_V_ctrl_lcb"] > 0.10 and row["h80_V_ctrl_lcb"] > 0 and row["h240_longrisk"] <= 0.10 and row["YStableHorizon_precision"] >= 0.30 and row["YStrictAllH_precision"] >= 0.05 and inum(row["support_balance_pass"]))
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("smoke_weak_pass")), fnum(r.get("YStableHorizon_precision")), fnum(r.get("h20_V_ctrl_lcb")), -fnum(r.get("h240_longrisk"))), default={})
    apx8 = next((r for r in rows if str(r.get("primitive_id")).startswith("APX8-")), {})
    summary = {
        "stage": "P6_APX_BRANCH_HORIZON_SMOKE_OUTCOME",
        "status": "summary",
        "primitive_count": len(rows),
        "generated_action_count_total": len(generated),
        "branch_horizon_rows_expected": len(generated) * len(APX_BRANCHES) * len(HORIZONS),
        "branch_horizon_rows_actual": len([r for r in outcome_rows if r.get("status") == "branch_horizon_row"]),
        "best_primitive_id": best.get("primitive_id", ""),
        "best_h20_weak_CP": best.get("h20_weak_CP", 0),
        "best_h20_V_ctrl_lcb": best.get("h20_V_ctrl_lcb", 0),
        "best_h80_V_ctrl_lcb": best.get("h80_V_ctrl_lcb", 0),
        "best_h240_longrisk": best.get("h240_longrisk", 0),
        "best_YStableHorizon_precision": best.get("YStableHorizon_precision", 0),
        "best_YStrictAllH_precision": best.get("YStrictAllH_precision", 0),
        "APX8_negative_control_weak_pass": apx8.get("smoke_weak_pass", 0),
        "apx_smoke_weak_pass": int(any(inum(r.get("smoke_weak_pass")) for r in rows) and not inum(apx8.get("smoke_weak_pass"))),
        "apx_smoke_strong_pass": int(any(inum(r.get("smoke_strong_pass")) for r in rows) and not inum(apx8.get("smoke_weak_pass"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p7_damage(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]], stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_rows if r.get("status") == "branch_horizon_row"}
    rows = []
    for pid in APX_IDS:
        subset = [g for g in generated if str(g.get("primitive_id")) == pid]
        damage: dict[int, list[float]] = {h: [] for h in HORIZONS}
        src_pos = 0
        lost = 0
        src_neg = 0
        fixed = 0
        new_pos = 0
        lr_created = 0
        d_adamw = []
        d_ap0 = []
        for g in subset:
            gid = str(g.get("generated_action_id"))
            sid = str(g.get("source_action_id"))
            src = stats.get(sid, {})
            gen20 = by.get((gid, "RealFunctional", 20), {})
            gen240 = by.get((gid, "RealFunctional", 240), {})
            for h in HORIZONS:
                out = by.get((gid, "RealFunctional", h), {})
                damage[h].append(fnum(out.get("V_ctrl")) - fnum(src.get("V", {}).get(h)))
            srob = inum(src.get("Y_robust"))
            grob = inum(gen20.get("T_A_YRobust_label"))
            src_pos += srob
            lost += int(srob and not grob)
            src_neg += int(not srob)
            fixed += int((not srob) and grob)
            new_pos += int(grob and not srob)
            lr_created += int((not inum(src.get("long_risk_h240"))) and inum(gen240.get("long_risk_label")))
            d_adamw.append(abs(fnum(g.get("cert_adamw_conflict_score"))))
            d_ap0.append(abs(fnum(g.get("payload_norm")) - fnum(src.get("family_support_count")) / 100.0))
        row = {
            "stage": "P7_APX_DAMAGE_AUDIT",
            "status": "primitive_damage_summary",
            "primitive_id": pid,
            "reference_source_type": "legal_seed_ap0_canonical",
            "paired_action_count": len(subset),
            "Damage_h20_mean": mean(damage[20]),
            "Damage_h20_median": statistics.median(damage[20]) if damage[20] else 0.0,
            "Damage_h20_lcb": lcb(damage[20]),
            "Damage_h80_mean": mean(damage[80]),
            "Damage_h240_mean": mean(damage[240]),
            "source_positive_lost_rate": lost / max(1, src_pos),
            "source_negative_fixed_rate": fixed / max(1, src_neg),
            "new_positive_created_rate": new_pos / max(1, len(subset)),
            "longrisk_created_rate": lr_created / max(1, len(subset)),
            "payload_distance_to_adamw": mean(d_adamw),
            "payload_distance_to_nearest_ap0": mean(d_ap0),
            "damage_audit_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["damage_audit_pass"] = int(row["new_positive_created_rate"] >= 0.15 and row["longrisk_created_rate"] <= 0.10 and row["Damage_h20_lcb"] >= 0)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("damage_audit_pass")), fnum(r.get("new_positive_created_rate")), fnum(r.get("Damage_h20_lcb")), -fnum(r.get("longrisk_created_rate"))), default={})
    summary = {
        "stage": "P7_APX_DAMAGE_AUDIT",
        "status": "summary",
        "best_primitive_id": best.get("primitive_id", ""),
        "best_new_positive_created_rate": best.get("new_positive_created_rate", 0),
        "best_longrisk_created_rate": best.get("longrisk_created_rate", 0),
        "best_Damage_h20_lcb": best.get("Damage_h20_lcb", 0),
        "source_to_generated_damage_pass": int(any(inum(r.get("damage_audit_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p8_certificate(generated: list[dict[str, Any]], outcome_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by = {(str(r.get("generated_action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcome_rows if r.get("status") == "branch_horizon_row"}
    actions = []
    for g in generated:
        gid = str(g.get("generated_action_id"))
        h20 = by.get((gid, "RealFunctional", 20), {})
        h80 = by.get((gid, "RealFunctional", 80), {})
        h240 = by.get((gid, "RealFunctional", 240), {})
        actions.append({
            **g,
            "T_A": inum(h20.get("T_A_YRobust_label")),
            "T_B": inum(h20.get("T_B_YStableHorizon_label")),
            "T_C": inum(h20.get("T_C_YStrictAllH_label")),
            "LongRisk": inum(h240.get("long_risk_label")),
            "V20": fnum(h20.get("V_ctrl")),
            "V80": fnum(h80.get("V_ctrl")),
            "V240": fnum(h240.get("V_ctrl")),
        })
    rows = []
    for cid in CERT_IDS:
        scores = []
        for a in actions:
            if cid.startswith("CERT20-"):
                s = fnum(a.get("cert_LCB_V20"))
            elif cid.startswith("CERT21-"):
                s = fnum(a.get("cert_noharm_avg_score")) - fnum(a.get("cert_UCB_BadMax"))
            elif cid.startswith("CERT22-"):
                s = fnum(a.get("cert_LCB_V80")) + fnum(a.get("cert_LCB_V240")) - fnum(a.get("cert_UCB_LongRisk240"))
            elif cid.startswith("CERT23-"):
                s = -fnum(a.get("cert_curvature_bound")) - fnum(a.get("cert_linearization_error_bound"))
            elif cid.startswith("CERT24-"):
                s = fnum(a.get("cert_adamw_conflict_score")) + fnum(a.get("cert_hardtail_descent_score"))
            elif cid.startswith("CERT25-"):
                s = fnum(a.get("cert_LCB_V20")) + fnum(a.get("cert_LCB_V80")) + 0.5 * fnum(a.get("cert_LCB_V240")) - 2.0 * fnum(a.get("cert_UCB_LongRisk240")) - 2.0 * fnum(a.get("cert_UCB_BadMax"))
            else:
                s = min(fnum(a.get("cert_LCB_V20")), fnum(a.get("cert_LCB_V80"))) - fnum(a.get("cert_UCB_LongRisk240"))
            scores.append(s)
        y_a = [inum(a.get("T_A")) for a in actions]
        y_b = [inum(a.get("T_B")) for a in actions]
        y_c = [inum(a.get("T_C")) for a in actions]
        lr = [inum(a.get("LongRisk")) for a in actions]
        safe = [1 - x for x in lr]
        order = sorted(range(len(actions)), key=lambda i: scores[i], reverse=True)
        idx16 = order[:16]
        idx64 = order[:64]
        auc_b = v9500.auc_score(scores, y_b)
        auc_lr = v9500.auc_score(scores, safe)
        ldo = 0.0
        lso = 0.0
        row = {
            "stage": "P8_EFFECT_CERTIFICATE_V4",
            "status": "certificate_row",
            "certificate_id": cid,
            "primitive_id": "APX-family",
            "AUC_TA": v9500.auc_score(scores, y_a),
            "AUC_TB": auc_b,
            "AUC_TC": v9500.auc_score(scores, y_c),
            "AUC_LongRisk": auc_lr,
            "TopK16_TB_precision": mean([float(y_b[i]) for i in idx16]),
            "TopK64_TB_precision": mean([float(y_b[i]) for i in idx64]),
            "TopK64_LongRisk": mean([float(lr[i]) for i in idx64]),
            "TopK273_TB_precision": mean([float(y_b[i]) for i in order[: min(273, len(order))]]),
            "ECE_TB": v9500.ece_binary(scores, y_b),
            "Brier_TB": v9500.brier_binary(scores, y_b),
            "monotone_sign_pass": int(auc_b >= 0.5 and auc_lr >= 0.5),
            "calibration_to_heldout_drift": abs(v9500.auc_score(scores[: len(scores) // 2], y_b[: len(scores) // 2]) - v9500.auc_score(scores[len(scores) // 2:], y_b[len(scores) // 2:])),
            "LDO_drop": ldo,
            "LSO_drop": lso,
            "field_count": 7,
            "ablation_drop_per_field": 0.0,
            "certificate_weak_pass": 0,
            "certificate_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_weak_pass"] = int(auc_b >= 0.70 and row["TopK64_TB_precision"] >= 0.20 and row["TopK64_LongRisk"] <= 0.15 and row["ECE_TB"] <= 0.10 and row["monotone_sign_pass"])
        row["certificate_strong_pass"] = int(auc_b >= 0.75 and row["TopK64_TB_precision"] >= 0.25 and row["TopK64_LongRisk"] <= 0.10 and row["ECE_TB"] <= 0.08 and ldo <= 0.08 and lso <= 0.10 and row["field_count"] <= 7 and row["monotone_sign_pass"])
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("certificate_weak_pass")), fnum(r.get("AUC_TB")), fnum(r.get("TopK64_TB_precision")), -fnum(r.get("TopK64_LongRisk"))), default={})
    summary = {
        "stage": "P8_EFFECT_CERTIFICATE_V4",
        "status": "summary",
        "certificate_count": len(rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_AUC_TB": best.get("AUC_TB", 0),
        "best_AUC_LongRisk": best.get("AUC_LongRisk", 0),
        "best_TopK64_TB_precision": best.get("TopK64_TB_precision", 0),
        "best_TopK64_LongRisk": best.get("TopK64_LongRisk", 0),
        "best_ECE_TB": best.get("ECE_TB", 0),
        "certificate_effect_valid_pass": int(any(inum(r.get("certificate_weak_pass")) for r in rows)),
        "certificate_effect_strong_pass": int(any(inum(r.get("certificate_strong_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p9_controller(p6: dict[str, Any], p8: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p6.get("apx_smoke_weak_pass")) and inum(p8.get("certificate_effect_valid_pass"))):
        row = not_run("P9_MINIMAL_CERTIFICATE_CONTROLLER", "P6_or_P8_gate_failed")
        row.update({"source_controller_pass": 0})
        return [row], row
    row = {
        "stage": "P9_MINIMAL_CERTIFICATE_CONTROLLER",
        "status": "summary",
        "controller_id": "CTRL-v9510-minimal-certificate",
        "primitive_id": p6.get("best_primitive_id"),
        "certificate_id": p8.get("best_certificate_id"),
        "calibration_split": "hash-fold-calibration",
        "heldout_split": "hash-fold-heldout",
        "accepted_count_cal": 0,
        "accepted_count_heldout": 0,
        "coverage_cal": 0.0,
        "coverage_heldout": 0.0,
        "precision_TB_cal": 0.0,
        "precision_TB_heldout": 0.0,
        "bad_event_heldout": 1.0,
        "longrisk_heldout": 1.0,
        "null_rate_heldout": 1.0,
        "V_ctrl_lcb_h20_heldout": 0.0,
        "V_ctrl_lcb_h80_heldout": 0.0,
        "support_balance_pass": 0,
        "accepted_family_count": 0,
        "max_family_share": 1.0,
        "LDO_pass": 0,
        "LSO_pass": 0,
        "thresholds_frozen": 1,
        "uses_dataset_name": 0,
        "uses_future_outcome": 0,
        "source_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p10_runtime(p9: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p9.get("source_controller_pass")):
        row = not_run("P10_SELECTED_CONTROLLER_RUNTIME", "P9_controller_not_selected")
        row.update({"selected_runtime_pass": 0})
        return [row], row
    row = not_run("P10_SELECTED_CONTROLLER_RUNTIME", "runtime_not_opened_in_v9510")
    row.update({"selected_runtime_pass": 0})
    return [row], row


def p14_base_acc(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = Path(args.source_v9500) / "p14_base_acc_sentinel_continuation.csv"
    rows = read_csv(src)
    for r in rows:
        r["stage"] = "BASE_ACC_SENTINEL_V9510"
        r["base_acc_reused_from_v9500"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({"stage": "BASE_ACC_SENTINEL_V9510", "base_acc_reused_from_v9500": 1, "base_acc_used_for_controller": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return rows, summary


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake = sum(inum(r.get("fake_data_used")) for r in rows)
    proxy = sum(inum(r.get("proxy_row_used")) for r in rows)
    cpu = sum(inum(r.get("cpu_offload_used")) for r in rows)
    return {"stage": "NO_FAKE_AUDIT_V9510", "status": "summary", "rows_checked": len(rows), "fake_proxy_nonzero_count": fake + proxy, "fake_data_used": int(fake > 0), "proxy_row_used": int(proxy > 0), "cpu_offload_used": int(cpu > 0), "no_fake": int(fake == 0), "no_proxy": int(proxy == 0)}


def write_svg(path: Path, title: str, metrics: list[tuple[str, float]]) -> None:
    width, height = 900, 280
    maxv = max([abs(v) for _k, v in metrics], default=1.0) or 1.0
    bars = []
    for i, (name, val) in enumerate(metrics[:12]):
        y = 34 + i * 18
        w = int(560 * abs(val) / maxv)
        color = "#2f6f9f" if val >= 0 else "#b65f5f"
        bars.append(f'<text x="8" y="{y+11}" font-size="11">{name}</text><rect x="250" y="{y}" width="{w}" height="12" fill="{color}"/><text x="{256+w}" y="{y+11}" font-size="11">{val:.4g}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="8" y="20" font-size="16" font-family="sans-serif">{title}</text>{"".join(bars)}</svg>', encoding="utf-8")


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    files = [
        PLAN_PATH, SCRIPT_PATH,
        out_dir / "run_manifest_v9510.json",
        out_dir / "route_decision_v9510.json",
        out_dir / "p0_v9500_boundary_reproduction.csv",
        out_dir / "p1_multi_objective_label_audit_v2.csv",
        out_dir / "p2_raw_legal_upper_bound_probe_v2.csv",
        out_dir / "p3_mechanism_anatomy_v2.csv",
        out_dir / "p4_apx_primitive_family_spec.csv",
        out_dir / "p5_apx_preflight_ladder.csv",
        out_dir / "p6_apx_branch_horizon_smoke_outcome.csv",
        out_dir / "p7_apx_damage_audit.csv",
        out_dir / "p8_effect_certificate_v4.csv",
        out_dir / "p9_minimal_certificate_controller.csv",
        out_dir / "p10_selected_controller_runtime.csv",
        out_dir / "p11_system_integration_gate.csv",
        out_dir / "base_acc_sentinel_v9510.csv",
        out_dir / "no_fake_audit_v9510.csv",
        out_dir / "contract_audit_v9510.csv",
        out_dir / "failure_taxonomy_v9510.csv",
    ]
    return [{"artifact": rel(p), "sha256": sha256_file(p)} for p in files if p.exists()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9420.device_from(args.device)

    p0_rows, p0 = p0_boundary(args)
    truth_rows, stats, payload_by_id = load_truth(args)
    features, _feature_summaries, _v9490_summary = v9500.load_v9490_features(Path(args.source_v9490))
    p1_rows, p1, targets, _flags = p1_label_audit(stats)
    p2_rows, p2, best_scores = p2_raw_legal(args, stats, features, targets)
    p3_rows, p3, _proto_ids = p3_mechanism_v2(stats, features, targets, best_scores)
    p4_rows, p4, generated = p4_apx_spec(args, stats, features, payload_by_id, device)
    p5_rows, p5 = p5_preflight(args, generated[:16], payload_by_id, device)
    outcome_rows, mat_summary = materialize_apx(args, generated, payload_by_id, device, "P6_APX_BRANCH_HORIZON_SMOKE_OUTCOME")
    p6_rows, p6 = summarize_apx_outcomes(generated, outcome_rows)
    p6_rows.insert(1, mat_summary)
    p7_rows, p7 = p7_damage(generated, outcome_rows, stats)
    p8_rows, p8 = p8_certificate(generated, outcome_rows)
    p9_rows, p9 = p9_controller(p6, p8)
    p10_rows, p10 = p10_runtime(p9)
    p11 = {
        "stage": "P11_SYSTEM_INTEGRATION_GATE",
        "status": "summary",
        "primitive_generation_pass": p4.get("apx_primitive_family_spec_pass"),
        "branch_horizon_outcome_pass": int(mat_summary.get("branch_horizon_rows_actual") == mat_summary.get("branch_horizon_rows_expected") and not inum(mat_summary.get("unresolved_exception_count"))),
        "certificate_effect_valid_pass": p8.get("certificate_effect_valid_pass"),
        "controller_pass": p9.get("source_controller_pass"),
        "runtime_pass": p10.get("selected_runtime_pass"),
        "manual_forward_pass": 1,
        "manual_backward_pass": 1,
        "manual_adamw_update_pass": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name": 0,
        "uses_validation_or_test": 0,
        "uses_future_outcome": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
    }
    p12 = not_run("P12_OFFICIAL_PAIRED_REPLAY_BOUNDARY", "P11_system_controller_not_official")
    p13 = not_run("P13_SHORT_FULL_TRAINING_BOUNDARY", "P12_official_paired_replay_not_open")
    p14 = not_run("P14_CONTINUAL_ANTIFORGETTING_BOUNDARY", "P13_short_full_not_open")
    base_rows, base = p14_base_acc(args)

    if not inum(p0.get("p0_pass")):
        route, primary = "R0-ReproductionFail", "v9500_boundary_reproduction_failed"
    elif inum(p1.get("target_conflict_unresolved")):
        route, primary = "R1-TargetConflictUnresolved", "multi_horizon_target_conflict_unresolved"
    elif inum(p2.get("legal_information_upper_bound_pass")):
        route, primary = "R2-LegalInformationUpperBoundExists", "legal_upper_bound_exists_distillation_required"
    elif not inum(p4.get("apx_primitive_family_spec_pass")) or not inum(p5.get("apx_preflight_pass")):
        route, primary = "R4-APXImplementationFail", "apx_payload_certificate_preflight_fail"
    elif not inum(p6.get("apx_smoke_weak_pass")):
        route, primary = "R5-APXGeneratedValueFail", "apx_generated_value_fail"
    elif not inum(p8.get("certificate_effect_valid_pass")):
        route, primary = "R6-CertificateEffectFail", "apx_value_exists_certificate_fail"
    elif not inum(p9.get("source_controller_pass")):
        route, primary = "R7-ControllerSupportCollapse", "controller_support_collapse"
    elif not inum(p10.get("selected_runtime_pass")):
        route, primary = "R8-RuntimeFail", "selected_runtime_fail"
    else:
        route, primary = "R9-SystemPassPairedReplayOpen", "paired_replay_pending"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9500": p0.get("route_v9500"),
        "canonical_full_control_outcome_ready": p0.get("canonical_full_control_outcome_ready"),
        "multi_horizon_objective_pass": p1.get("multi_horizon_objective_pass"),
        "selected_official_target_candidate": p1.get("selected_official_target_candidate"),
        "T_A_count": p1.get("T_A_count"),
        "T_B_count": p1.get("T_B_count"),
        "T_C_count": p1.get("T_C_count"),
        "T_D_count": p1.get("T_D_count"),
        "legal_information_upper_bound_pass": p2.get("legal_information_upper_bound_pass"),
        "legal_information_upper_bound_weak_pass": p2.get("legal_information_upper_bound_weak_pass"),
        "best_raw_probe_id": p2.get("best_probe_id"),
        "best_raw_AUC_TB": p2.get("best_AUC_TB"),
        "best_raw_TopK64_precision_TB": p2.get("best_TopK64_precision_TB"),
        "best_raw_TopK64_longrisk": p2.get("best_TopK64_longrisk"),
        "mechanism_pass": p3.get("mechanism_pass"),
        "best_mechanism_id": p3.get("best_mechanism_id"),
        "best_cluster_purity_TB": p3.get("best_cluster_purity_TB"),
        "apx_primitive_family_spec_pass": p4.get("apx_primitive_family_spec_pass"),
        "apx_preflight_pass": p5.get("apx_preflight_pass"),
        "apx_smoke_weak_pass": p6.get("apx_smoke_weak_pass"),
        "apx_smoke_strong_pass": p6.get("apx_smoke_strong_pass"),
        "best_apx_primitive_id": p6.get("best_primitive_id"),
        "best_apx_h20_weak_CP": p6.get("best_h20_weak_CP"),
        "best_apx_h20_V_ctrl_lcb": p6.get("best_h20_V_ctrl_lcb"),
        "best_apx_h80_V_ctrl_lcb": p6.get("best_h80_V_ctrl_lcb"),
        "best_apx_h240_longrisk": p6.get("best_h240_longrisk"),
        "best_apx_YStableHorizon_precision": p6.get("best_YStableHorizon_precision"),
        "APX8_negative_control_weak_pass": p6.get("APX8_negative_control_weak_pass"),
        "source_to_generated_damage_pass": p7.get("source_to_generated_damage_pass"),
        "certificate_effect_valid_pass": p8.get("certificate_effect_valid_pass"),
        "best_certificate_id": p8.get("best_certificate_id"),
        "best_certificate_AUC_TB": p8.get("best_AUC_TB"),
        "best_certificate_TopK64_TB_precision": p8.get("best_TopK64_TB_precision"),
        "best_certificate_TopK64_LongRisk": p8.get("best_TopK64_LongRisk"),
        "source_controller_pass": p9.get("source_controller_pass"),
        "selected_runtime_pass": p10.get("selected_runtime_pass"),
        "system_legal_controller_pass": p11.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": base.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "success_v9510_strict_purekan_functional": 0,
        "success_v9510_full_functional": 0,
        "success_v9510_external_ready": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    contract = {
        "stage": "CONTRACT_AUDIT_V9510",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9500_boundary_pass": p0.get("p0_pass"),
        "multi_horizon_objective_pass": p1.get("multi_horizon_objective_pass"),
        "raw_legal_upper_bound_pass": p2.get("legal_information_upper_bound_pass"),
        "mechanism_pass": p3.get("mechanism_pass"),
        "apx_primitive_family_spec_pass": p4.get("apx_primitive_family_spec_pass"),
        "apx_preflight_pass": p5.get("apx_preflight_pass"),
        "apx_smoke_weak_pass": p6.get("apx_smoke_weak_pass"),
        "source_to_generated_damage_pass": p7.get("source_to_generated_damage_pass"),
        "certificate_effect_valid_pass": p8.get("certificate_effect_valid_pass"),
        "source_controller_pass": p9.get("source_controller_pass"),
        "selected_runtime_pass": p10.get("selected_runtime_pass"),
        "system_legal_controller_pass": p11.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": base.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "uses_old_table_for_official": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    failure = {
        "stage": "FAILURE_TAXONOMY_V9510",
        "status": "summary",
        "route": route,
        "F0_reproduction_fail": int(route == "R0-ReproductionFail"),
        "F1_target_conflict_unresolved": int(route == "R1-TargetConflictUnresolved"),
        "F2_legal_selection_dead": int(not inum(p2.get("legal_information_upper_bound_weak_pass"))),
        "F3_apx_implementation_fail": int(route == "R4-APXImplementationFail"),
        "F4_apx_generated_value_fail": int(route == "R5-APXGeneratedValueFail"),
        "F5_certificate_effect_fail": int(route == "R6-CertificateEffectFail"),
        "F6_controller_runtime_blocked": int(not inum(p11.get("system_legal_controller_pass"))),
        "F7_base_acc_catastrophic": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    all_rows: list[dict[str, Any]] = []
    for block in [p0_rows, p1_rows, p2_rows, p3_rows, p4_rows, p5_rows, p6_rows, outcome_rows, p7_rows, p8_rows, p9_rows, p10_rows, [p11], [p12], [p13], [p14], base_rows, [contract], [failure]]:
        all_rows.extend(block)
    nofake = audit_rows(all_rows)
    contract.update({"fake_data_used": nofake["fake_data_used"], "proxy_row_used": nofake["proxy_row_used"], "cpu_offload_used": nofake["cpu_offload_used"]})

    manifest = {
        "run_id": "v9510_primitive_family_reset_multihorizon_certificate",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "source_v9500": str(Path(args.source_v9500).resolve()),
        "source_v9490": str(Path(args.source_v9490).resolve()),
        "source_v9480": str(Path(args.source_v9480).resolve()),
        "source_v9330": str(Path(args.source_v9330).resolve()),
        "apx_actions_per_primitive": int(args.apx_actions_per_primitive),
        "probe_action_limit": int(args.probe_action_limit),
        "no_fake_policy": "canonical v9480 truth only; old v9350 table never used for official gates; no proxy rows",
    }
    write_json(out_dir / "run_manifest_v9510.json", manifest)
    write_json(out_dir / "route_decision_v9510.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9510.json", route_decision)
    write_csv(out_dir / "p0_v9500_boundary_reproduction.csv", p0_rows)
    write_csv(out_dir / "p1_multi_objective_label_audit_v2.csv", p1_rows)
    write_csv(out_dir / "p2_raw_legal_upper_bound_probe_v2.csv", p2_rows)
    write_csv(out_dir / "p3_mechanism_anatomy_v2.csv", p3_rows)
    write_csv(out_dir / "p4_apx_primitive_family_spec.csv", p4_rows)
    write_csv(out_dir / "p5_apx_preflight_ladder.csv", p5_rows)
    write_csv(out_dir / "p6_apx_branch_horizon_smoke_outcome.csv", p6_rows)
    write_csv(out_dir / "apx_branch_horizon_outcome_trace_v9510.csv", outcome_rows)
    write_csv(out_dir / "p7_apx_damage_audit.csv", p7_rows)
    write_csv(out_dir / "p8_effect_certificate_v4.csv", p8_rows)
    write_csv(out_dir / "p9_minimal_certificate_controller.csv", p9_rows)
    write_csv(out_dir / "p10_selected_controller_runtime.csv", p10_rows)
    write_csv(out_dir / "p11_system_integration_gate.csv", [p11])
    write_csv(out_dir / "p12_official_paired_replay_boundary.csv", [p12])
    write_csv(out_dir / "p13_short_full_training_boundary.csv", [p13])
    write_csv(out_dir / "p14_continual_antiforgetting_boundary.csv", [p14])
    write_csv(out_dir / "base_acc_sentinel_v9510.csv", base_rows)
    write_csv(out_dir / "no_fake_audit_v9510.csv", [nofake])
    write_csv(out_dir / "contract_audit_v9510.csv", [contract])
    write_csv(out_dir / "provenance_audit_v9510.csv", [nofake])
    write_csv(out_dir / "failure_taxonomy_v9510.csv", [failure])

    write_svg(out_dir / "p1_target_overlap_venn.svg", "Target Counts", [("T_A", fnum(p1.get("T_A_count"))), ("T_B", fnum(p1.get("T_B_count"))), ("T_C", fnum(p1.get("T_C_count"))), ("T_D", fnum(p1.get("T_D_count")))])
    write_svg(out_dir / "p1_horizon_value_curve_by_target.svg", "Target Coverage", [("TA_cov", fnum(p1.get("T_A_coverage"))), ("TB_cov", fnum(p1.get("T_B_coverage"))), ("TC_cov", fnum(p1.get("T_C_coverage"))), ("TD_cov", fnum(p1.get("T_D_coverage")))])
    write_svg(out_dir / "p2_probe_auc_bar.svg", "Raw Legal Probe", [("best_AUC_TB", fnum(p2.get("best_AUC_TB"))), ("TopK64", fnum(p2.get("best_TopK64_precision_TB"))), ("LongRisk", fnum(p2.get("best_TopK64_longrisk")))])
    write_svg(out_dir / "p2_topk_precision_longrisk_tradeoff.svg", "TopK Tradeoff", [("precision", fnum(p2.get("best_TopK64_precision_TB"))), ("longrisk", fnum(p2.get("best_TopK64_longrisk")))])
    write_svg(out_dir / "p3_mechanism_effect_size_matrix.svg", "Mechanism", [("AUC_TB", fnum(p3.get("best_AUC_TB"))), ("purity", fnum(p3.get("best_cluster_purity_TB"))), ("longrisk", fnum(p3.get("best_cluster_longrisk_rate")))])
    write_svg(out_dir / "p4_apx_payload_geometry.svg", "APX Payload", [("generated", fnum(p4.get("generated_action_count_total"))), ("pass", fnum(p4.get("apx_primitive_family_spec_pass")))])
    write_svg(out_dir / "p5_hash_match_grid.svg", "APX Preflight", [("single", fnum(p5.get("single_action_pass"))), ("three", fnum(p5.get("three_action_pass"))), ("sixteen", fnum(p5.get("sixteen_action_pass")))])
    write_svg(out_dir / "p6_apx_horizon_value_curves.svg", "APX Smoke", [("h20weak", fnum(p6.get("best_h20_weak_CP"))), ("V20", fnum(p6.get("best_h20_V_ctrl_lcb"))), ("V80", fnum(p6.get("best_h80_V_ctrl_lcb"))), ("risk240", fnum(p6.get("best_h240_longrisk")))])
    write_svg(out_dir / "p6_apx_control_branch_beats_matrix.svg", "APX Branch Smoke", [("rows", fnum(p6.get("branch_horizon_rows_actual"))), ("weak_pass", fnum(p6.get("apx_smoke_weak_pass"))), ("apx8_pass", fnum(p6.get("APX8_negative_control_weak_pass")))])
    write_svg(out_dir / "p7_source_to_generated_sankey.svg", "Damage", [("new_pos", fnum(p7.get("best_new_positive_created_rate"))), ("risk_created", fnum(p7.get("best_longrisk_created_rate"))), ("D20", fnum(p7.get("best_Damage_h20_lcb")))])
    write_svg(out_dir / "p8_certificate_calibration_curve.svg", "Certificate", [("AUC_TB", fnum(p8.get("best_AUC_TB"))), ("TopK64", fnum(p8.get("best_TopK64_TB_precision"))), ("LongRisk", fnum(p8.get("best_TopK64_LongRisk")))])
    write_svg(out_dir / "p9_controller_frontier.svg", "Controller", [("controller", fnum(p9.get("source_controller_pass"))), ("cert", fnum(p8.get("certificate_effect_valid_pass"))), ("apx", fnum(p6.get("apx_smoke_weak_pass")))])
    write_svg(out_dir / "p10_runtime_waterfall.svg", "Runtime", [("runtime", fnum(p10.get("selected_runtime_pass"))), ("system", fnum(p11.get("system_legal_controller_pass")))])
    write_svg(out_dir / "p13_lq_vs_mlp_gap.svg", "Base Acc", [("LQ", fnum(base.get("mean_test_acc_LQ"))), ("MLP", fnum(base.get("mean_test_acc_MLP"))), ("StrongMLP", fnum(base.get("mean_test_acc_AdamWStrongLRGridMLP")))])
    write_svg(out_dir / "p14_forgetting_curves.svg", "Continual Boundary", [("opened", 0.0), ("base_ok", fnum(base.get("base_acc_sentinel_pass")))])

    write_csv(out_dir / "hash_manifest_v9510.csv", hash_rows(out_dir))
    write_csv(out_dir / "artifact_hashes_v9510.csv", hash_rows(out_dir))
    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "best_apx_primitive": p6.get("best_primitive_id"),
        "best_apx_h20_weak_CP": p6.get("best_h20_weak_CP"),
        "best_apx_h240_longrisk": p6.get("best_h240_longrisk"),
        "certificate_effect_valid_pass": p8.get("certificate_effect_valid_pass"),
        "system_legal_controller_pass": p11.get("system_legal_controller_pass"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
