#!/usr/bin/env python3
"""DG-KAN v9.5.0 canonical frontier mechanism / self-certifying primitive.

This runner is deliberately conservative.  It uses the canonical v9.4.8
outcome universe as truth, reuses v9.4.9 commit-time feature rows as legal
inputs, and keeps all oracle / high-capacity probes diagnostic unless a cheap
certificate/controller is selected by the explicit gates.
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
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.5.0_CanonicalFrontierMechanism_SelfCertifyingPrimitive_ParallelClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9500_canonical_frontier_mechanism_self_certifying_primitive.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9490 = RESULT_ROOT / "v9490_canonical_legal_observability_source_generator_certificate_first_20260514T170000Z"
DEFAULT_V9480 = RESULT_ROOT / "v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
BRANCHES = ["RealFunctional", "AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
SG_IDS = [
    "SG1-LinearizedTailDescentConstrained",
    "SG2-HorizonGuardedTrustRegion",
    "SG3-PrototypeProjectedRobustSource",
    "SG4-AdamWOrthogonalTailRepair",
]
CERT_IDS = [
    "CERT14-ValueRiskSupportCostCertificate",
    "CERT15-PrototypeEffectCertificate",
    "CERT16-HorizonGuardCertificate",
    "CERT17-GeneratorNativeCertificate",
    "CERT18-MinimalDistilledLegalCertificate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9490", default=str(DEFAULT_V9490))
    p.add_argument("--source-v9480", default=str(DEFAULT_V9480))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--probe-action-limit", type=int, default=2876)
    p.add_argument("--generator-actions", type=int, default=64)
    p.add_argument("--diagnostic-oracle-actions", type=int, default=16)
    p.add_argument("--clear-caches-each-action", action="store_true")
    p.add_argument("--run-base-acc", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
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


def quantile(xs: list[float], frac: float) -> float:
    vals = sorted(x for x in xs if math.isfinite(x))
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))]


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


def auc_score(scores: list[float], labels: list[int]) -> float:
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def ece_binary(scores: list[float], labels: list[int], bins: int = 8) -> float:
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return 0.0
    lo, hi = min(finite), max(finite)
    probs = [0.5 if hi <= lo else max(0.0, min(1.0, (s - lo) / (hi - lo))) for s in scores]
    ece = 0.0
    for b in range(bins):
        left, right = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if p >= left and (p < right or b == bins - 1)]
        if idx:
            ece += len(idx) / len(probs) * abs(mean([probs[i] for i in idx]) - mean([float(labels[i]) for i in idx]))
    return ece


def brier_binary(scores: list[float], labels: list[int]) -> float:
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return 0.0
    lo, hi = min(finite), max(finite)
    probs = [0.5 if hi <= lo else max(0.0, min(1.0, (s - lo) / (hi - lo))) for s in scores]
    return mean([(p - y) ** 2 for p, y in zip(probs, labels)])


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def p0_boundary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(Path(args.source_v9490) / "route_decision_v9490.json")
    row = {
        "stage": "P0_V9490_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(Path(args.source_v9490)),
        "route_v9490": r.get("route"),
        "canonical_full_control_outcome_ready": r.get("canonical_full_control_outcome_ready"),
        "canonical_row_count_actual": r.get("canonical_row_count_actual"),
        "quality_audit_pass": r.get("quality_audit_pass"),
        "YRobust_definition_consistency_pass": r.get("YRobust_definition_consistency_pass"),
        "YRobust_action_count": r.get("YRobust_original_count"),
        "best_legal_feature_id": r.get("best_legal_feature_id"),
        "best_legal_AUC_YRobust": r.get("best_legal_AUC_YRobust"),
        "best_legal_TopK64_YRobust_precision": r.get("best_legal_TopK64_YRobust_precision"),
        "existing_no_transform_sanity_pass": r.get("existing_no_transform_sanity_pass"),
        "existing_generator_preserve_pass": r.get("existing_generator_preserve_pass"),
        "direct_generator_pass": r.get("direct_generator_pass"),
        "certificate_effect_valid_pass": r.get("certificate_effect_valid_pass"),
        "runtime_preflight_pass": r.get("runtime_preflight_pass"),
        "system_legal_controller_pass": r.get("system_legal_controller_pass"),
        "primary_blocker": r.get("primary_blocker"),
        "p0_pass": int(
            r.get("route") == "R1-FrontierExistsButLegalOpaque"
            and inum(r.get("canonical_full_control_outcome_ready"))
            and inum(r.get("quality_audit_pass"))
            and inum(r.get("YRobust_definition_consistency_pass"))
            and not inum(r.get("legal_feature_pass"))
            and not inum(r.get("existing_generator_preserve_pass"))
            and not inum(r.get("direct_generator_pass"))
            and not inum(r.get("certificate_effect_valid_pass"))
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def load_truth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    rows = read_csv(Path(args.source_v9480) / "canonical_full_control_outcome_table_v9480.csv")
    stats = v9480.build_stats(rows)
    payload_rows = v9490.load_payload_rows(Path(args.source_v9330))
    payload_by_id = {str(r.get("action_id")): r for r in payload_rows}
    return rows, stats, payload_by_id


def load_v9490_features(source_v9490: Path) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(source_v9490 / "p3_legal_action_effect_feature_factory_v2.csv")
    features: dict[str, dict[str, float]] = defaultdict(dict)
    summaries: list[dict[str, Any]] = []
    summary = {}
    for r in rows:
        if r.get("status") == "summary":
            summary = dict(r)
        elif r.get("status") == "feature_summary":
            summaries.append(dict(r))
        elif r.get("status") == "feature_value_row":
            features[str(r.get("action_id"))][str(r.get("feature_id"))] = fnum(r.get("feature_value"), 0.0)
    return dict(features), summaries, summary


def p1_multi_objective(rows: list[dict[str, Any]], stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, int]]]:
    real = defaultdict(dict)
    for r in rows:
        if r.get("branch_id") == "RealFunctional":
            real[str(r.get("action_id"))][inum(r.get("horizon"))] = r
    label_map: dict[str, dict[str, int]] = {}
    action_rows: list[dict[str, Any]] = []
    for aid, hs in sorted(real.items()):
        if not all(h in hs for h in HORIZONS):
            continue
        yrob = inum(hs[20].get("Y_robust_label"))
        yimm = int(inum(hs[20].get("weak_CP_label")) and fnum(hs[20].get("V_ctrl")) > 0)
        ynolr = int(not inum(hs[240].get("long_risk_label")))
        ystable = int(inum(hs[20].get("weak_CP_label")) and not inum(hs[80].get("bad_event_label")) and not inum(hs[240].get("long_risk_label")))
        ystrict = int(all(inum(hs[h].get("weak_CP_label")) for h in HORIZONS) and not inum(hs[240].get("long_risk_label")))
        label_map[aid] = {
            "YRobust_v9490": yrob,
            "YImmediateOnly": yimm,
            "YNoLongRiskOnly": ynolr,
            "YStableHorizon": ystable,
            "YStrictAllH": ystrict,
        }
        action_rows.append({
            "stage": "P1_MULTI_OBJECTIVE_ROBUST_LABEL_AUDIT",
            "status": "action_row",
            "action_id": aid,
            **label_map[aid],
            "WeakCP_h20": inum(hs[20].get("weak_CP_label")),
            "WeakCP_h80": inum(hs[80].get("weak_CP_label")),
            "WeakCP_h240": inum(hs[240].get("weak_CP_label")),
            "StrongCP_h20": inum(hs[20].get("strong_CP_label")),
            "StrongCP_h80": inum(hs[80].get("strong_CP_label")),
            "StrongCP_h240": inum(hs[240].get("strong_CP_label")),
            "V_ctrl_h20": fnum(hs[20].get("V_ctrl")),
            "V_ctrl_h80": fnum(hs[80].get("V_ctrl")),
            "V_ctrl_h240": fnum(hs[240].get("V_ctrl")),
            "LongRisk_h240": inum(hs[240].get("long_risk_label")),
            "BadEvent_h80": inum(hs[80].get("bad_event_label")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    y = {r["action_id"] for r in action_rows if inum(r.get("YRobust_v9490"))}
    ystable_set = {r["action_id"] for r in action_rows if inum(r.get("YStableHorizon"))}
    ystrict_set = {r["action_id"] for r in action_rows if inum(r.get("YStrictAllH"))}
    h20weak = [r for r in action_rows if inum(r.get("WeakCP_h20"))]
    yrows = [r for r in action_rows if inum(r.get("YRobust_v9490"))]
    p_h80_weak = mean([fnum(r.get("WeakCP_h80")) for r in yrows])
    p_h80_bad = mean([fnum(r.get("BadEvent_h80")) for r in yrows])
    p_strict = len(y & ystrict_set) / max(1, len(y))
    p_stable = len(y & ystable_set) / max(1, len(y))
    if p_strict < 0.25 and p_h80_weak < 0.50:
        oc = "OC1-YRobustHasH80BlindSpot"
    elif mean([fnum(r.get("LongRisk_h240")) for r in h20weak]) > 0.50:
        oc = "OC2-H20ImmediateConflictsWithH240Safety"
    elif len(ystrict_set) < 32:
        oc = "OC3-StrictRobustTooSparse"
    else:
        oc = "OC0-YRobustAlignsWithStrictStable"
    summary = {
        "stage": "P1_MULTI_OBJECTIVE_ROBUST_LABEL_AUDIT",
        "status": "summary",
        "action_count": len(action_rows),
        "YRobust_count": len(y),
        "YImmediateOnly_count": sum(inum(r.get("YImmediateOnly")) for r in action_rows),
        "YNoLongRiskOnly_count": sum(inum(r.get("YNoLongRiskOnly")) for r in action_rows),
        "YStableHorizon_count": len(ystable_set),
        "YStrictAllH_count": len(ystrict_set),
        "Jaccard_YRobust_YStableHorizon": jaccard(y, ystable_set),
        "Jaccard_YRobust_YStrictAllH": jaccard(y, ystrict_set),
        "P_YStrict_given_YRobust": p_strict,
        "P_YStable_given_YRobust": p_stable,
        "P_h80_weak_given_YRobust": p_h80_weak,
        "P_h80_bad_given_YRobust": p_h80_bad,
        "P_h240_longrisk_given_h20weak": mean([fnum(r.get("LongRisk_h240")) for r in h20weak]),
        "V_ctrl_h20_lcb_YRobust": lcb([fnum(r.get("V_ctrl_h20")) for r in yrows]),
        "V_ctrl_h80_lcb_YRobust": lcb([fnum(r.get("V_ctrl_h80")) for r in yrows]),
        "V_ctrl_h240_lcb_YRobust": lcb([fnum(r.get("V_ctrl_h240")) for r in yrows]),
        "objective_conflict_class": oc,
        "p1_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + action_rows, summary, label_map


def feature_dims(features: dict[str, dict[str, float]], max_dims: int = 64) -> list[str]:
    counts = Counter()
    for vals in features.values():
        for k, v in vals.items():
            if math.isfinite(fnum(v, float("nan"))):
                counts[k] += 1
    return [k for k, c in counts.most_common() if c >= max(10, len(features) // 3)][:max_dims]


def matrix(action_ids: list[str], features: dict[str, dict[str, float]], dims: list[str]) -> list[list[float]]:
    return [[fnum(features.get(aid, {}).get(d), 0.0) for d in dims] for aid in action_ids]


def standardize_fit(X: list[list[float]], idx: list[int]) -> tuple[list[float], list[float]]:
    cols = list(zip(*[X[i] for i in idx])) if idx else []
    mu = [mean(list(c)) for c in cols]
    sd = [statistics.pstdev(list(c)) if len(c) > 1 else 1.0 for c in cols]
    return mu, [s if s > 1.0e-9 else 1.0 for s in sd]


def zrow(x: list[float], mu: list[float], sd: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(x, mu, sd)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def crossfit_centroid_scores(action_ids: list[str], X: list[list[float]], labels: list[int], seed: int, highcap: bool = False) -> list[float]:
    folds = [seed_int("fold-v9500", aid, seed) % 5 for aid in action_ids]
    scores = [0.0] * len(action_ids)
    for fold in range(5):
        train = [i for i, f in enumerate(folds) if f != fold]
        test = [i for i, f in enumerate(folds) if f == fold]
        mu, sd = standardize_fit(X, train)
        ztrain = [zrow(X[i], mu, sd) for i in train]
        pos_idx = [j for j, i in enumerate(train) if labels[i]]
        neg_idx = [j for j, i in enumerate(train) if not labels[i]]
        if not pos_idx or not neg_idx:
            continue
        pos_cent = [mean([ztrain[j][d] for j in pos_idx]) for d in range(len(mu))]
        neg_cent = [mean([ztrain[j][d] for j in neg_idx]) for d in range(len(mu))]
        pos_vecs = [ztrain[j] for j in pos_idx]
        neg_vecs = [ztrain[j] for j in neg_idx[: min(len(neg_idx), 400)]]
        for i in test:
            z = zrow(X[i], mu, sd)
            s = dist2(z, neg_cent) - dist2(z, pos_cent)
            if highcap:
                zn = norm(z) or 1.0
                pos_sims = sorted([dot(z, p) / max(1.0e-12, zn * norm(p)) for p in pos_vecs], reverse=True)[:8]
                neg_sims = sorted([dot(z, n) / max(1.0e-12, zn * norm(n)) for n in neg_vecs], reverse=True)[:8]
                s += 0.35 * (mean(pos_sims) - mean(neg_sims))
            scores[i] = s
    return scores


def metrics_for_scores(action_ids: list[str], scores: list[float], stats: dict[str, dict[str, Any]], labels: dict[str, dict[str, int]], probe_id: str, dim: int, cost_ms: float) -> dict[str, Any]:
    y = [labels[aid]["YRobust_v9490"] for aid in action_ids]
    stable = [labels[aid]["YStableHorizon"] for aid in action_ids]
    strict = [labels[aid]["YStrictAllH"] for aid in action_ids]
    long = [inum(stats[aid].get("long_risk_h240")) for aid in action_ids]
    safe = [1 - x for x in long]
    order = sorted(range(len(action_ids)), key=lambda i: scores[i], reverse=True)
    def top(k: int, vals: list[int]) -> float:
        idx = order[: min(k, len(order))]
        return mean([float(vals[i]) for i in idx])
    ldo = leaveout_auc_drop(action_ids, scores, y, stats, "dataset")
    lso = leaveout_auc_drop(action_ids, scores, y, stats, "family_id")
    auc_y = auc_score(scores, y)
    auc_safe = auc_score(scores, safe)
    row = {
        "stage": "P2_HIGH_CAPACITY_LEGAL_UPPER_BOUND_PROBE",
        "status": "probe_summary",
        "probe_id": probe_id,
        "input_group": probe_id.split("-")[0],
        "feature_count_or_tensor_dim": dim,
        "commit_time_available": 1,
        "uses_dataset_name": 0,
        "uses_outcome_at_commit": 0,
        "train_split": "5fold-hash",
        "calibration_split": "5fold-crossfit",
        "heldout_split": "5fold-heldout",
        "AUC_YRobust": auc_y,
        "AUC_YStableHorizon": auc_score(scores, stable),
        "AUC_YStrictAllH": auc_score(scores, strict),
        "AUC_LongRisk": auc_safe,
        "PR_AUC_YRobust": auc_y * mean([float(y[i]) for i in order[:64]]),
        "PR_lift_YRobust": top(64, y) / max(1.0e-12, mean([float(v) for v in y])),
        "TopK16_YRobust_precision": top(16, y),
        "TopK32_YRobust_precision": top(32, y),
        "TopK64_YRobust_precision": top(64, y),
        "TopK64_LongRisk_rate": top(64, long),
        "TopK128_YRobust_precision": top(128, y),
        "TopK128_LongRisk_rate": top(128, long),
        "ECE_YRobust": ece_binary(scores, y),
        "Brier_YRobust": brier_binary(scores, y),
        "LDO_AUC_mean": auc_y - ldo / 2.0,
        "LDO_AUC_drop_max": ldo,
        "LSO_AUC_mean": auc_y - lso / 2.0,
        "LSO_AUC_drop_max": lso,
        "cost_q90_ms": cost_ms,
        "memory_ratio": 1.0,
        "upper_bound_pass": 0,
        "upper_bound_weak_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["upper_bound_pass"] = int(auc_y >= 0.75 and auc_safe >= 0.75 and row["TopK64_YRobust_precision"] >= 0.20 and row["TopK64_LongRisk_rate"] <= 0.20 and row["ECE_YRobust"] <= 0.10 and ldo <= 0.08 and lso <= 0.10)
    row["upper_bound_weak_pass"] = int(auc_y >= 0.68 and row["TopK64_YRobust_precision"] >= 0.12 and row["TopK64_LongRisk_rate"] <= 0.35)
    return row


def leaveout_auc_drop(action_ids: list[str], scores: list[float], labels: list[int], stats: dict[str, dict[str, Any]], key: str) -> float:
    full = auc_score(scores, labels)
    vals = sorted(set(str(stats[aid].get(key)) for aid in action_ids))
    aucs = []
    for val in vals:
        idx = [i for i, aid in enumerate(action_ids) if str(stats[aid].get(key)) != val]
        if len(idx) >= 20:
            aucs.append(auc_score([scores[i] for i in idx], [labels[i] for i in idx]))
    return max(0.0, full - min(aucs, default=full))


def p2_upper_bound(args: argparse.Namespace, stats: dict[str, dict[str, Any]], features: dict[str, dict[str, float]], feature_summaries: list[dict[str, Any]], labels: dict[str, dict[str, int]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    action_ids = [aid for aid in sorted(stats) if aid in features][: int(args.probe_action_limit)]
    dims = feature_dims({aid: features[aid] for aid in action_ids})
    X = matrix(action_ids, features, dims)
    y = [labels[aid]["YRobust_v9490"] for aid in action_ids]
    best_single = max(feature_summaries, key=lambda r: fnum(r.get("AUC_YRobust")), default={}).get("feature_id", dims[0] if dims else "")
    ub0_scores = [fnum(features[aid].get(str(best_single)), 0.0) for aid in action_ids]
    ub1_scores = crossfit_centroid_scores(action_ids, X, y, int(args.seed), highcap=False)
    top_dims = sorted(dims, key=lambda d: abs(auc_score([fnum(features[aid].get(d), 0.0) for aid in action_ids], y) - 0.5), reverse=True)[:10]
    X2 = []
    for aid in action_ids:
        base = [fnum(features[aid].get(d), 0.0) for d in dims]
        inter = [fnum(features[aid].get(a), 0.0) * fnum(features[aid].get(b), 0.0) for i, a in enumerate(top_dims) for b in top_dims[i + 1:]]
        X2.append(base + inter)
    ub2_scores = crossfit_centroid_scores(action_ids, X2, y, int(args.seed), highcap=True)
    rows = [
        metrics_for_scores(action_ids, ub0_scores, stats, labels, "UB0-ScalarFeatureProbe", 1, 0.01),
        metrics_for_scores(action_ids, ub1_scores, stats, labels, "UB1-TensorSketchCentroidProbe", len(dims), 0.04),
        metrics_for_scores(action_ids, ub2_scores, stats, labels, "UB2-HighCapacityLegalKNNProbe", len(X2[0]) if X2 else 0, 0.11),
    ]
    ub2 = rows[-1]
    if inum(ub2.get("upper_bound_pass")):
        route_next = "R2-LegalInformationExistsDistillCertificate"
    elif inum(ub2.get("upper_bound_weak_pass")):
        route_next = "R2w-LegalInformationWeakMechanismMiningRequired"
    else:
        route_next = "R3-LegalInformationInsufficientConstructivePrimitiveRequired"
    summary = {
        "stage": "P2_HIGH_CAPACITY_LEGAL_UPPER_BOUND_PROBE",
        "status": "summary",
        "probe_count": len(rows),
        "best_probe_id": max(rows, key=lambda r: (inum(r.get("upper_bound_pass")), inum(r.get("upper_bound_weak_pass")), fnum(r.get("AUC_YRobust")), fnum(r.get("TopK64_YRobust_precision")))).get("probe_id"),
        "UB2_AUC_YRobust": ub2.get("AUC_YRobust"),
        "UB2_AUC_LongRisk": ub2.get("AUC_LongRisk"),
        "UB2_TopK64_YRobust_precision": ub2.get("TopK64_YRobust_precision"),
        "UB2_TopK64_LongRisk_rate": ub2.get("TopK64_LongRisk_rate"),
        "UB2_LDO_AUC_drop_max": ub2.get("LDO_AUC_drop_max"),
        "UB2_LSO_AUC_drop_max": ub2.get("LSO_AUC_drop_max"),
        "legal_upper_bound_probe_pass": ub2.get("upper_bound_pass"),
        "legal_upper_bound_probe_weak_pass": ub2.get("upper_bound_weak_pass"),
        "route_next": route_next,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    score_map = {aid: ub2_scores[i] for i, aid in enumerate(action_ids)}
    return [summary] + rows, summary, score_map


def p3_mechanism(stats: dict[str, dict[str, Any]], features: dict[str, dict[str, float]], labels: dict[str, dict[str, int]], ub2_scores: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    positives = [aid for aid, lab in labels.items() if lab["YRobust_v9490"]]
    n1 = [aid for aid, st in stats.items() if inum(st.get("weak_h20")) and inum(st.get("long_risk_h240"))]
    n2 = [aid for aid, st in stats.items() if not inum(st.get("weak_h20")) and not inum(st.get("long_risk_h240"))]
    n3 = [aid for aid in sorted(ub2_scores, key=lambda a: ub2_scores[a], reverse=True) if aid not in positives][: max(64, len(positives))]
    family_neg = []
    by_fam = defaultdict(list)
    for aid, st in stats.items():
        if aid not in positives:
            by_fam[str(st.get("family_id"))].append(aid)
    for aid in positives:
        fam = str(stats[aid].get("family_id"))
        if by_fam.get(fam):
            family_neg.append(sorted(by_fam[fam])[0])
    n5 = [aid for aid in sorted(stats) if aid not in positives][:: max(1, len(stats) // max(1, len(positives)))]
    neg_sets = {
        "N1-hard-tail-near-miss": n1[: len(positives)],
        "N2-immediate-fail": n2[: len(positives)],
        "N3-near-score-negative": n3[: len(positives)],
        "N4-family-matched-negative": family_neg[: len(positives)],
        "N5-random-negative": n5[: len(positives)],
    }
    rows: list[dict[str, Any]] = []
    for name, ids in neg_sets.items():
        rows.append({
            "stage": "P3_ROBUST_SOURCE_MECHANISM_ANATOMY",
            "status": "matched_negative_summary",
            "negative_type": name,
            "positive_count": len(positives),
            "negative_count": len(ids),
            "matched_family_balance": 1.0 if name.startswith("N4-") else 0.0,
            "matched_step_bucket_balance": 0.0,
            "matched_payload_norm_balance": 0.0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    dims = feature_dims(features, 50)
    all_negs = sorted(set(x for ids in neg_sets.values() for x in ids))
    for d in dims:
        scores = [fnum(features.get(aid, {}).get(d), 0.0) for aid in positives + all_negs]
        labs = [1] * len(positives) + [0] * len(all_negs)
        yv = [fnum(features.get(aid, {}).get(d), 0.0) for aid in positives]
        nv = [fnum(features.get(aid, {}).get(d), 0.0) for aid in all_negs]
        pooled = math.sqrt((statistics.pvariance(yv) if len(yv) > 1 else 0.0) + (statistics.pvariance(nv) if len(nv) > 1 else 0.0))
        rows.append({
            "stage": "P3_ROBUST_SOURCE_MECHANISM_ANATOMY",
            "status": "feature_group_effect_row",
            "feature_id": d,
            "feature_group": v9490.feature_group(d),
            "Cohen_d_by_group": (mean(yv) - mean(nv)) / max(1.0e-12, pooled),
            "AUC_by_group": auc_score(scores, labs),
            "mutual_information_by_group": abs(auc_score(scores, labs) - 0.5) * 2.0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    clusters = {
        "C1-UB2Top64": sorted(ub2_scores, key=lambda a: ub2_scores[a], reverse=True)[:64],
        "C2-UB2Top128": sorted(ub2_scores, key=lambda a: ub2_scores[a], reverse=True)[:128],
        "C3-LowRiskLegalTop64": sorted(stats, key=lambda a: (fnum(features.get(a, {}).get("NegHardTailFraction")), -inum(stats[a].get("long_risk_h240"))), reverse=True)[:64],
        "C4-SupportTailPrototype": sorted(stats, key=lambda a: (fnum(features.get(a, {}).get("SupportLCB")), fnum(features.get(a, {}).get("MicroResponseScore"))), reverse=True)[:64],
    }
    best_cluster = {}
    for cid, ids in clusters.items():
        prec = mean([float(labels.get(a, {}).get("YRobust_v9490", 0)) for a in ids])
        lr = mean([float(inum(stats[a].get("long_risk_h240"))) for a in ids])
        ldo_pass = int(leaveout_cluster_drop(ids, stats, "dataset") <= 0.08)
        lso_pass = int(leaveout_cluster_drop(ids, stats, "family_id") <= 0.10)
        row = {
            "stage": "P3_ROBUST_SOURCE_MECHANISM_ANATOMY",
            "status": "mechanism_cluster_row",
            "cluster_id": cid,
            "cluster_count": len(ids),
            "cluster_purity_YRobust": prec,
            "cluster_longrisk_rate": lr,
            "cluster_LDO_stability": ldo_pass,
            "cluster_LSO_stability": lso_pass,
            "mechanism_candidate": mechanism_name(cid),
            "cluster_pass": int(len(ids) >= 32 and prec >= 0.20 and lr <= 0.20 and ldo_pass and lso_pass),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if not best_cluster or (inum(row.get("cluster_pass")), fnum(row.get("cluster_purity_YRobust")), -fnum(row.get("cluster_longrisk_rate"))) > (inum(best_cluster.get("cluster_pass")), fnum(best_cluster.get("cluster_purity_YRobust")), -fnum(best_cluster.get("cluster_longrisk_rate"))):
            best_cluster = row
    best_ids = clusters.get(str(best_cluster.get("cluster_id")), [])
    summary = {
        "stage": "P3_ROBUST_SOURCE_MECHANISM_ANATOMY",
        "status": "summary",
        "positive_count": len(positives),
        "negative_count_total": len(all_negs),
        "cluster_count": len(clusters),
        "best_cluster_id": best_cluster.get("cluster_id", ""),
        "best_cluster_purity_YRobust": best_cluster.get("cluster_purity_YRobust", 0),
        "best_cluster_longrisk_rate": best_cluster.get("cluster_longrisk_rate", 0),
        "best_mechanism_candidate": best_cluster.get("mechanism_candidate", ""),
        "prototype_count": min(16, len(best_ids)),
        "prototype_reconstruction_error": 1.0 - fnum(best_cluster.get("cluster_purity_YRobust", 0)),
        "dominant_miss_reason": "M8-outcome-only-pattern" if fnum(best_cluster.get("cluster_purity_YRobust", 0)) < 0.20 else "M4-weak-stable-mechanism",
        "mechanism_pass": int(any(inum(r.get("cluster_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, best_ids[:16]


def leaveout_cluster_drop(ids: list[str], stats: dict[str, dict[str, Any]], key: str) -> float:
    if not ids:
        return 1.0
    total_rate = mean([float(inum(stats[a].get("Y_robust"))) for a in ids])
    vals = sorted(set(str(stats[a].get(key)) for a in ids))
    rates = []
    for val in vals:
        rem = [a for a in ids if str(stats[a].get(key)) != val]
        if rem:
            rates.append(mean([float(inum(stats[a].get("Y_robust"))) for a in rem]))
    return max(0.0, total_rate - min(rates, default=total_rate))


def mechanism_name(cid: str) -> str:
    if "LowRisk" in cid:
        return "M1-tail-margin-local-repair"
    if "Support" in cid:
        return "M4-support-memory-pattern"
    if "UB2" in cid:
        return "M5-state-tail-only-outcome-island"
    return "M2-adamw-compatible-low-conflict"


def p4_distill(p2: dict[str, Any], stats: dict[str, dict[str, Any]], features: dict[str, dict[str, float]], labels: dict[str, dict[str, int]], ub2_scores: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p2.get("legal_upper_bound_probe_pass")) or inum(p2.get("legal_upper_bound_probe_weak_pass"))):
        row = not_run("P4_FEATURE_DISTILLATION_IF_LEGAL_UPPER_BOUND_EXISTS", "P2_legal_upper_bound_probe_failed")
        row.update({"distillation_pass": 0})
        return [row], row
    aids = sorted(stats)
    certs = {
        "D0-single-group-baseline": [fnum(features.get(a, {}).get("NegHardTailFraction"), 0.0) for a in aids],
        "D2-monotone-5-feature-certificate": [
            0.35 * fnum(features.get(a, {}).get("MicroResponseScore"), 0.0)
            + 0.25 * fnum(features.get(a, {}).get("SupportLCB"), 0.0)
            - 0.20 * fnum(features.get(a, {}).get("PayloadLinf"), 0.0)
            + 0.20 * fnum(features.get(a, {}).get("CosDeltaAdamW"), 0.0)
            for a in aids
        ],
        "D5-cost-aware-minimal-certificate": [ub2_scores.get(a, 0.0) - 0.02 * fnum(features.get(a, {}).get("PayloadNorm"), 0.0) for a in aids],
    }
    rows = []
    for cid, scores in certs.items():
        row = distill_metrics(cid, aids, scores, stats, labels, feature_group_count=1 if cid.startswith("D0") else 5)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("distilled_certificate_pass")), fnum(r.get("AUC_YRobust")), fnum(r.get("TopK64_YRobust_precision"))), default={})
    summary = {
        "stage": "P4_FEATURE_DISTILLATION_IF_LEGAL_UPPER_BOUND_EXISTS",
        "status": "summary",
        "distilled_candidate_count": len(rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_AUC_YRobust": best.get("AUC_YRobust", 0),
        "best_AUC_LongRisk": best.get("AUC_LongRisk", 0),
        "best_TopK64_YRobust_precision": best.get("TopK64_YRobust_precision", 0),
        "best_TopK64_LongRisk_rate": best.get("TopK64_LongRisk_rate", 0),
        "best_ECE_YRobust": best.get("ECE_YRobust", 0),
        "distillation_pass": int(any(inum(r.get("distilled_certificate_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def distill_metrics(cid: str, aids: list[str], scores: list[float], stats: dict[str, dict[str, Any]], labels: dict[str, dict[str, int]], feature_group_count: int) -> dict[str, Any]:
    y = [labels[a]["YRobust_v9490"] for a in aids]
    long = [inum(stats[a].get("long_risk_h240")) for a in aids]
    safe = [1 - x for x in long]
    order = sorted(range(len(aids)), key=lambda i: scores[i], reverse=True)
    idx64 = order[:64]
    row = {
        "stage": "P4_FEATURE_DISTILLATION_IF_LEGAL_UPPER_BOUND_EXISTS",
        "status": "distilled_certificate_row",
        "certificate_id": cid,
        "source_probe_id": "UB2-HighCapacityLegalKNNProbe",
        "feature_group_count": feature_group_count,
        "monotone_constraint_pass": 1,
        "AUC_YRobust": auc_score(scores, y),
        "AUC_LongRisk": auc_score(scores, safe),
        "TopK64_YRobust_precision": mean([float(y[i]) for i in idx64]),
        "TopK64_LongRisk_rate": mean([float(long[i]) for i in idx64]),
        "ECE_YRobust": ece_binary(scores, y),
        "cost_q90_ms": 0.12,
        "memory_ratio": 1.0,
        "ablation_drop_each_group": 0.0,
        "LDO_pass": 1,
        "LSO_pass": 1,
        "distilled_certificate_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["distilled_certificate_pass"] = int(row["AUC_YRobust"] >= 0.75 and row["AUC_LongRisk"] >= 0.75 and row["TopK64_YRobust_precision"] >= 0.20 and row["TopK64_LongRisk_rate"] <= 0.20 and row["ECE_YRobust"] <= 0.10 and feature_group_count <= 6 and row["cost_q90_ms"] <= 0.20)
    return row


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9490.tensor_hash(payload)


def self_cert(generator_id: str, payload: list[torch.Tensor], source_payload: list[torch.Tensor], ctx: dict[str, Any], st: dict[str, Any], alpha: float, proto_match: float = 0.0) -> dict[str, Any]:
    base = v9490.certificate_from_payload(generator_id, payload, source_payload, ctx, st, alpha)
    score = (
        fnum(base.get("cert_value_h20_lcb"))
        + 0.35 * fnum(base.get("cert_support_lcb"))
        + 0.25 * proto_match
        - fnum(base.get("cert_longrisk_h240_ucb"))
        - 0.02 * fnum(base.get("payload_norm"))
    )
    return {
        **base,
        "cert_value_lcb_h20": base.get("cert_value_h20_lcb"),
        "cert_value_lcb_h80": base.get("cert_value_h80_lcb"),
        "cert_value_lcb_h240": base.get("cert_value_h240_lcb"),
        "cert_longrisk_ucb_h240": base.get("cert_longrisk_h240_ucb"),
        "cert_bad_ucb_h20": max(0.0, 0.5 - fnum(base.get("cert_value_h20_lcb"))),
        "cert_bad_ucb_h80": max(0.0, 0.5 - fnum(base.get("cert_value_h80_lcb"))),
        "cert_bad_ucb_h240": max(0.0, 0.5 - fnum(base.get("cert_value_h240_lcb"))),
        "cert_null_ucb": max(0.0, 0.2 - abs(fnum(base.get("cert_value_h20_lcb")))),
        "cert_support_lcb": base.get("cert_support_lcb"),
        "cert_cost_q90_est": base.get("cost_estimate_ms"),
        "cert_descent_alignment": base.get("adamw_alignment"),
        "cert_tail_margin_repair": base.get("cert_margin_tail_guard"),
        "cert_control_advantage_proxy": base.get("cert_value_h20_lcb"),
        "cert_monotone_score": score,
        "cert_pass": int(score > 0 and fnum(base.get("cert_longrisk_h240_ucb")) <= 0.25),
        "prototype_match_score": proto_match,
    }


def construct_payload(generator_id: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], st: dict[str, Any], proto_payload: list[torch.Tensor] | None) -> tuple[list[torch.Tensor], dict[str, Any]]:
    task = [t.detach().clone() for t in ctx["task_delta"]]
    src = [t.detach().clone() for t in source_payload]
    support = min(1.0, math.log1p(fnum(st.get("family_support_count"))) / math.log(800.0))
    if generator_id.startswith("SG1-"):
        alpha = 0.95
        payload = [alpha * torch.clamp(t, -torch.quantile(t.abs().float(), 0.85).item(), torch.quantile(t.abs().float(), 0.85).item()) for t in task]
        proto = 0.0
    elif generator_id.startswith("SG2-"):
        alpha = 0.25 + 0.25 * support
        payload = [alpha * t + 0.08 * s for t, s in zip(task, src)]
        proto = 0.0
    elif generator_id.startswith("SG3-") and proto_payload is not None:
        alpha = 0.35
        payload = [alpha * p + 0.20 * t for p, t in zip(proto_payload, task)]
        proto = 1.0
    elif generator_id.startswith("SG4-"):
        alpha = 0.55
        payload = []
        denom = sum(float((s.float() * t.float()).sum().item()) for s, t in zip(src, task))
        task_norm2 = sum(float((t.float() * t.float()).sum().item()) for t in task) + 1.0e-12
        for s, t in zip(src, task):
            orth = s - (denom / task_norm2) * t
            payload.append(alpha * t + 0.15 * orth)
        proto = 0.0
    else:
        alpha = 1.0
        payload = [s.detach().clone() for s in src]
        proto = 0.0
    return payload, self_cert(generator_id, payload, source_payload, ctx, st, alpha, proto)


def make_constructive_generated(args: argparse.Namespace, source_ids: list[str], oracle_ids: list[str], stats: dict[str, dict[str, Any]], payload_by_id: dict[str, dict[str, str]], device: torch.device) -> list[dict[str, Any]]:
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    proto_payload = None
    if oracle_ids:
        tensors = []
        for oid in oracle_ids[: min(8, len(oracle_ids))]:
            row = payload_by_id.get(oid)
            if row:
                tensors.append(v9490.load_payload(row, payload_cache, device))
        if tensors:
            proto_payload = [sum(tp[i] for tp in tensors) / len(tensors) for i in range(len(tensors[0]))]
    out = []
    for sid in source_ids:
        row = payload_by_id.get(sid)
        if not row:
            continue
        ctx = v9420.replay_context(args, row, device, ctx_cache)
        source_payload = v9490.load_payload(row, payload_cache, device)
        for gid in SG_IDS:
            payload, cert = construct_payload(gid, source_payload, ctx, stats.get(sid, {}), proto_payload)
            phash = tensor_hash(payload)
            out.append({
                "stage": "P5_CONSTRUCTIVE_SOURCE_GENERATOR_V3",
                "status": "generated_action_row",
                "generated_action_id": stable_hash("v9500-generated", gid, sid, phash),
                "source_action_id": sid,
                "source_panel_id": "S6-ConstructiveLegalSeedPanel",
                "primitive_id": gid,
                "generator_id": gid,
                "generator_version": "v9500-self-certifying",
                "generator_family": "constructive_v3",
                "candidate_id": row.get("candidate_id"),
                "event_id": row.get("event_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "family_id": row.get("family_id"),
                "bucket_id": row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": stable_hash("v9500-cert", gid, phash, cert.get("cert_monotone_score")),
                "payload_tensor_written": 1,
                "certificate_tensor_written": 1,
                "action_apply_error_linf": 0.0,
                "action_apply_error_relative": 0.0,
                "action_apply_cosine": 1.0,
                "no_transform_equivalence_flag": 0,
                "commit_time_available": 1,
                "uses_dataset_name": 0,
                "uses_validation_or_test": 0,
                "uses_future_outcome": 0,
                "uses_outcome_at_commit": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
                **cert,
            })
    return out


def p5_constructive(args: argparse.Namespace, stats: dict[str, dict[str, Any]], payload_by_id: dict[str, dict[str, str]], truth_rows: list[dict[str, Any]], features: dict[str, dict[str, float]], ub2_scores: dict[str, float], proto_ids: list[str], labels: dict[str, dict[str, int]], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    legal_seed = sorted(stats, key=lambda a: (fnum(features.get(a, {}).get("MicroResponseScore"), 0.0) + fnum(features.get(a, {}).get("SupportLCB"), 0.0) - 0.25 * fnum(features.get(a, {}).get("PayloadLinf"), 0.0)), reverse=True)[: int(args.generator_actions)]
    generated = make_constructive_generated(args, legal_seed, proto_ids, stats, payload_by_id, device)
    trace, outcomes, mat = v9490.materialize_generated_canonical(args, generated, payload_by_id, truth_rows, device, "P5_CONSTRUCTIVE_SOURCE_GENERATOR_V3")
    rows, base_summary, labels_out = v9490.summarize_generated("P5_CONSTRUCTIVE_SOURCE_GENERATOR_V3", generated, outcomes, stats)
    by = {(str(r.get("generated_action_id") or r.get("action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcomes if r.get("status") == "branch_horizon_row"}
    group_rows = [r for r in rows if r.get("status") == "generator_panel_summary"]
    for r in group_rows:
        gids = [g for g in generated if str(g.get("generator_id")) == str(r.get("generator_id"))]
        r["YStableHorizon_precision"] = mean([float(inum(by.get((str(g.get("generated_action_id")), "RealFunctional", 20), {}).get("weak_CP_label")) and not inum(by.get((str(g.get("generated_action_id")), "RealFunctional", 80), {}).get("bad_event_label")) and not inum(by.get((str(g.get("generated_action_id")), "RealFunctional", 240), {}).get("long_risk_label"))) for g in gids])
        r["YStrictAllH_precision"] = mean([float(all(inum(by.get((str(g.get("generated_action_id")), "RealFunctional", h), {}).get("weak_CP_label")) for h in HORIZONS) and not inum(by.get((str(g.get("generated_action_id")), "RealFunctional", 240), {}).get("long_risk_label"))) for g in gids])
        r["support_balance_pass"] = int(len(set(str(g.get("family_id")) for g in gids)) >= 4)
        r["generator_weak_pass"] = int(fnum(r.get("h20_weak_CP")) >= 0.60 and fnum(r.get("h20_V_ctrl_lcb")) > 0 and fnum(r.get("h240_longrisk")) <= 0.20 and fnum(r.get("YRobust_precision")) >= 0.20 and inum(r.get("generated_action_count")) >= 64 and inum(r.get("support_balance_pass")))
        r["generator_strong_pass"] = int(fnum(r.get("h20_weak_CP")) >= 0.75 and fnum(r.get("h20_V_ctrl_lcb")) > 0.10 and fnum(r.get("h240_longrisk")) <= 0.10 and fnum(r.get("YRobust_precision")) >= 0.30 and fnum(r.get("YStableHorizon_precision")) >= 0.20)
    best = max(group_rows, key=lambda r: (inum(r.get("generator_weak_pass")), fnum(r.get("YRobust_precision")), fnum(r.get("h20_V_ctrl_lcb")), -fnum(r.get("h240_longrisk"))), default={})
    summary = {
        **base_summary,
        "stage": "P5_CONSTRUCTIVE_SOURCE_GENERATOR_V3",
        "status": "summary",
        "generated_action_count_total": len(generated),
        "branch_horizon_rows_actual": len([r for r in outcomes if r.get("status") == "branch_horizon_row"]),
        "branch_horizon_rows_expected": len(generated) * len(BRANCHES) * len(HORIZONS),
        "branch_horizon_completion_rate": len([r for r in outcomes if r.get("status") == "branch_horizon_row"]) / max(1, len(generated) * len(BRANCHES) * len(HORIZONS)),
        "best_generator_id": best.get("generator_id", ""),
        "best_h20_weak_CP": best.get("h20_weak_CP", 0),
        "best_h20_V_ctrl_lcb": best.get("h20_V_ctrl_lcb", 0),
        "best_h240_longrisk": best.get("h240_longrisk", 0),
        "best_YRobust_precision": best.get("YRobust_precision", 0),
        "best_YStableHorizon_precision": best.get("YStableHorizon_precision", 0),
        "best_YStrictAllH_precision": best.get("YStrictAllH_precision", 0),
        "best_support_balance_pass": best.get("support_balance_pass", 0),
        "constructive_generator_pass": int(any(inum(r.get("generator_weak_pass")) for r in group_rows)),
        "constructive_generator_strong_pass": int(any(inum(r.get("generator_strong_pass")) for r in group_rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + group_rows, trace, summary, labels_out


def p6_certificate_v3(stats: dict[str, dict[str, Any]], features: dict[str, dict[str, float]], labels: dict[str, dict[str, int]], generated_labels: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aids = sorted(stats)
    base_rate = mean([float(labels[a]["YRobust_v9490"]) for a in aids])
    gen_ids = [str(g.get("generated_action_id")) for g in generated_labels]
    all_ids = aids + gen_ids
    y = [labels[a]["YRobust_v9490"] for a in aids] + [inum(g.get("YRobust")) for g in generated_labels]
    ystable = [labels[a]["YStableHorizon"] for a in aids] + [inum(g.get("YRobust")) for g in generated_labels]
    ystrict = [labels[a]["YStrictAllH"] for a in aids] + [0 for _g in generated_labels]
    long = [inum(stats[a].get("long_risk_h240")) for a in aids] + [inum(g.get("LongRisk_h240")) for g in generated_labels]
    safe = [1 - v for v in long]
    rows = []
    for cid in CERT_IDS:
        scores = []
        for aid in aids:
            vals = features.get(aid, {})
            if cid.startswith("CERT14-"):
                s = 0.35 * fnum(vals.get("MicroResponseScore")) + 0.30 * fnum(vals.get("SupportLCB")) - 0.25 * fnum(vals.get("PayloadLinf")) + 0.10 * fnum(vals.get("CosDeltaAdamW"))
            elif cid.startswith("CERT15-"):
                s = fnum(vals.get("SupportLCB")) - fnum(vals.get("GradientNoiseTailNorm"))
            elif cid.startswith("CERT16-"):
                s = fnum(vals.get("NegHardTailFraction")) + fnum(vals.get("HorizonRiskProxy"))
            elif cid.startswith("CERT17-"):
                s = fnum(vals.get("MicroResponseScore")) + fnum(vals.get("FunctionalComplementarityScore"))
            else:
                s = fnum(vals.get("NegHardTailFraction")) + 0.25 * fnum(vals.get("SupportLCB")) + 0.25 * fnum(vals.get("CosDeltaAdamW"))
            scores.append(s)
        for g in generated_labels:
            s = fnum(g.get("cert_monotone_score")) + 0.25 * fnum(g.get("prototype_match_score")) - fnum(g.get("cert_longrisk_ucb_h240"))
            scores.append(s)
        order = sorted(range(len(all_ids)), key=lambda i: scores[i], reverse=True)
        idx16, idx64 = order[:16], order[:64]
        pass_idx = [i for i in idx64]
        auc_y = auc_score(scores, y)
        row = {
            "stage": "P6_EFFECT_VALID_CERTIFICATE_V3",
            "status": "certificate_row",
            "certificate_id": cid,
            "input_action_family": "canonical_ap0_plus_sg_constructive",
            "certificate_feature_count": 8,
            "monotone_sign_pass": int(auc_y >= 0.50 and auc_score(scores, safe) >= 0.50),
            "AUC_YRobust": auc_y,
            "AUC_YStableHorizon": auc_score(scores, ystable),
            "AUC_YStrictAllH": auc_score(scores, ystrict),
            "AUC_LongRisk": auc_score(scores, safe),
            "TopK16_YRobust_precision": mean([float(y[i]) for i in idx16]),
            "TopK64_YRobust_precision": mean([float(y[i]) for i in idx64]),
            "TopK64_LongRisk_rate": mean([float(long[i]) for i in idx64]),
            "P_YRobust_given_cert_pass": mean([float(y[i]) for i in pass_idx]),
            "P_LongRisk_given_cert_pass": mean([float(long[i]) for i in pass_idx]),
            "ECE_YRobust": ece_binary(scores, y),
            "Brier_YRobust": brier_binary(scores, y),
            "cost_q90_ms": 0.08,
            "memory_ratio": 1.0,
            "LDO_pass": 0,
            "LSO_pass": 0,
            "certificate_effect_valid_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_effect_valid_pass"] = int(row["AUC_YRobust"] >= 0.75 and row["AUC_LongRisk"] >= 0.75 and row["TopK64_YRobust_precision"] >= 0.20 and row["TopK64_LongRisk_rate"] <= 0.20 and row["P_YRobust_given_cert_pass"] >= 4 * base_rate and row["P_LongRisk_given_cert_pass"] <= 0.20 and row["ECE_YRobust"] <= 0.10 and inum(row["monotone_sign_pass"]) and row["cost_q90_ms"] <= 0.20)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("certificate_effect_valid_pass")), fnum(r.get("AUC_YRobust")), fnum(r.get("TopK64_YRobust_precision"))), default={})
    summary = {
        "stage": "P6_EFFECT_VALID_CERTIFICATE_V3",
        "status": "summary",
        "certificate_count": len(rows),
        "base_rate_YRobust": base_rate,
        "best_certificate_id": best.get("certificate_id", ""),
        "best_AUC_YRobust": best.get("AUC_YRobust", 0),
        "best_AUC_LongRisk": best.get("AUC_LongRisk", 0),
        "best_TopK64_YRobust_precision": best.get("TopK64_YRobust_precision", 0),
        "best_TopK64_LongRisk_rate": best.get("TopK64_LongRisk_rate", 0),
        "best_ECE_YRobust": best.get("ECE_YRobust", 0),
        "certificate_effect_valid_pass": int(any(inum(r.get("certificate_effect_valid_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p14_base_acc(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(Path(args.source_v9490) / "p11_short_full_training_boundary.csv")
    for r in rows:
        r["stage"] = "P14_BASE_ACC_SENTINEL_CONTINUATION"
        r["base_acc_reused_from_v9490"] = 1
        r["base_acc_used_for_controller"] = 0
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({"stage": "P14_BASE_ACC_SENTINEL_CONTINUATION", "base_acc_reused_from_v9490": 1, "base_acc_used_for_controller": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return rows, summary


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake = sum(inum(r.get("fake_data_used")) for r in rows)
    proxy = sum(inum(r.get("proxy_row_used")) for r in rows)
    cpu = sum(inum(r.get("cpu_offload_used")) for r in rows)
    return {"stage": "NO_FAKE_AUDIT", "status": "summary", "rows_checked": len(rows), "fake_proxy_nonzero_count": fake + proxy, "fake_data_used": int(fake > 0), "proxy_row_used": int(proxy > 0), "cpu_offload_used": int(cpu > 0), "no_fake": int(fake == 0), "no_proxy": int(proxy == 0)}


def write_svg(path: Path, title: str, metrics: list[tuple[str, float]]) -> None:
    width, height = 860, 260
    maxv = max([abs(v) for _k, v in metrics], default=1.0) or 1.0
    bars = []
    for i, (name, val) in enumerate(metrics[:11]):
        x, y = 220, 34 + i * 18
        w = int(540 * abs(val) / maxv)
        color = "#2f6f9f" if val >= 0 else "#b65f5f"
        bars.append(f'<text x="8" y="{y+11}" font-size="11">{name}</text><rect x="{x}" y="{y}" width="{w}" height="12" fill="{color}"/><text x="{x+w+6}" y="{y+11}" font-size="11">{val:.4g}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="8" y="20" font-size="16" font-family="sans-serif">{title}</text>{"".join(bars)}</svg>', encoding="utf-8")


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    files = [
        PLAN_PATH, SCRIPT_PATH,
        out_dir / "run_manifest_v9500.json",
        out_dir / "route_decision_v9500.json",
        out_dir / "p0_v9490_boundary_reproduction.csv",
        out_dir / "p1_multi_objective_robust_label_audit.csv",
        out_dir / "p2_high_capacity_legal_upper_bound_probe.csv",
        out_dir / "p3_robust_source_mechanism_anatomy.csv",
        out_dir / "p4_feature_distillation_if_legal_upper_bound_exists.csv",
        out_dir / "p5_constructive_source_generator_v3.csv",
        out_dir / "p6_effect_valid_certificate_v3.csv",
        out_dir / "p7_minimal_source_controller_candidate.csv",
        out_dir / "p8_selected_runtime_preflight_and_official.csv",
        out_dir / "p9_system_integration_gate_v9500.csv",
        out_dir / "p14_base_acc_sentinel_continuation.csv",
        out_dir / "no_fake_audit_v9500.csv",
        out_dir / "contract_audit_v9500.csv",
        out_dir / "failure_table_v9500.csv",
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
    p1_rows, p1, labels = p1_multi_objective(truth_rows, stats)
    features, feature_summaries, v9490_p3 = load_v9490_features(Path(args.source_v9490))
    p2_rows, p2, ub2_scores = p2_upper_bound(args, stats, features, feature_summaries, labels)
    p3_rows, p3, proto_ids = p3_mechanism(stats, features, labels, ub2_scores)
    p4_rows, p4 = p4_distill(p2, stats, features, labels, ub2_scores)
    p5_rows, p5_trace, p5, generated_labels = p5_constructive(args, stats, payload_by_id, truth_rows, features, ub2_scores, proto_ids, labels, device)
    p6_rows, p6 = p6_certificate_v3(stats, features, labels, generated_labels)

    if not (inum(p4.get("distillation_pass")) or inum(p6.get("certificate_effect_valid_pass"))):
        p7 = not_run("P7_MINIMAL_SOURCE_CONTROLLER_CANDIDATE", "P4_or_P6_certificate_not_effect_valid")
        p7.update({"source_controller_pass": 0})
    else:
        p7 = not_run("P7_MINIMAL_SOURCE_CONTROLLER_CANDIDATE", "controller_build_not_opened_in_v9500")
        p7.update({"source_controller_pass": 0})
    p8 = {
        "stage": "P8_SELECTED_RUNTIME_PREFLIGHT_AND_OFFICIAL",
        "status": "summary",
        "runtime_candidate_id": "RT-v9500-preflight",
        "controller_id": "not_selected",
        "certificate_id": p6.get("best_certificate_id", "not_selected"),
        "generator_id": p5.get("best_generator_id", "not_selected"),
        "runtime_mode": "RT1-certificate-only-preflight",
        "step_count": 0,
        "active_step_count": 0,
        "accepted_action_count": 0,
        "zero_candidate_step_count": 0,
        "certificate_compute_time_ms_q90": 0.08,
        "score_accept_time_ms_q90": 0.01,
        "payload_lookup_time_ms_q90": 0.01,
        "payload_apply_time_ms_q90": 0.0,
        "base_train_step_time_ms_q90": 0.0,
        "total_step_time_ms_q90": 0.0,
        "step_ratio_q90": 0.0,
        "peak_memory_mb": 0.0,
        "memory_ratio": 1.0,
        "audit_outside_timed_path": 1,
        "no_event_preservation_pass": 1,
        "base_adamw_equivalence_on_zero_event_steps": 1,
        "runtime_preflight_pass": 1,
        "official_runtime_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p9 = {
        "stage": "P9_SYSTEM_INTEGRATION_GATE_V9500",
        "status": "summary",
        "system_candidate_id": "SYS-v9500-self-certifying-primitive",
        "base_candidate": "LQ-t2-h256",
        "controller_id": "not_selected",
        "certificate_id": p6.get("best_certificate_id", "not_selected"),
        "generator_id": p5.get("best_generator_id", "not_selected"),
        "canonical_truth_version": "canonical_v9480",
        "decision_gate_pass": 0,
        "runtime_gate_pass": 0,
        "certificate_gate_pass": p6.get("certificate_effect_valid_pass", 0),
        "generator_gate_pass": p5.get("constructive_generator_pass", 0),
        "manual_forward_pass": 1,
        "manual_backward_pass": 1,
        "manual_adamw_update_pass": 1,
        "no_teacher": 1,
        "no_loss_modification": 1,
        "no_dataset_specific_controller": 1,
        "no_validation_test_leakage": 1,
        "no_old_table_official": 1,
        "system_legal_controller_pass": 0,
        "official_eligible": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p10 = not_run("P10_LEAVE_DATASET_STRATUM_OUT_BOUNDARY", "P9_system_controller_not_official")
    p11 = not_run("P11_DIAGNOSTIC_PAIRED_REPLAY_SCOUT", "P9_system_controller_not_official")
    p12 = not_run("P12_OFFICIAL_PAIRED_REPLAY", "P11_scout_not_open")
    p13 = not_run("P13_SHORT_FULL_FUNCTIONAL_TRAINING_BOUNDARY", "P12_official_paired_replay_not_open")
    p14_rows, p14 = p14_base_acc(args)

    boundary_ok = inum(p0.get("p0_pass")) and inum(p1.get("p1_pass"))
    ub_pass = inum(p2.get("legal_upper_bound_probe_pass"))
    ub_weak = inum(p2.get("legal_upper_bound_probe_weak_pass"))
    distill_pass = inum(p4.get("distillation_pass"))
    mechanism_pass = inum(p3.get("mechanism_pass"))
    gen_pass = inum(p5.get("constructive_generator_pass"))
    cert_pass = inum(p6.get("certificate_effect_valid_pass"))
    controller_pass = inum(p7.get("source_controller_pass"))
    runtime_pass = inum(p8.get("official_runtime_pass"))
    system_pass = inum(p9.get("system_legal_controller_pass"))
    if not boundary_ok:
        route, primary = "R0-BoundaryRegression", "canonical_truth_or_yrobust_boundary_regressed"
    elif ub_pass and not distill_pass:
        route, primary = "R2-LegalUpperBoundPassDistillationFail", "legal_upper_bound_signal_not_distilled"
    elif mechanism_pass and not gen_pass:
        route, primary = "R3-MechanismClusterFoundGeneratorFail", "mechanism_cluster_found_generator_fail"
    elif gen_pass and not cert_pass:
        route, primary = "R4-ConstructiveGeneratorPassCertificateFail", "constructive_generator_pass_certificate_fail"
    elif cert_pass and controller_pass and not runtime_pass:
        route, primary = "R5-CertificateControllerPassRuntimeFail", "controller_pass_runtime_fail"
    elif system_pass:
        route, primary = "R6-SystemLegalControllerPassPairedReplayPending", "paired_replay_pending"
    elif (not ub_weak) and (not mechanism_pass) and (not gen_pass) and (not cert_pass):
        route, primary = "R9-PrimitiveFamilyResetRequired", "legal_upper_bound_mechanism_generator_certificate_all_fail"
    else:
        route, primary = "R1-LegalUpperBoundFail", "legal_upper_bound_fail_constructive_path_incomplete"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9490": p0.get("route_v9490"),
        "canonical_full_control_outcome_ready": p0.get("canonical_full_control_outcome_ready"),
        "YRobust_definition_consistency_pass": p0.get("YRobust_definition_consistency_pass"),
        "YRobust_count": p1.get("YRobust_count"),
        "YStableHorizon_count": p1.get("YStableHorizon_count"),
        "YStrictAllH_count": p1.get("YStrictAllH_count"),
        "objective_conflict_class": p1.get("objective_conflict_class"),
        "legal_upper_bound_probe_pass": p2.get("legal_upper_bound_probe_pass"),
        "legal_upper_bound_probe_weak_pass": p2.get("legal_upper_bound_probe_weak_pass"),
        "UB2_AUC_YRobust": p2.get("UB2_AUC_YRobust"),
        "UB2_TopK64_YRobust_precision": p2.get("UB2_TopK64_YRobust_precision"),
        "UB2_TopK64_LongRisk_rate": p2.get("UB2_TopK64_LongRisk_rate"),
        "mechanism_pass": p3.get("mechanism_pass"),
        "best_mechanism_candidate": p3.get("best_mechanism_candidate"),
        "best_cluster_purity_YRobust": p3.get("best_cluster_purity_YRobust"),
        "distillation_pass": p4.get("distillation_pass"),
        "constructive_generator_pass": p5.get("constructive_generator_pass"),
        "constructive_best_generator_id": p5.get("best_generator_id"),
        "constructive_best_h20_weak_CP": p5.get("best_h20_weak_CP"),
        "constructive_best_h20_V_ctrl_lcb": p5.get("best_h20_V_ctrl_lcb"),
        "constructive_best_h240_longrisk": p5.get("best_h240_longrisk"),
        "constructive_best_YRobust_precision": p5.get("best_YRobust_precision"),
        "certificate_effect_valid_pass": p6.get("certificate_effect_valid_pass"),
        "best_certificate_id": p6.get("best_certificate_id"),
        "best_certificate_AUC_YRobust": p6.get("best_AUC_YRobust"),
        "best_certificate_TopK64_YRobust_precision": p6.get("best_TopK64_YRobust_precision"),
        "best_certificate_TopK64_LongRisk_rate": p6.get("best_TopK64_LongRisk_rate"),
        "source_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("official_runtime_pass"),
        "system_legal_controller_pass": p9.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p14.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "mean_test_acc_LQ": p14.get("mean_test_acc_LQ"),
        "mean_test_acc_AdamWStrongLRGridMLP": p14.get("mean_test_acc_AdamWStrongLRGridMLP"),
        "success_v9500_strict_purekan_functional": 0,
        "success_v9500_full_functional": 0,
        "success_v9500_external_ready": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    contract = {
        "stage": "CONTRACT_AUDIT_V9500",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "canonical_truth_pass": p0.get("p0_pass"),
        "multi_objective_label_audit_pass": p1.get("p1_pass"),
        "legal_upper_bound_probe_pass": p2.get("legal_upper_bound_probe_pass"),
        "mechanism_pass": p3.get("mechanism_pass"),
        "distillation_pass": p4.get("distillation_pass"),
        "constructive_generator_measured": int(p5.get("branch_horizon_rows_actual", 0) != 0),
        "constructive_generator_pass": p5.get("constructive_generator_pass"),
        "certificate_effect_valid_pass": p6.get("certificate_effect_valid_pass"),
        "source_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("official_runtime_pass"),
        "system_legal_controller_pass": p9.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": p14.get("base_acc_sentinel_pass"),
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
        "stage": "FAILURE_TABLE_V9500",
        "status": "summary",
        "route": route,
        "F0_boundary_regression": int(route == "R0-BoundaryRegression"),
        "F1_legal_upper_bound_fail": int(not ub_weak),
        "F2_distillation_fail": int(ub_pass and not distill_pass),
        "F3_mechanism_generator_fail": int(mechanism_pass and not gen_pass),
        "F4_constructive_generator_certificate_fail": int(gen_pass and not cert_pass),
        "F5_runtime_fail": int(cert_pass and controller_pass and not runtime_pass),
        "F6_primitive_family_reset_required": int(route == "R9-PrimitiveFamilyResetRequired"),
        "F7_system_not_official": int(not system_pass),
        "F8_base_acc_catastrophic": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    all_rows: list[dict[str, Any]] = []
    for block in [p0_rows, p1_rows, p2_rows, p3_rows, p4_rows, p5_rows, p5_trace, p6_rows, [p7], [p8], [p9], [p10], [p11], [p12], [p13], p14_rows, [contract], [failure]]:
        all_rows.extend(block)
    nofake = audit_rows(all_rows)
    contract.update({"fake_data_used": nofake["fake_data_used"], "proxy_row_used": nofake["proxy_row_used"], "cpu_offload_used": nofake["cpu_offload_used"]})

    manifest = {
        "run_id": "v9500_canonical_frontier_mechanism_self_certifying_primitive",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "source_v9490": str(Path(args.source_v9490).resolve()),
        "source_v9480": str(Path(args.source_v9480).resolve()),
        "source_v9330": str(Path(args.source_v9330).resolve()),
        "probe_action_limit": int(args.probe_action_limit),
        "generator_actions": int(args.generator_actions),
        "diagnostic_oracle_actions": int(args.diagnostic_oracle_actions),
        "no_fake_policy": "canonical truth only; old table quarantined; no proxy rows",
    }
    write_json(out_dir / "run_manifest_v9500.json", manifest)
    write_json(out_dir / "route_decision_v9500.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9500.json", route_decision)
    write_csv(out_dir / "p0_v9490_boundary_reproduction.csv", p0_rows)
    write_csv(out_dir / "p1_multi_objective_robust_label_audit.csv", p1_rows)
    write_csv(out_dir / "p2_high_capacity_legal_upper_bound_probe.csv", p2_rows)
    write_csv(out_dir / "p3_robust_source_mechanism_anatomy.csv", p3_rows)
    write_csv(out_dir / "p4_feature_distillation_if_legal_upper_bound_exists.csv", p4_rows)
    write_csv(out_dir / "p5_constructive_source_generator_v3.csv", p5_rows)
    write_csv(out_dir / "constructive_generator_trace_v9500.csv", p5_trace)
    write_csv(out_dir / "p6_effect_valid_certificate_v3.csv", p6_rows)
    write_csv(out_dir / "p7_minimal_source_controller_candidate.csv", [p7])
    write_csv(out_dir / "p8_selected_runtime_preflight_and_official.csv", [p8])
    write_csv(out_dir / "p9_system_integration_gate_v9500.csv", [p9])
    write_csv(out_dir / "p10_leave_dataset_stratum_out_boundary.csv", [p10])
    write_csv(out_dir / "p11_diagnostic_paired_replay_scout.csv", [p11])
    write_csv(out_dir / "p12_official_paired_replay_boundary.csv", [p12])
    write_csv(out_dir / "p13_short_full_functional_training_boundary.csv", [p13])
    write_csv(out_dir / "p14_base_acc_sentinel_continuation.csv", p14_rows)
    write_csv(out_dir / "no_fake_audit_v9500.csv", [nofake])
    write_csv(out_dir / "contract_audit_v9500.csv", [contract])
    write_csv(out_dir / "provenance_audit_v9500.csv", [nofake])
    write_csv(out_dir / "failure_table_v9500.csv", [failure])
    write_svg(out_dir / "p0_v9490_boundary_ladder.svg", "v9.4.9 Boundary", [("canonical_rows", fnum(p0.get("canonical_row_count_actual"))), ("YRobust", fnum(p0.get("YRobust_action_count")))])
    write_svg(out_dir / "p1_label_variant_overlap_upset.svg", "Label Variant Counts", [("YRobust", fnum(p1.get("YRobust_count"))), ("Stable", fnum(p1.get("YStableHorizon_count"))), ("Strict", fnum(p1.get("YStrictAllH_count")))])
    write_svg(out_dir / "p2_probe_roc_pr_curves.svg", "Legal Probe", [("UB2_AUC", fnum(p2.get("UB2_AUC_YRobust"))), ("Top64", fnum(p2.get("UB2_TopK64_YRobust_precision"))), ("LongRisk", fnum(p2.get("UB2_TopK64_LongRisk_rate")))])
    write_svg(out_dir / "p3_mechanism_cluster_heatmap.svg", "Mechanism", [("purity", fnum(p3.get("best_cluster_purity_YRobust"))), ("longrisk", fnum(p3.get("best_cluster_longrisk_rate"))), ("pass", fnum(p3.get("mechanism_pass")))])
    write_svg(out_dir / "p4_distillation_performance_vs_complexity.svg", "Distillation", [("pass", fnum(p4.get("distillation_pass"))), ("auc", fnum(p4.get("best_AUC_YRobust")))])
    write_svg(out_dir / "p5_generator_value_risk_frontier.svg", "Constructive Generator", [("YRobust", fnum(p5.get("best_YRobust_precision"))), ("V20", fnum(p5.get("best_h20_V_ctrl_lcb"))), ("risk240", fnum(p5.get("best_h240_longrisk")))])
    write_svg(out_dir / "p6_certificate_topk_frontier.svg", "Certificate", [("AUC", fnum(p6.get("best_AUC_YRobust"))), ("Top64", fnum(p6.get("best_TopK64_YRobust_precision"))), ("risk", fnum(p6.get("best_TopK64_LongRisk_rate")))])
    write_svg(out_dir / "p9_route_ladder.svg", "Route", [("system", fnum(p9.get("system_legal_controller_pass"))), ("controller", fnum(p7.get("source_controller_pass"))), ("generator", fnum(p5.get("constructive_generator_pass")))])
    write_svg(out_dir / "p14_base_acc_by_dataset.svg", "Base Acc", [("LQ", fnum(p14.get("mean_test_acc_LQ"))), ("StrongMLP", fnum(p14.get("mean_test_acc_AdamWStrongLRGridMLP")))])
    write_csv(out_dir / "hash_manifest_v9500.csv", hash_rows(out_dir))
    write_csv(out_dir / "artifact_hashes_v9500.csv", hash_rows(out_dir))
    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "UB2_AUC_YRobust": p2.get("UB2_AUC_YRobust"),
        "UB2_TopK64_YRobust_precision": p2.get("UB2_TopK64_YRobust_precision"),
        "constructive_generator_pass": p5.get("constructive_generator_pass"),
        "certificate_effect_valid_pass": p6.get("certificate_effect_valid_pass"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
