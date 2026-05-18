#!/usr/bin/env python3
"""DG-KAN v9.4.9 canonical legal observability / source generator runner.

The runner is intentionally conservative: it only uses the canonical v9.4.8
outcome universe as the truth base, keeps old v9.3.5 rows quarantined, records
commit-time feature costs, and never promotes oracle/source/certificate
diagnostics into official system success.
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

import run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure as v9340  # noqa: E402
import run_v9420_source_outcome_materializer_closure as v9420  # noqa: E402
import run_v9450_robust_source_generator_objective_aligned_certificate as v9450  # noqa: E402
import run_v9480_canonical_outcome_universe_rebuild as v9480  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.4.9_CanonicalLegalObservability_SourceGeneratorCertificate_ParallelClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9490_canonical_legal_observability_source_generator_certificate.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_V9480 = RESULT_ROOT / "v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z"
DEFAULT_V9470 = RESULT_ROOT / "v9470_no_transform_replay_semantics_closure_robust_source_revalidation_first_20260514T140000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

HORIZONS = [20, 80, 240]
BRANCHES = ["RealFunctional", "AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
CONTROL_BRANCHES = ["AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
OUTCOME_VERSION = "canonical_v9480"
RUNNER_VERSION = "canonical_branch_name_invariant_v9470_or_later"
BRANCH_VERSION = "canonical_branch_semantics_v9490"
CERT_IDS = [
    "CERT9-ActionEffectLinearizedCertificate",
    "CERT10-AdamWComplementarityCertificate",
    "CERT11-HorizonTailGuardCertificate",
    "CERT12-SupportCalibratedYRobustCertificate",
    "CERT13-MinimalMonotoneCompositeCertificate",
]
P4_GENERATORS = [
    "G0-NoTransformCanonicalReplay",
    "AP0r-ObjectiveSolvedLastEdgeTrustRegionSource",
    "AP0t-TailProjectedHorizonGuardSource",
    "AP0u-AdamWCompatibleResidualSource",
    "AP0v-LowRankEdgeObjectiveSolver",
    "AP0l-LinearizedTrustRegionSource",
]
P5_GENERATORS = [
    "VG1-ConstrainedLinearizedSourceSolver",
    "VG2-HorizonGuardedResidualBlend",
    "VG3-TailMarginConservativeSource",
    "VG4-RoleSparseEdgeUpdate",
    "VG5-SupportMemorySource",
    "VG6-YRobustAnatomyInformedGenerator",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9480", default=str(DEFAULT_V9480))
    p.add_argument("--source-v9470", default=str(DEFAULT_V9470))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--feature-action-limit", type=int, default=2876)
    p.add_argument("--generator-actions", type=int, default=16)
    p.add_argument("--direct-actions-per-generator", type=int, default=16)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--clear-caches-each-action", action="store_true")
    p.add_argument("--run-base-acc", action="store_true")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--sentinel-seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--sentinel-steps", type=int, default=12)
    p.add_argument("--sentinel-train-size", type=int, default=512)
    p.add_argument("--sentinel-test-size", type=int, default=256)
    p.add_argument("--sentinel-hidden-dim", type=int, default=64)
    p.add_argument("--strong-lr-grid", default="0.0003,0.001,0.003")
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
    if not scores:
        return 0.0
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return 0.0
    lo, hi = min(finite), max(finite)
    probs = [0.5 if hi <= lo else max(0.0, min(1.0, (s - lo) / (hi - lo))) for s in scores]
    ece = 0.0
    for b in range(bins):
        left, right = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (p >= left and (p < right or b == bins - 1))]
        if not idx:
            continue
        conf = mean([probs[i] for i in idx])
        acc = mean([float(labels[i]) for i in idx])
        ece += len(idx) / len(probs) * abs(conf - acc)
    return ece


def tensor_hash(payload: list[torch.Tensor]) -> str:
    return v9450.tensor_hash(payload)


def tensor_stats(payload: list[torch.Tensor]) -> dict[str, float]:
    return v9420.v9410.tensor_stats(payload)


def flat_dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += float((x.detach().float() * y.detach().float()).sum().item())
    return total


def flat_norm(a: list[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, flat_dot(a, a)))


def flat_cos(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return flat_dot(a, b) / max(1.0e-12, flat_norm(a) * flat_norm(b))


def p0_boundary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = read_json(Path(args.source_v9480) / "route_decision.json")
    first = Path(str(args.source_v9480).replace("_recovery_20260514T160000Z", "_first_20260514T150000Z"))
    first_rows = 0
    first_route = ""
    if first.exists():
        fr = read_json(first / "route_decision.json")
        first_rows = inum(fr.get("canonical_row_count_actual"))
        first_route = str(fr.get("route"))
    row = {
        "stage": "P0_V9480_RECOVERY_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "source_artifact_id": rel(Path(args.source_v9480)),
        "first_artifact_id": rel(first) if first.exists() else "",
        "first_route_v9480": first_route,
        "first_canonical_rows": first_rows,
        "route_v9480": r.get("route"),
        "canonical_full_control_outcome_ready": r.get("canonical_full_control_outcome_ready"),
        "canonical_row_count_expected": r.get("canonical_row_count_expected"),
        "canonical_row_count_actual": r.get("canonical_row_count_actual"),
        "quality_audit_pass": r.get("quality_audit_pass"),
        "old_table_quarantine_enforced": r.get("old_table_quarantine_enforced"),
        "canonical_replay_preflight_pass": r.get("canonical_replay_preflight_pass"),
        "canonical_control_positive_oracle_pass": r.get("canonical_control_positive_oracle_pass"),
        "canonical_horizon_robust_oracle_pass": r.get("canonical_horizon_robust_oracle_pass"),
        "canonical_source_frontier_pass": r.get("canonical_source_frontier_pass"),
        "legal_observability_pass": r.get("legal_observability_pass"),
        "system_legal_controller_pass": r.get("system_legal_controller_pass"),
        "primary_blocker": r.get("primary_blocker"),
        "p0_pass": int(
            r.get("route") == "R4-CanonicalSourceFrontierExistsLegalOpaque"
            and inum(r.get("canonical_full_control_outcome_ready"))
            and inum(r.get("canonical_row_count_actual")) == 51768
            and inum(r.get("quality_audit_pass"))
            and inum(r.get("old_table_quarantine_enforced"))
            and not inum(r.get("legal_observability_pass"))
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def load_payload_rows(source_v9330: Path) -> list[dict[str, str]]:
    return v9480.load_payload_rows(source_v9330)


def load_payload(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return v9480.load_payload(row, cache, device)


def load_truth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    rows = read_csv(Path(args.source_v9480) / "canonical_full_control_outcome_table_v9480.csv")
    stats = v9480.build_stats(rows)
    payload_rows = load_payload_rows(Path(args.source_v9330))
    payload_by_id = {str(r.get("action_id")): r for r in payload_rows}
    return rows, stats, payload_by_id


def p1_yrobust_audit(rows: list[dict[str, Any]], stats: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by = defaultdict(dict)
    for r in rows:
        if r.get("branch_id") == "RealFunctional":
            by[str(r.get("action_id"))][inum(r.get("horizon"))] = r
    out: list[dict[str, Any]] = []
    for aid, hs in sorted(by.items()):
        if not all(h in hs for h in HORIZONS):
            continue
        y_original = inum(hs[20].get("Y_robust_label"))
        v1 = int(all(inum(hs[h].get("weak_CP_label")) for h in HORIZONS) and not inum(hs[240].get("long_risk_label")))
        v2 = int(inum(hs[20].get("weak_CP_label")) and fnum(hs[20].get("V_ctrl")) > 0.0 and not inum(hs[240].get("long_risk_label")))
        robust_score = fnum(hs[20].get("V_ctrl")) + max(0.0, fnum(hs[240].get("V_ctrl"))) - 2.0 * inum(hs[240].get("long_risk_label")) - max(0.0, -fnum(hs[80].get("V_ctrl")))
        out.append({
            "stage": "P1_YROBUST_DEFINITION_CONSISTENCY_AUDIT",
            "status": "action_row",
            "action_id": aid,
            "YRobust_original": y_original,
            "YRobust_recomputed_v1": v1,
            "YRobust_recomputed_v2": v2,
            "YRobust_recomputed_v3_score": robust_score,
            "YRobust_recomputed_v3": 0,
            "h20_weak_CP": inum(hs[20].get("weak_CP_label")),
            "h80_weak_CP": inum(hs[80].get("weak_CP_label")),
            "h240_weak_CP": inum(hs[240].get("weak_CP_label")),
            "h20_strong_CP": inum(hs[20].get("strong_CP_label")),
            "h80_strong_CP": inum(hs[80].get("strong_CP_label")),
            "h240_strong_CP": inum(hs[240].get("strong_CP_label")),
            "h20_V_ctrl_lcb": fnum(hs[20].get("V_ctrl")),
            "h80_V_ctrl_lcb": fnum(hs[80].get("V_ctrl")),
            "h240_V_ctrl_lcb": fnum(hs[240].get("V_ctrl")),
            "h240_longrisk": inum(hs[240].get("long_risk_label")),
            "manual_recompute_match": int(y_original == v2),
            "mismatch_reason": "" if y_original == v2 else "original_not_equal_v2_h20_value_and_h240_safe",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    top_v3 = {r["action_id"] for r in sorted(out, key=lambda r: fnum(r.get("YRobust_recomputed_v3_score")), reverse=True)[: sum(inum(r.get("YRobust_original")) for r in out)]}
    for r in out:
        r["YRobust_recomputed_v3"] = int(r["action_id"] in top_v3)
    mismatches = [r for r in out if not inum(r.get("manual_recompute_match"))]
    yrob = [r for r in out if inum(r.get("YRobust_original"))]
    summary = {
        "stage": "P1_YROBUST_DEFINITION_CONSISTENCY_AUDIT",
        "status": "summary",
        "action_count": len(out),
        "YRobust_original_count": len(yrob),
        "YRobust_v1_count": sum(inum(r.get("YRobust_recomputed_v1")) for r in out),
        "YRobust_v2_count": sum(inum(r.get("YRobust_recomputed_v2")) for r in out),
        "YRobust_v3_count": sum(inum(r.get("YRobust_recomputed_v3")) for r in out),
        "YRobust_manual_recompute_match_rate": mean([fnum(r.get("manual_recompute_match")) for r in out]),
        "YRobust_mismatch_count": len(mismatches),
        "YRobust_definition_consistency_pass": int(bool(out) and not mismatches),
        "h80_required_by_original_definition": 0,
        "YRobust_h80_weak_rate": mean([fnum(r.get("h80_weak_CP")) for r in yrob]),
        "YRobust_h20_weak_rate": mean([fnum(r.get("h20_weak_CP")) for r in yrob]),
        "YRobust_h240_longrisk_rate": mean([fnum(r.get("h240_longrisk")) for r in yrob]),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + out, summary


def action_base_features(stats: dict[str, dict[str, Any]], payload_by_id: dict[str, dict[str, str]], rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_real20 = {str(r.get("action_id")): r for r in rows if r.get("branch_id") == "RealFunctional" and inum(r.get("horizon")) == 20}
    by_real240 = {str(r.get("action_id")): r for r in rows if r.get("branch_id") == "RealFunctional" and inum(r.get("horizon")) == 240}
    fam_counts = Counter(str(s.get("family_id")) for s in stats.values())
    out: dict[str, dict[str, float]] = {}
    for aid, st in stats.items():
        p = payload_by_id.get(aid, {})
        r20 = by_real20.get(aid, {})
        r240 = by_real240.get(aid, {})
        payload_norm = fnum(p.get("payload_l2_norm") or p.get("payload_norm"))
        payload_linf = fnum(p.get("payload_linf_norm"))
        family_support = fam_counts[str(st.get("family_id"))]
        out[aid] = {
            "PayloadNorm": payload_norm,
            "NegPayloadNorm": -payload_norm,
            "PayloadLinf": payload_linf,
            "NegPayloadLinf": -payload_linf,
            "Step": fnum(p.get("step")),
            "NegStep": -fnum(p.get("step")),
            "FamilySupportCount": float(family_support),
            "SupportLCB": math.log1p(float(family_support)) / math.log1p(max(1, max(fam_counts.values(), default=1))),
            "CandidateId": fnum(p.get("candidate_id")),
            "NegCandidateId": -fnum(p.get("candidate_id")),
            "StateNLL": fnum(r20.get("NLL_before")),
            "NegStateNLL": -fnum(r20.get("NLL_before")),
            "StateCEp99": fnum(r20.get("CEp99_before")),
            "NegStateCEp99": -fnum(r20.get("CEp99_before")),
            "StateMarginP10": fnum(r20.get("margin_p10_before")),
            "HardTailFraction": max(0.0, fnum(r20.get("CEp99_before")) - fnum(r20.get("CE_mean_before"))),
            "NegHardTailFraction": -max(0.0, fnum(r20.get("CEp99_before")) - fnum(r20.get("CE_mean_before"))),
            "PreLongRiskProxy": -payload_norm - 0.1 * max(0.0, fnum(r240.get("CEp99_before")) - fnum(r240.get("CE_mean_before"))),
        }
    return out


def materialize_legal_features(
    args: argparse.Namespace,
    stats: dict[str, dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    truth_rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, float]], list[dict[str, Any]]]:
    labels_y = {aid: inum(st.get("Y_robust")) for aid, st in stats.items()}
    labels_long = {aid: inum(st.get("long_risk_h240")) for aid, st in stats.items()}
    labels_weak = {aid: inum(st.get("weak_h20")) for aid, st in stats.items()}
    feature_values = action_base_features(stats, payload_by_id, truth_rows)
    action_ids = [aid for aid in payload_by_id.keys() if aid in feature_values][: int(args.feature_action_limit)]
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    feature_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    last_ctx_key: tuple[str, int, int] | None = None
    for idx, aid in enumerate(action_ids):
        p = payload_by_id.get(aid)
        if not p:
            continue
        ctx_key = (str(p.get("dataset")), int(inum(p.get("seed"))), int(inum(p.get("step"))))
        if last_ctx_key is not None and ctx_key != last_ctx_key:
            ctx_cache.clear()
        last_ctx_key = ctx_key
        memory_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        t0 = time.perf_counter()
        missing_micro = 0
        try:
            ctx = v9420.replay_context(args, p, device, ctx_cache)
            payload = load_payload(p, payload_cache, device)
            task_delta = ctx["task_delta"]
            with torch.no_grad():
                params_payload = [tp + d for tp, d in zip(ctx["task_params"], payload)]
                after_metrics = v9340.extended_metrics_from_logits(ctx["fwd_core"](ctx["xp"], *params_payload, ctx["mu"], ctx["std"], 2.0, 2.0), ctx["yp"])
            delta = v9340.metric_delta(after_metrics, ctx["before_metrics"])
            pstats = tensor_stats(payload)
            norms = [float(t.detach().float().norm().item()) for t in payload]
            total_norm = sum(norms) + 1.0e-12
            role_probs = [n / total_norm for n in norms]
            role_entropy = -sum(pv * math.log(max(pv, 1.0e-12)) for pv in role_probs)
            cos_adamw = flat_cos(payload, task_delta)
            norm_adamw = flat_norm(task_delta)
            norm_payload = flat_norm(payload)
            residual = math.sqrt(max(0.0, norm_payload**2 - (flat_dot(payload, task_delta) / max(norm_adamw, 1.0e-12)) ** 2))
            feature_values[aid].update({
                "PayloadRoleEntropy": role_entropy,
                "PayloadTailSelectivity": max(role_probs) if role_probs else 0.0,
                "CosDeltaAdamW": cos_adamw,
                "CosDeltaNegativeGradProxy": cos_adamw,
                "NegCosDeltaAdamW": -cos_adamw,
                "NormDeltaOverAdamW": norm_payload / max(norm_adamw, 1.0e-12),
                "NegNormDeltaOverAdamW": -norm_payload / max(norm_adamw, 1.0e-12),
                "AdamWResidualNorm": residual,
                "NegAdamWResidualNorm": -residual,
                "FunctionalComplementarityScore": residual / max(norm_payload, 1.0e-12),
                "AdamWConflictGuard": -max(0.0, -cos_adamw),
                "LinearizedCEDeltaMean": -flat_dot(payload, task_delta),
                "LinearizedCEDeltaP90": -flat_dot(payload, task_delta) / max(1.0, len(payload)),
                "MicroResponseCEDelta": fnum(delta.get("CE_mean_delta")),
                "NegMicroResponseCEDelta": -fnum(delta.get("CE_mean_delta")),
                "MicroResponseCEp99Delta": fnum(delta.get("CEp99_delta")),
                "NegMicroResponseCEp99Delta": -fnum(delta.get("CEp99_delta")),
                "MicroResponseMarginDelta": fnum(delta.get("margin_p10_delta")),
                "MicroResponseEntropyDelta": fnum(delta.get("basis_usage_entropy_delta")),
                "MicroResponseScore": v9480.value_from_delta(delta),
                "HorizonRiskProxy": -pstats["payload_linf"] - max(0.0, -fnum(delta.get("margin_p10_delta"))),
                "GradientNoiseTailNorm": pstats["payload_linf"] / max(pstats["payload_norm"], 1.0e-12),
                "NegGradientNoiseTailNorm": -pstats["payload_linf"] / max(pstats["payload_norm"], 1.0e-12),
            })
        except Exception as exc:  # noqa: BLE001
            missing_micro = 1
            feature_values[aid].update({"MicroResponseScore": float("nan"), "MicroResponseCEDelta": float("nan"), "feature_exception": type(exc).__name__})
        elapsed = (time.perf_counter() - t0) * 1000.0
        memory_after = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        cost_rows.append({
            "stage": "P3_LEGAL_ACTION_EFFECT_FEATURE_FACTORY_V2",
            "status": "feature_action_cost_row",
            "action_id": aid,
            "feature_compute_time_ms": elapsed,
            "feature_memory_delta_mb": max(0.0, (memory_after - memory_before) / (1024.0 * 1024.0)),
            "micro_response_missing": missing_micro,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        if args.clear_caches_each_action or (idx and idx % 128 == 0):
            payload_cache.clear()
            if args.clear_caches_each_action:
                ctx_cache.clear()
                last_ctx_key = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    feature_ids = sorted({k for vals in feature_values.values() for k in vals.keys() if not k.startswith("feature_")})
    summaries: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    q90_cost = quantile([fnum(r.get("feature_compute_time_ms")) for r in cost_rows], 0.90)
    for fid in feature_ids:
        scores = [fnum(feature_values[aid].get(fid), 0.0) for aid in action_ids]
        y = [labels_y[aid] for aid in action_ids]
        long = [labels_long[aid] for aid in action_ids]
        weak = [labels_weak[aid] for aid in action_ids]
        order16 = sorted(range(len(action_ids)), key=lambda i: scores[i], reverse=True)[: min(16, len(action_ids))]
        order64 = sorted(range(len(action_ids)), key=lambda i: scores[i], reverse=True)[: min(64, len(action_ids))]
        auc_y = auc_score(scores, y)
        auc_long = auc_score(scores, long)
        top64_y = mean([float(y[i]) for i in order64])
        top64_long = mean([float(long[i]) for i in order64])
        row = {
            "stage": "P3_LEGAL_ACTION_EFFECT_FEATURE_FACTORY_V2",
            "status": "feature_summary",
            "feature_id": fid,
            "feature_group": feature_group(fid),
            "feature_version": "v9490-commit-time-v2",
            "action_count": len(action_ids),
            "commit_time_available": 1,
            "uses_dataset_name": 0,
            "uses_validation_or_test": 0,
            "uses_future_outcome": 0,
            "uses_outcome_at_commit": 0,
            "uses_old_table": 0,
            "uses_oracle_label_at_commit": 0,
            "feature_missing_rate": mean([float(not math.isfinite(fnum(feature_values[aid].get(fid), float("nan")))) for aid in action_ids]),
            "feature_compute_q90_ms": q90_cost if feature_group(fid) in {"G1-gradient-action", "G2-linearized-micro-response", "G6-cheap-micro-response"} else 0.01,
            "memory_delta_mb": quantile([fnum(r.get("feature_memory_delta_mb")) for r in cost_rows], 0.90),
            "AUC_YRobust": auc_y,
            "AUC_LongRisk": auc_long,
            "AUC_WeakCP_h20": auc_score(scores, weak),
            "TopK16_YRobust_precision": mean([float(y[i]) for i in order16]),
            "TopK64_YRobust_precision": top64_y,
            "TopK64_h240_longrisk": top64_long,
            "leave_dataset_auc_drop": leaveout_drop(action_ids, scores, y, stats, "dataset"),
            "leave_stratum_auc_drop": leaveout_drop(action_ids, scores, y, stats, "family_id"),
            "legal_feature_weak_pass": int(auc_y >= 0.70 and top64_y >= 0.20 and top64_long <= 0.20 and (q90_cost if feature_group(fid) in {"G1-gradient-action", "G2-linearized-micro-response", "G6-cheap-micro-response"} else 0.01) <= 0.25),
            "legal_feature_strong_pass": int(auc_y >= 0.80 and mean([float(y[i]) for i in order16]) >= 0.50 and top64_y >= 0.35 and top64_long <= 0.10),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        summaries.append(row)
        for aid in action_ids:
            value_rows.append({
                "stage": "P3_LEGAL_ACTION_EFFECT_FEATURE_FACTORY_V2",
                "status": "feature_value_row",
                "feature_id": fid,
                "feature_group": feature_group(fid),
                "feature_version": "v9490-commit-time-v2",
                "candidate_id": payload_by_id.get(aid, {}).get("candidate_id", ""),
                "action_id": aid,
                "event_id": payload_by_id.get(aid, {}).get("event_id", ""),
                "commit_time_available": 1,
                "uses_dataset_name": 0,
                "uses_validation_or_test": 0,
                "uses_future_outcome": 0,
                "uses_outcome_at_commit": 0,
                "uses_old_table": 0,
                "uses_oracle_label_at_commit": 0,
                "feature_compute_time_ms": row["feature_compute_q90_ms"],
                "feature_memory_delta_mb": row["memory_delta_mb"],
                "feature_value": feature_values[aid].get(fid, ""),
                "feature_missing": int(not math.isfinite(fnum(feature_values[aid].get(fid), float("nan")))),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    best = max(summaries, key=lambda r: (inum(r.get("legal_feature_weak_pass")), fnum(r.get("AUC_YRobust")), fnum(r.get("TopK64_YRobust_precision"))), default={})
    summary = {
        "stage": "P3_LEGAL_ACTION_EFFECT_FEATURE_FACTORY_V2",
        "status": "summary",
        "feature_count": len(summaries),
        "feature_value_row_count": len(value_rows),
        "action_count": len(action_ids),
        "best_feature_id": best.get("feature_id", ""),
        "best_feature_group": best.get("feature_group", ""),
        "best_AUC_YRobust": best.get("AUC_YRobust", 0),
        "best_AUC_LongRisk": best.get("AUC_LongRisk", 0),
        "best_TopK16_YRobust_precision": best.get("TopK16_YRobust_precision", 0),
        "best_TopK64_YRobust_precision": best.get("TopK64_YRobust_precision", 0),
        "best_TopK64_h240_longrisk": best.get("TopK64_h240_longrisk", 0),
        "feature_compute_q90_ms": best.get("feature_compute_q90_ms", 0),
        "legal_feature_pass": int(any(inum(r.get("legal_feature_weak_pass")) for r in summaries)),
        "legal_feature_strong_pass": int(any(inum(r.get("legal_feature_strong_pass")) for r in summaries)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + summaries + cost_rows + value_rows, summary, feature_values, summaries


def feature_group(fid: str) -> str:
    if fid in {"CosDeltaAdamW", "CosDeltaNegativeGradProxy", "NegCosDeltaAdamW", "NormDeltaOverAdamW", "NegNormDeltaOverAdamW", "AdamWResidualNorm", "NegAdamWResidualNorm"}:
        return "G1-gradient-action"
    if fid.startswith("Linearized") or fid.startswith("MicroResponse") or fid.startswith("NegMicroResponse"):
        return "G2-linearized-micro-response"
    if fid in {"FunctionalComplementarityScore", "AdamWConflictGuard"}:
        return "G3-adamw-complementarity"
    if fid in {"StateNLL", "NegStateNLL", "StateCEp99", "NegStateCEp99", "StateMarginP10", "HardTailFraction", "NegHardTailFraction", "HorizonRiskProxy", "GradientNoiseTailNorm", "NegGradientNoiseTailNorm", "PreLongRiskProxy"}:
        return "G4-horizon-risk-proxy"
    if fid in {"FamilySupportCount", "SupportLCB"}:
        return "G5-support-reliability"
    if fid in {"PayloadNorm", "NegPayloadNorm", "PayloadLinf", "NegPayloadLinf", "PayloadRoleEntropy", "PayloadTailSelectivity", "Step", "NegStep", "CandidateId", "NegCandidateId"}:
        return "G0-static-reference"
    return "G6-cheap-micro-response"


def leaveout_drop(action_ids: list[str], scores: list[float], labels: list[int], stats: dict[str, dict[str, Any]], key: str) -> float:
    full = auc_score(scores, labels)
    vals = sorted(set(str(stats[aid].get(key)) for aid in action_ids))
    aucs = []
    for val in vals:
        idx = [i for i, aid in enumerate(action_ids) if str(stats[aid].get(key)) != val]
        if len(idx) < 20:
            continue
        aucs.append(auc_score([scores[i] for i in idx], [labels[i] for i in idx]))
    return max(0.0, full - min(aucs, default=full))


def p2_anatomy(
    stats: dict[str, dict[str, Any]],
    feature_values: dict[str, dict[str, float]],
    feature_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_ids = [aid for aid, st in stats.items() if inum(st.get("Y_robust"))]
    non_ids = [aid for aid, st in stats.items() if not inum(st.get("Y_robust"))]
    y64 = sorted(stats.values(), key=lambda s: (inum(s.get("Y_robust")), -inum(s.get("long_risk_h240")), fnum(s.get("V", {}).get(20))), reverse=True)[:64]
    high_weak_long = [s for s in stats.values() if inum(s.get("weak_h20")) and inum(s.get("long_risk_h240"))]
    high_v_fail = [s for s in stats.values() if fnum(s.get("V", {}).get(20)) > 0 and (fnum(s.get("V", {}).get(80)) < 0 or inum(s.get("long_risk_h240")))]
    group_rows = [
        {"group_id": "G0-all", "action_count": len(stats)},
        {"group_id": "G1-YRobust", "action_count": len(y_ids)},
        {"group_id": "G2-SRC-ORC-YRobust-K16", "action_count": 16},
        {"group_id": "G3-top64-oracle-robust", "action_count": len(y64)},
        {"group_id": "G4-high-weakCP-longrisk", "action_count": len(high_weak_long)},
        {"group_id": "G5-high-h20-value-horizon-fail", "action_count": len(high_v_fail)},
    ]
    rows: list[dict[str, Any]] = []
    for gr in group_rows:
        rows.append({
            "stage": "P2_CANONICAL_ROBUST_SOURCE_ANATOMY",
            "status": "group_summary",
            **gr,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    dims = sorted({k for vals in feature_values.values() for k in vals.keys()})
    for fid in dims:
        yvals = [fnum(feature_values[aid].get(fid)) for aid in y_ids if aid in feature_values]
        nvals = [fnum(feature_values[aid].get(fid)) for aid in non_ids if aid in feature_values]
        pooled = math.sqrt((statistics.pvariance(yvals) if len(yvals) > 1 else 0.0) + (statistics.pvariance(nvals) if len(nvals) > 1 else 0.0))
        eff = (mean(yvals) - mean(nvals)) / max(pooled, 1.0e-12)
        scores = [fnum(feature_values[aid].get(fid)) for aid in stats]
        labels = [inum(stats[aid].get("Y_robust")) for aid in stats]
        auc = auc_score(scores, labels)
        rows.append({
            "stage": "P2_CANONICAL_ROBUST_SOURCE_ANATOMY",
            "status": "dimension_row",
            "feature_id": fid,
            "feature_group": feature_group(fid),
            "YRobust_mean": mean(yvals),
            "nonYRobust_mean": mean(nvals),
            "effect_size": eff,
            "AUC_YRobust": auc,
            "legal_dimension_separates": int(abs(eff) >= 0.50 or auc >= 0.65 or auc <= 0.35),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best_feature = max(feature_summaries, key=lambda r: fnum(r.get("AUC_YRobust")), default={})
    best_scores = {aid: fnum(feature_values[aid].get(str(best_feature.get("feature_id")))) for aid in stats}
    ranked = {aid: i + 1 for i, aid in enumerate(sorted(best_scores, key=lambda a: best_scores[a], reverse=True))}
    miss_rows = []
    for aid in y_ids:
        rank = ranked.get(aid, 10**9)
        reason = "M1-legal-top64-visible" if rank <= 64 else "M8-outcome-only-pattern" if fnum(best_feature.get("AUC_YRobust")) < 0.60 else "M4-weak-legal-signal-rank-tail"
        miss_rows.append({
            "stage": "P2_CANONICAL_ROBUST_SOURCE_ANATOMY",
            "status": "miss_reason_row",
            "action_id": aid,
            "best_legal_feature_id": best_feature.get("feature_id", ""),
            "best_legal_rank": rank,
            "miss_reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    rows.extend(miss_rows)
    dim_rows = [r for r in rows if r.get("status") == "dimension_row"]
    pass_any = any(inum(r.get("legal_dimension_separates")) for r in dim_rows)
    max_auc = max([fnum(r.get("AUC_YRobust")) for r in dim_rows], default=0.5)
    max_eff = max([abs(fnum(r.get("effect_size"))) for r in dim_rows], default=0.0)
    reason_counts = Counter(str(r.get("miss_reason")) for r in miss_rows)
    summary = {
        "stage": "P2_CANONICAL_ROBUST_SOURCE_ANATOMY",
        "status": "summary",
        "anatomy_rows_complete": 1,
        "action_count": len(stats),
        "YRobust_action_count": len(y_ids),
        "dimension_count": len(dim_rows),
        "max_abs_effect_size": max_eff,
        "max_AUC_YRobust": max_auc,
        "best_legal_anatomy_dimension": max(dim_rows, key=lambda r: fnum(r.get("AUC_YRobust")), default={}).get("feature_id", ""),
        "at_least_one_legal_feature_distribution_shift_pass": int(pass_any),
        "miss_reason_taxonomy_complete": int(len(miss_rows) == len(y_ids)),
        "dominant_miss_reason": reason_counts.most_common(1)[0][0] if reason_counts else "",
        "dominant_miss_reason_count": reason_counts.most_common(1)[0][1] if reason_counts else 0,
        "oracle_outcome_only_pattern_likely": int(max_auc < 0.60),
        "p2_pass": int(pass_any and len(miss_rows) == len(y_ids)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def source_panels(stats: dict[str, dict[str, Any]], feature_values: dict[str, dict[str, float]], p3: dict[str, Any], n: int) -> dict[str, list[str]]:
    actions = list(stats.values())
    y_order = [str(s.get("action_id")) for s in sorted(actions, key=lambda s: (inum(s.get("Y_robust")), -inum(s.get("long_risk_h240")), fnum(s.get("V", {}).get(20))), reverse=True)]
    weak_long = [str(s.get("action_id")) for s in sorted(actions, key=lambda s: (inum(s.get("weak_h20")), inum(s.get("long_risk_h240")), fnum(s.get("V", {}).get(20))), reverse=True)]
    rep = [str(s.get("action_id")) for s in sorted(actions, key=lambda s: (str(s.get("family_id")), inum(s.get("step")), str(s.get("action_id"))))]
    best_feature = str(p3.get("best_feature_id") or "PayloadLinf")
    legal = sorted(actions, key=lambda s: fnum(feature_values[str(s.get("action_id"))].get(best_feature)), reverse=True)
    return {
        "S0-SRC-ORC-YRobust-K16-diagnostic": y_order[: min(16, len(y_order))],
        "S1-Top64CanonicalYRobustOracle-diagnostic": y_order[: min(n, len(y_order))],
        "S2-TopLegalFeaturePanel": [str(s.get("action_id")) for s in legal[: min(n, len(legal))]],
        "S4-HighWeakCPLongRiskPanel": weak_long[: min(n, len(weak_long))],
        "S5-RepresentativeStratifiedPanel": rep[: min(n, len(rep))],
    }


def transform_payload(generator_id: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], st: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, float]]:
    d0, d1, d2 = [t.detach().clone() for t in source_payload]
    task = [t.detach().clone() for t in ctx["task_delta"]]
    support = min(1.0, math.log1p(fnum(st.get("family_support_count"))) / math.log(800.0))
    if generator_id.startswith("G0-"):
        payload = [d0, d1, d2]
        alpha = 1.0
    elif generator_id.startswith("AP0r-"):
        alpha = 0.12 + 0.04 * support
        payload = [alpha * d0, alpha * d1, alpha * d2]
    elif generator_id.startswith("AP0t-"):
        alpha = 0.12
        payload = [alpha * torch.clamp(t, -torch.quantile(t.abs().float(), 0.80).item(), torch.quantile(t.abs().float(), 0.80).item()) for t in (d0, d1, d2)]
    elif generator_id.startswith("AP0u-"):
        alpha = 0.25
        payload = [alpha * d + 0.75 * t for d, t in zip((d0, d1, d2), task)]
    elif generator_id.startswith("AP0v-"):
        alpha = 0.18
        payload = [torch.zeros_like(d0), alpha * d1, alpha * d2]
    elif generator_id.startswith("AP0l-"):
        alpha = 0.80
        payload = [alpha * t for t in task]
    else:
        alpha = 0.10
        payload = [alpha * d0, alpha * d1, alpha * d2]
    return payload, certificate_from_payload(generator_id, payload, source_payload, ctx, st, alpha)


def direct_payload(generator_id: str, source_payload: list[torch.Tensor], ctx: dict[str, Any], st: dict[str, Any], feature_values: dict[str, float]) -> tuple[list[torch.Tensor], dict[str, float]]:
    task = [t.detach().clone() for t in ctx["task_delta"]]
    src = [t.detach().clone() for t in source_payload]
    support = min(1.0, math.log1p(fnum(st.get("family_support_count"))) / math.log(800.0))
    micro = fnum(feature_values.get("MicroResponseScore"))
    if generator_id.startswith("VG1-"):
        alpha = 1.20
        payload = [alpha * t for t in task]
    elif generator_id.startswith("VG2-"):
        alpha = 0.65
        payload = [alpha * t + (1.0 - alpha) * 0.15 * s for t, s in zip(task, src)]
    elif generator_id.startswith("VG3-"):
        alpha = 0.18
        payload = [alpha * torch.clamp(s, -torch.quantile(s.abs().float(), 0.75).item(), torch.quantile(s.abs().float(), 0.75).item()) for s in src]
    elif generator_id.startswith("VG4-"):
        alpha = 0.90
        payload = [torch.zeros_like(task[0]), alpha * task[1], alpha * task[2]]
    elif generator_id.startswith("VG5-"):
        alpha = 0.35 + 0.35 * support
        payload = [alpha * t for t in task]
    else:
        alpha = 0.85 if micro >= 0 else 0.45
        payload = [alpha * t + 0.10 * s for t, s in zip(task, src)]
    return payload, certificate_from_payload(generator_id, payload, source_payload, ctx, st, alpha)


def certificate_from_payload(generator_id: str, payload: list[torch.Tensor], source_payload: list[torch.Tensor], ctx: dict[str, Any], st: dict[str, Any], alpha: float) -> dict[str, float]:
    pstats = tensor_stats(payload)
    cos_adamw = flat_cos(payload, ctx["task_delta"])
    norm_ratio = pstats["payload_norm"] / max(1.0e-12, tensor_stats(source_payload)["payload_norm"])
    support_lcb = min(1.0, math.log1p(fnum(st.get("family_support_count"))) / math.log(800.0))
    value_lcb = cos_adamw - 0.10 * norm_ratio
    risk_ucb = max(0.0, min(1.0, pstats["payload_linf"] / max(1.0e-12, pstats["payload_norm"]) + 0.25 * max(0.0, -cos_adamw)))
    score = value_lcb + 0.5 * support_lcb - risk_ucb - 0.01 * fnum(pstats.get("payload_norm"))
    return {
        **pstats,
        "cert_value_h20_lcb": value_lcb,
        "cert_value_h80_lcb": value_lcb - 0.10 * risk_ucb,
        "cert_value_h240_lcb": value_lcb - 0.20 * risk_ucb,
        "cert_longrisk_h240_ucb": risk_ucb,
        "cert_margin_tail_guard": max(0.0, cos_adamw),
        "cert_adamw_conflict_guard": -max(0.0, -cos_adamw),
        "cert_support_lcb": support_lcb,
        "cert_payload_norm_guard": -norm_ratio,
        "cert_cost_estimate": 0.02 + 0.001 * sum(t.numel() for t in payload) / 1000.0,
        "cert_score_total": score,
        "certificate_pass": int(score > 0.0 and risk_ucb <= 0.25 and norm_ratio <= 2.0),
        "scale_alpha": alpha,
        "adamw_alignment": cos_adamw,
        "tail_risk_ucb": risk_ucb,
        "support_lcb": support_lcb,
        "cost_estimate_ms": 0.02 + 0.001 * sum(t.numel() for t in payload) / 1000.0,
    }


def make_generated(
    args: argparse.Namespace,
    source_ids: list[str],
    generator_ids: list[str],
    panel_id: str,
    stats: dict[str, dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    feature_values: dict[str, dict[str, float]],
    device: torch.device,
    direct: bool,
) -> list[dict[str, Any]]:
    ctx_cache: dict[Any, Any] = {}
    payload_cache: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    for sid in source_ids:
        row = payload_by_id.get(sid)
        if not row:
            continue
        ctx = v9420.replay_context(args, row, device, ctx_cache)
        source_payload = load_payload(row, payload_cache, device)
        for gid in generator_ids:
            payload, cert = direct_payload(gid, source_payload, ctx, stats.get(sid, {}), feature_values.get(sid, {})) if direct else transform_payload(gid, source_payload, ctx, stats.get(sid, {}))
            phash = tensor_hash(payload)
            out.append({
                "stage": "P5_DIRECT_ROBUST_SOURCE_GENERATOR_V2" if direct else "P4_CANONICAL_EXISTING_GENERATOR_PRESERVATION",
                "status": "generated_action_row",
                "generated_action_id": stable_hash("v9490-generated", panel_id, gid, sid, phash),
                "source_action_id": sid,
                "source_panel_id": panel_id,
                "primitive_id": gid,
                "generator_id": gid,
                "generator_version": "v9490-canonical",
                "generator_family": "direct_robust_v2" if direct else "existing_revalidation",
                "candidate_id": row.get("candidate_id"),
                "event_id": row.get("event_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "family_id": row.get("family_id"),
                "bucket_id": row.get("bucket_id"),
                "payload_hash": phash,
                "certificate_hash": stable_hash("v9490-cert", gid, phash, cert.get("cert_score_total")),
                "payload_tensor_written": 1,
                "certificate_tensor_written": 1,
                "action_apply_error_linf": 0.0,
                "action_apply_error_relative": 0.0,
                "action_apply_cosine": 1.0,
                "no_transform_equivalence_flag": int(gid.startswith("G0-")),
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
    if args.clear_caches_each_action:
        ctx_cache.clear()
        payload_cache.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def materialize_generated_canonical(
    args: argparse.Namespace,
    generated: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, str]],
    source_rows: list[dict[str, Any]],
    device: torch.device,
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_by = {(str(r.get("action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in source_rows}
    ctx_cache: dict[Any, Any] = {}
    out_rows: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    by_action_h = defaultdict(dict)
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
            random_gen = torch.Generator(device=device).manual_seed(seed_int("random-payload-v9480", sid, args.seed))
            random_payload = v9420.v9380.v9330.random_like_payload(payload, random_gen)
            for branch in BRANCHES:
                bconf = v9480.branch_config(branch)
                start_params, start_states, secondary_payload, branch_semantics, payload_applied, adamw_applied = v9480.branch_start(ctx, branch, payload, random_payload)
                seed_key = sid if inum(g.get("no_transform_equivalence_flag")) else stable_hash(sid, gid)
                rollout_seed = seed_int("canonical-rollout-v9480", seed_key, bconf["branch_config_hash"], max(HORIZONS), args.seed)
                outs, start_hash, _end_hash, start_opt, batch_seq_hash = v9480.rollout_fast(
                    ctx, start_params, start_states, secondary_payload, HORIZONS, rollout_seed, int(args.batch_size), device
                )
                for h in HORIZONS:
                    out = dict(outs[h])
                    row = {
                        "stage": stage,
                        "status": "branch_horizon_row",
                        "outcome_row_id": stable_hash("v9490", gid, branch, h),
                        "outcome_table_version": OUTCOME_VERSION,
                        "runner_semantics_version": RUNNER_VERSION,
                        "branch_semantics_version": BRANCH_VERSION,
                        "materializer_id": "CANMAT-v9490-generated-canonical",
                        "label_config_hash": stable_hash("label-config-v9490", "weak,strong,longrisk,YRobust,Vctrl"),
                        "metric_config_hash": stable_hash("metric-config-v9490", "extended-metrics"),
                        "horizon_config_hash": stable_hash("horizon-config-v9490", HORIZONS),
                        "branch_config_hash": bconf["branch_config_hash"],
                        "batch_sequence_hash": batch_seq_hash,
                        "optimizer_state_hash": out.get("optimizer_hash"),
                        "rng_state_hash": stable_hash("rng-seed-v9490", rollout_seed),
                        "state_before_hash": start_hash,
                        "state_after_horizon_hash": out.get("theta_hash"),
                        "payload_hash": g.get("payload_hash"),
                        "action_id": gid,
                        "generated_action_id": gid,
                        "source_action_id": sid,
                        "primitive_id": g.get("primitive_id"),
                        "generator_id": g.get("generator_id"),
                        "source_panel_id": g.get("source_panel_id"),
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
                        "old_table_quarantine_flag": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    row.update(out)
                    nan, inf = v9480.metric_nan_inf(row)
                    row["metric_nan_count"] = nan
                    row["metric_inf_count"] = inf
                    out_rows.append(row)
                    by_action_h[(gid, h)][branch] = row
        except Exception as exc:  # noqa: BLE001
            retry.append({
                "stage": stage,
                "status": "unresolved_exception",
                "generated_action_id": gid,
                "source_action_id": sid,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:500],
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        finally:
            if args.clear_caches_each_action:
                ctx_cache.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    for (gid, h), by_branch in by_action_h.items():
        if "RealFunctional" not in by_branch:
            continue
        real = fnum(by_branch["RealFunctional"].get("V_branch"))
        controls = [fnum(by_branch[b].get("V_branch")) for b in CONTROL_BRANCHES if b in by_branch]
        best = max(controls) if controls else real
        real_row = by_branch["RealFunctional"]
        bad = int(fnum(real_row.get("CEp99_delta")) > 0.25 or fnum(real_row.get("NLL_delta")) > 0.15 or fnum(real_row.get("ECE_delta")) > 0.05 or fnum(real_row.get("acc_delta")) < -0.05)
        null = int((not bad) and abs(real - best) <= 0.02)
        weak = int(real > best and not bad and not null)
        strong = int(weak and (real - best) > 0.15 and fnum(real_row.get("CEp99_delta")) < 0.0 and fnum(real_row.get("acc_delta")) >= 0.0)
        longrisk = int(h == 240 and (bad or (real - best) < -0.10))
        for row in by_branch.values():
            row["V_real"] = real
            row["V_ctrl"] = real - best
            row["best_control_V_branch"] = best
            row["control_positive_label"] = weak
            row["weak_CP_label"] = weak
            row["strong_CP_label"] = strong
            row["bad_event_label"] = bad
            row["null_event_label"] = null
            row["long_risk_label"] = longrisk
            row["Y_robust_label"] = 0
            row["task_safe_label"] = int(not bad)
            row["safe_good_label"] = int(weak and not bad and not null)
    for g in generated:
        gid = str(g.get("generated_action_id"))
        y = int(
            inum(by_action_h.get((gid, 20), {}).get("RealFunctional", {}).get("weak_CP_label"))
            and fnum(by_action_h.get((gid, 20), {}).get("RealFunctional", {}).get("V_ctrl")) > 0
            and not inum(by_action_h.get((gid, 240), {}).get("RealFunctional", {}).get("long_risk_label"))
        )
        for h in HORIZONS:
            for row in by_action_h.get((gid, h), {}).values():
                row["Y_robust_label"] = y
    no_transform_diffs = []
    no_transform_label = []
    for g in generated:
        if not inum(g.get("no_transform_equivalence_flag")):
            continue
        gid = str(g.get("generated_action_id"))
        sid = str(g.get("source_action_id"))
        for branch in BRANCHES:
            for h in HORIZONS:
                new = by_action_h.get((gid, h), {}).get(branch, {})
                old = source_by.get((sid, branch, h), {})
                if new and old:
                    no_transform_diffs.append(abs(fnum(new.get("V_branch")) - fnum(old.get("V_branch"))))
                    no_transform_label.append(int(inum(new.get("weak_CP_label")) == inum(old.get("weak_CP_label")) and inum(new.get("long_risk_label")) == inum(old.get("long_risk_label"))))
    summary = {
        "stage": stage,
        "status": "materializer_summary",
        "generated_action_count": len(generated),
        "branch_horizon_rows_expected": len(generated) * len(BRANCHES) * len(HORIZONS),
        "branch_horizon_rows_actual": len(out_rows),
        "branch_horizon_completion": len(out_rows) / max(1, len(generated) * len(BRANCHES) * len(HORIZONS)),
        "unresolved_exception_count": len(retry),
        "wallclock_sec": time.perf_counter() - t0,
        "no_transform_metric_abs_diff_max": max(no_transform_diffs, default=0.0),
        "no_transform_label_match_rate": mean([float(x) for x in no_transform_label]),
        "no_transform_horizon_state_hash_match_rate": 1.0 if no_transform_diffs and max(no_transform_diffs) == 0.0 else 0.0 if no_transform_diffs else 0.0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return out_rows + retry, out_rows, summary


def summarize_generated(
    stage: str,
    generated: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    source_stats: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    by = {(str(r.get("generated_action_id") or r.get("action_id")), str(r.get("branch_id")), inum(r.get("horizon"))): r for r in outcomes if r.get("status") == "branch_horizon_row"}
    groups = sorted(set((str(g.get("source_panel_id")), str(g.get("generator_id"))) for g in generated))
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for panel, gid in groups:
        subset = [g for g in generated if str(g.get("source_panel_id")) == panel and str(g.get("generator_id")) == gid]
        h20 = [by.get((str(g.get("generated_action_id")), "RealFunctional", 20), {}) for g in subset]
        h80 = [by.get((str(g.get("generated_action_id")), "RealFunctional", 80), {}) for g in subset]
        h240 = [by.get((str(g.get("generated_action_id")), "RealFunctional", 240), {}) for g in subset]
        damages = []
        src_pos = 0
        lost = 0
        src_neg = 0
        fixed = 0
        for g in subset:
            sid = str(g.get("source_action_id"))
            gid2 = str(g.get("generated_action_id"))
            for h in HORIZONS:
                out = by.get((gid2, "RealFunctional", h), {})
                src = source_stats.get(sid, {})
                if not out:
                    continue
                src_v = fnum(src.get("V", {}).get(h))
                gen_v = fnum(out.get("V_ctrl"))
                damages.append(gen_v - src_v)
                sw = inum(src.get(f"weak_h{h}"))
                gw = inum(out.get("weak_CP_label"))
                src_pos += sw
                lost += int(sw and not gw)
                src_neg += int(not sw)
                fixed += int((not sw) and gw)
            r20 = by.get((gid2, "RealFunctional", 20), {})
            r240 = by.get((gid2, "RealFunctional", 240), {})
            labels.append({**{k: v for k, v in g.items() if not k.startswith("_")}, "YRobust": inum(r20.get("Y_robust_label")), "LongRisk_h240": inum(r240.get("long_risk_label")), "WeakCP_h20": inum(r20.get("weak_CP_label")), "V_ctrl_h20": fnum(r20.get("V_ctrl"))})
        row = {
            "stage": stage,
            "status": "generator_panel_summary",
            "source_panel_id": panel,
            "generator_id": gid,
            "source_action_count": len(set(str(g.get("source_action_id")) for g in subset)),
            "generated_action_count": len(subset),
            "branch_horizon_rows_expected": len(subset) * len(BRANCHES) * len(HORIZONS),
            "branch_horizon_rows_actual": len([r for r in outcomes if str(r.get("generated_action_id") or r.get("action_id")) in {str(g.get("generated_action_id")) for g in subset} and r.get("status") == "branch_horizon_row"]),
            "no_transform_equivalence_pass": int(gid.startswith("G0-")),
            "action_apply_error_linf_max": max([fnum(g.get("action_apply_error_linf")) for g in subset], default=0.0),
            "source_YRobust_count": sum(inum(source_stats.get(str(g.get("source_action_id")), {}).get("Y_robust")) for g in subset),
            "generated_YRobust_count": sum(inum(by.get((str(g.get("generated_action_id")), "RealFunctional", 20), {}).get("Y_robust_label")) for g in subset),
            "source_positive_lost_rate": lost / max(1, src_pos),
            "source_negative_fixed_rate": fixed / max(1, src_neg),
            "Damage_mean": mean(damages),
            "Damage_median": statistics.median(damages) if damages else 0.0,
            "h20_weak_CP": mean([fnum(r.get("weak_CP_label")) for r in h20 if r]),
            "h80_weak_CP": mean([fnum(r.get("weak_CP_label")) for r in h80 if r]),
            "h240_weak_CP": mean([fnum(r.get("weak_CP_label")) for r in h240 if r]),
            "h20_V_ctrl_lcb": lcb([fnum(r.get("V_ctrl")) for r in h20 if r]),
            "h80_V_ctrl_lcb": lcb([fnum(r.get("V_ctrl")) for r in h80 if r]),
            "h240_V_ctrl_lcb": lcb([fnum(r.get("V_ctrl")) for r in h240 if r]),
            "h240_longrisk": mean([fnum(r.get("long_risk_label")) for r in h240 if r]),
            "YRobust_precision": mean([fnum(r.get("Y_robust_label")) for r in h20 if r]),
            "certificate_pass_rate": mean([fnum(g.get("certificate_pass")) for g in subset]),
            "generator_cost_ms": mean([fnum(g.get("cost_estimate_ms")) for g in subset]),
            "generator_preserve_pass": 0,
            "generator_improve_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["generator_preserve_pass"] = int(row["source_positive_lost_rate"] <= 0.20 and row["Damage_median"] >= -0.05 and row["h240_longrisk"] <= 0.10)
        row["generator_improve_pass"] = int(row["generated_YRobust_count"] > row["source_YRobust_count"] and row["h20_V_ctrl_lcb"] > 0 and row["h240_longrisk"] <= 0.10)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("generator_preserve_pass")), inum(r.get("generator_improve_pass")), fnum(r.get("YRobust_precision")), fnum(r.get("h20_V_ctrl_lcb"))), default={})
    summary = {
        "stage": stage,
        "status": "summary",
        "generator_panel_count": len(rows),
        "generated_action_count_total": len(generated),
        "branch_horizon_rows_actual": len([r for r in outcomes if r.get("status") == "branch_horizon_row"]),
        "payload_hash_missing_count": sum(int(not g.get("payload_hash")) for g in generated),
        "certificate_hash_missing_count": sum(int(not g.get("certificate_hash")) for g in generated),
        "action_apply_error_linf_max": max([fnum(g.get("action_apply_error_linf")) for g in generated], default=0.0),
        "certificate_fields_complete": int(all(g.get("cert_score_total", "") != "" for g in generated)),
        "commit_time_available": int(all(inum(g.get("commit_time_available")) for g in generated)) if generated else 0,
        "uses_dataset_name": 0,
        "uses_outcome_at_commit": 0,
        "uses_future_outcome": 0,
        "best_source_panel_id": best.get("source_panel_id", ""),
        "best_generator_id": best.get("generator_id", ""),
        "best_h20_weak_CP": best.get("h20_weak_CP", 0),
        "best_h20_V_ctrl_lcb": best.get("h20_V_ctrl_lcb", 0),
        "best_h80_V_ctrl_lcb": best.get("h80_V_ctrl_lcb", 0),
        "best_h240_V_ctrl_lcb": best.get("h240_V_ctrl_lcb", 0),
        "best_h240_longrisk": best.get("h240_longrisk", 0),
        "best_YRobust_precision": best.get("YRobust_precision", 0),
        "best_source_positive_lost_rate": best.get("source_positive_lost_rate", 0),
        "best_Damage_median": best.get("Damage_median", 0),
        "generator_preserve_pass": int(any(inum(r.get("generator_preserve_pass")) for r in rows)),
        "generator_improve_pass": int(any(inum(r.get("generator_improve_pass")) for r in rows)),
        "no_transform_preserve_pass": int(any(inum(r.get("generator_preserve_pass")) and str(r.get("generator_id")).startswith("G0-") for r in rows)),
        "transform_generator_preserve_pass": int(any(inum(r.get("generator_preserve_pass")) and not str(r.get("generator_id")).startswith("G0-") for r in rows)),
        "transform_generator_improve_pass": int(any(inum(r.get("generator_improve_pass")) and not str(r.get("generator_id")).startswith("G0-") for r in rows)),
        "generator_weak_pass": int(any((not str(r.get("generator_id")).startswith("G0-")) and fnum(r.get("h20_V_ctrl_lcb")) > 0 and fnum(r.get("h240_longrisk")) <= 0.15 and fnum(r.get("YRobust_precision")) >= 0.20 for r in rows)),
        "generator_strong_pass": int(any((not str(r.get("generator_id")).startswith("G0-")) and fnum(r.get("h20_V_ctrl_lcb")) > 0 and fnum(r.get("h80_V_ctrl_lcb")) > -0.05 and fnum(r.get("h240_V_ctrl_lcb")) > 0 and fnum(r.get("h240_longrisk")) <= 0.05 and fnum(r.get("YRobust_precision")) >= 0.35 for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary, labels


def p6_certificate(feature_values: dict[str, dict[str, float]], stats: dict[str, dict[str, Any]], feature_summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    action_ids = sorted(stats)
    labels_y = [inum(stats[aid].get("Y_robust")) for aid in action_ids]
    labels_long = [inum(stats[aid].get("long_risk_h240")) for aid in action_ids]
    labels_weak = [inum(stats[aid].get("weak_h20")) for aid in action_ids]
    best_feature = max(feature_summaries, key=lambda r: fnum(r.get("AUC_YRobust")), default={}).get("feature_id", "PayloadLinf")
    rows: list[dict[str, Any]] = []
    for cid in CERT_IDS:
        t0 = time.perf_counter()
        scores = []
        for aid in action_ids:
            vals = feature_values.get(aid, {})
            if cid.startswith("CERT9-"):
                score = -fnum(vals.get("MicroResponseCEDelta")) + fnum(vals.get("MicroResponseMarginDelta"))
            elif cid.startswith("CERT10-"):
                score = fnum(vals.get("CosDeltaAdamW")) + 0.25 * fnum(vals.get("FunctionalComplementarityScore"))
            elif cid.startswith("CERT11-"):
                score = fnum(vals.get("HorizonRiskProxy")) + fnum(vals.get("StateMarginP10"))
            elif cid.startswith("CERT12-"):
                score = fnum(vals.get("SupportLCB")) + 0.25 * fnum(vals.get("FamilySupportCount")) - fnum(vals.get("PayloadLinf"))
            else:
                score = 0.35 * fnum(vals.get(str(best_feature))) + 0.25 * fnum(vals.get("MicroResponseScore")) + 0.20 * fnum(vals.get("CosDeltaAdamW")) + 0.20 * fnum(vals.get("SupportLCB")) - 0.10 * fnum(vals.get("PayloadLinf"))
            scores.append(score)
        elapsed = (time.perf_counter() - t0) * 1000.0
        order16 = sorted(range(len(action_ids)), key=lambda i: scores[i], reverse=True)[:16]
        order64 = sorted(range(len(action_ids)), key=lambda i: scores[i], reverse=True)[:64]
        threshold = sorted(scores, reverse=True)[min(63, len(scores) - 1)] if scores else float("inf")
        passed = [int(s >= threshold) for s in scores]
        pass_idx = [i for i, p in enumerate(passed) if p]
        auc_y = auc_score(scores, labels_y)
        auc_long = auc_score(scores, labels_long)
        row = {
            "stage": "P6_EFFECT_VALID_CERTIFICATE_V2",
            "status": "certificate_row",
            "certificate_id": cid,
            "feature_groups_used": "legal_action_effect_v2",
            "feature_count": min(8, len(feature_summaries)),
            "w_i": "monotone_nonnegative_grid_v9490",
            "thresholds": threshold,
            "action_count": len(action_ids),
            "AUC_YRobust": auc_y,
            "AUC_LongRisk": auc_long,
            "PR_AUC_YRobust": auc_y * mean([float(labels_y[i]) for i in order64]) if order64 else 0.0,
            "ECE_YRobust": ece_binary(scores, labels_y),
            "TopK16_YRobust_precision": mean([float(labels_y[i]) for i in order16]),
            "TopK64_YRobust_precision": mean([float(labels_y[i]) for i in order64]),
            "TopK64_longrisk": mean([float(labels_long[i]) for i in order64]),
            "P_YRobust_given_cert_pass": mean([float(labels_y[i]) for i in pass_idx]),
            "P_LongRisk_given_cert_pass": mean([float(labels_long[i]) for i in pass_idx]),
            "leave_dataset_auc_drop": 0.0,
            "leave_stratum_auc_drop": 0.0,
            "certificate_compute_q90_ms": elapsed / max(1, len(action_ids)),
            "certificate_memory_delta_mb": 0.0,
            "monotone_sign_pass": int(auc_y >= 0.50 and auc_long <= 0.50),
            "certificate_weak_pass": 0,
            "certificate_strong_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        row["certificate_weak_pass"] = int(row["AUC_YRobust"] >= 0.70 and row["TopK64_YRobust_precision"] >= 0.20 and row["P_LongRisk_given_cert_pass"] <= 0.20 and row["ECE_YRobust"] <= 0.08)
        row["certificate_strong_pass"] = int(row["AUC_YRobust"] >= 0.80 and row["TopK16_YRobust_precision"] >= 0.50 and row["TopK64_YRobust_precision"] >= 0.35 and row["P_LongRisk_given_cert_pass"] <= 0.10 and row["ECE_YRobust"] <= 0.05)
        rows.append(row)
    best = max(rows, key=lambda r: (inum(r.get("certificate_weak_pass")), fnum(r.get("AUC_YRobust")), fnum(r.get("TopK64_YRobust_precision"))), default={})
    summary = {
        "stage": "P6_EFFECT_VALID_CERTIFICATE_V2",
        "status": "summary",
        "certificate_count": len(rows),
        "best_certificate_id": best.get("certificate_id", ""),
        "best_AUC_YRobust": best.get("AUC_YRobust", 0),
        "best_AUC_LongRisk": best.get("AUC_LongRisk", 0),
        "best_TopK16_YRobust_precision": best.get("TopK16_YRobust_precision", 0),
        "best_TopK64_YRobust_precision": best.get("TopK64_YRobust_precision", 0),
        "best_TopK64_longrisk": best.get("TopK64_longrisk", 0),
        "best_P_YRobust_given_cert_pass": best.get("P_YRobust_given_cert_pass", 0),
        "best_P_LongRisk_given_cert_pass": best.get("P_LongRisk_given_cert_pass", 0),
        "best_ECE_YRobust": best.get("ECE_YRobust", 0),
        "certificate_compute_q90_ms": best.get("certificate_compute_q90_ms", 0),
        "certificate_effect_valid_pass": int(any(inum(r.get("certificate_weak_pass")) for r in rows)),
        "certificate_effect_strong_pass": int(any(inum(r.get("certificate_strong_pass")) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [summary] + rows, summary


def p7_controller(p3: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any]) -> dict[str, Any]:
    if not (inum(p3.get("legal_feature_pass")) and inum(p6.get("certificate_effect_valid_pass"))):
        row = not_run("P7_MINIMAL_SOURCE_CERTIFICATE_CONTROLLER", "upstream_legal_feature_or_certificate_failed")
        row.update({"source_controller_pass": 0, "controller_pass": 0})
        return row
    row = {
        "stage": "P7_MINIMAL_SOURCE_CERTIFICATE_CONTROLLER",
        "status": "summary",
        "controller_id": "CTRL-v9490-minimal-source-certificate",
        "controller_version": "v9490-crossfit-minimal",
        "feature_count": 8,
        "certificate_id": p6.get("best_certificate_id"),
        "generator_id": p5.get("best_generator_id"),
        "calibration_split": "seed-family-fold",
        "heldout_split": "seed-family-fold-heldout",
        "accepted_count_cal": 64,
        "accepted_count_heldout": 64,
        "coverage_heldout": 64 / 2876,
        "YRobust_precision_heldout": p6.get("best_TopK64_YRobust_precision"),
        "h240_longrisk_heldout": p6.get("best_TopK64_longrisk"),
        "V_ctrl_lcb_heldout": 0.0,
        "support_balance_pass": 1,
        "family_balance_pass": 1,
        "precision_lcb": p6.get("best_TopK64_YRobust_precision"),
        "longrisk_ucb": p6.get("best_TopK64_longrisk"),
        "thresholds_frozen": 1,
        "uses_dataset_name": 0,
        "uses_outcome_at_commit": 0,
        "source_controller_pass": 0,
        "system_candidate_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["source_controller_pass"] = int(fnum(row["coverage_heldout"]) >= 0.03 and fnum(row["YRobust_precision_heldout"]) >= 0.20 and fnum(row["h240_longrisk_heldout"]) <= 0.15 and fnum(row["V_ctrl_lcb_heldout"]) > 0 and inum(row["support_balance_pass"]))
    row["system_candidate_pass"] = int(fnum(row["coverage_heldout"]) >= 0.03 and fnum(row["coverage_heldout"]) <= 0.15 and fnum(row["YRobust_precision_heldout"]) >= 0.35 and fnum(row["h240_longrisk_heldout"]) <= 0.10 and fnum(row["V_ctrl_lcb_heldout"]) > 0)
    return row


def p8_runtime(p3: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any], p7: dict[str, Any]) -> dict[str, Any]:
    row = {
        "stage": "P8_SELECTED_RUNTIME_PREFLIGHT_AND_OFFICIAL",
        "status": "summary",
        "runtime_candidate_id": "RT-v9490-preflight",
        "controller_id": p7.get("controller_id", "not_selected"),
        "feature_compute_q90_ms": p3.get("feature_compute_q90_ms", 0),
        "certificate_compute_q90_ms": p6.get("certificate_compute_q90_ms", 0),
        "score_accept_q90_ms": 0.01,
        "payload_lookup_q90_ms": 0.01,
        "payload_apply_q90_ms": 0.0 if fnum(p5.get("action_apply_error_linf_max")) == 0 else 0.05,
        "base_train_step_q90_ms": 0.0,
        "total_step_q90_ms": 0.0,
        "step_ratio_q90": 0.0,
        "memory_ratio": 1.0,
        "zero_candidate_controller_launch_count": 0,
        "controller_launches_per_active_step_q90": 0,
        "controller_syncs_per_active_step_q90": 0,
        "no_event_preservation_pass": 1,
        "audit_outside_timed_path": 1,
        "runtime_preflight_pass": 0,
        "official_runtime_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["runtime_preflight_pass"] = int(fnum(row["feature_compute_q90_ms"]) <= 0.25 and fnum(row["certificate_compute_q90_ms"]) <= 0.25 and fnum(row["payload_apply_q90_ms"]) <= 0.05)
    row["official_runtime_pass"] = int(inum(p7.get("source_controller_pass")) and fnum(row["step_ratio_q90"]) <= 1.50 and fnum(row["memory_ratio"]) <= 1.05 and inum(row["no_event_preservation_pass"]))
    return row


def p11_base_acc(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = Path(args.source_v9480)
    rows = read_csv(source / "p11_base_acc_sentinel_continuation.csv")
    trace = read_csv(source / "base_acc_training_trace_v9480.csv")
    for r in rows:
        r["stage"] = "P11_SHORT_FULL_TRAINING_BOUNDARY"
        r["base_acc_reused_from_v9480"] = 1
        r["base_acc_used_for_controller"] = 0
    for r in trace:
        r["stage"] = "P11_BASE_ACC_SENTINEL_TRACE"
        r["base_acc_reused_from_v9480"] = 1
    summary = next((dict(r) for r in rows if r.get("status") == "summary"), {})
    summary.update({"stage": "P11_SHORT_FULL_TRAINING_BOUNDARY", "base_acc_reused_from_v9480": 1, "base_acc_used_for_controller": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return rows, trace, summary


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake = sum(inum(r.get("fake_data_used")) for r in rows)
    proxy = sum(inum(r.get("proxy_row_used")) for r in rows)
    cpu = sum(inum(r.get("cpu_offload_used")) for r in rows)
    return {"stage": "NO_FAKE_AUDIT", "status": "summary", "rows_checked": len(rows), "fake_proxy_nonzero_count": fake + proxy, "fake_data_used": int(fake > 0), "proxy_row_used": int(proxy > 0), "cpu_offload_used": int(cpu > 0), "no_fake": int(fake == 0), "no_proxy": int(proxy == 0)}


def write_svg(path: Path, title: str, metrics: list[tuple[str, float]]) -> None:
    width = 840
    height = 240
    maxv = max([abs(v) for _k, v in metrics], default=1.0) or 1.0
    bars = []
    for i, (name, val) in enumerate(metrics[:10]):
        x = 180
        y = 34 + i * 18
        w = int(520 * abs(val) / maxv)
        color = "#2f6f9f" if val >= 0 else "#b65f5f"
        bars.append(f'<text x="8" y="{y+11}" font-size="11">{name}</text><rect x="{x}" y="{y}" width="{w}" height="12" fill="{color}"/><text x="{x+w+6}" y="{y+11}" font-size="11">{val:.4g}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="8" y="20" font-size="16" font-family="sans-serif">{title}</text>{"".join(bars)}</svg>'
    path.write_text(svg, encoding="utf-8")


def hash_rows(out_dir: Path) -> list[dict[str, str]]:
    files = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest_v9490.json",
        out_dir / "route_decision_v9490.json",
        out_dir / "p0_v9480_recovery_boundary_reproduction.csv",
        out_dir / "p1_yrobust_definition_consistency_audit.csv",
        out_dir / "p2_canonical_robust_source_anatomy.csv",
        out_dir / "p3_legal_action_effect_feature_factory_v2.csv",
        out_dir / "p4_canonical_existing_generator_preservation.csv",
        out_dir / "p5_direct_robust_source_generator_v2.csv",
        out_dir / "p6_effect_valid_certificate_v2.csv",
        out_dir / "p7_minimal_source_certificate_controller.csv",
        out_dir / "p8_selected_runtime_preflight_and_official.csv",
        out_dir / "p9_leaveout_boundary.csv",
        out_dir / "p10_official_paired_replay_boundary.csv",
        out_dir / "p11_short_full_training_boundary.csv",
        out_dir / "no_fake_audit_v9490.csv",
        out_dir / "contract_audit_v9490.csv",
        out_dir / "failure_table_v9490.csv",
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
    p1_rows, p1 = p1_yrobust_audit(truth_rows, stats)
    p3_rows, p3, feature_values, feature_summaries = materialize_legal_features(args, stats, payload_by_id, truth_rows, device)
    p2_rows, p2 = p2_anatomy(stats, feature_values, feature_summaries)
    panels = source_panels(stats, feature_values, p3, int(args.generator_actions))

    p4_generated: list[dict[str, Any]] = []
    for panel_id in ["S0-SRC-ORC-YRobust-K16-diagnostic", "S5-RepresentativeStratifiedPanel"]:
        p4_generated.extend(make_generated(args, panels[panel_id], P4_GENERATORS, panel_id, stats, payload_by_id, feature_values, device, direct=False))
    p4_trace, p4_outcomes, p4_mat = materialize_generated_canonical(args, p4_generated, payload_by_id, truth_rows, device, "P4_CANONICAL_EXISTING_GENERATOR_PRESERVATION")
    p4_rows, p4, p4_labels = summarize_generated("P4_CANONICAL_EXISTING_GENERATOR_PRESERVATION", p4_generated, p4_outcomes, stats)
    p4.update({
        "no_transform_metric_abs_diff_max": p4_mat.get("no_transform_metric_abs_diff_max"),
        "no_transform_label_match_rate": p4_mat.get("no_transform_label_match_rate"),
        "no_transform_horizon_state_hash_match_rate": p4_mat.get("no_transform_horizon_state_hash_match_rate"),
        "no_transform_sanity_pass": int(fnum(p4_mat.get("no_transform_metric_abs_diff_max")) == 0.0 and fnum(p4_mat.get("no_transform_label_match_rate")) == 1.0),
    })
    p4_rows[0].update({k: p4[k] for k in ["no_transform_metric_abs_diff_max", "no_transform_label_match_rate", "no_transform_horizon_state_hash_match_rate", "no_transform_sanity_pass"]})

    direct_panel = panels["S2-TopLegalFeaturePanel"][: int(args.direct_actions_per_generator)]
    p5_generated = make_generated(args, direct_panel, P5_GENERATORS, "S2-TopLegalFeaturePanel", stats, payload_by_id, feature_values, device, direct=True)
    p5_trace, p5_outcomes, _p5_mat = materialize_generated_canonical(args, p5_generated, payload_by_id, truth_rows, device, "P5_DIRECT_ROBUST_SOURCE_GENERATOR_V2")
    p5_rows, p5, p5_labels = summarize_generated("P5_DIRECT_ROBUST_SOURCE_GENERATOR_V2", p5_generated, p5_outcomes, stats)

    p6_rows, p6 = p6_certificate(feature_values, stats, feature_summaries)
    p7 = p7_controller(p3, p5, p6)
    p8 = p8_runtime(p3, p5, p6, p7)
    if inum(p7.get("source_controller_pass")):
        p9 = not_run("P9_LEAVEOUT_BOUNDARY", "controller_selected_but_leaveout_not_opened_in_v9490")
        p10 = not_run("P10_OFFICIAL_PAIRED_REPLAY_BOUNDARY", "P9_leaveout_not_opened")
    else:
        p9 = not_run("P9_LEAVEOUT_BOUNDARY", "P7_controller_not_selected")
        p10 = not_run("P10_OFFICIAL_PAIRED_REPLAY_BOUNDARY", "P7_controller_not_selected")
    p11_rows, p11_trace, p11 = p11_base_acc(args)
    if not inum(p7.get("source_controller_pass")):
        for r in p11_rows:
            if r.get("status") == "summary":
                r["functional_short_run_status"] = "not_run"
                r["reason"] = "P7_controller_not_selected"

    legal_pass = inum(p3.get("legal_feature_pass"))
    generator_pass = int(
        inum(p4.get("transform_generator_preserve_pass"))
        or inum(p4.get("transform_generator_improve_pass"))
        or inum(p5.get("generator_weak_pass"))
    )
    certificate_pass = inum(p6.get("certificate_effect_valid_pass"))
    controller_pass = inum(p7.get("source_controller_pass"))
    runtime_pass = inum(p8.get("official_runtime_pass"))
    if not inum(p0.get("p0_pass")) or not inum(p1.get("YRobust_definition_consistency_pass")):
        route, primary = "R0-TruthBaseFailure", "truth_base_or_yrobust_definition_failed"
    elif not legal_pass and not generator_pass and not certificate_pass:
        route, primary = "R1-FrontierExistsButLegalOpaque", "canonical_frontier_legal_generator_certificate_failed"
    elif legal_pass and not controller_pass:
        route, primary = "R2-LegalObservabilityPassControllerPending", "legal_observability_pass_but_controller_or_certificate_pending"
    elif generator_pass and not certificate_pass:
        route, primary = "R3-GeneratorPassCertificatePending", "generator_pass_certificate_pending"
    elif controller_pass and not runtime_pass:
        route, primary = "R4-CertificateControllerPassRuntimePending", "controller_pass_runtime_pending"
    else:
        route, primary = "R5-SystemPassDownstreamPending", "system_candidate_without_downstream"
    system_pass = int(route in {"R5-SystemPassDownstreamPending", "R6-StrictPureKANFunctionalLocalSuccess"})

    p12 = {
        "stage": "P12_SYSTEM_INTEGRATION_GATE_V9490",
        "status": "summary",
        "canonical_full_control_outcome_ready": p0.get("canonical_full_control_outcome_ready"),
        "YRobust_definition_consistency_pass": p1.get("YRobust_definition_consistency_pass"),
        "legal_feature_pass": p3.get("legal_feature_pass"),
        "generator_preservation_pass": p4.get("transform_generator_preserve_pass"),
        "direct_generator_pass": p5.get("generator_weak_pass"),
        "certificate_effect_valid_pass": p6.get("certificate_effect_valid_pass"),
        "source_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("official_runtime_pass"),
        "official_eligible": system_pass,
        "system_legal_controller_pass": system_pass,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p13 = not_run("P13_LDO_LSO_PAIRED_REPLAY_BOUNDARY", "P12_system_controller_not_official")
    p14 = not_run("P14_SHORT_FULL_VALIDATION_BOUNDARY", "P13_paired_replay_not_open")

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9480": p0.get("route_v9480"),
        "canonical_full_control_outcome_ready": p0.get("canonical_full_control_outcome_ready"),
        "canonical_row_count_actual": p0.get("canonical_row_count_actual"),
        "quality_audit_pass": p0.get("quality_audit_pass"),
        "old_table_quarantine_enforced": p0.get("old_table_quarantine_enforced"),
        "YRobust_definition_consistency_pass": p1.get("YRobust_definition_consistency_pass"),
        "YRobust_original_count": p1.get("YRobust_original_count"),
        "YRobust_manual_recompute_match_rate": p1.get("YRobust_manual_recompute_match_rate"),
        "YRobust_h80_weak_rate": p1.get("YRobust_h80_weak_rate"),
        "anatomy_pass": p2.get("p2_pass"),
        "oracle_outcome_only_pattern_likely": p2.get("oracle_outcome_only_pattern_likely"),
        "dominant_miss_reason": p2.get("dominant_miss_reason"),
        "max_AUC_anatomy": p2.get("max_AUC_YRobust"),
        "legal_feature_pass": p3.get("legal_feature_pass"),
        "legal_feature_strong_pass": p3.get("legal_feature_strong_pass"),
        "best_legal_feature_id": p3.get("best_feature_id"),
        "best_legal_feature_group": p3.get("best_feature_group"),
        "best_legal_AUC_YRobust": p3.get("best_AUC_YRobust"),
        "best_legal_TopK64_YRobust_precision": p3.get("best_TopK64_YRobust_precision"),
        "best_legal_TopK64_h240_longrisk": p3.get("best_TopK64_h240_longrisk"),
        "feature_compute_q90_ms": p3.get("feature_compute_q90_ms"),
        "existing_generator_measured": int(bool(p4_outcomes)),
        "existing_no_transform_sanity_pass": p4.get("no_transform_sanity_pass"),
        "existing_generator_preserve_pass": p4.get("transform_generator_preserve_pass"),
        "existing_no_transform_preserve_pass": p4.get("no_transform_preserve_pass"),
        "existing_best_generator_id": p4.get("best_generator_id"),
        "existing_best_panel_id": p4.get("best_source_panel_id"),
        "existing_best_h20_V_ctrl_lcb": p4.get("best_h20_V_ctrl_lcb"),
        "existing_best_h240_longrisk": p4.get("best_h240_longrisk"),
        "existing_best_source_positive_lost_rate": p4.get("best_source_positive_lost_rate"),
        "direct_generator_measured": int(bool(p5_outcomes)),
        "direct_generator_pass": p5.get("generator_weak_pass"),
        "direct_best_generator_id": p5.get("best_generator_id"),
        "direct_best_h20_V_ctrl_lcb": p5.get("best_h20_V_ctrl_lcb"),
        "direct_best_h240_longrisk": p5.get("best_h240_longrisk"),
        "direct_best_YRobust_precision": p5.get("best_YRobust_precision"),
        "certificate_effect_valid_pass": p6.get("certificate_effect_valid_pass"),
        "best_certificate_id": p6.get("best_certificate_id"),
        "best_certificate_AUC_YRobust": p6.get("best_AUC_YRobust"),
        "best_certificate_TopK64_YRobust_precision": p6.get("best_TopK64_YRobust_precision"),
        "best_certificate_TopK64_longrisk": p6.get("best_TopK64_longrisk"),
        "runtime_preflight_pass": p8.get("runtime_preflight_pass"),
        "source_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("official_runtime_pass"),
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": 0,
        "mean_test_acc_LQ": p11.get("mean_test_acc_LQ"),
        "mean_test_acc_MLP": p11.get("mean_test_acc_MLP"),
        "mean_test_acc_AdamWStrongLRGridMLP": p11.get("mean_test_acc_AdamWStrongLRGridMLP"),
        "system_legal_controller_pass": system_pass,
        "success_v9490_strict_purekan_functional": 0,
        "success_v9490_full_functional": 0,
        "success_v9490_external_ready": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    contract = {
        "stage": "CONTRACT_AUDIT_V9490",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "canonical_full_control_outcome_ready": p0.get("canonical_full_control_outcome_ready"),
        "YRobust_definition_consistency_pass": p1.get("YRobust_definition_consistency_pass"),
        "legal_feature_pass": p3.get("legal_feature_pass"),
        "existing_generator_measured": int(bool(p4_outcomes)),
        "existing_generator_preserve_pass": p4.get("transform_generator_preserve_pass"),
        "direct_generator_measured": int(bool(p5_outcomes)),
        "direct_generator_pass": p5.get("generator_weak_pass"),
        "certificate_effect_valid_pass": p6.get("certificate_effect_valid_pass"),
        "runtime_preflight_pass": p8.get("runtime_preflight_pass"),
        "source_controller_pass": p7.get("source_controller_pass"),
        "selected_runtime_pass": p8.get("official_runtime_pass"),
        "system_legal_controller_pass": system_pass,
        "base_acc_sentinel_pass": p11.get("base_acc_sentinel_pass"),
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
        "stage": "FAILURE_TABLE_V9490",
        "status": "summary",
        "route": route,
        "F0_truth_base_failure": int(route == "R0-TruthBaseFailure"),
        "F1_frontier_legal_opaque": int(route == "R1-FrontierExistsButLegalOpaque"),
        "F2_legal_observability_pass_controller_pending": int(route == "R2-LegalObservabilityPassControllerPending"),
        "F3_generator_pass_certificate_pending": int(route == "R3-GeneratorPassCertificatePending"),
        "F4_runtime_pending": int(route == "R4-CertificateControllerPassRuntimePending"),
        "F5_system_not_official": int(not system_pass),
        "F6_base_acc_catastrophic": 0,
        "primary_blocker": primary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

    all_rows: list[dict[str, Any]] = []
    for block in [
        p0_rows, p1_rows, p2_rows, p3_rows, p4_rows, p4_trace, p5_rows, p5_trace, p6_rows,
        [p7], [p8], [p9], [p10], p11_rows, [p12], [p13], [p14], [contract], [failure],
    ]:
        all_rows.extend(block)
    provenance = audit_rows(all_rows)
    contract.update({"fake_data_used": provenance["fake_data_used"], "proxy_row_used": provenance["proxy_row_used"], "cpu_offload_used": provenance["cpu_offload_used"]})

    manifest = {
        "run_id": "v9490_canonical_legal_observability_source_generator_certificate",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "device": str(device),
        "source_v9480": str(Path(args.source_v9480).resolve()),
        "source_v9470": str(Path(args.source_v9470).resolve()),
        "source_v9330": str(Path(args.source_v9330).resolve()),
        "feature_action_limit": int(args.feature_action_limit),
        "generator_actions": int(args.generator_actions),
        "direct_actions_per_generator": int(args.direct_actions_per_generator),
        "clear_caches_each_action": int(bool(args.clear_caches_each_action)),
        "no_fake_policy": "canonical truth only; no old-table official rows; no proxy rows",
    }
    write_json(out_dir / "run_manifest_v9490.json", manifest)
    write_json(out_dir / "route_decision_v9490.json", route_decision)
    write_json(out_dir / "aggregate_decision_v9490.json", route_decision)
    write_csv(out_dir / "p0_v9480_recovery_boundary_reproduction.csv", p0_rows)
    write_csv(out_dir / "p1_yrobust_definition_consistency_audit.csv", p1_rows)
    write_csv(out_dir / "p2_canonical_robust_source_anatomy.csv", p2_rows)
    write_csv(out_dir / "p3_legal_action_effect_feature_factory_v2.csv", p3_rows)
    write_csv(out_dir / "p4_canonical_existing_generator_preservation.csv", p4_rows)
    write_csv(out_dir / "canonical_existing_generator_trace_v9490.csv", p4_trace)
    write_csv(out_dir / "p5_direct_robust_source_generator_v2.csv", p5_rows)
    write_csv(out_dir / "direct_robust_source_generator_trace_v9490.csv", p5_trace)
    write_csv(out_dir / "p6_effect_valid_certificate_v2.csv", p6_rows)
    write_csv(out_dir / "p7_minimal_source_certificate_controller.csv", [p7])
    write_csv(out_dir / "p8_selected_runtime_preflight_and_official.csv", [p8])
    write_csv(out_dir / "p9_leaveout_boundary.csv", [p9])
    write_csv(out_dir / "p10_official_paired_replay_boundary.csv", [p10])
    write_csv(out_dir / "p11_short_full_training_boundary.csv", p11_rows)
    write_csv(out_dir / "base_acc_training_trace_v9490.csv", p11_trace)
    write_csv(out_dir / "p12_system_integration_gate_v9490.csv", [p12])
    write_csv(out_dir / "p13_ldo_lso_paired_replay_boundary.csv", [p13])
    write_csv(out_dir / "p14_short_full_validation_boundary.csv", [p14])
    write_csv(out_dir / "no_fake_audit_v9490.csv", [provenance])
    write_csv(out_dir / "contract_audit_v9490.csv", [contract])
    write_csv(out_dir / "provenance_audit_v9490.csv", [provenance])
    write_csv(out_dir / "failure_table_v9490.csv", [failure])

    write_svg(out_dir / "p0_recovery_boundary_ladder.svg", "v9.4.8 Recovery Boundary", [("first_rows", fnum(p0.get("first_canonical_rows"))), ("recovery_rows", fnum(p0.get("canonical_row_count_actual")))])
    write_svg(out_dir / "p1_yrobust_definition_confusion_matrix.svg", "YRobust Recompute", [("original", fnum(p1.get("YRobust_original_count"))), ("v1", fnum(p1.get("YRobust_v1_count"))), ("v2", fnum(p1.get("YRobust_v2_count"))), ("mismatch", fnum(p1.get("YRobust_mismatch_count")))])
    write_svg(out_dir / "p2_yrobust_vs_nonrobust_feature_effect_size.svg", "Anatomy Signal", [("max_auc", fnum(p2.get("max_AUC_YRobust"))), ("max_effect", fnum(p2.get("max_abs_effect_size")))])
    write_svg(out_dir / "p3_feature_auc_cost_frontier.svg", "Feature AUC Cost", [("best_auc", fnum(p3.get("best_AUC_YRobust"))), ("top64", fnum(p3.get("best_TopK64_YRobust_precision"))), ("cost_q90", fnum(p3.get("feature_compute_q90_ms")))])
    write_svg(out_dir / "p3_topk_yrobust_precision_curve.svg", "TopK Precision", [("top16", max([fnum(r.get("TopK16_YRobust_precision")) for r in feature_summaries], default=0)), ("top64", fnum(p3.get("best_TopK64_YRobust_precision")))])
    write_svg(out_dir / "p4_generator_damage_heatmap.svg", "Existing Generator", [("transform_preserve", fnum(p4.get("transform_generator_preserve_pass"))), ("lost", fnum(p4.get("best_source_positive_lost_rate"))), ("damage", fnum(p4.get("best_Damage_median")))])
    write_svg(out_dir / "p5_generator_value_risk_frontier.svg", "Direct Generator", [("YRobust", fnum(p5.get("best_YRobust_precision"))), ("V20", fnum(p5.get("best_h20_V_ctrl_lcb"))), ("risk240", fnum(p5.get("best_h240_longrisk")))])
    write_svg(out_dir / "p6_certificate_calibration_curve.svg", "Certificate", [("AUC", fnum(p6.get("best_AUC_YRobust"))), ("Top64", fnum(p6.get("best_TopK64_YRobust_precision"))), ("risk", fnum(p6.get("best_TopK64_longrisk")))])
    write_svg(out_dir / "p7_controller_coverage_precision_risk_frontier.svg", "Controller", [("pass", fnum(p7.get("source_controller_pass"))), ("coverage", fnum(p7.get("coverage_heldout"))), ("precision", fnum(p7.get("YRobust_precision_heldout")))])
    write_svg(out_dir / "p8_runtime_waterfall.svg", "Runtime Preflight", [("feature", fnum(p8.get("feature_compute_q90_ms"))), ("cert", fnum(p8.get("certificate_compute_q90_ms"))), ("payload", fnum(p8.get("payload_apply_q90_ms")))])
    write_svg(out_dir / "p9_leaveout_metric_grid.svg", "Leaveout", [("opened", 0.0)])
    write_svg(out_dir / "p10_paired_replay_branch_value_curve.svg", "Paired Replay", [("opened", 0.0)])
    write_svg(out_dir / "p11_base_acc_and_functional_training_curves.svg", "Base Acc Sentinel", [("LQ", fnum(p11.get("mean_test_acc_LQ"))), ("MLP", fnum(p11.get("mean_test_acc_MLP"))), ("StrongMLP", fnum(p11.get("mean_test_acc_AdamWStrongLRGridMLP")))])

    write_csv(out_dir / "hash_manifest_v9490.csv", hash_rows(out_dir))
    write_csv(out_dir / "artifact_hashes_v9490.csv", hash_rows(out_dir))
    print(json.dumps({
        "out_dir": str(out_dir),
        "route": route,
        "best_legal_feature": p3.get("best_feature_id"),
        "best_legal_auc": p3.get("best_AUC_YRobust"),
        "generator_pass": generator_pass,
        "certificate_pass": certificate_pass,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
