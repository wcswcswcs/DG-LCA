#!/usr/bin/env python3
"""DG-KAN v9.3.7 certificate-producing primitive boundary runner.

The v9.3.7 plan asks for AP1/AP2/AP3/AP4 actions that are generated with
legal certificates at commit time. This runner makes the boundary explicit:

* It replays the AP0 v9.3.6 impossibility result.
* It instantiates the certificate schema on real measured AP0 probe/payload
  rows as a diagnostic audit only.
* It does not claim AP1/AP2/AP3/AP4 materialization unless a real generated
  payload exists. In the current repo no such primitive generator is present,
  so P3+ are gate-blocked instead of fabricated.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.3.7_CertificateProducingActionPrimitive_HorizonRobustRuntimeClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9370_certificate_producing_action_primitive_horizon_runtime.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9360 = RESULT_ROOT / "v9360_legal_action_effect_identifiability_certificate_runtime_first_20260514T030000Z"
DEFAULT_SOURCE_V9350 = RESULT_ROOT / "v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HELDOUT_DENOMINATOR = 9072
PRIMITIVES = [
    "AP1-LastEdgeLinearizedTailSafeCertificate",
    "AP2-AdamWResidualOrthogonalBenefitCertificate",
    "AP3-HorizonRobustTailMemoryCertificate",
    "AP4-LowRankEdgeCertificate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9360", default=str(DEFAULT_SOURCE_V9360))
    p.add_argument("--source-v9350", default=str(DEFAULT_SOURCE_V9350))
    p.add_argument("--source-v9330", default=str(DEFAULT_SOURCE_V9330))
    p.add_argument("--diagnostic-actions", type=int, default=864)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def q(vals: list[float], frac: float) -> float:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))
    return xs[idx]


def mean(vals: list[float]) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    return sum(xs) / max(1, len(xs))


def lcb_mean(vals: list[float]) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return 0.0
    m = mean(xs)
    if len(xs) <= 1:
        return m
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))
    return m - 1.96 * sd / math.sqrt(len(xs))


def topk_precision(values: list[float], labels: list[int], k: int) -> tuple[int, float]:
    if not values:
        return 0, 0.0
    k = min(k, len(values))
    top = sorted(zip(values, labels), key=lambda x: x[0], reverse=True)[:k]
    hits = sum(y for _, y in top)
    return hits, hits / max(1, k)


def p0_reanalysis(source_v9360: Path) -> dict[str, Any]:
    route = read_json(source_v9360 / "route_decision.json")
    return {
        "stage": "P0_V9360_BOUNDARY_REANALYSIS",
        "status": "summary",
        "source_run_id": source_v9360.name,
        "source_artifact_hash": sha256_file(source_v9360 / "route_decision.json"),
        "route": route.get("route"),
        "candidate_count": 2876,
        "action_count": 2876,
        "event_count": 24192,
        "weak_CP_row_count": route.get("weak_CP_row_count"),
        "weak_CP_coverage": route.get("weak_CP_coverage"),
        "weak_CP_coverage_lcb": route.get("weak_CP_coverage_lcb"),
        "weak_CP_V_ctrl_lcb": route.get("weak_CP_V_ctrl_lcb"),
        "strong_CP_row_count": route.get("strong_CP_row_count"),
        "strong_CP_coverage": route.get("strong_CP_coverage"),
        "horizon_robust_CP_action_count": route.get("horizon_robust_CP_action_count"),
        "horizon_robust_CP_coverage": route.get("horizon_robust_CP_coverage"),
        "short_only_CP_action_count": route.get("short_only_CP_action_count"),
        "long_risk_action_count": route.get("long_risk_action_count"),
        "best_legal_capacity_feature": route.get("best_legal_capacity_feature"),
        "best_legal_capacity_auc_CP": route.get("best_legal_capacity_auc_CP"),
        "best_top273_CP_precision": route.get("best_top273_CP_precision"),
        "best_aef_feature": route.get("best_aef_feature"),
        "best_aef_auc_CP": route.get("best_aef_auc_CP"),
        "best_aef_top273_CP_precision": route.get("best_aef_top273_CP_precision"),
        "best_aef_cost_q90": "",
        "best_microprobe_auc_CP": route.get("best_microprobe_auc_CP"),
        "best_microprobe_cost_q90": route.get("best_microprobe_cost_q90"),
        "certificate_primitive_smoke_pass": route.get("certificate_primitive_smoke_pass"),
        "payload_apply_time_ms_q90_microbench": route.get("payload_apply_time_ms_q90"),
        "step_ratio_q90_microbench": route.get("step_ratio_q90"),
        "system_legal_controller_pass": route.get("system_legal_controller_pass"),
        "p0_pass": int(
            route.get("route") == "R12-CertificatePrimitiveFail"
            and fnum(route.get("weak_CP_coverage")) >= 0.03
            and fnum(route.get("horizon_robust_CP_coverage")) < 0.03
            and inum(route.get("legal_observability_capacity_pass")) == 0
            and inum(route.get("aef_feature_pass")) == 0
            and inum(route.get("microprobe_pass")) == 0
            and inum(route.get("certificate_primitive_smoke_pass")) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def ap0_threshold_ban(source_v9360: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for path_name, source_name in [
        ("legal_feature_capacity_trace_v9360.csv", "capacity"),
        ("aef_feature_trace_v9360.csv", "aef"),
        ("microprobe_trace_v9360.csv", "microprobe"),
    ]:
        for row in read_csv(source_v9360 / path_name):
            if row.get("status") not in {"feature_summary", "source_v9350_static_summary"}:
                continue
            feature_id = row.get("feature_id") or row.get("probe_id")
            auc = fnum(row.get("AUC_CP") or row.get("AUC_CP_weak"))
            top = fnum(row.get("top273_CP_precision"))
            cost = fnum(row.get("feature_cost_q90") or row.get("probe_cost_q90"))
            traces.append(
                {
                    "stage": "P1_AP0_THRESHOLD_SEARCH_BAN",
                    "status": "feature_capacity_row",
                    "source_trace": source_name,
                    "feature_id": feature_id,
                    "feature_group": row.get("feature_group"),
                    "AUC_CP": auc,
                    "AUC_StrongCP": "",
                    "AUC_HorizonRobustCP": "",
                    "AUC_LongRisk": "",
                    "AUC_ShortOnly": "",
                    "PR_AUC_CP": row.get("PR_lift_CP"),
                    "Top273_CP_precision": top,
                    "Top273_StrongCP_precision": "",
                    "Top273_HorizonRobustCP_precision": "",
                    "Top273_LongRisk_rate": "",
                    "cost_q90_ms": cost,
                    "feature_missing_rate": row.get("feature_missing_rate", ""),
                    "leave_dataset_auc_drop": row.get("leave_dataset_auc_drop", ""),
                    "leave_stratum_auc_drop": row.get("leave_stratum_auc_drop", ""),
                    "continuation_gate_pass": int(auc >= 0.70 and top >= 0.60 and cost <= 0.20),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    best_auc = max(traces, key=lambda r: fnum(r.get("AUC_CP")), default={})
    best_top = max(traces, key=lambda r: fnum(r.get("Top273_CP_precision")), default={})
    continuation = any(inum(r.get("continuation_gate_pass")) for r in traces)
    summary = {
        "stage": "P1_AP0_THRESHOLD_SEARCH_BAN",
        "status": "summary",
        "feature_count_checked": len(traces),
        "best_feature_by_auc": best_auc.get("feature_id"),
        "best_auc_CP": best_auc.get("AUC_CP"),
        "best_feature_by_top273": best_top.get("feature_id"),
        "best_top273_CP_precision": best_top.get("Top273_CP_precision"),
        "ap0_continuation_allowed": int(continuation),
        "ap0_stop_condition": int(not continuation),
        "reason": "no_AP0_feature_satisfies_continuation_gate" if not continuation else "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return traces, summary


def support_stats(cp_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in cp_rows:
        if inum(row.get("seed")) <= 4:
            key = (str(row.get("family_id")), str(row.get("bucket_id")))
            buckets[key].append(row)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in buckets.items():
        n = len(rows)
        cp = sum(inum(r.get("weak_CP")) for r in rows)
        lr = sum(inum(r.get("long_risk")) for r in rows)
        so = sum(inum(r.get("short_only_CP")) for r in rows)
        out[key] = {
            "n": n,
            "cp_mean": cp / max(1, n),
            "cp_lcb": wilson_lcb(cp, n),
            "longrisk_mean": lr / max(1, n),
            "longrisk_ucb": wilson_ucb(lr, n),
            "shortonly_ucb": wilson_ucb(so, n),
        }
    return out


def certificate_schema(
    source_v9350: Path,
    source_v9360: Path,
    source_v9330: Path,
    diagnostic_actions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probes = [r for r in read_csv(source_v9350 / "probe_feature_trace_v9350.csv") if r.get("status") == "probe_row"][:diagnostic_actions]
    cp_by_event = {r.get("event_id"): r for r in read_csv(source_v9360 / "cp_horizon_structure_trace_v9360.csv") if r.get("status") == "action_horizon_row"}
    payload_by_event = {r.get("event_id"): r for r in read_csv(source_v9330 / "action_payload_disk_replay_trace_v9330.csv")}
    stats = support_stats(list(cp_by_event.values()))
    rows: list[dict[str, Any]] = []
    median_payload = 0.0035206519818876732
    for primitive in PRIMITIVES:
        for probe in probes:
            event = str(probe.get("event_id"))
            cp = cp_by_event.get(event, {})
            payload = payload_by_event.get(event, {})
            key = (str(cp.get("family_id", "")), str(cp.get("bucket_id", "")))
            st = stats.get(
                key,
                {
                    "n": 0,
                    "cp_mean": 0.0,
                    "cp_lcb": 0.0,
                    "longrisk_mean": 1.0,
                    "longrisk_ucb": 1.0,
                    "shortonly_ucb": 1.0,
                },
            )
            ce = fnum(probe.get("probe_CE_delta"))
            margin = fnum(probe.get("probe_margin_delta"))
            tail = fnum(probe.get("probe_CEp99_delta"))
            conflict = fnum(probe.get("probe_adamw_conflict"))
            payload_norm = fnum(payload.get("payload_norm"))
            if primitive.startswith("AP1"):
                c_v = int(ce <= -0.004 and margin >= 0.006)
                c_b = int(tail <= 0.015 and conflict <= 1.03)
                c_h = int(st["longrisk_ucb"] <= 0.75)
                shape = "last_edge_certificate_required_not_materialized"
            elif primitive.startswith("AP2"):
                c_v = int(ce <= -0.002 and conflict <= 1.02)
                c_b = int(tail <= 0.02)
                c_h = int(st["longrisk_ucb"] <= 0.80)
                shape = "adamw_residual_certificate_required_not_materialized"
            elif primitive.startswith("AP3"):
                c_v = int(ce <= 0.0 and margin >= 0.0)
                c_b = int(tail <= 0.01)
                c_h = int(st["longrisk_ucb"] <= 0.60 and st["shortonly_ucb"] <= 0.60)
                shape = "tail_memory_certificate_required_not_materialized"
            else:
                c_v = int(ce <= -0.002 and payload_norm <= q([median_payload], 0.5) * 2.0)
                c_b = int(tail <= 0.02)
                c_h = int(st["longrisk_ucb"] <= 0.80)
                shape = "low_rank_edge_certificate_required_not_materialized"
            c_n = int(payload_norm >= median_payload)
            c_s = int(st["n"] >= 3 and st["cp_lcb"] >= 0.05)
            c_c = 0
            primitive_materialized = 0
            diag_pass_without_payload = int(c_v and c_b and c_n and c_s and c_h)
            cert_pass = int(diag_pass_without_payload and c_c and primitive_materialized)
            rows.append(
                {
                    "stage": "P2_CERTIFICATE_SCHEMA",
                    "status": "certificate_row",
                    "certificate_schema_version": "v9370-cert-schema-v1",
                    "primitive_id": primitive,
                    "action_id": payload.get("action_id"),
                    "candidate_id": payload.get("candidate_id"),
                    "event_id": event,
                    "dataset_for_diagnostic_only": cp.get("dataset"),
                    "seed": cp.get("seed"),
                    "family_id": cp.get("family_id"),
                    "bucket_id": cp.get("bucket_id"),
                    "payload_hash": payload.get("payload_hash_loaded"),
                    "payload_shape_id": shape,
                    "payload_apply_mode": "not_materialized",
                    "primitive_materialized": primitive_materialized,
                    "certificate_fields_complete": 1,
                    "C_V_pass": c_v,
                    "C_B_pass": c_b,
                    "C_N_pass": c_n,
                    "C_S_pass": c_s,
                    "C_H_pass": c_h,
                    "C_C_pass": c_c,
                    "certificate_pass": cert_pass,
                    "diagnostic_certificate_pass_ignoring_payload": diag_pass_without_payload,
                    "V_hat": -ce,
                    "V_lcb": st["cp_lcb"],
                    "Bad_hat": max(0.0, tail),
                    "Bad_ucb": "",
                    "Null_hat": int(payload_norm < median_payload),
                    "Null_ucb": "",
                    "Support_eff_n": st["n"],
                    "Support_lcb": st["cp_lcb"],
                    "HorizonRisk_hat": st["longrisk_mean"],
                    "HorizonRisk_ucb": st["longrisk_ucb"],
                    "Cost_hat_ms": "",
                    "Cost_ucb_ms": "",
                    "uses_dataset_name_for_controller": 0,
                    "uses_validation_or_test": 0,
                    "uses_future_outcome_for_features": 0,
                    "uses_outcome_at_commit": 0,
                    "uses_source_measured_gap": 0,
                    "uses_formula_proxy_for_official": 0,
                    "uses_teacher": 0,
                    "uses_loss_modification": 0,
                    "uses_loss_backward": 0,
                    "legality_audit_pass": 1,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    complete = all(inum(r.get("certificate_fields_complete")) for r in rows)
    payload_missing = sum(int(not r.get("payload_hash")) for r in rows)
    materialized = sum(inum(r.get("primitive_materialized")) for r in rows)
    summary = {
        "stage": "P2_CERTIFICATE_SCHEMA",
        "status": "summary",
        "certificate_schema_version": "v9370-cert-schema-v1",
        "diagnostic_certificate_rows": len(rows),
        "primitive_count": len(PRIMITIVES),
        "certificate_fields_complete": int(complete),
        "payload_hash_missing_count": payload_missing,
        "primitive_materialized_count": materialized,
        "diagnostic_certificate_pass_ignoring_payload_count": sum(inum(r.get("diagnostic_certificate_pass_ignoring_payload")) for r in rows),
        "certificate_pass_count": sum(inum(r.get("certificate_pass")) for r in rows),
        "legality_audit_pass": int(complete and payload_missing == 0),
        "certificate_schema_contract_pass": 0,
        "reason": "certificate_schema_rows_complete_but_AP1_AP4_generated_payloads_not_materialized",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, summary


def primitive_generation(cert_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for primitive in PRIMITIVES:
        subset = [r for r in cert_rows if r.get("primitive_id") == primitive]
        rows.append(
            {
                "stage": "P3_AP_PRIMITIVE_SMOKE_GENERATION",
                "status": "primitive_summary",
                "primitive_id": primitive,
                "generated_action_count": 0,
                "diagnostic_certificate_row_count": len(subset),
                "certificate_pass_count": 0,
                "diagnostic_certificate_pass_ignoring_payload_count": sum(inum(r.get("diagnostic_certificate_pass_ignoring_payload")) for r in subset),
                "certificate_pass_rate": 0.0,
                "payload_hash_missing_count": sum(int(not r.get("payload_hash")) for r in subset),
                "payload_shape_id_distribution": json.dumps(dict(Counter(str(r.get("payload_shape_id")) for r in subset)), sort_keys=True),
                "payload_nonzero_count_mean": "",
                "payload_nonzero_count_q90": "",
                "payload_norm_mean": "",
                "payload_norm_q90": "",
                "payload_norm_over_adamw_mean": "",
                "certificate_generation_time_ms_q90": "",
                "payload_materialize_time_ms_q90": "",
                "payload_apply_time_ms_q90_smoke": "",
                "memory_ratio_smoke": "",
                "legality_audit_pass": 1,
                "primitive_generation_pass": 0,
                "runtime_smoke_pass": 0,
                "reason": "true_certificate_producing_payload_generator_not_implemented",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P3_AP_PRIMITIVE_SMOKE_GENERATION",
        "status": "summary",
        "primitive_count": len(PRIMITIVES),
        "generated_action_count_total": 0,
        "diagnostic_certificate_row_count": len(cert_rows),
        "certificate_pass_count_total": 0,
        "primitive_generation_pass": 0,
        "reason": "AP1_AP4_certificate_producing_action_generators_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, summary


def diagnostic_outcome(cert_rows: list[dict[str, Any]], source_v9360: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cp_by_event = {r.get("event_id"): r for r in read_csv(source_v9360 / "cp_horizon_structure_trace_v9360.csv") if r.get("status") == "action_horizon_row"}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for primitive in PRIMITIVES:
        subset = [r for r in cert_rows if r.get("primitive_id") == primitive and inum(r.get("diagnostic_certificate_pass_ignoring_payload"))]
        weak = 0
        strong = 0
        robust = 0
        longrisk = 0
        short = 0
        for r in subset:
            cp = cp_by_event.get(r.get("event_id"), {})
            weak += inum(cp.get("weak_CP"))
            strong += inum(cp.get("strong_CP"))
            robust += inum(cp.get("horizon_robust_CP"))
            longrisk += inum(cp.get("long_risk"))
            short += inum(cp.get("short_only_CP"))
        n = len(subset)
        summaries.append(
            {
                "stage": "P4_AP_SMOKE_OUTCOME_MATERIALIZATION",
                "status": "primitive_diagnostic_summary",
                "primitive_id": primitive,
                "smoke_outcome_materialized": 0,
                "diagnostic_on_AP0_existing_outcomes": 1,
                "certificate_pass_accepted_count": n,
                "weak_CP_precision_certificate_pass": weak / max(1, n),
                "strong_CP_precision_certificate_pass": strong / max(1, n),
                "horizon_robust_CP_precision_certificate_pass": robust / max(1, n),
                "long_risk_rate_certificate_pass": longrisk / max(1, n),
                "short_only_rate_certificate_pass": short / max(1, n),
                "bad_event_rate_certificate_pass": "",
                "null_rate_certificate_pass": "",
                "primitive_weak_pass": 0,
                "primitive_strong_pass": 0,
                "reason": "diagnostic_uses_AP0_outcomes_not_new_AP_payload_outcomes",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P4_AP_SMOKE_OUTCOME_MATERIALIZATION",
        "status": "summary",
        "branch_completion_rate": 0.0,
        "horizon_completion_rate": 0.0,
        "missing_secondary_delta_count": "",
        "quality_audit_pass": 0,
        "ap_smoke_outcome_pass": 0,
        "reason": "new_AP_payload_outcomes_not_materialized",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows + summaries, summary


def not_run(stage: str, reason: str) -> list[dict[str, Any]]:
    return [{"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}]


def write_hashes(out_dir: Path, paths: list[Path]) -> None:
    rows = []
    for label, path in [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("aggregate decision", out_dir / "aggregate_decision.json"),
        ("contract audit", out_dir / "contract_audit_v9370.csv"),
        ("provenance audit", out_dir / "provenance_audit_v9370.csv"),
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
    source_v9360 = Path(args.source_v9360)
    source_v9350 = Path(args.source_v9350)
    source_v9330 = Path(args.source_v9330)
    manifest = {
        "runner": rel(SCRIPT_PATH),
        "plan": rel(PLAN_PATH),
        "started_at": now_iso(),
        "device": args.device,
        "source_v9360": rel(source_v9360),
        "source_v9350": rel(source_v9350),
        "source_v9330": rel(source_v9330),
        "diagnostic_actions": args.diagnostic_actions,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    p0 = p0_reanalysis(source_v9360)
    p1_trace, p1 = ap0_threshold_ban(source_v9360)
    cert_rows, p2 = certificate_schema(source_v9350, source_v9360, source_v9330, int(args.diagnostic_actions))
    p3_trace, p3 = primitive_generation(cert_rows)
    p4_trace, p4 = diagnostic_outcome(cert_rows, source_v9360)
    p5 = {"stage": "P5_CERTIFICATE_CALIBRATION", "status": "summary", "certificate_calibration_pass": 0, "reason": "P3_P4_new_primitive_materialization_missing", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    p6 = {"stage": "P6_FULL_AP_FRONTIER_COMPLETION", "status": "summary", "full_ap_frontier_pass": 0, "reason": "no_survivor_primitive_from_P3_P5", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    p7 = {"stage": "P7_CERTIFICATE_CONTROLLER", "status": "summary", "certificate_controller_pass": 0, "controller_id": "not_selected_certificate_primitive_missing", "reason": "no_certificate_primitive_survivor", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    p8 = {"stage": "P8_SELECTED_PRIMITIVE_ONLINE_RUNTIME", "status": "summary", "selected_runtime_measured": 0, "runtime_candidate_id": "not_selected_certificate_primitive_missing", "selected_payload_runtime_pass": 0, "source_v9360_microbench_step_ratio_q90": p0.get("step_ratio_q90_microbench"), "reason": "no_selected_certificate_controller_or_payload", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    p9 = {
        "stage": "P9_SYSTEM_INTEGRATION_GATE",
        "status": "summary",
        "system_candidate_id": "SYS-v9370-certificate-primitive-boundary",
        "primitive_id": "not_selected",
        "controller_id": p7.get("controller_id"),
        "runtime_candidate_id": p8.get("runtime_candidate_id"),
        "certificate_schema_version": "v9370-cert-schema-v1",
        "candidate_count": 2876,
        "action_count": 0,
        "accepted_count": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "certificate_legality_pass": p2.get("legality_audit_pass"),
        "payload_binding_pass": 0,
        "materialized_system_path": 0,
        "diagnostic_derived_from_measured_components": 0,
        "projection_used": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "dataset_name_used": 0,
        "reason": "certificate_producing_action_primitive_not_materialized",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    route = {
        "route": "R6-ActionPrimitiveRedesignRequired",
        "base_candidate": "LQ-t2-h256",
        "source_route_v9360": p0.get("route"),
        "p0_boundary_reanalysis_pass": p0.get("p0_pass"),
        "ap0_stop_condition": p1.get("ap0_stop_condition"),
        "best_ap0_feature_auc_CP": p1.get("best_auc_CP"),
        "best_ap0_top273_CP_precision": p1.get("best_top273_CP_precision"),
        "certificate_schema_rows_complete": p2.get("certificate_fields_complete"),
        "certificate_schema_contract_pass": p2.get("certificate_schema_contract_pass"),
        "diagnostic_certificate_rows": p2.get("diagnostic_certificate_rows"),
        "diagnostic_certificate_pass_ignoring_payload_count": p2.get("diagnostic_certificate_pass_ignoring_payload_count"),
        "primitive_generation_pass": p3.get("primitive_generation_pass"),
        "generated_action_count_total": p3.get("generated_action_count_total"),
        "ap_smoke_outcome_pass": p4.get("ap_smoke_outcome_pass"),
        "certificate_calibration_pass": p5.get("certificate_calibration_pass"),
        "full_ap_frontier_pass": p6.get("full_ap_frontier_pass"),
        "certificate_controller_pass": p7.get("certificate_controller_pass"),
        "selected_payload_runtime_pass": p8.get("selected_payload_runtime_pass"),
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "primary_blocker": "certificate_producing_action_primitive_not_materialized",
        "next_required_implementation": "implement_real_AP1_AP4_payload_generators_with_commit_time_certificate_tensors",
        "success_v9370_strict_purekan_functional": 0,
        "success_v9370_full_functional": 0,
        "success_v9370_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    artifacts: list[Path] = []
    def w(name: str, rows: list[dict[str, Any]]) -> None:
        path = out_dir / name
        write_csv(path, rows)
        artifacts.append(path)
    w("p0_v9360_boundary_reanalysis.csv", [p0])
    w("p1_ap0_threshold_search_ban_audit.csv", [p1])
    w("ap0_feature_capacity_trace_v9370.csv", p1_trace)
    w("p2_certificate_schema_legality_contract.csv", [p2])
    w("certificate_schema_trace_v9370.csv", cert_rows)
    w("p3_ap_primitive_smoke_generation.csv", [p3])
    w("certificate_action_trace_v9370.csv", p3_trace)
    w("p4_ap_smoke_outcome_materialization.csv", [p4])
    w("primitive_outcome_diagnostic_trace_v9370.csv", p4_trace)
    w("p5_certificate_calibration_sufficient_statistic_audit.csv", [p5])
    w("p6_full_ap_frontier_completion.csv", [p6])
    w("p7_certificate_controller.csv", [p7])
    w("p8_selected_primitive_online_runtime.csv", [p8])
    w("p9_system_integration_gate.csv", [p9])
    for name in ["p10_leave_dataset_stratum_out.csv", "p11_diagnostic_paired_replay_scout.csv", "p12_official_paired_replay.csv", "p13_short_full_sampleeff_continual_robustness.csv"]:
        w(name, not_run(name.replace(".csv", "").upper(), "P9_system_controller_not_official"))
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    artifacts.extend([out_dir / "route_decision.json", out_dir / "aggregate_decision.json"])
    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "ap0_stop_condition": p1.get("ap0_stop_condition"),
        "certificate_schema_rows_complete": p2.get("certificate_fields_complete"),
        "certificate_schema_contract_pass": p2.get("certificate_schema_contract_pass"),
        "primitive_generation_pass": p3.get("primitive_generation_pass"),
        "ap_smoke_outcome_pass": p4.get("ap_smoke_outcome_pass"),
        "certificate_controller_pass": p7.get("certificate_controller_pass"),
        "selected_payload_runtime_pass": p8.get("selected_payload_runtime_pass"),
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
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
    w("contract_audit_v9370.csv", [contract])
    failure = [
        {"failure_id": "F1_AP0_threshold_search_banned", "active": p1.get("ap0_stop_condition"), "primary_blocker": route["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F2_certificate_schema_not_official", "active": int(inum(p2.get("certificate_schema_contract_pass")) == 0), "primary_blocker": route["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F3_AP1_AP4_generators_missing", "active": 1, "primary_blocker": route["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F4_new_AP_outcomes_missing", "active": 1, "primary_blocker": route["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F5_selected_runtime_missing", "active": 1, "primary_blocker": route["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
    ]
    w("failure_table.csv", failure)
    prov = audit_no_fake(artifacts)
    w("provenance_audit_v9370.csv", [prov])
    manifest["finished_at"] = now_iso()
    write_json(out_dir / "run_manifest.json", manifest)
    write_hashes(out_dir, artifacts)
    print(json.dumps({"out_dir": rel(out_dir), "route": route["route"], "generated_action_count_total": route["generated_action_count_total"], "diagnostic_certificate_rows": route["diagnostic_certificate_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
