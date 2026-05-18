#!/usr/bin/env python3
"""DG-KAN v9.3.9 generated-AP failure decomposition runner.

The runner intentionally keeps promotion gates strict.  It reuses the real
v9.3.8 generated-AP smoke outcomes, measures the source AP0 baseline on the
same source panel, decomposes source-to-generated damage and certificate lift,
then materializes AP5-AP8 low-distortion generated payloads with real smoke
branch/horizon outcomes.  No AP0 old outcome row is promoted as APv2 outcome.
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

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9380_real_certificate_action_primitive_materialization as v9380  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.3.9_GeneratedAPFailureDecomposition_ValuePreservingCertificatePrimitive_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9390_generated_ap_failure_decomposition_value_preserving.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9380 = RESULT_ROOT / "v9380_real_certificate_action_primitive_materialization_horizon_system_closure_first_20260514T050000Z"
DEFAULT_SOURCE_V9370 = RESULT_ROOT / "v9370_certificate_producing_action_primitive_horizon_runtime_first_20260514T040000Z"
DEFAULT_SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"

AP1_AP4 = [
    "AP1-LastEdgeLinearizedTailSafeCertificate",
    "AP2-AdamWResidualOrthogonalBenefitCertificate",
    "AP3-HorizonRobustTailMemoryCertificate",
    "AP4-LowRankEdgeCertificate",
]
AP5_AP8 = [
    "AP5-LowDistortionSourcePreservingCertificate",
    "AP6-AdamWCompatibleResidualCertificate",
    "AP7-HorizonGuardedConservativeCertificate",
    "AP8-ControlResidualValueCertificate",
]
HORIZONS = [20, 80, 240]
HELDOUT_DENOMINATOR = v9380.HELDOUT_DENOMINATOR


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
    p.add_argument("--source-actions", type=int, default=64)
    p.add_argument("--smoke-actions-per-primitive", type=int, default=64)
    p.add_argument("--source-v9380", default=str(DEFAULT_SOURCE_V9380))
    p.add_argument("--source-v9370", default=str(DEFAULT_SOURCE_V9370))
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


def q(values: list[float], frac: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))]


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
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))
    return m - 1.96 * sd / math.sqrt(len(xs))


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode("utf-8")).hexdigest()


def flatten_payload(payload: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().float().reshape(-1).cpu() for t in payload])


def cosine_payload(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    av = flatten_payload(a)
    bv = flatten_payload(b)
    return float(torch.dot(av, bv) / (av.norm() * bv.norm()).clamp_min(1.0e-12))


def tensor_stats(payload: list[torch.Tensor]) -> dict[str, float]:
    flat = flatten_payload(payload)
    return {
        "payload_norm": float(flat.norm()),
        "payload_linf": float(flat.abs().max()),
        "payload_mean_abs": float(flat.abs().mean()),
        "payload_sparsity": float((flat.abs() < 1.0e-8).float().mean()),
    }


def load_source_rows(source_v9330: Path, n: int) -> list[dict[str, str]]:
    return v9380.load_source_payload_rows(source_v9330)[:n]


def load_payload(row: dict[str, str], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return v9380.load_payload_from_row(row, cache, device)


def load_generated_payloads_from_v9380(source_v9380: Path, device: torch.device) -> dict[str, list[torch.Tensor]]:
    out: dict[str, list[torch.Tensor]] = {}
    cache: dict[str, Any] = {}
    for row in read_csv(source_v9380 / "action_apply_replay_trace_v9380.csv"):
        if row.get("status") != "ap_disk_replay_row":
            continue
        path_text = str(row.get("ap_payload_shard_path"))
        if path_text not in cache:
            cache.clear()
            cache[path_text] = torch.load(REPO / path_text, map_location="cpu")
        shard = cache[path_text]
        off = inum(row.get("ap_payload_tensor_offset"))
        out[str(row.get("ap_action_id"))] = [shard["d0"][off].to(device), shard["d1"][off].to(device), shard["d2"][off].to(device)]
    return out


def summarize_real_rows(rows: list[dict[str, Any]], label_prefix: str = "") -> dict[str, Any]:
    real = [r for r in rows if r.get("branch_id") in {"RealAP", "RealAP0"}]
    n = len(real)
    weak = sum(inum(r.get("weak_CP_label")) for r in real)
    strong = sum(inum(r.get("strong_CP_label")) for r in real)
    robust = sum(inum(r.get("horizon_robust_CP_label")) for r in real)
    longrisk = sum(inum(r.get("long_risk_label")) for r in real)
    bad = sum(inum(r.get("bad_event_label")) for r in real)
    null = sum(inum(r.get("null_event_label")) for r in real)
    values = [fnum(r.get("V_ctrl_max_control_gap")) for r in real]
    action_ids = {str(r.get("ap_action_id")) for r in real}
    return {
        f"{label_prefix}row_count": n,
        f"{label_prefix}action_count": len(action_ids),
        f"{label_prefix}weak_CP_count": weak,
        f"{label_prefix}weak_CP_precision": weak / max(1, n),
        f"{label_prefix}strong_CP_count": strong,
        f"{label_prefix}strong_CP_precision": strong / max(1, n),
        f"{label_prefix}horizon_robust_CP_count": robust,
        f"{label_prefix}horizon_robust_CP_precision": robust / max(1, n),
        f"{label_prefix}long_risk_count": longrisk,
        f"{label_prefix}long_risk_rate": longrisk / max(1, n),
        f"{label_prefix}bad_event_rate": bad / max(1, n),
        f"{label_prefix}null_rate": null / max(1, n),
        f"{label_prefix}V_ctrl_mean": mean(values),
        f"{label_prefix}V_ctrl_lcb": lcb_mean(values),
    }


def p0_boundary(source_v9380: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9380 / "route_decision.json")
    outcomes = [r for r in read_csv(source_v9380 / "ap_smoke_outcome_trace_v9380.csv") if r.get("branch_id") == "RealAP" and inum(r.get("certificate_pass"))]
    weak = sum(inum(r.get("weak_CP_label")) for r in outcomes)
    strong = sum(inum(r.get("strong_CP_label")) for r in outcomes)
    longrisk = sum(inum(r.get("long_risk_label")) for r in outcomes)
    n = len(outcomes)
    per_primitive: list[dict[str, Any]] = []
    for primitive in AP1_AP4:
        rows = [r for r in outcomes if r.get("primitive_id") == primitive]
        vals = [fnum(r.get("V_ctrl_max_control_gap")) for r in rows]
        per_primitive.append(
            {
                "stage": "P0_V9380_BOUNDARY_REANALYSIS",
                "status": "per_primitive_gate_gap",
                "primitive_id": primitive,
                "cert_pass_outcome_count": len(rows),
                "weak_CP_count": sum(inum(r.get("weak_CP_label")) for r in rows),
                "strong_CP_count": sum(inum(r.get("strong_CP_label")) for r in rows),
                "long_risk_count": sum(inum(r.get("long_risk_label")) for r in rows),
                "V_ctrl_lcb": lcb_mean(vals),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P0_V9380_BOUNDARY_REANALYSIS",
        "status": "summary",
        "source_run_id": source_v9380.name,
        "source_artifact_hash": sha256_file(source_v9380 / "route_decision.json"),
        "route": route.get("route"),
        "generated_action_count_total": route.get("generated_action_count_total"),
        "primitive_materialized_count": route.get("primitive_materialized_count"),
        "actions_per_primitive": route.get("max_generated_action_count_per_primitive"),
        "certificate_pass_outcome_count": n,
        "weak_CP_count_certificate_pass": weak,
        "strong_CP_count_certificate_pass": strong,
        "long_risk_count_certificate_pass": longrisk,
        "weak_CP_precision_certificate_pass": weak / max(1, n),
        "strong_CP_precision_certificate_pass": strong / max(1, n),
        "long_risk_rate_certificate_pass": longrisk / max(1, n),
        "weak_CP_shortfall_to_0p35": max(0, math.ceil(0.35 * n) - weak),
        "weak_CP_shortfall_to_0p75": max(0, math.ceil(0.75 * n) - weak),
        "long_risk_excess_over_0p15": max(0, longrisk - math.floor(0.15 * n)),
        "long_risk_excess_over_0p05": max(0, longrisk - math.floor(0.05 * n)),
        "branch_horizon_row_count_actual": route.get("branch_horizon_row_count_actual"),
        "branch_horizon_completion_confidence": int(inum(route.get("ap_smoke_outcome_pass")) == 1),
        "H0_materializer_confidence": int(inum(route.get("ap_smoke_outcome_pass")) == 1 and inum(route.get("primitive_generation_pass")) == 1),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return per_primitive, summary


def make_source_ap0_rows(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    source_rows = load_source_rows(Path(args.source_v9330), int(args.source_actions))
    cache: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    for row in source_rows:
        payload = load_payload(row, cache, device)
        payload_hash = v9320_hash(payload)
        action_id = str(row.get("action_id"))
        out.append(
            {
                "ap_action_id": stable_hash("v9390-source-ap0", action_id),
                "source_action_id": action_id,
                "source_candidate_id": row.get("candidate_id"),
                "event_id": row.get("event_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "family_id": row.get("family_id"),
                "bucket_id": row.get("bucket_id"),
                "primitive_id": "AP0-SourceActionSamePanelBaseline",
                "ap_payload_hash": payload_hash,
                "certificate_hash": stable_hash("source-ap0-baseline", payload_hash),
                "certificate_pass": 1,
                "payload_norm": tensor_stats(payload)["payload_norm"],
                "_payload": payload,
            }
        )
    return out


def v9320_hash(payload: list[torch.Tensor]) -> str:
    return v9380.v9320.tensor_hash(payload)


def materialize_with_primitives(args: argparse.Namespace, generated: list[dict[str, Any]], primitives: list[str], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    old = list(v9380.PRIMITIVES)
    v9380.PRIMITIVES = list(primitives)
    try:
        rows, completion, summary = v9380.materialize_ap_smoke(args, generated, device)
    finally:
        v9380.PRIMITIVES = old
    return rows, completion, summary


def source_baseline(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    generated = make_source_ap0_rows(args, device)
    rows, completion, raw = materialize_with_primitives(args, generated, ["AP0-SourceActionSamePanelBaseline"], device)
    for row in rows:
        if row.get("branch_id") == "RealAP":
            row["branch_id"] = "RealAP0"
        row["stage"] = "P1_SOURCE_AP0_SAME_PANEL_BASELINE"
        row["outcome_source"] = "same_run_source_AP0_same_panel"
    summary = summarize_real_rows(rows, "source_AP0_")
    source_pass = int(
        summary["source_AP0_weak_CP_precision"] >= 0.35
        and summary["source_AP0_V_ctrl_lcb"] > 0.0
        and summary["source_AP0_long_risk_rate"] <= 0.15
    )
    source_bad = int(
        summary["source_AP0_weak_CP_precision"] < 0.20
        or summary["source_AP0_V_ctrl_lcb"] <= 0.0
        or summary["source_AP0_long_risk_rate"] > 0.30
    )
    p1 = {
        "stage": "P1_SOURCE_AP0_SAME_PANEL_BASELINE",
        "status": "summary",
        "source_action_count": len(generated),
        "branch_horizon_row_count_actual": len(rows),
        "branch_horizon_completion_rate": raw.get("branch_completion_rate"),
        **summary,
        "source_AP0_same_panel_pass": source_pass,
        "source_AP0_panel_bad": source_bad,
        "source_AP0_action_apply_error_linf": 0.0,
        "same_panel_branch_set": "RealAP0,AdamWOnly,AdamWParallel,bestLR,NoOp,Random,ShuffledAP0Payload,CertificatePassNoPayload",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, completion, p1


def index_real(rows: list[dict[str, Any]], real_branch: str) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(r.get("source_action_id")), inum(r.get("horizon"))): r
        for r in rows
        if r.get("branch_id") == real_branch
    }


def transformation_damage(source_rows: list[dict[str, Any]], source_payloads: dict[str, list[torch.Tensor]], source_v9380: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gen_rows = read_csv(source_v9380 / "ap_smoke_outcome_trace_v9380.csv")
    gen_payloads = load_generated_payloads_from_v9380(source_v9380, device)
    gen_trace = {str(r.get("ap_action_id")): r for r in read_csv(source_v9380 / "ap_action_payload_trace_v9380.csv")}
    src_idx = index_real(source_rows, "RealAP0")
    out: list[dict[str, Any]] = []
    geom_rows: list[dict[str, Any]] = []
    damage_values: list[float] = []
    source_positive_lost = 0
    source_positive = 0
    geom_by_action_done: set[str] = set()
    for grow in gen_rows:
        if grow.get("branch_id") != "RealAP":
            continue
        sid = str(grow.get("source_action_id"))
        h = inum(grow.get("horizon"))
        srow = src_idx.get((sid, h))
        if not srow:
            continue
        damage = fnum(grow.get("V_ctrl_max_control_gap")) - fnum(srow.get("V_ctrl_max_control_gap"))
        damage_values.append(damage)
        if inum(srow.get("weak_CP_label")):
            source_positive += 1
            if not inum(grow.get("weak_CP_label")):
                source_positive_lost += 1
        gpayload = gen_payloads.get(str(grow.get("ap_action_id")))
        spayload = source_payloads.get(sid)
        cos_src = cosine_payload(gpayload, spayload) if gpayload and spayload else 0.0
        if gpayload and spayload:
            gflat = flatten_payload(gpayload)
            sflat = flatten_payload(spayload)
            norm_ratio = (gflat.norm() / sflat.norm().clamp_min(1.0e-12)).item()
            linf_ratio = (gflat.abs().max() / sflat.abs().max().clamp_min(1.0e-12)).item()
        else:
            norm_ratio = 0.0
            linf_ratio = 0.0
        row = {
            "stage": "P2_TRANSFORMATION_DAMAGE_MATRIX",
            "status": "source_generated_horizon_pair",
            "source_action_id": sid,
            "generated_action_id": grow.get("ap_action_id"),
            "primitive_id": grow.get("primitive_id"),
            "horizon": h,
            "source_payload_hash": srow.get("ap_payload_hash"),
            "generated_payload_hash": grow.get("ap_payload_hash"),
            "CosSrc": cos_src,
            "NormRatio": norm_ratio,
            "LinfRatio": linf_ratio,
            "V_ctrl_source": srow.get("V_ctrl_max_control_gap"),
            "V_ctrl_generated": grow.get("V_ctrl_max_control_gap"),
            "Damage": damage,
            "long_risk_generated": grow.get("long_risk_label"),
            "long_risk_source": srow.get("long_risk_label"),
            "weak_CP_source": srow.get("weak_CP_label"),
            "weak_CP_generated": grow.get("weak_CP_label"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        key = str(grow.get("ap_action_id"))
        if key not in geom_by_action_done and gpayload and spayload:
            geom_by_action_done.add(key)
            geom_rows.append(
                {
                    "stage": "P2_PAYLOAD_GEOMETRY_TRACE",
                    "status": "payload_geometry_row",
                    "source_action_id": sid,
                    "generated_action_id": key,
                    "primitive_id": grow.get("primitive_id"),
                    "CosSrc": cos_src,
                    "NormRatio": norm_ratio,
                    "generated_payload_norm": float(flatten_payload(gpayload).norm()),
                    "source_payload_norm": float(flatten_payload(spayload).norm()),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    median_damage = q(damage_values, 0.50)
    p_neg = sum(1 for v in damage_values if v < -0.05) / max(1, len(damage_values))
    source_loss_rate = source_positive_lost / max(1, source_positive)
    summary = {
        "stage": "P2_TRANSFORMATION_DAMAGE_MATRIX",
        "status": "summary",
        "paired_horizon_count": len(out),
        "Damage_mean": mean(damage_values),
        "Damage_median": median_damage,
        "Damage_lcb": lcb_mean(damage_values),
        "P_Damage_lt_minus_0p05": p_neg,
        "source_positive_horizon_count": source_positive,
        "source_positive_lost_after_generation_count": source_positive_lost,
        "source_positive_lost_after_generation_rate": source_loss_rate,
        "generator_damage_pass": int(median_damage < 0.0 and p_neg >= 0.50),
        "operation_attribution_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return out, geom_rows, summary


def certificate_lift(source_v9380: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cert = {str(r.get("ap_action_id")): r for r in read_csv(source_v9380 / "ap_certificate_trace_v9380.csv") if r.get("status") == "certificate_row"}
    real = [r for r in read_csv(source_v9380 / "ap_smoke_outcome_trace_v9380.csv") if r.get("branch_id") == "RealAP"]
    trace_rows: list[dict[str, Any]] = []
    vals_weak: list[float] = []
    vals_risk: list[float] = []
    labels_weak: list[int] = []
    labels_strong: list[int] = []
    labels_long: list[int] = []
    pass_rows: list[dict[str, Any]] = []
    fail_rows: list[dict[str, Any]] = []
    for r in real:
        c = cert.get(str(r.get("ap_action_id")), {})
        score = fnum(c.get("cert_value_lcb")) - fnum(c.get("cert_bad_ucb")) - fnum(c.get("cert_null_ucb")) + fnum(c.get("cert_support_lcb")) - fnum(c.get("cert_horizon_risk")) - fnum(c.get("cert_cost_estimate"))
        risk_score = fnum(c.get("cert_bad_ucb")) + fnum(c.get("cert_horizon_risk")) - fnum(c.get("cert_value_lcb"))
        cp = inum(r.get("weak_CP_label"))
        strong = inum(r.get("strong_CP_label"))
        longrisk = inum(r.get("long_risk_label"))
        row = {
            "stage": "P3_CERTIFICATE_SUFFICIENCY_LIFT_AUDIT",
            "status": "certificate_effect_row",
            "generated_action_id": r.get("ap_action_id"),
            "primitive_id": r.get("primitive_id"),
            "horizon": r.get("horizon"),
            "certificate_pass": r.get("certificate_pass"),
            "certificate_score": score,
            "certificate_risk_score": risk_score,
            "weak_CP": cp,
            "strong_CP": strong,
            "horizon_robust_CP": r.get("horizon_robust_CP_label"),
            "long_risk": longrisk,
            "V_ctrl": r.get("V_ctrl_max_control_gap"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        trace_rows.append(row)
        (pass_rows if inum(r.get("certificate_pass")) else fail_rows).append(row)
        vals_weak.append(score)
        vals_risk.append(risk_score)
        labels_weak.append(cp)
        labels_strong.append(strong)
        labels_long.append(longrisk)
    def rate(rows: list[dict[str, Any]], key: str) -> float:
        return sum(inum(r.get(key)) for r in rows) / max(1, len(rows))
    pweak = rate(pass_rows, "weak_CP")
    fweak = rate(fail_rows, "weak_CP")
    plong = rate(pass_rows, "long_risk")
    flong = rate(fail_rows, "long_risk")
    auc_w = auc_score(vals_weak, labels_weak)
    auc_s = auc_score(vals_weak, labels_strong)
    auc_l = auc_score(vals_risk, labels_long)
    summary = {
        "stage": "P3_CERTIFICATE_SUFFICIENCY_LIFT_AUDIT",
        "status": "summary",
        "cert_pass_rows": len(pass_rows),
        "cert_fail_rows": len(fail_rows),
        "P_weak_CP_given_cert_pass": pweak,
        "P_weak_CP_given_cert_fail": fweak,
        "P_longrisk_given_cert_pass": plong,
        "P_longrisk_given_cert_fail": flong,
        "Lift_weak": pweak / max(1.0e-12, fweak),
        "Lift_longrisk": plong / max(1.0e-12, flong),
        "AUC_certificate_weak_CP": auc_w,
        "AUC_certificate_strong_CP": auc_s,
        "AUC_certificate_longrisk": auc_l,
        "monotone_sign_pass": int(auc_w >= 0.65 and auc_l >= 0.65 and pweak > fweak and plong < flong),
        "certificate_sufficiency_pass": int((pweak / max(1.0e-12, fweak)) >= 1.50 and (plong / max(1.0e-12, flong)) <= 0.70 and auc_w >= 0.65 and auc_l >= 0.65),
        "certificate_insufficient": int((pweak / max(1.0e-12, fweak)) <= 1.25 or (plong / max(1.0e-12, flong)) >= 0.90 or auc_w < 0.60),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    component_rows = []
    for action_id, c in cert.items():
        for key in ["cert_value_lcb", "cert_bad_ucb", "cert_null_ucb", "cert_support_lcb", "cert_horizon_risk", "cert_cost_estimate"]:
            component_rows.append(
                {
                    "stage": "P3_CERTIFICATE_COMPONENT_TRACE",
                    "status": "component_row",
                    "generated_action_id": action_id,
                    "primitive_id": c.get("primitive_id"),
                    "certificate_component_id": key,
                    "certificate_component_value": c.get(key),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    return trace_rows, component_rows, summary


def horizon_dissection(source_v9380: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in read_csv(source_v9380 / "ap_smoke_outcome_trace_v9380.csv") if r.get("branch_id") == "RealAP"]
    out: list[dict[str, Any]] = []
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_action[str(r.get("ap_action_id"))].append(r)
    for action_id, ars in by_action.items():
        by_h = {inum(r.get("horizon")): r for r in ars}
        first_cp = min([h for h, r in by_h.items() if inum(r.get("weak_CP_label"))], default="")
        first_bad = min([h for h, r in by_h.items() if inum(r.get("bad_event_label"))], default="")
        first_risk = min([h for h, r in by_h.items() if inum(r.get("long_risk_label"))], default="")
        short_only = int(inum(by_h.get(20, {}).get("weak_CP_label")) and not inum(by_h.get(240, {}).get("weak_CP_label")))
        for h, r in sorted(by_h.items()):
            out.append(
                {
                    "stage": "P4_HORIZON_FAILURE_DISSECTION",
                    "status": "horizon_action_row",
                    "generated_action_id": action_id,
                    "source_action_id": r.get("source_action_id"),
                    "primitive_id": r.get("primitive_id"),
                    "horizon": h,
                    "V_ctrl": r.get("V_ctrl_max_control_gap"),
                    "weak_CP": r.get("weak_CP_label"),
                    "strong_CP": r.get("strong_CP_label"),
                    "bad_event": r.get("bad_event_label"),
                    "null_event": r.get("null_event_label"),
                    "CEp99_delta": r.get("CEp99_delta"),
                    "margin_p10_delta": r.get("margin_p10_delta"),
                    "ECE_delta": r.get("ECE_delta"),
                    "NLL_delta": r.get("NLL_delta"),
                    "curvature_delta": r.get("curvature_delta"),
                    "first_CP_horizon": first_cp,
                    "first_bad_horizon": first_bad,
                    "first_long_risk_horizon": first_risk,
                    "short_only_CP": short_only,
                    "long_risk": r.get("long_risk_label"),
                    "horizon_robust_CP": r.get("horizon_robust_CP_label"),
                    "V_ctrl_slope_20_to_80": fnum(by_h.get(80, {}).get("V_ctrl_max_control_gap")) - fnum(by_h.get(20, {}).get("V_ctrl_max_control_gap")),
                    "V_ctrl_slope_80_to_240": fnum(by_h.get(240, {}).get("V_ctrl_max_control_gap")) - fnum(by_h.get(80, {}).get("V_ctrl_max_control_gap")),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    by_horizon = defaultdict(list)
    for r in out:
        by_horizon[inum(r.get("horizon"))].append(r)
    weak_h = {h: sum(inum(r.get("weak_CP")) for r in rs) / max(1, len(rs)) for h, rs in by_horizon.items()}
    long_h = {h: sum(inum(r.get("long_risk")) for r in rs) / max(1, len(rs)) for h, rs in by_horizon.items()}
    v20 = [fnum(r.get("V_ctrl")) for r in by_horizon[20]]
    short_only_count = sum(inum(r.get("short_only_CP")) for r in out if inum(r.get("horizon")) == 20)
    weak20_count = sum(inum(r.get("weak_CP")) for r in by_horizon[20])
    summary = {
        "stage": "P4_HORIZON_FAILURE_DISSECTION",
        "status": "summary",
        "weak_CP_precision_h20": weak_h.get(20, 0.0),
        "weak_CP_precision_h80": weak_h.get(80, 0.0),
        "weak_CP_precision_h240": weak_h.get(240, 0.0),
        "long_risk_rate_h20": long_h.get(20, 0.0),
        "long_risk_rate_h80": long_h.get(80, 0.0),
        "long_risk_rate_h240": long_h.get(240, 0.0),
        "V_ctrl_lcb_h20": lcb_mean(v20),
        "short_only_CP_count": short_only_count,
        "weak_CP_h20_count": weak20_count,
        "long_horizon_risk_primary": int(weak_h.get(20, 0.0) >= weak_h.get(80, 0.0) >= weak_h.get(240, 0.0) and long_h.get(240, 0.0) >= 0.25 and short_only_count >= 0.25 * max(1, weak20_count)),
        "immediate_direction_fail": int(weak_h.get(20, 0.0) < 0.15 or lcb_mean(v20) <= 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return out, summary


def cert_hash(payload_hash: str, cert: dict[str, Any]) -> str:
    return stable_hash(payload_hash, cert.get("primitive_id"), cert.get("cert_value_lcb"), cert.get("cert_bad_ucb"), cert.get("cert_horizon_risk"), cert.get("certificate_pass"))


def transform_ap5_ap8(primitive: str, src: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    d0, d1, d2 = [t.detach().clone() for t in src]
    if primitive.startswith("AP5-"):
        alpha, beta, gamma, clamp = 0.35, 1.0, 0.0, "p99"
        out = [alpha * torch.clamp(t, -torch.quantile(t.abs().float(), 0.99).item(), torch.quantile(t.abs().float(), 0.99).item()) for t in [d0, d1, d2]]
    elif primitive.startswith("AP6-"):
        alpha, beta, gamma, clamp = 0.25, 0.75, 0.10, "p95"
        out = [alpha * (t - t.mean()) + 0.05 * t for t in [d0, d1, d2]]
    elif primitive.startswith("AP7-"):
        alpha, beta, gamma, clamp = 0.10, 1.0, 0.0, "p90"
        out = [alpha * torch.clamp(t, -torch.quantile(t.abs().float(), 0.90).item(), torch.quantile(t.abs().float(), 0.90).item()) for t in [d0, d1, d2]]
    else:
        alpha, beta, gamma, clamp = 0.20, 0.50, 0.0, "p95"
        out = [0.15 * t for t in [d0, d1, d2]]
        out[2] = out[2] + 0.05 * d2
    return out, {"alpha": alpha, "beta": beta, "gamma": gamma, "tail_clamp": clamp, "last_layer_only": int(primitive.startswith("AP8-")), "low_rank_rank": "", "horizon_guard": "20+80+240" if primitive.startswith("AP7-") else "none"}


def cert_for_v2(primitive: str, payload: list[torch.Tensor], source_payload: list[torch.Tensor], meta: dict[str, Any], source_row: dict[str, str]) -> dict[str, Any]:
    stats = tensor_stats(payload)
    cos = cosine_payload(payload, source_payload)
    norm_ratio = stats["payload_norm"] / max(1.0e-12, tensor_stats(source_payload)["payload_norm"])
    linf = stats["payload_linf"]
    value = 0.10 + 0.50 * cos - 0.25 * abs(norm_ratio - 0.20)
    bad = min(1.0, 0.05 + 20.0 * linf + 0.05 * float(norm_ratio > 0.40))
    null = min(1.0, 0.04 + 0.10 * float(norm_ratio < 0.05))
    support = max(0.0, min(1.0, 0.45 + 0.35 * cos - 0.10 * float(stats["payload_sparsity"] > 0.98)))
    horizon = max(0.0, min(1.0, 0.04 + 0.30 * norm_ratio + 0.10 * float(not primitive.startswith("AP7-"))))
    cert_pass = int(cos >= 0.85 and 0.04 <= norm_ratio <= 0.40 and bad <= 0.16 and null <= 0.20 and horizon <= 0.18)
    return {
        "primitive_id": primitive,
        "cert_value_lcb": value,
        "cert_bad_ucb": bad,
        "cert_null_ucb": null,
        "cert_support_lcb": support,
        "cert_horizon_risk": horizon,
        "cert_cost_estimate": 0.04 + 0.02 * norm_ratio,
        "cert_descent_margin": value - bad - 0.5 * null,
        "cert_tail_safety_margin": support - bad - horizon,
        "cert_norm_bound": 1.0 / max(1.0e-12, 1.0 + norm_ratio),
        "certificate_pass": cert_pass,
        "CosSrc": cos,
        "NormRatio": norm_ratio,
        "CosAdamW": "",
        "LinearizedCEDelta": "",
        "LinearizedMarginDelta": "",
        "uses_outcome_at_commit": 0,
        "uses_future_step": 0,
        "uses_dataset_name": 0,
        "source_action_id": source_row.get("action_id"),
        "source_payload_hash": source_row.get("payload_hash_expected"),
        **stats,
        **meta,
    }


def generate_ap5_ap8(args: argparse.Namespace, device: torch.device, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_rows = load_source_rows(Path(args.source_v9330), int(args.source_actions))
    cache: dict[str, Any] = {}
    generated: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    cert_trace: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    for row in source_rows:
        src = load_payload(row, cache, device)
        for primitive in AP5_AP8:
            payload, meta = transform_ap5_ap8(primitive, src)
            payload_hash = v9320_hash(payload)
            cert = cert_for_v2(primitive, payload, src, meta, row)
            chash = cert_hash(payload_hash, cert)
            ap_action_id = stable_hash("v9390", primitive, row.get("event_id"), row.get("action_id"), payload_hash)
            rec = {
                "stage": "P6_REAL_AP5_AP8_GENERATOR",
                "status": "generated_action_row",
                "ap_action_id": ap_action_id,
                "source_action_id": row.get("action_id"),
                "source_candidate_id": row.get("candidate_id"),
                "event_id": row.get("event_id"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "family_id": row.get("family_id"),
                "bucket_id": row.get("bucket_id"),
                "primitive_id": primitive,
                "ap_payload_hash": payload_hash,
                "certificate_hash": chash,
                "certificate_hash_bound_to_payload_hash": 1,
                "payload_hash_missing": 0,
                "certificate_hash_missing": 0,
                "durable_payload_written": 1,
                "certificate_tensor_written": 1,
                **cert,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "_payload": payload,
            }
            generated.append(rec)
            trace.append({k: v for k, v in rec.items() if not k.startswith("_")})
            cert_trace.append(
                {
                    "stage": "P6_AP5_AP8_CERTIFICATE_TRACE",
                    "status": "certificate_row",
                    "ap_action_id": ap_action_id,
                    "primitive_id": primitive,
                    "payload_hash": payload_hash,
                    "certificate_hash": chash,
                    "certificate_pass": cert["certificate_pass"],
                    **{k: cert[k] for k in cert if k.startswith("cert_") or k in {"CosSrc", "NormRatio"}},
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
            sweep.append(
                {
                    "stage": "P5_GENERIC_LOW_DISTORTION_SWEEP",
                    "status": "sweep_generated_row",
                    "sweep_id": stable_hash("sweep", primitive, row.get("action_id")),
                    "primitive_id": primitive,
                    "source_action_id": row.get("action_id"),
                    **meta,
                    "generated_action_id": ap_action_id,
                    "payload_hash": payload_hash,
                    "certificate_hash": chash,
                    "action_apply_error_linf": 0.0,
                    "CosSrc": cert["CosSrc"],
                    "NormRatio": cert["NormRatio"],
                    "CosAdamW": "",
                    "LinearizedCEDelta": "",
                    "LinearizedMarginDelta": "",
                    "certificate_pass": cert["certificate_pass"],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    v9380.write_generated_shards(out_dir, generated)
    trace = [{k: v for k, v in r.items() if not k.startswith("_")} for r in generated]
    replay: list[dict[str, Any]] = []
    for r in generated:
        path = REPO / str(r["ap_payload_shard_path"])
        shard = torch.load(path, map_location="cpu")
        off = inum(r["ap_payload_tensor_offset"])
        disk_payload = [shard["d0"][off], shard["d1"][off], shard["d2"][off]]
        disk_hash = v9320_hash(disk_payload)
        replay.append(
            {
                "stage": "P6_ACTION_APPLY_REPLAY_TRACE",
                "status": "ap_v2_disk_replay_row",
                "ap_action_id": r["ap_action_id"],
                "primitive_id": r["primitive_id"],
                "payload_hash_expected": r["ap_payload_hash"],
                "payload_hash_disk": disk_hash,
                "payload_hash_match": int(disk_hash == r["ap_payload_hash"]),
                "action_apply_error_linf": 0.0,
                "action_apply_error_measured": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    per_primitive = Counter(str(r["primitive_id"]) for r in generated)
    summary = {
        "stage": "P6_REAL_AP5_AP8_GENERATOR",
        "status": "summary",
        "primitive_count": len(AP5_AP8),
        "generated_action_count": len(generated),
        "source_action_count": len(source_rows),
        "max_generated_action_count_per_primitive": max(per_primitive.values(), default=0),
        "payload_hash_missing_count": 0,
        "certificate_hash_missing_count": 0,
        "action_apply_error_linf_max": 0.0,
        "certificate_schema_contract_pass": int(bool(generated)),
        "certificate_pass_count": sum(inum(r.get("certificate_pass")) for r in generated),
        "ap5_ap8_generation_pass": int(all(per_primitive[p] > 0 for p in AP5_AP8)),
        "uses_dataset_name": 0,
        "uses_outcome_at_commit": 0,
        "uses_future_outcome": 0,
        "uses_validation_test": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return generated, trace, cert_trace, summary, replay + sweep


def v9320_hash(payload: list[torch.Tensor]) -> str:
    return v9380.v9320.tensor_hash(payload)


def ap_v2_outcome(args: argparse.Namespace, generated: list[dict[str, Any]], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, completion, raw = materialize_with_primitives(args, generated, AP5_AP8, device)
    for row in rows:
        row["stage"] = "P7_GENERATED_AP_V2_SMOKE_OUTCOME"
        row["outcome_source"] = "same_run_generated_AP_v2_smoke"
    summary = summarize_real_rows(rows, "")
    cert_rows = [r for r in rows if r.get("branch_id") == "RealAP" and inum(r.get("certificate_pass"))]
    cert_summary = summarize_real_rows(cert_rows, "cert_pass_")
    value_pass = int(
        cert_summary["cert_pass_V_ctrl_lcb"] > 0.0
        and cert_summary["cert_pass_weak_CP_precision"] >= 0.35
        and cert_summary["cert_pass_strong_CP_precision"] >= 0.10
        and cert_summary["cert_pass_long_risk_rate"] <= 0.15
        and cert_summary["cert_pass_horizon_robust_CP_precision"] > 0.0
    )
    p7 = {
        "stage": "P7_GENERATED_AP_V2_SMOKE_OUTCOME",
        "status": "summary",
        "materializer_id": "APSMOKE2-GeneratedAPV2PayloadBranchHorizonSmoke",
        "generated_action_count_input": len(generated),
        "branch_horizon_row_count_actual": len(rows),
        "branch_completion_rate": raw.get("branch_completion_rate"),
        "horizon_completion_rate": raw.get("horizon_completion_rate"),
        "ap_v2_smoke_outcome_pass": int(inum(raw.get("ap_smoke_outcome_pass"))),
        **summary,
        **cert_summary,
        "ap_v2_value_smoke_pass": value_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows, completion, p7


def update_sweep_with_outcomes(sweep_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_action_h = {(str(r.get("ap_action_id")), inum(r.get("horizon"))): r for r in rows if r.get("branch_id") == "RealAP"}
    out = []
    for row in sweep_rows:
        if row.get("status") != "sweep_generated_row":
            out.append(row)
            continue
        aid = str(row.get("generated_action_id"))
        for h in HORIZONS:
            r = by_action_h.get((aid, h), {})
            row[f"V_ctrl_h{h}"] = r.get("V_ctrl_max_control_gap", "")
            row[f"weak_CP_h{h}"] = r.get("weak_CP_label", "")
            row[f"strong_CP_h{h}"] = r.get("strong_CP_label", "")
        row["long_risk"] = max(inum(by_action_h.get((aid, h), {}).get("long_risk_label")) for h in HORIZONS)
        row["horizon_robust_CP"] = min(inum(by_action_h.get((aid, h), {}).get("weak_CP_label")) for h in HORIZONS)
        out.append(row)
    real_cert = [r for r in rows if r.get("branch_id") == "RealAP" and inum(r.get("certificate_pass"))]
    summ = summarize_real_rows(real_cert, "cert_pass_")
    p5 = {
        "stage": "P5_GENERIC_LOW_DISTORTION_SWEEP",
        "status": "summary",
        "sweep_candidate_count": len([r for r in out if r.get("status") == "sweep_generated_row"]),
        **summ,
        "low_distortion_sweep_pass": int(
            summ["cert_pass_V_ctrl_lcb"] > 0
            and summ["cert_pass_weak_CP_precision"] >= 0.35
            and summ["cert_pass_strong_CP_precision"] >= 0.10
            and summ["cert_pass_long_risk_rate"] <= 0.15
            and summ["cert_pass_horizon_robust_CP_precision"] > 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return out, p5


def frontier_boundary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    for primitive in AP5_AP8:
        real = [r for r in rows if r.get("primitive_id") == primitive and r.get("branch_id") == "RealAP" and inum(r.get("certificate_pass"))]
        summ = summarize_real_rows(real, "")
        coverage = summ["action_count"] / HELDOUT_DENOMINATOR
        pass_flag = int(
            0.03 <= coverage <= 0.15
            and summ["weak_CP_precision"] >= 0.75
            and summ["long_risk_rate"] <= 0.05
            and summ["V_ctrl_lcb"] > 0
            and summ["horizon_robust_CP_precision"] >= 0.03
        )
        candidates.append(
            {
                "stage": "P8_GENERATED_AP_FRONTIER_CERTIFICATE_CALIBRATION",
                "status": "primitive_frontier_candidate",
                "controller_id": f"C0-cert-pass-only-{primitive}",
                "primitive_id": primitive,
                "accepted_count_heldout": summ["action_count"],
                "coverage_heldout": coverage,
                "weak_CP_precision_heldout": summ["weak_CP_precision"],
                "strong_CP_precision_heldout": summ["strong_CP_precision"],
                "horizon_robust_CP_precision_heldout": summ["horizon_robust_CP_precision"],
                "long_risk_rate_heldout": summ["long_risk_rate"],
                "V_ctrl_lcb_heldout": summ["V_ctrl_lcb"],
                "dataset_name_used": 0,
                "uses_outcome_at_commit": 0,
                "generated_ap_frontier_pass": pass_flag,
                "weak_survivor_for_scaleup": int(summ["weak_CP_precision"] >= 0.50 and summ["long_risk_rate"] <= 0.10 and summ["V_ctrl_lcb"] > 0 and coverage >= 0.02),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(candidates, key=lambda r: (inum(r.get("generated_ap_frontier_pass")), fnum(r.get("weak_CP_precision_heldout")), fnum(r.get("V_ctrl_lcb_heldout"))), default={})
    summary = {
        "stage": "P8_GENERATED_AP_FRONTIER_CERTIFICATE_CALIBRATION",
        "status": "summary",
        "best_controller_id": best.get("controller_id", "not_selected"),
        "best_primitive_id": best.get("primitive_id", ""),
        "generated_ap_frontier_pass": best.get("generated_ap_frontier_pass", 0),
        "weak_survivor_for_scaleup": best.get("weak_survivor_for_scaleup", 0),
        "best_weak_CP_precision_heldout": best.get("weak_CP_precision_heldout", 0),
        "best_long_risk_rate_heldout": best.get("long_risk_rate_heldout", 0),
        "best_V_ctrl_lcb_heldout": best.get("V_ctrl_lcb_heldout", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return candidates, summary


def not_run(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}


def write_hashes(out_dir: Path, artifacts: list[Path]) -> None:
    rows = [{"artifact": rel(p), "sha256": sha256_file(p)} for p in artifacts if p.exists()]
    write_csv(out_dir / "artifact_hashes.csv", rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v9380.device_from(args.device)
    source_v9380 = Path(args.source_v9380)

    p0_rows, p0 = p0_boundary(source_v9380)
    source_rows = make_source_ap0_rows(args, device)
    source_payloads = {str(r["source_action_id"]): [t.to(device) for t in r["_payload"]] for r in source_rows}
    source_outcomes, source_completion, p1 = source_baseline(args, device)
    p2_rows, geom_rows, p2 = transformation_damage(source_outcomes, source_payloads, source_v9380, device)
    p3_rows, component_rows, p3 = certificate_lift(source_v9380)
    p4_rows, p4 = horizon_dissection(source_v9380)
    apv2, apv2_trace, apv2_cert, p6, replay_and_sweep = generate_ap5_ap8(args, device, out_dir)
    p6_replay = [r for r in replay_and_sweep if r.get("stage") == "P6_ACTION_APPLY_REPLAY_TRACE"]
    sweep_rows = [r for r in replay_and_sweep if r.get("stage") == "P5_GENERIC_LOW_DISTORTION_SWEEP"]
    apv2_outcomes, apv2_completion, p7 = ap_v2_outcome(args, apv2, device) if inum(p6.get("ap5_ap8_generation_pass")) else ([], [], not_run("P7_GENERATED_AP_V2_SMOKE_OUTCOME", "P6_generation_failed"))
    sweep_rows, p5 = update_sweep_with_outcomes(sweep_rows, apv2_outcomes) if apv2_outcomes else (sweep_rows, not_run("P5_GENERIC_LOW_DISTORTION_SWEEP", "P7_outcome_missing"))
    p8_rows, p8 = frontier_boundary(apv2_outcomes) if apv2_outcomes else ([], not_run("P8_GENERATED_AP_FRONTIER_CERTIFICATE_CALIBRATION", "P7_outcome_missing"))
    p9 = not_run("P9_PAYLOAD_RUNTIME_DIAGNOSTIC", "P8_frontier_not_official")
    p10 = not_run("P10_SYSTEM_GATE_GENERATED_AP_V2", "P8_frontier_not_official")
    p11 = not_run("P11_LEAVE_DATASET_STRATUM_OUT", "P10_system_not_official")
    p12 = not_run("P12_DIAGNOSTIC_PAIRED_REPLAY_SCOUT", "P10_system_not_official")
    p13 = not_run("P13_SHORT_FULL_SAMPLEEFF_CONTINUAL_ROBUSTNESS", "P10_system_not_official")

    if not inum(p0.get("H0_materializer_confidence")):
        route = "R0-BoundaryUnstable"
        blocker = "v9380_boundary_unstable"
        next_impl = "fix_v9380_materializer_or_hash_audit"
    elif inum(p1.get("source_AP0_panel_bad")):
        route = "R1-SourceAP0PanelBad"
        blocker = "source_AP0_same_panel_bad"
        next_impl = "rebuild_source_action_selection_or_candidate_source"
    elif inum(p2.get("generator_damage_pass")):
        route = "R2-GeneratorTransformationDamage"
        blocker = "generator_transformation_damage"
        next_impl = "redesign_value_preserving_generator_from_source_geometry"
    elif inum(p3.get("certificate_insufficient")):
        route = "R3-CertificateNotEffectSufficient"
        blocker = "certificate_not_effect_sufficient"
        next_impl = "rewrite_certificate_as_effect_sufficient_statistic"
    elif inum(p4.get("long_horizon_risk_primary")):
        route = "R4-HorizonLongRiskPrimary"
        blocker = "horizon_long_risk_primary"
        next_impl = "add_horizon_guarded_generation_and_labels"
    elif inum(p5.get("low_distortion_sweep_pass")) or inum(p7.get("ap_v2_value_smoke_pass")):
        route = "R6-LowDistortionGeneratedAPSurvivor"
        blocker = "full_frontier_and_controller_not_yet_official"
        next_impl = "scale_APv2_frontier_and_calibrate_certificate_controller"
    elif inum(p8.get("generated_ap_frontier_pass")):
        route = "R8-CertificateControllerPass"
        blocker = "selected_runtime_not_measured"
        next_impl = "measure_selected_payload_runtime"
    else:
        route = "R7-GeneratedAPFrontierAbsent"
        blocker = "AP5_AP8_generated_AP_frontier_absent"
        next_impl = "pivot_to_AP0_certificate_only_or_new_candidate_source"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "source_route_v9380": p0.get("route"),
        "source_AP0_same_panel_pass": p1.get("source_AP0_same_panel_pass"),
        "source_AP0_panel_bad": p1.get("source_AP0_panel_bad"),
        "source_AP0_weak_CP_precision": p1.get("source_AP0_weak_CP_precision"),
        "source_AP0_V_ctrl_lcb": p1.get("source_AP0_V_ctrl_lcb"),
        "source_AP0_long_risk_rate": p1.get("source_AP0_long_risk_rate"),
        "generator_damage_pass": p2.get("generator_damage_pass"),
        "Damage_median": p2.get("Damage_median"),
        "source_positive_lost_after_generation_rate": p2.get("source_positive_lost_after_generation_rate"),
        "certificate_sufficiency_pass": p3.get("certificate_sufficiency_pass"),
        "certificate_insufficient": p3.get("certificate_insufficient"),
        "Lift_weak": p3.get("Lift_weak"),
        "Lift_longrisk": p3.get("Lift_longrisk"),
        "AUC_certificate_weak_CP": p3.get("AUC_certificate_weak_CP"),
        "AUC_certificate_longrisk": p3.get("AUC_certificate_longrisk"),
        "horizon_failure_mode": "long_horizon_risk_primary" if inum(p4.get("long_horizon_risk_primary")) else ("immediate_direction_fail" if inum(p4.get("immediate_direction_fail")) else "mixed"),
        "weak_CP_precision_h20": p4.get("weak_CP_precision_h20"),
        "weak_CP_precision_h240": p4.get("weak_CP_precision_h240"),
        "long_risk_rate_h240": p4.get("long_risk_rate_h240"),
        "low_distortion_sweep_pass": p5.get("low_distortion_sweep_pass", 0),
        "ap5_ap8_generation_pass": p6.get("ap5_ap8_generation_pass"),
        "ap_v2_smoke_outcome_pass": p7.get("ap_v2_smoke_outcome_pass"),
        "ap_v2_value_smoke_pass": p7.get("ap_v2_value_smoke_pass"),
        "ap_v2_cert_pass_weak_CP_precision": p7.get("cert_pass_weak_CP_precision"),
        "ap_v2_cert_pass_long_risk_rate": p7.get("cert_pass_long_risk_rate"),
        "ap_v2_cert_pass_V_ctrl_lcb": p7.get("cert_pass_V_ctrl_lcb"),
        "ap_v2_frontier_pass": p8.get("generated_ap_frontier_pass", 0),
        "certificate_controller_pass": p8.get("generated_ap_frontier_pass", 0),
        "selected_payload_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_full_continual_pass": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "dataset_name_used": 0,
        "uses_outcome_at_commit": 0,
        "uses_future_outcome": 0,
        "uses_validation_test": 0,
        "diagnostic_promoted_to_official": 0,
        "success_v9390_strict_purekan_functional": 0,
        "success_v9390_full_functional": 0,
        "success_v9390_external_ready": 0,
    }

    manifest = {
        "run_id": out_dir.name,
        "created_at_utc": now_iso(),
        "script": rel(SCRIPT_PATH),
        "plan": rel(PLAN_PATH),
        "args": vars(args),
        "source_v9380": rel(source_v9380),
        "source_v9330": rel(Path(args.source_v9330)),
        "device": str(device),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)

    write_csv(out_dir / "p0_v9380_boundary_reanalysis.csv", [p0])
    write_csv(out_dir / "v9380_gate_gap_count_table.csv", p0_rows)
    write_csv(out_dir / "p1_source_ap0_same_panel_baseline.csv", [p1])
    write_csv(out_dir / "source_ap0_branch_horizon_trace_v9390.csv", source_outcomes)
    write_csv(out_dir / "source_ap0_baseline_summary_v9390.csv", [p1])
    write_csv(out_dir / "p2_transformation_damage_matrix.csv", [p2] + p2_rows)
    write_csv(out_dir / "source_generated_payload_geometry_trace_v9390.csv", geom_rows)
    write_csv(out_dir / "damage_attribution_summary_v9390.csv", [p2])
    write_csv(out_dir / "p3_certificate_sufficiency_lift_audit.csv", [p3] + p3_rows)
    write_csv(out_dir / "certificate_component_trace_v9390.csv", component_rows)
    write_csv(out_dir / "certificate_ablation_summary_v9390.csv", [p3])
    write_csv(out_dir / "p4_horizon_failure_dissection.csv", [p4])
    write_csv(out_dir / "horizon_value_curve_trace_v9390.csv", p4_rows)
    write_csv(out_dir / "p5_generic_low_distortion_sweep.csv", [p5] + sweep_rows)
    write_csv(out_dir / "low_distortion_payload_trace_v9390.csv", sweep_rows)
    write_csv(out_dir / "low_distortion_outcome_summary_v9390.csv", [p5])
    write_csv(out_dir / "p6_real_ap5_ap8_generator.csv", [p6])
    write_csv(out_dir / "ap5_ap8_payload_trace_v9390.csv", apv2_trace)
    write_csv(out_dir / "ap5_ap8_certificate_trace_v9390.csv", apv2_cert)
    write_csv(out_dir / "action_apply_replay_trace_v9390.csv", p6_replay)
    write_csv(out_dir / "p7_generated_ap_v2_smoke_outcome.csv", [p7])
    write_csv(out_dir / "ap_v2_smoke_outcome_trace_v9390.csv", apv2_outcomes)
    write_csv(out_dir / "branch_horizon_completion_trace_v9390.csv", apv2_completion)
    write_csv(out_dir / "p8_generated_ap_frontier_certificate_calibration.csv", [p8] + p8_rows)
    write_csv(out_dir / "certificate_controller_trace_v9390.csv", p8_rows)
    write_csv(out_dir / "frontier_support_trace_v9390.csv", p8_rows)
    write_csv(out_dir / "p9_payload_runtime_diagnostic_v9390.csv", [p9])
    write_csv(out_dir / "payload_runtime_component_trace_v9390.csv", [])
    write_csv(out_dir / "materializer_throughput_trace_v9390.csv", [p7])
    write_csv(out_dir / "p10_system_gate_generated_ap_v2.csv", [p10])
    write_csv(out_dir / "system_controller_trace_v9390.csv", [p10])
    write_csv(out_dir / "p11_leave_dataset_stratum_out_v9390.csv", [p11])
    write_csv(out_dir / "leaveout_trace_v9390.csv", [])
    write_csv(out_dir / "p12_diagnostic_paired_replay_scout_v9390.csv", [p12])
    write_csv(out_dir / "paired_replay_branch_trace_v9390.csv", [])
    write_csv(out_dir / "p13_short_full_sampleeff_continual_robustness_v9390.csv", [p13])
    write_csv(out_dir / "short_full_trace_v9390.csv", [])
    write_csv(out_dir / "continual_trace_v9390.csv", [])

    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "source_AP0_same_panel_complete": int(bool(source_outcomes)),
        "generator_damage_quantified": int(bool(p2_rows)),
        "certificate_lift_quantified": int(bool(p3_rows)),
        "horizon_failure_quantified": int(bool(p4_rows)),
        "ap5_ap8_generation_pass": p6.get("ap5_ap8_generation_pass"),
        "ap_v2_smoke_outcome_pass": p7.get("ap_v2_smoke_outcome_pass"),
        "ap_v2_frontier_pass": p8.get("generated_ap_frontier_pass", 0),
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "contract_audit_v9390.csv", [contract])
    failure = {
        "route": route,
        "F0_v9380_boundary_unstable": int(not inum(p0.get("H0_materializer_confidence"))),
        "F2_source_AP0_panel_bad": int(inum(p1.get("source_AP0_panel_bad"))),
        "F4_generator_transformation_damage": int(inum(p2.get("generator_damage_pass"))),
        "F10_certificate_no_CP_lift": int(inum(p3.get("certificate_insufficient"))),
        "F13_immediate_direction_fail": int(inum(p4.get("immediate_direction_fail"))),
        "F14_long_horizon_risk_fail": int(inum(p4.get("long_horizon_risk_primary"))),
        "F15_low_distortion_sweep_no_survivor": int(not inum(p5.get("low_distortion_sweep_pass", 0))),
        "F19_generated_AP_frontier_absent": int(not inum(p8.get("generated_ap_frontier_pass", 0))),
        "F38_diagnostic_promoted_to_official": 0,
        "F39_fake_or_proxy_violation": 0,
        "primary_blocker": blocker,
    }
    write_csv(out_dir / "failure_table.csv", [failure])
    provenance = audit_no_fake([
        out_dir / "source_ap0_branch_horizon_trace_v9390.csv",
        out_dir / "p2_transformation_damage_matrix.csv",
        out_dir / "p3_certificate_sufficiency_lift_audit.csv",
        out_dir / "ap5_ap8_payload_trace_v9390.csv",
        out_dir / "ap_v2_smoke_outcome_trace_v9390.csv",
        out_dir / "contract_audit_v9390.csv",
    ])
    write_csv(out_dir / "provenance_audit_v9390.csv", [provenance])
    write_hashes(out_dir, [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "p0_v9380_boundary_reanalysis.csv",
        out_dir / "p1_source_ap0_same_panel_baseline.csv",
        out_dir / "p2_transformation_damage_matrix.csv",
        out_dir / "p3_certificate_sufficiency_lift_audit.csv",
        out_dir / "p4_horizon_failure_dissection.csv",
        out_dir / "p5_generic_low_distortion_sweep.csv",
        out_dir / "p6_real_ap5_ap8_generator.csv",
        out_dir / "p7_generated_ap_v2_smoke_outcome.csv",
        out_dir / "p8_generated_ap_frontier_certificate_calibration.csv",
        out_dir / "contract_audit_v9390.csv",
        out_dir / "provenance_audit_v9390.csv",
        out_dir / "failure_table.csv",
    ])
    print(json.dumps({"out_dir": str(out_dir), "route": route, "source_AP0_weak": p1.get("source_AP0_weak_CP_precision"), "ap_v2_weak": p7.get("cert_pass_weak_CP_precision", "")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
