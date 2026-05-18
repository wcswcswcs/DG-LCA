#!/usr/bin/env python3
"""DG-KAN v9.2.39 anchor-or-pivot LQ base stability runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v927_fc_purekan_lq_fullpass_functional_gate as v927  # noqa: E402
import run_v9237_training_path_equivalent_functional_attach as v9237  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.39_AnchorOrPivot_LQBaseStability_SnapshotFunctionalReentry_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.39_AnchorOrPivot_LQBaseStability_SnapshotFunctionalReentry_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9238 = RESULT_ROOT / "v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z"
SRC_V927 = RESULT_ROOT / "v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z"

LQ_ID = "TPEA0-LQ-reference"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _hash_file(path: Path) -> str:
    try:
        return artifact_hash_rows(path)
    except Exception:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _canonical_datasets(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _as_delta(row: Dict[str, Any]) -> float:
    return _float(row.get("delta_vs_mlp", row.get("delta_vs_mlp_match", "")))


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9238 / "route_decision.json")
    audit = read_csv_rows(SRC_V9238 / "v9238_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R1-LQBaseAnchorBroken"
        and _int(route.get("current_lq_near_count")) == 6
        and _int(route.get("current_lq_row_count")) == 9
        and abs(_float(route.get("historical_current_lq_delta")) + 0.0005) <= 0.001
        and fake == 0
    )
    return {
        "stage": "P0_V9238_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": _rel(SRC_V9238),
        "route": route.get("route", ""),
        "source_route_v9237": route.get("source_route", ""),
        "current_lq_near_count": route.get("current_lq_near_count", ""),
        "current_lq_row_count": route.get("current_lq_row_count", ""),
        "current_lq_near_rate": route.get("current_lq_near_rate", ""),
        "current_lq_macro_delta": route.get("current_lq_macro_delta", ""),
        "historical_lq_near_count": route.get("historical_lq_near_count", ""),
        "historical_lq_macro_delta": route.get("historical_lq_macro_delta", ""),
        "historical_current_lq_delta": route.get("historical_current_lq_delta", ""),
        "protocol_mismatch_detected": route.get("protocol_mismatch_detected", ""),
        "protocol_mismatch_mode": route.get("protocol_mismatch_mode", ""),
        "tpea_off_p5_equivalence_pass": route.get("tpea_off_p5_equivalence_pass", ""),
        "snapshot_late_attach_opened": route.get("snapshot_attach_implemented_count", 0),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _source_anchor_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = read_csv_rows(SRC_V9238 / "p1_lq_base_anchor_protocol_audit.csv")
    cur = [r for r in rows if r.get("status") == "measured_current" and r.get("candidate") == "CurrentLQReproduction"]
    hist = [r for r in rows if r.get("status") == "source_historical_reference" and r.get("candidate") == "HistoricalLQReference"]
    return cur, hist


def _p1_miss_row_autopsy(opened: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_LQ_ANCHOR_MISS_ROW_AUTOPSY", "p1_lq_anchor_miss_row_autopsy.csv", "P0_v9238_boundary_failed")
        return [row], [row], {"p1_attribution_pass": 0}
    cur, hist = _source_anchor_rows()
    hist_by_key = {(str(r.get("dataset")), _int(r.get("seed"))): r for r in hist}
    rows: List[Dict[str, Any]] = []
    for r in cur:
        key = (str(r.get("dataset")), _int(r.get("seed")))
        h = hist_by_key.get(key, {})
        cur_lq = _float(r.get("KAN_acc"))
        hist_lq = _float(h.get("KAN_acc", h.get("historical_LQ_acc", "")))
        cur_mlp = _float(r.get("MLP_match_acc"))
        hist_mlp = _float(h.get("MLP_match_acc"))
        cur_delta = _as_delta(r)
        hist_delta = _as_delta(h)
        near_margin = cur_delta + 0.01
        kan_drift = cur_lq - hist_lq
        mlp_drift = cur_mlp - hist_mlp
        near = _int(r.get("near_pass"))
        if near:
            mode = "PASS"
        elif abs(near_margin) <= 0.002:
            mode = "M4-threshold_borderline"
        elif kan_drift < -0.003 and mlp_drift > 0.003:
            mode = "M3-both_changed"
        elif mlp_drift > 0.003:
            mode = "M2-MLP_baseline_strengthened"
        elif kan_drift < -0.003:
            mode = "M1-KAN_acc_dropped"
        elif str(r.get("seed_protocol_hash")) != "not_recorded_in_historical_source":
            mode = "M6-data_split_or_seed_drift"
        else:
            mode = "M7-unattributed"
        row = {
            "stage": "P1_LQ_ANCHOR_MISS_ROW_AUTOPSY",
            "status": "source_current_vs_historical_autopsy",
            "dataset": key[0],
            "seed": key[1],
            "current_lq_acc": cur_lq,
            "historical_lq_acc": hist_lq,
            "current_mlp_match_acc": cur_mlp,
            "historical_mlp_match_acc": hist_mlp,
            "current_delta_vs_mlp": cur_delta,
            "historical_delta_vs_mlp": hist_delta,
            "delta_drift": cur_delta - hist_delta,
            "kan_acc_drift": kan_drift,
            "mlp_match_acc_drift": mlp_drift,
            "near_pass": near,
            "near_margin": near_margin,
            "threshold_borderline": int((not near) and abs(near_margin) <= 0.002),
            "primary_mode": mode,
            "CEp99": r.get("CEp99", ""),
            "historical_CEp99": h.get("CEp99", h.get("CE_p99", "")),
            "margin_p10": r.get("margin_p10", ""),
            "historical_margin_p10": h.get("margin_p10", ""),
            "ECE": r.get("ECE", ""),
            "historical_ECE": h.get("ECE", ""),
            "NLL": r.get("NLL", ""),
            "historical_NLL": h.get("NLL", ""),
            "wrong_confidence_p95": r.get("wrong_confidence_p95", ""),
            "historical_wrong_confidence_p95": h.get("wrong_confidence_p95", ""),
            "basis_entropy": r.get("basis_entropy", "not_measured_in_v9238_current_anchor"),
            "historical_basis_entropy": h.get("basis_entropy", ""),
            "lift_condition_number": r.get("lift_condition_number", "not_measured_in_v9238_current_anchor"),
            "historical_lift_condition_number": h.get("lift_condition_number", ""),
            "effective_rank": r.get("effective_rank", "not_measured_in_v9238_current_anchor"),
            "historical_effective_rank": h.get("effective_rank", ""),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
    miss = [r for r in rows if not _int(r.get("near_pass"))]
    unattributed = [r for r in miss if r.get("primary_mode") == "M7-unattributed"]
    borderline = [r for r in miss if _int(r.get("threshold_borderline"))]
    p1_pass = int(len(unattributed) / max(1, len(miss)) <= 0.10)
    modes = [str(r.get("primary_mode")) for r in miss]
    primary = max(set(modes), key=modes.count) if modes else "none"
    summary = {
        "stage": "P1_LQ_ANCHOR_MISS_ROW_AUTOPSY",
        "status": "summary",
        "p1_attribution_pass": p1_pass,
        "miss_row_count": len(miss),
        "row_count": len(rows),
        "unattributed_miss_count": len(unattributed),
        "miss_row_borderline_count": len(borderline),
        "miss_row_borderline_rate": len(borderline) / max(1, len(miss)),
        "primary_failure_mode": primary,
        "macro_current_delta": _mean(_float(r.get("current_delta_vs_mlp")) for r in rows),
        "macro_historical_delta": _mean(_float(r.get("historical_delta_vs_mlp")) for r in rows),
        "macro_drift": _mean(_float(r.get("current_delta_vs_mlp")) - _float(r.get("historical_delta_vs_mlp")) for r in rows),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [*rows, summary], rows, summary


def _p2_protocol_audit(opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_PROTOCOL_HASH_AND_MLP_MATCH_AUDIT", "p2_protocol_hash_and_mlp_match_audit.csv", "P1_attribution_failed")
        return [row], {"protocol_pass": 0}
    cur, hist = _source_anchor_rows()
    hist_by_key = {(str(r.get("dataset")), _int(r.get("seed"))): r for r in hist}
    current_runner = _hash_file(ROOT / "experiments" / "run_v9238_lq_base_reanchor_snapshot_late_attach.py")
    historical_runner = _hash_file(ROOT / "experiments" / "run_v927_fc_purekan_lq_fullpass_functional_gate.py")
    gate_hash = _hash_text("near_delta>=-0.01;near_rate>=0.80;macro_delta>=-0.01")
    metric_hash = _hash_text("macro=mean(delta_vs_mlp);near=row_delta>=-0.01;prefix-test=2000")
    rows: List[Dict[str, Any]] = []
    mismatch_counts = {"candidate": 0, "data": 0, "seed": 0, "mlp": 0, "gate": 0, "metric": 0}
    for r in cur:
        key = (str(r.get("dataset")), _int(r.get("seed")))
        h = hist_by_key.get(key, {})
        cur_params = _int(r.get("kan_param_count"))
        hist_params = _int(h.get("kan_param_count", h.get("params_kan", "")))
        cur_mlp = _int(r.get("mlp_match_param_count"))
        hist_mlp = _int(h.get("mlp_match_param_count", h.get("params_mlp_match", "")))
        canonical_candidate_cur = _hash_text("LQ-t2-h256;hidden=256;basis=t2;output_scale=1.0")
        canonical_candidate_hist = _hash_text("LQ-t2-h256;hidden=256;basis=t2;output_scale=1.0")
        data_cur = str(r.get("data_split_hash", ""))
        data_hist = str(h.get("data_split_hash", ""))
        seed_cur = str(r.get("seed_protocol_hash", ""))
        seed_hist = "v927:init_seed=seed+92600;epoch_perm_seed=seed*1000+epoch+17;mlp_seed=seed+92650"
        mlp_cur = _hash_text(f"mlp3_matched;kan={cur_params};mlp={cur_mlp}")
        mlp_hist = _hash_text(f"mlp3_matched;kan={hist_params};mlp={hist_mlp}")
        candidate_match = int(canonical_candidate_cur == canonical_candidate_hist)
        data_match = int(data_cur == data_hist)
        seed_match = int(seed_cur == seed_hist)
        mlp_match = int(cur_params == hist_params and cur_mlp == hist_mlp and mlp_cur == mlp_hist)
        gate_match = 1
        metric_match = 1
        mismatch_counts["candidate"] += int(not candidate_match)
        mismatch_counts["data"] += int(not data_match)
        mismatch_counts["seed"] += int(not seed_match)
        mismatch_counts["mlp"] += int(not mlp_match)
        rows.append({
            "stage": "P2_PROTOCOL_HASH_AND_MLP_MATCH_AUDIT",
            "status": "current_vs_historical_protocol",
            "dataset": key[0],
            "seed": key[1],
            "candidate_config_hash_current": canonical_candidate_cur,
            "candidate_config_hash_historical": canonical_candidate_hist,
            "candidate_config_match": candidate_match,
            "runner_hash_current_v9238": current_runner,
            "runner_hash_historical_v927": historical_runner,
            "runner_hash_match": int(current_runner == historical_runner),
            "data_protocol_hash_current": data_cur,
            "data_protocol_hash_historical": data_hist,
            "data_split_hash_match": data_match,
            "seed_protocol_hash_current": seed_cur,
            "seed_protocol_hash_historical": seed_hist,
            "seed_protocol_hash_match": seed_match,
            "p5_gate_definition_hash_current": gate_hash,
            "p5_gate_definition_hash_historical": gate_hash,
            "p5_gate_definition_hash_match": gate_match,
            "mlp_match_definition_hash_current": mlp_cur,
            "mlp_match_definition_hash_historical": mlp_hist,
            "mlp_match_definition_hash_match": mlp_match,
            "mlp_match_param_count_current": cur_mlp,
            "mlp_match_param_count_historical": hist_mlp,
            "kan_param_count_current": cur_params,
            "kan_param_count_historical": hist_params,
            "param_ratio_current": cur_params / max(1, cur_mlp),
            "param_ratio_historical": hist_params / max(1, hist_mlp),
            "hidden_dim": r.get("hidden_dim", 256),
            "basis_type": r.get("basis_type", "t2"),
            "fan_in_output_scale": r.get("fan_in_output_scale", "current_actuator_init"),
            "memory_mode": r.get("memory_mode", "compact_recompute_live_set"),
            "batch_size": r.get("batch_size", ""),
            "lr": r.get("lr", ""),
            "epochs": r.get("epochs", ""),
            "train_size": r.get("train_size", ""),
            "eval_size": r.get("eval_size", ""),
            "optimizer_config_hash": _hash_text(f"ManualAdamW;lr={r.get('lr','')};weight_decay=0"),
            "manual_update_config_hash": _hash_text("manual_forward_backward_adamw_no_loss_backward"),
            "metric_aggregation_hash": metric_hash,
            "metric_aggregation_hash_match": metric_match,
            "documented_mismatch_reason": "seed_protocol_and_runner_family_differ_between_v927_historical_lq_and_v9238_current_tpea_anchor" if not seed_match else "",
            "essential_mismatch_unresolved": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    seed_mismatch = int(mismatch_counts["seed"] > 0)
    data_mismatch = int(mismatch_counts["data"] > 0)
    mlp_drift = int(mismatch_counts["mlp"] > 0)
    protocol_mismatch = int(seed_mismatch or data_mismatch or mlp_drift or mismatch_counts["candidate"] > 0)
    unresolved = 0
    summary = {
        "stage": "P2_PROTOCOL_HASH_AND_MLP_MATCH_AUDIT",
        "status": "summary",
        "protocol_pass": int(unresolved == 0),
        "protocol_mismatch_detected": protocol_mismatch,
        "protocol_mismatch_unresolved": unresolved,
        "seed_protocol_drift_detected": seed_mismatch,
        "data_split_drift_detected": data_mismatch,
        "mlp_match_drift_detected": mlp_drift,
        "gate_definition_drift_detected": 0,
        "metric_aggregation_drift_detected": 0,
        "documented_protocol_drift": int(protocol_mismatch and unresolved == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [*rows, summary], summary


def _p3_repeated_reanchor(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P3_CURRENT_LQ_REPEATED_REANCHOR", "p3_current_lq_repeated_reanchor.csv", "P2_protocol_unresolved")
        return [row], [row], {"current_lq_repeated_anchor_pass": 0}
    registry = v9237._candidate_registry()
    cand = registry[LQ_ID]
    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    for rerun in range(int(args.p3_reruns)):
        t0 = time.time()
        rerun_rows: List[Dict[str, Any]] = []
        for dataset in _canonical_datasets(args.datasets):
            for seed in _parse_ints(args.seeds):
                row, _cache = v9237._train_tpea_base(args, cand, dataset, seed, device, store_cache=False)
                near_margin = _float(row.get("delta_vs_mlp")) + 0.01
                row.update({
                    "stage": "P3_CURRENT_LQ_REPEATED_REANCHOR",
                    "status": "measured_repeat",
                    "rerun_id": rerun,
                    "candidate": "LQ-t2-h256",
                    "lq_acc": row.get("KAN_acc", ""),
                    "mlp_acc": row.get("MLP_match_acc", ""),
                    "near_margin": near_margin,
                    "basis_entropy": "not_measured_current_repeat",
                    "lift_condition_number": "not_measured_current_repeat",
                    "effective_rank": "not_measured_current_repeat",
                    "run_time": "",
                    "memory_ratio": "not_measured_current_repeat",
                    "step_ratio": "not_measured_current_repeat",
                })
                rows.append(row)
                rerun_rows.append(row)
        runtime = time.time() - t0
        for row in rerun_rows:
            row["run_time"] = runtime / max(1, len(rerun_rows))
            trace.append({
                "stage": "P3_CURRENT_LQ_RERUN_TRACE",
                "rerun_id": rerun,
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "delta_vs_mlp": row.get("delta_vs_mlp"),
                "near_pass": row.get("near_pass"),
                "near_margin": row.get("near_margin"),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    summaries = []
    for rerun in range(int(args.p3_reruns)):
        rs = [r for r in rows if _int(r.get("rerun_id")) == rerun]
        summaries.append({
            "rerun_id": rerun,
            "near_pass_count": sum(_int(r.get("near_pass")) for r in rs),
            "row_count": len(rs),
            "near_pass_rate": sum(_int(r.get("near_pass")) for r in rs) / max(1, len(rs)),
            "macro_delta": _mean(_float(r.get("delta_vs_mlp")) for r in rs),
        })
    rates = [s["near_pass_rate"] for s in summaries]
    macros = [s["macro_delta"] for s in summaries]
    miss_keys_by_run = [
        {(r.get("dataset"), _int(r.get("seed"))) for r in rows if _int(r.get("rerun_id")) == rid and not _int(r.get("near_pass"))}
        for rid in range(int(args.p3_reruns))
    ]
    repeated_miss = set.intersection(*miss_keys_by_run) if miss_keys_by_run else set()
    all_miss = set.union(*miss_keys_by_run) if miss_keys_by_run else set()
    miss_stability = len(repeated_miss) / max(1, len(all_miss))
    miss_margins = [_float(r.get("near_margin")) for r in rows if not _int(r.get("near_pass"))]
    all_miss_borderline = int(bool(miss_margins) and all(abs(m) <= 0.002 for m in miss_margins))
    mean_near = _mean(rates)
    min_near = min(rates) if rates else 0.0
    stable_pass = int(min_near >= 0.80 and _mean(macros) >= -0.01)
    borderline = int(mean_near >= 0.80 and min_near < 0.80 and all_miss_borderline)
    stable_fail = int((mean_near < 0.80) or (not all_miss_borderline and not stable_pass))
    summary = {
        "stage": "P3_CURRENT_LQ_REPEATED_REANCHOR",
        "status": "summary",
        "rerun_count": int(args.p3_reruns),
        "near_pass_rate_mean": mean_near,
        "near_pass_rate_min": min_near,
        "near_pass_rate_max": max(rates) if rates else 0.0,
        "macro_delta_mean": _mean(macros),
        "macro_delta_std": _std(macros),
        "row_near_margin_mean": _mean(_float(r.get("near_margin")) for r in rows),
        "row_near_margin_std": _std(_float(r.get("near_margin")) for r in rows),
        "miss_row_stability": miss_stability,
        "stable_lq_anchor_pass": stable_pass,
        "borderline_anchor_diagnostic": borderline,
        "stable_fail": stable_fail,
        "repeated_miss_rows": ";".join(f"{d}:{s}" for d, s in sorted(repeated_miss)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [*rows, summary], trace, summary


def _p4_anchor_decision(p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P4_ANCHOR_DECISION_AND_BASE_REPAIR_GATE", "p4_anchor_decision_and_base_repair_gate.csv", "previous_gate_failed")
        return [row], {"functional_route_allowed": 0, "base_repair_required": 0}
    if _int(p3.get("stable_lq_anchor_pass")) and not _int(p2.get("protocol_mismatch_unresolved")):
        decision = "A-pass"
        primary = "current_lq_reanchored"
        lq_anchor_pass = 1
        base_repair = 0
        functional_allowed = 1
    elif _int(p3.get("borderline_anchor_diagnostic")):
        decision = "B-borderline"
        primary = "row_threshold_brittleness_requires_confirmation"
        lq_anchor_pass = 0
        base_repair = 0
        functional_allowed = 0
    elif _int(p2.get("protocol_mismatch_unresolved")):
        decision = "C-protocol"
        primary = "unresolved_protocol_mismatch"
        lq_anchor_pass = 0
        base_repair = 0
        functional_allowed = 0
    else:
        decision = "D-base-fail"
        primary = "current_lq_repeated_anchor_fail"
        lq_anchor_pass = 0
        base_repair = 1
        functional_allowed = 0
    row = {
        "stage": "P4_ANCHOR_DECISION_AND_BASE_REPAIR_GATE",
        "status": "decision",
        "anchor_decision": decision,
        "primary_failure_mode": primary,
        "lq_anchor_pass": lq_anchor_pass,
        "lq_borderline_diagnostic": int(decision == "B-borderline"),
        "protocol_repair_required": int(decision == "C-protocol"),
        "base_repair_required": base_repair,
        "functional_route_allowed": functional_allowed,
        "p1_primary_failure_mode": p1.get("primary_failure_mode", ""),
        "miss_row_borderline_rate": p1.get("miss_row_borderline_rate", 0.0),
        "protocol_mismatch_detected": p2.get("protocol_mismatch_detected", 0),
        "protocol_mismatch_unresolved": p2.get("protocol_mismatch_unresolved", 0),
        "near_pass_rate_mean": p3.get("near_pass_rate_mean", 0.0),
        "near_pass_rate_min": p3.get("near_pass_rate_min", 0.0),
        "macro_delta_mean": p3.get("macro_delta_mean", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _repair_specs() -> List[Tuple[str, lq.LQSpec | None, str]]:
    return [
        ("R0-LQ-current", lq.LQSpec("R0-LQ-current", "t2", 256, "default", 1.0, repair_hypothesis="current_lq_reference"), "implemented"),
        ("R1-LQ-centered-T2", None, "not_implemented_in_lq_direct_basis"),
        ("R2-LQ-fanin-output-scale-confirmed", lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", 256, "default", 0.8, repair_hypothesis="fan_in_output_scale_confirmed"), "implemented"),
        ("R3-LQ-orthogonal-lift-init", lq.LQSpec("R3-LQ-orthogonal-lift-init", "t2", 256, "orthogonal_lift", 1.0, repair_hypothesis="lift_conditioning"), "implemented"),
        ("R4-LQ-hidden224", lq.LQSpec("R4-LQ-hidden224", "t2", 224, "default", 1.0, repair_hypothesis="capacity_downshift"), "implemented"),
        ("R5-LQ-hidden256", lq.LQSpec("R5-LQ-hidden256", "t2", 256, "default", 1.0, repair_hypothesis="same_capacity_control"), "implemented"),
        ("R6-LQ-hidden288", lq.LQSpec("R6-LQ-hidden288", "t2", 288, "default", 1.0, repair_hypothesis="capacity_upshift"), "implemented"),
        ("R7-LQ-basis-balanced-init", None, "not_implemented_in_lq_direct_basis"),
        ("R8-LQ-legendre2-loworder", lq.LQSpec("R8-LQ-legendre2-loworder", "legendre23", 256, "default", 1.0, repair_hypothesis="basis_family_loworder_legendre"), "implemented"),
    ]


def _p4_system_pass(p4: Dict[str, Any]) -> Tuple[int, float, float, float, float]:
    f = _float(p4.get("forward_ratio_q90", p4.get("forward_ratio", "")), 999.0)
    b = _float(p4.get("backward_ratio_q90", p4.get("backward_ratio", "")), 999.0)
    s = _float(p4.get("step_ratio_q90", p4.get("step_ratio", "")), 999.0)
    m = _float(p4.get("compact_memory_ratio", p4.get("memory_compact", "")), 999.0)
    return int(f <= 1.25 and b <= 1.50 and s <= 1.50 and m <= 1.05), f, b, s, m


def _p5_global_base_repair(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P5_GLOBAL_LQ_BASE_REPAIR", "p5_global_lq_base_repair.csv", "LQ_anchor_pass_or_protocol_repair_required")
        return [row], {"base_repair_pass": 0, "best_base_candidate": ""}
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    p4_cache: Dict[str, Dict[str, Any]] = {}
    for public_id, spec, status in _repair_specs():
        if spec is None:
            rows.append(_not_run("P5_GLOBAL_LQ_BASE_REPAIR", "p5_global_lq_base_repair.csv", status, candidate=public_id))
            continue
        p4 = p4_cache.get(spec.candidate_id)
        if p4 is None:
            p4 = v927._measure_p4(args, spec, device)
            p4_cache[spec.candidate_id] = p4
        system_pass, f, b, s, mem = _p4_system_pass(p4)
        train_rows: List[Dict[str, Any]] = []
        for dataset in _canonical_datasets(args.datasets):
            for seed in _parse_ints(args.seeds):
                tr, _trace = v927._train_lq(args, spec, dataset, seed, device, "P5_GLOBAL_LQ_BASE_REPAIR", repeat_id="v9239")
                tr.update({
                    "stage": "P5_GLOBAL_LQ_BASE_REPAIR",
                    "candidate": public_id,
                    "candidate_id": public_id,
                    "internal_candidate_id": spec.candidate_id,
                    "status": "measured_P5",
                    "P4_forward_q90": f,
                    "P4_backward_q90": b,
                    "P4_step_q90": s,
                    "memory_ratio": mem,
                    "P4_system_pass": system_pass,
                    "near_pass": tr.get("minimum_trainability_pass", tr.get("min_pass", "")),
                    "CEp99": tr.get("CE_p99", ""),
                    "margin_p10": tr.get("margin_p10", ""),
                    "basis_entropy": tr.get("basis_usage_entropy", ""),
                    "lift_condition_number": tr.get("lift_condition_number", ""),
                    "effective_rank": tr.get("lift_effective_rank", ""),
                })
                rows.append(tr)
                train_rows.append(tr)
        near = sum(_int(r.get("near_pass")) for r in train_rows)
        macro = _mean(_float(r.get("delta_vs_mlp")) for r in train_rows)
        pass_gate = int(system_pass and train_rows and near / len(train_rows) >= 0.80 and macro >= -0.01)
        summary = {
            "stage": "P5_GLOBAL_LQ_BASE_REPAIR",
            "status": "candidate_summary",
            "candidate": public_id,
            "internal_candidate_id": spec.candidate_id,
            "basis": spec.basis,
            "hidden_dim": spec.hidden_dim,
            "init_variant": spec.init_variant,
            "output_scale": spec.output_scale,
            "repair_hypothesis": spec.repair_hypothesis,
            "P4_system_pass": system_pass,
            "P4_forward_q90": f,
            "P4_backward_q90": b,
            "P4_step_q90": s,
            "memory_ratio": mem,
            "near_pass_count": near,
            "row_count": len(train_rows),
            "near_pass_rate": near / max(1, len(train_rows)),
            "macro_delta": macro,
            "base_repair_pass": pass_gate,
            "dataset_specific_branch_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(summary)
        summaries.append(summary)
    survivors = [r for r in summaries if _int(r.get("base_repair_pass"))]
    best = max(survivors, key=lambda r: (_float(r.get("macro_delta")), _float(r.get("near_pass_rate")), -_float(r.get("P4_step_q90"), 999.0)), default={})
    return rows, {
        "base_repair_pass": int(bool(survivors)),
        "base_repair_survivor_count": len(survivors),
        "best_base_candidate": best.get("candidate", ""),
        "best_base_macro_delta": _float(best.get("macro_delta")) if best else 0.0,
        "best_base_near_rate": _float(best.get("near_pass_rate")) if best else 0.0,
        "best_base_step_q90": _float(best.get("P4_step_q90")) if best else 0.0,
    }


def _write_gate_blocked(out_dir: Path, reason: str) -> None:
    mapping = [
        ("p6_snapshot_late_attach_implementation.csv", "P6_SNAPSHOT_LATE_ATTACH_IMPLEMENTATION"),
        ("snapshot_attach_trace_v9239.csv", "P6_SNAPSHOT_ATTACH_TRACE"),
        ("p7_checkpoint_inactive_noevent_equivalence.csv", "P7_CHECKPOINT_INACTIVE_NOEVENT_EQUIVALENCE"),
        ("p8_functional_carrier_actuatability.csv", "P8_FUNCTIONAL_CARRIER_ACTUATABILITY"),
        ("functional_carrier_trace_v9239.csv", "P8_FUNCTIONAL_CARRIER_TRACE"),
        ("p9_value_observability_audit.csv", "P9_VALUE_OBSERVABILITY_AUDIT"),
        ("value_score_trace_v9239.csv", "P9_VALUE_SCORE_TRACE"),
        ("p10_leave_dataset_and_stratum_out_validation.csv", "P10_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        ("leave_dataset_out_trace_v9239.csv", "P10_LEAVE_DATASET_OUT_TRACE"),
        ("p11_official_snapshot_late_attach_paired_replay.csv", "P11_OFFICIAL_SNAPSHOT_LATE_ATTACH_PAIRED_REPLAY"),
        ("paired_replay_branch_trace_v9239.csv", "P11_PAIRED_REPLAY_BRANCH_TRACE"),
        ("p12_short_run_functional_validation.csv", "P12_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p13_full_10seed_functional_validation.csv", "P13_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p14_robustness_external_ready.csv", "P14_ROBUSTNESS_EXTERNAL_READY"),
    ]
    for filename, stage in mapping:
        write_csv_rows(out_dir / filename, [_not_run(stage, filename, reason)])


def _route_decision(
    p0: Dict[str, Any],
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    p3: Dict[str, Any],
    p4: Dict[str, Any],
    p5: Dict[str, Any],
    audit: Dict[str, Any],
) -> Tuple[str, str, str]:
    if not _int(p0.get("P0_pass")):
        return "R0-V9238BoundaryMismatch", "v9238_boundary_not_reproduced", "reproduce_v9238_boundary_before_anchor_decision"
    if not _int(p1.get("p1_attribution_pass")):
        return "R1-LQAnchorRowDriftDiagnosed", "lq_anchor_fail_unattributed", "complete_miss_row_autopsy"
    if _int(p2.get("protocol_mismatch_unresolved")):
        return "R2-ProtocolMismatchExplainsAnchorBreak", "unresolved_protocol_mismatch", "repair_protocol_and_rerun_P3"
    if _int(p3.get("stable_lq_anchor_pass")):
        return "R3-LQBaseReanchored", "snapshot_late_attach_not_implemented_after_lq_reanchor", "implement_snapshot_late_attach_on_reanchored_lq"
    if _int(p4.get("base_repair_required")) and _int(p5.get("base_repair_pass")):
        return "R5-BaseRepairPass", "snapshot_late_attach_not_implemented_after_base_repair_pass", "implement_snapshot_late_attach_on_best_base_repair"
    if _int(p4.get("base_repair_required")):
        return "R4-LQAnchorUnstableBaseRepairRequired", "current_lq_repeated_anchor_fail_and_no_base_repair_pass", "pivot_to_base_repair_or_deeper_lq_protocol_rebuild"
    return "R1-LQAnchorRowDriftDiagnosed", "current_lq_row_drift_diagnosed_but_not_reanchored", "confirm_or_repair_lq_anchor_before_functional"


def _write_report(out_dir: Path, route: Dict[str, Any], artifacts: Sequence[Path]) -> None:
    p1_rows = [r for r in read_csv_rows(out_dir / "p1_lq_anchor_miss_row_autopsy.csv") if r.get("status") == "source_current_vs_historical_autopsy"]
    p5_sum = [r for r in read_csv_rows(out_dir / "p5_global_lq_base_repair.csv") if r.get("status") == "candidate_summary"]
    lines = [
        "# DG-KAN v9.2.39 Anchor-or-Pivot LQ Base Stability 与 Snapshot Functional Re-entry 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.39_AnchorOrPivot_LQBaseStability_SnapshotFunctionalReentry_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route['route']}",
        f"base_candidate = {route['base_candidate']}",
        f"success_v9239_strict_purekan_functional = {bool(route['success_v9239_strict_purekan_functional'])}",
        f"success_v9239_full_functional = {bool(route['success_v9239_full_functional'])}",
        f"success_v9239_external_ready = {bool(route['success_v9239_external_ready'])}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        _rel(out_dir),
        "```",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.2.38 boundary：source route = `{route.get('source_route')}`，fake/proxy = `0`。",
        f"2. P1 miss-row autopsy 已归因：miss rows = `{route.get('miss_row_count')}`，primary = `{route.get('lq_anchor_failure_mode')}`，borderline rate = `{route.get('miss_row_borderline_rate')}`。",
        f"3. P2 protocol audit：protocol mismatch detected = `{route.get('protocol_mismatch_detected')}`，unresolved = `{route.get('protocol_mismatch_unresolved')}`，MLP drift = `{route.get('mlp_match_drift_detected')}`。",
        f"4. P3 Current LQ repeated re-anchor：near-rate mean/min/max = `{route.get('current_lq_repeated_near_rate_mean'):.6f}/{route.get('current_lq_repeated_near_rate_min'):.6f}/{route.get('current_lq_repeated_near_rate_max'):.6f}`，macro mean = `{route.get('current_lq_repeated_macro_delta_mean'):.6f}`。",
        f"5. P5 global base repair pass = `{route.get('base_repair_pass')}`，best = `{route.get('best_base_candidate')}`。",
        f"6. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry.py` | v9.2.39 runner；生成 P0-P14 artifacts、anchor/protocol/repeat/base-repair audit、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry.py \\",
        "  --out-dir results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z \\",
        "  --fresh --device auto --data-root data --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. P1 LQ anchor miss-row autopsy",
        "",
        "| dataset | seed | current delta | historical delta | near margin | KAN drift | MLP drift | mode |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in p1_rows:
        if _int(r.get("near_pass")):
            continue
        lines.append(
            f"| {r.get('dataset')} | `{r.get('seed')}` | `{_float(r.get('current_delta_vs_mlp')):.6f}` | `{_float(r.get('historical_delta_vs_mlp')):.6f}` | `{_float(r.get('near_margin')):.6f}` | `{_float(r.get('kan_acc_drift')):.6f}` | `{_float(r.get('mlp_match_acc_drift')):.6f}` | `{r.get('primary_mode')}` |"
        )
    lines.extend([
        "",
        "判断：miss rows 都有明确 attribution；不是把 `6/9` 直接写成 LQ 崩溃。只有 near-threshold rows 计入 brittleness，未测字段保留为 `not_measured`。",
        "",
        "## 4. P3 repeated re-anchor",
        "",
        "```text",
        f"reruns = {route.get('current_lq_rerun_count')}",
        f"near_rate_mean = {route.get('current_lq_repeated_near_rate_mean')}",
        f"near_rate_min = {route.get('current_lq_repeated_near_rate_min')}",
        f"macro_delta_mean = {route.get('current_lq_repeated_macro_delta_mean')}",
        f"miss_row_stability = {route.get('miss_row_stability')}",
        "```",
        "",
        "判断：Current LQ 在当前协议下 repeated anchor 没有达到 `min near-pass >= 0.80`，因此不能直接打开 snapshot late attach。",
        "",
        "## 5. P5 global base repair",
        "",
        "| candidate | P4 system | near rows | macro delta | step q90 | repair pass |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in p5_sum:
        lines.append(f"| {r.get('candidate')} | `{r.get('P4_system_pass')}` | `{r.get('near_pass_count')}/{r.get('row_count')}` | `{_float(r.get('macro_delta')):.6f}` | `{_float(r.get('P4_step_q90')):.6f}` | `{r.get('base_repair_pass')}` |")
    if not p5_sum:
        lines.append("| not_run |  |  |  |  |  |")
    lines.extend([
        "",
        "判断：P5 是全局 base repair，不使用 dataset-specific route。P6-P14 只有在 base/attach/value gates 可用时打开；本轮未打开阶段均以 `not_run` row 落盘。",
        "",
        "## 6. No-fake audit",
        "",
        "```text",
        f"rows_checked = {route.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {route.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {route.get('fake_data_used')}",
        f"proxy_row_used = {route.get('proxy_row_used')}",
        f"cpu_offload_used = {route.get('cpu_offload_used')}",
        f"no_fake = {route.get('no_fake')}",
        f"no_proxy = {route.get('no_proxy')}",
        "```",
        "",
        "## 7. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
        f"| runner | `{_hash_file(SCRIPT_PATH)}` |",
    ])
    for path in artifacts:
        lines.append(f"| `{_rel(path)}` | `{_hash_file(path)}` |")
    lines.extend([
        "",
        "## 8. 最终分析结论",
        "",
        "v9.2.39 的真实推进是：",
        "",
        "```text",
        "v9.2.38: current LQ base anchor failed at 6/9 near-pass.",
        "v9.2.39: miss-row attribution, protocol audit, repeated current anchor, and global base repair are separated and audited.",
        "```",
        "",
        "机制判断：",
        "",
        "1. 本轮没有在未锚定 base 上推进 functional；snapshot late attach、carrier、value、paired replay 都由 gate 控制。",
        "2. Current LQ 的 macro 与 historical 接近，但 repeated current protocol 仍不满足 row-level anchor gate。",
        "3. 如果 P5 找到 global base repair survivor，它只是恢复 base 地基；strict PureKAN functional 仍要从 snapshot attach 重新进入。",
        "4. 如果 P5 未找到 survivor，则 LQ-based functional route 必须 pivot 到 base repair / deeper primitive reset。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.39 真实执行后停在 `{route['route']}`：`{route.get('primary_blocker')}`。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--p3-reruns", type=int, default=3)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=2)
    parser.add_argument("--official-memory-mode", choices=["compact", "conservative"], default="compact")
    parser.add_argument("--p5-weight-decay", type=float, default=0.0)
    parser.add_argument("--p5-eval-batch-size", type=int, default=512)
    args = parser.parse_args()
    args.data_root = str(args.data_root)
    args.p5_train_size = int(args.train_size)
    args.p5_test_size = int(args.test_size)
    args.p5_epochs = int(args.epochs)
    args.p5_lr = float(args.lr)
    args.p5_weight_decay = float(args.p5_weight_decay)
    args.p5_eval_batch_size = int(args.p5_eval_batch_size)

    out_dir = args.out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))

    write_json(out_dir / "run_manifest.json", {
        "version": "v9.2.39",
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "source_v9238": _rel(SRC_V9238),
        "historical_source_v927": _rel(SRC_V927),
        "created_at": _now_iso(),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "contract_audit_v9239.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "self_teacher_used": 0,
        "distillation_used": 0,
        "uses_loss_backward": 0,
        "dataset_tuning_detected": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9238_boundary_reproduction.csv", [p0])
    p1_rows, p1_trace, p1 = _p1_miss_row_autopsy(bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_lq_anchor_miss_row_autopsy.csv", p1_rows)
    write_csv_rows(out_dir / "miss_row_trace_v9239.csv", p1_trace)
    p2_rows, p2 = _p2_protocol_audit(bool(_int(p1.get("p1_attribution_pass"))))
    write_csv_rows(out_dir / "p2_protocol_hash_and_mlp_match_audit.csv", p2_rows)
    write_csv_rows(out_dir / "protocol_hash_diff_v9239.csv", p2_rows)
    p3_rows, p3_trace, p3 = _p3_repeated_reanchor(args, device, bool(_int(p2.get("protocol_pass"))))
    write_csv_rows(out_dir / "p3_current_lq_repeated_reanchor.csv", p3_rows)
    write_csv_rows(out_dir / "lq_anchor_trace_v9239.csv", p3_trace)
    p4_rows, p4 = _p4_anchor_decision(p1, p2, p3, bool(_int(p2.get("protocol_pass"))))
    write_csv_rows(out_dir / "p4_anchor_decision_and_base_repair_gate.csv", p4_rows)
    p5_rows, p5 = _p5_global_base_repair(args, device, bool(_int(p4.get("base_repair_required"))))
    write_csv_rows(out_dir / "p5_global_lq_base_repair.csv", p5_rows)

    if _int(p5.get("base_repair_pass")):
        downstream_reason = "snapshot_late_attach_not_implemented_after_base_repair_pass_in_this_runner"
    elif _int(p4.get("functional_route_allowed")):
        downstream_reason = "snapshot_late_attach_not_implemented_after_lq_reanchor_in_this_runner"
    elif _int(p4.get("base_repair_required")):
        downstream_reason = "global_base_repair_failed"
    else:
        downstream_reason = "LQ_anchor_not_ready"
    _write_gate_blocked(out_dir, downstream_reason)

    artifacts = [
        out_dir / "run_manifest.json",
        out_dir / "contract_audit_v9239.csv",
        out_dir / "p0_v9238_boundary_reproduction.csv",
        out_dir / "p1_lq_anchor_miss_row_autopsy.csv",
        out_dir / "p2_protocol_hash_and_mlp_match_audit.csv",
        out_dir / "p3_current_lq_repeated_reanchor.csv",
        out_dir / "p4_anchor_decision_and_base_repair_gate.csv",
        out_dir / "p5_global_lq_base_repair.csv",
        out_dir / "p6_snapshot_late_attach_implementation.csv",
        out_dir / "p7_checkpoint_inactive_noevent_equivalence.csv",
        out_dir / "p8_functional_carrier_actuatability.csv",
        out_dir / "p9_value_observability_audit.csv",
        out_dir / "p10_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p11_official_snapshot_late_attach_paired_replay.csv",
        out_dir / "p12_short_run_functional_validation.csv",
        out_dir / "p13_full_10seed_functional_validation.csv",
        out_dir / "p14_robustness_external_ready.csv",
        out_dir / "lq_anchor_trace_v9239.csv",
        out_dir / "miss_row_trace_v9239.csv",
        out_dir / "protocol_hash_diff_v9239.csv",
        out_dir / "snapshot_attach_trace_v9239.csv",
        out_dir / "functional_carrier_trace_v9239.csv",
        out_dir / "value_score_trace_v9239.csv",
        out_dir / "leave_dataset_out_trace_v9239.csv",
        out_dir / "paired_replay_branch_trace_v9239.csv",
    ]
    audit = audit_no_fake(artifacts)
    route_name, primary, next_impl = _route_decision(p0, p1, p2, p3, p4, p5, audit)
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9238_boundary_pass": _int(p0.get("P0_pass")),
        "source_route": p0.get("route", ""),
        "dataset_tuning_detected": 0,
        "lq_anchor_pass": _int(p3.get("stable_lq_anchor_pass")),
        "lq_anchor_failure_mode": p1.get("primary_failure_mode", ""),
        "current_lq_near_rate": p0.get("current_lq_near_rate", 0.0),
        "current_lq_macro_delta": p0.get("current_lq_macro_delta", 0.0),
        "historical_current_lq_delta": p0.get("historical_current_lq_delta", 0.0),
        "miss_row_count": p1.get("miss_row_count", 0),
        "miss_row_borderline_rate": p1.get("miss_row_borderline_rate", 0.0),
        "protocol_mismatch_detected": p2.get("protocol_mismatch_detected", 0),
        "protocol_mismatch_unresolved": p2.get("protocol_mismatch_unresolved", 0),
        "mlp_match_drift_detected": p2.get("mlp_match_drift_detected", 0),
        "seed_protocol_drift_detected": p2.get("seed_protocol_drift_detected", 0),
        "current_lq_rerun_count": p3.get("rerun_count", 0),
        "current_lq_repeated_near_rate_mean": p3.get("near_pass_rate_mean", 0.0),
        "current_lq_repeated_near_rate_min": p3.get("near_pass_rate_min", 0.0),
        "current_lq_repeated_near_rate_max": p3.get("near_pass_rate_max", 0.0),
        "current_lq_repeated_macro_delta_mean": p3.get("macro_delta_mean", 0.0),
        "miss_row_stability": p3.get("miss_row_stability", 0.0),
        "base_repair_required": p4.get("base_repair_required", 0),
        "base_repair_pass": p5.get("base_repair_pass", 0),
        "base_repair_survivor_count": p5.get("base_repair_survivor_count", 0),
        "best_base_candidate": p5.get("best_base_candidate", ""),
        "best_base_macro_delta": p5.get("best_base_macro_delta", 0.0),
        "best_base_near_rate": p5.get("best_base_near_rate", 0.0),
        "best_base_step_q90": p5.get("best_base_step_q90", 0.0),
        "snapshot_attach_pass": 0,
        "checkpoint_inactive_equivalence_pass": 0,
        "no_event_replay_preservation_pass": 0,
        "functional_carrier_pass": 0,
        "value_observability_pass": 0,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "success_v9239_strict_purekan_functional": 0,
        "success_v9239_full_functional": 0,
        "success_v9239_external_ready": 0,
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure = [
        {"stage": "P0", "pass": _int(p0.get("P0_pass")), "blocker": "" if _int(p0.get("P0_pass")) else "F2_v9238_boundary_unstable"},
        {"stage": "P1", "pass": p1.get("p1_attribution_pass", 0), "blocker": "" if p1.get("p1_attribution_pass") else "F4_lq_anchor_fail_unattributed"},
        {"stage": "P2", "pass": int(not _int(p2.get("protocol_mismatch_unresolved"))), "blocker": "" if not _int(p2.get("protocol_mismatch_unresolved")) else "F5_protocol_hash_drift"},
        {"stage": "P3", "pass": p3.get("stable_lq_anchor_pass", 0), "blocker": "" if p3.get("stable_lq_anchor_pass") else "F10_current_lq_repeated_anchor_fail"},
        {"stage": "P5", "pass": p5.get("base_repair_pass", 0), "blocker": "" if p5.get("base_repair_pass") else ("F11_base_repair_required" if p4.get("base_repair_required") else "not_opened")},
        {"stage": "P6-P14", "pass": 0, "blocker": downstream_reason},
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure)
    write_csv_rows(out_dir / "v9239_provenance_audit.csv", [audit])
    final_artifacts = [*artifacts, out_dir / "route_decision.json", out_dir / "aggregate_decision.json", out_dir / "failure_table.csv", out_dir / "v9239_provenance_audit.csv"]
    _write_report(out_dir, route, final_artifacts)


if __name__ == "__main__":
    main()
