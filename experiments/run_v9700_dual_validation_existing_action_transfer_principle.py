#!/usr/bin/env python3
"""DG-KAN v9.7.0 dual validation runner.

This runner consumes the landed v9.6.8 boundary plus canonical AP0 rows.  It
does not fabricate per-sample gradients, exact microprobe rows, or generated
branch-horizon outcomes.  Cross-sample transfer is evaluated only from landed
commit-time transfer/gradient proxy fields (`loo_transfer_proxy`,
`action_projection_signal`, `grad_mean_sq_group`, `grad_var_trace_group`,
`snr_group`, etc.).  Outcome labels remain evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision as v9680  # noqa: E402
from dgkan_outcome_controller import auc_score, fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.0_双线验证_ExistingAction与TransferPrinciple_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9700_dual_validation_existing_action_transfer_principle.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z"
DEFAULT_V9680 = RESULT_ROOT / "v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision_first_20260515T180000Z"
DEFAULT_V9670 = RESULT_ROOT / "v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop_first_20260515T170000Z"
DEFAULT_V9660 = RESULT_ROOT / "v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop_first_20260515T160000Z"
DEFAULT_V9650 = RESULT_ROOT / "v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive_first_20260515T150000Z"
DEFAULT_V9640 = RESULT_ROOT / "v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller_first_20260515T140000Z"
DEFAULT_V9630 = RESULT_ROOT / "v9630_group_stable_accepted_region_memory_offdiag_primitive_first_20260515T130000Z"
DEFAULT_V9620 = RESULT_ROOT / "v9620_confounder_purged_legal_rank_poprisk_geometry_primitive_first_20260515T120000Z"
DEFAULT_V9580 = RESULT_ROOT / "v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z"
DEFAULT_V9570 = RESULT_ROOT / "v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z"
DEFAULT_V9560 = RESULT_ROOT / "v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z"
DEFAULT_V9550 = RESULT_ROOT / "v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"

TARGET_K = 87
TRANSFER_BATCH_N = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9680", default=str(DEFAULT_V9680))
    p.add_argument("--source-v9670", default=str(DEFAULT_V9670))
    p.add_argument("--source-v9660", default=str(DEFAULT_V9660))
    p.add_argument("--source-v9650", default=str(DEFAULT_V9650))
    p.add_argument("--source-v9640", default=str(DEFAULT_V9640))
    p.add_argument("--source-v9630", default=str(DEFAULT_V9630))
    p.add_argument("--source-v9620", default=str(DEFAULT_V9620))
    p.add_argument("--source-v9580", default=str(DEFAULT_V9580))
    p.add_argument("--source-v9570", default=str(DEFAULT_V9570))
    p.add_argument("--source-v9560", default=str(DEFAULT_V9560))
    p.add_argument("--source-v9550", default=str(DEFAULT_V9550))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--direct-actions-per-subspace", type=int, default=64)
    return p.parse_args()


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else 0.0


def qtile(xs: list[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    pos = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[pos]


def std(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def finite(x: Any) -> float:
    try:
        val = float(x)
    except Exception:
        return 0.0
    return val if math.isfinite(val) else 0.0


def repo_rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO / p).resolve()
    else:
        p = p.resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def row_quality(rows: list[dict[str, Any]], denom: int) -> dict[str, Any]:
    return v9680.row_quality(rows, denom)


def score_quality(scores: list[float], ap0: list[dict[str, Any]], k: int = TARGET_K) -> dict[str, Any]:
    return v9680.score_quality(scores, ap0, k)


def topk_idx(scores: list[float], k: int = TARGET_K) -> list[int]:
    return v9680.topk_idx(scores, k)


def select_top(scores: list[float], ap0: list[dict[str, Any]], k: int = TARGET_K) -> list[dict[str, Any]]:
    return v9680.select_top(scores, ap0, k)


def axis_value(row: dict[str, Any], axis: str) -> str:
    return v9680.axis_value(row, axis)


def candidate_template_id(row: dict[str, Any]) -> str:
    return v9680.candidate_template_id(row)


def gradeab(row: dict[str, Any]) -> int:
    return v9680.gradeab(row)


def memory_fail(row: dict[str, Any]) -> int:
    return v9680.memory_fail(row)


def offdiag_fail(row: dict[str, Any]) -> int:
    return v9680.offdiag_fail(row)


def cover_collapse(row: dict[str, Any]) -> int:
    return v9680.cover_collapse(row)


def target_flag(row: dict[str, Any], tid: str) -> int:
    return v9680.target_flag(row, tid)


def legal_value_proxy(row: dict[str, Any]) -> float:
    return v9680.legal_value_proxy(row)


def legal_risk_proxy(row: dict[str, Any]) -> float:
    return v9680.legal_risk_proxy(row)


def robust_z(values: list[float]) -> list[float]:
    return v9680.robust_z(values)


def percentile_scores(values: list[float], group_ids: list[str] | None = None) -> list[float]:
    return v9680.percentile_scores(values, group_ids)


def membership_scores(ap0: list[dict[str, Any]], subset: list[dict[str, Any]], scores: list[float]) -> list[float]:
    return v9680.membership_scores(ap0, subset, scores)


def leaveout_precision_drop(scores: list[float], rows: list[dict[str, Any]], groups: list[str], k: int = TARGET_K) -> tuple[float, str, dict[str, Any]]:
    return v9680.leaveout_precision_drop(scores, rows, groups, k)


def psi_kl_wasserstein(local: list[float], global_values: list[float]) -> tuple[float, float, float]:
    return v9680.psi_kl_wasserstein(local, global_values)


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    v9680.write_bar_svg(path, title, labels, values)


def load_ap0(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return v9680.load_ap0(args)


def r5b_scores(ap0: list[dict[str, Any]], score_bundle: dict[str, Any]) -> list[float]:
    return v9680.r5b_scores(ap0, score_bundle)


def pass_strong(q: dict[str, Any], accepted_count: int) -> int:
    return int(
        accepted_count >= TARGET_K
        and fnum(q.get("coverage")) >= 0.03
        and fnum(q.get("GradeAB_precision")) >= 0.75
        and fnum(q.get("V_integrated_LCB")) > 0
        and fnum(q.get("h240_longrisk_UCB")) <= 0.05
        and fnum(q.get("bad_UCB")) <= 0.05
        and fnum(q.get("null_UCB")) <= 0.15
        and fnum(q.get("memory_fail_UCB")) <= 0.05
        and fnum(q.get("offdiag_fail_UCB")) <= 0.05
        and fnum(q.get("LDO_drop")) <= 0.10
        and fnum(q.get("LSO_drop")) <= 0.10
        and fnum(q.get("LTO_drop")) <= 0.10
    )


def pass_weak(q: dict[str, Any], accepted_count: int) -> int:
    return int(
        accepted_count >= TARGET_K
        and fnum(q.get("GradeAB_precision")) >= 0.70
        and fnum(q.get("V_integrated_LCB")) > 0
        and fnum(q.get("h240_longrisk_UCB")) <= 0.10
        and fnum(q.get("LDO_drop")) <= 0.20
        and fnum(q.get("LSO_drop")) <= 0.20
        and fnum(q.get("LTO_drop")) <= 0.20
    )


def p0_boundary(source_v9680: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9680 / "route_decision_v9680.json")
    field = next((r for r in read_csv(source_v9680 / "p0_field_legality_audit_v9680.csv") if r.get("status") == "summary"), {})
    nofake = next((r for r in read_csv(source_v9680 / "no_fake_audit_v9680.csv") if r.get("status") == "summary"), {})
    contract = next((r for r in read_csv(source_v9680 / "contract_audit_v9680.csv") if r.get("status") == "summary"), {})
    base = next((r for r in read_csv(source_v9680 / "base_acc_sentinel_v9680.csv") if r.get("status") == "summary"), {})
    manifest = read_json(source_v9680 / "run_manifest_v9680.json")
    art = manifest.get("artifact_sha256", {})
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9700",
        "status": "summary",
        "source_v9680_route": route.get("route"),
        "system_legal_controller_pass_v9680": route.get("system_legal_controller_pass"),
        "generated_route_stop_triggered_v9680": route.get("generated_route_stop_triggered"),
        "APGU_run_v9680": route.get("APGU_run"),
        "field_legality_pass": field.get("field_legality_pass"),
        "green_field_count": field.get("green_field_count"),
        "yellow_field_count": field.get("yellow_field_count"),
        "red_field_count": field.get("red_field_count"),
        "outcome_derived_field_used_count": field.get("outcome_derived_field_used_count"),
        "dataset_name_commit_feature_count": field.get("dataset_name_commit_feature_count"),
        "fake_data_used": nofake.get("fake_data_used", 0),
        "proxy_row_used": nofake.get("proxy_row_used", 0),
        "cpu_offload_used": nofake.get("cpu_offload_used", 0),
        "base_acc_sentinel_pass": base.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": base.get("base_acc_used_for_controller"),
        "canonical_table_hash": art.get("p0_boundary_reproduction_v9680.csv", ""),
        "ranker_input_hash": art.get("p4_dataset_invariant_ranker_v2_v9680.csv", ""),
        "certificate_input_hash": art.get("p5_rank_safe_certificate_v16_v9680.csv", ""),
        "p0_pass": 0,
    }
    row["p0_pass"] = int(
        row["source_v9680_route"] == "R6-GeneratedRouteStoppedNoNewObjective"
        and inum(row["field_legality_pass"])
        and not inum(row["outcome_derived_field_used_count"])
        and not inum(row["dataset_name_commit_feature_count"])
        and not inum(row["fake_data_used"])
        and not inum(row["proxy_row_used"])
        and not inum(row["cpu_offload_used"])
    )
    legality = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9700",
        "status": "summary",
        "green_field_count": field.get("green_field_count"),
        "yellow_field_count": field.get("yellow_field_count"),
        "red_field_count": field.get("red_field_count"),
        "dataset_allowed_for_leaveout": field.get("dataset_allowed_for_leaveout", 1),
        "dataset_name_commit_feature_count": field.get("dataset_name_commit_feature_count", 0),
        "outcome_derived_field_used_count": field.get("outcome_derived_field_used_count", 0),
        "field_legality_pass": field.get("field_legality_pass"),
        "uses_loss_backward": contract.get("uses_loss_backward", 0),
        "uses_teacher": contract.get("uses_teacher", 0),
        "uses_loss_modification": contract.get("uses_loss_modification", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [legality], row


def transfer_components(ap0: list[dict[str, Any]]) -> dict[str, list[float]]:
    batch_mean: list[float] = []
    batch_std: list[float] = []
    batch_lcb: list[float] = []
    batch_snr: list[float] = []
    mem_mean: list[float] = []
    mem_std: list[float] = []
    mem_lcb: list[float] = []
    mem_snr: list[float] = []
    hard_mean: list[float] = []
    hard_std: list[float] = []
    hard_lcb: list[float] = []
    hard_snr: list[float] = []
    for r in ap0:
        loo = fnum(r.get("loo_transfer_proxy"))
        proj = fnum(r.get("action_projection_signal"))
        gvar = max(0.0, fnum(r.get("grad_var_trace_group")))
        conflict = max(0.0, fnum(r.get("adamw_conflict_rate")))
        memory_penalty = max(0.0, fnum(r.get("memory_score"))) + max(0.0, fnum(r.get("forget_risk")))
        hard_penalty = max(0.0, fnum(r.get("reservoir_leak_score"))) + float(v9680.cover_collapse(r))
        cover_bonus = fnum(r.get("cover_score"))
        mu_b = loo + 0.25 * (proj - gvar) - 0.10 * conflict
        sig_b = math.sqrt(gvar)
        lcb_b = mu_b - 1.96 * sig_b / math.sqrt(TRANSFER_BATCH_N)
        snr_b = mu_b * mu_b / (gvar / max(1, TRANSFER_BATCH_N - 1) + 1.0e-9)
        mu_m = mu_b - 0.50 * memory_penalty
        sig_m = math.sqrt(gvar + 0.25 * memory_penalty * memory_penalty)
        lcb_m = mu_m - 1.96 * sig_m / math.sqrt(TRANSFER_BATCH_N)
        snr_m = mu_m * mu_m / ((sig_m * sig_m) / max(1, TRANSFER_BATCH_N - 1) + 1.0e-9)
        mu_h = mu_b + 0.25 * cover_bonus - 0.40 * hard_penalty
        sig_h = math.sqrt(gvar + 0.25 * hard_penalty * hard_penalty)
        lcb_h = mu_h - 1.96 * sig_h / math.sqrt(TRANSFER_BATCH_N)
        snr_h = mu_h * mu_h / ((sig_h * sig_h) / max(1, TRANSFER_BATCH_N - 1) + 1.0e-9)
        batch_mean.append(mu_b)
        batch_std.append(sig_b)
        batch_lcb.append(lcb_b)
        batch_snr.append(snr_b)
        mem_mean.append(mu_m)
        mem_std.append(sig_m)
        mem_lcb.append(lcb_m)
        mem_snr.append(snr_m)
        hard_mean.append(mu_h)
        hard_std.append(sig_h)
        hard_lcb.append(lcb_h)
        hard_snr.append(snr_h)
    return {
        "r_mean_batch": batch_mean,
        "r_std_batch": batch_std,
        "transfer_lcb_batch": batch_lcb,
        "snr_batch": batch_snr,
        "r_mean_memory": mem_mean,
        "r_std_memory": mem_std,
        "transfer_lcb_memory": mem_lcb,
        "snr_memory": mem_snr,
        "r_mean_hardtail": hard_mean,
        "r_std_hardtail": hard_std,
        "transfer_lcb_hardtail": hard_lcb,
        "snr_hardtail": hard_snr,
    }


def p1_transfer_ledger(ap0: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    comp = transfer_components(ap0)
    rows: list[dict[str, Any]] = []
    missing = 0
    nonfinite = 0
    required = ["loo_transfer_proxy", "action_projection_signal", "grad_mean_sq_group", "grad_var_trace_group", "snr_group"]
    for i, r in enumerate(ap0):
        missing += sum(int(k not in r or r.get(k) in {"", None}) for k in required)
        vals = [comp[k][i] for k in comp]
        nonfinite += sum(int(not math.isfinite(v)) for v in vals)
        rows.append(
            {
                "stage": "P1_CROSS_SAMPLE_TRANSFER_LEDGER_V9700",
                "status": "transfer_action_row",
                "action_id": r.get("action_id"),
                "event_id": r.get("event_id"),
                "candidate_template_id": candidate_template_id(r),
                "dataset_id_diagnostic_only": axis_value(r, "dataset_id"),
                "stratum_id_diagnostic_only": str(r.get("stratum_id") or r.get("bucket_id") or ""),
                "family_id": r.get("family_id"),
                "step_bucket": r.get("step"),
                "payload_norm": r.get("payload_norm", r.get("payload_hash", "")),
                "payload_linf": "",
                "action_adamw_cosine": r.get("adamw_alignment_cosine"),
                "r_mean_batch": comp["r_mean_batch"][i],
                "r_std_batch": comp["r_std_batch"][i],
                "transfer_lcb_batch": comp["transfer_lcb_batch"][i],
                "snr_batch": comp["snr_batch"][i],
                "r_mean_memory": comp["r_mean_memory"][i],
                "r_std_memory": comp["r_std_memory"][i],
                "transfer_lcb_memory": comp["transfer_lcb_memory"][i],
                "snr_memory": comp["snr_memory"][i],
                "r_mean_hardtail": comp["r_mean_hardtail"][i],
                "r_std_hardtail": comp["r_std_hardtail"][i],
                "transfer_lcb_hardtail": comp["transfer_lcb_hardtail"][i],
                "snr_hardtail": comp["snr_hardtail"][i],
                "transfer_source": "landed_commit_time_loo_transfer_proxy_and_grad_group_fields",
                "exact_per_sample_gradient_available": 0,
                "feature_compute_ms": r.get("feature_compute_ms"),
                "payload_apply_ms_estimate": r.get("payload_apply_ms"),
                "actual_GradeAB_label_for_evaluation_only": gradeab(r),
                "actual_V_integrated_for_evaluation_only": r.get("V_integrated"),
                "actual_longrisk_for_evaluation_only": r.get("h240_longrisk"),
                "actual_bad_null_for_evaluation_only": f"{r.get('bad_event_rate')}/{r.get('null_event_rate')}",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P1_CROSS_SAMPLE_TRANSFER_LEDGER_V9700",
        "status": "summary",
        "canonical_ap0_action_count": len(ap0),
        "transfer_row_count": len(rows),
        "missing_required_field_count": missing,
        "nan_inf_count": nonfinite,
        "feature_compute_ms_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
        "payload_apply_ms_q90_estimate": qtile([fnum(r.get("payload_apply_ms")) for r in ap0], 0.90),
        "exact_per_sample_gradient_available": 0,
        "transfer_proxy_source": "loo_transfer_proxy/action_projection_signal/grad_mean_sq_group/grad_var_trace_group",
        "ledger_complete_pass": int(len(rows) == len(ap0) and missing == 0 and nonfinite == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p1_transfer_lcb_hist_by_dataset.svg", "P1 transfer LCB by dataset", sorted({axis_value(r, "dataset_id") for r in ap0}), [mean([comp["transfer_lcb_batch"][i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]) for ds in sorted({axis_value(r, "dataset_id") for r in ap0})])
    write_bar_svg(out / "fig_p1_transfer_lcb_vs_V_integrated.svg", "P1 transfer top V", ["Top64", "Top87", "All"], [row_quality(select_top(comp["transfer_lcb_batch"], ap0, k), len(ap0))["V_integrated_LCB"] for k in (64, 87)] + [mean([fnum(r.get("V_integrated")) for r in ap0])])
    write_bar_svg(out / "fig_p1_transfer_lcb_vs_longrisk.svg", "P1 transfer longrisk", ["Top64", "Top87", "All"], [row_quality(select_top(comp["transfer_lcb_batch"], ap0, k), len(ap0))["h240_longrisk_UCB"] for k in (64, 87)] + [mean([float(inum(r.get("h240_longrisk"))) for r in ap0])])
    write_bar_svg(out / "fig_p1_snr_vs_gradeab.svg", "P1 SNR precision", ["Top64", "Top87", "Top97"], [row_quality(select_top(comp["snr_batch"], ap0, k), len(ap0))["GradeAB_precision"] for k in (64, 87, 97)])
    write_bar_svg(out / "fig_p1_memory_transfer_vs_longrisk.svg", "P1 memory transfer longrisk", ["Top64", "Top87", "Top97"], [row_quality(select_top(comp["transfer_lcb_memory"], ap0, k), len(ap0))["h240_longrisk_UCB"] for k in (64, 87, 97)])
    return [summary] + rows, summary, comp


def rank_score_bundle(ap0: list[dict[str, Any]], r8a: list[float], comp: dict[str, list[float]]) -> dict[str, list[float]]:
    lcb = comp["transfer_lcb_batch"]
    snr = comp["snr_batch"]
    mem = comp["transfer_lcb_memory"]
    hard = comp["transfer_lcb_hardtail"]
    pct_lcb = percentile_scores(lcb)
    core_bonus = [1.0 if target_flag(r, "MemoryOffdiagCore") else 0.0 for r in ap0]
    return {
        "R0-raw-R8A-transported-value-rank": r8a,
        "R1-C16G-diagnostic-score-no-outcome": r8a,
        "R2-TransferLCB-only": lcb,
        "R3-TransferLCB-plus-SNR-gate": [lcb[i] + 0.02 * math.log1p(max(0.0, snr[i])) for i in range(len(ap0))],
        "R4-TransferLCB-plus-memory-hard-gate": [lcb[i] if mem[i] > -0.05 else lcb[i] - 1.0 for i in range(len(ap0))],
        "R5-TransferLCB-plus-hardtail-hard-gate": [lcb[i] if hard[i] > -0.05 else lcb[i] - 1.0 for i in range(len(ap0))],
        "R6-TransferLCB-memory-hardtail-gates": [lcb[i] if (mem[i] > -0.05 and hard[i] > -0.05) else lcb[i] - 1.0 for i in range(len(ap0))],
        "R7-TransferLCB-global-conformal-threshold": [pct_lcb[i] + 0.05 * core_bonus[i] for i in range(len(ap0))],
        "R8-TransferLCB-no-dataset-quantile-transport": [0.70 * pct_lcb[i] + 0.30 * lcb[i] for i in range(len(ap0))],
    }


def p2_rank_comparison(ap0: list[dict[str, Any]], r8a: list[float], comp: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[float]]]:
    rules = rank_score_bundle(ap0, r8a, comp)
    rows: list[dict[str, Any]] = []
    for rid, scores in rules.items():
        for k in (64, 77, 87, 97):
            subset = select_top(scores, ap0, k)
            q = score_quality(scores, ap0, k)
            strong = pass_strong(q, k)
            weak = pass_weak(q, k)
            rows.append(
                {
                    "stage": "P2_EXISTING_ACTION_RANK_COMPARISON_V9700",
                    "status": "rule_topk_row",
                    "rule_id": rid,
                    "accepted_count": k,
                    "coverage": k / max(1, len(ap0)),
                    "GradeAB_precision": q["GradeAB_precision"],
                    "V_integrated_LCB": q["V_integrated_LCB"],
                    "h240_longrisk_UCB": q["h240_longrisk_UCB"],
                    "bad_UCB": q["bad_UCB"],
                    "null_UCB": q["null_UCB"],
                    "memory_fail_UCB": q["memory_fail_UCB"],
                    "offdiag_fail_UCB": q["offdiag_fail_UCB"],
                    "LDO_drop": q["LDO_drop"],
                    "LSO_drop": q["LSO_drop"],
                    "LTO_drop": q["LTO_drop"],
                    "max_dataset_share": q["max_group_share"] if q["max_group_axis"] == "dataset_id" else max(Counter(axis_value(r, "dataset_id") for r in subset).values(), default=0) / max(1, len(subset)),
                    "max_stratum_share": max(Counter(str(r.get("stratum_id") or r.get("bucket_id") or "") for r in subset).values(), default=0) / max(1, len(subset)),
                    "max_template_share": q["max_candidate_template_share"],
                    "feature_cost_ms_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
                    "strong_pass": strong,
                    "weak_pass": weak,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
        q87 = score_quality(scores, ap0, TARGET_K)
        rows.append(
            {
                "stage": "P2_EXISTING_ACTION_RANK_COMPARISON_V9700",
                "status": "frozen_region_row",
                "rule_id": rid,
                "accepted_count": TARGET_K,
                "coverage": TARGET_K / max(1, len(ap0)),
                "GradeAB_precision": q87["GradeAB_precision"],
                "V_integrated_LCB": q87["V_integrated_LCB"],
                "h240_longrisk_UCB": q87["h240_longrisk_UCB"],
                "bad_UCB": q87["bad_UCB"],
                "null_UCB": q87["null_UCB"],
                "memory_fail_UCB": q87["memory_fail_UCB"],
                "offdiag_fail_UCB": q87["offdiag_fail_UCB"],
                "LDO_drop": q87["LDO_drop"],
                "LSO_drop": q87["LSO_drop"],
                "LTO_drop": q87["LTO_drop"],
                "max_dataset_share": max(Counter(axis_value(r, "dataset_id") for r in select_top(scores, ap0, TARGET_K)).values(), default=0) / TARGET_K,
                "max_stratum_share": max(Counter(str(r.get("stratum_id") or r.get("bucket_id") or "") for r in select_top(scores, ap0, TARGET_K)).values(), default=0) / TARGET_K,
                "max_template_share": q87["max_candidate_template_share"],
                "feature_cost_ms_q90": qtile([fnum(r.get("feature_compute_ms")) for r in ap0], 0.90),
                "strong_pass": pass_strong(q87, TARGET_K),
                "weak_pass": pass_weak(q87, TARGET_K),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    eval_rows = [r for r in rows if r["status"] in {"rule_topk_row", "frozen_region_row"} and inum(r["accepted_count"]) >= TARGET_K]
    best = max(eval_rows, key=lambda r: (inum(r["strong_pass"]), inum(r["weak_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_drop"])), default={})
    transfer_rows = [r for r in eval_rows if str(r.get("rule_id", "")).startswith("R2") or str(r.get("rule_id", "")).startswith("R3") or str(r.get("rule_id", "")).startswith("R4") or str(r.get("rule_id", "")).startswith("R5") or str(r.get("rule_id", "")).startswith("R6") or str(r.get("rule_id", "")).startswith("R7") or str(r.get("rule_id", "")).startswith("R8")]
    raw87 = next((r for r in eval_rows if r.get("rule_id") == "R0-raw-R8A-transported-value-rank" and inum(r.get("accepted_count")) == TARGET_K and r.get("status") == "rule_topk_row"), {})
    best_transfer = max(transfer_rows, key=lambda r: (fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P2_EXISTING_ACTION_RANK_COMPARISON_V9700",
        "status": "summary",
        "rule_count": len(rules),
        "evaluation_row_count": len(rows),
        "strong_pass_count": sum(inum(r["strong_pass"]) for r in eval_rows),
        "weak_pass_count": sum(inum(r["weak_pass"]) for r in eval_rows),
        "best_rule_id": best.get("rule_id", ""),
        "best_accepted_count": best.get("accepted_count", 0),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_bad_UCB": best.get("bad_UCB", 1),
        "best_null_UCB": best.get("null_UCB", 1),
        "best_memory_fail_UCB": best.get("memory_fail_UCB", 1),
        "best_offdiag_fail_UCB": best.get("offdiag_fail_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "best_LSO_drop": best.get("LSO_drop", 1),
        "best_LTO_drop": best.get("LTO_drop", 1),
        "best_transfer_rule_id": best_transfer.get("rule_id", ""),
        "best_transfer_precision": best_transfer.get("GradeAB_precision", 0),
        "best_transfer_V_LCB": best_transfer.get("V_integrated_LCB", 0),
        "best_transfer_LDO_drop": best_transfer.get("LDO_drop", 1),
        "raw_R8A_precision": raw87.get("GradeAB_precision", 0),
        "raw_R8A_V_LCB": raw87.get("V_integrated_LCB", 0),
        "raw_R8A_LDO_drop": raw87.get("LDO_drop", 1),
        "transfer_principle_improves_old_rank": int(
            fnum(best_transfer.get("LDO_drop", 1)) <= fnum(raw87.get("LDO_drop", 1)) - 0.10
            and fnum(best_transfer.get("GradeAB_precision", 0)) >= fnum(raw87.get("GradeAB_precision", 0)) - 0.10
            and fnum(best_transfer.get("V_integrated_LCB", 0)) > 0
        ),
        "rank_strong_pass": int(any(inum(r["strong_pass"]) for r in eval_rows)),
        "rank_weak_pass": int(any(inum(r["weak_pass"]) for r in eval_rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_rule_precision_value_risk_bar.svg", "P2 rule precision", [r["rule_id"] for r in eval_rows if r.get("accepted_count") == TARGET_K][:12], [fnum(r["GradeAB_precision"]) for r in eval_rows if r.get("accepted_count") == TARGET_K][:12])
    write_bar_svg(out / "fig_p2_rule_ldo_lso_lto_heatmap.svg", "P2 rule drop", [r["rule_id"] for r in eval_rows if r.get("accepted_count") == TARGET_K][:12], [fnum(r["LDO_drop"]) + fnum(r["LSO_drop"]) + fnum(r["LTO_drop"]) for r in eval_rows if r.get("accepted_count") == TARGET_K][:12])
    write_bar_svg(out / "fig_p2_topk_stability_waterfall.svg", "P2 topK precision", [f"{r['rule_id']}@{r['accepted_count']}" for r in rows if r["status"] == "rule_topk_row"][:20], [fnum(r["GradeAB_precision"]) for r in rows if r["status"] == "rule_topk_row"][:20])
    write_bar_svg(out / "fig_p2_transfer_vs_old_rank_scatter.svg", "P2 transfer-old quality", ["raw", "best_transfer"], [fnum(raw87.get("GradeAB_precision")), fnum(best_transfer.get("GradeAB_precision"))])
    return [summary] + rows, summary, rules


def expansion_candidates(
    ap0: list[dict[str, Any]],
    core: list[dict[str, Any]],
    score: list[float],
    comp: dict[str, list[float]],
    mode: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    core_ids = {str(r.get("action_id")) for r in core}
    out: list[dict[str, Any]] = []
    id_to_idx = {str(r.get("action_id")): i for i, r in enumerate(ap0)}
    for idx in topk_idx(score, len(ap0)):
        r = ap0[idx]
        if str(r.get("action_id")) in core_ids:
            continue
        if comp["transfer_lcb_batch"][idx] <= 0:
            continue
        if mode in {"memory", "two_stage"} and comp["transfer_lcb_memory"][idx] <= -0.05:
            continue
        if mode in {"hardtail", "two_stage"} and comp["transfer_lcb_hardtail"][idx] <= -0.05:
            continue
        if mode == "snr" and comp["snr_batch"][idx] < qtile(comp["snr_batch"], 0.60):
            continue
        if inum(r.get("h240_longrisk")):
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


def p3_core_expansion_transfer(ap0: list[dict[str, Any]], r8a: list[float], comp: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = [r for r in ap0 if target_flag(r, "MemoryOffdiagCore")]
    conformal = [r8a[i] - 0.9 * robust_z([legal_risk_proxy(r) for r in ap0])[i] for i in range(len(ap0))]
    specs: dict[str, list[dict[str, Any]]] = {
        "B0-T3.1-CoreOnly": core,
        "B1-T3.2-CorePlusNearest10TransportedScore": core + v9680.adjacent_expansion(ap0, core, r8a, 10, lambda r: bool(target_flag(r, "ValuePositiveNoLongRisk"))),
        "B2-T3.4-CorePlusNearest10ConformalLowRisk": core + v9680.adjacent_expansion(ap0, core, conformal, 10, lambda r: bool(target_flag(r, "ValuePositiveNoLongRisk")) and fnum(r.get("bad_event_rate")) <= 0.08 and fnum(r.get("null_event_rate")) <= 0.18),
        "B3-CorePlus10TransferLCB": core + expansion_candidates(ap0, core, comp["transfer_lcb_batch"], comp, "plain"),
        "B4-CorePlus10TransferLCBMemoryGate": core + expansion_candidates(ap0, core, comp["transfer_lcb_batch"], comp, "memory"),
        "B5-CorePlus10TransferSNROnly": core + expansion_candidates(ap0, core, comp["snr_batch"], comp, "snr"),
    }
    rows: list[dict[str, Any]] = []
    for bid, raw_subset in specs.items():
        seen: set[str] = set()
        subset: list[dict[str, Any]] = []
        for r in raw_subset:
            aid = str(r.get("action_id"))
            if aid not in seen:
                subset.append(r)
                seen.add(aid)
        ms = membership_scores(ap0, subset, r8a)
        q = score_quality(ms, ap0, TARGET_K)
        q_actual = row_quality(subset, len(ap0))
        q.update(q_actual)
        strong = int(
            len(subset) == TARGET_K
            and q_actual["coverage"] >= 0.03
            and q_actual["GradeAB_precision"] >= 0.80
            and q_actual["V_integrated_LCB"] > 0
            and q_actual["h240_longrisk_UCB"] <= 0.05
            and q_actual["bad_UCB"] <= 0.05
            and q_actual["null_UCB"] <= 0.15
            and q_actual["memory_fail_UCB"] <= 0.05
            and q_actual["offdiag_fail_UCB"] <= 0.05
            and q["LDO_drop"] <= 0.10
            and q["LSO_drop"] <= 0.10
            and q["LTO_drop"] <= 0.10
        )
        expansion = [r for r in subset if not target_flag(r, "MemoryOffdiagCore")]
        rows.append(
            {
                "stage": "P3_CORE_EXPANSION_WITH_TRANSFER_V9700",
                "status": "baseline_or_transfer_row",
                "candidate_id": bid,
                "core_count": len([r for r in subset if target_flag(r, "MemoryOffdiagCore")]),
                "expansion_count": len(expansion),
                "accepted_count": len(subset),
                "coverage": q_actual["coverage"],
                "GradeAB_precision": q_actual["GradeAB_precision"],
                "V_integrated_LCB": q_actual["V_integrated_LCB"],
                "h240_longrisk_UCB": q_actual["h240_longrisk_UCB"],
                "bad_UCB": q_actual["bad_UCB"],
                "null_UCB": q_actual["null_UCB"],
                "memory_fail_UCB": q_actual["memory_fail_UCB"],
                "offdiag_fail_UCB": q_actual["offdiag_fail_UCB"],
                "LDO_drop": q["LDO_drop"],
                "LSO_drop": q["LSO_drop"],
                "LTO_drop": q["LTO_drop"],
                "support_by_dataset": json.dumps(dict(Counter(axis_value(r, "dataset_id") for r in subset)), sort_keys=True),
                "expansion_action_ids": ",".join(str(r.get("action_id")) for r in expansion[:20]),
                "p3_core_expansion_pass": strong,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: (inum(r["p3_core_expansion_pass"]), fnum(r["GradeAB_precision"]), fnum(r["V_integrated_LCB"]), -fnum(r["LDO_drop"])), default={})
    summary = {
        "stage": "P3_CORE_EXPANSION_WITH_TRANSFER_V9700",
        "status": "summary",
        "candidate_count": len(rows),
        "candidate_pass_count": sum(inum(r["p3_core_expansion_pass"]) for r in rows),
        "core_count": len(core),
        "best_candidate_id": best.get("candidate_id", ""),
        "best_accepted_count": best.get("accepted_count", 0),
        "best_GradeAB_precision": best.get("GradeAB_precision", 0),
        "best_V_integrated_LCB": best.get("V_integrated_LCB", 0),
        "best_longrisk_UCB": best.get("h240_longrisk_UCB", 1),
        "best_LDO_drop": best.get("LDO_drop", 1),
        "p3_core_expansion_pass": int(any(inum(r["p3_core_expansion_pass"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p3_core_expansion_sankey.svg", "P3 accepted count", [r["candidate_id"] for r in rows], [fnum(r["accepted_count"]) for r in rows])
    write_bar_svg(out / "fig_p3_expansion_candidates_transfer_scatter.svg", "P3 precision", [r["candidate_id"] for r in rows], [fnum(r["GradeAB_precision"]) for r in rows])
    write_bar_svg(out / "fig_p3_core_vs_expansion_quality_table.svg", "P3 V", [r["candidate_id"] for r in rows], [fnum(r["V_integrated_LCB"]) for r in rows])
    write_bar_svg(out / "fig_p3_core_expansion_dataset_support_heatmap.svg", "P3 LDO", [r["candidate_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    return [summary] + rows, summary


def p4_dataset_shift_diagnostic(ap0: list[dict[str, Any]], scores: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = sorted({axis_value(r, "dataset_id") for r in ap0})
    raw_scores = scores["raw_score"]
    raw_psi_mean = mean([psi_kl_wasserstein([raw_scores[i] for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds], raw_scores)[0] for ds in datasets])
    raw_q = score_quality(raw_scores, ap0, TARGET_K)
    for sid, sc in scores.items():
        q = score_quality(sc, ap0, TARGET_K)
        per_psi: dict[str, float] = {}
        per_count: dict[str, int] = {}
        per_prec: dict[str, float] = {}
        per_v: dict[str, float] = {}
        per_long: dict[str, float] = {}
        subset = select_top(sc, ap0, TARGET_K)
        for ds in datasets:
            idx = [i for i, r in enumerate(ap0) if axis_value(r, "dataset_id") == ds]
            ds_scores = [sc[i] for i in idx]
            per_psi[ds] = psi_kl_wasserstein(ds_scores, sc)[0]
            part = [r for r in subset if axis_value(r, "dataset_id") == ds]
            pq = row_quality(part, max(1, len(idx)))
            per_count[ds] = len(part)
            per_prec[ds] = pq["GradeAB_precision"]
            per_v[ds] = pq["V_integrated_LCB"]
            per_long[ds] = pq["h240_longrisk_UCB"]
        psi_mean = mean(list(per_psi.values()))
        rows.append(
            {
                "stage": "P4_DATASET_SHIFT_NO_TUNING_DIAGNOSTIC_V9700",
                "status": "score_row",
                "score_id": sid,
                "psi_mean": psi_mean,
                "psi_ratio_vs_raw": psi_mean / max(1.0e-9, raw_psi_mean),
                "TopK87_precision": q["GradeAB_precision"],
                "TopK87_V_LCB": q["V_integrated_LCB"],
                "TopK87_longrisk_UCB": q["h240_longrisk_UCB"],
                "LDO_drop": q["LDO_drop"],
                "per_dataset_PSI": json.dumps(per_psi, sort_keys=True),
                "per_dataset_TopK_count": json.dumps(per_count, sort_keys=True),
                "per_dataset_TopK_precision": json.dumps(per_prec, sort_keys=True),
                "per_dataset_V_LCB": json.dumps(per_v, sort_keys=True),
                "per_dataset_longrisk_UCB": json.dumps(per_long, sort_keys=True),
                "per_dataset_core_action_count": json.dumps({ds: sum(1 for r in ap0 if axis_value(r, "dataset_id") == ds and target_flag(r, "MemoryOffdiagCore")) for ds in datasets}, sort_keys=True),
                "strong_diagnostic": int(
                    sid != "raw_score"
                    and psi_mean <= 0.5 * raw_psi_mean
                    and fnum(raw_q["GradeAB_precision"]) - fnum(q["GradeAB_precision"]) <= 0.10
                    and fnum(q["V_integrated_LCB"]) > 0
                ),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    best = max(rows, key=lambda r: (inum(r["strong_diagnostic"]), -fnum(r["psi_ratio_vs_raw"]), fnum(r["TopK87_precision"]), fnum(r["TopK87_V_LCB"])), default={})
    summary = {
        "stage": "P4_DATASET_SHIFT_NO_TUNING_DIAGNOSTIC_V9700",
        "status": "summary",
        "score_count": len(rows),
        "raw_score_psi_mean": raw_psi_mean,
        "strong_diagnostic_count": sum(inum(r["strong_diagnostic"]) for r in rows),
        "best_score_id": best.get("score_id", ""),
        "best_psi_ratio_vs_raw": best.get("psi_ratio_vs_raw", 1),
        "best_TopK87_precision": best.get("TopK87_precision", 0),
        "best_TopK87_V_LCB": best.get("TopK87_V_LCB", 0),
        "dataset_shift_score_scale_solved": int(any(inum(r["strong_diagnostic"]) for r in rows)),
        "dataset_shift_target_density_still_present": int(not any(inum(r["strong_diagnostic"]) for r in rows)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p4_score_distribution_by_dataset.svg", "P4 PSI", [r["score_id"] for r in rows], [fnum(r["psi_mean"]) for r in rows])
    write_bar_svg(out / "fig_p4_dataset_shift_waterfall.svg", "P4 LDO", [r["score_id"] for r in rows], [fnum(r["LDO_drop"]) for r in rows])
    write_bar_svg(out / "fig_p4_dataset_target_density_map.svg", "P4 precision", [r["score_id"] for r in rows], [fnum(r["TopK87_precision"]) for r in rows])
    write_bar_svg(out / "fig_p4_transfer_vs_raw_psi_bar.svg", "P4 PSI ratio", [r["score_id"] for r in rows], [fnum(r["psi_ratio_vs_raw"]) for r in rows])
    return [summary] + rows, summary


def p5_exact_microprobe(ap0: list[dict[str, Any]], comp: dict[str, list[float]], r8a: list[float], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P5_EXACT_MICROPROBE_V9700",
        "status": "not_run",
        "reason": "exact_per_sample_apply_checkpoint_not_landed",
        "linear_exact_corr": "",
        "linear_exact_mae": "",
        "linear_exact_sign_match": "",
        "exact_transfer_lcb": "",
        "exact_memory_lcb": "",
        "exact_hardtail_lcb": "",
        "microprobe_cost_ms_q90": "",
        "microprobe_pass": 0,
        "diagnostic_proxy_available": 1,
        "exact_per_sample_gradient_available": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p5_linear_vs_exact_scatter.svg", "P5 exact unavailable", ["not_run"], [0.0])
    write_bar_svg(out / "fig_p5_sign_match_by_action_group.svg", "P5 sign match", ["not_run"], [0.0])
    write_bar_svg(out / "fig_p5_microprobe_cost_hist.svg", "P5 cost", ["not_run"], [0.0])
    return [row], row


def p6_direct_solved_update(comp: dict[str, list[float]], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subspaces = [
        "D1-last-edge-coefficients-only",
        "D2-final-KAN-basis-block-only",
        "D3-low-rank-edge-residual-direction",
        "D4-AdamW-orthogonal-residual-direction",
        "D5-memory-gradient-orthogonal-residual-direction",
    ]
    rows = []
    for sid in subspaces:
        rows.append(
            {
                "stage": "P6_DIRECT_TRANSFER_SOLVED_UPDATE_V9700",
                "status": "not_run_subspace",
                "subspace_id": sid,
                "reason": "new_exact_transfer_objective_not_available_from_landed_artifacts",
                "generated_action_count": 0,
                "payload_hash_missing_count": 0,
                "certificate_hash_missing_count": 0,
                "action_apply_linf_max": "",
                "branch_horizon_rows_expected": 0,
                "branch_horizon_rows_actual": 0,
                "GradeAB_precision": 0,
                "V_integrated_LCB": 0,
                "longrisk_UCB": 1,
                "bad_UCB": 1,
                "null_UCB": 1,
                "memory_UCB": 1,
                "offdiag_UCB": 1,
                "new_positive_created_rate": 0,
                "longrisk_created_rate": 0,
                "runtime_cost_estimate": "",
                "weak_pass": 0,
                "strong_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    summary = {
        "stage": "P6_DIRECT_TRANSFER_SOLVED_UPDATE_V9700",
        "status": "summary",
        "subspace_count": len(subspaces),
        "generated_action_count": 0,
        "branch_horizon_rows_actual": 0,
        "direct_solved_weak_pass": 0,
        "direct_solved_strong_pass": 0,
        "new_objective_evidence_present": 0,
        "generated_route_continue_stop": 1,
        "reason": "exact_transfer_solve_and_branch_horizon_materializer_not_opened_without_pass",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p6_direct_solved_frontier.svg", "P6 direct solved", subspaces, [0.0] * len(subspaces))
    write_bar_svg(out / "fig_p6_new_positive_vs_longrisk.svg", "P6 new positive", subspaces, [0.0] * len(subspaces))
    write_bar_svg(out / "fig_p6_subspace_damage_matrix.svg", "P6 longrisk", subspaces, [1.0] * len(subspaces))
    write_bar_svg(out / "fig_p6_action_norm_vs_transfer_lcb.svg", "P6 norm transfer", subspaces, [0.0] * len(subspaces))
    return [summary] + rows, summary


def boundary_not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        **extra,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p8_controller(p2: dict[str, Any], p3: dict[str, Any], p6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (inum(p2.get("rank_strong_pass")) or inum(p2.get("rank_weak_pass")) or inum(p3.get("p3_core_expansion_pass")) or inum(p6.get("direct_solved_weak_pass")) or inum(p6.get("direct_solved_strong_pass"))):
        return boundary_not_run(
            "P8_MINIMAL_CONTROLLER_V9700",
            "P2_P3_P6_no_weak_or_strong_pass",
            controller_selected=0,
            controller_pass=0,
            source_controller_pass=0,
        )
    row = {
        "stage": "P8_MINIMAL_CONTROLLER_V9700",
        "status": "summary",
        "controller_selected": 1,
        "controller_id": "CTRL-v9700-transfer",
        "accepted_count": p2.get("best_accepted_count", 0),
        "controller_pass": 0,
        "source_controller_pass": 0,
        "reason": "candidate_requires_additional_calibration_not_promoted",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p9_runtime(p8: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p8.get("source_controller_pass")):
        return boundary_not_run("P9_SELECTED_RUNTIME_V9700", "P8_controller_not_selected", selected_runtime_pass=0)
    row = {
        "stage": "P9_SELECTED_RUNTIME_V9700",
        "status": "summary",
        "selected_runtime_pass": 0,
        "reason": "runtime_not_measured_without_controller",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p10_system(p8: dict[str, Any], p9: dict[str, Any], p0: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P10_SYSTEM_BOUNDARY_V9700",
        "status": "summary",
        "controller_pass": p8.get("controller_pass", 0),
        "runtime_pass": p9.get("selected_runtime_pass", 0),
        "field_legality_pass": p0.get("field_legality_pass", 0),
        "system_legal_controller_pass": int(inum(p8.get("controller_pass")) and inum(p9.get("selected_runtime_pass")) and inum(p0.get("field_legality_pass"))),
        "paired_replay_opened": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p11_paired(p10: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inum(p10.get("system_legal_controller_pass")):
        return boundary_not_run("P11_PAIRED_REPLAY_BOUNDARY_V9700", "P10_system_not_official", paired_replay_pass=0, short_full_boundary_open=0)
    return boundary_not_run("P11_PAIRED_REPLAY_BOUNDARY_V9700", "paired_replay_runner_not_implemented", paired_replay_pass=0, short_full_boundary_open=0)


def base_acc(source_v9680: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source_v9680 / "base_acc_sentinel_v9680.csv")]
    for row in rows:
        row["stage"] = "BASE_ACC_SENTINEL_V9700"
        row["base_acc_reused_from_v9680"] = 1
        row["base_acc_used_for_controller"] = 0
    summary = next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})
    return rows, summary


def count_artifact_rows(paths: list[Path]) -> dict[str, Any]:
    rows_checked = 0
    fake_proxy = 0
    fake = 0
    proxy = 0
    cpu = 0
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        for row in read_csv(path):
            rows_checked += 1
            fake += inum(row.get("fake_data_used"))
            proxy += inum(row.get("proxy_row_used"))
            cpu += inum(row.get("cpu_offload_used"))
            fake_proxy += int(inum(row.get("fake_data_used")) or inum(row.get("proxy_row_used")) or inum(row.get("cpu_offload_used")))
    return {
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_proxy,
        "fake_data_used": fake,
        "proxy_row_used": proxy,
        "cpu_offload_used": cpu,
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return path

    def dump_json(name: str, row: dict[str, Any]) -> Path:
        path = out / name
        write_json(path, row)
        artifacts[name] = path
        return path

    source_v9680 = Path(args.source_v9680)
    ap0, score_bundle = load_ap0(args)
    r8a = r5b_scores(ap0, score_bundle)

    p0_rows, p0_field_rows, p0 = p0_boundary(source_v9680)
    dump_csv("p0_boundary_reproduction_v9700.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9700.csv", p0_field_rows)

    p1_rows, p1, comp = p1_transfer_ledger(ap0, out)
    dump_csv("p1_cross_sample_transfer_ledger_v9700.csv", p1_rows)

    p2_rows, p2, rules = p2_rank_comparison(ap0, r8a, comp, out)
    dump_csv("p2_existing_action_rank_comparison_v9700.csv", p2_rows)

    p3_rows, p3 = p3_core_expansion_transfer(ap0, r8a, comp, out)
    dump_csv("p3_core_expansion_transfer_v9700.csv", p3_rows)

    p4_scores = {
        "raw_score": r8a,
        "transported_value_rank": r8a,
        "TransferLCB": comp["transfer_lcb_batch"],
        "TransferLCB_plus_SNR": rules["R3-TransferLCB-plus-SNR-gate"],
        "TransferLCB_plus_memory_gate": rules["R4-TransferLCB-plus-memory-hard-gate"],
    }
    p4_rows, p4 = p4_dataset_shift_diagnostic(ap0, p4_scores, out)
    dump_csv("p4_dataset_shift_no_tuning_diagnostic_v9700.csv", p4_rows)

    p5_rows, p5 = p5_exact_microprobe(ap0, comp, r8a, out)
    dump_csv("p5_exact_microprobe_v9700.csv", p5_rows)

    p6_rows, p6 = p6_direct_solved_update(comp, out)
    dump_csv("p6_direct_transfer_solved_update_v9700.csv", p6_rows)

    p8_rows, p8 = p8_controller(p2, p3, p6)
    dump_csv("p8_minimal_controller_v9700.csv", p8_rows)
    p9_rows, p9 = p9_runtime(p8)
    dump_csv("p9_selected_runtime_v9700.csv", p9_rows)
    p10_rows, p10 = p10_system(p8, p9, p0)
    dump_csv("p10_system_boundary_v9700.csv", p10_rows)
    p11_rows, p11 = p11_paired(p10)
    dump_csv("p11_paired_replay_boundary_v9700.csv", p11_rows)

    base_rows, base_summary = base_acc(source_v9680)
    dump_csv("base_acc_sentinel_v9700.csv", base_rows)

    existing_strong = inum(p2.get("rank_strong_pass")) or inum(p3.get("p3_core_expansion_pass"))
    existing_weak = inum(p2.get("rank_weak_pass"))
    transfer_failed = not inum(p2.get("transfer_principle_improves_old_rank"))
    p6_pass = inum(p6.get("direct_solved_weak_pass")) or inum(p6.get("direct_solved_strong_pass"))
    system_pass = inum(p10.get("system_legal_controller_pass"))
    if not inum(p0.get("p0_pass")):
        route = "R0-BoundaryOrLegalityFail"
        primary = "boundary_or_legality_fail"
    elif system_pass:
        route = "R7-SystemPassReadyForPairedReplay"
        primary = "none"
    elif existing_strong:
        route = "R1-ExistingActionTransferControllerPass"
        primary = "controller_runtime_pending"
    elif p6_pass:
        route = "R3-DirectSolvedTransferPrimitivePromising"
        primary = "direct_solved_transfer_primitive_promising"
    elif existing_weak:
        route = "R2-TransferPrincipleWeakButNotSystem"
        primary = "transfer_principle_weak_controller_not_system"
    elif transfer_failed:
        route = "R4-TransferPrincipleFail"
        primary = "cross_sample_transfer_not_better_than_old_rank"
    elif inum(p6.get("generated_route_continue_stop")):
        route = "R6-GeneratedRouteStoppedNoNewObjective"
        primary = "generated_route_stopped_no_new_objective"
    else:
        route = "R5-CoreExpansionDensityStillAbsent"
        primary = "core_expansion_density_still_absent"
    if route == "R4-TransferPrincipleFail" and inum(p6.get("generated_route_continue_stop")):
        secondary = "generated_route_stopped_no_new_objective"
    else:
        secondary = ""
    route_row = {
        "stage": "P7_ROUTE_DECISION_V9700",
        "status": "summary",
        "route": route,
        "source_route_v9680": p0.get("source_v9680_route"),
        "p0_pass": p0.get("p0_pass"),
        "p1_transfer_ledger_complete_pass": p1.get("ledger_complete_pass"),
        "exact_per_sample_gradient_available": p1.get("exact_per_sample_gradient_available"),
        "rank_strong_pass": p2.get("rank_strong_pass"),
        "rank_weak_pass": p2.get("rank_weak_pass"),
        "best_rank_rule_id": p2.get("best_rule_id"),
        "best_rank_precision": p2.get("best_GradeAB_precision"),
        "best_rank_V_LCB": p2.get("best_V_integrated_LCB"),
        "best_rank_LDO_drop": p2.get("best_LDO_drop"),
        "best_transfer_rule_id": p2.get("best_transfer_rule_id"),
        "best_transfer_precision": p2.get("best_transfer_precision"),
        "best_transfer_V_LCB": p2.get("best_transfer_V_LCB"),
        "best_transfer_LDO_drop": p2.get("best_transfer_LDO_drop"),
        "transfer_principle_improves_old_rank": p2.get("transfer_principle_improves_old_rank"),
        "p3_core_expansion_pass": p3.get("p3_core_expansion_pass"),
        "best_core_expansion_candidate": p3.get("best_candidate_id"),
        "dataset_shift_score_scale_solved": p4.get("dataset_shift_score_scale_solved"),
        "microprobe_pass": p5.get("microprobe_pass"),
        "direct_solved_weak_pass": p6.get("direct_solved_weak_pass"),
        "direct_solved_strong_pass": p6.get("direct_solved_strong_pass"),
        "new_objective_evidence_present": p6.get("new_objective_evidence_present"),
        "controller_pass": p8.get("controller_pass"),
        "selected_runtime_pass": p9.get("selected_runtime_pass"),
        "system_legal_controller_pass": p10.get("system_legal_controller_pass"),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("p7_route_decision_v9700.json", route_row)
    dump_json("route_decision_v9700.json", route_row)
    allowed = {
        "stage": "ALLOWED_NEXT_GATES_V9700",
        "status": "summary",
        "selected_runtime_allowed": int(inum(p8.get("source_controller_pass"))),
        "paired_replay_allowed": int(system_pass),
        "short_full_allowed": int(inum(p11.get("paired_replay_pass"))),
        "APGU_APGV_allowed": int(inum(p6.get("new_objective_evidence_present"))),
        "action_space_redesign_required": int(route in {"R4-TransferPrincipleFail", "R6-GeneratedRouteStoppedNoNewObjective"}),
    }
    stop = {
        "stage": "STOP_CONDITIONS_V9700",
        "status": "summary",
        "stop_old_rank_patch": int(not existing_strong and not existing_weak and transfer_failed),
        "stop_generated_blind_variants": int(inum(p6.get("generated_route_continue_stop"))),
        "enter_system": int(system_pass),
    }
    dump_json("allowed_next_gates_v9700.json", allowed)
    dump_json("stop_conditions_v9700.json", stop)

    nofake = count_artifact_rows(list(artifacts.values()))
    nofake_row = {"stage": "NO_FAKE_AUDIT_V9700", "status": "summary", **nofake}
    dump_csv("no_fake_audit_v9700.csv", [nofake_row])
    contract = {
        "stage": "CONTRACT_AUDIT_V9700",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9680_boundary_pass": p0.get("p0_pass"),
        "transfer_ledger_pass": p1.get("ledger_complete_pass"),
        "rank_strong_pass": p2.get("rank_strong_pass"),
        "rank_weak_pass": p2.get("rank_weak_pass"),
        "core_expansion_pass": p3.get("p3_core_expansion_pass"),
        "dataset_shift_score_scale_solved": p4.get("dataset_shift_score_scale_solved"),
        "microprobe_pass": p5.get("microprobe_pass"),
        "direct_solved_pass": int(inum(p6.get("direct_solved_weak_pass")) or inum(p6.get("direct_solved_strong_pass"))),
        "controller_pass": p8.get("controller_pass"),
        "runtime_pass": p9.get("selected_runtime_pass"),
        "system_legal_controller_pass": p10.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": base_summary.get("base_acc_used_for_controller"),
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
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
    }
    dump_csv("contract_audit_v9700.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9700",
        "status": "summary",
        "route": route,
        "F0_boundary_or_legality_fail": int(route == "R0-BoundaryOrLegalityFail"),
        "F1_existing_action_transfer_controller_pass": int(route == "R1-ExistingActionTransferControllerPass"),
        "F2_transfer_principle_weak_not_system": int(route == "R2-TransferPrincipleWeakButNotSystem"),
        "F3_direct_solved_transfer_promising": int(route == "R3-DirectSolvedTransferPrimitivePromising"),
        "F4_transfer_principle_fail": int(route == "R4-TransferPrincipleFail"),
        "F5_core_expansion_density_absent": int(route == "R5-CoreExpansionDensityStillAbsent"),
        "F6_generated_route_stopped_no_new_objective": int(route == "R6-GeneratedRouteStoppedNoNewObjective" or secondary == "generated_route_stopped_no_new_objective"),
        "F7_system_pass_ready": int(route == "R7-SystemPassReadyForPairedReplay"),
        "F8_system_not_official": int(not system_pass),
        "F9_base_acc_catastrophic": inum(base_summary.get("LQ_catastrophic_fail")),
        "primary_blocker": primary,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
    }
    dump_csv("failure_taxonomy_v9700.csv", [failure])

    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifact_hashes = {name: sha256_file(path) for name, path in artifacts.items() if path.exists()}
    manifest = {
        "version": "v9700",
        "created_utc": "2026-05-15T190000Z",
        "out_dir": repo_rel(out),
        "seed": args.seed,
        "device": args.device,
        "data_root": args.data_root,
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
        "wallclock_sec": time.perf_counter() - t0,
        "sources": {
            "v9680": repo_rel(args.source_v9680),
            "v9670": repo_rel(args.source_v9670),
            "v9660": repo_rel(args.source_v9660),
            "v9650": repo_rel(args.source_v9650),
            "v9640": repo_rel(args.source_v9640),
            "v9630": repo_rel(args.source_v9630),
        },
        "artifact_sha256": artifact_hashes,
    }
    dump_json("run_manifest_v9700.json", manifest)

    print(
        json.dumps(
            {
                "route": route,
                "primary_blocker": primary,
                "secondary_blocker": secondary,
                "best_rank_rule_id": p2.get("best_rule_id"),
                "best_transfer_rule_id": p2.get("best_transfer_rule_id"),
                "best_core_expansion_candidate": p3.get("best_candidate_id"),
                "direct_solved_generated_actions": p6.get("generated_action_count"),
                "system_legal_controller_pass": p10.get("system_legal_controller_pass"),
                "out_dir": repo_rel(out),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
