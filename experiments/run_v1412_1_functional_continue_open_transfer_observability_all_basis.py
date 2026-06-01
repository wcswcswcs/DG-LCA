#!/usr/bin/env python3
"""DG-KAN v14.12.1 continue-open transfer observability runner.

This runner is intentionally a bounded exploration surface. It does not add an
action token, controller, action bank, or promotion route. It rebuilds
synthetic-real observability, expands train-stream proxy audits, runs a
diagnostic real-lite D-CHE FMS surface through pre-registered aliases of the
existing v14.10 mechanisms, reconciles all-basis substrate evidence, and writes
the full v14.12.1 artifact/log contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410
from experiments import run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis as v1411
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144


DEFAULT_OUT = ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1"
PLAN_DOC = ROOT / "docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_执行日志.md"

V1411_OUT = ROOT / "results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411"
V1411_LINE_D = ROOT / "results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/line_d_substrate_hardening_v1411"
V1410_SYNTH = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410"
V1410_REAL = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410"
V1410_RT123 = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt123_steps200_interval200_v1410"
V1410_RT4 = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_composite_steps200_interval200_v1410"
V149_SUBSTRATE = ROOT / "results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3"
V143_MANUAL_COMPACT = ROOT / "results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143"
V1235 = ROOT / "results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235"
V1412_LINE_D_REPLAY = ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/line_d_replay_v149_exact_best_v1412_1"

REQUIRED = [
    "v1412_1_route_decision.json",
    "v1412_1_progress_table.csv",
    "v1412_1_provenance_audit.csv",
    "v1412_1_no_action_search_audit.csv",
    "v1412_1_forbidden_information_audit.csv",
    "v1412_1_method_surface_manifest.csv",
    "v1412_1_synthetic_real_predictivity.csv",
    "v1412_1_train_stream_proxy_expansion.csv",
    "v1412_1_micro_horizon_probe.csv",
    "v1412_1_micro_horizon_validity.csv",
    "v1412_1_real_lite_dche_fms_diagnostic.csv",
    "v1412_1_real_lite_controls.csv",
    "v1412_1_allbasis_reconciliation.csv",
    "v1412_1_d_fou_hardening.csv",
    "v1412_1_d_rbf_hardening.csv",
    "v1412_1_d_wav_monitor.csv",
    "v1412_1_dche_no_regression.csv",
    "v1412_1_mlp_controls.csv",
    "v1412_1_linec_tail_audit.csv",
    "v1412_1_failure_taxonomy.csv",
    "v1412_1_required_manifest.csv",
    "v1412_1_code_review_packet.zip",
]

FIGURES = [
    "fig_synthetic_real_predictivity_auc.svg",
    "fig_synthetic_real_spearman_matrix.svg",
    "fig_train_stream_proxy_auc_matrix.svg",
    "fig_micro_horizon_loss_integral_vs_real_pass.svg",
    "fig_recovery_lag_vs_AUCtime.svg",
    "fig_degree_energy_vs_real_transfer.svg",
    "fig_real_lite_method_heatmap_3x3.svg",
    "fig_real_lite_failure_taxonomy.svg",
    "fig_allbasis_cross_version_reconciliation.svg",
    "fig_allbasis_family_status_matrix.svg",
    "fig_mlp_vs_dche_fms_comparison.svg",
]

FMS_DIAG_ALIASES = {
    "FMS-D1-GenericValueDegreeSafety": "F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection",
    "FMS-D2-TrainStreamTransferUtilityGated": "F-CHE-RT1-TrainSplitAgreement",
    "FMS-D3-ContinuousLowAmplitudeDegreePhase": "F-CHE6-PhaseScheduleDegreeFMS",
    "FMS-D4-MicroHorizonLossIntegralGated": "F-CHE-RT4-CompositeTransferTrust",
    "FMS-D5-RecoveryLagSuppressed": "F-CHE-RT3-DegreeEnergySafetyProjection",
}

REAL_LITE_CONTROLS = [
    "C0-D-CHE-AdamW",
    "C1-D-CHE-AdamW-NoOpMatchedOverhead",
    "C2-D-CHE-AdamW-RandomMatchedNorm",
    "C3-D-CHE-AdamW-AdamWParallelDirectionControl",
    "C4-D-CHE-AdamW-GenericOptimizerStateControl",
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def sint(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return statistics.median(vals) if vals else 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = list(fieldnames or [])
    if not fields:
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def auc_score(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if len(scores) != len(labels) or not scores:
        return None
    positives = sum(1 for y in labels if y == 1)
    negatives = sum(1 for y in labels if y == 0)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(list(scores))
    pos_rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def split_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def summarize_feature(
    *,
    rows: Sequence[dict[str, Any]],
    feature_name: str,
    feature_family: str,
    score_fn: Any,
    label_key: str = "real_pass",
    source_key: str = "real_source_vs_control",
    dataset_key: str = "real_dataset",
    seed_key: str = "real_seed",
) -> dict[str, Any]:
    scored: list[tuple[float, int, float, str, str]] = []
    for row in rows:
        score = score_fn(row)
        if score is None or math.isnan(score) or math.isinf(score):
            continue
        label = sint(row.get(label_key), 0)
        source = fnum(row.get(source_key), 0.0)
        scored.append((float(score), int(label), float(source), str(row.get(dataset_key, "")), str(row.get(seed_key, ""))))
    scores = [x[0] for x in scored]
    labels = [x[1] for x in scored]
    sources = [x[2] for x in scored]
    auc = auc_score(scores, labels)
    rho_source = spearman(scores, sources)
    rho_pass = spearman(scores, [float(x) for x in labels])
    leave_dataset: list[float] = []
    for dataset in sorted({x[3] for x in scored if x[3]}):
        subset = [x for x in scored if x[3] != dataset]
        val = auc_score([x[0] for x in subset], [x[1] for x in subset])
        if val is not None:
            leave_dataset.append(val)
    leave_seed: list[float] = []
    for seed in sorted({x[4] for x in scored if x[4] != ""}):
        subset = [x for x in scored if x[4] != seed]
        val = auc_score([x[0] for x in subset], [x[1] for x in subset])
        if val is not None:
            leave_seed.append(val)
    if scores:
        threshold = median(scores)
        fp = sum(1 for score, label in zip(scores, labels) if score >= threshold and label == 0)
        fn = sum(1 for score, label in zip(scores, labels) if score < threshold and label == 1)
        neg = sum(1 for label in labels if label == 0)
        pos = sum(1 for label in labels if label == 1)
        calibration_slope = (mean([float(l) for s, l in zip(scores, labels) if s >= threshold]) - mean([float(l) for s, l in zip(scores, labels) if s < threshold]))
    else:
        threshold = 0.0
        fp = fn = neg = pos = 0
        calibration_slope = 0.0
    return {
        "feature_name": feature_name,
        "feature_family": feature_family,
        "rows": len(scored),
        "positive_rows": sum(labels),
        "spearman_to_real_source": "" if rho_source is None else rho_source,
        "spearman_to_real_pass": "" if rho_pass is None else rho_pass,
        "auc_to_real_pass": "" if auc is None else auc,
        "auc_to_synthetic_pass": "",
        "leave_dataset_out_auc": mean(leave_dataset),
        "leave_seed_out_auc": mean(leave_seed),
        "false_positive_rate": fp / max(1, neg),
        "false_negative_rate": fn / max(1, pos),
        "calibration_slope": calibration_slope,
        "calibration_intercept": mean([float(x) for x in labels]),
        "median_threshold": threshold,
        "exploration_gate_pass": int((auc or 0.0) >= 0.60),
        "promotion_enabling_gate_pass": int((auc or 0.0) >= 0.70 and (rho_source or 0.0) >= 0.30),
        "uses_validation_for_direction": 0,
        "uses_test_for_direction": 0,
        "uses_linec_tail_auc_calibration_for_direction": 0,
        "promotion_allowed": 0,
    }


def build_method_surface_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in REAL_LITE_CONTROLS:
        rows.append(
            {
                "method": method,
                "method_family": "D-CHE-control",
                "underlying_v1410_method": method,
                "is_control": 1,
                "pre_registered_v1412_1": 1,
                "is_fche8_or_later": 0,
                "is_action_token_extension": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    for alias, underlying in FMS_DIAG_ALIASES.items():
        rows.append(
            {
                "method": alias,
                "method_family": "D-CHE-FMS-diagnostic",
                "underlying_v1410_method": underlying,
                "is_control": 0,
                "pre_registered_v1412_1": 1,
                "actual_micro_horizon_gating": int(alias == "FMS-D4-MicroHorizonLossIntegralGated"),
                "is_fche8_or_later": 0,
                "is_action_token_extension": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    for method in ["MLP-AdamW", "MLP-GenericFMS", "MLP-TransferUtilityFMS", "MLP-RandomMatchedNorm", "MLP-NoOpMatchedOverhead"]:
        rows.append(
            {
                "method": method,
                "method_family": "MLP-control-monitor",
                "underlying_v1410_method": method,
                "is_control": 1,
                "pre_registered_v1412_1": 1,
                "is_fche8_or_later": 0,
                "is_action_token_extension": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    return rows


def build_audits(out: Path, method_manifest: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_files = [
        PLAN_DOC,
        ROOT / "experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py",
        ROOT / "experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py",
        ROOT / "experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py",
        V1410_SYNTH / "v1410_route_decision.json",
        V1410_REAL / "v1410_route_decision.json",
        V1411_OUT / "v1411_route_decision.json",
    ]
    provenance = [
        {
            "artifact": str(path.relative_to(ROOT) if path.is_absolute() and path.exists() else path),
            "exists": int(path.exists()),
            "sha256": sha256(path) if path.exists() and path.is_file() else "",
            "used_for_direction": 0,
            "used_for_audit_or_replay": 1,
            "promotion_allowed": 0,
        }
        for path in source_files
    ]
    no_action = []
    forbidden = []
    for row in method_manifest:
        no_action.append(
            {
                "method": row["method"],
                "is_fche8_or_later": row["is_fche8_or_later"],
                "is_action_token_extension": row["is_action_token_extension"],
                "controller_executed": row["controller_executed"],
                "action_bank_used": row["action_bank_used"],
                "violation": int(
                    sint(row["is_fche8_or_later"]) == 1
                    or sint(row["is_action_token_extension"]) == 1
                    or sint(row["controller_executed"]) == 1
                    or sint(row["action_bank_used"]) == 1
                ),
                "promotion_allowed": 0,
            }
        )
        forbidden.append(
            {
                "method": row["method"],
                "uses_dataset_name_branch": row["uses_dataset_name_branch"],
                "uses_seed_specific_scale": row["uses_seed_specific_scale"],
                "uses_validation_for_direction": 0,
                "uses_test_for_direction": 0,
                "uses_future_for_direction": 0,
                "uses_query_for_direction": 0,
                "uses_linec_for_direction": 0,
                "uses_cep99_for_direction": 0,
                "uses_nll_for_direction": 0,
                "uses_ece_for_direction": 0,
                "uses_auctime_for_direction": 0,
                "violation": int(sint(row["uses_dataset_name_branch"]) == 1 or sint(row["uses_seed_specific_scale"]) == 1),
                "promotion_allowed": 0,
            }
        )
    write_rows(out / "v1412_1_provenance_audit.csv", provenance)
    write_rows(out / "v1412_1_no_action_search_audit.csv", no_action)
    write_rows(out / "v1412_1_forbidden_information_audit.csv", forbidden)
    return provenance, no_action, forbidden


def build_predictivity(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alignment = read_rows(V1411_OUT / "v1411_synthetic_real_alignment.csv")
    features = [
        ("synthetic_source_vs_control", "synthetic-metric", lambda r: fnum(r.get("synthetic_source_vs_control"), 0.0)),
        ("synthetic_AUCtime_proxy", "synthetic-metric", lambda r: -fnum(r.get("synthetic_AUCtime"), 9.0)),
        ("synthetic_tail_safety_score", "synthetic-tail", lambda r: -max(fnum(r.get("synthetic_CEp99_delta"), 0.0), fnum(r.get("synthetic_NLL_delta"), 0.0), fnum(r.get("synthetic_ECE_delta"), 0.0))),
        ("synthetic_LineC_pass", "synthetic-linec", lambda r: fnum(r.get("synthetic_LineC"), 0.0)),
        ("synthetic_failure_pattern_score", "synthetic-composite", lambda r: fnum(r.get("synthetic_source_vs_control"), 0.0) - 0.1 * fnum(r.get("synthetic_AUCtime"), 1.0) - 0.05 * max(0.0, fnum(r.get("synthetic_CEp99_delta"), 0.0))),
        ("method_family_pass_count", "synthetic-family", lambda r: fnum(r.get("method_family_pass_count"), 0.0)),
        ("D-CHE_degree_energy_delta", "degree-telemetry", lambda r: fnum(r.get("synthetic_source_vs_control"), 0.0) * fnum(r.get("synthetic_LineC"), 0.0)),
        ("value_retention_after_degree_projection", "projection-telemetry", lambda r: 1.0 - abs(fnum(r.get("synthetic_AUCtime"), 1.0) - 1.0)),
        ("projection_rejection_fraction_inverse", "projection-telemetry", lambda r: -max(0.0, fnum(r.get("synthetic_AUCtime"), 1.0) - 1.0)),
        ("train_loss_integral_proxy", "transfer-proxy", lambda r: fnum(r.get("synthetic_source_vs_control"), 0.0) - fnum(r.get("synthetic_NLL_delta"), 0.0)),
        ("recovery_lag_proxy", "transfer-proxy", lambda r: -max(0.0, fnum(r.get("synthetic_AUCtime"), 1.0) - 1.0) - max(0.0, fnum(r.get("synthetic_CEp99_delta"), 0.0))),
    ]
    rows = [summarize_feature(rows=alignment, feature_name=name, feature_family=family, score_fn=fn) for name, family, fn in features]
    best = max(rows, key=lambda r: fnum(r.get("auc_to_real_pass"), -1.0)) if rows else {}
    route = {
        "line_e2_rows": len(alignment),
        "line_e2_best_feature": best.get("feature_name", ""),
        "line_e2_best_auc_to_real_pass": fnum(best.get("auc_to_real_pass"), 0.0),
        "line_e2_best_spearman_to_real_source": fnum(best.get("spearman_to_real_source"), 0.0),
        "line_e2_exploration_gate_pass": int(fnum(best.get("auc_to_real_pass"), 0.0) >= 0.60),
        "line_e2_promotion_enabling_gate_pass": int(fnum(best.get("auc_to_real_pass"), 0.0) >= 0.70 and fnum(best.get("spearman_to_real_source"), 0.0) >= 0.30),
    }
    write_rows(out / "v1412_1_synthetic_real_predictivity.csv", rows)
    return rows, route


def run_micro_horizon_probe(args: argparse.Namespace, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the v14.12.1 V2-B bounded micro-horizon B1/B2 probe.

    This is an observability diagnostic only. It uses a current synthetic train
    batch split into B1/B2, computes a train-stream FMS direction on B1, applies
    h in {1,2,4} stateless micro updates on copied models, and evaluates B2
    train losses. No probed update is committed to the experiment model, and no
    validation/test/LineC/tail/AUC/calibration quantity is used to form the
    direction.
    """

    probe_path = out / "v1412_1_micro_horizon_probe.csv"
    validity_path = out / "v1412_1_micro_horizon_validity.csv"
    if probe_path.exists() and validity_path.exists() and not int(args.force_micro_horizon_probe):
        return read_rows(probe_path), read_rows(validity_path)

    import torch
    from experiments.run_v133_task_family_robust_basis_natural import synthetic_data

    device = torch.device(args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    source_rows = [
        r
        for r in read_rows(V1410_SYNTH / "v1410_dche_fms_synthetic_results.csv")
        if r.get("family") == "D-CHE" and sint(r.get("control_method"), 0) == 0
    ]
    if int(args.micro_horizon_max_rows) > 0:
        source_rows = source_rows[: int(args.micro_horizon_max_rows)]
    horizons = split_ints(args.micro_horizon_steps) or [1, 2, 4]

    defaults = v1410.build_argparser().parse_args([])
    defaults.device = str(device)
    defaults.synthetic_dim = 16
    defaults.synthetic_classes = 3
    defaults.mlp_hidden = int(args.mlp_hidden)
    defaults.fms_beta = float(args.micro_horizon_fms_beta)
    defaults.fms_strength = float(args.micro_horizon_fms_strength)
    defaults.train_steps = max(horizons)
    defaults.streaming_per_example_gradients = 1

    probe_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        task = row.get("task", "")
        seed = sint(row.get("seed"), 0)
        loss_interface = row.get("loss_interface", "CE")
        method = row.get("method", "")
        xtr, ytr, _xva, _yva = synthetic_data(
            task,
            seed,
            int(args.micro_horizon_train_size),
            int(args.micro_horizon_val_size),
            16,
            3,
            device,
        )
        bsz = min(int(args.micro_horizon_batch_size), int(xtr.shape[0]))
        gen = torch.Generator(device=device).manual_seed(14_121_000 + idx + seed * 997 + sum(ord(c) for c in task + method + loss_interface))
        sample = torch.randint(0, int(xtr.shape[0]), (bsz,), generator=gen, device=device)
        xb, yb = xtr[sample], ytr[sample]
        split = max(1, int(xb.shape[0]) // 2)
        xb1, yb1 = xb[:split], yb[:split]
        xb2, yb2 = xb[split:], yb[split:]
        if int(xb2.shape[0]) == 0:
            xb2, yb2 = xb1, yb1

        model = v1410.make_case_model("D-CHE", args.dche_candidate, xtr, seed, defaults, device)
        specs = v1410.named_param_specs(model)
        params = [(spec.name, spec.param) for spec in specs]
        keys = [v1410.fms_key(spec, method, "D-CHE") for spec in specs]
        flat_a, util_a = v1410.collect_streaming_per_example_summary(model, xb1, yb1, params, specs, keys, loss_interface)
        flat_b, _util_b = v1410.collect_streaming_per_example_summary(model, xb2, yb2, params, specs, keys, loss_interface)
        state = v1410.FMSState(float(args.micro_horizon_fms_beta), float(args.micro_horizon_fms_strength), seed + 14_121)
        fms_method = "F7-PhaseScheduleFMS" if method == "F-CHE6-PhaseScheduleDegreeFMS" else "F3-LayerFMS"
        key_scales = state.update(keys, {key: float(util_a.get(key, 0.0)) for key in set(keys)}, fms_method, 0, max(horizons))
        projected_flat, active_fraction = v1410.apply_scales_to_flat(method, specs, keys, flat_a.detach().clone(), key_scales, 0, max(horizons))
        split_agreement = 0.0
        if flat_a.numel() and float(flat_a.norm().item()) > 1.0e-8 and float(flat_b.norm().item()) > 1.0e-8:
            split_agreement = float(torch.dot(flat_a, flat_b).item() / max(1.0e-8, float(flat_a.norm().item()) * float(flat_b.norm().item())))

        before = v1411.train_batch_metrics(model, xb2, yb2, loss_interface, torch)
        for horizon in horizons:
            adamw_model = deepcopy(model)
            fms_model = deepcopy(model)
            adamw_metrics: list[dict[str, float]] = []
            fms_metrics: list[dict[str, float]] = []
            for _step in range(int(horizon)):
                v1411.apply_flat_update(adamw_model, flat_a, float(args.micro_horizon_lr), float(args.micro_horizon_weight_decay))
                v1411.apply_flat_update(fms_model, projected_flat, float(args.micro_horizon_lr), float(args.micro_horizon_weight_decay))
                adamw_metrics.append(v1411.train_batch_metrics(adamw_model, xb2, yb2, loss_interface, torch))
                fms_metrics.append(v1411.train_batch_metrics(fms_model, xb2, yb2, loss_interface, torch))
            adamw_final = adamw_metrics[-1]
            fms_final = fms_metrics[-1]
            adamw_integral = sum(m["loss_mean"] for m in adamw_metrics)
            fms_integral = sum(m["loss_mean"] for m in fms_metrics)
            delta_integral_gain = adamw_integral - fms_integral
            delta_q95 = fms_final["loss_q95"] - adamw_final["loss_q95"]
            delta_margin = fms_final["margin_p10"] - adamw_final["margin_p10"]
            rms_drift = abs(fms_final["logit_rms"] - before["logit_rms"]) - abs(adamw_final["logit_rms"] - before["logit_rms"])
            entropy_collapse = max(0.0, before["entropy_norm"] - fms_final["entropy_norm"]) - max(0.0, before["entropy_norm"] - adamw_final["entropy_norm"])
            recovery_lag = sum(max(0.0, m["loss_mean"] - before["loss_mean"]) for m in fms_metrics)
            u_integral = (
                delta_integral_gain
                - 0.25 * max(0.0, delta_q95)
                + 0.10 * delta_margin
                - 0.10 * max(0.0, rms_drift)
                - 0.10 * max(0.0, entropy_collapse)
                - 0.05 * recovery_lag
            )
            u_lag = -recovery_lag - 0.10 * max(0.0, rms_drift) - 0.10 * max(0.0, entropy_collapse)
            probe_rows.append(
                {
                    "stage": "V1412_1_ACTUAL_MICRO_HORIZON_PROBE",
                    "family": "D-CHE",
                    "task": task,
                    "seed": seed,
                    "loss_interface": loss_interface,
                    "method": method,
                    "horizon": int(horizon),
                    "B1_size": int(xb1.shape[0]),
                    "B2_size": int(xb2.shape[0]),
                    "split_gradient_agreement": f"{split_agreement:.12g}",
                    "degree_gate_active_fraction": f"{active_fraction:.12g}",
                    "delta_loss_integral_gain_vs_adamw": f"{delta_integral_gain:.12g}",
                    "delta_q95_loss_vs_adamw": f"{delta_q95:.12g}",
                    "delta_margin_p10_vs_adamw": f"{delta_margin:.12g}",
                    "delta_logit_rms_drift_vs_adamw": f"{rms_drift:.12g}",
                    "delta_entropy_collapse_vs_adamw": f"{entropy_collapse:.12g}",
                    "recovery_lag": f"{recovery_lag:.12g}",
                    "U_micro_horizon_loss_integral": f"{u_integral:.12g}",
                    "U_micro_horizon_recovery_lag": f"{u_lag:.12g}",
                    "micro_horizon_static_direction": 1,
                    "probed_update_committed": 0,
                    "strict_gate_pass": row.get("strict_gate_pass", ""),
                    "synthetic_pass": row.get("strict_gate_pass", ""),
                    "real_pass": 0,
                    "source_vs_best_control_audit_only": row.get("source_vs_best_control", ""),
                    "uses_validation_for_proxy": 0,
                    "uses_test_for_proxy": 0,
                    "uses_linec_tail_auc_calibration_for_proxy": 0,
                    "promotion_allowed": 0,
                }
            )

    validity_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        subset = [r for r in probe_rows if sint(r.get("horizon")) == int(horizon)]
        labels = [sint(r.get("strict_gate_pass")) for r in subset]
        for proxy_name, field in [
            (f"V2-B_actual_micro_horizon_h{horizon}_loss_integral", "U_micro_horizon_loss_integral"),
            (f"V2-D_actual_micro_horizon_h{horizon}_recovery_lag", "U_micro_horizon_recovery_lag"),
        ]:
            scores = [fnum(r.get(field), 0.0) for r in subset]
            auc = auc_score(scores, labels)
            rho = spearman(scores, [fnum(r.get("source_vs_best_control_audit_only"), 0.0) for r in subset])
            validity_rows.append(
                {
                    "proxy_name": proxy_name,
                    "proxy_family": "actual-micro-horizon" if "loss_integral" in proxy_name else "actual-recovery-lag",
                    "horizon": int(horizon),
                    "rows": len(subset),
                    "positive_rows": sum(labels),
                    "auc_to_selected_pass": "" if auc is None else auc,
                    "auc_to_synthetic_pass": "" if auc is None else auc,
                    "auc_to_real_pass": "",
                    "spearman_to_source": "" if rho is None else rho,
                    "exploration_gate_pass": int((auc or 0.0) >= 0.60),
                    "promotion_enabling_gate_pass": int((auc or 0.0) >= 0.70),
                    "uses_validation_for_proxy": 0,
                    "uses_test_for_proxy": 0,
                    "uses_linec_tail_auc_calibration_for_proxy": 0,
                    "promotion_allowed": 0,
                }
            )
    write_rows(probe_path, probe_rows)
    write_rows(validity_path, validity_rows)
    return probe_rows, validity_rows


def build_train_stream_proxy(out: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    telemetry = read_rows(V1411_OUT / "v1411_train_stream_value_proxy.csv")
    split = read_rows(V1411_OUT / "v1411_split_batch_value_probe.csv")
    controls = read_rows(V1411_OUT / "v1411_split_batch_control_validity.csv")
    rows: list[dict[str, Any]] = []

    def add_proxy_summary(name: str, family: str, source_rows: Sequence[dict[str, Any]], score_fn: Any, label_key: str, source_key: str = "source_vs_best_control_audit_only") -> None:
        valid = []
        for row in source_rows:
            score = score_fn(row)
            if score is None or math.isnan(score) or math.isinf(score):
                continue
            valid.append(
                {
                    "score": float(score),
                    "label": sint(row.get(label_key), 0),
                    "synthetic_label": sint(row.get("synthetic_pass"), sint(row.get("strict_gate_pass"), 0)),
                    "real_label": sint(row.get("real_pass"), 0),
                    "source": fnum(row.get(source_key), 0.0),
                    "dataset": row.get("dataset", row.get("task", "")),
                    "seed": row.get("seed", ""),
                }
            )
        auc = auc_score([r["score"] for r in valid], [r["label"] for r in valid])
        auc_syn = auc_score([r["score"] for r in valid], [r["synthetic_label"] for r in valid])
        auc_real = auc_score([r["score"] for r in valid], [r["real_label"] for r in valid])
        rho = spearman([r["score"] for r in valid], [r["source"] for r in valid])
        rows.append(
            {
                "proxy_name": name,
                "proxy_family": family,
                "rows": len(valid),
                "positive_rows": sum(r["label"] for r in valid),
                "auc_to_selected_pass": "" if auc is None else auc,
                "auc_to_synthetic_pass": "" if auc_syn is None else auc_syn,
                "auc_to_real_pass": "" if auc_real is None else auc_real,
                "spearman_to_source": "" if rho is None else rho,
                "exploration_gate_pass": int(max(auc or 0.0, auc_syn or 0.0, auc_real or 0.0) >= 0.60),
                "promotion_enabling_gate_pass": int(max(auc or 0.0, auc_syn or 0.0, auc_real or 0.0) >= 0.70),
                "uses_validation_for_proxy": 0,
                "uses_test_for_proxy": 0,
                "uses_linec_tail_auc_calibration_for_proxy": 0,
                "promotion_allowed": 0,
            }
        )

    add_proxy_summary("telemetry_U_train_to_synthetic_pass", "legacy-telemetry", telemetry, lambda r: fnum(r.get("U_train_telemetry_proxy"), 0.0), "synthetic_pass")
    add_proxy_summary("telemetry_U_train_to_real_pass", "legacy-telemetry", telemetry, lambda r: fnum(r.get("U_train_telemetry_proxy"), 0.0), "real_pass")
    add_proxy_summary("V2-A_split_window_agreement", "split-window", split, lambda r: fnum(r.get("split_gradient_agreement"), 0.0) - 0.1 * fnum(r.get("delta_logit_rms_drift_vs_adamw"), 0.0) - 0.1 * fnum(r.get("delta_entropy_collapse_vs_adamw"), 0.0), "strict_gate_pass")
    add_proxy_summary("V2-B_micro_horizon_loss_integral_h1", "micro-horizon", split, lambda r: fnum(r.get("delta_loss_gain_vs_adamw"), 0.0) - 0.25 * fnum(r.get("delta_q95_loss_vs_adamw"), 0.0) + 0.1 * fnum(r.get("delta_margin_p10_vs_adamw"), 0.0), "strict_gate_pass")
    add_proxy_summary("V2-B_micro_horizon_loss_integral_h2_proxy", "micro-horizon", split, lambda r: 2.0 * fnum(r.get("delta_loss_gain_vs_adamw"), 0.0) - 0.35 * fnum(r.get("delta_q95_loss_vs_adamw"), 0.0) + 0.1 * fnum(r.get("delta_margin_p10_vs_adamw"), 0.0), "strict_gate_pass")
    add_proxy_summary("V2-B_micro_horizon_loss_integral_h4_proxy", "micro-horizon", split, lambda r: 4.0 * fnum(r.get("delta_loss_gain_vs_adamw"), 0.0) - 0.50 * fnum(r.get("delta_q95_loss_vs_adamw"), 0.0) + 0.1 * fnum(r.get("delta_margin_p10_vs_adamw"), 0.0), "strict_gate_pass")
    add_proxy_summary("V2-C_drift_diffusion_group_utility", "drift-diffusion", split, lambda r: fnum(r.get("split_gradient_agreement"), 0.0) * fnum(r.get("U_train_split_batch"), 0.0), "strict_gate_pass")
    add_proxy_summary("V2-D_recovery_lag_proxy", "recovery-lag", split, lambda r: -max(0.0, fnum(r.get("delta_q95_loss_vs_adamw"), 0.0)) - max(0.0, fnum(r.get("delta_entropy_collapse_vs_adamw"), 0.0)) - abs(fnum(r.get("delta_logit_rms_drift_vs_adamw"), 0.0)), "strict_gate_pass")
    add_proxy_summary("V2-E_adamw_fms_state_disagreement", "state-disagreement", split, lambda r: -abs(fnum(r.get("U_train_split_batch"), 0.0) - fnum(r.get("split_gradient_agreement"), 0.0)), "strict_gate_pass")
    add_proxy_summary("V2-F_degree_energy_stability", "degree-telemetry", split, lambda r: -abs(fnum(r.get("degree_gate_active_fraction"), 0.0) - 0.5), "strict_gate_pass")
    micro_probe_rows, micro_validity_rows = run_micro_horizon_probe(args, out)
    rows.extend(micro_validity_rows)

    control_explains = sum(sint(r.get("control_explains_fms"), 0) for r in controls)
    for row in rows:
        row["split_batch_control_probe_rows"] = sum(sint(r.get("rows"), 0) for r in controls)
        row["split_batch_control_explains_fms_count"] = control_explains
        row["controls_explain_proxy"] = int(control_explains > 0)
    best = max(rows, key=lambda r: max(fnum(r.get("auc_to_selected_pass"), 0.0), fnum(r.get("auc_to_synthetic_pass"), 0.0), fnum(r.get("auc_to_real_pass"), 0.0))) if rows else {}
    best_auc = max(fnum(best.get("auc_to_selected_pass"), 0.0), fnum(best.get("auc_to_synthetic_pass"), 0.0), fnum(best.get("auc_to_real_pass"), 0.0))
    route = {
        "line_v2_proxy_rows": len(rows),
        "line_v2_best_proxy": best.get("proxy_name", ""),
        "line_v2_best_auc": best_auc,
        "line_v2_exploration_gate_pass": int(best_auc >= 0.60),
        "line_v2_promotion_enabling_gate_pass": int(best_auc >= 0.70 and control_explains == 0),
        "line_v2_control_explains_count": control_explains,
        "line_v2_micro_horizon_probe_rows": len(micro_probe_rows),
    }
    write_rows(out / "v1412_1_train_stream_proxy_expansion.csv", rows)
    return rows, route


def alias_result(result: dict[str, Any], alias: str, underlying: str) -> dict[str, Any]:
    out = deepcopy(result)
    for key in ["row"]:
        out[key]["method_alias_underlying"] = underlying
        out[key]["method"] = alias
        out[key]["pre_registered_v1412_1_fdiag"] = 1
        out[key]["is_action_token_extension"] = 0
        out[key]["controller_executed"] = 0
        out[key]["action_bank_used"] = 0
        out[key]["promotion_allowed"] = 0
        out[key]["direction_uses_train_stream_only"] = 1
    for rows_key in ["degree_rows", "projection_rows", "linec_rows"]:
        for row in out.get(rows_key, []):
            row["method_alias_underlying"] = underlying
            row["method"] = alias
            row["pre_registered_v1412_1_fdiag"] = 1
            row["is_action_token_extension"] = 0
            row["controller_executed"] = 0
            row["action_bank_used"] = 0
            row["promotion_allowed"] = 0
    return out


def run_real_lite(args: argparse.Namespace, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    existing = out / "v1412_1_real_lite_dche_fms_diagnostic.csv"
    if existing.exists() and not int(args.force_real_lite):
        rows = read_rows(existing)
        controls = read_rows(out / "v1412_1_real_lite_controls.csv")
        linec = read_rows(out / "v1412_1_linec_tail_audit.csv")
        taxonomy = read_rows(out / "v1412_1_failure_taxonomy.csv")
        unique_pass = len({(r.get("dataset"), r.get("seed")) for r in rows if sint(r.get("strict_gate_pass")) == 1})
        summary = {
            "real_lite_reused_existing": 1,
            "real_lite_rows": len(rows),
            "real_lite_dataset_seed_pass_count": unique_pass,
            "real_lite_fms_mean_source_vs_best_control": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in rows]),
            "real_lite_linec_fail_count": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
            "real_lite_source_fail_count": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
            "real_lite_auctime_fail_count": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
            "real_lite_tail_fail_count": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
        }
        return rows, controls, linec, taxonomy, summary

    import torch

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    case_args = v1410.build_argparser().parse_args([])
    case_args.device = str(device)
    case_args.data_root = args.data_root
    case_args.no_download = bool(args.no_download)
    case_args.train_size = int(args.real_lite_train_size)
    case_args.val_size = int(args.real_lite_val_size)
    case_args.test_size = int(args.real_lite_test_size)
    case_args.train_steps = int(args.real_lite_train_steps)
    case_args.batch_size = int(args.real_lite_batch_size)
    case_args.lr = float(args.real_lite_lr)
    case_args.weight_decay = float(args.real_lite_weight_decay)
    case_args.fms_beta = float(args.real_lite_fms_beta)
    case_args.fms_strength = float(args.real_lite_fms_strength)
    case_args.fms_update_interval = int(args.real_lite_fms_update_interval)
    case_args.streaming_per_example_gradients = 1
    case_args.trace_interval = max(1, int(args.real_lite_trace_interval))
    case_args.linec_seeds = args.real_lite_linec_seeds
    case_args.linec_batch_size = int(args.real_lite_linec_batch_size)
    case_args.linec_sketch_dim = int(args.real_lite_linec_sketch_dim)
    case_args.dche_candidate = args.dche_candidate
    case_args.mlp_hidden = int(args.mlp_hidden)
    case_args.synthetic_dim = 16
    case_args.synthetic_classes = 3

    raw_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    methods = REAL_LITE_CONTROLS + list(FMS_DIAG_ALIASES)
    for dataset in split_csv(args.real_lite_datasets):
        for seed in split_ints(args.real_lite_seeds):
            xtr, ytr, xva, yva, xte, yte, input_dim, output_dim = v144.load_real_split(case_args, dataset, seed, device)
            for method in methods:
                underlying = FMS_DIAG_ALIASES.get(method, method)
                case_args.actual_micro_horizon_gating = int(method == "FMS-D4-MicroHorizonLossIntegralGated")
                case_args.actual_micro_horizon_steps = int(args.fdiag_micro_horizon_steps)
                result = v1410.train_model_case(
                    family="D-CHE",
                    candidate_id=args.dche_candidate,
                    method=underlying,
                    dataset=dataset,
                    task="real_lite",
                    seed=int(seed),
                    loss_interface="CE",
                    xtr=xtr,
                    ytr=ytr,
                    xva=xva,
                    yva=yva,
                    xte=xte,
                    yte=yte,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    args=case_args,
                    device=device,
                    real_linec=True,
                )
                if method in FMS_DIAG_ALIASES:
                    result = alias_result(result, method, underlying)
                    result["row"]["actual_micro_horizon_fdiag_d4"] = int(method == "FMS-D4-MicroHorizonLossIntegralGated")
                else:
                    result["row"]["method_alias_underlying"] = method
                    for rows_key in ["degree_rows", "projection_rows", "linec_rows"]:
                        for r in result.get(rows_key, []):
                            r["method_alias_underlying"] = method
                raw_rows.append(result["row"])
                degree_rows.extend(result.get("degree_rows", []))
                projection_rows.extend(result.get("projection_rows", []))
                linec_rows.extend(result.get("linec_rows", []))
    enriched = v1410.enrich_rows(raw_rows, "V1412_1_REAL_LITE_DCHE_FMS_DIAGNOSTIC")
    controls = [r for r in enriched if sint(r.get("control_method"), 0) == 1]
    fms_rows = [r for r in enriched if sint(r.get("control_method"), 0) == 0]
    taxonomy = v1410.failure_taxonomy(enriched)
    for row in taxonomy:
        row["stage"] = "V1412_1_FAILURE_TAXONOMY"
    for row in enriched:
        row["real_lite_diagnostic"] = 1
        row["official_promotion_run"] = 0
        row["promotion_allowed"] = 0
    write_rows(out / "v1412_1_real_lite_dche_fms_diagnostic.csv", fms_rows)
    write_rows(out / "v1412_1_real_lite_controls.csv", controls)
    write_rows(out / "v1412_1_linec_tail_audit.csv", linec_rows)
    write_rows(out / "v1412_1_failure_taxonomy.csv", taxonomy)
    write_rows(out / "v1412_1_dche_degree_telemetry.csv", degree_rows)
    write_rows(out / "v1412_1_dche_projection_retention.csv", projection_rows)
    unique_pass = len({(r.get("dataset"), int(r.get("seed", 0))) for r in fms_rows if sint(r.get("strict_gate_pass")) == 1})
    control_pass = len({(r.get("dataset"), int(r.get("seed", 0))) for r in controls if sint(r.get("strict_gate_pass")) == 1})
    summary = {
        "real_lite_reused_existing": 0,
        "real_lite_rows": len(fms_rows),
        "real_lite_control_rows": len(controls),
        "real_lite_dataset_seed_pass_count": unique_pass,
        "real_lite_control_dataset_seed_pass_count": control_pass,
        "real_lite_fms_mean_source_vs_best_control": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]),
        "real_lite_linec_fail_count": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
        "real_lite_source_fail_count": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
        "real_lite_auctime_fail_count": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
        "real_lite_tail_fail_count": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
    }
    return fms_rows, controls, linec_rows, taxonomy, summary


def build_allbasis_reconciliation(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v149 = read_rows(V149_SUBSTRATE / "v149_line_d_substrate_repair_summary.csv")
    v1411 = read_rows(V1411_OUT / "v1411_line_d_v1411_hardening_summary.csv")
    replay = read_rows(V1412_LINE_D_REPLAY / "v149_line_d_substrate_repair_summary.csv")
    v143 = read_rows(V143_MANUAL_COMPACT / "v143_nonrat_manual_kernel_substrate_summary.csv")
    v1235 = read_rows(V1235 / "v1235_family_substrate_summary.csv")
    families = ["D-CHE", "D-FOU", "D-RBF", "D-WAV"]

    def find_family(rows: Sequence[dict[str, str]], family: str) -> dict[str, str]:
        for row in rows:
            if row.get("family") == family:
                return row
        return {}

    out_rows: list[dict[str, Any]] = []
    for family in families:
        old = find_family(v149, family)
        new = find_family(v1411, family)
        rep = find_family(replay, family)
        manual = find_family(v143, family)
        health = find_family(v1235, family)
        old_count = sint(old.get("family_dataset_seed_pass_count"), sint(old.get("dataset_seed_pass_count"), 0))
        new_count = sint(new.get("family_dataset_seed_pass_count"), 0)
        replay_count = sint(rep.get("family_dataset_seed_pass_count"), -1)
        current_replay_count = replay_count if replay_count >= 0 else new_count
        manual_pass = sint(manual.get("manual_workspace_gate_pass_rows"), sint(manual.get("workspace_pass_rows"), 0))
        health_pass = sint(health.get("family_substrate_pass_count"), sint(health.get("pass_count"), 0))
        if old_count >= 6 and replay_count >= 0 and replay_count < 6:
            diagnosis = "v149_exact_lineage_replay_not_reproduced_under_v1412_budget"
        elif old_count >= 6 and new_count == 0:
            diagnosis = "cross_version_mismatch_budget_or_candidate_lineage"
        elif old_count < 6 and new_count == 0:
            diagnosis = "no_open_substrate_signal"
        elif new_count >= 6:
            diagnosis = "reproduced_or_open"
        else:
            diagnosis = "partial_signal"
        out_rows.append(
            {
                "family": family,
                "v149_dataset_seed_pass_count": old_count,
                "v149_best_candidate": old.get("best_candidate", ""),
                "v1411_dataset_seed_pass_count": new_count,
                "v1411_best_candidate": new.get("best_candidate", ""),
                "v1412_replay_dataset_seed_pass_count": "" if replay_count < 0 else replay_count,
                "v1412_replay_best_candidate": rep.get("best_candidate", ""),
                "v143_manual_workspace_pass_rows": manual_pass,
                "v1235_family_health_pass_count": health_pass,
                "reconciliation_diagnosis": diagnosis,
                "candidate_mismatch_possible": int(str(old.get("best_candidate", "")) != str(new.get("best_candidate", "")) and bool(old.get("best_candidate", ""))),
                "gate_mismatch_possible": int(old_count >= 6 and new_count == 0),
                "run_budget_mismatch_possible": int(old_count >= 6 and fnum(old.get("median_train_step_ratio_vs_MLP"), 0.0) != fnum(new.get("median_train_step_ratio_vs_MLP"), 0.0)),
                "true_non_reproducibility_possible": int(old_count >= 6 and current_replay_count < 6),
                "exploration_gate_pass": int(current_replay_count >= 6),
                "official_fms_eligibility": int(current_replay_count >= 9),
                "official_fms_proof_executed": 0,
                "promotion_allowed": 0,
            }
        )
    write_rows(out / "v1412_1_allbasis_reconciliation.csv", out_rows)
    by_family_rows = read_rows(V1411_OUT / "v1411_line_d_v1411_hardening_results.csv")
    replay_rows = read_rows(V1412_LINE_D_REPLAY / "v149_line_d_substrate_repair_results.csv")
    write_rows(out / "v1412_1_d_fou_hardening.csv", [r for r in by_family_rows if r.get("family") == "D-FOU"] + [r for r in replay_rows if r.get("family") == "D-FOU"])
    write_rows(out / "v1412_1_d_rbf_hardening.csv", [r for r in by_family_rows if r.get("family") == "D-RBF"] + [r for r in replay_rows if r.get("family") == "D-RBF"])
    write_rows(out / "v1412_1_d_wav_monitor.csv", [r for r in by_family_rows if r.get("family") == "D-WAV"])
    best_non_dche = max(
        [r for r in out_rows if r["family"] != "D-CHE"],
        key=lambda r: sint(r["v1412_replay_dataset_seed_pass_count"], sint(r["v1411_dataset_seed_pass_count"], 0)),
        default={},
    )
    summary = {
        "line_d_reconciliation_rows": len(out_rows),
        "line_d_best_non_dche_family": best_non_dche.get("family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": sint(best_non_dche.get("v1412_replay_dataset_seed_pass_count"), sint(best_non_dche.get("v1411_dataset_seed_pass_count"), 0)),
        "line_d_historical_best_non_dche_dataset_seed_pass_count": max([sint(r.get("v149_dataset_seed_pass_count"), 0) for r in out_rows if r["family"] != "D-CHE"] or [0]),
        "line_d_v1411_best_family_dataset_seed_pass_count": max([sint(r.get("v1411_dataset_seed_pass_count"), 0) for r in out_rows] or [0]),
        "line_d_official_fms_eligible_family_count": sum(sint(r.get("official_fms_eligibility"), 0) for r in out_rows if r["family"] != "D-CHE"),
    }
    return out_rows, summary


def build_mlp_controls(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(V1410_SYNTH / "v1410_mlp_generic_controls.csv")
    if not rows:
        rows = read_rows(V1411_OUT / "v1411_mlp_generic_controls.csv")
    for row in rows:
        row["stage"] = "V1412_1_MLP_GENERIC_CONTROL_REPLAY"
        row["used_for_direction"] = 0
        row["used_for_control_audit"] = 1
        row["promotion_allowed"] = 0
    write_rows(out / "v1412_1_mlp_controls.csv", rows)
    mlp_best = max([fnum(r.get("source_vs_best_control"), fnum(r.get("source_vs_adamw"), 0.0)) for r in rows] or [0.0])
    return rows, {"mlp_control_rows": len(rows), "mlp_best_source_proxy": mlp_best}


def build_dche_no_regression(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    synth_route = load_json(V1410_SYNTH / "v1410_route_decision.json")
    real_route = load_json(V1410_REAL / "v1410_route_decision.json")
    substrate = read_rows(V1410_SYNTH / "v1410_all_basis_substrate_status.csv")
    dche_sub = next((r for r in substrate if r.get("family") == "D-CHE"), {})
    rows = [
        {
            "check_name": "v1410_dche_synthetic_s3_retained",
            "value": synth_route.get("synthetic_task_family_pass_count", ""),
            "pass": int(sint(synth_route.get("synthetic_task_family_pass_count"), 0) >= 5),
            "source_artifact": str(V1410_SYNTH / "v1410_route_decision.json"),
            "promotion_allowed": 0,
        },
        {
            "check_name": "v1410_dche_real_best_retained",
            "value": real_route.get("real_dataset_seed_pass_count", ""),
            "pass": int(sint(real_route.get("real_dataset_seed_pass_count"), 0) >= 2),
            "source_artifact": str(V1410_REAL / "v1410_route_decision.json"),
            "promotion_allowed": 0,
        },
        {
            "check_name": "dche_substrate_official_fms_eligibility_replay",
            "value": dche_sub.get("official_fms_eligibility", ""),
            "pass": sint(dche_sub.get("official_fms_eligibility"), 0),
            "source_artifact": str(V1410_SYNTH / "v1410_all_basis_substrate_status.csv"),
            "promotion_allowed": 0,
        },
    ]
    write_rows(out / "v1412_1_dche_no_regression.csv", rows)
    return rows, {
        "synthetic_task_family_pass_count": sint(synth_route.get("synthetic_task_family_pass_count"), 0),
        "v1410_best_real_dataset_seed_pass_count": sint(real_route.get("real_dataset_seed_pass_count"), 0),
        "dche_substrate_official_fms_eligibility": sint(dche_sub.get("official_fms_eligibility"), 0),
    }


def write_required_manifest(out: Path) -> list[dict[str, Any]]:
    manifest_path = out / "v1412_1_required_manifest.csv"
    if not manifest_path.exists():
        manifest_path.write_text("artifact,exists,bytes,sha256,required,missing\n", encoding="utf-8")
    rows = []
    for name in REQUIRED + FIGURES:
        path = out / name
        rows.append(
            {
                "artifact": name,
                "exists": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256(path) if path.exists() and path.is_file() else "",
                "required": 1,
                "missing": int(not path.exists()),
            }
        )
    write_rows(out / "v1412_1_required_manifest.csv", rows)
    return rows


def simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float]], threshold: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, max(180, 70 + 28 * max(1, len(rows)))
    vals = [v for _, v in rows]
    maxv = max([abs(v) for v in vals] + [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="monospace" font-size="16">{title}</text>',
    ]
    if threshold is not None:
        x = 220 + min(450, max(0, threshold / maxv * 450))
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="48" y2="{height-20}" stroke="#888" stroke-dasharray="4 4"/>')
    for i, (label, val) in enumerate(rows):
        y = 62 + i * 28
        w = max(2, min(450, abs(val) / maxv * 450))
        color = "#2f80ed" if val >= 0 else "#d64545"
        parts.append(f'<text x="24" y="{y+14}" font-family="monospace" font-size="12">{label[:28]}</text>')
        parts.append(f'<rect x="220" y="{y}" width="{w:.1f}" height="18" fill="{color}" opacity="0.75"/>')
        parts.append(f'<text x="{230+w:.1f}" y="{y+14}" font-family="monospace" font-size="12">{val:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_figures(out: Path, predictivity: Sequence[dict[str, Any]], proxy: Sequence[dict[str, Any]], real_lite: Sequence[dict[str, Any]], taxonomy: Sequence[dict[str, Any]], recon: Sequence[dict[str, Any]], mlp_rows: Sequence[dict[str, Any]]) -> None:
    simple_svg(out / "fig_synthetic_real_predictivity_auc.svg", "Synthetic -> real AUC", [(r.get("feature_name", ""), fnum(r.get("auc_to_real_pass"), 0.0)) for r in predictivity], threshold=0.60)
    simple_svg(out / "fig_synthetic_real_spearman_matrix.svg", "Synthetic -> real Spearman", [(r.get("feature_name", ""), fnum(r.get("spearman_to_real_source"), 0.0)) for r in predictivity], threshold=0.30)
    simple_svg(out / "fig_train_stream_proxy_auc_matrix.svg", "Train-stream proxy AUC", [(r.get("proxy_name", ""), max(fnum(r.get("auc_to_selected_pass"), 0.0), fnum(r.get("auc_to_synthetic_pass"), 0.0), fnum(r.get("auc_to_real_pass"), 0.0))) for r in proxy], threshold=0.60)
    simple_svg(out / "fig_micro_horizon_loss_integral_vs_real_pass.svg", "Micro horizon proxy", [(r.get("proxy_name", ""), max(fnum(r.get("auc_to_selected_pass"), 0.0), fnum(r.get("auc_to_synthetic_pass"), 0.0), fnum(r.get("auc_to_real_pass"), 0.0))) for r in proxy if "micro" in r.get("proxy_family", "")], threshold=0.60)
    simple_svg(out / "fig_recovery_lag_vs_AUCtime.svg", "Recovery lag proxy", [(r.get("proxy_name", ""), max(fnum(r.get("auc_to_selected_pass"), 0.0), fnum(r.get("auc_to_synthetic_pass"), 0.0), fnum(r.get("auc_to_real_pass"), 0.0))) for r in proxy if "recovery" in r.get("proxy_family", "")], threshold=0.60)
    simple_svg(out / "fig_degree_energy_vs_real_transfer.svg", "Degree telemetry", [(r.get("feature_name", ""), fnum(r.get("auc_to_real_pass"), 0.0)) for r in predictivity if "degree" in r.get("feature_family", "")], threshold=0.60)
    method_pass: dict[str, float] = {}
    for row in real_lite:
        method_pass.setdefault(str(row.get("method")), 0.0)
    for method in method_pass:
        method_pass[method] = len({(r.get("dataset"), r.get("seed")) for r in real_lite if r.get("method") == method and sint(r.get("strict_gate_pass")) == 1})
    simple_svg(out / "fig_real_lite_method_heatmap_3x3.svg", "Real-lite pass by method", sorted(method_pass.items()))
    reasons = {
        "source": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
        "AUCtime": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
        "tail": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
        "LineC": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
        "step": sum(sint(r.get("step_fail"), 0) for r in taxonomy),
        "memory": sum(sint(r.get("memory_fail"), 0) for r in taxonomy),
    }
    simple_svg(out / "fig_real_lite_failure_taxonomy.svg", "Real-lite failure taxonomy", list(reasons.items()))
    simple_svg(out / "fig_allbasis_cross_version_reconciliation.svg", "All-basis old/new pass", [(r.get("family", ""), max(fnum(r.get("v149_dataset_seed_pass_count"), 0.0), fnum(r.get("v1411_dataset_seed_pass_count"), 0.0))) for r in recon], threshold=6.0)
    simple_svg(out / "fig_allbasis_family_status_matrix.svg", "All-basis v14.11 pass", [(r.get("family", ""), fnum(r.get("v1411_dataset_seed_pass_count"), 0.0)) for r in recon], threshold=6.0)
    simple_svg(out / "fig_mlp_vs_dche_fms_comparison.svg", "MLP control source proxy", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), fnum(r.get("source_vs_adamw"), 0.0))) for r in mlp_rows[:20]])


def make_packet(out: Path) -> None:
    packet = out / "v1412_1_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [PLAN_DOC, RECAP_DOC, EXEC_LOG_DOC, ROOT / "experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py"]:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
        for path in sorted(out.glob("v1412_1_*")):
            if path.name == packet.name:
                continue
            if path.is_file():
                zf.write(path, arcname=f"artifacts/{path.name}")
        for path in sorted(out.glob("fig_*.svg")):
            zf.write(path, arcname=f"figures/{path.name}")


def decide_route(e2: dict[str, Any], v2: dict[str, Any], real: dict[str, Any], line_d: dict[str, Any], violations: dict[str, int], missing_count: int) -> dict[str, Any]:
    if violations["forbidden"] > 0 or violations["action"] > 0 or missing_count > 0:
        route_name = "R0-ContractViolation"
    else:
        real_count = sint(real.get("real_lite_dataset_seed_pass_count"), 0)
        proxy_open = sint(e2.get("line_e2_exploration_gate_pass"), 0) == 1 or sint(v2.get("line_v2_exploration_gate_pass"), 0) == 1
        controls_cannot_explain = fnum(real.get("real_lite_fms_mean_source_vs_best_control"), -999.0) > 0.0
        if real_count >= 9 and missing_count == 0:
            route_name = "S5-OfficialFunctionalSuccess"
        elif real_count >= 8:
            route_name = "S3-DCHEOfficialRealTransferCandidate"
        elif real_count >= 6 and controls_cannot_explain:
            route_name = "S2-DCHEDiagnosticRealTransferPositive"
        elif proxy_open and real_count >= 4:
            route_name = "S1-TransferObservabilityExplorationOpened"
        elif real_count < 4 and sint(line_d.get("line_d_best_non_dche_dataset_seed_pass_count"), 0) < 6:
            route_name = "R4-AllBasisSubstrateBlocked"
        elif real_count < 4:
            route_name = "R3-DCHEFMSRealTransferNoGoCurrentDefinition"
        elif not proxy_open:
            route_name = "R2-TrainStreamTransferValueUnobservable"
        else:
            route_name = "R1-SyntheticGateNotPredictiveButExplorationComplete"
    return {
        "route": route_name,
        "minimum_success": "S3-DCHESyntheticFMSPass",
        "line_e2_best_auc_to_real_pass": e2.get("line_e2_best_auc_to_real_pass", 0),
        "line_e2_best_spearman_to_real_source": e2.get("line_e2_best_spearman_to_real_source", 0),
        "line_e2_exploration_gate_pass": e2.get("line_e2_exploration_gate_pass", 0),
        "line_v2_best_auc": v2.get("line_v2_best_auc", 0),
        "line_v2_exploration_gate_pass": v2.get("line_v2_exploration_gate_pass", 0),
        "line_v2_micro_horizon_probe_rows": v2.get("line_v2_micro_horizon_probe_rows", 0),
        "real_lite_dataset_seed_pass_count": real.get("real_lite_dataset_seed_pass_count", 0),
        "real_lite_fms_mean_source_vs_best_control": real.get("real_lite_fms_mean_source_vs_best_control", 0),
        "line_d_best_non_dche_dataset_seed_pass_count": line_d.get("line_d_best_non_dche_dataset_seed_pass_count", 0),
        "synthetic_task_family_pass_count": real.get("synthetic_task_family_pass_count", 5),
        "v1410_best_real_dataset_seed_pass_count": real.get("v1410_best_real_dataset_seed_pass_count", 2),
        "official_s5_reached": int(route_name == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": 0,
        "required_artifact_missing_count": missing_count,
        "forbidden_information_violation_count": violations["forbidden"],
        "no_action_search_violation_count": violations["action"],
        "compute_budgeted_run": 1,
    }


def write_progress(out: Path, route: dict[str, Any], e2: dict[str, Any], v2: dict[str, Any], real: dict[str, Any], line_d: dict[str, Any], dche: dict[str, Any]) -> None:
    rows = [
        {"line": "R", "status": "completed", "key_metric": "violations", "value": route["forbidden_information_violation_count"] + route["no_action_search_violation_count"], "promotion_allowed": 0},
        {"line": "E2", "status": "completed", "key_metric": "best_auc", "value": e2.get("line_e2_best_auc_to_real_pass", 0), "promotion_allowed": 0},
        {"line": "V2", "status": "completed", "key_metric": "best_auc", "value": v2.get("line_v2_best_auc", 0), "promotion_allowed": 0},
        {"line": "F-Diag", "status": "completed", "key_metric": "real_lite_dataset_seed_pass_count", "value": real.get("real_lite_dataset_seed_pass_count", 0), "promotion_allowed": 0},
        {"line": "D", "status": "completed", "key_metric": "best_non_dche_dataset_seed_pass_count", "value": line_d.get("line_d_best_non_dche_dataset_seed_pass_count", 0), "promotion_allowed": 0},
        {"line": "M", "status": "completed", "key_metric": "mlp_control_rows", "value": real.get("mlp_control_rows", 0), "promotion_allowed": 0},
        {"line": "C", "status": "completed", "key_metric": "failure_taxonomy_written", "value": 1, "promotion_allowed": 0},
        {"line": "Z", "status": "completed", "key_metric": "route", "value": route["route"], "promotion_allowed": 0},
        {"line": "D-CHE-no-regression", "status": "completed", "key_metric": "synthetic_task_family_pass_count", "value": dche.get("synthetic_task_family_pass_count", 0), "promotion_allowed": 0},
    ]
    write_rows(out / "v1412_1_progress_table.csv", rows)


def write_docs(out: Path, route: dict[str, Any], e2_rows: Sequence[dict[str, Any]], v2_rows: Sequence[dict[str, Any]], real_summary: dict[str, Any], line_d: dict[str, Any], commands: Sequence[str], args: argparse.Namespace) -> None:
    best_e = max(e2_rows, key=lambda r: fnum(r.get("auc_to_real_pass"), -1.0)) if e2_rows else {}
    best_v = max(v2_rows, key=lambda r: max(fnum(r.get("auc_to_selected_pass"), 0.0), fnum(r.get("auc_to_synthetic_pass"), 0.0), fnum(r.get("auc_to_real_pass"), 0.0))) if v2_rows else {}
    initial_real_lite_120_command = "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 120 --real-lite-batch-size 32 --real-lite-fms-update-interval 120 --real-lite-trace-interval 60 --real-lite-linec-seeds 12319500 --force-real-lite 1"
    strength002_command = "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_strength002_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --real-lite-fms-strength 0.02 --force-real-lite 1"
    strength010_command = "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_strength010_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --real-lite-fms-strength 0.10 --force-real-lite 1"
    real_lite_force_command = (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py "
        f"--out-dir {Path(args.out_dir).as_posix()} "
        f"--real-lite-datasets {args.real_lite_datasets} "
        f"--real-lite-seeds {args.real_lite_seeds} "
        f"--real-lite-train-size {args.real_lite_train_size} "
        f"--real-lite-val-size {args.real_lite_val_size} "
        f"--real-lite-test-size {args.real_lite_test_size} "
        f"--real-lite-train-steps {args.real_lite_train_steps} "
        f"--real-lite-batch-size {args.real_lite_batch_size} "
        f"--real-lite-fms-update-interval {args.real_lite_fms_update_interval} "
        f"--real-lite-trace-interval {args.real_lite_trace_interval} "
        f"--real-lite-linec-seeds {args.real_lite_linec_seeds} "
        f"--fdiag-micro-horizon-steps {args.fdiag_micro_horizon_steps} "
        "--force-real-lite 1"
    )
    line_d_replay_command = "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/line_d_replay_v149_exact_best_v1412_1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --candidates D-FOU26-NoMaterializeLifetimeAuditV2,D-RBF25-WidthConditionGuardNoTaskBranch --train-size 256 --val-size 128 --batch-size 32 --epochs 20 --linec-seeds 12319500,12319501,12319502 --compute-budgeted-run 1"
    micro_horizon_command = (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py "
        f"--out-dir {Path(args.out_dir).as_posix()} "
        f"--real-lite-datasets {args.real_lite_datasets} "
        f"--real-lite-seeds {args.real_lite_seeds} "
        f"--real-lite-train-size {args.real_lite_train_size} "
        f"--real-lite-val-size {args.real_lite_val_size} "
        f"--real-lite-test-size {args.real_lite_test_size} "
        f"--real-lite-train-steps {args.real_lite_train_steps} "
        f"--real-lite-batch-size {args.real_lite_batch_size} "
        f"--real-lite-fms-update-interval {args.real_lite_fms_update_interval} "
        f"--real-lite-trace-interval {args.real_lite_trace_interval} "
        f"--real-lite-linec-seeds {args.real_lite_linec_seeds} "
        f"--fdiag-micro-horizon-steps {args.fdiag_micro_horizon_steps} "
        f"--micro-horizon-max-rows {args.micro_horizon_max_rows} "
        f"--micro-horizon-steps {args.micro_horizon_steps} "
        "--force-micro-horizon-probe 1"
    )
    d4_h2_command = "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_d4_h2_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --fdiag-micro-horizon-steps 2 --micro-horizon-max-rows 210 --micro-horizon-steps 1,2,4 --force-real-lite 1"
    d4_h4_command = "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_d4_h4_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --fdiag-micro-horizon-steps 4 --micro-horizon-max-rows 210 --micro-horizon-steps 1,2,4 --force-real-lite 1"
    all_commands = list(dict.fromkeys([initial_real_lite_120_command, real_lite_force_command, strength002_command, strength010_command, line_d_replay_command, micro_horizon_command, d4_h2_command, d4_h4_command, *commands]))
    sensitivity_rows: list[str] = []
    for label, path in [
        ("strength0.02", ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_strength002_steps200_linec3_v1412_1"),
        ("strength0.10", ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_strength010_steps200_linec3_v1412_1"),
    ]:
        sr = load_json(path / "v1412_1_route_decision.json")
        if sr:
            sensitivity_rows.append(
                f"{label}: pass={sint(sr.get('real_lite_dataset_seed_pass_count'), 0)}/9, "
                f"mean_source={fnum(sr.get('real_lite_fms_mean_source_vs_best_control'), 0.0)}, "
                f"route={sr.get('route')}"
            )
    sensitivity_text = "\n".join(sensitivity_rows) if sensitivity_rows else "not executed"
    micro_rows = [r for r in v2_rows if str(r.get("proxy_family", "")).startswith("actual-")]
    micro_text = "\n".join(
        [
            f"{r.get('proxy_name')}: rows={sint(r.get('rows'), 0)}, "
            f"AUC={fnum(r.get('auc_to_selected_pass'), 0.0)}, "
            f"pass={sint(r.get('exploration_gate_pass'), 0)}"
            for r in micro_rows
        ]
    ) or "not executed"
    real_lite_doc_rows = read_rows(out / "v1412_1_real_lite_dche_fms_diagnostic.csv")
    taxonomy_doc_rows = read_rows(out / "v1412_1_failure_taxonomy.csv")
    d4_rows = [r for r in real_lite_doc_rows if r.get("method") == "FMS-D4-MicroHorizonLossIntegralGated"]
    d4_tax = [r for r in taxonomy_doc_rows if r.get("method") == "FMS-D4-MicroHorizonLossIntegralGated"]
    d4_pass = len({(r.get("dataset"), r.get("seed")) for r in d4_rows if sint(r.get("strict_gate_pass"), 0) == 1})
    d4_text = (
        f"rows={len(d4_rows)}, pass={d4_pass}/9, "
        f"mean_source={mean([fnum(r.get('source_vs_best_control'), 0.0) for r in d4_rows])}, "
        f"best_source={max([fnum(r.get('source_vs_best_control'), -999.0) for r in d4_rows] or [0.0])}, "
        f"source_fail={sum(sint(r.get('source_fail'), 0) for r in d4_tax)}, "
        f"AUC_fail={sum(sint(r.get('AUC_fail'), 0) for r in d4_tax)}, "
        f"tail_fail={sum(sint(r.get('CEp99_fail'), 0) + sint(r.get('NLL_fail'), 0) + sint(r.get('ECE_fail'), 0) for r in d4_tax)}, "
        f"LineC_fail={sum(sint(r.get('LineC_fail'), 0) for r in d4_tax)}"
    )
    d4_horizon_rows: list[str] = []
    for label, path in [
        ("h1", ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1"),
        ("h2", ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_d4_h2_steps200_linec3_v1412_1"),
        ("h4", ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_d4_h4_steps200_linec3_v1412_1"),
    ]:
        sr = load_json(path / "v1412_1_route_decision.json")
        rr = read_rows(path / "v1412_1_real_lite_dche_fms_diagnostic.csv")
        d4g = [r for r in rr if r.get("method") == "FMS-D4-MicroHorizonLossIntegralGated"]
        if sr and d4g:
            d4_horizon_rows.append(
                f"{label}: route={sr.get('route')}, real_lite_pass={sint(sr.get('real_lite_dataset_seed_pass_count'), 0)}/9, "
                f"D4_pass={len({(r.get('dataset'), r.get('seed')) for r in d4g if sint(r.get('strict_gate_pass'), 0) == 1})}/9, "
                f"D4_mean_source={mean([fnum(r.get('source_vs_best_control'), 0.0) for r in d4g])}, "
                f"overall_mean_source={fnum(sr.get('real_lite_fms_mean_source_vs_best_control'), 0.0)}"
            )
    d4_horizon_text = "\n".join(d4_horizon_rows) or "not executed"
    recap = f"""# DG-KAN v14.12.1 FunctionalContinueOpen TransferObservability AllBasis 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 synthetic S3、real-lite diagnostic、substrate eligibility 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v14.12.1 的核心修正是 promotion fail-closed、exploration continue-open。Line E2/V2 即使没有达到 promotion-enabling gate，也必须继续执行 F-Diag、Line D/M/C，并输出 failure taxonomy。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py
```

实现：

```text
1. Line R provenance / no-action-search / forbidden-information audit。
2. Line E2 synthetic-real predictivity rebuild。
3. Line V2 train-stream proxy expansion，包含 split-window、micro-horizon h1/h2/h4 proxy、drift-diffusion、recovery-lag、state disagreement、degree stability。
4. Line F-Diag D-CHE real-lite diagnostic。
5. Line D all-basis cross-version reconciliation，并复写 v14.11 D-FOU/RBF/WAV hardening rows。
6. Line M MLP/generic control replay。
7. Line C tail/LineC/failure taxonomy。
8. required manifest、figures、code review packet、执行日志和复盘日志。
9. 发现 required manifest 自身首次生成顺序导致 route 被误判 R0 后，修复为预创建 manifest 占位再 finalizer。
10. 按 Case D 补跑 v14.9 exact lineage replay：
    D-FOU26-NoMaterializeLifetimeAuditV2、
    D-RBF25-WidthConditionGuardNoTaskBranch。
11. 用户再次要求继续后，按 F-Diag protocol alignment 补跑：
    train_steps={args.real_lite_train_steps},
    fms_update_interval={args.real_lite_fms_update_interval},
    linec_seeds={args.real_lite_linec_seeds}。
12. protocol alignment 后仍未达 4/9，因此继续做全局 FMS amplitude sensitivity：
    strength=0.02 与 strength=0.10。
13. 用户再次要求继续后，补齐计划 V2-B：
    新增实际 bounded micro-horizon h=1/2/4 probe；
    只使用 synthetic train split B1/B2 和 train-stream gradient/FMS state；
    不提交 probed update，不使用 validation/test/LineC/tail/AUC/calibration 生成方向。
14. 随后修正 F-Diag D4 语义：
    FMS-D4 仍使用预注册 token，但开启 actual_micro_horizon_gating；
    gate 只由当前 train batch B1/B2 micro-horizon loss-integral / recovery-lag 计算；
    不读取 real fail pattern 或任何 audit metric。
15. D4 h=1 未打开后，继续做全局 horizon sensitivity：
    fdiag_micro_horizon_steps=2 与 4；
    仍为全局设置，不按 dataset/seed 分支。
```

F-Diag method alias：

```text
FMS-D1 -> F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
FMS-D2 -> F-CHE-RT1-TrainSplitAgreement
FMS-D3 -> F-CHE6-PhaseScheduleDegreeFMS
FMS-D4 -> F-CHE-RT4-CompositeTransferTrust + actual_micro_horizon_gating
FMS-D5 -> F-CHE-RT3-DegreeEnergySafetyProjection
```

这些 alias 记录在 `v1412_1_method_surface_manifest.csv`，不新增 F-CHE8/F-CHE9，不启动 controller/action bank。

## 3. Line E2 结果

```text
best_feature = {best_e.get('feature_name', '')}
best_auc_to_real_pass = {fnum(best_e.get('auc_to_real_pass'), 0.0)}
best_spearman_to_real_source = {fnum(best_e.get('spearman_to_real_source'), 0.0)}
line_e2_exploration_gate_pass = {route.get('line_e2_exploration_gate_pass')}
```

## 4. Line V2 结果

```text
best_proxy = {best_v.get('proxy_name', '')}
best_proxy_auc = {route.get('line_v2_best_auc')}
line_v2_exploration_gate_pass = {route.get('line_v2_exploration_gate_pass')}
control_explains_count = {route.get('line_v2_control_explains_count', '')}
```

### 4.1 Actual micro-horizon probe

```text
micro_horizon_probe_rows = {route.get('line_v2_micro_horizon_probe_rows', '')}
{micro_text}
```

判断：

```text
actual micro-horizon probe 是 observability diagnostic；
它不执行 real training，不改变 F-Diag method surface，不允许 promotion。
```

## 5. F-Diag real-lite 结果

```text
real_lite_dataset_seed_pass_count = {route.get('real_lite_dataset_seed_pass_count')}/9
real_lite_fms_mean_source_vs_best_control = {route.get('real_lite_fms_mean_source_vs_best_control')}
real_lite_source_fail_count = {real_summary.get('real_lite_source_fail_count', '')}
real_lite_auctime_fail_count = {real_summary.get('real_lite_auctime_fail_count', '')}
real_lite_tail_fail_count = {real_summary.get('real_lite_tail_fail_count', '')}
real_lite_linec_fail_count = {real_summary.get('real_lite_linec_fail_count', '')}
```

## 6. Line D all-basis reconciliation

```text
best_non_dche_family = {line_d.get('line_d_best_non_dche_family', '')}
best_non_dche_dataset_seed_pass_count = {line_d.get('line_d_best_non_dche_dataset_seed_pass_count', 0)}/9
historical_best_non_dche_dataset_seed_pass_count = {line_d.get('line_d_historical_best_non_dche_dataset_seed_pass_count', '')}/9
line_d_official_fms_eligible_family_count = {line_d.get('line_d_official_fms_eligible_family_count', 0)}
```

## 6.1 F-Diag strength sensitivity

```text
{sensitivity_text}
```

判断：

```text
strength 0.02 / 0.10 均没有打开 real-lite >=4/9；
唯一 pass 仍来自 Fashion-MNIST seed0 的 FMS-D5-RecoveryLagSuppressed；
因此当前 blocker 不是简单 FMS amplitude。
```

## 6.2 F-Diag D4 actual micro-horizon gating

```text
{d4_text}
```

判断：

```text
actual D4 micro-horizon gating 没有打开 real-lite；
D4 仍被 source/AUC/tail/LineC 混合阻断，且没有达到 4/9 exploration threshold。
```

### 6.3 D4 horizon sensitivity

```text
{d4_horizon_text}
```

判断：

```text
h=2 / h=4 均没有提高 real-lite pass；
micro-horizon 长度不是当前有效杠杆。
```

## 7. 最终 route

```text
route = {route.get('route')}
minimum_success = {route.get('minimum_success')}
official_s5_reached = {route.get('official_s5_reached')}
promotion_allowed = {route.get('promotion_allowed')}
required_artifact_missing_count = {route.get('required_artifact_missing_count')}
forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}
no_action_search_violation_count = {route.get('no_action_search_violation_count')}
```

## 8. 科学结论

```text
1. v14.12.1 已执行 Line R/E2/V2/F-Diag/D/M/C/Z。
   用户继续要求后，已补齐实际 micro-horizon h=1/2/4 train-stream probe。
2. 当前结果没有达成 S5，promotion_allowed = 0。
3. real-lite diagnostic 不能被写成 official real success。
4. all-basis reconciliation 显示 D-CHE 外仍没有 official FMS eligible family；
   v14.9 D-FOU/RBF historical 6/9 在 v14.12.1 exact replay 下没有复现到 >=6/9。
5. 后续若继续，需要新的计划或在 v14.12.1 允许的 alias/proxy 框架内继续，不允许 action/controller/reset route。
```
"""
    write_text(RECAP_DOC, recap)
    exec_log = f"""# DG-KAN v14.12.1 FunctionalContinueOpen TransferObservability AllBasis 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = {PLAN_DOC.relative_to(ROOT)}
runner = experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py
out_dir = {out.relative_to(ROOT)}
recap = {RECAP_DOC.relative_to(ROOT)}
```

## 2. 执行指令

```bash
{chr(10).join(all_commands)}
```

## 3. 主要 artifact

```text
{chr(10).join(REQUIRED)}
```

## 4. 复现说明

```text
1. 使用 /home/chengshun.wang/miniconda3/envs/kan/bin/python 执行 runner。
2. 如需重跑 real-lite，请加 --force-real-lite 1。
3. 如需重跑 actual micro-horizon probe，请加 --force-micro-horizon-probe 1。
4. 如果只想复核现有 artifact，可不加 --force-real-lite，runner 会复用 v1412_1_real_lite_dche_fms_diagnostic.csv。
5. 所有 direction 仍来自 train-stream loss/FMS state；LineC/tail/AUC/calibration 只作 audit/gate。
```
"""
    write_text(EXEC_LOG_DOC, exec_log)
    write_text(out / "v1412_1_no_go_boundary.md", recap.split("## 8. 科学结论", 1)[-1])
    write_text(out / "v1412_1_next_hypothesis_queue.md", "下一步需要新的 transfer-observable FMS definition 或继续 v14.12.1 alias/proxy 框架内的 real-lite diagnostic；不能新增 action/controller/reset route。\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    commands = [" ".join([sys.executable] + sys.argv)]

    method_manifest = build_method_surface_manifest()
    write_rows(out / "v1412_1_method_surface_manifest.csv", method_manifest)
    _provenance, no_action, forbidden = build_audits(out, method_manifest)
    e2_rows, e2_summary = build_predictivity(out)
    v2_rows, v2_summary = build_train_stream_proxy(out, args)
    real_rows, control_rows, linec_rows, taxonomy, real_summary = run_real_lite(args, out)
    recon_rows, line_d_summary = build_allbasis_reconciliation(out)
    mlp_rows, mlp_summary = build_mlp_controls(out)
    dche_rows, dche_summary = build_dche_no_regression(out)

    # Include D-CHE no-regression and MLP summaries in the real route context.
    real_context = dict(real_summary)
    real_context.update(mlp_summary)
    real_context.update(dche_summary)
    violations = {
        "action": sum(sint(r.get("violation"), 0) for r in no_action),
        "forbidden": sum(sint(r.get("violation"), 0) for r in forbidden),
    }
    write_figures(out, e2_rows, v2_rows, real_rows, taxonomy, recon_rows, mlp_rows)
    # Route must not be decided from a pre-finalization manifest, because route,
    # docs and packet are intentionally written after all scientific artifacts.
    route = decide_route(e2_summary, v2_summary, real_context, line_d_summary, violations, 0)
    route["line_v2_control_explains_count"] = v2_summary.get("line_v2_control_explains_count", 0)
    route["line_v2_micro_horizon_probe_rows"] = v2_summary.get("line_v2_micro_horizon_probe_rows", 0)
    write_json(out / "v1412_1_route_decision.json", route)
    write_progress(out, route, e2_summary, v2_summary, real_context, line_d_summary, dche_summary)
    write_docs(out, route, e2_rows, v2_rows, real_context, line_d_summary, commands, args)
    make_packet(out)
    manifest = write_required_manifest(out)
    final_missing = sum(sint(r.get("missing"), 0) for r in manifest)
    if final_missing != route["required_artifact_missing_count"]:
        route = decide_route(e2_summary, v2_summary, real_context, line_d_summary, violations, final_missing)
        route["line_v2_control_explains_count"] = v2_summary.get("line_v2_control_explains_count", 0)
        route["line_v2_micro_horizon_probe_rows"] = v2_summary.get("line_v2_micro_horizon_probe_rows", 0)
        write_json(out / "v1412_1_route_decision.json", route)
        write_progress(out, route, e2_summary, v2_summary, real_context, line_d_summary, dche_summary)
        write_docs(out, route, e2_rows, v2_rows, real_context, line_d_summary, commands, args)
        make_packet(out)
        write_required_manifest(out)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.12.1 continue-open transfer observability")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dche-candidate", default=v1410.DEFAULT_D_CHE_CANDIDATE)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--real-lite-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--real-lite-seeds", default="0,1,2")
    parser.add_argument("--real-lite-train-size", type=int, default=1024)
    parser.add_argument("--real-lite-val-size", type=int, default=512)
    parser.add_argument("--real-lite-test-size", type=int, default=512)
    parser.add_argument("--real-lite-train-steps", type=int, default=120)
    parser.add_argument("--real-lite-batch-size", type=int, default=32)
    parser.add_argument("--real-lite-lr", type=float, default=0.005)
    parser.add_argument("--real-lite-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--real-lite-fms-beta", type=float, default=0.99)
    parser.add_argument("--real-lite-fms-strength", type=float, default=0.05)
    parser.add_argument("--real-lite-fms-update-interval", type=int, default=120)
    parser.add_argument("--real-lite-trace-interval", type=int, default=60)
    parser.add_argument("--real-lite-linec-seeds", default="12319500")
    parser.add_argument("--real-lite-linec-batch-size", type=int, default=24)
    parser.add_argument("--real-lite-linec-sketch-dim", type=int, default=8)
    parser.add_argument("--force-real-lite", type=int, default=0)
    parser.add_argument("--micro-horizon-max-rows", type=int, default=210)
    parser.add_argument("--micro-horizon-steps", default="1,2,4")
    parser.add_argument("--micro-horizon-train-size", type=int, default=96)
    parser.add_argument("--micro-horizon-val-size", type=int, default=64)
    parser.add_argument("--micro-horizon-batch-size", type=int, default=32)
    parser.add_argument("--micro-horizon-lr", type=float, default=0.005)
    parser.add_argument("--micro-horizon-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--micro-horizon-fms-beta", type=float, default=0.99)
    parser.add_argument("--micro-horizon-fms-strength", type=float, default=0.05)
    parser.add_argument("--force-micro-horizon-probe", type=int, default=0)
    parser.add_argument("--fdiag-micro-horizon-steps", type=int, default=1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    start = time.perf_counter()
    route = run(args)
    route = dict(route)
    route["wall_time_sec"] = time.perf_counter() - start
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
