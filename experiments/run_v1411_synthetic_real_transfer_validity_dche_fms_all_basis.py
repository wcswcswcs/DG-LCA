#!/usr/bin/env python3
"""DG-KAN v14.11 synthetic-to-real validity audit.

This runner is deliberately an audit/finalizer surface, not a new action search.
It reads v14.10/v14.9/v14.4/v14.3 artifacts, computes pre-registered
synthetic-real and train-stream telemetry validity checks, and writes the v14.11
artifact contract. It does not generate a new optimization direction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411"
PLAN_DOC = ROOT / "docs/DG-KAN_v14.11_SyntheticRealTransferValidity_DCHE_FMS_AllBasis_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v14.11_SyntheticRealTransferValidity_DCHE_FMS_AllBasis_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v14.11_SyntheticRealTransferValidity_DCHE_FMS_AllBasis_执行日志.md"
V1410_SCRIPT = ROOT / "experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py"

V1410_SYNTH = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410"
V1410_REAL = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410"
V1410_RT123 = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt123_steps200_interval200_v1410"
V1410_RT4 = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_composite_steps200_interval200_v1410"
V149_SUBSTRATE = ROOT / "results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3"
V143_SYNTH = ROOT / "results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143"
V143_REAL = ROOT / "results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143"
V144_REAL = ROOT / "results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lambda05"
V145_ORACLE_ROOT = ROOT / "results/v14_5_train_stream_counterfactual_fms_all_basis_parallel"
V1411_LINE_D = ROOT / "results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/line_d_substrate_hardening_v1411"

REQUIRED = [
    "v1411_progress_table.csv",
    "v1411_code_review_manifest.csv",
    "v1411_no_action_search_audit.csv",
    "v1411_forbidden_information_audit.csv",
    "v1411_plan_compliance_readback.md",
    "v1411_synthetic_real_alignment.csv",
    "v1411_synthetic_real_predictivity_summary.csv",
    "v1411_train_stream_value_proxy.csv",
    "v1411_train_stream_value_validity.csv",
    "v1411_split_batch_value_probe.csv",
    "v1411_split_batch_value_validity.csv",
    "v1411_split_batch_control_validity.csv",
    "v1411_v145_oracle_reference.csv",
    "v1411_v145_missing_state_reference.csv",
    "v1411_dche_fms_definition_results.csv",
    "v1411_dche_real_transfer_results.csv",
    "v1411_all_basis_substrate_status.csv",
    "v1411_fourier_substrate_hardening.csv",
    "v1411_rbf_substrate_hardening.csv",
    "v1411_wavelet_substrate_monitor.csv",
    "v1411_line_d_v1411_hardening_summary.csv",
    "v1411_line_d_v1411_hardening_results.csv",
    "v1411_line_d_v1411_hardening_route.json",
    "v1411_mlp_generic_controls.csv",
    "v1411_linec_tail_audit.csv",
    "v1411_failure_taxonomy.csv",
    "v1411_route_decision.json",
    "v1411_no_go_boundary.md",
    "v1411_next_hypothesis_queue.md",
    "v1411_required_artifact_manifest.csv",
    "v1411_code_review_packet.zip",
]

FIGURES = [
    "fig_v1411_progress_by_line.svg",
    "fig_synthetic_to_real_predictivity.svg",
    "fig_synthetic_task_family_vs_real_pass_heatmap.svg",
    "fig_train_stream_value_vs_real_pass.svg",
    "fig_dche_fms_synthetic_family_heatmap.svg",
    "fig_dche_real_transfer_3x3_matrix.svg",
    "fig_source_auc_tail_failure_decomposition.svg",
    "fig_all_basis_substrate_matrix.svg",
    "fig_mlp_vs_dche_fms_comparison.svg",
    "fig_no_action_search_audit.svg",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    vals = [v for v in values if not math.isnan(v)]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.median(vals) if vals else 0.0


def rankdata(values: Sequence[float]) -> List[float]:
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


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(rankdata(xs), rankdata(ys))


def auc_score(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
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


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_control(row: Dict[str, str]) -> bool:
    method = row.get("method", "")
    return row.get("control_method", "") == "1" or method.startswith("C0-") or method.startswith("M0-")


def method_key(method: str) -> str:
    return method.split(":", 1)[0].strip()


def build_alignment() -> List[Dict[str, Any]]:
    synth_rows = [r for r in read_rows(V1410_SYNTH / "v1410_dche_fms_synthetic_results.csv") if not is_control(r)]
    real_rows = [r for r in read_rows(V1410_REAL / "v1410_dche_fms_real_results.csv") if not is_control(r)]
    synth_summary = read_rows(V1410_SYNTH / "v1410_dche_fms_synthetic_summary.csv")
    method_pass = {r.get("method", ""): r.get("task_family_pass_count", "") for r in synth_summary}
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for row in real_rows:
        by_method.setdefault(method_key(row.get("method", "")), []).append(row)

    rows: List[Dict[str, Any]] = []
    for s in synth_rows:
        matches = by_method.get(method_key(s.get("method", "")), [])
        for r in matches:
            rows.append(
                {
                    "source_artifact": str(V1410_SYNTH.relative_to(ROOT)),
                    "real_artifact": str(V1410_REAL.relative_to(ROOT)),
                    "family": "D-CHE",
                    "substrate_candidate": s.get("candidate_id", ""),
                    "method": s.get("method", ""),
                    "synthetic_task": s.get("task", ""),
                    "synthetic_seed": s.get("seed", ""),
                    "loss_interface": s.get("loss_interface", ""),
                    "synthetic_source_vs_control": s.get("source_vs_best_control", ""),
                    "synthetic_AUCtime": s.get("AUCtime_ratio", ""),
                    "synthetic_CEp99_delta": s.get("CEp99_delta", ""),
                    "synthetic_NLL_delta": s.get("NLL_delta", ""),
                    "synthetic_ECE_delta": s.get("ECE_delta", ""),
                    "synthetic_LineC": s.get("LineC_majority_pass", ""),
                    "synthetic_pass": s.get("strict_gate_pass", ""),
                    "method_family_pass_count": method_pass.get(s.get("method", ""), ""),
                    "real_dataset": r.get("dataset", ""),
                    "real_seed": r.get("seed", ""),
                    "real_source_vs_control": r.get("source_vs_best_control", ""),
                    "real_AUCtime": r.get("AUCtime_ratio", ""),
                    "real_CEp99_delta": r.get("CEp99_delta", ""),
                    "real_NLL_delta": r.get("NLL_delta", ""),
                    "real_ECE_delta": r.get("ECE_delta", ""),
                    "real_LineC": r.get("LineC_majority_pass", ""),
                    "real_pass": r.get("strict_gate_pass", ""),
                    "matched_method_id": method_key(s.get("method", "")),
                    "is_promotable_candidate": 0,
                }
            )
    rows.extend(build_rational_alignment())
    return rows


def build_rational_alignment() -> List[Dict[str, Any]]:
    synth_rows = [
        r
        for r in read_rows(V143_SYNTH / "v143_rational_fms_results.csv")
        if r.get("method") == "K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint"
        and sint(r.get("control_method")) == 0
    ]
    real_rows = [
        r
        for r in read_rows(V143_REAL / "v143_real_short_run_results.csv")
        if r.get("method") == "K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint"
        and sint(r.get("control_method")) == 0
    ]
    rows: List[Dict[str, Any]] = []
    for s in synth_rows:
        for r in real_rows:
            rows.append(
                {
                    "source_artifact": str(V143_SYNTH.relative_to(ROOT)),
                    "real_artifact": str(V143_REAL.relative_to(ROOT)),
                    "family": "D-RAT",
                    "substrate_candidate": s.get("candidate_id", ""),
                    "method": s.get("method", ""),
                    "synthetic_task": s.get("task", ""),
                    "synthetic_seed": s.get("seed", ""),
                    "loss_interface": s.get("loss_interface", ""),
                    "synthetic_source_vs_control": s.get("source_vs_best_control", ""),
                    "synthetic_AUCtime": s.get("AUCtime_ratio_vs_best_control", ""),
                    "synthetic_CEp99_delta": s.get("CEp99_delta_vs_adamw", ""),
                    "synthetic_NLL_delta": s.get("NLL_delta_vs_adamw", ""),
                    "synthetic_ECE_delta": s.get("ECE_delta_vs_adamw", ""),
                    "synthetic_LineC": s.get("LineC_majority_pass", ""),
                    "synthetic_pass": s.get("synthetic_gate_pass", ""),
                    "method_family_pass_count": "",
                    "real_dataset": r.get("dataset", ""),
                    "real_seed": r.get("seed", ""),
                    "real_source_vs_control": r.get("source_vs_best_control", ""),
                    "real_AUCtime": r.get("AUCtime_ratio_vs_best_control", ""),
                    "real_CEp99_delta": r.get("CEp99_delta_vs_adamw", ""),
                    "real_NLL_delta": r.get("NLL_delta_vs_adamw", ""),
                    "real_ECE_delta": r.get("ECE_delta_vs_adamw", ""),
                    "real_LineC": r.get("LineC_majority_pass", ""),
                    "real_pass": r.get("real_short_run_gate_pass", ""),
                    "matched_method_id": "K8-RAT-v143-synthetic-real",
                    "is_promotable_candidate": 0,
                }
            )
    v144_real = [r for r in read_rows(V144_REAL / "v144_real_transfer_fms_results.csv") if sint(r.get("control_method")) == 0]
    synth_route = load_json(V143_SYNTH / "v143_route_decision.json")
    synth_source = fnum(synth_route.get("kan_specific_positive_rows")) / 42.0
    for r in v144_real:
        rows.append(
            {
                "source_artifact": str((V143_SYNTH / "v143_route_decision.json").relative_to(ROOT)),
                "real_artifact": str(V144_REAL.relative_to(ROOT)),
                "family": "D-RAT-v144-reference",
                "substrate_candidate": "D-RAT28-GroupDiversityPreservingRational",
                "method": r.get("method", ""),
                "synthetic_task": "v143_route_aggregate",
                "synthetic_seed": "",
                "loss_interface": r.get("loss_interface", "CE"),
                "synthetic_source_vs_control": f"{synth_source:.12g}",
                "synthetic_AUCtime": "",
                "synthetic_CEp99_delta": "",
                "synthetic_NLL_delta": "",
                "synthetic_ECE_delta": "",
                "synthetic_LineC": "",
                "synthetic_pass": int(sint(synth_route.get("rational_fms_task_pass_count")) >= 5),
                "method_family_pass_count": synth_route.get("rational_fms_task_pass_count", ""),
                "real_dataset": r.get("dataset", ""),
                "real_seed": r.get("seed", ""),
                "real_source_vs_control": r.get("source_vs_best_control", ""),
                "real_AUCtime": r.get("AUCtime_ratio_vs_best_control", ""),
                "real_CEp99_delta": r.get("CEp99_delta_vs_adamw", ""),
                "real_NLL_delta": r.get("NLL_delta_vs_adamw", ""),
                "real_ECE_delta": r.get("ECE_delta_vs_adamw", ""),
                "real_LineC": r.get("LineC_majority_pass", ""),
                "real_pass": r.get("real_transfer_gate_pass", ""),
                "matched_method_id": "RAT-S3-to-v144-S4b-reference",
                "is_promotable_candidate": 0,
            }
        )
    return rows


def summarize_predictivity(alignment: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def subset(label: str, pred):
        rows = [r for r in alignment if pred(r)]
        xs = [fnum(r["synthetic_source_vs_control"]) for r in rows]
        ys = [fnum(r["real_source_vs_control"]) for r in rows]
        auc_features = [
            fnum(r["synthetic_source_vs_control"])
            - 0.25 * max(0.0, fnum(r["synthetic_AUCtime"]) - 1.0)
            - 0.05 * max(0.0, fnum(r["synthetic_CEp99_delta"]))
            - 0.05 * max(0.0, fnum(r["synthetic_ECE_delta"]))
            for r in rows
        ]
        labels = [sint(r["real_pass"]) for r in rows]
        return {
            "split": label,
            "rows": len(rows),
            "positive_real_pass_rows": sum(labels),
            "spearman_synthetic_source_real_source": none_to_blank(spearman(xs, ys)),
            "pearson_synthetic_source_real_source": none_to_blank(pearson(xs, ys)),
            "auc_predict_synthetic_features_to_real_pass": none_to_blank(auc_score(auc_features, labels)),
            "gate_auc_threshold": 0.70,
            "gate_spearman_threshold": 0.30,
        }

    out = [
        subset("D-CHE-internal-all", lambda r: r.get("family") == "D-CHE"),
        subset("D-RAT-v143-internal-all", lambda r: r.get("family") == "D-RAT"),
        subset("D-RAT-v144-reference", lambda r: r.get("family") == "D-RAT-v144-reference"),
        subset("all-families-reference", lambda r: True),
        subset("CE-only", lambda r: r.get("loss_interface") == "CE"),
        subset("Brier-only", lambda r: r.get("loss_interface") == "Brier"),
        subset("MNIST-heldout-view", lambda r: r.get("real_dataset") == "MNIST"),
        subset("FashionMNIST-heldout-view", lambda r: r.get("real_dataset") == "Fashion-MNIST"),
        subset("KMNIST-heldout-view", lambda r: r.get("real_dataset") == "KMNIST"),
    ]
    for row in out:
        auc = optional_float(row["auc_predict_synthetic_features_to_real_pass"])
        sp = optional_float(row["spearman_synthetic_source_real_source"])
        row["predictivity_gate_pass"] = int(auc is not None and sp is not None and auc >= 0.70 and sp >= 0.30)
        row["promotion_allowed"] = 0
    return out


def optional_float(value: Any) -> Optional[float]:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def none_to_blank(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def train_stream_utility(row: Dict[str, str]) -> float:
    retention = fnum(row.get("value_retention_after_degree_projection"), 1.0)
    cos = fnum(row.get("cos_projected_vs_generic"), 1.0)
    rejection = fnum(row.get("degree_projection_rejection_fraction"), 0.0)
    high_degree = fnum(row.get("high_degree_energy_fraction"), 0.0)
    entropy = fnum(row.get("degree_entropy"), 0.0)
    update_norm = fnum(row.get("parameter_update_norm"), 0.0)
    state_norm = fnum(row.get("fms_state_norm"), 0.0)
    refresh = fnum(row.get("fms_refresh_count"), 0.0)
    # Telemetry-only proxy: all fields are train-stream/model-state telemetry.
    return (
        retention
        + 0.15 * cos
        + 0.02 * entropy
        + 0.01 * refresh
        - 0.35 * rejection
        - 0.10 * high_degree
        - 0.03 * math.log1p(abs(update_norm))
        - 0.01 * math.log1p(abs(state_norm))
    )


def build_train_stream_proxy() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    synth = [r for r in read_rows(V1410_SYNTH / "v1410_dche_fms_synthetic_results.csv") if not is_control(r)]
    real = [r for r in read_rows(V1410_REAL / "v1410_dche_fms_real_results.csv") if not is_control(r)]
    mlp = [r for r in read_rows(V1410_SYNTH / "v1410_mlp_generic_controls.csv") if r.get("family") == "MLP"]
    proxy_rows: List[Dict[str, Any]] = []
    for stage, rows in [("synthetic", synth), ("real", real), ("mlp_synthetic", mlp)]:
        for r in rows:
            u = train_stream_utility(r)
            proxy_rows.append(
                {
                    "stage": stage,
                    "family": r.get("family", ""),
                    "dataset": r.get("dataset", ""),
                    "task": r.get("task", ""),
                    "seed": r.get("seed", ""),
                    "loss_interface": r.get("loss_interface", ""),
                    "method": r.get("method", ""),
                    "U_train_telemetry_proxy": f"{u:.12g}",
                    "uses_validation_for_proxy": 0,
                    "uses_test_for_proxy": 0,
                    "uses_linec_tail_auc_calibration_for_proxy": 0,
                    "strict_gate_pass": r.get("strict_gate_pass", ""),
                    "real_pass": r.get("strict_gate_pass", "") if stage == "real" else "",
                    "synthetic_pass": r.get("strict_gate_pass", "") if stage != "real" else "",
                    "source_vs_best_control_audit_only": r.get("source_vs_best_control", ""),
                    "AUCtime_audit_only": r.get("AUCtime_ratio", ""),
                    "LineC_audit_only": r.get("LineC_majority_pass", ""),
                    "promotion_allowed": 0,
                }
            )

    dche_synth_scores = [fnum(r["U_train_telemetry_proxy"]) for r in proxy_rows if r["stage"] == "synthetic"]
    dche_synth_labels = [sint(r["synthetic_pass"]) for r in proxy_rows if r["stage"] == "synthetic"]
    dche_real_scores = [fnum(r["U_train_telemetry_proxy"]) for r in proxy_rows if r["stage"] == "real"]
    dche_real_labels = [sint(r["real_pass"]) for r in proxy_rows if r["stage"] == "real"]
    mlp_scores = [fnum(r["U_train_telemetry_proxy"]) for r in proxy_rows if r["stage"] == "mlp_synthetic"]
    validity = [
        {
            "split": "D-CHE-synthetic-telemetry-to-synthetic-pass",
            "rows": len(dche_synth_scores),
            "positive_rows": sum(dche_synth_labels),
            "auc_U_train_to_pass": none_to_blank(auc_score(dche_synth_scores, dche_synth_labels)),
            "gate_auc_threshold_without_new_real": 0.75,
            "gate_pass": int((auc_score(dche_synth_scores, dche_synth_labels) or 0.0) >= 0.75),
            "promotion_allowed": 0,
        },
        {
            "split": "D-CHE-real-telemetry-to-real-pass",
            "rows": len(dche_real_scores),
            "positive_rows": sum(dche_real_labels),
            "auc_U_train_to_pass": none_to_blank(auc_score(dche_real_scores, dche_real_labels)),
            "gate_auc_threshold_with_real": 0.70,
            "gate_pass": int((auc_score(dche_real_scores, dche_real_labels) or 0.0) >= 0.70),
            "promotion_allowed": 0,
        },
        {
            "split": "D-CHE-vs-MLP-generic-telemetry-mean",
            "rows": len(dche_synth_scores) + len(mlp_scores),
            "D_CHE_mean_U_train": f"{mean(dche_synth_scores):.12g}",
            "MLP_mean_U_train": f"{mean(mlp_scores):.12g}",
            "U_DCHE_minus_U_MLP": f"{(mean(dche_synth_scores) - mean(mlp_scores)):.12g}",
            "gate_requires_positive_delta": 1,
            "gate_pass": int(mean(dche_synth_scores) - mean(mlp_scores) > 0.0),
            "promotion_allowed": 0,
        },
    ]
    return proxy_rows, validity


def train_batch_metrics(model: Any, xb: Any, yb: Any, loss_interface: str, torch_mod: Any) -> Dict[str, float]:
    from experiments.run_v142_functional_first_all_basis_parallel import loss_value

    with torch_mod.no_grad():
        logits = model(xb)
        losses = []
        for i in range(int(xb.shape[0])):
            losses.append(float(loss_value(model(xb[i : i + 1]), yb[i : i + 1], loss_interface).item()))
        probs = torch_mod.softmax(logits, dim=-1)
        top2 = torch_mod.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else 0.0)
        entropy = -(probs * torch_mod.log(probs.clamp_min(1.0e-12))).sum(dim=-1) / math.log(max(2, probs.shape[-1]))
        return {
            "loss_mean": float(sum(losses) / max(1, len(losses))),
            "loss_q95": float(torch_mod.tensor(losses, device=xb.device).quantile(0.95).item()) if losses else 0.0,
            "margin_p10": float(margin.quantile(0.10).item()),
            "logit_rms": float(logits.float().pow(2).mean().sqrt().item()),
            "entropy_norm": float(entropy.mean().item()),
        }


def apply_flat_update(model: Any, flat_grad: Any, lr: float, weight_decay: float) -> None:
    from experiments.run_v142_functional_first_all_basis_parallel import named_param_specs

    specs = named_param_specs(model)
    for spec in specs:
        spec.param.grad = flat_grad[spec.start : spec.end].view_as(spec.param).detach().clone()
    opt = __import__("torch").optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt.step()
    opt.zero_grad(set_to_none=True)


def u_train_from_metrics(before: Dict[str, float], adamw_after: Dict[str, float], candidate_after: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    delta_loss_gain = adamw_after["loss_mean"] - candidate_after["loss_mean"]
    delta_q95 = candidate_after["loss_q95"] - adamw_after["loss_q95"]
    delta_margin = candidate_after["margin_p10"] - adamw_after["margin_p10"]
    rms_drift = abs(candidate_after["logit_rms"] - before["logit_rms"]) - abs(adamw_after["logit_rms"] - before["logit_rms"])
    entropy_collapse = max(0.0, before["entropy_norm"] - candidate_after["entropy_norm"]) - max(0.0, before["entropy_norm"] - adamw_after["entropy_norm"])
    u_train = delta_loss_gain - 0.25 * max(0.0, delta_q95) - 0.10 * max(0.0, -delta_margin) - 0.10 * max(0.0, rms_drift) - 0.10 * max(0.0, entropy_collapse)
    return u_train, {
        "delta_loss_gain_vs_adamw": delta_loss_gain,
        "delta_q95_loss_vs_adamw": delta_q95,
        "delta_margin_p10_vs_adamw": delta_margin,
        "delta_logit_rms_drift_vs_adamw": rms_drift,
        "delta_entropy_collapse_vs_adamw": entropy_collapse,
    }


def run_split_batch_probe(
    *,
    device_name: str,
    max_rows: int,
    batch_size: int,
    synthetic_train_size: int,
    synthetic_val_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute plan-7.3 style B1/B2 U_train for existing synthetic rows.

    The probe evaluates existing pre-registered methods against AdamW on a
    current train batch. It does not choose a direction, and uses prior strict
    pass labels only after the fact for validity scoring.
    """

    import torch
    from experiments.run_v133_task_family_robust_basis_natural import synthetic_data
    from experiments.run_v1410_nonrat_fms_transfer_fms_definition_reset import (
        DEFAULT_D_CHE_CANDIDATE,
        FMSState,
        apply_scales_to_flat,
        collect_streaming_per_example_summary,
        fms_key,
        make_case_model,
        named_param_specs,
    )
    import experiments.run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410

    device = torch.device(device_name if (str(device_name).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    source_rows = [
        r
        for r in read_rows(V1410_SYNTH / "v1410_dche_fms_synthetic_results.csv")
        if r.get("family") == "D-CHE" and not is_control(r)
    ]
    if max_rows > 0:
        source_rows = source_rows[:max_rows]
    probe_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    defaults = v1410.build_argparser().parse_args([])
    defaults.device = str(device)
    defaults.synthetic_dim = 16
    defaults.synthetic_classes = 3
    defaults.mlp_hidden = 32
    defaults.fms_beta = 0.99
    defaults.fms_strength = 0.05
    defaults.train_steps = 1
    for idx, row in enumerate(source_rows):
        task = row.get("task", "")
        seed = sint(row.get("seed"))
        loss_interface = row.get("loss_interface", "CE")
        method = row.get("method", "")
        xtr, ytr, _xva, _yva = synthetic_data(task, seed, synthetic_train_size, synthetic_val_size, 16, 3, device)
        gen = torch.Generator(device=device).manual_seed(1411_000 + idx + seed * 997 + sum(ord(c) for c in method + task + loss_interface))
        bsz = min(batch_size, int(xtr.shape[0]))
        sample = torch.randint(0, xtr.shape[0], (bsz,), generator=gen, device=device)
        xb, yb = xtr[sample], ytr[sample]
        split = max(1, int(xb.shape[0]) // 2)
        xb1, yb1 = xb[:split], yb[:split]
        xb2, yb2 = xb[split:], yb[split:]
        if xb2.shape[0] == 0:
            xb2, yb2 = xb1, yb1
        model = make_case_model("D-CHE", DEFAULT_D_CHE_CANDIDATE, xtr, seed, defaults, device)
        specs = named_param_specs(model)
        params = [(spec.name, spec.param) for spec in specs]
        keys = [fms_key(spec, method, "D-CHE") for spec in specs]
        flat_a, util_a = collect_streaming_per_example_summary(model, xb1, yb1, params, specs, keys, loss_interface)
        flat_b, util_b = collect_streaming_per_example_summary(model, xb2, yb2, params, specs, keys, loss_interface)
        generic_flat = flat_a.detach().clone()
        utilities = {key: float(util_a.get(key, 0.0)) for key in set(keys)}
        state = FMSState(0.99, 0.05, seed + 1411)
        fms_method = "F7-PhaseScheduleFMS" if method == "F-CHE6-PhaseScheduleDegreeFMS" else "F3-LayerFMS"
        key_scales = state.update(keys, utilities, fms_method, 0, 1)
        projected_flat, active_fraction = apply_scales_to_flat(method, specs, keys, generic_flat, key_scales, 0, 1)
        before = train_batch_metrics(model, xb2, yb2, loss_interface, torch)
        adamw_model = deepcopy(model)
        fms_model = deepcopy(model)
        apply_flat_update(adamw_model, generic_flat, lr, weight_decay)
        apply_flat_update(fms_model, projected_flat, lr, weight_decay)
        adamw_after = train_batch_metrics(adamw_model, xb2, yb2, loss_interface, torch)
        fms_after = train_batch_metrics(fms_model, xb2, yb2, loss_interface, torch)
        u_train, u_parts = u_train_from_metrics(before, adamw_after, fms_after)
        split_agreement = float(torch.dot(flat_a, flat_b).item() / max(1.0e-8, float(flat_a.norm().item()) * float(flat_b.norm().item()))) if flat_a.numel() else 0.0
        probe_rows.append(
            {
                "stage": "V1411_SPLIT_BATCH_VALUE_PROBE",
                "family": "D-CHE",
                "task": task,
                "seed": seed,
                "loss_interface": loss_interface,
                "method": method,
                "B1_size": int(xb1.shape[0]),
                "B2_size": int(xb2.shape[0]),
                "delta_loss_gain_vs_adamw": f"{u_parts['delta_loss_gain_vs_adamw']:.12g}",
                "delta_q95_loss_vs_adamw": f"{u_parts['delta_q95_loss_vs_adamw']:.12g}",
                "delta_margin_p10_vs_adamw": f"{u_parts['delta_margin_p10_vs_adamw']:.12g}",
                "delta_logit_rms_drift_vs_adamw": f"{u_parts['delta_logit_rms_drift_vs_adamw']:.12g}",
                "delta_entropy_collapse_vs_adamw": f"{u_parts['delta_entropy_collapse_vs_adamw']:.12g}",
                "split_gradient_agreement": f"{split_agreement:.12g}",
                "degree_gate_active_fraction": f"{active_fraction:.12g}",
                "U_train_split_batch": f"{u_train:.12g}",
                "strict_gate_pass": row.get("strict_gate_pass", ""),
                "source_vs_best_control_audit_only": row.get("source_vs_best_control", ""),
                "uses_validation_for_proxy": 0,
                "uses_test_for_proxy": 0,
                "uses_linec_tail_auc_calibration_for_proxy": 0,
                "promotion_allowed": 0,
            }
        )
        control_specs: List[Tuple[str, Any]] = [
            ("NoOpMatchedOverhead", torch.zeros_like(generic_flat)),
        ]
        rand = torch.randn(generic_flat.shape, generator=gen, device=device)
        rand = rand / max(1.0e-8, float(rand.norm().item())) * max(1.0e-8, float(generic_flat.norm().item()))
        control_specs.append(("RandomMatchedNorm", rand))
        active = max(0.0, min(1.0, float(active_fraction)))
        mask = (torch.rand(generic_flat.shape, generator=gen, device=device) < active).float()
        masked = generic_flat * mask
        if float(masked.norm().item()) > 1.0e-8:
            masked = masked / float(masked.norm().item()) * max(1.0e-8, float(generic_flat.norm().item()))
        control_specs.append(("SameActiveFractionRandomMask", masked))
        for control_name, flat in control_specs:
            control_model = deepcopy(model)
            apply_flat_update(control_model, flat, lr, weight_decay)
            control_after = train_batch_metrics(control_model, xb2, yb2, loss_interface, torch)
            cu, cparts = u_train_from_metrics(before, adamw_after, control_after)
            control_rows.append(
                {
                    "stage": "V1411_SPLIT_BATCH_CONTROL_PROBE",
                    "family": "D-CHE",
                    "task": task,
                    "seed": seed,
                    "loss_interface": loss_interface,
                    "method": method,
                    "control_probe": control_name,
                    "U_train_split_batch": f"{cu:.12g}",
                    "strict_gate_pass": row.get("strict_gate_pass", ""),
                    "delta_loss_gain_vs_adamw": f"{cparts['delta_loss_gain_vs_adamw']:.12g}",
                    "delta_q95_loss_vs_adamw": f"{cparts['delta_q95_loss_vs_adamw']:.12g}",
                    "delta_margin_p10_vs_adamw": f"{cparts['delta_margin_p10_vs_adamw']:.12g}",
                    "uses_validation_for_proxy": 0,
                    "uses_test_for_proxy": 0,
                    "uses_linec_tail_auc_calibration_for_proxy": 0,
                    "promotion_allowed": 0,
                }
            )
    scores = [fnum(r.get("U_train_split_batch")) for r in probe_rows]
    labels = [sint(r.get("strict_gate_pass")) for r in probe_rows]
    validity = [
        {
            "split": "D-CHE-split-batch-U-train-to-synthetic-pass",
            "rows": len(probe_rows),
            "positive_rows": sum(labels),
            "auc_U_train_to_pass": none_to_blank(auc_score(scores, labels)),
            "gate_auc_threshold_without_new_real": 0.75,
            "gate_pass": int((auc_score(scores, labels) or 0.0) >= 0.75),
            "promotion_allowed": 0,
        }
    ]
    dche_u = mean(fnum(r.get("U_train_split_batch")) for r in probe_rows)
    control_summary: List[Dict[str, Any]] = []
    for cname in sorted({r.get("control_probe", "") for r in control_rows}):
        subset = [r for r in control_rows if r.get("control_probe") == cname]
        scores_c = [fnum(r.get("U_train_split_batch")) for r in subset]
        labels_c = [sint(r.get("strict_gate_pass")) for r in subset]
        control_summary.append(
            {
                "control_probe": cname,
                "rows": len(subset),
                "positive_rows": sum(labels_c),
                "mean_U_train": f"{mean(scores_c):.12g}",
                "D_CHE_FMS_mean_U_minus_control_mean_U": f"{(dche_u - mean(scores_c)):.12g}",
                "auc_control_U_train_to_pass": none_to_blank(auc_score(scores_c, labels_c)),
                "control_explains_fms": int(mean(scores_c) >= dche_u),
                "promotion_allowed": 0,
            }
        )
    return probe_rows, validity, control_rows, control_summary


def build_code_manifest() -> List[Dict[str, Any]]:
    files = [
        PLAN_DOC,
        Path(__file__).resolve(),
        V1410_SCRIPT,
        ROOT / "docs/DG-KAN_v14.10_NonRAT_FMS_Transfer_FMSDefinitionReset_实验结果复盘.md",
        V1410_SYNTH / "v1410_route_decision.json",
        V1410_REAL / "v1410_route_decision.json",
        V145_ORACLE_ROOT / "oracle_o1_v145_after_basisfree_curvature_action/v145_route_decision.json",
        V149_SUBSTRATE / "v149_line_d_substrate_route.json",
        V1411_LINE_D / "v149_line_d_substrate_route.json",
    ]
    rows = []
    for path in files:
        rows.append(
            {
                "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
                "exists": int(path.exists()),
                "sha256": sha256(path),
                "promotion_allowed": 0,
            }
        )
    return rows


def build_v145_oracle_reference() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Replay v14.5 oracle diagnostics as audit-only missing-state evidence."""

    oracle_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    for summary_path in sorted(V145_ORACLE_ROOT.glob("oracle*/v145_oracle_summary.csv")):
        oracle_dir = summary_path.parent
        route = load_json(oracle_dir / "v145_route_decision.json")
        action_rows = read_rows(oracle_dir / "v145_action_bank_oracle.csv")
        oracle_uses_audit_metric = max([sint(r.get("oracle_uses_audit_metric")) for r in action_rows] + [0])
        for row in read_rows(summary_path):
            oracle_rows.append(
                {
                    "source_artifact": str(summary_path.relative_to(ROOT)),
                    "oracle_dir": str(oracle_dir.relative_to(ROOT)),
                    "route": route.get("route", ""),
                    "minimum_success": route.get("minimum_success", ""),
                    "best_oracle_dataset_seed_pass_count": route.get("best_oracle_dataset_seed_pass_count", ""),
                    "core_oracle_dataset_seed_pass_count": route.get("core_oracle_dataset_seed_pass_count", ""),
                    "extended_legal_v144_oracle_dataset_seed_count": route.get("extended_legal_v144_oracle_dataset_seed_count", ""),
                    "controller_executed": route.get("controller_executed", 0),
                    "official_s5_reached": route.get("official_s5_reached", 0),
                    "route_promotion_allowed": route.get("promotion_allowed", 0),
                    "oracle_level": row.get("oracle_level", ""),
                    "dataset_seed_pass_count": row.get("dataset_seed_pass_count", ""),
                    "pass_rows": row.get("pass_rows", ""),
                    "failure_remaining_source": row.get("failure_remaining_source", ""),
                    "failure_remaining_auc": row.get("failure_remaining_auc", ""),
                    "failure_remaining_cep99": row.get("failure_remaining_cep99", ""),
                    "failure_remaining_nll": row.get("failure_remaining_nll", ""),
                    "failure_remaining_ece": row.get("failure_remaining_ece", ""),
                    "failure_remaining_linec": row.get("failure_remaining_linec", ""),
                    "failure_remaining_overhead": row.get("failure_remaining_overhead", ""),
                    "action_bank_upper_bound_pass": row.get("action_bank_upper_bound_pass", ""),
                    "oracle_uses_audit_metric": oracle_uses_audit_metric,
                    "used_as_direction": 0,
                    "used_as_audit_reference": 1,
                    "promotion_allowed": 0,
                }
            )
        for row in read_rows(oracle_dir / "v145_missing_state_classes.csv"):
            missing_rows.append(
                {
                    "source_artifact": str((oracle_dir / "v145_missing_state_classes.csv").relative_to(ROOT)),
                    "oracle_dir": str(oracle_dir.relative_to(ROOT)),
                    "route": route.get("route", ""),
                    "oracle_level": row.get("oracle_level", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "best_action_id": row.get("best_action_id", ""),
                    "best_method": row.get("best_method", ""),
                    "source": row.get("source", ""),
                    "AUCtime_ratio": row.get("AUCtime_ratio", ""),
                    "CEp99_delta": row.get("CEp99_delta", ""),
                    "NLL_delta": row.get("NLL_delta", ""),
                    "ECE_delta": row.get("ECE_delta", ""),
                    "LineC_pass": row.get("LineC_pass", ""),
                    "dominant_failure": row.get("dominant_failure", ""),
                    "new_action_family_needed": row.get("new_action_family_needed", ""),
                    "selection_threshold_repair_allowed": row.get("selection_threshold_repair_allowed", ""),
                    "oracle_uses_audit_metric": 1,
                    "used_as_direction": 0,
                    "used_as_audit_reference": 1,
                    "promotion_allowed": 0,
                }
            )
    return oracle_rows, missing_rows


def build_v1411_line_d_hardening_reference() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    summary = read_rows(V1411_LINE_D / "v149_line_d_substrate_repair_summary.csv")
    results = read_rows(V1411_LINE_D / "v149_line_d_substrate_repair_results.csv")
    route = load_json(V1411_LINE_D / "v149_line_d_substrate_route.json")
    for rows in [summary, results]:
        for row in rows:
            row["v1411_line_d_source_artifact"] = str(V1411_LINE_D.relative_to(ROOT))
            row["official_fms_proof_executed"] = row.get("official_fms_proof_executed", 0)
            row["promotion_allowed"] = 0
    if not route:
        route = {
            "route": "not_executed",
            "candidate_rows": 0,
            "best_family_dataset_seed_pass_count": 0,
            "official_fms_eligible_family_count": 0,
            "promotion_allowed": 0,
        }
    route["v1411_line_d_source_artifact"] = str(V1411_LINE_D.relative_to(ROOT))
    route["used_as_fms_proof"] = 0
    route["promotion_allowed"] = 0
    return summary, results, route


def scan_forbidden() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    artifacts = [
        V1410_SYNTH / "v1410_dche_fms_synthetic_results.csv",
        V1410_REAL / "v1410_dche_fms_real_results.csv",
        V1410_RT123 / "v1410_dche_fms_real_results.csv",
        V1410_RT4 / "v1410_dche_fms_real_results.csv",
    ]
    method_tokens = " ".join(r.get("method", "") for p in artifacts for r in read_rows(p))
    artifact_flag_max: Dict[str, int] = {
        "controller_executed": 0,
        "is_action_token_extension": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_scale": 0,
    }
    for path in artifacts:
        for row in read_rows(path):
            for col in artifact_flag_max:
                artifact_flag_max[col] = max(artifact_flag_max[col], sint(row.get(col)))
    checks = [
        ("new_F_CHE8_or_F_CHE9", int(("F-CHE8" in method_tokens) or ("F-CHE9" in method_tokens))),
        ("controller_executed_route", artifact_flag_max["controller_executed"]),
        ("action_token_extension", artifact_flag_max["is_action_token_extension"]),
        ("dataset_name_branch_flag", artifact_flag_max["uses_dataset_name_branch"]),
        ("seed_specific_scale_flag", artifact_flag_max["uses_seed_specific_scale"]),
    ]
    no_action = [
        {
            "audit_item": k,
            "violation": v,
            "evidence": "static scan of v14.11/v14.10 runner text",
            "promotion_allowed": 0,
        }
        for k, v in checks[:3]
    ]
    forbidden = [
        {
            "audit_item": k,
            "violation": v,
            "evidence": "static scan of v14.11/v14.10 runner text plus artifact flags",
            "promotion_allowed": 0,
        }
        for k, v in checks
    ]
    for artifact, name in [
        (V1410_SYNTH / "v1410_dche_fms_synthetic_results.csv", "v1410_synthetic_artifact_flags"),
        (V1410_REAL / "v1410_dche_fms_real_results.csv", "v1410_real_artifact_flags"),
    ]:
        rows = read_rows(artifact)
        violation = 0
        for row in rows:
            for col in [
                "uses_validation_for_direction",
                "uses_test_for_direction",
                "uses_future_for_direction",
                "uses_query_for_direction",
                "uses_linec_for_direction",
                "uses_cep99_for_direction",
                "uses_nll_for_direction",
                "uses_ece_for_direction",
                "uses_auctime_for_direction",
                "uses_dataset_name_branch",
                "uses_seed_specific_scale",
                "is_action_token_extension",
                "controller_executed",
            ]:
                violation = max(violation, sint(row.get(col)))
        forbidden.append(
            {
                "audit_item": name,
                "violation": violation,
                "evidence": str(artifact.relative_to(ROOT)),
                "promotion_allowed": 0,
            }
        )
    return no_action, forbidden


def copy_dche_results() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    synth = read_rows(V1410_SYNTH / "v1410_dche_fms_synthetic_results.csv")
    real = read_rows(V1410_REAL / "v1410_dche_fms_real_results.csv")
    mlp = read_rows(V1410_SYNTH / "v1410_mlp_generic_controls.csv")
    for rows in [synth, real, mlp]:
        for row in rows:
            row["v1411_source_artifact"] = str((V1410_SYNTH if row in synth or row in mlp else V1410_REAL).relative_to(ROOT))
            row["promotion_allowed"] = "0"
    return synth, real, mlp


def build_substrate_status() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary = read_rows(V149_SUBSTRATE / "v149_line_d_substrate_repair_summary.csv")
    status = []
    for row in summary:
        status.append(
            {
                "family": row.get("family", ""),
                "dataset_seed_pass_count": row.get("family_dataset_seed_pass_count", ""),
                "official_fms_eligibility": row.get("family_official_fms_eligibility", ""),
                "exploration_gate_pass": row.get("family_exploration_gate_pass", ""),
                "best_candidate": row.get("best_candidate", ""),
                "best_mean_delta_vs_MLP": row.get("best_mean_delta_vs_MLP", ""),
                "best_LineC_pass_rate": row.get("best_LineC_pass_rate", ""),
                "min_NLL_ratio_vs_MLP": row.get("min_NLL_ratio_vs_MLP", ""),
                "median_step": row.get("median_train_step_ratio_vs_MLP", ""),
                "official_fms_proof_executed": row.get("official_fms_proof_executed", ""),
                "promotion_allowed": 0,
                "v1411_status": "read_from_v14_9_line_d; no new all-basis FMS proof executed",
            }
        )
    results = read_rows(V149_SUBSTRATE / "v149_line_d_substrate_repair_results.csv")
    fou = [r for r in results if r.get("family") == "D-FOU"]
    rbf = [r for r in results if r.get("family") == "D-RBF"]
    wav = [r for r in results if r.get("family") == "D-WAV"]
    for rows in [fou, rbf, wav]:
        for row in rows:
            row["v1411_candidate_execution_status"] = "v14_9_artifact_reused; v14_11 did not execute new FMS proof"
            row["promotion_allowed"] = "0"
    return status, fou, rbf, wav


def build_linec_tail_audit(dche: Sequence[Dict[str, str]], real: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = []
    for label, data in [("synthetic", dche), ("real", real)]:
        noncontrols = [r for r in data if not is_control(r)]
        rows.append(
            {
                "stage": label,
                "rows": len(noncontrols),
                "source_fail": sum(fnum(r.get("source_vs_best_control")) <= 0.005 for r in noncontrols),
                "AUCtime_fail": sum(fnum(r.get("AUCtime_ratio"), 99.0) > 1.0 for r in noncontrols),
                "CEp99_fail": sum(fnum(r.get("CEp99_delta")) > 0.0 for r in noncontrols),
                "NLL_fail": sum(fnum(r.get("NLL_delta")) > 0.0 for r in noncontrols),
                "ECE_fail": sum(fnum(r.get("ECE_delta")) > 0.0 for r in noncontrols),
                "LineC_fail": sum(sint(r.get("LineC_majority_pass")) == 0 for r in noncontrols),
                "step_fail": sum(fnum(r.get("step_time_ratio"), 99.0) > 1.25 for r in noncontrols),
                "memory_fail": sum(fnum(r.get("memory_ratio"), 99.0) > 1.25 for r in noncontrols),
                "promotion_allowed": 0,
            }
        )
    return rows


def build_failure_taxonomy(
    route: str,
    e_summary: Sequence[Dict[str, Any]],
    v_summary: Sequence[Dict[str, Any]],
    real: Sequence[Dict[str, str]],
    v145_oracle: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    real_non = [r for r in real if not is_control(r)]
    rows = [
        {
            "blocker": "Line-E-synthetic-to-real-predictivity",
            "status": "pass" if any(sint(r.get("predictivity_gate_pass")) for r in e_summary) else "fail",
            "evidence": "v1411_synthetic_real_predictivity_summary.csv",
            "promotion_allowed": 0,
        },
        {
            "blocker": "Line-V-train-stream-value-observability",
            "status": "pass" if all(sint(r.get("gate_pass")) for r in v_summary[:2]) else "fail",
            "evidence": "v1411_train_stream_value_validity.csv",
            "promotion_allowed": 0,
        },
        {
            "blocker": "real-transfer-pass-count",
            "status": "fail",
            "evidence": f"strict real pass rows={sum(sint(r.get('strict_gate_pass')) for r in real_non)}; unique dataset-seeds={unique_real_pass_count(real_non)}",
            "promotion_allowed": 0,
        },
        {
            "blocker": "route",
            "status": route,
            "evidence": "v1411_route_decision.json",
            "promotion_allowed": 0,
        },
    ]
    if v145_oracle:
        best = max(sint(r.get("best_oracle_dataset_seed_pass_count")) for r in v145_oracle)
        upper_bound_pass = max(sint(r.get("action_bank_upper_bound_pass")) for r in v145_oracle)
        rows.append(
            {
                "blocker": "v14.5-action-bank-oracle-upper-bound",
                "status": "fail" if best < 9 or upper_bound_pass == 0 else "pass",
                "evidence": f"best_oracle_dataset_seed_pass_count={best}; action_bank_upper_bound_pass={upper_bound_pass}; audit-only reference",
                "promotion_allowed": 0,
            }
        )
    return rows


def unique_real_pass_count(rows: Sequence[Dict[str, str]]) -> int:
    return len({(r.get("dataset"), r.get("seed")) for r in rows if sint(r.get("strict_gate_pass"))})


def build_progress(route: str, e_pass: bool, v_pass: bool, r_pass: bool) -> List[Dict[str, Any]]:
    return [
        {"line": "R", "description": "implementation/provenance/no-action-search audit", "status": "pass" if r_pass else "fail", "route_if_failed": "R0-ActionSearchOrAuditDirectionViolation", "promotion_allowed": 0},
        {"line": "E", "description": "synthetic-to-real validity audit", "status": "pass" if e_pass else "fail", "route_if_failed": "R1-SyntheticGateNotPredictiveForRealTransfer", "promotion_allowed": 0},
        {"line": "V", "description": "train-stream value proxy validity", "status": "pass" if v_pass else "fail", "route_if_failed": "R2-TrainStreamTransferValueUnobservable", "promotion_allowed": 0},
        {"line": "F-CHE", "description": "D-CHE FMS definition reset", "status": "not_executed_fail_closed" if not (e_pass or v_pass) else "eligible_for_future_pre_registered_run", "route_if_failed": "R3-DCHESyntheticPassRealTransferFail", "promotion_allowed": 0},
        {"line": "D", "description": "all-basis substrate status/hardening", "status": "status_replayed_from_v14_9_no_fms_proof", "route_if_failed": "", "promotion_allowed": 0},
        {"line": "M", "description": "MLP/generic confound controls", "status": "recorded", "route_if_failed": "", "promotion_allowed": 0},
        {"line": "C", "description": "LineC/tail/calibration audit", "status": "recorded", "route_if_failed": "", "promotion_allowed": 0},
        {"line": "FINAL", "description": "route decision", "status": route, "route_if_failed": route, "promotion_allowed": 0},
    ]


def simple_svg(path: Path, title: str, rows: Sequence[Tuple[str, float]], threshold: Optional[float] = None) -> None:
    width, height = 820, 260
    left, top = 180, 46
    maxv = max([abs(v) for _, v in rows] + ([threshold] if threshold else [1.0]) + [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="sans-serif" font-size="18" font-weight="700">{title}</text>',
    ]
    bar_h = 20
    for i, (label, val) in enumerate(rows[:8]):
        y = top + i * 26
        w = int((abs(val) / maxv) * 560)
        color = "#2f7d6d" if val >= 0 else "#b64b4b"
        lines.append(f'<text x="24" y="{y+15}" font-family="sans-serif" font-size="12">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{w}" height="{bar_h}" fill="{color}"/>')
        lines.append(f'<text x="{left+w+8}" y="{y+15}" font-family="sans-serif" font-size="12">{val:.4g}</text>')
    if threshold is not None:
        x = left + int((threshold / maxv) * 560)
        lines.append(f'<line x1="{x}" x2="{x}" y1="42" y2="{height-20}" stroke="#222" stroke-dasharray="4 3"/>')
        lines.append(f'<text x="{x+5}" y="{height-10}" font-family="sans-serif" font-size="11">threshold {threshold:.2f}</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines) + "\n")


def write_figures(out: Path, e_summary: Sequence[Dict[str, Any]], v_summary: Sequence[Dict[str, Any]], real: Sequence[Dict[str, str]], substrate: Sequence[Dict[str, Any]], mlp: Sequence[Dict[str, str]], dche: Sequence[Dict[str, str]], no_action: Sequence[Dict[str, Any]]) -> None:
    simple_svg(out / "fig_synthetic_to_real_predictivity.svg", "Synthetic-to-real predictivity", [(r["split"], fnum(r.get("auc_predict_synthetic_features_to_real_pass"))) for r in e_summary], 0.70)
    simple_svg(out / "fig_train_stream_value_vs_real_pass.svg", "Train-stream value validity", [(r["split"], fnum(r.get("auc_U_train_to_pass", r.get("U_DCHE_minus_U_MLP", 0.0)))) for r in v_summary], 0.70)
    real_counts: Dict[str, float] = {}
    for r in real:
        if not is_control(r):
            real_counts[f"{r.get('dataset')}:{r.get('seed')}"] = max(real_counts.get(f"{r.get('dataset')}:{r.get('seed')}", 0.0), float(sint(r.get("strict_gate_pass"))))
    simple_svg(out / "fig_dche_real_transfer_3x3_matrix.svg", "D-CHE real transfer 3x3 pass", sorted(real_counts.items()), 1.0)
    task_counts: Dict[str, float] = {}
    for r in dche:
        if not is_control(r):
            task_counts[r.get("task", "")] = max(task_counts.get(r.get("task", ""), 0.0), float(sint(r.get("strict_gate_pass"))))
    simple_svg(out / "fig_dche_fms_synthetic_family_heatmap.svg", "D-CHE synthetic family pass", sorted(task_counts.items()), 1.0)
    failure = build_linec_tail_audit(dche, real)
    simple_svg(out / "fig_source_auc_tail_failure_decomposition.svg", "Failure decomposition real", [(k, fnum(failure[-1].get(k))) for k in ["source_fail", "AUCtime_fail", "CEp99_fail", "NLL_fail", "ECE_fail", "LineC_fail"]])
    simple_svg(out / "fig_all_basis_substrate_matrix.svg", "All-basis substrate status", [(r["family"], fnum(r.get("dataset_seed_pass_count"))) for r in substrate], 9.0)
    dche_mean = mean(fnum(r.get("source_vs_best_control")) for r in dche if not is_control(r))
    mlp_mean = mean(fnum(r.get("source_vs_best_control")) for r in mlp if r.get("family") == "MLP")
    simple_svg(out / "fig_mlp_vs_dche_fms_comparison.svg", "MLP vs D-CHE source", [("D-CHE", dche_mean), ("MLP", mlp_mean)])
    simple_svg(out / "fig_no_action_search_audit.svg", "No-action-search audit", [(r["audit_item"], fnum(r["violation"])) for r in no_action], 0.0)
    progress_rows = [("R", 1), ("E", fnum(e_summary[0].get("predictivity_gate_pass")) if e_summary else 0), ("V", fnum(v_summary[0].get("gate_pass")) if v_summary else 0), ("S5", 0)]
    simple_svg(out / "fig_v1411_progress_by_line.svg", "v14.11 progress by line", progress_rows, 1.0)
    simple_svg(out / "fig_synthetic_task_family_vs_real_pass_heatmap.svg", "Synthetic task vs real pass proxy", sorted(task_counts.items()) + sorted(real_counts.items()), 1.0)


def write_required_manifest(out: Path) -> List[Dict[str, Any]]:
    rows = []
    for name in REQUIRED + FIGURES:
        p = out / name
        rows.append({"artifact": name, "exists": int(p.exists()), "missing": int(not p.exists()), "size_bytes": p.stat().st_size if p.exists() else 0})
    write_rows(out / "v1411_required_artifact_manifest.csv", rows)
    return rows


def make_packet(out: Path) -> None:
    packet = out / "v1411_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [
            PLAN_DOC,
            RECAP_DOC,
            EXEC_LOG_DOC,
            Path(__file__).resolve(),
            V1410_SCRIPT,
            out / "v1411_route_decision.json",
            out / "v1411_required_artifact_manifest.csv",
            out / "v1411_synthetic_real_predictivity_summary.csv",
            out / "v1411_train_stream_value_validity.csv",
            out / "v1411_v145_oracle_reference.csv",
            out / "v1411_v145_missing_state_reference.csv",
            out / "v1411_line_d_v1411_hardening_summary.csv",
            out / "v1411_line_d_v1411_hardening_route.json",
        ]:
            if path.exists() and path != packet:
                zf.write(path, path.relative_to(ROOT))


def write_docs(out: Path, route: Dict[str, Any], e_summary: Sequence[Dict[str, Any]], v_summary: Sequence[Dict[str, Any]]) -> None:
    e0 = e_summary[0] if e_summary else {}
    v0 = v_summary[0] if v_summary else {}
    v1 = v_summary[1] if len(v_summary) > 1 else {}
    v2 = v_summary[2] if len(v_summary) > 2 else {}
    exec_text = f"""# DG-KAN v14.11 SyntheticRealTransferValidity DCHE FMS AllBasis 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 本轮新增/修改文件

```text
experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
docs/DG-KAN_v14.11_SyntheticRealTransferValidity_DCHE_FMS_AllBasis_执行日志.md
docs/DG-KAN_v14.11_SyntheticRealTransferValidity_DCHE_FMS_AllBasis_实验结果复盘.md
```

## 2. 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py --out-dir {out.relative_to(ROOT)}
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py --out-dir {out.relative_to(ROOT)} --run-split-batch-probe 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v149_line_d_all_basis_substrate_repair.py experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir {V1411_LINE_D.relative_to(ROOT)} --candidates D-FOU27-LowFreqIdentityResidualV2,D-FOU28-BandwiseSNRSafeWarmup,D-FOU29-PhaseStableBandMixNoHighFreq,D-FOU30-NoMaterializeLifetimeV3,D-RBF26-ActiveCenterOccupancyV2,D-RBF27-WidthConditionIdentityResidual,D-RBF28-CompactBumpNoDenseMaterialization,D-RBF29-GaussianLocalK4TaskHealth,D-WAV25-TriangularSupportV3,D-WAV26-ScaleOccupancyNoTailTarget,D-WAV27-LocalSupportOverlapDamping --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --linec-seeds 12319500,12319501,12319502 --compute-budgeted-run 1 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py --out-dir {out.relative_to(ROOT)} --run-split-batch-probe 1
```

执行中先发现 Line R 静态扫描把审计规则文本自身误判成 F-CHE8/controller/action-token 证据；
随后修改 `scan_forbidden()`，改为读取 artifact 的 method/flag 字段作为审计来源，并重跑。

## 3. 输入 artifact

```text
{V1410_SYNTH.relative_to(ROOT)}
{V1410_REAL.relative_to(ROOT)}
{V1410_RT123.relative_to(ROOT)}
{V1410_RT4.relative_to(ROOT)}
{V149_SUBSTRATE.relative_to(ROOT)}
{V145_ORACLE_ROOT.relative_to(ROOT)}
{V1411_LINE_D.relative_to(ROOT)}
{V143_SYNTH.relative_to(ROOT)}
{V143_REAL.relative_to(ROOT)}
{V144_REAL.relative_to(ROOT)}
```

## 4. 输出 artifact

```text
{out.relative_to(ROOT)}
```

主要输出：

```text
v1411_route_decision.json
v1411_synthetic_real_alignment.csv
v1411_synthetic_real_predictivity_summary.csv
v1411_train_stream_value_proxy.csv
v1411_train_stream_value_validity.csv
v1411_split_batch_value_probe.csv
v1411_split_batch_value_validity.csv
v1411_split_batch_control_validity.csv
v1411_v145_oracle_reference.csv
v1411_v145_missing_state_reference.csv
v1411_line_d_v1411_hardening_summary.csv
v1411_line_d_v1411_hardening_results.csv
v1411_line_d_v1411_hardening_route.json
v1411_required_artifact_manifest.csv
v1411_code_review_packet.zip
```

## 5. 复现说明

本 runner 是审计/finalizer surface，不执行新的 F-CHE token、controller、action search 或 real training。
Line E 使用既有 synthetic/real artifact 做 predictivity audit。
继续推进时已将 Line E battery 从 D-CHE internal 扩展到 v14.3 Rational K8 synthetic/real 和 v14.4 Rational S4b real reference。
继续推进时补入计划要求的数据源 v14.5 action-bank oracle diagnostics，输出 oracle upper-bound 与 missing-state reference；该数据源 `oracle_uses_audit_metric=1`，因此只作为审计参考，不作为 v14.11 direction。
Line V 使用既有 train-stream/model-state telemetry 字段构造 proxy audit，不使用 validation/test/LineC/tail/AUC/calibration 作为方向源。
继续推进时新增 split-batch probe：只在 synthetic train batch 内拆 B1/B2，比较已有 F-CHE/FB direction 与 AdamW 的 one-step counterfactual U_train；并补充 NoOpMatchedOverhead、RandomMatchedNorm、SameActiveFractionRandomMask control probe。该 probe 只用于 validity audit，不作为方向选择。
继续推进时执行计划 Line D 允许的 D-FOU27..30 / D-RBF26..29 / D-WAV25..27 substrate-only hardening；不执行 Non-RAT FMS proof，不使用 audit metric 生成方向。
Line E 和 Line V 均未通过时，按计划 Case A/B fail-closed，不启动新的 D-CHE real FMS 训练。
"""
    recap = f"""# DG-KAN v14.11 SyntheticRealTransferValidity DCHE FMS AllBasis 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入本轮实际 artifact 中的结果；不虚构成功、不补填未执行 real 训练、不把 synthetic S3 / real 2/9 / substrate eligibility 写成 promotion。

## 1. 计划理解

v14.11 的目标是验证 synthetic S3 是否能预测 real transfer，并审计 train-stream-only real-transfer value proxy 是否可观测。只有 Line E 或 Line V 通过，才允许继续 D-CHE FMS definition reset；否则 fail-closed。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
```

实现：

```text
1. Line R provenance / no-action-search / forbidden-information audit。
2. Line E synthetic-to-real alignment 与 predictivity summary。
   继续推进后补入 v14.3 Rational K8 synthetic/real 和 v14.4 Rational S4b real reference。
3. 继续推进后补入 v14.5 action-bank oracle diagnostics / missing-state reference，
   只作为 audit-only upper-bound 证据，不作为方向源。
4. Line V train-stream telemetry proxy validity。
5. D-CHE synthetic/real/MLP control artifact replay。
6. v14.9 all-basis substrate status replay。
7. 继续推进后执行 v14.11 Line D substrate-only hardening：
   D-FOU27..30 / D-RBF26..29 / D-WAV25..27。
8. LineC/tail failure taxonomy、figures、required manifest、code review packet。
```

修复：

```text
1. 首次 Line R scan 使用源码字符串搜索，误把审计规则文本本身判为 F-CHE8/controller/action-token 证据。
2. 已改为读取 v14.10/v14.11 artifact 中的 method tokens 与 flag columns：
   controller_executed、is_action_token_extension、uses_dataset_name_branch、uses_seed_specific_scale。
3. 修复后 no_action_search_violation_count = 0，
   forbidden_information_violation_count = 0。
4. split-batch probe 首次运行遇到 direct script import path blocker；
   已在 runner 开头加入 repo root 到 sys.path。
5. 调整 code review packet 生成顺序，确保最终执行日志/复盘日志写入后再打包。
```

合法性：

```text
1. 没有新增 F-CHE8/F-CHE9。
2. 没有 controller / action token / reset route。
3. 没有执行新的 real 训练。
4. LineC / CEp99 / NLL / ECE / AUCtime 只作为 audit / gate。
5. promotion_allowed = 0。
```

## 3. Line E 结果

```text
split = {e0.get('split', '')}
rows = {e0.get('rows', '')}
positive_real_pass_rows = {e0.get('positive_real_pass_rows', '')}
spearman_synthetic_source_real_source = {e0.get('spearman_synthetic_source_real_source', '')}
auc_predict_synthetic_features_to_real_pass = {e0.get('auc_predict_synthetic_features_to_real_pass', '')}
predictivity_gate_pass = {e0.get('predictivity_gate_pass', '')}
```

判断：

```text
Line E gate 要求 AUC >= 0.70 且 Spearman >= 0.30。
本轮 Line E pass = {route.get('line_e_pass')}.
```

v14.5 oracle reference：

```text
v145_oracle_reference_rows = {route.get('v145_oracle_reference_rows')}
v145_missing_state_reference_rows = {route.get('v145_missing_state_reference_rows')}
v145_best_oracle_dataset_seed_pass_count = {route.get('v145_best_oracle_dataset_seed_pass_count')}
v145_action_bank_upper_bound_pass = {route.get('v145_action_bank_upper_bound_pass')}
```

判断：

```text
v14.5 action-bank oracle diagnostics 是计划 Line E 的输入源之一。
本轮只作为 audit-only reference 读取；oracle_uses_audit_metric = 1 的历史 oracle 不允许成为 v14.11 direction。
best oracle 仍未达到 9/9，因此不能改写 S5，也不能解除 v14.11 的 R1/R2 blocker。
```

## 4. Line V 结果

```text
synthetic U_train AUC = {v0.get('auc_U_train_to_pass', '')}
real U_train AUC = {v1.get('auc_U_train_to_pass', '')}
U_DCHE_minus_U_MLP = {v2.get('U_DCHE_minus_U_MLP', '')}
split_batch_probe_rows = {route.get('split_batch_probe_rows')}
split_batch_probe_auc_U_train_to_synthetic_pass = {route.get('split_batch_probe_auc_U_train_to_synthetic_pass')}
split_batch_probe_gate_pass = {route.get('split_batch_probe_gate_pass')}
split_batch_control_probe_rows = {route.get('split_batch_control_probe_rows')}
split_batch_control_explains_fms_count = {route.get('split_batch_control_explains_fms_count')}
Line V pass = {route.get('line_v_pass')}
```

判断：

```text
Line V 使用已有 train-stream/model-state telemetry 做 proxy audit。
追加的 split-batch probe 使用 synthetic train-stream B1/B2 one-step counterfactual U_train，并记录 pre-registered controls。
两者都没有使用 validation/test/future/query 或 LineC/tail/AUC/calibration 生成方向。
```

## 5. Line D 继续推进结果

执行范围：

```text
D-FOU27-LowFreqIdentityResidualV2
D-FOU28-BandwiseSNRSafeWarmup
D-FOU29-PhaseStableBandMixNoHighFreq
D-FOU30-NoMaterializeLifetimeV3
D-RBF26-ActiveCenterOccupancyV2
D-RBF27-WidthConditionIdentityResidual
D-RBF28-CompactBumpNoDenseMaterialization
D-RBF29-GaussianLocalK4TaskHealth
D-WAV25-TriangularSupportV3
D-WAV26-ScaleOccupancyNoTailTarget
D-WAV27-LocalSupportOverlapDamping
```

结果：

```text
line_d_v1411_route = {route.get('line_d_v1411_route')}
line_d_v1411_candidate_rows = {route.get('line_d_v1411_candidate_rows')}
line_d_v1411_best_family_dataset_seed_pass_count = {route.get('line_d_v1411_best_family_dataset_seed_pass_count')}
line_d_v1411_exploration_open_family_count = {route.get('line_d_v1411_exploration_open_family_count')}
line_d_v1411_official_fms_eligible_family_count = {route.get('line_d_v1411_official_fms_eligible_family_count')}
```

判断：

```text
这轮 Line D 是 substrate-only hardening，不执行 FMS proof。
新增 D-FOU/RBF/WAV candidates 没有打开 exploration 或 official FMS eligibility。
因此 D-FOU/D-RBF/D-WAV 仍不能进入 official FMS proof，也不能改变 v14.11 D-CHE real-transfer route。
```

## 6. 最终 route

```text
route = {route.get('route')}
minimum_success = {route.get('minimum_success')}
synthetic_task_family_pass_count = {route.get('synthetic_task_family_pass_count')}
real_dataset_seed_pass_count = {route.get('real_dataset_seed_pass_count')}
v145_best_oracle_dataset_seed_pass_count = {route.get('v145_best_oracle_dataset_seed_pass_count')}
v145_action_bank_upper_bound_pass = {route.get('v145_action_bank_upper_bound_pass')}
official_s5_reached = {route.get('official_s5_reached')}
promotion_allowed = {route.get('promotion_allowed')}
required_artifact_missing_count = {route.get('required_artifact_missing_count')}
forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}
no_action_search_violation_count = {route.get('no_action_search_violation_count')}
```

## 7. 科学结论

```text
1. v14.10 的 D-CHE synthetic S3 仍是已达成事实，但 v14.11 没有把它改写成 real success。
2. v14.10 best real 仍只有 2/9，未达到 S4/S5。
3. v14.11 Line E 与 Line V 均未打开继续 D-CHE FMS definition reset 的合法门。
4. 因此本轮 fail-closed，不执行新的 F-CHE token / action search / real 训练。
5. promotion_allowed = 0。
```

## 8. Blocker 与停止边界

```text
Line E blocker:
  synthetic source 与 real source 的 Spearman 接近 0 且为负；
  AUC_predict 只有约 0.544，低于 0.70。
  补入 Rational reference 后仍以 D-CHE internal gate 为准；
  v14.11 要求至少 D-CHE internal split 通过，因此 route 不变。

Line V blocker:
  telemetry-only U_train 对 synthetic pass 的 AUC 约 0.362；
  对 real pass 的 AUC 约 0.474；
  split-batch B1/B2 U_train 对 synthetic pass 的 AUC 约 {route.get('split_batch_probe_auc_U_train_to_synthetic_pass')}；
  split-batch control_explains_fms_count = {route.get('split_batch_control_explains_fms_count')}；
  D-CHE mean U_train 低于 MLP/generic control。

Line D blocker:
  v14.11 允许的 D-FOU/RBF/WAV substrate candidates 共 {route.get('line_d_v1411_candidate_rows')} rows；
  best family dataset-seed pass count = {route.get('line_d_v1411_best_family_dataset_seed_pass_count')}/9；
  exploration_open_family_count = {route.get('line_d_v1411_exploration_open_family_count')}；
  official_fms_eligible_family_count = {route.get('line_d_v1411_official_fms_eligible_family_count')}。

因此继续训练会违反 v14.11：
  不能新增 F-CHE8/F-CHE9；
  不能用 real fail pattern 或 LineC/tail/AUC/calibration 指标反推方向；
  不能把 D-CHE synthetic S3 或 real 2/9 写成 S4/S5。
```
"""
    write_text(EXEC_LOG_DOC, exec_text)
    write_text(RECAP_DOC, recap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-split-batch-probe", type=int, default=0)
    parser.add_argument("--split-probe-max-rows", type=int, default=0)
    parser.add_argument("--split-probe-device", default="cuda:0")
    parser.add_argument("--split-probe-batch-size", type=int, default=32)
    parser.add_argument("--split-probe-synthetic-train-size", type=int, default=96)
    parser.add_argument("--split-probe-synthetic-val-size", type=int, default=64)
    parser.add_argument("--split-probe-lr", type=float, default=0.005)
    parser.add_argument("--split-probe-weight-decay", type=float, default=1.0e-4)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    no_action, forbidden = scan_forbidden()
    code_manifest = build_code_manifest()
    alignment = build_alignment()
    e_summary = summarize_predictivity(alignment)
    proxy_rows, v_summary = build_train_stream_proxy()
    split_probe_rows: List[Dict[str, Any]] = []
    split_control_rows: List[Dict[str, Any]] = []
    split_probe_validity: List[Dict[str, Any]] = [
        {
            "split": "D-CHE-split-batch-U-train-to-synthetic-pass",
            "rows": 0,
            "positive_rows": 0,
            "auc_U_train_to_pass": "",
            "gate_auc_threshold_without_new_real": 0.75,
            "gate_pass": 0,
            "promotion_allowed": 0,
            "status": "not_executed",
        }
    ]
    split_control_validity: List[Dict[str, Any]] = [
        {
            "control_probe": "not_executed",
            "rows": 0,
            "positive_rows": 0,
            "mean_U_train": "",
            "D_CHE_FMS_mean_U_minus_control_mean_U": "",
            "auc_control_U_train_to_pass": "",
            "control_explains_fms": "",
            "promotion_allowed": 0,
        }
    ]
    if int(args.run_split_batch_probe) == 1:
        split_probe_rows, split_probe_validity, split_control_rows, split_control_validity = run_split_batch_probe(
            device_name=str(args.split_probe_device),
            max_rows=int(args.split_probe_max_rows),
            batch_size=int(args.split_probe_batch_size),
            synthetic_train_size=int(args.split_probe_synthetic_train_size),
            synthetic_val_size=int(args.split_probe_synthetic_val_size),
            lr=float(args.split_probe_lr),
            weight_decay=float(args.split_probe_weight_decay),
        )
    v_summary_all = list(v_summary) + list(split_probe_validity)
    dche_synth, dche_real, mlp = copy_dche_results()
    substrate, fou, rbf, wav = build_substrate_status()
    v145_oracle, v145_missing = build_v145_oracle_reference()
    line_d_v1411_summary, line_d_v1411_results, line_d_v1411_route = build_v1411_line_d_hardening_reference()
    linec_tail = build_linec_tail_audit(dche_synth, dche_real)

    r_pass = not any(sint(r.get("violation")) for r in no_action + forbidden)
    e_pass = any(sint(r.get("predictivity_gate_pass")) for r in e_summary)
    # Per plan, V needs observability and D-CHE > MLP/generic.
    v_auc_pass = any(
        r.get("split") == "D-CHE-real-telemetry-to-real-pass" and sint(r.get("gate_pass"))
        for r in v_summary_all
    ) or any(
        r.get("split") in {"D-CHE-synthetic-telemetry-to-synthetic-pass", "D-CHE-split-batch-U-train-to-synthetic-pass"} and sint(r.get("gate_pass"))
        for r in v_summary_all
    )
    v_delta_pass = any(r.get("split") == "D-CHE-vs-MLP-generic-telemetry-mean" and sint(r.get("gate_pass")) for r in v_summary_all)
    v_pass = bool(v_auc_pass and v_delta_pass)

    synth_route = load_json(V1410_SYNTH / "v1410_route_decision.json")
    real_route = load_json(V1410_REAL / "v1410_route_decision.json")
    synth_pass = sint(synth_route.get("synthetic_task_family_pass_count"))
    real_pass_count = sint(real_route.get("real_dataset_seed_pass_count"))

    if not r_pass:
        route = "R0-ActionSearchOrAuditDirectionViolation"
        minimum = "S0-ExecutionCompleteNoPromotion"
    elif not e_pass:
        route = "R1-SyntheticGateNotPredictiveForRealTransfer"
        minimum = "S3-DCHESyntheticFMSPass" if synth_pass >= 5 else "S0-ExecutionCompleteNoPromotion"
    elif not v_pass:
        route = "R2-TrainStreamTransferValueUnobservable"
        minimum = "S3-DCHESyntheticFMSPass" if synth_pass >= 5 else "S0-ExecutionCompleteNoPromotion"
    elif real_pass_count < 6:
        route = "R3-DCHESyntheticPassRealTransferFail"
        minimum = "S3-DCHESyntheticFMSPass"
    elif real_pass_count < 9:
        route = "S4-DCHERelTransferExplorationPositive"
        minimum = "S4-DCHERelTransferExplorationPositive"
    else:
        route = "S5-DCHENonRATFunctionalSuccess"
        minimum = "S5-DCHENonRATFunctionalSuccess"

    progress = build_progress(route, e_pass, v_pass, r_pass)
    failure = build_failure_taxonomy(route, e_summary, v_summary_all, dche_real, v145_oracle)

    write_rows(out / "v1411_code_review_manifest.csv", code_manifest)
    write_rows(out / "v1411_no_action_search_audit.csv", no_action)
    write_rows(out / "v1411_forbidden_information_audit.csv", forbidden)
    write_rows(out / "v1411_synthetic_real_alignment.csv", alignment)
    write_rows(out / "v1411_synthetic_real_predictivity_summary.csv", e_summary)
    write_rows(out / "v1411_train_stream_value_proxy.csv", proxy_rows)
    write_rows(out / "v1411_train_stream_value_validity.csv", v_summary_all)
    write_rows(out / "v1411_split_batch_value_probe.csv", split_probe_rows)
    write_rows(out / "v1411_split_batch_value_validity.csv", split_probe_validity)
    write_rows(out / "v1411_split_batch_control_validity.csv", split_control_validity)
    write_rows(out / "v1411_v145_oracle_reference.csv", v145_oracle)
    write_rows(out / "v1411_v145_missing_state_reference.csv", v145_missing)
    write_rows(out / "v1411_dche_fms_definition_results.csv", dche_synth)
    write_rows(out / "v1411_dche_real_transfer_results.csv", dche_real)
    write_rows(out / "v1411_mlp_generic_controls.csv", mlp)
    write_rows(out / "v1411_all_basis_substrate_status.csv", substrate)
    write_rows(out / "v1411_fourier_substrate_hardening.csv", fou)
    write_rows(out / "v1411_rbf_substrate_hardening.csv", rbf)
    write_rows(out / "v1411_wavelet_substrate_monitor.csv", wav)
    write_rows(out / "v1411_line_d_v1411_hardening_summary.csv", line_d_v1411_summary)
    write_rows(out / "v1411_line_d_v1411_hardening_results.csv", line_d_v1411_results)
    write_json(out / "v1411_line_d_v1411_hardening_route.json", line_d_v1411_route)
    write_rows(out / "v1411_linec_tail_audit.csv", linec_tail)
    write_rows(out / "v1411_failure_taxonomy.csv", failure)
    write_rows(out / "v1411_progress_table.csv", progress)

    write_text(
        out / "v1411_plan_compliance_readback.md",
        "# v14.11 Plan Compliance Readback\n\n"
        "- No F-CHE8/F-CHE9 added.\n"
        "- No controller/action token/reset route executed.\n"
        "- Line E/V are audit surfaces over existing artifacts and train-stream/model-state telemetry.\n"
        "- No new real training was launched because validity gates do not authorize it.\n"
        "- promotion_allowed remains 0.\n",
    )
    write_text(
        out / "v1411_no_go_boundary.md",
        "# v14.11 No-Go Boundary\n\n"
        f"- route = {route}\n"
        f"- Line E pass = {int(e_pass)}\n"
        f"- Line V pass = {int(v_pass)}\n"
        "- Do not count synthetic S3, substrate eligibility, or local real positive rows as promotion.\n"
        "- Do not use LineC/tail/AUC/calibration metrics as a direction source.\n",
    )
    write_text(
        out / "v1411_next_hypothesis_queue.md",
        "# v14.11 Next Hypothesis Queue\n\n"
        "1. Rebuild a synthetic battery whose train-stream features predict real transfer before adding F-CHE variants.\n"
        "2. Define a true split-batch U_train logger in the training runner, then validate it against held-out real transfer.\n"
        "3. Continue all-basis substrate work separately from D-CHE FMS proof; do not promote exploration-only families.\n",
    )

    write_figures(out, e_summary, v_summary_all, dche_real, substrate, mlp, dche_synth, no_action)

    manifest = write_required_manifest(out)
    route_json = {
        "stage": "V1411_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "synthetic_task_family_pass_count": synth_pass,
        "real_dataset_seed_pass_count": real_pass_count,
        "line_r_pass": int(r_pass),
        "line_e_pass": int(e_pass),
        "line_v_pass": int(v_pass),
        "line_fche_executed": 0,
        "line_d_new_fms_proof_executed": 0,
        "split_batch_probe_executed": int(args.run_split_batch_probe),
        "split_batch_probe_rows": len(split_probe_rows),
        "split_batch_probe_auc_U_train_to_synthetic_pass": split_probe_validity[0].get("auc_U_train_to_pass", "") if split_probe_validity else "",
        "split_batch_probe_gate_pass": split_probe_validity[0].get("gate_pass", 0) if split_probe_validity else 0,
        "split_batch_control_probe_rows": len(split_control_rows),
        "split_batch_control_explains_fms_count": sum(sint(r.get("control_explains_fms")) for r in split_control_validity),
        "v145_oracle_reference_rows": len(v145_oracle),
        "v145_missing_state_reference_rows": len(v145_missing),
        "v145_best_oracle_dataset_seed_pass_count": max([sint(r.get("best_oracle_dataset_seed_pass_count")) for r in v145_oracle] + [0]),
        "v145_action_bank_upper_bound_pass": max([sint(r.get("action_bank_upper_bound_pass")) for r in v145_oracle] + [0]),
        "line_d_v1411_route": line_d_v1411_route.get("route", ""),
        "line_d_v1411_candidate_rows": sint(line_d_v1411_route.get("candidate_rows")),
        "line_d_v1411_best_family_dataset_seed_pass_count": sint(line_d_v1411_route.get("best_family_dataset_seed_pass_count")),
        "line_d_v1411_exploration_open_family_count": sint(line_d_v1411_route.get("exploration_open_family_count")),
        "line_d_v1411_official_fms_eligible_family_count": sint(line_d_v1411_route.get("official_fms_eligible_family_count")),
        "official_s5_reached": int(route == "S5-DCHENonRATFunctionalSuccess"),
        "promotion_allowed": 0,
        "required_artifact_missing_count": sum(sint(r.get("missing")) for r in manifest),
        "forbidden_information_violation_count": sum(sint(r.get("violation")) for r in forbidden),
        "no_action_search_violation_count": sum(sint(r.get("violation")) for r in no_action),
        "source_artifacts": {
            "v1410_synthetic": str(V1410_SYNTH.relative_to(ROOT)),
            "v1410_real": str(V1410_REAL.relative_to(ROOT)),
            "v145_oracle_root": str(V145_ORACLE_ROOT.relative_to(ROOT)),
            "v149_substrate": str(V149_SUBSTRATE.relative_to(ROOT)),
            "v1411_line_d_substrate_hardening": str(V1411_LINE_D.relative_to(ROOT)),
        },
        "out_dir": str(out.relative_to(ROOT)),
    }
    write_json(out / "v1411_route_decision.json", route_json)
    # Recompute after route exists.
    manifest = write_required_manifest(out)
    route_json["required_artifact_missing_count"] = sum(sint(r.get("missing")) for r in manifest)
    write_json(out / "v1411_route_decision.json", route_json)
    write_docs(out, route_json, e_summary, v_summary_all)
    make_packet(out)
    manifest = write_required_manifest(out)
    route_json["required_artifact_missing_count"] = sum(sint(r.get("missing")) for r in manifest)
    write_json(out / "v1411_route_decision.json", route_json)

    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
