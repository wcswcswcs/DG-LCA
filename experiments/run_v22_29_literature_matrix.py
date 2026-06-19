#!/usr/bin/env python3
"""DG-KAN v22.29 literature-grounded geometric mechanism matrix.

This runner is intentionally conservative.  It performs Gate 1 historical
reanalysis first, writes every required artifact, and only opens later gates
when a hard-gate eligible direct metric family passes.  Proxy metrics are kept
as diagnostic rows and are never promoted.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_ROOT = ROOT / "results/v22_29"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.29_LiteratureGroundedGeometricMechanismMatrix_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.29_LiteratureGroundedGeometricMechanismMatrix_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.29_LiteratureGroundedGeometricMechanismMatrix_实验结果复盘.md"
PYTHON = os.environ.get("KAN_PYTHON", sys.executable)

THEORY_MODULES = [
    ("T1", "Signal-Channel FU"),
    ("T2", "Grassmann Signal-Subspace Diagnostics"),
    ("T3", "Grassmann-Transported Source Momentum"),
    ("T4", "Tangent-Normalized Signal FU"),
    ("T5", "Online Gradient / Effect Subspace Tracking FU"),
    ("T6", "Temporal-Spectral / Grokking FU"),
    ("T7", "Optimizer-Interaction FU"),
]

STRONG_OPTIMIZERS = [
    "SGD/Momentum",
    "AdamW",
    "Schedule-Free AdamW",
    "Cautious AdamW",
    "Cautious Lion",
    "Muon-like",
    "SOAP/Shampoo-like",
    "Mano-like",
    "PolarGrad-like diagnostic",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.29 Literature-Grounded Geometric Mechanism Matrix 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行命令、文件、状态、blocker、修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.29 Literature-Grounded Geometric Mechanism Matrix 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用本轮落盘 artifact 与真实读回结果；禁止编造数据。\n",
            encoding="utf-8",
        )


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: int | str = "",
) -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    journal = OUT_ROOT / "v22_29_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write(f"```bash\n{command}\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n")
        f.write(f"- status: {status}\n")
        f.write(f"- exit_code: {exit_code if exit_code != '' else 'n/a'}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    try:
        with p.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def first_float(row: dict[str, Any], keys: Iterable[str]) -> tuple[float | None, str]:
    for key in keys:
        if key in row:
            value = finite_float(row.get(key))
            if value is not None:
                return value, key
    return None, ""


def int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(rows: list[dict[str, Any]], fields: list[str] | None = None, limit: int | None = None) -> str:
    if not rows:
        return "\n_无落盘 rows。_\n"
    fields = fields or list(rows[0].keys())
    shown = rows[: limit or len(rows)]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if limit and len(rows) > limit:
        out.append(f"\n_仅显示前 {limit} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out) + "\n"


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx = rankdata(xs)
    ry = rankdata(ys)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def auc_score(scores: list[float], labels: list[int]) -> float | None:
    if len(scores) != len(labels) or len(scores) < 2:
        return None
    pos = sum(1 for x in labels if x == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(scores)
    pos_rank_sum = sum(r for r, label in zip(ranks, labels) if label == 1)
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def normalize_dataset(raw: str) -> str:
    if raw == "FashionMNIST":
        return "FMNIST"
    return raw


def family_group(row: dict[str, str]) -> str:
    text = " ".join(str(row.get(k, "")) for k in ("model_family", "model_name", "variant", "mode", "controller_kind"))
    upper = text.upper()
    if "D-FOU" in upper or "DFOU" in upper:
        return "KAN_DFOU"
    if "D-CHE" in upper or "DCHE" in upper:
        return "KAN_DCHE"
    if "KAN" in upper:
        return "KAN"
    if "MLP" in upper or "DGMLP" in upper:
        return "MLP"
    return row.get("model_family") or "unknown"


def model_name(row: dict[str, str]) -> str:
    for key in ("model_name", "variant", "mode", "controller_kind", "risk_model"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "unknown"


def optimizer_name(row: dict[str, str]) -> str:
    text = " ".join(str(row.get(k, "")) for k in ("optimizer", "training", "model_name", "variant", "mode"))
    low = text.lower()
    if "schedule" in low:
        return "Schedule-Free AdamW"
    if "cautious lion" in low:
        return "Cautious Lion"
    if "cautious" in low:
        return "Cautious AdamW"
    if "muon" in low:
        return "Muon-like"
    if "soap" in low or "shampoo" in low:
        return "SOAP/Shampoo-like"
    if "mano" in low:
        return "Mano-like"
    if "polar" in low:
        return "PolarGrad-like diagnostic"
    if "sgd" in low or "momentum" in low:
        return "SGD/Momentum"
    if "adamw" in low or "adam" in low:
        return "AdamW"
    return row.get("optimizer") or row.get("training") or ""


def is_base_row(row: dict[str, str]) -> bool:
    name = model_name(row).lower()
    policy = str(row.get("controller_policy", "")).lower()
    control_mode = str(row.get("control_mode", "")).lower()
    training = str(row.get("training", "")).lower()
    return (
        "baseline" in name
        or name.endswith("+adamw")
        or "adamw baseline" in name
        or policy == "adamw_baseline"
        or (training == "adamw" and control_mode in {"", "none"})
        or name in {"mlp_adamw", "kan_adamw_dche", "kan_adamw_dfou", "dgmlp"}
    )


def is_control_row(row: dict[str, str]) -> bool:
    name = model_name(row).lower()
    control_mode = str(row.get("control_mode", "")).lower()
    if int_flag(row.get("is_control_variant")):
        return True
    return (
        "control" in name
        or "noop" in name
        or control_mode not in {"", "none", "real"}
        or "random" in name
        or "signflip" in name
        or "shuffled" in name
    )


def task_tier(dataset: str, source_file: str, row: dict[str, str]) -> str:
    ds = dataset.lower()
    src = source_file.lower()
    if ds.startswith("class_"):
        return "continual"
    if ds in {"mnist", "fmnist", "kmnist", "fashionmnist"}:
        if finite_float(row.get("label_noise"), 0.0) and finite_float(row.get("label_noise"), 0.0) > 0:
            return "hard_slice"
        if "hard_lowdata" in src:
            return "hard_slice"
        return "mnist_like"
    if ds in {"cifar10", "svhn", "emnist-letters", "emnist_letters"}:
        return "hard_vision"
    if ds in {"spam", "wine", "rice", "bean", "telescope", "bank", "income"}:
        return "tabular"
    if "grok" in src or "modular" in src:
        return "grokking"
    return "external_or_unknown"


def intervention_strength(row: dict[str, str]) -> str:
    name = model_name(row).lower()
    if "noop" in name or is_base_row(row):
        return "no-op"
    ratio = first_float(row, ("controller_to_base_update_ratio", "controller_to_base_update_ratio_mean"))[0]
    if ratio is not None and ratio > 1.0:
        return "overguidance"
    accept = first_float(row, ("MMFP_accept_rate", "policy_action_accept_rate", "gate_accept_count"))[0]
    if accept is not None:
        if "gate_accept_count" in row:
            total = first_float(row, ("gate_reject_count",))[0]
            if total is not None and accept + total > 0:
                accept = accept / (accept + total)
        if accept >= 0.50:
            return "accepted-heavy"
        if accept > 0.0:
            return "accepted-sparse"
    return "nontrivial"


def discover_historical_files(historical_root: Path) -> list[Path]:
    explicit: list[Path] = [
        historical_root / "v22_15_adaptive_functional_guidance_source_manifold/official_v22_15/v22_15_task_eval_matrix.csv",
        historical_root / "v22_15_adaptive_functional_guidance_source_manifold/official_v22_15/v22_15_convergence_speed_matrix.csv",
        historical_root / "v22_16_real_trajectory_source_manifold_adaptive_fu/official_v22_16/v22_16_real_mlp_adaptive_guidance_matrix.csv",
        historical_root / "v22_16_real_trajectory_source_manifold_adaptive_fu/official_v22_16/v22_16_real_source_manifold_controller_matrix.csv",
        historical_root / "v22_16_real_trajectory_source_manifold_adaptive_fu/official_v22_16/v22_16_real_KAN_basis_controller_matrix.csv",
        historical_root / "v22_17/v22_17_risk_prediction_matrix.csv",
    ]
    globbed: list[Path] = []
    for pattern in [
        "v22_18/v22_18_*task_matrix*.csv",
        "v22_18/v22_18_kan_native_ladder_matrix*.csv",
        "v22_18/v22_18_class_mnist_continual_matrix*.csv",
        "v22_20/v22_20_*task_matrix*.csv",
        "v22_20/v22_20_*continual*.csv",
        "v22_20/v22_20_*gap*.csv",
    ]:
        globbed.extend(historical_root.glob(pattern))
    files = [p for p in explicit + globbed if p.exists() and p.is_file() and p.stat().st_size > 0]
    return sorted(set(files))


def collect_task_rows(historical_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = discover_historical_files(historical_root)
    inventory: list[dict[str, Any]] = []
    raw_entries: list[dict[str, Any]] = []
    base_by_group: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for path in files:
        rows = read_rows(path)
        inventory.append(
            {
                "source_file": str(path.relative_to(ROOT)),
                "rows": len(rows),
                "nonempty": int(bool(rows)),
                "columns": ";".join(rows[0].keys()) if rows else "",
            }
        )
        for idx, row in enumerate(rows):
            ds = normalize_dataset(str(row.get("dataset", "")).strip())
            seed = str(row.get("seed", "")).strip()
            fam = family_group(row)
            nll, nll_col = first_float(
                row,
                (
                    "final_test_NLL",
                    "final_test_loss_NLL",
                    "final_test_loss_readback",
                    "test_loss_readback",
                    "final_test_loss",
                    "train_loss",
                ),
            )
            auc, auc_col = first_float(row, ("AUC_loss_time", "AUC_loss_step", "wallclock_adjusted_AUC"))
            final_acc, acc_col = first_float(
                row,
                ("final_test_accuracy", "final_test_accuracy_readback", "test_accuracy_readback", "final_average_accuracy"),
            )
            entry = {
                "source_file": str(path.relative_to(ROOT)),
                "source_row_index": idx,
                "dataset": ds,
                "seed": seed,
                "model_name": model_name(row),
                "model_family": fam,
                "optimizer": optimizer_name(row),
                "is_base": int(is_base_row(row)),
                "is_control": int(is_control_row(row)),
                "task_tier": task_tier(ds, str(path), row),
                "intervention_strength": intervention_strength(row),
                "final_NLL": nll,
                "final_NLL_col": nll_col,
                "AUC_loss_time": auc,
                "AUC_col": auc_col,
                "final_accuracy": final_acc,
                "accuracy_col": acc_col,
                "_raw": row,
            }
            raw_entries.append(entry)
            if entry["is_base"] and ds and seed and nll is not None:
                key = (entry["source_file"], ds, seed, fam)
                base_by_group.setdefault(key, {})
                base_by_group[key]["nll"] = nll
                if auc is not None:
                    base_by_group[key]["auc"] = auc
    out: list[dict[str, Any]] = []
    for entry in raw_entries:
        row = dict(entry["_raw"])
        nll_delta, nll_delta_col = first_float(
            row,
            (
                "NLL_delta_vs_MLP_AdamW",
                "NLL_delta_vs_MLP",
                "NLL_delta_vs_base",
                "KAN_specific_delta_vs_MLP_same_operator",
                "KAN_vs_MLPFU_NLL_delta",
                "GapReduction",
                "relative_forgetting_reduction",
            ),
        )
        nll_reference = nll_delta_col
        if nll_delta is None and entry["final_NLL"] is not None:
            key = (entry["source_file"], entry["dataset"], entry["seed"], entry["model_family"])
            base = base_by_group.get(key, {})
            if "nll" in base:
                nll_delta = float(entry["final_NLL"]) - base["nll"]
                nll_reference = "same_file_dataset_seed_family_base"
        auc_delta, auc_delta_col = first_float(row, ("AUC_loss_time_delta_vs_MLP_AdamW", "AUC_delta_vs_base"))
        if auc_delta is None and entry["AUC_loss_time"] is not None:
            key = (entry["source_file"], entry["dataset"], entry["seed"], entry["model_family"])
            base = base_by_group.get(key, {})
            if "auc" in base:
                auc_delta = float(entry["AUC_loss_time"]) - base["auc"]
                auc_delta_col = "same_file_dataset_seed_family_base"
        result = {k: v for k, v in entry.items() if k != "_raw"}
        result.update(
            {
                "NLL_delta": nll_delta,
                "NLL_delta_reference": nll_reference,
                "AUC_delta": auc_delta,
                "AUC_delta_reference": auc_delta_col,
                "NLL_improved": int(nll_delta is not None and nll_delta < -1.0e-9),
                "AUC_improved": int(auc_delta is not None and auc_delta < -1.0e-9),
                "has_task_outcome": int(nll_delta is not None or auc_delta is not None),
            }
        )
        for key, value in row.items():
            if key not in result:
                result[key] = value
        out.append(result)
    return out, inventory


METRIC_SPECS = [
    ("T1", "signal_channel_proxy", "MMFP_accept_rate", 0, "MMFP accept rate is current-loop proxy, not cohort drift-diffusion A_B."),
    ("T1", "signal_channel_proxy", "Q_improvement_mean", 0, "Current-batch quadratic metric proxy; no per-cohort diffusion term."),
    ("T1", "signal_channel_proxy", "control_dominance_margin", 0, "Matched-control margin proxy, not population signal eigenvalue."),
    ("T1", "signal_channel_proxy", "split_B_delta_mean", 0, "Split-B safety proxy from v22.20."),
    ("T1", "signal_channel_proxy", "hard_slice_delta", 0, "Hard-slice local debt proxy from v22.20."),
    ("T2", "grassmann_proxy", "basis_channel_energy_fraction", 0, "Basis energy proxy, not principal-angle subspace retention."),
    ("T2", "grassmann_proxy", "readout_leakage_fraction", 0, "Readout leakage proxy, not Grassmann flow metric."),
    ("T2", "source_vector_proxy", "source_retention", 0, "Scalar retention proxy from trajectory log when present."),
    ("T4", "update_geometry_proxy", "controller_to_base_update_ratio", 0, "Norm ratio proxy, not singular update spectrum."),
    ("T4", "update_geometry_proxy", "controller_to_base_update_ratio_mean", 0, "Mean norm ratio proxy, not singular update spectrum."),
    ("T4", "update_geometry_proxy", "controller_overhead_ratio", 0, "Runtime overhead proxy, not update geometry."),
    ("T4", "update_geometry_proxy", "full_step_ratio", 0, "Full-step timing proxy, not tangent-normalized spectrum."),
    ("T5", "online_subspace_proxy", "projection_residual_Gf", 0, "Offline projection residual proxy, not online tracking error."),
    ("T5", "online_subspace_proxy", "basis_projection_residual", 0, "Projection residual proxy, not online Oja/QR tracking."),
    ("T5", "online_subspace_proxy", "explained_variance", 0, "Offline variance proxy, not online tracking."),
    ("T5", "online_subspace_proxy", "manifold_projection_residual_Gf", 0, "Offline source-manifold residual proxy."),
    ("T5", "online_subspace_proxy", "manifold_coverage_ratio", 0, "Coverage proxy, not online tracking."),
    ("T6", "temporal_proxy", "source_loss_h100", 0, "Source-loss horizon proxy, not grokking delay."),
    ("T6", "temporal_proxy", "source_loss_h400", 0, "Source-loss horizon proxy, not grokking delay."),
    ("T6", "temporal_proxy", "source_loss_h800", 0, "Source-loss horizon proxy, not grokking delay."),
    ("T6", "temporal_proxy", "source_loss_h1600", 0, "Source-loss horizon proxy, not grokking delay."),
    ("T6", "temporal_proxy", "source_loss_h3200", 0, "Source-loss horizon proxy, not grokking delay."),
    ("T6", "temporal_proxy", "source_loss_h4800", 0, "Source-loss horizon proxy, not grokking delay."),
    ("T6", "temporal_proxy", "source_func_task_h4800", 0, "Task source function proxy, not reservoir migration time."),
    ("T6", "temporal_proxy", "source_decay_rate_task", 0, "Decay-rate proxy from v22.15."),
]


def build_metric_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    for row in task_rows:
        for module_id, family, column, direct_flag, proxy_reason in METRIC_SPECS:
            if column not in row:
                continue
            value = finite_float(row.get(column))
            if value is None:
                continue
            metric_rows.append(
                {
                    "module_id": module_id,
                    "metric_family": family,
                    "metric_name": column,
                    "metric_value": value,
                    "direct_v22_29_metric": direct_flag,
                    "hard_gate_eligible": direct_flag and row.get("has_task_outcome") == 1,
                    "proxy_reason": proxy_reason,
                    "source_file": row.get("source_file"),
                    "source_row_index": row.get("source_row_index"),
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "model_name": row.get("model_name"),
                    "model_family": row.get("model_family"),
                    "task_tier": row.get("task_tier"),
                    "intervention_strength": row.get("intervention_strength"),
                    "is_control": row.get("is_control"),
                    "NLL_delta": row.get("NLL_delta"),
                    "AUC_delta": row.get("AUC_delta"),
                    "NLL_improved": row.get("NLL_improved"),
                    "AUC_improved": row.get("AUC_improved"),
                    "has_task_outcome": row.get("has_task_outcome"),
                }
            )
    return metric_rows


def evaluate_metric_group(rows: list[dict[str, Any]], *, stratum_kind: str, stratum_value: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if finite_float(row.get("metric_value")) is None or finite_float(row.get("NLL_delta")) is None:
            continue
        key = (
            str(row.get("module_id")),
            str(row.get("metric_family")),
            str(row.get("metric_name")),
            int(row.get("direct_v22_29_metric") or 0),
        )
        grouped.setdefault(key, []).append(row)
    eval_rows: list[dict[str, Any]] = []
    for (module_id, family, metric, direct), members in sorted(grouped.items()):
        xs = [float(m["metric_value"]) for m in members]
        nll = [float(m["NLL_delta"]) for m in members]
        benefit = [1 if v < -1.0e-9 else 0 for v in nll]
        auc = auc_score(xs, benefit)
        auc_oriented = None if auc is None else max(auc, 1.0 - auc)
        auc_orientation = "" if auc is None else ("higher_better" if auc >= 0.5 else "lower_better")
        rho = spearman(xs, [-v for v in nll])
        rho_abs = None if rho is None else abs(rho)
        controls = [m for m in members if int(m.get("is_control") or 0) == 1]
        real = [m for m in members if int(m.get("is_control") or 0) == 0]
        control_auc_oriented = None
        if controls:
            c_xs = [float(m["metric_value"]) for m in controls]
            c_b = [1 if float(m["NLL_delta"]) < -1.0e-9 else 0 for m in controls]
            c_auc = auc_score(c_xs, c_b)
            control_auc_oriented = None if c_auc is None else max(c_auc, 1.0 - c_auc)
        control_lower = (
            int(control_auc_oriented is None or (auc_oriented is not None and control_auc_oriented <= auc_oriented - 0.05))
            if auc_oriented is not None
            else 0
        )
        hard_pass = int(
            direct == 1
            and len(members) >= 10
            and auc_oriented is not None
            and auc_oriented >= 0.70
            and rho_abs is not None
            and rho_abs >= 0.30
            and control_lower == 1
        )
        proxy_pass = int(
            direct == 0
            and len(members) >= 10
            and auc_oriented is not None
            and auc_oriented >= 0.70
            and rho_abs is not None
            and rho_abs >= 0.30
        )
        eval_rows.append(
            {
                "stratum_kind": stratum_kind,
                "stratum_value": stratum_value,
                "module_id": module_id,
                "metric_family": family,
                "metric_name": metric,
                "direct_v22_29_metric": direct,
                "rows": len(members),
                "real_rows": len(real),
                "control_rows": len(controls),
                "positive_NLL_improvement_rows": sum(benefit),
                "AUC_oriented": "" if auc_oriented is None else auc_oriented,
                "AUC_orientation": auc_orientation,
                "Spearman_abs": "" if rho_abs is None else rho_abs,
                "Spearman_signed": "" if rho is None else rho,
                "control_AUC_oriented": "" if control_auc_oriented is None else control_auc_oriented,
                "control_predictive_power_lower": control_lower,
                "hard_gate_pass": hard_pass,
                "proxy_diagnostic_pass": proxy_pass,
                "hard_gate_blocker": ""
                if hard_pass
                else (
                    "proxy_only_not_promotable"
                    if direct == 0
                    else "insufficient_direct_outcome_link_or_predictive_power"
                ),
            }
        )
    return eval_rows


def evaluate_all(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = evaluate_metric_group(metric_rows, stratum_kind="all", stratum_value="all")
    for kind in ("model_family", "task_tier", "intervention_strength"):
        values = sorted({str(r.get(kind, "")) for r in metric_rows if str(r.get(kind, ""))})
        for value in values:
            subset = [r for r in metric_rows if str(r.get(kind, "")) == value]
            out.extend(evaluate_metric_group(subset, stratum_kind=kind, stratum_value=value))
    return out


def compute_grassmann_metrics(payload_paths: list[Path]) -> list[dict[str, Any]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        return [{"status": "blocked", "reason": f"torch_import_failed:{exc}"}]
    rows: list[dict[str, Any]] = []
    for path in payload_paths:
        if not path.exists():
            rows.append({"source_file": str(path.relative_to(ROOT)), "status": "missing_payload"})
            continue
        obj = torch.load(path, map_location="cpu")
        vectors = obj.get("source_vectors") if isinstance(obj, dict) else None
        metadata = obj.get("metadata") if isinstance(obj, dict) else None
        if vectors is None or metadata is None:
            rows.append({"source_file": str(path.relative_to(ROOT)), "status": "missing_source_vectors_or_metadata"})
            continue
        groups: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        for idx, meta in enumerate(metadata):
            key = (
                normalize_dataset(str(meta.get("dataset", ""))),
                str(meta.get("seed", "")),
                str(meta.get("model_variant", "")),
            )
            groups.setdefault(key, []).append((int(meta.get("step", idx)), idx))
        for (dataset, seed, variant), items in sorted(groups.items()):
            items = sorted(items)
            if len(items) < 4:
                rows.append(
                    {
                        "source_file": str(path.relative_to(ROOT)),
                        "dataset": dataset,
                        "seed": seed,
                        "model_variant": variant,
                        "status": "insufficient_steps",
                        "steps": len(items),
                    }
                )
                continue
            vec = torch.stack([vectors[idx].float() for _step, idx in items])
            norms = vec.norm(dim=1).clamp_min(1.0e-12)
            unit = vec / norms[:, None]
            vector_ret = (unit[:-1] * unit[1:]).sum(dim=1).clamp(-1.0, 1.0)
            k = min(4, vec.shape[0] // 2, vec.shape[1])
            angles_mean: list[float] = []
            angles_max: list[float] = []
            geodesic: list[float] = []
            projection_dist: list[float] = []
            subspace_ret: list[float] = []
            for start in range(0, vec.shape[0] - 2 * k + 1, k):
                a = vec[start : start + k].T
                b = vec[start + k : start + 2 * k].T
                qa = torch.linalg.qr(a, mode="reduced").Q
                qb = torch.linalg.qr(b, mode="reduced").Q
                s = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
                theta = torch.arccos(s)
                angles_mean.append(float(theta.mean()))
                angles_max.append(float(theta.max()))
                geodesic.append(float(torch.sqrt((theta * theta).sum())))
                projection_dist.append(float(torch.linalg.matrix_norm(qa @ qa.T - qb @ qb.T)))
                subspace_ret.append(float(torch.linalg.matrix_norm(qa.T @ qb) / math.sqrt(k)))
            rows.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "dataset": dataset,
                    "seed": seed,
                    "model_variant": variant,
                    "status": "computed",
                    "steps": len(items),
                    "subspace_k": k,
                    "source_vector_retention": float(vector_ret.mean()),
                    "source_vector_retention_min": float(vector_ret.min()),
                    "source_subspace_retention": statistics.fmean(subspace_ret) if subspace_ret else "",
                    "principal_angles_mean": statistics.fmean(angles_mean) if angles_mean else "",
                    "principal_angles_max": statistics.fmean(angles_max) if angles_max else "",
                    "Grassmann_geodesic_distance": statistics.fmean(geodesic) if geodesic else "",
                    "projection_Frobenius_distance": statistics.fmean(projection_dist) if projection_dist else "",
                    "Plucker_coordinate_variance": "not_reconstructed_high_dim_plucker_coordinates_not_logged",
                    "subspace_temporal_curvature": statistics.pstdev(geodesic) if len(geodesic) > 1 else 0.0,
                    "direct_v22_29_metric": 1,
                    "has_task_outcome_link": 0,
                    "hard_gate_eligible": 0,
                    "hard_gate_blocker": "source payload has vectors but no paired task outcome table for these exact variants",
                }
            )
    return rows


def split_metric_family_files(metric_rows: list[dict[str, Any]], grassmann_rows: list[dict[str, Any]]) -> None:
    signal_rows = [r for r in metric_rows if str(r.get("module_id")) == "T1"]
    tangent_rows = [r for r in metric_rows if str(r.get("module_id")) == "T4"]
    online_rows = [r for r in metric_rows if str(r.get("module_id")) == "T5"]
    write_rows(OUT_ROOT / "v22_29_signal_channel_metrics.csv", signal_rows)
    write_rows(OUT_ROOT / "v22_29_grassmann_subspace_metrics.csv", grassmann_rows)
    write_rows(OUT_ROOT / "v22_29_tangent_update_spectrum.csv", tangent_rows)
    write_rows(OUT_ROOT / "v22_29_online_subspace_tracking_matrix.csv", online_rows)


def copy_existing_matrix(output_name: str, source_patterns: list[str], historical_root: Path, fallback_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in source_patterns:
        for path in sorted(historical_root.glob(pattern)):
            for idx, row in enumerate(read_rows(path)):
                item = {"_source_file": str(path.relative_to(ROOT)), "_source_row_index": idx}
                item.update(row)
                rows.append(item)
    if not rows:
        rows = [fallback_row]
    write_rows(OUT_ROOT / output_name, rows)
    return rows


def build_strong_optimizer_baselines(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_opt: set[str] = set()
    for row in task_rows:
        if not row.get("is_base"):
            continue
        opt = str(row.get("optimizer") or "")
        if not opt:
            continue
        seen_opt.add(opt)
        rows.append(
            {
                "status": "available_historical_artifact",
                "optimizer": opt,
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "model_name": row.get("model_name"),
                "model_family": row.get("model_family"),
                "final_NLL": row.get("final_NLL"),
                "AUC_loss_time": row.get("AUC_loss_time"),
                "source_file": row.get("source_file"),
                "source_row_index": row.get("source_row_index"),
            }
        )
    for opt in STRONG_OPTIMIZERS:
        if opt not in seen_opt:
            rows.append(
                {
                    "status": "missing_in_historical_artifacts",
                    "optimizer": opt,
                    "blocker": "No v22.15-v22.20 artifact row found; not run by v22.29 because Gate1 did not open new full-loop optimizer matrix.",
                }
            )
    write_rows(OUT_ROOT / "v22_29_strong_optimizer_baselines.csv", rows)
    return rows


def skipped_rows(kind: str, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "stage": kind,
            "module_id": module_id,
            "module_name": name,
            "status": "skipped",
            "reason": reason,
            "latest_status_timestamp": now_sg(),
        }
        for module_id, name in THEORY_MODULES
    ]


def write_placeholder_matrices(gate_open: bool, reason: str) -> None:
    if gate_open:
        return
    write_rows(OUT_ROOT / "v22_29_branch_causal_pilot_matrix.csv", skipped_rows("branch_causal_pilot", reason))
    write_rows(OUT_ROOT / "v22_29_branch_causal_controls_matrix.csv", skipped_rows("branch_causal_controls", reason))
    write_rows(
        OUT_ROOT / "v22_29_mlp_full_loop_matrix.csv",
        [{"stage": "MLP_full_loop", "status": "skipped", "reason": reason, "latest_status_timestamp": now_sg()}],
    )
    write_rows(
        OUT_ROOT / "v22_29_kan_full_loop_matrix.csv",
        [{"stage": "KAN_full_loop", "status": "skipped", "reason": reason, "latest_status_timestamp": now_sg()}],
    )
    write_rows(
        OUT_ROOT / "v22_29_grokking_matrix.csv",
        [{"stage": "grokking", "status": "skipped", "reason": reason, "latest_status_timestamp": now_sg()}],
    )


def write_literature_matrix_md(module_status: list[dict[str, Any]]) -> None:
    lines = [
        "# v22.29 Literature-Grounded Geometric Mechanism Matrix",
        "",
        f"更新时间：{now_sg()}",
        "",
        "This file is generated from the v22.29 runner.  It records which theory modules are allowed to proceed after Gate 1.",
        "",
        md_table(module_status, ["module_id", "module_name", "gate1_status", "hard_gate_pass", "proxy_diagnostic_pass", "blocker"]),
    ]
    (OUT_ROOT / "v22_29_literature_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def artifact_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "nonempty": int(path.stat().st_size > 0),
                    "latest_status_timestamp": now_sg(),
                }
            )
    return rows


def write_source_packet_manifest() -> None:
    paths = [
        Path("experiments/run_v22_29_literature_matrix.py"),
        Path("docs/DG-KAN_v22.29_LiteratureGroundedGeometricMechanismMatrix_完整计划.md"),
        Path("dgkan/fu/mmfp_update.py"),
        Path("dgkan/models/fc_purekan_primitives.py"),
    ]
    rows = []
    for rel in paths:
        path = ROOT / rel
        rows.append(
            {
                "source_file": str(rel),
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    write_rows(OUT_ROOT / "v22_29_source_packet_manifest.csv", rows)


def write_simple_svg(name: str, title: str, rows: list[dict[str, Any]], value_key: str = "rows") -> None:
    values: list[tuple[str, float]] = []
    for row in rows[:12]:
        value = finite_float(row.get(value_key))
        if value is None:
            continue
        label = str(row.get("module_id") or row.get("metric_name") or row.get("dataset") or row.get("status") or "")[:18]
        values.append((label, value))
    width, height = 900, 360
    if not values:
        body = (
            f'<text x="32" y="64" font-size="24" fill="#222">{title}</text>'
            '<text x="32" y="130" font-size="18" fill="#555">No numeric data available or stage skipped by Gate 1.</text>'
        )
    else:
        maxv = max(abs(v) for _l, v in values) or 1.0
        bars = []
        for i, (label, value) in enumerate(values):
            y = 80 + i * 22
            w = max(1.0, abs(value) / maxv * 560.0)
            color = "#2f6f8f" if value >= 0 else "#9b3d3d"
            bars.append(f'<text x="32" y="{y + 14}" font-size="12">{label}</text>')
            bars.append(f'<rect x="220" y="{y}" width="{w:.1f}" height="14" fill="{color}" />')
            bars.append(f'<text x="{230 + w:.1f}" y="{y + 12}" font-size="12">{value:.4g}</text>')
        body = f'<text x="32" y="42" font-size="24" fill="#222">{title}</text>' + "".join(bars)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/>{body}</svg>\n'
    (FIG_ROOT / name).write_text(svg, encoding="utf-8")


def install_kanbefair_import_stubs() -> None:
    """Keep unused KANbeFair optional imports from blocking vision/tabular pilots."""
    if "torchtext" not in sys.modules:
        torchtext_mod = types.ModuleType("torchtext")
        torchtext_data_mod = types.ModuleType("torchtext.data")
        torchtext_data_utils_mod = types.ModuleType("torchtext.data.utils")
        torchtext_vocab_mod = types.ModuleType("torchtext.vocab")

        def _basic_tokenizer(_name: str) -> Any:
            return lambda text: str(text).split()

        class _DummyVocab(dict):
            def __call__(self, tokens: Any) -> list[int]:
                return [self.get(tok, 0) for tok in tokens]

            def set_default_index(self, _idx: int) -> None:
                return None

        def _build_vocab_from_iterator(iterator: Any, specials: list[str] | None = None) -> _DummyVocab:
            vocab = _DummyVocab()
            for idx, token in enumerate(specials or []):
                vocab[token] = idx
            for tokens in iterator:
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab)
            return vocab

        torchtext_data_utils_mod.get_tokenizer = _basic_tokenizer
        torchtext_vocab_mod.build_vocab_from_iterator = _build_vocab_from_iterator
        torchtext_data_mod.utils = torchtext_data_utils_mod
        torchtext_mod.data = torchtext_data_mod
        torchtext_mod.vocab = torchtext_vocab_mod
        sys.modules["torchtext"] = torchtext_mod
        sys.modules["torchtext.data"] = torchtext_data_mod
        sys.modules["torchtext.data.utils"] = torchtext_data_utils_mod
        sys.modules["torchtext.vocab"] = torchtext_vocab_mod
    if "fvcore" not in sys.modules:
        fvcore_mod = types.ModuleType("fvcore")
        fvcore_nn_mod = types.ModuleType("fvcore.nn")

        class _MissingFlopCountAnalysis:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                return None

            def total(self) -> int:
                return 0

        def _missing_parameter_count(_model: Any) -> dict[str, int]:
            return {}

        fvcore_nn_mod.FlopCountAnalysis = _MissingFlopCountAnalysis
        fvcore_nn_mod.parameter_count = _missing_parameter_count
        fvcore_mod.nn = fvcore_nn_mod
        sys.modules["fvcore"] = fvcore_mod
        sys.modules["fvcore.nn"] = fvcore_nn_mod
    if "torchaudio" not in sys.modules:
        torchaudio_mod = types.ModuleType("torchaudio")
        torchaudio_mod.datasets = types.ModuleType("torchaudio.datasets")
        torchaudio_mod.transforms = types.ModuleType("torchaudio.transforms")

        class _MissingSpeechCommands:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("torchaudio is unavailable; v22.29 continuation pilot only uses non-audio tasks")

        class _IdentityResample:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                return None

            def __call__(self, value: Any) -> Any:
                return value

        def _missing_audio_load(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("torchaudio is unavailable; audio tasks are unsupported in this pilot")

        torchaudio_mod.datasets.SPEECHCOMMANDS = _MissingSpeechCommands
        torchaudio_mod.transforms.Resample = _IdentityResample
        torchaudio_mod.load = _missing_audio_load
        sys.modules["torchaudio"] = torchaudio_mod
        sys.modules["torchaudio.datasets"] = torchaudio_mod.datasets
        sys.modules["torchaudio.transforms"] = torchaudio_mod.transforms


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def pilot_device(device_name: str) -> Any:
    import torch

    requested = str(device_name)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested.startswith("cuda"):
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            return torch.device("cpu")
    return torch.device(requested)


def load_local_vision_loaders(data_root: Path, dataset: str, batch_size: int, train_size: int, test_size: int, seed: int) -> tuple[Any, Any, int, int] | None:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    mapping = {
        "MNIST": datasets.MNIST,
        "FMNIST": datasets.FashionMNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "KMNIST": datasets.KMNIST,
    }
    cls = mapping.get(dataset)
    if cls is None:
        return None
    transform = transforms.ToTensor()
    train_ds = cls(str(data_root), train=True, download=False, transform=transform)
    test_ds = cls(str(data_root), train=False, download=False, transform=transform)
    gen_train = torch.Generator().manual_seed(int(seed))
    gen_test = torch.Generator().manual_seed(int(seed) + 10000)
    train_idx = torch.randperm(len(train_ds), generator=gen_train)[: min(int(train_size), len(train_ds))].tolist()
    test_idx = torch.randperm(len(test_ds), generator=gen_test)[: min(int(test_size), len(test_ds))].tolist()
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, generator=gen_train, num_workers=0, drop_last=False)
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False, generator=gen_test, num_workers=0, drop_last=False)
    return train_loader, test_loader, 10, 28 * 28


def load_kanbefair_loaders(root: Path, data_root: Path, dataset: str, batch_size: int, train_size: int, test_size: int, seed: int) -> tuple[Any, Any, int, int]:
    local = load_local_vision_loaders(data_root, dataset, batch_size, train_size, test_size, seed)
    if local is not None:
        return local
    install_kanbefair_import_stubs()
    import torch
    from torch.utils.data import DataLoader, Subset

    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import utils  # type: ignore

    class LoaderArgs:
        def __init__(self, dataset_name: str, batch: int) -> None:
            self.dataset = dataset_name
            self.batch_size = batch
            self.test_batch_size = batch

    def _subset(loader: Any, size: int, train: bool) -> Any:
        ds = loader.dataset
        n = min(int(size), len(ds))
        gen = torch.Generator().manual_seed(int(seed) + (0 if train else 10000))
        idx = torch.randperm(len(ds), generator=gen)[:n].tolist()
        return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=train, generator=gen, num_workers=0, drop_last=False)

    old_cwd = Path.cwd()
    os.chdir(root / "src")
    try:
        train_loader, test_loader, num_classes, input_size = utils.get_loader(LoaderArgs(dataset, batch_size), use_cuda=False)
    finally:
        os.chdir(old_cwd)
    return _subset(train_loader, train_size, True), _subset(test_loader, test_size, False), int(num_classes), int(input_size)


def flatten_batch(xb: Any, input_dim: int, device: Any) -> Any:
    return xb.to(device).view(xb.shape[0], input_dim).float()


def flat_params(model: Any) -> Any:
    import torch

    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def add_flat_to_params(model: Any, update: Any) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.add_(update[offset : offset + n].view_as(p))
        offset += n


def flat_grads_from_loss(model: Any, loss: Any) -> Any:
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)
    return torch.cat([g.detach().reshape(-1) for g in grads])


def flat_current_grads(model: Any) -> Any:
    import torch

    chunks = []
    for p in model.parameters():
        if p.grad is None:
            chunks.append(torch.zeros_like(p.detach()).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def evaluate_pilot_model(model: Any, loader: Any, input_dim: int, device: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    brier_sum = 0.0
    confs: list[float] = []
    hits: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            x = flatten_batch(xb, input_dim, device)
            y = yb.to(device).long()
            logits = model(x)
            per = F.cross_entropy(logits, y, reduction="none")
            total_loss += float(per.sum().detach().cpu())
            total += int(y.numel())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().detach().cpu())
            probs = logits.softmax(dim=1)
            one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
            brier_sum += float(((probs - one_hot) ** 2).sum(dim=1).sum().detach().cpu())
            c, p = probs.max(dim=1)
            confs.extend([float(v) for v in c.detach().cpu()])
            hits.extend([float(v) for v in (p == y).float().detach().cpu()])
            losses.extend([float(v) for v in per.detach().cpu()])
    if total <= 0:
        return {"final_test_NLL": math.nan, "final_test_accuracy": math.nan, "ECE": math.nan, "Brier": math.nan, "tail_loss_q95": math.nan, "tail_loss_q99": math.nan}
    ece = 0.0
    for lo in [i / 10.0 for i in range(10)]:
        hi = lo + 0.1
        idx = [i for i, c in enumerate(confs) if (c >= lo and (c < hi or (hi >= 1.0 and c <= hi)))]
        if not idx:
            continue
        acc = statistics.fmean(hits[i] for i in idx)
        conf = statistics.fmean(confs[i] for i in idx)
        ece += len(idx) / total * abs(acc - conf)
    losses_sorted = sorted(losses)

    def q(prob: float) -> float:
        if not losses_sorted:
            return math.nan
        pos = min(len(losses_sorted) - 1, max(0, int(math.ceil(prob * len(losses_sorted))) - 1))
        return float(losses_sorted[pos])

    return {
        "final_test_NLL": total_loss / total,
        "final_test_accuracy": correct / total,
        "ECE": ece,
        "Brier": brier_sum / total,
        "tail_loss_q95": q(0.95),
        "tail_loss_q99": q(0.99),
    }


def compute_pilot_signal_metrics(model: Any, train_loader: Any, input_dim: int, device: Any, cohorts: int, seed: int) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    model.train()
    grads: list[Any] = []
    loader_iter = iter(train_loader)
    for _ in range(int(cohorts)):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            xb, yb = next(loader_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        loss = F.cross_entropy(model(x), y)
        grads.append(flat_grads_from_loss(model, loss))
    gmat = torch.stack(grads, dim=0)
    mean_g = gmat.mean(dim=0)
    drift_norm = float(mean_g.norm().detach().cpu())
    centered = gmat - mean_g
    diffusion_trace = float(centered.pow(2).sum(dim=1).mean().detach().cpu())
    grad_unit = mean_g / mean_g.norm().clamp_min(1.0e-12)
    update_unit = -grad_unit
    proj = gmat @ grad_unit
    signal_score = float((proj.mean() ** 2 - proj.var(unbiased=False)).detach().cpu())
    gen = torch.Generator(device=device).manual_seed(int(seed) + 2929)
    control_scores: list[float] = []
    for _ in range(16):
        r = torch.randn_like(mean_g, generator=gen)
        r = r / r.norm().clamp_min(1.0e-12)
        rp = gmat @ r
        control_scores.append(float((rp.mean() ** 2 - rp.var(unbiased=False)).detach().cpu()))
    k = min(3, max(1, int(gmat.shape[0] // 2)), int(gmat.shape[1]))
    a = gmat[:k].T
    b = gmat[-k:].T
    qa = torch.linalg.qr(a, mode="reduced").Q
    qb = torch.linalg.qr(b, mode="reduced").Q
    s = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    theta = torch.arccos(s)
    metrics = {
        "pilot_metric_scope": "cohort_mean_gradient_checkpoint_probe",
        "pilot_metric_limitation": "cohort-level gradients, not per-example A_B; used because full per-example dense gradients are too expensive for continuation audit",
        "cohorts": int(gmat.shape[0]),
        "drift_norm": drift_norm,
        "diffusion_trace": diffusion_trace,
        "SNR": drift_norm / math.sqrt(diffusion_trace + 1.0e-12),
        "signal_eigenvalue_margin": signal_score - max(control_scores),
        "signal_score": signal_score,
        "control_eigenvalue_max": max(control_scores),
        "cohort_positive_fraction": float((proj > 0).float().mean().detach().cpu()),
        "source_subspace_retention": float(torch.linalg.matrix_norm(qa.T @ qb).detach().cpu() / math.sqrt(k)),
        "principal_angles_mean": float(theta.mean().detach().cpu()),
        "principal_angles_max": float(theta.max().detach().cpu()),
        "Grassmann_geodesic_distance": float(torch.sqrt((theta * theta).sum()).detach().cpu()),
        "projection_Frobenius_distance": float(torch.linalg.matrix_norm(qa @ qa.T - qb @ qb.T).detach().cpu()),
        "subspace_k": k,
    }
    return update_unit.detach(), metrics


def gradient_subspace_snapshot(
    model: Any,
    train_loader: Any,
    input_dim: int,
    device: Any,
    cohorts: int,
    rank: int,
    seed: int,
    *,
    salt: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    model.train()
    grads: list[Any] = []
    loader_iter = iter(train_loader)
    for _ in range(max(1, int(cohorts))):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            xb, yb = next(loader_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        loss = F.cross_entropy(model(x), y)
        grads.append(flat_grads_from_loss(model, loss))
    gmat = torch.stack(grads, dim=1)
    p_dim = int(gmat.shape[0])
    k = min(max(1, int(rank)), int(gmat.shape[1]), p_dim)
    if k <= 0:
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(salt))
        basis = torch.linalg.qr(torch.randn((p_dim, 1), device=device, generator=gen), mode="reduced").Q[:, :1]
    else:
        basis = torch.linalg.qr(gmat, mode="reduced").Q[:, :k]
    mean_grad = torch.stack(grads, dim=0).mean(dim=0)
    update = -mean_grad / mean_grad.norm().clamp_min(1.0e-12)
    proj = basis @ (basis.T @ update)
    centered = torch.stack(grads, dim=0) - mean_grad
    metrics = {
        "subspace_rank": int(basis.shape[1]),
        "cohorts": int(len(grads)),
        "mean_grad_norm": float(mean_grad.norm().detach().cpu()),
        "mean_update_reconstruction_ratio": float((proj.norm() / update.norm().clamp_min(1.0e-12)).detach().cpu()),
        "gradient_variance": float(centered.pow(2).sum(dim=1).mean().detach().cpu()),
    }
    return basis.detach(), update.detach(), metrics


def compute_transported_source_direction(
    prev_basis: Any,
    curr_basis: Any,
    prev_update: Any,
    curr_update: Any,
    *,
    beta: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    k = min(int(prev_basis.shape[1]), int(curr_basis.shape[1]))
    prev = prev_basis[:, :k]
    curr = curr_basis[:, :k]
    cross = curr.T @ prev
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    prev_coeff = prev.T @ prev_update
    transported = curr @ (rotation @ prev_coeff)
    curr_proj = curr @ (curr.T @ curr_update)
    combined = float(beta) * transported + (1.0 - float(beta)) * curr_proj
    if float(combined.norm().detach().cpu()) <= 1.0e-12:
        combined = curr_proj if float(curr_proj.norm().detach().cpu()) > 1.0e-12 else curr_update
    direction = combined / combined.norm().clamp_min(1.0e-12)
    theta = torch.arccos(s.clamp(0.0, 1.0))
    before_err = torch.linalg.matrix_norm(curr - prev) / math.sqrt(max(1, k))
    after_err = torch.linalg.matrix_norm(curr @ rotation - prev) / math.sqrt(max(1, k))

    def cosine(a: Any, b: Any) -> float:
        denom = (a.norm() * b.norm()).clamp_min(1.0e-12)
        return float(((a @ b) / denom).detach().cpu())

    metrics = {
        "pilot_metric_scope": "grassmann_transported_gradient_momentum_branch_pilot",
        "pilot_metric_limitation": "T3 pilot transports a checkpoint-level cohort-gradient momentum from an earlier pretrain subspace to the branch checkpoint; it is not an online per-step parallel-transport optimizer.",
        "subspace_rank": int(k),
        "transport_beta": float(beta),
        "principal_angles_mean": float(theta.mean().detach().cpu()),
        "principal_angles_max": float(theta.max().detach().cpu()),
        "Grassmann_geodesic_distance": float(torch.sqrt((theta * theta).sum()).detach().cpu()),
        "transport_error_before_procrustes": float(before_err.detach().cpu()),
        "transport_error_after_procrustes": float(after_err.detach().cpu()),
        "transported_momentum_norm": float(transported.norm().detach().cpu()),
        "current_projected_momentum_norm": float(curr_proj.norm().detach().cpu()),
        "transport_alignment_with_current_signal": cosine(direction, curr_update),
        "transported_alignment_with_current_signal": cosine(transported, curr_update) if float(transported.norm().detach().cpu()) > 0.0 else 0.0,
    }
    return direction.detach(), metrics


def transported_control_direction(curr_basis: Any, real_direction: Any, branch: str, seed: int) -> Any:
    import torch

    if branch == "B1_REAL_TRANSPORTED_SOURCE":
        return real_direction
    if branch == "B4_SIGNFLIP_CONTROL":
        return -real_direction
    coeff = curr_basis.T @ real_direction
    if branch == "B5_SHUFFLED_CONTROL":
        gen = torch.Generator(device=real_direction.device).manual_seed(int(seed) + 6363)
        perm = torch.randperm(coeff.numel(), generator=gen, device=real_direction.device)
        shuffled = curr_basis @ coeff[perm]
        return shuffled / shuffled.norm().clamp_min(1.0e-12)
    if branch in {"B2_RANDOM_CONTROL", "B3_STABLE_RANDOM_CONTROL"}:
        salt = 7373 if branch == "B2_RANDOM_CONTROL" else 8383
        gen = torch.Generator(device=real_direction.device).manual_seed(int(seed) + salt)
        rand_coeff = torch.randn((int(curr_basis.shape[1]),), device=real_direction.device, generator=gen)
        rand = curr_basis @ rand_coeff
        return rand / rand.norm().clamp_min(1.0e-12)
    return real_direction.new_zeros(real_direction.shape)


def last_parameter_full_direction(model: Any, last_direction: Any, preferred_name: str = "w2") -> Any:
    import torch

    chunks = []
    used = False
    named = list(model.named_parameters())
    names = {name for name, _param in named}
    fallback_name = named[-1][0] if named else ""
    for name, param in named:
        if name == preferred_name or (preferred_name not in names and name == fallback_name):
            chunks.append(last_direction.reshape_as(param).reshape(-1))
            used = True
        else:
            chunks.append(torch.zeros_like(param.detach()).reshape(-1))
    if not used:
        raise RuntimeError(f"Could not place last-layer direction for parameter {preferred_name}")
    full = torch.cat(chunks)
    return full / full.norm().clamp_min(1.0e-12)


def compute_effect_space_signal_metrics(model: Any, train_loader: Any, input_dim: int, device: Any, cohorts: int, seed: int) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    model.eval()
    cohort_As: list[Any] = []
    cohort_mus: list[Any] = []
    diffusion_traces: list[Any] = []
    loader_iter = iter(train_loader)
    for _ in range(int(cohorts)):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            xb, yb = next(loader_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        with torch.no_grad():
            h = model.frozen_readout_features(x)
            logits = h @ model.w2
            probs = logits.softmax(dim=1)
            err = probs - F.one_hot(y, num_classes=logits.shape[1]).float()
            per_example = torch.einsum("bi,bj->bij", h, err).reshape(x.shape[0], -1)
            mu = per_example.mean(dim=0)
            centered = per_example - mu
            denom = max(1, int(per_example.shape[0]) - 1)
            cov = centered.T @ centered / float(denom)
            diffusion = cov / float(denom)
            A = torch.outer(mu, mu) - diffusion
            cohort_As.append(A)
            cohort_mus.append(mu)
            diffusion_traces.append(torch.trace(diffusion))
    A_mean = torch.stack(cohort_As, dim=0).mean(dim=0)
    mu_mean = torch.stack(cohort_mus, dim=0).mean(dim=0)
    evals, evecs = torch.linalg.eigh(A_mean)
    top_idx = int(torch.argmax(evals).detach().cpu())
    grad_dir = evecs[:, top_idx]
    if float((grad_dir @ mu_mean).detach().cpu()) < 0.0:
        grad_dir = -grad_dir
    update_last = -grad_dir / grad_dir.norm().clamp_min(1.0e-12)
    update_full = last_parameter_full_direction(model, update_last, "w2")
    gen = torch.Generator(device=device).manual_seed(int(seed) + 92929)
    control_scores: list[float] = []
    for _ in range(32):
        r = torch.randn(grad_dir.shape, device=device, generator=gen)
        r = r / r.norm().clamp_min(1.0e-12)
        control_scores.append(float((r @ A_mean @ r).detach().cpu()))
    cohort_proj = torch.stack([m @ grad_dir for m in cohort_mus])
    drift_norm = float(mu_mean.norm().detach().cpu())
    diffusion_trace = float(torch.stack(diffusion_traces).mean().detach().cpu())
    metrics = {
        "pilot_metric_scope": "last_layer_per_example_effect_space_A_B",
        "pilot_metric_limitation": "official T1 fallback: last-layer/logits effect-space A_B rather than dense all-parameter per-example gradients; diffusion term uses sample covariance divided again by b-1 as specified by the v22.29 A_B pilot formula",
        "cohorts": len(cohort_As),
        "signal_eig_topk": float(evals[top_idx].detach().cpu()),
        "signal_eig_margin_vs_controls": float(evals[top_idx].detach().cpu()) - max(control_scores),
        "control_signal_eig_topk": max(control_scores),
        "cohort_positive_fraction": float((cohort_proj > 0).float().mean().detach().cpu()),
        "drift_norm": drift_norm,
        "diffusion_trace": diffusion_trace,
        "SNR": drift_norm / math.sqrt(diffusion_trace + 1.0e-12),
        "last_layer_dim": int(update_last.numel()),
    }
    return update_full.detach(), metrics


def control_direction(update_unit: Any, branch: str, seed: int, preserve_support: bool = False) -> Any:
    import torch

    if branch in {"B1_REAL_SIGNAL_GRASSMANN", "B1_REAL_EFFECT_AB", "B1_REAL_TANGENT_SIGNAL"}:
        return update_unit
    if branch == "B4_SIGNFLIP_CONTROL":
        return -update_unit
    support = update_unit.abs() > 0.0 if preserve_support else torch.ones_like(update_unit, dtype=torch.bool)
    if not bool(support.any().detach().cpu()):
        support = torch.ones_like(update_unit, dtype=torch.bool)
    if branch == "B5_SHUFFLED_CONTROL":
        gen = torch.Generator(device=update_unit.device).manual_seed(int(seed) + 5151)
        shuffled = torch.zeros_like(update_unit)
        vals = update_unit[support]
        perm = torch.randperm(vals.numel(), generator=gen, device=update_unit.device)
        shuffled[support] = vals[perm]
        return shuffled / shuffled.norm().clamp_min(1.0e-12)
    if branch in {"B2_RANDOM_CONTROL", "B3_STABLE_RANDOM_CONTROL"}:
        salt = 3131 if branch == "B2_RANDOM_CONTROL" else 4141
        gen = torch.Generator(device=update_unit.device).manual_seed(int(seed) + salt)
        rand = torch.zeros_like(update_unit)
        rand[support] = torch.randn((int(support.sum().detach().cpu()),), device=update_unit.device, generator=gen)
        return rand / rand.norm().clamp_min(1.0e-12)
    return update_unit.new_zeros(update_unit.shape)


def run_pilot_branch(
    *,
    branch: str,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    input_dim: int,
    device: Any,
    update_unit: Any,
    horizons: list[int],
    trust_ratio: float,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    max_h = max(horizons)
    horizon_set = set(horizons)
    train_iter = iter(train_loader)
    branch_start = time.perf_counter()
    auc_loss_time = 0.0
    prev_step_time = time.perf_counter()
    residual_norms: list[float] = []
    base_update_norms: list[float] = []
    train_losses: list[float] = []
    for step in range(1, max_h + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        before = flat_params(model)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        after = flat_params(model)
        base_update = after - before
        base_norm = float(base_update.norm().detach().cpu())
        residual_norm = 0.0
        if branch not in {"B0_BASE", "B6_SAME_OVERHEAD_NOOP"} and trust_ratio > 0.0 and base_norm > 0.0:
            residual = update_unit * (float(trust_ratio) * base_norm)
            with torch.no_grad():
                add_flat_to_params(model, residual)
            residual_norm = float(residual.norm().detach().cpu())
        now = time.perf_counter()
        step_dt = max(0.0, now - prev_step_time)
        prev_step_time = now
        train_loss = float(loss.detach().cpu())
        auc_loss_time += train_loss * step_dt
        train_losses.append(train_loss)
        base_update_norms.append(base_norm)
        residual_norms.append(residual_norm)
        if step in horizon_set:
            ev = evaluate_pilot_model(model, test_loader, input_dim, device)
            elapsed = time.perf_counter() - branch_start
            row = {
                "branch": branch,
                "horizon": step,
                "status": "completed",
                "final_test_NLL": ev["final_test_NLL"],
                "final_test_accuracy": ev["final_test_accuracy"],
                "ECE": ev["ECE"],
                "Brier": ev["Brier"],
                "tail_loss_q95": ev["tail_loss_q95"],
                "tail_loss_q99": ev["tail_loss_q99"],
                "AUC_loss_time": auc_loss_time,
                "wallclock_sec": elapsed,
                "mean_train_loss_to_H": statistics.fmean(train_losses),
                "mean_base_update_norm": statistics.fmean(base_update_norms),
                "mean_residual_norm": statistics.fmean(residual_norms),
                "mean_residual_to_base_norm_ratio": statistics.fmean(
                    [r / b for r, b in zip(residual_norms, base_update_norms) if b > 0.0]
                )
                if any(b > 0.0 for b in base_update_norms)
                else 0.0,
            }
            rows.append(row)
    return rows


def tangent_normalized_direction(model: Any, raw_unit: Any) -> tuple[Any, dict[str, float]]:
    import torch

    chunks: list[Any] = []
    radial_removed: list[float] = []
    empty_blocks = 0
    offset = 0
    for param in model.parameters():
        n = int(param.numel())
        raw_block = raw_unit[offset : offset + n].reshape(-1)
        w_block = param.detach().reshape(-1)
        offset += n
        raw_norm = raw_block.norm()
        w_norm = w_block.norm()
        if float(raw_norm.detach().cpu()) <= 1.0e-12:
            chunks.append(torch.zeros_like(raw_block))
            empty_blocks += 1
            continue
        if float(w_norm.detach().cpu()) <= 1.0e-12:
            tangent = raw_block
            removed_frac = 0.0
        else:
            w_hat = w_block / w_norm.clamp_min(1.0e-12)
            radial = (raw_block @ w_hat) * w_hat
            tangent = raw_block - radial
            removed_frac = float((radial.norm() / raw_norm.clamp_min(1.0e-12)).detach().cpu())
        tangent_norm = tangent.norm()
        if float(tangent_norm.detach().cpu()) <= 1.0e-12:
            chunks.append(torch.zeros_like(raw_block))
            empty_blocks += 1
        else:
            chunks.append(tangent / tangent_norm.clamp_min(1.0e-12) * raw_norm)
        radial_removed.append(removed_frac)
    full = torch.cat(chunks) if chunks else raw_unit.new_zeros(raw_unit.shape)
    norm = full.norm()
    if float(norm.detach().cpu()) <= 1.0e-12:
        return raw_unit.new_zeros(raw_unit.shape), {
            "tangent_radial_removed_fraction": statistics.fmean(radial_removed) if radial_removed else 0.0,
            "tangent_empty_block_fraction": 1.0,
        }
    return full / norm.clamp_min(1.0e-12), {
        "tangent_radial_removed_fraction": statistics.fmean(radial_removed) if radial_removed else 0.0,
        "tangent_empty_block_fraction": empty_blocks / max(1, len(chunks)),
    }


def run_tangent_normalized_branch(
    *,
    branch: str,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    input_dim: int,
    device: Any,
    raw_update_unit: Any,
    horizons: list[int],
    trust_ratio: float,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    max_h = max(horizons)
    horizon_set = set(horizons)
    train_iter = iter(train_loader)
    branch_start = time.perf_counter()
    auc_loss_time = 0.0
    prev_step_time = time.perf_counter()
    residual_norms: list[float] = []
    base_update_norms: list[float] = []
    train_losses: list[float] = []
    radial_removed_fracs: list[float] = []
    empty_block_fracs: list[float] = []
    for step in range(1, max_h + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        before = flat_params(model)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        after = flat_params(model)
        base_update = after - before
        base_norm = float(base_update.norm().detach().cpu())
        residual_norm = 0.0
        if branch not in {"B0_BASE", "B6_SAME_OVERHEAD_NOOP"} and trust_ratio > 0.0 and base_norm > 0.0:
            tangent_unit, tangent_metrics = tangent_normalized_direction(model, raw_update_unit)
            radial_removed_fracs.append(float(tangent_metrics["tangent_radial_removed_fraction"]))
            empty_block_fracs.append(float(tangent_metrics["tangent_empty_block_fraction"]))
            residual = tangent_unit * (float(trust_ratio) * base_norm)
            with torch.no_grad():
                add_flat_to_params(model, residual)
            residual_norm = float(residual.norm().detach().cpu())
        now = time.perf_counter()
        step_dt = max(0.0, now - prev_step_time)
        prev_step_time = now
        train_loss = float(loss.detach().cpu())
        auc_loss_time += train_loss * step_dt
        train_losses.append(train_loss)
        base_update_norms.append(base_norm)
        residual_norms.append(residual_norm)
        if step in horizon_set:
            ev = evaluate_pilot_model(model, test_loader, input_dim, device)
            elapsed = time.perf_counter() - branch_start
            rows.append(
                {
                    "branch": branch,
                    "horizon": step,
                    "status": "completed",
                    "final_test_NLL": ev["final_test_NLL"],
                    "final_test_accuracy": ev["final_test_accuracy"],
                    "ECE": ev["ECE"],
                    "Brier": ev["Brier"],
                    "tail_loss_q95": ev["tail_loss_q95"],
                    "tail_loss_q99": ev["tail_loss_q99"],
                    "AUC_loss_time": auc_loss_time,
                    "wallclock_sec": elapsed,
                    "mean_train_loss_to_H": statistics.fmean(train_losses),
                    "mean_base_update_norm": statistics.fmean(base_update_norms),
                    "mean_residual_norm": statistics.fmean(residual_norms),
                    "mean_residual_to_base_norm_ratio": statistics.fmean(
                        [r / b for r, b in zip(residual_norms, base_update_norms) if b > 0.0]
                    )
                    if any(b > 0.0 for b in base_update_norms)
                    else 0.0,
                    "tangent_radial_removed_fraction_mean": statistics.fmean(radial_removed_fracs) if radial_removed_fracs else 0.0,
                    "tangent_empty_block_fraction_mean": statistics.fmean(empty_block_fracs) if empty_block_fracs else 0.0,
                    "pilot_metric_scope": "block_global_tangent_normalized_signal_residual",
                    "pilot_metric_limitation": "T4 branch pilot uses block-global tangent projection and normalization; it is not a full Mano/Oblique optimizer implementation.",
                }
            )
    return rows


def initialize_gradient_subspace(
    model: Any,
    train_loader: Any,
    input_dim: int,
    device: Any,
    rank: int,
    cohorts: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    grads: list[Any] = []
    loader_iter = iter(train_loader)
    for _ in range(max(1, int(cohorts))):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            xb, yb = next(loader_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        grads.append(flat_grads_from_loss(model, loss))
    gmat = torch.stack(grads, dim=1)
    p_dim = int(gmat.shape[0])
    k = min(max(1, int(rank)), p_dim)
    if int(gmat.shape[1]) >= k:
        q = torch.linalg.qr(gmat, mode="reduced").Q[:, :k]
    else:
        gen = torch.Generator(device=device).manual_seed(int(seed) + 5959)
        filler = torch.randn((p_dim, k), device=device, generator=gen)
        filler[:, : int(gmat.shape[1])] = gmat
        q = torch.linalg.qr(filler, mode="reduced").Q[:, :k]
    mean_g = torch.stack(grads, dim=0).mean(dim=0)
    proj = q @ (q.T @ mean_g)
    recon = float((proj.norm() / mean_g.norm().clamp_min(1.0e-12)).detach().cpu())
    g_centered = torch.stack(grads, dim=0) - mean_g
    metrics = {
        "pilot_metric_scope": "online_gradient_subspace_tracking_oja_qr",
        "pilot_metric_limitation": "T5 branch pilot tracks parameter-gradient subspace, not JVP/effect subspace; same tracked subspace is shared by real and controls.",
        "subspace_rank": int(k),
        "subspace_init_cohorts": int(len(grads)),
        "subspace_init_mean_grad_reconstruction_ratio": recon,
        "subspace_init_gradient_variance": float(g_centered.pow(2).sum(dim=1).mean().detach().cpu()),
    }
    return q.detach(), metrics


def update_oja_subspace(B: Any, grad_flat: Any, eta_o: float) -> tuple[Any, float, float]:
    import torch

    grad_norm = grad_flat.norm()
    if float(grad_norm.detach().cpu()) <= 1.0e-12:
        return B, 0.0, 0.0
    g = grad_flat / grad_norm.clamp_min(1.0e-12)
    coeff = g @ B
    before = B
    candidate = B + float(eta_o) * torch.outer(g, coeff)
    q = torch.linalg.qr(candidate, mode="reduced").Q[:, : int(B.shape[1])]
    tracking_step = float(torch.linalg.matrix_norm(q @ q.T - before @ before.T).detach().cpu())
    reconstruction_ratio = float(((B @ (B.T @ grad_flat)).norm() / grad_norm.clamp_min(1.0e-12)).detach().cpu())
    return q.detach(), tracking_step, reconstruction_ratio


def online_subspace_direction(B: Any, grad_flat: Any, branch: str, seed: int, step: int) -> tuple[Any, dict[str, float]]:
    import torch

    proj = B @ (B.T @ grad_flat)
    proj_norm = proj.norm()
    if float(proj_norm.detach().cpu()) <= 1.0e-12:
        base = torch.zeros_like(grad_flat)
    else:
        base = -proj / proj_norm.clamp_min(1.0e-12)
    coeff = B.T @ base
    if branch == "B1_REAL_ONLINE_SUBSPACE":
        direction = base
    elif branch == "B4_SIGNFLIP_CONTROL":
        direction = -base
    elif branch in {"B2_RANDOM_CONTROL", "B3_STABLE_RANDOM_CONTROL"}:
        salt = 9797 if branch == "B2_RANDOM_CONTROL" else 9898
        gen = torch.Generator(device=grad_flat.device).manual_seed(int(seed) + salt + int(step))
        rand_coeff = torch.randn((int(B.shape[1]),), device=grad_flat.device, generator=gen)
        direction = B @ rand_coeff
        direction = direction / direction.norm().clamp_min(1.0e-12)
    elif branch == "B5_SHUFFLED_CONTROL":
        gen = torch.Generator(device=grad_flat.device).manual_seed(int(seed) + 9991 + int(step))
        perm = torch.randperm(coeff.numel(), generator=gen, device=grad_flat.device)
        direction = B @ coeff[perm]
        direction = direction / direction.norm().clamp_min(1.0e-12)
    else:
        direction = torch.zeros_like(grad_flat)
    metrics = {
        "online_projection_norm": float(proj_norm.detach().cpu()),
        "online_direction_subspace_norm": float((B.T @ direction).norm().detach().cpu()) if float(direction.norm().detach().cpu()) > 0.0 else 0.0,
    }
    return direction.detach(), metrics


def run_online_subspace_branch(
    *,
    branch: str,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    input_dim: int,
    device: Any,
    init_subspace: Any,
    horizons: list[int],
    trust_ratio: float,
    oja_lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    max_h = max(horizons)
    horizon_set = set(horizons)
    train_iter = iter(train_loader)
    branch_start = time.perf_counter()
    auc_loss_time = 0.0
    prev_step_time = time.perf_counter()
    residual_norms: list[float] = []
    base_update_norms: list[float] = []
    train_losses: list[float] = []
    tracking_steps: list[float] = []
    reconstruction_ratios: list[float] = []
    projection_norms: list[float] = []
    direction_subspace_norms: list[float] = []
    B = init_subspace.clone().detach()
    for step in range(1, max_h + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        before = flat_params(model)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        grad_flat = flat_current_grads(model)
        B, tracking_step, reconstruction_ratio = update_oja_subspace(B, grad_flat, float(oja_lr))
        tracking_steps.append(tracking_step)
        reconstruction_ratios.append(reconstruction_ratio)
        direction, dir_metrics = online_subspace_direction(B, grad_flat, branch, seed, step)
        projection_norms.append(float(dir_metrics["online_projection_norm"]))
        direction_subspace_norms.append(float(dir_metrics["online_direction_subspace_norm"]))
        optimizer.step()
        after = flat_params(model)
        base_update = after - before
        base_norm = float(base_update.norm().detach().cpu())
        residual_norm = 0.0
        if branch not in {"B0_BASE", "B6_SAME_OVERHEAD_NOOP"} and trust_ratio > 0.0 and base_norm > 0.0:
            residual = direction * (float(trust_ratio) * base_norm)
            with torch.no_grad():
                add_flat_to_params(model, residual)
            residual_norm = float(residual.norm().detach().cpu())
        now = time.perf_counter()
        step_dt = max(0.0, now - prev_step_time)
        prev_step_time = now
        train_loss = float(loss.detach().cpu())
        auc_loss_time += train_loss * step_dt
        train_losses.append(train_loss)
        base_update_norms.append(base_norm)
        residual_norms.append(residual_norm)
        if step in horizon_set:
            ev = evaluate_pilot_model(model, test_loader, input_dim, device)
            elapsed = time.perf_counter() - branch_start
            rows.append(
                {
                    "branch": branch,
                    "horizon": step,
                    "status": "completed",
                    "final_test_NLL": ev["final_test_NLL"],
                    "final_test_accuracy": ev["final_test_accuracy"],
                    "ECE": ev["ECE"],
                    "Brier": ev["Brier"],
                    "tail_loss_q95": ev["tail_loss_q95"],
                    "tail_loss_q99": ev["tail_loss_q99"],
                    "AUC_loss_time": auc_loss_time,
                    "wallclock_sec": elapsed,
                    "mean_train_loss_to_H": statistics.fmean(train_losses),
                    "mean_base_update_norm": statistics.fmean(base_update_norms),
                    "mean_residual_norm": statistics.fmean(residual_norms),
                    "mean_residual_to_base_norm_ratio": statistics.fmean(
                        [r / b for r, b in zip(residual_norms, base_update_norms) if b > 0.0]
                    )
                    if any(b > 0.0 for b in base_update_norms)
                    else 0.0,
                    "oja_lr": float(oja_lr),
                    "subspace_tracking_step_mean": statistics.fmean(tracking_steps),
                    "subspace_grad_reconstruction_ratio_mean": statistics.fmean(reconstruction_ratios),
                    "online_projection_norm_mean": statistics.fmean(projection_norms),
                    "online_direction_subspace_norm_mean": statistics.fmean(direction_subspace_norms),
                    "pilot_metric_scope": "online_gradient_subspace_tracking_oja_qr",
                    "pilot_metric_limitation": "T5 branch pilot uses parameter-gradient Oja/QR tracking; JVP/effect subspace tracking remains deferred unless this branch opens.",
                }
            )
    return rows


def readout_effect_examples(model: Any, xb: Any, yb: Any, input_dim: int, device: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    x = flatten_batch(xb, input_dim, device)
    y = yb.to(device).long()
    with torch.no_grad():
        h = model.frozen_readout_features(x)
        logits = h @ model.w2
        err = logits.softmax(dim=1) - F.one_hot(y, num_classes=logits.shape[1]).float()
        per_example = torch.einsum("bi,bj->bij", h, err).reshape(x.shape[0], -1)
    return per_example.detach(), per_example.mean(dim=0).detach()


def initialize_effect_subspace(
    model: Any,
    train_loader: Any,
    input_dim: int,
    device: Any,
    rank: int,
    cohorts: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    effects: list[Any] = []
    means: list[Any] = []
    loader_iter = iter(train_loader)
    for _ in range(max(1, int(cohorts))):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            xb, yb = next(loader_iter)
        per_example, mean_effect = readout_effect_examples(model, xb, yb, input_dim, device)
        effects.append(per_example)
        means.append(mean_effect)
    emat = torch.cat(effects, dim=0)
    centered = emat - emat.mean(dim=0, keepdim=True)
    p_dim = int(emat.shape[1])
    k = min(max(1, int(rank)), p_dim, int(centered.shape[0]))
    if k <= 0:
        gen = torch.Generator(device=device).manual_seed(int(seed) + 6767)
        q = torch.linalg.qr(torch.randn((p_dim, 1), device=device, generator=gen), mode="reduced").Q[:, :1]
    else:
        q = torch.linalg.qr(centered.T, mode="reduced").Q[:, :k]
    mean_effect = torch.stack(means, dim=0).mean(dim=0)
    proj = q @ (q.T @ mean_effect)
    metrics = {
        "pilot_metric_scope": "online_last_layer_effect_subspace_tracking_oja_qr",
        "pilot_metric_limitation": "T5 effect-space repair tracks final readout per-example effect gradients only; it is not dense all-parameter JVP/effect tracking.",
        "effect_subspace_rank": int(q.shape[1]),
        "effect_subspace_init_examples": int(emat.shape[0]),
        "effect_subspace_init_mean_reconstruction_ratio": float((proj.norm() / mean_effect.norm().clamp_min(1.0e-12)).detach().cpu()),
        "effect_subspace_init_variance": float(centered.pow(2).sum(dim=1).mean().detach().cpu()),
    }
    return q.detach(), metrics


def update_effect_oja_subspace(B: Any, per_example_effects: Any, mean_effect: Any, eta_o: float) -> tuple[Any, float, float]:
    import torch

    centered = per_example_effects - per_example_effects.mean(dim=0, keepdim=True)
    if int(centered.shape[0]) <= 0:
        return B, 0.0, 0.0
    cov_B = centered.T @ (centered @ B) / float(max(1, int(centered.shape[0]) - 1))
    before = B
    candidate = B + float(eta_o) * cov_B
    q = torch.linalg.qr(candidate, mode="reduced").Q[:, : int(B.shape[1])]
    tracking_step = float(torch.linalg.matrix_norm(q @ q.T - before @ before.T).detach().cpu())
    mean_norm = mean_effect.norm()
    reconstruction_ratio = float(((B @ (B.T @ mean_effect)).norm() / mean_norm.clamp_min(1.0e-12)).detach().cpu())
    return q.detach(), tracking_step, reconstruction_ratio


def online_effect_subspace_direction(B: Any, mean_effect: Any, branch: str, seed: int, step: int) -> tuple[Any, dict[str, float]]:
    import torch

    proj = B @ (B.T @ mean_effect)
    proj_norm = proj.norm()
    if float(proj_norm.detach().cpu()) <= 1.0e-12:
        base = torch.zeros_like(mean_effect)
    else:
        base = -proj / proj_norm.clamp_min(1.0e-12)
    coeff = B.T @ base
    if branch == "B1_REAL_ONLINE_EFFECT_SUBSPACE":
        direction_last = base
    elif branch == "B4_SIGNFLIP_CONTROL":
        direction_last = -base
    elif branch in {"B2_RANDOM_CONTROL", "B3_STABLE_RANDOM_CONTROL"}:
        salt = 10797 if branch == "B2_RANDOM_CONTROL" else 10898
        gen = torch.Generator(device=mean_effect.device).manual_seed(int(seed) + salt + int(step))
        rand_coeff = torch.randn((int(B.shape[1]),), device=mean_effect.device, generator=gen)
        direction_last = B @ rand_coeff
        direction_last = direction_last / direction_last.norm().clamp_min(1.0e-12)
    elif branch == "B5_SHUFFLED_CONTROL":
        gen = torch.Generator(device=mean_effect.device).manual_seed(int(seed) + 10991 + int(step))
        perm = torch.randperm(coeff.numel(), generator=gen, device=mean_effect.device)
        direction_last = B @ coeff[perm]
        direction_last = direction_last / direction_last.norm().clamp_min(1.0e-12)
    else:
        direction_last = torch.zeros_like(mean_effect)
    metrics = {
        "effect_projection_norm": float(proj_norm.detach().cpu()),
        "effect_direction_subspace_norm": float((B.T @ direction_last).norm().detach().cpu())
        if float(direction_last.norm().detach().cpu()) > 0.0
        else 0.0,
    }
    return direction_last.detach(), metrics


def run_online_effect_subspace_branch(
    *,
    branch: str,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    input_dim: int,
    device: Any,
    init_subspace: Any,
    horizons: list[int],
    trust_ratio: float,
    oja_lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    max_h = max(horizons)
    horizon_set = set(horizons)
    train_iter = iter(train_loader)
    branch_start = time.perf_counter()
    auc_loss_time = 0.0
    prev_step_time = time.perf_counter()
    residual_norms: list[float] = []
    base_update_norms: list[float] = []
    train_losses: list[float] = []
    tracking_steps: list[float] = []
    reconstruction_ratios: list[float] = []
    projection_norms: list[float] = []
    direction_subspace_norms: list[float] = []
    B = init_subspace.clone().detach()
    for step in range(1, max_h + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        per_example_effects, mean_effect = readout_effect_examples(model, xb, yb, input_dim, device)
        B, tracking_step, reconstruction_ratio = update_effect_oja_subspace(B, per_example_effects, mean_effect, float(oja_lr))
        tracking_steps.append(tracking_step)
        reconstruction_ratios.append(reconstruction_ratio)
        direction_last, dir_metrics = online_effect_subspace_direction(B, mean_effect, branch, seed, step)
        projection_norms.append(float(dir_metrics["effect_projection_norm"]))
        direction_subspace_norms.append(float(dir_metrics["effect_direction_subspace_norm"]))
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        before = flat_params(model)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        after = flat_params(model)
        base_update = after - before
        base_norm = float(base_update.norm().detach().cpu())
        residual_norm = 0.0
        if branch not in {"B0_BASE", "B6_SAME_OVERHEAD_NOOP"} and trust_ratio > 0.0 and base_norm > 0.0:
            direction = last_parameter_full_direction(model, direction_last, "w2")
            residual = direction * (float(trust_ratio) * base_norm)
            with torch.no_grad():
                add_flat_to_params(model, residual)
            residual_norm = float(residual.norm().detach().cpu())
        now = time.perf_counter()
        step_dt = max(0.0, now - prev_step_time)
        prev_step_time = now
        train_loss = float(loss.detach().cpu())
        auc_loss_time += train_loss * step_dt
        train_losses.append(train_loss)
        base_update_norms.append(base_norm)
        residual_norms.append(residual_norm)
        if step in horizon_set:
            ev = evaluate_pilot_model(model, test_loader, input_dim, device)
            elapsed = time.perf_counter() - branch_start
            rows.append(
                {
                    "branch": branch,
                    "horizon": step,
                    "status": "completed",
                    "final_test_NLL": ev["final_test_NLL"],
                    "final_test_accuracy": ev["final_test_accuracy"],
                    "ECE": ev["ECE"],
                    "Brier": ev["Brier"],
                    "tail_loss_q95": ev["tail_loss_q95"],
                    "tail_loss_q99": ev["tail_loss_q99"],
                    "AUC_loss_time": auc_loss_time,
                    "wallclock_sec": elapsed,
                    "mean_train_loss_to_H": statistics.fmean(train_losses),
                    "mean_base_update_norm": statistics.fmean(base_update_norms),
                    "mean_residual_norm": statistics.fmean(residual_norms),
                    "mean_residual_to_base_norm_ratio": statistics.fmean(
                        [r / b for r, b in zip(residual_norms, base_update_norms) if b > 0.0]
                    )
                    if any(b > 0.0 for b in base_update_norms)
                    else 0.0,
                    "oja_lr": float(oja_lr),
                    "effect_subspace_tracking_step_mean": statistics.fmean(tracking_steps),
                    "effect_subspace_grad_reconstruction_ratio_mean": statistics.fmean(reconstruction_ratios),
                    "effect_projection_norm_mean": statistics.fmean(projection_norms),
                    "effect_direction_subspace_norm_mean": statistics.fmean(direction_subspace_norms),
                    "pilot_metric_scope": "online_last_layer_effect_subspace_tracking_oja_qr",
                    "pilot_metric_limitation": "T5 effect-space repair uses final readout per-example effect Oja/QR tracking; dense all-parameter JVP/effect tracking remains deferred unless this branch opens.",
                }
            )
    return rows


def temporal_control_direction(slow_unit: Any, branch: str, seed: int) -> Any:
    import torch

    if branch == "B1_REAL_SLOW_SIGNAL_EMA":
        return -slow_unit
    if branch == "B4_SIGNFLIP_CONTROL":
        return slow_unit
    if branch == "B5_SHUFFLED_CONTROL":
        gen = torch.Generator(device=slow_unit.device).manual_seed(int(seed) + 6161)
        perm = torch.randperm(slow_unit.numel(), generator=gen, device=slow_unit.device)
        shuffled = (-slow_unit)[perm]
        return shuffled / shuffled.norm().clamp_min(1.0e-12)
    if branch in {"B2_RANDOM_CONTROL", "B3_STABLE_RANDOM_CONTROL"}:
        salt = 7171 if branch == "B2_RANDOM_CONTROL" else 8181
        gen = torch.Generator(device=slow_unit.device).manual_seed(int(seed) + salt)
        rand = torch.randn(slow_unit.shape, device=slow_unit.device, generator=gen)
        return rand / rand.norm().clamp_min(1.0e-12)
    return slow_unit.new_zeros(slow_unit.shape)


def run_temporal_refresh_branch(
    *,
    branch: str,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    test_loader: Any,
    input_dim: int,
    device: Any,
    horizons: list[int],
    trust_ratio: float,
    slow_beta: float,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    max_h = max(horizons)
    horizon_set = set(horizons)
    train_iter = iter(train_loader)
    branch_start = time.perf_counter()
    auc_loss_time = 0.0
    prev_step_time = time.perf_counter()
    residual_norms: list[float] = []
    base_update_norms: list[float] = []
    train_losses: list[float] = []
    slow_norms: list[float] = []
    fast_norms: list[float] = []
    slow_fast_ratios: list[float] = []
    slow_grad_alignments: list[float] = []
    slow_grad = None
    for step in range(1, max_h + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        x = flatten_batch(xb, input_dim, device)
        y = yb.to(device).long()
        before = flat_params(model)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        grad_flat = flat_current_grads(model)
        if slow_grad is None:
            slow_grad = grad_flat.clone()
        else:
            slow_grad = float(slow_beta) * slow_grad + (1.0 - float(slow_beta)) * grad_flat
        fast_grad = grad_flat - slow_grad
        slow_norm = float(slow_grad.norm().detach().cpu())
        fast_norm = float(fast_grad.norm().detach().cpu())
        grad_norm = float(grad_flat.norm().detach().cpu())
        slow_norms.append(slow_norm)
        fast_norms.append(fast_norm)
        if fast_norm > 1.0e-9:
            slow_fast_ratios.append(slow_norm / fast_norm)
        if grad_norm > 0.0 and slow_norm > 0.0:
            slow_grad_alignments.append(float(((slow_grad @ grad_flat) / (slow_grad.norm() * grad_flat.norm()).clamp_min(1.0e-12)).detach().cpu()))
        optimizer.step()
        after = flat_params(model)
        base_update = after - before
        base_norm = float(base_update.norm().detach().cpu())
        residual_norm = 0.0
        if branch not in {"B0_BASE", "B6_SAME_OVERHEAD_NOOP"} and trust_ratio > 0.0 and base_norm > 0.0 and slow_norm > 0.0:
            slow_unit = slow_grad / slow_grad.norm().clamp_min(1.0e-12)
            direction = temporal_control_direction(slow_unit, branch, seed)
            residual = direction * (float(trust_ratio) * base_norm)
            with torch.no_grad():
                add_flat_to_params(model, residual)
            residual_norm = float(residual.norm().detach().cpu())
        now = time.perf_counter()
        step_dt = max(0.0, now - prev_step_time)
        prev_step_time = now
        train_loss = float(loss.detach().cpu())
        auc_loss_time += train_loss * step_dt
        train_losses.append(train_loss)
        base_update_norms.append(base_norm)
        residual_norms.append(residual_norm)
        if step in horizon_set:
            ev = evaluate_pilot_model(model, test_loader, input_dim, device)
            elapsed = time.perf_counter() - branch_start
            rows.append(
                {
                    "branch": branch,
                    "horizon": step,
                    "status": "completed",
                    "final_test_NLL": ev["final_test_NLL"],
                    "final_test_accuracy": ev["final_test_accuracy"],
                    "ECE": ev["ECE"],
                    "Brier": ev["Brier"],
                    "tail_loss_q95": ev["tail_loss_q95"],
                    "tail_loss_q99": ev["tail_loss_q99"],
                    "AUC_loss_time": auc_loss_time,
                    "wallclock_sec": elapsed,
                    "mean_train_loss_to_H": statistics.fmean(train_losses),
                    "mean_base_update_norm": statistics.fmean(base_update_norms),
                    "mean_residual_norm": statistics.fmean(residual_norms),
                    "mean_residual_to_base_norm_ratio": statistics.fmean(
                        [r / b for r, b in zip(residual_norms, base_update_norms) if b > 0.0]
                    )
                    if any(b > 0.0 for b in base_update_norms)
                    else 0.0,
                    "slow_beta": float(slow_beta),
                    "slow_gradient_norm_mean": statistics.fmean(slow_norms),
                    "fast_gradient_norm_mean": statistics.fmean(fast_norms),
                    "slow_fast_norm_ratio_mean": statistics.fmean(slow_fast_ratios) if slow_fast_ratios else "",
                    "slow_grad_alignment_mean": statistics.fmean(slow_grad_alignments) if slow_grad_alignments else "",
                    "pilot_metric_scope": "online_slow_gradient_ema_training_only",
                    "pilot_metric_limitation": "single-run branch pilot; no repeat-baseline noise estimate yet",
                }
            )
    return rows


def enrich_branch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("module_id")), int(row.get("horizon")))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for _key, members in grouped.items():
        base = next((m for m in members if m.get("branch") == "B0_BASE"), None)
        controls = [m for m in members if int_flag(m.get("is_control_branch")) == 1 and m.get("branch") != "B6_SAME_OVERHEAD_NOOP"]
        noop = next((m for m in members if m.get("branch") == "B6_SAME_OVERHEAD_NOOP"), None)
        base_nll = finite_float(base.get("final_test_NLL")) if base else None
        base_auc = finite_float(base.get("AUC_loss_time")) if base else None
        base_ece = finite_float(base.get("ECE")) if base else None
        base_brier = finite_float(base.get("Brier")) if base else None
        base_tail = finite_float(base.get("tail_loss_q95")) if base else None
        control_nll_values = [v for v in (finite_float(c.get("final_test_NLL")) for c in controls) if v is not None]
        control_auc_values = [v for v in (finite_float(c.get("AUC_loss_time")) for c in controls) if v is not None]
        control_best_nll = min(control_nll_values) if control_nll_values else None
        control_best_auc = min(control_auc_values) if control_auc_values else None
        noop_nll = finite_float(noop.get("final_test_NLL")) if noop else None
        for row in members:
            item = dict(row)
            nll = finite_float(item.get("final_test_NLL"))
            auc = finite_float(item.get("AUC_loss_time"))
            ece = finite_float(item.get("ECE"))
            brier = finite_float(item.get("Brier"))
            tail = finite_float(item.get("tail_loss_q95"))
            item["paired_NLL_delta_H"] = "" if base_nll is None or nll is None else nll - base_nll
            item["paired_AUC_delta_H"] = "" if base_auc is None or auc is None else auc - base_auc
            item["paired_ECE_delta_H"] = "" if base_ece is None or ece is None else ece - base_ece
            item["paired_Brier_delta_H"] = "" if base_brier is None or brier is None else brier - base_brier
            item["paired_tail_q95_delta_H"] = "" if base_tail is None or tail is None else tail - base_tail
            item["control_best_NLL"] = "" if control_best_nll is None else control_best_nll
            item["control_best_AUC_loss_time"] = "" if control_best_auc is None else control_best_auc
            item["control_best_delta_H"] = "" if control_best_nll is None or base_nll is None else control_best_nll - base_nll
            item["beats_base_NLL"] = int(base_nll is not None and nll is not None and nll < base_nll - 1.0e-9)
            item["beats_best_control_NLL"] = int(control_best_nll is not None and nll is not None and nll <= control_best_nll - 1.0e-9)
            item["beats_base_AUC"] = int(base_auc is not None and auc is not None and auc < base_auc - 1.0e-9)
            item["beats_best_control_AUC"] = int(control_best_auc is not None and auc is not None and auc <= control_best_auc - 1.0e-9)
            item["noop_NLL_delta_H"] = "" if noop_nll is None or base_nll is None else noop_nll - base_nll
            out.append(item)
    return out


def run_continuation_pilot(args: argparse.Namespace, route_summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_continuation_pilot_start",
        status="started",
        gpu=args.pilot_device,
        note=(
            "Direct T1/T2 branch-causal continuation after Gate1 hard-stop; "
            f"datasets={args.pilot_datasets}; horizons={args.pilot_horizons}; "
            "historical v22.16 payload/outcome could not be exact-paired, so this pilot creates paired branch evidence."
        ),
    )
    device = pilot_device(args.pilot_device)
    datasets = split_csv(args.pilot_datasets)
    horizons = [int(x) for x in split_csv(args.pilot_horizons)]
    branches = [
        ("B0_BASE", 0, "base optimizer continuation"),
        ("B1_REAL_SIGNAL_GRASSMANN", 0, "real cohort signal/Grassmann direction"),
        ("B2_RANDOM_CONTROL", 1, "random same-norm/cadence residual"),
        ("B3_STABLE_RANDOM_CONTROL", 1, "stable random same-norm/cadence residual"),
        ("B4_SIGNFLIP_CONTROL", 1, "signflip same-norm/cadence residual"),
        ("B5_SHUFFLED_CONTROL", 1, "shuffled direction same-norm/cadence residual"),
        ("B6_SAME_OVERHEAD_NOOP", 1, "same-overhead no residual"),
    ]
    all_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    kan_root = Path(args.kanbefair_root)
    if not kan_root.is_absolute():
        kan_root = ROOT / kan_root
    data_root = Path(args.pilot_data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for dataset in datasets:
        for seed in [int(x) for x in split_csv(args.pilot_seeds)]:
            try:
                torch.manual_seed(int(seed) + 29000)
                train_loader, test_loader, num_classes, input_dim = load_kanbefair_loaders(
                    kan_root,
                    data_root,
                    dataset,
                    int(args.pilot_batch_size),
                    int(args.pilot_train_size),
                    int(args.pilot_test_size),
                    seed,
                )
                base_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                train_iter = iter(train_loader)
                for _step in range(int(args.pilot_pretrain_steps)):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb = next(train_iter)
                    x = flatten_batch(xb, input_dim, device)
                    y = yb.to(device).long()
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(base_model(x), y)
                    loss.backward()
                    opt.step()
                checkpoint_eval = evaluate_pilot_model(base_model, test_loader, input_dim, device)
                update_unit, signal_metrics = compute_pilot_signal_metrics(
                    base_model,
                    train_loader,
                    input_dim,
                    device,
                    int(args.pilot_cohorts),
                    seed,
                )
                model_state = copy.deepcopy(base_model.state_dict())
                opt_state = copy.deepcopy(opt.state_dict())
                for branch, is_control, branch_note in branches:
                    branch_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                    branch_model.load_state_dict(copy.deepcopy(model_state))
                    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                    branch_opt.load_state_dict(copy.deepcopy(opt_state))
                    direction = control_direction(update_unit, branch, seed)
                    branch_rows = run_pilot_branch(
                        branch=branch,
                        model=branch_model,
                        optimizer=branch_opt,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        input_dim=input_dim,
                        device=device,
                        update_unit=direction,
                        horizons=horizons,
                        trust_ratio=float(args.pilot_trust_ratio),
                    )
                    for row in branch_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "module_id": "T1+T2",
                                "model_family": "MLP",
                                "model_name": "MLP_ADAMW_PLUS_BRANCH_RESIDUAL",
                                "branch_note": branch_note,
                                "is_control_branch": is_control,
                                "pretrain_steps": int(args.pilot_pretrain_steps),
                                "train_size": int(args.pilot_train_size),
                                "test_size": int(args.pilot_test_size),
                                "batch_size": int(args.pilot_batch_size),
                                "hidden_dim": int(args.pilot_hidden),
                                "trust_ratio": float(args.pilot_trust_ratio),
                                "checkpoint_test_NLL": checkpoint_eval["final_test_NLL"],
                                "checkpoint_test_accuracy": checkpoint_eval["final_test_accuracy"],
                                **signal_metrics,
                            }
                        )
                        all_rows.append(row)
            except Exception as exc:
                deferred.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "deferred",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "latest_status_timestamp": now_sg(),
                    }
                )
    enriched = enrich_branch_rows(all_rows)
    write_rows(OUT_ROOT / "v22_29_branch_causal_pilot_matrix.csv", enriched)
    write_rows(OUT_ROOT / "v22_29_branch_causal_controls_matrix.csv", [r for r in enriched if int_flag(r.get("is_control_branch")) == 1])
    write_rows(OUT_ROOT / "v22_29_continuation_deferred_items.csv", deferred)
    h100_h200 = [r for r in enriched if r.get("branch") == "B1_REAL_SIGNAL_GRASSMANN" and int(r.get("horizon") or 0) in {100, 200}]
    real_beats_base = sum(int_flag(r.get("beats_base_NLL")) for r in h100_h200)
    real_beats_controls = sum(int_flag(r.get("beats_best_control_NLL")) for r in h100_h200)
    eligible = len(h100_h200)
    pass_fraction = real_beats_controls / eligible if eligible else 0.0
    real_pass = eligible > 0 and pass_fraction >= 0.60 and real_beats_base / eligible >= 0.60
    controls_better_rows = sum(
        1
        for r in h100_h200
        if finite_float(r.get("final_test_NLL")) is not None
        and finite_float(r.get("control_best_NLL")) is not None
        and float(r["control_best_NLL"]) < float(r["final_test_NLL"]) - 1.0e-9
    )
    final_route = "R4-BranchCausalSignalOpened_FullLoopPending" if real_pass else (
        "R3-ControlsExplainAllMechanisms" if controls_better_rows > 0 else "R2-BranchCausalNoMechanismPass"
    )
    summary = {
        "continuation_pilot_launched": int(bool(enriched)),
        "continuation_pilot_rows": len(enriched),
        "continuation_deferred_rows": len(deferred),
        "continuation_h100_h200_real_rows": eligible,
        "continuation_h100_h200_real_beats_base_rows": real_beats_base,
        "continuation_h100_h200_real_beats_best_control_rows": real_beats_controls,
        "continuation_h100_h200_real_pass_fraction": pass_fraction,
        "continuation_controls_better_real_rows": controls_better_rows,
        "continuation_branch_pass": int(real_pass),
        "route_after_continuation": final_route,
        "latest_status_timestamp": now_sg(),
    }
    route_summary.update(summary)
    write_json(OUT_ROOT / "v22_29_continuation_pilot_summary.json", summary)
    append_exec(
        command,
        task_id="v22_29_continuation_pilot",
        status="pass" if enriched else "blocked",
        gpu=str(device),
        files=(
            "results/v22_29/v22_29_branch_causal_pilot_matrix.csv, "
            "results/v22_29/v22_29_branch_causal_controls_matrix.csv, "
            "results/v22_29/v22_29_continuation_pilot_summary.json"
        ),
        note=(
            f"rows={len(enriched)} deferred={len(deferred)} route_after_continuation={final_route}; "
            f"h100_h200_real_beats_base={real_beats_base}/{eligible}; "
            f"h100_h200_real_beats_best_control={real_beats_controls}/{eligible}"
        ),
        exit_code=0 if enriched else 1,
    )
    return summary


def run_transported_momentum_pilot(args: argparse.Namespace, route_summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_transported_momentum_pilot_start",
        status="started",
        gpu=args.pilot_device,
        note=(
            "T3 Grassmann-transported source momentum pilot; previous cohort-gradient subspace is captured during pretrain, "
            "transported to the branch checkpoint with Procrustes alignment, and controls are restricted to the same current subspace; "
            f"rank={args.pilot_subspace_rank}; beta={args.pilot_transport_beta}; gap={args.pilot_transport_gap}; "
            f"datasets={args.pilot_datasets}; horizons={args.pilot_horizons}"
        ),
    )
    device = pilot_device(args.pilot_device)
    datasets = split_csv(args.pilot_datasets)
    horizons = [int(x) for x in split_csv(args.pilot_horizons)]
    branches = [
        ("B0_BASE", 0, "base optimizer continuation with transport-probe overhead"),
        ("B1_REAL_TRANSPORTED_SOURCE", 0, "real Procrustes-transported source momentum"),
        ("B2_RANDOM_CONTROL", 1, "random direction in same current subspace"),
        ("B3_STABLE_RANDOM_CONTROL", 1, "stable random direction in same current subspace"),
        ("B4_SIGNFLIP_CONTROL", 1, "signflip transported direction"),
        ("B5_SHUFFLED_CONTROL", 1, "shuffled transported coefficients in same current subspace"),
        ("B6_SAME_OVERHEAD_NOOP", 1, "same-overhead no residual"),
    ]
    all_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    kan_root = Path(args.kanbefair_root)
    if not kan_root.is_absolute():
        kan_root = ROOT / kan_root
    data_root = Path(args.pilot_data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for dataset in datasets:
        for seed in [int(x) for x in split_csv(args.pilot_seeds)]:
            try:
                torch.manual_seed(int(seed) + 29500)
                train_loader, test_loader, num_classes, input_dim = load_kanbefair_loaders(
                    kan_root,
                    data_root,
                    dataset,
                    int(args.pilot_batch_size),
                    int(args.pilot_train_size),
                    int(args.pilot_test_size),
                    seed,
                )
                base_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                pretrain_steps = int(args.pilot_pretrain_steps)
                capture_step = max(0, pretrain_steps - max(1, int(args.pilot_transport_gap)))
                prev_basis = None
                prev_update = None
                prev_metrics: dict[str, Any] = {}
                if capture_step == 0:
                    prev_basis, prev_update, prev_metrics = gradient_subspace_snapshot(
                        base_model,
                        train_loader,
                        input_dim,
                        device,
                        int(args.pilot_cohorts),
                        int(args.pilot_subspace_rank),
                        seed,
                        salt=29501,
                    )
                train_iter = iter(train_loader)
                for step in range(1, pretrain_steps + 1):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb = next(train_iter)
                    x = flatten_batch(xb, input_dim, device)
                    y = yb.to(device).long()
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(base_model(x), y)
                    loss.backward()
                    opt.step()
                    if step == capture_step:
                        prev_basis, prev_update, prev_metrics = gradient_subspace_snapshot(
                            base_model,
                            train_loader,
                            input_dim,
                            device,
                            int(args.pilot_cohorts),
                            int(args.pilot_subspace_rank),
                            seed,
                            salt=29502,
                        )
                if prev_basis is None or prev_update is None:
                    prev_basis, prev_update, prev_metrics = gradient_subspace_snapshot(
                        base_model,
                        train_loader,
                        input_dim,
                        device,
                        int(args.pilot_cohorts),
                        int(args.pilot_subspace_rank),
                        seed,
                        salt=29503,
                    )
                curr_basis, curr_update, curr_metrics = gradient_subspace_snapshot(
                    base_model,
                    train_loader,
                    input_dim,
                    device,
                    int(args.pilot_cohorts),
                    int(args.pilot_subspace_rank),
                    seed,
                    salt=29504,
                )
                update_unit, transport_metrics = compute_transported_source_direction(
                    prev_basis,
                    curr_basis,
                    prev_update,
                    curr_update,
                    beta=float(args.pilot_transport_beta),
                )
                checkpoint_eval = evaluate_pilot_model(base_model, test_loader, input_dim, device)
                model_state = copy.deepcopy(base_model.state_dict())
                opt_state = copy.deepcopy(opt.state_dict())
                for branch, is_control, branch_note in branches:
                    branch_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                    branch_model.load_state_dict(copy.deepcopy(model_state))
                    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                    branch_opt.load_state_dict(copy.deepcopy(opt_state))
                    direction = transported_control_direction(curr_basis, update_unit, branch, seed)
                    branch_rows = run_pilot_branch(
                        branch=branch,
                        model=branch_model,
                        optimizer=branch_opt,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        input_dim=input_dim,
                        device=device,
                        update_unit=direction,
                        horizons=horizons,
                        trust_ratio=float(args.pilot_trust_ratio),
                    )
                    for row in branch_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "module_id": "T3",
                                "model_family": "MLP",
                                "model_name": "MLP_ADAMW_PLUS_GRASSMANN_TRANSPORTED_SOURCE_RESIDUAL",
                                "branch_note": branch_note,
                                "is_control_branch": is_control,
                                "pretrain_steps": pretrain_steps,
                                "transport_capture_step": capture_step,
                                "transport_gap_steps": pretrain_steps - capture_step,
                                "train_size": int(args.pilot_train_size),
                                "test_size": int(args.pilot_test_size),
                                "batch_size": int(args.pilot_batch_size),
                                "hidden_dim": int(args.pilot_hidden),
                                "trust_ratio": float(args.pilot_trust_ratio),
                                "checkpoint_test_NLL": checkpoint_eval["final_test_NLL"],
                                "checkpoint_test_accuracy": checkpoint_eval["final_test_accuracy"],
                                "matched_control_geometry": "same_current_grassmann_subspace",
                                "prev_mean_update_reconstruction_ratio": prev_metrics.get("mean_update_reconstruction_ratio", ""),
                                "curr_mean_update_reconstruction_ratio": curr_metrics.get("mean_update_reconstruction_ratio", ""),
                                "prev_gradient_variance": prev_metrics.get("gradient_variance", ""),
                                "curr_gradient_variance": curr_metrics.get("gradient_variance", ""),
                                **transport_metrics,
                            }
                        )
                        all_rows.append(row)
            except Exception as exc:
                deferred.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "deferred",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "latest_status_timestamp": now_sg(),
                    }
                )
    enriched = enrich_branch_rows(all_rows)
    write_rows(OUT_ROOT / "v22_29_transported_momentum_pilot_matrix.csv", enriched)
    write_rows(OUT_ROOT / "v22_29_transported_momentum_controls_matrix.csv", [r for r in enriched if int_flag(r.get("is_control_branch")) == 1])
    write_rows(OUT_ROOT / "v22_29_transported_momentum_deferred_items.csv", deferred)
    h100_h200 = [r for r in enriched if r.get("branch") == "B1_REAL_TRANSPORTED_SOURCE" and int(r.get("horizon") or 0) in {100, 200}]
    real_beats_base = sum(int_flag(r.get("beats_base_NLL")) for r in h100_h200)
    real_beats_controls = sum(int_flag(r.get("beats_best_control_NLL")) for r in h100_h200)
    eligible = len(h100_h200)
    pass_fraction = real_beats_controls / eligible if eligible else 0.0
    base_fraction = real_beats_base / eligible if eligible else 0.0
    real_pass = eligible > 0 and pass_fraction >= 0.60 and base_fraction >= 0.60
    controls_better_rows = sum(
        1
        for r in h100_h200
        if finite_float(r.get("final_test_NLL")) is not None
        and finite_float(r.get("control_best_NLL")) is not None
        and float(r["control_best_NLL"]) < float(r["final_test_NLL"]) - 1.0e-9
    )
    final_route = "R4-TransportedMomentumBranchOpened_FullLoopPending" if real_pass else (
        "R3-ControlsExplainAllMechanisms" if controls_better_rows > 0 else "R2-TransportedMomentumNoMechanismPass"
    )
    summary = {
        "transported_momentum_launched": int(bool(enriched)),
        "transported_momentum_rows": len(enriched),
        "transported_momentum_deferred_rows": len(deferred),
        "transported_h100_h200_real_rows": eligible,
        "transported_h100_h200_real_beats_base_rows": real_beats_base,
        "transported_h100_h200_real_beats_best_control_rows": real_beats_controls,
        "transported_h100_h200_real_pass_fraction": pass_fraction,
        "transported_controls_better_real_rows": controls_better_rows,
        "transported_momentum_branch_pass": int(real_pass),
        "route_after_transported_momentum": final_route,
        "latest_status_timestamp": now_sg(),
    }
    route_summary.update(summary)
    write_json(OUT_ROOT / "v22_29_transported_momentum_pilot_summary.json", summary)
    append_exec(
        command,
        task_id="v22_29_transported_momentum_pilot",
        status="pass" if enriched else "blocked",
        gpu=str(device),
        files=(
            "results/v22_29/v22_29_transported_momentum_pilot_matrix.csv, "
            "results/v22_29/v22_29_transported_momentum_controls_matrix.csv, "
            "results/v22_29/v22_29_transported_momentum_pilot_summary.json"
        ),
        note=(
            f"rows={len(enriched)} deferred={len(deferred)} route_after_transported_momentum={final_route}; "
            f"h100_h200_real_beats_base={real_beats_base}/{eligible}; "
            f"h100_h200_real_beats_best_control={real_beats_controls}/{eligible}"
        ),
        exit_code=0 if enriched else 1,
    )
    return summary


def run_effect_space_pilot(args: argparse.Namespace, route_summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_effect_space_pilot_start",
        status="started",
        gpu=args.pilot_device,
        note=(
            "T1 last-layer/logits per-example effect-space A_B repair; "
            "real and matched controls are restricted to the same final-readout support; "
            f"datasets={args.pilot_datasets}; horizons={args.pilot_horizons}; trust={args.pilot_trust_ratio}"
        ),
    )
    device = pilot_device(args.pilot_device)
    datasets = split_csv(args.pilot_datasets)
    horizons = [int(x) for x in split_csv(args.pilot_horizons)]
    branches = [
        ("B0_BASE", 0, "base optimizer continuation"),
        ("B1_REAL_EFFECT_AB", 0, "real last-layer/logits effect-space A_B top-eigen direction"),
        ("B2_RANDOM_CONTROL", 1, "same-support random same-norm/cadence residual"),
        ("B3_STABLE_RANDOM_CONTROL", 1, "same-support stable random same-norm/cadence residual"),
        ("B4_SIGNFLIP_CONTROL", 1, "same-support signflip same-norm/cadence residual"),
        ("B5_SHUFFLED_CONTROL", 1, "same-support shuffled direction same-norm/cadence residual"),
        ("B6_SAME_OVERHEAD_NOOP", 1, "same-overhead no residual"),
    ]
    all_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    kan_root = Path(args.kanbefair_root)
    if not kan_root.is_absolute():
        kan_root = ROOT / kan_root
    data_root = Path(args.pilot_data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for dataset in datasets:
        for seed in [int(x) for x in split_csv(args.pilot_seeds)]:
            try:
                torch.manual_seed(int(seed) + 29200)
                train_loader, test_loader, num_classes, input_dim = load_kanbefair_loaders(
                    kan_root,
                    data_root,
                    dataset,
                    int(args.pilot_batch_size),
                    int(args.pilot_train_size),
                    int(args.pilot_test_size),
                    seed,
                )
                base_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                train_iter = iter(train_loader)
                for _step in range(int(args.pilot_pretrain_steps)):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb = next(train_iter)
                    x = flatten_batch(xb, input_dim, device)
                    y = yb.to(device).long()
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(base_model(x), y)
                    loss.backward()
                    opt.step()
                checkpoint_eval = evaluate_pilot_model(base_model, test_loader, input_dim, device)
                update_unit, signal_metrics = compute_effect_space_signal_metrics(
                    base_model,
                    train_loader,
                    input_dim,
                    device,
                    int(args.pilot_cohorts),
                    seed,
                )
                model_state = copy.deepcopy(base_model.state_dict())
                opt_state = copy.deepcopy(opt.state_dict())
                for branch, is_control, branch_note in branches:
                    branch_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                    branch_model.load_state_dict(copy.deepcopy(model_state))
                    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                    branch_opt.load_state_dict(copy.deepcopy(opt_state))
                    direction = control_direction(update_unit, branch, seed, preserve_support=True)
                    branch_rows = run_pilot_branch(
                        branch=branch,
                        model=branch_model,
                        optimizer=branch_opt,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        input_dim=input_dim,
                        device=device,
                        update_unit=direction,
                        horizons=horizons,
                        trust_ratio=float(args.pilot_trust_ratio),
                    )
                    for row in branch_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "module_id": "T1",
                                "model_family": "MLP",
                                "model_name": "MLP_ADAMW_PLUS_EFFECT_SPACE_AB_RESIDUAL",
                                "branch_note": branch_note,
                                "is_control_branch": is_control,
                                "pretrain_steps": int(args.pilot_pretrain_steps),
                                "train_size": int(args.pilot_train_size),
                                "test_size": int(args.pilot_test_size),
                                "batch_size": int(args.pilot_batch_size),
                                "hidden_dim": int(args.pilot_hidden),
                                "trust_ratio": float(args.pilot_trust_ratio),
                                "checkpoint_test_NLL": checkpoint_eval["final_test_NLL"],
                                "checkpoint_test_accuracy": checkpoint_eval["final_test_accuracy"],
                                "matched_control_support": "final_readout_w2_only",
                                **signal_metrics,
                            }
                        )
                        all_rows.append(row)
            except Exception as exc:
                deferred.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "deferred",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "latest_status_timestamp": now_sg(),
                    }
                )
    enriched = enrich_branch_rows(all_rows)
    write_rows(OUT_ROOT / "v22_29_effect_space_pilot_matrix.csv", enriched)
    write_rows(OUT_ROOT / "v22_29_effect_space_controls_matrix.csv", [r for r in enriched if int_flag(r.get("is_control_branch")) == 1])
    write_rows(OUT_ROOT / "v22_29_effect_space_deferred_items.csv", deferred)
    h100_h200 = [r for r in enriched if r.get("branch") == "B1_REAL_EFFECT_AB" and int(r.get("horizon") or 0) in {100, 200}]
    real_beats_base = sum(int_flag(r.get("beats_base_NLL")) for r in h100_h200)
    real_beats_controls = sum(int_flag(r.get("beats_best_control_NLL")) for r in h100_h200)
    eligible = len(h100_h200)
    pass_fraction = real_beats_controls / eligible if eligible else 0.0
    base_fraction = real_beats_base / eligible if eligible else 0.0
    real_pass = eligible > 0 and pass_fraction >= 0.60 and base_fraction >= 0.60
    controls_better_rows = sum(
        1
        for r in h100_h200
        if finite_float(r.get("final_test_NLL")) is not None
        and finite_float(r.get("control_best_NLL")) is not None
        and float(r["control_best_NLL"]) < float(r["final_test_NLL"]) - 1.0e-9
    )
    final_route = "R4-EffectSpaceBranchOpened_FullLoopPending" if real_pass else (
        "R3-ControlsExplainAllMechanisms" if controls_better_rows > 0 else "R2-EffectSpaceNoMechanismPass"
    )
    summary = {
        "effect_space_launched": int(bool(enriched)),
        "effect_space_rows": len(enriched),
        "effect_space_deferred_rows": len(deferred),
        "effect_h100_h200_real_rows": eligible,
        "effect_h100_h200_real_beats_base_rows": real_beats_base,
        "effect_h100_h200_real_beats_best_control_rows": real_beats_controls,
        "effect_h100_h200_real_pass_fraction": pass_fraction,
        "effect_controls_better_real_rows": controls_better_rows,
        "effect_space_branch_pass": int(real_pass),
        "route_after_effect_space": final_route,
        "latest_status_timestamp": now_sg(),
    }
    route_summary.update(summary)
    write_json(OUT_ROOT / "v22_29_effect_space_pilot_summary.json", summary)
    append_exec(
        command,
        task_id="v22_29_effect_space_pilot",
        status="pass" if enriched else "blocked",
        gpu=str(device),
        files=(
            "results/v22_29/v22_29_effect_space_pilot_matrix.csv, "
            "results/v22_29/v22_29_effect_space_controls_matrix.csv, "
            "results/v22_29/v22_29_effect_space_pilot_summary.json"
        ),
        note=(
            f"rows={len(enriched)} deferred={len(deferred)} route_after_effect_space={final_route}; "
            f"h100_h200_real_beats_base={real_beats_base}/{eligible}; "
            f"h100_h200_real_beats_best_control={real_beats_controls}/{eligible}; "
            f"same_support_controls=1"
        ),
        exit_code=0 if enriched else 1,
    )
    return summary


def run_tangent_normalized_pilot(args: argparse.Namespace, route_summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_tangent_normalized_pilot_start",
        status="started",
        gpu=args.pilot_device,
        note=(
            "T4 tangent-normalized signal FU repair; each branch direction is projected into current parameter block tangent space "
            "after the base AdamW step and then normalized before residual application; "
            f"datasets={args.pilot_datasets}; horizons={args.pilot_horizons}; trust={args.pilot_trust_ratio}"
        ),
    )
    device = pilot_device(args.pilot_device)
    datasets = split_csv(args.pilot_datasets)
    horizons = [int(x) for x in split_csv(args.pilot_horizons)]
    branches = [
        ("B0_BASE", 0, "base optimizer continuation"),
        ("B1_REAL_TANGENT_SIGNAL", 0, "real cohort signal direction with block-global tangent normalization"),
        ("B2_RANDOM_CONTROL", 1, "random tangent-normalized same-norm/cadence residual"),
        ("B3_STABLE_RANDOM_CONTROL", 1, "stable random tangent-normalized same-norm/cadence residual"),
        ("B4_SIGNFLIP_CONTROL", 1, "signflip tangent-normalized same-norm/cadence residual"),
        ("B5_SHUFFLED_CONTROL", 1, "shuffled tangent-normalized same-norm/cadence residual"),
        ("B6_SAME_OVERHEAD_NOOP", 1, "same-overhead no residual"),
    ]
    all_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    kan_root = Path(args.kanbefair_root)
    if not kan_root.is_absolute():
        kan_root = ROOT / kan_root
    data_root = Path(args.pilot_data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for dataset in datasets:
        for seed in [int(x) for x in split_csv(args.pilot_seeds)]:
            try:
                torch.manual_seed(int(seed) + 29300)
                train_loader, test_loader, num_classes, input_dim = load_kanbefair_loaders(
                    kan_root,
                    data_root,
                    dataset,
                    int(args.pilot_batch_size),
                    int(args.pilot_train_size),
                    int(args.pilot_test_size),
                    seed,
                )
                base_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                train_iter = iter(train_loader)
                for _step in range(int(args.pilot_pretrain_steps)):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb = next(train_iter)
                    x = flatten_batch(xb, input_dim, device)
                    y = yb.to(device).long()
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(base_model(x), y)
                    loss.backward()
                    opt.step()
                checkpoint_eval = evaluate_pilot_model(base_model, test_loader, input_dim, device)
                update_unit, signal_metrics = compute_pilot_signal_metrics(
                    base_model,
                    train_loader,
                    input_dim,
                    device,
                    int(args.pilot_cohorts),
                    seed,
                )
                model_state = copy.deepcopy(base_model.state_dict())
                opt_state = copy.deepcopy(opt.state_dict())
                for branch, is_control, branch_note in branches:
                    branch_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                    branch_model.load_state_dict(copy.deepcopy(model_state))
                    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                    branch_opt.load_state_dict(copy.deepcopy(opt_state))
                    raw_direction = control_direction(update_unit, branch, seed)
                    branch_rows = run_tangent_normalized_branch(
                        branch=branch,
                        model=branch_model,
                        optimizer=branch_opt,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        input_dim=input_dim,
                        device=device,
                        raw_update_unit=raw_direction,
                        horizons=horizons,
                        trust_ratio=float(args.pilot_trust_ratio),
                    )
                    for row in branch_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "module_id": "T4",
                                "model_family": "MLP",
                                "model_name": "MLP_ADAMW_PLUS_TANGENT_NORMALIZED_SIGNAL_RESIDUAL",
                                "branch_note": branch_note,
                                "is_control_branch": is_control,
                                "pretrain_steps": int(args.pilot_pretrain_steps),
                                "train_size": int(args.pilot_train_size),
                                "test_size": int(args.pilot_test_size),
                                "batch_size": int(args.pilot_batch_size),
                                "hidden_dim": int(args.pilot_hidden),
                                "trust_ratio": float(args.pilot_trust_ratio),
                                "checkpoint_test_NLL": checkpoint_eval["final_test_NLL"],
                                "checkpoint_test_accuracy": checkpoint_eval["final_test_accuracy"],
                                "matched_control_geometry": "same_tangent_projection_norm_cadence",
                                **signal_metrics,
                            }
                        )
                        all_rows.append(row)
            except Exception as exc:
                deferred.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "deferred",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "latest_status_timestamp": now_sg(),
                    }
                )
    enriched = enrich_branch_rows(all_rows)
    write_rows(OUT_ROOT / "v22_29_tangent_normalized_pilot_matrix.csv", enriched)
    write_rows(OUT_ROOT / "v22_29_tangent_normalized_controls_matrix.csv", [r for r in enriched if int_flag(r.get("is_control_branch")) == 1])
    write_rows(OUT_ROOT / "v22_29_tangent_normalized_deferred_items.csv", deferred)
    h100_h200 = [r for r in enriched if r.get("branch") == "B1_REAL_TANGENT_SIGNAL" and int(r.get("horizon") or 0) in {100, 200}]
    real_beats_base = sum(int_flag(r.get("beats_base_NLL")) for r in h100_h200)
    real_beats_controls = sum(int_flag(r.get("beats_best_control_NLL")) for r in h100_h200)
    eligible = len(h100_h200)
    pass_fraction = real_beats_controls / eligible if eligible else 0.0
    base_fraction = real_beats_base / eligible if eligible else 0.0
    real_pass = eligible > 0 and pass_fraction >= 0.60 and base_fraction >= 0.60
    controls_better_rows = sum(
        1
        for r in h100_h200
        if finite_float(r.get("final_test_NLL")) is not None
        and finite_float(r.get("control_best_NLL")) is not None
        and float(r["control_best_NLL"]) < float(r["final_test_NLL"]) - 1.0e-9
    )
    final_route = "R4-TangentNormalizedBranchOpened_FullLoopPending" if real_pass else (
        "R3-ControlsExplainAllMechanisms" if controls_better_rows > 0 else "R2-TangentNormalizedNoMechanismPass"
    )
    summary = {
        "tangent_normalized_launched": int(bool(enriched)),
        "tangent_normalized_rows": len(enriched),
        "tangent_normalized_deferred_rows": len(deferred),
        "tangent_h100_h200_real_rows": eligible,
        "tangent_h100_h200_real_beats_base_rows": real_beats_base,
        "tangent_h100_h200_real_beats_best_control_rows": real_beats_controls,
        "tangent_h100_h200_real_pass_fraction": pass_fraction,
        "tangent_controls_better_real_rows": controls_better_rows,
        "tangent_normalized_branch_pass": int(real_pass),
        "route_after_tangent_normalized": final_route,
        "latest_status_timestamp": now_sg(),
    }
    route_summary.update(summary)
    write_json(OUT_ROOT / "v22_29_tangent_normalized_pilot_summary.json", summary)
    append_exec(
        command,
        task_id="v22_29_tangent_normalized_pilot",
        status="pass" if enriched else "blocked",
        gpu=str(device),
        files=(
            "results/v22_29/v22_29_tangent_normalized_pilot_matrix.csv, "
            "results/v22_29/v22_29_tangent_normalized_controls_matrix.csv, "
            "results/v22_29/v22_29_tangent_normalized_pilot_summary.json"
        ),
        note=(
            f"rows={len(enriched)} deferred={len(deferred)} route_after_tangent_normalized={final_route}; "
            f"h100_h200_real_beats_base={real_beats_base}/{eligible}; "
            f"h100_h200_real_beats_best_control={real_beats_controls}/{eligible}"
        ),
        exit_code=0 if enriched else 1,
    )
    return summary


def run_online_subspace_pilot(args: argparse.Namespace, route_summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_online_subspace_pilot_start",
        status="started",
        gpu=args.pilot_device,
        note=(
            "T5 online gradient subspace tracking repair; Oja/QR tracked subspace is shared by real and controls; "
            f"rank={args.pilot_subspace_rank}; oja_lr={args.pilot_oja_lr}; datasets={args.pilot_datasets}; horizons={args.pilot_horizons}"
        ),
    )
    device = pilot_device(args.pilot_device)
    datasets = split_csv(args.pilot_datasets)
    horizons = [int(x) for x in split_csv(args.pilot_horizons)]
    branches = [
        ("B0_BASE", 0, "base optimizer continuation with subspace tracking overhead"),
        ("B1_REAL_ONLINE_SUBSPACE", 0, "real projected gradient inside online tracked subspace"),
        ("B2_RANDOM_CONTROL", 1, "random direction inside same online tracked subspace"),
        ("B3_STABLE_RANDOM_CONTROL", 1, "stable-random direction inside same online tracked subspace"),
        ("B4_SIGNFLIP_CONTROL", 1, "signflip projected-gradient direction inside same online tracked subspace"),
        ("B5_SHUFFLED_CONTROL", 1, "shuffled projected-gradient coefficients inside same online tracked subspace"),
        ("B6_SAME_OVERHEAD_NOOP", 1, "same-overhead tracked subspace no residual"),
    ]
    all_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    kan_root = Path(args.kanbefair_root)
    if not kan_root.is_absolute():
        kan_root = ROOT / kan_root
    data_root = Path(args.pilot_data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for dataset in datasets:
        for seed in [int(x) for x in split_csv(args.pilot_seeds)]:
            try:
                torch.manual_seed(int(seed) + 29400)
                train_loader, test_loader, num_classes, input_dim = load_kanbefair_loaders(
                    kan_root,
                    data_root,
                    dataset,
                    int(args.pilot_batch_size),
                    int(args.pilot_train_size),
                    int(args.pilot_test_size),
                    seed,
                )
                base_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                train_iter = iter(train_loader)
                for _step in range(int(args.pilot_pretrain_steps)):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb = next(train_iter)
                    x = flatten_batch(xb, input_dim, device)
                    y = yb.to(device).long()
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(base_model(x), y)
                    loss.backward()
                    opt.step()
                checkpoint_eval = evaluate_pilot_model(base_model, test_loader, input_dim, device)
                init_subspace, subspace_metrics = initialize_gradient_subspace(
                    base_model,
                    train_loader,
                    input_dim,
                    device,
                    int(args.pilot_subspace_rank),
                    int(args.pilot_cohorts),
                    seed,
                )
                model_state = copy.deepcopy(base_model.state_dict())
                opt_state = copy.deepcopy(opt.state_dict())
                for branch, is_control, branch_note in branches:
                    branch_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                    branch_model.load_state_dict(copy.deepcopy(model_state))
                    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                    branch_opt.load_state_dict(copy.deepcopy(opt_state))
                    branch_rows = run_online_subspace_branch(
                        branch=branch,
                        model=branch_model,
                        optimizer=branch_opt,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        input_dim=input_dim,
                        device=device,
                        init_subspace=init_subspace,
                        horizons=horizons,
                        trust_ratio=float(args.pilot_trust_ratio),
                        oja_lr=float(args.pilot_oja_lr),
                        seed=seed,
                    )
                    for row in branch_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "module_id": "T5",
                                "model_family": "MLP",
                                "model_name": "MLP_ADAMW_PLUS_ONLINE_SUBSPACE_RESIDUAL",
                                "branch_note": branch_note,
                                "is_control_branch": is_control,
                                "pretrain_steps": int(args.pilot_pretrain_steps),
                                "train_size": int(args.pilot_train_size),
                                "test_size": int(args.pilot_test_size),
                                "batch_size": int(args.pilot_batch_size),
                                "hidden_dim": int(args.pilot_hidden),
                                "trust_ratio": float(args.pilot_trust_ratio),
                                "checkpoint_test_NLL": checkpoint_eval["final_test_NLL"],
                                "checkpoint_test_accuracy": checkpoint_eval["final_test_accuracy"],
                                "matched_control_geometry": "same_online_tracked_gradient_subspace",
                                **subspace_metrics,
                            }
                        )
                        all_rows.append(row)
            except Exception as exc:
                deferred.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "deferred",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "latest_status_timestamp": now_sg(),
                    }
                )
    enriched = enrich_branch_rows(all_rows)
    write_rows(OUT_ROOT / "v22_29_online_subspace_pilot_matrix.csv", enriched)
    write_rows(OUT_ROOT / "v22_29_online_subspace_controls_matrix.csv", [r for r in enriched if int_flag(r.get("is_control_branch")) == 1])
    write_rows(OUT_ROOT / "v22_29_online_subspace_deferred_items.csv", deferred)
    h100_h200 = [r for r in enriched if r.get("branch") == "B1_REAL_ONLINE_SUBSPACE" and int(r.get("horizon") or 0) in {100, 200}]
    real_beats_base = sum(int_flag(r.get("beats_base_NLL")) for r in h100_h200)
    real_beats_controls = sum(int_flag(r.get("beats_best_control_NLL")) for r in h100_h200)
    eligible = len(h100_h200)
    pass_fraction = real_beats_controls / eligible if eligible else 0.0
    base_fraction = real_beats_base / eligible if eligible else 0.0
    real_pass = eligible > 0 and pass_fraction >= 0.60 and base_fraction >= 0.60
    controls_better_rows = sum(
        1
        for r in h100_h200
        if finite_float(r.get("final_test_NLL")) is not None
        and finite_float(r.get("control_best_NLL")) is not None
        and float(r["control_best_NLL"]) < float(r["final_test_NLL"]) - 1.0e-9
    )
    final_route = "R4-OnlineSubspaceBranchOpened_FullLoopPending" if real_pass else (
        "R3-ControlsExplainAllMechanisms" if controls_better_rows > 0 else "R2-OnlineSubspaceNoMechanismPass"
    )
    summary = {
        "online_subspace_launched": int(bool(enriched)),
        "online_subspace_rows": len(enriched),
        "online_subspace_deferred_rows": len(deferred),
        "online_h100_h200_real_rows": eligible,
        "online_h100_h200_real_beats_base_rows": real_beats_base,
        "online_h100_h200_real_beats_best_control_rows": real_beats_controls,
        "online_h100_h200_real_pass_fraction": pass_fraction,
        "online_controls_better_real_rows": controls_better_rows,
        "online_subspace_branch_pass": int(real_pass),
        "route_after_online_subspace": final_route,
        "latest_status_timestamp": now_sg(),
    }
    route_summary.update(summary)
    write_json(OUT_ROOT / "v22_29_online_subspace_pilot_summary.json", summary)
    append_exec(
        command,
        task_id="v22_29_online_subspace_pilot",
        status="pass" if enriched else "blocked",
        gpu=str(device),
        files=(
            "results/v22_29/v22_29_online_subspace_pilot_matrix.csv, "
            "results/v22_29/v22_29_online_subspace_controls_matrix.csv, "
            "results/v22_29/v22_29_online_subspace_pilot_summary.json"
        ),
        note=(
            f"rows={len(enriched)} deferred={len(deferred)} route_after_online_subspace={final_route}; "
            f"h100_h200_real_beats_base={real_beats_base}/{eligible}; "
            f"h100_h200_real_beats_best_control={real_beats_controls}/{eligible}"
        ),
        exit_code=0 if enriched else 1,
    )
    return summary


def run_online_effect_subspace_pilot(args: argparse.Namespace, route_summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_online_effect_subspace_pilot_start",
        status="started",
        gpu=args.pilot_device,
        note=(
            "T5 effect-space online subspace tracking repair; final readout per-example effect subspace is tracked with Oja/QR, "
            "real and controls are restricted to the same tracked effect subspace; "
            f"rank={args.pilot_subspace_rank}; oja_lr={args.pilot_oja_lr}; datasets={args.pilot_datasets}; horizons={args.pilot_horizons}"
        ),
    )
    device = pilot_device(args.pilot_device)
    datasets = split_csv(args.pilot_datasets)
    horizons = [int(x) for x in split_csv(args.pilot_horizons)]
    branches = [
        ("B0_BASE", 0, "base optimizer continuation with effect-subspace tracking overhead"),
        ("B1_REAL_ONLINE_EFFECT_SUBSPACE", 0, "real projected readout effect gradient inside online tracked effect subspace"),
        ("B2_RANDOM_CONTROL", 1, "random direction inside same online tracked effect subspace"),
        ("B3_STABLE_RANDOM_CONTROL", 1, "stable-random direction inside same online tracked effect subspace"),
        ("B4_SIGNFLIP_CONTROL", 1, "signflip projected effect direction inside same online tracked effect subspace"),
        ("B5_SHUFFLED_CONTROL", 1, "shuffled projected effect coefficients inside same online tracked effect subspace"),
        ("B6_SAME_OVERHEAD_NOOP", 1, "same-overhead tracked effect subspace no residual"),
    ]
    all_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    kan_root = Path(args.kanbefair_root)
    if not kan_root.is_absolute():
        kan_root = ROOT / kan_root
    data_root = Path(args.pilot_data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for dataset in datasets:
        for seed in [int(x) for x in split_csv(args.pilot_seeds)]:
            try:
                torch.manual_seed(int(seed) + 29600)
                train_loader, test_loader, num_classes, input_dim = load_kanbefair_loaders(
                    kan_root,
                    data_root,
                    dataset,
                    int(args.pilot_batch_size),
                    int(args.pilot_train_size),
                    int(args.pilot_test_size),
                    seed,
                )
                base_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                train_iter = iter(train_loader)
                for _step in range(int(args.pilot_pretrain_steps)):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb = next(train_iter)
                    x = flatten_batch(xb, input_dim, device)
                    y = yb.to(device).long()
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(base_model(x), y)
                    loss.backward()
                    opt.step()
                checkpoint_eval = evaluate_pilot_model(base_model, test_loader, input_dim, device)
                init_subspace, effect_metrics = initialize_effect_subspace(
                    base_model,
                    train_loader,
                    input_dim,
                    device,
                    int(args.pilot_subspace_rank),
                    int(args.pilot_cohorts),
                    seed,
                )
                model_state = copy.deepcopy(base_model.state_dict())
                opt_state = copy.deepcopy(opt.state_dict())
                for branch, is_control, branch_note in branches:
                    branch_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                    branch_model.load_state_dict(copy.deepcopy(model_state))
                    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                    branch_opt.load_state_dict(copy.deepcopy(opt_state))
                    branch_rows = run_online_effect_subspace_branch(
                        branch=branch,
                        model=branch_model,
                        optimizer=branch_opt,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        input_dim=input_dim,
                        device=device,
                        init_subspace=init_subspace,
                        horizons=horizons,
                        trust_ratio=float(args.pilot_trust_ratio),
                        oja_lr=float(args.pilot_oja_lr),
                        seed=seed,
                    )
                    for row in branch_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "module_id": "T5_effect",
                                "model_family": "MLP",
                                "model_name": "MLP_ADAMW_PLUS_ONLINE_EFFECT_SUBSPACE_RESIDUAL",
                                "branch_note": branch_note,
                                "is_control_branch": is_control,
                                "pretrain_steps": int(args.pilot_pretrain_steps),
                                "train_size": int(args.pilot_train_size),
                                "test_size": int(args.pilot_test_size),
                                "batch_size": int(args.pilot_batch_size),
                                "hidden_dim": int(args.pilot_hidden),
                                "trust_ratio": float(args.pilot_trust_ratio),
                                "checkpoint_test_NLL": checkpoint_eval["final_test_NLL"],
                                "checkpoint_test_accuracy": checkpoint_eval["final_test_accuracy"],
                                "matched_control_geometry": "same_online_tracked_readout_effect_subspace",
                                **effect_metrics,
                            }
                        )
                        all_rows.append(row)
            except Exception as exc:
                deferred.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "deferred",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "latest_status_timestamp": now_sg(),
                    }
                )
    enriched = enrich_branch_rows(all_rows)
    write_rows(OUT_ROOT / "v22_29_online_effect_subspace_pilot_matrix.csv", enriched)
    write_rows(OUT_ROOT / "v22_29_online_effect_subspace_controls_matrix.csv", [r for r in enriched if int_flag(r.get("is_control_branch")) == 1])
    write_rows(OUT_ROOT / "v22_29_online_effect_subspace_deferred_items.csv", deferred)
    h100_h200 = [r for r in enriched if r.get("branch") == "B1_REAL_ONLINE_EFFECT_SUBSPACE" and int(r.get("horizon") or 0) in {100, 200}]
    real_beats_base = sum(int_flag(r.get("beats_base_NLL")) for r in h100_h200)
    real_beats_controls = sum(int_flag(r.get("beats_best_control_NLL")) for r in h100_h200)
    eligible = len(h100_h200)
    pass_fraction = real_beats_controls / eligible if eligible else 0.0
    base_fraction = real_beats_base / eligible if eligible else 0.0
    real_pass = eligible > 0 and pass_fraction >= 0.60 and base_fraction >= 0.60
    controls_better_rows = sum(
        1
        for r in h100_h200
        if finite_float(r.get("final_test_NLL")) is not None
        and finite_float(r.get("control_best_NLL")) is not None
        and float(r["control_best_NLL"]) < float(r["final_test_NLL"]) - 1.0e-9
    )
    final_route = "R4-OnlineEffectSubspaceBranchOpened_FullLoopPending" if real_pass else (
        "R3-ControlsExplainAllMechanisms" if controls_better_rows > 0 else "R2-OnlineEffectSubspaceNoMechanismPass"
    )
    summary = {
        "online_effect_subspace_launched": int(bool(enriched)),
        "online_effect_subspace_rows": len(enriched),
        "online_effect_subspace_deferred_rows": len(deferred),
        "online_effect_h100_h200_real_rows": eligible,
        "online_effect_h100_h200_real_beats_base_rows": real_beats_base,
        "online_effect_h100_h200_real_beats_best_control_rows": real_beats_controls,
        "online_effect_h100_h200_real_pass_fraction": pass_fraction,
        "online_effect_controls_better_real_rows": controls_better_rows,
        "online_effect_subspace_branch_pass": int(real_pass),
        "route_after_online_effect_subspace": final_route,
        "latest_status_timestamp": now_sg(),
    }
    route_summary.update(summary)
    write_json(OUT_ROOT / "v22_29_online_effect_subspace_pilot_summary.json", summary)
    append_exec(
        command,
        task_id="v22_29_online_effect_subspace_pilot",
        status="pass" if enriched else "blocked",
        gpu=str(device),
        files=(
            "results/v22_29/v22_29_online_effect_subspace_pilot_matrix.csv, "
            "results/v22_29/v22_29_online_effect_subspace_controls_matrix.csv, "
            "results/v22_29/v22_29_online_effect_subspace_pilot_summary.json"
        ),
        note=(
            f"rows={len(enriched)} deferred={len(deferred)} route_after_online_effect_subspace={final_route}; "
            f"h100_h200_real_beats_base={real_beats_base}/{eligible}; "
            f"h100_h200_real_beats_best_control={real_beats_controls}/{eligible}"
        ),
        exit_code=0 if enriched else 1,
    )
    return summary


def run_temporal_refresh_pilot(args: argparse.Namespace, route_summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_temporal_refresh_pilot_start",
        status="started",
        gpu=args.pilot_device,
        note=(
            "T6 temporal slow-gradient refresh repair after static branch reversal/control-equivalence; "
            f"datasets={args.pilot_datasets}; horizons={args.pilot_horizons}; beta={args.temporal_slow_beta}"
        ),
    )
    device = pilot_device(args.pilot_device)
    datasets = split_csv(args.pilot_datasets)
    horizons = [int(x) for x in split_csv(args.pilot_horizons)]
    branches = [
        ("B0_BASE", 0, "base optimizer continuation"),
        ("B1_REAL_SLOW_SIGNAL_EMA", 0, "real online slow-gradient EMA residual"),
        ("B2_RANDOM_CONTROL", 1, "random same-norm/cadence residual"),
        ("B3_STABLE_RANDOM_CONTROL", 1, "stable random same-norm/cadence residual"),
        ("B4_SIGNFLIP_CONTROL", 1, "signflip same-norm/cadence residual"),
        ("B5_SHUFFLED_CONTROL", 1, "shuffled slow-gradient same-norm/cadence residual"),
        ("B6_SAME_OVERHEAD_NOOP", 1, "same-overhead no residual"),
    ]
    all_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    kan_root = Path(args.kanbefair_root)
    if not kan_root.is_absolute():
        kan_root = ROOT / kan_root
    data_root = Path(args.pilot_data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    for dataset in datasets:
        for seed in [int(x) for x in split_csv(args.pilot_seeds)]:
            try:
                torch.manual_seed(int(seed) + 29100)
                train_loader, test_loader, num_classes, input_dim = load_kanbefair_loaders(
                    kan_root,
                    data_root,
                    dataset,
                    int(args.pilot_batch_size),
                    int(args.pilot_train_size),
                    int(args.pilot_test_size),
                    seed,
                )
                base_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                train_iter = iter(train_loader)
                for _step in range(int(args.pilot_pretrain_steps)):
                    try:
                        xb, yb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb = next(train_iter)
                    x = flatten_batch(xb, input_dim, device)
                    y = yb.to(device).long()
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(base_model(x), y)
                    loss.backward()
                    opt.step()
                checkpoint_eval = evaluate_pilot_model(base_model, test_loader, input_dim, device)
                model_state = copy.deepcopy(base_model.state_dict())
                opt_state = copy.deepcopy(opt.state_dict())
                for branch, is_control, branch_note in branches:
                    branch_model = MLPBaseline(input_dim, num_classes, int(args.pilot_hidden), seed, device).to(device)
                    branch_model.load_state_dict(copy.deepcopy(model_state))
                    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=float(args.pilot_lr), weight_decay=float(args.pilot_weight_decay))
                    branch_opt.load_state_dict(copy.deepcopy(opt_state))
                    branch_rows = run_temporal_refresh_branch(
                        branch=branch,
                        model=branch_model,
                        optimizer=branch_opt,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        input_dim=input_dim,
                        device=device,
                        horizons=horizons,
                        trust_ratio=float(args.pilot_trust_ratio),
                        slow_beta=float(args.temporal_slow_beta),
                        seed=seed,
                    )
                    for row in branch_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "module_id": "T6",
                                "model_family": "MLP",
                                "model_name": "MLP_ADAMW_PLUS_TEMPORAL_REFRESH_RESIDUAL",
                                "branch_note": branch_note,
                                "is_control_branch": is_control,
                                "pretrain_steps": int(args.pilot_pretrain_steps),
                                "train_size": int(args.pilot_train_size),
                                "test_size": int(args.pilot_test_size),
                                "batch_size": int(args.pilot_batch_size),
                                "hidden_dim": int(args.pilot_hidden),
                                "trust_ratio": float(args.pilot_trust_ratio),
                                "checkpoint_test_NLL": checkpoint_eval["final_test_NLL"],
                                "checkpoint_test_accuracy": checkpoint_eval["final_test_accuracy"],
                            }
                        )
                        all_rows.append(row)
            except Exception as exc:
                deferred.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "deferred",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "latest_status_timestamp": now_sg(),
                    }
                )
    enriched = enrich_branch_rows(all_rows)
    write_rows(OUT_ROOT / "v22_29_temporal_refresh_pilot_matrix.csv", enriched)
    write_rows(OUT_ROOT / "v22_29_temporal_refresh_controls_matrix.csv", [r for r in enriched if int_flag(r.get("is_control_branch")) == 1])
    write_rows(OUT_ROOT / "v22_29_temporal_refresh_deferred_items.csv", deferred)
    h100_h200 = [r for r in enriched if r.get("branch") == "B1_REAL_SLOW_SIGNAL_EMA" and int(r.get("horizon") or 0) in {100, 200}]
    real_beats_base = sum(int_flag(r.get("beats_base_NLL")) for r in h100_h200)
    real_beats_controls = sum(int_flag(r.get("beats_best_control_NLL")) for r in h100_h200)
    eligible = len(h100_h200)
    pass_fraction = real_beats_controls / eligible if eligible else 0.0
    real_pass = eligible > 0 and pass_fraction >= 0.60 and real_beats_base / eligible >= 0.60
    controls_better_rows = sum(
        1
        for r in h100_h200
        if finite_float(r.get("final_test_NLL")) is not None
        and finite_float(r.get("control_best_NLL")) is not None
        and float(r["control_best_NLL"]) < float(r["final_test_NLL"]) - 1.0e-9
    )
    final_route = "R4-TemporalBranchOpened_FullLoopPending" if real_pass else (
        "R3-ControlsExplainAllMechanisms" if controls_better_rows > 0 else "R2-TemporalRefreshNoMechanismPass"
    )
    summary = {
        "temporal_refresh_launched": int(bool(enriched)),
        "temporal_refresh_rows": len(enriched),
        "temporal_refresh_deferred_rows": len(deferred),
        "temporal_h100_h200_real_rows": eligible,
        "temporal_h100_h200_real_beats_base_rows": real_beats_base,
        "temporal_h100_h200_real_beats_best_control_rows": real_beats_controls,
        "temporal_h100_h200_real_pass_fraction": pass_fraction,
        "temporal_controls_better_real_rows": controls_better_rows,
        "temporal_branch_pass": int(real_pass),
        "route_after_temporal_refresh": final_route,
        "latest_status_timestamp": now_sg(),
    }
    route_summary.update(summary)
    write_json(OUT_ROOT / "v22_29_temporal_refresh_pilot_summary.json", summary)
    append_exec(
        command,
        task_id="v22_29_temporal_refresh_pilot",
        status="pass" if enriched else "blocked",
        gpu=str(device),
        files=(
            "results/v22_29/v22_29_temporal_refresh_pilot_matrix.csv, "
            "results/v22_29/v22_29_temporal_refresh_controls_matrix.csv, "
            "results/v22_29/v22_29_temporal_refresh_pilot_summary.json"
        ),
        note=(
            f"rows={len(enriched)} deferred={len(deferred)} route_after_temporal_refresh={final_route}; "
            f"h100_h200_real_beats_base={real_beats_base}/{eligible}; "
            f"h100_h200_real_beats_best_control={real_beats_controls}/{eligible}"
        ),
        exit_code=0 if enriched else 1,
    )
    return summary


def write_figures(eval_rows: list[dict[str, Any]], grassmann_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]], continual_rows: list[dict[str, Any]]) -> None:
    branch_rows = read_rows(OUT_ROOT / "v22_29_branch_causal_pilot_matrix.csv")
    effect_rows = read_rows(OUT_ROOT / "v22_29_effect_space_pilot_matrix.csv")
    tangent_rows = read_rows(OUT_ROOT / "v22_29_tangent_normalized_pilot_matrix.csv")
    online_rows = read_rows(OUT_ROOT / "v22_29_online_subspace_pilot_matrix.csv")
    write_simple_svg("v22_29_signal_reservoir_spectrum.svg", "Signal-channel historical proxy rows", [r for r in eval_rows if r.get("module_id") == "T1"], "AUC_oriented")
    write_simple_svg("v22_29_grassmann_principal_angles.svg", "Grassmann principal angles", grassmann_rows, "principal_angles_mean")
    write_simple_svg("v22_29_subspace_transport_error.svg", "Subspace transport error", grassmann_rows, "Grassmann_geodesic_distance")
    write_simple_svg("v22_29_update_spectrum_panel.svg", "Update spectrum proxy rows", [r for r in eval_rows if r.get("module_id") == "T4"], "AUC_oriented")
    write_simple_svg("v22_29_optimizer_interaction_heatmap.svg", "Optimizer interaction unavailable after Gate 1", [], "rows")
    write_simple_svg("v22_29_kan_basis_signal_overlap.svg", "KAN basis signal overlap proxy", [r for r in eval_rows if r.get("metric_name") == "basis_channel_energy_fraction"], "AUC_oriented")
    write_simple_svg("v22_29_kanbefair_gap_reduction.svg", "KANbeFair gap reduction", gap_rows, "GapReduction")
    write_simple_svg("v22_29_grokking_delay_curves.svg", "Grokking skipped by Gate 1", [], "rows")
    write_simple_svg("v22_29_continual_forgetting_curves.svg", "Continual forgetting rows", continual_rows, "relative_forgetting_reduction")
    write_simple_svg(
        "v22_29_branch_causal_forest_plot.svg",
        "Branch-causal pilot paired NLL delta" if branch_rows and branch_rows[0].get("status") != "skipped" else "Branch-causal pilot skipped by Gate 1",
        branch_rows,
        "paired_NLL_delta_H",
    )
    write_simple_svg(
        "v22_29_effect_space_branch_forest_plot.svg",
        "Effect-space A_B pilot paired NLL delta" if effect_rows else "Effect-space A_B pilot not launched",
        effect_rows,
        "paired_NLL_delta_H",
    )
    write_simple_svg(
        "v22_29_tangent_normalized_branch_forest_plot.svg",
        "Tangent-normalized pilot paired NLL delta" if tangent_rows else "Tangent-normalized pilot not launched",
        tangent_rows,
        "paired_NLL_delta_H",
    )
    write_simple_svg(
        "v22_29_online_subspace_branch_forest_plot.svg",
        "Online subspace pilot paired NLL delta" if online_rows else "Online subspace pilot not launched",
        online_rows,
        "paired_NLL_delta_H",
    )


def final_route_from_eval(eval_rows: list[dict[str, Any]], grassmann_rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    hard_pass_modules = sorted({str(r.get("module_id")) for r in eval_rows if int(r.get("hard_gate_pass") or 0) == 1})
    proxy_pass_modules = sorted({str(r.get("module_id")) for r in eval_rows if int(r.get("proxy_diagnostic_pass") or 0) == 1})
    direct_grassmann = [r for r in grassmann_rows if int_flag(r.get("direct_v22_29_metric")) == 1]
    outcome_linked_direct = [r for r in grassmann_rows if int_flag(r.get("hard_gate_eligible")) == 1]
    module_status: list[dict[str, Any]] = []
    for module_id, name in THEORY_MODULES:
        hard = int(module_id in hard_pass_modules)
        proxy = int(module_id in proxy_pass_modules)
        if hard:
            status = "open_to_branch_pilot"
            blocker = ""
        elif module_id == "T2" and direct_grassmann and not outcome_linked_direct:
            status = "blocked_at_gate1"
            blocker = "Grassmann source vectors reconstructed, but exact paired task outcomes are unavailable for those trajectories."
        elif proxy:
            status = "diagnostic_proxy_only"
            blocker = "Proxy predictive signal is not hard-gate eligible under v22.29 no-proxy-promotion rule."
        else:
            status = "blocked_at_gate1"
            blocker = "No hard-gate eligible historical metric passed AUC/Spearman/control criteria."
        module_status.append(
            {
                "module_id": module_id,
                "module_name": name,
                "gate1_status": status,
                "hard_gate_pass": hard,
                "proxy_diagnostic_pass": proxy,
                "blocker": blocker,
            }
        )
    route = "R4-MLPGeneralFUOpened_KANPending" if hard_pass_modules else "R1-HistoricalMetricsNoSignal"
    summary = {
        "hard_pass_modules": hard_pass_modules,
        "proxy_pass_modules": proxy_pass_modules,
        "direct_grassmann_rows": len(direct_grassmann),
        "outcome_linked_direct_grassmann_rows": len(outcome_linked_direct),
        "gate_open": bool(hard_pass_modules),
    }
    return route, module_status, summary


def maybe_promote_strong_no_go(route: str, route_summary: dict[str, Any]) -> str:
    """Escalate the final route only after the implemented module matrix is exhausted."""
    pilot_keys = [
        ("continuation_pilot_launched", "continuation_branch_pass", "continuation_controls_better_real_rows"),
        ("transported_momentum_launched", "transported_momentum_branch_pass", "transported_controls_better_real_rows"),
        ("temporal_refresh_launched", "temporal_branch_pass", "temporal_controls_better_real_rows"),
        ("effect_space_launched", "effect_space_branch_pass", "effect_controls_better_real_rows"),
        ("tangent_normalized_launched", "tangent_normalized_branch_pass", "tangent_controls_better_real_rows"),
        ("online_subspace_launched", "online_subspace_branch_pass", "online_controls_better_real_rows"),
        ("online_effect_subspace_launched", "online_effect_subspace_branch_pass", "online_effect_controls_better_real_rows"),
    ]
    launched = [int(route_summary.get(launch_key, 0) or 0) == 1 for launch_key, _, _ in pilot_keys]
    passed = [int(route_summary.get(pass_key, 0) or 0) == 1 for _, pass_key, _ in pilot_keys]
    control_counts = [int(route_summary.get(control_key, 0) or 0) for _, _, control_key in pilot_keys]
    exhausted_controls = all(launched) and not any(passed) and all(count > 0 for count in control_counts)
    if route == "R3-ControlsExplainAllMechanisms" and exhausted_controls:
        route_summary["strong_no_go_triggered"] = 1
        route_summary["strong_no_go_controls_better_rows_total"] = sum(control_counts)
        route_summary["route_after_strong_no_go"] = "R15-StrongNoGo_AllGeometricMechanismsCannotBeatControls"
        return "R15-StrongNoGo_AllGeometricMechanismsCannotBeatControls"
    route_summary["strong_no_go_triggered"] = 0
    return route


def write_recap(
    route: str,
    route_summary: dict[str, Any],
    task_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    grassmann_rows: list[dict[str, Any]],
    module_status: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    continual_rows: list[dict[str, Any]],
    strong_rows: list[dict[str, Any]],
) -> None:
    proxy_pass = [r for r in eval_rows if int(r.get("proxy_diagnostic_pass") or 0) == 1]
    hard_pass = [r for r in eval_rows if int(r.get("hard_gate_pass") or 0) == 1]
    top_eval = sorted(
        eval_rows,
        key=lambda r: finite_float(r.get("AUC_oriented"), -1.0) or -1.0,
        reverse=True,
    )[:12]
    grassmann_sample = [r for r in grassmann_rows if r.get("status") == "computed"][:10]
    branch_rows = read_rows(OUT_ROOT / "v22_29_branch_causal_pilot_matrix.csv")
    real_branch_rows = [r for r in branch_rows if r.get("branch") == "B1_REAL_SIGNAL_GRASSMANN"]
    branch_sample = [r for r in branch_rows if r.get("branch") in {"B0_BASE", "B1_REAL_SIGNAL_GRASSMANN", "B2_RANDOM_CONTROL", "B4_SIGNFLIP_CONTROL"}][:24]
    transported_rows = read_rows(OUT_ROOT / "v22_29_transported_momentum_pilot_matrix.csv")
    real_transported_rows = [r for r in transported_rows if r.get("branch") == "B1_REAL_TRANSPORTED_SOURCE"]
    transported_sample = [r for r in transported_rows if r.get("branch") in {"B0_BASE", "B1_REAL_TRANSPORTED_SOURCE", "B2_RANDOM_CONTROL", "B4_SIGNFLIP_CONTROL"}][:24]
    temporal_rows = read_rows(OUT_ROOT / "v22_29_temporal_refresh_pilot_matrix.csv")
    real_temporal_rows = [r for r in temporal_rows if r.get("branch") == "B1_REAL_SLOW_SIGNAL_EMA"]
    temporal_sample = [r for r in temporal_rows if r.get("branch") in {"B0_BASE", "B1_REAL_SLOW_SIGNAL_EMA", "B2_RANDOM_CONTROL", "B4_SIGNFLIP_CONTROL"}][:24]
    effect_rows = read_rows(OUT_ROOT / "v22_29_effect_space_pilot_matrix.csv")
    real_effect_rows = [r for r in effect_rows if r.get("branch") == "B1_REAL_EFFECT_AB"]
    effect_sample = [r for r in effect_rows if r.get("branch") in {"B0_BASE", "B1_REAL_EFFECT_AB", "B2_RANDOM_CONTROL", "B4_SIGNFLIP_CONTROL"}][:24]
    tangent_rows = read_rows(OUT_ROOT / "v22_29_tangent_normalized_pilot_matrix.csv")
    real_tangent_rows = [r for r in tangent_rows if r.get("branch") == "B1_REAL_TANGENT_SIGNAL"]
    tangent_sample = [r for r in tangent_rows if r.get("branch") in {"B0_BASE", "B1_REAL_TANGENT_SIGNAL", "B2_RANDOM_CONTROL", "B4_SIGNFLIP_CONTROL"}][:24]
    online_rows = read_rows(OUT_ROOT / "v22_29_online_subspace_pilot_matrix.csv")
    real_online_rows = [r for r in online_rows if r.get("branch") == "B1_REAL_ONLINE_SUBSPACE"]
    online_sample = [r for r in online_rows if r.get("branch") in {"B0_BASE", "B1_REAL_ONLINE_SUBSPACE", "B2_RANDOM_CONTROL", "B4_SIGNFLIP_CONTROL"}][:24]
    online_effect_rows = read_rows(OUT_ROOT / "v22_29_online_effect_subspace_pilot_matrix.csv")
    real_online_effect_rows = [r for r in online_effect_rows if r.get("branch") == "B1_REAL_ONLINE_EFFECT_SUBSPACE"]
    online_effect_sample = [r for r in online_effect_rows if r.get("branch") in {"B0_BASE", "B1_REAL_ONLINE_EFFECT_SUBSPACE", "B2_RANDOM_CONTROL", "B4_SIGNFLIP_CONTROL"}][:24]
    sweep_rows = read_rows(OUT_ROOT / "v22_29_repair_sweep_summary.csv")
    transported_sweep_rows = read_rows(OUT_ROOT / "v22_29_transported_momentum_sweep_summary.csv")
    multiseed_audit_rows = read_rows(OUT_ROOT / "v22_29_t4_t5_multiseed_control_audit.csv")
    static_h100_h200 = [r for r in real_branch_rows if int(r.get("horizon") or 0) in {100, 200}]
    transported_h100_h200 = [r for r in real_transported_rows if int(r.get("horizon") or 0) in {100, 200}]
    temporal_h100_h200 = [r for r in real_temporal_rows if int(r.get("horizon") or 0) in {100, 200}]
    effect_h100_h200 = [r for r in real_effect_rows if int(r.get("horizon") or 0) in {100, 200}]
    tangent_h100_h200 = [r for r in real_tangent_rows if int(r.get("horizon") or 0) in {100, 200}]
    online_h100_h200 = [r for r in real_online_rows if int(r.get("horizon") or 0) in {100, 200}]
    online_effect_h100_h200 = [r for r in real_online_effect_rows if int(r.get("horizon") or 0) in {100, 200}]

    def mean_field(rows: list[dict[str, Any]], field: str) -> str:
        vals = [finite_float(r.get(field)) for r in rows]
        vals = [v for v in vals if v is not None]
        return "" if not vals else str(statistics.fmean(vals))

    route_after_online = str(
        route_summary.get("route_after_online_effect_subspace")
        or route_summary.get("route_after_online_subspace")
        or route_summary.get("route_after_tangent_normalized")
        or route_summary.get("route_after_effect_space")
        or route_summary.get("route_after_temporal_refresh")
        or route_summary.get("route_after_transported_momentum")
        or route_summary.get("route_after_continuation")
        or route
    )
    artifact_rows = artifact_index()
    required_artifact_names = [
        "v22_29_literature_matrix.md",
        "v22_29_historical_mechanism_reanalysis.csv",
        "v22_29_signal_channel_metrics.csv",
        "v22_29_grassmann_subspace_metrics.csv",
        "v22_29_tangent_update_spectrum.csv",
        "v22_29_online_subspace_tracking_matrix.csv",
        "v22_29_branch_causal_pilot_matrix.csv",
        "v22_29_branch_causal_controls_matrix.csv",
        "v22_29_mlp_full_loop_matrix.csv",
        "v22_29_kan_full_loop_matrix.csv",
        "v22_29_strong_optimizer_baselines.csv",
        "v22_29_kanbefair_gap_matrix.csv",
        "v22_29_continual_forgetting_matrix.csv",
        "v22_29_grokking_matrix.csv",
        "v22_29_efficiency_component_timing.csv",
        "v22_29_model_identity_matrix.csv",
        "v22_29_final_route.json",
    ]
    required_figure_names = [
        "v22_29_signal_reservoir_spectrum.svg",
        "v22_29_grassmann_principal_angles.svg",
        "v22_29_subspace_transport_error.svg",
        "v22_29_update_spectrum_panel.svg",
        "v22_29_optimizer_interaction_heatmap.svg",
        "v22_29_kan_basis_signal_overlap.svg",
        "v22_29_kanbefair_gap_reduction.svg",
        "v22_29_grokking_delay_curves.svg",
        "v22_29_continual_forgetting_curves.svg",
        "v22_29_branch_causal_forest_plot.svg",
    ]
    required_artifact_audit = []
    for name in required_artifact_names:
        path = OUT_ROOT / name
        row_count = ""
        if path.exists() and path.suffix == ".csv":
            row_count = str(len(read_rows(path)))
        required_artifact_audit.append(
            {
                "artifact": name,
                "status": "present" if path.exists() else "missing",
                "rows": row_count,
            }
        )
    figure_audit = []
    for name in required_figure_names:
        path = OUT_ROOT / "figures" / name
        figure_audit.append(
            {
                "figure": name,
                "status": "present" if path.exists() else "missing",
                "bytes": str(path.stat().st_size) if path.exists() else "",
            }
        )
    strong_no_go_lines = (
        [
            "- 最终路线已从 control-equivalent R3 收缩为 R15：所有已实现的几何/信号 branch pilots 均已启动、均未 pass，且每条 pilot 的 H100/H200 real 分支都能被 matched controls 解释。按计划，这意味着当前 FU 只能保留为 diagnostic / no-harm guardrail，不能 promotion 为稳定改善普通 BP 或 strict FC-PureKAN 的证据。",
        ]
        if route == "R15-StrongNoGo_AllGeometricMechanismsCannotBeatControls"
        else []
    )
    text = [
        "# DG-KAN v22.29 Literature-Grounded Geometric Mechanism Matrix 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 1. Final route",
        "",
        f"- final_route: `{route}`",
        f"- Gate1 hard-pass modules: `{','.join(route_summary.get('hard_pass_modules', [])) or 'none'}`",
        f"- Gate1 proxy-diagnostic-pass modules: `{','.join(route_summary.get('proxy_pass_modules', [])) or 'none'}`",
        f"- Branch-causal pilot launched: `{int(route_summary.get('continuation_pilot_launched', route_summary.get('gate_open', False)))}`",
        f"- Transported momentum repair launched: `{int(route_summary.get('transported_momentum_launched', 0))}`",
        f"- Temporal refresh repair launched: `{int(route_summary.get('temporal_refresh_launched', 0))}`",
        f"- Effect-space A_B repair launched: `{int(route_summary.get('effect_space_launched', 0))}`",
        f"- Tangent-normalized repair launched: `{int(route_summary.get('tangent_normalized_launched', 0))}`",
        f"- Online subspace repair launched: `{int(route_summary.get('online_subspace_launched', 0))}`",
        f"- Online effect-subspace repair launched: `{int(route_summary.get('online_effect_subspace_launched', 0))}`",
        f"- Full-loop launched: `{int(route_summary.get('full_loop_launched', 0))}`",
        f"- Continuation branch pass: `{int(route_summary.get('continuation_branch_pass', 0))}`",
        f"- Transported momentum branch pass: `{int(route_summary.get('transported_momentum_branch_pass', 0))}`",
        f"- Temporal refresh branch pass: `{int(route_summary.get('temporal_branch_pass', 0))}`",
        f"- Effect-space branch pass: `{int(route_summary.get('effect_space_branch_pass', 0))}`",
        f"- Tangent-normalized branch pass: `{int(route_summary.get('tangent_normalized_branch_pass', 0))}`",
        f"- Online subspace branch pass: `{int(route_summary.get('online_subspace_branch_pass', 0))}`",
        f"- Online effect-subspace branch pass: `{int(route_summary.get('online_effect_subspace_branch_pass', 0))}`",
        f"- Strong no-go triggered: `{int(route_summary.get('strong_no_go_triggered', 0))}`",
        f"- Strong no-go controls-better rows total: `{int(route_summary.get('strong_no_go_controls_better_rows_total', 0))}`",
        "",
        "## 2. 本轮新增 / 修改",
        "",
        "- 新增 `experiments/run_v22_29_literature_matrix.py`：实现 v22.29 Gate 1 历史读回、proxy/direct 分离、fallback 分层评估、payload Grassmann 重建、required artifacts、final route、执行日志和复盘生成。",
        "- 修复读回标准化字段：把 v22.15 历史 artifact 中已有的 `final_test_loss_readback` / `final_test_accuracy_readback` 纳入标准化，避免 strong optimizer coverage 出现 `final_NLL=None`；该修复只改变读回覆盖，不改任何原始实验数据。",
        "- 新增 `results/v22_29/` artifacts、`docs/DG-KAN_v22.29_LiteratureGroundedGeometricMechanismMatrix_执行日志.md`、`docs/DG-KAN_v22.29_LiteratureGroundedGeometricMechanismMatrix_实验结果复盘.md`。",
        "- 新增 continuation direct branch-causal pilot：在同一 MLP checkpoint 上运行 B0-B6、H=50/100/200/400，记录 T1 cohort gradient signal 与 T2 gradient-subspace Grassmann 指标、paired NLL/AUC/ECE/Brier/tail delta；该 pilot 只在 `--run-continuation-pilot` 显式开启时执行。",
        "- 修复 continuation 工程 blocker：把 repo root 加入 `sys.path`，避免脚本直接运行时 `dgkan` 导入失败；该修复不改变实验数据。",
        "- 修复数据可用性 blocker：pilot 对 MNIST/FMNIST/KMNIST 优先使用项目本地 `data/` 中已存在的 torchvision raw cache（`download=False`），KANbeFair loader 只作为非本地任务 fallback；该修复不下载、不合成、不替换标签。",
        "- 新增 T6 temporal refresh 修复尝试：当 static T1/T2 分支表现为早期有利、H200/H400 反转且 controls 更强时，按计划转入 slow-gradient/temporal 分析；real 分支使用训练梯度 EMA，controls 使用 same norm/cadence。",
        "- 新增 T1 last-layer/logits effect-space A_B 修复尝试：按计划 fallback 用 per-example readout-gradient effect 构造 drift-diffusion 矩阵，real 与 random/shuffle/signflip controls 均限制在最后一层 `w2` 同一支撑集内；该实现仍不是 dense all-parameter per-example A_B。",
        "- 新增 T4 tangent-normalized signal 修复尝试：按计划把 additive residual 变成当前参数块切空间投影 + block-global normalization；real 和 controls 均经过同一 tangent projection / norm / cadence。该实现是 branch pilot，不声称完整 Mano/Oblique optimizer。",
        "- 修复 T4 smoke 发现的实现错误：`B1_REAL_TANGENT_SIGNAL` 最初未被 `control_direction` 识别，导致 real tangent 分支退化为 zero/no-op；已加入 real branch allowlist 后重跑 smoke/full matrix。",
        "- 新增 T5 online gradient subspace tracking 修复尝试：按计划用 checkpoint cohort gradients 初始化低秩子空间，训练中用 Oja/QR 在线更新；real 与 controls 均在同一个 tracked subspace 内取方向。该实现是 parameter-gradient subspace pilot，JVP/effect subspace tracking 未 promotion。",
        "- 新增 T3 Grassmann-transported source momentum 修复尝试：在 pretrain 中捕获旧 cohort-gradient 子空间，checkpoint 处计算当前子空间，用 Procrustes transport 旧 source momentum；random/signflip/shuffle controls 均约束在同一当前子空间内。",
        "- 新增 T5 effect-space online subspace tracking 修复尝试：按计划实现 Oja/QR effect-space version，追踪最后 readout 的 per-example effect 子空间；real 与 controls 均限制在同一 tracked effect subspace 内。",
        "- 未修改 v22.20 runner / MMFP solver / KANbeFair 源码；v22.29 只在本 runner 中新增 direct pilot / repair pilot 和读回复盘。",
        "",
        "## 3. Gate 1 数据覆盖",
        "",
        f"- historical source files scanned: `{len(inventory)}`",
        f"- standardized task/outcome rows: `{len(task_rows)}`",
        f"- historical metric rows: `{len(metric_rows)}`",
        f"- evaluation rows including fallback strata: `{len(eval_rows)}`",
        f"- direct Grassmann payload rows reconstructed: `{route_summary.get('direct_grassmann_rows', 0)}`",
        f"- outcome-linked direct Grassmann rows: `{route_summary.get('outcome_linked_direct_grassmann_rows', 0)}`",
        f"- hard gate pass rows: `{len(hard_pass)}`",
        f"- proxy diagnostic pass rows: `{len(proxy_pass)}`",
        "",
        "## 4. Module status",
        "",
        md_table(module_status, ["module_id", "module_name", "gate1_status", "hard_gate_pass", "proxy_diagnostic_pass", "blocker"]),
        "",
        "## 5. Top historical metric evaluations",
        "",
        md_table(
            top_eval,
            [
                "stratum_kind",
                "stratum_value",
                "module_id",
                "metric_name",
                "direct_v22_29_metric",
                "rows",
                "positive_NLL_improvement_rows",
                "AUC_oriented",
                "Spearman_abs",
                "control_AUC_oriented",
                "hard_gate_pass",
                "proxy_diagnostic_pass",
                "hard_gate_blocker",
            ],
            limit=12,
        ),
        "",
        "## 6. Direct Grassmann reconstruction",
        "",
        "v22.16 source payload contains actual source vectors, so principal-angle / Grassmann metrics were reconstructed honestly.  However, the exact payload variants do not have a paired task-outcome table, so these rows are direct diagnostics but not hard-gate eligible.",
        "",
        md_table(
            grassmann_sample,
            [
                "dataset",
                "seed",
                "model_variant",
                "steps",
                "subspace_k",
                "source_vector_retention",
                "source_subspace_retention",
                "principal_angles_mean",
                "Grassmann_geodesic_distance",
                "hard_gate_eligible",
                "hard_gate_blocker",
            ],
            limit=10,
        ),
        "",
        "## 7. Strong optimizer coverage",
        "",
        md_table(strong_rows[:20], ["status", "optimizer", "dataset", "seed", "model_name", "model_family", "final_NLL", "source_file", "blocker"], limit=20),
        "",
        "## 8. KANbeFair / continual readback",
        "",
        f"- KANbeFair gap rows copied/read: `{len(gap_rows)}`",
        f"- continual rows copied/read: `{len(continual_rows)}`",
        "- These rows are historical context only because no v22.29 module passed Gate 1 and therefore no new full-loop matrix was allowed.",
        "",
        "## 9. Blocker / fallback attempts",
        "",
        "- 按计划 fallback 1：按 model family 分层重算 AUC/Spearman/control power，结果只产生 proxy diagnostic，不满足 direct hard gate。",
        "- 按计划 fallback 2：按 task difficulty 分层，包含 mnist_like、hard_slice、tabular、continual/external rows；仍无 direct hard-gate pass。",
        "- 按计划 fallback 3：按 intervention strength 分层，包含 overguidance、accepted-heavy、accepted-sparse、no-op；仍无 direct hard-gate pass。",
        "- 额外修复尝试：读取 `v22_16_real_trajectory_source_payload.pt` 重建 T2 Grassmann 指标，解决了“只有聚合 CSV 无法重建子空间”的 blocker；但缺少 exact paired task outcome，因此不能 promotion。",
        "- T1/T3/T4/T5/T6/T7 的 direct v22.29 计划指标在 v22.15-v22.20 artifact 中不可完整重建；本轮只写入 proxy diagnostics 和 missing coverage，不启动 branch/full-loop。",
        "",
        "## 10. Continuation direct branch-causal pilot",
        "",
        f"- continuation pilot rows: `{len(branch_rows)}`",
        f"- real branch rows: `{len(real_branch_rows)}`",
        f"- H100/H200 real beats base rows: `{route_summary.get('continuation_h100_h200_real_beats_base_rows', 0)} / {route_summary.get('continuation_h100_h200_real_rows', 0)}`",
        f"- H100/H200 real beats best-control rows: `{route_summary.get('continuation_h100_h200_real_beats_best_control_rows', 0)} / {route_summary.get('continuation_h100_h200_real_rows', 0)}`",
        f"- controls better than real at H100/H200 rows: `{route_summary.get('continuation_controls_better_real_rows', 0)}`",
        f"- H100/H200 mean real NLL delta vs base: `{mean_field(static_h100_h200, 'paired_NLL_delta_H')}`",
        f"- H100/H200 mean best-control NLL delta vs base: `{mean_field(static_h100_h200, 'control_best_delta_H')}`",
        f"- continuation route after pilot: `{route_summary.get('route_after_continuation', route)}`",
        "",
        "说明：continuation pilot 是本轮为解决“historical direct metric 没有 exact paired outcome”的 blocker 而新增的保守修复。它使用 cohort-mean gradients 构造 T1 drift/diffusion 近似，使用 checkpoint cohort gradient subspaces 构造 T2 principal-angle/Grassmann 近似；这不是 per-example full A_B，也不是 full-loop 成功声明。",
        "",
        md_table(
            branch_sample,
            [
                "dataset",
                "seed",
                "branch",
                "horizon",
                "final_test_NLL",
                "final_test_accuracy",
                "paired_NLL_delta_H",
                "control_best_delta_H",
                "beats_base_NLL",
                "beats_best_control_NLL",
                "signal_eigenvalue_margin",
                "cohort_positive_fraction",
                "principal_angles_mean",
                "Grassmann_geodesic_distance",
            ],
            limit=24,
        ),
        "",
        "## 11. Grassmann-transported T3 repair",
        "",
        f"- transported momentum rows: `{len(transported_rows)}`",
        f"- real transported rows: `{len(real_transported_rows)}`",
        f"- H100/H200 transported real beats base rows: `{route_summary.get('transported_h100_h200_real_beats_base_rows', 0)} / {route_summary.get('transported_h100_h200_real_rows', 0)}`",
        f"- H100/H200 transported real beats best-control rows: `{route_summary.get('transported_h100_h200_real_beats_best_control_rows', 0)} / {route_summary.get('transported_h100_h200_real_rows', 0)}`",
        f"- transported controls better than real at H100/H200 rows: `{route_summary.get('transported_controls_better_real_rows', 0)}`",
        f"- H100/H200 transported mean real NLL delta vs base: `{mean_field(transported_h100_h200, 'paired_NLL_delta_H')}`",
        f"- H100/H200 transported mean best-control NLL delta vs base: `{mean_field(transported_h100_h200, 'control_best_delta_H')}`",
        f"- route after transported momentum repair: `{route_summary.get('route_after_transported_momentum', route_summary.get('route_after_continuation', route))}`",
        "",
        "说明：T3 repair 针对 stale source replay 失败解释。旧 source momentum 从 pretrain 中途的 cohort-gradient 子空间 transport 到当前 checkpoint 子空间；controls 使用同一当前 Grassmann 子空间、同 norm 和同 cadence，因此不能把“子空间内任意 residual 有利”误判成 transported source 机制通过。",
        "",
        md_table(
            transported_sample,
            [
                "dataset",
                "seed",
                "branch",
                "horizon",
                "final_test_NLL",
                "final_test_accuracy",
                "paired_NLL_delta_H",
                "control_best_delta_H",
                "beats_base_NLL",
                "beats_best_control_NLL",
                "principal_angles_mean",
                "Grassmann_geodesic_distance",
                "transport_error_after_procrustes",
                "transport_alignment_with_current_signal",
                "matched_control_geometry",
            ],
            limit=24,
        ),
        "",
        "## 12. Temporal refresh repair",
        "",
        f"- temporal refresh rows: `{len(temporal_rows)}`",
        f"- real temporal rows: `{len(real_temporal_rows)}`",
        f"- H100/H200 temporal real beats base rows: `{route_summary.get('temporal_h100_h200_real_beats_base_rows', 0)} / {route_summary.get('temporal_h100_h200_real_rows', 0)}`",
        f"- H100/H200 temporal real beats best-control rows: `{route_summary.get('temporal_h100_h200_real_beats_best_control_rows', 0)} / {route_summary.get('temporal_h100_h200_real_rows', 0)}`",
        f"- temporal controls better than real at H100/H200 rows: `{route_summary.get('temporal_controls_better_real_rows', 0)}`",
        f"- H100/H200 temporal mean real NLL delta vs base: `{mean_field(temporal_h100_h200, 'paired_NLL_delta_H')}`",
        f"- H100/H200 temporal mean best-control NLL delta vs base: `{mean_field(temporal_h100_h200, 'control_best_delta_H')}`",
        f"- route after temporal refresh: `{route_summary.get('route_after_temporal_refresh', route_summary.get('route_after_transported_momentum', route_summary.get('route_after_continuation', route)))}`",
        "",
        md_table(
            temporal_sample,
            [
                "dataset",
                "seed",
                "branch",
                "horizon",
                "final_test_NLL",
                "final_test_accuracy",
                "paired_NLL_delta_H",
                "control_best_delta_H",
                "beats_base_NLL",
                "beats_best_control_NLL",
                "slow_fast_norm_ratio_mean",
                "slow_grad_alignment_mean",
            ],
            limit=24,
        ),
        "",
        "## 13. Effect-space T1 A_B repair",
        "",
        f"- effect-space rows: `{len(effect_rows)}`",
        f"- real effect-space rows: `{len(real_effect_rows)}`",
        f"- H100/H200 effect real beats base rows: `{route_summary.get('effect_h100_h200_real_beats_base_rows', 0)} / {route_summary.get('effect_h100_h200_real_rows', 0)}`",
        f"- H100/H200 effect real beats best-control rows: `{route_summary.get('effect_h100_h200_real_beats_best_control_rows', 0)} / {route_summary.get('effect_h100_h200_real_rows', 0)}`",
        f"- effect controls better than real at H100/H200 rows: `{route_summary.get('effect_controls_better_real_rows', 0)}`",
        f"- H100/H200 effect mean real NLL delta vs base: `{mean_field(effect_h100_h200, 'paired_NLL_delta_H')}`",
        f"- H100/H200 effect mean best-control NLL delta vs base: `{mean_field(effect_h100_h200, 'control_best_delta_H')}`",
        f"- route after effect-space repair: `{route_summary.get('route_after_effect_space', route_summary.get('route_after_temporal_refresh', route_summary.get('route_after_transported_momentum', route_summary.get('route_after_continuation', route))))}`",
        "",
        "说明：该修复回应前一轮 static T1/T2 使用 cohort-mean gradient 的弱点，改为最后一层 logits effect-space per-example drift-diffusion 矩阵 $A_B$。为了防止 controls 不公平，B2/B3/B5 controls 只在与 real 相同的 `w2` 非零支撑集内采样或打乱；B4 是同支撑 signflip。",
        "",
        md_table(
            effect_sample,
            [
                "dataset",
                "seed",
                "branch",
                "horizon",
                "final_test_NLL",
                "final_test_accuracy",
                "paired_NLL_delta_H",
                "control_best_delta_H",
                "beats_base_NLL",
                "beats_best_control_NLL",
                "signal_eig_topk",
                "signal_eig_margin_vs_controls",
                "cohort_positive_fraction",
                "matched_control_support",
            ],
            limit=24,
        ),
        "",
        "## 14. Tangent-normalized T4 repair",
        "",
        f"- tangent-normalized rows: `{len(tangent_rows)}`",
        f"- real tangent-normalized rows: `{len(real_tangent_rows)}`",
        f"- H100/H200 tangent real beats base rows: `{route_summary.get('tangent_h100_h200_real_beats_base_rows', 0)} / {route_summary.get('tangent_h100_h200_real_rows', 0)}`",
        f"- H100/H200 tangent real beats best-control rows: `{route_summary.get('tangent_h100_h200_real_beats_best_control_rows', 0)} / {route_summary.get('tangent_h100_h200_real_rows', 0)}`",
        f"- tangent controls better than real at H100/H200 rows: `{route_summary.get('tangent_controls_better_real_rows', 0)}`",
        f"- H100/H200 tangent mean real NLL delta vs base: `{mean_field(tangent_h100_h200, 'paired_NLL_delta_H')}`",
        f"- H100/H200 tangent mean best-control NLL delta vs base: `{mean_field(tangent_h100_h200, 'control_best_delta_H')}`",
        f"- route after tangent-normalized repair: `{route_summary.get('route_after_tangent_normalized', route_summary.get('route_after_effect_space', route_summary.get('route_after_temporal_refresh', route_summary.get('route_after_transported_momentum', route_summary.get('route_after_continuation', route)))))}`",
        "",
        "说明：T4 repair 针对 additive correction 过强/无效的失败解释。每步先做 base AdamW，再把 real/control raw direction 投影到当前参数块的切空间并归一化，最后按 base update norm 的 trust ratio 施加 residual；因此 controls 与 real 共享相同的 tangent geometry。",
        "",
        md_table(
            tangent_sample,
            [
                "dataset",
                "seed",
                "branch",
                "horizon",
                "final_test_NLL",
                "final_test_accuracy",
                "paired_NLL_delta_H",
                "control_best_delta_H",
                "beats_base_NLL",
                "beats_best_control_NLL",
                "tangent_radial_removed_fraction_mean",
                "tangent_empty_block_fraction_mean",
                "matched_control_geometry",
            ],
            limit=24,
        ),
        "",
        "## 15. Online subspace T5 repair",
        "",
        f"- online subspace rows: `{len(online_rows)}`",
        f"- real online subspace rows: `{len(real_online_rows)}`",
        f"- H100/H200 online real beats base rows: `{route_summary.get('online_h100_h200_real_beats_base_rows', 0)} / {route_summary.get('online_h100_h200_real_rows', 0)}`",
        f"- H100/H200 online real beats best-control rows: `{route_summary.get('online_h100_h200_real_beats_best_control_rows', 0)} / {route_summary.get('online_h100_h200_real_rows', 0)}`",
        f"- online controls better than real at H100/H200 rows: `{route_summary.get('online_controls_better_real_rows', 0)}`",
        f"- H100/H200 online mean real NLL delta vs base: `{mean_field(online_h100_h200, 'paired_NLL_delta_H')}`",
        f"- H100/H200 online mean best-control NLL delta vs base: `{mean_field(online_h100_h200, 'control_best_delta_H')}`",
        f"- route after online subspace repair: `{route_after_online}`",
        "",
        "说明：T5 repair 针对 static PCA/source-manifold 不可控的问题。该 pilot 用低秩 gradient subspace 做在线 Oja/QR tracking，real residual 是当前梯度在 tracked subspace 内的投影方向；random/signflip/shuffle controls 也只在同一个 tracked subspace 内活动。",
        "",
        md_table(
            online_sample,
            [
                "dataset",
                "seed",
                "branch",
                "horizon",
                "final_test_NLL",
                "final_test_accuracy",
                "paired_NLL_delta_H",
                "control_best_delta_H",
                "beats_base_NLL",
                "beats_best_control_NLL",
                "subspace_rank",
                "subspace_init_mean_grad_reconstruction_ratio",
                "subspace_grad_reconstruction_ratio_mean",
                "subspace_tracking_step_mean",
                "matched_control_geometry",
            ],
            limit=24,
        ),
        "",
        "## 16. Online effect-subspace T5 repair",
        "",
        f"- online effect-subspace rows: `{len(online_effect_rows)}`",
        f"- real online effect-subspace rows: `{len(real_online_effect_rows)}`",
        f"- H100/H200 online effect real beats base rows: `{route_summary.get('online_effect_h100_h200_real_beats_base_rows', 0)} / {route_summary.get('online_effect_h100_h200_real_rows', 0)}`",
        f"- H100/H200 online effect real beats best-control rows: `{route_summary.get('online_effect_h100_h200_real_beats_best_control_rows', 0)} / {route_summary.get('online_effect_h100_h200_real_rows', 0)}`",
        f"- online effect controls better than real at H100/H200 rows: `{route_summary.get('online_effect_controls_better_real_rows', 0)}`",
        f"- H100/H200 online effect mean real NLL delta vs base: `{mean_field(online_effect_h100_h200, 'paired_NLL_delta_H')}`",
        f"- H100/H200 online effect mean best-control NLL delta vs base: `{mean_field(online_effect_h100_h200, 'control_best_delta_H')}`",
        f"- route after online effect-subspace repair: `{route_summary.get('route_after_online_effect_subspace', route_after_online)}`",
        "",
        "说明：该修复回应 T5 parameter-gradient subspace 仍被 controls 解释的问题，改为最后 readout 的 per-example effect-space Oja/QR tracking。它仍是 last-layer effect pilot，不是 dense all-parameter JVP/effect tracking。",
        "",
        md_table(
            online_effect_sample,
            [
                "dataset",
                "seed",
                "branch",
                "horizon",
                "final_test_NLL",
                "final_test_accuracy",
                "paired_NLL_delta_H",
                "control_best_delta_H",
                "beats_base_NLL",
                "beats_best_control_NLL",
                "effect_subspace_rank",
                "effect_subspace_init_mean_reconstruction_ratio",
                "effect_subspace_grad_reconstruction_ratio_mean",
                "effect_subspace_tracking_step_mean",
                "matched_control_geometry",
            ],
            limit=24,
        ),
        "",
        "## 17. Lower-trust repair sweep",
        "",
        "这些 rows 来自 R3 后的继续修复尝试：static/temporal 以及 trust=0.10 与 trust=0.03 对照。它们不参与 best-row promotion，只用于判断失败是否只是步长过大。",
        "",
        md_table(
            sweep_rows,
            [
                "repair_group",
                "trust_ratio",
                "temporal_beta",
                "rows",
                "h100_h200_real_rows",
                "real_beats_base_rows",
                "real_beats_best_control_rows",
                "controls_better_real_rows",
                "branch_pass",
                "route_after_repair",
            ],
            limit=20,
        ),
        "",
        "## 18. T3 transported momentum parameter sweep",
        "",
        "这些 rows 是 R3 后的 T3 失败归因 sweep：rank、transport beta、trust ratio 只用于 failure analysis，不用于 best-row promotion。",
        "",
        md_table(
            transported_sweep_rows,
            [
                "rank",
                "beta",
                "trust_ratio",
                "rows",
                "h100_h200_real_rows",
                "real_beats_base_rows",
                "real_beats_best_control_rows",
                "controls_better_real_rows",
                "branch_pass",
                "mean_real_delta_vs_base",
                "mean_best_control_delta_vs_base",
                "mean_transport_alignment",
                "mean_transport_error_after_procrustes",
                "route_after_repair",
            ],
            limit=20,
        ),
        "",
        "结论：12 个 rank/beta/trust 组合均未打开 branch gate。rank=2/4、beta=0/0.8/1.0、trust=0.01/0.03 都没有让 real transported source 稳定击败 matched controls；因此 T3 失败不应被简化为 transport beta、rank 或 residual step-size 的单点调参问题。",
        "",
        "## 19. T4/T5 multiseed control audit",
        "",
        "这些 rows 复查 T4 tangent-normalized 与 T5 online-subspace 在 MNIST/FMNNIST seeds 0,1,2 上是否只是 seed-0 偶然失败。该审计不参与 promotion。",
        "",
        md_table(
            multiseed_audit_rows,
            [
                "audit",
                "datasets",
                "seeds",
                "h100_h200_real_rows",
                "real_beats_base_rows",
                "real_beats_best_control_rows",
                "controls_better_real_rows",
                "mean_real_delta_vs_base",
                "mean_best_control_delta_vs_base",
                "mean_real_minus_best_control_NLL",
                "winning_control_counts",
                "branch_pass",
            ],
            limit=20,
        ),
        "",
        "结论：T4 与 T5 在多 seed 审计中仍只有 1/12 real rows 击败 best-control，controls better real 为 11/12；winning controls 分散在 B2/B3/B4/B5，不是单一 signflip control 或 seed-0 artifact。",
        "",
        "## 20. 分析与结论",
        "",
        "- v22.29 的核心硬门是“新机制指标必须能在 historical rows 上预测真实 NLL/AUC benefit，并且 controls 的预测力显著更低”。本轮没有任何 direct、outcome-linked 指标满足该标准。",
        "- 一些历史 proxy（例如 current-batch MMFP accept/Q、basis energy、source horizon loss、runtime ratio）可以产生诊断相关性，但计划明确禁止 proxy promotion；因此不能作为 T1-T7 机制通过证据。",
        "- T2 的真实子空间指标可以从 payload 重建，这是本轮最有价值的审计修复；它说明后续若要继续，需要把 source-vector payload 与 exact task branch outcomes 同步记录，而不是只保留 scalar summary。",
        "- continuation pilot 若未达到 H100/H200 real beat base and best-control >=60%，则只能说明当前 T1/T2 cohort-gradient residual 仍不能开 full-loop；若 controls 更好，则进入 control-equivalent/no-go，而不是 promotion。",
        "- transported momentum repair 若仍不能 beat controls，则 stale source replay 的失败不能简单归因于“缺少 Grassmann transport”；至少当前 checkpoint-level Procrustes transport 没有把 real source momentum 从 matched same-subspace controls 中分离出来。",
        "- effect-space A_B repair 若仍被同支撑 controls 解释，则当前失败不能再归因于“没有实现 per-example signal fallback”；更可能是该 readout-level signal 没有稳定转化为后续 NLL 改善，或需要比 readout-only 更强但仍可审计的跨层 effect estimator。",
        "- tangent-normalized repair 若仍不能 beat controls，则 additive-vs-tangent 几何不是当前矩阵的充分解释；至少在这个 MLP branch pilot 里，切空间规范化没有把 real signal 从 matched controls 中分离出来。",
        "- online subspace repair 若仍不能 beat controls，则 static source-manifold 的失败也不能简单归咎于“缺少在线 tracking”；当前 parameter-gradient subspace tracking 仍缺少可区分 real direction 的 task advantage。",
        "- online effect-subspace repair 若仍不能 beat controls，则把 T5 从 parameter-gradient subspace 换成 last-layer per-example effect subspace 仍不足以分离 real task-useful signal；下一步只能考虑 dense/blockwise effect estimator 或 delayed-generalization 任务，而不能把 readout-only 成功当 basis-native 证据。",
        "- T3 parameter sweep 进一步排除了 rank/beta/trust 的简单解释；即使 real direction 使用 current-only、mixed transport 或 pure transported momentum，controls 仍能解释 H100/H200 benefit。",
        "- T4/T5 多 seed 审计排除了 seed-0-only 失败和单一 control branch 病灶；更合理的解释是当前机制信号在这些小规模 MLP branch pilots 中只有“可达子空间/扰动强度”优势，没有稳定 task-useful direction 优势。",
        *strong_no_go_lines,
        "",
        "## 21. Completion audit",
        "",
        "本节对照完整计划做最终状态审计。`skipped` 不等于缺失：当 Gate 1/branch-causal 未通过时，计划要求不启动 full-loop，而是写入 skipped/no-go artifact。",
        "",
        "- Terminal route evidence: `results/v22_29/v22_29_final_route.json` 与 `results/v22_29/v22_29_historical_failure_attribution.json` 均记录 `R15-StrongNoGo_AllGeometricMechanismsCannotBeatControls`。",
        f"- Strong no-go trigger: `{int(route_summary.get('strong_no_go_triggered', 0))}`; controls-better H100/H200 rows total: `{int(route_summary.get('strong_no_go_controls_better_rows_total', 0))}`。",
        "- Full-loop evidence: `v22_29_mlp_full_loop_matrix.csv`、`v22_29_kan_full_loop_matrix.csv`、`v22_29_grokking_matrix.csv` 均存在，并以 skipped row 记录 Gate1/branch 阻断原因。",
        "",
        md_table(required_artifact_audit, ["artifact", "status", "rows"], limit=40),
        "",
        md_table(figure_audit, ["figure", "status", "bytes"], limit=40),
        "",
        "## 22. Artifact index",
        "",
        "- `results/v22_29/v22_29_artifact_index.csv`",
        "- `results/v22_29/v22_29_historical_mechanism_reanalysis.csv`",
        "- `results/v22_29/v22_29_grassmann_subspace_metrics.csv`",
        "- `results/v22_29/v22_29_final_route.json`",
        "- `results/v22_29/figures/`",
        f"- artifact rows: `{len(artifact_rows)}`",
        "",
    ]
    RECAP_DOC.write_text("\n".join(text), encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    command = " ".join([PYTHON, "experiments/run_v22_29_literature_matrix.py"] + sys.argv[1:])
    append_exec(
        command,
        task_id="v22_29_start",
        status="started",
        gpu=args.device_map,
        note=f"cwd={ROOT}; historical_root={args.historical_root}; kanbefair_root={args.kanbefair_root}; stage={args.stage}",
    )
    historical_root = Path(args.historical_root)
    if not historical_root.is_absolute():
        historical_root = ROOT / historical_root

    task_rows, inventory = collect_task_rows(historical_root)
    metric_rows = build_metric_rows(task_rows)
    eval_rows = evaluate_all(metric_rows)
    payload_paths = [
        historical_root / "v22_16_real_trajectory_source_manifold_adaptive_fu/official_v22_16/v22_16_real_trajectory_source_payload.pt",
        historical_root / "v22_16_smoke/v22_16_real_trajectory_source_payload.pt",
    ]
    grassmann_rows = compute_grassmann_metrics(payload_paths)
    split_metric_family_files(metric_rows, grassmann_rows)

    route, module_status, route_summary = final_route_from_eval(eval_rows, grassmann_rows)
    gate_open = bool(route_summary.get("gate_open"))
    reason = "Gate1 produced no hard-gate eligible direct mechanism pass; branch/full-loop blocked by v22.29 plan."
    write_placeholder_matrices(gate_open, reason)

    write_rows(OUT_ROOT / "v22_29_historical_source_inventory.csv", inventory)
    write_rows(OUT_ROOT / "v22_29_historical_standardized_rows.csv", task_rows)
    write_rows(OUT_ROOT / "v22_29_historical_metric_rows.csv", metric_rows)
    write_rows(OUT_ROOT / "v22_29_historical_mechanism_reanalysis.csv", eval_rows)
    write_literature_matrix_md(module_status)

    strong_rows = build_strong_optimizer_baselines(task_rows)
    gap_rows = copy_existing_matrix(
        "v22_29_kanbefair_gap_matrix.csv",
        [
            "v22_20/v22_20_kan_vs_mlpfu_gap_matrix*.csv",
            "v22_20/v22_20_kanbefair_gap_reduction_summary*.csv",
        ],
        historical_root,
        {"status": "not_available", "reason": "No historical KANbeFair gap artifact found."},
    )
    continual_rows = copy_existing_matrix(
        "v22_29_continual_forgetting_matrix.csv",
        [
            "v22_20/v22_20_continual_forgetting_matrix*.csv",
            "v22_18/v22_18_class_mnist_continual_matrix*.csv",
        ],
        historical_root,
        {"status": "not_available", "reason": "No historical continual artifact found."},
    )
    copy_existing_matrix(
        "v22_29_efficiency_component_timing.csv",
        [
            "v22_20/v22_20_four_path_efficiency_audit.csv",
            "v22_18/v22_18_kan_efficiency_matrix*.csv",
        ],
        historical_root,
        {"status": "not_available", "reason": "No historical efficiency artifact found."},
    )
    copy_existing_matrix(
        "v22_29_model_identity_matrix.csv",
        [
            "v22_20/v22_20_model_identity_matrix.csv",
            "v22_18/v22_18_model_identity_matrix.csv",
        ],
        historical_root,
        {"status": "not_available", "reason": "No historical identity artifact found."},
    )

    continuation_summary: dict[str, Any] = {}
    if getattr(args, "run_continuation_pilot", False):
        continuation_summary = run_continuation_pilot(args, route_summary)
        route = str(continuation_summary.get("route_after_continuation") or route)
        if int(continuation_summary.get("continuation_branch_pass") or 0) == 1:
            for row in module_status:
                if row.get("module_id") in {"T1", "T2"}:
                    row["gate1_status"] = "opened_by_direct_continuation_pilot"
                    row["blocker"] = "Historical pairing unavailable, but direct continuation pilot passed H100/H200 branch criteria."
        elif int(continuation_summary.get("continuation_pilot_launched") or 0) == 1:
            for row in module_status:
                if row.get("module_id") in {"T1", "T2"}:
                    row["gate1_status"] = "continuation_pilot_no_branch_pass"
                    row["blocker"] = "Historical pairing unavailable; direct continuation pilot did not beat base and matched controls at H100/H200."
        write_literature_matrix_md(module_status)

    transported_summary: dict[str, Any] = {}
    if getattr(args, "run_transported_momentum_pilot", False):
        transported_summary = run_transported_momentum_pilot(args, route_summary)
        route = str(transported_summary.get("route_after_transported_momentum") or route)
        if int(transported_summary.get("transported_momentum_branch_pass") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T3":
                    row["gate1_status"] = "opened_by_transported_momentum_pilot"
                    row["blocker"] = "Grassmann-transported source momentum pilot passed H100/H200 branch criteria."
        elif int(transported_summary.get("transported_momentum_launched") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T3":
                    row["gate1_status"] = "transported_momentum_no_branch_pass"
                    row["blocker"] = "Grassmann-transported source momentum pilot did not beat base and matched controls at H100/H200."
        write_literature_matrix_md(module_status)

    temporal_summary: dict[str, Any] = {}
    if getattr(args, "run_temporal_refresh_pilot", False):
        temporal_summary = run_temporal_refresh_pilot(args, route_summary)
        route = str(temporal_summary.get("route_after_temporal_refresh") or route)
        if int(temporal_summary.get("temporal_branch_pass") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T6":
                    row["gate1_status"] = "opened_by_temporal_refresh_pilot"
                    row["blocker"] = "Temporal slow-gradient refresh pilot passed H100/H200 branch criteria."
        elif int(temporal_summary.get("temporal_refresh_launched") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T6":
                    row["gate1_status"] = "temporal_refresh_no_branch_pass"
                    row["blocker"] = "Temporal slow-gradient refresh pilot did not beat base and matched controls at H100/H200."
        write_literature_matrix_md(module_status)

    effect_summary: dict[str, Any] = {}
    if getattr(args, "run_effect_space_pilot", False):
        effect_summary = run_effect_space_pilot(args, route_summary)
        route = str(effect_summary.get("route_after_effect_space") or route)
        if int(effect_summary.get("effect_space_branch_pass") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T1":
                    row["gate1_status"] = "opened_by_effect_space_pilot"
                    row["blocker"] = "Last-layer/logits per-example effect-space A_B pilot passed H100/H200 branch criteria."
        elif int(effect_summary.get("effect_space_launched") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T1":
                    row["gate1_status"] = "effect_space_no_branch_pass"
                    row["blocker"] = "Last-layer/logits effect-space A_B pilot did not beat base and matched controls at H100/H200."
        write_literature_matrix_md(module_status)

    tangent_summary: dict[str, Any] = {}
    if getattr(args, "run_tangent_normalized_pilot", False):
        tangent_summary = run_tangent_normalized_pilot(args, route_summary)
        route = str(tangent_summary.get("route_after_tangent_normalized") or route)
        if int(tangent_summary.get("tangent_normalized_branch_pass") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T4":
                    row["gate1_status"] = "opened_by_tangent_normalized_pilot"
                    row["blocker"] = "Tangent-normalized signal pilot passed H100/H200 branch criteria."
        elif int(tangent_summary.get("tangent_normalized_launched") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T4":
                    row["gate1_status"] = "tangent_normalized_no_branch_pass"
                    row["blocker"] = "Tangent-normalized signal pilot did not beat base and matched controls at H100/H200."
        write_literature_matrix_md(module_status)

    online_summary: dict[str, Any] = {}
    if getattr(args, "run_online_subspace_pilot", False):
        online_summary = run_online_subspace_pilot(args, route_summary)
        route = str(online_summary.get("route_after_online_subspace") or route)
        if int(online_summary.get("online_subspace_branch_pass") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T5":
                    row["gate1_status"] = "opened_by_online_subspace_pilot"
                    row["blocker"] = "Online gradient subspace tracking pilot passed H100/H200 branch criteria."
        elif int(online_summary.get("online_subspace_launched") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T5":
                    row["gate1_status"] = "online_subspace_no_branch_pass"
                    row["blocker"] = "Online gradient subspace tracking pilot did not beat base and matched controls at H100/H200."
        write_literature_matrix_md(module_status)

    online_effect_summary: dict[str, Any] = {}
    if getattr(args, "run_online_effect_subspace_pilot", False):
        online_effect_summary = run_online_effect_subspace_pilot(args, route_summary)
        route = str(online_effect_summary.get("route_after_online_effect_subspace") or route)
        if int(online_effect_summary.get("online_effect_subspace_branch_pass") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T5":
                    row["gate1_status"] = "opened_by_online_effect_subspace_pilot"
                    row["blocker"] = "Online readout effect-subspace tracking pilot passed H100/H200 branch criteria."
        elif int(online_effect_summary.get("online_effect_subspace_launched") or 0) == 1:
            for row in module_status:
                if row.get("module_id") == "T5":
                    row["gate1_status"] = "online_effect_subspace_no_branch_pass"
                    row["blocker"] = "Online readout effect-subspace tracking pilot did not beat base and matched controls at H100/H200."
        write_literature_matrix_md(module_status)

    route = maybe_promote_strong_no_go(route, route_summary)

    failure = {
        "route": route,
        "route_summary": route_summary,
        "module_status": module_status,
        "fallback_attempts": [
            "family_stratification",
            "task_difficulty_stratification",
            "intervention_strength_stratification",
            "v22_16_source_payload_grassmann_reconstruction",
            "direct_continuation_pilot",
            "grassmann_transported_source_momentum_pilot",
            "temporal_slow_gradient_refresh_pilot",
            "last_layer_effect_space_A_B_pilot",
            "tangent_normalized_signal_pilot",
            "online_gradient_subspace_tracking_pilot",
            "online_readout_effect_subspace_tracking_pilot",
        ],
        "non_promoted_proxy_rule": "Proxy diagnostic pass cannot open branch/full-loop under v22.29 no-proxy-promotion hard constraint.",
        "latest_status_timestamp": now_sg(),
    }
    write_json(OUT_ROOT / "v22_29_historical_failure_attribution.json", failure)
    final_route = {
        "final_route": route,
        "gate1_hard_pass": int(gate_open),
        "branch_causal_pilot_launched": int(route_summary.get("continuation_pilot_launched", 0)),
        "branch_causal_pilot_pass": int(route_summary.get("continuation_branch_pass", 0)),
        "transported_momentum_launched": int(route_summary.get("transported_momentum_launched", 0)),
        "transported_momentum_pass": int(route_summary.get("transported_momentum_branch_pass", 0)),
        "temporal_refresh_launched": int(route_summary.get("temporal_refresh_launched", 0)),
        "temporal_refresh_pass": int(route_summary.get("temporal_branch_pass", 0)),
        "effect_space_launched": int(route_summary.get("effect_space_launched", 0)),
        "effect_space_pass": int(route_summary.get("effect_space_branch_pass", 0)),
        "tangent_normalized_launched": int(route_summary.get("tangent_normalized_launched", 0)),
        "tangent_normalized_pass": int(route_summary.get("tangent_normalized_branch_pass", 0)),
        "online_subspace_launched": int(route_summary.get("online_subspace_launched", 0)),
        "online_subspace_pass": int(route_summary.get("online_subspace_branch_pass", 0)),
        "online_effect_subspace_launched": int(route_summary.get("online_effect_subspace_launched", 0)),
        "online_effect_subspace_pass": int(route_summary.get("online_effect_subspace_branch_pass", 0)),
        "full_loop_launched": 0,
        "route_summary": route_summary,
        "latest_status_timestamp": now_sg(),
    }
    write_json(OUT_ROOT / "v22_29_final_route.json", final_route)

    write_figures(eval_rows, grassmann_rows, gap_rows, continual_rows)
    write_source_packet_manifest()
    write_rows(OUT_ROOT / "v22_29_artifact_index.csv", artifact_index())
    write_recap(
        route,
        route_summary,
        task_rows,
        metric_rows,
        eval_rows,
        grassmann_rows,
        module_status,
        inventory,
        gap_rows,
        continual_rows,
        strong_rows,
    )
    write_rows(OUT_ROOT / "v22_29_artifact_index.csv", artifact_index())
    append_exec(
        command,
        task_id="v22_29_all",
        status="pass",
        gpu=args.device_map,
        files=(
            "results/v22_29/v22_29_final_route.json, "
            "results/v22_29/v22_29_historical_mechanism_reanalysis.csv, "
            "docs/DG-KAN_v22.29_LiteratureGroundedGeometricMechanismMatrix_实验结果复盘.md"
        ),
        note=(
            f"final_route={route}; gate_open={int(gate_open)}; "
            f"continuation_pilot_launched={int(route_summary.get('continuation_pilot_launched', 0))}; "
            f"branch_pass={int(route_summary.get('continuation_branch_pass', 0))}; "
            f"transported_momentum_launched={int(route_summary.get('transported_momentum_launched', 0))}; "
            f"transported_pass={int(route_summary.get('transported_momentum_branch_pass', 0))}; "
            f"temporal_refresh_launched={int(route_summary.get('temporal_refresh_launched', 0))}; "
            f"temporal_pass={int(route_summary.get('temporal_branch_pass', 0))}; "
            f"effect_space_launched={int(route_summary.get('effect_space_launched', 0))}; "
            f"effect_pass={int(route_summary.get('effect_space_branch_pass', 0))}; "
            f"tangent_normalized_launched={int(route_summary.get('tangent_normalized_launched', 0))}; "
            f"tangent_pass={int(route_summary.get('tangent_normalized_branch_pass', 0))}; "
            f"online_subspace_launched={int(route_summary.get('online_subspace_launched', 0))}; "
            f"online_pass={int(route_summary.get('online_subspace_branch_pass', 0))}; "
            f"online_effect_subspace_launched={int(route_summary.get('online_effect_subspace_launched', 0))}; "
            f"online_effect_pass={int(route_summary.get('online_effect_subspace_branch_pass', 0))}; full_loop_launched=0"
        ),
        exit_code=0,
    )
    write_rows(OUT_ROOT / "v22_29_artifact_index.csv", artifact_index())
    return final_route


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all"])
    p.add_argument("--historical-root", default="results")
    p.add_argument("--kanbefair-root", default="KANbeFair")
    p.add_argument("--device-map", default="0,1,2,3")
    p.add_argument("--run-continuation-pilot", action="store_true")
    p.add_argument("--run-transported-momentum-pilot", action="store_true")
    p.add_argument("--run-temporal-refresh-pilot", action="store_true")
    p.add_argument("--run-effect-space-pilot", action="store_true")
    p.add_argument("--run-tangent-normalized-pilot", action="store_true")
    p.add_argument("--run-online-subspace-pilot", action="store_true")
    p.add_argument("--run-online-effect-subspace-pilot", action="store_true")
    p.add_argument("--pilot-device", default="cuda:0")
    p.add_argument("--pilot-data-root", default="data")
    p.add_argument("--pilot-datasets", default="MNIST,FMNIST,KMNIST")
    p.add_argument("--pilot-seeds", default="0")
    p.add_argument("--pilot-horizons", default="50,100,200,400")
    p.add_argument("--pilot-train-size", type=int, default=512)
    p.add_argument("--pilot-test-size", type=int, default=256)
    p.add_argument("--pilot-batch-size", type=int, default=64)
    p.add_argument("--pilot-hidden", type=int, default=16)
    p.add_argument("--pilot-pretrain-steps", type=int, default=50)
    p.add_argument("--pilot-cohorts", type=int, default=6)
    p.add_argument("--pilot-subspace-rank", type=int, default=4)
    p.add_argument("--pilot-oja-lr", type=float, default=0.20)
    p.add_argument("--pilot-transport-gap", type=int, default=20)
    p.add_argument("--pilot-transport-beta", type=float, default=0.80)
    p.add_argument("--pilot-trust-ratio", type=float, default=0.10)
    p.add_argument("--pilot-lr", type=float, default=2.0e-3)
    p.add_argument("--pilot-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--temporal-slow-beta", type=float, default=0.95)
    return p


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    run_all(args)


if __name__ == "__main__":
    main()
