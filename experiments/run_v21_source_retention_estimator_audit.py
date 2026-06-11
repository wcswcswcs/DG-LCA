#!/usr/bin/env python3
"""F5 source-retention estimator audit for v21.

This is an offline audit. It reads completed v21 matrix/trace artifacts, aligns
early train-stream diagnostics with later source readback, and decides whether
any estimator is good enough to become a direction selector.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    read_rows,
    write_rows,
)


CONTROL_MECHANISMS = {
    "CTRL-AdamW",
    "CTRL-SGD",
    "CTRL-NoOpMatchedOverhead",
    "CTRL-RandomMatchedNorm",
    "CTRL-RandomSameRankBlock",
    "CTRL-RecoveryOnly",
}

HORIZONS = ("800", "1600", "3200", "4800")
PREDICTORS = {
    "P0_early_source_readback": "source_h800",
    "P1_parameter_snr_baseline": "PopRisk_SNR_score_mean_h800",
    "P2_output_transfer_b2_gain": "B2_transfer_gain_mean_h800",
    "P3_split_consensus_signal_minus_corrupt": "source_state_signal_minus_corrupt_mean_h800",
    "P4_linec_channel_quality": "LineC_channel_quality_h800",
    "P5_matrix_block_norm": "block_update_norm_mean_h800",
    "P6_slow_state_stability": "source_state_current_cos_mean_h800",
    "P7_random_or_control_indicator": "non_control_indicator",
    "P8_train_source_readback": "train_source_h800",
    "P9_train_loss_drop": "train_loss_drop_h800",
    "P10_train_source_x_linec_quality": "train_source_x_linec_quality_h800",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def job_key(row: dict[str, Any]) -> str:
    return f"{row.get('run_label')}:{row.get('job_index')}"


def measured_non_smoke(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if "smoke" in str(row.get("run_label", "")).lower():
            continue
        if str(row.get("execution_status", "measured")) not in {"", "measured"}:
            continue
        out.append(row)
    return out


def add_trace_horizons(rows: list[dict[str, Any]], traces: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_job: dict[str, list[dict[str, str]]] = {}
    by_step: dict[str, dict[str, dict[str, str]]] = {}
    for tr in traces:
        key = job_key(tr)
        by_job.setdefault(key, []).append(tr)
        step = str(tr.get("step"))
        if step in HORIZONS:
            by_step.setdefault(key, {})[step] = tr
    fields = [
        "val_loss",
        "train_loss",
        "LineC_channel_loss",
        "LineC_fast_loss",
        "ActuationR2",
        "B1_gain",
        "B2_transfer_gain",
        "B3_safety_gain",
        "source_state_gate_accept",
        "source_state_signal_gain",
        "source_state_corrupt_gain",
        "source_state_current_cos",
        "source_state_consensus_density",
        "generalization_signal_gain",
        "generalization_gate_accept",
        "PopRisk_SNR_score",
        "block_update_norm",
    ]
    out = []
    for row in rows:
        item = dict(row)
        key = job_key(row)
        for step in HORIZONS:
            tr = by_step.get(key, {}).get(step, {})
            for field in fields:
                if field in tr:
                    item[f"{field}_h{step}"] = tr.get(field)
        early = [tr for tr in by_job.get(key, []) if 0 <= finite_float(tr.get("step"), -1.0) <= 800]
        for field in [
            "ActuationR2",
            "B2_transfer_gain",
            "source_state_signal_gain",
            "source_state_corrupt_gain",
            "source_state_current_cos",
            "source_state_consensus_density",
            "PopRisk_SNR_score",
            "block_update_norm",
        ]:
            vals = [finite_float(tr.get(field)) for tr in early]
            vals = [v for v in vals if v == v]
            item[f"{field}_mean_h800"] = sum(vals) / len(vals) if vals else ""
            item[f"{field}_max_h800"] = max(vals) if vals else ""
        sig = finite_float(item.get("source_state_signal_gain_mean_h800"))
        corrupt = finite_float(item.get("source_state_corrupt_gain_mean_h800"))
        item["source_state_signal_minus_corrupt_mean_h800"] = sig - max(0.0, corrupt) if sig == sig and corrupt == corrupt else ""
        linec = finite_float(item.get("LineC_channel_loss_h800"))
        item["LineC_channel_quality_h800"] = 1.0 - linec if linec == linec else ""
        item["non_control_indicator"] = 0 if str(item.get("mechanism")) in CONTROL_MECHANISMS else 1
        out.append(item)
    return out


def enrich_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str, str, str], float] = {}
    train_best: dict[tuple[str, str, str, str, str], float] = {}
    for row in rows:
        if str(row.get("mechanism")) not in CONTROL_MECHANISMS:
            continue
        for h in HORIZONS:
            val = finite_float(row.get(f"val_loss_h{h}"))
            if val == val:
                key = (str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("dataset")), str(row.get("seed")), h)
                best[key] = min(best.get(key, float("inf")), val)
            train = finite_float(row.get(f"train_loss_h{h}"))
            if train == train:
                key = (str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("dataset")), str(row.get("seed")), h)
                train_best[key] = min(train_best.get(key, float("inf")), train)
    out = []
    for row in rows:
        item = dict(row)
        for h in HORIZONS:
            if finite_float(item.get(f"source_h{h}")) == finite_float(item.get(f"source_h{h}")):
                continue
            val = finite_float(item.get(f"val_loss_h{h}"))
            base = best.get((str(item.get("carrier")), str(item.get("basis_repair_variant")), str(item.get("dataset")), str(item.get("seed")), h), float("nan"))
            item[f"source_h{h}"] = base - val if val == val and base == base else ""
            train = finite_float(item.get(f"train_loss_h{h}"))
            train_base = train_best.get((str(item.get("carrier")), str(item.get("basis_repair_variant")), str(item.get("dataset")), str(item.get("seed")), h), float("nan"))
            item[f"train_source_h{h}"] = train_base - train if train == train and train_base == train_base else ""
        train0 = finite_float(item.get("train_loss"))
        train800 = finite_float(item.get("train_loss_h800"))
        item["train_loss_drop_h800"] = train0 - train800 if train0 == train0 and train800 == train800 else ""
        train_source = finite_float(item.get("train_source_h800"))
        linec_quality = finite_float(item.get("LineC_channel_quality_h800"))
        item["train_source_x_linec_quality_h800"] = train_source * linec_quality if train_source == train_source and linec_quality == linec_quality else ""
        out.append(item)
    return out


def load_audit_rows(out_dir: Path) -> list[dict[str, Any]]:
    mlp = measured_non_smoke(read_rows(out_dir / "v21_mlp_source_dynamics_raw_matrix.csv"))
    mlp_traces = measured_non_smoke(read_rows(out_dir / "v21_mlp_source_dynamics_raw_traces.csv"))
    kan = measured_non_smoke(read_rows(out_dir / "v21_kan_source_writer_raw_matrix.csv"))
    kan_traces = measured_non_smoke(read_rows(out_dir / "v21_kan_source_writer_raw_traces.csv"))
    rows = enrich_sources(add_trace_horizons(mlp, mlp_traces)) + enrich_sources(add_trace_horizons(kan, kan_traces))
    out = []
    for row in rows:
        h800 = finite_float(row.get("source_h800"))
        h1600 = finite_float(row.get("source_h1600"))
        h3200 = finite_float(row.get("source_h3200"))
        h4800 = finite_float(row.get("source_h4800"))
        if not all(v == v for v in [h800, h1600, h3200, h4800]):
            continue
        retained_h1600 = int(h800 > 0.0 and h1600 >= 0.005)
        retained_h3200 = int(h800 > 0.0 and h1600 >= 0.005 and h3200 >= 0.005)
        retained_h4800 = int(retained_h3200 and h4800 >= 0.005)
        washout_h3200 = int(h800 > 0.0 and (h1600 < 0.005 or h3200 < 0.005))
        late_rebound_h3200 = int(h800 <= 0.0 and h3200 >= 0.005)
        item = {
            "scope": row.get("scope", ""),
            "run_label": row.get("run_label", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "carrier": row.get("carrier", ""),
            "basis_repair_variant": row.get("basis_repair_variant", ""),
            "v21_id": row.get("v21_id", ""),
            "mechanism": row.get("mechanism", ""),
            "hypothesis": row.get("hypothesis", ""),
            "source_h800": h800,
            "source_h1600": h1600,
            "source_h3200": h3200,
            "source_h4800": h4800,
            "retained_h1600_label": retained_h1600,
            "retained_h3200_label": retained_h3200,
            "retained_h4800_label": retained_h4800,
            "washout_h3200_label": washout_h3200,
            "late_rebound_h3200_label": late_rebound_h3200,
        }
        for _name, field in PREDICTORS.items():
            value = finite_float(row.get(field))
            item[field] = value if value == value else ""
        out.append(item)
    return out


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(rows: list[dict[str, Any]], predictor: str, target: str) -> float:
    pairs = []
    for row in rows:
        x = finite_float(row.get(predictor))
        y = finite_float(row.get(target))
        if x == x and y == y:
            pairs.append((x, y))
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    return pearson(ranks(list(xs)), ranks(list(ys)))


def auc_score(rows: list[dict[str, Any]], predictor: str, label: str) -> float:
    pos = []
    neg = []
    for row in rows:
        x = finite_float(row.get(predictor))
        y = int(finite_float(row.get(label), 0.0))
        if x != x:
            continue
        if y:
            pos.append(x)
        else:
            neg.append(x)
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else float("nan")


def auc_retained_vs_washout(rows: list[dict[str, Any]], predictor: str) -> float:
    subset = []
    for row in rows:
        if int(row.get("retained_h3200_label", 0)):
            item = dict(row)
            item["retained_vs_washout_label"] = 1
            subset.append(item)
        elif int(row.get("washout_h3200_label", 0)):
            item = dict(row)
            item["retained_vs_washout_label"] = 0
            subset.append(item)
    return auc_score(subset, predictor, "retained_vs_washout_label")


def partition_auc(rows: list[dict[str, Any]], predictor: str, partition: str) -> tuple[float, float]:
    values = sorted({str(r.get(partition, "")) for r in rows})
    aucs = []
    for value in values:
        subset = [r for r in rows if str(r.get(partition, "")) == value]
        auc = auc_score(subset, predictor, "retained_h3200_label")
        if auc == auc:
            aucs.append(auc)
    if not aucs:
        return float("nan"), float("nan")
    return sum(aucs) / len(aucs), min(aucs)


def rate_errors(rows: list[dict[str, Any]], predictor: str) -> tuple[float, float]:
    vals = [finite_float(r.get(predictor)) for r in rows]
    vals = sorted(v for v in vals if v == v)
    if not vals:
        return float("nan"), float("nan")
    threshold = vals[len(vals) // 2]
    fp = fn = pos = neg = 0
    for row in rows:
        x = finite_float(row.get(predictor))
        if x != x:
            continue
        label = int(row.get("retained_h3200_label", 0))
        pred = int(x >= threshold)
        if label:
            pos += 1
            fn += int(not pred)
        else:
            neg += 1
            fp += int(pred)
    return (fp / neg if neg else float("nan"), fn / pos if pos else float("nan"))


def blank_nan(value: float) -> Any:
    return value if value == value else ""


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control_auc = auc_score(rows, "non_control_indicator", "retained_h3200_label")
    summaries = []
    for name, field in PREDICTORS.items():
        auc_retained = auc_retained_vs_washout(rows, field)
        auc_label = auc_score(rows, field, "retained_h3200_label")
        ds_mean, ds_min = partition_auc(rows, field, "dataset")
        seed_mean, seed_min = partition_auc(rows, field, "seed")
        fpr, fnr = rate_errors(rows, field)
        inc = auc_label - control_auc if auc_label == auc_label and control_auc == control_auc else float("nan")
        row = {
            "predictor": name,
            "field": field,
            "finite_rows": sum(1 for r in rows if finite_float(r.get(field)) == finite_float(r.get(field))),
            "Spearman_h1600": blank_nan(spearman(rows, field, "source_h1600")),
            "Spearman_h3200": blank_nan(spearman(rows, field, "source_h3200")),
            "Spearman_h4800": blank_nan(spearman(rows, field, "source_h4800")),
            "AUC_retained_vs_washout": blank_nan(auc_retained),
            "AUC_retained_h3200": blank_nan(auc_label),
            "leave_dataset_out_AUC_mean": blank_nan(ds_mean),
            "leave_dataset_out_AUC_min": blank_nan(ds_min),
            "leave_seed_out_AUC_mean": blank_nan(seed_mean),
            "leave_seed_out_AUC_min": blank_nan(seed_min),
            "incremental_AUC_vs_controls": blank_nan(inc),
            "false_positive_rate": blank_nan(fpr),
            "false_negative_rate": blank_nan(fnr),
        }
        row["estimator_pass"] = int(
            finite_float(row["leave_dataset_out_AUC_min"]) >= 0.65
            and finite_float(row["leave_seed_out_AUC_min"]) >= 0.65
            and finite_float(row["Spearman_h3200"]) >= 0.30
            and finite_float(row["incremental_AUC_vs_controls"]) >= 0.10
        )
        summaries.append(row)
    return summaries



def stratify(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        for key in [
            ("carrier", str(row.get("carrier", ""))),
            ("carrier_dataset", str(row.get("carrier", "")), str(row.get("dataset", ""))),
            ("carrier_v21_id", str(row.get("carrier", "")), str(row.get("v21_id", ""))),
        ]:
            groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        kind = key[0]
        vals = {"group_kind": kind, "carrier": "", "dataset": "", "v21_id": ""}
        if kind == "carrier":
            vals["carrier"] = key[1]
        elif kind == "carrier_dataset":
            vals["carrier"] = key[1]
            vals["dataset"] = key[2]
        elif kind == "carrier_v21_id":
            vals["carrier"] = key[1]
            vals["v21_id"] = key[2]
        def mean(field: str) -> Any:
            xs = [finite_float(r.get(field)) for r in group]
            xs = [x for x in xs if x == x]
            return sum(xs) / len(xs) if xs else ""
        rows_n = len(group)
        h800_pos = sum(1 for r in group if finite_float(r.get("source_h800")) > 0.0)
        retained_h3200 = sum(int(r.get("retained_h3200_label", 0)) for r in group)
        washout = sum(int(r.get("washout_h3200_label", 0)) for r in group)
        late = sum(int(r.get("late_rebound_h3200_label", 0)) for r in group)
        vals.update(
            {
                "rows": rows_n,
                "h800_positive_rows": h800_pos,
                "retained_h1600_rows": sum(int(r.get("retained_h1600_label", 0)) for r in group),
                "retained_h3200_rows": retained_h3200,
                "retained_h4800_rows": sum(int(r.get("retained_h4800_label", 0)) for r in group),
                "washout_h3200_rows": washout,
                "late_rebound_h3200_rows": late,
                "source_h800_mean": mean("source_h800"),
                "source_h1600_mean": mean("source_h1600"),
                "source_h3200_mean": mean("source_h3200"),
                "source_h4800_mean": mean("source_h4800"),
                "h800_positive_rate": h800_pos / rows_n if rows_n else "",
                "retained_h3200_rate": retained_h3200 / rows_n if rows_n else "",
                "washout_given_h800_positive_rate": washout / h800_pos if h800_pos else "",
                "late_rebound_rate": late / rows_n if rows_n else "",
            }
        )
        out.append(vals)
    return out

def decision(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passing = [r for r in summaries if int(r.get("estimator_pass", 0))]
    train_only_passing = [
        r for r in passing
        if str(r.get("predictor")) not in {"P0_early_source_readback", "P7_random_or_control_indicator"}
    ]
    best = max(summaries, key=lambda r: finite_float(r.get("Spearman_h3200"), -999.0), default={})
    return [
        {
            "audit": "F5-source-retention-estimator",
            "rows": len(rows),
            "retained_h1600_labels": sum(int(r.get("retained_h1600_label", 0)) for r in rows),
            "retained_h3200_labels": sum(int(r.get("retained_h3200_label", 0)) for r in rows),
            "retained_h4800_labels": sum(int(r.get("retained_h4800_label", 0)) for r in rows),
            "washout_h3200_labels": sum(int(r.get("washout_h3200_label", 0)) for r in rows),
            "late_rebound_h3200_labels": sum(int(r.get("late_rebound_h3200_label", 0)) for r in rows),
            "passing_predictors": len(passing),
            "passing_train_only_predictors": len(train_only_passing),
            "best_predictor_by_spearman_h3200": best.get("predictor", ""),
            "best_spearman_h3200": best.get("Spearman_h3200", ""),
            "retention_estimator_valid": int(bool(passing)),
            "direction_selector_allowed": int(bool(train_only_passing)),
            "promotion_allowed": 0,
            "decision": (
                "TrainOnlyEstimatorUsableForNextDirection"
                if train_only_passing
                else "OnlyEarlySourceReadbackPredictsRetention_NoTrainOnlyDirectionSelector"
                if passing
                else "RetentionEstimatorInvalid_CurrentImplementation"
            ),
        }
    ]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_source_retention_estimator_audit.py --out-dir {out_dir}", status="started")
    rows = load_audit_rows(out_dir)
    summary_rows = summarize(rows)
    decision_rows = decision(rows, summary_rows)
    stratification_rows = stratify(rows)
    write_rows(out_dir / "v21_f5_source_retention_estimator_rows.csv", rows)
    write_rows(out_dir / "v21_f5_source_retention_estimator_summary.csv", summary_rows)
    write_rows(out_dir / "v21_f5_source_retention_estimator_decision.csv", decision_rows)
    write_rows(out_dir / "v21_f5_early_source_stratification.csv", stratification_rows)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v21_source_retention_estimator_audit.py --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(rows)} stratification_rows={len(stratification_rows)} passing_predictors={decision_rows[0]['passing_predictors']} train_only_passing={decision_rows[0]['passing_train_only_predictors']} decision={decision_rows[0]['decision']}",
    )


if __name__ == "__main__":
    main()
