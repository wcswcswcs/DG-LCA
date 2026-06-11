#!/usr/bin/env python3
"""v22 C4 source-observability audit.

This script is offline-only: it reads v22 source-chain rows and checks whether
legal train-only diagnostics can select early-source rows without relying on
validation/test/future/source readback.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402


PREDICTORS = [
    ("train_loss_h100", "train_only", "higher_bad"),
    ("train_loss_h400", "train_only", "higher_bad"),
    ("train_loss_h800", "train_only", "higher_bad"),
    ("train_loss_drop_h100_h800", "train_only", "higher_good"),
    ("CEp99_h800", "audit_readback", "higher_good"),
    ("Brier_h800", "audit_readback", "higher_bad"),
    ("LineC_channel_loss_h800", "audit_readback", "higher_bad"),
    ("ActuationR2_h800", "function_readback", "higher_good"),
    ("B2_transfer_gain_h800", "function_readback", "higher_good"),
    ("source_channel_projection_h800", "train_stream_diagnostic", "higher_good"),
    ("source_state_gate_accept_h800", "train_stream_diagnostic", "higher_good"),
    ("generalization_gate_accept_h800", "train_stream_diagnostic", "higher_good"),
    ("C4_E1_split_fisher_snr_h800", "train_stream_diagnostic", "higher_good"),
    ("C4_E6_split_fisher_coherence_h800", "train_stream_diagnostic", "higher_good"),
    ("C4_E7_noise_reservoir_separation_h800", "train_stream_diagnostic", "higher_good"),
    ("source_h400", "source_readback_forbidden", "higher_good"),
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def score(row: dict[str, Any], name: str, direction: str) -> float:
    if name == "train_loss_drop_h100_h800":
        value = finite_float(row.get("train_loss_h100")) - finite_float(row.get("train_loss_h800"))
    elif name == "C4_E1_split_fisher_snr_h800":
        signal = finite_float(row.get("source_state_signal_gain_h800"))
        corrupt = finite_float(row.get("source_state_corrupt_gain_h800"))
        density = finite_float(row.get("source_state_consensus_density_h800"))
        balance = finite_float(row.get("source_state_balance_mean_h800"))
        if not all(math.isfinite(v) for v in [signal, corrupt, density, balance]):
            value = float("nan")
        else:
            value = (signal - max(0.0, corrupt)) * max(0.0, density) * max(0.0, balance)
    elif name == "C4_E6_split_fisher_coherence_h800":
        density = finite_float(row.get("source_state_consensus_density_h800"))
        balance = finite_float(row.get("source_state_balance_mean_h800"))
        current_cos = finite_float(row.get("source_state_current_cos_h800"))
        if not all(math.isfinite(v) for v in [density, balance, current_cos]):
            value = float("nan")
        else:
            value = max(0.0, density) * max(0.0, balance) * max(0.0, current_cos)
    elif name == "C4_E7_noise_reservoir_separation_h800":
        signal = finite_float(row.get("source_state_signal_gain_h800"))
        corrupt = finite_float(row.get("source_state_corrupt_gain_h800"))
        corrupt_cos = finite_float(row.get("source_state_corrupt_cos_h800"))
        if not all(math.isfinite(v) for v in [signal, corrupt, corrupt_cos]):
            value = float("nan")
        else:
            value = signal - max(0.0, corrupt) - max(0.0, corrupt_cos) * max(abs(signal), 1.0e-12)
    else:
        value = finite_float(row.get(name))
    if not math.isfinite(value):
        return float("nan")
    return -value if direction == "higher_bad" else value


def auc(scores: list[float], labels: list[int]) -> float | str:
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = sum(1 for _s, y in pairs if y)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return ""
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def spearman(xs: list[float], ys: list[float]) -> float | str:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return ""
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and vals[order[j]] == vals[order[i]]:
                j += 1
            rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                out[order[k]] = rank
            i = j
        return out
    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0.0 or vy <= 0.0:
        return ""
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def min_group_auc(rows: list[dict[str, Any]], scores: dict[int, float], label: str, group_key: str) -> float | str:
    vals = []
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row.get(group_key, ""))].append((scores.get(idx, float("nan")), int_flag(row.get(label))))
    for items in groups.values():
        a = auc([x for x, _y in items], [y for _x, y in items])
        if a != "":
            vals.append(float(a))
    return min(vals) if vals else ""


def best_threshold(scores: list[float], labels: list[int]) -> tuple[float | str, int, int, int, int]:
    pairs = sorted((s, y) for s, y in zip(scores, labels) if math.isfinite(s))
    if not pairs:
        return "", 0, 0, 0, 0
    best = ("", -1.0, 0, 0, 0, 0)
    thresholds = sorted({s for s, _y in pairs})
    for t in thresholds:
        tp = sum(1 for s, y in pairs if s >= t and y)
        fp = sum(1 for s, y in pairs if s >= t and not y)
        fn = sum(1 for s, y in pairs if s < t and y)
        tn = sum(1 for s, y in pairs if s < t and not y)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        youden = tpr - fpr
        if youden > best[1]:
            best = (t, youden, tp, fp, fn, tn)
    return best[0], best[2], best[3], best[4], best[5]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_source_observability_audit.py", status="started")
    rows = read_rows(out_dir / "v22_source_chain_matrix.csv")
    candidates = [r for r in rows if str(r.get("v22_id", "")).startswith(("F", "KSW", "MLP"))]
    scan = []
    for pred, kind, direction in PREDICTORS:
        pred_rows = candidates if kind != "source_readback_forbidden" else rows
        scores = [score(r, pred, direction) for r in pred_rows]
        labels = [int_flag(r.get("early_source_chain")) for r in pred_rows]
        continuous_labels = [int_flag(r.get("continuous_retention_chain")) for r in pred_rows]
        score_map = {idx: s for idx, s in enumerate(scores)}
        threshold, tp, fp, fn, tn = best_threshold(scores, labels)
        auc_early = auc(scores, labels)
        auc_continuous = auc(scores, continuous_labels)
        h800 = [finite_float(r.get("source_h800")) for r in pred_rows]
        sp_h800 = spearman(scores, h800)
        dataset_min = min_group_auc(pred_rows, score_map, "early_source_chain", "dataset")
        carrier_min = min_group_auc(pred_rows, score_map, "early_source_chain", "carrier")
        finite_rows = sum(1 for s in scores if math.isfinite(s))
        pass_flag = int(
            kind in {"train_only", "train_stream_diagnostic"}
            and finite_rows >= 30
            and auc_early != ""
            and float(auc_early) >= 0.75
            and dataset_min != ""
            and float(dataset_min) >= 0.65
            and carrier_min != ""
            and float(carrier_min) >= 0.65
        )
        scan.append(
            {
                "predictor": pred,
                "kind": kind,
                "finite_rows": finite_rows,
                "positive_rows": sum(labels),
                "continuous_positive_rows": sum(continuous_labels),
                "AUC_early_chain": auc_early,
                "AUC_continuous_retention": auc_continuous,
                "dataset_min_AUC": dataset_min,
                "carrier_min_AUC": carrier_min,
                "Spearman_h800": sp_h800,
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "pass": pass_flag,
            }
        )
    write_rows(out_dir / "v22_source_observability_predictor_scan.csv", scan)
    legal_pass = sum(int_flag(r.get("pass")) for r in scan if r.get("kind") in {"train_only", "train_stream_diagnostic"})
    train_pass = sum(int_flag(r.get("pass")) for r in scan if r.get("kind") == "train_only")
    candidate_continuous_rows = sum(int_flag(r.get("continuous_retention_chain")) for r in candidates)
    if legal_pass and candidate_continuous_rows:
        route = "TrainOnlySelectorCandidateNeedsHeldOut"
    elif legal_pass:
        route = "TrainOnlyEarlySelectorNoContinuousRetention"
    else:
        route = "SourceObservabilityNoGo"
    decision = {
        "audit_rows": len(rows),
        "candidate_rows": len(candidates),
        "candidate_early_rows": sum(int_flag(r.get("early_source_chain")) for r in candidates),
        "candidate_continuous_rows": candidate_continuous_rows,
        "passing_train_only_predictors": train_pass,
        "passing_legal_predictors": legal_pass,
        "decision": route,
        "promotion_allowed": 0,
    }
    write_json(out_dir / "v22_source_observability_decision.json", decision)
    write_rows(out_dir / "v22_source_observability_decision.csv", [decision])
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_source_observability_audit.py", status="completed", note=f"decision={decision['decision']} train_only_pass={train_pass} legal_pass={legal_pass}")


if __name__ == "__main__":
    main()
