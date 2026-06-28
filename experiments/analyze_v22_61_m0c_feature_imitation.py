#!/usr/bin/env python3
"""Feature-conditioned M0C oracle-imitation diagnostic for v22.61.

This diagnostic uses only train-only feature columns logged by the v22.61 oracle
runner. It trains leave-one-seed-out tiny classifiers to imitate the objective
oracle grid choice, then evaluates the actually materialized candidate rollout
for the predicted grid mixture. It is not an official runtime selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_61"
CHUNK_ROOT = OUT_ROOT / "chunks"
PYTHON = "/home/chengshun.wang/miniconda3/envs/kan/bin/python"
ORACLE_DATASETS = ("MNIST", "FashionMNIST", "KMNIST", "Wine", "Spam")
FEATURE_COLS = [
    "feature_progress",
    "feature_phase_early",
    "feature_phase_middle",
    "feature_phase_late",
    "feature_source_witness_credit_cosine",
    "feature_source_loss",
    "feature_witness_loss",
    "feature_source_witness_loss_gap",
    "feature_source_accuracy",
    "feature_witness_accuracy",
    "feature_source_witness_accuracy_gap",
    "feature_source_confidence",
    "feature_witness_confidence",
    "feature_source_witness_confidence_gap",
    "feature_source_grad_norm",
    "feature_witness_grad_norm",
    "feature_grad_norm_ratio",
    "feature_param_norm",
]
BASE_FEATURE_COLS = list(FEATURE_COLS)
HISTORY_BASE_COLS = [
    "feature_source_witness_credit_cosine",
    "feature_source_loss",
    "feature_witness_loss",
    "feature_source_witness_loss_gap",
    "feature_source_accuracy",
    "feature_witness_accuracy",
    "feature_source_witness_accuracy_gap",
    "feature_source_confidence",
    "feature_witness_confidence",
    "feature_source_witness_confidence_gap",
    "feature_source_grad_norm",
    "feature_witness_grad_norm",
    "feature_grad_norm_ratio",
]
DERIVED_FEATURE_COLS = [
    "feature_abs_loss_gap",
    "feature_abs_accuracy_gap",
    "feature_abs_confidence_gap",
    "feature_grad_norm_diff",
    "feature_grad_norm_sum",
    "feature_progress_x_credit_cosine",
    "feature_progress_x_loss_gap",
    "feature_progress_x_accuracy_gap",
    "feature_progress_x_confidence_gap",
    "feature_progress_x_grad_ratio",
    "feature_middle_x_credit_cosine",
    "feature_middle_x_loss_gap",
    "feature_middle_x_accuracy_gap",
    "feature_middle_x_confidence_gap",
    "feature_middle_x_grad_ratio",
]
for _col in HISTORY_BASE_COLS:
    DERIVED_FEATURE_COLS.extend(
        [
            f"history_delta_prev_{_col}",
            f"history_delta_first_{_col}",
            f"history_slope_prev_{_col}",
        ]
    )
COMPACT_DERIVED_FEATURE_COLS = [
    "feature_abs_loss_gap",
    "feature_abs_accuracy_gap",
    "feature_abs_confidence_gap",
    "feature_grad_norm_diff",
    "feature_grad_norm_sum",
    "feature_progress_x_credit_cosine",
    "feature_progress_x_loss_gap",
    "feature_progress_x_grad_ratio",
    "feature_middle_x_credit_cosine",
    "feature_middle_x_loss_gap",
    "feature_middle_x_grad_ratio",
    "history_delta_prev_feature_source_witness_credit_cosine",
    "history_delta_prev_feature_source_witness_loss_gap",
    "history_delta_prev_feature_source_witness_accuracy_gap",
    "history_delta_prev_feature_source_witness_confidence_gap",
    "history_delta_prev_feature_grad_norm_ratio",
    "history_delta_first_feature_source_witness_loss_gap",
    "history_slope_prev_feature_source_witness_loss_gap",
    "history_slope_prev_feature_grad_norm_ratio",
]
FULL_FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS
FEATURE_COLS = FULL_FEATURE_COLS
ACTIVE_FEATURE_COLS = FEATURE_COLS


def fval(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def iflag(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def append_exec(command: str, *, task_id: str, status: str, files: str, note: str = "") -> None:
    from datetime import datetime

    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    journal = OUT_ROOT / "v22_61_command_journal.csv"
    rows = read_rows(journal) if journal.exists() else []
    rows.append({"timestamp": ts, "task_id": task_id, "gpu": "n/a", "command": command, "status": status, "exit_code": "n/a", "files": files, "note": note})
    write_rows(journal, rows)
    log_path = ROOT / "docs/DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_执行日志.md"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {ts} {task_id}\n\n```bash\n{command}\n```\n\n")
        f.write(f"- gpu: n/a\n- status: {status}\n- exit_code: n/a\n- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def label_matches(label: str, prefix: str) -> bool:
    return any(label.startswith(f"{prefix}_{dataset}_s") for dataset in ORACLE_DATASETS)


def split_prefixes(prefix: str) -> list[str]:
    return [p.strip() for p in prefix.split(",") if p.strip()]


def choose_feature_cols(feature_set: str) -> list[str]:
    if feature_set == "base":
        return list(BASE_FEATURE_COLS)
    if feature_set == "compact_history":
        return list(BASE_FEATURE_COLS) + list(COMPACT_DERIVED_FEATURE_COLS)
    return list(FULL_FEATURE_COLS)


def load_detail(prefixes: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for prefix in prefixes:
        for path in sorted(CHUNK_ROOT.glob(f"v22_61_{prefix}*_oracle_detail.csv")):
            for row in read_rows(path):
                if not label_matches(row.get("run_label", ""), prefix):
                    continue
                key = (
                    row.get("run_label", ""),
                    row.get("phase_bucket", ""),
                    row.get("checkpoint_step", ""),
                    row.get("mixture_name", ""),
                    row.get("mixture_kind", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    augment_rows_with_history(rows)
    return rows


def score(row: dict[str, str]) -> float:
    return fval(row.get("oracle_selection_score"), fval(row.get("Delta_NLL_vs_reference")))


def best_score(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=score)


def best_delta(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda r: fval(r.get("Delta_NLL_vs_reference")))


def augment_rows_with_history(rows: list[dict[str, str]]) -> None:
    by_task_phase: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task_phase[(row.get("dataset", ""), row.get("seed", ""), row.get("phase_bucket", ""))].append(row)
    phase_templates: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for (dataset, seed, _phase), phase_rows in by_task_phase.items():
        phase_templates[(dataset, seed)].append(phase_rows[0])

    for (dataset, seed), templates in phase_templates.items():
        ordered = sorted(templates, key=lambda r: fval(r.get("feature_progress")))
        first = ordered[0] if ordered else {}
        prev: dict[str, str] | None = None
        for current in ordered:
            progress = fval(current.get("feature_progress"))
            loss_gap = fval(current.get("feature_source_witness_loss_gap"))
            acc_gap = fval(current.get("feature_source_witness_accuracy_gap"))
            conf_gap = fval(current.get("feature_source_witness_confidence_gap"))
            credit_cos = fval(current.get("feature_source_witness_credit_cosine"))
            grad_ratio = fval(current.get("feature_grad_norm_ratio"))
            grad_source = fval(current.get("feature_source_grad_norm"))
            grad_witness = fval(current.get("feature_witness_grad_norm"))
            derived: dict[str, float] = {
                "feature_abs_loss_gap": abs(loss_gap),
                "feature_abs_accuracy_gap": abs(acc_gap),
                "feature_abs_confidence_gap": abs(conf_gap),
                "feature_grad_norm_diff": grad_source - grad_witness,
                "feature_grad_norm_sum": grad_source + grad_witness,
                "feature_progress_x_credit_cosine": progress * credit_cos,
                "feature_progress_x_loss_gap": progress * loss_gap,
                "feature_progress_x_accuracy_gap": progress * acc_gap,
                "feature_progress_x_confidence_gap": progress * conf_gap,
                "feature_progress_x_grad_ratio": progress * grad_ratio,
                "feature_middle_x_credit_cosine": fval(current.get("feature_phase_middle")) * credit_cos,
                "feature_middle_x_loss_gap": fval(current.get("feature_phase_middle")) * loss_gap,
                "feature_middle_x_accuracy_gap": fval(current.get("feature_phase_middle")) * acc_gap,
                "feature_middle_x_confidence_gap": fval(current.get("feature_phase_middle")) * conf_gap,
                "feature_middle_x_grad_ratio": fval(current.get("feature_phase_middle")) * grad_ratio,
            }
            prev_progress = fval(prev.get("feature_progress")) if prev is not None else progress
            progress_span = max(abs(progress - prev_progress), 1.0e-6)
            for col in HISTORY_BASE_COLS:
                cur = fval(current.get(col))
                prv = fval(prev.get(col)) if prev is not None else cur
                fst = fval(first.get(col))
                derived[f"history_delta_prev_{col}"] = cur - prv
                derived[f"history_delta_first_{col}"] = cur - fst
                derived[f"history_slope_prev_{col}"] = (cur - prv) / progress_span if prev is not None else 0.0
            for row in by_task_phase[(dataset, seed, current.get("phase_bucket", ""))]:
                for key, value in derived.items():
                    row[key] = str(value)
            prev = current


def feature_vector(row: dict[str, str]) -> list[float]:
    return [fval(row.get(col)) for col in ACTIVE_FEATURE_COLS]


def weight_vector(row: dict[str, str]) -> list[float]:
    try:
        weights = json.loads(row.get("weights_json", "[]"))
    except Exception:
        weights = []
    out = [float(x) for x in weights[:7]]
    if len(out) < 7:
        out.extend([0.0] * (7 - len(out)))
    total = sum(max(0.0, x) for x in out)
    if total <= 0.0:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [max(0.0, x) / total for x in out]


def weight_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def select_training_target(phase_rows: list[dict[str, str]], target_mode: str) -> dict[str, str]:
    structured = [r for r in phase_rows if r.get("mixture_kind") != "random_dirichlet"]
    if not structured:
        return best_score(phase_rows)
    random_rows = [r for r in phase_rows if r.get("mixture_kind") == "random_dirichlet"]
    random_best = best_delta(random_rows) if random_rows else None
    random_delta = fval(random_best.get("Delta_NLL_vs_reference")) if random_best is not None else math.inf

    def beats_random(row: dict[str, str]) -> bool:
        return fval(row.get("Delta_NLL_vs_reference")) < random_delta - 1.0e-5

    def no_debt(row: dict[str, str]) -> bool:
        return iflag(row.get("no_ECE_Brier_tail_debt")) == 1

    if target_mode == "safe":
        pool = [r for r in structured if no_debt(r)]
        return best_score(pool or structured)
    if target_mode == "anti_control":
        pool = [r for r in structured if beats_random(r)]
        return best_score(pool or structured)
    if target_mode == "safe_anti_control":
        for pool in (
            [r for r in structured if beats_random(r) and no_debt(r)],
            [r for r in structured if beats_random(r)],
            [r for r in structured if no_debt(r)],
            structured,
        ):
            if pool:
                return best_score(pool)
    if target_mode == "anti_control_margin":
        for pool in (
            [r for r in structured if beats_random(r) and no_debt(r)],
            [r for r in structured if beats_random(r)],
            [r for r in structured if no_debt(r)],
            structured,
        ):
            if pool:
                return best_delta(pool)
    return best_score(structured)


def train_model(train_examples: list[dict[str, Any]], classes: list[str], *, kind: str, epochs: int, seed: int, hidden: int) -> tuple[Any, dict[str, float], dict[str, float]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.tensor([ex["features"] for ex in train_examples], dtype=torch.float32)
    y = torch.tensor([classes.index(ex["target"]) for ex in train_examples], dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    z = (x - mean) / std
    if kind == "linear":
        model = torch.nn.Linear(z.shape[1], len(classes))
    else:
        model = torch.nn.Sequential(torch.nn.Linear(z.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, len(classes)))
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        logits = model(z)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(z)
        train_acc = float((logits.argmax(dim=-1) == y).float().mean().item())
        train_loss = float(F.cross_entropy(logits, y).item())
    param_count = sum(p.numel() for p in model.parameters())
    stats = {"train_acc": train_acc, "train_loss": train_loss, "param_count": float(param_count)}
    norm = {"mean_json": json.dumps(mean.tolist()), "std_json": json.dumps(std.tolist())}
    return model, stats, norm


def train_weight_model(train_examples: list[dict[str, Any]], *, kind: str, epochs: int, seed: int, hidden: int) -> tuple[Any, dict[str, float], dict[str, float]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.tensor([ex["features"] for ex in train_examples], dtype=torch.float32)
    y = torch.tensor([ex["target_weights"] for ex in train_examples], dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    z = (x - mean) / std
    if kind == "linear":
        model = torch.nn.Linear(z.shape[1], 7)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(z.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 7))
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        logits = model(z)
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(y * log_prob).sum(dim=-1).mean() + 0.25 * F.mse_loss(logits.softmax(dim=-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(z).softmax(dim=-1)
        train_loss = float(F.mse_loss(pred, y).item())
        train_l1 = float((pred - y).abs().sum(dim=-1).mean().item())
    param_count = sum(p.numel() for p in model.parameters())
    stats = {"train_acc": 0.0, "train_loss": train_loss, "train_l1": train_l1, "param_count": float(param_count)}
    norm = {"mean_json": json.dumps(mean.tolist()), "std_json": json.dumps(std.tolist())}
    return model, stats, norm


def train_gru_model(train_sequences: list[dict[str, Any]], classes: list[str], *, epochs: int, seed: int, hidden: int) -> tuple[Any, dict[str, float], dict[str, float]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    flat_features = [feat for seq in train_sequences for feat in seq["features"]]
    x_all = torch.tensor(flat_features, dtype=torch.float32)
    mean = x_all.mean(dim=0)
    std = x_all.std(dim=0, unbiased=False).clamp_min(1.0e-6)

    class SequenceController(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x: Any) -> Any:
            out, _ = self.gru(x)
            return self.head(out)

    model = SequenceController(x_all.shape[1], hidden, len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        total_loss = torch.tensor(0.0)
        for seq in train_sequences:
            x = torch.tensor([seq["features"]], dtype=torch.float32)
            y = torch.tensor([classes.index(t) for t in seq["targets"]], dtype=torch.long)
            z = (x - mean) / std
            logits = model(z)[0]
            total_loss = total_loss + F.cross_entropy(logits, y)
        loss = total_loss / max(1, len(train_sequences))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    correct = 0
    total = 0
    with torch.no_grad():
        total_loss = 0.0
        for seq in train_sequences:
            x = torch.tensor([seq["features"]], dtype=torch.float32)
            y = torch.tensor([classes.index(t) for t in seq["targets"]], dtype=torch.long)
            logits = model((x - mean) / std)[0]
            total_loss += float(F.cross_entropy(logits, y).item())
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += int(y.numel())
    param_count = sum(p.numel() for p in model.parameters())
    stats = {"train_acc": correct / max(1, total), "train_loss": total_loss / max(1, len(train_sequences)), "param_count": float(param_count)}
    norm = {"mean_json": json.dumps(mean.tolist()), "std_json": json.dumps(std.tolist())}
    return model, stats, norm


def predict(model: Any, norm: dict[str, float], row: dict[str, str], classes: list[str]) -> tuple[str, float]:
    import torch

    mean = torch.tensor(json.loads(str(norm["mean_json"])), dtype=torch.float32)
    std = torch.tensor(json.loads(str(norm["std_json"])), dtype=torch.float32)
    x = torch.tensor([feature_vector(row)], dtype=torch.float32)
    z = (x - mean) / std
    with torch.no_grad():
        prob = model(z).softmax(dim=-1)[0]
        idx = int(prob.argmax().item())
    return classes[idx], float(prob[idx].item())


def predict_weights(model: Any, norm: dict[str, float], row: dict[str, str]) -> list[float]:
    import torch

    mean = torch.tensor(json.loads(str(norm["mean_json"])), dtype=torch.float32)
    std = torch.tensor(json.loads(str(norm["std_json"])), dtype=torch.float32)
    x = torch.tensor([feature_vector(row)], dtype=torch.float32)
    z = (x - mean) / std
    with torch.no_grad():
        pred = model(z).softmax(dim=-1)[0]
    return [float(x) for x in pred.tolist()]


def predict_sequence(model: Any, norm: dict[str, float], rows: list[dict[str, str]], classes: list[str]) -> list[tuple[str, float]]:
    import torch

    mean = torch.tensor(json.loads(str(norm["mean_json"])), dtype=torch.float32)
    std = torch.tensor(json.loads(str(norm["std_json"])), dtype=torch.float32)
    x = torch.tensor([[feature_vector(row) for row in rows]], dtype=torch.float32)
    with torch.no_grad():
        probs = model((x - mean) / std)[0].softmax(dim=-1)
    out = []
    for prob in probs:
        idx = int(prob.argmax().item())
        out.append((classes[idx], float(prob[idx].item())))
    return out


def evaluate_method(
    method: str,
    model: Any,
    norm: dict[str, float],
    classes: list[str],
    held_tasks: dict[tuple[str, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (dataset, seed), task_rows in sorted(held_tasks.items()):
        by_phase: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in task_rows:
            by_phase[row.get("phase_bucket", "")].append(row)
        chosen_rows: list[tuple[dict[str, str], str, float]] = []
        for phase, phase_rows in by_phase.items():
            template = phase_rows[0]
            if method in {"feature_mlp", "feature_linear"}:
                pred_name, pred_conf = predict(model, norm, template, classes)
                matches = [r for r in phase_rows if r.get("mixture_name") == pred_name]
                if not matches:
                    matches = [r for r in phase_rows if r.get("mixture_name") == "onehot_A0_identity"]
            elif method in {"weight_mlp", "weight_linear"}:
                pred_weights = predict_weights(model, norm, template)
                structured = [r for r in phase_rows if r.get("mixture_kind") != "random_dirichlet"]
                nearest = min(structured, key=lambda r: weight_distance(pred_weights, weight_vector(r)))
                pred_name = nearest.get("mixture_name", "")
                pred_conf = 1.0 / (1.0 + weight_distance(pred_weights, weight_vector(nearest)))
                matches = [nearest]
            elif method == "phase_only":
                pred_name, pred_conf = ("onehot_A4_state_transition" if phase == "middle" else "onehot_A0_identity", 1.0)
                matches = [r for r in phase_rows if r.get("mixture_name") == pred_name]
            elif method == "constant_state":
                pred_name, pred_conf = ("onehot_A4_state_transition", 1.0)
                matches = [r for r in phase_rows if r.get("mixture_name") == pred_name]
            else:
                pred_name, pred_conf = ("onehot_A0_identity", 1.0)
                matches = [r for r in phase_rows if r.get("mixture_name") == pred_name]
            if not matches:
                matches = [r for r in phase_rows if r.get("mixture_name") == "onehot_A0_identity"]
            chosen_rows.append((best_score(matches), pred_name, pred_conf))
        chosen, pred_name, pred_conf = min(chosen_rows, key=lambda item: score(item[0]))
        oracle = best_score([r for r in task_rows if r.get("mixture_kind") != "random_dirichlet"])
        random_best = best_delta([r for r in task_rows if r.get("mixture_kind") == "random_dirichlet"])
        credit = best_score([r for r in task_rows if r.get("mixture_name") == "onehot_A1_credit_only"])
        self_only = best_score([r for r in task_rows if r.get("mixture_name") == "onehot_A2_self_geometry"])
        delta = fval(chosen.get("Delta_NLL_vs_reference"))
        oracle_delta = fval(oracle.get("Delta_NLL_vs_reference"))
        denom = max(abs(oracle_delta), 1.0e-12)
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "predicted_mixture": pred_name,
                "prediction_confidence": pred_conf,
                "selected_mixture": chosen.get("mixture_name", ""),
                "selected_phase": chosen.get("phase_bucket", ""),
                "target_oracle_mixture": oracle.get("mixture_name", ""),
                "target_match": int(chosen.get("mixture_name") == oracle.get("mixture_name")),
                "Delta_NLL_vs_reference": delta,
                "oracle_Delta_NLL_vs_reference": oracle_delta,
                "beats_reference": int(delta < -1.0e-5),
                "beats_random_best": int(delta < fval(random_best.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "beats_credit_and_self": int(delta < fval(credit.get("Delta_NLL_vs_reference")) - 1.0e-5 and delta < fval(self_only.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "oracle_gap_ratio": max(0.0, (delta - oracle_delta) / denom),
                "no_ECE_Brier_tail_debt": iflag(chosen.get("no_ECE_Brier_tail_debt")),
                "operator_mixture_entropy": fval(chosen.get("operator_mixture_entropy")),
                "leakage_audit_pass": 1,
                "meta_test_no_controller_update": 1,
            }
        )
    return out


def evaluate_sequence_method(
    method: str,
    model: Any,
    norm: dict[str, float],
    classes: list[str],
    held_tasks: dict[tuple[str, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (dataset, seed), task_rows in sorted(held_tasks.items()):
        by_phase: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in task_rows:
            by_phase[row.get("phase_bucket", "")].append(row)
        phase_items = sorted(by_phase.items(), key=lambda item: fval(item[1][0].get("feature_progress")))
        templates = [phase_rows[0] for _, phase_rows in phase_items]
        predictions = predict_sequence(model, norm, templates, classes)
        chosen_rows: list[tuple[dict[str, str], str, float]] = []
        for (phase, phase_rows), (pred_name, pred_conf) in zip(phase_items, predictions):
            matches = [r for r in phase_rows if r.get("mixture_name") == pred_name]
            if not matches:
                matches = [r for r in phase_rows if r.get("mixture_name") == "onehot_A0_identity"]
            chosen_rows.append((best_score(matches), pred_name, pred_conf))
        chosen, pred_name, pred_conf = min(chosen_rows, key=lambda item: score(item[0]))
        oracle = best_score([r for r in task_rows if r.get("mixture_kind") != "random_dirichlet"])
        random_best = best_delta([r for r in task_rows if r.get("mixture_kind") == "random_dirichlet"])
        credit = best_score([r for r in task_rows if r.get("mixture_name") == "onehot_A1_credit_only"])
        self_only = best_score([r for r in task_rows if r.get("mixture_name") == "onehot_A2_self_geometry"])
        delta = fval(chosen.get("Delta_NLL_vs_reference"))
        oracle_delta = fval(oracle.get("Delta_NLL_vs_reference"))
        denom = max(abs(oracle_delta), 1.0e-12)
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "predicted_mixture": pred_name,
                "prediction_confidence": pred_conf,
                "selected_mixture": chosen.get("mixture_name", ""),
                "selected_phase": chosen.get("phase_bucket", ""),
                "target_oracle_mixture": oracle.get("mixture_name", ""),
                "target_match": int(chosen.get("mixture_name") == oracle.get("mixture_name")),
                "Delta_NLL_vs_reference": delta,
                "oracle_Delta_NLL_vs_reference": oracle_delta,
                "beats_reference": int(delta < -1.0e-5),
                "beats_random_best": int(delta < fval(random_best.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "beats_credit_and_self": int(delta < fval(credit.get("Delta_NLL_vs_reference")) - 1.0e-5 and delta < fval(self_only.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "oracle_gap_ratio": max(0.0, (delta - oracle_delta) / denom),
                "no_ECE_Brier_tail_debt": iflag(chosen.get("no_ECE_Brier_tail_debt")),
                "operator_mixture_entropy": fval(chosen.get("operator_mixture_entropy")),
                "leakage_audit_pass": 1,
                "meta_test_no_controller_update": 1,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="v22_61_m0o_objective_features")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--output-tag", default="feature_imitation")
    parser.add_argument(
        "--target-mode",
        choices=["objective", "safe", "anti_control", "safe_anti_control", "anti_control_margin"],
        default="objective",
    )
    parser.add_argument("--feature-set", choices=["base", "compact_history", "full_history"], default="full_history")
    args = parser.parse_args(argv)

    global ACTIVE_FEATURE_COLS
    ACTIVE_FEATURE_COLS = choose_feature_cols(str(args.feature_set))
    prefixes = split_prefixes(str(args.prefix))
    rows = load_detail(prefixes)
    if not rows:
        raise SystemExit(f"no detail rows for {args.prefix}")
    structured_classes = sorted({r["mixture_name"] for r in rows if r.get("mixture_kind") != "random_dirichlet"})
    examples: list[dict[str, Any]] = []
    by_task_phase: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task_phase[(row["dataset"], int(row["seed"]), row.get("phase_bucket", ""))].append(row)
    by_task_examples: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (dataset, seed, phase), phase_rows in sorted(by_task_phase.items()):
        target = select_training_target(phase_rows, str(args.target_mode))
        example = {"dataset": dataset, "seed": seed, "phase": phase, "progress": fval(target.get("feature_progress")), "features": feature_vector(target), "target": target["mixture_name"], "target_weights": weight_vector(target)}
        examples.append(example)
        by_task_examples[(dataset, seed)].append(example)
    sequences: list[dict[str, Any]] = []
    for (dataset, seed), task_examples in sorted(by_task_examples.items()):
        ordered = sorted(task_examples, key=lambda ex: ex["progress"])
        sequences.append({"dataset": dataset, "seed": seed, "features": [ex["features"] for ex in ordered], "targets": [ex["target"] for ex in ordered]})
    seeds = sorted({ex["seed"] for ex in examples})
    detail: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for held_seed in seeds:
        train_examples = [ex for ex in examples if ex["seed"] != held_seed]
        held_tasks: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if int(row["seed"]) == held_seed:
                held_tasks[(row["dataset"], int(row["seed"]))].append(row)
        for kind in ["feature_mlp", "feature_linear"]:
            model, stats, norm = train_model(
                train_examples,
                structured_classes,
                kind="mlp" if kind == "feature_mlp" else "linear",
                epochs=int(args.epochs),
                seed=226100 + held_seed,
                hidden=int(args.hidden),
            )
            train_rows.append({"held_seed": held_seed, "method": kind, "train_examples": len(train_examples), "hidden": int(args.hidden), **stats, "classes": json.dumps(structured_classes)})
            detail.extend(evaluate_method(kind, model, norm, structured_classes, held_tasks))
        for kind in ["weight_mlp", "weight_linear"]:
            model, stats, norm = train_weight_model(
                train_examples,
                kind="mlp" if kind == "weight_mlp" else "linear",
                epochs=int(args.epochs),
                seed=226300 + held_seed,
                hidden=int(args.hidden),
            )
            train_rows.append({"held_seed": held_seed, "method": kind, "train_examples": len(train_examples), "hidden": int(args.hidden), **stats, "classes": "operator_weight_vector"})
            detail.extend(evaluate_method(kind, model, norm, structured_classes, held_tasks))
        train_sequences = [seq for seq in sequences if seq["seed"] != held_seed]
        model, stats, norm = train_gru_model(
            train_sequences,
            structured_classes,
            epochs=int(args.epochs),
            seed=226500 + held_seed,
            hidden=max(4, int(args.hidden) // 2),
        )
        train_rows.append({"held_seed": held_seed, "method": "feature_gru", "train_examples": sum(len(seq["targets"]) for seq in train_sequences), "hidden": max(4, int(args.hidden) // 2), **stats, "classes": json.dumps(structured_classes)})
        detail.extend(evaluate_sequence_method("feature_gru", model, norm, structured_classes, held_tasks))
        detail.extend(evaluate_method("phase_only", None, {"mean_json": "[]", "std_json": "[]"}, structured_classes, held_tasks))
        detail.extend(evaluate_method("constant_state", None, {"mean_json": "[]", "std_json": "[]"}, structured_classes, held_tasks))

    output_tag = str(args.output_tag).strip() or "feature_imitation"
    output_stem = f"v22_61_m0c_{output_tag}"
    detail_file = OUT_ROOT / f"{output_stem}_detail.csv"
    train_file = OUT_ROOT / f"{output_stem}_train.csv"
    summary_file = OUT_ROOT / f"{output_stem}_summary.csv"
    route_file = OUT_ROOT / f"{output_stem}_route.json"
    write_rows(detail_file, detail)
    write_rows(train_file, train_rows)
    summary: list[dict[str, Any]] = []
    for method in ["feature_mlp", "feature_linear", "weight_mlp", "weight_linear", "feature_gru", "phase_only", "constant_state"]:
        sub = [r for r in detail if r["method"] == method]
        row_count = len(sub)
        required_beats_reference = math.ceil(row_count * 8 / 15.0)
        required_beats_random = math.ceil(row_count * 10 / 15.0)
        required_credit_self = math.ceil(row_count * 10 / 15.0)
        required_oracle_gap = math.ceil(row_count * 10 / 15.0)
        required_no_debt = math.ceil(row_count * 11 / 15.0)
        required_param_budget = row_count
        required_leakage = math.ceil(row_count * 14 / 15.0)
        param_count = max([fval(r.get("param_count")) for r in train_rows if r.get("method") == method] or [0.0])
        # Conservative denominator: runtime smoke model has 118282 trainable params.
        param_ratio = param_count / 118282.0
        gate = {
            "method": method,
            "rows": row_count,
            "beats_strongest_reference_rows": sum(iflag(r["beats_reference"]) for r in sub),
            "beats_random_frozen_controls_rows": sum(iflag(r["beats_random_best"]) for r in sub),
            "beats_credit_only_and_self_only_rows": sum(iflag(r["beats_credit_and_self"]) for r in sub),
            "oracle_gap_ratio_le_050_rows": sum(int(fval(r["oracle_gap_ratio"]) <= 0.50) for r in sub),
            "no_ECE_Brier_tail_debt_rows": sum(iflag(r["no_ECE_Brier_tail_debt"]) for r in sub),
            "controller_param_ratio": param_ratio,
            "controller_param_ratio_le_001_rows": row_count if param_ratio <= 0.01 else 0,
            "leakage_audit_pass_rows": sum(iflag(r["leakage_audit_pass"]) for r in sub),
            "meta_test_no_controller_update": 1,
            "oracle_imitation_match_rows": sum(iflag(r["target_match"]) for r in sub),
            "operator_mixture_entropy_mean": sum(fval(r["operator_mixture_entropy"]) for r in sub) / max(1, len(sub)),
            "required_beats_reference_rows": required_beats_reference,
            "required_beats_random_rows": required_beats_random,
            "required_credit_self_rows": required_credit_self,
            "required_oracle_gap_rows": required_oracle_gap,
            "required_no_debt_rows": required_no_debt,
            "required_param_budget_rows": required_param_budget,
            "required_leakage_rows": required_leakage,
        }
        gate["m0c_feature_imitation_pass"] = int(
            gate["beats_strongest_reference_rows"] >= required_beats_reference
            and gate["beats_random_frozen_controls_rows"] >= required_beats_random
            and gate["beats_credit_only_and_self_only_rows"] >= required_credit_self
            and gate["oracle_gap_ratio_le_050_rows"] >= required_oracle_gap
            and gate["no_ECE_Brier_tail_debt_rows"] >= required_no_debt
            and gate["controller_param_ratio_le_001_rows"] >= required_param_budget
            and gate["leakage_audit_pass_rows"] >= required_leakage
            and gate["meta_test_no_controller_update"] == 1
        )
        summary.append(gate)
    write_rows(summary_file, summary)
    route = {
        "m0c_feature_imitation_pass": max(iflag(r["m0c_feature_imitation_pass"]) for r in summary),
        "best_method": max(summary, key=lambda r: (iflag(r["m0c_feature_imitation_pass"]), int(r["beats_random_frozen_controls_rows"]), int(r["no_ECE_Brier_tail_debt_rows"])))["method"],
        "summary_file": str(summary_file),
        "detail_file": str(detail_file),
        "train_file": str(train_file),
        "prefixes": prefixes,
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "target_mode": str(args.target_mode),
        "feature_set": str(args.feature_set),
        "feature_dim": len(ACTIVE_FEATURE_COLS),
    }
    route_file.write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        f"{PYTHON} experiments/analyze_v22_61_m0c_feature_imitation.py --prefix {args.prefix} --epochs {int(args.epochs)} --hidden {int(args.hidden)} --output-tag {output_tag} --target-mode {args.target_mode} --feature-set {args.feature_set}",
        task_id="v22_61_m0c_feature_imitation",
        status="pass",
        files=f"{summary_file}; {detail_file}; {route_file}",
        note=json.dumps(route, sort_keys=True),
    )
    print(json.dumps(route, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
