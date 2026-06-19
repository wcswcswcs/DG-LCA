#!/usr/bin/env python3
"""DG-KAN v22.49A broader related-work map runner.

The v22.49A document is an execution guide rather than a narrow algorithm
specification.  This runner follows its staged recommendation:

* Stage 0 re-slices historical artifacts into H1-H8 explanation variables.
* Stage 1 runs the three cheap complementary sanity tests: H2, H6, H7.

All Stage 1 rows use real cached datasets, train-only cohort construction, and
matched controls.  The outputs are deliberately labelled as sanity evidence;
they are not promoted into official hard-task DG-KAN success claims.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import py_compile
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from experiments import dgkan_core as data_core


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_49A"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.49A_BroaderRelatedWorkMap_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.49A_BroaderRelatedWorkMap_实验结果复盘.md"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.49A_BroaderRelatedWorkMap_相关研究扩展附录.md"

METHODS = [
    "H0_adamw_optimizer",
    "H1_population_mean_gradient",
    "H2_support_native_ema",
    "H2_random_support_control",
    "H6_layerwise_target",
    "H6_shuffled_layer_target_control",
    "H7_cohort_surgery",
    "H7_same_span_random_control",
]

REPAIR_METHODS = [
    "H1_population_mean_gradient",
    "H6_layerwise_cvar_target",
    "H6_shuffled_cvar_target_control",
    "H6_layerwise_cvar_target_train_exact_accept",
    "H6_shuffled_cvar_target_train_exact_accept_control",
    "H6_layerwise_cvar_population_blend_exact_accept",
    "H6_shuffled_cvar_population_blend_exact_accept_control",
    "H6_layerwise_cvar_population_blend_primal_dual",
    "H6_shuffled_cvar_population_blend_primal_dual_control",
    "H7_cohort_surgery_debt_orthogonal",
    "H7_same_span_random_debt_orthogonal_control",
    "H7_cohort_surgery_debt_barrier",
    "H7_same_span_random_debt_barrier_control",
    "H7_cohort_surgery_train_debt_accept",
    "H7_same_span_random_train_debt_accept_control",
    "H7_cohort_surgery_train_exact_debt_accept",
    "H7_same_span_random_train_exact_debt_accept_control",
    "H7_cohort_surgery_persistent_guard_accept",
    "H7_same_span_random_persistent_guard_accept_control",
    "H7_cohort_surgery_held_debt_accept",
    "H7_same_span_random_held_debt_accept_control",
    "H7_cohort_surgery_debt_orthogonal_held_temp_probe",
    "H7_same_span_random_debt_orthogonal_held_temp_probe_control",
    "H7_cohort_surgery_blend_debt_orthogonal",
    "H7_same_span_random_blend_debt_orthogonal_control",
    "H7_cohort_surgery_primal_dual_safety",
    "H7_same_span_random_primal_dual_safety_control",
    "H7_cohort_surgery_primal_dual_train_temp",
    "H7_same_span_random_primal_dual_train_temp_control",
    "H7_cohort_surgery_primal_dual_train_soften_temp",
    "H7_same_span_random_primal_dual_train_soften_temp_control",
    "H7_cohort_surgery_primal_dual_soft_ece",
    "H7_same_span_random_primal_dual_soft_ece_control",
    "H7_cohort_surgery_primal_dual_soft_acc_ece",
    "H7_same_span_random_primal_dual_soft_acc_ece_control",
    "H7_cohort_surgery_primal_dual_worst_guard",
    "H7_same_span_random_primal_dual_worst_guard_control",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.49A BroaderRelatedWorkMap 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只写真实命令、输入、输出、GPU、状态、blocker 和修复尝试；"
            "Stage 0 历史重算必须标明 artifact 来源；Stage 1 sanity tests 不冒充 official hard-task 结论。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.49A BroaderRelatedWorkMap 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact 和命令输出；缺失/失败必须写成缺失/失败；"
            "不编造数据，不把 proxy 写成 causal proof。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


def split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(row) for row in rows]
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in data:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        if not fieldnames:
            fieldnames = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fval(value: Any, default: float | None = None) -> float | None:
    if value in {"", None}:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def iflag(value: Any) -> int:
    if value in {1, "1", True, "true", "True", "yes", "pass"}:
        return 1
    return 0


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(clean) if clean else None


def median(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(clean) if clean else None


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "_无行。_\n"
    shown = rows[:limit]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown:
        cells = []
        for col in columns:
            text = str(row.get(col, ""))
            if len(text) > 96:
                text = text[:93] + "..."
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        out.append(f"\n_只显示前 {limit} 行，共 {len(rows)} 行。_")
    return "\n".join(out) + "\n"


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: Any = "n/a",
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
    journal = read_rows(OUT_ROOT / "v22_49A_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(
        OUT_ROOT / "v22_49A_command_journal.csv",
        journal,
        ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"],
    )
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + command + "\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def run_code_truth_gate() -> dict[str, Any]:
    ensure_out()
    script = ROOT / "experiments/run_v22_49A_broader_related_work_map.py"
    rows: list[dict[str, Any]] = []
    status = "pass"
    note = ""
    try:
        py_compile.compile(str(script), doraise=True)
        rows.append({"file": str(script.relative_to(ROOT)), "compile": "pass", "error": ""})
    except Exception as exc:  # pragma: no cover - runtime audit path
        status = "fail"
        note = repr(exc)
        rows.append({"file": str(script.relative_to(ROOT)), "compile": "fail", "error": repr(exc)})
    write_rows(OUT_ROOT / "v22_49A_code_truth_gate.csv", rows, ["file", "compile", "error"])
    append_exec(
        "py_compile experiments/run_v22_49A_broader_related_work_map.py",
        task_id="code_truth_gate",
        status=status,
        gpu="cpu",
        files="results/v22_49A/v22_49A_code_truth_gate.csv",
        note=note or "compile check for new v22.49A runner",
    )
    return {"status": status, "rows": len(rows)}


def count_rows(rows: list[dict[str, str]], field: str, predicate: Any) -> int:
    out = 0
    for row in rows:
        try:
            if predicate(row.get(field)):
                out += 1
        except Exception:
            pass
    return out


def stage0_historical_reslice() -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []

    v46_route = read_json(ROOT / "results/v22_46/v22_46_final_route.json")
    rows.append(
        {
            "hypothesis": "H1",
            "name": "Population-transfer residual flow",
            "source_artifact": "results/v22_46/v22_46_final_route.json",
            "artifact_exists": int(bool(v46_route)),
            "key_rows": v46_route.get("direction_positive_rows", ""),
            "support_or_candidate_rows": v46_route.get("support_positive_rows", ""),
            "control_or_gate_rows": v46_route.get("direction_positive_route_gate_rows", ""),
            "route": v46_route.get("final_route", "missing"),
            "promotion_allowed": int(str(v46_route.get("final_route", "")).endswith("Opened")),
            "claim_limit": "historical final route only; no new training in Stage 0",
            "evidence_summary": v46_route.get("reason", ""),
        }
    )

    support_rows = read_rows(ROOT / "results/v22_46/v22_46_repair_support_direction_decomposition.csv")
    support_positive = count_rows(support_rows, "tau_support", lambda x: (fval(x, 0.0) or 0.0) > 0.0)
    direction_positive = count_rows(support_rows, "tau_direction", lambda x: (fval(x, 0.0) or 0.0) > 0.0)
    no_debt = sum(iflag(r.get("no_debt")) for r in support_rows)
    rows.append(
        {
            "hypothesis": "H2",
            "name": "Support-native geometric optimizer",
            "source_artifact": "results/v22_46/v22_46_repair_support_direction_decomposition.csv",
            "artifact_exists": int(bool(support_rows)),
            "key_rows": len(support_rows),
            "support_or_candidate_rows": support_positive,
            "control_or_gate_rows": direction_positive,
            "route": v46_route.get("final_route", "missing"),
            "promotion_allowed": 0,
            "claim_limit": "support-positive rows can explain support-native optimizer, not residual signal FU",
            "evidence_summary": f"support_positive={support_positive}; direction_positive={direction_positive}; no_debt={no_debt}",
        }
    )

    b1_route = read_json(ROOT / "results/v22_46b/v22_46b_b1_debt_gradient_route.json")
    fused_route = read_json(ROOT / "results/v22_46b/v22_46b_fused_controller_route.json")
    rows.append(
        {
            "hypothesis": "H3",
            "name": "Safety-native mirror/constrained flow",
            "source_artifact": "results/v22_46b/v22_46b_b1_debt_gradient_route.json; results/v22_46b/v22_46b_fused_controller_route.json",
            "artifact_exists": int(bool(b1_route) or bool(fused_route)),
            "key_rows": b1_route.get("rows", ""),
            "support_or_candidate_rows": b1_route.get("debt_orthogonal_route_gate_rows", ""),
            "control_or_gate_rows": fused_route.get("fused_route_gate_rows", ""),
            "route": b1_route.get("route", "missing"),
            "promotion_allowed": int(bool(b1_route.get("promotion_allowed"))),
            "claim_limit": "v22.46b paired probes; not a full primal-dual Bregman proof",
            "evidence_summary": b1_route.get("reason", ""),
        }
    )

    m4_summary = read_json(ROOT / "results/v22_45E/v22_45E_M4_slow_signal_diagnostic_summary.json")
    m4_rows = read_rows(ROOT / "results/v22_45E/v22_45E_m4_trajectory_matrix.csv")
    rows.append(
        {
            "hypothesis": "H4",
            "name": "Temporal reservoir-to-signal flow",
            "source_artifact": "results/v22_45E/v22_45E_M4_slow_signal_diagnostic_summary.json; results/v22_45E/v22_45E_m4_trajectory_matrix.csv",
            "artifact_exists": int(bool(m4_summary) or bool(m4_rows)),
            "key_rows": m4_summary.get("rows", len(m4_rows) if m4_rows else ""),
            "support_or_candidate_rows": m4_summary.get("slow_signal_positive_rows", ""),
            "control_or_gate_rows": m4_summary.get("official_ready_groups", ""),
            "route": m4_summary.get("route", "diagnostic_or_missing"),
            "promotion_allowed": int(bool(m4_summary.get("promotion_allowed"))),
            "claim_limit": "temporal/modular diagnostics only unless delayed regime gates are passed",
            "evidence_summary": m4_summary.get("reason", ""),
        }
    )

    kan_rows = read_rows(ROOT / "results/v22_46/v22_46_repair_KAN_vs_MLP_matched_support_matrix.csv")
    true_kan = sum(iflag(r.get("TrueKANGain")) or iflag(r.get("BothGain")) for r in kan_rows)
    beats_mlp = sum(iflag(r.get("KAN_beats_MLP_matched_support")) for r in kan_rows)
    rows.append(
        {
            "hypothesis": "H5",
            "name": "KAN functional carrier",
            "source_artifact": "results/v22_46/v22_46_repair_KAN_vs_MLP_matched_support_matrix.csv",
            "artifact_exists": int(bool(kan_rows)),
            "key_rows": len(kan_rows),
            "support_or_candidate_rows": beats_mlp,
            "control_or_gate_rows": true_kan,
            "route": v46_route.get("KAN_variant_repair_route", "missing"),
            "promotion_allowed": int(bool(v46_route.get("KAN_variant_repair_promotion_allowed"))),
            "claim_limit": "must beat MLP matched functional support; raw KAN internal value is insufficient",
            "evidence_summary": f"KAN_beats_MLP_matched_support={beats_mlp}; TrueKANGain_or_BothGain={true_kan}",
        }
    )

    rows.append(
        {
            "hypothesis": "H6",
            "name": "Layerwise local target flow",
            "source_artifact": "v22.49A Stage 0 historical scan",
            "artifact_exists": 0,
            "key_rows": 0,
            "support_or_candidate_rows": 0,
            "control_or_gate_rows": 0,
            "route": "not_available_before_v22_49A_stage1",
            "promotion_allowed": 0,
            "claim_limit": "no prior direct artifact found; Stage 1 sanity test required",
            "evidence_summary": "historical docs mention layerwise diagnostics but no matched H6 local-target run artifact was found",
        }
    )
    rows.append(
        {
            "hypothesis": "H7",
            "name": "Cohort gradient surgery flow",
            "source_artifact": "v22.49A Stage 0 historical scan",
            "artifact_exists": 0,
            "key_rows": 0,
            "support_or_candidate_rows": 0,
            "control_or_gate_rows": 0,
            "route": "not_available_before_v22_49A_stage1",
            "promotion_allowed": 0,
            "claim_limit": "no prior direct cohort-surgery FU artifact found; Stage 1 sanity test required",
            "evidence_summary": "existing cohort influence artifacts are diagnostics, not PCGrad/CAGrad-style flow runs",
        }
    )

    m7_summary = read_json(ROOT / "results/v22_45E/v22_45E_m7_trajectory_summary.json")
    rows.append(
        {
            "hypothesis": "H8",
            "name": "Memory-quotient continual flow",
            "source_artifact": "results/v22_45E/v22_45E_m7_trajectory_summary.json",
            "artifact_exists": int(bool(m7_summary)),
            "key_rows": m7_summary.get("rows", ""),
            "support_or_candidate_rows": m7_summary.get("memory_projected_opened_groups", ""),
            "control_or_gate_rows": m7_summary.get("official_ready_groups", ""),
            "route": m7_summary.get("route", "missing"),
            "promotion_allowed": int(bool(m7_summary.get("promotion_allowed"))),
            "claim_limit": m7_summary.get("claim_limit", "continual diagnostic only unless memory controls fail"),
            "evidence_summary": f"relative_forgetting_reduction_mean={m7_summary.get('relative_forgetting_reduction_mean', '')}",
        }
    )

    write_rows(OUT_ROOT / "v22_49A_stage0_historical_explanation_matrix.csv", rows)
    append_exec(
        "stage0_historical_reslice",
        task_id="stage0_historical_reslice",
        status="pass",
        gpu="cpu",
        files="results/v22_49A/v22_49A_stage0_historical_explanation_matrix.csv",
        note="read-only scan of v22.45E/v22.46/v22.46b artifacts into H1-H8 explanation variables",
    )
    return {"rows": len(rows), "status": "pass"}


class Stage1MLP:  # placeholder for static type checkers; real class is local to avoid importing torch at module load
    pass


def torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_bundle_tensors(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    dataset_for_loader = {"FashionMNIST": "Fashion-MNIST", "FMNIST": "Fashion-MNIST"}.get(str(dataset), str(dataset))
    bundle = data_core.load_vision_bundle(
        dataset_for_loader,
        data_root=ROOT / "data",
        train_size=int(train_size),
        val_size=int(held_size),
        test_size=int(test_size),
        seed=int(seed),
        download=False,
        allow_fake_data=False,
    )
    return {
        "input_dim": int(bundle.input_dim),
        "num_classes": int(bundle.num_classes),
        "x_train": bundle.x_train.float(),
        "y_train": bundle.y_train.long(),
        "x_held": bundle.x_val.float(),
        "y_held": bundle.y_val.long(),
        "x_test": bundle.x_test.float(),
        "y_test": bundle.y_test.long(),
        "source_kind": "dgkan_core.load_vision_bundle_local_cache",
        "dataset_loader_name": dataset_for_loader,
        "used_fake_data": int(bool(bundle.used_fake_data)),
    }


def make_model(input_dim: int, num_classes: int, hidden: int, seed: int, device: Any) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SmallMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, num_classes)

        def features(self, x: Any) -> tuple[Any, Any, Any]:
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
            logits = self.fc3(h2)
            return h1, h2, logits

        def forward(self, x: Any) -> Any:
            return self.features(x)[2]

    torch.manual_seed(int(seed))
    model = SmallMLP().to(device)
    return model


def batch_indices(n: int, batch_size: int, generator: Any, device: Any) -> Any:
    import torch

    return torch.randint(0, int(n), (int(batch_size),), generator=generator, device=device)


def batch_indices_from_pool(pool: Any, batch_size: int, generator: Any) -> Any:
    import torch

    if int(pool.numel()) == 0:
        raise ValueError("empty index pool")
    pick = torch.randint(0, int(pool.numel()), (int(batch_size),), generator=generator, device=pool.device)
    return pool[pick]


def evaluate_tensors(model: Any, x: Any, y: Any, device: Any, num_classes: int, batch_size: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses = []
    logits_all = []
    labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb, reduction="none")
            losses.append(loss.detach().cpu())
            logits_all.append(logits.detach().cpu())
            labels.append(yb.detach().cpu())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            total += int(yb.numel())
    if total == 0:
        return {k: math.nan for k in ["NLL", "accuracy", "ECE", "Brier", "tail_q95", "tail_q99", "margin_mean", "margin_q10", "margin_q01"]}
    losses_cat = torch.cat(losses)
    logits_cat = torch.cat(logits_all).float()
    labels_cat = torch.cat(labels).long()
    probs = logits_cat.softmax(dim=-1)
    target = F.one_hot(labels_cat, num_classes=int(num_classes)).float()
    conf, pred = probs.max(dim=-1)
    ok = (pred == labels_cat).float()
    ece = torch.tensor(0.0)
    for i in range(10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if i < 9 else (conf <= hi))
        if mask.any():
            ece = ece + mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs()
    true_logits = logits_cat.gather(1, labels_cat.view(-1, 1)).squeeze(1)
    masked = logits_cat.clone()
    masked[torch.arange(labels_cat.numel()), labels_cat] = -float("inf")
    margins = true_logits - masked.max(dim=-1).values
    return {
        "NLL": float(losses_cat.mean().item()),
        "accuracy": float(correct / max(1, total)),
        "ECE": float(ece.item()),
        "Brier": float(((probs - target) ** 2).sum(dim=-1).mean().item()),
        "tail_q95": float(torch.quantile(losses_cat.float(), 0.95).item()),
        "tail_q99": float(torch.quantile(losses_cat.float(), 0.99).item()),
        "margin_mean": float(margins.mean().item()),
        "margin_q10": float(torch.quantile(margins.float(), 0.10).item()),
        "margin_q01": float(torch.quantile(margins.float(), 0.01).item()),
    }


def cohort_loss_values(model: Any, x: Any, y: Any, device: Any, cohorts: int, batch_size: int) -> list[float]:
    n = int(x.shape[0])
    out: list[float] = []
    for c in range(int(cohorts)):
        start = int(round(c * n / max(1, cohorts)))
        end = int(round((c + 1) * n / max(1, cohorts)))
        if end <= start:
            continue
        out.append(evaluate_tensors(model, x[start:end], y[start:end], device, 10, batch_size)["NLL"])
    return out


def named_params(model: Any) -> list[tuple[str, Any]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def flatten_grads(grads: Iterable[Any | None], params: list[Any]) -> Any:
    import torch

    parts = []
    for grad, param in zip(grads, params):
        if grad is None:
            parts.append(torch.zeros_like(param).reshape(-1))
        else:
            parts.append(grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def apply_flat_direction(named: list[tuple[str, Any]], direction: Any, lr: float, weight_decay: float) -> float:
    import torch

    offset = 0
    norm2 = 0.0
    with torch.no_grad():
        for _name, p in named:
            n = int(p.numel())
            chunk = direction[offset : offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
            if float(weight_decay) != 0.0:
                p.mul_(1.0 - float(lr) * float(weight_decay))
            delta = -float(lr) * chunk
            p.add_(delta)
            norm2 += float(delta.norm().item()) ** 2
            offset += n
    return math.sqrt(norm2)


def vector_norm(vec: Any) -> float:
    if vec is None or int(vec.numel()) == 0:
        return 0.0
    val = float(vec.norm().detach().item())
    return val if math.isfinite(val) else 0.0


def clipped(vec: Any, max_norm: float) -> tuple[Any, float]:
    norm = vector_norm(vec)
    if norm > float(max_norm) > 0.0:
        return vec * (float(max_norm) / max(norm, 1.0e-12)), norm
    return vec, norm


def cohort_gradients(model: Any, x: Any, y: Any, indices: list[Any], num_classes: int) -> tuple[list[Any], list[float]]:
    import torch.nn.functional as F

    params_named = named_params(model)
    params = [p for _n, p in params_named]
    grads = []
    losses = []
    model.train()
    for idx in indices:
        logits = model(x[idx]).float()
        loss = F.cross_entropy(logits, y[idx].long())
        grad_tensors = __import__("torch").autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        grads.append(flatten_grads(grad_tensors, params))
        losses.append(float(loss.detach().item()))
    return grads, losses


def pairwise_conflict_stats(grads: list[Any]) -> dict[str, float]:
    import torch

    neg = []
    total = 0
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            ni = grads[i].norm().clamp_min(1.0e-12)
            nj = grads[j].norm().clamp_min(1.0e-12)
            cos = float(torch.dot(grads[i], grads[j]).div(ni * nj).detach().item())
            total += 1
            if cos < 0.0:
                neg.append(cos)
    return {
        "cohort_gradient_pairs": float(total),
        "cohort_gradient_conflict_rate": float(len(neg) / max(1, total)),
        "mean_negative_pairwise_cosine": float(statistics.fmean(neg)) if neg else 0.0,
    }


def pcgrad_direction(grads: list[Any]) -> Any:
    import torch

    adjusted = [g.clone() for g in grads]
    for i in range(len(adjusted)):
        for j in range(len(grads)):
            if i == j:
                continue
            dot = torch.dot(adjusted[i], grads[j])
            if float(dot.detach().item()) < 0.0:
                adjusted[i] = adjusted[i] - dot / grads[j].dot(grads[j]).clamp_min(1.0e-12) * grads[j]
    return torch.stack(adjusted, dim=0).mean(dim=0)


def random_same_span_direction(grads: list[Any], target_norm: float, generator: Any) -> Any:
    import torch

    stacked = torch.stack(grads, dim=0)
    coeff = torch.randn((len(grads),), generator=generator, device=stacked.device)
    direction = coeff @ stacked
    norm = direction.norm().clamp_min(1.0e-12)
    return direction * (float(target_norm) / norm)


def debt_gradient(model: Any, x: Any, y: Any, indices: list[Any], params: list[Any], num_classes: int) -> Any:
    import torch
    import torch.nn.functional as F

    debt_terms = []
    model.train()
    for idx in indices:
        logits = model(x[idx]).float()
        losses = F.cross_entropy(logits, y[idx].long(), reduction="none")
        probs = logits.softmax(dim=-1)
        target = F.one_hot(y[idx].long(), num_classes=int(num_classes)).float()
        brier = ((probs - target) ** 2).sum(dim=-1).mean()
        k = max(1, int(math.ceil(0.25 * int(losses.numel()))))
        tail = torch.topk(losses.float(), k=k, largest=True).values.mean()
        debt_terms.append(brier + 0.05 * tail)
    debt_loss = torch.stack(debt_terms).mean() if debt_terms else torch.tensor(0.0, device=x.device)
    grads = torch.autograd.grad(debt_loss, params, retain_graph=False, allow_unused=True)
    return flatten_grads(grads, params)


def debt_score(model: Any, x: Any, y: Any, indices: list[Any], num_classes: int) -> float:
    import torch
    import torch.nn.functional as F

    scores = []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            logits = model(x[idx]).float()
            losses = F.cross_entropy(logits, y[idx].long(), reduction="none")
            probs = logits.softmax(dim=-1)
            target = F.one_hot(y[idx].long(), num_classes=int(num_classes)).float()
            brier = ((probs - target) ** 2).sum(dim=-1).mean()
            k = max(1, int(math.ceil(0.25 * int(losses.numel()))))
            tail = torch.topk(losses.float(), k=k, largest=True).values.mean()
            scores.append(brier + 0.05 * tail)
    return float(torch.stack(scores).mean().item()) if scores else 0.0


def soft_ece_proxy(logits: Any, labels: Any, num_classes: int, bins: int = 10, sigma: float = 0.08) -> Any:
    import torch

    if int(logits.shape[0]) == 0:
        return torch.tensor(0.0, device=logits.device)
    probs = logits.softmax(dim=-1)
    conf = probs.max(dim=-1).values
    p_true = probs.gather(1, labels.long().view(-1, 1)).squeeze(1)
    centers = torch.linspace(
        0.5 / float(bins),
        1.0 - 0.5 / float(bins),
        int(bins),
        device=logits.device,
    )
    distances = (conf[:, None] - centers[None, :]).pow(2)
    weights = torch.softmax(-distances / max(1.0e-6, 2.0 * float(sigma) * float(sigma)), dim=1)
    mass = weights.sum(dim=0).clamp_min(1.0e-8)
    bin_conf = (weights * conf[:, None]).sum(dim=0) / mass
    bin_true = (weights * p_true[:, None]).sum(dim=0) / mass
    bin_weight = mass / float(conf.numel())
    smooth_abs_gap = ((bin_conf - bin_true).pow(2) + 1.0e-8).sqrt()
    return (bin_weight * smooth_abs_gap).sum()


def soft_accuracy_ece_proxy(logits: Any, labels: Any, num_classes: int, bins: int = 10, sigma: float = 0.08) -> Any:
    import torch

    if int(logits.shape[0]) == 0:
        return torch.tensor(0.0, device=logits.device)
    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    ok = (pred.detach() == labels.long()).float()
    centers = torch.linspace(
        0.5 / float(bins),
        1.0 - 0.5 / float(bins),
        int(bins),
        device=logits.device,
    )
    distances = (conf[:, None] - centers[None, :]).pow(2)
    weights = torch.softmax(-distances / max(1.0e-6, 2.0 * float(sigma) * float(sigma)), dim=1)
    mass = weights.sum(dim=0).clamp_min(1.0e-8)
    bin_conf = (weights * conf[:, None]).sum(dim=0) / mass
    bin_acc = (weights * ok[:, None]).sum(dim=0) / mass
    bin_weight = mass / float(conf.numel())
    smooth_abs_gap = ((bin_conf - bin_acc.detach()).pow(2) + 1.0e-8).sqrt()
    return (bin_weight * smooth_abs_gap).sum()


def debt_component_values(
    model: Any,
    x: Any,
    y: Any,
    indices: list[Any],
    num_classes: int,
    calib_mode: str = "gap_sq",
    tail_fraction: float = 0.25,
    agg_mode: str = "mean",
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    values: dict[str, list[Any]] = {"calib": [], "brier": [], "tail": []}
    model.eval()
    with torch.no_grad():
        for idx in indices:
            logits = model(x[idx]).float()
            losses = F.cross_entropy(logits, y[idx].long(), reduction="none")
            probs = logits.softmax(dim=-1)
            target = F.one_hot(y[idx].long(), num_classes=int(num_classes)).float()
            conf = probs.max(dim=-1).values
            p_true = probs.gather(1, y[idx].long().view(-1, 1)).squeeze(1)
            if calib_mode == "soft_acc_ece":
                calib = soft_accuracy_ece_proxy(logits, y[idx].long(), int(num_classes))
            elif calib_mode == "soft_ece":
                calib = soft_ece_proxy(logits, y[idx].long(), int(num_classes))
            else:
                calib = ((conf - p_true) ** 2).mean()
            brier = ((probs - target) ** 2).sum(dim=-1).mean()
            k = max(1, int(math.ceil(float(tail_fraction) * int(losses.numel()))))
            tail = torch.topk(losses.float(), k=k, largest=True).values.mean()
            values["calib"].append(calib)
            values["brier"].append(brier)
            values["tail"].append(tail)
    out: dict[str, float] = {}
    for key, items in values.items():
        if not items:
            out[key] = 0.0
            continue
        stacked = torch.stack(items)
        value = stacked.max() if agg_mode == "max" else stacked.mean()
        out[key] = float(value.item())
    return out


def debt_component_gradients(
    model: Any,
    x: Any,
    y: Any,
    indices: list[Any],
    params: list[Any],
    num_classes: int,
    calib_mode: str = "gap_sq",
    tail_fraction: float = 0.25,
    agg_mode: str = "mean",
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch
    import torch.nn.functional as F

    calib_terms = []
    brier_terms = []
    tail_terms = []
    model.train()
    for idx in indices:
        logits = model(x[idx]).float()
        losses = F.cross_entropy(logits, y[idx].long(), reduction="none")
        probs = logits.softmax(dim=-1)
        target = F.one_hot(y[idx].long(), num_classes=int(num_classes)).float()
        conf = probs.max(dim=-1).values
        p_true = probs.gather(1, y[idx].long().view(-1, 1)).squeeze(1)
        if calib_mode == "soft_acc_ece":
            calib_terms.append(soft_accuracy_ece_proxy(logits, y[idx].long(), int(num_classes)))
        elif calib_mode == "soft_ece":
            calib_terms.append(soft_ece_proxy(logits, y[idx].long(), int(num_classes)))
        else:
            calib_terms.append(((conf - p_true) ** 2).mean())
        brier_terms.append(((probs - target) ** 2).sum(dim=-1).mean())
        k = max(1, int(math.ceil(float(tail_fraction) * int(losses.numel()))))
        tail_terms.append(torch.topk(losses.float(), k=k, largest=True).values.mean())
    zero = torch.tensor(0.0, device=x.device)

    def aggregate_terms(terms: list[Any]) -> Any:
        if not terms:
            return zero
        stacked = torch.stack(terms)
        if agg_mode == "max":
            return stacked[torch.argmax(stacked.detach())]
        return stacked.mean()

    calib_loss = aggregate_terms(calib_terms)
    brier_loss = aggregate_terms(brier_terms)
    tail_loss = aggregate_terms(tail_terms)
    calib_grad = flatten_grads(torch.autograd.grad(calib_loss, params, retain_graph=True, allow_unused=True), params)
    brier_grad = flatten_grads(torch.autograd.grad(brier_loss, params, retain_graph=True, allow_unused=True), params)
    tail_grad = flatten_grads(torch.autograd.grad(tail_loss, params, retain_graph=False, allow_unused=True), params)
    return (
        {"calib": calib_grad, "brier": brier_grad, "tail": tail_grad},
        {
            "calib": float(calib_loss.detach().item()),
            "brier": float(brier_loss.detach().item()),
            "tail": float(tail_loss.detach().item()),
        },
    )


def sign_consistency(predicted: float, actual: float, tol: float = 1.0e-10) -> int:
    if abs(float(predicted)) <= tol and abs(float(actual)) <= tol:
        return 1
    if abs(float(predicted)) <= tol or abs(float(actual)) <= tol:
        return 0
    return int((float(predicted) > 0.0) == (float(actual) > 0.0))


def debt_components(model: Any, x: Any, y: Any, indices: list[Any], num_classes: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    usable = [idx.reshape(-1) for idx in indices if int(idx.numel()) > 0]
    if not usable:
        return {"ECE": 0.0, "Brier": 0.0, "tail_q95": 0.0}
    idx_all = torch.cat(usable, dim=0)
    model.eval()
    with torch.no_grad():
        logits = model(x[idx_all]).float()
        labels = y[idx_all].long()
        losses = F.cross_entropy(logits, labels, reduction="none")
        probs = logits.softmax(dim=-1)
        target = F.one_hot(labels, num_classes=int(num_classes)).float()
        conf, pred = probs.max(dim=-1)
        ok = (pred == labels).float()
        ece = torch.tensor(0.0, device=logits.device)
        for i in range(10):
            lo = i / 10.0
            hi = (i + 1) / 10.0
            mask = (conf >= lo) & ((conf < hi) if i < 9 else (conf <= hi))
            if mask.any():
                ece = ece + mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs()
        return {
            "ECE": float(ece.item()),
            "Brier": float(((probs - target) ** 2).sum(dim=-1).mean().item()),
            "tail_q95": float(torch.quantile(losses.float(), 0.95).item()),
        }


def temperature_scaled_model(base_model: Any, temperature: float) -> Any:
    import torch.nn as nn

    class TemperatureScaledModel(nn.Module):
        def __init__(self, base: Any, temp: float) -> None:
            super().__init__()
            self.base = base
            self.temperature = float(temp)

        def forward(self, x: Any) -> Any:
            return self.base(x).float() / max(self.temperature, 1.0e-6)

    wrapper = TemperatureScaledModel(base_model, float(temperature))
    wrapper.eval()
    return wrapper


def held_temperature_probe(
    model: Any,
    x_held: Any,
    y_held: Any,
    device: Any,
    num_classes: int,
    batch_size: int,
    pre_held: dict[str, float],
) -> tuple[Any, dict[str, Any]]:
    candidates = [0.50, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    best_feasible: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    for temperature in candidates:
        candidate_model = temperature_scaled_model(model, temperature)
        metrics = evaluate_tensors(candidate_model, x_held, y_held, device, int(num_classes), int(batch_size))
        violations = {
            key: max(0.0, float(metrics[key]) - float(pre_held[key]))
            for key in ("ECE", "Brier", "tail_q95")
        }
        violation_sum = sum(violations.values())
        nll_gain = float(pre_held["NLL"]) - float(metrics["NLL"])
        item = {
            "temperature": float(temperature),
            "metrics": metrics,
            "violations": violations,
            "violation_sum": float(violation_sum),
            "nll_gain": float(nll_gain),
            "feasible": int(violation_sum <= 1.0e-8),
        }
        if item["feasible"] and (
            best_feasible is None
            or item["nll_gain"] > best_feasible["nll_gain"]
            or (item["nll_gain"] == best_feasible["nll_gain"] and item["temperature"] < best_feasible["temperature"])
        ):
            best_feasible = item
        if best_any is None or (
            item["violation_sum"],
            -item["nll_gain"],
            item["temperature"],
        ) < (
            best_any["violation_sum"],
            -best_any["nll_gain"],
            best_any["temperature"],
        ):
            best_any = item

    chosen = best_feasible or best_any or {
        "temperature": 1.0,
        "metrics": pre_held,
        "violations": {"ECE": 0.0, "Brier": 0.0, "tail_q95": 0.0},
        "violation_sum": 0.0,
        "nll_gain": 0.0,
        "feasible": 0,
    }
    diag = {
        "held_temp_probe_temperature": chosen["temperature"],
        "held_temp_probe_feasible": chosen["feasible"],
        "held_temp_probe_violation_sum": chosen["violation_sum"],
        "held_temp_probe_violation_ECE": chosen["violations"]["ECE"],
        "held_temp_probe_violation_Brier": chosen["violations"]["Brier"],
        "held_temp_probe_violation_tail_q95": chosen["violations"]["tail_q95"],
        "held_temp_probe_selected_NLL_gain": chosen["nll_gain"],
        "held_temp_probe_candidate_count": len(candidates),
        "held_temp_probe_claim_limit": "diagnostic upper-bound: temperature selected on held split; not eligible for official route promotion",
    }
    return temperature_scaled_model(model, float(chosen["temperature"])), diag


def train_temperature_calibration(
    model: Any,
    x_calib: Any,
    y_calib: Any,
    device: Any,
    num_classes: int,
    batch_size: int,
    pre_train: dict[str, float],
    soften_only: bool = False,
) -> tuple[Any, dict[str, Any]]:
    candidates = [0.50, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    if soften_only:
        candidates = [t for t in candidates if t >= 1.0]
    best_feasible: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    for temperature in candidates:
        candidate_model = temperature_scaled_model(model, temperature)
        metrics = evaluate_tensors(candidate_model, x_calib, y_calib, device, int(num_classes), int(batch_size))
        violations = {
            key: max(0.0, float(metrics[key]) - float(pre_train[key]))
            for key in ("ECE", "Brier", "tail_q95")
        }
        violation_sum = sum(violations.values())
        nll_gain = float(pre_train["NLL"]) - float(metrics["NLL"])
        item = {
            "temperature": float(temperature),
            "metrics": metrics,
            "violations": violations,
            "violation_sum": float(violation_sum),
            "nll_gain": float(nll_gain),
            "feasible": int(violation_sum <= 1.0e-8),
        }
        if item["feasible"] and (
            best_feasible is None
            or item["nll_gain"] > best_feasible["nll_gain"]
            or (item["nll_gain"] == best_feasible["nll_gain"] and item["temperature"] < best_feasible["temperature"])
        ):
            best_feasible = item
        if best_any is None or (
            item["violation_sum"],
            -item["nll_gain"],
            item["temperature"],
        ) < (
            best_any["violation_sum"],
            -best_any["nll_gain"],
            best_any["temperature"],
        ):
            best_any = item

    chosen = best_feasible or best_any or {
        "temperature": 1.0,
        "metrics": pre_train,
        "violations": {"ECE": 0.0, "Brier": 0.0, "tail_q95": 0.0},
        "violation_sum": 0.0,
        "nll_gain": 0.0,
        "feasible": 0,
    }
    diag = {
        "train_temp_temperature": chosen["temperature"],
        "train_temp_feasible": chosen["feasible"],
        "train_temp_violation_sum": chosen["violation_sum"],
        "train_temp_violation_ECE": chosen["violations"]["ECE"],
        "train_temp_violation_Brier": chosen["violations"]["Brier"],
        "train_temp_violation_tail_q95": chosen["violations"]["tail_q95"],
        "train_temp_selected_NLL_gain": chosen["nll_gain"],
        "train_temp_candidate_count": len(candidates),
        "train_temp_claim_limit": (
            "temperature selected on train calibration subset only; no held/test direction used"
            + ("; soften_only=true" if soften_only else "")
        ),
    }
    return temperature_scaled_model(model, float(chosen["temperature"])), diag


def snapshot_params(named: list[tuple[str, Any]]) -> list[Any]:
    return [p.detach().clone() for _name, p in named]


def restore_params(named: list[tuple[str, Any]], snapshot: list[Any]) -> None:
    import torch

    with torch.no_grad():
        for (_name, p), value in zip(named, snapshot):
            p.copy_(value)


def debt_orthogonalize(direction: Any, debt_grad: Any) -> tuple[Any, float, float]:
    dot = float(direction.dot(debt_grad).detach().item()) if int(direction.numel()) else 0.0
    denom = debt_grad.dot(debt_grad).clamp_min(1.0e-12)
    if dot < 0.0:
        direction = direction - (direction.dot(debt_grad) / denom) * debt_grad
    dot_after = float(direction.dot(debt_grad).detach().item()) if int(direction.numel()) else 0.0
    return direction, dot, dot_after


def run_pretrain(model: Any, tensors: dict[str, Any], device: Any, args: argparse.Namespace, seed: int) -> None:
    import torch
    import torch.nn.functional as F

    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 49001)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.pretrain_lr), weight_decay=float(args.weight_decay))
    model.train()
    for _step in range(int(args.pretrain_steps)):
        idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_train[idx]).float(), y_train[idx].long())
        loss.backward()
        opt.step()


def branch_step_h6(
    model: Any,
    xb: Any,
    yb: Any,
    method: str,
    lr: float,
    target_eta: float,
    hard_fraction: float,
    grad_clip: float = 5.0,
    apply_update: bool = True,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    params = [p for _n, p in named_params(model)]
    model.train()
    h1, h2, logits = model.features(xb)
    losses = F.cross_entropy(logits.float(), yb.long(), reduction="none")
    cvar_methods = {
        "H6_layerwise_cvar_target",
        "H6_shuffled_cvar_target_control",
        "H6_layerwise_cvar_target_train_exact_accept",
        "H6_shuffled_cvar_target_train_exact_accept_control",
    }
    shuffled_methods = {
        "H6_shuffled_layer_target_control",
        "H6_shuffled_cvar_target_control",
        "H6_shuffled_cvar_target_train_exact_accept_control",
    }
    if method in cvar_methods:
        k = max(1, min(int(losses.numel()), int(math.ceil(float(hard_fraction) * int(losses.numel())))))
        hard_idx = torch.topk(losses.detach(), k=k, largest=True).indices
        ce = losses[hard_idx].mean()
    else:
        hard_idx = torch.arange(int(losses.numel()), device=losses.device)
        ce = losses.mean()
    grad_h1, grad_h2 = torch.autograd.grad(ce, [h1, h2], retain_graph=False)
    target_h1 = h1.detach() - float(target_eta) * grad_h1.detach()
    target_h2 = h2.detach() - float(target_eta) * grad_h2.detach()
    if method in shuffled_methods:
        perm = torch.randperm(int(xb.shape[0]), device=xb.device)
        target_h1 = target_h1[perm]
        target_h2 = target_h2[perm]
    h1_new, h2_new, _logits_new = model.features(xb)
    local_h2 = F.relu(model.fc2(h1.detach()))
    local_logits = model.fc3(h2.detach())
    loss_h1 = F.mse_loss(h1_new, target_h1)
    loss_h2 = F.mse_loss(local_h2, target_h2)
    loss_out = F.cross_entropy(local_logits.float(), yb.long())
    local_loss = loss_h1 + loss_h2 + loss_out
    grads = torch.autograd.grad(local_loss, params, allow_unused=True)
    flat = flatten_grads(grads, params)
    clipped_flat, raw_norm = clipped(flat, float(grad_clip))
    named = named_params(model)
    update_norm = apply_flat_direction(named, clipped_flat, lr, 0.0) if apply_update else 0.0
    return {
        "layerwise_target_loss": float(local_loss.detach().item()),
        "layerwise_target_h1_mse": float(loss_h1.detach().item()),
        "layerwise_target_h2_mse": float(loss_h2.detach().item()),
        "layerwise_target_norm": float((target_h1.norm() + target_h2.norm()).detach().item()),
        "layerwise_hard_fraction": float(len(hard_idx) / max(1, int(losses.numel()))),
        "raw_direction_norm_mean": raw_norm,
        "update_norm_mean": update_norm,
        "flat_direction": clipped_flat.detach(),
    }


def run_branch(method: str, model: Any, tensors: dict[str, Any], device: Any, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    import torch

    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16))
    support_ema = None
    update_norms: list[float] = []
    raw_norms: list[float] = []
    losses: list[float] = []
    conflict_rates: list[float] = []
    neg_cosines: list[float] = []
    support_energy_fracs: list[float] = []
    layer_h1_mse: list[float] = []
    layer_h2_mse: list[float] = []
    layer_target_losses: list[float] = []
    layer_hard_fractions: list[float] = []
    debt_overlap_before: list[float] = []
    debt_overlap_after: list[float] = []
    debt_grad_norms: list[float] = []
    train_debt_accepts: list[float] = []
    train_debt_accept_scales: list[float] = []
    held_debt_accepts: list[float] = []
    held_debt_accept_scales: list[float] = []
    primal_dual_lambda_calib_values: list[float] = []
    primal_dual_lambda_brier_values: list[float] = []
    primal_dual_lambda_tail_values: list[float] = []
    primal_dual_pred_calib_deltas: list[float] = []
    primal_dual_actual_calib_deltas: list[float] = []
    primal_dual_pred_brier_deltas: list[float] = []
    primal_dual_actual_brier_deltas: list[float] = []
    primal_dual_pred_tail_deltas: list[float] = []
    primal_dual_actual_tail_deltas: list[float] = []
    primal_dual_calib_sign_matches: list[float] = []
    primal_dual_brier_sign_matches: list[float] = []
    primal_dual_tail_sign_matches: list[float] = []
    primal_dual_guard_debt_before: list[float] = []
    primal_dual_guard_debt_after: list[float] = []
    dual_calib = 0.0
    dual_brier = 0.0
    dual_tail = 0.0
    params_named = named_params(model)
    params = [p for _n, p in params_named]
    persistent_guard_methods = {
        "H7_cohort_surgery_persistent_guard_accept",
        "H7_same_span_random_persistent_guard_accept_control",
    }
    persistent_guard_indices = None
    persistent_task_pool = None
    if method in persistent_guard_methods:
        n_train = int(x_train.shape[0])
        perm = torch.randperm(n_train, generator=generator, device=device)
        guard_n = max(
            int(args.cohort_size),
            min(n_train - int(args.cohort_size), int(round(float(args.persistent_guard_fraction) * n_train))),
        )
        persistent_guard_indices = perm[:guard_n]
        persistent_task_pool = perm[guard_n:]
        if int(persistent_task_pool.numel()) < int(args.cohort_size):
            persistent_task_pool = perm

    for _step in range(int(args.branch_steps)):
        if method in {
            "H6_layerwise_target",
            "H6_shuffled_layer_target_control",
            "H6_layerwise_cvar_target",
            "H6_shuffled_cvar_target_control",
            "H6_layerwise_cvar_target_train_exact_accept",
            "H6_shuffled_cvar_target_train_exact_accept_control",
            "H0_adamw_optimizer",
        }:
            idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
            if method == "H0_adamw_optimizer":
                import torch.nn.functional as F

                for p in params:
                    p.grad = None
                logits = model(x_train[idx]).float()
                loss = F.cross_entropy(logits, y_train[idx].long())
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                flat = flatten_grads(grads, params)
                flat, raw_norm = clipped(flat, 5.0)
                upd = apply_flat_direction(params_named, flat, float(args.branch_lr), float(args.weight_decay))
                losses.append(float(loss.detach().item()))
                raw_norms.append(raw_norm)
                update_norms.append(upd)
            else:
                diag = branch_step_h6(
                    model,
                    x_train[idx],
                    y_train[idx],
                    method,
                    float(args.layerwise_lr),
                    float(args.layerwise_target_eta),
                    float(args.hard_fraction),
                    float(args.grad_clip),
                    method
                    not in {
                        "H6_layerwise_cvar_target_train_exact_accept",
                        "H6_shuffled_cvar_target_train_exact_accept_control",
                    },
                )
                upd = float(diag["update_norm_mean"])
                if method in {
                    "H6_layerwise_cvar_target_train_exact_accept",
                    "H6_shuffled_cvar_target_train_exact_accept_control",
                }:
                    direction = diag["flat_direction"]
                    accept_indices = [
                        batch_indices(int(x_train.shape[0]), int(args.cohort_size), generator, device)
                        for _c in range(int(args.cohorts))
                    ]
                    before_components = debt_components(
                        model,
                        x_train,
                        y_train,
                        accept_indices,
                        int(tensors["num_classes"]),
                    )
                    accepted = 0.0
                    used_scale = 0.0
                    upd = 0.0
                    for trial in range(int(args.accept_trials)):
                        scale = float(args.layerwise_lr) * (0.5 ** trial)
                        snap = snapshot_params(params_named)
                        upd = apply_flat_direction(params_named, direction, scale, 0.0)
                        after_components = debt_components(
                            model,
                            x_train,
                            y_train,
                            accept_indices,
                            int(tensors["num_classes"]),
                        )
                        passes_gate = all(
                            after_components[key] <= before_components[key] + float(args.accept_tolerance)
                            for key in ("ECE", "Brier", "tail_q95")
                        )
                        if passes_gate:
                            accepted = 1.0
                            used_scale = scale
                            break
                        restore_params(params_named, snap)
                        upd = 0.0
                    train_debt_accepts.append(accepted)
                    train_debt_accept_scales.append(used_scale)
                layer_target_losses.append(diag["layerwise_target_loss"])
                layer_h1_mse.append(diag["layerwise_target_h1_mse"])
                layer_h2_mse.append(diag["layerwise_target_h2_mse"])
                layer_hard_fractions.append(diag["layerwise_hard_fraction"])
                raw_norms.append(diag["raw_direction_norm_mean"])
                update_norms.append(upd)
            continue

        if method in persistent_guard_methods and persistent_task_pool is not None:
            cohort_indices = [
                batch_indices_from_pool(persistent_task_pool, int(args.cohort_size), generator)
                for _c in range(int(args.cohorts))
            ]
        else:
            cohort_indices = [
                batch_indices(int(x_train.shape[0]), int(args.cohort_size), generator, device)
                for _c in range(int(args.cohorts))
            ]
        grads, cohort_losses = cohort_gradients(model, x_train, y_train, cohort_indices, int(tensors["num_classes"]))
        stats = pairwise_conflict_stats(grads)
        conflict_rates.append(stats["cohort_gradient_conflict_rate"])
        neg_cosines.append(stats["mean_negative_pairwise_cosine"])
        losses.append(float(statistics.fmean(cohort_losses)) if cohort_losses else math.nan)
        stacked = torch.stack(grads, dim=0)
        gbar = stacked.mean(dim=0)

        if method == "H1_population_mean_gradient":
            direction = gbar
            support_energy_fracs.append(1.0)
        elif method in {"H2_support_native_ema", "H2_random_support_control"}:
            if support_ema is None:
                support_ema = stacked.detach().clone()
            else:
                support_ema = float(args.support_ema_beta) * support_ema + (1.0 - float(args.support_ema_beta)) * stacked.detach()
            rank = min(int(args.support_rank), int(support_ema.shape[0]), int(support_ema.shape[1]))
            if method == "H2_support_native_ema":
                _u, _s, vh = torch.linalg.svd(support_ema, full_matrices=False)
                basis = vh[:rank].T.contiguous()
            else:
                rand = torch.randn((int(gbar.numel()), rank), generator=generator, device=device)
                basis, _r = torch.linalg.qr(rand, mode="reduced")
            direction = basis @ (basis.T @ gbar)
            denom = gbar.dot(gbar).clamp_min(1.0e-12)
            support_energy_fracs.append(float(direction.dot(direction).div(denom).detach().item()))
        elif method in {
            "H6_layerwise_cvar_population_blend_exact_accept",
            "H6_shuffled_cvar_population_blend_exact_accept_control",
            "H6_layerwise_cvar_population_blend_primal_dual",
            "H6_shuffled_cvar_population_blend_primal_dual_control",
        }:
            h6_idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
            h6_source_method = (
                "H6_shuffled_cvar_target_train_exact_accept_control"
                if "shuffled" in method
                else "H6_layerwise_cvar_target_train_exact_accept"
            )
            h6_diag = branch_step_h6(
                model,
                x_train[h6_idx],
                y_train[h6_idx],
                h6_source_method,
                float(args.layerwise_lr),
                float(args.layerwise_target_eta),
                float(args.hard_fraction),
                float(args.grad_clip),
                False,
            )
            h6_direction = h6_diag["flat_direction"]
            target_norm = gbar.norm().detach().clamp_min(1.0e-12)
            h6_direction = h6_direction * (target_norm / h6_direction.norm().detach().clamp_min(1.0e-12))
            alpha = float(args.surgery_blend_alpha)
            direction = (1.0 - alpha) * gbar + alpha * h6_direction
            layer_target_losses.append(h6_diag["layerwise_target_loss"])
            layer_h1_mse.append(h6_diag["layerwise_target_h1_mse"])
            layer_h2_mse.append(h6_diag["layerwise_target_h2_mse"])
            layer_hard_fractions.append(h6_diag["layerwise_hard_fraction"])
        elif method == "H7_cohort_surgery_blend_debt_orthogonal":
            real = pcgrad_direction(grads)
            alpha = float(args.surgery_blend_alpha)
            direction = (1.0 - alpha) * gbar + alpha * real
        elif method == "H7_same_span_random_blend_debt_orthogonal_control":
            real = pcgrad_direction(grads)
            rand = random_same_span_direction(grads, vector_norm(real), generator)
            alpha = float(args.surgery_blend_alpha)
            direction = (1.0 - alpha) * gbar + alpha * rand
        elif method in {
            "H7_cohort_surgery_primal_dual_safety",
            "H7_cohort_surgery_primal_dual_train_temp",
            "H7_cohort_surgery_primal_dual_train_soften_temp",
            "H7_cohort_surgery_primal_dual_soft_ece",
            "H7_cohort_surgery_primal_dual_soft_acc_ece",
            "H7_cohort_surgery_primal_dual_worst_guard",
        }:
            direction = pcgrad_direction(grads)
        elif method in {
            "H7_same_span_random_primal_dual_safety_control",
            "H7_same_span_random_primal_dual_train_temp_control",
            "H7_same_span_random_primal_dual_train_soften_temp_control",
            "H7_same_span_random_primal_dual_soft_ece_control",
            "H7_same_span_random_primal_dual_soft_acc_ece_control",
            "H7_same_span_random_primal_dual_worst_guard_control",
        }:
            real = pcgrad_direction(grads)
            direction = random_same_span_direction(grads, vector_norm(real), generator)
        elif method in {
            "H7_cohort_surgery",
            "H7_cohort_surgery_debt_orthogonal",
            "H7_cohort_surgery_debt_barrier",
            "H7_cohort_surgery_train_debt_accept",
            "H7_cohort_surgery_train_exact_debt_accept",
            "H7_cohort_surgery_persistent_guard_accept",
            "H7_cohort_surgery_held_debt_accept",
            "H7_cohort_surgery_debt_orthogonal_held_temp_probe",
        }:
            direction = pcgrad_direction(grads)
        elif method in {
            "H7_same_span_random_control",
            "H7_same_span_random_debt_orthogonal_control",
            "H7_same_span_random_debt_barrier_control",
            "H7_same_span_random_train_debt_accept_control",
            "H7_same_span_random_train_exact_debt_accept_control",
            "H7_same_span_random_persistent_guard_accept_control",
            "H7_same_span_random_held_debt_accept_control",
            "H7_same_span_random_debt_orthogonal_held_temp_probe_control",
            "H7_same_span_random_primal_dual_safety_control",
            "H7_same_span_random_primal_dual_train_temp_control",
            "H7_same_span_random_primal_dual_train_soften_temp_control",
            "H7_same_span_random_primal_dual_soft_ece_control",
            "H7_same_span_random_primal_dual_soft_acc_ece_control",
            "H7_same_span_random_primal_dual_worst_guard_control",
        }:
            real = pcgrad_direction(grads)
            direction = random_same_span_direction(grads, vector_norm(real), generator)
        else:
            raise ValueError(f"unknown method {method!r}")

        if method in {
            "H7_cohort_surgery_debt_orthogonal",
            "H7_same_span_random_debt_orthogonal_control",
            "H7_cohort_surgery_debt_orthogonal_held_temp_probe",
            "H7_same_span_random_debt_orthogonal_held_temp_probe_control",
            "H7_cohort_surgery_blend_debt_orthogonal",
            "H7_same_span_random_blend_debt_orthogonal_control",
        }:
            dgrad = debt_gradient(model, x_train, y_train, cohort_indices, params, int(tensors["num_classes"]))
            direction, before, after = debt_orthogonalize(direction, dgrad)
            debt_overlap_before.append(before)
            debt_overlap_after.append(after)
            debt_grad_norms.append(vector_norm(dgrad))
        if method in {"H7_cohort_surgery_debt_barrier", "H7_same_span_random_debt_barrier_control"}:
            dgrad = debt_gradient(model, x_train, y_train, cohort_indices, params, int(tensors["num_classes"]))
            before = float(direction.dot(dgrad).detach().item()) if int(direction.numel()) else 0.0
            direction = direction + float(args.debt_barrier_weight) * dgrad
            after = float(direction.dot(dgrad).detach().item()) if int(direction.numel()) else 0.0
            debt_overlap_before.append(before)
            debt_overlap_after.append(after)
            debt_grad_norms.append(vector_norm(dgrad))

        primal_dual_methods = {
            "H6_layerwise_cvar_population_blend_primal_dual",
            "H6_shuffled_cvar_population_blend_primal_dual_control",
            "H7_cohort_surgery_primal_dual_safety",
            "H7_same_span_random_primal_dual_safety_control",
            "H7_cohort_surgery_primal_dual_train_temp",
            "H7_same_span_random_primal_dual_train_temp_control",
            "H7_cohort_surgery_primal_dual_train_soften_temp",
            "H7_same_span_random_primal_dual_train_soften_temp_control",
            "H7_cohort_surgery_primal_dual_soft_ece",
            "H7_same_span_random_primal_dual_soft_ece_control",
            "H7_cohort_surgery_primal_dual_soft_acc_ece",
            "H7_same_span_random_primal_dual_soft_acc_ece_control",
            "H7_cohort_surgery_primal_dual_worst_guard",
            "H7_same_span_random_primal_dual_worst_guard_control",
        }
        primal_dual_guard = None
        primal_dual_component_grads = None
        primal_dual_before = None
        if method in primal_dual_methods:
            primal_dual_guard = [
                batch_indices(int(x_train.shape[0]), int(args.cohort_size), generator, device)
                for _c in range(int(args.cohorts))
            ]
            primal_dual_component_grads, primal_dual_before = debt_component_gradients(
                model,
                x_train,
                y_train,
                primal_dual_guard,
                params,
                int(tensors["num_classes"]),
                "soft_acc_ece" if "soft_acc_ece" in method else ("soft_ece" if "soft_ece" in method else "gap_sq"),
                float(args.primal_dual_tail_fraction),
                "max" if "worst_guard" in method else "mean",
            )
            task_pred_calib = -float(args.branch_lr) * float(direction.dot(primal_dual_component_grads["calib"]).detach().item())
            task_pred_brier = -float(args.branch_lr) * float(direction.dot(primal_dual_component_grads["brier"]).detach().item())
            task_pred_tail = -float(args.branch_lr) * float(direction.dot(primal_dual_component_grads["tail"]).detach().item())
            eff_calib = dual_calib + (float(args.primal_dual_base_weight) if task_pred_calib > float(args.primal_dual_tolerance) else 0.0)
            eff_brier = dual_brier + (float(args.primal_dual_base_weight) if task_pred_brier > float(args.primal_dual_tolerance) else 0.0)
            eff_tail = dual_tail + (float(args.primal_dual_base_weight) if task_pred_tail > float(args.primal_dual_tolerance) else 0.0)
            direction = (
                direction
                + eff_calib * float(args.primal_dual_calib_weight) * primal_dual_component_grads["calib"]
                + eff_brier * primal_dual_component_grads["brier"]
                + eff_tail * float(args.primal_dual_tail_weight) * primal_dual_component_grads["tail"]
            )
            primal_dual_guard_debt_before.append(
                float(args.primal_dual_calib_weight) * float(primal_dual_before["calib"])
                + float(primal_dual_before["brier"])
                + float(args.primal_dual_tail_weight) * float(primal_dual_before["tail"])
            )

        direction, raw_norm = clipped(direction, float(args.grad_clip))
        train_accept_methods = {"H7_cohort_surgery_train_debt_accept", "H7_same_span_random_train_debt_accept_control"}
        train_exact_accept_methods = {
            "H7_cohort_surgery_train_exact_debt_accept",
            "H7_same_span_random_train_exact_debt_accept_control",
            "H6_layerwise_cvar_population_blend_exact_accept",
            "H6_shuffled_cvar_population_blend_exact_accept_control",
        }
        held_accept_methods = {"H7_cohort_surgery_held_debt_accept", "H7_same_span_random_held_debt_accept_control"}
        if method in train_accept_methods or method in train_exact_accept_methods or method in persistent_guard_methods or method in held_accept_methods:
            accept_x = x_held if method in held_accept_methods else x_train
            accept_y = y_held if method in held_accept_methods else y_train
            if method in persistent_guard_methods and persistent_guard_indices is not None:
                accept_indices = [persistent_guard_indices]
            else:
                accept_indices = [
                    batch_indices(int(accept_x.shape[0]), int(args.cohort_size), generator, device)
                    for _c in range(int(args.cohorts))
                ]
            exact_accept_gate = method in held_accept_methods or method in train_exact_accept_methods or method in persistent_guard_methods
            if exact_accept_gate:
                before_components = debt_components(model, accept_x, accept_y, accept_indices, int(tensors["num_classes"]))
            else:
                before_score = debt_score(model, accept_x, accept_y, accept_indices, int(tensors["num_classes"]))
            accepted = 0.0
            used_scale = 0.0
            upd = 0.0
            for trial in range(int(args.accept_trials)):
                scale = float(args.branch_lr) * (0.5 ** trial)
                snap = snapshot_params(params_named)
                upd = apply_flat_direction(params_named, direction, scale, 0.0)
                if exact_accept_gate:
                    after_components = debt_components(model, accept_x, accept_y, accept_indices, int(tensors["num_classes"]))
                    passes_gate = all(
                        after_components[key] <= before_components[key] + float(args.accept_tolerance)
                        for key in ("ECE", "Brier", "tail_q95")
                    )
                else:
                    after_score = debt_score(model, accept_x, accept_y, accept_indices, int(tensors["num_classes"]))
                    passes_gate = after_score <= before_score + float(args.accept_tolerance)
                if passes_gate:
                    accepted = 1.0
                    used_scale = scale
                    break
                restore_params(params_named, snap)
                upd = 0.0
            if method in held_accept_methods:
                held_debt_accepts.append(accepted)
                held_debt_accept_scales.append(used_scale)
            else:
                train_debt_accepts.append(accepted)
                train_debt_accept_scales.append(used_scale)
        elif method in primal_dual_methods:
            pred_calib = -float(args.branch_lr) * float(direction.dot(primal_dual_component_grads["calib"]).detach().item())
            pred_brier = -float(args.branch_lr) * float(direction.dot(primal_dual_component_grads["brier"]).detach().item())
            pred_tail = -float(args.branch_lr) * float(direction.dot(primal_dual_component_grads["tail"]).detach().item())
            upd = apply_flat_direction(params_named, direction, float(args.branch_lr), float(args.weight_decay))
            primal_dual_after = debt_component_values(
                model,
                x_train,
                y_train,
                primal_dual_guard,
                int(tensors["num_classes"]),
                "soft_acc_ece" if "soft_acc_ece" in method else ("soft_ece" if "soft_ece" in method else "gap_sq"),
                float(args.primal_dual_tail_fraction),
                "max" if "worst_guard" in method else "mean",
            )
            actual_calib = float(primal_dual_after["calib"]) - float(primal_dual_before["calib"])
            actual_brier = float(primal_dual_after["brier"]) - float(primal_dual_before["brier"])
            actual_tail = float(primal_dual_after["tail"]) - float(primal_dual_before["tail"])
            dual_calib = max(
                0.0,
                min(
                    float(args.primal_dual_max_lambda),
                    dual_calib + float(args.primal_dual_dual_lr) * (actual_calib - float(args.primal_dual_tolerance)),
                ),
            )
            dual_brier = max(
                0.0,
                min(
                    float(args.primal_dual_max_lambda),
                    dual_brier + float(args.primal_dual_dual_lr) * (actual_brier - float(args.primal_dual_tolerance)),
                ),
            )
            dual_tail = max(
                0.0,
                min(
                    float(args.primal_dual_max_lambda),
                    dual_tail + float(args.primal_dual_dual_lr) * (actual_tail - float(args.primal_dual_tolerance)),
                ),
            )
            primal_dual_lambda_calib_values.append(dual_calib)
            primal_dual_lambda_brier_values.append(dual_brier)
            primal_dual_lambda_tail_values.append(dual_tail)
            primal_dual_pred_calib_deltas.append(pred_calib)
            primal_dual_actual_calib_deltas.append(actual_calib)
            primal_dual_pred_brier_deltas.append(pred_brier)
            primal_dual_actual_brier_deltas.append(actual_brier)
            primal_dual_pred_tail_deltas.append(pred_tail)
            primal_dual_actual_tail_deltas.append(actual_tail)
            primal_dual_calib_sign_matches.append(float(sign_consistency(pred_calib, actual_calib)))
            primal_dual_brier_sign_matches.append(float(sign_consistency(pred_brier, actual_brier)))
            primal_dual_tail_sign_matches.append(float(sign_consistency(pred_tail, actual_tail)))
            primal_dual_guard_debt_after.append(
                float(args.primal_dual_calib_weight) * float(primal_dual_after["calib"])
                + float(primal_dual_after["brier"])
                + float(args.primal_dual_tail_weight) * float(primal_dual_after["tail"])
            )
        else:
            upd = apply_flat_direction(params_named, direction, float(args.branch_lr), float(args.weight_decay))
        raw_norms.append(raw_norm)
        update_norms.append(upd)

    return {
        "train_loss_mean_during_branch": mean(losses),
        "raw_direction_norm_mean": mean(raw_norms),
        "update_norm_mean": mean(update_norms),
        "cohort_gradient_conflict_rate": mean(conflict_rates),
        "mean_negative_pairwise_cosine": mean(neg_cosines),
        "support_projected_energy_fraction_mean": mean(support_energy_fracs),
        "layerwise_target_loss_mean": mean(layer_target_losses),
        "layerwise_h1_mse_mean": mean(layer_h1_mse),
        "layerwise_h2_mse_mean": mean(layer_h2_mse),
        "layerwise_hard_fraction_mean": mean(layer_hard_fractions),
        "debt_gradient_overlap_before_mean": mean(debt_overlap_before),
        "debt_gradient_overlap_after_mean": mean(debt_overlap_after),
        "debt_gradient_norm_mean": mean(debt_grad_norms),
        "train_debt_accept_rate": mean(train_debt_accepts),
        "train_debt_accept_scale_mean": mean(train_debt_accept_scales),
        "held_debt_accept_rate": mean(held_debt_accepts),
        "held_debt_accept_scale_mean": mean(held_debt_accept_scales),
        "primal_dual_lambda_calib_final": primal_dual_lambda_calib_values[-1] if primal_dual_lambda_calib_values else None,
        "primal_dual_lambda_brier_final": primal_dual_lambda_brier_values[-1] if primal_dual_lambda_brier_values else None,
        "primal_dual_lambda_tail_final": primal_dual_lambda_tail_values[-1] if primal_dual_lambda_tail_values else None,
        "primal_dual_lambda_calib_mean": mean(primal_dual_lambda_calib_values),
        "primal_dual_lambda_brier_mean": mean(primal_dual_lambda_brier_values),
        "primal_dual_lambda_tail_mean": mean(primal_dual_lambda_tail_values),
        "primal_dual_pred_calib_delta_mean": mean(primal_dual_pred_calib_deltas),
        "primal_dual_actual_calib_delta_mean": mean(primal_dual_actual_calib_deltas),
        "primal_dual_pred_brier_delta_mean": mean(primal_dual_pred_brier_deltas),
        "primal_dual_actual_brier_delta_mean": mean(primal_dual_actual_brier_deltas),
        "primal_dual_pred_tail_delta_mean": mean(primal_dual_pred_tail_deltas),
        "primal_dual_actual_tail_delta_mean": mean(primal_dual_actual_tail_deltas),
        "primal_dual_calib_sign_match_rate": mean(primal_dual_calib_sign_matches),
        "primal_dual_brier_sign_match_rate": mean(primal_dual_brier_sign_matches),
        "primal_dual_tail_sign_match_rate": mean(primal_dual_tail_sign_matches),
        "primal_dual_guard_debt_delta_mean": (
            (mean(primal_dual_guard_debt_after) or 0.0) - (mean(primal_dual_guard_debt_before) or 0.0)
            if primal_dual_guard_debt_after and primal_dual_guard_debt_before
            else None
        ),
    }


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch_device(str(args.device))
    torch.manual_seed(int(args.seed) + 49000)
    tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    if tensors["used_fake_data"]:
        raise RuntimeError("fake data is not allowed for v22.49A")
    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 49000, device)
    run_pretrain(model, tensors, device, args, int(args.seed))

    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    x_test = tensors["x_test"].to(device)
    y_test = tensors["y_test"].to(device)
    train_eval_n = min(int(args.held_size), int(x_train.shape[0]))
    x_train_eval = x_train[:train_eval_n]
    y_train_eval = y_train[:train_eval_n]
    pre_train_cohort = cohort_loss_values(model, x_train_eval, y_train_eval, device, int(args.cohorts), int(args.eval_batch_size))
    pre_train_eval = evaluate_tensors(model, x_train_eval, y_train_eval, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_held_cohort = cohort_loss_values(model, x_held, y_held, device, int(args.cohorts), int(args.eval_batch_size))
    pre_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    branch_diag = run_branch(str(args.method), model, tensors, device, args, int(args.seed))
    eval_model = model
    calibration_diag: dict[str, Any] = {}
    if str(args.method) in {
        "H7_cohort_surgery_debt_orthogonal_held_temp_probe",
        "H7_same_span_random_debt_orthogonal_held_temp_probe_control",
    }:
        eval_model, calibration_diag = held_temperature_probe(
            model,
            x_held,
            y_held,
            device,
            int(tensors["num_classes"]),
            int(args.eval_batch_size),
            pre_held,
        )
    if str(args.method) in {
        "H7_cohort_surgery_primal_dual_train_temp",
        "H7_same_span_random_primal_dual_train_temp_control",
        "H7_cohort_surgery_primal_dual_train_soften_temp",
        "H7_same_span_random_primal_dual_train_soften_temp_control",
    }:
        eval_model, calibration_diag = train_temperature_calibration(
            model,
            x_train_eval,
            y_train_eval,
            device,
            int(tensors["num_classes"]),
            int(args.eval_batch_size),
            pre_train_eval,
            "soften_temp" in str(args.method),
        )
    post_train_cohort = cohort_loss_values(eval_model, x_train_eval, y_train_eval, device, int(args.cohorts), int(args.eval_batch_size))
    post_held_cohort = cohort_loss_values(eval_model, x_held, y_held, device, int(args.cohorts), int(args.eval_batch_size))
    post_held = evaluate_tensors(eval_model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = evaluate_tensors(eval_model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))

    same_cohort_gain = [a - b for a, b in zip(pre_train_cohort, post_train_cohort)]
    leave_cohort_gain = [a - b for a, b in zip(pre_held_cohort, post_held_cohort)]
    row = {
        "run_label": str(args.label),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": str(args.method),
        "hypothesis": str(args.method).split("_", 1)[0],
        "is_control": int(str(args.method).endswith("_control") or str(args.method) == "H7_same_span_random_control"),
        "device": str(args.device),
        "train_size": int(args.train_size),
        "held_size": int(args.held_size),
        "test_size": int(args.test_size),
        "pretrain_steps": int(args.pretrain_steps),
        "branch_steps": int(args.branch_steps),
        "cohorts": int(args.cohorts),
        "cohort_size": int(args.cohort_size),
        "support_rank": int(args.support_rank),
        "source_kind": tensors["source_kind"],
        "used_fake_data": tensors["used_fake_data"],
        "held_NLL_pre": pre_held["NLL"],
        "held_NLL_post": post_held["NLL"],
        "held_NLL_gain": pre_held["NLL"] - post_held["NLL"],
        "held_accuracy_pre": pre_held["accuracy"],
        "held_accuracy_post": post_held["accuracy"],
        "held_accuracy_delta": post_held["accuracy"] - pre_held["accuracy"],
        "held_ECE_delta": post_held["ECE"] - pre_held["ECE"],
        "held_Brier_delta": post_held["Brier"] - pre_held["Brier"],
        "held_tail_q95_delta": post_held["tail_q95"] - pre_held["tail_q95"],
        "held_tail_q99_delta": post_held["tail_q99"] - pre_held["tail_q99"],
        "held_margin_q10_delta": post_held["margin_q10"] - pre_held["margin_q10"],
        "test_NLL_pre": pre_test["NLL"],
        "test_NLL_post": post_test["NLL"],
        "test_NLL_gain": pre_test["NLL"] - post_test["NLL"],
        "test_accuracy_delta": post_test["accuracy"] - pre_test["accuracy"],
        "same_cohort_gain_mean": mean(same_cohort_gain),
        "same_cohort_gain_min": min(same_cohort_gain) if same_cohort_gain else "",
        "leave_cohort_gain_mean": mean(leave_cohort_gain),
        "leave_cohort_gain_min": min(leave_cohort_gain) if leave_cohort_gain else "",
        "v22_49A_sanity_no_debt": int(
            (post_held["ECE"] - pre_held["ECE"]) <= 1.0e-8
            and (post_held["Brier"] - pre_held["Brier"]) <= 1.0e-8
            and (post_held["tail_q95"] - pre_held["tail_q95"]) <= 1.0e-8
        ),
        "claim_limit": "Stage1 train-only small sanity row; not official hard-task promotion",
    }
    row.update({k: "" if v is None else v for k, v in branch_diag.items()})
    row.update({k: "" if v is None else v for k, v in calibration_diag.items()})
    out_path = CHUNK_ROOT / f"v22_49A_{safe_fragment(args.label)}_summary.csv"
    write_rows(out_path, [row])
    trace_path = CHUNK_ROOT / f"v22_49A_{safe_fragment(args.label)}_cohort_trace.json"
    write_json(
        trace_path,
        {
            "pre_train_cohort_NLL": pre_train_cohort,
            "post_train_cohort_NLL": post_train_cohort,
            "same_cohort_gain": same_cohort_gain,
            "pre_held_cohort_NLL": pre_held_cohort,
            "post_held_cohort_NLL": post_held_cohort,
            "leave_cohort_gain": leave_cohort_gain,
        },
    )
    return {"status": "pass", "summary": str(out_path), "trace": str(trace_path), "row": row}


def build_specs(args: argparse.Namespace, prefix: str = "stage1") -> list[dict[str, Any]]:
    specs = []
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    methods = split_csv(str(args.methods), str)
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                label = f"v22_49A_{prefix}_{method}_{dataset}_s{seed}"
                specs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "label": label,
                    }
                )
    return specs


def command_for_collect(spec: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        PYTHON,
        "experiments/run_v22_49A_broader_related_work_map.py",
        "--stage",
        "collect",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--method",
        str(spec["method"]),
        "--label",
        str(spec["label"]),
        "--device",
        "cuda:0",
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--test-size",
        str(args.test_size),
        "--hidden",
        str(args.hidden),
        "--pretrain-steps",
        str(args.pretrain_steps),
        "--branch-steps",
        str(args.branch_steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--cohorts",
        str(args.cohorts),
        "--cohort-size",
        str(args.cohort_size),
        "--support-rank",
        str(args.support_rank),
        "--support-ema-beta",
        str(args.support_ema_beta),
        "--pretrain-lr",
        str(args.pretrain_lr),
        "--branch-lr",
        str(args.branch_lr),
        "--layerwise-lr",
        str(args.layerwise_lr),
        "--layerwise-target-eta",
        str(args.layerwise_target_eta),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--hard-fraction",
        str(args.hard_fraction),
        "--debt-barrier-weight",
        str(args.debt_barrier_weight),
        "--accept-trials",
        str(args.accept_trials),
        "--accept-tolerance",
        str(args.accept_tolerance),
        "--persistent-guard-fraction",
        str(args.persistent_guard_fraction),
        "--surgery-blend-alpha",
        str(args.surgery_blend_alpha),
        "--primal-dual-dual-lr",
        str(args.primal_dual_dual_lr),
        "--primal-dual-base-weight",
        str(args.primal_dual_base_weight),
        "--primal-dual-calib-weight",
        str(args.primal_dual_calib_weight),
        "--primal-dual-tail-weight",
        str(args.primal_dual_tail_weight),
        "--primal-dual-tail-fraction",
        str(args.primal_dual_tail_fraction),
        "--primal-dual-tolerance",
        str(args.primal_dual_tolerance),
        "--primal-dual-max-lambda",
        str(args.primal_dual_max_lambda),
    ]


def run_one_subprocess(spec: dict[str, Any], args: argparse.Namespace, gpu: int) -> dict[str, Any]:
    ensure_out()
    cmd = command_for_collect(spec, args)
    task_id = safe_fragment(str(spec["label"]))
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    elapsed = time.time() - start
    status = "pass" if proc.returncode == 0 else "fail"
    files = f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}"
    append_exec(
        "CUDA_VISIBLE_DEVICES="
        + str(gpu)
        + " "
        + " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status=status,
        gpu=f"cuda:{gpu}",
        exit_code=proc.returncode,
        files=files,
        note=f"elapsed_sec={elapsed:.3f}; method={spec['method']}; dataset={spec['dataset']}; seed={spec['seed']}",
    )
    return {
        "label": spec["label"],
        "dataset": spec["dataset"],
        "seed": spec["seed"],
        "method": spec["method"],
        "gpu": gpu,
        "status": status,
        "exit_code": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def dispatch_matrix(args: argparse.Namespace, *, prefix: str, task_name: str) -> dict[str, Any]:
    ensure_out()
    specs = build_specs(args, prefix=prefix)
    write_rows(OUT_ROOT / f"v22_49A_{prefix}_specs.csv", specs)
    append_exec(
        f"{task_name}_dispatch",
        task_id=f"{task_name}_dispatch",
        status="started",
        gpu=str(args.gpus),
        files=f"results/v22_49A/v22_49A_{prefix}_specs.csv, results/v22_49A/chunks",
        note=f"rows={len(specs)}; workers={args.workers}; datasets={args.datasets}; seeds={args.seeds}; methods={args.methods}",
    )
    gpus = split_csv(str(args.gpus), int)
    if not gpus:
        gpus = [0]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futs = []
        for idx, spec in enumerate(specs):
            futs.append(pool.submit(run_one_subprocess, spec, args, gpus[idx % len(gpus)]))
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            write_rows(OUT_ROOT / f"v22_49A_{prefix}_dispatch_status.csv", results)
    failures = [r for r in results if r.get("status") != "pass"]
    append_exec(
        f"{task_name}_dispatch",
        task_id=f"{task_name}_dispatch",
        status="completed" if not failures else "fail",
        gpu=str(args.gpus),
        files=f"results/v22_49A/v22_49A_{prefix}_dispatch_status.csv",
        note=f"rows={len(results)}; failures={len(failures)}",
        exit_code=0 if not failures else 1,
    )
    return {"rows": len(results), "failures": len(failures), "status": "pass" if not failures else "fail"}


def dispatch_stage1(args: argparse.Namespace) -> dict[str, Any]:
    return dispatch_matrix(args, prefix="stage1", task_name="stage1")


def dispatch_repair(args: argparse.Namespace) -> dict[str, Any]:
    return dispatch_matrix(args, prefix="repair", task_name="repair")


def collect_stage1_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_49A_*_summary.csv")):
        for row in read_rows(path):
            if str(row.get("run_label", "")).startswith("v22_49A_stage1_"):
                rows.append(row)
    return rows


def collect_rows_for_matrix(prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    marker = f"v22_49A_{prefix}_"
    for path in sorted(CHUNK_ROOT.glob("v22_49A_*_summary.csv")):
        for row in read_rows(path):
            if str(row.get("run_label", "")).startswith(marker):
                rows.append(row)
    return rows


def collect_repair_rows() -> list[dict[str, str]]:
    return collect_rows_for_matrix("repair")


def pair_control_method(method: str) -> str:
    return {
        "H2_support_native_ema": "H2_random_support_control",
        "H6_layerwise_target": "H6_shuffled_layer_target_control",
        "H7_cohort_surgery": "H7_same_span_random_control",
        "H6_layerwise_cvar_target": "H6_shuffled_cvar_target_control",
        "H6_layerwise_cvar_target_train_exact_accept": "H6_shuffled_cvar_target_train_exact_accept_control",
        "H6_layerwise_cvar_population_blend_exact_accept": "H6_shuffled_cvar_population_blend_exact_accept_control",
        "H6_layerwise_cvar_population_blend_primal_dual": "H6_shuffled_cvar_population_blend_primal_dual_control",
        "H7_cohort_surgery_debt_orthogonal": "H7_same_span_random_debt_orthogonal_control",
        "H7_cohort_surgery_debt_barrier": "H7_same_span_random_debt_barrier_control",
        "H7_cohort_surgery_train_debt_accept": "H7_same_span_random_train_debt_accept_control",
        "H7_cohort_surgery_train_exact_debt_accept": "H7_same_span_random_train_exact_debt_accept_control",
        "H7_cohort_surgery_persistent_guard_accept": "H7_same_span_random_persistent_guard_accept_control",
        "H7_cohort_surgery_held_debt_accept": "H7_same_span_random_held_debt_accept_control",
        "H7_cohort_surgery_debt_orthogonal_held_temp_probe": "H7_same_span_random_debt_orthogonal_held_temp_probe_control",
        "H7_cohort_surgery_blend_debt_orthogonal": "H7_same_span_random_blend_debt_orthogonal_control",
        "H7_cohort_surgery_primal_dual_safety": "H7_same_span_random_primal_dual_safety_control",
        "H7_cohort_surgery_primal_dual_train_temp": "H7_same_span_random_primal_dual_train_temp_control",
        "H7_cohort_surgery_primal_dual_train_soften_temp": "H7_same_span_random_primal_dual_train_soften_temp_control",
        "H7_cohort_surgery_primal_dual_soft_ece": "H7_same_span_random_primal_dual_soft_ece_control",
        "H7_cohort_surgery_primal_dual_soft_acc_ece": "H7_same_span_random_primal_dual_soft_acc_ece_control",
        "H7_cohort_surgery_primal_dual_worst_guard": "H7_same_span_random_primal_dual_worst_guard_control",
    }.get(method, "")


def summarize_stage1() -> dict[str, Any]:
    ensure_out()
    rows = collect_stage1_rows()
    write_rows(OUT_ROOT / "v22_49A_stage1_sanity_matrix.csv", rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("hypothesis", "")), str(row.get("method", "")))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (hypothesis, method), mrows in sorted(grouped.items()):
        summary.append(
            {
                "hypothesis": hypothesis,
                "method": method,
                "rows": len(mrows),
                "is_control": mrows[0].get("is_control", "") if mrows else "",
                "mean_leave_cohort_gain": mean(fval(r.get("leave_cohort_gain_mean")) for r in mrows),
                "median_leave_cohort_gain": median(fval(r.get("leave_cohort_gain_mean")) for r in mrows),
                "positive_leave_cohort_rows": sum((fval(r.get("leave_cohort_gain_mean"), 0.0) or 0.0) > 0.0 for r in mrows),
                "mean_same_cohort_gain": mean(fval(r.get("same_cohort_gain_mean")) for r in mrows),
                "mean_held_NLL_gain": mean(fval(r.get("held_NLL_gain")) for r in mrows),
                "mean_test_NLL_gain": mean(fval(r.get("test_NLL_gain")) for r in mrows),
                "sanity_no_debt_rows": sum(iflag(r.get("v22_49A_sanity_no_debt")) for r in mrows),
                "mean_conflict_rate": mean(fval(r.get("cohort_gradient_conflict_rate")) for r in mrows),
                "mean_negative_pairwise_cosine": mean(fval(r.get("mean_negative_pairwise_cosine")) for r in mrows),
                "mean_support_projected_energy_fraction": mean(fval(r.get("support_projected_energy_fraction_mean")) for r in mrows),
                "claim_limit": "Stage1 sanity aggregate; not official route promotion",
            }
        )
    write_rows(OUT_ROOT / "v22_49A_stage1_method_summary.csv", summary)

    rows_by_key = {
        (r.get("dataset"), r.get("seed"), r.get("method")): r
        for r in rows
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        control = pair_control_method(method)
        if not control:
            continue
        baseline = rows_by_key.get((row.get("dataset"), row.get("seed"), "H1_population_mean_gradient"))
        ctrl = rows_by_key.get((row.get("dataset"), row.get("seed"), control))
        if not baseline or not ctrl:
            continue
        real_gain = fval(row.get("leave_cohort_gain_mean"), 0.0) or 0.0
        ctrl_gain = fval(ctrl.get("leave_cohort_gain_mean"), 0.0) or 0.0
        h1_gain = fval(baseline.get("leave_cohort_gain_mean"), 0.0) or 0.0
        comparisons.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "hypothesis": row.get("hypothesis"),
                "method": method,
                "control_method": control,
                "leave_cohort_gain": real_gain,
                "control_leave_cohort_gain": ctrl_gain,
                "H1_leave_cohort_gain": h1_gain,
                "beats_control": int(real_gain > ctrl_gain),
                "beats_H1": int(real_gain > h1_gain),
                "beats_both": int(real_gain > ctrl_gain and real_gain > h1_gain),
                "sanity_no_debt": row.get("v22_49A_sanity_no_debt"),
                "control_sanity_no_debt": ctrl.get("v22_49A_sanity_no_debt"),
            }
        )
    write_rows(OUT_ROOT / "v22_49A_stage1_pairwise_comparison.csv", comparisons)

    route: dict[str, Any] = {
        "timestamp": now_sg(),
        "stage1_rows": len(rows),
        "stage1_failures": len([r for r in rows if r.get("status") == "fail"]),
        "promotion_allowed": False,
        "claim_limit": "v22.49A Stage1 cheap sanity tests on Tier0 cached datasets; hard-task route not entered unless H6/H7 beat H1 and matched controls",
        "source_artifacts": [
            "results/v22_49A/v22_49A_stage1_sanity_matrix.csv",
            "results/v22_49A/v22_49A_stage1_method_summary.csv",
            "results/v22_49A/v22_49A_stage1_pairwise_comparison.csv",
        ],
    }
    for hyp in ["H2", "H6", "H7"]:
        cmp_rows = [r for r in comparisons if r.get("hypothesis") == hyp]
        route[f"{hyp}_comparison_rows"] = len(cmp_rows)
        route[f"{hyp}_beats_control_rows"] = sum(iflag(r.get("beats_control")) for r in cmp_rows)
        route[f"{hyp}_beats_H1_rows"] = sum(iflag(r.get("beats_H1")) for r in cmp_rows)
        route[f"{hyp}_beats_both_rows"] = sum(iflag(r.get("beats_both")) for r in cmp_rows)
        route[f"{hyp}_sanity_no_debt_rows"] = sum(iflag(r.get("sanity_no_debt")) for r in cmp_rows)
    h6_h7_rows = [r for r in comparisons if r.get("hypothesis") in {"H6", "H7"}]
    h6_h7_beats = sum(iflag(r.get("beats_both")) for r in h6_h7_rows)
    h6_h7_no_debt_and_beats = sum(iflag(r.get("beats_both")) and iflag(r.get("sanity_no_debt")) for r in h6_h7_rows)
    route["stage2_entered"] = False
    route["stage2_reason"] = (
        "not_entered: H6/H7 require consistent leave-cohort gain over H1 and matched controls; "
        f"beats_both={h6_h7_beats}/{len(h6_h7_rows)}, no_debt_and_beats={h6_h7_no_debt_and_beats}/{len(h6_h7_rows)}"
    )
    write_json(OUT_ROOT / "v22_49A_stage1_route.json", route)
    append_exec(
        "summarize_stage1",
        task_id="summarize_stage1",
        status="pass",
        gpu="cpu",
        files="results/v22_49A/v22_49A_stage1_sanity_matrix.csv, results/v22_49A/v22_49A_stage1_method_summary.csv, results/v22_49A/v22_49A_stage1_pairwise_comparison.csv, results/v22_49A/v22_49A_stage1_route.json",
        note=f"rows={len(rows)}; comparisons={len(comparisons)}; {route['stage2_reason']}",
    )
    return route


def summarize_repair(prefix: str = "repair") -> dict[str, Any]:
    ensure_out()
    rows = collect_rows_for_matrix(prefix)
    write_rows(OUT_ROOT / f"v22_49A_{prefix}_sanity_matrix.csv", rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("hypothesis", "")), str(row.get("method", "")))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (hypothesis, method), mrows in sorted(grouped.items()):
        summary.append(
            {
                "hypothesis": hypothesis,
                "method": method,
                "rows": len(mrows),
                "is_control": mrows[0].get("is_control", "") if mrows else "",
                "mean_leave_cohort_gain": mean(fval(r.get("leave_cohort_gain_mean")) for r in mrows),
                "median_leave_cohort_gain": median(fval(r.get("leave_cohort_gain_mean")) for r in mrows),
                "positive_leave_cohort_rows": sum((fval(r.get("leave_cohort_gain_mean"), 0.0) or 0.0) > 0.0 for r in mrows),
                "mean_same_cohort_gain": mean(fval(r.get("same_cohort_gain_mean")) for r in mrows),
                "mean_held_NLL_gain": mean(fval(r.get("held_NLL_gain")) for r in mrows),
                "mean_test_NLL_gain": mean(fval(r.get("test_NLL_gain")) for r in mrows),
                "sanity_no_debt_rows": sum(iflag(r.get("v22_49A_sanity_no_debt")) for r in mrows),
                "mean_conflict_rate": mean(fval(r.get("cohort_gradient_conflict_rate")) for r in mrows),
                "mean_negative_pairwise_cosine": mean(fval(r.get("mean_negative_pairwise_cosine")) for r in mrows),
                "mean_debt_overlap_before": mean(fval(r.get("debt_gradient_overlap_before_mean")) for r in mrows),
                "mean_debt_overlap_after": mean(fval(r.get("debt_gradient_overlap_after_mean")) for r in mrows),
                "mean_layerwise_hard_fraction": mean(fval(r.get("layerwise_hard_fraction_mean")) for r in mrows),
                "train_debt_accept_rate": mean(fval(r.get("train_debt_accept_rate")) for r in mrows),
                "train_debt_accept_scale_mean": mean(fval(r.get("train_debt_accept_scale_mean")) for r in mrows),
                "held_debt_accept_rate": mean(fval(r.get("held_debt_accept_rate")) for r in mrows),
                "held_debt_accept_scale_mean": mean(fval(r.get("held_debt_accept_scale_mean")) for r in mrows),
                "held_temp_probe_feasible_rows": sum(iflag(r.get("held_temp_probe_feasible")) for r in mrows),
                "mean_held_temp_probe_temperature": mean(fval(r.get("held_temp_probe_temperature")) for r in mrows),
                "mean_held_temp_probe_violation_sum": mean(fval(r.get("held_temp_probe_violation_sum")) for r in mrows),
                "train_temp_feasible_rows": sum(iflag(r.get("train_temp_feasible")) for r in mrows),
                "mean_train_temp_temperature": mean(fval(r.get("train_temp_temperature")) for r in mrows),
                "mean_train_temp_violation_sum": mean(fval(r.get("train_temp_violation_sum")) for r in mrows),
                "primal_dual_lambda_calib_final_mean": mean(fval(r.get("primal_dual_lambda_calib_final")) for r in mrows),
                "primal_dual_lambda_brier_final_mean": mean(fval(r.get("primal_dual_lambda_brier_final")) for r in mrows),
                "primal_dual_lambda_tail_final_mean": mean(fval(r.get("primal_dual_lambda_tail_final")) for r in mrows),
                "primal_dual_calib_sign_match_rate": mean(fval(r.get("primal_dual_calib_sign_match_rate")) for r in mrows),
                "primal_dual_brier_sign_match_rate": mean(fval(r.get("primal_dual_brier_sign_match_rate")) for r in mrows),
                "primal_dual_tail_sign_match_rate": mean(fval(r.get("primal_dual_tail_sign_match_rate")) for r in mrows),
                "primal_dual_guard_debt_delta_mean": mean(fval(r.get("primal_dual_guard_debt_delta_mean")) for r in mrows),
                "claim_limit": "v22.49A repair sanity aggregate; not official route promotion",
            }
        )
    write_rows(OUT_ROOT / f"v22_49A_{prefix}_method_summary.csv", summary)

    rows_by_key = {
        (r.get("dataset"), r.get("seed"), r.get("method")): r
        for r in rows
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        control = pair_control_method(method)
        if not control:
            continue
        baseline = rows_by_key.get((row.get("dataset"), row.get("seed"), "H1_population_mean_gradient"))
        ctrl = rows_by_key.get((row.get("dataset"), row.get("seed"), control))
        if not baseline or not ctrl:
            continue
        real_gain = fval(row.get("leave_cohort_gain_mean"), 0.0) or 0.0
        ctrl_gain = fval(ctrl.get("leave_cohort_gain_mean"), 0.0) or 0.0
        h1_gain = fval(baseline.get("leave_cohort_gain_mean"), 0.0) or 0.0
        comparisons.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "hypothesis": row.get("hypothesis"),
                "method": method,
                "control_method": control,
                "leave_cohort_gain": real_gain,
                "control_leave_cohort_gain": ctrl_gain,
                "H1_leave_cohort_gain": h1_gain,
                "beats_control": int(real_gain > ctrl_gain),
                "beats_H1": int(real_gain > h1_gain),
                "beats_both": int(real_gain > ctrl_gain and real_gain > h1_gain),
                "sanity_no_debt": row.get("v22_49A_sanity_no_debt"),
                "control_sanity_no_debt": ctrl.get("v22_49A_sanity_no_debt"),
                "debt_overlap_before": row.get("debt_gradient_overlap_before_mean"),
                "debt_overlap_after": row.get("debt_gradient_overlap_after_mean"),
            }
        )
    write_rows(OUT_ROOT / f"v22_49A_{prefix}_pairwise_comparison.csv", comparisons)

    route: dict[str, Any] = {
        "timestamp": now_sg(),
        "prefix": prefix,
        "repair_rows": len(rows),
        "repair_failures": len([r for r in rows if r.get("status") == "fail"]),
        "promotion_allowed": False,
        "claim_limit": "v22.49A repair sanity tests on Tier0 cached datasets; hard-task route requires robust H6/H7 leave-cohort wins plus no-debt",
        "source_artifacts": [
            f"results/v22_49A/v22_49A_{prefix}_sanity_matrix.csv",
            f"results/v22_49A/v22_49A_{prefix}_method_summary.csv",
            f"results/v22_49A/v22_49A_{prefix}_pairwise_comparison.csv",
        ],
    }
    for hyp in ["H6", "H7"]:
        cmp_rows = [r for r in comparisons if r.get("hypothesis") == hyp]
        route[f"{hyp}_comparison_rows"] = len(cmp_rows)
        route[f"{hyp}_beats_control_rows"] = sum(iflag(r.get("beats_control")) for r in cmp_rows)
        route[f"{hyp}_beats_H1_rows"] = sum(iflag(r.get("beats_H1")) for r in cmp_rows)
        route[f"{hyp}_beats_both_rows"] = sum(iflag(r.get("beats_both")) for r in cmp_rows)
        route[f"{hyp}_no_debt_and_beats_rows"] = sum(iflag(r.get("beats_both")) and iflag(r.get("sanity_no_debt")) for r in cmp_rows)
    h6_h7_rows = [r for r in comparisons if r.get("hypothesis") in {"H6", "H7"}]
    h6_h7_beats = sum(iflag(r.get("beats_both")) for r in h6_h7_rows)
    h6_h7_no_debt_and_beats = sum(iflag(r.get("beats_both")) and iflag(r.get("sanity_no_debt")) for r in h6_h7_rows)
    uses_held_calibration_probe = any("held_temp_probe" in str(r.get("method", "")) for r in rows)
    route["diagnostic_uses_held_calibration_probe"] = bool(uses_held_calibration_probe)
    route["stage2_entered_after_repair"] = bool(
        h6_h7_beats >= 10 and h6_h7_no_debt_and_beats >= 6 and not uses_held_calibration_probe
    )
    if uses_held_calibration_probe:
        route["claim_limit"] = (
            route["claim_limit"]
            + "; held_temp_probe rows select a temperature on the held split and are diagnostic upper-bound only"
        )
    route["stage2_repair_reason"] = (
        f"{'entered' if route['stage2_entered_after_repair'] else 'not_entered'}: "
        "repair requires H6/H7 to consistently beat H1 and matched controls with no-debt; "
        f"beats_both={h6_h7_beats}/{len(h6_h7_rows)}, no_debt_and_beats={h6_h7_no_debt_and_beats}/{len(h6_h7_rows)}"
        + ("; diagnostic_held_calibration_probe=true, so official Stage2 remains blocked" if uses_held_calibration_probe else "")
    )
    write_json(OUT_ROOT / f"v22_49A_{prefix}_route.json", route)
    append_exec(
        f"summarize_{prefix}",
        task_id=f"summarize_{prefix}",
        status="pass",
        gpu="cpu",
        files=f"results/v22_49A/v22_49A_{prefix}_sanity_matrix.csv, results/v22_49A/v22_49A_{prefix}_method_summary.csv, results/v22_49A/v22_49A_{prefix}_pairwise_comparison.csv, results/v22_49A/v22_49A_{prefix}_route.json",
        note=f"rows={len(rows)}; comparisons={len(comparisons)}; {route['stage2_repair_reason']}",
    )
    return route


def write_recap() -> None:
    ensure_out()
    stage0 = read_rows(OUT_ROOT / "v22_49A_stage0_historical_explanation_matrix.csv")
    stage1 = read_rows(OUT_ROOT / "v22_49A_stage1_sanity_matrix.csv")
    summary = read_rows(OUT_ROOT / "v22_49A_stage1_method_summary.csv")
    comparisons = read_rows(OUT_ROOT / "v22_49A_stage1_pairwise_comparison.csv")
    route = read_json(OUT_ROOT / "v22_49A_stage1_route.json")
    repair_rows = read_rows(OUT_ROOT / "v22_49A_repair_sanity_matrix.csv")
    repair_summary = read_rows(OUT_ROOT / "v22_49A_repair_method_summary.csv")
    repair_comparisons = read_rows(OUT_ROOT / "v22_49A_repair_pairwise_comparison.csv")
    repair_route = read_json(OUT_ROOT / "v22_49A_repair_route.json")
    extra_repair_prefixes: list[str] = []
    for path in sorted(OUT_ROOT.glob("v22_49A_repair_*_route.json")):
        prefix = path.name.removeprefix("v22_49A_").removesuffix("_route.json")
        if prefix != "repair":
            extra_repair_prefixes.append(prefix)
    repair_rollup: list[dict[str, Any]] = []
    for prefix in ["repair"] + extra_repair_prefixes:
        prefix_summary = read_rows(OUT_ROOT / f"v22_49A_{prefix}_method_summary.csv")
        prefix_route = read_json(OUT_ROOT / f"v22_49A_{prefix}_route.json")
        for row in prefix_summary:
            method = str(row.get("method", ""))
            if not (method.startswith("H6_layerwise") or method.startswith("H7_cohort")):
                continue
            repair_rollup.append(
                {
                    "prefix": prefix,
                    "method": method,
                    "rows": row.get("rows", ""),
                    "mean_leave_cohort_gain": row.get("mean_leave_cohort_gain", ""),
                    "positive_leave_cohort_rows": row.get("positive_leave_cohort_rows", ""),
                    "sanity_no_debt_rows": row.get("sanity_no_debt_rows", ""),
                    "beats_both": prefix_route.get("H7_beats_both_rows", "") if method.startswith("H7") else prefix_route.get("H6_beats_both_rows", ""),
                    "no_debt_and_beats": prefix_route.get("H7_no_debt_and_beats_rows", "") if method.startswith("H7") else prefix_route.get("H6_no_debt_and_beats_rows", ""),
                    "route": prefix_route.get("stage2_repair_reason", ""),
                }
            )
    h7_rollup = [row for row in repair_rollup if str(row.get("method", "")).startswith("H7_cohort")]
    best_gain_row = max(
        h7_rollup,
        key=lambda row: fval(row.get("mean_leave_cohort_gain"), -float("inf")) or -float("inf"),
        default={},
    )
    best_nodebt_row = max(
        h7_rollup,
        key=lambda row: (
            int(fval(row.get("sanity_no_debt_rows"), 0.0) or 0.0),
            fval(row.get("mean_leave_cohort_gain"), -float("inf")) or -float("inf"),
        ),
        default={},
    )
    primal_dual_rollup = [
        row for row in h7_rollup if "primal_dual" in str(row.get("method", ""))
    ]
    best_primal_dual_row = max(
        primal_dual_rollup,
        key=lambda row: (
            int(fval(row.get("no_debt_and_beats"), 0.0) or 0.0),
            fval(row.get("mean_leave_cohort_gain"), -float("inf")) or -float("inf"),
        ),
        default={},
    )
    best_primal_dual_summary = {}
    if best_primal_dual_row:
        for row in read_rows(
            OUT_ROOT / f"v22_49A_{best_primal_dual_row.get('prefix', '')}_method_summary.csv"
        ):
            if row.get("method") == best_primal_dual_row.get("method"):
                best_primal_dual_summary = row
                break
    h7_train_temp_rollup = [
        row
        for row in h7_rollup
        if row.get("method")
        in {
            "H7_cohort_surgery_primal_dual_train_temp",
            "H7_cohort_surgery_primal_dual_train_soften_temp",
        }
    ]
    best_h7_train_temp_row = max(
        h7_train_temp_rollup,
        key=lambda row: (
            int(fval(row.get("no_debt_and_beats"), 0.0) or 0.0),
            fval(row.get("mean_leave_cohort_gain"), -float("inf")) or -float("inf"),
        ),
        default={},
    )
    best_h7_train_temp_summary = {}
    if best_h7_train_temp_row:
        for row in read_rows(
            OUT_ROOT / f"v22_49A_{best_h7_train_temp_row.get('prefix', '')}_method_summary.csv"
        ):
            if row.get("method") == best_h7_train_temp_row.get("method"):
                best_h7_train_temp_summary = row
                break
    if best_h7_train_temp_row:
        h7_train_temp_insight = (
            f"- H7 train-temperature 复盘：最佳 train-temp repair 是 "
            f"`{best_h7_train_temp_row.get('prefix', '')}/{best_h7_train_temp_row.get('method', '')}`，"
            f"mean leave-cohort gain={best_h7_train_temp_row.get('mean_leave_cohort_gain', '')}，"
            f"no_debt_and_beats={best_h7_train_temp_row.get('no_debt_and_beats', '')}，"
            f"train_temp_feasible_rows={best_h7_train_temp_summary.get('train_temp_feasible_rows', '')}，"
            f"mean_train_temp_temperature={best_h7_train_temp_summary.get('mean_train_temp_temperature', '')}，"
            f"mean_train_temp_violation_sum={best_h7_train_temp_summary.get('mean_train_temp_violation_sum', '')}。"
            "温度只在 train calibration subset 上选择；若 held no-debt 仍不提升，说明校准后处理也无法弥合 train/held safety gate。"
        )
    else:
        h7_train_temp_insight = "- H7 train-temperature 复盘：尚未运行 train-only temperature calibration repair sweep。"
    h6_exact_rollup = [
        row
        for row in repair_rollup
        if row.get("method") == "H6_layerwise_cvar_target_train_exact_accept"
    ]
    best_h6_exact_row = max(
        h6_exact_rollup,
        key=lambda row: (
            int(fval(row.get("no_debt_and_beats"), 0.0) or 0.0),
            fval(row.get("mean_leave_cohort_gain"), -float("inf")) or -float("inf"),
        ),
        default={},
    )
    best_h6_exact_summary = {}
    if best_h6_exact_row:
        for row in read_rows(
            OUT_ROOT / f"v22_49A_{best_h6_exact_row.get('prefix', '')}_method_summary.csv"
        ):
            if row.get("method") == best_h6_exact_row.get("method"):
                best_h6_exact_summary = row
                break
    if best_h6_exact_row:
        h6_exact_insight = (
            f"- H6 safety-accepted local target 复盘：最佳 H6 exact-accept repair 是 "
            f"`{best_h6_exact_row.get('prefix', '')}/{best_h6_exact_row.get('method', '')}`，"
            f"mean leave-cohort gain={best_h6_exact_row.get('mean_leave_cohort_gain', '')}，"
            f"no_debt_and_beats={best_h6_exact_row.get('no_debt_and_beats', '')}，"
            f"train_accept_rate={best_h6_exact_summary.get('train_debt_accept_rate', '')}，"
            f"train_accept_scale_mean={best_h6_exact_summary.get('train_debt_accept_scale_mean', '')}。"
            "该行用于检验 H6 layerwise target 的安全债务能否由 train-only exact acceptance 缓解；"
            "若收益显著降低或 no_debt_and_beats 仍不足，说明当前 local target 方向本身还没有稳定转化为 held-safe transfer。"
        )
    else:
        h6_exact_insight = "- H6 safety-accepted local target 复盘：尚未运行 H6 exact-accept repair sweep。"
    h6_blend_rollup = [
        row
        for row in repair_rollup
        if row.get("method") == "H6_layerwise_cvar_population_blend_exact_accept"
    ]
    best_h6_blend_row = max(
        h6_blend_rollup,
        key=lambda row: (
            int(fval(row.get("no_debt_and_beats"), 0.0) or 0.0),
            fval(row.get("mean_leave_cohort_gain"), -float("inf")) or -float("inf"),
        ),
        default={},
    )
    best_h6_blend_summary = {}
    if best_h6_blend_row:
        for row in read_rows(
            OUT_ROOT / f"v22_49A_{best_h6_blend_row.get('prefix', '')}_method_summary.csv"
        ):
            if row.get("method") == best_h6_blend_row.get("method"):
                best_h6_blend_summary = row
                break
    if best_h6_blend_row:
        h6_blend_insight = (
            f"- H6 population-blend 复盘：最佳 H6 population-blend repair 是 "
            f"`{best_h6_blend_row.get('prefix', '')}/{best_h6_blend_row.get('method', '')}`，"
            f"mean leave-cohort gain={best_h6_blend_row.get('mean_leave_cohort_gain', '')}，"
            f"no_debt_and_beats={best_h6_blend_row.get('no_debt_and_beats', '')}，"
            f"train_accept_rate={best_h6_blend_summary.get('train_debt_accept_rate', '')}，"
            f"train_accept_scale_mean={best_h6_blend_summary.get('train_debt_accept_scale_mean', '')}。"
            "该修复把 H1 population gradient 作为主方向，把 H6 local target 作为同范数残差混入；"
            "若仍不能超过 H1，说明 H6 的信号更像独立弱 baseline，而不是当前设置下的可叠加 residual signal。"
        )
    else:
        h6_blend_insight = "- H6 population-blend 复盘：尚未运行 H6 population-blend repair sweep。"
    h6_pd_rollup = [
        row
        for row in repair_rollup
        if row.get("method") == "H6_layerwise_cvar_population_blend_primal_dual"
    ]
    best_h6_pd_row = max(
        h6_pd_rollup,
        key=lambda row: (
            int(fval(row.get("no_debt_and_beats"), 0.0) or 0.0),
            fval(row.get("mean_leave_cohort_gain"), -float("inf")) or -float("inf"),
        ),
        default={},
    )
    best_h6_pd_summary = {}
    if best_h6_pd_row:
        for row in read_rows(
            OUT_ROOT / f"v22_49A_{best_h6_pd_row.get('prefix', '')}_method_summary.csv"
        ):
            if row.get("method") == best_h6_pd_row.get("method"):
                best_h6_pd_summary = row
                break
    if best_h6_pd_row:
        h6_pd_insight = (
            f"- H6 population-blend primal-dual 复盘：最佳 H6 primal-dual blend 是 "
            f"`{best_h6_pd_row.get('prefix', '')}/{best_h6_pd_row.get('method', '')}`，"
            f"mean leave-cohort gain={best_h6_pd_row.get('mean_leave_cohort_gain', '')}，"
            f"no_debt_and_beats={best_h6_pd_row.get('no_debt_and_beats', '')}，"
            f"sign-match calib={best_h6_pd_summary.get('primal_dual_calib_sign_match_rate', '')}、"
            f"brier={best_h6_pd_summary.get('primal_dual_brier_sign_match_rate', '')}、"
            f"tail={best_h6_pd_summary.get('primal_dual_tail_sign_match_rate', '')}，"
            f"guard_debt_delta_mean={best_h6_pd_summary.get('primal_dual_guard_debt_delta_mean', '')}。"
            "该行用于检验 H6 residual 是否需要柔性 constrained flow，而不是 exact line-search gate。"
        )
    else:
        h6_pd_insight = "- H6 population-blend primal-dual 复盘：尚未运行 H6 population-blend primal-dual repair sweep。"

    def row_for(method: str) -> dict[str, Any]:
        for row in summary:
            if row.get("method") == method:
                return row
        return {}

    h1 = row_for("H1_population_mean_gradient")
    h2 = row_for("H2_support_native_ema")
    h2c = row_for("H2_random_support_control")
    h6 = row_for("H6_layerwise_target")
    h6c = row_for("H6_shuffled_layer_target_control")
    h7 = row_for("H7_cohort_surgery")
    h7c = row_for("H7_same_span_random_control")

    lines = [
        "# DG-KAN v22.49A BroaderRelatedWorkMap 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 结论",
        "",
        f"- Stage 0 完成：历史 artifact 重算 H1-H8 共 {len(stage0)} 行，未新增训练数据，所有结论只限于既有 artifact。",
        f"- Stage 1 完成：便宜 sanity tests 共 {len(stage1)} 行；覆盖 H2 support-native、H6 layerwise target、H7 cohort surgery，并包含 H1 与 matched controls。",
        f"- Stage 2 是否进入：{route.get('stage2_entered', False)}；原因：{route.get('stage2_reason', '')}",
        f"- Repair 是否执行：{bool(repair_rows)}；repair rows={len(repair_rows)}；repair 后 Stage 2 是否进入：{repair_route.get('stage2_entered_after_repair', '')}；原因：{repair_route.get('stage2_repair_reason', '')}",
        "- 本轮没有把任何 Stage 1 row 写成 official hard-task success；所有 promotion_allowed 均保持 false。",
        "",
        "## Stage 0 历史解释矩阵",
        "",
        md_table(stage0, ["hypothesis", "name", "artifact_exists", "key_rows", "support_or_candidate_rows", "control_or_gate_rows", "route", "promotion_allowed", "claim_limit"], 12),
        "",
        "## Stage 1 方法汇总",
        "",
        md_table(summary, ["hypothesis", "method", "rows", "mean_leave_cohort_gain", "positive_leave_cohort_rows", "mean_held_NLL_gain", "sanity_no_debt_rows", "mean_conflict_rate", "mean_support_projected_energy_fraction"], 20),
        "",
        "## Pairwise 证据链",
        "",
        md_table(comparisons, ["dataset", "seed", "hypothesis", "method", "control_method", "leave_cohort_gain", "control_leave_cohort_gain", "H1_leave_cohort_gain", "beats_control", "beats_H1", "beats_both", "sanity_no_debt"], 24),
        "",
        "## Repair 结果",
        "",
        md_table(repair_summary, ["hypothesis", "method", "rows", "mean_leave_cohort_gain", "positive_leave_cohort_rows", "mean_held_NLL_gain", "sanity_no_debt_rows", "mean_debt_overlap_before", "mean_debt_overlap_after", "mean_layerwise_hard_fraction", "train_debt_accept_rate", "train_debt_accept_scale_mean", "held_debt_accept_rate", "held_debt_accept_scale_mean", "held_temp_probe_feasible_rows", "mean_held_temp_probe_temperature", "mean_held_temp_probe_violation_sum", "train_temp_feasible_rows", "mean_train_temp_temperature", "mean_train_temp_violation_sum", "primal_dual_lambda_calib_final_mean", "primal_dual_lambda_brier_final_mean", "primal_dual_lambda_tail_final_mean", "primal_dual_calib_sign_match_rate", "primal_dual_brier_sign_match_rate", "primal_dual_tail_sign_match_rate", "primal_dual_guard_debt_delta_mean"], 20),
        "",
        md_table(repair_comparisons, ["dataset", "seed", "hypothesis", "method", "control_method", "leave_cohort_gain", "control_leave_cohort_gain", "H1_leave_cohort_gain", "beats_control", "beats_H1", "beats_both", "sanity_no_debt", "debt_overlap_before", "debt_overlap_after"], 24),
        "",
        "## Repair Sweep 横向复盘",
        "",
        md_table(repair_rollup, ["prefix", "method", "rows", "mean_leave_cohort_gain", "positive_leave_cohort_rows", "sanity_no_debt_rows", "beats_both", "no_debt_and_beats"], 40),
        "",
    ]
    for prefix in extra_repair_prefixes:
        extra_summary = read_rows(OUT_ROOT / f"v22_49A_{prefix}_method_summary.csv")
        extra_comparisons = read_rows(OUT_ROOT / f"v22_49A_{prefix}_pairwise_comparison.csv")
        extra_route = read_json(OUT_ROOT / f"v22_49A_{prefix}_route.json")
        lines.extend(
            [
                f"## Repair Sweep {prefix}",
                "",
                f"- route: {extra_route.get('stage2_repair_reason', '')}",
                "",
                md_table(extra_summary, ["hypothesis", "method", "rows", "mean_leave_cohort_gain", "positive_leave_cohort_rows", "mean_held_NLL_gain", "sanity_no_debt_rows", "mean_debt_overlap_before", "mean_debt_overlap_after", "train_debt_accept_rate", "train_debt_accept_scale_mean", "held_debt_accept_rate", "held_debt_accept_scale_mean", "held_temp_probe_feasible_rows", "mean_held_temp_probe_temperature", "mean_held_temp_probe_violation_sum", "train_temp_feasible_rows", "mean_train_temp_temperature", "mean_train_temp_violation_sum", "primal_dual_lambda_calib_final_mean", "primal_dual_lambda_brier_final_mean", "primal_dual_lambda_tail_final_mean", "primal_dual_calib_sign_match_rate", "primal_dual_brier_sign_match_rate", "primal_dual_tail_sign_match_rate", "primal_dual_guard_debt_delta_mean"], 20),
                "",
                md_table(extra_comparisons, ["dataset", "seed", "hypothesis", "method", "control_method", "leave_cohort_gain", "control_leave_cohort_gain", "H1_leave_cohort_gain", "beats_control", "beats_H1", "beats_both", "sanity_no_debt"], 24),
                "",
            ]
        )
    lines.extend(
        [
            "## 分析与 Insight",
            "",
            f"- H2 support-native sanity：mean leave-cohort gain={h2.get('mean_leave_cohort_gain', '')}，matched random support={h2c.get('mean_leave_cohort_gain', '')}，H1 population mean={h1.get('mean_leave_cohort_gain', '')}。该证据只能说明 cohort-EMA support projection 在本小规模设置中的相对表现，不能替代 Fisher/K-FAC official baseline。",
        f"- H6 layerwise target sanity：real={h6.get('mean_leave_cohort_gain', '')}，shuffled same-layer target control={h6c.get('mean_leave_cohort_gain', '')}。如果 real 没稳定超过 control/H1，说明当前 local target 构造还不足以支持进入 hard task。",
        h6_exact_insight,
        h6_blend_insight,
        h6_pd_insight,
        f"- H7 cohort surgery sanity：real={h7.get('mean_leave_cohort_gain', '')}，same-span random control={h7c.get('mean_leave_cohort_gain', '')}，mean conflict rate={h7.get('mean_conflict_rate', '')}。若冲突率存在但收益不超过 H1/control，则 PCGrad-like surgery 只能作为 diagnostic 或 baseline。",
        "- no-debt 解释：`v22_49A_sanity_no_debt` 是 held-train ECE/Brier/tail_q95 三项同时不增的严格 sanity gate，不是 v22.46 official no-debt 复用；失败时不得改写成成功。",
        "- Stage 0 与 Stage 1 的共同指向会由 route JSON 固化：只有 H6/H7 在 leave-cohort transfer 同时超过 H1 和 matched control，才允许按计划进入 hard task；否则停在 sanity/diagnostic。",
        f"- Repair 后路线判断：{repair_route.get('stage2_repair_reason', '未运行 repair')}。该判断保守使用 matched-control 与 H1 双比较，并要求 no-debt 不只是单点偶然通过。",
        f"- Repair 横向结论：H7 最高 mean leave-cohort gain 来自 `{best_gain_row.get('prefix', '')}/{best_gain_row.get('method', '')}`，mean={best_gain_row.get('mean_leave_cohort_gain', '')}，no-debt rows={best_gain_row.get('sanity_no_debt_rows', '')}；H7 no-debt rows 最多的是 `{best_nodebt_row.get('prefix', '')}/{best_nodebt_row.get('method', '')}`，no-debt rows={best_nodebt_row.get('sanity_no_debt_rows', '')}，mean={best_nodebt_row.get('mean_leave_cohort_gain', '')}。只要 route 仍是 not_entered，就不能把局部最强候选改写成 official hard-task success。",
        f"- primal-dual 复盘：最佳 primal-dual repair 是 `{best_primal_dual_row.get('prefix', '')}/{best_primal_dual_row.get('method', '')}`，mean leave-cohort gain={best_primal_dual_row.get('mean_leave_cohort_gain', '')}，no_debt_and_beats={best_primal_dual_row.get('no_debt_and_beats', '')}；其 train-guard sign-match rate 为 calib={best_primal_dual_summary.get('primal_dual_calib_sign_match_rate', '')}、brier={best_primal_dual_summary.get('primal_dual_brier_sign_match_rate', '')}、tail={best_primal_dual_summary.get('primal_dual_tail_sign_match_rate', '')}，guard_debt_delta_mean={best_primal_dual_summary.get('primal_dual_guard_debt_delta_mean', '')}。这说明 explicit primal-dual state 能在 train guard 上压低 proxy debt，但 held no-debt gate 仍未稳定打开。",
        h7_train_temp_insight,
        "- heldaccept/tempprobe/blend/primal-dual 的证据链：held acceptance 明显降低收益，temperature probe 不能把 no-debt 修成可行，H1/H7 blend 保留了一部分收益但 no-debt 更差；primal-dual guard8/tail05 仍是最强 H7 候选，能稳定 6/6 beat H1/control，但 no_debt_and_beats 最高停在 3/6。后续针对失败行做过多条修复：H6 exact-accept 有弱信号但 0/6 beat H1，H6 population blend / primal-dual blend 仍 0/6 beat H1；H7 calib3/tail02、lr0025+calib3/tail02、soft-ECE、soft accuracy-ECE 都保持或接近 6/6 beat，但 no_debt_and_beats 只有 2/6 或 3/6；unrestricted train-temp 在 train 上可行但选择 T<1 后 held tail 爆炸，soften-only train-temp 又退化为 T=1 且 no-debt 不改善；train exact ECE/Brier/tail acceptance 把 H7 压到 0/6 beat H1，worst-guard primal-dual 虽让 train guard debt 降得更强但 no_debt_and_beats 只有 1/6，persistent train safety guard 则把 mean gain 压到 0.00937951977054278 且 no-debt 仍为 0/6。因此当前 blocker 更像 train-only safety/debt proxy 与 held ECE/Brier/tail gate 的结构性泛化错位，而不是单纯步长、后处理校准、PCGrad 强度、guard 数量、exact train line-search、fixed train safety split、dual 聚合方式、H6 局部目标或 ECE proxy 形式不足。",
        "",
        "## 修改与修复记录",
        "",
        "- 新增 `experiments/run_v22_49A_broader_related_work_map.py`：实现 Stage 0 历史 artifact 重算、Stage 1 H2/H6/H7 并行 sanity runner、pairwise 汇总、route JSON 与两份日志生成。",
        "- blocker 修复：首轮全量 FashionMNIST 16/16 子任务失败，stderr 显示 `dgkan_core.load_vision_bundle` 不接受无连字符别名 `FashionMNIST`；已在 `load_bundle_tensors` 中把 `FashionMNIST/FMNIST` 风格输入规范化为 `Fashion-MNIST` 后重跑。",
        "- H2 实现说明：使用 train-only 多 cohort gradient matrix 的 EMA + SVD top-r support projection；matched control 使用同 rank random support，记录 support projected energy fraction。",
        "- H6 实现说明：用 train-only batch 的 hidden activation target 做 layerwise local MSE solver，并用 same-layer shuffled target 做控制；该实现是最小可证伪 sanity，不是完整 LocoProp/TargetProp。",
        "- H7 实现说明：对 train-only cohort gradients 做 PCGrad-style negative conflict projection；matched control 在同 cohort-gradient span 内随机采样同 norm 方向。",
        "- Repair 修改：新增 `H6_layerwise_cvar_target` / `H6_shuffled_cvar_target_control`，用 train-only hard-cohort/CVaR local target 测试 layerwise flow 是否能优于 H1 与 shuffled same-layer control。",
        "- Repair 修改：新增 `H6_layerwise_cvar_target_train_exact_accept` / `H6_shuffled_cvar_target_train_exact_accept_control`，先由 H6 CVaR local target 生成方向，再用独立 train guard cohorts 的 exact ECE/Brier/tail_q95 三项同时不增做 line-search acceptance；用于测试 H6 layerwise target 是否能在不使用 held/test direction 的条件下缓解 safety debt。",
        "- Repair 修改：新增 `H6_layerwise_cvar_population_blend_exact_accept` / `H6_shuffled_cvar_population_blend_exact_accept_control`，把 H1 population mean gradient 与同范数 H6 CVaR local-target 方向按 `surgery_blend_alpha` 混合，再用 train-only exact ECE/Brier/tail_q95 gate 做 line-search；用于测试 H6 是否更适合作为 H1 的残差信号而非独立替代优化器。",
        "- Repair 修改：新增 `H6_layerwise_cvar_population_blend_primal_dual` / `H6_shuffled_cvar_population_blend_primal_dual_control`，同样使用 H1+H6 residual blend，但把 safety 从 exact acceptance 改成 train-only primal-dual calib/Brier/tail constrained penalty，并记录 predicted/actual debt delta sign-match；用于测试 exact gate 损失收益后，柔性 constrained flow 是否能保留 H1 主干收益。",
        "- Repair 修改：新增 `H7_cohort_surgery_debt_orthogonal` / `H7_same_span_random_debt_orthogonal_control`，在 cohort surgery 后用 train-only Brier+tail debt gradient 做 orthogonalization，并记录 debt overlap before/after。",
        "- Repair 修改：新增 `H7_cohort_surgery_debt_barrier` / `H7_same_span_random_debt_barrier_control`，把 train-only Brier+tail debt gradient 作为 barrier 项直接加入更新方向，回应 v22.49A/H3 中 safety debt 进入更新方程的要求。",
        "- Repair 修改：新增 `H7_cohort_surgery_train_debt_accept` / `H7_same_span_random_train_debt_accept_control`，用独立 train cohorts 做 debt line-search acceptance，记录 accept rate 与 accept scale，防止把 safe no-op 写成成功。",
        "- Repair 修改：新增 `H7_cohort_surgery_train_exact_debt_accept` / `H7_same_span_random_train_exact_debt_accept_control`，用独立 train guard cohorts 的 exact ECE/Brier/tail_q95 三项同时不增做 line-search acceptance；该修复用于验证 held no-debt blocker 是否能被更忠实的 train-only safety gate 缓解。",
        "- Repair 修改：新增 `H7_cohort_surgery_persistent_guard_accept` / `H7_same_span_random_persistent_guard_accept_control`，把 train split 固定拆成 task pool 与 persistent safety guard；方向只由 task pool cohort 构造，line-search 在固定 guard 的 exact ECE/Brier/tail_q95 上执行，用于测试随机 train guard 噪声是否导致 held no-debt 泛化失败。",
        "- Repair 修改：新增 `H7_cohort_surgery_held_debt_accept` / `H7_same_span_random_held_debt_accept_control`，用独立 held-train cohorts 的 ECE/Brier/tail_q95 三项同时不增作为 line-search acceptance gate，并记录 held accept rate 与 accept scale；方向仍只由 train cohort gradients 构造。",
        "- Repair 诊断修改：新增 `H7_cohort_surgery_debt_orthogonal_held_temp_probe` / `H7_same_span_random_debt_orthogonal_held_temp_probe_control`，在 debt-orthogonal 更新后只对评估 logits 做 held-split 温度搜索，记录 feasible、temperature 和 violation；该方法使用 held split 选择温度，只能判断 calibration/tail debt 是否可后处理缓解，不能作为 official Stage 2 promotion。",
        "- Repair 修改：新增 `H7_cohort_surgery_blend_debt_orthogonal` / `H7_same_span_random_blend_debt_orthogonal_control`，把 H7 cohort surgery 作为 H1 population gradient 的残差修正 `direction=(1-alpha)*H1+alpha*H7`，再做 train-only debt orthogonalization；用于测试 PCGrad 完全替代 H1 是否过激。",
        "- Repair 修改：新增 `H7_cohort_surgery_primal_dual_safety` / `H7_same_span_random_primal_dual_safety_control`，在每个 run 内维护 train-only `lambda_calib/lambda_brier/lambda_tail` dual state；`calib=mean((max_prob-p_true)^2)` 只是可微置信度校准 proxy，不是真实 ECE 梯度；更新前记录 predicted debt delta，更新后在独立 train guard cohorts 上记录 actual debt delta 和 sign-match rate，用于检验 H3 safety-native constrained flow 是否比后验 filter 更合理。",
        "- Repair 修改：新增 `H7_cohort_surgery_primal_dual_train_temp` / `H7_same_span_random_primal_dual_train_temp_control`，在 H7 primal-dual 更新后只用 train calibration subset 选择 logits temperature，选择准则是 train ECE/Brier/tail_q95 相对更新前不增并优先 NLL gain；该方法不同于 held-temp probe，不使用 held/test 方向，用于测试输出温度后处理是否能缓解 held no-debt blocker。",
        "- Repair 修改：新增 `H7_cohort_surgery_primal_dual_train_soften_temp` / `H7_same_span_random_primal_dual_train_soften_temp_control`，在 train-only temperature calibration 中只允许 `T>=1.0` 的 softening candidates；该修复回应 unrestricted train-temp 选择 `T<1` 后在 held tail_q95 上严重过拟合的问题。",
        "- Repair 修改：新增 `H7_cohort_surgery_primal_dual_soft_ece` / `H7_same_span_random_primal_dual_soft_ece_control`，把 primal-dual calibration 分量替换为 train-only soft-binned ECE proxy；用于测试 guard8 失败行中的 held ECE 增长是否来自原校准 proxy 过粗。",
        "- Repair 修改：新增 `H7_cohort_surgery_primal_dual_soft_acc_ece` / `H7_same_span_random_primal_dual_soft_acc_ece_control`，把 calibration proxy 改成 soft-binned accuracy-ECE：bin accuracy 使用 train labels 上 detached correctness，梯度只经 confidence 与 soft bin weights；用于更贴近 held ECE gate，同时保持 train-only。",
        "- Repair 修改：新增 `H7_cohort_surgery_primal_dual_worst_guard` / `H7_same_span_random_primal_dual_worst_guard_control`，把 primal-dual 的 train guard 聚合从 mean 改成 max-over-guard-cohorts；用于测试 mean guard debt 下降是否掩盖了局部 bad cohort，从而导致 held no-debt 泛化失败。",
        "- Repair 修改：新增 `--primal-dual-tail-fraction`，使 primal-dual tail 分量可以从默认 top-25% train loss 调成更贴近 held `tail_q95` gate 的 top-5% train loss；用于测试 MNIST seed1 等失败行是否主要来自 tail proxy 粒度过粗。",
        "- 未使用 fake data；`dgkan_core.load_vision_bundle(..., download=False, allow_fake_data=False)` 从本地缓存读取 MNIST/FashionMNIST/KMNIST。",
        "",
        "## 原始产物",
        "",
        "- `results/v22_49A/v22_49A_stage0_historical_explanation_matrix.csv`",
        "- `results/v22_49A/v22_49A_stage1_sanity_matrix.csv`",
        "- `results/v22_49A/v22_49A_stage1_method_summary.csv`",
        "- `results/v22_49A/v22_49A_stage1_pairwise_comparison.csv`",
        "- `results/v22_49A/v22_49A_stage1_route.json`",
        "- `results/v22_49A/v22_49A_repair_sanity_matrix.csv`",
        "- `results/v22_49A/v22_49A_repair_method_summary.csv`",
        "- `results/v22_49A/v22_49A_repair_pairwise_comparison.csv`",
        "- `results/v22_49A/v22_49A_repair_route.json`",
        ]
    )
    for prefix in extra_repair_prefixes:
        lines.extend(
            [
                f"- `results/v22_49A/v22_49A_{prefix}_sanity_matrix.csv`",
                f"- `results/v22_49A/v22_49A_{prefix}_method_summary.csv`",
                f"- `results/v22_49A/v22_49A_{prefix}_pairwise_comparison.csv`",
                f"- `results/v22_49A/v22_49A_{prefix}_route.json`",
            ]
        )
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_exec(
        "write_recap",
        task_id="write_recap",
        status="pass",
        gpu="cpu",
        files="docs/DG-KAN_v22.49A_BroaderRelatedWorkMap_实验结果复盘.md",
        note=f"stage0_rows={len(stage0)}; stage1_rows={len(stage1)}; comparisons={len(comparisons)}",
    )


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join(shlex.quote(x) for x in sys.argv),
        task_id="v22_49A_all",
        status="started",
        gpu=str(args.gpus),
        files="results/v22_49A, docs/DG-KAN_v22.49A_BroaderRelatedWorkMap_执行日志.md, docs/DG-KAN_v22.49A_BroaderRelatedWorkMap_实验结果复盘.md",
        note=f"plan={PLAN_DOC.relative_to(ROOT)}",
    )
    truth = run_code_truth_gate()
    if truth.get("status") != "pass":
        raise RuntimeError("code truth gate failed")
    stage0_historical_reslice()
    dispatch = dispatch_stage1(args)
    if dispatch.get("failures", 0):
        raise RuntimeError(f"stage1 dispatch failures={dispatch.get('failures')}")
    route = summarize_stage1()
    write_recap()
    append_exec(
        " ".join(shlex.quote(x) for x in sys.argv),
        task_id="v22_49A_all",
        status="completed",
        gpu=str(args.gpus),
        exit_code=0,
        files="results/v22_49A/v22_49A_stage1_route.json, docs/DG-KAN_v22.49A_BroaderRelatedWorkMap_实验结果复盘.md",
        note=str(route.get("stage2_reason", "")),
    )
    return {"status": "pass", "route": route}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["all", "code-truth", "stage0", "stage1", "repair", "collect", "summarize"])
    parser.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--matrix-prefix", default="")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="H2_support_native_ema")
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--held-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--pretrain-steps", type=int, default=80)
    parser.add_argument("--branch-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--cohorts", type=int, default=4)
    parser.add_argument("--cohort-size", type=int, default=32)
    parser.add_argument("--support-rank", type=int, default=3)
    parser.add_argument("--support-ema-beta", type=float, default=0.85)
    parser.add_argument("--pretrain-lr", type=float, default=1.0e-3)
    parser.add_argument("--branch-lr", type=float, default=5.0e-3)
    parser.add_argument("--layerwise-lr", type=float, default=2.0e-2)
    parser.add_argument("--layerwise-target-eta", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hard-fraction", type=float, default=0.50)
    parser.add_argument("--debt-barrier-weight", type=float, default=0.50)
    parser.add_argument("--accept-trials", type=int, default=4)
    parser.add_argument("--accept-tolerance", type=float, default=0.0)
    parser.add_argument("--persistent-guard-fraction", type=float, default=0.25)
    parser.add_argument("--surgery-blend-alpha", type=float, default=0.50)
    parser.add_argument("--primal-dual-dual-lr", type=float, default=5.0)
    parser.add_argument("--primal-dual-base-weight", type=float, default=0.25)
    parser.add_argument("--primal-dual-calib-weight", type=float, default=1.0)
    parser.add_argument("--primal-dual-tail-weight", type=float, default=0.05)
    parser.add_argument("--primal-dual-tail-fraction", type=float, default=0.25)
    parser.add_argument("--primal-dual-tolerance", type=float, default=0.0)
    parser.add_argument("--primal-dual-max-lambda", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage == "repair" and args.methods == ",".join(METHODS):
        args.methods = ",".join(REPAIR_METHODS)
    if not args.label:
        args.label = f"v22_49A_{args.stage}_{args.method}_{args.dataset}_s{args.seed}"
    try:
        if args.stage == "all":
            run_all(args)
        elif args.stage == "code-truth":
            run_code_truth_gate()
        elif args.stage == "stage0":
            stage0_historical_reslice()
        elif args.stage == "stage1":
            dispatch_stage1(args)
            summarize_stage1()
            write_recap()
        elif args.stage == "repair":
            prefix = args.matrix_prefix.strip() or "repair"
            dispatch = dispatch_matrix(args, prefix=prefix, task_name=prefix)
            if dispatch.get("failures", 0):
                raise RuntimeError(f"repair dispatch failures={dispatch.get('failures')}")
            summarize_repair(prefix)
            write_recap()
        elif args.stage == "collect":
            run_collect(args)
        elif args.stage == "summarize":
            summarize_stage1()
            summarize_repair()
            write_recap()
        else:
            raise ValueError(args.stage)
    except Exception:
        ensure_out()
        err_path = LOG_ROOT / f"{safe_fragment(args.label)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            " ".join(shlex.quote(x) for x in sys.argv),
            task_id=safe_fragment(args.label),
            status="fail",
            gpu=str(args.device),
            exit_code=1,
            files=str(err_path.relative_to(ROOT)),
            note="exception recorded; see log",
        )
        raise


if __name__ == "__main__":
    main()
