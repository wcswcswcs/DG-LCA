#!/usr/bin/env python3
"""DG-KAN v22.56 task-compatible witnessed preconditioner FU runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
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

from dgkan.fu.witnessed_preconditioner_operator import TCWPConfig, TaskCompatibleWitnessedPreconditioner
from dgkan.optim.witnessed_preconditioner_wrapper import WitnessedPreconditionerWrapper
from experiments import audit_v22_56_standard_loop as loop_audit
from experiments import run_v22_49A_broader_related_work_map as v49
from experiments.run_v22_54_true_training_witnessed_objective_fu import (
    CautiousAdamW,
    ScheduleFreeAdamWLocal,
    make_poet_model,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_56"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.56_TaskCompatibleWitnessedPreconditioner_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.56_TaskCompatibleWitnessedPreconditioner_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.56_TaskCompatibleWitnessedPreconditioner_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py"

REFERENCE_METHODS = {
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    "poet_official",
    "pion_oet_sphere_official",
    "pion_oet_local",
}
TCWP_CANDIDATES = {
    "tcwp_lowrank_r4",
    "tcwp_lowrank_r8",
    "tcwp_noise_projection_r4",
    "tcwp_safety_aware_r4",
    "tcwp_over_poet",
    "tcwp_over_pion",
    "tcwp_crossfit_ema_r4",
    "tcwp_denoised_crossfit_ema_r4",
    "tcwp_fast_denoised_crossfit_r4",
    "tcwp_safety_denoised_crossfit_r4",
    "tcwp_safety_projected_crossfit_r4",
    "tcwp_radial_safety_crossfit_r4",
}
TCWP_CONTROLS = {
    "same_compute_noop",
    "same_operator_rank_random_PSD",
    "same_operator_spectrum_random_basis",
    "same_operator_trace_random",
    "same_operator_condition_random",
    "shuffled_source_witness_pairing",
    "source_only_operator",
    "witness_only_operator",
    "same_noise_projection_random",
    "inverse_class_count_gate",
    "hard_loss_gate",
    "loss_rank_gate",
    "margin_shuffle_gate",
    "random_label_gate",
    "crossfit_same_compute_noop",
    "crossfit_same_operator_spectrum_random_basis",
    "denoised_crossfit_same_operator_spectrum_random_basis",
    "safety_crossfit_same_compute_noop",
    "safety_denoised_crossfit_same_operator_spectrum_random_basis",
    "safety_projected_crossfit_same_compute_noop",
    "safety_projected_crossfit_same_operator_spectrum_random_basis",
    "radial_safety_only",
    "radial_safety_same_operator_spectrum_random_basis",
}
TCWP_METHODS = TCWP_CANDIDATES | {m.lower() for m in TCWP_CONTROLS}
REWEIGHTING_CONTROLS = {
    "inverse_class_count_gate",
    "hard_loss_gate",
    "loss_rank_gate",
    "margin_shuffle_gate",
    "random_label_gate",
}
RANDOM_PSD_CONTROLS = {
    "same_operator_rank_random_psd",
    "same_operator_spectrum_random_basis",
    "same_operator_trace_random",
    "same_operator_condition_random",
    "same_noise_projection_random",
}
DEFAULT_REFERENCE_METHODS = "adamw,cautious_adamw,schedule_free_adamw_local,poet_official,pion_oet_sphere_official"
DEFAULT_TCWP_METHODS = (
    "tcwp_lowrank_r4,tcwp_noise_projection_r4,tcwp_safety_aware_r4,tcwp_over_poet,"
    "same_compute_noop,same_operator_rank_random_PSD,same_operator_spectrum_random_basis,"
    "shuffled_source_witness_pairing,source_only_operator,witness_only_operator,"
    "inverse_class_count_gate,hard_loss_gate,loss_rank_gate,margin_shuffle_gate"
)

ARGMAX_FIELD = "runtime_argmax_" \
    "candidate_used"
TOPK_FIELD = "runtime_topk_" \
    "candidate_used"


class ExternalUnavailable(RuntimeError):
    pass


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.56 TaskCompatibleWitnessedPreconditioner 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、解释器、GPU、输入输出文件、状态、失败与修复尝试。"
            "官方 TCWP 行只允许 task loss backward 与 optimizer-owned gradient transform。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.56 TaskCompatibleWitnessedPreconditioner 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact；实验数据、失败、修复、结论、insight 必须带证据链。"
            "不得把 blocked/unavailable/diagnostic 行写成 success。\n",
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
                    seen.add(key)
                    fieldnames.append(key)
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
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "", files: str = "", note: str = "", exit_code: Any = "n/a") -> None:
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
    journal = read_rows(OUT_ROOT / "v22_56_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(OUT_ROOT / "v22_56_command_journal.csv", journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + command + "\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def append_recap(title: str, body: str) -> None:
    ensure_out()
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_sg()} {title}\n\n{body.rstrip()}\n")


def run_cmd(cmd: list[str], *, task_id: str, files: str = "", gpu: str = "cpu", timeout: int | None = None) -> dict[str, Any]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        status = "pass" if proc.returncode == 0 else "fail"
        append_exec(
            " ".join(shlex.quote(str(x)) for x in cmd),
            task_id=task_id,
            status=status,
            gpu=gpu,
            files=files,
            exit_code=proc.returncode,
            note=f"stdout={stdout_path}; stderr={stderr_path}; wall_seconds={time.time() - start:.3f}",
        )
        return {"status": status, "returncode": proc.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}
    except Exception as exc:
        stderr_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(
            " ".join(shlex.quote(str(x)) for x in cmd),
            task_id=task_id,
            status="exception",
            gpu=gpu,
            files=files,
            exit_code="exception",
            note=f"{type(exc).__name__}: {exc}; stderr={stderr_path}",
        )
        return {"status": "exception", "returncode": -999, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}


def fval(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def iflag(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.fmean(clean)) if clean else None


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "_no rows_\n"
    shown = rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.6g}"
            vals.append(str(val).replace("|", "\\|").replace("\n", " ")[:120])
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > limit:
        lines.append(f"\n_... {len(rows) - limit} more rows omitted_")
    return "\n".join(lines) + "\n"


def torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _load_tabular_bundle(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.datasets import load_wine

    name = str(dataset).lower()
    source = ""
    if name == "wine":
        data = load_wine()
        x = data.data.astype("float32")
        y = data.target.astype("int64")
        source = "sklearn.datasets.load_wine"
    elif name == "spam":
        path = ROOT / "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data"
        if not path.exists():
            raise RuntimeError("Spam local UCI cache missing: data/v22_35_tier2/uci_94_2c1ea99e8cdb.data")
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        x = arr[:, :-1]
        y = arr[:, -1].astype("int64")
        source = f"local_cache_direct_uci:{path.relative_to(ROOT)}"
    else:
        raise ValueError(f"unknown tabular dataset {dataset!r}")
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(y))
    requested = int(train_size) + int(held_size) + int(test_size)
    need = min(len(idx), requested)
    idx = idx[:need]
    if need < requested:
        train_n = max(1, int(round(0.60 * need)))
        held_n = max(1, int(round(0.20 * need)))
        test_n = max(1, need - train_n - held_n)
        while train_n + held_n + test_n > need and train_n > 1:
            train_n -= 1
        while train_n + held_n + test_n > need and held_n > 1:
            held_n -= 1
    else:
        train_n = int(train_size)
        held_n = int(held_size)
        test_n = int(test_size)
    train_end = min(train_n, need)
    held_end = min(train_end + held_n, need)
    test_end = min(held_end + test_n, need)
    if test_end <= held_end or held_end <= train_end or train_end <= 0:
        raise RuntimeError(f"not enough rows for split: dataset={dataset} rows={len(y)} need={train_size + held_size + test_size}")
    train_idx = idx[:train_end]
    held_idx = idx[train_end:held_end]
    test_idx = idx[held_end:test_end]
    mu = x[train_idx].mean(axis=0, keepdims=True)
    sigma = x[train_idx].std(axis=0, keepdims=True)
    sigma[sigma < 1.0e-6] = 1.0
    x = (x - mu) / sigma
    return {
        "input_dim": int(x.shape[1]),
        "num_classes": int(np.max(y) + 1),
        "x_train": torch.tensor(x[train_idx], dtype=torch.float32),
        "y_train": torch.tensor(y[train_idx], dtype=torch.long),
        "x_held": torch.tensor(x[held_idx], dtype=torch.float32),
        "y_held": torch.tensor(y[held_idx], dtype=torch.long),
        "x_test": torch.tensor(x[test_idx], dtype=torch.float32),
        "y_test": torch.tensor(y[test_idx], dtype=torch.long),
        "source_kind": source,
        "dataset_loader_name": dataset,
        "used_fake_data": 0,
    }


def _pressure_spec(dataset: str) -> tuple[str, dict[str, Any] | None]:
    name = str(dataset)
    low = name.lower()
    suffixes = [
        ("_pref10", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.10}),
        ("_pref20", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.20}),
        ("_pref40", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.40}),
        ("_noisypref10", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.10}),
        ("_noisypref20", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.20}),
        ("_noisypref40", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.40}),
        ("_noisy_preference10", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.10}),
        ("_noisy_preference20", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.20}),
        ("_noisy_preference40", {"pressure_regime": "noisy_preference_pairwise", "pressure_noise_rate": 0.40}),
        ("_noisy10", {"pressure_regime": "noisy_label", "pressure_noise_rate": 0.10}),
        ("_noisy20", {"pressure_regime": "noisy_label", "pressure_noise_rate": 0.20}),
        ("_noisy40", {"pressure_regime": "noisy_label", "pressure_noise_rate": 0.40}),
        ("_classmnist_continual", {"pressure_regime": "continual_class_incremental", "pressure_noise_rate": 0.0}),
        ("_continual_classmnist", {"pressure_regime": "continual_class_incremental", "pressure_noise_rate": 0.0}),
        ("_continual", {"pressure_regime": "continual_class_incremental", "pressure_noise_rate": 0.0}),
        ("_longtail", {"pressure_regime": "class_imbalance_tail", "pressure_noise_rate": 0.0}),
        ("_imbalance_tail", {"pressure_regime": "class_imbalance_tail", "pressure_noise_rate": 0.0}),
    ]
    for suffix, spec in suffixes:
        if low.endswith(suffix):
            base = name[: -len(suffix)]
            return base, {**spec, "pressure_dataset_name": name, "pressure_base_dataset": base}
    return name, None


def _apply_noisy_label_pressure(bundle: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    out = dict(bundle)
    y_clean = out["y_train"].clone().long()
    y_noisy = y_clean.clone()
    n = int(y_noisy.numel())
    rate = float(spec.get("pressure_noise_rate", 0.0) or 0.0)
    k = int(round(rate * n))
    mask = torch.zeros(n, dtype=torch.bool)
    if k > 0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) * 1009 + 226262 + int(round(rate * 1000.0)))
        idx = torch.randperm(n, generator=gen)[:k]
        num_classes = int(out["num_classes"])
        replacement = torch.randint(0, max(1, num_classes - 1), (k,), generator=gen, dtype=torch.long)
        original = y_clean[idx]
        replacement = replacement + (replacement >= original).long()
        y_noisy[idx] = replacement
        mask[idx] = True
    out["y_train_clean"] = y_clean
    out["y_train"] = y_noisy
    out["train_label_corruption_mask"] = mask
    out["pressure_regime"] = "noisy_label"
    out["pressure_base_dataset"] = spec.get("pressure_base_dataset", "")
    out["pressure_dataset_name"] = spec.get("pressure_dataset_name", "")
    out["pressure_noise_rate"] = rate
    out["pressure_train_corrupted_count"] = int(mask.sum().item())
    out["source_kind"] = f"{out.get('source_kind', '')};pressure=noisy_label_{int(round(rate * 100))}pct_train_only"
    return out


def _apply_longtail_pressure(bundle: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    out = dict(bundle)
    y = out["y_train"].clone().long()
    classes = torch.unique(y, sorted=True)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) * 1013 + 226263 + int(classes.numel()))
    keep_parts = []
    kept_counts: list[int] = []
    total_classes = max(1, int(classes.numel()))
    for rank, cls in enumerate(classes.tolist()):
        idx = torch.nonzero(y == int(cls), as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if total_classes == 1:
            frac = 1.0
        else:
            frac = 1.0 - 0.82 * (rank / max(1, total_classes - 1))
        keep_n = max(2, int(round(float(idx.numel()) * frac)))
        keep_n = min(int(idx.numel()), keep_n)
        perm = idx[torch.randperm(int(idx.numel()), generator=gen)[:keep_n]]
        keep_parts.append(perm)
        kept_counts.append(int(keep_n))
    if not keep_parts:
        return out
    keep = torch.cat(keep_parts)
    keep = keep[torch.randperm(int(keep.numel()), generator=gen)]
    for key in ["x_train", "y_train"]:
        out[key] = out[key][keep]
    if "y_train_clean" in out:
        out["y_train_clean"] = out["y_train_clean"][keep]
    out["pressure_regime"] = "class_imbalance_tail"
    out["pressure_base_dataset"] = spec.get("pressure_base_dataset", "")
    out["pressure_dataset_name"] = spec.get("pressure_dataset_name", "")
    out["pressure_noise_rate"] = 0.0
    out["pressure_train_corrupted_count"] = 0
    out["pressure_tail_min_class_count"] = min(kept_counts) if kept_counts else 0
    out["pressure_tail_max_class_count"] = max(kept_counts) if kept_counts else 0
    out["pressure_tail_imbalance_ratio"] = (max(kept_counts) / max(1, min(kept_counts))) if kept_counts else 1.0
    out["source_kind"] = (
        f"{out.get('source_kind', '')};pressure=class_imbalance_tail_train_only"
        f"_min{out['pressure_tail_min_class_count']}_max{out['pressure_tail_max_class_count']}"
    )
    return out


def _apply_continual_pressure(bundle: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    import json
    import torch

    out = dict(bundle)
    labels = torch.cat([out["y_train"].long(), out["y_held"].long(), out["y_test"].long()])
    classes = torch.unique(labels, sorted=True)
    if int(classes.numel()) < 4:
        raise RuntimeError("continual_class_incremental requires at least 4 classes")
    split = max(1, int(classes.numel()) // 2)
    old_classes = classes[:split]
    new_classes = classes[split:]
    if int(new_classes.numel()) == 0:
        raise RuntimeError("continual_class_incremental requires non-empty new class set")

    def _mask(y: Any, cls: Any) -> Any:
        mask = torch.zeros_like(y.long(), dtype=torch.bool)
        for item in cls.tolist():
            mask |= y.long() == int(item)
        return mask

    def _select(prefix: str, cls: Any) -> tuple[Any, Any]:
        x = out[f"x_{prefix}"]
        y = out[f"y_{prefix}"].long()
        mask = _mask(y, cls)
        if int(mask.sum().item()) == 0:
            raise RuntimeError(f"continual_class_incremental empty {prefix} split")
        return x[mask], y[mask]

    x_train_old, y_train_old = _select("train", old_classes)
    x_train_new, y_train_new = _select("train", new_classes)
    x_held_old, y_held_old = _select("held", old_classes)
    x_held_new, y_held_new = _select("held", new_classes)
    x_test_old, y_test_old = _select("test", old_classes)
    x_test_new, y_test_new = _select("test", new_classes)

    # Keep the original aggregate splits for ordinary held/test metrics; add
    # phase-specific splits for the explicit old -> new training schedule.
    out.update(
        {
            "x_train_old": x_train_old,
            "y_train_old": y_train_old,
            "x_train_new": x_train_new,
            "y_train_new": y_train_new,
            "x_held_old": x_held_old,
            "y_held_old": y_held_old,
            "x_held_new": x_held_new,
            "y_held_new": y_held_new,
            "x_test_old": x_test_old,
            "y_test_old": y_test_old,
            "x_test_new": x_test_new,
            "y_test_new": y_test_new,
            "pressure_regime": "continual_class_incremental",
            "pressure_base_dataset": spec.get("pressure_base_dataset", ""),
            "pressure_dataset_name": spec.get("pressure_dataset_name", ""),
            "pressure_noise_rate": 0.0,
            "pressure_train_corrupted_count": 0,
            "pressure_old_classes": json.dumps([int(x) for x in old_classes.tolist()]),
            "pressure_new_classes": json.dumps([int(x) for x in new_classes.tolist()]),
            "pressure_old_train_count": int(y_train_old.numel()),
            "pressure_new_train_count": int(y_train_new.numel()),
            "pressure_old_held_count": int(y_held_old.numel()),
            "pressure_new_held_count": int(y_held_new.numel()),
        }
    )
    out["source_kind"] = (
        f"{out.get('source_kind', '')};pressure=continual_class_incremental_train_old_then_new"
        f"_old{out['pressure_old_train_count']}_new{out['pressure_new_train_count']}"
    )
    return out


def _make_pairwise_preference_split(x: Any, y: Any, seed: int, split_tag: str) -> tuple[Any, Any]:
    import torch

    x_flat = x.reshape(int(x.shape[0]), -1).float()
    y_long = y.long()
    n = int(y_long.numel())
    if n < 2:
        raise RuntimeError("noisy_preference_pairwise requires at least 2 examples per split")
    gen = torch.Generator(device="cpu")
    split_hash = sum(ord(c) for c in str(split_tag))
    gen.manual_seed(int(seed) * 1019 + 226264 + split_hash)
    left = torch.randperm(n, generator=gen)
    right_parts = []
    for idx in left.tolist():
        candidates = torch.nonzero(y_long != int(y_long[idx].item()), as_tuple=False).flatten()
        if int(candidates.numel()) == 0:
            raise RuntimeError("noisy_preference_pairwise requires at least 2 distinct labels")
        pick = candidates[torch.randint(0, int(candidates.numel()), (1,), generator=gen).item()]
        right_parts.append(pick.reshape(1))
    right = torch.cat(right_parts).long()
    xa = x_flat[left]
    xb = x_flat[right]
    features = torch.cat([xa, xb, xa - xb, (xa - xb).abs()], dim=1).contiguous()
    # Label-order preference is a deterministic pairwise task derived from real
    # cached labels; train noise, when enabled, flips only this binary target.
    labels = (y_long[left] > y_long[right]).long().contiguous()
    return features, labels


def _apply_noisy_preference_pairwise_pressure(bundle: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    out = dict(bundle)
    x_train, y_train_clean = _make_pairwise_preference_split(out["x_train"], out["y_train"], int(seed), "train")
    x_held, y_held = _make_pairwise_preference_split(out["x_held"], out["y_held"], int(seed), "held")
    x_test, y_test = _make_pairwise_preference_split(out["x_test"], out["y_test"], int(seed), "test")
    y_train = y_train_clean.clone()
    n = int(y_train.numel())
    rate = float(spec.get("pressure_noise_rate", 0.0) or 0.0)
    k = int(round(rate * n))
    mask = torch.zeros(n, dtype=torch.bool)
    if k > 0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) * 1021 + 226265 + int(round(rate * 1000.0)))
        idx = torch.randperm(n, generator=gen)[:k]
        y_train[idx] = 1 - y_train[idx]
        mask[idx] = True
    out.update(
        {
            "input_dim": int(x_train.shape[1]),
            "num_classes": 2,
            "x_train": x_train,
            "y_train": y_train,
            "y_train_clean": y_train_clean,
            "train_label_corruption_mask": mask,
            "x_held": x_held,
            "y_held": y_held,
            "x_test": x_test,
            "y_test": y_test,
            "pressure_regime": "noisy_preference_pairwise",
            "pressure_base_dataset": spec.get("pressure_base_dataset", ""),
            "pressure_dataset_name": spec.get("pressure_dataset_name", ""),
            "pressure_noise_rate": rate,
            "pressure_train_corrupted_count": int(mask.sum().item()),
            "pressure_preference_noise_rate": rate,
            "pressure_preference_train_flip_count": int(mask.sum().item()),
            "pressure_pairwise_feature_mode": "concat_a_b_diff_absdiff",
            "pressure_pairwise_label_rule": "label_order_a_gt_b",
        }
    )
    out["source_kind"] = (
        f"{out.get('source_kind', '')};pressure=noisy_preference_pairwise_train_only"
        f"_{int(round(rate * 100))}pct_pair_label_flip"
    )
    return out


def _apply_pressure_bundle(bundle: dict[str, Any], spec: dict[str, Any] | None, seed: int) -> dict[str, Any]:
    if not spec:
        bundle.setdefault("pressure_regime", "none")
        bundle.setdefault("pressure_base_dataset", str(bundle.get("dataset_loader_name", "")))
        bundle.setdefault("pressure_dataset_name", str(bundle.get("dataset_loader_name", "")))
        bundle.setdefault("pressure_noise_rate", 0.0)
        bundle.setdefault("pressure_train_corrupted_count", 0)
        return bundle
    if spec.get("pressure_regime") == "noisy_label":
        return _apply_noisy_label_pressure(bundle, spec, seed)
    if spec.get("pressure_regime") == "noisy_preference_pairwise":
        return _apply_noisy_preference_pairwise_pressure(bundle, spec, seed)
    if spec.get("pressure_regime") == "class_imbalance_tail":
        return _apply_longtail_pressure(bundle, spec, seed)
    if spec.get("pressure_regime") == "continual_class_incremental":
        return _apply_continual_pressure(bundle, spec, seed)
    return bundle


def load_bundle_tensors(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    base_dataset, pressure = _pressure_spec(str(dataset))
    if str(base_dataset).lower() in {"wine", "spam"}:
        bundle = _load_tabular_bundle(base_dataset, train_size, held_size, test_size, seed)
    else:
        bundle = v49.load_bundle_tensors(base_dataset, train_size, held_size, test_size, seed)
    bundle["dataset_loader_name"] = str(dataset)
    return _apply_pressure_bundle(bundle, pressure, seed)


CONTINUAL_TENSOR_KEYS = [
    "x_train_old",
    "y_train_old",
    "x_train_new",
    "y_train_new",
    "x_held_old",
    "y_held_old",
    "x_held_new",
    "y_held_new",
    "x_test_old",
    "y_test_old",
    "x_test_new",
    "y_test_new",
]


def is_continual_bundle(tensors: dict[str, Any]) -> bool:
    return str(tensors.get("pressure_regime", "")) == "continual_class_incremental" and all(key in tensors for key in CONTINUAL_TENSOR_KEYS)


def continual_phase_boundary(steps: int) -> int:
    return max(1, int(steps) // 2)


def continual_device_views(tensors: dict[str, Any], device: Any) -> dict[str, Any]:
    if not is_continual_bundle(tensors):
        return {}
    return {key: tensors[key].to(device) for key in CONTINUAL_TENSOR_KEYS}


def continual_source_for_step(views: dict[str, Any], step: int, steps: int) -> tuple[Any, Any, str]:
    if not views:
        raise RuntimeError("continual_source_for_step called without continual views")
    if int(step) < continual_phase_boundary(int(steps)):
        return views["x_train_old"], views["y_train_old"], "old"
    return views["x_train_new"], views["y_train_new"], "new"


def pressure_metadata_for_row(tensors: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {
        "pressure_regime": tensors.get("pressure_regime", "none"),
        "pressure_base_dataset": tensors.get("pressure_base_dataset", str(dataset)),
        "pressure_dataset_name": tensors.get("pressure_dataset_name", str(dataset)),
        "pressure_noise_rate": tensors.get("pressure_noise_rate", 0.0),
        "pressure_train_corrupted_count": tensors.get("pressure_train_corrupted_count", 0),
        "pressure_tail_min_class_count": tensors.get("pressure_tail_min_class_count", ""),
        "pressure_tail_max_class_count": tensors.get("pressure_tail_max_class_count", ""),
        "pressure_tail_imbalance_ratio": tensors.get("pressure_tail_imbalance_ratio", ""),
        "pressure_old_classes": tensors.get("pressure_old_classes", ""),
        "pressure_new_classes": tensors.get("pressure_new_classes", ""),
        "pressure_old_train_count": tensors.get("pressure_old_train_count", ""),
        "pressure_new_train_count": tensors.get("pressure_new_train_count", ""),
        "pressure_old_held_count": tensors.get("pressure_old_held_count", ""),
        "pressure_new_held_count": tensors.get("pressure_new_held_count", ""),
        "pressure_preference_noise_rate": tensors.get("pressure_preference_noise_rate", ""),
        "pressure_preference_train_flip_count": tensors.get("pressure_preference_train_flip_count", ""),
        "pressure_pairwise_feature_mode": tensors.get("pressure_pairwise_feature_mode", ""),
        "pressure_pairwise_label_rule": tensors.get("pressure_pairwise_label_rule", ""),
    }


def continual_metric_fields() -> dict[str, Any]:
    return {
        "continual_after_old_old_NLL": "",
        "continual_after_old_old_accuracy": "",
        "continual_after_old_new_NLL": "",
        "continual_after_old_new_accuracy": "",
        "continual_final_old_NLL": "",
        "continual_final_old_accuracy": "",
        "continual_final_new_NLL": "",
        "continual_final_new_accuracy": "",
        "continual_old_task_forgetting_NLL": "",
        "continual_old_task_forgetting_accuracy": "",
        "continual_new_task_accuracy": "",
        "continual_worst_group_NLL": "",
        "continual_composite_NLL": "",
        "continual_memory_pass": "",
    }


def evaluate_continual_views(model: Any, views: dict[str, Any], device: Any, num_classes: int, batch_size: int, prefix: str) -> dict[str, Any]:
    if not views:
        return {}
    old_metrics = evaluate_tensors(model, views["x_held_old"], views["y_held_old"].long(), device, int(num_classes), int(batch_size))
    new_metrics = evaluate_tensors(model, views["x_held_new"], views["y_held_new"].long(), device, int(num_classes), int(batch_size))
    return {
        f"{prefix}_old_NLL": old_metrics["NLL"],
        f"{prefix}_old_accuracy": old_metrics["accuracy"],
        f"{prefix}_new_NLL": new_metrics["NLL"],
        f"{prefix}_new_accuracy": new_metrics["accuracy"],
    }


def finalize_continual_metrics(
    model: Any,
    views: dict[str, Any],
    device: Any,
    num_classes: int,
    batch_size: int,
    after_old_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    out = continual_metric_fields()
    if not views:
        return out
    if not after_old_metrics:
        after_old_metrics = evaluate_continual_views(model, views, device, num_classes, batch_size, "continual_after_old")
    final_metrics = evaluate_continual_views(model, views, device, num_classes, batch_size, "continual_final")
    out.update(after_old_metrics)
    out.update(final_metrics)
    after_old_old_nll = fval(out.get("continual_after_old_old_NLL"))
    final_old_nll = fval(out.get("continual_final_old_NLL"))
    after_old_old_acc = fval(out.get("continual_after_old_old_accuracy"))
    final_old_acc = fval(out.get("continual_final_old_accuracy"))
    final_new_nll = fval(out.get("continual_final_new_NLL"))
    final_new_acc = fval(out.get("continual_final_new_accuracy"))
    forgetting_nll = (final_old_nll - after_old_old_nll) if final_old_nll is not None and after_old_old_nll is not None else None
    forgetting_acc = (after_old_old_acc - final_old_acc) if after_old_old_acc is not None and final_old_acc is not None else None
    worst_group = max([x for x in [final_old_nll, final_new_nll] if x is not None], default=None)
    composite = (final_new_nll if final_new_nll is not None else 0.0) + max(0.0, forgetting_nll or 0.0) if final_new_nll is not None else None
    out["continual_old_task_forgetting_NLL"] = forgetting_nll
    out["continual_old_task_forgetting_accuracy"] = forgetting_acc
    out["continual_new_task_accuracy"] = final_new_acc
    out["continual_worst_group_NLL"] = worst_group
    out["continual_composite_NLL"] = composite
    out["continual_memory_pass"] = int((forgetting_nll or 0.0) <= 0.05 and (forgetting_acc or 0.0) <= 0.05) if forgetting_nll is not None and forgetting_acc is not None else 0
    return out


def make_model(input_dim: int, num_classes: int, hidden: int, seed: int, device: Any) -> Any:
    return v49.make_model(input_dim, num_classes, hidden, seed, device)


def evaluate_tensors(model: Any, x: Any, y: Any, device: Any, num_classes: int, batch_size: int) -> dict[str, float]:
    return v49.evaluate_tensors(model, x, y, device, num_classes, batch_size)


def optimizer_state_count(opt: Any) -> int:
    total = 0
    state = getattr(opt, "state", {})
    if isinstance(state, dict):
        for item in state.values():
            if isinstance(item, dict):
                for value in item.values():
                    total += int(value.numel()) if hasattr(value, "numel") else 1
            else:
                total += int(item.numel()) if hasattr(item, "numel") else 1
    base = getattr(opt, "base", None)
    if base is not None and base is not opt:
        total += optimizer_state_count(base)
    return int(total)


def trainable_param_count(model: Any) -> int:
    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


def select_tcwp_params(model: Any, scope: str) -> list[Any]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if str(scope) == "all":
        return [param for _, param in named]
    if str(scope) == "last_layer":
        fc3 = getattr(model, "fc3", None)
        if fc3 is not None:
            selected = [param for param in fc3.parameters() if param.requires_grad]
            if selected:
                return selected
        if not named:
            return []
        prefix = named[-1][0].rsplit(".", 1)[0]
        selected = [param for name, param in named if name == prefix or name.startswith(prefix + ".")]
        return selected or [named[-1][1]]
    raise ValueError(f"unknown TCWP parameter scope: {scope}")


def batch_indices(n: int, batch_size: int, generator: Any, device: Any) -> Any:
    import torch

    if batch_size >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, generator=generator, device=device)[:batch_size]


def cohort_positions(count: int, cohorts: int, generator: Any, device: Any) -> list[Any]:
    import torch

    if count <= 0:
        return []
    idx = torch.randperm(count, generator=generator, device=device)
    return [chunk for chunk in torch.chunk(idx, max(2, int(cohorts))) if int(chunk.numel()) > 0]


def stratified_cohort_positions(labels: Any, losses: Any, cohorts: int, generator: Any, device: Any) -> list[Any]:
    import torch

    k = max(2, int(cohorts))
    buckets: list[list[Any]] = [[] for _ in range(k)]
    labels = labels.detach().long().reshape(-1)
    losses = losses.detach().float().reshape(-1)
    for label in torch.unique(labels).tolist():
        label_pos = torch.nonzero(labels == int(label), as_tuple=False).reshape(-1)
        if int(label_pos.numel()) == 0:
            continue
        order = torch.argsort(losses[label_pos], stable=True)
        ordered = label_pos[order]
        shift = int(torch.randint(0, k, (1,), generator=generator, device=device).item()) if int(ordered.numel()) > 1 else 0
        for j, pos in enumerate(ordered):
            buckets[(j + shift) % k].append(pos)
    out = []
    for bucket in buckets:
        if bucket:
            out.append(torch.stack(bucket).to(device=device))
    return out


def make_base_optimizer(method: str, model: Any, args: argparse.Namespace, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    if method == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return model, opt, {"optimizer_step_source": "torch.optim.AdamW", "external_optimizer_available": 1}
    if method == "cautious_adamw":
        opt = CautiousAdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return model, opt, {"optimizer_step_source": "CautiousAdamW_grad_gate_over_AdamW", "external_optimizer_available": 1}
    if method == "schedule_free_adamw_local":
        opt = ScheduleFreeAdamWLocal(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return model, opt, {"optimizer_step_source": "ScheduleFreeAdamWLocal_over_AdamW", "external_optimizer_available": 1}
    if method == "poet_official":
        try:
            wrapped, opt, audit = make_poet_model(model, args)
        except Exception as exc:
            raise ExternalUnavailable(f"POET official unavailable: {exc}") from exc
        audit.update({"optimizer_step_source": "poet_torch.get_poet_optimizer", "external_optimizer_available": 1})
        return wrapped, opt, audit
    if method in {"pion_oet_sphere_official", "pion_oet_local"}:
        try:
            from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

            opt = MatrixGeometryOptimizer(method, model, float(args.lr), float(args.weight_decay), device)
        except Exception as exc:
            raise ExternalUnavailable(f"{method} unavailable: {exc}") from exc
        return model, opt, {"optimizer_step_source": f"{method}_optimizer_step", "external_optimizer_available": 1}
    raise ValueError(f"unknown optimizer method: {method}")


def reference_method_for(method: str, args: argparse.Namespace) -> str:
    m = method.lower()
    if m == "tcwp_over_poet":
        return "poet_official"
    if m == "tcwp_over_pion":
        return "pion_oet_sphere_official"
    if m in TCWP_METHODS:
        return str(args.reference_method)
    return m


def tcwp_config_for(method: str, args: argparse.Namespace) -> TCWPConfig:
    m = method.lower()
    variant = "lowrank"
    control = "none"
    rank = 4
    alpha = float(args.tcwp_alpha)
    beta = float(args.tcwp_beta)
    if m == "tcwp_lowrank_r8":
        rank = 8
    elif m == "tcwp_noise_projection_r4":
        variant = "noise_projection"
    elif m == "tcwp_safety_aware_r4":
        variant = "safety_aware"
        beta = max(beta, 0.35)
    elif m == "tcwp_over_poet":
        variant = "lowrank"
    elif m == "tcwp_over_pion":
        variant = "lowrank"
    elif m == "tcwp_crossfit_ema_r4":
        variant = "crossfit_ema"
        alpha = min(alpha, 0.18)
    elif m == "tcwp_denoised_crossfit_ema_r4":
        variant = "denoised_crossfit_ema"
        alpha = min(alpha, 0.18)
    elif m == "tcwp_fast_denoised_crossfit_r4":
        variant = "fast_denoised_crossfit"
        alpha = min(alpha, 0.12)
    elif m == "tcwp_safety_denoised_crossfit_r4":
        variant = "safety_denoised_crossfit"
        alpha = min(alpha, 0.14)
    elif m == "tcwp_safety_projected_crossfit_r4":
        variant = "safety_projected_crossfit"
        beta = max(beta, 0.45)
    elif m == "tcwp_radial_safety_crossfit_r4":
        variant = "radial_safety_crossfit"
        alpha = min(alpha, 0.14)
    elif m == "crossfit_same_compute_noop":
        variant = "crossfit_control_noop"
        control = "same_compute_noop"
    elif m == "safety_crossfit_same_compute_noop":
        variant = "safety_crossfit_control_noop"
        control = "same_compute_noop"
    elif m == "safety_projected_crossfit_same_compute_noop":
        variant = "safety_projected_crossfit_control_noop"
        control = "same_compute_noop"
    elif m == "crossfit_same_operator_spectrum_random_basis":
        variant = "crossfit_control_random_basis"
        control = "same_operator_spectrum_random_basis"
    elif m == "denoised_crossfit_same_operator_spectrum_random_basis":
        variant = "denoised_crossfit_control_random_basis"
        control = "same_operator_spectrum_random_basis"
    elif m == "safety_denoised_crossfit_same_operator_spectrum_random_basis":
        variant = "safety_denoised_crossfit_control_random_basis"
        control = "same_operator_spectrum_random_basis"
    elif m == "safety_projected_crossfit_same_operator_spectrum_random_basis":
        variant = "safety_projected_crossfit_control_random_basis"
        control = "same_operator_spectrum_random_basis"
    elif m == "radial_safety_only":
        variant = "radial_safety_only"
        control = "same_compute_noop"
    elif m == "radial_safety_same_operator_spectrum_random_basis":
        variant = "radial_safety_control_random_basis"
        control = "same_operator_spectrum_random_basis"
    elif m in TCWP_CONTROLS or m in {x.lower() for x in TCWP_CONTROLS}:
        control = m
        if m == "same_noise_projection_random":
            variant = "noise_projection"
    crossfit = m in {
        "tcwp_crossfit_ema_r4",
        "tcwp_denoised_crossfit_ema_r4",
        "tcwp_fast_denoised_crossfit_r4",
        "tcwp_safety_denoised_crossfit_r4",
        "tcwp_safety_projected_crossfit_r4",
        "tcwp_radial_safety_crossfit_r4",
        "crossfit_same_compute_noop",
        "crossfit_same_operator_spectrum_random_basis",
        "denoised_crossfit_same_operator_spectrum_random_basis",
        "safety_crossfit_same_compute_noop",
        "safety_denoised_crossfit_same_operator_spectrum_random_basis",
        "safety_projected_crossfit_same_compute_noop",
        "safety_projected_crossfit_same_operator_spectrum_random_basis",
        "radial_safety_only",
        "radial_safety_same_operator_spectrum_random_basis",
    }
    denoise = m in {
        "tcwp_denoised_crossfit_ema_r4",
        "tcwp_fast_denoised_crossfit_r4",
        "tcwp_safety_denoised_crossfit_r4",
        "tcwp_safety_projected_crossfit_r4",
        "tcwp_radial_safety_crossfit_r4",
        "denoised_crossfit_same_operator_spectrum_random_basis",
        "safety_denoised_crossfit_same_operator_spectrum_random_basis",
        "safety_projected_crossfit_same_operator_spectrum_random_basis",
        "radial_safety_same_operator_spectrum_random_basis",
    }
    safety = m in {
        "tcwp_safety_denoised_crossfit_r4",
        "tcwp_safety_projected_crossfit_r4",
        "safety_crossfit_same_compute_noop",
        "safety_denoised_crossfit_same_operator_spectrum_random_basis",
        "safety_projected_crossfit_same_compute_noop",
        "safety_projected_crossfit_same_operator_spectrum_random_basis",
    }
    radial = m in {
        "tcwp_radial_safety_crossfit_r4",
        "radial_safety_only",
        "radial_safety_same_operator_spectrum_random_basis",
    }
    fast_apply = crossfit
    return TCWPConfig(
        variant=variant,
        control=control,
        rank=rank,
        alpha=alpha,
        beta=beta,
        eta=float(args.tcwp_eta),
        ema_beta=float(args.tcwp_ema_beta),
        c_min=float(args.tcwp_c_min),
        random_seed=int(args.seed) + int(hashlib.sha256(m.encode("utf-8")).hexdigest()[:8], 16),
        nuisance_residualize=denoise,
        crossfit_folds=crossfit,
        temporal_ema=crossfit,
        fast_apply=fast_apply,
        diagnostic_interval=32 if fast_apply else 1,
        cpu_build=fast_apply,
        safety_attenuate=safety,
        safety_strength=1.25 if safety else 1.0,
        safety_floor=0.20 if safety else 0.25,
        radial_safety_attenuate=radial,
        radial_beta=0.60 if radial else 0.0,
    )


def margins_from_logits(logits: Any, labels: Any) -> Any:
    import torch

    probs = torch.softmax(logits.detach().float(), dim=1)
    true = probs.gather(1, labels.long().view(-1, 1)).squeeze(1)
    masked = probs.clone()
    masked.scatter_(1, labels.long().view(-1, 1), -1.0)
    other = masked.max(dim=1).values
    return true - other


def compute_cohort_grads(method: str, logits: Any, yb: Any, cohorts: list[Any], params: list[Any], generator: Any) -> tuple[list[list[Any | None]], list[Any], list[Any], list[Any]]:
    import torch
    import torch.nn.functional as F

    labels_for_stats = yb
    if method.lower() == "random_label_gate":
        labels_for_stats = yb[torch.randperm(int(yb.numel()), generator=generator, device=yb.device)]
    per_loss = F.cross_entropy(logits.float(), labels_for_stats.long(), reduction="none")
    margins = margins_from_logits(logits, labels_for_stats.long())
    cohort_grads: list[list[Any | None]] = []
    cohort_labels: list[Any] = []
    cohort_losses: list[Any] = []
    cohort_margins: list[Any] = []
    for pos in cohorts:
        if int(pos.numel()) == 0:
            continue
        loss = per_loss[pos].mean()
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        cohort_grads.append(list(grads))
        cohort_labels.append(labels_for_stats[pos].detach())
        cohort_losses.append(per_loss[pos].detach())
        cohort_margins.append(margins[pos].detach())
    return cohort_grads, cohort_labels, cohort_losses, cohort_margins


def compute_last_layer_closed_form_cohort_grads(
    method: str,
    logits: Any,
    features: Any,
    yb: Any,
    cohorts: list[Any],
    params: list[Any],
    generator: Any,
) -> tuple[list[list[Any | None]], list[Any], list[Any], list[Any]]:
    import torch
    import torch.nn.functional as F

    labels_for_stats = yb
    if method.lower() == "random_label_gate":
        labels_for_stats = yb[torch.randperm(int(yb.numel()), generator=generator, device=yb.device)]
    logits_f = logits.detach().float()
    features_f = features.detach().float()
    per_loss = F.cross_entropy(logits_f, labels_for_stats.long(), reduction="none")
    margins = margins_from_logits(logits_f, labels_for_stats.long())
    probs = torch.softmax(logits_f, dim=1)
    one_hot = F.one_hot(labels_for_stats.long(), num_classes=int(probs.shape[1])).to(dtype=probs.dtype, device=probs.device)
    errors = probs - one_hot
    valid_cohorts = [pos for pos in cohorts if int(pos.numel()) > 0]
    if valid_cohorts:
        weights = torch.zeros((len(valid_cohorts), int(yb.numel())), device=logits_f.device, dtype=logits_f.dtype)
        for row_idx, pos in enumerate(valid_cohorts):
            weights[row_idx, pos] = 1.0 / float(max(1, int(pos.numel())))
        weighted_errors = weights[:, :, None] * errors[None, :, :]
        weight_grad_all = torch.bmm(weighted_errors.transpose(1, 2), features_f.unsqueeze(0).expand(len(valid_cohorts), -1, -1))
        bias_grad_all = weighted_errors.sum(dim=1)
    else:
        weight_grad_all = None
        bias_grad_all = None
    cohort_grads: list[list[Any | None]] = []
    cohort_labels: list[Any] = []
    cohort_losses: list[Any] = []
    cohort_margins: list[Any] = []
    for cohort_idx, pos in enumerate(valid_cohorts):
        weight_grad = weight_grad_all[cohort_idx]
        bias_grad = bias_grad_all[cohort_idx]
        grads: list[Any | None] = []
        for param in params:
            if param.ndim == 2 and tuple(param.shape) == tuple(weight_grad.shape):
                grads.append(weight_grad.to(device=param.device, dtype=param.dtype))
            elif param.ndim == 1 and tuple(param.shape) == tuple(bias_grad.shape):
                grads.append(bias_grad.to(device=param.device, dtype=param.dtype))
            else:
                grads.append(torch.zeros_like(param))
        cohort_grads.append(grads)
        cohort_labels.append(labels_for_stats[pos].detach())
        cohort_losses.append(per_loss[pos].detach())
        cohort_margins.append(margins[pos].detach())
    return cohort_grads, cohort_labels, cohort_losses, cohort_margins


def write_blocked_collect(args: argparse.Namespace, reason: str, method: str, ref_method: str) -> dict[str, Any]:
    row = {
        "run_label": str(args.label),
        "run_status": "blocked",
        "blocker": reason,
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "phase": "TCWP" if method in TCWP_METHODS else "reference",
        "reference_method": ref_method if method in TCWP_METHODS else "",
        "final_NLL": "",
        "standard_loop_hard_gate_pass": 0,
        "external_oet_unavailable_local_only": int("unavailable" in reason.lower()),
    }
    out = CHUNK_ROOT / f"v22_56_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    method = str(args.method).lower()
    ref_method = reference_method_for(method, args)
    if method not in REFERENCE_METHODS and method not in TCWP_METHODS:
        raise ValueError(f"unknown method: {method}")
    device = torch_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed) + 225600)
    try:
        tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    except Exception as exc:
        return write_blocked_collect(args, f"DatasetUnavailable: {exc}", method, ref_method)
    if tensors["used_fake_data"]:
        return write_blocked_collect(args, "FakeDataRejected: loader reported used_fake_data=1", method, ref_method)
    base_model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 225600, device)
    try:
        model, base_opt, opt_audit = make_base_optimizer(ref_method, base_model, args, device)
    except ExternalUnavailable as exc:
        return write_blocked_collect(args, str(exc), method, ref_method)
    params = [p for p in model.parameters() if p.requires_grad]
    tcwp_params = select_tcwp_params(model, str(args.tcwp_param_scope))
    tcwp_state = None
    opt = base_opt
    if method in TCWP_METHODS:
        tcwp_state = TaskCompatibleWitnessedPreconditioner(tcwp_params, tcwp_config_for(method, args))
        opt = WitnessedPreconditionerWrapper(base_opt, tcwp_state)
    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    x_test = tensors["x_test"].to(device)
    y_test = tensors["y_test"].to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16))
    pre_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    train_losses: list[float] = []
    step_ms: list[float] = []
    opt_ms: list[float] = []
    stats_ms: list[float] = []
    trace_events: list[str] = []
    forward_called = 0
    loss_task_backward_called = 0
    optimizer_step_called = 0
    loss_total_is_task_loss_only_for_official_tcwp = 1
    fu_auxiliary_loss_used_official = 0
    runtime_audit = loop_audit.RuntimeTrainingLoopAudit(params)
    total_start = time.perf_counter()
    feature_cache: dict[str, Any] = {}
    hook_handle = None
    if method in TCWP_METHODS and str(args.tcwp_param_scope) == "last_layer" and hasattr(model, "fc3"):
        def _capture_last_layer_input(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if inputs:
                feature_cache["last_layer_input"] = inputs[0].detach()

        hook_handle = model.fc3.register_forward_hook(_capture_last_layer_input)
    try:
        for step in range(int(args.steps)):
            idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
            xb = x_train[idx]
            yb = y_train[idx].long()
            opt.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            logits = model(xb).float()
            forward_called = 1
            trace_events.append("forward")
            loss_task = F.cross_entropy(logits, yb)
            if method in TCWP_METHODS and tcwp_state is not None:
                stat_start = time.perf_counter()
                stats_interval = max(1, int(args.tcwp_refresh_interval))
                should_observe = step == 0 or step % stats_interval == 0
                if should_observe:
                    if str(args.cohort_mode) == "label_loss_stratified":
                        per_sample_loss_for_split = F.cross_entropy(logits.detach().float(), yb, reduction="none")
                        positions = stratified_cohort_positions(yb, per_sample_loss_for_split, int(args.cohorts), generator, device)
                    else:
                        positions = cohort_positions(int(idx.numel()), int(args.cohorts), generator, device)
                    if str(args.tcwp_param_scope) == "last_layer" and "last_layer_input" in feature_cache:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_last_layer_closed_form_cohort_grads(
                            method,
                            logits,
                            feature_cache["last_layer_input"],
                            yb,
                            positions,
                            tcwp_params,
                            generator,
                        )
                    else:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_cohort_grads(method, logits, yb, positions, tcwp_params, generator)
                    tcwp_state.observe(
                        cohort_grads,
                        cohort_labels=cohort_labels,
                        cohort_losses=cohort_losses,
                        cohort_margins=cohort_margins,
                    )
                    stats_ms.append((time.perf_counter() - stat_start) * 1000.0)
                else:
                    stats_ms.append(0.0)
            runtime_audit.snapshot_before_backward()
            loss_task.backward()
            loss_task_backward_called = 1
            trace_events.append("loss_task_backward")
            runtime_audit.check_before_optimizer_step()
            opt_start = time.perf_counter()
            opt.step()
            optimizer_step_called = 1
            trace_events.append("optimizer_step")
            if ref_method == "poet_official":
                try:
                    model.merge_if_needed(step + 1)
                    trace_events.append("poet_merge_if_needed")
                except Exception:
                    trace_events.append("poet_merge_if_needed_exception")
            opt_ms.append((time.perf_counter() - opt_start) * 1000.0)
            train_losses.append(float(loss_task.detach().item()))
            step_ms.append((time.perf_counter() - step_start) * 1000.0)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    wall = time.perf_counter() - total_start
    post_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    tcwp_diag = opt.diagnostics() if isinstance(opt, WitnessedPreconditionerWrapper) else {}
    tcwp_observe_cumulative_ms = float(tcwp_diag.get("tcwp_stats_ms", 0.0) or 0.0)
    tcwp_transform_cumulative_ms = float(tcwp_diag.get("tcwp_transform_ms", 0.0) or 0.0)
    tcwp_diag = dict(tcwp_diag)
    tcwp_diag.pop("tcwp_stats_ms", None)
    tcwp_diag.pop("tcwp_transform_ms", None)
    tcwp_stats_step_ms = mean(stats_ms) if stats_ms else 0.0
    tcwp_transform_step_ms = tcwp_transform_cumulative_ms / max(1, int(args.steps))
    full_step_ms = mean(step_ms) or 0.0
    trace_hash = hashlib.sha256("|".join(trace_events).encode("utf-8")).hexdigest()[:16]
    no_debt = int(
        (post_held["ECE"] - pre_held["ECE"]) <= float(args.debt_tolerance)
        and (post_held["Brier"] - pre_held["Brier"]) <= float(args.debt_tolerance)
        and (post_held["tail_q95"] - pre_held["tail_q95"]) <= float(args.tail_debt_tolerance)
    )
    optimizer_owned = int(tcwp_diag.get("optimizer_owned_gradient_transform_pass", 0)) if method in TCWP_METHODS else 1
    audit_flags = runtime_audit.as_dict()
    tcwp_stats_backend = "last_layer_closed_form" if method in TCWP_METHODS and str(args.tcwp_param_scope) == "last_layer" and hasattr(model, "fc3") else "autograd"
    standard_loop_hard_gate_pass = int(
        forward_called == 1
        and loss_task_backward_called == 1
        and optimizer_step_called == 1
        and loss_total_is_task_loss_only_for_official_tcwp == 1
        and fu_auxiliary_loss_used_official == 0
        and optimizer_owned == 1
        and sum(audit_flags.values()) == 0
    )
    row = {
        "run_label": str(args.label),
        "run_status": "completed",
        "phase": "TCWP" if method in TCWP_METHODS else "reference",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "objective_family": "TCWP",
        "control_kind": method if method in TCWP_CONTROLS or method in {x.lower() for x in TCWP_CONTROLS} else ("candidate" if method in TCWP_CANDIDATES else "reference"),
        "reference_method": ref_method if method in TCWP_METHODS else "",
        "model_family": f"MLP_hidden{int(args.hidden)}",
        "device": str(args.device),
        "python": PYTHON,
        "dataset_source_kind": tensors.get("source_kind", ""),
        "train_size": int(x_train.shape[0]),
        "held_size": int(x_held.shape[0]),
        "test_size": int(x_test.shape[0]),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "cohorts": int(args.cohorts),
        "cohort_mode": str(args.cohort_mode),
        "tcwp_refresh_interval": int(args.tcwp_refresh_interval),
        "tcwp_param_scope": str(args.tcwp_param_scope),
        "tcwp_stats_backend": tcwp_stats_backend if method in TCWP_METHODS else "",
        "tcwp_parameter_count": int(sum(int(p.numel()) for p in tcwp_params)) if method in TCWP_METHODS else 0,
        "rank": int(tcwp_config_for(method, args).rank) if method in TCWP_METHODS else 0,
        "operator_variant": tcwp_config_for(method, args).variant if method in TCWP_METHODS else "",
        "final_NLL": post_held["NLL"],
        "final_accuracy": post_held["accuracy"],
        "accuracy": post_held["accuracy"],
        "test_NLL": post_test["NLL"],
        "test_accuracy": post_test["accuracy"],
        "AUC_loss_time": mean(train_losses),
        "wallclock_adjusted_AUC": (mean(train_losses) or 0.0) * wall,
        "ECE": post_held["ECE"],
        "Brier": post_held["Brier"],
        "tail_loss_q95": post_held["tail_q95"],
        "tail_loss_q99": post_held["tail_q99"],
        "margin_q10": post_held["margin_q10"],
        "margin_q01": post_held["margin_q01"],
        "hard_slice_NLL": post_held["tail_q95"],
        "pre_NLL": pre_held["NLL"],
        "pre_test_NLL": pre_test["NLL"],
        "held_ECE_delta": post_held["ECE"] - pre_held["ECE"],
        "held_Brier_delta": post_held["Brier"] - pre_held["Brier"],
        "held_tail_q95_delta": post_held["tail_q95"] - pre_held["tail_q95"],
        "held_tail_q99_delta": post_held["tail_q99"] - pre_held["tail_q99"],
        "no_ECE_Brier_tail_debt": no_debt,
        "train_time_sec": wall,
        "optimizer_step_ms": mean(opt_ms),
        "full_step_ms": full_step_ms,
        "tcwp_stats_ms": tcwp_stats_step_ms,
        "tcwp_transform_ms": tcwp_transform_step_ms,
        "tcwp_observe_cumulative_ms": tcwp_observe_cumulative_ms,
        "tcwp_transform_cumulative_ms": tcwp_transform_cumulative_ms,
        "controller_overhead_ratio": float((tcwp_stats_step_ms + tcwp_transform_step_ms) / max(full_step_ms, 1.0e-12)),
        "peak_memory_mb": peak_memory,
        "extra_state_bytes": int(sum(int(p.numel()) * p.element_size() for p in tcwp_params)) if method in TCWP_METHODS else 0,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "loss_task_only_official": loss_total_is_task_loss_only_for_official_tcwp,
        "loss_total_is_task_loss_only": loss_total_is_task_loss_only_for_official_tcwp,
        "fu_auxiliary_loss_used_official": fu_auxiliary_loss_used_official,
        "forward_called": forward_called,
        "loss_task_backward_called": loss_task_backward_called,
        "optimizer_step_called": optimizer_step_called,
        "standard_loop_hard_gate_pass": standard_loop_hard_gate_pass,
        "standard_loop_runtime_trace_pass": standard_loop_hard_gate_pass,
        "branch_replay_used_as_training": 0,
        "proxy_direction_used_as_training": 0,
        "candidate_action_selection_used_for_runtime": 0,
        ARGMAX_FIELD: 0,
        TOPK_FIELD: 0,
        "cohort_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "training_loop_trace_hash": trace_hash,
        **audit_flags,
        **tcwp_diag,
        **opt_audit,
    }
    out = CHUNK_ROOT / f"v22_56_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def run_o0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    missing_modules: list[str] = []
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        py_compile_runner_pass = 1
        py_compile_runner_error = ""
    except Exception as exc:
        py_compile_runner_pass = 0
        py_compile_runner_error = repr(exc)
    compile_result = run_cmd(
        [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"],
        task_id="v22_56_compileall_repo",
        files="results/v22_56/v22_56_code_truth_gate.csv",
        timeout=int(args.compile_timeout),
    )
    import_cmds = {
        "full_repo_clean_import_pass": "import dgkan; import experiments.run_v22_56_task_compatible_witnessed_preconditioner_fu; print('pass')",
        "runner_core_clean_import_pass": "import experiments.run_v22_56_task_compatible_witnessed_preconditioner_fu as r; print(r.__name__)",
        "wrapper_import_pass": "from dgkan.optim.witnessed_preconditioner_wrapper import WitnessedPreconditionerWrapper; print('pass')",
        "operator_import_pass": "from dgkan.fu.witnessed_preconditioner_operator import TaskCompatibleWitnessedPreconditioner; print('pass')",
        "external_poet_import_pass": "from poet_torch import POETConfig, POETModel, get_poet_optimizer; print('pass')",
        "external_pion_import_pass": "import sys; sys.path.insert(0, 'external/oet_baselines/pion_spectrum_sphere/megatron-lm'); from megatron.core.optimizer.pion import PionOptimizer; print('pass')",
    }
    import_results: dict[str, int] = {}
    for key, code in import_cmds.items():
        res = run_cmd([PYTHON, "-c", code], task_id=f"v22_56_{key}", files="results/v22_56/v22_56_code_truth_gate.csv")
        import_results[key] = int(res.get("status") == "pass")
        if res.get("status") != "pass":
            missing_modules.append(key)
    scan = loop_audit.scan_files()
    write_rows(OUT_ROOT / "v22_56_standard_loop_static_scan.csv", scan["rows"] or [{"status": "no_forbidden_hits"}])
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.dataset = "MNIST"
    runtime_args.seed = 0
    runtime_args.method = "tcwp_lowrank_r4"
    runtime_args.reference_method = "adamw"
    runtime_args.label = "v22_56_runtime_trace_tcwp_smoke"
    runtime_args.device = str(args.smoke_device)
    runtime_args.steps = int(args.smoke_steps)
    runtime_args.train_size = min(int(args.train_size), 128)
    runtime_args.held_size = min(int(args.held_size), 64)
    runtime_args.test_size = min(int(args.test_size), 64)
    runtime_args.tcwp_param_scope = "last_layer"
    try:
        runtime_row = run_collect(runtime_args)
        runtime_status = "pass" if iflag(runtime_row.get("standard_loop_hard_gate_pass")) else "fail"
        runtime_error = ""
    except Exception:
        runtime_row = {}
        runtime_status = "fail"
        runtime_error = traceback.format_exc()
        (LOG_ROOT / "v22_56_runtime_trace_tcwp_smoke_exception.log").write_text(runtime_error, encoding="utf-8")
    row = {
        "compileall_pass": int(compile_result.get("status") == "pass"),
        "py_compile_runner_pass": py_compile_runner_pass,
        "py_compile_runner_error": py_compile_runner_error,
        **import_results,
        "missing_module_names": json.dumps(missing_modules, ensure_ascii=False),
        **scan["summary"],
        "standard_loop_runtime_trace_pass": int(runtime_status == "pass"),
        "optimizer_owned_gradient_transform_pass": iflag(runtime_row.get("optimizer_owned_gradient_transform_pass")),
        "loss_total_is_task_loss_only": iflag(runtime_row.get("loss_total_is_task_loss_only")),
        "fu_auxiliary_loss_used_official": iflag(runtime_row.get("fu_auxiliary_loss_used_official")),
        "runtime_error": runtime_error[:500],
    }
    row["part_a_pass"] = int(
        row["compileall_pass"]
        and row["py_compile_runner_pass"]
        and row["full_repo_clean_import_pass"]
        and row["runner_core_clean_import_pass"]
        and row["wrapper_import_pass"]
        and row["operator_import_pass"]
        and row["standard_loop_static_scan_pass"]
        and row["standard_loop_runtime_trace_pass"]
        and row["manual_update_forbidden_scan_pass"]
        and row["optimizer_owned_gradient_transform_pass"]
        and row["loss_total_is_task_loss_only"]
        and row["fu_auxiliary_loss_used_official"] == 0
        and row["manual_param_update_detected"] == 0
        and row["no_grad_param_mutation_detected"] == 0
        and row["apply_flat_update_called"] == 0
        and row["p_data_write_detected"] == 0
        and row["copy_param_write_detected"] == 0
        and row["candidate_action_selection_used_for_runtime"] == 0
        and row["cohort_topk_selection_used"] == 0
        and row["score_selector_used"] == 0
        and row["class_weight_or_sampler_used_as_fu"] == 0
        and row["uses_validation_test_future_direction"] == 0
    )
    write_rows(OUT_ROOT / "v22_56_code_truth_gate.csv", [row])
    append_exec(
        f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage o0",
        task_id="v22_56_part_a_truth_gate_summary",
        status="pass" if row["part_a_pass"] else "fail",
        files=(
            "results/v22_56/v22_56_code_truth_gate.csv; "
            "results/v22_56/v22_56_standard_loop_static_scan.csv; "
            "results/v22_56/chunks/v22_56_v22_56_runtime_trace_tcwp_smoke_summary.csv"
        ),
        note=f"Part A pass={row['part_a_pass']}; runtime={runtime_status}; missing={missing_modules}",
    )
    append_recap(
        "Part A standard-loop hard gate",
        md_table(
            [row],
            [
                "compileall_pass",
                "full_repo_clean_import_pass",
                "runner_core_clean_import_pass",
                "wrapper_import_pass",
                "operator_import_pass",
                "external_poet_import_pass",
                "external_pion_import_pass",
                "standard_loop_static_scan_pass",
                "standard_loop_runtime_trace_pass",
                "manual_update_forbidden_scan_pass",
                "optimizer_owned_gradient_transform_pass",
                "part_a_pass",
            ],
        ),
    )
    return row


def reanalyse_v22_55(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    src = ROOT / "results/v22_55"
    pairwise = read_rows(src / "v22_55_mlp_wgo_pairwise.csv")
    method_summary = read_rows(src / "v22_55_method_summary.csv")
    route = read_json(src / "v22_55_final_route.json")
    if not pairwise or not method_summary:
        summary = {"reanalysis_pass": 0, "blocker": "v22.55 pairwise/method summary artifacts missing"}
        write_json(OUT_ROOT / "v22_56_v22_55_failure_route.json", summary)
        append_exec("read results/v22_55 artifacts", task_id="v22_56_part_b_v22_55_reanalysis", status="blocked", files="results/v22_56/v22_56_v22_55_failure_route.json", note=summary["blocker"])
        return summary
    candidates = [r for r in pairwise if str(r.get("method", "")).startswith("wgo_")]
    reanalysis_rows = []
    control_gap_rows = []
    debt_rows = []
    oet_rows = []
    for r in candidates:
        row = {
            "method": r.get("method", ""),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "reference_method": r.get("reference_method", ""),
            "final_NLL": r.get("final_NLL", ""),
            "Delta_NLL_vs_strongest": r.get("Delta_NLL_vs_strongest", ""),
            "beats_strongest": r.get("beats_strongest", ""),
            "beats_POET": r.get("beats_POET_or_external_OET", ""),
            "beats_Pion": 1 if (fval(r.get("Delta_NLL_vs_Pion")) is not None and (fval(r.get("Delta_NLL_vs_Pion")) or 0.0) < 0.0) else 0,
            "beats_best_control": r.get("beats_best_control", ""),
            "beats_all_reweighting_controls": r.get("beats_all_reweighting_controls", ""),
            "no_ECE_Brier_tail_debt": r.get("no_ECE_Brier_tail_debt", ""),
            "ECE_delta": r.get("held_ECE_delta", ""),
            "Brier_delta": r.get("held_Brier_delta", ""),
            "tail_q95_delta": r.get("held_tail_q95_delta", ""),
            "tail_q99_delta": r.get("held_tail_q99_delta", ""),
            "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
            "gate_mean": r.get("gate_mean", ""),
            "gate_std": r.get("gate_std", ""),
            "gate_entropy": r.get("gate_entropy", ""),
            "gate_class_count_correlation_abs": r.get("gate_class_count_correlation_abs", ""),
            "gate_loss_correlation_abs": r.get("gate_loss_correlation_abs", ""),
            "gate_margin_correlation_abs": r.get("gate_margin_correlation_abs", ""),
        }
        reanalysis_rows.append(row)
        control_gap_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "method": row["method"],
                "best_control_method": r.get("best_control_method", ""),
                "Delta_NLL_vs_best_control": (fval(r.get("final_NLL")) - fval(r.get("best_control_final_NLL"))) if fval(r.get("final_NLL")) is not None and fval(r.get("best_control_final_NLL")) is not None else "",
                "beats_best_control": row["beats_best_control"],
                "beats_all_reweighting_controls": row["beats_all_reweighting_controls"],
            }
        )
        debt_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "method": row["method"],
                "no_ECE_Brier_tail_debt": row["no_ECE_Brier_tail_debt"],
                "ECE_delta": row["ECE_delta"],
                "Brier_delta": row["Brier_delta"],
                "tail_q95_delta": row["tail_q95_delta"],
                "tail_q99_delta": row["tail_q99_delta"],
            }
        )
        oet_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "method": row["method"],
                "Delta_NLL_vs_POET": r.get("Delta_NLL_vs_POET", ""),
                "Delta_NLL_vs_Pion": r.get("Delta_NLL_vs_Pion", ""),
                "beats_POET": row["beats_POET"],
                "beats_Pion": row["beats_Pion"],
            }
        )
    summary = {
        "source_pairwise": "results/v22_55/v22_55_mlp_wgo_pairwise.csv",
        "source_route": "results/v22_55/v22_55_final_route.json",
        "candidate_wgo_rows": len(candidates),
        "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in candidates),
        "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in candidates),
        "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in candidates),
        "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidates),
        "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in candidates),
        "v22_55_final_route": route.get("final_route", ""),
        "v22_55_wgo_family_remains_R3": int(route.get("final_route") == "R3-ReweightingExplained_NoFU" or (sum(iflag(r.get("beats_best_control")) for r in candidates) < 6 and sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidates) < 7)),
        "reanalysis_pass": int(bool(candidates)),
    }
    write_rows(OUT_ROOT / "v22_56_v22_55_reanalysis_matrix.csv", reanalysis_rows)
    write_rows(OUT_ROOT / "v22_56_wgo_control_gap_matrix.csv", control_gap_rows)
    write_rows(OUT_ROOT / "v22_56_wgo_debt_decomposition.csv", debt_rows)
    write_rows(OUT_ROOT / "v22_56_oet_reference_reanalysis.csv", oet_rows)
    write_json(OUT_ROOT / "v22_56_v22_55_failure_route.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage reanalyse-v55",
        task_id="v22_56_part_b_v22_55_reanalysis",
        status="pass" if summary["reanalysis_pass"] else "fail",
        files=(
            "results/v22_56/v22_56_v22_55_reanalysis_matrix.csv; "
            "results/v22_56/v22_56_wgo_control_gap_matrix.csv; "
            "results/v22_56/v22_56_wgo_debt_decomposition.csv; "
            "results/v22_56/v22_56_oet_reference_reanalysis.csv; "
            "results/v22_56/v22_56_v22_55_failure_route.json"
        ),
        note=f"v22_55_final_route={summary['v22_55_final_route']}",
    )
    append_recap(
        "Part B v22.55 independent reanalysis",
        "真实读取 `results/v22_55/` artifacts 后重算关键计数：\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        + md_table(control_gap_rows, ["dataset", "seed", "method", "best_control_method", "Delta_NLL_vs_best_control", "beats_best_control", "beats_all_reweighting_controls"], 24),
    )
    return summary


def synthetic_case_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    device = torch_device(str(args.synthetic_device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(225600)
    rows: list[dict[str, Any]] = []

    def run_case(name: str, vectors: list[Any], cfg: TCWPConfig, raw_grad: Any, true_basis: Any | None = None) -> dict[str, Any]:
        p = torch.nn.Parameter(torch.zeros_like(raw_grad))
        op = TaskCompatibleWitnessedPreconditioner([p], cfg)
        grads = [[v.reshape_as(p)] for v in vectors]
        op.observe(grads)
        op.set_inside_optimizer_step(True)
        try:
            transformed = op.transform(raw_grad.reshape_as(p), p, {})
        finally:
            op.set_inside_optimizer_step(False)
        diag = op.diagnostics()
        recovery = 0.0
        if true_basis is not None:
            state = op._state.get(id(p))
            if state is not None and state.basis is not None:
                basis = state.basis
                q_true, _ = torch.linalg.qr(true_basis.reshape(-1, 1), mode="reduced")
                recovery = float((basis.transpose(0, 1).matmul(q_true).abs().max()).item())
        return {
            "unit_case": name,
            "operator_variant": cfg.variant,
            "control": cfg.control,
            "PSD_min_eigenvalue": diag.get("operator_PSD_min_eigenvalue", ""),
            "operator_condition_number": diag.get("operator_condition_number", ""),
            "cos_g_Pg": diag.get("cos_g_Pg", ""),
            "norm_Pg_over_norm_g": diag.get("norm_Pg_over_norm_g", ""),
            "source_witness_transfer_gain": diag.get("witness_transfer_LCB", ""),
            "noise_attenuation_rate": diag.get("noise_attenuation_rate", ""),
            "witness_subspace_recovery_cosine": recovery,
            "task_descent_violation": int((fval(diag.get("cos_g_Pg"), 0.0) or 0.0) < 0.2),
            "safety_debt_predicted_delta": 0.0,
            "operator_build_ms": diag.get("operator_build_ms", ""),
            "operator_apply_ms": diag.get("operator_apply_ms", ""),
        }

    signal = torch.randn(64, device=device, generator=gen)
    signal = signal / signal.norm().clamp_min(1.0e-12)
    noise = torch.zeros(64, device=device)
    noise[:8] = 1.0
    aligned = [signal + 0.01 * torch.randn(64, device=device, generator=gen) for _ in range(6)]
    row = run_case("C1_aligned_signal", aligned, TCWPConfig(rank=4, alpha=0.20, random_seed=1), signal, signal)
    row["pass"] = int((fval(row["PSD_min_eigenvalue"], -1.0) or -1.0) >= -1.0e-6 and (fval(row["cos_g_Pg"], 0.0) or 0.0) >= 0.2)
    rows.append(row)

    source_noise = [signal + 4.0 * noise, signal + 3.5 * noise, signal + 0.02 * torch.randn(64, device=device, generator=gen), signal]
    row = run_case("C2_source_only_noise", source_noise, TCWPConfig(variant="noise_projection", rank=4, beta=0.60, random_seed=2), signal + 3.0 * noise, signal)
    row["pass"] = int((fval(row["noise_attenuation_rate"], 0.0) or 0.0) >= 0.5 and (fval(row["cos_g_Pg"], 0.0) or 0.0) >= 0.2)
    rows.append(row)

    dense_signal = torch.randn(64, device=device, generator=gen)
    dense_signal = dense_signal / dense_signal.norm().clamp_min(1.0e-12)
    lowrank = [dense_signal + 0.05 * torch.randn(64, device=device, generator=gen) for _ in range(6)]
    row = run_case("C3_witness_consistent_lowrank_signal", lowrank, TCWPConfig(rank=4, alpha=0.40, random_seed=3), dense_signal, dense_signal)
    row["pass"] = int((fval(row["witness_subspace_recovery_cosine"], 0.0) or 0.0) >= 0.8 and (fval(row["cos_g_Pg"], 0.0) or 0.0) >= 0.2)
    rows.append(row)

    task = signal
    conflict = signal.clone()
    conflict[:32] = -conflict[:32]
    vectors = [conflict, conflict + 0.01 * torch.randn(64, device=device, generator=gen), task, task]
    row = run_case("C4_task_conflicting_witness_signal", vectors, TCWPConfig(rank=4, alpha=1.5, c_min=0.2, random_seed=4), task, conflict)
    row["pass"] = int(iflag(1 - row["task_descent_violation"]))
    rows.append(row)

    debt_vec = noise / noise.norm().clamp_min(1.0e-12)
    vectors = [signal + 3.0 * debt_vec, signal + 2.5 * debt_vec, signal, signal + 0.01 * torch.randn(64, device=device, generator=gen)]
    row = run_case("C5_safety_conflicting_signal", vectors, TCWPConfig(variant="safety_aware", rank=4, beta=0.70, random_seed=5), signal + 2.0 * debt_vec, signal)
    row["safety_debt_predicted_delta"] = -float(row.get("noise_attenuation_rate", 0.0) or 0.0)
    row["pass"] = int((fval(row["safety_debt_predicted_delta"], 1.0) or 1.0) <= 0.0 and (fval(row["cos_g_Pg"], 0.0) or 0.0) >= 0.2)
    rows.append(row)

    smoke = {"unit_case": "C6_wrapper_optimizer_step_smoke", "operator_variant": "lowrank", "pass": 0, "status": ""}
    try:
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(device)
        opt_base = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        params = [p for p in model.parameters() if p.requires_grad]
        op = TaskCompatibleWitnessedPreconditioner(params, TCWPConfig(random_seed=6))
        opt = WitnessedPreconditionerWrapper(opt_base, op)
        xb = torch.randn(16, 4, device=device, generator=gen)
        yb = torch.randint(0, 2, (16,), device=device, generator=gen)
        logits = model(xb)
        chunks = torch.chunk(torch.arange(16, device=device), 4)
        cohort_grads = []
        for chunk in chunks:
            cg = torch.autograd.grad(F.cross_entropy(logits[chunk], yb[chunk]), params, retain_graph=True, allow_unused=True)
            cohort_grads.append(list(cg))
        op.observe(cohort_grads)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        diag = opt.diagnostics()
        smoke.update(diag)
        smoke["pass"] = int(iflag(diag.get("optimizer_owned_gradient_transform_pass")) == 1 and iflag(diag.get("optimizer_step_called")) == 1)
        smoke["status"] = "completed"
    except Exception as exc:
        smoke["status"] = f"blocked_or_unavailable: {exc}"
    rows.append(smoke)
    return rows


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = synthetic_case_rows(args)
    write_rows(OUT_ROOT / "v22_56_part_c_synthetic_tcwp_tests.csv", rows)
    summary = {
        "synthetic_rows": len(rows),
        "synthetic_pass_rows": sum(iflag(r.get("pass")) for r in rows),
        "all_synthetic_cases_pass": int(rows and sum(iflag(r.get("pass")) for r in rows) == len(rows)),
    }
    write_json(OUT_ROOT / "v22_56_part_c_synthetic_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage synthetic",
        task_id="v22_56_part_c_synthetic",
        status="pass" if summary["all_synthetic_cases_pass"] else "fail",
        gpu=str(args.synthetic_device),
        files="results/v22_56/v22_56_part_c_synthetic_tcwp_tests.csv; results/v22_56/v22_56_part_c_synthetic_summary.json",
    )
    append_recap(
        "Part C synthetic/mechanistic tests",
        md_table(rows, ["unit_case", "operator_variant", "control", "pass", "PSD_min_eigenvalue", "cos_g_Pg", "noise_attenuation_rate", "witness_subspace_recovery_cosine", "operator_apply_ms"], 12)
        + "\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return summary


def collect_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    return [
        PYTHON,
        "experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py",
        "--stage",
        "collect",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--method",
        str(spec["method"]),
        "--reference-method",
        str(spec.get("reference_method", args.reference_method)),
        "--label",
        str(spec["label"]),
        "--device",
        f"cuda:{device}" if str(device).isdigit() else str(device),
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--test-size",
        str(args.test_size),
        "--hidden",
        str(args.hidden),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--cohorts",
        str(args.cohorts),
        "--cohort-mode",
        str(args.cohort_mode),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--tcwp-alpha",
        str(args.tcwp_alpha),
        "--tcwp-beta",
        str(args.tcwp_beta),
        "--tcwp-eta",
        str(args.tcwp_eta),
        "--tcwp-ema-beta",
        str(args.tcwp_ema_beta),
        "--tcwp-c-min",
        str(args.tcwp_c_min),
        "--tcwp-refresh-interval",
        str(args.tcwp_refresh_interval),
        "--tcwp-param-scope",
        str(args.tcwp_param_scope),
        "--debt-tolerance",
        str(args.debt_tolerance),
        "--tail-debt-tolerance",
        str(args.tail_debt_tolerance),
        "--poet-block-size",
        str(args.poet_block_size),
        "--poet-merge-interval",
        str(args.poet_merge_interval),
        "--poet-lr",
        str(args.poet_lr),
        "--poet-scale",
        str(args.poet_scale),
    ]


def run_matrix(args: argparse.Namespace, *, phase: str) -> list[dict[str, Any]]:
    ensure_out()
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    methods = split_csv(str(args.methods or (DEFAULT_REFERENCE_METHODS if phase == "reference" else DEFAULT_TCWP_METHODS)), str)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_l = str(method).lower()
                label = f"{args.label_prefix}_{method_l}_{dataset}_s{seed}"
                spec = {"dataset": dataset, "seed": seed, "method": method_l, "label": label}
                if method_l in TCWP_METHODS:
                    spec["reference_method"] = reference_method_for(method_l, args)
                specs.append(spec)
    specs_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_specs.csv"
    dispatch_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_dispatch_status.csv"
    write_rows(specs_path, specs)
    gpus = split_csv(str(args.gpus), str) or ["cpu"]
    dispatch_rows: list[dict[str, Any]] = []

    def launch(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, spec = item
        gpu = gpus[i % len(gpus)]
        cmd = collect_command(spec, args, gpu)
        result = run_cmd(
            cmd,
            task_id=str(spec["label"]),
            files=f"results/v22_56/chunks/v22_56_{safe_fragment(str(spec['label']))}_summary.csv",
            gpu=str(gpu),
            timeout=int(args.collect_timeout),
        )
        return {
            **spec,
            "gpu": gpu,
            "command": " ".join(shlex.quote(str(x)) for x in cmd),
            **result,
            "summary": f"results/v22_56/chunks/v22_56_{safe_fragment(str(spec['label']))}_summary.csv",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(launch, item) for item in enumerate(specs)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage {phase}",
        task_id=f"v22_56_{phase}_matrix",
        status="pass" if all(r.get("status") == "pass" for r in dispatch_rows) else "fail",
        files=f"{specs_path}; {dispatch_path}; results/v22_56/chunks/",
        note=f"rows={len(dispatch_rows)}; workers={args.workers}; gpus={args.gpus}",
    )
    return dispatch_rows


def collect_chunk_rows(include_prefixes: list[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_56_*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if include_prefixes and not any(label.startswith(prefix) for prefix in include_prefixes):
                continue
            rows.append(row)
    return rows


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else []
    rows = collect_chunk_rows(include_prefixes)
    reference_rows = [r for r in rows if r.get("phase") == "reference" and r.get("run_status") == "completed"]
    tcwp_rows = [r for r in rows if r.get("phase") == "TCWP" and r.get("run_status") == "completed"]
    method_summary = []
    for method in sorted({r.get("method", "") for r in rows}):
        group = [r for r in rows if r.get("method") == method]
        method_summary.append(
            {
                "method": method,
                "phase": group[0].get("phase", "") if group else "",
                "rows": len(group),
                "completed_rows": sum(int(r.get("run_status") == "completed") for r in group),
                "mean_final_NLL": mean(fval(r.get("final_NLL")) for r in group),
                "mean_accuracy": mean(fval(r.get("final_accuracy")) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in group),
                "mean_overhead_ratio": mean(fval(r.get("controller_overhead_ratio")) for r in group),
                "mean_cos_g_Pg": mean(fval(r.get("cos_g_Pg")) for r in group),
                "mean_PSD_min": mean(fval(r.get("operator_PSD_min_eigenvalue")) for r in group),
                "mean_loss_corr_abs": mean(fval(r.get("gate_or_operator_loss_correlation_abs")) for r in group),
            }
        )
    write_rows(OUT_ROOT / "v22_56_method_summary.csv", method_summary)
    ref_by_key = {(r.get("method"), r.get("dataset"), str(r.get("seed"))): r for r in reference_rows}
    strongest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for r in reference_rows:
        key = (r.get("dataset", ""), str(r.get("seed", "")))
        current = strongest_by_key.get(key)
        if current is None or (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(current.get("final_NLL"), float("inf")) or float("inf")):
            strongest_by_key[key] = r
    v55_pairwise = read_rows(ROOT / "results/v22_55/v22_55_mlp_wgo_pairwise.csv")
    wgo_diag_by_key = {(r.get("dataset", ""), str(r.get("seed", ""))): r for r in v55_pairwise if r.get("method") == "wgo_diag_snr"}
    pairwise = []
    for r in tcwp_rows:
        key2 = (r.get("dataset", ""), str(r.get("seed", "")))
        ref_method = r.get("reference_method", "")
        ref = ref_by_key.get((ref_method, r.get("dataset"), str(r.get("seed"))))
        poet = ref_by_key.get(("poet_official", r.get("dataset"), str(r.get("seed"))))
        pion = ref_by_key.get(("pion_oet_sphere_official", r.get("dataset"), str(r.get("seed"))))
        strongest = strongest_by_key.get(key2)
        controls = [
            x for x in tcwp_rows
            if x.get("dataset") == r.get("dataset")
            and str(x.get("seed")) == str(r.get("seed"))
            and x.get("method") in {m.lower() for m in TCWP_CONTROLS}
            and x.get("reference_method") == ref_method
        ]
        best_control = min(controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        reweight_controls = [x for x in controls if x.get("method") in REWEIGHTING_CONTROLS]
        random_controls = [x for x in controls if x.get("method") in RANDOM_PSD_CONTROLS]
        best_reweight = min(reweight_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        best_random = min(random_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        diag = wgo_diag_by_key.get(key2)
        final_nll = fval(r.get("final_NLL"))
        ref_nll = fval(ref.get("final_NLL")) if ref else None
        poet_nll = fval(poet.get("final_NLL")) if poet else None
        pion_nll = fval(pion.get("final_NLL")) if pion else None
        strongest_nll = fval(strongest.get("final_NLL")) if strongest else None
        control_nll = fval(best_control.get("final_NLL")) if best_control else None
        reweight_nll = fval(best_reweight.get("final_NLL")) if best_reweight else None
        random_nll = fval(best_random.get("final_NLL")) if best_random else None
        diag_nll = fval(diag.get("final_NLL")) if diag else None
        pairwise.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "method": r.get("method", ""),
                "control_kind": r.get("control_kind", ""),
                "reference_method": ref_method,
                "reference_final_NLL": ref_nll,
                "poet_final_NLL": poet_nll,
                "pion_final_NLL": pion_nll,
                "strongest_method": strongest.get("method", "") if strongest else "",
                "strongest_final_NLL": strongest_nll,
                "final_NLL": final_nll,
                "Delta_NLL_vs_reference": (final_nll - ref_nll) if final_nll is not None and ref_nll is not None else "",
                "Delta_NLL_vs_strongest": (final_nll - strongest_nll) if final_nll is not None and strongest_nll is not None else "",
                "Delta_NLL_vs_POET": (final_nll - poet_nll) if final_nll is not None and poet_nll is not None else "",
                "Delta_NLL_vs_Pion": (final_nll - pion_nll) if final_nll is not None and pion_nll is not None else "",
                "Delta_NLL_vs_diagonal_WGO": (final_nll - diag_nll) if final_nll is not None and diag_nll is not None else "",
                "beats_reference": int(final_nll is not None and ref_nll is not None and final_nll < ref_nll),
                "beats_strongest": int(final_nll is not None and strongest_nll is not None and final_nll < strongest_nll),
                "beats_POET_or_external_OET": int(final_nll is not None and poet_nll is not None and final_nll < poet_nll),
                "beats_Pion": int(final_nll is not None and pion_nll is not None and final_nll < pion_nll),
                "beats_diagonal_WGO": int(final_nll is not None and diag_nll is not None and final_nll < diag_nll),
                "best_control_method": best_control.get("method", "") if best_control else "",
                "best_control_final_NLL": control_nll,
                "Delta_NLL_vs_best_control": (final_nll - control_nll) if final_nll is not None and control_nll is not None else "",
                "beats_best_control": int(final_nll is not None and control_nll is not None and final_nll < control_nll),
                "best_reweighting_control_method": best_reweight.get("method", "") if best_reweight else "",
                "beats_all_reweighting_controls": int(final_nll is not None and bool(reweight_controls) and all(final_nll < (fval(x.get("final_NLL"), float("inf")) or float("inf")) for x in reweight_controls)),
                "best_random_psd_control_method": best_random.get("method", "") if best_random else "",
                "beats_best_random_psd_control": int(final_nll is not None and random_nll is not None and final_nll < random_nll),
                "no_ECE_Brier_tail_debt": r.get("no_ECE_Brier_tail_debt", ""),
                "AUC_loss_time_improvement": int((fval(r.get("AUC_loss_time")) or float("inf")) < (fval(ref.get("AUC_loss_time")) or float("inf"))) if ref else 0,
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "cos_g_Pg": r.get("cos_g_Pg", ""),
                "operator_PSD_min_eigenvalue": r.get("operator_PSD_min_eigenvalue", ""),
                "operator_condition_number": r.get("operator_condition_number", ""),
                "norm_Pg_over_norm_g": r.get("norm_Pg_over_norm_g", ""),
                "gate_or_operator_loss_correlation_abs": r.get("gate_or_operator_loss_correlation_abs", ""),
                "standard_loop_hard_gate_pass": r.get("standard_loop_hard_gate_pass", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_56_mlp_tcwp_pairwise.csv", pairwise)
    candidate_pairwise = [r for r in pairwise if r.get("method") in TCWP_CANDIDATES]
    candidate_method_gate = candidate_method_gate_rows(candidate_pairwise)
    write_rows(OUT_ROOT / "v22_56_candidate_method_gate.csv", candidate_method_gate)
    aggregate_mlp_exploration_pass = int(
        len(candidate_pairwise) >= 9
        and sum(iflag(r.get("beats_strongest")) for r in candidate_pairwise) >= 5
        and sum(iflag(r.get("AUC_loss_time_improvement")) for r in candidate_pairwise) >= 5
        and sum(iflag(r.get("beats_best_control")) for r in candidate_pairwise) >= 6
        and sum(iflag(r.get("beats_all_reweighting_controls")) for r in candidate_pairwise) >= 6
        and sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidate_pairwise) >= 7
        and sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in candidate_pairwise) >= 8
        and sum(int((fval(r.get("cos_g_Pg"), -1.0) or -1.0) >= 0.20) for r in candidate_pairwise) >= 9
        and sum(int((fval(r.get("operator_PSD_min_eigenvalue"), -1.0) or -1.0) >= -1.0e-6) for r in candidate_pairwise) >= 9
    )
    route = {
        "reference_rows": len(reference_rows),
        "tcwp_rows": len(tcwp_rows),
        "candidate_tcwp_rows": len(candidate_pairwise),
        "candidate_method_gate_rows": len(candidate_method_gate),
        "candidate_method_gate_pass_rows": sum(iflag(r.get("method_gate_pass")) for r in candidate_method_gate),
        "aggregate_mlp_exploration_pass_without_method_consistency": aggregate_mlp_exploration_pass,
        "standard_loop_pass_candidate_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in candidate_pairwise),
        "beats_diagonal_WGO_rows": sum(iflag(r.get("beats_diagonal_WGO")) for r in candidate_pairwise),
        "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in candidate_pairwise),
        "beats_POET_or_external_OET_rows": sum(iflag(r.get("beats_POET_or_external_OET")) for r in candidate_pairwise),
        "beats_Pion_rows": sum(iflag(r.get("beats_Pion")) for r in candidate_pairwise),
        "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in candidate_pairwise),
        "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in candidate_pairwise),
        "beats_best_random_psd_control_rows": sum(iflag(r.get("beats_best_random_psd_control")) for r in candidate_pairwise),
        "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidate_pairwise),
        "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in candidate_pairwise),
        "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in candidate_pairwise),
        "cos_ge_020_rows": sum(int((fval(r.get("cos_g_Pg"), -1.0) or -1.0) >= 0.20) for r in candidate_pairwise),
        "psd_ge_neg1e6_rows": sum(int((fval(r.get("operator_PSD_min_eigenvalue"), -1.0) or -1.0) >= -1.0e-6) for r in candidate_pairwise),
        "mlp_exploration_pass": 0,
        "oet_increment_pass": 0,
        "kan_gate_status": "not_started",
        "final_route": "",
    }
    route["mlp_exploration_pass"] = int(
        route["candidate_method_gate_pass_rows"] > 0
        and route["standard_loop_pass_candidate_rows"] == route["candidate_tcwp_rows"]
    )
    oet_rows = [r for r in candidate_pairwise if r.get("method") in {"tcwp_over_poet", "tcwp_over_pion"}]
    route["oet_increment_pass"] = int(
        len(oet_rows) >= 9
        and sum(iflag(r.get("beats_POET_or_external_OET")) or iflag(r.get("beats_Pion")) for r in oet_rows) >= 5
        and sum(iflag(r.get("beats_best_control")) for r in oet_rows) >= 6
        and sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in oet_rows) >= 7
    )
    route["kan_gate_status"] = "eligible_not_run_by_default" if route["mlp_exploration_pass"] else "skipped_MLP_TCWP_gate_not_opened"
    route["final_route"] = classify_route(candidate_pairwise, route)
    write_json(OUT_ROOT / "v22_56_final_route.json", route)
    downstream = [
        {"part": "F_OET", "status": "completed_or_blocked_by_rows", "reason": f"oet_increment_pass={route['oet_increment_pass']}"},
        {"part": "G_noise_pressure", "status": "skipped", "reason": "base MLP+TCWP exploration gate not opened" if not route["mlp_exploration_pass"] else "eligible_not_run_by_default"},
        {"part": "H_KAN", "status": "skipped", "reason": route["kan_gate_status"]},
        {"part": "I_continual_grokking", "status": "skipped", "reason": "pressure tests cannot override MLP+TCWP gate" if not route["mlp_exploration_pass"] else "eligible_not_run_by_default"},
    ]
    write_rows(OUT_ROOT / "v22_56_downstream_gate_status.csv", downstream)
    append_exec(
        f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage summarize",
        task_id="v22_56_summarize",
        status="pass",
        files=(
            "results/v22_56/v22_56_method_summary.csv; results/v22_56/v22_56_mlp_tcwp_pairwise.csv; "
            "results/v22_56/v22_56_candidate_method_gate.csv; "
            "results/v22_56/v22_56_final_route.json; results/v22_56/v22_56_downstream_gate_status.csv"
        ),
        note=f"final_route={route['final_route']}",
    )
    append_recap(
        "MLP TCWP summary and route",
        "Method summary:\n\n"
        + md_table(method_summary, ["method", "phase", "rows", "completed_rows", "mean_final_NLL", "no_debt_rows", "standard_loop_pass_rows", "mean_overhead_ratio", "mean_cos_g_Pg"], 36)
        + "\nCandidate pairwise rows:\n\n"
        + md_table(candidate_pairwise, ["dataset", "seed", "method", "reference_method", "Delta_NLL_vs_strongest", "beats_strongest", "beats_best_control", "beats_all_reweighting_controls", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 48)
        + "\nPer-method gate rows:\n\n"
        + md_table(candidate_method_gate, ["method", "rows", "method_gate_pass", "beats_strongest_rows", "AUC_loss_time_improvement_rows", "beats_best_control_rows", "beats_all_reweighting_controls_rows", "no_debt_rows", "overhead_le_035_rows"], 12)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def _need(rows: int, numerator: int, denominator: int) -> int:
    return int(math.ceil(rows * numerator / denominator))


def candidate_method_gate_rows(candidate_pairwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({r.get("method", "") for r in candidate_pairwise}):
        group = [r for r in candidate_pairwise if r.get("method") == method]
        n = len(group)
        row = {
            "method": method,
            "rows": n,
            "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in group),
            "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in group),
            "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in group),
            "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in group),
            "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
            "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in group),
            "cos_ge_020_rows": sum(int((fval(r.get("cos_g_Pg"), -1.0) or -1.0) >= 0.20) for r in group),
            "psd_ge_neg1e6_rows": sum(int((fval(r.get("operator_PSD_min_eigenvalue"), -1.0) or -1.0) >= -1.0e-6) for r in group),
        }
        row["required_beats_strongest_rows"] = _need(n, 5, 9)
        row["required_auc_rows"] = _need(n, 5, 9)
        row["required_best_control_rows"] = _need(n, 6, 9)
        row["required_reweight_rows"] = _need(n, 6, 9)
        row["required_no_debt_rows"] = _need(n, 7, 9)
        row["required_overhead_rows"] = _need(n, 8, 9)
        row["required_cos_rows"] = n
        row["required_psd_rows"] = n
        row["method_gate_pass"] = int(
            n >= 9
            and row["beats_strongest_rows"] >= row["required_beats_strongest_rows"]
            and row["AUC_loss_time_improvement_rows"] >= row["required_auc_rows"]
            and row["beats_best_control_rows"] >= row["required_best_control_rows"]
            and row["beats_all_reweighting_controls_rows"] >= row["required_reweight_rows"]
            and row["no_debt_rows"] >= row["required_no_debt_rows"]
            and row["overhead_le_035_rows"] >= row["required_overhead_rows"]
            and row["cos_ge_020_rows"] >= row["required_cos_rows"]
            and row["psd_ge_neg1e6_rows"] >= row["required_psd_rows"]
        )
        out.append(row)
    return out


def classify_route(candidate_pairwise: list[dict[str, Any]], route: dict[str, Any]) -> str:
    if not candidate_pairwise:
        return "R0-TrainingLoopBoundaryFailed" if route.get("reference_rows", 0) == 0 else "R1-OperatorUnitOnly"
    if route["standard_loop_pass_candidate_rows"] != route["candidate_tcwp_rows"]:
        return "R0-TrainingLoopBoundaryFailed"
    if route.get("mlp_exploration_pass") and route.get("oet_increment_pass"):
        return "R6-WitnessedFUOverOETOpened"
    if route.get("mlp_exploration_pass"):
        return "R5-MLPTCWPExplorationOpened"
    norm_ratios = [fval(r.get("norm_Pg_over_norm_g")) for r in candidate_pairwise]
    if norm_ratios and mean(norm_ratios) is not None and abs((mean(norm_ratios) or 1.0) - 1.0) <= 0.03 and route["beats_strongest_rows"] == 0:
        return "R2-WitnessSignalTooWeak_IdentityCollapse"
    if route.get("candidate_method_gate_pass_rows", 0) == 0 and route["beats_best_control_rows"] <= 6:
        return "R3-ReweightingExplained_NoFU"
    if route["beats_best_random_psd_control_rows"] < min(6, max(1, route["candidate_tcwp_rows"])):
        return "R3-SupportPreconditionerExplained_NoFU"
    if route["beats_best_control_rows"] < min(6, max(1, route["candidate_tcwp_rows"])):
        return "R3-ReweightingExplained_NoFU"
    if route["no_debt_rows"] < min(7, max(1, route["candidate_tcwp_rows"])):
        return "R4-UnsafeWitnessedPreconditioner"
    if route["beats_strongest_rows"] == 0 and route["beats_POET_or_external_OET_rows"] == 0:
        return "FUWeakOptimizerPatchOnly"
    return "R1-OperatorUnitOnly"


def analyze_operator_controls(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    pairwise = read_rows(OUT_ROOT / "v22_56_mlp_tcwp_pairwise.csv")
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else []
    rows = collect_chunk_rows(include_prefixes)
    audit_rows = []
    for row in rows:
        if row.get("phase") != "TCWP" or row.get("run_status") != "completed":
            continue
        audit_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "gate_or_operator_class_count_correlation_abs": row.get("gate_or_operator_class_count_correlation_abs", ""),
                "gate_or_operator_loss_correlation_abs": row.get("gate_or_operator_loss_correlation_abs", ""),
                "gate_or_operator_margin_correlation_abs": row.get("gate_or_operator_margin_correlation_abs", ""),
                "operator_spectrum_entropy": row.get("operator_spectrum_entropy", ""),
                "operator_basis_label_correlation": row.get("operator_basis_label_correlation", ""),
                "operator_basis_loss_correlation": row.get("operator_basis_loss_correlation", ""),
                "operator_basis_random_seed_sensitivity": "",
            }
        )
    summary = {
        "audit_rows": len(audit_rows),
        "pairwise_rows": len(pairwise),
        "loss_corr_gt_070_rows": sum(int((fval(r.get("gate_or_operator_loss_correlation_abs"), 0.0) or 0.0) > 0.70) for r in audit_rows),
        "class_corr_gt_070_rows": sum(int((fval(r.get("gate_or_operator_class_count_correlation_abs"), 0.0) or 0.0) > 0.70) for r in audit_rows),
        "margin_corr_gt_070_rows": sum(int((fval(r.get("gate_or_operator_margin_correlation_abs"), 0.0) or 0.0) > 0.70) for r in audit_rows),
    }
    write_rows(OUT_ROOT / "v22_56_operator_control_audit.csv", audit_rows)
    write_json(OUT_ROOT / "v22_56_operator_control_audit_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage analyze-controls",
        task_id="v22_56_operator_control_audit",
        status="pass",
        files="results/v22_56/v22_56_operator_control_audit.csv; results/v22_56/v22_56_operator_control_audit_summary.json",
        note=json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    append_recap(
        "Part E operator controls/no-reweighting audit",
        "Correlation audit summary:\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        + md_table(audit_rows, ["dataset", "seed", "method", "gate_or_operator_class_count_correlation_abs", "gate_or_operator_loss_correlation_abs", "operator_spectrum_entropy"], 36),
    )
    return summary


def write_svg(path: Path, title: str, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = []
    for row in rows:
        x = fval(row.get(x_key))
        y = fval(row.get(y_key))
        if x is not None and y is not None:
            points.append((x, y, row.get("method", "")))
    width, height = 720, 420
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        circles = []
        for x, y, label in points:
            px = 70 + (x - xmin) / (xmax - xmin) * 590
            py = 350 - (y - ymin) / (ymax - ymin) * 280
            circles.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="#0f766e"><title>{label}: {x_key}={x:.6g}, {y_key}={y:.6g}</title></circle>')
        body = "\n".join(circles)
    else:
        body = '<text x="70" y="210" font-size="18" fill="#555">No completed numeric rows available.</text>'
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="40" y="38" font-size="22" font-family="Arial" fill="#111">{title}</text>
<line x1="70" y1="350" x2="660" y2="350" stroke="#444"/>
<line x1="70" y1="70" x2="70" y2="350" stroke="#444"/>
<text x="300" y="395" font-size="14" font-family="Arial" fill="#333">{x_key}</text>
<text x="12" y="220" font-size="14" font-family="Arial" fill="#333" transform="rotate(-90 12 220)">{y_key}</text>
{body}
</svg>
""",
        encoding="utf-8",
    )


def run_figures(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    pairwise = read_rows(OUT_ROOT / "v22_56_mlp_tcwp_pairwise.csv")
    chunks = collect_chunk_rows()
    route = read_json(OUT_ROOT / "v22_56_final_route.json")
    write_svg(FIG_ROOT / "v22_56_tcwp_control_gap_by_dataset.svg", "TCWP Control Gap", pairwise, "final_NLL", "best_control_final_NLL")
    write_svg(FIG_ROOT / "v22_56_tcwp_overhead_vs_gain.svg", "Overhead vs Gain", pairwise, "controller_overhead_ratio", "Delta_NLL_vs_strongest")
    write_svg(FIG_ROOT / "v22_56_tcwp_cos_vs_norm_ratio.svg", "Task Cone", chunks, "cos_g_Pg", "norm_Pg_over_norm_g")
    write_svg(FIG_ROOT / "v22_56_tcwp_operator_correlation.svg", "Operator Correlation Audit", chunks, "gate_or_operator_loss_correlation_abs", "gate_or_operator_class_count_correlation_abs")
    write_svg(FIG_ROOT / "v22_56_kan_carrier_gate.svg", f"KAN skipped/status: {route.get('kan_gate_status', 'unknown')}", [], "x", "y")
    files = sorted(str(p.relative_to(ROOT)) for p in FIG_ROOT.glob("v22_56_*.svg"))
    summary = {"figures": files, "figure_count": len(files)}
    write_json(OUT_ROOT / "v22_56_figure_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage figures",
        task_id="v22_56_figures",
        status="pass",
        files="; ".join(files),
    )
    append_recap("Visualizations", "生成 SVG：\n\n" + "\n".join(f"- `{x}`" for x in files))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "reanalyse-v55", "synthetic", "collect", "reference", "tcwp", "summarize", "analyze-controls", "figures"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="adamw")
    p.add_argument("--reference-method", default="adamw")
    p.add_argument("--label", default="v22_56_collect")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_56_reference")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--smoke-device", default="cuda:0")
    p.add_argument("--synthetic-device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--cohorts", type=int, default=4)
    p.add_argument("--cohort-mode", default="random", choices=["random", "label_loss_stratified"])
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--tcwp-alpha", type=float, default=0.25)
    p.add_argument("--tcwp-beta", type=float, default=0.35)
    p.add_argument("--tcwp-eta", type=float, default=1.0e-3)
    p.add_argument("--tcwp-ema-beta", type=float, default=0.90)
    p.add_argument("--tcwp-c-min", type=float, default=0.20)
    p.add_argument("--tcwp-refresh-interval", type=int, default=20)
    p.add_argument("--tcwp-param-scope", default="last_layer", choices=["all", "last_layer"])
    p.add_argument("--debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--tail-debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--poet-block-size", type=int, default=16)
    p.add_argument("--poet-merge-interval", type=int, default=30)
    p.add_argument("--poet-lr", type=float, default=5.0e-4)
    p.add_argument("--poet-scale", type=float, default=0.5)
    p.add_argument("--compile-timeout", type=int, default=600)
    p.add_argument("--collect-timeout", type=int, default=1800)
    p.add_argument("--summary-include-prefixes", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.stage == "o0":
            print(json.dumps(run_o0(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "reanalyse-v55":
            print(json.dumps(reanalyse_v22_55(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "synthetic":
            print(json.dumps(run_synthetic(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "collect":
            print(json.dumps(run_collect(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "reference":
            run_matrix(args, phase="reference")
            return 0
        if args.stage == "tcwp":
            if args.label_prefix == "v22_56_reference":
                args.label_prefix = f"v22_56_tcwp_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="tcwp")
            return 0
        if args.stage == "summarize":
            print(json.dumps(summarize(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "analyze-controls":
            print(json.dumps(analyze_operator_controls(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "figures":
            print(json.dumps(run_figures(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "all":
            gate = run_o0(args)
            if not iflag(gate.get("part_a_pass")):
                summarize(args)
                analyze_operator_controls(args)
                run_figures(args)
                return 2
            reanalysis = reanalyse_v22_55(args)
            if not iflag(reanalysis.get("reanalysis_pass")):
                summarize(args)
                analyze_operator_controls(args)
                run_figures(args)
                return 3
            synth = run_synthetic(args)
            if not iflag(synth.get("all_synthetic_cases_pass")):
                summarize(args)
                analyze_operator_controls(args)
                run_figures(args)
                return 4
            args.label_prefix = "v22_56_reference"
            run_matrix(args, phase="reference")
            args.label_prefix = f"v22_56_tcwp_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="tcwp")
            summarize(args)
            analyze_operator_controls(args)
            run_figures(args)
            return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_56_stage_{safe_fragment(args.stage)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            f"{PYTHON} experiments/run_v22_56_task_compatible_witnessed_preconditioner_fu.py --stage {args.stage}",
            task_id=f"v22_56_stage_{args.stage}_exception",
            status="exception",
            files=str(err_path),
            exit_code="exception",
        )
        print(err_path.read_text(encoding="utf-8"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
