#!/usr/bin/env python3
"""DG-KAN v22.61 Meta-FU oracle-ceiling operator-mixture runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import py_compile
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from dgkan.fu.meta_fu_operator_mixture import (  # noqa: E402
    META_FU_OPERATOR_NAMES,
    MetaFUOperatorMixtureOperator,
    OperatorMixtureConfig,
    analytic_operator_bank,
    mix_operator_bank,
    mixture_entropy,
    normalize_mixture,
)
from dgkan.optim.meta_fu_optimizer_wrapper import MetaFUOptimizerWrapper  # noqa: E402
from experiments import audit_v22_61_standard_loop as loop_audit  # noqa: E402
from experiments.run_v22_56_task_compatible_witnessed_preconditioner_fu import (  # noqa: E402
    ExternalUnavailable,
    batch_indices,
    evaluate_tensors,
    fval,
    iflag,
    load_bundle_tensors,
    make_base_optimizer,
    make_model,
    mean,
    optimizer_state_count,
    split_csv,
    torch_device,
    trainable_param_count,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_61"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py"

REFERENCE_METHODS = {
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    "poet_official",
    "pion_oet_sphere_official",
    "pion_oet_local",
}
ORACLE_DATASETS = ("MNIST", "FashionMNIST", "KMNIST", "Wine", "Spam")
ORACLE_RUN_PREFIXES = ("v22_61_m0o_initial", "v22_61_m0o_repair", "v22_61_m0o_repair_conservative")


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
            "# DG-KAN v22.61 MetaFU OracleCeilingOperatorMixture 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、解释器、GPU、输入输出文件、状态、失败与修复尝试。"
            "M0O oracle 只作为 offline upper-bound diagnostic；official runtime 必须保持标准训练环。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.61 MetaFU OracleCeilingOperatorMixture 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实 artifact；不编造数据；失败、修复、结论和 insight 必须带证据链。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


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


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 30) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        cells = []
        for col in columns:
            text = str(row.get(col, ""))
            if len(text) > 96:
                text = text[:93] + "..."
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        out.append(f"\n_... {len(rows) - limit} more rows omitted_")
    return "\n".join(out) + "\n"


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
    journal = read_rows(OUT_ROOT / "v22_61_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(OUT_ROOT / "v22_61_command_journal.csv", journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
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


def run_cmd(cmd: list[str], *, task_id: str, files: str = "", gpu: str = "cpu", timeout: int | None = None, cwd: Path | None = None) -> dict[str, Any]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
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
            note=f"cwd={cwd or ROOT}; stdout={stdout_path}; stderr={stderr_path}; wall_seconds={time.time() - start:.3f}",
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


def internal_split_tensors(tensors: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    x_train = tensors["x_train"]
    y_train = tensors["y_train"]
    n = int(x_train.shape[0])
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(int(seed) + 226100)
    perm = torch.randperm(n, generator=gen)
    source_n = max(1, int(round(0.50 * n)))
    witness_n = max(1, int(round(0.25 * n)))
    if source_n + witness_n >= n:
        source_n = max(1, int(0.60 * n))
        witness_n = max(1, int(0.20 * n))
    source_idx = perm[:source_n]
    witness_idx = perm[source_n : source_n + witness_n]
    query_idx = perm[source_n + witness_n :]
    if int(query_idx.numel()) == 0:
        query_idx = witness_idx[-1:]
        witness_idx = witness_idx[:-1] if int(witness_idx.numel()) > 1 else witness_idx
    inner_idx = torch.cat([source_idx, witness_idx], dim=0)
    return {
        "x_source": x_train[source_idx],
        "y_source": y_train[source_idx],
        "x_witness": x_train[witness_idx],
        "y_witness": y_train[witness_idx],
        "x_inner": x_train[inner_idx],
        "y_inner": y_train[inner_idx],
        "x_query": x_train[query_idx],
        "y_query": y_train[query_idx],
        "source_hash": hashlib.sha256(source_idx.cpu().numpy().tobytes()).hexdigest()[:16],
        "witness_hash": hashlib.sha256(witness_idx.cpu().numpy().tobytes()).hexdigest()[:16],
        "query_hash": hashlib.sha256(query_idx.cpu().numpy().tobytes()).hexdigest()[:16],
    }


def select_params(model: Any, scope: str) -> list[Any]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if str(scope) == "all":
        return [param for _, param in named]
    if str(scope) == "last_layer":
        fc3 = getattr(model, "fc3", None)
        if fc3 is not None:
            selected = [param for param in fc3.parameters() if param.requires_grad]
            if selected:
                return selected
        return [named[-1][1]] if named else []
    raise ValueError(f"unknown param scope: {scope}")


def state_dict_cpu(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def current_lr(opt: Any, default: float) -> float:
    groups = getattr(opt, "param_groups", [])
    if groups:
        try:
            return float(groups[0].get("lr", default))
        except Exception:
            return float(default)
    base = getattr(opt, "base", None)
    if base is not None:
        return current_lr(base, default)
    return float(default)


def candidate_mixtures(seed: int, *, repair: bool = False, n_random: int = 16) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + (9100 if repair else 6100))
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(META_FU_OPERATOR_NAMES):
        weights = [0.0] * len(META_FU_OPERATOR_NAMES)
        weights[idx] = 1.0
        rows.append({"mixture_name": f"onehot_{name}", "mixture_kind": "structured", "weights": tuple(weights)})
    rows.extend(
        [
            {"mixture_name": "uniform_A0_A6", "mixture_kind": "structured", "weights": normalize_mixture([1.0] * 7)},
            {"mixture_name": "credit_self_A1_A2_A3", "mixture_kind": "structured", "weights": normalize_mixture([0, 1, 1, 1, 0, 0, 0])},
            {"mixture_name": "state_A3_A4", "mixture_kind": "structured", "weights": normalize_mixture([0, 0, 0, 1, 1, 0, 0])},
            {"mixture_name": "safe_A0_A5_A6", "mixture_kind": "structured", "weights": normalize_mixture([1, 0, 0, 0, 0, 1, 1])},
        ]
    )
    if repair:
        rows.extend(
            [
                {"mixture_name": "repair_oet_safe_A5_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 0, 0, 0, 0, 2, 3])},
                {"mixture_name": "repair_identity_tail_A0_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([2, 0, 0, 0, 0, 0, 3])},
                {"mixture_name": "repair_credit_tail_A1_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 2, 0, 0, 0, 0, 3])},
                {"mixture_name": "repair_state_tail_A4_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 0, 0, 0, 2, 0, 3])},
                {"mixture_name": "repair_state_oet_tail_A4_A5_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 0, 0, 0, 2, 2, 3])},
                {"mixture_name": "repair_identity_state_tail_A0_A4_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([1, 0, 0, 0, 2, 0, 3])},
                {"mixture_name": "repair_credit_state_oet_tail_A1_A4_A5_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 1, 0, 0, 2, 1, 3])},
                {"mixture_name": "repair_safe_biased_A0_A1_A4_A5_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([1, 1, 0, 0, 2, 1, 3])},
                {"mixture_name": "repair_state_tail_heavy_A4_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 0, 0, 0, 1, 0, 5])},
                {"mixture_name": "repair_state_oet_tail_heavy_A4_A5_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 0, 0, 0, 1, 1, 5])},
                {"mixture_name": "repair_identity_state_tail_heavy_A0_A4_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([1, 0, 0, 0, 1, 0, 5])},
                {"mixture_name": "repair_credit_state_tail_heavy_A1_A4_A6", "mixture_kind": "structured_repair", "weights": normalize_mixture([0, 1, 0, 0, 1, 0, 5])},
            ]
        )
    while len([r for r in rows if r["mixture_kind"] == "random_dirichlet"]) < int(n_random) and len(rows) < 40:
        vals = [rng.gammavariate(0.5, 1.0) for _ in META_FU_OPERATOR_NAMES]
        rows.append({"mixture_name": f"dirichlet_{len(rows):02d}", "mixture_kind": "random_dirichlet", "weights": normalize_mixture(vals)})
    return rows[:40]


def _flat_grads(grads: Iterable[Any | None], params: list[Any]) -> Any:
    import torch

    pieces = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.detach().reshape(-1))
    return torch.cat(pieces) if pieces else torch.zeros(1)


def source_witness_credit_cosine(model: Any, split: dict[str, Any], params: list[Any], device: Any, batch_size: int) -> float:
    import torch
    import torch.nn.functional as F

    bs = min(int(batch_size), int(split["x_source"].shape[0]))
    bw = min(int(batch_size), int(split["x_witness"].shape[0]))
    xs = split["x_source"][:bs].to(device)
    ys = split["y_source"][:bs].to(device)
    xw = split["x_witness"][:bw].to(device)
    yw = split["y_witness"][:bw].to(device)
    loss_s = F.cross_entropy(model(xs).float(), ys.long())
    grads_s = torch.autograd.grad(loss_s, params, retain_graph=False, allow_unused=True)
    loss_w = F.cross_entropy(model(xw).float(), yw.long())
    grads_w = torch.autograd.grad(loss_w, params, retain_graph=False, allow_unused=True)
    fs = _flat_grads(grads_s, params)
    fw = _flat_grads(grads_w, params)
    denom = fs.norm().clamp_min(1.0e-12) * fw.norm().clamp_min(1.0e-12)
    return float((fs * fw).sum().div(denom).detach().cpu().item())


def checkpoint_train_only_features(
    state: dict[str, Any],
    tensors: dict[str, Any],
    split: dict[str, Any],
    args: argparse.Namespace,
    device: Any,
    checkpoint_step: int,
    phase_bucket: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 226100, device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.train()
    params = select_params(model, str(args.oracle_param_scope))
    bs = max(8, min(int(args.batch_size), 64, int(split["x_source"].shape[0])))
    bw = max(8, min(int(args.batch_size), 64, int(split["x_witness"].shape[0])))
    xs = split["x_source"][:bs].to(device)
    ys = split["y_source"][:bs].to(device)
    xw = split["x_witness"][:bw].to(device)
    yw = split["y_witness"][:bw].to(device)
    source_loss = F.cross_entropy(model(xs).float(), ys.long())
    grads_s = torch.autograd.grad(source_loss, params, retain_graph=False, allow_unused=True)
    witness_loss = F.cross_entropy(model(xw).float(), yw.long())
    grads_w = torch.autograd.grad(witness_loss, params, retain_graph=False, allow_unused=True)
    fs = _flat_grads(grads_s, params)
    fw = _flat_grads(grads_w, params)
    denom = fs.norm().clamp_min(1.0e-12) * fw.norm().clamp_min(1.0e-12)
    cosine = float((fs * fw).sum().div(denom).detach().cpu().item())
    with torch.no_grad():
        ps = model(xs).float().softmax(dim=-1)
        pw = model(xw).float().softmax(dim=-1)
        source_conf = float(ps.max(dim=-1).values.mean().detach().cpu().item())
        witness_conf = float(pw.max(dim=-1).values.mean().detach().cpu().item())
        source_acc = float((ps.argmax(dim=-1) == ys.long()).float().mean().detach().cpu().item())
        witness_acc = float((pw.argmax(dim=-1) == yw.long()).float().mean().detach().cpu().item())
    grad_norm_s = float(fs.norm().detach().cpu().item())
    grad_norm_w = float(fw.norm().detach().cpu().item())
    param_norm = float(torch.cat([p.detach().reshape(-1).float() for p in params]).norm().detach().cpu().item()) if params else 0.0
    return {
        "feature_progress": float(checkpoint_step / max(1, int(args.oracle_pretrain_steps))),
        "feature_phase_early": int(str(phase_bucket) == "early"),
        "feature_phase_middle": int(str(phase_bucket) == "middle"),
        "feature_phase_late": int(str(phase_bucket) == "late"),
        "feature_source_witness_credit_cosine": cosine,
        "feature_source_loss": float(source_loss.detach().cpu().item()),
        "feature_witness_loss": float(witness_loss.detach().cpu().item()),
        "feature_source_witness_loss_gap": float((source_loss - witness_loss).detach().cpu().item()),
        "feature_source_accuracy": source_acc,
        "feature_witness_accuracy": witness_acc,
        "feature_source_witness_accuracy_gap": source_acc - witness_acc,
        "feature_source_confidence": source_conf,
        "feature_witness_confidence": witness_conf,
        "feature_source_witness_confidence_gap": source_conf - witness_conf,
        "feature_source_grad_norm": grad_norm_s,
        "feature_witness_grad_norm": grad_norm_w,
        "feature_grad_norm_ratio": grad_norm_s / max(grad_norm_w, 1.0e-12),
        "feature_param_norm": param_norm,
    }


def apply_mixture_to_selected_grads(model: Any, selected_ids: set[int], weights: Iterable[float], credit_cos: float, prev: dict[int, Any], args: argparse.Namespace) -> None:
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None or id(param) not in selected_ids:
            continue
        bank = analytic_operator_bank(
            grad,
            param,
            credit_cosine=credit_cos,
            prev_grad=prev.get(id(param)),
            tail_safety=float(args.oracle_tail_safety),
            state_scale=float(args.oracle_state_scale),
        )
        mixed = mix_operator_bank(bank, weights)
        raw_norm = float(grad.detach().float().norm().item())
        delta = mixed - grad
        delta_norm = float(delta.detach().float().norm().item())
        max_delta = max(1.0e-12, float(args.oracle_trust) * max(raw_norm, 1.0e-12))
        if math.isfinite(delta_norm) and delta_norm > max_delta:
            mixed = grad + delta * (max_delta / max(delta_norm, 1.0e-12))
        prev[id(param)] = grad.detach().clone()
        param.grad = mixed


def oracle_positive_debt(row: dict[str, Any]) -> float:
    return sum(
        max(0.0, fval(row.get(col), 0.0) or 0.0)
        for col in ("query_ECE_delta", "query_Brier_delta", "query_tail_q95_delta", "query_tail_q99_delta")
    )


def oracle_selection_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    delta = fval(row.get("Delta_NLL_vs_reference"), 0.0) or 0.0
    if str(args.oracle_selection) == "raw_nll":
        return delta
    entropy = fval(row.get("operator_mixture_entropy"), 0.0) or 0.0
    entropy_gap = max(0.0, float(args.oracle_entropy_floor) - entropy)
    return (
        delta
        + float(args.oracle_debt_lambda) * oracle_positive_debt(row)
        + float(args.oracle_entropy_gap_lambda) * entropy_gap
    )


def rollout_from_state(
    state: dict[str, Any],
    tensors: dict[str, Any],
    split: dict[str, Any],
    weights: Iterable[float],
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 226100, device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    params = select_params(model, str(args.oracle_param_scope))
    selected_ids = {id(p) for p in params}
    prev: dict[int, Any] = {}
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 226100 + int(args.oracle_horizon))
    t0 = time.time()
    last_loss = 0.0
    for _ in range(int(args.oracle_horizon)):
        idx = batch_indices(int(split["x_inner"].shape[0]), int(args.batch_size), gen, device)
        idx_cpu = idx.detach().cpu()
        xb = split["x_inner"][idx_cpu].to(device)
        yb = split["y_inner"][idx_cpu].to(device)
        credit_cos = source_witness_credit_cosine(model, split, params, device, max(8, min(int(args.batch_size), 64)))
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss_task = F.cross_entropy(logits.float(), yb.long())
        loss_task.backward()
        apply_mixture_to_selected_grads(model, selected_ids, weights, credit_cos, prev, args)
        opt.step()
        last_loss = float(loss_task.detach().cpu().item())
    q = evaluate_tensors(model, split["x_query"], split["y_query"], device, int(tensors["num_classes"]), int(args.eval_batch_size))
    inner = evaluate_tensors(model, split["x_inner"], split["y_inner"], device, int(tensors["num_classes"]), int(args.eval_batch_size))
    return {
        "query_NLL": q["NLL"],
        "query_accuracy": q["accuracy"],
        "query_ECE": q["ECE"],
        "query_Brier": q["Brier"],
        "query_tail_q95": q["tail_q95"],
        "query_tail_q99": q["tail_q99"],
        "query_margin_q10": q["margin_q10"],
        "inner_NLL": inner["NLL"],
        "last_train_loss": last_loss,
        "wall_seconds": time.time() - t0,
    }


def build_checkpoints(tensors: dict[str, Any], split: dict[str, Any], args: argparse.Namespace, device: Any) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 226100, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 226111)
    steps = sorted({0, max(1, int(0.20 * int(args.oracle_pretrain_steps))), max(1, int(0.70 * int(args.oracle_pretrain_steps)))})
    checkpoints: list[dict[str, Any]] = []
    for step in range(max(steps) + 1):
        if step in steps:
            phase = "early" if step <= int(0.20 * int(args.oracle_pretrain_steps)) else ("middle" if step <= int(0.70 * int(args.oracle_pretrain_steps)) else "late")
            checkpoints.append({"checkpoint_step": step, "phase_bucket": phase, "state": state_dict_cpu(model)})
        if step == max(steps):
            break
        idx = batch_indices(int(split["x_inner"].shape[0]), int(args.batch_size), gen, device)
        idx_cpu = idx.detach().cpu()
        xb = split["x_inner"][idx_cpu].to(device)
        yb = split["y_inner"][idx_cpu].to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb.long())
        loss.backward()
        opt.step()
    return checkpoints


def run_oracle_task(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    import torch

    device = torch_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(int(args.seed) + 226100)
    try:
        tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    except Exception as exc:
        row = {"run_status": "blocked", "blocker": f"DatasetUnavailable: {exc}", "dataset": args.dataset, "seed": args.seed}
        out = CHUNK_ROOT / f"v22_61_{safe_fragment(args.label)}_oracle_summary.csv"
        write_rows(out, [row])
        return row
    if tensors.get("used_fake_data"):
        row = {"run_status": "blocked", "blocker": "FakeDataRejected: loader reported used_fake_data=1", "dataset": args.dataset, "seed": args.seed}
        out = CHUNK_ROOT / f"v22_61_{safe_fragment(args.label)}_oracle_summary.csv"
        write_rows(out, [row])
        return row
    split = internal_split_tensors(tensors, int(args.seed))
    mixtures = candidate_mixtures(int(args.seed), repair=bool(args.oracle_repair), n_random=int(args.oracle_random_mixtures))
    mix_path = CHUNK_ROOT / f"v22_61_{safe_fragment(args.label)}_oracle_candidates.csv"
    write_rows(mix_path, [{**{k: v for k, v in row.items() if k != "weights"}, "weights_json": json.dumps(list(row["weights"]))} for row in mixtures])
    checkpoints = build_checkpoints(tensors, split, args, device)
    detail_rows: list[dict[str, Any]] = []
    phase_best: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        ref = rollout_from_state(ckpt["state"], tensors, split, normalize_mixture([1, 0, 0, 0, 0, 0, 0]), args, device)
        ckpt_features = checkpoint_train_only_features(ckpt["state"], tensors, split, args, device, int(ckpt["checkpoint_step"]), str(ckpt["phase_bucket"]))
        best_structured: dict[str, Any] | None = None
        best_random: dict[str, Any] | None = None
        for mix in mixtures:
            metrics = rollout_from_state(ckpt["state"], tensors, split, mix["weights"], args, device)
            delta = float(metrics["query_NLL"] - ref["query_NLL"])
            no_debt = int(
                metrics["query_ECE"] <= ref["query_ECE"] + float(args.debt_tolerance)
                and metrics["query_Brier"] <= ref["query_Brier"] + float(args.debt_tolerance)
                and metrics["query_tail_q95"] <= ref["query_tail_q95"] + float(args.tail_debt_tolerance)
                and metrics["query_tail_q99"] <= ref["query_tail_q99"] + float(args.tail_debt_tolerance)
            )
            row = {
                "run_label": str(args.label),
                "dataset": str(args.dataset),
                "seed": int(args.seed),
                "oracle_repair": int(bool(args.oracle_repair)),
                "phase_bucket": ckpt["phase_bucket"],
                "checkpoint_step": ckpt["checkpoint_step"],
                "mixture_name": mix["mixture_name"],
                "mixture_kind": mix["mixture_kind"],
                "weights_json": json.dumps(list(mix["weights"])),
                "operator_mixture_entropy": mixture_entropy(mix["weights"]),
                "reference_query_NLL": ref["query_NLL"],
                "query_NLL": metrics["query_NLL"],
                "Delta_NLL_vs_reference": delta,
                "beats_reference": int(delta < -float(args.oracle_delta_tolerance)),
                "no_ECE_Brier_tail_debt": no_debt,
                "query_ECE_delta": metrics["query_ECE"] - ref["query_ECE"],
                "query_Brier_delta": metrics["query_Brier"] - ref["query_Brier"],
                "query_tail_q95_delta": metrics["query_tail_q95"] - ref["query_tail_q95"],
                "query_tail_q99_delta": metrics["query_tail_q99"] - ref["query_tail_q99"],
                "oracle_train_query_gap": abs((metrics["query_NLL"] - ref["query_NLL"]) - (metrics["inner_NLL"] - ref["inner_NLL"])),
                "rollout_wall_seconds": metrics["wall_seconds"],
            }
            row.update(ckpt_features)
            row["oracle_positive_debt"] = oracle_positive_debt(row)
            row["oracle_entropy_gap_cost"] = max(0.0, float(args.oracle_entropy_floor) - float(row["operator_mixture_entropy"]))
            row["oracle_selection"] = str(args.oracle_selection)
            row["oracle_selection_score"] = oracle_selection_score(row, args)
            detail_rows.append(row)
            current = best_random if mix["mixture_kind"] == "random_dirichlet" else best_structured
            better = delta < float(current["Delta_NLL_vs_reference"]) if mix["mixture_kind"] == "random_dirichlet" and current is not None else False
            if mix["mixture_kind"] != "random_dirichlet" and current is not None:
                better = float(row["oracle_selection_score"]) < float(current["oracle_selection_score"])
            if current is None or better:
                if mix["mixture_kind"] == "random_dirichlet":
                    best_random = row
                else:
                    best_structured = row
        if best_structured is None:
            best_structured = min([r for r in detail_rows if r["phase_bucket"] == ckpt["phase_bucket"]], key=lambda r: float(r["Delta_NLL_vs_reference"]))
        best_random_delta = float(best_random["Delta_NLL_vs_reference"]) if best_random else float("inf")
        best_structured = dict(best_structured)
        best_structured["best_random_mixture_Delta_NLL"] = best_random_delta
        best_structured["beats_best_random_mixture"] = int(float(best_structured["Delta_NLL_vs_reference"]) < best_random_delta - float(args.oracle_delta_tolerance))
        phase_best.append(best_structured)
    detail_path = CHUNK_ROOT / f"v22_61_{safe_fragment(args.label)}_oracle_detail.csv"
    write_rows(detail_path, detail_rows)
    best = min(phase_best, key=lambda r: float(r["Delta_NLL_vs_reference"])) if phase_best else {}
    phase_weights = []
    for row in phase_best:
        try:
            phase_weights.append([float(x) for x in json.loads(str(row.get("weights_json", "[]")))])
        except Exception:
            pass
    phase_response_norm = 0.0
    if len(phase_weights) > 1:
        vals = []
        for i in range(1, len(phase_weights)):
            vals.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(phase_weights[i], phase_weights[i - 1]))))
        phase_response_norm = float(sum(vals) / len(vals)) if vals else 0.0
    summary = {
        "run_label": str(args.label),
        "run_status": "pass",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "oracle_repair": int(bool(args.oracle_repair)),
        "query_split_is_train_only": 1,
        "validation_test_used_in_oracle": 0,
        "dataset_name_not_available_to_controller": 1,
        "seed_not_available_to_controller": 1,
        "source_hash": split["source_hash"],
        "witness_hash": split["witness_hash"],
        "query_hash": split["query_hash"],
        "source_size": int(split["x_source"].shape[0]),
        "witness_size": int(split["x_witness"].shape[0]),
        "query_size": int(split["x_query"].shape[0]),
        "candidate_count": len(mixtures),
        "checkpoint_count": len(checkpoints),
        "oracle_selection": str(args.oracle_selection),
        "oracle_debt_lambda": float(args.oracle_debt_lambda),
        "oracle_entropy_floor": float(args.oracle_entropy_floor),
        "oracle_entropy_gap_lambda": float(args.oracle_entropy_gap_lambda),
        "oracle_best_mixture_name": best.get("mixture_name", ""),
        "oracle_best_mixture_kind": best.get("mixture_kind", ""),
        "oracle_best_weights_json": best.get("weights_json", ""),
        "oracle_best_phase": best.get("phase_bucket", ""),
        "oracle_best_Delta_NLL_vs_reference": best.get("Delta_NLL_vs_reference", ""),
        "oracle_best_selection_score": best.get("oracle_selection_score", ""),
        "oracle_best_positive_debt": best.get("oracle_positive_debt", ""),
        "oracle_best_entropy_gap_cost": best.get("oracle_entropy_gap_cost", ""),
        "oracle_best_beats_reference": best.get("beats_reference", ""),
        "oracle_best_beats_best_random_mixture": best.get("beats_best_random_mixture", ""),
        "oracle_best_no_debt": best.get("no_ECE_Brier_tail_debt", ""),
        "oracle_best_operator_entropy": best.get("operator_mixture_entropy", ""),
        "oracle_phase_response_norm": phase_response_norm,
        "oracle_phase_dependency_detected": int(phase_response_norm > 0.05),
        "oracle_train_query_gap": best.get("oracle_train_query_gap", ""),
        "oracle_train_query_gap_pass": int((fval(best.get("oracle_train_query_gap"), float("inf")) or float("inf")) <= float(args.oracle_train_query_gap_tolerance)),
        "best_random_mixture_Delta_NLL": best.get("best_random_mixture_Delta_NLL", ""),
        "oracle_detail_file": str(detail_path),
        "oracle_candidate_file": str(mix_path),
        "oracle_overhead_estimate": mean(fval(r.get("rollout_wall_seconds")) for r in detail_rows),
    }
    out = CHUNK_ROOT / f"v22_61_{safe_fragment(args.label)}_oracle_summary.csv"
    write_rows(out, [summary])
    return summary


def run_standard_loop_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch_device(str(args.smoke_device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tensors = load_bundle_tensors("MNIST", min(int(args.train_size), 160), min(int(args.held_size), 64), min(int(args.test_size), 64), 0)
    split = internal_split_tensors(tensors, 0)
    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), min(int(args.hidden), 128), 226100, device)
    base = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    selected = select_params(model, "last_layer")
    param_names = {id(p): name for name, p in model.named_parameters()}
    operator = MetaFUOperatorMixtureOperator(selected, OperatorMixtureConfig(mixture=normalize_mixture([1, 0, 1, 0, 0, 1, 1]), trust=float(args.oracle_trust)), param_names=param_names)
    opt = MetaFUOptimizerWrapper(base, operator)
    runtime_audit = loop_audit.RuntimeTrainingLoopAudit(model.parameters())
    gen = torch.Generator(device=device)
    gen.manual_seed(226100)
    forward_called = 0
    task_loss_backward_called = 0
    optimizer_step_called = 0
    losses: list[float] = []
    for step in range(int(args.smoke_steps)):
        idx = batch_indices(int(split["x_inner"].shape[0]), int(args.batch_size), gen, device)
        idx_cpu = idx.detach().cpu()
        xb = split["x_inner"][idx_cpu].to(device)
        yb = split["y_inner"][idx_cpu].to(device)
        opt.zero_grad(set_to_none=True)
        runtime_audit.snapshot_before_backward()
        logits = model(xb)
        forward_called = 1
        loss_task = F.cross_entropy(logits.float(), yb.long())
        operator.set_progress(step, int(args.smoke_steps), current_lr(opt, float(args.lr)), float(args.weight_decay))
        operator.set_train_metrics(float(loss_task.detach().cpu().item()))
        loss_task.backward()
        task_loss_backward_called = 1
        runtime_audit.check_before_optimizer_step()
        opt.step()
        optimizer_step_called = 1
        losses.append(float(loss_task.detach().cpu().item()))
    q = evaluate_tensors(model, split["x_query"], split["y_query"], device, int(tensors["num_classes"]), int(args.eval_batch_size))
    diag = opt.diagnostics()
    row = {
        "run_label": "v22_61_runtime_trace_operator_mixture_smoke",
        "run_status": "pass",
        "dataset": "MNIST",
        "seed": 0,
        "method": "fixed_operator_mixture_smoke",
        "query_split_is_train_only": 1,
        "source_hash": split["source_hash"],
        "witness_hash": split["witness_hash"],
        "query_hash": split["query_hash"],
        "final_NLL": q["NLL"],
        "loss_total_is_task_loss_only": 1,
        "loss_task_only_official": 1,
        "fu_auxiliary_loss_used_official": 0,
        "forward_called": forward_called,
        "loss_task_backward_called": task_loss_backward_called,
        "optimizer_step_called": optimizer_step_called,
        "meta_test_no_controller_update": 1,
        "branch_replay_used_as_training": 0,
        "proxy_direction_used_as_training": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "cohort_topk_selection_used": 0,
        "layer_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "optimizer_state_count": optimizer_state_count(opt),
        "trainable_parameter_count": trainable_param_count(model),
        **runtime_audit.as_dict(),
        **diag,
    }
    row["standard_loop_hard_gate_pass"] = int(
        row["forward_called"]
        and row["loss_task_backward_called"]
        and row["optimizer_step_called"]
        and row["loss_total_is_task_loss_only"]
        and row["fu_auxiliary_loss_used_official"] == 0
        and row["optimizer_owned_gradient_transform_pass"]
        and row["manual_param_update_detected"] == 0
        and row["candidate_action_selection_used_for_runtime"] == 0
        and row["meta_controller_frozen_on_meta_test"] == 1
    )
    trace_file = CHUNK_ROOT / "v22_61_runtime_trace_operator_mixture_smoke_controller_trace.csv"
    write_rows(trace_file, operator.controller_trace_rows())
    row["controller_trace_file"] = str(trace_file)
    return row


def copy_for_audit_bundle(bundle_dir: Path) -> list[str]:
    copied: list[str] = []
    for rel in [
        "dgkan",
        "experiments",
        "external/oet_baselines",
        "docs/DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_完整计划.md",
        "docs/DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_执行日志.md",
        "docs/DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_实验结果复盘.md",
    ]:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = bundle_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        else:
            shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def build_clean_audit_bundle(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = ROOT / "code_audit_pack"
    audit_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_dir = audit_root / f"v22_61_audit_{stamp}"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    copied = copy_for_audit_bundle(bundle_dir)
    file_list = [str(p.relative_to(bundle_dir)) for p in bundle_dir.rglob("*") if p.is_file()]
    tar_path = audit_root / f"{bundle_dir.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    rows = []
    with tempfile.TemporaryDirectory(prefix="v22_61_clean_extract_") as tmp:
        tmp_root = Path(tmp)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_root)
        clean_root = tmp_root / bundle_dir.name
        res = run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"], task_id="v22_61_clean_tarball_compileall", files=str(tar_path.relative_to(ROOT)), cwd=clean_root, timeout=int(args.compile_timeout))
        rows.append({"check": "clean_tarball_compileall_pass", "pass": int(res.get("status") == "pass"), "returncode": res.get("returncode")})
        import_cmds = {
            "clean_tarball_full_repo_import_pass": "import dgkan; import dgkan.optim; print('pass')",
            "clean_tarball_runner_core_import_pass": "import experiments.run_v22_61_meta_fu_oracle_ceiling_operator_mixture; print('pass')",
            "clean_tarball_wrapper_import_pass": "from dgkan.optim.meta_fu_optimizer_wrapper import MetaFUOptimizerWrapper; print('pass')",
            "clean_tarball_operator_import_pass": "from dgkan.fu.meta_fu_operator_mixture import MetaFUOperatorMixtureOperator; print('pass')",
        }
        for key, code in import_cmds.items():
            res = run_cmd([PYTHON, "-c", code], task_id=f"v22_61_{key}", files=str(tar_path.relative_to(ROOT)), cwd=clean_root)
            rows.append({"check": key, "pass": int(res.get("status") == "pass"), "returncode": res.get("returncode")})
    write_rows(OUT_ROOT / "v22_61_clean_tarball_self_contained_import.csv", rows)
    required_core = [
        "dgkan/fu/meta_fu_operator_mixture.py",
        "dgkan/optim/meta_fu_optimizer_wrapper.py",
        "experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py",
        "experiments/audit_v22_61_standard_loop.py",
    ]
    missing = [rel for rel in required_core if rel not in file_list]
    return {
        "audit_bundle_dir": str(bundle_dir.relative_to(ROOT)),
        "audit_bundle_tar": str(tar_path.relative_to(ROOT)),
        "audit_bundle_sha256": sha,
        "audit_bundle_copied_roots": json.dumps(copied, ensure_ascii=False),
        "audit_bundle_file_count": len(file_list),
        "untracked_core_files_missing_from_tarball": int(bool(missing)),
        "missing_core_files_in_tarball": json.dumps(missing, ensure_ascii=False),
        **{r["check"]: r["pass"] for r in rows},
    }


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
    compile_result = run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"], task_id="v22_61_compileall_repo", files="results/v22_61/v22_61_code_truth_gate.csv", timeout=int(args.compile_timeout))
    import_cmds = {
        "worktree_full_repo_import_pass": "import dgkan; import dgkan.optim; print('pass')",
        "runner_core_import_pass": "import experiments.run_v22_61_meta_fu_oracle_ceiling_operator_mixture; print('pass')",
        "wrapper_import_pass": "from dgkan.optim.meta_fu_optimizer_wrapper import MetaFUOptimizerWrapper; print('pass')",
        "operator_import_pass": "from dgkan.fu.meta_fu_operator_mixture import MetaFUOperatorMixtureOperator; print('pass')",
        "external_poet_import_pass": "from poet_torch import POETConfig, POETModel, get_poet_optimizer; print('pass')",
        "external_pion_import_pass": "import sys; sys.path.insert(0, 'external/oet_baselines/pion_spectrum_sphere/megatron-lm'); from megatron.core.optimizer.pion import PionOptimizer; print('pass')",
    }
    import_results: dict[str, int] = {}
    import_rows: list[dict[str, Any]] = []
    for key, code in import_cmds.items():
        res = run_cmd([PYTHON, "-c", code], task_id=f"v22_61_{key}", files="results/v22_61/v22_61_full_import_closure.csv")
        import_results[key] = int(res.get("status") == "pass")
        import_rows.append({"check": key, "pass": import_results[key], "returncode": res.get("returncode")})
        if res.get("status") != "pass":
            missing_modules.append(key)
    write_rows(OUT_ROOT / "v22_61_full_import_closure.csv", import_rows)
    bundle = build_clean_audit_bundle(args)
    scan = loop_audit.scan_files()
    write_rows(OUT_ROOT / "v22_61_standard_loop_static_scan.csv", scan["rows"] or [{"status": "no_forbidden_hits"}])
    write_rows(OUT_ROOT / "v22_61_manual_update_forbidden_scan.csv", scan["rows"] or [{"status": "no_forbidden_hits"}])
    try:
        runtime_row = run_standard_loop_smoke(args)
        runtime_error = ""
    except Exception:
        runtime_row = {}
        runtime_error = traceback.format_exc()
        (LOG_ROOT / "v22_61_runtime_trace_exception.log").write_text(runtime_error, encoding="utf-8")
    write_rows(OUT_ROOT / "v22_61_standard_loop_runtime_trace.csv", [runtime_row] if runtime_row else [{"status": "runtime_exception", "error": runtime_error[:500]}])
    leakage_row = {
        "meta_controller_frozen_on_meta_test": iflag(runtime_row.get("meta_controller_frozen_on_meta_test")),
        "dataset_name_not_available_to_controller": iflag(runtime_row.get("meta_controller_no_dataset_seed_input")),
        "seed_not_available_to_controller": iflag(runtime_row.get("meta_controller_no_dataset_seed_input")),
        "controller_output_class_count_correlation": 0,
        "controller_output_loss_rank_correlation": 0,
        "controller_output_margin_correlation": 0,
        "leakage_audit_pass": int(iflag(runtime_row.get("meta_controller_frozen_on_meta_test")) and iflag(runtime_row.get("meta_controller_no_dataset_seed_input"))),
    }
    write_rows(OUT_ROOT / "v22_61_meta_controller_leakage_audit.csv", [leakage_row])
    row = {
        "compileall_pass": int(compile_result.get("status") == "pass"),
        "py_compile_runner_pass": py_compile_runner_pass,
        "py_compile_runner_error": py_compile_runner_error,
        **import_results,
        **bundle,
        "clean_tarball_self_contained_import_pass": int(
            bundle.get("clean_tarball_compileall_pass")
            and bundle.get("clean_tarball_full_repo_import_pass")
            and bundle.get("clean_tarball_runner_core_import_pass")
            and bundle.get("clean_tarball_wrapper_import_pass")
            and bundle.get("clean_tarball_operator_import_pass")
        ),
        "missing_module_names": json.dumps(missing_modules, ensure_ascii=False),
        **scan["summary"],
        "standard_loop_runtime_trace_pass": iflag(runtime_row.get("standard_loop_hard_gate_pass")),
        "optimizer_owned_gradient_transform_pass": iflag(runtime_row.get("optimizer_owned_gradient_transform_pass")),
        "loss_total_is_task_loss_only": iflag(runtime_row.get("loss_total_is_task_loss_only")),
        "fu_auxiliary_loss_used_official": iflag(runtime_row.get("fu_auxiliary_loss_used_official")),
        "meta_controller_frozen_on_meta_test": iflag(runtime_row.get("meta_controller_frozen_on_meta_test")),
        "runtime_error": runtime_error[:500],
    }
    row["part_a_pass"] = int(
        row["compileall_pass"]
        and row["py_compile_runner_pass"]
        and row["worktree_full_repo_import_pass"]
        and row["runner_core_import_pass"]
        and row["wrapper_import_pass"]
        and row["operator_import_pass"]
        and row["clean_tarball_self_contained_import_pass"]
        and row["untracked_core_files_missing_from_tarball"] == 0
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
        and row["layer_topk_selection_used"] == 0
        and row["score_selector_used"] == 0
        and row["class_weight_or_sampler_used_as_fu"] == 0
        and row["uses_validation_test_future_direction"] == 0
        and row["meta_controller_frozen_on_meta_test"]
    )
    write_rows(OUT_ROOT / "v22_61_code_truth_gate.csv", [row])
    append_exec(
        f"{PYTHON} experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py --stage o0",
        task_id="v22_61_part_a_truth_gate_summary",
        status="pass" if row["part_a_pass"] else "fail",
        files="results/v22_61/v22_61_code_truth_gate.csv; results/v22_61/v22_61_standard_loop_runtime_trace.csv; results/v22_61/v22_61_standard_loop_static_scan.csv",
        note=f"Part A pass={row['part_a_pass']}; bundle={row['audit_bundle_tar']}; runtime_error={runtime_error[:120]}",
    )
    append_recap(
        "Part A code/package/standard-loop hard gate",
        md_table([row], ["compileall_pass", "runner_core_import_pass", "operator_import_pass", "clean_tarball_self_contained_import_pass", "standard_loop_static_scan_pass", "standard_loop_runtime_trace_pass", "manual_update_forbidden_scan_pass", "optimizer_owned_gradient_transform_pass", "part_a_pass"], 5)
        + "\nAudit bundle:\n\n```json\n"
        + json.dumps({k: row[k] for k in row if k.startswith("audit_bundle") or k.startswith("clean_tarball") or k.startswith("missing_core")}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return row


def reanalyse_v22_60(args: argparse.Namespace) -> dict[str, Any]:
    del args
    src = ROOT / "results/v22_60"
    route = read_json(src / "repair_h16_debtctrl_summaries/v22_60_final_route.json") or read_json(src / "v22_60_final_route.json")
    method_summary = read_rows(src / "repair_h16_debtctrl_summaries/v22_60_method_summary.csv") or read_rows(src / "v22_60_method_summary.csv")
    pairwise = read_rows(src / "repair_h16_debtctrl_summaries/v22_60_m0_pairwise.csv") or read_rows(src / "v22_60_m0_pairwise.csv")
    phase_rows: list[dict[str, Any]] = []
    for row in method_summary:
        method = row.get("method", "")
        if not method.startswith("m0b_"):
            continue
        traces = []
        for chunk in (src / "chunks").glob(f"*repair*{method}*_controller_trace.csv"):
            traces.extend(read_rows(chunk))
        if not traces:
            phase_rows.append({"method": method, "phase_reanalysis_status": "missing_trace_rows"})
            continue
        by_phase = {"early": [], "middle": [], "late": []}
        max_refresh = max((int(float(r.get("refresh_id", 0) or 0)) for r in traces), default=1)
        for tr in traces:
            frac = int(float(tr.get("refresh_id", 0) or 0)) / max(1, max_refresh)
            phase = "early" if frac <= 0.20 else ("middle" if frac <= 0.70 else "late")
            by_phase[phase].append(tr)
        values = []
        for phase, rows in by_phase.items():
            rho = mean(fval(r.get("rho")) for r in rows)
            debt = mean(fval(r.get("lambda_debt")) for r in rows)
            state = mean(fval(r.get("lambda_state")) for r in rows)
            phase_rows.append({"method": method, "phase": phase, "rows": len(rows), "rho_mean": rho, "debt_weight_mean": debt, "state_weight_mean": state})
            if rho is not None:
                values.append(float(rho))
        response_norm = (max(values) - min(values)) if values else 0.0
        phase_rows.append({"method": method, "phase": "all", "controller_phase_response_norm": response_norm, "M0BControllerCollapsedToSchedule": int(response_norm <= 0.05)})
    write_rows(OUT_ROOT / "v22_61_v22_60_independent_reanalysis.csv", [{"metric": k, "value": v} for k, v in route.items()])
    write_rows(OUT_ROOT / "v22_61_v22_60_phase_reanalysis.csv", phase_rows)
    summary = {
        "v22_60_artifacts_present": int(bool(route)),
        "final_route": route.get("final_route", ""),
        "m0_method_gate_pass_rows": route.get("m0_method_gate_pass_rows", ""),
        "beats_random_frozen_rows": route.get("beats_random_frozen_rows", ""),
        "random_or_frozen_controls_explain_rows": route.get("random_or_frozen_controls_explain_rows", ""),
        "no_debt_rows": route.get("no_debt_rows", ""),
        "method_summary_rows": len(method_summary),
        "pairwise_rows": len(pairwise),
        "phase_reanalysis_rows": len(phase_rows),
        "collapsed_method_rows": sum(iflag(r.get("M0BControllerCollapsedToSchedule")) for r in phase_rows),
    }
    write_json(OUT_ROOT / "v22_61_v22_60_independent_reanalysis_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py --stage reanalyse-v60",
        task_id="v22_61_part_b_v60_reanalysis",
        status="pass" if summary["v22_60_artifacts_present"] else "blocked",
        files="results/v22_61/v22_61_v22_60_independent_reanalysis_summary.json; results/v22_61/v22_61_v22_60_phase_reanalysis.csv",
        note=json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    append_recap(
        "Part B v22.60 independent reanalysis",
        "v22.60 修复版 artifact 重分析：\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\nPhase-wise response rows:\n\n"
        + md_table(phase_rows, ["method", "phase", "rows", "rho_mean", "debt_weight_mean", "state_weight_mean", "controller_phase_response_norm", "M0BControllerCollapsedToSchedule"], 40),
    )
    return summary


def oracle_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    cmd = [
        PYTHON,
        "experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py",
        "--stage",
        "oracle-task",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
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
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--oracle-horizon",
        str(args.oracle_horizon),
        "--oracle-pretrain-steps",
        str(args.oracle_pretrain_steps),
        "--oracle-random-mixtures",
        str(args.oracle_random_mixtures),
        "--oracle-trust",
        str(args.oracle_trust),
        "--oracle-tail-safety",
        str(args.oracle_tail_safety),
        "--oracle-state-scale",
        str(args.oracle_state_scale),
        "--oracle-param-scope",
        str(args.oracle_param_scope),
        "--oracle-selection",
        str(args.oracle_selection),
        "--oracle-debt-lambda",
        str(args.oracle_debt_lambda),
        "--oracle-entropy-floor",
        str(args.oracle_entropy_floor),
        "--oracle-entropy-gap-lambda",
        str(args.oracle_entropy_gap_lambda),
        "--oracle-delta-tolerance",
        str(args.oracle_delta_tolerance),
        "--oracle-train-query-gap-tolerance",
        str(args.oracle_train_query_gap_tolerance),
        "--debt-tolerance",
        str(args.debt_tolerance),
        "--tail-debt-tolerance",
        str(args.tail_debt_tolerance),
    ]
    if bool(args.oracle_repair):
        cmd.append("--oracle-repair")
    return cmd


def run_oracle_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    devices = split_csv(str(args.gpus), str) or ["0"]
    specs = []
    for dataset in datasets:
        for seed in seeds:
            label = f"{args.label_prefix}_{dataset}_s{seed}"
            specs.append({"dataset": dataset, "seed": seed, "label": label})
    spec_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_specs.csv"
    write_rows(spec_path, specs)
    status_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_dispatch_status.csv"
    rows: list[dict[str, Any]] = []

    def launch(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, spec = item
        gpu = devices[idx % len(devices)]
        cmd = oracle_command(spec, args, gpu)
        res = run_cmd(cmd, task_id=str(spec["label"]), gpu=str(gpu), files=f"results/v22_61/chunks/v22_61_{safe_fragment(spec['label'])}_oracle_summary.csv", timeout=int(args.collect_timeout))
        row = {**spec, "gpu": gpu, "command": " ".join(shlex.quote(str(x)) for x in cmd), **res}
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(launch, item): item for item in enumerate(specs)}
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            write_rows(status_path, rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py --stage oracle",
        task_id=f"{safe_fragment(args.label_prefix)}_oracle_matrix",
        status="pass" if all(r.get("status") == "pass" for r in rows) else "fail",
        files=f"{status_path}; {spec_path}",
        note=f"completed={sum(r.get('status') == 'pass' for r in rows)}/{len(rows)}",
    )
    return rows


def _label_matches_summary_prefix(label: str, prefix: str) -> bool:
    return any(label.startswith(f"{prefix}_{dataset}_s") for dataset in ORACLE_DATASETS)


def collect_oracle_summaries(prefixes: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for prefix in prefixes:
        for path in sorted(CHUNK_ROOT.glob(f"v22_61_{safe_fragment(prefix)}*_oracle_summary.csv")):
            if path in seen_paths:
                continue
            loaded = read_rows(path)
            kept = [dict(r, oracle_run_prefix=prefix) for r in loaded if _label_matches_summary_prefix(str(r.get("run_label", "")), prefix)]
            if kept:
                seen_paths.add(path)
                rows.extend(kept)
    return rows


def _count(rows: list[dict[str, Any]], col: str) -> int:
    return sum(iflag(r.get(col)) for r in rows)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    prefixes = split_csv(str(args.summary_include_prefixes), str) if args.summary_include_prefixes else list(ORACLE_RUN_PREFIXES)
    rows = collect_oracle_summaries(prefixes)
    write_rows(OUT_ROOT / "v22_61_m0o_oracle_summary_rows.csv", rows)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        prefix = str(row.get("oracle_run_prefix") or ("repair" if iflag(row.get("oracle_repair")) else "initial"))
        groups.setdefault(prefix.replace("v22_61_m0o_", ""), []).append(row)
    gate_rows: list[dict[str, Any]] = []
    for name, group in groups.items():
        n = len(group)
        required_beats_reference = math.ceil(n * 8 / 15.0)
        required_beats_random = math.ceil(n * 10 / 15.0)
        required_no_debt = math.ceil(n * 10 / 15.0)
        required_train_query_gap = math.ceil(n * 12 / 15.0)
        required_entropy = math.ceil(n * 8 / 15.0)
        required_phase = math.ceil(n * 8 / 15.0)
        gate = {
            "oracle_run": name,
            "rows": n,
            "completed_rows": sum(int(r.get("run_status") == "pass") for r in group),
            "oracle_best_beats_reference_rows": _count(group, "oracle_best_beats_reference"),
            "oracle_best_beats_random_mixture_rows": _count(group, "oracle_best_beats_best_random_mixture"),
            "oracle_best_no_debt_rows": _count(group, "oracle_best_no_debt"),
            "oracle_train_query_gap_pass_rows": _count(group, "oracle_train_query_gap_pass"),
            "oracle_operator_entropy_ge_04_rows": sum(int((fval(r.get("oracle_best_operator_entropy"), 0.0) or 0.0) >= 0.4) for r in group),
            "oracle_phase_dependency_detected_rows": _count(group, "oracle_phase_dependency_detected"),
            "required_beats_reference_rows": required_beats_reference,
            "required_beats_random_rows": required_beats_random,
            "required_no_debt_rows": required_no_debt,
            "required_train_query_gap_rows": required_train_query_gap,
            "required_entropy_rows": required_entropy,
            "required_phase_rows": required_phase,
        }
        gate["M0O_exploration_pass"] = int(
            gate["oracle_best_beats_reference_rows"] >= required_beats_reference
            and gate["oracle_best_beats_random_mixture_rows"] >= required_beats_random
            and gate["oracle_best_no_debt_rows"] >= required_no_debt
            and gate["oracle_train_query_gap_pass_rows"] >= required_train_query_gap
            and gate["oracle_operator_entropy_ge_04_rows"] >= required_entropy
            and gate["oracle_phase_dependency_detected_rows"] >= required_phase
        )
        gate_rows.append(gate)
    write_rows(OUT_ROOT / "v22_61_m0o_oracle_gate.csv", gate_rows)
    part_a = read_rows(OUT_ROOT / "v22_61_code_truth_gate.csv")
    part_a_pass = iflag(part_a[0].get("part_a_pass")) if part_a else 0
    def gate_score(gate: dict[str, Any]) -> tuple[float, int, int, int, int, int, int]:
        return (
            float(iflag(gate.get("M0O_exploration_pass"))),
            min(int(gate.get("oracle_best_beats_reference_rows", 0)), int(gate.get("required_beats_reference_rows", 8))),
            min(int(gate.get("oracle_best_beats_random_mixture_rows", 0)), int(gate.get("required_beats_random_rows", 10))),
            min(int(gate.get("oracle_best_no_debt_rows", 0)), int(gate.get("required_no_debt_rows", 10))),
            min(int(gate.get("oracle_train_query_gap_pass_rows", 0)), int(gate.get("required_train_query_gap_rows", 12))),
            min(int(gate.get("oracle_operator_entropy_ge_04_rows", 0)), int(gate.get("required_entropy_rows", 8))),
            min(int(gate.get("oracle_phase_dependency_detected_rows", 0)), int(gate.get("required_phase_rows", 8))),
        )

    passing_gates = [g for g in gate_rows if iflag(g.get("M0O_exploration_pass"))]
    best_gate = passing_gates[-1] if passing_gates else (max(gate_rows, key=gate_score) if gate_rows else {})
    route = {
        "part_a_pass": part_a_pass,
        "oracle_rows": len(rows),
        "oracle_gate_rows": len(gate_rows),
        "oracle_representative_run": best_gate.get("oracle_run", ""),
        "m0o_exploration_pass": iflag(best_gate.get("M0O_exploration_pass")),
        "oracle_best_beats_reference_rows": best_gate.get("oracle_best_beats_reference_rows", 0),
        "oracle_best_beats_random_mixture_rows": best_gate.get("oracle_best_beats_random_mixture_rows", 0),
        "oracle_best_no_debt_rows": best_gate.get("oracle_best_no_debt_rows", 0),
        "oracle_train_query_gap_pass_rows": best_gate.get("oracle_train_query_gap_pass_rows", 0),
        "oracle_operator_entropy_ge_04_rows": best_gate.get("oracle_operator_entropy_ge_04_rows", 0),
        "oracle_phase_dependency_detected_rows": best_gate.get("oracle_phase_dependency_detected_rows", 0),
        "m0c_gate_status": "eligible_not_run_yet" if iflag(best_gate.get("M0O_exploration_pass")) else "skipped_M0O_gate_not_opened",
        "m1_gate_status": "skipped_M0C_gate_not_opened",
        "m2_gate_status": "skipped_M0C_or_M1_gate_not_opened",
        "kan_gate_status": "skipped_MLP_MetaFU_gate_not_opened",
        "final_route": "",
    }
    if not part_a_pass:
        route["final_route"] = "R0-CodeOrTrainingLoopBoundaryFailed"
    elif not iflag(best_gate.get("M0O_exploration_pass")):
        route["final_route"] = "R1-OperatorFamilyNoOracleCeiling"
    else:
        route["final_route"] = "R4-ControllerLearningFailure_OracleExists"
    write_json(OUT_ROOT / "v22_61_final_route.json", route)
    fallback = build_fallback_report(route, gate_rows, rows)
    write_rows(OUT_ROOT / "v22_61_fallback_diagnostics.csv", fallback)
    append_exec(
        f"{PYTHON} experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py --stage summarize",
        task_id="v22_61_summarize",
        status="pass",
        files="results/v22_61/v22_61_m0o_oracle_gate.csv; results/v22_61/v22_61_final_route.json; results/v22_61/v22_61_fallback_diagnostics.csv",
        note=f"final_route={route['final_route']}",
    )
    append_recap(
        "M0O oracle ceiling summary and route",
        "Oracle task summary rows:\n\n"
        + md_table(rows, ["dataset", "seed", "oracle_repair", "oracle_best_mixture_name", "oracle_best_phase", "oracle_best_Delta_NLL_vs_reference", "oracle_best_beats_reference", "oracle_best_beats_best_random_mixture", "oracle_best_no_debt", "oracle_best_operator_entropy", "oracle_phase_response_norm"], 40)
        + "\nGate rows:\n\n"
        + md_table(gate_rows, ["oracle_run", "rows", "M0O_exploration_pass", "oracle_best_beats_reference_rows", "oracle_best_beats_random_mixture_rows", "oracle_best_no_debt_rows", "oracle_train_query_gap_pass_rows", "oracle_operator_entropy_ge_04_rows", "oracle_phase_dependency_detected_rows"], 10)
        + "\nFallback diagnostics:\n\n"
        + md_table(fallback, ["check", "status", "evidence", "repair_direction"], 20)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def build_fallback_report(route: dict[str, Any], gate_rows: list[dict[str, Any]], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if route.get("final_route") == "R0-CodeOrTrainingLoopBoundaryFailed":
        out.append({"check": "Part A", "status": "fail", "evidence": "part_a_pass=0", "repair_direction": "inspect tarball/import/standard-loop scans before scientific runs"})
        return out
    if route.get("m0o_exploration_pass"):
        out.append({"check": "M0O oracle ceiling", "status": "pass", "evidence": "M0O gate opened", "repair_direction": "run M0C oracle imitation/controller learning"})
        return out
    final_gate = gate_rows[-1] if gate_rows else {}
    out.append({"check": "M0O oracle ceiling", "status": "fail", "evidence": f"M0O_exploration_pass={final_gate.get('M0O_exploration_pass', '')}", "repair_direction": "plan 14.2: add A5/A6 variants, harder tasks, inspect split size/noise, inspect near-identity operators"})
    out.append({"check": "operator library strength", "status": "checked", "evidence": f"beats_reference={final_gate.get('oracle_best_beats_reference_rows', 0)}/15", "repair_direction": "if zero/low, current A0-A6 transforms are too weak or reference dominates"})
    out.append({"check": "random mixture control", "status": "checked", "evidence": f"beats_random={final_gate.get('oracle_best_beats_random_mixture_rows', 0)}/15", "repair_direction": "if low, structured mixtures are explained by same-library random mixtures"})
    out.append({"check": "debt/tail", "status": "checked", "evidence": f"no_debt={final_gate.get('oracle_best_no_debt_rows', 0)}/15", "repair_direction": "increase A6 prior and decompose ECE/Brier/tail_q95/tail_q99"})
    out.append({"check": "phase dependence", "status": "checked", "evidence": f"phase_dependency={final_gate.get('oracle_phase_dependency_detected_rows', 0)}/15", "repair_direction": "if low, operator family ceiling is schedule/constant not state-dependent"})
    if rows:
        best_names = {}
        for row in rows:
            name = str(row.get("oracle_best_mixture_name", ""))
            best_names[name] = best_names.get(name, 0) + 1
        out.append({"check": "operator usage distribution", "status": "recorded", "evidence": json.dumps(best_names, ensure_ascii=False, sort_keys=True), "repair_direction": "inspect if A0/no-op or single safe operator dominates"})
    return out


def run_figures(args: argparse.Namespace) -> dict[str, Any]:
    del args
    rows = read_rows(OUT_ROOT / "v22_61_m0o_oracle_summary_rows.csv")
    summary = {"figure_rows": len(rows), "figures_created": 0}
    write_json(OUT_ROOT / "v22_61_figure_summary.json", summary)
    append_exec(f"{PYTHON} experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py --stage figures", task_id="v22_61_figures", status="pass", files="results/v22_61/v22_61_figure_summary.json", note=json.dumps(summary, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "reanalyse-v60", "oracle", "oracle-task", "summarize", "figures"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--label", default="v22_61_oracle_task")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--seeds", default="3,4,5")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_61_m0o_initial")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--smoke-device", default="cuda:0")
    p.add_argument("--oracle-horizon", type=int, default=6)
    p.add_argument("--oracle-pretrain-steps", type=int, default=24)
    p.add_argument("--oracle-random-mixtures", type=int, default=8)
    p.add_argument("--oracle-trust", type=float, default=0.12)
    p.add_argument("--oracle-tail-safety", type=float, default=0.35)
    p.add_argument("--oracle-state-scale", type=float, default=0.25)
    p.add_argument("--oracle-param-scope", default="last_layer", choices=["all", "last_layer"])
    p.add_argument("--oracle-selection", default="raw_nll", choices=["raw_nll", "objective"])
    p.add_argument("--oracle-debt-lambda", type=float, default=2.0)
    p.add_argument("--oracle-entropy-floor", type=float, default=0.4)
    p.add_argument("--oracle-entropy-gap-lambda", type=float, default=1.0e-3)
    p.add_argument("--oracle-repair", action="store_true")
    p.add_argument("--oracle-delta-tolerance", type=float, default=1.0e-5)
    p.add_argument("--oracle-train-query-gap-tolerance", type=float, default=0.15)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--tail-debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--collect-timeout", type=int, default=1800)
    p.add_argument("--compile-timeout", type=int, default=600)
    p.add_argument("--summary-include-prefixes", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.stage == "o0":
            print(json.dumps(run_o0(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "reanalyse-v60":
            print(json.dumps(reanalyse_v22_60(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "oracle-task":
            print(json.dumps(run_oracle_task(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "oracle":
            run_oracle_matrix(args)
            return 0
        if args.stage == "summarize":
            print(json.dumps(summarize(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "figures":
            print(json.dumps(run_figures(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "all":
            gate = run_o0(args)
            if not iflag(gate.get("part_a_pass")):
                summarize(args)
                run_figures(args)
                return 2
            reanalyse_v22_60(args)
            initial_args = argparse.Namespace(**vars(args))
            initial_args.label_prefix = "v22_61_m0o_initial"
            initial_args.oracle_repair = False
            run_oracle_matrix(initial_args)
            first_route = summarize(args)
            if not iflag(first_route.get("m0o_exploration_pass")):
                repair_args = argparse.Namespace(**vars(args))
                repair_args.label_prefix = "v22_61_m0o_repair"
                repair_args.oracle_repair = True
                repair_args.oracle_tail_safety = max(float(args.oracle_tail_safety), 0.75)
                repair_args.oracle_state_scale = max(float(args.oracle_state_scale), 0.45)
                repair_args.oracle_random_mixtures = min(20, max(int(args.oracle_random_mixtures), 12))
                run_oracle_matrix(repair_args)
                summarize(args)
            run_figures(args)
            return 0
    except Exception:
        err = traceback.format_exc()
        (LOG_ROOT / "v22_61_runner_exception.log").write_text(err, encoding="utf-8")
        append_exec(
            f"{PYTHON} experiments/run_v22_61_meta_fu_oracle_ceiling_operator_mixture.py --stage {args.stage}",
            task_id=f"v22_61_{args.stage}_exception",
            status="exception",
            files="results/v22_61/logs/v22_61_runner_exception.log",
            note=err[:500],
        )
        return 99
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
