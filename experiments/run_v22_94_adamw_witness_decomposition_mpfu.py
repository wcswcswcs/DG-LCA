#!/usr/bin/env python3
"""DG-KAN v22.94 AdamW witness decomposition and fixed-scheme MPFU runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import re
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
from dgkan.fu.layer_composite_metric import (
    EPS,
    composite_metric,
    condition_number,
    effective_rank,
    matrix_to_w1,
    metric_matrix,
    relative_fro_error,
    sym,
    w1_to_matrix,
)


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_94_adamw_witness_decomposition_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.94_AdamW_Witness_Decomposition_MultiScheme_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.94_AdamW_Witness_Decomposition_MultiScheme_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.94_AdamW_Witness_Decomposition_MultiScheme_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2294_OUT_ROOT", str(ROOT / "results/v22_94"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"
CKPT_ROOT = OUT_ROOT / "checkpoints"

AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "candidate_runtime_selection_used": 0,
    "winner_metric_selection_used": 0,
    "winner_update_selection_used": 0,
    "mlp_readout_used": 0,
    "mlp_initial_layer_used": 0,
    "new_edge_function_added": 0,
    "external_product_feature_used": 0,
    "changed_w1": 0,
    "changed_w2": 0,
    "changed_mlp": 0,
    "dche_enabled": 0,
    "dfour_enabled": 0,
    "basis_family": "",
}

VISUAL_TASKS = ["stroke_hv", "local_patch_interaction", "two_patch_composition", "rotation_sensitive"]
BASIS_KEYS = ["dche_k5", "dche_k9", "dfour_default"]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, CKPT_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def command_text(items: Iterable[Any]) -> str:
    return " ".join(str(item) for item in items)


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    clean: list[float] = []
    for value in values:
        out = fval(value, float("nan"))
        if math.isfinite(out):
            clean.append(out)
    clean.sort()
    if not clean:
        return float(default)
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    clean = [fval(v, float("nan")) for v in values]
    clean = [v for v in clean if math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else float(default)


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for item_idx, item in enumerate(items) if item_idx % count == idx]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**AUDIT_DEFAULTS, **data}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys = sorted({key for row in enriched for key in row.keys()} or set(AUDIT_DEFAULTS.keys()))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.94 AdamW Witness Decomposition MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- Python：`{PYTHON}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 非编造约束：只记录真实命令、真实 artifact、真实错误与观测；缺失项写 missing/unavailable/skipped。\n"
            "- 复现提示：Part C/E 支持 `--shard-count/--shard-index`，建议用 `CUDA_VISIBLE_DEVICES=0..3` 并行。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.94 AdamW Witness Decomposition MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录真实 artifact、真实指标、实际修复与分析；不补造缺失数据。\n\n"
            "## 1. 证据链、修复与分析\n",
            encoding="utf-8",
        )


def append_exec(stage: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {stage} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, bullets: list[str]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for item in bullets:
            fh.write(f"- {item}\n")


def write_next_actions(part: str, route: str, blocker: str, actions: list[dict[str, Any]], evidence: list[str] | None = None) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part.upper(),
            "route": route,
            "dominant_blocker": blocker,
            "evidence_files": evidence or [],
            "allowed_actions": actions,
            "forbidden_actions": [
                "do not add a new edge function",
                "do not add MLP stem/readout to PureKAN",
                "do not weaken C2 gate",
                "do not delete MLP controls",
                "do not use held/test data for training or response fitting",
                "do not promote single-seed probe",
                "do not choose basis family winner at runtime",
                "do not choose scheme winner per row/seed/dataset",
                "do not reinterpret skipped Part E/F/G as success",
            ],
            "max_repair_rounds": 2,
            "rerun_commands": [],
            "stop_condition": "rerun required gate and compare only real artifacts",
        },
    )


def basis_args(args: argparse.Namespace, basis_key: str) -> argparse.Namespace:
    out = copy(args)
    if basis_key == "dche_k5":
        out.pc_basis = "D-CHE"
        out.basis_k_override = 5
    elif basis_key == "dche_k9":
        out.pc_basis = "D-CHE"
        out.basis_k_override = 9
    elif basis_key == "dfour_default":
        out.pc_basis = "D-FOU"
        out.basis_k_override = int(args.dfour_k)
    else:
        raise ValueError(f"unknown basis key {basis_key}")
    return out


def basis_audit_fields(basis_key: str, model: Any | None = None) -> dict[str, Any]:
    basis_family = basis_key
    if model is not None:
        basis_family = f"{basis_key}:{getattr(model, 'basis_name', '')}:K{getattr(model, 'k', '')}"
    return {
        "basis_family": basis_family,
        "dche_enabled": int(basis_key.startswith("dche")),
        "dfour_enabled": int(basis_key.startswith("dfour")),
    }


def scheme_base_name(scheme: str) -> str:
    out = str(scheme)
    for marker in ["_tailsafe_w025", "_tailsafe_w05", "_tailsafe_w1", "_cal_w025", "_cal_w05", "_cal_w1", "_calibrated", "_tailsafe"]:
        out = out.replace(marker, "")
    return out


def scheme_calibration_weight(scheme: str, args: argparse.Namespace) -> float:
    text = str(scheme)
    if "_cal_w025" in text or "_tailsafe_w025" in text:
        return 0.25
    if "_cal_w05" in text or "_tailsafe_w05" in text:
        return 0.50
    if "_cal_w1" in text or "_tailsafe_w1" in text:
        return 1.00
    if "_calibrated" in text or "_tailsafe" in text:
        return float(args.scheme_calibration_weight)
    return 0.0


def scheme_tail_margin_weights(scheme: str, args: argparse.Namespace) -> tuple[float, float]:
    if "_tailsafe" in str(scheme):
        return float(args.scheme_calibration_tail_weight), float(args.scheme_calibration_margin_weight)
    return 0.0, 0.0


def tail_margin_surrogate_loss(logits: torch.Tensor, y: torch.Tensor, baseline: dict[str, float], budget: float, tail_weight: float, margin_weight: float) -> torch.Tensor:
    if float(tail_weight) <= 0.0 and float(margin_weight) <= 0.0:
        return logits.float().sum() * 0.0
    prob = F.softmax(logits.float(), dim=1)
    true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1)
    soft_tail = torch.logsumexp(10.0 * (1.0 - true_prob), dim=0) / 10.0 - math.log(max(1, int(y.numel()))) / 10.0
    if int(logits.shape[1]) > 1:
        true_logit = logits.float().gather(1, y.long().reshape(-1, 1)).reshape(-1)
        other_logits = logits.float().clone()
        other_logits.scatter_(1, y.long().reshape(-1, 1), float("-inf"))
        margin = true_logit - other_logits.max(dim=1).values
    else:
        margin = logits.float().reshape(-1)
    k_low = max(1, int(math.ceil(0.10 * int(margin.numel()))))
    low_margin = torch.topk(margin, k=k_low, largest=False).values.mean()
    tail_budget = float(baseline["tail95"]) + float(budget)
    margin_floor = float(baseline["margin10"]) - float(budget)
    return float(tail_weight) * F.relu(soft_tail - tail_budget).square() + float(margin_weight) * F.relu(margin_floor - low_margin).square()


def static_scan(files: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_architecture_winner_selection": re.compile(r"\b(select_arch|choose_arch|best_family|winner_family)\s*\("),
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|metric_winner|winner_metric)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|candidate_score|winner_update)\s*\("),
        "runtime_topk_edge_selection": re.compile(r"\.topk\s*\("),
        "external_product_feature": re.compile(r"\b(make_external_product_feature|external_product_feature_transform|patch_product_feature)\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for check, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                hits.append({"file": rel(path), "check": check, "match": match.group(0)})
    return int(not hits), hits


def visual_data(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2293.visual_synthetic_task(
        task,
        seed,
        int(args.synthetic_train_size),
        int(args.synthetic_guard_size),
        int(args.visual_side),
        int(args.num_classes),
        device,
        fixed_patch_features=bool(int(args.visual_fixed_patch_features)),
        task_version=str(args.visual_task_version),
    )


def synthetic_data(teacher: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2293.synthetic_task(
        teacher,
        seed,
        int(args.synthetic_train_size),
        int(args.synthetic_guard_size),
        int(args.input_dim),
        int(args.num_classes),
        device,
    )


def make_model(depth: str, input_dim: int, classes: int, model_seed: int, args: argparse.Namespace, device: torch.device) -> v2293.TrueDeepPureKAN:
    return v2293.make_kan_family(depth, input_dim, classes, model_seed, args, device)


def model_seed_for(basis_key: str, depth: str, task: str, seed: int) -> int:
    basis_idx = BASIS_KEYS.index(basis_key) if basis_key in BASIS_KEYS else 9
    depth_idx = {"shallow": 1, "depth2": 2, "depth3": 3}.get(depth, 1)
    task_idx = VISUAL_TASKS.index(task) if task in VISUAL_TASKS else 17
    return 2294000 + basis_idx * 100000 + depth_idx * 10000 + task_idx * 1000 + int(seed)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(dtype=torch.float64).numpy()


def param_mats(model: v2293.TrueDeepPureKAN) -> list[torch.Tensor]:
    return [w1_to_matrix(p.detach()).to(device=p.device, dtype=torch.float64) for p in model.coeffs]


def grad_mats(model: v2293.TrueDeepPureKAN) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for p in model.coeffs:
        if p.grad is None:
            out.append(torch.zeros_like(w1_to_matrix(p.detach()).to(dtype=torch.float64)))
        else:
            out.append(w1_to_matrix(p.grad.detach()).to(device=p.device, dtype=torch.float64))
    return out


def set_model_mats(model: v2293.TrueDeepPureKAN, mats: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, mat in zip(model.coeffs, mats):
            param.copy_(matrix_to_w1(mat, param.shape, dtype=param.dtype, device=param.device))


def flatten_mats(mats: list[torch.Tensor]) -> torch.Tensor:
    if not mats:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat([m.detach().reshape(-1).to(dtype=torch.float64).cpu() for m in mats])


def cosine_mats(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    va, vb = flatten_mats(a), flatten_mats(b)
    if va.numel() == 0 or vb.numel() == 0:
        return 0.0
    denom = va.norm() * vb.norm()
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((va @ vb / denom).item())


def mode_metric_diag(model: v2293.TrueDeepPureKAN, layer_idx: int, args: argparse.Namespace) -> torch.Tensor:
    k = int(model.k)
    if model.basis_name == "chebyshev":
        vals = [
            1.0 + float(args.mode_lambda_partial) * (m**2) + float(args.mode_lambda_partial2) * (m**4)
            for m in range(k)
        ]
    else:
        vals = [1.0]
        freq = 1
        while len(vals) < k:
            vals.append(1.0 + float(args.mode_lambda_omega) * (freq**2))
            if len(vals) < k:
                vals.append(1.0 + float(args.mode_lambda_omega) * (freq**2))
            freq += 1
    return torch.tensor(vals[:k], device=model.coeffs[layer_idx].device, dtype=torch.float64).repeat(int(model.dims[layer_idx]))


def op_spectrum(a: torch.Tensor, mode_diag: torch.Tensor) -> torch.Tensor:
    d = mode_diag.to(device=a.device, dtype=torch.float64).reshape(-1, 1)
    return sym(a.T @ (a * d))


def data_c_matrix(model: v2293.TrueDeepPureKAN, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    with torch.no_grad():
        _logits, activations = model.forward_with_activations(x)
        phi = model.layer_phi(activations[layer_idx].detach(), layer_idx)
        c = phi.T @ phi / float(max(1, int(phi.shape[0])))
        c = c + EPS * torch.eye(int(c.shape[1]), device=x.device, dtype=torch.float64)
    return sym(c)


def matrix_summary(mat: torch.Tensor) -> dict[str, float]:
    eig = torch.linalg.eigvalsh(sym(mat).to(dtype=torch.float64)).detach()
    eig_abs = eig.abs().clamp_min(EPS)
    top = torch.sort(eig_abs, descending=True).values
    return {
        "trace": float(torch.trace(mat).detach().cpu().item()),
        "fro": float(mat.norm().detach().cpu().item()),
        "condition": condition_number(mat),
        "top1": float(top[0].cpu().item()) if int(top.numel()) else 0.0,
        "top3_sum": float(top[: min(3, int(top.numel()))].sum().cpu().item()) if int(top.numel()) else 0.0,
        "rank_eff": effective_rank(mat),
    }


def log_spectral_drift(before: torch.Tensor, after: torch.Tensor) -> float:
    def eig_abs(mat: torch.Tensor) -> torch.Tensor | None:
        m = sym(mat).to(dtype=torch.float64)
        if not bool(torch.isfinite(m).all().detach().cpu().item()):
            return None
        try:
            return torch.linalg.eigvalsh(m).abs().clamp_min(EPS)
        except Exception:
            eye = torch.eye(int(m.shape[0]), device=m.device, dtype=m.dtype)
            scale = float(m.detach().abs().max().clamp_min(1.0).cpu().item())
            try:
                return torch.linalg.eigvalsh(m + (1.0e-8 * scale) * eye).abs().clamp_min(EPS)
            except Exception:
                try:
                    return torch.linalg.svdvals(m).abs().clamp_min(EPS)
                except Exception:
                    return None

    eb = eig_abs(before)
    ea = eig_abs(after)
    if eb is None or ea is None:
        return float("inf")
    return float((ea.log() - eb.log()).norm().div(math.sqrt(max(1, int(eb.numel())))).detach().cpu().item())


def activation_entropy(x: torch.Tensor) -> float:
    flat = x.detach().to(dtype=torch.float64).reshape(-1)
    if int(flat.numel()) == 0:
        return 0.0
    hist = torch.histc(flat.float(), bins=32, min=float(flat.min().cpu().item()), max=float(flat.max().cpu().item()))
    p = hist.to(dtype=torch.float64)
    p = p / p.sum().clamp_min(EPS)
    return float((-(p * (p + EPS).log()).sum()).cpu().item())


def safe_activation_rank(x: torch.Tensor) -> float:
    if not bool(torch.isfinite(x.detach()).all().detach().cpu().item()):
        return 0.0
    try:
        out = float(v2293.activation_rank(x))
        return out if math.isfinite(out) else 0.0
    except Exception:
        return 0.0


def activation_summary(x: torch.Tensor) -> dict[str, float]:
    xd = x.detach().to(dtype=torch.float64)
    q = torch.quantile(xd.reshape(-1), torch.tensor([0.1, 0.5, 0.9], device=xd.device, dtype=torch.float64))
    sign_div = 0.0
    if xd.ndim == 2 and int(xd.shape[1]) <= 24:
        signs = (xd > 0).to(torch.int8).detach().cpu().numpy()
        sign_div = float(len({tuple(row.tolist()) for row in signs}) / max(1, int(signs.shape[0])))
    return {
        "mean": float(xd.mean().cpu().item()),
        "variance": float(xd.var(unbiased=False).cpu().item()),
        "entropy": activation_entropy(xd),
        "q10": float(q[0].cpu().item()),
        "q50": float(q[1].cpu().item()),
        "q90": float(q[2].cpu().item()),
        "cov_rank": safe_activation_rank(xd) if xd.ndim == 2 else 0.0,
        "offdiag_mixing": v2293.offdiag_mixing(xd) if xd.ndim == 2 else 0.0,
        "sign_pattern_diversity": sign_div,
    }


def mode_band_energy(param: torch.Tensor, basis_name: str) -> dict[str, float]:
    p = param.detach().to(dtype=torch.float64)
    k = int(p.shape[2])
    energies = [float(p[:, :, m].square().sum().cpu().item()) for m in range(k)]
    out: dict[str, float] = {}
    if basis_name == "chebyshev":
        bands = {
            "degree_0_1": range(0, min(k, 2)),
            "degree_2_3": range(2, min(k, 4)),
            "degree_4_5": range(4, min(k, 6)),
            "degree_6_plus": range(6, k),
        }
    else:
        bands = {
            "constant": range(0, min(k, 1)),
            "low_harmonic": range(1, min(k, 3)),
            "mid_harmonic": range(3, min(k, 5)),
            "high_harmonic": range(5, k),
        }
    for name, idxs in bands.items():
        out[name] = float(sum(energies[i] for i in idxs))
    if basis_name != "chebyshev":
        pairs = []
        for idx in range(1, k, 2):
            pairs.append(energies[idx] + (energies[idx + 1] if idx + 1 < k else 0.0))
        out["phase_pair_energy"] = float(sum(pairs))
    out["total"] = float(sum(energies))
    return out


def checkpoint_paths(run_id: str, step: int) -> tuple[Path, Path]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    return CKPT_ROOT / f"{safe}_step{int(step):03d}.npz", CKPT_ROOT / f"{safe}_step{int(step):03d}.json"


def save_checkpoint(
    run_id: str,
    step: int,
    model: v2293.TrueDeepPureKAN,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    a_before: list[torch.Tensor],
    a_after: list[torch.Tensor],
    grads: list[torch.Tensor],
    deltas: list[torch.Tensor],
    diag: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    json_layers: dict[str, Any] = {}
    with torch.no_grad():
        _logits, activations = model.forward_with_activations(x_train)
    for layer_idx, param in enumerate(model.coeffs):
        a0 = a_before[layer_idx].detach().to(device=param.device, dtype=torch.float64)
        a1 = a_after[layer_idx].detach().to(device=param.device, dtype=torch.float64)
        g = grads[layer_idx].detach().to(device=param.device, dtype=torch.float64)
        d = deltas[layer_idx].detach().to(device=param.device, dtype=torch.float64)
        c_data = data_c_matrix(model, x_train, layer_idx)
        r_data = composite_metric(a1, c_data)
        s_op = op_spectrum(a1, mode_metric_diag(model, layer_idx, args))
        arrays[f"A_before_layer{layer_idx}"] = tensor_to_np(a0)
        arrays[f"A_after_layer{layer_idx}"] = tensor_to_np(a1)
        arrays[f"grad_layer{layer_idx}"] = tensor_to_np(g)
        arrays[f"delta_layer{layer_idx}"] = tensor_to_np(d)
        arrays[f"R_data_layer{layer_idx}"] = tensor_to_np(r_data)
        arrays[f"S_op_layer{layer_idx}"] = tensor_to_np(s_op)
        json_layers[str(layer_idx)] = {
            "A_norm": float(a1.norm().detach().cpu().item()),
            "grad_norm": float(g.norm().detach().cpu().item()),
            "delta_norm": float(d.norm().detach().cpu().item()),
            "data_R": matrix_summary(r_data),
            "op_S": matrix_summary(s_op),
            "mode_band_energy": mode_band_energy(param, model.basis_name),
            "activation": activation_summary(activations[layer_idx + 1].detach()),
        }
    train_metrics = v2293.metrics_for_model(model, x_train, y_train)
    guard_metrics = v2293.metrics_for_model(model, x_guard, y_guard)
    npz_path, json_path = checkpoint_paths(run_id, step)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    meta = {
        "run_id": run_id,
        "step": int(step),
        "train_NLL": train_metrics["nll"],
        "guard_NLL": guard_metrics["nll"],
        "visual_accuracy": guard_metrics["accuracy"],
        "visual_coverage": guard_metrics["coverage_CVaR25"],
        "diag": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in diag.items()},
        "layers": json_layers,
        "npz": rel(npz_path),
    }
    write_json(json_path, meta)
    return npz_path, json_path, meta


def zero_like_mats(model: v2293.TrueDeepPureKAN) -> list[torch.Tensor]:
    return [torch.zeros_like(w1_to_matrix(p.detach()).to(dtype=torch.float64)) for p in model.coeffs]


def train_witness_run(
    basis_key: str,
    depth: str,
    optimizer_kind: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    bargs = basis_args(args, basis_key)
    xtr, ytr, xg, yg = visual_data(task, seed, args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = model_seed_for(basis_key, depth, task, seed)
    model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
    run_id = f"{basis_key}_{depth}_{optimizer_kind}_{task}_seed{seed}"
    initial_train = v2293.metrics_for_model(model, xtr, ytr)
    initial_guard = v2293.metrics_for_model(model, xg, yg)
    save_steps = {int(s) for s in csv_items(args.checkpoint_steps) if int(s) <= int(args.train_steps)}
    ckpt_jsons: list[str] = []
    zero = zero_like_mats(model)
    if 0 in save_steps:
        mats = param_mats(model)
        _npz, js, _meta = save_checkpoint(run_id, 0, model, xtr, ytr, xg, yg, mats, mats, zero, zero, {}, bargs)
        ckpt_jsons.append(rel(js))
    opt: torch.optim.Optimizer | None = None
    if optimizer_kind in {"adamw", "projected_adamw_tube"}:
        lr = float(args.adamw_lr) if optimizer_kind == "adamw" else float(args.metric_projected_adamw_lr)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
    metric_state: dict[int, dict[str, torch.Tensor | int]] = {}
    trace: dict[str, float] = {"velocity_emitted": 0.0, "changed_w1": 0.0, "metric_drift_after_max": 0.0}
    start = time.time()
    for step in range(1, int(args.train_steps) + 1):
        xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
        before = param_mats(model)
        diag: dict[str, Any]
        if optimizer_kind == "metric":
            diag = v2293.projected_metric_step(model, xb, yb, float(args.metric_lr), bargs, metric_state=metric_state, step_idx=step)
        elif optimizer_kind == "projected_adamw_tube":
            assert opt is not None
            diag = v2293.projected_adamw_tube_step(model, opt, xb, yb, float(args.metric_projected_adamw_lr), bargs)
        else:
            assert opt is not None
            model.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb.long())
            loss.backward()
            opt.step()
            diag = {"loss": float(loss.detach().cpu().item()), "changed_w1": 1.0}
        grads = grad_mats(model)
        after = param_mats(model)
        deltas = [a1 - a0 for a0, a1 in zip(before, after)]
        for key, value in diag.items():
            if isinstance(value, (int, float)):
                if key.endswith("_max") or key in {"velocity_emitted", "changed_w1", "metric_drift_after_max"}:
                    trace[key] = max(fval(trace.get(key)), fval(value))
                else:
                    trace[key] = fval(value)
        if step in save_steps:
            _npz, js, _meta = save_checkpoint(run_id, step, model, xtr, ytr, xg, yg, before, after, grads, deltas, diag, bargs)
            ckpt_jsons.append(rel(js))
    elapsed = time.time() - start
    final_train = v2293.metrics_for_model(model, xtr, ytr)
    final_guard = v2293.metrics_for_model(model, xg, yg)
    row = {
        "run_id": run_id,
        "basis_key": basis_key,
        "depth": depth,
        "optimizer_kind": optimizer_kind,
        "visual_synthetic_task": task,
        "visual_task_version": str(args.visual_task_version),
        "seed": seed,
        "model_seed": model_seed,
        "train_steps": int(args.train_steps),
        "checkpoint_steps": ",".join(str(s) for s in sorted(save_steps)),
        "checkpoint_jsons": ";".join(ckpt_jsons),
        "checkpoint_dir": rel(CKPT_ROOT),
        "initial_train_NLL": initial_train["nll"],
        "initial_guard_NLL": initial_guard["nll"],
        "initial_visual_accuracy": initial_guard["accuracy"],
        "initial_visual_coverage": initial_guard["coverage_CVaR25"],
        "final_train_NLL": final_train["nll"],
        "final_guard_NLL": final_guard["nll"],
        "final_visual_accuracy": final_guard["accuracy"],
        "final_visual_coverage": final_guard["coverage_CVaR25"],
        "visual_accuracy_improvement": final_guard["accuracy"] - initial_guard["accuracy"],
        "visual_coverage_improvement": final_guard["coverage_CVaR25"] - initial_guard["coverage_CVaR25"],
        "wall_time_s": elapsed,
        "changed_w1": 1,
        "changed_w2": 0,
        "changed_mlp": 0,
        "mlp_readout_used": 0,
        "mlp_initial_layer_used": 0,
        "metric_drift_after_max": trace.get("metric_drift_after_max", 0.0),
        **basis_audit_fields(basis_key, model),
    }
    return row


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    compile_pass = 1
    compile_error = ""
    try:
        py_compile.compile(str(RUNNER), doraise=True)
    except Exception as exc:
        compile_pass = 0
        compile_error = repr(exc)
    import_pass = 1
    scan_pass, scan_hits = static_scan([RUNNER])
    rows: list[dict[str, Any]] = []
    x = torch.randn(24, int(args.input_dim), device=device)
    y = torch.randint(0, int(args.num_classes), (24,), device=device)
    for basis_key in BASIS_KEYS:
        bargs = basis_args(args, basis_key)
        for depth in ["depth2", "depth3"]:
            model = make_model(depth, int(args.input_dim), int(args.num_classes), 2294 + len(rows), bargs, device)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            diag = v2293.projected_metric_step(model, x, y, float(args.metric_lr), bargs)
            changed = {n: int(not torch.allclose(before[n], p.detach())) for n, p in model.named_parameters()}
            rows.append(
                {
                    "model_name": f"{basis_key}_{depth}",
                    "depth": model.depth,
                    "widths": ",".join(str(v) for v in model.dims),
                    "basis_count_per_layer": model.k,
                    "has_mlp_readout": 0,
                    "has_mlp_initial_layer": 0,
                    "external_product_feature_used": 0,
                    "new_edge_function_added": 0,
                    "w2_readout_in_forward": 0,
                    "changed_w1_edge_tensors": int(any(changed.values())),
                    "changed_w2_readout_tensors": 0,
                    "changed_mlp_tensors": 0,
                    "all_trainable_param_names": ";".join(name for name, _p in model.named_parameters()),
                    "velocity_emitted": diag.get("velocity_emitted", 0),
                    "changed_w1": int(any(changed.values())),
                    **basis_audit_fields(basis_key, model),
                }
            )
    pure_gate = int(
        rows
        and all(ival(r.get("velocity_emitted")) == 1 for r in rows)
        and all(ival(r.get("changed_w1_edge_tensors")) == 1 for r in rows)
        and all(ival(r.get("changed_w2_readout_tensors")) == 0 for r in rows)
        and all(ival(r.get("has_mlp_readout")) == 0 and ival(r.get("has_mlp_initial_layer")) == 0 for r in rows)
    )
    gate = int(compile_pass and import_pass and scan_pass and pure_gate)
    matrix = OUT_ROOT / "part_a_identity_matrix.csv"
    summary = OUT_ROOT / "part_a_identity.json"
    out = {
        "gate": "v22_94_part_a_identity",
        "compile_pass": compile_pass,
        "compile_error": compile_error,
        "import_pass": import_pass,
        "static_scan_pass": scan_pass,
        "static_scan_hits": scan_hits,
        "purekan_depth2_constructed": int(any(r.get("depth") == 2 for r in rows)),
        "purekan_depth3_constructed": int(any(r.get("depth") == 3 for r in rows)),
        "part_a_gate_pass": gate,
        "rows": rows,
        "changed_w1": int(any(ival(r.get("changed_w1")) for r in rows)),
        "dche_enabled": int(any(ival(r.get("dche_enabled")) for r in rows)),
        "dfour_enabled": int(any(ival(r.get("dfour_enabled")) for r in rows)),
    }
    write_rows(matrix, rows)
    write_json(summary, out)
    route = "A_Pass" if gate else "A_FailedIdentity"
    next_path = write_next_actions("a", route, "none" if gate else "identity", [] if gate else [{"action": "repair identity/static scan before Part C", "reason": "Part A hard gate failed"}], [rel(matrix), rel(summary)])
    append_exec("part-a", command_text(sys.argv), "done" if gate else "failed", gpu=str(device), files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part A identity",
        [
            f"gate_pass={gate}; compile={compile_pass}; import={import_pass}; static_scan={scan_pass}; rows={len(rows)}",
            "analysis: constructed depth2/depth3 PureKAN rows for D-CHE K=5, D-CHE K=9, and D-FOUR; PureKAN primary rows update only coefficient tensors and do not use MLP stem/readout or external product features.",
        ],
    )
    return out


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    required = {
        "v22_90_part_e": ROOT / "results/v22_90/part_e_summary.json",
        "v22_90_part_f": ROOT / "results/v22_90/part_f_summary.json",
        "v22_90_pullback": ROOT / "results/v22_90/part_f_output_safe_pullback_probe_summary.json",
        "v22_91_final": ROOT / "results/v22_91/final_route.json",
        "v22_91_c": ROOT / "results/v22_91/part_c_summary_dist_h20.json",
        "v22_93_final": ROOT / "results/v22_93/final_route.json",
        "v22_93_c": ROOT / "results/v22_93/part_c_architecture_capability_summary.json",
        "v22_93_repair": ROOT / "results/v22_93/part_c_repair_comparison_latest.json",
    }
    v90e = read_json(required["v22_90_part_e"])
    v90f = read_json(required["v22_90_part_f"])
    v90pull = read_json(required["v22_90_pullback"])
    v91final = read_json(required["v22_91_final"])
    v91c = read_json(required["v22_91_c"])
    v93final = read_json(required["v22_93_final"])
    v93c = read_json(required["v22_93_c"])
    repair = []
    try:
        repair = json.loads(required["v22_93_repair"].read_text(encoding="utf-8"))
    except Exception:
        repair = []
    best_che_k5 = "missing"
    if isinstance(repair, list):
        vals = [fval(r.get("visual_improvement_median"), float("nan")) for r in repair if "che_k5_full" in str(r.get("variant", ""))]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            best_che_k5 = max(vals)
    pull_norm = v90pull.get("by_direction", {}).get("output_safe_pullback_norm", {}) if isinstance(v90pull.get("by_direction"), dict) else {}
    out = {
        "gate": "v22_94_part_b_history_lock",
        "history_files": {k: rel(v) for k, v in required.items()},
        "missing_history_files": [rel(v) for v in required.values() if not v.exists()],
        "v22_90_part_e_pass": v90e.get("part_e_gate_pass", "missing"),
        "v22_90_part_f_route": v90f.get("part_f_route", v90f.get("dominant_blocker", "missing")),
        "v22_90_beats_MLP_composite": v90f.get("beats_MLP_composite_coordinate", v90f.get("beats_MLP_composite", "missing")),
        "v22_90_no_debt": v90f.get("no_debt", "missing"),
        "v22_90_visual_coverage": v90f.get("visual_coverage_pass", v90f.get("visual_coverage", "missing")),
        "v22_90_debt_sign_agreement": read_json(ROOT / "results/v22_91/part_b_history_lock.json").get("debt_surrogate_sign_agreement", "missing"),
        "v22_90_output_safe_pullback_residual": pull_norm.get("composite_pullback_residual_median", v90pull.get("output_safe_pullback_residual", "missing")),
        "v22_91_final_route": v91final.get("final_route", "missing"),
        "v22_91_response_debt_sign": v91c.get("response_debt_sign_agreement", "missing"),
        "v22_91_debt_R2": v91c.get("debt_R2", "missing"),
        "v22_91_leave_family_debt_R2": v91c.get("leave_dataset_family_debt_R2", "missing"),
        "v22_93_final_route": v93final.get("final_route", "missing"),
        "v22_93_part_c_route": v93c.get("part_c_route", "missing"),
        "v22_93_c2_adamw_architecture_signal": v93c.get("c2_adamw_architecture_signal", "missing"),
        "v22_93_c2_metric_visual_gain": v93c.get("visual_synthetic_coverage_improvement_median", "missing"),
        "v22_93_best_che_k5_gain": best_che_k5,
        "history_lock_statement": "v22.94 cannot treat v22.90 Part E positive-control as real-task success and cannot treat v22.93 AdamW architecture signal as metric branch success.",
    }
    gate = int(not out["missing_history_files"] and ival(out["v22_90_part_e_pass"]) == 1 and ival(out["v22_93_c2_adamw_architecture_signal"]) == 1)
    out["part_b_gate_pass"] = gate
    summary = OUT_ROOT / "part_b_history_lock.json"
    write_json(summary, out)
    route = "B_Pass" if gate else "B_HistoryLockFailed"
    next_path = write_next_actions("b", route, "none" if gate else "history_lock", [] if gate else [{"action": "compat-parse or locate missing historical artifact; do not infer values from memory", "reason": "history lock failed"}], [rel(summary)])
    append_exec("part-b", command_text(sys.argv), "done" if gate else "failed", files=f"{rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part B history lock",
        [
            f"gate_pass={gate}; v22_90_part_e={out['v22_90_part_e_pass']}; v22_90_part_f_route={out['v22_90_part_f_route']}; v22_91_final_route={out['v22_91_final_route']}; v22_93_final_route={out['v22_93_final_route']}",
            f"v22_93 AdamW signal={out['v22_93_c2_adamw_architecture_signal']}; metric C2 gain={out['v22_93_c2_metric_visual_gain']}; best_che_k5_gain={out['v22_93_best_che_k5_gain']}",
            "analysis: history lock preserves the route boundary: positive-control success and AdamW capacity are not metric-branch success.",
        ],
    )
    return out


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int]]:
    primary = [
        (basis, depth, opt, task, seed)
        for basis in BASIS_KEYS
        for depth in ["depth2", "depth3"]
        for opt in ["adamw", "metric", "projected_adamw_tube"]
        for task in VISUAL_TASKS
        for seed in range(int(args.visual_seed_count))
    ]
    # Diagnostic baseline used only to reproduce the v22.93 architecture-signal
    # definition: deep PureKAN AdamW/metric final coverage relative to shallow.
    shallow = [
        (basis, "shallow", opt, task, seed)
        for basis in BASIS_KEYS
        for opt in ["adamw", "metric"]
        for task in VISUAL_TASKS
        for seed in range(int(args.visual_seed_count))
    ]
    return primary + shallow


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for basis_key, depth, opt, task, seed in shard_items(part_c_jobs(args), args):
        rows.append(train_witness_run(basis_key, depth, opt, task, seed, args, device))
    path = OUT_ROOT / f"part_c_witness_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-c-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def aggregate_best_by_job(rows: list[dict[str, Any]], optimizer: str, basis_key: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r.get("basis_key"), r.get("visual_synthetic_task"), r.get("seed")) for r in rows if r.get("optimizer_kind") == optimizer and (basis_key is None or r.get("basis_key") == basis_key)})
    for basis, task, seed in keys:
        cand = [
            r
            for r in rows
            if r.get("optimizer_kind") == optimizer
            and r.get("basis_key") == basis
            and r.get("visual_synthetic_task") == task
            and str(r.get("seed")) == str(seed)
            and r.get("depth") in {"depth2", "depth3"}
        ]
        if not cand:
            continue
        best = max(cand, key=lambda r: fval(r.get("visual_accuracy_improvement")))
        cov_best = max(cand, key=lambda r: fval(r.get("visual_coverage_improvement")))
        merged = dict(best)
        merged["best_depth_by_accuracy"] = best.get("depth")
        merged["visual_coverage_improvement"] = cov_best.get("visual_coverage_improvement")
        merged["best_depth_by_coverage"] = cov_best.get("depth")
        out.append(merged)
    return out


def aggregate_deep_vs_shallow(rows: list[dict[str, Any]], optimizer: str, basis_key: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r.get("basis_key"), r.get("visual_synthetic_task"), r.get("seed")) for r in rows if r.get("optimizer_kind") == optimizer and (basis_key is None or r.get("basis_key") == basis_key)})
    for basis, task, seed in keys:
        deep = [
            r
            for r in rows
            if r.get("optimizer_kind") == optimizer
            and r.get("basis_key") == basis
            and r.get("visual_synthetic_task") == task
            and str(r.get("seed")) == str(seed)
            and r.get("depth") in {"depth2", "depth3"}
        ]
        shallow = [
            r
            for r in rows
            if r.get("optimizer_kind") == optimizer
            and r.get("basis_key") == basis
            and r.get("visual_synthetic_task") == task
            and str(r.get("seed")) == str(seed)
            and r.get("depth") == "shallow"
        ]
        if not deep or not shallow:
            continue
        best_acc = max(deep, key=lambda r: fval(r.get("final_visual_accuracy")))
        best_cov = max(deep, key=lambda r: fval(r.get("final_visual_coverage")))
        base = shallow[0]
        out.append(
            {
                "basis_key": basis,
                "visual_synthetic_task": task,
                "seed": seed,
                "best_depth_by_accuracy": best_acc.get("depth"),
                "best_depth_by_coverage": best_cov.get("depth"),
                "visual_accuracy_improvement": fval(best_acc.get("final_visual_accuracy")) - fval(base.get("final_visual_accuracy")),
                "visual_coverage_improvement": fval(best_cov.get("final_visual_coverage")) - fval(base.get("final_visual_coverage")),
                "deep_final_visual_accuracy": best_acc.get("final_visual_accuracy"),
                "shallow_final_visual_accuracy": base.get("final_visual_accuracy"),
                "deep_final_visual_coverage": best_cov.get("final_visual_coverage"),
                "shallow_final_visual_coverage": base.get("final_visual_coverage"),
            }
        )
    return out


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_witness_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = OUT_ROOT / "part_c_witness_matrix.csv"
    write_rows(matrix, rows)
    per_basis: dict[str, Any] = {}
    adamw_best_all: list[dict[str, Any]] = []
    metric_best_all: list[dict[str, Any]] = []
    for basis_key in BASIS_KEYS:
        adamw_best = aggregate_deep_vs_shallow(rows, "adamw", basis_key)
        metric_best = aggregate_deep_vs_shallow(rows, "metric", basis_key)
        adamw_best_all.extend(adamw_best)
        metric_best_all.extend(metric_best)
        per_basis[basis_key] = {
            "adamw_rows": len(adamw_best),
            "metric_rows": len(metric_best),
            "adamw_visual_accuracy_improvement_median": median([r.get("visual_accuracy_improvement") for r in adamw_best]),
            "adamw_visual_coverage_improvement_median": median([r.get("visual_coverage_improvement") for r in adamw_best]),
            "adamw_architecture_signal_rows": sum(int(fval(r.get("visual_accuracy_improvement")) > 0.0) for r in adamw_best),
            "adamw_visual_accuracy_rows_ge_0p20": sum(int(fval(r.get("visual_accuracy_improvement")) >= 0.20) for r in adamw_best),
            "adamw_visual_coverage_rows_ge_0p03": sum(int(fval(r.get("visual_coverage_improvement")) >= 0.03) for r in adamw_best),
            "metric_branch_visual_coverage_improvement_median": median([r.get("visual_coverage_improvement") for r in metric_best]),
            "metric_rows_ge_0p02": sum(int(fval(r.get("visual_coverage_improvement")) >= 0.02) for r in metric_best),
        }
    adamw_acc = median([r.get("visual_accuracy_improvement") for r in adamw_best_all])
    adamw_cov = median([r.get("visual_coverage_improvement") for r in adamw_best_all])
    adamw_signal_rows = sum(int(fval(r.get("visual_accuracy_improvement")) > 0.0) for r in adamw_best_all)
    metric_cov = median([r.get("visual_coverage_improvement") for r in metric_best_all])
    expected = int(len(BASIS_KEYS) * len(VISUAL_TASKS) * int(args.visual_seed_count))
    witness_gate = int(adamw_acc >= 0.20 and adamw_cov >= 0.03 and adamw_signal_rows >= math.ceil((8.0 / 12.0) * expected))
    metric_unopened = int(metric_cov < 0.02)
    if not witness_gate:
        route, blocker = "A_WitnessMissing", "AdamWWitnessMissing"
    elif not metric_unopened:
        route, blocker = "UnexpectedMetricBranchC2PassRequiresAudit", "metric_branch_unexpected_c2"
    else:
        route, blocker = "C_WitnessValidMetricGapConfirmed", "none"
    gate = int(witness_gate and metric_unopened)
    out = {
        "gate": "v22_94_part_c_adamw_witness_capture",
        "rows": len(rows),
        "expected_rows": len(part_c_jobs(args)),
        "basis_families": BASIS_KEYS,
        "architecture_signal_definition": "deep final (max depth2/depth3) minus shallow final for same basis/task/seed/optimizer, matching v22.93 Part C architecture-signal audit",
        "per_basis": per_basis,
        "adamw_visual_accuracy_improvement_median": adamw_acc,
        "adamw_visual_coverage_improvement_median": adamw_cov,
        "adamw_architecture_signal_rows": adamw_signal_rows,
        "adamw_visual_accuracy_rows_ge_0p20": sum(int(fval(r.get("visual_accuracy_improvement")) >= 0.20) for r in adamw_best_all),
        "adamw_visual_coverage_rows_ge_0p03": sum(int(fval(r.get("visual_coverage_improvement")) >= 0.03) for r in adamw_best_all),
        "adamw_architecture_signal_expected_rows": expected,
        "metric_branch_visual_coverage_improvement_median": metric_cov,
        "metric_branch_visual_rows_ge_0p02": sum(int(fval(r.get("visual_coverage_improvement")) >= 0.02) for r in metric_best_all),
        "adamw_witness_valid": witness_gate,
        "metric_branch_unopened": metric_unopened,
        "part_c_gate_pass": gate,
        "part_c_route": route,
        "dominant_blocker": blocker,
        "checkpoint_root": rel(CKPT_ROOT),
        "changed_w1": int(any(ival(r.get("changed_w1")) for r in rows)),
        "dche_enabled": int(any(ival(r.get("dche_enabled")) for r in rows)),
        "dfour_enabled": int(any(ival(r.get("dfour_enabled")) for r in rows)),
    }
    summary = OUT_ROOT / "part_c_witness_summary.json"
    write_json(summary, out)
    if blocker == "AdamWWitnessMissing":
        actions = [{"action": "check C2 generator, seed split, basis settings, training budget, visual shard balance before Part D", "reason": blocker}]
    elif blocker == "metric_branch_unexpected_c2":
        actions = [{"action": "audit metric branch unexpected C2 pass; do Part D before any promotion", "reason": blocker}]
    else:
        actions = []
    next_path = write_next_actions("c", route, blocker, actions, [rel(matrix), rel(summary), rel(CKPT_ROOT)])
    append_exec("part-c-merge", command_text(sys.argv), "done" if gate else "failed", files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part C AdamW witness",
        [
            f"gate_pass={gate}; route={route}; rows={len(rows)}/{len(part_c_jobs(args))}; AdamW acc improve median={adamw_acc}; AdamW coverage improve median={adamw_cov}; signal_rows={adamw_signal_rows}/{expected}",
            f"metric_branch coverage improve median={metric_cov}; metric_rows_ge_0p02={out['metric_branch_visual_rows_ge_0p02']}",
            f"per_basis={json.dumps(per_basis, sort_keys=True)}",
            "analysis: Part C used fixed basis/depth/optimizer rows. Depth aggregation is offline diagnostic only for witness validity; no runtime basis/depth/scheme winner is introduced.",
        ],
    )
    return out


def load_npz(run_id: str, step: int) -> dict[str, np.ndarray]:
    path, _json_path = checkpoint_paths(run_id, step)
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_json_meta(run_id: str, step: int) -> dict[str, Any]:
    _path, json_path = checkpoint_paths(run_id, step)
    return read_json(json_path)


def mats_from_npz(data: dict[str, np.ndarray], prefix: str, device: torch.device) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    idx = 0
    while f"{prefix}_layer{idx}" in data:
        out.append(torch.tensor(data[f"{prefix}_layer{idx}"], device=device, dtype=torch.float64))
        idx += 1
    return out


def project_tangent(c: torch.Tensor, a: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if float(d.norm().detach().cpu().item()) <= 0.0:
        z = torch.zeros_like(d)
        return z, z, 0.0, 0.0
    c = sym(c.to(device=a.device, dtype=torch.float64))
    a = a.to(dtype=torch.float64)
    d = d.to(dtype=torch.float64)
    r = sym(a.T @ c @ a) + EPS * torch.eye(int(a.shape[1]), device=a.device, dtype=torch.float64)
    b = sym(a.T @ c @ d + d.T @ c @ a)
    n = int(r.shape[0])
    eye = torch.eye(n, device=a.device, dtype=torch.float64)
    k = torch.kron(eye, r) + torch.kron(r, eye) + EPS * torch.eye(n * n, device=a.device, dtype=torch.float64)
    rhs = b.reshape(-1)
    m_vec: torch.Tensor | None = None
    if bool(torch.isfinite(k).all().detach().cpu().item()) and bool(torch.isfinite(rhs).all().detach().cpu().item()):
        for ridge in [0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2]:
            try:
                kk = k if ridge == 0.0 else k + ridge * torch.eye(n * n, device=a.device, dtype=torch.float64)
                candidate = torch.linalg.solve(kk, rhs)
                if bool(torch.isfinite(candidate).all().detach().cpu().item()):
                    m_vec = candidate
                    break
            except Exception:
                pass
        if m_vec is None:
            try:
                candidate = torch.linalg.lstsq(k + 1.0e-4 * torch.eye(n * n, device=a.device, dtype=torch.float64), rhs).solution
                if bool(torch.isfinite(candidate).all().detach().cpu().item()):
                    m_vec = candidate
            except Exception:
                m_vec = None
    if m_vec is None:
        normal = torch.zeros_like(d)
        tangent = d
        return tangent, normal, float("inf"), 0.0
    m = m_vec.reshape(n, n)
    normal = a @ m
    if not bool(torch.isfinite(normal).all().detach().cpu().item()):
        normal = torch.zeros_like(d)
    tangent = d - normal
    residual = float((a.T @ c @ tangent + tangent.T @ c @ a).norm().div((b.norm() + EPS)).detach().cpu().item())
    recon = float((tangent + normal - d).norm().div(d.norm().clamp_min(EPS)).detach().cpu().item())
    return tangent, normal, residual, recon


def apply_delta_and_eval(
    model: v2293.TrueDeepPureKAN,
    base_mats: list[torch.Tensor],
    deltas: list[torch.Tensor],
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    scale: float,
    layers: set[int] | None = None,
) -> dict[str, float]:
    mats = [m.clone() for m in base_mats]
    for idx, delta in enumerate(deltas):
        if layers is None or idx in layers:
            mats[idx] = mats[idx] + float(scale) * delta.to(device=mats[idx].device, dtype=torch.float64)
    set_model_mats(model, mats)
    return v2293.metrics_for_model(model, x_guard, y_guard)


def decomp_jobs_from_part_c(args: argparse.Namespace) -> list[tuple[dict[str, Any], int]]:
    rows = read_rows(OUT_ROOT / "part_c_witness_matrix.csv")
    adamw = [r for r in rows if r.get("optimizer_kind") == "adamw" and r.get("depth") in {"depth2", "depth3"}]
    steps = [int(s) for s in csv_items(args.decomp_steps)]
    jobs: list[tuple[dict[str, Any], int]] = []
    for row in adamw:
        for step in steps:
            path, _meta = checkpoint_paths(str(row.get("run_id")), step)
            if path.exists():
                jobs.append((row, step))
    return jobs


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    csum = read_json(OUT_ROOT / "part_c_witness_summary.json")
    if not csum.get("part_c_gate_pass"):
        out = {
            "gate": "v22_94_part_d_missing_freedom_decomposition",
            "part_d_gate_pass": 0,
            "part_d_route": "D_SkippedBecausePartCFailed",
            "skip_reason": csum.get("part_c_route", "missing_part_c"),
            "rows": 0,
        }
        summary = OUT_ROOT / "part_d_decomposition_summary.json"
        write_json(summary, out)
        write_rows(OUT_ROOT / "part_d_decomposition_matrix.csv", [])
        next_path = write_next_actions("d", out["part_d_route"], "part_c", [{"action": "repair Part C witness before decomposition", "reason": str(out["skip_reason"])}], [rel(summary)])
        append_exec("part-d", command_text(sys.argv), "skipped", files=f"{rel(summary)}; {rel(next_path)}")
        append_recap("Part D decomposition", [f"gate_pass=0; skipped because Part C route={out['skip_reason']}"])
        return out
    device = device_from_args(args)
    rows_c = read_rows(OUT_ROOT / "part_c_witness_matrix.csv")
    metric_lookup = {(r.get("basis_key"), r.get("depth"), r.get("visual_synthetic_task"), str(r.get("seed"))): r for r in rows_c if r.get("optimizer_kind") == "metric"}
    rows: list[dict[str, Any]] = []
    for adamw_row, step in decomp_jobs_from_part_c(args):
        basis_key = str(adamw_row.get("basis_key"))
        depth = str(adamw_row.get("depth"))
        task = str(adamw_row.get("visual_synthetic_task"))
        seed = int(fval(adamw_row.get("seed")))
        bargs = basis_args(args, basis_key)
        xtr, ytr, xg, yg = visual_data(task, seed, args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = make_model(depth, int(xtr.shape[1]), classes, int(fval(adamw_row.get("model_seed"))), bargs, device)
        adata = load_npz(str(adamw_row.get("run_id")), step)
        a_before = mats_from_npz(adata, "A_before", device)
        adamw_delta = mats_from_npz(adata, "delta", device)
        set_model_mats(model, a_before)
        base_metrics = v2293.metrics_for_model(model, xg, yg)
        key = (basis_key, depth, task, str(seed))
        metric_row = metric_lookup.get(key, {})
        metric_delta = zero_like_mats(model)
        if metric_row:
            try:
                mdata = load_npz(str(metric_row.get("run_id")), step)
                metric_delta = mats_from_npz(mdata, "delta", device)
            except Exception:
                metric_delta = zero_like_mats(model)
        proj_deltas: list[torch.Tensor] = []
        deleted_deltas: list[torch.Tensor] = []
        data_tan_fracs: list[float] = []
        data_norm_fracs: list[float] = []
        op_tan_fracs: list[float] = []
        op_norm_fracs: list[float] = []
        residuals: list[float] = []
        recons: list[float] = []
        data_drifts: list[float] = []
        op_drifts: list[float] = []
        mode_deleted_fracs: list[float] = []
        for layer_idx, (a, d) in enumerate(zip(a_before, adamw_delta)):
            c_data = data_c_matrix(model, xtr, layer_idx)
            t_data, n_data, res, recon = project_tangent(c_data, a, d)
            mode_diag = mode_metric_diag(model, layer_idx, bargs)
            c_op = torch.diag(mode_diag)
            t_op, n_op, res_op, recon_op = project_tangent(c_op, a, d)
            proj_deltas.append(t_data)
            deleted_deltas.append(n_data)
            dnorm = d.norm().clamp_min(EPS)
            data_tan_fracs.append(float(t_data.norm().div(dnorm).detach().cpu().item()))
            data_norm_fracs.append(float(n_data.norm().div(dnorm).detach().cpu().item()))
            op_tan_fracs.append(float(t_op.norm().div(dnorm).detach().cpu().item()))
            op_norm_fracs.append(float(n_op.norm().div(dnorm).detach().cpu().item()))
            residuals.extend([res, res_op])
            recons.extend([recon, recon_op])
            data_drifts.append(log_spectral_drift(composite_metric(a, c_data), composite_metric(a + d, c_data)))
            op_drifts.append(log_spectral_drift(op_spectrum(a, mode_diag), op_spectrum(a + d, mode_diag)))
            mode_deleted_fracs.append(float(n_data.norm().div(dnorm).detach().cpu().item()))
        metric_cos = cosine_mats(adamw_delta, metric_delta)
        metric_eval = apply_delta_and_eval(model, a_before, metric_delta, xg, yg, 1.0)
        set_model_mats(model, a_before)
        alpha_rows: list[dict[str, float]] = []
        for alpha in [float(x) for x in csv_items(args.replay_alphas)]:
            gain_ad = apply_delta_and_eval(model, a_before, adamw_delta, xg, yg, alpha)["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
            gain_proj = apply_delta_and_eval(model, a_before, proj_deltas, xg, yg, alpha)["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
            gain_deleted = apply_delta_and_eval(model, a_before, deleted_deltas, xg, yg, alpha)["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
            for beta in [float(x) for x in csv_items(args.replay_betas)]:
                combo = [p + beta * dd for p, dd in zip(proj_deltas, deleted_deltas)]
                gain_combo = apply_delta_and_eval(model, a_before, combo, xg, yg, alpha)["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
                alpha_rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "gain_adamw": gain_ad,
                        "gain_proj": gain_proj,
                        "gain_deleted": gain_deleted,
                        "gain_proj_plus_deleted": gain_combo,
                        "deleted_replay_error": abs(gain_combo - gain_ad) / max(abs(gain_ad), EPS) if beta == 1.0 else float("nan"),
                    }
                )
        layer_synergies: list[float] = []
        for layer_idx in range(max(0, len(adamw_delta) - 1)):
            alpha = 1.0
            g_l = apply_delta_and_eval(model, a_before, adamw_delta, xg, yg, alpha, {layer_idx})["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
            g_r = apply_delta_and_eval(model, a_before, adamw_delta, xg, yg, alpha, {layer_idx + 1})["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
            g_joint = apply_delta_and_eval(model, a_before, adamw_delta, xg, yg, alpha, {layer_idx, layer_idx + 1})["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
            layer_synergies.append(g_joint - g_l - g_r)
        replay_main = [r for r in alpha_rows if abs(fval(r.get("alpha")) - 0.08) < 1.0e-12 and abs(fval(r.get("beta")) - 1.0) < 1.0e-12]
        replay_error = replay_main[0]["deleted_replay_error"] if replay_main else 1.0
        gain_adamw = replay_main[0]["gain_adamw"] if replay_main else 0.0
        gain_metric = metric_eval["coverage_CVaR25"] - base_metrics["coverage_CVaR25"]
        gap = gain_adamw - gain_metric
        explained = max(0.0, 1.0 - replay_error) if abs(gap) > 1.0e-12 else 0.0
        rows.append(
            {
                "basis_key": basis_key,
                "depth": depth,
                "visual_synthetic_task": task,
                "seed": seed,
                "step": step,
                "data_tangent_fraction": mean(data_tan_fracs),
                "data_normal_fraction": mean(data_norm_fracs),
                "data_gram_log_spectral_drift": mean(data_drifts),
                "op_tangent_fraction": mean(op_tan_fracs),
                "op_normal_fraction": mean(op_norm_fracs),
                "op_spectrum_log_drift": mean(op_drifts),
                "mode_band_deleted_by_metric_projection": mean(mode_deleted_fracs),
                "cosine_adamw_to_metric_update": metric_cos,
                "cosine_metric_update_to_data_tangent": cosine_mats(metric_delta, proj_deltas),
                "cross_layer_synergy": median(layer_synergies),
                "projection_reconstruction_error": median(recons),
                "decomposition_numerical_residual": median(residuals),
                "deleted_component_replay_error": replay_error,
                "gain_adamw": gain_adamw,
                "gain_metric": gain_metric,
                "gain_gap_adamw_minus_metric": gap,
                "explained_gain_fraction": explained,
                "deleted_fraction": mean(data_norm_fracs),
                "replay_json": json.dumps(alpha_rows, sort_keys=True),
                "changed_w1": 1,
                **basis_audit_fields(basis_key, model),
            }
        )
    matrix = OUT_ROOT / "part_d_decomposition_matrix.csv"
    summary = OUT_ROOT / "part_d_decomposition_summary.json"
    write_rows(matrix, rows)
    valid = [r for r in rows if math.isfinite(fval(r.get("decomposition_numerical_residual"), float("nan")))]
    explained_median = median([r.get("explained_gain_fraction") for r in valid])
    valid_ratio = len(valid) / max(1, len(rows))
    residual_median = median([r.get("decomposition_numerical_residual") for r in valid], default=1.0)
    recon_median = median([r.get("projection_reconstruction_error") for r in valid], default=1.0)
    replay_err_median = median([r.get("deleted_component_replay_error") for r in valid], default=1.0)
    witness_rows = int(csum.get("adamw_architecture_signal_rows", 0))
    witness_expected = int(csum.get("adamw_architecture_signal_expected_rows", 1))
    gate = int(
        rows
        and explained_median >= 0.60
        and valid_ratio >= 0.80
        and witness_rows >= math.ceil((8.0 / 12.0) * witness_expected)
        and residual_median <= 0.05
        and recon_median <= 0.05
        and replay_err_median <= 0.10
    )
    if gate:
        route, blocker = "D_DecompositionExplained", "none"
    else:
        route, blocker = "D_DecompositionUnexplained", "DecompositionUnexplained"
    out = {
        "gate": "v22_94_part_d_missing_freedom_decomposition",
        "rows": len(rows),
        "valid_replay_rows": len(valid),
        "valid_replay_ratio": valid_ratio,
        "adamw_witness_rows": witness_rows,
        "adamw_witness_expected_rows": witness_expected,
        "explained_gain_fraction_median": explained_median,
        "decomposition_numerical_residual_median": residual_median,
        "projection_reconstruction_error_median": recon_median,
        "deleted_component_replay_error_median": replay_err_median,
        "data_normal_fraction_median": median([r.get("data_normal_fraction") for r in valid]),
        "op_normal_fraction_median": median([r.get("op_normal_fraction") for r in valid]),
        "cross_layer_synergy_median": median([r.get("cross_layer_synergy") for r in valid]),
        "part_d_gate_pass": gate,
        "part_d_route": route,
        "dominant_blocker": blocker,
        "changed_w1": 1,
    }
    write_json(summary, out)
    actions = [] if gate else [{"action": "increase checkpoints, causal replay scales, optimizer moment logging, and layer-pair replay; do not propose a new optimizer before gap is explained", "reason": blocker}]
    next_path = write_next_actions("d", route, blocker, actions, [rel(matrix), rel(summary)])
    append_exec("part-d", command_text(sys.argv), "done" if gate else "failed", gpu=str(device), files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part D decomposition",
        [
            f"gate_pass={gate}; route={route}; rows={len(rows)}; valid_ratio={valid_ratio}; explained_gain_fraction_median={explained_median}",
            f"residual_median={residual_median}; reconstruction_error_median={recon_median}; deleted_replay_error_median={replay_err_median}",
            f"data_normal_fraction_median={out['data_normal_fraction_median']}; op_normal_fraction_median={out['op_normal_fraction_median']}; cross_layer_synergy_median={out['cross_layer_synergy_median']}",
            "analysis: projection-deletion replay uses saved AdamW deltas and data-tangent projection as the metric-preservation proxy; failures remain blockers, not success evidence.",
        ],
    )
    return out


def cap_delta(a: torch.Tensor, d: torch.Tensor, ratio: float) -> torch.Tensor:
    cap = float(ratio) * max(float(a.norm().detach().cpu().item()), EPS)
    n = float(d.norm().detach().cpu().item())
    if n > cap:
        return d * (cap / max(n, EPS))
    return d


def in_band_mask(param: torch.Tensor, basis_name: str) -> torch.Tensor:
    mask = torch.zeros_like(param.detach(), dtype=torch.float64)
    k = int(param.shape[2])
    if basis_name == "chebyshev":
        idxs = list(range(2, min(k, 6)))
    else:
        idxs = list(range(1, min(k, 5)))
    if idxs:
        mask[:, :, idxs] = 1.0
    return w1_to_matrix(mask.to(device=param.device, dtype=param.dtype)).to(device=param.device, dtype=torch.float64)


def scheme_step(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    step: int,
    total_steps: int,
    scheme: str,
    args: argparse.Namespace,
    *,
    debt_baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    logits, _acts = model.forward_with_activations(x)
    ce_loss = F.cross_entropy(logits.float(), y.long())
    cal_weight = scheme_calibration_weight(scheme, args)
    cal_active = debt_baseline is not None and cal_weight > 0.0 and step >= int(math.ceil(float(args.scheme_calibration_start_frac) * float(total_steps)))
    if cal_active:
        debt_loss = v2293.v2289.debt_surrogate_loss(
            logits,
            y,
            debt_baseline,
            float(args.no_debt_budget),
            ece_weight=float(args.scheme_calibration_ece_weight),
            bucketed_ece=True,
        )
        tail_weight, margin_weight = scheme_tail_margin_weights(scheme, args)
        debt_loss = debt_loss + tail_margin_surrogate_loss(logits, y, debt_baseline, float(args.no_debt_budget), tail_weight, margin_weight)
        loss = ce_loss + cal_weight * debt_loss
    else:
        debt_loss = logits.float().sum() * 0.0
        loss = ce_loss
    loss.backward()
    before = param_mats(model)
    grads = grad_mats(model)
    new_mats: list[torch.Tensor] = []
    data_drifts: list[float] = []
    op_drifts: list[float] = []
    preserved: list[float] = []
    t_form = 0.0
    base_scheme = scheme_base_name(scheme)
    if base_scheme == "formation_20_op_tube":
        t_form = 0.20
    elif base_scheme == "formation_35_op_tube" or base_scheme == "activation_domain_freedom":
        t_form = 0.35
    formation = t_form > 0.0 and step <= int(math.ceil(t_form * total_steps))
    for layer_idx, (param, a, g) in enumerate(zip(model.coeffs, before, grads)):
        raw = -g
        mode_diag = mode_metric_diag(model, layer_idx, args)
        c_op = torch.diag(mode_diag)
        tangent, normal, _res, _recon = project_tangent(c_op, a, raw)
        if base_scheme == "op_strict":
            delta = tangent
        elif base_scheme in {"op_tube", "cross_layer_coupled_proxy"}:
            delta = tangent + float(args.scheme_op_normal_scale) * normal
        elif base_scheme in {"formation_20_op_tube", "formation_35_op_tube", "activation_domain_freedom"}:
            delta = tangent + (0.35 if formation else float(args.scheme_op_normal_scale)) * normal
        elif base_scheme == "mode_band_freedom":
            band = in_band_mask(param, model.basis_name)
            delta = tangent + normal * band
        else:
            delta = tangent
        delta = cap_delta(a, float(args.scheme_max_norm_ratio) * float(args.metric_lr) * delta, float(args.metric_max_norm_ratio))
        c_data = data_c_matrix(model, x, layer_idx)
        data_drifts.append(log_spectral_drift(composite_metric(a, c_data), composite_metric(a + delta, c_data)))
        op_drifts.append(log_spectral_drift(op_spectrum(a, mode_diag), op_spectrum(a + delta, mode_diag)))
        if float(g.norm().detach().cpu().item()) > 0.0 and float(delta.norm().detach().cpu().item()) > 0.0:
            preserved.append(float(((-g) * delta).sum().div(g.norm() * delta.norm()).detach().cpu().item()))
        new_mats.append(a + delta)
    set_model_mats(model, new_mats)
    return {
        "loss": float(loss.detach().cpu().item()),
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "debt_surrogate_loss": float(debt_loss.detach().cpu().item()),
        "calibration_active": int(cal_active),
        "data_gram_drift": max(data_drifts or [0.0]),
        "op_spectrum_drift": max(op_drifts or [0.0]),
        "task_gradient_preserved_fraction": min(preserved or [0.0]),
    }


def train_scheme(
    scheme: str,
    basis_key: str,
    depth: str,
    teacher_type: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    bargs = basis_args(args, basis_key)
    if teacher_type == "c2_visual_synthetic":
        xtr, ytr, xg, yg = visual_data(task, seed, args, device)
    else:
        xtr, ytr, xg, yg = synthetic_data(teacher_type, seed, args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    mseed = 2294500 + BASIS_KEYS.index(basis_key) * 100000 + int(seed) * 31 + (2 if depth == "depth2" else 3)
    model = make_model(depth, int(xtr.shape[1]), classes, mseed, bargs, device)
    before = v2293.metrics_for_model(model, xg, yg)
    before_train = v2293.metrics_for_model(model, xtr, ytr)
    mlp_nll = ""
    if teacher_type == "c1c_mlp_friendly":
        mlp = v2293.MatchedMLP(int(xtr.shape[1]), classes, int(args.mlp_hidden), 2, mseed + 777, device)
        mlp_res = v2293.train_model(mlp, "mlp_raw", "mlp", xtr, ytr, xg, yg, seed, bargs)
        mlp_nll = mlp_res["guard_NLL"]
    trace = {"data_gram_drift": 0.0, "op_spectrum_drift": 0.0, "task_gradient_preserved_fraction": 1.0, "debt_surrogate_loss": 0.0, "calibration_steps": 0}
    for step in range(1, int(args.train_steps) + 1):
        xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
        diag = scheme_step(model, xb, yb, step, int(args.train_steps), scheme, bargs, debt_baseline=before_train)
        trace["data_gram_drift"] = max(trace["data_gram_drift"], diag["data_gram_drift"])
        trace["op_spectrum_drift"] = max(trace["op_spectrum_drift"], diag["op_spectrum_drift"])
        trace["task_gradient_preserved_fraction"] = min(trace["task_gradient_preserved_fraction"], diag["task_gradient_preserved_fraction"])
        trace["debt_surrogate_loss"] = max(trace["debt_surrogate_loss"], diag["debt_surrogate_loss"])
        trace["calibration_steps"] += int(diag["calibration_active"])
    after = v2293.metrics_for_model(model, xg, yg)
    after_train = v2293.metrics_for_model(model, xtr, ytr)
    with torch.no_grad():
        _logits, acts = model.forward_with_activations(xtr)
        act_rank = median([safe_activation_rank(a) for a in acts[1:]])
    return {
        "scheme": scheme,
        "basis_key": basis_key,
        "depth": depth,
        "teacher_type": teacher_type,
        "visual_synthetic_task": task,
        "seed": seed,
        "guard_NLL": after["nll"],
        "guard_accuracy": after["accuracy"],
        "guard_NLL_delta": after["nll"] - before["nll"],
        "guard_debt_before": before["debt_metric"],
        "guard_debt_after": after["debt_metric"],
        "train_debt_before": before_train["debt_metric"],
        "train_debt_after": after_train["debt_metric"],
        "train_debt_delta": after_train["debt_metric"] - before_train["debt_metric"],
        "no_debt": int(after["debt_metric"] <= before["debt_metric"] + float(args.no_debt_budget)),
        "Brier_before": before.get("brier", 0.0),
        "Brier_after": after.get("brier", 0.0),
        "Brier_delta": after.get("brier", 0.0) - before.get("brier", 0.0),
        "ECE_before": before.get("ece", 0.0),
        "ECE_after": after.get("ece", 0.0),
        "ECE_delta": after.get("ece", 0.0) - before.get("ece", 0.0),
        "tail95_before": before.get("tail95", 0.0),
        "tail95_after": after.get("tail95", 0.0),
        "tail95_delta": after.get("tail95", 0.0) - before.get("tail95", 0.0),
        "margin10_before": before.get("margin10", 0.0),
        "margin10_after": after.get("margin10", 0.0),
        "margin10_delta": after.get("margin10", 0.0) - before.get("margin10", 0.0),
        "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
        "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
        "train_tail95_delta": after_train.get("tail95", 0.0) - before_train.get("tail95", 0.0),
        "train_margin10_delta": after_train.get("margin10", 0.0) - before_train.get("margin10", 0.0),
        "visual_accuracy_improvement": after["accuracy"] - before["accuracy"],
        "visual_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
        "final_visual_coverage": after["coverage_CVaR25"],
        "mlp_raw_guard_NLL": mlp_nll,
        "c1c_negative_ok": int(mlp_nll != "" and float(mlp_nll) <= after["nll"] + 0.05) if teacher_type == "c1c_mlp_friendly" else "",
        "data_gram_drift_max": trace["data_gram_drift"],
        "op_spectrum_drift_max": trace["op_spectrum_drift"],
        "task_gradient_preserved_fraction_min": trace["task_gradient_preserved_fraction"],
        "debt_surrogate_loss_max": trace["debt_surrogate_loss"],
        "calibration_steps": trace["calibration_steps"],
        "activation_cov_rank_final": act_rank,
        "changed_w1": 1,
        **basis_audit_fields(basis_key, model),
    }


def part_e_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, str, int]]:
    schemes = csv_items(args.part_e_schemes)
    depths = csv_items(args.part_e_depths)
    jobs: list[tuple[str, str, str, str, str, int]] = []
    for scheme in schemes:
        for basis in BASIS_KEYS:
            for depth in depths:
                for teacher in ["c1a_shallow_additive", "c1b_deep_coordinate", "c1c_mlp_friendly"]:
                    for seed in range(int(args.part_e_seed_count)):
                        jobs.append((scheme, basis, depth, teacher, "", seed))
                for task in VISUAL_TASKS:
                    for seed in range(int(args.visual_seed_count)):
                        jobs.append((scheme, basis, depth, "c2_visual_synthetic", task, seed))
    return jobs


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    dsum = read_json(OUT_ROOT / "part_d_decomposition_summary.json")
    if not dsum.get("part_d_gate_pass"):
        out = {
            "gate": "v22_94_part_e_multi_scheme_c2_repair",
            "part_e_gate_pass": 0,
            "part_e_route": "E_SkippedBecausePartDFailed",
            "skip_reason": dsum.get("part_d_route", "missing_part_d"),
            "rows": 0,
        }
        summary = OUT_ROOT / "part_e_multi_scheme_summary.json"
        write_json(summary, out)
        write_rows(OUT_ROOT / "part_e_multi_scheme_matrix.csv", [])
        next_path = write_next_actions("e", out["part_e_route"], "part_d", [{"action": "complete/explain Part D before running repair matrix", "reason": str(out["skip_reason"])}], [rel(summary)])
        append_exec("part-e", command_text(sys.argv), "skipped", files=f"{rel(summary)}; {rel(next_path)}")
        append_recap("Part E multi-scheme", [f"gate_pass=0; skipped because Part D route={out['skip_reason']}"])
        return out
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for scheme, basis, depth, teacher, task, seed in shard_items(part_e_jobs(args), args):
        rows.append(train_scheme(scheme, basis, depth, teacher, task, seed, args, device))
    path = OUT_ROOT / f"part_e_multi_scheme_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-e-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_e_multi_scheme_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = OUT_ROOT / "part_e_multi_scheme_matrix.csv"
    write_rows(matrix, rows)
    groups = sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in rows})
    summaries: list[dict[str, Any]] = []
    for scheme, basis, depth in groups:
        gr = [r for r in rows if r.get("scheme") == scheme and r.get("basis_key") == basis and r.get("depth") == depth]
        c1a = [r for r in gr if r.get("teacher_type") == "c1a_shallow_additive"]
        c1b = [r for r in gr if r.get("teacher_type") == "c1b_deep_coordinate"]
        c1c = [r for r in gr if r.get("teacher_type") == "c1c_mlp_friendly"]
        c2 = [r for r in gr if r.get("teacher_type") == "c2_visual_synthetic"]
        c1a_pass = int(c1a and median([r.get("guard_accuracy") for r in c1a]) >= 0.65)
        c1b_pass = int(c1b and median([r.get("guard_accuracy") for r in c1b]) >= 0.60 and median([r.get("guard_NLL_delta") for r in c1b]) < 0.0)
        c1c_pass = int(c1c and sum(ival(r.get("c1c_negative_ok")) for r in c1c) >= math.ceil(0.5 * len(c1c)))
        cov_med = median([r.get("visual_coverage_improvement") for r in c2])
        acc_med = median([r.get("visual_accuracy_improvement") for r in c2])
        rows_ge = sum(int(fval(r.get("visual_coverage_improvement")) >= 0.02) for r in c2)
        c2_pass = int(c2 and cov_med >= 0.02 and rows_ge >= math.ceil((8.0 / 12.0) * len(c2)) and acc_med > 0.0)
        op_drift = median([r.get("op_spectrum_drift_max") for r in gr])
        task_pres = median([r.get("task_gradient_preserved_fraction_min") for r in gr])
        gate = int(c1a_pass and c1b_pass and c1c_pass and c2_pass and task_pres >= 0.30)
        summaries.append(
            {
                "scheme": scheme,
                "basis_key": basis,
                "depth": depth,
                "rows": len(gr),
                "c1a_pass": c1a_pass,
                "c1b_pass": c1b_pass,
                "c1c_negative_control_pass": c1c_pass,
                "c2_visual_pass": c2_pass,
                "visual_synthetic_coverage_improvement_median": cov_med,
                "visual_rows_ge_0p02": rows_ge,
                "metric_visual_accuracy_improvement_median": acc_med,
                "op_spectrum_log_drift_median": op_drift,
                "task_gradient_preserved_fraction_median": task_pres,
                "scheme_gate_pass": gate,
            }
        )
    pass_groups = [s for s in summaries if ival(s.get("scheme_gate_pass")) == 1]
    if pass_groups:
        route, blocker = "E_C2RepairSchemePass", "none"
    else:
        route, blocker = "E_AllSchemesFail", "c2_visual"
    out = {
        "gate": "v22_94_part_e_multi_scheme_c2_repair",
        "rows": len(rows),
        "scheme_groups": summaries,
        "passing_scheme_groups": pass_groups,
        "part_e_gate_pass": int(bool(pass_groups)),
        "part_e_route": route,
        "dominant_blocker": blocker,
        "changed_w1": int(any(ival(r.get("changed_w1")) for r in rows)),
    }
    summary = OUT_ROOT / "part_e_multi_scheme_summary.json"
    write_json(summary, out)
    next_path = write_next_actions("e", route, blocker, [] if pass_groups else [{"action": "analyze each fixed scheme failure against Part D dominant freedom; do not weaken C2 gate or promote single seed", "reason": blocker}], [rel(matrix), rel(summary)])
    append_exec("part-e-merge", command_text(sys.argv), "done" if pass_groups else "failed", files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part E multi-scheme",
        [
            f"gate_pass={int(bool(pass_groups))}; route={route}; rows={len(rows)}; passing_groups={len(pass_groups)}",
            f"scheme_groups={json.dumps(summaries, sort_keys=True)}",
            "analysis: Part E reports fixed scheme+basis+depth groups. Passing one fixed group is not row-level runtime selection; failing groups remain evidence against their registered hypothesis.",
        ],
    )
    return out


def part_f_passing_groups() -> list[dict[str, Any]]:
    e = read_json(OUT_ROOT / "part_e_multi_scheme_summary.json")
    groups = e.get("passing_scheme_groups", [])
    return groups if isinstance(groups, list) else []


def part_f_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    requested_parts = set(csv_items(str(args.part_f_parts)))
    for group_idx, group in enumerate(part_f_passing_groups()):
        scheme = str(group.get("scheme", ""))
        basis = str(group.get("basis_key", ""))
        depth = str(group.get("depth", ""))
        if scheme not in csv_items(args.part_e_schemes) or basis not in BASIS_KEYS or depth not in {"depth2", "depth3"}:
            continue
        for seed in range(int(args.part_f_seed_count)):
            if "F1" in requested_parts:
                jobs.append({"group_idx": group_idx, "scheme": scheme, "basis_key": basis, "depth": depth, "part": "F1", "teacher_type": "c1a_shallow_additive", "task": "", "seed": seed})
            if "F2" in requested_parts:
                jobs.append({"group_idx": group_idx, "scheme": scheme, "basis_key": basis, "depth": depth, "part": "F2", "teacher_type": "c1b_deep_coordinate", "task": "", "seed": seed})
            if "F4" in requested_parts:
                jobs.append({"group_idx": group_idx, "scheme": scheme, "basis_key": basis, "depth": depth, "part": "F4", "teacher_type": "c1c_mlp_friendly", "task": "", "seed": seed})
            if "F5" in requested_parts:
                jobs.append({"group_idx": group_idx, "scheme": scheme, "basis_key": basis, "depth": depth, "part": "F5", "teacher_type": "f5_no_debt_calibration", "task": "", "seed": seed})
        if "F3" in requested_parts:
            for task in VISUAL_TASKS:
                for seed in range(int(args.visual_seed_count)):
                    jobs.append({"group_idx": group_idx, "scheme": scheme, "basis_key": basis, "depth": depth, "part": "F3", "teacher_type": "c2_visual_synthetic", "task": task, "seed": seed})
    return jobs


def part_f_data(job: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher = str(job.get("teacher_type"))
    seed = int(job.get("seed", 0))
    if teacher == "c2_visual_synthetic":
        return visual_data(str(job.get("task")), seed, args, device)
    if teacher == "f5_no_debt_calibration":
        # Fixed held-out synthetic calibration target.  It reuses the registered
        # KAN-native generator with a disjoint seed range so F5 is not a replay
        # of F1 rows.
        return synthetic_data("c1a_shallow_additive", seed + 50000, args, device)
    return synthetic_data(teacher, seed, args, device)


def train_adamw_control(job: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    basis_key = str(job.get("basis_key"))
    depth = str(job.get("depth"))
    bargs = basis_args(args, basis_key)
    xtr, ytr, xg, yg = part_f_data(job, args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    seed = int(job.get("seed", 0))
    mseed = 2294600 + int(job.get("group_idx", 0)) * 100000 + seed * 37 + (2 if depth == "depth2" else 3)
    model = make_model(depth, int(xtr.shape[1]), classes, mseed, bargs, device)
    return v2293.train_model(model, "kan_adamw_control", "adamw", xtr, ytr, xg, yg, seed, bargs)


def train_mlp_control(job: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    basis_key = str(job.get("basis_key"))
    bargs = basis_args(args, basis_key)
    xtr, ytr, xg, yg = part_f_data(job, args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    seed = int(job.get("seed", 0))
    mseed = 2294700 + int(job.get("group_idx", 0)) * 100000 + seed * 41
    model = v2293.MatchedMLP(int(xtr.shape[1]), classes, int(args.mlp_hidden), 2, mseed, device)
    return v2293.train_model(model, "mlp_raw_control", "mlp", xtr, ytr, xg, yg, seed, bargs)


def run_part_f_row(job: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme_args = copy(args)
    if str(job.get("teacher_type")) == "f5_no_debt_calibration":
        scheme_args = copy(args)
        original_synthetic_data = synthetic_data

        def _f5_data(_teacher: str, seed: int, fargs: argparse.Namespace, fdevice: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return original_synthetic_data("c1a_shallow_additive", int(seed) + 50000, fargs, fdevice)

        xtr, ytr, xg, yg = _f5_data("f5_no_debt_calibration", int(job.get("seed", 0)), args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        basis_key = str(job.get("basis_key"))
        depth = str(job.get("depth"))
        bargs = basis_args(scheme_args, basis_key)
        mseed = 2294500 + BASIS_KEYS.index(basis_key) * 100000 + int(job.get("seed", 0)) * 31 + (2 if depth == "depth2" else 3)
        model = make_model(depth, int(xtr.shape[1]), classes, mseed, bargs, device)
        before = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        trace = {"data_gram_drift": 0.0, "op_spectrum_drift": 0.0, "task_gradient_preserved_fraction": 1.0, "debt_surrogate_loss": 0.0, "calibration_steps": 0}
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(job.get("seed", 0)))
            diag = scheme_step(model, xb, yb, step, int(args.train_steps), str(job.get("scheme")), bargs, debt_baseline=before_train)
            trace["data_gram_drift"] = max(trace["data_gram_drift"], diag["data_gram_drift"])
            trace["op_spectrum_drift"] = max(trace["op_spectrum_drift"], diag["op_spectrum_drift"])
            trace["task_gradient_preserved_fraction"] = min(trace["task_gradient_preserved_fraction"], diag["task_gradient_preserved_fraction"])
            trace["debt_surrogate_loss"] = max(trace["debt_surrogate_loss"], diag["debt_surrogate_loss"])
            trace["calibration_steps"] += int(diag["calibration_active"])
        after = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        with torch.no_grad():
            _logits, acts = model.forward_with_activations(xtr)
            act_rank = median([safe_activation_rank(a) for a in acts[1:]])
        scheme_row: dict[str, Any] = {
            "scheme": str(job.get("scheme")),
            "basis_key": basis_key,
            "depth": depth,
            "teacher_type": "f5_no_debt_calibration",
            "visual_synthetic_task": "",
            "seed": int(job.get("seed", 0)),
            "guard_NLL": after["nll"],
            "guard_accuracy": after["accuracy"],
            "guard_NLL_delta": after["nll"] - before["nll"],
            "guard_debt_before": before["debt_metric"],
            "guard_debt_after": after["debt_metric"],
            "train_debt_before": before_train["debt_metric"],
            "train_debt_after": after_train["debt_metric"],
            "train_debt_delta": after_train["debt_metric"] - before_train["debt_metric"],
            "no_debt": int(after["debt_metric"] <= before["debt_metric"] + float(args.no_debt_budget)),
            "Brier_before": before.get("brier", 0.0),
            "Brier_after": after.get("brier", 0.0),
            "Brier_delta": after.get("brier", 0.0) - before.get("brier", 0.0),
            "ECE_before": before.get("ece", 0.0),
            "ECE_after": after.get("ece", 0.0),
            "ECE_delta": after.get("ece", 0.0) - before.get("ece", 0.0),
            "tail95_before": before.get("tail95", 0.0),
            "tail95_after": after.get("tail95", 0.0),
            "tail95_delta": after.get("tail95", 0.0) - before.get("tail95", 0.0),
            "margin10_before": before.get("margin10", 0.0),
            "margin10_after": after.get("margin10", 0.0),
            "margin10_delta": after.get("margin10", 0.0) - before.get("margin10", 0.0),
            "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
            "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
            "train_tail95_delta": after_train.get("tail95", 0.0) - before_train.get("tail95", 0.0),
            "train_margin10_delta": after_train.get("margin10", 0.0) - before_train.get("margin10", 0.0),
            "visual_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "visual_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "final_visual_coverage": after["coverage_CVaR25"],
            "data_gram_drift_max": trace["data_gram_drift"],
            "op_spectrum_drift_max": trace["op_spectrum_drift"],
            "task_gradient_preserved_fraction_min": trace["task_gradient_preserved_fraction"],
            "debt_surrogate_loss_max": trace["debt_surrogate_loss"],
            "calibration_steps": trace["calibration_steps"],
            "activation_cov_rank_final": act_rank,
            "changed_w1": 1,
            **basis_audit_fields(basis_key, model),
        }
    else:
        scheme_row = train_scheme(
            str(job.get("scheme")),
            str(job.get("basis_key")),
            str(job.get("depth")),
            str(job.get("teacher_type")),
            str(job.get("task")),
            int(job.get("seed", 0)),
            scheme_args,
            device,
        )
    adamw = train_adamw_control(job, args, device)
    mlp = train_mlp_control(job, args, device)
    teacher = str(job.get("teacher_type"))
    part = str(job.get("part"))
    if part == "F1":
        primary_pass = int(fval(scheme_row.get("guard_accuracy")) >= 0.65 and fval(scheme_row.get("guard_NLL_delta")) < 0.0)
    elif part == "F2":
        primary_pass = int(fval(scheme_row.get("guard_accuracy")) >= 0.60 and fval(scheme_row.get("guard_NLL_delta")) < 0.0)
    elif part == "F3":
        primary_pass = int(fval(scheme_row.get("visual_coverage_improvement")) >= 0.05)
    elif part == "F4":
        primary_pass = int(fval(mlp.get("guard_NLL")) <= fval(scheme_row.get("guard_NLL")) + 0.05)
    elif part == "F5":
        primary_pass = ival(scheme_row.get("no_debt"))
    else:
        primary_pass = 0
    beats_adamw = int(fval(scheme_row.get("guard_NLL")) <= fval(adamw.get("guard_NLL")))
    beats_mlp = int(fval(scheme_row.get("guard_NLL")) <= fval(mlp.get("guard_NLL")))
    beats_best = int(fval(scheme_row.get("guard_NLL")) <= min(fval(adamw.get("guard_NLL")), fval(mlp.get("guard_NLL"))))
    return {
        "group_idx": int(job.get("group_idx", 0)),
        "part": part,
        "scheme": str(job.get("scheme")),
        "basis_key": str(job.get("basis_key")),
        "depth": str(job.get("depth")),
        "teacher_type": teacher,
        "visual_synthetic_task": str(job.get("task")),
        "seed": int(job.get("seed", 0)),
        "scheme_guard_NLL": scheme_row.get("guard_NLL"),
        "scheme_guard_accuracy": scheme_row.get("guard_accuracy"),
        "scheme_guard_NLL_delta": scheme_row.get("guard_NLL_delta"),
        "scheme_no_debt": scheme_row.get("no_debt", ""),
        "scheme_guard_debt_before": scheme_row.get("guard_debt_before", ""),
        "scheme_guard_debt_after": scheme_row.get("guard_debt_after", ""),
        "scheme_train_debt_before": scheme_row.get("train_debt_before", ""),
        "scheme_train_debt_after": scheme_row.get("train_debt_after", ""),
        "scheme_train_debt_delta": scheme_row.get("train_debt_delta", ""),
        "scheme_Brier_before": scheme_row.get("Brier_before", ""),
        "scheme_Brier_after": scheme_row.get("Brier_after", ""),
        "scheme_Brier_delta": scheme_row.get("Brier_delta", ""),
        "scheme_ECE_before": scheme_row.get("ECE_before", ""),
        "scheme_ECE_after": scheme_row.get("ECE_after", ""),
        "scheme_ECE_delta": scheme_row.get("ECE_delta", ""),
        "scheme_tail95_before": scheme_row.get("tail95_before", ""),
        "scheme_tail95_after": scheme_row.get("tail95_after", ""),
        "scheme_tail95_delta": scheme_row.get("tail95_delta", ""),
        "scheme_margin10_before": scheme_row.get("margin10_before", ""),
        "scheme_margin10_after": scheme_row.get("margin10_after", ""),
        "scheme_margin10_delta": scheme_row.get("margin10_delta", ""),
        "scheme_train_Brier_delta": scheme_row.get("train_Brier_delta", ""),
        "scheme_train_ECE_delta": scheme_row.get("train_ECE_delta", ""),
        "scheme_train_tail95_delta": scheme_row.get("train_tail95_delta", ""),
        "scheme_train_margin10_delta": scheme_row.get("train_margin10_delta", ""),
        "scheme_visual_coverage_improvement": scheme_row.get("visual_coverage_improvement"),
        "scheme_visual_accuracy_improvement": scheme_row.get("visual_accuracy_improvement"),
        "scheme_op_spectrum_drift_max": scheme_row.get("op_spectrum_drift_max"),
        "scheme_data_gram_drift_max": scheme_row.get("data_gram_drift_max"),
        "scheme_task_gradient_preserved_fraction_min": scheme_row.get("task_gradient_preserved_fraction_min"),
        "scheme_debt_surrogate_loss_max": scheme_row.get("debt_surrogate_loss_max", ""),
        "scheme_calibration_steps": scheme_row.get("calibration_steps", ""),
        "adamw_guard_NLL": adamw.get("guard_NLL"),
        "adamw_guard_accuracy": adamw.get("guard_accuracy"),
        "adamw_no_debt": adamw.get("no_debt"),
        "mlp_raw_guard_NLL": mlp.get("guard_NLL"),
        "mlp_raw_guard_accuracy": mlp.get("guard_accuracy"),
        "mlp_negative_ok": int(fval(mlp.get("guard_NLL")) <= fval(scheme_row.get("guard_NLL")) + 0.05) if part == "F4" else "",
        "primary_pass": primary_pass,
        "beats_AdamW": beats_adamw,
        "beats_MLP": beats_mlp,
        "beats_best_control": beats_best,
        "changed_w1": 1,
        **basis_audit_fields(str(job.get("basis_key"))),
    }


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    e = read_json(OUT_ROOT / "part_e_multi_scheme_summary.json")
    if not e.get("part_e_gate_pass"):
        route = "F_SkippedBecausePartEFailed"
        blocker = "part_e"
        reason = e.get("part_e_route", "missing_part_e")
        out = {"gate": "v22_94_part_f_deep_positive_control", "part_f_gate_pass": 0, "part_f_route": route, "dominant_blocker": blocker, "skip_reason": reason, "rows": 0}
        summary = OUT_ROOT / "part_f_deep_positive_control_summary.json"
        write_json(summary, out)
        write_rows(OUT_ROOT / "part_f_deep_positive_control_matrix.csv", [])
        next_path = write_next_actions("f", route, blocker, [{"action": "run Part F only after a fixed Part E scheme passes", "reason": str(reason)}], [rel(summary)])
        append_exec("part-f", command_text(sys.argv), "skipped", files=f"{rel(summary)}; {rel(next_path)}")
        append_recap("Part F deep positive-control", [f"gate_pass=0; route={route}; reason={reason}", "analysis: Part F is gated behind Part E; skipped parts are not success evidence."])
        return out
    device = device_from_args(args)
    rows = [run_part_f_row(job, args, device) for job in shard_items(part_f_jobs(args), args)]
    path = OUT_ROOT / f"part_f_deep_positive_control_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-f-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_deep_positive_control_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = OUT_ROOT / "part_f_deep_positive_control_matrix.csv"
    write_rows(matrix, rows)
    group_keys = sorted({(r.get("group_idx"), r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in rows}, key=lambda x: (ival(x[0]), str(x[1]), str(x[2]), str(x[3])))
    group_summaries: list[dict[str, Any]] = []
    for group_idx, scheme, basis, depth in group_keys:
        gr = [r for r in rows if r.get("group_idx") == group_idx and r.get("scheme") == scheme and r.get("basis_key") == basis and r.get("depth") == depth]
        by_part = {part: [r for r in gr if r.get("part") == part] for part in ["F1", "F2", "F3", "F4", "F5"]}
        f1_pass_rows = sum(ival(r.get("primary_pass")) for r in by_part["F1"])
        f2_pass_rows = sum(ival(r.get("primary_pass")) for r in by_part["F2"])
        f4_negative_ok_rows = sum(ival(r.get("primary_pass")) for r in by_part["F4"])
        f5_no_debt_rows = sum(ival(r.get("primary_pass")) for r in by_part["F5"])
        f3_cov_median = median([r.get("scheme_visual_coverage_improvement") for r in by_part["F3"]])
        kan_native = by_part["F1"] + by_part["F2"] + by_part["F5"]
        f1_beats_adamw = sum(ival(r.get("beats_AdamW")) for r in by_part["F1"])
        f2_beats_adamw = sum(ival(r.get("beats_AdamW")) for r in by_part["F2"])
        f5_beats_adamw = sum(ival(r.get("beats_AdamW")) for r in by_part["F5"])
        f1_beats_best = sum(ival(r.get("beats_best_control")) for r in by_part["F1"])
        f2_beats_best = sum(ival(r.get("beats_best_control")) for r in by_part["F2"])
        f5_beats_best = sum(ival(r.get("beats_best_control")) for r in by_part["F5"])
        f1_beats_mlp = sum(ival(r.get("beats_MLP")) for r in by_part["F1"])
        f2_beats_mlp = sum(ival(r.get("beats_MLP")) for r in by_part["F2"])
        f5_beats_mlp = sum(ival(r.get("beats_MLP")) for r in by_part["F5"])
        f1_gate = int(len(by_part["F1"]) >= 15 and f1_pass_rows >= 12 and f1_beats_adamw >= 12 and f1_beats_best >= 12 and f1_beats_mlp >= 12)
        f2_gate = int(len(by_part["F2"]) >= 15 and f2_pass_rows >= 12 and f2_beats_adamw >= 12 and f2_beats_best >= 12 and f2_beats_mlp >= 12)
        f3_gate = int(bool(by_part["F3"]) and f3_cov_median >= 0.05)
        f4_gate = int(len(by_part["F4"]) >= 15 and f4_negative_ok_rows >= 12)
        f5_gate = int(len(by_part["F5"]) >= 15 and f5_no_debt_rows >= 12 and f5_beats_adamw >= 12 and f5_beats_best >= 12 and f5_beats_mlp >= 12)
        gate = int(f1_gate and f2_gate and f3_gate and f4_gate and f5_gate)
        group_summaries.append(
            {
                "group_idx": ival(group_idx),
                "scheme": scheme,
                "basis_key": basis,
                "depth": depth,
                "rows": len(gr),
                "f1_pass_rows": f1_pass_rows,
                "f1_beats_AdamW_rows": f1_beats_adamw,
                "f1_beats_best_control_rows": f1_beats_best,
                "f1_beats_MLP_rows": f1_beats_mlp,
                "f1_gate_pass": f1_gate,
                "f2_pass_rows": f2_pass_rows,
                "f2_beats_AdamW_rows": f2_beats_adamw,
                "f2_beats_best_control_rows": f2_beats_best,
                "f2_beats_MLP_rows": f2_beats_mlp,
                "f2_gate_pass": f2_gate,
                "f3_visual_coverage_improvement_median": f3_cov_median,
                "f3_gate_pass": f3_gate,
                "f4_negative_ok_rows": f4_negative_ok_rows,
                "f4_gate_pass": f4_gate,
                "f5_no_debt_rows": f5_no_debt_rows,
                "f5_beats_AdamW_rows": f5_beats_adamw,
                "f5_beats_best_control_rows": f5_beats_best,
                "f5_beats_MLP_rows": f5_beats_mlp,
                "f5_gate_pass": f5_gate,
                "kan_native_rows": len(kan_native),
                "part_f_group_gate_pass": gate,
            }
        )
    pass_groups = [g for g in group_summaries if ival(g.get("part_f_group_gate_pass")) == 1]
    if pass_groups:
        route, blocker = "F_DeepPositiveControlPass", "none"
    else:
        all_f3_fail = group_summaries and not any(ival(g.get("f3_gate_pass")) for g in group_summaries)
        all_f4_fail = group_summaries and not any(ival(g.get("f4_gate_pass")) for g in group_summaries)
        all_f5_fail = group_summaries and not any(ival(g.get("f5_gate_pass")) for g in group_summaries)
        any_f4_fail = any(not ival(g.get("f4_gate_pass")) for g in group_summaries)
        any_f5_fail = any(not ival(g.get("f5_gate_pass")) for g in group_summaries)
        if all_f3_fail:
            route, blocker = "F3_VisualPositiveControlFailed", "f3_visual_overfit_or_undertransfer"
        elif all_f5_fail:
            route, blocker = "F5_NoDebtOrControlDominanceFailed", "f5_no_debt_or_controls"
        elif all_f4_fail:
            route, blocker = "F4_MLPFriendlyNegativeControlFailed", "f4_negative_control"
        elif any_f4_fail:
            route, blocker = "F_MixedNegativeControlAndNoDebtFailed", "mixed_f4_f5"
        elif any_f5_fail:
            route, blocker = "F5_NoDebtOrControlDominanceFailed", "f5_no_debt_or_controls"
        else:
            route, blocker = "F_KANNativeControlDominanceFailed", "beats_adamw_or_mlp_controls"
    out = {
        "gate": "v22_94_part_f_deep_positive_control",
        "rows": len(rows),
        "expected_rows": len(part_f_jobs(args)),
        "group_summaries": group_summaries,
        "passing_part_f_groups": pass_groups,
        "part_f_gate_pass": int(bool(pass_groups)),
        "part_f_route": route,
        "dominant_blocker": blocker,
        "changed_w1": int(any(ival(r.get("changed_w1")) for r in rows)),
    }
    summary = OUT_ROOT / "part_f_deep_positive_control_summary.json"
    write_json(summary, out)
    next_path = write_next_actions(
        "f",
        route,
        blocker,
        [] if pass_groups else [{"action": "analyze Part E passing groups against F1-F5 controls before any real-task preflight", "reason": blocker}],
        [rel(matrix), rel(summary)],
    )
    append_exec("part-f-merge", command_text(sys.argv), "done" if pass_groups else "failed", files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part F deep positive-control",
        [
            f"gate_pass={int(bool(pass_groups))}; route={route}; rows={len(rows)}/{len(part_f_jobs(args))}; passing_groups={len(pass_groups)}",
            f"group_summaries={json.dumps(group_summaries, sort_keys=True)}",
            "analysis: Part F evaluates every Part E passing fixed group independently on F1-F5. Passing/failing groups are not runtime-selected rows; no skipped row is success evidence.",
        ],
    )
    return out


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    f = read_json(OUT_ROOT / "part_f_deep_positive_control_summary.json")
    route = "G_SkippedBecausePartFFailed"
    reason = f.get("part_f_route", "missing_part_f")
    out = {"gate": "v22_94_part_g_real_task_preflight", "part_g_gate_pass": 0, "part_g_route": route, "skip_reason": reason, "rows": 0}
    summary = OUT_ROOT / "part_g_real_task_preflight_summary.json"
    write_json(summary, out)
    write_rows(OUT_ROOT / "part_g_real_task_preflight_matrix.csv", [])
    next_path = write_next_actions("g", route, "part_f", [{"action": "run real-task preflight only after Part F passes", "reason": str(reason)}], [rel(summary)])
    append_exec("part-g", command_text(sys.argv), "skipped", files=f"{rel(summary)}; {rel(next_path)}")
    append_recap("Part G real-task preflight", [f"gate_pass=0; route={route}; reason={reason}", "analysis: real-task preflight remains blocked unless Part F passes."])
    return out


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    g = read_json(OUT_ROOT / "part_g_real_task_preflight_summary.json")
    route = "H_SkippedBecausePartGFailed"
    reason = g.get("part_g_route", "missing_part_g")
    out = {"gate": "v22_94_part_h_limited_full_loop", "part_h_gate_pass": 0, "part_h_route": route, "skip_reason": reason, "rows": 0}
    summary = OUT_ROOT / "part_h_limited_full_loop_summary.json"
    write_json(summary, out)
    next_path = write_next_actions("h", route, "part_g", [{"action": "run H-step only after Part G passes", "reason": str(reason)}], [rel(summary)])
    append_exec("part-h", command_text(sys.argv), "skipped", files=f"{rel(summary)}; {rel(next_path)}")
    append_recap("Part H limited full-loop", [f"gate_pass=0; route={route}; reason={reason}", "analysis: H-step cannot be used as a repair for earlier gate failures."])
    return out


def run_part_i(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    parts = {
        "a": read_json(OUT_ROOT / "part_a_identity.json"),
        "b": read_json(OUT_ROOT / "part_b_history_lock.json"),
        "c": read_json(OUT_ROOT / "part_c_witness_summary.json"),
        "d": read_json(OUT_ROOT / "part_d_decomposition_summary.json"),
        "e": read_json(OUT_ROOT / "part_e_multi_scheme_summary.json"),
        "f": read_json(OUT_ROOT / "part_f_deep_positive_control_summary.json"),
        "g": read_json(OUT_ROOT / "part_g_real_task_preflight_summary.json"),
        "h": read_json(OUT_ROOT / "part_h_limited_full_loop_summary.json"),
    }
    lines = ["# v22.94 failure decomposition and next actions\n\n"]
    for part, data in parts.items():
        route = data.get(f"part_{part}_route", data.get(f"part_{part}_gate_pass", "missing"))
        lines.append(f"- Part {part.upper()}: route=`{route}`, gate=`{data.get(f'part_{part}_gate_pass', 'missing')}`, blocker=`{data.get('dominant_blocker', data.get('skip_reason', 'missing'))}`\n")
    c = parts["c"]
    d = parts["d"]
    e = parts["e"]
    f = parts["f"]
    f_groups = f.get("group_summaries", [])
    if not isinstance(f_groups, list):
        f_groups = []
    f3_best = max([fval(g.get("f3_visual_coverage_improvement_median")) for g in f_groups] or [0.0])
    f4_best = max([ival(g.get("f4_negative_ok_rows")) for g in f_groups] or [0])
    f5_best = max([ival(g.get("f5_no_debt_rows")) for g in f_groups] or [0])
    lines.extend(
        [
            "\n## Evidence chain\n",
            f"- Part C AdamW witness: acc improvement median `{c.get('adamw_visual_accuracy_improvement_median', 'missing')}`, coverage improvement median `{c.get('adamw_visual_coverage_improvement_median', 'missing')}`, metric coverage improvement median `{c.get('metric_branch_visual_coverage_improvement_median', 'missing')}`.\n",
            f"- Part D decomposition: explained fraction median `{d.get('explained_gain_fraction_median', 'missing')}`, residual median `{d.get('decomposition_numerical_residual_median', 'missing')}`, deleted replay error median `{d.get('deleted_component_replay_error_median', 'missing')}`.\n",
            f"- Part E route: `{e.get('part_e_route', 'missing')}`; passing scheme groups `{len(e.get('passing_scheme_groups', []) if isinstance(e.get('passing_scheme_groups'), list) else [])}`.\n",
            f"- Part F route: `{f.get('part_f_route', 'missing')}`; passing groups `{len(f.get('passing_part_f_groups', []) if isinstance(f.get('passing_part_f_groups'), list) else [])}`; best F3 coverage median `{f3_best}`; best F4 negative-ok rows `{f4_best}/15`; best F5 no-debt rows `{f5_best}/15`.\n",
            "- No skipped Part F/G/H artifact is treated as success evidence.\n",
        ]
    )
    path = OUT_ROOT / "part_i_failure_decomposition.md"
    path.write_text("".join(lines), encoding="utf-8")
    out = {"gate": "v22_94_part_i_closeout", "part_i_written": 1, "part_i_path": rel(path)}
    write_json(OUT_ROOT / "part_i_summary.json", out)
    append_exec("part-i", command_text(sys.argv), "done", files=f"{rel(path)}; {rel(OUT_ROOT / 'part_i_summary.json')}")
    append_recap("Part I closeout", [f"written={rel(path)}", "analysis: closeout records blocker route and next actions without inventing missing data."])
    return out


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    c = read_json(OUT_ROOT / "part_c_witness_summary.json")
    d = read_json(OUT_ROOT / "part_d_decomposition_summary.json")
    e = read_json(OUT_ROOT / "part_e_multi_scheme_summary.json")
    f = read_json(OUT_ROOT / "part_f_deep_positive_control_summary.json")
    g = read_json(OUT_ROOT / "part_g_real_task_preflight_summary.json")
    h = read_json(OUT_ROOT / "part_h_limited_full_loop_summary.json")
    if not a.get("part_a_gate_pass"):
        route, reason = "A_FailedIdentity", "part_a"
    elif not b.get("part_b_gate_pass"):
        route, reason = "B_HistoryLockFailed", "part_b"
    elif not c.get("part_c_gate_pass"):
        route, reason = c.get("part_c_route", "A_WitnessMissing"), c.get("dominant_blocker", "part_c")
    elif not d.get("part_d_gate_pass"):
        route, reason = d.get("part_d_route", "D_DecompositionUnexplained"), d.get("dominant_blocker", "part_d")
    elif not e.get("part_e_gate_pass"):
        route, reason = e.get("part_e_route", "E_AllSchemesFail"), e.get("dominant_blocker", "part_e")
    elif not f.get("part_f_gate_pass"):
        route, reason = f.get("part_f_route", "C2PassButPositiveControlMissing"), f.get("dominant_blocker", "part_f")
    elif not g.get("part_g_gate_pass"):
        route, reason = g.get("part_g_route", "G_RealTaskFailed"), g.get("skip_reason", "part_g")
    elif not h.get("part_h_gate_pass"):
        route, reason = h.get("part_h_route", "H_LimitedLoopFailed"), h.get("skip_reason", "part_h")
    else:
        route, reason = "OfficialCandidateGatePass", "all_gates_passed"
    out = {
        "gate": "v22_94_final_route",
        "official_candidate_gate_pass": int(route == "OfficialCandidateGatePass"),
        "final_route": route,
        "route_reason": reason,
        "part_a": {"part_a_gate_pass": a.get("part_a_gate_pass")},
        "part_b": {"part_b_gate_pass": b.get("part_b_gate_pass")},
        "part_c": {"part_c_gate_pass": c.get("part_c_gate_pass"), "part_c_route": c.get("part_c_route"), "dominant_blocker": c.get("dominant_blocker")},
        "part_d": {"part_d_gate_pass": d.get("part_d_gate_pass"), "part_d_route": d.get("part_d_route"), "dominant_blocker": d.get("dominant_blocker")},
        "part_e": {"part_e_gate_pass": e.get("part_e_gate_pass"), "part_e_route": e.get("part_e_route"), "passing_scheme_groups": e.get("passing_scheme_groups", [])},
        "part_f": {"part_f_gate_pass": f.get("part_f_gate_pass"), "part_f_route": f.get("part_f_route")},
        "part_g": {"part_g_gate_pass": g.get("part_g_gate_pass"), "part_g_route": g.get("part_g_route")},
        "part_h": {"part_h_gate_pass": h.get("part_h_gate_pass"), "part_h_route": h.get("part_h_route")},
    }
    path = OUT_ROOT / "final_route.json"
    write_json(path, out)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        f"# v22.94 reproduction manifest\n\n"
        f"Python: `{PYTHON}`\n"
        f"Runner: `{rel(RUNNER)}`\n"
        f"Plan: `{rel(PLAN)}`\n"
        f"Output: `{rel(OUT_ROOT)}`\n\n"
        "## Typical commands\n\n"
        f"- Part A: `{PYTHON} {rel(RUNNER)} --mode part-a --device cuda`\n"
        f"- Part B: `{PYTHON} {rel(RUNNER)} --mode part-b --device cuda`\n"
        f"- Part C shard: `CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-c --device cuda --shard-count 4 --shard-index 0`\n"
        f"- Part C merge: `{PYTHON} {rel(RUNNER)} --mode part-c-merge --device cuda`\n"
        f"- Part D: `{PYTHON} {rel(RUNNER)} --mode part-d --device cuda`\n"
        f"- Part E repair shard: `CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-e --device cuda --shard-count 4 --shard-index 0 --train-steps 120 --metric-lr 0.1 --scheme-max-norm-ratio 4.0`\n"
        f"- Part E repair merge: `{PYTHON} {rel(RUNNER)} --mode part-e-merge --device cuda --train-steps 120 --metric-lr 0.1 --scheme-max-norm-ratio 4.0`\n"
        f"- Part F shard: `CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-f --device cuda --shard-count 4 --shard-index 0 --train-steps 120 --metric-lr 0.1 --scheme-max-norm-ratio 4.0 --part-f-seed-count 15`\n"
        f"- Part F merge: `{PYTHON} {rel(RUNNER)} --mode part-f-merge --device cuda --train-steps 120 --metric-lr 0.1 --scheme-max-norm-ratio 4.0 --part-f-seed-count 15`\n"
        f"- Part G/H/I: `{PYTHON} {rel(RUNNER)} --mode part-g --device cuda`; `{PYTHON} {rel(RUNNER)} --mode part-h --device cuda`; `{PYTHON} {rel(RUNNER)} --mode part-i --device cuda`\n"
        f"- Finalize: `{PYTHON} {rel(RUNNER)} --mode finalize --device cuda`\n",
        encoding="utf-8",
    )
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(path)}; {rel(manifest)}")
    append_recap(
        "Final route",
        [
            f"official_candidate_gate_pass={out['official_candidate_gate_pass']}; final_route={route}; reason={reason}",
            "analysis: Final route is based only on current artifacts. Missing/skipped/failing Parts are not success evidence.",
        ],
    )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--pc-basis", default="D-FOU")
    p.add_argument("--basis-k-override", type=int, default=0)
    p.add_argument("--dfour-k", type=int, default=3)
    p.add_argument("--input-dim", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--deep-width", type=int, default=12)
    p.add_argument("--mlp-hidden", type=int, default=24)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--c-condition-budget", type=float, default=1.0e5)
    p.add_argument("--metric-lr", type=float, default=0.05)
    p.add_argument("--metric-update-mode", default="projected_metric")
    p.add_argument("--metric-projected-adamw-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=0.02)
    p.add_argument("--mlp-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metric-max-norm-ratio", type=float, default=2.0)
    p.add_argument("--metric-shape-scale-branch", type=int, default=1)
    p.add_argument("--shape-radial-fraction-cap", type=float, default=2.0)
    p.add_argument("--metric-tube-gradient-residual", type=int, default=1)
    p.add_argument("--metric-tube-combine-tangent-residual", type=int, default=0)
    p.add_argument("--metric-tube-adamw-residual", type=int, default=0)
    p.add_argument("--metric-tube-adamw-beta1", type=float, default=0.90)
    p.add_argument("--metric-tube-adamw-beta2", type=float, default=0.999)
    p.add_argument("--metric-tube-adamw-eps", type=float, default=1.0e-8)
    p.add_argument("--metric-tube-adamw-norm-match", type=int, default=1)
    p.add_argument("--metric-tube-drift-budget", type=float, default=0.01)
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--no-debt-budget", type=float, default=0.01)
    p.add_argument("--synthetic-train-size", type=int, default=192)
    p.add_argument("--synthetic-guard-size", type=int, default=128)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-seed-count", type=int, default=3)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--checkpoint-steps", default="0,1,5,10,20,40,80")
    p.add_argument("--decomp-steps", default="1,5,10,20,40,80")
    p.add_argument("--replay-alphas", default="0.02,0.08")
    p.add_argument("--replay-betas", default="0.25,0.5,1.0")
    p.add_argument("--mode-lambda-partial", type=float, default=1.0e-4)
    p.add_argument("--mode-lambda-partial2", type=float, default=1.0e-6)
    p.add_argument("--mode-lambda-omega", type=float, default=1.0e-4)
    p.add_argument("--part-e-schemes", default="op_strict,op_tube,formation_20_op_tube,formation_35_op_tube,cross_layer_coupled_proxy,mode_band_freedom,activation_domain_freedom")
    p.add_argument("--part-e-depths", default="depth2,depth3")
    p.add_argument("--part-e-seed-count", type=int, default=3)
    p.add_argument("--part-f-seed-count", type=int, default=15)
    p.add_argument("--part-f-parts", default="F1,F2,F3,F4,F5")
    p.add_argument("--scheme-max-norm-ratio", type=float, default=1.0)
    p.add_argument("--scheme-op-normal-scale", type=float, default=0.10)
    p.add_argument("--scheme-calibration-weight", type=float, default=0.50)
    p.add_argument("--scheme-calibration-ece-weight", type=float, default=1.0)
    p.add_argument("--scheme-calibration-start-frac", type=float, default=0.50)
    p.add_argument("--scheme-calibration-tail-weight", type=float, default=4.0)
    p.add_argument("--scheme-calibration-margin-weight", type=float, default=4.0)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode)
    if mode == "part-a":
        return run_part_a(args)
    if mode == "part-b":
        return run_part_b(args)
    if mode == "part-c":
        return run_part_c(args)
    if mode == "part-c-merge":
        return merge_part_c(args)
    if mode == "part-d":
        return run_part_d(args)
    if mode == "part-e":
        return run_part_e(args)
    if mode == "part-e-merge":
        return merge_part_e(args)
    if mode == "part-f":
        return run_part_f(args)
    if mode == "part-f-merge":
        return merge_part_f(args)
    if mode == "part-g":
        return run_part_g(args)
    if mode == "part-h":
        return run_part_h(args)
    if mode == "part-i":
        return run_part_i(args)
    if mode == "finalize":
        return finalize(args)
    if mode == "all":
        run_part_a(args)
        run_part_b(args)
        run_part_c(args)
        merge_part_c(args)
        run_part_d(args)
        run_part_e(args)
        merge_part_e(args)
        run_part_f(args)
        merge_part_f(args)
        run_part_g(args)
        run_part_h(args)
        run_part_i(args)
        return finalize(args)
    raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
