#!/usr/bin/env python3
"""DG-KAN v23.07 edge-function residual inverse flow runner."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import math
import os
import py_compile
import re
import sys
import time
import tokenize
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
from dgkan.fu.edge_sobolev_metrics import functional_edge_gram
from dgkan.models.fc_purekan_primitives import _basis_eval
from dgkan.optim.edge_sobolev_population_flow import EdgeSobolevPopulationFlow, mark_kan_edge_params


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.07_EdgeFunctionResidualInverseFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.07_EdgeFunctionResidualInverseFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.07_EdgeFunctionResidualInverseFlow_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2307_OUT_ROOT", str(ROOT / "results/v23_07"))).resolve()

AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "new_edge_function_added": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "structure_transform_in_forward": 0,
}

PART_B_SCHEMES = [
    "B0_ordinary_gradient",
    "B1_preconditioned_gradient",
    "B2_EFRF_L2",
    "B3_EFRF_s025",
    "B4_EFRF_population_weighted",
    "B5_random_projected_EFRF",
    "B6_shuffled_design_EFRF",
    "B7_H10_structure_weighted_diagnostic",
    "B8_EFRF_identity_G_diagnostic",
]

EFRF_PART_B_CANDIDATES = {"B2_EFRF_L2", "B3_EFRF_s025", "B4_EFRF_population_weighted"}
CONTROL_PART_B_RANDOM = "B5_random_projected_EFRF"
CONTROL_PART_B_SHUFFLE = "B6_shuffled_design_EFRF"

PART_C_SCHEMES = [
    "C0_ordinary_gradient_actual",
    "C1_EFRF_L2_last_layer",
    "C2_EFRF_L2_all_layers",
    "C3_EFRF_L2_gauss_seidel",
    "C4_EFRF_s025_gauss_seidel",
    "C5_EFRF_Wpop_gauss_seidel",
    "C6_random_projected_EFRF",
    "C8_same_compute_noop",
    "C9_FunctionalGram_actual_step",
]
EFRF_PART_C_CANDIDATES = {
    "C1_EFRF_L2_last_layer",
    "C2_EFRF_L2_all_layers",
    "C3_EFRF_L2_gauss_seidel",
    "C4_EFRF_s025_gauss_seidel",
    "C5_EFRF_Wpop_gauss_seidel",
}
CONTROL_PART_C_GRADIENT = "C0_ordinary_gradient_actual"
CONTROL_PART_C_RANDOM = "C6_random_projected_EFRF"
CONTROL_PART_C_NOOP = "C8_same_compute_noop"
CONTROL_PART_C_FUNCTIONALGRAM = "C9_FunctionalGram_actual_step"

PART_D_SCHEMES = [
    "D0_AdamW_coefficient_baseline",
    "D1_FunctionalGram_AdamW",
    "D2_FunctionalGram_BlockSNR_population_proxy",
    "D3_H10_known_structure_synthetic_control_proxy",
    "D4_EFRF_L2_identityW_gauss_seidel",
    "D5_EFRF_s025_identityW_gauss_seidel",
    "D6_EFRF_L2_Wpop_gauss_seidel",
    "D7_EFRF_s025_Wpop_gauss_seidel",
    "D8_EFRF_random_projected_residual_flow",
    "D9_EFRF_shuffled_design_residual_flow",
    "D10_EFRF_identity_G_diagnostic",
    "D11_same_compute_noop",
]
EFRF_PART_D_CANDIDATES = {
    "D4_EFRF_L2_identityW_gauss_seidel",
    "D5_EFRF_s025_identityW_gauss_seidel",
    "D6_EFRF_L2_Wpop_gauss_seidel",
    "D7_EFRF_s025_Wpop_gauss_seidel",
}
CONTROL_PART_D_FUNCTIONALGRAM = "D1_FunctionalGram_AdamW"
CONTROL_PART_D_BLOCKSNR = "D2_FunctionalGram_BlockSNR_population_proxy"
CONTROL_PART_D_H10 = "D3_H10_known_structure_synthetic_control_proxy"
CONTROL_PART_D_RANDOM = "D8_EFRF_random_projected_residual_flow"
CONTROL_PART_D_NOOP = "D11_same_compute_noop"

FINAL_ROUTE_ALIASES = {
    "TaskwiseC2Failed": "D_C2TaskwiseFailed",
    "ResidualFlowUnsafe": "D_ResidualFlowUnsafe",
}


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    RECAP_LOG.parent.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "x"


def command_text(argv: Iterable[str]) -> str:
    cmd = " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])
    env = os.environ.get("V2307_OUT_ROOT")
    return f"V2307_OUT_ROOT={env} {cmd}" if env else cmd


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else float(default)


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def csv_items(text: Any) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def float_items(text: Any) -> list[float]:
    out: list[float] = []
    for item in csv_items(text):
        try:
            out.append(float(item))
        except Exception:
            pass
    return out or [1.0e-2]


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
    payload = {**AUDIT_DEFAULTS, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
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
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow({k: row.get(k, "") for k in keys})
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.07 EdgeFunctionResidualInverseFlow 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- Python：`{PYTHON}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误和真实指标；缺失写 `missing`，跳过写 `skipped`，不补造。\n"
            "- 复现提示：Part B 支持 `--shard-count/--shard-index`；本轮优先使用 GPU 2/3 并行。\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.07 EdgeFunctionResidualInverseFlow 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n尚未 final。本文件只记录真实 artifact、真实指标、实际修复与分析，不补造数据。\n\n",
            encoding="utf-8",
        )


def append_exec(part: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {part} {status}\n\n")
        fh.write(f"- command: `{command}`\n")
        if gpu:
            fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def next_actions(part: str, gate: int, blocker: str, actions: list[str], rerun_commands: list[str] | None = None) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part.upper(),
            "blocked": int(not bool(gate)),
            "gate_pass": int(gate),
            "dominant_blocker": blocker,
            "allowed_next_actions": actions,
            "forbidden_actions": [
                "do_not_use_held_or_test_selector",
                "do_not_add_new_edge_function",
                "do_not_add_mlp_stem_or_readout",
                "do_not_drop_random_or_shuffled_controls",
                "do_not_lower_gate_thresholds_after_seeing_results",
                "do_not_promote_rowwise_or_seedwise_winners",
            ],
            "rerun_commands": rerun_commands or [],
            "promotion_allowed": int(bool(gate)),
        },
    )


def strip_code(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING}:
                out.append("\n" if token.type in {tokenize.NL, tokenize.NEWLINE} else " ")
            else:
                out.append(token.string)
    except tokenize.TokenError:
        return text
    return "".join(out)


def static_identity_scan(files: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_winner_selection": re.compile(r"\b(best_|winner_|select_best|choose_best|argmax)\b"),
        "held_test_induction": re.compile(r"\b(held|test)\b.*\b(select|choose|gate|lambda|rank|metric)\b", re.IGNORECASE),
        "new_edge_function_hint": re.compile(r"\b(new_edge|bspline|product_edge|convkan|mlp_readout|mlp_stem)\b", re.IGNORECASE),
    }
    hits: list[dict[str, str]] = []
    for path in files:
        code = strip_code(path)
        for check, pattern in patterns.items():
            match = pattern.search(code)
            if match:
                hits.append({"file": rel(path), "check": check, "match": match.group(0)})
    serious = [h for h in hits if h["check"] in {"held_test_induction", "new_edge_function_hint"}]
    return int(not serious), hits


def basis_for_key(basis_key: str) -> tuple[str, int]:
    key = str(basis_key).lower()
    if "four" in key or "fou" in key:
        return "fourier_lowfreq", 9 if "9" in key else 5
    match = re.search(r"k(\d+)", key)
    return "chebyshev", int(match.group(1)) if match else 9


def make_model(
    basis_key: str,
    depth: int,
    input_dim: int,
    output_dim: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    basis_input_gain: float | None = None,
) -> v2293.TrueDeepPureKAN:
    basis_name, k = basis_for_key(basis_key)
    model = v2293.TrueDeepPureKAN(
        int(input_dim),
        int(output_dim),
        int(args.width),
        int(depth),
        basis_name,
        int(k),
        int(seed),
        device,
        basis_input_gain=float(args.basis_input_gain if basis_input_gain is None else basis_input_gain),
    ).to(device)
    mark_kan_edge_params(model, basis_key=basis_key)
    return model


def visual_data(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2293.visual_synthetic_task(
        task,
        int(seed),
        int(args.train_size),
        int(args.guard_size),
        int(args.visual_side),
        int(args.num_classes),
        device,
        fixed_patch_features=bool(int(args.visual_fixed_patch_features)),
        task_version=str(args.visual_task_version),
    )


def iter_batch(x: torch.Tensor, y: torch.Tensor, step: int, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    return v2293.v2289.iter_train_batches(x, y, int(step), int(batch_size), int(seed))


def train_checkpoint(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    checkpoint: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if checkpoint == "random_init":
        return {"checkpoint_train_steps": 0, "checkpoint_optimizer": "none"}
    steps = int(args.checkpoint_steps)
    batch = int(args.batch_size)
    if checkpoint == "functionalgram_partial":
        opt = EdgeSobolevPopulationFlow(
            list(model.coeffs),
            lr=float(args.checkpoint_lr),
            weight_decay=float(args.weight_decay),
            sobolev_exponent=0.0,
            edge_metric_type="functional_gram",
            edge_weight_normalization="trace",
            edge_weight_ridge=float(args.functional_gram_ridge),
            functional_gram_quadrature_points=int(args.quadrature_points),
            use_population_gate=False,
        )
        opt_name = "EdgeSobolevPopulationFlow(functional_gram_s0)"
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.checkpoint_lr), weight_decay=float(args.weight_decay))
        opt_name = "AdamW"
    for step in range(steps):
        xb, yb = iter_batch(x, y, step, batch, seed)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb.long())
        loss.backward()
        opt.step()
    return {"checkpoint_train_steps": steps, "checkpoint_optimizer": opt_name}


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def eig_condition(x: torch.Tensor, eps: float = 1.0e-12) -> tuple[float, float, float]:
    if int(x.numel()) == 0:
        return 0.0, 0.0, 0.0
    vals = torch.linalg.eigvalsh(sym(x).to(dtype=torch.float64))
    mn = float(vals.min().detach().cpu().item())
    mx = float(vals.max().detach().cpu().item())
    if mn <= eps:
        return mn, mx, float("inf")
    return mn, mx, float(mx / max(mn, eps))


def expand_edge_gram(model: v2293.TrueDeepPureKAN, layer_idx: int, basis_key: str, sobolev_order: float, args: argparse.Namespace, *, device: torch.device) -> torch.Tensor:
    in_dim = int(model.dims[int(layer_idx)])
    k = int(model.k)
    gram = functional_edge_gram(
        basis_key,
        k,
        sobolev_order=float(sobolev_order),
        quadrature_points=int(args.quadrature_points),
        normalization="trace",
        ridge=float(args.functional_gram_ridge),
        device=device,
        dtype=torch.float64,
    )
    eye = torch.eye(in_dim, device=device, dtype=torch.float64)
    return torch.kron(eye, gram)


def solve_exact(K: torch.Tensor, B: torch.Tensor, *, jitter: float = 1.0e-10) -> tuple[torch.Tensor, dict[str, Any]]:
    kk = sym(K).to(dtype=torch.float64)
    rhs = B.to(dtype=torch.float64)
    eye = torch.eye(int(kk.shape[0]), device=kk.device, dtype=kk.dtype)
    used = 0.0
    for attempt in range(8):
        try:
            chol = torch.linalg.cholesky(kk + used * eye)
            sol = torch.cholesky_solve(rhs, chol)
            resid = float(((kk + used * eye) @ sol - rhs).norm().div(rhs.norm().clamp_min(1.0e-12)).detach().cpu().item())
            return sol, {"cholesky_success": 1, "jitter_used": used, "solve_residual": resid, "solve_attempts": attempt + 1}
        except RuntimeError:
            used = float(jitter) if used == 0.0 else used * 10.0
    sol = torch.linalg.solve(kk + used * eye, rhs)
    resid = float(((kk + used * eye) @ sol - rhs).norm().div(rhs.norm().clamp_min(1.0e-12)).detach().cpu().item())
    return sol, {"cholesky_success": 0, "jitter_used": used, "solve_residual": resid, "solve_attempts": 8}


def solve_cg(K: torch.Tensor, B: torch.Tensor, *, tol: float = 1.0e-6, max_iter: int = 512) -> tuple[torch.Tensor, dict[str, Any]]:
    kk = sym(K).to(dtype=torch.float64)
    rhs = B.to(dtype=torch.float64)
    x = torch.zeros_like(rhs)
    r = rhs - kk @ x
    p = r.clone()
    rsold = (r * r).sum(dim=0).clamp_min(1.0e-30)
    rhs_norm = rhs.norm(dim=0).clamp_min(1.0e-12)
    it = 0
    for it in range(1, int(max_iter) + 1):
        ap = kk @ p
        alpha = rsold / (p * ap).sum(dim=0).clamp_min(1.0e-30)
        x = x + p * alpha.reshape(1, -1)
        r = r - ap * alpha.reshape(1, -1)
        rsnew = (r * r).sum(dim=0).clamp_min(1.0e-30)
        rel = (rsnew.sqrt() / rhs_norm).max()
        if float(rel.detach().cpu().item()) <= float(tol):
            rsold = rsnew
            break
        beta = rsnew / rsold
        p = r + p * beta.reshape(1, -1)
        rsold = rsnew
    resid = float((kk @ x - rhs).norm().div(rhs.norm().clamp_min(1.0e-12)).detach().cpu().item())
    return x, {"cg_iterations": it, "cg_residual": resid}


def solve_blockdiag(K: torch.Tensor, B: torch.Tensor, block: int) -> tuple[torch.Tensor, dict[str, Any]]:
    kk = sym(K).to(dtype=torch.float64)
    rhs = B.to(dtype=torch.float64)
    out = torch.zeros_like(rhs)
    blocks = 0
    for start in range(0, int(kk.shape[0]), int(block)):
        end = min(start + int(block), int(kk.shape[0]))
        sol, _diag = solve_exact(kk[start:end, start:end], rhs[start:end])
        out[start:end] = sol
        blocks += 1
    resid = float((kk @ out - rhs).norm().div(rhs.norm().clamp_min(1.0e-12)).detach().cpu().item())
    return out, {"blockdiag_blocks": blocks, "blockdiag_residual_full_K": resid}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).to(dtype=torch.float64)
    bb = b.reshape(-1).to(dtype=torch.float64)
    denom = aa.norm() * bb.norm()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    return float((aa @ bb).div(denom).detach().cpu().item())


def solve_normal(
    phi: torch.Tensor,
    residual: torch.Tensor,
    g_edge: torch.Tensor,
    lam: float,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    w = torch.ones(int(phi.shape[0]), device=phi.device, dtype=torch.float64) if weights is None else weights.to(device=phi.device, dtype=torch.float64).reshape(-1)
    wp = phi.to(dtype=torch.float64) * w.sqrt().reshape(-1, 1)
    wr = residual.to(dtype=torch.float64) * w.reshape(-1, 1)
    design = wp.T @ wp
    rhs = phi.to(dtype=torch.float64).T @ wr
    _mn0, _mx0, cond0 = eig_condition(design)
    K = sym(design + float(lam) * g_edge.to(device=phi.device, dtype=torch.float64))
    mn, mx, cond = eig_condition(K)
    sol, diag = solve_exact(K, rhs)
    diag.update(
        {
            "condition_number_before_ridge": cond0,
            "condition_number_after_ridge": cond,
            "min_eig_after_ridge": mn,
            "max_eig_after_ridge": mx,
            "lambda_used": float(lam),
        }
    )
    return sol, diag


def class_population_weights(residual: torch.Tensor, labels: torch.Tensor, beta: float, mode: str = "sample_coherence") -> torch.Tensor:
    rr = residual.detach().to(dtype=torch.float64)
    y = labels.detach().reshape(-1)
    weights = torch.ones(int(rr.shape[0]), device=rr.device, dtype=torch.float64)
    mode_key = str(mode).lower()
    for cls in sorted(set(int(v) for v in y.detach().cpu().tolist())):
        idx = (y == int(cls)).nonzero(as_tuple=False).reshape(-1)
        if int(idx.numel()) <= 1:
            continue
        block = rr[idx]
        mu = block.mean(dim=0)
        centered = block - mu
        if mode_key == "class_snr":
            denom = centered.square().sum().div(float(max(1, int(idx.numel()) - 1))).clamp_min(1.0e-12)
            snr = mu.square().sum() / denom
            weights[idx] = torch.sigmoid(float(beta) * (snr - 1.0))
        else:
            signal = mu.square().mean().clamp_min(1.0e-12)
            noise = centered.square().mean(dim=1).clamp_min(1.0e-12)
            direction = (block * mu.reshape(1, -1)).sum(dim=1) / (block.norm(dim=1) * mu.norm()).clamp_min(1.0e-12)
            coherence = signal / noise * direction.clamp_min(0.0).square()
            weights[idx] = torch.sigmoid(float(beta) * (coherence - 1.0))
    return weights.clamp(0.05, 1.0)


def residual_bundle(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    residual_target: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    logits, acts = model.forward_with_activations(x)
    for act in acts[1:]:
        act.retain_grad()
    target = str(residual_target).lower()
    if target == "gn":
        prob = torch.softmax(logits.float(), dim=1)
        onehot = F.one_hot(y.long(), num_classes=int(logits.shape[1])).to(dtype=prob.dtype, device=prob.device)
        rz = (onehot - prob).to(dtype=logits.dtype) / float(max(1, int(x.shape[0])))
        torch.autograd.backward(logits, grad_tensors=rz)
        loss_value = float(F.cross_entropy(logits.float(), y.long()).detach().cpu().item())
        residuals = [act.grad.detach().clone().to(dtype=torch.float64) for act in acts[1:]]
    else:
        loss = F.cross_entropy(logits.float(), y.long())
        loss.backward()
        loss_value = float(loss.detach().cpu().item())
        residuals = [(-act.grad.detach().clone()).to(dtype=torch.float64) for act in acts[1:]]
        if target == "norm":
            residuals = [r / r.square().mean().sqrt().clamp_min(1.0e-12) for r in residuals]
    return [a.detach() for a in acts], residuals, logits.detach(), loss_value


def improvement_metrics(phi: torch.Tensor, residual: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    pred = phi.to(dtype=torch.float64) @ delta.to(dtype=torch.float64)
    before = residual.to(dtype=torch.float64).square().mean().clamp_min(1.0e-12)
    after = (residual.to(dtype=torch.float64) - pred).square().mean()
    align = cosine(pred, residual)
    return {
        "residual_mse_before": float(before.detach().cpu().item()),
        "residual_mse_after": float(after.detach().cpu().item()),
        "fit_improvement_ratio": float(((before - after) / before).detach().cpu().item()),
        "residual_alignment_cosine": align,
        "predicted_loss_delta": float((-(residual.to(dtype=torch.float64) * pred).mean()).detach().cpu().item()),
        "activation_residual_norm": float(residual.norm().detach().cpu().item()),
    }


def mode_energy(delta: torch.Tensor, k: int) -> tuple[float, float]:
    d = delta.detach().to(dtype=torch.float64)
    if int(d.numel()) == 0:
        return 0.0, 0.0
    rows = int(d.shape[0]) // int(k)
    shaped = d[: rows * int(k)].reshape(rows, int(k), -1)
    energy = shaped.square().sum().clamp_min(1.0e-12)
    split = max(1, int(math.ceil(int(k) / 3.0)))
    low = shaped[:, :split, :].square().sum() / energy
    high = shaped[:, -split:, :].square().sum() / energy
    return float(low.detach().cpu().item()), float(high.detach().cpu().item())


def effective_rank(x: torch.Tensor) -> float:
    xx = x.detach().to(dtype=torch.float64)
    if int(xx.ndim) == 1:
        xx = xx.reshape(-1, 1)
    if min(int(xx.shape[0]), int(xx.shape[1])) <= 1:
        return 1.0
    centered = xx - xx.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / float(max(1, int(xx.shape[0]) - 1))
    vals = torch.linalg.eigvalsh(sym(cov)).clamp_min(0.0)
    total = vals.sum()
    if float(total.detach().cpu().item()) <= 1.0e-12:
        return 0.0
    p = vals / total
    entropy = -(p[p > 0] * p[p > 0].log()).sum()
    return float(torch.exp(entropy).detach().cpu().item())


def activation_diagnostics(act: torch.Tensor, basis_input_gain: float) -> dict[str, float]:
    aa = act.detach().to(dtype=torch.float64)
    zz = aa * float(basis_input_gain)
    tt = torch.tanh(zz)
    return {
        "activation_rms": float(aa.square().mean().sqrt().detach().cpu().item()),
        "activation_abs_mean": float(aa.abs().mean().detach().cpu().item()),
        "activation_effective_rank": effective_rank(aa.reshape(int(aa.shape[0]), -1)),
        "basis_input_abs_mean": float(zz.abs().mean().detach().cpu().item()),
        "basis_input_tanh_abs_mean": float(tt.abs().mean().detach().cpu().item()),
        "basis_input_tanh_saturation_frac_095": float((tt.abs() >= 0.95).to(dtype=torch.float64).mean().detach().cpu().item()),
        "basis_input_tanh_saturation_frac_099": float((tt.abs() >= 0.99).to(dtype=torch.float64).mean().detach().cpu().item()),
    }


def gradient_like_delta(phi: torch.Tensor, residual: torch.Tensor, g_edge: torch.Tensor | None = None) -> torch.Tensor:
    raw = phi.to(dtype=torch.float64).T @ residual.to(dtype=torch.float64)
    if g_edge is None:
        return raw
    sol, _diag = solve_exact(g_edge.to(device=phi.device, dtype=torch.float64), raw)
    return sol


def optimal_nonnegative_scale(phi: torch.Tensor, residual: torch.Tensor, delta: torch.Tensor) -> float:
    pred = phi.to(dtype=torch.float64) @ delta.to(dtype=torch.float64)
    denom = (pred * pred).sum().clamp_min(1.0e-12)
    eta = (pred * residual.to(dtype=torch.float64)).sum() / denom
    return max(0.0, float(eta.detach().cpu().item()))


def part0(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    code_files = [
        RUNNER,
        ROOT / "dgkan/fu/edge_sobolev_metrics.py",
        ROOT / "dgkan/fu/edge_functional_gram.py",
        ROOT / "dgkan/models/fc_purekan_primitives.py",
    ]
    evidence_files = [
        *code_files,
        PLAN,
    ]
    compile_ok = 1
    errors: list[str] = []
    for path in [RUNNER, ROOT / "dgkan/fu/edge_sobolev_metrics.py"]:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_ok = 0
            errors.append(f"{rel(path)}: {exc!r}")
    import_ok = 1
    for module in [
        "dgkan.fu.edge_sobolev_metrics",
        "dgkan.fu.edge_functional_gram",
        "experiments.run_v23_07_edge_function_residual_inverse_flow",
    ]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            import_ok = 0
            errors.append(f"{module}: {exc!r}")
    scan_ok, scan_hits = static_identity_scan(code_files)
    h10_official = read_json(ROOT / "results/v23_03_official_h10_full_positive_control_seed15_steps80/part_h_full_positive_control_summary.json")
    historical_evidence = {
        "v22_94_recap": rel(ROOT / "docs/DG-KAN_v22.94_AdamW_Witness_Decomposition_MultiScheme_MPFU_实验结果复盘.md"),
        "v22_97_source_guard_summary": rel(ROOT / "results/v22_97/current/part_e_sourceguard_pullback_summary.json"),
        "v23_03_final_route": read_json(ROOT / "results/v23_03/final_route.json"),
        "v23_03_h10_official_positive_control": h10_official,
        "v23_05_final_route": read_json(ROOT / "results/v23_05/final_route.json"),
        "v23_06_part_b_summary": read_json(ROOT / "results/v23_06/part_b_capacity_bridge_summary.json"),
    }
    row = {
        "selected_mainline": "edge_function_residual_inverse_flow",
        "v23_03_h10_available": int((ROOT / "results/v23_03_official_h10_full_positive_control_seed15_steps80").exists()),
        "v23_03_h10_gate_pass": int(h10_official.get("gate_pass", 0)),
        "v23_05_blind_observer_failed": int(read_json(ROOT / "results/v23_05/final_route.json").get("route") == "BlindObserverBridgeFailed"),
        "v23_06_sidecar_only": 1,
        "new_edge_function_added": 0,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "metric_winner_selection_used": 0,
        "compile_ok": compile_ok,
        "import_ok": import_ok,
        "identity_scan_ok": scan_ok,
        "identity_scan_hits": json.dumps(scan_hits, ensure_ascii=False),
        "identity_scan_scope": ";".join(rel(p) for p in code_files),
        "evidence_files": ";".join(rel(p) for p in evidence_files),
        "error_messages": "; ".join(errors),
    }
    matrix = write_rows(OUT_ROOT / "part_0_identity_matrix.csv", [row])
    gate = int(compile_ok and import_ok and scan_ok)
    summary = {
        "part": "0",
        "gate_pass": gate,
        "route": "Part0Pass" if gate else "Part0IdentityFailed",
        "dominant_blocker": "none" if gate else "compile_import_or_identity_scan",
        "history_lock": {
            "v22_94_adamw_witness_missing_freedom_recorded": "local_docs_present",
            "v22_96_97_98_signal_channel_quotient_pullback_instability_recorded": "local_docs_present",
            "v23_00R_23_02_functional_geometry_trust_recorded": "local_docs_present",
            "v23_03_h10_synthetic_control_recorded": "local_results_present",
            "v23_05_v23_06_observer_capacity_not_mainline_recorded": "local_results_present",
        },
        "historical_evidence": historical_evidence,
        "identity_matrix": rel(matrix),
        **row,
    }
    write_json(OUT_ROOT / "part_0_history_lock.json", summary)
    next_path = next_actions("0", gate, summary["dominant_blocker"], [] if gate else ["fix compile/import/static identity scan only"])
    write_json(
        OUT_ROOT / "part_0_next_actions_for_codex.json",
        {
            **read_json(next_path),
            "selected_mainline": "edge_function_residual_inverse_flow",
        },
    )
    append_exec("Part 0", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_0_history_lock.json')}", gpu=str(args.device), note=f"gate={gate}")
    append_recap("Part 0 history lock and identity", summary)
    return summary


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    p0 = read_json(OUT_ROOT / "part_0_history_lock.json")
    if p0 and not ival(p0.get("gate_pass")):
        summary = {
            "part": "A",
            "gate_pass": 0,
            "route": "A_BlockedByPart0",
            "dominant_blocker": p0.get("dominant_blocker", "part0_failed"),
        }
        write_json(OUT_ROOT / "part_a_summary.json", summary)
        next_actions("a", 0, str(summary["dominant_blocker"]), ["rerun Part 0 after allowed identity fixes"])
        append_recap("Part A blocked", summary)
        return summary
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    gen = torch.Generator(device=device).manual_seed(2307001)

    def run_case(case: str, phi: torch.Tensor, g: torch.Tensor, r: torch.Tensor, lam: float, w: torch.Tensor | None, block: int) -> None:
        try:
            wp = phi * (torch.ones(int(phi.shape[0]), device=device, dtype=torch.float64) if w is None else w.to(device=device, dtype=torch.float64)).sqrt().reshape(-1, 1)
            wr = r * (torch.ones(int(phi.shape[0]), device=device, dtype=torch.float64) if w is None else w.to(device=device, dtype=torch.float64)).reshape(-1, 1)
            design = wp.T @ wp
            rhs = phi.T @ wr
            K = sym(design + float(lam) * g)
            mn0, mx0, cond0 = eig_condition(design)
            mn, mx, cond = eig_condition(K)
            exact, ediag = solve_exact(K, rhs)
            cg, cdiag = solve_cg(K, rhs, tol=float(args.cg_tol), max_iter=int(args.cg_max_iter))
            bdiag, bdiag_diag = solve_blockdiag(K, rhs, block)
            vals, vecs = torch.linalg.eigh(sym(g))
            white = (vecs * vals.clamp_min(1.0e-12).rsqrt().reshape(1, -1)) @ vecs.T
            unwhite = (vecs * vals.clamp_min(1.0e-12).sqrt().reshape(1, -1)) @ vecs.T
            probe = torch.randn((int(g.shape[0]), 3), generator=gen, device=device, dtype=torch.float64)
            whiten_error = float((unwhite @ (white @ probe) - probe).norm().div(probe.norm().clamp_min(1.0e-12)).detach().cpu().item())
            rows.append(
                {
                    "part": "A",
                    "status": "ok",
                    "case": case,
                    "toy_solve_residual": ediag["solve_residual"],
                    "rank_deficiency_detected": int(not math.isfinite(cond0) or mn0 <= 1.0e-10),
                    "ridge_used": float(lam),
                    "condition_number_before_ridge": cond0,
                    "condition_number_after_ridge": cond,
                    "condition_number_min_eig_before": mn0,
                    "condition_number_max_eig_before": mx0,
                    "cholesky_success": ediag["cholesky_success"],
                    "cg_iterations": cdiag["cg_iterations"],
                    "cg_residual": cdiag["cg_residual"],
                    "exact_vs_cg_cosine": cosine(exact, cg),
                    "exact_vs_blockdiag_cosine": cosine(exact, bdiag),
                    "blockdiag_residual_full_K": bdiag_diag["blockdiag_residual_full_K"],
                    "functional_gram_min_eig": mn,
                    "functional_gram_max_eig": mx,
                    "whiten_unwhiten_error": whiten_error,
                    "solve_residual": ediag["solve_residual"],
                }
            )
        except Exception as exc:
            rows.append({"part": "A", "status": "error", "case": case, "error_message": repr(exc)})

    n, m, o = int(args.part_a_n), int(args.part_a_m), int(args.part_a_outputs)
    phi = torch.randn((n, m), generator=gen, device=device, dtype=torch.float64)
    a = torch.randn((m, m), generator=gen, device=device, dtype=torch.float64)
    g = sym(a.T @ a) + 0.1 * torch.eye(m, device=device, dtype=torch.float64)
    r = torch.randn((n, o), generator=gen, device=device, dtype=torch.float64)
    run_case("toy_well_conditioned", phi, g, r, float(args.lambda_default), None, max(1, m // 4))

    phi_rank = phi.clone()
    phi_rank[:, 1] = phi_rank[:, 0]
    run_case("toy_rank_deficient", phi_rank, g, r, float(args.lambda_default), None, max(1, m // 4))

    q, _ = torch.linalg.qr(torch.randn((n, m), generator=gen, device=device, dtype=torch.float64))
    phi_full = q[:, :m]
    run_case("toy_identity_limit", phi_full, torch.eye(m, device=device, dtype=torch.float64), r, 1.0e-8, None, max(1, m // 4))

    grad_rhs = phi.T @ r
    grad_delta, _ = solve_exact(torch.eye(m, device=device, dtype=torch.float64), grad_rhs)
    rows.append(
        {
            "part": "A",
            "status": "ok",
            "case": "toy_gradient_limit",
            "toy_solve_residual": 0.0,
            "rank_deficiency_detected": 0,
            "ridge_used": 0.0,
            "condition_number_before_ridge": 1.0,
            "condition_number_after_ridge": 1.0,
            "cholesky_success": 1,
            "cg_iterations": 0,
            "cg_residual": 0.0,
            "exact_vs_cg_cosine": 1.0,
            "exact_vs_blockdiag_cosine": 1.0,
            "functional_gram_min_eig": 1.0,
            "functional_gram_max_eig": 1.0,
            "whiten_unwhiten_error": 0.0,
            "gradient_limit_cosine_vs_phiT_R": cosine(grad_delta, grad_rhs),
            "solve_residual": 0.0,
        }
    )

    for basis_key in ["dche_k9", "dfour_default"]:
        basis_name, k = basis_for_key(basis_key)
        z = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float64)
        centers = torch.linspace(-1.0, 1.0, k, device=device, dtype=torch.float64)
        scales = torch.tensor([max(0.2, 2.0 / max(1, k - 1))], device=device, dtype=torch.float64)
        local_phi = _basis_eval(z, basis_name, k, centers, scales).reshape(n, k)
        g0 = functional_edge_gram(basis_key, k, sobolev_order=0.0, quadrature_points=int(args.quadrature_points), normalization="trace", ridge=float(args.functional_gram_ridge), device=device, dtype=torch.float64)
        r0 = torch.randn((n, o), generator=gen, device=device, dtype=torch.float64)
        run_case(f"real_basis_{basis_key}_L2", local_phi, g0, r0, float(args.lambda_default), None, k)
        gs = functional_edge_gram(basis_key, k, sobolev_order=0.25, quadrature_points=int(args.quadrature_points), normalization="trace", ridge=float(args.functional_gram_ridge), device=device, dtype=torch.float64)
        run_case(f"real_basis_{basis_key}_s025", local_phi, gs, r0, float(args.lambda_default), None, k)

    matrix = write_rows(OUT_ROOT / "part_a_solver_unit_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    exact_med = median(r.get("solve_residual") for r in ok)
    cg_med = median(r.get("cg_residual") for r in ok)
    white_med = median(r.get("whiten_unwhiten_error") for r in ok)
    gate = int(
        len(ok) == len(rows)
        and exact_med < 1.0e-6
        and cg_med < 1.0e-4
        and white_med < 1.0e-6
        and all(fval(r.get("functional_gram_min_eig"), 1.0) > 0.0 for r in ok)
    )
    summary = {
        "part": "A",
        "gate_pass": gate,
        "route": "PartASolverPass" if gate else "A_SolverFailed",
        "dominant_blocker": "none" if gate else "solver_residual_or_gram_pd_failure",
        "row_count": len(rows),
        "error_rows": sum(1 for r in rows if r.get("status") == "error"),
        "exact_solve_residual_median": exact_med,
        "cg_residual_median": cg_med,
        "whiten_unwhiten_error_median": white_med,
        "matrix": rel(matrix),
    }
    write_json(OUT_ROOT / "part_a_summary.json", summary)
    next_actions(
        "a",
        gate,
        summary["dominant_blocker"],
        [] if gate else ["fix symmetrization/ridge/jitter/basis quadrature/CG tolerance only"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-a --device {args.device}"],
    )
    append_exec("Part A", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_a_summary.json')}", gpu=str(args.device), note=f"gate={gate}")
    append_recap("Part A residual inverse solver unit tests", summary)
    return summary


def part_b_basis_input_gains(args: argparse.Namespace) -> list[float]:
    text = str(getattr(args, "part_b_basis_input_gains", "")).strip()
    return float_items(text) if text else [float(args.basis_input_gain)]


def part_b_jobs(args: argparse.Namespace) -> list[tuple[str, int, str, str, float]]:
    tasks = csv_items(args.part_b_tasks)
    targets = csv_items(args.part_b_residual_targets)
    checkpoints = csv_items(args.part_b_checkpoints)
    gains = part_b_basis_input_gains(args)
    return [
        (task, seed, target, ckpt, gain)
        for task in tasks
        for seed in range(int(args.part_b_seed_count))
        for target in targets
        for ckpt in checkpoints
        for gain in gains
    ]


def run_part_b_job(job: tuple[str, int, str, str, float], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    task, seed, residual_target, checkpoint, basis_input_gain = job
    basis_key = str(args.basis_key)
    rows: list[dict[str, Any]] = []
    xtr, ytr, xg, yg = visual_data(task, int(seed), args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = 23070000 + int(seed) * 1009 + sum(ord(c) for c in task) + sum(ord(c) for c in checkpoint)
    model = make_model(basis_key, int(args.depth), int(xtr.shape[1]), classes, model_seed, args, device, basis_input_gain=float(basis_input_gain))
    ckpt_info = train_checkpoint(model, xtr, ytr, checkpoint, int(seed), args)
    source_acts, source_residuals, _logits, source_loss = residual_bundle(model, xtr, ytr, residual_target)
    guard_acts, guard_residuals, _glogits, guard_loss = residual_bundle(model, xg, yg, residual_target)
    gen = torch.Generator(device=device).manual_seed(23071000 + int(seed) * 997 + sum(ord(c) for c in residual_target + checkpoint + task))
    for layer_idx in range(len(model.coeffs)):
        phi_s = model.layer_phi(source_acts[layer_idx], layer_idx)
        phi_g = model.layer_phi(guard_acts[layer_idx], layer_idx)
        r_s = source_residuals[layer_idx].to(device=device, dtype=torch.float64)
        r_g = guard_residuals[layer_idx].to(device=device, dtype=torch.float64)
        in_dim = int(model.dims[layer_idx])
        out_dim = int(model.dims[layer_idx + 1])
        k = int(model.k)
        g_l2 = expand_edge_gram(model, layer_idx, basis_key, 0.0, args, device=device)
        g_s025 = expand_edge_gram(model, layer_idx, basis_key, 0.25, args, device=device)
        g_identity = torch.eye(int(g_l2.shape[0]), device=device, dtype=torch.float64)
        source_rank = int(torch.linalg.matrix_rank(phi_s.to(dtype=torch.float64), tol=1.0e-8).detach().cpu().item())
        design_rank_deficient = int(source_rank < min(int(phi_s.shape[0]), int(phi_s.shape[1])))
        src_act_diag = activation_diagnostics(source_acts[layer_idx], float(basis_input_gain))
        grd_act_diag = activation_diagnostics(guard_acts[layer_idx], float(basis_input_gain))
        weights_pop = class_population_weights(r_s, ytr, float(args.population_beta), str(args.population_weight_mode))
        random_q, _ = torch.linalg.qr(torch.randn((int(r_s.shape[0]), min(int(r_s.shape[0]), max(1, int(r_s.shape[1])))), generator=gen, device=device, dtype=torch.float64))
        r_rand = random_q @ (random_q.T @ r_s)
        perm = torch.randperm(int(phi_s.shape[0]), generator=gen, device=device)
        for scheme in PART_B_SCHEMES:
            if scheme == "B7_H10_structure_weighted_diagnostic" and task not in {"local_patch_interaction", "rotation_sensitive"}:
                continue
            lambda_values = [0.0] if scheme in {"B0_ordinary_gradient", "B1_preconditioned_gradient"} else float_items(args.part_b_lambdas)
            for lam in lambda_values:
              try:
                lambda_key = "gradient" if scheme in {"B0_ordinary_gradient", "B1_preconditioned_gradient"} else f"{float(lam):.0e}"
                weights = None
                phi_for_solve = phi_s
                residual_for_solve = r_s
                g_used = g_l2
                solve_diag: dict[str, Any] = {}
                if scheme == "B0_ordinary_gradient":
                    delta = gradient_like_delta(phi_s, r_s, None)
                    eta = optimal_nonnegative_scale(phi_s, r_s, delta)
                    delta = eta * delta
                    solve_diag = {
                        "solve_residual": 0.0,
                        "condition_number_before_ridge": "",
                        "condition_number_after_ridge": "",
                        "lambda_used": 0.0,
                        "gradient_eta": eta,
                        "cholesky_success": "",
                    }
                    g_used = g_identity
                elif scheme == "B1_preconditioned_gradient":
                    delta = gradient_like_delta(phi_s, r_s, g_l2)
                    eta = optimal_nonnegative_scale(phi_s, r_s, delta)
                    delta = eta * delta
                    solve_diag = {
                        "solve_residual": 0.0,
                        "condition_number_before_ridge": "",
                        "condition_number_after_ridge": "",
                        "lambda_used": 0.0,
                        "gradient_eta": eta,
                        "cholesky_success": 1,
                    }
                else:
                    if scheme == "B3_EFRF_s025":
                        g_used = g_s025
                    elif scheme == "B4_EFRF_population_weighted":
                        weights = weights_pop
                    elif scheme == "B5_random_projected_EFRF":
                        residual_for_solve = r_rand
                    elif scheme == "B6_shuffled_design_EFRF":
                        phi_for_solve = phi_s[perm]
                    elif scheme == "B7_H10_structure_weighted_diagnostic":
                        weights = class_population_weights(r_s, ytr, float(args.population_beta) * 2.0, str(args.population_weight_mode))
                    elif scheme == "B8_EFRF_identity_G_diagnostic":
                        g_used = g_identity
                    delta, solve_diag = solve_normal(phi_for_solve, residual_for_solve, g_used, lam, weights)
                sm = improvement_metrics(phi_s, r_s, delta)
                gm = improvement_metrics(phi_g, r_g, delta)
                low, high = mode_energy(delta, k)
                edge_norm_g = float(((delta.T @ g_used @ delta).trace()).clamp_min(0.0).sqrt().detach().cpu().item())
                edge_norm_l2 = float(delta.norm().detach().cpu().item())
                row = {
                    "part": "B",
                    "status": "ok",
                    "task": task,
                    "seed": int(seed),
                    "basis_key": basis_key,
                    "depth": int(args.depth),
                    "train_size": int(args.train_size),
                    "guard_size": int(args.guard_size),
                    "basis_input_gain": float(basis_input_gain),
                    "checkpoint": checkpoint,
                    "checkpoint_isomorphic_synthetic": int(checkpoint == "h10_isomorphic_partial"),
                    "residual_target": residual_target,
                    "scheme": scheme,
                    "lambda_key": lambda_key,
                    "layer_idx": int(layer_idx),
                    "input_dim": in_dim,
                    "output_dim": out_dim,
                    "k": k,
                    "source_loss": source_loss,
                    "guard_loss": guard_loss,
                    "source_residual_mse_before": sm["residual_mse_before"],
                    "source_residual_mse_after": sm["residual_mse_after"],
                    "guard_residual_mse_before": gm["residual_mse_before"],
                    "guard_residual_mse_after": gm["residual_mse_after"],
                    "source_fit_improvement_ratio": sm["fit_improvement_ratio"],
                    "guard_fit_improvement_ratio": gm["fit_improvement_ratio"],
                    "source_guard_fit_gap": sm["fit_improvement_ratio"] - gm["fit_improvement_ratio"],
                    "source_predicted_loss_delta": sm["predicted_loss_delta"],
                    "guard_predicted_loss_delta": gm["predicted_loss_delta"],
                    "residual_alignment_cosine": gm["residual_alignment_cosine"],
                    "activation_residual_norm": gm["activation_residual_norm"],
                    "edge_norm_G": edge_norm_g,
                    "edge_norm_L2": edge_norm_l2,
                    "high_mode_energy_fraction": high,
                    "low_mode_energy_fraction": low,
                    "condition_number": solve_diag.get("condition_number_after_ridge", ""),
                    "condition_number_before_ridge": solve_diag.get("condition_number_before_ridge", ""),
                    "condition_number_after_ridge": solve_diag.get("condition_number_after_ridge", ""),
                    "solve_residual": solve_diag.get("solve_residual", 0.0),
                    "lambda_used": solve_diag.get("lambda_used", 0.0),
                    "cholesky_success": solve_diag.get("cholesky_success", ""),
                    "gradient_eta": solve_diag.get("gradient_eta", ""),
                    "source_rank": source_rank,
                    "design_rank_deficient": design_rank_deficient,
                    "source_activation_rms": src_act_diag["activation_rms"],
                    "source_activation_abs_mean": src_act_diag["activation_abs_mean"],
                    "source_activation_effective_rank": src_act_diag["activation_effective_rank"],
                    "source_basis_input_abs_mean": src_act_diag["basis_input_abs_mean"],
                    "source_basis_input_tanh_abs_mean": src_act_diag["basis_input_tanh_abs_mean"],
                    "source_basis_input_tanh_saturation_frac_095": src_act_diag["basis_input_tanh_saturation_frac_095"],
                    "source_basis_input_tanh_saturation_frac_099": src_act_diag["basis_input_tanh_saturation_frac_099"],
                    "guard_activation_rms": grd_act_diag["activation_rms"],
                    "guard_activation_abs_mean": grd_act_diag["activation_abs_mean"],
                    "guard_activation_effective_rank": grd_act_diag["activation_effective_rank"],
                    "guard_basis_input_abs_mean": grd_act_diag["basis_input_abs_mean"],
                    "guard_basis_input_tanh_abs_mean": grd_act_diag["basis_input_tanh_abs_mean"],
                    "guard_basis_input_tanh_saturation_frac_095": grd_act_diag["basis_input_tanh_saturation_frac_095"],
                    "guard_basis_input_tanh_saturation_frac_099": grd_act_diag["basis_input_tanh_saturation_frac_099"],
                    "population_weight_mean": float(weights_pop.mean().detach().cpu().item()),
                    "population_weight_mode": str(args.population_weight_mode),
                    **ckpt_info,
                }
                rows.append(row)
              except Exception as exc:
                rows.append(
                    {
                        "part": "B",
                        "status": "error",
                        "task": task,
                        "seed": int(seed),
                        "basis_key": basis_key,
                        "depth": int(args.depth),
                        "train_size": int(args.train_size),
                        "guard_size": int(args.guard_size),
                        "basis_input_gain": float(basis_input_gain),
                        "checkpoint": checkpoint,
                        "residual_target": residual_target,
                        "scheme": scheme,
                        "lambda_key": f"{float(lam):.0e}" if scheme not in {"B0_ordinary_gradient", "B1_preconditioned_gradient"} else "gradient",
                        "layer_idx": int(layer_idx),
                        "error_message": repr(exc),
                    }
                )
    by_key: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("task"), row.get("seed"), row.get("checkpoint"), row.get("residual_target"), row.get("layer_idx"), row.get("lambda_key"))
        by_key[(key, row["scheme"])] = {
            "guard": fval(row.get("guard_fit_improvement_ratio")),
            "source": fval(row.get("source_fit_improvement_ratio")),
        }
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("task"), row.get("seed"), row.get("checkpoint"), row.get("residual_target"), row.get("layer_idx"), row.get("lambda_key"))
        grad_key = (row.get("task"), row.get("seed"), row.get("checkpoint"), row.get("residual_target"), row.get("layer_idx"), "gradient")
        grad = by_key.get((grad_key, "B0_ordinary_gradient"), {})
        rand = by_key.get((key, CONTROL_PART_B_RANDOM), {})
        shuf = by_key.get((key, CONTROL_PART_B_SHUFFLE), {})
        row["gradient_guard_fit_improvement_ratio"] = grad.get("guard", "")
        row["random_projected_guard_fit_improvement_ratio"] = rand.get("guard", "")
        row["shuffled_design_guard_fit_improvement_ratio"] = shuf.get("guard", "")
        row["EFRF_minus_gradient_guard_fit"] = fval(row.get("guard_fit_improvement_ratio")) - fval(grad.get("guard"))
        row["EFRF_minus_random_projected_guard_fit"] = fval(row.get("guard_fit_improvement_ratio")) - fval(rand.get("guard"))
        row["EFRF_minus_shuffled_design_guard_fit"] = fval(row.get("guard_fit_improvement_ratio")) - fval(shuf.get("guard"))
        row["EFRF_source_guard_retention"] = fval(row.get("guard_fit_improvement_ratio")) / max(abs(fval(row.get("source_fit_improvement_ratio"))), 1.0e-12)
    return rows


def part_b(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    if not ival(pa.get("gate_pass")):
        summary = {"part": "B", "gate_pass": 0, "route": "B_BlockedByPartA", "dominant_blocker": pa.get("dominant_blocker", "part_a_missing_or_failed")}
        write_json(OUT_ROOT / "part_b_summary.json", summary)
        next_actions("b", 0, str(summary["dominant_blocker"]), ["rerun/fix Part A before Part B"])
        append_recap("Part B blocked", summary)
        return summary
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for job in shard_items(part_b_jobs(args), args):
        rows.extend(run_part_b_job(job, args, device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = write_rows(OUT_ROOT / f"part_b_frozen_residual_matrix{suffix}.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "part": "B",
        "gate_pass": 0,
        "route": "PartBShardOnly" if int(args.shard_count) > 1 else "PartBNeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "matrix": rel(matrix),
    }
    write_json(OUT_ROOT / f"part_b_frozen_residual_summary{suffix}.json", summary)
    append_exec("Part B", command_text(sys.argv), "done", files=rel(matrix), gpu=str(args.device), note=f"rows={len(rows)}")
    append_recap("Part B frozen residual fitting shard", summary)
    return summary


def part_b_merge(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    shard_paths = sorted(OUT_ROOT.glob("part_b_frozen_residual_matrix_shard*_of_*.csv"))
    if shard_paths:
        rows = [row for path in shard_paths for row in read_rows(path)]
    else:
        rows = read_rows(OUT_ROOT / "part_b_frozen_residual_matrix.csv")
    matrix = write_rows(OUT_ROOT / "part_b_frozen_residual_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    base_group_keys = ["scheme", "basis_key", "depth", "train_size", "guard_size", "basis_input_gain", "population_weight_mode", "residual_target", "checkpoint", "lambda_key"]
    group_keys = ["layer_policy", *base_group_keys]
    groups: list[dict[str, Any]] = []
    layer_policies = {
        "all_layers": lambda r: True,
        "last_layer": lambda r: int(float(r.get("layer_idx", 0))) == int(float(r.get("depth", 1))) - 1,
    }
    for layer_policy, policy_predicate in layer_policies.items():
        policy_rows = [r for r in ok if policy_predicate(r)]
        for base_values in sorted({tuple(r.get(k, "") for k in base_group_keys) for r in policy_rows}):
            group = [r for r in policy_rows if tuple(r.get(k, "") for k in base_group_keys) == base_values]
            if not group:
                continue
            values = (layer_policy, *base_values)
            scheme = str(base_values[0])
            candidate = scheme in EFRF_PART_B_CANDIDATES
            guard = median(r.get("guard_fit_improvement_ratio") for r in group)
            grad = median(r.get("gradient_guard_fit_improvement_ratio") for r in group)
            rand_gap = median(r.get("EFRF_minus_random_projected_guard_fit") for r in group)
            shuf_gap = median(r.get("EFRF_minus_shuffled_design_guard_fit") for r in group)
            sg_gap = median(r.get("source_guard_fit_gap") for r in group)
            solve = median(r.get("solve_residual") for r in group)
            cond = median(r.get("condition_number_after_ridge") for r in group if str(r.get("condition_number_after_ridge", "")) != "")
            primary = int(base_values[1] == "dche_k9" and int(float(base_values[2])) == 3)
            pass_gate = int(
                candidate
                and primary
                and guard >= grad + 0.15
                and rand_gap >= 0.10
                and shuf_gap >= 0.10
                and sg_gap <= 0.20
                and solve <= 1.0e-4
                and cond <= 1.0e6
            )
            groups.append(
                {
                    **dict(zip(group_keys, values)),
                    "rows": len(group),
                    "candidate_EFRF": int(candidate),
                    "primary_row": primary,
                    "guard_fit_improvement_ratio_median": guard,
                    "gradient_guard_fit_improvement_ratio_median": grad,
                    "EFRF_minus_gradient_guard_fit_median": median(r.get("EFRF_minus_gradient_guard_fit") for r in group),
                    "EFRF_minus_random_projected_guard_fit_median": rand_gap,
                    "EFRF_minus_shuffled_design_guard_fit_median": shuf_gap,
                    "source_guard_fit_gap_median": sg_gap,
                    "solve_residual_median": solve,
                    "condition_number_after_ridge_median": cond,
                    "design_rank_deficient_rows": sum(ival(r.get("design_rank_deficient")) for r in group),
                    "source_activation_effective_rank_median": median(r.get("source_activation_effective_rank") for r in group),
                    "source_basis_input_tanh_saturation_frac_095_median": median(r.get("source_basis_input_tanh_saturation_frac_095") for r in group),
                    "guard_activation_effective_rank_median": median(r.get("guard_activation_effective_rank") for r in group),
                    "guard_basis_input_tanh_saturation_frac_095_median": median(r.get("guard_basis_input_tanh_saturation_frac_095") for r in group),
                    "scheme_gate_pass": pass_gate,
                }
            )
    group_csv = write_rows(OUT_ROOT / "part_b_frozen_residual_group_summary.csv", groups)
    pass_groups = [g for g in groups if ival(g.get("scheme_gate_pass")) == 1]
    efrf_groups = [g for g in groups if ival(g.get("candidate_EFRF")) == 1 and ival(g.get("primary_row")) == 1]
    if pass_groups:
        route = "PartBFrozenResidualInversePass"
        blocker = "none"
        gate = 1
    else:
        gate = 0
        if any(g for g in efrf_groups if fval(g.get("guard_fit_improvement_ratio_median")) < fval(g.get("gradient_guard_fit_improvement_ratio_median")) + 0.15):
            route, blocker = "B_FrozenResidualInverseFailed", "efrf_not_better_than_gradient"
        elif any(g for g in efrf_groups if fval(g.get("EFRF_minus_random_projected_guard_fit_median")) < 0.10):
            route, blocker = "B_RandomResidualAssimilation", "random_projected_gap_low"
        elif any(g for g in efrf_groups if fval(g.get("source_guard_fit_gap_median")) > 0.20):
            route, blocker = "FrozenResidualOverfit", "source_guard_fit_gap_high"
        elif any(g for g in efrf_groups if ival(g.get("design_rank_deficient_rows")) > 0):
            route, blocker = "B_FrozenResidualInverseFailed", "design_rank_deficient"
        else:
            route, blocker = "B_FrozenResidualInverseFailed", "no_fixed_efrf_scheme_passed"
    summary = {
        "part": "B",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "passing_scheme_groups": pass_groups,
        "matrix": rel(matrix),
        "group_summary": rel(group_csv),
        "primary_efrf_groups": efrf_groups,
    }
    write_json(OUT_ROOT / "part_b_frozen_residual_summary.json", summary)
    actions = [] if gate else [
        "compare residual targets act/norm/GN already present in this run",
        "sweep fixed lambda values 1e-4,1e-3,1e-2,1e-1,1 as independent schemes",
        "compare G_L2, G_s025 and identity G",
        "inspect Phi rank and activation saturation; fixed basis_input_gain sweep is supported through --part-b-basis-input-gains",
        "if random-projected is close, write RandomResidualAssimilation and do not continue to Part C",
    ]
    next_actions(
        "b",
        gate,
        blocker,
        actions,
        [f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-b --device cuda:0 --shard-count 2 --shard-index 0",
         f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-b --device cuda:0 --shard-count 2 --shard-index 1",
         f"{PYTHON} {rel(RUNNER)} --mode part-b-merge --device cpu"],
    )
    append_exec("Part B merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_b_frozen_residual_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part B frozen residual fitting merge", summary)
    return summary


def coeff_delta_to_param(delta: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    in_dim, out_dim, k = int(param.shape[0]), int(param.shape[1]), int(param.shape[2])
    return delta.to(device=param.device, dtype=param.dtype).reshape(in_dim, k, out_dim).permute(0, 2, 1).contiguous()


def model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def load_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({name: tensor.detach().clone() for name, tensor in state.items()})


def actual_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    metrics = v2293.metrics_for_model(model, x, y)
    return {
        "loss": fval(metrics.get("nll")),
        "coverage": fval(metrics.get("coverage_CVaR25")),
        "accuracy": fval(metrics.get("accuracy")),
        "debt": fval(metrics.get("debt_metric")),
        "brier": fval(metrics.get("brier")),
        "ece": fval(metrics.get("ece")),
        "tail95": fval(metrics.get("tail95")),
        "tail99": fval(metrics.get("tail99")),
        "margin10": fval(metrics.get("margin10")),
    }


def apply_coeff_deltas(model: v2293.TrueDeepPureKAN, deltas: dict[int, torch.Tensor], alpha: float) -> None:
    with torch.no_grad():
        for layer_idx, delta in deltas.items():
            if int(layer_idx) < 0 or int(layer_idx) >= len(model.coeffs):
                continue
            model.coeffs[int(layer_idx)].add_(float(alpha) * coeff_delta_to_param(delta, model.coeffs[int(layer_idx)]))


def part_c_basis_input_gains(args: argparse.Namespace) -> list[float]:
    text = str(getattr(args, "part_c_basis_input_gains", "")).strip()
    return float_items(text) if text else part_b_basis_input_gains(args)


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, int, str, str, float]]:
    tasks = csv_items(args.part_c_tasks)
    targets = csv_items(args.part_c_residual_targets)
    checkpoints = csv_items(args.part_c_checkpoints)
    gains = part_c_basis_input_gains(args)
    return [
        (task, seed, target, ckpt, gain)
        for task in tasks
        for seed in range(int(args.part_c_seed_count))
        for target in targets
        for ckpt in checkpoints
        for gain in gains
    ]


def part_c_layer_indices(model: v2293.TrueDeepPureKAN, scheme: str) -> list[int]:
    if scheme == "C1_EFRF_L2_last_layer":
        return [len(model.coeffs) - 1]
    return list(range(len(model.coeffs)))


def part_c_scheme_delta(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    residual_target: str,
    scheme: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
    task: str,
    checkpoint: str,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    basis_key = str(args.basis_key)
    selected_layers = part_c_layer_indices(model, scheme)
    deltas: dict[int, torch.Tensor] = {}
    solve_residuals: list[float] = []
    conds: list[float] = []
    predicted_guard_loss_deltas: list[float] = []
    predicted_guard_fit: list[float] = []
    rank_deficient = 0
    gauss = scheme in {
        "C3_EFRF_L2_gauss_seidel",
        "C4_EFRF_s025_gauss_seidel",
        "C5_EFRF_Wpop_gauss_seidel",
        "D4_EFRF_L2_identityW_gauss_seidel",
        "D5_EFRF_s025_identityW_gauss_seidel",
        "D6_EFRF_L2_Wpop_gauss_seidel",
        "D7_EFRF_s025_Wpop_gauss_seidel",
        "D8_EFRF_random_projected_residual_flow",
        "D9_EFRF_shuffled_design_residual_flow",
        "D10_EFRF_identity_G_diagnostic",
    }
    temp_state = model_state(model)
    gen = torch.Generator(device=device).manual_seed(23072000 + int(seed) * 1009 + sum(ord(c) for c in task + checkpoint + residual_target + scheme))

    for layer_idx in selected_layers:
        if gauss:
            # Recompute residuals after previous layer updates in a fixed top-down pass.
            source_acts, source_residuals, _logits, _loss = residual_bundle(model, x, y, residual_target)
        elif layer_idx == selected_layers[0]:
            source_acts, source_residuals, _logits, _loss = residual_bundle(model, x, y, residual_target)
        phi_s = model.layer_phi(source_acts[layer_idx], layer_idx)
        r_s = source_residuals[layer_idx].to(device=device, dtype=torch.float64)
        g_used = expand_edge_gram(model, layer_idx, basis_key, 0.0, args, device=device)
        weights = None
        residual_for_solve = r_s
        phi_for_solve = phi_s
        if scheme == "C0_ordinary_gradient_actual":
            delta = gradient_like_delta(phi_s, r_s, None)
            eta = optimal_nonnegative_scale(phi_s, r_s, delta)
            delta = eta * delta
            solve_diag = {"solve_residual": 0.0, "condition_number_after_ridge": 0.0}
        else:
            if scheme in {"C4_EFRF_s025_gauss_seidel", "D5_EFRF_s025_identityW_gauss_seidel", "D7_EFRF_s025_Wpop_gauss_seidel"}:
                g_used = expand_edge_gram(model, layer_idx, basis_key, 0.25, args, device=device)
            if scheme in {"C5_EFRF_Wpop_gauss_seidel", "D6_EFRF_L2_Wpop_gauss_seidel", "D7_EFRF_s025_Wpop_gauss_seidel"}:
                weights = class_population_weights(r_s, y, float(args.population_beta), str(args.population_weight_mode))
            if scheme in {"C6_random_projected_EFRF", "D8_EFRF_random_projected_residual_flow"}:
                random_q, _ = torch.linalg.qr(
                    torch.randn((int(r_s.shape[0]), min(int(r_s.shape[0]), max(1, int(r_s.shape[1])))), generator=gen, device=device, dtype=torch.float64)
                )
                residual_for_solve = random_q @ (random_q.T @ r_s)
            if scheme == "D9_EFRF_shuffled_design_residual_flow":
                perm = torch.randperm(int(phi_s.shape[0]), generator=gen, device=device)
                phi_for_solve = phi_s[perm]
            if scheme == "D10_EFRF_identity_G_diagnostic":
                g_used = torch.eye(int(g_used.shape[0]), device=device, dtype=torch.float64)
            delta, solve_diag = solve_normal(phi_for_solve, residual_for_solve, g_used, float(args.part_c_lambda), weights)
        deltas[layer_idx] = delta
        solve_residuals.append(fval(solve_diag.get("solve_residual")))
        conds.append(fval(solve_diag.get("condition_number_after_ridge")))
        source_rank = int(torch.linalg.matrix_rank(phi_s.to(dtype=torch.float64), tol=1.0e-8).detach().cpu().item())
        rank_deficient += int(source_rank < min(int(phi_s.shape[0]), int(phi_s.shape[1])))

        # Guard prediction uses current model state for GS and base state for simultaneous.
        _acts, _resid, _logits, _loss = residual_bundle(model, x_guard, y_guard, residual_target)
        pm = improvement_metrics(model.layer_phi(_acts[layer_idx], layer_idx), _resid[layer_idx].to(device=device, dtype=torch.float64), delta)
        predicted_guard_loss_deltas.append(pm["predicted_loss_delta"])
        predicted_guard_fit.append(pm["fit_improvement_ratio"])
        if gauss:
            apply_coeff_deltas(model, {layer_idx: delta}, 1.0)

    if gauss:
        load_model_state(model, temp_state)
    return deltas, {
        "selected_layers": ",".join(str(i) for i in selected_layers),
        "layer_count": len(selected_layers),
        "solve_residual_max": max(solve_residuals or [0.0]),
        "condition_number_after_ridge_max": max(conds or [0.0]),
        "design_rank_deficient_layers": rank_deficient,
        "predicted_guard_loss_delta_sum": sum(predicted_guard_loss_deltas),
        "predicted_guard_fit_mean": mean(predicted_guard_fit),
        "gauss_seidel_internal_unit_alpha": int(gauss),
    }


def choose_alpha_for_deltas(
    model: v2293.TrueDeepPureKAN,
    base_state: dict[str, torch.Tensor],
    deltas: dict[int, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    tol = float(args.part_c_trust_loss_tolerance)
    best_source = source_before
    best_guard = guard_before
    for alpha in float_items(args.part_c_alphas):
        load_model_state(model, base_state)
        apply_coeff_deltas(model, deltas, alpha)
        source_after = actual_metrics(model, x_source, y_source)
        guard_after = actual_metrics(model, x_guard, y_guard)
        if source_after["loss"] <= source_before["loss"] + tol and guard_after["loss"] <= guard_before["loss"] + tol:
            return float(alpha), 1, "accepted", source_after, guard_after
        best_source, best_guard = source_after, guard_after
    load_model_state(model, base_state)
    return 0.0, 0, "loss_trust_reject", source_before, guard_before


def part_d_objective_trust_accept(
    part: str,
    source_before: dict[str, float],
    source_after: dict[str, float],
    guard_before: dict[str, float],
    guard_after: dict[str, float],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    tol = float(args.part_d_trust_loss_tolerance)
    if part == "D_C2":
        ok = (
            source_after["coverage"] >= source_before["coverage"] - tol
            and guard_after["coverage"] >= guard_before["coverage"] - tol
        )
        return ok, "c2_floor_trust_reject"
    if part == "D_F5":
        budget = float(args.no_debt_budget) + tol
        source_ok = no_debt_ok(debt_deltas(source_before, source_after), budget)
        guard_ok = no_debt_ok(debt_deltas(guard_before, guard_after), budget)
        return bool(source_ok and guard_ok), "f5_no_debt_trust_reject"
    ok = source_after["loss"] <= source_before["loss"] + tol and guard_after["loss"] <= guard_before["loss"] + tol
    return ok, "loss_trust_reject"


def choose_alpha_for_deltas_part_d(
    model: v2293.TrueDeepPureKAN,
    base_state: dict[str, torch.Tensor],
    deltas: dict[int, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    part: str,
    args: argparse.Namespace,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    reject_reason = "no_alpha_candidates"
    for alpha in float_items(args.part_d_alphas):
        load_model_state(model, base_state)
        apply_coeff_deltas(model, deltas, alpha)
        source_after = actual_metrics(model, x_source, y_source)
        guard_after = actual_metrics(model, x_guard, y_guard)
        accepted, reject_reason = part_d_objective_trust_accept(part, source_before, source_after, guard_before, guard_after, args)
        if accepted:
            return float(alpha), 1, "accepted", source_after, guard_after
    load_model_state(model, base_state)
    return 0.0, 0, reject_reason, source_before, guard_before


def choose_alpha_functionalgram(
    model: v2293.TrueDeepPureKAN,
    base_state: dict[str, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    tol = float(args.part_c_trust_loss_tolerance)
    for alpha in float_items(args.part_c_alphas):
        load_model_state(model, base_state)
        opt = EdgeSobolevPopulationFlow(
            list(model.coeffs),
            lr=float(alpha) * float(args.part_c_functionalgram_lr),
            weight_decay=float(args.weight_decay),
            sobolev_exponent=0.0,
            edge_metric_type="functional_gram",
            edge_weight_normalization="trace",
            edge_weight_ridge=float(args.functional_gram_ridge),
            functional_gram_quadrature_points=int(args.quadrature_points),
            use_population_gate=False,
        )
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_source).float(), y_source.long())
        loss.backward()
        opt.step()
        source_after = actual_metrics(model, x_source, y_source)
        guard_after = actual_metrics(model, x_guard, y_guard)
        if source_after["loss"] <= source_before["loss"] + tol and guard_after["loss"] <= guard_before["loss"] + tol:
            return float(alpha), 1, "accepted", source_after, guard_after
    load_model_state(model, base_state)
    return 0.0, 0, "loss_trust_reject", source_before, guard_before


def run_part_c_job(job: tuple[str, int, str, str, float], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    task, seed, residual_target, checkpoint, basis_input_gain = job
    basis_key = str(args.basis_key)
    xtr, ytr, xg, yg = visual_data(task, int(seed), args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = 23070000 + int(seed) * 1009 + sum(ord(c) for c in task) + sum(ord(c) for c in checkpoint)
    model = make_model(basis_key, int(args.depth), int(xtr.shape[1]), classes, model_seed, args, device, basis_input_gain=float(basis_input_gain))
    ckpt_info = train_checkpoint(model, xtr, ytr, checkpoint, int(seed), args)
    base_state = model_state(model)
    source_before = actual_metrics(model, xtr, ytr)
    guard_before = actual_metrics(model, xg, yg)
    rows: list[dict[str, Any]] = []
    for scheme in csv_items(args.part_c_schemes):
        try:
            load_model_state(model, base_state)
            diag: dict[str, Any] = {
                "selected_layers": "",
                "layer_count": 0,
                "solve_residual_max": 0.0,
                "condition_number_after_ridge_max": 0.0,
                "design_rank_deficient_layers": 0,
                "predicted_guard_loss_delta_sum": 0.0,
                "predicted_guard_fit_mean": 0.0,
                "gauss_seidel_internal_unit_alpha": 0,
            }
            if scheme == CONTROL_PART_C_NOOP:
                alpha, accepted, reject_reason = 0.0, 1, "same_compute_noop"
                source_after, guard_after = source_before, guard_before
            elif scheme == CONTROL_PART_C_FUNCTIONALGRAM:
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_functionalgram(model, base_state, source_before, guard_before, xtr, ytr, xg, yg, args)
            else:
                deltas, diag = part_c_scheme_delta(model, xtr, ytr, xg, yg, residual_target, scheme, args, device, seed=int(seed), task=task, checkpoint=checkpoint)
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_for_deltas(model, base_state, deltas, source_before, guard_before, xtr, ytr, xg, yg, args)
            rows.append(
                {
                    "part": "C",
                    "status": "ok",
                    "task": task,
                    "seed": int(seed),
                    "basis_key": basis_key,
                    "depth": int(args.depth),
                    "train_size": int(args.train_size),
                    "guard_size": int(args.guard_size),
                    "basis_input_gain": float(basis_input_gain),
                    "population_weight_mode": str(args.population_weight_mode),
                    "checkpoint": checkpoint,
                    "residual_target": residual_target,
                    "scheme": scheme,
                    "lambda_key": f"{float(args.part_c_lambda):.0e}" if scheme not in {CONTROL_PART_C_GRADIENT, CONTROL_PART_C_FUNCTIONALGRAM, CONTROL_PART_C_NOOP} else "actual",
                    "alpha_selected": alpha,
                    "finite_step_accepted": accepted,
                    "reject_reason": reject_reason,
                    "source_loss_before": source_before["loss"],
                    "source_loss_after": source_after["loss"],
                    "guard_loss_before": guard_before["loss"],
                    "guard_loss_after": guard_after["loss"],
                    "actual_source_loss_delta": source_after["loss"] - source_before["loss"],
                    "actual_guard_loss_delta": guard_after["loss"] - guard_before["loss"],
                    "source_C2_coverage_before": source_before["coverage"],
                    "source_C2_coverage_after": source_after["coverage"],
                    "guard_C2_coverage_before": guard_before["coverage"],
                    "guard_C2_coverage_after": guard_after["coverage"],
                    "actual_C2_coverage_delta": guard_after["coverage"] - guard_before["coverage"],
                    "guard_accuracy_before": guard_before["accuracy"],
                    "guard_accuracy_after": guard_after["accuracy"],
                    "actual_C2_accuracy_delta": guard_after["accuracy"] - guard_before["accuracy"],
                    "guard_debt_delta": guard_after["debt"] - guard_before["debt"],
                    **diag,
                    **ckpt_info,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "part": "C",
                    "status": "error",
                    "task": task,
                    "seed": int(seed),
                    "basis_key": basis_key,
                    "depth": int(args.depth),
                    "train_size": int(args.train_size),
                    "guard_size": int(args.guard_size),
                    "basis_input_gain": float(basis_input_gain),
                    "checkpoint": checkpoint,
                    "residual_target": residual_target,
                    "scheme": scheme,
                    "error_message": repr(exc),
                }
            )
    return rows


def corrcoef(xs: Iterable[Any], ys: Iterable[Any]) -> float:
    x = np.asarray([fval(v, float("nan")) for v in xs], dtype=float)
    y = np.asarray([fval(v, float("nan")) for v in ys], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def enrich_part_c_controls(rows: list[dict[str, Any]]) -> None:
    base_keys = ["task", "seed", "checkpoint", "residual_target", "basis_input_gain"]
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(k, "") for k in base_keys)
        by_key.setdefault(key, {})[str(row.get("scheme"))] = row
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(k, "") for k in base_keys)
        controls = by_key.get(key, {})
        grad = controls.get(CONTROL_PART_C_GRADIENT, {})
        rand = controls.get(CONTROL_PART_C_RANDOM, {})
        noop = controls.get(CONTROL_PART_C_NOOP, {})
        fg = controls.get(CONTROL_PART_C_FUNCTIONALGRAM, {})
        row["gradient_guard_loss_delta"] = grad.get("actual_guard_loss_delta", "")
        row["functionalgram_actual_C2_coverage_delta"] = fg.get("actual_C2_coverage_delta", "")
        row["random_projected_actual_C2_coverage_delta"] = rand.get("actual_C2_coverage_delta", "")
        row["same_compute_noop_C2_coverage_delta"] = noop.get("actual_C2_coverage_delta", "")
        row["guard_loss_gap_vs_gradient"] = fval(grad.get("actual_guard_loss_delta")) - fval(row.get("actual_guard_loss_delta"))
        row["functionalgram_coverage_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(fg.get("actual_C2_coverage_delta"))
        row["random_projected_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(rand.get("actual_C2_coverage_delta"))
        row["same_compute_noop_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(noop.get("actual_C2_coverage_delta"))


def part_c(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pb = read_json(OUT_ROOT / "part_b_frozen_residual_summary.json")
    if not ival(pb.get("gate_pass")):
        return write_blocked_summary("C", "C_BlockedByPartB", str(pb.get("dominant_blocker", "part_b_missing_or_failed")))
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for job in shard_items(part_c_jobs(args), args):
        rows.extend(run_part_c_job(job, args, device))
    enrich_part_c_controls(rows)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = write_rows(OUT_ROOT / f"part_c_actual_update_matrix{suffix}.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "part": "C",
        "gate_pass": 0,
        "route": "PartCShardOnly" if int(args.shard_count) > 1 else "PartCNeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "matrix": rel(matrix),
    }
    write_json(OUT_ROOT / f"part_c_summary{suffix}.json", summary)
    append_exec("Part C", command_text(sys.argv), "done", files=rel(matrix), gpu=str(args.device), note=f"rows={len(rows)}")
    append_recap("Part C actual update shard", summary)
    return summary


def part_c_merge(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    shard_paths = sorted(OUT_ROOT.glob("part_c_actual_update_matrix_shard*_of_*.csv"))
    if shard_paths:
        rows = [row for path in shard_paths for row in read_rows(path)]
    else:
        rows = read_rows(OUT_ROOT / "part_c_actual_update_matrix.csv")
    enrich_part_c_controls(rows)
    matrix = write_rows(OUT_ROOT / "part_c_actual_update_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    group_keys = ["scheme", "basis_key", "depth", "train_size", "guard_size", "basis_input_gain", "population_weight_mode", "residual_target", "checkpoint", "lambda_key"]
    groups: list[dict[str, Any]] = []
    for values in sorted({tuple(r.get(k, "") for k in group_keys) for r in ok}):
        group = [r for r in ok if tuple(r.get(k, "") for k in group_keys) == values]
        if not group:
            continue
        scheme = str(values[0])
        candidate = scheme in EFRF_PART_C_CANDIDATES
        primary = int(values[1] == "dche_k9" and int(float(values[2])) == 3)
        accept_rate = mean(ival(r.get("finite_step_accepted")) for r in group)
        scale_mean = mean(r.get("alpha_selected") for r in group)
        loss_gap = median(r.get("guard_loss_gap_vs_gradient") for r in group)
        fg_gap = median(r.get("functionalgram_coverage_gap") for r in group)
        rand_gap = median(r.get("random_projected_gap") for r in group)
        pred_loss_corr = corrcoef([r.get("predicted_guard_loss_delta_sum") for r in group], [r.get("actual_guard_loss_delta") for r in group])
        pred_cov_corr = corrcoef([r.get("predicted_guard_fit_mean") for r in group], [r.get("actual_C2_coverage_delta") for r in group])
        pass_gate = int(
            candidate
            and primary
            and loss_gap >= float(args.part_c_loss_margin)
            and fg_gap >= float(args.part_c_functionalgram_coverage_margin)
            and rand_gap >= float(args.part_c_random_gap)
            and pred_loss_corr >= 0.30
            and accept_rate >= 0.50
            and scale_mean >= 0.30
        )
        groups.append(
            {
                **dict(zip(group_keys, values)),
                "rows": len(group),
                "candidate_EFRF": int(candidate),
                "primary_row": primary,
                "actual_guard_loss_delta_median": median(r.get("actual_guard_loss_delta") for r in group),
                "gradient_guard_loss_delta_median": median(r.get("gradient_guard_loss_delta") for r in group),
                "guard_loss_gap_vs_gradient_median": loss_gap,
                "actual_C2_coverage_delta_median": median(r.get("actual_C2_coverage_delta") for r in group),
                "functionalgram_actual_C2_coverage_delta_median": median(r.get("functionalgram_actual_C2_coverage_delta") for r in group),
                "functionalgram_coverage_gap_median": fg_gap,
                "random_projected_gap_median": rand_gap,
                "same_compute_noop_gap_median": median(r.get("same_compute_noop_gap") for r in group),
                "actual_C2_accuracy_delta_median": median(r.get("actual_C2_accuracy_delta") for r in group),
                "predicted_vs_actual_loss_corr": pred_loss_corr,
                "predicted_vs_actual_coverage_corr": pred_cov_corr,
                "finite_step_accept_rate": accept_rate,
                "finite_step_scale_mean": scale_mean,
                "finite_step_skip_count": sum(1 - ival(r.get("finite_step_accepted")) for r in group),
                "reject_reason_counts": json.dumps({reason: sum(1 for r in group if str(r.get("reject_reason", "")) == reason) for reason in sorted({str(r.get("reject_reason", "")) for r in group})}, sort_keys=True),
                "scheme_gate_pass": pass_gate,
            }
        )
    group_csv = write_rows(OUT_ROOT / "part_c_actual_update_group_summary.csv", groups)
    pass_groups = [g for g in groups if ival(g.get("scheme_gate_pass")) == 1]
    candidate_groups = [g for g in groups if ival(g.get("candidate_EFRF")) == 1 and ival(g.get("primary_row")) == 1]
    if pass_groups:
        gate, route, blocker = 1, "PartCOneStepActualPass", "none"
    else:
        gate = 0
        if any(fval(g.get("finite_step_accept_rate")) < 0.50 or fval(g.get("finite_step_scale_mean")) < 0.30 for g in candidate_groups):
            route, blocker = "C_ActualStepFailed", "finite_step_trust_low"
        elif any(fval(g.get("guard_loss_gap_vs_gradient_median")) < float(args.part_c_loss_margin) for g in candidate_groups):
            route, blocker = "C_ActualStepFailed", "actual_guard_loss_not_better_than_gradient"
        elif any(fval(g.get("functionalgram_coverage_gap_median")) < float(args.part_c_functionalgram_coverage_margin) for g in candidate_groups):
            route, blocker = "C_ActualStepFailed", "functionalgram_coverage_gap_low"
        elif any(fval(g.get("random_projected_gap_median")) < float(args.part_c_random_gap) for g in candidate_groups):
            route, blocker = "C_ActualStepFailed", "random_projected_gap_low"
        else:
            route, blocker = "C_ActualStepFailed", "predicted_actual_corr_low_or_no_fixed_scheme"
    summary = {
        "part": "C",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "passing_scheme_groups": pass_groups,
        "matrix": rel(matrix),
        "group_summary": rel(group_csv),
        "candidate_groups": candidate_groups,
    }
    write_json(OUT_ROOT / "part_c_summary.json", summary)
    actions = [] if gate else ["check step scale/trust, activation drift, simultaneous vs Gauss-Seidel, residual recomputation frequency, and solve damping"]
    next_actions(
        "c",
        gate,
        blocker,
        actions,
        [f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-c --device cuda:0 --shard-count 2 --shard-index 0",
         f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-c --device cuda:0 --shard-count 2 --shard-index 1",
         f"{PYTHON} {rel(RUNNER)} --mode part-c-merge --device cpu"],
    )
    append_exec("Part C merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_c_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part C actual update merge", summary)
    return summary


def part_d_basis_input_gains(args: argparse.Namespace) -> list[float]:
    text = str(getattr(args, "part_d_basis_input_gains", "")).strip()
    return float_items(text) if text else part_c_basis_input_gains(args)


def part_d_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, int, float]]:
    schemes = csv_items(args.part_d_schemes)
    tasks = csv_items(args.part_d_tasks)
    gains = part_d_basis_input_gains(args)
    jobs: list[tuple[str, str, str, int, float]] = []
    if "D_C2" in set(csv_items(args.part_d_parts)):
        for scheme in schemes:
            for task in tasks:
                for seed in range(int(args.part_d_seed_count)):
                    for gain in gains:
                        jobs.append((scheme, "D_C2", task, seed, gain))
    if "D_F5" in set(csv_items(args.part_d_parts)):
        for scheme in schemes:
            for seed in range(int(args.part_d_f5_seed_count)):
                task = tasks[seed % max(1, len(tasks))]
                for gain in gains:
                    jobs.append((scheme, "D_F5", task, seed, gain))
    return jobs


def part_d_run_args(args: argparse.Namespace) -> argparse.Namespace:
    dargs = argparse.Namespace(**vars(args))
    dargs.part_c_lambda = float(args.part_d_lambda)
    dargs.part_c_alphas = str(args.part_d_alphas)
    dargs.part_c_trust_loss_tolerance = float(args.part_d_trust_loss_tolerance)
    return dargs


def choose_alpha_optimizer_step(
    model: v2293.TrueDeepPureKAN,
    base_state: dict[str, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    scheme: str,
    part: str,
    args: argparse.Namespace,
) -> tuple[float, int, str, dict[str, float], dict[str, float], dict[str, Any]]:
    diag = {
        "optimizer_kind": scheme,
        "blocksnr_population_proxy_observed_gradients": 0,
        "selected_layers": "optimizer",
        "layer_count": len(model.coeffs),
        "solve_residual_max": 0.0,
        "condition_number_after_ridge_max": 0.0,
        "design_rank_deficient_layers": 0,
        "predicted_guard_loss_delta_sum": 0.0,
        "predicted_guard_fit_mean": 0.0,
    }
    reject_reason = "no_alpha_candidates"
    for alpha in float_items(args.part_d_alphas):
        load_model_state(model, base_state)
        if scheme in {"D1_FunctionalGram_AdamW", "D2_FunctionalGram_BlockSNR_population_proxy"}:
            opt = EdgeSobolevPopulationFlow(
                list(model.coeffs),
                lr=float(alpha) * float(args.part_d_functionalgram_lr),
                weight_decay=float(args.weight_decay),
                sobolev_exponent=0.0,
                edge_metric_type="functional_gram",
                edge_weight_normalization="trace",
                edge_weight_ridge=float(args.functional_gram_ridge),
                functional_gram_quadrature_points=int(args.quadrature_points),
                use_population_gate=(scheme == "D2_FunctionalGram_BlockSNR_population_proxy"),
                gate_family="block",
                strict_edge_params=True,
            )
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=float(alpha) * float(args.part_d_adam_lr), weight_decay=float(args.weight_decay))
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_source).float(), y_source.long())
        loss.backward()
        opt.step()
        source_after = actual_metrics(model, x_source, y_source)
        guard_after = actual_metrics(model, x_guard, y_guard)
        accepted, reject_reason = part_d_objective_trust_accept(part, source_before, source_after, guard_before, guard_after, args)
        if accepted:
            return float(alpha), 1, "accepted", source_after, guard_after, diag
    load_model_state(model, base_state)
    return 0.0, 0, reject_reason, source_before, guard_before, diag


def debt_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        "Brier_delta": after["brier"] - before["brier"],
        "ECE_delta": after["ece"] - before["ece"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "tail99_delta": after["tail99"] - before["tail99"],
        "margin10_delta": after["margin10"] - before["margin10"],
        "debt_delta": after["debt"] - before["debt"],
    }


def no_debt_ok(deltas: dict[str, float], budget: float) -> int:
    return int(
        deltas["Brier_delta"] <= float(budget)
        and deltas["ECE_delta"] <= float(budget)
        and deltas["tail95_delta"] <= float(budget)
        and deltas["tail99_delta"] <= float(budget)
        and deltas["margin10_delta"] >= -float(budget)
        and deltas["debt_delta"] <= float(budget)
    )


def run_part_d_row(job: tuple[str, str, str, int, float], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, part, task, seed, basis_input_gain = job
    start = time.time()
    dargs = part_d_run_args(args)
    basis_key = str(args.basis_key)
    data_seed = int(seed) if part == "D_C2" else int(seed) + 50000
    try:
        xtr, ytr, xg, yg = visual_data(task, data_seed, args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 23073000 + int(seed) * 1009 + sum(ord(c) for c in task + scheme + part)
        model = make_model(basis_key, int(args.depth), int(xtr.shape[1]), classes, model_seed, args, device, basis_input_gain=float(basis_input_gain))
        train_before = actual_metrics(model, xtr, ytr)
        guard_before = actual_metrics(model, xg, yg)
        finite_accept = 0
        finite_skip = 0
        finite_attempt = 0
        finite_scales: list[float] = []
        reject_reasons: dict[str, int] = {}
        solve_residuals: list[float] = []
        conds: list[float] = []
        rank_deficient = 0
        predicted_loss: list[float] = []
        predicted_fit: list[float] = []
        for step in range(1, int(args.part_d_steps) + 1):
            source_before = actual_metrics(model, xtr, ytr)
            step_guard_before = actual_metrics(model, xg, yg)
            base_state = model_state(model)
            diag: dict[str, Any] = {}
            if scheme == CONTROL_PART_D_NOOP:
                _deltas, diag = part_c_scheme_delta(model, xtr, ytr, xg, yg, str(args.part_d_residual_target), "D6_EFRF_L2_Wpop_gauss_seidel", dargs, device, seed=int(seed), task=task, checkpoint=part)
                load_model_state(model, base_state)
                alpha, accepted, reject_reason = 0.0, 0, "same_compute_noop"
                source_after, guard_after = source_before, step_guard_before
            elif scheme in {"D0_AdamW_coefficient_baseline", "D1_FunctionalGram_AdamW", "D2_FunctionalGram_BlockSNR_population_proxy", "D3_H10_known_structure_synthetic_control_proxy"}:
                alpha, accepted, reject_reason, source_after, guard_after, diag = choose_alpha_optimizer_step(
                    model, base_state, source_before, step_guard_before, xtr, ytr, xg, yg, scheme, part, args
                )
            else:
                deltas, diag = part_c_scheme_delta(model, xtr, ytr, xg, yg, str(args.part_d_residual_target), scheme, dargs, device, seed=int(seed), task=task, checkpoint=part)
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_for_deltas_part_d(
                    model, base_state, deltas, source_before, step_guard_before, xtr, ytr, xg, yg, part, args
                )
            finite_accept += int(accepted)
            finite_skip += int(not accepted)
            finite_attempt += 1
            finite_scales.append(float(alpha))
            reject_reasons[str(reject_reason)] = reject_reasons.get(str(reject_reason), 0) + 1
            solve_residuals.append(fval(diag.get("solve_residual_max")))
            conds.append(fval(diag.get("condition_number_after_ridge_max")))
            rank_deficient += ival(diag.get("design_rank_deficient_layers"))
            predicted_loss.append(fval(diag.get("predicted_guard_loss_delta_sum")))
            predicted_fit.append(fval(diag.get("predicted_guard_fit_mean")))
        train_after = actual_metrics(model, xtr, ytr)
        guard_after = actual_metrics(model, xg, yg)
        deltas = debt_deltas(guard_before, guard_after)
        component_row = int(
            deltas["Brier_delta"] <= float(args.no_debt_budget)
            and deltas["ECE_delta"] <= float(args.no_debt_budget)
            and deltas["tail95_delta"] <= float(args.no_debt_budget)
            and deltas["tail99_delta"] <= float(args.no_debt_budget)
            and deltas["margin10_delta"] >= -float(args.no_debt_budget)
        )
        return {
            "part": part,
            "status": "ok",
            "scheme": scheme,
            "candidate_EFRF": int(scheme in EFRF_PART_D_CANDIDATES),
            "basis_key": basis_key,
            "depth": int(args.depth),
            "task": task,
            "seed": int(seed),
            "data_seed": data_seed,
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "basis_input_gain": float(basis_input_gain),
            "population_weight_mode": str(args.population_weight_mode),
            "residual_target": str(args.part_d_residual_target),
            "lambda_key": f"{float(args.part_d_lambda):.0e}" if scheme not in {"D0_AdamW_coefficient_baseline", CONTROL_PART_D_FUNCTIONALGRAM, CONTROL_PART_D_BLOCKSNR, CONTROL_PART_D_H10, CONTROL_PART_D_NOOP} else "optimizer",
            "train_steps": int(args.part_d_steps),
            "C2_accuracy_initial": guard_before["accuracy"],
            "C2_accuracy_final": guard_after["accuracy"],
            "C2_accuracy_improvement": guard_after["accuracy"] - guard_before["accuracy"],
            "C2_coverage_initial": guard_before["coverage"],
            "C2_coverage_final": guard_after["coverage"],
            "C2_coverage_improvement": guard_after["coverage"] - guard_before["coverage"],
            "guard_nll_initial": guard_before["loss"],
            "guard_nll_final": guard_after["loss"],
            "guard_nll_delta": guard_after["loss"] - guard_before["loss"],
            "train_nll_delta": train_after["loss"] - train_before["loss"],
            "source_guard_gap": (train_after["loss"] - train_before["loss"]) - (guard_after["loss"] - guard_before["loss"]),
            **deltas,
            "component_non_positive_row": component_row,
            "F5_no_debt": no_debt_ok(deltas, float(args.no_debt_budget)),
            "finite_step_accept_count": finite_accept,
            "finite_step_skip_count": finite_skip,
            "finite_step_attempt_count": finite_attempt,
            "finite_step_accept_rate": finite_accept / max(1, finite_attempt),
            "finite_step_scale_mean": mean(finite_scales),
            "finite_step_reject_reasons": json.dumps(reject_reasons, sort_keys=True),
            "solve_residual_max": max(solve_residuals or [0.0]),
            "condition_number_after_ridge_max": max(conds or [0.0]),
            "design_rank_deficient_layers_total": rank_deficient,
            "predicted_guard_loss_delta_sum_mean": mean(predicted_loss),
            "predicted_guard_fit_mean": mean(predicted_fit),
            "blocksnr_population_proxy_observed_gradients": 0 if scheme == CONTROL_PART_D_BLOCKSNR else "",
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {
            "part": part,
            "status": "error",
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": int(args.depth),
            "task": task,
            "seed": int(seed),
            "basis_input_gain": float(basis_input_gain),
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
        }


def part_d(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pc = read_json(OUT_ROOT / "part_c_summary.json")
    if not ival(pc.get("gate_pass")):
        return write_blocked_summary("D", "D_BlockedByPartC", str(pc.get("dominant_blocker", "part_c_missing_or_failed")))
    device = device_from_args(args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix_path = OUT_ROOT / f"part_d_full_positive_control_matrix{suffix}.csv"
    summary_path = OUT_ROOT / f"part_d_summary{suffix}.json"
    jobs = shard_items(part_d_jobs(args), args)
    rows = read_rows(matrix_path) if int(args.part_d_resume) else []
    done = {
        (r.get("scheme"), r.get("part"), r.get("task"), str(r.get("seed")), str(r.get("basis_input_gain")), str(r.get("train_size")), str(r.get("guard_size")), str(r.get("train_steps")))
        for r in rows
        if r.get("status") == "ok"
    }
    flush_every = max(1, int(args.part_d_flush_every))
    for job_index, job in enumerate(jobs, start=1):
        scheme, part, task, seed, gain = job
        key = (scheme, part, task, str(seed), str(float(gain)), str(int(args.train_size)), str(int(args.guard_size)), str(int(args.part_d_steps)))
        if key in done:
            continue
        rows.append(run_part_d_row(job, args, device))
        if len(rows) % flush_every == 0:
            matrix = write_rows(matrix_path, rows)
            ok = [r for r in rows if r.get("status") == "ok"]
            partial = {
                "part": "D",
                "gate_pass": 0,
                "route": "PartDShardPartial",
                "dominant_blocker": "shard_in_progress",
                "row_count": len(rows),
                "ok_rows": len(ok),
                "error_rows": len(rows) - len(ok),
                "total_jobs_in_shard": len(jobs),
                "last_job_index_seen": job_index,
                "matrix": rel(matrix),
            }
            write_json(summary_path, partial)
            print(json.dumps({"part": "D", "shard": suffix or "single", "rows_written": len(rows), "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    matrix = write_rows(matrix_path, rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "part": "D",
        "gate_pass": 0,
        "route": "PartDShardOnly" if int(args.shard_count) > 1 else "PartDNeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "total_jobs_in_shard": len(jobs),
        "resume_used": int(args.part_d_resume),
        "matrix": rel(matrix),
    }
    write_json(summary_path, summary)
    append_exec("Part D", command_text(sys.argv), "done", files=rel(matrix), gpu=str(args.device), note=f"rows={len(rows)}")
    append_recap("Part D full positive-control shard", summary)
    return summary


def part_d_merge(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    shard_paths = sorted(OUT_ROOT.glob("part_d_full_positive_control_matrix_shard*_of_*.csv"))
    if shard_paths:
        rows = [row for path in shard_paths for row in read_rows(path)]
    else:
        rows = read_rows(OUT_ROOT / "part_d_full_positive_control_matrix.csv")
    matrix = write_rows(OUT_ROOT / "part_d_full_positive_control_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    c2_rows = [r for r in ok if r.get("part") == "D_C2"]
    f5_rows = [r for r in ok if r.get("part") == "D_F5"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("task")) for r in c2_rows}):
        group = [r for r in c2_rows if (r.get("scheme"), r.get("task")) == key]
        task_summaries.append(
            {
                "scheme": key[0],
                "task": key[1],
                "rows": len(group),
                "C2_coverage_improvement_median": median(r.get("C2_coverage_improvement") for r in group),
                "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group),
                "taskwise_pass": int(bool(group) and median(r.get("C2_coverage_improvement") for r in group) >= float(args.part_d_task_coverage_gate)),
            }
        )
    task_by = {(t["scheme"], t["task"]): t for t in task_summaries}
    schemes = sorted({r.get("scheme") for r in ok})

    def task_cov(scheme: str, task: str) -> float:
        return fval(task_by.get((scheme, task), {}).get("C2_coverage_improvement_median"))

    def scheme_cov(scheme: str) -> float:
        tasks = [t for t in task_summaries if t.get("scheme") == scheme]
        return median(t.get("C2_coverage_improvement_median") for t in tasks)

    fg_cov = scheme_cov(CONTROL_PART_D_FUNCTIONALGRAM)
    bs_cov = scheme_cov(CONTROL_PART_D_BLOCKSNR)
    rand_cov = scheme_cov(CONTROL_PART_D_RANDOM)
    noop_cov = scheme_cov(CONTROL_PART_D_NOOP)
    h10_cov = scheme_cov(CONTROL_PART_D_H10)
    fg_wall = median(r.get("wall_time_s") for r in ok if r.get("scheme") == CONTROL_PART_D_FUNCTIONALGRAM)
    groups: list[dict[str, Any]] = []
    for scheme in schemes:
        group = [r for r in ok if r.get("scheme") == scheme]
        c2 = [r for r in group if r.get("part") == "D_C2"]
        f5 = [r for r in group if r.get("part") == "D_F5"]
        local_cov = task_cov(str(scheme), "local_patch_interaction")
        rotation_cov = task_cov(str(scheme), "rotation_sensitive")
        cov = scheme_cov(str(scheme))
        local_rand_gap = local_cov - task_cov(CONTROL_PART_D_RANDOM, "local_patch_interaction")
        rotation_rand_gap = rotation_cov - task_cov(CONTROL_PART_D_RANDOM, "rotation_sensitive")
        random_gap = cov - rand_cov
        beats_fg = cov - fg_cov
        beats_bs = cov - bs_cov
        noop_gap = cov - noop_cov
        wall = median(r.get("wall_time_s") for r in group)
        overhead = wall / max(fg_wall, 1.0e-12) if fg_wall > 0 else 99.0
        f5_count = sum(ival(r.get("F5_no_debt")) for r in f5)
        component_count = sum(ival(r.get("component_non_positive_row")) for r in f5)
        accept = median(r.get("finite_step_accept_rate") for r in f5 or group)
        scale = median(r.get("finite_step_scale_mean") for r in f5 or group)
        skip = median(r.get("finite_step_skip_count") for r in f5 or group)
        pass_gate = int(
            scheme in EFRF_PART_D_CANDIDATES
            and local_cov >= float(args.part_d_task_coverage_gate)
            and rotation_cov >= float(args.part_d_task_coverage_gate)
            and local_rand_gap >= float(args.part_d_task_random_gap_gate)
            and rotation_rand_gap >= float(args.part_d_task_random_gap_gate)
            and random_gap >= float(args.part_d_overall_random_gap_gate)
            and beats_fg >= float(args.part_d_baseline_gap_gate)
            and beats_bs >= float(args.part_d_baseline_gap_gate)
            and f5_count >= int(args.part_d_f5_no_debt_gate)
            and component_count >= int(args.part_d_component_non_positive_gate)
            and accept >= float(args.part_d_accept_rate_gate)
            and scale >= float(args.part_d_scale_mean_gate)
            and skip <= float(args.part_d_skip_count_gate)
            and overhead <= float(args.part_d_overhead_gate)
        )
        groups.append(
            {
                "scheme": scheme,
                "rows": len(group),
                "candidate_EFRF": int(scheme in EFRF_PART_D_CANDIDATES),
                "basis_key": sorted({str(r.get("basis_key", "")) for r in group})[0] if group else "",
                "depth": sorted({str(r.get("depth", "")) for r in group})[0] if group else "",
                "train_steps": sorted({str(r.get("train_steps", "")) for r in group})[0] if group else "",
                "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in c2),
                "C2_coverage_improvement_median": cov,
                "local_patch_C2_coverage": local_cov,
                "rotation_C2_coverage": rotation_cov,
                "local_patch_random_projected_gap": local_rand_gap,
                "rotation_random_projected_gap": rotation_rand_gap,
                "overall_random_projected_gap": random_gap,
                "beats_FunctionalGram_gap": beats_fg,
                "beats_BlockSNR_gap": beats_bs,
                "gap_to_H10_known_structure": h10_cov - cov,
                "same_compute_noop_gap": noop_gap,
                "taskwise_all_pass": int(local_cov >= float(args.part_d_task_coverage_gate) and rotation_cov >= float(args.part_d_task_coverage_gate)),
                "F5_no_debt_count": f5_count,
                "component_non_positive_rows": component_count,
                "Brier_delta_median": median(r.get("Brier_delta") for r in f5),
                "ECE_delta_median": median(r.get("ECE_delta") for r in f5),
                "tail95_delta_median": median(r.get("tail95_delta") for r in f5),
                "tail99_delta_median": median(r.get("tail99_delta") for r in f5),
                "margin10_delta_median": median(r.get("margin10_delta") for r in f5),
                "finite_step_accept_rate_median": accept,
                "finite_step_scale_mean_median": scale,
                "finite_step_skip_count_median": skip,
                "random_veto_matched_gap": random_gap,
                "overhead_ratio_vs_FunctionalGram": overhead,
                "wall_time_s_median": wall,
                "solve_residual_max_median": median(r.get("solve_residual_max") for r in group),
                "condition_number_after_ridge_max_median": median(r.get("condition_number_after_ridge_max") for r in group),
                "scheme_gate_pass": pass_gate,
            }
        )
    task_csv = write_rows(OUT_ROOT / "part_d_full_positive_control_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_d_full_positive_control_group_summary.csv", groups)
    pass_groups = [g for g in groups if ival(g.get("scheme_gate_pass")) == 1]
    candidate_groups = [g for g in groups if ival(g.get("candidate_EFRF")) == 1]
    if pass_groups:
        gate, route, blocker = 1, "PartDFullPositiveControlPass", "none"
    else:
        gate = 0
        if any(ival(g.get("taskwise_all_pass")) == 0 for g in candidate_groups):
            route, blocker = "TaskwiseC2Failed", "local_or_rotation_c2_coverage_low"
        elif any(fval(g.get("overall_random_projected_gap")) < float(args.part_d_overall_random_gap_gate) for g in candidate_groups):
            route, blocker = "RandomProjectedResidualAssimilation", "random_projected_gap_low"
        elif any(fval(g.get("beats_FunctionalGram_gap")) < float(args.part_d_baseline_gap_gate) or fval(g.get("beats_BlockSNR_gap")) < float(args.part_d_baseline_gap_gate) for g in candidate_groups):
            route, blocker = "FunctionalGramOrBlockSNRDominates", "baseline_gap_low"
        elif any(ival(g.get("F5_no_debt_count")) < int(args.part_d_f5_no_debt_gate) or ival(g.get("component_non_positive_rows")) < int(args.part_d_component_non_positive_gate) for g in candidate_groups):
            route, blocker = "ResidualFlowUnsafe", "f5_no_debt_or_component_rows_low"
        elif any(fval(g.get("finite_step_accept_rate_median")) < float(args.part_d_accept_rate_gate) or fval(g.get("finite_step_scale_mean_median")) < float(args.part_d_scale_mean_gate) for g in candidate_groups):
            route, blocker = "SafetyDiagnosticOnlyHeavyRejection", "finite_step_accept_or_scale_low"
        elif any(fval(g.get("overhead_ratio_vs_FunctionalGram")) > float(args.part_d_overhead_gate) for g in candidate_groups):
            route, blocker = "SolverOverheadTooHigh", "overhead_ratio_high"
        else:
            route, blocker = "D_FullPositiveControlFailed", "no_fixed_efrf_scheme_passed"
    summary = {
        "part": "D",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "passing_scheme_groups": pass_groups,
        "candidate_groups": candidate_groups,
        "matrix": rel(matrix),
        "task_summary": rel(task_csv),
        "group_summary": rel(group_csv),
        "blocksnr_baseline_note": "D2 is a fixed FunctionalGram population-gate proxy; no held/test induction and no observer search were added.",
    }
    write_json(OUT_ROOT / "part_d_summary.json", summary)
    actions = [] if gate else [
        "decompose local_patch vs rotation failures taskwise",
        "compare last-layer/all-layer/Gauss-Seidel schedules already present through Part C and D4-D7",
        "if F5 fails, only repair objective-aligned finite-step trust scale/component budgets/margin sign",
        "if overhead fails, continue to Part F solver scaling rather than lowering gates",
    ]
    next_actions(
        "d",
        gate,
        blocker,
        actions,
        [f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-d --device cuda:0 --shard-count 2 --shard-index 0",
         f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-d --device cuda:0 --shard-count 2 --shard-index 1",
         f"{PYTHON} {rel(RUNNER)} --mode part-d-merge --device cpu"],
    )
    append_exec("Part D merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_d_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part D full positive-control merge", summary)
    return summary


def write_blocked_summary(part: str, route: str, blocker: str) -> dict[str, Any]:
    summary = {
        "part": part.upper(),
        "gate_pass": 0,
        "route": route,
        "dominant_blocker": blocker,
        "blocked": 1,
        "promotion_allowed": 0,
    }
    write_json(OUT_ROOT / f"part_{part.lower()}_summary.json", summary)
    next_actions(part.lower(), 0, blocker, ["resolve previous part blocker before running this part"])
    append_recap(f"Part {part.upper()} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    p0 = read_json(OUT_ROOT / "part_0_history_lock.json")
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    pb = read_json(OUT_ROOT / "part_b_frozen_residual_summary.json")
    pc = read_json(OUT_ROOT / "part_c_summary.json")
    pd = read_json(OUT_ROOT / "part_d_summary.json")
    pf = read_json(OUT_ROOT / "part_f_summary.json")
    pg = read_json(OUT_ROOT / "part_g_summary.json")
    if not ival(p0.get("gate_pass")):
        route, blocker = "A_SolverFailed", str(p0.get("dominant_blocker", "part0_failed"))
    elif not ival(pa.get("gate_pass")):
        route, blocker = "A_SolverFailed", str(pa.get("dominant_blocker", "part_a_failed"))
    elif not ival(pb.get("gate_pass")):
        route = str(pb.get("route", "B_FrozenResidualInverseFailed"))
        blocker = str(pb.get("dominant_blocker", "part_b_failed"))
    else:
        if pc and ival(pc.get("gate_pass")):
            if pd and ival(pd.get("gate_pass")):
                if pg and ival(pg.get("gate_pass")):
                    route, blocker = "G_LimitedRealTaskPreflightPass_NoOfficialYet", "none"
                else:
                    route, blocker = "D_PositiveControlPass_NoRealTaskYet", "part_f_or_part_g_not_executed"
            elif pd:
                route = FINAL_ROUTE_ALIASES.get(str(pd.get("route", "D_FullPositiveControlFailed")), str(pd.get("route", "D_FullPositiveControlFailed")))
                blocker = str(pd.get("dominant_blocker", "part_d_failed"))
            else:
                route, blocker = "C_OneStepActualPass_PartDRequired", "part_d_not_executed"
        else:
            route, blocker = "C_PendingAfterPartBPass", "part_c_not_executed"
    evidence = {
        "part_0": rel(OUT_ROOT / "part_0_history_lock.json"),
        "part_a": rel(OUT_ROOT / "part_a_summary.json"),
        "part_b": rel(OUT_ROOT / "part_b_frozen_residual_summary.json"),
    }
    if pc:
        evidence["part_c"] = rel(OUT_ROOT / "part_c_summary.json")
    if pd:
        evidence["part_d"] = rel(OUT_ROOT / "part_d_summary.json")
    skipped_by_plan: dict[str, str] = {}
    if pd and not ival(pd.get("gate_pass")):
        skipped_by_plan["part_f"] = "skipped_after_part_d_gate_failed"
        skipped_by_plan["part_g"] = "skipped_after_part_d_gate_failed"
    elif pf:
        evidence["part_f"] = rel(OUT_ROOT / "part_f_summary.json")
    if pg:
        evidence["part_g"] = rel(OUT_ROOT / "part_g_summary.json")
    summary = {
        "final_route": route,
        "dominant_blocker": blocker,
        "part_0_gate_pass": ival(p0.get("gate_pass")),
        "part_a_gate_pass": ival(pa.get("gate_pass")),
        "part_b_gate_pass": ival(pb.get("gate_pass")),
        "part_c_gate_pass": ival(pc.get("gate_pass")),
        "part_d_gate_pass": ival(pd.get("gate_pass")),
        "part_f_gate_pass": ival(pf.get("gate_pass")),
        "part_g_gate_pass": ival(pg.get("gate_pass")),
        "promotion_allowed": int(route == "G_LimitedRealTaskPreflightPass_NoOfficialYet"),
        "evidence": evidence,
        "skipped_by_plan": skipped_by_plan,
    }
    write_json(OUT_ROOT / "final_route.json", summary)
    append_exec("Finalize", command_text(sys.argv), "done", files=rel(OUT_ROOT / "final_route.json"), gpu=str(args.device), note=f"route={route}; blocker={blocker}")
    append_recap("Final route decision", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--basis-key", default="dche_k9")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--train-size", type=int, default=192)
    p.add_argument("--guard-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--checkpoint-steps", type=int, default=20)
    p.add_argument("--checkpoint-lr", type=float, default=0.02)
    p.add_argument("--quadrature-points", type=int, default=257)
    p.add_argument("--functional-gram-ridge", type=float, default=1.0e-6)
    p.add_argument("--lambda-default", type=float, default=1.0e-2)
    p.add_argument("--part-b-lambdas", default="1e-2")
    p.add_argument("--population-beta", type=float, default=1.0)
    p.add_argument("--population-weight-mode", default="sample_coherence")
    p.add_argument("--cg-tol", type=float, default=1.0e-6)
    p.add_argument("--cg-max-iter", type=int, default=512)
    p.add_argument("--part-a-n", type=int, default=96)
    p.add_argument("--part-a-m", type=int, default=24)
    p.add_argument("--part-a-outputs", type=int, default=5)
    p.add_argument("--part-b-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-b-seed-count", type=int, default=3)
    p.add_argument("--part-b-residual-targets", default="act,norm,gn")
    p.add_argument("--part-b-checkpoints", default="random_init,functionalgram_partial,h10_isomorphic_partial")
    p.add_argument("--part-b-basis-input-gains", default="")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-seed-count", type=int, default=3)
    p.add_argument("--part-c-residual-targets", default="norm")
    p.add_argument("--part-c-checkpoints", default="functionalgram_partial,h10_isomorphic_partial")
    p.add_argument("--part-c-basis-input-gains", default="")
    p.add_argument("--part-c-schemes", default=",".join(PART_C_SCHEMES))
    p.add_argument("--part-c-alphas", default="1,0.5,0.25,0.125")
    p.add_argument("--part-c-lambda", type=float, default=1.0)
    p.add_argument("--part-c-loss-margin", type=float, default=1.0e-3)
    p.add_argument("--part-c-functionalgram-coverage-margin", type=float, default=0.02)
    p.add_argument("--part-c-random-gap", type=float, default=0.03)
    p.add_argument("--part-c-trust-loss-tolerance", type=float, default=0.0)
    p.add_argument("--part-c-functionalgram-lr", type=float, default=0.02)
    p.add_argument("--part-d-parts", default="D_C2,D_F5")
    p.add_argument("--part-d-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-d-seed-count", type=int, default=15)
    p.add_argument("--part-d-f5-seed-count", type=int, default=15)
    p.add_argument("--part-d-steps", type=int, default=80)
    p.add_argument("--part-d-schemes", default=",".join(PART_D_SCHEMES))
    p.add_argument("--part-d-residual-target", default="norm")
    p.add_argument("--part-d-basis-input-gains", default="")
    p.add_argument("--part-d-lambda", type=float, default=1.0)
    p.add_argument("--part-d-alphas", default="1,0.5,0.25,0.125")
    p.add_argument("--part-d-trust-loss-tolerance", type=float, default=0.0)
    p.add_argument("--part-d-functionalgram-lr", type=float, default=0.02)
    p.add_argument("--part-d-adam-lr", type=float, default=0.02)
    p.add_argument("--part-d-flush-every", type=int, default=5)
    p.add_argument("--part-d-resume", type=int, default=1)
    p.add_argument("--no-debt-budget", type=float, default=0.0)
    p.add_argument("--part-d-task-coverage-gate", type=float, default=0.02)
    p.add_argument("--part-d-task-random-gap-gate", type=float, default=0.03)
    p.add_argument("--part-d-overall-random-gap-gate", type=float, default=0.05)
    p.add_argument("--part-d-baseline-gap-gate", type=float, default=0.03)
    p.add_argument("--part-d-f5-no-debt-gate", type=int, default=12)
    p.add_argument("--part-d-component-non-positive-gate", type=int, default=12)
    p.add_argument("--part-d-accept-rate-gate", type=float, default=0.50)
    p.add_argument("--part-d-scale-mean-gate", type=float, default=0.30)
    p.add_argument("--part-d-skip-count-gate", type=float, default=30.0)
    p.add_argument("--part-d-overhead-gate", type=float, default=2.5)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode).lower().replace("_", "-")
    if mode in {"init", "init-logs"}:
        init_logs()
        append_exec("Init", command_text(sys.argv), "done", files=f"{rel(EXEC_LOG)}; {rel(RECAP_LOG)}", gpu=str(args.device))
        return {"mode": mode}
    if mode in {"part-0", "part0"}:
        return part0(args)
    if mode in {"part-a", "parta"}:
        return part_a(args)
    if mode in {"part-b", "partb"}:
        return part_b(args)
    if mode in {"part-b-merge", "partb-merge"}:
        return part_b_merge(args)
    if mode in {"part-c", "partc"}:
        return part_c(args)
    if mode in {"part-c-merge", "partc-merge"}:
        return part_c_merge(args)
    if mode in {"part-d", "partd"}:
        return part_d(args)
    if mode in {"part-d-merge", "partd-merge"}:
        return part_d_merge(args)
    if mode in {"part-g", "partg"}:
        return write_blocked_summary("G", "G_BlockedByPartD", "part_d_not_passed")
    if mode == "finalize":
        return finalize(args)
    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
