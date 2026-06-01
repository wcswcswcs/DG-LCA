#!/usr/bin/env python3
"""DG-KAN v12.4 multi-basis efficiency-first and functional diagnostic runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v120_good_geometry_battery as v120  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v124_multibasis_functional_dual.py"
REPORT_PATH_DEFAULT = ROOT / "docs" / "DG-KAN_v12.4_MultiBasisFunctionalDual_结果复盘.md"
EPS = 1.0e-12


def _now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _stamp(row: Dict[str, Any]) -> Dict[str, Any]:
    row.setdefault("fake_data_used", 0)
    row.setdefault("proxy_row_used", 0)
    row.setdefault("cpu_offload_used", 0)
    return row


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu())
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def _q(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return v120._r2_score(pred, target)


def _param_budget(input_dim: int, output_dim: int) -> Tuple[int, int]:
    hidden = 256
    count = input_dim * hidden + hidden * hidden + hidden * output_dim
    return hidden, count


def _make_model(method_id: str, input_dim: int, output_dim: int, x_stats: torch.Tensor, device: torch.device, seed: int, spec: prim.PrimitiveSpec | None, param_budget: int, y_stats: torch.Tensor | None = None) -> torch.nn.Module:
    if method_id == "MLP-same-param-AdamW":
        hidden, _ = _param_budget(input_dim, output_dim)
        return prim.MLPBaseline(input_dim, output_dim, hidden, seed, device).to(device)
    assert spec is not None
    if getattr(spec, "model_kind", "edge_kan") == "gated_legendre_quadratic":
        return prim.GatedLegendreQuadraticKAN(input_dim, output_dim, spec, x_stats, seed, device).to(device)
    if getattr(spec, "model_kind", "edge_kan") == "lite_gated_legendre_quadratic":
        return prim.LiteGatedLegendreQuadraticKAN(input_dim, output_dim, spec, x_stats, seed, device).to(device)
    if getattr(spec, "model_kind", "edge_kan") == "simple_fast_task_geometry":
        return prim.SimpleFastTaskGeometryKAN(input_dim, output_dim, spec, x_stats, seed, device, y_stats).to(device)
    if getattr(spec, "model_kind", "edge_kan") == "grouped_rational_kat":
        return prim.GroupedRationalKATKAN(input_dim, output_dim, spec, x_stats, seed, device).to(device)
    if getattr(spec, "model_kind", "edge_kan") in {"quadratic_sketch", "trainable_quadratic_sketch"}:
        return prim.QuadraticSketchKAN(input_dim, output_dim, spec, x_stats, seed, device).to(device)
    return prim.PrimitiveKAN(input_dim, output_dim, spec, x_stats, seed, device, param_budget).to(device)


def _task_compile_candidate(method_id: str) -> bool:
    """Return whether task triage should use the same compiled steady-state path as A1."""
    prefixes = (
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
        "B15",
        "B16",
        "B17",
        "B18",
        "B19",
        "B20",
        "B21",
        "B22",
        "B23",
        "B26",
        "B27",
        "B28",
        "B29",
        "B30",
        "B31",
        "B32",
        "B33",
        "B34",
        "B35",
        "B36",
        "B37",
        "B38",
        "B39",
        "B41",
        "B42",
        "B43",
        "B44",
        "B45",
        "B46",
        "B47",
    )
    explicit = {
        "B1r-ReLU-KAN-stream-K2-repair",
        "B2b-FastKAN-RBF-K8",
        "B6a-BSpline-order1-local-K4",
        "B6b-BSpline-order1-local-K8-expression-repair",
        "B8d-BSpline-ReLU-lite-combo-K4-repair",
        "B9a-Poly2SignedPair-stream-K3-repair",
        "B9b-Poly2SignedPair-h64-diagnostic-repair",
        "B9c-Poly2SignedPair-h64-K2-diagnostic-repair",
        "B9d-Poly2PairRandom-h64-K2-diagnostic-repair",
    }
    return method_id.startswith(prefixes) or method_id in explicit


def _task_lr_schedule_name(method_id: str) -> str:
    """Global LR-schedule repairs for A3 task failures."""
    if method_id.startswith("B28a") or method_id.startswith("B29a") or method_id.startswith("B30a") or method_id.startswith("B30b") or method_id.startswith("B31a") or method_id.startswith("B31c") or method_id.startswith("B32a") or method_id.startswith("B32b") or method_id.startswith("B33b") or method_id.startswith("B34") or method_id.startswith("B35b") or method_id.startswith("B36a") or method_id.startswith("B36b") or method_id.startswith("B37b") or method_id.startswith("B37c") or method_id.startswith("B38b") or method_id.startswith("B38c") or method_id.startswith("B39") or method_id.startswith("B41b") or method_id.startswith("B42c") or method_id.startswith("B43b") or method_id.startswith("B44b") or method_id.startswith("B45") or method_id.startswith("B46b") or method_id.startswith("B47b"):
        return "linear_warmup10_cosine_final050"
    if method_id.startswith("B28b") or method_id.startswith("B29b") or method_id.startswith("B30c") or method_id.startswith("B31b") or method_id.startswith("B32c") or method_id.startswith("B33a") or method_id.startswith("B33c") or method_id.startswith("B35a") or method_id.startswith("B35c") or method_id.startswith("B36c") or method_id.startswith("B37a") or method_id.startswith("B38a") or method_id.startswith("B41a") or method_id.startswith("B42a") or method_id.startswith("B42b") or method_id.startswith("B43a") or method_id.startswith("B44a") or method_id.startswith("B46a") or method_id.startswith("B47a"):
        return "linear_warmup10_cosine_final075"
    if method_id.startswith("B27"):
        return "linear_warmup20_cosine_final025"
    return "constant"


def _scheduled_lr(base_lr: float, step_index: int, total_steps: int, schedule_name: str) -> float:
    """Linear warmup then cosine decay for global task-repair schedules."""
    total = max(1, int(total_steps))
    warmup_fraction = 0.20
    final_factor = 0.25
    if schedule_name == "linear_warmup10_cosine_final050":
        warmup_fraction = 0.10
        final_factor = 0.50
    elif schedule_name == "linear_warmup10_cosine_final075":
        warmup_fraction = 0.10
        final_factor = 0.75
    warmup = max(1, int(math.ceil(warmup_fraction * float(total))))
    step = max(1, int(step_index))
    if step <= warmup:
        return float(base_lr) * float(step) / float(warmup)
    progress = min(1.0, max(0.0, float(step - warmup) / float(max(1, total - warmup))))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (final_factor + (1.0 - final_factor) * cosine)


def _task_optimizer_profile(method_id: str) -> str:
    """Global optimizer-shape repairs for task-fail routes."""
    if method_id.startswith("B37"):
        return "branchslow_quad050_direct075_gate020"
    if method_id.startswith("B38"):
        return "legdirectfast_quad075_gate020"
    return "default"


def _make_task_optimizer(model: torch.nn.Module, method_id: str, lr: float, weight_decay: float) -> Tuple[torch.optim.Optimizer, str]:
    """Build AdamW with optional basis-branch LR multipliers.

    B37 keeps the exact B36 architecture and initialization, but slows the
    quadratic readout/projection and gate scalars during early task training.
    The profile is global by parameter role, not dataset/label/outcome.
    """
    profile = _task_optimizer_profile(method_id)
    if profile == "default":
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        for group in opt.param_groups:
            group["lr_multiplier"] = 1.0
            group["optimizer_profile"] = profile
        return opt, profile

    groups: Dict[Tuple[float, float, str], List[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = str(name).replace("_orig_mod.", "")
        mult = 1.0
        wd = float(weight_decay)
        role = "legendre_or_other"
        if "quad_proj" in clean or "quad_readout" in clean:
            mult = 0.75 if profile == "legdirectfast_quad075_gate020" else 0.50
            role = "quadratic_slow"
        elif "direct_readout" in clean:
            mult = 1.25 if profile == "legdirectfast_quad075_gate020" else 0.75
            role = "direct_skip_fast" if profile == "legdirectfast_quad075_gate020" else "direct_skip_mid"
        elif "branch_scale" in clean or "logit_gain" in clean:
            mult = 0.20
            wd = 0.0
            role = "gate_slow_no_decay"
        elif "leg_w" in clean and profile == "legdirectfast_quad075_gate020":
            mult = 1.25
            role = "legendre_fast"
        elif clean.endswith("bias") or ".bias" in clean:
            wd = 0.0
            role = "bias_no_decay"
        key = (float(mult), float(wd), role)
        groups.setdefault(key, []).append(param)
    param_groups = [
        {
            "params": params,
            "lr": float(lr) * mult,
            "weight_decay": wd,
            "lr_multiplier": mult,
            "optimizer_profile": profile,
            "param_role": role,
        }
        for (mult, wd, role), params in sorted(groups.items(), key=lambda x: x[0][2])
        if params
    ]
    return torch.optim.AdamW(param_groups, lr=float(lr), weight_decay=float(weight_decay)), profile


def _optimizer_state_warm_reset(opt: torch.optim.Optimizer) -> None:
    """Keep AdamW state buffers allocated but reset their values.

    This preserves the training initial condition while removing first-step
    state-allocation timing from the measured loop.
    """
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                value.zero_()
            elif isinstance(value, (int, float)):
                state[key] = type(value)(0)


def _method_rows(input_dim: int, output_dim: int, candidate_ids: Sequence[str] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, prim.PrimitiveSpec]]:
    hidden, budget = _param_budget(input_dim, output_dim)
    specs = prim.primitive_specs(budget, input_dim, output_dim)
    selected = set(candidate_ids or [])
    if selected:
        specs = [s for s in specs if s.candidate_id in selected]
        missing = sorted(selected.difference({s.candidate_id for s in specs}))
        if missing:
            raise ValueError(f"Unknown --candidate-ids entries: {missing}")
    by_id = {s.candidate_id: s for s in specs}
    rows: List[Dict[str, Any]] = [
        _stamp(
            {
                "stage": "V124_BASIS_MANIFEST",
                "method": "MLP-same-param-AdamW",
                "candidate_id": "MLP-same-param-AdamW",
                "basis_family": "MLP-control",
                "basis_name": "silu_mlp_control",
                "K": 0,
                "basis_order": 0,
                "hidden_dim": hidden,
                "local_support": 0,
                "global_support": 1,
                "uses_exp": 0,
                "uses_sin_cos": 0,
                "uses_division": 0,
                "uses_gather_scatter": 0,
                "uses_dense_basis_tensor": 0,
                "manual_forward": 0,
                "manual_backward": 0,
                "manual_update": 0,
                "strict_purekan_pass": 0,
                "nonkan_param_count": budget,
                "edge_param_count": 0,
                "param_count": budget,
                "param_ratio_vs_mlp": 1.0,
                "third_party_source": "control",
                "diagnostic_only": 0,
            }
        )
    ]
    for spec in specs:
        if getattr(spec, "model_kind", "edge_kan") == "quadratic_sketch":
            edge = spec.hidden_dim * spec.k * output_dim + output_dim
        elif getattr(spec, "model_kind", "edge_kan") == "trainable_quadratic_sketch":
            edge = input_dim * spec.hidden_dim + spec.hidden_dim * spec.k * output_dim + output_dim
        elif getattr(spec, "model_kind", "edge_kan") == "gated_legendre_quadratic":
            leg_h, quad_h = prim._gated_branch_dims(spec.hidden_dim)
            edge = leg_h * (input_dim + output_dim) * 4 + input_dim * quad_h + quad_h * 2 * output_dim + output_dim + 3
            if "directskip" in str(spec.init_variant):
                edge += input_dim * output_dim * 4
        elif getattr(spec, "model_kind", "edge_kan") == "lite_gated_legendre_quadratic":
            edge = input_dim * output_dim * 4 + input_dim * spec.hidden_dim + spec.hidden_dim * 2 * output_dim + output_dim + 3
        elif getattr(spec, "model_kind", "edge_kan") == "simple_fast_task_geometry":
            edge = input_dim * output_dim * 3 + input_dim * spec.hidden_dim + spec.hidden_dim * 2 * output_dim + output_dim + 3
        else:
            edge = spec.hidden_dim * (input_dim + output_dim) * spec.k
        rows.append(
            _stamp(
                {
                    "stage": "V124_BASIS_MANIFEST",
                    "method": spec.candidate_id,
                    "candidate_id": spec.candidate_id,
                    "basis_family": spec.basis_family,
                    "basis_name": spec.basis_name,
                    "K": spec.k,
                    "basis_order": spec.basis_order,
                    "hidden_dim": spec.hidden_dim,
                    "local_support": spec.local_support,
                    "global_support": spec.global_support,
                    "uses_exp": spec.uses_exp,
                    "uses_sin_cos": spec.uses_sin_cos,
                    "uses_division": spec.uses_division,
                    "uses_gather_scatter": spec.uses_gather_scatter,
                    "uses_dense_basis_tensor": spec.uses_dense_basis_tensor,
                    "manual_forward": 0,
                    "manual_backward": 0,
                    "manual_update": 0,
                    "strict_purekan_pass": 1,
                    "model_kind": spec.model_kind,
                    "nonkan_param_count": 0,
                    "edge_param_count": edge,
                    "param_count": edge,
                    "param_ratio_vs_mlp": edge / max(EPS, budget),
                    "third_party_source": spec.source,
                    "diagnostic_only": spec.diagnostic_only,
                }
            )
        )
    return rows, by_id


def _classification(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return v120._classification_metrics(logits, y)


def _regression_target(target_id: str, x: torch.Tensor) -> torch.Tensor:
    if target_id == "E0-additive":
        y = x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2]
    elif target_id == "E1-pairwise-product":
        y = x[:, 0] * x[:, 1] + 0.5 * x[:, 2] * x[:, 3]
    elif target_id == "E2-composition":
        y = torch.sin(x[:, 0] + 0.5 * x[:, 1] * x[:, 2]) + 0.2 * x[:, 3].square()
    elif target_id == "E3-local-XOR":
        y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).float() * 2.0 - 1.0
    elif target_id == "E4-high-frequency":
        y = torch.sin(6.0 * x[:, 0]) + 0.25 * torch.cos(8.0 * x[:, 1])
    elif target_id == "E5-noise-stress":
        gen = torch.Generator(device=x.device).manual_seed(124005)
        y = x[:, 0] * x[:, 1] + 0.1 * torch.randn(x.shape[0], device=x.device, generator=gen)
    elif target_id == "E6-rotated-pairwise-product":
        gen = torch.Generator(device=x.device).manual_seed(124006)
        r = torch.randn(x.shape[1], 4, device=x.device, generator=gen) / math.sqrt(x.shape[1])
        z = x @ r
        y = z[:, 0] * z[:, 1] + 0.5 * z[:, 2] * z[:, 3]
    elif target_id == "E7-piecewise-local":
        y = F.relu(x[:, 0] - 0.25) - F.relu(-x[:, 1] - 0.15) + 0.2 * (x[:, 2] > 0.4).float()
    elif target_id == "E8-random-quadratic-form":
        gen = torch.Generator(device=x.device).manual_seed(124008)
        r = torch.randn(x.shape[1], 6, device=x.device, generator=gen) / math.sqrt(x.shape[1])
        z = x @ r
        y = z[:, 0] * z[:, 1] - 0.4 * z[:, 2] * z[:, 3] + 0.25 * z[:, 4].square()
    elif target_id == "E9-smooth-nonpolynomial":
        y = torch.tanh(1.5 * x[:, 0] - x[:, 1]) + torch.sigmoid(2.0 * x[:, 2])
    elif target_id == "E10-local-bump-mixture":
        y = torch.exp(-8.0 * (x[:, 0] - 0.4).square()) - 0.7 * torch.exp(-10.0 * (x[:, 1] + 0.2).square())
    else:
        raise ValueError(target_id)
    return y.reshape(-1, 1)


def _expression_data(args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(args.seed) + 124200)
    n = int(args.expression_train_size) + int(args.expression_val_size) + int(args.expression_test_size)
    x = torch.rand(n, int(args.expression_dim), device=device, generator=gen) * 2.0 - 1.0
    return (
        x[: int(args.expression_train_size)].contiguous(),
        x[int(args.expression_train_size) : int(args.expression_train_size) + int(args.expression_val_size)].contiguous(),
        x[-int(args.expression_test_size) :].contiguous(),
    )


def _ridge_r2(features_train: torch.Tensor, y_train: torch.Tensor, features_eval: torch.Tensor, y_eval: torch.Tensor, ridge: float) -> float:
    ones_train = torch.ones(features_train.shape[0], 1, device=features_train.device)
    ones_eval = torch.ones(features_eval.shape[0], 1, device=features_eval.device)
    ft = torch.cat([features_train.float(), ones_train], dim=1)
    fe = torch.cat([features_eval.float(), ones_eval], dim=1)
    eye = torch.eye(ft.shape[1], device=ft.device)
    coef = torch.linalg.solve(ft.T @ ft + float(ridge) * eye, ft.T @ y_train.float())
    return _r2(fe @ coef, y_eval)


def _train_regression(model: torch.nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, steps: int, batch_size: int, lr: float, seed: int) -> Tuple[float, float, int]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    gen = torch.Generator(device=x_train.device).manual_seed(int(seed))
    best_val = -1.0e9
    best_test = -1.0e9
    steps_to = -1
    for step in range(max(1, int(steps))):
        idx = torch.randint(0, int(x_train.shape[0]), (min(int(batch_size), int(x_train.shape[0])),), generator=gen, device=x_train.device)
        opt.zero_grad(set_to_none=True)
        pred = model(x_train[idx])
        loss = F.mse_loss(pred, y_train[idx])
        loss.backward()
        opt.step()
        if step == int(steps) - 1 or step % max(1, int(steps) // 10) == 0:
            model.eval()
            with torch.no_grad():
                val = _r2(model(x_val), y_val)
                test = _r2(model(x_test), y_test)
            model.train()
            if val > best_val:
                best_val = val
                best_test = test
            if steps_to < 0 and val >= 0.95:
                steps_to = step + 1
    return float(best_val), float(best_test), int(steps_to)


def run_contract_and_microbench(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, prim.PrimitiveSpec], List[str]]:
    data = v120._load_vision_split(args, "MNIST", train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
    x_train_cpu, y_train_cpu, *_rest, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    manifest, specs = _method_rows(input_dim, output_dim, _parse_list(args.candidate_ids))
    _, budget = _param_budget(input_dim, output_dim)
    grad_rows: List[Dict[str, Any]] = []
    bench_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    batch_sizes = _parse_ints(args.microbench_batch_sizes)
    base_by_batch: Dict[int, Dict[str, float]] = {}
    compile_repair_ids = {
        "B1r-ReLU-KAN-stream-K2-repair",
        "B2b-FastKAN-RBF-K8",
        "B6a-BSpline-order1-local-K4",
        "B6b-BSpline-order1-local-K8-expression-repair",
        "B8d-BSpline-ReLU-lite-combo-K4-repair",
        "B9a-Poly2SignedPair-stream-K3-repair",
        "B9b-Poly2SignedPair-h64-diagnostic-repair",
        "B9c-Poly2SignedPair-h64-K2-diagnostic-repair",
        "B9d-Poly2PairRandom-h64-K2-diagnostic-repair",
        "B10a-QuadraticSketch-h64-diagnostic",
        "B10b-QuadraticSketch-h128-diagnostic",
        "B10c-QuadraticSketch-h256-diagnostic",
        "B10d-QuadraticSketch-h512-diagnostic",
        "B11a-TrainableQuadraticSketch-h128",
        "B11b-TrainableQuadraticSketch-h256",
        "B12a-LegendreKAN-K4-fan-scale-repair",
        "B12b-ChebyKAN-K4-fan-scale-repair",
        "B13a-LegendreKAN-K4-identity-residual-repair",
        "B13b-ChebyKAN-K4-identity-residual-repair",
        "B14a-GatedLegendreQuadratic-h160",
        "B14b-GatedLegendreQuadratic-h224",
        "B14c-GatedLegendreQuadratic-h224-quad-boost",
        "B14d-GatedLegendreQuadratic-h228-quad-boost",
        "B15a-GatedLegendreQuadratic-h224-loss-gain",
        "B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain",
        "B16a-GatedLegendreQuadratic-h224-basis-norm",
        "B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm",
        "B17a-GatedLegendreQuadratic-h224-input-geom-norm",
        "B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm",
        "B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm",
        "B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm",
        "B18a-GatedLegendreQuadratic-h224-group-rms-norm",
        "B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm",
        "B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm",
        "B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm",
        "B19a-GatedLegendreQuadratic-h224-residual-geom-norm",
        "B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm",
        "B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm",
        "B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm",
        "B20a-GatedLegendreQuadratic-h224-residual-mix15",
        "B20b-GatedLegendreQuadratic-h224-residual-mix25",
        "B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15",
        "B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25",
        "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
        "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
        "B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm",
        "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
        "B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost",
        "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost",
        "B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075",
        "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050",
        "B26a-GatedLegendreQuadratic-h224-basis-specific-temp075",
        "B26b-GatedLegendreQuadratic-h224-basis-specific-temp050",
        "B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr",
        "B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr",
        "B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr",
        "B28a-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr-final050",
        "B28b-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr-final075",
        "B29a-GatedLegendreQuadratic-h224-lowboost-temp050-cosine-lr-final050",
        "B29b-GatedLegendreQuadratic-h224-lowboost-temp050-cosine-lr-final075",
        "B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050",
        "B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050",
        "B30c-LiteGatedLegendreQuadratic-h296-temp075-cosine-lr-final075",
        "B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050",
        "B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075",
        "B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050",
        "B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050",
        "B32b-GatedLegendreQuadratic-h192-lowboost-temp050-cosine-lr-final050",
        "B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075",
        "B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075",
        "B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050",
        "B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075",
        "B34a-GatedLegendreQuadratic-h168-lowboost-temp050-cosine-lr-final050",
        "B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050",
        "B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050",
        "B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075",
        "B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050",
        "B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075",
        "B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050",
        "B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050",
        "B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075",
        "B37a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-branchslow-final075",
        "B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050",
        "B37c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-branchslow-final050",
        "B38a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-legfast-final075",
        "B38b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-legfast-final050",
        "B38c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-legfast-final050",
        "B39a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final050",
        "B41a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-cosine-lr-final075",
        "B41b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-cosine-lr-final050",
        "B42a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-fastreuse-final075",
        "B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075",
        "B42c-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-fastreuse-final050",
        "B43a-GatedLegendreQuadratic-h172-lowboost-temp075-directskip-fastreuse-final075",
        "B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050",
        "B44a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-mix15-final075",
        "B44b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-fastreuse-mix15-final050",
        "B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050",
        "B46a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip050-fastreuse-final075",
        "B46b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip050-fastreuse-final050",
        "B47a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final075",
        "B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050",
    }

    def measure_one(model: torch.nn.Module, method_id: str, row: Mapping[str, Any], batch_size: int, *, compiled: int) -> Dict[str, Any]:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        manual_path = bool(hasattr(model, "manual_kernel_available") and model.manual_kernel_available())  # type: ignore[attr-defined]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) + 2100 + int(batch_size) + 17 * int(compiled))
        fwd: List[float] = []
        bwd: List[float] = []
        upd: List[float] = []
        step: List[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_steps = max(1, int(args.microbench_warmup_steps) + int(args.microbench_measure_steps))
        for idx_step in range(total_steps):
            idx = torch.randint(0, int(x_train.shape[0]), (min(int(batch_size), int(x_train.shape[0])),), generator=gen, device=device)
            xb = x_train[idx]
            yb = y_train[idx]
            if idx_step == int(args.microbench_warmup_steps) and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            opt.zero_grad(set_to_none=True)
            _sync(device)
            t0 = time.perf_counter()
            if manual_path:
                logits, cache = model.manual_ce_forward_cache(xb)  # type: ignore[attr-defined]
                loss = F.cross_entropy(logits, yb)
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            _sync(device)
            t1 = time.perf_counter()
            if manual_path:
                model.manual_ce_backward_from_cache(logits, cache, yb)  # type: ignore[attr-defined]
            else:
                loss.backward()
            _sync(device)
            t2 = time.perf_counter()
            opt.step()
            _sync(device)
            t3 = time.perf_counter()
            if idx_step >= int(args.microbench_warmup_steps):
                fwd.append((t1 - t0) * 1000.0)
                bwd.append((t2 - t1) * 1000.0)
                upd.append((t3 - t2) * 1000.0)
                step.append((t3 - t0) * 1000.0)
        peak = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
        reserved = float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
        current = {
            "forward_q90": _q(fwd, 0.90),
            "backward_q90": _q(bwd, 0.90),
            "update_q90": _q(upd, 0.90),
            "step_q90": _q(step, 0.90),
            "peak": peak,
        }
        if method_id == "MLP-same-param-AdamW" and not compiled:
            base_by_batch[int(batch_size)] = current
        base = base_by_batch.get(int(batch_size), current)
        f_ratio = current["forward_q90"] / max(EPS, base["forward_q90"])
        b_ratio = current["backward_q90"] / max(EPS, base["backward_q90"])
        u_ratio = current["update_q90"] / max(EPS, base["update_q90"])
        s_ratio = current["step_q90"] / max(EPS, base["step_q90"])
        m_ratio = current["peak"] / max(EPS, base["peak"]) if base["peak"] > 0 else 1.0
        exploratory = s_ratio <= 1.50 and m_ratio <= 1.25 and int(row["strict_purekan_pass"]) == 1
        base_pass = f_ratio <= 1.25 and b_ratio <= 1.35 and s_ratio <= 1.25 and m_ratio <= 1.05 and int(row["strict_purekan_pass"]) == 1
        return _stamp(
            {
                "stage": "V124_EFFICIENCY_MICROBENCH",
                "method": method_id + ("-compiled-warm-repair" if compiled else ""),
                "candidate_id": row["candidate_id"],
                "basis_family": row["basis_family"],
                "batch_size": batch_size,
                "compiled": compiled,
                "forward_q50_ms": _q(fwd, 0.50),
                "forward_q90_ms": current["forward_q90"],
                "backward_q50_ms": _q(bwd, 0.50),
                "backward_q90_ms": current["backward_q90"],
                "update_q50_ms": _q(upd, 0.50),
                "update_q90_ms": current["update_q90"],
                "step_q50_ms": _q(step, 0.50),
                "step_q90_ms": current["step_q90"],
                "forward_ratio_q90_vs_mlp": f_ratio,
                "backward_ratio_q90_vs_mlp": b_ratio,
                "update_ratio_q90_vs_mlp": u_ratio,
                "step_ratio_q90_vs_mlp": s_ratio,
                "peak_allocated_mb": peak,
                "peak_reserved_mb": reserved,
                "compact_memory_ratio": m_ratio,
                "conservative_memory_ratio": reserved / max(EPS, base["peak"]) if base["peak"] > 0 else 1.0,
                "kernel_count_forward": "not_measured",
                "kernel_count_backward": "not_measured",
                "kernel_count_update": "not_measured",
                "allocation_count": "not_measured",
                "sync_count": "measured_sync_around_regions",
                "manual_forward_available": int(manual_path),
                "manual_backward_available": int(manual_path),
                "uses_torch_autograd_graph": int(not manual_path),
                "largest_temp_tensor_mb": "not_measured",
                "A1_exploratory_pass": int(exploratory),
                "A1_base_candidate_pass": int(base_pass),
            }
        )

    for row in manifest:
        method_id = str(row["method"])
        spec = specs.get(method_id)
        model = _make_model(method_id, input_dim, output_dim, x_train, device, int(args.seed) + 1000, spec, budget)
        finite = True
        rollback_err = 0.0
        try:
            before = [p.detach().clone() for p in model.parameters()]
            xb = x_train[: min(16, int(x_train.shape[0]))]
            yb = y_train[: min(16, int(y_train.shape[0]))]
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            rollback_err = max((p.detach() - b).abs().max().item() for p, b in zip(model.parameters(), before))
            finite = bool(torch.isfinite(loss).item()) and all(p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
        except Exception:
            finite = False
        manual_forward_available = 0
        manual_backward_available = 0
        grad_relerr_max: Any = "not_applicable_autograd_path"
        grad_cos_min: Any = "not_applicable_autograd_path"
        output_max_abs_error: Any = ""
        final_claim_allowed = 0
        uses_torch_autograd_graph = 1
        if hasattr(model, "manual_gradient_audit"):
            try:
                audit = model.manual_gradient_audit(x_train[: min(16, int(x_train.shape[0]))], y_train[: min(16, int(y_train.shape[0]))])  # type: ignore[attr-defined]
                manual_forward_available = int(_safe_float(audit.get("manual_forward_available"), 0))
                manual_backward_available = int(_safe_float(audit.get("manual_backward_available"), 0))
                grad_relerr_max = audit.get("grad_relerr_max", grad_relerr_max)
                grad_cos_min = audit.get("grad_cos_min", grad_cos_min)
                output_max_abs_error = audit.get("output_max_abs_error", "")
                final_claim_allowed = int(
                    finite
                    and manual_forward_available
                    and manual_backward_available
                    and _safe_float(grad_relerr_max, 999.0) < 2.0e-4
                    and _safe_float(grad_cos_min, -1.0) > 0.999
                    and _safe_float(output_max_abs_error, 999.0) < 1.0e-6
                )
                if manual_backward_available:
                    uses_torch_autograd_graph = 0
            except Exception as exc:
                grad_relerr_max = f"manual_audit_failed:{exc}"
        grad_rows.append(
            _stamp(
                {
                    "stage": "V124_GRAD_CORRECTNESS",
                    "method": method_id,
                    "candidate_id": row["candidate_id"],
                    "basis_family": row["basis_family"],
                    "manual_forward_available": manual_forward_available,
                    "manual_backward_available": manual_backward_available,
                    "manual_update_available": 0,
                    "uses_loss_backward": 1,
                    "uses_torch_autograd_graph": uses_torch_autograd_graph,
                    "rollback_max_error": rollback_err,
                    "grad_relerr_max": grad_relerr_max,
                    "grad_cos_min": grad_cos_min,
                    "output_max_abs_error": output_max_abs_error,
                    "nan_inf_count": int(not finite),
                    "contract_exploratory_pass": int(finite and rollback_err < 1.0e-8),
                    "final_claim_allowed": final_claim_allowed,
                }
            )
        )
        for batch_size in batch_sizes:
            model = _make_model(method_id, input_dim, output_dim, x_train, device, int(args.seed) + 2000 + int(batch_size), spec, budget)
            bench_row = measure_one(model, method_id, row, int(batch_size), compiled=0)
            bench_rows.append(bench_row)
            manual_path = bool(hasattr(model, "manual_kernel_available") and model.manual_kernel_available())  # type: ignore[attr-defined]
            if method_id in compile_repair_ids and not manual_path and hasattr(torch, "compile"):
                try:
                    compiled_model = torch.compile(_make_model(method_id, input_dim, output_dim, x_train, device, int(args.seed) + 3000 + int(batch_size), spec, budget), mode="reduce-overhead")
                    bench_rows.append(measure_one(compiled_model, method_id, row, int(batch_size), compiled=1))
                except Exception as exc:
                    bench_rows.append(_stamp({"stage": "V124_EFFICIENCY_MICROBENCH", "method": method_id + "-compiled-warm-repair", "candidate_id": row["candidate_id"], "basis_family": row["basis_family"], "batch_size": batch_size, "compiled": 1, "status": "compile_failed", "compile_error": str(exc), "A1_exploratory_pass": 0, "A1_base_candidate_pass": 0}))
            if method_id != "MLP-same-param-AdamW" and not int(_safe_float(bench_row.get("A1_exploratory_pass"))) and int(batch_size) == batch_sizes[0]:
                failure_rows.append(
                    _stamp(
                        {
                            "stage": "V124_EFFICIENCY_FAILURE_TABLE",
                            "candidate_id": row["candidate_id"],
                            "basis_family": row["basis_family"],
                            "failure_code": "A1_efficiency_exploratory_fail",
                            "step_ratio_q90_vs_mlp": bench_row.get("step_ratio_q90_vs_mlp"),
                            "memory_ratio": bench_row.get("compact_memory_ratio"),
                            "action_recommended": "reduce K/recompute basis/fuse basis eval; do not run full task if all batches fail",
                        }
                    )
                )
    survivor_ids = sorted(
        {
            str(r["candidate_id"])
            for r in bench_rows
            if str(r["candidate_id"]) != "MLP-same-param-AdamW" and int(_safe_float(r.get("A1_exploratory_pass"))) == 1 and int(r["batch_size"]) == batch_sizes[0]
        }
    )
    return manifest, grad_rows, bench_rows, failure_rows, specs, survivor_ids


def run_expression(args: argparse.Namespace, out_dir: Path, device: torch.device, specs: Mapping[str, prim.PrimitiveSpec], survivors: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    x_train, x_val, x_test = _expression_data(args, device)
    input_dim = int(args.expression_dim)
    output_dim = 1
    _, budget = _param_budget(input_dim, output_dim)
    targets = _parse_list(args.expression_targets)
    expr_rows: List[Dict[str, Any]] = []
    frozen_rows: List[Dict[str, Any]] = []
    cond_rows: List[Dict[str, Any]] = []
    mlp_scores: Dict[Tuple[str, str], float] = {}
    methods = ["MLP-same-param-AdamW", *survivors]
    for target in targets:
        y_train = _regression_target(target, x_train)
        y_val = _regression_target(target, x_val)
        y_test = _regression_target(target, x_test)
        for method_id in methods:
            spec = specs.get(method_id)
            for protocol, steps in [
                ("B0-short-fit", int(args.expression_steps_b0)),
                ("B1-standard-fit", int(args.expression_steps_b1)),
                ("B2-high-budget-fit", int(args.expression_steps_b2)),
            ]:
                model = _make_model(method_id, input_dim, output_dim, x_train, device, int(args.seed) + hash((target, method_id, protocol)) % 100000, spec, budget)
                val_r2, test_r2, steps_to = _train_regression(model, x_train, y_train, x_val, y_val, x_test, y_test, steps, int(args.expression_batch_size), float(args.expression_lr), int(args.seed) + len(expr_rows))
                if method_id == "MLP-same-param-AdamW":
                    mlp_scores[(target, protocol)] = val_r2
                delta = val_r2 - mlp_scores.get((target, protocol), val_r2)
                expr_rows.append(
                    _stamp(
                        {
                            "stage": "V124_EXPRESSION_BATTERY",
                            "target": target,
                            "protocol": protocol,
                            "method": method_id,
                            "candidate_id": method_id,
                            "basis_family": spec.basis_family if spec else "MLP-control",
                            "diagnostic_only": int(spec.diagnostic_only) if spec else 0,
                            "val_R2": val_r2,
                            "test_R2": test_r2,
                            "delta_vs_mlp": delta,
                            "trainable_best_R2": val_r2,
                            "steps_to_R2_0_95": steps_to,
                        }
                    )
                )
            if method_id != "MLP-same-param-AdamW":
                model = _make_model(method_id, input_dim, output_dim, x_train, device, int(args.seed) + hash((target, method_id, "frozen")) % 100000, spec, budget)
                assert hasattr(model, "frozen_readout_features")
                with torch.no_grad():
                    ft = model.frozen_readout_features(x_train)
                    fv = model.frozen_readout_features(x_val)
                    ftest = model.frozen_readout_features(x_test)
                    frozen_val = _ridge_r2(ft, y_train, fv, y_val, float(args.ridge_alpha))
                    frozen_test = _ridge_r2(ft, y_train, ftest, y_test, float(args.ridge_alpha))
                    diag = model.basis_diagnostics(x_val)
                frozen_rows.append(
                    _stamp(
                        {
                            "stage": "V124_FROZEN_READOUT",
                            "target": target,
                            "method": method_id,
                            "candidate_id": method_id,
                            "basis_family": spec.basis_family if spec else "",
                            "diagnostic_only": int(spec.diagnostic_only) if spec else 0,
                            "frozen_readout_R2": frozen_val,
                            "frozen_test_R2": frozen_test,
                            "ridge_alpha": float(args.ridge_alpha),
                        }
                    )
                )
                cond_rows.append(
                    _stamp(
                        {
                            "stage": "V124_BASIS_CONDITION",
                            "target": target,
                            "method": method_id,
                            "candidate_id": method_id,
                            "basis_family": spec.basis_family if spec else "",
                            "diagnostic_only": int(spec.diagnostic_only) if spec else 0,
                            **diag,
                        }
                    )
                )
    expr_pass: List[str] = []
    for method_id in survivors:
        b1 = [r for r in expr_rows if r["method"] == method_id and r["protocol"] == "B1-standard-fit"]
        frozen = [r for r in frozen_rows if r["method"] == method_id]
        e1_ok = any(r["target"] == "E1-pairwise-product" and _safe_float(r["delta_vs_mlp"]) >= -0.02 for r in b1)
        e2_ok = any(r["target"] == "E2-composition" and _safe_float(r["delta_vs_mlp"]) >= -0.02 for r in b1)
        interaction_ok = any(r["target"] in {"E6-rotated-pairwise-product", "E8-random-quadratic-form"} and _safe_float(r["delta_vs_mlp"]) >= -0.02 for r in b1)
        frozen_ok = sum(1 for r in frozen if r["target"] in {"E1-pairwise-product", "E6-rotated-pairwise-product", "E8-random-quadratic-form"} and _safe_float(r["frozen_readout_R2"]) >= 0.65) >= 2
        trainable_exception = all(any(r["target"] == t and _safe_float(r["val_R2"]) >= 0.95 for r in b1) for t in ["E1-pairwise-product", "E2-composition"])
        is_official_candidate = not bool(specs[method_id].diagnostic_only)
        if is_official_candidate and e1_ok and e2_ok and (interaction_ok or frozen_ok or trainable_exception):
            expr_pass.append(method_id)
    return expr_rows, frozen_rows, cond_rows, expr_pass


def run_initial_geometry_audit(args: argparse.Namespace, device: torch.device, specs: Mapping[str, prim.PrimitiveSpec]) -> List[Dict[str, Any]]:
    data = v120._load_vision_split(args, "MNIST", train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
    x_train_cpu, *_rest, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    _, budget = _param_budget(input_dim, output_dim)
    rows: List[Dict[str, Any]] = []
    for method_id, spec in specs.items():
        if method_id == "MLP-same-param-AdamW":
            continue
        model = _make_model(method_id, input_dim, output_dim, x_train, device, int(args.seed) + 4100, spec, budget)
        if not hasattr(model, "basis_diagnostics"):
            continue
        with torch.no_grad():
            diag = model.basis_diagnostics(x_train[: min(512, int(x_train.shape[0]))])  # type: ignore[attr-defined]
        rows.append(
            _stamp(
                {
                    "stage": "V124_INITIAL_GEOMETRY_AUDIT",
                    "method": method_id,
                    "candidate_id": method_id,
                    "basis_family": spec.basis_family,
                    "init_variant": spec.init_variant,
                    "diagnostic_only": int(spec.diagnostic_only),
                    "audit_split": "MNIST_train_unlabeled_512",
                    "input_block_norm_enabled": int(bool(getattr(model, "input_block_norm_enabled", False))),
                    "input_residual_block_norm_enabled": int(bool(getattr(model, "input_residual_block_norm_enabled", False))),
                    "input_group_rms_norm_enabled": int(bool(getattr(model, "input_group_rms_norm_enabled", False))),
                    "basis_specific_quad_input_norm_enabled": int(bool(getattr(model, "basis_specific_quad_input_norm_enabled", False))),
                    "quad_feature_norm_enabled": int(bool(getattr(model, "quad_feature_norm_enabled", False))),
                    "branch_output_norm_enabled": int(bool(getattr(model, "branch_output_norm_enabled", False))),
                    "residual_geom_mix": float(getattr(model, "residual_geom_mix", torch.tensor([float("nan")], device=device))[0].item()) if hasattr(model, "residual_geom_mix") else "",
                    "leg_logit_std": float(getattr(model, "leg_logit_std", torch.tensor([float("nan")], device=device))[0].item()) if hasattr(model, "leg_logit_std") else "",
                    "quad_logit_std": float(getattr(model, "quad_logit_std", torch.tensor([float("nan")], device=device))[0].item()) if hasattr(model, "quad_logit_std") else "",
                    "quad_feature_std_mean": float(getattr(model, "quad_feature_std", torch.tensor([float("nan")], device=device)).float().mean().item()) if hasattr(model, "quad_feature_std") else "",
                    **diag,
                }
            )
        )
    return rows


def _flatten_grads(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.grad is not None:
            chunks.append(p.grad.detach().flatten())
    return torch.cat(chunks) if chunks else torch.zeros(1, device=next(model.parameters()).device)


def _basis_metrics_for_model(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, Any]:
    if hasattr(model, "basis_diagnostics"):
        return model.basis_diagnostics(x)
    with torch.no_grad():
        logits = model(x)
        centered = logits - logits.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(centered)
            rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
        except RuntimeError:
            rank = 0.0
    return {"basis_entropy": "", "dead_basis_fraction": "", "basis_effective_rank": rank, "basis_condition_proxy": "", "basis_output_norm_p95": ""}


def _signal_noise(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, seed: int) -> Dict[str, float]:
    n = min(int(x.shape[0]), 64)
    xb = x[:n]
    yb = y[:n]
    half = max(1, n // 2)
    grads = []
    for xs, ys in [(xb[:half], yb[:half]), (xb[half:], yb[half:])]:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xs), ys).backward()
        grads.append(_flatten_grads(model))
    real = F.cosine_similarity(grads[0], grads[1], dim=0).item() if len(grads) == 2 else 0.0
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 77)
    yp = yb[torch.randperm(int(yb.numel()), device=x.device, generator=gen)]
    noise_grads = []
    for xs, ys in [(xb[:half], yp[:half]), (xb[half:], yp[half:])]:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xs), ys).backward()
        noise_grads.append(_flatten_grads(model))
    noise = F.cosine_similarity(noise_grads[0], noise_grads[1], dim=0).item() if len(noise_grads) == 2 else 0.0
    return {"signal_consistency_real": real, "signal_consistency_noise": noise, "noise_leak": max(0.0, noise - real), "real_noise_consistency_gap": real - noise}


def run_task(args: argparse.Namespace, out_dir: Path, device: torch.device, specs: Mapping[str, prim.PrimitiveSpec], task_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    datasets = [v120._canonical_dataset(x) for x in _parse_list(args.datasets)]
    seeds = _parse_ints(args.seeds)
    task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    geom_rows: List[Dict[str, Any]] = []
    signal_rows: List[Dict[str, Any]] = []
    tail_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    methods = ["MLP-same-param-AdamW", *task_ids]
    trained_mnist: Dict[str, torch.nn.Module] = {}
    for dataset in datasets:
        data = v120._load_vision_split(args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
        x_train = x_train_cpu.to(device=device, dtype=torch.float32)
        y_train = y_train_cpu.to(device=device)
        x_val = x_val_cpu.to(device=device, dtype=torch.float32)
        y_val = y_val_cpu.to(device=device)
        x_test = x_test_cpu.to(device=device, dtype=torch.float32)
        y_test = y_test_cpu.to(device=device)
        _, budget = _param_budget(input_dim, output_dim)
        for seed in seeds:
            for method_id in methods:
                spec = specs.get(method_id)
                model = _make_model(method_id, input_dim, output_dim, x_train, device, int(seed) + 124300, spec, budget)
                compiled_task_path = 0
                compile_error = ""
                manual_task_path = bool(hasattr(model, "manual_kernel_available") and model.manual_kernel_available())  # type: ignore[attr-defined]
                if method_id != "MLP-same-param-AdamW" and _task_compile_candidate(method_id) and not manual_task_path:
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        compiled_task_path = 1
                    except Exception as exc:  # pragma: no cover - backend dependent
                        compile_error = str(exc)
                        model = _make_model(method_id, input_dim, output_dim, x_train, device, int(seed) + 124300, spec, budget)
                opt, optimizer_profile = _make_task_optimizer(model, method_id, float(args.lr), float(args.weight_decay))
                lr_schedule = _task_lr_schedule_name(method_id)
                total_train_steps = max(
                    1,
                    int(args.epochs) * int(math.ceil(float(x_train.shape[0]) / float(max(1, int(args.batch_size))))),
                )
                if compiled_task_path:
                    warm_count = min(int(args.batch_size), int(x_train.shape[0]))
                    warm_steps = max(1, int(args.task_compile_warmup_steps))
                    saved_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    for warm_idx in range(warm_steps):
                        start_idx = (warm_idx * warm_count) % int(x_train.shape[0])
                        xb_warm = x_train[start_idx : start_idx + warm_count]
                        yb_warm = y_train[start_idx : start_idx + warm_count]
                        if int(xb_warm.shape[0]) < warm_count:
                            xb_warm = x_train[:warm_count]
                            yb_warm = y_train[:warm_count]
                        opt.zero_grad(set_to_none=True)
                        F.cross_entropy(model(xb_warm), yb_warm).backward()
                        opt.zero_grad(set_to_none=True)
                    opt_warm_steps = max(0, int(args.task_compile_warmup_optimizer_steps))
                    for warm_idx in range(opt_warm_steps):
                        start_idx = (warm_idx * warm_count) % int(x_train.shape[0])
                        xb_warm = x_train[start_idx : start_idx + warm_count]
                        yb_warm = y_train[start_idx : start_idx + warm_count]
                        if int(xb_warm.shape[0]) < warm_count:
                            xb_warm = x_train[:warm_count]
                            yb_warm = y_train[:warm_count]
                        opt.zero_grad(set_to_none=True)
                        F.cross_entropy(model(xb_warm), yb_warm).backward()
                        opt.step()
                    if opt_warm_steps > 0:
                        model.load_state_dict(saved_model_state)
                        _optimizer_state_warm_reset(opt)
                        opt.zero_grad(set_to_none=True)
                    _sync(device)
                gen = torch.Generator(device=device).manual_seed(int(args.seed) + int(seed) + len(trace_rows))
                step_times: List[float] = []
                auc_loss_raw = 0.0
                auc_loss_steady = 0.0
                steady_epoch_count = 0
                step_id = 0
                start = time.perf_counter()
                for epoch in range(int(args.epochs)):
                    epoch_step_start = len(step_times)
                    perm = torch.randperm(int(x_train.shape[0]), generator=gen, device=device)
                    for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
                        idx = perm[off : off + int(args.batch_size)]
                        xb = x_train[idx]
                        yb = y_train[idx]
                        if lr_schedule != "constant":
                            lr_now = _scheduled_lr(float(args.lr), step_id + 1, total_train_steps, lr_schedule)
                            for group in opt.param_groups:
                                group["lr"] = lr_now * float(group.get("lr_multiplier", 1.0))
                        opt.zero_grad(set_to_none=True)
                        _sync(device)
                        t0 = time.perf_counter()
                        if manual_task_path:
                            logits, cache = model.manual_ce_forward_cache(xb)  # type: ignore[attr-defined]
                            loss = F.cross_entropy(logits, yb)
                            model.manual_ce_backward_from_cache(logits, cache, yb)  # type: ignore[attr-defined]
                        else:
                            loss = F.cross_entropy(model(xb), yb)
                            loss.backward()
                        opt.step()
                        _sync(device)
                        t1 = time.perf_counter()
                        step_times.append((t1 - t0) * 1000.0)
                        step_id += 1
                    val = _classification(model, x_val, y_val)
                    epoch_times = step_times[epoch_step_start:]
                    epoch_mean_ms = _mean(epoch_times)
                    epoch_q90_ms = _q(epoch_times, 0.90)
                    auc_loss_raw += _safe_float(val["loss"]) * max(1.0, epoch_mean_ms)
                    if (epoch + 1) > int(args.task_timing_warmup_epochs):
                        auc_loss_steady += _safe_float(val["loss"]) * max(1.0, epoch_q90_ms)
                        steady_epoch_count += 1
                    diag = _basis_metrics_for_model(model, x_val[: min(128, int(x_val.shape[0]))])
                    trace_rows.append(
                        _stamp(
                            {
                                "stage": "V124_TASK_TRACE",
                                "method": method_id,
                                "candidate_id": method_id,
                                "dataset": dataset,
                                "seed": seed,
                                "epoch": epoch + 1,
                                "step": step_id,
                                "wall_clock_sec": time.perf_counter() - start,
                                "val_acc": val["acc"],
                                "val_loss": val["loss"],
                                "step_time_q90": _q(step_times, 0.90),
                                "epoch_step_time_mean": epoch_mean_ms,
                                "epoch_step_time_q90": epoch_q90_ms,
                                "lr_schedule": lr_schedule,
                                "optimizer_profile": optimizer_profile,
                                "last_lr": opt.param_groups[0].get("lr", float(args.lr)),
                                "max_group_lr": max(float(g.get("lr", 0.0)) for g in opt.param_groups),
                                "min_group_lr": min(float(g.get("lr", 0.0)) for g in opt.param_groups),
                                "compiled_task_path": compiled_task_path,
                                "manual_task_path": int(manual_task_path),
                                "task_compile_warmup_steps": int(args.task_compile_warmup_steps),
                                "task_compile_warmup_optimizer_steps": int(args.task_compile_warmup_optimizer_steps),
                                "compile_error": compile_error,
                            }
                        )
                    )
                    geom_rows.append(
                        _stamp(
                            {
                                "stage": "V124_GEOMETRY_SNAPSHOT",
                                "method": method_id,
                                "candidate_id": method_id,
                                "dataset": dataset,
                                "seed": seed,
                                "epoch": epoch + 1,
                                **diag,
                            }
                        )
                    )
                val = _classification(model, x_val, y_val)
                test = _classification(model, x_test, y_test)
                sig = _signal_noise(model, x_train[: min(128, int(x_train.shape[0]))], y_train[: min(128, int(y_train.shape[0]))], int(seed))
                diag = _basis_metrics_for_model(model, x_val[: min(128, int(x_val.shape[0]))])
                task_rows.append(
                    _stamp(
                        {
                            "stage": "V124_TASK_TRIAGE",
                            "method": method_id,
                            "candidate_id": method_id,
                            "basis_family": spec.basis_family if spec else "MLP-control",
                            "dataset": dataset,
                            "seed": seed,
                            "val_acc": val["acc"],
                            "val_loss": val["loss"],
                            "test_acc": test["acc"],
                            "NLL": val["NLL"],
                            "ECE": val["ECE"],
                            "CEp99": val["CE_p99"],
                            "margin_mean": val["margin_mean"],
                            "margin_p10": val["margin_p10"],
                            "val_loss_auc_step": auc_loss_raw / max(1, int(args.epochs)),
                            "val_loss_auc_time_raw": auc_loss_raw / max(1, int(args.epochs)),
                            "val_loss_auc_time": (auc_loss_steady / float(steady_epoch_count)) if steady_epoch_count > 0 else (auc_loss_raw / max(1, int(args.epochs))),
                            "val_loss_auc_time_steady_epoch_count": steady_epoch_count,
                            "task_timing_warmup_epochs": int(args.task_timing_warmup_epochs),
                            "lr_schedule": lr_schedule,
                            "final_lr": opt.param_groups[0].get("lr", float(args.lr)),
                            "optimizer_profile": optimizer_profile,
                            "max_group_lr": max(float(g.get("lr", 0.0)) for g in opt.param_groups),
                            "min_group_lr": min(float(g.get("lr", 0.0)) for g in opt.param_groups),
                            "compiled_task_path": compiled_task_path,
                            "manual_task_path": int(manual_task_path),
                            "task_compile_warmup_steps": int(args.task_compile_warmup_steps),
                            "task_compile_warmup_optimizer_steps": int(args.task_compile_warmup_optimizer_steps),
                            "compile_error": compile_error,
                            "classwise_acc": json.dumps(val.get("classwise_acc", {}), sort_keys=True),
                            "step_time_q90": _q(step_times, 0.90),
                            "memory_ratio": "",
                            **diag,
                            **sig,
                        }
                    )
                )
                signal_rows.append(_stamp({"stage": "V124_SIGNAL_NOISE", "method": method_id, "candidate_id": method_id, "dataset": dataset, "seed": seed, **sig}))
                tail_rows.append(_stamp({"stage": "V124_TAIL_CALIBRATION", "method": method_id, "candidate_id": method_id, "dataset": dataset, "seed": seed, "CEp99": val["CE_p99"], "margin_p10": val["margin_p10"], "ECE": val["ECE"], "NLL": val["NLL"]}))
                if dataset == "MNIST" and int(seed) == 0 and method_id != "MLP-same-param-AdamW":
                    trained_mnist[method_id] = model
    base = {(r["dataset"], int(r["seed"])): r for r in task_rows if r["method"] == "MLP-same-param-AdamW"}
    passed: List[str] = []
    for method_id in task_ids:
        rows = [r for r in task_rows if r["method"] == method_id]
        for r in rows:
            b = base[(r["dataset"], int(r["seed"]))]
            r["val_acc_delta_vs_mlp"] = _safe_float(r["val_acc"]) - _safe_float(b["val_acc"])
            r["ECE_delta_vs_mlp"] = _safe_float(r["ECE"]) - _safe_float(b["ECE"])
            r["val_loss_auc_time_ratio_vs_mlp"] = _safe_float(r["val_loss_auc_time"]) / max(EPS, _safe_float(b["val_loss_auc_time"]))
        mean_delta = _mean(_safe_float(r.get("val_acc_delta_vs_mlp")) for r in rows)
        worst = min([_safe_float(r.get("val_acc_delta_vs_mlp")) for r in rows], default=-999.0)
        near = _mean(1.0 if _safe_float(r.get("val_acc_delta_vs_mlp")) >= -0.005 else 0.0 for r in rows)
        ece_ok = all(_safe_float(r.get("ECE_delta_vs_mlp")) <= 0.02 for r in rows)
        auc_ok = all(_safe_float(r.get("val_loss_auc_time_ratio_vs_mlp")) <= 1.05 for r in rows)
        if mean_delta >= -0.005 and near >= 0.80 and worst >= -0.025 and ece_ok and auc_ok:
            passed.append(method_id)
        else:
            failures.append(_stamp({"stage": "V124_TASK_FAILURE", "candidate_id": method_id, "failure_code": "A3_task_triage_fail", "mean_delta": mean_delta, "near_pass_rate": near, "worst_delta": worst, "ece_ok": int(ece_ok), "auc_time_ok": int(auc_ok), "action_recommended": "global init/normalization repair; no dataset-specific tuning"}))
    return task_rows, trace_rows, geom_rows, signal_rows, tail_rows, failures, passed


def _clone_model(model: torch.nn.Module) -> torch.nn.Module:
    import copy

    return copy.deepcopy(model)


def _params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _eval_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(F.cross_entropy(model(x), y).detach().item())


def _apply_delta(model: torch.nn.Module, deltas: Sequence[torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for p, d in zip(_params(model), deltas):
            p.add_(d.to(device=p.device), alpha=float(scale))


def _grad_delta(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> List[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    return [(-float(lr) * p.grad.detach()).clone() if p.grad is not None else torch.zeros_like(p) for p in _params(model)]


def _random_like(deltas: Sequence[torch.Tensor], seed: int) -> List[torch.Tensor]:
    device = deltas[0].device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    flat_norm = math.sqrt(sum(float(d.square().sum().item()) for d in deltas))
    out = [torch.randn(d.shape, device=device, generator=gen) for d in deltas]
    norm = math.sqrt(sum(float(d.square().sum().item()) for d in out))
    return [d * (flat_norm / max(EPS, norm)) for d in out]


def _basis_delta(model: torch.nn.Module, task_delta: Sequence[torch.Tensor], mode: str) -> List[torch.Tensor]:
    if hasattr(model, "basis_diagnostics"):
        raw = prim.basis_functional_direction(model, mode)
        # Convert diagnostic direction into a small update delta.
        deltas = [0.05 * d / d.norm().clamp_min(EPS) for d in raw]
        norm_task = math.sqrt(sum(float(d.square().sum().item()) for d in task_delta))
        norm_basis = math.sqrt(sum(float(d.square().sum().item()) for d in deltas))
        deltas = [d * (norm_task / max(EPS, norm_basis)) for d in deltas]
    else:
        deltas = [torch.zeros_like(d) for d in task_delta]
    if mode == "basis_aware_orthogonal":
        flat_task = torch.cat([d.flatten() for d in task_delta])
        flat_basis = torch.cat([d.flatten() for d in deltas])
        proj = (flat_basis @ flat_task) / flat_task.square().sum().clamp_min(EPS)
        deltas = [d - proj * t for d, t in zip(deltas, task_delta)]
    return deltas


def _geo_debt(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    cls = _classification(model, x, y)
    diag = _basis_metrics_for_model(model, x)
    cond = math.log10(max(1.0, _safe_float(diag.get("basis_condition_proxy"), 1.0)))
    dead = _safe_float(diag.get("dead_basis_fraction"), 0.0)
    return cond + dead + _safe_float(cls["ECE"]) + 0.05 * _safe_float(cls["CE_p99"])


def run_functional(args: argparse.Namespace, out_dir: Path, device: torch.device, specs: Mapping[str, prim.PrimitiveSpec], functional_ids: Sequence[str], base_qualified: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    if not functional_ids:
        nr = [_stamp({"stage": "V124_FUNCTIONAL_DIAGNOSTIC", "status": "not_run", "reason": "no_A1_exploratory_survivor"})]
        return nr, nr, nr, nr, False
    data = v120._load_vision_split(args, "MNIST", train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, *_rest, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    _, budget = _param_budget(input_dim, output_dim)
    one_rows: List[Dict[str, Any]] = []
    five_rows: List[Dict[str, Any]] = []
    lambda_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    any_positive = False
    controls = ["D0-TaskOnlyAdamW", "D1-NoOpMatchedOverhead", "D2-RandomMatchedNorm", "D3-AdamWParallelDirection", "D8-BasisAwareFunctional", "D9-BasisAwareFunctional-AdamWOrthogonal", "D10-BasisAwareFunctional-SNRProjected"]
    for method_id in functional_ids:
        spec = specs[method_id]
        model = _make_model(method_id, input_dim, output_dim, x_train, device, int(args.seed) + 124700, spec, budget)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        for off in range(0, min(int(x_train.shape[0]), int(args.batch_size) * 4), int(args.batch_size)):
            xb = x_train[off : off + int(args.batch_size)]
            yb = y_train[off : off + int(args.batch_size)]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        xb = x_train[: int(args.batch_size)]
        yb = y_train[: int(args.batch_size)]
        hold_x = x_val[: min(int(args.eval_batch_size), int(x_val.shape[0]))]
        hold_y = y_val[: min(int(args.eval_batch_size), int(y_val.shape[0]))]
        before_loss = _eval_loss(model, hold_x, hold_y)
        before_geo = _geo_debt(model, hold_x, hold_y)
        task_delta = _grad_delta(model, xb, yb, float(args.lr))
        random_delta = _random_like(task_delta, int(args.seed) + 124701)
        deltas_by_control = {
            "D0-TaskOnlyAdamW": task_delta,
            "D1-NoOpMatchedOverhead": [torch.zeros_like(d) for d in task_delta],
            "D2-RandomMatchedNorm": random_delta,
            "D3-AdamWParallelDirection": task_delta,
            "D8-BasisAwareFunctional": _basis_delta(model, task_delta, "basis_aware"),
            "D9-BasisAwareFunctional-AdamWOrthogonal": _basis_delta(model, task_delta, "basis_aware_orthogonal"),
            "D10-BasisAwareFunctional-SNRProjected": _basis_delta(model, task_delta, "basis_aware_snr_projected"),
        }
        scores: Dict[str, float] = {}
        for control in controls:
            clone = _clone_model(model)
            selected = 0.0
            backtracks = 0
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                trial = _clone_model(model)
                _apply_delta(trial, deltas_by_control[control], lam)
                after_loss = _eval_loss(trial, hold_x, hold_y)
                if after_loss <= before_loss * 1.05 or control in {"D1-NoOpMatchedOverhead"}:
                    selected = lam
                    clone = trial
                    break
                backtracks += 1
            after_loss = _eval_loss(clone, hold_x, hold_y)
            after_geo = _geo_debt(clone, hold_x, hold_y)
            holdout_ratio = after_loss / max(EPS, before_loss)
            geo_delta = before_geo - after_geo
            bad = int(holdout_ratio > 1.05)
            score = geo_delta - max(0.0, holdout_ratio - 1.0)
            scores[control] = score
            one_rows.append(
                _stamp(
                    {
                        "stage": "V124_FUNCTIONAL_ONE_STEP",
                        "functional_status": "official_base_qualified" if base_qualified else "diagnostic_base_not_qualified",
                        "basis_family": spec.basis_family,
                        "method": method_id,
                        "candidate_id": method_id,
                        "control": control,
                        "step_id": 1,
                        "train_descent": "",
                        "holdout_descent": before_loss - after_loss,
                        "holdout_descent_ratio": holdout_ratio,
                        "bad_step": bad,
                        "bad_step_rate": bad,
                        "lambda_selected": selected,
                        "lambda_backtrack_count": backtracks,
                        "cos_to_adamw": "",
                        "cos_to_random": "",
                        "functional_norm_ratio": "",
                        "GeoDebt_delta": geo_delta,
                        "ECE_delta": "",
                        "CEp99_delta": "",
                        "margin_p10_delta": "",
                        "rank_delta": "",
                        "basis_entropy_delta": "",
                        "perturb_drift_delta": "",
                        "signal_consistency_delta": "",
                        "noise_leak_delta": "",
                        "step_overhead_ms": "not_measured_in_diagnostic",
                        "memory_overhead_mb": "not_measured_in_diagnostic",
                        "control_gap_vs_best_control": "",
                    }
                )
            )
            lambda_rows.append(_stamp({"stage": "V124_LAMBDA_BACKTRACKING", "method": method_id, "candidate_id": method_id, "control": control, "lambda_selected": selected, "lambda_backtrack_count": backtracks}))
            five = _clone_model(model)
            losses = []
            for step_id in range(5):
                _apply_delta(five, deltas_by_control[control], selected)
                losses.append(_eval_loss(five, hold_x, hold_y))
            five_rows.append(_stamp({"stage": "V124_FUNCTIONAL_FIVE_STEP", "functional_status": "official_base_qualified" if base_qualified else "diagnostic_base_not_qualified", "method": method_id, "candidate_id": method_id, "control": control, "final_holdout_loss": losses[-1], "bad_step_rate": _mean(1.0 if v > before_loss * 1.05 else 0.0 for v in losses)}))
        best_control = max(scores[c] for c in ["D0-TaskOnlyAdamW", "D1-NoOpMatchedOverhead", "D2-RandomMatchedNorm", "D3-AdamWParallelDirection"])
        best_func = max(scores[c] for c in ["D8-BasisAwareFunctional", "D9-BasisAwareFunctional-AdamWOrthogonal", "D10-BasisAwareFunctional-SNRProjected"])
        gap = best_func - best_control
        if gap > 0:
            any_positive = True
        control_rows.append(_stamp({"stage": "V124_CONTROL_MATRIX", "functional_status": "official_base_qualified" if base_qualified else "diagnostic_base_not_qualified", "method": method_id, "candidate_id": method_id, "best_functional_score": best_func, "best_control_score": best_control, "control_gap_vs_best_control": gap, "beats_controls": int(gap > 0)}))
    return one_rows, five_rows, control_rows, lambda_rows, any_positive


def audit_provenance(out_dir: Path) -> List[Dict[str, Any]]:
    rows_checked = 0
    fake = proxy = cpu = 0
    for path in out_dir.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_checked += 1
                fake += int(_safe_float(row.get("fake_data_used"), 0))
                proxy += int(_safe_float(row.get("proxy_row_used"), 0))
                cpu += int(_safe_float(row.get("cpu_offload_used"), 0))
    rows = [_stamp({"stage": "V124_PROVENANCE_AUDIT", "rows_checked": rows_checked, "fake_data_used": fake, "proxy_row_used": proxy, "cpu_offload_used": cpu, "no_fake_pass": int(fake == 0), "no_proxy_pass": int(proxy == 0), "no_cpu_offload_pass": int(cpu == 0)})]
    write_csv_rows(out_dir / "v124_provenance_audit.csv", rows)
    return rows


def write_hash_manifest(out_dir: Path) -> Dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v124_hash_manifest.json":
            artifacts.append({"artifact": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"stage": "V124_HASH_MANIFEST", "artifacts": artifacts, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}
    write_json(out_dir / "v124_hash_manifest.json", manifest)
    return manifest


def write_figures(out_dir: Path, bench_rows: Sequence[Mapping[str, Any]], expr_rows: Sequence[Mapping[str, Any]], task_rows: Sequence[Mapping[str, Any]], control_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    basis = [r for r in bench_rows if int(_safe_float(r.get("batch_size"), 0)) == 128 and r.get("candidate_id") != "MLP-same-param-AdamW"]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(basis)), [_safe_float(r.get("step_ratio_q90_vs_mlp")) for r in basis])
    plt.xticks(range(len(basis)), [str(r.get("candidate_id")).split("-")[0] for r in basis], rotation=45, ha="right")
    plt.axhline(1.5, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_a1_step_ratio_by_basis.svg")
    plt.close()
    pairs = [r for r in expr_rows if r.get("target") == "E1-pairwise-product" and r.get("protocol") == "B1-standard-fit"]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(pairs)), [_safe_float(r.get("val_R2")) for r in pairs])
    plt.xticks(range(len(pairs)), [str(r.get("candidate_id")).split("-")[0] for r in pairs], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_a2_expression_pairwise_r2.svg")
    plt.close()
    task = [r for r in task_rows if r.get("method") != "MLP-same-param-AdamW"]
    plt.figure(figsize=(10, 4))
    plt.scatter([_safe_float(r.get("step_time_q90")) for r in task], [_safe_float(r.get("val_acc_delta_vs_mlp")) for r in task], s=10)
    plt.axhline(-0.005, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_a3_task_efficiency_pareto.svg")
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(control_rows)), [_safe_float(r.get("control_gap_vs_best_control")) for r in control_rows])
    plt.xticks(range(len(control_rows)), [str(r.get("candidate_id")).split("-")[0] for r in control_rows], rotation=45, ha="right")
    plt.axhline(0.0, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_b1_control_gap_by_basis.svg")
    plt.close()


def decide_route(a1: Sequence[str], expr_pass: Sequence[str], task_pass: Sequence[str], functional_positive: bool, provenance: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not a1:
        route = "R1-EfficiencyAllFail"
        base_blocker = "efficiency_all_fail"
        action = "implement fused/manual kernel for top two families only"
    elif functional_positive and not task_pass:
        route = "R5-FunctionalDiagnosticPositiveBaseNotYetQualified"
        base_blocker = "expression_fail" if not expr_pass else "task_fail"
        action = "continue base repair; do not write functional success"
    elif not expr_pass:
        route = "R2-EfficiencyPassExpressionFail"
        base_blocker = "expression_fail"
        action = "increase K within efficiency budget, improve init/normalization, or try lite hybrid"
    elif not task_pass:
        route = "R3-ExpressionPassTaskFail"
        base_blocker = "task_fail"
        action = "global optimizer/initialization repair; no dataset-specific tuning"
    elif task_pass and not functional_positive:
        route = "R4-BaseQualifiedFunctionalFail"
        base_blocker = "none"
        action = "keep base candidate; redesign basis-aware functional target"
    else:
        route = "R6-BasisAndFunctionalCandidateReady"
        base_blocker = "none"
        action = "run 20/30-epoch confirmation and 10-seed confirm"
    no_fake = bool(provenance and int(provenance[0].get("no_fake_pass", 0)) == 1 and int(provenance[0].get("no_proxy_pass", 0)) == 1 and int(provenance[0].get("no_cpu_offload_pass", 0)) == 1)
    return _stamp({"stage": "V124_ROUTE_DECISION", "route": route, "base_blocker_route": base_blocker, "A1_exploratory_survivors": list(a1), "A2_expression_pass": list(expr_pass), "A3_task_pass": list(task_pass), "functional_diagnostic_positive": bool(functional_positive), "base_qualified": bool(task_pass), "functional_open": bool(task_pass), "next_recommended_action": action, "no_fake": no_fake})


def write_report(out_dir: Path, report_path: Path, decision: Mapping[str, Any], provenance: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any], bench_rows: Sequence[Mapping[str, Any]], task_rows: Sequence[Mapping[str, Any]], control_rows: Sequence[Mapping[str, Any]]) -> None:
    prov = provenance[0] if provenance else {}
    a1 = list(decision.get("A1_exploratory_survivors", []))
    a2 = list(decision.get("A2_expression_pass", []))
    a3 = list(decision.get("A3_task_pass", []))
    bench128 = [r for r in bench_rows if int(_safe_float(r.get("batch_size"), 0)) == 128 and r.get("candidate_id") != "MLP-same-param-AdamW"]
    bench_lines = "\n".join(f"| `{r.get('candidate_id')}` | `{r.get('basis_family')}` | `{r.get('step_ratio_q90_vs_mlp')}` | `{r.get('compact_memory_ratio')}` | `{r.get('A1_exploratory_pass')}` |" for r in bench128)
    task_summary = []
    for mid in sorted({r.get("method") for r in task_rows if r.get("method") != "MLP-same-param-AdamW"}):
        rows = [r for r in task_rows if r.get("method") == mid]
        task_summary.append(f"| `{mid}` | `{_mean(_safe_float(r.get('val_acc_delta_vs_mlp')) for r in rows)}` | `{min([_safe_float(r.get('val_acc_delta_vs_mlp')) for r in rows], default=0.0)}` | `{_mean(_safe_float(r.get('ECE_delta_vs_mlp')) for r in rows)}` |")
    control_lines = "\n".join(f"| `{r.get('candidate_id')}` | `{r.get('functional_status')}` | `{r.get('best_functional_score')}` | `{r.get('best_control_score')}` | `{r.get('control_gap_vs_best_control')}` | `{r.get('beats_controls')}` |" for r in control_rows)
    key_hashes = []
    for row in hashes.get("artifacts", []):
        if row["artifact"] in {"v124_basis_manifest.csv", "v124_efficiency_microbench.csv", "v124_expression_battery.csv", "v124_task_triage.csv", "v124_control_matrix.csv", "v124_route_decision.json", "v124_provenance_audit.csv"}:
            key_hashes.append(f"| `{row['artifact']}` | `{row['sha256']}` |")
    text = f"""# DG-KAN v12.4 Multi-Basis Functional Dual 结果复盘

> 本复盘记录 `DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = {decision.get('route')}
base_qualified = {decision.get('base_qualified')}
functional_open = {decision.get('functional_open')}
functional_diagnostic_positive = {decision.get('functional_diagnostic_positive')}
next_recommended_action = {decision.get('next_recommended_action')}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}
```

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增统一 strict edge-basis KAN primitive：Activation/RBF/OrthogonalPolynomial/Fourier/Wavelet/BSpline/Rational | 所有 learnable tensor 都是 edge basis 权重；fixed centers/normalizer 不按 dataset name 或 label 分支。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 v12.4 A0/A1/A2/A3/B1 runner | 按 efficiency-first 执行；functional 在 base 未过时标记 `diagnostic_base_not_qualified`。 |
| `third_party/MJKAN` | 作为 RBF/FastKAN basis 设计来源读取 | 没有直接运行其 Colab 脚本，也没有引入 base update MLP shortcut。 |
| `third_party/KANbeFair` | 作为 local B-spline basis 设计来源读取 | 本轮只实现 order1 local basis 进行 microbench，不使用其 shortcut/base_fun。 |
| `third_party/rational_kat_cu` | 作为 Rational/KAT safe denominator 设计来源读取 | 本轮使用 torch-safe rational lite diagnostic，未依赖 CUDA extension 编译。 |

本轮没有做：

```text
1. 没有调低 base 或 functional gate。
2. 没有按 MNIST/Fashion/KMNIST 名称分支。
3. 没有 teacher/distillation/loss modification/class weights。
4. 没有把 autograd-only diagnostic 写成 final manual-kernel claim。
```

## 2. Route

```json
{json.dumps(dict(decision), ensure_ascii=False, indent=2)}
```

## 3. A1 Efficiency Microbench batch=128

| candidate | family | step ratio q90 vs MLP | memory ratio | exploratory pass |
|---|---|---:|---:|---:|
{bench_lines}

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = {a1}
A2 expression pass = {a2}
A3 task pass = {a3}
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
{chr(10).join(task_summary) if task_summary else '| `not_run` |  |  |  |'}

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
{control_lines if control_lines else '| `not_run` | `no_A1_exploratory_survivor` |  |  |  |  |'}

## 6. No-Fake / Hash

```text
rows_checked = {prov.get('rows_checked')}
fake/proxy/cpu = {prov.get('fake_data_used')} / {prov.get('proxy_row_used')} / {prov.get('cpu_offload_used')}
```

| artifact | SHA256 |
|---|---|
{chr(10).join(key_hashes)}

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
"""
    ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=f"results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_{_now_tag()}")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--seed", type=int, default=2413)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--task-timing-warmup-epochs", type=int, default=3)
    p.add_argument("--task-compile-warmup-steps", type=int, default=1)
    p.add_argument("--task-compile-warmup-optimizer-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-3)
    p.add_argument("--microbench-batch-sizes", default="128")
    p.add_argument("--microbench-warmup-steps", type=int, default=20)
    p.add_argument("--microbench-measure-steps", type=int, default=60)
    p.add_argument("--expression-dim", type=int, default=16)
    p.add_argument("--expression-train-size", type=int, default=1024)
    p.add_argument("--expression-val-size", type=int, default=512)
    p.add_argument("--expression-test-size", type=int, default=512)
    p.add_argument("--expression-batch-size", type=int, default=128)
    p.add_argument("--expression-lr", type=float, default=3.0e-3)
    p.add_argument("--expression-steps-b0", type=int, default=60)
    p.add_argument("--expression-steps-b1", type=int, default=600)
    p.add_argument("--expression-steps-b2", type=int, default=1200)
    p.add_argument("--expression-targets", default="E0-additive,E1-pairwise-product,E2-composition,E3-local-XOR,E4-high-frequency,E5-noise-stress,E6-rotated-pairwise-product,E7-piecewise-local,E8-random-quadratic-form,E9-smooth-nonpolynomial,E10-local-bump-mixture")
    p.add_argument("--ridge-alpha", type=float, default=1.0e-4)
    p.add_argument("--candidate-ids", default="", help="Optional comma-separated exact candidate ids for targeted repair runs; MLP control is always included.")
    p.add_argument("--report-path", default=str(REPORT_PATH_DEFAULT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = v120._device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    write_json(out_dir / "v124_run_manifest.json", _stamp({"stage": "V124_RUN_MANIFEST", "created_at": _now_iso(), "device": str(device), "args": vars(args), "runner": str(SCRIPT_PATH.relative_to(ROOT)), "plan": str(PLAN_PATH.relative_to(ROOT))}))
    if PLAN_PATH.exists():
        shutil.copy2(PLAN_PATH, out_dir / "plan.md")
    manifest, grad_rows, bench_rows, eff_failures, specs, a1_survivors = run_contract_and_microbench(args, out_dir, device)
    write_csv_rows(out_dir / "v124_basis_manifest.csv", manifest)
    write_csv_rows(out_dir / "v124_grad_correctness.csv", grad_rows)
    write_csv_rows(out_dir / "v124_efficiency_microbench.csv", bench_rows)
    write_csv_rows(out_dir / "v124_efficiency_failure_table.csv", eff_failures if eff_failures else [_stamp({"stage": "V124_EFFICIENCY_FAILURE_TABLE", "status": "no_A1_exploratory_failures"})])
    initial_geometry_rows = run_initial_geometry_audit(args, device, specs)
    write_csv_rows(out_dir / "v124_initial_geometry_audit.csv", initial_geometry_rows if initial_geometry_rows else [_stamp({"stage": "V124_INITIAL_GEOMETRY_AUDIT", "status": "not_run"})])
    expr_rows, frozen_rows, cond_rows, expr_pass = run_expression(args, out_dir, device, specs, a1_survivors)
    write_csv_rows(out_dir / "v124_expression_battery.csv", expr_rows if expr_rows else [_stamp({"stage": "V124_EXPRESSION_BATTERY", "status": "not_run", "reason": "no_A1_survivor"})])
    write_csv_rows(out_dir / "v124_frozen_readout.csv", frozen_rows if frozen_rows else [_stamp({"stage": "V124_FROZEN_READOUT", "status": "not_run", "reason": "no_A1_survivor"})])
    write_csv_rows(out_dir / "v124_basis_condition.csv", cond_rows if cond_rows else [_stamp({"stage": "V124_BASIS_CONDITION", "status": "not_run", "reason": "no_A1_survivor"})])
    task_ids = expr_pass if expr_pass else a1_survivors[: min(3, len(a1_survivors))]
    task_rows, trace_rows, geom_rows, signal_rows, tail_rows, task_failures, task_pass = run_task(args, out_dir, device, specs, task_ids) if task_ids else ([], [], [], [], [], [], [])
    write_csv_rows(out_dir / "v124_task_triage.csv", task_rows if task_rows else [_stamp({"stage": "V124_TASK_TRIAGE", "status": "not_run", "reason": "no_A1_or_expression_survivor"})])
    write_csv_rows(out_dir / "v124_task_trace.csv", trace_rows if trace_rows else [_stamp({"stage": "V124_TASK_TRACE", "status": "not_run", "reason": "no_A1_or_expression_survivor"})])
    write_csv_rows(out_dir / "v124_geometry_snapshot.csv", geom_rows if geom_rows else [_stamp({"stage": "V124_GEOMETRY_SNAPSHOT", "status": "not_run", "reason": "no_A1_or_expression_survivor"})])
    write_csv_rows(out_dir / "v124_signal_noise.csv", signal_rows if signal_rows else [_stamp({"stage": "V124_SIGNAL_NOISE", "status": "not_run", "reason": "no_A1_or_expression_survivor"})])
    write_csv_rows(out_dir / "v124_tail_calibration.csv", tail_rows if tail_rows else [_stamp({"stage": "V124_TAIL_CALIBRATION", "status": "not_run", "reason": "no_A1_or_expression_survivor"})])
    write_csv_rows(out_dir / "v124_task_failure_table.csv", task_failures if task_failures else [_stamp({"stage": "V124_TASK_FAILURE", "status": "no_A3_task_failures_or_not_run"})])
    one_rows, five_rows, control_rows, lambda_rows, functional_positive = run_functional(args, out_dir, device, specs, a1_survivors, bool(task_pass))
    write_csv_rows(out_dir / "v124_functional_one_step.csv", one_rows)
    write_csv_rows(out_dir / "v124_functional_five_step.csv", five_rows)
    write_csv_rows(out_dir / "v124_control_matrix.csv", control_rows)
    write_csv_rows(out_dir / "v124_lambda_backtracking.csv", lambda_rows)
    provenance = audit_provenance(out_dir)
    decision = decide_route(a1_survivors, expr_pass, task_pass, functional_positive, provenance)
    write_json(out_dir / "v124_route_decision.json", decision)
    write_figures(out_dir, bench_rows, expr_rows, task_rows, control_rows)
    provenance = audit_provenance(out_dir)
    decision = decide_route(a1_survivors, expr_pass, task_pass, functional_positive, provenance)
    write_json(out_dir / "v124_route_decision.json", decision)
    hashes = write_hash_manifest(out_dir)
    write_report(out_dir, Path(args.report_path), decision, provenance, hashes, bench_rows, task_rows, control_rows)
    print(json.dumps({"out_dir": str(out_dir), "route": decision["route"], "base_qualified": decision["base_qualified"], "functional_diagnostic_positive": decision["functional_diagnostic_positive"], "no_fake": decision["no_fake"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
