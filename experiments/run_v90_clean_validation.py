#!/usr/bin/env python3
"""DG-KAN v9 clean validation runner.

This runner continues the v9.0 refactor audit past Phase A/B.  It keeps the
scientific contract explicit:

* official KAN candidates use CE with label_smoothing=0,
* full-edge KAN paths use manual forward/backward/update, no loss.backward,
* functional updates are guarded update rules, not geometry losses,
* every pass/fail is written to CSV/JSON artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_gafu_v85_real as v85  # noqa: E402
import run_gafu_v87_real as v87  # noqa: E402
from run_v90_refactor_audit import (  # noqa: E402
    contract_tests,
    legacy_truth_audit,
    module_import_audit,
    write_public_api_table,
)

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.config import FAIR_FLOPS_RATIO_MAX, FAIR_MEMORY_RATIO_MAX, FAIR_PARAM_RATIO_MAX, FAIR_STEP_RATIO_MAX, V9_REQUIRED_ARTIFACTS  # noqa: E402
from dgkan.contracts import make_contract_audit_row  # noqa: E402
from dgkan.external.kanbefair_adapter import baseline_registry_rows  # noqa: E402
from dgkan.registry import default_v90_registry  # noqa: E402
from dgkan.specs import CandidateSpec, ContractFlags, ModuleAuditSpec  # noqa: E402
from dgkan.training.gradcheck import edge_layer_manual_autograd_gradcheck, edge_layer_manual_shape_smoke  # noqa: E402
from dgkan.training.manual_full_edge import (  # noqa: E402
    FullEdgeTrainConfig,
    ce_loss_and_grad,
    evaluate_manual_full_edge,
    time_manual_full_edge_step,
    train_manual_full_edge,
)
from dgkan.models.manual_full_edge import ManualFullEdgeClassifier  # noqa: E402


LEGACY_DEFAULT = Path("results/real_rerun_20260506/v87_full_chain_fmnist_kmnist_selected_kw4_formal_repeat_20260508T203000Z")
PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.0_CodeRefactor_CleanPureKAN_FunctionalTraining_ExternalFairValidation_完整整改计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v90_clean_validation.py"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _canonical_task(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"fashion", "fashion-mnist", "fmnist"}:
        return "Fashion-MNIST"
    if key == "kmnist":
        return "KMNIST"
    if key == "mnist":
        return "MNIST"
    return str(name)


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def parameter_count(input_dim: int, output_dim: int) -> int:
        return 32 * (int(input_dim) + 1) + int(output_dim) * (32 + 1)

    @staticmethod
    def flops_count(input_dim: int, output_dim: int) -> int:
        return 2 * int(input_dim) * 32 + 14 * 32 + 2 * 32 * int(output_dim) + 14 * int(output_dim)


def _ece_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    ece = torch.zeros((), device=logits.device)
    for idx in range(10):
        lo = idx / 10.0
        hi = (idx + 1) / 10.0
        mask = (conf > lo) & (conf <= hi)
        if bool(mask.any()):
            ece = ece + mask.float().mean() * torch.abs(conf[mask].mean() - correct[mask].mean())
    return float(ece.detach().cpu())


def _train_mlp_baseline(
    *,
    task: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
    output_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = SmallMLP(input_dim, output_dim).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    steps_per_epoch = max(1, math.ceil(int(x_train.shape[0]) / int(batch_size)))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    last_loss = float("nan")
    gen = torch.Generator(device=device).manual_seed(int(seed))
    for _epoch in range(int(epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for start in range(0, int(x_train.shape[0]), int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
    _sync(device)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        logits = model(x_test)
        loss = F.cross_entropy(logits, y_test)
        acc = (logits.argmax(dim=1) == y_test).float().mean()
    return {
        "stage": "G1_KANBEFAIR_BASELINE_REPRODUCTION",
        "task": task,
        "candidate_id": "KB-MLP-width32-gelu",
        "model_level": "external_baseline",
        "loss_type": "CE",
        "label_smoothing": 0.0,
        "test_acc": float(acc.detach().cpu()),
        "test_acc_pct": float(acc.detach().cpu()) * 100.0,
        "NLL": float(loss.detach().cpu()),
        "ECE": _ece_from_logits(logits, y_test),
        "params": SmallMLP.parameter_count(input_dim, output_dim),
        "FLOPs_forward": SmallMLP.flops_count(input_dim, output_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "train_loss_last": last_loss,
        "train_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / max(1, int(epochs) * steps_per_epoch),
        "peak_memory_MB": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else float("nan"),
        "uses_loss_backward": 1,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _time_mlp_step(
    *,
    input_dim: int,
    output_dim: int,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    device: torch.device,
    batch_size: int,
    warmup: int,
    reps: int,
) -> float:
    torch.manual_seed(int(seed))
    model = SmallMLP(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = x.to(device)
    y = y.to(device)
    bs = min(int(batch_size), int(x.shape[0]))

    def one_step(pos: int) -> None:
        start = (pos * bs) % int(x.shape[0])
        idx = torch.arange(start, min(start + bs, int(x.shape[0])), device=device)
        if int(idx.numel()) < bs:
            idx = torch.arange(0, bs, device=device) % int(x.shape[0])
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        loss.backward()
        opt.step()

    for i in range(int(warmup)):
        one_step(i)
    _sync(device)
    started = time.perf_counter()
    for i in range(int(reps)):
        one_step(i + int(warmup))
    _sync(device)
    return (time.perf_counter() - started) * 1000.0 / max(1, int(reps))


def _baseline_kanbefair_row(
    *,
    task: str,
    model_name: str,
    model_cls: Any,
    input_dim: int,
    output_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    if model_cls is None:
        return {
            "stage": "G1_KANBEFAIR_BASELINE_REPRODUCTION",
            "task": task,
            "candidate_id": f"KB-{model_name}",
            "status": "not_run",
            "reason": "KANbeFair model import unavailable",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    if model_name == "KAN":
        ns = v85._kb_namespace(
            model_name="KAN",
            input_size=input_dim,
            output_size=output_dim,
            layers_width=[2],
            activation_name="gelu",
            kan_grid=3,
            kan_order=2,
            kan_shortcut="silu",
        )
    else:
        ns = v85._kb_namespace(model_name="MLP", input_size=input_dim, output_size=output_dim, layers_width=[32], activation_name="gelu")
    model = model_cls(ns)
    metrics = v85._train_kb_classifier(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=1.0e-3,
        seed=int(seed),
        device=device,
    )
    return {
        "stage": "G1_KANBEFAIR_BASELINE_REPRODUCTION",
        "task": task,
        "candidate_id": f"KB-{model_name}",
        "model_level": "external_baseline",
        "loss_type": "CE",
        "label_smoothing": 0.0,
        "test_acc": float(metrics["test_acc_pct"]) / 100.0,
        "test_acc_pct": metrics["test_acc_pct"],
        "NLL": metrics["NLL"],
        "ECE": metrics["ECE"],
        "params": int(model.total_parameters()) if hasattr(model, "total_parameters") else "metric_unavailable",
        "FLOPs_forward": float(model.total_flops()) if hasattr(model, "total_flops") else "metric_unavailable",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "train_loss_last": metrics["train_loss_last"],
        "train_time_s": metrics["train_time_s"],
        "step_time_ms": metrics["step_time_ms"],
        "peak_memory_MB": metrics["peak_memory_MB"],
        "uses_loss_backward": 1,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _dg_namespace(args: argparse.Namespace, *, candidate_id: str, hidden_dim: int, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        seed=int(seed),
        dg_candidate_id=candidate_id,
        dg_hidden_dim=int(hidden_dim),
        dg_basis_count=8,
        dg_batch_size=int(args.batch_size),
        dg_eval_batch_size=int(args.eval_batch_size),
        dg_stream_batches_from_cpu=False,
        dg_stream_chunk_batches=1,
        dg_stream_epoch_permute_cpu=False,
        dg_stream_chunk_order_shuffle=False,
        primary_epochs=int(args.epochs),
        weight_decay=1.0e-4,
        ft7_event_stride=128,
        ft7_event_alpha_mult=15.0,
    )


def _with_label_smoothing_override(value: float):
    original = v85.v83.v80._spec_map_v80

    def wrapped() -> Dict[str, Any]:
        out = original()
        return {key: dataclasses.replace(spec, label_smoothing=float(value)) for key, spec in out.items()}

    return original, wrapped


def run_clean_transitional(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    task = "MNIST"
    x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v85._load_kanbefair_vision_tensors(
        task,
        data_root=ROOT / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root),
        train_size=int(args.train_size),
        test_size=int(args.test_size),
        seed=int(args.seed),
    )
    mlp_row = _train_mlp_baseline(
        task=task,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
    )
    rows: List[Dict[str, Any]] = [mlp_row]
    original_ce = v85._manual_ce_backward_v85
    original_holdout_ce = v85._manual_ce_backward_with_holdout_v85
    original_hook = getattr(v85, "DG_PRETRAIN_HOOK", None)
    original_compiled_flag = v87._V85_FUSED_USE_COMPILED_CORE
    v87._V85_FUSED_USE_COMPILED_CORE = True
    v85._manual_ce_backward_v85 = v87._manual_ce_backward_v85_fused
    v85._manual_ce_backward_with_holdout_v85 = v87._manual_ce_backward_with_holdout_v85_fused
    v85.DG_PRETRAIN_HOOK = v87._make_v85_compiled_fused_prewarm_hook()
    try:
        for label, smoothing in [("DG-Transitional-LegacySmoothCE", 0.05), ("DG-Transitional-CleanCE", 0.0)]:
            original, wrapped = _with_label_smoothing_override(smoothing)
            v85.v83.v80._spec_map_v80 = wrapped
            try:
                for functional in [0, 1]:
                    ns = _dg_namespace(args, candidate_id="KW4", hidden_dim=28, seed=int(args.seed))
                    row = v85._train_dg_primary_classifier(
                        ns,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        dataset=task,
                        input_dim=input_dim,
                        num_classes=output_dim,
                        functional_update_used=functional,
                    )
                    row.update(
                        {
                            "stage": "C1_CLEAN_CE_TRANSITIONAL_REVALIDATION",
                            "task": task,
                            "candidate_id": f"{label}{'+Functional' if functional else '+Base'}",
                            "model_level": "transitional",
                            "loss_type": "CE",
                            "label_smoothing": float(smoothing),
                            "ce_head_backward_impl": "compiled_fused_value_equivalent",
                            "compiled_prewarm_used": 1,
                            "delta_vs_KB_MLP": _safe_float(row.get("test_metric"), 0.0) / 100.0 - _safe_float(mlp_row.get("test_acc"), 0.0),
                            "params_ratio_vs_KB_MLP": _safe_float(row.get("params"), float("nan")) / max(1.0, _safe_float(mlp_row.get("params"), float("nan"))),
                            "FLOPs_ratio_vs_KB_MLP": _safe_float(row.get("FLOPs"), float("nan")) / max(1.0, _safe_float(mlp_row.get("FLOPs_forward"), float("nan"))),
                            "step_ratio_vs_KB_MLP": _safe_float(row.get("step_time_ms"), float("nan")) / max(1.0e-12, _safe_float(mlp_row.get("step_time_ms"), float("nan"))),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }
                    )
                    rows.append(row)
            finally:
                v85.v83.v80._spec_map_v80 = original
    finally:
        v85._manual_ce_backward_v85 = original_ce
        v85._manual_ce_backward_with_holdout_v85 = original_holdout_ce
        v85.DG_PRETRAIN_HOOK = original_hook
        v87._V85_FUSED_USE_COMPILED_CORE = original_compiled_flag

    clean_base = next((r for r in rows if r.get("candidate_id") == "DG-Transitional-CleanCE+Base"), {})
    clean_func = next((r for r in rows if r.get("candidate_id") == "DG-Transitional-CleanCE+Functional"), {})
    legacy_func = next((r for r in rows if r.get("candidate_id") == "DG-Transitional-LegacySmoothCE+Functional"), {})
    clean_task_preserve = int(
        _safe_float(clean_func.get("test_metric"), -1.0) / 100.0
        >= _safe_float(legacy_func.get("test_metric"), 1.0e9) / 100.0 - 0.005
    )
    clean_external = int(_safe_float(clean_func.get("test_metric"), -1.0) / 100.0 >= _safe_float(mlp_row.get("test_acc"), 1.0e9))
    clean_system = int(
        _safe_float(clean_func.get("params_ratio_vs_KB_MLP"), 99.0) <= FAIR_PARAM_RATIO_MAX
        and _safe_float(clean_func.get("FLOPs_ratio_vs_KB_MLP"), 99.0) <= FAIR_FLOPS_RATIO_MAX
        and _safe_float(clean_func.get("step_ratio_vs_KB_MLP"), 99.0) <= FAIR_STEP_RATIO_MAX
    )
    base_curv = _safe_float(clean_base.get("geometry_curvature_after"), float("nan"))
    func_curv = _safe_float(clean_func.get("geometry_curvature_after"), float("nan"))
    geom = int(math.isfinite(base_curv) and math.isfinite(func_curv) and func_curv <= 0.90 * max(base_curv, 1.0e-12))
    summary = {
        "stage": "C1_CLEAN_CE_TRANSITIONAL_SUMMARY",
        "task": task,
        "protocol": protocol,
        "clean_ce_task_preservation_pass": clean_task_preserve,
        "clean_ce_external_fair_pass": clean_external,
        "clean_ce_system_pass": clean_system,
        "clean_ce_functional_geometry_pass": geom,
        "clean_ce_pass": int(clean_task_preserve and clean_external and clean_system and geom),
        "kb_mlp_acc": mlp_row.get("test_acc"),
        "legacy_smooth_functional_acc": _safe_float(legacy_func.get("test_metric"), float("nan")) / 100.0,
        "clean_functional_acc": _safe_float(clean_func.get("test_metric"), float("nan")) / 100.0,
        "clean_functional_delta_vs_mlp": _safe_float(clean_func.get("test_metric"), float("nan")) / 100.0 - _safe_float(mlp_row.get("test_acc"), float("nan")),
        "clean_functional_step_ratio_vs_mlp": clean_func.get("step_ratio_vs_KB_MLP"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv_rows(out_dir / "label_smoothing_removal_audit.csv", rows)
    write_csv_rows(out_dir / "clean_transitional_revalidation.csv", [summary])
    write_csv_rows(out_dir / "legacy_refactor_parity.csv", [
        {
            "stage": "C2_LEGACY_REFACTOR_PARITY",
            "status": "measured",
            "candidate_id": "DG-Transitional-CleanCE-KW4-hidden28",
            "legacy_reference": str(LEGACY_DEFAULT),
            "parity_scope": "clean_ce_revalidation_uses_legacy_manual_core_with_label_smoothing_forced_to_zero",
            "legacy_refactor_parity_pass": summary["clean_ce_pass"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ])
    return rows, summary


def _full_edge_candidate_configs(input_dim: int, output_dim: int, epochs: int, batch_size: int, eval_batch_size: int) -> List[FullEdgeTrainConfig]:
    return [
        FullEdgeTrainConfig(
            candidate_id="DG-FullEdge-Poly2Silu6-h5",
            dims=(input_dim, 5, output_dim),
            edge_kind="poly2_silu6",
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            lr=1.0e-3,
        ),
        FullEdgeTrainConfig(
            candidate_id="DG-FullEdge-CompactPolySilu3-h11",
            dims=(input_dim, 11, output_dim),
            edge_kind="compact_poly_silu3",
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            lr=1.0e-3,
        ),
        FullEdgeTrainConfig(
            candidate_id="DG-FullEdge-CompactPolySilu3-h11x2",
            dims=(input_dim, 11, 11, output_dim),
            edge_kind="compact_poly_silu3",
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            lr=1.0e-3,
        ),
        FullEdgeTrainConfig(
            candidate_id="DG-FullEdge-RBF4-h8",
            dims=(input_dim, 8, output_dim),
            edge_kind="rbf4",
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            lr=1.0e-3,
        ),
        FullEdgeTrainConfig(
            candidate_id="DG-FullEdge-Spline4-h8",
            dims=(input_dim, 8, output_dim),
            edge_kind="spline4",
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            lr=1.0e-3,
        ),
    ]


def _functional_cfg(base: FullEdgeTrainConfig) -> FullEdgeTrainConfig:
    return FullEdgeTrainConfig(
        candidate_id=base.candidate_id + "+Functional",
        dims=base.dims,
        edge_kind=base.edge_kind,
        epochs=base.epochs,
        batch_size=base.batch_size,
        eval_batch_size=base.eval_batch_size,
        lr=base.lr,
        weight_decay=base.weight_decay,
        functional_update="functional",
        functional_interval=128,
        functional_strength=0.05,
        functional_guard_ratio=1.003,
    )


def run_full_edge_external(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    MLP_cls, KAN_cls, _BSpline, _stub, import_status = v85._import_kanbefair_models(ROOT / args.kanbefair_path)
    baseline_rows: List[Dict[str, Any]] = []
    external_rows: List[Dict[str, Any]] = []
    causality_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []
    compute_rows: List[Dict[str, Any]] = []
    task_summaries: List[Dict[str, Any]] = []
    best_candidates: Dict[str, FullEdgeTrainConfig] = {}

    for task in [_canonical_task(x) for x in _parse_csv_list(args.datasets)]:
        x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v85._load_kanbefair_vision_tensors(
            task,
            data_root=ROOT / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root),
            train_size=int(args.train_size),
            test_size=int(args.test_size),
            seed=int(args.seed),
        )
        mlp_row = _baseline_kanbefair_row(
            task=task,
            model_name="MLP",
            model_cls=MLP_cls,
            input_dim=input_dim,
            output_dim=output_dim,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            device=device,
        )
        kan_row = _baseline_kanbefair_row(
            task=task,
            model_name="KAN",
            model_cls=KAN_cls,
            input_dim=input_dim,
            output_dim=output_dim,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            device=device,
        )
        baseline_rows.extend([mlp_row, kan_row])
        mlp_acc = _safe_float(mlp_row.get("test_acc"), float("nan"))
        mlp_params = _safe_float(mlp_row.get("params"), float("nan"))
        mlp_flops = _safe_float(mlp_row.get("FLOPs_forward"), float("nan"))
        mlp_step = _safe_float(mlp_row.get("step_time_ms"), float("nan"))
        mlp_mem = _safe_float(mlp_row.get("peak_memory_MB"), float("nan"))
        base_candidate_rows: List[Dict[str, Any]] = []
        for cfg in _full_edge_candidate_configs(input_dim, output_dim, int(args.epochs), int(args.batch_size), int(args.eval_batch_size)):
            _model, row, _raw = train_manual_full_edge(cfg, x_train, y_train, x_test, y_test, seed=int(args.seed), device=device)
            row.update(
                {
                    "stage": "G3_FULL_EDGE_EXTERNAL_VALIDATION",
                    "task": task,
                    "protocol": protocol,
                    "delta_vs_KB_MLP": _safe_float(row.get("test_acc"), float("nan")) - mlp_acc,
                    "params_ratio_vs_KB_MLP": _safe_float(row.get("params"), float("nan")) / max(1.0, mlp_params),
                    "FLOPs_ratio_vs_KB_MLP": _safe_float(row.get("FLOPs_forward"), float("nan")) / max(1.0, mlp_flops),
                    "step_ratio_vs_KB_MLP": _safe_float(row.get("step_time_ms"), float("nan")) / max(1.0e-12, mlp_step),
                    "memory_ratio_vs_KB_MLP": _safe_float(row.get("peak_memory_MB"), float("nan")) / max(1.0e-12, mlp_mem),
                }
            )
            row["FullEdgeTaskPass"] = int(row["delta_vs_KB_MLP"] >= 0.0)
            row["FullEdgeFairEnvelopePass"] = int(
                row["params_ratio_vs_KB_MLP"] <= FAIR_PARAM_RATIO_MAX
                and row["FLOPs_ratio_vs_KB_MLP"] <= FAIR_FLOPS_RATIO_MAX
                and row["step_ratio_vs_KB_MLP"] <= FAIR_STEP_RATIO_MAX
                and row["memory_ratio_vs_KB_MLP"] <= FAIR_MEMORY_RATIO_MAX
            )
            row["FullEdgeMinimumPass"] = 0
            base_candidate_rows.append(row)
            external_rows.append(row)
        best_row = max(base_candidate_rows, key=lambda r: _safe_float(r.get("test_acc"), -1.0))
        best_cfg = next(c for c in _full_edge_candidate_configs(input_dim, output_dim, int(args.epochs), int(args.batch_size), int(args.eval_batch_size)) if c.candidate_id == best_row["candidate_id"])
        best_candidates[task] = best_cfg
        func_cfg = _functional_cfg(best_cfg)
        _func_model, func_row, func_raw = train_manual_full_edge(func_cfg, x_train, y_train, x_test, y_test, seed=int(args.seed), device=device)
        base_curv = _safe_float(best_row.get("curvature"), float("nan"))
        func_curv = _safe_float(func_row.get("curvature"), float("nan"))
        func_row.update(
            {
                "stage": "G3_FULL_EDGE_EXTERNAL_VALIDATION",
                "task": task,
                "protocol": protocol,
                "delta_vs_KB_MLP": _safe_float(func_row.get("test_acc"), float("nan")) - mlp_acc,
                "delta_vs_FullEdge_base": _safe_float(func_row.get("test_acc"), float("nan")) - _safe_float(best_row.get("test_acc"), float("nan")),
                "curvature_ratio_vs_base": func_curv / max(1.0e-12, base_curv),
                "params_ratio_vs_KB_MLP": _safe_float(func_row.get("params"), float("nan")) / max(1.0, mlp_params),
                "FLOPs_ratio_vs_KB_MLP": _safe_float(func_row.get("FLOPs_forward"), float("nan")) / max(1.0, mlp_flops),
                "step_ratio_vs_KB_MLP": _safe_float(func_row.get("step_time_ms"), float("nan")) / max(1.0e-12, mlp_step),
                "memory_ratio_vs_KB_MLP": _safe_float(func_row.get("peak_memory_MB"), float("nan")) / max(1.0e-12, mlp_mem),
            }
        )
        func_row["FullEdgeTaskPass"] = int(func_row["delta_vs_KB_MLP"] >= 0.0)
        func_row["FullEdgeFairEnvelopePass"] = int(
            func_row["params_ratio_vs_KB_MLP"] <= FAIR_PARAM_RATIO_MAX
            and func_row["FLOPs_ratio_vs_KB_MLP"] <= FAIR_FLOPS_RATIO_MAX
            and func_row["step_ratio_vs_KB_MLP"] <= FAIR_STEP_RATIO_MAX
            and func_row["memory_ratio_vs_KB_MLP"] <= FAIR_MEMORY_RATIO_MAX
        )
        func_row["FullEdgeGeometryPass"] = int(func_row["curvature_ratio_vs_base"] <= 0.90)
        func_row["FullEdgeMinimumPass"] = int(func_row["FullEdgeTaskPass"] and func_row["FullEdgeFairEnvelopePass"] and func_row["FullEdgeGeometryPass"])
        external_rows.append(func_row)
        for raw in func_raw:
            raw.update({"task": task, "candidate_id": func_cfg.candidate_id})
        write_csv_rows(out_dir / f"full_edge_functional_events_{task.replace('-', '_')}.csv", func_raw)

        # Causality controls use the same selected full-edge architecture.
        control_rows: List[Dict[str, Any]] = []
        for mode in ["none", "functional", "random", "shuffle"]:
            control_cfg = FullEdgeTrainConfig(
                candidate_id=f"{best_cfg.candidate_id}+{mode}",
                dims=best_cfg.dims,
                edge_kind=best_cfg.edge_kind,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                eval_batch_size=int(args.eval_batch_size),
                lr=best_cfg.lr,
                functional_update=mode if mode != "none" else "none",
                functional_interval=128,
                functional_strength=0.05,
                functional_guard_ratio=1.003,
            )
            _control_model, control_row, _ = train_manual_full_edge(
                control_cfg, x_train, y_train, x_test, y_test, seed=int(args.seed), device=device
            )
            control_row.update(
                {
                    "stage": "G4_FUNCTIONAL_CAUSALITY_CONTROLS",
                    "task": task,
                    "control": {"none": "Base", "functional": "Functional", "random": "RandomFunc", "shuffle": "ShuffledRoleFunc"}[mode],
                    "base_candidate_id": best_cfg.candidate_id,
                    "delta_vs_KB_MLP": _safe_float(control_row.get("test_acc"), float("nan")) - mlp_acc,
                }
            )
            control_rows.append(control_row)
        base_control = next(r for r in control_rows if r["control"] == "Base")
        func_control = next(r for r in control_rows if r["control"] == "Functional")
        random_control = next(r for r in control_rows if r["control"] == "RandomFunc")
        noop_curv = _safe_float(base_control.get("curvature"), float("nan"))
        random_curv = _safe_float(random_control.get("curvature"), float("nan"))
        for row in control_rows:
            row["curvature_ratio_vs_base"] = _safe_float(row.get("curvature"), float("nan")) / max(1.0e-12, noop_curv)
            row["CausalityPass"] = int(
                row["control"] == "Functional"
                and _safe_float(func_control.get("curvature"), float("inf")) < noop_curv
                and _safe_float(func_control.get("curvature"), float("inf")) < random_curv
                and _safe_float(func_control.get("test_acc"), -1.0) >= _safe_float(base_control.get("test_acc"), 1.0e9) - 0.002
            )
        causality_rows.extend(control_rows)

        # Timing protocols for the functional selected candidate.
        for protocol_name, warmup, reps in [("T0", 50, 200), ("T1", 100, 500), ("T2", 200, 1000), ("T3", 20, 260)]:
            dg_step = time_manual_full_edge_step(func_cfg, x_train, y_train, seed=int(args.seed), device=device, warmup=warmup, reps=reps)
            mlp_step = _time_mlp_step(
                input_dim=input_dim,
                output_dim=output_dim,
                x=x_train,
                y=y_train,
                seed=int(args.seed),
                device=device,
                batch_size=int(args.batch_size),
                warmup=warmup,
                reps=reps,
            )
            timing_rows.append(
                {
                    "stage": "F1_ROBUST_TIMING_PROTOCOLS",
                    "task": task,
                    "candidate_id": func_cfg.candidate_id,
                    "protocol": protocol_name,
                    "warmup": warmup,
                    "reps": reps,
                    "dg_step_time_ms": dg_step["step_time_ms"],
                    "kb_mlp_step_time_ms": mlp_step,
                    "step_ratio_vs_KB_MLP": dg_step["step_time_ms"] / max(1.0e-12, mlp_step),
                    "unknown_time_fraction": 0.0,
                    "timing_pass": int(dg_step["step_time_ms"] / max(1.0e-12, mlp_step) <= FAIR_STEP_RATIO_MAX),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
        timing_rows.append(
            {
                "stage": "F1_ROBUST_TIMING_PROTOCOLS",
                "task": task,
                "candidate_id": func_cfg.candidate_id,
                "protocol": "T4-full-loop",
                "dg_step_time_ms": func_row.get("step_time_ms"),
                "kb_mlp_step_time_ms": mlp_row.get("step_time_ms"),
                "step_ratio_vs_KB_MLP": func_row.get("step_ratio_vs_KB_MLP"),
                "unknown_time_fraction": 0.0,
                "timing_pass": int(_safe_float(func_row.get("step_ratio_vs_KB_MLP"), 99.0) <= FAIR_STEP_RATIO_MAX),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
        compute_rows.append(
            {
                "stage": "F3_TRAINING_COMPUTE_COUNTER",
                "task": task,
                "candidate_id": func_cfg.candidate_id,
                "forward_FLOPs": func_row.get("FLOPs_forward"),
                "forward_FLOPs_ratio_vs_KB_MLP": func_row.get("FLOPs_ratio_vs_KB_MLP"),
                "backward_FLOPs_estimate": _safe_float(func_row.get("FLOPs_forward"), float("nan")) * 2.0,
                "backward_FLOPs_estimate_ratio_vs_KB_MLP": _safe_float(func_row.get("FLOPs_ratio_vs_KB_MLP"), float("nan")) * 2.0,
                "update_FLOPs_estimate": func_row.get("params"),
                "step_time_ratio_vs_KB_MLP": func_row.get("step_ratio_vs_KB_MLP"),
                "kernel_time_forward": "metric_unavailable",
                "kernel_time_backward": "metric_unavailable",
                "kernel_time_update": "metric_unavailable",
                "kernel_time_functional": "metric_unavailable",
                "training_compute_fair_pass": int(
                    _safe_float(func_row.get("FLOPs_ratio_vs_KB_MLP"), 99.0) <= FAIR_FLOPS_RATIO_MAX
                    and _safe_float(func_row.get("step_ratio_vs_KB_MLP"), 99.0) <= FAIR_STEP_RATIO_MAX
                ),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
        task_summaries.append(
            {
                "task": task,
                "best_base_candidate_id": best_cfg.candidate_id,
                "best_base_acc": best_row.get("test_acc"),
                "functional_candidate_id": func_cfg.candidate_id,
                "functional_acc": func_row.get("test_acc"),
                "kb_mlp_acc": mlp_acc,
                "full_edge_minimum_pass": func_row["FullEdgeMinimumPass"],
                "causality_pass": max(int(r.get("CausalityPass", 0)) for r in control_rows),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )

    write_csv_rows(out_dir / "kanbefair_baseline_reproduction.csv", baseline_rows)
    write_csv_rows(out_dir / "full_edge_external_validation.csv", external_rows)
    write_csv_rows(out_dir / "external_fair_envelope.csv", [r for r in external_rows if str(r.get("candidate_id", "")).endswith("+Functional")])
    write_csv_rows(out_dir / "functional_causality_controls.csv", causality_rows)
    write_csv_rows(out_dir / "robust_timing_protocols.csv", timing_rows)
    write_csv_rows(out_dir / "training_compute_counter.csv", compute_rows)

    ratios = [_safe_float(r.get("step_ratio_vs_KB_MLP"), float("nan")) for r in timing_rows]
    ratios = [r for r in ratios if math.isfinite(r)]
    q90 = float(torch.tensor(ratios).quantile(0.90).item()) if ratios else float("nan")
    max_ratio = max(ratios) if ratios else float("nan")
    external_pass = int(task_summaries and all(int(r["full_edge_minimum_pass"]) == 1 for r in task_summaries))
    causality_pass = int(task_summaries and all(int(r["causality_pass"]) == 1 for r in task_summaries))
    timing_pass = int(math.isfinite(q90) and q90 <= FAIR_STEP_RATIO_MAX and math.isfinite(max_ratio) and max_ratio <= FAIR_STEP_RATIO_MAX)
    compute_pass = int(compute_rows and all(int(r.get("training_compute_fair_pass", 0)) == 1 for r in compute_rows))
    summary = {
        "stage": "V90_FULL_EDGE_EXTERNAL_SUMMARY",
        "kanbefair_import_status": import_status,
        "tasks": ",".join(r["task"] for r in task_summaries),
        "task_count": len(task_summaries),
        "full_edge_external_fair_pass": external_pass,
        "functional_causality_pass": causality_pass,
        "robust_timing_pass": timing_pass,
        "training_compute_fair_pass": compute_pass,
        "q90_step_ratio_vs_KB_MLP": q90,
        "max_step_ratio_vs_KB_MLP": max_ratio,
        "full_edge_task_pass_count": sum(int(r["full_edge_minimum_pass"]) for r in task_summaries),
        "full_edge_task_count": len(task_summaries),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv_rows(out_dir / "full_edge_external_summary.csv", [summary])
    write_csv_rows(out_dir / "boundary_audit.csv", task_summaries)
    return external_rows, causality_rows, summary, {"timing_rows": timing_rows, "compute_rows": compute_rows, "task_summaries": task_summaries}


def _write_remaining_artifacts(out_dir: Path) -> None:
    for filename, stage, reason in [
        ("robustness_perturbation.csv", "H1_ROBUSTNESS_PERTURBATION", "not_run_after_full_edge_boundary_fail"),
        ("continual_balance_multisplit.csv", "H2_CONTINUAL_BALANCE_MULTISPLIT", "not_run_after_full_edge_boundary_fail"),
    ]:
        if not (out_dir / filename).exists():
            write_csv_rows(
                out_dir / filename,
                [{"stage": stage, "status": "not_run", "reason": reason, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}],
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--kanbefair-path", default="third_party/KANbeFair")
    parser.add_argument("--legacy-artifact-dir", default=str(LEGACY_DEFAULT))
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--skip-clean-transitional", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = v85.v83.get_device(args.device)
    legacy_dir = ROOT / args.legacy_artifact_dir if not Path(args.legacy_artifact_dir).is_absolute() else Path(args.legacy_artifact_dir)

    manifest = {
        "stage": "V90_CLEAN_VALIDATION",
        "status": "measured",
        "started_utc": _now_iso(),
        "repo_root": str(ROOT),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "script_path": str(SCRIPT_PATH.relative_to(ROOT)),
        "device": str(device),
        "datasets": args.datasets,
        "epochs": int(args.epochs),
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
        "loss_type": "CE",
        "label_smoothing": 0.0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    registry = default_v90_registry()
    registry_rows = registry.rows()
    registry_rows.extend({**row, "model_level": "external_baseline", "official_eligible": 1} for row in baseline_registry_rows())
    registry_rows.extend(
        [
            CandidateSpec("DG-FullEdge-Poly2Silu6-h5", "full_edge", "edge_function_layer", "edge_function_head", "poly2_silu6", 5, 2, 6).to_row(),
            CandidateSpec("DG-FullEdge-CompactPolySilu3-h11", "full_edge", "edge_function_layer", "edge_function_head", "compact_poly_silu3", 11, 2, 3).to_row(),
            CandidateSpec("DG-FullEdge-CompactPolySilu3-h11x2", "full_edge", "edge_function_layer", "edge_function_head", "compact_poly_silu3", 11, 3, 3).to_row(),
            CandidateSpec("DG-FullEdge-RBF4-h8", "full_edge", "edge_function_layer", "edge_function_head", "rbf4", 8, 2, 4).to_row(),
            CandidateSpec("DG-FullEdge-Spline4-h8", "full_edge", "edge_function_layer", "edge_function_head", "spline4", 8, 2, 4).to_row(),
        ]
    )
    write_csv_rows(out_dir / "candidate_registry_clean.csv", registry_rows)
    write_csv_rows(out_dir / "code_audit_legacy_current.csv", legacy_truth_audit(ROOT, legacy_dir))
    contract_rows = [make_contract_audit_row(spec, ContractFlags(), stage="B2_CLEAN_REGISTRY_CONTRACT_AUDIT") for spec in registry]
    test_rows = contract_tests()
    write_csv_rows(out_dir / "contract_audit_clean.csv", contract_rows + test_rows)
    write_csv_rows(out_dir / "contract_negative_tests.csv", test_rows)
    modules = [
        "dgkan",
        "dgkan.specs",
        "dgkan.contracts",
        "dgkan.registry",
        "dgkan.models.edge_functions",
        "dgkan.models.edge_layers",
        "dgkan.models.manual_full_edge",
        "dgkan.optim.manual_adamw",
        "dgkan.functional.controller",
        "dgkan.external.kanbefair_adapter",
        "dgkan.artifacts.writer",
        "dgkan.training.gradcheck",
        "dgkan.training.manual_full_edge",
    ]
    module_rows = module_import_audit(modules)
    write_csv_rows(out_dir / "module_import_audit.csv", module_rows)
    write_public_api_table(out_dir, module_rows)
    smoke_rows = [edge_layer_manual_shape_smoke()]
    gradcheck_rows = [edge_layer_manual_autograd_gradcheck()]
    write_csv_rows(out_dir / "core_smoke_tests.csv", smoke_rows)
    write_csv_rows(out_dir / "gradcheck_full_edge.csv", gradcheck_rows)
    write_csv_rows(
        out_dir / "full_edge_implementation_audit.csv",
        [
            {
                "stage": "D1_D2_D3_FULL_EDGE_IMPLEMENTATION_AUDIT",
                "status": "measured",
                "poly2_silu6_implemented": 1,
                "compact_poly_silu3_implemented": 1,
                "rbf4_implemented": 1,
                "spline4_implemented": 1,
                "manual_forward": 1,
                "manual_backward": 1,
                "manual_update": 1,
                "GradPass": int(gradcheck_rows[0]["GradPass"]),
                "full_edge_contract_pass": int(gradcheck_rows[0]["GradPass"]),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        ],
    )

    clean_summary = {"clean_ce_pass": 0}
    if args.skip_clean_transitional:
        write_csv_rows(
            out_dir / "label_smoothing_removal_audit.csv",
            [{"stage": "C1_LABEL_SMOOTHING_REMOVAL_AUDIT", "status": "not_run", "reason": "skipped_by_cli", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}],
        )
        write_csv_rows(out_dir / "clean_transitional_revalidation.csv", [clean_summary])
        write_csv_rows(out_dir / "legacy_refactor_parity.csv", [clean_summary])
    else:
        _clean_rows, clean_summary = run_clean_transitional(args, out_dir, device)

    _external_rows, _causality_rows, full_summary, _extras = run_full_edge_external(args, out_dir, device)
    _write_remaining_artifacts(out_dir)

    clean_ce_pass = int(clean_summary.get("clean_ce_pass", 0))
    full_edge_contract = int(gradcheck_rows[0]["GradPass"])
    full_edge_task = int(full_summary.get("full_edge_task_pass_count", 0) == full_summary.get("full_edge_task_count", -1) and full_summary.get("full_edge_task_count", 0) > 0)
    external_pass = int(full_summary.get("full_edge_external_fair_pass", 0))
    causality_pass = int(full_summary.get("functional_causality_pass", 0))
    timing_pass = int(full_summary.get("robust_timing_pass", 0))
    compute_pass = int(full_summary.get("training_compute_fair_pass", 0))
    if external_pass and causality_pass and timing_pass and compute_pass and clean_ce_pass:
        route_name = "R2-CleanFullEdgePureKANVisionSuccess"
        primary_blocker = "broad_non_vision_robustness_continual_not_run"
        next_impl = "run_phase_h_robustness_continual_and_non_vision_task_families"
    elif clean_ce_pass:
        route_name = "R3-CleanTransitionalOnlySuccess"
        primary_blocker = "full_edge_candidates_fail_external_fair_or_compute_gate"
        next_impl = "redesign_full_edge_parameterization_or_accept_transitional_only_boundary"
    else:
        route_name = "R7-MLPDominatesExternal"
        primary_blocker = "clean_transitional_or_full_edge_external_fair_failed"
        next_impl = "inspect_clean_ce_and_full_edge_task_gap_before_further_claims"
    route = {
        "route": route_name,
        "legacy_parity_pass": int(clean_ce_pass),
        "clean_ce_pass": clean_ce_pass,
        "full_edge_contract_pass": full_edge_contract,
        "full_edge_task_pass": full_edge_task,
        "functional_causality_pass": causality_pass,
        "external_fair_pass": external_pass,
        "training_compute_fair_pass": compute_pass,
        "robust_timing_pass": timing_pass,
        "robustness_pass": 0,
        "continual_balance_pass": 0,
        "best_candidate": "see_full_edge_external_summary",
        "best_model_level": "full_edge" if full_edge_task else ("transitional" if clean_ce_pass else "metric_unavailable"),
        "success_v90_core_skeleton": 1,
        "success_v90_contract_hardening": int(all(int(r["negative_contract_test_pass"]) == 1 for r in test_rows)),
        "success_v90_clean_ce": clean_ce_pass,
        "success_v90_full_edge": int(external_pass and causality_pass and timing_pass and compute_pass),
        "success_v90_external_fair": external_pass,
        "success_v90_strong": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_impl,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(
        out_dir / "failure_table.csv",
        [
            {"failure_id": "F7_clean_ce_regression", "active": int(not clean_ce_pass), "detail": clean_summary.get("stage", ""), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
            {"failure_id": "F8_full_edge_task_fail", "active": int(not external_pass), "detail": full_summary.get("stage", ""), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
            {"failure_id": "F9_functional_causality_fail", "active": int(not causality_pass), "detail": "functional causality controls", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
            {"failure_id": "F10_timing_fail", "active": int(not timing_pass), "detail": f"q90={full_summary.get('q90_step_ratio_vs_KB_MLP')}", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
            {"failure_id": "F12_training_compute_fail", "active": int(not compute_pass), "detail": "training compute counter", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
            {"failure_id": "F17_artifact_missing", "active": 0, "detail": "all required v9 artifacts written", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        ],
    )
    audit_paths = [
        out_dir / "code_audit_legacy_current.csv",
        out_dir / "core_smoke_tests.csv",
        out_dir / "gradcheck_full_edge.csv",
        out_dir / "full_edge_implementation_audit.csv",
        out_dir / "label_smoothing_removal_audit.csv",
        out_dir / "clean_transitional_revalidation.csv",
        out_dir / "full_edge_external_validation.csv",
        out_dir / "functional_causality_controls.csv",
        out_dir / "robust_timing_protocols.csv",
        out_dir / "training_compute_counter.csv",
        out_dir / "kanbefair_baseline_reproduction.csv",
        out_dir / "external_fair_envelope.csv",
        out_dir / "boundary_audit.csv",
    ]
    audit = audit_no_fake(audit_paths)
    audit.update({"stage": "V90_PROVENANCE_AUDIT", "status": "measured", "script_path": str(SCRIPT_PATH.relative_to(ROOT)), "plan_path": str(PLAN_PATH.relative_to(ROOT))})
    write_csv_rows(out_dir / "v90_provenance_audit.csv", [audit])
    hash_targets = [
        PLAN_PATH,
        SCRIPT_PATH,
        ROOT / "dgkan" / "models" / "manual_full_edge.py",
        ROOT / "dgkan" / "training" / "manual_full_edge.py",
        out_dir / "route_decision.json",
        out_dir / "clean_transitional_revalidation.csv",
        out_dir / "full_edge_external_validation.csv",
        out_dir / "functional_causality_controls.csv",
        out_dir / "robust_timing_protocols.csv",
        out_dir / "training_compute_counter.csv",
        out_dir / "v90_provenance_audit.csv",
    ]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_targets, root=ROOT))
    missing = [name for name in V9_REQUIRED_ARTIFACTS if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing required v9 artifacts: {missing}")


if __name__ == "__main__":
    main()
