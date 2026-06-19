#!/usr/bin/env python3
"""v22.18 Class_MNIST continual-learning smoke for DG MLP/FU."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shlex
import sys
import time
import math
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from dgkan.fu.continual_source_state import forgetting_reduction  # noqa: E402
from dgkan.fu.mlp_adaptive_controller import MLPFUState, apply_mlp_fu_update  # noqa: E402
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments.run_v22_16_common import ce_cotangent, selected_named_parameters  # noqa: E402
from experiments.run_v22_17_kanbefair_dgkan_eval import _flat_grads, _get_loaders, _assign_flat_update, _virtual_train_loss  # noqa: E402
from experiments.run_v22_18_common import (  # noqa: E402
    OUT_ROOT,
    PYTHON,
    WORKTREE_ROOT,
    append_exec,
    ensure_out,
    finite_float,
    run_logged,
    write_rows,
)


TASKS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="DGMLP,DGMLP_FU_RELEASE")
    p.add_argument("--seeds", default="0")
    p.add_argument("--physical-gpu", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=1536)
    p.add_argument("--test-size", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--steps-per-task", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--boundary-memory-steps", type=int, default=120)
    p.add_argument("--boundary-memory-window", type=int, default=32)
    p.add_argument("--fu-ratio-cap", type=float, default=0.01)
    p.add_argument("--continual-virtual-loss-tolerance", type=float, default=-1.0)
    p.add_argument("--continual-virtual-gate-scale", type=float, default=0.002)
    p.add_argument("--run-kanbefair-original-dryrun", action="store_true")
    p.add_argument("--output-suffix", default="v22_18_class_mnist_dg_smoke")
    return p


def _out(name: str, suffix: str) -> Path:
    clean = suffix.strip().strip("_")
    path = OUT_ROOT / name
    if clean:
        return path.with_name(f"{path.stem}_{clean}{path.suffix}")
    return path


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _task_loader(loader: DataLoader, task_idx: int, batch_size: int, seed: int, train: bool) -> DataLoader:
    wanted = set(TASKS[task_idx])
    labels = []
    for idx in range(len(loader.dataset)):
        item = loader.dataset[idx]
        y = int(item[1])
        if y in wanted:
            labels.append(idx)
    gen = torch.Generator().manual_seed(int(seed) + 7000 + task_idx * 101 + (0 if train else 10000))
    if labels:
        perm = torch.randperm(len(labels), generator=gen).tolist()
        labels = [labels[i] for i in perm]
    return DataLoader(Subset(loader.dataset, labels), batch_size=batch_size, shuffle=train, generator=gen, num_workers=0, drop_last=False)


def _eval_task(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            mask = yb < 9
            if not mask.any():
                continue
            logits = model(xb[mask]).float()
            y = yb[mask]
            losses.append(F.cross_entropy(logits, y, reduction="sum").detach().cpu())
            pred = logits.argmax(dim=-1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())
    loss = float(torch.stack(losses).sum().item() / max(1, total)) if losses else 0.0
    return correct / max(1, total), loss


def _unit_param_source(vec: torch.Tensor) -> torch.Tensor:
    flat = vec.detach().float().reshape(-1)
    norm = torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
    return flat / norm


def _continual_source_mode(model_name: str) -> str:
    upper = str(model_name).upper()
    if "BOUNDARY_MEMORY" in upper:
        return "boundary_memory"
    if "PARAM_SOURCE" in upper:
        return "param_update"
    return "output_cotangent"


def _train_one(model_name: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    train_loader, test_loader, _classes, input_dim = _get_loaders(WORKTREE_ROOT, "MNIST", args.batch_size, args.train_size, args.test_size, seed)
    task_train = [_task_loader(train_loader, i, args.batch_size, seed, True) for i in range(3)]
    task_test = [_task_loader(test_loader, i, args.batch_size, seed, False) for i in range(3)]
    model = MLPBaseline(input_dim, 9, args.hidden, seed + 2218, device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    fu_enabled = "_FU" in model_name
    state = MLPFUState()
    selected = selected_named_parameters(model, "all")
    all_params = [p for p in model.parameters() if p.requires_grad]
    source_mode = _continual_source_mode(model_name)
    acc_matrix: dict[tuple[int, int], float] = {}
    nll_matrix: dict[tuple[int, int], float] = {}
    source_loss_boundary = []
    source_retention_boundary = []
    release_near_boundary = 0
    action_near_boundary = 0
    boundary_memory_refresh_count = 0
    boundary_memory_active_steps = 0
    overhead = 0.0
    total_time = 0.0
    auc_losses = []
    boundary_memory: torch.Tensor | None = None
    virtual_loss_base_values: list[float] = []
    virtual_loss_guided_values: list[float] = []
    virtual_reject_count = 0

    for task_idx in range(3):
        recent_task_sources: list[torch.Tensor] = []
        if fu_enabled and source_mode == "boundary_memory" and boundary_memory is not None and task_idx > 0:
            state.source_state = boundary_memory.to(device).detach().clone()
            state.source_age = 0
            boundary_memory_refresh_count += 1
        train_iter = iter(task_train[task_idx])
        for step in range(1, int(args.steps_per_task) + 1):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(task_train[task_idx])
                xb, yb = next(train_iter)
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            mask = yb < 9
            if not mask.any():
                continue
            xb = xb[mask]
            yb = yb[mask]
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            auc_losses.append(float(loss.detach().item()))
            if fu_enabled:
                t0 = time.perf_counter()
                grad_flat = _flat_grads(selected).detach().float()
                update_flat = -grad_flat.reshape(-1)
                current_param_source = _unit_param_source(update_flat).to(device)
                recent_task_sources.append(current_param_source.detach().cpu())
                if len(recent_task_sources) > int(args.boundary_memory_window):
                    recent_task_sources.pop(0)
                use_boundary_memory = (
                    source_mode == "boundary_memory"
                    and boundary_memory is not None
                    and task_idx > 0
                    and step <= int(args.boundary_memory_steps)
                )
                if use_boundary_memory:
                    source_signal = boundary_memory.to(device).detach().clone()
                    boundary_memory_active_steps += 1
                elif source_mode in {"param_update", "boundary_memory"}:
                    source_signal = current_param_source
                else:
                    delta = ce_cotangent(logits.detach(), yb).reshape(-1)
                    source_signal = -delta
                if source_signal.numel() == grad_flat.numel():
                    src_unit = _unit_param_source(source_signal).to(device)
                    source_loss_gain = float((-torch.dot(grad_flat.reshape(-1).to(device), src_unit)).div(max(1.0, math.sqrt(float(grad_flat.numel())))).item())
                else:
                    delta = ce_cotangent(logits.detach(), yb).reshape(-1)
                    source_loss_gain = float(delta.square().mean().item())
                old_state = copy.deepcopy(state)
                guided, state, diag = apply_mlp_fu_update(
                    grad_flat,
                    state,
                    variant="M6 MLP+SourceReleaseController",
                    source_signal=source_signal,
                    source_loss_gain=source_loss_gain,
                    step=task_idx * int(args.steps_per_task) + step,
                    seed=seed,
                    ratio_cap=float(args.fu_ratio_cap),
                )
                virtual_rejected = False
                if float(args.continual_virtual_loss_tolerance) >= 0.0:
                    base_virtual_loss = _virtual_train_loss(model, selected, xb, yb, update_flat, float(args.continual_virtual_gate_scale))
                    guided_virtual_loss = _virtual_train_loss(model, selected, xb, yb, guided, float(args.continual_virtual_gate_scale))
                    virtual_loss_base_values.append(float(base_virtual_loss))
                    virtual_loss_guided_values.append(float(guided_virtual_loss))
                    diag["continual_virtual_loss_base"] = base_virtual_loss
                    diag["continual_virtual_loss_guided"] = guided_virtual_loss
                    diag["continual_virtual_loss_delta"] = guided_virtual_loss - base_virtual_loss
                    if guided_virtual_loss > base_virtual_loss + float(args.continual_virtual_loss_tolerance):
                        virtual_rejected = True
                        virtual_reject_count += 1
                        state = old_state
                        diag["intervention_flag"] = 0
                        diag["lambda_t"] = 0.0
                        diag["continual_gate_reject_reason"] = "continual_virtual_train_loss_gate_failed"
                if not virtual_rejected:
                    _assign_flat_update(selected, guided)
                overhead += time.perf_counter() - t0
                if step >= int(args.steps_per_task) - 5:
                    action_near_boundary += int(diag.get("intervention_flag", 0))
                    release_near_boundary += int(diag.get("source_release_count", 0))
                    source_loss_boundary.append(source_loss_gain)
                    source_retention_boundary.append(float(diag.get("source_func", 0.0)))
            torch.nn.utils.clip_grad_norm_(all_params, 2.0)
            opt.step()
            total_time += time.perf_counter() - start
        for eval_task in range(task_idx + 1):
            acc, nll = _eval_task(model, task_test[eval_task], device)
            acc_matrix[(task_idx, eval_task)] = acc
            nll_matrix[(task_idx, eval_task)] = nll
        if fu_enabled and source_mode == "boundary_memory" and recent_task_sources:
            memory = torch.stack([x.float().reshape(-1) for x in recent_task_sources], dim=0).mean(dim=0)
            boundary_memory = _unit_param_source(memory).to(device).detach().clone()

    row: dict[str, Any] = {
        "dataset": "Class_MNIST",
        "protocol": "DG_reimplementation_of_KANbeFair_digits_0-2_3-5_6-8_num_classes_9",
        "model_name": model_name,
        "origin": "DGKAN_v22_18",
        "seed": seed,
        "steps_per_task": args.steps_per_task,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "hidden": args.hidden,
        "average_accuracy": sum(acc_matrix.get((2, i), 0.0) for i in range(3)) / 3.0,
        "average_NLL_after_T2": sum(nll_matrix.get((2, i), 0.0) for i in range(3)) / 3.0,
        "AUC_loss_time_per_task": sum(auc_losses) / max(1, len(auc_losses)),
        "controller_overhead_ratio": overhead / max(1.0e-12, total_time),
        "source_retention_across_task_boundary": sum(source_retention_boundary) / max(1, len(source_retention_boundary)) if source_retention_boundary else "",
        "source_loss_across_task_boundary": sum(source_loss_boundary) / max(1, len(source_loss_boundary)) if source_loss_boundary else "",
        "controller_action_count_near_boundary": action_near_boundary,
        "source_release_count_near_boundary": release_near_boundary,
        "continual_source_mode": source_mode,
        "boundary_memory_refresh_count": boundary_memory_refresh_count,
        "boundary_memory_active_steps": boundary_memory_active_steps,
        "boundary_memory_steps": int(args.boundary_memory_steps),
        "boundary_memory_window": int(args.boundary_memory_window),
        "fu_ratio_cap": float(args.fu_ratio_cap),
        "continual_virtual_loss_tolerance": float(args.continual_virtual_loss_tolerance),
        "continual_virtual_gate_scale": float(args.continual_virtual_gate_scale),
        "continual_virtual_reject_count": virtual_reject_count,
        "mean_continual_virtual_loss_base": sum(virtual_loss_base_values) / max(1, len(virtual_loss_base_values)) if virtual_loss_base_values else "",
        "mean_continual_virtual_loss_guided": sum(virtual_loss_guided_values) / max(1, len(virtual_loss_guided_values)) if virtual_loss_guided_values else "",
        "dgkan_official_evidence": 1,
    }
    for after_task in range(3):
        for eval_task in range(after_task + 1):
            row[f"ACC_after_task_{after_task}_on_task_{eval_task}"] = acc_matrix.get((after_task, eval_task), "")
            row[f"NLL_after_task_{after_task}_on_task_{eval_task}"] = nll_matrix.get((after_task, eval_task), "")
    if (0, 0) in acc_matrix and (2, 0) in acc_matrix:
        row["Forgetting_T0_after_T2"] = max(0.0, acc_matrix[(0, 0)] - acc_matrix[(2, 0)])
    if (1, 1) in acc_matrix and (2, 1) in acc_matrix:
        row["Forgetting_T1_after_T2"] = max(0.0, acc_matrix[(1, 1)] - acc_matrix[(2, 1)])
    row["average_backward_transfer"] = -(
        finite_float(row.get("Forgetting_T0_after_T2")) + finite_float(row.get("Forgetting_T1_after_T2"))
    ) / 2.0
    return row


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    if args.run_kanbefair_original_dryrun:
        run_logged(
            [PYTHON, "experiments/run_v22_17_kanbefair_continual_eval.py", "--models", "MLP,KAN", "--dry-run"],
            task_id="H-kanbefair-original-continual-dryrun",
            gpu="external script default",
            timeout=None,
        )
    device = _device(args.device)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        for model_name in models:
            rows.append(_train_one(model_name, seed, args, device))
    summary = []
    for seed in seeds:
        base = next((r for r in rows if r.get("seed") == seed and r.get("model_name") == "DGMLP"), None)
        fu_rows = [r for r in rows if r.get("seed") == seed and str(r.get("model_name", "")).startswith("DGMLP_FU")]
        for fu in fu_rows:
            if not base:
                continue
            metrics = forgetting_reduction(base, fu)
            summary.append(
                {
                    "seed": seed,
                    "base_model_name": base.get("model_name", ""),
                    "fu_model_name": fu.get("model_name", ""),
                    "continual_source_mode": fu.get("continual_source_mode", ""),
                    "controller_overhead_ratio": fu.get("controller_overhead_ratio", ""),
                    "controller_action_count_near_boundary": fu.get("controller_action_count_near_boundary", ""),
                    "source_release_count_near_boundary": fu.get("source_release_count_near_boundary", ""),
                    "boundary_memory_refresh_count": fu.get("boundary_memory_refresh_count", ""),
                    "boundary_memory_active_steps": fu.get("boundary_memory_active_steps", ""),
                    "continual_virtual_loss_tolerance": fu.get("continual_virtual_loss_tolerance", ""),
                    "continual_virtual_reject_count": fu.get("continual_virtual_reject_count", ""),
                    "mean_continual_virtual_loss_base": fu.get("mean_continual_virtual_loss_base", ""),
                    "mean_continual_virtual_loss_guided": fu.get("mean_continual_virtual_loss_guided", ""),
                    **metrics,
                    "H_exploration_pass": int(metrics["absolute_forgetting_reduction"] > 0.0 and metrics["average_accuracy_delta"] >= -0.005),
                    "H_official_pass": int(metrics["relative_forgetting_reduction"] >= 0.10 and metrics["average_accuracy_delta"] >= -0.005 and len(seeds) >= 3),
                }
            )
    write_rows(_out("v22_18_class_mnist_continual_matrix.csv", args.output_suffix), rows)
    write_rows(_out("v22_18_forgetting_transfer_summary.csv", args.output_suffix), summary)
    cmdline = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmdline,
        task_id="H-continual-dg-smoke",
        status="pass" if rows else "fail",
        gpu=f"physical {args.physical_gpu or 'default'}; script device {args.device}",
        exit_code=0 if rows else 1,
        files=(
            f"{_out('v22_18_class_mnist_continual_matrix.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_forgetting_transfer_summary.csv', args.output_suffix).relative_to(ROOT)}"
        ),
        note=f"rows={len(rows)}; summary_rows={len(summary)}; exact KANbeFair Class_MNIST task definition used, DG implementation",
    )


if __name__ == "__main__":
    main()
