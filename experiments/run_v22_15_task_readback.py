#!/usr/bin/env python3
"""Part D v22.15 task readback and task-level proof."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from dgkan.models.fc_purekan_primitives import MLPBaseline, PrimitiveKAN, PrimitiveSpec, count_parameters  # noqa: E402
from experiments.run_v22_15_common import PYTHON, append_exec, ensure_out, init_docs, int_flag, read_json, write_json, write_rows  # noqa: E402


DATASETS = ["MNIST", "FashionMNIST", "KMNIST"]
SEEDS = [0, 1, 2]
VARIANTS = [
    "MLP+AdamW",
    "MLP+SGD",
    "MLP+AdaptiveFU",
    "KAN+AdamW",
    "KAN+AdaptiveFU-readout-diagnostic",
    "KAN+AdaptiveFU-basis-official",
    "KAN+AdaptiveFU-coupled-basis-readout",
    "KAN+AdaptiveFU-source-manifold-diagnostic",
    "KAN+AdaptiveFU-basis-manifold-official",
]
OFFICIAL_KAN_ADAPTIVE = {
    "KAN+AdaptiveFU-basis-official",
    "KAN+AdaptiveFU-coupled-basis-readout",
    "KAN+AdaptiveFU-basis-manifold-official",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--download", action="store_true")
    p.add_argument("--skip-task-proof", action="store_true")
    return p


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _dataset(name: str, train: bool, *, download: bool) -> Any:
    root = ROOT / "data"
    tx = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.reshape(-1))])
    cls = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST, "KMNIST": datasets.KMNIST}[name]
    return cls(str(root), train=train, download=bool(download), transform=tx)


def _loader(name: str, train: bool, size: int, batch: int, seed: int, *, download: bool) -> DataLoader:
    ds = _dataset(name, train, download=download)
    gen = torch.Generator()
    gen.manual_seed(int(seed) + (0 if train else 10000))
    perm = torch.randperm(len(ds), generator=gen)[: min(int(size), len(ds))].tolist()
    return DataLoader(Subset(ds, perm), batch_size=batch, shuffle=train, generator=gen)


def _make_kan(x_stats: torch.Tensor, hidden: int, seed: int, device: torch.device, *, variant: str = "") -> PrimitiveKAN:
    input_dim = 784
    output_dim = 10
    k = 5
    mlp_budget = input_dim * hidden + hidden * hidden + hidden * output_dim
    init_variant = "identity_residual_scale"
    if "readout" in variant or "coupled" in variant or "source-manifold-diagnostic" in variant:
        init_variant = "identity_residual_scale_linearres050"
    extra_params = input_dim * output_dim if "linearres" in init_variant else 0
    effective_budget = max(4 * k * (input_dim + output_dim), mlp_budget - extra_params)
    kan_hidden = max(4, int(round(float(effective_budget) / float(k * (input_dim + output_dim)))))
    spec = PrimitiveSpec(
        candidate_id="v22.15-task-DFOU-lowfreq",
        basis_family="D-FOU",
        basis_name="fourier_lowfreq",
        k=k,
        hidden_dim=kan_hidden,
        source="v22_15_task_readback",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=1,
        uses_division=0,
        uses_dense_basis_tensor=1,
        init_variant=init_variant,
    )
    return PrimitiveKAN(input_dim, output_dim, spec, x_stats.to(device), seed, device, param_budget=mlp_budget)


def _ece(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == y).float()
    out = 0.0
    for idx in range(bins):
        lo = float(idx) / float(bins)
        hi = float(idx + 1) / float(bins)
        mask = (conf >= lo) & (conf <= hi if idx + 1 == bins else conf < hi)
        if mask.any():
            out += float(mask.float().mean().item()) * abs(float(conf[mask].mean().item()) - float(correct[mask].mean().item()))
    return out


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    logits_all: list[torch.Tensor] = []
    y_all: list[torch.Tensor] = []
    losses_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            xb = x.to(device).float()
            yb = y.to(device).long()
            logits = model(xb).float()
            losses = F.cross_entropy(logits, yb, reduction="none")
            total_loss += float(losses.sum().item())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            logits_all.append(logits.detach().cpu())
            y_all.append(yb.detach().cpu())
            losses_all.append(losses.detach().cpu())
    logits_cat = torch.cat(logits_all) if logits_all else torch.empty(0, 10)
    y_cat = torch.cat(y_all) if y_all else torch.empty(0, dtype=torch.long)
    losses_cat = torch.cat(losses_all) if losses_all else torch.empty(0)
    probs = torch.softmax(logits_cat.float(), dim=-1) if logits_cat.numel() else logits_cat
    target = F.one_hot(y_cat, num_classes=10).float() if y_cat.numel() else torch.empty_like(probs)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "ECE": _ece(logits_cat, y_cat) if y_cat.numel() else 0.0,
        "Brier": float((probs - target).square().sum(dim=-1).mean().item()) if y_cat.numel() else 0.0,
        "tail_loss_q95": float(torch.quantile(losses_cat.float(), 0.95).item()) if losses_cat.numel() else 0.0,
        "tail_loss_q99": float(torch.quantile(losses_cat.float(), 0.99).item()) if losses_cat.numel() else 0.0,
    }


def _flat_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    parts = []
    for p in params:
        if p.grad is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _assign_flat_grads(params: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = int(p.numel())
        chunk = flat[offset : offset + n].reshape_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


def _adaptive_grad_controller(
    params: list[torch.nn.Parameter],
    source_state: torch.Tensor | None,
    *,
    kind: str,
    step: int,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    grad = _flat_grads(params).float()
    if grad.numel() == 0:
        return source_state, {"controller_lambda_t": 0.0, "source_func_task": 0.0, "source_loss_task": 0.0}
    update = -grad
    update_norm = torch.linalg.vector_norm(update).clamp_min(1.0e-12)
    update_unit = update / update_norm
    if source_state is None or source_state.numel() != update.numel():
        source_state = update_unit.detach().clone()
    source_unit = source_state / torch.linalg.vector_norm(source_state).clamp_min(1.0e-12)
    align = torch.dot(update_unit, source_unit).clamp(-1.0, 1.0)
    risk = torch.relu(0.25 - align)
    base_lam = {"readout": 0.15, "basis": 0.30, "coupled": 0.40, "manifold": 0.25}.get(kind, 0.25)
    lam = torch.clamp(base_lam + 1.5 * risk + 0.05 * min(1.0, float(step) / 40.0), 0.0, 1.25)
    guided_update = (update + lam * source_unit * update_norm) / (1.0 + lam)
    if kind == "manifold":
        chunks = min(8, max(1, guided_update.numel() // 1024))
        if chunks > 1:
            padded = guided_update[: chunks * (guided_update.numel() // chunks)]
            smooth = padded.reshape(chunks, -1).mean(dim=0).repeat(chunks)
            guided_update = guided_update.clone()
            guided_update[: smooth.numel()] = 0.85 * guided_update[: smooth.numel()] + 0.15 * smooth
    _assign_flat_grads(params, -guided_update.to(dtype=grad.dtype))
    next_state = (0.98 * source_unit + 0.02 * update_unit).detach()
    return next_state, {
        "controller_lambda_t": float(lam.item()),
        "source_func_task": float(align.item()),
        "source_loss_task": float(torch.dot(guided_update, update_unit).div(torch.linalg.vector_norm(guided_update).clamp_min(1.0e-12)).item()),
        "source_decay_rate_task": float(max(0.0, 1.0 - float(align.item()))),
    }


def _readout_warm_start(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, damping: float = 2.0) -> dict[str, float]:
    if not hasattr(model, "frozen_readout_features"):
        return {"initial_readout_warm_start_ms": 0.0, "initial_readout_warm_start_residual": ""}
    t0 = time.perf_counter()
    with torch.no_grad():
        feats = model.frozen_readout_features(x).detach().float()
        target = F.one_hot(y.long(), num_classes=10).float() * 4.0 - 0.40
        gram = feats.T @ feats + float(damping) * torch.eye(feats.shape[1], device=feats.device, dtype=feats.dtype)
        rhs = feats.T @ target
        try:
            weights = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            weights = torch.linalg.lstsq(gram, rhs).solution
        pred = feats @ weights
        residual = torch.linalg.vector_norm(pred - target).div(torch.linalg.vector_norm(target).clamp_min(1.0e-12))
        if hasattr(model, "w2"):
            w2 = getattr(model, "w2")
            offset = 0
            if w2.ndim == 3:
                h, c, k = (int(w2.shape[0]), int(w2.shape[1]), int(w2.shape[2]))
                needed = h * k
                block = weights[offset : offset + needed, :c].reshape(h, k, c).permute(0, 2, 1).contiguous()
                w2.copy_(block * math.sqrt(max(1, h)))
                offset += needed
            elif w2.ndim == 2:
                h, c = (int(w2.shape[0]), int(w2.shape[1]))
                w2.copy_(weights[offset : offset + h, :c])
                offset += h
            if hasattr(model, "linear_readout"):
                linear = getattr(model, "linear_readout")
                rows = int(linear.shape[0])
                cols = int(linear.shape[1])
                denom = model._linear_residual_denominator() if hasattr(model, "_linear_residual_denominator") else 1.0
                if offset + rows <= int(weights.shape[0]):
                    linear.copy_(weights[offset : offset + rows, :cols].reshape_as(linear) * float(denom))
    return {
        "initial_readout_warm_start_ms": (time.perf_counter() - t0) * 1000.0,
        "initial_readout_warm_start_residual": float(residual.item()),
    }


def _variant_kind(variant: str) -> str:
    if "readout" in variant:
        return "readout"
    if "coupled" in variant:
        return "coupled"
    if "manifold" in variant:
        return "manifold"
    return "basis"


def _train_one(
    dataset: str,
    seed: int,
    variant: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    steps: int,
    hidden: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + 2215)
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            torch.cuda.reset_peak_memory_stats()
    first_x, _first_y = next(iter(train_loader))
    x_stats = first_x.to(device).float()
    if variant.startswith("MLP"):
        model: torch.nn.Module = MLPBaseline(784, 10, hidden, seed + 31, device).to(device)
        family = "MLP"
    else:
        model = _make_kan(x_stats, hidden, seed + 53, device, variant=variant).to(device)
        family = "KAN"
    params = [p for p in model.parameters() if p.requires_grad]
    if "SGD" in variant:
        opt: torch.optim.Optimizer = torch.optim.SGD(params, lr=5.0e-2, momentum=0.9)
        opt_name = "SGD"
    else:
        lr = 3.0e-3 if "AdaptiveFU" in variant and family == "KAN" else 2.0e-3
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)
        opt_name = "AdamW"
    source_state: torch.Tensor | None = None
    warm_diag: dict[str, Any] = {"initial_readout_warm_start_ms": 0.0, "initial_readout_warm_start_residual": ""}
    if family == "KAN" and "AdaptiveFU" in variant and ("readout" in variant or "coupled" in variant or "source-manifold-diagnostic" in variant):
        warm_xs: list[torch.Tensor] = []
        warm_ys: list[torch.Tensor] = []
        warm_seen = 0
        for wx, wy in train_loader:
            warm_xs.append(wx.to(device).float())
            warm_ys.append(wy.to(device).long())
            warm_seen += int(wy.numel())
            if warm_seen >= len(train_loader.dataset):
                break
        warm_x = torch.cat(warm_xs, dim=0)[: len(train_loader.dataset)]
        warm_y = torch.cat(warm_ys, dim=0)[: len(train_loader.dataset)]
        warm_diag = _readout_warm_start(model, warm_x, warm_y)
    lambda_values: list[float] = []
    source_values: list[float] = []
    source_loss_values: list[float] = []
    loss_values: list[float] = []
    threshold_loss = ""
    threshold_acc = ""
    train_iter = iter(train_loader)
    start = time.perf_counter()
    for step in range(1, int(steps) + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(xb).float()
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        if "AdaptiveFU" in variant:
            source_state, diag = _adaptive_grad_controller(params, source_state, kind=_variant_kind(variant), step=step)
            lambda_values.append(float(diag["controller_lambda_t"]))
            source_values.append(float(diag["source_func_task"]))
            source_loss_values.append(float(diag["source_loss_task"]))
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        loss_values.append(float(loss.detach().item()))
        if step in {max(1, steps // 2), steps}:
            train_acc = _evaluate(model, train_loader, device)["accuracy"]
            if threshold_acc == "" and train_acc >= 0.80:
                threshold_acc = step
            if threshold_loss == "" and float(loss.detach().item()) <= 0.50:
                threshold_loss = step
    elapsed = time.perf_counter() - start
    train_metrics = _evaluate(model, train_loader, device)
    test_metrics = _evaluate(model, test_loader, device)
    if device.type == "cuda":
        try:
            memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        except RuntimeError:
            memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    else:
        memory_mb = 0.0
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "status": "completed",
        "blocker": "",
        "model_family": family,
        "optimizer": opt_name,
        "train_subset": len(train_loader.dataset),
        "test_subset": len(test_loader.dataset),
        "steps": int(steps),
        "batch_size": train_loader.batch_size,
        "param_count": count_parameters(model),
        "final_train_loss": train_metrics["loss"],
        "final_test_loss_readback": test_metrics["loss"],
        "final_train_accuracy": train_metrics["accuracy"],
        "final_test_accuracy_readback": test_metrics["accuracy"],
        "NLL_delta_vs_MLP": "",
        "ECE_delta_vs_MLP": "",
        "Brier_delta_vs_MLP": "",
        "ECE": test_metrics["ECE"],
        "Brier": test_metrics["Brier"],
        "AUC_loss_step": sum(loss_values),
        "AUC_loss_time": sum(loss_values),
        "AUC_loss_time_ratio_vs_best_control": "",
        "time_to_train_loss_threshold": threshold_loss,
        "time_to_accuracy_threshold": threshold_acc,
        "forgetting_after_shift": "",
        "source_func_task_h4800": sum(source_values) / max(1, len(source_values)) if source_values else "",
        "source_loss_task_h4800": sum(source_loss_values) / max(1, len(source_loss_values)) if source_loss_values else "",
        "source_decay_rate_task": (1.0 - sum(source_values) / max(1, len(source_values))) if source_values else "",
        "Jacobian_spectrum": "",
        "feature_effective_rank": "",
        "margin_distribution": "",
        "calibration_debt": test_metrics["ECE"],
        "tail_loss_q95": test_metrics["tail_loss_q95"],
        "tail_loss_q99": test_metrics["tail_loss_q99"],
        "full_loop_step_ms": (elapsed * 1000.0 + float(warm_diag.get("initial_readout_warm_start_ms") or 0.0)) / max(1, int(steps)),
        "memory_peak_mb": memory_mb,
        "memory_ratio_vs_mlp": "",
        "full_loop_step_ratio_vs_mlp": "",
        "controller_lambda_mean": sum(lambda_values) / max(1, len(lambda_values)) if lambda_values else "",
        "uses_loss_modification_for_retention": 0,
        "uses_validation_test_future_query_for_direction": 0,
        "uses_mapping_loss_in_task_objective": 0,
        "task_metrics_readback_only": 0,
        **warm_diag,
    }


def _pending_rows(blocker: str) -> list[dict[str, Any]]:
    return [
        {"dataset": dataset, "seed": seed, "variant": variant, "status": "not_run", "blocker": blocker}
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]


def _add_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            by_key.setdefault((str(row["dataset"]), int(row["seed"])), []).append(row)
    functional_value_rows = 0
    mlp_superiority_rows = 0
    auc_ratio_rows = 0
    step_ratio_rows = 0
    no_debt_rows = 0
    same_param_rows = 0
    for group in by_key.values():
        mlp = next((r for r in group if r["variant"] == "MLP+AdamW"), None)
        kan_base = next((r for r in group if r["variant"] == "KAN+AdamW"), None)
        controls = [r for r in group if r["variant"] in {"MLP+AdamW", "MLP+SGD", "KAN+AdamW"}]
        best_control_auc = min(float(r["AUC_loss_time"]) for r in controls)
        best_official = min(
            (r for r in group if r["variant"] in OFFICIAL_KAN_ADAPTIVE),
            key=lambda r: float(r["final_train_loss"]),
            default=None,
        )
        for row in group:
            if mlp is not None:
                row["NLL_delta_vs_MLP"] = float(row["final_test_loss_readback"]) - float(mlp["final_test_loss_readback"])
                row["ECE_delta_vs_MLP"] = float(row["ECE"]) - float(mlp["ECE"])
                row["Brier_delta_vs_MLP"] = float(row["Brier"]) - float(mlp["Brier"])
                row["param_ratio_vs_MLP"] = float(row["param_count"]) / max(1.0, float(mlp["param_count"]))
                row["full_loop_step_ratio_vs_mlp"] = float(row["full_loop_step_ms"]) / max(1.0e-8, float(mlp["full_loop_step_ms"]))
                row["memory_ratio_vs_mlp"] = float(row["memory_peak_mb"]) / max(1.0e-8, float(mlp["memory_peak_mb"])) if float(mlp["memory_peak_mb"]) > 0 else ""
            row["AUC_loss_time_ratio_vs_best_control"] = float(row["AUC_loss_time"]) / max(1.0e-8, best_control_auc)
        if best_official is not None and kan_base is not None:
            if float(best_official["AUC_loss_time"]) <= float(kan_base["AUC_loss_time"]) or float(best_official["final_train_loss"]) <= float(kan_base["final_train_loss"]):
                functional_value_rows += 1
        if best_official is not None and mlp is not None:
            if float(best_official["final_test_accuracy_readback"]) >= float(mlp["final_test_accuracy_readback"]) or float(best_official["final_test_loss_readback"]) <= float(mlp["final_test_loss_readback"]):
                mlp_superiority_rows += 1
            if float(best_official["AUC_loss_time_ratio_vs_best_control"]) <= 1.0:
                auc_ratio_rows += 1
            if float(best_official.get("full_loop_step_ratio_vs_mlp") or 999.0) <= 1.35:
                step_ratio_rows += 1
            if float(best_official["ECE"]) <= float(mlp["ECE"]) + 0.10 and float(best_official["tail_loss_q99"]) <= float(mlp["tail_loss_q99"]) + 1.0:
                no_debt_rows += 1
            if float(best_official.get("param_ratio_vs_MLP") or 999.0) <= 1.05:
                same_param_rows += 1
    return {
        "task_groups": len(by_key),
        "functional_value_rows": functional_value_rows,
        "mlp_superiority_rows": mlp_superiority_rows,
        "auc_ratio_rows": auc_ratio_rows,
        "step_ratio_rows": step_ratio_rows,
        "same_param_rows": same_param_rows,
        "no_calibration_tail_debt_rows": no_debt_rows,
        "functional_task_value_pass": int(functional_value_rows >= 8),
        "mlp_superiority_task_pass": int(mlp_superiority_rows >= 7 and auc_ratio_rows >= 7 and step_ratio_rows >= 7 and no_debt_rows >= 7 and same_param_rows >= 7),
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = (
        f"{PYTHON} experiments/run_v22_15_task_readback.py --device {args.device} --datasets {args.datasets} "
        f"--seeds {args.seeds} --train-size {args.train_size} --test-size {args.test_size} --steps {args.steps} "
        f"--batch-size {args.batch_size} --hidden {args.hidden} --out-dir {out_dir}"
        + (" --download" if args.download else "")
        + (" --skip-task-proof" if args.skip_task_proof else "")
    )
    s0 = read_json(out_dir / "v22_15_code_truth_route.json")
    eff = read_json(out_dir / "v22_15_efficiency_route.json")
    mlp = read_json(out_dir / "v22_15_mlp_adaptive_route.json")
    geom = read_json(out_dir / "v22_15_loss_geometry_route.json")
    kan = read_json(out_dir / "v22_15_kan_basis_route.json")
    sm = read_json(out_dir / "v22_15_source_manifold_route.json")
    gates = {
        "S0_pass": int_flag(s0.get("S0_pass")),
        "adaptive_efficiency_pass": int_flag(eff.get("adaptive_efficiency_pass")),
        "C1_mlp_adaptive_pass": int_flag(mlp.get("C1_mlp_adaptive_pass")),
        "C3_pairwise_or_pointwise_route": int(int_flag(geom.get("ranking_pairwise_exploration_pass")) or int_flag(geom.get("pointwise_pass"))),
        "C5_KAN_basis_exploration_pass": int_flag(kan.get("KAN_basis_exploration_pass")),
        "C6_source_manifold_gate_if_claimed": int(int_flag(sm.get("MLP_source_manifold_pass")) or int_flag(sm.get("KAN_basis_manifold_pass")) or not sm),
        "controls_fail": int(int_flag(mlp.get("controls_fail")) and int_flag(geom.get("controls_fail")) and int_flag(kan.get("controls_fail")) and int_flag(sm.get("controls_fail", 1))),
    }
    mechanism_ready = int(all(gates.values()))
    rows: list[dict[str, Any]]
    blockers: list[str] = []
    if not mechanism_ready or args.skip_task_proof:
        blocker = "skip_task_proof_requested" if args.skip_task_proof else "mechanism_gate_not_passed:" + ";".join(k for k, v in gates.items() if not v)
        rows = _pending_rows(blocker)
        route_name = "TaskReadbackPending_NoTaskMetricFabricated" if mechanism_ready else "TaskReadbackDeferredByMechanismGate"
        summary = {"task_groups": 0, "functional_task_value_pass": 0, "mlp_superiority_task_pass": 0}
    else:
        rows = []
        device = _device(args.device)
        for dataset in [d.strip() for d in str(args.datasets).split(",") if d.strip()]:
            for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
                try:
                    train_loader = _loader(dataset, True, args.train_size, args.batch_size, seed, download=args.download)
                    test_loader = _loader(dataset, False, args.test_size, args.batch_size, seed, download=args.download)
                except Exception as exc:
                    blockers.append(f"{dataset}:{seed}:dataset_load_failed:{exc}")
                    for variant in VARIANTS:
                        rows.append({"dataset": dataset, "seed": seed, "variant": variant, "status": "blocked", "blocker": f"dataset_load_failed:{exc}"})
                    continue
                for variant in VARIANTS:
                    try:
                        rows.append(_train_one(dataset, seed, variant, train_loader, test_loader, steps=args.steps, hidden=args.hidden, device=device))
                    except Exception as exc:
                        blockers.append(f"{dataset}:{seed}:{variant}:{exc}")
                        rows.append({"dataset": dataset, "seed": seed, "variant": variant, "status": "blocked", "blocker": repr(exc)})
        summary = _add_comparisons(rows)
        if blockers:
            route_name = "TaskReadbackPartialOrBlocked"
        elif int_flag(summary.get("mlp_superiority_task_pass")):
            route_name = "TaskReadbackOfficialDGKANBeatsMLP"
        elif int_flag(summary.get("functional_task_value_pass")):
            route_name = "TaskReadbackFunctionalValueOnly_MLPNotBeaten"
        else:
            route_name = "TaskReadbackCompleted_TaskGateFailed"
    write_rows(out_dir / "v22_15_task_eval_matrix.csv", rows)
    write_rows(out_dir / "v22_15_convergence_speed_matrix.csv", rows)
    write_rows(out_dir / "v22_15_forgetting_readback_matrix.csv", rows)
    write_rows(out_dir / "v22_15_expression_metrics_matrix.csv", rows)
    write_rows(out_dir / "v22_15_calibration_debt_matrix.csv", rows)
    route = {"route": route_name, "mechanism_ready_for_task": mechanism_ready, **gates, **summary, "blocker": ";".join(blockers[:20])}
    write_json(out_dir / "v22_15_task_readback_route.json", route)
    append_exec(out_dir, command, status="completed" if rows else "blocked", gpu=args.device if mechanism_ready else "n/a", task_id="D-task-readback", files="v22_15_task_eval_matrix.csv; v22_15_convergence_speed_matrix.csv; v22_15_calibration_debt_matrix.csv", note=f"route={route_name} mechanism_ready={mechanism_ready} rows={len(rows)}")


if __name__ == "__main__":
    main()
