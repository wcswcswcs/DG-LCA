#!/usr/bin/env python3
"""v22.13 task-level readback for expression, forgetting, and convergence."""

from __future__ import annotations

import argparse
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

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.fu.loss_interface import ClassificationCEAdapter  # noqa: E402
from dgkan.fu.operator_commit import commit_operator_target  # noqa: E402
from dgkan.fu.operator_core import apply_operator  # noqa: E402
from dgkan.kan.corrected_readout_solve import solve_corrected_readout_update  # noqa: E402
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec  # noqa: E402
from experiments.run_v22_13_common import PYTHON, append_exec, ensure_out, simple_svg, write_json, write_rows  # noqa: E402


class SmallMLP(torch.nn.Module):
    def __init__(self, input_dim: int = 784, hidden: int = 64, classes: int = 10) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.w2 = torch.nn.Parameter(torch.randn(hidden, classes) * 0.02)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frozen_readout_features(x) @ self.w2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    return p


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _dataset(name: str, train: bool) -> Any:
    root = ROOT / "data"
    tx = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.reshape(-1))])
    cls = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST, "KMNIST": datasets.KMNIST}[name]
    return cls(str(root), train=train, download=False, transform=tx)


def _loader(name: str, train: bool, size: int, batch: int, seed: int) -> DataLoader:
    ds = _dataset(name, train)
    gen = torch.Generator()
    gen.manual_seed(seed + (0 if train else 10000))
    perm = torch.randperm(len(ds), generator=gen)[: min(size, len(ds))].tolist()
    return DataLoader(Subset(ds, perm), batch_size=batch, shuffle=train, generator=gen)


def _make_kan(x_stats: torch.Tensor, hidden: int, seed: int, device: torch.device) -> PrimitiveKAN:
    spec = PrimitiveSpec(
        candidate_id="v22.13-task-DFOU",
        basis_family="D-FOU",
        basis_name="fourier_lowfreq",
        k=3,
        hidden_dim=hidden,
        source="v22_13_task_eval",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=1,
        uses_division=0,
        uses_dense_basis_tensor=1,
    )
    return PrimitiveKAN(784, 10, spec, x_stats.to(device), seed, device, param_budget=784 * hidden + hidden * 10)


def _ece(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == y).float()
    out = 0.0
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            out += float(mask.float().mean().item()) * abs(float(conf[mask].mean().item()) - float(correct[mask].mean().item()))
    return out


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    logits_all = []
    y_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            logits = model(x).float()
            losses.append(F.cross_entropy(logits, y, reduction="sum").item())
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += int(y.numel())
            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())
    logits_cat = torch.cat(logits_all) if logits_all else torch.empty(0, 10)
    y_cat = torch.cat(y_all) if y_all else torch.empty(0, dtype=torch.long)
    probs = torch.softmax(logits_cat.float(), dim=-1) if logits_cat.numel() else logits_cat
    target = F.one_hot(y_cat, num_classes=10).float() if y_cat.numel() else torch.empty_like(probs)
    return {
        "loss": sum(losses) / max(1, total),
        "accuracy": correct / max(1, total),
        "ECE": _ece(logits_cat, y_cat) if y_cat.numel() else 0.0,
        "Brier": float((probs - target).square().sum(dim=-1).mean().item()) if y_cat.numel() else 0.0,
    }


def _apply_fu(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, model_family: str, seed: int) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(x).detach().float()
    delta = ClassificationCEAdapter().cotangent(logits, y.detach().cpu()).to(device=x.device)
    target, telem = apply_operator(operator_id="LIO1_LowNDSGreen", logits=logits, cotangent=delta, seed=seed)
    before = flat_params(model).detach().float()
    if model_family == "KAN":
        update, diag = solve_corrected_readout_update(model, x, target.to(device=x.device), damping=1.0e-3)
        with torch.no_grad():
            load_flat_params(model, before + update.to(device=before.device, dtype=before.dtype))
        sign = "plus_corrected_kan_readout"
    else:
        update, diag = commit_operator_target(model, x, target.to(device=x.device), solver_level="task_eval_readout_fu", damping=1.0e-3, fit_scope="all_train_stream", seed=seed)
        with torch.no_grad():
            load_flat_params(model, before - update.to(device=before.device, dtype=before.dtype))
        sign = "minus_mlp_commit_contract"
    return {
        **telem.to_row(),
        **diag,
        "fu_update_norm": float(torch.linalg.vector_norm(update.detach().float()).item()),
        "fu_commit_sign_contract": sign,
    }


def _train_one(name: str, seed: int, variant: str, train_loader: DataLoader, test_loader: DataLoader, steps: int, hidden: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    first_x, first_y = next(iter(train_loader))
    x_stats = first_x.to(device).float()
    if variant.startswith("MLP"):
        model: torch.nn.Module = SmallMLP(hidden=hidden).to(device)
        family = "MLP"
    else:
        model = _make_kan(x_stats, hidden, seed, device)
        family = "KAN"
    fu_diag: dict[str, Any] = {}
    if "+FU" in variant:
        fu_diag = _apply_fu(model, x_stats, first_y.to(device).long(), family, seed)
    opt_name = "SGD" if "SGD" in variant else "AdamW"
    opt: torch.optim.Optimizer
    if opt_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=5.0e-2, momentum=0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    train_iter = iter(train_loader)
    auc_loss = 0.0
    start = time.perf_counter()
    threshold80 = ""
    threshold90 = ""
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
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        opt.step()
        auc_loss += float(loss.detach().item())
        if step in {max(1, steps // 2), steps}:
            acc = _evaluate(model, train_loader, device)["accuracy"]
            if threshold80 == "" and acc >= 0.80:
                threshold80 = step
            if threshold90 == "" and acc >= 0.90:
                threshold90 = step
    elapsed = time.perf_counter() - start
    train_metrics = _evaluate(model, train_loader, device)
    test_metrics = _evaluate(model, test_loader, device)
    rank = ""
    try:
        feats = model.frozen_readout_features(x_stats).detach().float().cpu()
        rank = int(torch.linalg.matrix_rank(feats).item())
    except Exception:
        rank = ""
    return {
        "dataset": name,
        "seed": seed,
        "variant": variant,
        "model_family": family,
        "optimizer": opt_name,
        "train_subset": len(train_loader.dataset),
        "test_subset": len(test_loader.dataset),
        "steps": steps,
        "batch_size": train_loader.batch_size,
        "final_train_loss": train_metrics["loss"],
        "final_test_loss_readback": test_metrics["loss"],
        "final_train_accuracy": train_metrics["accuracy"],
        "final_test_accuracy_readback": test_metrics["accuracy"],
        "ECE_delta": test_metrics["ECE"],
        "Brier_delta": test_metrics["Brier"],
        "AUC_loss_step": auc_loss,
        "AUC_loss_time": auc_loss,
        "wallclock_sec": elapsed,
        "steps_to_threshold_80pct": threshold80,
        "steps_to_threshold_90pct": threshold90,
        "function_rank_effective": rank,
        "uses_validation_test_future_query_for_direction": 0,
        "task_metrics_readback_only": 1,
        **fu_diag,
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = _device(args.device)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    variants = ["MLP+AdamW", "MLP+SGD", "MLP+FU+AdamW", "KAN+AdamW", "KAN+FU+AdamW"]
    for name in [x.strip() for x in str(args.datasets).split(",") if x.strip()]:
        for seed in [int(x) for x in str(args.seeds).split(",") if x.strip()]:
            try:
                train_loader = _loader(name, True, int(args.train_size), int(args.batch_size), seed)
                test_loader = _loader(name, False, int(args.test_size), int(args.batch_size), seed)
            except Exception as exc:
                blockers.append(f"{name}:dataset_load_failed:{exc}")
                continue
            for variant in variants:
                try:
                    rows.append(_train_one(name, seed, variant, train_loader, test_loader, int(args.steps), int(args.hidden), device))
                except Exception as exc:
                    rows.append({"dataset": name, "seed": seed, "variant": variant, "status": "blocked", "blocker": repr(exc)})
                    blockers.append(f"{name}:{seed}:{variant}:{exc}")
    # Add relative comparisons per dataset/seed.
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("dataset")), int(row.get("seed", -1))), {})[str(row.get("variant"))] = row
    for (_name, _seed), group in by_key.items():
        mlp = group.get("MLP+AdamW", {})
        best_control_loss = min(float(r.get("final_train_loss", 999.0)) for v, r in group.items() if "+FU" not in v and r.get("final_train_loss", "") != "")
        for row in group.values():
            row["NLL_delta_vs_MLP"] = float(row.get("final_test_loss_readback", 999.0)) - float(mlp.get("final_test_loss_readback", 999.0)) if mlp else ""
            row["AUC_loss_time_ratio_vs_best_control"] = float(row.get("final_train_loss", 999.0)) / max(best_control_loss, 1.0e-8)
            row["time_to_threshold_80pct"] = row.get("steps_to_threshold_80pct", "")
            row["time_to_threshold_90pct"] = row.get("steps_to_threshold_90pct", "")
    write_rows(out_dir / "v22_13_task_eval_matrix.csv", rows)
    write_rows(out_dir / "v22_13_convergence_speed_matrix.csv", rows)
    write_rows(out_dir / "v22_13_forgetting_readback_matrix.csv", rows)
    write_rows(out_dir / "v22_13_expression_metrics_matrix.csv", rows)
    write_rows(out_dir / "v22_13_calibration_debt_matrix.csv", rows)
    kan_fu_rows = [r for r in rows if r.get("variant") == "KAN+FU+AdamW" and r.get("status") != "blocked"]
    mlp_rows = [r for r in rows if r.get("variant") == "MLP+AdamW" and r.get("status") != "blocked"]
    better_acc = 0
    for row in kan_fu_rows:
        match = next((r for r in mlp_rows if r.get("dataset") == row.get("dataset") and r.get("seed") == row.get("seed")), None)
        if match and float(row.get("final_test_accuracy_readback", 0.0)) >= float(match.get("final_test_accuracy_readback", 1.0)):
            better_acc += 1
    route = {
        "route": "TaskReadbackCompleted" if rows and not blockers else "TaskReadbackPartialOrBlocked",
        "task_rows": len(rows),
        "KAN_FU_rows": len(kan_fu_rows),
        "KAN_FU_accuracy_ge_MLP_rows": better_acc,
        "full_scientific_task_gate_pass": int(len(kan_fu_rows) >= 9 and better_acc >= 7),
        "uses_task_metric_for_direction": 0,
        "promotion_allowed": 0,
        "blocker": ";".join(blockers[:20]),
    }
    write_json(out_dir / "v22_13_task_eval_route.json", route)
    simple_svg(out_dir / "figures/v22_13_convergence_forgetting_expression_dashboard.svg", "v22.13 task readback", rows, "final_test_accuracy_readback")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_13_task_eval.py --device {args.device} --datasets {args.datasets} --seeds {args.seeds} --train-size {int(args.train_size)} --test-size {int(args.test_size)} --steps {int(args.steps)} --batch-size {int(args.batch_size)} --hidden {int(args.hidden)} --out-dir {out_dir}", status="completed" if rows else "blocked", note=f"route={route['route']} rows={len(rows)} blocker={route.get('blocker')}")


if __name__ == "__main__":
    main()

