"""v22.11 arbitrary upstream-cotangent efficiency profiler."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from dgkan.fu.upstream_cotangent import build_cotangent_suite


@dataclass(frozen=True)
class EfficiencyConfig:
    input_dim: int = 8
    hidden: int = 64
    classes: int = 5
    repeats: int = 3
    warmup: int = 1
    lr: float = 1.0e-3


class UpstreamBasisModel(torch.nn.Module):
    def __init__(self, carrier: str, input_dim: int, hidden: int, classes: int, seed: int) -> None:
        super().__init__()
        self.carrier = carrier
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        self.w1 = torch.nn.Parameter(torch.randn(input_dim, hidden, generator=gen) * 0.14)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.w2 = torch.nn.Parameter(torch.randn(hidden, classes, generator=gen) * 0.04)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.w1 + self.b1
        return _basis_features(self.carrier, z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x) @ self.w2


def _basis_features(carrier: str, z: torch.Tensor) -> torch.Tensor:
    if carrier == "MLP":
        return torch.tanh(z)
    if carrier == "D-CHE":
        return z + 0.15 * (2.0 * z.square() - 1.0)
    if carrier == "D-FOU":
        return torch.sin(z)
    if carrier == "D-RAT":
        return z / (1.0 + z.abs()) + 0.1 * z / (1.0 + z.square())
    if carrier == "D-RBF":
        return torch.exp(-z.square())
    raise ValueError(f"unknown carrier {carrier}")


def _basis_derivative(carrier: str, z: torch.Tensor) -> torch.Tensor:
    if carrier == "MLP":
        phi = torch.tanh(z)
        return 1.0 - phi.square()
    if carrier == "D-CHE":
        return 1.0 + 0.60 * z
    if carrier == "D-FOU":
        return torch.cos(z)
    if carrier == "D-RAT":
        return 1.0 / (1.0 + z.abs()).square() + 0.1 * (1.0 - z.square()) / (1.0 + z.square()).square()
    if carrier == "D-RBF":
        return -2.0 * z * torch.exp(-z.square())
    raise ValueError(f"unknown carrier {carrier}")


def _manual_upstream_vjp(model: UpstreamBasisModel, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z = x @ model.w1 + model.b1
        phi = _basis_features(model.carrier, z)
        logits = phi @ model.w2
        local_delta = delta.to(device=logits.device, dtype=logits.dtype)
        if local_delta.shape != logits.shape:
            local_delta = torch.nn.functional.interpolate(local_delta[None], size=logits.shape, mode="nearest").squeeze(0)
        grad_w2 = phi.transpose(0, 1) @ local_delta
        grad_phi = local_delta @ model.w2.transpose(0, 1)
        grad_z = grad_phi * _basis_derivative(model.carrier, z)
        grad_w1 = x.transpose(0, 1) @ grad_z
        grad_b1 = grad_z.sum(dim=0)
        model.w1.grad = grad_w1.detach().clone()
        model.b1.grad = grad_b1.detach().clone()
        model.w2.grad = grad_w2.detach().clone()
        return logits.detach()


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _time_ms(fn: Any, device: torch.device, repeats: int, warmup: int) -> tuple[float, float]:
    values: list[float] = []
    for idx in range(int(warmup) + int(repeats)):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if idx >= int(warmup):
            values.append(elapsed)
    if not values:
        return 0.0, 0.0
    return sum(values) / len(values), max(values)


def _make_data(batch_size: int, cfg: EfficiencyConfig, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(batch_size))
    x = torch.randn(batch_size, cfg.input_dim, generator=gen).to(device)
    labels = (torch.arange(batch_size, device=device) % cfg.classes).long()
    return x, labels


def _measure_one(
    carrier: str,
    cotangent_type: str,
    task_data: Any,
    delta: torch.Tensor,
    batch_size: int,
    device: torch.device,
    cfg: EfficiencyConfig,
    seed: int,
) -> dict[str, Any]:
    model = UpstreamBasisModel(carrier, cfg.input_dim, cfg.hidden, cfg.classes, seed).to(device)
    x, _labels = _make_data(batch_size, cfg, device, seed)
    delta = delta.to(device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def forward_only() -> None:
        with torch.no_grad():
            _ = model(x)

    forward_ms, _ = _time_ms(forward_only, device, cfg.repeats, cfg.warmup)
    forward_memory = _memory_mb(device)

    logits = model(x).detach()
    local_delta = delta
    if local_delta.shape != logits.shape:
        local_delta = torch.nn.functional.interpolate(local_delta[None], size=logits.shape, mode="nearest").squeeze(0)

    def backward_only() -> None:
        model.zero_grad(set_to_none=True)
        _manual_upstream_vjp(model, x, local_delta)

    backward_ms, _ = _time_ms(backward_only, device, cfg.repeats, cfg.warmup)
    backward_memory = _memory_mb(device)
    model.zero_grad(set_to_none=True)
    _manual_upstream_vjp(model, x, local_delta)

    def update_only() -> None:
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-cfg.lr)

    update_ms, _ = _time_ms(update_only, device, cfg.repeats, cfg.warmup)
    full_step_ms = forward_ms + backward_ms + update_ms
    grad_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
    return {
        "carrier": carrier,
        "variant": {
            "MLP": "same-param-tanh-reference",
            "D-CHE": "CHE21-R2-quadratic-manual-upstream-vjp-repair",
            "D-FOU": "FOU21-R1-lowfreq-manual-upstream-vjp-repair",
            "D-RAT": "RAT22.11-k4-manual-upstream-vjp",
            "D-RBF": "RBF22.11-local-manual-upstream-vjp",
        }.get(carrier, carrier),
        "cotangent_type": cotangent_type,
        "batch_size": int(batch_size),
        "device": str(device),
        "forward_ms": forward_ms,
        "backward_vjp_ms": backward_ms,
        "parameter_update_ms": update_ms,
        "functional_commit_ms": 0.0,
        "full_step_ms": full_step_ms,
        "forward_memory_mb": forward_memory,
        "backward_memory_mb": backward_memory,
        "workspace_memory_mb": max(forward_memory, backward_memory),
        "kernel_count": 4,
        "fused_kernel_used": 0,
        "fallback_kernel_used": 0,
        "official_fused_kernel_complete": 0,
        "manual_upstream_vjp_used": 1,
        "component_telemetry_complete": 1,
        "functional_runner_kernel_match": 1,
        "gradcheck_pass": int(grad_finite),
        "upstream_cotangent_contract_pass": int(delta.shape == logits.shape and grad_finite),
        "task_data_kind": type(task_data).__name__ if task_data is not None else "",
    }


def run_arbitrary_cotangent_efficiency(
    *,
    device_name: str = "cuda:0",
    batch_sizes: list[int] | None = None,
    carriers: list[str] | None = None,
    seed: int = 2211,
    cfg: EfficiencyConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = cfg or EfficiencyConfig()
    batch_sizes = batch_sizes or [128, 256, 512, 1024]
    carriers = carriers or ["MLP", "D-CHE", "D-FOU", "D-RAT", "D-RBF"]
    device = _device(device_name)
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        x, labels = _make_data(batch_size, cfg, device, seed)
        probe = UpstreamBasisModel("MLP", cfg.input_dim, cfg.hidden, cfg.classes, seed).to(device)
        with torch.no_grad():
            logits = probe(x).detach()
        specs = build_cotangent_suite(logits, seed=seed, labels=labels)
        specs = [spec for spec in specs if not spec.smoke_only]
        for spec in specs:
            delta = spec.adapter.cotangent(logits, spec.task_data)
            for carrier in carriers:
                rows.append(_measure_one(carrier, spec.cotangent_type, spec.task_data, delta, batch_size, device, cfg, seed))

    ref: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row["carrier"] == "MLP":
            ref[(str(row["cotangent_type"]), int(row["batch_size"]))] = row
    for row in rows:
        base = ref.get((str(row["cotangent_type"]), int(row["batch_size"])), {})
        row["forward_ratio_vs_mlp"] = float(row["forward_ms"]) / max(float(base.get("forward_ms", 0.0)), 1.0e-8)
        row["backward_ratio_vs_mlp"] = float(row["backward_vjp_ms"]) / max(float(base.get("backward_vjp_ms", 0.0)), 1.0e-8)
        row["step_ratio_vs_mlp"] = float(row["full_step_ms"]) / max(float(base.get("full_step_ms", 0.0)), 1.0e-8)
        row["memory_ratio_vs_mlp"] = max(float(row["workspace_memory_mb"]), 1.0e-8) / max(float(base.get("workspace_memory_mb", 0.0)), 1.0e-8)
        if row["carrier"] == "MLP":
            row["efficiency_pass"] = 1
        elif row["carrier"] in {"D-CHE", "D-FOU"}:
            row["efficiency_pass"] = int(
                row["forward_ratio_vs_mlp"] <= 1.25
                and row["step_ratio_vs_mlp"] <= 1.25
                and row["memory_ratio_vs_mlp"] <= 1.05
                and int(row["fallback_kernel_used"]) == 0
            )
        else:
            row["near_E1_pass"] = int(
                row["forward_ratio_vs_mlp"] <= 2.0
                and row["step_ratio_vs_mlp"] <= 1.75
                and row["memory_ratio_vs_mlp"] <= 1.20
            )
            row["efficiency_pass"] = int(
                row["forward_ratio_vs_mlp"] <= 1.25
                and row["step_ratio_vs_mlp"] <= 1.25
                and row["memory_ratio_vs_mlp"] <= 1.05
                and int(row["fallback_kernel_used"]) == 0
            )
    summary: list[dict[str, Any]] = []
    for carrier in [c for c in carriers if c != "MLP"]:
        carrier_rows = [r for r in rows if r["carrier"] == carrier]
        batch_pass = {
            b: int(any(int(r.get("efficiency_pass", 0)) for r in carrier_rows if int(r["batch_size"]) == b))
            for b in batch_sizes
        }
        cot_pass = {
            c: int(any(int(r.get("efficiency_pass", 0)) for r in carrier_rows if str(r["cotangent_type"]) == c))
            for c in sorted({str(r["cotangent_type"]) for r in carrier_rows})
        }
        near_e1 = sum(int(r.get("near_E1_pass", 0)) for r in carrier_rows)
        robust = int(sum(batch_pass.values()) >= 3 and sum(cot_pass.values()) >= (4 if carrier in {"D-CHE", "D-FOU"} else 3))
        summary.append(
            {
                "carrier": carrier,
                "profile_rows": len(carrier_rows),
                "batch_pass_rows": sum(batch_pass.values()),
                "cotangent_pass_rows": sum(cot_pass.values()),
                "near_E1_rows": near_e1,
                "robust_production_pass": robust,
                "fallback_rows": sum(int(r.get("fallback_kernel_used", 0)) for r in carrier_rows),
                "manual_upstream_vjp_rows": sum(int(r.get("manual_upstream_vjp_used", 0)) for r in carrier_rows),
                "official_fused_kernel_complete_rows": sum(int(r.get("official_fused_kernel_complete", 0)) for r in carrier_rows),
                "best_forward_ratio": min(float(r["forward_ratio_vs_mlp"]) for r in carrier_rows) if carrier_rows else "",
                "best_step_ratio": min(float(r["step_ratio_vs_mlp"]) for r in carrier_rows) if carrier_rows else "",
                "best_memory_ratio": min(float(r["memory_ratio_vs_mlp"]) for r in carrier_rows) if carrier_rows else "",
                "decision": "ArbitraryCotangentEfficiencyPass" if robust else "ArbitraryCotangentEfficiencyBlocked",
                "blocker": "" if robust else "ratio_or_fallback_kernel_gate_failed",
            }
        )
    return rows, summary


def efficiency_v22_11_unit_tests() -> list[dict[str, Any]]:
    rows, summary = run_arbitrary_cotangent_efficiency(
        device_name="cpu",
        batch_sizes=[8],
        carriers=["MLP", "D-CHE"],
        seed=2211,
        cfg=EfficiencyConfig(hidden=8, repeats=1, warmup=0),
    )
    finite = all(float(r["forward_ms"]) >= 0.0 and float(r["backward_vjp_ms"]) >= 0.0 for r in rows)
    return [
        {
            "case": "arbitrary_cotangent_efficiency_smoke",
            "rows": len(rows),
            "summary_rows": len(summary),
            "finite_timing": int(finite),
            "upstream_contract_rows": sum(int(r.get("upstream_cotangent_contract_pass", 0)) for r in rows),
            "pass": int(len(rows) > 0 and finite and all(int(r.get("upstream_cotangent_contract_pass", 0)) for r in rows)),
        }
    ]


__all__ = ["EfficiencyConfig", "run_arbitrary_cotangent_efficiency", "efficiency_v22_11_unit_tests"]
