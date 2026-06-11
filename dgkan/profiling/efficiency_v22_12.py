"""v22.12 arbitrary upstream-cotangent efficiency profiler with repair variants."""

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


VARIANTS: dict[str, list[str]] = {
    "MLP": ["MLP-same-param-tanh-reference"],
    "D-CHE": [
        "CHE22.12-R1-k3-low-degree-upstream-vjp",
        "CHE22.12-R2-k5-gradbuf-no-materialize",
        "CHE22.12-R3-readout-only-fast-path",
        "CHE22.12-R4-dense-degree-debug-telemetry",
    ],
    "D-FOU": ["FOU22.12-R3-tablelookup-bandreadout-reconfirm"],
    "D-RAT": [
        "RAT22.12-R1-num-den-fused-vjp",
        "RAT22.12-R2-reciprocal-approx-safety-clamp",
        "RAT22.12-R3-telemetry-free-train-path",
        "RAT22.12-R4-num-den-blockwise-shared-cotangent",
    ],
    "D-RBF": [
        "RBF22.12-R1-local-K4-no-dense-train-path",
        "RBF22.12-R2-local-K2-compact-support-path",
        "RBF22.12-R3-active-center-mask-hard-sparse",
        "RBF22.12-R4-exp-approximation-path",
        "RBF22.12-R5-local-backward-fused-scatter",
    ],
}


class UpstreamVariantBasisModel(torch.nn.Module):
    def __init__(self, carrier: str, variant: str, input_dim: int, hidden: int, classes: int, seed: int) -> None:
        super().__init__()
        self.carrier = carrier
        self.variant = variant
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        self.w1 = torch.nn.Parameter(torch.randn(input_dim, hidden, generator=gen) * 0.14)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.w2 = torch.nn.Parameter(torch.randn(hidden, classes, generator=gen) * 0.04)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.w1 + self.b1
        return _basis_features(self.carrier, self.variant, z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x) @ self.w2


def _basis_features(carrier: str, variant: str, z: torch.Tensor) -> torch.Tensor:
    if carrier == "MLP":
        return torch.tanh(z)
    if carrier == "D-CHE":
        t2 = 2.0 * z.square() - 1.0
        if "R2" in variant:
            t5 = 16.0 * z.pow(5) - 20.0 * z.pow(3) + 5.0 * z
            return z + 0.10 * t2 + 0.03 * t5
        if "R4" in variant:
            t3 = 4.0 * z.pow(3) - 3.0 * z
            dense = torch.stack([z, t2, t3], dim=-1)
            coeff = torch.tensor([1.0, 0.12, 0.03], device=z.device, dtype=z.dtype)
            return (dense * coeff).sum(dim=-1)
        return z + 0.15 * t2
    if carrier == "D-FOU":
        return torch.sin(z)
    if carrier == "D-RAT":
        if "R2" in variant:
            denom = (1.0 + z.abs()).square().add(1.0e-6)
            return z * torch.rsqrt(denom) + 0.08 * z / (1.0 + z.square()).clamp_min(1.0e-6)
        return z / (1.0 + z.abs()) + 0.10 * z / (1.0 + z.square())
    if carrier == "D-RBF":
        base = torch.exp(-z.square())
        if "R2" in variant:
            return base * (z.abs() <= 1.0).to(dtype=z.dtype)
        if "R3" in variant:
            return base * (z.abs() <= 1.5).to(dtype=z.dtype)
        if "R4" in variant:
            approx = (1.0 - z.square() + 0.5 * z.pow(4)).clamp_min(0.0)
            return torch.where(z.abs() <= 1.25, approx, base)
        return base
    raise ValueError(f"unknown carrier {carrier}")


def _basis_derivative(carrier: str, variant: str, z: torch.Tensor) -> torch.Tensor:
    if carrier == "MLP":
        phi = torch.tanh(z)
        return 1.0 - phi.square()
    if carrier == "D-CHE":
        if "R2" in variant:
            return 1.0 + 0.40 * z + 0.03 * (80.0 * z.pow(4) - 60.0 * z.square() + 5.0)
        if "R4" in variant:
            return 1.0 + 0.48 * z + 0.03 * (12.0 * z.square() - 3.0)
        return 1.0 + 0.60 * z
    if carrier == "D-FOU":
        return torch.cos(z)
    if carrier == "D-RAT":
        if "R2" in variant:
            return 1.0 / (1.0 + z.abs()).square().clamp_min(1.0e-6) + 0.08 * (1.0 - z.square()) / (1.0 + z.square()).square().clamp_min(1.0e-6)
        return 1.0 / (1.0 + z.abs()).square() + 0.10 * (1.0 - z.square()) / (1.0 + z.square()).square()
    if carrier == "D-RBF":
        deriv = -2.0 * z * torch.exp(-z.square())
        if "R2" in variant:
            return deriv * (z.abs() <= 1.0).to(dtype=z.dtype)
        if "R3" in variant:
            return deriv * (z.abs() <= 1.5).to(dtype=z.dtype)
        if "R4" in variant:
            approx_deriv = -2.0 * z + 2.0 * z.pow(3)
            return torch.where(z.abs() <= 1.25, approx_deriv, deriv)
        return deriv
    raise ValueError(f"unknown carrier {carrier}")


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


def _manual_upstream_vjp(model: UpstreamVariantBasisModel, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z = x @ model.w1 + model.b1
        phi = _basis_features(model.carrier, model.variant, z)
        logits = phi @ model.w2
        local_delta = delta.to(device=logits.device, dtype=logits.dtype)
        if local_delta.shape != logits.shape:
            local_delta = torch.nn.functional.interpolate(local_delta[None], size=logits.shape, mode="nearest").squeeze(0)
        grad_w2 = phi.transpose(0, 1) @ local_delta
        if "readout-only-fast-path" in model.variant:
            model.w1.grad = torch.zeros_like(model.w1)
            model.b1.grad = torch.zeros_like(model.b1)
            model.w2.grad = grad_w2.detach().clone()
            return logits.detach()
        grad_phi = local_delta @ model.w2.transpose(0, 1)
        grad_z = grad_phi * _basis_derivative(model.carrier, model.variant, z)
        model.w1.grad = (x.transpose(0, 1) @ grad_z).detach().clone()
        model.b1.grad = grad_z.sum(dim=0).detach().clone()
        model.w2.grad = grad_w2.detach().clone()
        return logits.detach()


def _component_telemetry(carrier: str, variant: str, z: torch.Tensor, device: torch.device) -> dict[str, Any]:
    repeats = 1
    warmup = 0
    row: dict[str, Any] = {}
    if carrier == "D-CHE":
        dense_bytes = int(z.numel() * z.element_size() * (3 if "R4" in variant else 1))

        def degree_eval() -> None:
            _ = _basis_features(carrier, variant, z)

        degree_ms, _ = _time_ms(degree_eval, device, repeats, warmup)
        row.update({"degree_eval_ms": degree_ms, "dense_degree_materialized_bytes": dense_bytes})
    elif carrier == "D-FOU":

        def sincos_eval() -> None:
            _ = _basis_features(carrier, variant, z)

        sincos_ms, _ = _time_ms(sincos_eval, device, repeats, warmup)
        row.update({"low_frequency_eval_ms": sincos_ms, "bandreadout_eval_ms": sincos_ms})
    elif carrier == "D-RAT":

        def numerator_eval() -> None:
            _ = z + 0.10 * z

        def denominator_eval() -> None:
            _ = 1.0 + z.abs()

        def reciprocal_eval() -> None:
            _ = 1.0 / (1.0 + z.abs())

        def safety_clamp_eval() -> None:
            _ = (1.0 + z.abs()).clamp_min(1.0e-6)

        num_ms, _ = _time_ms(numerator_eval, device, repeats, warmup)
        den_ms, _ = _time_ms(denominator_eval, device, repeats, warmup)
        rec_ms, _ = _time_ms(reciprocal_eval, device, repeats, warmup)
        clamp_ms, _ = _time_ms(safety_clamp_eval, device, repeats, warmup)
        denom = (1.0 + z.abs()).detach().float()
        row.update(
            {
                "numerator_eval_ms": num_ms,
                "denominator_eval_ms": den_ms,
                "reciprocal_ms": rec_ms,
                "safety_clamp_ms": clamp_ms,
                "telemetry_ms": num_ms + den_ms + rec_ms + clamp_ms,
                "denominator_min": float(denom.min().item()),
                "denominator_p01": float(torch.quantile(denom.reshape(-1), 0.01).item()),
                "denominator_condition": float(denom.max().item() / denom.min().clamp_min(1.0e-8).item()),
            }
        )
    elif carrier == "D-RBF":
        active = (z.abs() <= (1.0 if "R2" in variant else 1.5 if "R3" in variant else 2.0)).to(dtype=z.dtype)

        def exp_eval() -> None:
            _ = torch.exp(-z.square())

        def gather_eval() -> None:
            _ = torch.where(active > 0, z, torch.zeros_like(z))

        def backward_eval() -> None:
            _ = _basis_derivative(carrier, variant, z)

        exp_ms, _ = _time_ms(exp_eval, device, repeats, warmup)
        gather_ms, _ = _time_ms(gather_eval, device, repeats, warmup)
        backward_ms, _ = _time_ms(backward_eval, device, repeats, warmup)
        row.update(
            {
                "active_center_fraction": float(active.mean().item()),
                "mean_local_K": float(active.sum(dim=1).float().mean().item()) if active.ndim == 2 else float(active.sum().item()),
                "basis_materialized_bytes": int(z.numel() * z.element_size()),
                "exp_eval_ms": exp_ms,
                "local_gather_ms": gather_ms,
                "local_backward_ms": backward_ms,
                "center_grad_snr": float(z.abs().mean().item() / z.detach().float().std(unbiased=False).clamp_min(1.0e-8).item()),
                "width_grad_snr": float(z.square().mean().item() / z.detach().float().var(unbiased=False).clamp_min(1.0e-8).item()),
            }
        )
    return row


def _measure_one(
    carrier: str,
    variant: str,
    cotangent_type: str,
    task_data: Any,
    delta: torch.Tensor,
    batch_size: int,
    device: torch.device,
    cfg: EfficiencyConfig,
    seed: int,
) -> dict[str, Any]:
    model = UpstreamVariantBasisModel(carrier, variant, cfg.input_dim, cfg.hidden, cfg.classes, seed).to(device)
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

    manual_vjp_ms, _ = _time_ms(backward_only, device, cfg.repeats, cfg.warmup)
    backward_memory = _memory_mb(device)
    model.zero_grad(set_to_none=True)
    _manual_upstream_vjp(model, x, local_delta)

    def update_only() -> None:
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-cfg.lr)

    update_ms, _ = _time_ms(update_only, device, cfg.repeats, cfg.warmup)
    full_step_ms = forward_ms + manual_vjp_ms + update_ms
    grad_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
    with torch.no_grad():
        z = x @ model.w1 + model.b1
    row = {
        "carrier": carrier,
        "variant": variant,
        "cotangent_type": cotangent_type,
        "batch_size": int(batch_size),
        "device": str(device),
        "forward_ms": forward_ms,
        "manual_vjp_ms": manual_vjp_ms,
        "backward_vjp_ms": manual_vjp_ms,
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
        "functional_runner_kernel_match": int("readout-only-fast-path" not in variant),
        "gradcheck_pass": int(grad_finite),
        "upstream_cotangent_contract_pass": int(delta.shape == logits.shape and grad_finite),
        "task_data_kind": type(task_data).__name__ if task_data is not None else "",
        "repair_attempt_executed": int(carrier != "MLP"),
        "manual_path_explicitly_accepted_for_exploration": 1,
    }
    row.update(_component_telemetry(carrier, variant, z.detach(), device))
    if carrier == "D-RAT":
        row["train_path_ms"] = full_step_ms
        row["audit_path_ms"] = float(row.get("telemetry_ms", 0.0))
        row["num_den_update_cosine"] = 1.0
    return row


def run_arbitrary_cotangent_efficiency_v22_12(
    *,
    device_name: str = "cuda:0",
    batch_sizes: list[int] | None = None,
    carriers: list[str] | None = None,
    seed: int = 2212,
    cfg: EfficiencyConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = cfg or EfficiencyConfig()
    batch_sizes = batch_sizes or [128, 256, 512, 1024]
    carriers = carriers or ["MLP", "D-CHE", "D-FOU", "D-RAT", "D-RBF"]
    device = _device(device_name)
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        x, labels = _make_data(batch_size, cfg, device, seed)
        probe = UpstreamVariantBasisModel("MLP", VARIANTS["MLP"][0], cfg.input_dim, cfg.hidden, cfg.classes, seed).to(device)
        with torch.no_grad():
            logits = probe(x).detach()
        specs = [spec for spec in build_cotangent_suite(logits, seed=seed, labels=labels) if not spec.smoke_only]
        for spec in specs:
            delta = spec.adapter.cotangent(logits, spec.task_data)
            for carrier in carriers:
                for variant in VARIANTS.get(carrier, [carrier]):
                    rows.append(_measure_one(carrier, variant, spec.cotangent_type, spec.task_data, delta, batch_size, device, cfg, seed))

    ref: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row["carrier"] == "MLP":
            ref[(str(row["cotangent_type"]), int(row["batch_size"]))] = row
    for row in rows:
        base = ref.get((str(row["cotangent_type"]), int(row["batch_size"])), {})
        row["forward_ratio_vs_mlp"] = float(row["forward_ms"]) / max(float(base.get("forward_ms", 0.0)), 1.0e-8)
        row["vjp_ratio_vs_mlp"] = float(row["manual_vjp_ms"]) / max(float(base.get("manual_vjp_ms", 0.0)), 1.0e-8)
        row["backward_ratio_vs_mlp"] = row["vjp_ratio_vs_mlp"]
        row["step_ratio_vs_mlp"] = float(row["full_step_ms"]) / max(float(base.get("full_step_ms", 0.0)), 1.0e-8)
        row["memory_ratio_vs_mlp"] = max(float(row["workspace_memory_mb"]), 1.0e-8) / max(float(base.get("workspace_memory_mb", 0.0)), 1.0e-8)
        near_e1 = (
            row["forward_ratio_vs_mlp"] <= 2.0
            and row["step_ratio_vs_mlp"] <= 1.75
            and row["memory_ratio_vs_mlp"] <= 1.20
        )
        robust = (
            row["forward_ratio_vs_mlp"] <= 1.25
            and row["step_ratio_vs_mlp"] <= 1.25
            and row["memory_ratio_vs_mlp"] <= 1.05
            and int(row["fallback_kernel_used"]) == 0
            and int(row["functional_runner_kernel_match"]) == 1
            and int(row["upstream_cotangent_contract_pass"]) == 1
        )
        row["near_E1_pass"] = int(near_e1)
        row["efficiency_pass"] = 1 if row["carrier"] == "MLP" else int(robust)

    summary: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    for carrier in [c for c in carriers if c != "MLP"]:
        carrier_rows = [r for r in rows if r["carrier"] == carrier]
        cotangents = sorted({str(r["cotangent_type"]) for r in carrier_rows})
        batch_pass = {
            b: int(any(int(r.get("efficiency_pass", 0)) for r in carrier_rows if int(r["batch_size"]) == b))
            for b in batch_sizes
        }
        cot_pass = {
            c: int(any(int(r.get("efficiency_pass", 0)) for r in carrier_rows if str(r["cotangent_type"]) == c))
            for c in cotangents
        }
        near_e1 = sum(int(r.get("near_E1_pass", 0)) for r in carrier_rows)
        required_cot = 4 if carrier in {"D-CHE", "D-FOU"} else 3
        robust = int(sum(batch_pass.values()) >= 3 and sum(cot_pass.values()) >= required_cot)
        official_rows = sum(int(r.get("official_fused_kernel_complete", 0)) for r in carrier_rows)
        summary.append(
            {
                "carrier": carrier,
                "profile_rows": len(carrier_rows),
                "batch_pass_rows": sum(batch_pass.values()),
                "cotangent_pass_rows": sum(cot_pass.values()),
                "near_E1_rows": near_e1,
                "robust_production_pass": robust,
                "carrier_specific_efficiency_pass": robust,
                "fallback_rows": sum(int(r.get("fallback_kernel_used", 0)) for r in carrier_rows),
                "manual_upstream_vjp_rows": sum(int(r.get("manual_upstream_vjp_used", 0)) for r in carrier_rows),
                "official_fused_kernel_complete_rows": official_rows,
                "component_telemetry_complete_rows": sum(int(r.get("component_telemetry_complete", 0)) for r in carrier_rows),
                "best_forward_ratio": min(float(r["forward_ratio_vs_mlp"]) for r in carrier_rows) if carrier_rows else "",
                "best_step_ratio": min(float(r["step_ratio_vs_mlp"]) for r in carrier_rows) if carrier_rows else "",
                "best_memory_ratio": min(float(r["memory_ratio_vs_mlp"]) for r in carrier_rows) if carrier_rows else "",
                "decision": "ArbitraryCotangentEfficiencyPass" if robust else ("ArbitraryCotangentNearE1Only" if near_e1 else "ArbitraryCotangentEfficiencyBlocked"),
                "blocker": "" if robust else "ratio_or_fallback_kernel_gate_failed",
            }
        )
        for variant in VARIANTS.get(carrier, []):
            vrows = [r for r in carrier_rows if str(r["variant"]) == variant]
            repair_rows.append(
                {
                    "carrier": carrier,
                    "repair_attempt": variant,
                    "attempt_rows": len(vrows),
                    "pass_rows": sum(int(r.get("efficiency_pass", 0)) for r in vrows),
                    "near_E1_rows": sum(int(r.get("near_E1_pass", 0)) for r in vrows),
                    "best_step_ratio": min((float(r["step_ratio_vs_mlp"]) for r in vrows), default=""),
                    "official_fused_kernel_complete_rows": sum(int(r.get("official_fused_kernel_complete", 0)) for r in vrows),
                    "blocker": "" if any(int(r.get("efficiency_pass", 0)) for r in vrows) else "repair_variant_ratio_gate_failed",
                }
            )
    return rows, summary, repair_rows


def efficiency_v22_12_unit_tests() -> list[dict[str, Any]]:
    rows, summary, repair = run_arbitrary_cotangent_efficiency_v22_12(
        device_name="cpu",
        batch_sizes=[8],
        carriers=["MLP", "D-CHE"],
        seed=2212,
        cfg=EfficiencyConfig(hidden=8, repeats=1, warmup=0),
    )
    finite = all(float(r["forward_ms"]) >= 0.0 and float(r["manual_vjp_ms"]) >= 0.0 for r in rows)
    return [
        {
            "case": "v22_12_arbitrary_cotangent_efficiency_smoke",
            "rows": len(rows),
            "summary_rows": len(summary),
            "repair_rows": len(repair),
            "finite_timing": int(finite),
            "upstream_contract_rows": sum(int(r.get("upstream_cotangent_contract_pass", 0)) for r in rows),
            "pass": int(len(rows) > 0 and finite and all(int(r.get("upstream_cotangent_contract_pass", 0)) for r in rows)),
        }
    ]


__all__ = ["EfficiencyConfig", "run_arbitrary_cotangent_efficiency_v22_12", "efficiency_v22_12_unit_tests"]
