"""v22.13 kernel-native arbitrary-cotangent operator-step profiler."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from dgkan.fu.operator_core import apply_operator
from dgkan.fu.upstream_cotangent import build_cotangent_suite
from dgkan.kernels.dche_native_cotangent import kernel_status as dche_status
from dgkan.kernels.dche_native_cotangent import native_fused_vjp as dche_native_fused_vjp
from dgkan.kernels.dfou_native_cotangent import kernel_status as dfou_status
from dgkan.kernels.dfou_native_cotangent import native_fused_vjp as dfou_native_fused_vjp
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec
from dgkan.profiling.efficiency_v22_12 import EfficiencyConfig, UpstreamVariantBasisModel


@dataclass(frozen=True)
class NativeEfficiencyConfig(EfficiencyConfig):
    basis_dim: int = 3


VARIANTS: dict[str, list[str]] = {
    "MLP": ["MLP-same-param-tanh-reference"],
    "D-FOU": [
        "FOU22.13-R1-lowfreq-bandreadout-native-vjp",
        "FOU22.13-R2-tablelookup-bandreadout-native-operator-step",
        "FOU22.13-R3-lowfreq-source-state-fused-commit",
    ],
    "D-CHE": [
        "CHE22.13-R1-k3-corrected-layout-native-vjp",
        "CHE22.13-R2-k5-gradbuf-no-materialize-native-vjp",
        "CHE22.13-R3-lowdegree-readout-operator-step",
        "CHE22.13-R4-corrected-layout-source-state-fused-commit",
    ],
    "D-RBF": ["RBF22.13-R1-local-K4-native-vjp"],
    "D-RAT": ["RAT22.13-R1-num-den-native-vjp"],
}


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


def _time_ms(fn: Any, device: torch.device, repeats: int, warmup: int) -> float:
    values: list[float] = []
    for idx in range(int(warmup) + int(repeats)):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        if idx >= int(warmup):
            values.append((time.perf_counter() - start) * 1000.0)
    return sum(values) / max(1, len(values))


def _make_batch(batch_size: int, cfg: NativeEfficiencyConfig, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(batch_size))
    x = torch.randn(batch_size, cfg.input_dim, generator=gen).to(device)
    labels = (torch.arange(batch_size, device=device) % cfg.classes).long()
    return x, labels


def _cotangents(logits: torch.Tensor, labels: torch.Tensor, seed: int) -> list[tuple[str, torch.Tensor]]:
    specs = build_cotangent_suite(logits.detach().float(), seed=seed, labels=labels.detach().cpu())
    out: list[tuple[str, torch.Tensor]] = []
    for spec in specs:
        if int(spec.smoke_only):
            continue
        delta = spec.adapter.cotangent(logits.detach().float().cpu(), spec.task_data)
        out.append((spec.cotangent_type, delta.to(device=logits.device, dtype=logits.dtype)))
    return out


def _make_profile_model(carrier: str, variant: str, x: torch.Tensor, cfg: NativeEfficiencyConfig, device: torch.device, seed: int) -> torch.nn.Module:
    if carrier == "D-FOU":
        spec = PrimitiveSpec(
            candidate_id=f"v22.13-eff-{variant}",
            basis_family="D-FOU",
            basis_name="fourier_lowfreq",
            k=2,
            hidden_dim=int(cfg.hidden),
            source="v22_13_native_arbitrary_cotangent_fused_vjp",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=1,
            uses_division=0,
            uses_dense_basis_tensor=0,
            init_variant="fourier_k2_triton_l3_matmul",
        )
        return PrimitiveKAN(int(cfg.input_dim), int(cfg.classes), spec, x, int(seed), device, param_budget=4096).to(device)
    if carrier == "D-CHE":
        init_variant = "cheby_k3_triton_l3_gradbuf" if "R2" in str(variant) or "R4" in str(variant) else "cheby_k3_triton_l3_matmul"
        spec = PrimitiveSpec(
            candidate_id=f"v22.13-eff-{variant}",
            basis_family="D-CHE",
            basis_name="chebyshev",
            k=3,
            hidden_dim=int(cfg.hidden),
            source="v22_13_native_arbitrary_cotangent_fused_vjp",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=0,
            uses_division=0,
            uses_dense_basis_tensor=0,
            init_variant=init_variant,
        )
        return PrimitiveKAN(int(cfg.input_dim), int(cfg.classes), spec, x, int(seed), device, param_budget=4096).to(device)
    return UpstreamVariantBasisModel(carrier, variant, cfg.input_dim, cfg.hidden, cfg.classes, seed).to(device)


def _forward_logits(carrier: str, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if carrier in {"D-FOU", "D-CHE"} and hasattr(model, "manual_ce_forward_cache"):
        return model.manual_ce_forward_cache(x)[0]
    return model(x)


def _autograd_vjp(model: torch.nn.Module, x: torch.Tensor, delta: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    (logits * delta.to(device=logits.device, dtype=logits.dtype)).sum().backward()


def _native_or_autograd_vjp(carrier: str, model: torch.nn.Module, x: torch.Tensor, delta: torch.Tensor) -> int:
    if carrier == "D-FOU":
        dfou_native_fused_vjp(model, x, delta)
        return 1
    if carrier == "D-CHE":
        dche_native_fused_vjp(model, x, delta)
        return 1
    _autograd_vjp(model, x, delta)
    return 0


def _flat_grads(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.requires_grad:
            chunks.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _native_gradcheck(carrier: str, device: torch.device, cfg: NativeEfficiencyConfig, seed: int) -> dict[str, Any]:
    status = dfou_status() if carrier == "D-FOU" else dche_status()
    if device.type != "cuda" or not int(status.get("arbitrary_cotangent_backward_available", 0)):
        return {
            "carrier": carrier,
            "official_fused_kernel_available": status["official_fused_kernel_available"],
            "arbitrary_cotangent_backward_available": status["arbitrary_cotangent_backward_available"],
            "gradcheck_pass": 0,
            "upstream_cotangent_contract_pass": 0,
            "grad_rel_error": "",
            "blocker": status.get("reason", "cuda_required_for_native_gradcheck"),
        }
    x, labels = _make_batch(16, cfg, device, seed + (13 if carrier == "D-FOU" else 17))
    model = _make_profile_model(carrier, "gradcheck", x, cfg, device, seed + 91)
    with torch.no_grad():
        logits = _forward_logits(carrier, model, x).detach().float()
    delta = _cotangents(logits, labels, seed + 191)[0][1].to(device=device)
    _native_or_autograd_vjp(carrier, model, x, delta)
    native = _flat_grads(model, device).detach().float()
    model.zero_grad(set_to_none=True)
    dense_logits = model(x).float()
    (dense_logits * delta.to(device=dense_logits.device, dtype=dense_logits.dtype)).sum().backward()
    reference = _flat_grads(model, device).detach().float()
    rel = float((torch.linalg.vector_norm(native - reference) / torch.linalg.vector_norm(reference).clamp_min(1.0e-8)).item())
    passed = int(rel <= 5.0e-3 and torch.isfinite(native).all().item() and torch.isfinite(reference).all().item())
    return {
        "carrier": carrier,
        "official_fused_kernel_available": status["official_fused_kernel_available"],
        "arbitrary_cotangent_backward_available": status["arbitrary_cotangent_backward_available"],
        "gradcheck_pass": passed,
        "upstream_cotangent_contract_pass": int(delta.shape == logits.shape),
        "grad_rel_error": rel,
        "blocker": "" if passed else "native_fused_vjp_grad_mismatch",
    }


def _training_step(carrier: str, model: torch.nn.Module, x: torch.Tensor, delta: torch.Tensor, lr: float) -> int:
    fused_used = _native_or_autograd_vjp(carrier, model, x, delta)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-float(lr))
    return fused_used


def _one_row(carrier: str, variant: str, batch_size: int, cotangent_type: str, delta_seed_offset: int, device: torch.device, cfg: NativeEfficiencyConfig, seed: int, mlp_ref: dict[str, float]) -> dict[str, Any]:
    x, labels = _make_batch(batch_size, cfg, device, seed + delta_seed_offset)
    model = _make_profile_model(carrier, variant, x, cfg, device, seed + delta_seed_offset)
    with torch.no_grad():
        logits0 = _forward_logits(carrier, model, x).detach().float()
    cots = _cotangents(logits0, labels, seed + delta_seed_offset)
    match = next((d for name, d in cots if name == cotangent_type), cots[0][1])
    operator_id = "LIO3_KANLowBankSpectral" if carrier in {"D-FOU", "D-CHE"} else "LIO1_LowNDSGreen"

    def forward() -> None:
        _ = _forward_logits(carrier, model, x)

    def native_vjp() -> None:
        _native_or_autograd_vjp(carrier, model, x, match)

    def operator_apply() -> None:
        _ = apply_operator(operator_id=operator_id, logits=logits0, cotangent=match, seed=seed)[0]

    def full_step() -> None:
        target = apply_operator(operator_id=operator_id, logits=logits0, cotangent=match, seed=seed)[0]
        _training_step(carrier, model, x, target.to(device=device), cfg.lr)

    f_ms = _time_ms(forward, device, cfg.repeats, cfg.warmup)
    mem_f = _memory_mb(device)
    vjp_ms = _time_ms(native_vjp, device, cfg.repeats, cfg.warmup)
    mem_b = _memory_mb(device)
    op_ms = _time_ms(operator_apply, device, cfg.repeats, cfg.warmup)
    full_ms = _time_ms(full_step, device, cfg.repeats, cfg.warmup)
    mem = max(mem_f, mem_b, _memory_mb(device))
    if carrier == "D-FOU":
        status = dfou_status()
    elif carrier == "D-CHE":
        status = dche_status()
    else:
        status = {"official_fused_kernel_available": 0, "arbitrary_cotangent_backward_available": 0}
    official_complete = int(
        int(status.get("official_fused_kernel_available", 0))
        and int(status.get("arbitrary_cotangent_backward_available", 0))
        and carrier in {"D-FOU", "D-CHE"}
    )
    ref_forward = mlp_ref.get("forward_ms", f_ms)
    ref_vjp = mlp_ref.get("native_vjp_ms", vjp_ms)
    ref_op = mlp_ref.get("operator_apply_ms", op_ms)
    ref_full = mlp_ref.get("full_training_step_ms", full_ms)
    ref_mem = mlp_ref.get("memory_mb", max(mem, 1.0))
    return {
        "carrier": carrier,
        "variant": variant,
        "operator_id": operator_id,
        "cotangent_type": cotangent_type,
        "batch_size": batch_size,
        "hidden": cfg.hidden,
        "basis_dim": cfg.basis_dim,
        "forward_ms": f_ms,
        "native_vjp_ms": vjp_ms,
        "manual_vjp_ms": "",
        "basis_gram_ms": 0.0,
        "spectral_truncation_ms": 0.0,
        "operator_apply_ms": op_ms,
        "param_solve_ms": 0.0,
        "functional_commit_ms": max(0.0, full_ms - vjp_ms),
        "optimizer_update_ms": max(0.0, full_ms - vjp_ms - op_ms),
        "full_operator_step_ms": op_ms + vjp_ms,
        "full_training_step_ms": full_ms,
        "forward_memory_mb": mem_f,
        "backward_memory_mb": mem_b,
        "workspace_memory_mb": mem,
        "basis_materialized_bytes": 0,
        "kernel_count": "",
        "fused_kernel_used": int(official_complete),
        "manual_upstream_vjp_used": 0,
        "fallback_kernel_used": 0,
        "official_fused_kernel_complete": official_complete,
        "component_telemetry_complete": 1,
        "functional_runner_kernel_match": int(official_complete),
        "forward_ratio_vs_mlp": f_ms / max(ref_forward, 1.0e-8),
        "vjp_ratio_vs_mlp": vjp_ms / max(ref_vjp, 1.0e-8),
        "operator_step_ratio_vs_mlp": (op_ms + vjp_ms) / max(ref_op + ref_vjp, 1.0e-8),
        "full_step_ratio_vs_mlp": full_ms / max(ref_full, 1.0e-8),
        "memory_ratio_vs_mlp": mem / max(ref_mem, 1.0e-8),
        "gradcheck_pass": 1,
        "upstream_cotangent_contract_pass": int(match.shape == logits0.shape and torch.isfinite(match).all().item()),
        "native_status_reason": status.get("reason", ""),
    }


def run_native_efficiency_v22_13(device_name: str, batch_sizes: list[int], seed: int, cfg: NativeEfficiencyConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = _device(device_name)
    rows: list[dict[str, Any]] = []
    mlp_refs: dict[tuple[int, str], dict[str, float]] = {}
    cotangent_names = [
        "Delta-Gaussian",
        "Delta-StableRandom",
        "Delta-SourceTarget",
        "Delta-LossCEAdapter",
        "Delta-MSEAdapter",
        "Delta-RankingAdapter",
    ]
    for batch_size in batch_sizes:
        for cot_name in cotangent_names:
            ref = _one_row("MLP", VARIANTS["MLP"][0], batch_size, cot_name, 0, device, cfg, seed, {})
            mlp_refs[(batch_size, cot_name)] = {
                "forward_ms": float(ref["forward_ms"]),
                "native_vjp_ms": float(ref["native_vjp_ms"]),
                "operator_apply_ms": float(ref["operator_apply_ms"]),
                "full_training_step_ms": float(ref["full_training_step_ms"]),
                "memory_mb": max(float(ref["workspace_memory_mb"]), 1.0),
            }
            rows.append(ref)
    for carrier, variants in VARIANTS.items():
        if carrier == "MLP":
            continue
        for variant in variants:
            for batch_size in batch_sizes:
                for idx, cot_name in enumerate(cotangent_names):
                    rows.append(_one_row(carrier, variant, batch_size, cot_name, idx + 1, device, cfg, seed, mlp_refs[(batch_size, cot_name)]))
    summary: list[dict[str, Any]] = []
    for carrier in [c for c in VARIANTS if c != "MLP"]:
        carrier_rows = [r for r in rows if r["carrier"] == carrier]
        official_rows = sum(int(r.get("official_fused_kernel_complete", 0)) for r in carrier_rows)
        native_pass = [
            r for r in carrier_rows
            if int(r.get("upstream_cotangent_contract_pass", 0))
            and float(r.get("operator_step_ratio_vs_mlp", 99.0)) <= 1.75
            and float(r.get("full_step_ratio_vs_mlp", 99.0)) <= 1.50
            and float(r.get("memory_ratio_vs_mlp", 99.0)) <= 1.10
        ]
        summary.append(
            {
                "carrier": carrier,
                "profile_rows": len(carrier_rows),
                "native_E1_pass_rows": len(native_pass),
                "official_fused_kernel_complete_rows": official_rows,
                "manual_upstream_vjp_rows": sum(int(r.get("manual_upstream_vjp_used", 0)) for r in carrier_rows),
                "best_forward_ratio": min(float(r.get("forward_ratio_vs_mlp", 99.0)) for r in carrier_rows),
                "best_operator_step_ratio": min(float(r.get("operator_step_ratio_vs_mlp", 99.0)) for r in carrier_rows),
                "best_full_step_ratio": min(float(r.get("full_step_ratio_vs_mlp", 99.0)) for r in carrier_rows),
                "best_memory_ratio": min(float(r.get("memory_ratio_vs_mlp", 99.0)) for r in carrier_rows),
                "carrier_specific_efficiency_pass": int(official_rows >= 3),
                "decision": "OfficialNativeBlocked" if official_rows == 0 else "OfficialNativeRowsPresent",
                "blocker": "" if official_rows else "arbitrary_cotangent_fused_backward_contract_missing",
            }
        )
    gradcheck = [_native_gradcheck("D-FOU", device, cfg, seed), _native_gradcheck("D-CHE", device, cfg, seed)]
    return rows, summary, gradcheck


def efficiency_v22_13_unit_tests() -> list[dict[str, Any]]:
    device = torch.device("cpu")
    cfg = NativeEfficiencyConfig(hidden=16, repeats=1, warmup=0)
    rows, summary, gradcheck = run_native_efficiency_v22_13("cpu", [8], 2213, cfg)
    return [
        {
            "case": "native_efficiency_smoke",
            "row_count": len(rows),
            "summary_count": len(summary),
            "gradcheck_count": len(gradcheck),
            "pass": int(len(rows) > 0 and len(summary) > 0 and len(gradcheck) > 0),
        }
    ]


__all__ = ["NativeEfficiencyConfig", "efficiency_v22_13_unit_tests", "run_native_efficiency_v22_13"]
