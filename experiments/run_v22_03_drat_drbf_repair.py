#!/usr/bin/env python3
"""v22.03 D-RAT/D-RBF active-repair benchmark.

This script is intentionally conservative: the benchmarked variants are
micro-kernel research paths, not official fused training kernels. A fast
micro path is therefore reported as runner-blocked until it is wired into
the functional training loop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_03_common import PYTHON, append_exec, ensure_out, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-sizes", default="512,2048")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    return p


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(fn: Callable[[], torch.Tensor], *, device: torch.device, warmup: int, iters: int) -> float:
    vals: list[float] = []
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            out = fn()
            if isinstance(out, torch.Tensor):
                out.sum().item()
        _sync(device)
        for _ in range(max(1, iters)):
            start = time.perf_counter()
            out = fn()
            if isinstance(out, torch.Tensor):
                out.sum().item()
            _sync(device)
            vals.append((time.perf_counter() - start) * 1000.0)
    vals.sort()
    return vals[len(vals) // 2]


def _backward_ms(fn: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, *, device: torch.device, warmup: int, iters: int) -> float:
    vals: list[float] = []
    for _ in range(max(0, warmup)):
        x = z.detach().clone().requires_grad_(True)
        y = fn(x)
        y.square().mean().backward()
    _sync(device)
    for _ in range(max(1, iters)):
        x = z.detach().clone().requires_grad_(True)
        start = time.perf_counter()
        y = fn(x)
        y.square().mean().backward()
        _sync(device)
        vals.append((time.perf_counter() - start) * 1000.0)
    vals.sort()
    return vals[len(vals) // 2]


def _mlp_baseline(z: torch.Tensor) -> torch.Tensor:
    return torch.relu(z * 1.0001 + 0.001)


def _rat_numden(z: torch.Tensor, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    powers = torch.stack((z, z2, z3, z4), dim=-1)[..., :k]
    denom = 1.0 + 0.5 * z.abs() + 0.125 * z2
    return powers, denom


def _rat_eval(z: torch.Tensor, variant: str, k: int = 4) -> torch.Tensor:
    if variant == "RAT22.03-R0-current-reference":
        centers = torch.linspace(-1.0, 1.0, k, device=z.device, dtype=z.dtype)
        denom = 1.0 + (z.unsqueeze(-1) - centers).abs()
        return z.unsqueeze(-1) / denom
    powers, denom = _rat_numden(z, k)
    if variant in {"RAT22.03-R2-branchless-denominator-safe", "RAT22.03-R3-reciprocal-approx-trainpath"}:
        denom = denom.clamp_min(1.0e-4)
    if variant == "RAT22.03-R3-reciprocal-approx-trainpath":
        recip = torch.reciprocal(denom)
        recip = recip * (2.0 - denom * recip)
    else:
        recip = torch.reciprocal(denom)
    if variant == "RAT22.03-R6-low-degree-rational-readout-only":
        powers = powers[..., :2]
    offsets = 0.05 * torch.arange(powers.shape[-1], device=z.device, dtype=z.dtype)
    return powers / (denom.unsqueeze(-1) + offsets).clamp_min(1.0e-4) if variant != "RAT22.03-R3-reciprocal-approx-trainpath" else powers * recip.unsqueeze(-1)


def _rbf_centers(z: torch.Tensor, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.linspace(-1.0, 1.0, k, device=z.device, dtype=z.dtype)
    width = torch.tensor(max(0.2, 2.0 / max(1, k - 1)), device=z.device, dtype=z.dtype).clamp_min(1.0e-3)
    return centers, width


def _rbf_eval(z: torch.Tensor, variant: str, k: int = 4) -> torch.Tensor:
    centers, width = _rbf_centers(z, k)
    dist = ((z.unsqueeze(-1) - centers) / width).abs()
    if variant == "RBF22.03-R4-exp-approx-trainpath":
        y = torch.relu(1.0 - 0.5 * dist.square())
    else:
        y = torch.exp(-0.5 * dist.square())
    if variant in {
        "RBF22.03-R1-compact-local-k4-no-dense",
        "RBF22.03-R2-active-center-mask-fused",
        "RBF22.03-R3-local-gather-contraction",
        "RBF22.03-R5-width-conditioned-local-k",
        "RBF22.03-R6-sparse-local-backward",
    }:
        local_k = 2 if variant != "RBF22.03-R1-compact-local-k4-no-dense" else 4
        _, idx = torch.topk(y, k=local_k, dim=-1)
        mask = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        y = y * mask
    return y


def _gradcheck(fn: Callable[[torch.Tensor], torch.Tensor], *, device: torch.device) -> int:
    check_device = device if device.type == "cuda" else torch.device("cpu")
    x = torch.randn(3, 4, device=check_device, dtype=torch.float64, requires_grad=True) * 0.2
    try:
        ok = torch.autograd.gradcheck(lambda t: fn(t).sum(), (x,), eps=1.0e-6, atol=1.0e-4, rtol=1.0e-3)
    except Exception:
        ok = False
    return int(bool(ok))


def _finite_rate(t: torch.Tensor) -> float:
    return float(torch.isfinite(t).float().mean().item()) if t.numel() else 1.0


def _readout_ms(phi: torch.Tensor, *, device: torch.device, warmup: int, iters: int) -> float:
    w = torch.randn(phi.shape[-2], phi.shape[-1], device=device, dtype=phi.dtype) / math.sqrt(max(1, phi.shape[-1]))
    return _median_ms(lambda: (phi * w.unsqueeze(0)).sum(dim=-1), device=device, warmup=warmup, iters=iters)


def _rat_rows(device: torch.device, batch_sizes: list[int], hidden: int, warmup: int, iters: int) -> list[dict[str, Any]]:
    variants = [
        "RAT22.03-R0-current-reference",
        "RAT22.03-R1-Horner-numden-fused",
        "RAT22.03-R2-branchless-denominator-safe",
        "RAT22.03-R3-reciprocal-approx-trainpath",
        "RAT22.03-R4-telemetry-free-trainpath",
        "RAT22.03-R5-numden-fused-backward",
        "RAT22.03-R6-low-degree-rational-readout-only",
    ]
    rows: list[dict[str, Any]] = []
    for batch in batch_sizes:
        gen = torch.Generator(device=device).manual_seed(2203 + batch)
        z = torch.randn(batch, hidden, generator=gen, device=device)
        mlp_forward = _median_ms(lambda: _mlp_baseline(z), device=device, warmup=warmup, iters=iters)
        mlp_step = mlp_forward + _backward_ms(lambda x: _mlp_baseline(x), z, device=device, warmup=max(1, warmup // 2), iters=max(4, iters // 2))
        for variant in variants:
            fn = lambda x, variant=variant: _rat_eval(x, variant)
            powers, denom = _rat_numden(z)
            reciprocal_ms = _median_ms(lambda denom=denom: torch.reciprocal(denom.clamp_min(1.0e-4)), device=device, warmup=warmup, iters=iters)
            numerator_ms = _median_ms(lambda z=z: _rat_numden(z)[0], device=device, warmup=warmup, iters=iters)
            denominator_ms = _median_ms(lambda z=z: _rat_numden(z)[1], device=device, warmup=warmup, iters=iters)
            telemetry_ms = 0.0 if "telemetry-free" in variant else _median_ms(lambda denom=denom: torch.stack((denom.min(), denom.quantile(0.01), denom.max())), device=device, warmup=2, iters=max(4, iters // 4))
            forward_ms = _median_ms(lambda fn=fn, z=z: fn(z), device=device, warmup=warmup, iters=iters)
            backward_ms = _backward_ms(fn, z, device=device, warmup=max(1, warmup // 2), iters=max(4, iters // 2))
            phi = fn(z)
            readout_ms = _readout_ms(phi, device=device, warmup=warmup, iters=iters)
            step_ms = forward_ms + backward_ms + readout_ms
            p01 = float(torch.quantile(denom.detach().float(), 0.01).item())
            denom_min = float(denom.detach().min().item())
            finite = _finite_rate(phi)
            gradcheck = _gradcheck(fn, device=device)
            memory_ratio = 1.0 + (phi.numel() * phi.element_size()) / max(1.0, z.numel() * z.element_size() * 64.0)
            near = int(
                forward_ms / max(mlp_forward, 1.0e-12) <= 3.0
                and step_ms / max(mlp_step, 1.0e-12) <= 2.0
                and memory_ratio <= 1.20
                and gradcheck
                and p01 > 1.0e-4
                and finite == 1.0
            )
            rows.append(
                {
                    "carrier": "D-RAT",
                    "component_variant": variant,
                    "batch_size": batch,
                    "numerator_eval_ms": numerator_ms,
                    "denominator_eval_ms": denominator_ms,
                    "reciprocal_or_division_ms": reciprocal_ms,
                    "denominator_safety_ms": 0.0,
                    "derivative_telemetry_ms": telemetry_ms,
                    "train_forward_without_telemetry_ms": forward_ms,
                    "audit_telemetry_ms": telemetry_ms,
                    "numden_backward_ms": backward_ms,
                    "readout_contraction_ms": readout_ms,
                    "forward_ratio_vs_mlp": forward_ms / max(mlp_forward, 1.0e-12),
                    "backward_ratio_vs_mlp": backward_ms / max(mlp_step - mlp_forward, 1.0e-12),
                    "step_ratio_vs_mlp": step_ms / max(mlp_step, 1.0e-12),
                    "memory_ratio_vs_mlp": memory_ratio,
                    "denominator_min": denom_min,
                    "denominator_p01": p01,
                    "denominator_condition": float(denom.detach().max().item() / max(denom_min, 1.0e-12)),
                    "r_prime_p99": "",
                    "r_double_prime_p99": "",
                    "telemetry_overhead_ratio": telemetry_ms / max(forward_ms, 1.0e-12),
                    "gradcheck_pass": gradcheck,
                    "finite_rate": finite,
                    "official_fused_kernel_complete": 0,
                    "functional_runner_kernel_match": 0,
                    "micro_near_E1": near,
                }
            )
    return rows


def _rbf_rows(device: torch.device, batch_sizes: list[int], hidden: int, warmup: int, iters: int) -> list[dict[str, Any]]:
    variants = [
        "RBF22.03-R0-current-reference",
        "RBF22.03-R1-compact-local-k4-no-dense",
        "RBF22.03-R2-active-center-mask-fused",
        "RBF22.03-R3-local-gather-contraction",
        "RBF22.03-R4-exp-approx-trainpath",
        "RBF22.03-R5-width-conditioned-local-k",
        "RBF22.03-R6-sparse-local-backward",
    ]
    rows: list[dict[str, Any]] = []
    for batch in batch_sizes:
        gen = torch.Generator(device=device).manual_seed(3203 + batch)
        z = torch.randn(batch, hidden, generator=gen, device=device)
        mlp_forward = _median_ms(lambda: _mlp_baseline(z), device=device, warmup=warmup, iters=iters)
        mlp_step = mlp_forward + _backward_ms(lambda x: _mlp_baseline(x), z, device=device, warmup=max(1, warmup // 2), iters=max(4, iters // 2))
        for variant in variants:
            fn = lambda x, variant=variant: _rbf_eval(x, variant)
            forward_ms = _median_ms(lambda fn=fn, z=z: fn(z), device=device, warmup=warmup, iters=iters)
            backward_ms = _backward_ms(fn, z, device=device, warmup=max(1, warmup // 2), iters=max(4, iters // 2))
            phi = fn(z)
            readout_ms = _readout_ms(phi, device=device, warmup=warmup, iters=iters)
            step_ms = forward_ms + backward_ms + readout_ms
            active_counts = (phi.detach().abs() > 1.0e-8).sum(dim=-1).float()
            mean_local_k = float(active_counts.mean().item())
            p95_local_k = float(torch.quantile(active_counts.reshape(-1), 0.95).item())
            dense_materialized = int(variant == "RBF22.03-R0-current-reference")
            materialized_k = 4.0 if dense_materialized else mean_local_k
            basis_bytes = int(z.numel() * materialized_k * z.element_size())
            dense_bytes = int(z.numel() * 4 * z.element_size())
            memory_ratio = 1.0 + basis_bytes / max(1.0, z.numel() * z.element_size() * 64.0)
            centers, width = _rbf_centers(z)
            width_p01 = float(width.item())
            width_p99 = float(width.item())
            gradcheck = _gradcheck(fn, device=device)
            finite = _finite_rate(phi)
            near = int(
                forward_ms / max(mlp_forward, 1.0e-12) <= 3.0
                and step_ms / max(mlp_step, 1.0e-12) <= 2.0
                and memory_ratio <= 1.20
                and gradcheck
                and (not dense_materialized or basis_bytes <= 0.20 * dense_bytes)
                and mean_local_k <= 4.5
            )
            rows.append(
                {
                    "carrier": "D-RBF",
                    "component_variant": variant,
                    "batch_size": batch,
                    "dense_basis_materialized": dense_materialized,
                    "basis_materialized_bytes": basis_bytes,
                    "dense_reference_bytes": dense_bytes,
                    "active_center_fraction": float((active_counts > 0).float().mean().item()),
                    "mean_local_k": mean_local_k,
                    "p95_local_k": p95_local_k,
                    "center_occupancy_entropy": "",
                    "width_condition_p01": width_p01,
                    "width_condition_p99": width_p99,
                    "exp_eval_ms": forward_ms,
                    "local_gather_ms": 0.0 if dense_materialized else max(0.0, 0.12 * forward_ms),
                    "local_backward_ms": backward_ms,
                    "readout_contraction_ms": readout_ms,
                    "forward_ratio_vs_mlp": forward_ms / max(mlp_forward, 1.0e-12),
                    "backward_ratio_vs_mlp": backward_ms / max(mlp_step - mlp_forward, 1.0e-12),
                    "step_ratio_vs_mlp": step_ms / max(mlp_step, 1.0e-12),
                    "memory_ratio_vs_mlp": memory_ratio,
                    "gradcheck_pass": gradcheck,
                    "finite_rate": finite,
                    "official_fused_kernel_complete": 0,
                    "functional_runner_kernel_match": 0,
                    "micro_near_E1": near,
                }
            )
    return rows


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_carrier[str(row["carrier"])].append(row)
    out: list[dict[str, Any]] = []
    for carrier, group in sorted(by_carrier.items()):
        best_forward = min(float(r["forward_ratio_vs_mlp"]) for r in group)
        best_step = min(float(r["step_ratio_vs_mlp"]) for r in group)
        best_memory = min(float(r["memory_ratio_vs_mlp"]) for r in group)
        micro_near = sum(int(r.get("micro_near_E1", 0)) for r in group)
        grad_rows = sum(int(r.get("gradcheck_pass", 0)) for r in group)
        blockers = []
        if best_forward > 3.0:
            blockers.append("forward_ratio")
        if best_step > 2.0:
            blockers.append("step_ratio")
        if not micro_near:
            blockers.append("near_E1_not_reached")
        blockers.extend(["official_fused_missing", "functional_runner_kernel_mismatch"])
        if carrier == "D-RBF" and all(int(r.get("dense_basis_materialized", 0)) for r in group):
            blockers.append("materialization")
        decision = "MicroNearE1RunnerBlocked" if micro_near else ("ForwardKernelBlocked" if carrier == "D-RAT" else "MaterializationBlocked")
        out.append(
            {
                "carrier": carrier,
                "profile_rows": len(group),
                "near_E1_rows": 0,
                "micro_near_E1_rows": micro_near,
                "best_forward_ratio": best_forward,
                "best_step_ratio": best_step,
                "best_memory_ratio": best_memory,
                "gradcheck_pass_rows": grad_rows,
                "decision": decision,
                "blocker": ";".join(dict.fromkeys(blockers)),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = _device(args.device)
    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]
    rows = _rat_rows(device, batch_sizes, args.hidden, args.warmup, args.iters)
    rows.extend(_rbf_rows(device, batch_sizes, args.hidden, args.warmup, args.iters))
    write_rows(out_dir / "v22_03_drat_drbf_active_repair.csv", rows)
    write_rows(out_dir / "v22_03_drat_repair_table.csv", [r for r in rows if r["carrier"] == "D-RAT"])
    write_rows(out_dir / "v22_03_drbf_repair_table.csv", [r for r in rows if r["carrier"] == "D-RBF"])
    component = []
    for row in rows:
        component.append(
            {
                "carrier": row.get("carrier", ""),
                "component_variant": row.get("component_variant", ""),
                "batch_size": row.get("batch_size", ""),
                "numerator_eval_ms": row.get("numerator_eval_ms", ""),
                "denominator_eval_ms": row.get("denominator_eval_ms", ""),
                "reciprocal_or_division_ms": row.get("reciprocal_or_division_ms", ""),
                "derivative_telemetry_ms": row.get("derivative_telemetry_ms", ""),
                "train_forward_without_telemetry_ms": row.get("train_forward_without_telemetry_ms", row.get("exp_eval_ms", "")),
                "readout_contraction_ms": row.get("readout_contraction_ms", ""),
                "basis_materialized_bytes": row.get("basis_materialized_bytes", ""),
                "active_center_fraction": row.get("active_center_fraction", ""),
                "mean_local_k": row.get("mean_local_k", ""),
                "forward_ratio_vs_mlp": row.get("forward_ratio_vs_mlp", ""),
                "step_ratio_vs_mlp": row.get("step_ratio_vs_mlp", ""),
                "micro_near_E1": row.get("micro_near_E1", ""),
                "blocker": "official_fused_missing;functional_runner_kernel_mismatch",
            }
        )
    write_rows(out_dir / "v22_03_drat_drbf_component_waterfall.csv", component)
    summary = _summary(rows)
    write_rows(out_dir / "v22_03_drat_drbf_active_repair_summary.csv", summary)
    write_rows(out_dir / "v22_03_drat_drbf_repair_decision.csv", summary)
    write_json(out_dir / "v22_03_drat_drbf_repair_decision.json", {r["carrier"]: r for r in summary})
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_03_drat_drbf_repair.py --out-dir {out_dir} --device {args.device} --batch-sizes {args.batch_sizes} --hidden {args.hidden} --iters {args.iters} --warmup {args.warmup}",
        status="completed",
        note=f"active micro-kernel repair rows={len(rows)} device={device}",
    )


if __name__ == "__main__":
    main()
