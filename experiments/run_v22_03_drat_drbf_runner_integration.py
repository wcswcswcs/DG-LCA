#!/usr/bin/env python3
"""v22.03 D-RAT/D-RBF limited same-kernel runner integration.

This is not a functional-update promotion runner. It only checks whether the
same active-repair micro-kernel callable can sit inside a CUDA autograd
training step with a readout head. Official fused-kernel status remains false
until the implementation is wired as a production fused kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_03_common import PYTHON, append_exec, ensure_out, read_json, write_json, write_rows  # noqa: E402
from experiments.run_v22_03_drat_drbf_repair import _device, _rat_eval, _rbf_eval  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--rat-variant", default="RAT22.03-R0-current-reference")
    p.add_argument("--rbf-variant", default="RBF22.03-R4-exp-approx-trainpath")
    return p


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _source_hash(fn: Callable[..., torch.Tensor]) -> str:
    try:
        src = inspect.getsource(fn)
    except OSError:
        src = repr(fn)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _kernel_features(z: torch.Tensor, carrier: str, variant: str) -> torch.Tensor:
    if carrier == "D-RAT":
        return _rat_eval(z, variant)
    if carrier == "D-RBF":
        return _rbf_eval(z, variant)
    raise ValueError(carrier)


def _run_limited_runner(
    *,
    carrier: str,
    variant: str,
    device: torch.device,
    batch_size: int,
    hidden: int,
    classes: int,
    steps: int,
    warmup: int,
) -> dict[str, Any]:
    gen = torch.Generator(device=device).manual_seed(220303 + (17 if carrier == "D-RAT" else 31))
    z = torch.randn(batch_size, hidden, generator=gen, device=device)
    y = torch.randint(0, classes, (batch_size,), generator=gen, device=device)
    probe_phi = _kernel_features(z, carrier, variant)
    coeff = torch.nn.Parameter(
        torch.randn(hidden, probe_phi.shape[-1], classes, generator=gen, device=device) / math.sqrt(max(1, hidden * probe_phi.shape[-1]))
    )
    bias = torch.nn.Parameter(torch.zeros(classes, device=device))
    opt = torch.optim.SGD([coeff, bias], lr=0.05, momentum=0.0)

    losses: list[float] = []
    step_ms: list[float] = []
    finite_loss_rows = 0
    nonzero_grad_rows = 0
    same_shape_rows = 0
    for step in range(max(1, steps + warmup)):
        opt.zero_grad(set_to_none=True)
        start = time.perf_counter()
        phi = _kernel_features(z, carrier, variant)
        logits = torch.einsum("bhk,hkc->bc", phi, coeff) + bias
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        _sync(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if step >= warmup:
            losses.append(float(loss.detach().item()))
            step_ms.append(elapsed)
            finite_loss_rows += int(bool(torch.isfinite(loss).item()))
            grad_norm = float(coeff.grad.detach().norm().item()) if coeff.grad is not None else 0.0
            nonzero_grad_rows += int(grad_norm > 0.0 and math.isfinite(grad_norm))
            same_shape_rows += int(tuple(phi.shape) == tuple(probe_phi.shape))
    step_ms_sorted = sorted(step_ms)
    median_step_ms = step_ms_sorted[len(step_ms_sorted) // 2] if step_ms_sorted else float("nan")
    loss_delta = losses[-1] - losses[0] if len(losses) >= 2 else float("nan")
    source_fn = _rat_eval if carrier == "D-RAT" else _rbf_eval
    return {
        "carrier": carrier,
        "component_variant": variant,
        "device": str(device),
        "batch_size": batch_size,
        "hidden": hidden,
        "basis_k": int(probe_phi.shape[-1]),
        "classes": classes,
        "runner_steps": steps,
        "finite_loss_rows": finite_loss_rows,
        "nonzero_grad_rows": nonzero_grad_rows,
        "same_shape_rows": same_shape_rows,
        "loss_start": losses[0] if losses else "",
        "loss_end": losses[-1] if losses else "",
        "loss_delta": loss_delta,
        "median_step_ms": median_step_ms,
        "same_kernel_callable_hash": _source_hash(source_fn),
        "same_kernel_callable_used": 1,
        "no_cpu_offload": int(device.type == "cuda"),
        "limited_runner_kernel_match": int(finite_loss_rows == steps and nonzero_grad_rows == steps and same_shape_rows == steps),
        "official_fused_kernel_complete": 0,
        "promotion_allowed": 0,
        "blocker": "official_fused_missing" if finite_loss_rows == steps and nonzero_grad_rows == steps and same_shape_rows == steps else "limited_runner_kernel_failed",
    }


def _merge_decision(out_dir: Path, runner_rows: list[dict[str, Any]]) -> None:
    decision = read_json(out_dir / "v22_03_drat_drbf_repair_decision.json")
    for row in runner_rows:
        carrier = str(row.get("carrier", ""))
        if carrier not in decision:
            continue
        base = dict(decision.get(carrier, {}) or {})
        runner_pass = int(row.get("limited_runner_kernel_match", 0))
        base["limited_runner_kernel_match"] = runner_pass
        base["limited_runner_variant"] = row.get("component_variant", "")
        base["limited_runner_median_step_ms"] = row.get("median_step_ms", "")
        if int(base.get("micro_near_E1_rows", 0) or 0) > 0 and runner_pass:
            base["decision"] = "MicroNearE1LimitedRunnerPassedOfficialFusedBlocked"
            base["blocker"] = "official_fused_missing"
        elif runner_pass:
            base["decision"] = "LimitedRunnerPassedKernelStillNotNearE1"
            base["blocker"] = "near_E1_not_reached;official_fused_missing"
        else:
            blockers = [x for x in str(base.get("blocker", "")).split(";") if x]
            blockers.append("limited_runner_kernel_failed")
            base["blocker"] = ";".join(dict.fromkeys(blockers))
        decision[carrier] = base
    write_json(out_dir / "v22_03_drat_drbf_repair_decision.json", decision)
    write_rows(out_dir / "v22_03_drat_drbf_repair_decision.csv", [decision[k] for k in sorted(decision)])


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = _device(args.device)
    rows = [
        _run_limited_runner(
            carrier="D-RAT",
            variant=args.rat_variant,
            device=device,
            batch_size=args.batch_size,
            hidden=args.hidden,
            classes=args.classes,
            steps=args.steps,
            warmup=args.warmup,
        ),
        _run_limited_runner(
            carrier="D-RBF",
            variant=args.rbf_variant,
            device=device,
            batch_size=args.batch_size,
            hidden=args.hidden,
            classes=args.classes,
            steps=args.steps,
            warmup=args.warmup,
        ),
    ]
    write_rows(out_dir / "v22_03_drat_drbf_runner_integration_summary.csv", rows)
    write_json(out_dir / "v22_03_drat_drbf_runner_integration_decision.json", {r["carrier"]: r for r in rows})
    _merge_decision(out_dir, rows)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_03_drat_drbf_runner_integration.py --out-dir {out_dir} --device {args.device} --batch-size {args.batch_size} --hidden {args.hidden} --classes {args.classes} --steps {args.steps} --warmup {args.warmup} --rat-variant {args.rat_variant} --rbf-variant {args.rbf_variant}",
        status="completed",
        note=f"limited same-kernel runner integration rows={len(rows)} device={device}",
    )


if __name__ == "__main__":
    main()
