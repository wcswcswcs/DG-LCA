#!/usr/bin/env python3
"""v22.17 KAN basis VJP/JVP gradcheck on real KANbeFair batches."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.integration.kanbefair_adapter import canonical_dataset_name  # noqa: E402
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec  # noqa: E402
from experiments.run_v22_16_common import selected_named_parameters  # noqa: E402
from experiments.run_v22_17_common import OUT_ROOT, WORKTREE_ROOT, append_exec, ensure_out, write_rows  # noqa: E402
from experiments.run_v22_17_kanbefair_dgkan_eval import _get_loaders, _make_model  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--kanbefair-root", default=str(WORKTREE_ROOT))
    p.add_argument("--datasets", default="MNIST,FMNIST,KMNIST")
    p.add_argument("--models", default="DGKAN_DFOU,DGKAN_DCHE")
    p.add_argument("--implementation-modes", default="bridge_dense_cache,native_no_dense_basis")
    p.add_argument("--seeds", default="0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--selectors", default="basis,readout,all")
    p.add_argument("--directions", type=int, default=3)
    p.add_argument("--fd-eps", type=float, default=1.0e-3)
    p.add_argument("--timing-repeats", type=int, default=3)
    p.add_argument("--output-suffix", default="")
    return p


def _split(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _out_path(name: str, suffix: str) -> Path:
    path = OUT_ROOT / name
    clean = str(suffix or "").strip().strip("_")
    if not clean:
        return path
    return path.with_name(f"{path.stem}_{clean}{path.suffix}")


def _batch_fingerprint(xb: torch.Tensor, yb: torch.Tensor) -> str:
    x_cpu = xb.detach().cpu().contiguous()
    y_cpu = yb.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x_cpu.shape)).encode("utf-8"))
    h.update(x_cpu.numpy().tobytes())
    h.update(str(tuple(y_cpu.shape)).encode("utf-8"))
    h.update(y_cpu.numpy().tobytes())
    return h.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn: Any, device: torch.device, repeats: int) -> float:
    values: list[float] = []
    for idx in range(int(repeats) + 1):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        if idx > 0:
            values.append((time.perf_counter() - start) * 1000.0)
    return float(sum(values) / max(1, len(values)))


def _make_native_no_dense_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    hidden: int,
    seed: int,
    device: torch.device,
    x_stats: torch.Tensor,
) -> PrimitiveKAN:
    carrier = "D-CHE" if "DCHE" in model_name else "D-FOU"
    basis_name = "chebyshev" if carrier == "D-CHE" else "fourier_lowfreq"
    init_variant = "cheby_k3_triton_l3_gradbuf" if carrier == "D-CHE" else "fourier_k3_triton_l3_matmul"
    spec = PrimitiveSpec(
        candidate_id=f"v22.17-{carrier}-basis-gradcheck-native-no-dense",
        basis_family=carrier,
        basis_name=basis_name,
        k=3,
        hidden_dim=max(4, int(hidden)),
        source="v22_17_basis_jvp_vjp_gradcheck",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=int(carrier == "D-FOU"),
        uses_division=0,
        uses_dense_basis_tensor=0,
        init_variant=init_variant,
    )
    budget = int(input_dim) * int(hidden) + int(hidden) * int(hidden) + int(hidden) * int(output_dim)
    return PrimitiveKAN(input_dim, output_dim, spec, x_stats.to(device), seed, device, param_budget=budget).to(device)


def _make_audit_model(
    model_name: str,
    implementation_mode: str,
    input_dim: int,
    output_dim: int,
    hidden: int,
    seed: int,
    device: torch.device,
    x_stats: torch.Tensor,
) -> torch.nn.Module:
    if implementation_mode == "bridge_dense_cache":
        return _make_model(model_name, input_dim, output_dim, hidden, seed, device, x_stats).to(device)
    if implementation_mode == "native_no_dense_basis":
        return _make_native_no_dense_model(model_name, input_dim, output_dim, hidden, seed, device, x_stats).to(device)
    raise ValueError(f"unknown implementation_mode={implementation_mode}")


def _warmup_train(model: torch.nn.Module, loader: Any, steps: int, lr: float, device: torch.device) -> int:
    if int(steps) <= 0:
        return 0
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    train_iter = iter(loader)
    done = 0
    for _idx in range(int(steps)):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        opt.step()
        done += 1
    return done


def _flat_grads(named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    parts = []
    for _name, p in named:
        parts.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _manual_ce_grad(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    named: list[tuple[str, torch.nn.Parameter]],
) -> tuple[torch.Tensor, dict[str, float]]:
    model.zero_grad(set_to_none=True)
    if not hasattr(model, "manual_ce_forward_cache") or not hasattr(model, "manual_ce_backward_from_cache"):
        return torch.empty(0, device=xb.device), {
            "manual_forward_available": 0.0,
            "manual_backward_available": 0.0,
            "ce_loss_manual": float("nan"),
            "output_max_abs_error": float("nan"),
        }
    logits, cache = model.manual_ce_forward_cache(xb)
    manual_logits = logits.detach().float()
    loss = model.manual_ce_backward_from_cache(logits, cache, yb)
    grads = _flat_grads(named).detach().float()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        ref_logits = model(xb).detach().float()
    return grads, {
        "manual_forward_available": 1.0,
        "manual_backward_available": 1.0,
        "ce_loss_manual": float(loss.detach().float().item()) if torch.is_tensor(loss) else float(loss),
        "output_max_abs_error": float((manual_logits - ref_logits).abs().max().item()),
    }


def _autograd_ce_grad(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    named: list[tuple[str, torch.nn.Parameter]],
) -> tuple[torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    logits = model(xb).float()
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    grads = _flat_grads(named).detach().float()
    model.zero_grad(set_to_none=True)
    return grads, float(loss.detach().float().item())


def _direction_chunks(
    named: list[tuple[str, torch.nn.Parameter]],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    flat_parts = [torch.randn(int(p.numel()), generator=gen, dtype=torch.float32) for _name, p in named]
    flat = torch.cat(flat_parts) if flat_parts else torch.empty(0, dtype=torch.float32)
    flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
    chunks: list[torch.Tensor] = []
    offset = 0
    for _name, p in named:
        n = int(p.numel())
        chunks.append(flat[offset : offset + n].to(device=device, dtype=p.dtype).reshape_as(p))
        offset += n
    return flat.to(device=device), chunks


@torch.no_grad()
def _add_direction(named: list[tuple[str, torch.nn.Parameter]], chunks: list[torch.Tensor], scale: float) -> None:
    for (_name, p), d in zip(named, chunks):
        p.add_(d, alpha=float(scale))


def _loss_value(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(model(xb).float(), yb).detach().float().item())


def _directional_jvp_stats(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    named: list[tuple[str, torch.nn.Parameter]],
    manual_grads: torch.Tensor,
    ref_grads: torch.Tensor,
    directions: int,
    eps: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    relerrs: list[float] = []
    ref_relerrs: list[float] = []
    abs_errors: list[float] = []
    fd_abs_values: list[float] = []
    manual_dot_abs_values: list[float] = []
    ref_dot_abs_values: list[float] = []
    for idx in range(int(directions)):
        v_flat, chunks = _direction_chunks(named, seed + idx * 9973, device)
        _add_direction(named, chunks, float(eps))
        loss_plus = _loss_value(model, xb, yb)
        _add_direction(named, chunks, -2.0 * float(eps))
        loss_minus = _loss_value(model, xb, yb)
        _add_direction(named, chunks, float(eps))
        fd = (loss_plus - loss_minus) / (2.0 * float(eps))
        manual_dot = float(torch.dot(manual_grads.to(device), v_flat).detach().item()) if manual_grads.numel() else float("nan")
        ref_dot = float(torch.dot(ref_grads.to(device), v_flat).detach().item()) if ref_grads.numel() else float("nan")
        denom = max(abs(fd), abs(ref_dot), 1.0e-8)
        relerrs.append(abs(manual_dot - fd) / denom)
        ref_relerrs.append(abs(ref_dot - fd) / denom)
        abs_errors.append(abs(manual_dot - fd))
        fd_abs_values.append(abs(fd))
        manual_dot_abs_values.append(abs(manual_dot))
        ref_dot_abs_values.append(abs(ref_dot))
    return {
        "jvp_direction_count": float(int(directions)),
        "jvp_relerr_max": max(relerrs) if relerrs else float("inf"),
        "jvp_relerr_mean": float(sum(relerrs) / max(1, len(relerrs))),
        "jvp_autograd_relerr_max": max(ref_relerrs) if ref_relerrs else float("inf"),
        "jvp_dot_abs_error_max": max(abs_errors) if abs_errors else float("inf"),
        "jvp_fd_abs_max": max(fd_abs_values) if fd_abs_values else 0.0,
        "jvp_fd_abs_min": min(fd_abs_values) if fd_abs_values else 0.0,
        "jvp_manual_dot_abs_max": max(manual_dot_abs_values) if manual_dot_abs_values else 0.0,
        "jvp_ref_dot_abs_max": max(ref_dot_abs_values) if ref_dot_abs_values else 0.0,
    }


def _func_jvp_stats(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    named: list[tuple[str, torch.nn.Parameter]],
    manual_grads: torch.Tensor,
    ref_grads: torch.Tensor,
    directions: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    try:
        from torch.func import functional_call, jvp
    except Exception as exc:
        return {
            "func_jvp_available": 0,
            "func_jvp_error": f"{type(exc).__name__}: {exc}",
            "func_jvp_relerr_max": "",
            "func_jvp_relerr_mean": "",
            "func_jvp_manual_dot_abs_error_max": "",
            "func_jvp_ref_relerr_max": "",
        }
    param_map = dict(model.named_parameters())
    names = [name for name, _p in named]
    base_tuple = tuple(param_map[name] for name in names)

    def loss_fn(*selected_values: torch.Tensor) -> torch.Tensor:
        patched = dict(param_map)
        for name, value in zip(names, selected_values):
            patched[name] = value
        logits = functional_call(model, patched, (xb,), strict=False).float()
        return F.cross_entropy(logits, yb)

    relerrs: list[float] = []
    ref_relerrs: list[float] = []
    abs_errors: list[float] = []
    try:
        for idx in range(int(directions)):
            v_flat, chunks = _direction_chunks(named, seed + idx * 7919, device)
            tangent_tuple = tuple(chunks)
            _loss, tangent = jvp(loss_fn, base_tuple, tangent_tuple)
            exact = float(tangent.detach().float().item())
            manual_dot = float(torch.dot(manual_grads.to(device), v_flat).detach().item()) if manual_grads.numel() else float("nan")
            ref_dot = float(torch.dot(ref_grads.to(device), v_flat).detach().item()) if ref_grads.numel() else float("nan")
            denom = max(abs(exact), abs(ref_dot), 1.0e-8)
            relerrs.append(abs(manual_dot - exact) / denom)
            ref_relerrs.append(abs(ref_dot - exact) / denom)
            abs_errors.append(abs(manual_dot - exact))
    except Exception as exc:
        return {
            "func_jvp_available": 0,
            "func_jvp_error": f"{type(exc).__name__}: {exc}",
            "func_jvp_relerr_max": "",
            "func_jvp_relerr_mean": "",
            "func_jvp_manual_dot_abs_error_max": "",
            "func_jvp_ref_relerr_max": "",
        }
    return {
        "func_jvp_available": 1,
        "func_jvp_error": "",
        "func_jvp_relerr_max": max(relerrs) if relerrs else float("inf"),
        "func_jvp_relerr_mean": float(sum(relerrs) / max(1, len(relerrs))),
        "func_jvp_manual_dot_abs_error_max": max(abs_errors) if abs_errors else float("inf"),
        "func_jvp_ref_relerr_max": max(ref_relerrs) if ref_relerrs else float("inf"),
    }


def _audit_selector(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    selector: str,
    *,
    directions: int,
    fd_eps: float,
    timing_repeats: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    named = selected_named_parameters(model, selector)
    manual_grads, manual_meta = _manual_ce_grad(model, xb, yb, named)
    ref_grads, ref_loss = _autograd_ce_grad(model, xb, yb, named)
    grad_denom = torch.linalg.vector_norm(ref_grads).clamp_min(1.0e-8)
    grad_relerr = float(torch.linalg.vector_norm(manual_grads - ref_grads).div(grad_denom).detach().item()) if manual_grads.numel() == ref_grads.numel() else float("inf")
    grad_cos = float(F.cosine_similarity(manual_grads.flatten(), ref_grads.flatten(), dim=0, eps=1.0e-8).detach().item()) if manual_grads.numel() and ref_grads.numel() else -1.0
    jvp_stats = _directional_jvp_stats(model, xb, yb, named, manual_grads, ref_grads, directions, fd_eps, seed, device)
    func_jvp = _func_jvp_stats(model, xb, yb, named, manual_grads, ref_grads, directions, seed + 310000, device)
    vjp_ms = _time_ms(lambda: _manual_ce_grad(model, xb, yb, named), device, timing_repeats)
    v_flat, chunks = _direction_chunks(named, seed + 900001, device)

    def jvp_once() -> float:
        _add_direction(named, chunks, float(fd_eps))
        loss_plus = _loss_value(model, xb, yb)
        _add_direction(named, chunks, -2.0 * float(fd_eps))
        loss_minus = _loss_value(model, xb, yb)
        _add_direction(named, chunks, float(fd_eps))
        return (loss_plus - loss_minus) / (2.0 * float(fd_eps))

    _ = v_flat
    jvp_ms = _time_ms(jvp_once, device, timing_repeats)
    manual_kernel = model.manual_kernel_variant() if hasattr(model, "manual_kernel_variant") else ""
    uses_dense_basis = getattr(getattr(model, "spec", None), "uses_dense_basis_tensor", "")
    row: dict[str, Any] = {
        "selector": selector,
        "selected_param_count": sum(int(p.numel()) for _name, p in named),
        "manual_kernel_variant": manual_kernel,
        "uses_dense_basis_tensor_spec": uses_dense_basis,
        "uses_dense_output_jacobian_official": 0,
        "autograd_reference_used_for_verification": 1,
        "manual_logits_vjp_available": int(hasattr(model, "manual_logits_backward_from_cache")),
        "ce_loss_manual": manual_meta.get("ce_loss_manual", ""),
        "ce_loss_autograd": ref_loss,
        "ce_loss_abs_error": abs(float(manual_meta.get("ce_loss_manual", float("nan"))) - ref_loss),
        "output_max_abs_error": manual_meta.get("output_max_abs_error", ""),
        "manual_forward_available": int(manual_meta.get("manual_forward_available", 0.0)),
        "manual_backward_available": int(manual_meta.get("manual_backward_available", 0.0)),
        "grad_relerr": grad_relerr,
        "grad_cos": grad_cos,
        "analytic_vs_autograd_rel_error_audit": grad_relerr,
        "finite_diff_eps": fd_eps,
        "sketch_dim": int(directions),
        "sketch_residual_error": func_jvp["func_jvp_relerr_max"] if int(func_jvp["func_jvp_available"]) else jvp_stats["jvp_relerr_max"],
        "basis_jvp_ms": jvp_ms if selector == "basis" else "",
        "basis_vjp_ms": vjp_ms if selector == "basis" else "",
        "readout_jvp_ms": jvp_ms if selector == "readout" else "",
        "readout_vjp_ms": vjp_ms if selector == "readout" else "",
        "all_jvp_ms": jvp_ms if selector == "all" else "",
        "all_vjp_ms": vjp_ms if selector == "all" else "",
        **jvp_stats,
        **func_jvp,
    }
    row["jvp_vjp_gradcheck_pass"] = int(
        int(row["manual_forward_available"]) == 1
        and int(row["manual_backward_available"]) == 1
        and int(row["uses_dense_output_jacobian_official"]) == 0
        and float(row["analytic_vs_autograd_rel_error_audit"]) <= 1.0e-4
        and int(row["func_jvp_available"]) == 1
        and float(row["func_jvp_relerr_max"]) <= 1.0e-4
    )
    row["official_basis_jvp_vjp_gradcheck_pass"] = int(
        selector == "basis"
        and int(row["jvp_vjp_gradcheck_pass"]) == 1
        and str(row["uses_dense_basis_tensor_spec"]) in {"0", "0.0"}
    )
    return row


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    device = _device(args.device)
    datasets = [canonical_dataset_name(x) for x in _split(args.datasets)]
    models = _split(args.models)
    modes = _split(args.implementation_modes)
    seeds = _split(args.seeds, int)
    selectors = _split(args.selectors)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in datasets:
        train_loader, _test_loader, output_dim, input_dim = _get_loaders(
            Path(args.kanbefair_root),
            dataset,
            int(args.batch_size),
            int(args.train_size),
            int(args.test_size),
            int(seeds[0] if seeds else 0),
        )
        first_x, first_y = next(iter(train_loader))
        initial_batch_fingerprint = _batch_fingerprint(first_x, first_y)
        for seed in seeds:
            for model_name in models:
                for mode in modes:
                    try:
                        model = _make_audit_model(
                            model_name,
                            mode,
                            input_dim,
                            output_dim,
                            int(args.hidden),
                            int(seed) + 2217,
                            device,
                            first_x.float(),
                        )
                        warmup_done = _warmup_train(model, train_loader, int(args.warmup_steps), float(args.lr), device)
                        xb, yb = next(iter(train_loader))
                        audit_batch_fingerprint = _batch_fingerprint(xb, yb)
                        xb = xb.to(device).float()
                        yb = yb.to(device).long()
                        model.eval()
                        for selector in selectors:
                            row = _audit_selector(
                                model,
                                xb,
                                yb,
                                selector,
                                directions=int(args.directions),
                                fd_eps=float(args.fd_eps),
                                timing_repeats=int(args.timing_repeats),
                                seed=int(seed) + 7000,
                                device=device,
                            )
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "model_name": model_name,
                                    "implementation_mode": mode,
                                    "input_dim": input_dim,
                                    "output_dim": output_dim,
                                    "hidden": args.hidden,
                                    "train_size": int(args.train_size),
                                    "test_size": int(args.test_size),
                                    "batch_size": int(xb.shape[0]),
                                    "initial_data_batch_fingerprint": initial_batch_fingerprint,
                                    "audit_data_batch_fingerprint": audit_batch_fingerprint,
                                    "warmup_steps_done": warmup_done,
                                    **row,
                                }
                            )
                    except Exception as exc:
                        failures.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model_name": model_name,
                                "implementation_mode": mode,
                                "failure_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
    out_path = _out_path("v22_17_basis_jvp_vjp_gradcheck.csv", args.output_suffix)
    fail_path = _out_path("v22_17_basis_jvp_vjp_gradcheck_failures.csv", args.output_suffix)
    write_rows(out_path, rows if rows else [{"status": "no_rows"}])
    if failures:
        write_rows(fail_path, failures)
    command = " ".join(
        shlex.quote(x)
        for x in [
            sys.executable,
            "experiments/run_v22_17_basis_jvp_vjp_gradcheck.py",
            "--kanbefair-root",
            args.kanbefair_root,
            "--datasets",
            args.datasets,
            "--models",
            args.models,
            "--implementation-modes",
            args.implementation_modes,
            "--seeds",
            args.seeds,
            "--device",
            args.device,
            "--train-size",
            str(args.train_size),
            "--test-size",
            str(args.test_size),
            "--batch-size",
            str(args.batch_size),
            "--hidden",
            str(args.hidden),
            "--warmup-steps",
            str(args.warmup_steps),
            "--lr",
            str(args.lr),
            "--selectors",
            args.selectors,
            "--directions",
            str(args.directions),
            "--fd-eps",
            str(args.fd_eps),
            "--timing-repeats",
            str(args.timing_repeats),
            "--output-suffix",
            args.output_suffix,
        ]
    )
    status = "pass" if rows and not failures else ("partial" if rows else "fail")
    append_exec(
        command,
        task_id="basis-jvp-vjp-gradcheck",
        status=status,
        gpu=args.device,
        exit_code=0 if rows else 1,
        files=str(out_path.relative_to(ROOT)) + (f", {fail_path.relative_to(ROOT)}" if failures else ""),
        note=f"rows={len(rows)}; failures={len(failures)}; no fabricated metrics",
    )


if __name__ == "__main__":
    main()
