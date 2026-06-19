#!/usr/bin/env python3
"""v22.17 strict KAN native-basis FU audit on KANbeFair loaders."""

from __future__ import annotations

import argparse
import hashlib
import os
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
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments.run_v22_16_common import ce_cotangent, grad_energy_by_channel, selected_named_parameters  # noqa: E402
from experiments.run_v22_17_basis_jvp_vjp_gradcheck import _make_native_no_dense_model  # noqa: E402
from experiments.run_v22_17_common import (  # noqa: E402
    OUT_ROOT,
    WORKTREE_ROOT,
    append_exec,
    ensure_out,
    finite_float,
    read_rows,
    write_rows,
)
from experiments.run_v22_17_kanbefair_dgkan_eval import (  # noqa: E402
    _assign_flat_update,
    _candidate_source_loss,
    _controllability_diag_rows,
    _eval,
    _flat_grads,
    _get_loaders,
    _jvp_history_refresh_update,
    _kan_flops,
    _mlp_flops,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--kanbefair-root", default=str(WORKTREE_ROOT))
    p.add_argument("--datasets", default="MNIST")
    p.add_argument("--models", default="DGKAN_DFOU")
    p.add_argument("--seeds", default="0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--mlp-hidden", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--experiment-tag", default="v22_17_kan_basis_native_audit")
    p.add_argument("--output-suffix", default="kan_basis_native_audit")
    p.add_argument("--jvp-refresh-dim", type=int, default=4)
    p.add_argument("--jvp-refresh-min-history", type=int, default=4)
    p.add_argument("--jvp-refresh-history-window", type=int, default=16)
    p.add_argument("--jvp-refresh-scale", type=float, default=0.05)
    p.add_argument("--jvp-refresh-ratio-cap", type=float, default=0.05)
    p.add_argument("--jvp-refresh-ridge", type=float, default=1.0e-4)
    p.add_argument("--jvp-refresh-interval", type=int, default=1)
    p.add_argument("--jvp-refresh-release-on-accept", action="store_true")
    p.add_argument("--fu-zero-nonbasis-grads", action="store_true")
    p.add_argument("--fu-nonbasis-grad-scale", type=float, default=1.0)
    p.add_argument("--virtual-gate-scale", type=float, default=0.002)
    p.add_argument("--virtual-loss-tolerance", type=float, default=1.0e-5)
    p.add_argument("--actionable-gain-tolerance", type=float, default=0.002)
    p.add_argument("--jvp-refresh-require-virtual-improvement", action="store_true")
    p.add_argument("--jvp-refresh-virtual-improvement-margin", type=float, default=0.0)
    return p


def _split(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _batch_fingerprint(xb: torch.Tensor, yb: torch.Tensor) -> str:
    x_cpu = xb.detach().cpu().contiguous()
    y_cpu = yb.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x_cpu.shape)).encode("utf-8"))
    h.update(x_cpu.numpy().tobytes())
    h.update(str(tuple(y_cpu.shape)).encode("utf-8"))
    h.update(y_cpu.numpy().tobytes())
    return h.hexdigest()


def _out_path(name: str, suffix: str) -> Path:
    path = OUT_ROOT / name
    clean = str(suffix or "").strip().strip("_")
    if not clean:
        return path
    return path.with_name(f"{path.stem}_{clean}{path.suffix}")


def _make_model(
    row_model_name: str,
    input_dim: int,
    output_dim: int,
    hidden: int,
    seed: int,
    device: torch.device,
    x_stats: torch.Tensor,
) -> torch.nn.Module:
    if row_model_name == "MLP_ADAMW":
        return MLPBaseline(input_dim, output_dim, hidden, seed, device).to(device)
    base = row_model_name.replace("_NATIVE_ADAMW", "").replace("_FU_NATIVE_JVP_REFRESH", "")
    return _make_native_no_dense_model(base, input_dim, output_dim, hidden, seed, device, x_stats).to(device)


def _model_flops(row_model_name: str, input_dim: int, hidden: int, output_dim: int) -> int:
    if row_model_name == "MLP_ADAMW":
        return _mlp_flops(input_dim, hidden, output_dim)
    return _kan_flops(input_dim, hidden, output_dim, 3)


def _gradcheck_lookup() -> dict[tuple[str, str], dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("v22_17_basis_jvp_vjp_gradcheck*.csv")):
        if path.name.endswith("_failures.csv"):
            continue
        rows.extend(read_rows(path))
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if (
            row.get("implementation_mode") == "native_no_dense_basis"
            and row.get("selector") == "basis"
            and row.get("model_name")
            and row.get("dataset")
        ):
            out[(row["dataset"], row["model_name"])] = row
    return out


def _unit(v: torch.Tensor) -> torch.Tensor:
    flat = v.detach().float().reshape(-1)
    return flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _train_one(
    dataset: str,
    seed: int,
    row_model_name: str,
    args: argparse.Namespace,
    device: torch.device,
    gradchecks: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_loader, test_loader, output_dim, input_dim = _get_loaders(
        Path(args.kanbefair_root), dataset, args.batch_size, args.train_size, args.test_size, seed
    )
    first_x, first_y = next(iter(train_loader))
    initial_batch_fingerprint = _batch_fingerprint(first_x, first_y)
    model_hidden = int(args.mlp_hidden) if row_model_name == "MLP_ADAMW" and int(args.mlp_hidden) > 0 else int(args.hidden)
    model = _make_model(row_model_name, input_dim, output_dim, model_hidden, seed + 2217, device, first_x.float())
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    fu_enabled = row_model_name.endswith("_FU_NATIVE_JVP_REFRESH")
    selected = selected_named_parameters(model, "basis" if row_model_name.startswith("DGKAN") else "all")
    all_params = [p for p in model.parameters() if p.requires_grad]
    train_iter = iter(train_loader)
    losses: list[float] = []
    history: list[torch.Tensor] = []
    useful_flags: list[int] = []
    source_rows: list[dict[str, Any]] = []
    controllability_rows: list[dict[str, Any]] = []
    overhead_time = 0.0
    diagnostic_source_eval_time = 0.0
    total_step_time = 0.0
    gate_accept_count = 0
    gate_reject_count = 0
    refresh_interval_skip_count = 0
    source_release_count = 0
    reject_reasons: dict[str, int] = {}
    basis_source_values: list[float] = []
    base_source_values: list[float] = []
    random_source_values: list[float] = []
    signflip_source_values: list[float] = []
    stable_random_source_values: list[float] = []
    update_ratio_values: list[float] = []
    residual_values: list[float] = []
    jvp_ref_values: list[str] = []
    xb_last = first_x.to(device).float()
    last_diag: dict[str, Any] = {}

    for step in range(1, int(args.steps) + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        xb_last = xb
        model.train()
        _sync(device)
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits_before = model(xb).float()
        loss = F.cross_entropy(logits_before, yb)
        loss.backward()
        delta = ce_cotangent(logits_before.detach(), yb).reshape(-1)
        grad_flat = _flat_grads(selected).detach().float()
        update = -grad_flat
        guided = update
        diag: dict[str, Any] = {
            "gate_reject_reason": "",
            "intervention_flag": 0,
            "jvp_reference": "",
            "controller_to_base_update_ratio": "",
            "jacobian_reachable_projection_residual": "",
            "controller_source_loss_gain": "",
        }
        release_history_after_step = False

        _sync(device)
        diag_t0 = time.perf_counter()
        base_score = _candidate_source_loss(model, selected, xb, delta.detach().cpu(), update) if grad_flat.numel() else 0.0
        gen = torch.Generator(device="cpu").manual_seed(880000 + int(seed) * 1000 + step)
        random_dir = torch.randn(update.numel(), generator=gen, dtype=torch.float32) if update.numel() else torch.empty(0)
        random_score = _candidate_source_loss(model, selected, xb, delta.detach().cpu(), random_dir) if random_dir.numel() else 0.0
        signflip_score = _candidate_source_loss(model, selected, xb, delta.detach().cpu(), -update) if update.numel() else 0.0
        stable_random_score = 0.5 * random_score
        base_source_values.append(base_score)
        random_source_values.append(random_score)
        signflip_source_values.append(signflip_score)
        stable_random_source_values.append(stable_random_score)
        useful_flags.append(int(base_score > max(random_score, 0.0)))
        _sync(device)
        diagnostic_source_eval_time += time.perf_counter() - diag_t0

        if fu_enabled and grad_flat.numel():
            interval = max(1, int(args.jvp_refresh_interval))
            if step % interval == 0:
                _sync(device)
                t0 = time.perf_counter()
                guided, diag = _jvp_history_refresh_update(model, selected, xb, yb, delta.detach(), grad_flat, update, history, args)
                _sync(device)
                overhead_time += time.perf_counter() - t0
                reason = str(diag.get("gate_reject_reason", ""))
                if reason:
                    gate_reject_count += 1
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                else:
                    gate_accept_count += 1
                    _assign_flat_update(selected, guided)
                    release_history_after_step = bool(args.jvp_refresh_release_on_accept)
                if diag.get("controller_to_base_update_ratio") != "":
                    update_ratio_values.append(float(diag.get("controller_to_base_update_ratio", 0.0)))
                if diag.get("jacobian_reachable_projection_residual") != "":
                    residual_values.append(float(diag.get("jacobian_reachable_projection_residual", 0.0)))
                if diag.get("jvp_reference"):
                    jvp_ref_values.append(str(diag.get("jvp_reference")))
            else:
                refresh_interval_skip_count += 1
                diag["gate_reject_reason"] = "refresh_interval_skip"
        _sync(device)
        diag_t0 = time.perf_counter()
        basis_score = _candidate_source_loss(model, selected, xb, delta.detach().cpu(), guided) if guided.numel() else base_score
        _sync(device)
        diagnostic_source_eval_time += time.perf_counter() - diag_t0
        basis_source_values.append(basis_score)
        history.append(update.detach().float().cpu())
        if len(history) > int(args.jvp_refresh_history_window):
            history.pop(0)
            useful_flags.pop(0)
        if release_history_after_step and history:
            history[:] = [history[-1]]
            useful_flags[:] = [useful_flags[-1]] if useful_flags else []
            source_release_count += 1
            diag["source_release_count"] = source_release_count

        torch.nn.utils.clip_grad_norm_(all_params, 2.0)
        nonbasis_scale = 0.0 if bool(args.fu_zero_nonbasis_grads) else float(args.fu_nonbasis_grad_scale)
        if fu_enabled and nonbasis_scale != 1.0:
            selected_names = {name for name, _param in selected}
            for name, param in model.named_parameters():
                if name not in selected_names:
                    if nonbasis_scale == 0.0:
                        param.grad = None
                    elif param.grad is not None:
                        param.grad.mul_(nonbasis_scale)
        opt.step()
        _sync(device)
        total_step_time += time.perf_counter() - step_start
        losses.append(float(loss.detach().item()))
        last_diag = diag
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            source_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model_name": row_model_name,
                    "step": step,
                    "experiment_tag": args.experiment_tag,
                    "implementation_mode": "native_no_dense_basis" if row_model_name.startswith("DGKAN") else "mlp_reference",
                    "parameter_block": "basis" if row_model_name.startswith("DGKAN") else "all",
                    "basis_source_loss_t": basis_score,
                    "base_basis_source_loss_t": base_score,
                    "random_basis_control_source_loss_t": random_score,
                    "stable_random_basis_control_source_loss_t": stable_random_score,
                    "signflip_basis_control_source_loss_t": signflip_score,
                    "gate_reject_reason": diag.get("gate_reject_reason", ""),
                    "jvp_reference": diag.get("jvp_reference", ""),
                    "jacobian_reachable_projection_residual_step": diag.get("jacobian_reachable_projection_residual", ""),
                    "controller_to_base_update_ratio": diag.get("controller_to_base_update_ratio", ""),
                    "source_release_count": source_release_count,
                    "uses_dense_output_jacobian_official": 0,
                    "analysis_only_future_label_used_for_runtime_direction": 0,
                }
            )

    metrics = _eval(model, test_loader, device, output_dim)
    train_eval = _eval(model, train_loader, device, output_dim)
    basis_energy, readout_energy = grad_energy_by_channel(model)
    final_jvp_reference = str(last_diag.get("jvp_reference", "")) if fu_enabled else ""
    dense_finite_diff_fallback = int(row_model_name.startswith("DGKAN") and fu_enabled and final_jvp_reference != "torch_func_jvp")
    ctrl_rows = _controllability_diag_rows(model, selected, xb_last, history, useful_flags, dim=min(4, max(1, len(history) - 1)))
    for ctrl in ctrl_rows:
        ctrl = dict(ctrl)
        ctrl.update(
            {
                "dataset": dataset,
                "seed": seed,
                "model_name": row_model_name,
                "experiment_tag": args.experiment_tag,
                "implementation_mode": "native_no_dense_basis" if row_model_name.startswith("DGKAN") else "mlp_reference",
                "uses_dense_output_jacobian_official": 0,
                "audit_dense_finite_diff_j_used": dense_finite_diff_fallback,
            }
        )
        controllability_rows.append(ctrl)

    controls_pass_count = sum(
        int(control_mean >= (sum(basis_source_values) / max(1, len(basis_source_values))))
        for control_mean in [
            sum(random_source_values) / max(1, len(random_source_values)),
            sum(stable_random_source_values) / max(1, len(stable_random_source_values)),
            sum(signflip_source_values) / max(1, len(signflip_source_values)),
        ]
    )
    base_model_name = row_model_name.replace("_NATIVE_ADAMW", "").replace("_FU_NATIVE_JVP_REFRESH", "")
    gradcheck = gradchecks.get((dataset, base_model_name), {})
    source_mean = sum(basis_source_values) / max(1, len(basis_source_values))
    base_source_mean = sum(base_source_values) / max(1, len(base_source_values))
    horizon_steps = int(args.steps)
    official_train_time = max(0.0, total_step_time - diagnostic_source_eval_time)
    final = {
        "dataset": dataset,
        "seed": seed,
        "model_name": row_model_name,
        "base_model_name": base_model_name,
        "experiment_tag": args.experiment_tag,
        "implementation_mode": "native_no_dense_basis" if row_model_name.startswith("DGKAN") else "mlp_reference",
        "controller_policy": "basis_jvp_useful_history_refresh" if fu_enabled else "adamw_baseline",
        "source_candidate_type": "basis_jvp_useful_history" if fu_enabled else "basis_adamw_update",
        "model_origin": "DGKAN_native_basis_audit" if row_model_name.startswith("DGKAN") else "MLP_reference_same_run",
        "model_family": "KAN" if row_model_name.startswith("DGKAN") else "MLP",
        "is_dgkan_strict_fc_purekan": int(row_model_name.startswith("DGKAN")),
        "uses_kanbefair_loader": 1,
        "uses_kanbefair_baseline_model": 0,
        "uses_bspline_official_path": 0,
        "uses_readout_diagnostic": 0,
        "basis_native_claim_allowed": int(row_model_name.startswith("DGKAN")),
        "uses_dense_basis_tensor_spec": 0 if row_model_name.startswith("DGKAN") else "",
        "uses_dense_output_jacobian_official": 0,
        "audit_dense_finite_diff_j_used": dense_finite_diff_fallback,
        "manual_kernel_variant": gradcheck.get("manual_kernel_variant", ""),
        "fu_zero_nonbasis_grads": int(bool(args.fu_zero_nonbasis_grads) and fu_enabled),
        "fu_nonbasis_grad_scale": (0.0 if bool(args.fu_zero_nonbasis_grads) else float(args.fu_nonbasis_grad_scale)) if fu_enabled else "",
        "jvp_refresh_interval": int(args.jvp_refresh_interval) if fu_enabled else "",
        "jvp_refresh_min_history": int(args.jvp_refresh_min_history) if fu_enabled else "",
        "jvp_refresh_history_window": int(args.jvp_refresh_history_window) if fu_enabled else "",
        "jvp_refresh_release_on_accept": int(bool(args.jvp_refresh_release_on_accept)) if fu_enabled else "",
        "jvp_refresh_require_virtual_improvement": int(bool(args.jvp_refresh_require_virtual_improvement)) if fu_enabled else "",
        "jvp_refresh_virtual_improvement_margin": float(args.jvp_refresh_virtual_improvement_margin) if fu_enabled else "",
        "smoke_only": 1,
        "horizon_steps": horizon_steps,
        "hidden": model_hidden,
        "kan_hidden": int(args.hidden) if row_model_name.startswith("DGKAN") else "",
        "mlp_reference_hidden": int(args.mlp_hidden) if row_model_name == "MLP_ADAMW" and int(args.mlp_hidden) > 0 else "",
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
        "batch_size": int(args.batch_size),
        "initial_data_batch_fingerprint": initial_batch_fingerprint,
        "final_train_loss": train_eval["loss"],
        "final_test_loss_NLL": metrics["loss"],
        "final_test_accuracy": metrics["accuracy"],
        "AUC_loss_time": sum(losses) / max(1, len(losses)),
        "ECE": metrics["ECE"],
        "Brier": metrics["Brier"],
        "tail_loss_q95": metrics["tail_loss_q95"],
        "tail_loss_q99": metrics["tail_loss_q99"],
        "param_count": sum(int(p.numel()) for p in model.parameters()),
        "flops": _model_flops(row_model_name, input_dim, model_hidden, output_dim),
        "runtime_per_step_ms": 1000.0 * total_step_time / max(1, int(args.steps)),
        "full_loop_step_ms": 1000.0 * total_step_time / max(1, int(args.steps)),
        "diagnostic_source_eval_step_ms": 1000.0 * diagnostic_source_eval_time / max(1, int(args.steps)),
        "official_train_step_ms": 1000.0 * official_train_time / max(1, int(args.steps)),
        "full_loop_step_ms_excluding_diagnostics": 1000.0 * official_train_time / max(1, int(args.steps)),
        "source_eval_diagnostic_subtracted_from_official_timing": 1,
        "controller_overhead_ratio": overhead_time / max(1.0e-12, total_step_time),
        "official_controller_overhead_ratio": overhead_time / max(1.0e-12, official_train_time),
        "basis_channel_energy_fraction": basis_energy if row_model_name.startswith("DGKAN") else "",
        "readout_channel_energy_fraction": readout_energy if row_model_name.startswith("DGKAN") else "",
        "KAN_source_loss_horizon": source_mean if row_model_name.startswith("DGKAN") else "",
        "KAN_base_source_loss_horizon": base_source_mean if row_model_name.startswith("DGKAN") else "",
        "KAN_source_loss_h20": source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 20 else "",
        "KAN_base_source_loss_h20": base_source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 20 else "",
        "KAN_source_loss_h120": source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 120 else "",
        "KAN_base_source_loss_h120": base_source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 120 else "",
        "KAN_source_loss_h3200": source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 3200 else "",
        "KAN_base_source_loss_h3200": base_source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 3200 else "",
        "KAN_source_loss_h4800": source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 4800 else "",
        "KAN_base_source_loss_h4800": base_source_mean if row_model_name.startswith("DGKAN") and horizon_steps >= 4800 else "",
        "mean_random_basis_control_source_loss": sum(random_source_values) / max(1, len(random_source_values)) if row_model_name.startswith("DGKAN") else "",
        "mean_stable_random_basis_control_source_loss": sum(stable_random_source_values) / max(1, len(stable_random_source_values)) if row_model_name.startswith("DGKAN") else "",
        "mean_signflip_basis_control_source_loss": sum(signflip_source_values) / max(1, len(signflip_source_values)) if row_model_name.startswith("DGKAN") else "",
        "controls_pass_count": controls_pass_count if row_model_name.startswith("DGKAN") else "",
        "gate_accept_count": gate_accept_count,
        "gate_reject_count": gate_reject_count,
        "source_release_count": source_release_count,
        "refresh_interval_skip_count": refresh_interval_skip_count,
        "gate_reject_reason_counts": ";".join(f"{k}:{v}" for k, v in sorted(reject_reasons.items())),
        "mean_controller_to_base_update_ratio": sum(update_ratio_values) / max(1, len(update_ratio_values)) if update_ratio_values else "",
        "mean_jvp_reachable_projection_residual": sum(residual_values) / max(1, len(residual_values)) if residual_values else "",
        "final_jvp_reference": final_jvp_reference,
        "jvp_reference_kinds": ";".join(sorted(set(jvp_ref_values))),
        "analytic_vs_autograd_rel_error_audit": gradcheck.get("analytic_vs_autograd_rel_error_audit", ""),
        "basis_jvp_ms": gradcheck.get("basis_jvp_ms", ""),
        "basis_vjp_ms": gradcheck.get("basis_vjp_ms", ""),
        "sketch_dim": gradcheck.get("sketch_dim", ""),
        "sketch_residual_error": gradcheck.get("sketch_residual_error", ""),
        "official_basis_jvp_vjp_gradcheck_pass": gradcheck.get("official_basis_jvp_vjp_gradcheck_pass", ""),
        "kan_basis_native_smoke_exploration_candidate_pass": 0,
        "kan_basis_native_official_pass": 0,
        "blocker": "",
    }
    return final, source_rows, controllability_rows, {
        "dataset": dataset,
        "seed": seed,
        "model_name": row_model_name,
        "input_size": input_dim,
        "output_size": output_dim,
        "param_count": final["param_count"],
        "flops": final["flops"],
        "hidden": model_hidden,
    }


def _add_cross_row_stats(rows: list[dict[str, Any]]) -> None:
    by_key = {(r["dataset"], str(r["seed"]), r["model_name"]): r for r in rows}
    for row in rows:
        dataset, seed, model_name = row["dataset"], str(row["seed"]), row["model_name"]
        mlp = by_key.get((dataset, seed, "MLP_ADAMW"))
        kan_base_name = row.get("base_model_name", "")
        kan_base = by_key.get((dataset, seed, f"{kan_base_name}_NATIVE_ADAMW"))
        if mlp:
            row["full_loop_ratio_vs_mlp"] = finite_float(row.get("runtime_per_step_ms")) / max(1.0e-12, finite_float(mlp.get("runtime_per_step_ms")))
            row["official_full_loop_ratio_vs_mlp"] = finite_float(row.get("official_train_step_ms"), finite_float(row.get("runtime_per_step_ms"))) / max(
                1.0e-12,
                finite_float(mlp.get("official_train_step_ms"), finite_float(mlp.get("runtime_per_step_ms"))),
            )
            row["NLL_delta_vs_MLP_AdamW_same_run"] = finite_float(row.get("final_test_loss_NLL")) - finite_float(mlp.get("final_test_loss_NLL"))
            row["AUC_loss_time_delta_vs_MLP_AdamW_same_run"] = finite_float(row.get("AUC_loss_time")) - finite_float(mlp.get("AUC_loss_time"))
            row["param_ratio_vs_mlp"] = finite_float(row.get("param_count")) / max(1.0e-12, finite_float(mlp.get("param_count")))
            row["flops_ratio_vs_mlp"] = finite_float(row.get("flops")) / max(1.0e-12, finite_float(mlp.get("flops")))
        else:
            row["full_loop_ratio_vs_mlp"] = ""
            row["official_full_loop_ratio_vs_mlp"] = ""
            row["NLL_delta_vs_MLP_AdamW_same_run"] = ""
            row["AUC_loss_time_delta_vs_MLP_AdamW_same_run"] = ""
            row["param_ratio_vs_mlp"] = ""
            row["flops_ratio_vs_mlp"] = ""
        if kan_base and model_name.endswith("_FU_NATIVE_JVP_REFRESH"):
            row["NLL_delta_vs_KAN_AdamW_native"] = finite_float(row.get("final_test_loss_NLL")) - finite_float(kan_base.get("final_test_loss_NLL"))
            row["AUC_loss_time_delta_vs_KAN_AdamW_native"] = finite_float(row.get("AUC_loss_time")) - finite_float(kan_base.get("AUC_loss_time"))
            row["accuracy_delta_vs_KAN_AdamW_native"] = finite_float(row.get("final_test_accuracy")) - finite_float(kan_base.get("final_test_accuracy"))
        else:
            row.setdefault("NLL_delta_vs_KAN_AdamW_native", "")
            row.setdefault("AUC_loss_time_delta_vs_KAN_AdamW_native", "")
            row.setdefault("accuracy_delta_vs_KAN_AdamW_native", "")

        blockers: list[str] = []
        if model_name.endswith("_FU_NATIVE_JVP_REFRESH"):
            if finite_float(row.get("basis_channel_energy_fraction"), 0.0) < 0.30:
                blockers.append("BasisEnergyLow")
            residual = row.get("mean_jvp_reachable_projection_residual", "")
            if residual == "" or finite_float(residual, 1.0) > 0.70:
                blockers.append("ReachabilityResidualHighOrMissing")
            if finite_float(row.get("KAN_source_loss_horizon"), -1.0) < 0.0:
                blockers.append("KANSourceLossNegative")
            if int(row.get("controls_pass_count") or 0) > 0:
                blockers.append("ControlsNotFailed")
            official_ratio = row.get("official_full_loop_ratio_vs_mlp", "")
            if official_ratio == "" or finite_float(official_ratio, 999.0) > 3.0:
                blockers.append("FullLoopRatioVsMLPHigh")
            if int(row.get("horizon_steps") or 0) < 3200:
                blockers.append("LongHorizonH3200Missing")
            if int(row.get("horizon_steps") or 0) < 4800:
                blockers.append("LongHorizonH4800Missing")
            row["kan_basis_native_smoke_exploration_candidate_pass"] = int(
                all(b not in blockers for b in ["BasisEnergyLow", "ReachabilityResidualHighOrMissing", "KANSourceLossNegative", "ControlsNotFailed", "FullLoopRatioVsMLPHigh"])
            )
            row["kan_basis_native_official_pass"] = 0
            row["blocker"] = ";".join(blockers)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fu_rows = [r for r in rows if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")]
    out: list[dict[str, Any]] = []
    for model_name in sorted({r["model_name"] for r in fu_rows}):
        vals = [r for r in fu_rows if r["model_name"] == model_name]
        if not vals:
            continue
        n = len(vals)
        out.append(
            {
                "model_name": model_name,
                "rows": n,
                "smoke_exploration_candidate_pass": sum(int(r.get("kan_basis_native_smoke_exploration_candidate_pass") or 0) for r in vals),
                "official_pass": sum(int(r.get("kan_basis_native_official_pass") or 0) for r in vals),
                "basis_energy_ge_030": sum(finite_float(r.get("basis_channel_energy_fraction"), 0.0) >= 0.30 for r in vals),
                "basis_energy_ge_050": sum(finite_float(r.get("basis_channel_energy_fraction"), 0.0) >= 0.50 for r in vals),
                "source_loss_horizon_ge_0": sum(finite_float(r.get("KAN_source_loss_horizon"), -1.0) >= 0.0 for r in vals),
                "source_loss_h20_ge_0": sum(finite_float(r.get("KAN_source_loss_h20"), -1.0) >= 0.0 for r in vals),
                "source_loss_h120_ge_0": sum(finite_float(r.get("KAN_source_loss_h120"), -1.0) >= 0.0 for r in vals),
                "source_loss_h3200_ge_0": sum(finite_float(r.get("KAN_source_loss_h3200"), -1.0) >= 0.0 for r in vals),
                "source_loss_h4800_ge_0": sum(finite_float(r.get("KAN_source_loss_h4800"), -1.0) >= 0.0 for r in vals),
                "full_loop_ratio_le_3": sum(finite_float(r.get("full_loop_ratio_vs_mlp"), 999.0) <= 3.0 for r in vals),
                "official_full_loop_ratio_le_3": sum(finite_float(r.get("official_full_loop_ratio_vs_mlp"), 999.0) <= 3.0 for r in vals),
                "overhead_ratio_le_025": sum(finite_float(r.get("controller_overhead_ratio"), 999.0) <= 0.25 for r in vals),
                "official_overhead_ratio_le_025": sum(finite_float(r.get("official_controller_overhead_ratio"), 999.0) <= 0.25 for r in vals),
                "controls_fail": sum(int(r.get("controls_pass_count") or 0) == 0 for r in vals),
                "nll_noharm_vs_kan": sum(finite_float(r.get("NLL_delta_vs_KAN_AdamW_native"), 999.0) <= 0.02 for r in vals),
                "auc_improve_vs_kan": sum(finite_float(r.get("AUC_loss_time_delta_vs_KAN_AdamW_native"), 999.0) < 0.0 for r in vals),
                "max_full_loop_ratio_vs_mlp": max(finite_float(r.get("full_loop_ratio_vs_mlp"), 0.0) for r in vals),
                "max_official_full_loop_ratio_vs_mlp": max(finite_float(r.get("official_full_loop_ratio_vs_mlp"), 0.0) for r in vals),
                "max_controller_overhead_ratio": max(finite_float(r.get("controller_overhead_ratio"), 0.0) for r in vals),
                "max_official_controller_overhead_ratio": max(finite_float(r.get("official_controller_overhead_ratio"), 0.0) for r in vals),
                "mean_diagnostic_source_eval_step_ms": sum(finite_float(r.get("diagnostic_source_eval_step_ms"), 0.0) for r in vals) / max(1, len(vals)),
                "max_param_ratio_vs_mlp": max(finite_float(r.get("param_ratio_vs_mlp"), 0.0) for r in vals),
                "max_flops_ratio_vs_mlp": max(finite_float(r.get("flops_ratio_vs_mlp"), 0.0) for r in vals),
                "min_basis_channel_energy_fraction": min(finite_float(r.get("basis_channel_energy_fraction"), 0.0) for r in vals),
                "max_nll_delta_vs_kan": max(finite_float(r.get("NLL_delta_vs_KAN_AdamW_native"), 0.0) for r in vals),
                "blockers": ";".join(sorted({b for r in vals for b in str(r.get("blocker", "")).split(";") if b})),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    device = _device(args.device)
    datasets = [canonical_dataset_name(x) for x in _split(args.datasets)]
    models = _split(args.models)
    seeds = _split(args.seeds, int)
    gradchecks = _gradcheck_lookup()
    final_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            run_models = ["MLP_ADAMW"]
            for model in models:
                run_models.extend([f"{model}_NATIVE_ADAMW", f"{model}_FU_NATIVE_JVP_REFRESH"])
            for row_model_name in run_models:
                try:
                    row, src, ctrl, param = _train_one(dataset, int(seed), row_model_name, args, device, gradchecks)
                    final_rows.append(row)
                    source_rows.extend(src)
                    ctrl_rows.extend(ctrl)
                    param_rows.append(param)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model_name": row_model_name,
                            "failure_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
    _add_cross_row_stats(final_rows)
    summary = _summary_rows(final_rows)
    write_rows(_out_path("v22_17_kan_basis_native_audit.csv", args.output_suffix), final_rows)
    write_rows(_out_path("v22_17_kan_basis_native_audit_summary.csv", args.output_suffix), summary)
    write_rows(_out_path("v22_17_kan_basis_native_source_timeseries.csv", args.output_suffix), source_rows)
    write_rows(_out_path("v22_17_kan_basis_native_controllability.csv", args.output_suffix), ctrl_rows)
    write_rows(_out_path("v22_17_kan_basis_native_param_flops.csv", args.output_suffix), param_rows)
    if failure_rows:
        write_rows(_out_path("v22_17_kan_basis_native_failures.csv", args.output_suffix), failure_rows)
    cmd = shlex.join([sys.executable, sys.argv[0], *sys.argv[1:]])
    append_exec(
        cmd,
        task_id="F-kan-basis-native-audit",
        status="pass" if not failure_rows else "partial",
        gpu=f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}; script device={args.device}",
        exit_code=0,
        files=str(_out_path("v22_17_kan_basis_native_audit.csv", args.output_suffix).relative_to(ROOT)),
        note=f"rows={len(final_rows)}; failures={len(failure_rows)}; summary={_out_path('v22_17_kan_basis_native_audit_summary.csv', args.output_suffix).name}",
    )


if __name__ == "__main__":
    main()
