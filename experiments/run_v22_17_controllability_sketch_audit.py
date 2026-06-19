#!/usr/bin/env python3
"""Sketched J-space controllability audit for v22.17.

This script is diagnostic: it uses real KANbeFair batches and real long-horizon
training on a small subset, but the subset loop is not promoted as an official
full benchmark.
"""

from __future__ import annotations

import argparse
import copy
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
from experiments.run_v22_16_common import ce_cotangent, selected_named_parameters  # noqa: E402
from experiments.run_v22_17_common import OUT_ROOT, WORKTREE_ROOT, append_exec, ensure_out, write_rows  # noqa: E402
from experiments.run_v22_17_kanbefair_dgkan_eval import _get_loaders, _make_model  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--kanbefair-root", default=str(WORKTREE_ROOT))
    p.add_argument("--datasets", default="MNIST,FMNIST")
    p.add_argument("--models", default="DGMLP")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--collect-steps", type=int, default=80)
    p.add_argument("--history-window", type=int, default=32)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--ridge", type=float, default=1.0e-4)
    p.add_argument("--ratio-cap", type=float, default=0.05)
    p.add_argument("--horizons", default="3200,4800")
    p.add_argument("--selector", default="all")
    p.add_argument("--experiment-tag", default="v22_17_sketched_jspace_h4800_smoke")
    p.add_argument("--output-name", default="v22_17_controllability_sketch_audit.csv")
    p.add_argument("--virtual-loss-gate", action="store_true")
    p.add_argument("--virtual-loss-tolerance", type=float, default=1.0e-5)
    p.add_argument("--controller-compose", choices=["replace", "add_to_base"], default="replace")
    p.add_argument("--controller-line-search", action="store_true")
    p.add_argument("--line-search-scales", default="-1.0,-0.5,0.0,0.25,0.5,1.0")
    p.add_argument("--line-search-loss-tolerance", type=float, default=0.02)
    p.add_argument("--controller-refresh-interval", type=int, default=0)
    p.add_argument("--controller-refresh-max-count", type=int, default=0)
    p.add_argument("--refresh-update-mode", choices=["step_update", "controller_only"], default="step_update")
    p.add_argument("--refresh-scale", type=float, default=1.0)
    p.add_argument(
        "--source-candidate",
        choices=[
            "gradient",
            "split_consensus",
            "split_consensus_positive",
            "jvp_useful_history",
            "jvp_gain_history",
            "jvp_weighted_history",
        ],
        default="gradient",
    )
    p.add_argument(
        "--jspace-target",
        choices=["base_effect", "source_signal", "source_gain_residual", "source_gain_direct"],
        default="base_effect",
    )
    return p


def _split(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flat_grads(named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    parts = []
    for _name, p in named:
        parts.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _unit(v: torch.Tensor) -> torch.Tensor:
    vf = v.detach().float().reshape(-1)
    return vf / torch.linalg.vector_norm(vf).clamp_min(1.0e-12)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    n = min(int(af.numel()), int(bf.numel()))
    if n == 0:
        return 0.0
    af = af[:n]
    bf = bf[:n].to(af.device)
    return float(torch.dot(af, bf).div(torch.linalg.vector_norm(af).clamp_min(1.0e-12) * torch.linalg.vector_norm(bf).clamp_min(1.0e-12)).clamp(-1, 1).item())


def _direction_chunks(
    named: list[tuple[str, torch.nn.Parameter]],
    flat: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    chunks: list[torch.Tensor] = []
    offset = 0
    flat = flat.detach().float().reshape(-1)
    for _name, p in named:
        n = int(p.numel())
        chunks.append(flat[offset : offset + n].to(device=device, dtype=p.dtype).reshape_as(p))
        offset += n
    return tuple(chunks)


def _add_direction(named: list[tuple[str, torch.nn.Parameter]], flat: torch.Tensor, scale: float) -> None:
    chunks = _direction_chunks(named, flat, next(iter(named))[1].device if named else torch.device("cpu"))
    with torch.no_grad():
        for (_name, p), d in zip(named, chunks):
            p.add_(d, alpha=float(scale))


def _assign_grad_update(named: list[tuple[str, torch.nn.Parameter]], update_flat: torch.Tensor) -> None:
    chunks = _direction_chunks(named, update_flat, next(iter(named))[1].device if named else torch.device("cpu"))
    for (_name, p), update in zip(named, chunks):
        grad = -update.to(device=p.device, dtype=p.dtype)
        if p.grad is None:
            p.grad = grad.clone()
        else:
            p.grad.copy_(grad)


def _batch_list(loader: Any, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(x.to(device).float(), y.to(device).long()) for x, y in loader]


def _cycle_batch(batches: list[tuple[torch.Tensor, torch.Tensor]], idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return batches[int(idx) % max(1, len(batches))]


def _train_steps(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    start_idx: int,
    steps: int,
) -> None:
    model.train()
    for local in range(int(steps)):
        xb, yb = _cycle_batch(batches, int(start_idx) + local)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
        opt.step()


def _train_steps_with_refresh(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    selected: list[tuple[str, torch.nn.Parameter]],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    start_idx: int,
    steps: int,
    *,
    update_flat: torch.Tensor,
    selector: str,
    refresh_interval: int,
    refresh_max_count: int,
    refresh_count: int,
    virtual_loss_gate: bool,
    virtual_loss_tolerance: float,
    lr: float,
    weight_decay: float,
    refresh_skipped_count: int,
) -> tuple[int, int]:
    if int(refresh_interval) <= 0:
        _train_steps(model, opt, batches, int(start_idx), int(steps))
        return int(refresh_count), int(refresh_skipped_count)
    for local in range(int(steps)):
        global_step = int(start_idx) + local
        should_refresh = (global_step > int(start_idx)) and ((global_step - int(start_idx)) % int(refresh_interval) == 0)
        if int(refresh_max_count) > 0 and int(refresh_count) >= int(refresh_max_count):
            should_refresh = False
        if should_refresh:
            batch = _cycle_batch(batches, global_step)
            if bool(virtual_loss_gate):
                opt_state = copy.deepcopy(opt.state_dict())
                base_probe = _one_step_with_update(
                    model,
                    opt_state,
                    str(selector),
                    batch,
                    None,
                    float(lr),
                    float(weight_decay),
                )
                ctrl_probe = _one_step_with_update(
                    model,
                    opt_state,
                    str(selector),
                    batch,
                    update_flat,
                    float(lr),
                    float(weight_decay),
                )
                with torch.no_grad():
                    base_loss = float(F.cross_entropy(base_probe(batch[0]).float(), batch[1]).item())
                    ctrl_loss = float(F.cross_entropy(ctrl_probe(batch[0]).float(), batch[1]).item())
                if ctrl_loss > base_loss + float(virtual_loss_tolerance):
                    should_refresh = False
                    refresh_skipped_count += 1
            if not should_refresh:
                _train_steps(model, opt, batches, global_step, 1)
                continue
            _controller_train_step(
                model,
                opt,
                selected_named_parameters(model, str(selector)),
                batch,
                update_flat.to(next(model.parameters()).device),
            )
            refresh_count += 1
        else:
            _train_steps(model, opt, batches, global_step, 1)
    return int(refresh_count), int(refresh_skipped_count)


def _controller_train_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    selected: list[tuple[str, torch.nn.Parameter]],
    batch: tuple[torch.Tensor, torch.Tensor],
    update_flat: torch.Tensor,
) -> None:
    model.train()
    xb, yb = batch
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb).float(), yb)
    loss.backward()
    _assign_grad_update(selected, update_flat)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
    opt.step()


def _one_step_with_update(
    model: torch.nn.Module,
    opt_state: dict[str, Any],
    selector: str,
    batch: tuple[torch.Tensor, torch.Tensor],
    update_flat: torch.Tensor | None,
    lr: float,
    weight_decay: float,
) -> torch.nn.Module:
    probe = copy.deepcopy(model)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    probe_opt.load_state_dict(copy.deepcopy(opt_state))
    if update_flat is None:
        _train_steps(probe, probe_opt, [batch], 0, 1)
    else:
        _controller_train_step(probe, probe_opt, selected_named_parameters(probe, str(selector)), batch, update_flat)
    return probe


def _test_nll(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for xb, yb in batches:
            logits = model(xb).float()
            total_loss += float(F.cross_entropy(logits, yb, reduction="sum").item())
            total_n += int(yb.numel())
    return total_loss / max(1, total_n)


def _pca_residual(target: torch.Tensor, vectors: list[torch.Tensor], dim: int) -> float | str:
    if len(vectors) < 2:
        return ""
    mat = torch.stack([_unit(v).cpu() for v in vectors])
    mat = mat - mat.mean(dim=0, keepdim=True)
    try:
        _u, _s, vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        return ""
    basis = vh[: min(int(dim), int(vh.shape[0]))]
    tgt = _unit(target).cpu()
    coeff = basis @ tgt
    proj = coeff @ basis
    return float(torch.linalg.vector_norm(tgt - proj).div(torch.linalg.vector_norm(tgt).clamp_min(1.0e-12)).item())


def _jvp_logits(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    try:
        from torch.func import functional_call, jvp

        param_map = dict(model.named_parameters())
        names = [name for name, _p in named]
        base_tuple = tuple(param_map[name] for name in names)
        tangent_tuple = _direction_chunks(named, direction, xb.device)

        def logits_fn(*selected_values: torch.Tensor) -> torch.Tensor:
            patched = dict(param_map)
            for name, value in zip(names, selected_values):
                patched[name] = value
            return functional_call(model, patched, (xb,), strict=False).float().reshape(-1)

        _base, tangent = jvp(logits_fn, base_tuple, tangent_tuple)
        return tangent.detach().float().reshape(-1), "torch_func_jvp"
    except Exception:
        eps = 1.0e-3
        with torch.no_grad():
            base = model(xb).detach().float().reshape(-1)
            _add_direction(named, direction, eps)
            moved = model(xb).detach().float().reshape(-1)
            _add_direction(named, direction, -eps)
        return ((moved - base) / eps).detach().float().reshape(-1), "finite_diff_fallback"


def _effect_stats(effect_matrix: torch.Tensor) -> dict[str, Any]:
    if effect_matrix.numel() == 0:
        return {"jacobian_effect_rank": 0, "basis_condition_number": "", "basis_stability_energy": ""}
    try:
        s = torch.linalg.svdvals(effect_matrix.detach().float().cpu())
    except RuntimeError:
        return {"jacobian_effect_rank": "", "basis_condition_number": "", "basis_stability_energy": "", "blocker": "EffectSVDFailed"}
    if s.numel() == 0:
        return {"jacobian_effect_rank": 0, "basis_condition_number": "", "basis_stability_energy": ""}
    smax = float(s.max().item())
    keep = s > max(1.0e-8, smax * 1.0e-4)
    rank = int(keep.sum().item())
    cond = (smax / max(float(s[keep].min().item()), 1.0e-12)) if rank else ""
    return {
        "jacobian_effect_rank": rank,
        "basis_condition_number": cond,
        "basis_stability_energy": float(effect_matrix.detach().float().square().mean().item()),
    }


def _flat_from_grad_list(grads: tuple[torch.Tensor | None, ...], named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    parts = []
    for grad, (_name, p) in zip(grads, named):
        parts.append(torch.zeros_like(p).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _split_consensus_update(
    logits: torch.Tensor,
    yb: torch.Tensor,
    named: list[tuple[str, torch.nn.Parameter]],
) -> tuple[torch.Tensor | None, float | str]:
    if logits.shape[0] < 2 or not named:
        return None, ""
    mid = int(logits.shape[0] // 2)
    if mid <= 0 or mid >= int(logits.shape[0]):
        return None, ""
    params = [p for _name, p in named]
    loss_a = F.cross_entropy(logits[:mid], yb[:mid])
    loss_b = F.cross_entropy(logits[mid:], yb[mid:])
    grads_a = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    grads_b = torch.autograd.grad(loss_b, params, retain_graph=True, allow_unused=True)
    update_a = -_flat_from_grad_list(grads_a, named).detach().float()
    update_b = -_flat_from_grad_list(grads_b, named).detach().float()
    if not update_a.numel() or not update_b.numel():
        return None, ""
    cosine = _cos(update_a, update_b)
    if float(cosine) > 0.0:
        return _unit(update_a) + _unit(update_b), cosine
    return _unit(update_a + update_b), cosine


def _solve_jspace(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    delta: torch.Tensor,
    target: torch.Tensor,
    history: list[torch.Tensor],
    *,
    dim: int,
    ridge: float,
    ratio_cap: float,
    jspace_target: str,
) -> dict[str, Any]:
    cols = [_unit(v).to(xb.device) for v in history[-int(dim) :]]
    target_unit = _unit(target).to(xb.device)
    base_effect, jvp_kind = _jvp_logits(model, named, xb, target_unit)
    source_signal = _unit(-delta.detach().float().reshape(-1)).cpu()
    base_effect_cpu = base_effect.detach().float().cpu().reshape(-1)
    if jspace_target == "source_signal":
        target_effect = source_signal
    elif jspace_target in {"source_gain_residual", "source_gain_direct"}:
        base_unit = _unit(base_effect_cpu)
        coeff = torch.dot(source_signal, base_unit).clamp_min(0.0)
        residual = source_signal - coeff * base_unit
        target_effect = _unit(residual if float(torch.linalg.vector_norm(residual).item()) > 1.0e-12 else source_signal)
    else:
        target_effect = base_effect_cpu
    effects = []
    for vec in cols:
        eff, kind = _jvp_logits(model, named, xb, vec)
        jvp_kind = jvp_kind if jvp_kind == kind else f"{jvp_kind}+{kind}"
        effects.append(eff.cpu())
    if not effects:
        return {"blocker": "InsufficientHistory", "jvp_reference": jvp_kind}
    e_mat = torch.stack(effects, dim=1).float()
    tgt = target_effect.detach().float().cpu().reshape(-1)
    eye = torch.eye(e_mat.shape[1], dtype=e_mat.dtype)
    if jspace_target == "source_gain_direct":
        source_vec = source_signal.to(dtype=e_mat.dtype)
        coeff = e_mat.T @ source_vec
        coeff = coeff / torch.linalg.vector_norm(coeff).clamp_min(1.0e-12)
    else:
        try:
            coeff = torch.linalg.solve(e_mat.T @ e_mat + float(ridge) * eye, e_mat.T @ tgt)
        except RuntimeError:
            coeff = torch.linalg.lstsq(e_mat, tgt).solution
    proj_effect = e_mat @ coeff
    residual = float(torch.linalg.vector_norm(tgt - proj_effect).div(torch.linalg.vector_norm(tgt).clamp_min(1.0e-12)).item())
    u = torch.zeros_like(target_unit.detach().cpu())
    for c, v in zip(coeff, cols):
        u += float(c.item()) * v.detach().cpu()
    u_norm = float(torch.linalg.vector_norm(u).item())
    cap_scale = min(1.0, float(ratio_cap) / max(u_norm, 1.0e-12))
    u_capped = u * cap_scale
    capped_effect, capped_kind = _jvp_logits(model, named, xb, u_capped.to(xb.device))
    jvp_kind = jvp_kind if capped_kind == jvp_kind else f"{jvp_kind}+{capped_kind}"
    source_loss = float((-(delta.detach().float().cpu().reshape(-1) * capped_effect.detach().float().cpu().reshape(-1))).mean().item())
    base_source_loss = float((-(delta.detach().float().cpu().reshape(-1) * base_effect_cpu.reshape(-1))).mean().item())
    gen = torch.Generator(device="cpu").manual_seed(221712)
    random_cols = [_unit(torch.randn(target_unit.numel(), generator=gen, dtype=torch.float32)).to(xb.device) for _ in cols]
    random_effects = [_jvp_logits(model, named, xb, v)[0].cpu() for v in random_cols]
    random_mat = torch.stack(random_effects, dim=1).float()
    try:
        rc = torch.linalg.solve(random_mat.T @ random_mat + float(ridge) * eye, random_mat.T @ tgt)
    except RuntimeError:
        rc = torch.linalg.lstsq(random_mat, tgt).solution
    random_res = float(torch.linalg.vector_norm(tgt - random_mat @ rc).div(torch.linalg.vector_norm(tgt).clamp_min(1.0e-12)).item())
    perm = torch.randperm(int(target_unit.numel()), generator=gen)
    shuffled_cols = [_unit(v.detach().cpu()[perm]).to(xb.device) for v in cols]
    shuffled_effects = [_jvp_logits(model, named, xb, v)[0].cpu() for v in shuffled_cols]
    shuffled_mat = torch.stack(shuffled_effects, dim=1).float()
    try:
        sc = torch.linalg.solve(shuffled_mat.T @ shuffled_mat + float(ridge) * eye, shuffled_mat.T @ tgt)
    except RuntimeError:
        sc = torch.linalg.lstsq(shuffled_mat, tgt).solution
    shuffled_res = float(torch.linalg.vector_norm(tgt - shuffled_mat @ sc).div(torch.linalg.vector_norm(tgt).clamp_min(1.0e-12)).item())
    return {
        "jvp_reference": jvp_kind,
        "jspace_target": jspace_target,
        "dim": len(cols),
        "history_window": len(history),
        "jacobian_reachable_projection_residual": residual,
        "JBa_source_cosine": _cos(proj_effect, tgt),
        "base_effect_source_loss": base_source_loss,
        "JBa_source_loss": source_loss,
        "JBa_source_loss_gain_vs_base_effect": source_loss - base_source_loss,
        "U_task_usefulness_after_projection": source_loss,
        "controller_to_base_update_ratio": float(torch.linalg.vector_norm(u_capped).item()),
        "uncapped_controller_norm": u_norm,
        "controller_cap_scale": cap_scale,
        "random_basis_projection_residual": random_res,
        "shuffled_basis_projection_residual": shuffled_res,
        "controls_pass_count": int(random_res <= residual) + int(shuffled_res <= residual),
        **_effect_stats(e_mat),
        "blocker": "ControllabilityResidualHigh" if residual > 0.35 else "",
        "_controller_update": u_capped,
    }


def _source_metrics(model: torch.nn.Module, xb0: torch.Tensor, logits0: torch.Tensor, delta0: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        effect = model(xb0).detach().float().reshape(-1) - logits0.detach().float().reshape(-1)
    effect_cpu = effect.detach().float().cpu()
    delta_cpu = delta0.detach().float().cpu().reshape(-1)
    source_func = _cos(effect_cpu, -delta_cpu)
    source_loss = float((-(delta_cpu * effect_cpu).mean()).item())
    return source_func, source_loss


def _candidate_source_loss(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    delta: torch.Tensor,
    candidate: torch.Tensor,
) -> float:
    effect, _kind = _jvp_logits(model, named, xb, _unit(candidate).to(xb.device))
    delta_cpu = delta.detach().float().cpu().reshape(-1)
    effect_cpu = effect.detach().float().cpu().reshape(-1)
    return float((-(delta_cpu * effect_cpu)).mean().item())


def _audit_one(dataset: str, seed: int, model_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    train_loader, test_loader, output_dim, input_dim = _get_loaders(
        Path(args.kanbefair_root),
        dataset,
        int(args.batch_size),
        int(args.train_size),
        int(args.test_size),
        int(seed),
    )
    train_batches = _batch_list(train_loader, device)
    test_batches = _batch_list(test_loader, device)
    first_x, _first_y = train_batches[0]
    model = _make_model(model_name, input_dim, output_dim, int(args.hidden), int(seed) + 2217, device, first_x.float()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    selected = selected_named_parameters(model, str(args.selector))
    history: list[torch.Tensor] = []
    history_source_losses: list[float] = []
    split_cosines: list[float] = []
    collect_start = time.perf_counter()
    for step in range(int(args.collect_steps)):
        xb, yb = _cycle_batch(train_batches, step)
        opt.zero_grad(set_to_none=True)
        logits = model(xb).float()
        loss = F.cross_entropy(logits, yb)
        candidate_update: torch.Tensor | None = None
        split_cosine: float | str = ""
        skip_history = False
        if str(args.source_candidate) in {"split_consensus", "split_consensus_positive"}:
            candidate_update, split_cosine = _split_consensus_update(logits, yb, selected)
            if str(args.source_candidate) == "split_consensus_positive" and (not isinstance(split_cosine, float) or float(split_cosine) <= 0.0):
                candidate_update = None
                skip_history = True
        loss.backward()
        grad = _flat_grads(selected)
        if str(args.source_candidate) in {"jvp_useful_history", "jvp_gain_history", "jvp_weighted_history"}:
            candidate_update = -grad.detach().float().cpu() if grad.numel() else None
        elif candidate_update is None and not skip_history:
            candidate_update = -grad.detach().float().cpu()
        else:
            candidate_update = candidate_update.detach().float().cpu() if candidate_update is not None else None
        if isinstance(split_cosine, float):
            split_cosines.append(float(split_cosine))
        if grad.numel() and candidate_update is not None:
            if str(args.source_candidate) in {"jvp_useful_history", "jvp_gain_history", "jvp_weighted_history"}:
                delta_step = ce_cotangent(logits.detach(), yb).detach().float().reshape(-1).cpu()
                candidate_score = _candidate_source_loss(model, selected, xb, delta_step, candidate_update)
                if candidate_score > 0.0:
                    history.append(candidate_update)
                    history_source_losses.append(candidate_score)
            else:
                history.append(candidate_update)
            while len(history) > int(args.history_window):
                history.pop(0)
                if history_source_losses:
                    history_source_losses.pop(0)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
        opt.step()
    collect_sec = time.perf_counter() - collect_start
    xb0, yb0 = _cycle_batch(train_batches, int(args.collect_steps))
    model.eval()
    opt.zero_grad(set_to_none=True)
    logits0 = model(xb0).float()
    loss0 = F.cross_entropy(logits0, yb0)
    target_candidate: torch.Tensor | None = None
    target_split_cosine: float | str = ""
    if str(args.source_candidate) in {"split_consensus", "split_consensus_positive"}:
        target_candidate, target_split_cosine = _split_consensus_update(logits0, yb0, selected)
        if str(args.source_candidate) == "split_consensus_positive" and (
            not isinstance(target_split_cosine, float) or float(target_split_cosine) <= 0.0
        ):
            target_candidate = None
    loss0.backward()
    grad0 = _flat_grads(selected)
    delta0 = ce_cotangent(logits0.detach(), yb0).detach().float().reshape(-1).cpu()
    target_history_source_loss: float | str = ""
    target_current_base_source_loss: float | str = ""
    target_history_gain_vs_current_base: float | str = ""
    if str(args.source_candidate) in {"jvp_useful_history", "jvp_gain_history", "jvp_weighted_history"} and history:
        base_candidate = -grad0.detach().float().cpu()
        target_current_base_source_loss = _candidate_source_loss(model, selected, xb0, delta0, base_candidate) if base_candidate.numel() else ""
        target_scores = [_candidate_source_loss(model, selected, xb0, delta0, h) for h in history]
        if str(args.source_candidate) == "jvp_weighted_history":
            positive = [(i, max(0.0, float(score))) for i, score in enumerate(target_scores) if float(score) > 0.0]
            if positive:
                denom = sum(score for _i, score in positive)
                target = torch.zeros_like(history[positive[0][0]].detach().float().cpu())
                for i, score in positive:
                    target += (score / max(denom, 1.0e-12)) * _unit(history[i]).cpu()
                target_history_source_loss = _candidate_source_loss(model, selected, xb0, delta0, target)
                target_history_gain_vs_current_base = (
                    float(target_history_source_loss) - float(target_current_base_source_loss)
                    if target_current_base_source_loss != ""
                    else ""
                )
            else:
                target = base_candidate
                target_history_source_loss = max(target_scores) if target_scores else ""
        else:
            best_idx = max(range(len(target_scores)), key=lambda i: target_scores[i])
            best_gain = (
                float(target_scores[best_idx]) - float(target_current_base_source_loss)
                if target_current_base_source_loss != ""
                else ""
            )
            target_history_gain_vs_current_base = best_gain
            use_history = float(target_scores[best_idx]) > 0.0
            if str(args.source_candidate) == "jvp_gain_history":
                use_history = use_history and best_gain != "" and float(best_gain) > 0.0
            if use_history:
                target = history[best_idx].detach().float().cpu()
                target_history_source_loss = float(target_scores[best_idx])
            else:
                target = base_candidate
                target_history_source_loss = float(target_scores[best_idx])
    elif str(args.source_candidate) in {"jvp_useful_history", "jvp_gain_history", "jvp_weighted_history"}:
        target = -grad0.detach().float().cpu()
    else:
        target = target_candidate.detach().float().cpu() if target_candidate is not None else -grad0.detach().float().cpu()
    raw_res = _pca_residual(target, history, int(args.dim))
    sketch = _solve_jspace(
        model,
        selected,
        xb0,
        delta0.to(device),
        target,
        history,
        dim=int(args.dim),
        ridge=float(args.ridge),
        ratio_cap=float(args.ratio_cap),
        jspace_target=str(args.jspace_target),
    )
    controller_update = sketch.pop("_controller_update", torch.zeros_like(target))
    step_update = target + controller_update if str(args.controller_compose) == "add_to_base" else controller_update
    line_search_enabled = int(bool(args.controller_line_search))
    line_search_selected_scale: float | str = ""
    line_search_best_train_source_gain: float | str = ""
    line_search_best_virtual_loss_delta: float | str = ""
    if bool(args.controller_line_search):
        opt_state = copy.deepcopy(opt.state_dict())
        base_line_probe = _one_step_with_update(
            model,
            opt_state,
            str(args.selector),
            (xb0, yb0),
            None,
            float(args.lr),
            float(args.weight_decay),
        )
        with torch.no_grad():
            base_line_virtual_loss = float(F.cross_entropy(base_line_probe(xb0).float(), yb0).item())
        _base_line_func, base_line_source_loss = _source_metrics(base_line_probe, xb0, logits0.detach(), delta0)
        best_tuple: tuple[float, float, float, torch.Tensor, torch.Tensor] | None = None
        for scale in _split(args.line_search_scales, float):
            scaled_controller = controller_update * float(scale)
            candidate_step = target + scaled_controller if str(args.controller_compose) == "add_to_base" else scaled_controller
            probe = _one_step_with_update(
                model,
                opt_state,
                str(args.selector),
                (xb0, yb0),
                candidate_step.to(device),
                float(args.lr),
                float(args.weight_decay),
            )
            with torch.no_grad():
                probe_loss = float(F.cross_entropy(probe(xb0).float(), yb0).item())
            _probe_func, probe_source_loss = _source_metrics(probe, xb0, logits0.detach(), delta0)
            loss_delta = probe_loss - base_line_virtual_loss
            source_gain = probe_source_loss - base_line_source_loss
            if loss_delta <= float(args.line_search_loss_tolerance):
                key = (float(source_gain), -abs(float(scale)), -float(loss_delta))
                if best_tuple is None or key > (best_tuple[0], best_tuple[1], best_tuple[2]):
                    best_tuple = (float(source_gain), -abs(float(scale)), -float(loss_delta), scaled_controller, candidate_step)
                    line_search_selected_scale = float(scale)
                    line_search_best_train_source_gain = float(source_gain)
                    line_search_best_virtual_loss_delta = float(loss_delta)
        if best_tuple is not None:
            controller_update = best_tuple[3]
            step_update = best_tuple[4]
            sketch["controller_to_base_update_ratio"] = float(torch.linalg.vector_norm(controller_update.detach().float()).item())
            sketch["controller_cap_scale"] = float(sketch.get("controller_cap_scale", 1.0)) * abs(float(line_search_selected_scale))
    base_probe = copy.deepcopy(model)
    ctrl_probe = copy.deepcopy(model)
    base_probe_opt = torch.optim.AdamW(base_probe.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ctrl_probe_opt = torch.optim.AdamW(ctrl_probe.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    base_probe_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    ctrl_probe_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    _train_steps(base_probe, base_probe_opt, [(xb0, yb0)], 0, 1)
    _controller_train_step(
        ctrl_probe,
        ctrl_probe_opt,
        selected_named_parameters(ctrl_probe, str(args.selector)),
        (xb0, yb0),
        step_update.to(device),
    )
    with torch.no_grad():
        virtual_loss_base = float(F.cross_entropy(base_probe(xb0).float(), yb0).item())
        virtual_loss_controller = float(F.cross_entropy(ctrl_probe(xb0).float(), yb0).item())
    controller_released = int(bool(args.virtual_loss_gate) and virtual_loss_controller > virtual_loss_base + float(args.virtual_loss_tolerance))
    if controller_released:
        controller_update = torch.zeros_like(controller_update)
        step_update = target if str(args.controller_compose) == "add_to_base" else torch.zeros_like(step_update)
        sketch["controller_to_base_update_ratio"] = 0.0
        sketch["controller_cap_scale"] = 0.0
    horizons = sorted(_split(args.horizons, int))
    base_model = copy.deepcopy(model)
    ctrl_model = copy.deepcopy(model)
    base_opt = torch.optim.AdamW(base_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ctrl_opt = torch.optim.AdamW(ctrl_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    base_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    ctrl_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    ctrl_selected = selected_named_parameters(ctrl_model, str(args.selector))
    base_start_nll = _test_nll(base_model, test_batches)
    ctrl_start_nll = _test_nll(ctrl_model, test_batches)
    prev = 0
    horizon_metrics: dict[str, Any] = {}
    refresh_count = 0
    refresh_skipped_count = 0
    long_start = time.perf_counter()
    for horizon in horizons:
        if prev == 0 and int(horizon) > 0:
            _train_steps(base_model, base_opt, train_batches, int(args.collect_steps), 1)
            if controller_released:
                _train_steps(ctrl_model, ctrl_opt, train_batches, int(args.collect_steps), 1)
            else:
                _controller_train_step(
                    ctrl_model,
                    ctrl_opt,
                    ctrl_selected,
                    _cycle_batch(train_batches, int(args.collect_steps)),
                    step_update.to(device),
                )
                refresh_count += 1
            prev = 1
        delta_steps = int(horizon) - int(prev)
        _train_steps(base_model, base_opt, train_batches, int(args.collect_steps) + int(prev), delta_steps)
        refresh_update = controller_update if str(args.refresh_update_mode) == "controller_only" else step_update
        refresh_update = refresh_update * float(args.refresh_scale)
        refresh_count, refresh_skipped_count = _train_steps_with_refresh(
            ctrl_model,
            ctrl_opt,
            ctrl_selected,
            train_batches,
            int(args.collect_steps) + int(prev),
            delta_steps,
            update_flat=refresh_update.to(device),
            selector=str(args.selector),
            refresh_interval=int(args.controller_refresh_interval),
            refresh_max_count=int(args.controller_refresh_max_count),
            refresh_count=int(refresh_count),
            virtual_loss_gate=bool(args.virtual_loss_gate),
            virtual_loss_tolerance=float(args.virtual_loss_tolerance),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            refresh_skipped_count=int(refresh_skipped_count),
        )
        base_func, base_loss = _source_metrics(base_model, xb0, logits0.detach(), delta0)
        ctrl_func, ctrl_loss = _source_metrics(ctrl_model, xb0, logits0.detach(), delta0)
        base_nll = _test_nll(base_model, test_batches)
        ctrl_nll = _test_nll(ctrl_model, test_batches)
        horizon_metrics[f"base_source_func_h{horizon}"] = base_func
        horizon_metrics[f"base_source_loss_h{horizon}"] = base_loss
        horizon_metrics[f"source_func_h{horizon}"] = ctrl_func
        horizon_metrics[f"source_loss_h{horizon}"] = ctrl_loss
        horizon_metrics[f"controller_minus_base_source_loss_h{horizon}"] = ctrl_loss - base_loss
        horizon_metrics[f"base_test_nll_h{horizon}"] = base_nll
        horizon_metrics[f"controller_test_nll_h{horizon}"] = ctrl_nll
        horizon_metrics[f"controller_nll_delta_vs_base_h{horizon}"] = ctrl_nll - base_nll
        prev = int(horizon)
    long_sec = time.perf_counter() - long_start
    source_h4800 = horizon_metrics.get("source_loss_h4800", "")
    source_gain_h4800 = horizon_metrics.get("controller_minus_base_source_loss_h4800", "")
    nll_delta_h4800 = horizon_metrics.get("controller_nll_delta_vs_base_h4800", "")
    plan_mlp_candidate = int(
        model_name.startswith("DGMLP")
        and sketch.get("jacobian_reachable_projection_residual", 1.0) != ""
        and float(sketch.get("jacobian_reachable_projection_residual", 1.0)) <= 0.35
        and source_h4800 != ""
        and float(source_h4800) >= 0.0
        and nll_delta_h4800 != ""
        and float(nll_delta_h4800) <= 0.02
        and int(sketch.get("controls_pass_count", 99)) == 0
        and controller_released == 0
    )
    official_candidate = int(
        plan_mlp_candidate
        and source_gain_h4800 != ""
        and float(source_gain_h4800) > 1.0e-8
    )
    blocker = sketch.get("blocker", "")
    if not official_candidate:
        if source_h4800 == "":
            blocker = "LongHorizonH4800Missing"
        elif float(sketch.get("jacobian_reachable_projection_residual", 1.0)) > 0.35:
            blocker = "ControllabilityResidualHigh"
        elif float(source_h4800) < 0.0:
            blocker = "SourceLossH4800Negative"
        elif source_gain_h4800 != "" and float(source_gain_h4800) <= 1.0e-8:
            blocker = "NoSourceGainVsBase"
        elif nll_delta_h4800 != "" and float(nll_delta_h4800) > 0.02:
            blocker = "TaskNLLDegradation"
        elif int(sketch.get("controls_pass_count", 99)) != 0:
            blocker = "ControlsNotFailed"
        elif controller_released:
            blocker = "VirtualLossGateReleasedController"
    return {
        "dataset": dataset,
        "seed": seed,
        "model_name": model_name,
        "manifold_family": "SketchedJSpaceLeastSquares_torch_func_jvp",
        "selector": args.selector,
        "source_candidate": args.source_candidate,
        "target_split_consensus_cosine": target_split_cosine,
        "mean_history_split_consensus_cosine": (sum(split_cosines) / max(1, len(split_cosines))) if split_cosines else "",
        "target_history_source_loss": target_history_source_loss,
        "mean_history_source_loss": (sum(history_source_losses) / max(1, len(history_source_losses))) if history_source_losses else "",
        "target_current_base_source_loss": target_current_base_source_loss,
        "target_history_gain_vs_current_base": target_history_gain_vs_current_base,
        "experiment_tag": args.experiment_tag,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden": args.hidden,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "collect_steps": args.collect_steps,
        "long_horizon_smoke_subset": 1,
        "optimizer_state_cloned": 1,
        "uses_dense_output_jacobian_official": 0,
        "raw_history_projection_residual": raw_res,
        "useful_source_count": len(history),
        "source_loss_t": float((-(delta0 * _jvp_logits(model, selected, xb0, target.to(device))[0].detach().cpu()).mean()).item()),
        "base_start_nll": base_start_nll,
        "controller_start_nll": ctrl_start_nll,
        "virtual_loss_gate_enabled": int(bool(args.virtual_loss_gate)),
        "virtual_loss_base": virtual_loss_base,
        "virtual_loss_controller": virtual_loss_controller,
        "virtual_loss_delta": virtual_loss_controller - virtual_loss_base,
        "controller_released_by_virtual_loss_gate": controller_released,
        "controller_line_search_enabled": line_search_enabled,
        "line_search_selected_scale": line_search_selected_scale,
        "line_search_best_train_source_gain": line_search_best_train_source_gain,
        "line_search_best_virtual_loss_delta": line_search_best_virtual_loss_delta,
        "controller_refresh_interval": args.controller_refresh_interval,
        "controller_refresh_max_count": args.controller_refresh_max_count,
        "refresh_update_mode": args.refresh_update_mode,
        "refresh_scale": args.refresh_scale,
        "controller_refresh_count": refresh_count,
        "controller_refresh_skipped_count": refresh_skipped_count,
        "controller_compose": args.controller_compose,
        "composed_update_norm": float(torch.linalg.vector_norm(step_update.detach().float()).item()),
        "base_update_norm": float(torch.linalg.vector_norm(target.detach().float()).item()),
        "collect_elapsed_sec": collect_sec,
        "long_horizon_elapsed_sec": long_sec,
        "plan_mlp_controllability_criteria_pass": plan_mlp_candidate,
        "strict_source_gain_over_base_pass": int(source_gain_h4800 != "" and float(source_gain_h4800) > 1.0e-8),
        "mlp_controllability_official_candidate_pass": official_candidate,
        "official_promotion_allowed": 0,
        **{k: v for k, v in sketch.items() if not k.startswith("_")},
        **horizon_metrics,
        "blocker": blocker,
    }


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    device = _device(args.device)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in [canonical_dataset_name(x) for x in _split(args.datasets)]:
        for seed in _split(args.seeds, int):
            for model_name in _split(args.models):
                try:
                    rows.append(_audit_one(dataset, int(seed), model_name, args, device))
                except Exception as exc:
                    failures.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model_name": model_name,
                            "failure_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
    output_path = OUT_ROOT / args.output_name
    write_rows(output_path, rows if rows else [{"status": "no_rows"}])
    failure_path = OUT_ROOT / str(args.output_name).replace(".csv", "_failures.csv")
    if failures:
        write_rows(failure_path, failures)
    elif failure_path.exists():
        failure_path.unlink()
    command = " ".join(
        shlex.quote(x)
        for x in [
            sys.executable,
            "experiments/run_v22_17_controllability_sketch_audit.py",
            "--kanbefair-root",
            args.kanbefair_root,
            "--datasets",
            args.datasets,
            "--models",
            args.models,
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
            "--collect-steps",
            str(args.collect_steps),
            "--history-window",
            str(args.history_window),
            "--dim",
            str(args.dim),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--ridge",
            str(args.ridge),
            "--ratio-cap",
            str(args.ratio_cap),
            "--horizons",
            args.horizons,
            "--selector",
            args.selector,
            "--experiment-tag",
            args.experiment_tag,
            "--output-name",
            args.output_name,
            "--controller-compose",
            args.controller_compose,
            "--source-candidate",
            args.source_candidate,
            "--jspace-target",
            args.jspace_target,
            *(
                [
                    "--controller-refresh-interval",
                    str(args.controller_refresh_interval),
                    "--controller-refresh-max-count",
                    str(args.controller_refresh_max_count),
                    "--refresh-update-mode",
                    args.refresh_update_mode,
                    "--refresh-scale",
                    str(args.refresh_scale),
                ]
                if int(args.controller_refresh_interval) > 0
                else []
            ),
            *(
                [
                    "--controller-line-search",
                    "--line-search-scales",
                    args.line_search_scales,
                    "--line-search-loss-tolerance",
                    str(args.line_search_loss_tolerance),
                ]
                if bool(args.controller_line_search)
                else []
            ),
            *(
                ["--virtual-loss-gate", "--virtual-loss-tolerance", str(args.virtual_loss_tolerance)]
                if bool(args.virtual_loss_gate)
                else []
            ),
        ]
    )
    status = "pass" if rows and not failures else ("partial" if rows else "fail")
    append_exec(
        command,
        task_id="controllability-sketch-audit",
        status=status,
        gpu=args.device,
        exit_code=0 if rows else 1,
        files=str(output_path.relative_to(ROOT)) + (", " + str(failure_path.relative_to(ROOT)) if failures else ""),
        note=f"rows={len(rows)}; failures={len(failures)}; long_horizon_smoke_subset=1",
    )


if __name__ == "__main__":
    main()
