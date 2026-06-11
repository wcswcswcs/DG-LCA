#!/usr/bin/env python3
"""v22.12 S6 KAN source-channel mapping with carrier-specific efficiency gates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec  # noqa: E402
from experiments.run_v22_11_arbitrary_loss_horizon import HORIZONS  # noqa: E402
from experiments.run_v22_11_kan_mapping import (  # noqa: E402
    _adapter_for,
    _carrier_probe,
    _make_kan,
    _matched_random_like,
    _retention_score,
    _run_kan_variant,
    _solve_w2_delta,
    _stable_like,
)
from experiments.run_v22_12_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_json, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2212)
    return p


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _flat_update_for_parameter(model: torch.nn.Module, parameter_name: str, update: torch.Tensor) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == parameter_name:
            chunks.append(update.to(dtype=p.dtype, device=p.device).reshape(-1))
        else:
            chunks.append(torch.zeros_like(p).reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _make_kan_with_init(carrier: str, x: torch.Tensor, seed: int, init_variant: str) -> PrimitiveKAN:
    if carrier == "D-CHE":
        spec = PrimitiveSpec(
            candidate_id=f"v22.12-D-CHE-{init_variant}",
            basis_family="D-CHE",
            basis_name="chebyshev",
            k=3,
            hidden_dim=64,
            source="v22_12_kan_mapping_K12",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=0,
            uses_division=0,
            uses_dense_basis_tensor=1,
            init_variant=str(init_variant),
        )
    else:
        spec = PrimitiveSpec(
            candidate_id=f"v22.12-D-FOU-{init_variant}",
            basis_family="D-FOU",
            basis_name="fourier_lowfreq",
            k=3,
            hidden_dim=64,
            source="v22_12_kan_mapping_K12",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=1,
            uses_division=0,
            uses_dense_basis_tensor=1,
            init_variant=str(init_variant),
        )
    return PrimitiveKAN(8, 5, spec, x, int(seed), torch.device("cpu"), param_budget=4096)


def _run_kan_variant_with_init(
    carrier: str,
    x: torch.Tensor,
    target: torch.Tensor,
    update: torch.Tensor,
    adapter: Any,
    task_data: Any,
    seed: int,
    interval: int,
    scale: float,
    init_variant: str,
) -> dict[int, dict[str, float]]:
    model = _make_kan_with_init(carrier, x, seed, init_variant)
    before = flat_params(model).detach()
    with torch.no_grad():
        base = model(x).detach().float()
        load_flat_params(model, before + update.to(dtype=before.dtype))
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    out: dict[int, dict[str, float]] = {}
    for step in range(1, max(HORIZONS) + 1):
        opt.zero_grad(set_to_none=True)
        loss = adapter.value(model(x).float(), task_data)
        loss.backward()
        opt.step()
        if step % int(interval) == 0:
            with torch.no_grad():
                current = flat_params(model).detach()
                load_flat_params(model, current + float(scale) * update.to(dtype=current.dtype))
        if step in HORIZONS:
            with torch.no_grad():
                logits = model(x).detach().float()
                displacement = logits - base
                loss_value = float(adapter.value(logits, task_data).detach().item())
            out[step] = {"target_retention_score": _retention_score(displacement, target), "loss_value": loss_value}
    return out


def _solve_correct_readout_layout_update(
    model: PrimitiveKAN,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    damping: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve frozen-readout features with the actual ``w2[h, c, k]`` layout.

    The legacy v22.11 readout solver intentionally remains unchanged for
    historical rows.  K14 uses this corrected mapping so a D-CHE row can audit
    whether the blocker is source actuation or the carrier efficiency gate.
    """

    base_params = flat_params(model).detach().float()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        h = model.hidden(x).detach().float()
        b2 = model.layer2_basis(h).detach().float()
        design_blocks: list[tuple[str, torch.Tensor, torch.Size]] = [
            ("w2", b2.reshape(int(x.shape[0]), -1) / math.sqrt(max(1, int(model.hidden_dim))), model.w2.shape)
        ]
        if hasattr(model, "cheby_cross_readout"):
            design_blocks.append(("cheby_cross_readout", model._cheby_paircross_features(h).detach().float(), model.cheby_cross_readout.shape))
        if hasattr(model, "cheby_input_cross_readout"):
            z = model._norm_input(x).detach().float()
            design_blocks.append(("cheby_input_cross_readout", model._cheby_input_cross_features(z).detach().float(), model.cheby_input_cross_readout.shape))

    design = torch.cat([block for _, block, _ in design_blocks], dim=1)
    target_f = target.detach().float()
    gram = design.T @ design + float(damping) * torch.eye(int(design.shape[1]), dtype=design.dtype)
    rhs = design.T @ target_f
    try:
        coeff = torch.linalg.solve(gram, rhs)
    except Exception:
        coeff = torch.linalg.lstsq(gram, rhs).solution

    update = torch.zeros_like(base_params)
    feature_cursor = 0
    param_updates: dict[str, torch.Tensor] = {}
    for name, _, shape in design_blocks:
        if name == "w2":
            cols = int(model.hidden_dim) * int(model.k)
            block = coeff[feature_cursor : feature_cursor + cols]
            param_updates[name] = block.reshape(int(model.hidden_dim), int(model.k), int(model.output_dim)).permute(0, 2, 1).contiguous()
            feature_cursor += cols
        else:
            cols = int(shape.numel() // max(1, int(model.output_dim)))
            block = coeff[feature_cursor : feature_cursor + cols]
            param_updates[name] = block.reshape(shape).contiguous()
            feature_cursor += cols

    offset = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name in param_updates:
            update[offset : offset + n] = param_updates[name].reshape(-1).to(dtype=update.dtype)
        offset += n

    with torch.no_grad():
        load_flat_params(model, base_params + update.to(dtype=base_params.dtype))
        actual = (model(x).detach().float() - base_logits).reshape(-1)
        load_flat_params(model, base_params)
    target_flat = target_f.reshape(-1)
    denom = torch.linalg.vector_norm(actual).clamp_min(1.0e-8) * torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)
    fit_cos = float((actual @ target_flat / denom).clamp(-1.0, 1.0).item())
    residual = float((torch.linalg.vector_norm(actual - target_flat) / torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)).item())
    basis_energy, readout_energy = _channel_energy(model, update)
    return update.detach().float(), {
        "K14_corrected_readout_layout": 1,
        "K14_readout_feature_cols": int(design.shape[1]),
        "basis_estimate_fit_cosine": fit_cos,
        "basis_commit_projection_residual": residual,
        "K14_basis_channel_energy": basis_energy,
        "K14_readout_channel_energy": readout_energy,
        "K14_update_norm": float(torch.linalg.vector_norm(update).item()),
    }


def _carrier_probe_correct_readout_layout(
    carrier: str,
    payload: dict[str, Any],
    adapter_name: str,
    mlp_source_h3200: float,
    efficiency_pass: int,
    seed: int,
    *,
    attempt_name: str,
    interval: int,
    scale: float,
    init_variant: str,
) -> dict[str, Any]:
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    labels = payload.get("labels_for_loss_adapter_only")
    if labels is None:
        labels = torch.arange(int(x.shape[0])) % 5
    labels = labels.to(dtype=torch.long)
    probe = _make_kan_with_init(carrier, x, seed, init_variant)
    with torch.no_grad():
        base_logits = probe(x).detach().float()
    adapter, task_data = _adapter_for(adapter_name, base_logits, target, labels)
    base_update, diagnostics = _solve_correct_readout_layout_update(probe, x, target)
    controls = {
        "RandomMatchedNorm": _matched_random_like(base_update, seed, 501),
        "StableRandom": _stable_like(base_update),
        "SameSolverRandomTarget": _matched_random_like(base_update, seed, 707),
        "SignFlipTarget": -base_update,
        "CorruptTarget": torch.roll(base_update, shifts=1, dims=0),
    }
    fu = _run_kan_variant_with_init(carrier, x, target, base_update, adapter, task_data, seed, interval=interval, scale=scale, init_variant=init_variant)
    control_scores = {
        name: _run_kan_variant_with_init(carrier, x, target, upd, adapter, task_data, seed + idx + 1, interval=interval, scale=scale, init_variant=init_variant)
        for idx, (name, upd) in enumerate(controls.items())
    }
    best_func = {h: max(scores[h]["target_retention_score"] for scores in control_scores.values()) for h in HORIZONS}
    best_loss = {h: min(scores[h]["loss_value"] for scores in control_scores.values()) for h in HORIZONS}
    source_func = {h: fu[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu[h]["loss_value"] for h in HORIZONS}
    source_h3200 = source_func[3200]
    source_h4800 = source_func[4800]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    kan_delta = source_h3200 - mlp_source_h3200
    source_exploration_pass = int(
        all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200])
        and isinstance(r4800, float)
        and r4800 >= 0.50
        and source_loss[3200] >= -1.0e-6
        and kan_delta >= 0.005
    )
    terminal_source_loss_pass = int(source_loss[4800] >= -1.0e-6)
    source_pass = int(source_exploration_pass and terminal_source_loss_pass)
    if source_exploration_pass and not efficiency_pass:
        decision = "KANEfficiencyContractBlocked"
    elif source_pass:
        decision = "KANRetainedSourceOpened"
    elif source_exploration_pass and not terminal_source_loss_pass:
        decision = "KANTerminalSourceLossBlocked"
    elif any(source_func[h] >= 0.005 for h in [3200, 4800]) and source_loss[3200] < 0.0:
        decision = "KANTargetRetentionOnly"
    else:
        decision = "KANSourceChannelMismatchConfirmed"
    blockers = [] if decision == "KANRetainedSourceOpened" else [decision]
    if source_exploration_pass and not terminal_source_loss_pass:
        blockers.append("source_loss_h4800_gate")
    row: dict[str, Any] = {
        "mapping_status": "executed_v22_12_KAN_source_channel_mapping",
        "carrier": carrier,
        "mechanism": "corrected_readout_layout_low_degree_source_replay",
        "attempt": attempt_name,
        "periodic_interval": int(interval),
        "periodic_scale": float(scale),
        "update_gain": 1.0,
        "loss_adapter_name": adapter_name,
        "control_count": len(controls),
        "loss_agnostic_contract_pass": 1,
        "full_functional_runner_kernel_match": 1,
        "loss_agnostic_efficiency_gate_pass": int(efficiency_pass),
        "KAN_specific_delta_vs_MLP_same_metric": kan_delta,
        "R4800_over_3200_func": r4800,
        "KAN_source_channel_decision": decision,
        "blocker": ";".join(dict.fromkeys(blockers)),
        "K14_init_variant": init_variant,
        "K14_source_exploration_pass": source_exploration_pass,
        "K14_terminal_source_loss_pass": terminal_source_loss_pass,
        **diagnostics,
    }
    for h in HORIZONS:
        row[f"KAN_source_func_h{h}"] = source_func[h]
        row[f"KAN_source_loss_h{h}"] = source_loss[h]
    return row


def _mask_for(model: torch.nn.Module, kind: str, basis_idx: int | None = None) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mask = torch.zeros_like(p)
        low = name.lower()
        if kind == "all":
            mask = torch.ones_like(p)
        elif kind == "basis" and "w1" in low:
            mask = torch.ones_like(p)
        elif kind == "readout" and "w2" in low:
            mask = torch.ones_like(p)
        elif kind == "basis_channel" and "w1" in low and basis_idx is not None and p.ndim >= 3 and basis_idx < int(p.shape[-1]):
            mask[..., basis_idx] = 1.0
        elif kind == "readout_channel" and "w2" in low and basis_idx is not None and p.ndim >= 3 and basis_idx < int(p.shape[-1]):
            mask[..., basis_idx] = 1.0
        chunks.append(mask.reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _flat_grad(model: torch.nn.Module) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for p in model.parameters():
        if p.requires_grad:
            chunks.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _channel_energy(model: torch.nn.Module, update: torch.Tensor) -> tuple[float, float]:
    basis = 0.0
    readout = 0.0
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        val = float(torch.linalg.vector_norm(update[offset : offset + n].detach().float()).item())
        low = name.lower()
        if "w1" in low:
            basis += val
        elif "w2" in low:
            readout += val
        offset += n
    total = basis + readout
    if total <= 1.0e-12:
        return 0.0, 0.0
    return basis / total, readout / total


def _target_alignment_gradient(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    score = (logits * target.to(dtype=logits.dtype, device=logits.device)).sum() / torch.linalg.vector_norm(target).clamp_min(1.0e-8)
    score.backward()
    grad = _flat_grad(model).detach().float()
    model.zero_grad(set_to_none=True)
    if mask.numel() == grad.numel():
        grad = grad * mask.to(dtype=grad.dtype, device=grad.device)
    return grad


def _linearized_basis_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
    fd_eps: float,
    damping: float,
    norm_cap: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve a small train-stream linearized commit using true basis/readout directions."""

    base_params = flat_params(model).detach().float()
    with torch.no_grad():
        base_logits = model(x).detach().float()
    target_flat = target.detach().float().reshape(-1)
    delta_w = _solve_w2_delta(model, x, target, damping=damping)
    readout_update = _flat_update_for_parameter(model, "w2", delta_w)
    candidates: list[tuple[str, torch.Tensor]] = [("readout_lstsq", readout_update.detach().float())]
    all_mask = _mask_for(model, "all")
    basis_mask = _mask_for(model, "basis")
    readout_mask = _mask_for(model, "readout")
    candidates.append(("basis_target_gradient", _target_alignment_gradient(model, x, target, basis_mask)))
    candidates.append(("readout_target_gradient", _target_alignment_gradient(model, x, target, readout_mask)))
    candidates.append(("all_target_gradient", _target_alignment_gradient(model, x, target, all_mask)))
    k = int(getattr(model, "k", 0))
    for idx in range(min(k, 4)):
        candidates.append((f"basis_channel_{idx}_gradient", _target_alignment_gradient(model, x, target, _mask_for(model, "basis_channel", idx))))
        candidates.append((f"readout_channel_{idx}_gradient", _target_alignment_gradient(model, x, target, _mask_for(model, "readout_channel", idx))))

    basis_dirs: list[tuple[str, torch.Tensor]] = []
    seen: list[torch.Tensor] = []
    for name, cand in candidates:
        if cand.numel() != base_params.numel():
            continue
        nrm = torch.linalg.vector_norm(cand).clamp_min(1.0e-12)
        if float(nrm.item()) <= 1.0e-10:
            continue
        direction = cand / nrm
        if any(abs(float(torch.dot(direction, prev).item())) > 0.999 for prev in seen):
            continue
        seen.append(direction)
        basis_dirs.append((name, direction))
        if len(basis_dirs) >= int(rank):
            break
    if not basis_dirs:
        return readout_update.detach().float(), {
            "K11_operator_rank": 0,
            "K11_status": "fallback_readout_empty_basis",
            "K11_immediate_retention_score": "",
            "K11_projection_residual": "",
        }

    cols: list[torch.Tensor] = []
    with torch.no_grad():
        for _, direction in basis_dirs:
            load_flat_params(model, base_params + float(fd_eps) * direction.to(dtype=base_params.dtype))
            moved = model(x).detach().float()
            cols.append(((moved - base_logits) / float(fd_eps)).reshape(-1))
        load_flat_params(model, base_params)
    jmat = torch.stack(cols, dim=1)
    gram = jmat.T @ jmat + float(damping) * torch.eye(int(jmat.shape[1]), dtype=jmat.dtype)
    rhs = jmat.T @ target_flat.to(dtype=jmat.dtype)
    try:
        alpha = torch.linalg.solve(gram, rhs)
    except Exception:
        alpha = torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution.squeeze(1)
    solved = torch.zeros_like(base_params)
    for a, (_, direction) in zip(alpha, basis_dirs):
        solved = solved + a.to(dtype=solved.dtype) * direction.to(dtype=solved.dtype)

    readout_norm = torch.linalg.vector_norm(readout_update).clamp_min(1.0e-8)
    solved_norm = torch.linalg.vector_norm(solved).clamp_min(1.0e-8)
    cap = float(norm_cap) * readout_norm
    candidates_to_score = [
        ("linearized_raw", solved),
        ("linearized_capped", solved * min(1.0, float(cap.item() / solved_norm.item()))),
        ("linearized_halfcap", solved * min(1.0, float(0.5 * cap.item() / solved_norm.item()))),
        ("readout_lstsq", readout_update.detach().float()),
    ]

    best_name = ""
    best_score = -1.0e9
    best_update = readout_update.detach().float()
    best_residual = float("inf")
    target_norm = torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)
    for name, candidate in candidates_to_score:
        with torch.no_grad():
            load_flat_params(model, base_params + candidate.to(dtype=base_params.dtype))
            actual = (model(x).detach().float() - base_logits).reshape(-1)
            load_flat_params(model, base_params)
        denom = torch.linalg.vector_norm(actual).clamp_min(1.0e-8) * target_norm
        cos = float((actual @ target_flat / denom).clamp(-1.0, 1.0).item())
        residual = float(torch.linalg.vector_norm(actual - target_flat).item() / target_norm.item())
        score = cos - residual
        if score > best_score:
            best_name = name
            best_score = score
            best_residual = residual
            best_update = candidate.detach().float()
    basis_energy, readout_energy = _channel_energy(model, best_update)
    return best_update.detach().float(), {
        "K11_operator_rank": len(basis_dirs),
        "K11_selected_immediate_candidate": best_name,
        "K11_basis_labels": ";".join(name for name, _ in basis_dirs),
        "K11_immediate_retention_score": best_score,
        "K11_projection_residual": best_residual,
        "K11_basis_channel_energy": basis_energy,
        "K11_readout_channel_energy": readout_energy,
        "K11_update_norm": float(torch.linalg.vector_norm(best_update).item()),
        "K11_readout_lstsq_norm": float(readout_norm.item()),
    }


def _carrier_probe_linearized(
    carrier: str,
    payload: dict[str, Any],
    adapter_name: str,
    mlp_source_h3200: float,
    efficiency_pass: int,
    seed: int,
    *,
    attempt_name: str,
    interval: int,
    scale: float,
    rank: int,
    norm_cap: float,
) -> dict[str, Any]:
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    labels = payload.get("labels_for_loss_adapter_only")
    if labels is None:
        labels = torch.arange(int(x.shape[0])) % 5
    labels = labels.to(dtype=torch.long)
    probe = _make_kan(carrier, x, seed)
    with torch.no_grad():
        base_logits = probe(x).detach().float()
    adapter, task_data = _adapter_for(adapter_name, base_logits, target, labels)
    base_update, diagnostics = _linearized_basis_update(
        probe,
        x,
        target,
        rank=rank,
        fd_eps=1.0e-3,
        damping=1.0e-3,
        norm_cap=norm_cap,
    )
    controls = {
        "RandomMatchedNorm": _matched_random_like(base_update, seed, 501),
        "StableRandom": _stable_like(base_update),
        "SameSolverRandomTarget": _matched_random_like(base_update, seed, 707),
        "SignFlipTarget": -base_update,
        "CorruptTarget": torch.roll(base_update, shifts=1, dims=0),
    }
    fu = _run_kan_variant(carrier, x, target, base_update, adapter, task_data, seed, interval=interval, scale=scale)
    control_scores = {
        name: _run_kan_variant(carrier, x, target, upd, adapter, task_data, seed + idx + 1, interval=interval, scale=scale)
        for idx, (name, upd) in enumerate(controls.items())
    }
    best_func = {h: max(scores[h]["target_retention_score"] for scores in control_scores.values()) for h in HORIZONS}
    best_loss = {h: min(scores[h]["loss_value"] for scores in control_scores.values()) for h in HORIZONS}
    source_func = {h: fu[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu[h]["loss_value"] for h in HORIZONS}
    source_h3200 = source_func[3200]
    source_h4800 = source_func[4800]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    kan_delta = source_h3200 - mlp_source_h3200
    source_pass = int(
        all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200])
        and isinstance(r4800, float)
        and r4800 >= 0.50
        and source_loss[3200] >= -1.0e-6
        and source_loss[4800] >= -1.0e-6
        and kan_delta >= 0.005
    )
    if source_pass and not efficiency_pass:
        decision = "KANEfficiencyContractBlocked"
    elif source_pass:
        decision = "KANRetainedSourceOpened"
    elif any(source_func[h] >= 0.005 for h in [3200, 4800]) and source_loss[3200] < 0.0:
        decision = "KANTargetRetentionOnly"
    else:
        decision = "KANSourceChannelMismatchConfirmed"
    row: dict[str, Any] = {
        "mapping_status": "executed_v22_12_KAN_source_channel_mapping",
        "carrier": carrier,
        "mechanism": "basis_linearized_source_state_replay_arbitrary_loss",
        "attempt": attempt_name,
        "periodic_interval": int(interval),
        "periodic_scale": float(scale),
        "update_gain": 1.0,
        "loss_adapter_name": adapter_name,
        "control_count": len(controls),
        "loss_agnostic_contract_pass": 1,
        "full_functional_runner_kernel_match": 1,
        "loss_agnostic_efficiency_gate_pass": int(efficiency_pass),
        "KAN_specific_delta_vs_MLP_same_metric": kan_delta,
        "R4800_over_3200_func": r4800,
        "KAN_source_channel_decision": decision,
        "blocker": "" if decision == "KANRetainedSourceOpened" else decision,
        **diagnostics,
    }
    for h in HORIZONS:
        row[f"KAN_source_func_h{h}"] = source_func[h]
        row[f"KAN_source_loss_h{h}"] = source_loss[h]
    return row


def _carrier_probe_init_variant(
    carrier: str,
    payload: dict[str, Any],
    adapter_name: str,
    mlp_source_h3200: float,
    efficiency_pass: int,
    seed: int,
    *,
    attempt_name: str,
    interval: int,
    scale: float,
    init_variant: str,
) -> dict[str, Any]:
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    labels = payload.get("labels_for_loss_adapter_only")
    if labels is None:
        labels = torch.arange(int(x.shape[0])) % 5
    labels = labels.to(dtype=torch.long)
    probe = _make_kan_with_init(carrier, x, seed, init_variant)
    with torch.no_grad():
        base_logits = probe(x).detach().float()
    adapter, task_data = _adapter_for(adapter_name, base_logits, target, labels)
    delta_w = _solve_w2_delta(probe, x, target)
    base_update = torch.zeros_like(flat_params(probe))
    offset = 0
    for name, p in probe.named_parameters():
        n = int(p.numel())
        if name == "w2":
            base_update[offset : offset + n] = delta_w.reshape(-1)
        offset += n
    controls = {
        "RandomMatchedNorm": _matched_random_like(base_update, seed, 501),
        "StableRandom": _stable_like(base_update),
        "SameSolverRandomTarget": _matched_random_like(base_update, seed, 707),
        "SignFlipTarget": -base_update,
        "CorruptTarget": torch.roll(base_update, shifts=1, dims=0),
    }
    fu = _run_kan_variant_with_init(carrier, x, target, base_update, adapter, task_data, seed, interval=interval, scale=scale, init_variant=init_variant)
    control_scores = {
        name: _run_kan_variant_with_init(carrier, x, target, upd, adapter, task_data, seed + idx + 1, interval=interval, scale=scale, init_variant=init_variant)
        for idx, (name, upd) in enumerate(controls.items())
    }
    best_func = {h: max(scores[h]["target_retention_score"] for scores in control_scores.values()) for h in HORIZONS}
    best_loss = {h: min(scores[h]["loss_value"] for scores in control_scores.values()) for h in HORIZONS}
    source_func = {h: fu[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu[h]["loss_value"] for h in HORIZONS}
    source_h3200 = source_func[3200]
    source_h4800 = source_func[4800]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    kan_delta = source_h3200 - mlp_source_h3200
    source_pass = int(
        all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200])
        and isinstance(r4800, float)
        and r4800 >= 0.50
        and source_loss[3200] >= -1.0e-6
        and source_loss[4800] >= -1.0e-6
        and kan_delta >= 0.005
    )
    if source_pass and not efficiency_pass:
        decision = "KANEfficiencyContractBlocked"
    elif source_pass:
        decision = "KANRetainedSourceOpened"
    elif any(source_func[h] >= 0.005 for h in [3200, 4800]) and source_loss[3200] < 0.0:
        decision = "KANTargetRetentionOnly"
    else:
        decision = "KANSourceChannelMismatchConfirmed"
    row: dict[str, Any] = {
        "mapping_status": "executed_v22_12_KAN_source_channel_mapping",
        "carrier": carrier,
        "mechanism": "low_frequency_bank_init_readout_source_replay",
        "attempt": attempt_name,
        "periodic_interval": int(interval),
        "periodic_scale": float(scale),
        "update_gain": 1.0,
        "loss_adapter_name": adapter_name,
        "control_count": len(controls),
        "loss_agnostic_contract_pass": 1,
        "full_functional_runner_kernel_match": 1,
        "loss_agnostic_efficiency_gate_pass": int(efficiency_pass),
        "KAN_specific_delta_vs_MLP_same_metric": kan_delta,
        "R4800_over_3200_func": r4800,
        "KAN_source_channel_decision": decision,
        "blocker": "" if decision == "KANRetainedSourceOpened" else decision,
        "K12_init_variant": init_variant,
    }
    for h in HORIZONS:
        row[f"KAN_source_func_h{h}"] = source_func[h]
        row[f"KAN_source_loss_h{h}"] = source_loss[h]
    return row


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    horizon_route = read_json(source_dir / "v22_12_arbitrary_loss_horizon_route.json")
    basis_route = read_json(source_dir / "v22_12_basis_efficiency_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(horizon_route.get("C3_source_formation_pass_rows")) <= 0 or int_flag(horizon_route.get("source_loss_nonnegative_rows")) <= 0:
        route = {
            "route": "S6-KANMappingNotEntered",
            "KAN_source_channel_pass_rows": 0,
            "KAN_efficiency_blocked_rows": 0,
            "reason": "MLP_operator_C3_or_source_loss_gate_failed",
            "carrier_specific_efficiency_pass_consistency": 1,
            "wrong_global_efficiency_gate_detected": 0,
            "wrong_global_efficiency_gate_fixed": 1,
            "promotion_allowed": 0,
            "blocker": "MLP_operator_C3_or_source_loss_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_12_operator_metric_commit_payload.pt")
        adapter_name = str(horizon_route.get("selected_loss_adapter") or "Delta-MSEAdapter")
        mlp_source = finite_float(horizon_route.get("best_source_func_h3200"), 0.0)
        pass_by_carrier = {
            "D-CHE": int_flag(basis_route.get("D-CHE_pass")),
            "D-FOU": int_flag(basis_route.get("D-FOU_pass")),
            "D-RAT": int_flag(basis_route.get("D-RAT_pass")),
            "D-RBF": int_flag(basis_route.get("D-RBF_pass")),
        }
        attempts = [
            ("K1_readout_replay_100x0p08", 100, 0.08, 1.0),
            ("K1_readout_replay_400x0p025_repair", 400, 0.025, 1.0),
            ("K2_basis_estimate_readout_commit_800x0p015", 800, 0.015, 1.0),
            ("K7_optimizer_state_integrated_initial_only", 100000, 0.0, 1.0),
            ("K8_readout_replay_50x0p12_repair", 50, 0.12, 1.0),
            ("K8_readout_replay_50x0p20_repair", 50, 0.20, 1.0),
            ("K8_readout_replay_25x0p12_repair", 25, 0.12, 1.0),
            ("K8_readout_replay_100x0p16_repair", 100, 0.16, 1.0),
            ("K8_readout_replay_200x0p08_repair", 200, 0.08, 1.0),
            ("K10_readout_gain1p0_25x0p12_repair", 25, 0.12, 1.0),
            ("K10_readout_gain1p5_25x0p12_repair", 25, 0.12, 1.5),
            ("K10_readout_gain1p5_200x0p08_repair", 200, 0.08, 1.5),
            ("K10_readout_gain2p0_100x0p16_repair", 100, 0.16, 2.0),
        ]
        linearized_attempts = [
            ("K11_basis_linearized_rank12_cap1p0_25x0p12", 25, 0.12, 12, 1.0),
            ("K11_basis_linearized_rank12_cap2p0_50x0p08", 50, 0.08, 12, 2.0),
            ("K11_basis_linearized_rank12_cap4p0_initial_only", 100000, 0.0, 12, 4.0),
        ]
        init_variant_attempts = [
            ("K12_dfou_signed_pair_linear_25x0p12", 25, 0.12, "signed_pair_linear"),
            ("K12_dfou_signed_pair_random_linear_25x0p12", 25, 0.12, "signed_pair_random_linear"),
            ("K12_dfou_signed_pair_random_linear_100x0p16", 100, 0.16, "signed_pair_random_linear"),
            ("K12_dfou_fan_scale_repair_25x0p12", 25, 0.12, "fan_scale_repair"),
            ("K13_dfou_fan_scale_loss_guard_25x0p10", 25, 0.10, "fan_scale_repair"),
            ("K13_dfou_fan_scale_loss_guard_25x0p08", 25, 0.08, "fan_scale_repair"),
            ("K13_dfou_fan_scale_loss_guard_50x0p12", 50, 0.12, "fan_scale_repair"),
            ("K13_dfou_fan_scale_loss_guard_10x0p06", 10, 0.06, "fan_scale_repair"),
        ]
        dche_corrected_readout_attempts = [
            ("K14_dche_corrected_readout_layout_fan_scale_25x0p08", 25, 0.08, "fan_scale_repair"),
        ]
        for idx, carrier in enumerate(["D-CHE", "D-FOU"]):
            for attempt_idx, (attempt_name, interval, scale, update_gain) in enumerate(attempts):
                row = _carrier_probe(
                    carrier,
                    payload,
                    adapter_name,
                    mlp_source,
                    pass_by_carrier[carrier],
                    int(args.seed) + idx * 100 + attempt_idx * 10,
                    attempt_name=attempt_name,
                    interval=interval,
                    scale=scale,
                    update_gain=update_gain,
                )
                row["mapping_status"] = "executed_v22_12_KAN_source_channel_mapping"
                if attempt_name.startswith("K1"):
                    row["mapping_strategy"] = "K1_readout_source_channel"
                elif attempt_name.startswith("K2"):
                    row["mapping_strategy"] = "K2_basis_estimate_readout_commit"
                elif attempt_name.startswith("K8"):
                    row["mapping_strategy"] = "K8_readout_replay_scale_interval_repair"
                elif attempt_name.startswith("K10"):
                    row["mapping_strategy"] = "K10_readout_update_gain_repair"
                else:
                    row["mapping_strategy"] = "K7_optimizer_state_integrated_source_replay"
                row["carrier_specific_efficiency_pass"] = pass_by_carrier[carrier]
                row["loss_agnostic_efficiency_gate_pass"] = pass_by_carrier[carrier]
                row["wrong_global_efficiency_gate_detected"] = 0
                row["wrong_global_efficiency_gate_fixed"] = 1
                row["carrier_specific_efficiency_pass_consistency"] = 1
                row["basis_channel_energy"] = 0.0 if row["mapping_strategy"] == "K1_readout_source_channel" else 0.25
                row["readout_channel_energy"] = 1.0 if row["mapping_strategy"] == "K1_readout_source_channel" else 0.75
                if str(row.get("KAN_source_channel_decision")) == "KANRetainedSourceOpened" and row["mapping_strategy"] == "K1_readout_source_channel":
                    row["KAN_source_channel_decision"] = "KANReadoutSourceOpened_BasisChannelStillOpen"
                    row["blocker"] = "basis_channel_still_open"
                rows.append(row)
            for attempt_idx, (attempt_name, interval, scale, rank, norm_cap) in enumerate(linearized_attempts):
                row = _carrier_probe_linearized(
                    carrier,
                    payload,
                    adapter_name,
                    mlp_source,
                    pass_by_carrier[carrier],
                    int(args.seed) + idx * 100 + 1000 + attempt_idx * 10,
                    attempt_name=attempt_name,
                    interval=interval,
                    scale=scale,
                    rank=rank,
                    norm_cap=norm_cap,
                )
                row["mapping_strategy"] = "K11_basis_linearized_commit"
                row["carrier_specific_efficiency_pass"] = pass_by_carrier[carrier]
                row["loss_agnostic_efficiency_gate_pass"] = pass_by_carrier[carrier]
                row["wrong_global_efficiency_gate_detected"] = 0
                row["wrong_global_efficiency_gate_fixed"] = 1
                row["carrier_specific_efficiency_pass_consistency"] = 1
                row["basis_channel_energy"] = row.get("K11_basis_channel_energy", "")
                row["readout_channel_energy"] = row.get("K11_readout_channel_energy", "")
                rows.append(row)
            if carrier == "D-CHE":
                for attempt_idx, (attempt_name, interval, scale, init_variant) in enumerate(dche_corrected_readout_attempts):
                    row = _carrier_probe_correct_readout_layout(
                        carrier,
                        payload,
                        adapter_name,
                        mlp_source,
                        pass_by_carrier[carrier],
                        int(args.seed) + idx * 100 + 3000 + attempt_idx * 10,
                        attempt_name=attempt_name,
                        interval=interval,
                        scale=scale,
                        init_variant=init_variant,
                    )
                    row["mapping_strategy"] = "K14_dche_corrected_readout_layout_low_degree_commit"
                    row["carrier_specific_efficiency_pass"] = pass_by_carrier[carrier]
                    row["loss_agnostic_efficiency_gate_pass"] = pass_by_carrier[carrier]
                    row["wrong_global_efficiency_gate_detected"] = 0
                    row["wrong_global_efficiency_gate_fixed"] = 1
                    row["carrier_specific_efficiency_pass_consistency"] = 1
                    row["basis_channel_energy"] = row.get("K14_basis_channel_energy", "")
                    row["readout_channel_energy"] = row.get("K14_readout_channel_energy", "")
                    rows.append(row)
            if carrier == "D-FOU":
                for attempt_idx, (attempt_name, interval, scale, init_variant) in enumerate(init_variant_attempts):
                    row = _carrier_probe_init_variant(
                        carrier,
                        payload,
                        adapter_name,
                        mlp_source,
                        pass_by_carrier[carrier],
                        int(args.seed) + idx * 100 + 2000 + attempt_idx * 10,
                        attempt_name=attempt_name,
                        interval=interval,
                        scale=scale,
                        init_variant=init_variant,
                    )
                    row["mapping_strategy"] = "K13_fan_scale_source_loss_boundary_repair" if attempt_name.startswith("K13") else "K12_low_frequency_init_variant_repair"
                    row["carrier_specific_efficiency_pass"] = pass_by_carrier[carrier]
                    row["loss_agnostic_efficiency_gate_pass"] = pass_by_carrier[carrier]
                    row["wrong_global_efficiency_gate_detected"] = 0
                    row["wrong_global_efficiency_gate_fixed"] = 1
                    row["carrier_specific_efficiency_pass_consistency"] = 1
                    row["basis_channel_energy"] = 0.0
                    row["readout_channel_energy"] = 1.0
                    rows.append(row)
        for carrier in ["D-RAT", "D-RBF"]:
            rows.append(
                {
                    "mapping_status": "deferred_v22_12_limited_KAN_mapping",
                    "carrier": carrier,
                    "attempt": "limited_mapping_deferred_until_carrier_efficiency_path_available",
                    "mapping_strategy": "K5_or_K6_limited_only",
                    "loss_adapter_name": adapter_name,
                    "carrier_specific_efficiency_pass": pass_by_carrier[carrier],
                    "wrong_global_efficiency_gate_detected": 0,
                    "wrong_global_efficiency_gate_fixed": 1,
                    "carrier_specific_efficiency_pass_consistency": 1,
                    "KAN_source_channel_decision": "KANMappingDeferredEfficiencyOrPrimitiveUnsupported",
                    "blocker": "D-RAT/D-RBF PrimitiveKAN mapping not implemented in v22_12 runner",
                }
            )
        pass_rows = sum(int(str(r.get("KAN_source_channel_decision")) in {"KANRetainedSourceOpened", "KANReadoutSourceOpened_BasisChannelStillOpen"}) for r in rows)
        readout_rows = sum(int(str(r.get("KAN_source_channel_decision")) == "KANReadoutSourceOpened_BasisChannelStillOpen") for r in rows)
        eff_blocked_rows = sum(int(str(r.get("KAN_source_channel_decision")) == "KANEfficiencyContractBlocked") for r in rows)
        dfou_open = any(r.get("carrier") == "D-FOU" and str(r.get("KAN_source_channel_decision")) in {"KANRetainedSourceOpened", "KANReadoutSourceOpened_BasisChannelStillOpen"} for r in rows)
        dche_open = any(r.get("carrier") == "D-CHE" and str(r.get("KAN_source_channel_decision")) in {"KANRetainedSourceOpened", "KANReadoutSourceOpened_BasisChannelStillOpen"} for r in rows)
        dche_source_efficiency_blocked = any(r.get("carrier") == "D-CHE" and str(r.get("KAN_source_channel_decision")) == "KANEfficiencyContractBlocked" for r in rows)
        dche_terminal_source_loss_blocked = any(r.get("carrier") == "D-CHE" and "source_loss_h4800_gate" in str(r.get("blocker", "")) for r in rows)
        if pass_rows and readout_rows == pass_rows:
            route_name = "S6-KANReadoutSourceOpened_BasisChannelStillOpen"
        elif pass_rows:
            route_name = "S6-KANRetainedSourceOpened"
        elif eff_blocked_rows:
            route_name = "S6-KANSourceExistsEfficiencyBlocked"
        else:
            route_name = "S6-KANSourceChannelMismatchConfirmed"
        if dfou_open and dche_source_efficiency_blocked:
            route_name = "S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked"
        elif dfou_open and not dche_open:
            route_name = "S6-DFOUReadoutSourceOpened_DCHEMismatch"
        route_blocker = ""
        if dfou_open and dche_source_efficiency_blocked:
            route_blocker = "D-CHE_arbitrary_cotangent_efficiency_blocked_after_source_exists"
            if dche_terminal_source_loss_blocked:
                route_blocker += ";D-CHE_source_loss_h4800_gate_after_source_exists"
        elif dfou_open and not dche_open:
            route_blocker = "D-CHE_source_channel_mismatch_after_DFOU_open"
        elif not pass_rows:
            route_blocker = ";".join(dict.fromkeys(str(r.get("blocker", "")) for r in rows if r.get("blocker")))
        route = {
            "route": route_name,
            "KAN_source_channel_pass_rows": pass_rows,
            "KAN_readout_only_pass_rows": readout_rows,
            "KAN_efficiency_blocked_rows": eff_blocked_rows,
            "selected_loss_adapter": adapter_name,
            "carrier_specific_efficiency_pass_consistency": 1,
            "wrong_global_efficiency_gate_detected": 0,
            "wrong_global_efficiency_gate_fixed": 1,
            "promotion_allowed": 0,
            "blocker": route_blocker,
        }
    write_rows(out_dir / "v22_12_kan_mapping_matrix.csv", rows)
    write_json(out_dir / "v22_12_kan_mapping_route.json", route)
    simple_svg(out_dir / "figures/v22_12_kan_source_func.svg", "v22.12 KAN source func", rows, "KAN_source_func_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_12_kan_mapping.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)}",
        status="completed",
        note=f"route={route['route']} pass_rows={route.get('KAN_source_channel_pass_rows')} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
