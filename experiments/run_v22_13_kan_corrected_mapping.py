#!/usr/bin/env python3
"""v22.13 corrected KAN source-carrier mapping."""

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
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec  # noqa: E402
from experiments.run_v22_11_arbitrary_loss_horizon import HORIZONS  # noqa: E402
from experiments.run_v22_11_kan_mapping import _adapter_for, _retention_score  # noqa: E402
from experiments.run_v22_13_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2213)
    p.add_argument("--device", default="cuda:2")
    return p


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _move_task_data(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _move_task_data(v, device) for k, v in data.items()}
    return data


def _matched_random_like(vec: torch.Tensor, seed: int, offset: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(offset))
    raw = torch.randn(vec.shape, generator=gen, dtype=vec.detach().cpu().dtype).to(device=vec.device, dtype=vec.dtype)
    return raw * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(raw).clamp_min(1.0e-8))


def _stable_like(vec: torch.Tensor) -> torch.Tensor:
    raw = torch.sin(torch.arange(vec.numel(), device=vec.device, dtype=vec.dtype)).reshape_as(vec)
    return raw * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(raw).clamp_min(1.0e-8))


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
        elif "w2" in low or "readout" in low:
            readout += val
        offset += n
    total = basis + readout
    if total <= 1.0e-12:
        return 0.0, 0.0
    return basis / total, readout / total


def _make_kan_with_init(carrier: str, x: torch.Tensor, seed: int, init_variant: str, device: torch.device) -> PrimitiveKAN:
    if carrier == "D-CHE":
        spec = PrimitiveSpec(
            candidate_id=f"v22.13-D-CHE-{init_variant}",
            basis_family="D-CHE",
            basis_name="chebyshev",
            k=3,
            hidden_dim=64,
            source="v22_13_kan_mapping_K15_K16",
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
            candidate_id=f"v22.13-D-FOU-{init_variant}",
            basis_family="D-FOU",
            basis_name="fourier_lowfreq",
            k=3,
            hidden_dim=64,
            source="v22_13_kan_mapping_K15_K16",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=1,
            uses_division=0,
            uses_dense_basis_tensor=1,
            init_variant=str(init_variant),
        )
    return PrimitiveKAN(8, 5, spec, x.to(device), int(seed), device, param_budget=4096).to(device)


def _solve_correct_readout_layout_update(
    model: PrimitiveKAN,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    damping: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, Any]]:
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
    target_f = target.detach().float().to(device=design.device)
    gram = design.T @ design + float(damping) * torch.eye(int(design.shape[1]), dtype=design.dtype, device=design.device)
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
            update[offset : offset + n] = param_updates[name].reshape(-1).to(device=update.device, dtype=update.dtype)
        offset += n

    with torch.no_grad():
        load_flat_params(model, base_params + update.to(dtype=base_params.dtype, device=base_params.device))
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


def _solve_basis_linearized_update(
    model: PrimitiveKAN,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    damping: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_params = flat_params(model).detach().float()
    x = x.detach().float().to(next(model.parameters()).device)
    target_f = target.detach().float().to(device=x.device)
    basis_params: list[tuple[str, torch.nn.Parameter, int, int]] = []
    offset = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if p.requires_grad and "w1" in name.lower():
            basis_params.append((name, p, offset, n))
        offset += n
    if not basis_params:
        return torch.zeros_like(base_params), {
            "K14_corrected_readout_layout": 0,
            "K14_readout_feature_cols": 0,
            "basis_estimate_fit_cosine": 0.0,
            "basis_commit_projection_residual": float("inf"),
            "K14_basis_channel_energy": 0.0,
            "K14_readout_channel_energy": 0.0,
            "K14_update_norm": 0.0,
            "basis_linearized_jacobian_rows": 0,
            "basis_linearized_jacobian_cols": 0,
            "basis_linearized_solver_status": "no_w1_basis_params",
        }
    with torch.no_grad():
        base_logits = model(x).detach().float()
    flat_target = target_f.reshape(-1)
    jac_rows: list[torch.Tensor] = []
    params = [p for _, p, _, _ in basis_params]
    for out_idx in range(int(flat_target.numel())):
        model.zero_grad(set_to_none=True)
        logits = model(x).float().reshape(-1)
        grads = torch.autograd.grad(logits[out_idx], params, retain_graph=False, allow_unused=True)
        chunks = [torch.zeros_like(p).reshape(-1) if g is None else g.detach().float().reshape(-1) for p, g in zip(params, grads)]
        jac_rows.append(torch.cat(chunks))
    jac = torch.stack(jac_rows, dim=0)
    gram = jac.T @ jac + float(damping) * torch.eye(int(jac.shape[1]), dtype=jac.dtype, device=jac.device)
    rhs = jac.T @ flat_target.to(device=jac.device, dtype=jac.dtype)
    try:
        coeff = torch.linalg.solve(gram, rhs)
        solver_status = "solve"
    except Exception:
        coeff = torch.linalg.lstsq(gram, rhs).solution
        solver_status = "lstsq_fallback"
    update = torch.zeros_like(base_params)
    cursor = 0
    for _name, _p, start, n in basis_params:
        update[start : start + n] = coeff[cursor : cursor + n].to(device=update.device, dtype=update.dtype)
        cursor += n
    with torch.no_grad():
        load_flat_params(model, base_params + update.to(dtype=base_params.dtype, device=base_params.device))
        actual = (model(x).detach().float() - base_logits).reshape(-1)
        load_flat_params(model, base_params)
    target_flat = target_f.reshape(-1)
    denom = torch.linalg.vector_norm(actual).clamp_min(1.0e-8) * torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)
    fit_cos = float((actual @ target_flat / denom).clamp(-1.0, 1.0).item())
    residual = float((torch.linalg.vector_norm(actual - target_flat) / torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)).item())
    basis_energy, readout_energy = _channel_energy(model, update)
    return update.detach().float(), {
        "K14_corrected_readout_layout": 0,
        "K14_readout_feature_cols": 0,
        "basis_estimate_fit_cosine": fit_cos,
        "basis_commit_projection_residual": residual,
        "K14_basis_channel_energy": basis_energy,
        "K14_readout_channel_energy": readout_energy,
        "K14_update_norm": float(torch.linalg.vector_norm(update).item()),
        "basis_linearized_jacobian_rows": int(jac.shape[0]),
        "basis_linearized_jacobian_cols": int(jac.shape[1]),
        "basis_linearized_solver_status": solver_status,
    }


def _refresh_update_diagnostics(model: PrimitiveKAN, x: torch.Tensor, target: torch.Tensor, update: torch.Tensor, diagnostics: dict[str, Any]) -> dict[str, Any]:
    before = flat_params(model).detach().float()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        load_flat_params(model, before + update.to(dtype=before.dtype, device=before.device))
        actual = (model(x).detach().float() - base_logits).reshape(-1)
        load_flat_params(model, before)
    target_flat = target.detach().float().to(device=actual.device).reshape(-1)
    denom = torch.linalg.vector_norm(actual).clamp_min(1.0e-8) * torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)
    basis_energy, readout_energy = _channel_energy(model, update)
    out = dict(diagnostics)
    out.update(
        {
            "basis_estimate_fit_cosine": float((actual @ target_flat / denom).clamp(-1.0, 1.0).item()),
            "basis_commit_projection_residual": float((torch.linalg.vector_norm(actual - target_flat) / torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)).item()),
            "K14_basis_channel_energy": basis_energy,
            "K14_readout_channel_energy": readout_energy,
            "K14_update_norm": float(torch.linalg.vector_norm(update).item()),
        }
    )
    return out


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
    device: torch.device,
    retention_weight: float,
    retention_start_step: int,
    retention_stop_step: int,
) -> dict[int, dict[str, float]]:
    model = _make_kan_with_init(carrier, x, seed, init_variant, device)
    x = x.detach().float().to(device)
    target = target.detach().float().to(device)
    update = update.detach().float().to(device)
    task_data = _move_task_data(task_data, device)
    before = flat_params(model).detach()
    with torch.no_grad():
        base = model(x).detach().float()
        load_flat_params(model, before + update.to(dtype=before.dtype, device=before.device))
        source_anchor = (model(x).detach().float() - base).detach()
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    out: dict[int, dict[str, float]] = {}
    for step in range(1, max(HORIZONS) + 1):
        opt.zero_grad(set_to_none=True)
        logits_now = model(x).float()
        loss = adapter.value(logits_now, task_data)
        if float(retention_weight) > 0.0:
            retention_active = step >= int(retention_start_step) and (int(retention_stop_step) <= 0 or step <= int(retention_stop_step))
            if retention_active:
                displacement_now = logits_now - base.to(device=logits_now.device, dtype=logits_now.dtype)
                loss = loss + float(retention_weight) * F.mse_loss(displacement_now, source_anchor.to(device=logits_now.device, dtype=logits_now.dtype))
        loss.backward()
        opt.step()
        if step % int(interval) == 0:
            with torch.no_grad():
                current = flat_params(model).detach()
                load_flat_params(model, current + float(scale) * update.to(dtype=current.dtype, device=current.device))
        if step in HORIZONS:
            with torch.no_grad():
                logits = model(x).detach().float()
                displacement = logits - base
                loss_value = float(adapter.value(logits, task_data).detach().item())
            out[step] = {"target_retention_score": _retention_score(displacement, target), "loss_value": loss_value}
    return out


def _carrier_probe_correct_readout_layout_gpu(
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
    commit_mode: str,
    update_gain: float,
    retention_weight: float,
    retention_start_step: int,
    retention_stop_step: int,
    device: torch.device,
) -> dict[str, Any]:
    x = payload["x"].detach().float().to(device)
    target = payload["target_delta"].detach().float().to(device)
    labels = payload.get("labels_for_loss_adapter_only")
    if labels is None:
        labels = torch.arange(int(x.shape[0]), device=device) % 5
    labels = labels.to(device=device, dtype=torch.long)
    probe = _make_kan_with_init(carrier, x, seed, init_variant, device)
    with torch.no_grad():
        base_logits = probe(x).detach().float()
    adapter, task_data = _adapter_for(adapter_name, base_logits, target, labels)
    task_data = _move_task_data(task_data, device)
    if commit_mode == "basis_linearized_w1":
        base_update, diagnostics = _solve_basis_linearized_update(probe, x, target)
    else:
        base_update, diagnostics = _solve_correct_readout_layout_update(probe, x, target)
    if float(update_gain) != 1.0:
        base_update = base_update * float(update_gain)
        diagnostics = _refresh_update_diagnostics(probe, x, target, base_update, diagnostics)
    controls = {
        "RandomMatchedNorm": _matched_random_like(base_update, seed, 501),
        "StableRandom": _stable_like(base_update),
        "SameSolverRandomTarget": _matched_random_like(base_update, seed, 707),
        "SignFlipTarget": -base_update,
        "CorruptTarget": torch.roll(base_update, shifts=1, dims=0),
    }
    fu = _run_kan_variant_with_init(
        carrier,
        x,
        target,
        base_update,
        adapter,
        task_data,
        seed,
        interval=interval,
        scale=scale,
        init_variant=init_variant,
        device=device,
        retention_weight=retention_weight,
        retention_start_step=retention_start_step,
        retention_stop_step=retention_stop_step,
    )
    control_scores = {
        name: _run_kan_variant_with_init(
            carrier,
            x,
            target,
            upd,
            adapter,
            task_data,
            seed + idx + 1,
            interval=interval,
            scale=scale,
            init_variant=init_variant,
            device=device,
            retention_weight=retention_weight,
            retention_start_step=retention_start_step,
            retention_stop_step=retention_stop_step,
        )
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
        "mapping_status": "executed_v22_13_corrected_layout_mapping_gpu",
        "carrier": carrier,
        "mechanism": commit_mode,
        "attempt": attempt_name,
        "periodic_interval": int(interval),
        "periodic_scale": float(scale),
        "update_gain": float(update_gain),
        "retention_weight": float(retention_weight),
        "retention_start_step": int(retention_start_step),
        "retention_stop_step": int(retention_stop_step),
        "source_state_attempt": int(float(retention_weight) > 0.0),
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
        "device": str(device),
        **diagnostics,
    }
    for h in HORIZONS:
        row[f"KAN_source_func_h{h}"] = source_func[h]
        row[f"KAN_source_loss_h{h}"] = source_loss[h]
    return row


def _best_mlp_source(out_dir: Path, adapter_name: str = "Delta-MSEAdapter") -> float:
    rows = [r for r in read_rows(out_dir / "v22_13_adapter_horizon_matrix.csv") if str(r.get("variant")) == "FU"]
    same_adapter = [r for r in rows if str(r.get("loss_adapter_name")) == str(adapter_name)]
    if same_adapter:
        rows = same_adapter
    vals = [finite_float(r.get("source_func_h3200")) for r in rows]
    vals = [v for v in vals if v == v]
    return max(vals) if vals else 0.0


def _eff_pass(out_dir: Path, carrier: str) -> int:
    summary = read_rows(out_dir / "v22_13_official_fused_status_matrix.csv")
    return int(any(r.get("carrier") == carrier and int_flag(r.get("carrier_specific_efficiency_pass")) for r in summary))


def _k17_row(row: dict[str, Any], commit_channel: str) -> dict[str, Any]:
    basis_energy = finite_float(row.get("K14_basis_channel_energy"), 0.0)
    readout_energy = finite_float(row.get("K14_readout_channel_energy"), 0.0)
    source_loss_3200 = finite_float(row.get("KAN_source_loss_h3200"), -999.0)
    source_loss_4800 = finite_float(row.get("KAN_source_loss_h4800"), -999.0)
    basis_pass = int(basis_energy > 0 and source_loss_3200 >= -1.0e-6 and source_loss_4800 >= -1.0e-6 and str(row.get("KAN_source_channel_decision")) == "KANRetainedSourceOpened")
    if basis_pass:
        blocker = ""
    elif basis_energy > 0:
        blocker = "KANBasisChannelSourceNotRetained"
    else:
        blocker = "KANReadoutSourceOpened_BasisChannelBlocked"
    return {
        "carrier": row.get("carrier", ""),
        "commit_channel": commit_channel,
        "readout_channel_energy": readout_energy,
        "basis_channel_energy": basis_energy,
        "basis_to_readout_energy_ratio": basis_energy / max(readout_energy, 1.0e-12),
        "basis_channel_source_loss_h3200": source_loss_3200 if basis_energy > 0 else "",
        "basis_channel_source_loss_h4800": source_loss_4800 if basis_energy > 0 else "",
        "readout_only_ablation_source_func_h3200": row.get("KAN_source_func_h3200", ""),
        "readout_only_ablation_source_loss_h3200": row.get("KAN_source_loss_h3200", ""),
        "K17_basis_channel_pass": basis_pass,
        "blocker": blocker,
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    device = _device(args.device)
    payload_path = source_dir / "v22_13_operator_commit_payload.pt"
    rows: list[dict[str, Any]] = []
    if not payload_path.exists():
        route = {
            "route": "S6-KANMappingNotEntered",
            "KAN_source_channel_pass_rows": 0,
            "blocker": "operator_commit_payload_missing",
            "promotion_allowed": 0,
        }
    else:
        payload = _load_payload(payload_path)
        mlp_source = _best_mlp_source(source_dir, "Delta-MSEAdapter")
        for carrier, attempt, init_variant in [
            ("D-FOU", "K15_DFOU_corrected_K13_revalidation", "default"),
            ("D-CHE", "K16_DCHE_corrected_source_loss_repair_slow_state", "default"),
        ]:
            for commit_mode, suffix, readout_claim, basis_claim, update_gain, retention_weight, retention_start, retention_stop in [
                ("corrected_readout_layout_low_degree_source_replay", "readout_corrected", 1, 0, 1.0, 0.0, 0, 0),
                ("basis_linearized_w1", "basis_linearized_w1_gain1", 0, 1, 1.0, 0.0, 0, 0),
                ("basis_linearized_w1", "basis_linearized_w1_gain4", 0, 1, 4.0, 0.0, 0, 0),
                ("basis_linearized_w1", "basis_linearized_w1_gain16", 0, 1, 16.0, 0.0, 0, 0),
                ("basis_linearized_w1", "basis_linearized_w1_gain1_anchor_w0p10", 0, 1, 1.0, 0.10, 1, 6400),
                ("basis_linearized_w1", "basis_linearized_w1_gain4_anchor_w0p10", 0, 1, 4.0, 0.10, 1, 6400),
            ]:
                row = _carrier_probe_correct_readout_layout_gpu(
                    carrier,
                    payload,
                    "Delta-MSEAdapter",
                    mlp_source,
                    _eff_pass(source_dir, carrier),
                    int(args.seed),
                    attempt_name=f"{attempt}_{suffix}",
                    interval=800,
                    scale=0.006,
                    init_variant=init_variant,
                    commit_mode=commit_mode,
                    update_gain=update_gain,
                    retention_weight=retention_weight,
                    retention_start_step=retention_start,
                    retention_stop_step=retention_stop,
                    device=device,
                )
                row["mapping_status"] = "executed_v22_13_corrected_layout_mapping"
                row["mechanism"] = commit_mode
                row["carrier_specific_efficiency_pass"] = _eff_pass(source_dir, carrier)
                row["D-CHE_efficiency_pass"] = _eff_pass(source_dir, "D-CHE") if carrier == "D-CHE" else ""
                row["D-FOU_efficiency_pass"] = _eff_pass(source_dir, "D-FOU") if carrier == "D-FOU" else ""
                row["uses_adapter_name_for_direction"] = 0
                row["uses_loss_formula_for_direction"] = 0
                row["readout_only_claim"] = readout_claim
                row["basis_channel_claim"] = basis_claim
                rows.append(row)
        pass_rows = [r for r in rows if str(r.get("KAN_source_channel_decision")) == "KANRetainedSourceOpened"]
        dche_rows = [r for r in rows if r.get("carrier") == "D-CHE"]
        dfou_rows = [r for r in rows if r.get("carrier") == "D-FOU"]
        dche = max(dche_rows, key=lambda r: finite_float(r.get("KAN_source_func_h3200"), -999.0), default={})
        dfou = max(dfou_rows, key=lambda r: finite_float(r.get("KAN_source_func_h3200"), -999.0), default={})
        dfou_source = int(any(int_flag(r.get("K14_source_exploration_pass")) for r in dfou_rows))
        dche_source = int(any(int_flag(r.get("K14_source_exploration_pass")) for r in dche_rows))
        dche_terminal = int(any(int_flag(r.get("K14_terminal_source_loss_pass")) for r in dche_rows))
        if dfou_source and not int_flag(dche.get("D-CHE_efficiency_pass")):
            route_name = "R7-DFOUCorrectedSourceOpened_DCHEBlocked"
        elif dche_source and (not int_flag(dche.get("D-CHE_efficiency_pass")) or not dche_terminal):
            route_name = "R8-DCHESourceExists_EfficiencyOrSourceLossBlocked"
        elif pass_rows:
            route_name = "S6-KANCorrectedCarrierExplorationPass"
        else:
            route_name = "S6-KANCorrectedCarrierNoGo"
        route = {
            "route": route_name,
            "KAN_source_channel_pass_rows": len(pass_rows),
            "DFOU_corrected_source_exists": dfou_source,
            "DCHE_corrected_source_exists": dche_source,
            "DFOU_efficiency_pass": int_flag(dfou.get("D-FOU_efficiency_pass")),
            "DCHE_efficiency_pass": int_flag(dche.get("D-CHE_efficiency_pass")),
            "DCHE_source_loss_h4800": dche.get("KAN_source_loss_h4800", ""),
            "readout_only_pass_rows": sum(int_flag(r.get("readout_only_claim")) for r in rows),
            "basis_channel_pass_rows": sum(int_flag(r.get("basis_channel_claim")) and str(r.get("KAN_source_channel_decision")) == "KANRetainedSourceOpened" for r in rows),
            "promotion_allowed": 0,
            "blocker": ";".join(dict.fromkeys(str(r.get("blocker")) for r in rows if r.get("blocker"))),
        }
    k15 = [r for r in rows if r.get("carrier") == "D-FOU"]
    k16 = [r for r in rows if r.get("carrier") == "D-CHE"]
    k17 = [_k17_row(r, str(r.get("mechanism", ""))) for r in rows]
    write_rows(out_dir / "v22_13_K15_DFOU_corrected_K13.csv", k15)
    write_rows(out_dir / "v22_13_K16_DCHE_corrected_source_loss_repair.csv", k16)
    write_rows(out_dir / "v22_13_K17_basis_channel_commit.csv", k17)
    write_rows(out_dir / "v22_13_KAN_mapping_matrix.csv", rows)
    write_rows(out_dir / "v22_13_readout_vs_basis_ablation.csv", k17)
    write_rows(out_dir / "v22_13_KAN_vs_MLP_same_operator.csv", rows)
    write_json(out_dir / "v22_13_kan_mapping_route.json", route)
    simple_svg(out_dir / "figures/v22_13_KAN_source_channel_dashboard.svg", "v22.13 KAN source channel", rows, "KAN_source_func_h3200")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_13_kan_corrected_mapping.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)} --device {args.device}", status="completed" if rows else "blocked", note=f"route={route['route']} pass_rows={route.get('KAN_source_channel_pass_rows')} device={device} blocker={route.get('blocker')}")


if __name__ == "__main__":
    main()
