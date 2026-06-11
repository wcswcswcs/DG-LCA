#!/usr/bin/env python3
"""v22.14 repair continuation for strict source-state and KAN basis-state blockers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.fu.source_loss import source_gate_row  # noqa: E402
from dgkan.profiling.efficiency_v22_13 import NativeEfficiencyConfig, run_native_efficiency_v22_13  # noqa: E402
from experiments import run_v22_13_kan_corrected_mapping as k13  # noqa: E402
from experiments import run_v22_13_operator_horizon as h13  # noqa: E402
from experiments import run_v22_14_basis_state_operator_fu as v14  # noqa: E402
from experiments.run_v22_14_common import (  # noqa: E402
    PYTHON,
    V2214_RECAP_DOC,
    append_exec,
    artifact_index,
    build_code_review_packet,
    build_results_bundle,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)

HORIZONS = [100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400]
REQUIRED_COTANGENTS = {"Delta-Gaussian", "Delta-StableRandom", "Delta-SourceTarget", "Delta-LossCEAdapter", "Delta-MSEAdapter", "Delta-RankingAdapter"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "efficiency_repair", "f1_repair", "kan_repair", "finalize_repair"])
    p.add_argument("--out-dir", default="results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14")
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2213)
    p.add_argument("--device-efficiency", default="cuda:0")
    p.add_argument("--device-f1", default="cuda:1")
    p.add_argument("--device-kan", default="cuda:2")
    p.add_argument("--norm-scale", type=float, default=5.12)
    p.add_argument("--batch-sizes", default="128,256,512,1024")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--f1-seeds", default="2213")
    p.add_argument("--f1-adapters", default="Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter,Delta-PreferenceAdapter-smoke,Delta-StableRandom-control,Delta-RandomMatched-control")
    p.add_argument("--f1-modes", default="optimizer_prox_jacobian_rank16,optimizer_prox_jacobian_rank32,source_state_projector_jacobian_rank32")
    return p


def _items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _ints(text: str) -> list[int]:
    return [int(x) for x in _items(text)]


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _readout_add_update_mlp(model: torch.nn.Module, x: torch.Tensor, residual: torch.Tensor, *, damping: float, rank_cap: int) -> tuple[torch.Tensor, dict[str, Any]]:
    base_params = flat_params(model).detach().float()
    with torch.no_grad():
        features = model.frozen_readout_features(x).detach().float()
        current_logits = model(x).detach().float()
    residual = residual.detach().float().to(device=features.device)
    hidden, classes = int(features.shape[1]), int(residual.shape[1])
    rank = min(int(rank_cap), hidden, int(features.shape[0])) if int(rank_cap) > 0 else min(hidden, int(features.shape[0]))
    start = time.perf_counter()
    try:
        u, s, vh = torch.linalg.svd(features, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        z = (s_r[:, None] * (u_r.T @ residual)) / (s_r[:, None].square() + float(damping))
        coeff = vh_r.T @ z
        status = "svd_rank_cap"
    except Exception:
        gram = features.T @ features + float(damping) * torch.eye(hidden, device=features.device, dtype=features.dtype)
        rhs = features.T @ residual
        coeff = torch.linalg.lstsq(gram, rhs).solution
        status = "lstsq_fallback"
    solve_ms = (time.perf_counter() - start) * 1000.0
    update = torch.zeros_like(base_params)
    offset = 0
    wrote = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name == "w2":
            update[offset : offset + n] = coeff.reshape(-1).to(device=update.device, dtype=update.dtype)
            wrote = 1
        offset += n
    with torch.no_grad():
        load_flat_params(model, base_params + update.to(device=base_params.device, dtype=base_params.dtype))
        actual = (model(x).detach().float() - current_logits).reshape(-1)
        load_flat_params(model, base_params)
    target = residual.reshape(-1)
    denom = torch.linalg.vector_norm(target).clamp_min(1.0e-8)
    return update, {
        "jacobian_source_state_solver": status,
        "jacobian_rank_cap": rank,
        "jacobian_solve_ms": solve_ms,
        "jacobian_readout_update_written": wrote,
        "jacobian_residual_projection_residual": float((torch.linalg.vector_norm(actual - target) / denom).item()),
        "jacobian_residual_projection_cosine": h13._safe_cos(actual, target),
        "jacobian_update_norm": float(torch.linalg.vector_norm(update).item()),
    }


def _repair_mode_cfg(mode: str) -> dict[str, Any]:
    table = {
        "optimizer_prox_jacobian_rank16": {"jacobian_prox_scale": 0.08, "jacobian_projector_scale": 0.0, "jacobian_interval": 100, "jacobian_damping": 1.0e-2, "jacobian_rank_cap": 16},
        "optimizer_prox_jacobian_rank32": {"jacobian_prox_scale": 0.10, "jacobian_projector_scale": 0.0, "jacobian_interval": 100, "jacobian_damping": 1.0e-3, "jacobian_rank_cap": 32},
        "source_state_projector_jacobian_rank32": {"jacobian_prox_scale": 0.0, "jacobian_projector_scale": 0.12, "jacobian_interval": 100, "jacobian_damping": 1.0e-3, "jacobian_rank_cap": 32},
    }
    cfg = dict(table[mode])
    cfg.update({"lr_scale": 0.02, "uses_loss_modification_for_retention": 0, "anchor_mechanism_type": mode, "source_state_attempt": 1})
    return cfg


def _train_anchor_repair(payload: dict[str, Any], *, update_vec: torch.Tensor, optimizer_name: str, adapter: Any, task_data: Any, seed: int, cfg: dict[str, Any], device: torch.device) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, Any]]]:
    torch.manual_seed(int(seed))
    model = v14._make_model(payload).to(device)
    x = payload["x"].detach().float().to(device)
    target_delta = payload["target_delta"].detach().float().to(device)
    update = update_vec.detach().float().to(device)
    task_data = h13._move_task_data(task_data, device)
    before = flat_params(model).detach()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        load_flat_params(model, before - update.to(dtype=before.dtype, device=before.device))
        source_anchor = (model(x).detach().float() - base_logits).detach()
        load_flat_params(model, before)
    opt = h13._make_optimizer(model, optimizer_name, float(cfg["lr_scale"]))
    snapshots: dict[int, dict[str, float]] = {}
    source_rows: dict[int, dict[str, Any]] = {}
    prev_displacement = source_anchor.clone()
    last_diag: dict[str, Any] = {}
    for step in range(1, max(HORIZONS) + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            logits_now = model(x).float()
            loss = adapter.value(logits_now, task_data)
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits_after = model(x).detach().float()
            displacement = logits_after - base_logits
            anchor_norm2 = source_anchor.square().sum().clamp_min(1.0e-8)
            projected_gain = float(((displacement * source_anchor).sum() / anchor_norm2).item())
            prox_residual = max(0.0, 1.0 - projected_gain)
            destructive = float((((displacement - prev_displacement) * source_anchor).sum() / anchor_norm2).item())
        apply_jacobian = step % int(cfg["jacobian_interval"]) == 0
        if apply_jacobian and (float(cfg["jacobian_prox_scale"]) > 0.0 and prox_residual > 1.0e-4):
            residual = source_anchor - displacement
            add_update, last_diag = _readout_add_update_mlp(model, x, residual, damping=float(cfg["jacobian_damping"]), rank_cap=int(cfg["jacobian_rank_cap"]))
            current = flat_params(model).detach()
            scale = float(cfg["jacobian_prox_scale"]) * min(1.0, prox_residual)
            load_flat_params(model, current + scale * add_update.to(device=current.device, dtype=current.dtype))
        elif apply_jacobian and (float(cfg["jacobian_projector_scale"]) > 0.0 and destructive < -1.0e-5):
            residual = source_anchor - displacement
            add_update, last_diag = _readout_add_update_mlp(model, x, residual, damping=float(cfg["jacobian_damping"]), rank_cap=int(cfg["jacobian_rank_cap"]))
            current = flat_params(model).detach()
            scale = float(cfg["jacobian_projector_scale"]) * min(1.0, -destructive)
            load_flat_params(model, current + scale * add_update.to(device=current.device, dtype=current.dtype))
        with torch.no_grad():
            prev_displacement = (model(x).detach().float() - base_logits).detach()
        if step in HORIZONS:
            snapshots[step] = h13._snapshot_gpu(model, x, base_logits, target_delta, adapter, task_data)
            source_rows[step] = {
                "source_state_alignment": h13._safe_cos(prev_displacement, source_anchor),
                "source_state_decay_rate": max(0.0, 1.0 - projected_gain),
                "optimizer_destructive_projection": max(0.0, -destructive),
                "optimizer_prox_residual": prox_residual,
                "uses_loss_modification_for_retention": 0,
                **{f"last_{k}": v for k, v in last_diag.items()},
            }
    return snapshots, source_rows


def _evaluate_anchor_repair(payload: dict[str, Any], adapter_name: str, adapter: Any, task_data: Any, adapter_seen: int, operator_id: str, seed: int, mode: str, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = _repair_mode_cfg(mode)
    adapter_payload, commit_diag = h13._payload_for_adapter(payload, operator_id, adapter, task_data, int(seed), device, float(payload.get("norm_scale", 5.12)))
    commit_update = adapter_payload["update_vec"].detach().float().to(device)
    controls = h13._control_updates_gpu(adapter_payload, commit_update, int(seed), device)
    fu_snaps, fu_state = _train_anchor_repair(adapter_payload, update_vec=commit_update, optimizer_name="AdamW", adapter=adapter, task_data=task_data, seed=int(seed), cfg=cfg, device=device)
    control_snaps: dict[str, dict[int, dict[str, float]]] = {}
    control_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for idx, (control_name, update, optimizer_name) in enumerate(controls):
        snaps, states = _train_anchor_repair(adapter_payload, update_vec=update.detach().float().to(device), optimizer_name=optimizer_name, adapter=adapter, task_data=task_data, seed=int(seed) + idx + 1, cfg=cfg, device=device)
        control_snaps[control_name] = snaps
        control_rows.append({"loss_adapter_name": adapter_name, "anchor_mechanism_type": mode, "variant": control_name, "optimizer": optimizer_name, "adapter_is_control": 1, "device": str(device), **{f"target_retention_score_h{h}": snaps[h]["target_retention_score"] for h in HORIZONS}})
        for h in HORIZONS:
            state_rows.append({"loss_adapter_name": adapter_name, "anchor_mechanism_type": mode, "variant": control_name, "horizon": h, **states[h]})
    best_func = {h: max(snaps[h]["target_retention_score"] for snaps in control_snaps.values()) for h in HORIZONS}
    best_loss = {h: min(snaps[h]["loss_value"] for snaps in control_snaps.values()) for h in HORIZONS}
    source_func = {h: fu_snaps[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu_snaps[h]["loss_value"] for h in HORIZONS}
    positive_counts = {h: sum(int(fu_snaps[h]["target_retention_score"] > snaps[h]["target_retention_score"] + 0.005) for snaps in control_snaps.values()) for h in HORIZONS}
    r4800 = source_func[4800] / source_func[3200] if abs(source_func[3200]) > 1.0e-12 else ""
    r6400 = source_func[6400] / source_func[4800] if abs(source_func[4800]) > 1.0e-12 else ""
    c3 = int(all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200]) and all(source_loss[h] >= -1.0e-6 for h in [800, 1600, 3200]) and positive_counts[3200] >= 6)
    c4 = int(c3 and source_func[4800] >= 0.005 and isinstance(r4800, float) and r4800 >= 0.50 and source_loss[4800] >= -1.0e-6 and positive_counts[4800] >= 7)
    c5 = int(c4 and source_func[6400] >= 0.005 and isinstance(r6400, float) and r6400 >= 0.50 and source_loss[6400] >= -1.0e-6)
    row: dict[str, Any] = {
        "loss_adapter_name": adapter_name,
        "operator_id": operator_id,
        "anchor_mechanism_type": mode,
        "uses_loss_modification_for_retention": 0,
        "official_strict_anchor_allowed": 1,
        "jacobian_prox_scale": cfg["jacobian_prox_scale"],
        "jacobian_projector_scale": cfg["jacobian_projector_scale"],
        "jacobian_interval": cfg["jacobian_interval"],
        "jacobian_damping": cfg["jacobian_damping"],
        "jacobian_rank_cap": cfg["jacobian_rank_cap"],
        "adapter_seen_in_operator_tuning": adapter_seen,
        "adapter_is_control": int(adapter_name.endswith("-control")),
        "variant": "FU",
        "device": str(device),
        "R4800_over_3200_func": r4800,
        "R6400_over_4800_func": r6400,
        "C3_source_formation_pass": c3,
        "C4_terminal_retention_pass": c4,
        "C5_h6400_retention_pass": c5,
        "TargetRetentionOnly_NotTaskUseful": int(any(source_func[h] >= 0.005 for h in [3200, 4800, 6400]) and source_loss[3200] < -1.0e-6),
        "blocker": "" if c3 else "source_func_or_source_loss_or_control_gate_failed",
        **{f"adapter_commit_{k}": v for k, v in commit_diag.items() if k in {"projection_residual_Gf", "ActuationR2", "function_displacement_cos_with_target"}},
    }
    for h in HORIZONS:
        row[f"source_func_h{h}"] = source_func[h]
        row[f"source_loss_h{h}"] = source_loss[h]
        row[f"row_positive_count_h{h}"] = positive_counts[h]
        row[f"FU_loss_value_h{h}"] = fu_snaps[h]["loss_value"]
        row[f"best_control_loss_value_h{h}"] = best_loss[h]
        for key, value in fu_state[h].items():
            row[f"{key}_h{h}"] = value
    row.update(source_gate_row({h: source_func[h] for h in HORIZONS}, {h: source_loss[h] for h in HORIZONS}))
    state_rows = [{"loss_adapter_name": adapter_name, "anchor_mechanism_type": mode, "variant": "FU", "horizon": h, **fu_state[h]} for h in HORIZONS] + state_rows
    return row, control_rows, state_rows


def stage_f1_repair(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    device = _device(args.device_f1)
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    payload_path = source_dir / "v22_14_operator_commit_payload.pt"
    if not payload_path.exists():
        route = {"route": "F1Repair-BlockedBeforePayload", "strict_operator_pass": 0, "blocker": "operator_commit_payload_missing"}
    else:
        payload = _load_payload(payload_path)
        payload["norm_scale"] = float(args.norm_scale)
        operator_id = str(payload.get("operator_variational_route", {}).get("selected_operators", "")).split(";")[0] or "LIO8_FisherSobolevResolvent"
        allow_adapters = set(_items(args.f1_adapters))
        modes = _items(args.f1_modes)
        for seed in _ints(args.f1_seeds):
            for adapter_idx, (name, adapter, task_data, seen) in enumerate(v14._adapters(payload, seed)):
                if name not in allow_adapters:
                    continue
                for mode in modes:
                    row, controls, states = _evaluate_anchor_repair(payload, name, adapter, task_data, seen, operator_id, seed + adapter_idx * 1000, mode, device)
                    row["seed"] = seed
                    for item in controls:
                        item["seed"] = seed
                    for item in states:
                        item["seed"] = seed
                    rows.append(row)
                    control_rows.extend(controls)
                    state_rows.extend(states)
        non_random = [r for r in rows if not int_flag(r.get("adapter_is_control"))]
        strict_c3 = sorted({r["loss_adapter_name"] for r in non_random if int_flag(r.get("C3_source_formation_pass"))})
        strict_c4 = sorted({r["loss_adapter_name"] for r in non_random if int_flag(r.get("C4_terminal_retention_pass"))})
        controls_pass = [r for r in rows if int_flag(r.get("adapter_is_control")) and (int_flag(r.get("C3_source_formation_pass")) or int_flag(r.get("C4_terminal_retention_pass")))]
        strict_pass = int(len(strict_c3) >= 3 and len(strict_c4) >= 2 and not controls_pass)
        route = {
            "route": "F1Repair-StrictJacobianSourceStatePass" if strict_pass else "F1Repair-JacobianSourceStateNoGo",
            "FU_repair_rows": len(rows),
            "strict_C3_adapter_count": len(strict_c3),
            "strict_C4_adapter_count": len(strict_c4),
            "strict_C3_adapter_list": ";".join(strict_c3),
            "strict_C4_adapter_list": ";".join(strict_c4),
            "controls_pass_count": len(controls_pass),
            "strict_operator_pass": strict_pass,
            "uses_loss_modification_for_strict_pass": 0,
            "blocker": "" if strict_pass else "JacobianSourceStateNoGo",
        }
    write_rows(out_dir / "v22_14_source_state_repair_matrix.csv", rows)
    write_rows(out_dir / "v22_14_source_state_repair_dynamics.csv", state_rows)
    write_rows(out_dir / "v22_14_control_attribution_repair_matrix.csv", control_rows)
    write_json(out_dir / "v22_14_source_state_repair_route.json", route)
    simple_svg(out_dir / "figures/v22_14_source_state_repair_source_func.svg", "v22.14 source-state repair", rows, "source_func_h3200")
    append_exec(out_dir, " ".join([PYTHON, "experiments/run_v22_14_repair_continuation.py", "--stage", "f1_repair", "--out-dir", str(out_dir), "--seed", str(args.seed), "--device-f1", args.device_f1, "--f1-modes", args.f1_modes]), status="completed" if rows else "blocked", note=f"route={route.get('route')} rows={len(rows)} blocker={route.get('blocker')}", gpu=args.device_f1, task_id="f1_repair")


def _direct_basis_operator_update_dual(model: torch.nn.Module, x: torch.Tensor, delta: torch.Tensor, damping: float) -> tuple[torch.Tensor, dict[str, Any]]:
    jac, spans, base_params = v14._basis_jacobian(model, x, "w1")
    rhs = -delta.detach().float().reshape(-1).to(jac.device, jac.dtype)
    gram = jac @ jac.T + float(damping) * torch.eye(int(jac.shape[0]), device=jac.device, dtype=jac.dtype)
    start = time.perf_counter()
    try:
        alpha = torch.linalg.solve(gram, rhs)
        status = "dual_solve"
    except Exception:
        alpha = torch.linalg.lstsq(gram, rhs).solution
        status = "dual_lstsq_fallback"
    coeff = jac.T @ alpha
    solve_ms = (time.perf_counter() - start) * 1000.0
    update = torch.zeros_like(base_params)
    cursor = 0
    for _name, start_idx, n in spans:
        update[start_idx : start_idx + n] = coeff[cursor : cursor + n].to(device=update.device, dtype=update.dtype)
        cursor += n
    residual = float(torch.linalg.vector_norm(jac @ coeff - rhs).item() / torch.linalg.vector_norm(rhs).clamp_min(1.0e-8).item())
    return update, {"basis_operator_solve_ms": solve_ms, "basis_operator_residual": residual, "basis_linear_solver_status": status, "basis_jacobian_rows": int(jac.shape[0]), "basis_jacobian_cols": int(jac.shape[1]), "basis_cg_iterations": 0}


def _cap_readout_to_basis(model: torch.nn.Module, update: torch.Tensor, max_readout_to_basis: float = 1.0) -> torch.Tensor:
    out = update.detach().float().clone()
    basis_norm = 0.0
    readout_spans: list[tuple[int, int]] = []
    offset = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        val = float(torch.linalg.vector_norm(out[offset : offset + n]).item())
        low = name.lower()
        if "w1" in low:
            basis_norm += val
        elif "w2" in low or "readout" in low:
            readout_spans.append((offset, n))
        offset += n
    readout_norm = sum(float(torch.linalg.vector_norm(out[start : start + n]).item()) for start, n in readout_spans)
    limit = float(max_readout_to_basis) * max(basis_norm, 1.0e-12)
    if readout_norm > limit:
        scale = limit / max(readout_norm, 1.0e-12)
        for start, n in readout_spans:
            out[start : start + n] *= scale
    return out


def _evaluate_kan_update(carrier: str, x: torch.Tensor, target: torch.Tensor, adapter: Any, task_data: Any, update: torch.Tensor, seed: int, device: torch.device, *, basis_operator_id: str, interval: int, scale: float, diag: dict[str, Any]) -> dict[str, Any]:
    probe = k13._make_kan_with_init(carrier, x, int(seed), "default", device)
    basis_energy, readout_energy = k13._channel_energy(probe, update)
    fu_scores = k13._run_kan_variant_with_init(carrier, x, target, update, adapter, task_data, int(seed), interval=interval, scale=scale, init_variant="default", device=device, retention_weight=0.0, retention_start_step=0, retention_stop_step=0)
    controls = {"RandomMatchedNorm": k13._matched_random_like(update, int(seed), 301), "StableRandom": k13._stable_like(update), "SignFlipTarget": -update}
    control_scores = {name: k13._run_kan_variant_with_init(carrier, x, target, upd, adapter, task_data, int(seed) + idx + 1, interval=interval, scale=scale, init_variant="default", device=device, retention_weight=0.0, retention_start_step=0, retention_stop_step=0) for idx, (name, upd) in enumerate(controls.items())}
    best_func = {h: max(scores[h]["target_retention_score"] for scores in control_scores.values()) for h in HORIZONS}
    best_loss = {h: min(scores[h]["loss_value"] for scores in control_scores.values()) for h in HORIZONS}
    source_func = {h: fu_scores[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu_scores[h]["loss_value"] for h in HORIZONS}
    r4800 = source_func[4800] / source_func[3200] if abs(source_func[3200]) > 1.0e-12 else ""
    positive_controls = {h: sum(int(fu_scores[h]["target_retention_score"] > scores[h]["target_retention_score"] + 0.005) for scores in control_scores.values()) for h in HORIZONS}
    kpass = int(basis_energy >= 0.50 and all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200]) and source_loss[3200] >= -1.0e-6 and isinstance(r4800, float) and r4800 >= 0.50 and positive_controls[3200] >= 2)
    row: dict[str, Any] = {"carrier": carrier, "basis_operator_id": basis_operator_id, "basis_state_type": "w1_basis_repair", "training_interval": interval, "training_scale": scale, "basis_channel_energy": basis_energy, "readout_channel_energy": readout_energy, "basis_to_readout_energy_ratio": basis_energy / max(readout_energy, 1.0e-12), "R4800_over_3200_func": r4800, "K19_basis_state_operator_pass": kpass, "blocker": "" if kpass else "KAN_basis_state_source_or_retention_gate_failed", **diag}
    for h in HORIZONS:
        row[f"KAN_source_func_h{h}"] = source_func[h]
        row[f"KAN_source_loss_h{h}"] = source_loss[h]
        row[f"positive_control_count_h{h}"] = positive_controls[h]
    return row


def _run_kan_transfer(carrier: str, x: torch.Tensor, target: torch.Tensor, readout_update: torch.Tensor, basis_update: torch.Tensor, adapter: Any, task_data: Any, seed: int, tau: int, interval: int, scale: float, device: torch.device) -> dict[int, dict[str, float]]:
    model = k13._make_kan_with_init(carrier, x, seed, "default", device)
    x = x.detach().float().to(device)
    target = target.detach().float().to(device)
    readout_update = readout_update.detach().float().to(device)
    basis_update = basis_update.detach().float().to(device)
    task_data = k13._move_task_data(task_data, device)
    with torch.no_grad():
        base = model(x).detach().float()
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    out: dict[int, dict[str, float]] = {}
    for step in range(1, max(HORIZONS) + 1):
        opt.zero_grad(set_to_none=True)
        logits_now = model(x).float()
        loss = adapter.value(logits_now, task_data)
        loss.backward()
        opt.step()
        if step % int(interval) == 0:
            alpha = min(1.0, float(step) / max(1.0, float(tau)))
            update = (1.0 - alpha) * readout_update + alpha * basis_update
            with torch.no_grad():
                current = flat_params(model).detach()
                load_flat_params(model, current + float(scale) * update.to(dtype=current.dtype, device=current.device))
        if step in HORIZONS:
            with torch.no_grad():
                logits = model(x).detach().float()
                displacement = logits - base
                loss_value = float(adapter.value(logits, task_data).detach().item())
            out[step] = {"target_retention_score": k13._retention_score(displacement, target), "loss_value": loss_value}
    return out


def _evaluate_k20_transfer(carrier: str, x: torch.Tensor, target: torch.Tensor, readout_update: torch.Tensor, basis_update: torch.Tensor, adapter: Any, task_data: Any, seed: int, tau: int, device: torch.device) -> dict[str, Any]:
    interval, scale = 800, 0.006
    fu = _run_kan_transfer(carrier, x, target, readout_update, basis_update, adapter, task_data, seed, tau, interval, scale, device)
    controls = {
        "RandomMatchedTransfer": (k13._matched_random_like(readout_update, seed, 401), k13._matched_random_like(basis_update, seed, 402)),
        "SignFlipTransfer": (-readout_update, -basis_update),
    }
    control_scores = {name: _run_kan_transfer(carrier, x, target, ru, bu, adapter, task_data, seed + idx + 1, tau, interval, scale, device) for idx, (name, (ru, bu)) in enumerate(controls.items())}
    best_func = {h: max(scores[h]["target_retention_score"] for scores in control_scores.values()) for h in HORIZONS}
    best_loss = {h: min(scores[h]["loss_value"] for scores in control_scores.values()) for h in HORIZONS}
    source_func = {h: fu[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu[h]["loss_value"] for h in HORIZONS}
    alpha3200 = min(1.0, 3200.0 / max(1.0, float(tau)))
    alpha4800 = min(1.0, 4800.0 / max(1.0, float(tau)))
    probe = k13._make_kan_with_init(carrier, x, seed, "default", device)
    b3200, r3200 = k13._channel_energy(probe, (1.0 - alpha3200) * readout_update + alpha3200 * basis_update)
    b4800, r4800 = k13._channel_energy(probe, (1.0 - alpha4800) * readout_update + alpha4800 * basis_update)
    passed = int(b3200 >= 0.50 and b4800 >= 0.50 and source_func[4800] >= 0.005 and source_loss[4800] >= -1.0e-6)
    row: dict[str, Any] = {"carrier": carrier, "transfer_variant": f"K20-real-tau{tau}", "tau": tau, "interval": interval, "scale": scale, "alpha_t_h3200": alpha3200, "alpha_t_h4800": alpha4800, "basis_source_energy_h3200": b3200, "readout_source_energy_h3200": r3200, "basis_source_energy_h4800": b4800, "readout_source_energy_h4800": r4800, "source_transfer_success_h4800": passed, "blocker": "" if passed else "readout_source_cannot_migrate_to_basis"}
    for h in HORIZONS:
        row[f"source_func_h{h}"] = source_func[h]
        row[f"source_loss_h{h}"] = source_loss[h]
    return row


def stage_kan_repair(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    device = _device(args.device_kan)
    payload_path = source_dir / "v22_14_operator_commit_payload.pt"
    k19_rows: list[dict[str, Any]] = []
    k20_rows: list[dict[str, Any]] = []
    k21_rows: list[dict[str, Any]] = []
    if not payload_path.exists():
        route = {"route": "KANRepair-BlockedBeforePayload", "KAN_basis_pass_rows": 0, "blocker": "operator_commit_payload_missing"}
    else:
        payload = _load_payload(payload_path)
        x = payload["x"].detach().float().to(device)
        target = payload["target_delta"].detach().float().to(device)
        labels = payload.get("labels_for_loss_adapter_only", torch.arange(int(x.shape[0])) % 5).to(device)
        for carrier in ["D-FOU", "D-CHE"]:
            probe = k13._make_kan_with_init(carrier, x, int(args.seed), "default", device)
            adapter, task_data = k13._adapter_for("Delta-MSEAdapter", probe(x).detach().float(), target, labels)
            task_data = k13._move_task_data(task_data, device)
            cot = adapter.cotangent(probe(x).detach().float(), task_data).to(device)
            readout_update, _readout_diag = k13._solve_correct_readout_layout_update(probe, x, target)
            primal_update, primal_diag = v14._direct_basis_operator_update(probe, x, cot, damping=1.0e-2)
            k19_rows.append(_evaluate_kan_update(carrier, x, target, adapter, task_data, primal_update, int(args.seed), device, basis_operator_id="K19-A-primal-d1e-2", interval=800, scale=0.006, diag=primal_diag))
            for damping in [1.0e-3, 1.0e-2, 1.0e-1]:
                dual_update, dual_diag = _direct_basis_operator_update_dual(probe, x, cot, damping=damping)
                k19_rows.append(_evaluate_kan_update(carrier, x, target, adapter, task_data, dual_update, int(args.seed) + int(damping * 10000), device, basis_operator_id=f"K19-B-dual-d{damping:g}", interval=800, scale=0.006, diag=dual_diag))
                residual_target = target
                readout_residual, readout_diag = k13._solve_correct_readout_layout_update(probe, x, residual_target)
                coupled = _cap_readout_to_basis(probe, dual_update + readout_residual, max_readout_to_basis=1.0)
                cdiag = dict(dual_diag)
                cdiag.update({"readout_residual_solver": "corrected_readout_layout_capped", "readout_residual_cols": readout_diag.get("K14_readout_feature_cols", "")})
                k19_rows.append(_evaluate_kan_update(carrier, x, target, adapter, task_data, coupled, int(args.seed) + int(damping * 20000), device, basis_operator_id=f"K19-D-basis-readout-coupled-cap50-d{damping:g}", interval=800, scale=0.006, diag=cdiag))
            best_basis = max([r for r in k19_rows if r.get("carrier") == carrier], key=lambda r: finite_float(r.get("KAN_source_func_h3200"), -999.0))
            best_update, _ = _direct_basis_operator_update_dual(probe, x, cot, damping=1.0e-2)
            for tau in [1600, 3200, 6400, 9600]:
                k20_rows.append(_evaluate_k20_transfer(carrier, x, target, readout_update, best_update, adapter, task_data, int(args.seed) + tau, tau, device))
            k21_rows.append({"carrier": carrier, "diagnostic": "K21-A-optimizer-only-basis-source-buffer", "diagnostic_only": 1, "official_strict_eligible": 0, "reason": "source-state buffer can be retained outside inference parameters but is not a model-parameter basis carrier", "linked_best_k19_source_func_h3200": best_basis.get("KAN_source_func_h3200", ""), "linked_best_k19_source_loss_h3200": best_basis.get("KAN_source_loss_h3200", "")})
        pass_rows = sum(int_flag(r.get("K19_basis_state_operator_pass")) for r in k19_rows)
        transfer_pass_rows = sum(int_flag(r.get("source_transfer_success_h4800")) for r in k20_rows)
        if pass_rows:
            route_name = "KANRepair-K19BasisStatePass"
        elif transfer_pass_rows:
            route_name = "KANRepair-K20ReadoutToBasisTransferPass"
        else:
            route_name = "KANRepair-BasisStateNoGo"
        route = {"route": route_name, "K19_repair_rows": len(k19_rows), "K19_basis_state_operator_pass_rows": pass_rows, "K20_repair_rows": len(k20_rows), "K20_transfer_success_rows": transfer_pass_rows, "KAN_basis_pass_rows": pass_rows + transfer_pass_rows, "blocker": "" if pass_rows + transfer_pass_rows else "KAN_basis_state_source_or_transfer_gate_failed"}
    write_rows(out_dir / "v22_14_K19_basis_state_operator_repair.csv", k19_rows)
    write_rows(out_dir / "v22_14_K20_readout_to_basis_transfer_repair.csv", k20_rows)
    write_rows(out_dir / "v22_14_K21_basis_source_state_boundary.csv", k21_rows)
    write_json(out_dir / "v22_14_kan_basis_repair_route.json", route)
    simple_svg(out_dir / "figures/v22_14_kan_repair_source_func.svg", "v22.14 KAN repair", k19_rows, "KAN_source_func_h3200")
    append_exec(out_dir, " ".join([PYTHON, "experiments/run_v22_14_repair_continuation.py", "--stage", "kan_repair", "--out-dir", str(out_dir), "--seed", str(args.seed), "--device-kan", args.device_kan]), status="completed" if k19_rows or k20_rows else "blocked", note=f"route={route.get('route')} KAN_basis_pass_rows={route.get('KAN_basis_pass_rows')} blocker={route.get('blocker')}", gpu=args.device_kan, task_id="kan_repair")


def stage_efficiency_repair(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    device = args.device_efficiency
    rows, summary, gradcheck = run_native_efficiency_v22_13(device_name=device, batch_sizes=_ints(args.batch_sizes), seed=int(args.seed), cfg=NativeEfficiencyConfig(hidden=int(args.hidden), repeats=int(args.repeats), warmup=int(args.warmup)))
    variant_rows: list[dict[str, Any]] = []
    batches = set(_ints(args.batch_sizes))
    for variant in sorted({str(r.get("variant")) for r in rows if r.get("carrier") in {"D-FOU", "D-CHE"}}):
        vrows = [r for r in rows if str(r.get("variant")) == variant]
        cots = {str(r.get("cotangent_type")) for r in vrows}
        pass_rows = [r for r in vrows if int_flag(r.get("official_fused_kernel_complete")) and not int_flag(r.get("manual_upstream_vjp_used")) and not int_flag(r.get("fallback_kernel_used")) and float(r.get("forward_ratio_vs_mlp", 99.0)) <= 1.35 and float(r.get("vjp_ratio_vs_mlp", 99.0)) <= 1.35 and float(r.get("operator_step_ratio_vs_mlp", 99.0)) <= 1.35 and float(r.get("full_step_ratio_vs_mlp", 99.0)) <= 1.35 and float(r.get("memory_ratio_vs_mlp", 99.0)) <= 1.10]
        suite_complete = int(REQUIRED_COTANGENTS.issubset(cots))
        grid_complete = int({128, 256, 512, 1024}.issubset(batches) and int(args.hidden) >= 128)
        robust = int(len(vrows) > 0 and len(pass_rows) == len(vrows) and suite_complete and grid_complete)
        variant_rows.append({"carrier": vrows[0].get("carrier", "") if vrows else "", "variant": variant, "profile_rows": len(vrows), "ratio_pass_rows": len(pass_rows), "cotangent_types": ";".join(sorted(cots)), "planned_cotangent_suite_complete": suite_complete, "planned_hidden_grid_complete": int(int(args.hidden) >= 128), "planned_batch_1024_complete": int(1024 in batches), "planned_batch_grid_complete": int({128, 256, 512, 1024}.issubset(batches)), "variant_robust_pass": robust, "exploration_pass": int(len(vrows) > 0 and len(pass_rows) >= math.ceil(0.80 * len(vrows)) and all(int_flag(r.get("official_fused_kernel_complete")) for r in vrows)), "blocker": "" if robust else "ratio_outlier_or_planned_grid_not_closed"})
    route = {"route": "E1Repair-VariantRobustEfficiencyPass" if any(int_flag(r.get("variant_robust_pass")) for r in variant_rows) else "R3-EfficiencyVariantRobustnessBlocked", "profile_rows": len(rows), "official_fused_kernel_complete_rows": sum(int_flag(r.get("official_fused_kernel_complete")) for r in rows), "variant_robust_pass_rows": sum(int_flag(r.get("variant_robust_pass")) for r in variant_rows), "exploration_pass_rows": sum(int_flag(r.get("exploration_pass")) for r in variant_rows), "blocker": "" if any(int_flag(r.get("variant_robust_pass")) for r in variant_rows) else "planned_grid_or_ratio_gate_not_closed"}
    write_rows(out_dir / "v22_14_native_efficiency_truth_table_repair.csv", rows)
    write_rows(out_dir / "v22_14_variant_robust_efficiency_repair_matrix.csv", variant_rows)
    write_rows(out_dir / "v22_14_native_kernel_gradcheck_repair.csv", gradcheck)
    write_rows(out_dir / "v22_14_official_fused_status_repair_matrix.csv", summary)
    write_json(out_dir / "v22_14_efficiency_repair_route.json", route)
    simple_svg(out_dir / "figures/v22_14_efficiency_repair_dashboard.svg", "v22.14 efficiency repair", rows, "operator_step_ratio_vs_mlp")
    append_exec(out_dir, " ".join([PYTHON, "experiments/run_v22_14_repair_continuation.py", "--stage", "efficiency_repair", "--out-dir", str(out_dir), "--seed", str(args.seed), "--device-efficiency", args.device_efficiency, "--batch-sizes", args.batch_sizes, "--hidden", str(args.hidden), "--repeats", str(args.repeats), "--warmup", str(args.warmup)]), status="completed", note=f"route={route.get('route')} profile_rows={len(rows)} robust={route.get('variant_robust_pass_rows')} blocker={route.get('blocker')}", gpu=args.device_efficiency, task_id="efficiency_repair")


def _best_route(base: dict[str, Any], repair: dict[str, Any], pass_key: str) -> dict[str, Any]:
    if repair and int_flag(repair.get(pass_key)) > int_flag(base.get(pass_key)):
        return repair
    return repair or base


def stage_finalize_repair(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    base_decision = read_json(out_dir / "v22_14_final_decision.json")
    code = read_json(out_dir / "v22_14_code_truth_route.json")
    eff_base = read_json(out_dir / "v22_14_efficiency_route.json")
    eff_repair = read_json(out_dir / "v22_14_efficiency_repair_route.json")
    op_base = read_json(out_dir / "v22_14_operator_horizon_route.json")
    op_repair = read_json(out_dir / "v22_14_source_state_repair_route.json")
    kan_base = read_json(out_dir / "v22_14_kan_basis_route.json")
    kan_repair = read_json(out_dir / "v22_14_kan_basis_repair_route.json")
    task = read_json(out_dir / "v22_14_task_eval_route.json")
    eff_pass = max(int_flag(eff_base.get("variant_robust_pass_rows")), int_flag(eff_repair.get("variant_robust_pass_rows")))
    strict_operator = max(int_flag(op_base.get("strict_operator_pass")), int_flag(op_repair.get("strict_operator_pass")))
    kan_pass = max(int_flag(kan_base.get("KAN_basis_pass_rows")), int_flag(kan_repair.get("KAN_basis_pass_rows")))
    if not int_flag(code.get("S0_pass")):
        route = "R0-CodeTruthFailed"
    elif not eff_pass:
        route = "R3-EfficiencyVariantRobustnessBlocked"
    elif not strict_operator:
        route = "R5-OptimizerProxAnchorNoGo"
    elif not kan_pass:
        route = "R8-KANBasisStateReachableButNotRetained"
    elif not int_flag(task.get("full_scientific_task_gate_pass")):
        route = "R12-OfficialOperatorAndCarrierReady_TaskEvidencePending"
    else:
        route = "R13-FullScientificPromotionReady"
    blockers = []
    if not eff_pass:
        blockers.append(eff_repair.get("blocker") or eff_base.get("blocker") or "efficiency_gate")
    if not strict_operator:
        blockers.append(op_repair.get("blocker") or op_base.get("blocker") or "strict_operator_gate")
    if not kan_pass:
        blockers.append(kan_repair.get("blocker") or kan_base.get("blocker") or "kan_basis_gate")
    if kan_pass and not int_flag(task.get("full_scientific_task_gate_pass")):
        blockers.append(task.get("blocker", "task_gate"))
    decision = dict(base_decision)
    decision.update({"route": route, "official_operator_promotion_allowed_strict": int(strict_operator), "official_kan_carrier_promotion_allowed": int(strict_operator and eff_pass and kan_pass), "scientific_claim_allowed": int(strict_operator and eff_pass and kan_pass and int_flag(task.get("full_scientific_task_gate_pass"))), "blocking_metric": ";".join(x for x in blockers if x), "next_codex_action": "continue from remaining repair blocker; do not promote diagnostic K21 or loss-modified auxiliary rows"})
    write_json(out_dir / "v22_14_repair_final_decision.json", decision)
    write_json(out_dir / "v22_14_final_decision.json", decision)
    write_rows(out_dir / "v22_14_artifact_index.csv", artifact_index(out_dir))
    packet = build_code_review_packet(out_dir)
    bundle = build_results_bundle(out_dir)
    repair_anchor = read_rows(out_dir / "v22_14_source_state_repair_matrix.csv")
    repair_eff = read_rows(out_dir / "v22_14_variant_robust_efficiency_repair_matrix.csv")
    repair_k19 = read_rows(out_dir / "v22_14_K19_basis_state_operator_repair.csv")
    repair_k20 = read_rows(out_dir / "v22_14_K20_readout_to_basis_transfer_repair.csv")
    repair_k21 = read_rows(out_dir / "v22_14_K21_basis_source_state_boundary.csv")
    with V2214_RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write("\n## 9. Repair Continuation v22.14-R1\n\n")
        f.write(f"生成时间：{now_sg()}\n\n")
        f.write("### 9.1 Repair Route\n\n")
        for key in ["route", "official_operator_promotion_allowed_strict", "official_kan_carrier_promotion_allowed", "scientific_claim_allowed", "blocking_metric", "next_codex_action"]:
            f.write(f"- {key}: `{decision.get(key, '')}`\n")
        f.write(f"- refreshed results bundle: `{bundle}`\n")
        f.write(f"- refreshed code review packet: `{packet}`\n\n")
        f.write("### 9.2 修改说明\n\n")
        f.write("- 修复 efficiency profiler：不再截断 cotangent suite 前 4 项，纳入 `Delta-SourceTarget` 与 `Delta-RankingAdapter` 的 non-smoke robust grid。\n")
        f.write("- 新增 `experiments/run_v22_14_repair_continuation.py`，按计划尝试 Jacobian source-prox/projector、K19 dual/coupled basis operator、K20 real tau transfer 和 K21 boundary diagnostic。\n")
        f.write("- 所有 repair row 均写入独立 `*_repair*.csv`，旧 v22.14 原始 no-go 矩阵未覆盖；K21 明确 `diagnostic_only=1`，不计 official promotion。\n\n")
        f.write("### 9.3 Source-State Repair\n\n")
        f.write(md_table(repair_anchor, ["seed", "loss_adapter_name", "anchor_mechanism_type", "jacobian_rank_cap", "source_func_h3200", "source_func_h4800", "source_loss_h3200", "source_loss_h4800", "C3_source_formation_pass", "C4_terminal_retention_pass", "blocker"], max_rows=80))
        f.write("\nAnalysis: 这些 row 是 no-loss-modification optimizer dynamics；如果仍不过，说明 v22.13 auxiliary loss anchor 不能简单转写为 readout-Jacobian source-state correction。\n\n")
        f.write("### 9.4 Efficiency Repair\n\n")
        f.write(md_table(repair_eff, ["carrier", "variant", "profile_rows", "ratio_pass_rows", "cotangent_types", "planned_cotangent_suite_complete", "planned_batch_grid_complete", "planned_hidden_grid_complete", "variant_robust_pass", "blocker"], max_rows=60))
        f.write("\nAnalysis: robust gate 只按真实 profile rows 计数；ratio 或 grid 任一未闭合仍保持 blocked。\n\n")
        f.write("### 9.5 KAN Repair\n\n")
        f.write(md_table(repair_k19, ["carrier", "basis_operator_id", "basis_channel_energy", "readout_channel_energy", "basis_operator_residual", "KAN_source_func_h3200", "KAN_source_func_h4800", "KAN_source_loss_h3200", "KAN_source_loss_h4800", "K19_basis_state_operator_pass", "blocker"], max_rows=80))
        f.write("\n")
        f.write(md_table(repair_k20, ["carrier", "transfer_variant", "alpha_t_h3200", "basis_source_energy_h3200", "basis_source_energy_h4800", "source_func_h4800", "source_loss_h4800", "source_transfer_success_h4800", "blocker"], max_rows=80))
        f.write("\n")
        f.write(md_table(repair_k21, ["carrier", "diagnostic", "diagnostic_only", "official_strict_eligible", "linked_best_k19_source_func_h3200", "linked_best_k19_source_loss_h3200", "reason"], max_rows=20))
        f.write("\nAnalysis: K19/K20 repair 仍需 basis channel 同时满足 source_func/source_loss；readout-coupled 或 K21 diagnostic 不会被当成 KAN basis official success。\n")
    write_rows(out_dir / "v22_14_artifact_index.csv", artifact_index(out_dir))
    build_code_review_packet(out_dir)
    build_results_bundle(out_dir)
    append_exec(out_dir, " ".join([PYTHON, "experiments/run_v22_14_repair_continuation.py", "--stage", "finalize_repair", "--out-dir", str(out_dir), "--seed", str(args.seed)]), status="completed", note=f"route={decision.get('route')} blocker={decision.get('blocking_metric')}", gpu="n/a", task_id="finalize_repair")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    if args.stage in {"all", "efficiency_repair"}:
        stage_efficiency_repair(args)
    if args.stage in {"all", "f1_repair"}:
        stage_f1_repair(args)
    if args.stage in {"all", "kan_repair"}:
        stage_kan_repair(args)
    if args.stage in {"all", "finalize_repair"}:
        stage_finalize_repair(args)


if __name__ == "__main__":
    main()
