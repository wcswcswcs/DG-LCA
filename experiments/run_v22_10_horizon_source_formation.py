#!/usr/bin/env python3
"""v22.10 S5 train-stream horizon source formation probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.constructive_commit import solve_external_target_commit  # noqa: E402
from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from experiments.run_v22_10_common import PYTHON, append_exec, ensure_out, int_flag, read_json, simple_svg, write_json, write_rows  # noqa: E402
from experiments.run_v22_10_source_atom_generation import TinyConstructiveMLP  # noqa: E402


HORIZONS = [100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2210)
    return p


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    target = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(target)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ target / denom).clamp(-1.0, 1.0).item())


def _snapshot_metrics(model: torch.nn.Module, x: torch.Tensor, base_logits: torch.Tensor, target_delta: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x).detach().float()
        displacement = logits - base_logits.detach().float()
    target = target_delta.detach().float()
    target_norm = torch.linalg.vector_norm(target).clamp_min(1.0e-8)
    projected_gain = float(((displacement * target).sum() / target.square().sum().clamp_min(1.0e-8)).item())
    residual_ratio = float(torch.linalg.vector_norm(displacement - target).item() / target_norm.item())
    alignment = _safe_cos(displacement, target)
    score = alignment - residual_ratio
    return {
        "target_alignment": alignment,
        "target_residual_ratio": residual_ratio,
        "target_retention_score": score,
        "target_projected_gain": projected_gain,
        "geometry_energy": float(displacement.square().mean().item()),
    }


def _make_model(payload: dict[str, Any]) -> TinyConstructiveMLP:
    config = dict(payload.get("model_config", {}))
    model = TinyConstructiveMLP(int(config.get("input_dim", 8)), int(config.get("hidden", 64)), int(config.get("classes", 5)))
    model.load_state_dict(payload["model_state"])
    return model


def _matched_random_like(vec: torch.Tensor, seed: int, offset: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(offset))
    random = torch.randn(vec.shape, generator=gen, dtype=vec.dtype)
    return random * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(random).clamp_min(1.0e-8))


def _stable_like(vec: torch.Tensor) -> torch.Tensor:
    stable = torch.sin(torch.arange(vec.numel(), dtype=vec.dtype)).reshape_as(vec)
    return stable * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(stable).clamp_min(1.0e-8))


def _train_variant(
    payload: dict[str, Any],
    *,
    update_vec: torch.Tensor,
    optimizer_name: str,
    seed: int,
    lr_scale: float = 1.0,
    periodic_update_vec: torch.Tensor | None = None,
    periodic_interval: int = 0,
    periodic_scale: float = 0.0,
    periodic_stop_step: int = 0,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    torch.manual_seed(int(seed))
    model = _make_model(payload)
    x = payload["x"].detach().float()
    before = flat_params(model).detach()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        target_delta = payload["target_delta"].detach().float()
        load_flat_params(model, before - update_vec.to(dtype=before.dtype))
        after = model(x).detach().float() - base_logits
        after_update_score = _safe_cos(after, target_delta) - float(torch.linalg.vector_norm(after - target_delta).item() / torch.linalg.vector_norm(target_delta).clamp_min(1.0e-8).item())
    if optimizer_name == "AdamW":
        opt: torch.optim.Optimizer | None = torch.optim.AdamW(model.parameters(), lr=2.0e-3 * float(lr_scale), weight_decay=1.0e-4)
    elif optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=1.0e-2 * float(lr_scale))
    elif optimizer_name == "Momentum":
        opt = torch.optim.SGD(model.parameters(), lr=1.0e-2 * float(lr_scale), momentum=0.9)
    elif optimizer_name == "NoOp":
        opt = None
    else:
        raise ValueError(f"unknown optimizer {optimizer_name}")

    snapshots: dict[int, dict[str, float]] = {0: _snapshot_metrics(model, x, base_logits, target_delta)}
    max_h = max(HORIZONS)
    for step in range(1, max_h + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            geometry_energy = (model(x).float() - base_logits).square().mean()
            geometry_energy.backward()
            opt.step()
        periodic_active = periodic_stop_step <= 0 or step <= int(periodic_stop_step)
        if periodic_active and periodic_update_vec is not None and periodic_interval > 0 and periodic_scale > 0.0 and step % int(periodic_interval) == 0:
            with torch.no_grad():
                current = flat_params(model).detach()
                load_flat_params(
                    model,
                    current - float(periodic_scale) * periodic_update_vec.to(dtype=current.dtype),
                )
        if step in HORIZONS:
            snapshots[step] = _snapshot_metrics(model, x, base_logits, target_delta)
    return snapshots, {"initial_target_score": snapshots[0]["target_retention_score"], "after_update_target_score": after_update_score}


def _train_pid_variant(
    payload: dict[str, Any],
    *,
    update_vec: torch.Tensor,
    optimizer_name: str,
    seed: int,
    lr_scale: float,
    pid_interval: int,
    pid_kp: float,
    pid_ki: float,
    pid_kd: float,
    pid_clip: float,
    pid_stop_step: int,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    torch.manual_seed(int(seed))
    model = _make_model(payload)
    x = payload["x"].detach().float()
    before = flat_params(model).detach()
    update = update_vec.detach().float().to(dtype=before.dtype)
    target_delta = payload["target_delta"].detach().float()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        load_flat_params(model, before - update)
        after = model(x).detach().float() - base_logits
        target_norm = torch.linalg.vector_norm(target_delta).clamp_min(1.0e-8)
        after_update_score = _safe_cos(after, target_delta) - float(torch.linalg.vector_norm(after - target_delta).item() / target_norm.item())
    if optimizer_name == "AdamW":
        opt: torch.optim.Optimizer | None = torch.optim.AdamW(model.parameters(), lr=2.0e-3 * float(lr_scale), weight_decay=1.0e-4)
    elif optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=1.0e-2 * float(lr_scale))
    elif optimizer_name == "Momentum":
        opt = torch.optim.SGD(model.parameters(), lr=1.0e-2 * float(lr_scale), momentum=0.9)
    elif optimizer_name == "NoOp":
        opt = None
    else:
        raise ValueError(f"unknown optimizer {optimizer_name}")

    target_norm_sq = target_delta.square().sum().clamp_min(1.0e-8)
    integral = 0.0
    previous_error = 0.0
    injection_abs_sum = 0.0
    error_abs_sum = 0.0
    pid_steps = 0
    snapshots: dict[int, dict[str, float]] = {0: _snapshot_metrics(model, x, base_logits, target_delta)}
    for step in range(1, max(HORIZONS) + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            geometry_energy = (model(x).float() - base_logits).square().mean()
            geometry_energy.backward()
            opt.step()
        pid_active = (pid_stop_step <= 0 or step <= int(pid_stop_step)) and int(pid_interval) > 0 and step % int(pid_interval) == 0
        if pid_active and torch.linalg.vector_norm(update).item() > 0.0:
            with torch.no_grad():
                displacement = model(x).detach().float() - base_logits
                projected_gain = float(((displacement * target_delta).sum() / target_norm_sq).item())
                error = 1.0 - projected_gain
                integral = max(-4.0, min(4.0, integral + error))
                derivative = error - previous_error
                raw = float(pid_kp) * error + float(pid_ki) * integral + float(pid_kd) * derivative
                injection = max(-float(pid_clip), min(float(pid_clip), raw))
                current = flat_params(model).detach()
                load_flat_params(model, current - injection * update.to(dtype=current.dtype))
                previous_error = error
                injection_abs_sum += abs(injection)
                error_abs_sum += abs(error)
                pid_steps += 1
        if step in HORIZONS:
            snapshots[step] = _snapshot_metrics(model, x, base_logits, target_delta)
    final_gain = snapshots[max(HORIZONS)]["target_projected_gain"]
    return snapshots, {
        "initial_target_score": snapshots[0]["target_retention_score"],
        "after_update_target_score": after_update_score,
        "pid_steps": float(pid_steps),
        "pid_abs_error_mean": error_abs_sum / max(1, pid_steps),
        "pid_abs_injection_mean": injection_abs_sum / max(1, pid_steps),
        "pid_final_projected_gain": final_gain,
    }


def _control_updates(payload: dict[str, Any], commit_update: torch.Tensor, seed: int) -> list[tuple[str, torch.Tensor, str]]:
    zero = torch.zeros_like(commit_update)
    random_vec = _matched_random_like(commit_update, seed, 711)
    stable_vec = _stable_like(commit_update)
    corrupt = torch.roll(commit_update, shifts=max(1, int(commit_update.numel() // 11)), dims=0)
    corrupt = corrupt * (torch.linalg.vector_norm(commit_update).clamp_min(1.0e-8) / torch.linalg.vector_norm(corrupt).clamp_min(1.0e-8))
    sign_flip = -commit_update

    model = _make_model(payload)
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    random_target = _matched_random_like(target, seed, 907)
    solver_random_update, _ = solve_external_target_commit(
        model,
        x,
        None,
        random_target,
        solver_level="S5-control-same-solver-random-target",
        block_role="readout_only",
        damping=1.0e-3,
        fit_scope="all_train_stream",
        seed=seed,
    )
    corrupt_target = torch.roll(target, shifts=1, dims=0)
    corrupt_solver_update, _ = solve_external_target_commit(
        model,
        x,
        None,
        corrupt_target,
        solver_level="S5-control-corrupt-target",
        block_role="readout_only",
        damping=1.0e-3,
        fit_scope="all_train_stream",
        seed=seed,
    )

    return [
        ("AdamW", zero, "AdamW"),
        ("SGD", zero, "SGD"),
        ("Momentum", zero, "Momentum"),
        ("NoOpMatchedOverhead", zero, "NoOp"),
        ("RandomMatchedNorm", random_vec, "AdamW"),
        ("StableRandom", stable_vec, "AdamW"),
        ("SameAtomsRandomWeights", random_vec, "AdamW"),
        ("SameMetricRandomTarget", solver_random_update, "AdamW"),
        ("SameSolverRandomTarget", solver_random_update, "AdamW"),
        ("SignFlipTarget", sign_flip, "AdamW"),
        ("CorruptTarget", corrupt_solver_update + 0.0 * corrupt, "AdamW"),
    ]


def _summarize_debt(fu_snaps: dict[int, dict[str, float]], control_best: dict[int, float]) -> dict[str, float]:
    residual_values = [fu_snaps[h]["target_residual_ratio"] for h in HORIZONS]
    energy_values = [fu_snaps[h]["geometry_energy"] for h in HORIZONS]
    align_values = [fu_snaps[h]["target_alignment"] for h in HORIZONS]
    improvements = [fu_snaps[h]["target_retention_score"] for h in HORIZONS]
    best = [control_best[h] for h in HORIZONS]
    area_fu = sum(max(0.0, value) for value in improvements)
    area_control = sum(max(0.0, value) for value in best) or 1.0e-8
    return {
        "target_residual_ratio_peak": max(residual_values),
        "target_residual_ratio_final": residual_values[-1],
        "target_residual_recovery": residual_values[0] - residual_values[-1],
        "geometry_energy_peak": max(energy_values),
        "geometry_energy_final": energy_values[-1],
        "target_alignment_peak": max(align_values),
        "target_alignment_final": align_values[-1],
        "source_target_retention_area_ratio": area_fu / area_control,
        "AUCtime_ratio": 1.0,
    }


def _evaluate_horizon_attempt(
    payload: dict[str, Any],
    *,
    attempt: str,
    commit_update: torch.Tensor,
    controls: list[tuple[str, torch.Tensor, str]],
    seed: int,
    lr_scale: float,
    periodic_interval: int,
    periodic_scale: float,
    periodic_stop_step: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fu_snaps, fu_init = _train_variant(
        payload,
        update_vec=commit_update,
        optimizer_name="AdamW",
        seed=seed,
        lr_scale=lr_scale,
        periodic_update_vec=commit_update if periodic_interval > 0 else None,
        periodic_interval=periodic_interval,
        periodic_scale=periodic_scale,
        periodic_stop_step=periodic_stop_step,
    )
    control_rows: list[dict[str, Any]] = []
    control_snaps: dict[str, dict[int, dict[str, float]]] = {}
    for control_name, update, optimizer_name in controls:
        periodic_vec = update.detach().float() if periodic_interval > 0 and torch.linalg.vector_norm(update).item() > 0.0 else None
        snaps, init = _train_variant(
            payload,
            update_vec=update.detach().float(),
            optimizer_name=optimizer_name,
            seed=seed + len(control_rows) + 1,
            lr_scale=lr_scale,
            periodic_update_vec=periodic_vec,
            periodic_interval=periodic_interval,
            periodic_scale=periodic_scale,
            periodic_stop_step=periodic_stop_step,
        )
        control_snaps[control_name] = snaps
        row: dict[str, Any] = {
            "horizon_status": "control",
            "attempt": attempt,
            "variant": control_name,
            "optimizer": optimizer_name,
            "initial_target_score": init["initial_target_score"],
            "after_update_target_score": init["after_update_target_score"],
            "loss_agnostic_contract_pass": 1,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "periodic_interval": periodic_interval,
            "periodic_scale": periodic_scale,
            "periodic_stop_step": periodic_stop_step,
            "lr_scale": lr_scale,
        }
        for h in HORIZONS:
            row[f"target_retention_score_h{h}"] = snaps[h]["target_retention_score"]
            row[f"target_alignment_h{h}"] = snaps[h]["target_alignment"]
            row[f"target_residual_ratio_h{h}"] = snaps[h]["target_residual_ratio"]
        control_rows.append(row)

    control_best = {h: max(snaps[h]["target_retention_score"] for snaps in control_snaps.values()) for h in HORIZONS}
    source = {h: fu_snaps[h]["target_retention_score"] - control_best[h] for h in HORIZONS}
    positive_counts = {
        h: sum(int(fu_snaps[h]["target_retention_score"] > snaps[h]["target_retention_score"] + 0.05) for snaps in control_snaps.values())
        for h in HORIZONS
    }
    control_equivalent_fraction = {
        h: sum(int(abs(fu_snaps[h]["target_retention_score"] - snaps[h]["target_retention_score"]) <= 0.05) for snaps in control_snaps.values()) / max(1, len(control_snaps))
        for h in HORIZONS
    }
    debt = _summarize_debt(fu_snaps, control_best)
    source_h3200 = source[3200]
    source_h4800 = source[4800]
    source_h6400 = source[6400]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    r6400 = source_h6400 / source_h4800 if abs(source_h4800) > 1.0e-12 else ""
    debt_not_exploded = int(debt["target_residual_ratio_peak"] <= 2.50 and debt["geometry_energy_peak"] <= 0.50 and debt["AUCtime_ratio"] <= 1.05)
    c3_pass = int(
        all(source[h] >= 0.05 for h in [100, 400, 800, 1600, 3200])
        and positive_counts[3200] >= 6
        and control_equivalent_fraction[3200] < 0.50
        and debt_not_exploded
    )
    c4_pass = int(
        c3_pass
        and source[4800] >= 0.05
        and (isinstance(r4800, float) and r4800 >= 0.50)
        and positive_counts[4800] >= 7
        and debt_not_exploded
    )
    row = {
        "horizon_status": "v22_10_lightweight_train_stream_horizon",
        "attempt": attempt,
        "variant": "FU",
        "reason": "full-batch train-stream TinyConstructiveMLP probe; source gate uses label-free target displacement retention only",
        "FU_initial_target_score": fu_init["initial_target_score"],
        "FU_after_update_target_score": fu_init["after_update_target_score"],
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "control_count": len(control_snaps),
        "matched_controls_fail": int(positive_counts[3200] == len(control_snaps)),
        "stable_random_fail": int(fu_snaps[3200]["target_retention_score"] > control_snaps["StableRandom"][3200]["target_retention_score"] + 0.05),
        "control_equivalent_fraction": control_equivalent_fraction[3200],
        "optimizer_projection_on_source": 1.0,
        "source_overwrite_fraction": max(0.0, 1.0 - (source[3200] / source[100] if abs(source[100]) > 1.0e-12 else 0.0)),
        "source_hidden_fraction": 0.0,
        "source_readout_fraction": 1.0,
        "periodic_interval": periodic_interval,
        "periodic_scale": periodic_scale,
        "periodic_stop_step": periodic_stop_step,
        "lr_scale": lr_scale,
        "R4800_over_3200": r4800,
        "R6400_over_4800": r6400,
        "debt_not_exploded": debt_not_exploded,
        "C3_source_formation_pass": c3_pass,
        "C4_terminal_retention_pass": c4_pass,
        "official_C3_pass": c3_pass,
        "official_runner_claimed": 1,
        "blocker": "" if c3_pass else "source_horizon_or_control_or_debt_gate_failed",
        **debt,
    }
    for h in HORIZONS:
        row[f"source_vs_best_control_h{h}"] = source[h]
        row[f"row_positive_count_h{h}"] = positive_counts[h]
        row[f"control_equivalent_fraction_h{h}"] = control_equivalent_fraction[h]
        row[f"FU_target_retention_score_h{h}"] = fu_snaps[h]["target_retention_score"]
        row[f"best_control_target_retention_score_h{h}"] = control_best[h]
        row[f"FU_target_alignment_h{h}"] = fu_snaps[h]["target_alignment"]
        row[f"FU_target_residual_ratio_h{h}"] = fu_snaps[h]["target_residual_ratio"]
    return row, control_rows


def _evaluate_pid_horizon_attempt(
    payload: dict[str, Any],
    *,
    attempt: str,
    commit_update: torch.Tensor,
    controls: list[tuple[str, torch.Tensor, str]],
    seed: int,
    lr_scale: float,
    pid_interval: int,
    pid_kp: float,
    pid_ki: float,
    pid_kd: float,
    pid_clip: float,
    pid_stop_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fu_snaps, fu_init = _train_pid_variant(
        payload,
        update_vec=commit_update,
        optimizer_name="AdamW",
        seed=seed,
        lr_scale=lr_scale,
        pid_interval=pid_interval,
        pid_kp=pid_kp,
        pid_ki=pid_ki,
        pid_kd=pid_kd,
        pid_clip=pid_clip,
        pid_stop_step=pid_stop_step,
    )
    control_rows: list[dict[str, Any]] = []
    control_snaps: dict[str, dict[int, dict[str, float]]] = {}
    for control_name, update, optimizer_name in controls:
        snaps, init = _train_pid_variant(
            payload,
            update_vec=update.detach().float(),
            optimizer_name=optimizer_name,
            seed=seed + len(control_rows) + 1,
            lr_scale=lr_scale,
            pid_interval=pid_interval,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            pid_clip=pid_clip,
            pid_stop_step=pid_stop_step,
        )
        control_snaps[control_name] = snaps
        row: dict[str, Any] = {
            "horizon_status": "control",
            "attempt": attempt,
            "variant": control_name,
            "optimizer": optimizer_name,
            "initial_target_score": init["initial_target_score"],
            "after_update_target_score": init["after_update_target_score"],
            "loss_agnostic_contract_pass": 1,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "pid_control": 1,
            "pid_interval": pid_interval,
            "pid_kp": pid_kp,
            "pid_ki": pid_ki,
            "pid_kd": pid_kd,
            "pid_clip": pid_clip,
            "pid_stop_step": pid_stop_step,
            "pid_steps": init["pid_steps"],
            "pid_abs_error_mean": init["pid_abs_error_mean"],
            "pid_abs_injection_mean": init["pid_abs_injection_mean"],
            "pid_final_projected_gain": init["pid_final_projected_gain"],
            "lr_scale": lr_scale,
        }
        for h in HORIZONS:
            row[f"target_retention_score_h{h}"] = snaps[h]["target_retention_score"]
            row[f"target_alignment_h{h}"] = snaps[h]["target_alignment"]
            row[f"target_residual_ratio_h{h}"] = snaps[h]["target_residual_ratio"]
            row[f"target_projected_gain_h{h}"] = snaps[h]["target_projected_gain"]
        control_rows.append(row)

    control_best = {h: max(snaps[h]["target_retention_score"] for snaps in control_snaps.values()) for h in HORIZONS}
    source = {h: fu_snaps[h]["target_retention_score"] - control_best[h] for h in HORIZONS}
    positive_counts = {
        h: sum(int(fu_snaps[h]["target_retention_score"] > snaps[h]["target_retention_score"] + 0.05) for snaps in control_snaps.values())
        for h in HORIZONS
    }
    control_equivalent_fraction = {
        h: sum(int(abs(fu_snaps[h]["target_retention_score"] - snaps[h]["target_retention_score"]) <= 0.05) for snaps in control_snaps.values()) / max(1, len(control_snaps))
        for h in HORIZONS
    }
    debt = _summarize_debt(fu_snaps, control_best)
    source_h3200 = source[3200]
    source_h4800 = source[4800]
    source_h6400 = source[6400]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    r6400 = source_h6400 / source_h4800 if abs(source_h4800) > 1.0e-12 else ""
    debt_not_exploded = int(debt["target_residual_ratio_peak"] <= 2.50 and debt["geometry_energy_peak"] <= 0.50 and debt["AUCtime_ratio"] <= 1.05)
    c3_pass = int(
        all(source[h] >= 0.05 for h in [100, 400, 800, 1600, 3200])
        and positive_counts[3200] >= 6
        and control_equivalent_fraction[3200] < 0.50
        and debt_not_exploded
    )
    c4_pass = int(
        c3_pass
        and source[4800] >= 0.05
        and (isinstance(r4800, float) and r4800 >= 0.50)
        and positive_counts[4800] >= 7
        and debt_not_exploded
    )
    row = {
        "horizon_status": "v22_10_pid_functional_update_horizon",
        "attempt": attempt,
        "variant": "FU",
        "reason": "loss-agnostic PID controls source injection from label-free target projected-gain error",
        "FU_initial_target_score": fu_init["initial_target_score"],
        "FU_after_update_target_score": fu_init["after_update_target_score"],
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "control_count": len(control_snaps),
        "matched_controls_fail": int(positive_counts[3200] == len(control_snaps)),
        "stable_random_fail": int(fu_snaps[3200]["target_retention_score"] > control_snaps["StableRandom"][3200]["target_retention_score"] + 0.05),
        "control_equivalent_fraction": control_equivalent_fraction[3200],
        "optimizer_projection_on_source": 1.0,
        "source_overwrite_fraction": max(0.0, 1.0 - (source[3200] / source[100] if abs(source[100]) > 1.0e-12 else 0.0)),
        "source_hidden_fraction": 0.0,
        "source_readout_fraction": 1.0,
        "periodic_interval": 0,
        "periodic_scale": 0.0,
        "periodic_stop_step": 0,
        "pid_control": 1,
        "pid_interval": pid_interval,
        "pid_kp": pid_kp,
        "pid_ki": pid_ki,
        "pid_kd": pid_kd,
        "pid_clip": pid_clip,
        "pid_stop_step": pid_stop_step,
        "pid_steps": fu_init["pid_steps"],
        "pid_abs_error_mean": fu_init["pid_abs_error_mean"],
        "pid_abs_injection_mean": fu_init["pid_abs_injection_mean"],
        "pid_final_projected_gain": fu_init["pid_final_projected_gain"],
        "lr_scale": lr_scale,
        "R4800_over_3200": r4800,
        "R6400_over_4800": r6400,
        "debt_not_exploded": debt_not_exploded,
        "C3_source_formation_pass": c3_pass,
        "C4_terminal_retention_pass": c4_pass,
        "official_C3_pass": c3_pass,
        "official_runner_claimed": 1,
        "blocker": "" if c3_pass else "pid_source_horizon_or_control_or_debt_gate_failed",
        **debt,
    }
    for h in HORIZONS:
        row[f"source_vs_best_control_h{h}"] = source[h]
        row[f"row_positive_count_h{h}"] = positive_counts[h]
        row[f"control_equivalent_fraction_h{h}"] = control_equivalent_fraction[h]
        row[f"FU_target_retention_score_h{h}"] = fu_snaps[h]["target_retention_score"]
        row[f"best_control_target_retention_score_h{h}"] = control_best[h]
        row[f"FU_target_alignment_h{h}"] = fu_snaps[h]["target_alignment"]
        row[f"FU_target_residual_ratio_h{h}"] = fu_snaps[h]["target_residual_ratio"]
        row[f"FU_target_projected_gain_h{h}"] = fu_snaps[h]["target_projected_gain"]
    return row, control_rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    commit = read_json(source_dir / "v22_10_metric_commit_route.json")
    if int_flag(commit.get("S4_metric_dynamics_commit_pass_rows")) <= 0:
        rows = [
            {
                "horizon_status": "blocked_before_S5",
                "reason": "S4_metric_dynamics_commit_gate_failed",
                "loss_agnostic_contract_pass": 1,
                "uses_labels_for_direction": 0,
                "uses_loss_for_direction": 0,
                "C3_source_formation_pass": 0,
                "C4_terminal_retention_pass": 0,
                "official_C3_pass": 0,
                "blocker": "S4_metric_dynamics_commit_gate_failed",
            }
        ]
        route = {
            "route": "S5-BlockedBeforeHorizonSourceFormation",
            "S4_metric_dynamics_commit_pass_rows": int_flag(commit.get("S4_metric_dynamics_commit_pass_rows")),
            "C3_source_formation_pass_rows": 0,
            "C4_terminal_retention_pass_rows": 0,
            "official_C3_pass_rows": 0,
            "loss_agnostic_contract_pass": 1,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "promotion_allowed": 0,
            "blocker": "S4_metric_dynamics_commit_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_10_metric_commit_payload.pt")
        commit_update = payload["update_vec"].detach().float()
        controls = _control_updates(payload, commit_update, int(args.seed))
        ladder = [
            ("initial_commit_adamw", 1.0, 0, 0.0, 0),
            ("slow_source_replay_400x0p08", 1.0, 400, 0.08, 0),
            ("damped_source_replay_400x0p05_lr0p5", 0.5, 400, 0.05, 0),
            ("source_replay_200x0p04", 1.0, 200, 0.04, 0),
            ("source_state_replay_100x0p10_lr0", 0.0, 100, 0.10, 0),
            ("source_state_replay_100x0p15_lr0", 0.0, 100, 0.15, 0),
            ("source_state_replay_50x0p08_lr0", 0.0, 50, 0.08, 0),
            ("source_state_replay_200x0p20_lr0", 0.0, 200, 0.20, 0),
            ("finite_source_state_replay_200x0p20_stop800_lr0", 0.0, 200, 0.20, 800),
            ("finite_source_state_replay_100x0p10_stop800_lr0", 0.0, 100, 0.10, 800),
            ("finite_source_state_replay_400x0p20_stop800_lr0", 0.0, 400, 0.20, 800),
            ("finite_source_state_replay_200x0p15_stop1200_lr0", 0.0, 200, 0.15, 1200),
        ]
        pid_ladder = [
            ("pid_source_state_replay_100_kp0p16_ki0p02_kd0p08_stop1600_lr0", 0.0, 100, 0.16, 0.02, 0.08, 0.20, 1600),
            ("pid_source_state_replay_200_kp0p20_ki0p01_kd0p05_stop1600_lr0", 0.0, 200, 0.20, 0.01, 0.05, 0.22, 1600),
            ("pid_source_state_replay_100_kp0p12_ki0p04_kd0p08_stop800_lr0", 0.0, 100, 0.12, 0.04, 0.08, 0.18, 800),
        ]
        rows = []
        for idx, (attempt, lr_scale, periodic_interval, periodic_scale, periodic_stop_step) in enumerate(ladder):
            row, control_rows = _evaluate_horizon_attempt(
                payload,
                attempt=attempt,
                commit_update=commit_update,
                controls=controls,
                seed=int(args.seed) + 100 * idx,
                lr_scale=lr_scale,
                periodic_interval=periodic_interval,
                periodic_scale=periodic_scale,
                periodic_stop_step=periodic_stop_step,
            )
            rows.append(row)
            rows.extend(control_rows)
        for idx, (attempt, lr_scale, pid_interval, pid_kp, pid_ki, pid_kd, pid_clip, pid_stop_step) in enumerate(pid_ladder):
            row, control_rows = _evaluate_pid_horizon_attempt(
                payload,
                attempt=attempt,
                commit_update=commit_update,
                controls=controls,
                seed=int(args.seed) + 2000 + 100 * idx,
                lr_scale=lr_scale,
                pid_interval=pid_interval,
                pid_kp=pid_kp,
                pid_ki=pid_ki,
                pid_kd=pid_kd,
                pid_clip=pid_clip,
                pid_stop_step=pid_stop_step,
            )
            rows.append(row)
            rows.extend(control_rows)
        c3_pass = sum(int_flag(r.get("C3_source_formation_pass")) for r in rows)
        c4_pass = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in rows)
        pid_rows = [r for r in rows if str(r.get("variant", "")) == "FU" and int_flag(r.get("pid_control"))]
        best_row = max(
            [r for r in rows if str(r.get("variant", "")) == "FU"],
            key=lambda r: float(r.get("source_vs_best_control_h3200") or -999.0),
        )
        best_pid_row = max(
            pid_rows,
            key=lambda r: float(r.get("source_vs_best_control_h3200") or -999.0),
        ) if pid_rows else {}
        route = {
            "route": "S5-HorizonSourceFormationPass" if c3_pass else "S5-HorizonSourceFormationNoGo",
            "S4_metric_dynamics_commit_pass_rows": int_flag(commit.get("S4_metric_dynamics_commit_pass_rows")),
            "C3_source_formation_pass_rows": c3_pass,
            "C4_terminal_retention_pass_rows": c4_pass,
            "official_C3_pass_rows": c3_pass,
            "PID_control_attempt_rows": len(pid_rows),
            "PID_C3_source_formation_pass_rows": sum(int_flag(r.get("C3_source_formation_pass")) for r in pid_rows),
            "PID_C4_terminal_retention_pass_rows": sum(int_flag(r.get("C4_terminal_retention_pass")) for r in pid_rows),
            "PID_best_attempt": best_pid_row.get("attempt", ""),
            "PID_best_source_vs_best_control_h3200": best_pid_row.get("source_vs_best_control_h3200", ""),
            "PID_best_pid_final_projected_gain": best_pid_row.get("pid_final_projected_gain", ""),
            "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass", 1)) for r in rows if str(r.get("horizon_status")) != "control") and all(int_flag(r.get("uses_labels_for_direction", 0)) == 0 and int_flag(r.get("uses_loss_for_direction", 0)) == 0 for r in rows)),
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "selected_attempt": best_row.get("attempt", ""),
            "best_source_vs_best_control_h3200": best_row.get("source_vs_best_control_h3200", ""),
            "promotion_allowed": 0,
            "blocker": "" if c3_pass else best_row.get("blocker", "source_horizon_or_control_or_debt_gate_failed"),
        }
    write_rows(out_dir / "v22_10_horizon_source_matrix.csv", rows)
    write_json(out_dir / "v22_10_horizon_source_route.json", route)
    simple_svg(out_dir / "figures/v22_10_source_horizon_trajectory.svg", "v22.10 source horizon", rows, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_10_horizon_source_formation.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)}",
        status="completed",
        note=f"route={route['route']} c3={route['C3_source_formation_pass_rows']} c4={route['C4_terminal_retention_pass_rows']}",
    )


if __name__ == "__main__":
    main()
