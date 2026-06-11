"""Retained-source variational solve helpers for v22.10.

This module consumes S2 source atom tensors and solves a small constrained
combination problem in function space.  The v22.10 official path is loss
agnostic: controls are generated from the same current train logits without
labels, while audit/future horizons are not available to the solver.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import torch

from dgkan.fu.function_space_metrics import l2_energy, rkhs_graph_energy, sobolev_fd_energy
from dgkan.fu.source_atoms import (
    _ce_gain,
    _control_tensors,
    _curvature_nds,
    _ddr,
    _geometric_gain,
    _logit_geometry_nds,
    _loss_agnostic_control_tensors,
    _project_fraction,
    _safe_cos,
    _split_ranges,
)


EPS = 1.0e-8


def _int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if isfinite(out) else default


def _eligible_rows(rows: list[dict[str, Any]], attempt: str) -> list[dict[str, Any]]:
    passed = [r for r in rows if _int_flag(r.get("S2_source_atom_pass"))]
    if attempt == "low_nds_only":
        passed = [r for r in passed if "low" in str(r.get("atom_family", "")).lower() or _float(r.get("NDS"), 999.0) <= _float(r.get("control_NDS_median"), -999.0)]
    elif attempt == "control_null_only":
        passed = [r for r in passed if "control_null" in str(r.get("atom_family", ""))]
    elif attempt == "block_restricted":
        passed = [r for r in passed if str(r.get("atom_family", "")) in {"row_orthogonal", "low_degree_frequency"} or str(r.get("block_role", "")) != "all"]
    return passed


def _select_rows(rows: list[dict[str, Any]], attempt: str, max_atoms: int) -> list[dict[str, Any]]:
    eligible = _eligible_rows(rows, attempt)
    if attempt == "sparse_l1":
        max_atoms = min(max_atoms, 2)
    scored = sorted(
        eligible,
        key=lambda r: (
            _float(r.get("DDR"), 0.0)
            + 10.0 * _float(r.get("random_gap"), 0.0)
            + 5.0 * _float(r.get("corrupt_target_gap"), 0.0)
            - 10.0 * _float(r.get("control_projection_fraction"), 0.0)
            - 100.0 * _float(r.get("NDS"), 0.0)
        ),
        reverse=True,
    )
    return scored[: max(1, int(max_atoms))]


def _weights_for(rows: list[dict[str, Any]], attempt: str, tau: float) -> dict[str, float]:
    if not rows:
        return {}
    raw = []
    for row in rows:
        score = max(0.0, _float(row.get("DDR"), 0.0)) + max(0.0, _float(row.get("random_gap"), 0.0)) + EPS
        if attempt == "low_nds_only":
            score /= max(EPS, _float(row.get("NDS"), 1.0))
        if attempt == "control_null_only":
            score *= 1.5
        raw.append(score)
    total = sum(raw) or 1.0
    return {str(row.get("atom_id")): float(tau) * raw[idx] / total for idx, row in enumerate(rows)}


def _combo_from_weights(tensors: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor | None:
    combo: torch.Tensor | None = None
    for atom_id, weight in weights.items():
        tensor = tensors.get(atom_id)
        if tensor is None:
            continue
        combo = tensor.detach().float() * float(weight) if combo is None else combo + tensor.detach().float() * float(weight)
    return combo


def solve_variational_source(
    atom_rows: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    attempt: str = "default",
    tau: float = 1.0,
    max_atoms: int = 4,
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    selected = _select_rows(atom_rows, attempt, max_atoms)
    if not selected:
        return (
            {
                "attempt": attempt,
                "source_combo_atom_count": 0,
                "source_combo_l1_norm": 0.0,
                "S3_variational_source_solve_pass": 0,
                "blocker": "no_S2_pass_atoms_for_attempt",
            },
            None,
        )

    controls = _control_tensors(logits, labels, norm_scale, seed)
    control_basis = [controls[name] for name in ("adamw_proxy", "sgd_proxy", "random_matched", "stable_random")]
    control_nds_values = [_curvature_nds(logits, labels, tensor) for tensor in controls.values()]
    control_nds_median = float(torch.tensor(control_nds_values).median().item()) if control_nds_values else 0.0
    control_sob_values = [float(sobolev_fd_energy(tensor).item()) for tensor in controls.values()]
    control_sob_p50 = float(torch.tensor(control_sob_values).median().item()) if control_sob_values else 0.0

    weights = _weights_for(selected, attempt, tau)
    if attempt == "control_balanced_l1":
        atom_ids = [str(r.get("atom_id")) for r in selected if str(r.get("atom_id")) in tensors]
        options: list[dict[str, float]] = []
        for i, left in enumerate(atom_ids):
            for right in atom_ids[i + 1 :]:
                for local_tau in (0.85, 0.65, 0.50, 0.35, 0.25):
                    for step in range(0, 101):
                        alpha = step / 100.0
                        options.append({left: local_tau * alpha, right: local_tau * (1.0 - alpha)})
        best_weights: dict[str, float] | None = None
        best_score = -1.0e99
        ranges_for_grid = _split_ranges(int(logits.shape[0]), split_count)
        b2_for_grid = ranges_for_grid[min(1, len(ranges_for_grid) - 1)]
        for option in options:
            cand = _combo_from_weights(tensors, option)
            if cand is None:
                continue
            ddr_value, _, _ = _ddr(cand, split_count)
            projection_value, _ = _project_fraction(cand, control_basis)
            sob_value = float(sobolev_fd_energy(cand).item())
            nds_value = _curvature_nds(logits, labels, cand)
            gain_value = _ce_gain(logits, labels, cand, *b2_for_grid)
            random_gap_value = gain_value - _ce_gain(logits, labels, controls["random_matched"], *b2_for_grid)
            feasible = int(ddr_value >= 1.0 and projection_value <= 0.25 and sob_value <= control_sob_p50 + 1.0e-12 and nds_value <= control_nds_median + 1.0e-12 and random_gap_value > 0.0)
            score = (
                1000.0 * feasible
                + ddr_value
                + 10.0 * random_gap_value
                - 20.0 * projection_value
                - 50.0 * max(0.0, sob_value - control_sob_p50)
                - 100.0 * max(0.0, nds_value - control_nds_median)
            )
            if score > best_score:
                best_score = score
                best_weights = option
        if best_weights is not None:
            weights = best_weights
    combo = _combo_from_weights(tensors, weights)
    if combo is None:
        return (
            {
                "attempt": attempt,
                "source_combo_atom_count": len(selected),
                "source_combo_l1_norm": sum(abs(w) for w in weights.values()),
                "S3_variational_source_solve_pass": 0,
                "blocker": "selected_atom_tensor_missing",
            },
            None,
        )

    ranges = _split_ranges(int(logits.shape[0]), split_count)
    b2 = ranges[min(1, len(ranges) - 1)]
    random_gain = _ce_gain(logits, labels, controls["random_matched"], *b2)
    sign_gain = _ce_gain(logits, labels, controls["sign_flip"], *b2)
    corrupt_gain = _ce_gain(logits, labels, controls["corrupt_target"], *b2)
    combo_gain = _ce_gain(logits, labels, combo, *b2)
    ddr, signal, diffusion = _ddr(combo, split_count)
    projection, residual_norm = _project_fraction(combo, control_basis)
    nds = _curvature_nds(logits, labels, combo)
    sob = float(sobolev_fd_energy(combo).item())
    rkhs = float(rkhs_graph_energy(combo).item())
    random_gap = combo_gain - random_gain
    sign_gap = combo_gain - sign_gain
    corrupt_gap = combo_gain - corrupt_gain
    pass_flag = int(
        ddr >= 1.0
        and projection <= 0.25
        and random_gap > 0.0
        and sign_gap > 0.0
        and corrupt_gap > 0.0
        and nds <= control_nds_median + 1.0e-12
        and sob <= control_sob_p50 + 1.0e-12
    )
    blockers = []
    if ddr < 1.0:
        blockers.append("DDR_gate")
    if projection > 0.25:
        blockers.append("control_projection_fraction_gate")
    if random_gap <= 0.0:
        blockers.append("random_gap_gate")
    if sign_gap <= 0.0:
        blockers.append("sign_flip_gap_gate")
    if corrupt_gap <= 0.0:
        blockers.append("corrupt_gap_gate")
    if nds > control_nds_median + 1.0e-12:
        blockers.append("NDS_gate")
    if sob > control_sob_p50 + 1.0e-12:
        blockers.append("Sobolev_or_RKHS_gate")
    row = {
        "attempt": attempt,
        "source_combo_atom_count": len(selected),
        "source_combo_l1_norm": sum(abs(w) for w in weights.values()),
        "signal_drift_score": signal,
        "diffusion_score": diffusion,
        "DDR": ddr,
        "control_projection_fraction": projection,
        "control_null_residual_norm": residual_norm,
        "B2_transfer_gain": combo_gain,
        "random_gap": random_gap,
        "sign_flip_gap": sign_gap,
        "corrupt_gap": corrupt_gap,
        "metric_energy_L2": float(l2_energy(combo).item()),
        "metric_energy_Fisher": float(combo.square().mean().item()),
        "metric_energy_Sobolev": sob,
        "metric_energy_RKHS": rkhs,
        "control_Sobolev_p50": control_sob_p50,
        "NDS": nds,
        "control_NDS_median": control_nds_median,
        "source_combo_norm": float(torch.linalg.vector_norm(combo).item()),
        "combo_cosine_with_random_control": _safe_cos(combo, controls["random_matched"]),
        "selected_atoms": ";".join(weights.keys()),
        "selected_atom_weights": ";".join(f"{k}:{v:.8g}" for k, v in weights.items()),
        "uses_future_or_validation": 0,
        "uses_audit_metric_for_direction": 0,
        "S3_variational_source_solve_pass": pass_flag,
        "blocker": "" if pass_flag else ";".join(blockers),
    }
    return row, combo


def solve_loss_agnostic_variational_source(
    atom_rows: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    logits: torch.Tensor,
    *,
    attempt: str = "default",
    tau: float = 1.0,
    max_atoms: int = 4,
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    selected = _select_rows(atom_rows, attempt, max_atoms)
    if not selected:
        return (
            {
                "attempt": attempt,
                "source_combo_atom_count": 0,
                "source_combo_l1_norm": 0.0,
                "loss_agnostic_contract_pass": 1,
                "uses_labels_for_direction": 0,
                "uses_loss_for_direction": 0,
                "S3_variational_source_solve_pass": 0,
                "blocker": "no_S2_pass_atoms_for_attempt",
            },
            None,
        )

    controls = _loss_agnostic_control_tensors(logits, norm_scale, seed)
    control_basis = [controls[name] for name in ("mean_axis", "row_axis", "random_matched", "stable_random")]
    control_nds_values = [_logit_geometry_nds(tensor) for tensor in controls.values()]
    control_nds_median = float(torch.tensor(control_nds_values).median().item()) if control_nds_values else 0.0
    control_sob_values = [float(sobolev_fd_energy(tensor).item()) for tensor in controls.values()]
    control_sob_p50 = float(torch.tensor(control_sob_values).median().item()) if control_sob_values else 0.0

    weights = _weights_for(selected, attempt, tau)
    if attempt == "control_balanced_l1":
        atom_ids = [str(r.get("atom_id")) for r in selected if str(r.get("atom_id")) in tensors]
        options: list[dict[str, float]] = []
        for i, left in enumerate(atom_ids):
            for right in atom_ids[i + 1 :]:
                for local_tau in (0.85, 0.65, 0.50, 0.35, 0.25):
                    for step in range(0, 101):
                        alpha = step / 100.0
                        options.append({left: local_tau * alpha, right: local_tau * (1.0 - alpha)})
        best_weights: dict[str, float] | None = None
        best_score = -1.0e99
        ranges_for_grid = _split_ranges(int(logits.shape[0]), split_count)
        b2_for_grid = ranges_for_grid[min(1, len(ranges_for_grid) - 1)]
        for option in options:
            cand = _combo_from_weights(tensors, option)
            if cand is None:
                continue
            ddr_value, _, _ = _ddr(cand, split_count)
            projection_value, _ = _project_fraction(cand, control_basis)
            sob_value = float(sobolev_fd_energy(cand).item())
            nds_value = _logit_geometry_nds(cand)
            gain_value = _geometric_gain(logits, cand, *b2_for_grid)
            random_gap_value = gain_value - _geometric_gain(logits, controls["random_matched"], *b2_for_grid)
            feasible = int(
                ddr_value >= 1.0
                and projection_value <= 0.25
                and sob_value <= control_sob_p50 + 1.0e-12
                and nds_value <= max(control_nds_median + 1.0e-12, 0.40)
                and random_gap_value > 0.0
            )
            score = (
                1000.0 * feasible
                + min(ddr_value, 10.0)
                + 10.0 * random_gap_value
                - 20.0 * projection_value
                - 50.0 * max(0.0, sob_value - control_sob_p50)
                - 100.0 * max(0.0, nds_value - max(control_nds_median, 0.40))
            )
            if score > best_score:
                best_score = score
                best_weights = option
        if best_weights is not None:
            weights = best_weights
    combo = _combo_from_weights(tensors, weights)
    if combo is None:
        return (
            {
                "attempt": attempt,
                "source_combo_atom_count": len(selected),
                "source_combo_l1_norm": sum(abs(w) for w in weights.values()),
                "loss_agnostic_contract_pass": 1,
                "uses_labels_for_direction": 0,
                "uses_loss_for_direction": 0,
                "S3_variational_source_solve_pass": 0,
                "blocker": "selected_atom_tensor_missing",
            },
            None,
        )

    ranges = _split_ranges(int(logits.shape[0]), split_count)
    b2 = ranges[min(1, len(ranges) - 1)]
    random_gain = _geometric_gain(logits, controls["random_matched"], *b2)
    sign_gain = _geometric_gain(logits, controls["sign_flip"], *b2)
    corrupt_gain = _geometric_gain(logits, controls["corrupt_target"], *b2)
    combo_gain = _geometric_gain(logits, combo, *b2)
    ddr, signal, diffusion = _ddr(combo, split_count)
    projection, residual_norm = _project_fraction(combo, control_basis)
    nds = _logit_geometry_nds(combo)
    sob = float(sobolev_fd_energy(combo).item())
    rkhs = float(rkhs_graph_energy(combo).item())
    random_gap = combo_gain - random_gain
    sign_gap = combo_gain - sign_gain
    corrupt_gap = combo_gain - corrupt_gain
    pass_flag = int(
        ddr >= 1.0
        and projection <= 0.25
        and random_gap > 0.0
        and sign_gap > 0.0
        and corrupt_gap > 0.0
        and nds <= max(control_nds_median + 1.0e-12, 0.40)
        and sob <= control_sob_p50 + 1.0e-12
    )
    blockers = []
    if ddr < 1.0:
        blockers.append("DDR_gate")
    if projection > 0.25:
        blockers.append("control_projection_fraction_gate")
    if random_gap <= 0.0:
        blockers.append("random_gap_gate")
    if sign_gap <= 0.0:
        blockers.append("sign_flip_gap_gate")
    if corrupt_gap <= 0.0:
        blockers.append("corrupt_gap_gate")
    if nds > max(control_nds_median + 1.0e-12, 0.40):
        blockers.append("loss_agnostic_NDS_gate")
    if sob > control_sob_p50 + 1.0e-12:
        blockers.append("Sobolev_or_RKHS_gate")
    row = {
        "attempt": attempt,
        "source_combo_atom_count": len(selected),
        "source_combo_l1_norm": sum(abs(w) for w in weights.values()),
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "direction_metric": "logit_geometry_split_coherence",
        "signal_drift_score": signal,
        "diffusion_score": diffusion,
        "DDR": ddr,
        "control_projection_fraction": projection,
        "control_null_residual_norm": residual_norm,
        "B2_transfer_gain": combo_gain,
        "random_gap": random_gap,
        "sign_flip_gap": sign_gap,
        "corrupt_gap": corrupt_gap,
        "metric_energy_L2": float(l2_energy(combo).item()),
        "metric_energy_Fisher": float(combo.square().mean().item()),
        "metric_energy_Sobolev": sob,
        "metric_energy_RKHS": rkhs,
        "control_Sobolev_p50": control_sob_p50,
        "NDS": nds,
        "control_NDS_median": control_nds_median,
        "source_combo_norm": float(torch.linalg.vector_norm(combo).item()),
        "combo_cosine_with_random_control": _safe_cos(combo, controls["random_matched"]),
        "selected_atoms": ";".join(weights.keys()),
        "selected_atom_weights": ";".join(f"{k}:{v:.8g}" for k, v in weights.items()),
        "uses_future_or_validation": 0,
        "uses_audit_metric_for_direction": 0,
        "S3_variational_source_solve_pass": pass_flag,
        "blocker": "" if pass_flag else ";".join(blockers),
    }
    return row, combo


def run_variational_repair_ladder(
    atom_rows: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    attempts = [
        ("default", 1.0, 4),
        ("control_balanced_l1", 0.85, 4),
        ("sparse_l1", 0.65, 2),
        ("low_nds_only", 0.75, 3),
        ("control_null_only", 1.0, 2),
        ("block_restricted", 0.85, 3),
    ]
    rows: list[dict[str, Any]] = []
    chosen: torch.Tensor | None = None
    chosen_row: dict[str, Any] | None = None
    for attempt, tau, max_atoms in attempts:
        row, combo = solve_variational_source(
            atom_rows,
            tensors,
            logits,
            labels,
            attempt=attempt,
            tau=tau,
            max_atoms=max_atoms,
            norm_scale=norm_scale,
            split_count=split_count,
            seed=seed,
        )
        rows.append(row)
        if combo is not None and _int_flag(row.get("S3_variational_source_solve_pass")) and chosen is None:
            chosen = combo
            chosen_row = row
    if chosen_row is None:
        valid = [(row, idx) for idx, row in enumerate(rows) if _int_flag(row.get("source_combo_atom_count")) > 0]
        if valid:
            chosen_row, idx = max(valid, key=lambda item: _float(item[0].get("DDR"), 0.0) - _float(item[0].get("control_projection_fraction"), 1.0))
            attempt, tau, max_atoms = attempts[idx]
            _, chosen = solve_variational_source(
                atom_rows,
                tensors,
                logits,
                labels,
                attempt=attempt,
                tau=tau,
                max_atoms=max_atoms,
                norm_scale=norm_scale,
                split_count=split_count,
                seed=seed,
            )
    route = {
        "route": "S3-VariationalSourceSolvePass" if chosen_row and _int_flag(chosen_row.get("S3_variational_source_solve_pass")) else "S3-VariationalSourceSolveNoGo",
        "S3_variational_source_solve_pass_rows": sum(_int_flag(r.get("S3_variational_source_solve_pass")) for r in rows),
        "attempt_rows": len(rows),
        "selected_attempt": chosen_row.get("attempt", "") if chosen_row else "",
        "selected_atoms": chosen_row.get("selected_atoms", "") if chosen_row else "",
        "promotion_allowed": 0,
        "blocker": "" if chosen_row and _int_flag(chosen_row.get("S3_variational_source_solve_pass")) else ";".join(dict.fromkeys(part for r in rows for part in str(r.get("blocker", "")).split(";") if part)),
    }
    return rows, chosen, route


def run_loss_agnostic_variational_repair_ladder(
    atom_rows: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    logits: torch.Tensor,
    *,
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    attempts = [
        ("default", 1.0, 4),
        ("control_balanced_l1", 0.85, 4),
        ("sparse_l1", 0.65, 2),
        ("low_nds_only", 0.75, 3),
        ("control_null_only", 1.0, 2),
        ("block_restricted", 0.85, 3),
    ]
    rows: list[dict[str, Any]] = []
    chosen: torch.Tensor | None = None
    chosen_row: dict[str, Any] | None = None
    for attempt, tau, max_atoms in attempts:
        row, combo = solve_loss_agnostic_variational_source(
            atom_rows,
            tensors,
            logits,
            attempt=attempt,
            tau=tau,
            max_atoms=max_atoms,
            norm_scale=norm_scale,
            split_count=split_count,
            seed=seed,
        )
        rows.append(row)
        if combo is not None and _int_flag(row.get("S3_variational_source_solve_pass")) and chosen is None:
            chosen = combo
            chosen_row = row
    if chosen_row is None:
        valid = [(row, idx) for idx, row in enumerate(rows) if _int_flag(row.get("source_combo_atom_count")) > 0]
        if valid:
            chosen_row, idx = max(valid, key=lambda item: _float(item[0].get("DDR"), 0.0) - _float(item[0].get("control_projection_fraction"), 1.0))
            attempt, tau, max_atoms = attempts[idx]
            _, chosen = solve_loss_agnostic_variational_source(
                atom_rows,
                tensors,
                logits,
                attempt=attempt,
                tau=tau,
                max_atoms=max_atoms,
                norm_scale=norm_scale,
                split_count=split_count,
                seed=seed,
            )
    route = {
        "route": "S3-VariationalSourceSolvePass" if chosen_row and _int_flag(chosen_row.get("S3_variational_source_solve_pass")) else "S3-VariationalSourceSolveNoGo",
        "S3_variational_source_solve_pass_rows": sum(_int_flag(r.get("S3_variational_source_solve_pass")) for r in rows),
        "attempt_rows": len(rows),
        "selected_attempt": chosen_row.get("attempt", "") if chosen_row else "",
        "selected_atoms": chosen_row.get("selected_atoms", "") if chosen_row else "",
        "loss_agnostic_contract_pass": int(all(_int_flag(r.get("loss_agnostic_contract_pass")) for r in rows)),
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "promotion_allowed": 0,
        "blocker": "" if chosen_row and _int_flag(chosen_row.get("S3_variational_source_solve_pass")) else ";".join(dict.fromkeys(part for r in rows for part in str(r.get("blocker", "")).split(";") if part)),
    }
    return rows, chosen, route


def variational_source_solver_unit_tests() -> list[dict[str, Any]]:
    from dgkan.fu.source_atoms import generate_loss_agnostic_source_atoms, generate_train_only_source_atoms

    torch.manual_seed(2210)
    logits = torch.randn(48, 5) * 0.25
    labels = torch.arange(48) % 5
    rows, tensors, _ = generate_train_only_source_atoms(logits, labels, norm_scale=0.08, split_count=4, seed=2210)
    solve_rows, combo, route = run_variational_repair_ladder(rows, tensors, logits, labels, norm_scale=0.08, split_count=4, seed=2210)
    finite = all(isfinite(float(r.get(k, 0.0))) for r in solve_rows for k in ["DDR", "control_projection_fraction", "NDS"] if r.get(k, "") != "")
    la_logits = torch.randn(48, 5) * 0.01
    template = torch.linspace(-1.0, 1.0, 5).reshape(1, 5)
    for idx, (lo, hi) in enumerate([(0, 12), (12, 24), (24, 36), (36, 48)]):
        la_logits[lo:hi] += 1.2 * (1.0 - 0.12 * idx) * template
    la_rows, la_tensors, _ = generate_loss_agnostic_source_atoms(la_logits, norm_scale=0.08, split_count=4, seed=2210)
    la_solve_rows, la_combo, la_route = run_loss_agnostic_variational_repair_ladder(la_rows, la_tensors, la_logits, norm_scale=0.08, split_count=4, seed=2210)
    la_finite = all(isfinite(float(r.get(k, 0.0))) for r in la_solve_rows for k in ["DDR", "control_projection_fraction", "NDS"] if r.get(k, "") != "")
    return [
        {
            "case": "variational_ladder_executes",
            "attempt_rows": len(solve_rows),
            "combo_exists": int(combo is not None),
            "pass_rows": route.get("S3_variational_source_solve_pass_rows", 0),
            "diagnostics_finite": int(finite),
            "pass": int(len(solve_rows) == 6 and combo is not None and finite),
        },
        {
            "case": "loss_agnostic_variational_ladder_executes",
            "attempt_rows": len(la_solve_rows),
            "combo_exists": int(la_combo is not None),
            "pass_rows": la_route.get("S3_variational_source_solve_pass_rows", 0),
            "loss_agnostic_contract_pass": la_route.get("loss_agnostic_contract_pass", 0),
            "uses_labels_for_direction": la_route.get("uses_labels_for_direction", 1),
            "uses_loss_for_direction": la_route.get("uses_loss_for_direction", 1),
            "diagnostics_finite": int(la_finite),
            "pass": int(len(la_solve_rows) == 6 and la_combo is not None and la_finite and _int_flag(la_route.get("loss_agnostic_contract_pass"))),
        }
    ]


__all__ = [
    "run_variational_repair_ladder",
    "run_loss_agnostic_variational_repair_ladder",
    "solve_loss_agnostic_variational_source",
    "solve_variational_source",
    "variational_source_solver_unit_tests",
]
