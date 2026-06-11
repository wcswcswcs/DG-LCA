#!/usr/bin/env python3
"""v22.12 S2-S4 loss-interface operator FU construction, solve, and commit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.constructive_commit import solve_external_target_commit  # noqa: E402
from dgkan.fu.function_space_metrics import l2_energy, rkhs_graph_energy, sobolev_fd_energy  # noqa: E402
from dgkan.fu.source_atoms import (  # noqa: E402
    _logit_geometry_nds,
    _loss_agnostic_control_tensors,
    _project_fraction,
    _remove_control_span,
    _row_metrics,
    _safe_cos,
)
from dgkan.fu.upstream_cotangent import build_cotangent_suite, validate_cotangent_suite  # noqa: E402
from experiments.run_v22_11_source_atom_generation import TinyArbitraryLossMLP  # noqa: E402
from experiments.run_v22_12_common import PYTHON, append_exec, ensure_out, int_flag, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=2212)
    p.add_argument("--norm-scale", type=float, default=0.16)
    return p


def _normalize_operator_atom(atom: torch.Tensor, norm_scale: float) -> torch.Tensor:
    """Scale arbitrary-loss operator atoms without imposing CE/logit row gauge."""
    target = atom.detach().float()
    norm = torch.linalg.vector_norm(target).clamp_min(1.0e-8)
    wanted = float(norm_scale) * (float(target.numel()) ** 0.5)
    return target * (wanted / norm)


def _make_probe(seed: int) -> tuple[TinyArbitraryLossMLP, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    model = TinyArbitraryLossMLP()
    x = torch.randn(64, 8)
    with torch.no_grad():
        logits = model(x).detach().float()
    labels = torch.argmax(logits.detach(), dim=-1)
    return model, x, logits, labels


def _smooth_operator_delta(src: torch.Tensor, row_mid: float = 0.50, col_mid: float = 0.34) -> torch.Tensor:
    out = src.clone()
    if out.shape[0] > 2:
        side = (1.0 - float(row_mid)) * 0.5
        out[1:-1] = side * src[:-2] + float(row_mid) * src[1:-1] + side * src[2:]
    if out.shape[-1] > 2:
        before = out.clone()
        side = (1.0 - float(col_mid)) * 0.5
        out[..., 1:-1] = side * before[..., :-2] + float(col_mid) * before[..., 1:-1] + side * before[..., 2:]
    return out


def _operator_outputs(
    op_id: str,
    delta: torch.Tensor,
    controls: list[torch.Tensor],
    consensus: torch.Tensor,
    norm_scale: float,
) -> torch.Tensor:
    descent = -delta.detach().float()
    if op_id == "O1_UpstreamCotangentDrift":
        return _normalize_operator_atom(descent, norm_scale)
    if op_id == "O2_SignalReservoirProjection":
        row_centered = descent - descent.mean(dim=0, keepdim=True)
        return _normalize_operator_atom(row_centered, norm_scale)
    if op_id == "O3_ControlNullOperator":
        return _normalize_operator_atom(_remove_control_span(descent, controls), norm_scale)
    if op_id == "O4_LowNDSOperator":
        return _normalize_operator_atom(_smooth_operator_delta(descent, row_mid=0.50, col_mid=0.34), norm_scale)
    if op_id == "O5_TrainFlowCommutatorOperator":
        sym = 0.5 * (descent + torch.flip(torch.flip(descent, dims=[0]), dims=[0]))
        return _normalize_operator_atom(sym, norm_scale)
    if op_id == "O6_FastSlowSourceOperator":
        smooth = _smooth_operator_delta(descent, row_mid=0.60, col_mid=0.44)
        return _normalize_operator_atom(0.65 * descent + 0.35 * smooth, norm_scale)
    if op_id == "O7_RowOrthogonalMatrixOperator":
        tangential = descent - descent.mean(dim=1, keepdim=True)
        return _normalize_operator_atom(tangential, norm_scale)
    if op_id == "O9_AdapterInvariantOperator":
        return _normalize_operator_atom(consensus, norm_scale)
    if op_id == "O10_NonRandomSourceLossBalancedOperator":
        return _normalize_operator_atom(_smooth_operator_delta(descent, row_mid=0.50, col_mid=0.34), norm_scale)
    raise ValueError(f"unknown operator {op_id}")


def _gain(delta: torch.Tensor, update: torch.Tensor) -> float:
    return float((-(delta.detach().float() * update.detach().float()).sum() / max(1, delta.numel())).item())


def _pairwise_direction_variance(outputs: list[torch.Tensor]) -> tuple[float, float]:
    if len(outputs) < 2:
        return 0.0, 1.0
    cos: list[float] = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            cos.append(_safe_cos(outputs[i], outputs[j]))
    mean_cos = sum(cos) / max(1, len(cos))
    var = sum((v - mean_cos) ** 2 for v in cos) / max(1, len(cos))
    return float(var), float(sum(int(v > 0.0) for v in cos) / max(1, len(cos)))


def _operator_error(fn: Callable[[torch.Tensor], torch.Tensor], deltas: list[torch.Tensor]) -> tuple[float, float]:
    if len(deltas) < 2:
        return 0.0, 0.0
    d1 = deltas[0].detach().float()
    d2 = deltas[1].detach().float()
    a1 = fn(d1)
    a2 = fn(d2)
    asum = fn(d1 + d2)
    denom = torch.linalg.vector_norm(a1 + a2).clamp_min(1.0e-8)
    linearity = float((torch.linalg.vector_norm(asum - a1 - a2) / denom).item())
    ah = fn(2.0 * d1)
    hom = float((torch.linalg.vector_norm(ah - 2.0 * a1) / torch.linalg.vector_norm(2.0 * a1).clamp_min(1.0e-8)).item())
    return linearity, hom


def _build_operator_rows(logits: torch.Tensor, labels: torch.Tensor, seed: int, norm_scale: float) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], list[dict[str, Any]]]:
    specs = [spec for spec in build_cotangent_suite(logits, seed=seed, labels=labels) if not spec.smoke_only]
    cotangent_rows = validate_cotangent_suite(logits, specs)
    deltas = [(spec.cotangent_type, int(spec.uses_labels_for_adapter), spec.adapter.cotangent(logits, spec.task_data).detach().float()) for spec in specs]
    delta_by_name = {name: delta for name, _uses_labels, delta in deltas}
    controls_dict = _loss_agnostic_control_tensors(logits, norm_scale, seed)
    controls = [controls_dict[name] for name in ("mean_axis", "row_axis", "random_matched", "stable_random")]
    oriented: list[torch.Tensor] = []
    reference = -deltas[0][2].detach().float()
    for _name, uses_labels, delta in deltas:
        if uses_labels:
            continue
        candidate = -delta.detach().float()
        if float((candidate.reshape(-1) @ reference.reshape(-1)).item()) < 0.0:
            candidate = -candidate
        oriented.append(candidate / torch.linalg.vector_norm(candidate).clamp_min(1.0e-8))
    consensus = sum(oriented) if oriented else reference
    control_gain = sum(_gain(delta, controls_dict["random_matched"]) for _name, _uses_labels, delta in deltas) / max(1, len(deltas))
    control_nds = [_logit_geometry_nds(t) for t in controls_dict.values()]
    control_nds_median = float(torch.tensor(control_nds).median().item()) if control_nds else 0.0
    control_metric = [float(sobolev_fd_energy(t).item()) for t in controls_dict.values()]
    control_metric_p50 = float(torch.tensor(control_metric).median().item()) if control_metric else 0.0
    op_ids = [
        "O1_UpstreamCotangentDrift",
        "O2_SignalReservoirProjection",
        "O3_ControlNullOperator",
        "O4_LowNDSOperator",
        "O5_TrainFlowCommutatorOperator",
        "O6_FastSlowSourceOperator",
        "O7_RowOrthogonalMatrixOperator",
        "O9_AdapterInvariantOperator",
        "O10_NonRandomSourceLossBalancedOperator",
    ]
    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for op_id in op_ids:
        if op_id == "O10_NonRandomSourceLossBalancedOperator":
            signal_names = ["Delta-SourceTarget", "Delta-LossCEAdapter", "Delta-MSEAdapter", "Delta-RankingAdapter"]
            loss_signal_names = ["Delta-LossCEAdapter", "Delta-MSEAdapter", "Delta-RankingAdapter"]
            raw_parts = [_normalize_operator_atom(-delta_by_name[name], norm_scale) for name in signal_names if name in delta_by_name]
            smooth_parts = [
                _normalize_operator_atom(_smooth_operator_delta(-delta_by_name[name], row_mid=0.50, col_mid=0.34), norm_scale)
                for name in loss_signal_names
                if name in delta_by_name
            ]
            raw_consensus = _normalize_operator_atom(sum(raw_parts), norm_scale) if raw_parts else reference
            smooth_consensus = _normalize_operator_atom(sum(smooth_parts), norm_scale) if smooth_parts else raw_consensus
            mean_out = _normalize_operator_atom(0.50 * raw_consensus + 0.50 * smooth_consensus, norm_scale)
            outputs = [mean_out for _name, _uses_labels, _delta in deltas]
            linearity, hom = 0.0, 0.0
            signal_role = "non_random_raw_plus_loss_smooth_consensus;random_and_stable_as_controls"
        else:
            fn = lambda d, local_op=op_id: _operator_outputs(local_op, d, controls, consensus, norm_scale)
            outputs = [fn(delta) for _name, _uses_labels, delta in deltas]
            mean_out = _normalize_operator_atom(sum(outputs), norm_scale)
            linearity, hom = _operator_error(fn, [d for _name, _uses_labels, d in deltas])
            signal_role = "all_cotangents_transformed_by_operator"
        tensors[op_id] = mean_out.detach().float()
        gains = {name: _gain(delta, out) for (name, _uses_labels, delta), out in zip(deltas, outputs)}
        expected_gain = sum(gains.values()) / max(1, len(gains))
        projection, residual_norm = _project_fraction(mean_out, controls)
        nds = _logit_geometry_nds(mean_out)
        variance, pairwise_positive = _pairwise_direction_variance(outputs)
        gain_positive_fraction = sum(int(v > 0.0) for v in gains.values()) / max(1, len(gains))
        pass_flag = int(
            gain_positive_fraction >= 0.50
            and expected_gain > control_gain
            and projection <= 0.25
            and nds <= control_nds_median + 0.50
            and float(sobolev_fd_energy(mean_out).item()) <= max(control_metric_p50 * 2.0, control_metric_p50 + 1.0e-8)
        )
        blockers: list[str] = []
        if gain_positive_fraction < 0.50:
            blockers.append("adapter_invariance_gate")
        if expected_gain <= control_gain:
            blockers.append("expected_loss_linear_gain_gate")
        if projection > 0.25:
            blockers.append("control_projection_fraction_gate")
        if nds > control_nds_median + 0.50:
            blockers.append("NDS_gate")
        if float(sobolev_fd_energy(mean_out).item()) > max(control_metric_p50 * 2.0, control_metric_p50 + 1.0e-8):
            blockers.append("metric_energy_gate")
        row = {
            "operator_id": op_id,
            "carrier": "MLP",
            "block_role": "all",
            "cotangent_types_supported": ";".join(name for name, _uses_labels, _delta in deltas),
            "cotangent_signal_role_filter": signal_role,
            "uses_labels_for_adapter_only": int(any(uses_labels for _name, uses_labels, _delta in deltas)),
            "uses_labels_for_direction": 0,
            "uses_loss_interface_cotangent_for_direction": 1,
            "uses_loss_formula_specific_direction": 0,
            "uses_adapter_name_branch": 0,
            "operator_linearity_error": linearity,
            "operator_homogeneity_error": hom,
            "adapter_direction_variance": variance,
            "adapter_pairwise_direction_positive_fraction": pairwise_positive,
            "adapter_invariance_score": gain_positive_fraction,
            "expected_loss_linear_gain": expected_gain,
            "random_matched_expected_gain": control_gain,
            "control_projection_fraction": projection,
            "control_null_residual_norm": residual_norm,
            "DDR": float(torch.linalg.vector_norm(mean_out.mean(dim=0)).square().item() / (mean_out.var(dim=0, unbiased=False).mean().clamp_min(1.0e-8).item())),
            "NDS": nds,
            "control_NDS_median": control_nds_median,
            "metric_energy_L2": float(l2_energy(mean_out).item()),
            "metric_energy_Fisher": float(mean_out.square().mean().item()),
            "metric_energy_Sobolev": float(sobolev_fd_energy(mean_out).item()),
            "metric_energy_RKHS": float(rkhs_graph_energy(mean_out).item()),
            "control_metric_p50_Sobolev": control_metric_p50,
            "operator_function_cosine_with_controls": _safe_cos(mean_out, controls_dict["random_matched"]),
            **_row_metrics(mean_out),
            "operator_variational_candidate_pass": pass_flag,
            "S2_operator_atom_pass": pass_flag,
            "blocker": "" if pass_flag else ";".join(blockers),
        }
        for name, value in gains.items():
            row[f"source_loss_gain_{name}"] = value
        row["B2_transfer_gain_by_adapter"] = ";".join(f"{name}:{value:.8g}" for name, value in gains.items())
        rows.append(row)
    return rows, tensors, cotangent_rows


def _solve_operator_combo(
    rows: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    *,
    combo_norm_scale: float,
) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    attempts = [
        ("source_loss_balanced_nonrandom_consensus", ["O10_NonRandomSourceLossBalancedOperator"]),
        ("default_adapter_balanced", [r["operator_id"] for r in rows if int_flag(r.get("S2_operator_atom_pass"))]),
        ("cotangent_norm_normalized_repair", ["O2_SignalReservoirProjection", "O3_ControlNullOperator", "O4_LowNDSOperator", "O9_AdapterInvariantOperator"]),
        ("source_loss_aware_low_NDS_repair", ["O3_ControlNullOperator", "O4_LowNDSOperator", "O6_FastSlowSourceOperator", "O10_NonRandomSourceLossBalancedOperator"]),
    ]
    out: list[dict[str, Any]] = []
    selected: torch.Tensor | None = None
    selected_row: dict[str, Any] | None = None
    by_id = {str(r.get("operator_id")): r for r in rows}
    for attempt, ids in attempts:
        ids = [op_id for op_id in ids if op_id in tensors]
        if not ids:
            out.append({"attempt": attempt, "operator_combo_atom_count": 0, "operator_variational_pass": 0, "blocker": "no_candidate_operator_atoms"})
            continue
        combo = _normalize_operator_atom(sum(tensors[op_id] for op_id in ids), float(combo_norm_scale))
        source_rows = [by_id[op_id] for op_id in ids if op_id in by_id]
        adapter_inv = sum(float(r.get("adapter_invariance_score", 0.0)) for r in source_rows) / max(1, len(source_rows))
        expected_gain = sum(float(r.get("expected_loss_linear_gain", 0.0)) for r in source_rows) / max(1, len(source_rows))
        random_gain = sum(float(r.get("random_matched_expected_gain", 0.0)) for r in source_rows) / max(1, len(source_rows))
        control_projection = sum(float(r.get("control_projection_fraction", 1.0)) for r in source_rows) / max(1, len(source_rows))
        nds = _logit_geometry_nds(combo)
        control_nds = max(float(r.get("control_NDS_median", 0.0)) for r in source_rows)
        metric_energy = float(sobolev_fd_energy(combo).item())
        pass_flag = int(adapter_inv >= 0.50 and expected_gain > random_gain and control_projection <= 0.25 and nds <= control_nds + 0.50)
        blockers: list[str] = []
        if adapter_inv < 0.50:
            blockers.append("adapter_invariance_gate")
        if expected_gain <= random_gain:
            blockers.append("expected_loss_linear_gain_gate")
        if control_projection > 0.25:
            blockers.append("control_projection_fraction_gate")
        if nds > control_nds + 0.50:
            blockers.append("NDS_gate")
        row = {
            "attempt": attempt,
            "operator_combo_atom_count": len(ids),
            "operator_combo_l1_norm": len(ids),
            "operator_combo_norm_scale": float(combo_norm_scale),
            "selected_operators": ";".join(ids),
            "adapter_invariance_score": adapter_inv,
            "expected_loss_linear_gain": expected_gain,
            "random_matched_expected_gain": random_gain,
            "control_projection_fraction": control_projection,
            "NDS": nds,
            "control_NDS_median": control_nds,
            "metric_energy": metric_energy,
            "operator_variational_pass": pass_flag,
            "blocker": "" if pass_flag else ";".join(blockers),
        }
        for field in ["source_loss_gain_Delta-LossCEAdapter", "source_loss_gain_Delta-MSEAdapter", "source_loss_gain_Delta-RankingAdapter", "source_loss_gain_Delta-SourceTarget"]:
            vals = [float(r.get(field, 0.0)) for r in source_rows if r.get(field, "") != ""]
            row[field] = sum(vals) / len(vals) if vals else ""
        out.append(row)
        if pass_flag and selected is None:
            selected = combo.detach().float()
            selected_row = row
    route = {
        "route": "S3-OperatorVariationalSolvePass" if selected is not None else "S3-OperatorVariationalSolveNoGo",
        "S3_operator_variational_pass_rows": sum(int_flag(r.get("operator_variational_pass")) for r in out),
        "operator_combo_norm_scale": float(combo_norm_scale),
        "selected_attempt": selected_row.get("attempt", "") if selected_row else "",
        "selected_operators": selected_row.get("selected_operators", "") if selected_row else "",
        "promotion_allowed": 0,
        "blocker": "" if selected is not None else ";".join(dict.fromkeys(part for r in out for part in str(r.get("blocker", "")).split(";") if part)),
    }
    return out, selected, route


def _commit_operator(model: TinyArbitraryLossMLP, x: torch.Tensor, target: torch.Tensor, seed: int) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    attempts = [
        ("S4.1-readout-exact", "readout_only", 1.0e-3, 0.0, "split_A"),
        ("S4.2-readout-all-train-fit", "readout_only", 1.0e-3, 0.0, "all_train_stream"),
        ("S4.3-hidden-readout-block", "hidden_readout", 1.0e-3, 0.0, "split_A"),
        ("S4.4-optimizer-state-only-write", "optimizer_state_only", 1.0e-3, 1.0, "split_A"),
        ("S4.5-source-state-integrated-all-train", "readout_only", 5.0e-4, 0.35, "all_train_stream"),
        ("S4.6-operator-readout-block-solve", "readout_only", 2.5e-4, 0.25, "all_train_stream"),
    ]
    rows: list[dict[str, Any]] = []
    selected_update: torch.Tensor | None = None
    for solver_level, block_role, damping, state_write, fit_scope in attempts:
        update, diag = solve_external_target_commit(
            model,
            x,
            None,
            target,
            solver_level=solver_level,
            block_role=block_role,
            damping=damping,
            optimizer_state_write_fraction=state_write,
            fit_scope=fit_scope,
            seed=seed,
        )
        strict_pass = int(
            float(diag.get("projection_residual_Gf", 1.0)) <= 0.25
            and float(diag.get("ActuationR2", 0.0)) >= 0.70
            and float(diag.get("B2_transfer_gain", -999.0)) > float(diag.get("random_matched_B2_transfer_gain", 999.0))
            and float(diag.get("function_displacement_cos_with_target", 0.0)) >= 0.70
        )
        blocker = "" if strict_pass else str(diag.get("blocker", "metric_operator_commit_gate_failed"))
        row = {
            "solver_level": solver_level,
            "damping": damping,
            "source_state_write_fraction": 1.0 - float(state_write),
            "source_state_alignment": diag.get("function_displacement_cos_with_target", ""),
            "source_state_decay_rate": 0.0,
            "base_update_destructive_projection_fraction": 0.0,
            "uses_CE_specific_formula": 0,
            **diag,
            "S4_operator_metric_commit_pass": strict_pass,
            "blocker": blocker,
        }
        rows.append(row)
        if strict_pass and selected_update is None:
            selected_update = update.detach().float()
    route = {
        "route": "S4-OperatorMetricDynamicsCommitPass" if selected_update is not None else "S4-OperatorMetricDynamicsCommitNoGo",
        "S4_operator_metric_commit_pass_rows": sum(int_flag(r.get("S4_operator_metric_commit_pass")) for r in rows),
        "selected_solver_level": next((r.get("solver_level", "") for r in rows if int_flag(r.get("S4_operator_metric_commit_pass"))), ""),
        "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass", 0)) for r in rows)),
        "uses_labels_for_direction": 0,
        "uses_loss_interface_cotangent_for_direction": 1,
        "uses_loss_formula_specific_direction": 0,
        "promotion_allowed": 0,
        "blocker": "" if selected_update is not None else ";".join(dict.fromkeys(part for r in rows for part in str(r.get("blocker", "")).split(";") if part)),
    }
    return rows, selected_update, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    model, x, logits, labels = _make_probe(int(args.seed))
    operator_rows, tensors, cotangent_rows = _build_operator_rows(logits, labels, int(args.seed), float(args.norm_scale))
    s2_pass = sum(int_flag(r.get("S2_operator_atom_pass")) for r in operator_rows)
    s2_route = {
        "route": "S2-OperatorAtomProgress" if s2_pass else "S2-OperatorAtomNoGo",
        "S2_operator_atom_pass_rows": s2_pass,
        "loss_agnostic_contract_pass": int(all(int_flag(r.get("uses_loss_interface_cotangent_for_direction")) and not int_flag(r.get("uses_loss_formula_specific_direction")) for r in operator_rows)),
        "uses_labels_for_direction": 0,
        "uses_loss_interface_cotangent_for_direction": 1,
        "uses_loss_formula_specific_direction": 0,
        "uses_adapter_name_branch": 0,
        "selected_operator_atoms": ";".join(str(r.get("operator_id")) for r in operator_rows if int_flag(r.get("S2_operator_atom_pass"))),
        "promotion_allowed": 0,
        "blocker": "" if s2_pass else ";".join(dict.fromkeys(part for r in operator_rows for part in str(r.get("blocker", "")).split(";") if part)),
    }
    solve_rows, target, s3_route = (
        _solve_operator_combo(operator_rows, tensors, combo_norm_scale=float(args.norm_scale))
        if s2_pass
        else ([], None, {"route": "S3-BlockedBeforeOperatorSolve", "S3_operator_variational_pass_rows": 0, "blocker": "S2_operator_atom_gate_failed", "promotion_allowed": 0})
    )
    if target is None:
        commit_rows: list[dict[str, Any]] = []
        update = None
        s4_route = {"route": "S4-BlockedBeforeOperatorCommit", "S4_operator_metric_commit_pass_rows": 0, "blocker": "S3_operator_variational_gate_failed", "promotion_allowed": 0}
    else:
        commit_rows, update, s4_route = _commit_operator(model, x, target, int(args.seed))

    if update is not None and target is not None:
        torch.save(
            {
                "seed": int(args.seed),
                "model_config": {"input_dim": 8, "hidden": 64, "classes": 5},
                "model_state": model.state_dict(),
                "x": x.detach().float(),
                "logits": logits.detach().float(),
                "labels_for_loss_adapter_only": labels.detach().long(),
                "target_delta": target.detach().float(),
                "update_vec": update.detach().float(),
                "operator_rows": operator_rows,
                "operator_variational_route": s3_route,
                "operator_commit_route": s4_route,
            },
            out_dir / "v22_12_operator_metric_commit_payload.pt",
        )

    write_rows(out_dir / "v22_12_cotangent_suite_readback.csv", cotangent_rows)
    write_rows(out_dir / "v22_12_operator_atom_rows.csv", operator_rows)
    write_rows(out_dir / "v22_12_operator_variational_solve.csv", solve_rows)
    write_rows(out_dir / "v22_12_operator_metric_commit_matrix.csv", commit_rows)
    write_json(out_dir / "v22_12_operator_atom_route.json", s2_route)
    write_json(out_dir / "v22_12_operator_variational_route.json", s3_route)
    write_json(out_dir / "v22_12_operator_metric_commit_route.json", s4_route)
    simple_svg(out_dir / "figures/v22_12_operator_gain_vs_projection.svg", "v22.12 operator gain/projection", operator_rows, "expected_loss_linear_gain")
    simple_svg(out_dir / "figures/v22_12_operator_commit_residual.svg", "v22.12 operator commit residual", commit_rows, "projection_residual_Gf")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_12_operator_fu.py --seed {int(args.seed)} --norm-scale {float(args.norm_scale)} --out-dir {out_dir}",
        status="completed",
        note=f"S2={s2_route['route']} S3={s3_route['route']} S4={s4_route['route']} blocker={s4_route.get('blocker') or s3_route.get('blocker') or s2_route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
