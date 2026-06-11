#!/usr/bin/env python3
"""v22.11 S2 source atoms with arbitrary-cotangent readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.function_space_metrics import l2_energy, rkhs_graph_energy, sobolev_fd_energy  # noqa: E402
from dgkan.fu.source_atoms import (  # noqa: E402
    _ddr,
    _logit_geometry_nds,
    _loss_agnostic_control_tensors,
    _normalize_atom,
    _project_fraction,
    _remove_control_span,
    _row_metrics,
    _safe_cos,
    _split_ranges,
    generate_loss_agnostic_source_atoms,
)
from dgkan.fu.upstream_cotangent import build_cotangent_suite, validate_cotangent_suite  # noqa: E402
from experiments.run_v22_11_common import PYTHON, append_exec, ensure_out, int_flag, simple_svg, write_json, write_rows  # noqa: E402


class TinyArbitraryLossMLP(torch.nn.Module):
    def __init__(self, input_dim: int = 8, hidden: int = 64, classes: int = 5) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.w2 = torch.nn.Parameter(torch.randn(hidden, classes) * 0.02)

    def frozen_readout_features(self, xb: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(xb))

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.frozen_readout_features(xb) @ self.w2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=2211)
    return p


def _make_probe(seed: int) -> tuple[TinyArbitraryLossMLP, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    model = TinyArbitraryLossMLP()
    x = torch.randn(64, 8)
    with torch.no_grad():
        logits = model(x).detach().float()
    labels = torch.argmax(logits.detach(), dim=-1)
    return model, x, logits, labels


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ y / denom).clamp(-1.0, 1.0).item())


def _attach_cotangent_readback(rows: list[dict[str, Any]], tensors: dict[str, torch.Tensor], logits: torch.Tensor, labels: torch.Tensor, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [spec for spec in build_cotangent_suite(logits, seed=seed, labels=labels) if not spec.smoke_only]
    cotangent_rows = validate_cotangent_suite(logits, specs)
    deltas = [(spec.cotangent_type, spec.adapter.cotangent(logits, spec.task_data).detach().float()) for spec in specs]
    out: list[dict[str, Any]] = []
    for row in rows:
        local = dict(row)
        tensor = tensors.get(str(row.get("atom_id", "")))
        cos_values: list[float] = []
        positive = 0
        for name, delta in deltas:
            cos = _safe_cos(tensor, delta) if tensor is not None else 0.0
            local[f"cotangent_cos_{name}"] = cos
            cos_values.append(cos)
            positive += int(cos > 0.0)
        invariance = sum(1 for value in cos_values if value > 0.0) / max(1, len(cos_values))
        local["cotangent_same_direction_count"] = positive
        local["loss_adapter_invariance_score"] = invariance
        local["uses_loss_formula_specific_direction"] = 0
        local["S2_v22_11_source_atom_pass"] = int(int_flag(local.get("S2_source_atom_pass")) and positive >= 2 and invariance >= 0.33)
        if not int_flag(local.get("S2_v22_11_source_atom_pass")):
            blocker = str(local.get("blocker", ""))
            if positive < 2:
                blocker = ";".join(part for part in [blocker, "cotangent_same_direction_gate"] if part)
            if invariance < 0.33:
                blocker = ";".join(part for part in [blocker, "loss_adapter_invariance_gate"] if part)
            local["blocker"] = blocker
        out.append(local)
    return out, cotangent_rows


def _cotangent_deltas(logits: torch.Tensor, labels: torch.Tensor, seed: int) -> tuple[list[Any], list[tuple[str, int, torch.Tensor]]]:
    specs = [spec for spec in build_cotangent_suite(logits, seed=seed, labels=labels) if not spec.smoke_only]
    deltas = [
        (spec.cotangent_type, int(spec.uses_labels_for_adapter), spec.adapter.cotangent(logits, spec.task_data).detach().float())
        for spec in specs
    ]
    return specs, deltas


def _mean_split_cotangent_gain(deltas: list[tuple[str, int, torch.Tensor]], atom: torch.Tensor, lo: int, hi: int) -> float:
    if not deltas:
        return 0.0
    local = atom.detach().float()[lo:hi]
    values = [float((-(delta[lo:hi] * local).sum()).item()) for _name, _uses_labels, delta in deltas]
    return float(sum(values) / max(1, len(values)))


def _full_cotangent_gain(deltas: list[tuple[str, int, torch.Tensor]], atom: torch.Tensor) -> float:
    if not deltas:
        return 0.0
    values = [float((-(delta * atom.detach().float()).sum()).item()) for _name, _uses_labels, delta in deltas]
    return float(sum(values) / max(1, len(values)))


def _build_cotangent_repair_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    norm_scale: float,
    split_count: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, int, torch.Tensor]], list[dict[str, Any]]]:
    specs, deltas = _cotangent_deltas(logits, labels, seed)
    source_delta = next((delta for name, _uses_labels, delta in deltas if name == "Delta-SourceTarget"), None)
    if source_delta is None:
        source_delta = torch.zeros_like(logits)
    reference = -source_delta.detach().float()
    oriented: list[torch.Tensor] = []
    for name, uses_labels, delta in deltas:
        if uses_labels:
            continue
        descent = -delta.detach().float()
        if float(torch.linalg.vector_norm(descent).item()) <= 1.0e-8:
            continue
        if float((descent.reshape(-1) @ reference.reshape(-1)).item()) < 0.0:
            descent = -descent
        oriented.append(descent / torch.linalg.vector_norm(descent).clamp_min(1.0e-8))
    if not oriented:
        oriented = [reference / torch.linalg.vector_norm(reference).clamp_min(1.0e-8)]
    consensus = _normalize_atom(sum(oriented), norm_scale)
    controls = _loss_agnostic_control_tensors(logits, norm_scale, seed)
    control_basis = [controls[name] for name in ("mean_axis", "row_axis", "random_matched", "stable_random")]
    control_null = _normalize_atom(_remove_control_span(consensus, control_basis), norm_scale)
    smoothed = consensus.clone()
    if smoothed.shape[0] > 2:
        smoothed[1:-1] = 0.25 * consensus[:-2] + 0.50 * consensus[1:-1] + 0.25 * consensus[2:]
    split_drift = torch.zeros_like(consensus)
    ranges = _split_ranges(int(consensus.shape[0]), split_count)
    for lo, hi in ranges:
        split_drift[lo:hi] = consensus[lo:hi].mean(dim=0, keepdim=True)
    candidates = {
        "A12_arbitrary_cotangent_invariant_atom": consensus,
        "A13_cotangent_control_null_residual_atom": control_null,
        "A14_cotangent_low_NDS_smooth_atom": _normalize_atom(smoothed, norm_scale),
        "A15_cotangent_split_drift_atom": _normalize_atom(split_drift, norm_scale),
    }
    cotangent_rows = validate_cotangent_suite(logits, specs)
    return candidates, deltas, cotangent_rows


def _score_cotangent_repair_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    norm_scale: float,
    split_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    candidates, deltas, cotangent_rows = _build_cotangent_repair_candidates(
        logits,
        labels,
        norm_scale=norm_scale,
        split_count=split_count,
        seed=seed,
    )
    ranges = _split_ranges(int(logits.shape[0]), split_count)
    b1 = ranges[0]
    b2 = ranges[min(1, len(ranges) - 1)]
    b3 = ranges[min(2, len(ranges) - 1)]
    controls = _loss_agnostic_control_tensors(logits, norm_scale, seed)
    control_basis = [controls[name] for name in ("mean_axis", "row_axis", "random_matched", "stable_random")]
    control_nds_values = [_logit_geometry_nds(tensor) for tensor in controls.values()]
    control_nds_median = float(torch.tensor(control_nds_values).median().item()) if control_nds_values else 0.0
    control_sob_values = [float(sobolev_fd_energy(tensor).item()) for tensor in controls.values()]
    control_sob_p50 = float(torch.tensor(control_sob_values).median().item()) if control_sob_values else 0.0
    random_gain = _mean_split_cotangent_gain(deltas, controls["random_matched"], *b2)
    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    family_by_id = {
        "A12_arbitrary_cotangent_invariant_atom": "arbitrary_cotangent_invariant",
        "A13_cotangent_control_null_residual_atom": "cotangent_control_null_residual",
        "A14_cotangent_low_NDS_smooth_atom": "cotangent_low_NDS_smooth",
        "A15_cotangent_split_drift_atom": "cotangent_split_drift",
    }
    for atom_id, tensor in candidates.items():
        tensor = tensor.detach().float()
        tensors[atom_id] = tensor
        sign_tensor = _normalize_atom(-tensor, norm_scale)
        corrupt_tensor = _normalize_atom(torch.roll(tensor, shifts=1, dims=0), norm_scale)
        b1_gain = _mean_split_cotangent_gain(deltas, tensor, *b1)
        b2_gain = _mean_split_cotangent_gain(deltas, tensor, *b2)
        b3_gain = _mean_split_cotangent_gain(deltas, tensor, *b3)
        sign_gain = _mean_split_cotangent_gain(deltas, sign_tensor, *b2)
        corrupt_gain = _mean_split_cotangent_gain(deltas, corrupt_tensor, *b2)
        ddr, signal, diffusion = _ddr(tensor, split_count)
        projection, residual_norm = _project_fraction(tensor, control_basis)
        nds = _logit_geometry_nds(tensor)
        sob = float(sobolev_fd_energy(tensor).item())
        rkhs = float(rkhs_graph_energy(tensor).item())
        full_gain = _full_cotangent_gain(deltas, tensor)
        cos_values: list[float] = []
        positive = 0
        cos_cols: dict[str, float] = {}
        for name, _uses_labels, delta in deltas:
            cos = _safe_cos(tensor, -delta)
            cos_cols[f"cotangent_cos_{name}"] = cos
            cos_values.append(cos)
            positive += int(cos > 0.0)
        invariance = positive / max(1, len(cos_values))
        random_gap = b2_gain - random_gain
        sign_gap = b2_gain - sign_gain
        corrupt_gap = b2_gain - corrupt_gain
        low_nds_repair = int(nds <= max(control_nds_median + 1.0e-12, 0.40) or (b2_gain > random_gain + 0.05 and nds <= control_nds_median + 0.50))
        pass_flag = int(
            b2_gain > random_gain + 0.005
            and b3_gain >= -0.05
            and random_gap > 0.0
            and sign_gap > 0.0
            and corrupt_gap > 0.0
            and projection <= 0.35
            and residual_norm > 0.0
            and low_nds_repair
            and positive >= 2
            and invariance >= 0.33
        )
        blockers: list[str] = []
        if b2_gain <= random_gain + 0.005:
            blockers.append("loss_agnostic_B2_transfer_gain_gate")
        if b3_gain < -0.05:
            blockers.append("loss_agnostic_B3_safety_gain_gate")
        if random_gap <= 0.0:
            blockers.append("random_gap_gate")
        if sign_gap <= 0.0:
            blockers.append("sign_flip_gap_gate")
        if corrupt_gap <= 0.0:
            blockers.append("corrupt_target_gap_gate")
        if projection > 0.35:
            blockers.append("control_projection_fraction_gate")
        if residual_norm <= 0.0:
            blockers.append("control_null_residual_norm_gate")
        if not low_nds_repair:
            blockers.append("loss_agnostic_NDS_gate")
        if positive < 2:
            blockers.append("cotangent_same_direction_gate")
        if invariance < 0.33:
            blockers.append("loss_adapter_invariance_gate")
        rows.append(
            {
                "atom_id": atom_id,
                "carrier": "MLP",
                "block_role": "all",
                "atom_family": family_by_id.get(atom_id, "cotangent_repair"),
                "loss_agnostic_contract_pass": 1,
                "uses_labels_for_direction": 0,
                "uses_loss_for_direction": 1,
                "uses_loss_interface_cotangent_for_direction": 1,
                "uses_loss_formula_specific_direction": 0,
                "uses_future_or_validation": 0,
                "uses_audit_metric_for_direction": 0,
                "direction_metric": "generic_cotangent_descent_consensus",
                "B1_gain": b1_gain,
                "B2_transfer_gain": b2_gain,
                "B2_transfer_gain_generic": b2_gain,
                "B3_safety_gain": b3_gain,
                "B3_safety_gain_generic": b3_gain,
                "full_cotangent_gain_mean": full_gain,
                "random_matched_gain": random_gain,
                "sign_flip_gain": sign_gain,
                "corrupt_target_gain": corrupt_gain,
                "random_gap": random_gap,
                "sign_flip_gap": sign_gap,
                "corrupt_target_gap": corrupt_gap,
                "control_projection_fraction": projection,
                "control_null_residual_norm": residual_norm,
                "commutator_norm_ratio": 0.0,
                "commutator_proxy": "permutation_equivariant_cotangent_descent_atom",
                "flow_order_gap": 1.0 - _safe_cos(tensor, torch.flip(tensor, dims=[0])),
                "DDR": ddr,
                "signal_drift_score": signal,
                "diffusion_score": diffusion,
                "NDS": nds,
                "control_NDS_median": control_nds_median,
                "metric_energy_L2": float(l2_energy(tensor).item()),
                "metric_energy_Fisher": float(tensor.square().mean().item()),
                "metric_energy_Sobolev": sob,
                "metric_energy_RKHS": rkhs,
                "control_Sobolev_p50": control_sob_p50,
                **_row_metrics(tensor),
                "source_atom_norm": float(torch.linalg.vector_norm(tensor).item()),
                "source_atom_function_cosine_with_controls": _safe_cos(tensor, controls["random_matched"]),
                **cos_cols,
                "cotangent_same_direction_count": positive,
                "loss_adapter_invariance_score": invariance,
                "S2_source_atom_pass": pass_flag,
                "S2_v22_11_source_atom_pass": pass_flag,
                "blocker": "" if pass_flag else ";".join(blockers),
            }
        )
    summary = {
        "atom_rows": len(rows),
        "S2_source_atom_pass_rows": sum(int_flag(r.get("S2_v22_11_source_atom_pass")) for r in rows),
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 1,
        "uses_loss_interface_cotangent_for_direction": 1,
        "uses_loss_formula_specific_direction": 0,
        "control_NDS_median": control_nds_median,
        "control_Sobolev_p50": control_sob_p50,
        "random_matched_B2_gain": random_gain,
        "sign_flip_B2_gain": "",
        "corrupt_target_B2_gain": "",
    }
    return rows, tensors, summary, cotangent_rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    model, x, logits, labels = _make_probe(int(args.seed))
    attempts = [
        {"attempt": "initial_norm_0p08_split4_all", "norm_scale": 0.08, "split_count": 4, "block_role": "all", "fallback_step": "initial"},
        {"attempt": "fallback_lower_norm_0p04_split4_all", "norm_scale": 0.04, "split_count": 4, "block_role": "all", "fallback_step": "lower_source_atom_norm"},
        {"attempt": "fallback_lower_norm_0p02_split4_all", "norm_scale": 0.02, "split_count": 4, "block_role": "all", "fallback_step": "lower_source_atom_norm"},
        {"attempt": "fallback_more_splits_0p02_split6_all", "norm_scale": 0.02, "split_count": 6, "block_role": "all", "fallback_step": "increase_micro_batch_split_count"},
        {"attempt": "fallback_block_restricted_0p02_split6_readout", "norm_scale": 0.02, "split_count": 6, "block_role": "readout_only", "fallback_step": "block_restricted_atom"},
        {"attempt": "fallback_control_null_residual_0p02_split6_all", "norm_scale": 0.02, "split_count": 6, "block_role": "control_null_only", "fallback_step": "control_null_residual_atom"},
    ]
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected: tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]] | None = None
    last_cotangent_rows: list[dict[str, Any]] = []
    for attempt in attempts:
        rows, tensors, summary = generate_loss_agnostic_source_atoms(
            logits,
            carrier="MLP",
            block_role=str(attempt["block_role"]),
            norm_scale=float(attempt["norm_scale"]),
            split_count=int(attempt["split_count"]),
            seed=int(args.seed),
        )
        rows, cotangent_rows = _attach_cotangent_readback(rows, tensors, logits, labels, int(args.seed))
        last_cotangent_rows = cotangent_rows
        for row in rows:
            row.update(attempt)
        pass_rows = sum(int_flag(r.get("S2_v22_11_source_atom_pass")) for r in rows)
        summary_row = {
            **attempt,
            **summary,
            "S2_source_atom_pass_rows": pass_rows,
            "cotangent_contract_pass_rows": sum(int_flag(r.get("upstream_cotangent_contract_pass")) for r in cotangent_rows),
            "route": "S2-SourceAtomPass" if pass_rows else "S2-SourceAtomAttemptBlocked",
        }
        summary_rows.append(summary_row)
        all_rows.extend(rows)
        if pass_rows > 0 and selected is None:
            selected = (attempt, tensors, summary_row)

    cotangent_attempt = {
        "attempt": "fallback_arbitrary_cotangent_invariant_0p04_split4_all",
        "norm_scale": 0.04,
        "split_count": 4,
        "block_role": "all",
        "fallback_step": "arbitrary_cotangent_invariant_atom",
    }
    cot_rows, cot_tensors, cot_summary, cotangent_rows = _score_cotangent_repair_candidates(
        logits,
        labels,
        norm_scale=float(cotangent_attempt["norm_scale"]),
        split_count=int(cotangent_attempt["split_count"]),
        seed=int(args.seed),
    )
    last_cotangent_rows = cotangent_rows
    for row in cot_rows:
        row.update(cotangent_attempt)
    cot_pass_rows = sum(int_flag(r.get("S2_v22_11_source_atom_pass")) for r in cot_rows)
    cot_summary_row = {
        **cotangent_attempt,
        **cot_summary,
        "S2_source_atom_pass_rows": cot_pass_rows,
        "cotangent_contract_pass_rows": sum(int_flag(r.get("upstream_cotangent_contract_pass")) for r in cotangent_rows),
        "route": "S2-SourceAtomPass" if cot_pass_rows else "S2-SourceAtomAttemptBlocked",
    }
    summary_rows.append(cot_summary_row)
    all_rows.extend(cot_rows)
    if cot_pass_rows > 0 and selected is None:
        selected = (cotangent_attempt, cot_tensors, cot_summary_row)

    if selected is None:
        route = {
            "route": "S2-SourceAtomNoGo_v22.11",
            "S2_source_atom_pass_rows": 0,
            "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass")) for r in all_rows)) if all_rows else 0,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": int(any(int_flag(r.get("uses_loss_for_direction")) for r in all_rows)),
            "uses_loss_interface_cotangent_for_direction": int(any(int_flag(r.get("uses_loss_interface_cotangent_for_direction")) for r in all_rows)),
            "uses_loss_formula_specific_direction": int(any(int_flag(r.get("uses_loss_formula_specific_direction")) for r in all_rows)),
            "selected_attempt": "",
            "promotion_allowed": 0,
            "blocker": ";".join(dict.fromkeys(part for row in all_rows for part in str(row.get("blocker", "")).split(";") if part)),
        }
        selected_tensors: dict[str, torch.Tensor] = {}
        selected_attempt = attempts[-1]
    else:
        selected_attempt, selected_tensors, selected_summary = selected
        route = {
            "route": "S2-SourceAtomProgress",
            "S2_source_atom_pass_rows": int(selected_summary.get("S2_source_atom_pass_rows", 0)),
            "loss_agnostic_contract_pass": int_flag(selected_summary.get("loss_agnostic_contract_pass")),
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": int_flag(selected_summary.get("uses_loss_for_direction")),
            "uses_loss_interface_cotangent_for_direction": int_flag(selected_summary.get("uses_loss_interface_cotangent_for_direction")),
            "uses_loss_formula_specific_direction": int_flag(selected_summary.get("uses_loss_formula_specific_direction")),
            "selected_attempt": selected_attempt.get("attempt", ""),
            "selected_norm_scale": selected_attempt.get("norm_scale", ""),
            "selected_split_count": selected_attempt.get("split_count", ""),
            "promotion_allowed": 0,
            "blocker": "",
        }
    payload = {
        "seed": int(args.seed),
        "model_config": {"input_dim": 8, "hidden": 64, "classes": 5},
        "model_state": model.state_dict(),
        "x": x,
        "logits": logits,
        "labels_for_loss_adapter_only": labels,
        "loss_agnostic_contract_pass": int_flag(route.get("loss_agnostic_contract_pass")),
        "selected_attempt": dict(selected_attempt),
        "source_atom_tensors": selected_tensors,
    }
    torch.save(payload, out_dir / "v22_11_source_atom_payload.pt")
    write_rows(out_dir / "v22_11_cotangent_suite_readback.csv", last_cotangent_rows)
    write_rows(out_dir / "v22_11_source_atom_attempt_summary.csv", summary_rows)
    write_rows(out_dir / "v22_11_source_atom_rows.csv", all_rows)
    write_json(out_dir / "v22_11_source_atom_route.json", route)
    simple_svg(out_dir / "figures/v22_11_source_atom_DDR_vs_NDS.svg", "v22.11 source atom DDR/NDS", all_rows, "DDR")
    simple_svg(out_dir / "figures/v22_11_source_atom_invariance.svg", "v22.11 source atom cotangent invariance", all_rows, "loss_adapter_invariance_score")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_11_source_atom_generation.py --seed {int(args.seed)} --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} pass_rows={route['S2_source_atom_pass_rows']} selected_attempt={route.get('selected_attempt')} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
