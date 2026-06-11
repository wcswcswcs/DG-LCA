#!/usr/bin/env python3
"""v22.13 role-blind operator construction, solve, and commit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.operator_atoms_v22_13 import build_operator_atom_rows  # noqa: E402
from dgkan.fu.operator_commit import commit_operator_target  # noqa: E402
from dgkan.fu.upstream_cotangent import build_cotangent_suite, validate_cotangent_suite  # noqa: E402
from experiments.run_v22_11_source_atom_generation import TinyArbitraryLossMLP  # noqa: E402
from experiments.run_v22_13_common import PYTHON, append_exec, ensure_out, int_flag, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=2213)
    p.add_argument("--norm-scale", type=float, default=0.16)
    p.add_argument("--prefer-attempt", default="")
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


def _solve_combo(rows: list[dict[str, Any]], tensors: dict[str, torch.Tensor], prefer_attempt: str = "", norm_scale: float = 0.16) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    attempts = [
        ("low_nds_green_only", ["LIO1_LowNDSGreen"]),
        ("split_control_null_only", ["LIO2_SplitCoherentControlNull"]),
        ("kan_lowbank_spectral_only", ["LIO3_KANLowBankSpectral"]),
        ("source_loss_boundary_only", ["LIO6_SourceLossBoundary"]),
        ("low_nds_control_null_only", ["LIO7_LowNDSControlNull"]),
        ("dual_memory_boundary_combo", ["LIO5_DualMemoryRetained", "LIO6_SourceLossBoundary"]),
        ("all_passed_role_blind_atoms", [str(r.get("operator_id")) for r in rows if int_flag(r.get("S2_operator_atom_pass"))]),
    ]
    if prefer_attempt:
        attempts = sorted(attempts, key=lambda item: 0 if item[0] == prefer_attempt else 1)
    out = []
    selected: torch.Tensor | None = None
    selected_row: dict[str, Any] | None = None
    by_id = {str(r.get("operator_id")): r for r in rows}
    for name, ids in attempts:
        ids = [x for x in ids if x in tensors]
        if not ids:
            out.append({"attempt": name, "operator_combo_atom_count": 0, "operator_variational_pass": 0, "blocker": "no_atoms"})
            continue
        combo = sum(tensors[x] for x in ids) / max(1, len(ids))
        norm = torch.linalg.vector_norm(combo).clamp_min(1.0e-8)
        combo = combo * (float(norm_scale) * (float(combo.numel()) ** 0.5) / norm)
        atom_rows = [by_id[x] for x in ids if x in by_id]
        gain_fraction = sum(float(r.get("gain_positive_fraction", 0.0)) for r in atom_rows) / max(1, len(atom_rows))
        projection = sum(float(r.get("control_projection_after", 1.0)) for r in atom_rows) / max(1, len(atom_rows))
        law_pass = int(all(int_flag(r.get("S2_operator_atom_pass")) for r in atom_rows))
        pass_flag = int(law_pass and gain_fraction >= 0.50 and projection <= 0.25)
        row = {
            "attempt": name,
            "operator_combo_atom_count": len(ids),
            "selected_operators": ";".join(ids),
            "norm_scale": float(norm_scale),
            "gain_positive_fraction": gain_fraction,
            "control_projection_after": projection,
            "operator_variational_pass": pass_flag,
            "blocker": "" if pass_flag else "operator_atom_or_projection_gate_failed",
        }
        out.append(row)
        if pass_flag and selected is None:
            selected = combo.detach().float()
            selected_row = row
    route = {
        "route": "S3-OperatorVariationalSolvePass" if selected is not None else "S3-OperatorVariationalSolveNoGo",
        "S3_operator_variational_pass_rows": sum(int_flag(r.get("operator_variational_pass")) for r in out),
        "selected_attempt": selected_row.get("attempt", "") if selected_row else "",
        "selected_operators": selected_row.get("selected_operators", "") if selected_row else "",
        "norm_scale": float(norm_scale),
        "promotion_allowed": 0,
        "blocker": "" if selected is not None else ";".join(dict.fromkeys(str(r.get("blocker")) for r in out if r.get("blocker"))),
    }
    return out, selected, route


def _commit(model: TinyArbitraryLossMLP, x: torch.Tensor, target: torch.Tensor, seed: int) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    attempts = [
        ("S4.1-v2213-readout-all-train", "readout_only", 1.0e-3),
        ("S4.2-v2213-readout-damped-lowcurv", "readout_only", 2.5e-3),
        ("S4.3-v2213-hidden-readout-audit", "hidden_readout", 1.0e-3),
    ]
    rows = []
    selected: torch.Tensor | None = None
    for solver, block, damping in attempts:
        update, diag = commit_operator_target(model, x, target, solver_level=solver, block_role=block, damping=damping, fit_scope="all_train_stream", seed=seed)
        pass_flag = int(
            float(diag.get("projection_residual_Gf", 1.0)) <= 0.25
            and float(diag.get("ActuationR2", 0.0)) >= 0.70
            and float(diag.get("function_displacement_cos_with_target", 0.0)) >= 0.70
        )
        row = {**diag, "S4_operator_metric_commit_pass": pass_flag, "blocker": "" if pass_flag else str(diag.get("blocker", "commit_gate_failed"))}
        rows.append(row)
        if pass_flag and selected is None:
            selected = update.detach().float()
    route = {
        "route": "S4-OperatorMetricCommitPass" if selected is not None else "S4-OperatorMetricCommitNoGo",
        "S4_operator_metric_commit_pass_rows": sum(int_flag(r.get("S4_operator_metric_commit_pass")) for r in rows),
        "selected_solver_level": next((str(r.get("solver_level")) for r in rows if int_flag(r.get("S4_operator_metric_commit_pass"))), ""),
        "promotion_allowed": 0,
        "blocker": "" if selected is not None else ";".join(dict.fromkeys(str(r.get("blocker")) for r in rows if r.get("blocker"))),
    }
    return rows, selected, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    model, x, logits, labels = _make_probe(int(args.seed))
    specs = [s for s in build_cotangent_suite(logits, seed=int(args.seed), labels=labels) if not s.smoke_only]
    cotangent_rows = validate_cotangent_suite(logits, specs)
    cotangents = [(s.cotangent_type, s.adapter.cotangent(logits, s.task_data).detach().float(), int(s.uses_labels_for_adapter)) for s in specs]
    atom_rows, tensors = build_operator_atom_rows(logits, cotangents, norm_scale=float(args.norm_scale), seed=int(args.seed))
    s2_pass = sum(int_flag(r.get("S2_operator_atom_pass")) for r in atom_rows)
    s2_route = {
        "route": "S2-RoleBlindOperatorAtomProgress" if s2_pass else "S2-RoleBlindOperatorAtomNoGo",
        "S2_operator_atom_pass_rows": s2_pass,
        "official_path_role_blind": 1,
        "O10_counted_as_official": 0,
        "promotion_allowed": 0,
        "blocker": "" if s2_pass else ";".join(dict.fromkeys(str(r.get("blocker")) for r in atom_rows if r.get("blocker"))),
    }
    solve_rows, target, s3_route = _solve_combo(atom_rows, tensors, str(args.prefer_attempt), float(args.norm_scale)) if s2_pass else ([], None, {"route": "S3-BlockedBeforeOperatorSolve", "S3_operator_variational_pass_rows": 0, "blocker": "S2_gate_failed", "promotion_allowed": 0})
    if target is None:
        commit_rows: list[dict[str, Any]] = []
        update = None
        s4_route = {"route": "S4-BlockedBeforeOperatorCommit", "S4_operator_metric_commit_pass_rows": 0, "blocker": "S3_gate_failed", "promotion_allowed": 0}
    else:
        commit_rows, update, s4_route = _commit(model, x, target, int(args.seed))
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
                "operator_atom_rows": atom_rows,
                "operator_variational_route": s3_route,
                "operator_commit_route": s4_route,
            },
            out_dir / "v22_13_operator_commit_payload.pt",
        )
    write_rows(out_dir / "v22_13_cotangent_suite_readback.csv", cotangent_rows)
    write_rows(out_dir / "v22_13_operator_atom_matrix.csv", atom_rows)
    write_rows(out_dir / "v22_13_metric_geometry_matrix.csv", atom_rows)
    write_rows(out_dir / "v22_13_variational_operator_solve.csv", solve_rows)
    write_rows(out_dir / "v22_13_operator_commit_matrix.csv", commit_rows)
    write_json(out_dir / "v22_13_operator_atom_route.json", s2_route)
    write_json(out_dir / "v22_13_operator_variational_route.json", s3_route)
    write_json(out_dir / "v22_13_operator_commit_route.json", s4_route)
    simple_svg(out_dir / "figures/v22_13_operator_law_dashboard.svg", "v22.13 role-blind operator atoms", atom_rows, "gain_positive_fraction")
    simple_svg(out_dir / "figures/v22_13_metric_as_operator_pareto.svg", "v22.13 metric/operator target", atom_rows, "NDS_reduction")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_13_operator_construct.py --seed {int(args.seed)} --norm-scale {float(args.norm_scale)} --prefer-attempt {args.prefer_attempt} --out-dir {out_dir}", status="completed" if int_flag(s4_route.get("S4_operator_metric_commit_pass_rows")) else "blocked", note=f"S2={s2_route['route']} S3={s3_route['route']} S4={s4_route['route']} blocker={s4_route.get('blocker') or s3_route.get('blocker') or s2_route.get('blocker')}")


if __name__ == "__main__":
    main()
