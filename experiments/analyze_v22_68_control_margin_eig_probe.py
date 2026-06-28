#!/usr/bin/env python3
"""Train-only generalized-eigen chart probe for v22.68 Part E failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

import torch  # noqa: E402

import experiments.run_v22_68_control_contrastive_kan_gauge_carrier_mpfu as r  # noqa: E402


OUT_ROOT = ROOT / "results" / "v22_68"


def norm_cols(phi: Any) -> Any:
    return phi / phi.float().norm(dim=0).clamp_min(1.0e-12).view(1, -1)


def chart_stats(phi: Any) -> tuple[float, float]:
    visible = r.v67.part_c_column_visibility(phi.float())
    gram = phi.double().transpose(0, 1) @ phi.double() / max(1, int(phi.shape[0]))
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    evals = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(0, 1)) + 1.0e-8 * eye)
    cond = float((evals.max() / evals.min().clamp_min(1.0e-8)).detach().cpu().item())
    return float(visible["cvar25"]), cond


def eig_probe(phi_source: Any, source_target: Any, phi_witness: Any, witness_target: Any, *, rank: int, lam: float, seed: int) -> dict[str, Any]:
    x = phi_source.double()
    xw = phi_witness.double()
    target = source_target.double().reshape(-1, 1)
    energy = x.square().sum(dim=0)
    visible = energy / energy.max().clamp_min(1.0e-12)
    mask = visible >= 0.25
    if int(mask.sum().detach().cpu().item()) < int(rank):
        keep = torch.argsort(visible, descending=True)[: int(rank)]
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[keep] = True
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    x = x.index_select(1, idx)
    xw = xw.index_select(1, idx)

    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed) + 777)
    random_target = torch.randn(target.shape, device=x.device, dtype=x.dtype, generator=gen)
    task_unit = target / target.norm().clamp_min(1.0e-12)
    ctrl_unit = random_target / random_target.norm().clamp_min(1.0e-12)
    s_task = x.transpose(0, 1) @ (task_unit @ task_unit.transpose(0, 1)) @ x
    s_ctrl = x.transpose(0, 1) @ (ctrl_unit @ ctrl_unit.transpose(0, 1)) @ x
    gram = x.transpose(0, 1) @ x / max(1, int(x.shape[0])) + 1.0e-5 * torch.eye(int(x.shape[1]), device=x.device, dtype=x.dtype)
    evals, evecs = torch.linalg.eigh(0.5 * (gram + gram.transpose(0, 1)))
    inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(1.0e-5))) @ evecs.transpose(0, 1)
    contrast = s_task - float(lam) * s_ctrl
    matrix = inv_sqrt @ (0.5 * (contrast + contrast.transpose(0, 1))) @ inv_sqrt
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.transpose(0, 1)))
    order = torch.argsort(values, descending=True)[: int(rank)]
    selected = norm_cols((xw @ (inv_sqrt @ vectors[:, order])).float())

    selected_energy = selected.square().sum(dim=0)
    gen.manual_seed(int(seed) + 999)
    mix = torch.randn(int(xw.shape[1]), int(rank), device=xw.device, dtype=xw.dtype, generator=gen)
    q, _ = torch.linalg.qr(mix, mode="reduced")
    random_phi = (xw @ q[:, : int(rank)]).float()
    random_phi = random_phi * torch.sqrt(selected_energy / random_phi.square().sum(dim=0).clamp_min(1.0e-12)).view(1, -1)

    ctrl_values, ctrl_vectors = torch.linalg.eigh(inv_sqrt @ (0.5 * (s_ctrl + s_ctrl.transpose(0, 1))) @ inv_sqrt)
    ctrl_order = torch.argsort(ctrl_values, descending=True)[: int(rank)]
    control_phi = norm_cols((xw @ (inv_sqrt @ ctrl_vectors[:, ctrl_order])).float())

    task_capacity = r.projection_capacity(selected, witness_target)["capacity"]
    random_capacity = r.projection_capacity(random_phi, witness_target)["capacity"]
    control_capacity = r.projection_capacity(control_phi, witness_target)["capacity"]
    cvar25, condition = chart_stats(selected)
    return {
        "control_contrastive_margin": float(task_capacity) - max(float(random_capacity), float(control_capacity)),
        "task_capacity": float(task_capacity),
        "same_energy_random_capacity": float(random_capacity),
        "same_control_margin_random_capacity": float(control_capacity),
        "beats_same_energy_random_preflight": int(float(task_capacity) > float(random_capacity)),
        "beats_same_control_margin_random_preflight": int(float(task_capacity) > float(control_capacity)),
        "readout_visible_energy_CVaR25": cvar25,
        "basis_Gram_condition": condition,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(math.floor((len(values) - 1) * p))))
    return values[idx]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    parser.add_argument("--lambdas", default="0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0")
    parser.add_argument("--ranks", default="4,8")
    args = parser.parse_args()
    device = r.torch_device(args.device)
    lambdas = [float(x) for x in str(args.lambdas).split(",") if x.strip()]
    ranks = [int(x) for x in str(args.ranks).split(",") if x.strip()]
    detail: list[dict[str, Any]] = []
    for dataset in r.DATASETS_KAN:
        for seed in r.SEEDS:
            bundle = r.v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_source, y_source, x_witness, y_witness = r.split_source_witness(bundle, device, int(args.metric_batch_size))
            for arch in r.KAN_ARCHES:
                base = r.make_base_for_arch(arch, bundle, device, int(args.hidden), int(seed), args)
                phi_source, _source_diag = r.design_matrix(base, x_source, y_source, bundle, args)
                phi_witness, _witness_diag = r.design_matrix(base, x_witness, y_witness, bundle, args)
                source_target = r.task_target(base, x_source, y_source, bundle)
                witness_target = r.task_target(base, x_witness, y_witness, bundle)
                for rank in ranks:
                    for lam in lambdas:
                        out = eig_probe(
                            phi_source,
                            source_target,
                            phi_witness,
                            witness_target,
                            rank=int(rank),
                            lam=float(lam),
                            seed=int(seed) * 1009 + sum(ord(c) for c in f"{dataset}:{arch}:{rank}:{lam}"),
                        )
                        detail.append({"dataset": dataset, "seed": int(seed), "architecture": arch, "rank": int(rank), "lambda_ctrl": float(lam), **out})
    summary_rows: list[dict[str, Any]] = []
    for arch in r.KAN_ARCHES:
        for rank in ranks:
            for lam in lambdas:
                sub = [row for row in detail if row["architecture"] == arch and int(row["rank"]) == int(rank) and float(row["lambda_ctrl"]) == float(lam)]
                margins = [float(row["control_contrastive_margin"]) for row in sub]
                row = {
                    "architecture": arch,
                    "rank": int(rank),
                    "lambda_ctrl": float(lam),
                    "completed_rows": len(sub),
                    "control_contrastive_margin_mean": statistics.fmean(margins) if margins else None,
                    "control_contrastive_margin_p10": percentile(margins, 0.10),
                    "margin_positive_rows": sum(1 for row in sub if float(row["control_contrastive_margin"]) > 0.0),
                    "beats_same_energy_random_preflight_rows": sum(int(row["beats_same_energy_random_preflight"]) for row in sub),
                    "beats_same_control_margin_random_preflight_rows": sum(int(row["beats_same_control_margin_random_preflight"]) for row in sub),
                    "readout_visible_energy_CVaR25_ge_025_rows": sum(1 for row in sub if float(row["readout_visible_energy_CVaR25"]) >= 0.25),
                    "basis_Gram_condition_pass_rows": sum(1 for row in sub if float(row["basis_Gram_condition"]) <= 1.0e6),
                }
                row["probe_gate_pass"] = int(
                    row["completed_rows"] == 15
                    and row["control_contrastive_margin_p10"] is not None
                    and float(row["control_contrastive_margin_p10"]) > 0.0
                    and row["beats_same_energy_random_preflight_rows"] >= 10
                    and row["beats_same_control_margin_random_preflight_rows"] >= 10
                    and row["readout_visible_energy_CVaR25_ge_025_rows"] >= 10
                    and row["basis_Gram_condition_pass_rows"] >= 12
                )
                summary_rows.append(row)
    best = max(
        summary_rows,
        key=lambda row: (
            int(row["probe_gate_pass"]),
            float(row["control_contrastive_margin_p10"] if row["control_contrastive_margin_p10"] is not None else -999.0),
            int(row["margin_positive_rows"]),
        ),
    )
    detail_path = OUT_ROOT / "v22_68_part_e_generalized_eig_margin_probe_detail.csv"
    summary_path = OUT_ROOT / "v22_68_part_e_generalized_eig_margin_probe_summary.csv"
    json_path = OUT_ROOT / "v22_68_part_e_generalized_eig_margin_probe.json"
    write_rows(detail_path, detail)
    write_rows(summary_path, summary_rows)
    route = {
        "gate": "v22_68_part_e_generalized_eig_margin_probe",
        "detail_csv": str(detail_path.relative_to(ROOT)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "best_probe": best,
        "any_probe_gate_pass": int(any(int(row["probe_gate_pass"]) for row in summary_rows)),
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
