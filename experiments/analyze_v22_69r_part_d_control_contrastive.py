#!/usr/bin/env python3
"""Control-contrastive Part D repair probe for v22.69R.

Diagnostic only.  It uses the train-only MLP target from Part B to select a
readout-visible KAN column subset with a fixed lambda grid, then evaluates the
lift against matched random controls.  It does not create an official runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit as r69  # noqa: E402


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def current_command() -> str:
    return r69.command_text([r69.PYTHON, r69.rel(Path(__file__)), *sys.argv[1:]])


def select_control_contrastive(phi: Any, target: Any, lam: float, seed: int, keep_fraction: float) -> dict[str, Any]:
    import torch

    col_count = int(phi.shape[1])
    if col_count == 0:
        return {"selected_phi": phi[:, :0], "keep_indices": [], "score_p10": 0.0}
    target = target.float().reshape(-1)
    col_norm = phi.float().norm(dim=0).clamp_min(1.0e-12)
    target_norm = target.norm().clamp_min(1.0e-12)
    align = torch.abs(phi.float().transpose(0, 1) @ target) / (col_norm * target_norm)
    energy = phi.float().square().sum(dim=0)
    readout = energy / energy.max().clamp_min(1.0e-12)
    gen = torch.Generator(device=phi.device)
    gen.manual_seed(int(seed) + int(float(lam) * 1000) + 8801)
    ctrl_best = torch.zeros_like(align)
    for _ in range(16):
        ctrl = torch.randn_like(target, generator=gen)
        ctrl = ctrl / ctrl.norm().clamp_min(1.0e-12)
        ctrl_align = torch.abs(phi.float().transpose(0, 1) @ ctrl) / col_norm
        ctrl_best = torch.maximum(ctrl_best, ctrl_align)
    score = readout * align.square() - float(lam) * readout * ctrl_best.square()
    keep = max(2, min(col_count, int(math.ceil(float(keep_fraction) * col_count))))
    idx = torch.argsort(score, descending=True)[:keep]
    idx = torch.sort(idx).values
    vals = sorted(float(x) for x in score.detach().cpu().tolist())
    return {
        "selected_phi": phi.index_select(1, idx),
        "keep_indices": [int(x) for x in idx.detach().cpu().tolist()],
        "score_p10": vals[max(0, min(len(vals) - 1, int(math.floor(0.10 * (len(vals) - 1)))))],
        "score_mean": sum(vals) / len(vals),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66
    import experiments.run_v22_67_kanaware_metric_compatible_generator_carrier_fu as v67
    import experiments.run_v22_68_control_contrastive_kan_gauge_carrier_mpfu as v68

    r69.ensure_out()
    base_args = r69.build_parser().parse_args([])
    for key, value in vars(args).items():
        setattr(base_args, key, value)
    device = r69.torch_device(str(args.device))
    lambdas = [float(x) for x in parse_csv(args.lambdas)]
    rows_b = [row for row in r69.read_rows(r69.OUT_ROOT / "v22_69R_part_b_mlp_target_extraction.csv") if row.get("run_status") == "completed" and row.get("target_artifact")]
    arches = parse_csv(args.part_d_arches)
    suffix = f"_{r69.safe_fragment(str(args.artifact_tag))}" if str(args.artifact_tag or "").strip() else ""
    out_csv = r69.OUT_ROOT / f"v22_69R_part_d_control_contrastive_repair{suffix}.csv"
    out_json = r69.OUT_ROOT / f"v22_69R_part_d_control_contrastive_repair{suffix}_summary.json"
    r69.append_exec(
        "D_control_contrastive_repair_start",
        current_command(),
        "start",
        gpu=str(args.device),
        files=f"{r69.rel(out_csv)}; {r69.rel(out_json)}",
        note=f"rows_b={len(rows_b)}; arches={arches}; lambdas={lambdas}; artifact_tag={args.artifact_tag}; diagnostic_only=1",
    )
    rows: list[dict[str, Any]] = []
    for rb in rows_b:
        dataset = str(rb["dataset"])
        seed = int(float(rb["seed"]))
        method = str(rb["method"])
        ckpt = int(float(rb["checkpoint_step"]))
        bundle = v66.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
        splits = r69.split_train_only(bundle, device, int(args.probe_batch_size), seed)
        x_probe = splits["x_probe"]
        y_probe = splits["y_probe"]
        pack = torch.load(ROOT / str(rb["target_artifact"]), map_location=device)
        target = pack["target_fd"].to(device).float().reshape(-1)
        target = target / target.norm().clamp_min(1.0e-12)
        for arch in arches:
            sub_arches = [part.strip() for part in str(arch).split("+") if part.strip()]
            if len(sub_arches) > 1:
                bases = []
                phis = []
                sub_diags = []
                for sub_arch in sub_arches:
                    sub_base = v68.make_base_for_arch(sub_arch, bundle, device, int(args.hidden), seed, base_args)
                    sub_phi, sub_diag = v68.design_matrix(sub_base, x_probe, y_probe, bundle, base_args)
                    bases.append(sub_base)
                    phis.append(sub_phi)
                    sub_diags.append(sub_diag)
                base = bases[0]
                phi = torch.cat(phis, dim=1)
                diag_raw = {"basis_Gram_condition": ";".join(str(d.get("basis_Gram_condition", "")) for d in sub_diags)}
            else:
                base = v68.make_base_for_arch(arch, bundle, device, int(args.hidden), seed, base_args)
                phi, diag_raw = v68.design_matrix(base, x_probe, y_probe, bundle, base_args)
            for lam in lambdas:
                selected = select_control_contrastive(phi, target, lam, seed * 1009 + ckpt + sum(ord(c) for c in arch + method), float(args.keep_fraction))
                sel_phi = selected["selected_phi"]
                cap = v67.part_c_projection_capacity(sel_phi, target)
                proj = cap.get("projection")
                if proj is None:
                    proj = torch.zeros_like(target)
                visibility = v67.part_c_column_visibility(sel_phi)
                rand_phi = r69.same_energy_random_phi(sel_phi, seed * 1709 + ckpt + int(lam * 100))
                rand_cap = v67.part_c_projection_capacity(rand_phi, target)
                spectrum = float(cap.get("coef_norm", 0.0) or 0.0) / max(1.0, math.sqrt(max(1, int(sel_phi.shape[1]))))
                debt = r69.target_norms_and_debt(base(x_probe).detach().float(), proj.reshape(-1).to(device).float(), y_probe)["mlp_target_debt_predicted_delta"]
                score = r69.projection_score(cap, 1.0, debt, spectrum)
                rand_score = r69.projection_score(rand_cap, 1.0, 0.0, float(rand_cap.get("coef_norm", 0.0) or 0.0))
                rows.append(
                    {
                        "run_status": "completed",
                        "diagnostic_only": 1,
                        "official_runtime_eligible": 0,
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "checkpoint_step": ckpt,
                        "architecture": arch,
                        "mixed_bank_diagnostic": int(len(sub_arches) > 1),
                        "mixed_bank_sub_arches": "+".join(sub_arches) if len(sub_arches) > 1 else "",
                        "lambda_ctrl": lam,
                        "keep_fraction": float(args.keep_fraction),
                        "raw_projection_dim": int(phi.shape[1]),
                        "selected_projection_dim": int(sel_phi.shape[1]),
                        "selected_indices": json.dumps(selected["keep_indices"]),
                        "selection_score_p10": selected["score_p10"],
                        "KAN_projection_energy_to_MLP_winner": cap["capacity"],
                        "KAN_target_alignment_cosine": r69.cosine(proj.reshape(-1).to(device).float(), target),
                        "KAN_lift_reconstruction_error": float((torch.linalg.norm(proj.reshape(-1).to(device).float() - target) / target.norm().clamp_min(1.0e-12)).detach().cpu().item()),
                        "readout_visible_energy_CVaR25": visibility["cvar25"],
                        "KAN_lift_predicted_debt_delta": debt,
                        "KAN_lift_predicted_spectrum_drift": spectrum,
                        "KAN_score": score,
                        "same_energy_random_score": rand_score,
                        "KAN_control_margin": score - rand_score,
                        "beats_same_energy_random_lift": int(score > rand_score),
                        "basis_Gram_condition_raw": diag_raw.get("basis_Gram_condition", ""),
                    }
                )
        if len(rows) % 50 == 0:
            r69.write_rows(out_csv, rows)
    r69.write_rows(out_csv, rows)
    summary_rows = []
    for lam in lambdas:
        sub = [row for row in rows if float(row["lambda_ctrl"]) == float(lam)]
        n = max(1, len(sub))
        margins = [float(row["KAN_control_margin"]) for row in sub]
        summary_rows.append(
            {
                "lambda_ctrl": lam,
                "completed_rows": len(sub),
                "projection_energy_ge_025_rows": sum(float(row["KAN_projection_energy_to_MLP_winner"]) >= 0.25 for row in sub),
                "alignment_ge_035_rows": sum(float(row["KAN_target_alignment_cosine"]) >= 0.35 for row in sub),
                "reconstruction_le_075_rows": sum(float(row["KAN_lift_reconstruction_error"]) <= 0.75 for row in sub),
                "readout_visible_CVaR25_ge_025_rows": sum(float(row["readout_visible_energy_CVaR25"]) >= 0.25 for row in sub),
                "control_margin_positive_rows": sum(float(row["KAN_control_margin"]) > 0.0 for row in sub),
                "control_margin_p10": r69.percentile(margins, 0.10),
                "control_margin_CVaR25": r69.mean(sorted(margins)[: max(1, int(math.ceil(0.25 * len(margins))))]) if margins else 0.0,
                "beats_same_energy_random_lift_rows": sum(int(row["beats_same_energy_random_lift"]) for row in sub),
                "predicted_no_debt_rows": sum(float(row["KAN_lift_predicted_debt_delta"]) <= 0.0 for row in sub),
                "predicted_spectrum_drift_pass_rows": sum(float(row["KAN_lift_predicted_spectrum_drift"]) <= float(args.functional_spectrum_drift_threshold) for row in sub),
                "control_margin_positive_fraction": sum(float(row["KAN_control_margin"]) > 0.0 for row in sub) / n,
            }
        )
    best = max(summary_rows, key=lambda row: (float(row["control_margin_p10"]), int(row["projection_energy_ge_025_rows"])), default={})
    summary = {
        "gate": "v22_69R_part_d_control_contrastive_repair",
        "generated_at_sg": r69.now_sg(),
        "diagnostic_only": 1,
        "official_runtime_eligible": 0,
        "completed_rows": len(rows),
        "summary_rows": summary_rows,
        "best_lambda_by_margin_p10": best,
        "route": "R3-KANCarrierCapacityNotEstablished_NoFullLoop",
        "conclusion": "control-contrastive repair is diagnostic; official full-loop remains forbidden unless Part D gate is passed by a fixed audited candidate.",
    }
    r69.write_json(out_json, summary)
    r69.append_exec(
        "D_control_contrastive_repair",
        current_command(),
        "done",
        gpu=str(args.device),
        files=f"{r69.rel(out_csv)}; {r69.rel(out_json)}",
        note=json.dumps(summary, ensure_ascii=False),
    )
    r69.append_recap(
        "Part D control-contrastive repair probe",
        [
            f"diagnostic_only=1; completed_rows={len(rows)}; lambdas={lambdas}.",
            f"best_lambda_by_margin_p10={json.dumps(best, ensure_ascii=False)}.",
            "This implements the planned control-margin fallback: readout/target/control-contrastive column scoring with fixed lambda grid 2,4,8,16; no official full-loop run.",
            f"Evidence: `{r69.rel(out_csv)}`, `{r69.rel(out_json)}`.",
        ],
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lambdas", default="2,4,8,16")
    p.add_argument("--keep-fraction", type=float, default=0.25)
    p.add_argument("--part-d-arches", default="DGKAN_DCHE,DGKAN_DFOU")
    p.add_argument("--artifact-tag", default="")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--held-size", type=int, default=64)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--probe-batch-size", type=int, default=32)
    p.add_argument("--metric-batch-size", type=int, default=96)
    p.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    args = p.parse_args()
    out = run(args)
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
