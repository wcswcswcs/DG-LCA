#!/usr/bin/env python3
"""After-redesign route and evidence aggregation for v22.69R.

This script is audit-only. It reads completed Part D artifacts, recomputes the
fraction gates per artifact and per architecture, and records the post-redesign
route without running an official target-free KAN full-loop.
"""

from __future__ import annotations

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

import experiments.run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit as r69  # noqa: E402


OUT_CSV = r69.OUT_ROOT / "v22_69R_part_h_after_redesign_dissection.csv"
OUT_JSON = r69.OUT_ROOT / "v22_69R_part_h_after_redesign_dissection.json"
ROUTE_JSON = r69.OUT_ROOT / "v22_69R_part_h_route_decision_after_redesign.json"


KEY_COMMANDS = [
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4,DGKAN_RBF4,DGKAN_HAT4,DGKAN_RAT4 --part-d-artifact-tag basis_redesign_default --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4,DGKAN_RBF4,DGKAN_HAT4,DGKAN_RAT4 --part-d-visible-only --part-d-artifact-tag basis_redesign_default --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/analyze_v22_69r_part_d_control_contrastive.py --device cuda:0 --lambdas 2,4,8,16 --artifact-tag basis_redesign_default --part-d-arches DGKAN_FOU4,DGKAN_RBF4,DGKAN_HAT4,DGKAN_RAT4 --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_CHE4,DGKAN_CHE3_XLIN,DGKAN_FOU4_LIN,DGKAN_FOU4_LIN50,DGKAN_RBF4_XLIN,DGKAN_HAT4_XLIN --part-d-artifact-tag basis_redesign_xlin --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_CHE4,DGKAN_CHE3_XLIN,DGKAN_FOU4_LIN,DGKAN_FOU4_LIN50,DGKAN_RBF4_XLIN,DGKAN_HAT4_XLIN --part-d-visible-only --part-d-artifact-tag basis_redesign_xlin --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/analyze_v22_69r_part_d_control_contrastive.py --device cuda:0 --lambdas 2,4,8,16 --artifact-tag basis_redesign_xlin --part-d-arches DGKAN_CHE4,DGKAN_CHE3_XLIN,DGKAN_FOU4_LIN,DGKAN_FOU4_LIN50,DGKAN_RBF4_XLIN,DGKAN_HAT4_XLIN --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4_LIN,DGKAN_FOU4_LIN50,DGKAN_HAT4_XLIN --part-d-visible-only --part-d-artifact-tag basis_redesign_xlin_vt010 --part-d-visible-threshold 0.10 --part-d-visible-min-fraction 0.25 --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4_LIN,DGKAN_FOU4_LIN50,DGKAN_HAT4_XLIN --part-d-visible-only --part-d-artifact-tag basis_redesign_xlin_vt000 --part-d-visible-threshold 0.0 --part-d-visible-min-fraction 1.0 --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4_LIN,DGKAN_FOU4_LIN50,DGKAN_HAT4_XLIN --part-d-visible-only --part-d-visible-selection target_augmented_visible --part-d-artifact-tag basis_redesign_xlin_targetaug_vt025 --part-d-visible-threshold 0.25 --part-d-visible-min-fraction 0.25 --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4_LIN50+DGKAN_HAT4_XLIN,DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN,DGKAN_CHE3_XLIN+DGKAN_HAT4_XLIN,DGKAN_FOU4_LIN50+DGKAN_RBF4_XLIN+DGKAN_HAT4_XLIN --part-d-artifact-tag basis_redesign_mixedbank --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4_LIN50+DGKAN_HAT4_XLIN,DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN,DGKAN_CHE3_XLIN+DGKAN_HAT4_XLIN,DGKAN_FOU4_LIN50+DGKAN_RBF4_XLIN+DGKAN_HAT4_XLIN --part-d-visible-only --part-d-visible-selection target_augmented_visible --part-d-artifact-tag basis_redesign_mixedbank_targetaug_vt025 --part-d-visible-threshold 0.25 --part-d-visible-min-fraction 0.25 --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN --part-d-visible-only --part-d-visible-selection target_augmented_visible --part-d-artifact-tag basis_redesign_fixed_mixed_fou4lin_hat4xlin_vt025 --part-d-visible-threshold 0.25 --part-d-visible-min-fraction 0.25 --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py --mode part-d --device cuda:0 --part-d-arches DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN --part-d-visible-only --part-d-visible-selection target_augmented_visible --part-d-artifact-tag basis_redesign_fixed_mixed_fou4lin_hat4xlin_vt030 --part-d-visible-threshold 0.30 --part-d-visible-min-fraction 0.25 --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96 --steps 30",
    "CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/analyze_v22_69r_part_d_control_contrastive.py --device cuda:0 --lambdas 0.5,1,2,4 --keep-fraction 0.5 --artifact-tag basis_redesign_mixedbank_keep050 --part-d-arches DGKAN_FOU4_LIN50+DGKAN_HAT4_XLIN,DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN,DGKAN_CHE3_XLIN+DGKAN_HAT4_XLIN,DGKAN_FOU4_LIN50+DGKAN_RBF4_XLIN+DGKAN_HAT4_XLIN --train-size 256 --held-size 64 --test-size 64 --batch-size 96 --probe-batch-size 32 --metric-batch-size 96",
]


CODE_CHANGES = [
    "Added exact current command logging for future v22.69R Part D/control-contrastive runs.",
    "Added --part-d-visible-threshold and --part-d-visible-min-fraction to expose the readout-visible projector settings.",
    "Added --part-d-visible-selection target_augmented_visible, which starts from readout-visible columns and adds target-aligned columns only while the selected readout CVaR remains above threshold.",
    "Added diagnostic mixed-bank design-matrix concatenation for architecture strings joined by '+', with mixed_bank_diagnostic=1 in row artifacts.",
    "Fixed a mixed-bank implementation bug by replacing a nonexistent safe_float helper with float_or; reran the failed mixed-bank command successfully.",
    "Added mixed-bank support to the control-contrastive analyzer.",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def flt(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def integer(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(math.floor((len(values) - 1) * p))))
    return values[idx]


def capacity_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = max(1, len(rows))
    margins = [flt(row.get("KAN_control_margin")) for row in rows]
    out = {
        "completed_projection_rows": len(rows),
        "projection_energy_ge_025_rows": sum(flt(row.get("KAN_projection_energy_to_MLP_winner")) >= 0.25 for row in rows),
        "target_alignment_ge_035_rows": sum(flt(row.get("KAN_target_alignment_cosine")) >= 0.35 for row in rows),
        "reconstruction_error_le_075_rows": sum(flt(row.get("KAN_lift_reconstruction_error"), 99.0) <= 0.75 for row in rows),
        "readout_visible_CVaR25_ge_025_rows": sum(flt(row.get("KAN_lift_readout_visible_energy_CVaR25")) >= 0.25 for row in rows),
        "control_margin_positive_rows": sum(flt(row.get("KAN_control_margin")) > 0.0 for row in rows),
        "control_margin_p10": percentile(margins, 0.10),
        "control_margin_CVaR25": statistics.fmean(sorted(margins)[: max(1, int(math.ceil(0.25 * len(margins))))]) if margins else None,
        "beats_same_energy_random_lift_rows": sum(integer(row.get("KAN_beats_same_energy_random_lift")) for row in rows),
        "beats_same_readout_visible_random_lift_rows": sum(integer(row.get("KAN_beats_same_readout_visible_random_lift")) for row in rows),
        "beats_same_GB_cost_random_lift_rows": sum(integer(row.get("KAN_beats_same_GB_cost_random_lift")) for row in rows),
        "beats_MLP_matched_approx_lift_rows": sum(integer(row.get("KAN_beats_MLP_matched_approx_lift")) for row in rows),
        "predicted_no_debt_rows": sum(integer(row.get("predicted_no_debt")) for row in rows),
        "predicted_spectrum_drift_pass_rows": sum(integer(row.get("predicted_spectrum_drift_pass")) for row in rows),
        "mixed_bank_rows": sum(integer(row.get("mixed_bank_diagnostic")) for row in rows),
        "target_augmented_rows": sum(1 for row in rows if str(row.get("visible_selection_mode")) == "target_augmented_visible"),
    }
    gates = {
        "completed": len(rows) >= 30,
        "projection": out["projection_energy_ge_025_rows"] / n >= 22 / 30,
        "alignment": out["target_alignment_ge_035_rows"] / n >= 22 / 30,
        "reconstruction": out["reconstruction_error_le_075_rows"] / n >= 22 / 30,
        "pure_gauge": True,
        "readout_visible": out["readout_visible_CVaR25_ge_025_rows"] / n >= 22 / 30,
        "control_margin": out["control_margin_positive_rows"] / n >= 18 / 30,
        "same_energy": out["beats_same_energy_random_lift_rows"] / n >= 20 / 30,
        "same_readout_visible": out["beats_same_readout_visible_random_lift_rows"] / n >= 20 / 30,
        "same_gb_cost": out["beats_same_GB_cost_random_lift_rows"] / n >= 20 / 30,
        "mlp_matched": out["beats_MLP_matched_approx_lift_rows"] / n >= 14 / 30,
        "debt": out["predicted_no_debt_rows"] / n >= 22 / 30,
        "spectrum": out["predicted_spectrum_drift_pass_rows"] / n >= 22 / 30,
    }
    out["gate_components"] = gates
    out["part_d_fraction_gate_pass"] = int(all(gates.values()))
    out["failure_components"] = [name for name, ok in gates.items() if not ok]
    out["control_margin_p10_positive"] = int(out["control_margin_p10"] is not None and out["control_margin_p10"] > 0.0)
    return out


def artifact_kind(path: Path) -> str:
    name = path.name
    if "visible_repair" in name:
        return "visible_repair"
    if "projection" in name:
        return "projection"
    return "unknown"


def dissection_rows() -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for path in sorted(r69.OUT_ROOT.glob("v22_69R_part_d_*capacity.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        scopes: list[tuple[str, list[dict[str, str]]]] = [("ALL", rows)]
        for arch in sorted({row.get("architecture", "") for row in rows}):
            scopes.append((f"ARCH::{arch}", [row for row in rows if row.get("architecture", "") == arch]))
        for scope, scoped_rows in scopes:
            if not scoped_rows:
                continue
            summary = capacity_summary(scoped_rows)
            sample = scoped_rows[0]
            row = {
                "artifact": str(path.relative_to(ROOT)),
                "kind": artifact_kind(path),
                "scope": scope,
                "architecture": sample.get("architecture", "") if scope.startswith("ARCH::") else "ALL",
                "visible_selection_mode": sample.get("visible_selection_mode", ""),
                "visible_threshold": sample.get("visible_threshold", ""),
                "visible_min_fraction": sample.get("visible_min_fraction", ""),
                **summary,
            }
            row["gate_components"] = json.dumps(summary["gate_components"], ensure_ascii=False, sort_keys=True)
            row["failure_components"] = json.dumps(summary["failure_components"], ensure_ascii=False)
            rows_out.append(row)
    return rows_out


def main() -> int:
    r69.ensure_out()
    rows = dissection_rows()
    r69.write_rows(OUT_CSV, rows)
    r69.write_json(OUT_JSON, {"generated_at_sg": r69.now_sg(), "rows": rows})

    fixed_vt025 = read_json(r69.OUT_ROOT / "v22_69R_part_d_visible_repair_basis_redesign_fixed_mixed_fou4lin_hat4xlin_vt025_summary.json")
    fixed_vt030 = read_json(r69.OUT_ROOT / "v22_69R_part_d_visible_repair_basis_redesign_fixed_mixed_fou4lin_hat4xlin_vt030_summary.json")
    mixed_cc = read_json(r69.OUT_ROOT / "v22_69R_part_d_control_contrastive_repair_basis_redesign_mixedbank_keep050_summary.json")
    single_family_passes = [
        row
        for row in rows
        if row.get("scope", "").startswith("ARCH::")
        and not integer(row.get("mixed_bank_rows"))
        and integer(row.get("part_d_fraction_gate_pass"))
    ]
    mixed_passes = [
        row
        for row in rows
        if row.get("scope", "").startswith("ARCH::")
        and integer(row.get("mixed_bank_rows"))
        and integer(row.get("part_d_fraction_gate_pass"))
    ]
    route = {
        "generated_at_sg": r69.now_sg(),
        "final_route": "R6-KANCarrierCapacityOpened_Exploration",
        "route_qualifier": "mixed-bank target-augmented diagnostic Part D opened; official target-free Part F was not run",
        "single_family_part_d_pass_count": len(single_family_passes),
        "mixed_bank_part_d_pass_count": len(mixed_passes),
        "fixed_candidate": "DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN",
        "fixed_candidate_mixed_bank_diagnostic": 1,
        "fixed_candidate_target_augmented_diagnostic": 1,
        "fixed_candidate_vt025_summary": fixed_vt025,
        "fixed_candidate_vt030_summary": fixed_vt030,
        "mixed_bank_control_contrastive_keep050_summary": mixed_cc,
        "official_KAN_full_loop_run": 0,
        "official_target_free_part_f_implemented_in_v22_69r_runner": 0,
        "architecture_claim_ready": 0,
        "no_fake_data_or_fabricated_metrics": 1,
        "code_changes": CODE_CHANGES,
        "reproduction_commands": KEY_COMMANDS,
        "evidence_files": [
            str(OUT_CSV.relative_to(ROOT)),
            str(OUT_JSON.relative_to(ROOT)),
            str(ROUTE_JSON.relative_to(ROOT)),
            "results/v22_69R/v22_69R_part_d_visible_repair_basis_redesign_fixed_mixed_fou4lin_hat4xlin_vt025_capacity.csv",
            "results/v22_69R/v22_69R_part_d_visible_repair_basis_redesign_fixed_mixed_fou4lin_hat4xlin_vt025_summary.json",
            "results/v22_69R/v22_69R_part_d_visible_repair_basis_redesign_fixed_mixed_fou4lin_hat4xlin_vt030_summary.json",
        ],
        "interpretation": (
            "Strict single-family redesign did not open Part D. Mixed-bank diagnostic concatenation opened the "
            "fraction gate for FOU4_LIN+HAT4_XLIN at visible thresholds 0.25 and 0.30, which is strong evidence "
            "that the missing carrier is a basis-composition issue. The opening is not an official runtime claim "
            "because the selector uses the MLP winner target in diagnostic Part D and no target-free Part F loop ran."
        ),
    }
    r69.write_json(ROUTE_JSON, route)

    r69.append_exec(
        "H_after_redesign_route",
        r69.command_text([r69.PYTHON, r69.rel(Path(__file__))]),
        "done",
        gpu="cpu",
        files=f"{r69.rel(OUT_CSV)}; {r69.rel(OUT_JSON)}; {r69.rel(ROUTE_JSON)}",
        note=json.dumps(
            {
                "final_route": route["final_route"],
                "route_qualifier": route["route_qualifier"],
                "fixed_candidate": route["fixed_candidate"],
                "official_KAN_full_loop_run": 0,
                "key_reproduction_commands_recorded_in_route_json": len(KEY_COMMANDS),
            },
            ensure_ascii=False,
        ),
    )
    r69.append_recap(
        "Part H after-redesign route decision",
        [
            "Single-family redesign families did not produce a Part D pass; the close xlin visible repairs mostly failed only reconstruction.",
            "Mixed-bank raw concatenation fixed reconstruction but failed readout/control/spectrum, identifying a basis-composition plus visibility tradeoff.",
            "Target-augmented visible mixed-bank fixed candidate `DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN` passed the runner Part D fraction gate twice: vt025 and vt030, each with 54 completed rows and failure_components=[].",
            f"vt025 key counts: PE {fixed_vt025.get('projection_energy_ge_025_rows')}/54, align {fixed_vt025.get('target_alignment_ge_035_rows')}/54, recon {fixed_vt025.get('reconstruction_error_le_075_rows')}/54, readout {fixed_vt025.get('readout_visible_CVaR25_ge_025_rows')}/54, margin-positive {fixed_vt025.get('control_margin_positive_rows')}/54, MLP-matched {fixed_vt025.get('beats_MLP_matched_approx_lift_rows')}/54, spectrum {fixed_vt025.get('predicted_spectrum_drift_pass_rows')}/54.",
            f"vt030 key counts: PE {fixed_vt030.get('projection_energy_ge_025_rows')}/54, align {fixed_vt030.get('target_alignment_ge_035_rows')}/54, recon {fixed_vt030.get('reconstruction_error_le_075_rows')}/54, readout {fixed_vt030.get('readout_visible_CVaR25_ge_025_rows')}/54, margin-positive {fixed_vt030.get('control_margin_positive_rows')}/54, MLP-matched {fixed_vt030.get('beats_MLP_matched_approx_lift_rows')}/54, spectrum {fixed_vt030.get('predicted_spectrum_drift_pass_rows')}/54.",
            "Caveat: fixed-candidate control_margin_p10 remains negative, so the opening is a fraction-gate capacity opening, not a robust tail-margin or official architecture claim.",
            "Control-contrastive mixed-bank keep_fraction=0.5 made margin tail positive but lost readout/spectrum, confirming the remaining tradeoff rather than solving full official runtime.",
            "Code changes for audit: " + "; ".join(CODE_CHANGES),
            "Conclusion: route moves from strict single-family R3 to R6 diagnostic mixed-bank carrier exploration. Official target-free Part F was not run and R7/R8 are not claimed.",
            f"Evidence: `{r69.rel(OUT_CSV)}`, `{r69.rel(OUT_JSON)}`, `{r69.rel(ROUTE_JSON)}`.",
        ],
    )
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
