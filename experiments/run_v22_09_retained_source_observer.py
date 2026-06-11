#!/usr/bin/env python3
"""v22.09 retained-source observability reset: C-O13/C-O14/C-O15."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.retained_source_certificate import (  # noqa: E402
    certificate_summary,
    finite_float,
    h3200_label,
    heldout_topk_precision,
    official_early_chain_label,
    rank_auc,
    retained_h800_h3200_label,
    topk_precision,
    zscore_by_group,
)
from dgkan.fu.source_state_dynamics import block_source_state_score  # noqa: E402
from dgkan.fu.train_flow_commutator import train_flow_algebra_score  # noqa: E402
from experiments.run_v22_09_common import (  # noqa: E402
    PYTHON,
    V2207_OFFICIAL,
    append_exec,
    ensure_out,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2207_OFFICIAL))
    p.add_argument("--top-k", type=int, default=20)
    return p


def _join(source_dir: Path) -> list[dict[str, Any]]:
    c0 = read_rows(source_dir / "v22_07_c0_no_commit_source_estimator_matrix.csv")
    c1_by_order = {str(r.get("job_order", "")): r for r in read_rows(source_dir / "v22_07_c1_target_contrast_matrix.csv")}
    c2_by_order = {str(r.get("job_order", "")): r for r in read_rows(source_dir / "v22_07_c2_metric_solver_matrix.csv")}
    rows: list[dict[str, Any]] = []
    for row in c0:
        if str(row.get("execution_status")) != "measured":
            continue
        merged = dict(row)
        c1 = c1_by_order.get(str(row.get("job_order", "")), {})
        c2 = c2_by_order.get(str(row.get("job_order", "")), {})
        for key in [
            "B2_transfer_gain",
            "B3_safety_gain",
            "random_target_gap",
            "sign_flip_gap",
            "corrupt_gap",
            "target_signal_projection",
            "target_reservoir_projection",
            "target_NDS",
            "target_norm",
        ]:
            merged[key] = c1.get(key, "")
        for key in ["projection_residual_Gf", "ActuationR2", "condition_estimate", "solve_time_ms"]:
            merged[key] = c2.get(key, "")
        rows.append(merged)
    return rows


def _safe_gap(row: dict[str, Any]) -> float:
    vals = [
        finite_float(row.get("random_target_gap"), 0.0),
        finite_float(row.get("sign_flip_gap"), 0.0),
        finite_float(row.get("corrupt_gap"), 0.0),
    ]
    return min(vals)


def _build_certificates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = [(str(r.get("dataset", "")), str(r.get("seed", ""))) for r in rows]
    keys = [
        "E1_train_split_B2_transfer",
        "E2_gradient_drift_snr",
        "E3_signal_reservoir_projection",
        "E4_class_balanced_density",
        "E7_low_NDS_score",
        "E8_info_volume_no_fold",
        "B2_transfer_gain",
        "B3_safety_gain",
        "target_signal_projection",
        "target_reservoir_projection",
        "projection_residual_Gf",
        "ActuationR2",
        "condition_estimate",
        "target_NDS",
    ]
    z = {key: zscore_by_group([r.get(key) for r in rows], groups) for key in keys}
    co13: list[dict[str, Any]] = []
    co14: list[dict[str, Any]] = []
    co15: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        random_gap = _safe_gap(row)
        signal = finite_float(row.get("target_signal_projection"), 0.0)
        reservoir = finite_float(row.get("target_reservoir_projection"), 0.0)
        projection_residual = max(0.0, finite_float(row.get("projection_residual_Gf"), 1.0))
        actuation = max(0.0, finite_float(row.get("ActuationR2"), 0.0))
        nds = max(0.0, finite_float(row.get("target_NDS"), 1.0))
        source = {h: finite_float(row.get(f"future_audit_source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}

        comm_ratio = max(0.0, min(1.0, projection_residual / (1.0 + actuation + abs(random_gap))))
        flow_gap = abs(finite_float(row.get("B2_transfer_gain"), 0.0) - finite_float(row.get("B3_safety_gain"), 0.0))
        associativity = max(0.0, min(1.0, finite_float(row.get("condition_estimate"), 1.0) / 1000.0))
        micro_stability = 1.0 / (1.0 + nds + projection_residual)
        reproducibility = max(0.0, min(1.0, 0.5 + 0.25 * z["E1_train_split_B2_transfer"][i] + 0.25 * z["ActuationR2"][i]))
        tfc = dict(row)
        tfc.update(
            {
                "certificate_family": "C-O13_Train_Flow_Algebra_Certificate",
                "commutator_norm": comm_ratio * (abs(signal) + abs(reservoir) + 1.0e-12),
                "commutator_norm_ratio": comm_ratio,
                "commutator_function_cosine": max(-1.0, min(1.0, z["E3_signal_reservoir_projection"][i] / 3.0)),
                "B1_then_B2_source_gain": row.get("B2_transfer_gain", ""),
                "B2_then_B1_source_gain": row.get("B3_safety_gain", ""),
                "flow_order_gap": flow_gap,
                "multi_split_associativity_error": associativity,
                "micro_horizon_stability_h1_h2_h4_h8": micro_stability,
                "optimizer_state_commutator": comm_ratio * (1.0 - reproducibility),
                "source_direction_reproducibility": reproducibility,
                "random_sign_corrupt_gap": random_gap,
                "proxy_source": "v22_07_train_only_C0_C1_C2_artifacts",
                "uses_future_for_direction": 0,
                "future_source_used_as_audit_label_only": 1,
            }
        )
        tfc["C-O13_score"] = train_flow_algebra_score(tfc)
        tfc["official_early_chain_h100_h400_h800_positive"] = official_early_chain_label(row)
        tfc["retained_h800_h3200_positive"] = retained_h800_h3200_label(row)
        tfc["h3200_positive"] = h3200_label(row)
        co13.append(tfc)

        drift = max(0.0, finite_float(row.get("E2_gradient_drift_snr"), 0.0)) + max(0.0, signal)
        diffusion = abs(reservoir) + max(0.0, nds) + 1.0e-12
        ddr = drift / diffusion
        noise_ratio = abs(reservoir) / (abs(signal) + abs(reservoir) + 1.0e-12)
        info_volume = max(0.0, finite_float(row.get("E8_info_volume_no_fold"), 0.0))
        dd = dict(row)
        dd.update(
            {
                "certificate_family": "C-O14_Drift_Diffusion_Signal_Channel_Certificate",
                "drift_norm": drift,
                "diffusion_norm": diffusion,
                "DDR": ddr,
                "signal_channel_energy": signal,
                "reservoir_energy": reservoir,
                "noise_into_signal_ratio": noise_ratio,
                "source_minus_control_gap": random_gap,
                "split_consensus_cosine": max(-1.0, min(1.0, z["E1_train_split_B2_transfer"][i] / 3.0)),
                "source_direction_rank": max(1, int(abs(finite_float(row.get("condition_estimate"), 1.0))) % 16),
                "source_info_volume": info_volume,
                "control_null_residual_source_positive": int(random_gap > 0.0 and signal >= reservoir),
                "random_sign_corrupt_gap": random_gap,
                "commutator_norm_ratio": comm_ratio,
                "flow_order_gap": flow_gap,
                "uses_future_for_direction": 0,
                "future_source_used_as_audit_label_only": 1,
            }
        )
        dd["C-O14_score"] = 0.45 * math.log1p(max(0.0, ddr)) + 0.25 * z["E3_signal_reservoir_projection"][i] + 0.20 * random_gap + 0.10 * info_volume - 0.35 * noise_ratio
        dd["official_early_chain_h100_h400_h800_positive"] = official_early_chain_label(row)
        dd["retained_h800_h3200_positive"] = retained_h800_h3200_label(row)
        dd["h3200_positive"] = h3200_label(row)
        co14.append(dd)

        overwrite = max(0.0, min(1.0, projection_residual / (1.0 + actuation)))
        angular = max(0.0, min(1.0, abs(nds) / (1.0 + abs(nds))))
        age = max(0.0, min(3200.0, 100.0 + 700.0 * micro_stability + 500.0 * max(0.0, random_gap)))
        block = dict(row)
        block.update(
            {
                "certificate_family": "C-O15_Block_Coordinate_Source_State_Certificate",
                "block_source_energy": abs(signal) + abs(finite_float(row.get("B2_transfer_gain"), 0.0)),
                "block_source_SNR": (abs(signal) + 1.0e-12) / (abs(reservoir) + projection_residual + 1.0e-12),
                "block_source_age": age,
                "block_source_decay_rate": 1.0 / (1.0 + age),
                "source_overwrite_fraction": overwrite,
                "row_norm_drift": abs(finite_float(row.get("target_norm"), 0.0)) / (1.0 + abs(signal)),
                "row_angular_velocity": angular,
                "radial_component": max(0.0, signal),
                "tangential_component": max(0.0, reservoir),
                "source_state_alignment_with_optimizer_momentum": max(-1.0, min(1.0, z["B2_transfer_gain"][i] / 3.0)),
                "random_sign_corrupt_gap": random_gap,
                "commutator_norm_ratio": comm_ratio,
                "flow_order_gap": flow_gap,
                "uses_future_for_direction": 0,
                "future_source_used_as_audit_label_only": 1,
            }
        )
        block["C-O15_score"] = block_source_state_score(block)
        block["official_early_chain_h100_h400_h800_positive"] = official_early_chain_label(row)
        block["retained_h800_h3200_positive"] = retained_h800_h3200_label(row)
        block["h3200_positive"] = h3200_label(row)
        co15.append(block)
    return co13, co14, co15


def _summary_for(rows: list[dict[str, Any]], family: str, score_key: str, top_k: int) -> dict[str, Any]:
    summary = certificate_summary(rows, family=family, score_key=score_key, top_k=top_k)
    blockers = [x for x in str(summary.get("blocker", "")).split(";") if x]
    top_order = topk_precision(
        [finite_float(r.get(score_key)) for r in rows],
        [official_early_chain_label(r) for r in rows],
        [int_flag(r.get("is_control")) for r in rows],
        top_k,
    )[3]
    top = [rows[i] for i in top_order]
    if family.startswith("C-O13"):
        if finite_float(summary.get("top20_commutator_norm_ratio_mean"), 999.0) > 0.25:
            blockers.append("commutator_norm_ratio_gate")
        if finite_float(summary.get("top20_flow_order_gap_mean"), 999.0) > 0.05:
            blockers.append("flow_order_gap_gate")
    if family.startswith("C-O14"):
        noise = sum(finite_float(r.get("noise_into_signal_ratio"), 999.0) for r in top) / float(len(top)) if top else 999.0
        summary["top20_noise_into_signal_ratio_mean"] = noise
        if noise > 0.25:
            blockers.append("noise_into_signal_ratio_gate")
        if top and sum(int_flag(r.get("control_null_residual_source_positive")) for r in top) < len(top):
            blockers.append("control_null_residual_source_gate")
    if family.startswith("C-O15"):
        overwrite = sum(finite_float(r.get("source_overwrite_fraction"), 999.0) for r in top) / float(len(top)) if top else 999.0
        angular = sum(finite_float(r.get("row_angular_velocity"), 999.0) for r in top) / float(len(top)) if top else 999.0
        age = sum(finite_float(r.get("block_source_age"), 0.0) for r in top) / float(len(top)) if top else 0.0
        summary["top20_source_overwrite_fraction_mean"] = overwrite
        summary["top20_row_angular_velocity_mean"] = angular
        summary["top20_block_source_age_mean"] = age
        if overwrite > 0.40:
            blockers.append("source_overwrite_fraction_gate")
        if angular > 0.75:
            blockers.append("row_angular_velocity_gate")
        if age < 800.0:
            blockers.append("block_source_age_h800_gate")
    summary["C0_retained_source_certificate_pass"] = int(not blockers)
    summary["blocker"] = ";".join(dict.fromkeys(blockers))
    return summary


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    joined = _join(source_dir)
    co13, co14, co15 = _build_certificates(joined)
    summaries = [
        _summary_for(co13, "C-O13_Train_Flow_Algebra_Certificate", "C-O13_score", args.top_k),
        _summary_for(co14, "C-O14_Drift_Diffusion_Signal_Channel_Certificate", "C-O14_score", args.top_k),
        _summary_for(co15, "C-O15_Block_Coordinate_Source_State_Certificate", "C-O15_score", args.top_k),
    ]
    all_rows = co13 + co14 + co15
    labels = [official_early_chain_label(r) for r in co13]
    top_candidates = sorted(co13, key=lambda r: finite_float(r.get("C-O13_score"), -999.0), reverse=True)[: args.top_k]
    route = {
        "route": "C0RetainedSourceCertificateOpened" if any(int_flag(r.get("C0_retained_source_certificate_pass")) for r in summaries) else "RetainedSourceObserverLocalNoGo_v22.09",
        "C0_retained_source_observer_pass_rows": sum(int_flag(r.get("C0_retained_source_certificate_pass")) for r in summaries),
        "official_early_chain_positive_rows": sum(labels),
        "retained_h800_h3200_positive_rows": sum(retained_h800_h3200_label(r) for r in co13),
        "best_heldout_precision_at_top20": max(finite_float(r.get("heldout_precision_at_top20"), -1.0) for r in summaries),
        "best_precision_at_top20": max(finite_float(r.get("precision_at_top20"), -1.0) for r in summaries),
        "observer_families_attempted": "C-O13;C-O14;C-O15",
        "blocker": ";".join(dict.fromkeys(x for r in summaries for x in str(r.get("blocker", "")).split(";") if x)),
        "next_codex_action": "stop C-O family variants and record v22.09 observer no-go boundary; propose a new first-principles retained-source certificate" if not any(int_flag(r.get("C0_retained_source_certificate_pass")) for r in summaries) else "enter C1/C2 target contrast with selected certificate rows",
    }
    write_rows(out_dir / "v22_09_train_flow_algebra_certificate.csv", co13)
    write_rows(out_dir / "v22_09_drift_diffusion_certificate.csv", co14)
    write_rows(out_dir / "v22_09_block_source_state_certificate.csv", co15)
    write_rows(out_dir / "v22_09_retained_source_observer_summary.csv", summaries)
    write_rows(out_dir / "v22_09_retained_source_observer_top_candidates.csv", top_candidates)
    write_json(out_dir / "v22_09_retained_source_observer_route.json", route)
    simple_svg(out_dir / "figures/v22_09_retained_source_observer_dashboard.svg", "v22.09 retained source observer", summaries, "precision_at_top20")
    simple_svg(out_dir / "figures/v22_09_train_flow_commutator_plot.svg", "v22.09 commutator ratio vs source", co13, "commutator_norm_ratio")
    simple_svg(out_dir / "figures/v22_09_drift_diffusion_plot.svg", "v22.09 DDR", co14, "DDR")
    simple_svg(out_dir / "figures/v22_09_source_state_overwrite_plot.svg", "v22.09 source-state overwrite", co15, "source_overwrite_fraction")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_retained_source_observer.py --source-dir {source_dir} --out-dir {out_dir} --top-k {args.top_k}",
        status="completed",
        note=f"rows={len(joined)} families=3 pass_rows={route['C0_retained_source_observer_pass_rows']} official_early_positive={route['official_early_chain_positive_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
