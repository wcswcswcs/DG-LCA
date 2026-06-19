#!/usr/bin/env python3
"""Run or convert the v22.18 MLP+FU benefit mainline."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_18_common import (  # noqa: E402
    OUT_ROOT,
    PYTHON,
    V22_17_ROOT,
    WORKTREE_ROOT,
    append_exec,
    ensure_out,
    finite_float,
    read_rows,
    run_logged,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="MNIST,FMNIST,KMNIST")
    p.add_argument(
        "--models",
        default=(
            "DGMLP,DGMLP_FU,DGMLP_FU_GATED,DGMLP_FU_ACTIONABLE,DGMLP_FU_RELEASE,"
            "DGMLP_FU_SPLIT_VIRTUAL,DGMLP_FU_JVP_REFRESH,DGMLP_FU_CONTROL_RANDOM,"
            "DGMLP_FU_CONTROL_STABLE_RANDOM,DGMLP_FU_CONTROL_SIGNFLIP"
        ),
    )
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--physical-gpu", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clear-proxy-env", action="store_true")
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=440)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--log-interval", type=int, default=110)
    p.add_argument("--jvp-refresh-dim", type=int, default=8)
    p.add_argument("--jvp-refresh-min-history", type=int, default=8)
    p.add_argument("--jvp-refresh-history-window", type=int, default=32)
    p.add_argument("--jvp-refresh-scale", type=float, default=0.05)
    p.add_argument("--jvp-refresh-ratio-cap", type=float, default=0.05)
    p.add_argument("--actionable-gain-tolerance", type=float, default=0.002)
    p.add_argument("--virtual-loss-tolerance", type=float, default=1.0e-5)
    p.add_argument("--virtual-gate-scale", type=float, default=0.002)
    p.add_argument("--benefit-p3-ratio-cap", type=float, default=0.01)
    p.add_argument("--benefit-p3-strong-ratio-cap", type=float, default=0.03)
    p.add_argument("--benefit-p3-min-score", type=float, default=0.0)
    p.add_argument("--benefit-p3-strong-min-score", type=float, default=0.005)
    p.add_argument("--benefit-p3-jvp-strong-scale", type=float, default=0.20)
    p.add_argument("--benefit-p3-jvp-strong-ratio-cap", type=float, default=0.20)
    p.add_argument("--benefit-p3-jvp-interval", type=int, default=1)
    p.add_argument("--benefit-policy-export-json", default="")
    p.add_argument("--benefit-policy-export-threshold", default="")
    p.add_argument("--experiment-tag", default="v22_18_mlp_fu_benefit_d1_h440")
    p.add_argument("--output-suffix", default="v22_18_mlp_fu_benefit_d1_h440")
    p.add_argument("--reuse-v2217-suffix", default="")
    return p


def _out(name: str, suffix: str) -> Path:
    clean = suffix.strip().strip("_")
    path = OUT_ROOT / name
    if clean:
        return path.with_name(f"{path.stem}_{clean}{path.suffix}")
    return path


def _v17(name: str, suffix: str) -> Path:
    clean = suffix.strip().strip("_")
    path = V22_17_ROOT / name
    if clean:
        return path.with_name(f"{path.stem}_{clean}{path.suffix}")
    return path


def _is_control(model: str) -> bool:
    upper = model.upper()
    return "CONTROL" in upper or "RANDOM" in upper or "SIGNFLIP" in upper or "CORRUPT" in upper or "SHUFFLED" in upper


def _variant_name(model: str) -> str:
    if model == "DGMLP":
        return "MLP+AdamW"
    mapping = {
        "DGMLP_FU": "MLP+FU_RISK_ONLY",
        "DGMLP_FU_GATED": "MLP+FU_SOURCE_GATED",
        "DGMLP_FU_ACTIONABLE": "MLP+FU_BENEFIT_P3_PROXY",
        "DGMLP_FU_RELEASE": "MLP+FU_RELEASE",
        "DGMLP_FU_SPLIT_VIRTUAL": "MLP+FU_SPLIT_CONSENSUS_BENEFIT_PROXY",
        "DGMLP_FU_JVP_REFRESH": "MLP+FU_JVP_REFRESH_BENEFIT_PROXY",
        "DGMLP_FU_BENEFIT_P3": "MLP+FU_BENEFIT_P3_RUNTIME",
        "DGMLP_FU_BENEFIT_P3_STRONG": "MLP+FU_BENEFIT_P3_RUNTIME_STRONG",
        "DGMLP_FU_BENEFIT_P3_SPLIT": "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME",
        "DGMLP_FU_BENEFIT_P3_SPLIT_STRONG": "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG",
        "DGMLP_FU_BENEFIT_P3_JVP_REFRESH": "MLP+FU_BENEFIT_P3_JVP_RUNTIME",
        "DGMLP_FU_BENEFIT_P3_JVP_REFRESH_STRONG": "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG",
        "DGMLP_FU_CONTROL_RANDOM": "MLP+FU_RANDOM_CONTROL",
        "DGMLP_FU_CONTROL_STABLE_RANDOM": "MLP+FU_STABLE_RANDOM_CONTROL",
        "DGMLP_FU_CONTROL_SIGNFLIP": "MLP+FU_SIGNFLIP_CONTROL",
    }
    return mapping.get(model, model)


def _convert_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("model_family") != "MLP":
            continue
        converted = dict(row)
        converted["v22_18_variant"] = _variant_name(row.get("model_name", ""))
        converted["is_control_variant"] = int(_is_control(row.get("model_name", "")))
        converted["stage"] = "D1_internal_smoke" if int(float(row.get("steps") or 0)) <= 1600 else "D2_internal_long_horizon"
        converted["benefit_conditioned_runtime_per_step"] = int(float(row.get("runtime_per_step_policy_integrated") or 0))
        converted["benefit_policy_export_runtime_integrated"] = int(float(row.get("benefit_policy_export_runtime_integrated") or 0))
        converted["benefit_policy_proxy_only"] = int("P3_PROXY" in converted["v22_18_variant"])
        out.append(converted)
    return out


def _summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in rows}
    summary: list[dict[str, Any]] = []
    debt: list[dict[str, Any]] = []
    dist: list[dict[str, Any]] = []
    for model in sorted({r.get("model_name", "") for r in rows if r.get("model_name", "").startswith("DGMLP_FU")}):
        vals = []
        for row in [r for r in rows if r.get("model_name") == model]:
            base = by_key.get((row.get("dataset"), str(row.get("seed")), "DGMLP"))
            if not base:
                continue
            nll_delta = finite_float(row.get("final_test_loss_NLL")) - finite_float(base.get("final_test_loss_NLL"))
            acc_delta = finite_float(row.get("final_test_accuracy")) - finite_float(base.get("final_test_accuracy"))
            auc_delta = finite_float(row.get("AUC_loss_time")) - finite_float(base.get("AUC_loss_time"))
            tail_delta = finite_float(row.get("tail_loss_q99")) - finite_float(base.get("tail_loss_q99"))
            ece_delta = finite_float(row.get("ECE")) - finite_float(base.get("ECE"))
            vals.append((nll_delta, acc_delta, auc_delta, tail_delta, ece_delta, finite_float(row.get("controller_overhead_ratio"))))
            debt.append(
                {
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "model_name": model,
                    "v22_18_variant": row.get("v22_18_variant", ""),
                    "NLL_delta_vs_MLP_AdamW": nll_delta,
                    "accuracy_delta_vs_MLP_AdamW": acc_delta,
                    "AUC_delta_vs_MLP_AdamW": auc_delta,
                    "tail_q99_delta": tail_delta,
                    "ECE_delta": ece_delta,
                    "tail_debt_pass": int(tail_delta <= 0.05),
                    "calibration_debt_pass": int(ece_delta <= 0.02),
                }
            )
            dist.append(
                {
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "model_name": model,
                    "v22_18_variant": row.get("v22_18_variant", ""),
                "gate_accept_count": row.get("gate_accept_count", ""),
                "gate_reject_count": row.get("gate_reject_count", ""),
                "intervention_count": row.get("intervention_count", ""),
                "controller_policy": row.get("controller_policy", ""),
                "source_candidate_type": row.get("source_candidate_type", ""),
                "benefit_policy_export_runtime_integrated": row.get("benefit_policy_export_runtime_integrated", ""),
                "mean_benefit_policy_export_score": row.get("mean_benefit_policy_export_score", ""),
                "mean_benefit_policy_export_accept_score": row.get("mean_benefit_policy_export_accept_score", ""),
                "benefit_policy_export_threshold": row.get("benefit_policy_export_threshold", ""),
                "benefit_policy_export_json": row.get("benefit_policy_export_json", ""),
            }
        )
        n = len(vals)
        if not vals:
            continue
        summary.append(
            {
                "model_name": model,
                "v22_18_variant": _variant_name(model),
                "rows": n,
                "is_control_variant": int(_is_control(model)),
                "NLL_noharm_rows": sum(v[0] <= 0.02 for v in vals),
                "accuracy_noharm_rows": sum(v[1] >= -0.005 for v in vals),
                "tail_q99_noharm_rows": sum(v[3] <= 0.05 for v in vals),
                "ECE_noharm_rows": sum(v[4] <= 0.02 for v in vals),
                "NLL_improvement_rows": sum(v[0] < -0.01 for v in vals),
                "AUC_improvement_rows": sum(v[2] < 0.0 for v in vals),
                "accuracy_improvement_rows": sum(v[1] >= 0.0 for v in vals),
                "max_NLL_delta": max(v[0] for v in vals),
                "mean_NLL_delta": sum(v[0] for v in vals) / n,
                "max_AUC_delta": max(v[2] for v in vals),
                "max_tail_q99_delta": max(v[3] for v in vals),
                "max_ECE_delta": max(v[4] for v in vals),
                "max_controller_overhead_ratio": max(v[5] for v in vals),
                "D_noharm_pass": int(n >= 9 and sum(v[0] <= 0.02 for v in vals) >= 8 and sum(v[1] >= -0.005 for v in vals) >= 8 and sum(v[3] <= 0.05 for v in vals) >= 8 and sum(v[4] <= 0.02 for v in vals) >= 8),
                "D_improvement_metric_pass_before_controls": int(n >= 9 and sum(v[0] < -0.01 for v in vals) >= 5 and sum(v[2] < 0.0 for v in vals) >= 5 and sum(v[1] >= 0.0 for v in vals) >= 4 and max(v[5] for v in vals) <= 0.25),
            }
        )
    real_best = max((int(r["NLL_improvement_rows"]) for r in summary if not int(r["is_control_variant"])), default=0)
    control_best = max((int(r["NLL_improvement_rows"]) for r in summary if int(r["is_control_variant"])), default=0)
    for row in summary:
        row["trained_controls_best_NLL_improvement_rows"] = control_best
        row["D_improvement_pass"] = int(int(row["D_improvement_metric_pass_before_controls"]) and real_best - control_best >= 3 and not int(row["is_control_variant"]))
    return summary, dist, debt


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    suffix = args.reuse_v2217_suffix or args.output_suffix
    if not args.reuse_v2217_suffix:
        cmd = [
            PYTHON,
            "experiments/run_v22_17_kanbefair_dgkan_eval.py",
            "--kanbefair-root",
            str(WORKTREE_ROOT),
            "--datasets",
            args.datasets,
            "--models",
            args.models,
            "--seeds",
            args.seeds,
            "--device",
            args.device,
            "--train-size",
            str(args.train_size),
            "--test-size",
            str(args.test_size),
            "--batch-size",
            str(args.batch_size),
            "--steps",
            str(args.steps),
            "--hidden",
            str(args.hidden),
            "--log-interval",
            str(args.log_interval),
            "--experiment-tag",
            args.experiment_tag,
            "--output-suffix",
            args.output_suffix,
            "--jvp-refresh-dim",
            str(args.jvp_refresh_dim),
            "--jvp-refresh-min-history",
            str(args.jvp_refresh_min_history),
            "--jvp-refresh-history-window",
            str(args.jvp_refresh_history_window),
            "--jvp-refresh-scale",
            str(args.jvp_refresh_scale),
            "--jvp-refresh-ratio-cap",
            str(args.jvp_refresh_ratio_cap),
            "--actionable-gain-tolerance",
            str(args.actionable_gain_tolerance),
            "--virtual-loss-tolerance",
            str(args.virtual_loss_tolerance),
            "--virtual-gate-scale",
            str(args.virtual_gate_scale),
            "--benefit-p3-ratio-cap",
            str(args.benefit_p3_ratio_cap),
            "--benefit-p3-strong-ratio-cap",
            str(args.benefit_p3_strong_ratio_cap),
            "--benefit-p3-min-score",
            str(args.benefit_p3_min_score),
            "--benefit-p3-strong-min-score",
            str(args.benefit_p3_strong_min_score),
            "--benefit-p3-jvp-strong-scale",
            str(args.benefit_p3_jvp_strong_scale),
            "--benefit-p3-jvp-strong-ratio-cap",
            str(args.benefit_p3_jvp_strong_ratio_cap),
            "--benefit-p3-jvp-interval",
            str(args.benefit_p3_jvp_interval),
        ]
        if args.benefit_policy_export_json:
            cmd.extend(["--benefit-policy-export-json", args.benefit_policy_export_json])
            if str(args.benefit_policy_export_threshold).strip():
                cmd.extend(["--benefit-policy-export-threshold", str(args.benefit_policy_export_threshold)])
        env = {"CUDA_VISIBLE_DEVICES": args.physical_gpu} if args.physical_gpu else {}
        if args.clear_proxy_env:
            for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
                env[key] = ""
        proc = run_logged(
            cmd,
            task_id="D-MLP-FU-mainline-bridge",
            env=env or None,
            gpu=f"physical {args.physical_gpu or 'default'}; script device {args.device}",
            timeout=None,
        )
        if proc.returncode != 0:
            append_exec(
                " ".join(shlex.quote(x) for x in cmd),
                task_id="D-MLP-FU-mainline-convert-skipped",
                status="fail",
                gpu=f"physical {args.physical_gpu or 'default'}; script device {args.device}",
                exit_code=proc.returncode,
                note="v22_17 bridge subprocess failed; conversion still tries to read any partial artifact",
            )

    bridge = _v17("v22_17_kanbefair_dgkan_bridge_matrix.csv", suffix)
    source = _v17("v22_17_task_useful_source_matrix.csv", suffix)
    neutral = _v17("v22_17_task_neutral_control_source_matrix.csv", suffix)
    rows = _convert_rows(read_rows(bridge))
    summary, dist, debt = _summaries(rows)
    write_rows(_out("v22_18_mlp_fu_task_matrix.csv", args.output_suffix), rows)
    write_rows(_out("v22_18_mlp_fu_controls_matrix.csv", args.output_suffix), [r for r in rows if int(r.get("is_control_variant") or 0)])
    write_rows(_out("v22_18_mlp_fu_noharm_improvement_summary.csv", args.output_suffix), summary)
    write_rows(_out("v22_18_mlp_fu_action_distribution.csv", args.output_suffix), dist)
    write_rows(_out("v22_18_mlp_fu_tail_calibration_debt.csv", args.output_suffix), debt)
    write_rows(_out("v22_18_mlp_fu_source_artifact_pointers.csv", args.output_suffix), [{"bridge_csv": str(bridge), "source_csv": str(source), "neutral_control_csv": str(neutral), "exists": int(bridge.exists())}])
    cmdline = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmdline,
        task_id="D-MLP-FU-mainline-convert",
        status="pass" if rows else "fail",
        exit_code=0 if rows else 1,
        files=(
            f"{_out('v22_18_mlp_fu_task_matrix.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_mlp_fu_noharm_improvement_summary.csv', args.output_suffix).relative_to(ROOT)}"
        ),
        note=f"rows={len(rows)}; source={source.relative_to(ROOT) if source.exists() else source}; neutral={neutral.relative_to(ROOT) if neutral.exists() else neutral}",
    )


if __name__ == "__main__":
    main()
