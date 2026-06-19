#!/usr/bin/env python3
"""Run or convert v22.18 KAN native-basis ladder evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.kan_native_jvp import summarize_native_ladder  # noqa: E402
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
    p.add_argument("--models", default="DGKAN_DCHE")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--physical-gpu", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--mlp-hidden", type=int, default=80)
    p.add_argument("--steps", type=int, default=440)
    p.add_argument("--log-interval", type=int, default=110)
    p.add_argument("--jvp-refresh-interval", type=int, default=40)
    p.add_argument("--jvp-refresh-scale", type=float, default=0.05)
    p.add_argument("--jvp-refresh-ratio-cap", type=float, default=0.05)
    p.add_argument("--jvp-refresh-dim", type=int, default=4)
    p.add_argument("--jvp-refresh-min-history", type=int, default=8)
    p.add_argument("--jvp-refresh-history-window", type=int, default=32)
    p.add_argument("--jvp-refresh-release-on-accept", action="store_true")
    p.add_argument("--virtual-loss-tolerance", type=float, default=1.0e-5)
    p.add_argument("--actionable-gain-tolerance", type=float, default=0.002)
    p.add_argument("--fu-nonbasis-grad-scale", type=float, default=0.1)
    p.add_argument("--jvp-refresh-require-virtual-improvement", action="store_true")
    p.add_argument("--jvp-refresh-virtual-improvement-margin", type=float, default=0.0)
    p.add_argument("--experiment-tag", default="v22_18_kan_native_dche_h8_h440")
    p.add_argument("--output-suffix", default="v22_18_kan_native_dche_h8_h440")
    p.add_argument("--reuse-v2217-suffix", default="")
    p.add_argument("--mlp-task-csv", default="")
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


def _best_mlpfu(rows: list[dict[str, str]], dataset: str, seed: str) -> dict[str, str] | None:
    candidates = [
        r
        for r in rows
        if r.get("dataset") == dataset
        and str(r.get("seed")) == str(seed)
        and r.get("model_name", "").startswith("DGMLP_FU")
        and not int(float(r.get("is_control_variant") or 0))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: finite_float(r.get("final_test_loss_NLL"), 1.0e9))


def _kan_vs_mlp(rows: list[dict[str, str]], mlp_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("model_name", "").endswith("_FU_NATIVE_JVP_REFRESH"):
            continue
        mlp = _best_mlpfu(mlp_rows, row.get("dataset", ""), str(row.get("seed", "")))
        if not mlp:
            continue
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "kan_model_name": row.get("model_name", ""),
                "mlpfu_model_name": mlp.get("model_name", ""),
                "KAN_vs_MLPFU_NLL_delta": finite_float(row.get("final_test_loss_NLL")) - finite_float(mlp.get("final_test_loss_NLL")),
                "KAN_vs_MLPFU_AUC_delta": finite_float(row.get("AUC_loss_time")) - finite_float(mlp.get("AUC_loss_time")),
                "KAN_vs_MLPFU_accuracy_delta": finite_float(row.get("final_test_accuracy")) - finite_float(mlp.get("final_test_accuracy")),
                "KAN_final_NLL": row.get("final_test_loss_NLL", ""),
                "MLPFU_final_NLL": mlp.get("final_test_loss_NLL", ""),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    suffix = args.reuse_v2217_suffix or args.output_suffix
    if not args.reuse_v2217_suffix:
        cmd = [
            PYTHON,
            "experiments/run_v22_17_kan_basis_native_audit.py",
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
            "--hidden",
            str(args.hidden),
            "--mlp-hidden",
            str(args.mlp_hidden),
            "--steps",
            str(args.steps),
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
            "--jvp-refresh-interval",
            str(args.jvp_refresh_interval),
            "--fu-nonbasis-grad-scale",
            str(args.fu_nonbasis_grad_scale),
            "--virtual-loss-tolerance",
            str(args.virtual_loss_tolerance),
            "--actionable-gain-tolerance",
            str(args.actionable_gain_tolerance),
        ]
        if args.jvp_refresh_release_on_accept:
            cmd.append("--jvp-refresh-release-on-accept")
        if args.jvp_refresh_require_virtual_improvement:
            cmd.append("--jvp-refresh-require-virtual-improvement")
            cmd.extend(["--jvp-refresh-virtual-improvement-margin", str(args.jvp_refresh_virtual_improvement_margin)])
        env = {"CUDA_VISIBLE_DEVICES": args.physical_gpu} if args.physical_gpu else None
        run_logged(
            cmd,
            task_id="E-KAN-native-ladder",
            env=env,
            gpu=f"physical {args.physical_gpu or 'default'}; script device {args.device}",
            timeout=None,
        )

    native_path = _v17("v22_17_kan_basis_native_audit.csv", suffix)
    source_path = _v17("v22_17_kan_basis_native_source_timeseries.csv", suffix)
    ctrl_path = _v17("v22_17_kan_basis_native_controllability.csv", suffix)
    rows = read_rows(native_path)
    source_rows = read_rows(source_path)
    ctrl_rows = read_rows(ctrl_path)
    summary = summarize_native_ladder(rows)
    mlp_rows = read_rows(Path(args.mlp_task_csv) if args.mlp_task_csv else OUT_ROOT / "v22_18_mlp_fu_task_matrix.csv")
    kan_vs_mlp = _kan_vs_mlp(rows, mlp_rows)
    write_rows(_out("v22_18_kan_native_ladder_matrix.csv", args.output_suffix), rows)
    write_rows(_out("v22_18_kan_native_controls_matrix.csv", args.output_suffix), ctrl_rows)
    write_rows(_out("v22_18_kan_vs_mlpfu_matrix.csv", args.output_suffix), kan_vs_mlp)
    write_rows(
        _out("v22_18_basis_channel_energy_matrix.csv", args.output_suffix),
        [
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "model_name": r.get("model_name", ""),
                "basis_channel_energy_fraction": r.get("basis_channel_energy_fraction", ""),
                "readout_channel_energy_fraction": r.get("readout_channel_energy_fraction", ""),
                "basis_to_readout_leakage": r.get("readout_channel_energy_fraction", ""),
                "uses_dense_output_jacobian_official": r.get("uses_dense_output_jacobian_official", ""),
                "audit_dense_finite_diff_j_used": r.get("audit_dense_finite_diff_j_used", ""),
            }
            for r in rows
            if r.get("model_family") == "KAN"
        ],
    )
    write_rows(
        _out("v22_18_kan_efficiency_matrix.csv", args.output_suffix),
        [
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "model_name": r.get("model_name", ""),
                "full_loop_step_ms": r.get("full_loop_step_ms", ""),
                "official_train_step_ms": r.get("official_train_step_ms", ""),
                "diagnostic_source_eval_step_ms": r.get("diagnostic_source_eval_step_ms", ""),
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "official_controller_overhead_ratio": r.get("official_controller_overhead_ratio", ""),
                "full_loop_ratio_vs_matched_MLP": r.get("full_loop_ratio_vs_mlp", ""),
                "official_full_loop_ratio_vs_matched_MLP": r.get("official_full_loop_ratio_vs_mlp", ""),
                "FLOPs_ratio_vs_matched_MLP": r.get("flops_ratio_vs_mlp", ""),
                "param_ratio_vs_matched_MLP": r.get("param_ratio_vs_mlp", ""),
                "uses_dense_output_jacobian": r.get("uses_dense_output_jacobian_official", ""),
                "native_jvp_vjp_gradcheck_rel_error": r.get("analytic_vs_autograd_rel_error_audit", ""),
            }
            for r in rows
        ],
    )
    write_rows(_out("v22_18_kan_native_ladder_summary.csv", args.output_suffix), summary)
    cmdline = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmdline,
        task_id="E-KAN-native-ladder-convert",
        status="pass" if rows else "fail",
        exit_code=0 if rows else 1,
        files=(
            f"{_out('v22_18_kan_native_ladder_matrix.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_kan_native_ladder_summary.csv', args.output_suffix).relative_to(ROOT)}"
        ),
        note=f"rows={len(rows)}; source_rows={len(source_rows)}; ctrl_rows={len(ctrl_rows)}; KAN-vs-MLPFU rows={len(kan_vs_mlp)}",
    )


if __name__ == "__main__":
    main()
