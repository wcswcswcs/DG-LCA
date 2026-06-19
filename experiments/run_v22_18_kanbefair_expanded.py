#!/usr/bin/env python3
"""Run v22.18 KANbeFair expanded availability and seed0 task matrix."""

from __future__ import annotations

import argparse
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
    p.add_argument("--datasets", default="MNIST,FMNIST,KMNIST,EMNIST-Letters,CIFAR10,SVHN,Bean,Rice,Spam,Titanic,Wine,Income,Bank,Telescope")
    p.add_argument("--models", default="DGMLP,DGMLP_FU_SPLIT_VIRTUAL,DGKAN_DFOU,DGKAN_DFOU_FU_SPLIT_VIRTUAL")
    p.add_argument("--physical-gpu", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-run-datasets", type=int, default=6)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--output-suffix", default="v22_18_kanbefair_expanded_seed0_h120")
    p.add_argument("--reuse-v2217-suffix", default="")
    p.add_argument("--skip-task-run", action="store_true")
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


def _family(dataset: str) -> str:
    if dataset in {"MNIST", "FMNIST", "KMNIST", "EMNIST-Letters", "CIFAR10", "SVHN"}:
        return "vision"
    if dataset in {"Bean", "Rice", "Spam", "Titanic", "Wine", "Income", "Bank", "Telescope"}:
        return "tabular"
    return "diagnostic"


def _base_dgkan_model(model_name: str) -> str:
    if model_name.startswith("DGKAN_DCHE"):
        return "DGKAN_DCHE"
    if model_name.startswith("DGKAN_DFOU"):
        return "DGKAN_DFOU"
    return model_name


def _gradcheck_lookup() -> dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, str]] = {}
    for path in sorted(V22_17_ROOT.glob("v22_17_basis_jvp_vjp_gradcheck*.csv")):
        if path.name.endswith("_failures.csv"):
            continue
        for row in read_rows(path):
            if row.get("implementation_mode") not in {"bridge_dense_cache", "native_no_dense_basis"} or row.get("selector") != "basis":
                continue
            key = (
                row.get("implementation_mode", ""),
                row.get("dataset", ""),
                str(row.get("seed", "")),
                row.get("model_name", ""),
                str(row.get("hidden", "")),
                str(row.get("train_size", "")),
                str(row.get("test_size", "")),
                str(row.get("batch_size", "")),
                row.get("initial_data_batch_fingerprint", ""),
            )
            out[key] = row
    return out


def _attach_bridge_gradcheck(rows: list[dict[str, str]]) -> None:
    lookup = _gradcheck_lookup()
    for row in rows:
        model = row.get("model_name", "")
        if not model.startswith("DGKAN_"):
            continue
        dense_spec = str(row.get("uses_dense_basis_tensor_spec", "")).strip()
        manual_kernel = str(row.get("manual_kernel_variant", ""))
        wanted_impl = "native_no_dense_basis" if dense_spec in {"0", "0.0"} and ("triton" in manual_kernel or "matmul" in manual_kernel) else "bridge_dense_cache"
        key = (
            wanted_impl,
            row.get("dataset", ""),
            str(row.get("seed", "")),
            _base_dgkan_model(model),
            str(row.get("hidden", "")),
            str(row.get("train_size", "")),
            str(row.get("test_size", "")),
            str(row.get("batch_size", "")),
            row.get("initial_data_batch_fingerprint", ""),
        )
        gradcheck = lookup.get(key)
        if not gradcheck:
            continue
        row["bridge_gradcheck_source"] = gradcheck.get("_source_file", "")
        row["manual_kernel_variant"] = gradcheck.get("manual_kernel_variant", "")
        row["uses_dense_basis_tensor_spec"] = gradcheck.get("uses_dense_basis_tensor_spec", "")
        row["func_jvp_available"] = gradcheck.get("func_jvp_available", "")
        row["func_jvp_relerr_max"] = gradcheck.get("func_jvp_relerr_max", "")
        row["analytic_vs_autograd_rel_error_audit"] = gradcheck.get("analytic_vs_autograd_rel_error_audit", "")
        row["official_basis_jvp_vjp_gradcheck_pass"] = gradcheck.get("official_basis_jvp_vjp_gradcheck_pass", "")
        row["bridge_gradcheck_matched_same_initial_batch"] = 1


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in rows}
    out: list[dict[str, Any]] = []
    for dataset in sorted({r.get("dataset", "") for r in rows}):
        seed = "0"
        mlp = by_key.get((dataset, seed, "DGMLP"))
        mlpfu = by_key.get((dataset, seed, "DGMLP_FU_SPLIT_VIRTUAL"))
        kan = by_key.get((dataset, seed, "DGKAN_DFOU"))
        kanfu = by_key.get((dataset, seed, "DGKAN_DFOU_FU_SPLIT_VIRTUAL"))
        out.append(
            {
                "dataset": dataset,
                "dataset_family": _family(dataset),
                "MLP_NLL": "" if not mlp else mlp.get("final_test_loss_NLL", ""),
                "MLPFU_NLL": "" if not mlpfu else mlpfu.get("final_test_loss_NLL", ""),
                "MLPFU_NLL_delta_vs_DGMLP": "" if not mlp or not mlpfu else finite_float(mlpfu.get("final_test_loss_NLL")) - finite_float(mlp.get("final_test_loss_NLL")),
                "MLPFU_AUC_delta_vs_DGMLP": "" if not mlp or not mlpfu else finite_float(mlpfu.get("AUC_loss_time")) - finite_float(mlp.get("AUC_loss_time")),
                "KAN_NLL": "" if not kan else kan.get("final_test_loss_NLL", ""),
                "KANFU_NLL": "" if not kanfu else kanfu.get("final_test_loss_NLL", ""),
                "KANFU_NLL_delta_vs_DGKAN": "" if not kan or not kanfu else finite_float(kanfu.get("final_test_loss_NLL")) - finite_float(kan.get("final_test_loss_NLL")),
                "KANFU_AUC_delta_vs_DGKAN": "" if not kan or not kanfu else finite_float(kanfu.get("AUC_loss_time")) - finite_float(kan.get("AUC_loss_time")),
                "KANFU_vs_MLPFU_NLL_delta": "" if not kanfu or not mlpfu else finite_float(kanfu.get("final_test_loss_NLL")) - finite_float(mlpfu.get("final_test_loss_NLL")),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    env_cmd = [PYTHON, "experiments/run_v22_17_kanbefair_env_smoke.py", "--datasets", args.datasets]
    run_logged(env_cmd, task_id="G-expanded-availability", gpu="CPU/dataset loaders", timeout=None)
    availability = read_rows(V22_17_ROOT / "v22_17_kanbefair_dataset_availability_matrix.csv")
    for row in availability:
        row["dataset_family"] = _family(row.get("dataset", ""))
        row["algorithm_failure"] = 0 if row.get("availability") != "available" else ""
    write_rows(OUT_ROOT / "v22_18_kanbefair_dataset_availability.csv", availability)
    available = [r.get("dataset", "") for r in availability if r.get("availability") == "available"]
    run_datasets = available[: max(0, int(args.max_run_datasets))]
    suffix = args.reuse_v2217_suffix or args.output_suffix
    if run_datasets and not args.skip_task_run and not args.reuse_v2217_suffix:
        cmd = [
            PYTHON,
            "experiments/run_v22_17_kanbefair_dgkan_eval.py",
            "--kanbefair-root",
            str(WORKTREE_ROOT),
            "--datasets",
            ",".join(run_datasets),
            "--models",
            args.models,
            "--seeds",
            "0",
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
            str(max(1, args.steps // 3)),
            "--experiment-tag",
            "v22_18_kanbefair_expanded_seed0",
            "--output-suffix",
            args.output_suffix,
        ]
        env = {"CUDA_VISIBLE_DEVICES": args.physical_gpu} if args.physical_gpu else None
        run_logged(cmd, task_id="G-expanded-task-matrix", env=env, gpu=f"physical {args.physical_gpu or 'default'}; script device {args.device}", timeout=None)
    task_rows = read_rows(_v17("v22_17_kanbefair_dgkan_bridge_matrix.csv", suffix))
    for row in task_rows:
        row["dataset_family"] = _family(row.get("dataset", ""))
        row["external_expanded_seed0"] = 1
    _attach_bridge_gradcheck(task_rows)
    summary = _summary(task_rows)
    available_total = len(available)
    run_dataset_count = len(run_datasets)
    mlpfu_noharm = sum(finite_float(r.get("MLPFU_NLL_delta_vs_DGMLP"), 999.0) <= 0.02 for r in summary if r.get("MLPFU_NLL_delta_vs_DGMLP") != "")
    mlpfu_improve = sum(finite_float(r.get("MLPFU_NLL_delta_vs_DGMLP"), 0.0) < -0.01 or finite_float(r.get("MLPFU_AUC_delta_vs_DGMLP"), 0.0) < 0.0 for r in summary if r.get("MLPFU_NLL_delta_vs_DGMLP") != "")
    kanfu_noharm = sum(finite_float(r.get("KANFU_NLL_delta_vs_DGKAN"), 999.0) <= 0.02 for r in summary if r.get("KANFU_NLL_delta_vs_DGKAN") != "")
    kanfu_improve = sum(finite_float(r.get("KANFU_NLL_delta_vs_DGKAN"), 0.0) < -0.01 or finite_float(r.get("KANFU_AUC_delta_vs_DGKAN"), 0.0) < 0.0 for r in summary if r.get("KANFU_NLL_delta_vs_DGKAN") != "")
    pass_rows = [
        {
            "available_datasets": available_total,
            "run_dataset_count": run_dataset_count,
            "run_datasets": ",".join(run_datasets),
            "task_rows": len(task_rows),
            "MLPFU_noharm_rows": mlpfu_noharm,
            "MLPFU_improvement_rows": mlpfu_improve,
            "KANFU_noharm_rows": kanfu_noharm,
            "KANFU_improvement_rows": kanfu_improve,
            "G_external_noharm_pass": int(run_dataset_count > 0 and mlpfu_noharm / max(1, run_dataset_count) >= 0.70 and kanfu_noharm / max(1, run_dataset_count) >= 0.70),
            "G_external_improvement_pass": int(run_dataset_count > 0 and mlpfu_improve / max(1, run_dataset_count) >= 0.40 and kanfu_improve / max(1, run_dataset_count) >= 0.50),
        }
    ]
    write_rows(_out("v22_18_kanbefair_expanded_task_matrix.csv", args.output_suffix), task_rows)
    write_rows(_out("v22_18_kanbefair_four_square_summary.csv", args.output_suffix), summary)
    write_rows(_out("v22_18_kanbefair_expanded_pass_summary.csv", args.output_suffix), pass_rows)
    cmdline = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmdline,
        task_id="G-expanded-convert",
        status="pass" if task_rows else "partial",
        exit_code=0,
        files=(
            "results/v22_18/v22_18_kanbefair_dataset_availability.csv, "
            f"{_out('v22_18_kanbefair_expanded_task_matrix.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_kanbefair_four_square_summary.csv', args.output_suffix).relative_to(ROOT)}"
        ),
        note=f"available_total={available_total}; run_dataset_count={run_dataset_count}; run_datasets={','.join(run_datasets) if run_datasets else 'none'}",
    )


if __name__ == "__main__":
    main()
