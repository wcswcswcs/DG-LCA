#!/usr/bin/env python
"""v12.31 Rational workspace-pass AUC/task hardening audit.

This runner is intentionally narrow: it only executes Rational candidates after
the workspace gate has been rechecked.  Trajectory/AUC helper logic lives in
dgkan.diagnostics.basis_workspace; this file orchestrates datasets and artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402
from experiments.run_v1231_basis_kernel_workspace import linec_metrics, make_adamw, make_basis_model  # noqa: E402
from dgkan.diagnostics.basis_workspace import (  # noqa: E402
    V1231_BASIS_CANDIDATES,
    V1232_BASIS_CANDIDATES,
    V1233_BASIS_CANDIDATES,
    V12342_BASIS_CANDIDATES,
    V1235_BASIS_CANDIDATES,
    finite_mean,
    family_telemetry_metrics,
    fnum,
    parse_csv,
    parse_ints,
    rational_telemetry_metrics,
    rational_tail_metrics,
    train_epoch_timed,
    trajectory_auc,
    trajectory_time_auc,
)
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from dgkan.training.eval import classification_basic  # noqa: E402


DEFAULT_CANDIDATES = ",".join([
    "D-RAT7-FusedGroupRationalNoMaterialize",
    "D-RAT8-RecomputeDenominatorBackward",
    "D-RAT10-FusedDenNumReadoutGrad",
    "D-RAT11-LowMemGroupSharedDenomPlusLineC",
    "D-RAT12-RationalWorkspaceMinStrongDiag",
])


def workspace_pass_map(path: Path) -> dict[tuple[str, str, int], int]:
    if not path.exists():
        return {}
    rows = v1223.read_csv_rows(path)
    out: dict[tuple[str, str, int], int] = {}
    for row in rows:
        out[(str(row.get("candidate_id", "")), str(row.get("dataset", "")), int(float(row.get("seed", 0) or 0)))] = int(float(row.get("workspace_gate_pass", 0) or 0))
    return out


def train_trajectory(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    rng_seed: int,
    device: torch.device,
    architecture: str,
    candidate_id: str,
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    gen = torch.Generator(device=device).manual_seed(int(rng_seed))
    rows: list[dict[str, Any]] = []
    accs: list[float] = []
    nlls: list[float] = []
    ce99s: list[float] = []
    step_times: list[float] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = train_epoch_timed(model, opt, x_train, y_train, int(args.batch_size), gen, device)
        ev = classification_basic(model, x_val, y_val)
        accs.append(float(ev["acc"]))
        nlls.append(float(ev["NLL"]))
        ce99s.append(float(ev["CEp99"]))
        step_times.append(float(train_stats["step_time_q90_ms"]))
        rows.append({
            "stage": "V1231_RATIONAL_AUC_TRAJECTORY",
            "architecture": architecture,
                "candidate_id": candidate_id,
                "dataset": dataset,
                "seed": int(seed),
            "epoch": epoch,
            "train_loss_mean": train_stats["train_loss_mean"],
            "step_time_q90_ms": train_stats["step_time_q90_ms"],
            "epoch_steps": train_stats["epoch_steps"],
            "val_acc": ev["acc"],
            "NLL": ev["NLL"],
            "ECE": ev["ECE"],
            "CEp99": ev["CEp99"],
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    summary = {
        "final_val_acc": accs[-1] if accs else float("nan"),
        "final_NLL": nlls[-1] if nlls else float("nan"),
        "final_CEp99": ce99s[-1] if ce99s else float("nan"),
        "acc_auc_step": trajectory_auc(accs),
        "NLL_AUC_step": trajectory_auc(nlls),
        "NLL_AUC_time": trajectory_time_auc(nlls, step_times),
        "step_time_q90_mean_ms": finite_mean(step_times),
    }
    return rows, summary


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v1231_rational_auc_repair")
    ap.add_argument("--run-id", default="v1231_rational_auc_repair")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--candidate-registry", choices=["v1231", "v1232", "v1233", "v12342", "v1235"], default="v1231")
    ap.add_argument("--workspace-csv", default="")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="12319500,12320600,12321600")
    ap.add_argument("--ce-p99-tolerance", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    v1223.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.31 Rational AUC hardening requires CUDA")
    torch.cuda.set_device(device)

    datasets = [v1223.v120._canonical_dataset(d) for d in parse_csv(args.datasets)]
    seeds = parse_ints(args.seeds)
    candidate_registry = (
        V1235_BASIS_CANDIDATES
        if args.candidate_registry == "v1235"
        else V12342_BASIS_CANDIDATES
        if args.candidate_registry == "v12342"
        else V1233_BASIS_CANDIDATES
        if args.candidate_registry == "v1233"
        else V1232_BASIS_CANDIDATES
        if args.candidate_registry == "v1232"
        else V1231_BASIS_CANDIDATES
    )
    selected = [candidate_registry[c] for c in parse_csv(args.candidates)]
    stage_prefix = "V1235" if args.candidate_registry == "v1235" else "V12342" if args.candidate_registry == "v12342" else "V1233" if args.candidate_registry == "v1233" else "V1232" if args.candidate_registry == "v1232" else "V1231"
    wpass = workspace_pass_map(Path(args.workspace_csv)) if args.workspace_csv else {}

    candidate_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []

    mlp_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
            data = v1223.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)

            mlp = MLPBaseline(int(input_dim), int(output_dim), int(args.mlp_hidden), int(seed) + 1231810, device).to(device)
            mlp_opt = make_adamw(args, mlp.parameters())
            mlp_traj, mlp_summary = train_trajectory(
                mlp,
                mlp_opt,
                x_train,
                y_train,
                x_val,
                y_val,
                args,
                int(seed),
                int(seed) + 1231820,
                device,
                "MLP",
                "MLPReference",
                dataset,
            )
            trajectory_rows.extend(mlp_traj)
            mlp_cache[(dataset, int(seed))] = mlp_summary

            for cand in selected:
                gate_key = (cand.candidate_id, dataset, int(seed))
                if wpass and not int(wpass.get(gate_key, 0)):
                    candidate_rows.append({
                        "stage": f"{stage_prefix}_RATIONAL_AUC_CANDIDATE",
                        "run_id": args.run_id,
                        "candidate_id": cand.candidate_id,
                        "family": cand.family,
                        "mapped_method_id": cand.method_id,
                        "dataset": dataset,
                        "seed": int(seed),
                        "executed": 0,
                        "skip_reason": "workspace_gate_fail_in_input_workspace_csv",
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    })
                    continue

                model, _spec = make_basis_model(cand.method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 1231830)
                opt = make_adamw(args, [p for p in model.parameters() if p.requires_grad])
                traj, summary = train_trajectory(
                    model,
                    opt,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    args,
                    int(seed),
                    int(seed) + 1231840,
                    device,
                    f"Classic{cand.family}",
                    cand.candidate_id,
                    dataset,
                )
                trajectory_rows.extend(traj)
                b = min(int(args.linec_batch_size), int(x_train.shape[0]), int(x_val.shape[0]))
                this_linec: list[dict[str, Any]] = []
                for ls in parse_ints(args.linec_seeds):
                    lm = linec_metrics(model, x_train[:b], y_train[:b], x_val[:b], y_val[:b], int(ls), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                    row = {
                        "stage": f"{stage_prefix}_RATIONAL_AUC_LINEC",
                        "run_id": args.run_id,
                        "candidate_id": cand.candidate_id,
                        "family": cand.family,
                        "mapped_method_id": cand.method_id,
                        "dataset": dataset,
                        "seed": int(seed),
                        "linec_seed": int(ls),
                        **lm,
                        "label_used_for_audit_only": 1,
                        "uses_linec_hard_target_for_direction": 0,
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    }
                    this_linec.append(row)
                    linec_rows.append(row)
                linec_pass = sum(
                    int(fnum(r.get("CouplingR2"), -999.0) >= 0.15 and fnum(r.get("NoiseSignalLeak"), 999.0) <= 0.20 and fnum(r.get("RealSignalReservoirRatio"), 999.0) <= 0.70)
                    for r in this_linec
                )
                mlp_summary = mlp_cache[(dataset, int(seed))]
                nll_auc_step_ratio = summary["NLL_AUC_step"] / mlp_summary["NLL_AUC_step"] if fnum(mlp_summary["NLL_AUC_step"]) > 0 else float("nan")
                nll_auc_time_ratio = summary["NLL_AUC_time"] / mlp_summary["NLL_AUC_time"] if fnum(mlp_summary["NLL_AUC_time"]) > 0 else float("nan")
                delta = summary["final_val_acc"] - mlp_summary["final_val_acc"]
                ce_delta = summary["final_CEp99"] - mlp_summary["final_CEp99"]
                if args.candidate_registry in ("v12342", "v1235"):
                    tail = family_telemetry_metrics(model, x_train[: min(int(args.batch_size), int(x_train.shape[0]))], cand.family, int(seed) + 1233420)
                elif args.candidate_registry == "v1233":
                    tail = rational_telemetry_metrics(model, x_train[: min(int(args.batch_size), int(x_train.shape[0]))], int(seed) + 1233330)
                elif args.candidate_registry == "v1232":
                    tail = rational_tail_metrics(model, x_train[: min(int(args.batch_size), int(x_train.shape[0]))])
                else:
                    tail = {}
                candidate_rows.append({
                    "stage": f"{stage_prefix}_RATIONAL_AUC_CANDIDATE",
                    "run_id": args.run_id,
                    "candidate_id": cand.candidate_id,
                    "family": cand.family,
                    "mapped_method_id": cand.method_id,
                    "dataset": dataset,
                    "seed": int(seed),
                    "executed": 1,
                    "epochs": int(args.epochs),
                    "adamw_foreach": str(args.adamw_foreach),
                    "train_size": int(args.train_size),
                    "val_size": int(args.val_size),
                    "val_acc": summary["final_val_acc"],
                    "mlp_val_acc": mlp_summary["final_val_acc"],
                    "final_delta_vs_MLP": delta,
                    "NLL_AUC_step": summary["NLL_AUC_step"],
                    "mlp_NLL_AUC_step": mlp_summary["NLL_AUC_step"],
                    "AUC_step_ratio_vs_MLP": nll_auc_step_ratio,
                    "NLL_AUC_time": summary["NLL_AUC_time"],
                    "mlp_NLL_AUC_time": mlp_summary["NLL_AUC_time"],
                    "AUC_time_ratio_vs_MLP": nll_auc_time_ratio,
                    "AUC_metric_source": "epoch_mean_NLL_and_step_q90_time",
                    "step_time_q90_mean_ms": summary["step_time_q90_mean_ms"],
                    "mlp_step_time_q90_mean_ms": mlp_summary["step_time_q90_mean_ms"],
                    "CEp99": summary["final_CEp99"],
                    "mlp_CEp99": mlp_summary["final_CEp99"],
                    "CEp99_delta_vs_MLP": ce_delta,
                    "LineC_pass_count": linec_pass,
                    "LineC_seed_count": len(this_linec),
                    **tail,
                    "row_auc_probe_pass": int(delta >= -0.015 and nll_auc_time_ratio <= 1.10 and linec_pass >= math.ceil(len(this_linec) / 2.0) and ce_delta <= float(args.ce_p99_tolerance)),
                    "promotion_allowed": 0,
                    "no_fake": 1,
                })
                torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_candidate.setdefault(str(row.get("candidate_id", "")), []).append(row)
    for cid, rows in sorted(by_candidate.items()):
        executed = [r for r in rows if int(float(r.get("executed", 0) or 0))]
        skipped = len(rows) - len(executed)
        deltas = [fnum(r.get("final_delta_vs_MLP")) for r in executed if math.isfinite(fnum(r.get("final_delta_vs_MLP")))]
        auc_time = [fnum(r.get("AUC_time_ratio_vs_MLP")) for r in executed if math.isfinite(fnum(r.get("AUC_time_ratio_vs_MLP")))]
        auc_step = [fnum(r.get("AUC_step_ratio_vs_MLP")) for r in executed if math.isfinite(fnum(r.get("AUC_step_ratio_vs_MLP")))]
        ce_delta = [fnum(r.get("CEp99_delta_vs_MLP")) for r in executed if math.isfinite(fnum(r.get("CEp99_delta_vs_MLP")))]
        linec_pass = sum(int(float(r.get("LineC_pass_count", 0) or 0)) for r in executed)
        linec_total = sum(int(float(r.get("LineC_seed_count", 0) or 0)) for r in executed)
        mean_delta = finite_mean(deltas)
        worst_delta = min(deltas) if deltas else float("nan")
        max_auc_time = max(auc_time) if auc_time else float("nan")
        max_auc_step = max(auc_step) if auc_step else float("nan")
        max_ce_delta = max(ce_delta) if ce_delta else float("nan")
        linec_required = math.ceil((5.0 / 9.0) * linec_total) if linec_total else 1
        near = int(
            skipped == 0
            and len(executed) == len(datasets) * len(seeds)
            and mean_delta >= -0.015
            and worst_delta >= -0.035
            and max_auc_time <= 1.10
            and linec_pass >= linec_required
            and max_ce_delta <= float(args.ce_p99_tolerance)
        )
        near_v1232 = int(
            skipped == 0
            and len(executed) == len(datasets) * len(seeds)
            and mean_delta >= 0.0
            and worst_delta >= -0.01
            and max_auc_time <= 1.00
            and linec_pass >= 25
            and max_ce_delta <= 0.05
        )
        near_v1233 = int(
            skipped == 0
            and len(executed) == len(datasets) * len(seeds)
            and mean_delta >= -0.005
            and worst_delta >= -0.025
            and max_auc_time <= 1.05
            and linec_pass >= 20
            and max_ce_delta <= 0.50
        )
        official_v1233 = int(
            skipped == 0
            and len(executed) == len(datasets) * len(seeds)
            and mean_delta >= 0.0
            and worst_delta >= -0.010
            and max_auc_time <= 1.00
            and linec_pass >= linec_total
            and max_ce_delta <= 0.05
        )
        summary_rows.append({
            "stage": f"{stage_prefix}_RATIONAL_AUC_SUMMARY",
            "candidate_id": cid,
            "family": (executed[0] if executed else rows[0]).get("family", "") if rows else "",
            "rows": len(rows),
            "executed_rows": len(executed),
            "skipped_rows": skipped,
            "mean_delta_vs_MLP": mean_delta,
            "worst_delta_vs_MLP": worst_delta,
            "max_AUC_step_ratio_vs_MLP": max_auc_step,
            "max_AUC_time_ratio_vs_MLP": max_auc_time,
            "AUC_metric_source": "epoch_mean_NLL_and_step_q90_time",
            "max_CEp99_delta_vs_MLP": max_ce_delta,
            "LineC_pass_count": linec_pass,
            "LineC_seed_count": linec_total,
            "LineC_required_count": linec_required,
            "row_auc_probe_pass_count": sum(int(float(r.get("row_auc_probe_pass", 0) or 0)) for r in executed),
            "candidate_auc_near_pass": near,
            "candidate_auc_near_pass_v1232": near_v1232,
            "candidate_auc_near_pass_v1233": near_v1233,
            "candidate_auc_official_pass_v1233": official_v1233,
            "max_logit_norm_p99": max((fnum(r.get("logit_norm_p99")) for r in executed if math.isfinite(fnum(r.get("logit_norm_p99")))), default=float("nan")),
            "min_unlabeled_entropy_p05": min((fnum(r.get("unlabeled_entropy_p05")) for r in executed if math.isfinite(fnum(r.get("unlabeled_entropy_p05")))), default=float("nan")),
            "min_unlabeled_top1_top2_gap_p01": min((fnum(r.get("unlabeled_top1_top2_gap_p01")) for r in executed if math.isfinite(fnum(r.get("unlabeled_top1_top2_gap_p01")))), default=float("nan")),
            "min_den_p01_batch": min((fnum(r.get("den_p01_batch")) for r in executed if math.isfinite(fnum(r.get("den_p01_batch")))), default=float("nan")),
            "max_den_condition_batch": max((fnum(r.get("den_condition_batch")) for r in executed if math.isfinite(fnum(r.get("den_condition_batch")))), default=float("nan")),
            "max_r_prime_p99": max((fnum(r.get("r_prime_p99")) for r in executed if math.isfinite(fnum(r.get("r_prime_p99")))), default=float("nan")),
            "max_r_double_prime_p99": max((fnum(r.get("r_double_prime_p99")) for r in executed if math.isfinite(fnum(r.get("r_double_prime_p99")))), default=float("nan")),
            "min_tangent_effective_rank": min((fnum(r.get("tangent_effective_rank")) for r in executed if math.isfinite(fnum(r.get("tangent_effective_rank")))), default=float("nan")),
            "max_tangent_top_eigen_share": max((fnum(r.get("tangent_top_eigen_share")) for r in executed if math.isfinite(fnum(r.get("tangent_top_eigen_share")))), default=float("nan")),
            "telemetry_available_rows": sum(int(float(r.get("telemetry_available", 0) or 0)) for r in executed),
            "promotion_allowed": 0,
            "no_fake": 1,
        })

    candidate_path = out_dir / f"{args.artifact_prefix}_candidate.csv"
    summary_path = out_dir / f"{args.artifact_prefix}_summary.csv"
    trajectory_path = out_dir / f"{args.artifact_prefix}_trajectory.csv"
    linec_path = out_dir / f"{args.artifact_prefix}_linec.csv"
    v1223.write_csv_rows(candidate_path, candidate_rows)
    v1223.write_csv_rows(summary_path, summary_rows)
    v1223.write_csv_rows(trajectory_path, trajectory_rows)
    v1223.write_csv_rows(linec_path, linec_rows)
    result = {
        "stage": f"{stage_prefix}_RATIONAL_AUC_ROUTE",
        "candidate_csv": v1223.rel(candidate_path),
        "summary_csv": v1223.rel(summary_path),
        "trajectory_csv": v1223.rel(trajectory_path),
        "linec_csv": v1223.rel(linec_path),
        "candidate_rows": len(candidate_rows),
        "summary_rows": len(summary_rows),
        "trajectory_rows": len(trajectory_rows),
        "linec_rows": len(linec_rows),
        "candidate_auc_near_pass_rows": sum(int(r.get("candidate_auc_near_pass", 0)) for r in summary_rows),
        "candidate_auc_near_pass_v1232_rows": sum(int(r.get("candidate_auc_near_pass_v1232", 0)) for r in summary_rows),
        "candidate_auc_near_pass_v1233_rows": sum(int(r.get("candidate_auc_near_pass_v1233", 0)) for r in summary_rows),
        "candidate_auc_official_pass_v1233_rows": sum(int(r.get("candidate_auc_official_pass_v1233", 0)) for r in summary_rows),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    route_path = out_dir / f"{args.artifact_prefix}_route.json"
    v1223.write_json(route_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
