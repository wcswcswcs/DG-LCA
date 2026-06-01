#!/usr/bin/env python
"""v12.30 loss-agnostic functional diagnostic on MLP controls.

The supervised task optimizer is ordinary AdamW and therefore uses training
labels.  The functional source candidates from dgkan.functional.mlp_functional use only precommit
model-state features from unlabeled train-stream inputs.  Label/CE gradients are
used only by the AdamWParallel control row and are marked as non-promotable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402
import experiments.run_v1252_efficiency_functional_manifold as v1252  # noqa: E402
from dgkan.diagnostics.classic_basis import fnum, parse_csv, parse_ints  # noqa: E402
from dgkan.functional.mlp_functional import (  # noqa: E402
    CONTROL_IDS,
    SOURCE_CANDIDATES,
    apply_delta,
    functional_objective,
    grad_delta_from_objective,
    hidden_forward,
    param_norm,
    random_delta_like,
    scale_delta,
)
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402
from dgkan.training.eval import classification_basic  # noqa: E402


def linec_metrics(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, xq: torch.Tensor, yq: torch.Tensor, seed: int, sketch_dim: int, lr: float, weight_decay: float) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        before_b = model(xb).detach()
        before_q = model(xq).detach()
    updated = v1252._take_adamw_window(model, xb, yb, float(lr), float(weight_decay)).eval()
    with torch.no_grad():
        after_b = updated(xb).detach()
        after_q = updated(xq).detach()
    r2, corr, resid, pred_norm = v1252._ridge_coupling(after_b - before_b, after_q - before_q, 1.0e-3)
    sig = v1252._signal_reservoir_metrics(updated, xb, yb, int(sketch_dim), int(seed))
    return {
        "CouplingR2": r2,
        "CouplingCorr": corr,
        "coupling_residual_norm": resid,
        "coupling_prediction_norm": pred_norm,
        "NoiseSignalLeak": sig["NoiseSignalLeak"],
        "RealSignalReservoirRatio": sig["RealSignalReservoirRatio"],
        "signal_effective_rank": sig["signal_effective_rank"],
        "reservoir_fraction": sig["reservoir_fraction"],
        "top_eigen_share": sig["top_eigen_share"],
    }


def train_branch(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    train_seed_base: int,
    candidate_id: str,
    branch_id: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    model = prim.MLPBaseline(input_dim, output_dim, int(args.hidden), int(seed) + int(train_seed_base), device).to(device)
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(train_seed_base) + 1230)
    times: list[float] = []
    func_times: list[float] = []
    functional_events = 0
    for epoch in range(int(args.epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off:off + int(args.batch_size)]
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        xb = x_train[: min(int(args.functional_batch), int(x_train.shape[0]))]
        yb = y_train[: xb.shape[0]]
        if branch_id == "C0-TaskOnlyAdamW":
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ft0 = time.perf_counter()
        objective = functional_objective(model, xb, candidate_id, anchor, int(seed) + int(train_seed_base) + epoch)
        source_delta = grad_delta_from_objective(model, objective)
        target_norm = max(1.0e-12, float(args.functional_norm_frac) * param_norm(model))
        source_delta = scale_delta(source_delta, target_norm)
        if branch_id == candidate_id:
            delta = source_delta
        elif branch_id == "C1-NoOpMatchedOverhead":
            delta = [torch.zeros_like(d) for d in source_delta]
        elif branch_id == "C2-RandomMatchedNorm":
            delta = random_delta_like(source_delta, target_norm, int(seed) + int(train_seed_base) + epoch + 991)
        elif branch_id == "C3-AdamWParallelDirection":
            ce = F.cross_entropy(model(xb), yb)
            delta = scale_delta(grad_delta_from_objective(model, ce), target_norm)
        elif branch_id == "C4-SNROnlyAudit":
            logits, _h1, h2 = hidden_forward(model, xb)
            probs = logits.softmax(dim=1)
            snr_obj = -h2.float().var(dim=0).mean() + 0.05 * probs.float().var(dim=0).mean()
            delta = scale_delta(grad_delta_from_objective(model, snr_obj), target_norm)
        else:
            raise KeyError(branch_id)
        apply_delta(model, delta)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        func_times.append((time.perf_counter() - ft0) * 1000.0)
        functional_events += 1
    ev = classification_basic(model, x_val, y_val)
    q90 = float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan")
    fq90 = float(torch.tensor(func_times, device=device).quantile(0.90).item()) if func_times else 0.0
    return {
        "candidate_id": candidate_id,
        "branch_id": branch_id,
        "dataset": dataset,
        "seed": int(seed),
        "train_seed_base": int(train_seed_base),
        "acc": ev["acc"],
        "NLL": ev["NLL"],
        "ECE": ev["ECE"],
        "CEp99": ev["CEp99"],
        "margin_p10": ev["margin_p10"],
        "step_time_q90_ms": q90,
        "functional_time_q90_ms": fq90,
        "functional_overhead_ratio": (q90 + fq90) / max(q90, 1.0e-12),
        "functional_events": functional_events,
        "window_epochs": int(args.epochs),
        "uses_label_for_direction": int(branch_id == "C3-AdamWParallelDirection"),
        "uses_ce_vector_for_direction": int(branch_id == "C3-AdamWParallelDirection"),
        "uses_validation_for_commit": 0,
        "uses_linec_hard_target_for_direction": 0,
        "uses_query_batch_for_commit": 0,
        "uses_optimizer_update_as_observable": 0,
        "loss_agnostic_direction": int(branch_id != "C3-AdamWParallelDirection"),
        "promotion_allowed": 0,
        "no_fake": 1,
        "_model": model,
    }


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v1230_mlp_functional")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-seed-bases", default="12300400,12301400,12302400")
    ap.add_argument("--candidates", default=",".join(SOURCE_CANDIDATES))
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--functional-batch", type=int, default=128)
    ap.add_argument("--functional-norm-frac", type=float, default=1.0e-4)
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="12309500,12310600,12311600,12312600,12313600")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    v1223.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.30 MLP functional diagnostic requires CUDA")
    torch.cuda.set_device(device)
    datasets = [v1223.v120._canonical_dataset(d) for d in parse_csv(args.datasets)]
    seeds = parse_ints(args.seeds)
    train_seed_bases = parse_ints(args.train_seed_bases)
    candidates = parse_csv(args.candidates)
    linec_seeds = parse_ints(args.linec_seeds)

    branch_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
            data = v1223.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)
            for train_seed_base in train_seed_bases:
                for cid in candidates:
                    branches = [cid, *CONTROL_IDS]
                    results: dict[str, dict[str, Any]] = {}
                    for branch in branches:
                        row = train_branch(args, dataset, int(seed), int(train_seed_base), cid, branch, x_train, y_train, x_val, y_val, int(input_dim), int(output_dim), device)
                        model = row.pop("_model")
                        results[branch] = row
                        branch_rows.append({"stage": "V1230_MLP_FUNCTIONAL_BRANCH", "architecture": "MLP", "functional_candidate_id": cid, **row})
                        if branch in {cid, "C1-NoOpMatchedOverhead"}:
                            b = min(int(args.linec_batch_size), int(x_train.shape[0]) // 2, int(x_val.shape[0]))
                            xb, yb = x_train[:b], y_train[:b]
                            xq, yq = x_val[:b], y_val[:b]
                            for ls in linec_seeds:
                                lm = linec_metrics(model, xb, yb, xq, yq, int(ls), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                                linec_rows.append({
                                    "stage": "V1230_MLP_FUNCTIONAL_LINEC",
                                    "architecture": "MLP",
                                    "functional_candidate_id": cid,
                                    "branch_id": branch,
                                    "dataset": dataset,
                                    "seed": int(seed),
                                    "train_seed_base": int(train_seed_base),
                                    "window_epochs": int(args.epochs),
                                    "linec_seed": int(ls),
                                    **lm,
                                    "label_used_for_audit_only": 1,
                                    "uses_label_for_direction": int(branch == "C3-AdamWParallelDirection"),
                                    "uses_ce_vector_for_direction": int(branch == "C3-AdamWParallelDirection"),
                                    "uses_linec_hard_target_for_direction": 0,
                                    "promotion_allowed": 0,
                                    "no_fake": 1,
                                })
                        del model
                        torch.cuda.empty_cache()
                    source = results[cid]
                    noop = results["C1-NoOpMatchedOverhead"]
                    control_ids = ["C0-TaskOnlyAdamW", "C2-RandomMatchedNorm", "C3-AdamWParallelDirection", "C4-SNROnlyAudit"]
                    best_control = max((results[c] for c in control_ids), key=lambda r: fnum(r.get("acc"), -999.0))
                    source_linec = [
                        r for r in linec_rows
                        if r.get("functional_candidate_id") == cid
                        and r.get("branch_id") == cid
                        and r.get("dataset") == dataset
                        and int(r.get("seed")) == int(seed)
                        and int(r.get("train_seed_base")) == int(train_seed_base)
                    ]
                    noop_linec = [
                        r for r in linec_rows
                        if r.get("functional_candidate_id") == cid
                        and r.get("branch_id") == "C1-NoOpMatchedOverhead"
                        and r.get("dataset") == dataset
                        and int(r.get("seed")) == int(seed)
                        and int(r.get("train_seed_base")) == int(train_seed_base)
                    ]
                    linec_pass_count = 0
                    for sr, nr in zip(source_linec, noop_linec):
                        linec_pass_count += int(
                            fnum(sr.get("CouplingR2"), -999.0) >= fnum(nr.get("CouplingR2"), 0.0) - 0.02
                            and fnum(sr.get("NoiseSignalLeak"), 999.0) <= fnum(nr.get("NoiseSignalLeak"), 0.0) + 0.02
                            and fnum(sr.get("RealSignalReservoirRatio"), 999.0) <= fnum(nr.get("RealSignalReservoirRatio"), 0.0) + 0.03
                        )
                    source_vs_noop = fnum(source.get("acc"), -999.0) - fnum(noop.get("acc"), -999.0)
                    source_vs_control = fnum(source.get("acc"), -999.0) - fnum(best_control.get("acc"), -999.0)
                    coupling_delta = (sum(fnum(r.get("CouplingR2"), 0.0) for r in source_linec) - sum(fnum(r.get("CouplingR2"), 0.0) for r in noop_linec)) / max(1, min(len(source_linec), len(noop_linec)))
                    noise_delta = (sum(fnum(r.get("NoiseSignalLeak"), 0.0) for r in source_linec) - sum(fnum(r.get("NoiseSignalLeak"), 0.0) for r in noop_linec)) / max(1, min(len(source_linec), len(noop_linec)))
                    reservoir_delta = (sum(fnum(r.get("RealSignalReservoirRatio"), 0.0) for r in source_linec) - sum(fnum(r.get("RealSignalReservoirRatio"), 0.0) for r in noop_linec)) / max(1, min(len(source_linec), len(noop_linec)))
                    exploration = int(
                        source_vs_noop >= 0.0
                        and source_vs_control >= -0.003
                        and coupling_delta >= 0.01
                        and fnum(source.get("CEp99"), 999.0) <= fnum(noop.get("CEp99"), 0.0) + 0.50
                        and fnum(source.get("NLL"), 999.0) <= fnum(noop.get("NLL"), 0.0) + 0.05
                        and fnum(source.get("ECE"), 999.0) <= fnum(noop.get("ECE"), 0.0) + 0.03
                        and linec_pass_count >= math.ceil(len(linec_seeds) / 2.0)
                        and fnum(source.get("functional_overhead_ratio"), 999.0) <= 1.05
                    )
                    official = int(
                        source_vs_control >= 0.005
                        and coupling_delta >= 0.02
                        and noise_delta <= 0.0
                        and reservoir_delta <= 0.0
                        and linec_pass_count >= math.ceil(len(linec_seeds) / 2.0)
                        and fnum(source.get("CEp99"), 999.0) <= fnum(noop.get("CEp99"), 0.0)
                        and fnum(source.get("NLL"), 999.0) <= fnum(noop.get("NLL"), 0.0)
                        and fnum(source.get("ECE"), 999.0) <= fnum(noop.get("ECE"), 0.0)
                    )
                    candidate_rows.append({
                        "stage": "V1230_MLP_FUNCTIONAL_CANDIDATE",
                        "architecture": "MLP",
                        "functional_candidate_id": cid,
                        "functional_description": SOURCE_CANDIDATES.get(cid, ""),
                        "dataset": dataset,
                        "seed": int(seed),
                        "train_seed_base": int(train_seed_base),
                        "window_epochs": int(args.epochs),
                        "source_vs_noop_acc_delta": source_vs_noop,
                        "source_vs_control_acc_delta": source_vs_control,
                        "best_control_id": best_control.get("branch_id", ""),
                        "NLL_delta_vs_noop": fnum(source.get("NLL"), 999.0) - fnum(noop.get("NLL"), 999.0),
                        "ECE_delta_vs_noop": fnum(source.get("ECE"), 999.0) - fnum(noop.get("ECE"), 999.0),
                        "CEp99_delta_vs_noop": fnum(source.get("CEp99"), 999.0) - fnum(noop.get("CEp99"), 999.0),
                        "CouplingR2_delta": coupling_delta,
                        "NoiseSignalLeak_delta": noise_delta,
                        "RealSignalReservoirRatio_delta": reservoir_delta,
                        "LineC_pass_count": linec_pass_count,
                        "LineC_seed_count": len(linec_seeds),
                        "functional_overhead_ratio": source.get("functional_overhead_ratio", ""),
                        "exploration_gate_pass": exploration,
                        "official_gate_pass": official,
                        "uses_label_for_direction": 0,
                        "uses_ce_vector_for_direction": 0,
                        "uses_validation_for_commit": 0,
                        "uses_linec_hard_target_for_direction": 0,
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    })
    cand_csv = out_dir / f"{args.artifact_prefix}_candidates.csv"
    ctrl_csv = out_dir / f"{args.artifact_prefix}_controls.csv"
    linec_csv = out_dir / f"{args.artifact_prefix}_linec.csv"
    gate_csv = out_dir / f"{args.artifact_prefix}_gate.csv"
    v1223.write_csv_rows(cand_csv, candidate_rows)
    v1223.write_csv_rows(ctrl_csv, branch_rows)
    v1223.write_csv_rows(linec_csv, linec_rows)
    by_cand: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_cand.setdefault(str(row["functional_candidate_id"]), []).append(row)
    gate_rows = []
    for cid, group in sorted(by_cand.items()):
        gate_rows.append({
            "stage": "V1230_MLP_FUNCTIONAL_GATE",
            "functional_candidate_id": cid,
            "window_epochs": int(args.epochs),
            "rows": len(group),
            "exploration_pass_rows": sum(int(r.get("exploration_gate_pass", 0)) for r in group),
            "official_pass_rows": sum(int(r.get("official_gate_pass", 0)) for r in group),
            "mean_source_vs_noop": sum(fnum(r.get("source_vs_noop_acc_delta"), 0.0) for r in group) / max(1, len(group)),
            "mean_source_vs_control": sum(fnum(r.get("source_vs_control_acc_delta"), 0.0) for r in group) / max(1, len(group)),
            "mean_CouplingR2_delta": sum(fnum(r.get("CouplingR2_delta"), 0.0) for r in group) / max(1, len(group)),
            "max_LineC_pass_count": max(int(r.get("LineC_pass_count", 0)) for r in group),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    v1223.write_csv_rows(gate_csv, gate_rows)
    result = {
        "stage": "V1230_MLP_FUNCTIONAL_AGGREGATE",
        "candidate_csv": v1223.rel(cand_csv),
        "control_csv": v1223.rel(ctrl_csv),
        "linec_csv": v1223.rel(linec_csv),
        "gate_csv": v1223.rel(gate_csv),
        "candidate_rows": len(candidate_rows),
        "control_rows": len(branch_rows),
        "linec_rows": len(linec_rows),
        "gate_rows": len(gate_rows),
        "any_exploration_pass": int(any(int(r.get("exploration_gate_pass", 0)) for r in candidate_rows)),
        "any_official_pass": int(any(int(r.get("official_gate_pass", 0)) for r in candidate_rows)),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    json_path = out_dir / f"{args.artifact_prefix}_aggregate.json"
    v1223.write_json(json_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
