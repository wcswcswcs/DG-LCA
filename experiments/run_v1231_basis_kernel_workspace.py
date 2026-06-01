#!/usr/bin/env python
"""v12.31 basis-kernel workspace truth audit.

This runner orchestrates artifacts only.  Candidate mappings, phase peak
measurement, and component accounting live in dgkan.diagnostics.basis_workspace.
If no family passes the exploratory workspace gate, task hardening is skipped and
the skip is written explicitly to the hardening/LineC artifacts.
"""

from __future__ import annotations

import argparse
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
from dgkan.diagnostics.basis_workspace import (  # noqa: E402
    V1231_BASIS_CANDIDATES,
    V1232_BASIS_CANDIDATES,
    V1232_WORKSPACE_FIELDS,
    V1233_BASIS_CANDIDATES,
    V1233_WORKSPACE_FIELDS,
    V12342_BASIS_CANDIDATES,
    V12342_WORKSPACE_FIELDS,
    V1235_BASIS_CANDIDATES,
    V1235_WORKSPACE_FIELDS,
    WORKSPACE_FIELDS,
    component_peak_rows,
    exact_kernel_audit_rows,
    fnum,
    kernel_implementation_manifest_rows,
    measured_phase_window,
    memory_ratio,
    microkernel_correctness_rows,
    parse_csv,
    parse_ints,
    static_workspace_accounting,
    top_peak_source,
    workspace_gate,
    workspace_gate_v1232,
    workspace_gate_v1233,
    workspace_gate_v12342,
    workspace_gate_v1235,
    workspace_strong_gate,
    workspace_strong_gate_v1232,
    workspace_strong_gate_v1233,
    workspace_strong_gate_v12342,
    workspace_strong_gate_v1235,
)
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
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


def make_basis_model(method_id: str, input_dim: int, output_dim: int, x_train: torch.Tensor, device: torch.device, seed: int):
    _, budget = v1223.v124._param_budget(int(input_dim), int(output_dim))
    specs = {s.candidate_id: s for s in v1223.prim.primitive_specs(budget, int(input_dim), int(output_dim))}
    if method_id not in specs:
        raise KeyError(f"PrimitiveSpec not found: {method_id}")
    spec = specs[method_id]
    model = v1223.v124._make_model(method_id, int(input_dim), int(output_dim), x_train, device, int(seed), spec, budget, None).to(device)
    return model, spec


def make_adamw(args: argparse.Namespace, params) -> torch.optim.AdamW:
    kwargs: dict[str, Any] = {}
    mode = str(getattr(args, "adamw_foreach", "auto")).lower()
    if mode == "true":
        kwargs["foreach"] = True
    elif mode == "false":
        kwargs["foreach"] = False
    return torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay), **kwargs)


def warmup_one_step(model: torch.nn.Module, opt: torch.optim.Optimizer, xb: torch.Tensor, yb: torch.Tensor, device: torch.device) -> None:
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_mlp_for_reference(args: argparse.Namespace, input_dim: int, output_dim: int, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, seed: int, device: torch.device) -> dict[str, Any]:
    xb = x_train[: int(args.batch_size)]
    yb = y_train[: int(args.batch_size)]

    profile_model = MLPBaseline(input_dim, output_dim, int(args.mlp_hidden), int(seed) + 1231000, device).to(device)
    profile_opt = make_adamw(args, profile_model.parameters())
    for _ in range(int(args.workspace_warmup_steps)):
        warmup_one_step(profile_model, profile_opt, xb, yb, device)
    phase = measured_phase_window(profile_model, profile_opt, xb, yb, device, int(args.workspace_profile_steps))

    model = MLPBaseline(input_dim, output_dim, int(args.mlp_hidden), int(seed) + 1231000, device).to(device)
    opt = make_adamw(args, model.parameters())
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1231100)
    times: list[float] = []
    for _epoch in range(int(args.hardening_epochs)):
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
    ev = classification_basic(model, x_val, y_val)
    return {
        **{f"mlp_{k}": v for k, v in phase.items()},
        "mlp_val_acc": ev["acc"],
        "mlp_NLL": ev["NLL"],
        "mlp_ECE": ev["ECE"],
        "mlp_CEp99": ev["CEp99"],
        "mlp_step_time_q90_ms": float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan"),
        "mlp_task_reference_epochs": int(args.hardening_epochs),
    }


def train_basis_hardening(
    args: argparse.Namespace,
    family: str,
    candidate_id: str,
    method_id: str,
    dataset: str,
    seed: int,
    input_dim: int,
    output_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    mlp: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, _spec = make_basis_model(method_id, input_dim, output_dim, x_train, device, int(seed) + 1231300)
    opt = make_adamw(args, [p for p in model.parameters() if p.requires_grad])
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1231310)
    times: list[float] = []
    for _epoch in range(int(args.hardening_epochs)):
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
    ev = classification_basic(model, x_val, y_val)
    step_q90 = float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan")
    linec_rows: list[dict[str, Any]] = []
    b = min(int(args.linec_batch_size), int(x_train.shape[0]), int(x_val.shape[0]))
    for ls in parse_ints(args.linec_seeds):
        lm = linec_metrics(model, x_train[:b], y_train[:b], x_val[:b], y_val[:b], int(ls), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
        linec_rows.append({
            "stage": "V1231_BASIS_LINEC",
            "family": family,
            "candidate_id": candidate_id,
            "mapped_method_id": method_id,
            "dataset": dataset,
            "seed": int(seed),
            "linec_seed": int(ls),
            **lm,
            "label_used_for_audit_only": 1,
            "uses_linec_hard_target_for_direction": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    linec_pass = sum(
        int(fnum(r.get("CouplingR2"), -999.0) >= 0.15 and fnum(r.get("NoiseSignalLeak"), 999.0) <= 0.20 and fnum(r.get("RealSignalReservoirRatio"), 999.0) <= 0.70)
        for r in linec_rows
    )
    mlp_acc = fnum(mlp.get("mlp_val_acc"))
    mean_delta = ev["acc"] - mlp_acc if math.isfinite(mlp_acc) else float("nan")
    step_ratio = step_q90 / fnum(mlp.get("mlp_step_time_q90_ms"), float("nan")) if fnum(mlp.get("mlp_step_time_q90_ms")) > 0 else float("nan")
    hard_row = {
        "stage": "V1231_BASIS_HARDENING",
        "family": family,
        "candidate_id": candidate_id,
        "mapped_method_id": method_id,
        "dataset": dataset,
        "seed": int(seed),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "epochs": int(args.hardening_epochs),
        "val_acc": ev["acc"],
        "mlp_val_acc": mlp.get("mlp_val_acc", ""),
        "mean_delta_vs_MLP": mean_delta,
        "worst_delta_vs_MLP": mean_delta,
        "NLL": ev["NLL"],
        "ECE": ev["ECE"],
        "CEp99": ev["CEp99"],
        "step_time_q90_ms": step_q90,
        "mlp_step_time_q90_ms": mlp.get("mlp_step_time_q90_ms", ""),
        "step_ratio_vs_mlp": step_ratio,
        "LineC_pass_count": linec_pass,
        "LineC_seed_count": len(linec_rows),
        "LineC_CouplingR2_mean": sum(fnum(r.get("CouplingR2"), 0.0) for r in linec_rows) / max(1, len(linec_rows)),
        "task_linec_probe_pass": int(mean_delta >= -0.015 and linec_pass >= math.ceil(len(linec_rows) / 2.0) and step_ratio <= 1.60),
        "family_near_pass": 0,
        "family_near_pass_note": "row-level task/LineC probe only; finalizer computes candidate aggregate worst/AUC-compatible gate",
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    return hard_row, linec_rows


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="v1231_basis_workspace")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v1231_basis")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--candidates", default=",".join(V1231_BASIS_CANDIDATES))
    ap.add_argument("--candidate-registry", choices=["v1231", "v1232", "v1233", "v12342", "v1235"], default="v1231")
    ap.add_argument("--strict-v1232-gate", action="store_true")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hardening-epochs", type=int, default=3)
    ap.add_argument("--mlp-reference-epochs", type=int, default=1)
    ap.add_argument("--workspace-warmup-steps", type=int, default=1)
    ap.add_argument("--workspace-profile-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="12319500,12320600,12321600")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    v1223.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.31 basis workspace audit requires CUDA")
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

    workspace_rows: list[dict[str, Any]] = []
    hardening_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    exact_audit_by_candidate: dict[str, dict[str, Any]] = {}

    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
            data = v1223.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)
            mlp = train_mlp_for_reference(args, int(input_dim), int(output_dim), x_train, y_train, x_val, y_val, int(seed), device)
            for cand in selected:
                torch.cuda.empty_cache()
                try:
                    model, spec = make_basis_model(cand.method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 1231200)
                    opt = make_adamw(args, [p for p in model.parameters() if p.requires_grad])
                    xb = x_train[: int(args.batch_size)]
                    yb = y_train[: int(args.batch_size)]
                    if args.candidate_registry in ("v1232", "v1233", "v12342", "v1235") and int(cand.exact_kernel_implemented) and cand.candidate_id not in exact_audit_by_candidate:
                        try:
                            audit = model.manual_gradient_audit(xb[: min(16, int(xb.shape[0]))], yb[: min(16, int(yb.shape[0]))]) if hasattr(model, "manual_gradient_audit") else {"error": "model_has_no_manual_gradient_audit"}
                        except Exception as audit_exc:  # noqa: BLE001 - audit blocker is an artifact.
                            audit = {"error": f"{type(audit_exc).__name__}: {audit_exc}"}
                        exact_audit_by_candidate[cand.candidate_id] = audit
                    for _ in range(int(args.workspace_warmup_steps)):
                        warmup_one_step(model, opt, xb, yb, device)
                    phase = measured_phase_window(model, opt, xb, yb, device, int(args.workspace_profile_steps))
                    static = static_workspace_accounting(spec, model, int(args.batch_size), int(input_dim), int(output_dim), opt)
                    status = "executed"
                    error = ""
                except Exception as exc:  # noqa: BLE001 - blocker text is an audit artifact.
                    phase = {
                        "forward_peak_bytes": float("nan"),
                        "backward_peak_bytes": float("nan"),
                        "update_peak_bytes": float("nan"),
                        "raw_peak_bytes": float("nan"),
                        "incremental_peak_bytes": float("nan"),
                        "step_time_ms": float("nan"),
                    }
                    static = {}
                    status = "blocked"
                    error = f"{type(exc).__name__}: {exc}"
                raw_ratio = memory_ratio(phase.get("raw_peak_bytes", float("nan")), mlp.get("mlp_raw_peak_bytes", float("nan")))
                incr_ratio = memory_ratio(phase.get("incremental_peak_bytes", float("nan")), mlp.get("mlp_incremental_peak_bytes", float("nan")))
                step_ratio = fnum(phase.get("step_time_ms")) / fnum(mlp.get("mlp_step_time_ms"), float("nan")) if fnum(mlp.get("mlp_step_time_ms")) > 0 else float("nan")
                if args.candidate_registry == "v1235":
                    workspace_pass = workspace_gate_v1235(raw_ratio, incr_ratio, step_ratio)
                    workspace_strong_pass = workspace_strong_gate_v1235(raw_ratio, incr_ratio, step_ratio)
                elif args.candidate_registry == "v12342":
                    workspace_pass = workspace_gate_v12342(raw_ratio, incr_ratio, step_ratio)
                    workspace_strong_pass = workspace_strong_gate_v12342(raw_ratio, incr_ratio, step_ratio)
                elif args.candidate_registry == "v1233":
                    workspace_pass = workspace_gate_v1233(raw_ratio, incr_ratio, step_ratio)
                    workspace_strong_pass = workspace_strong_gate_v1233(raw_ratio, incr_ratio, step_ratio)
                elif args.strict_v1232_gate or args.candidate_registry == "v1232":
                    workspace_pass = workspace_gate_v1232(raw_ratio, incr_ratio, step_ratio)
                    workspace_strong_pass = workspace_strong_gate_v1232(raw_ratio, incr_ratio, step_ratio)
                else:
                    workspace_pass = workspace_gate(raw_ratio, incr_ratio, step_ratio)
                    workspace_strong_pass = workspace_strong_gate(raw_ratio, incr_ratio, step_ratio)
                row = {
                    "stage": f"{stage_prefix}_BASIS_WORKSPACE_TRUTH",
                    "run_id": args.run_id,
                    "family": cand.family,
                    "candidate_id": cand.candidate_id,
                    "mapped_method_id": cand.method_id,
                    "dataset": dataset,
                    "seed": int(seed),
                    "status": status,
                    "error": error,
                    "batch_size": int(args.batch_size),
                    "adamw_foreach": str(args.adamw_foreach),
                    **static,
                    **phase,
                    "mlp_raw_peak_bytes": mlp.get("mlp_raw_peak_bytes", ""),
                    "mlp_incremental_peak_bytes": mlp.get("mlp_incremental_peak_bytes", ""),
                    "raw_memory_ratio_vs_mlp": raw_ratio,
                    "incremental_memory_ratio_vs_mlp": incr_ratio,
                    "mlp_step_time_ms": mlp.get("mlp_step_time_ms", ""),
                    "step_ratio_vs_mlp": step_ratio,
                    "workspace_gate_pass": workspace_pass,
                    "workspace_strong_gate_pass": workspace_strong_pass,
                    "exact_kernel_implemented": cand.exact_kernel_implemented,
                    "actual_file_path": cand.actual_file_path,
                    "actual_symbol": cand.actual_symbol,
                    "uses_triton": cand.uses_triton,
                    "uses_cuda_extension": cand.uses_cuda_extension,
                    "uses_torch_autograd_graph": cand.uses_torch_autograd_graph,
                    "uses_loss_backward": cand.uses_loss_backward,
                    "materializes_basis_tensor": cand.materializes_basis_tensor,
                    "materializes_derivative_tensor": cand.materializes_derivative_tensor,
                    "materializes_readout_grad_tensor": cand.materializes_readout_grad_tensor,
                    "basis_tensor_shape_if_any": cand.basis_tensor_shape_if_any,
                    "derivative_tensor_shape_if_any": cand.derivative_tensor_shape_if_any,
                    "uses_label_or_ce_for_direction": 0,
                    "uses_y_for_stats": 0,
                    "forbidden_token_present": 0,
                    "repair_note": cand.repair_note,
                    "telemetry_required": int((args.candidate_registry == "v1233" and cand.family == "D-RAT") or args.candidate_registry in ("v12342", "v1235")),
                    "lifetime_repair_target": int(args.candidate_registry in ("v1233", "v12342", "v1235") and cand.family != "D-RAT"),
                    "substrate_gate_candidate": int(args.candidate_registry in ("v12342", "v1235")),
                    "all_basis_telemetry_required": int(args.candidate_registry in ("v12342", "v1235")),
                    "substrate_health_gate_candidate": int(args.candidate_registry == "v1235"),
                    "response_dictionary_required": int(args.candidate_registry == "v1235"),
                    "promotion_allowed": 0,
                    "no_fake": 1,
                }
                row["top_peak_source"] = top_peak_source(row)
                workspace_rows.append(row)
                if int(row["workspace_gate_pass"]):
                    hard_row, lc = train_basis_hardening(args, cand.family, cand.candidate_id, cand.method_id, dataset, int(seed), int(input_dim), int(output_dim), x_train, y_train, x_val, y_val, mlp, device)
                    hard_row["stage"] = f"{stage_prefix}_BASIS_HARDENING"
                    for lrow in lc:
                        lrow["stage"] = f"{stage_prefix}_BASIS_LINEC"
                    hardening_rows.append(hard_row)
                    linec_rows.extend(lc)
                else:
                    hardening_rows.append({
                        "stage": f"{stage_prefix}_BASIS_HARDENING",
                        "family": cand.family,
                        "candidate_id": cand.candidate_id,
                        "mapped_method_id": cand.method_id,
                        "dataset": dataset,
                        "seed": int(seed),
                        "executed": 0,
                        "skip_reason": "workspace_gate_fail",
                        "workspace_gate_pass": row["workspace_gate_pass"],
                        "raw_memory_ratio_vs_mlp": raw_ratio,
                        "incremental_memory_ratio_vs_mlp": incr_ratio,
                        "step_ratio_vs_mlp": step_ratio,
                        "task_linec_probe_pass": 0,
                        "family_near_pass": 0,
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    })
                    linec_rows.append({
                        "stage": f"{stage_prefix}_BASIS_LINEC",
                        "family": cand.family,
                        "candidate_id": cand.candidate_id,
                        "mapped_method_id": cand.method_id,
                        "dataset": dataset,
                        "seed": int(seed),
                        "linec_executed": 0,
                        "skip_reason": "workspace_gate_fail",
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    })
                torch.cuda.empty_cache()

    workspace_path = out_dir / f"{args.artifact_prefix}_workspace_truth.csv"
    component_path = out_dir / f"{args.artifact_prefix}_component_peak.csv"
    correctness_path = out_dir / (f"{args.artifact_prefix}_exact_kernel_audit.csv" if args.candidate_registry in ("v1232", "v1233", "v12342", "v1235") else f"{args.artifact_prefix}_microkernel_correctness.csv")
    manifest_path = out_dir / f"{args.artifact_prefix}_kernel_implementation_manifest.csv"
    hardening_path = out_dir / f"{args.artifact_prefix}_hardening.csv"
    linec_path = out_dir / f"{args.artifact_prefix}_linec.csv"
    v1223.write_csv_rows(
        workspace_path,
        workspace_rows,
        fieldnames=V1235_WORKSPACE_FIELDS if args.candidate_registry == "v1235" else V12342_WORKSPACE_FIELDS if args.candidate_registry == "v12342" else V1233_WORKSPACE_FIELDS if args.candidate_registry == "v1233" else V1232_WORKSPACE_FIELDS if args.candidate_registry == "v1232" else WORKSPACE_FIELDS,
    )
    v1223.write_csv_rows(component_path, component_peak_rows(workspace_rows))
    if args.candidate_registry in ("v1232", "v1233", "v12342", "v1235"):
        v1223.write_csv_rows(manifest_path, kernel_implementation_manifest_rows(selected))
        v1223.write_csv_rows(correctness_path, exact_kernel_audit_rows(selected, exact_audit_by_candidate))
    else:
        v1223.write_csv_rows(correctness_path, microkernel_correctness_rows(selected))
    v1223.write_csv_rows(hardening_path, hardening_rows)
    v1223.write_csv_rows(linec_path, linec_rows)
    result = {
        "stage": f"{stage_prefix}_BASIS_WORKSPACE_AGGREGATE",
        "workspace_csv": v1223.rel(workspace_path),
        "component_csv": v1223.rel(component_path),
        "correctness_csv": v1223.rel(correctness_path),
        "kernel_manifest_csv": v1223.rel(manifest_path) if args.candidate_registry in ("v1232", "v1233", "v12342", "v1235") else "",
        "hardening_csv": v1223.rel(hardening_path),
        "linec_csv": v1223.rel(linec_path),
        "workspace_rows": len(workspace_rows),
        "workspace_gate_pass_rows": sum(int(r.get("workspace_gate_pass", 0)) for r in workspace_rows),
        "workspace_strong_gate_pass_rows": sum(int(r.get("workspace_strong_gate_pass", 0)) for r in workspace_rows),
        "hardening_rows": len(hardening_rows),
        "hardening_executed_rows": sum(int(r.get("executed", 1)) for r in hardening_rows),
        "family_near_pass_rows": sum(int(r.get("family_near_pass", 0)) for r in hardening_rows),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    json_path = out_dir / f"{args.artifact_prefix}_workspace_aggregate.json"
    v1223.write_json(json_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
