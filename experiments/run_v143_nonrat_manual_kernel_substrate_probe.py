#!/usr/bin/env python
"""v14.3 Non-RAT exact/manual-kernel substrate probe.

This is a substrate-only diagnostic for the R4-NonRATSubstrateMissing branch.
It does not run FMS proof and does not use LineC/tail metrics as direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES, fnum, memory_ratio, parse_csv, parse_ints, top_peak_source  # noqa: E402
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments import run_v124_multibasis_functional_dual as v124  # noqa: E402
from experiments.run_v1231_basis_kernel_workspace import make_adamw, make_basis_model  # noqa: E402
from experiments.run_v132_loss_interface_basis_natural_update import synthetic_data  # noqa: E402


DEFAULT_CANDIDATES = (
    "D-CHE13-FusedReadoutGradNoMaterialize-K3,"
    "D-CHE15-FullStepNoMaterialize-K3,"
    "D-FOU12-LifetimeRecomputeBackward-K2,"
    "D-FOU13-FusedReadoutGradNoMaterialize-K2,"
    "D-FOU14-SincosSharedWorkspace-K2,"
    "D-FOU15-FullStepNoMaterialize-K2"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cuda_peak() -> float:
    return float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else float("nan")


def standard_step(model: torch.nn.Module, opt: torch.optim.Optimizer, xb: torch.Tensor, yb: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = float("nan")
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    loss = F.cross_entropy(model(xb), yb)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_peak = cuda_peak()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_peak = cuda_peak()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    update_peak = cuda_peak()
    raw_peak = max(forward_peak, backward_peak, update_peak)
    return {
        "memory_baseline_bytes": baseline,
        "forward_peak_bytes": forward_peak,
        "backward_peak_bytes": backward_peak,
        "update_peak_bytes": update_peak,
        "raw_peak_bytes": raw_peak,
        "incremental_peak_bytes": max(0.0, raw_peak - baseline) if math.isfinite(raw_peak) and math.isfinite(baseline) else float("nan"),
        "step_time_ms": (time.perf_counter() - t0) * 1000.0,
    }


def manual_step(model: torch.nn.Module, opt: torch.optim.Optimizer, xb: torch.Tensor, yb: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.train()
    if not hasattr(model, "manual_ce_forward_cache") or not hasattr(model, "manual_ce_backward_from_cache"):
        raise RuntimeError("model has no manual CE backward path")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = float("nan")
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    logits, cache = model.manual_ce_forward_cache(xb)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_peak = cuda_peak()
    _loss = model.manual_ce_backward_from_cache(logits, cache, yb)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_peak = cuda_peak()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    update_peak = cuda_peak()
    raw_peak = max(forward_peak, backward_peak, update_peak)
    return {
        "memory_baseline_bytes": baseline,
        "forward_peak_bytes": forward_peak,
        "backward_peak_bytes": backward_peak,
        "update_peak_bytes": update_peak,
        "raw_peak_bytes": raw_peak,
        "incremental_peak_bytes": max(0.0, raw_peak - baseline) if math.isfinite(raw_peak) and math.isfinite(baseline) else float("nan"),
        "step_time_ms": (time.perf_counter() - t0) * 1000.0,
    }


def window(measure_fn, model: torch.nn.Module, opt: torch.optim.Optimizer, xb: torch.Tensor, yb: torch.Tensor, device: torch.device, steps: int) -> dict[str, float]:
    samples = [measure_fn(model, opt, xb, yb, device) for _ in range(max(1, int(steps)))]
    out: dict[str, float] = {}
    for key in ["memory_baseline_bytes", "forward_peak_bytes", "backward_peak_bytes", "update_peak_bytes", "raw_peak_bytes", "incremental_peak_bytes"]:
        vals = [fnum(s.get(key)) for s in samples if math.isfinite(fnum(s.get(key)))]
        out[key] = max(vals) if vals else float("nan")
    times = [fnum(s.get("step_time_ms")) for s in samples if math.isfinite(fnum(s.get("step_time_ms")))]
    out["step_time_ms"] = float(torch.tensor(times).quantile(0.90).item()) if times else float("nan")
    return out


def make_probe_model(candidate_id: str, input_dim: int, output_dim: int, x_train: torch.Tensor, device: torch.device, seed: int, hidden_override: int):
    cand = V1235_BASIS_CANDIDATES[candidate_id]
    model, spec = make_basis_model(cand.method_id, input_dim, output_dim, x_train, device, seed)
    if int(hidden_override) <= 0:
        return model, spec
    compact_spec = replace(spec, hidden_dim=int(hidden_override), source=f"{spec.source}+v14.3_capacity_reduced_substrate_probe")
    _, budget = v124._param_budget(int(input_dim), int(output_dim))
    compact_model = v124._make_model(cand.method_id, int(input_dim), int(output_dim), x_train, device, int(seed), compact_spec, budget, None).to(device)
    return compact_model, compact_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--synthetic-task", default="X1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=128)
    ap.add_argument("--val-size", type=int, default=64)
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--profile-steps", type=int, default=1)
    ap.add_argument("--workspace-warmup-steps", type=int, default=1)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--hidden-override", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="false")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("manual kernel substrate probe requires CUDA")
    torch.cuda.set_device(device)

    xtr, ytr, _xva, _yva = synthetic_data(args.synthetic_task, int(args.seed), int(args.train_size), int(args.val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    xb = xtr[: int(args.batch_size)]
    yb = ytr[: int(args.batch_size)]
    opt_args = argparse.Namespace(lr=float(args.lr), weight_decay=float(args.weight_decay), adamw_foreach=str(args.adamw_foreach))

    mlp = MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), int(args.seed) + 14_300, device).to(device)
    mlp_opt = make_adamw(opt_args, mlp.parameters())
    for _ in range(int(args.workspace_warmup_steps)):
        standard_step(mlp, mlp_opt, xb, yb, device)
    mlp_phase = window(standard_step, mlp, mlp_opt, xb, yb, device, int(args.profile_steps))

    rows: list[dict[str, Any]] = []
    for candidate_id in parse_csv(args.candidates):
        cand = V1235_BASIS_CANDIDATES[candidate_id]
        for mode in ["standard_autograd", "manual_no_materialize"]:
            try:
                model, spec = make_probe_model(cand.candidate_id, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(args.seed) + 14_310, int(args.hidden_override))
                opt = make_adamw(opt_args, [p for p in model.parameters() if p.requires_grad])
                measure = manual_step if mode == "manual_no_materialize" else standard_step
                for _ in range(int(args.workspace_warmup_steps)):
                    measure(model, opt, xb, yb, device)
                phase = window(measure, model, opt, xb, yb, device, int(args.profile_steps))
                status = "executed"
                error = ""
            except Exception as exc:  # noqa: BLE001
                spec = None
                phase = {
                    "memory_baseline_bytes": float("nan"),
                    "forward_peak_bytes": float("nan"),
                    "backward_peak_bytes": float("nan"),
                    "update_peak_bytes": float("nan"),
                    "raw_peak_bytes": float("nan"),
                    "incremental_peak_bytes": float("nan"),
                    "step_time_ms": float("nan"),
                }
                status = "blocked"
                error = f"{type(exc).__name__}: {exc}"
            raw_ratio = memory_ratio(phase["raw_peak_bytes"], mlp_phase["raw_peak_bytes"])
            inc_ratio = memory_ratio(phase["incremental_peak_bytes"], mlp_phase["incremental_peak_bytes"])
            step_ratio = phase["step_time_ms"] / mlp_phase["step_time_ms"] if fnum(mlp_phase["step_time_ms"]) > 0 else float("nan")
            row = {
                "stage": "V143_NONRAT_MANUAL_KERNEL_SUBSTRATE_PROBE",
                "family": cand.family,
                "candidate_id": cand.candidate_id,
                "mapped_method_id": cand.method_id,
                "probe_mode": mode,
                "status": status,
                "error": error,
                "synthetic_task": args.synthetic_task,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "mlp_hidden": int(args.mlp_hidden),
                "basis_name": getattr(spec, "basis_name", "") if spec is not None else "",
                "init_variant": getattr(spec, "init_variant", "") if spec is not None else "",
                "spec_hidden_dim": int(getattr(spec, "hidden_dim", 0)) if spec is not None else 0,
                "hidden_override": int(args.hidden_override),
                "capacity_reduced_substrate_probe": int(int(args.hidden_override) > 0),
                "manual_backward_available": int(hasattr(model, "manual_ce_backward_from_cache")) if status == "executed" else 0,
                **{f"mlp_{k}": v for k, v in mlp_phase.items()},
                **phase,
                "raw_memory_ratio_vs_mlp": raw_ratio,
                "incremental_memory_ratio_vs_mlp": inc_ratio,
                "step_ratio_vs_mlp": step_ratio,
                "manual_workspace_gate_pass": int(raw_ratio <= 1.75 and inc_ratio <= 1.75 and step_ratio <= 1.75),
                "v143_official_fms_proof_executed": 0,
                "promotion_allowed": 0,
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece": 0,
            }
            row["top_peak_source"] = top_peak_source(row)
            rows.append(row)
            torch.cuda.empty_cache()

    write_rows(out_dir / "v143_nonrat_manual_kernel_substrate_probe.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for family in sorted({str(r["family"]) for r in rows}):
        fam_rows = [r for r in rows if str(r["family"]) == family]
        summary_rows.append({
            "stage": "V143_NONRAT_MANUAL_KERNEL_SUBSTRATE_SUMMARY",
            "family": family,
            "rows": len(fam_rows),
            "manual_workspace_gate_pass_rows": sum(int(r["manual_workspace_gate_pass"]) for r in fam_rows),
            "best_raw_memory_ratio_vs_mlp": min(fnum(r["raw_memory_ratio_vs_mlp"], 999.0) for r in fam_rows),
            "best_incremental_memory_ratio_vs_mlp": min(fnum(r["incremental_memory_ratio_vs_mlp"], 999.0) for r in fam_rows),
            "best_step_ratio_vs_mlp": min(fnum(r["step_ratio_vs_mlp"], 999.0) for r in fam_rows),
            "promotion_allowed": 0,
        })
    write_rows(out_dir / "v143_nonrat_manual_kernel_substrate_summary.csv", summary_rows)
    route = {
        "stage": "V143_NONRAT_MANUAL_KERNEL_SUBSTRATE_ROUTE",
        "diagnostic_route": "D5-NonRATManualKernelWorkspaceProbe",
        "official_route_unchanged": "R4-NonRATSubstrateMissing",
        "new_training_executed": 0,
        "official_fms_proof_executed": 0,
        "manual_workspace_gate_pass_rows": sum(int(r["manual_workspace_gate_pass"]) for r in rows),
        "candidate_rows": len(rows),
        "promotion_allowed": 0,
        "real_short_run_open_allowed": 0,
    }
    (out_dir / "v143_nonrat_manual_kernel_substrate_route.json").write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
