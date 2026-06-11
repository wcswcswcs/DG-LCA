#!/usr/bin/env python3
"""v22.01 D-RAT/D-RBF active efficiency repair and component telemetry."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.kernels.v17_basis import _rational_branchless, _rbf_local_k, kernel_correctness_row  # noqa: E402
from dgkan.profiling.efficiency_v22_01 import v22_01_efficiency_gate  # noqa: E402
from experiments.run_v22_01_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--batches", default="8,32")
    p.add_argument("--run-profile", action="store_true")
    return p


def time_op(fn: Callable[[], torch.Tensor], device: torch.device, repeats: int = 20) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(3):
        out = fn()
        if out.requires_grad:
            out.sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def run_profile(out_dir: Path, device: str, batches: str) -> None:
    cmd = [
        PYTHON,
        "experiments/run_v21_efficiency_officialization.py",
        "--out-dir",
        str(out_dir),
        "--device",
        device,
        "--families",
        "D-RAT,D-RBF",
        "--batches",
        batches,
        "--train-size",
        "128",
        "--val-size",
        "64",
        "--profiler-repeats",
        "1",
        "--profiler-warmup",
        "1",
        "--shard-count",
        "1",
        "--shard-index",
        "0",
    ]
    append_exec(out_dir, " ".join(cmd), status="started", note="D-RAT/D-RBF active repair full-loop profile")
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=1800)
    write_rows(out_dir / "v22_01_drat_drbf_profile_subprocess.csv", [{"command": " ".join(cmd), "returncode": proc.returncode, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]}])
    append_exec(out_dir, " ".join(cmd), status=f"exit={proc.returncode}", note="profile subprocess finished")
    merge = cmd[:]
    merge.append("--merge-only")
    append_exec(out_dir, " ".join(merge), status="started", note="merge D-RAT/D-RBF profile shards")
    proc2 = subprocess.run(merge, cwd=str(ROOT), text=True, capture_output=True, timeout=300)
    write_rows(out_dir / "v22_01_drat_drbf_merge_subprocess.csv", [{"command": " ".join(merge), "returncode": proc2.returncode, "stdout_tail": proc2.stdout[-1000:], "stderr_tail": proc2.stderr[-1000:]}])
    append_exec(out_dir, " ".join(merge), status=f"exit={proc2.returncode}", note="merge subprocess finished")


def rat_telemetry(device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device=device).manual_seed(2201)
    z = torch.randn(4096, 64, generator=gen, device=device)
    denom = 1.0 + 0.5 * z.abs() + 0.125 * z.square()
    reciprocal_ms = time_op(lambda: 1.0 / denom, device)
    numerator_ms = time_op(lambda: torch.stack([z, z.square(), z.square() * z, z.square().square()], dim=-1), device)
    denominator_ms = time_op(lambda: 1.0 + 0.5 * z.abs() + 0.125 * z.square(), device)
    full_ms = time_op(lambda: _rational_branchless(z, 4), device)
    z64 = torch.randn(256, 16, generator=gen, device=device, dtype=torch.float64, requires_grad=True)
    out = _rational_branchless(z64, 4).sum()
    grad = torch.autograd.grad(out, z64, create_graph=True)[0]
    rprime = grad.detach().abs().flatten()
    r2 = torch.autograd.grad(grad.sum(), z64)[0].detach().abs().flatten()
    den_flat = denom.detach().flatten().sort().values
    return {
        "carrier": "D-RAT",
        "component_variant": "RAT22-R4-telemetry-separated-training-path",
        "numerator_eval_ms": numerator_ms,
        "denominator_eval_ms": denominator_ms,
        "reciprocal_ms": reciprocal_ms,
        "component_full_eval_ms": full_ms,
        "derivative_telemetry_ms": "",
        "denominator_safety_ms": denominator_ms,
        "training_path_telemetry_free": 1,
        "rational_den_min": float(den_flat[0].item()),
        "rational_den_p01": float(den_flat[max(0, int(0.01 * den_flat.numel()) - 1)].item()),
        "rational_den_condition": float((den_flat[-1] / den_flat[0].clamp_min(1.0e-12)).item()),
        "r_prime_p99": float(rprime.sort().values[max(0, int(0.99 * rprime.numel()) - 1)].item()),
        "r_double_prime_p99": float(r2.sort().values[max(0, int(0.99 * r2.numel()) - 1)].item()),
        "denominator_safety_pass": int(float(den_flat[0].item()) > 0.1 and torch.isfinite(denom).all().item()),
    }


def rbf_telemetry(device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device=device).manual_seed(2211)
    z = torch.randn(4096, 64, generator=gen, device=device)
    basis = _rbf_local_k(z, 4)
    active = basis > 1.0e-3
    active_count = active.sum(dim=-1).float()
    dense_bytes = int(basis.numel() * basis.element_size())
    local_effective = float(active_count.mean().item())
    active_fraction = float(active.float().mean().item())
    exp_ms = time_op(lambda: _rbf_local_k(z, 4), device)
    z64 = torch.randn(256, 16, generator=gen, device=device, dtype=torch.float64, requires_grad=True)
    out = _rbf_local_k(z64, 4).sum()
    grad = torch.autograd.grad(out, z64)[0].detach()
    return {
        "carrier": "D-RBF",
        "component_variant": "RBF22-R2-compact-local-k4",
        "basis_materialized_bytes": dense_bytes,
        "basis_materialized_reduction_fraction": 0.0,
        "active_center_fraction": active_fraction,
        "local_k_effective": local_effective,
        "empty_center_fraction": float((active_count == 0).float().mean().item()),
        "width_p01": 2.0 / 3.0,
        "width_p50": 2.0 / 3.0,
        "width_p99": 2.0 / 3.0,
        "rbf_exp_eval_ms": exp_ms,
        "local_gather_ms": "",
        "local_backward_ms": "",
        "center_grad_snr": float(grad.abs().mean().item() / grad.std().clamp_min(1.0e-12).item()),
        "width_grad_snr": "",
        "width_safety_pass": int(torch.isfinite(basis).all().item() and active_fraction > 0.0),
    }


def summarize(out_dir: Path, component_rows: list[dict[str, Any]]) -> None:
    prof_rows = []
    for row in read_rows(out_dir / "v21_efficiency_truth_table.csv"):
        if str(row.get("carrier")) not in {"D-RAT", "D-RBF"}:
            continue
        item = dict(row)
        item["forward_ratio_vs_mlp"] = item.get("forward_ratio_vs_mlp", item.get("forward_ratio", ""))
        item["step_ratio_vs_mlp"] = item.get("step_ratio_vs_mlp", item.get("training_step_ratio", ""))
        item["memory_ratio_vs_mlp"] = item.get("memory_ratio_vs_mlp", item.get("memory_ratio", ""))
        comp = next((r for r in component_rows if r.get("carrier") == item.get("carrier")), {})
        item.update(comp)
        item.update(v22_01_efficiency_gate(item))
        prof_rows.append(item)
    write_rows(out_dir / "v22_01_drat_drbf_active_repair.csv", prof_rows)
    write_rows(out_dir / "v22_01_drat_drbf_component_waterfall.csv", component_rows)
    grad_rows = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for fam in ["D-RAT", "D-RBF"]:
        try:
            row = kernel_correctness_row(fam, device=device)
            grad_rows.append({"carrier": fam, **row, "gradcheck_pass": row.get("kernel_correctness_exploration_pass", 0)})
        except Exception as exc:
            grad_rows.append({"carrier": fam, "gradcheck_pass": 0, "blocker": f"{type(exc).__name__}:{exc}"})
    write_rows(out_dir / "v22_01_drat_drbf_kernel_gradcheck.csv", grad_rows)
    summary = []
    for fam in ["D-RAT", "D-RBF"]:
        fam_rows = [r for r in prof_rows if str(r.get("carrier")) == fam]
        fwd = [finite_float(r.get("forward_ratio_vs_mlp")) for r in fam_rows if math.isfinite(finite_float(r.get("forward_ratio_vs_mlp")))]
        step = [finite_float(r.get("step_ratio_vs_mlp")) for r in fam_rows if math.isfinite(finite_float(r.get("step_ratio_vs_mlp")))]
        near_key = "v22_01_NearE1_RAT_gate" if fam == "D-RAT" else "v22_01_NearE1_RBF_gate"
        near = sum(int_flag(r.get(near_key)) for r in fam_rows)
        blockers = sorted({str(r.get("v22_01_efficiency_blocker", "")) for r in fam_rows if str(r.get("v22_01_efficiency_blocker", "")) not in {"", "pass"}})
        summary.append(
            {
                "carrier": fam,
                "profile_rows": len(fam_rows),
                "near_E1_rows": near,
                "best_forward_ratio": min(fwd or [999.0]),
                "best_step_ratio": min(step or [999.0]),
                "decision": "NearE1Reached" if near else "RejectedForThisVersion",
                "blocker": ";".join(blockers) if blockers else ("ProfileMissing" if not fam_rows else "NearE1GateFailed"),
            }
        )
    write_rows(out_dir / "v22_01_drat_drbf_active_repair_summary.csv", summary)
    write_json(out_dir / "v22_01_drat_drbf_active_repair_decision.json", {"D-RAT": next((r for r in summary if r["carrier"] == "D-RAT"), {}), "D-RBF": next((r for r in summary if r["carrier"] == "D-RBF"), {})})
    simple_svg(out_dir / "figures" / "efficiency_component_waterfall_DRAT_DRBF.svg", "D-RAT/D-RBF component waterfall", component_rows, "component_full_eval_ms")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.run_profile:
        run_profile(out_dir, args.device, args.batches)
    component_rows = []
    try:
        component_rows.append(rat_telemetry(device))
    except Exception as exc:
        component_rows.append({"carrier": "D-RAT", "component_variant": "RAT22-telemetry", "blocker": f"{type(exc).__name__}:{exc}"})
    try:
        component_rows.append(rbf_telemetry(device))
    except Exception as exc:
        component_rows.append({"carrier": "D-RBF", "component_variant": "RBF22-telemetry", "blocker": f"{type(exc).__name__}:{exc}"})
    summarize(out_dir, component_rows)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_01_drat_drbf_active_repair.py --device {args.device} --batches {args.batches} {'--run-profile' if args.run_profile else ''}", status="completed", note=f"component_rows={len(component_rows)}")


if __name__ == "__main__":
    main()
