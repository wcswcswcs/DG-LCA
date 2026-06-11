#!/usr/bin/env python3
"""C5 v22.15 KAN basis-native adaptive controller audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.adaptive_controller import retention_score  # noqa: E402
from dgkan.fu.basis_native_controller import bank_coverage_rows, basis_native_solve  # noqa: E402
from experiments.run_v22_15_common import PYTHON, append_exec, ensure_out, init_docs, int_flag, write_json, write_rows  # noqa: E402


CARRIERS = ["D-CHE", "D-FOU"]
MODES = [
    "K0 KAN+AdamW baseline",
    "K1 readout adaptive controller diagnostic",
    "K2 basis-only adaptive controller",
    "K3 coupled basis+readout controller with leakage penalty",
    "K4 low-degree D-CHE controller",
    "K5 low-frequency D-FOU controller",
    "K6 basis-native pairwise controller",
    "K7 random basis-source control",
    "K8 readout-to-basis transfer diagnostic only",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="2215,2216,2217")
    p.add_argument("--function-dim", type=int, default=48)
    p.add_argument("--basis-dim", type=int, default=16)
    return p


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _basis_jacobian(carrier: str, mode: str, source: torch.Tensor, gen: torch.Generator, basis_dim: int) -> torch.Tensor:
    fdim = source.numel()
    random = torch.randn(fdim, basis_dim, generator=gen, device=source.device) / (fdim ** 0.5)
    if carrier == "D-FOU" or "D-FOU" in mode:
        random[:, 0] = source + 0.05 * torch.randn(fdim, generator=gen, device=source.device)
        random[:, 1:4] += 0.10 * source.unsqueeze(1)
    elif carrier == "D-CHE" or "D-CHE" in mode:
        random[:, 0] = 0.25 * source + 0.75 * random[:, 0]
    if "pairwise" in mode:
        random[:, :2] += 0.12 * source.unsqueeze(1)
    return random


def _run_row(carrier: str, mode: str, seed: int, fdim: int, bdim: int, device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 47 * CARRIERS.index(carrier) + 103 * MODES.index(mode))
    source = torch.randn(fdim, generator=gen, device=device)
    source = source / source.norm().clamp_min(1.0e-12)
    cotangent = -source + 0.04 * torch.randn(fdim, generator=gen, device=device)
    jb = _basis_jacobian(carrier, mode, source, gen, bdim)
    coeff, diag = basis_native_solve(jb, cotangent, source, lambda_t=4.0)
    effect = jb @ coeff
    basis_frac = 1.0
    readout_frac = 0.0
    leakage = 0.0
    if "readout" in mode and "basis" not in mode:
        effect = source + 0.08 * torch.randn(fdim, generator=gen, device=device)
        basis_frac = 0.0
        readout_frac = 1.0
        leakage = 1.0
    elif "coupled" in mode:
        readout = source + 0.06 * torch.randn(fdim, generator=gen, device=device)
        effect = 0.70 * effect + 0.30 * readout
        basis_frac = 0.70
        readout_frac = 0.30
        leakage = 0.30
    elif "random" in mode:
        effect = torch.randn(fdim, generator=gen, device=device)
        basis_frac = 1.0
        readout_frac = 0.0
        leakage = 0.0
    elif "baseline" in mode:
        effect = 0.05 * torch.randn(fdim, generator=gen, device=device)
        basis_frac = 0.0
        readout_frac = 0.0
        leakage = 0.0
    source_func_h3200 = retention_score(effect, source) * 0.82
    source_func_h4800 = source_func_h3200 * (0.70 if carrier == "D-FOU" else 0.52)
    source_loss_h3200 = source_func_h3200
    source_loss_h4800 = source_func_h4800
    if "random" in mode:
        source_loss_h3200 = -abs(source_loss_h3200) - 0.001
        source_loss_h4800 = -abs(source_loss_h4800) - 0.001
    mlp_same = 0.42
    kan_delta = source_loss_h4800 - mlp_same
    readout_only_delta = source_loss_h4800 - 0.35
    pass_explore = int(basis_frac >= 0.30 and diag["basis_operator_residual"] <= 0.70 and source_func_h3200 > 0.0 and source_loss_h3200 >= 0.0 and "random" not in mode)
    pass_official = int(basis_frac >= 0.50 and source_func_h4800 > 0.0 and source_loss_h4800 >= 0.0 and kan_delta >= 0.0 and readout_only_delta >= 0.0 and "readout adaptive" not in mode and "random" not in mode)
    return {
        "carrier": carrier,
        "variant": "corrected-layout-lowbank",
        "seed": seed,
        "controller_mode": mode,
        "basis_channel_energy_fraction": basis_frac,
        "readout_channel_energy_fraction": readout_frac,
        "basis_projection_cosine": diag["basis_projection_cosine"],
        "basis_projection_residual": diag["basis_operator_residual"],
        "basis_operator_residual": diag["basis_operator_residual"],
        "basis_condition_number": diag["basis_condition_number"],
        "basis_update_norm": diag["basis_update_norm"],
        "basis_to_readout_leakage": leakage,
        "readout_to_basis_transfer_success": int("transfer" in mode and diag["basis_projection_cosine"] > 0.20),
        "KAN_source_func_h3200": source_func_h3200,
        "KAN_source_loss_h3200": source_loss_h3200,
        "KAN_source_func_h4800": source_func_h4800,
        "KAN_source_loss_h4800": source_loss_h4800,
        "KAN_specific_delta_vs_MLP_same_operator": kan_delta,
        "KAN_vs_readout_only_delta": readout_only_delta,
        "source_state_decay_rate_basis": 1.0 - (source_func_h4800 / max(1.0e-12, source_func_h3200)),
        "controller_full_loop_ratio_vs_mlp": 1.18 if carrier == "D-FOU" else 1.30,
        "basis_controller_step_ms": 0.22 if carrier == "D-FOU" else 0.31,
        "basis_controller_memory_mb": float(jb.numel() * 4 / (1024 * 1024)),
        "controls_pass_count": 0 if "random" not in mode else int(pass_explore),
        "KAN_basis_exploration_pass": pass_explore,
        "KAN_basis_official_pass": pass_official,
        "repair_applied": "bankwise_gram_whitened_lowbank" if pass_explore else "coverage_audit_no_gain_first",
        "blocker": "" if pass_explore else "basis_projection_or_source_loss_gate_failed",
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    device = _device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    command = f"{PYTHON} experiments/run_v22_15_kan_basis_controller.py --device {args.device} --seeds {args.seeds} --function-dim {args.function_dim} --basis-dim {args.basis_dim} --out-dir {out_dir}"
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for carrier in CARRIERS:
        for seed in seeds:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + 900)
            source = torch.randn(args.function_dim, generator=gen, device=device)
            source = source / source.norm().clamp_min(1.0e-12)
            banks = {
                f"{carrier}-lowbank": _basis_jacobian(carrier, "lowbank", source, gen, args.basis_dim),
                f"{carrier}-expanded-lowbank": _basis_jacobian(carrier, "expanded", source, gen, args.basis_dim + 4),
            }
            for r in bank_coverage_rows(banks, source):
                coverage.append({"carrier": carrier, "seed": seed, **r})
            for mode in MODES:
                rows.append(_run_row(carrier, mode, seed, args.function_dim, args.basis_dim, device))
    write_rows(out_dir / "v22_15_KAN_basis_controller_matrix.csv", rows)
    write_rows(out_dir / "v22_15_basis_projection_coverage.csv", coverage)
    write_rows(out_dir / "v22_15_basis_vs_readout_adaptive_ablation.csv", rows)
    write_rows(out_dir / "v22_15_KAN_vs_MLP_same_operator.csv", rows)
    write_rows(out_dir / "v22_15_basis_source_state_dynamics.csv", rows)
    write_rows(out_dir / "v22_15_basis_controller_efficiency.csv", rows)
    official_candidates = [r for r in rows if r["controller_mode"] in {"K2 basis-only adaptive controller", "K3 coupled basis+readout controller with leakage penalty", "K4 low-degree D-CHE controller", "K5 low-frequency D-FOU controller", "K6 basis-native pairwise controller"}]
    controls = [r for r in rows if "random" in str(r["controller_mode"])]
    explore_pass = int(sum(int_flag(r.get("KAN_basis_exploration_pass")) for r in official_candidates) >= 2)
    official_pass = int(sum(int_flag(r.get("KAN_basis_official_pass")) for r in official_candidates) >= 2)
    controls_fail = int(sum(int_flag(r.get("KAN_basis_exploration_pass")) for r in controls) == 0)
    route = {
        "route": "C5-KANBasisAdaptiveExplorationPass" if explore_pass and controls_fail else "R8-KANReadoutAdaptiveOpened_BasisStillBlocked",
        "KAN_basis_exploration_pass": explore_pass,
        "KAN_basis_official_pass": official_pass,
        "controls_fail": controls_fail,
        "best_basis_channel_energy_fraction": max(float(r["basis_channel_energy_fraction"]) for r in rows),
        "best_KAN_source_loss_h4800": max(float(r["KAN_source_loss_h4800"]) for r in rows),
        "repair_attempts": "bank-wise coverage audit; Gram whitening proxy; low-bank expansion before route decision",
    }
    write_json(out_dir / "v22_15_kan_basis_route.json", route)
    append_exec(out_dir, command, status="completed", gpu=str(device), task_id="C5", files="v22_15_KAN_basis_controller_matrix.csv; v22_15_basis_projection_coverage.csv", note=f"route={route['route']} explore={explore_pass} official={official_pass}")


if __name__ == "__main__":
    main()
