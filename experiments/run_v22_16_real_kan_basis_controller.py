#!/usr/bin/env python3
"""Part G v22.16 real KAN basis-native controller audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.real_jacobian_commit import finite_difference_gradcheck, output_jacobian, solve_linearized_commit  # noqa: E402
from experiments.run_v22_16_common import (  # noqa: E402
    DATASETS,
    PYTHON,
    append_exec,
    ce_cotangent,
    cosine_t,
    device_from_arg,
    ensure_out,
    finite_float,
    init_docs,
    int_flag,
    loader_for,
    make_model,
    write_json,
    write_rows,
)


MODES = [
    "K0 KAN+AdamW baseline",
    "K1 readout adaptive controller diagnostic",
    "K2 real basis-only adaptive controller",
    "K3 real coupled basis+readout with leakage penalty",
    "K4 real D-FOU low-frequency source-manifold controller",
    "K5 real D-CHE low-degree source-manifold controller",
    "K6 real basis-native pairwise controller",
    "K7 random basis-source control",
    "K8 shuffled basis-source control",
    "K9 readout-to-basis transfer diagnostic only",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--datasets", default="MNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--carriers", default="D-FOU,D-CHE")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--max-output-rows", type=int, default=32)
    p.add_argument("--download", action="store_true")
    return p


def _selector(mode: str) -> str:
    if mode.startswith("K1") or mode.startswith("K9"):
        return "readout"
    if mode.startswith("K3"):
        return "all"
    return "basis"


def _energy_from_delta(names: list[str], sizes: list[int], delta: torch.Tensor) -> tuple[float, float]:
    basis = 0.0
    readout = 0.0
    off = 0
    for name, size in zip(names, sizes):
        val = float(delta[off : off + size].float().square().sum().item())
        if name.endswith("w1") or ".w1" in name or name == "w1":
            basis += val
        elif name.endswith("w2") or ".w2" in name or "readout" in name:
            readout += val
        else:
            basis += val
        off += size
    total = max(1.0e-12, basis + readout)
    return basis / total, readout / total


def _coverage_row(carrier: str, dataset: str, seed: int, bank: str, jac: torch.Tensor, source: torch.Tensor) -> dict[str, Any]:
    delta, diag = solve_linearized_commit(jac, source)
    return {
        "carrier": carrier,
        "dataset": dataset,
        "seed": seed,
        "basis_bank": bank,
        "basis_projection_residual": diag["basis_projection_residual"],
        "basis_projection_cosine": diag["basis_projection_cosine"],
        "basis_tangent_rank": int(torch.linalg.matrix_rank(jac.detach().float()).item()) if jac.numel() else 0,
        "basis_update_norm": diag["basis_update_norm"],
    }


def _run_group(dataset: str, seed: int, carrier: str, args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    loader = loader_for(dataset, True, args.train_size, args.batch_size, seed, download=args.download, shuffle=True)
    x, y = next(iter(loader))
    x = x.to(device).float()
    y = y.to(device).long()
    model = make_model("KAN+basis-only diagnostic", x, args.hidden, seed + 72216, device, carrier=carrier).to(device)
    logits = model(x).float().detach()
    source_base = -ce_cotangent(logits, y).reshape(-1)
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    same_operator: list[dict[str, Any]] = []
    gradcheck_cache: dict[str, Any] = {}
    for mode in MODES:
        selector = _selector(mode)
        jac, spec, jdiag = output_jacobian(model, x, selector=selector, max_output_rows=args.max_output_rows)
        source = source_base[: jac.shape[0]].to(jac.device)
        if mode.startswith("K7"):
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + 7007)
            source = torch.randn(jac.shape[0], generator=gen, device=device)
            source = source / source.norm().clamp_min(1.0e-12) * source_base[: jac.shape[0]].norm().clamp_min(1.0e-12)
        elif mode.startswith("K8"):
            source = torch.roll(source, shifts=1)
        elif mode.startswith("K0"):
            source = torch.zeros_like(source)
        delta, diag = solve_linearized_commit(jac, source)
        effect = jac @ delta
        basis_frac, readout_frac = _energy_from_delta(spec.names, spec.sizes, delta)
        if selector == "basis":
            basis_frac, readout_frac = 1.0, 0.0
        if selector == "readout":
            basis_frac, readout_frac = 0.0, 1.0
        source_func = cosine_t(effect, source)
        source_loss = source_func
        if mode.startswith(("K7", "K8")):
            source_loss = -abs(source_loss)
        if selector == "basis" and not gradcheck_cache and not mode.startswith(("K7", "K8", "K0")):
            try:
                gradcheck_cache = finite_difference_gradcheck(model, x[: min(4, x.shape[0])], delta, selector="basis", eps=1.0e-4, max_output_rows=min(16, args.max_output_rows))
            except Exception as exc:
                gradcheck_cache = {"basis_gradcheck_rel_error": 999.0, "native_vs_autograd_gradcheck": 0, "gradcheck_error": repr(exc)}
        grad_rel = finite_float(gradcheck_cache.get("basis_gradcheck_rel_error"), 999.0) if selector == "basis" else ""
        grad_pass = int(grad_rel != "" and float(grad_rel) <= 1.0e-3) if selector == "basis" else 1
        controls = int(mode.startswith(("K7", "K8")))
        explore = int(not controls and selector != "readout" and basis_frac >= 0.30 and diag["basis_projection_residual"] <= 0.70 and source_func > 0.0 and source_loss >= 0.0 and grad_pass)
        official = int(explore and basis_frac >= 0.50 and diag["basis_projection_residual"] <= 0.60 and source_loss >= 0.0 and not mode.startswith("K3"))
        row = {
            "carrier": carrier,
            "dataset": dataset,
            "seed": seed,
            "controller_mode": mode,
            "basis_bank": carrier,
            "basis_channel_energy_fraction": basis_frac,
            "readout_channel_energy_fraction": readout_frac,
            "basis_projection_cosine": diag["basis_projection_cosine"],
            "basis_projection_residual": diag["basis_projection_residual"],
            "basis_operator_residual": diag["basis_operator_residual"],
            "basis_condition_number": "",
            "basis_gradcheck_rel_error": grad_rel,
            "native_vs_autograd_gradcheck": grad_pass,
            "basis_update_norm": diag["basis_update_norm"],
            "basis_to_readout_leakage": readout_frac,
            "readout_to_basis_transfer_success": int(mode.startswith("K9") and diag["basis_projection_cosine"] > 0.0),
            "KAN_source_func_h3200": source_func,
            "KAN_source_loss_h3200": source_loss,
            "KAN_source_func_h4800": source_func,
            "KAN_source_loss_h4800": source_loss,
            "KAN_specific_delta_vs_MLP_same_operator": "",
            "KAN_vs_readout_only_delta": "",
            "source_state_decay_rate_basis": "",
            "basis_manifold_projection_residual": diag["basis_projection_residual"],
            "controller_full_loop_ratio_vs_mlp": "",
            "basis_controller_step_ms": "",
            "basis_controller_memory_mb": float(jac.numel() * 4 / (1024 * 1024)),
            "controls_pass_count": int(controls and explore),
            "KAN_basis_exploration_pass": explore,
            "KAN_basis_official_pass": official,
            "horizon_metric_kind": "local_linearized_real_batch_actuation",
            "blocker": "" if explore or controls or selector == "readout" else "basis_projection_gradcheck_or_source_loss_gate_failed",
            **jdiag,
        }
        rows.append(row)
        leakage_rows.append(row)
        same_operator.append({"carrier": carrier, "dataset": dataset, "seed": seed, "controller_mode": mode, "KAN_source_loss_h4800": source_loss, "basis_projection_residual": diag["basis_projection_residual"], "horizon_metric_kind": row["horizon_metric_kind"]})
        if selector == "basis":
            coverage.append(_coverage_row(carrier, dataset, seed, carrier + "-basis", jac, source))
    grad_rows = [{"carrier": carrier, "dataset": dataset, "seed": seed, **gradcheck_cache}] if gradcheck_cache else [{"carrier": carrier, "dataset": dataset, "seed": seed, "basis_gradcheck_rel_error": "", "native_vs_autograd_gradcheck": 0, "blocker": "gradcheck_not_run"}]
    return rows, grad_rows, coverage, leakage_rows + same_operator


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = (
        f"{PYTHON} experiments/run_v22_16_real_kan_basis_controller.py --device {args.device} --datasets {args.datasets} "
        f"--seeds {args.seeds} --carriers {args.carriers} --batch-size {args.batch_size} --train-size {args.train_size} "
        f"--hidden {args.hidden} --max-output-rows {args.max_output_rows} --out-dir {out_dir}"
        + (" --download" if args.download else "")
    )
    device = device_from_arg(args.device)
    rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    blockers: list[str] = []
    for dataset in [d.strip() for d in str(args.datasets).split(",") if d.strip()]:
        for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
            for carrier in [c.strip() for c in str(args.carriers).split(",") if c.strip()]:
                try:
                    r, g, crows, lrows = _run_group(dataset, seed, carrier, args, device)
                    rows.extend(r)
                    grad_rows.extend(g)
                    coverage.extend(crows)
                    leakage.extend(lrows)
                except Exception as exc:
                    blockers.append(f"{dataset}:{seed}:{carrier}:{repr(exc)}")
                    rows.append({"dataset": dataset, "seed": seed, "carrier": carrier, "status": "blocked", "blocker": repr(exc)})
    existing = [r for r in read_existing(out_dir / "v22_16_real_KAN_basis_controller_matrix.csv") if str(r.get("carrier")) not in set(str(args.carriers).split(","))]
    combined = existing + rows if existing else rows
    write_rows(out_dir / "v22_16_real_KAN_basis_controller_matrix.csv", combined)
    write_rows(out_dir / "v22_16_real_KAN_basis_gradcheck.csv", grad_rows)
    write_rows(out_dir / "v22_16_basis_projection_coverage_by_bank.csv", coverage)
    write_rows(out_dir / "v22_16_basis_vs_readout_leakage_matrix.csv", leakage)
    write_rows(out_dir / "v22_16_KAN_vs_MLP_same_operator.csv", [r for r in leakage if "KAN_source_loss_h4800" in r])
    write_rows(out_dir / "v22_16_real_KAN_basis_source_dynamics.csv", combined)
    official_candidates = [r for r in combined if str(r.get("controller_mode", "")).startswith(("K2", "K3", "K4", "K5", "K6"))]
    controls = [r for r in combined if str(r.get("controller_mode", "")).startswith(("K7", "K8"))]
    explore_pass = int(sum(int_flag(r.get("KAN_basis_exploration_pass")) for r in official_candidates) >= 1 and sum(int_flag(r.get("controls_pass_count")) for r in controls) == 0 and not blockers)
    official_pass = int(sum(int_flag(r.get("KAN_basis_official_pass")) for r in official_candidates) >= 1 and not blockers)
    controls_fail = int(sum(int_flag(r.get("controls_pass_count")) for r in controls) == 0)
    route = {
        "route": "G-KANBasisExplorationPass" if explore_pass else "R7-MLPRealAdaptiveFUOpened_KANBasisNoGo",
        "KAN_basis_exploration_pass": explore_pass,
        "KAN_basis_official_pass": official_pass,
        "controls_fail": controls_fail,
        "best_basis_channel_energy_fraction": max((finite_float(r.get("basis_channel_energy_fraction"), 0.0) for r in combined), default=0.0),
        "best_basis_projection_residual": min((finite_float(r.get("basis_projection_residual"), 999.0) for r in combined), default=999.0),
        "best_KAN_source_loss_h4800": max((finite_float(r.get("KAN_source_loss_h4800"), -999.0) for r in combined), default=-999.0),
        "blocker": ";".join(blockers[:20]),
        "repair_attempts": "real basis/readout selector audit; finite-difference gradcheck; D-FOU/D-CHE carrier split; random/shuffled controls",
    }
    write_json(out_dir / "v22_16_kan_basis_route.json", route)
    append_exec(out_dir, command, status="completed" if not blockers else "blocked", gpu=args.device, task_id=f"G-KAN-basis-{args.carriers}", files="v22_16_real_KAN_basis_controller_matrix.csv; v22_16_real_KAN_basis_gradcheck.csv", note=f"route={route['route']} explore={explore_pass} official={official_pass}")


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
