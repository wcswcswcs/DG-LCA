#!/usr/bin/env python3
"""v22.69R Part F target-free mixed-bank KAN full-loop exploration.

This runner is official-eligible in the narrow Part F sense: it does not use
the MLP winner target during runtime.  It registers the Part-D-discovered
mixed bank as an actual trainable base by concatenating the feature maps of
DGKAN_FOU4_LIN and DGKAN_HAT4_XLIN and attaching one shared readout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66  # noqa: E402
import experiments.run_v22_67_kanaware_metric_compatible_generator_carrier_fu as v67  # noqa: E402
import experiments.run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit as r69  # noqa: E402


OUT_PREFIX = "v22_69R_part_f_target_free_mixedbank"
MIXED_ARCH = "DGKAN_MIXED_FOU4LIN_HAT4XLIN"
MIXED_SUB_ARCHES = ["DGKAN_FOU4_LIN", "DGKAN_HAT4_XLIN"]
DEFAULT_KAN_REFERENCE_METHODS = ["adamw", "cautious_adamw"]
DEFAULT_KAN_CANDIDATE_METHODS = ["mcga_gradcoh_fsclip_generator_rank4", "mcga_cvargrad_fsclip_generator_rank4"]
DEFAULT_KAN_CONTROL_METHODS = ["same_generator_descent_energy_random", "same_C_skew_spectrum_random", "same_rank_random_coordinate"]
DEFAULT_MLP_MATCHED_METHODS = ["mcga_gradcoh_fsclip_generator_rank4", "mcga_cvargrad_fsclip_generator_rank4"]


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def unique_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def method_lists(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], list[str]]:
    refs = parse_csv(args.reference_methods)
    candidates = parse_csv(args.candidate_methods)
    controls = parse_csv(args.control_methods)
    mlp = parse_csv(args.mlp_matched_methods)
    return (
        refs or list(DEFAULT_KAN_REFERENCE_METHODS),
        candidates or list(DEFAULT_KAN_CANDIDATE_METHODS),
        controls or list(DEFAULT_KAN_CONTROL_METHODS),
        mlp or list(DEFAULT_MLP_MATCHED_METHODS),
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def flt(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out
    except Exception:
        return default


def ifinite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values: list[float]) -> float | str:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return "" if not vals else float(statistics.fmean(vals))


def percentile(values: list[float], p: float) -> float | str:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return ""
    idx = max(0, min(len(vals) - 1, int(math.floor((len(vals) - 1) * p))))
    return float(vals[idx])


def make_mixed_bank_base(bundle: dict[str, Any], device: Any, hidden: int, seed: int) -> Any:
    import torch
    import torch.nn as nn

    bases = [
        v67.make_redesigned_kan_base(arch, bundle, device, int(hidden), int(seed))[0]
        for arch in MIXED_SUB_ARCHES
    ]

    class MixedBankReadoutBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.banks = nn.ModuleList(bases)
            out_dim = int(bases[0].fc3.out_features)
            in_dim = sum(int(base.fc3.in_features) for base in bases)
            self.fc3 = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                self.fc3.weight.zero_()
                start = 0
                for base in bases:
                    width = int(base.fc3.in_features)
                    self.fc3.weight[:, start : start + width].copy_(base.fc3.weight.detach() / float(len(bases)))
                    start += width
                self.fc3.bias.copy_(sum(base.fc3.bias.detach() for base in bases) / float(len(bases)))
            self.mixed_bank_architecture = MIXED_ARCH
            self.mixed_bank_sub_arches = "+".join(MIXED_SUB_ARCHES)
            self.kan_readout_linearization_max_abs_error = max(
                float(getattr(base, "kan_readout_linearization_max_abs_error", 0.0) or 0.0)
                for base in bases
            )

        def features(self, x: Any) -> Any:
            return torch.cat([base.features(x) for base in self.banks], dim=1)

        def forward(self, x: Any) -> Any:
            return self.fc3(self.features(x))

    return MixedBankReadoutBase().to(device)


def install_v66_hooks() -> None:
    import torch

    original_make_base_model = v66.make_base_model
    original_capacity = v66.compute_capacity_diagnostics

    def readout_weight(model: Any) -> Any:
        fc3 = getattr(model, "fc3", None)
        if fc3 is not None and hasattr(fc3, "effective_weight"):
            return fc3.effective_weight().detach().float()
        if fc3 is not None and hasattr(fc3, "weight"):
            return fc3.weight.detach().float()
        coord = v66.coord_model_of(model)
        fc3 = getattr(coord, "fc3", None)
        if fc3 is not None and hasattr(fc3, "effective_weight"):
            return fc3.effective_weight().detach().float()
        if fc3 is not None and hasattr(fc3, "weight"):
            return fc3.weight.detach().float()
        raise AttributeError("model readout has neither weight nor effective_weight")

    def stable_projection_capacity(design: Any, target: Any, eps: float = 1.0e-10) -> dict[str, Any]:
        work_phi = design.detach().double()
        work_target = target.detach().double().reshape(-1, 1)
        if int(work_phi.numel()) == 0 or int(work_phi.shape[1]) == 0 or int(work_target.numel()) == 0:
            return {"capacity": 0.0, "projected_norm": 0.0, "target_norm": 0.0, "rank": 0, "condition": 0.0}
        u, s, _ = torch.linalg.svd(work_phi, full_matrices=False)
        if int(s.numel()) == 0:
            return {"capacity": 0.0, "projected_norm": 0.0, "target_norm": float(torch.linalg.norm(work_target).detach().cpu().item()), "rank": 0, "condition": 0.0}
        keep = s > (float(eps) * s.max().clamp_min(float(eps)))
        if int(keep.sum().item()) == 0:
            return {"capacity": 0.0, "projected_norm": 0.0, "target_norm": float(torch.linalg.norm(work_target).detach().cpu().item()), "rank": 0, "condition": 0.0}
        basis = u[:, keep]
        proj = basis @ (basis.transpose(0, 1) @ work_target)
        target_norm = torch.linalg.norm(work_target).clamp_min(float(eps))
        proj_norm = torch.linalg.norm(proj)
        kept_s = s[keep]
        return {
            "capacity": float((proj_norm.square() / target_norm.square()).clamp(min=0.0, max=1.0).detach().cpu().item()),
            "projected_norm": float(proj_norm.detach().cpu().item()),
            "target_norm": float(target_norm.detach().cpu().item()),
            "rank": int(keep.sum().detach().cpu().item()),
            "condition": float((kept_s.max() / kept_s.min().clamp_min(float(eps))).detach().cpu().item()),
            "projection": proj.reshape(-1),
        }

    def weighted_lower_tail(values: Any, weights: Any, frac: float = 0.25) -> float:
        vals = values.detach().float().reshape(-1)
        w = weights.detach().float().reshape(-1).clamp_min(0.0)
        mask = torch.isfinite(vals) & torch.isfinite(w) & (w > 0)
        if int(mask.sum().item()) == 0:
            return 0.0
        vals = vals[mask]
        w = w[mask]
        order = torch.argsort(vals)
        vals = vals[order]
        w = w[order]
        target_mass = torch.clamp(w.sum() * float(frac), min=1.0e-12)
        acc = torch.zeros((), device=vals.device, dtype=vals.dtype)
        mass = torch.zeros((), device=vals.device, dtype=vals.dtype)
        for val, weight in zip(vals, w):
            take = torch.minimum(weight, target_mass - mass)
            acc = acc + take * val
            mass = mass + take
            if bool((mass >= target_mass - 1.0e-12).detach().cpu().item()):
                break
        return float((acc / mass.clamp_min(1.0e-12)).detach().cpu().item())

    def active_or_raw_design(model: Any, phi: Any) -> tuple[Any, str, dict[str, Any]]:
        layer = v66.get_coord_layer(model)
        if layer is not None and all(hasattr(layer, name) for name in ["output_basis", "input_basis", "coord_scale"]):
            output_basis = layer.output_basis.detach().float().to(phi.device)
            input_basis = layer.input_basis.detach().float().to(phi.device)
            coord_scale = layer.coord_scale.detach().float().to(phi.device).view(-1)
            rank = min(int(output_basis.shape[1]), int(input_basis.shape[1]), int(coord_scale.numel()))
            if rank > 0:
                h_v = phi.float() @ input_basis[:, :rank]
                cols = []
                for i in range(rank):
                    out_col = output_basis[:, i].view(1, -1)
                    for j in range(rank):
                        col_logits = h_v[:, j : j + 1] * coord_scale[j] * out_col
                        cols.append(col_logits.reshape(-1))
                if cols:
                    extras: dict[str, Any] = {}
                    try:
                        eye = torch.eye(rank, device=phi.device, dtype=output_basis.dtype)
                        update = layer.iso_operator().detach().float().to(phi.device) - eye
                        if hasattr(layer, "shape_operator"):
                            update = update + layer.shape_operator().detach().float().to(phi.device)
                        update = update[:rank, :rank] * coord_scale[:rank].view(1, -1)
                        extras["active_coord_weights"] = update.abs().reshape(-1)
                        extras["active_delta_coord_l1_mass"] = float(update.abs().sum().detach().cpu().item())
                        extras["active_delta_coord_l2_mass"] = float(update.square().sum().sqrt().detach().cpu().item())
                        threshold = update.abs().max().clamp_min(1.0e-12) * 1.0e-4
                        extras["active_delta_coord_active_fraction"] = float((update.abs() > threshold).float().mean().detach().cpu().item())
                    except Exception as exc:
                        extras["active_delta_weight_error"] = f"{type(exc).__name__}: {exc}"
                    return torch.stack(cols, dim=1), "active_atlas_coordinate_design", extras
        cols = []
        weight = readout_weight(model).to(phi.device)
        for j in range(int(phi.shape[1])):
            col_logits = phi[:, j : j + 1].float() @ weight[:, j : j + 1].transpose(0, 1)
            cols.append(col_logits.reshape(-1))
        design = torch.stack(cols, dim=1) if cols else torch.zeros((int(phi.shape[0]), 0), device=phi.device)
        return design, "raw_readout_feature_design_fallback", {}

    def hooked_make_base_model(args: argparse.Namespace, bundle: dict[str, Any], device: Any) -> Any:
        if str(getattr(args, "architecture", "")) == MIXED_ARCH:
            return make_mixed_bank_base(bundle, device, int(args.hidden), int(args.seed))
        return original_make_base_model(args, bundle, device)

    def hooked_capacity(model: Any, x: Any, y: Any, batch_size: int = 512) -> dict[str, Any]:
        out = dict(original_capacity(model, x, y, batch_size))
        try:
            device = next(model.parameters()).device
            xb = x[: int(batch_size)].to(device)
            yb = y[: int(batch_size)].to(device).long()
            with torch.no_grad():
                phi = model.features(xb).detach().float()
                logits = model(xb).detach().float()
            target = v67.part_c_task_target(model, xb, yb, int(logits.shape[1])).to(device).float()
            target = target / target.norm().clamp_min(1.0e-12)
            logits_dim = int(target.numel())
            output_dim = int(logits_dim // max(1, int(xb.shape[0])))
            with torch.no_grad():
                design, design_source, design_extra = active_or_raw_design(model, phi)
            visibility = v67.part_c_column_visibility(design)
            vis_values = visibility["normalized"].detach().float().reshape(-1)
            atlas_readout_mean = float(visibility["mean"])
            atlas_readout_cvar = float(visibility["cvar25"])
            atlas_readout_p10 = percentile([float(v) for v in vis_values.detach().cpu().tolist()], 0.10)
            readout_mean = atlas_readout_mean
            readout_cvar = atlas_readout_cvar
            readout_p10 = atlas_readout_p10
            gram_design = design
            diagnostic_source = design_source
            weights = design_extra.get("active_coord_weights")
            if weights is not None and int(design.shape[1]) == int(weights.numel()):
                weights = weights.detach().float().to(design.device).reshape(-1)
                total_weight = float(weights.sum().detach().cpu().item())
                if total_weight > 1.0e-12:
                    w = weights / weights.sum().clamp_min(1.0e-12)
                    readout_mean = float((vis_values.to(design.device) * w).sum().detach().cpu().item())
                    readout_cvar = weighted_lower_tail(vis_values.to(design.device), weights, 0.25)
                    keep = weights > weights.max().clamp_min(1.0e-12) * 1.0e-4
                    if int(keep.sum().item()) > 0:
                        gram_design = design[:, keep] * torch.sqrt(weights[keep].clamp_min(1.0e-12)).view(1, -1)
                        diagnostic_source = f"{design_source}::active_delta_weighted"
            gram = gram_design.double().transpose(0, 1) @ gram_design.double() / max(1, int(gram_design.shape[0]))
            eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
            evals = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(0, 1)) + 1.0e-8 * eye)
            cond = float((evals.max() / evals.min().clamp_min(1.0e-8)).detach().cpu().item()) if int(evals.numel()) else 0.0
            eff_rank = float((evals.sum().square() / evals.square().sum().clamp_min(1.0e-12)).detach().cpu().item()) if int(evals.numel()) else 0.0
            cap = stable_projection_capacity(design, target)
            gen = torch.Generator(device=device)
            gen.manual_seed(69269 + int(phi.shape[1]) + int(xb.shape[0]) + output_dim)
            rand = torch.randn_like(design, generator=gen) if int(design.numel()) else design
            rand_cap = stable_projection_capacity(rand, target)
            margin = float(cap.get("capacity", 0.0) or 0.0) - float(rand_cap.get("capacity", 0.0) or 0.0)
            out.update(
                {
                    "pure_gauge_energy_fraction": 0.0,
                    "horizontal_energy_fraction": 1.0,
                    "quotient_projection_residual": float(max(0.0, 1.0 - float(cap.get("capacity", 0.0) or 0.0))),
                    "readout_visible_energy_mean": readout_mean,
                    "readout_visible_energy_CVaR25": readout_cvar,
                    "readout_visible_energy_p10": readout_p10,
                    "atlas_coordinate_readout_visible_energy_mean": atlas_readout_mean,
                    "atlas_coordinate_readout_visible_energy_CVaR25": atlas_readout_cvar,
                    "atlas_coordinate_readout_visible_energy_p10": atlas_readout_p10,
                    "control_contrastive_margin_p10": margin,
                    "control_contrastive_margin_CVaR25": margin,
                    "basis_Gram_condition": cond,
                    "basis_Gram_effective_rank": eff_rank,
                    "basis_Gram_condition_source": diagnostic_source,
                    "basis_design_rank": int(cap.get("rank", 0) or 0),
                    "basis_design_svd_condition": float(cap.get("condition", 0.0) or 0.0),
                    "active_delta_coord_l1_mass": design_extra.get("active_delta_coord_l1_mass", ""),
                    "active_delta_coord_l2_mass": design_extra.get("active_delta_coord_l2_mass", ""),
                    "active_delta_coord_active_fraction": design_extra.get("active_delta_coord_active_fraction", ""),
                    "active_delta_weight_error": design_extra.get("active_delta_weight_error", ""),
                    "G_B_skew_residual": out.get("C_skew_projection_error", ""),
                    "C_skew_residual": out.get("C_skew_projection_error", ""),
                    "MLP_target_used_in_official_runtime": 0,
                    "part_f_extra_capacity_source": f"train_only_task_target_from_current_KAN_logits_no_MLP_target::{diagnostic_source}",
                }
            )
        except Exception as exc:
            out.update({"part_f_extra_capacity_error": f"{type(exc).__name__}: {exc}"})
        return out

    v66.make_base_model = hooked_make_base_model
    v66.compute_capacity_diagnostics = hooked_capacity


def row_args_from_cli(args: argparse.Namespace) -> argparse.Namespace:
    row_args = v66.build_parser().parse_args([])
    row_args.mode = "row"
    row_args.dataset = str(args.dataset)
    row_args.seed = int(args.seed)
    row_args.method = str(args.method)
    row_args.architecture = str(args.architecture)
    row_args.run_label = str(args.run_label)
    row_args.device = str(args.device)
    row_args.train_size = int(args.train_size)
    row_args.held_size = int(args.held_size)
    row_args.test_size = int(args.test_size)
    row_args.hidden = int(args.hidden)
    row_args.steps = int(args.steps)
    row_args.batch_size = int(args.batch_size)
    row_args.eval_batch_size = int(args.eval_batch_size)
    row_args.metric_batch_size = int(args.metric_batch_size)
    row_args.refresh = int(args.refresh)
    row_args.lr = float(args.lr)
    row_args.weight_decay = float(args.weight_decay)
    row_args.functional_spectrum_drift_threshold = float(args.functional_spectrum_drift_threshold)
    row_args.iso_eta = float(args.iso_eta)
    row_args.shaping_budget = float(args.shaping_budget)
    return row_args


def run_row(args: argparse.Namespace) -> int:
    install_v66_hooks()
    row_args = row_args_from_cli(args)
    row = v66.train_row(row_args)
    print(json.dumps({"row_path": str(v66.row_path(row_args).relative_to(ROOT)), "status": row.get("run_status", "")}, ensure_ascii=False))
    return 0


def matrix_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = parse_csv(args.datasets)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    ref_methods, candidate_methods, control_methods, mlp_methods = method_lists(args)
    all_kan_methods = unique_preserve(ref_methods + candidate_methods + control_methods)
    tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for method in all_kan_methods:
                tasks.append({"architecture": MIXED_ARCH, "dataset": dataset, "seed": seed, "method": method})
            for method in mlp_methods:
                tasks.append({"architecture": "MLP", "dataset": dataset, "seed": seed, "method": method})
    return tasks


def run_one_subprocess(base_args: argparse.Namespace, task: dict[str, Any], gpu: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).relative_to(ROOT)),
        "--mode",
        "row",
        "--architecture",
        str(task["architecture"]),
        "--dataset",
        str(task["dataset"]),
        "--seed",
        str(task["seed"]),
        "--method",
        str(task["method"]),
        "--run-label",
        str(base_args.run_label),
        "--device",
        "cuda:0" if str(gpu) != "cpu" else "cpu",
        "--train-size",
        str(base_args.train_size),
        "--held-size",
        str(base_args.held_size),
        "--test-size",
        str(base_args.test_size),
        "--hidden",
        str(base_args.hidden),
        "--steps",
        str(base_args.steps),
        "--batch-size",
        str(base_args.batch_size),
        "--eval-batch-size",
        str(base_args.eval_batch_size),
        "--metric-batch-size",
        str(base_args.metric_batch_size),
        "--refresh",
        str(base_args.refresh),
        "--lr",
        str(base_args.lr),
        "--weight-decay",
        str(base_args.weight_decay),
        "--functional-spectrum-drift-threshold",
        str(base_args.functional_spectrum_drift_threshold),
    ]
    env = dict(os.environ)
    if str(gpu) != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=int(base_args.row_timeout))
    return {
        **task,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - t0, 3),
        "cmd": " ".join(cmd),
        "stdout": proc.stdout.strip()[-1000:],
        "stderr": proc.stderr.strip()[-2000:],
    }


def run_matrix(args: argparse.Namespace) -> int:
    r69.ensure_out()
    tasks = matrix_tasks(args)
    status_path = r69.OUT_ROOT / f"{OUT_PREFIX}_subprocess_status.csv"
    r69.append_exec(
        "F_target_free_mixedbank_matrix_start",
        r69.command_text([sys.executable, r69.rel(Path(__file__)), *sys.argv[1:]]),
        "start",
        gpu=str(args.gpus),
        files=f"{r69.rel(status_path)}; results/v22_66/chunks/*{args.run_label}*.csv",
        note=f"tasks={len(tasks)}; datasets={args.datasets}; seeds={args.seeds}; reference_methods={method_lists(args)[0]}; candidate_methods={method_lists(args)[1]}; control_methods={method_lists(args)[2]}; mlp_methods={method_lists(args)[3]}; target_free=1",
    )
    gpus = parse_csv(args.gpus) or ["cpu"]
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as pool:
        futs = []
        for idx, task in enumerate(tasks):
            futs.append(pool.submit(run_one_subprocess, args, task, gpus[idx % len(gpus)]))
        for fut in as_completed(futs):
            status = fut.result()
            statuses.append(status)
            write_rows(status_path, statuses)
    failed = [s for s in statuses if int(s["returncode"]) != 0]
    r69.append_exec(
        "F_target_free_mixedbank_matrix",
        r69.command_text([sys.executable, r69.rel(Path(__file__)), *sys.argv[1:]]),
        "fail" if failed else "done",
        gpu=str(args.gpus),
        files=f"{r69.rel(status_path)}; results/v22_66/chunks/*{args.run_label}*.csv",
        note=json.dumps({"tasks": len(tasks), "failed": len(failed), "status_path": r69.rel(status_path)}, ensure_ascii=False),
    )
    if failed:
        print(json.dumps({"failed": len(failed), "examples": failed[:3]}, ensure_ascii=False, indent=2))
        return 1
    return run_analyze(args)


def collect_rows(run_label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    label = v66.safe_fragment(str(run_label))
    for path in sorted(v66.CHUNK_ROOT.glob(f"{label}_*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            row["source_chunk"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def annotate_candidate_rows(
    rows: list[dict[str, str]],
    threshold: float,
    ref_methods: list[str],
    candidate_methods: list[str],
    control_methods: list[str],
    mlp_methods: list[str],
) -> list[dict[str, Any]]:
    out = [dict(r) for r in rows]
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in out:
        if row.get("run_status") == "completed":
            by_group.setdefault((str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    for group in by_group.values():
        kan_refs = [r for r in group if r.get("architecture_key") == MIXED_ARCH and r.get("method") in ref_methods]
        kan_controls = [r for r in group if r.get("architecture_key") == MIXED_ARCH and r.get("method") in control_methods]
        mlp_rows = [r for r in group if r.get("architecture_key") == "MLP" and r.get("method") in mlp_methods]
        best_ref = min((flt(r.get("held_NLL")) for r in kan_refs if ifinite(r.get("held_NLL"))), default=float("inf"))
        best_control = min((flt(r.get("held_NLL")) for r in kan_controls if ifinite(r.get("held_NLL"))), default=float("inf"))
        same_gen = {str(r.get("method")): flt(r.get("held_NLL")) for r in kan_controls if ifinite(r.get("held_NLL"))}
        best_mlp = min((flt(r.get("held_NLL")) for r in mlp_rows if ifinite(r.get("held_NLL"))), default=float("inf"))
        best_ref_debt = min((flt(r.get("ECE")) + flt(r.get("Brier")) + flt(r.get("tail_loss_q95")) for r in kan_refs if all(ifinite(r.get(k)) for k in ["ECE", "Brier", "tail_loss_q95"])), default=float("inf"))
        best_mlp_ref = min((flt(r.get("initial_held_NLL")) for r in mlp_rows if ifinite(r.get("initial_held_NLL"))), default=float("inf"))
        for row in group:
            if row.get("architecture_key") != MIXED_ARCH or row.get("method") not in candidate_methods:
                continue
            nll = flt(row.get("held_NLL"))
            debt = flt(row.get("ECE")) + flt(row.get("Brier")) + flt(row.get("tail_loss_q95"))
            row["Delta_NLL_vs_KAN_own_strong_optimizer"] = nll - best_ref if math.isfinite(best_ref) else ""
            row["Delta_NLL_vs_best_KAN_control"] = nll - best_control if math.isfinite(best_control) else ""
            row["Delta_NLL_vs_MLP_matched_coordinate"] = nll - best_mlp if math.isfinite(best_mlp) else ""
            row["KAN_improves_own"] = int(math.isfinite(best_ref) and nll < best_ref)
            row["KAN_beats_MLP_matched"] = int(math.isfinite(best_mlp) and nll < best_mlp)
            row["KAN_beats_MLP_MCGA_winner"] = row["KAN_beats_MLP_matched"]
            row["KAN_beats_best_KAN_control"] = int(math.isfinite(best_control) and nll < best_control)
            row["KAN_beats_same_generator_controls"] = int(
                bool(control_methods)
                and all(method in same_gen and math.isfinite(same_gen[method]) and nll < same_gen[method] for method in control_methods)
            )
            row["no_ECE_Brier_tail_debt"] = int(math.isfinite(best_ref_debt) and debt <= best_ref_debt + 1.0e-9)
            row["overhead_le_035"] = int(flt(row.get("controller_overhead_ratio"), 999.0) <= 0.35)
            row["overhead_le_025"] = int(flt(row.get("controller_overhead_ratio"), 999.0) <= 0.25)
            row["active_Gram_drift_le_005"] = int(flt(row.get("active_Gram_drift_max"), 999.0) <= 0.05)
            row["functional_spectrum_drift_le_threshold"] = int(flt(row.get("functional_spectrum_drift_max"), 999.0) <= float(threshold))
            row["readout_visible_CVaR25_ge_025"] = int(flt(row.get("readout_visible_energy_CVaR25"), 0.0) >= 0.25)
            row["control_contrastive_margin_p10_positive"] = int(flt(row.get("control_contrastive_margin_p10"), -1.0) > 0.0)
            row["basis_Gram_condition_pass"] = int(math.isfinite(flt(row.get("basis_Gram_condition"))) and flt(row.get("basis_Gram_condition")) <= 1.0e8)
            row["TrueKANGain"] = int(row["KAN_improves_own"] and row["KAN_beats_MLP_matched"])
            row["BothGain"] = int(row["KAN_improves_own"] and math.isfinite(best_mlp_ref) and best_mlp < best_mlp_ref)
            row["ControlExplained"] = int(not row["KAN_beats_best_KAN_control"])
            row["MLPDegradationDriven"] = int(row["KAN_beats_MLP_matched"] and math.isfinite(best_mlp_ref) and best_mlp >= best_mlp_ref)
            row["NoGain"] = int(not row["KAN_improves_own"])
            row["TrueKANGain_plus_BothGain"] = int(row["TrueKANGain"] or row["BothGain"])
    return out


def summarize(
    rows: list[dict[str, Any]],
    threshold: float,
    ref_methods: list[str],
    candidate_methods: list[str],
    control_methods: list[str],
    mlp_methods: list[str],
) -> dict[str, Any]:
    candidates = [
        r for r in rows
        if r.get("run_status") == "completed"
        and r.get("architecture_key") == MIXED_ARCH
        and r.get("method") in candidate_methods
    ]
    n = max(1, len(candidates))

    def count(key: str) -> int:
        return sum(int(flt(r.get(key), 0.0)) for r in candidates)

    control_explained_pct = 100.0 * count("ControlExplained") / n
    mlp_degradation_pct = 100.0 * count("MLPDegradationDriven") / n
    summary = {
        "gate": "v22_69R_part_f_target_free_mixedbank_full_loop",
        "generated_at_sg": r69.now_sg(),
        "run_label": candidates[0].get("run_label", "") if candidates else "",
        "architecture": MIXED_ARCH,
        "sub_arches": "+".join(MIXED_SUB_ARCHES),
        "official_runtime_eligible": 1,
        "MLP_target_used_in_runtime": 0,
        "loss_total_is_task_loss_only_rows": sum(int(flt(r.get("loss_total_is_task_loss_only"), 0.0)) for r in candidates),
        "completed_rows": len(candidates),
        "total_rows_collected": len(rows),
        "reference_methods": ref_methods,
        "candidate_methods": candidate_methods,
        "control_methods": control_methods,
        "mlp_matched_methods": mlp_methods,
        "KAN_improves_own_rows": count("KAN_improves_own"),
        "KAN_beats_MLP_matched_rows": count("KAN_beats_MLP_matched"),
        "KAN_beats_MLP_MCGA_winner_rows": count("KAN_beats_MLP_MCGA_winner"),
        "KAN_beats_best_KAN_control_rows": count("KAN_beats_best_KAN_control"),
        "KAN_beats_same_generator_controls_rows": count("KAN_beats_same_generator_controls"),
        "TrueKANGain_plus_BothGain_rows": count("TrueKANGain_plus_BothGain"),
        "ControlExplained_pct": control_explained_pct,
        "MLPDegradationDriven_pct": mlp_degradation_pct,
        "no_debt_rows": count("no_ECE_Brier_tail_debt"),
        "overhead_le_035_rows": count("overhead_le_035"),
        "overhead_le_025_rows": count("overhead_le_025"),
        "active_Gram_drift_le_005_rows": count("active_Gram_drift_le_005"),
        "functional_spectrum_drift_le_threshold_rows": count("functional_spectrum_drift_le_threshold"),
        "readout_visible_CVaR25_ge_025_rows": count("readout_visible_CVaR25_ge_025"),
        "control_contrastive_margin_p10_positive_rows": count("control_contrastive_margin_p10_positive"),
        "basis_Gram_condition_pass_rows": count("basis_Gram_condition_pass"),
        "mean_Delta_NLL_vs_KAN_own_strong_optimizer": mean([flt(r.get("Delta_NLL_vs_KAN_own_strong_optimizer")) for r in candidates]),
        "mean_Delta_NLL_vs_best_KAN_control": mean([flt(r.get("Delta_NLL_vs_best_KAN_control")) for r in candidates]),
        "mean_Delta_NLL_vs_MLP_matched_coordinate": mean([flt(r.get("Delta_NLL_vs_MLP_matched_coordinate")) for r in candidates]),
        "control_margin_p10": percentile([flt(r.get("control_contrastive_margin_p10")) for r in candidates], 0.10),
        "readout_CVaR25_p10": percentile([flt(r.get("readout_visible_energy_CVaR25")) for r in candidates], 0.10),
    }
    gates = {
        "completed": summary["completed_rows"] >= 30,
        "KAN_improves_own": summary["KAN_improves_own_rows"] >= 16,
        "KAN_beats_MLP_matched": summary["KAN_beats_MLP_matched_rows"] >= 14,
        "KAN_beats_best_KAN_control": summary["KAN_beats_best_KAN_control_rows"] >= 16,
        "KAN_beats_same_generator_controls": summary["KAN_beats_same_generator_controls_rows"] >= 16,
        "TrueKANGain_plus_BothGain": summary["TrueKANGain_plus_BothGain_rows"] >= 8,
        "ControlExplained_pct": summary["ControlExplained_pct"] <= 40.0,
        "MLPDegradationDriven_pct": summary["MLPDegradationDriven_pct"] <= 20.0,
        "no_debt": summary["no_debt_rows"] >= 22,
        "overhead_le_035": summary["overhead_le_035_rows"] >= 24,
        "active_Gram_drift_le_005": summary["active_Gram_drift_le_005_rows"] >= 24,
        "functional_spectrum_drift": summary["functional_spectrum_drift_le_threshold_rows"] >= 24,
        "readout_visible": summary["readout_visible_CVaR25_ge_025_rows"] >= 22,
        "control_contrastive_margin": summary["control_contrastive_margin_p10_positive_rows"] >= 22,
        "basis_Gram_condition": summary["basis_Gram_condition_pass_rows"] >= 24,
    }
    summary["gate_components"] = gates
    summary["part_f_exploration_gate_pass"] = int(all(gates.values()))
    summary["failure_components"] = [k for k, v in gates.items() if not v]
    summary["final_route"] = "R6-KANCarrierCapacityOpened_Exploration" if summary["part_f_exploration_gate_pass"] else "R4-KANDiagnosticLiftOpened_TargetFreeFullLoopFailed"
    summary["official_candidate_gate_pass"] = int(
        summary["completed_rows"] >= 60
        and summary["KAN_improves_own_rows"] >= 42
        and summary["KAN_beats_MLP_matched_rows"] >= 36
        and summary["KAN_beats_MLP_MCGA_winner_rows"] >= 30
        and summary["KAN_beats_best_KAN_control_rows"] >= 42
        and summary["KAN_beats_same_generator_controls_rows"] >= 42
        and summary["TrueKANGain_plus_BothGain_rows"] >= 24
        and summary["no_debt_rows"] >= 48
        and summary["overhead_le_025_rows"] >= 48
        and summary["readout_visible_CVaR25_ge_025_rows"] >= 48
        and summary["control_contrastive_margin_p10_positive_rows"] >= 48
    )
    return summary


def run_analyze(args: argparse.Namespace) -> int:
    rows = collect_rows(str(args.run_label))
    ref_methods, candidate_methods, control_methods, mlp_methods = method_lists(args)
    annotated = annotate_candidate_rows(
        rows,
        float(args.functional_spectrum_drift_threshold),
        ref_methods,
        candidate_methods,
        control_methods,
        mlp_methods,
    )
    label = v66.safe_fragment(str(args.run_label))
    artifact_prefix = f"{OUT_PREFIX}_{label}" if label else OUT_PREFIX
    detail_path = r69.OUT_ROOT / f"{artifact_prefix}_detail.csv"
    rows_path = r69.OUT_ROOT / f"{artifact_prefix}_rows.csv"
    summary_path = r69.OUT_ROOT / f"{artifact_prefix}_summary.json"
    write_rows(rows_path, annotated)
    candidate_detail = [
        r for r in annotated
        if r.get("architecture_key") == MIXED_ARCH and r.get("method") in candidate_methods and r.get("run_status") == "completed"
    ]
    write_rows(detail_path, candidate_detail)
    summary = summarize(annotated, float(args.functional_spectrum_drift_threshold), ref_methods, candidate_methods, control_methods, mlp_methods)
    summary["rows_path"] = str(rows_path.relative_to(ROOT))
    summary["detail_path"] = str(detail_path.relative_to(ROOT))
    summary["summary_path"] = str(summary_path.relative_to(ROOT))
    r69.write_json(summary_path, summary)
    r69.append_exec(
        "F_target_free_mixedbank_analyze",
        r69.command_text([sys.executable, r69.rel(Path(__file__)), "--mode", "analyze", "--run-label", str(args.run_label)]),
        "pass" if summary["part_f_exploration_gate_pass"] else "fail",
        gpu="cpu",
        files=f"{r69.rel(rows_path)}; {r69.rel(detail_path)}; {r69.rel(summary_path)}",
        note=json.dumps({k: summary[k] for k in ["final_route", "completed_rows", "part_f_exploration_gate_pass", "failure_components"]}, ensure_ascii=False),
    )
    r69.append_recap(
        "Part F target-free mixed-bank full-loop exploration",
        [
            f"run_label={args.run_label}; official_runtime_eligible=1; MLP_target_used_in_runtime=0; completed_candidate_rows={summary['completed_rows']}; total_rows_collected={summary['total_rows_collected']}.",
            "Gate counts: improves_own={own}, beats_MLP_matched={mlp}, beats_best_KAN_control={control}, beats_same_generator_controls={same}, TrueKANGain+BothGain={true}, no_debt={debt}, overhead<=0.35={overhead}, Gram<=0.05={gram}, spectrum<=threshold={spectrum}, readout={readout}, margin={margin}, basis_cond={basis}.".format(
                own=summary["KAN_improves_own_rows"],
                mlp=summary["KAN_beats_MLP_matched_rows"],
                control=summary["KAN_beats_best_KAN_control_rows"],
                same=summary["KAN_beats_same_generator_controls_rows"],
                true=summary["TrueKANGain_plus_BothGain_rows"],
                debt=summary["no_debt_rows"],
                overhead=summary["overhead_le_035_rows"],
                gram=summary["active_Gram_drift_le_005_rows"],
                spectrum=summary["functional_spectrum_drift_le_threshold_rows"],
                readout=summary["readout_visible_CVaR25_ge_025_rows"],
                margin=summary["control_contrastive_margin_p10_positive_rows"],
                basis=summary["basis_Gram_condition_pass_rows"],
            ),
            f"part_f_exploration_gate_pass={summary['part_f_exploration_gate_pass']}; failure_components={summary['failure_components']}; final_route={summary['final_route']}; official_candidate_gate_pass={summary['official_candidate_gate_pass']}.",
            f"Means: Delta_NLL_vs_own={summary['mean_Delta_NLL_vs_KAN_own_strong_optimizer']}; Delta_NLL_vs_control={summary['mean_Delta_NLL_vs_best_KAN_control']}; Delta_NLL_vs_MLP={summary['mean_Delta_NLL_vs_MLP_matched_coordinate']}.",
            "Modification audited: Part F registers the Part-D fixed mixed bank as a trainable feature-concatenation KAN base and reuses v22.66 task-loss-only training; no MLP target artifact is loaded.",
            f"Evidence: `{r69.rel(rows_path)}`, `{r69.rel(detail_path)}`, `{r69.rel(summary_path)}`.",
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="matrix", choices=["row", "matrix", "analyze"])
    p.add_argument("--architecture", default=MIXED_ARCH)
    p.add_argument("--dataset", default="Wine")
    p.add_argument("--datasets", default="Wine,Spam,MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--method", default=DEFAULT_KAN_CANDIDATE_METHODS[0])
    p.add_argument("--reference-methods", default=",".join(DEFAULT_KAN_REFERENCE_METHODS))
    p.add_argument("--candidate-methods", default=",".join(DEFAULT_KAN_CANDIDATE_METHODS))
    p.add_argument("--control-methods", default=",".join(DEFAULT_KAN_CONTROL_METHODS))
    p.add_argument("--mlp-matched-methods", default=",".join(DEFAULT_MLP_MATCHED_METHODS))
    p.add_argument("--run-label", default="v22_69R_part_f_mixedbank_st30")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=900)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--held-size", type=int, default=64)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--metric-batch-size", type=int, default=96)
    p.add_argument("--refresh", type=int, default=100)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--iso-eta", type=float, default=1.0)
    p.add_argument("--shaping-budget", type=float, default=0.05)
    p.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "row":
        return run_row(args)
    if args.mode == "matrix":
        return run_matrix(args)
    return run_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
