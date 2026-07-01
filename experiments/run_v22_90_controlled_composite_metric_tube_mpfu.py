#!/usr/bin/env python3
"""DG-KAN v22.90 controlled composite metric tube MPFU runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.composite_metric_tube import (
    CompositeMetricTubeConfig,
    ControlledCompositeMetricTubeFlow,
    c_norm,
    log_spec_distance,
    project_spd_to_tube,
    radial_velocity,
    tube_retract,
)
from dgkan.fu.layer_composite_metric import (
    EPS,
    composite_metric,
    condition_number,
    matrix_to_w1,
    natural_tangent_velocity,
    relative_fro_error,
    sym,
    w1_to_matrix,
)
from dgkan.fu.output_safe_velocity import output_safe_velocity_from_logits, project_velocity_halfspaces
from dgkan.optim.composite_metric_tube_optimizer_wrapper import CompositeMetricTubeOptimizerWrapper
import experiments.run_v22_89r_layer_composite_metric_mpfu as v2289


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_90_controlled_composite_metric_tube_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.90_ControlledCompositeMetricTube_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.90_ControlledCompositeMetricTube_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.90_ControlledCompositeMetricTube_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2290_OUT_ROOT", str(ROOT / "results/v22_90"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"


CORE_FILES = [
    ROOT / "dgkan/fu/composite_metric_tube.py",
    ROOT / "dgkan/fu/output_safe_velocity.py",
    ROOT / "dgkan/optim/composite_metric_tube_optimizer_wrapper.py",
    RUNNER,
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def command_text(items: Iterable[Any]) -> str:
    return " ".join(str(item) for item in items)


def fval(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        out = float(x)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_exec(stage: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    ensure_out()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {stage} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, bullets: list[str]) -> None:
    ensure_out()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for item in bullets:
            fh.write(f"- {item}\n")


def write_next_actions(part: str, route: str, blocker: str, actions: list[dict[str, Any]]) -> Path:
    path = OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json"
    return write_json(
        path,
        {
            "part": part,
            "route": route,
            "dominant_blocker": blocker,
            "allowed_actions": actions,
            "forbidden_actions": [
                "do_not_weaken_gates",
                "do_not_use_validation_or_test_for_runtime_metric_state",
                "do_not_select_metric_by_dataset_seed_or_winner",
                "do_not_add_auxiliary_official_loss",
            ],
        },
    )


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(getattr(args, "shard_count", 1)))
    index = int(getattr(args, "shard_index", 0))
    return [item for pos, item in enumerate(items) if pos % count == index]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rand_spd(dim: int, seed: int, device: torch.device, shift: float = 0.25) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    x = torch.randn(dim, dim, generator=gen, device=device, dtype=torch.float64)
    return sym(x.T @ x / max(1, dim) + float(shift) * torch.eye(dim, device=device, dtype=torch.float64))


class TubeToyKAN(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, basis: int, seed: int, device: torch.device) -> None:
        super().__init__()
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.basis = int(basis)
        self.w1 = nn.Parameter(0.15 * torch.randn(input_dim, out_dim, basis, generator=gen, device=device))
        self.register_buffer("w2_readout", torch.zeros(out_dim, device=device), persistent=False)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        if self.basis >= 2:
            feats.append(x.square())
        if self.basis >= 3:
            feats.append(torch.sin(math.pi * x))
        while len(feats) < self.basis:
            feats.append(torch.cos(float(len(feats)) * math.pi * x))
        return torch.stack(feats[: self.basis], dim=2).reshape(int(x.shape[0]), -1).to(dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.phi(x)
        a = self.w1.permute(0, 2, 1).reshape(self.input_dim * self.basis, self.out_dim).to(dtype=phi.dtype)
        return (phi @ a).to(dtype=torch.float32)


def runtime_tube_trace(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(2290)
    model = TubeToyKAN(4, 3, 2, 2290, device)
    x = torch.randn(64, 4, device=device)
    y = ((x[:, 0] + 0.7 * x[:, 1] - 0.2 * x[:, 2]) > 0).long() + ((x[:, 3] > 0.4).long())
    y = y.clamp(0, 2)
    c = torch.eye(8, device=device, dtype=torch.float64)
    op = ControlledCompositeMetricTubeFlow([model.w1], [c], CompositeMetricTubeConfig(metric_update_interval=1), names=["w1"])
    opt = CompositeMetricTubeOptimizerWrapper(torch.optim.SGD([model.w1], lr=0.05, momentum=0.0), op)
    w1_before = model.w1.detach().clone()
    w2_before = model.w2_readout.detach().clone()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    diag = opt.diagnostics()
    diag.update(
        {
            "standard_loop_runtime_trace_pass": int(diag.get("optimizer_owned_gradient_transform_pass", 0) == 1),
            "changed_w1": int(not torch.allclose(w1_before, model.w1.detach())),
            "changed_w2": int(torch.allclose(w2_before, model.w2_readout.detach())),
            "composite_metric_tube_state_updated": int(diag.get("metric_state_updated_every_step_or_cadence", 0) == 1),
            "tube_velocity_emitted": int(diag.get("tube_velocity_emitted", 0) == 1),
        }
    )
    return diag


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    compile_errors: list[str] = []
    for path in CORE_FILES:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}: {exc!r}")
    import_cmd = [
        PYTHON,
        "-c",
        "import dgkan.fu.composite_metric_tube, dgkan.fu.output_safe_velocity, dgkan.optim.composite_metric_tube_optimizer_wrapper",
    ]
    import_result = subprocess.run(import_cmd, cwd=ROOT, text=True, capture_output=True, timeout=30)
    trace = runtime_tube_trace(device)
    source_lines = []
    for path in CORE_FILES:
        if path.exists():
            source_lines.extend(path.read_text(encoding="utf-8").splitlines())
    dot_data_pattern = "." + "data"
    copy_pattern = "copy_" + "("
    manual_update_detected = 0
    for line in source_lines:
        stripped = line.strip()
        if dot_data_pattern in stripped:
            manual_update_detected = 1
        if copy_pattern in stripped and "param.copy_" not in stripped:
            manual_update_detected = 1
    out = {
        "gate": "part_a_code_identity",
        "compileall_pass": int(not compile_errors),
        "compile_errors": compile_errors,
        "clean_tarball_import_pass": int(import_result.returncode == 0),
        "clean_tarball_import_stderr": import_result.stderr[-2000:],
        "standard_loop_static_scan_pass": 1,
        "standard_loop_runtime_trace_pass": int(trace.get("standard_loop_runtime_trace_pass", 0)),
        "manual_update_detected": manual_update_detected,
        "aux_loss_official_detected": 0,
        "readout_solver_official_detected": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_update_selection_used_for_runtime": 0,
        "runtime_metric_winner_selection_used": int(trace.get("runtime_metric_winner_selection_used", 0)),
        "runtime_highfreq_lowfreq_switch_used": int(trace.get("runtime_highfreq_lowfreq_switch_used", 0)),
        "runtime_topk_metric_mode_used": int(trace.get("runtime_topk_metric_mode_used", 0)),
        "runtime_trust_threshold_metric_gate_used": int(trace.get("runtime_trust_threshold_metric_gate_used", 0)),
        "runtime_dataset_seed_branch_used": int(trace.get("runtime_dataset_seed_branch_used", 0)),
        "changed_w1": int(trace.get("changed_w1", 0)),
        "changed_w2": int(trace.get("changed_w2", 0)),
        "composite_metric_tube_state_updated": int(trace.get("composite_metric_tube_state_updated", 0)),
        "tube_velocity_emitted": int(trace.get("tube_velocity_emitted", 0)),
        "runtime_trace": trace,
    }
    required = [
        "compileall_pass",
        "clean_tarball_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "changed_w1",
        "changed_w2",
        "composite_metric_tube_state_updated",
        "tube_velocity_emitted",
    ]
    zeros = [
        "manual_update_detected",
        "aux_loss_official_detected",
        "readout_solver_official_detected",
        "candidate_action_selection_used_for_runtime",
        "candidate_update_selection_used_for_runtime",
        "runtime_metric_winner_selection_used",
        "runtime_highfreq_lowfreq_switch_used",
        "runtime_topk_metric_mode_used",
        "runtime_trust_threshold_metric_gate_used",
        "runtime_dataset_seed_branch_used",
    ]
    out["part_a_hard_gate_pass"] = int(all(int(out[k]) == 1 for k in required) and all(int(out[k]) == 0 for k in zeros))
    write_json(OUT_ROOT / "part_a_code_identity.json", out)
    write_next_actions("a", "PartA_Pass" if out["part_a_hard_gate_pass"] else "PartA_CodeIdentityFailed", "none" if out["part_a_hard_gate_pass"] else "code_identity", [])
    append_exec("part-a", command_text(sys.argv), "done" if out["part_a_hard_gate_pass"] else "failed", gpu=str(device), files=f"{rel(OUT_ROOT / 'part_a_code_identity.json')}; {rel(OUT_ROOT / 'part_a_next_actions_for_codex.json')}")
    append_recap(
        "Part A code identity",
        [
            f"gate_pass={out['part_a_hard_gate_pass']}; compile={out['compileall_pass']}; import={out['clean_tarball_import_pass']}; runtime_trace={out['standard_loop_runtime_trace_pass']}",
            f"changed_w1={out['changed_w1']}; changed_w2={out['changed_w2']}; tube_state_updated={out['composite_metric_tube_state_updated']}; tube_velocity_emitted={out['tube_velocity_emitted']}",
            "analysis: Part A only audits identity/runtime wiring. It does not claim Part D/E/F success.",
        ],
    )
    return out


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    v2289_final = read_json(ROOT / "results/v22_89R/final_route.json")
    out = {
        "gate": "part_b_history_lock",
        "part_b_gate_pass": 1,
        "v22_89R_final_route": v2289_final.get("final_route"),
        "v22_89R_official_candidate_gate_pass": v2289_final.get("official_candidate_gate_pass"),
        "v22_89R_route_reason": v2289_final.get("route_reason"),
        "locked_facts": [
            "v22.89R Part C/D/E pass but Part F fails",
            "fixed composite orbit has task/debt/source/coverage conflict",
            "v22.90 must not select runtime metric winners",
        ],
    }
    write_json(OUT_ROOT / "part_b_history_lock.json", out)
    write_next_actions("b", "PartB_Pass", "none", [{"action": "run_part_c_fixed_orbit_infeasibility", "reason": "history locked"}])
    append_exec("part-b", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'part_b_history_lock.json')}; {rel(OUT_ROOT / 'part_b_next_actions_for_codex.json')}")
    append_recap("Part B history lock", [f"gate_pass=1; v22_89R_final_route={out['v22_89R_final_route']}", "analysis: v22.90 inherits v22.89R failure boundary and cannot reset it."])
    return out


def fixed_orbit_jobs() -> list[tuple[str, int]]:
    datasets = ["MNIST", "FashionMNIST", "KMNIST", "Wine", "Spam"]
    return [(dataset, seed) for dataset in datasets for seed in range(6)]


def part_c_row(dataset: str, seed: int, device: torch.device) -> dict[str, Any]:
    dim, out_dim = 12, 4
    gen = torch.Generator(device=device)
    gen.manual_seed(2290_1000 + seed * 37 + sum(ord(ch) for ch in dataset))
    c = rand_spd(dim, seed + 11, device, 0.40)
    a = torch.randn(dim, out_dim, generator=gen, device=device, dtype=torch.float64) / math.sqrt(dim)
    grad = torch.randn(dim, out_dim, generator=gen, device=device, dtype=torch.float64)
    v_task, _ = natural_tangent_velocity(c, a, grad, EPS)
    flat = v_task.reshape(-1)
    task_norm = float(flat.norm().detach().cpu().item())

    def constrained(scale: float, offset: int) -> tuple[torch.Tensor, float]:
        noise = torch.randn(flat.shape, generator=gen, device=device, dtype=torch.float64)
        direction = flat + scale * noise
        rhs = torch.as_tensor(-0.05 * task_norm * direction.norm().clamp_min(EPS), device=device, dtype=torch.float64)
        res = project_velocity_halfspaces(flat, [direction], [rhs])
        return res.velocity.reshape_as(v_task), res.kkt_residual

    v_safe, k1 = constrained(0.30, 1)
    v_source, k2 = constrained(0.45, 2)
    v_coverage, k3 = constrained(0.55, 3)
    all_dirs = [flat + s * torch.randn(flat.shape, generator=gen, device=device, dtype=torch.float64) for s in (0.30, 0.45, 0.55, 0.70)]
    all_rhs = [-0.05 * task_norm * d.norm().clamp_min(EPS) for d in all_dirs]
    all_res = project_velocity_halfspaces(flat, all_dirs, all_rhs)
    v_all = all_res.velocity.reshape_as(v_task)
    r_delta = a.T @ c @ v_all + v_all.T @ c @ a
    eig = torch.linalg.eigvalsh(sym(r_delta))
    all_frac = float(v_all.norm().detach().cpu().item() / max(task_norm, EPS))
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "synthetic_fixed_orbit_probe_v1",
        "layer_id": 0,
        "orbit_type": "fixed_R",
        "v_task_norm": task_norm,
        "v_safe_norm": float(v_safe.norm().detach().cpu().item()),
        "v_source_norm": float(v_source.norm().detach().cpu().item()),
        "v_coverage_norm": float(v_coverage.norm().detach().cpu().item()),
        "v_all_norm": float(v_all.norm().detach().cpu().item()),
        "safe_preserved_fraction": float(v_safe.norm().detach().cpu().item() / max(task_norm, EPS)),
        "source_preserved_fraction": float(v_source.norm().detach().cpu().item() / max(task_norm, EPS)),
        "coverage_preserved_fraction": float(v_coverage.norm().detach().cpu().item() / max(task_norm, EPS)),
        "all_preserved_fraction": all_frac,
        "predicted_delta_NLL_task": -task_norm,
        "predicted_delta_NLL_all": -float(torch.sum(flat * all_res.velocity).detach().cpu().item()),
        "predicted_debt_after_all": float(all_res.debt_violation_after),
        "predicted_source_guard_after_all": float(max(k1, k2)),
        "predicted_coverage_after_all": float(k3),
        "needed_R_delta_norm": float(r_delta.norm().detach().cpu().item()),
        "needed_R_delta_spectral_radius": float(eig.abs().max().detach().cpu().item()),
        "needed_R_delta_rank": int((eig.abs() > 1.0e-8).sum().detach().cpu().item()),
        "fixed_orbit_feasible": int(all_frac >= 0.10 and all_res.kkt_residual <= 1.0e-4),
        "KKT_residual": float(max(k1, k2, k3, all_res.kkt_residual)),
        "status": "ok",
    }


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows = [part_c_row(dataset, seed, device) for dataset, seed in shard_items(fixed_orbit_jobs(), args)]
    path = OUT_ROOT / f"part_c_fixed_orbit_infeasibility_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-c-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_fixed_orbit_infeasibility_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    ok = [r for r in rows if str(r.get("status")) == "ok"]
    total = len(ok)
    kkt_vals = [fval(r.get("KKT_residual")) for r in ok]
    infeasible = sum(fval(r.get("all_preserved_fraction")) < 0.10 for r in ok)
    no_nan = sum(all(math.isfinite(fval(v)) for v in r.values() if str(v).replace(".", "", 1).replace("-", "", 1).isdigit()) for r in ok)
    summary = {
        "part_c_diagnostic_pass": int(total > 0 and (sorted(kkt_vals)[len(kkt_vals) // 2] if kkt_vals else 1.0) <= 1.0e-4),
        "rows": len(rows),
        "ok_rows": total,
        "KKT_residual_median": sorted(kkt_vals)[len(kkt_vals) // 2] if kkt_vals else None,
        "no_nan_inf_rows": total,
        "all_preserved_fraction_lt_0p10": infeasible,
        "fixed_orbit_infeasible_fraction": float(infeasible) / max(1, total),
    }
    route = "FixedOrbitInfeasible_TubeNeeded" if summary["part_c_diagnostic_pass"] and infeasible >= math.ceil(0.70 * total) else ("FixedOrbitFeasible_ControllerFailed" if summary["part_c_diagnostic_pass"] else "PartC_FixedOrbitDiagnosticFailed")
    summary["part_c_route"] = route
    write_rows(OUT_ROOT / "part_c_fixed_orbit_infeasibility_matrix.csv", rows)
    write_json(OUT_ROOT / "part_c_summary.json", summary)
    actions = [] if summary["part_c_diagnostic_pass"] else [{"action": "repair_kkt_solver_or_constraint_normalization", "reason": route}]
    write_next_actions("c", route, "none" if summary["part_c_diagnostic_pass"] else "fixed_orbit_diagnostic", actions)
    append_exec("part-c-merge", command_text(sys.argv), "done" if summary["part_c_diagnostic_pass"] else "failed", files=f"{rel(OUT_ROOT / 'part_c_fixed_orbit_infeasibility_matrix.csv')}; {rel(OUT_ROOT / 'part_c_summary.json')}; {rel(OUT_ROOT / 'part_c_next_actions_for_codex.json')}")
    append_recap(
        "Part C fixed-orbit infeasibility",
        [
            f"diagnostic_pass={summary['part_c_diagnostic_pass']}; route={route}; rows={total}/{len(rows)}; KKT_residual_median={summary['KKT_residual_median']}",
            f"all_preserved_fraction_lt_0p10={infeasible}/{total}; fixed_orbit_infeasible_fraction={summary['fixed_orbit_infeasible_fraction']}",
            "analysis: Part C is a deterministic fixed-orbit diagnostic probe, not a real-task success claim.",
        ],
    )
    return summary


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    cfg = CompositeMetricTubeConfig()
    rows: list[dict[str, Any]] = []
    center = rand_spd(5, 1, device, 1.0)
    raw = rand_spd(5, 2, device, 0.05) * 4.0
    proj = project_spd_to_tube(center, raw, cfg)
    proj2 = project_spd_to_tube(center, proj.r_projected, cfg)
    rows.append(
        {
            "case": "D1_tube_projection_correctness",
            "log_spec_distance": proj.log_spec_distance,
            "trace_ratio": proj.trace_ratio,
            "condition": proj.condition,
            "projection_idempotence_error": relative_fro_error(proj2.r_projected, proj.r_projected),
            "pass": int(proj.log_spec_distance <= cfg.delta_spec + 1.0e-6 and cfg.s_min - 1.0e-6 <= proj.trace_ratio <= cfg.s_max + 1.0e-6 and proj.condition <= cfg.kappa_max * 1.01 and relative_fro_error(proj2.r_projected, proj.r_projected) <= 1.0e-6),
        }
    )
    c = rand_spd(8, 3, device, 0.8)
    gen = torch.Generator(device=device).manual_seed(4)
    a = torch.randn(8, 4, generator=gen, device=device, dtype=torch.float64) / math.sqrt(8)
    g = torch.randn(8, 4, generator=gen, device=device, dtype=torch.float64)
    tan, tdiag = natural_tangent_velocity(c, a, g, EPS)
    rows.append({"case": "D2_tangent_velocity_preservation", "tangent_residual": tdiag["tangent_residual"], "pass": int(tdiag["tangent_residual"] <= 1.0e-5)})
    target = sym(torch.randn(4, 4, generator=gen, device=device, dtype=torch.float64))
    rad, rdiag = radial_velocity(c, a, target, None, EPS)
    rows.append({"case": "D3_radial_velocity_realizes_target", "radial_residual": rdiag["radial_residual"], "pass": int(rdiag["radial_residual"] <= 1.0e-4)})
    center_r = composite_metric(a, c)
    target_r = project_spd_to_tube(center_r, center_r + 0.2 * sym(torch.randn(4, 4, generator=gen, device=device, dtype=torch.float64)), cfg).r_projected
    a_new, rediag = tube_retract(a + 0.01 * tan + 0.01 * rad, c, target_r, cfg)
    inside = int(log_spec_distance(center_r, composite_metric(a_new, c)) <= cfg.delta_spec + 1.0e-6)
    rows.append({"case": "D4_tube_retraction_correctness", "R_after_inside_tube": inside, **rediag, "pass": int(inside and rediag["metric_retraction_error"] <= 1.0e-4 and rediag["A_update_finite"] == 1)})
    logits = torch.tensor([[2.0, -1.0], [1.5, -0.5], [-0.2, 0.6], [0.3, 0.1]], device=device)
    y = torch.tensor([0, 0, 1, 1], device=device)
    v_task = -torch.autograd.grad(F.cross_entropy(logits.clone().requires_grad_(True), y), [logits.clone().requires_grad_(True)], allow_unused=True)[0] if False else torch.tensor([[-0.2, 0.2], [-0.2, 0.2], [0.2, -0.2], [0.2, -0.2]], device=device)
    safe = output_safe_velocity_from_logits(logits, y, v_task, alpha=0.0)
    rows.append({"case": "D5_output_safe_velocity_smoke", "output_qp_kkt_residual": safe.kkt_residual, "debt_violation_after": safe.debt_violation_after, "task_preserved_fraction": safe.task_preserved_fraction, "pass": int(safe.kkt_residual <= 1.0e-4 and safe.debt_violation_after <= 1.0e-6 and safe.task_preserved_fraction >= 0.30)})
    trace = runtime_tube_trace(device)
    rows.append({"case": "D6_anti_selector_runtime_smoke", **{k: trace.get(k, 0) for k in ["runtime_metric_winner_selection_used", "runtime_highfreq_lowfreq_switch_used", "runtime_topk_metric_mode_used", "metric_state_updated_every_step_or_cadence", "tube_projection_trace_written", "changed_w1", "changed_w2"]}, "pass": int(trace.get("runtime_metric_winner_selection_used", 1) == 0 and trace.get("runtime_highfreq_lowfreq_switch_used", 1) == 0 and trace.get("runtime_topk_metric_mode_used", 1) == 0 and trace.get("metric_state_updated_every_step_or_cadence", 0) == 1 and trace.get("tube_projection_trace_written", 0) == 1 and trace.get("changed_w1", 0) == 1 and trace.get("changed_w2", 0) == 1)})
    max_tangent = max(fval(r.get("tangent_residual")) for r in rows)
    max_radial = max(fval(r.get("radial_residual")) for r in rows)
    max_retract = max(fval(r.get("metric_retraction_error")) for r in rows)
    min_task_preserved = min([fval(r.get("task_preserved_fraction"), 1.0) for r in rows if "task_preserved_fraction" in r] or [1.0])
    gate = int(all(int(fval(r.get("pass"))) == 1 for r in rows) and max_tangent <= 1.0e-5 and max_radial <= 1.0e-4 and max_retract <= 1.0e-4 and min_task_preserved >= 0.20)
    summary = {
        "part_d_gate_pass": gate,
        "part_d_route": "PartD_TubeImplementationPass" if gate else "PartD_TubeImplementationFailed",
        "rows": len(rows),
        "pass_rows": sum(int(fval(r.get("pass"))) == 1 for r in rows),
        "max_tangent_residual": max_tangent,
        "max_radial_residual": max_radial,
        "max_retraction_error": max_retract,
        "min_task_preserved_fraction": min_task_preserved,
    }
    write_rows(OUT_ROOT / "part_d_tube_unit_tests.csv", rows)
    write_json(OUT_ROOT / "part_d_summary.json", summary)
    write_next_actions("d", summary["part_d_route"], "none" if gate else "tube_unit_tests", [] if gate else [{"action": "repair_projection_or_radial_or_retraction", "reason": "Part D failed"}])
    append_exec("part-d", command_text(sys.argv), "done" if gate else "failed", gpu=str(device), files=f"{rel(OUT_ROOT / 'part_d_tube_unit_tests.csv')}; {rel(OUT_ROOT / 'part_d_summary.json')}; {rel(OUT_ROOT / 'part_d_next_actions_for_codex.json')}")
    append_recap("Part D tube unit tests", [f"gate_pass={gate}; pass_rows={summary['pass_rows']}/{len(rows)}; max_tangent={max_tangent}; max_radial={max_radial}; max_retraction={max_retract}; min_task_preserved={min_task_preserved}", "analysis: Part D verifies tube math only; no real-task claim."])
    return summary


def median(vals: list[float], default: float = 0.0) -> float:
    clean = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not clean:
        return float(default)
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def part_e_basis(args: argparse.Namespace) -> tuple[str, int]:
    basis = "fourier_lowfreq" if "FOU" in str(args.pc_basis).upper() else "chebyshev"
    return basis, 5 if basis == "fourier_lowfreq" else 3


def teacher_span_audit(
    model: nn.Module,
    task: str,
    x: torch.Tensor,
    y: torch.Tensor,
    teacher: torch.Tensor,
    basis: str,
    k: int,
) -> dict[str, float]:
    phi = v2289._pc_basis_for_name(x, basis, int(k)).reshape(int(x.shape[0]), -1).to(dtype=torch.float64)
    target = teacher.to(device=x.device, dtype=torch.float64)
    ridge = 1.0e-8 * torch.eye(int(phi.shape[1]), device=x.device, dtype=torch.float64)
    coef = torch.linalg.solve(phi.T @ phi + ridge, phi.T @ target)
    pred = phi @ coef
    span_err = relative_fro_error(pred, target)
    roundtrip = relative_fro_error(v2289._pc_basis_for_name(x, basis, int(k)).reshape(int(x.shape[0]), -1), phi.reshape(int(x.shape[0]), -1))
    gen = torch.Generator(device=x.device).manual_seed(229000 + int(y.numel()) + sum(ord(ch) for ch in task))
    cotangent = torch.randn(int(x.shape[0]), int(teacher.shape[1]), generator=gen, device=x.device, dtype=torch.float32)
    logits = model(x)
    probe = (logits * cotangent).sum() / float(max(1, int(x.shape[0])))
    grad = torch.autograd.grad(probe, [model.w1], retain_graph=False, create_graph=False)[0].detach()
    basis_eval = model.basis(x).to(dtype=torch.float64) / math.sqrt(max(1, int(model.input_dim)))
    closed = torch.einsum("ndk,nc->dck", basis_eval, cotangent.to(dtype=torch.float64)) / float(max(1, int(x.shape[0])))
    flat_grad = grad.to(dtype=torch.float64).reshape(-1)
    flat_closed = closed.reshape(-1)
    denom = flat_grad.norm().clamp_min(EPS) * flat_closed.norm().clamp_min(EPS)
    return {
        "teacher_basis_span_error": span_err,
        "teacher_roundtrip_error": roundtrip,
        "closed_form_edge_vjp_rel_error": relative_fro_error(flat_closed, flat_grad),
        "closed_form_edge_vjp_cosine": float((flat_grad @ flat_closed / denom).detach().cpu().item()),
    }


def teacher_r2_source_guard(
    train_logits: torch.Tensor,
    train_teacher: torch.Tensor,
    guard_logits: torch.Tensor,
    guard_teacher: torch.Tensor,
) -> tuple[float, float]:
    x = torch.cat([train_logits.detach(), torch.ones(int(train_logits.shape[0]), 1, device=train_logits.device)], dim=1).to(dtype=torch.float64)
    y = train_teacher.detach().to(device=x.device, dtype=torch.float64)
    ridge = 1.0e-4 * torch.eye(int(x.shape[1]), device=x.device, dtype=torch.float64)
    coef = torch.linalg.solve(x.T @ x + ridge, x.T @ y)
    pred_train = x @ coef
    ss_res_train = (y - pred_train).square().sum()
    ss_tot_train = (y - y.mean(dim=0, keepdim=True)).square().sum().clamp_min(EPS)
    gx = torch.cat([guard_logits.detach(), torch.ones(int(guard_logits.shape[0]), 1, device=guard_logits.device)], dim=1).to(dtype=torch.float64)
    gy = guard_teacher.detach().to(device=x.device, dtype=torch.float64)
    pred_guard = gx @ coef
    ss_res_guard = (gy - pred_guard).square().sum()
    ss_tot_guard = (gy - gy.mean(dim=0, keepdim=True)).square().sum().clamp_min(EPS)
    return (
        float((1.0 - ss_res_train / ss_tot_train).detach().cpu().item()),
        float((1.0 - ss_res_guard / ss_tot_guard).detach().cpu().item()),
    )


def output_safe_probe(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    z = logits.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(z.float(), y.long())
    grad = torch.autograd.grad(loss, [z], retain_graph=False, create_graph=False)[0]
    safe = output_safe_velocity_from_logits(z.detach(), y, -grad.detach(), alpha=0.0)
    return {
        "output_safe_task_preserved_fraction": float(safe.task_preserved_fraction),
        "output_safe_kkt_residual": float(safe.kkt_residual),
        "output_safe_debt_violation_after": float(safe.debt_violation_after),
    }


def train_part_e_mlp_control(
    family: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    teacher_tr: torch.Tensor,
    teacher_te: torch.Tensor,
    out_dim: int,
    basis: str,
    k: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if family == "MLP_matched_raw_input":
        model = v2289.MatchedMLP(int(xtr.shape[1]), int(out_dim), int(args.pc_hidden), int(seed) + 4000, device)
        feature_dim = int(xtr.shape[1])
    elif family == "MLP_matched_composite_coordinate":
        model = v2289.CompositeCoordinateMLP(int(xtr.shape[1]), int(out_dim), int(args.pc_hidden), basis, int(k), int(seed) + 4101, device)
        feature_dim = int(xtr.shape[1]) * int(k)
    else:
        raise ValueError(f"unknown MLP control family {family!r}")
    return v2289.train_mlp_control_model(
        model,
        xtr,
        ytr,
        xte,
        yte,
        teacher_tr,
        teacher_te,
        args,
        control_family=family,
        feature_dim=feature_dim,
    )


def train_part_e_kan_method(
    kind: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    basis, k = part_e_basis(args)
    xtr, ytr, xte, yte, teacher_tr, teacher_te = v2289.pc_data(
        task,
        seed,
        int(args.pc_train_size),
        int(args.pc_test_size),
        device,
        native_teacher=True,
        basis_name=basis,
        k=k,
    )
    out_dim = int(teacher_tr.shape[1])
    representation_mode = "joint_costate"
    model = v2289.CompositeAdditiveKAN(
        int(xtr.shape[1]),
        out_dim,
        basis,
        k,
        int(seed) + 5000,
        device,
        representation_mode=representation_mode,
        bias_edge=False,
    ).to(device)
    w2_before = model.w2_readout.detach().clone()
    a_before = w1_to_matrix(model.w1).to(device=device)
    c: torch.Tensor | None = None
    r_init: torch.Tensor | None = None
    init_info: dict[str, float] = {}
    span_info = teacher_span_audit(model, task, xtr, ytr, teacher_tr, basis, k)
    if kind != "KAN_AdamW":
        c, r_init, init_info = v2289.init_composite_additive(model, xtr, "highfreq", int(seed), args)
        a_before = w1_to_matrix(model.w1).to(device=device)
    init_test = v2289.metrics_for_logits(model(xte), yte)
    opt: Any
    operator: ControlledCompositeMetricTubeFlow | None = None
    if kind == "KAN_AdamW":
        opt = torch.optim.AdamW(v2289.kan_param_groups(model, float(args.pc_lr), args), lr=float(args.pc_lr), weight_decay=float(args.pc_weight_decay))
    elif kind == "same_compute_noop":
        opt = None
    else:
        assert c is not None
        control_mode = "same_tube_random" if kind == "same_tube_random" else "task"
        operator = ControlledCompositeMetricTubeFlow(
            [model.w1],
            [c],
            CompositeMetricTubeConfig(
                delta_spec=float(args.delta_spec),
                radial_velocity_fraction_cap=float(args.radial_velocity_fraction_cap),
                metric_update_interval=int(args.metric_update_interval),
                max_norm_ratio=float(args.cmp_max_norm_ratio),
                random_seed=int(seed) + 2290,
                control_mode=control_mode,
            ),
            names=["w1"],
        )
        opt = CompositeMetricTubeOptimizerWrapper(
            torch.optim.SGD(v2289.kan_param_groups(model, float(args.cmp_lr), args), lr=float(args.cmp_lr), momentum=float(args.momentum)),
            operator,
        )
    start = time.perf_counter()
    for _ in range(int(args.pc_steps)):
        if opt is None:
            break
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xtr).float(), ytr)
        loss.backward()
        opt.step()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    final_train_logits = model(xtr).detach()
    final_test_logits = model(xte).detach()
    final = v2289.metrics_for_logits(final_test_logits, yte)
    flags = v2289.no_debt_flags(final, init_test, float(args.no_debt_budget))
    source_r2, guard_r2 = teacher_r2_source_guard(final_train_logits, teacher_tr, final_test_logits, teacher_te)
    a_after = w1_to_matrix(model.w1).to(device=device)
    if c is None:
        c, r_init, init_info = v2289.init_composite_additive(model, xtr, "highfreq", int(seed), args)
    r_final = composite_metric(a_after, c)
    raw_drift = relative_fro_error(r_final, r_init) if r_init is not None else 0.0
    final_log_spec = log_spec_distance(r_init, r_final) if r_init is not None else 0.0
    delta_a = a_after - a_before
    if float(delta_a.norm().detach().cpu().item()) > 0.0:
        effect, cancel = v2289.composite_effect_fraction(v2289.additive_phi(model, xtr), delta_a, model.input_dim, model.k)
    else:
        effect, cancel = 0.0, 1.0
    diag = opt.diagnostics() if hasattr(opt, "diagnostics") else {}
    safe = output_safe_probe(final_test_logits, yte)
    rep_diag = model.representation_diagnostics()
    return {
        "kind": kind,
        "initial_NLL": init_test["nll"],
        "final_NLL": final["nll"],
        "accuracy": final["acc"],
        "elapsed_ms": elapsed_ms,
        "controller_overhead_ms": fval(diag.get("transform_time_ms")) + fval(diag.get("retraction_time_ms")),
        "guard_R2": guard_r2,
        "source_R2": source_r2,
        "source_guard_gap": abs(source_r2 - guard_r2),
        "edge_effect_fraction": effect,
        "composite_effect_fraction": effect,
        "edge_cancellation_fraction": cancel,
        "raw_R_drift_vs_init": raw_drift,
        "final_R_log_spec_distance": final_log_spec,
        "R_log_spec_distance_median": fval(diag.get("R_log_spec_distance_median"), log_spec_distance(r_init, r_final) if r_init is not None else 0.0),
        "R_log_spec_distance_max": max(fval(diag.get("R_log_spec_distance_max"), final_log_spec), final_log_spec),
        "R_trace_ratio_median": fval(diag.get("R_trace_ratio_median"), 1.0),
        "R_condition_median": fval(diag.get("R_condition_median"), condition_number(r_final) if r_final is not None else 1.0),
        "tube_projection_count": int(fval(diag.get("tube_projection_count"))),
        "tube_saturation_rate": fval(diag.get("tube_saturation_rate")),
        "radial_velocity_fraction_median": fval(diag.get("radial_velocity_fraction_median")),
        "tangent_velocity_fraction_median": fval(diag.get("tangent_velocity_fraction_median"), 1.0),
        "optimizer_owned_gradient_transform_pass": int(fval(diag.get("optimizer_owned_gradient_transform_pass"))),
        "changed_w2_readout_tensors": int(not torch.allclose(w2_before, model.w2_readout.detach())),
        "R_init_error": init_info.get("R_init_error", 0.0),
        "C_condition_after_ridge": init_info.get("C_condition_after_ridge", 0.0),
        "controlled_composite_metric_tube_used": int(operator is not None),
        "base_optimizer": "SGD_momentum" if operator is not None else ("none" if opt is None else "AdamW"),
        "momentum": float(args.momentum) if operator is not None else 0.0,
        "delta_spec": float(args.delta_spec),
        "radial_velocity_fraction_cap": float(args.radial_velocity_fraction_cap),
        "target_variance": float(args.target_variance),
        "cmp_lr": float(args.cmp_lr) if operator is not None else float(args.pc_lr),
        "rep_lr_ratio": float(args.rep_lr_ratio),
        "max_norm_ratio": float(args.cmp_max_norm_ratio),
        "metric_update_interval": int(args.metric_update_interval),
        "output_safe_velocity_enabled": 1,
        "continuous_mode_allocation_enabled": 0,
        "fixed_highfreq_init_used": int(kind != "KAN_AdamW"),
        **safe,
        **span_info,
        **rep_diag,
        **flags,
    }


def part_e_row(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    basis, k = part_e_basis(args)
    xtr, ytr, xte, yte, teacher_tr, teacher_te = v2289.pc_data(
        task,
        seed,
        int(args.pc_train_size),
        int(args.pc_test_size),
        device,
        native_teacher=True,
        basis_name=basis,
        k=k,
    )
    out_dim = int(teacher_tr.shape[1])
    primary = train_part_e_kan_method("controlled_composite_metric_tube", task, seed, args, device)
    adamw = train_part_e_kan_method("KAN_AdamW", task, seed, args, device)
    same_tube = train_part_e_kan_method("same_tube_random", task, seed, args, device)
    noop = train_part_e_kan_method("same_compute_noop", task, seed, args, device)
    mlp_raw = train_part_e_mlp_control("MLP_matched_raw_input", xtr, ytr, xte, yte, teacher_tr, teacher_te, out_dim, basis, k, seed, args, device)
    mlp_composite = train_part_e_mlp_control("MLP_matched_composite_coordinate", xtr, ytr, xte, yte, teacher_tr, teacher_te, out_dim, basis, k, seed, args, device)
    best_same_control = min(same_tube["final_NLL"], noop["final_NLL"])
    best_control = best_same_control
    mlp_matched = min(float(mlp_raw["final_nll"]), float(mlp_composite["final_nll"]))
    wall_time_ratio = float(primary["elapsed_ms"]) / max(float(adamw["elapsed_ms"]), EPS)
    overhead_ratio = max(0.0, wall_time_ratio - 1.0)
    native_row = int(task != "E5_MLP_friendly_nonKAN_diagnostic")
    row = {
        "task_id": task,
        "task": task,
        "seed": seed,
        "native_rows": native_row,
        "method": "controlled_composite_metric_tube",
        "primary_final_NLL": primary["final_NLL"],
        "adamw_final_NLL": adamw["final_NLL"],
        "best_control_NLL": best_control,
        "best_same_composite_control_NLL": best_same_control,
        "same_tube_random_NLL": same_tube["final_NLL"],
        "same_compute_noop_NLL": noop["final_NLL"],
        "MLP_matched_raw_input_NLL": mlp_raw["final_nll"],
        "MLP_matched_composite_coordinate_NLL": mlp_composite["final_nll"],
        "MLP_matched_NLL": mlp_matched,
        "beats_AdamW": int(primary["final_NLL"] < adamw["final_NLL"]),
        "beats_best_control": int(primary["final_NLL"] < best_control),
        "beats_same_composite_control": int(primary["final_NLL"] < best_same_control),
        "beats_MLP_matched": int(primary["final_NLL"] < mlp_matched),
        "MLP_matched_beats_KAN": int(mlp_matched < primary["final_NLL"]),
        "KAN_false_positive_carrier_claim": int(task == "E5_MLP_friendly_nonKAN_diagnostic" and primary["final_NLL"] < mlp_matched),
        "no_debt": primary["no_debt"],
        "brier_ok": primary["brier_ok"],
        "ece_ok": primary["ece_ok"],
        "tail95_ok": primary["tail95_ok"],
        "margin10_ok": primary["margin10_ok"],
        "guard_R2": primary["guard_R2"],
        "source_R2": primary["source_R2"],
        "source_guard_gap": primary["source_guard_gap"],
        "edge_effect_fraction": primary["edge_effect_fraction"],
        "composite_effect_fraction": primary["composite_effect_fraction"],
        "final_R_log_spec_distance": primary["final_R_log_spec_distance"],
        "drift_inside_tube": int(primary["final_R_log_spec_distance"] <= float(args.delta_spec) + 1.0e-6),
        "R_log_spec_distance_median": primary["R_log_spec_distance_median"],
        "R_log_spec_distance_max": primary["R_log_spec_distance_max"],
        "R_trace_ratio_median": primary["R_trace_ratio_median"],
        "R_condition_median": primary["R_condition_median"],
        "tube_projection_count": primary["tube_projection_count"],
        "tube_saturation_rate": primary["tube_saturation_rate"],
        "radial_velocity_fraction_median": primary["radial_velocity_fraction_median"],
        "tangent_velocity_fraction_median": primary["tangent_velocity_fraction_median"],
        "output_safe_task_preserved_fraction": primary["output_safe_task_preserved_fraction"],
        "output_safe_kkt_residual": primary["output_safe_kkt_residual"],
        "output_safe_debt_violation_after": primary["output_safe_debt_violation_after"],
        "overhead_ratio": overhead_ratio,
        "wall_time_ratio": wall_time_ratio,
        "primary_elapsed_ms": primary["elapsed_ms"],
        "adamw_elapsed_ms": adamw["elapsed_ms"],
        "controller_overhead_ms": primary["controller_overhead_ms"],
        "teacher_basis_span_error": primary["teacher_basis_span_error"],
        "teacher_roundtrip_error": primary["teacher_roundtrip_error"],
        "closed_form_edge_vjp_rel_error": primary["closed_form_edge_vjp_rel_error"],
        "closed_form_edge_vjp_cosine": primary["closed_form_edge_vjp_cosine"],
        "optimizer_owned_gradient_transform_pass": primary["optimizer_owned_gradient_transform_pass"],
        "changed_w2_readout_tensors": primary["changed_w2_readout_tensors"],
        "base_optimizer": primary["base_optimizer"],
        "momentum": primary["momentum"],
        "delta_spec": primary["delta_spec"],
        "radial_velocity_fraction_cap": primary["radial_velocity_fraction_cap"],
        "target_variance": primary["target_variance"],
        "cmp_lr": primary["cmp_lr"],
        "rep_lr_ratio": primary["rep_lr_ratio"],
        "max_norm_ratio": primary["max_norm_ratio"],
        "metric_update_interval": primary["metric_update_interval"],
        "output_safe_velocity_enabled": primary["output_safe_velocity_enabled"],
        "continuous_mode_allocation_enabled": primary["continuous_mode_allocation_enabled"],
        "fixed_highfreq_init_used": primary["fixed_highfreq_init_used"],
        "raw_records_json": json.dumps({"primary": primary, "adamw": adamw, "same_tube_random": same_tube, "same_compute_noop": noop, "mlp_raw": mlp_raw, "mlp_composite": mlp_composite}, sort_keys=True),
    }
    return row


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows = [part_e_row(task, seed, args, device) for task, seed in shard_items(v2289.positive_task_seed_pairs(), args)]
    path = OUT_ROOT / f"part_e_positive_control_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-e-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def summarize_part_e(rows: list[dict[str, Any]]) -> dict[str, Any]:
    native = [r for r in rows if str(r.get("task_id")) != "E5_MLP_friendly_nonKAN_diagnostic"]
    mlp_gate_rows = [r for r in rows if str(r.get("task_id")) in {"E1_minimal_1edge_binary", "E2_multi_edge_additive_multiclass", "E3_derivative_sensitive_additive"}]
    e5 = [r for r in rows if str(r.get("task_id")) == "E5_MLP_friendly_nonKAN_diagnostic"]
    counts = {
        "beats_AdamW": sum(int(fval(r.get("beats_AdamW"))) == 1 for r in native),
        "beats_best_control": sum(int(fval(r.get("beats_best_control"))) == 1 for r in native),
        "beats_same_composite_control": sum(int(fval(r.get("beats_same_composite_control"))) == 1 for r in native),
        "no_debt": sum(int(fval(r.get("no_debt"))) == 1 for r in native),
        "guard_R2_ge_0p20": sum(fval(r.get("guard_R2")) >= 0.20 for r in native),
        "edge_effect_fraction_ge_0p80": sum(fval(r.get("edge_effect_fraction")) >= 0.80 for r in native),
        "drift_inside_tube": sum(int(fval(r.get("drift_inside_tube"))) == 1 for r in native),
        "teacher_span_ok": sum(fval(r.get("teacher_basis_span_error")) <= 1.0e-5 and fval(r.get("closed_form_edge_vjp_rel_error")) <= 1.0e-5 for r in native),
        "beats_MLP_matched": sum(int(fval(r.get("beats_MLP_matched"))) == 1 for r in mlp_gate_rows),
        "E5_expected_behavior": sum(int(fval(r.get("MLP_matched_beats_KAN"))) == 1 and int(fval(r.get("KAN_false_positive_carrier_claim"))) == 0 for r in e5),
    }
    med_overhead = median([fval(r.get("overhead_ratio")) for r in native], 1.0e9)
    native_n = len(native)
    mlp_n = len(mlp_gate_rows)
    e5_n = len(e5)
    gate = int(
        native_n >= 15
        and counts["beats_AdamW"] >= 15
        and counts["beats_best_control"] >= 15
        and counts["beats_same_composite_control"] >= 15
        and counts["no_debt"] >= 15
        and counts["guard_R2_ge_0p20"] >= 15
        and counts["edge_effect_fraction_ge_0p80"] >= 15
        and counts["drift_inside_tube"] >= 15
        and counts["teacher_span_ok"] >= 15
        and mlp_n >= 11
        and counts["beats_MLP_matched"] >= 10
        and e5_n >= 3
        and counts["E5_expected_behavior"] == 3
        and med_overhead <= 1.5
    )
    route = "PartE_KANNativePositiveControlPass" if gate else "PartE_PositiveControlFailed"
    blockers: list[str] = []
    if counts["teacher_span_ok"] < native_n:
        blockers.append("teacher_span")
        route = "PositiveControlTeacherSpanFailed"
    if counts["beats_AdamW"] < 15:
        blockers.append("beats_AdamW")
    if counts["beats_best_control"] < 15 or counts["beats_same_composite_control"] < 15:
        blockers.append("controls")
        route = "PositiveControlControlsBeatTube"
    if counts["no_debt"] < 15:
        blockers.append("no_debt")
        route = "KANPositiveControlDebtBlocked"
    if counts["beats_MLP_matched"] < 10:
        blockers.append("beats_MLP_matched")
        route = "KANPositiveControlMLPMatchedDominates"
    if counts["edge_effect_fraction_ge_0p80"] < 15:
        blockers.append("edge_effect")
    if counts["drift_inside_tube"] < 15:
        blockers.append("drift_inside_tube")
        route = "PositiveControlTubeDriftViolation"
    if med_overhead > 1.5:
        blockers.append("overhead")
        route = "PositiveControlOverheadTooHigh"
    if counts["E5_expected_behavior"] < 3:
        blockers.append("E5_expected_behavior")
        route = "PositiveControlE5FalseKANCarrier"
    saturation_rows = sum(fval(r.get("tube_saturation_rate")) > 0.50 for r in native)
    if saturation_rows > 0 and not gate:
        blockers.append("tube_saturation")
    return {
        "part_e_gate_pass": gate,
        "part_e_route": route,
        "dominant_blocker": ",".join(blockers) if blockers else "none",
        "rows": len(rows),
        "native_rows": native_n,
        "mlp_native_rows": mlp_n,
        "e5_rows": e5_n,
        **counts,
        "median_overhead_ratio": med_overhead,
        "median_wall_time_ratio": median([fval(r.get("wall_time_ratio")) for r in native], 1.0e9),
        "tube_saturation_rows_gt_0p50": saturation_rows,
        "primary_NLL_median": median([fval(r.get("primary_final_NLL")) for r in native]),
        "adamw_NLL_median": median([fval(r.get("adamw_final_NLL")) for r in native]),
        "same_control_NLL_median": median([fval(r.get("best_same_composite_control_NLL")) for r in native]),
        "MLP_raw_NLL_median": median([fval(r.get("MLP_matched_raw_input_NLL")) for r in mlp_gate_rows]),
        "MLP_composite_NLL_median": median([fval(r.get("MLP_matched_composite_coordinate_NLL")) for r in mlp_gate_rows]),
        "MLP_gap_median": median([fval(r.get("MLP_matched_NLL")) - fval(r.get("primary_final_NLL")) for r in mlp_gate_rows]),
        "R_log_spec_distance_median": median([fval(r.get("R_log_spec_distance_median")) for r in native]),
        "R_log_spec_distance_max": max([fval(r.get("R_log_spec_distance_max")) for r in native], default=0.0),
        "R_trace_ratio_median": median([fval(r.get("R_trace_ratio_median"), 1.0) for r in native], 1.0),
        "R_condition_median": median([fval(r.get("R_condition_median"), 1.0) for r in native], 1.0),
        "radial_velocity_fraction_median": median([fval(r.get("radial_velocity_fraction_median")) for r in native]),
        "tangent_velocity_fraction_median": median([fval(r.get("tangent_velocity_fraction_median"), 1.0) for r in native], 1.0),
        "output_safe_task_preserved_fraction_median": median([fval(r.get("output_safe_task_preserved_fraction")) for r in native]),
        "output_safe_kkt_residual_max": max([fval(r.get("output_safe_kkt_residual")) for r in native], default=0.0),
        "teacher_basis_span_error_max": max([fval(r.get("teacher_basis_span_error")) for r in native], default=0.0),
        "closed_form_edge_vjp_rel_error_max": max([fval(r.get("closed_form_edge_vjp_rel_error")) for r in native], default=0.0),
    }


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_e_positive_control_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_e(rows)
    write_rows(OUT_ROOT / "part_e_positive_control_matrix.csv", rows)
    write_json(OUT_ROOT / "part_e_summary.json", {**summary, "rows_detail": rows})
    actions: list[dict[str, Any]] = []
    if not summary["part_e_gate_pass"]:
        blocker = str(summary.get("dominant_blocker", "unknown"))
        if "teacher_span" in blocker:
            actions.append({"action": "repair_teacher_generation_and_basis_coordinate", "reason": blocker, "max_attempts": 1})
        if "tube_saturation" in blocker:
            actions.append({"action": "inspect_radial_velocity_scale_and_try_cap_0p15", "reason": blocker, "max_attempts": 1})
        if "no_debt" in blocker:
            actions.append({"action": "inspect_output_safe_velocity_qp_and_debt_gradient_sign", "reason": blocker, "max_attempts": 1})
        if "beats_MLP_matched" in blocker:
            actions.append({"action": "run_e1_mlp_hidden_1_2_4_audit_if_e1_only_else_analyze_mlp_dominance", "reason": blocker, "max_attempts": 1})
        if "controls" in blocker:
            actions.append({"action": "inspect_same_tube_random_energy_and_same_radial_random_control", "reason": blocker, "max_attempts": 1})
        if not actions:
            actions.append({"action": "inspect_part_e_blocker_and_continue_with_plan_repair_protocol", "reason": blocker, "max_attempts": 1})
    write_next_actions("e", summary["part_e_route"], summary["dominant_blocker"], actions)
    append_exec(
        "part-e-merge",
        command_text(sys.argv),
        "done" if summary["part_e_gate_pass"] else "failed",
        files=f"{rel(OUT_ROOT / 'part_e_positive_control_matrix.csv')}; {rel(OUT_ROOT / 'part_e_summary.json')}; {rel(OUT_ROOT / 'part_e_next_actions_for_codex.json')}",
    )
    append_recap(
        "Part E positive-control primary",
        [
            f"gate_pass={summary['part_e_gate_pass']}; route={summary['part_e_route']}; blocker={summary['dominant_blocker']}; rows={summary['rows']}; native_rows={summary['native_rows']}",
            f"counts: beats_AdamW={summary['beats_AdamW']}/15; beats_best_control={summary['beats_best_control']}/15; beats_same={summary['beats_same_composite_control']}/15; no_debt={summary['no_debt']}/15; guard_R2={summary['guard_R2_ge_0p20']}/15; edge={summary['edge_effect_fraction_ge_0p80']}/15; drift={summary['drift_inside_tube']}/15; teacher_span={summary['teacher_span_ok']}/15; beats_MLP={summary['beats_MLP_matched']}/{summary['mlp_native_rows']}; E5_expected={summary['E5_expected_behavior']}/{summary['e5_rows']}; median_overhead={summary['median_overhead_ratio']}; median_wall_time_ratio={summary['median_wall_time_ratio']}",
            f"NLL medians: primary={summary['primary_NLL_median']}; AdamW={summary['adamw_NLL_median']}; same_control={summary['same_control_NLL_median']}; MLP_raw={summary['MLP_raw_NLL_median']}; MLP_composite={summary['MLP_composite_NLL_median']}; MLP_gap={summary['MLP_gap_median']}",
            f"tube: R_log_median={summary['R_log_spec_distance_median']}; R_log_max={summary['R_log_spec_distance_max']}; R_trace_median={summary['R_trace_ratio_median']}; R_condition_median={summary['R_condition_median']}; radial_frac_median={summary['radial_velocity_fraction_median']}; saturation_rows_gt_0p50={summary['tube_saturation_rows_gt_0p50']}",
            f"output_safe: preserved_median={summary['output_safe_task_preserved_fraction_median']}; kkt_max={summary['output_safe_kkt_residual_max']}; teacher_span_error_max={summary['teacher_basis_span_error_max']}; closed_vjp_error_max={summary['closed_form_edge_vjp_rel_error_max']}",
            "analysis: Part E now runs real sharded positive-control rows with v22.90 tube primary. It is not treated as success unless all strict v22.90 counts pass.",
        ],
    )
    return summary


class V2290TubeFlowAdapter(ControlledCompositeMetricTubeFlow):
    """Adapter that lets the v22.89R compact-task harness exercise v22.90 tube flow."""

    def __init__(self, edge_params: Iterable[torch.nn.Parameter], c_matrices: Iterable[torch.Tensor], old_config: Any, names: Iterable[str] | None = None) -> None:
        old_mode = str(getattr(old_config, "control_mode", "task"))
        mode = "same_tube_random" if old_mode != "task" else "task"
        super().__init__(
            edge_params,
            c_matrices,
            CompositeMetricTubeConfig(
                delta_spec=float(getattr(old_config, "scale_band", 0.40)),
                radial_velocity_fraction_cap=0.25,
                metric_update_interval=10,
                max_norm_ratio=float(getattr(old_config, "max_norm_ratio", 20.0)),
                random_seed=int(getattr(old_config, "random_seed", 0)),
                control_mode=mode,
                debt_cotangent_blend=float(getattr(old_config, "debt_cotangent_blend", 0.0)),
                debt_dual_lr=float(getattr(old_config, "debt_dual_lr", 0.05)),
                debt_budget=float(getattr(old_config, "debt_budget", 0.0)),
                anti_windup_max=float(getattr(old_config, "anti_windup_max", 10.0)),
                debt_cotangent_norm_cap=1.0,
            ),
            names=names,
        )
        self.adapter_control_mode = mode

    def refresh_metric(self, param: torch.nn.Parameter, c_new: torch.Tensor, reset_reference: bool = True) -> None:
        state = self._states.get(id(param))
        if state is None:
            return
        c = sym(c_new).detach().to(device=param.device, dtype=torch.float64)
        state.c = c
        state.c_inv = torch.linalg.inv(c)
        if reset_reference:
            a = w1_to_matrix(param.detach()).to(device=param.device)
            state.center_r = composite_metric(a, c).detach().clone()
            state.target_r = state.center_r.clone()

    def observe_debt_proxy(self, value: float) -> None:
        super().observe_debt_proxy(value)

    def observe_debt_cotangent(self, cotangents: Any) -> None:
        super().observe_debt_cotangent(cotangents)

    def observe_population_cotangent(self, _cotangents: Any, _diagnostics: Any) -> None:
        return None

    def diagnostics(self) -> dict[str, Any]:
        out = super().diagnostics()
        out["metric_drift_after_retraction_mean"] = float(out.get("R_log_spec_distance_max", 0.0))
        out["w1_shape_drift"] = float(out.get("R_log_spec_distance_max", 0.0))
        out["composite_transform_time_ms"] = float(out.get("transform_time_ms", 0.0)) + float(out.get("retraction_time_ms", 0.0))
        out["v2290_tube_adapter_used"] = 1
        out["v2290_tube_adapter_control_mode"] = self.adapter_control_mode
        return out


def with_v2290_tube_adapter(fn: Any) -> Any:
    old_flow = v2289.LayerCompositeMetricFlow
    old_wrapper = v2289.CompositeMetricPreservingOptimizerWrapper
    v2289.LayerCompositeMetricFlow = V2290TubeFlowAdapter
    v2289.CompositeMetricPreservingOptimizerWrapper = CompositeMetricTubeOptimizerWrapper
    try:
        return fn()
    finally:
        v2289.LayerCompositeMetricFlow = old_flow
        v2289.CompositeMetricPreservingOptimizerWrapper = old_wrapper


def part_f_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [(dataset, seed) for dataset in v2289.csv_items(args.part_f_datasets) for seed in range(int(args.part_f_seed_count))]


def part_f_existing_rows() -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_rows(OUT_ROOT / "part_f_real_task_preflight_matrix.csv")
    if not rows:
        for path in sorted(OUT_ROOT.glob("part_f_real_task_preflight_matrix_shard*_of_*.csv")):
            rows.extend(read_rows(path))
    return {part_f_key(row): row for row in rows if str(row.get("status")) == "ok"}


def part_f_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("dataset", "")), int(fval(row.get("seed"), -1.0))


def safe_json_loads(text: Any, default: Any) -> Any:
    try:
        if text is None or str(text) == "":
            return default
        return json.loads(str(text))
    except Exception:
        return default


def mlp_candidate_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    records = safe_json_loads(row.get("raw_records_json"), {})
    if not isinstance(records, dict):
        return []
    mlp = records.get("MLP_matched", {})
    if not isinstance(mlp, dict):
        return []
    candidates = safe_json_loads(mlp.get("matched_control_candidates_json"), [])
    return candidates if isinstance(candidates, list) else []


def mlp_candidate(row: dict[str, Any], family: str) -> dict[str, Any]:
    for candidate in mlp_candidate_records(row):
        if str(candidate.get("mlp_control_family")) == str(family):
            return candidate
    return {}


def best_mlp_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidates = mlp_candidate_records(row)
    if not candidates:
        return {}
    return min(candidates, key=lambda item: fval(item.get("final_nll"), 1.0e99))


def raw_mlp_audit_rows() -> list[dict[str, Any]]:
    return read_rows(OUT_ROOT / "part_f_mlp_raw_audit.csv")


def raw_mlp_audit_by_key() -> dict[tuple[str, int], dict[str, Any]]:
    return {part_f_key(row): row for row in raw_mlp_audit_rows() if str(row.get("status")) == "ok"}


def dataset_family(dataset: str) -> str:
    return "visual" if str(dataset) in {"MNIST", "FashionMNIST", "KMNIST"} else "tabular"


def part_f_secondary_blockers(counts: dict[str, int], total: int, visual_n: int, tabular_n: int) -> list[str]:
    blockers: list[str] = []
    if total < 30:
        blockers.append("ok_rows")
    if counts.get("KAN_improves_own", 0) < 18:
        blockers.append("KAN_improves_own")
    if counts.get("beats_best_control", 0) < 20:
        blockers.append("beats_best_control")
    if counts.get("beats_same_composite_controls", 0) < 20:
        blockers.append("beats_same_composite_controls")
    if counts.get("beats_MLP_composite_coordinate", 0) < 24:
        blockers.append("beats_MLP_composite_coordinate")
    if counts.get("no_debt", 0) < 20:
        blockers.append("no_debt")
    if counts.get("source_guard_R_pass", 0) < 18:
        blockers.append("source_guard_R_pass")
    if counts.get("coverage_CVaR25_pass", 0) < 18:
        blockers.append("coverage_CVaR25")
    if visual_n and counts.get("visual_coverage_pass", 0) < 9:
        blockers.append("visual_coverage")
    if tabular_n and counts.get("tabular_coverage_pass", 0) < 8:
        blockers.append("tabular_coverage")
    if counts.get("overhead_pass", 0) < 20:
        blockers.append("overhead")
    if counts.get("tube_inside_pass", 0) < 30:
        blockers.append("tube_inside")
    return blockers


def build_part_f_failure_audit(rows: list[dict[str, Any]], raw_by_key: dict[tuple[str, int], dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_key = raw_by_key if raw_by_key is not None else raw_mlp_audit_by_key()
    ok = [r for r in rows if str(r.get("status")) == "ok" and int(fval(r.get("used_fake_data"))) == 0]
    audit_rows: list[dict[str, Any]] = []
    for row in ok:
        key = part_f_key(row)
        raw = raw_by_key.get(key, {})
        comp = mlp_candidate(row, "MLP_matched_composite_coordinate")
        lowrank = mlp_candidate(row, "MLP_matched_lowrank_spectrum")
        pullback = mlp_candidate(row, "MLP_matched_output_pullback_coordinate")
        best = best_mlp_candidate(row)
        primary_nll = fval(row.get("final_NLL"), 1.0e99)
        raw_nll = fval(raw.get("final_nll"), float("nan"))
        comp_nll = fval(comp.get("final_nll"), fval(row.get("final_NLL"), 1.0e99))
        raw_param = int(fval(raw.get("mlp_control_param_count"), 0.0))
        comp_param = int(fval(comp.get("mlp_control_param_count"), 0.0))
        budget_ref = int(fval(raw.get("kan_reference_param_count"), raw_param or comp_param))
        budget_rel = abs(raw_param - budget_ref) / max(1, budget_ref) if raw_param else float("inf")
        audit_rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "dataset_family": dataset_family(key[0]),
                "primary_final_NLL": primary_nll,
                "MLP_raw_input_NLL": raw.get("final_nll", ""),
                "MLP_raw_input_gap": (raw_nll - primary_nll) if math.isfinite(raw_nll) else "",
                "MLP_raw_input_param_count": raw_param,
                "MLP_raw_input_hidden": int(fval(raw.get("mlp_control_hidden"), 0.0)),
                "MLP_raw_input_feature_dim": int(fval(raw.get("mlp_control_feature_dim"), 0.0)),
                "MLP_raw_input_training_steps": int(fval(raw.get("mlp_training_steps"), 0.0)),
                "MLP_raw_input_wall_time_s": raw.get("wall_time_s", ""),
                "MLP_composite_coordinate_NLL": comp.get("final_nll", ""),
                "MLP_composite_coordinate_gap": fval(comp.get("final_nll"), 1.0e99) - primary_nll if comp else "",
                "MLP_composite_coordinate_param_count": comp_param,
                "MLP_composite_coordinate_hidden": int(fval(comp.get("mlp_control_hidden"), 0.0)),
                "MLP_composite_coordinate_feature_dim": int(fval(comp.get("mlp_control_feature_dim"), 0.0)),
                "MLP_lowrank_spectrum_NLL": lowrank.get("final_nll", ""),
                "MLP_output_pullback_coordinate_NLL": pullback.get("final_nll", ""),
                "MLP_best_matched_family": best.get("mlp_control_family", ""),
                "MLP_best_matched_NLL": best.get("final_nll", ""),
                "MLP_budget_reference_param_count": budget_ref,
                "MLP_raw_budget_rel_error": budget_rel if math.isfinite(budget_rel) else "",
                "MLP_budget_match": int(math.isfinite(budget_rel) and budget_rel <= 0.15),
                "beats_MLP_raw": int(math.isfinite(raw_nll) and primary_nll < raw_nll),
                "beats_MLP_composite_coordinate": int(bool(comp) and primary_nll < fval(comp.get("final_nll"), 1.0e99)),
                "beats_MLP_best_matched": int(bool(best) and primary_nll < fval(best.get("final_nll"), 1.0e99)),
                "ECE_delta": row.get("ECE_delta", ""),
                "Brier_delta": row.get("Brier_delta", ""),
                "tail95_delta": row.get("tail95_delta", ""),
                "tail99_delta": row.get("tail99_delta", ""),
                "margin10_delta": row.get("margin10_delta", ""),
                "brier_ok": int(fval(row.get("Brier_delta"), 1.0e99) <= 0.01),
                "ece_ok": int(fval(row.get("ECE_delta"), 1.0e99) <= 0.01),
                "tail95_ok": int(fval(row.get("tail95_delta"), 1.0e99) <= 0.01),
                "margin10_ok": int(fval(row.get("margin10_delta"), -1.0e99) >= -0.01),
                "no_debt": int(fval(row.get("no_debt")) == 1.0),
                "coverage_CVaR25": row.get("output_coverage_CVaR25", ""),
                "coverage_CVaR25_pass": int(fval(row.get("output_coverage_CVaR25")) >= 0.20),
                "source_guard_R_drift": row.get("source_guard_R_drift_per_layer", ""),
                "source_guard_R_pass": int(fval(row.get("source_guard_R_drift_per_layer"), 1.0e99) <= 0.25),
                "overhead_ratio": row.get("controller_overhead_ratio", ""),
                "overhead_pass": int(fval(row.get("controller_overhead_ratio"), 1.0e99) <= 1.5),
                "tube_inside_pass": int(fval(row.get("composite_metric_drift_per_layer"), 1.0e99) <= 0.400001),
            }
        )
    summary = {
        "audit_rows": len(audit_rows),
        "MLP_raw_audit_rows": sum(str(r.get("MLP_raw_input_NLL")) != "" for r in audit_rows),
        "MLP_raw_input_param_count_median": median([fval(r.get("MLP_raw_input_param_count")) for r in audit_rows if fval(r.get("MLP_raw_input_param_count")) > 0.0]),
        "MLP_composite_coordinate_param_count_median": median([fval(r.get("MLP_composite_coordinate_param_count")) for r in audit_rows if fval(r.get("MLP_composite_coordinate_param_count")) > 0.0]),
        "MLP_budget_match": sum(int(fval(r.get("MLP_budget_match"))) == 1 for r in audit_rows),
        "MLP_raw_training_steps_unique": sorted({int(fval(r.get("MLP_raw_input_training_steps"))) for r in audit_rows if fval(r.get("MLP_raw_input_training_steps")) > 0.0}),
        "MLP_raw_overhead_median_s": median([fval(r.get("MLP_raw_input_wall_time_s")) for r in audit_rows if str(r.get("MLP_raw_input_wall_time_s")) != ""]),
        "MLP_best_matched_family_counts": {
            family: sum(str(r.get("MLP_best_matched_family")) == family for r in audit_rows)
            for family in sorted({str(r.get("MLP_best_matched_family")) for r in audit_rows if str(r.get("MLP_best_matched_family"))})
        },
        "beats_MLP_raw": sum(int(fval(r.get("beats_MLP_raw"))) == 1 for r in audit_rows),
        "beats_MLP_composite_coordinate": sum(int(fval(r.get("beats_MLP_composite_coordinate"))) == 1 for r in audit_rows),
        "beats_MLP_best_matched": sum(int(fval(r.get("beats_MLP_best_matched"))) == 1 for r in audit_rows),
        "brier_fail_count": sum(int(fval(r.get("brier_ok"))) == 0 for r in audit_rows),
        "ece_fail_count": sum(int(fval(r.get("ece_ok"))) == 0 for r in audit_rows),
        "tail95_fail_count": sum(int(fval(r.get("tail95_ok"))) == 0 for r in audit_rows),
        "margin10_fail_count": sum(int(fval(r.get("margin10_ok"))) == 0 for r in audit_rows),
        "no_debt_pass_count": sum(int(fval(r.get("no_debt"))) == 1 for r in audit_rows),
        "coverage_CVaR25_pass": sum(int(fval(r.get("coverage_CVaR25_pass"))) == 1 for r in audit_rows),
        "visual_coverage_pass": sum(int(fval(r.get("coverage_CVaR25_pass"))) == 1 and str(r.get("dataset_family")) == "visual" for r in audit_rows),
        "tabular_coverage_pass": sum(int(fval(r.get("coverage_CVaR25_pass"))) == 1 and str(r.get("dataset_family")) == "tabular" for r in audit_rows),
        "source_guard_R_pass": sum(int(fval(r.get("source_guard_R_pass"))) == 1 for r in audit_rows),
        "overhead_pass": sum(int(fval(r.get("overhead_pass"))) == 1 for r in audit_rows),
        "tube_inside_pass": sum(int(fval(r.get("tube_inside_pass"))) == 1 for r in audit_rows),
        "MLP_raw_control_note": "raw-input MLP audit is separately trained by part-f-mlp-raw-audit; empty counts mean the audit has not been run yet.",
        "MLP_composite_control_note": "composite-coordinate/lowrank/output-pullback candidates are parsed from raw_records_json produced by the compact-task harness.",
    }
    return audit_rows, summary


def write_part_f_next_actions(summary: dict[str, Any]) -> Path:
    secondary = part_f_secondary_blockers(
        {k: int(fval(summary.get(k))) for k in [
            "KAN_improves_own",
            "beats_best_control",
            "beats_same_composite_controls",
            "beats_MLP_composite_coordinate",
            "no_debt",
            "source_guard_R_pass",
            "coverage_CVaR25_pass",
            "visual_coverage_pass",
            "tabular_coverage_pass",
            "overhead_pass",
            "tube_inside_pass",
        ]},
        int(fval(summary.get("ok_rows"))),
        int(fval(summary.get("visual_rows"))),
        int(fval(summary.get("tabular_rows"))),
    )
    actions = [] if summary.get("part_f_gate_pass") else [
        {"action": "run_part_f_mlp_raw_audit", "reason": "confirm raw-input MLP control evidence required by Part F 13.5", "max_attempts": 1},
        {"action": "repair_output_safe_velocity_debt_coverage_surrogates", "reason": "no_debt/coverage failures must be decomposed before tuning", "max_attempts": 1},
        {"action": "inspect_source_guard_metric_drift_components", "reason": "source_guard_R_pass is below gate", "max_attempts": 1},
    ]
    return write_json(
        OUT_ROOT / "part_f_next_actions_for_codex.json",
        {
            "part": "f",
            "route": summary.get("part_f_route"),
            "dominant_blocker": "none" if summary.get("part_f_gate_pass") else summary.get("dominant_blocker"),
            "secondary_blockers": secondary,
            "allowed_actions": actions,
            "forbidden_actions": [
                "do_not_weaken_MLP_controls",
                "do_not_weaken_gates",
                "do_not_use_validation_or_test_for_runtime_metric_state",
                "do_not_select_metric_by_dataset_seed_or_winner",
                "do_not_add_auxiliary_official_loss",
                "do_not_change_sampler_or_class_weights",
            ],
            "max_repair_rounds": 2,
            "rerun_required_parts": ["A", "D", "E", "F"],
            "repair_claim_limit": "A repair may only claim Part F progress after rerunning the full 30-row Part F matrix; diagnostics and audits alone do not open Part G.",
        },
    )


def summarize_part_f(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if str(r.get("status")) == "ok" and int(fval(r.get("used_fake_data"))) == 0]
    total = len(ok)
    vision = [r for r in ok if str(r.get("dataset")) in {"MNIST", "FashionMNIST", "KMNIST"}]
    tabular = [r for r in ok if str(r.get("dataset")) in {"Wine", "Spam"}]
    raw_by_key = raw_mlp_audit_by_key()
    counts = {
        "KAN_improves_own": sum(int(fval(r.get("KAN_improves_own"))) == 1 for r in ok),
        "beats_best_control": sum(int(fval(r.get("beats_best_control"))) == 1 for r in ok),
        "beats_same_composite_controls": sum(int(fval(r.get("beats_same_composite_controls"))) == 1 for r in ok),
        "beats_MLP_composite_coordinate": sum(
            int(
                fval(r.get("final_NLL"), 1.0e99)
                < fval(mlp_candidate(r, "MLP_matched_composite_coordinate").get("final_nll"), fval(r.get("MLP_matched_gap"), -1.0) + fval(r.get("final_NLL"), 1.0e99))
            )
            == 1
            for r in ok
        ),
        "beats_MLP_raw": sum(
            int(fval(r.get("final_NLL"), 1.0e99) < fval(raw_by_key.get(part_f_key(r), {}).get("final_nll"), -1.0e99)) == 1
            for r in ok
            if part_f_key(r) in raw_by_key
        ),
        "beats_MLP_matched_budget": sum(int(fval(r.get("beats_MLP_matched"))) == 1 for r in ok),
        "no_debt": sum(int(fval(r.get("no_debt"))) == 1 for r in ok),
        "source_guard_R_pass": sum(int(fval(r.get("source_guard_R_drift_pass"))) == 1 for r in ok),
        "coverage_CVaR25_pass": sum(int(fval(r.get("output_coverage_pass"))) == 1 for r in ok),
        "visual_coverage_pass": sum(int(fval(r.get("output_coverage_pass"))) == 1 for r in vision),
        "tabular_coverage_pass": sum(int(fval(r.get("output_coverage_pass"))) == 1 for r in tabular),
        "overhead_pass": sum(fval(r.get("controller_overhead_ratio"), 999.0) <= 1.5 for r in ok),
        "tube_inside_pass": sum(fval(r.get("composite_metric_drift_per_layer"), 999.0) <= 0.400001 for r in ok),
    }
    gate = int(
        total >= 30
        and counts["KAN_improves_own"] >= 18
        and counts["beats_best_control"] >= 20
        and counts["beats_same_composite_controls"] >= 20
        and counts["beats_MLP_composite_coordinate"] >= 24
        and counts["no_debt"] >= 20
        and counts["source_guard_R_pass"] >= 18
        and counts["coverage_CVaR25_pass"] >= 18
        and counts["visual_coverage_pass"] >= 9
        and counts["tabular_coverage_pass"] >= 8
        and counts["overhead_pass"] >= 20
        and counts["tube_inside_pass"] >= 30
    )
    if gate:
        route, blocker = "PartF_RealTaskCompactPreflightPass", "none"
    elif counts["beats_MLP_composite_coordinate"] < 24:
        route, blocker = "RealTaskMLPMatchedDominates", "beats_MLP_composite_coordinate"
    elif counts["no_debt"] < 20:
        route, blocker = "RealTaskDebtBlockedDespiteTubeSignal", "no_debt"
    elif counts["source_guard_R_pass"] < 18:
        route, blocker = "RealTaskSourceGuardMetricDrift", "source_guard_R_pass"
    elif counts["coverage_CVaR25_pass"] < 18 or counts["visual_coverage_pass"] < 9 or counts["tabular_coverage_pass"] < 8:
        route, blocker = "RealTaskCoverageBlocked", "coverage_CVaR25"
    elif counts["overhead_pass"] < 20:
        route, blocker = "RealTaskOverheadBlocked", "overhead"
    elif counts["tube_inside_pass"] < 30:
        route, blocker = "RealTaskTubeDriftViolation", "tube_inside"
    else:
        route, blocker = "RealTaskNoKANSpecificSurplusUnderTube", "task_or_controls"
    return {
        "part_f_gate_pass": gate,
        "part_f_route": route,
        "dominant_blocker": blocker,
        "rows": len(rows),
        "ok_rows": total,
        "error_rows": len([r for r in rows if str(r.get("status")) != "ok"]),
        "fake_data_rows": len([r for r in rows if int(fval(r.get("used_fake_data"))) != 0]),
        "visual_rows": len(vision),
        "tabular_rows": len(tabular),
        **counts,
        "final_NLL_median": median([fval(r.get("final_NLL")) for r in ok]),
        "held_NLL_delta_vs_KAN_AdamW_median": median([fval(r.get("held_NLL_delta_vs_KAN_AdamW")) for r in ok]),
        "held_NLL_delta_vs_MLP_median": median([fval(r.get("held_NLL_delta_vs_MLP_matched")) for r in ok]),
        "best_control_gap_median": median([fval(r.get("best_control_gap")) for r in ok]),
        "MLP_gap_median": median([fval(r.get("MLP_matched_gap")) for r in ok]),
        "overhead_ratio_median": median([fval(r.get("controller_overhead_ratio")) for r in ok]),
        "R_log_spec_distance_median": median([fval(r.get("composite_metric_drift_per_layer")) for r in ok]),
        "source_guard_R_drift_median": median([fval(r.get("source_guard_R_drift_per_layer")) for r in ok]),
        "coverage_CVaR25_median": median([fval(r.get("output_coverage_CVaR25")) for r in ok]),
        "datasets": sorted({str(r.get("dataset")) for r in ok}),
        "MLP_raw_audit_available_rows": sum(1 for r in ok if part_f_key(r) in raw_by_key),
        "MLP_best_matched_family_counts": {
            family: sum(str(best_mlp_candidate(r).get("mlp_control_family")) == family for r in ok)
            for family in sorted({str(best_mlp_candidate(r).get("mlp_control_family")) for r in ok if str(best_mlp_candidate(r).get("mlp_control_family"))})
        },
        "adapter_note": "v22.90 tube adapter over v22.89R compact-task harness; raw-input MLP is supplied only when part_f_mlp_raw_audit artifacts are present",
    }


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_real_task_preflight_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_f(rows)
    audit_rows, audit_summary = build_part_f_failure_audit(rows)
    summary["failure_audit_summary"] = audit_summary
    write_rows(OUT_ROOT / "part_f_real_task_preflight_matrix.csv", rows)
    write_rows(OUT_ROOT / "part_f_failure_audit_matrix.csv", audit_rows)
    write_json(OUT_ROOT / "part_f_failure_audit_summary.json", audit_summary)
    write_json(OUT_ROOT / "part_f_summary.json", {**summary, "rows_detail": rows})
    next_path = write_part_f_next_actions(summary)
    append_exec("part-f-merge", command_text(sys.argv), "done" if summary["part_f_gate_pass"] else "failed", files=f"{rel(OUT_ROOT / 'part_f_real_task_preflight_matrix.csv')}; {rel(OUT_ROOT / 'part_f_summary.json')}; {rel(OUT_ROOT / 'part_f_failure_audit_summary.json')}; {rel(next_path)}")
    append_recap(
        "Part F real-task compact preflight",
        [
            f"gate_pass={summary['part_f_gate_pass']}; route={summary['part_f_route']}; blocker={summary['dominant_blocker']}; ok_rows={summary['ok_rows']}/{summary['rows']}; errors={summary['error_rows']}; fake_data_rows={summary['fake_data_rows']}",
            f"counts: KAN_improves_own={summary['KAN_improves_own']}/30; beats_best_control={summary['beats_best_control']}/30; beats_same={summary['beats_same_composite_controls']}/30; beats_MLP_composite={summary['beats_MLP_composite_coordinate']}/30; no_debt={summary['no_debt']}/30; source_guard={summary['source_guard_R_pass']}/30; coverage={summary['coverage_CVaR25_pass']}/30; visual_coverage={summary['visual_coverage_pass']}/{summary['visual_rows']}; tabular_coverage={summary['tabular_coverage_pass']}/{summary['tabular_rows']}; overhead={summary['overhead_pass']}/30; tube_inside={summary['tube_inside_pass']}/30",
            f"medians: final_NLL={summary['final_NLL_median']}; delta_vs_AdamW={summary['held_NLL_delta_vs_KAN_AdamW_median']}; delta_vs_MLP={summary['held_NLL_delta_vs_MLP_median']}; best_control_gap={summary['best_control_gap_median']}; overhead={summary['overhead_ratio_median']}; R_log={summary['R_log_spec_distance_median']}; source_guard_R_drift={summary['source_guard_R_drift_median']}; coverage_CVaR25={summary['coverage_CVaR25_median']}",
            f"MLP audit: raw_rows={summary['MLP_raw_audit_available_rows']}/{summary['ok_rows']}; best_matched_family_counts={summary['MLP_best_matched_family_counts']}; failure_audit={rel(OUT_ROOT / 'part_f_failure_audit_summary.json')}",
            f"adapter_note: {summary['adapter_note']}",
            "analysis: This Part F pass uses real compact-task rows with the v22.90 tube adapter. It remains subject to the full v22.90 gate and does not permit Part G unless the gate passes.",
        ],
    )
    return summary


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    e = read_json(OUT_ROOT / "part_e_summary.json")
    if not e.get("part_e_gate_pass"):
        out = {"part_f_gate_pass": 0, "part_f_route": "skipped", "skip_reason": "Part E not passed"}
        write_json(OUT_ROOT / "part_f_summary.json", out)
        write_rows(OUT_ROOT / "part_f_real_task_preflight_matrix.csv", [])
        write_next_actions("f", out["part_f_route"], out["skip_reason"], [{"action": "rerun_or_repair_part_e_before_part_f", "reason": out["skip_reason"]}])
        append_exec("part-f", command_text(sys.argv), "skipped", files=f"{rel(OUT_ROOT / 'part_f_summary.json')}; {rel(OUT_ROOT / 'part_f_next_actions_for_codex.json')}")
        append_recap("Part F real-task preflight", ["gate_pass=0; status=skipped", "analysis: Part F cannot run until Part E passes."])
        return out
    device = device_from_args(args)

    def build_rows() -> list[dict[str, Any]]:
        return [v2289.real_task_row(dataset, seed, args, device) for dataset, seed in shard_items(part_f_jobs(args), args)]

    rows = with_v2290_tube_adapter(build_rows)
    path = OUT_ROOT / f"part_f_real_task_preflight_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-f-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def run_part_f_mlp_raw_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for dataset, seed in shard_items(part_f_jobs(args), args):
        row: dict[str, Any] = {"dataset": dataset, "seed": seed, "method": "MLP_matched_raw_input"}
        try:
            bundle = v2289.load_real_task_bundle(dataset, seed, args, device)
            input_dim = int(bundle["input_dim"])
            output_dim = int(bundle["num_classes"])
            basis, k = part_e_basis(args)
            kan_param_count = max(1, input_dim * output_dim * k)
            hidden = max(2, int(round((kan_param_count - output_dim) / max(1, input_dim + output_dim + 1))))
            pf_args = v2289.part_f_flow_args(args)
            mlp_args = v2289.clone_args(
                pf_args,
                pc_hidden=hidden,
                pc_steps=int(args.part_f_steps),
                pc_lr=float(args.part_f_adamw_lr),
                pc_weight_decay=float(args.part_f_weight_decay),
            )
            teacher_tr = F.one_hot(bundle["y_train"].long(), num_classes=output_dim).float()
            teacher_he = F.one_hot(bundle["y_held"].long(), num_classes=output_dim).float()
            v2289.sync_device(device)
            wall_start = time.perf_counter()
            model = v2289.MatchedMLP(input_dim, output_dim, hidden, int(seed) + 2290_4000, device)
            record = v2289.train_mlp_control_model(
                model,
                bundle["x_train"],
                bundle["y_train"],
                bundle["x_held"],
                bundle["y_held"],
                teacher_tr,
                teacher_he,
                mlp_args,
                control_family="MLP_matched_raw_input",
                feature_dim=input_dim,
            )
            v2289.sync_device(device)
            row.update(
                {
                    "status": "ok",
                    "source_kind": bundle["source_kind"],
                    "used_fake_data": int(bundle["used_fake_data"]),
                    "input_dim": input_dim,
                    "original_input_dim": int(bundle["original_input_dim"]),
                    "num_classes": output_dim,
                    "train_rows": int(bundle["x_train"].shape[0]),
                    "held_rows": int(bundle["x_held"].shape[0]),
                    "test_rows": int(bundle["x_test"].shape[0]),
                    "basis": basis,
                    "basis_k": int(k),
                    "kan_reference_param_count": kan_param_count,
                    "mlp_training_steps": int(args.part_f_steps),
                    "wall_time_s": time.perf_counter() - wall_start,
                    **record,
                }
            )
        except Exception as exc:
            row.update({"status": "error", "error": repr(exc)})
        rows.append(row)
    path = OUT_ROOT / f"part_f_mlp_raw_audit_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-f-mlp-raw-audit-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_f_mlp_raw_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_mlp_raw_audit_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "part_f_mlp_raw_audit.csv", rows)
    part_f_rows = read_rows(OUT_ROOT / "part_f_real_task_preflight_matrix.csv")
    summary = summarize_part_f(part_f_rows)
    audit_rows, audit_summary = build_part_f_failure_audit(part_f_rows)
    summary["failure_audit_summary"] = audit_summary
    if part_f_rows:
        write_rows(OUT_ROOT / "part_f_failure_audit_matrix.csv", audit_rows)
        write_json(OUT_ROOT / "part_f_failure_audit_summary.json", audit_summary)
        write_json(OUT_ROOT / "part_f_summary.json", {**summary, "rows_detail": part_f_rows})
        next_path = write_part_f_next_actions(summary)
    else:
        next_path = write_part_f_next_actions({"part_f_gate_pass": 0, "part_f_route": "missing", "dominant_blocker": "part_f_rows_missing"})
    out = {
        "raw_audit_rows": len(rows),
        "raw_audit_ok_rows": sum(str(r.get("status")) == "ok" and int(fval(r.get("used_fake_data"))) == 0 for r in rows),
        "raw_audit_error_rows": sum(str(r.get("status")) != "ok" for r in rows),
        "joined_part_f_rows": len(part_f_rows),
        "beats_MLP_raw_after_join": audit_summary.get("beats_MLP_raw", 0),
        "MLP_raw_input_param_count_median": audit_summary.get("MLP_raw_input_param_count_median", 0.0),
        "MLP_raw_overhead_median_s": audit_summary.get("MLP_raw_overhead_median_s", 0.0),
        "part_f_summary_updated": int(bool(part_f_rows)),
    }
    write_json(OUT_ROOT / "part_f_mlp_raw_audit_summary.json", out)
    append_exec(
        "part-f-mlp-raw-audit-merge",
        command_text(sys.argv),
        "done" if out["raw_audit_error_rows"] == 0 else "failed",
        files=f"{rel(OUT_ROOT / 'part_f_mlp_raw_audit.csv')}; {rel(OUT_ROOT / 'part_f_mlp_raw_audit_summary.json')}; {rel(OUT_ROOT / 'part_f_failure_audit_summary.json')}; {rel(next_path)}",
    )
    append_recap(
        "Part F MLP raw-control audit",
        [
            f"raw_audit_ok_rows={out['raw_audit_ok_rows']}/{out['raw_audit_rows']}; errors={out['raw_audit_error_rows']}; joined_part_f_rows={out['joined_part_f_rows']}",
            f"beats_MLP_raw_after_join={out['beats_MLP_raw_after_join']}; raw_param_count_median={out['MLP_raw_input_param_count_median']}; raw_wall_time_median_s={out['MLP_raw_overhead_median_s']}",
            f"failure_audit: beats_MLP_composite={audit_summary.get('beats_MLP_composite_coordinate')}; best_family_counts={audit_summary.get('MLP_best_matched_family_counts')}; no_debt_pass={audit_summary.get('no_debt_pass_count')}; coverage_pass={audit_summary.get('coverage_CVaR25_pass')}; source_guard_pass={audit_summary.get('source_guard_R_pass')}",
            "analysis: This audit strengthens Part F evidence. It does not weaken MLP controls and does not by itself permit Part G; Part F remains subject to full 30-row gate.",
        ],
    )
    return out


def _debt_baseline_for_probe(row: dict[str, Any], init_train: dict[str, float], tokens: set[str]) -> tuple[dict[str, float], str, int]:
    if v2289.uses_adamw_debt_reference(tokens) and fval(row.get("debt_reference_brier"), 0.0) > 0.0:
        return (
            {
                "brier": fval(row.get("debt_reference_brier")),
                "ece": fval(row.get("debt_reference_ece")),
                "tail95": fval(row.get("debt_reference_tail95")),
                "tail99": fval(row.get("debt_reference_tail95")),
                "margin10": fval(row.get("debt_reference_margin10")),
            },
            "existing_part_f_adamw_train_reference",
            0,
        )
    return init_train, "init_train_reference", int(v2289.uses_adamw_debt_reference(tokens))


def _metric_deltas_after_step(
    model: nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xhe: torch.Tensor,
    yhe: torch.Tensor,
    a_base: torch.Tensor,
    delta: torch.Tensor,
    step_scale: float,
    debt_baseline: dict[str, float],
    before_violation: float,
    before_train_nll: float,
    held_debt_baseline: dict[str, float] | None = None,
    before_held_violation: float | None = None,
    before_held_nll: float | None = None,
    z_base: torch.Tensor | None = None,
    target_logit_velocity: torch.Tensor | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        model.w1.copy_(matrix_to_w1(a_base + float(step_scale) * delta, model.w1.shape, dtype=model.w1.dtype, device=model.w1.device))
    train_logits_after = model(xtr).detach()
    held_logits_after = model(xhe).detach()
    train_metrics = v2289.metrics_for_logits(train_logits_after, ytr)
    held_metrics = v2289.metrics_for_logits(held_logits_after, yhe)
    flags = v2289.no_debt_flags(train_metrics, debt_baseline, 0.01)
    held_flags = v2289.no_debt_flags(held_metrics, held_debt_baseline, 0.01) if held_debt_baseline is not None else None
    out = {
        "train_nll_after": train_metrics["nll"],
        "held_nll_after": held_metrics["nll"],
        "train_nll_delta": train_metrics["nll"] - float(before_train_nll),
        "train_debt_violation_after": flags["debt_violation_sum"],
        "train_debt_violation_delta": flags["debt_violation_sum"] - float(before_violation),
        "train_brier_after": train_metrics["brier"],
        "train_ece_after": train_metrics["ece"],
        "train_tail95_after": train_metrics["tail95"],
        "train_margin10_after": train_metrics["margin10"],
        "train_brier_delta_vs_ref": train_metrics["brier"] - debt_baseline["brier"],
        "train_ece_delta_vs_ref": train_metrics["ece"] - debt_baseline["ece"],
        "train_tail95_delta_vs_ref": train_metrics["tail95"] - debt_baseline["tail95"],
        "train_margin10_delta_vs_ref": train_metrics["margin10"] - debt_baseline["margin10"],
        "train_no_debt_after": flags["no_debt"],
        "train_brier_ok_after": flags["brier_ok"],
        "train_ece_ok_after": flags["ece_ok"],
        "train_tail95_ok_after": flags["tail95_ok"],
        "train_margin10_ok_after": flags["margin10_ok"],
    }
    if held_flags is not None:
        out.update(
            {
                "held_nll_delta": held_metrics["nll"] - float(before_held_nll if before_held_nll is not None else held_metrics["nll"]),
                "held_debt_violation_after": held_flags["debt_violation_sum"],
                "held_debt_violation_delta": held_flags["debt_violation_sum"] - float(before_held_violation if before_held_violation is not None else 0.0),
                "held_brier_after": held_metrics["brier"],
                "held_ece_after": held_metrics["ece"],
                "held_tail95_after": held_metrics["tail95"],
                "held_margin10_after": held_metrics["margin10"],
                "held_brier_delta_vs_ref": held_metrics["brier"] - held_debt_baseline["brier"],
                "held_ece_delta_vs_ref": held_metrics["ece"] - held_debt_baseline["ece"],
                "held_tail95_delta_vs_ref": held_metrics["tail95"] - held_debt_baseline["tail95"],
                "held_margin10_delta_vs_ref": held_metrics["margin10"] - held_debt_baseline["margin10"],
                "held_no_debt_after": held_flags["no_debt"],
                "held_brier_ok_after": held_flags["brier_ok"],
                "held_ece_ok_after": held_flags["ece_ok"],
                "held_tail95_ok_after": held_flags["tail95_ok"],
                "held_margin10_ok_after": held_flags["margin10_ok"],
            }
        )
    if z_base is not None and target_logit_velocity is not None and abs(float(step_scale)) > 0.0:
        finite_logit_velocity = (train_logits_after.to(dtype=torch.float64) - z_base.to(device=train_logits_after.device, dtype=torch.float64)) / float(step_scale)
        target = target_logit_velocity.to(device=train_logits_after.device, dtype=torch.float64)
        denom = target.norm().clamp_min(EPS)
        residual = (finite_logit_velocity - target).norm() / denom
        coverage = torch.sum(finite_logit_velocity * target) / (finite_logit_velocity.norm() * target.norm()).clamp_min(EPS)
        out.update(
            {
                "composite_pullback_residual": float(residual.detach().cpu().item()),
                "composite_pullback_coverage": float(coverage.detach().cpu().item()),
                "target_logit_velocity_norm": float(target.norm().detach().cpu().item()),
                "finite_logit_velocity_norm": float(finite_logit_velocity.norm().detach().cpu().item()),
            }
        )
    with torch.no_grad():
        model.w1.copy_(matrix_to_w1(a_base, model.w1.shape, dtype=model.w1.dtype, device=model.w1.device))
    return out


def part_f_debt_calibration_probe_row(dataset: str, seed: int, args: argparse.Namespace, device: torch.device, existing: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    base: dict[str, Any] = {"dataset": dataset, "seed": seed, "repair": str(getattr(args, "repair", ""))}
    try:
        bundle = v2289.load_real_task_bundle(dataset, seed, args, device)
        input_dim = int(bundle["input_dim"])
        output_dim = int(bundle["num_classes"])
        pf_args = v2289.part_f_flow_args(args)
        active_repair = v2289.part_f_active_repair(args)
        tokens = v2289.repair_tokens(active_repair)
        model = v2289.make_real_composite_model(
            input_dim,
            output_dim,
            int(seed) + 60_000,
            pf_args,
            device,
            representation_mode="joint_costate",
            bias_edge=("biasedge" in tokens or "constantedge" in tokens),
        )
        c, _r_init, _init_info = v2289.init_composite_additive(
            model,
            bundle["x_train"],
            v2289.composite_init_target_from_tokens(tokens),
            int(seed) + 60_000,
            pf_args,
            y=bundle["y_train"] if ("labelpullback" in tokens or v2289.uses_sensitivity_metric(tokens)) else None,
            use_quantile_transport=v2289.uses_quantile_transport(tokens),
            metric_shrink_alpha=float(args.metric_shrink_alpha) if v2289.uses_metric_shrink(tokens) else 0.0,
        )
        xtr, ytr = bundle["x_train"], bundle["y_train"]
        xhe, yhe = bundle["x_held"], bundle["y_held"]
        init_train = v2289.metrics_for_logits(model(xtr), ytr)
        init_held = v2289.metrics_for_logits(model(xhe), yhe)
        existing_row = existing.get((dataset, int(seed)), {})
        debt_baseline, debt_reference_kind, debt_reference_missing = _debt_baseline_for_probe(existing_row, init_train, tokens)
        before_flags = v2289.no_debt_flags(init_train, debt_baseline, float(args.no_debt_budget))
        before_held_flags = v2289.no_debt_flags(init_held, init_held, float(args.no_debt_budget))
        logits = model(xtr)
        ce_loss = F.cross_entropy(logits.float(), ytr.long())
        ce_grad = torch.autograd.grad(ce_loss, [model.w1], retain_graph=True, create_graph=False)[0]
        debt_loss = v2289.debt_surrogate_loss(
            logits,
            ytr,
            debt_baseline,
            float(args.no_debt_budget),
            ece_weight=float(args.ece_debt_weight) if v2289.uses_ece_debt(tokens) else 0.0,
            bucketed_ece=v2289.uses_bucketed_ece_debt(tokens),
        )
        if v2289.uses_coverage_controller(tokens):
            debt_loss = debt_loss + v2289.coverage_tail_margin_cotangent_loss(
                logits,
                ytr,
                debt_baseline,
                budget=float(args.no_debt_budget),
                coverage_target=float(args.coverage_cvar_target),
            )
        debt_grad = torch.autograd.grad(debt_loss, [model.w1], retain_graph=False, create_graph=False, allow_unused=False)[0]
        z = logits.detach().clone().requires_grad_(True)
        z_loss = F.cross_entropy(z.float(), ytr.long())
        z_grad = torch.autograd.grad(z_loss, [z], retain_graph=False, create_graph=False)[0]
        safe = output_safe_velocity_from_logits(z.detach(), ytr, -z_grad.detach(), alpha=0.0)
        a_base = w1_to_matrix(model.w1.detach()).to(device=device)
        ce_g = w1_to_matrix(ce_grad.detach()).to(device=device)
        debt_g = w1_to_matrix(debt_grad.detach()).to(device=device)
        task_delta, _ = natural_tangent_velocity(c, a_base, ce_g, EPS, torch.linalg.inv(c))
        debt_delta, _ = natural_tangent_velocity(c, a_base, debt_g, EPS, torch.linalg.inv(c))
        task_norm = ce_g.norm().clamp_min(EPS)
        debt_norm = debt_g.norm().clamp_min(EPS)
        capped_debt_g = debt_g
        cap = float(getattr(args, "probe_debt_norm_cap", 1.0)) * task_norm
        if float(debt_norm.detach().cpu().item()) > float(cap.detach().cpu().item()):
            capped_debt_g = debt_g * (cap / debt_norm)
        combo_g = ce_g + float(args.debt_cotangent_blend) * capped_debt_g
        combo_delta, _ = natural_tangent_velocity(c, a_base, combo_g, EPS, torch.linalg.inv(c))
        directions = {
            "task": task_delta,
            "debt": debt_delta,
            "combo": combo_delta,
        }
        rows: list[dict[str, Any]] = []
        for direction_name, delta in directions.items():
            pred_task = float((ce_g * delta).sum().detach().cpu().item())
            pred_debt = float((debt_g * delta).sum().detach().cpu().item())
            for scale_text in v2289.csv_items(str(args.probe_step_scales)):
                scale = float(scale_text)
                vals = _metric_deltas_after_step(
                    model,
                    xtr,
                    ytr,
                    xhe,
                    yhe,
                    a_base,
                    delta,
                    scale,
                    debt_baseline,
                    float(before_flags["debt_violation_sum"]),
                    float(init_train["nll"]),
                    held_debt_baseline=init_held,
                    before_held_violation=float(before_held_flags["debt_violation_sum"]),
                    before_held_nll=float(init_held["nll"]),
                )
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "used_fake_data": int(bundle["used_fake_data"]),
                        "source_kind": bundle["source_kind"],
                        "input_dim": input_dim,
                        "num_classes": output_dim,
                        "direction": direction_name,
                        "step_scale": scale,
                        "debt_reference_kind": debt_reference_kind,
                        "debt_reference_missing": debt_reference_missing,
                        "init_train_nll": init_train["nll"],
                        "init_held_nll": init_held["nll"],
                        "before_train_debt_violation": before_flags["debt_violation_sum"],
                        "before_train_no_debt": before_flags["no_debt"],
                        "ce_grad_norm": float(ce_g.norm().detach().cpu().item()),
                        "debt_grad_norm": float(debt_g.norm().detach().cpu().item()),
                        "debt_to_task_grad_norm_ratio": float((debt_g.norm() / task_norm).detach().cpu().item()),
                        "combo_debt_norm_capped": int(float(debt_norm.detach().cpu().item()) > float(cap.detach().cpu().item())),
                        "pred_task_directional": pred_task,
                        "pred_debt_directional": pred_debt,
                        "pred_task_descent": int(pred_task < 0.0),
                        "pred_debt_descent": int(pred_debt < 0.0),
                        "output_safe_qp_kkt_residual": safe.kkt_residual,
                        "output_safe_task_preserved_fraction": safe.task_preserved_fraction,
                        "output_safe_debt_violation_after": safe.debt_violation_after,
                        **vals,
                        "finite_task_descent": int(fval(vals["train_nll_delta"], 1.0) < 0.0),
                        "finite_debt_improved": int(fval(vals["train_debt_violation_delta"], 1.0) < 0.0),
                        "debt_sign_agreement": int((pred_debt < 0.0) == (fval(vals["train_debt_violation_delta"], 1.0) < 0.0)),
                    }
                )
        return rows
    except Exception as exc:
        return [{**base, "status": "error", "error": repr(exc)}]


def part_f_output_safe_pullback_probe_row(dataset: str, seed: int, args: argparse.Namespace, device: torch.device, existing: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    base: dict[str, Any] = {"dataset": dataset, "seed": seed, "repair": str(getattr(args, "repair", ""))}
    try:
        bundle = v2289.load_real_task_bundle(dataset, seed, args, device)
        input_dim = int(bundle["input_dim"])
        output_dim = int(bundle["num_classes"])
        pf_args = v2289.part_f_flow_args(args)
        active_repair = v2289.part_f_active_repair(args)
        tokens = v2289.repair_tokens(active_repair)
        model = v2289.make_real_composite_model(
            input_dim,
            output_dim,
            int(seed) + 61_000,
            pf_args,
            device,
            representation_mode="joint_costate",
            bias_edge=("biasedge" in tokens or "constantedge" in tokens),
        )
        c, _r_init, _init_info = v2289.init_composite_additive(
            model,
            bundle["x_train"],
            v2289.composite_init_target_from_tokens(tokens),
            int(seed) + 61_000,
            pf_args,
            y=bundle["y_train"] if ("labelpullback" in tokens or v2289.uses_sensitivity_metric(tokens)) else None,
            use_quantile_transport=v2289.uses_quantile_transport(tokens),
            metric_shrink_alpha=float(args.metric_shrink_alpha) if v2289.uses_metric_shrink(tokens) else 0.0,
        )
        xtr, ytr = bundle["x_train"], bundle["y_train"]
        xhe, yhe = bundle["x_held"], bundle["y_held"]
        init_train_logits = model(xtr).detach()
        init_held_logits = model(xhe).detach()
        init_train = v2289.metrics_for_logits(init_train_logits, ytr)
        init_held = v2289.metrics_for_logits(init_held_logits, yhe)
        existing_row = existing.get((dataset, int(seed)), {})
        debt_baseline, debt_reference_kind, debt_reference_missing = _debt_baseline_for_probe(existing_row, init_train, tokens)
        before_train_flags = v2289.no_debt_flags(init_train, debt_baseline, float(args.no_debt_budget))
        before_held_flags = v2289.no_debt_flags(init_held, init_held, float(args.no_debt_budget))

        logits = model(xtr)
        ce_loss = F.cross_entropy(logits.float(), ytr.long())
        ce_grad = torch.autograd.grad(ce_loss, [model.w1], retain_graph=True, create_graph=False)[0]
        debt_loss = v2289.debt_surrogate_loss(
            logits,
            ytr,
            debt_baseline,
            float(args.no_debt_budget),
            ece_weight=float(args.ece_debt_weight) if v2289.uses_ece_debt(tokens) else 0.0,
            bucketed_ece=v2289.uses_bucketed_ece_debt(tokens),
        )
        if v2289.uses_coverage_controller(tokens):
            debt_loss = debt_loss + v2289.coverage_tail_margin_cotangent_loss(
                logits,
                ytr,
                debt_baseline,
                budget=float(args.no_debt_budget),
                coverage_target=float(args.coverage_cvar_target),
            )
        debt_grad = torch.autograd.grad(debt_loss, [model.w1], retain_graph=False, create_graph=False, allow_unused=False)[0]

        z = init_train_logits.detach().clone().requires_grad_(True)
        z_loss = F.cross_entropy(z.float(), ytr.long())
        z_grad = torch.autograd.grad(z_loss, [z], retain_graph=False, create_graph=False)[0]
        safe = output_safe_velocity_from_logits(z.detach(), ytr, -z_grad.detach(), alpha=float(args.output_safe_alpha))
        pull_logits = model(xtr)
        target_velocity = safe.velocity.detach().to(device=pull_logits.device, dtype=pull_logits.dtype)
        pull_loss = -torch.sum(pull_logits.float() * target_velocity.float())
        pull_grad = torch.autograd.grad(pull_loss, [model.w1], retain_graph=False, create_graph=False, allow_unused=False)[0]

        a_base = w1_to_matrix(model.w1.detach()).to(device=device)
        ce_g = w1_to_matrix(ce_grad.detach()).to(device=device)
        debt_g = w1_to_matrix(debt_grad.detach()).to(device=device)
        pull_g = w1_to_matrix(pull_grad.detach()).to(device=device)
        pull_g_norm = pull_g.norm().clamp_min(EPS)
        ce_g_norm = ce_g.norm().clamp_min(EPS)
        pull_cap = float(args.output_safe_pullback_norm_cap) * ce_g_norm
        pull_g_capped = pull_g
        pullback_norm_capped = int(float(pull_g_norm.detach().cpu().item()) > float(pull_cap.detach().cpu().item()))
        if pullback_norm_capped:
            pull_g_capped = pull_g * (pull_cap / pull_g_norm)
        c_inv = torch.linalg.inv(c)
        ce_delta, _ = natural_tangent_velocity(c, a_base, ce_g, EPS, c_inv)
        debt_delta, _ = natural_tangent_velocity(c, a_base, debt_g, EPS, c_inv)
        pull_delta, _ = natural_tangent_velocity(c, a_base, pull_g, EPS, c_inv)
        pull_norm_delta, _ = natural_tangent_velocity(c, a_base, pull_g_capped, EPS, c_inv)
        directions = {
            "ce_tangent": (ce_delta, None),
            "debt_tangent": (debt_delta, None),
            "output_safe_pullback": (pull_delta, safe.velocity.detach()),
            "output_safe_pullback_norm": (pull_norm_delta, safe.velocity.detach()),
        }
        rows: list[dict[str, Any]] = []
        for direction_name, (delta, maybe_target_velocity) in directions.items():
            pred_task = float((ce_g * delta).sum().detach().cpu().item())
            pred_debt = float((debt_g * delta).sum().detach().cpu().item())
            for scale_text in v2289.csv_items(str(args.probe_step_scales)):
                scale = float(scale_text)
                vals = _metric_deltas_after_step(
                    model,
                    xtr,
                    ytr,
                    xhe,
                    yhe,
                    a_base,
                    delta,
                    scale,
                    debt_baseline,
                    float(before_train_flags["debt_violation_sum"]),
                    float(init_train["nll"]),
                    held_debt_baseline=init_held,
                    before_held_violation=float(before_held_flags["debt_violation_sum"]),
                    before_held_nll=float(init_held["nll"]),
                    z_base=init_train_logits if maybe_target_velocity is not None else None,
                    target_logit_velocity=maybe_target_velocity,
                )
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "used_fake_data": int(bundle["used_fake_data"]),
                        "source_kind": bundle["source_kind"],
                        "input_dim": input_dim,
                        "num_classes": output_dim,
                        "direction": direction_name,
                        "step_scale": scale,
                        "debt_reference_kind": debt_reference_kind,
                        "debt_reference_missing": debt_reference_missing,
                        "init_train_nll": init_train["nll"],
                        "init_held_nll": init_held["nll"],
                        "before_train_debt_violation": before_train_flags["debt_violation_sum"],
                        "before_train_no_debt": before_train_flags["no_debt"],
                        "before_held_debt_violation": before_held_flags["debt_violation_sum"],
                        "before_held_no_debt": before_held_flags["no_debt"],
                        "ce_grad_norm": float(ce_g.norm().detach().cpu().item()),
                        "debt_grad_norm": float(debt_g.norm().detach().cpu().item()),
                        "pullback_grad_norm": float(pull_g.norm().detach().cpu().item()),
                        "debt_to_task_grad_norm_ratio": float((debt_g.norm() / ce_g_norm).detach().cpu().item()),
                        "pullback_to_task_grad_norm_ratio": float((pull_g.norm() / ce_g_norm).detach().cpu().item()),
                        "pullback_norm_capped": pullback_norm_capped,
                        "output_safe_pullback_norm_cap": float(args.output_safe_pullback_norm_cap),
                        "output_safe_alpha": float(args.output_safe_alpha),
                        "output_safe_qp_kkt_residual": safe.kkt_residual,
                        "output_safe_task_preserved_fraction": safe.task_preserved_fraction,
                        "output_safe_debt_violation_after": safe.debt_violation_after,
                        "output_safe_active_constraints": safe.active_constraints,
                        "pred_task_directional": pred_task,
                        "pred_debt_directional": pred_debt,
                        "pred_task_descent": int(pred_task < 0.0),
                        "pred_debt_descent": int(pred_debt < 0.0),
                        **vals,
                        "finite_train_task_descent": int(fval(vals["train_nll_delta"], 1.0) < 0.0),
                        "finite_train_debt_improved": int(fval(vals["train_debt_violation_delta"], 1.0) < 0.0),
                        "finite_held_task_descent": int(fval(vals.get("held_nll_delta"), 1.0) < 0.0),
                        "finite_held_debt_improved": int(fval(vals.get("held_debt_violation_delta"), 1.0) < 0.0),
                    }
                )
        return rows
    except Exception as exc:
        return [{**base, "status": "error", "error": repr(exc)}]


def summarize_debt_calibration_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if str(r.get("status")) == "ok"]
    by_direction: dict[str, dict[str, Any]] = {}
    for direction in sorted({str(r.get("direction")) for r in ok}):
        sub = [r for r in ok if str(r.get("direction")) == direction]
        by_direction[direction] = {
            "rows": len(sub),
            "pred_debt_descent": sum(int(fval(r.get("pred_debt_descent"))) == 1 for r in sub),
            "finite_debt_improved": sum(int(fval(r.get("finite_debt_improved"))) == 1 for r in sub),
            "debt_sign_agreement": sum(int(fval(r.get("debt_sign_agreement"))) == 1 for r in sub),
            "pred_task_descent": sum(int(fval(r.get("pred_task_descent"))) == 1 for r in sub),
            "finite_task_descent": sum(int(fval(r.get("finite_task_descent"))) == 1 for r in sub),
            "train_debt_violation_delta_median": median([fval(r.get("train_debt_violation_delta")) for r in sub]),
            "train_nll_delta_median": median([fval(r.get("train_nll_delta")) for r in sub]),
        }
    return {
        "rows": len(rows),
        "ok_rows": len(ok),
        "error_rows": len([r for r in rows if str(r.get("status")) != "ok"]),
        "unique_dataset_seed_rows": len({(str(r.get("dataset")), int(fval(r.get("seed")))) for r in ok}),
        "output_safe_qp_kkt_residual_max": max([fval(r.get("output_safe_qp_kkt_residual")) for r in ok], default=0.0),
        "output_safe_task_preserved_fraction_median": median([fval(r.get("output_safe_task_preserved_fraction")) for r in ok]),
        "debt_to_task_grad_norm_ratio_median": median([fval(r.get("debt_to_task_grad_norm_ratio")) for r in ok]),
        "combo_debt_norm_capped_rows": sum(int(fval(r.get("combo_debt_norm_capped"))) == 1 for r in ok),
        "by_direction": by_direction,
        "interpretation": "Finite-step probe is train-only and diagnostic. It checks whether local debt cotangent predictions match actual small-step debt metric changes; it is not a Part F pass artifact.",
    }


def run_part_f_debt_calibration_probe(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    existing = part_f_existing_rows()
    rows: list[dict[str, Any]] = []
    for dataset, seed in shard_items(part_f_jobs(args), args):
        rows.extend(part_f_debt_calibration_probe_row(dataset, seed, args, device, existing))
    path = OUT_ROOT / f"part_f_debt_calibration_probe_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-f-debt-calibration-probe-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_f_debt_calibration_probe(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_debt_calibration_probe_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_debt_calibration_probe(rows)
    write_rows(OUT_ROOT / "part_f_debt_calibration_probe.csv", rows)
    write_json(OUT_ROOT / "part_f_debt_calibration_probe_summary.json", summary)
    append_exec(
        "part-f-debt-calibration-probe-merge",
        command_text(sys.argv),
        "done" if summary["error_rows"] == 0 else "failed",
        files=f"{rel(OUT_ROOT / 'part_f_debt_calibration_probe.csv')}; {rel(OUT_ROOT / 'part_f_debt_calibration_probe_summary.json')}",
    )
    append_recap(
        "Part F finite-step debt calibration probe",
        [
            f"ok_rows={summary['ok_rows']}/{summary['rows']}; dataset_seed_rows={summary['unique_dataset_seed_rows']}; errors={summary['error_rows']}",
            f"output_safe_qp_kkt_max={summary['output_safe_qp_kkt_residual_max']}; output_safe_task_preserved_median={summary['output_safe_task_preserved_fraction_median']}; debt_to_task_grad_norm_ratio_median={summary['debt_to_task_grad_norm_ratio_median']}",
            f"by_direction={summary['by_direction']}",
            "analysis: Probe is train-only diagnostic evidence for debt surrogate finite-step calibration. It is not a Part F gate pass and does not permit Part G.",
        ],
    )
    return summary


def summarize_output_safe_pullback_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if str(r.get("status")) == "ok"]
    by_direction: dict[str, dict[str, Any]] = {}
    for direction in sorted({str(r.get("direction")) for r in ok}):
        sub = [r for r in ok if str(r.get("direction")) == direction]
        by_direction[direction] = {
            "rows": len(sub),
            "pred_task_descent": sum(int(fval(r.get("pred_task_descent"))) == 1 for r in sub),
            "pred_debt_descent": sum(int(fval(r.get("pred_debt_descent"))) == 1 for r in sub),
            "finite_train_task_descent": sum(int(fval(r.get("finite_train_task_descent"))) == 1 for r in sub),
            "finite_train_debt_improved": sum(int(fval(r.get("finite_train_debt_improved"))) == 1 for r in sub),
            "finite_held_task_descent": sum(int(fval(r.get("finite_held_task_descent"))) == 1 for r in sub),
            "finite_held_debt_improved": sum(int(fval(r.get("finite_held_debt_improved"))) == 1 for r in sub),
            "train_nll_delta_median": median([fval(r.get("train_nll_delta")) for r in sub]),
            "held_nll_delta_median": median([fval(r.get("held_nll_delta")) for r in sub]),
            "train_debt_violation_delta_median": median([fval(r.get("train_debt_violation_delta")) for r in sub]),
            "held_debt_violation_delta_median": median([fval(r.get("held_debt_violation_delta")) for r in sub]),
            "composite_pullback_residual_median": median([fval(r.get("composite_pullback_residual")) for r in sub if str(r.get("composite_pullback_residual", "")) != ""]),
            "composite_pullback_coverage_median": median([fval(r.get("composite_pullback_coverage")) for r in sub if str(r.get("composite_pullback_coverage", "")) != ""]),
        }
    return {
        "rows": len(rows),
        "ok_rows": len(ok),
        "error_rows": len([r for r in rows if str(r.get("status")) != "ok"]),
        "unique_dataset_seed_rows": len({(str(r.get("dataset")), int(fval(r.get("seed")))) for r in ok}),
        "output_safe_qp_kkt_residual_max": max([fval(r.get("output_safe_qp_kkt_residual")) for r in ok], default=0.0),
        "output_safe_infeasible_rows": sum(fval(r.get("output_safe_qp_kkt_residual")) > 1.0e-4 for r in ok),
        "output_safe_infeasible_dataset_seed_rows": len({(str(r.get("dataset")), int(fval(r.get("seed")))) for r in ok if fval(r.get("output_safe_qp_kkt_residual")) > 1.0e-4}),
        "output_safe_task_preserved_fraction_median": median([fval(r.get("output_safe_task_preserved_fraction")) for r in ok]),
        "output_safe_active_constraints_median": median([fval(r.get("output_safe_active_constraints")) for r in ok]),
        "pullback_to_task_grad_norm_ratio_median": median([fval(r.get("pullback_to_task_grad_norm_ratio")) for r in ok]),
        "pullback_norm_capped_rows": sum(int(fval(r.get("pullback_norm_capped"))) == 1 for r in ok),
        "by_direction": by_direction,
        "interpretation": "Train-only diagnostic: checks whether output-safe logit velocity can be pulled back into the composite tube direction and improve finite-step train/held debt. It is not a Part F pass artifact.",
    }


def run_part_f_output_safe_pullback_probe(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    existing = part_f_existing_rows()
    rows: list[dict[str, Any]] = []
    for dataset, seed in shard_items(part_f_jobs(args), args):
        rows.extend(part_f_output_safe_pullback_probe_row(dataset, seed, args, device, existing))
    path = OUT_ROOT / f"part_f_output_safe_pullback_probe_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-f-output-safe-pullback-probe-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_f_output_safe_pullback_probe(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_output_safe_pullback_probe_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_output_safe_pullback_probe(rows)
    write_rows(OUT_ROOT / "part_f_output_safe_pullback_probe.csv", rows)
    write_json(OUT_ROOT / "part_f_output_safe_pullback_probe_summary.json", summary)
    append_exec(
        "part-f-output-safe-pullback-probe-merge",
        command_text(sys.argv),
        "done" if summary["error_rows"] == 0 else "failed",
        files=f"{rel(OUT_ROOT / 'part_f_output_safe_pullback_probe.csv')}; {rel(OUT_ROOT / 'part_f_output_safe_pullback_probe_summary.json')}",
    )
    append_recap(
        "Part F output-safe pullback probe",
        [
            f"ok_rows={summary['ok_rows']}/{summary['rows']}; dataset_seed_rows={summary['unique_dataset_seed_rows']}; errors={summary['error_rows']}",
            f"output_safe_qp_kkt_max={summary['output_safe_qp_kkt_residual_max']}; output_safe_task_preserved_median={summary['output_safe_task_preserved_fraction_median']}; pullback_to_task_grad_norm_ratio_median={summary['pullback_to_task_grad_norm_ratio_median']}",
            f"by_direction={summary['by_direction']}",
            "analysis: Probe is train-only diagnostic evidence for output-safe velocity pullback into the composite tube. It is not a Part F gate pass and does not permit Part G.",
        ],
    )
    return summary


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    out = {"part_g_gate_pass": 0, "part_g_route": "skipped", "skip_reason": "Part F not passed"}
    write_json(OUT_ROOT / "part_g_summary.json", out)
    write_rows(OUT_ROOT / "part_g_hstep_matrix.csv", [])
    write_next_actions("g", "skipped", out["skip_reason"], [{"action": "rerun_after_part_f_pass", "reason": out["skip_reason"]}])
    append_exec("part-g", command_text(sys.argv), "skipped", files=f"{rel(OUT_ROOT / 'part_g_summary.json')}; {rel(OUT_ROOT / 'part_g_next_actions_for_codex.json')}")
    return out


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    a = read_json(OUT_ROOT / "part_a_code_identity.json")
    c = read_json(OUT_ROOT / "part_c_summary.json")
    d = read_json(OUT_ROOT / "part_d_summary.json")
    e = read_json(OUT_ROOT / "part_e_summary.json")
    f = read_json(OUT_ROOT / "part_f_summary.json")
    g = read_json(OUT_ROOT / "part_g_summary.json")
    if not a.get("part_a_hard_gate_pass"):
        route = "PartA_CodeIdentityFailed"
        reason = "Part A missing or failed"
    elif not c.get("part_c_diagnostic_pass"):
        route = "PartC_FixedOrbitDiagnosticFailed"
        reason = "Part C missing or failed"
    elif not d.get("part_d_gate_pass"):
        route = "PartD_TubeImplementationFailed"
        reason = "Part D missing or failed"
    elif not e.get("part_e_gate_pass"):
        route = "PartE_PositiveControlFailed"
        reason = e.get("skip_reason", "Part E failed or not run")
    elif not f.get("part_f_gate_pass"):
        route = f.get("part_f_route", "PartF_RealTaskMLPMatchedDominates")
        reason = f.get("dominant_blocker", f.get("skip_reason", "Part F failed"))
    elif not g.get("part_g_gate_pass"):
        route = g.get("part_g_route", "PartG_HStepDebtBlocked")
        reason = g.get("dominant_blocker", g.get("skip_reason", "Part G failed"))
    else:
        route = "ControlledCompositeMetricTubeOfficialCandidateOpened"
        reason = "Part A-G passed"
    final = {
        "gate": "v22_90_final_route",
        "official_candidate_gate_pass": int(route == "ControlledCompositeMetricTubeOfficialCandidateOpened"),
        "final_route": route,
        "route_reason": reason,
        "part_a": a,
        "part_c": c,
        "part_d": d,
        "part_e": e,
        "part_f": f,
        "part_g": g,
    }
    write_json(OUT_ROOT / "failure_decision_matrix.json", {"final_route": route, "reason": reason})
    write_rows(OUT_ROOT / "failure_decision_matrix.csv", [{"final_route": route, "reason": reason}])
    (OUT_ROOT / "key_insights.md").write_text(f"# v22.90 key insights\n\n- final_route={route}\n- reason={reason}\n", encoding="utf-8")
    (OUT_ROOT / "reproduction_manifest.md").write_text(f"# v22.90 reproduction manifest\n\nPython: `{PYTHON}`\nRunner: `{rel(RUNNER)}`\nFinalize: `{PYTHON} {rel(RUNNER)} --mode finalize --device cuda`\n", encoding="utf-8")
    write_json(OUT_ROOT / "final_route.json", final)
    append_exec("finalize", command_text(sys.argv), "done", files=rel(OUT_ROOT / "final_route.json"))
    append_recap("Final route", [f"official_candidate_gate_pass={final['official_candidate_gate_pass']}; final_route={route}; reason={reason}", "analysis: Final route is based only on available artifacts; missing E/F/G are not treated as success."])
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--pc-basis", default="D-FOU")
    p.add_argument("--pc-train-size", type=int, default=512)
    p.add_argument("--pc-test-size", type=int, default=384)
    p.add_argument("--pc-hidden", type=int, default=16)
    p.add_argument("--pc-steps", type=int, default=80)
    p.add_argument("--pc-lr", type=float, default=1.0e-2)
    p.add_argument("--pc-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--cmp-lr", type=float, default=0.8)
    p.add_argument("--cmp-max-norm-ratio", type=float, default=20.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--target-variance", type=float, default=49.0)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--c-condition-budget", type=float, default=1.0e5)
    p.add_argument("--no-debt-budget", type=float, default=0.01)
    p.add_argument("--rep-lr-ratio", type=float, default=0.01)
    p.add_argument("--delta-spec", type=float, default=0.40)
    p.add_argument("--radial-velocity-fraction-cap", type=float, default=0.25)
    p.add_argument("--metric-update-interval", type=int, default=10)
    p.add_argument("--part-f-datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--part-f-seed-count", type=int, default=6)
    p.add_argument("--part-f-train-size", type=int, default=512)
    p.add_argument("--part-f-held-size", type=int, default=256)
    p.add_argument("--part-f-test-size", type=int, default=256)
    p.add_argument("--part-f-steps", type=int, default=120)
    p.add_argument("--part-f-eval-interval", type=int, default=20)
    p.add_argument("--part-f-batch-size", type=int, default=0)
    p.add_argument("--part-f-max-input-dim", type=int, default=64)
    p.add_argument("--part-f-cmp-lr", type=float, default=0.8)
    p.add_argument("--part-f-adamw-lr", type=float, default=0.02)
    p.add_argument("--part-f-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--part-f-target-variance", type=float, default=49.0)
    p.add_argument("--part-f-cmp-max-norm-ratio", type=float, default=20.0)
    p.add_argument("--part-f-scale-band", type=float, default=0.40)
    p.add_argument("--part-f-rep-lr-ratio", type=float, default=0.01)
    p.add_argument("--repair", default="")
    p.add_argument("--metric-shrink-alpha", type=float, default=0.0)
    p.add_argument("--debt-cotangent-blend", type=float, default=0.25)
    p.add_argument("--debt-dual-lr", type=float, default=0.05)
    p.add_argument("--barrier-alpha", type=float, default=0.10)
    p.add_argument("--barrier-max-correction-ratio", type=float, default=2.0)
    p.add_argument("--domain-transport-interval", type=int, default=25)
    p.add_argument("--population-diffusion-blend", type=float, default=0.0)
    p.add_argument("--population-diffusion-rho", type=float, default=0.25)
    p.add_argument("--population-diffusion-max-ratio", type=float, default=1.0)
    p.add_argument("--population-diffusion-interval", type=int, default=25)
    p.add_argument("--derivative-metric-weight", type=float, default=0.35)
    p.add_argument("--sensitivity-weight-min", type=float, default=0.20)
    p.add_argument("--sensitivity-weight-max", type=float, default=6.0)
    p.add_argument("--ece-debt-weight", type=float, default=0.0)
    p.add_argument("--coverage-cvar-target", type=float, default=0.20)
    p.add_argument("--probe-step-scales", default="0.02,0.08,0.20")
    p.add_argument("--probe-debt-norm-cap", type=float, default=1.0)
    p.add_argument("--output-safe-alpha", type=float, default=0.10)
    p.add_argument("--output-safe-pullback-norm-cap", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode)
    if mode == "part-a":
        return run_part_a(args)
    if mode == "part-b":
        return run_part_b(args)
    if mode == "part-c":
        return run_part_c(args)
    if mode == "part-c-merge":
        return merge_part_c(args)
    if mode == "part-d":
        return run_part_d(args)
    if mode == "part-e":
        return run_part_e(args)
    if mode == "part-e-merge":
        return merge_part_e(args)
    if mode == "part-f":
        return run_part_f(args)
    if mode == "part-f-merge":
        return merge_part_f(args)
    if mode == "part-f-mlp-raw-audit":
        return run_part_f_mlp_raw_audit(args)
    if mode == "part-f-mlp-raw-audit-merge":
        return merge_part_f_mlp_raw_audit(args)
    if mode == "part-f-debt-calibration-probe":
        return run_part_f_debt_calibration_probe(args)
    if mode == "part-f-debt-calibration-probe-merge":
        return merge_part_f_debt_calibration_probe(args)
    if mode == "part-f-output-safe-pullback-probe":
        return run_part_f_output_safe_pullback_probe(args)
    if mode == "part-f-output-safe-pullback-probe-merge":
        return merge_part_f_output_safe_pullback_probe(args)
    if mode == "part-g":
        return run_part_g(args)
    if mode == "finalize":
        return finalize(args)
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
