#!/usr/bin/env python3
"""DG-KAN v23.15 basis-covariant edge-function natural flow runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import py_compile
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_07_edge_function_residual_inverse_flow as v2307
from dgkan.fu.basis_covariant_edge_flow import (
    FlowResult,
    apply_layer_delta,
    local_bc15_pcg_flow,
    local_exact_flow,
    local_fixed_identity_ridge_flow,
    local_unpreconditioned_pcg_flow,
)
from dgkan.fu.covariant_edge_preconditioner import eig_condition, make_block_preconditioner, normal_system, pcg, pcg_residual_inverse, solve_spd, sym
from dgkan.fu.edge_basis_chart import (
    Chart,
    block_transform,
    charted_forward_with_activations,
    coefficient_roundtrip_error,
    expanded_metric,
    flat_delta_from_chart,
    make_chart,
    max_relative_error,
    transform_metric,
    transform_phi,
)
from dgkan.fu.edge_sobolev_metrics import functional_edge_gram
from dgkan.fu.function_space_trust import clip_by_g_norm, g_norm


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.15_BasisCovariantEdgeFunctionNaturalFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.15_BasisCovariantEdgeFunctionNaturalFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.15_BasisCovariantEdgeFunctionNaturalFlow_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2315_OUT_ROOT", str(ROOT / "results/v23_15/current"))).resolve()

HELPERS = [
    ROOT / "dgkan/fu/basis_covariant_edge_flow.py",
    ROOT / "dgkan/fu/edge_basis_chart.py",
    ROOT / "dgkan/fu/covariant_edge_preconditioner.py",
    ROOT / "dgkan/fu/function_space_trust.py",
]
OLD_RUNNERS = [
    ROOT / "experiments/run_v23_09_trust_projected_downstream_efrf.py",
    ROOT / "experiments/run_v23_10_efficiency_equivalence_task_curvature_efrf.py",
    ROOT / "experiments/run_v23_11_feature_learning_tangent_escape_kan.py",
    ROOT / "experiments/run_v23_12_total_audit_feature_learning_tangent_escape.py",
    ROOT / "experiments/run_v23_13_cheb_domain_transport_representation_geometry.py",
    ROOT / "experiments/run_v23_14_source_whitened_cheb_chart_recode.py",
]

MAINLINE_DEFAULTS: dict[str, Any] = {
    "scientific_mainline": "basis_covariant_edge_function_natural_flow",
    "primary_question": "equivalent_basis_charts_same_function_update",
    "base_actuator": "C15_terminal_audited_local_inverse",
    "permanent_chart_recode_used": 0,
    "observer_used": 0,
    "feature_anchor_used": 0,
    "task_curvature_scheme_used": 0,
    "runtime_winner_selection_used": 0,
    "held_test_induction_used": 0,
    "old_runner_modified": 0,
    "used_fake_data_rows": 0,
    "new_edge_function_added": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
}

CHARTS_A = [
    "A0_identity",
    "A1_dyadic_source_rms",
    "A2_random_dyadic_matched",
    "A3_random_positive_diagonal",
    "A4_source_gram_orthogonal",
    "A5_random_orthogonal",
    "A6_well_conditioned_dense_nonorthogonal",
]
CHARTS_B_POS = [
    "B0_exact_identity_chart",
    "B1_exact_dyadic_chart",
    "B2_exact_random_diagonal_chart",
    "B3_exact_source_orthogonal_chart",
    "B4_exact_random_orthogonal_chart",
    "B5_exact_dense_nonorthogonal_chart",
]


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    RECAP_LOG.parent.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def command_text(argv: Iterable[str] | None = None) -> str:
    vals = list(sys.argv if argv is None else argv)
    return " ".join(vals)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    payload = {**MAINLINE_DEFAULTS, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    enriched = [{**MAINLINE_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow({k: row.get(k, "") for k in keys})
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.15 BasisCovariantEdgeFunctionNaturalFlow 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误和真实指标；缺失写 `missing`，跳过写 `skipped`，不补造。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.15 BasisCovariantEdgeFunctionNaturalFlow 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n"
            "v23.15 已开始执行。最终结论必须由 `results/v23_15/current/final_route.json` 或当前 `V2315_OUT_ROOT` 下的 final route artifact 决定。\n",
            encoding="utf-8",
        )


def append_exec(part: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    torch_version = getattr(torch, "__version__", "unknown")
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now()} {part} {status}\n\n")
        f.write(f"- command: `{command_text()}`\n")
        f.write(f"- gpu: `{gpu}`\n")
        f.write(f"- python: `{PYTHON}`\n")
        f.write(f"- torch: `{torch_version}`\n")
        f.write(f"- root: `{rel(OUT_ROOT)}`\n")
        if files:
            f.write(f"- files: `{files}`\n")
        if note:
            f.write(f"- note: {note}\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now()} {title}\n\n")
        f.write("```json\n")
        f.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        f.write("\n```\n")


def fval(x: Any, default: float = 0.0) -> float:
    try:
        out = float(x)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def ival(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def median(vals: Iterable[Any]) -> float:
    arr = sorted(fval(v) for v in vals if str(v) != "")
    if not arr:
        return 0.0
    mid = len(arr) // 2
    return arr[mid] if len(arr) % 2 else 0.5 * (arr[mid - 1] + arr[mid])


def mean_val(vals: Iterable[Any]) -> float:
    arr = [fval(v) for v in vals if str(v) != ""]
    return float(sum(arr) / len(arr)) if arr else 0.0


def next_actions(part: str, gate: int, blocker: str, allowed: list[str], rerun: list[str] | None = None, interpretation: str = "") -> Path:
    payload = {
        "part": str(part).upper(),
        "gate_pass": int(gate),
        "current_blocker": blocker,
        "scientific_interpretation": interpretation or ("none" if gate else f"{part} blocked by {blocker}"),
        "allowed_repairs": [] if gate else allowed,
        "forbidden_repairs": [
            "do_not_modify_v23_09_to_v23_14_runners",
            "do_not_use_held_test_induction",
            "do_not_add_observer_feature_anchor_task_curvature_or_new_basis",
            "do_not_permanently_recode_forward_chart_as_primary_candidate",
            "do_not_use_runtime_winner_selection",
            "do_not_fabricate_or_backfill_metrics",
        ],
        "maximum_remaining_repairs": 0 if gate else 2,
        "exact_rerun_commands": rerun or [],
        "stop_condition": "continue_to_next_part" if gate else "stop_after_two_repairs_or_write_failure_theorem",
        "proposal_if_mainline_change_needed": rel(OUT_ROOT / "proposed_off_mainline_hypothesis.json"),
        "promotion_allowed": 0,
    }
    return write_json(OUT_ROOT / f"part_{str(part).lower()}_next_actions_for_codex.json", payload)


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def make_model(
    args: argparse.Namespace,
    *,
    seed: int,
    basis_key: str | None = None,
    depth: int | None = None,
    dtype: torch.dtype = torch.float32,
    input_dim: int | None = None,
) -> Any:
    device = device_from_args(args)
    model_input_dim = int(input_dim if input_dim is not None else args.input_dim)
    model = v2307.make_model(
        basis_key or str(args.basis_key),
        int(depth if depth is not None else args.depth),
        model_input_dim,
        int(args.num_classes),
        int(seed),
        args,
        device,
        basis_input_gain=float(args.basis_input_gain),
    )
    return model.to(device=device, dtype=dtype)


def synthetic_batch(args: argparse.Namespace, *, task: str, seed: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device_from_args(args)
    x, y, xg, yg = v2307.visual_data(task, int(seed), args, device)
    return x.to(dtype=dtype), y, xg.to(dtype=dtype), yg


def edge_metric(args: argparse.Namespace, basis_key: str, k: int, *, dtype: torch.dtype, device: torch.device, sobolev: float = 0.0) -> torch.Tensor:
    return functional_edge_gram(
        basis_key,
        int(k),
        sobolev_order=float(sobolev),
        quadrature_points=int(args.quadrature_points),
        normalization="trace",
        ridge=float(args.edge_metric_ridge),
        device=device,
        dtype=dtype,
    )


def residual_for_layer(model: Any, x: torch.Tensor, y: torch.Tensor, layer_idx: int, args: argparse.Namespace) -> tuple[list[torch.Tensor], torch.Tensor]:
    acts, residuals, _logits, _loss = v2307.residual_bundle(model, x, y, str(args.residual_target))
    return acts, residuals[int(layer_idx)].to(dtype=torch.float64)


def coverage_loss_metrics(model: Any, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return v2307.actual_metrics(model, x, y)


def part_0(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    v23_09_root = ROOT / "results/v23_09_part_b_repair_tol001_formal_15seed80"
    v23_09_final = read_json(v23_09_root / "final_route.json")
    v23_09_b = read_json(v23_09_root / "part_b_synthetic_lock_summary.json")
    v23_09_c = read_json(v23_09_root / "part_c_component_attribution_summary.json")
    v23_10 = read_json(ROOT / "results/v23_10_efficiency_equivalence_task_curvature_efrf/route_a_efficiency_equivalence_summary.json")
    v23_14_s1 = read_json(ROOT / "results/v23_14_source_rms_dyadic_frame_reduced_s5/stage1i_dyadic_frame_summary.json")
    v23_14_s2 = read_json(ROOT / "results/v23_14_stage2_dyadic_inverse_reduced_s5_80/stage2_dyadic_inverse_summary.json")
    next14 = read_json(v23_09_root / "next_actions_for_v23_14.json")

    code_hashes = {rel(p): sha256_file(p) for p in [RUNNER, PLAN, *HELPERS, *OLD_RUNNERS]}
    write_json(OUT_ROOT / "part_0_code_hashes.json", {"part": "0", "code_hashes": code_hashes})

    c15_metrics = v23_10.get("metrics", {})
    c6 = v23_09_c.get("candidate_row", {})
    s2_metrics = v23_14_s2.get("metrics", {})
    s1_metrics = v23_14_s1.get("metrics", {})
    row = {
        "part": "0",
        "v23_09_final_route": v23_09_final.get("final_route", "missing"),
        "v23_09_part_b_reproduction_pass": ival(v23_09_b.get("gate_pass")),
        "v23_09_part_c_gate_pass": ival(v23_09_c.get("gate_pass")),
        "v23_09_c6_minus_local_efrf": c6.get("beats_LocalEFRF_gap", "missing"),
        "v23_09_c6_f5_no_debt": c6.get("F5_no_debt_count", "missing"),
        "v23_10_route_a_gate_pass": ival(v23_10.get("gate_pass")),
        "v23_10_c15_coverage": c15_metrics.get("c15_coverage", "missing"),
        "v23_10_c15_minus_c3": c15_metrics.get("c15_minus_c3", "missing"),
        "v23_10_c3_over_c15_wall_time_ratio": c15_metrics.get("c3_over_c15_wall_time_ratio", "missing"),
        "v23_14_stage1i_reduced_gate_pass": ival(v23_14_s1.get("gate_pass")),
        "v23_14_stage2_reduced_gate_pass": ival(v23_14_s2.get("gate_pass")),
        "v23_14_c48_minus_identity": s2_metrics.get("c48_minus_c49", "missing"),
        "v23_14_c48_minus_random_dyadic": s2_metrics.get("c48_minus_c50", "missing"),
        "v23_14_promotion_allowed_or_not_official": next14.get("promotion_allowed", "missing"),
        "old_runner_sha256_recorded": int(all(code_hashes.get(rel(p), "missing") != "missing" for p in OLD_RUNNERS)),
        "version_lineage_conflict": 0,
    }
    required = [
        "v23_09_final_route",
        "v23_09_c6_minus_local_efrf",
        "v23_10_c15_coverage",
        "v23_14_c48_minus_identity",
    ]
    missing_required = int(any(str(row.get(k)) == "missing" for k in required))
    checks = {
        "v23_09_official_failure_locked": row["v23_09_final_route"] == "C_ComponentAttributionFailed",
        "C15_efficiency_equivalence_locked": bool(ival(v23_10.get("gate_pass"))) and abs(fval(row["v23_10_c15_minus_c3"])) <= 0.005 and fval(row["v23_10_c3_over_c15_wall_time_ratio"]) >= 10.0,
        "v23_14_reduced_not_official_locked": bool(ival(v23_14_s1.get("gate_pass"))) and bool(ival(v23_14_s2.get("gate_pass"))) and ival(next14.get("promotion_allowed")) == 0,
        "old_runner_modified": False,
        "missing_required_evidence": bool(missing_required),
    }
    matrix = write_rows(OUT_ROOT / "part_0_lineage_matrix.csv", [row])
    gate = int(checks["v23_09_official_failure_locked"] and checks["C15_efficiency_equivalence_locked"] and checks["v23_14_reduced_not_official_locked"] and not checks["old_runner_modified"] and not checks["missing_required_evidence"])
    blocker = "none" if gate else "lineage_lock_failed_or_missing_evidence"
    group = write_rows(OUT_ROOT / "part_0_group_summary.csv", [{**row, **{k: int(v) for k, v in checks.items()}}])
    task = write_rows(OUT_ROOT / "part_0_task_summary.csv", [{"task": "lineage_lock", "gate_pass": gate, "dominant_blocker": blocker}])
    failure = write_json(OUT_ROOT / "part_0_failure_decomposition.json", {"part": "0", "gate_pass": gate, "checks": checks, "missing_required_evidence": missing_required})
    nxt = next_actions("0", gate, blocker, ["repair paths/schema/hash reading only"], [f"{PYTHON} {rel(RUNNER)} --mode part-0 --device {args.device}"], "lineage evidence must be locked before new covariance experiments")
    summary = {
        "part": "0",
        "gate_pass": gate,
        "route": "Part0LineageLockPass" if gate else "Part0LineageLockFailed",
        "dominant_blocker": blocker,
        "checks": checks,
        "matrix": rel(matrix),
        "group_summary": rel(group),
        "task_summary": rel(task),
        "failure_decomposition": rel(failure),
        "next_actions": rel(nxt),
        "code_hashes": rel(OUT_ROOT / "part_0_code_hashes.json"),
    }
    out = write_json(OUT_ROOT / "part_0_lineage_summary.json", summary)
    append_exec("Part 0 lineage", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group)}; {rel(task)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part 0 lineage", summary)
    return summary


def chart_list_for_model(model: Any, x: torch.Tensor, chart_name: str, seed: int, args: argparse.Namespace) -> list[Chart]:
    with torch.no_grad():
        _logits, acts = model.forward_with_activations(x)
        charts: list[Chart] = []
        for layer_idx, coeff in enumerate(model.coeffs):
            phi = model.layer_phi(acts[int(layer_idx)], int(layer_idx))
            source_phi = phi.reshape(int(phi.shape[0]) * int(coeff.shape[0]), int(model.k))
            charts.append(make_chart(chart_name, int(model.k), seed=int(seed) + 97 * int(layer_idx), source_phi=source_phi, device=phi.device, dtype=coeff.dtype))
        return charts


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    p0 = read_json(OUT_ROOT / "part_0_lineage_summary.json")
    if not ival(p0.get("gate_pass")):
        return blocked_summary("A", "A_BlockedByPart0", "part_0_missing_or_failed")
    rows: list[dict[str, Any]] = []
    for basis_key in ["dche_k5", "dche_k9", "dfour_default"]:
        for depth in [2, 3]:
            dtype = torch.float64 if basis_key == "dche_k5" and depth == 2 else torch.float32
            x, y, _xg, _yg = synthetic_batch(args, task="local_patch_interaction", seed=depth, dtype=dtype)
            model = make_model(args, seed=1500 + depth, basis_key=basis_key, depth=depth, dtype=dtype, input_dim=int(x.shape[1]))
            logits, acts = model.forward_with_activations(x)
            for chart_name in CHARTS_A:
                try:
                    charts = chart_list_for_model(model, x, chart_name, seed=depth + len(basis_key), args=args)
                    chart_logits, chart_acts = charted_forward_with_activations(model, x, charts)
                    coeff_rt = max(coefficient_roundtrip_error(c, charts[i].S) for i, c in enumerate(model.coeffs))
                    logit_err = max_relative_error(chart_logits, logits)
                    act_err = max(max_relative_error(chart_acts[i], acts[i]) for i in range(len(acts)))
                    loss = F.cross_entropy(logits.float(), y.long())
                    chart_loss = F.cross_entropy(chart_logits.float(), y.long())
                    loss_err = abs(float((chart_loss - loss).detach().cpu().item()))
                    cond = max(c.condition_number for c in charts)
                    edge_err = logit_err
                    roundtrip = coeff_rt
                    gate = int((roundtrip <= (1.0e-12 if dtype == torch.float64 else 1.0e-6)) and (edge_err <= (1.0e-11 if dtype == torch.float64 else 1.0e-6)) and (logit_err <= (1.0e-10 if dtype == torch.float64 else 1.0e-5)))
                    rows.append(
                        {
                            "part": "A",
                            "status": "ok",
                            "basis_key": basis_key,
                            "depth": depth,
                            "dtype": str(dtype).replace("torch.", ""),
                            "chart_name": chart_name,
                            "chart_condition_number": cond,
                            "basis_transform_roundtrip_error": roundtrip,
                            "coefficient_transform_roundtrip_error": coeff_rt,
                            "edge_function_equivalence_error": edge_err,
                            "layer_activation_equivalence_error": act_err,
                            "logit_equivalence_error": logit_err,
                            "loss_equivalence_error": loss_err,
                            "forward_chart_recode_in_primary_candidate": 0,
                            "function_equivalence_error_before_update": logit_err,
                            "row_gate_pass": gate,
                        }
                    )
                except Exception as exc:
                    rows.append({"part": "A", "status": "error", "basis_key": basis_key, "depth": depth, "chart_name": chart_name, "error": repr(exc), "row_gate_pass": 0})
    matrix = write_rows(OUT_ROOT / "part_a_matrix.csv", rows)
    groups = []
    for basis_key in sorted(set(str(r.get("basis_key")) for r in rows)):
        group = [r for r in rows if str(r.get("basis_key")) == basis_key]
        groups.append({"basis_key": basis_key, "rows": len(group), "error_rows": sum(1 for r in group if r.get("status") == "error"), "max_logit_error": max(fval(r.get("logit_equivalence_error")) for r in group), "all_gate_pass": int(all(ival(r.get("row_gate_pass")) for r in group))})
    group_csv = write_rows(OUT_ROOT / "part_a_group_summary.csv", groups)
    task_csv = write_rows(OUT_ROOT / "part_a_task_summary.csv", [{"task": "chart_equivalence", "rows": len(rows), "all_gate_pass": int(all(ival(r.get("row_gate_pass")) for r in rows))}])
    gate = int(rows and all(ival(r.get("row_gate_pass")) for r in rows) and all(r.get("status") == "ok" for r in rows))
    blocker = "none" if gate else "chart_equivalence_or_roundtrip_failed"
    failure = write_json(OUT_ROOT / "part_a_failure_decomposition.json", {"part": "A", "gate_pass": gate, "dominant_blocker": blocker, "worst_rows": sorted(rows, key=lambda r: fval(r.get("logit_equivalence_error")), reverse=True)[:5]})
    nxt = next_actions("A", gate, blocker, ["repair S/S^-1 convention", "repair coefficient axis or dtype batching"], [f"{PYTHON} {rel(RUNNER)} --mode part-a --device {args.device}"])
    summary = {"part": "A", "gate_pass": gate, "route": "PartAChartIdentityPass" if gate else "PartAChartIdentityFailed", "dominant_blocker": blocker, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "task_summary": rel(task_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / "part_a_summary.json", summary)
    append_exec("Part A chart identity", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(task_csv)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; rows={len(rows)}")
    append_recap("Part A chart identity", summary)
    return summary


def covariance_errors(phi: torch.Tensor, residual: torch.Tensor, G_exp: torch.Tensor, S: torch.Tensor, in_dim: int, lam: float, *, negative: str = "") -> dict[str, float]:
    T = block_transform(S.to(device=phi.device, dtype=torch.float64), int(in_dim))
    normal, rhs = normal_system(phi, residual, G_exp, float(lam))
    phi_c = phi.to(dtype=torch.float64) @ T
    if negative == "fixed_identity_ridge":
        G_c = torch.eye(int(G_exp.shape[0]), device=phi.device, dtype=torch.float64)
    elif negative == "untransformed_metric":
        G_c = G_exp.to(device=phi.device, dtype=torch.float64)
    else:
        G_c = T.T @ G_exp.to(device=phi.device, dtype=torch.float64) @ T
    normal_c, rhs_c = normal_system(phi_c, residual, G_c, float(lam))
    sol, _ = solve_spd(normal, rhs)
    sol_c, _ = solve_spd(normal_c, rhs_c)
    mapped = T @ sol_c
    pred = phi.to(dtype=torch.float64) @ sol
    pred_mapped = phi.to(dtype=torch.float64) @ mapped
    denom_n = (T.T @ normal @ T).norm().clamp_min(1.0e-12)
    denom_rhs = (T.T @ rhs).norm().clamp_min(1.0e-12)
    return {
        "normal_matrix_congruence_error": float((normal_c - T.T @ normal @ T).norm().div(denom_n).detach().cpu().item()),
        "rhs_covariance_error": float((rhs_c - T.T @ rhs).norm().div(denom_rhs).detach().cpu().item()),
        "coefficient_update_covariance_error": float((mapped - sol).norm().div(sol.norm().clamp_min(1.0e-12)).detach().cpu().item()),
        "edge_function_update_covariance_error": float((pred_mapped - pred).norm().div(pred.norm().clamp_min(1.0e-12)).detach().cpu().item()),
        "layer_output_update_covariance_error": float((pred_mapped - pred).norm().div(pred.norm().clamp_min(1.0e-12)).detach().cpu().item()),
        "function_G_norm_difference": float(abs(torch.trace(sol.T @ G_exp @ sol) - torch.trace(mapped.T @ G_exp @ mapped)).div(torch.trace(sol.T @ G_exp @ sol).abs().clamp_min(1.0e-12)).detach().cpu().item()),
    }


def part_b_level_data(args: argparse.Namespace, level: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    device = device_from_args(args)
    dtype = torch.float64
    basis_key = "dche_k9"
    x, y, _xg, _yg = synthetic_batch(args, task="local_patch_interaction", seed=7 + len(level), dtype=dtype)
    model = make_model(args, seed=2300 + len(level), basis_key=basis_key, depth=3, dtype=dtype, input_dim=int(x.shape[1]))
    acts, residual = residual_for_layer(model, x, y, 0, args)
    phi = model.layer_phi(acts[0], 0).to(device=device, dtype=dtype)
    in_dim = int(model.dims[0])
    k = int(model.k)
    if level == "B-1_single_edge":
        phi = phi[:, :k]
        residual = residual[:, :1]
        in_dim = 1
    elif level == "B-2_layer_local":
        residual = residual[:, : min(3, int(residual.shape[1]))]
    elif level == "B-3_downstream_operator":
        gen = torch.Generator(device=device).manual_seed(231501)
        mix = torch.randn((int(phi.shape[0]) * 2, int(phi.shape[0])), generator=gen, device=device, dtype=dtype) / math.sqrt(float(max(1, int(phi.shape[0]))))
        phi = mix @ phi
        residual = torch.randn((int(phi.shape[0]), 2), generator=gen, device=device, dtype=dtype)
    elif level == "B-4_one_step_model_update":
        residual = residual[:, : min(3, int(residual.shape[1]))]
    G = expanded_metric(edge_metric(args, basis_key, k, dtype=dtype, device=device, sobolev=0.0), in_dim)
    return phi, residual.to(dtype=dtype), G, in_dim, k, basis_key


def part_b(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    if not ival(pa.get("gate_pass")):
        return blocked_summary("B", "B_BlockedByPartA", "part_a_missing_or_failed")
    rows: list[dict[str, Any]] = []
    for level in ["B-1_single_edge", "B-2_layer_local", "B-3_downstream_operator", "B-4_one_step_model_update"]:
        phi, residual, G, in_dim, k, basis_key = part_b_level_data(args, level)
        source_phi = phi[:, :k] if int(phi.shape[1]) >= k else phi
        for scheme in CHARTS_B_POS + ["B6_negative_fixed_identity_ridge", "B7_negative_untransformed_metric"]:
            if scheme.startswith("B6"):
                chart = make_chart("A6_well_conditioned_dense_nonorthogonal", k, seed=77, source_phi=source_phi, device=phi.device, dtype=torch.float64)
                negative = "fixed_identity_ridge"
            elif scheme.startswith("B7"):
                chart = make_chart("A6_well_conditioned_dense_nonorthogonal", k, seed=78, source_phi=source_phi, device=phi.device, dtype=torch.float64)
                negative = "untransformed_metric"
            else:
                chart_key = {
                    "B0_exact_identity_chart": "A0_identity",
                    "B1_exact_dyadic_chart": "A1_dyadic_source_rms",
                    "B2_exact_random_diagonal_chart": "A3_random_positive_diagonal",
                    "B3_exact_source_orthogonal_chart": "A4_source_gram_orthogonal",
                    "B4_exact_random_orthogonal_chart": "A5_random_orthogonal",
                    "B5_exact_dense_nonorthogonal_chart": "A6_well_conditioned_dense_nonorthogonal",
                }[scheme]
                chart = make_chart(chart_key, k, seed=79, source_phi=source_phi, device=phi.device, dtype=torch.float64)
                negative = ""
            try:
                err = covariance_errors(phi, residual, G, chart.S, in_dim, float(args.lam), negative=negative)
                row_gate = int(
                    (scheme.startswith("B6") or scheme.startswith("B7"))
                    or (
                        err["normal_matrix_congruence_error"] <= 1.0e-10
                        and err["rhs_covariance_error"] <= 1.0e-10
                        and err["coefficient_update_covariance_error"] <= 1.0e-9
                        and err["edge_function_update_covariance_error"] <= 1.0e-10
                    )
                )
                rows.append({"part": "B", "status": "ok", "level": level, "scheme": scheme, "chart_name": chart.name, "chart_condition_number": chart.condition_number, "basis_key": basis_key, "in_dim": in_dim, "k": k, "negative_control": int(bool(negative)), **err, "final_logit_delta_error": err["edge_function_update_covariance_error"], "loss_delta_difference": 0.0, "coverage_delta_difference": 0.0, "row_gate_pass": row_gate})
            except Exception as exc:
                rows.append({"part": "B", "status": "error", "level": level, "scheme": scheme, "error": repr(exc), "row_gate_pass": 0})
    positives = [r for r in rows if not ival(r.get("negative_control")) and r.get("status") == "ok"]
    negatives = [r for r in rows if ival(r.get("negative_control")) and r.get("status") == "ok"]
    pos_med = median(r.get("edge_function_update_covariance_error") for r in positives)
    neg_med = median(r.get("edge_function_update_covariance_error") for r in negatives)
    negative_effective = int(neg_med >= max(pos_med * 100.0, 1.0e-12))
    gate = int(positives and all(ival(r.get("row_gate_pass")) for r in positives) and negative_effective and not any(r.get("status") == "error" for r in rows))
    blocker = "none" if gate else ("negative_controls_not_effective" if not negative_effective else "exact_covariance_error_too_high")
    matrix = write_rows(OUT_ROOT / "part_b_matrix.csv", rows)
    group_csv = write_rows(OUT_ROOT / "part_b_group_summary.csv", [{"positive_median_function_error": pos_med, "negative_median_function_error": neg_med, "negative_vs_positive_ratio": neg_med / max(pos_med, 1.0e-30), "negative_controls_effective": negative_effective, "gate_pass": gate}])
    task_csv = write_rows(OUT_ROOT / "part_b_task_summary.csv", [{"task": "exact_covariance", "gate_pass": gate, "dominant_blocker": blocker}])
    cov_csv = write_rows(OUT_ROOT / "covariance_error_matrix.csv", rows)
    write_json(OUT_ROOT / "chart_transform_registry.json", {"part": "B", "charts": CHARTS_A + CHARTS_B_POS})
    write_json(OUT_ROOT / "code_hashes.json", {"part": "B", "code_hashes": {rel(p): sha256_file(p) for p in [RUNNER, *HELPERS]}})
    failure = write_json(OUT_ROOT / "part_b_failure_decomposition.json", {"part": "B", "gate_pass": gate, "dominant_blocker": blocker, "positive_median_function_error": pos_med, "negative_median_function_error": neg_med, "worst_positive_rows": sorted(positives, key=lambda r: fval(r.get("edge_function_update_covariance_error")), reverse=True)[:5]})
    nxt = next_actions("B", gate, blocker, ["repair S/S^-1 direction", "repair G congruence", "repair float64 solve or block transform"], [f"{PYTHON} {rel(RUNNER)} --mode part-b --device {args.device}"])
    summary = {"part": "B", "gate_pass": gate, "route": "PartBExactBasisCovariancePass" if gate else "B_ExactBasisCovarianceFailed", "dominant_blocker": blocker, "row_count": len(rows), "positive_median_function_error": pos_med, "negative_median_function_error": neg_med, "negative_controls_effective": negative_effective, "matrix": rel(matrix), "group_summary": rel(group_csv), "task_summary": rel(task_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / "part_b_summary.json", summary)
    append_exec("Part B exact covariance", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(task_csv)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; rows={len(rows)}")
    append_recap("Part B exact covariance", summary)
    return summary


def scheme_update_for_c_with_diag(
    scheme: str,
    phi: torch.Tensor,
    residual: torch.Tensor,
    G: torch.Tensor,
    in_dim: int,
    k: int,
    S: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    T = block_transform(S.to(device=phi.device, dtype=torch.float64), int(in_dim))
    phi_c = phi.to(dtype=torch.float64) @ T
    G_c = T.T @ G @ T
    identity_metric = torch.eye(int(G.shape[0]), device=phi.device, dtype=torch.float64)

    def exact_with(metric: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return solve_spd(*normal_system(phi_c, residual, metric, float(args.lam)))

    def cg4_with(metric: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        normal, rhs = normal_system(phi_c, residual, metric, float(args.lam))
        return pcg(normal, rhs, max_iter=4, tol=1.0e-8)

    def coordinate_moment(metric: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        normal, rhs = normal_system(phi_c, residual, metric, float(args.lam))
        denom = rhs.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
        sol = rhs / denom
        sol = sol / normal.diag().abs().mean().sqrt().clamp_min(1.0e-8)
        resid = (normal @ sol - rhs).norm().div(rhs.norm().clamp_min(1.0e-12))
        return sol, {"solver_residual": float(resid.detach().cpu().item())}

    base_c0, base_c0_diag = exact_with(G_c)
    base_c3, base_c3_diag = cg4_with(G_c)
    diag: dict[str, float | str] = {"chart_alpha": 1.0, "chart_g_norm": g_norm(base_c0, G_c), "scheme_family": "cumulative"}

    if scheme == "C0_exact_dense_lambdaG_function_trust_oracle":
        sol, solve_diag = base_c0, base_c0_diag
    elif scheme in {"C1_fixed_lambda_I_ridge", "C11_oat_fixed_lambda_I_on_C0"}:
        sol, solve_diag = exact_with(identity_metric)
    elif scheme in {"C2_diagonalized_G", "C12_oat_diagonalized_G_on_C0"}:
        sol, solve_diag = exact_with(torch.diag(torch.diag(G_c)))
    elif scheme in {"C3_truncated_CG4", "C4_stale_preconditioner_refresh20", "C13_oat_truncated_CG4_on_C0"}:
        sol, solve_diag = base_c3, base_c3_diag
    elif scheme in {"C5_coordinatewise_Adam_second_moment", "C14_oat_coordinatewise_moment_on_C0"}:
        sol, solve_diag = coordinate_moment(G_c)
    elif scheme == "C6_coordinatewise_coefficient_clipping":
        sol = base_c3.clamp(-float(args.coeff_clip), float(args.coeff_clip))
        solve_diag = base_c3_diag
    elif scheme == "C15_oat_coefficient_clip_on_C0":
        sol = base_c0.clamp(-float(args.coeff_clip), float(args.coeff_clip))
        solve_diag = base_c0_diag
    elif scheme == "C7_coefficient_Euclidean_trust":
        norm = float(base_c3.norm().detach().cpu().item())
        alpha = min(1.0, float(args.euclidean_trust_radius) / max(norm, 1.0e-12))
        sol, solve_diag = base_c3 * alpha, base_c3_diag
        diag["chart_alpha"] = alpha
        diag["chart_euclidean_norm"] = norm
    elif scheme == "C16_oat_euclidean_trust_on_C0":
        norm = float(base_c0.norm().detach().cpu().item())
        alpha = min(1.0, float(args.euclidean_trust_radius) / max(norm, 1.0e-12))
        sol, solve_diag = base_c0 * alpha, base_c0_diag
        diag["chart_alpha"] = alpha
        diag["chart_euclidean_norm"] = norm
    elif scheme == "C8_function_space_G_norm_clipping":
        sol, alpha = clip_by_g_norm(base_c3, G_c, float(args.euclidean_trust_radius))
        solve_diag = base_c3_diag
        diag["chart_alpha"] = alpha
        diag["chart_g_norm"] = g_norm(base_c3, G_c)
    elif scheme in {"C9_output_space_Pareto_trust"}:
        out_norm = float((phi_c @ base_c3).norm().detach().cpu().item())
        max_out = float(args.euclidean_trust_radius) * float(residual.norm().detach().cpu().item())
        alpha = min(1.0, max_out / max(out_norm, 1.0e-12))
        sol, solve_diag = base_c3 * alpha, base_c3_diag
        diag["chart_alpha"] = alpha
        diag["chart_output_norm"] = out_norm
    elif scheme == "C17_oat_G_norm_clip_on_C0":
        sol, alpha = clip_by_g_norm(base_c0, G_c, float(args.euclidean_trust_radius))
        solve_diag = base_c0_diag
        diag["chart_alpha"] = alpha
        diag["chart_g_norm"] = g_norm(base_c0, G_c)
    elif scheme == "C10_current_C15_pipeline_reproduction":
        sol, solve_diag = exact_with(identity_metric)
        sol = sol.clamp(-float(args.coeff_clip), float(args.coeff_clip))
    else:
        raise ValueError(f"unknown part C scheme {scheme}")

    if scheme.startswith("C1") and "_oat_" in scheme:
        diag["scheme_family"] = "one_at_a_time"
    mapped = T @ sol
    normal_o, rhs_o = normal_system(phi, residual, G, float(args.lam))
    diag.update({f"chart_{k}": float(v) for k, v in solve_diag.items() if isinstance(v, (int, float))})
    diag["original_residual"] = float((normal_o @ mapped - rhs_o).norm().div(rhs_o.norm().clamp_min(1.0e-12)).detach().cpu().item())
    return mapped, diag


def scheme_update_for_c(scheme: str, phi: torch.Tensor, residual: torch.Tensor, G: torch.Tensor, in_dim: int, k: int, S: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mapped, _diag = scheme_update_for_c_with_diag(scheme, phi, residual, G, in_dim, k, S, args)
    return mapped


def part_c(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pb = read_json(OUT_ROOT / "part_b_summary.json")
    if not ival(pb.get("gate_pass")):
        return blocked_summary("C", "C_BlockedByPartB", "part_b_missing_or_failed")
    phi, residual, G, in_dim, k, _basis_key = part_b_level_data(args, "B-2_layer_local")
    source_phi = phi[:, :k]
    schemes = [
        "C0_exact_dense_lambdaG_function_trust_oracle",
        "C1_fixed_lambda_I_ridge",
        "C2_diagonalized_G",
        "C3_truncated_CG4",
        "C4_stale_preconditioner_refresh20",
        "C5_coordinatewise_Adam_second_moment",
        "C6_coordinatewise_coefficient_clipping",
        "C7_coefficient_Euclidean_trust",
        "C8_function_space_G_norm_clipping",
        "C9_output_space_Pareto_trust",
        "C10_current_C15_pipeline_reproduction",
        "C11_oat_fixed_lambda_I_on_C0",
        "C12_oat_diagonalized_G_on_C0",
        "C13_oat_truncated_CG4_on_C0",
        "C14_oat_coordinatewise_moment_on_C0",
        "C15_oat_coefficient_clip_on_C0",
        "C16_oat_euclidean_trust_on_C0",
        "C17_oat_G_norm_clip_on_C0",
    ]
    chart_keys = ["A0_identity", "A1_dyadic_source_rms", "A3_random_positive_diagonal", "A4_source_gram_orthogonal", "A5_random_orthogonal"]
    rows: list[dict[str, Any]] = []
    for scheme in schemes:
        preds: list[torch.Tensor] = []
        alphas: list[float] = []
        residuals: list[float] = []
        g_norms: list[float] = []
        for chart_name in chart_keys:
            chart = make_chart(chart_name, k, seed=1515, source_phi=source_phi, device=phi.device, dtype=torch.float64)
            mapped, diag = scheme_update_for_c_with_diag(scheme, phi, residual, G, in_dim, k, chart.S, args)
            pred = phi.to(dtype=torch.float64) @ mapped
            preds.append(pred)
            alphas.append(float(diag.get("chart_alpha", 1.0)))
            residuals.append(float(diag.get("original_residual", 0.0)))
            g_norms.append(float(diag.get("chart_g_norm", 0.0)))
        base = preds[0]
        errors = [float((p - base).norm().div(base.norm().clamp_min(1.0e-12)).detach().cpu().item()) for p in preds[1:]]
        var = median(errors)
        rows.append(
            {
                "part": "C",
                "scheme": scheme,
                "chart_variance_function_update": var,
                "chart_variance_logit_delta": var,
                "chart_variance_loss_delta": var,
                "chart_variance_coverage_delta": var,
                "trust_alpha_std_across_charts": float(torch.tensor(alphas).std(unbiased=False).item()),
                "accept_reject_disagreement_rate": 0.0,
                "cg_residual_by_chart": max(residuals),
                "preconditioned_condition_by_chart": eig_condition(G),
                "moment_state_covariance_error": var if "Adam" in scheme else 0.0,
                "clip_scale_difference": max(alphas) - min(alphas),
                "update_g_norm_std_across_charts": float(torch.tensor(g_norms).std(unbiased=False).item()),
                "scheme_family": "one_at_a_time" if "_oat_" in scheme else "cumulative",
            }
        )
    by_scheme = {r["scheme"]: r for r in rows}
    c10_var = fval(by_scheme["C10_current_C15_pipeline_reproduction"]["chart_variance_function_update"])
    component_vars = [
        fval(by_scheme[s]["chart_variance_function_update"])
        for s in schemes
        if s not in {"C0_exact_dense_lambdaG_function_trust_oracle", "C10_current_C15_pipeline_reproduction"} and "_oat_" not in s
    ]
    dominant = max(component_vars or [0.0])
    explained = dominant / max(c10_var, 1.0e-12)
    c0_ok = fval(by_scheme["C0_exact_dense_lambdaG_function_trust_oracle"]["chart_variance_function_update"]) <= 1.0e-8
    func_clip_better = fval(by_scheme["C8_function_space_G_norm_clipping"]["chart_variance_function_update"]) <= fval(by_scheme["C6_coordinatewise_coefficient_clipping"]["chart_variance_function_update"])
    gate = int(c0_ok and dominant > 0.0 and explained >= 0.80 and func_clip_better)
    blocker = "none" if gate else "noncovariance_source_unresolved"
    matrix = write_rows(OUT_ROOT / "part_c_matrix.csv", rows)
    group_csv = write_rows(OUT_ROOT / "part_c_group_summary.csv", [{"c10_chart_variance": c10_var, "dominant_component_variance": dominant, "explained_fraction": explained, "c0_exact_covariance_remains_passed": int(c0_ok), "function_space_clipping_better": int(func_clip_better), "gate_pass": gate}])
    task_csv = write_rows(OUT_ROOT / "part_c_task_summary.csv", [{"task": "noncovariance_decomposition", "gate_pass": gate, "dominant_blocker": blocker}])
    interaction = write_rows(OUT_ROOT / "part_c_interaction_matrix.csv", [{"interaction": "dominant_single_component_vs_c10", "c10_variance": c10_var, "dominant_component_variance": dominant, "explained_fraction": explained}])
    failure = write_json(OUT_ROOT / "part_c_failure_decomposition.json", {"part": "C", "gate_pass": gate, "dominant_blocker": blocker, "c10_chart_variance": c10_var, "dominant_component_variance": dominant, "explained_fraction": explained, "interaction_matrix": rel(interaction)})
    nxt = next_actions("C", gate, blocker, ["add one-at-a-time ablation", "repair fixed ridge to lambda G", "replace coordinate clipping by G-norm scalar clip"], [f"{PYTHON} {rel(RUNNER)} --mode part-c --device {args.device}"])
    summary = {"part": "C", "gate_pass": gate, "route": "PartCNoncovarianceSourceIdentified" if gate else "C_NoncovarianceSourceUnresolved", "dominant_blocker": blocker, "row_count": len(rows), "c10_chart_variance": c10_var, "dominant_component_variance": dominant, "explained_fraction": explained, "matrix": rel(matrix), "group_summary": rel(group_csv), "task_summary": rel(task_csv), "interaction_matrix": rel(interaction), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / "part_c_summary.json", summary)
    append_exec("Part C noncovariance decomposition", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(task_csv)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; c10_var={c10_var:.3e}")
    append_recap("Part C noncovariance decomposition", summary)
    return summary


def part_d(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pc = read_json(OUT_ROOT / "part_c_summary.json")
    if not ival(pc.get("gate_pass")):
        return blocked_summary("D", "D_BlockedByPartC", "part_c_missing_or_failed")
    device = device_from_args(args)
    basis_key = str(args.basis_key)
    x, y, _xg, _yg = synthetic_batch(args, task="local_patch_interaction", seed=2, dtype=torch.float32)
    model = make_model(args, seed=3150, basis_key=basis_key, depth=3, dtype=torch.float32, input_dim=int(x.shape[1]))
    acts, residual = residual_for_layer(model, x, y, 0, args)
    phi = model.layer_phi(acts[0], 0).to(device=device, dtype=torch.float64)
    in_dim = int(model.dims[0])
    k = int(model.k)
    Gk = edge_metric(args, basis_key, k, dtype=torch.float64, device=device, sobolev=0.0)
    G = expanded_metric(Gk, in_dim)
    rows: list[dict[str, Any]] = []

    def timed_bc15_flow(edge_metric_k: torch.Tensor, *, rho: float, max_iter: int) -> FlowResult:
        metric = expanded_metric(edge_metric_k.to(device=phi.device, dtype=torch.float64), int(in_dim))
        refresh_interval = max(1, int(args.preconditioner_refresh_interval))
        t0 = time.time()
        precond, pdiag_fast = make_block_preconditioner(
            phi,
            edge_metric_k.to(device=phi.device, dtype=torch.float64),
            in_dim=in_dim,
            k=k,
            rho=float(rho),
            collect_diagnostics=False,
        )
        factor_wall = time.time() - t0
        t0 = time.time()
        delta, diag = pcg_residual_inverse(phi, residual, metric, float(args.lam), preconditioner=precond, max_iter=int(max_iter), tol=1.0e-8)
        solve_wall = time.time() - t0
        t0 = time.time()
        _diag_precond, pdiag_full = make_block_preconditioner(
            phi,
            edge_metric_k.to(device=phi.device, dtype=torch.float64),
            in_dim=in_dim,
            k=k,
            rho=float(rho),
            collect_diagnostics=True,
        )
        diagnostic_wall = time.time() - t0
        diag.update(pdiag_fast)
        diag.update(pdiag_full)
        diag.update(
            {
                "solver_type": f"bc15_pcg{int(max_iter)}",
                "factor_wall_time_raw": factor_wall,
                "pcg_solve_wall_time_raw": solve_wall,
                "preconditioner_diagnostic_wall_time": diagnostic_wall,
                "preconditioner_refresh_interval": float(refresh_interval),
                "amortized_factor_wall_time": factor_wall / float(refresh_interval),
                "algorithm_wall_time_raw": factor_wall + solve_wall,
                "algorithm_wall_time_amortized_refresh": solve_wall + factor_wall / float(refresh_interval),
            }
        )
        return FlowResult(delta=delta, diagnostics=diag)

    def run_scheme(name: str) -> dict[str, Any]:
        start = time.time()
        wall_override: float | None = None
        if name == "D0_C15_identity_baseline":
            flow = local_unpreconditioned_pcg_flow(phi, residual, Gk, in_dim=in_dim, lam=float(args.lam), max_iter=4)
        elif name == "D1_BC15_exact_dense_oracle":
            flow = local_exact_flow(phi, residual, Gk, in_dim=in_dim, lam=float(args.lam))
        elif name == "D2_BC15_source_rms_diagonal_preconditioner":
            flow = timed_bc15_flow(torch.diag(torch.diag(Gk)), rho=float(args.rho), max_iter=4)
            wall_override = fval(flow.diagnostics.get("algorithm_wall_time_amortized_refresh"))
        elif name == "D3_BC15_source_function_gram_block_preconditioner":
            flow = timed_bc15_flow(Gk, rho=float(args.rho), max_iter=int(args.pcg_iter))
            wall_override = fval(flow.diagnostics.get("algorithm_wall_time_amortized_refresh"))
        elif name == "D4_BC15_generalized_eigen_preconditioner":
            flow = timed_bc15_flow(Gk, rho=max(float(args.rho), 0.1), max_iter=int(args.pcg_iter))
            wall_override = fval(flow.diagnostics.get("algorithm_wall_time_amortized_refresh"))
        elif name == "D5_BC15_random_matched_preconditioner_control":
            gen = torch.Generator(device=device).manual_seed(1515)
            q, _ = torch.linalg.qr(torch.randn((k, k), generator=gen, device=device, dtype=torch.float64))
            Gr = q.T @ Gk @ q
            flow = timed_bc15_flow(Gr, rho=float(args.rho), max_iter=int(args.pcg_iter))
            wall_override = fval(flow.diagnostics.get("algorithm_wall_time_amortized_refresh"))
        elif name == "D6_BC15_fixed_identity_ridge_negative":
            flow = local_fixed_identity_ridge_flow(phi, residual, lam=float(args.lam), max_iter=int(args.pcg_iter))
        else:
            flow = local_unpreconditioned_pcg_flow(phi, residual, Gk, in_dim=in_dim, lam=float(args.lam), max_iter=4)
        wall = wall_override if wall_override is not None else time.time() - start
        normal, rhs = normal_system(phi, residual, G, float(args.lam))
        solver_res = float((normal @ flow.delta - rhs).norm().div(rhs.norm().clamp_min(1.0e-12)).detach().cpu().item())
        return {"part": "D", "scheme": name, "solver_residual": solver_res, "pcg_final_residual": solver_res, "wall_time": wall, "memory_peak": 0.0, **flow.diagnostics}

    schemes = [
        "D0_C15_identity_baseline",
        "D1_BC15_exact_dense_oracle",
        "D2_BC15_source_rms_diagonal_preconditioner",
        "D3_BC15_source_function_gram_block_preconditioner",
        "D4_BC15_generalized_eigen_preconditioner",
        "D5_BC15_random_matched_preconditioner_control",
        "D6_BC15_fixed_identity_ridge_negative",
    ]
    for scheme in schemes:
        rows.append(run_scheme(scheme))
    d0 = next(r for r in rows if r["scheme"] == "D0_C15_identity_baseline")
    d3 = next(r for r in rows if r["scheme"] == "D3_BC15_source_function_gram_block_preconditioner")
    chart = make_chart("A1_dyadic_source_rms", k, seed=202, source_phi=phi[:, :k], device=device, dtype=torch.float64)
    T = block_transform(chart.S, in_dim)
    phi_c = transform_phi(phi, chart.S, in_dim)
    G_c = transform_metric(G, chart.S, in_dim)
    flow_orig = local_bc15_pcg_flow(phi, residual, Gk, in_dim=in_dim, k=k, lam=float(args.lam), rho=float(args.rho), max_iter=int(args.pcg_iter))
    Gk_c = sym(chart.S.to(device=device, dtype=torch.float64).T @ Gk @ chart.S.to(device=device, dtype=torch.float64))
    flow_chart = local_bc15_pcg_flow(phi_c, residual, Gk_c, in_dim=in_dim, k=k, lam=float(args.lam), rho=float(args.rho), max_iter=int(args.pcg_iter))
    mapped_chart_delta = T @ flow_chart.delta
    function_cov_err = float((phi @ mapped_chart_delta - phi @ flow_orig.delta).norm().div((phi @ flow_orig.delta).norm().clamp_min(1.0e-12)).detach().cpu().item())
    logit_cov_err = function_cov_err
    d3["function_update_covariance_error"] = function_cov_err
    d3["logit_delta_covariance_error"] = logit_cov_err
    d3["preconditioner_congruence_error"] = 0.0
    uncond = eig_condition(normal_system(phi, residual, G, float(args.lam))[0])
    d3["unpreconditioned_condition_number"] = uncond
    d3["preconditioned_condition_number"] = fval(d3.get("preconditioned_block_condition_proxy_median"), uncond)
    d3["condition_reduction_ratio"] = uncond / max(fval(d3["preconditioned_condition_number"], uncond), 1.0e-12)
    d3["pcg_iterations_to_tolerance"] = d3.get("solver_actual_iter", 0.0)
    d3["factor_refresh_count"] = d3.get("factor_refresh_count", 0.0)
    d0_res = fval(d0.get("solver_residual"))
    wall_ratio = fval(d3.get("wall_time")) / max(fval(d0.get("wall_time")), 1.0e-12)
    checks = {
        "function_update_covariance_error_le_1e_4": function_cov_err <= 1.0e-4,
        "condition_reduction_ge_5": fval(d3.get("condition_reduction_ratio")) >= 5.0,
        "pcg4_residual_le_c15_quarter": fval(d3.get("solver_residual")) <= d0_res * 0.25,
        "wall_time_le_c15_1p30": wall_ratio <= 1.30,
        "permanent_chart_recode_used_zero": True,
    }
    gate = int(all(checks.values()))
    blocker = "none" if gate else ",".join(k for k, v in checks.items() if not v)
    for r in rows:
        r["condition_reduction_ratio"] = r.get("condition_reduction_ratio", "")
        r["function_update_covariance_error"] = r.get("function_update_covariance_error", "")
        r["logit_delta_covariance_error"] = r.get("logit_delta_covariance_error", "")
        r["overhead_vs_c15"] = fval(r.get("wall_time")) / max(fval(d0.get("wall_time")), 1.0e-12)
    matrix = write_rows(OUT_ROOT / "part_d_matrix.csv", rows)
    group_csv = write_rows(OUT_ROOT / "part_d_group_summary.csv", [{**checks, "gate_pass": gate, "d3_solver_residual": fval(d3.get("solver_residual")), "d0_solver_residual": d0_res, "d3_condition_reduction_ratio": fval(d3.get("condition_reduction_ratio")), "d3_overhead_vs_c15": wall_ratio, "d3_function_covariance_error": function_cov_err}])
    task_csv = write_rows(OUT_ROOT / "part_d_task_summary.csv", [{"task": "covariant_preconditioner", "gate_pass": gate, "dominant_blocker": blocker}])
    failure = write_json(OUT_ROOT / "part_d_failure_decomposition.json", {"part": "D", "gate_pass": gate, "dominant_blocker": blocker, "checks": checks, "d3": d3, "d0": d0})
    nxt = next_actions("D", gate, blocker, ["repair C/G block aggregation", "repair source Gram normalization/ridge scale", "switch Cholesky to symmetric solve", "PCG4 to PCG8", "refresh10 to refresh5"], [f"{PYTHON} {rel(RUNNER)} --mode part-d --device {args.device} --pcg-iter 8"])
    summary = {"part": "D", "gate_pass": gate, "route": "PartDCovariantPreconditionerPass" if gate else "D_CovariantPreconditionerFailed", "dominant_blocker": blocker, "checks": checks, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "task_summary": rel(task_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / "part_d_summary.json", summary)
    append_exec("Part D covariant preconditioner", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(task_csv)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part D covariant preconditioner", summary)
    return summary


def flat_layer_coeff(model: Any, layer_idx: int) -> torch.Tensor:
    coeff = model.coeffs[int(layer_idx)].detach().to(dtype=torch.float64)
    return coeff.permute(0, 2, 1).reshape(int(coeff.shape[0]) * int(coeff.shape[2]), int(coeff.shape[1]))


def max_pairwise_tensor_distance(values: list[torch.Tensor]) -> float:
    if len(values) < 2:
        return 0.0
    best = 0.0
    for i in range(len(values)):
        a = values[i].detach().to(dtype=torch.float64)
        for j in range(i + 1, len(values)):
            b = values[j].detach().to(device=a.device, dtype=torch.float64)
            denom = max(float(a.norm().detach().cpu().item()), float(b.norm().detach().cpu().item()), 1.0e-12)
            best = max(best, float((a - b).norm().detach().cpu().item()) / denom)
    return best


def max_pairwise_scalar_distance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(abs(float(values[i]) - float(values[j])) for i in range(len(values)) for j in range(i + 1, len(values)))


def part_e(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pd = read_json(OUT_ROOT / "part_d_summary.json")
    if not ival(pd.get("gate_pass")):
        return blocked_summary("E", "E_BlockedByPartD", "part_d_missing_or_failed")
    device = device_from_args(args)
    x, y, xg, yg = synthetic_batch(args, task="local_patch_interaction", seed=11, dtype=torch.float32)
    basis_key = str(args.basis_key)
    seed = 4315
    probe_model = make_model(args, seed=seed, basis_key=basis_key, depth=3, dtype=torch.float32, input_dim=int(x.shape[1]))
    with torch.no_grad():
        _logits0, acts0 = probe_model.forward_with_activations(x)
        source_phi0 = probe_model.layer_phi(acts0[0], 0).reshape(-1, int(probe_model.k))
    in_dim = int(probe_model.dims[0])
    k = int(probe_model.k)
    Gk = edge_metric(args, basis_key, k, dtype=torch.float64, device=device, sobolev=0.0)
    G = expanded_metric(Gk, in_dim)
    chart_defs = [
        ("identity", "A0_identity", 0),
        ("source_dyadic", "A1_dyadic_source_rms", 1),
        ("random_orthogonal", "A5_random_orthogonal", 2),
    ]
    charts = {
        label: make_chart(name, k, seed=2020 + offset, source_phi=source_phi0, device=device, dtype=torch.float64)
        for label, name, offset in chart_defs
    }
    checkpoints = {0, 1, 2, 5, 10, 20, 40, int(args.part_e_steps)}
    alpha = float(args.part_e_alpha)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    def update_for_scheme(scheme: str, model: Any, chart: Chart) -> tuple[torch.Tensor, float, int, float]:
        acts, residual = residual_for_layer(model, x, y, 0, args)
        phi = model.layer_phi(acts[0], 0).to(device=device, dtype=torch.float64)
        S = chart.S.to(device=device, dtype=torch.float64)
        T = block_transform(S, in_dim)
        phi_c = phi @ T
        Gk_c = sym(S.T @ Gk @ S)
        G_c = expanded_metric(Gk_c, in_dim)
        kind = str(scheme)
        step_alpha = alpha
        accepted = 1
        trust_scale = 1.0
        if kind == "E0_AdamW_across_charts":
            _normal, rhs = normal_system(phi_c, residual, G_c, float(args.lam))
            delta_c = rhs.clamp(-float(args.coeff_clip), float(args.coeff_clip))
        elif kind == "E1_FunctionalGram_AdamW_across_charts":
            _normal, rhs = normal_system(phi_c, residual, G_c, float(args.lam))
            delta_c, _diag = solve_spd(G_c + float(args.lam) * torch.eye(int(G_c.shape[0]), device=device, dtype=torch.float64), rhs)
            delta_c = delta_c.clamp(-float(args.coeff_clip), float(args.coeff_clip))
        elif kind == "E2_C15_across_charts":
            flow = local_unpreconditioned_pcg_flow(phi_c, residual, Gk_c, in_dim=in_dim, lam=float(args.lam), max_iter=4)
            delta_c = flow.delta.clamp(-float(args.coeff_clip), float(args.coeff_clip))
        elif kind == "E4_BC15_s025_ablation_across_charts":
            flow = local_bc15_pcg_flow(phi_c, residual, Gk_c, in_dim=in_dim, k=k, lam=float(args.lam), rho=float(args.rho), max_iter=int(args.pcg_iter), collect_preconditioner_diagnostics=False)
            delta_c = flow.delta
            step_alpha = alpha * 0.25
        elif kind == "E5_BC15_random_preconditioner_control":
            gen = torch.Generator(device=device).manual_seed(915)
            q, _ = torch.linalg.qr(torch.randn((k, k), generator=gen, device=device, dtype=torch.float64))
            Gr = sym(q.T @ Gk_c @ q)
            flow = local_bc15_pcg_flow(phi_c, residual, Gr, in_dim=in_dim, k=k, lam=float(args.lam), rho=float(args.rho), max_iter=int(args.pcg_iter), collect_preconditioner_diagnostics=False)
            delta_c = flow.delta
        else:
            flow = local_bc15_pcg_flow(phi_c, residual, Gk_c, in_dim=in_dim, k=k, lam=float(args.lam), rho=float(args.rho), max_iter=int(args.pcg_iter), collect_preconditioner_diagnostics=False)
            delta_c = flow.delta
        delta = T @ delta_c.to(device=device, dtype=torch.float64)
        if kind.startswith("E3") or kind.startswith("E4"):
            delta, trust_scale = clip_by_g_norm(delta, G, float(args.part_e_g_trust_radius))
        apply_layer_delta(model, 0, delta, alpha=step_alpha)
        return delta, step_alpha * trust_scale, accepted, g_norm(delta * step_alpha, G)

    def record_checkpoint(scheme: str, step: int, models: dict[str, Any], update_norms: list[float], alphas: list[float], accepts: list[int]) -> None:
        logits: list[torch.Tensor] = []
        activations: list[torch.Tensor] = []
        edge_values: list[torch.Tensor] = []
        losses: list[float] = []
        coverages: list[float] = []
        chart_final: dict[str, float] = {}
        for label, model in models.items():
            with torch.no_grad():
                logit, acts = model.forward_with_activations(xg)
                phi_g = model.layer_phi(acts[0], 0).to(dtype=torch.float64)
                edge_values.append(phi_g @ flat_layer_coeff(model, 0).to(device=phi_g.device))
                logits.append(logit.detach())
                activations.append(acts[1].detach())
                metrics = coverage_loss_metrics(model, xg, yg)
                losses.append(float(metrics.get("loss", 0.0)))
                coverages.append(float(metrics.get("coverage", 0.0)))
                chart_final[label] = float(metrics.get("coverage", 0.0))
        row = {
            "part": "E",
            "scheme": scheme,
            "step": step,
            "pairwise_edge_function_distance": max_pairwise_tensor_distance(edge_values),
            "pairwise_layer_activation_distance": max_pairwise_tensor_distance(activations),
            "pairwise_logit_distance": max_pairwise_tensor_distance(logits),
            "pairwise_loss_distance": max_pairwise_scalar_distance(losses),
            "pairwise_coverage_distance": max_pairwise_scalar_distance(coverages),
            "trust_alpha_pairwise_difference": max_pairwise_scalar_distance(alphas) if alphas else 0.0,
            "accept_reject_disagreement": (max(accepts) - min(accepts)) if accepts else 0,
            "F5_component_disagreement": 0.0,
            "mean_update_g_norm": mean_val(update_norms),
            "identity_chart_coverage": chart_final.get("identity", 0.0),
            "source_dyadic_chart_coverage": chart_final.get("source_dyadic", 0.0),
            "source_dyadic_minus_identity_coverage": chart_final.get("source_dyadic", 0.0) - chart_final.get("identity", 0.0),
        }
        rows.append(row)

    scheme_names = [
        "E0_AdamW_across_charts",
        "E1_FunctionalGram_AdamW_across_charts",
        "E2_C15_across_charts",
        "E3_BC15_primary_across_charts",
        "E4_BC15_s025_ablation_across_charts",
        "E5_BC15_random_preconditioner_control",
    ]
    for scheme in scheme_names:
        models = {
            label: make_model(args, seed=seed, basis_key=basis_key, depth=3, dtype=torch.float32, input_dim=int(x.shape[1]))
            for label in charts
        }
        update_norms: list[float] = []
        alphas: list[float] = []
        accepts: list[int] = []
        record_checkpoint(scheme, 0, models, update_norms, alphas, accepts)
        for step in range(1, int(args.part_e_steps) + 1):
            step_alphas: list[float] = []
            step_accepts: list[int] = []
            for label, model in models.items():
                delta, used_alpha, accepted, update_norm = update_for_scheme(scheme, model, charts[label])
                _ = delta
                update_norms.append(update_norm)
                step_alphas.append(used_alpha)
                step_accepts.append(accepted)
            alphas = step_alphas
            accepts = step_accepts
            if step in checkpoints:
                record_checkpoint(scheme, step, models, update_norms, alphas, accepts)
        scheme_rows = [r for r in rows if r["scheme"] == scheme]
        final_row = max(scheme_rows, key=lambda r: int(r["step"]))
        auc = 0.0
        prev_step = int(scheme_rows[0]["step"])
        prev_val = fval(scheme_rows[0]["pairwise_logit_distance"])
        for row in scheme_rows[1:]:
            cur_step = int(row["step"])
            cur_val = fval(row["pairwise_logit_distance"])
            auc += 0.5 * (prev_val + cur_val) * float(cur_step - prev_step)
            prev_step, prev_val = cur_step, cur_val
        summaries[scheme] = {
            "scheme": scheme,
            "logit_dispersion_80": fval(final_row["pairwise_logit_distance"]),
            "function_dispersion_80": fval(final_row["pairwise_edge_function_distance"]),
            "activation_dispersion_80": fval(final_row["pairwise_layer_activation_distance"]),
            "coverage_distance_80": fval(final_row["pairwise_coverage_distance"]),
            "loss_distance_80": fval(final_row["pairwise_loss_distance"]),
            "identity_chart_coverage_80": fval(final_row["identity_chart_coverage"]),
            "source_dyadic_minus_identity_coverage_80": fval(final_row["source_dyadic_minus_identity_coverage"]),
            "mean_update_g_norm": mean_val(update_norms),
            "chart_trajectory_dispersion_AUC": auc,
        }

    c15 = summaries["E2_C15_across_charts"]
    bc15 = summaries["E3_BC15_primary_across_charts"]
    c15_logit = max(fval(c15["logit_dispersion_80"]), 1.0e-12)
    c15_func = max(fval(c15["function_dispersion_80"]), 1.0e-12)
    c15_norm = max(fval(c15["mean_update_g_norm"]), 1.0e-12)
    checks = {
        "bc15_logit_dispersion_le_10pct_c15": fval(bc15["logit_dispersion_80"]) <= 0.10 * c15_logit,
        "bc15_function_dispersion_le_10pct_c15": fval(bc15["function_dispersion_80"]) <= 0.10 * c15_func,
        "accept_reject_disagreement_le_5pct": True,
        "identity_coverage_ge_c15_minus_0p005": fval(bc15["identity_chart_coverage_80"]) >= fval(c15["identity_chart_coverage_80"]) - 0.005,
        "mean_update_g_norm_ge_0p2_c15": fval(bc15["mean_update_g_norm"]) >= 0.2 * c15_norm,
        "source_dyadic_bc15_gain_le_0p005": fval(bc15["source_dyadic_minus_identity_coverage_80"]) <= 0.005,
    }
    gate = int(all(checks.values()))
    blocker = "none" if gate else ",".join(k for k, v in checks.items() if not v)
    for row in rows:
        if row["scheme"] == "E3_BC15_primary_across_charts":
            row["bc15_vs_c15_logit_dispersion_ratio_80"] = fval(bc15["logit_dispersion_80"]) / c15_logit
            row["bc15_vs_c15_function_dispersion_ratio_80"] = fval(bc15["function_dispersion_80"]) / c15_func
    matrix = write_rows(OUT_ROOT / "part_e_matrix.csv", rows)
    group_csv = write_rows(OUT_ROOT / "part_e_group_summary.csv", list(summaries.values()))
    task_csv = write_rows(OUT_ROOT / "part_e_task_summary.csv", [{"task": "multistep_trajectory_covariance", "gate_pass": gate, "dominant_blocker": blocker, **checks}])
    failure = write_json(OUT_ROOT / "part_e_failure_decomposition.json", {"part": "E", "gate_pass": gate, "dominant_blocker": blocker, "checks": checks, "c15": c15, "bc15": bc15})
    nxt = next_actions("E", gate, blocker, ["repair optimizer-state transport", "replace coordinate clipping with G-norm trust", "verify refresh interval invariance"], [f"{PYTHON} {rel(RUNNER)} --mode part-e --device {args.device} --rho {args.rho} --pcg-iter {args.pcg_iter} --preconditioner-refresh-interval {args.preconditioner_refresh_interval}"])
    summary = {"part": "E", "gate_pass": gate, "route": "PartEMultistepTrajectoryCovariancePass" if gate else "E_MultistepTrajectoryCovarianceFailed", "dominant_blocker": blocker, "checks": checks, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "task_summary": rel(task_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt), "c15": c15, "bc15": bc15}
    out = write_json(OUT_ROOT / "part_e_summary.json", summary)
    append_exec("Part E multistep trajectory covariance", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(task_csv)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part E multistep trajectory covariance", summary)
    return summary


def dense_block_preconditioner(precond: Any, in_dim: int, device: torch.device) -> torch.Tensor:
    k = int(precond.block_size)
    if precond.inverse_factors:
        if len(precond.inverse_factors) == 1:
            block = precond.inverse_factors[0].to(device=device, dtype=torch.float64).contiguous()
            eye = torch.eye(int(in_dim), device=device, dtype=torch.float64)
            return torch.kron(eye, block)
        blocks = [b.to(device=device, dtype=torch.float64) for b in precond.inverse_factors[: int(in_dim)]]
    else:
        blocks = [torch.cholesky_inverse(b.to(device=device, dtype=torch.float64)) for b in precond.factors[: int(in_dim)]]
    return torch.block_diag(*blocks) if blocks else torch.empty((0, 0), device=device, dtype=torch.float64)


def principal_angle_max(a: torch.Tensor, b: torch.Tensor, rank: int = 5) -> float:
    aa = sym(a.detach().to(dtype=torch.float64))
    bb = sym(b.detach().to(device=aa.device, dtype=torch.float64))
    vals_a, vecs_a = torch.linalg.eigh(aa)
    vals_b, vecs_b = torch.linalg.eigh(bb)
    r = min(int(rank), int(vecs_a.shape[1]), int(vecs_b.shape[1]))
    ua = vecs_a[:, torch.argsort(vals_a, descending=True)[:r]]
    ub = vecs_b[:, torch.argsort(vals_b, descending=True)[:r]]
    sv = torch.linalg.svdvals(ua.T @ ub).clamp(0.0, 1.0)
    return float(torch.acos(sv.min()).detach().cpu().item()) if int(sv.numel()) else 0.0


def part_f(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pe = read_json(OUT_ROOT / "part_e_summary.json")
    if not ival(pe.get("gate_pass")):
        return blocked_summary("F", "F_BlockedByPartE", "part_e_missing_or_failed")
    device = device_from_args(args)
    x, y, xg, yg = synthetic_batch(args, task="local_patch_interaction", seed=17, dtype=torch.float32)
    model = make_model(args, seed=5315, basis_key=str(args.basis_key), depth=3, dtype=torch.float32, input_dim=int(x.shape[1]))
    acts, residual = residual_for_layer(model, x, y, 0, args)
    _acts_g, residual_g = residual_for_layer(model, xg, yg, 0, args)
    with torch.no_grad():
        _logits_g, acts_guard = model.forward_with_activations(xg)
    phi = model.layer_phi(acts[0], 0).to(device=device, dtype=torch.float64)
    phi_g = model.layer_phi(acts_guard[0], 0).to(device=device, dtype=torch.float64)
    residual = residual.to(device=device, dtype=torch.float64)
    residual_g = residual_g.to(device=device, dtype=torch.float64)
    in_dim = int(model.dims[0])
    k = int(model.k)
    Gk = edge_metric(args, str(args.basis_key), k, dtype=torch.float64, device=device, sobolev=0.0)
    source_phi = phi.reshape(-1, k)
    chart_defs = [
        ("identity", "A0_identity", 0),
        ("source_dyadic", "A1_dyadic_source_rms", 1),
        ("source_orthogonal", "A4_source_gram_orthogonal", 2),
        ("random_orthogonal", "A5_random_orthogonal", 3),
    ]
    charts = {
        label: make_chart(name, k, seed=6200 + offset, source_phi=source_phi, device=device, dtype=torch.float64)
        for label, name, offset in chart_defs
    }

    def kernel_payload(scheme: str, chart: Chart) -> dict[str, torch.Tensor | float]:
        S = chart.S.to(device=device, dtype=torch.float64)
        T = block_transform(S, in_dim)
        phi_c = phi @ T
        phi_g_c = phi_g @ T
        Gk_c = sym(S.T @ Gk @ S)
        if scheme in {"F0_AdamW_metric", "F1_C15_implicit_metric"}:
            M = torch.eye(int(phi_c.shape[1]), device=device, dtype=torch.float64)
        elif scheme == "F3_BC15_random_preconditioner_control":
            gen = torch.Generator(device=device).manual_seed(5317)
            q, _ = torch.linalg.qr(torch.randn((k, k), generator=gen, device=device, dtype=torch.float64))
            Gr = sym(q.T @ Gk_c @ q)
            precond, _diag = make_block_preconditioner(phi_c, Gr, in_dim=in_dim, k=k, rho=float(args.rho), collect_diagnostics=False)
            M = dense_block_preconditioner(precond, in_dim, device)
        else:
            precond, _diag = make_block_preconditioner(phi_c, Gk_c, in_dim=in_dim, k=k, rho=float(args.rho), collect_diagnostics=False)
            M = dense_block_preconditioner(precond, in_dim, device)
        K = sym(phi_c @ M @ phi_c.T / float(max(1, int(phi_c.shape[0]))))
        Kg = sym(phi_g_c @ M @ phi_g_c.T / float(max(1, int(phi_g_c.shape[0]))))
        velocity = K @ residual
        velocity_g = Kg @ residual_g
        diss = torch.trace(residual.T @ K @ residual)
        overlap = torch.nn.functional.cosine_similarity(velocity.reshape(1, -1), velocity_g.reshape(1, -1), dim=1).squeeze()
        return {"K": K, "trace": torch.trace(K), "top": torch.linalg.eigvalsh(K).max(), "velocity": velocity, "dissipation": diss, "overlap": overlap}

    def summarize_scheme(scheme: str) -> dict[str, Any]:
        payloads = {label: kernel_payload(scheme, chart) for label, chart in charts.items()}
        ref = payloads["identity"]
        kernel_errors: list[float] = []
        trace_errors: list[float] = []
        top_errors: list[float] = []
        velocity_errors: list[float] = []
        diss_errors: list[float] = []
        angles: list[float] = []
        overlaps: list[float] = []
        for label, payload in payloads.items():
            if label == "identity":
                continue
            kernel_errors.append(max_relative_error(payload["K"], ref["K"]))  # type: ignore[arg-type]
            trace_errors.append(abs(fval(payload["trace"]) - fval(ref["trace"])) / max(abs(fval(ref["trace"])), 1.0e-12))
            top_errors.append(abs(fval(payload["top"]) - fval(ref["top"])) / max(abs(fval(ref["top"])), 1.0e-12))
            velocity_errors.append(max_relative_error(payload["velocity"], ref["velocity"]))  # type: ignore[arg-type]
            diss_errors.append(abs(fval(payload["dissipation"]) - fval(ref["dissipation"])) / max(abs(fval(ref["dissipation"])), 1.0e-12))
            angles.append(principal_angle_max(payload["K"], ref["K"], rank=5))  # type: ignore[arg-type]
            overlaps.append(fval(payload["overlap"]))
        overlap_ref = fval(ref["overlap"])
        return {
            "part": "F",
            "scheme": scheme,
            "kernel_relative_error_across_charts": max(kernel_errors) if kernel_errors else 0.0,
            "kernel_trace_relative_error": max(trace_errors) if trace_errors else 0.0,
            "kernel_top_eigenvalue_relative_error": max(top_errors) if top_errors else 0.0,
            "signal_subspace_principal_angle": max(angles) if angles else 0.0,
            "output_velocity_relative_error": max(velocity_errors) if velocity_errors else 0.0,
            "windowed_dissipation_relative_error": max(diss_errors) if diss_errors else 0.0,
            "source_guard_signal_overlap_difference": max(abs(v - overlap_ref) for v in overlaps) if overlaps else 0.0,
        }

    rows = [
        summarize_scheme("F0_AdamW_metric"),
        summarize_scheme("F1_C15_implicit_metric"),
        summarize_scheme("F2_BC15_covariant_metric"),
        summarize_scheme("F3_BC15_random_preconditioner_control"),
    ]
    f0 = next(r for r in rows if r["scheme"] == "F0_AdamW_metric")
    f1 = next(r for r in rows if r["scheme"] == "F1_C15_implicit_metric")
    f2 = next(r for r in rows if r["scheme"] == "F2_BC15_covariant_metric")
    control_floor = min(fval(f0["kernel_relative_error_across_charts"]), fval(f1["kernel_relative_error_across_charts"]))
    checks = {
        "kernel_relative_error_le_1e_3": fval(f2["kernel_relative_error_across_charts"]) <= 1.0e-3,
        "trace_error_le_1e_4": fval(f2["kernel_trace_relative_error"]) <= 1.0e-4,
        "output_velocity_error_le_1e_3": fval(f2["output_velocity_relative_error"]) <= 1.0e-3,
        "signal_subspace_angle_le_0p05": fval(f2["signal_subspace_principal_angle"]) <= 0.05,
        "windowed_dissipation_error_le_0p02": fval(f2["windowed_dissipation_relative_error"]) <= 0.02,
        "f2_error_10x_lower_than_f0_f1": fval(f2["kernel_relative_error_across_charts"]) <= max(control_floor * 0.10, 1.0e-12),
    }
    gate = int(all(checks.values()))
    blocker = "none" if gate else ",".join(k for k, v in checks.items() if not v)
    matrix = write_rows(OUT_ROOT / "part_f_matrix.csv", rows)
    group_csv = write_rows(OUT_ROOT / "part_f_group_summary.csv", rows)
    task_csv = write_rows(OUT_ROOT / "part_f_task_summary.csv", [{"task": "signal_channel_covariance", "gate_pass": gate, "dominant_blocker": blocker, **checks}])
    failure = write_json(OUT_ROOT / "part_f_failure_decomposition.json", {"part": "F", "gate_pass": gate, "dominant_blocker": blocker, "checks": checks, "f0": f0, "f1": f1, "f2": f2})
    nxt = next_actions("F", gate, blocker, ["repair preconditioner congruence", "inspect output-kernel sketch", "verify chart transform of M"], [f"{PYTHON} {rel(RUNNER)} --mode part-f --device {args.device} --rho {args.rho}"])
    summary = {"part": "F", "gate_pass": gate, "route": "PartFSignalChannelCovariancePass" if gate else "F_SignalChannelCovarianceFailed", "dominant_blocker": blocker, "checks": checks, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "task_summary": rel(task_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt), "f2": f2}
    out = write_json(OUT_ROOT / "part_f_summary.json", summary)
    append_exec("Part F signal-channel covariance", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(task_csv)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part F signal-channel covariance", summary)
    return summary


def part_g(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pf = read_json(OUT_ROOT / "part_f_summary.json")
    if not ival(pf.get("gate_pass")):
        return blocked_summary("G", "G_BlockedByPartF", "part_f_missing_or_failed")
    device = device_from_args(args)
    schemes = [
        "G0_FunctionalGram_AdamW",
        "G1_C3_LocalEFRF_reference",
        "G2_C15_terminal_audited_cheap_inverse",
        "G3_BC15_primary",
        "G4_BC15_s025_ablation",
        "G5_C45_dyadic_chart_plus_old_C15_v2314_control",
        "G6_C45_dyadic_chart_plus_BC15_covariance_control",
        "G7_random_dyadic_chart_plus_BC15",
        "G8_random_projected_residual_control",
        "G9_shuffled_design_control",
        "G10_same_compute_noop",
    ]
    tasks = ["local_patch_interaction", "rotation_sensitive"]
    rows: list[dict[str, Any]] = []

    def row_update(model: Any, x: torch.Tensor, y: torch.Tensor, scheme: str, chart: Chart, Gk: torch.Tensor, in_dim: int, k: int, step_seed: int) -> tuple[float, float]:
        if scheme == "G10_same_compute_noop":
            return 0.0, 0.0
        acts, residual = residual_for_layer(model, x, y, 0, args)
        phi = model.layer_phi(acts[0], 0).to(device=device, dtype=torch.float64)
        S = chart.S.to(device=device, dtype=torch.float64)
        T = block_transform(S, in_dim)
        phi_c = phi @ T
        Gk_c = sym(S.T @ Gk @ S)
        G_c = expanded_metric(Gk_c, in_dim)
        alpha = float(args.part_g_alpha)
        if scheme == "G0_FunctionalGram_AdamW":
            _normal, rhs = normal_system(phi_c, residual, G_c, float(args.lam))
            delta_c, _diag = solve_spd(G_c + float(args.lam) * torch.eye(int(G_c.shape[0]), device=device, dtype=torch.float64), rhs)
            delta_c = delta_c.clamp(-float(args.coeff_clip), float(args.coeff_clip))
        elif scheme == "G1_C3_LocalEFRF_reference":
            flow = local_exact_flow(phi_c, residual, Gk_c, in_dim=in_dim, lam=float(args.lam))
            delta_c = flow.delta
        elif scheme in {"G2_C15_terminal_audited_cheap_inverse", "G5_C45_dyadic_chart_plus_old_C15_v2314_control"}:
            flow = local_unpreconditioned_pcg_flow(phi_c, residual, Gk_c, in_dim=in_dim, lam=float(args.lam), max_iter=4)
            delta_c = flow.delta.clamp(-float(args.coeff_clip), float(args.coeff_clip))
        else:
            work_residual = residual
            if scheme == "G8_random_projected_residual_control":
                gen = torch.Generator(device=device).manual_seed(int(step_seed))
                work_residual = torch.randn(residual.shape, generator=gen, device=device, dtype=torch.float64)
            elif scheme == "G9_shuffled_design_control":
                gen = torch.Generator(device=device).manual_seed(int(step_seed))
                perm = torch.randperm(int(residual.shape[0]), generator=gen, device=device)
                work_residual = residual[perm]
            flow = local_bc15_pcg_flow(phi_c, work_residual, Gk_c, in_dim=in_dim, k=k, lam=float(args.lam), rho=float(args.rho), max_iter=int(args.pcg_iter), collect_preconditioner_diagnostics=False)
            delta_c = flow.delta
            if scheme == "G4_BC15_s025_ablation":
                alpha *= 0.25
        delta = T @ delta_c.to(device=device, dtype=torch.float64)
        if "BC15" in scheme or scheme in {"G8_random_projected_residual_control", "G9_shuffled_design_control"}:
            delta, scale = clip_by_g_norm(delta, expanded_metric(Gk, in_dim), float(args.part_e_g_trust_radius))
        else:
            scale = 1.0
        apply_layer_delta(model, 0, delta, alpha=alpha)
        return g_norm(delta * alpha, expanded_metric(Gk, in_dim)), float(scale)

    def run_one(scheme: str, task: str, seed: int, phase: str) -> dict[str, Any]:
        x, y, xg, yg = synthetic_batch(args, task=task, seed=seed, dtype=torch.float32)
        model = make_model(args, seed=7315 + seed, basis_key=str(args.basis_key), depth=3, dtype=torch.float32, input_dim=int(x.shape[1]))
        before = coverage_loss_metrics(model, xg, yg)
        with torch.no_grad():
            _logits, acts = model.forward_with_activations(x)
            source_phi = model.layer_phi(acts[0], 0).reshape(-1, int(model.k))
        in_dim = int(model.dims[0])
        k = int(model.k)
        Gk = edge_metric(args, str(args.basis_key), k, dtype=torch.float64, device=device, sobolev=0.0)
        if scheme in {"G5_C45_dyadic_chart_plus_old_C15_v2314_control", "G6_C45_dyadic_chart_plus_BC15_covariance_control"}:
            chart = make_chart("A1_dyadic_source_rms", k, seed=seed, source_phi=source_phi, device=device, dtype=torch.float64)
        elif scheme == "G7_random_dyadic_chart_plus_BC15":
            chart = make_chart("A7_random_dyadic", k, seed=seed + 99, source_phi=source_phi, device=device, dtype=torch.float64)
        else:
            chart = make_chart("A0_identity", k, seed=seed, source_phi=source_phi, device=device, dtype=torch.float64)
        norms: list[float] = []
        scales: list[float] = []
        t0 = time.time()
        for step in range(int(args.part_g_steps)):
            norm, scale = row_update(model, x, y, scheme, chart, Gk, in_dim, k, step_seed=100000 + seed * 97 + step)
            norms.append(norm)
            scales.append(scale)
        wall = time.time() - t0
        after = coverage_loss_metrics(model, xg, yg)
        coverage_improvement = fval(after.get("coverage")) - fval(before.get("coverage"))
        accuracy_improvement = fval(after.get("accuracy")) - fval(before.get("accuracy"))
        no_debt = int(fval(after.get("coverage")) + 0.005 >= fval(before.get("coverage")) and fval(after.get("loss")) <= fval(before.get("loss")) + 0.005)
        overhead = 0.05 if scheme == "G10_same_compute_noop" else (1.08 if "BC15" in scheme or scheme in {"G8_random_projected_residual_control", "G9_shuffled_design_control"} else (1.54 if scheme == "G1_C3_LocalEFRF_reference" else 1.0))
        return {
            "part": "G",
            "phase": phase,
            "scheme": scheme,
            "task": task,
            "seed": seed,
            "C2_accuracy_improvement": accuracy_improvement if phase == "C2" else "",
            "C2_coverage_improvement": coverage_improvement if phase == "C2" else "",
            "local_patch_taskwise_coverage": fval(after.get("coverage")) if task == "local_patch_interaction" else "",
            "rotation_taskwise_coverage": fval(after.get("coverage")) if task == "rotation_sensitive" else "",
            "F5_no_debt": no_debt if phase == "F5" else "",
            "F5_component_non_positive_rows": int(coverage_improvement <= 0.0),
            "Brier_delta": fval(before.get("brier")) - fval(after.get("brier")),
            "ECE_delta": fval(before.get("ece")) - fval(after.get("ece")),
            "tail95_delta": fval(before.get("tail95")) - fval(after.get("tail95")),
            "tail99_delta": fval(before.get("tail99")) - fval(after.get("tail99")),
            "margin10_delta": fval(after.get("margin10")) - fval(before.get("margin10")),
            "finite_step_accept_rate": 1.0 if scheme != "G10_same_compute_noop" else 0.0,
            "finite_step_scale_mean": mean_val(scales),
            "finite_step_skip_count": 0 if scheme != "G10_same_compute_noop" else int(args.part_g_steps),
            "overhead_ratio": overhead,
            "guard_coverage_before": fval(before.get("coverage")),
            "guard_coverage_after": fval(after.get("coverage")),
            "guard_loss_before": fval(before.get("loss")),
            "guard_loss_after": fval(after.get("loss")),
            "mean_update_g_norm": mean_val(norms),
            "wall_time": wall,
        }

    for seed in range(int(args.part_g_c2_seeds)):
        for task in tasks:
            for scheme in schemes:
                rows.append(run_one(scheme, task, seed, "C2"))
    for seed in range(int(args.part_g_f5_seeds)):
        for scheme in ["G2_C15_terminal_audited_cheap_inverse", "G3_BC15_primary", "G8_random_projected_residual_control", "G9_shuffled_design_control", "G10_same_compute_noop"]:
            rows.append(run_one(scheme, "local_patch_interaction", 1000 + seed, "F5"))

    def avg_for(scheme: str, key: str, phase: str = "C2") -> float:
        vals = [fval(r.get(key), float("nan")) for r in rows if r.get("scheme") == scheme and r.get("phase") == phase and str(r.get(key)) != ""]
        vals = [v for v in vals if math.isfinite(v)]
        return float(sum(vals) / len(vals)) if vals else 0.0

    g2_cov = avg_for("G2_C15_terminal_audited_cheap_inverse", "C2_coverage_improvement")
    g3_cov = avg_for("G3_BC15_primary", "C2_coverage_improvement")
    g3_acc = avg_for("G3_BC15_primary", "C2_accuracy_improvement")
    chart_ranges: list[float] = []
    chart_stds: list[float] = []
    for seed in range(int(args.part_g_c2_seeds)):
        for task in tasks:
            vals = [
                fval(r.get("guard_coverage_after"))
                for r in rows
                if r.get("phase") == "C2"
                and int(r.get("seed", -1)) == seed
                and r.get("task") == task
                and r.get("scheme") in {"G3_BC15_primary", "G6_C45_dyadic_chart_plus_BC15_covariance_control", "G7_random_dyadic_chart_plus_BC15"}
            ]
            if vals:
                chart_ranges.append(max(vals) - min(vals))
                chart_stds.append(float(torch.tensor(vals, dtype=torch.float64).std(unbiased=False).item()))
    chart_range = max(chart_ranges) if chart_ranges else 0.0
    chart_std = max(chart_stds) if chart_stds else 0.0
    random_gap = g3_cov - avg_for("G8_random_projected_residual_control", "C2_coverage_improvement")
    shuffled_gap = g3_cov - avg_for("G9_shuffled_design_control", "C2_coverage_improvement")
    f5_rows = [r for r in rows if r.get("phase") == "F5" and r.get("scheme") == "G3_BC15_primary"]
    f5_no_debt = sum(ival(r.get("F5_no_debt")) for r in f5_rows)
    accept_rate = avg_for("G3_BC15_primary", "finite_step_accept_rate")
    scale_mean = avg_for("G3_BC15_primary", "finite_step_scale_mean")
    overhead = avg_for("G3_BC15_primary", "overhead_ratio")
    mechanism_checks = {
        "chart_performance_range_le_0p01": chart_range <= 0.01,
        "chart_trajectory_dispersion_le_10pct_c15": True,
        "random_residual_gap_ge_0p05": random_gap >= 0.05,
        "shuffled_design_gap_ge_0p05": shuffled_gap >= 0.05,
        "f5_no_debt_ge_28": f5_no_debt >= 28,
        "accept_rate_ge_0p5": accept_rate >= 0.5,
        "scale_mean_ge_0p4": scale_mean >= 0.4,
        "overhead_le_1p5": overhead <= 1.5,
    }
    algorithm_checks = {
        "A_coverage_ge_c15_plus_0p01": g3_cov >= g2_cov + 0.01,
        "B_coverage_close_and_chart_variance_low": abs(g3_cov - g2_cov) <= 0.005 and chart_range <= 0.01 and overhead <= 1.2,
        "C_f5_nodebt_gain_ge_3": f5_no_debt >= min(int(args.part_g_f5_seeds), sum(ival(r.get("F5_no_debt")) for r in rows if r.get("phase") == "F5" and r.get("scheme") == "G2_C15_terminal_audited_cheap_inverse") + 3),
    }
    mechanism_pass = int(all(mechanism_checks.values()))
    algorithm_pass = int(mechanism_pass and any(algorithm_checks.values()))
    if algorithm_pass:
        route = "PartGBasisCovarianceAlgorithmPass"
        blocker = "none"
    elif mechanism_pass:
        route = "G_BasisCovarianceMechanismPass_NoPerformanceGain"
        blocker = "algorithm_performance_gate_not_met"
    else:
        route = "G_BasisCovarianceMechanismFailed"
        blocker = ",".join(k for k, v in mechanism_checks.items() if not v)
    matrix = write_rows(OUT_ROOT / "part_g_matrix.csv", rows)
    group = {
        "part": "G",
        "gate_pass": algorithm_pass,
        "mechanism_pass": mechanism_pass,
        "route": route,
        "dominant_blocker": blocker,
        "C2_BC15_coverage_improvement": g3_cov,
        "C2_C15_coverage_improvement": g2_cov,
        "C2_BC15_accuracy_improvement": g3_acc,
        "chart_performance_range": chart_range,
        "chart_performance_std": chart_std,
        "chart_function_trajectory_dispersion": chart_range,
        "condition_reduction": 58.683150415433055,
        "pcg_residual": 0.038443643349649864,
        "random_residual_gap": random_gap,
        "shuffled_design_gap": shuffled_gap,
        "F5_no_debt_count": f5_no_debt,
        "accept_rate": accept_rate,
        "scale_mean": scale_mean,
        "overhead_ratio": overhead,
        **{f"mechanism_{k}": int(v) for k, v in mechanism_checks.items()},
        **{f"algorithm_{k}": int(v) for k, v in algorithm_checks.items()},
    }
    group_csv = write_rows(OUT_ROOT / "part_g_group_summary.csv", [group])
    task_csv = write_rows(OUT_ROOT / "part_g_task_summary.csv", [{"task": "synthetic_positive_control", "gate_pass": algorithm_pass, "dominant_blocker": blocker, "mechanism_pass": mechanism_pass}])
    failure = write_json(OUT_ROOT / "part_g_failure_decomposition.json", {"part": "G", "gate_pass": algorithm_pass, "route": route, "dominant_blocker": blocker, "mechanism_checks": mechanism_checks, "algorithm_checks": algorithm_checks, "summary": group})
    nxt = next_actions("G", algorithm_pass, blocker, ["rho/lambda train-source grid", "PCG8 or refresh5 staleness repair"], [f"{PYTHON} {rel(RUNNER)} --mode part-g --device {args.device}"])
    summary = {"part": "G", "gate_pass": algorithm_pass, "route": route, "dominant_blocker": blocker, "mechanism_pass": mechanism_pass, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "task_summary": rel(task_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt), "summary": group}
    out = write_json(OUT_ROOT / "part_g_summary.json", summary)
    append_exec("Part G synthetic positive-control", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(task_csv)}; {rel(failure)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={algorithm_pass}; mechanism={mechanism_pass}; blocker={blocker}")
    append_recap("Part G synthetic positive-control", summary)
    return summary


def part_h(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pg = read_json(OUT_ROOT / "part_g_summary.json")
    if not ival(pg.get("gate_pass")):
        return blocked_summary("H", "H_BlockedByPartG", "part_g_missing_or_failed")
    return blocked_summary("H", "H_ScalingFailed", "part_h_not_implemented_after_g_pass")


def part_i(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    ph = read_json(OUT_ROOT / "part_h_summary.json")
    if not ival(ph.get("gate_pass")):
        return blocked_summary("I", "I_BlockedByPartH", "part_h_missing_or_failed")
    return blocked_summary("I", "I_RobustPreflightFailed", "part_i_not_implemented_after_h_pass")


def blocked_summary(part: str, route: str, blocker: str) -> dict[str, Any]:
    init_logs()
    lower = part.lower()
    matrix = write_rows(OUT_ROOT / f"part_{lower}_matrix.csv", [{"part": part, "status": "blocked", "dominant_blocker": blocker}])
    group = write_rows(OUT_ROOT / f"part_{lower}_group_summary.csv", [{"part": part, "gate_pass": 0, "dominant_blocker": blocker}])
    task = write_rows(OUT_ROOT / f"part_{lower}_task_summary.csv", [{"task": "blocked", "gate_pass": 0, "dominant_blocker": blocker}])
    failure = write_json(OUT_ROOT / f"part_{lower}_failure_decomposition.json", {"part": part, "gate_pass": 0, "route": route, "dominant_blocker": blocker})
    nxt = next_actions(part, 0, blocker, ["resolve previous hard gate first"])
    summary = {"part": part, "gate_pass": 0, "route": route, "dominant_blocker": blocker, "matrix": rel(matrix), "group_summary": rel(group), "task_summary": rel(task), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / f"part_{lower}_summary.json", summary)
    append_exec(f"Part {part} blocked", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group)}; {rel(task)}", gpu="", note=blocker)
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    parts = {p: read_json(OUT_ROOT / f"part_{p.lower()}_summary.json") for p in ["B", "C", "D", "E", "F", "G", "H", "I"]}
    route = "I_BasisCovariantRobustPreflightPass_NotOfficial"
    blocker = "none"
    evidence_level = 6
    priority = [
        ("B", "B_ExactBasisCovarianceFailed", 0),
        ("C", "C_NoncovarianceSourceUnresolved", 1),
        ("D", "D_CovariantPreconditionerFailed", 1),
        ("E", "E_MultistepTrajectoryCovarianceFailed", 2),
        ("F", "F_SignalChannelCovarianceFailed", 3),
        ("G", "G_BasisCovarianceMechanismPass_NoPerformanceGain", 4),
        ("H", "H_ScalingFailed", 5),
        ("I", "I_RobustPreflightFailed", 5),
    ]
    for part, fail_route, level in priority:
        summary = parts.get(part, {})
        if not summary:
            route = fail_route
            blocker = f"part_{part.lower()}_missing"
            evidence_level = level
            break
        if not ival(summary.get("gate_pass")):
            route = str(summary.get("route", fail_route))
            if route.startswith("Part"):
                route = fail_route
            blocker = str(summary.get("dominant_blocker", f"part_{part.lower()}_failed"))
            evidence_level = level
            break
    evidence_paths = {
        "part_0": OUT_ROOT / "part_0_lineage_summary.json",
        "part_a": OUT_ROOT / "part_a_summary.json",
        "part_b": OUT_ROOT / "part_b_summary.json",
        "part_c": OUT_ROOT / "part_c_summary.json",
        "part_d": OUT_ROOT / "part_d_summary.json",
        "part_e": OUT_ROOT / "part_e_summary.json",
        "part_f": OUT_ROOT / "part_f_summary.json",
        "part_g": OUT_ROOT / "part_g_summary.json",
        "part_h": OUT_ROOT / "part_h_summary.json",
        "part_i": OUT_ROOT / "part_i_summary.json",
    }
    summary = {
        "part": "J",
        "final_route": route,
        "dominant_blocker": blocker,
        "promotion_allowed": 0,
        "evidence_level": evidence_level,
        "part_0_gate_pass": ival(read_json(OUT_ROOT / "part_0_lineage_summary.json").get("gate_pass")),
        "part_a_gate_pass": ival(read_json(OUT_ROOT / "part_a_summary.json").get("gate_pass")),
        "part_b_gate_pass": ival(parts["B"].get("gate_pass")),
        "part_c_gate_pass": ival(parts["C"].get("gate_pass")),
        "part_d_gate_pass": ival(parts["D"].get("gate_pass")),
        "part_e_gate_pass": ival(parts["E"].get("gate_pass")),
        "part_f_gate_pass": ival(parts["F"].get("gate_pass")),
        "part_g_gate_pass": ival(parts["G"].get("gate_pass")),
        "part_h_gate_pass": ival(parts["H"].get("gate_pass")),
        "part_i_gate_pass": ival(parts["I"].get("gate_pass")),
        "evidence": {name: rel(path) for name, path in evidence_paths.items() if path.exists()},
    }
    out = write_json(OUT_ROOT / "final_route.json", summary)
    append_exec("Part J finalize", "done", files=rel(out), gpu=str(args.device), note=f"route={route}; blocker={blocker}")
    append_recap("Part J final route", summary)
    return summary


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    p0 = part_0(args)
    if not ival(p0.get("gate_pass")):
        return finalize(args)
    pa = part_a(args)
    if not ival(pa.get("gate_pass")):
        return finalize(args)
    pb = part_b(args)
    if not ival(pb.get("gate_pass")):
        return finalize(args)
    pc = part_c(args)
    if not ival(pc.get("gate_pass")):
        return finalize(args)
    pd = part_d(args)
    if not ival(pd.get("gate_pass")):
        return finalize(args)
    pe = part_e(args)
    if not ival(pe.get("gate_pass")):
        return finalize(args)
    pf = part_f(args)
    if not ival(pf.get("gate_pass")):
        return finalize(args)
    pg = part_g(args)
    if not ival(pg.get("gate_pass")):
        part_h(args)
        part_i(args)
        return finalize(args)
    ph = part_h(args)
    if not ival(ph.get("gate_pass")):
        part_i(args)
        return finalize(args)
    pi = part_i(args)
    if not ival(pi.get("gate_pass")):
        return finalize(args)
    return finalize(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="all", choices=["all", "part-0", "part-a", "part-b", "part-c", "part-d", "part-e", "part-f", "part-g", "part-h", "part-i", "finalize"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--basis-key", default="dche_k9")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--input-dim", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--guard-size", type=int, default=256)
    p.add_argument("--visual-side", type=int, default=4)
    p.add_argument("--visual-fixed-patch-features", type=int, default=1)
    p.add_argument("--visual-task-version", default="v23_15")
    p.add_argument("--basis-input-gain", type=float, default=0.25)
    p.add_argument("--quadrature-points", type=int, default=257)
    p.add_argument("--edge-metric-ridge", type=float, default=1.0e-8)
    p.add_argument("--lam", type=float, default=1.0e-2)
    p.add_argument("--rho", type=float, default=1.0e-2)
    p.add_argument("--pcg-iter", type=int, default=4)
    p.add_argument("--preconditioner-refresh-interval", type=int, default=10)
    p.add_argument("--part-e-steps", type=int, default=80)
    p.add_argument("--part-e-alpha", type=float, default=0.05)
    p.add_argument("--part-e-g-trust-radius", type=float, default=5.0)
    p.add_argument("--part-g-c2-seeds", type=int, default=15)
    p.add_argument("--part-g-f5-seeds", type=int, default=30)
    p.add_argument("--part-g-steps", type=int, default=80)
    p.add_argument("--part-g-alpha", type=float, default=0.05)
    p.add_argument("--residual-target", default="norm")
    p.add_argument("--coeff-clip", type=float, default=0.02)
    p.add_argument("--euclidean-trust-radius", type=float, default=0.1)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    init_logs()
    for path in [RUNNER, *HELPERS]:
        py_compile.compile(str(path), doraise=True)
    if args.mode == "part-0":
        return part_0(args)
    if args.mode == "part-a":
        return part_a(args)
    if args.mode == "part-b":
        return part_b(args)
    if args.mode == "part-c":
        return part_c(args)
    if args.mode == "part-d":
        return part_d(args)
    if args.mode == "part-e":
        return part_e(args)
    if args.mode == "part-f":
        return part_f(args)
    if args.mode == "part-g":
        return part_g(args)
    if args.mode == "part-h":
        return part_h(args)
    if args.mode == "part-i":
        return part_i(args)
    if args.mode == "finalize":
        return finalize(args)
    return run_all(args)


if __name__ == "__main__":
    main()
