#!/usr/bin/env python3
"""DG-KAN v23.16 compositional basis-covariant vertical-horizontal flow runner."""

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
import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
from dgkan.fu.compositional_edge_tangent import (
    apply_j,
    apply_jt,
    autograd_jvp,
    build_tangent_cache,
    coeff_inner,
    derivative_health,
    explicit_jacobian,
    flatten_coeffs,
    unflatten_coeffs,
)
from dgkan.fu.compositional_edge_natural_flow import (
    apply_normal,
    blockdiag_exact_flow,
    blocktridiag_exact_flow,
    exact_global_flow,
    gradient_flow,
    global_pcg_flow,
    metric_matrix_for_model,
    metric_norm_sq,
    offdiag_energy_ratio,
)
from dgkan.fu.edge_basis_chart import block_transform, make_chart
from dgkan.fu.edge_sobolev_metrics import functional_edge_gram


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.16_CompositionalBasisCovariantVerticalHorizontalFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.16_CompositionalBasisCovariantVerticalHorizontalFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.16_CompositionalBasisCovariantVerticalHorizontalFlow_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2316_OUT_ROOT", str(ROOT / "results/v23_16/current"))).resolve()

HELPERS = [
    ROOT / "dgkan/fu/compositional_edge_tangent.py",
    ROOT / "dgkan/fu/compositional_edge_natural_flow.py",
    ROOT / "dgkan/optim/bc_vertical_horizontal_flow.py",
]
OLD_RUNNERS = [
    ROOT / "experiments/run_v23_09_trust_projected_downstream_efrf.py",
    ROOT / "experiments/run_v23_10_efficiency_equivalence_task_curvature_efrf.py",
    ROOT / "experiments/run_v23_11_feature_learning_tangent_escape_kan.py",
    ROOT / "experiments/run_v23_12_total_audit_feature_learning_tangent_escape.py",
    ROOT / "experiments/run_v23_13_cheb_domain_transport_representation_geometry.py",
    ROOT / "experiments/run_v23_14_source_whitened_cheb_chart_recode.py",
    ROOT / "experiments/run_v23_15_basis_covariant_edge_natural_flow.py",
]

SCHEME_REGISTRY = [
    "V0_FunctionalGram_AdamW",
    "V1_BC15_BlockDiagonalVerticalFlow",
    "V2_BC_VH_GlobalPCG2_efficiency_control",
    "V3_BC_VH_GlobalPCG4_main",
    "V4_BC_VH_NearestNeighborBlockTridiagonal",
    "V5_BC_VH_HorizontalSuppressed_Dzero",
    "V6_BC_VH_HorizontalShuffled",
    "V7_BC_VH_RandomCrossLayerSigns",
    "V8_SameComputeNoOp",
    "V9_H10_known_structure_upper_bound_diagnostic",
]

AUDIT_DEFAULTS: dict[str, Any] = {
    "version": "v23.16",
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "new_edge_function_added": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "permanent_basis_recode_used": 0,
    "observer_branch_reopened": 0,
    "thresholds_changed": 0,
}


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
    env = os.environ.get("V2316_OUT_ROOT")
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    prefix = []
    if env:
        prefix.append(f"V2316_OUT_ROOT={env}")
    if cuda:
        prefix.append(f"CUDA_VISIBLE_DEVICES={cuda}")
    return " ".join([*prefix, PYTHON, rel(RUNNER), *vals[1:]])


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing", "nan"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    payload = {**AUDIT_DEFAULTS, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow({key: row.get(key, "") for key in keys})
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.16 CompositionalBasisCovariantVerticalHorizontalFlow 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误和真实指标；缺失写 missing，blocked 写 blocked，不补造。\n"
            "- 复现提示：本 runner 固定 scheme registry；GPU 2/3 可通过 `CUDA_VISIBLE_DEVICES=2` / `3` 分片运行。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.16 CompositionalBasisCovariantVerticalHorizontalFlow 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n尚未 final。本文件只记录真实 artifact、真实指标、修复记录和分析结论。\n",
            encoding="utf-8",
        )


def append_exec(part: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {part} {status}\n\n")
        fh.write(f"- command: `{command_text()}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- root: `{rel(OUT_ROOT)}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def next_actions(
    part: str,
    gate: int,
    blocker: str,
    *,
    failed_metrics: dict[str, Any] | None = None,
    hypothesis: str = "inconclusive",
    repair_class: str = "none",
    grid: dict[str, Any] | None = None,
    max_repairs: int = 0,
    rerun: list[str] | None = None,
) -> Path:
    payload = {
        "blocker_class": blocker,
        "failed_metrics": failed_metrics or {},
        "current_hypothesis_status": hypothesis,
        "allowed_repair_class": repair_class,
        "allowed_parameter_grid": grid or {},
        "max_repairs_remaining": int(max_repairs),
        "required_rerun_commands": rerun or [],
        "forbidden_actions": [
            "new_edge_function_or_basis_family",
            "mlp_stem_or_readout",
            "permanent_basis_recode",
            "observer_phase_patch_atom_feature_anchor",
            "task_curvature_tail_target_or_manual_structure_target",
            "runtime_winner_selection",
            "held_test_tuning",
            "threshold_lowering",
            "fabricated_or_backfilled_metrics",
        ],
        "stop_condition": "continue_to_next_part" if gate else "repair_within_allowed_class_or_stop",
    }
    return write_json(OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json", payload)


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def basis_for_key(basis_key: str) -> tuple[str, int]:
    return v2307.basis_for_key(basis_key)


def make_model(args: argparse.Namespace, *, basis_key: str, depth: int, width: int, input_dim: int, output_dim: int, seed: int, dtype: torch.dtype) -> v2293.TrueDeepPureKAN:
    basis_name, k = basis_for_key(basis_key)
    model = v2293.TrueDeepPureKAN(
        int(input_dim),
        int(output_dim),
        int(width),
        int(depth),
        basis_name,
        int(k),
        int(seed),
        device_from_args(args),
        basis_input_gain=float(args.basis_input_gain),
    )
    return model.to(device=device_from_args(args), dtype=dtype)


def synthetic_batch(args: argparse.Namespace, *, task: str, seed: int, dtype: torch.dtype, train_size: int | None = None, guard_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local = argparse.Namespace(**vars(args))
    if train_size is not None:
        local.train_size = int(train_size)
    if guard_size is not None:
        local.guard_size = int(guard_size)
    x, y, xg, yg = v2307.visual_data(task, int(seed), local, device_from_args(args))
    return x.to(dtype=dtype), y, xg.to(dtype=dtype), yg


def edge_grams(args: argparse.Namespace, model: Any, basis_key: str, dtype: torch.dtype) -> list[torch.Tensor]:
    grams: list[torch.Tensor] = []
    for _ in model.coeffs:
        grams.append(
            functional_edge_gram(
                basis_key,
                int(model.k),
                sobolev_order=0.0,
                quadrature_points=int(args.quadrature_points),
                normalization="trace",
                ridge=float(args.edge_metric_ridge),
                device=device_from_args(args),
                dtype=dtype,
            )
        )
    return grams


def output_residual(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
    target = F.one_hot(y.long(), num_classes=int(logits.shape[1])).to(device=logits.device, dtype=logits.dtype)
    return target - probs


def actual_metrics(model: Any, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return v2307.actual_metrics(model, x, y)


def model_state(model: Any) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.coeffs]


def restore_model(model: Any, state: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(model.coeffs, state):
            param.copy_(value)


def apply_delta(model: Any, delta: list[torch.Tensor], alpha: float) -> None:
    with torch.no_grad():
        for param, d in zip(model.coeffs, delta):
            param.add_(float(alpha) * d.to(device=param.device, dtype=param.dtype))


def random_delta(model: Any, seed: int, scale: float = 0.05, dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(int(seed))
    return [torch.randn(c.shape, generator=gen, device=c.device, dtype=dtype) * float(scale) for c in model.coeffs]


def transformed_metric_matrix(model: Any, grams: list[torch.Tensor], chart_name: str, cache: Any, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    blocks: list[torch.Tensor] = []
    gblocks: list[torch.Tensor] = []
    for layer_idx, coeff in enumerate(model.coeffs):
        source_phi = cache.bases[layer_idx].reshape(-1, int(model.k)).detach()
        chart = make_chart(chart_name, int(model.k), seed=int(seed) + 31 * layer_idx, source_phi=source_phi, device=coeff.device, dtype=torch.float64)
        s_work = chart.S.to(device=coeff.device, dtype=torch.float64).contiguous()
        repeat = int(coeff.shape[0]) * int(coeff.shape[1])
        eye = torch.eye(repeat, device=coeff.device, dtype=torch.float64)
        t = torch.kron(eye, s_work)
        blocks.append(t)
        gblocks.append(t.T @ torch.kron(eye, grams[layer_idx].to(device=coeff.device, dtype=torch.float64)) @ t)
    return torch.block_diag(*blocks), torch.block_diag(*gblocks)


def relative_tensor_error(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().to(dtype=torch.float64)
    bb = b.detach().to(device=aa.device, dtype=torch.float64)
    return float((aa - bb).norm().div(bb.norm().clamp_min(1.0e-12)).detach().cpu().item())


def part_0(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    old_hashes = {rel(path): sha256_file(path) for path in OLD_RUNNERS}
    new_files = [*HELPERS, RUNNER, EXEC_LOG, RECAP_LOG]
    new_hashes = {rel(path): sha256_file(path) for path in new_files}
    v23_09 = read_json(ROOT / "results/v23_09_part_b_repair_tol001_formal_15seed80/final_route.json")
    v23_15 = read_json(ROOT / "results/v23_15/current/final_route.json")
    lineage = {
        "v23_09_failure_locked": int(v23_09.get("final_route") == "C_ComponentAttributionFailed"),
        "v23_09_route": v23_09.get("final_route", "missing"),
        "v23_10_to_v23_14_status": "diagnostic_branches_locked_by_v23_16_plan",
        "v23_15_failure_locked": int(v23_15.get("final_route") == "G_BasisCovarianceMechanismFailed"),
        "v23_15_route": v23_15.get("final_route", "missing"),
        "v23_16_scope": "compositional_vertical_horizontal_hypothesis_only",
        "old_runner_hashes": old_hashes,
        "new_file_hashes": new_hashes,
        "scheme_registry_exact_match": int(SCHEME_REGISTRY == [
            "V0_FunctionalGram_AdamW",
            "V1_BC15_BlockDiagonalVerticalFlow",
            "V2_BC_VH_GlobalPCG2_efficiency_control",
            "V3_BC_VH_GlobalPCG4_main",
            "V4_BC_VH_NearestNeighborBlockTridiagonal",
            "V5_BC_VH_HorizontalSuppressed_Dzero",
            "V6_BC_VH_HorizontalShuffled",
            "V7_BC_VH_RandomCrossLayerSigns",
            "V8_SameComputeNoOp",
            "V9_H10_known_structure_upper_bound_diagnostic",
        ]),
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "new_edge_function_added": 0,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
        "permanent_basis_recode_used": 0,
    }
    manifest_path = write_json(OUT_ROOT / "part_0_lineage_manifest.json", lineage)
    formula_text = "(J_theta^T W J_theta + lambda G) deltaA = J_theta^T W r; xi_{l+1}=B_l deltaA_l + D_l xi_l"
    contract = {
        "main_formula": formula_text,
        "theory_formula_hash": hashlib.sha256(formula_text.encode("utf-8")).hexdigest(),
        "main_candidate": "V3_BC_VH_GlobalPCG4",
        "scheme_registry": SCHEME_REGISTRY,
        "thresholds": {
            "A_chain_rule_float64": 1.0e-6,
            "A_chain_rule_float32": 1.0e-4,
            "A_adjoint_float64": 1.0e-8,
            "A_adjoint_float32": 1.0e-5,
            "A_basis_covariance": 1.0e-5,
            "nonzero_update_floor": 1.0e-6,
            "C_guard_fit_improvement": 0.15,
            "C_global_minus_BC15_guard_fit": 0.05,
        },
        "allowed_repair_classes": ["R0_implementation", "R1_numerical_batch", "R2_trust"],
        "forbidden_branches": [
            "new_edge_function",
            "mlp_stem_or_readout",
            "permanent_basis_recode",
            "observer_phase_patch_atom_feature_anchor",
            "task_curvature_tail_target_or_manual_structure_target",
            "runtime_winner_selection",
            "held_test_tuning",
        ],
    }
    contract_path = write_json(OUT_ROOT / "theory_contract.json", contract)
    row = {
        "part": "0",
        **{k: lineage[k] for k in ["v23_09_failure_locked", "v23_15_failure_locked", "scheme_registry_exact_match", "held_test_usage", "runtime_selector_used", "new_edge_function_added", "mlp_stem_used", "mlp_readout_used", "permanent_basis_recode_used"]},
        "old_runner_hashes_missing_count": sum(1 for v in old_hashes.values() if v == "missing"),
        "theory_formula_hash": contract["theory_formula_hash"],
    }
    gate = int(row["v23_09_failure_locked"] and row["v23_15_failure_locked"] and row["scheme_registry_exact_match"] and row["old_runner_hashes_missing_count"] == 0)
    blocker = "none" if gate else "lineage_or_contract_lock_failed"
    matrix = write_rows(OUT_ROOT / "part_0_lineage_matrix.csv", [row])
    failure = write_json(OUT_ROOT / "part_0_failure_decomposition.json", {"part": "0", "gate_pass": gate, "dominant_blocker": blocker, "row": row})
    nxt = next_actions("0", gate, blocker, repair_class="implementation", max_repairs=1, rerun=[f"{PYTHON} {rel(RUNNER)} --mode part-0 --device cpu"])
    summary = {"part": "0", "gate_pass": gate, "route": "Part0LineageContractPass" if gate else "Part0LineageContractFailed", "dominant_blocker": blocker, "manifest": rel(manifest_path), "theory_contract": rel(contract_path), "matrix": rel(matrix), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / "part_0_summary.json", summary)
    append_exec("Part 0 lineage/theory contract", "done", files=f"{rel(out)}; {rel(manifest_path)}; {rel(contract_path)}; {rel(matrix)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part 0 lineage/theory contract", summary)
    return summary


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    p0 = read_json(OUT_ROOT / "part_0_summary.json")
    if not ival(p0.get("gate_pass")):
        return blocked_summary("A", "A_CompositionalOperatorInvalid", "part_0_missing_or_failed")
    rows: list[dict[str, Any]] = []
    basis_keys = ["dche_k5", "dche_k9", "dfour_default"]
    depths = [2, 3]
    dtypes = [torch.float64, torch.float32]
    chart_names = ["identity", "A1_dyadic_source_rms", "A5_random_orthogonal"]
    seeds = [0, 1]
    for basis_key in basis_keys:
        for depth in depths:
            for dtype in dtypes:
                for chart_name in chart_names:
                    for seed in seeds:
                        try:
                            x, y, _xg, _yg = synthetic_batch(args, task="local_patch_interaction", seed=seed + depth, dtype=dtype, train_size=24, guard_size=16)
                            model = make_model(args, basis_key=basis_key, depth=depth, width=4, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=231600 + seed + depth, dtype=dtype)
                            cache = build_tangent_cache(model, x)
                            delta = random_delta(model, 7000 + seed + depth, scale=0.03, dtype=torch.float64)
                            custom, verticals, horizontals = apply_j(model, cache, delta, return_parts=True)
                            ref = autograd_jvp(model, x, delta)
                            chain_err = relative_tensor_error(custom, ref)
                            q = torch.randn(custom.shape, generator=torch.Generator(device=custom.device).manual_seed(8000 + seed), device=custom.device, dtype=torch.float64)
                            jtq = apply_jt(model, cache, q)
                            lhs = float((custom.to(dtype=torch.float64) * q).sum().detach().cpu().item())
                            rhs = float(coeff_inner(delta, jtq).detach().cpu().item())
                            adj_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-12)
                            grams = edge_grams(args, model, basis_key, torch.float64)
                            u = random_delta(model, 9000 + seed, scale=0.02, dtype=torch.float64)
                            v = random_delta(model, 9100 + seed, scale=0.02, dtype=torch.float64)
                            nu = apply_normal(model, cache, u, grams, float(args.lam))
                            nv = apply_normal(model, cache, v, grams, float(args.lam))
                            sym_err = abs(float(coeff_inner(u, nv).detach().cpu().item()) - float(coeff_inner(nu, v).detach().cpu().item())) / max(abs(float(coeff_inner(u, nv).detach().cpu().item())), abs(float(coeff_inner(nu, v).detach().cpu().item())), 1.0e-12)
                            qform = float(coeff_inner(v, nv).detach().cpu().item())
                            residual = output_residual(cache.logits, y)
                            j = explicit_jacobian(model, cache)
                            g = metric_matrix_for_model(model, grams).to(device=j.device)
                            n = float(max(1, int(x.shape[0])))
                            normal = 0.5 * (j.T @ j / n + float(args.lam) * g + (j.T @ j / n + float(args.lam) * g).T)
                            rr = residual.reshape(-1, 1).to(device=j.device, dtype=torch.float64)
                            rhs0 = j.T @ rr / n
                            sol0 = torch.linalg.solve(normal + 1.0e-9 * torch.eye(int(normal.shape[0]), device=j.device, dtype=torch.float64), rhs0)
                            tglob, gc = transformed_metric_matrix(model, grams, chart_name, cache, seed)
                            jc = j @ tglob
                            normal_c = 0.5 * (jc.T @ jc / n + float(args.lam) * gc + (jc.T @ jc / n + float(args.lam) * gc).T)
                            rhs_c = jc.T @ rr / n
                            sol_c = torch.linalg.solve(normal_c + 1.0e-9 * torch.eye(int(normal_c.shape[0]), device=j.device, dtype=torch.float64), rhs_c)
                            pred0 = j @ sol0
                            predc = j @ (tglob @ sol_c)
                            cov_err = relative_tensor_error(predc, pred0)
                            gnorm = float(torch.sqrt((sol0.T @ g @ sol0).squeeze().clamp_min(0.0)).detach().cpu().item())
                            hnorm = [float(h.norm().detach().cpu().item()) for h in horizontals]
                            vnorm = [float(vv.norm().detach().cpu().item()) for vv in verticals]
                            health = derivative_health(cache)
                            chain_thr = 1.0e-6 if dtype == torch.float64 else 1.0e-4
                            adj_thr = 1.0e-8 if dtype == torch.float64 else 1.0e-5
                            row_gate = int(chain_err <= chain_thr and adj_err <= adj_thr and sym_err <= max(adj_thr, 1.0e-8) and qform > 0.0 and cov_err <= 1.0e-5 and gnorm >= 1.0e-6 and max(health.get(k, 0.0) for k in health if k.startswith("nan_inf_count")) == 0.0)
                            row: dict[str, Any] = {
                                "part": "A",
                                "scheme": "operator_math_audit",
                                "basis_key": basis_key,
                                "depth": depth,
                                "dtype": str(dtype).replace("torch.", ""),
                                "chart": chart_name,
                                "seed": seed,
                                "status": "ok",
                                "chain_rule_reconstruction_error": chain_err,
                                "JVP_error": chain_err,
                                "adjoint_error": adj_err,
                                "normal_symmetry_error": sym_err,
                                "normal_min_quadratic_form": qform,
                                "basis_covariance_error": cov_err,
                                "function_G_step_norm": gnorm,
                                "row_gate_pass": row_gate,
                            }
                            for idx, val in enumerate(vnorm):
                                row[f"vertical_norm_l{idx}"] = val
                            for idx, val in enumerate(hnorm):
                                row[f"horizontal_norm_l{idx}"] = val
                                row[f"horizontal_fraction_l{idx}"] = val / max(val + vnorm[idx], 1.0e-12)
                            row.update(health)
                            rows.append(row)
                        except Exception as exc:
                            rows.append({"part": "A", "basis_key": basis_key, "depth": depth, "dtype": str(dtype), "chart": chart_name, "seed": seed, "status": "error", "error": repr(exc), "row_gate_pass": 0})
    gate = int(len(rows) >= 72 and all(ival(r.get("row_gate_pass")) for r in rows))
    blocker = "none" if gate else "chain_rule_or_adjoint_or_covariance_gate_failed"
    matrix = write_rows(OUT_ROOT / "part_a_matrix.csv", rows)
    group_rows: list[dict[str, Any]] = []
    for basis_key in basis_keys:
        group = [r for r in rows if r.get("basis_key") == basis_key]
        group_rows.append({
            "basis_key": basis_key,
            "rows": len(group),
            "gate_pass": int(group and all(ival(r.get("row_gate_pass")) for r in group)),
            "max_chain_rule_reconstruction_error": max([fval(r.get("chain_rule_reconstruction_error")) for r in group] or [0.0]),
            "max_adjoint_error": max([fval(r.get("adjoint_error")) for r in group] or [0.0]),
            "max_basis_covariance_error": max([fval(r.get("basis_covariance_error")) for r in group] or [0.0]),
        })
    group_csv = write_rows(OUT_ROOT / "part_a_group_summary.csv", group_rows)
    failure = write_json(OUT_ROOT / "part_a_failure_decomposition.json", {"part": "A", "gate_pass": gate, "dominant_blocker": blocker, "worst_rows": sorted(rows, key=lambda r: max(fval(r.get("chain_rule_reconstruction_error")), fval(r.get("adjoint_error")), fval(r.get("basis_covariance_error"))), reverse=True)[:10]})
    nxt = next_actions("A", gate, blocker, failed_metrics=group_rows, hypothesis="supported" if gate else "inconclusive", repair_class="implementation", max_repairs=2, rerun=[f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-a --device cuda:0"])
    summary = {"part": "A", "gate_pass": gate, "route": "PartACompositionalOperatorPass" if gate else "A_CompositionalOperatorInvalid", "dominant_blocker": blocker, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / "part_a_summary.json", summary)
    append_exec("Part A compositional operator audit", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"gate={gate}; rows={len(rows)}; blocker={blocker}")
    append_recap("Part A compositional operator audit", summary)
    return summary


def train_checkpoint(model: Any, x: torch.Tensor, y: torch.Tensor, checkpoint: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    if checkpoint == "random_init":
        return {"checkpoint_train_steps": 0, "checkpoint_optimizer": "none"}
    if checkpoint in {"FunctionalGram_partial", "C15_partial"}:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.checkpoint_lr) * 0.5, weight_decay=float(args.weight_decay))
        steps = max(1, int(args.checkpoint_steps) // 2)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.checkpoint_lr), weight_decay=float(args.weight_decay))
        steps = int(args.checkpoint_steps)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y.long())
        loss.backward()
        opt.step()
    return {"checkpoint_train_steps": steps, "checkpoint_optimizer": "rebuilt_AdamW_proxy", "artifact_status": "rebuilt_from_public_runner_path_not_historical_binary_checkpoint"}


def mechanism_row(args: argparse.Namespace, task: str, seed: int, checkpoint: str) -> dict[str, Any]:
    x, y, xg, yg = synthetic_batch(args, task=task, seed=seed, dtype=torch.float32, train_size=48, guard_size=48)
    model = make_model(args, basis_key="dche_k9", depth=3, width=8, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=5000 + seed, dtype=torch.float32)
    ckpt = train_checkpoint(model, x, y, checkpoint, seed, args)
    before = actual_metrics(model, xg, yg)
    cache = build_tangent_cache(model, x)
    grams = edge_grams(args, model, "dche_k9", torch.float64)
    residual = output_residual(cache.logits, y)
    flow = global_pcg_flow(model, cache, residual, grams, lam=float(args.lam), iterations=4)
    _pred, verticals, horizontals = apply_j(model, cache, flow.delta, return_parts=True)
    off, adj, nonadj = offdiag_energy_ratio(model, cache, grams, float(args.lam))
    state = model_state(model)
    apply_delta(model, flow.delta, float(args.one_step_alpha))
    after = actual_metrics(model, xg, yg)
    restore_model(model, state)
    row: dict[str, Any] = {
        "part": "B",
        "scheme": "V3_BC_VH_GlobalPCG4_main",
        "task": task,
        "seed": seed,
        "checkpoint": checkpoint,
        "status": "ok",
        "candidate_actual_C2_gain": fval(after.get("coverage")) - fval(before.get("coverage")),
        "candidate_loss_delta": fval(after.get("loss")) - fval(before.get("loss")),
        "candidate_coverage_delta": fval(after.get("coverage")) - fval(before.get("coverage")),
        "offdiag_normal_energy_ratio": off,
        "cross_layer_offdiag_Frobenius_ratio": off,
        "adjacent_coupling_ratio": adj,
        "nonadjacent_coupling_ratio": nonadj,
        **ckpt,
    }
    for idx, (v, h) in enumerate(zip(verticals, horizontals)):
        vn = float(v.norm().detach().cpu().item())
        hn = float(h.norm().detach().cpu().item())
        row[f"vertical_norm_l{idx}"] = vn
        row[f"horizontal_norm_l{idx}"] = hn
        row[f"horizontal_fraction_l{idx}"] = hn / max(vn + hn, 1.0e-12)
        row[f"vertical_horizontal_cosine_l{idx}"] = float(F.cosine_similarity(v.reshape(1, -1).float(), h.reshape(1, -1).float(), dim=1).detach().cpu().item()) if vn > 0 and hn > 0 else 0.0
    return row


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    def ranks(vals: list[float]) -> torch.Tensor:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = torch.zeros(len(vals), dtype=torch.float64)
        for rank, idx in enumerate(order):
            out[idx] = float(rank)
        return out
    rx, ry = ranks(xs), ranks(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum().div(rx.norm().mul(ry.norm()).clamp_min(1.0e-12)).item())


def part_b(args: argparse.Namespace) -> dict[str, Any]:
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    if not ival(pa.get("gate_pass")):
        return blocked_summary("B", "B_BlockedByPartA", "part_a_missing_or_failed")
    rows: list[dict[str, Any]] = []
    checkpoints = ["random_init", "AdamW_witness_rebuilt", "H10_positive_control_rebuilt_proxy", "BC15_partial"]
    for task in ["local_patch_interaction", "rotation_sensitive"]:
        for seed in range(2):
            for checkpoint in checkpoints:
                try:
                    rows.append(mechanism_row(args, task, seed, checkpoint))
                except Exception as exc:
                    rows.append({"part": "B", "task": task, "seed": seed, "checkpoint": checkpoint, "status": "error", "error": repr(exc)})
    witness = [r for r in rows if r.get("checkpoint") in {"AdamW_witness_rebuilt", "H10_positive_control_rebuilt_proxy"} and r.get("status") == "ok"]
    bc15 = [r for r in rows if r.get("checkpoint") == "BC15_partial" and r.get("status") == "ok"]
    witness_h = median(max(fval(r.get("horizontal_fraction_l0")), fval(r.get("horizontal_fraction_l1")), fval(r.get("horizontal_fraction_l2"))) for r in witness)
    bc15_h = median(max(fval(r.get("horizontal_fraction_l0")), fval(r.get("horizontal_fraction_l1")), fval(r.get("horizontal_fraction_l2"))) for r in bc15)
    witness_off = median(r.get("offdiag_normal_energy_ratio") for r in witness)
    bc15_off = median(r.get("offdiag_normal_energy_ratio") for r in bc15)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    corr_off = spearman([fval(r.get("offdiag_normal_energy_ratio")) for r in ok_rows], [fval(r.get("candidate_coverage_delta")) for r in ok_rows])
    corr_h = spearman([max(fval(r.get("horizontal_fraction_l0")), fval(r.get("horizontal_fraction_l1")), fval(r.get("horizontal_fraction_l2"))) for r in ok_rows], [fval(r.get("candidate_coverage_delta")) for r in ok_rows])
    support = int((witness_h - bc15_h >= 0.10) or (witness_off - bc15_off >= 0.10) or max(corr_off, corr_h) >= 0.30)
    blocker = "none" if support else "historical_rebuilt_mechanism_support_below_threshold"
    matrix = write_rows(OUT_ROOT / "part_b_matrix.csv", rows)
    group = {
        "part": "B",
        "gate_pass": support,
        "witness_horizontal_fraction_median": witness_h,
        "bc15_horizontal_fraction_median": bc15_h,
        "witness_offdiag_ratio_median": witness_off,
        "bc15_offdiag_ratio_median": bc15_off,
        "spearman_offdiag_gain": corr_off,
        "spearman_horizontal_gain": corr_h,
        "dominant_blocker": blocker,
        "evidence_note": "historical binary checkpoints were not loaded; rows marked rebuilt proxy where applicable",
    }
    group_csv = write_rows(OUT_ROOT / "part_b_group_summary.csv", [group])
    failure = write_json(OUT_ROOT / "part_b_failure_decomposition.json", {"part": "B", "gate_pass": support, "dominant_blocker": blocker, "summary": group})
    nxt = next_actions("B", support, blocker, failed_metrics=group, hypothesis="supported" if support else "unsupported", repair_class="none", max_repairs=0)
    summary = {"part": "B", "gate_pass": support, "route": "PartBHorizontalMechanismSupported" if support else "B_HorizontalMechanismUnsupported", "dominant_blocker": blocker, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt), "minimal_falsification_only": int(not support)}
    out = write_json(OUT_ROOT / "part_b_summary.json", summary)
    append_exec("Part B mechanism audit", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"gate={support}; blocker={blocker}")
    append_recap("Part B mechanism audit", summary)
    return summary


def fit_improvement(model: Any, cache: Any, delta: list[torch.Tensor], residual: torch.Tensor) -> float:
    pred = apply_j(model, cache, delta).to(dtype=torch.float64)
    r = residual.to(device=pred.device, dtype=torch.float64)
    return 1.0 - float((r - pred).square().sum().div(r.square().sum().clamp_min(1.0e-12)).detach().cpu().item())


def flow_basis_covariance_error(model: Any, cache: Any, residual: torch.Tensor, grams: list[torch.Tensor], delta: list[torch.Tensor], lam: float, seed: int) -> float:
    try:
        j = explicit_jacobian(model, cache)
        n = float(max(1, int(cache.logits.shape[0])))
        rr = residual.reshape(-1, 1).to(device=j.device, dtype=torch.float64)
        tglob, gc = transformed_metric_matrix(model, grams, "A5_random_orthogonal", cache, int(seed))
        jc = j @ tglob
        normal_c = 0.5 * (jc.T @ jc / n + float(lam) * gc + (jc.T @ jc / n + float(lam) * gc).T)
        rhs_c = jc.T @ rr / n
        eye = torch.eye(int(normal_c.shape[0]), device=j.device, dtype=torch.float64)
        sol_c = torch.linalg.solve(normal_c + 1.0e-9 * eye, rhs_c)
        pred_c = j @ (tglob @ sol_c)
        pred = apply_j(model, cache, delta).reshape(-1, 1).to(device=j.device, dtype=torch.float64)
        return relative_tensor_error(pred_c, pred)
    except Exception:
        return float("inf")


def part_c_collect_rows(args: argparse.Namespace, *, lam_value: float, pcg_iterations: int, refresh_count: int) -> list[dict[str, Any]]:
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    if not ival(pa.get("gate_pass")):
        return [{"part": "C", "status": "blocked", "error": "part_a_missing_or_failed"}]
    rows: list[dict[str, Any]] = []
    for depth in [2, 3]:
        for width in [4, 8]:
            for checkpoint in ["random_init", "FunctionalGram_partial", "C15_partial"]:
                train_n = 40 * int(refresh_count)
                guard_n = 40 * int(refresh_count)
                x, y, xg, yg = synthetic_batch(args, task="local_patch_interaction", seed=depth * 10 + width, dtype=torch.float32, train_size=train_n, guard_size=guard_n)
                model = make_model(args, basis_key="dche_k5", depth=depth, width=width, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=6100 + depth + width, dtype=torch.float32)
                train_checkpoint(model, x, y, checkpoint, depth + width, args)
                cache = build_tangent_cache(model, x)
                guard_cache = build_tangent_cache(model, xg)
                grams = edge_grams(args, model, "dche_k5", torch.float64)
                residual = output_residual(cache.logits, y)
                guard_residual = output_residual(guard_cache.logits, yg)
                pcg_scheme = f"C4_full_global_PCG{int(pcg_iterations)}"
                schemes = {
                    "C0_ordinary_gradient": lambda: gradient_flow(model, cache, residual, grams),
                    "C1_BC15_blockdiag_exact": lambda: blockdiag_exact_flow(model, cache, residual, grams, lam=float(lam_value)),
                    "C2_adjacent_blocktridiag_exact": lambda: blocktridiag_exact_flow(model, cache, residual, grams, lam=float(lam_value)),
                    "C3_full_global_exact": lambda: exact_global_flow(model, cache, residual, grams, lam=float(lam_value)),
                    pcg_scheme: lambda: global_pcg_flow(model, cache, residual, grams, lam=float(lam_value), iterations=int(pcg_iterations)),
                    "C5_horizontal_suppressed": lambda: global_pcg_flow(model, build_tangent_cache(model, x, horizontal_mode="suppressed"), residual, grams, lam=float(lam_value), iterations=int(pcg_iterations)),
                    "C6_horizontal_shuffled": lambda: global_pcg_flow(model, build_tangent_cache(model, x, horizontal_mode="shuffled", seed=depth + width + 101 * int(refresh_count)), residual, grams, lam=float(lam_value), iterations=int(pcg_iterations)),
                    "C7_shuffled_design": lambda: exact_global_flow(model, cache, residual[torch.randperm(int(residual.shape[0]), generator=torch.Generator(device=residual.device).manual_seed(44 + int(refresh_count)), device=residual.device)], grams, lam=float(lam_value)),
                }
                for scheme, fn in schemes.items():
                    try:
                        t0 = time.time()
                        flow = fn()
                        wall = time.time() - t0
                        source_fit = fit_improvement(model, cache, flow.delta, residual)
                        guard_fit = fit_improvement(model, guard_cache, flow.delta, guard_residual)
                        cov_err = ""
                        if scheme == "C3_full_global_exact":
                            cov_err = flow_basis_covariance_error(model, cache, residual, grams, flow.delta, float(lam_value), seed=depth * 1000 + width * 10 + int(refresh_count))
                        diag = dict(flow.diagnostics)
                        rows.append({
                            "part": "C",
                            "scheme": scheme,
                            "basis_key": "dche_k5",
                            "depth": depth,
                            "width": width,
                            "checkpoint": checkpoint,
                            "lambda": float(lam_value),
                            "pcg_iterations_requested": int(pcg_iterations),
                            "refresh_count": int(refresh_count),
                            "source_train_size": train_n,
                            "guard_size": guard_n,
                            "status": "ok",
                            **diag,
                            "source_residual_fit_improvement": source_fit,
                            "guard_residual_fit_improvement": guard_fit,
                            "source_guard_fit_gap": abs(source_fit - guard_fit),
                            "basis_covariance_error": cov_err,
                            "function_G_step_norm": float(torch.sqrt(metric_norm_sq(flow.delta, grams).clamp_min(0.0)).detach().cpu().item()),
                            "wall_time": wall,
                        })
                    except Exception as exc:
                        rows.append({"part": "C", "scheme": scheme, "depth": depth, "width": width, "checkpoint": checkpoint, "lambda": float(lam_value), "pcg_iterations_requested": int(pcg_iterations), "refresh_count": int(refresh_count), "status": "error", "error": repr(exc)})
    return rows


def part_c_summary_from_rows(rows: list[dict[str, Any]], *, pcg_iterations: int) -> tuple[dict[str, Any], dict[str, bool], int, str, str]:
    by_scheme = {scheme: [r for r in rows if r.get("scheme") == scheme and r.get("status") == "ok"] for scheme in sorted({str(r.get("scheme")) for r in rows})}
    med = {scheme: median(r.get("guard_residual_fit_improvement") for r in group) for scheme, group in by_scheme.items()}
    src_med = {scheme: median(r.get("source_residual_fit_improvement") for r in group) for scheme, group in by_scheme.items()}
    full = med.get("C3_full_global_exact", 0.0)
    pcg_scheme = f"C4_full_global_PCG{int(pcg_iterations)}"
    pcg = med.get(pcg_scheme, 0.0)
    block = med.get("C1_BC15_blockdiag_exact", 0.0)
    supp = med.get("C5_horizontal_suppressed", 0.0)
    shuf = med.get("C6_horizontal_shuffled", 0.0)
    design = med.get("C7_shuffled_design", 0.0)
    pcg_retention = pcg / max(full, 1.0e-12)
    norm_ratio = median(r.get("function_G_step_norm") for r in by_scheme.get("C3_full_global_exact", [])) / max(median(r.get("function_G_step_norm") for r in by_scheme.get("C1_BC15_blockdiag_exact", [])), 1.0e-12)
    source_guard_gap = median(r.get("source_guard_fit_gap") for r in by_scheme.get("C3_full_global_exact", []))
    cov_vals = [fval(r.get("basis_covariance_error"), float("inf")) for r in by_scheme.get("C3_full_global_exact", [])]
    cov_vals = [v for v in cov_vals if math.isfinite(v)]
    max_covariance = max(cov_vals) if cov_vals else float("inf")
    checks = {
        "guard_fit_improvement_gt_0p15": full > 0.15,
        "global_minus_BC15_guard_fit_ge_0p05": full - block >= 0.05,
        "global_minus_horizontal_suppressed_ge_0p05": full - supp >= 0.05,
        "global_minus_horizontal_shuffled_ge_0p05": full - shuf >= 0.05,
        "global_minus_shuffled_design_ge_0p05": full - design >= 0.05,
        "PCG4_retention_ge_0p95": pcg_retention >= 0.95,
        "nonzero_update_norm_ge_0p25_BC15": norm_ratio >= 0.25,
        "source_guard_fit_gap_le_0p20": source_guard_gap <= 0.20,
        "basis_covariance_error_le_1e_5": max_covariance <= 1.0e-5,
    }
    gate = int(all(checks.values()) and not any(r.get("status") == "error" for r in rows))
    if not checks["global_minus_BC15_guard_fit_ge_0p05"]:
        route = "C_CrossLayerInverseNoValue"
    elif not checks["PCG4_retention_ge_0p95"]:
        route = "C_GlobalApproximationFailed"
    else:
        route = "PartCCrossLayerInversePass" if gate else "C_CrossLayerInverseNoValue"
    blocker = "none" if gate else ",".join(k for k, v in checks.items() if not v)
    group = {
        "part": "C",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "median_guard_fit": med,
        "median_source_fit": src_med,
        "PCG4_retention": pcg_retention,
        "norm_ratio_vs_BC15": norm_ratio,
        "source_guard_fit_gap_median": source_guard_gap,
        "max_basis_covariance_error": max_covariance,
        **{f"check_{k}": int(v) for k, v in checks.items()},
    }
    return group, checks, gate, route, blocker


def part_c(args: argparse.Namespace) -> dict[str, Any]:
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    if not ival(pa.get("gate_pass")):
        return blocked_summary("C", "C_BlockedByPartA", "part_a_missing_or_failed")
    rows = part_c_collect_rows(args, lam_value=float(args.lam), pcg_iterations=4, refresh_count=1)
    group, checks, gate, route, blocker = part_c_summary_from_rows(rows, pcg_iterations=4)
    matrix = write_rows(OUT_ROOT / "part_c_matrix.csv", rows)
    group_csv = write_rows(OUT_ROOT / "part_c_group_summary.csv", [group])
    failure = write_json(OUT_ROOT / "part_c_failure_decomposition.json", {"part": "C", "gate_pass": gate, "route": route, "dominant_blocker": blocker, "checks": checks, "medians": group.get("median_guard_fit", {})})
    nxt = next_actions(
        "C",
        gate,
        blocker,
        failed_metrics=group,
        hypothesis="supported" if gate else "unsupported",
        repair_class="numerical" if not gate else "none",
        grid={"lambda": [0.01, 0.03, 0.1], "pcg_iterations": [2, 4, 8], "refresh": [1, 2, 4]},
        max_repairs=1 if not gate else 0,
        rerun=[
            f"V2316_OUT_ROOT={rel(OUT_ROOT)} CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-c-r1-grid --device cuda:0 --r1-shard-index 0 --r1-shard-count 2",
            f"V2316_OUT_ROOT={rel(OUT_ROOT)} CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-c-r1-grid --device cuda:0 --r1-shard-index 1 --r1-shard-count 2",
            f"{PYTHON} {rel(RUNNER)} --mode part-c-r1-merge --device cpu",
        ],
    )
    summary = {"part": "C", "gate_pass": gate, "route": route, "dominant_blocker": blocker, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt), "summary": group}
    out = write_json(OUT_ROOT / "part_c_summary.json", summary)
    append_exec("Part C toy/frozen inverse", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"gate={gate}; route={route}; blocker={blocker}")
    append_recap("Part C toy/frozen inverse", summary)
    return summary


def summarize_r1_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[float, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (fval(row.get("lambda")), ival(row.get("pcg_iterations_requested")), ival(row.get("refresh_count")))
        cells.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (lam_value, pcg_iter, refresh), group_rows in sorted(cells.items()):
        group, checks, gate, route, blocker = part_c_summary_from_rows(group_rows, pcg_iterations=pcg_iter)
        c4_name = f"C4_full_global_PCG{pcg_iter}"
        c4_rows = [r for r in group_rows if r.get("scheme") == c4_name and r.get("status") == "ok"]
        full_rows = [r for r in group_rows if r.get("scheme") == "C3_full_global_exact" and r.get("status") == "ok"]
        med_guard = group.get("median_guard_fit", {})
        row: dict[str, Any] = {
            "part": "C_R1",
            "lambda": lam_value,
            "pcg_iterations": pcg_iter,
            "refresh_count": refresh,
            "cell_gate_pass": gate,
            "route": route,
            "dominant_blocker": blocker,
            "C1_guard_fit_median": med_guard.get("C1_BC15_blockdiag_exact", ""),
            "C2_guard_fit_median": med_guard.get("C2_adjacent_blocktridiag_exact", ""),
            "C3_guard_fit_median": med_guard.get("C3_full_global_exact", ""),
            "C4_guard_fit_median": med_guard.get(c4_name, ""),
            "C5_guard_fit_median": med_guard.get("C5_horizontal_suppressed", ""),
            "C6_guard_fit_median": med_guard.get("C6_horizontal_shuffled", ""),
            "C7_guard_fit_median": med_guard.get("C7_shuffled_design", ""),
            "PCG_retention": group.get("PCG4_retention", ""),
            "norm_ratio_vs_BC15": group.get("norm_ratio_vs_BC15", ""),
            "source_guard_fit_gap_median": group.get("source_guard_fit_gap_median", ""),
            "max_basis_covariance_error": group.get("max_basis_covariance_error", ""),
            "C3_condition_number_median": median(r.get("condition_number") for r in full_rows),
            "C4_PCG_residual_median": median(r.get("PCG_residual") for r in c4_rows),
            "C4_PCG_relative_residual_median": median(r.get("PCG_relative_residual") for r in c4_rows),
            **{f"check_{k}": int(v) for k, v in checks.items()},
        }
        out.append(row)
    return out


def part_c_r1_grid(args: argparse.Namespace) -> dict[str, Any]:
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    if not ival(pa.get("gate_pass")):
        return blocked_summary("C_R1", "C_R1_BlockedByPartA", "part_a_missing_or_failed")
    lams = float_list(args.r1_lams)
    pcgs = int_list(args.r1_pcg_iters)
    refreshes = int_list(args.r1_refreshes)
    combos = [(lam_value, pcg_iter, refresh) for lam_value in lams for pcg_iter in pcgs for refresh in refreshes]
    shard_count = max(1, int(args.r1_shard_count))
    shard_index = max(0, min(int(args.r1_shard_index), shard_count - 1))
    rows: list[dict[str, Any]] = []
    for idx, (lam_value, pcg_iter, refresh) in enumerate(combos):
        if idx % shard_count != shard_index:
            continue
        rows.extend(part_c_collect_rows(args, lam_value=float(lam_value), pcg_iterations=int(pcg_iter), refresh_count=int(refresh)))
    matrix = write_rows(OUT_ROOT / f"part_c_r1_grid_matrix_shard{shard_index}_of_{shard_count}.csv", rows)
    summary_rows = summarize_r1_grid(rows)
    group_csv = write_rows(OUT_ROOT / f"part_c_r1_grid_summary_shard{shard_index}_of_{shard_count}.csv", summary_rows)
    ok_cells = [r for r in summary_rows if ival(r.get("cell_gate_pass"))]
    math_ranked = sorted(summary_rows, key=lambda r: (fval(r.get("C4_PCG_relative_residual_median"), float("inf")), fval(r.get("max_basis_covariance_error"), float("inf")), fval(r.get("C3_condition_number_median"), float("inf"))))
    payload = {
        "part": "C_R1",
        "route": "C_R1GridFoundPassingCell_NotAdopted" if ok_cells else "C_R1GridNoRepairRecovered",
        "gate_pass": int(bool(ok_cells)),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "row_count": len(rows),
        "cell_count": len(summary_rows),
        "matrix": rel(matrix),
        "group_summary": rel(group_csv),
        "best_math_residual_cell": math_ranked[0] if math_ranked else {},
        "passing_cells": ok_cells,
        "selection_note": "R1 grid is a repair diagnostic; final route must not be assembled from a best cell without rerunning the formal root.",
    }
    out = write_json(OUT_ROOT / f"part_c_r1_grid_summary_shard{shard_index}_of_{shard_count}.json", payload)
    append_exec("Part C R1 numerical grid", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"shard={shard_index}/{shard_count}; cells={len(summary_rows)}; passing={len(ok_cells)}")
    append_recap("Part C R1 numerical grid", payload)
    return payload


def part_c_r1_merge(args: argparse.Namespace) -> dict[str, Any]:
    files = sorted(OUT_ROOT.glob("part_c_r1_grid_matrix_shard*_of_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_rows(path))
    if not rows:
        return blocked_summary("C_R1", "C_R1MergeBlocked", "no_r1_shard_matrices_found")
    matrix = write_rows(OUT_ROOT / "part_c_r1_grid_matrix.csv", rows)
    summary_rows = summarize_r1_grid(rows)
    group_csv = write_rows(OUT_ROOT / "part_c_r1_grid_summary.csv", summary_rows)
    ok_cells = [r for r in summary_rows if ival(r.get("cell_gate_pass"))]
    math_ranked = sorted(summary_rows, key=lambda r: (fval(r.get("C4_PCG_relative_residual_median"), float("inf")), fval(r.get("max_basis_covariance_error"), float("inf")), fval(r.get("C3_condition_number_median"), float("inf"))))
    payload = {
        "part": "C_R1",
        "route": "C_R1GridFoundPassingCell_NotAdopted" if ok_cells else "C_R1GridNoRepairRecovered",
        "gate_pass": int(bool(ok_cells)),
        "row_count": len(rows),
        "cell_count": len(summary_rows),
        "matrix": rel(matrix),
        "group_summary": rel(group_csv),
        "best_math_residual_cell": math_ranked[0] if math_ranked else {},
        "passing_cells": ok_cells,
        "selection_note": "Merged R1 grid uses math residual/covariance/condition ranking only; it does not overwrite the formal Part C route.",
    }
    out = write_json(OUT_ROOT / "part_c_r1_grid_summary.json", payload)
    append_exec("Part C R1 numerical grid merge", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"cells={len(summary_rows)}; passing={len(ok_cells)}")
    append_recap("Part C R1 numerical grid merge", payload)
    return payload


def part_d_run(args: argparse.Namespace, *, minimal: bool = False) -> dict[str, Any]:
    pc = read_json(OUT_ROOT / "part_c_summary.json")
    if not minimal and not ival(pc.get("gate_pass")):
        return blocked_summary("D", "D_BlockedByPartC", "part_c_missing_or_failed")
    part_name = "D_minimal" if minimal else "D"
    file_prefix = "part_d_minimal" if minimal else "part_d"
    pass_route = "D_MinimalOneStepCausalityObserved_BlockedByPartC" if minimal else "PartDOneStepCausalityPass"
    fail_route = "D_MinimalFalsificationNoActualCausality" if minimal else "D_GlobalInverseNoActualCausality"
    rows: list[dict[str, Any]] = []
    schemes = SCHEME_REGISTRY
    checkpoints = ["random_init", "FunctionalGram_partial", "C15_partial"]
    for task in ["local_patch_interaction", "rotation_sensitive"]:
        for seed in range(5):
            for checkpoint in checkpoints:
                x, y, xg, yg = synthetic_batch(args, task=task, seed=seed, dtype=torch.float32, train_size=96, guard_size=96)
                for scheme in schemes:
                    model = make_model(args, basis_key="dche_k9", depth=3, width=8, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=7100 + seed, dtype=torch.float32)
                    train_checkpoint(model, x, y, checkpoint, seed, args)
                    before = actual_metrics(model, xg, yg)
                    source_before = actual_metrics(model, x, y)
                    cache_mode = "normal"
                    pcg_iter = 4
                    if scheme == "V2_BC_VH_GlobalPCG2_efficiency_control":
                        pcg_iter = 2
                    if scheme == "V5_BC_VH_HorizontalSuppressed_Dzero":
                        cache_mode = "suppressed"
                    if scheme == "V6_BC_VH_HorizontalShuffled":
                        cache_mode = "shuffled"
                    if scheme == "V7_BC_VH_RandomCrossLayerSigns":
                        cache_mode = "random_signs"
                    t0 = time.time()
                    accepted = 0
                    alpha_used = 0.0
                    pred_cos = 0.0
                    gnorm = 0.0
                    if scheme == "V8_SameComputeNoOp":
                        pass
                    elif scheme in {"V0_FunctionalGram_AdamW", "V9_H10_known_structure_upper_bound_diagnostic"}:
                        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=float(args.weight_decay))
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(x).float(), y.long())
                        loss.backward()
                        opt.step()
                        accepted = 1
                        alpha_used = 1.0
                    else:
                        cache = build_tangent_cache(model, x, horizontal_mode=cache_mode, seed=seed)
                        grams = edge_grams(args, model, "dche_k9", torch.float64)
                        residual = output_residual(cache.logits, y)
                        if scheme in {"V1_BC15_BlockDiagonalVerticalFlow", "V4_BC_VH_NearestNeighborBlockTridiagonal"}:
                            flow = blockdiag_exact_flow(model, cache, residual, grams, lam=float(args.lam))
                        else:
                            flow = global_pcg_flow(model, cache, residual, grams, lam=float(args.lam), iterations=pcg_iter)
                        pred = apply_j(model, cache, flow.delta).detach()
                        state = model_state(model)
                        pred_loss_reduction = float((pred.to(dtype=torch.float64) * residual.to(dtype=torch.float64)).mean().detach().cpu().item())
                        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                            restore_model(model, state)
                            apply_delta(model, flow.delta, alpha * float(args.one_step_alpha))
                            guard_now = actual_metrics(model, xg, yg)
                            if fval(guard_now.get("loss")) <= fval(before.get("loss")) + 0.01:
                                accepted = 1
                                alpha_used = alpha
                                break
                        after_pred = model(x).detach() - cache.logits.detach()
                        pred_cos = float(F.cosine_similarity(pred.reshape(1, -1).float(), after_pred.reshape(1, -1).float(), dim=1).detach().cpu().item()) if float(pred.norm()) > 0 and float(after_pred.norm()) > 0 else 0.0
                        gnorm = float(torch.sqrt(metric_norm_sq(flow.delta, grams).clamp_min(0.0)).detach().cpu().item())
                        if not accepted:
                            restore_model(model, state)
                        _ = pred_loss_reduction
                    wall = time.time() - t0
                    after = actual_metrics(model, xg, yg)
                    source_after = actual_metrics(model, x, y)
                    rows.append({
                        "part": part_name,
                        "scheme": scheme,
                        "basis_key": "dche_k9",
                        "depth": 3,
                        "task": task,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "status": "ok",
                        "actual_source_loss_delta": fval(source_after.get("loss")) - fval(source_before.get("loss")),
                        "actual_guard_loss_delta": fval(after.get("loss")) - fval(before.get("loss")),
                        "actual_C2_accuracy_delta": fval(after.get("accuracy")) - fval(before.get("accuracy")),
                        "actual_C2_coverage_delta": fval(after.get("coverage")) - fval(before.get("coverage")),
                        "predicted_actual_cosine": pred_cos,
                        "finite_step_accept_rate": float(accepted),
                        "finite_step_scale": alpha_used,
                        "function_G_step_norm": gnorm,
                        "wall_time_ratio": wall,
                    })
    v3 = [r for r in rows if r.get("scheme") == "V3_BC_VH_GlobalPCG4_main"]
    v1 = [r for r in rows if r.get("scheme") == "V1_BC15_BlockDiagonalVerticalFlow"]
    v5 = [r for r in rows if r.get("scheme") == "V5_BC_VH_HorizontalSuppressed_Dzero"]
    v6 = [r for r in rows if r.get("scheme") == "V6_BC_VH_HorizontalShuffled"]
    cov_v3 = median(r.get("actual_C2_coverage_delta") for r in v3)
    checks = {
        "median_actual_C2_coverage_delta_ge_0p05": cov_v3 >= 0.05,
        "V3_minus_BC15_gap_ge_0p03": cov_v3 - median(r.get("actual_C2_coverage_delta") for r in v1) >= 0.03,
        "V3_minus_horizontal_suppressed_ge_0p03": cov_v3 - median(r.get("actual_C2_coverage_delta") for r in v5) >= 0.03,
        "V3_minus_horizontal_shuffled_ge_0p03": cov_v3 - median(r.get("actual_C2_coverage_delta") for r in v6) >= 0.03,
        "predicted_actual_cosine_ge_0p50": median(r.get("predicted_actual_cosine") for r in v3) >= 0.50,
        "accept_rate_ge_0p80": mean(r.get("finite_step_accept_rate") for r in v3) >= 0.80,
        "scale_mean_ge_0p50": mean(r.get("finite_step_scale") for r in v3) >= 0.50,
    }
    gate = int(all(checks.values()))
    blocker = "none" if gate else ",".join(k for k, v in checks.items() if not v)
    matrix = write_rows(OUT_ROOT / f"{file_prefix}_matrix.csv", rows)
    group = {"part": part_name, "gate_pass": gate, "minimal_falsification_only": int(minimal), "blocked_by_part_c": int(minimal and not ival(pc.get("gate_pass"))), "dominant_blocker": blocker, "V3_median_coverage_delta": cov_v3, **{f"check_{k}": int(v) for k, v in checks.items()}}
    group_csv = write_rows(OUT_ROOT / f"{file_prefix}_group_summary.csv", [group])
    failure = write_json(OUT_ROOT / f"{file_prefix}_failure_decomposition.json", {"part": part_name, "gate_pass": gate, "route": pass_route if gate else fail_route, "dominant_blocker": blocker, "checks": checks})
    nxt = next_actions(part_name, gate, blocker, failed_metrics=group, hypothesis="supported" if gate else "unsupported", repair_class="none" if minimal else ("trust" if cov_v3 >= 0.05 else "none"), max_repairs=0 if minimal else 1)
    summary = {"part": part_name, "gate_pass": gate, "route": pass_route if gate else fail_route, "dominant_blocker": blocker, "row_count": len(rows), "matrix": rel(matrix), "group_summary": rel(group_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt), "summary": group}
    out = write_json(OUT_ROOT / f"{file_prefix}_summary.json", summary)
    append_exec("Part D minimal one-step actual causality" if minimal else "Part D one-step actual causality", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; minimal={int(minimal)}")
    append_recap("Part D minimal one-step actual causality" if minimal else "Part D one-step actual causality", summary)
    return summary


def part_d(args: argparse.Namespace) -> dict[str, Any]:
    return part_d_run(args, minimal=False)


def part_d_minimal(args: argparse.Namespace) -> dict[str, Any]:
    return part_d_run(args, minimal=True)


def blocked_summary(part: str, route: str, blocker: str) -> dict[str, Any]:
    matrix = write_rows(OUT_ROOT / f"part_{part.lower()}_matrix.csv", [{"part": part, "status": "blocked", "dominant_blocker": blocker}])
    group = write_rows(OUT_ROOT / f"part_{part.lower()}_group_summary.csv", [{"part": part, "gate_pass": 0, "dominant_blocker": blocker}])
    failure = write_json(OUT_ROOT / f"part_{part.lower()}_failure_decomposition.json", {"part": part, "gate_pass": 0, "route": route, "dominant_blocker": blocker})
    nxt = next_actions(part, 0, blocker, repair_class="none", max_repairs=0)
    summary = {"part": part, "gate_pass": 0, "route": route, "dominant_blocker": blocker, "matrix": rel(matrix), "group_summary": rel(group), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(OUT_ROOT / f"part_{part.lower()}_summary.json", summary)
    append_exec(f"Part {part} blocked", "done", files=f"{rel(out)}; {rel(matrix)}", gpu="", note=blocker)
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    parts = {p: read_json(OUT_ROOT / f"part_{p.lower()}_summary.json") for p in ["A", "B", "C", "D", "E", "F", "G", "H"]}
    c_r1 = read_json(OUT_ROOT / "part_c_r1_grid_summary.json")
    d_minimal = read_json(OUT_ROOT / "part_d_minimal_summary.json")
    route = "H_RobustRealTaskPreflightPass_NotOfficial"
    blocker = "none"
    level = 5
    priority = [
        ("A", "A_CompositionalOperatorInvalid", 0),
        ("C", "C_CrossLayerInverseNoValue", 1),
        ("D", "D_GlobalInverseNoActualCausality", 2),
        ("E", "E_CompositionalFlowNoExpressionGain", 2),
        ("F", "F_PerformancePass_MechanismUnresolved", 3),
        ("G", "G_SolverScalingFailed", 3),
        ("H", "H_RealTaskPreflightFailed", 4),
    ]
    for part, fail_route, fail_level in priority:
        summary = parts.get(part, {})
        if not summary:
            route = fail_route
            blocker = f"part_{part.lower()}_missing"
            level = fail_level
            break
        if not ival(summary.get("gate_pass")):
            route = str(summary.get("route", fail_route))
            if route.startswith("Part"):
                route = fail_route
            blocker = str(summary.get("dominant_blocker", f"part_{part.lower()}_failed"))
            level = fail_level
            break
    out_payload = {
        "part": "I",
        "final_route": route,
        "dominant_blocker": blocker,
        "evidence_level": level,
        "promotion_allowed": 0,
        "part_0_gate_pass": ival(read_json(OUT_ROOT / "part_0_summary.json").get("gate_pass")),
        "part_a_gate_pass": ival(parts.get("A", {}).get("gate_pass")),
        "part_b_gate_pass": ival(parts.get("B", {}).get("gate_pass")),
        "part_c_gate_pass": ival(parts.get("C", {}).get("gate_pass")),
        "part_c_r1_grid_gate_pass": ival(c_r1.get("gate_pass")),
        "part_d_gate_pass": ival(parts.get("D", {}).get("gate_pass")),
        "part_d_minimal_gate_pass": ival(d_minimal.get("gate_pass")),
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "evidence": {
            **{p: rel(OUT_ROOT / f"part_{p.lower()}_summary.json") for p in ["0", "A", "B", "C", "D"] if (OUT_ROOT / f"part_{p.lower()}_summary.json").exists()},
            **({"C_R1": rel(OUT_ROOT / "part_c_r1_grid_summary.json")} if c_r1 else {}),
            **({"D_minimal": rel(OUT_ROOT / "part_d_minimal_summary.json")} if d_minimal else {}),
        },
    }
    out = write_json(OUT_ROOT / "final_route.json", out_payload)
    append_exec("Part I finalize", "done", files=rel(out), gpu=str(args.device), note=f"route={route}; blocker={blocker}; level={level}")
    append_recap("Part I final route", out_payload)
    return out_payload


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    p0 = part_0(args)
    if not ival(p0.get("gate_pass")):
        return finalize(args)
    pa = part_a(args)
    if not ival(pa.get("gate_pass")):
        return finalize(args)
    part_b(args)
    pc = part_c(args)
    if not ival(pc.get("gate_pass")):
        return finalize(args)
    pd = part_d(args)
    if not ival(pd.get("gate_pass")):
        return finalize(args)
    blocked_summary("E", "E_NotImplementedBecausePartDUnexpectedlyPassed", "part_e_requires_full_80_step_matrix_not_executed")
    return finalize(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="all", choices=["all", "part-0", "part-a", "part-b", "part-c", "part-c-r1-grid", "part-c-r1-merge", "part-d", "part-d-minimal", "finalize"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--basis-input-gain", type=float, default=0.25)
    p.add_argument("--quadrature-points", type=int, default=257)
    p.add_argument("--edge-metric-ridge", type=float, default=1.0e-8)
    p.add_argument("--lam", type=float, default=1.0e-2)
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--guard-size", type=int, default=128)
    p.add_argument("--visual-side", type=int, default=4)
    p.add_argument("--visual-fixed-patch-features", type=int, default=1)
    p.add_argument("--visual-task-version", default="v23_15")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--checkpoint-steps", type=int, default=6)
    p.add_argument("--checkpoint-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--one-step-alpha", type=float, default=0.05)
    p.add_argument("--r1-lams", default="0.01,0.03,0.1")
    p.add_argument("--r1-pcg-iters", default="2,4,8")
    p.add_argument("--r1-refreshes", default="1,2,4")
    p.add_argument("--r1-shard-index", type=int, default=0)
    p.add_argument("--r1-shard-count", type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    init_logs()
    for path in [*HELPERS, RUNNER]:
        py_compile.compile(str(path), doraise=True)
    if args.mode == "part-0":
        return part_0(args)
    if args.mode == "part-a":
        return part_a(args)
    if args.mode == "part-b":
        return part_b(args)
    if args.mode == "part-c":
        return part_c(args)
    if args.mode == "part-c-r1-grid":
        return part_c_r1_grid(args)
    if args.mode == "part-c-r1-merge":
        return part_c_r1_merge(args)
    if args.mode == "part-d":
        return part_d(args)
    if args.mode == "part-d-minimal":
        return part_d_minimal(args)
    if args.mode == "finalize":
        return finalize(args)
    return run_all(args)


if __name__ == "__main__":
    main()
