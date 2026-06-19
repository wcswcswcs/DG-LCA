#!/usr/bin/env python3
"""Build v22.18 model-identity and efficiency-path audit artifacts.

This script is intentionally a readback/audit layer.  It classifies existing
v22.17/v22.18 artifacts and marks ambiguous or diagnostic paths as such instead
of upgrading them to official PureKAN evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_18_common import (  # noqa: E402
    OUT_ROOT,
    V22_17_ROOT,
    append_exec,
    ensure_out,
    finite_float,
    now_sg,
    read_rows,
    write_rows,
)


MODEL_FIELDS = [
    "row_id",
    "dataset",
    "seed",
    "model_name",
    "model_origin",
    "model_class_import_path",
    "model_config_sha256",
    "uses_kanbefair_baseline_model",
    "uses_pykan",
    "uses_bspline_official_path",
    "uses_bspline_diagnostic_only",
    "uses_primitivekan",
    "is_dgkan_strict_fc_purekan",
    "carrier",
    "basis_family",
    "basis_name",
    "basis_k",
    "readout_diagnostic_only",
    "basis_native_claim_allowed",
    "same_param_matched",
    "same_flops_matched",
    "param_count",
    "FLOPs",
    "model_identity_class",
    "model_identity_pass",
    "route_scope",
    "never_count_for_DGKAN_efficiency_or_superiority",
    "source_table",
    "source_row_index",
    "identity_blocker",
]


EFF_FIELDS = [
    "path_id",
    "path_scope",
    "model_name",
    "carrier",
    "inside_kanbefair_bridge",
    "inside_dglca_native_loop",
    "official_fused_kernel_complete",
    "manual_upstream_vjp_used",
    "uses_dense_output_jacobian",
    "uses_finite_diff_fallback",
    "uses_python_loop_controller",
    "uses_analytic_or_sketch_jvp",
    "native_jvp_vjp_gradcheck_rel_error",
    "forward_ms",
    "backward_ms",
    "cotangent_ms",
    "basis_JVP_ms",
    "basis_VJP_ms",
    "controller_ms",
    "commit_ms",
    "optimizer_ms",
    "data_loader_ms",
    "logging_ms",
    "full_step_ms",
    "controller_overhead_ratio",
    "memory_peak_mb",
    "dataset",
    "seed",
    "input_dim",
    "output_dim",
    "hidden",
    "train_size",
    "test_size",
    "batch_size",
    "steps",
    "initial_data_batch_fingerprint",
    "audit_data_batch_fingerprint",
    "source_table",
    "source_row_index",
    "implementation_mode",
    "manual_kernel_variant",
    "selector",
    "path_number",
    "path_complete_full_loop",
    "timing_mode",
    "model_identity_pass",
    "official_efficiency_hard_conditions_pass",
    "efficiency_path_official_pass",
    "bridge_overhead_ms",
    "bridge_overhead_ratio",
    "full_loop_ratio_vs_matched_flops_mlp",
    "efficiency_blocker",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v22_18_identity_audit")
    return p


def _stable_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _nonempty(value: Any) -> bool:
    return str(value or "").strip() != ""


def _glob_rows(path: Path, source_table: str) -> list[dict[str, str]]:
    rows = read_rows(path)
    for idx, row in enumerate(rows):
        row["_source_table"] = source_table
        row["_source_row_index"] = str(idx)
    return rows


def _model_paths() -> list[tuple[Path, str]]:
    paths = [
        (V22_17_ROOT / "v22_17_kanbefair_baseline_reproduction.csv", "v22_17_kanbefair_baseline_reproduction"),
        (V22_17_ROOT / "v22_17_kanbefair_model_complexity_smoke.csv", "v22_17_kanbefair_model_complexity_smoke"),
    ]
    for pattern in [
        "v22_18_mlp_fu_task_matrix*.csv",
        "v22_18_kan_native_ladder_matrix*.csv",
        "v22_18_kanbefair_expanded_task_matrix*.csv",
    ]:
        for path in sorted(OUT_ROOT.glob(pattern)):
            paths.append((path, path.stem))
    seen: set[Path] = set()
    out: list[tuple[Path, str]] = []
    for path, source in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append((path, source))
    return out


def _carrier(model_name: str) -> str:
    if model_name.startswith("DGKAN_DCHE"):
        return "D-CHE"
    if model_name.startswith("DGKAN_DFOU"):
        return "D-FOU"
    if model_name == "KAN":
        return "KANbeFair-BSpline"
    if "BSpline" in model_name:
        return "KANbeFair-BSpline"
    return ""


def _basis_fields(carrier: str) -> tuple[str, str, str]:
    if carrier == "D-CHE":
        return "D-CHE", "chebyshev_lowdegree", "3"
    if carrier == "D-FOU":
        return "D-FOU", "fourier_lowfreq", "3"
    if carrier == "KANbeFair-BSpline":
        return "bspline_external", "pykan_or_bspline", "5"
    return "", "", ""


def _class_path(model_name: str, source_table: str) -> str:
    if source_table.startswith("v22_17_kanbefair"):
        if model_name == "MLP":
            return "external.KANbeFair_worktree.src.models.mlp.MLP"
        if model_name == "KAN":
            return "external.KANbeFair_worktree.src.models.kanbefair.KANbeFair"
        if model_name == "BSpline_MLP":
            return "external.KANbeFair_worktree.src.models.bspline_mlp.BSpline_MLP"
        if model_name == "BSpline_First_MLP":
            return "external.KANbeFair_worktree.src.models.bspline_mlp.BSpline_First_MLP"
    if model_name.startswith("DGKAN_"):
        return "dgkan.models.fc_purekan_primitives.PrimitiveKAN"
    if model_name.startswith("DGMLP") or model_name == "MLP_ADAMW":
        return "dgkan.models.fc_purekan_primitives.MLPBaseline"
    return ""


def _model_family_name(row: dict[str, str]) -> str:
    return str(row.get("model_name") or row.get("model") or "").strip()


def _build_ratio_maps(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    refs: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        source = str(row.get("_source_table", ""))
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        model = _model_family_name(row)
        if model in {"DGMLP", "MLP_ADAMW", "MLP"}:
            key = (source, dataset, seed)
            refs[key] = {
                "param_count": finite_float(row.get("param_count"), 0.0),
                "FLOPs": finite_float(row.get("FLOPs") or row.get("flops"), 0.0),
            }
    return refs


def _matched_flags(row: dict[str, str], refs: dict[tuple[str, str, str], dict[str, float]]) -> tuple[str, str]:
    param_ratio = row.get("param_ratio_vs_mlp")
    flops_ratio = row.get("flops_ratio_vs_mlp")
    if _nonempty(param_ratio):
        p_ratio = finite_float(param_ratio, 999.0)
        same_param = int(0.80 <= p_ratio <= 1.25)
    else:
        ref = refs.get((str(row.get("_source_table", "")), str(row.get("dataset", "")), str(row.get("seed", ""))), {})
        p = finite_float(row.get("param_count"), 0.0)
        rp = ref.get("param_count", 0.0)
        same_param = int(p > 0 and rp > 0 and 0.80 <= p / rp <= 1.25)
    if _nonempty(flops_ratio):
        f_ratio = finite_float(flops_ratio, 999.0)
        same_flops = int(0.80 <= f_ratio <= 1.25)
    else:
        ref = refs.get((str(row.get("_source_table", "")), str(row.get("dataset", "")), str(row.get("seed", ""))), {})
        f = finite_float(row.get("FLOPs") or row.get("flops"), 0.0)
        rf = ref.get("FLOPs", 0.0)
        same_flops = int(f > 0 and rf > 0 and 0.80 <= f / rf <= 1.25)
    return str(same_param), str(same_flops)


def _identity_from_row(row: dict[str, str], refs: dict[tuple[str, str, str], dict[str, float]]) -> dict[str, Any]:
    source_table = str(row.get("_source_table", ""))
    source_idx = str(row.get("_source_row_index", ""))
    model = _model_family_name(row)
    carrier = _carrier(model)
    basis_family, basis_name, basis_k = _basis_fields(carrier)
    source_is_kanbefair_baseline = source_table.startswith("v22_17_kanbefair")
    is_dgkan = model.startswith("DGKAN_")
    is_mlp = model.startswith("DGMLP") or model == "MLP_ADAMW"
    is_fu = "_FU" in model or "FU" in str(row.get("v22_18_variant", ""))
    readout_diag = _int(row.get("uses_readout_diagnostic"))

    model_origin = "KANbeFair" if source_is_kanbefair_baseline else "DG-LCA"
    uses_kanbefair_baseline_model = int(source_is_kanbefair_baseline)
    uses_pykan = int(source_is_kanbefair_baseline and model == "KAN")
    uses_bspline_official_path = int(source_is_kanbefair_baseline and model in {"KAN", "BSpline_MLP", "BSpline_First_MLP"})
    uses_bspline_diagnostic_only = int(source_is_kanbefair_baseline and "BSpline" in model)
    uses_primitivekan = int(is_dgkan and not source_is_kanbefair_baseline)
    is_strict = int(is_dgkan and not source_is_kanbefair_baseline and _int(row.get("is_dgkan_strict_fc_purekan", 1)) == 1)
    basis_native_allowed = int(is_dgkan and not readout_diag and carrier in {"D-CHE", "D-FOU"})
    same_param, same_flops = _matched_flags(row, refs)

    identity_class = "RejectedAmbiguousModelIdentity"
    route_scope = "RejectedOrDiagnostic"
    identity_pass = 0
    blocker = ""
    never_count = 0
    if source_is_kanbefair_baseline:
        identity_class = "KANbeFairBaselineContext"
        route_scope = "ExternalBaselineContextOnly"
        never_count = 1
        identity_pass = 1
    elif is_mlp:
        identity_class = "DGMLPFUOfficialCandidate" if is_fu else "DGMLPBaseline"
        route_scope = "DGMLPOfficialCandidate"
        identity_pass = 1
    elif is_dgkan and readout_diag:
        identity_class = "DGKANReadoutDiagnostic"
        route_scope = "DGKANDiagnosticOnly"
        blocker = "ReadoutDiagnosticOnly"
    elif is_dgkan:
        strict_ok = (
            model_origin == "DG-LCA"
            and uses_kanbefair_baseline_model == 0
            and uses_pykan == 0
            and uses_bspline_official_path == 0
            and uses_primitivekan == 1
            and is_strict == 1
            and carrier in {"D-CHE", "D-FOU"}
            and basis_family in {"chebyshev_lowdegree", "fourier_lowfreq", "D-CHE", "D-FOU"}
            and readout_diag == 0
            and basis_native_allowed == 1
        )
        if strict_ok:
            identity_class = "DGKANPureKANOfficialCandidate"
            route_scope = "DGKANOfficialCandidate"
            identity_pass = 1
        else:
            blocker = "ModelIdentityAmbiguous"
    else:
        blocker = "UnknownModelFamily"

    config_hash = _stable_hash(
        {
            "source_table": source_table,
            "model_name": model,
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "hidden": row.get("hidden") or row.get("kan_hidden") or row.get("layers_width", ""),
            "carrier": carrier,
            "basis_family": basis_family,
            "param_count": row.get("param_count", ""),
            "FLOPs": row.get("FLOPs") or row.get("flops", ""),
            "implementation_mode": row.get("implementation_mode", ""),
        }
    )
    out = {
        "row_id": f"{source_table}:{source_idx}",
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "model_name": model,
        "model_origin": model_origin,
        "model_class_import_path": _class_path(model, source_table),
        "model_config_sha256": config_hash,
        "uses_kanbefair_baseline_model": uses_kanbefair_baseline_model,
        "uses_pykan": uses_pykan,
        "uses_bspline_official_path": uses_bspline_official_path,
        "uses_bspline_diagnostic_only": uses_bspline_diagnostic_only,
        "uses_primitivekan": uses_primitivekan,
        "is_dgkan_strict_fc_purekan": is_strict,
        "carrier": carrier,
        "basis_family": basis_family,
        "basis_name": basis_name,
        "basis_k": basis_k,
        "readout_diagnostic_only": readout_diag,
        "basis_native_claim_allowed": basis_native_allowed,
        "same_param_matched": same_param,
        "same_flops_matched": same_flops,
        "param_count": row.get("param_count", ""),
        "FLOPs": row.get("FLOPs") or row.get("flops", ""),
        "model_identity_class": identity_class,
        "model_identity_pass": identity_pass,
        "route_scope": route_scope,
        "never_count_for_DGKAN_efficiency_or_superiority": never_count,
        "source_table": source_table,
        "source_row_index": source_idx,
        "identity_blocker": blocker,
    }
    return {field: out.get(field, "") for field in MODEL_FIELDS}


def _eff_paths() -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    for path in sorted(V22_17_ROOT.glob("v22_17_basis_jvp_vjp_gradcheck*.csv")):
        if path.name.endswith("_failures.csv"):
            continue
        paths.append((path, path.stem))
    for pattern in [
        "v22_18_kan_native_ladder_matrix*.csv",
        "v22_18_kanbefair_expanded_task_matrix*.csv",
    ]:
        for path in sorted(OUT_ROOT.glob(pattern)):
            paths.append((path, path.stem))
    return paths


def _model_identity_lookup(model_rows: list[dict[str, Any]]) -> dict[str, int]:
    by_model: dict[str, int] = {}
    for row in model_rows:
        model = str(row.get("model_name", ""))
        if model.startswith("DGKAN_"):
            identity_pass = _int(row.get("model_identity_pass"))
            by_model[model] = max(by_model.get(model, 0), identity_pass)
            if model.startswith("DGKAN_DCHE"):
                by_model["DGKAN_DCHE"] = max(by_model.get("DGKAN_DCHE", 0), identity_pass)
            if model.startswith("DGKAN_DFOU"):
                by_model["DGKAN_DFOU"] = max(by_model.get("DGKAN_DFOU", 0), identity_pass)
    return by_model


def _eff_path_class(row: dict[str, str]) -> tuple[str, str, str, int, int]:
    source = str(row.get("_source_table", ""))
    model = _model_family_name(row)
    impl = str(row.get("implementation_mode", ""))
    selector = str(row.get("selector", ""))
    if source.startswith("v22_17_basis_jvp_vjp_gradcheck") and impl == "native_no_dense_basis":
        return "Path1InternalOptimizedPureKANProfiler", "P1", "internal_profiler", 1, 0
    if source.startswith("v22_17_basis_jvp_vjp_gradcheck") and impl == "bridge_dense_cache":
        return "Path3BridgeProfilerDiagnostic", "P3-profiler", "bridge_profiler", 3, 0
    if source.startswith("v22_18_kan_native_ladder_matrix") and impl == "native_no_dense_basis":
        return "Path2NativeDGLCAFullTrainLoop", "P2", "native_full_loop", 2, 1
    if source.startswith("v22_18_kanbefair_expanded_task_matrix") and model.startswith("DGKAN_") and "_FU" in model:
        return "Path4KANbeFairBridgeWithFU", "P4", "bridge_fu_full_loop", 4, 1
    if source.startswith("v22_18_kanbefair_expanded_task_matrix") and model.startswith("DGKAN_"):
        return "Path3KANbeFairBridgeNoFU", "P3", "bridge_no_fu_full_loop", 3, 1
    return "RejectedNonDGKANEfficiencyRow", "PX", selector or "unknown", 0, 0


def _eff_from_row(row: dict[str, str], identity_by_model: dict[str, int]) -> dict[str, Any] | None:
    model = _model_family_name(row)
    if not model.startswith("DGKAN_"):
        return None
    source = str(row.get("_source_table", ""))
    source_idx = str(row.get("_source_row_index", ""))
    path_scope, path_prefix, timing_mode, path_number, full_loop = _eff_path_class(row)
    if path_number == 0:
        return None
    carrier = _carrier(model)
    impl = str(row.get("implementation_mode", ""))
    selector = str(row.get("selector", ""))
    is_bridge = int(path_number in {3, 4})
    is_native_loop = int(path_number == 2)
    is_path1 = int(path_number == 1)
    no_dense_impl = impl in {"native_no_dense_basis", "kanbefair_bridge_no_dense_basis"}
    relerr = row.get("native_jvp_vjp_gradcheck_rel_error") or row.get("analytic_vs_autograd_rel_error_audit") or row.get("func_jvp_relerr_max") or ""
    relerr_value = finite_float(relerr, 999.0)
    uses_dense = _int(row.get("uses_dense_output_jacobian_official"))
    dense_basis = _int(row.get("uses_dense_basis_tensor_spec"))
    manual_kernel_variant = str(row.get("manual_kernel_variant", ""))
    optimized_kernel_marker = ("triton" in manual_kernel_variant) or ("matmul" in manual_kernel_variant)
    official_fused = int(
        (
            is_path1
            and no_dense_impl
            and selector == "basis"
            and dense_basis == 0
            and optimized_kernel_marker
        )
        or (
            is_native_loop
            and no_dense_impl
            and dense_basis == 0
            and optimized_kernel_marker
        )
        or (
            is_bridge
            and no_dense_impl
            and dense_basis == 0
            and optimized_kernel_marker
        )
    )
    finite_diff_fallback = int(source.startswith("v22_18_kan_native_ladder_matrix") and _int(row.get("audit_dense_finite_diff_j_used")) == 1)
    if source.startswith("v22_17_basis_jvp_vjp_gradcheck") and _int(row.get("func_jvp_available")) == 1:
        finite_diff_fallback = 0
    analytic_or_sketch = int(
        (_int(row.get("official_basis_jvp_vjp_gradcheck_pass")) == 1)
        or (_int(row.get("func_jvp_available")) == 1 and relerr_value <= 1.0e-4)
        or str(row.get("final_jvp_reference", "")) == "torch_func_jvp"
    )
    manual_upstream_vjp = 0
    controller_ratio = row.get("controller_overhead_ratio", "")
    full_step = row.get("full_loop_step_ms") or row.get("runtime_per_step_ms") or ""
    controller_ms = ""
    if _nonempty(full_step) and _nonempty(controller_ratio):
        controller_ms = f"{finite_float(full_step) * finite_float(controller_ratio):.9g}"
    hard_pass = int(
        official_fused == 1
        and manual_upstream_vjp == 0
        and uses_dense == 0
        and finite_diff_fallback == 0
        and analytic_or_sketch == 1
        and relerr_value <= 1.0e-4
    )
    path_official = int(hard_pass == 1 and full_loop == 1)
    blocker_parts: list[str] = []
    if identity_by_model.get(model, 0) != 1:
        blocker_parts.append("ModelIdentityAmbiguous")
    if official_fused != 1:
        blocker_parts.append("OfficialFusedFullLoopMissing" if full_loop else "ProfilerNotOfficialFullLoop")
    if finite_diff_fallback:
        blocker_parts.append("FiniteDiffAuditPathSeparated")
    if analytic_or_sketch != 1:
        blocker_parts.append("AnalyticOrSketchJVPMissing")
    if path_official != 1:
        blocker_parts.append("EfficiencyPathDiagnosticOnly")
    out = {
        "path_id": f"{path_prefix}:{source}:{source_idx}",
        "path_scope": path_scope,
        "model_name": model,
        "carrier": carrier,
        "inside_kanbefair_bridge": is_bridge,
        "inside_dglca_native_loop": is_native_loop,
        "official_fused_kernel_complete": official_fused,
        "manual_upstream_vjp_used": manual_upstream_vjp,
        "uses_dense_output_jacobian": uses_dense,
        "uses_finite_diff_fallback": finite_diff_fallback,
        "uses_python_loop_controller": int("_FU" in model),
        "uses_analytic_or_sketch_jvp": analytic_or_sketch,
        "native_jvp_vjp_gradcheck_rel_error": relerr,
        "forward_ms": row.get("forward_ms", ""),
        "backward_ms": row.get("backward_ms", ""),
        "cotangent_ms": row.get("cotangent_ms", ""),
        "basis_JVP_ms": row.get("basis_jvp_ms", ""),
        "basis_VJP_ms": row.get("basis_vjp_ms", ""),
        "controller_ms": controller_ms,
        "commit_ms": row.get("commit_ms", ""),
        "optimizer_ms": row.get("optimizer_ms", ""),
        "data_loader_ms": row.get("data_loader_ms", ""),
        "logging_ms": row.get("logging_ms", ""),
        "full_step_ms": full_step,
        "controller_overhead_ratio": controller_ratio,
        "memory_peak_mb": row.get("memory_peak_mb", ""),
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "input_dim": row.get("input_dim", ""),
        "output_dim": row.get("output_dim", ""),
        "hidden": row.get("hidden") or row.get("kan_hidden", ""),
        "train_size": row.get("train_size", ""),
        "test_size": row.get("test_size", ""),
        "batch_size": row.get("batch_size", ""),
        "steps": row.get("steps") or row.get("horizon_steps", ""),
        "initial_data_batch_fingerprint": row.get("initial_data_batch_fingerprint", ""),
        "audit_data_batch_fingerprint": row.get("audit_data_batch_fingerprint", ""),
        "source_table": source,
        "source_row_index": source_idx,
        "implementation_mode": impl,
        "manual_kernel_variant": manual_kernel_variant,
        "selector": selector,
        "path_number": path_number,
        "path_complete_full_loop": full_loop,
        "timing_mode": timing_mode,
        "model_identity_pass": identity_by_model.get(model, 0),
        "official_efficiency_hard_conditions_pass": hard_pass,
        "efficiency_path_official_pass": path_official,
        "bridge_overhead_ms": "",
        "bridge_overhead_ratio": "",
        "full_loop_ratio_vs_matched_flops_mlp": row.get("full_loop_ratio_vs_mlp", ""),
        "efficiency_blocker": ";".join(dict.fromkeys(blocker_parts)),
    }
    return {field: out.get(field, "") for field in EFF_FIELDS}


def _mean(values: list[float]) -> str:
    vals = [v for v in values if v == v and v not in {float("inf"), float("-inf")}]
    if not vals:
        return ""
    return f"{sum(vals) / len(vals):.9g}"


def _bridge_audit_rows(eff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_number in [1, 2, 3, 4]:
        vals = [r for r in eff_rows if _int(r.get("path_number")) == path_number]
        full_vals = [finite_float(r.get("full_step_ms"), float("nan")) for r in vals if _nonempty(r.get("full_step_ms"))]
        overhead_vals = [finite_float(r.get("controller_overhead_ratio"), float("nan")) for r in vals if _nonempty(r.get("controller_overhead_ratio"))]
        relerr_vals = [finite_float(r.get("native_jvp_vjp_gradcheck_rel_error"), float("nan")) for r in vals if _nonempty(r.get("native_jvp_vjp_gradcheck_rel_error"))]
        rows.append(
            {
                "path_number": path_number,
                "path_name": {
                    1: "internal optimized PureKAN profiler",
                    2: "native DG-LCA full train loop",
                    3: "KANbeFair bridge without FU",
                    4: "KANbeFair bridge with FU",
                }[path_number],
                "rows": len(vals),
                "full_loop_rows": sum(_int(r.get("path_complete_full_loop")) for r in vals),
                "official_identity_pass_rows": sum(_int(r.get("model_identity_pass")) for r in vals),
                "official_efficiency_hard_pass_rows": sum(_int(r.get("official_efficiency_hard_conditions_pass")) for r in vals),
                "efficiency_path_official_pass_rows": sum(_int(r.get("efficiency_path_official_pass")) for r in vals),
                "mean_full_step_ms": _mean(full_vals),
                "mean_controller_overhead_ratio": _mean(overhead_vals),
                "max_gradcheck_rel_error": f"{max(relerr_vals):.9g}" if relerr_vals else "",
                "uses_dense_output_jacobian_rows": sum(_int(r.get("uses_dense_output_jacobian")) for r in vals),
                "finite_diff_fallback_rows": sum(_int(r.get("uses_finite_diff_fallback")) for r in vals),
                "diagnostic_only_rows": sum(1 for r in vals if _int(r.get("efficiency_path_official_pass")) == 0),
                "blocker": "" if vals else "PathMissing",
            }
        )
    return rows


def _same_config_key(row: dict[str, Any], *, require_fingerprint: bool) -> tuple[str, ...] | None:
    fields = ["dataset", "seed", "carrier", "hidden", "train_size", "test_size", "batch_size"]
    values = [str(row.get(field, "")).strip() for field in fields]
    if any(not value for value in values):
        return None
    if require_fingerprint:
        fp = str(row.get("initial_data_batch_fingerprint", "")).strip()
        if not fp:
            return None
        values.append(fp)
    return tuple(values)


def _complete_key_count(eff_rows: list[dict[str, Any]], *, require_fingerprint: bool) -> tuple[int, str]:
    by_key: dict[tuple[str, ...], set[int]] = {}
    for row in eff_rows:
        key = _same_config_key(row, require_fingerprint=require_fingerprint)
        if key is None:
            continue
        by_key.setdefault(key, set()).add(_int(row.get("path_number")))
    complete = [key for key, paths in by_key.items() if {1, 2, 3, 4}.issubset(paths)]
    example = "|".join(complete[0]) if complete else ""
    return len(complete), example


def _summary_rows(model_rows: list[dict[str, Any]], eff_rows: list[dict[str, Any]], bridge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    official_dgkan = [
        r
        for r in model_rows
        if str(r.get("model_name", "")).startswith("DGKAN_")
        and str(r.get("model_origin")) == "DG-LCA"
        and _int(r.get("readout_diagnostic_only")) == 0
    ]
    kanbefair_original = [
        r
        for r in model_rows
        if str(r.get("model_origin")) == "KANbeFair" and str(r.get("model_name")) in {"KAN", "BSpline_MLP", "BSpline_First_MLP"}
    ]
    official_eff_rows = [r for r in eff_rows if _int(r.get("official_fused_kernel_complete")) == 1]
    path_counts = {str(r.get("path_number")): _int(r.get("rows")) for r in bridge_rows}
    full_loop_counts = {str(r.get("path_number")): _int(r.get("full_loop_rows")) for r in bridge_rows}
    carriers_by_path: dict[int, set[str]] = {}
    hidden_by_path: dict[int, set[str]] = {}
    for r in eff_rows:
        pn = _int(r.get("path_number"))
        carriers_by_path.setdefault(pn, set()).add(str(r.get("carrier", "")))
        hidden_by_path.setdefault(pn, set()).add(str(r.get("implementation_mode", "")))
    four_paths_present = all(path_counts.get(str(i), 0) > 0 for i in [1, 2, 3, 4])
    four_full_loop_present = all(full_loop_counts.get(str(i), 0) > 0 for i in [2, 3, 4]) and path_counts.get("1", 0) > 0
    same_config_count, same_config_example = _complete_key_count(eff_rows, require_fingerprint=False)
    same_data_count, same_data_example = _complete_key_count(eff_rows, require_fingerprint=True)
    same_data_batch_hidden_seed_complete = int(same_data_count > 0)
    f_four_path_explore = int(four_paths_present and four_full_loop_present and same_data_batch_hidden_seed_complete == 1)
    summary = {
        "timestamp": now_sg(),
        "model_identity_matrix_complete": int(bool(model_rows) and all(all(field in r for field in MODEL_FIELDS) for r in model_rows)),
        "model_identity_rows": len(model_rows),
        "official_DGKAN_rows": len(official_dgkan),
        "official_DGKAN_identity_pass_rows": sum(_int(r.get("model_identity_pass")) for r in official_dgkan),
        "official_DGKAN_rows_have_strict_FC_PureKAN_identity": int(bool(official_dgkan) and all(_int(r.get("model_identity_pass")) == 1 for r in official_dgkan)),
        "KANbeFair_original_KAN_rows": len(kanbefair_original),
        "KANbeFair_original_KAN_rows_marked_baseline_context_only": int(
            bool(kanbefair_original)
            and all(
                _int(r.get("uses_kanbefair_baseline_model")) == 1
                and str(r.get("route_scope")) == "ExternalBaselineContextOnly"
                and _int(r.get("never_count_for_DGKAN_efficiency_or_superiority")) == 1
                for r in kanbefair_original
            )
        ),
        "efficiency_path_identity_matrix_complete": int(bool(eff_rows) and all(all(field in r for field in EFF_FIELDS) for r in eff_rows)),
        "efficiency_path_rows": len(eff_rows),
        "official_efficiency_rows_count": len(official_eff_rows),
        "official_efficiency_rows_use_analytic_or_sketch_JVP": int(
            bool(official_eff_rows) and all(_int(r.get("uses_analytic_or_sketch_jvp")) == 1 for r in official_eff_rows)
        ),
        "official_efficiency_hard_conditions_pass_rows": sum(_int(r.get("official_efficiency_hard_conditions_pass")) for r in eff_rows),
        "efficiency_path_official_pass_rows": sum(_int(r.get("efficiency_path_official_pass")) for r in eff_rows),
        "path1_rows": path_counts.get("1", 0),
        "path2_rows": path_counts.get("2", 0),
        "path3_rows": path_counts.get("3", 0),
        "path4_rows": path_counts.get("4", 0),
        "four_path_audit_paths_present": int(four_paths_present),
        "four_path_audit_full_loop_present": int(four_full_loop_present),
        "same_dataset_seed_hidden_batch_complete_keys": same_config_count,
        "same_dataset_seed_hidden_batch_example_key": same_config_example,
        "same_data_batch_hidden_seed_complete_keys": same_data_count,
        "same_data_batch_hidden_seed_example_key": same_data_example,
        "same_data_batch_hidden_seed_complete": same_data_batch_hidden_seed_complete,
        "F_four_path_exploration_pass": f_four_path_explore,
        "identity_audit_blocker": "" if f_four_path_explore else "FourPathAuditNotSameDataBatchHiddenSeedOrNoOfficialFullLoop",
    }
    return [summary]


def _svg(path: Path, title: str, lines: list[str]) -> None:
    safe = [line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for line in lines]
    height = max(140, 70 + 24 * len(safe))
    body = "\n".join(
        f'<text x="24" y="{70 + 24 * i}" font-size="14" fill="#1f2937">{line}</text>' for i, line in enumerate(safe)
    )
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">\n'
            '<rect width="1200" height="100%" fill="#f8fafc"/>\n'
            f'<text x="24" y="36" font-size="22" font-family="sans-serif" fill="#111827">{title}</text>\n'
            f'<g font-family="monospace">{body}</g>\n'
            "</svg>\n"
        ),
        encoding="utf-8",
    )


def _write_figures(model_rows: list[dict[str, Any]], bridge_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fig = OUT_ROOT / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row in model_rows:
        key = str(row.get("model_identity_class", ""))
        counts[key] = counts.get(key, 0) + 1
    identity_lines = [
        f"{key}: {counts[key]}" for key in sorted(counts)
    ] + [
        f"DGKAN official identity pass rows: {summary.get('official_DGKAN_identity_pass_rows')}/{summary.get('official_DGKAN_rows')}",
        f"KANbeFair original KAN/BSpline baseline-only: {summary.get('KANbeFair_original_KAN_rows_marked_baseline_context_only')}",
    ]
    _svg(fig / "v22_18_model_identity_firewall.svg", "v22.18 model identity firewall", identity_lines)
    bridge_lines = [
        f"Path {r.get('path_number')} {r.get('path_name')}: rows={r.get('rows')} full_loop={r.get('full_loop_rows')} official_path={r.get('efficiency_path_official_pass_rows')} mean_full_step_ms={r.get('mean_full_step_ms')}"
        for r in bridge_rows
    ] + [
        f"same_data_batch_hidden_seed_complete={summary.get('same_data_batch_hidden_seed_complete')}",
        f"blocker={summary.get('identity_audit_blocker')}",
    ]
    _svg(fig / "v22_18_bridge_overhead_waterfall.svg", "v22.18 bridge overhead path audit", bridge_lines)


def main() -> None:
    _args = parser().parse_args()
    ensure_out()
    raw_model_rows: list[dict[str, str]] = []
    for path, source in _model_paths():
        raw_model_rows.extend(_glob_rows(path, source))
    refs = _build_ratio_maps(raw_model_rows)
    model_rows = [_identity_from_row(row, refs) for row in raw_model_rows if _model_family_name(row)]
    write_rows(OUT_ROOT / "v22_18_model_identity_matrix.csv", model_rows)

    identity_by_model = _model_identity_lookup(model_rows)
    eff_rows: list[dict[str, Any]] = []
    for path, source in _eff_paths():
        for row in _glob_rows(path, source):
            out = _eff_from_row(row, identity_by_model)
            if out is not None:
                eff_rows.append(out)
    write_rows(OUT_ROOT / "v22_18_efficiency_path_identity_matrix.csv", eff_rows)
    component_rows = [
        {field: row.get(field, "") for field in EFF_FIELDS}
        for row in eff_rows
        if _nonempty(row.get("full_step_ms")) or _nonempty(row.get("basis_JVP_ms")) or _nonempty(row.get("basis_VJP_ms"))
    ]
    write_rows(OUT_ROOT / "v22_18_efficiency_path_component_timing.csv", component_rows)
    bridge_rows = _bridge_audit_rows(eff_rows)
    write_rows(OUT_ROOT / "v22_18_efficiency_bridge_audit_matrix.csv", bridge_rows)
    summary_rows = _summary_rows(model_rows, eff_rows, bridge_rows)
    write_rows(OUT_ROOT / "v22_18_identity_audit_summary.csv", summary_rows)
    _write_figures(model_rows, bridge_rows, summary_rows[0])

    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="A-identity-efficiency-audit",
        status="pass",
        exit_code=0,
        files=(
            "results/v22_18/v22_18_model_identity_matrix.csv, "
            "results/v22_18/v22_18_efficiency_path_identity_matrix.csv, "
            "results/v22_18/v22_18_efficiency_bridge_audit_matrix.csv, "
            "results/v22_18/v22_18_efficiency_path_component_timing.csv, "
            "results/v22_18/v22_18_identity_audit_summary.csv, "
            "results/v22_18/figures/v22_18_model_identity_firewall.svg, "
            "results/v22_18/figures/v22_18_bridge_overhead_waterfall.svg"
        ),
        note=(
            f"model_rows={len(model_rows)}; efficiency_rows={len(eff_rows)}; "
            f"four_path_pass={summary_rows[0].get('F_four_path_exploration_pass')}"
        ),
    )


if __name__ == "__main__":
    main()
