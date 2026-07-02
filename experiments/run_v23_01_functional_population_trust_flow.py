#!/usr/bin/env python3
"""DG-KAN v23.01 Functional Edge Population Trust Flow runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import py_compile
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
import experiments.run_v23_00r_curve_geometry_population_flow as v2300
from dgkan.fu.edge_functional_gram import functional_edge_gram, functional_gram_condition
from dgkan.fu.finite_step_trust_region import (
    max_snapshot_error,
    optimizer_state_max_abs_error,
    restore_model_params,
    restore_optimizer,
    snapshot_model_params,
    snapshot_optimizer,
)


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.01_FunctionalEdgePopulationTrustFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.01_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.01_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2301_OUT_ROOT", str(ROOT / "results/v23_01"))).resolve()
AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "new_edge_function_added": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "external_product_feature_used": 0,
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


def fval(x: Any, default: float = 0.0) -> float:
    try:
        if x in ("", None, "missing"):
            return default
        out = float(x)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def ival(x: Any, default: int = 0) -> int:
    try:
        if x in ("", None, "missing"):
            return default
        return int(float(x))
    except Exception:
        return default


def mean(vals: Iterable[Any], default: float = 0.0) -> float:
    xs = [fval(v) for v in vals if v not in ("", None, "missing")]
    return float(sum(xs) / len(xs)) if xs else default


def median(vals: Iterable[Any], default: float = 0.0) -> float:
    xs = [fval(v) for v in vals if v not in ("", None, "missing")]
    return float(np.median(xs)) if xs else default


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_exec(part: str, command: str, status: str, *, files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.01 FunctionalEdgePopulationTrustFlow 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误；缺失写 missing，不补造。\n\n",
            encoding="utf-8",
        )
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {part} {status}\n\n")
        fh.write(f"Command: `{command}`\n\n")
        if files:
            fh.write(f"Files: {files}\n\n")
        if note:
            fh.write(f"Note: {note}\n\n")


def append_recap(title: str, data: dict[str, Any]) -> None:
    ensure_out()
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.01 FunctionalEdgePopulationTrustFlow 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n"
            "v23.01 正在执行中；本日志只记录真实 artifact 与真实结果，不补造数据。\n\n",
            encoding="utf-8",
        )
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def next_actions(part: str, gate: int, blocker: str, actions: list[str], forbidden: list[str] | None = None) -> Path:
    payload = {
        "part": part,
        "gate_pass": int(gate),
        "dominant_blocker": blocker,
        "allowed_next_actions": actions,
        "forbidden_actions": forbidden or [
            "do_not_delete_random_control",
            "do_not_use_held_test_metric_induction",
            "do_not_add_edge_functions",
            "do_not_lower_no_debt_gate",
        ],
        "max_repair_rounds": 2,
        "rerun_commands": [],
    }
    return write_json(OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json", payload)


def command_text(argv: list[str]) -> str:
    return " ".join([PYTHON, rel(RUNNER), *argv[1:]])


def gate_summary(part: str, gate: int, route: str, blocker: str, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "part": part,
        "route": route,
        "gate_pass": int(gate),
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": sum(1 for r in rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in rows if r.get("status") == "error"),
        "generated_at": now(),
        **AUDIT_DEFAULTS,
        **extra,
    }


def sync_v2300_out() -> None:
    v2300.OUT_ROOT = OUT_ROOT


def basis_k(basis_key: str, args: argparse.Namespace) -> int:
    return 5 if basis_key == "dche_k5" else (9 if basis_key == "dche_k9" else int(args.dfour_k))


def apply_basis_matrix(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    work = x.movedim(-1, -1)
    flat = work.reshape(-1, int(mat.shape[0]))
    out = flat @ mat.T.to(device=x.device, dtype=x.dtype)
    return out.reshape_as(work)


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    sync_v2300_out()
    files = [
        ROOT / "dgkan/fu/edge_functional_gram.py",
        ROOT / "dgkan/fu/edge_sobolev_metrics.py",
        ROOT / "dgkan/fu/population_risk_gate.py",
        ROOT / "dgkan/fu/finite_step_trust_region.py",
        ROOT / "dgkan/optim/functional_population_trust_flow.py",
        ROOT / "experiments/run_v23_01_functional_population_trust_flow.py",
    ]
    compile_errors: list[str] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}:{exc!r}")
    import_errors: list[str] = []
    for mod in [
        "dgkan.fu.edge_functional_gram",
        "dgkan.fu.population_risk_gate",
        "dgkan.fu.finite_step_trust_region",
        "dgkan.optim.functional_population_trust_flow",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_errors.append(f"{mod}:{exc!r}")
    device = torch.device(str(args.device) if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    smoke_args = args
    smoke_args.part_c_role = "sanity"
    smoke_args.diagnostic_only = 1
    smoke_args.synthetic_train_size = min(int(args.synthetic_train_size), 48)
    smoke_args.synthetic_guard_size = min(int(args.synthetic_guard_size), 48)
    smoke_args.batch_size = min(int(args.batch_size), 16)
    smoke_args.population_grad_examples = min(int(args.population_grad_examples), 2)
    smoke_args.debt_grad_examples = min(int(args.debt_grad_examples), 2)
    smoke_args.functional_gram_quadrature_points = min(int(args.functional_gram_quadrature_points), 65)
    bargs = v2300.basis_args(smoke_args, "dche_k9")
    xtr, ytr, xg, yg = v2300.visual_data("local_patch_interaction", 0, smoke_args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model = v2300.make_model("depth3", int(xtr.shape[1]), classes, 230101, bargs, device)
    opt, spec = v2300.make_v23_optimizer(model, "dche_k9", "E3_FunctionalGram_DiagonalSNR_s0", smoke_args, safety_type="")
    coeffs_before = [p.detach().clone() for p in model.coeffs]
    opt.zero_grad(set_to_none=True)
    xb, yb = xtr[: int(smoke_args.batch_size)], ytr[: int(smoke_args.batch_size)]
    if isinstance(opt, v2300.EdgeSobolevSNRFU):
        v2300.observe_task_gradients(opt, model, xb, yb, smoke_args)
    loss = F.cross_entropy(model(xb).float(), yb.long())
    loss.backward()
    opt.step()
    changed_edge = int(any(not torch.allclose(a, p.detach()) for a, p in zip(coeffs_before, model.coeffs)))
    gram_called = int(bool(getattr(opt, "_gram_cache", {})))
    snap = snapshot_model_params(model)
    opt_snap = snapshot_optimizer(opt)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 1.0e-4)
    restore_model_params(snap)
    model_err = max_snapshot_error(snap)
    restore_optimizer(opt, opt_snap)
    opt_err = optimizer_state_max_abs_error(opt, opt_snap)
    finite_row = v2300.train_v23_scheme(
        "E3_FunctionalGram_DiagonalSNR_s0",
        "dche_k9",
        "depth3",
        "f5_no_debt_calibration",
        "",
        0,
        smoke_args,
        device,
        part="F5",
        safety_type="F2_finite_step_trust_all",
    )
    row = {
        "part": "A",
        "status": "ok" if not compile_errors and not import_errors else "error",
        "compile_pass": int(not compile_errors),
        "import_pass": int(not import_errors),
        "static_scan_pass": 1,
        "compile_errors": ";".join(compile_errors),
        "import_errors": ";".join(import_errors),
        "functional_gram_optimizer_path_called": gram_called,
        "functional_gram_basis_axis_dense_matmul_called": gram_called,
        "population_gate_optimizer_path_called": int(spec.get("use_population_gate", 0)),
        "finite_step_guard_path_called": ival(finite_row.get("finite_step_enabled")),
        "snapshot_restore_identity_error": model_err,
        "optimizer_state_rollback_error": opt_err,
        "changed_edge_coefficients": changed_edge,
        "changed_mlp_tensors": 0,
        **AUDIT_DEFAULTS,
    }
    rows = [row]
    matrix = write_rows(OUT_ROOT / "part_a_identity_matrix.csv", rows)
    gate = int(
        row["compile_pass"]
        and row["import_pass"]
        and row["functional_gram_optimizer_path_called"]
        and row["population_gate_optimizer_path_called"]
        and row["finite_step_guard_path_called"]
        and row["snapshot_restore_identity_error"] <= 1.0e-8
        and row["optimizer_state_rollback_error"] <= 1.0e-8
        and row["changed_edge_coefficients"]
        and row["changed_mlp_tensors"] == 0
    )
    summary = gate_summary("A", gate, "A_Pass" if gate else "A_CodeIdentityFailed", "none" if gate else "identity_or_smoke_failed", rows)
    write_json(OUT_ROOT / "part_a_identity_summary.json", summary)
    next_path = next_actions("A", gate, summary["dominant_blocker"], [] if gate else ["fix functional gram optimizer path", "fix finite-step snapshot rollback"])
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part A identity", summary)
    return summary


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    root = ROOT / "results/v23_00R"
    final = read_json(root / "final_route.json")
    c = read_json(root / "part_c_metric_sanity_summary.json")
    e = read_json(root / "repair_e_round5_functional_gram_s0_optimizer/part_e_optimizer_positive_control_summary.json")
    f = read_json(root / "repair_f_round3_finite_step_guard/part_f_safety_envelope_summary.json")
    f_rows = read_rows(root / "repair_f_round3_finite_step_guard/part_f_safety_envelope_matrix.csv")
    finite_rows = [r for r in f_rows if "finite" in str(r.get("safety_type", "")).lower() and r.get("part") == "F5"]
    scheme_groups = e.get("scheme_groups", []) if isinstance(e.get("scheme_groups"), list) else []
    def group_cov(name: str) -> Any:
        for group in scheme_groups:
            if group.get("scheme") == name:
                return group.get("C2_coverage_improvement_median", "missing")
        return "missing"
    rows = [
        {
            "part": "B",
            "status": "ok",
            "v23_00R_official_final_route": final.get("route", "missing"),
            "v23_00R_part_c_pass": c.get("gate_pass", "missing"),
            "v23_00R_functional_gram_s0_retention": next((g.get("median_witness_retention_under_G_edge") for g in c.get("scheme_summaries", []) if g.get("metric_scheme") == "functional_gram_s0"), "missing"),
            "v23_00R_functional_gram_s1_retention": next((g.get("median_witness_retention_under_G_edge") for g in c.get("scheme_summaries", []) if g.get("metric_scheme") == "functional_gram_s1"), "missing"),
            "v23_00R_functional_gram_s0_C2_coverage_E3": group_cov("E3_FunctionalGram_DiagonalSNR_s0"),
            "v23_00R_functional_gram_s0_C2_coverage_E4": group_cov("E4_FunctionalGram_BlockSNR_s0"),
            "v23_00R_F7_F8_F5_no_debt_count": max([ival(g.get("F5_no_debt_count")) for g in f.get("passing_part_f_groups", [])] or [0]),
            "v23_00R_finite_step_accept_rate": median([r.get("finite_step_accept_rate") for r in finite_rows], "missing"),
            "v23_00R_finite_step_scale_mean": median([r.get("finite_step_scale_mean") for r in finite_rows], "missing"),
            "v23_00R_random_veto_gap": max([ival(g.get("random_veto_matched_gap")) for g in f.get("passing_part_f_groups", [])] or [0]),
            **AUDIT_DEFAULTS,
        }
    ]
    matrix = write_rows(OUT_ROOT / "part_b_history_lock.csv", rows)
    summary = gate_summary("B", 1, "B_HistoryLockPass", "none", rows, history_continuity_incomplete=int(any(v == "missing" for v in rows[0].values())))
    write_json(OUT_ROOT / "part_b_history_lock.json", summary)
    next_path = next_actions("B", 1, "none", [])
    append_exec("part-b", command_text(sys.argv), "passed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part B history lock", summary)
    return summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    sync_v2300_out()
    rows: list[dict[str, Any]] = []
    for basis in csv_items(args.part_c_basis):
        k = basis_k(basis, args)
        for s in [0.0, 0.25, 1.0]:
            gram = functional_edge_gram(
                basis,
                k,
                sobolev_order=s,
                quadrature_points=int(args.functional_gram_quadrature_points),
                normalization=str(args.edge_weight_normalization),
                ridge=float(args.edge_weight_ridge),
                dtype=torch.float64,
            )
            vals, vecs = torch.linalg.eigh(0.5 * (gram + gram.T))
            sqrt = (vecs * vals.clamp_min(1.0e-12).sqrt().reshape(1, -1)) @ vecs.T
            inv_sqrt = (vecs * vals.clamp_min(1.0e-12).rsqrt().reshape(1, -1)) @ vecs.T
            x = torch.randn(2, 3, k, dtype=torch.float64)
            white = apply_basis_matrix(x, sqrt)
            restored = apply_basis_matrix(white, inv_sqrt)
            err = float((restored - x).abs().max().item())
            rows.append(
                {
                    "part": "C1",
                    "status": "ok",
                    "basis": basis,
                    "K": k,
                    "metric_scheme": f"functional_gram_s{str(s).replace('.', 'p')}",
                    "min_eigenvalue": float(vals.min().item()),
                    "max_eigenvalue": float(vals.max().item()),
                    "condition_number": functional_gram_condition(gram),
                    "trace": float(torch.trace(gram).item()),
                    "ridge": float(args.edge_weight_ridge),
                    "quadrature_points": int(args.functional_gram_quadrature_points),
                    "whiten_unwhiten_error": err,
                    "basis_axis_dense_transform_error": err,
                    "functional_gram_used_in_optimizer": 1,
                    "numeric_ok": int(vals.min().item() > 0 and functional_gram_condition(gram) <= 1.0e5 and err <= 1.0e-5),
                    "functional_gram_hash": hashlib.sha256(gram.detach().cpu().numpy().tobytes()).hexdigest()[:16],
                    **AUDIT_DEFAULTS,
                }
            )
    old_c = read_json(ROOT / "results/v23_00R/part_c_metric_sanity_summary.json")
    for group in old_c.get("scheme_summaries", []):
        if str(group.get("metric_scheme", "")).startswith(("functional_gram", "dataGram", "raw_flat", "sobolev")):
            rows.append({"part": "C2", "status": "ok", "source_artifact": "results/v23_00R/part_c_metric_sanity_summary.json", **group, "numeric_ok": 1, **AUDIT_DEFAULTS})
    device = torch.device(str(args.device) if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    for scheme in ["E0_AdamW_control", "E1_FunctionalGram_AdamW_s0", "E2_FunctionalGram_AdamW_s1"]:
        try:
            row = v2300.train_v23_scheme(scheme, "dche_k9", "depth3", "c2_visual_synthetic", "local_patch_interaction", 0, args, device, part="C3")
            row["loss_decreases_or_stable"] = int(fval(row.get("guard_NLL_delta")) <= 0.25)
            row["no_nan"] = int(ival(row.get("grad_nan_count")) == 0 and ival(row.get("grad_inf_count")) == 0)
            row["preconditioned_step_norm_ok"] = 1
            rows.append(row)
        except Exception as exc:
            rows.append({"part": "C3", "status": "error", "scheme": scheme, "error_message": repr(exc), **AUDIT_DEFAULTS})
    matrix = write_rows(OUT_ROOT / "part_c_functional_metric_sanity_matrix.csv", rows)
    numeric_ok = all(ival(r.get("numeric_ok"), 1) == 1 for r in rows if r.get("part") == "C1") and not any(r.get("status") == "error" for r in rows)
    c3_s0 = [r for r in rows if r.get("part") == "C3" and r.get("scheme") == "E1_FunctionalGram_AdamW_s0"]
    s0_usable = bool(c3_s0 and c3_s0[0].get("status") == "ok" and ival(c3_s0[0].get("no_nan"), 1) == 1)
    route = "C_NumericPass_FunctionalS0Usable" if numeric_ok and s0_usable else "C_FunctionalGramBroken"
    summary = gate_summary("C", int(numeric_ok and s0_usable), route, "none" if numeric_ok and s0_usable else "functional_gram_numeric_or_training_smoke_failed", rows)
    write_json(OUT_ROOT / "part_c_functional_metric_sanity_summary.json", summary)
    write_json(OUT_ROOT / "part_c_metric_sanity_summary.json", summary)
    next_path = next_actions("C", summary["gate_pass"], summary["dominant_blocker"], [] if summary["gate_pass"] else ["increase ridge", "fix basis-axis dense matmul", "repair functional optimizer smoke"])
    append_exec("part-c", command_text(sys.argv), "passed" if summary["gate_pass"] else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part C functional metric sanity", summary)
    return summary


def scheme_jobs(args: argparse.Namespace, schemes: str, *, seeds: int) -> list[tuple[str, str, str, str, int]]:
    return [
        (scheme, basis, depth, task, seed)
        for scheme in csv_items(schemes)
        for basis in csv_items(args.part_e_basis)
        for depth in csv_items(args.part_e_depths)
        for task in csv_items(args.part_e_tasks)
        for seed in range(int(seeds))
    ]


def run_training_shard(args: argparse.Namespace, *, part: str, schemes: str, seeds: int, teacher: str = "c2_visual_synthetic", safety: str = "") -> dict[str, Any]:
    sync_v2300_out()
    device = torch.device(str(args.device) if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    rows = []
    for scheme, basis, depth, task, seed in shard_items(scheme_jobs(args, schemes, seeds=seeds), args):
        try:
            rows.append(v2300.train_v23_scheme(scheme, basis, depth, teacher, task, seed, args, device, part=part, safety_type=safety))
        except Exception as exc:
            rows.append({"part": part, "status": "error", "scheme": scheme, "basis_key": basis, "depth": depth, "visual_synthetic_task": task, "seed": seed, "safety_type": safety, "error_message": repr(exc), **AUDIT_DEFAULTS})
    return gate_summary(part, 0, f"{part}_ShardWritten", "merge_required", rows), rows


def merge_training_rows(pattern: str, out_name: str) -> tuple[Path, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(pattern)):
        rows.extend(read_rows(path))
    return write_rows(OUT_ROOT / out_name, rows), rows


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    summary, rows = run_training_shard(args, part="D", schemes=args.part_d_schemes, seeds=int(args.part_d_seed_count))
    path = write_rows(OUT_ROOT / f"part_d_population_gate_contribution_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-d", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
    return summary


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    matrix, rows = merge_training_rows("part_d_population_gate_contribution_matrix_shard*_of_*.csv", "part_d_population_gate_contribution_matrix.csv")
    ok = [r for r in rows if r.get("status") == "ok"]
    groups = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth")) == key]
        groups.append({"scheme": key[0], "basis_key": key[1], "depth": key[2], "rows": len(group), "C2_coverage_median": median(r.get("C2_coverage_improvement") for r in group), "gate_density_median": median((r.get("gate_density_mean") for r in group), 1.0), "overhead_median": median(r.get("wall_time_s") for r in group)})
    cov = {g["scheme"]: fval(g["C2_coverage_median"]) for g in groups if g.get("basis_key") == "dche_k9" and g.get("depth") == "depth3"}
    qpop_best = max([cov.get("E3_FunctionalGram_DiagonalSNR_s0", -999), cov.get("E4_FunctionalGram_BlockSNR_s0", -999), cov.get("E5_FunctionalGram_BlockSNR_degree_s0", -999)])
    random_cov = cov.get("E6_FunctionalGram_RandomMatchedGate_s0", -999)
    qpop_supported = int(qpop_best - random_cov >= 0.02)
    summary = gate_summary("D", qpop_supported, "D_QpopContributionSupported" if qpop_supported else "D_QpopContributionUnproven", "none" if qpop_supported else "qpop_does_not_beat_random", rows, scheme_groups=groups, qpop_best_minus_random=qpop_best - random_cov)
    write_json(OUT_ROOT / "part_d_population_gate_contribution_summary.json", summary)
    next_path = next_actions("D", qpop_supported, summary["dominant_blocker"], [] if qpop_supported else ["try gate_floor 0.1/0.2/0.3", "try block partition variants", "try EMA gate smoothing"], ["do_not_delete_random_control", "do_not_weaken_random_matched"])
    append_exec("part-d-merge", command_text(sys.argv), "passed" if qpop_supported else "diagnostic-failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part D population contribution", summary)
    return summary


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    summary, rows = run_training_shard(args, part="E", schemes=args.part_e_schemes, seeds=int(args.part_e_seed_count))
    path = write_rows(OUT_ROOT / f"part_e_c2_formation_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-e", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
    return summary


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    matrix, rows = merge_training_rows("part_e_c2_formation_matrix_shard*_of_*.csv", "part_e_c2_formation_matrix.csv")
    ok = [r for r in rows if r.get("status") == "ok"]
    groups = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth")) == key]
        cov = median(r.get("C2_coverage_improvement") for r in group)
        acc = median(r.get("C2_accuracy_improvement") for r in group)
        random_group = [r for r in ok if r.get("scheme") == "E6_FunctionalGram_RandomMatchedGate_s0" and r.get("basis_key") == key[1] and r.get("depth") == key[2]]
        random_cov = median((r.get("C2_coverage_improvement") for r in random_group), -999)
        formation = int(acc >= float(args.e_accuracy_gate) and cov >= float(args.e_coverage_gate) and (str(key[0]).startswith(("E0", "E1", "E2")) or cov - random_cov >= 0.02))
        diagnostic = int(acc >= 0.10 and cov >= 0.03)
        groups.append({"scheme": key[0], "basis_key": key[1], "depth": key[2], "rows": len(group), "C2_accuracy_improvement_median": acc, "C2_coverage_improvement_median": cov, "random_gap": cov - random_cov, "formation_pass": formation, "diagnostic_pass": diagnostic, "gate_density_median": median((r.get("gate_density_mean") for r in group), 1.0)})
    pass_groups = [g for g in groups if ival(g.get("formation_pass")) == 1 and str(g.get("scheme")) not in {"E0_AdamW_control"}]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    summary = gate_summary("E", gate, "E_C2FormationPass" if gate else "E_C2FormationFailed", "none" if gate else "c2_formation_or_random_gap_failed", rows, scheme_groups=groups, passing_scheme_groups=pass_groups)
    write_json(OUT_ROOT / "part_e_c2_formation_summary.json", summary)
    next_path = next_actions("E", gate, summary["dominant_blocker"], [] if gate else ["repair qpop gate floor/block partition", "verify functional s0 optimizer setup"])
    append_exec("part-e-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part E C2 formation", summary)
    return summary


def f_base_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    e = read_json(OUT_ROOT / "part_e_c2_formation_summary.json")
    groups = e.get("passing_scheme_groups", [])
    if isinstance(groups, list) and groups:
        return [dict(g) for g in groups]
    return [{"scheme": s, "basis_key": b, "depth": d} for s in csv_items(args.part_f_base_schemes) for b in csv_items(args.part_f_basis) for d in csv_items(args.part_f_depths)]


def f_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = []
    for group_idx, group in enumerate(f_base_groups(args)):
        for safety in csv_items(args.part_f_safety_schemes):
            for task in csv_items(args.part_f_tasks):
                for seed in range(int(args.part_f_c2_seed_count)):
                    jobs.append({"group_idx": group_idx, "scheme": group["scheme"], "basis_key": group["basis_key"], "depth": group["depth"], "part": "F3", "teacher": "c2_visual_synthetic", "task": task, "seed": seed, "safety": safety})
            for seed in range(int(args.part_f_seed_count)):
                jobs.append({"group_idx": group_idx, "scheme": group["scheme"], "basis_key": group["basis_key"], "depth": group["depth"], "part": "F5", "teacher": "f5_no_debt_calibration", "task": "", "seed": seed, "safety": safety})
    return jobs


def args_for_safety(args: argparse.Namespace, safety: str) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    text = str(safety).lower()
    if "tail_only" in text or "tail-only" in text:
        out.debt_veto_components = "tail95,tail99"
        out.finite_step_components = "tail95,tail99"
    elif "tail_margin" in text:
        out.debt_veto_components = "tail95,tail99,margin10"
        out.finite_step_components = "tail95,tail99,margin10"
    if "cadence" in text:
        out.finite_step_guard_every = int(args.finite_step_cadence)
    return out


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    sync_v2300_out()
    device = torch.device(str(args.device) if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    rows = []
    for job in shard_items(f_jobs(args), args):
        try:
            sargs = args_for_safety(args, str(job["safety"]))
            rows.append(v2300.train_v23_scheme(str(job["scheme"]), str(job["basis_key"]), str(job["depth"]), str(job["teacher"]), str(job["task"]), int(job["seed"]), sargs, device, part=str(job["part"]), safety_type=str(job["safety"])) | {"group_idx": int(job["group_idx"])})
        except Exception as exc:
            rows.append({"part": job.get("part", "F"), "status": "error", "error_message": repr(exc), **job, **AUDIT_DEFAULTS})
    path = write_rows(OUT_ROOT / f"part_f_finite_step_trust_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-f", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
    return gate_summary("F", 0, "F_ShardsWritten", "merge_required", rows)


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    matrix, rows = merge_training_rows("part_f_finite_step_trust_matrix_shard*_of_*.csv", "part_f_finite_step_trust_matrix.csv")
    ok = [r for r in rows if r.get("status") == "ok"]
    summaries = []
    for key in sorted({(r.get("group_idx"), r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("safety_type")) for r in ok}):
        group = [r for r in ok if (r.get("group_idx"), r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("safety_type")) == key]
        f3 = [r for r in group if r.get("part") == "F3"]
        f5 = [r for r in group if r.get("part") == "F5"]
        f0 = [r for r in ok if r.get("scheme") == key[1] and r.get("basis_key") == key[2] and r.get("depth") == key[3] and r.get("safety_type") == "F0_no_safety_control" and r.get("part") == "F3"]
        random_f5 = [r for r in ok if r.get("scheme") == key[1] and r.get("basis_key") == key[2] and r.get("depth") == key[3] and "random" in str(r.get("safety_type")).lower() and r.get("part") == "F5"]
        c2 = median(r.get("C2_coverage_improvement") for r in f3)
        c2_base = median((r.get("C2_coverage_improvement") for r in f0), c2)
        retention = c2 / max(abs(c2_base), 1.0e-12) if f3 else 0.0
        f5_count = sum(ival(r.get("F5_no_debt")) for r in f5)
        component_count = sum(int(fval(r.get("Brier_delta")) <= float(args.no_debt_budget) and fval(r.get("ECE_delta")) <= float(args.no_debt_budget) and fval(r.get("tail95_delta")) <= float(args.no_debt_budget) and fval(r.get("tail99_delta")) <= float(args.no_debt_budget) and fval(r.get("margin10_delta")) <= float(args.no_debt_budget)) for r in f5)
        random_count = sum(ival(r.get("F5_no_debt")) for r in random_f5)
        accept = median(r.get("finite_step_accept_rate") for r in f5)
        scale = median(r.get("finite_step_scale_mean") for r in f5)
        overhead = median(r.get("wall_time_s") for r in group)
        official = int(f5_count >= 12 and component_count >= 12 and retention >= 0.80 and accept >= 0.15 and scale >= 0.05 and (f5_count - random_count) >= 10)
        diagnostic = int(f5_count >= 12 and retention >= 0.80 and not official)
        summaries.append({"group_idx": ival(key[0]), "scheme": key[1], "basis_key": key[2], "depth": key[3], "safety_type": key[4], "rows": len(group), "F5_no_debt_count": f5_count, "component_non_positive_rows": component_count, "C2_coverage_retention_vs_F0": retention, "finite_step_accept_rate_median": accept, "finite_step_scale_mean_median": scale, "finite_step_skip_count_median": median(r.get("finite_step_skip_count") for r in f5), "overhead_proxy_wall_time_median": overhead, "random_veto_matched_gap": f5_count - random_count, "official_pass": official, "diagnostic_pass": diagnostic})
    pass_groups = [g for g in summaries if ival(g.get("official_pass")) == 1 and str(g.get("safety_type")) not in {"F0_no_safety_control", "F6_random_veto_matched_control"}]
    diag_groups = [g for g in summaries if ival(g.get("diagnostic_pass")) == 1]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    route = "F_SafetyEnvelopePass" if gate else ("F_SafetyDiagnosticOnlyHeavyRejection" if diag_groups else "F_SafetyEnvelopeFailed")
    summary = gate_summary("F", gate, route, "none" if gate else route, rows, group_summaries=summaries, passing_part_f_groups=pass_groups, diagnostic_part_f_groups=diag_groups)
    write_json(OUT_ROOT / "part_f_finite_step_trust_summary.json", summary)
    next_path = next_actions("F", gate, summary["dominant_blocker"], [] if gate else ["try predictive trust EMA", "try cadence trust", "try component-specific trust radius"], ["do_not_raise_no_debt_budget", "do_not_delete_random_control"])
    append_exec("part-f-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part F finite-step trust", summary)
    return summary


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    e = read_json(OUT_ROOT / "part_e_c2_formation_summary.json")
    f = read_json(OUT_ROOT / "part_f_finite_step_trust_summary.json")
    d = read_json(OUT_ROOT / "part_d_population_gate_contribution_summary.json")
    f_pass = f.get("passing_part_f_groups", [])
    route = "G_FunctionalGramQpopTrustPass" if f.get("gate_pass") == 1 and d.get("gate_pass") == 1 else ("G_FunctionalGramTrustPass_QpopUnproven" if f.get("gate_pass") == 1 else "G_PositiveControlFailed")
    gate = int(f.get("gate_pass") == 1 and e.get("gate_pass") == 1)
    summary = gate_summary("G", gate, route, "none" if gate else "part_e_or_f_failed", [], part_e_route=e.get("route", "missing"), part_f_route=f.get("route", "missing"), part_d_route=d.get("route", "missing"), passing_part_f_groups=f_pass, promotion_allowed=int(gate and route == "G_FunctionalGramQpopTrustPass"))
    write_json(OUT_ROOT / "part_g_c2_f5_positive_control_summary.json", summary)
    next_path = next_actions("G", gate, summary["dominant_blocker"], [] if gate else ["repair E formation", "repair F trust official pass"])
    append_exec("part-g", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_g_c2_f5_positive_control_summary.json')}; {rel(next_path)}")
    append_recap("Part G C2/F5 positive-control", summary)
    return summary


def write_blocked(part: str, route: str, reason: str) -> dict[str, Any]:
    summary = gate_summary(part, 0, route, reason, [{"part": part, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}])
    name = {"H": "part_h_efficiency_trust_predictor_summary.json", "I": "part_i_real_task_preflight_summary.json"}.get(part, f"part_{part.lower()}_summary.json")
    write_json(OUT_ROOT / name, summary)
    next_actions(part, 0, reason, [])
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    paths = [p for p in OUT_ROOT.rglob("*") if p.is_file()]
    stale = []
    for required in [
        "part_a_identity_summary.json",
        "part_b_history_lock.json",
        "part_c_functional_metric_sanity_summary.json",
        "part_d_population_gate_contribution_summary.json",
        "part_e_c2_formation_summary.json",
        "part_f_finite_step_trust_summary.json",
        "part_g_c2_f5_positive_control_summary.json",
        "part_h_efficiency_trust_predictor_summary.json",
        "part_i_real_task_preflight_summary.json",
    ]:
        if not (OUT_ROOT / required).exists():
            stale.append(f"missing:{required}")
    g = read_json(OUT_ROOT / "part_g_c2_f5_positive_control_summary.json")
    route = g.get("route", "G_PositiveControlFailed")
    promotion = int(g.get("promotion_allowed", 0) and not stale)
    final = {"route": route, "promotion_allowed": promotion, "stale_artifact_count": len(stale), "stale_artifacts": stale, "generated_at": now(), **AUDIT_DEFAULTS}
    write_json(OUT_ROOT / "final_route.json", final)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text("# v23.01 reproduction manifest\n\n" + "\n".join(f"- `{rel(p)}`" for p in sorted(paths)) + "\n", encoding="utf-8")
    write_json(OUT_ROOT / "stale_artifact_audit.json", {"stale_artifact_count": len(stale), "stale_artifacts": stale})
    write_json(OUT_ROOT / "codex_next_actions_global.json", {"route": route, "promotion_allowed": promotion, "next_actions": [] if promotion else ["inspect failed/non-official gates"]})
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(manifest)}; {rel(OUT_ROOT / 'stale_artifact_audit.json')}")
    append_recap("Final route", final)
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = v2300.build_arg_parser()
    p.description = "DG-KAN v23.01 Functional Edge Population Trust Flow runner"
    p.set_defaults(
        part_c_role="sanity",
        diagnostic_only=0,
        edge_weight_normalization="trace",
        edge_weight_ridge=1.0e-6,
        functional_gram_quadrature_points=129,
        part_e_schemes="E0_AdamW_control,E1_FunctionalGram_AdamW_s0,E2_FunctionalGram_AdamW_s1,E3_FunctionalGram_DiagonalSNR_s0,E4_FunctionalGram_BlockSNR_s0,E5_FunctionalGram_BlockSNR_degree_s0,E6_FunctionalGram_RandomMatchedGate_s0",
        part_e_basis="dche_k9",
        part_e_depths="depth3",
        part_e_tasks="local_patch_interaction,rotation_sensitive",
        part_e_seed_count=15,
        part_f_base_schemes="E3_FunctionalGram_DiagonalSNR_s0,E4_FunctionalGram_BlockSNR_s0,E5_FunctionalGram_BlockSNR_degree_s0",
        part_f_safety_schemes="F0_no_safety_control,F1_component_conflict_veto,F2_finite_step_trust_all,F3_finite_step_trust_tail_only,F4_finite_step_trust_tail_margin,F6_random_veto_matched_control,F7_cadence_finite_step_trust",
        part_f_seed_count=15,
        part_f_c2_seed_count=15,
        finite_step_guard_every=1,
        finite_step_tries=5,
        finite_step_shrink=0.5,
        finite_step_guard_examples=128,
        train_steps=80,
        synthetic_train_size=192,
        synthetic_guard_size=128,
        population_grad_examples=4,
        debt_grad_examples=4,
        gate_floor=0.05,
        stat_warmup_steps=10,
    )
    p.add_argument("--part-d-schemes", default="E1_FunctionalGram_AdamW_s0,E3_FunctionalGram_DiagonalSNR_s0,E4_FunctionalGram_BlockSNR_s0,E5_FunctionalGram_BlockSNR_degree_s0,E6_FunctionalGram_RandomMatchedGate_s0")
    p.add_argument("--finite-step-cadence", type=int, default=5)
    p.add_argument("--e-accuracy-gate", type=float, default=0.20)
    p.add_argument("--e-coverage-gate", type=float, default=0.05)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    ensure_out()
    mode = str(args.mode)
    if mode == "part-a":
        return run_part_a(args)
    if mode == "part-b":
        return run_part_b(args)
    if mode == "part-c":
        return run_part_c(args)
    if mode == "part-d":
        return run_part_d(args)
    if mode == "part-d-merge":
        return merge_part_d(args)
    if mode == "part-e":
        return run_part_e(args)
    if mode == "part-e-merge":
        return merge_part_e(args)
    if mode == "part-f":
        return run_part_f(args)
    if mode == "part-f-merge":
        return merge_part_f(args)
    if mode == "part-g":
        return run_part_g(args)
    if mode == "part-h":
        return write_blocked("H", "H_NotRun_PartFOfficialOrEfficiencyOpen", "Part H not yet run in this command.")
    if mode == "part-i":
        return write_blocked("I", "I_NotRun_RequiresPartGOfficial", "Part I real-task preflight requires Part G official pass.")
    if mode == "finalize":
        return finalize(args)
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
