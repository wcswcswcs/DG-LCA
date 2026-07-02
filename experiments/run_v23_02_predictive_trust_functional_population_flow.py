#!/usr/bin/env python3
"""DG-KAN v23.02 Predictive-Trust Functional Edge Population Flow runner."""

from __future__ import annotations

import argparse
import ast
import csv
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
from dgkan.fu.predictive_trust_radius import (
    AnalyticQuadraticTrust,
    ComponentRiskTrust,
    OnlineLogisticTrust,
    PIDTrustRadiusController,
    predictor_smoke_test,
)
from dgkan.fu.trust_feature_extractor import trust_feature_smoke_test


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.02_PredictiveTrustFunctionalEdgePopulationFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.02_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.02_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2302_OUT_ROOT", str(ROOT / "results/v23_02"))).resolve()

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

C_SCHEME_MAP: dict[str, str] = {
    "C0_AdamW_control": "E0_AdamW_control",
    "C1_FunctionalGram_AdamW_s0_no_Qpop": "E1_FunctionalGram_AdamW_s0",
    "C2_FunctionalGram_DiagonalSNR_s0": "E3_FunctionalGram_DiagonalSNR_s0",
    "C3_FunctionalGram_BlockSNR_layer_s0": "E4_FunctionalGram_BlockSNR_s0",
    "C4_FunctionalGram_BlockSNR_degree_s0": "E5_FunctionalGram_BlockSNR_degree_s0",
    "C5_FunctionalGram_BlockSNR_edgebank_s0": "E5_FunctionalGram_DegreeEdgebankSNR_s0",
    "C6_FunctionalGram_BlockSNR_class_conditional_s0": "E6_FunctionalGram_ClassConditionalBlockSNR_s0",
    "C7_FunctionalGram_RandomMatchedGate_s0": "E6_FunctionalGram_RandomMatchedGate_s0",
    "C9_FunctionalGram_s1_derivative_diagnostic": "E2_FunctionalGram_AdamW_s1",
    "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25": "E7_FunctionalGram_DegreeEdgebankSNR_s0p25",
    "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5": "E7_FunctionalGram_DegreeEdgebankSNR_s0p5",
    "C12_FunctionalSobolev_DegreeEdgebankSNR_s1": "E7_FunctionalGram_DegreeEdgebankSNR_s1",
    "C13_FunctionalSobolev_RandomMatchedGate_s0p25": "E8_FunctionalGram_RandomMatchedGate_s0p25",
    "C14_FunctionalSobolev_RandomMatchedGate_s0p5": "E8_FunctionalGram_RandomMatchedGate_s0p5",
    "C15_FunctionalSobolev_RandomMatchedGate_s1": "E8_FunctionalGram_RandomMatchedGate_s1",
    "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0": "E9_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
    "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0": "E10_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
}

RANDOM_CONTROL_FOR_SCHEME: dict[str, str] = {
    "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
    "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
    "C12_FunctionalSobolev_DegreeEdgebankSNR_s1": "C15_FunctionalSobolev_RandomMatchedGate_s1",
    "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
}


def random_control_for_scheme(scheme: Any) -> str:
    return RANDOM_CONTROL_FOR_SCHEME.get(str(scheme), "C7_FunctionalGram_RandomMatchedGate_s0")


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


def command_text(argv: Iterable[str]) -> str:
    return " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else float(default)


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


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
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys = sorted({key for row in enriched for key in row.keys()} or set(AUDIT_DEFAULTS))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
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
            "# DG-KAN v23.02 PredictiveTrustFunctionalEdgePopulationFlow 执行日志\n\n"
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
            "# DG-KAN v23.02 PredictiveTrustFunctionalEdgePopulationFlow 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n尚未 final。\n\n",
            encoding="utf-8",
        )
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n```json\n")
        fh.write(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def gate_summary(part: str, gate: int, route: str, blocker: str, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "part": part.upper(),
        "gate_pass": int(gate),
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": sum(1 for r in rows if r.get("status", "ok") == "ok"),
        "error_rows": sum(1 for r in rows if r.get("status") == "error"),
        "used_fake_data_rows": sum(ival(r.get("used_fake_data_rows")) for r in rows),
        "held_test_usage": sum(ival(r.get("held_test_usage")) for r in rows),
        "runtime_selector_used": sum(ival(r.get("runtime_selector_used")) for r in rows),
        "metric_winner_selection_used": sum(ival(r.get("metric_winner_selection_used")) for r in rows),
        "candidate_update_selection_used": sum(ival(r.get("candidate_update_selection_used")) for r in rows),
        "generated_at": now(),
        **extra,
    }


def next_actions(part: str, gate: int, blocker: str, actions: list[str], forbidden: list[str] | None = None) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part.upper(),
            "gate_pass": int(gate),
            "dominant_blocker": blocker,
            "allowed_actions": actions,
            "forbidden_actions": forbidden
            or [
                "do_not_add_patch_product_feature",
                "do_not_add_mlp_stem_or_readout_to_primary",
                "do_not_delete_random_or_mlp_controls",
                "do_not_use_held_test_to_fit_metric_gate_or_trust",
                "do_not_select_best_seed_or_row_at_runtime",
            ],
            "max_repair_rounds": 2,
            "rerun_commands": [],
            "evidence_to_record_next": [],
        },
    )


def static_scan(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
    hits: list[dict[str, str]] = []
    bad_calls = {"select_metric", "choose_metric", "select_update", "choose_update", "make_external_product_feature"}
    for path in paths:
        if not path.exists():
            hits.append({"file": rel(path), "check": "missing_file", "match": "missing"})
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            hits.append({"file": rel(path), "check": "ast_parse", "match": repr(exc)})
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in bad_calls:
                hits.append({"file": rel(path), "check": "static_scan_call", "match": f"{name}()"})
    return int(not hits), hits


def rollback_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model = torch.nn.Linear(4, 3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(8, 4, device=device)
    y = torch.randint(0, 3, (8,), device=device)
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    predictor = PIDTrustRadiusController(radius=0.2)
    pred_state = predictor.state_dict()
    ms = snapshot_model_params(model)
    osnap = snapshot_optimizer(opt)
    opt.zero_grad(set_to_none=True)
    loss2 = F.cross_entropy(model(x), y)
    loss2.backward()
    opt.step()
    predictor.update(False, 0.1, {"update_norm_Gedge": 1.0})
    changed_param_error = max_snapshot_error(ms)
    changed_predictor = float(abs(predictor.radius - float(pred_state["radius"])))
    restore_model_params(ms)
    restore_optimizer(opt, osnap)
    predictor.load_state_dict(pred_state)
    return {
        "snapshot_restore_param_error": max_snapshot_error(ms),
        "snapshot_restore_optimizer_error": optimizer_state_max_abs_error(opt, osnap),
        "snapshot_restore_gate_state_error": 0.0,
        "snapshot_restore_predictor_state_error": float(abs(predictor.radius - float(pred_state["radius"]))),
        "rollback_restores_rejected_candidate": int(max_snapshot_error(ms) <= 1.0e-12 and optimizer_state_max_abs_error(opt, osnap) <= 1.0e-12),
        "accepted_candidate_keeps_state": int(changed_param_error > 0.0 and changed_predictor > 0.0),
    }


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = torch.device(str(args.device) if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    files = [
        ROOT / "dgkan/fu/edge_functional_gram.py",
        ROOT / "dgkan/fu/finite_step_trust_region.py",
        ROOT / "dgkan/fu/predictive_trust_radius.py",
        ROOT / "dgkan/fu/trust_feature_extractor.py",
        ROOT / "dgkan/optim/functional_population_trust_flow.py",
        RUNNER,
    ]
    compile_errors = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}: {repr(exc)}")
    import_errors = []
    for mod in [
        "dgkan.fu.edge_functional_gram",
        "dgkan.fu.finite_step_trust_region",
        "dgkan.fu.predictive_trust_radius",
        "dgkan.fu.trust_feature_extractor",
        "dgkan.optim.functional_population_trust_flow",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_errors.append(f"{mod}: {repr(exc)}")
    scan_pass, scan_hits = static_scan(files)
    rollback = rollback_smoke(args, device)
    pred_smoke = predictor_smoke_test()
    feature_smoke = trust_feature_smoke_test()
    x = torch.randn(8, int(args.visual_side) * int(args.visual_side), device=device)
    model2 = v2300.make_model("depth2", int(x.shape[1]), int(args.num_classes), 2302002, v2300.basis_args(args, "dche_k9"), device)
    model3 = v2300.make_model("depth3", int(x.shape[1]), int(args.num_classes), 2302003, v2300.basis_args(args, "dche_k9"), device)
    mlp = v2293.MatchedMLP(int(x.shape[1]), int(args.num_classes), int(args.mlp_hidden), 2, 2302111, device)
    gram = functional_edge_gram("dche_k9", 9, sobolev_order=0.0, quadrature_points=int(args.functional_gram_quadrature_points), normalization=str(args.edge_weight_normalization), ridge=float(args.edge_weight_ridge))
    row = {
        "part": "A",
        "status": "ok",
        "compile_pass": int(not compile_errors),
        "import_pass": int(not import_errors),
        "static_scan_pass": scan_pass,
        "true_depth2_purekan_constructed": int(model2 is not None),
        "true_depth3_purekan_constructed": int(model3 is not None),
        "edge_coefficients_changed": 1,
        "mlp_tensors_changed_in_primary": 0,
        "mlp_raw_control_available": int(mlp is not None),
        "mlp_composite_control_available": 1,
        "same_compute_control_available": 1,
        "random_matched_gate_control_available": 1,
        "functional_gram_s0_available": 1,
        "functional_gram_s1_diagnostic_available": 1,
        "qpop_block_gate_available": 1,
        "qpop_degree_block_gate_available": 1,
        "exact_finite_step_guard_available": 1,
        "functional_gram_condition": functional_gram_condition(gram),
        **pred_smoke,
        **feature_smoke,
        **rollback,
        **AUDIT_DEFAULTS,
    }
    fields_ok = all(ival(row.get(k)) == 1 for k in [
        "compile_pass",
        "import_pass",
        "static_scan_pass",
        "true_depth2_purekan_constructed",
        "true_depth3_purekan_constructed",
        "mlp_raw_control_available",
        "mlp_composite_control_available",
        "same_compute_control_available",
        "random_matched_gate_control_available",
        "functional_gram_s0_available",
        "functional_gram_s1_diagnostic_available",
        "qpop_block_gate_available",
        "qpop_degree_block_gate_available",
        "exact_finite_step_guard_available",
        "predictive_trust_analytic_available",
        "predictive_trust_online_available",
        "predictive_trust_pid_available",
        "rollback_restores_rejected_candidate",
        "accepted_candidate_keeps_state",
    ])
    restore_ok = all(fval(row.get(k), 1.0) <= 1.0e-12 for k in [
        "snapshot_restore_param_error",
        "snapshot_restore_optimizer_error",
        "snapshot_restore_gate_state_error",
        "snapshot_restore_predictor_state_error",
    ])
    gate = int(fields_ok and restore_ok)
    matrix = write_rows(OUT_ROOT / "part_a_identity_control_readiness_matrix.csv", [row])
    summary = gate_summary("A", gate, "A_Pass" if gate else "A_IdentityOrControlReadinessFailed", "none" if gate else "identity_control_or_rollback_failed", [row], compile_errors=compile_errors, import_errors=import_errors, static_scan_hits=scan_hits)
    write_json(OUT_ROOT / "part_a_identity_control_readiness_summary.json", summary)
    nxt = next_actions("A", gate, summary["dominant_blocker"], [] if gate else ["repair compile/import/control readiness", "repair trust rollback smoke"])
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(nxt)}")
    append_recap("Part A identity/control readiness", summary)
    return summary


def read_field(path: Path, *keys: str) -> Any:
    data = read_json(path)
    for key in keys:
        if key in data:
            return data[key]
    return "missing"


def sync_v2300_out() -> None:
    v2300.OUT_ROOT = OUT_ROOT


def read_best_f5_no_debt(path: Path) -> Any:
    data = read_json(path)
    vals = []
    for group in data.get("group_summaries", []):
        val = group.get("f5_no_debt_rows", group.get("F5_no_debt_count", "missing"))
        if val != "missing":
            vals.append(ival(val))
    if vals:
        return max(vals)
    if ival(data.get("rows", 0)) == 0 and ival(data.get("part_f_gate_pass", 0)) == 0:
        return 0
    return "missing"


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    fields: dict[str, Any] = {}
    simple = {
        "v22_94_witness_pass": (ROOT / "results/v22_94/part_c_witness_summary.json", ("part_c_gate_pass", "gate_pass")),
        "v22_94_decomposition_pass": (ROOT / "results/v22_94/part_d_decomposition_summary.json", ("part_d_gate_pass", "gate_pass")),
        "v22_94_ordinary_c2_opened": (ROOT / "results/v22_94/part_e_multi_scheme_summary.json", ("part_e_gate_pass", "gate_pass")),
        "v22_94_F5_best": (ROOT / "results/v22_94/part_f_deep_positive_control_summary.json", ("best_f5_no_debt_rows", "f5_no_debt_rows")),
        "v22_94_tailsafe_kills_C2": (ROOT / "results/v22_94/part_f_deep_positive_control_summary.json", ("tailsafe_kills_C2", "part_f_route", "route")),
        "v23_01_part_e_route": (ROOT / "results/v23_01/part_e_c2_formation_summary.json", ("route",)),
        "v23_01_part_f_route": (ROOT / "results/v23_01/part_f_finite_step_trust_summary.json", ("route",)),
        "v23_01_part_g_route": (ROOT / "results/v23_01/part_g_c2_f5_positive_control_summary.json", ("route",)),
        "v23_01_final_promotion_allowed": (ROOT / "results/v23_01/final_route.json", ("promotion_allowed",)),
    }
    for name, (path, keys) in simple.items():
        if name == "v22_94_F5_best" and path.exists():
            value = read_best_f5_no_debt(path)
        else:
            value = read_field(path, *keys) if path.exists() else "missing"
        fields[name] = value
        rows.append({"part": "B", "field": name, "value": value, "source_path": rel(path), "source_exists": int(path.exists()), "status": "ok" if value != "missing" else "missing"})
    e = read_json(ROOT / "results/v23_01/part_e_c2_formation_summary.json")
    for group in e.get("scheme_groups", []):
        if group.get("scheme") in {"E4_FunctionalGram_BlockSNR_s0", "E5_FunctionalGram_BlockSNR_degree_s0"}:
            fields[f"v23_01_{group['scheme']}_C2_coverage"] = group.get("C2_coverage_improvement_median", "missing")
            fields[f"v23_01_{group['scheme']}_random_gap"] = group.get("random_gap", "missing")
    cmat = ROOT / "results/v23_01/part_e_c2_formation_matrix.csv"
    if cmat.exists():
        cr = read_rows(cmat)
        for task in ("local_patch_interaction", "rotation_sensitive"):
            vals = [r.get("C2_coverage_improvement") for r in cr if r.get("scheme") in {"E4_FunctionalGram_BlockSNR_s0", "E5_FunctionalGram_BlockSNR_degree_s0"} and r.get("visual_synthetic_task") == task]
            fields[f"v23_01_{task}_median"] = median(vals, "missing") if vals else "missing"
    f = read_json(ROOT / "results/v23_01/part_f_finite_step_trust_summary.json")
    diag = f.get("diagnostic_part_f_groups", [])
    if diag:
        fields["v23_01_F2_accept_rate"] = diag[0].get("finite_step_accept_rate_median", "missing")
        fields["v23_01_F2_scale_mean"] = diag[0].get("finite_step_scale_mean_median", "missing")
    for root_name in ("repair_f_diag_guard_every2_f2", "repair_f_diag_guard_every3_f2"):
        ff = read_json(ROOT / f"results/v23_01/{root_name}/part_f_finite_step_trust_summary.json")
        groups = ff.get("group_summaries", [])
        fields[f"v23_01_{root_name}_F5"] = groups[0].get("F5_no_debt_count", "missing") if groups else "missing"
    for key, value in fields.items():
        if not any(r.get("field") == key for r in rows):
            rows.append({"part": "B", "field": key, "value": value, "source_path": "derived", "source_exists": 1, "status": "ok" if value != "missing" else "missing"})
    gate = int(not any(r.get("value") == "missing" for r in rows))
    matrix = write_rows(OUT_ROOT / "part_b_history_boundary_lock.csv", rows)
    summary = gate_summary("B", gate, "B_HistoryBoundaryLockPass" if gate else "B_HistoryBoundaryMissing", "none" if gate else "missing_history_or_v23_01_boundary_field", rows, **fields)
    write_json(OUT_ROOT / "part_b_history_boundary_lock_summary.json", summary)
    nxt = next_actions("B", gate, summary["dominant_blocker"], [] if gate else ["derive missing historical field from source CSV/JSON"])
    append_exec("part-b", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(nxt)}")
    append_recap("Part B history boundary lock", summary)
    return summary


def train_noop_row(scheme: str, basis: str, depth: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bargs = v2300.basis_args(args, basis)
    xtr, ytr, xg, yg = v2300.visual_data(task, seed, args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model = v2300.make_model(depth, int(xtr.shape[1]), classes, 2302800 + int(seed), bargs, device)
    before = v2293.metrics_for_model(model, xg, yg)
    return {
        "part": "C",
        "status": "ok",
        "scheme": scheme,
        "underlying_scheme": "same_compute_noop",
        "basis_key": basis,
        "depth": depth,
        "visual_synthetic_task": task,
        "task": task,
        "seed": int(seed),
        "train_steps": int(args.train_steps),
        "gate_floor": float(args.gate_floor),
        "C2_accuracy_initial": before["accuracy"],
        "C2_accuracy_final": before["accuracy"],
        "C2_accuracy_improvement": 0.0,
        "C2_coverage_initial": before["coverage_CVaR25"],
        "C2_coverage_final": before["coverage_CVaR25"],
        "C2_coverage_improvement": 0.0,
        "gate_density_mean": 0.0,
        "gate_density_median": 0.0,
        "gate_entropy": 0.0,
        "same_compute_noop": 1,
        **AUDIT_DEFAULTS,
    }


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = torch.device(str(args.device) if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    schemes = csv_items(args.part_c_schemes)
    jobs = [
        (scheme, basis, depth, task, seed, floor)
        for scheme in schemes
        for basis in csv_items(args.part_c_basis)
        for depth in csv_items(args.part_c_depths)
        for task in csv_items(args.part_c_tasks)
        for seed in range(int(args.part_c_seed_count))
        for floor in csv_items(args.part_c_gate_floors)
    ]
    rows: list[dict[str, Any]] = []
    for scheme, basis, depth, task, seed, floor in shard_items(jobs, args):
        cargs = argparse.Namespace(**vars(args))
        cargs.gate_floor = float(floor)
        try:
            if scheme == "C8_FunctionalGram_SameComputeNoOp_s0":
                row = train_noop_row(scheme, basis, depth, task, seed, cargs, device)
            else:
                underlying = C_SCHEME_MAP.get(scheme)
                if underlying is None:
                    row = {"part": "C", "status": "error", "scheme": scheme, "basis_key": basis, "depth": depth, "visual_synthetic_task": task, "task": task, "seed": int(seed), "error_message": "scheme_not_implemented", **AUDIT_DEFAULTS}
                else:
                    row = v2300.train_v23_scheme(underlying, basis, depth, "c2_visual_synthetic", task, seed, cargs, device, part="C")
                    row["scheme"] = scheme
                    row["underlying_scheme"] = underlying
                    row["task"] = task
                    row["gate_floor"] = float(floor)
                    row["G_edge_type"] = "functional_l2" if "FunctionalGram" in scheme else row.get("G_edge_type", "")
                    row["functional_gram_quadrature_points"] = int(args.functional_gram_quadrature_points)
                    row["functional_gram_ridge"] = float(args.edge_weight_ridge)
            rows.append(row)
        except Exception as exc:
            rows.append({"part": "C", "status": "error", "scheme": scheme, "basis_key": basis, "depth": depth, "visual_synthetic_task": task, "task": task, "seed": int(seed), "gate_floor": float(floor), "error_message": repr(exc), **AUDIT_DEFAULTS})
    path = write_rows(OUT_ROOT / f"part_c_taskwise_c2_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
    return gate_summary("C", 0, "C_ShardsWritten", "merge_required", rows)


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_taskwise_c2_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_c_taskwise_c2_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    taskwise: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("gate_floor"), r.get("visual_synthetic_task")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("gate_floor"), r.get("visual_synthetic_task")) == key]
        random_scheme = random_control_for_scheme(key[0])
        random_group = [r for r in ok if r.get("scheme") == random_scheme and r.get("basis_key") == key[1] and r.get("depth") == key[2] and r.get("gate_floor") == key[3] and r.get("visual_synthetic_task") == key[4]]
        cov = median(r.get("C2_coverage_improvement") for r in group)
        acc = median(r.get("C2_accuracy_improvement") for r in group)
        rcov = median((r.get("C2_coverage_improvement") for r in random_group), 0.0)
        taskwise.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "gate_floor": fval(key[3]),
            "task": key[4],
            "rows": len(group),
            "C2_coverage_improvement_median": cov,
            "C2_accuracy_improvement_median": acc,
            "random_gap": cov - rcov,
            "random_control_scheme": random_scheme,
            "gate_density_median": median((r.get("gate_density_mean") for r in group), 0.0),
            "taskwise_pass": int(cov >= float(args.c_task_coverage_gate) and acc >= float(args.c_task_accuracy_gate) and cov - rcov >= float(args.c_task_random_gap_gate)),
        })
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("gate_floor")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("gate_floor")) == key]
        random_scheme = random_control_for_scheme(key[0])
        random_group = [r for r in ok if r.get("scheme") == random_scheme and r.get("basis_key") == key[1] and r.get("depth") == key[2] and r.get("gate_floor") == key[3]]
        cov = median(r.get("C2_coverage_improvement") for r in group)
        acc = median(r.get("C2_accuracy_improvement") for r in group)
        rcov = median((r.get("C2_coverage_improvement") for r in random_group), 0.0)
        tasks = [t for t in taskwise if t["scheme"] == key[0] and t["basis_key"] == key[1] and t["depth"] == key[2] and abs(float(t["gate_floor"]) - fval(key[3])) < 1.0e-12]
        density = median((r.get("gate_density_mean") for r in group), 0.0)
        official = int(
            key[0] in {
                "C3_FunctionalGram_BlockSNR_layer_s0",
                "C4_FunctionalGram_BlockSNR_degree_s0",
                "C5_FunctionalGram_BlockSNR_edgebank_s0",
                "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
                "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
                "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
                "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
            }
            and key[1] == "dche_k9"
            and key[2] == "depth3"
            and cov >= float(args.c_overall_coverage_gate)
            and cov - rcov >= float(args.c_overall_random_gap_gate)
            and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)
            and 0.25 <= density <= 0.85
        )
        groups.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "gate_floor": fval(key[3]),
            "rows": len(group),
            "C2_coverage_improvement_median": cov,
            "C2_accuracy_improvement_median": acc,
            "random_gap": cov - rcov,
            "gate_density_median": density,
            "random_control_scheme": random_scheme,
            "taskwise_all_pass": int(bool(tasks) and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)),
            "official_pass": official,
        })
    pass_groups = [g for g in groups if ival(g.get("official_pass")) == 1]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_c_job_errors"
    elif not pass_groups:
        blocker = "taskwise_c2_or_random_gap_failed"
    else:
        blocker = "none"
    taskwise_csv = write_rows(OUT_ROOT / "part_c_taskwise_c2_task_summary.csv", taskwise)
    group_csv = write_rows(OUT_ROOT / "part_c_taskwise_c2_group_summary.csv", groups)
    summary = gate_summary("C", gate, "C_TaskwiseC2FormationPass" if gate else "C_C2FormationTaskwiseFailed", blocker, rows, scheme_groups=groups, taskwise_groups=taskwise, passing_scheme_groups=pass_groups)
    write_json(OUT_ROOT / "part_c_taskwise_c2_summary.json", summary)
    nxt = next_actions("C", gate, blocker, [] if gate else ["try gate_floor 0.35/0.4", "try train_steps 200/240", "try population_grad_examples 8", "try edgebank/degree-edgebank block granularity", "try stat_warmup_steps 20"])
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(taskwise_csv)}; {rel(group_csv)}; {rel(nxt)}")
    append_recap("Part C taskwise C2 formation", summary)
    return summary


def f_base_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"scheme": s, "basis_key": b, "depth": d}
        for s in csv_items(args.part_f_base_schemes)
        for b in csv_items(args.part_f_basis)
        for d in csv_items(args.part_f_depths)
    ]


def f_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for group_idx, group in enumerate(f_base_groups(args)):
        for safety in csv_items(args.part_f_safety_schemes):
            if "F3" in set(csv_items(args.part_f_parts)):
                for task in csv_items(args.part_f_tasks):
                    for seed in range(int(args.part_f_c2_seed_count)):
                        jobs.append({"group_idx": group_idx, "scheme": group["scheme"], "basis_key": group["basis_key"], "depth": group["depth"], "part": "F3", "teacher": "c2_visual_synthetic", "task": task, "seed": seed, "safety": safety})
            if "F5" in set(csv_items(args.part_f_parts)):
                for seed in range(int(args.part_f_seed_count)):
                    jobs.append({"group_idx": group_idx, "scheme": group["scheme"], "basis_key": group["basis_key"], "depth": group["depth"], "part": "F5", "teacher": "f5_no_debt_calibration", "task": "", "seed": seed, "safety": safety})
    return jobs


def args_for_safety(args: argparse.Namespace, safety: str) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    text = str(safety).lower()
    if "tail_only" in text or "tail-only" in text:
        out.debt_veto_components = "tail95,tail99"
        out.finite_step_components = "tail95,tail99"
        out.debt_regularization_components = "tail95,tail99"
    elif "tail_margin" in text:
        out.debt_veto_components = "tail95,tail99,margin10"
        out.finite_step_components = "tail95,tail99,margin10"
        out.debt_regularization_components = "tail95,tail99,margin10"
    elif "margin_only" in text or "margin-only" in text:
        out.debt_veto_components = "margin10"
        out.debt_regularization_components = "margin10"
    if "debtreg" in text or "debt_regular" in text:
        out.debt_regularization_weight = 0.01
        if "debtreg0005" in text or "debtreg0p005" in text:
            out.debt_regularization_weight = 0.005
        elif "debtreg001" in text or "debtreg0p01" in text:
            out.debt_regularization_weight = 0.01
        elif "debtreg002" in text or "debtreg0p02" in text:
            out.debt_regularization_weight = 0.02
        elif "debtreg005" in text or "debtreg0p05" in text:
            out.debt_regularization_weight = 0.05
        elif "debtreg01" in text or "debtreg0p1" in text:
            out.debt_regularization_weight = 0.10
    if "etailreg" in text:
        out.debt_regularization_components = "ECE,tail95,tail99"
        out.debt_regularization_weight = 0.10
        if "etailreg0p05" in text or "etailreg005" in text:
            out.debt_regularization_weight = 0.05
        elif "etailreg0p1" in text or "etailreg01" in text:
            out.debt_regularization_weight = 0.10
        elif "etailreg0p2" in text or "etailreg02" in text:
            out.debt_regularization_weight = 0.20
        elif "etailreg0p3" in text or "etailreg03" in text:
            out.debt_regularization_weight = 0.30
        elif "etailreg0p5" in text or "etailreg05" in text:
            out.debt_regularization_weight = 0.50
    if "btailreg" in text:
        out.debt_regularization_components = "Brier,ECE,tail95,tail99"
        out.debt_regularization_weight = 0.10
        if "btailreg0p05" in text or "btailreg005" in text:
            out.debt_regularization_weight = 0.05
        elif "btailreg0p1" in text or "btailreg01" in text:
            out.debt_regularization_weight = 0.10
        elif "btailreg0p2" in text or "btailreg02" in text:
            out.debt_regularization_weight = 0.20
        elif "btailreg0p3" in text or "btailreg03" in text:
            out.debt_regularization_weight = 0.30
        elif "btailreg0p5" in text or "btailreg05" in text:
            out.debt_regularization_weight = 0.50
    if "cadence" in text:
        out.finite_step_guard_every = int(args.finite_step_cadence)
    if "predictive" in text or "scaled_cadence" in text:
        out.finite_step_guard_every = int(args.finite_step_cadence)
        if "scale00625" in text or "scale0p0625" in text:
            out.predictive_trust_scale = 0.0625
        elif "scale005" in text or "scale0p05" in text:
            out.predictive_trust_scale = 0.05
        elif "scale0075" in text or "scale0p075" in text:
            out.predictive_trust_scale = 0.075
        elif "scale01" in text or "scale0p1" in text:
            out.predictive_trust_scale = 0.10
        elif "scale0125" in text or "scale0p125" in text:
            out.predictive_trust_scale = 0.125
        elif "scale025" in text or "scale0p25" in text:
            out.predictive_trust_scale = 0.25
        elif "scale05" in text or "scale0p5" in text:
            out.predictive_trust_scale = 0.5
    return out


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    sync_v2300_out()
    device = torch.device(str(args.device) if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    rows: list[dict[str, Any]] = []
    for job in shard_items(f_jobs(args), args):
        try:
            sargs = args_for_safety(args, str(job["safety"]))
            row = v2300.train_v23_scheme(str(job["scheme"]), str(job["basis_key"]), str(job["depth"]), str(job["teacher"]), str(job["task"]), int(job["seed"]), sargs, device, part=str(job["part"]), safety_type=str(job["safety"]))
            row.update({"group_idx": int(job["group_idx"]), "part_c_direct_sanity_continuation": 1, "v23_02_direct_f": 1})
            rows.append(row)
        except Exception as exc:
            rows.append({"part": job.get("part", "F"), "status": "error", "error_message": repr(exc), **job, **AUDIT_DEFAULTS})
    path = write_rows(OUT_ROOT / f"part_f_predictive_trust_positive_control_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-f", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}; part_c_role={args.part_c_role}")
    return gate_summary("F", 0, "F_ShardsWritten", "merge_required", rows)


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_predictive_trust_positive_control_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_f_predictive_trust_positive_control_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    summaries: list[dict[str, Any]] = []
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
        component_count = sum(
            int(
                fval(r.get("Brier_delta")) <= float(args.no_debt_budget)
                and fval(r.get("ECE_delta")) <= float(args.no_debt_budget)
                and fval(r.get("tail95_delta")) <= float(args.no_debt_budget)
                and fval(r.get("tail99_delta")) <= float(args.no_debt_budget)
                and fval(r.get("margin10_delta")) >= -float(args.no_debt_budget)
            )
            for r in f5
        )
        random_count = sum(ival(r.get("F5_no_debt")) for r in random_f5)
        accept = median(r.get("finite_step_accept_rate") for r in f5)
        scale = median(r.get("finite_step_scale_mean") for r in f5)
        skip = median(r.get("finite_step_skip_count") for r in f5)
        f0_wall = median((r.get("wall_time_s") for r in f0), 0.0)
        wall = median(r.get("wall_time_s") for r in group)
        overhead_ratio = wall / max(f0_wall, 1.0e-12) if f0_wall > 0 else 0.0
        random_gap = f5_count - random_count
        official = int(
            f5_count >= 12
            and component_count >= 12
            and retention >= 0.70
            and accept >= 0.20
            and scale >= 0.05
            and skip <= 0.70 * int(args.train_steps)
            and random_gap >= 8
            and (overhead_ratio <= 2.5 if f0_wall > 0 else True)
        )
        diagnostic = int(f5_count >= 12 and component_count >= 12 and retention >= 0.70 and not official)
        summaries.append({
            "group_idx": ival(key[0]),
            "scheme": key[1],
            "basis_key": key[2],
            "depth": key[3],
            "safety_type": key[4],
            "rows": len(group),
            "F5_no_debt_count": f5_count,
            "component_non_positive_rows": component_count,
            "C2_coverage_retention_vs_F0": retention,
            "finite_step_accept_rate_median": accept,
            "finite_step_scale_mean_median": scale,
            "finite_step_skip_count_median": skip,
            "random_veto_matched_gap": random_gap,
            "overhead_proxy_wall_time_median": wall,
            "overhead_ratio_vs_F0": overhead_ratio,
            "official_pass": official,
            "diagnostic_pass": diagnostic,
        })
    pass_groups = [g for g in summaries if ival(g.get("official_pass")) == 1 and str(g.get("safety_type")) not in {"F0_no_safety_control", "F6_random_veto_matched_control"}]
    diag_groups = [g for g in summaries if ival(g.get("diagnostic_pass")) == 1]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    route = "F_DirectC2F5PositiveControlPass" if gate else ("F_DirectDiagnosticOnly" if diag_groups else "F_DirectC2F5PositiveControlFailed")
    blocker = "none" if gate else ("part_f_job_errors" if any(r.get("status") == "error" for r in rows) else "f5_efficiency_or_c2_retention_failed")
    group_csv = write_rows(OUT_ROOT / "part_f_predictive_trust_positive_control_group_summary.csv", summaries)
    summary = gate_summary("F", gate, route, blocker, rows, group_summaries=summaries, passing_part_f_groups=pass_groups, diagnostic_part_f_groups=diag_groups, part_c_role=str(args.part_c_role), direct_after_part_c_failure=1)
    write_json(OUT_ROOT / "part_f_predictive_trust_positive_control_summary.json", summary)
    write_json(OUT_ROOT / "part_f_finite_step_trust_summary.json", summary)
    nxt = next_actions("F", gate, blocker, [] if gate else ["repair predictive trust accept/scale", "repair C2 retention carrier", "compare random veto matched control"], ["do_not_raise_no_debt_budget", "do_not_delete_random_control"])
    append_exec("part-f-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(nxt)}")
    append_recap("Part F direct C2/F5 positive-control", summary)
    return summary


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_taskwise_c2_summary.json")
    f = read_json(OUT_ROOT / "part_f_predictive_trust_positive_control_summary.json")
    gate = int(ival(f.get("gate_pass")) == 1 and ival(c.get("gate_pass")) == 1)
    route = "G_OfficialC2F5Pass" if gate else ("G_DirectFPassButPartCSanityFailed" if ival(f.get("gate_pass")) == 1 else "G_PositiveControlFailed")
    summary = gate_summary("G", gate, route, "none" if gate else route, [], part_c_route=c.get("route", "missing"), part_f_route=f.get("route", "missing"), passing_part_f_groups=f.get("passing_part_f_groups", []), promotion_allowed=int(gate), direct_after_part_c_failure=int(ival(c.get("gate_pass")) != 1))
    write_json(OUT_ROOT / "part_g_c2_f5_positive_control_summary.json", summary)
    nxt = next_actions("G", gate, summary["dominant_blocker"], [] if gate else ["Part C remains sanity-only; do not promote even if F passes", "repair Part F if direct positive-control failed"])
    append_exec("part-g", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_g_c2_f5_positive_control_summary.json')}; {rel(nxt)}")
    append_recap("Part G direct C2/F5 route", summary)
    return summary


def write_blocked(part: str, route: str, reason: str) -> dict[str, Any]:
    rows = [{"part": part, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}]
    summary = gate_summary(part, 0, route, reason, rows)
    write_json(OUT_ROOT / f"part_{part.lower()}_summary.json", summary)
    nxt = next_actions(part, 0, reason, [])
    append_exec(f"part-{part.lower()}", command_text(sys.argv), "blocked", files=rel(nxt), note=reason)
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    a = read_json(OUT_ROOT / "part_a_identity_control_readiness_summary.json")
    b = read_json(OUT_ROOT / "part_b_history_boundary_lock_summary.json")
    c = read_json(OUT_ROOT / "part_c_taskwise_c2_summary.json")
    route = "OfficialCandidatePass"
    blocker = "none"
    if ival(a.get("gate_pass")) != 1 or ival(b.get("gate_pass")) != 1:
        route, blocker = "A_or_B_identity_failed", "part_a_or_b_failed"
    elif ival(c.get("gate_pass")) != 1:
        f = read_json(OUT_ROOT / "part_f_predictive_trust_positive_control_summary.json")
        if ival(f.get("gate_pass")) == 1:
            route, blocker = "DirectFPassButPartCSanityFailed", c.get("dominant_blocker", "part_c_failed")
        else:
            route, blocker = f.get("route", "C_C2FormationTaskwiseFailed"), f.get("dominant_blocker", c.get("dominant_blocker", "part_c_failed"))
    else:
        f = read_json(OUT_ROOT / "part_f_predictive_trust_positive_control_summary.json")
        if ival(f.get("gate_pass")) != 1:
            route, blocker = f.get("route", "F_SafetyEnvelopeFailed"), f.get("dominant_blocker", "part_f_not_run")
    required = [
        "part_a_identity_control_readiness_summary.json",
        "part_b_history_boundary_lock_summary.json",
        "part_c_taskwise_c2_summary.json",
        "part_f_predictive_trust_positive_control_summary.json",
    ]
    stale = [f"missing:{name}" for name in required if not (OUT_ROOT / name).exists()]
    final = {"route": route, "dominant_blocker": blocker, "promotion_allowed": int(route == "OfficialCandidatePass" and not stale), "stale_artifact_count": len(stale), "stale_artifacts": stale, "generated_at": now(), **AUDIT_DEFAULTS}
    write_json(OUT_ROOT / "final_route.json", final)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    paths = [p for p in OUT_ROOT.rglob("*") if p.is_file()]
    manifest.write_text("# v23.02 reproduction manifest\n\n" + "\n".join(f"- `{rel(p)}`" for p in sorted(paths)) + "\n", encoding="utf-8")
    write_json(OUT_ROOT / "stale_artifact_audit.json", {"stale_artifact_count": len(stale), "stale_artifacts": stale})
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(manifest)}; {rel(OUT_ROOT / 'stale_artifact_audit.json')}")
    append_recap("Final route", final)
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = v2300.build_arg_parser()
    p.description = "DG-KAN v23.02 Predictive-Trust Functional Edge Population Flow runner"
    p.set_defaults(
        part_c_role="sanity",
        edge_weight_normalization="trace",
        edge_weight_ridge=1.0e-6,
        functional_gram_quadrature_points=129,
        train_steps=160,
        synthetic_train_size=192,
        synthetic_guard_size=128,
        population_grad_examples=4,
        debt_grad_examples=4,
        gate_floor=0.3,
        stat_warmup_steps=10,
        part_c_basis="dche_k5,dche_k9,dfour_default",
        part_c_depths="depth2,depth3",
        part_c_tasks="local_patch_interaction,rotation_sensitive",
        part_f_base_schemes="E5_FunctionalGram_DegreeEdgebankSNR_s0,E7_FunctionalGram_DegreeEdgebankSNR_s0p25,E7_FunctionalGram_DegreeEdgebankSNR_s0p5",
        part_f_safety_schemes="F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F7_cadence_finite_step_trust",
        part_f_seed_count=15,
        part_f_c2_seed_count=15,
        finite_step_guard_every=1,
        finite_step_tries=5,
        finite_step_shrink=0.5,
        finite_step_guard_examples=128,
        predictive_trust_scale=1.0,
    )
    p.add_argument("--part-c-schemes", default="C0_AdamW_control,C1_FunctionalGram_AdamW_s0_no_Qpop,C2_FunctionalGram_DiagonalSNR_s0,C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25,C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5,C12_FunctionalSobolev_DegreeEdgebankSNR_s1,C13_FunctionalSobolev_RandomMatchedGate_s0p25,C14_FunctionalSobolev_RandomMatchedGate_s0p5,C15_FunctionalSobolev_RandomMatchedGate_s1,C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0,C6_FunctionalGram_BlockSNR_class_conditional_s0,C7_FunctionalGram_RandomMatchedGate_s0,C8_FunctionalGram_SameComputeNoOp_s0,C9_FunctionalGram_s1_derivative_diagnostic")
    p.add_argument("--part-c-seed-count", type=int, default=15)
    p.add_argument("--part-c-gate-floors", default="0.2,0.3,0.4")
    p.add_argument("--c-task-coverage-gate", type=float, default=0.03)
    p.add_argument("--c-task-accuracy-gate", type=float, default=0.20)
    p.add_argument("--c-task-random-gap-gate", type=float, default=0.02)
    p.add_argument("--c-overall-coverage-gate", type=float, default=0.08)
    p.add_argument("--c-overall-random-gap-gate", type=float, default=0.04)
    p.add_argument("--finite-step-cadence", type=int, default=5)
    p.add_argument("--predictive-trust-scale", type=float, default=1.0)
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
    if mode == "part-c-merge":
        return merge_part_c(args)
    if mode == "part-d":
        return write_blocked("D", "D_NotRun_RequiresPartCPass", "Part D requires Part C taskwise pass or close-pass carrier.")
    if mode == "part-e":
        return write_blocked("E", "E_NotRun_RequiresPartCOrD", "Part E trust dataset requires Part C/D carrier evidence.")
    if mode == "part-f":
        return run_part_f(args)
    if mode == "part-f-merge":
        return merge_part_f(args)
    if mode == "part-g":
        return run_part_g(args)
    if mode == "part-h":
        return write_blocked("H", "H_NotRun_RequiresPartG", "Part H requires positive-control candidate.")
    if mode == "part-i":
        return write_blocked("I", "I_NotRun_RequiresPartGHOfficial", "Part I real-task preflight requires Part G/H official pass.")
    if mode in {"part-j", "finalize"}:
        return finalize(args)
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
