#!/usr/bin/env python3
"""DG-KAN v23.00R Curve-Geometry Population Flow runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import py_compile
import re
import sys
import time
import copy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
import experiments.run_v22_94_adamw_witness_decomposition_mpfu as v2294
import experiments.run_v22_96_solid_signal_channel_induced_metric_mpfu as v2296
from dgkan.fu.debt_conflict_veto import debt_conflict_veto_smoke_test
from dgkan.fu.edge_sobolev_metrics import (
    edge_sobolev_smoke_test,
    flatten_param_mode_weights,
    functional_edge_gram,
    functional_gram_condition,
    functional_gram_cost,
    functional_gram_inverse_retention,
    functional_gram_whitened_vector,
    inverse_weight_retention,
    make_mode_weights,
    matrix_mode_weights,
    mode_band_slices,
    mode_weight_summary,
)
from dgkan.fu.population_risk_gate import (
    block_snr_gate,
    cohort_stability_audit,
    diagonal_snr_gate,
    lowrank_ab_gate,
    population_risk_gate_smoke_test,
)
from dgkan.fu.signal_channel_estimators import per_example_gradient_matrix
from dgkan.optim.edge_sobolev_population_flow import EdgeSobolevAdamW, EdgeSobolevSNRFU, EdgeSobolevSNRFUWithDebtVeto, mark_kan_edge_params, optimizer_smoke_test


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.00R_CurveGeometryPopulationFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.00R_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.00R_实验结果复盘.md"
RESULT_ROOT = Path(os.environ.get("V2300R_RESULT_ROOT", str(ROOT / "results/v23_00R"))).resolve()
OUT_ROOT = Path(os.environ.get("V2300R_OUT_ROOT", str(RESULT_ROOT))).resolve()

BASIS_KEYS = ("dche_k5", "dche_k9", "dfour_default")
DEPTHS = ("depth2", "depth3")
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
    for path in (OUT_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def command_text(argv: Iterable[str]) -> str:
    return " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    clean = sorted(v for v in (fval(x, float("nan")) for x in values) if math.isfinite(v))
    if not clean:
        return float(default)
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    clean = [v for v in (fval(x, float("nan")) for x in values) if math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else float(default)


def pearson(xs: Iterable[Any], ys: Iterable[Any]) -> float:
    x = torch.tensor([fval(v, float("nan")) for v in xs], dtype=torch.float64)
    y = torch.tensor([fval(v, float("nan")) for v in ys], dtype=torch.float64)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[mask], y[mask]
    if int(x.numel()) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = x.norm().clamp_min(1.0e-12) * y.norm().clamp_min(1.0e-12)
    return float((x @ y / den).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64).cpu()
    bb = b.detach().reshape(-1).to(dtype=torch.float64).cpu()
    n = min(int(aa.numel()), int(bb.numel()))
    if n <= 0:
        return 0.0
    den = aa[:n].norm().clamp_min(1.0e-12) * bb[:n].norm().clamp_min(1.0e-12)
    return float((aa[:n] @ bb[:n] / den).item())


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(str(args.device))
    return torch.device("cpu")


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys = sorted({k for row in enriched for k in row.keys()} or set(AUDIT_DEFAULTS.keys()))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.00R CurveGeometryPopulationFlow 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误；缺失写 missing，不补造。\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.00R CurveGeometryPopulationFlow 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n尚未 final。\n\n",
            encoding="utf-8",
        )


def append_exec(stage: str, command: str, status: str, *, files: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {stage} {status}\n\n")
        fh.write(f"Command: `{command}`\n\n")
        if files:
            fh.write(f"Files: {files}\n\n")
        if note:
            fh.write(f"Note: {note}\n\n")


def append_recap(title: str, data: dict[str, Any] | list[str]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        if isinstance(data, dict):
            fh.write("```json\n")
            fh.write(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
            fh.write("\n```\n")
        else:
            for item in data:
                fh.write(f"- {item}\n")


def write_next_actions(part: str, route: str, blocker: str, actions: list[str], forbidden: list[str] | None = None, evidence: dict[str, Any] | None = None, max_rounds: int = 2) -> Path:
    payload = {
        "part": part.upper(),
        "route": route,
        "gate_pass": int(blocker == "none"),
        "dominant_blocker": blocker,
        "evidence": evidence or {},
        "allowed_actions": actions,
        "forbidden_actions": forbidden
        or [
            "add new edge function",
            "delete random-label/MLP controls",
            "use held/test to induce metric/gate",
            "promote reduced probe",
            "select best row/seed/basis at runtime",
        ],
        "max_repair_rounds": int(max_rounds),
        "rerun_commands": [f"{PYTHON} {rel(RUNNER)} --mode part-{part.lower()} --device cuda:0"],
        "required_new_metrics": [],
    }
    return write_json(OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json", payload)


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


def basis_args(args: argparse.Namespace, basis_key: str) -> argparse.Namespace:
    return v2294.basis_args(args, basis_key)


def make_model(depth: str, input_dim: int, classes: int, seed: int, args: argparse.Namespace, device: torch.device) -> v2293.TrueDeepPureKAN:
    return v2294.make_model(depth, input_dim, classes, seed, args, device)


def visual_data(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2294.visual_data(task, seed, args, device)


def model_seed_for(basis_key: str, depth: str, task: str, seed: int) -> int:
    return v2294.model_seed_for(basis_key, depth, task, seed)


def static_scan(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|metric_winner)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|winner_update)\s*\("),
        "external_product_feature": re.compile(r"\b(make_external_product_feature|patch_product_feature)\s*\("),
        "static_topk_mode_selection": re.compile(r"\.topk\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            hits.append({"file": rel(path), "check": "missing_file", "match": "missing"})
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            m = pattern.search(text)
            if m:
                hits.append({"file": rel(path), "check": name, "match": m.group(0)})
    return int(not hits), hits


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    files = [
        RUNNER,
        ROOT / "dgkan/fu/edge_sobolev_metrics.py",
        ROOT / "dgkan/fu/population_risk_gate.py",
        ROOT / "dgkan/fu/debt_conflict_veto.py",
        ROOT / "dgkan/optim/edge_sobolev_population_flow.py",
    ]
    compile_errors: list[str] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}: {repr(exc)}")
    compile_pass = int(not compile_errors)
    import_pass = 1
    smoke = {
        **edge_sobolev_smoke_test(),
        **{f"pop_{k}": v for k, v in population_risk_gate_smoke_test().items()},
        **{f"debt_{k}": v for k, v in debt_conflict_veto_smoke_test().items()},
    }
    scan_pass, scan_hits = static_scan(files)
    rows: list[dict[str, Any]] = []
    for basis_key in BASIS_KEYS:
        bargs = basis_args(args, basis_key)
        for depth in DEPTHS:
            try:
                x = torch.randn(8, int(args.visual_side) * int(args.visual_side), device=device)
                y = torch.randint(0, int(args.num_classes), (8,), device=device)
                model = make_model(depth, int(x.shape[1]), int(args.num_classes), 230000 + len(rows), bargs, device)
                diag = optimizer_smoke_test(model, x, y, basis_key=basis_key)
                rows.append(
                    {
                        "part": "A",
                        "status": "ok",
                        "basis_key": basis_key,
                        "depth": depth,
                        "compile_pass": compile_pass,
                        "import_pass": import_pass,
                        "static_scan_pass": scan_pass,
                        "purekan_depth2_constructed": int(depth == "depth2"),
                        "purekan_depth3_constructed": int(depth == "depth3"),
                        "dche_k5_available": int(basis_key == "dche_k5"),
                        "dche_k9_available": int(basis_key == "dche_k9"),
                        "dfour_default_available": int(basis_key == "dfour_default"),
                        "new_edge_function_added": 0,
                        "mlp_stem_used": 0,
                        "mlp_readout_used": 0,
                        "external_product_feature_used": 0,
                        "runtime_selector_used": 0,
                        "metric_winner_selection_used": 0,
                        "candidate_update_selection_used": 0,
                        "held_test_usage": 0,
                        "used_fake_data_rows": 0,
                        **diag,
                    }
                )
            except Exception as exc:
                rows.append({"part": "A", "status": "error", "basis_key": basis_key, "depth": depth, "error_message": repr(exc), "compile_pass": compile_pass, "import_pass": import_pass, "static_scan_pass": scan_pass, **AUDIT_DEFAULTS})
    gate = int(
        compile_pass
        and import_pass
        and scan_pass
        and any(ival(r.get("purekan_depth2_constructed")) for r in rows)
        and any(ival(r.get("purekan_depth3_constructed")) for r in rows)
        and all(r.get("status") == "ok" for r in rows)
        and all(ival(r.get("changed_edge_coefficients")) == 1 for r in rows if r.get("status") == "ok")
        and all(ival(r.get("changed_mlp_tensors")) == 0 for r in rows if r.get("status") == "ok")
    )
    blocker = "none" if gate else ("compile_or_static_scan_failed" if not (compile_pass and scan_pass) else "purekan_identity_or_optimizer_smoke_failed")
    matrix = write_rows(OUT_ROOT / "part_a_identity_matrix.csv", rows)
    summary = gate_summary("A", gate, "A_Pass" if gate else "A_CodeIdentityFailed", blocker, rows, compile_errors=compile_errors, static_scan_hits=scan_hits, smoke=smoke)
    write_json(OUT_ROOT / "part_a_identity_summary.json", summary)
    next_path = write_next_actions("a", summary["route"], blocker, ["fix compile/import/static scan identity issue", "ensure optimizer receives only edge coefficient tensors"] if not gate else [], evidence={"matrix": rel(matrix), "summary": rel(OUT_ROOT / "part_a_identity_summary.json")})
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part A identity audit", summary)
    return summary


def read_field(path: Path, *keys: str) -> Any:
    data = read_json(path)
    for key in keys:
        if key in data:
            return data[key]
    return "missing"


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    fields: dict[str, tuple[Path, tuple[str, ...]]] = {
        "v22_90_part_e_pass": (ROOT / "results/v22_90/part_e_summary.json", ("part_e_gate_pass", "gate_pass")),
        "v22_90_part_f_route": (ROOT / "results/v22_90/part_f_summary.json", ("part_f_route", "route", "dominant_blocker")),
        "v22_90_no_debt": (ROOT / "results/v22_90/part_f_summary.json", ("no_debt", "no_debt_count")),
        "v22_90_visual_coverage": (ROOT / "results/v22_90/part_f_summary.json", ("visual_coverage", "visual_coverage_pass")),
        "v22_91_final_route": (ROOT / "results/v22_91/final_route.json", ("final_route", "route")),
        "v22_91_h20_debt_sign": (ROOT / "results/v22_91/part_c_summary.json", ("h20_debt_sign", "response_debt_sign_agreement")),
        "v22_91_h20_debt_R2": (ROOT / "results/v22_91/part_c_summary.json", ("h20_debt_R2", "debt_R2")),
        "v22_91_leave_family_debt_R2": (ROOT / "results/v22_91/part_c_summary.json", ("leave_family_debt_R2", "leave_dataset_family_debt_R2")),
        "v22_93_final_route": (ROOT / "results/v22_93/final_route.json", ("final_route", "route")),
        "v22_93_adamw_architecture_signal": (ROOT / "results/v22_93/part_c_architecture_capability_summary.json", ("c2_adamw_architecture_signal", "adamw_architecture_signal")),
        "v22_93_metric_c2_gain": (ROOT / "results/v22_93/part_c_architecture_capability_summary.json", ("visual_synthetic_coverage_improvement_median", "metric_c2_gain")),
        "v22_94_part_c_witness_pass": (ROOT / "results/v22_94/part_c_witness_summary.json", ("part_c_gate_pass", "gate_pass")),
        "v22_94_part_d_decomposition_pass": (ROOT / "results/v22_94/part_d_decomposition_summary.json", ("part_d_gate_pass", "gate_pass")),
        "v22_94_ordinary_part_e_pass": (ROOT / "results/v22_94/part_e_multi_scheme_summary.json", ("part_e_gate_pass", "gate_pass")),
        "v22_94_ordinary_part_f_F5_best": (ROOT / "results/v22_94/part_f_deep_positive_control_summary.json", ("best_f5_no_debt_rows", "f5_no_debt_rows")),
        "v22_94_tailsafe_full_route": (ROOT / "results/v22_94/part_f_deep_positive_control_summary.json", ("part_f_route", "route")),
        "v22_96_best_repair_seed_stability": (ROOT / "results/v22_96/current/part_c_signal_channel_summary.json", ("seed_stability", "strict_offdiag_seed_stability")),
        "v22_96_best_repair_negative_snr": (ROOT / "results/v22_96/current/part_c_signal_channel_summary.json", ("negative_snr", "random_label_offdiag_score")),
        "v22_97_final_route": (ROOT / "results/v22_97/current/final_route.json", ("route", "final_route")),
        "v22_97_coeff_retention_ratio": (ROOT / "results/v22_97/current/part_e_sourceguard_pullback_summary.json", ("coefficient_retention_ratio", "source_guard_common_retention")),
        "v22_98_final_route": (ROOT / "results/v22_98/current/final_route.json", ("route", "final_route")),
        "v22_98_strict_offdiag_seed_stability": (ROOT / "results/v22_98/current/part_c_estimator_summary.json", ("strict_offdiag_seed_stability",)),
    }
    rows: list[dict[str, Any]] = []
    out: dict[str, Any] = {}
    for field, (path, keys) in fields.items():
        value = read_field(path, *keys) if path.exists() else "missing"
        out[field] = value
        rows.append({"part": "B", "field": field, "value": value, "source_path": rel(path), "source_exists": int(path.exists()), "status": "ok" if path.exists() else "missing"})
    gate = int(int(a.get("gate_pass", 0)) == 1)
    blocker = "none" if gate else "part_a_failed"
    matrix = write_rows(OUT_ROOT / "part_b_history_sources.csv", rows)
    summary = gate_summary("B", gate, "B_HistoryLockPass" if gate else "B_HistoryLockFailed", blocker, rows, missing_history_fields=sum(1 for r in rows if r["value"] == "missing"), **out)
    write_json(OUT_ROOT / "part_b_history_lock.json", summary)
    next_path = write_next_actions("b", summary["route"], blocker, ["rerun Part A before history lock"] if not gate else [], evidence={"sources": rel(matrix)})
    append_exec("part-b", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part B history lock", summary)
    return summary


def part_c_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = ROOT / "results/v22_94/part_c_witness_matrix.csv"
    rows = [r for r in read_rows(path) if r.get("optimizer_kind") == "adamw" and r.get("depth") in set(csv_items(args.part_c_depths)) and r.get("basis_key") in set(csv_items(args.part_c_basis))]
    tasks = set(csv_items(args.part_c_tasks))
    if tasks:
        rows = [r for r in rows if r.get("visual_synthetic_task") in tasks]
    return rows


def c_schemes(args: argparse.Namespace) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = [("raw_flat", "raw", 0.0)]
    for s in csv_items(args.sobolev_exponents):
        out.append((f"sobolev_s{s.replace('.', 'p')}", "sobolev", float(s)))
    for s in csv_items(args.functional_gram_exponents):
        out.append((f"functional_gram_s{s.replace('.', 'p')}", "functional_gram", float(s)))
    out.append(("derivative_grid_s1", "derivative", 1.0))
    out.append(("dataGram_diagnostic", "dataGram", 0.0))
    return out


def load_checkpoint_arrays(row: dict[str, Any]) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None, str, str]:
    js = [x for x in str(row.get("checkpoint_jsons", "")).split(";") if x]
    if not js:
        return None, None, "", ""
    final_json = ROOT / js[-1] if not Path(js[-1]).is_absolute() else Path(js[-1])
    prev_json = ROOT / js[-2] if len(js) > 1 and not Path(js[-2]).is_absolute() else (Path(js[-2]) if len(js) > 1 else final_json)
    if not final_json.exists():
        return None, None, rel(final_json), rel(prev_json)
    final_meta = read_json(final_json)
    prev_meta = read_json(prev_json) if prev_json.exists() else {}
    final_npz = ROOT / final_meta.get("npz", "") if not Path(str(final_meta.get("npz", ""))).is_absolute() else Path(str(final_meta.get("npz", "")))
    prev_npz = ROOT / prev_meta.get("npz", "") if prev_meta.get("npz") and not Path(str(prev_meta.get("npz"))).is_absolute() else Path(str(prev_meta.get("npz", "")))
    if not final_npz.exists():
        return None, None, rel(final_json), rel(prev_json)
    final_arr = dict(np.load(final_npz))
    prev_arr = dict(np.load(prev_npz)) if prev_npz.exists() else None
    return final_arr, prev_arr, rel(final_json), rel(prev_json)


def data_weights_for_row(row: dict[str, Any], arrays: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    basis_key = str(row.get("basis_key"))
    bargs = basis_args(args, basis_key)
    task = str(row.get("visual_synthetic_task"))
    seed = int(row.get("seed", 0))
    xtr, ytr, _xg, _yg = visual_data(task, seed, args, device)
    classes = int(ytr.max().detach().cpu().item()) + 1
    model_seed = int(row.get("model_seed", model_seed_for(basis_key, str(row.get("depth")), task, seed)))
    model = make_model(str(row.get("depth")), int(xtr.shape[1]), classes, model_seed, bargs, device)
    mats: list[torch.Tensor] = []
    layer_idx = 0
    while f"A_after_layer{layer_idx}" in arrays:
        mats.append(torch.from_numpy(arrays[f"A_after_layer{layer_idx}"]).to(device=device, dtype=torch.float64))
        layer_idx += 1
    v2294.set_model_mats(model, mats)
    weights: list[torch.Tensor] = []
    for idx, mat in enumerate(mats):
        c = v2294.data_c_matrix(model, xtr, idx).detach().to(dtype=torch.float64).cpu()
        diag = torch.diag(c).clamp_min(1.0e-8)
        diag = diag / diag.median().clamp_min(1.0e-8)
        weights.append(diag.reshape(-1, 1).expand(int(mat.shape[0]), int(mat.shape[1])).reshape(-1).contiguous())
    return weights


def mode_energy_fractions(delta: torch.Tensor, k: int, basis_key: str) -> tuple[float, float]:
    dd = delta.detach().to(dtype=torch.float64)
    if dd.ndim != 2:
        return 0.0, 0.0
    row_energy = dd.square().sum(dim=1)
    mode_energy = torch.zeros(k, dtype=torch.float64)
    for m in range(k):
        mode_energy[m] = row_energy[m::k].sum()
    total = mode_energy.sum().clamp_min(1.0e-12)
    bands = mode_band_slices(k, basis_key)
    low_idxs = next(iter(bands.values())) if bands else []
    high_idxs = list(bands.values())[-1] if bands else []
    low = float(mode_energy[torch.tensor(low_idxs, dtype=torch.long)].sum().div(total).item()) if low_idxs else 0.0
    high = float(mode_energy[torch.tensor(high_idxs, dtype=torch.long)].sum().div(total).item()) if high_idxs else 0.0
    return low, high


def run_part_c_row(row: dict[str, Any], scheme: tuple[str, str, float], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme_name, scheme_type, exponent = scheme
    start = time.time()
    try:
        arrays, prev_arrays, final_json, prev_json = load_checkpoint_arrays(row)
        if arrays is None:
            return {"part": "C", "status": "error", "metric_scheme": scheme_name, "basis_key": row.get("basis_key"), "depth": row.get("depth"), "seed": row.get("seed"), "error_message": "missing_checkpoint", "checkpoint_json": final_json, **AUDIT_DEFAULTS}
        data_weights = data_weights_for_row(row, arrays, args, device)
        basis_key = str(row.get("basis_key"))
        layer_metrics: list[dict[str, float]] = []
        all_delta: list[torch.Tensor] = []
        all_edge_w: list[torch.Tensor] = []
        all_data_w: list[torch.Tensor] = []
        all_prev_delta: list[torch.Tensor] = []
        dense_edge_costs: list[float] = []
        dense_edge_retentions: list[float] = []
        dense_edge_whites: list[torch.Tensor] = []
        dense_prev_edge_whites: list[torch.Tensor] = []
        layer_idx = 0
        while f"delta_layer{layer_idx}" in arrays:
            delta = torch.from_numpy(arrays[f"delta_layer{layer_idx}"]).to(dtype=torch.float64)
            k = 5 if basis_key == "dche_k5" else (9 if basis_key == "dche_k9" else int(args.dfour_k))
            if scheme_type == "raw":
                mode_w = torch.ones(k, dtype=torch.float64)
            elif scheme_type == "dataGram":
                mode_w = torch.ones(k, dtype=torch.float64)
            elif scheme_type == "functional_gram":
                gram = functional_edge_gram(
                    basis_key,
                    k,
                    sobolev_order=float(exponent),
                    quadrature_points=int(args.functional_gram_quadrature_points),
                    normalization=str(args.edge_weight_normalization),
                    ridge=float(args.edge_weight_ridge) if float(args.edge_weight_ridge) > 0.0 else 1.0e-6,
                )
                mode_w = torch.diag(gram).to(dtype=torch.float64)
            elif scheme_type == "derivative":
                mode_w = make_mode_weights(basis_key, k, exponent=1.0, normalization=str(args.edge_weight_normalization), ridge=float(args.edge_weight_ridge))
            else:
                mode_w = make_mode_weights(basis_key, k, exponent=float(exponent), normalization=str(args.edge_weight_normalization), ridge=float(args.edge_weight_ridge))
            edge_w = matrix_mode_weights(delta.shape, mode_w, mode_axis_period=k)
            data_w = data_weights[layer_idx]
            if scheme_type == "dataGram":
                edge_w = data_w
            if scheme_type == "functional_gram":
                dense_edge_costs.append(functional_gram_cost(delta, gram, mode_axis_period=k))
                dense_edge_retentions.append(functional_gram_inverse_retention(delta, gram, mode_axis_period=k))
                dense_edge_whites.append(functional_gram_whitened_vector(delta, gram, mode_axis_period=k))
            low, high = mode_energy_fractions(delta, k, basis_key)
            all_delta.append(delta.reshape(-1))
            all_edge_w.append(edge_w)
            all_data_w.append(data_w)
            if prev_arrays is not None and f"delta_layer{layer_idx}" in prev_arrays:
                prev_delta = torch.from_numpy(prev_arrays[f"delta_layer{layer_idx}"]).to(dtype=torch.float64)
                all_prev_delta.append(prev_delta.reshape(-1))
                if scheme_type == "functional_gram":
                    dense_prev_edge_whites.append(functional_gram_whitened_vector(prev_delta, gram, mode_axis_period=k))
            layer_metrics.append(
                {
                    "layer_idx": float(layer_idx),
                    "low_mode_energy_fraction": low,
                    "high_mode_energy_fraction": high,
                    "edge_condition": functional_gram_condition(gram) if scheme_type == "functional_gram" else float((edge_w.max() / edge_w.min().clamp_min(1.0e-12)).item()),
                }
            )
            layer_idx += 1
        delta_vec = torch.cat(all_delta) if all_delta else torch.zeros(0, dtype=torch.float64)
        edge_w_vec = torch.cat(all_edge_w) if all_edge_w else torch.ones(0, dtype=torch.float64)
        data_w_vec = torch.cat(all_data_w) if all_data_w else torch.ones(0, dtype=torch.float64)
        prev_vec = torch.cat(all_prev_delta) if all_prev_delta else delta_vec
        edge_ret = mean(dense_edge_retentions) if scheme_type == "functional_gram" else inverse_weight_retention(delta_vec, edge_w_vec)
        data_ret = inverse_weight_retention(delta_vec, data_w_vec)
        edge_cost = float(sum(dense_edge_costs)) if scheme_type == "functional_gram" else float((delta_vec.square() * edge_w_vec).sum().item())
        data_cost = float((delta_vec.square() * data_w_vec).sum().item())
        edge_white = torch.cat(dense_edge_whites) if scheme_type == "functional_gram" and dense_edge_whites else delta_vec / edge_w_vec.sqrt().clamp_min(1.0e-12)
        prev_edge_white = torch.cat(dense_prev_edge_whites) if scheme_type == "functional_gram" and dense_prev_edge_whites else prev_vec[: int(delta_vec.numel())] / edge_w_vec[: int(delta_vec.numel())].sqrt().clamp_min(1.0e-12)
        mw_summary = mode_weight_summary(edge_w_vec, basis_key, exponent, str(args.edge_weight_normalization))
        return {
            "part": "C",
            "status": "ok",
            "row_id": f"C_{scheme_name}_{row.get('run_id')}",
            "source_v22_94_run_id": row.get("run_id"),
            "metric_scheme": scheme_name,
            "metric_scheme_type": scheme_type,
            "basis_key": basis_key,
            "depth": row.get("depth"),
            "visual_synthetic_task": row.get("visual_synthetic_task"),
            "seed": row.get("seed"),
            "sobolev_s": exponent,
            "G_edge_condition_number": mw_summary.mode_weight_condition,
            "G_edge_min_weight": mw_summary.mode_weight_min,
            "G_edge_max_weight": mw_summary.mode_weight_max,
            "witness_retention_under_G_edge": edge_ret,
            "witness_retention_under_dataGram": data_ret,
            "witness_retention_gap_vs_dataGram": edge_ret - data_ret,
            "G_edge_cost": edge_cost,
            "dataGram_cost": data_cost,
            "C2_gain": fval(row.get("visual_coverage_improvement")),
            "C2_accuracy_gain": fval(row.get("visual_accuracy_improvement")),
            "source_guard_witness_cost_cosine": cosine(edge_white, prev_edge_white),
            "low_mode_energy_fraction": mean([m["low_mode_energy_fraction"] for m in layer_metrics]),
            "high_mode_energy_fraction": mean([m["high_mode_energy_fraction"] for m in layer_metrics]),
            "mode_energy_drift_by_degree_or_frequency": json.dumps({str(int(m["layer_idx"])): {"low": m["low_mode_energy_fraction"], "high": m["high_mode_energy_fraction"]} for m in layer_metrics}, sort_keys=True),
            "checkpoint_json": final_json,
            "previous_checkpoint_json": prev_json,
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {"part": "C", "status": "error", "row_id": f"C_{scheme_name}_{row.get('run_id')}", "metric_scheme": scheme_name, "basis_key": row.get("basis_key"), "depth": row.get("depth"), "seed": row.get("seed"), "error_message": repr(exc), "wall_time_s": time.time() - start, **AUDIT_DEFAULTS}


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    path = OUT_ROOT / f"part_c_metric_sanity_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    if int(a.get("gate_pass", 0)) != 1 or int(b.get("gate_pass", 0)) != 1:
        rows = [{"part": "C", "status": "blocked", "blocked_reason": "Part A/B failed", **AUDIT_DEFAULTS}]
        write_rows(path, rows)
        append_exec("part-c", command_text(sys.argv), "blocked", files=rel(path))
        return gate_summary("C", 0, "C_BlockedByPrerequisite", "part_a_or_b_failed", rows)
    device = device_from_args(args)
    jobs = [(r, s) for r in part_c_source_rows(args) for s in c_schemes(args)]
    rows = [run_part_c_row(row, scheme, args, device) for row, scheme in shard_items(jobs, args)]
    write_rows(path, rows)
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
    return gate_summary("C", 0, "C_ShardsWritten", "merge_required", rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_metric_sanity_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_c_metric_sanity_matrix.csv", rows)
    write_rows(OUT_ROOT / "part_c_edge_sobolev_metric_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    scheme_summaries: list[dict[str, Any]] = []
    for scheme in sorted({r.get("metric_scheme") for r in ok}):
        sr = [r for r in ok if r.get("metric_scheme") == scheme]
        edge_corr = pearson([r.get("G_edge_cost") for r in sr], [r.get("C2_gain") for r in sr])
        data_corr = pearson([r.get("dataGram_cost") for r in sr], [r.get("C2_gain") for r in sr])
        med_ret = median([r.get("witness_retention_under_G_edge") for r in sr])
        med_data = median([r.get("witness_retention_under_dataGram") for r in sr])
        med_cond = median([r.get("G_edge_condition_number") for r in sr])
        med_sg = median([r.get("source_guard_witness_cost_cosine") for r in sr])
        scheme_type = sr[0].get("metric_scheme_type", "") if sr else ""
        scheme_pass = int(
            scheme_type in {"sobolev", "derivative", "functional_gram"}
            and med_cond <= 1.0e4
            and med_ret >= med_data + 0.10
            and edge_corr >= data_corr
            and med_sg >= 0.50
            and math.isfinite(median([r.get("high_mode_energy_fraction") for r in sr], float("nan")))
            and math.isfinite(median([r.get("low_mode_energy_fraction") for r in sr], float("nan")))
        )
        scheme_summaries.append(
            {
                "metric_scheme": scheme,
                "metric_scheme_type": scheme_type,
                "rows": len(sr),
                "median_G_edge_condition_number": med_cond,
                "median_witness_retention_under_G_edge": med_ret,
                "median_witness_retention_under_dataGram": med_data,
                "median_witness_retention_gap_vs_dataGram": med_ret - med_data,
                "C2_gain_vs_G_edge_cost_corr": edge_corr,
                "C2_gain_vs_dataGram_cost_corr": data_corr,
                "source_guard_witness_cost_cosine": med_sg,
                "high_mode_energy_fraction": median([r.get("high_mode_energy_fraction") for r in sr]),
                "low_mode_energy_fraction": median([r.get("low_mode_energy_fraction") for r in sr]),
                "scheme_pass": scheme_pass,
            }
        )
    data_med = max([fval(s["median_witness_retention_under_G_edge"]) for s in scheme_summaries if s["metric_scheme_type"] == "dataGram"] or [0.0])
    pass_schemes = [s for s in scheme_summaries if ival(s.get("scheme_pass")) == 1 and data_med <= fval(s.get("median_witness_retention_under_G_edge")) + 0.10]
    gate = int(bool(pass_schemes) and not any(r.get("status") == "error" for r in rows))
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_c_job_errors"
    elif not pass_schemes:
        blocker = "edge_sobolev_retention_or_cost_gate_failed"
    else:
        blocker = "none"
    summary = gate_summary(
        "C",
        gate,
        "C_EdgeSobolevMetricPass" if gate else "C_EdgeSobolevMetricFailed",
        blocker,
        rows,
        scheme_summaries=scheme_summaries,
        passing_metric_schemes=[s["metric_scheme"] for s in pass_schemes],
        dataGram_best_retention=data_med,
    )
    write_json(OUT_ROOT / "part_c_metric_sanity_summary.json", summary)
    write_json(OUT_ROOT / "part_c_edge_sobolev_metric_summary.json", summary)
    next_path = write_next_actions(
        "c",
        summary["route"],
        blocker,
        ["try Sobolev exponent normalization/ridge/trace normalization", "try derivative-sample metric", "write metric_role_conflict_report if dataGram dominates"] if not gate else [],
        evidence={"matrix": rel(matrix), "passing_metric_schemes": summary["passing_metric_schemes"]},
    )
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part C edge-Sobolev metric sanity", summary)
    return summary


def passing_c_schemes() -> list[str]:
    c = read_json(OUT_ROOT / "part_c_metric_sanity_summary.json")
    schemes = c.get("passing_metric_schemes", [])
    return [str(s) for s in schemes] if isinstance(schemes, list) else []


def part_c_allows_continuation(args: argparse.Namespace) -> tuple[bool, int, str]:
    c = read_json(OUT_ROOT / "part_c_metric_sanity_summary.json")
    c_pass = int(c.get("gate_pass", 0)) == 1
    if c_pass:
        return True, 0, "part_c_pass"
    if str(getattr(args, "part_c_role", "hard")) == "sanity":
        return True, 1, str(c.get("dominant_blocker", "part_c_failed"))
    return False, 0, str(c.get("dominant_blocker", "part_c_failed"))


def scheme_sobolev_exponent(scheme: str) -> float:
    text = str(scheme)
    match = re.search(r"s([0-9]+)(?:p([0-9]+))?", text)
    if not match:
        return 0.0
    whole = match.group(1)
    frac = match.group(2) or ""
    return float(f"{whole}.{frac}") if frac else float(whole)


def continuation_metric_schemes(args: argparse.Namespace) -> list[str]:
    schemes = passing_c_schemes()
    if schemes:
        return schemes
    return csv_items(getattr(args, "continuation_metric_schemes", "sobolev_s0p0"))


def weights_for_model_params(model: v2293.TrueDeepPureKAN, basis_key: str, scheme: str, args: argparse.Namespace) -> torch.Tensor:
    weights: list[torch.Tensor] = []
    exponent = scheme_sobolev_exponent(scheme)
    if "derivative" in scheme:
        exponent = 1.0
    for p in model.coeffs:
        k = int(p.shape[-1])
        mw = make_mode_weights(basis_key, k, exponent=exponent, normalization=str(args.edge_weight_normalization), ridge=float(args.edge_weight_ridge), device=p.device, dtype=torch.float64)
        weights.append(flatten_param_mode_weights(p, mw, mode_axis=-1).cpu())
    return torch.cat(weights).to(dtype=torch.float64)


def d_blocks_for_model(model: v2293.TrueDeepPureKAN, basis_key: str) -> list[list[int]]:
    blocks: list[list[int]] = []
    offset = 0
    for p in model.coeffs:
        flat_n = int(p.numel())
        k = int(p.shape[-1])
        bands = mode_band_slices(k, basis_key)
        for idxs in bands.values():
            block: list[int] = []
            for flat_idx in range(flat_n):
                if flat_idx % k in idxs:
                    block.append(offset + flat_idx)
            if block:
                blocks.append(block)
        offset += flat_n
    return blocks


def historical_c2_gain(basis_key: str, depth: str, task: str, seed: int) -> float:
    rows = read_rows(ROOT / "results/v22_94/part_c_witness_matrix.csv")
    vals = [
        fval(r.get("visual_coverage_improvement"))
        for r in rows
        if r.get("optimizer_kind") == "adamw"
        and r.get("basis_key") == basis_key
        and r.get("depth") == depth
        and r.get("visual_synthetic_task") == task
        and str(r.get("seed")) == str(seed)
    ]
    return vals[0] if vals else 0.0


def run_part_d_row(job: tuple[str, str, str, int, str, str, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, basis_key, depth, seed, task, control, family = job
    start = time.time()
    try:
        bargs = basis_args(args, basis_key)
        xtr, ytr, _xg, _yg = v2296.controlled_data(task, control, seed, args, device)
        xsrc, ysrc, _xtraj, _ytraj, xwit, ywit, split = v2296.part_c_source_witness_split(xtr, ytr, control, seed, args)
        classes = int(max(ytr.max(), ywit.max()).detach().cpu().item()) + 1
        model = make_model(depth, int(xtr.shape[1]), classes, model_seed_for(basis_key, depth, task, seed), bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        g_source = per_example_gradient_matrix(model, xsrc, ysrc, max_examples=int(args.snr_examples), label_prior_correction=bool(int(args.label_prior_correction)))
        g_guard = per_example_gradient_matrix(model, xwit, ywit, max_examples=int(args.snr_examples), label_prior_correction=bool(int(args.label_prior_correction)))
        weights = weights_for_model_params(model, basis_key, scheme, args)
        n = min(int(g_source.shape[1]), int(weights.numel()))
        g_source = g_source[:, :n] / weights[:n].reshape(1, -1).sqrt().clamp_min(1.0e-8)
        g_guard = g_guard[:, :n] / weights[:n].reshape(1, -1).sqrt().clamp_min(1.0e-8)
        if family == "diagonal":
            src = diagonal_snr_gate(g_source, beta=float(args.snr_beta), use_log_snr=True)
            grd = diagonal_snr_gate(g_guard, beta=float(args.snr_beta), use_log_snr=True)
            gate_vec = src.gate
            guard_vec = grd.gate
            summary = src.summary
            update_mass = float((src.gate * src.mean.abs()).sum().item())
        elif family == "block":
            blocks = d_blocks_for_model(model, basis_key)
            src = block_snr_gate(g_source, blocks, beta=float(args.snr_beta))
            grd = block_snr_gate(g_guard, blocks, beta=float(args.snr_beta))
            gate_vec = src.gate
            guard_vec = grd.gate
            summary = src.summary
            update_mass = float((src.gate * src.mean.abs()).sum().item())
        elif family == "lowrank":
            src_lr = lowrank_ab_gate(g_source, rank=int(args.lowrank_rank))
            grd_lr = lowrank_ab_gate(g_guard, rank=int(args.lowrank_rank))
            op = src_lr["operator"]
            opg = grd_lr["operator"]
            gate_vec = torch.diag(op).clamp_min(0.0) if isinstance(op, torch.Tensor) and int(op.numel()) else torch.zeros(n, dtype=torch.float64)
            guard_vec = torch.diag(opg).clamp_min(0.0) if isinstance(opg, torch.Tensor) and int(opg.numel()) else torch.zeros(n, dtype=torch.float64)
            summary = src_lr["summary"] if isinstance(src_lr["summary"], dict) else {}
            update_mass = float(gate_vec.sum().item())
        else:
            cohorts = torch.chunk(g_source, max(2, int(args.cohort_count)), dim=0)
            gates = [diagonal_snr_gate(c, beta=float(args.snr_beta)).gate for c in cohorts if int(c.shape[0]) >= 2]
            audit = cohort_stability_audit(gates)
            src = diagonal_snr_gate(g_source, beta=float(args.snr_beta), use_log_snr=True)
            grd = diagonal_snr_gate(g_guard, beta=float(args.snr_beta), use_log_snr=True)
            gate_vec = src.gate
            guard_vec = grd.gate
            summary = {**src.summary, **audit, "gate_family": "cohort"}
            update_mass = float((src.gate * src.mean.abs()).sum().item())
        row = {
            "part": "D",
            "status": "ok",
            "row_id": f"D_{scheme}_{basis_key}_{depth}_{task}_s{seed}_{control}_{family}",
            "part_c_role": str(getattr(args, "part_c_role", "hard")),
            "forced_after_c_fail": part_c_allows_continuation(args)[1],
            "metric_scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "seed": int(seed),
            "visual_synthetic_task": task,
            "control": control,
            "gate_family": family,
            "snr_beta": float(args.snr_beta),
            "gate_density_mean": summary.get("gate_density_mean", summary.get("block_gate_density_mean", 0.0)),
            "snr_top_decile": summary.get("snr_top_decile", summary.get("block_snr_top_decile", 0.0)),
            "update_mass": update_mass,
            "snr_source_guard_cosine": cosine(gate_vec, guard_vec),
            "historical_C2_gain": historical_c2_gain(basis_key, depth, task, seed),
            "per_example_grad_mode": "microbatch_true",
            "per_example_count": int(g_source.shape[0]),
            "source_witness_disjoint": split.get("source_witness_disjoint", 0),
            "wall_time_s": time.time() - start,
            **{k: v for k, v in summary.items() if isinstance(v, (int, float, str))},
            **AUDIT_DEFAULTS,
        }
        return row
    except Exception as exc:
        return {"part": "D", "status": "error", "row_id": f"D_{scheme}_{basis_key}_{depth}_{task}_s{seed}_{control}_{family}", "metric_scheme": scheme, "basis_key": basis_key, "depth": depth, "seed": int(seed), "control": control, "gate_family": family, "error_message": repr(exc), "wall_time_s": time.time() - start, **AUDIT_DEFAULTS}


def part_d_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, int, str, str, str]]:
    schemes = continuation_metric_schemes(args)
    return [
        (scheme, basis, depth, seed, task, control, family)
        for scheme in schemes
        for basis in csv_items(args.part_d_basis)
        for depth in csv_items(args.part_d_depths)
        for seed in range(int(args.part_d_seed_count))
        for task in csv_items(args.part_d_tasks)
        for control in csv_items(args.part_d_controls)
        for family in csv_items(args.part_d_gate_families)
    ]


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    _c = read_json(OUT_ROOT / "part_c_metric_sanity_summary.json")
    path = OUT_ROOT / f"part_d_population_gate_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    c_continue, forced_after_c_fail, c_reason = part_c_allows_continuation(args)
    if not c_continue:
        rows = [{"part": "D", "status": "blocked", "blocked_reason": c_reason, **AUDIT_DEFAULTS}]
        write_rows(path, rows)
        write_blocked_summary("D", "D_BlockedByPartC", c_reason, "Part C edge geometry did not pass and --part-c-role was hard.")
        append_exec("part-d", command_text(sys.argv), "blocked", files=rel(path))
        return gate_summary("D", 0, "D_BlockedByPartC", c_reason, rows)
    device = device_from_args(args)
    rows = [run_part_d_row(job, args, device) for job in shard_items(part_d_jobs(args), args)]
    write_rows(path, rows)
    note = f"rows={len(rows)}; part_c_role={getattr(args, 'part_c_role', 'hard')}; forced_after_c_fail={forced_after_c_fail}; c_reason={c_reason}"
    append_exec("part-d", command_text(sys.argv), "shard-written", files=rel(path), note=note)
    return gate_summary("D", 0, "D_ShardsWritten", "merge_required", rows, part_c_role=str(getattr(args, "part_c_role", "hard")), forced_after_c_fail=forced_after_c_fail, part_c_reason=c_reason)


def seed_stability(rows: list[dict[str, Any]]) -> float:
    vals: list[float] = []
    for key in sorted({(r.get("metric_scheme"), r.get("basis_key"), r.get("depth"), r.get("visual_synthetic_task"), r.get("control"), r.get("gate_family")) for r in rows}):
        group = [r for r in rows if (r.get("metric_scheme"), r.get("basis_key"), r.get("depth"), r.get("visual_synthetic_task"), r.get("control"), r.get("gate_family")) == key and r.get("status") == "ok"]
        by_seed = {str(r.get("seed")): fval(r.get("update_mass")) for r in group}
        seeds = sorted(by_seed)
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1 :]:
                a, b = by_seed[s1], by_seed[s2]
                vals.append(1.0 - abs(a - b) / max(abs(a) + abs(b), 1.0e-12))
    return median(vals)


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_d_population_gate_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_d_population_gate_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    family_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("metric_scheme"), r.get("gate_family")) for r in ok}):
        group = [r for r in ok if (r.get("metric_scheme"), r.get("gate_family")) == key]
        c2 = [r for r in group if r.get("control") == "c2_positive"]
        rnd = [r for r in group if r.get("control") == "random_label"]
        mlp = [r for r in group if r.get("control") == "mlp_friendly_negative"]
        density = median([r.get("gate_density_mean") for r in c2])
        stability = seed_stability(c2)
        sg_cos = median([r.get("snr_source_guard_cosine") for r in c2])
        corr = pearson([r.get("update_mass") for r in c2], [r.get("historical_C2_gain") for r in c2])
        rnd_mass = median([r.get("update_mass") for r in rnd])
        c2_mass = median([r.get("update_mass") for r in c2])
        mlp_mass = median([r.get("update_mass") for r in mlp])
        family = str(key[1])
        if family == "block":
            pass_flag = int(0.05 <= density <= 0.80 and stability >= 0.65 and sg_cos >= 0.55 and corr > 0.15 and rnd_mass <= 0.75 * c2_mass)
        elif family == "lowrank":
            pass_flag = int(stability >= 0.55 and sg_cos >= 0.50 and corr > 0.15 and rnd_mass <= 0.75 * c2_mass)
        elif family == "cohort":
            pass_flag = int(stability >= 0.60 and 0.05 <= density <= 0.80 and sg_cos >= 0.50)
        else:
            pass_flag = int(0.05 <= density <= 0.80 and stability >= 0.60 and sg_cos >= 0.50 and corr > 0.10 and rnd_mass <= 0.75 * c2_mass)
        family_summaries.append(
            {
                "metric_scheme": key[0],
                "gate_family": family,
                "rows": len(group),
                "gate_density_mean": density,
                "snr_seed_stability": stability,
                "snr_source_guard_cosine": sg_cos,
                "snr_C2_gain_correlation": corr,
                "C2_update_mass": c2_mass,
                "random_label_update_mass": rnd_mass,
                "MLP_friendly_update_mass": mlp_mass,
                "family_pass": pass_flag,
                "generic_learnability_warning": int(mlp_mass > c2_mass),
            }
        )
    pass_families = [s for s in family_summaries if ival(s.get("family_pass")) == 1]
    gate = int(bool(pass_families) and not any(r.get("status") == "error" for r in rows))
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_d_job_errors"
    elif not pass_families:
        blocker = "population_gate_generic_or_unstable"
    else:
        blocker = "none"
    summary = gate_summary(
        "D",
        gate,
        "D_PopulationGatePass" if gate else "D_PopulationGateFailed",
        blocker,
        rows,
        part_c_role=str(getattr(args, "part_c_role", "hard")),
        forced_after_c_fail=max([ival(r.get("forced_after_c_fail")) for r in rows] or [0]),
        family_summaries=family_summaries,
        passing_gate_families=pass_families,
    )
    write_json(OUT_ROOT / "part_d_population_gate_summary.json", summary)
    next_path = write_next_actions("d", summary["route"], blocker, ["try block SNR/cohort-stable SNR/variance floor/gradient centering/fresh cohort split"] if not gate else [], evidence={"matrix": rel(matrix), "passing_gate_families": pass_families})
    append_exec("part-d-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part D population gate sanity", summary)
    return summary


def write_blocked_summary(part: str, route: str, blocker: str, reason: str) -> dict[str, Any]:
    rows = [{"part": part, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}]
    lower = part.lower()
    name_map = {
        "D": "part_d_population_gate",
        "E": "part_e_optimizer_positive_control",
        "F": "part_f_safety_envelope",
        "G": "part_g_phase",
        "H": "part_h_real_task_preflight",
        "I": "part_i_full_c2_f5",
        "J": "part_j_real_task_preflight",
    }
    stem = name_map.get(part.upper(), f"part_{lower}")
    matrix = write_rows(OUT_ROOT / f"{stem}_matrix.csv", rows)
    summary = gate_summary(part, 0, route, blocker, rows, blocked_reason=reason)
    write_json(OUT_ROOT / f"{stem}_summary.json", summary)
    if part.upper() == "D":
        write_json(OUT_ROOT / "part_d_population_gate_summary.json", summary)
    if part.upper() == "E":
        write_json(OUT_ROOT / "part_e_optimizer_positive_control_summary.json", summary)
    if part.upper() == "F":
        write_rows(OUT_ROOT / "part_f_phase_matrix.csv", rows)
        write_json(OUT_ROOT / "part_f_phase_summary.json", summary)
        write_json(OUT_ROOT / "part_f_safety_envelope_summary.json", summary)
    if part.upper() == "G":
        write_rows(OUT_ROOT / "part_g_full_positive_control_matrix.csv", rows)
        write_json(OUT_ROOT / "part_g_full_positive_control_summary.json", summary)
    if part.upper() == "H":
        write_json(OUT_ROOT / "part_h_real_task_preflight_summary.json", summary)
    if part.upper() == "I":
        write_json(OUT_ROOT / "part_i_failure_diagnostic_summary.json", summary)
    next_path = write_next_actions(lower, route, blocker, [], evidence={"matrix": rel(matrix), "reason": reason})
    append_recap(f"Part {part} blocked", summary)
    return summary


def synthetic_data(teacher: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2294.synthetic_data(teacher, seed, args, device)


def coeff_vector(model: v2293.TrueDeepPureKAN) -> torch.Tensor:
    parts = [p.detach().reshape(-1).to(dtype=torch.float64).cpu() for p in model.coeffs]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)


def split_flat_grads(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    offset = 0
    for param in params:
        n = int(param.numel())
        out.append(flat[:, offset : offset + n].reshape(int(flat.shape[0]), *tuple(param.shape)).contiguous())
        offset += n
    return out


def observe_task_gradients(opt: EdgeSobolevSNRFU, model: v2293.TrueDeepPureKAN, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace) -> tuple[int, float, float, int, int]:
    flat = per_example_gradient_matrix(model, x, y, max_examples=int(args.population_grad_examples), label_prior_correction=bool(int(args.label_prior_correction)))
    nan_count = int(torch.isnan(flat).sum().item()) if int(flat.numel()) else 0
    inf_count = int(torch.isinf(flat).sum().item()) if int(flat.numel()) else 0
    norms = flat.norm(dim=1) if flat.ndim == 2 and int(flat.shape[0]) else torch.zeros(0, dtype=torch.float64)
    params = list(model.coeffs)
    for param, grads in zip(params, split_flat_grads(flat, params)):
        opt.observe_per_example_gradients(param, grads.to(device=param.device, dtype=param.dtype))
    return (
        int(flat.shape[0]) if flat.ndim == 2 else 0,
        float(norms.median().item()) if int(norms.numel()) else 0.0,
        float(torch.quantile(norms, 0.95).item()) if int(norms.numel()) else 0.0,
        nan_count,
        inf_count,
    )


def debt_component_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    component: str,
    baseline: dict[str, float] | None = None,
    budget: float = 0.0,
    *,
    proactive: bool = False,
) -> torch.Tensor:
    prob = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(y.long(), num_classes=int(logits.shape[1])).to(dtype=prob.dtype)
    true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1)
    if component == "Brier":
        return (prob - one_hot).square().sum(dim=1).mean()
    if component == "ECE":
        conf = prob.max(dim=1).values
        correct = (prob.argmax(dim=1) == y.long()).to(dtype=prob.dtype)
        return (conf - correct).square().mean()
    if component in {"tail95", "tail99"}:
        q = 0.95 if component == "tail95" else 0.99
        k = max(1, int(math.ceil(float(q) * int(true_prob.numel()))))
        sorted_risk = torch.sort(1.0 - true_prob).values
        tail = sorted_risk[min(k - 1, int(sorted_risk.numel()) - 1) :].mean()
        if bool(proactive):
            return tail
        target = float((baseline or {}).get(component, 0.0)) + float(budget)
        return F.relu(tail - target).square()
    if component == "margin10":
        true_logit = logits.float().gather(1, y.long().reshape(-1, 1)).reshape(-1)
        other_logits = logits.float().clone()
        other_logits.scatter_(1, y.long().reshape(-1, 1), float("-inf"))
        margin = true_logit - other_logits.max(dim=1).values
        k = max(1, int(math.ceil(0.10 * int(margin.numel()))))
        low_margin = torch.sort(margin).values[:k].mean()
        if bool(proactive):
            return -low_margin
        floor = float((baseline or {}).get(component, 0.0)) - float(budget)
        return F.relu(floor - low_margin).square()
    return logits.float().sum() * 0.0


def per_example_debt_gradient_matrix(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    component: str,
    *,
    max_examples: int,
    baseline: dict[str, float],
    budget: float,
    proactive: bool = False,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    params = list(getattr(model, "coeffs", list(model.parameters())))
    n = min(int(max_examples), int(x.shape[0]))
    for idx in range(n):
        logits = model(x[idx : idx + 1])
        loss = debt_component_loss(logits, y[idx : idx + 1], component, baseline=baseline, budget=budget, proactive=proactive)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).to(dtype=torch.float64) for g, p in zip(grads, params)]).cpu())
    return torch.stack(rows, dim=0).to(dtype=torch.float64) if rows else torch.zeros((0, 0), dtype=torch.float64)


def observe_debt_gradients(
    opt: EdgeSobolevSNRFUWithDebtVeto,
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
    baseline: dict[str, float],
    *,
    randomize_alignment: bool = False,
    proactive: bool = False,
) -> dict[str, float | int | str]:
    params = list(model.coeffs)
    components = csv_items(args.debt_veto_components)
    out: dict[str, float | int | str] = {"debt_veto_components": ",".join(components), "debt_veto_proactive": int(bool(proactive))}
    for component in components:
        flat = per_example_debt_gradient_matrix(
            model,
            x,
            y,
            component,
            max_examples=int(args.debt_grad_examples),
            baseline=baseline,
            budget=float(args.no_debt_budget),
            proactive=bool(proactive),
        )
        if randomize_alignment and flat.ndim == 2 and int(flat.shape[1]) > 0:
            perm = torch.randperm(int(flat.shape[1]))
            flat = flat[:, perm]
        for param, grads in zip(params, split_flat_grads(flat, params)):
            opt.observe_debt_per_example_gradients(component, param, grads.to(device=param.device, dtype=param.dtype))
        out[f"{component}_debt_grad_norm"] = float(flat.norm().item()) if int(flat.numel()) else 0.0
    return out


def data_op_drifts(model: v2293.TrueDeepPureKAN, before_mats: list[torch.Tensor], x: torch.Tensor, args: argparse.Namespace) -> tuple[float, float]:
    data_drifts: list[float] = []
    op_drifts: list[float] = []
    after_mats = v2294.param_mats(model)
    for layer_idx, (a0, a1) in enumerate(zip(before_mats, after_mats)):
        try:
            c_data = v2294.data_c_matrix(model, x, layer_idx)
            data_drifts.append(v2294.log_spectral_drift(v2294.composite_metric(a0, c_data), v2294.composite_metric(a1, c_data)))
            mode_diag = v2294.mode_metric_diag(model, layer_idx, args)
            op_drifts.append(v2294.log_spectral_drift(v2294.op_spectrum(a0, mode_diag), v2294.op_spectrum(a1, mode_diag)))
        except Exception:
            continue
    return max(data_drifts or [0.0]), max(op_drifts or [0.0])


def debt_deltas(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "Brier": float(metrics.get("brier", 0.0)) - float(baseline.get("brier", 0.0)),
        "ECE": float(metrics.get("ece", 0.0)) - float(baseline.get("ece", 0.0)),
        "tail95": float(metrics.get("tail95", 0.0)) - float(baseline.get("tail95", 0.0)),
        "tail99": float(metrics.get("tail99", 0.0)) - float(baseline.get("tail99", 0.0)),
        "margin10": float(metrics.get("margin10", 0.0)) - float(baseline.get("margin10", 0.0)),
    }


def debt_component_ok(component: str, delta: float, budget: float) -> bool:
    if str(component) == "margin10":
        return float(delta) >= -float(budget)
    return float(delta) <= float(budget)


def debt_component_slack(component: str, delta: float, budget: float) -> float:
    if str(component) == "margin10":
        return float(delta) + float(budget)
    return float(budget) - float(delta)


def snapshot_model_params(model: torch.nn.Module) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    return [(p, p.detach().clone()) for p in model.parameters()]


@torch.no_grad()
def restore_model_params(snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    for param, value in snapshot:
        param.copy_(value)


def clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_state_value(v) for v in value)
    return copy.deepcopy(value)


def snapshot_optimizer(opt: torch.optim.Optimizer) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "state": {p: {k: clone_state_value(v) for k, v in state.items()} for p, state in opt.state.items()},
        "group_scalars": [
            {k: clone_state_value(v) for k, v in group.items() if k != "params" and not isinstance(v, torch.Tensor)}
            for group in opt.param_groups
        ],
    }
    if hasattr(opt, "_observed_grads"):
        snap["observed_grads"] = clone_state_value(getattr(opt, "_observed_grads"))
    if hasattr(opt, "_observed_debt_grads"):
        snap["observed_debt_grads"] = clone_state_value(getattr(opt, "_observed_debt_grads"))
    return snap


def restore_optimizer(opt: torch.optim.Optimizer, snapshot: dict[str, Any]) -> None:
    opt.state.clear()
    for param, state in snapshot.get("state", {}).items():
        opt.state[param] = {k: clone_state_value(v) for k, v in state.items()}
    for group, scalars in zip(opt.param_groups, snapshot.get("group_scalars", [])):
        for key, value in scalars.items():
            group[key] = clone_state_value(value)
    if "observed_grads" in snapshot and hasattr(opt, "_observed_grads"):
        setattr(opt, "_observed_grads", clone_state_value(snapshot["observed_grads"]))
    if "observed_debt_grads" in snapshot and hasattr(opt, "_observed_debt_grads"):
        setattr(opt, "_observed_debt_grads", clone_state_value(snapshot["observed_debt_grads"]))


def finite_step_guarded_step(
    opt: torch.optim.Optimizer,
    model: v2293.TrueDeepPureKAN,
    baseline: dict[str, float],
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
    *,
    part: str,
    safety_type: str,
    step: int,
) -> dict[str, float | int | str]:
    safety_lower = str(safety_type).lower()
    predictive_enabled = "predictive" in safety_lower or "scaled_cadence" in safety_lower
    finite_enabled = "finite" in str(safety_type).lower() or predictive_enabled
    predictive_scale = max(0.0, min(1.0, fval(getattr(args, "predictive_trust_scale", 1.0), 1.0)))
    guard_n = min(int(args.finite_step_guard_examples), int(xg.shape[0]))
    x_eval = xg[:guard_n]
    y_eval = yg[:guard_n]
    components = csv_items(args.finite_step_components)
    budget = float(args.no_debt_budget)

    def state_slack_adjusted_scale(base_scale: float) -> tuple[float, str]:
        if "slack" not in safety_lower and "component_risk" not in safety_lower:
            return float(base_scale), "predictive_unchecked"
        metrics = v2293.metrics_for_model(model, x_eval, y_eval)
        deltas = debt_deltas(metrics, baseline)
        if not components:
            return float(base_scale), "predictive_slack_no_components"
        slack = min(debt_component_slack(component, float(deltas.get(component, 0.0)), float(budget)) for component in components)
        denom = max(abs(float(budget)), 1.0e-8)
        slack_ratio = max(0.0, min(1.0, slack / denom))
        return float(base_scale) * slack_ratio, f"predictive_slack_ratio={slack_ratio:.6g}"

    if not finite_enabled or int(args.finite_step_guard_every) <= 0 or int(step) % int(args.finite_step_guard_every) != 0:
        if finite_enabled and predictive_enabled:
            predictive_scale, reason = state_slack_adjusted_scale(predictive_scale)
            if predictive_scale <= 0.0:
                if hasattr(opt, "clear_observed_gradients"):
                    opt.clear_observed_gradients()
                return {
                    "finite_step_enabled": int(finite_enabled),
                    "finite_step_accept": 0,
                    "finite_step_skip": 1,
                    "finite_step_scale": 0.0,
                    "finite_step_attempts": 1,
                    "finite_step_reject_reason": "predictive_slack_zero",
                }
            base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]
            for group, lr in zip(opt.param_groups, base_lrs):
                group["lr"] = lr * predictive_scale
            opt.step()
            for group, lr in zip(opt.param_groups, base_lrs):
                group["lr"] = lr
            return {
                "finite_step_enabled": int(finite_enabled),
                "finite_step_accept": 1,
                "finite_step_skip": 0,
                "finite_step_scale": predictive_scale,
                "finite_step_attempts": 1,
                "finite_step_reject_reason": reason,
            }
        opt.step()
        return {
            "finite_step_enabled": int(finite_enabled),
            "finite_step_accept": 1,
            "finite_step_skip": 0,
            "finite_step_scale": 1.0,
            "finite_step_attempts": 1,
            "finite_step_reject_reason": "not_checked",
        }

    model_snapshot = snapshot_model_params(model)
    opt_snapshot = snapshot_optimizer(opt)
    base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]
    enforce_debt = str(part).upper() == "F5" or bool(int(args.finite_step_apply_all_parts))
    last_reason = "no_attempt"
    attempts = max(1, int(args.finite_step_tries))
    predictive_scale, predictive_reason = state_slack_adjusted_scale(predictive_scale)
    if predictive_enabled and predictive_scale <= 0.0:
        restore_model_params(model_snapshot)
        restore_optimizer(opt, opt_snapshot)
        if hasattr(opt, "clear_observed_gradients"):
            opt.clear_observed_gradients()
        return {
            "finite_step_enabled": 1,
            "finite_step_accept": 0,
            "finite_step_skip": 1,
            "finite_step_scale": 0.0,
            "finite_step_attempts": 1,
            "finite_step_reject_reason": "predictive_slack_zero",
        }
    for attempt in range(attempts):
        restore_model_params(model_snapshot)
        restore_optimizer(opt, opt_snapshot)
        scale = (predictive_scale if predictive_enabled else 1.0) * (float(args.finite_step_shrink) ** attempt)
        for group, lr in zip(opt.param_groups, base_lrs):
            group["lr"] = lr * scale
        opt.step()
        metrics = v2293.metrics_for_model(model, x_eval, y_eval)
        deltas = debt_deltas(metrics, baseline)
        debt_ok = all(debt_component_ok(component, float(deltas.get(component, 0.0)), budget) for component in components)
        c2_delta = float(metrics.get("coverage_CVaR25", 0.0)) - float(baseline.get("coverage_CVaR25", 0.0))
        c2_ok = str(part).upper() != "F3" or c2_delta >= float(args.finite_step_c2_floor)
        if (debt_ok or not enforce_debt) and c2_ok:
            for group, lr in zip(opt.param_groups, base_lrs):
                group["lr"] = lr
            return {
                "finite_step_enabled": 1,
                "finite_step_accept": 1,
                "finite_step_skip": 0,
                "finite_step_scale": scale,
                "finite_step_attempts": attempt + 1,
                "finite_step_reject_reason": "accepted",
            }
        bad_components = [component for component in components if not debt_component_ok(component, float(deltas.get(component, 0.0)), budget)]
        if not c2_ok:
            last_reason = "c2_floor"
        elif bad_components:
            last_reason = "debt_" + "|".join(bad_components)
        else:
            last_reason = predictive_reason if predictive_enabled else "unknown"
    restore_model_params(model_snapshot)
    restore_optimizer(opt, opt_snapshot)
    for group, lr in zip(opt.param_groups, base_lrs):
        group["lr"] = lr
    if hasattr(opt, "clear_observed_gradients"):
        opt.clear_observed_gradients()
    return {
        "finite_step_enabled": 1,
        "finite_step_accept": 0,
        "finite_step_skip": 1,
        "finite_step_scale": 0.0,
        "finite_step_attempts": attempts,
        "finite_step_reject_reason": last_reason,
    }


def parse_v23_scheme(scheme: str, *, safety_type: str = "") -> dict[str, Any]:
    text = str(scheme)
    lower = text.lower()
    if "random" in lower:
        gate_family = "random_matched"
    elif "degree" in lower and "edgebank" in lower:
        gate_family = "degree_edgebank"
    elif "edgebank" in lower or "edge_bank" in lower:
        gate_family = "edgebank"
    elif "block" in lower:
        gate_family = "block"
    else:
        gate_family = "diagonal"
    return {
        "optimizer_family": "adamw" if text == "E0_AdamW_control" or "adamw_control" in lower else "edge_sobolev",
        "use_population_gate": int("snr" in lower or "random" in lower),
        "gate_family": gate_family,
        "sobolev_exponent": scheme_sobolev_exponent(text),
        "edge_metric_type": "functional_gram" if "functionalgram" in lower or "functional_gram" in lower else "mode_diag",
        "use_debt_veto": int("veto" in str(safety_type).lower()),
    }


def make_v23_optimizer(model: v2293.TrueDeepPureKAN, basis_key: str, scheme: str, args: argparse.Namespace, *, safety_type: str = ""):
    mark_kan_edge_params(model, basis_key=basis_key)
    spec = parse_v23_scheme(scheme, safety_type=safety_type)
    if spec["optimizer_family"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(args.adamw_lr), weight_decay=float(args.weight_decay)), spec
    cls = EdgeSobolevSNRFUWithDebtVeto if spec["use_debt_veto"] else (EdgeSobolevSNRFU if spec["use_population_gate"] else EdgeSobolevAdamW)
    opt = cls(
        list(model.coeffs),
        lr=float(args.edge_lr),
        weight_decay=float(args.weight_decay),
        sobolev_exponent=float(spec["sobolev_exponent"]),
        edge_metric_type=str(spec["edge_metric_type"]),
        edge_weight_normalization=str(args.edge_weight_normalization),
        edge_weight_ridge=float(args.edge_weight_ridge),
        functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
        gate_beta=float(args.snr_beta),
        gate_family=str(spec["gate_family"]),
        gate_floor=float(args.gate_floor),
        stat_warmup_steps=int(args.stat_warmup_steps),
    )
    if spec["use_debt_veto"]:
        mode = "descent_negative" if ("corrected" in str(safety_type).lower() or "proactive" in str(safety_type).lower()) else "legacy_positive"
        for group in opt.param_groups:
            group["debt_veto_conflict_mode"] = mode
    return opt, spec


def train_v23_scheme(
    scheme: str,
    basis_key: str,
    depth: str,
    teacher_type: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    part: str,
    safety_type: str = "",
) -> dict[str, Any]:
    start = time.time()
    bargs = basis_args(args, basis_key)
    if teacher_type == "c2_visual_synthetic":
        xtr, ytr, xg, yg = visual_data(task, seed, args, device)
    elif teacher_type == "f5_no_debt_calibration":
        xtr, ytr, xg, yg = synthetic_data("c1a_shallow_additive", seed + 50000, args, device)
    else:
        xtr, ytr, xg, yg = synthetic_data(teacher_type, seed, args, device)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = 2305000 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + int(seed)
    model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
    opt, spec = make_v23_optimizer(model, basis_key, scheme, args, safety_type=safety_type)
    before = v2293.metrics_for_model(model, xg, yg)
    before_train = v2293.metrics_for_model(model, xtr, ytr)
    before_vec = coeff_vector(model)
    before_mats = v2294.param_mats(model)
    gate_trace: list[float] = []
    veto_trace: list[float] = []
    preserved_trace: list[float] = []
    grad_norm_median: list[float] = []
    grad_norm_p95: list[float] = []
    grad_nan_count = 0
    grad_inf_count = 0
    per_example_count = 0
    debt_diag: dict[str, float | int | str] = {}
    finite_accept_count = 0
    finite_skip_count = 0
    finite_attempt_count = 0
    finite_scale_trace: list[float] = []
    finite_reject_reasons: dict[str, int] = {}
    for step in range(1, int(args.train_steps) + 1):
        xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
        opt.zero_grad(set_to_none=True)
        if isinstance(opt, EdgeSobolevSNRFU):
            pe_count, gmed, gp95, nnan, ninf = observe_task_gradients(opt, model, xb, yb, args)
            per_example_count = max(per_example_count, pe_count)
            grad_norm_median.append(gmed)
            grad_norm_p95.append(gp95)
            grad_nan_count += nnan
            grad_inf_count += ninf
            if isinstance(opt, EdgeSobolevSNRFUWithDebtVeto):
                debt_diag = observe_debt_gradients(
                    opt,
                    model,
                    xb,
                    yb,
                    args,
                    before_train,
                    randomize_alignment="random" in str(safety_type).lower(),
                    proactive=("proactive" in str(safety_type).lower() or "corrected" in str(safety_type).lower()),
                )
        logits = model(xb).float()
        loss = F.cross_entropy(logits, yb.long())
        reg_weight = fval(getattr(args, "debt_regularization_weight", 0.0), 0.0)
        if reg_weight > 0.0:
            reg_components = csv_items(getattr(args, "debt_regularization_components", getattr(args, "debt_veto_components", "")))
            reg_terms = [
                debt_component_loss(logits, yb, component, baseline=before_train, budget=float(args.no_debt_budget), proactive=True)
                for component in reg_components
            ]
            if reg_terms:
                loss = loss + float(reg_weight) * torch.stack([term.reshape(()) for term in reg_terms]).mean()
        loss.backward()
        finite_diag = finite_step_guarded_step(opt, model, before, xg, yg, args, part=part, safety_type=safety_type, step=step)
        finite_accept_count += ival(finite_diag.get("finite_step_accept"))
        finite_skip_count += ival(finite_diag.get("finite_step_skip"))
        finite_attempt_count += ival(finite_diag.get("finite_step_attempts"))
        finite_scale_trace.append(fval(finite_diag.get("finite_step_scale"), 1.0))
        reason = str(finite_diag.get("finite_step_reject_reason", ""))
        if reason and reason not in {"accepted", "not_checked"} and not reason.startswith("predictive_unchecked") and not reason.startswith("predictive_slack_ratio"):
            finite_reject_reasons[reason] = finite_reject_reasons.get(reason, 0) + 1
        if hasattr(opt, "last_stats"):
            gate_trace.append(float(opt.last_stats.gate_density_mean))
            veto_trace.append(float(opt.last_stats.veto_density_mean))
            preserved_trace.append(float(opt.last_stats.preserved_task_energy_fraction))
    after = v2293.metrics_for_model(model, xg, yg)
    after_train = v2293.metrics_for_model(model, xtr, ytr)
    after_vec = coeff_vector(model)
    data_drift, op_drift = data_op_drifts(model, before_mats, xtr, bargs)
    component_deltas = {
        "Brier": after.get("brier", 0.0) - before.get("brier", 0.0),
        "ECE": after.get("ece", 0.0) - before.get("ece", 0.0),
        "tail95": after.get("tail95", 0.0) - before.get("tail95", 0.0),
        "tail99": after.get("tail99", 0.0) - before.get("tail99", 0.0),
        "margin10": after.get("margin10", 0.0) - before.get("margin10", 0.0),
    }
    no_debt = int(all(debt_component_ok(k, float(component_deltas[k]), float(args.no_debt_budget)) for k in component_deltas))
    return {
        "part": part,
        "status": "ok",
        "scheme": scheme,
        "safety_type": safety_type,
        "basis_key": basis_key,
        "depth": depth,
        "teacher_type": teacher_type,
        "visual_synthetic_task": task,
        "seed": int(seed),
        "train_steps": int(args.train_steps),
        "part_c_role": str(args.part_c_role),
        "forced_after_c_fail": part_c_allows_continuation(args)[1],
        "diagnostic_only": int(args.diagnostic_only),
        "G_edge_type": str(spec["edge_metric_type"]),
        "snr_gate_type": spec["gate_family"] if spec["use_population_gate"] else "none",
        "sobolev_s": spec["sobolev_exponent"],
        "stat_warmup_steps": int(args.stat_warmup_steps),
        "edge_lr": float(args.edge_lr),
        "adamw_lr": float(args.adamw_lr),
        "no_debt_budget": float(args.no_debt_budget),
        "debt_veto_conflict_mode": str(opt.param_groups[0].get("debt_veto_conflict_mode", "none")) if opt.param_groups else "none",
        "debt_regularization_weight": fval(getattr(args, "debt_regularization_weight", 0.0), 0.0),
        "debt_regularization_components": str(getattr(args, "debt_regularization_components", "")),
        "gate_floor": float(args.gate_floor),
        "finite_step_enabled": int("finite" in str(safety_type).lower() or "predictive" in str(safety_type).lower() or "scaled_cadence" in str(safety_type).lower()),
        "finite_step_accept_count": finite_accept_count,
        "finite_step_skip_count": finite_skip_count,
        "finite_step_attempt_count": finite_attempt_count,
        "finite_step_accept_rate": finite_accept_count / max(1, finite_accept_count + finite_skip_count),
        "finite_step_scale_mean": mean(finite_scale_trace, 1.0),
        "finite_step_reject_reasons": ";".join(f"{k}:{v}" for k, v in sorted(finite_reject_reasons.items())),
        "gate_density_mean": mean(gate_trace, 1.0 if not spec["use_population_gate"] else 0.0),
        "gate_density_median": median(gate_trace, 1.0 if not spec["use_population_gate"] else 0.0),
        "gate_density_p10": float(np.quantile(gate_trace, 0.10)) if gate_trace else 0.0,
        "gate_density_p90": float(np.quantile(gate_trace, 0.90)) if gate_trace else 0.0,
        "veto_density_mean": mean(veto_trace),
        "preserved_task_energy_fraction": mean(preserved_trace, 1.0),
        "C2_accuracy_initial": before["accuracy"],
        "C2_accuracy_final": after["accuracy"],
        "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
        "C2_coverage_initial": before["coverage_CVaR25"],
        "C2_coverage_final": after["coverage_CVaR25"],
        "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
        "visual_rows_ge_0p02": int(after["coverage_CVaR25"] - before["coverage_CVaR25"] >= 0.02),
        "guard_NLL_before": before["nll"],
        "guard_NLL_after": after["nll"],
        "guard_NLL_delta": after["nll"] - before["nll"],
        "guard_accuracy": after["accuracy"],
        "Brier_delta": component_deltas["Brier"],
        "ECE_delta": component_deltas["ECE"],
        "tail95_delta": component_deltas["tail95"],
        "tail99_delta": component_deltas["tail99"],
        "margin10_delta": component_deltas["margin10"],
        "no_debt_row": no_debt,
        "F5_no_debt": no_debt,
        "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
        "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
        "train_tail95_delta": after_train.get("tail95", 0.0) - before_train.get("tail95", 0.0),
        "train_tail99_delta": after_train.get("tail99", 0.0) - before_train.get("tail99", 0.0),
        "train_margin10_delta": after_train.get("margin10", 0.0) - before_train.get("margin10", 0.0),
        "mode_spectrum_drift": float((after_vec - before_vec).norm().div(before_vec.norm().clamp_min(1.0e-12)).item()) if int(before_vec.numel()) else 0.0,
        "dataGram_drift": data_drift,
        "operator_norm_drift": op_drift,
        "wall_time_s": time.time() - start,
        "per_example_grad_mode": "microbatch_true" if spec["use_population_gate"] else "not_used",
        "per_example_count": per_example_count,
        "cohort_count": 0,
        "cohort_size_min": 0,
        "cohort_size_max": 0,
        "grad_nan_count": grad_nan_count,
        "grad_inf_count": grad_inf_count,
        "grad_norm_median": median(grad_norm_median),
        "grad_norm_p95": median(grad_norm_p95),
        "new_edge_function_added": 0,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
        "external_product_feature_used": 0,
        **debt_diag,
        **v2294.basis_audit_fields(basis_key, model),
        **AUDIT_DEFAULTS,
    }


def part_e_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int]]:
    return [
        (scheme, basis, depth, task, seed)
        for scheme in csv_items(args.part_e_schemes)
        for basis in csv_items(args.part_e_basis)
        for depth in csv_items(args.part_e_depths)
        for task in csv_items(args.part_e_tasks)
        for seed in range(int(args.part_e_seed_count))
    ]


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    d = read_json(OUT_ROOT / "part_d_population_gate_summary.json")
    if int(d.get("gate_pass", 0)) != 1 and not int(args.force_part_e_after_d_fail):
        return write_blocked_summary("E", "E_BlockedByPartD", str(d.get("dominant_blocker", "part_d_failed")), "Part D population gate did not pass and --force-part-e-after-d-fail=0.")
    device = device_from_args(args)
    rows = [
        train_v23_scheme(scheme, basis, depth, "c2_visual_synthetic", task, seed, args, device, part="E")
        for scheme, basis, depth, task, seed in shard_items(part_e_jobs(args), args)
    ]
    path = OUT_ROOT / f"part_e_optimizer_positive_control_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-e", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}; force_after_d_fail={int(args.force_part_e_after_d_fail)}")
    return gate_summary("E", 0, "E_ShardsWritten", "merge_required", rows)


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_e_optimizer_positive_control_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    e0_times = {
        (r.get("basis_key"), r.get("depth"), r.get("visual_synthetic_task"), str(r.get("seed"))): fval(r.get("wall_time_s"), 1.0)
        for r in rows
        if str(r.get("scheme")) == "E0_AdamW_control" and r.get("status") == "ok"
    }
    for row in rows:
        key = (row.get("basis_key"), row.get("depth"), row.get("visual_synthetic_task"), str(row.get("seed")))
        row["overhead_ratio"] = fval(row.get("wall_time_s")) / max(e0_times.get(key, fval(row.get("wall_time_s"), 1.0)), 1.0e-12)
    matrix = write_rows(OUT_ROOT / "part_e_optimizer_positive_control_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth")) == key]
        scheme_text = str(key[0])
        scheme_lower = scheme_text.lower()
        no_gate_scheme = scheme_text == "E0_AdamW_control" or ("snr" not in scheme_lower and "random" not in scheme_lower)
        coverage_med = median([r.get("C2_coverage_improvement") for r in group])
        accuracy_med = median([r.get("C2_accuracy_improvement") for r in group])
        rows_ge = sum(ival(r.get("visual_rows_ge_0p02")) for r in group)
        density = median([r.get("gate_density_mean") for r in group], 1.0)
        overhead = median([r.get("overhead_ratio") for r in group], 1.0)
        gate = int(
            group
            and coverage_med >= float(args.c2_coverage_gate)
            and rows_ge >= math.ceil(float(args.c2_rows_ge_fraction) * len(group))
            and overhead <= float(args.max_overhead_ratio)
            and (no_gate_scheme or 0.05 <= density <= 0.80)
        )
        summaries.append(
            {
                "scheme": key[0],
                "basis_key": key[1],
                "depth": key[2],
                "rows": len(group),
                "C2_coverage_improvement_median": coverage_med,
                "C2_accuracy_improvement_median": accuracy_med,
                "visual_rows_ge_0p02": rows_ge,
                "gate_density_mean": density,
                "overhead_ratio_median": overhead,
                "scheme_gate_pass": gate,
            }
        )
    pass_groups = [s for s in summaries if ival(s.get("scheme_gate_pass")) == 1 and str(s.get("scheme")) != "E0_AdamW_control"]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    blocker = "none" if gate else ("part_e_job_errors" if any(r.get("status") == "error" for r in rows) else "c2_formation_or_gate_density_failed")
    summary = gate_summary(
        "E",
        gate,
        "E_C2FormationPass" if gate else "E_C2FormationFailed",
        blocker,
        rows,
        part_c_role=str(args.part_c_role),
        forced_after_c_fail=max([ival(r.get("forced_after_c_fail")) for r in rows] or [0]),
        diagnostic_only=int(args.diagnostic_only),
        seed_count=int(args.part_e_seed_count),
        scheme_groups=summaries,
        passing_scheme_groups=pass_groups,
    )
    write_json(OUT_ROOT / "part_e_optimizer_positive_control_summary.json", summary)
    next_path = write_next_actions("e", summary["route"], blocker, [] if gate else ["try gate_floor 0.05/0.10, stat warmup 5/10, block gate, lower Sobolev exponent"], evidence={"matrix": rel(matrix), "passing_scheme_groups": pass_groups})
    append_exec("part-e-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part E optimizer positive-control", summary)
    return summary


def part_f_base_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    e = read_json(OUT_ROOT / "part_e_optimizer_positive_control_summary.json")
    groups = e.get("passing_scheme_groups", [])
    if isinstance(groups, list) and groups:
        return [dict(g, forced_after_e_fail=0) for g in groups]
    out = []
    for scheme in csv_items(args.part_f_base_schemes):
        for basis in csv_items(args.part_f_basis):
            for depth in csv_items(args.part_f_depths):
                out.append({"scheme": scheme, "basis_key": basis, "depth": depth, "forced_after_e_fail": 1})
    return out


def part_f_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    requested = set(csv_items(args.part_f_parts))
    for group_idx, group in enumerate(part_f_base_groups(args)):
        for safety in csv_items(args.part_f_safety_schemes):
            if "F3" in requested:
                for task in csv_items(args.part_f_tasks):
                    for seed in range(int(args.part_f_c2_seed_count)):
                        jobs.append({"group_idx": group_idx, "scheme": group["scheme"], "basis_key": group["basis_key"], "depth": group["depth"], "forced_after_e_fail": group.get("forced_after_e_fail", 0), "part": "F3", "safety_type": safety, "teacher_type": "c2_visual_synthetic", "task": task, "seed": seed})
            if "F5" in requested:
                for seed in range(int(args.part_f_seed_count)):
                    jobs.append({"group_idx": group_idx, "scheme": group["scheme"], "basis_key": group["basis_key"], "depth": group["depth"], "forced_after_e_fail": group.get("forced_after_e_fail", 0), "part": "F5", "safety_type": safety, "teacher_type": "f5_no_debt_calibration", "task": "", "seed": seed})
    return jobs


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    e = read_json(OUT_ROOT / "part_e_optimizer_positive_control_summary.json")
    if int(e.get("gate_pass", 0)) != 1 and not int(args.force_part_f_after_e_fail):
        return write_blocked_summary("F", "F_BlockedByPartE", str(e.get("dominant_blocker", "part_e_failed")), "Part E C2 formation did not pass and --force-part-f-after-e-fail=0.")
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for job in shard_items(part_f_jobs(args), args):
        try:
            row = train_v23_scheme(
                str(job["scheme"]),
                str(job["basis_key"]),
                str(job["depth"]),
                str(job["teacher_type"]),
                str(job["task"]),
                int(job["seed"]),
                args,
                device,
                part=str(job["part"]),
                safety_type=str(job["safety_type"]),
            )
            row["group_idx"] = int(job["group_idx"])
            row["forced_after_e_fail"] = int(job.get("forced_after_e_fail", 0))
            rows.append(row)
        except Exception as exc:
            rows.append({"part": str(job.get("part", "F")), "status": "error", "error_message": repr(exc), **job, **AUDIT_DEFAULTS})
    path = OUT_ROOT / f"part_f_safety_envelope_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-f", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}; force_after_e_fail={int(args.force_part_f_after_e_fail)}")
    return gate_summary("F", 0, "F_ShardsWritten", "merge_required", rows)


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_safety_envelope_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_f_safety_envelope_matrix.csv", rows)
    write_rows(OUT_ROOT / "part_f_phase_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    group_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("group_idx"), r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("safety_type")) for r in ok}):
        group = [r for r in ok if (r.get("group_idx"), r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("safety_type")) == key]
        f3 = [r for r in group if r.get("part") == "F3"]
        f5 = [r for r in group if r.get("part") == "F5"]
        f0 = [r for r in ok if r.get("scheme") == key[1] and r.get("basis_key") == key[2] and r.get("depth") == key[3] and r.get("safety_type") == "F0_no_safety_control" and r.get("part") == "F3"]
        c2_med = median([r.get("C2_coverage_improvement") for r in f3])
        c2_base = median([r.get("C2_coverage_improvement") for r in f0], c2_med)
        retention = c2_med / max(abs(c2_base), 1.0e-12) if f3 else 0.0
        f5_count = sum(ival(r.get("F5_no_debt")) for r in f5)
        component_nonpos = sum(
            int(
                fval(r.get("Brier_delta")) <= float(args.no_debt_budget)
                and fval(r.get("ECE_delta")) <= float(args.no_debt_budget)
                and fval(r.get("tail95_delta")) <= float(args.no_debt_budget)
                and fval(r.get("tail99_delta")) <= float(args.no_debt_budget)
                and fval(r.get("margin10_delta")) >= -float(args.no_debt_budget)
            )
            for r in f5
        )
        random_count = max(
            [
                sum(
                    ival(rr.get("F5_no_debt"))
                    for rr in ok
                    if rr.get("scheme") == key[1]
                    and rr.get("basis_key") == key[2]
                    and rr.get("depth") == key[3]
                    and rr.get("safety_type") == "F6_random_debt_veto_control"
                    and rr.get("part") == "F5"
                )
            ]
            or [0]
        )
        random_gap = f5_count - random_count
        gate = int(f5 and f5_count >= int(args.f5_diagnostic_gate) and retention >= 0.80 and random_gap > 0)
        group_summaries.append(
            {
                "group_idx": ival(key[0]),
                "scheme": key[1],
                "basis_key": key[2],
                "depth": key[3],
                "safety_type": key[4],
                "rows": len(group),
                "F5_no_debt_count": f5_count,
                "component_non_positive_rows": component_nonpos,
                "C2_coverage_improvement_after_veto_median": c2_med,
                "C2_retention_ratio_vs_no_safety": retention,
                "veto_density_mean": median([r.get("veto_density_mean") for r in group]),
                "random_veto_matched_gap": random_gap,
                "part_f_group_gate_pass": gate,
            }
        )
    pass_groups = [g for g in group_summaries if ival(g.get("part_f_group_gate_pass")) == 1 and str(g.get("safety_type")) not in {"F0_no_safety_control", "F6_random_debt_veto_control"}]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    blocker = "none" if gate else ("part_f_job_errors" if any(r.get("status") == "error" for r in rows) else "f5_no_debt_or_c2_retention_failed")
    summary = gate_summary(
        "F",
        gate,
        "F_SafetyEnvelopePass" if gate else "F_SafetyEnvelopeFailed",
        blocker,
        rows,
        part_c_role=str(args.part_c_role),
        forced_after_c_fail=max([ival(r.get("forced_after_c_fail")) for r in rows] or [0]),
        forced_after_e_fail=max([ival(r.get("forced_after_e_fail")) for r in rows] or [0]),
        diagnostic_only=int(args.diagnostic_only),
        seed_count=int(args.part_f_seed_count),
        group_summaries=group_summaries,
        passing_part_f_groups=pass_groups,
    )
    write_json(OUT_ROOT / "part_f_safety_envelope_summary.json", summary)
    write_json(OUT_ROOT / "part_f_phase_summary.json", summary)
    next_path = write_next_actions("f", summary["route"], blocker, [] if gate else ["component-wise veto threshold", "block-local veto", "debt SNR diagnostic", "finite-step guard veto"], evidence={"matrix": rel(matrix), "passing_part_f_groups": pass_groups})
    append_exec("part-f-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(next_path)}")
    append_recap("Part F safety envelope", summary)
    return summary


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    f = read_json(OUT_ROOT / "part_f_safety_envelope_summary.json")
    if int(f.get("gate_pass", 0)) != 1:
        return write_blocked_summary("G", "G_BlockedByPartF", str(f.get("dominant_blocker", "part_f_failed")), "Part F safety envelope did not pass.")
    return write_blocked_summary("G", "G_C2PassF5Fail", "full_matrix_not_run", "Part G full positive-control matrix requires Part F pass.")


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    g = read_json(OUT_ROOT / "part_g_full_positive_control_summary.json")
    if int(g.get("gate_pass", 0)) != 1:
        return write_blocked_summary("H", "H_BlockedByPartG", str(g.get("dominant_blocker", "part_g_failed")), "Part G did not pass, so real-task preflight is forbidden.")
    return write_blocked_summary("H", "J_RealTaskPreflightFailed", "not_run", "Real-task preflight not run.")


def run_part_i(args: argparse.Namespace) -> dict[str, Any]:
    g = read_json(OUT_ROOT / "part_g_full_positive_control_summary.json")
    return write_blocked_summary("I", "I_DiagnosticOnly", str(g.get("dominant_blocker", "no_part_g")), "Fiber/quotient diagnostic is non-official and was not run because main chain did not pass.")


def stale_artifact_audit() -> dict[str, Any]:
    expected = [
        "final_route.json",
        "reproduction_manifest.md",
        "stale_artifact_audit.json",
        "part_a_identity_summary.json",
        "part_b_history_lock.json",
        "part_c_metric_sanity_summary.json",
        "part_c_edge_sobolev_metric_summary.json",
        "part_d_population_gate_summary.json",
        "part_e_optimizer_positive_control_summary.json",
        "part_f_safety_envelope_summary.json",
        "part_f_phase_summary.json",
        "part_g_full_positive_control_summary.json",
        "part_g_phase_summary.json",
        "part_h_real_task_preflight_summary.json",
        "part_i_failure_decomposition.md",
    ]
    missing = [p for p in expected if not (OUT_ROOT / p).exists()]
    return {"expected_files": expected, "missing_files": missing, "stale_artifact_count": 0 if not missing else len(missing)}


def run_part_k(args: argparse.Namespace) -> dict[str, Any]:
    parts = {
        "A": read_json(OUT_ROOT / "part_a_identity_summary.json"),
        "B": read_json(OUT_ROOT / "part_b_history_lock.json"),
        "C": read_json(OUT_ROOT / "part_c_metric_sanity_summary.json"),
        "D": read_json(OUT_ROOT / "part_d_population_gate_summary.json"),
        "E": read_json(OUT_ROOT / "part_e_optimizer_positive_control_summary.json"),
        "F": read_json(OUT_ROOT / "part_f_safety_envelope_summary.json"),
        "G": read_json(OUT_ROOT / "part_g_full_positive_control_summary.json"),
        "H": read_json(OUT_ROOT / "part_h_real_task_preflight_summary.json"),
    }
    if int(parts["A"].get("gate_pass", 0)) != 1:
        route, blocker = "A_CodeIdentityFailed", parts["A"].get("dominant_blocker", "part_a")
    elif int(parts["B"].get("gate_pass", 0)) != 1:
        route, blocker = "B_HistoryLockFailed", parts["B"].get("dominant_blocker", "part_b")
    elif int(parts["C"].get("gate_pass", 0)) != 1:
        route, blocker = "C_EdgeSobolevMetricFailed", parts["C"].get("dominant_blocker", "part_c")
    elif int(parts["D"].get("gate_pass", 0)) != 1:
        route, blocker = "D_PopulationGateFailed", parts["D"].get("dominant_blocker", "part_d")
    elif int(parts["E"].get("gate_pass", 0)) != 1:
        route, blocker = "E_C2FormationFailed", parts["E"].get("dominant_blocker", "part_e")
    elif int(parts["F"].get("gate_pass", 0)) != 1:
        route, blocker = "F_SafetyEnvelopeFailed", parts["F"].get("dominant_blocker", "part_f")
    elif int(parts["G"].get("gate_pass", 0)) != 1:
        route, blocker = "G_C2PassF5Fail", parts["G"].get("dominant_blocker", "part_g")
    elif int(parts["H"].get("gate_pass", 0)) != 1:
        route, blocker = "J_RealTaskPreflightFailed", parts["H"].get("dominant_blocker", "part_h")
    else:
        route, blocker = "J_RealTaskPreflightProgress", "none"
    lines = [
        "# v23.00R failure decomposition",
        "",
        f"Final route: `{route}`",
        f"Dominant blocker: `{blocker}`",
        "",
        "## Part summaries",
    ]
    for key, data in parts.items():
        lines.append(f"- Part {key}: route=`{data.get('route', 'missing')}`, gate_pass=`{data.get('gate_pass', 0)}`, blocker=`{data.get('dominant_blocker', 'missing')}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "This file is generated from current artifacts only. Blocked downstream parts are not treated as success evidence.",
        ]
    )
    path = OUT_ROOT / "part_i_failure_decomposition.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_ROOT / "part_k_failure_decomposition.md").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    rows = [{"part": "K", "status": "ok", "route": route, "dominant_blocker": str(blocker), **AUDIT_DEFAULTS}]
    summary = gate_summary("K", 1, route, str(blocker), rows, part_routes={k: v.get("route", "missing") for k, v in parts.items()})
    write_json(OUT_ROOT / "part_k_summary.json", summary)
    final = {
        "version": "v23.00R",
        "route": route,
        "dominant_blocker": str(blocker),
        "official_candidate_gate_pass": int(route in {"I_C2F5PositiveControlPass", "J_RealTaskPreflightProgress"}),
        "promotion_allowed": bool(route in {"I_C2F5PositiveControlPass", "J_RealTaskPreflightProgress"}),
        "plan": rel(PLAN),
        "runner": rel(RUNNER),
        "execution_log": rel(EXEC_LOG),
        "recap_log": rel(RECAP_LOG),
        "used_fake_data_rows": sum(ival(v.get("used_fake_data_rows")) for v in parts.values()),
        "held_test_usage": sum(ival(v.get("held_test_usage")) for v in parts.values()),
        "generated_at": now(),
    }
    write_json(OUT_ROOT / "final_route.json", final)
    append_exec("part-k", command_text(sys.argv), "done", files=f"{rel(path)}; {rel(OUT_ROOT / 'final_route.json')}")
    append_recap("Part K failure decomposition", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    k = read_json(OUT_ROOT / "part_k_summary.json")
    final = read_json(OUT_ROOT / "final_route.json")
    audit = stale_artifact_audit()
    write_json(OUT_ROOT / "stale_artifact_audit.json", audit)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        "# v23.00R reproduction manifest\n\n"
        f"- Plan: `{rel(PLAN)}`\n"
        f"- Runner: `{rel(RUNNER)}`\n"
        f"- Final route: `{final.get('route', k.get('route', 'missing'))}`\n"
        f"- Output root: `{rel(OUT_ROOT)}`\n\n"
        "## Typical commands\n\n"
        f"- Part A: `{PYTHON} {rel(RUNNER)} --mode part-a --device cuda:0`\n"
        f"- Part B: `{PYTHON} {rel(RUNNER)} --mode part-b --device cuda:0`\n"
        f"- Part C shard: `CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 0 --device cuda:0`\n"
        f"- Part C merge: `{PYTHON} {rel(RUNNER)} --mode part-c-merge --device cuda:0`\n"
        f"- Part D shard: `CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-d --shard-count 4 --shard-index 0 --device cuda:0`\n"
        f"- Part D merge: `{PYTHON} {rel(RUNNER)} --mode part-d-merge --device cuda:0`\n"
        f"- Finalize: `{PYTHON} {rel(RUNNER)} --mode finalize --device cuda:0`\n",
        encoding="utf-8",
    )
    with (OUT_ROOT / "sha256_manifest.txt").open("w", encoding="utf-8") as fh:
        for p in sorted(OUT_ROOT.rglob("*")):
            if p.is_file():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                fh.write(f"{h}  {rel(p)}\n")
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(manifest)}; {rel(OUT_ROOT / 'stale_artifact_audit.json')}")
    append_recap("Final route", {**final, **audit})
    return {**final, **audit}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--pc-basis", default="D-FOU")
    p.add_argument("--basis-k-override", type=int, default=0)
    p.add_argument("--dfour-k", type=int, default=3)
    p.add_argument("--input-dim", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--deep-width", type=int, default=12)
    p.add_argument("--mlp-hidden", type=int, default=24)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--synthetic-train-size", type=int, default=72)
    p.add_argument("--synthetic-guard-size", type=int, default=72)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--batch-size", type=int, default=36)
    p.add_argument("--channel-source-size", type=int, default=16)
    p.add_argument("--fresh-cohort-split", type=int, default=1)
    p.add_argument("--snr-examples", type=int, default=16)
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--part-c-role", default="hard", choices=["hard", "sanity"])
    p.add_argument("--diagnostic-only", type=int, default=0)
    p.add_argument("--continuation-metric-schemes", default="sobolev_s0p0")
    p.add_argument("--part-c-basis", default="dche_k5,dche_k9,dfour_default")
    p.add_argument("--part-c-depths", default="depth2,depth3")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--sobolev-exponents", default="0.25,0.5,1.0")
    p.add_argument("--functional-gram-exponents", default="")
    p.add_argument("--functional-gram-quadrature-points", type=int, default=257)
    p.add_argument("--edge-weight-normalization", default="median", choices=["median", "mean", "trace"])
    p.add_argument("--edge-weight-ridge", type=float, default=0.0)
    p.add_argument("--part-d-basis", default="dche_k5,dche_k9,dfour_default")
    p.add_argument("--part-d-depths", default="depth2,depth3")
    p.add_argument("--part-d-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-d-controls", default="c2_positive,random_label,source_shuffle,mlp_friendly_negative")
    p.add_argument("--part-d-seed-count", type=int, default=3)
    p.add_argument("--part-d-gate-families", default="diagonal,block,lowrank,cohort")
    p.add_argument("--snr-beta", type=float, default=2.0)
    p.add_argument("--lowrank-rank", type=int, default=4)
    p.add_argument("--cohort-count", type=int, default=4)
    p.add_argument("--adamw-lr", type=float, default=0.02)
    p.add_argument("--edge-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--population-grad-examples", type=int, default=8)
    p.add_argument("--debt-grad-examples", type=int, default=8)
    p.add_argument("--debt-veto-components", default="Brier,ECE,tail95,tail99,margin10")
    p.add_argument("--debt-regularization-weight", type=float, default=0.0)
    p.add_argument("--debt-regularization-components", default="Brier,ECE,tail95,tail99,margin10")
    p.add_argument("--gate-floor", type=float, default=0.0)
    p.add_argument("--stat-warmup-steps", type=int, default=5)
    p.add_argument("--train-steps", type=int, default=40)
    p.add_argument("--no-debt-budget", type=float, default=0.01)
    p.add_argument("--finite-step-components", default="Brier,ECE,tail95,tail99,margin10")
    p.add_argument("--finite-step-tries", type=int, default=5)
    p.add_argument("--finite-step-shrink", type=float, default=0.5)
    p.add_argument("--finite-step-guard-examples", type=int, default=128)
    p.add_argument("--finite-step-guard-every", type=int, default=1)
    p.add_argument("--finite-step-apply-all-parts", type=int, default=0)
    p.add_argument("--finite-step-c2-floor", type=float, default=-0.05)
    p.add_argument("--part-e-schemes", default="E0_AdamW_control,E1_EdgeSobolev_AdamW_s0,E3_EdgeSobolev_DiagonalSNR_s0,E4_EdgeSobolev_BlockSNR_s0,E6_RandomMatchedGate_Control_s0")
    p.add_argument("--part-e-basis", default="dche_k9")
    p.add_argument("--part-e-depths", default="depth3")
    p.add_argument("--part-e-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-e-seed-count", type=int, default=3)
    p.add_argument("--force-part-e-after-d-fail", type=int, default=0)
    p.add_argument("--c2-coverage-gate", type=float, default=0.02)
    p.add_argument("--c2-rows-ge-fraction", type=float, default=0.50)
    p.add_argument("--max-overhead-ratio", type=float, default=2.0)
    p.add_argument("--part-f-base-schemes", default="E3_EdgeSobolev_DiagonalSNR_s0")
    p.add_argument("--part-f-basis", default="dche_k9")
    p.add_argument("--part-f-depths", default="depth3")
    p.add_argument("--part-f-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-f-safety-schemes", default="F0_no_safety_control,F2_component_conflict_veto,F6_random_debt_veto_control")
    p.add_argument("--part-f-parts", default="F3,F5")
    p.add_argument("--part-f-c2-seed-count", type=int, default=3)
    p.add_argument("--part-f-seed-count", type=int, default=15)
    p.add_argument("--force-part-f-after-e-fail", type=int, default=0)
    p.add_argument("--f5-diagnostic-gate", type=int, default=10)
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
        return run_part_h(args)
    if mode == "part-i":
        return run_part_i(args)
    if mode == "part-k":
        return run_part_k(args)
    if mode == "finalize":
        return finalize(args)
    if mode == "all":
        run_part_a(args)
        run_part_b(args)
        run_part_c(args)
        merge_part_c(args)
        run_part_d(args)
        merge_part_d(args)
        run_part_e(args)
        run_part_f(args)
        run_part_g(args)
        run_part_h(args)
        run_part_i(args)
        run_part_k(args)
        return finalize(args)
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
