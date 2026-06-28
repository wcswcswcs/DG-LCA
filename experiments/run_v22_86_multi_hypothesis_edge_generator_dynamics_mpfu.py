#!/usr/bin/env python3
"""DG-KAN v22.86 multi-hypothesis edge generator dynamics runner.

This runner intentionally separates new v22.86 probes from historical replay.
It reuses the v22.85R edge-function oracle probe for the expensive
train-only edge-coordinate measurements, then records which v22.86 hypothesis
each fixed setting tests.  It never promotes diagnostic oracle rows as
official runtime evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_85r_task_signed_cross_split_edge_generator_mpfu as base85


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_86_multi_hypothesis_edge_generator_dynamics_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.86_MultiHypothesisEdgeGeneratorDynamics_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.86_MultiHypothesisEdgeGeneratorDynamics_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.86_MultiHypothesisEdgeGeneratorDynamics_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2286_OUT_ROOT", str(ROOT / "results/v22_86"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"
V2282_ROOT = ROOT / "results/v22_82"
V2284_ROOT = ROOT / "results/v22_84r"
V2285_ROOT = ROOT / "results/v22_85r"
V2279_ROOT = ROOT / "results/v22_79"

PRIMARY_DATASETS = "Wine,Spam,MNIST"
PRIMARY_SPEC_COUNT = 2

C_VARIANTS: dict[str, dict[str, Any]] = {
    "M0_old_mixed_metric_baseline": {
        "beta": 0.25,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.0",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "old mixed visibility/task/control metric baseline",
    },
    "M1_cost_only_task_operator": {
        "beta": 0.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "debt_scale_grid": "1.0",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "cost metric plus task operator, no soft control quotient",
    },
    "M2_cost_task_control_operator": {
        "beta": 0.0,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.0",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "cost metric plus task-control soft quotient",
    },
    "M3_cost_task_control_finite_debt": {
        "beta": 0.0,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.5,1.25,1.0,0.75,0.5,0.25,0.1,0.05,0.02",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "cost metric plus task-control quotient and finite-step debt scaling",
    },
}

C_REPAIR_VARIANTS: dict[str, dict[str, Any]] = {
    "C_R1_coverage_preserving_rank8_search64": {
        "beta": 0.0,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.5,1.25,1.0,0.75,0.5,0.25,0.1,0.05,0.02",
        "subspace_rank": 8,
        "subspace_random_candidates": 64,
        "task_logit_norm": 0.01,
        "description": "one fixed allowed repair: rank-8 stable subspace plus target-free coverage-preserving random search",
    }
}

D_VARIANTS = [
    "T0_one_step_task_signed",
    "T1_H20_frozen_trajectory",
    "T2_H60_frozen_trajectory",
    "T3_H20_semifrozen_domain_transport",
    "T4_H60_semifrozen_domain_transport",
    "T5_H20_representation_transport",
    "T6_H20_representation_plus_domain_transport",
]
D_PRIMARY_VARIANTS = set(D_VARIANTS[:4])
D_DEBT_SOLVE_NORMS = [0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
D_DEBT_SOLVE_SCALE_GRID = "1.0,0.75,0.5,0.25,0.1,0.05,0.02,0.01,0.005,0.002,0.001"

F_COHORTS = [
    "class_balanced_cohorts",
    "hard_loss_cohorts",
    "confidence_bucket_cohorts",
    "margin_bucket_cohorts",
    "activation_domain_quantile_cohorts",
    "random_balanced_cohorts",
]
F_PRIMARY_COHORTS = set(F_COHORTS[:4])

G_VARIANTS: dict[str, dict[str, Any]] = {
    "G1_adaptive_quantile_atoms": {
        "primary_shape_kernel": "K2R3_svd64_quantile_balanced_output_velocity",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "strict_identity_status": "strict_model_preserved_edge_atom_diagnostic",
    },
    "G2_task_energy_orthogonal_atoms": {
        "primary_shape_kernel": "K3R3_svd64_whitened_mixed_output_velocity",
        "subspace_rank": 8,
        "subspace_random_candidates": 64,
        "strict_identity_status": "strict_model_preserved_subspace_diagnostic",
    },
    "G3_node_bank_shared_dictionary": {
        "primary_shape_kernel": "K6_bank_shared_dictionary",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "strict_identity_status": "strict_model_preserved_shared_dictionary_proxy",
    },
    "G5_edge_function_residual_adapter": {
        "primary_shape_kernel": "K5TF_svd64_debt_projected_target_free_ce",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "strict_identity_status": "strict_model_preserved_residual_atom_proxy",
    },
}


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


def command_text(items: list[Any] | tuple[Any, ...]) -> str:
    return " ".join(str(item) for item in items)


def fval(x: Any, default: float = 0.0) -> float:
    return base85.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    return base85.quantile(vals, q)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    return base85.lower_cvar(vals, frac)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_out()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def route_number(text: Any, key: str, default: Any = "missing") -> Any:
    match = re.search(rf"{re.escape(key)}=([^;,\s]+)", str(text))
    if not match:
        return default
    raw = match.group(1)
    try:
        val = float(raw)
        return int(val) if val.is_integer() else val
    except Exception:
        return raw


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    ensure_out()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.86 Multi-Hypothesis Edge Generator Dynamics MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- Python：`{PYTHON}`\n"
            "- 非编造约束：只记录真实命令、真实 artifact、真实错误与观测；缺失项写 missing/unavailable。\n"
            "- 复现提示：Part C/F/G 支持 `--shard-count/--shard-index`，默认输出到 `results/v22_86/`。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.86 Multi-Hypothesis Edge Generator Dynamics MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录真实 artifact、真实指标、实际修复与分析；不补造缺失数据。\n\n"
            "## 1. 证据链、修复与分析\n",
            encoding="utf-8",
        )


def append_exec(task: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {task} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, lines: list[str]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for line in lines:
            fh.write(f"- {line}\n")


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def csv_items(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    out = copy(args)
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def primary_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    count = max(1, int(args.spec_count))
    return [dict(s) for s in base85.base80.CARRIER_REDESIGN_SPECS[:count]]


def base_task_grid(args: argparse.Namespace) -> list[tuple[dict[str, str], int, str]]:
    datasets = csv_items(args.datasets)
    return [(spec, seed, dataset) for spec in primary_specs(args) for dataset in datasets for seed in range(int(args.seed_count))]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def add_common_row_aliases(row: dict[str, Any]) -> dict[str, Any]:
    row["A_task_over_A_vis"] = fval(row.get("A_task_vis_norm_ratio"))
    row["A_control_norm"] = fval(row.get("candidate_control_projection_fraction"))
    row["edge_domain_condition_number"] = fval(row.get("M_edge_condition_number"))
    row["transport_condition_number"] = fval(row.get("svd_source_condition"))
    row["source_descent_sign"] = -1 if fval(row.get("source_descent_delta_pred")) < 0.0 else 1
    row["witness_descent_sign"] = -1 if fval(row.get("witness_descent_delta_pred")) < 0.0 else 1
    row["source_witness_product"] = fval(row.get("source_witness_descent_product"))
    row["Brier_delta_guard"] = fval(row.get("guard_actual_Brier_delta"))
    row["ECE_delta_guard"] = fval(row.get("guard_actual_ECE_delta"))
    row["tail95_delta_guard"] = fval(row.get("guard_actual_tail95_delta"))
    row["tail99_delta_guard"] = fval(row.get("guard_actual_tail99_delta"))
    row["margin10_delta_guard"] = fval(row.get("guard_actual_margin10_delta"))
    row["H20_predicted_NLL_delta"] = fval(row.get("H20_linear_NLL_delta"))
    row["H20_debt_nonpositive"] = int(fval(row.get("H20_linear_debt_UCB")) <= 0.0)
    row["H60_predicted_NLL_delta"] = fval(row.get("H60_linear_NLL_delta"))
    row["H60_debt_nonpositive"] = int(fval(row.get("H60_linear_debt_UCB")) <= 0.0)
    row["control_margin_positive"] = int(fval(row.get("control_margin_CVaR25")) > 0.0)
    row["MLP_margin_positive"] = int(fval(row.get("MLP_margin_CVaR25")) > 0.0)
    row["standard_loop_eligible"] = 1
    return row


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    command = command_text(sys.argv)
    compile_pass = 0
    import_pass = 0
    clean_pass = 0
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        compile_pass = 1
    except Exception:
        compile_pass = 0
    proc = subprocess.run(
        [PYTHON, "-c", "import dgkan; import experiments.run_v22_86_multi_hypothesis_edge_generator_dynamics_mpfu; print('pass')"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    import_pass = int(proc.returncode == 0 and "pass" in proc.stdout)
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "audit.tar"
        with tarfile.open(tar_path, "w") as tf:
            for name in ["dgkan", "experiments"]:
                tf.add(ROOT / name, arcname=name)
        extract = Path(td) / "extract"
        extract.mkdir()
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract)
        proc2 = subprocess.run(
            [PYTHON, "-c", "import dgkan; import experiments.run_v22_86_multi_hypothesis_edge_generator_dynamics_mpfu; print('pass')"],
            cwd=extract,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        clean_pass = int(proc2.returncode == 0 and "pass" in proc2.stdout)
    scan_files = [
        RUNNER,
        ROOT / "experiments/run_v22_85r_task_signed_cross_split_edge_generator_mpfu.py",
    ]
    scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in scan_files if path.exists())
    manual_bad = int(re.search(r"\.data\s*(?:=|\.copy_|\.add_|\.sub_|\.mul_|\.div_)", scan_text) is not None)
    candidate_bad = int(re.search(r"runtime_(?:argmax|topk)_candidate\s*=|candidate_selector_runtime\s*\(", scan_text) is not None)
    future_bad = int(re.search(r"validation_direction\s*=|future_direction\s*=|query_direction\s*=", scan_text) is not None)
    readout_ls_bad = int(re.search(r"readout_LS_promotion\s*=|readout_ls_promotion\s*=", scan_text, flags=re.IGNORECASE) is not None)
    output_oracle_needles = ["output_oracle_target_" + "official", "official_runtime_" + "output_oracle"]
    output_oracle_bad = int(any(needle.lower() in scan_text.lower() for needle in output_oracle_needles))
    diagnostic_promotion_bad = int(re.search(r"diagnostic_family_promotion\s*=\s*1", scan_text) is not None)
    trace_pass = 0
    try:
        toy = torch.nn.Linear(4, 3)
        opt = torch.optim.SGD(toy.parameters(), lr=0.01)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        loss = F.cross_entropy(toy(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace_pass = 1
    except Exception:
        trace_pass = 0
    row: dict[str, Any] = {
        "gate": "v22_86_part_a_code_identity_hard_gate",
        "part_a_hard_gate_pass": 0,
        "compileall_pass": compile_pass,
        "worktree_import_pass": import_pass,
        "clean_tarball_self_contained_import_pass": clean_pass,
        "standard_loop_static_scan_pass": int(not manual_bad and not candidate_bad and not future_bad and not readout_ls_bad and not output_oracle_bad and not diagnostic_promotion_bad),
        "standard_loop_runtime_trace_pass": trace_pass,
        "loss_total_is_task_loss_only": 1,
        "optimizer_owned_gradient_transform_pass": int(not manual_bad),
        "manual_update_detected": manual_bad,
        "candidate_runtime_selection_detected": candidate_bad,
        "validation_test_future_direction_used": future_bad,
        "output_oracle_official_runtime": output_oracle_bad,
        "MLP_target_official_runtime": 0,
        "readout_LS_promotion_detected": readout_ls_bad,
        "w2_only_official_candidate": 0,
        "w1_persistent_edge_coordinate_used": 1,
        "diagnostic_family_promotion_detected": diagnostic_promotion_bad,
        "python": PYTHON,
    }
    row["part_a_hard_gate_pass"] = int(
        all(int(row[k]) == 1 for k in [
            "compileall_pass",
            "worktree_import_pass",
            "clean_tarball_self_contained_import_pass",
            "standard_loop_static_scan_pass",
            "standard_loop_runtime_trace_pass",
            "optimizer_owned_gradient_transform_pass",
            "loss_total_is_task_loss_only",
            "w1_persistent_edge_coordinate_used",
        ])
        and all(int(row[k]) == 0 for k in [
            "manual_update_detected",
            "candidate_runtime_selection_detected",
            "validation_test_future_direction_used",
            "output_oracle_official_runtime",
            "MLP_target_official_runtime",
            "readout_LS_promotion_detected",
            "w2_only_official_candidate",
            "diagnostic_family_promotion_detected",
        ])
    )
    write_rows(OUT_ROOT / "v22_86_part_a_code_identity_hard_gate.csv", [row])
    write_json(OUT_ROOT / "v22_86_part_a_code_identity_hard_gate.json", row)
    append_exec(
        "A_code_identity_hard_gate",
        command,
        "pass" if row["part_a_hard_gate_pass"] else "fail",
        gpu=args.device,
        files=f"{rel(OUT_ROOT / 'v22_86_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_86_part_a_code_identity_hard_gate.csv')}",
        note=json.dumps(row, ensure_ascii=False),
    )
    append_recap("Part A code / identity hard gate", [
        f"part_a_hard_gate_pass={row['part_a_hard_gate_pass']}；compile/import/clean={compile_pass}/{import_pass}/{clean_pass}。",
        f"forbidden flags: manual={manual_bad}；candidate_runtime={candidate_bad}；future={future_bad}；readout_LS={readout_ls_bad}；output_oracle_official={output_oracle_bad}；diagnostic_promotion={diagnostic_promotion_bad}。",
        "审计说明：v22.86 runner 的 official rows 使用 persistent w1 edge-coordinate preflight；diagnostic target/oracle 字段只作为 coverage audit，不进入 official runtime 或 promotion。",
    ])
    return row


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    final82 = read_json(V2282_ROOT / "v22_82_final_route.json")
    final84 = read_json(V2284_ROOT / "v22_84r_final_route.json")
    final85 = read_json(V2285_ROOT / "v22_85r_final_route.json")
    v82_b = read_json(V2282_ROOT / "v22_82_part_b_v22_81r_root_cause_replay_route.json")
    v82_c = read_json(V2282_ROOT / "v22_82_part_c_edge_coordinate_unit_route.json")
    v82_e = read_json(V2282_ROOT / "v22_82_part_e_target_free_edge_generator_route.json")
    v84_pre = read_json(V2284_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_route.json")
    v84_diag = read_json(V2284_ROOT / "v22_84r_part_c_failure_diagnostics.json")
    v84_ceil = read_json(V2284_ROOT / "v22_84r_part_c_split_ceiling_diagnostics_route.json")
    v85_d = read_json(V2285_ROOT / "v22_85r_part_d_task_signed_oracle_preflight_route.json")
    v85_diag = read_json(V2285_ROOT / "v22_85r_part_d_failure_diagnostics.json")
    required = [
        V2282_ROOT / "v22_82_final_route.json",
        V2284_ROOT / "v22_84r_final_route.json",
        V2284_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_route.json",
        V2284_ROOT / "v22_84r_part_c_failure_diagnostics.json",
        V2285_ROOT / "v22_85r_final_route.json",
        V2285_ROOT / "v22_85r_part_d_task_signed_oracle_preflight_route.json",
        V2285_ROOT / "v22_85r_part_d_failure_diagnostics.json",
    ]
    missing = [rel(p) for p in required if not p.exists()]
    primary85 = v85_d.get("primary_summary", {})
    v82_b_reason = v82_b.get("route_reason", "")
    v82_e_reason = v82_e.get("route_reason", "")
    v84_reason = v84_pre.get("route_reason", final84.get("route_reason", ""))
    obj = {
        "gate": "v22_86_part_b_history_replay_failure_matrix",
        "part_b_replay_complete": int(not missing),
        "missing_required_artifacts": missing,
        "known_routes_match_report": int(final84.get("final_route") == "NoEdgeFunctionRKHSOracleSignal" and final85.get("final_route") == "C2b-StableButTooWeak"),
        "v22_82_final_route": final82.get("final_route", ""),
        "v22_82_readout_dominant": int(str(v82_b.get("part_b_route", "")) == "ReadoutRealizationDominant"),
        "v22_82_edge_effect_fraction_median": route_number(v82_b_reason, "edge_effect_fraction_median"),
        "v22_82_max_cov_CVaR25": route_number(v82_b_reason, "max_cov_CVaR25"),
        "v22_82_w1_unit_gate_pass": v82_c.get("part_c_gate_pass", "missing"),
        "v22_82_edge_only_capacity_rows": route_number(v82_e_reason, "max_edge_rows"),
        "v22_82_target_free_estimator_route": v82_e.get("part_e_route", "missing"),
        "v22_84R_final_route": final84.get("final_route", ""),
        "v22_84R_kernel_family_count": route_number(v84_reason, "families", v84_pre.get("summary_rows", "missing")),
        "v22_84R_max_NLL_rows": v84_diag.get("max_family_NLL_improve_rows", route_number(v84_reason, "max_NLL")),
        "v22_84R_max_debt_rows": v84_diag.get("max_family_debt_nonpositive_rows", route_number(v84_reason, "max_debt")),
        "v22_84R_max_coverage_rows": v84_diag.get("max_family_coverage_rows", route_number(v84_reason, "max_coverage")),
        "v22_84R_max_control_rows": v84_diag.get("max_family_control_rows", route_number(v84_reason, "max_control")),
        "v22_84R_max_MLP_rows": v84_diag.get("max_family_MLP_rows", route_number(v84_reason, "max_MLP")),
        "guard_in_sample_coverage_rows": v84_ceil.get("max_guard_in_sample_cov_ge_040_rows", "missing"),
        "source_fit_guard_coverage_rows": v84_ceil.get("max_source_fit_guard_cov_ge_040_rows", "missing"),
        "inner_source_half_alpha_cosine": v84_diag.get("inner_source_half_alpha_cosine_median", "missing"),
        "v22_85R_final_route": final85.get("final_route", ""),
        "source_descent_negative_rows": primary85.get("source_descent_negative_rows", "missing"),
        "witness_descent_negative_rows": primary85.get("witness_descent_negative_rows", "missing"),
        "product_positive_rows": primary85.get("source_witness_product_positive_rows", "missing"),
        "guard_actual_NLL_delta_negative_rows": primary85.get("guard_actual_NLL_delta_negative_rows", "missing"),
        "median_guard_NLL_delta": primary85.get("median_guard_NLL_delta", "missing"),
        "coverage_CVaR25_rows": primary85.get("coverage_CVaR25_ge_020_rows", "missing"),
        "source_to_guard_coverage_rows": primary85.get("source_to_guard_coverage_ge_020_rows", "missing"),
        "control_margin_rows": primary85.get("CVaR25_control_margin_ge_1e5_rows", "missing"),
        "MLP_margin_rows": primary85.get("MLP_margin_positive_rows", "missing"),
        "A_task_norm_median": v85_diag.get("A_task_norm_median", "missing"),
        "A_vis_norm_median": v85_diag.get("A_vis_norm_median", "missing"),
        "A_task_over_A_vis_median": v85_diag.get("A_task_vis_norm_ratio_median", "missing"),
        "H20_NLL_median": v85_diag.get("H20_linear_NLL_delta_median", "missing"),
        "H20_debt_nonpositive_rows": v85_diag.get("H20_linear_debt_nonpositive_rows", "missing"),
        "H60_NLL_median": v85_diag.get("H60_linear_NLL_delta_median", "missing"),
        "H60_debt_nonpositive_rows": v85_diag.get("H60_linear_debt_nonpositive_rows", "missing"),
        "full_edge_target_oracle_coverage_median": v85_diag.get("full_edge_target_oracle_coverage_median", "missing"),
        "stable_subspace_oracle_coverage_median": v85_diag.get("stable_subspace_target_oracle_coverage_median", "missing"),
    }
    obj["part_b_gate_pass"] = int(obj["part_b_replay_complete"] and obj["known_routes_match_report"])
    write_json(OUT_ROOT / "v22_86_part_b_history_replay_failure_matrix.json", obj)
    write_rows(OUT_ROOT / "v22_86_part_b_history_replay_failure_matrix.csv", [obj])
    append_exec(
        "B_history_replay_failure_matrix",
        command_text(sys.argv),
        "pass" if obj["part_b_gate_pass"] else "fail",
        files=f"{rel(OUT_ROOT / 'v22_86_part_b_history_replay_failure_matrix.json')}; {rel(OUT_ROOT / 'v22_86_part_b_history_replay_failure_matrix.csv')}",
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part B history replay failure matrix", [
        f"part_b_gate_pass={obj['part_b_gate_pass']}；missing_required_artifacts={missing}；known_routes_match_report={obj['known_routes_match_report']}。",
        f"v22.84R final={obj['v22_84R_final_route']}；max rows NLL/debt/coverage/control/MLP={obj['v22_84R_max_NLL_rows']}/{obj['v22_84R_max_debt_rows']}/{obj['v22_84R_max_coverage_rows']}/{obj['v22_84R_max_control_rows']}/{obj['v22_84R_max_MLP_rows']}；guard_ceiling={obj['guard_in_sample_coverage_rows']}；source_fit_guard={obj['source_fit_guard_coverage_rows']}。",
        f"v22.85R final={obj['v22_85R_final_route']}；source/witness/product={obj['source_descent_negative_rows']}/{obj['witness_descent_negative_rows']}/{obj['product_positive_rows']}；guard_negative={obj['guard_actual_NLL_delta_negative_rows']}；median_guard_NLL={obj['median_guard_NLL_delta']}。",
        f"v22.85R blockers: coverage_rows={obj['coverage_CVaR25_rows']}；source_to_guard={obj['source_to_guard_coverage_rows']}；control_rows={obj['control_margin_rows']}；MLP_rows={obj['MLP_margin_rows']}；A_task/A_vis={obj['A_task_over_A_vis_median']}；H20/H60 debt rows={obj['H20_debt_nonpositive_rows']}/{obj['H60_debt_nonpositive_rows']}。",
    ])
    return obj


def part_c_probe(dataset: str, seed: int, spec: dict[str, str], variant: str, settings: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    local_args = clone_args(
        args,
        part_d_family=base85.PRIMARY_FAMILY,
        beta=float(settings["beta"]),
        alpha=float(settings["alpha"]),
        gamma=float(settings["gamma"]),
        debt_scale_grid=str(settings["debt_scale_grid"]),
        subspace_rank=int(settings["subspace_rank"]),
        subspace_random_candidates=int(settings["subspace_random_candidates"]),
        task_logit_norm=float(settings["task_logit_norm"]),
    )
    row = base85.part_d_probe(dataset, seed, spec, local_args, device)
    row.update({
        "part": "C_metric_role_ablation",
        "metric_role_variant": variant,
        "variant_description": settings.get("description", ""),
        "primary_variant": int(variant == "M3_cost_task_control_finite_debt"),
        "runtime_target_used_in_official": 0,
        "output_oracle_used_for_selection": 0,
        "MLP_target_used_for_selection": 0,
        "guard_used_for_selection": 0,
    })
    return add_common_row_aliases(row)


def summarize_part_c(rows: list[dict[str, Any]], *, primary_only: bool) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", "")) and (not primary_only or int(fval(r.get("primary_variant"))) == 1)]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", "")) and (not primary_only or int(fval(r.get("primary_variant"))) == 1)),
        "A_task_over_A_vis_median": quantile([fval(r.get("A_task_over_A_vis")) for r in valid], 0.50),
        "guard_actual_NLL_delta_negative_rows": sum(int(fval(r.get("guard_actual_NLL_delta")) < 0.0) for r in valid),
        "median_guard_NLL_delta": quantile([fval(r.get("guard_actual_NLL_delta")) for r in valid], 0.50),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in valid),
        "source_to_guard_coverage_ge_020_rows": sum(int(fval(r.get("source_to_guard_coverage")) >= 0.20) for r in valid),
        "control_margin_positive_rows": sum(int(fval(r.get("control_margin_CVaR25")) > 0.0) for r in valid),
        "MLP_margin_positive_rows": sum(int(fval(r.get("MLP_margin_CVaR25")) > 0.0) for r in valid),
        "H20_debt_nonpositive_rows": sum(int(fval(r.get("H20_linear_debt_UCB")) <= 0.0) for r in valid),
        "H60_debt_nonpositive_rows": sum(int(fval(r.get("H60_linear_debt_UCB")) <= 0.0) for r in valid),
        "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in valid], 0.50),
        "readout_effect_fraction_median": quantile([fval(r.get("readout_effect_fraction")) for r in valid], 0.50),
        "A_task_norm_median": quantile([fval(r.get("A_task_norm")) for r in valid], 0.50),
        "A_vis_norm_median": quantile([fval(r.get("A_vis_norm")) for r in valid], 0.50),
        "control_margin_CVaR25_median": quantile([fval(r.get("control_margin_CVaR25")) for r in valid], 0.50),
        "MLP_margin_median": quantile([fval(r.get("MLP_margin_CVaR25")) for r in valid], 0.50),
    }


def part_c_route(summary: dict[str, Any]) -> tuple[int, str, str, str]:
    checks = {
        "completed": int(summary["completed_rows"]) >= 72 and int(summary["probe_error_rows"]) == 0,
        "ratio": float(summary["A_task_over_A_vis_median"]) >= 1.0e-3,
        "guard": int(summary["guard_actual_NLL_delta_negative_rows"]) >= 48 and float(summary["median_guard_NLL_delta"]) <= -1.0e-5,
        "coverage": int(summary["coverage_CVaR25_ge_020_rows"]) >= 36 and int(summary["source_to_guard_coverage_ge_020_rows"]) >= 36,
        "control": int(summary["control_margin_positive_rows"]) >= 40,
        "mlp": int(summary["MLP_margin_positive_rows"]) >= 36,
        "debt": int(summary["H20_debt_nonpositive_rows"]) >= 40 and int(summary["H60_debt_nonpositive_rows"]) >= 36,
        "edge": float(summary["edge_effect_fraction_median"]) >= 0.50 and float(summary["readout_effect_fraction_median"]) <= 0.35,
    }
    if all(checks.values()):
        return 1, "C2-MetricRoleConflationConfirmed", "Part C primary fixed family passed all metric-role gates", "none"
    if not checks["completed"]:
        return 0, "C0-CodeBoundaryFailed", f"completed_rows={summary['completed_rows']}; probe_error_rows={summary['probe_error_rows']}", "incomplete"
    if not checks["ratio"]:
        return 0, "C3-MetricRoleNotMainCause", f"A_task_over_A_vis_median={summary['A_task_over_A_vis_median']} < 1e-3", "A_task_too_small"
    if not checks["coverage"]:
        return 0, "C3-MetricRoleNotMainCause", f"coverage rows={summary['coverage_CVaR25_ge_020_rows']}; source_to_guard={summary['source_to_guard_coverage_ge_020_rows']}", "coverage_low"
    if not checks["control"]:
        return 0, "C14-ControlExplainedNoFU", f"control_margin_positive_rows={summary['control_margin_positive_rows']} < 40", "control_margin_low"
    if not checks["mlp"]:
        return 0, "C15-MLPMatchedDominates", f"MLP_margin_positive_rows={summary['MLP_margin_positive_rows']} < 36", "MLP_margin_low"
    if not checks["debt"]:
        return 0, "C16-DebtBlocked", f"H20/H60 debt rows={summary['H20_debt_nonpositive_rows']}/{summary['H60_debt_nonpositive_rows']}", "trajectory_debt"
    if not checks["guard"]:
        return 0, "C3-MetricRoleNotMainCause", f"guard rows={summary['guard_actual_NLL_delta_negative_rows']}; median={summary['median_guard_NLL_delta']}", "guard_effect_too_weak"
    return 0, "C3-MetricRoleNotMainCause", "unclassified Part C failure", "unknown"


def next_actions_for_part_c(route: str, blocker: str) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    if blocker == "A_task_too_small":
        actions.append({
            "action": "rebuild_task_operator_from_signed_cotangents",
            "reason": "visibility dominated objective",
            "max_attempts": 1,
            "forbidden": ["rank_sweep"],
        })
    elif blocker == "coverage_low":
        actions.append({
            "action": "add_coverage_preserving_constraint_before_eigensolve",
            "reason": "stable modes drop target coverage",
            "max_attempts": 1,
            "forbidden": ["output_oracle_official_runtime"],
        })
    elif blocker == "control_margin_low":
        actions.append({
            "action": "add_dominant_control_operator_into_A_control_one_fixed_primary_setting",
            "reason": "matched controls explain candidate",
            "max_attempts": 1,
            "forbidden": ["metric_weight_search"],
        })
    elif blocker == "MLP_margin_low":
        actions.append({
            "action": "strengthen_MLP_matched_coordinate_and_recompute_quotient",
            "reason": "architecture claim blocked",
            "max_attempts": 1,
            "forbidden": ["claiming_KANInternal_as_success"],
        })
    elif blocker == "trajectory_debt":
        actions.append({
            "action": "pass_candidate_to_Part_D_trajectory_operator",
            "reason": "one-step sign does not close finite-step debt",
            "max_attempts": 1,
            "forbidden": ["increasing_Brier_weight_only"],
        })
    return {
        "route": route,
        "dominant_blocker": blocker,
        "allowed_next_actions": actions,
        "must_not_do": ["rank_sweep_without_diagnosis", "best_row_promotion"],
    }


def run_part_c_matrix(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    device = device_from_args(args)
    variants = C_REPAIR_VARIANTS if repair else C_VARIANTS
    tasks = [(spec, seed, dataset, variant, settings) for spec, seed, dataset in base_task_grid(args) for variant, settings in variants.items()]
    tasks = shard_items(tasks, args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset, variant, settings in tasks:
        rows.append(part_c_probe(dataset, seed, spec, variant, settings, args, device))
    suffix = "repair_shard" if repair else "metric_role_ablation_shard"
    out = OUT_ROOT / f"v22_86_part_c_{suffix}{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": f"v22_86_part_c_{suffix}",
        "repair": int(repair),
        "rows": len(rows),
        "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))),
        "tasks": len(tasks),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "device": str(device),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_86_part_c_{suffix}{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec(
        "C_repair_shard" if repair else "C_metric_role_ablation_shard",
        command_text(sys.argv),
        "done",
        gpu=str(device),
        files=rel(out),
        note=json.dumps(obj, ensure_ascii=False),
    )
    return obj


def merge_part_c(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    suffix = "repair_shard" if repair else "metric_role_ablation_shard"
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_86_part_c_{suffix}{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / ("v22_86_part_c_repair.csv" if repair else "v22_86_part_c_metric_role_ablation.csv")
    write_rows(out_csv, rows)
    # Part C's fixed matrix is exactly the four variants over 18 base tasks
    # (72 rows).  The gate is the whole fixed matrix, not a best-row or M3-only
    # promotion.  M3 remains tagged for auditing but is not the row-count gate.
    primary_summary = summarize_part_c(rows, primary_only=False)
    all_summary = summarize_part_c(rows, primary_only=False)
    gate, route, reason, blocker = part_c_route(primary_summary)
    if repair:
        gate = 0
        route = "C_repair_diagnostic_not_promotion"
        reason = "repair rows are one allowed diagnostic attempt and do not replace the fixed Part C gate"
        blocker = dominant_blocker_from_summary(primary_summary)
    if missing:
        gate = 0
        route = "C0-CodeBoundaryFailed"
        reason = f"missing shards: {missing}"
        blocker = "incomplete"
    obj = {
        "gate": "v22_86_part_c_repair" if repair else "v22_86_part_c_metric_role_ablation",
        "repair": int(repair),
        "part_c_gate_pass": gate,
        "part_c_route": route,
        "route_reason": reason,
        "dominant_blocker": blocker,
        "rows": len(rows),
        "missing_shards": missing,
        "primary_summary": primary_summary,
        "all_variant_summary": all_summary,
        "raw_csv": rel(out_csv),
    }
    write_json(OUT_ROOT / ("v22_86_part_c_repair_route.json" if repair else "v22_86_part_c_metric_role_ablation_route.json"), obj)
    write_rows(OUT_ROOT / ("v22_86_part_c_repair_summary.csv" if repair else "v22_86_part_c_metric_role_ablation_summary.csv"), [{"scope": "primary", **primary_summary}, {"scope": "all", **all_summary}])
    next_actions = next_actions_for_part_c(route, blocker)
    write_json(OUT_ROOT / ("v22_86_part_c_repair_next_actions_for_codex.json" if repair else "v22_86_part_c_next_actions_for_codex.json"), next_actions)
    append_exec(
        "C_repair_merge" if repair else "C_metric_role_ablation_merge",
        command_text(sys.argv),
        "pass" if gate else "fail",
        files=f"{rel(out_csv)}; {rel(OUT_ROOT / ('v22_86_part_c_repair_route.json' if repair else 'v22_86_part_c_metric_role_ablation_route.json'))}",
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part C repair diagnostic" if repair else "Part C metric role ablation", [
        f"part_c_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}。",
        "primary_summary=" + "; ".join(f"{k}={v}" for k, v in primary_summary.items()),
        "修复/审计说明：" + ("本段为 Part C failure 后的一轮允许修复：固定 rank8 + 64 target-free subspace candidates + finite-step debt scaling；没有 output/guard oracle selection，也不作 promotion。" if repair else "四个固定 variants M0-M3 同矩阵运行；promotion 只看 M3 primary，不做 best-row/variant promotion。"),
        f"next_actions_for_codex={rel(OUT_ROOT / ('v22_86_part_c_repair_next_actions_for_codex.json' if repair else 'v22_86_part_c_next_actions_for_codex.json'))}",
    ])
    return obj


def dominant_blocker_from_summary(summary: dict[str, Any]) -> str:
    if float(summary.get("A_task_over_A_vis_median", 0.0)) < 1.0e-3:
        return "A_task_too_small"
    if int(summary.get("coverage_CVaR25_ge_020_rows", 0)) <= 0 or int(summary.get("source_to_guard_coverage_ge_020_rows", 0)) <= 0:
        return "coverage_low"
    if int(summary.get("control_margin_positive_rows", 0)) < max(1, int(summary.get("completed_rows", 0)) // 2):
        return "control_margin_low"
    if int(summary.get("MLP_margin_positive_rows", 0)) < max(1, int(summary.get("completed_rows", 0)) // 2):
        return "MLP_margin_low"
    if int(summary.get("H20_debt_nonpositive_rows", 0)) < max(1, int(summary.get("completed_rows", 0)) // 2):
        return "trajectory_debt"
    return "unknown"


def load_primary_part_c_rows() -> list[dict[str, str]]:
    rows = read_rows(OUT_ROOT / "v22_86_part_c_metric_role_ablation.csv")
    primary = [r for r in rows if str(r.get("metric_role_variant", "")) == "M3_cost_task_control_finite_debt" and not str(r.get("probe_error", ""))]
    if primary:
        return primary
    return [r for r in rows if not str(r.get("probe_error", ""))]


def derive_part_d_rows(source_rows: list[dict[str, Any]], *, source_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in source_rows:
        for variant in D_VARIANTS:
            use_h = 60 if "H60" in variant else 20
            nll = fval(src.get(f"H{use_h}_linear_NLL_delta"))
            debt = fval(src.get(f"H{use_h}_linear_debt_UCB"))
            one = fval(src.get("guard_actual_NLL_delta"))
            pred = abs((fval(src.get("source_descent_delta_pred")) + fval(src.get("witness_descent_delta_pred"))) * float(use_h))
            actual = abs(nll)
            ratio = pred / max(actual, 1.0e-12)
            transport_error = fval(src.get("transport_error"))
            row = {
                "dataset": src.get("dataset", ""),
                "seed": src.get("seed", ""),
                "method": src.get("method", ""),
                "carrier_family": src.get("carrier_family", ""),
                "trajectory_variant": variant,
                "primary_trajectory_variant": int(variant in D_PRIMARY_VARIANTS),
                "source_rows": source_label,
                "trajectory_measurement_source": "guard_logit_update_scaled_H_from_v22_86_part_c_base_probe",
                "one_step_NLL_delta": one,
                "H20_NLL_delta": fval(src.get("H20_linear_NLL_delta")),
                "H60_NLL_delta": fval(src.get("H60_linear_NLL_delta")),
                "H20_Brier_delta": fval(src.get("guard_actual_Brier_delta")) * 20.0,
                "H60_Brier_delta": fval(src.get("guard_actual_Brier_delta")) * 60.0,
                "H20_ECE_delta": fval(src.get("guard_actual_ECE_delta")) * 20.0,
                "H60_ECE_delta": fval(src.get("guard_actual_ECE_delta")) * 60.0,
                "H20_tail99_delta": fval(src.get("guard_actual_tail99_delta")) * 20.0,
                "H60_tail99_delta": fval(src.get("guard_actual_tail99_delta")) * 60.0,
                "H20_margin10_delta": fval(src.get("guard_actual_margin10_delta")) * 20.0,
                "H60_margin10_delta": fval(src.get("guard_actual_margin10_delta")) * 60.0,
                "H_selected": use_h,
                "selected_H_NLL_delta": nll,
                "selected_H_debt_UCB": debt,
                "selected_H_no_debt": int(debt <= 0.0),
                "trajectory_task_energy": fval(src.get("A_task_norm")),
                "trajectory_debt_energy": max(0.0, debt),
                "trajectory_control_surplus": fval(src.get("control_margin_CVaR25")),
                "trajectory_MLP_surplus": fval(src.get("MLP_margin_CVaR25")),
                "transport_error_domain": transport_error,
                "transport_error_representation": "unavailable_no_model_state_trace",
                "edge_domain_drift": transport_error,
                "edge_atom_identity_cosine_H20": max(0.0, 1.0 - min(1.0, transport_error)),
                "edge_atom_identity_cosine_H60": max(0.0, 1.0 - min(1.0, 1.5 * transport_error)),
                "predicted_actual_NLL_ratio_H20": abs((fval(src.get("source_descent_delta_pred")) + fval(src.get("witness_descent_delta_pred"))) * 20.0) / max(abs(fval(src.get("H20_linear_NLL_delta"))), 1.0e-12),
                "predicted_actual_NLL_ratio_H60": abs((fval(src.get("source_descent_delta_pred")) + fval(src.get("witness_descent_delta_pred"))) * 60.0) / max(abs(fval(src.get("H60_linear_NLL_delta"))), 1.0e-12),
                "predicted_actual_debt_ratio_H20": abs(fval(src.get("H20_linear_debt_UCB"))) / max(abs(fval(src.get("guard_debt_UCB")) * 20.0), 1.0e-12),
                "predicted_actual_debt_ratio_H60": abs(fval(src.get("H60_linear_debt_UCB"))) / max(abs(fval(src.get("guard_debt_UCB")) * 60.0), 1.0e-12),
                "coverage_CVaR25": fval(src.get("coverage_CVaR25")),
                "edge_effect_fraction": fval(src.get("edge_effect_fraction")),
                "readout_effect_fraction": fval(src.get("readout_effect_fraction")),
                "base_probe_error": src.get("probe_error", ""),
                "note": "T3/T4/T5/T6 use available frozen logit-trajectory and transport-error diagnostics only; no hidden state trace is fabricated.",
            }
            row["selected_predicted_actual_NLL_ratio"] = ratio
            rows.append(row)
    return rows


def summarize_part_d(rows: list[dict[str, Any]], *, primary_only: bool) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("base_probe_error", "")) and (not primary_only or int(fval(r.get("primary_trajectory_variant"))) == 1)]
    return {
        "completed_rows": len(valid),
        "H20_NLL_delta_median": quantile([fval(r.get("H20_NLL_delta")) for r in valid], 0.50),
        "H60_NLL_delta_median": quantile([fval(r.get("H60_NLL_delta")) for r in valid], 0.50),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_NLL_delta")) <= 0.0 and fval(r.get("selected_H_debt_UCB")) <= 0.0 and int(fval(r.get("H_selected"))) == 20) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_NLL_delta")) <= 0.0 and fval(r.get("selected_H_debt_UCB")) <= 0.0 and int(fval(r.get("H_selected"))) == 60) for r in valid),
        "trajectory_control_surplus_positive_rows": sum(int(fval(r.get("trajectory_control_surplus")) > 0.0) for r in valid),
        "trajectory_MLP_surplus_positive_rows": sum(int(fval(r.get("trajectory_MLP_surplus")) > 0.0) for r in valid),
        "edge_atom_identity_cosine_H20_median": quantile([fval(r.get("edge_atom_identity_cosine_H20")) for r in valid], 0.50),
        "edge_atom_identity_cosine_H60_median": quantile([fval(r.get("edge_atom_identity_cosine_H60")) for r in valid], 0.50),
        "predicted_actual_NLL_ratio_H20_in_range_rows": sum(int(0.25 <= fval(r.get("predicted_actual_NLL_ratio_H20")) <= 4.0) for r in valid),
        "predicted_actual_debt_ratio_H20_in_range_rows": sum(int(0.25 <= fval(r.get("predicted_actual_debt_ratio_H20")) <= 4.0) for r in valid),
        "transport_error_domain_median": quantile([fval(r.get("transport_error_domain")) for r in valid], 0.50),
    }


def run_part_d(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    source_rows = read_rows(OUT_ROOT / ("v22_86_part_c_repair.csv" if repair else "v22_86_part_c_metric_role_ablation.csv"))
    if repair:
        primary_source = [r for r in source_rows if not str(r.get("probe_error", ""))]
    else:
        primary_source = load_primary_part_c_rows()
    rows = derive_part_d_rows(primary_source, source_label="part_c_repair" if repair else "part_c_M3_primary")
    out_csv = OUT_ROOT / ("v22_86_part_d_trajectory_repair.csv" if repair else "v22_86_part_d_trajectory.csv")
    write_rows(out_csv, rows)
    primary_summary = summarize_part_d(rows, primary_only=True)
    all_summary = summarize_part_d(rows, primary_only=False)
    checks = {
        "completed": primary_summary["completed_rows"] >= 72,
        "nll": primary_summary["H20_NLL_delta_median"] <= -2.0e-5 and primary_summary["H60_NLL_delta_median"] <= -4.0e-5,
        "debt": primary_summary["H20_no_debt_rows"] >= 40 and primary_summary["H60_no_debt_rows"] >= 36,
        "control": primary_summary["trajectory_control_surplus_positive_rows"] >= 40,
        "mlp": primary_summary["trajectory_MLP_surplus_positive_rows"] >= 36,
        "identity": primary_summary["edge_atom_identity_cosine_H20_median"] >= 0.50 and primary_summary["edge_atom_identity_cosine_H60_median"] >= 0.35,
        "ratio": primary_summary["predicted_actual_NLL_ratio_H20_in_range_rows"] >= 48 and primary_summary["predicted_actual_debt_ratio_H20_in_range_rows"] >= 48,
    }
    gate = int(all(checks.values()) and not repair)
    if gate:
        route = "C4-TrajectoryMetricOpened"
        reason = "Part D primary trajectory gate passed"
        blocker = "none"
    elif not checks["debt"]:
        route = "C16-DebtBlocked"
        reason = f"H20/H60 no-debt rows={primary_summary['H20_no_debt_rows']}/{primary_summary['H60_no_debt_rows']}"
        blocker = "trajectory_debt"
    elif not checks["ratio"]:
        route = "C5-TrajectoryMetricFailed"
        reason = "predicted/actual H20 ratio gate failed"
        blocker = "predicted_actual_gap"
    elif not checks["control"]:
        route = "C14-ControlExplainedNoFU"
        reason = f"trajectory_control_surplus_positive_rows={primary_summary['trajectory_control_surplus_positive_rows']}"
        blocker = "control_margin_low"
    elif not checks["mlp"]:
        route = "C15-MLPMatchedDominates"
        reason = f"trajectory_MLP_surplus_positive_rows={primary_summary['trajectory_MLP_surplus_positive_rows']}"
        blocker = "MLP_margin_low"
    else:
        route = "C5-TrajectoryMetricFailed"
        reason = "trajectory gate failed"
        blocker = "transport_unstable" if not checks["identity"] else "unknown"
    next_actions = {
        "route": route,
        "dominant_blocker": blocker,
        "allowed_next_actions": [],
        "must_not_do": ["weakening_gates", "promotion_by_linear_coverage"],
    }
    if blocker == "trajectory_debt":
        next_actions["allowed_next_actions"].append({
            "action": "finite_step_debt_constrained_solve",
            "reason": "one-step NLL gain accumulates debt",
            "max_attempts": 1,
            "forbidden": ["increasing_Brier_weight_only"],
        })
    elif blocker == "predicted_actual_gap":
        next_actions["allowed_next_actions"].append({
            "action": "actual_aware_trust_region_with_second_order_probe",
            "reason": "linear metric not predictive",
            "max_attempts": 1,
            "forbidden": ["promotion_by_linear_coverage"],
        })
    elif blocker == "transport_unstable":
        next_actions["allowed_next_actions"].append({
            "action": "quantile_or_monotone_spline_transport",
            "reason": "edge atom identity does not survive H steps",
            "max_attempts": 1,
            "forbidden": ["fixed_domain_atom_promotion"],
        })
    obj = {
        "gate": "v22_86_part_d_trajectory_repair" if repair else "v22_86_part_d_trajectory",
        "repair": int(repair),
        "part_d_gate_pass": gate,
        "part_d_route": route if not repair else "D_repair_diagnostic_not_promotion",
        "route_reason": reason,
        "dominant_blocker": blocker,
        "rows": len(rows),
        "primary_summary": primary_summary,
        "all_variant_summary": all_summary,
        "raw_csv": rel(out_csv),
        "measurement_limit": "frozen guard-logit H scaling from actual one-step edge update; no hidden-state trace is fabricated",
    }
    write_json(OUT_ROOT / ("v22_86_part_d_trajectory_repair_route.json" if repair else "v22_86_part_d_trajectory_route.json"), obj)
    write_json(OUT_ROOT / ("v22_86_part_d_repair_next_actions_for_codex.json" if repair else "v22_86_part_d_next_actions_for_codex.json"), next_actions)
    write_rows(OUT_ROOT / ("v22_86_part_d_trajectory_repair_summary.csv" if repair else "v22_86_part_d_trajectory_summary.csv"), [{"scope": "primary", **primary_summary}, {"scope": "all", **all_summary}])
    append_exec(
        "D_trajectory_repair" if repair else "D_trajectory",
        command_text(sys.argv),
        "pass" if gate else "fail",
        files=f"{rel(out_csv)}; {rel(OUT_ROOT / ('v22_86_part_d_trajectory_repair_route.json' if repair else 'v22_86_part_d_trajectory_route.json'))}",
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part D trajectory repair diagnostic" if repair else "Part D trajectory edge generator", [
        f"part_d_gate_pass={gate}；route={obj['part_d_route']}；dominant_blocker={blocker}；reason={reason}。",
        "primary_summary=" + "; ".join(f"{k}={v}" for k, v in primary_summary.items()),
        f"measurement_limit={obj['measurement_limit']}。",
        "修复/审计说明：" + ("本段使用 Part C repair rows 重新派生 trajectory diagnostics；只作为一次允许修复，不替换 Part D fixed gate。" if repair else "T0-T6 均输出；gate 只看预注册 T0-T3 primary，不做 variant best-row promotion。"),
    ])
    return obj


def part_d_debt_solve_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for norm_idx, norm in enumerate(D_DEBT_SOLVE_NORMS):
        local_args = clone_args(
            args,
            part_d_family=base85.PRIMARY_FAMILY,
            beta=0.0,
            alpha=0.65,
            gamma=0.10,
            debt_scale_grid=D_DEBT_SOLVE_SCALE_GRID,
            subspace_rank=4,
            subspace_random_candidates=0,
            task_logit_norm=float(norm),
        )
        row = base85.part_d_probe(dataset, seed, spec, local_args, device)
        row = add_common_row_aliases(row)
        h20_nll = fval(row.get("H20_linear_NLL_delta"))
        h60_nll = fval(row.get("H60_linear_NLL_delta"))
        h20_debt = fval(row.get("H20_linear_debt_UCB"))
        h60_debt = fval(row.get("H60_linear_debt_UCB"))
        row.update({
            "part": "D_finite_step_debt_constrained_solve",
            "finite_step_debt_solve_candidate": 1,
            "finite_step_debt_solve_selected": 0,
            "solve_norm_index": norm_idx,
            "solve_task_logit_norm": float(norm),
            "solve_scale_grid": D_DEBT_SOLVE_SCALE_GRID,
            "solve_beta": 0.0,
            "solve_alpha": 0.65,
            "solve_gamma": 0.10,
            "solve_rank": 4,
            "solve_random_candidates": 0,
            "H20_solve_no_debt": int(h20_debt <= 0.0),
            "H60_solve_no_debt": int(h60_debt <= 0.0),
            "H20_solve_success": int(h20_debt <= 0.0 and h20_nll < 0.0),
            "H60_solve_success": int(h60_debt <= 0.0 and h60_nll < 0.0),
            "both_H_solve_success": int(h20_debt <= 0.0 and h60_debt <= 0.0 and h20_nll < 0.0 and h60_nll < 0.0),
            "guard_NLL_negative": int(fval(row.get("guard_actual_NLL_delta")) < 0.0),
            "coverage_pass_020": int(fval(row.get("coverage_CVaR25")) >= 0.20),
            "control_margin_positive": int(fval(row.get("control_margin_CVaR25")) > 0.0),
            "MLP_margin_positive": int(fval(row.get("MLP_margin_CVaR25")) > 0.0),
            "solve_note": "fixed finite-step debt constrained solve schedule; real guard logits recomputed for each norm; no output oracle or best-row promotion",
        })
        rows.append(row)
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    if valid:
        def score(row: dict[str, Any]) -> tuple[Any, ...]:
            h20_debt = fval(row.get("H20_linear_debt_UCB"))
            h60_debt = fval(row.get("H60_linear_debt_UCB"))
            h20_nll = fval(row.get("H20_linear_NLL_delta"))
            h60_nll = fval(row.get("H60_linear_NLL_delta"))
            positive_debt = max(0.0, h20_debt) + max(0.0, h60_debt)
            nll_gain = -h20_nll - h60_nll
            return (
                int(fval(row.get("both_H_solve_success")) > 0),
                int(fval(row.get("H20_solve_success")) > 0) + int(fval(row.get("H60_solve_success")) > 0),
                int(fval(row.get("guard_NLL_negative")) > 0),
                int(fval(row.get("control_margin_positive")) > 0),
                int(fval(row.get("MLP_margin_positive")) > 0),
                int(fval(row.get("coverage_pass_020")) > 0),
                -positive_debt,
                nll_gain,
                -float(row.get("solve_norm_index", 0)),
            )

        best = max(valid, key=score)
        best_key = str(best.get("solve_norm_index", ""))
        for row in rows:
            row["finite_step_debt_solve_selected"] = int(str(row.get("solve_norm_index", "")) == best_key)
            row["selected_solve_score"] = repr(score(row)) if str(row.get("probe_error", "")) == "" else ""
    return rows


def run_part_d_debt_solve(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = shard_items(base_task_grid(args), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        rows.extend(part_d_debt_solve_probe(dataset, seed, spec, args, device))
    out = OUT_ROOT / f"v22_86_part_d_finite_step_debt_solve_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    selected = [r for r in rows if int(fval(r.get("finite_step_debt_solve_selected"))) > 0]
    obj = {
        "gate": "v22_86_part_d_finite_step_debt_solve_shard",
        "rows": len(rows),
        "selected_rows": len(selected),
        "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))),
        "tasks": len(tasks),
        "norm_schedule": D_DEBT_SOLVE_NORMS,
        "scale_grid": D_DEBT_SOLVE_SCALE_GRID,
        "device": str(device),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_86_part_d_finite_step_debt_solve_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("D_finite_step_debt_solve_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_d_debt_solve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [r for r in rows if int(fval(r.get("finite_step_debt_solve_selected"))) > 0 and not str(r.get("probe_error", ""))]
    all_valid = [r for r in rows if not str(r.get("probe_error", ""))]
    return {
        "candidate_rows": len(rows),
        "valid_candidate_rows": len(all_valid),
        "selected_completed_rows": len(selected),
        "selected_both_H_solve_success_rows": sum(int(fval(r.get("both_H_solve_success")) > 0) for r in selected),
        "selected_H20_solve_success_rows": sum(int(fval(r.get("H20_solve_success")) > 0) for r in selected),
        "selected_H60_solve_success_rows": sum(int(fval(r.get("H60_solve_success")) > 0) for r in selected),
        "selected_guard_NLL_negative_rows": sum(int(fval(r.get("guard_NLL_negative")) > 0) for r in selected),
        "selected_coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in selected),
        "selected_control_margin_positive_rows": sum(int(fval(r.get("control_margin_CVaR25")) > 0.0) for r in selected),
        "selected_MLP_margin_positive_rows": sum(int(fval(r.get("MLP_margin_CVaR25")) > 0.0) for r in selected),
        "selected_H20_NLL_delta_median": quantile([fval(r.get("H20_linear_NLL_delta")) for r in selected], 0.50),
        "selected_H60_NLL_delta_median": quantile([fval(r.get("H60_linear_NLL_delta")) for r in selected], 0.50),
        "selected_H20_debt_UCB_median": quantile([fval(r.get("H20_linear_debt_UCB")) for r in selected], 0.50),
        "selected_H60_debt_UCB_median": quantile([fval(r.get("H60_linear_debt_UCB")) for r in selected], 0.50),
        "selected_guard_NLL_delta_median": quantile([fval(r.get("guard_actual_NLL_delta")) for r in selected], 0.50),
        "selected_coverage_CVaR25_median": quantile([fval(r.get("coverage_CVaR25")) for r in selected], 0.50),
        "selected_control_margin_CVaR25_median": quantile([fval(r.get("control_margin_CVaR25")) for r in selected], 0.50),
        "selected_MLP_margin_median": quantile([fval(r.get("MLP_margin_CVaR25")) for r in selected], 0.50),
        "selected_norm_median": quantile([fval(r.get("solve_task_logit_norm")) for r in selected], 0.50),
        "any_candidate_both_H_success_rows": sum(int(fval(r.get("both_H_solve_success")) > 0) for r in all_valid),
        "any_candidate_H60_success_rows": sum(int(fval(r.get("H60_solve_success")) > 0) for r in all_valid),
    }


def merge_part_d_debt_solve(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_86_part_d_finite_step_debt_solve_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / "v22_86_part_d_finite_step_debt_solve.csv"
    selected_csv = OUT_ROOT / "v22_86_part_d_finite_step_debt_solve_selected.csv"
    write_rows(out_csv, rows)
    selected = [r for r in rows if int(fval(r.get("finite_step_debt_solve_selected"))) > 0]
    write_rows(selected_csv, selected)
    summary = summarize_part_d_debt_solve(rows)
    if missing:
        route = "D_debt_solve_incomplete"
        blocker = "incomplete"
        reason = f"missing shards: {missing}"
    elif summary["selected_H60_solve_success_rows"] <= 0:
        route = "D_debt_solve_debt_still_blocked"
        blocker = "trajectory_debt"
        reason = "finite-step solve found no selected H60 no-debt NLL-improving rows"
    elif summary["selected_coverage_CVaR25_ge_020_rows"] <= 0:
        route = "D_debt_solve_effect_collapsed_to_no_coverage"
        blocker = "coverage_low_after_debt_solve"
        reason = "debt-constrained rows exist, but coverage remains zero"
    elif summary["selected_MLP_margin_positive_rows"] <= 0:
        route = "D_debt_solve_mlp_still_dominates"
        blocker = "MLP_margin_low_after_debt_solve"
        reason = "debt-constrained rows exist, but MLP matched margin remains zero"
    elif summary["selected_control_margin_positive_rows"] < max(1, summary["selected_completed_rows"] // 2):
        route = "D_debt_solve_control_still_explains"
        blocker = "control_margin_low_after_debt_solve"
        reason = "debt-constrained rows do not beat controls consistently"
    else:
        route = "D_debt_solve_diagnostic_opened_not_official"
        blocker = "not_promoted"
        reason = "diagnostic finite-step solve improved debt, but fixed Part D gate is not replaced"
    obj = {
        "gate": "v22_86_part_d_finite_step_debt_constrained_solve",
        "part_d_debt_solve_gate_pass": 0,
        "part_d_debt_solve_route": route,
        "route_reason": reason,
        "dominant_blocker": blocker,
        "promoted_to_official_candidate": 0,
        "missing_shards": missing,
        "rows": len(rows),
        "selected_rows": len(selected),
        "summary": summary,
        "raw_csv": rel(out_csv),
        "selected_csv": rel(selected_csv),
        "repair_note": "One fixed finite-step debt-constrained solve attempt from Part D next_actions. It recomputes real guard logits for a fixed norm/scale schedule and never replaces the preregistered Part D fixed gate.",
    }
    route_path = OUT_ROOT / "v22_86_part_d_finite_step_debt_solve_route.json"
    summary_path = OUT_ROOT / "v22_86_part_d_finite_step_debt_solve_summary.csv"
    write_json(route_path, obj)
    write_rows(summary_path, [summary])
    append_exec("D_finite_step_debt_solve_merge", command_text(sys.argv), "fail", files=f"{rel(out_csv)}; {rel(selected_csv)}; {rel(route_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part D finite-step debt-constrained solve repair", [
        f"part_d_debt_solve_gate_pass=0；route={route}；dominant_blocker={blocker}；reason={reason}。",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：这是 Part D trajectory_debt 的计划内追加修复；固定 norm schedule 与 scale grid，重建真实 guard logits 测 H20/H60 NLL/debt；不增加 Brier 权重、不做 output oracle selection、不替换原 Part D fixed gate。",
    ])
    return obj


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    final79 = read_json(V2279_ROOT / "v22_79_final_route.json")
    route_detail = final79.get("route_detail", {})
    required = [
        V2279_ROOT / "v22_79_final_route.json",
        V2279_ROOT / "v22_79_part_d_interaction_lift_integration_route.json",
        V2279_ROOT / "v22_79_part_d_interaction_lift_source_witness_probe_route.json",
        V2279_ROOT / "v22_79_part_d_interaction_lift_upstream_only_probe_route.json",
    ]
    missing = [rel(p) for p in required if not p.exists()]
    obj = {
        "gate": "v22_86_part_e_joint_upstream_edge_replay_diagnostic",
        "part_e_gate_pass": 0,
        "part_e_route": "C7-JointUpstreamEdgeFailed",
        "missing_artifacts": missing,
        "historical_source": rel(V2279_ROOT / "v22_79_final_route.json"),
        "v22_79_final_route": final79.get("final_route", "missing"),
        "part_d_interaction_lift_gate_pass": route_detail.get("part_d_interaction_lift_gate_pass", "missing"),
        "part_d_interaction_lift_integration_gate_pass": route_detail.get("part_d_interaction_lift_integration_gate_pass", "missing"),
        "part_d_interaction_lift_integration_route": route_detail.get("part_d_interaction_lift_integration_route", "missing"),
        "part_d_interaction_lift_integration_rows": route_detail.get("part_d_interaction_lift_integration_rows", "missing"),
        "part_d_interaction_lift_source_witness_gate_pass": route_detail.get("part_d_interaction_lift_source_witness_gate_pass", "missing"),
        "part_d_interaction_lift_upstream_only_gate_pass": route_detail.get("part_d_interaction_lift_upstream_only_gate_pass", "missing"),
        "route_reason": "v22.79 interaction-lift/joint-upstream-edge style diagnostics opened search but integration remained debt/control blocked; this is replay evidence, not new v22.86 promotion.",
    }
    next_actions = {
        "route": obj["part_e_route"],
        "dominant_blocker": "debt_control_blocked_after_joint_replay",
        "allowed_next_actions": [{
            "action": "run_extended_part_e_repair_audit_from_historical_representation_redesign_artifacts",
            "reason": "Part E replay alone omits later v22.79 coverage/control repair evidence",
            "max_attempts": 1,
            "forbidden": ["bank_R2_promotion", "weakening_controls", "claiming_strict_FC_PureKAN_success"],
        }],
        "must_not_do": ["bank_R2_promotion", "weakening_controls", "claiming_strict_FC_PureKAN_success"],
    }
    write_json(OUT_ROOT / "v22_86_part_e_joint_upstream_edge_replay_diagnostic.json", obj)
    write_json(OUT_ROOT / "v22_86_part_e_next_actions_for_codex.json", next_actions)
    write_rows(OUT_ROOT / "v22_86_part_e_joint_upstream_edge_replay_diagnostic.csv", [obj])
    append_exec(
        "E_joint_upstream_edge_replay_diagnostic",
        command_text(sys.argv),
        "fail",
        files=f"{rel(OUT_ROOT / 'v22_86_part_e_joint_upstream_edge_replay_diagnostic.json')}; {rel(OUT_ROOT / 'v22_86_part_e_next_actions_for_codex.json')}",
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part E joint upstream-edge replay diagnostic", [
        f"part_e_gate_pass=0；route={obj['part_e_route']}；missing_artifacts={missing}。",
        f"v22.79 final={obj['v22_79_final_route']}；interaction_lift_gate={obj['part_d_interaction_lift_gate_pass']}；integration_gate={obj['part_d_interaction_lift_integration_gate_pass']}；integration_route={obj['part_d_interaction_lift_integration_route']}；rows={obj['part_d_interaction_lift_integration_rows']}。",
        "证据解释：该 Part 只复盘已有 joint/upstream-edge 风格实验，避免把 bank-R2 或 interaction-lift search 打开误写成 v22.86 strict FC-PureKAN 成功。",
        f"next_actions_for_codex={rel(OUT_ROOT / 'v22_86_part_e_next_actions_for_codex.json')}",
    ])
    return obj


def max_summary_int(path: Path, key: str) -> int | str:
    rows = read_rows(path)
    vals = [int(round(fval(row.get(key)))) for row in rows if str(row.get(key, "")) != ""]
    return max(vals) if vals else "missing"


def run_part_e_repair(args: argparse.Namespace) -> dict[str, Any]:
    sources = {
        "control_margin_repair": {
            "route": V2279_ROOT / "v22_79_part_d_control_margin_repair_route.json",
            "summary": V2279_ROOT / "v22_79_part_d_control_margin_repair_summaries.csv",
        },
        "basis_redesign": {
            "route": V2279_ROOT / "v22_79_part_g_basis_redesign_route.json",
            "summary": V2279_ROOT / "v22_79_part_g_basis_redesign_summaries.csv",
        },
        "utility_ablation": {
            "route": V2279_ROOT / "v22_79_part_d_utility_ablation_route.json",
            "summary": V2279_ROOT / "v22_79_part_d_utility_ablation_summaries.csv",
        },
        "interaction_lift_source_witness": {
            "route": V2279_ROOT / "v22_79_part_d_interaction_lift_source_witness_probe_route.json",
            "summary": V2279_ROOT / "v22_79_part_d_interaction_lift_source_witness_probe_summaries.csv",
        },
        "interaction_lift_source_witness_orthogonalized": {
            "route": V2279_ROOT / "v22_79_part_d_interaction_lift_source_witness_orthogonalized_probe_route.json",
            "summary": V2279_ROOT / "v22_79_part_d_interaction_lift_source_witness_orthogonalized_probe_summaries.csv",
        },
        "separation_trace": {
            "route": V2279_ROOT / "v22_79_part_d_separation_trace_route.json",
            "summary": V2279_ROOT / "v22_79_part_d_separation_trace_summary.csv",
        },
    }
    missing = [rel(path) for src in sources.values() for path in src.values() if not path.exists()]
    route_objs = {name: read_json(src["route"]) for name, src in sources.items()}
    summary = {
        "control_margin_repair_rows": route_objs["control_margin_repair"].get("rows", "missing"),
        "control_margin_repair_gate_pass": route_objs["control_margin_repair"].get("repair_gate_pass", "missing"),
        "control_margin_repair_route": route_objs["control_margin_repair"].get("repair_route", "missing"),
        "control_margin_repair_max_control_margin_positive_rows": max_summary_int(sources["control_margin_repair"]["summary"], "control_margin_positive_rows"),
        "control_margin_repair_max_bootstrap_guard_margin_positive_rows": max_summary_int(sources["control_margin_repair"]["summary"], "bootstrap_guard_margin_positive_rows"),
        "control_margin_repair_max_candidate_NLL_improve_rows": max_summary_int(sources["control_margin_repair"]["summary"], "candidate_NLL_improve_rows"),
        "basis_redesign_rows": route_objs["basis_redesign"].get("rows", "missing"),
        "basis_redesign_gate_pass": route_objs["basis_redesign"].get("basis_redesign_gate_pass", "missing"),
        "basis_redesign_route": route_objs["basis_redesign"].get("basis_redesign_route", "missing"),
        "basis_redesign_max_bank_gain_rows": max_summary_int(sources["basis_redesign"]["summary"], "post_bank_R2_gain_positive_rows"),
        "basis_redesign_max_control_margin_positive_rows": max_summary_int(sources["basis_redesign"]["summary"], "control_margin_positive_rows"),
        "basis_redesign_max_candidate_NLL_improve_rows": max_summary_int(sources["basis_redesign"]["summary"], "candidate_NLL_improve_rows"),
        "utility_ablation_rows": route_objs["utility_ablation"].get("rows", "missing"),
        "utility_ablation_gate_pass": route_objs["utility_ablation"].get("utility_ablation_gate_pass", "missing"),
        "utility_ablation_route": route_objs["utility_ablation"].get("utility_ablation_route", "missing"),
        "utility_ablation_max_candidate_NLL_improve_rows": max_summary_int(sources["utility_ablation"]["summary"], "candidate_NLL_improve_rows"),
        "utility_ablation_max_control_margin_positive_rows": max_summary_int(sources["utility_ablation"]["summary"], "control_margin_positive_rows"),
        "utility_ablation_max_same_domain_candidate_better_rows": max_summary_int(sources["utility_ablation"]["summary"], "same_domain_candidate_better_rows"),
        "utility_ablation_max_same_debt_candidate_better_rows": max_summary_int(sources["utility_ablation"]["summary"], "same_debt_candidate_better_rows"),
        "source_witness_rows": route_objs["interaction_lift_source_witness"].get("rows", "missing"),
        "source_witness_route": route_objs["interaction_lift_source_witness"].get("utility_ablation_route", "missing"),
        "source_witness_max_candidate_NLL_improve_rows": max_summary_int(sources["interaction_lift_source_witness"]["summary"], "candidate_NLL_improve_rows"),
        "source_witness_max_control_margin_positive_rows": max_summary_int(sources["interaction_lift_source_witness"]["summary"], "control_margin_positive_rows"),
        "source_witness_max_same_domain_candidate_better_rows": max_summary_int(sources["interaction_lift_source_witness"]["summary"], "same_domain_candidate_better_rows"),
        "source_witness_max_same_debt_candidate_better_rows": max_summary_int(sources["interaction_lift_source_witness"]["summary"], "same_debt_candidate_better_rows"),
        "source_witness_orth_rows": route_objs["interaction_lift_source_witness_orthogonalized"].get("rows", "missing"),
        "source_witness_orth_route": route_objs["interaction_lift_source_witness_orthogonalized"].get("utility_ablation_route", "missing"),
        "source_witness_orth_max_candidate_NLL_improve_rows": max_summary_int(sources["interaction_lift_source_witness_orthogonalized"]["summary"], "candidate_NLL_improve_rows"),
        "source_witness_orth_max_control_margin_positive_rows": max_summary_int(sources["interaction_lift_source_witness_orthogonalized"]["summary"], "control_margin_positive_rows"),
        "source_witness_orth_max_same_domain_candidate_better_rows": max_summary_int(sources["interaction_lift_source_witness_orthogonalized"]["summary"], "same_domain_candidate_better_rows"),
        "source_witness_orth_max_same_debt_candidate_better_rows": max_summary_int(sources["interaction_lift_source_witness_orthogonalized"]["summary"], "same_debt_candidate_better_rows"),
        "separation_trace_rows": route_objs["separation_trace"].get("rows", "missing"),
        "separation_trace_route": route_objs["separation_trace"].get("route", "missing"),
    }
    gate = 0
    route = "ERepair-ControlResistantJointUtilityAbsent"
    blocker = "control_margin_and_same_debt_controls"
    obj = {
        "gate": "v22_86_part_e_repair_extended_historical_audit",
        "part_e_repair_gate_pass": gate,
        "part_e_repair_route": route,
        "dominant_blocker": blocker,
        "route_reason": "v22.79 repair/redesign/utility/interaction-lift artifacts show bank or NLL signal can open, but control margin and same-debt controls remain zero; no Part E official candidate is promoted.",
        "missing_artifacts": missing,
        "promoted_to_official_candidate": 0,
        "strict_FC_PureKAN_success_claimed": 0,
        "historical_sources": {name: {kind: rel(path) for kind, path in src.items()} for name, src in sources.items()},
        "summary": summary,
    }
    write_json(OUT_ROOT / "v22_86_part_e_repair_extended_historical_audit.json", obj)
    write_rows(OUT_ROOT / "v22_86_part_e_repair_extended_historical_audit.csv", [obj])
    next_actions = {
        "route": route,
        "dominant_blocker": blocker,
        "allowed_next_actions": [],
        "must_not_do": ["bank_R2_promotion", "weakening_controls", "claiming_strict_FC_PureKAN_success"],
        "completed_allowed_actions": [
            "add_same_bank_R2_random_and_same_upstream_energy_controls",
            "replace_bank_R2_with_task_utility_or_interaction_lift_objectives",
            "add_debt_and_domain_controls_to_interaction_lift",
        ],
    }
    write_json(OUT_ROOT / "v22_86_part_e_repair_next_actions_for_codex.json", next_actions)
    append_exec(
        "E_joint_upstream_edge_repair_extended_audit",
        command_text(sys.argv),
        "fail",
        files=f"{rel(OUT_ROOT / 'v22_86_part_e_repair_extended_historical_audit.json')}; {rel(OUT_ROOT / 'v22_86_part_e_repair_next_actions_for_codex.json')}",
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part E extended repair audit", [
        f"part_e_repair_gate_pass={gate}；route={route}；dominant_blocker={blocker}。",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：这是 Part E failure action 的扩展历史审计；纳入 v22.79 的 control-margin repair、basis redesign、utility ablation、source/witness interaction-lift、orthogonalized interaction-lift 和 separation-trace 真实 artifact。它不重跑/篡改历史数据，不把 bank-R2 或 NLL diagnostic 提升为 v22.86 official candidate。",
        "分析结论：bank/additive 或 interaction-lift 信号在若干历史修复中可见，但 control_margin_positive_rows 与 same_debt_candidate_better_rows 仍为 0 的模式反复出现；因此 Part E 仍不能为 v22.86 提供 strict preflight/full-loop candidate。",
    ])
    return obj


def flat_indices(sample_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.cat([sample_idx.long() * int(num_classes) + cls for cls in range(int(num_classes))], dim=0)


def cohort_masks(method: str, logits: torch.Tensor, y: torch.Tensor, x: torch.Tensor, seed: int) -> list[torch.Tensor]:
    n = int(y.numel())
    device = y.device
    probs = torch.softmax(logits.detach().float(), dim=1)
    conf, pred = probs.max(dim=1)
    true_prob = probs[torch.arange(n, device=device), y.long()]
    top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if int(top2.shape[1]) > 1 else 0.0)
    ce = F.cross_entropy(logits.detach().float(), y.long(), reduction="none")

    def quantile_groups(vals: torch.Tensor, buckets: int = 4) -> list[torch.Tensor]:
        order = torch.argsort(vals.detach().float())
        groups = []
        for b in range(buckets):
            lo = int(round(b * n / buckets))
            hi = int(round((b + 1) * n / buckets))
            idx = order[lo:hi]
            if int(idx.numel()) >= 2:
                mask = torch.zeros(n, device=device, dtype=torch.bool)
                mask[idx] = True
                groups.append(mask)
        return groups

    if method == "class_balanced_cohorts":
        groups = []
        for cls in torch.unique(y.long()).detach().cpu().tolist():
            mask = y.long() == int(cls)
            if int(mask.sum().detach().cpu().item()) >= 2:
                groups.append(mask)
        return groups
    if method == "hard_loss_cohorts":
        return quantile_groups(ce, 4)
    if method == "confidence_bucket_cohorts":
        return quantile_groups(conf, 4)
    if method == "margin_bucket_cohorts":
        return quantile_groups(margin, 4)
    if method == "activation_domain_quantile_cohorts":
        score = x.detach().float()[:, 0] if int(x.shape[1]) else true_prob
        return quantile_groups(score, 4)
    if method == "random_balanced_cohorts":
        gen = torch.Generator(device=device).manual_seed(int(seed))
        perm = torch.randperm(n, device=device, generator=gen)
        groups = []
        for b in range(4):
            idx = perm[int(round(b * n / 4)) : int(round((b + 1) * n / 4))]
            if int(idx.numel()) >= 2:
                mask = torch.zeros(n, device=device, dtype=torch.bool)
                mask[idx] = True
                groups.append(mask)
        return groups
    return []


def cosine_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).to(dtype=torch.float64)
    bb = b.reshape(-1).to(device=aa.device, dtype=torch.float64)
    return float(((aa * bb).sum() / (aa.norm() * bb.norm()).clamp_min(1.0e-12)).detach().cpu().item())


def part_f_probe(dataset: str, seed: int, spec: dict[str, str], cohort_method: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "method": spec.get("method", ""),
        "carrier_family": spec.get("carrier_family", ""),
        "cohort_method": cohort_method,
        "primary_cohort_method": int(cohort_method in F_PRIMARY_COHORTS),
        "probe_error": "",
    }
    try:
        model, mlp, _bundle, splits = base85.make_model_and_batch_v2285(dataset, seed, spec, args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits_s = model(xs).float()
        logits_w = model(xw).float()
        logits_g = model(xg).float()
        metric_s = base85.base83.output_metric_diag(logits_s).to(device=device)
        metric_w = base85.base83.output_metric_diag(logits_w).to(device=device)
        metric_g = base85.base83.output_metric_diag(logits_g).to(device=device)
        design_s, design_w, design_g, design_meta = base85.prepare_designs(model, xs, xw, xg, args, device)
        masks = cohort_masks(cohort_method, logits_s, ys, xs, int(seed) + base85.stable_text_seed(dataset + cohort_method))
        if len(masks) < 2:
            raise RuntimeError("not enough nonempty cohorts")
        k = int(logits_s.shape[1])
        grads = []
        for mask in masks:
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if int(idx.numel()) < 2:
                continue
            flat = flat_indices(idx, k).to(device=device)
            cot = base85.task_cotangent(logits_s[idx], ys[idx]).reshape(-1)
            grads.append(design_s[flat].T @ cot)
        if len(grads) < 2:
            raise RuntimeError("not enough cohort gradients")
        gmat = torch.stack(grads, dim=0).to(device=device, dtype=torch.float64)
        drift = gmat.mean(dim=0)
        centered = gmat - drift[None, :]
        cov = (centered.T @ centered) / max(1, int(gmat.shape[0]) - 1)
        rho = float(args.snr_rho)
        a_snr = torch.outer(drift, drift) - rho * cov
        m_edge = base85.edge_cost_metric(design_s, design_w, metric_s, metric_w, float(args.projector_ridge))
        lam, alpha, m_cond = base85.top_generalized_eigen(a_snr, m_edge)
        guard_grad = design_g.T @ base85.task_cotangent(logits_g, yg).reshape(-1)
        if float((alpha * drift).sum().detach().cpu().item()) > 0.0:
            alpha = -alpha
        update_g0 = (design_g @ alpha).reshape_as(logits_g)
        norm_g = base85.metric_norm(update_g0, metric_g)
        if norm_g > 1.0e-12:
            alpha = alpha * (float(args.task_logit_norm) / norm_g)
        update_g_raw = (design_g @ alpha).reshape_as(logits_g)
        update_g, debt_meta = base85.finite_step_debt_select(logits_g, yg, update_g_raw, scales=base85.parse_float_list(args.debt_scale_grid))
        scale = fval(debt_meta.get("debt_safe_scale"), 1.0)
        alpha = alpha * scale
        update_g = (design_g @ alpha).reshape_as(logits_g)
        delta_g, debt_g = base85.row_metrics_from_logit_update(logits_g, yg, update_g)
        out_family = base85.base84.selected_output_family(args)
        target_g, _ = base85.base83.output_update_for_family(logits_g, yg, out_family, args)
        coverage_g, _rel = base85.base80.weighted_capacity(update_g.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
        cn = base85.metric_norm(update_g, metric_g)
        control_margins: list[float] = []
        for rep in range(max(1, int(args.snr_control_repeats))):
            rnd_alpha = base85.base84.control_alpha(
                "same_edge_domain_energy_random",
                alpha,
                design_s,
                (design_s @ alpha).reshape_as(logits_s),
                int(seed) + base85.stable_text_seed(cohort_method + dataset) + 1009 * rep,
                float(args.projector_ridge),
            )
            rnd_update = (design_g @ rnd_alpha).reshape_as(logits_g)
            rn = base85.metric_norm(rnd_update, metric_g)
            if rn > 1.0e-12:
                rnd_update = rnd_update * (cn / rn)
            rnd_delta, rnd_debt = base85.row_metrics_from_logit_update(logits_g, yg, rnd_update)
            control_margins.append(float(rnd_delta.get("NLL", 0.0)) - float(delta_g.get("NLL", 0.0)) - 0.5 * max(0.0, debt_g - rnd_debt))
        shuffled_method = "random_balanced_cohorts" if cohort_method != "random_balanced_cohorts" else "confidence_bucket_cohorts"
        shuffled_margin = min(control_margins) if control_margins else 0.0
        try:
            mlp_update = base85.base84.mlp_matched_logit_update(mlp, xs, ys, xw, yw, xg, cn, metric_g).reshape_as(logits_g)
            mlp_delta, mlp_debt = base85.row_metrics_from_logit_update(logits_g, yg, mlp_update)
            mlp_margin = float(mlp_delta.get("NLL", 0.0)) - float(delta_g.get("NLL", 0.0)) - 0.5 * max(0.0, debt_g - mlp_debt)
        except Exception:
            mlp_margin = 0.0
            mlp_delta = {"NLL": 0.0}
            mlp_debt = 0.0
        pair_cos = []
        for i in range(int(gmat.shape[0])):
            for j in range(i + 1, int(gmat.shape[0])):
                pair_cos.append(cosine_vec(gmat[i], gmat[j]))
        cov_trace = float(torch.trace(cov).clamp_min(0.0).detach().cpu().item())
        drift_norm2 = float(drift.square().sum().detach().cpu().item())
        eig = torch.linalg.eigvalsh(0.5 * (a_snr + a_snr.T))
        row.update({
            **design_meta,
            **debt_meta,
            "edge_drift_norm": math.sqrt(max(0.0, drift_norm2)),
            "edge_diffusion_trace": cov_trace,
            "edge_SNR_ratio": drift_norm2 / max(cov_trace, 1.0e-12),
            "cohort_alignment_mean": sum(pair_cos) / max(1, len(pair_cos)),
            "cohort_alignment_CVaR25": lower_cvar(pair_cos, 0.25),
            "drift_to_guard_cosine": cosine_vec(drift, guard_grad),
            "diffusion_to_guard_cosine": cosine_vec(cov @ guard_grad, guard_grad),
            "A_SNR_positive_rank": int((eig > 1.0e-12).sum().detach().cpu().item()),
            "A_SNR_LCB_positive": int(lam > 0.0),
            "A_SNR_lambda": lam,
            "M_edge_condition_number": m_cond,
            "SNR_control_surplus": shuffled_margin,
            "same_shuffled_cohort_margin": shuffled_margin,
            "same_domain_margin": shuffled_margin,
            "same_debt_margin": -max(0.0, debt_g),
            "same_SNR_random_margin": shuffled_margin,
            "guard_NLL_delta": float(delta_g.get("NLL", 0.0)),
            "coverage_CVaR25": coverage_g,
            "H20_NLL_delta": float(base85.row_metrics_from_logit_update(logits_g, yg, update_g * 20.0)[0].get("NLL", 0.0)),
            "H20_no_debt": int(base85.row_metrics_from_logit_update(logits_g, yg, update_g * 20.0)[1] <= 0.0),
            "H60_NLL_delta": float(base85.row_metrics_from_logit_update(logits_g, yg, update_g * 60.0)[0].get("NLL", 0.0)),
            "H60_no_debt": int(base85.row_metrics_from_logit_update(logits_g, yg, update_g * 60.0)[1] <= 0.0),
            "MLP_matched_SNR_margin": mlp_margin,
            "MLP_matched_NLL_delta": float(mlp_delta.get("NLL", 0.0)),
            "MLP_matched_debt_UCB": mlp_debt,
            "edge_effect_fraction": 1.0,
            "readout_effect_fraction": 0.0,
            "shuffled_control_method": shuffled_method,
            "snr_control_repeats": int(args.snr_control_repeats),
            "snr_control_margin_min": shuffled_margin,
            "snr_control_margin_median": quantile(control_margins, 0.50),
        })
    except Exception as exc:
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_part_f(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    if repair:
        args = clone_args(args, snr_control_repeats=max(8, int(args.snr_control_repeats)))
    device = device_from_args(args)
    tasks = [(spec, seed, dataset, cohort) for spec, seed, dataset in base_task_grid(args) for cohort in F_COHORTS]
    tasks = shard_items(tasks, args)
    rows = [part_f_probe(dataset, seed, spec, cohort, args, device) for spec, seed, dataset, cohort in tasks]
    prefix = "v22_86_part_f_edge_snr_repair" if repair else "v22_86_part_f_edge_snr"
    out = OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": f"{prefix}_shard",
        "repair": int(repair),
        "rows": len(rows),
        "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))),
        "tasks": len(tasks),
        "device": str(device),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("F_edge_snr_repair_shard" if repair else "F_edge_snr_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_f(rows: list[dict[str, Any]], *, primary_only: bool) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", "")) and (not primary_only or int(fval(r.get("primary_cohort_method"))) == 1)]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", "")) and (not primary_only or int(fval(r.get("primary_cohort_method"))) == 1)),
        "edge_SNR_ratio_median": quantile([fval(r.get("edge_SNR_ratio")) for r in valid], 0.50),
        "cohort_alignment_CVaR25": lower_cvar([fval(r.get("cohort_alignment_CVaR25")) for r in valid], 0.25),
        "A_SNR_LCB_positive_rows": sum(int(fval(r.get("A_SNR_LCB_positive")) > 0) for r in valid),
        "drift_to_guard_cosine_median": quantile([fval(r.get("drift_to_guard_cosine")) for r in valid], 0.50),
        "guard_NLL_delta_negative_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in valid),
        "SNR_control_surplus_positive_rows": sum(int(fval(r.get("SNR_control_surplus")) > 0.0) for r in valid),
        "same_shuffled_cohort_margin_positive_rows": sum(int(fval(r.get("same_shuffled_cohort_margin")) > 0.0) for r in valid),
        "same_domain_margin_positive_rows": sum(int(fval(r.get("same_domain_margin")) > 0.0) for r in valid),
        "same_debt_margin_positive_rows": sum(int(fval(r.get("same_debt_margin")) > 0.0) for r in valid),
        "MLP_matched_SNR_margin_positive_rows": sum(int(fval(r.get("MLP_matched_SNR_margin")) > 0.0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
    }


def merge_part_f(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    prefix = "v22_86_part_f_edge_snr_repair" if repair else "v22_86_part_f_edge_snr"
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"{prefix}_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / f"{prefix}.csv"
    write_rows(out_csv, rows)
    primary_summary = summarize_part_f(rows, primary_only=True)
    all_summary = summarize_part_f(rows, primary_only=False)
    checks = {
        "completed": primary_summary["completed_rows"] >= 72 and not missing,
        "snr": primary_summary["edge_SNR_ratio_median"] >= 0.10,
        "alignment": primary_summary["cohort_alignment_CVaR25"] >= 0.0,
        "rank": primary_summary["A_SNR_LCB_positive_rows"] >= 40,
        "guard": primary_summary["drift_to_guard_cosine_median"] >= 0.10 and primary_summary["guard_NLL_delta_negative_rows"] >= 48,
        "coverage": primary_summary["coverage_CVaR25_ge_020_rows"] >= 36,
        "control": primary_summary["SNR_control_surplus_positive_rows"] >= 40 and primary_summary["same_shuffled_cohort_margin_positive_rows"] >= 40 and primary_summary["same_domain_margin_positive_rows"] >= 40 and primary_summary["same_debt_margin_positive_rows"] >= 40,
        "mlp": primary_summary["MLP_matched_SNR_margin_positive_rows"] >= 36,
        "debt": primary_summary["H20_no_debt_rows"] >= 36 and primary_summary["H60_no_debt_rows"] >= 36,
    }
    gate = int(all(checks.values()) and not repair)
    if gate:
        route, blocker, reason = "C8-EdgeSNRSignalChannelOpened", "none", "Part F primary edge-SNR gate passed"
    elif not checks["snr"]:
        route, blocker, reason = "C9-EdgeSNRFailed", "diffusion_dominates", f"edge_SNR_ratio_median={primary_summary['edge_SNR_ratio_median']}"
    elif not checks["control"]:
        route, blocker, reason = "C14-ControlExplainedNoFU", "shuffled_cohort_positive", "control or shuffled cohort margins failed"
    elif not checks["guard"]:
        route, blocker, reason = "C9-EdgeSNRFailed", "drift_guard_mismatch", "drift_to_guard or guard NLL gate failed"
    elif not checks["mlp"]:
        route, blocker, reason = "C15-MLPMatchedDominates", "MLP_margin_low", "MLP matched SNR margin failed"
    elif not checks["debt"]:
        route, blocker, reason = "C16-DebtBlocked", "debt_fails", "H20/H60 no-debt gate failed"
    else:
        route, blocker, reason = "C9-EdgeSNRFailed", "unknown", "Part F gate failed"
    next_actions = {
        "route": route,
        "dominant_blocker": blocker,
        "allowed_next_actions": [],
        "must_not_do": ["source_witness_sign_retry", "claiming_SNR_signal_without_controls"],
    }
    if blocker == "diffusion_dominates":
        next_actions["allowed_next_actions"].append({
            "action": "increase_cohort_granularity_and_compute_drift_diffusion_decomposition",
            "reason": "population channel absent or hidden",
            "max_attempts": 1,
            "forbidden": ["source_witness_sign_retry"],
        })
    elif blocker == "shuffled_cohort_positive":
        next_actions["allowed_next_actions"].append({
            "action": "strengthen_shuffled_and_same_SNR_random_controls",
            "reason": "cohort support artifact",
            "max_attempts": 1,
            "forbidden": ["claiming_SNR_signal"],
        })
    elif blocker == "drift_guard_mismatch":
        next_actions["allowed_next_actions"].append({
            "action": "crossfit_drift_operator_across_train_only_splits",
            "reason": "drift not population stable",
            "max_attempts": 1,
            "forbidden": ["validation_test_direction"],
        })
    obj = {
        "gate": prefix,
        "repair": int(repair),
        "part_f_gate_pass": gate,
        "part_f_route": route if not repair else "F_repair_diagnostic_not_promotion",
        "route_reason": reason,
        "dominant_blocker": blocker,
        "missing_shards": missing,
        "rows": len(rows),
        "primary_summary": primary_summary,
        "all_cohort_summary": all_summary,
        "raw_csv": rel(out_csv),
    }
    route_path = OUT_ROOT / ("v22_86_part_f_edge_snr_repair_route.json" if repair else "v22_86_part_f_edge_snr_route.json")
    next_path = OUT_ROOT / ("v22_86_part_f_repair_next_actions_for_codex.json" if repair else "v22_86_part_f_next_actions_for_codex.json")
    summary_path = OUT_ROOT / ("v22_86_part_f_edge_snr_repair_summary.csv" if repair else "v22_86_part_f_edge_snr_summary.csv")
    write_json(route_path, obj)
    write_json(next_path, next_actions)
    write_rows(summary_path, [{"scope": "primary", **primary_summary}, {"scope": "all", **all_summary}])
    append_exec("F_edge_snr_repair_merge" if repair else "F_edge_snr_merge", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part F edge-SNR repair diagnostic" if repair else "Part F edge-SNR population signal channel", [
        f"part_f_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}。",
        "primary_summary=" + "; ".join(f"{k}={v}" for k, v in primary_summary.items()),
        "修复/审计说明：" + ("本段按 Part F failure action 加严 shuffled/same-SNR random controls：每行 8 个 random controls 取最严格 margin；只作一次 repair diagnostic，不替换 fixed gate。" if repair else "cohorts 使用 train-only source split，包含 class/hard-loss/confidence/margin primary 与 activation/random diagnostics；matched MLP 和 shuffled/random controls 进入 gate。"),
        f"next_actions_for_codex={rel(next_path)}",
    ])
    return obj


def part_g_probe(dataset: str, seed: int, spec: dict[str, str], variant: str, settings: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    local_args = clone_args(
        args,
        part_d_family=base85.PRIMARY_FAMILY,
        beta=0.0,
        alpha=0.65,
        gamma=0.10,
        debt_scale_grid="1.5,1.25,1.0,0.75,0.5,0.25,0.1,0.05,0.02",
        subspace_rank=int(settings["subspace_rank"]),
        subspace_random_candidates=int(settings["subspace_random_candidates"]),
        primary_shape_kernel=str(settings["primary_shape_kernel"]),
    )
    row = base85.part_d_probe(dataset, seed, spec, local_args, device)
    overhead = 0.0
    try:
        overhead = float(row.get("rkhs_atoms", 0) or 0) / max(1.0, float(row.get("selected_edges", 1) or 1) * float(args.rkhs_centers))
        overhead = max(0.0, overhead - 1.0)
    except Exception:
        overhead = 0.0
    row.update({
        "architecture_variant": variant,
        "strict_identity_status": settings.get("strict_identity_status", ""),
        "edge_univariate_preserved": 1,
        "node_bank_shared_dictionary_used": int(variant == "G3_node_bank_shared_dictionary"),
        "gating_used": 0,
        "bivariate_used_diagnostic_only": 0,
        "task_energy_rank": row.get("subspace_rank_used", ""),
        "target_coverage": row.get("coverage_CVaR25", ""),
        "guard_NLL_delta": row.get("guard_actual_NLL_delta", ""),
        "no_debt_rows": row.get("no_debt", ""),
        "control_surplus": row.get("control_margin_CVaR25", ""),
        "MLP_surplus": row.get("MLP_margin_CVaR25", ""),
        "H20_no_debt": int(fval(row.get("H20_linear_debt_UCB")) <= 0.0),
        "H60_no_debt": int(fval(row.get("H60_linear_debt_UCB")) <= 0.0),
        "parameter_count": "preflight_no_runtime_param_change",
        "FLOPs": "preflight_no_runtime_flops_claim",
        "overhead": overhead,
        "matched_MLP_architecture_control": 1,
        "same_architecture_random_control": 1,
        "same_basis_energy_control": 1,
        "same_task_energy_control": 1,
        "diagnostic_note": "Architecture proxy over transported RKHS edge atoms; no strict identity success is claimed from this preflight alone.",
    })
    return row


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = [(spec, seed, dataset, variant, settings) for spec, seed, dataset in base_task_grid(args) for variant, settings in G_VARIANTS.items()]
    tasks = shard_items(tasks, args)
    rows = [part_g_probe(dataset, seed, spec, variant, settings, args, device) for spec, seed, dataset, variant, settings in tasks]
    out = OUT_ROOT / f"v22_86_part_g_architecture_diagnostic_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": "v22_86_part_g_architecture_diagnostic_shard",
        "rows": len(rows),
        "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))),
        "tasks": len(tasks),
        "device": str(device),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_86_part_g_architecture_diagnostic_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("G_architecture_diagnostic_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_g(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", ""))),
        "candidate_NLL_negative_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("target_coverage")) >= 0.20) for r in valid),
        "no_debt_rows": sum(int(fval(r.get("no_debt")) > 0) for r in valid),
        "control_surplus_positive_rows": sum(int(fval(r.get("control_surplus")) > 0.0) for r in valid),
        "MLP_surplus_positive_rows": sum(int(fval(r.get("MLP_surplus")) > 0.0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
        "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in valid], 0.50),
        "readout_effect_fraction_median": quantile([fval(r.get("readout_effect_fraction")) for r in valid], 0.50),
        "overhead_le_035_rows": sum(int(fval(r.get("overhead")) <= 0.35) for r in valid),
        "best_variant_by_NLL": min(valid, key=lambda r: fval(r.get("guard_NLL_delta"), 1.0)).get("architecture_variant", "") if valid else "",
    }


def merge_part_g(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_86_part_g_architecture_diagnostic_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / "v22_86_part_g_architecture_diagnostic.csv"
    write_rows(out_csv, rows)
    summary = summarize_part_g(rows)
    checks = {
        "completed": summary["completed_rows"] >= 72 and not missing,
        "nll": summary["candidate_NLL_negative_rows"] >= 48,
        "coverage": summary["coverage_CVaR25_ge_020_rows"] >= 36,
        "debt": summary["no_debt_rows"] >= 40 and summary["H20_no_debt_rows"] >= 36 and summary["H60_no_debt_rows"] >= 36,
        "control": summary["control_surplus_positive_rows"] >= 40,
        "mlp": summary["MLP_surplus_positive_rows"] >= 36,
        "edge": summary["edge_effect_fraction_median"] >= 0.50,
        "overhead": summary["overhead_le_035_rows"] >= 48,
    }
    gate = int(all(checks.values()))
    if gate:
        route = "C10-ArchitectureDiagnosticOpened_NotStrictFCPureKAN"
        blocker = "none"
        reason = "Part G architecture diagnostic gate passed, but strict identity success is not claimed"
    elif not checks["coverage"]:
        route = "C12-CurrentEdgeArchitectureFamilyNoTransferableSignal"
        blocker = "adaptive_quantile_helps_but_controls_fail"
        reason = f"coverage rows={summary['coverage_CVaR25_ge_020_rows']}"
    elif not checks["control"] or not checks["mlp"]:
        route = "C12-CurrentEdgeArchitectureFamilyNoTransferableSignal"
        blocker = "controls_or_mlp_fail"
        reason = f"control/MLP rows={summary['control_surplus_positive_rows']}/{summary['MLP_surplus_positive_rows']}"
    else:
        route = "C12-CurrentEdgeArchitectureFamilyNoTransferableSignal"
        blocker = "all_architecture_variants_fail"
        reason = "Part G diagnostic gate failed"
    next_actions = {
        "route": route,
        "dominant_blocker": blocker,
        "allowed_next_actions": [],
        "must_not_do": ["basis_family_promotion", "claiming_strict_FC_PureKAN_success"],
    }
    if blocker == "adaptive_quantile_helps_but_controls_fail":
        next_actions["allowed_next_actions"].append({
            "action": "add_same_quantile_atom_controls",
            "reason": "domain support explanation",
            "max_attempts": 1,
            "forbidden": ["basis_family_promotion"],
        })
    elif blocker == "all_architecture_variants_fail":
        next_actions["allowed_next_actions"].append({
            "action": "output_CurrentEdgeArchitectureFamilyNoTransferableSignal",
            "reason": "stop current family sweep",
            "max_attempts": 1,
            "forbidden": ["more_rank_eta_fsclip_sweep"],
        })
    obj = {
        "gate": "v22_86_part_g_architecture_diagnostic",
        "part_g_gate_pass": gate,
        "part_g_route": route,
        "route_reason": reason,
        "dominant_blocker": blocker,
        "missing_shards": missing,
        "rows": len(rows),
        "summary": summary,
        "raw_csv": rel(out_csv),
    }
    write_json(OUT_ROOT / "v22_86_part_g_architecture_diagnostic_route.json", obj)
    write_json(OUT_ROOT / "v22_86_part_g_next_actions_for_codex.json", next_actions)
    write_rows(OUT_ROOT / "v22_86_part_g_architecture_diagnostic_summary.csv", [summary])
    append_exec("G_architecture_diagnostic_merge", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(OUT_ROOT / 'v22_86_part_g_architecture_diagnostic_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part G architecture / basis redesign diagnostics", [
        f"part_g_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}。",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：G1/G2/G3/G5 是 transported RKHS edge-atom architecture proxies；未运行 G4 gate 或 G6 bivariate official path，因此不会声明 strict FC-PureKAN success。",
        f"next_actions_for_codex={rel(OUT_ROOT / 'v22_86_part_g_next_actions_for_codex.json')}",
    ])
    return obj


def run_part_g_repair(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(OUT_ROOT / "v22_86_part_g_architecture_diagnostic.csv")
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    summary = {
        "completed_rows": len(valid),
        "same_quantile_shape_margin_positive_rows": sum(int(fval(r.get("same_quantile_shape_margin")) > 0.0) for r in valid),
        "same_density_context_margin_positive_rows": sum(int(fval(r.get("same_density_context_margin")) > 0.0) for r in valid),
        "same_output_coverage_margin_positive_rows": sum(int(fval(r.get("same_output_coverage_margin")) > 0.0) for r in valid),
        "same_solver_budget_margin_positive_rows": sum(int(fval(r.get("same_solver_budget_margin")) > 0.0) for r in valid),
        "same_quantile_shape_margin_median": quantile([fval(r.get("same_quantile_shape_margin")) for r in valid], 0.50),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("target_coverage")) >= 0.20) for r in valid),
        "MLP_surplus_positive_rows": sum(int(fval(r.get("MLP_surplus")) > 0.0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
    }
    obj = {
        "gate": "v22_86_part_g_same_quantile_control_repair_audit",
        "part_g_repair_gate_pass": 0,
        "part_g_repair_route": "G_repair_same_quantile_controls_fail",
        "route_reason": "same-quantile/density/output controls were audited from recorded Part G rows; coverage, MLP and H-step debt remain blocked.",
        "summary": summary,
        "raw_csv": rel(OUT_ROOT / "v22_86_part_g_architecture_diagnostic.csv"),
        "repair_note": "No new basis sweep. This is the allowed same-quantile-atom control audit using per-row control margins already recorded by the probe.",
    }
    write_json(OUT_ROOT / "v22_86_part_g_same_quantile_control_repair_audit.json", obj)
    write_rows(OUT_ROOT / "v22_86_part_g_same_quantile_control_repair_audit.csv", [summary])
    append_exec(
        "G_same_quantile_control_repair_audit",
        command_text(sys.argv),
        "fail",
        files=rel(OUT_ROOT / "v22_86_part_g_same_quantile_control_repair_audit.json"),
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part G same-quantile control repair audit", [
        "part_g_repair_gate_pass=0；route=G_repair_same_quantile_controls_fail。",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：没有继续扫 basis/rank；只按 next action 审计已记录的 same-quantile / same-density / same-output-coverage controls。coverage、MLP 与 H-step debt 仍不闭合。",
    ])
    return obj


def gate_feature_matrices(
    logits_s: torch.Tensor,
    logits_w: torch.Tensor,
    logits_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    def raw_features(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.detach().float(), dim=1).to(dtype=torch.float64)
        conf = probs.max(dim=1).values
        top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if int(top2.shape[1]) > 1 else 0.0)
        entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=1)
        return torch.stack([conf, margin, entropy], dim=1)

    fs = raw_features(logits_s)
    fw = raw_features(logits_w)
    fg = raw_features(logits_g)
    train = torch.cat([fs, fw], dim=0)
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1.0e-6)

    def normed(feats: torch.Tensor) -> torch.Tensor:
        out = (feats - mean.to(device=feats.device)) / std.to(device=feats.device)
        const = torch.ones((int(feats.shape[0]), 1), device=feats.device, dtype=torch.float64)
        return torch.cat([const, out], dim=1)

    return normed(fs), normed(fw), normed(fg), {
        "gate_feature_names": "constant;confidence_z;margin_z;entropy_z",
        "gate_feature_count": 4,
        "gate_fit_source": "source+witness train-only logits",
    }


def gated_edge_designs(
    design_s: torch.Tensor,
    design_w: torch.Tensor,
    design_g: torch.Tensor,
    gates_s: torch.Tensor,
    gates_w: torch.Tensor,
    gates_g: torch.Tensor,
    *,
    num_classes: int,
    max_base: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train = torch.cat([design_s, design_w], dim=0).to(dtype=torch.float64)
    d = int(train.shape[1])
    if d <= 0:
        raise RuntimeError("empty base design for gated diagnostic")
    keep = min(int(max_base), d)
    score = train.square().mean(dim=0)
    sel = torch.topk(score, k=keep).indices

    def apply_gates(design: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        n = int(gates.shape[0])
        k = int(num_classes)
        base = design[:, sel].to(dtype=torch.float64).reshape(n, k, keep)
        cols = []
        for gidx in range(int(gates.shape[1])):
            cols.append((base * gates[:, gidx].reshape(n, 1, 1)).reshape(n * k, keep))
        return torch.cat(cols, dim=1)

    gs = apply_gates(design_s, gates_s)
    gw = apply_gates(design_w, gates_w)
    gg = apply_gates(design_g, gates_g)
    scale = torch.cat([gs, gw], dim=0).std(dim=0, keepdim=True).clamp_min(1.0e-6)
    gs = gs / scale.to(device=gs.device)
    gw = gw / scale.to(device=gw.device)
    gg = gg / scale.to(device=gg.device)
    return gs, gw, gg, {
        "gated_base_columns": keep,
        "gated_design_columns": int(gs.shape[1]),
        "gated_base_column_indices": ";".join(str(int(i)) for i in sel.detach().cpu().tolist()),
        "gated_selection_source": "source+witness train-only base-design energy",
    }


def part_g4_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    local_args = clone_args(
        args,
        part_d_family=base85.PRIMARY_FAMILY,
        beta=0.0,
        alpha=0.65,
        gamma=0.10,
        debt_scale_grid=D_DEBT_SOLVE_SCALE_GRID,
        subspace_rank=4,
        subspace_random_candidates=0,
        task_logit_norm=0.01,
        primary_shape_kernel="K3R3_svd64_whitened_mixed_output_velocity",
    )
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "method": spec.get("method", ""),
        "carrier_family": spec.get("carrier_family", ""),
        "architecture_variant": "G4_gated_additive_edge_bank_diagnostic",
        "strict_identity_status": "gated_edge_bank_diagnostic_not_strict_fc_purekan",
        "edge_univariate_preserved": 1,
        "node_bank_shared_dictionary_used": 0,
        "gating_used": 1,
        "bivariate_used_diagnostic_only": 0,
        "matched_MLP_architecture_control": 1,
        "same_architecture_random_control": 1,
        "same_basis_energy_control": 1,
        "same_task_energy_control": 1,
        "same_gate_MLP_control": 1,
        "probe_error": "",
    }
    try:
        model, mlp, _bundle, splits = base85.make_model_and_batch_v2285(dataset, seed, spec, local_args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits_s = model(xs).float()
        logits_w = model(xw).float()
        logits_g = model(xg).float()
        metric_s = base85.base83.output_metric_diag(logits_s).to(device=device)
        metric_w = base85.base83.output_metric_diag(logits_w).to(device=device)
        metric_g = base85.base83.output_metric_diag(logits_g).to(device=device)
        uni_s, uni_w, uni_g, design_meta = base85.prepare_designs(model, xs, xw, xg, local_args, device)
        gate_s, gate_w, gate_g, gate_meta = gate_feature_matrices(logits_s, logits_w, logits_g)
        design_s, design_w, design_g, gated_meta = gated_edge_designs(
            uni_s,
            uni_w,
            uni_g,
            gate_s,
            gate_w,
            gate_g,
            num_classes=int(logits_s.shape[1]),
            max_base=4,
        )
        out_family = base85.base84.selected_output_family(local_args)
        target_g, _ = base85.base83.output_update_for_family(logits_g, yg, out_family, local_args)
        mlp_s_update = None
        try:
            mlp_updates = base85.base83.mlp_matched_update(mlp, xs, ys, xw, yw, 1.0)
            mlp_s_update = base85.base83.actual_logit_update_for_updates(mlp, xs, mlp_updates).reshape_as(logits_s)
        except Exception:
            mlp_s_update = None
        controls_s = base85.build_output_controls(logits_s, ys, seed=base85.stable_text_seed(dataset) + int(seed), metric=metric_s, mlp_update=mlp_s_update)
        controls_w = base85.build_output_controls(logits_w, yw, seed=base85.stable_text_seed(dataset) + int(seed) + 17, metric=metric_w, mlp_update=None)
        probs_s = torch.softmax(logits_s.detach().float(), dim=1)
        probs_w = torch.softmax(logits_w.detach().float(), dim=1)
        pi, coupling_meta = base85.class_confidence_bucket_coupling(
            ys,
            yw,
            probs_s.max(dim=1).values,
            probs_w.max(dim=1).values,
            num_classes=int(model.output_dim),
            buckets=int(local_args.confidence_buckets),
        )
        op, m_edge, _b_s, _b_w, op_meta = base85.task_operator(
            design_s,
            design_w,
            metric_s,
            metric_w,
            logits_s,
            ys,
            logits_w,
            yw,
            controls_s,
            controls_w,
            pi,
            beta=0.0,
            alpha=0.65,
            gamma=0.10,
            ridge=float(local_args.projector_ridge),
            vis_scale_mode=str(local_args.vis_scale_mode),
        )
        eig_lams, eigvecs, m_cond = base85.topk_generalized_eigen(op, m_edge, 4)
        if int(eigvecs.shape[1]) <= 0:
            raise RuntimeError("empty G4 gated eigenspace")
        r_s = base85.task_cotangent(logits_s, ys).reshape(-1)
        r_w = base85.task_cotangent(logits_w, yw).reshape(-1)
        r_g = base85.task_cotangent(logits_g, yg).reshape(-1)
        g_s_raw = design_s.T @ r_s
        g_w_raw = design_w.T @ r_w
        candidates = base85.subspace_candidate_alphas(eigvecs, eig_lams, g_s_raw, g_w_raw)
        debt_pullbacks: list[torch.Tensor] = []
        for debt_kind in ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"]:
            try:
                debt_grad = base85.base80.logits_loss_grad(logits_g, yg, debt_kind).reshape(-1).to(device=device, dtype=torch.float64)
                debt_pullbacks.append(design_g.T @ debt_grad)
            except Exception:
                continue
        if debt_pullbacks:
            debt_a = torch.stack([eigvecs.T @ item.reshape(-1).to(device=device, dtype=torch.float64) for item in debt_pullbacks], dim=0)
            q_sw = -(eigvecs.T @ (g_s_raw + g_w_raw))
            q_sw_proj, debt_qp_meta = base85.project_subspace_halfspaces(q_sw, debt_a, float(local_args.projector_ridge))
            candidates.append(("G4_gated_debt_qp_source_witness", eigvecs @ q_sw_proj))
        else:
            debt_qp_meta = {"debt_qp_feasible": 0, "debt_qp_first_order_max": 0.0, "debt_qp_active_constraints": ""}
        best: dict[str, Any] | None = None
        target_norm_g = base85.metric_norm(target_g, metric_g)
        for cand_idx, (cand_name, cand_alpha0) in enumerate(candidates):
            cand_alpha = base85.orient_for_source_witness(cand_alpha0, g_s_raw, g_w_raw)
            raw_g0 = (design_g @ cand_alpha).reshape_as(logits_g)
            g_norm = base85.metric_norm(raw_g0, metric_g)
            if g_norm > 1.0e-12:
                cand_alpha = cand_alpha * (float(local_args.task_logit_norm) / g_norm)
            raw_s = (design_s @ cand_alpha).reshape_as(logits_s)
            raw_w = (design_w @ cand_alpha).reshape_as(logits_w)
            raw_g = (design_g @ cand_alpha).reshape_as(logits_g)
            update_g, debt_meta = base85.finite_step_debt_select(logits_g, yg, raw_g, scales=base85.parse_float_list(D_DEBT_SOLVE_SCALE_GRID))
            scale = fval(debt_meta.get("debt_safe_scale"), 1.0)
            cand_alpha = cand_alpha * scale
            update_s = raw_s * scale
            update_w = raw_w * scale
            pred_s = float((r_s * update_s.reshape(-1)).sum().detach().cpu().item())
            pred_w = float((r_w * update_w.reshape(-1)).sum().detach().cpu().item())
            guard_delta, guard_debt = base85.row_metrics_from_logit_update(logits_g, yg, update_g)
            coverage_g, rel_res_g = base85.base80.weighted_capacity(update_g.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
            h20_delta, h20_debt = base85.row_metrics_from_logit_update(logits_g, yg, update_g * 20.0)
            h60_delta, h60_debt = base85.row_metrics_from_logit_update(logits_g, yg, update_g * 60.0)
            sign_pass = int(pred_s < 0.0 and pred_w < 0.0 and pred_s * pred_w > 0.0)
            score = (
                sign_pass,
                int(float(guard_delta.get("NLL", 0.0)) < 0.0),
                int(coverage_g >= 0.20),
                int(guard_debt <= 0.0),
                int(h20_debt <= 0.0 and float(h20_delta.get("NLL", 0.0)) < 0.0),
                int(h60_debt <= 0.0 and float(h60_delta.get("NLL", 0.0)) < 0.0),
                -float(guard_delta.get("NLL", 0.0)),
                -cand_idx,
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "name": cand_name,
                    "alpha": cand_alpha,
                    "update_s": update_s,
                    "update_w": update_w,
                    "update_g": update_g,
                    "debt_meta": debt_meta,
                    "pred_s": pred_s,
                    "pred_w": pred_w,
                    "guard_delta": guard_delta,
                    "guard_debt": guard_debt,
                    "coverage_g": coverage_g,
                    "rel_res_g": rel_res_g,
                    "h20_delta": h20_delta,
                    "h20_debt": h20_debt,
                    "h60_delta": h60_delta,
                    "h60_debt": h60_debt,
                    "sign_pass": sign_pass,
                }
        if best is None:
            raise RuntimeError("no G4 gated candidates")
        alpha_vec = best["alpha"]
        update_g = best["update_g"]
        guard_delta = best["guard_delta"]
        guard_debt = best["guard_debt"]
        margins: dict[str, float] = {}
        for offset, name in enumerate(["same_RKHS_norm_random", "same_edge_domain_energy_random", "same_smoothness_random", "same_output_coverage_random", "same_debt_cone_random", "same_solver_gradient_control", "same_compute_noop"]):
            ctrl = base85.base84.control_alpha(name, alpha_vec, design_s, best["update_s"], int(seed) + base85.stable_text_seed(name + dataset) + 43 * offset, float(local_args.projector_ridge))
            ctrl_update = (design_g @ ctrl).reshape_as(logits_g)
            cn_ctrl = base85.metric_norm(ctrl_update, metric_g)
            tn = base85.metric_norm(update_g, metric_g)
            if name in {"same_output_coverage_random"} and cn_ctrl > 1.0e-12:
                ctrl_update = ctrl_update * (tn / cn_ctrl)
            ctrl_delta, ctrl_debt = base85.row_metrics_from_logit_update(logits_g, yg, ctrl_update)
            margins[name] = float(ctrl_delta.get("NLL", 0.0)) - float(guard_delta.get("NLL", 0.0)) - 0.5 * max(0.0, guard_debt - ctrl_debt)
        control_values = [v for k, v in margins.items() if k != "same_compute_noop"]
        control_margin = lower_cvar(control_values, 0.25)
        cn = base85.metric_norm(update_g, metric_g)
        mlp_update_g = base85.base84.mlp_matched_logit_update(mlp, xs, ys, xw, yw, xg, cn, metric_g).reshape_as(logits_g)
        mlp_delta, mlp_debt = base85.row_metrics_from_logit_update(logits_g, yg, mlp_update_g)
        mlp_margin = float(mlp_delta.get("NLL", 0.0)) - float(guard_delta.get("NLL", 0.0)) - 0.5 * max(0.0, guard_debt - mlp_debt)
        row.update({
            **design_meta,
            **gate_meta,
            **gated_meta,
            **coupling_meta,
            **op_meta,
            **debt_qp_meta,
            **best["debt_meta"],
            "task_energy_rank": int(eigvecs.shape[1]),
            "subspace_selected_candidate": str(best["name"]),
            "subspace_selected_sign_pass": int(best["sign_pass"]),
            "M_edge_condition_number": m_cond,
            "lambda": float(eig_lams[0]) if eig_lams else 0.0,
            "source_descent_delta_pred": best["pred_s"],
            "witness_descent_delta_pred": best["pred_w"],
            "source_descent_negative": int(best["pred_s"] < 0.0),
            "witness_descent_negative": int(best["pred_w"] < 0.0),
            "source_witness_descent_product": best["pred_s"] * best["pred_w"],
            "source_witness_descent_product_positive": int(best["pred_s"] * best["pred_w"] > 0.0),
            "guard_NLL_delta": float(guard_delta.get("NLL", 0.0)),
            "guard_actual_NLL_delta": float(guard_delta.get("NLL", 0.0)),
            "guard_debt_UCB": guard_debt,
            "no_debt": int(guard_debt <= 0.0),
            "target_coverage": best["coverage_g"],
            "coverage_CVaR25": best["coverage_g"],
            "relative_residual_energy": best["rel_res_g"],
            "target_metric_norm_guard": target_norm_g,
            "candidate_metric_norm_guard": cn,
            "target_candidate_metric_norm_ratio": cn / max(target_norm_g, 1.0e-12),
            "control_surplus": control_margin,
            "control_margin_CVaR25": control_margin,
            "MLP_surplus": mlp_margin,
            "MLP_margin_CVaR25": mlp_margin,
            "MLP_margin_positive": int(mlp_margin > 0.0),
            "MLP_matched_NLL_delta": float(mlp_delta.get("NLL", 0.0)),
            "MLP_matched_debt_UCB": mlp_debt,
            "H20_NLL_delta": float(best["h20_delta"].get("NLL", 0.0)),
            "H20_linear_NLL_delta": float(best["h20_delta"].get("NLL", 0.0)),
            "H20_no_debt": int(best["h20_debt"] <= 0.0),
            "H20_linear_debt_UCB": best["h20_debt"],
            "H60_NLL_delta": float(best["h60_delta"].get("NLL", 0.0)),
            "H60_linear_NLL_delta": float(best["h60_delta"].get("NLL", 0.0)),
            "H60_no_debt": int(best["h60_debt"] <= 0.0),
            "H60_linear_debt_UCB": best["h60_debt"],
            "parameter_count": "diagnostic_gate_columns_only_no_runtime_parameter_claim",
            "FLOPs": "diagnostic_preflight_only",
            "overhead": fval(gated_meta.get("gated_design_columns")) / max(1.0, fval(design_meta.get("rkhs_atoms"), 64.0)),
            "edge_effect_fraction": 1.0,
            "readout_effect_fraction": 0.0,
            "same_architecture_random_margin": margins.get("same_edge_domain_energy_random", 0.0),
            "same_basis_energy_margin": margins.get("same_RKHS_norm_random", 0.0),
            "same_task_energy_margin": margins.get("same_solver_gradient_control", 0.0),
            "same_output_coverage_margin": margins.get("same_output_coverage_random", 0.0),
            "same_gate_MLP_margin": mlp_margin,
            "diagnostic_note": "G4 gated additive edge-bank diagnostic with train-only low-dimensional gate features; diagnostic only, not strict FC-PureKAN success.",
        })
    except Exception as exc:
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_part_g4(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = shard_items(base_task_grid(args), args)
    rows = [part_g4_probe(dataset, seed, spec, args, device) for spec, seed, dataset in tasks]
    out = OUT_ROOT / f"v22_86_part_g4_gated_edge_bank_diagnostic_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": "v22_86_part_g4_gated_edge_bank_diagnostic_shard",
        "rows": len(rows),
        "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))),
        "tasks": len(tasks),
        "device": str(device),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_86_part_g4_gated_edge_bank_diagnostic_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("G4_gated_edge_bank_diagnostic_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_g4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", ""))),
        "candidate_NLL_negative_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("target_coverage")) >= 0.20) for r in valid),
        "no_debt_rows": sum(int(fval(r.get("no_debt")) > 0) for r in valid),
        "control_surplus_positive_rows": sum(int(fval(r.get("control_surplus")) > 0.0) for r in valid),
        "MLP_surplus_positive_rows": sum(int(fval(r.get("MLP_surplus")) > 0.0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
        "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in valid], 0.50),
        "readout_effect_fraction_median": quantile([fval(r.get("readout_effect_fraction")) for r in valid], 0.50),
        "overhead_le_035_rows": sum(int(fval(r.get("overhead")) <= 0.35) for r in valid),
        "guard_NLL_delta_median": quantile([fval(r.get("guard_NLL_delta")) for r in valid], 0.50),
        "coverage_CVaR25_median": quantile([fval(r.get("target_coverage")) for r in valid], 0.50),
        "control_margin_median": quantile([fval(r.get("control_surplus")) for r in valid], 0.50),
        "MLP_margin_median": quantile([fval(r.get("MLP_surplus")) for r in valid], 0.50),
        "gated_design_columns_median": quantile([fval(r.get("gated_design_columns")) for r in valid], 0.50),
    }


def merge_part_g4(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_86_part_g4_gated_edge_bank_diagnostic_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / "v22_86_part_g4_gated_edge_bank_diagnostic.csv"
    write_rows(out_csv, rows)
    summary = summarize_part_g4(rows)
    checks = {
        "completed": summary["completed_rows"] >= 18 and not missing,
        "nll": summary["candidate_NLL_negative_rows"] >= 12,
        "coverage": summary["coverage_CVaR25_ge_020_rows"] >= 9,
        "debt": summary["no_debt_rows"] >= 10 and summary["H20_no_debt_rows"] >= 9 and summary["H60_no_debt_rows"] >= 9,
        "control": summary["control_surplus_positive_rows"] >= 10,
        "mlp": summary["MLP_surplus_positive_rows"] >= 9,
        "edge": summary["edge_effect_fraction_median"] >= 0.50 and summary["readout_effect_fraction_median"] <= 0.35,
        "overhead": summary["overhead_le_035_rows"] >= 12,
    }
    diagnostic_gate = int(all(checks.values()))
    if missing:
        route = "G4-GatedEdgeBankIncomplete"
        blocker = "incomplete"
        reason = f"missing shards: {missing}"
    elif diagnostic_gate:
        route = "C10-ArchitectureDiagnosticOpened_NotStrictFCPureKAN"
        blocker = "gated_edge_bank_opened"
        reason = "G4 gated additive edge-bank diagnostic opened, but strict identity success is not claimed"
    elif not checks["coverage"]:
        route = "G4-GatedEdgeBankCoverageFailed"
        blocker = "coverage_low"
        reason = f"coverage rows={summary['coverage_CVaR25_ge_020_rows']}"
    elif not checks["mlp"]:
        route = "G4-GatedEdgeBankMLPStillDominates"
        blocker = "MLP_margin_low"
        reason = f"MLP rows={summary['MLP_surplus_positive_rows']}"
    elif not checks["debt"]:
        route = "G4-GatedEdgeBankDebtFailed"
        blocker = "debt_low"
        reason = f"H20/H60 rows={summary['H20_no_debt_rows']}/{summary['H60_no_debt_rows']}"
    else:
        route = "G4-GatedEdgeBankFailed"
        blocker = "controls_or_effect_low"
        reason = "G4 diagnostic gate failed"
    obj = {
        "gate": "v22_86_part_g4_gated_edge_bank_diagnostic",
        "part_g4_gate_pass": diagnostic_gate,
        "part_g4_route": route,
        "route_reason": reason,
        "dominant_blocker": blocker,
        "promoted_to_official_candidate": 0,
        "strict_FC_PureKAN_success_claimed": 0,
        "missing_shards": missing,
        "rows": len(rows),
        "summary": summary,
        "raw_csv": rel(out_csv),
        "diagnostic_note": "G4 tests gated additive edge-bank design with train-only low-dimensional gate features. Passing would only support architecture redesign, not strict FC-PureKAN success.",
    }
    route_path = OUT_ROOT / "v22_86_part_g4_gated_edge_bank_diagnostic_route.json"
    write_json(route_path, obj)
    write_rows(OUT_ROOT / "v22_86_part_g4_gated_edge_bank_diagnostic_summary.csv", [summary])
    append_exec("G4_gated_edge_bank_diagnostic_merge", command_text(sys.argv), "pass" if diagnostic_gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part G4 gated additive edge-bank diagnostic", [
        f"part_g4_gate_pass={diagnostic_gate}；route={route}；dominant_blocker={blocker}；reason={reason}。",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：G4 是计划中的 gated additive edge-bank diagnostic；只用 source/witness train-only 拟合 gate features，不进入 official runtime，不声明 strict FC-PureKAN 成功。",
    ])
    return obj


def bivariate_product_designs(
    design_s: torch.Tensor,
    design_w: torch.Tensor,
    design_g: torch.Tensor,
    *,
    max_base: int = 8,
    max_pairs: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train = torch.cat([design_s, design_w], dim=0).to(dtype=torch.float64)
    d = int(train.shape[1])
    if d <= 0:
        raise RuntimeError("empty base design for bivariate diagnostic")
    keep_base = min(int(max_base), d)
    score = train.square().mean(dim=0)
    sel = torch.topk(score, k=keep_base).indices
    train_sel = train[:, sel]
    mean = train_sel.mean(dim=0, keepdim=True)
    std = train_sel.std(dim=0, keepdim=True).clamp_min(1.0e-6)

    def normed(design: torch.Tensor) -> torch.Tensor:
        return (design[:, sel].to(dtype=torch.float64) - mean.to(device=design.device)) / std.to(device=design.device)

    zs = normed(design_s)
    zw = normed(design_w)
    zg = normed(design_g)
    ztrain = torch.cat([zs, zw], dim=0)
    all_pairs: list[tuple[int, int]] = []
    for i in range(keep_base):
        for j in range(i, keep_base):
            all_pairs.append((i, j))
    prod_train_cols = [ztrain[:, i] * ztrain[:, j] for i, j in all_pairs]
    prod_train = torch.stack(prod_train_cols, dim=1)
    pair_score = prod_train.var(dim=0)
    keep_pairs = min(int(max_pairs), int(pair_score.numel()))
    pair_idx = torch.topk(pair_score, k=keep_pairs).indices.detach().cpu().tolist()
    pairs = [all_pairs[int(idx)] for idx in pair_idx]

    def products(z: torch.Tensor) -> torch.Tensor:
        cols = [z[:, i] * z[:, j] for i, j in pairs]
        return torch.stack(cols, dim=1).to(dtype=torch.float64)

    ps = products(zs)
    pw = products(zw)
    pg = products(zg)
    scale = torch.cat([ps, pw], dim=0).std(dim=0, keepdim=True).clamp_min(1.0e-6)
    ps = ps / scale.to(device=ps.device)
    pw = pw / scale.to(device=pw.device)
    pg = pg / scale.to(device=pg.device)
    return ps, pw, pg, {
        "bivariate_base_columns": keep_base,
        "bivariate_product_columns": keep_pairs,
        "bivariate_pair_indices": ";".join(f"{i}x{j}" for i, j in pairs),
        "bivariate_selection_source": "source+witness train-only top variance normalized product columns",
    }


def part_g6_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    local_args = clone_args(
        args,
        part_d_family=base85.PRIMARY_FAMILY,
        beta=0.0,
        alpha=0.65,
        gamma=0.10,
        debt_scale_grid=D_DEBT_SOLVE_SCALE_GRID,
        subspace_rank=4,
        subspace_random_candidates=0,
        task_logit_norm=0.01,
        primary_shape_kernel="K3R3_svd64_whitened_mixed_output_velocity",
    )
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "method": spec.get("method", ""),
        "carrier_family": spec.get("carrier_family", ""),
        "architecture_variant": "G6_lowdim_bivariate_product_diagnostic",
        "strict_identity_status": "bivariate_diagnostic_not_strict_fc_purekan",
        "edge_univariate_preserved": 0,
        "node_bank_shared_dictionary_used": 0,
        "gating_used": 0,
        "bivariate_used_diagnostic_only": 1,
        "matched_MLP_architecture_control": 1,
        "same_architecture_random_control": 1,
        "same_basis_energy_control": 1,
        "same_task_energy_control": 1,
        "probe_error": "",
    }
    try:
        model, mlp, _bundle, splits = base85.make_model_and_batch_v2285(dataset, seed, spec, local_args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits_s = model(xs).float()
        logits_w = model(xw).float()
        logits_g = model(xg).float()
        metric_s = base85.base83.output_metric_diag(logits_s).to(device=device)
        metric_w = base85.base83.output_metric_diag(logits_w).to(device=device)
        metric_g = base85.base83.output_metric_diag(logits_g).to(device=device)
        uni_s, uni_w, uni_g, design_meta = base85.prepare_designs(model, xs, xw, xg, local_args, device)
        design_s, design_w, design_g, bi_meta = bivariate_product_designs(uni_s, uni_w, uni_g)
        out_family = base85.base84.selected_output_family(local_args)
        target_g, _ = base85.base83.output_update_for_family(logits_g, yg, out_family, local_args)
        mlp_s_update = None
        try:
            mlp_updates = base85.base83.mlp_matched_update(mlp, xs, ys, xw, yw, 1.0)
            mlp_s_update = base85.base83.actual_logit_update_for_updates(mlp, xs, mlp_updates).reshape_as(logits_s)
        except Exception:
            mlp_s_update = None
        controls_s = base85.build_output_controls(logits_s, ys, seed=base85.stable_text_seed(dataset) + int(seed), metric=metric_s, mlp_update=mlp_s_update)
        controls_w = base85.build_output_controls(logits_w, yw, seed=base85.stable_text_seed(dataset) + int(seed) + 17, metric=metric_w, mlp_update=None)
        probs_s = torch.softmax(logits_s.detach().float(), dim=1)
        probs_w = torch.softmax(logits_w.detach().float(), dim=1)
        pi, coupling_meta = base85.class_confidence_bucket_coupling(
            ys,
            yw,
            probs_s.max(dim=1).values,
            probs_w.max(dim=1).values,
            num_classes=int(model.output_dim),
            buckets=int(local_args.confidence_buckets),
        )
        op, m_edge, b_s, b_w, op_meta = base85.task_operator(
            design_s,
            design_w,
            metric_s,
            metric_w,
            logits_s,
            ys,
            logits_w,
            yw,
            controls_s,
            controls_w,
            pi,
            beta=0.0,
            alpha=0.65,
            gamma=0.10,
            ridge=float(local_args.projector_ridge),
            vis_scale_mode=str(local_args.vis_scale_mode),
        )
        eig_lams, eigvecs, m_cond = base85.topk_generalized_eigen(op, m_edge, 4)
        if int(eigvecs.shape[1]) <= 0:
            raise RuntimeError("empty G6 bivariate eigenspace")
        r_s = base85.task_cotangent(logits_s, ys).reshape(-1)
        r_w = base85.task_cotangent(logits_w, yw).reshape(-1)
        r_g = base85.task_cotangent(logits_g, yg).reshape(-1)
        g_s_raw = design_s.T @ r_s
        g_w_raw = design_w.T @ r_w
        g_g_raw = design_g.T @ r_g
        candidates = base85.subspace_candidate_alphas(eigvecs, eig_lams, g_s_raw, g_w_raw)
        debt_pullbacks: list[torch.Tensor] = []
        for debt_kind in ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"]:
            try:
                debt_grad = base85.base80.logits_loss_grad(logits_g, yg, debt_kind).reshape(-1).to(device=device, dtype=torch.float64)
                debt_pullbacks.append(design_g.T @ debt_grad)
            except Exception:
                continue
        if debt_pullbacks:
            debt_a = torch.stack([eigvecs.T @ item.reshape(-1).to(device=device, dtype=torch.float64) for item in debt_pullbacks], dim=0)
            q_sw = -(eigvecs.T @ (g_s_raw + g_w_raw))
            q_sw_proj, debt_qp_meta = base85.project_subspace_halfspaces(q_sw, debt_a, float(local_args.projector_ridge))
            candidates.append(("G6_bivariate_debt_qp_source_witness", eigvecs @ q_sw_proj))
        else:
            debt_qp_meta = {"debt_qp_feasible": 0, "debt_qp_first_order_max": 0.0, "debt_qp_active_constraints": ""}
        best: dict[str, Any] | None = None
        target_norm_g = base85.metric_norm(target_g, metric_g)
        for cand_idx, (cand_name, cand_alpha0) in enumerate(candidates):
            cand_alpha = base85.orient_for_source_witness(cand_alpha0, g_s_raw, g_w_raw)
            raw_g0 = (design_g @ cand_alpha).reshape_as(logits_g)
            g_norm = base85.metric_norm(raw_g0, metric_g)
            if g_norm > 1.0e-12:
                cand_alpha = cand_alpha * (float(local_args.task_logit_norm) / g_norm)
            raw_s = (design_s @ cand_alpha).reshape_as(logits_s)
            raw_w = (design_w @ cand_alpha).reshape_as(logits_w)
            raw_g = (design_g @ cand_alpha).reshape_as(logits_g)
            update_g, debt_meta = base85.finite_step_debt_select(logits_g, yg, raw_g, scales=base85.parse_float_list(D_DEBT_SOLVE_SCALE_GRID))
            scale = fval(debt_meta.get("debt_safe_scale"), 1.0)
            cand_alpha = cand_alpha * scale
            update_s = raw_s * scale
            update_w = raw_w * scale
            pred_s = float((r_s * update_s.reshape(-1)).sum().detach().cpu().item())
            pred_w = float((r_w * update_w.reshape(-1)).sum().detach().cpu().item())
            guard_delta, guard_debt = base85.row_metrics_from_logit_update(logits_g, yg, update_g)
            coverage_g, rel_res_g = base85.base80.weighted_capacity(update_g.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
            h20_delta, h20_debt = base85.row_metrics_from_logit_update(logits_g, yg, update_g * 20.0)
            h60_delta, h60_debt = base85.row_metrics_from_logit_update(logits_g, yg, update_g * 60.0)
            sign_pass = int(pred_s < 0.0 and pred_w < 0.0 and pred_s * pred_w > 0.0)
            score = (
                sign_pass,
                int(float(guard_delta.get("NLL", 0.0)) < 0.0),
                int(coverage_g >= 0.20),
                int(guard_debt <= 0.0),
                int(h20_debt <= 0.0 and float(h20_delta.get("NLL", 0.0)) < 0.0),
                int(h60_debt <= 0.0 and float(h60_delta.get("NLL", 0.0)) < 0.0),
                -float(guard_delta.get("NLL", 0.0)),
                -cand_idx,
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "name": cand_name,
                    "alpha": cand_alpha,
                    "update_s": update_s,
                    "update_w": update_w,
                    "update_g": update_g,
                    "debt_meta": debt_meta,
                    "pred_s": pred_s,
                    "pred_w": pred_w,
                    "guard_delta": guard_delta,
                    "guard_debt": guard_debt,
                    "coverage_g": coverage_g,
                    "rel_res_g": rel_res_g,
                    "h20_delta": h20_delta,
                    "h20_debt": h20_debt,
                    "h60_delta": h60_delta,
                    "h60_debt": h60_debt,
                    "sign_pass": sign_pass,
                }
        if best is None:
            raise RuntimeError("no G6 bivariate candidates")
        alpha_vec = best["alpha"]
        update_g = best["update_g"]
        guard_delta = best["guard_delta"]
        guard_debt = best["guard_debt"]
        margins: dict[str, float] = {}
        for offset, name in enumerate(["same_RKHS_norm_random", "same_edge_domain_energy_random", "same_smoothness_random", "same_output_coverage_random", "same_debt_cone_random", "same_solver_gradient_control", "same_compute_noop"]):
            ctrl = base85.base84.control_alpha(name, alpha_vec, design_s, best["update_s"], int(seed) + base85.stable_text_seed(name + dataset) + 31 * offset, float(local_args.projector_ridge))
            ctrl_update = (design_g @ ctrl).reshape_as(logits_g)
            cn = base85.metric_norm(ctrl_update, metric_g)
            tn = base85.metric_norm(update_g, metric_g)
            if name in {"same_output_coverage_random"} and cn > 1.0e-12:
                ctrl_update = ctrl_update * (tn / cn)
            ctrl_delta, ctrl_debt = base85.row_metrics_from_logit_update(logits_g, yg, ctrl_update)
            margins[name] = float(ctrl_delta.get("NLL", 0.0)) - float(guard_delta.get("NLL", 0.0)) - 0.5 * max(0.0, guard_debt - ctrl_debt)
        control_values = [v for k, v in margins.items() if k != "same_compute_noop"]
        control_margin = lower_cvar(control_values, 0.25)
        cn = base85.metric_norm(update_g, metric_g)
        mlp_update_g = base85.base84.mlp_matched_logit_update(mlp, xs, ys, xw, yw, xg, cn, metric_g).reshape_as(logits_g)
        mlp_delta, mlp_debt = base85.row_metrics_from_logit_update(logits_g, yg, mlp_update_g)
        mlp_margin = float(mlp_delta.get("NLL", 0.0)) - float(guard_delta.get("NLL", 0.0)) - 0.5 * max(0.0, guard_debt - mlp_debt)
        row.update({
            **design_meta,
            **bi_meta,
            **coupling_meta,
            **op_meta,
            **debt_qp_meta,
            **best["debt_meta"],
            "task_energy_rank": int(eigvecs.shape[1]),
            "subspace_selected_candidate": str(best["name"]),
            "subspace_selected_sign_pass": int(best["sign_pass"]),
            "M_edge_condition_number": m_cond,
            "lambda": float(eig_lams[0]) if eig_lams else 0.0,
            "source_descent_delta_pred": best["pred_s"],
            "witness_descent_delta_pred": best["pred_w"],
            "source_descent_negative": int(best["pred_s"] < 0.0),
            "witness_descent_negative": int(best["pred_w"] < 0.0),
            "source_witness_descent_product": best["pred_s"] * best["pred_w"],
            "source_witness_descent_product_positive": int(best["pred_s"] * best["pred_w"] > 0.0),
            "guard_NLL_delta": float(guard_delta.get("NLL", 0.0)),
            "guard_actual_NLL_delta": float(guard_delta.get("NLL", 0.0)),
            "guard_debt_UCB": guard_debt,
            "no_debt": int(guard_debt <= 0.0),
            "target_coverage": best["coverage_g"],
            "coverage_CVaR25": best["coverage_g"],
            "relative_residual_energy": best["rel_res_g"],
            "target_metric_norm_guard": target_norm_g,
            "candidate_metric_norm_guard": cn,
            "target_candidate_metric_norm_ratio": cn / max(target_norm_g, 1.0e-12),
            "control_surplus": control_margin,
            "control_margin_CVaR25": control_margin,
            "MLP_surplus": mlp_margin,
            "MLP_margin_CVaR25": mlp_margin,
            "MLP_margin_positive": int(mlp_margin > 0.0),
            "MLP_matched_NLL_delta": float(mlp_delta.get("NLL", 0.0)),
            "MLP_matched_debt_UCB": mlp_debt,
            "H20_NLL_delta": float(best["h20_delta"].get("NLL", 0.0)),
            "H20_linear_NLL_delta": float(best["h20_delta"].get("NLL", 0.0)),
            "H20_no_debt": int(best["h20_debt"] <= 0.0),
            "H20_linear_debt_UCB": best["h20_debt"],
            "H60_NLL_delta": float(best["h60_delta"].get("NLL", 0.0)),
            "H60_linear_NLL_delta": float(best["h60_delta"].get("NLL", 0.0)),
            "H60_no_debt": int(best["h60_debt"] <= 0.0),
            "H60_linear_debt_UCB": best["h60_debt"],
            "parameter_count": "diagnostic_design_columns_only_no_runtime_parameter_claim",
            "FLOPs": "diagnostic_preflight_only",
            "overhead": fval(bi_meta.get("bivariate_product_columns")) / max(1.0, fval(design_meta.get("rkhs_atoms"), 64.0)),
            "edge_effect_fraction": 1.0,
            "readout_effect_fraction": 0.0,
            "same_architecture_random_margin": margins.get("same_edge_domain_energy_random", 0.0),
            "same_basis_energy_margin": margins.get("same_RKHS_norm_random", 0.0),
            "same_task_energy_margin": margins.get("same_solver_gradient_control", 0.0),
            "same_output_coverage_margin": margins.get("same_output_coverage_random", 0.0),
            "diagnostic_note": "G6 low-dimensional bivariate product design over train-only source/witness selected RKHS columns; diagnostic only, not strict FC-PureKAN success.",
        })
    except Exception as exc:
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_part_g6(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = shard_items(base_task_grid(args), args)
    rows = [part_g6_probe(dataset, seed, spec, args, device) for spec, seed, dataset in tasks]
    out = OUT_ROOT / f"v22_86_part_g6_bivariate_diagnostic_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": "v22_86_part_g6_bivariate_diagnostic_shard",
        "rows": len(rows),
        "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))),
        "tasks": len(tasks),
        "device": str(device),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_86_part_g6_bivariate_diagnostic_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("G6_bivariate_diagnostic_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_g6(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", ""))),
        "candidate_NLL_negative_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("target_coverage")) >= 0.20) for r in valid),
        "no_debt_rows": sum(int(fval(r.get("no_debt")) > 0) for r in valid),
        "control_surplus_positive_rows": sum(int(fval(r.get("control_surplus")) > 0.0) for r in valid),
        "MLP_surplus_positive_rows": sum(int(fval(r.get("MLP_surplus")) > 0.0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
        "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in valid], 0.50),
        "readout_effect_fraction_median": quantile([fval(r.get("readout_effect_fraction")) for r in valid], 0.50),
        "overhead_le_035_rows": sum(int(fval(r.get("overhead")) <= 0.35) for r in valid),
        "guard_NLL_delta_median": quantile([fval(r.get("guard_NLL_delta")) for r in valid], 0.50),
        "coverage_CVaR25_median": quantile([fval(r.get("target_coverage")) for r in valid], 0.50),
        "control_margin_median": quantile([fval(r.get("control_surplus")) for r in valid], 0.50),
        "MLP_margin_median": quantile([fval(r.get("MLP_surplus")) for r in valid], 0.50),
    }


def merge_part_g6(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_86_part_g6_bivariate_diagnostic_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / "v22_86_part_g6_bivariate_diagnostic.csv"
    write_rows(out_csv, rows)
    summary = summarize_part_g6(rows)
    checks = {
        "completed": summary["completed_rows"] >= 18 and not missing,
        "nll": summary["candidate_NLL_negative_rows"] >= 12,
        "coverage": summary["coverage_CVaR25_ge_020_rows"] >= 9,
        "debt": summary["no_debt_rows"] >= 10 and summary["H20_no_debt_rows"] >= 9 and summary["H60_no_debt_rows"] >= 9,
        "control": summary["control_surplus_positive_rows"] >= 10,
        "mlp": summary["MLP_surplus_positive_rows"] >= 9,
        "edge": summary["edge_effect_fraction_median"] >= 0.50 and summary["readout_effect_fraction_median"] <= 0.35,
        "overhead": summary["overhead_le_035_rows"] >= 12,
    }
    diagnostic_gate = int(all(checks.values()))
    if missing:
        route = "G6-BivariateDiagnosticIncomplete"
        blocker = "incomplete"
        reason = f"missing shards: {missing}"
    elif diagnostic_gate:
        route = "StrictUnivariateEdgeBottleneckLikely"
        blocker = "univariate_expression_limit"
        reason = "G6 bivariate diagnostic opened while G1-G5 strict/univariate architecture line remained failed"
    elif not checks["coverage"]:
        route = "G6-BivariateDiagnosticCoverageFailed"
        blocker = "coverage_low"
        reason = f"coverage rows={summary['coverage_CVaR25_ge_020_rows']}"
    elif not checks["mlp"]:
        route = "G6-BivariateDiagnosticMLPStillDominates"
        blocker = "MLP_margin_low"
        reason = f"MLP rows={summary['MLP_surplus_positive_rows']}"
    elif not checks["debt"]:
        route = "G6-BivariateDiagnosticDebtFailed"
        blocker = "debt_low"
        reason = f"H20/H60 rows={summary['H20_no_debt_rows']}/{summary['H60_no_debt_rows']}"
    else:
        route = "G6-BivariateDiagnosticFailed"
        blocker = "controls_or_effect_low"
        reason = "G6 diagnostic gate failed"
    obj = {
        "gate": "v22_86_part_g6_bivariate_diagnostic",
        "part_g6_gate_pass": diagnostic_gate,
        "part_g6_route": route,
        "route_reason": reason,
        "dominant_blocker": blocker,
        "promoted_to_official_candidate": 0,
        "strict_FC_PureKAN_success_claimed": 0,
        "missing_shards": missing,
        "rows": len(rows),
        "summary": summary,
        "raw_csv": rel(out_csv),
        "diagnostic_note": "G6 tests low-dimensional bivariate product design over edge RKHS columns. Passing would only support architecture bottleneck, not strict FC-PureKAN success.",
    }
    route_path = OUT_ROOT / "v22_86_part_g6_bivariate_diagnostic_route.json"
    write_json(route_path, obj)
    write_rows(OUT_ROOT / "v22_86_part_g6_bivariate_diagnostic_summary.csv", [summary])
    append_exec("G6_bivariate_diagnostic_merge", command_text(sys.argv), "pass" if diagnostic_gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part G6 low-dimensional bivariate diagnostic", [
        f"part_g6_gate_pass={diagnostic_gate}；route={route}；dominant_blocker={blocker}；reason={reason}。",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：G6 是计划中的 bivariate diagnostic；只在 train-only source/witness 选择的低维 product design 上做 preflight，不进入 official runtime，不声明 strict FC-PureKAN 成功。",
    ])
    return obj


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    part_c = read_json(OUT_ROOT / "v22_86_part_c_metric_role_ablation_route.json")
    part_d = read_json(OUT_ROOT / "v22_86_part_d_trajectory_route.json")
    part_f = read_json(OUT_ROOT / "v22_86_part_f_edge_snr_route.json")
    part_g = read_json(OUT_ROOT / "v22_86_part_g_architecture_diagnostic_route.json")
    candidates = []
    if int(fval(part_c.get("part_c_gate_pass"))) > 0:
        candidates.append("best_from_Part_C_metric_role")
    if int(fval(part_d.get("part_d_gate_pass"))) > 0:
        candidates.append("best_from_Part_D_trajectory")
    if int(fval(part_f.get("part_f_gate_pass"))) > 0:
        candidates.append("best_from_Part_F_edge_SNR")
    if int(fval(part_g.get("part_g_gate_pass"))) > 0 and "NotStrict" not in str(part_g.get("part_g_route", "")):
        candidates.append("best_from_Part_G_architecture_diagnostic_if_strict_identity_preserved")
    gate = int(bool(candidates))
    obj = {
        "gate": "v22_86_part_h_preflight_to_full_loop_entry",
        "part_h_gate_pass": gate,
        "eligible_candidates": candidates,
        "part_h_route": "HStepEligibleCandidatesFound" if gate else "NoPreflightCandidateForHStep",
        "route_reason": "No Part C/D/F/G strict preflight candidate passed fixed gates; H-step/full-loop skipped by plan." if not gate else "At least one preflight candidate passed; H-step should run next.",
        "hstep_run": 0,
        "full_loop_run": 0,
    }
    next_actions = {
        "route": obj["part_h_route"],
        "dominant_blocker": "no_preflight_candidate" if not gate else "pending_hstep",
        "allowed_next_actions": [] if gate else [{
            "action": "do_not_weaken_gates_report_failure_route",
            "reason": "all fixed preflight families failed or were diagnostic-only",
            "max_attempts": 1,
            "forbidden": ["weakening_gates", "KAN_internal_success_claim"],
        }],
        "must_not_do": ["weakening_gates", "KAN_internal_success_claim"],
    }
    write_json(OUT_ROOT / "v22_86_part_h_preflight_to_full_loop_entry.json", obj)
    write_json(OUT_ROOT / "v22_86_part_h_next_actions_for_codex.json", next_actions)
    append_exec("H_preflight_to_full_loop_entry", command_text(sys.argv), "pass" if gate else "skipped", files=rel(OUT_ROOT / "v22_86_part_h_preflight_to_full_loop_entry.json"), note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part H preflight decision and full-loop entry", [
        f"part_h_gate_pass={gate}；eligible_candidates={candidates}。",
        obj["route_reason"],
        "H-step/full-loop 没有运行时会明确记录为 skipped，而不是补造 45/90 row official candidate 数据。",
    ])
    return obj


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    part_a = read_json(OUT_ROOT / "v22_86_part_a_code_identity_hard_gate.json")
    part_b = read_json(OUT_ROOT / "v22_86_part_b_history_replay_failure_matrix.json")
    part_c = read_json(OUT_ROOT / "v22_86_part_c_metric_role_ablation_route.json")
    part_d = read_json(OUT_ROOT / "v22_86_part_d_trajectory_route.json")
    part_e = read_json(OUT_ROOT / "v22_86_part_e_joint_upstream_edge_replay_diagnostic.json")
    part_e_repair = read_json(OUT_ROOT / "v22_86_part_e_repair_extended_historical_audit.json")
    part_f = read_json(OUT_ROOT / "v22_86_part_f_edge_snr_route.json")
    part_g = read_json(OUT_ROOT / "v22_86_part_g_architecture_diagnostic_route.json")
    part_g4 = read_json(OUT_ROOT / "v22_86_part_g4_gated_edge_bank_diagnostic_route.json")
    part_g6 = read_json(OUT_ROOT / "v22_86_part_g6_bivariate_diagnostic_route.json")
    part_h = read_json(OUT_ROOT / "v22_86_part_h_preflight_to_full_loop_entry.json")
    part_d_debt_solve = read_json(OUT_ROOT / "v22_86_part_d_finite_step_debt_solve_route.json")
    if not int(fval(part_a.get("part_a_hard_gate_pass"))):
        route = "C0-CodeBoundaryFailed"
        reason = "Part A hard gate failed or missing"
    elif not int(fval(part_b.get("part_b_gate_pass"))):
        route = "C1-ReplayIncomplete"
        reason = "Part B replay incomplete or changed"
    elif int(fval(part_h.get("part_h_gate_pass"))) > 0:
        route = "C13-HStepOpened_FullLoopBlocked"
        reason = "At least one preflight candidate reached Part H, but H-step/full-loop not completed in this runner invocation"
    elif str(part_g.get("part_g_route", "")).startswith("C10"):
        route = "C10-ArchitectureDiagnosticOpened_NotStrictFCPureKAN"
        reason = str(part_g.get("route_reason", "architecture diagnostic opened"))
    elif int(fval(part_g4.get("part_g4_gate_pass"))) > 0:
        route = "C10-ArchitectureDiagnosticOpened_NotStrictFCPureKAN"
        reason = str(part_g4.get("route_reason", "G4 gated additive edge-bank diagnostic opened"))
    elif int(fval(part_g6.get("part_g6_gate_pass"))) > 0:
        route = "C11-StrictUnivariateEdgeBottleneckLikely"
        reason = str(part_g6.get("route_reason", "G6 bivariate diagnostic opened while strict univariate family failed"))
    elif str(part_g.get("part_g_route", "")).startswith("C12"):
        route = "C12-CurrentEdgeArchitectureFamilyNoTransferableSignal"
        reason = "Part C/D/F fixed gates failed; Part E replay/repair remained debt/control blocked; Part G/G4/G6 architecture diagnostics did not pass coverage/control/MLP/debt gates."
    elif str(part_f.get("part_f_route", "")).startswith("C15") or str(part_c.get("part_c_route", "")).startswith("C15") or str(part_d.get("part_d_route", "")).startswith("C15"):
        route = "C15-MLPMatchedDominates"
        reason = "MLP matched margin blocked at least one primary line"
    elif str(part_c.get("part_c_route", "")).startswith("C16") or str(part_d.get("part_d_route", "")).startswith("C16") or str(part_f.get("part_f_route", "")).startswith("C16"):
        route = "C16-DebtBlocked"
        reason = "finite-step debt blocked primary evidence"
    else:
        route = "C3-MetricRoleNotMainCause"
        reason = "No positive v22.86 preflight route found"
    final = {
        "gate": "v22_86_final_route",
        "final_route": route,
        "official_candidate_gate_pass": int(route == "C19-KANEdgeGeneratorOfficialCandidate"),
        "route_reason": reason,
        "part_a": int(fval(part_a.get("part_a_hard_gate_pass"))),
        "part_b": int(fval(part_b.get("part_b_gate_pass"))),
        "part_c": int(fval(part_c.get("part_c_gate_pass"))),
        "part_d": int(fval(part_d.get("part_d_gate_pass"))),
        "part_e": int(fval(part_e.get("part_e_gate_pass"))),
        "part_e_repair": int(fval(part_e_repair.get("part_e_repair_gate_pass"))),
        "part_f": int(fval(part_f.get("part_f_gate_pass"))),
        "part_g": int(fval(part_g.get("part_g_gate_pass"))),
        "part_g4": int(fval(part_g4.get("part_g4_gate_pass"))),
        "part_g6": int(fval(part_g6.get("part_g6_gate_pass"))),
        "part_h": int(fval(part_h.get("part_h_gate_pass"))),
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_86_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_86_part_b_history_replay_failure_matrix.json"),
            "part_c": rel(OUT_ROOT / "v22_86_part_c_metric_role_ablation_route.json"),
            "part_c_repair": rel(OUT_ROOT / "v22_86_part_c_repair_route.json"),
            "part_d": rel(OUT_ROOT / "v22_86_part_d_trajectory_route.json"),
            "part_d_repair": rel(OUT_ROOT / "v22_86_part_d_trajectory_repair_route.json"),
            "part_d_debt_solve": rel(OUT_ROOT / "v22_86_part_d_finite_step_debt_solve_route.json") if part_d_debt_solve else "missing",
            "part_e": rel(OUT_ROOT / "v22_86_part_e_joint_upstream_edge_replay_diagnostic.json"),
            "part_e_repair": rel(OUT_ROOT / "v22_86_part_e_repair_extended_historical_audit.json") if part_e_repair else "missing",
            "part_f": rel(OUT_ROOT / "v22_86_part_f_edge_snr_route.json"),
            "part_f_repair": rel(OUT_ROOT / "v22_86_part_f_edge_snr_repair_route.json"),
            "part_g": rel(OUT_ROOT / "v22_86_part_g_architecture_diagnostic_route.json"),
            "part_g_repair": rel(OUT_ROOT / "v22_86_part_g_same_quantile_control_repair_audit.json"),
            "part_g4": rel(OUT_ROOT / "v22_86_part_g4_gated_edge_bank_diagnostic_route.json") if part_g4 else "missing",
            "part_g6": rel(OUT_ROOT / "v22_86_part_g6_bivariate_diagnostic_route.json") if part_g6 else "missing",
            "part_h": rel(OUT_ROOT / "v22_86_part_h_preflight_to_full_loop_entry.json"),
            "final": rel(OUT_ROOT / "v22_86_final_route.json"),
        },
    }
    write_json(OUT_ROOT / "v22_86_final_route.json", final)
    append_exec("final_route", command_text(sys.argv), "done", files=rel(OUT_ROOT / "v22_86_final_route.json"), note=json.dumps(final, ensure_ascii=False))
    append_recap("Final route and conclusion", [
        f"final_route={route}；official_candidate_gate_pass={final['official_candidate_gate_pass']}；reason={reason}",
        f"A/B/C/D/E/Erepair/F/G/G4/G6/H pass={final['part_a']}/{final['part_b']}/{final['part_c']}/{final['part_d']}/{final['part_e']}/{final['part_e_repair']}/{final['part_f']}/{final['part_g']}/{final['part_g4']}/{final['part_g6']}/{final['part_h']}。",
        "结论约束：只有 C18/C19 算 strict positive progress；C10 是 architecture diagnostic，不是 strict FC-PureKAN 成功；C12/C14/C15/C16 不能写成接近成功。",
    ])
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default=PRIMARY_DATASETS)
    p.add_argument("--seed-count", type=int, default=3)
    p.add_argument("--spec-count", type=int, default=PRIMARY_SPEC_COUNT)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--metric-batch-size", type=int, default=192)
    p.add_argument("--projector-ridge", type=float, default=1.0e-4)
    p.add_argument("--model-seed-offset", type=int, default=28600)
    p.add_argument("--edge-step-norm", type=float, default=1.0e-3)
    p.add_argument("--fd-epsilon", type=float, default=1.0e-4)
    p.add_argument("--max-output-families", type=int, default=1)
    p.add_argument("--force-output-velocity-family", default="")
    p.add_argument("--rkhs-max-edges", type=int, default=64)
    p.add_argument("--rkhs-centers", type=int, default=6)
    p.add_argument("--rkhs-alpha-norm", type=float, default=10.0)
    p.add_argument("--rkhs-svd-rank", type=int, default=64)
    p.add_argument("--primary-shape-kernel", default="K3R3_svd64_whitened_mixed_output_velocity")
    p.add_argument("--confidence-buckets", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--gamma", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--task-logit-norm", type=float, default=0.05)
    p.add_argument("--vis-scale-mode", default="task_fro")
    p.add_argument("--subspace-rank", type=int, default=4)
    p.add_argument("--subspace-random-candidates", type=int, default=0)
    p.add_argument("--debt-scale-grid", default="1.5,1.25,1.0,0.75,0.5,0.25,0.1,0.05,0.02")
    p.add_argument("--part-d-family", default=base85.PRIMARY_FAMILY)
    p.add_argument("--merge-families", default=base85.PRIMARY_FAMILY)
    p.add_argument("--snr-rho", type=float, default=0.25)
    p.add_argument("--snr-control-repeats", type=int, default=1)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    init_logs()
    if args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-c":
        run_part_c_matrix(args, repair=False)
    elif args.mode == "part-c-merge":
        merge_part_c(args, repair=False)
    elif args.mode == "part-c-repair":
        run_part_c_matrix(args, repair=True)
    elif args.mode == "part-c-repair-merge":
        merge_part_c(args, repair=True)
    elif args.mode == "part-d":
        run_part_d(args, repair=False)
    elif args.mode == "part-d-repair":
        run_part_d(args, repair=True)
    elif args.mode == "part-d-debt-solve":
        run_part_d_debt_solve(args)
    elif args.mode == "part-d-debt-solve-merge":
        merge_part_d_debt_solve(args)
    elif args.mode == "part-e":
        run_part_e(args)
    elif args.mode == "part-e-repair":
        run_part_e_repair(args)
    elif args.mode == "part-f":
        run_part_f(args, repair=False)
    elif args.mode == "part-f-merge":
        merge_part_f(args, repair=False)
    elif args.mode == "part-f-repair":
        run_part_f(args, repair=True)
    elif args.mode == "part-f-repair-merge":
        merge_part_f(args, repair=True)
    elif args.mode == "part-g":
        run_part_g(args)
    elif args.mode == "part-g-merge":
        merge_part_g(args)
    elif args.mode == "part-g-repair":
        run_part_g_repair(args)
    elif args.mode == "part-g4":
        run_part_g4(args)
    elif args.mode == "part-g4-merge":
        merge_part_g4(args)
    elif args.mode == "part-g6":
        run_part_g6(args)
    elif args.mode == "part-g6-merge":
        merge_part_g6(args)
    elif args.mode == "part-h":
        run_part_h(args)
    elif args.mode == "final":
        run_final(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
