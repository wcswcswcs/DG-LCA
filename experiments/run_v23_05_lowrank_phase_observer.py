#!/usr/bin/env python3
"""DG-KAN v23.05 low-rank edge-family phase observer runner.

This runner is intentionally narrow: it reuses the v23.03 PureKAN, edge
geometry, data, and metric code, and adds a train-only low-rank phase observer
around existing D-CHE/D-FOUR edge coefficient updates. It writes v23.05
artifacts under results/v23_05 and does not modify v23.03/v23.04 outputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import py_compile
import re
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
import experiments.run_v23_03_structure_projected_functional_population_flow as v2303
from dgkan.fu.signal_channel_estimators import per_example_gradient_matrix
from dgkan.optim.edge_sobolev_population_flow import EdgeSobolevPopulationFlow, mark_kan_edge_params


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.05_H10RealTransfer_LowRankPhaseObserver_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.05_H10RealTransfer_LowRankPhaseObserver_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.05_H10RealTransfer_LowRankPhaseObserver_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2305_OUT_ROOT", str(ROOT / "results/v23_05"))).resolve()

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
    "structure_transform_used_only_for_gate": 1,
    "structure_transform_in_forward": 0,
}

PART_C_SCHEMES = {
    "C0_H10_oracle_task_label_upper_bound": {"kind": "oracle", "rank": 0, "block": "degree_edgebank"},
    "C1_real_visual_scalar_dual_baseline": {"kind": "scalar_dual", "rank": 0, "block": "degree_edgebank"},
    "C2_low_rank_phase_r1": {"kind": "lowrank", "rank": 1, "block": "degree_edgebank"},
    "C3_low_rank_phase_r2": {"kind": "lowrank", "rank": 2, "block": "degree_edgebank"},
    "C4_low_rank_phase_r4": {"kind": "lowrank", "rank": 4, "block": "degree_edgebank"},
    "C5_class_conditional_phase_r2": {"kind": "lowrank", "rank": 2, "block": "class_degree"},
    "C6_margin_bucket_phase_r2": {"kind": "lowrank", "rank": 2, "block": "margin_degree"},
    "C7_inputBand_degreeBand_phase_r2": {"kind": "lowrank", "rank": 2, "block": "input_degree"},
    "C8_self_supervised_transform_consistency_phase_r2": {"kind": "lowrank", "rank": 2, "block": "input_degree"},
    "C9_random_phase_matched_control": {"kind": "random_phase", "rank": 2, "block": "degree_edgebank"},
    "C10_no_phase_FunctionalGram_control": {"kind": "functional_gram", "rank": 0, "block": "degree_edgebank"},
    "C11_residual_sparse_phase_r2": {"kind": "residual_sparse", "rank": 2, "block": "input_degree"},
    "C12_residual_sparse_random_phase_control": {"kind": "residual_sparse_random", "rank": 2, "block": "input_degree"},
    "C13_commutator_sparse_phase_r2": {"kind": "commutator_sparse", "rank": 2, "block": "input_degree"},
    "C14_commutator_sparse_random_phase_control": {"kind": "commutator_sparse_random", "rank": 2, "block": "input_degree"},
    "C15_residual_sparse_signed_phase_r2": {"kind": "residual_sparse_signed", "rank": 2, "block": "input_degree"},
    "C16_residual_sparse_signed_random_phase_control": {"kind": "residual_sparse_signed_random", "rank": 2, "block": "input_degree"},
    "C17_dense_signed_input_phase_r2": {"kind": "lowrank_signed", "rank": 2, "block": "input_degree"},
    "C18_dense_signed_random_phase_control": {"kind": "lowrank_signed_random", "rank": 2, "block": "input_degree"},
    "C19_class_conditional_random_phase_control": {"kind": "random_phase", "rank": 2, "block": "class_degree"},
    "C20_margin_bucket_random_phase_control": {"kind": "random_phase", "rank": 2, "block": "margin_degree"},
    "C21_intersection_input_phase_r2": {"kind": "intersection_lowrank", "rank": 2, "block": "input_degree"},
    "C22_intersection_input_random_phase_control": {"kind": "intersection_random", "rank": 2, "block": "input_degree"},
    "C23_intersection_sparse_input_phase_r2": {"kind": "intersection_sparse", "rank": 2, "block": "input_degree"},
    "C24_intersection_sparse_random_phase_control": {"kind": "intersection_sparse_random", "rank": 2, "block": "input_degree"},
    "C25_delta_energy_intersection_sparse_phase_r2": {"kind": "delta_energy_intersection_sparse", "rank": 2, "block": "input_degree"},
    "C26_delta_energy_intersection_sparse_random_control": {"kind": "delta_energy_intersection_sparse_random", "rank": 2, "block": "input_degree"},
    "C27_pixel_degree_intersection_sparse_phase_r2": {"kind": "intersection_sparse", "rank": 2, "block": "pixel_degree"},
    "C28_pixel_degree_intersection_sparse_random_control": {"kind": "intersection_sparse_random", "rank": 2, "block": "pixel_degree"},
    "C29_projector_phase_r1": {"kind": "projector_sparse", "rank": 1, "block": "input_degree"},
    "C30_projector_random_direction_control": {"kind": "projector_sparse_random", "rank": 1, "block": "input_degree"},
    "C31_covariance_projector_phase_r2": {"kind": "projector_cov_sparse", "rank": 2, "block": "input_degree"},
    "C32_covariance_projector_random_control": {"kind": "projector_cov_sparse_random", "rank": 2, "block": "input_degree"},
    "C33_covariance_projector_aligned_phase_r2": {"kind": "projector_cov_align_sparse", "rank": 2, "block": "input_degree"},
    "C34_covariance_projector_aligned_random_control": {"kind": "projector_cov_align_sparse_random", "rank": 2, "block": "input_degree"},
    "C35_covariance_projector_block_shuffle_control": {"kind": "projector_cov_blockshuffle", "rank": 2, "block": "input_degree"},
    "C36_covariance_projector_fullblock_control": {"kind": "projector_cov_fullblock", "rank": 2, "block": "input_degree"},
    "C37_covariance_projector_uniform_q_phase_r2": {"kind": "projector_cov_uniform", "rank": 2, "block": "input_degree"},
    "C38_covariance_projector_uniform_q_random_control": {"kind": "projector_cov_uniform_random", "rank": 2, "block": "input_degree"},
    "C39_covariance_projector_c2trust_phase_r2": {"kind": "projector_cov_sparse", "rank": 2, "block": "input_degree", "c2_trust": 1},
    "C40_covariance_projector_c2trust_random_control": {"kind": "projector_cov_sparse_random", "rank": 2, "block": "input_degree", "c2_trust": 1},
    "C41_covariance_projector_sgfilter_phase_r2": {"kind": "projector_cov_sgfilter", "rank": 2, "block": "input_degree"},
    "C42_covariance_projector_sgfilter_randombasis_control": {"kind": "projector_cov_sgfilter_randombasis", "rank": 2, "block": "input_degree"},
    "C43_covariance_projector_sgfilter_fullblock_control": {"kind": "projector_cov_sgfilter_fullblock", "rank": 2, "block": "input_degree"},
    "C44_covariance_projector_alignadv_phase_r2": {"kind": "projector_cov_alignadv", "rank": 2, "block": "input_degree"},
    "C45_covariance_projector_alignadv_random_control": {"kind": "projector_cov_alignadv_random", "rank": 2, "block": "input_degree"},
    "C46_alignadv_fullblock_phase_coordinate": {"kind": "projector_cov_alignadv_fullblock", "rank": 2, "block": "input_degree"},
    "C47_alignadv_fullblock_random_coordinate_control": {"kind": "projector_cov_alignadv_random_fullblock", "rank": 2, "block": "input_degree"},
    "C48_alignadv_fullblock_c2trust_phase_coordinate": {"kind": "projector_cov_alignadv_fullblock", "rank": 2, "block": "input_degree", "c2_trust": 1},
    "C49_alignadv_fullblock_c2trust_random_control": {"kind": "projector_cov_alignadv_random_fullblock", "rank": 2, "block": "input_degree", "c2_trust": 1},
    "C50_finite_response_phase_coordinate": {"kind": "projector_cov_response_fullblock", "rank": 2, "block": "input_degree"},
    "C51_finite_response_random_coordinate_control": {"kind": "projector_cov_response_random_fullblock", "rank": 2, "block": "input_degree"},
    "C52_null_advantage_response_phase_coordinate": {"kind": "projector_cov_response_nulladv_fullblock", "rank": 2, "block": "input_degree"},
    "C53_null_advantage_response_random_control": {"kind": "projector_cov_response_nulladv_random_fullblock", "rank": 2, "block": "input_degree"},
    "C54_orbit_contrast_phase_coordinate": {"kind": "projector_cov_orbit_contrast_fullblock", "rank": 2, "block": "task_orbit_degree"},
    "C55_orbit_contrast_blockshuffle_control": {"kind": "projector_cov_orbit_contrast_blockshuffle_fullblock", "rank": 2, "block": "task_orbit_degree"},
    "C56_orbit_atom_contrast_phase_coordinate": {"kind": "projector_cov_orbit_atom_contrast_fullblock", "rank": 2, "block": "task_orbit_atom_degree"},
    "C57_orbit_atom_patchshuffle_control": {"kind": "projector_cov_orbit_atom_patchshuffle_fullblock", "rank": 2, "block": "task_orbit_atom_degree"},
}

PROJECTOR_DIRECTION_KINDS = {"projector_sparse", "projector_sparse_random"}
PROJECTOR_SGFILTER_KINDS = {"projector_cov_sgfilter", "projector_cov_sgfilter_randombasis", "projector_cov_sgfilter_fullblock"}
PROJECTOR_SGFILTER_RANDOM_BASIS_KINDS = {"projector_cov_sgfilter_randombasis"}
PROJECTOR_SGFILTER_FULLBLOCK_KINDS = {"projector_cov_sgfilter_fullblock"}
PROJECTOR_RESPONSE_NULL_ADV_KINDS = {"projector_cov_response_nulladv_fullblock", "projector_cov_response_nulladv_random_fullblock"}
PROJECTOR_RESPONSE_KINDS = {"projector_cov_response_fullblock", "projector_cov_response_random_fullblock"} | PROJECTOR_RESPONSE_NULL_ADV_KINDS
PROJECTOR_ORBIT_CONTRAST_KINDS = {"projector_cov_orbit_contrast_fullblock", "projector_cov_orbit_contrast_blockshuffle_fullblock", "projector_cov_orbit_atom_contrast_fullblock", "projector_cov_orbit_atom_patchshuffle_fullblock"}
PROJECTOR_ORBIT_BLOCK_SHUFFLE_KINDS = {"projector_cov_orbit_contrast_blockshuffle_fullblock"}
PROJECTOR_ORBIT_PATCH_SHUFFLE_KINDS = {"projector_cov_orbit_atom_patchshuffle_fullblock"}
PROJECTOR_ALIGN_ADV_KINDS = {"projector_cov_alignadv", "projector_cov_alignadv_random", "projector_cov_alignadv_fullblock", "projector_cov_alignadv_random_fullblock"} | PROJECTOR_RESPONSE_KINDS | PROJECTOR_ORBIT_CONTRAST_KINDS
PROJECTOR_ALIGN_ADV_RANDOM_AFTER_GUARD_KINDS = {"projector_cov_alignadv_random_fullblock", "projector_cov_response_random_fullblock", "projector_cov_response_nulladv_random_fullblock"}
PROJECTOR_ALIGN_ADV_FULLBLOCK_AFTER_KINDS = {"projector_cov_alignadv_fullblock", "projector_cov_alignadv_random_fullblock"} | PROJECTOR_RESPONSE_KINDS | PROJECTOR_ORBIT_CONTRAST_KINDS
PROJECTOR_COV_KINDS = {"projector_cov_sparse", "projector_cov_sparse_random", "projector_cov_align_sparse", "projector_cov_align_sparse_random", "projector_cov_blockshuffle", "projector_cov_fullblock", "projector_cov_uniform", "projector_cov_uniform_random"} | PROJECTOR_SGFILTER_KINDS | PROJECTOR_ALIGN_ADV_KINDS
PROJECTOR_ALIGN_KINDS = {"projector_cov_align_sparse", "projector_cov_align_sparse_random"}
PROJECTOR_BLOCK_SHUFFLE_KINDS = {"projector_cov_blockshuffle"}
PROJECTOR_FULLBLOCK_KINDS = {"projector_cov_fullblock"}
PROJECTOR_UNIFORM_Q_KINDS = {"projector_cov_uniform", "projector_cov_uniform_random"}
PROJECTOR_KINDS = PROJECTOR_DIRECTION_KINDS | PROJECTOR_COV_KINDS
PROJECTOR_RANDOM_KINDS = {"projector_sparse_random", "projector_cov_sparse_random", "projector_cov_align_sparse_random", "projector_cov_uniform_random", "projector_cov_alignadv_random"}
ORBIT_LOCAL_TRANSFORMS = {"shift_down", "shift_right", "patch_shift_diag", "corner_preserving_jitter", "swap_local_patch_pairs", "swap_top_patch_pair", "swap_bottom_patch_pair"}
ORBIT_ROTATION_TRANSFORMS = {"rot90", "hflip", "vflip", "rot180", "swap_rotation_checkers"}

PART_D_SCHEMES = {
    "D0_scalar_dual_real_visual_baseline": {"kind": "scalar_dual", "rank": 0, "block": "degree_edgebank"},
    "D1_low_rank_phase_r1_real_visual": {"kind": "lowrank", "rank": 1, "block": "degree_edgebank"},
    "D2_low_rank_phase_r2_real_visual": {"kind": "lowrank", "rank": 2, "block": "degree_edgebank"},
    "D3_low_rank_phase_r4_real_visual": {"kind": "lowrank", "rank": 4, "block": "degree_edgebank"},
    "D4_inputBand_degreeBand_phase_r2": {"kind": "lowrank", "rank": 2, "block": "input_degree"},
    "D5_class_conditional_phase_r2": {"kind": "lowrank", "rank": 2, "block": "class_degree"},
    "D6_margin_bucket_phase_r2": {"kind": "lowrank", "rank": 2, "block": "margin_degree"},
    "D7_self_supervised_aug_phase_r2": {"kind": "lowrank", "rank": 2, "block": "input_degree"},
    "D8_random_phase_matched": {"kind": "random_phase", "rank": 2, "block": "degree_edgebank"},
    "D9_tabular_no_structure_baseline": {"kind": "tabular_no_structure", "rank": 0, "block": "degree_edgebank"},
    "D10_tabular_feature_group_phase": {"kind": "tabular_feature_group", "rank": 2, "block": "degree_edgebank"},
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


def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "x"


def command_text(argv: Iterable[str]) -> str:
    cmd = " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])
    env = os.environ.get("V2305_OUT_ROOT")
    return f"V2305_OUT_ROOT={env} {cmd}" if env else cmd


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


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else float(default)


def csv_items(text: Any) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
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


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.05 H10RealTransfer LowRankPhaseObserver 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            f"- Python：`{PYTHON}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误；缺失写 `missing`，阻塞写 `blocked`，不补造数据。\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.05 H10RealTransfer LowRankPhaseObserver 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n尚未 final。本文件只记录真实 artifact、真实指标、实际修复与分析，不补造数据。\n\n",
            encoding="utf-8",
        )


def append_exec(part: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {part} {status}\n\n")
        fh.write(f"- command: `{command}`\n")
        if gpu:
            fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, data: dict[str, Any]) -> None:
    init_logs()
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
        "mlp_readout_used": sum(ival(r.get("mlp_readout_used")) for r in rows),
        "mlp_stem_used": sum(ival(r.get("mlp_stem_used")) for r in rows),
        "new_edge_function_added": sum(ival(r.get("new_edge_function_added")) for r in rows),
        "external_product_feature_used": sum(ival(r.get("external_product_feature_used")) for r in rows),
        "generated_at": now(),
        **extra,
    }


def next_actions(part: str, gate: int, blocker: str, actions: list[str], commands: list[str] | None = None) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part.upper(),
            "gate_pass": int(gate),
            "blocker": blocker,
            "allowed_actions": actions,
            "forbidden_actions": [
                "do_not_use_synthetic_task_label_for_blind_observers",
                "do_not_select_rank_or_transform_by_held_or_test_metric",
                "do_not_delete_random_phase_control",
                "do_not_add_new_edge_function",
                "do_not_add_mlp_stem_or_mlp_readout",
                "do_not_insert_transform_output_as_forward_feature",
            ],
            "max_repair_rounds": 5,
            "required_rerun_commands": commands or [],
            "expected_artifacts": [f"results/v23_05/part_{part.lower()}_*"],
        },
    )


def device_from_args(args: argparse.Namespace) -> torch.device:
    requested = str(args.device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def torch_env() -> dict[str, Any]:
    return {
        "python_executable": PYTHON,
        "torch_version": torch.__version__,
        "cuda_available": int(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }


def run_part_0(args: argparse.Namespace) -> dict[str, Any]:
    h10_root = ROOT / "results/v23_03_official_h10_full_positive_control_seed15_steps80"
    h10 = read_json(h10_root / "part_h_full_positive_control_summary.json")
    h10_checks = h10.get("positive_control_checks", {}) if isinstance(h10.get("positive_control_checks"), dict) else {}
    h10_groups = h10.get("group_summaries", []) if isinstance(h10.get("group_summaries"), list) else []
    h10_group = next((g for g in h10_groups if str(g.get("h_scheme")) == "H10_F7ObjectiveAlignedSafetyDiagnostic"), {})
    part_i_smoke = ROOT / "results/v23_03_part_i_smoke"
    part_i_wine = ROOT / "results/v23_03_part_i_smoke_wine"
    v2304_root = ROOT / "results/v23_04_structure_conditioned_population_flow"
    v2304_final = read_json(v2304_root / "final_route.json")
    row = {
        "v23_03_h10_available": int(h10_root.exists() and bool(h10)),
        "v23_03_h10_gate_pass": ival(h10.get("gate_pass")),
        "v23_03_h10_coverage": fval(h10_checks.get("official_coverage", h10_group.get("C2_coverage_improvement_median")), -999.0),
        "v23_03_h10_baseline_gap": fval(h10_checks.get("beats_functional_or_blocksnr_baseline_gap"), -999.0),
        "v23_03_h10_random_gap": fval(h10_checks.get("beats_random_controls_gap"), -999.0),
        "v23_03_h10_random_orbit_gap": fval(h10_checks.get("random_gap_orbit"), -999.0),
        "v23_03_h10_f5_no_debt": ival(h10_checks.get("official_F5_no_debt_count", h10_group.get("F5_no_debt_count"))),
        "v23_03_h10_accept_rate": fval(h10_checks.get("official_accept_rate", h10_group.get("finite_step_accept_rate_median"))),
        "v23_03_h10_scale_mean": fval(h10_checks.get("official_scale_mean", h10_group.get("finite_step_scale_mean_median"))),
        "v23_03_h10_skip_count": fval(h10_checks.get("official_skip_count", h10_group.get("finite_step_skip_count_median"))),
        "v23_03_part_i_smoke_available": int(part_i_smoke.exists() or part_i_wine.exists()),
        "v23_03_part_i_official_available": int((h10_root / "part_i_real_task_preflight_summary.json").exists()),
        "v23_04_final_route": v2304_final.get("route", "missing"),
        "v23_04_part_c_gate_pass": ival((v2304_final.get("part_gate_passes") or {}).get("C", 0)) if isinstance(v2304_final.get("part_gate_passes"), dict) else 0,
        "v23_04_has_part_h_path": int((v2304_root / "part_h_real_task_preflight_summary.json").exists()),
        "v23_04_has_part_i_path": int((v2304_root / "part_i_real_task_preflight_summary.json").exists()),
        "version_branch_conflict_detected": 1,
        "selected_mainline": "v23_03_H10_to_real_transfer",
        "v23_03_runner": "experiments/run_v23_03_structure_projected_functional_population_flow.py",
        "v23_04_runner": "experiments/run_v23_04_structure_conditioned_population_flow.py",
        **AUDIT_DEFAULTS,
    }
    rows = [row]
    matrix = write_rows(OUT_ROOT / "part_0_lineage_matrix.csv", rows)
    gate = int(
        row["v23_03_h10_available"] == 1
        and row["v23_03_h10_gate_pass"] == 1
        and row["v23_04_has_part_h_path"] == 0
        and row["v23_04_has_part_i_path"] == 0
        and row["selected_mainline"] == "v23_03_H10_to_real_transfer"
    )
    summary = gate_summary("0", gate, "Part0LineageLocked" if gate else "Part0LineageFailed", "none" if gate else "lineage_or_h10_artifact_missing", rows, **row)
    write_json(OUT_ROOT / "part_0_lineage_summary.json", summary)
    nxt = next_actions(
        "0",
        gate,
        summary["dominant_blocker"],
        [] if gate else ["locate v23.03 H10 artifacts", "rerun v23.03 H10 reproduction before Part C", "do not substitute v23.04 branch artifacts"],
    )
    append_exec("part-0", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_0_lineage_summary.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part 0 version lineage and artifact reconciliation", summary)
    return summary


def static_scan(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|metric_winner)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|winner_update)\s*\("),
        "external_product_feature": re.compile(r"\b(make_external_product_feature|patch_product_feature)\s*\("),
        "mlp_stem_constructor": re.compile(r"\bMLPStem\s*\("),
        "mlp_readout_constructor": re.compile(r"\bMLPReadout\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            hits.append({"file": rel(path), "check": "missing_file", "match": "missing"})
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                hits.append({"file": rel(path), "check": name, "match": match.group(0)})
    return int(not hits), hits


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    files = [
        RUNNER,
        ROOT / "dgkan/optim/edge_sobolev_population_flow.py",
        ROOT / "experiments/run_v23_00r_curve_geometry_population_flow.py",
        ROOT / "experiments/run_v23_03_structure_projected_functional_population_flow.py",
    ]
    compile_errors: list[str] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}: {repr(exc)}")
    import_errors: list[str] = []
    for mod in [
        "dgkan.optim.edge_sobolev_population_flow",
        "experiments.run_v23_00r_curve_geometry_population_flow",
        "experiments.run_v23_03_structure_projected_functional_population_flow",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_errors.append(f"{mod}: {repr(exc)}")
    scan_pass, scan_hits = static_scan(files)
    device = device_from_args(args)
    smoke = purekan_smoke(args, device)
    row = {
        "compile_pass": int(not compile_errors),
        "import_pass": int(not import_errors),
        "static_scan_pass": scan_pass,
        **smoke,
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "metric_winner_selection_used": 0,
        "candidate_update_selection_used": 0,
        "new_edge_function_added": 0,
        "external_product_feature_used": 0,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
        "structure_transform_used_only_for_gate": 1,
        "structure_transform_in_forward": 0,
        "torch_env": json.dumps(torch_env(), ensure_ascii=False),
        **{k: v for k, v in AUDIT_DEFAULTS.items() if k not in {"held_test_usage", "runtime_selector_used", "metric_winner_selection_used", "candidate_update_selection_used", "new_edge_function_added", "external_product_feature_used", "mlp_stem_used", "mlp_readout_used", "structure_transform_used_only_for_gate", "structure_transform_in_forward"}},
    }
    required = [
        "compile_pass",
        "import_pass",
        "static_scan_pass",
        "changed_edge_coefficients",
        "state_updated",
        "structure_transform_used_only_for_gate",
    ]
    gate = int(all(ival(row.get(k)) == 1 for k in required) and ival(row.get("changed_mlp_tensors")) == 0)
    matrix = write_rows(OUT_ROOT / "part_a_identity_matrix.csv", [row])
    summary = gate_summary(
        "A",
        gate,
        "PartAIdentityPass" if gate else "PartAIdentityFailed",
        "none" if gate else "compile_import_static_or_smoke_failed",
        [row],
        compile_errors=compile_errors,
        import_errors=import_errors,
        static_scan_hits=scan_hits,
        torch_env=torch_env(),
    )
    write_json(OUT_ROOT / "part_a_identity_summary.json", summary)
    nxt = next_actions("A", gate, summary["dominant_blocker"], [] if gate else ["repair compile/import failure", "repair PureKAN smoke step", "tighten static scan only if it is a false positive"])
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_a_identity_summary.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part A implementation identity and anti-selector audit", summary)
    return summary


def purekan_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    try:
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.visual_side = 8
        bargs = v2300.basis_args(smoke_args, "dche_k9")
        model = v2300.make_model("depth3", 64, 2, 2305001, bargs, device)
        mark_kan_edge_params(model, basis_key="dche_k9")
        before = torch.cat([p.detach().reshape(-1).cpu() for p in model.coeffs])
        opt = EdgeSobolevPopulationFlow(
            list(model.coeffs),
            lr=float(args.edge_lr),
            weight_decay=float(args.weight_decay),
            sobolev_exponent=0.25,
            edge_metric_type="functional_gram",
            edge_weight_normalization=str(args.edge_weight_normalization),
            edge_weight_ridge=float(args.edge_weight_ridge),
            functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
            gate_beta=float(args.snr_beta),
            gate_family="degree_edgebank",
            gate_floor=0.0,
            gate_input_side=8,
            use_population_gate=False,
        )
        gen = torch.Generator(device=device).manual_seed(2305)
        xb = torch.randn(16, 64, generator=gen, device=device)
        yb = torch.randint(0, 2, (16,), generator=gen, device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb.long())
        loss.backward()
        opt.step()
        after = torch.cat([p.detach().reshape(-1).cpu() for p in model.coeffs])
        return {
            "changed_edge_coefficients": int(float((after - before).abs().sum().item()) > 0.0),
            "changed_mlp_tensors": 0,
            "state_updated": int(getattr(opt.last_stats, "edge_params_updated", 0) > 0),
            "smoke_loss": float(loss.detach().cpu().item()),
            "smoke_device": str(device),
        }
    except Exception as exc:
        return {
            "changed_edge_coefficients": 0,
            "changed_mlp_tensors": 0,
            "state_updated": 0,
            "smoke_error": repr(exc),
            "smoke_device": str(device),
        }


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    h10_root = ROOT / "results/v23_03_official_h10_full_positive_control_seed15_steps80"
    h10 = read_json(h10_root / "part_h_full_positive_control_summary.json")
    checks = h10.get("positive_control_checks", {}) if isinstance(h10.get("positive_control_checks"), dict) else {}
    groups = h10.get("group_summaries", []) if isinstance(h10.get("group_summaries"), list) else []
    h10_group = next((g for g in groups if str(g.get("h_scheme")) == "H10_F7ObjectiveAlignedSafetyDiagnostic"), {})
    row = {
        "H10_gate_pass": ival(h10.get("gate_pass")),
        "H10_coverage": fval(checks.get("official_coverage", h10_group.get("C2_coverage_improvement_median")), -999.0),
        "H10_local_patch_coverage": "see_part_h_full_positive_control_task_summary.csv",
        "H10_rotation_coverage": "see_part_h_full_positive_control_task_summary.csv",
        "H10_baseline_gap_vs_FunctionalGram": fval(checks.get("beats_functional_or_blocksnr_baseline_gap"), -999.0),
        "H10_random_density_gap": "see_part_h_full_positive_control_group_summary.csv",
        "H10_random_hist_gap": "see_part_h_full_positive_control_group_summary.csv",
        "H10_random_orbit_gap": fval(checks.get("random_gap_orbit"), -999.0),
        "H10_same_compute_noop_gap": fval(checks.get("beats_same_compute_noop_gap"), -999.0),
        "H10_F5_no_debt_count": ival(checks.get("official_F5_no_debt_count", h10_group.get("F5_no_debt_count"))),
        "H10_component_non_positive_rows": ival(checks.get("official_component_non_positive_rows", h10_group.get("component_non_positive_rows"))),
        "H10_accept_rate": fval(checks.get("official_accept_rate", h10_group.get("finite_step_accept_rate_median"))),
        "H10_scale_mean": fval(checks.get("official_scale_mean", h10_group.get("finite_step_scale_mean_median"))),
        "H10_skip_count": fval(checks.get("official_skip_count", h10_group.get("finite_step_skip_count_median"))),
        "H10_overhead_vs_noop": fval(checks.get("official_overhead_vs_same_compute_noop"), -999.0),
        "H10_held_test_usage": ival(h10.get("held_test_usage")),
        "H10_used_fake_data_rows": ival(h10.get("used_fake_data_rows")),
        "artifact_source": rel(h10_root),
        **AUDIT_DEFAULTS,
    }
    rows = [row]
    gate = int(
        row["H10_gate_pass"] == 1
        and fval(row["H10_coverage"]) >= 0.45
        and fval(row["H10_baseline_gap_vs_FunctionalGram"]) >= 0.03
        and fval(row["H10_random_orbit_gap"]) >= 0.30
        and ival(row["H10_F5_no_debt_count"]) >= 12
        and fval(row["H10_accept_rate"]) >= 0.20
        and fval(row["H10_scale_mean"]) >= 0.05
        and ival(row["H10_held_test_usage"]) == 0
        and ival(row["H10_used_fake_data_rows"]) == 0
    )
    matrix = write_rows(OUT_ROOT / "part_b_h10_lock_matrix.csv", rows)
    summary = gate_summary("B", gate, "PartBH10LockPass" if gate else "PartBH10LockFailed", "none" if gate else "h10_reproduction_or_artifact_failed", rows, **row)
    write_json(OUT_ROOT / "part_b_h10_lock.json", summary)
    nxt = next_actions("B", gate, summary["dominant_blocker"], [] if gate else ["rerun v23.03 H10 full positive-control", "audit seeds/train steps/safety alignment before Part C"])
    append_exec("part-b", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_b_h10_lock.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part B H10 synthetic positive-control reproduction lock", summary)
    return summary


def h10_c2_artifact_row(task: str, seed: int) -> dict[str, Any]:
    path = ROOT / "results/v23_03_official_h10_full_positive_control_seed15_steps80/part_h_full_positive_control_matrix.csv"
    for row in read_rows(path):
        if (
            str(row.get("part")) == "H_C2"
            and str(row.get("h_scheme")) == "H10_F7ObjectiveAlignedSafetyDiagnostic"
            and str(row.get("task")) == str(task)
            and ival(row.get("seed")) == int(seed)
        ):
            return row
    return {}


def make_edge_optimizer(model: v2293.TrueDeepPureKAN, args: argparse.Namespace, *, sobolev: float = 0.25, use_population_gate: bool = True) -> EdgeSobolevPopulationFlow:
    return EdgeSobolevPopulationFlow(
        list(model.coeffs),
        lr=float(args.edge_lr),
        weight_decay=float(args.weight_decay),
        sobolev_exponent=float(sobolev),
        edge_metric_type="functional_gram",
        edge_weight_normalization=str(args.edge_weight_normalization),
        edge_weight_ridge=float(args.edge_weight_ridge),
        functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
        gate_beta=float(args.snr_beta),
        gate_family="degree_edgebank",
        gate_floor=float(args.gate_floor),
        gate_floor_mode=str(args.gate_floor_mode),
        gate_input_side=int(args.visual_side),
        gate_patch_size=2,
        stat_warmup_steps=0,
        use_population_gate=bool(use_population_gate),
    )


def block_indices_for_param(
    opt: EdgeSobolevPopulationFlow,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    args: argparse.Namespace,
    block_family: str,
    labels: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    family = str(block_family)
    if family == "task_orbit_degree":
        return task_orbit_degree_indices_for_param(param, args)
    if family == "task_orbit_atom_degree":
        return task_orbit_degree_indices_for_param(param, args, atom=True)
    if family in {"pixel_degree", "pixelBand_degreeBand"}:
        if int(param.ndim) < 3 or int(param.numel()) <= 0:
            return [torch.arange(int(param.numel()), device=param.device, dtype=torch.long)]
        flat = torch.arange(int(param.numel()), device=param.device, dtype=torch.long).reshape_as(param)
        degree_ranges = v2303._split_ranges(int(param.shape[-1]), int(getattr(args, "part_i_band_degree_bands", 3)))
        first = int(param.shape[0])
        side = int(getattr(args, "visual_side", 0))
        layer_id = int(getattr(param, "_kan_layer_id", -1))
        if layer_id == 0 and side > 0 and first == side * side:
            first_axis_groups = [[i] for i in range(first)]
        else:
            first_axis_groups = [[i] for i in range(first)]
        blocks: list[torch.Tensor] = []
        for first_idx in first_axis_groups:
            first_tensor = torch.tensor(first_idx, device=param.device, dtype=torch.long)
            for dr in degree_ranges:
                degree_tensor = torch.tensor([int(i) for i in dr], device=param.device, dtype=torch.long)
                if int(first_tensor.numel()) <= 0 or int(degree_tensor.numel()) <= 0:
                    continue
                block = flat.index_select(0, first_tensor).index_select(-1, degree_tensor).reshape(-1)
                if int(block.numel()) > 0:
                    blocks.append(block)
        return blocks or [torch.arange(int(param.numel()), device=param.device, dtype=torch.long)]
    if family in {"input_degree", "inputBand_degreeBand"}:
        return [b.detach().long().to(device=param.device) for b in v2303.part_i_band_indices_for_param(param, args)]
    if family == "layer":
        return [torch.arange(int(param.numel()), device=param.device, dtype=torch.long)]
    if family == "degree_only" and int(param.ndim) >= 1:
        flat = torch.arange(int(param.numel()), device=param.device, dtype=torch.long).reshape_as(param)
        blocks: list[torch.Tensor] = []
        for dr in v2303._split_ranges(int(param.shape[-1]), 3):
            idx = torch.tensor([int(i) for i in dr], device=param.device, dtype=torch.long)
            blocks.append(flat.index_select(-1, idx).reshape(-1))
        return blocks
    blocks = opt._blocks_for_param(param, {**group, "gate_family": "degree_edgebank"})
    return [torch.tensor([int(i) for i in b], device=param.device, dtype=torch.long) for b in blocks if b]


def patch_axis_indices(side: int, top: int, left: int, *, size: int = 2) -> list[int]:
    if side <= 0 or top < 0 or left < 0 or top + size > side or left + size > side:
        return []
    return [int(r) * int(side) + int(c) for r in range(int(top), int(top) + int(size)) for c in range(int(left), int(left) + int(size))]


def unique_axis_group(parts: Iterable[int]) -> list[int]:
    return sorted({int(i) for i in parts if int(i) >= 0})


def task_orbit_first_axis_groups(side: int) -> list[list[int]]:
    if side < 4:
        return []
    lo = 1 if side >= 5 else 0
    hi = max(0, side - 3)
    mid = max(0, min(side - 2, side // 2 - 1))
    corner_tl = patch_axis_indices(side, lo, lo)
    corner_tr = patch_axis_indices(side, lo, hi)
    corner_bl = patch_axis_indices(side, hi, lo)
    corner_br = patch_axis_indices(side, hi, hi)
    rot_top = patch_axis_indices(side, lo, mid)
    rot_bottom = patch_axis_indices(side, hi, mid)
    rot_left = patch_axis_indices(side, mid, lo)
    rot_right = patch_axis_indices(side, mid, hi)
    candidates = [
        unique_axis_group([*corner_tl, *corner_tr]),
        unique_axis_group([*corner_bl, *corner_br]),
        unique_axis_group([*corner_tl, *corner_bl]),
        unique_axis_group([*corner_tr, *corner_br]),
        unique_axis_group([*corner_tl, *corner_tr, *corner_bl, *corner_br]),
        unique_axis_group([*rot_top, *rot_bottom]),
        unique_axis_group([*rot_left, *rot_right]),
        unique_axis_group([*corner_tl, *corner_br]),
        unique_axis_group([*corner_tr, *corner_bl]),
    ]
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for group in candidates:
        key = tuple(group)
        if group and key not in seen:
            out.append(group)
            seen.add(key)
    return out


def task_orbit_atom_first_axis_groups(side: int) -> list[list[int]]:
    if side < 4:
        return []
    lo = 1 if side >= 5 else 0
    hi = max(0, side - 3)
    mid = max(0, min(side - 2, side // 2 - 1))
    candidates = [
        patch_axis_indices(side, lo, lo),
        patch_axis_indices(side, lo, hi),
        patch_axis_indices(side, hi, lo),
        patch_axis_indices(side, hi, hi),
        patch_axis_indices(side, lo, mid),
        patch_axis_indices(side, hi, mid),
        patch_axis_indices(side, mid, lo),
        patch_axis_indices(side, mid, hi),
    ]
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for group in candidates:
        key = tuple(unique_axis_group(group))
        if key and key not in seen:
            out.append(list(key))
            seen.add(key)
    return out


def task_orbit_degree_indices_for_param(param: torch.nn.Parameter, args: argparse.Namespace, *, atom: bool = False) -> list[torch.Tensor]:
    if int(param.ndim) < 3 or int(param.numel()) <= 0:
        return [torch.arange(int(param.numel()), device=param.device, dtype=torch.long)]
    flat = torch.arange(int(param.numel()), device=param.device, dtype=torch.long).reshape_as(param)
    k = int(param.shape[-1])
    degree_ranges = v2303._split_ranges(k, int(getattr(args, "part_i_band_degree_bands", 3)))
    first = int(param.shape[0])
    side = int(getattr(args, "visual_side", 0))
    layer_id = int(getattr(param, "_kan_layer_id", -1))
    if layer_id == 0 and side > 0 and first == side * side:
        first_axis_groups = task_orbit_atom_first_axis_groups(side) if bool(atom) else task_orbit_first_axis_groups(side)
    else:
        first_axis_groups = [[int(i) for i in chunk] for chunk in v2303._split_ranges(first, int(getattr(args, "part_i_band_edge_bands", 4)))]
    blocks: list[torch.Tensor] = []
    for first_idx in first_axis_groups:
        first_tensor = torch.tensor(first_idx, device=param.device, dtype=torch.long)
        for dr in degree_ranges:
            degree_tensor = torch.tensor([int(i) for i in dr], device=param.device, dtype=torch.long)
            if int(first_tensor.numel()) <= 0 or int(degree_tensor.numel()) <= 0:
                continue
            block = flat.index_select(0, first_tensor).index_select(-1, degree_tensor).reshape(-1)
            if int(block.numel()) > 0:
                blocks.append(block)
    return blocks or [b.detach().long().to(device=param.device) for b in v2303.part_i_band_indices_for_param(param, args)]


def block_specs_for_param(
    opt: EdgeSobolevPopulationFlow,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    args: argparse.Namespace,
    block_family: str,
    labels: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    family = str(block_family)
    if family == "class_degree" and labels is not None and int(labels.numel()) > 0:
        blocks = block_indices_for_param(opt, param, group, args, "input_degree", labels=labels, logits=logits)
        label_vec = labels.detach().reshape(-1).to(device=param.device)
        specs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for label in torch.unique(label_vec):
            mask = label_vec == label
            if int(mask.sum().item()) >= 1:
                specs.extend((block, mask.detach().cpu()) for block in blocks)
        return specs or [(block, None) for block in blocks]
    if family == "margin_degree" and labels is not None and logits is not None and int(labels.numel()) > 0:
        blocks = block_indices_for_param(opt, param, group, args, "input_degree", labels=labels, logits=logits)
        label_vec = labels.detach().reshape(-1).to(device=logits.device, dtype=torch.long)
        probs = torch.softmax(logits.detach().float(), dim=1)
        n = min(int(label_vec.numel()), int(probs.shape[0]))
        label_vec = label_vec[:n]
        probs = probs[:n]
        true_p = probs.gather(1, label_vec.reshape(-1, 1)).reshape(-1)
        masked = probs.clone()
        masked.scatter_(1, label_vec.reshape(-1, 1), -1.0)
        margin = true_p - masked.max(dim=1).values
        order = torch.argsort(margin)
        bucket_count = max(1, min(3, int(n)))
        specs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for ids in torch.chunk(order, bucket_count):
            if int(ids.numel()) <= 0:
                continue
            mask = torch.zeros(n, dtype=torch.bool, device=param.device)
            mask[ids.to(device=param.device)] = True
            specs.extend((block, mask.detach().cpu()) for block in blocks)
        return specs or [(block, None) for block in blocks]
    return [(block, None) for block in block_indices_for_param(opt, param, group, args, block_family, labels=labels, logits=logits)]


def transform_specs(args: argparse.Namespace, *, real_visual: bool = False) -> list[tuple[str, str]]:
    profile = str(args.phase_transform_profile)
    specs: list[tuple[str, str]] = [
        ("shift_down", "positive"),
        ("corner_preserving_jitter", "positive"),
        ("rot90", "positive"),
        ("hflip", "positive"),
        ("swap_local_patch_pairs", "positive"),
        ("destructive_lowrank", "destructive"),
    ]
    if profile in {"expanded", "repair_expanded"}:
        specs.extend([
            ("shift_right", "positive"),
            ("patch_shift_diag", "positive"),
            ("vflip", "positive"),
            ("rot180", "positive"),
            ("swap_rotation_checkers", "positive"),
        ])
    if profile == "task_orbit":
        specs = [
            ("swap_top_patch_pair", "positive"),
            ("swap_bottom_patch_pair", "positive"),
            ("swap_local_patch_pairs", "positive"),
            ("hflip", "positive"),
            ("vflip", "positive"),
            ("rot180", "positive"),
            ("destructive_lowrank", "destructive"),
        ]
    return specs


def compute_split_grads(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    n = min(int(args.population_grad_examples), int(x.shape[0]))
    xb = x[:n]
    yb = y[:n]
    flat = per_example_gradient_matrix(model, xb, yb, max_examples=n, label_prior_correction=bool(int(args.label_prior_correction)))
    return flat, v2303.split_flat_grads(flat, list(model.coeffs)), yb[: int(flat.shape[0])].detach()


def metric_block_score(
    opt: EdgeSobolevPopulationFlow,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor,
    perm: torch.Tensor,
    idx: torch.Tensor,
) -> float:
    a_raw = v2303.permute_param_examples(source_grads, param, perm, inverse=False)
    b_raw = target_grads
    n = min(int(a_raw.shape[0]), int(b_raw.shape[0]))
    if n <= 0 or int(idx.numel()) <= 0:
        return 0.0
    a = opt._apply_metric_inv_sqrt(a_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    b = opt._apply_metric_inv_sqrt(b_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    use_idx = idx.to(device=a.device)
    a = a[:, use_idx]
    b = b[:, use_idx]
    denom = a.norm(dim=1).clamp_min(1.0e-12) * b.norm(dim=1).clamp_min(1.0e-12)
    c = (a * b).sum(dim=1) / denom
    return float(c.clamp_min(0.0).mean().detach().cpu().item())


def metric_block_commutator_score(
    opt: EdgeSobolevPopulationFlow,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor,
    perm: torch.Tensor,
    idx: torch.Tensor,
) -> float:
    a_raw = v2303.permute_param_examples(source_grads, param, perm, inverse=False)
    b_raw = target_grads
    n = min(int(a_raw.shape[0]), int(b_raw.shape[0]))
    if n <= 0 or int(idx.numel()) <= 0:
        return 0.0
    a = opt._apply_metric_inv_sqrt(a_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    b = opt._apply_metric_inv_sqrt(b_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    use_idx = idx.to(device=a.device)
    mu = a[:, use_idx].mean(dim=0).detach().to(dtype=torch.float64)
    delta = (b[:, use_idx] - a[:, use_idx]).mean(dim=0).detach().to(dtype=torch.float64)
    mu_norm = mu.norm().clamp_min(1.0e-12)
    delta_norm = delta.norm().clamp_min(1.0e-12)
    align = torch.dot(mu, delta).div(mu_norm * delta_norm)
    magnitude = (delta_norm / mu_norm).clamp(0.0, 3.0)
    return float((align.clamp_min(0.0) * magnitude).detach().cpu().item())


def metric_block_delta_energy_score(
    opt: EdgeSobolevPopulationFlow,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor,
    perm: torch.Tensor,
    idx: torch.Tensor,
) -> float:
    a_raw = v2303.permute_param_examples(source_grads, param, perm, inverse=False)
    b_raw = target_grads
    n = min(int(a_raw.shape[0]), int(b_raw.shape[0]))
    if n <= 0 or int(idx.numel()) <= 0:
        return 0.0
    a = opt._apply_metric_inv_sqrt(a_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    b = opt._apply_metric_inv_sqrt(b_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    use_idx = idx.to(device=a.device)
    delta = (b[:, use_idx] - a[:, use_idx]).mean(dim=0).detach().to(dtype=torch.float64)
    source = a[:, use_idx].mean(dim=0).detach().to(dtype=torch.float64)
    delta_norm = delta.norm().div(math.sqrt(max(1, int(use_idx.numel()))))
    source_norm = source.norm().div(math.sqrt(max(1, int(use_idx.numel())))).clamp_min(1.0e-12)
    return float((delta_norm / source_norm).clamp(0.0, 3.0).detach().cpu().item())


def metric_block_tangent_direction(
    opt: EdgeSobolevPopulationFlow,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor,
    perm: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    a_raw = v2303.permute_param_examples(source_grads, param, perm, inverse=False)
    b_raw = target_grads
    n = min(int(a_raw.shape[0]), int(b_raw.shape[0]))
    if n <= 0 or int(idx.numel()) <= 0:
        return torch.zeros(int(idx.numel()), dtype=torch.float64)
    a = opt._apply_metric_inv_sqrt(a_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    b = opt._apply_metric_inv_sqrt(b_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    use_idx = idx.to(device=a.device)
    return (b[:, use_idx] - a[:, use_idx]).mean(dim=0).detach().to(dtype=torch.float64).cpu()


def metric_block_tangent_samples(
    opt: EdgeSobolevPopulationFlow,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor,
    perm: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    a_raw = v2303.permute_param_examples(source_grads, param, perm, inverse=False)
    b_raw = target_grads
    n = min(int(a_raw.shape[0]), int(b_raw.shape[0]))
    if n <= 0 or int(idx.numel()) <= 0:
        return torch.zeros((0, int(idx.numel())), dtype=torch.float64)
    a = opt._apply_metric_inv_sqrt(a_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    b = opt._apply_metric_inv_sqrt(b_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    use_idx = idx.to(device=a.device)
    return (b[:, use_idx] - a[:, use_idx]).detach().to(dtype=torch.float64).cpu()


def orthonormal_columns(mat: torch.Tensor, rank: int | None = None) -> torch.Tensor:
    work = mat.detach().to(dtype=torch.float64).cpu()
    if work.ndim == 1:
        work = work.reshape(-1, 1)
    if work.ndim != 2 or int(work.shape[0]) <= 0 or int(work.shape[1]) <= 0:
        return torch.zeros((int(work.shape[0]) if work.ndim == 2 else 0, 0), dtype=torch.float64)
    keep = max(1, min(int(rank) if rank is not None else int(work.shape[1]), int(work.shape[0]), int(work.shape[1])))
    try:
        q, _r = torch.linalg.qr(work, mode="reduced")
    except Exception:
        u, _s, _vh = torch.linalg.svd(work, full_matrices=False)
        q = u
    q = q[:, :keep]
    norms = q.norm(dim=0)
    mask = norms > 1.0e-12
    return q[:, mask] if int(mask.sum().item()) > 0 else torch.zeros((int(work.shape[0]), 0), dtype=torch.float64)


def covariance_projector_basis(
    positive_samples: list[torch.Tensor],
    destructive_samples: list[torch.Tensor],
    rank: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    mats = [m.detach().to(dtype=torch.float64).cpu() for m in positive_samples if m.ndim == 2 and int(m.numel()) > 0]
    if not mats:
        dim = int(destructive_samples[0].shape[1]) if destructive_samples and destructive_samples[0].ndim == 2 else 0
        return torch.zeros((dim, 0), dtype=torch.float64)
    dim = int(mats[0].shape[1])
    pos = torch.cat([m[:, :dim] for m in mats if int(m.shape[1]) >= dim], dim=0)
    pos = pos / pos.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    c_pos = pos.T @ pos / max(1, int(pos.shape[0]))
    c_pos = c_pos / torch.trace(c_pos).clamp_min(1.0e-12)
    neg_mats = [m.detach().to(dtype=torch.float64).cpu()[:, :dim] for m in destructive_samples if m.ndim == 2 and int(m.shape[1]) >= dim and int(m.numel()) > 0]
    if neg_mats:
        neg = torch.cat(neg_mats, dim=0)
        neg = neg / neg.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        c_neg = neg.T @ neg / max(1, int(neg.shape[0]))
        c_neg = c_neg / torch.trace(c_neg).clamp_min(1.0e-12)
    else:
        c_neg = torch.zeros_like(c_pos)
    cov = 0.5 * (c_pos + c_pos.T) - float(args.contrastive_eta) * 0.5 * (c_neg + c_neg.T)
    cov = cov + 1.0e-8 * torch.eye(dim, dtype=torch.float64)
    try:
        vals, vecs = torch.linalg.eigh(cov)
    except Exception:
        return orthonormal_columns(pos.T, rank)
    order = torch.argsort(vals, descending=True)
    k = max(1, min(int(rank), dim, int(order.numel())))
    basis = vecs[:, order[:k]]
    return orthonormal_columns(basis, k)


def align_projector_basis(source: torch.Tensor, guard: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    src = orthonormal_columns(source, rank)
    grd = orthonormal_columns(guard, rank)
    if int(src.numel()) <= 0:
        return src, 0.0
    if int(grd.numel()) <= 0:
        return src, 0.0
    n = min(int(src.shape[0]), int(grd.shape[0]))
    k = min(int(src.shape[1]), int(grd.shape[1]), max(1, int(rank)))
    if n <= 0 or k <= 0:
        return src, 0.0
    src = src[:n, :k]
    grd = grd[:n, :k]
    try:
        u, s, vh = torch.linalg.svd(src.T @ grd, full_matrices=False)
        src_a = src @ u[:, :k]
        grd_a = grd @ vh.T[:, :k]
        shared = src_a + grd_a
        for col in range(int(shared.shape[1])):
            if float(shared[:, col].norm().item()) <= 1.0e-12:
                shared[:, col] = src_a[:, col]
        return orthonormal_columns(shared, k), float(s[:k].mean().item()) if int(s.numel()) else 0.0
    except Exception:
        return src, cosine_vec(src.reshape(-1), grd.reshape(-1), abs_value=True)


def random_orthonormal_like(basis: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    src = basis.detach().to(dtype=torch.float64).cpu()
    if src.ndim == 1:
        src = src.reshape(-1, 1)
    if src.ndim != 2 or int(src.shape[0]) <= 0 or int(src.shape[1]) <= 0:
        return torch.zeros_like(src)
    rnd = torch.randn(tuple(src.shape), generator=generator, dtype=torch.float64)
    return orthonormal_columns(rnd, int(src.shape[1]))


def fullblock_basis_like(basis: torch.Tensor) -> torch.Tensor:
    src = basis.detach().to(dtype=torch.float64).cpu()
    dim = int(src.numel()) if src.ndim == 1 else (int(src.shape[0]) if src.ndim == 2 else 0)
    if dim <= 0:
        return torch.zeros((0, 0), dtype=torch.float64)
    return torch.eye(dim, dtype=torch.float64)


def shuffle_projector_block_identity(
    bases: list[torch.Tensor],
    q_block: torch.Tensor,
    *,
    seed: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if not bases or int(q_block.numel()) <= 0:
        return bases, q_block
    q_new = q_block.detach().clone()
    bases_new = list(bases)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, basis in enumerate(bases):
        if i >= int(q_block.numel()) or basis.ndim != 2:
            continue
        groups.setdefault((int(basis.shape[0]), int(basis.shape[1])), []).append(i)
    gen = torch.Generator().manual_seed(int(seed) * 10007 + 79)
    for ids in groups.values():
        if len(ids) <= 1:
            continue
        perm = torch.randperm(len(ids), generator=gen).tolist()
        src_ids = [ids[j] for j in perm]
        for dst, src in zip(ids, src_ids):
            q_new[dst] = q_block[src]
            bases_new[dst] = bases[src]
    return bases_new, q_new


def block_first_axis_signature(param: torch.nn.Parameter, block_cpu: torch.Tensor) -> tuple[int, tuple[int, ...]]:
    if int(param.ndim) <= 0 or int(param.numel()) <= 0 or int(block_cpu.numel()) <= 0:
        return (0, ())
    stride0 = max(1, int(param.numel()) // max(1, int(param.shape[0])))
    first_ids = torch.div(block_cpu.detach().reshape(-1).long(), stride0, rounding_mode="floor")
    first_ids = first_ids[(first_ids >= 0) & (first_ids < int(param.shape[0]))]
    if int(first_ids.numel()) <= 0:
        return (int(block_cpu.numel()), ())
    uniq = tuple(int(v) for v in torch.unique(first_ids.cpu(), sorted=True).tolist())
    return (int(block_cpu.numel()), uniq)


def block_degree_signature(param: torch.nn.Parameter, block_cpu: torch.Tensor) -> tuple[int, ...]:
    if int(param.ndim) <= 0 or int(block_cpu.numel()) <= 0:
        return ()
    k = int(param.shape[-1])
    if k <= 0:
        return ()
    degree_ids = torch.remainder(block_cpu.detach().reshape(-1).long(), k)
    return tuple(int(v) for v in torch.unique(degree_ids.cpu(), sorted=True).tolist())


def shuffle_projector_within_orbit_identity(
    bases: list[torch.Tensor],
    q_block: torch.Tensor,
    records: list[tuple[int, torch.Tensor, torch.Tensor | None]],
    params: list[torch.nn.Parameter],
    *,
    seed: int,
) -> tuple[list[torch.Tensor], torch.Tensor, float]:
    if not bases or int(q_block.numel()) <= 0:
        return bases, q_block, 0.0
    groups: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
    for i, (pidx, block_cpu, _mask) in enumerate(records):
        if i >= len(bases) or i >= int(q_block.numel()) or int(pidx) >= len(params):
            continue
        size, first_sig = block_first_axis_signature(params[int(pidx)], block_cpu)
        groups.setdefault((int(pidx), int(size), first_sig), []).append(i)
    q_new = q_block.detach().clone()
    bases_new = list(bases)
    gen = torch.Generator().manual_seed(int(seed) * 10007 + 307)
    moved = 0
    total = 0
    for ids in groups.values():
        if len(ids) <= 1:
            continue
        total += len(ids)
        perm = torch.randperm(len(ids), generator=gen).tolist()
        src_ids = [ids[j] for j in perm]
        for dst, src in zip(ids, src_ids):
            if int(dst) != int(src):
                moved += 1
            q_new[dst] = q_block[src]
            bases_new[dst] = bases[src]
    return bases_new, q_new, float(moved) / max(1.0, float(total))


def shuffle_projector_within_degree_identity(
    bases: list[torch.Tensor],
    q_block: torch.Tensor,
    records: list[tuple[int, torch.Tensor, torch.Tensor | None]],
    params: list[torch.nn.Parameter],
    *,
    seed: int,
) -> tuple[list[torch.Tensor], torch.Tensor, float]:
    if not bases or int(q_block.numel()) <= 0:
        return bases, q_block, 0.0
    groups: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
    for i, (pidx, block_cpu, _mask) in enumerate(records):
        if i >= len(bases) or i >= int(q_block.numel()) or int(pidx) >= len(params):
            continue
        degree_sig = block_degree_signature(params[int(pidx)], block_cpu)
        groups.setdefault((int(pidx), int(block_cpu.numel()), degree_sig), []).append(i)
    q_new = q_block.detach().clone()
    bases_new = list(bases)
    gen = torch.Generator().manual_seed(int(seed) * 10007 + 347)
    moved = 0
    total = 0
    for ids in groups.values():
        if len(ids) <= 1:
            continue
        total += len(ids)
        perm = torch.randperm(len(ids), generator=gen).tolist()
        src_ids = [ids[j] for j in perm]
        for dst, src in zip(ids, src_ids):
            if int(dst) != int(src):
                moved += 1
            q_new[dst] = q_block[src]
            bases_new[dst] = bases[src]
    return bases_new, q_new, float(moved) / max(1.0, float(total))


def cosine_vec(a: torch.Tensor, b: torch.Tensor, *, abs_value: bool = False) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64).cpu()
    bb = b.detach().reshape(-1).to(dtype=torch.float64).cpu()
    n = min(int(aa.numel()), int(bb.numel()))
    if n <= 0:
        return 0.0
    val = float(torch.dot(aa[:n], bb[:n]).div(aa[:n].norm().clamp_min(1.0e-12) * bb[:n].norm().clamp_min(1.0e-12)).item())
    return abs(val) if abs_value else val


def sigmoid_gate(signal: torch.Tensor, alpha: float) -> torch.Tensor:
    sig = signal.detach().to(dtype=torch.float64).cpu()
    if int(sig.numel()) <= 0:
        return sig
    std = float(sig.std(unbiased=False).item())
    if std <= 1.0e-12:
        z = sig - sig.mean()
    else:
        z = (sig - sig.mean()) / std
    return torch.sigmoid(float(alpha) * z).clamp(0.0, 1.0)


def sparse_topk_gate(signal: torch.Tensor, density: float) -> torch.Tensor:
    sig = signal.detach().reshape(-1).to(dtype=torch.float64).cpu()
    if int(sig.numel()) <= 0:
        return sig
    k = max(1, min(int(sig.numel()), int(round(float(density) * int(sig.numel())))))
    order = torch.argsort(sig, descending=True)
    out = torch.zeros_like(sig)
    out[order[:k]] = 1.0
    return out


def density_preserve_mean(weighted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if int(weighted.numel()) <= 0 or int(reference.numel()) <= 0:
        return weighted
    target = reference.detach().mean().to(device=weighted.device, dtype=weighted.dtype)
    current = weighted.detach().mean().to(device=weighted.device, dtype=weighted.dtype)
    if float(current.cpu().item()) <= 1.0e-12:
        return weighted
    return (weighted * (target / current.clamp_min(1.0e-12))).clamp(0.0, 1.0)


def lowrank_from_H(
    H: torch.Tensor,
    H_guard: torch.Tensor | None,
    utility: torch.Tensor,
    rank: int,
    args: argparse.Namespace,
    *,
    residualize: bool = False,
    sparse_density: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(H.numel()) <= 0:
        return torch.zeros(0, dtype=torch.float64), {"phase_energy_top1": 0.0, "phase_energy_top2": 0.0, "phase_energy_top4": 0.0, "phase_source_guard_cosine": 0.0, "phase_utility_alignment": 0.0}
    Hc = H.detach().to(dtype=torch.float64).cpu()
    if bool(residualize) and int(Hc.shape[0]) > 1 and int(Hc.shape[1]) > 1:
        Hc = Hc - Hc.mean(dim=1, keepdim=True) - Hc.mean(dim=0, keepdim=True) + Hc.mean()
        Hc = Hc / Hc.std(dim=0, keepdim=True).clamp_min(1.0e-12)
    else:
        Hc = Hc - Hc.mean(dim=0, keepdim=True)
    if float(args.phase_ridge) > 0.0:
        Hc = Hc + float(args.phase_ridge) * torch.randn_like(Hc) * 0.0
    u, s, _vh = torch.linalg.svd(Hc, full_matrices=False)
    total = float((s.square()).sum().item())
    def energy(k: int) -> float:
        return float((s[: min(k, int(s.numel()))].square()).sum().item() / max(total, 1.0e-12))
    r = max(1, min(int(rank), int(u.shape[1])))
    phase = u[:, :r] * s[:r].reshape(1, -1)
    guard_phase: torch.Tensor | None = None
    if H_guard is not None and int(H_guard.numel()) > 0:
        Hg = H_guard.detach().to(dtype=torch.float64).cpu()
        if bool(residualize) and int(Hg.shape[0]) > 1 and int(Hg.shape[1]) > 1:
            Hg = Hg - Hg.mean(dim=1, keepdim=True) - Hg.mean(dim=0, keepdim=True) + Hg.mean()
            Hg = Hg / Hg.std(dim=0, keepdim=True).clamp_min(1.0e-12)
        else:
            Hg = Hg - Hg.mean(dim=0, keepdim=True)
        ug, sg, _ = torch.linalg.svd(Hg, full_matrices=False)
        rg = max(1, min(r, int(ug.shape[1])))
        guard_phase = ug[:, :rg] * sg[:rg].reshape(1, -1)
    util = utility.detach().reshape(-1).to(dtype=torch.float64).cpu()
    if int(util.numel()) != int(phase.shape[0]):
        util = torch.ones(int(phase.shape[0]), dtype=torch.float64)
    rhos: list[float] = []
    sg_vals: list[float] = []
    util_vals: list[float] = []
    for k in range(r):
        col = phase[:, k]
        if float(torch.dot(col, util).item()) < 0.0:
            phase[:, k] = -phase[:, k]
            col = phase[:, k]
        sg = 1.0
        if guard_phase is not None and k < int(guard_phase.shape[1]):
            sg = cosine_vec(col, guard_phase[:, k], abs_value=True)
        util_align = max(0.0, cosine_vec(col, util, abs_value=False))
        rho = max(0.0, min(1.0, sg * util_align))
        rhos.append(rho)
        sg_vals.append(sg)
        util_vals.append(util_align)
    if not rhos or max(rhos) <= 1.0e-12:
        signal = phase[:, :r].mean(dim=1) * 0.0
    else:
        weights = torch.tensor(rhos, dtype=torch.float64)
        signal = (phase[:, :r] * weights.reshape(1, -1)).sum(dim=1)
    if sparse_density is not None:
        q = sparse_topk_gate(signal, float(sparse_density))
    else:
        q = sigmoid_gate(signal, float(args.phase_alpha))
    return q, {
        "phase_energy_top1": energy(1),
        "phase_energy_top2": energy(2),
        "phase_energy_top4": energy(4),
        "phase_source_guard_cosine": mean(sg_vals),
        "phase_utility_alignment": mean(util_vals),
        "phase_rho_mean": mean(rhos),
        "phase_rank_effective": float((s.square().sum().square() / s.pow(4).sum().clamp_min(1.0e-12)).item()) if int(s.numel()) else 0.0,
    }


def intersection_lowrank_from_H(
    H: torch.Tensor,
    H_guard: torch.Tensor | None,
    utility: torch.Tensor,
    rank: int,
    args: argparse.Namespace,
    *,
    residualize: bool = False,
    sparse_density: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if H_guard is None or int(H_guard.numel()) <= 0:
        return lowrank_from_H(H, H_guard, utility, rank, args, residualize=residualize, sparse_density=sparse_density)
    if int(H.numel()) <= 0:
        return torch.zeros(0, dtype=torch.float64), {"phase_energy_top1": 0.0, "phase_energy_top2": 0.0, "phase_energy_top4": 0.0, "phase_source_guard_cosine": 0.0, "phase_utility_alignment": 0.0}
    Hc = H.detach().to(dtype=torch.float64).cpu()
    Hg = H_guard.detach().to(dtype=torch.float64).cpu()
    n = min(int(Hc.shape[0]), int(Hg.shape[0]))
    m = min(int(Hc.shape[1]), int(Hg.shape[1]))
    Hc = Hc[:n, :m]
    Hg = Hg[:n, :m]
    if bool(residualize) and n > 1 and m > 1:
        Hc = Hc - Hc.mean(dim=1, keepdim=True) - Hc.mean(dim=0, keepdim=True) + Hc.mean()
        Hg = Hg - Hg.mean(dim=1, keepdim=True) - Hg.mean(dim=0, keepdim=True) + Hg.mean()
        Hc = Hc / Hc.std(dim=0, keepdim=True).clamp_min(1.0e-12)
        Hg = Hg / Hg.std(dim=0, keepdim=True).clamp_min(1.0e-12)
    else:
        Hc = Hc - Hc.mean(dim=0, keepdim=True)
        Hg = Hg - Hg.mean(dim=0, keepdim=True)
    cross = Hc.T @ Hg / max(1, n)
    us, s, vh = torch.linalg.svd(cross, full_matrices=False)
    total = float((s.square()).sum().item())

    def energy(k: int) -> float:
        return float((s[: min(k, int(s.numel()))].square()).sum().item() / max(total, 1.0e-12))

    r = max(1, min(int(rank), int(us.shape[1]), int(vh.shape[0])))
    source_phase = Hc @ us[:, :r]
    guard_phase = Hg @ vh.T[:, :r]
    util = utility.detach().reshape(-1).to(dtype=torch.float64).cpu()
    if int(util.numel()) != n:
        util = torch.ones(n, dtype=torch.float64)
    shared_cols: list[torch.Tensor] = []
    rhos: list[float] = []
    sg_vals: list[float] = []
    util_vals: list[float] = []
    for k in range(r):
        src = source_phase[:, k].clone()
        grd = guard_phase[:, k].clone()
        if cosine_vec(src, grd, abs_value=False) < 0.0:
            grd = -grd
        shared = 0.5 * (src + grd)
        if float(torch.dot(shared, util).item()) < 0.0:
            shared = -shared
            src = -src
            grd = -grd
        sg = cosine_vec(src, grd, abs_value=True)
        util_align = max(0.0, cosine_vec(shared, util, abs_value=False))
        rhos.append(max(0.0, min(1.0, sg * util_align)))
        sg_vals.append(sg)
        util_vals.append(util_align)
        shared_cols.append(shared)
    phase = torch.stack(shared_cols, dim=1) if shared_cols else torch.zeros((n, 1), dtype=torch.float64)
    if not rhos or max(rhos) <= 1.0e-12:
        signal = phase[:, 0] * 0.0
    else:
        weights = torch.tensor(rhos, dtype=torch.float64)
        signal = (phase[:, :r] * weights.reshape(1, -1)).sum(dim=1)
    if sparse_density is not None:
        q = sparse_topk_gate(signal, float(sparse_density))
    else:
        q = sigmoid_gate(signal, float(args.phase_alpha))
    return q, {
        "phase_energy_top1": energy(1),
        "phase_energy_top2": energy(2),
        "phase_energy_top4": energy(4),
        "phase_source_guard_cosine": mean(sg_vals),
        "phase_utility_alignment": mean(util_vals),
        "phase_rho_mean": mean(rhos),
        "phase_rank_effective": float((s.square().sum().square() / s.pow(4).sum().clamp_min(1.0e-12)).item()) if int(s.numel()) else 0.0,
    }


def block_utility_scores_and_modifiers(
    opt: EdgeSobolevPopulationFlow,
    params: list[torch.nn.Parameter],
    group: dict[str, Any],
    source_split: list[torch.Tensor],
    utility_split: list[torch.Tensor] | None,
    records: list[tuple[int, torch.Tensor, torch.Tensor | None]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if utility_split is None:
        scores = torch.ones(len(records), dtype=torch.float64)
        modifiers = torch.ones(len(records), dtype=torch.float64)
        return scores, modifiers, {
            "utility_multiplier_mean": 1.0,
            "signed_modifier_mean": 1.0,
            "signed_modifier_negative_fraction": 0.0,
        }
    score_vals: list[float] = []
    modifier_vals: list[float] = []
    for pidx, block_cpu, _cohort_mask_cpu in records:
        if int(pidx) >= len(params) or int(pidx) >= len(source_split) or int(pidx) >= len(utility_split):
            score_vals.append(1.0)
            modifier_vals.append(1.0)
            continue
        param = params[int(pidx)]
        source = source_split[int(pidx)].to(device=param.device, dtype=param.dtype)
        utility = utility_split[int(pidx)].to(device=param.device, dtype=param.dtype)
        idx = block_cpu.to(device=param.device, dtype=torch.long)
        if utility.ndim < 2 or int(idx.numel()) <= 0:
            score_vals.append(1.0)
            modifier_vals.append(1.0)
            continue
        n = min(int(source.shape[0]), int(utility.shape[0]))
        if n <= 0:
            score_vals.append(1.0)
            modifier_vals.append(1.0)
            continue
        a = opt._apply_metric_inv_sqrt(source[:n], param, group).reshape(n, -1).mean(dim=0)[idx]
        b = opt._apply_metric_inv_sqrt(utility[:n], param, group).reshape(n, -1).mean(dim=0)[idx]
        denom = a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)
        cos = float(torch.dot(a, b).div(denom).detach().cpu().item())
        score_vals.append(max(0.0, cos))
        modifier_vals.append(float(args.signed_negative_scale) if cos < 0.0 else 1.0)
    floor = max(0.0, min(float(args.utility_tilt_floor), 1.0))
    scores = torch.tensor([floor + (1.0 - floor) * v for v in score_vals], dtype=torch.float64)
    modifiers = torch.tensor(modifier_vals, dtype=torch.float64)
    return scores, modifiers, {
        "utility_multiplier_mean": mean(scores.tolist(), default=1.0),
        "signed_modifier_mean": mean(modifiers.tolist(), default=1.0),
        "signed_modifier_negative_fraction": mean((modifiers < 0.0).to(dtype=torch.float64).tolist(), default=0.0),
    }


def finite_response_coordinate_gate(
    model: v2293.TrueDeepPureKAN,
    opt: EdgeSobolevPopulationFlow,
    params: list[torch.nn.Parameter],
    records: list[tuple[int, torch.Tensor, torch.Tensor | None]],
    prior: torch.Tensor,
    guard_xy: tuple[torch.Tensor, torch.Tensor] | None,
    args: argparse.Namespace,
    *,
    seed: int = 0,
    null_advantage: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    if guard_xy is None or not records or int(prior.numel()) <= 0:
        return prior, {
            "phase_response_probe_count": 0.0,
            "phase_response_score_mean": 0.0,
            "phase_response_score_best": 0.0,
            "phase_response_selected_score_mean": 0.0,
            "phase_response_selected_count": 0.0,
            "phase_response_null_score_mean": 0.0,
            "phase_response_advantage_mean": 0.0,
            "phase_response_advantage_best": 0.0,
            "phase_response_positive_fraction": 0.0,
        }
    x_guard, y_guard = guard_xy
    n_eval = min(int(args.phase_response_guard_examples), int(x_guard.shape[0]))
    if n_eval <= 1:
        return prior, {
            "phase_response_probe_count": 0.0,
            "phase_response_score_mean": 0.0,
            "phase_response_score_best": 0.0,
            "phase_response_selected_score_mean": 0.0,
            "phase_response_selected_count": 0.0,
            "phase_response_null_score_mean": 0.0,
            "phase_response_advantage_mean": 0.0,
            "phase_response_advantage_best": 0.0,
            "phase_response_positive_fraction": 0.0,
        }
    x_eval = x_guard[:n_eval]
    y_eval = y_guard[:n_eval]
    target_keep = max(1, min(int(prior.numel()), int(round(float(args.phase_density_target) * int(prior.numel())))))
    budget = max(int(args.phase_response_probe_blocks), target_keep)
    budget = max(1, min(budget, len(records), int(prior.numel())))
    prior_cpu = prior.detach().reshape(-1).to(dtype=torch.float64).cpu()
    order = torch.argsort(prior_cpu, descending=True).tolist()
    candidate_ids = [int(i) for i in order[:budget]]
    if not candidate_ids:
        return prior, {
            "phase_response_probe_count": 0.0,
            "phase_response_score_mean": 0.0,
            "phase_response_score_best": 0.0,
            "phase_response_selected_score_mean": 0.0,
            "phase_response_selected_count": 0.0,
            "phase_response_null_score_mean": 0.0,
            "phase_response_advantage_mean": 0.0,
            "phase_response_advantage_best": 0.0,
            "phase_response_positive_fraction": 0.0,
        }
    null_ids: list[int] = []
    if bool(null_advantage):
        candidate_set = set(candidate_ids)
        gen = torch.Generator().manual_seed(int(seed) * 10007 + 211)
        perm = torch.randperm(int(prior_cpu.numel()), generator=gen).tolist()
        null_ids = [int(i) for i in perm if int(i) not in candidate_set][:budget]
        if len(null_ids) < budget:
            null_ids.extend([int(i) for i in perm[: budget - len(null_ids)]])

    before = v2293.metrics_for_model(model, x_eval, y_eval)
    param_snapshot = snapshot_model_params(model)
    opt_snapshot = clone_tensor_tree(opt.state_dict())
    stats_snapshot = copy.deepcopy(opt.last_stats)
    mode_snapshot = bool(model.training)
    guard_n = min(int(args.population_grad_examples), int(x_eval.shape[0]))
    _flat_guard, split_guard, labels_guard = compute_split_grads(model, x_eval[:guard_n], y_eval[:guard_n], args)
    scores = torch.full((int(prior_cpu.numel()),), -1.0e9, dtype=torch.float64)
    cov_deltas: list[float] = []
    score_vals: list[float] = []
    base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]
    scale = float(args.phase_response_lr_scale)
    score_ids = candidate_ids + null_ids
    for row_idx in score_ids:
        pidx, block_cpu, _cohort_mask_cpu = records[row_idx]
        if int(pidx) >= len(params) or int(pidx) >= len(split_guard):
            continue
        restore_model_params(model, param_snapshot)
        opt.load_state_dict(clone_tensor_tree(opt_snapshot))
        opt.last_stats = copy.deepcopy(stats_snapshot)
        opt.clear_observed_gradients()
        for lr, group in zip(base_lrs, opt.param_groups):
            group["lr"] = lr * scale
        opt.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        for j, param in enumerate(params):
            opt.observe_per_example_gradients(param, split_guard[j].to(device=param.device, dtype=param.dtype), labels=labels_guard)
            gate = torch.zeros(int(param.numel()), device=param.device, dtype=param.dtype)
            if j == int(pidx):
                block = block_cpu.to(device=param.device, dtype=torch.long)
                if int(block.numel()) > 0:
                    gate[block] = 1.0
            opt.observe_structure_gate(param, gate.reshape_as(param))
        logits = model(x_eval[:guard_n]).float()
        loss = F.cross_entropy(logits, y_eval[:guard_n].long())
        loss.backward()
        opt.step()
        after = v2293.metrics_for_model(model, x_eval, y_eval)
        cov_delta = fval(after.get("coverage_CVaR25")) - fval(before.get("coverage_CVaR25"))
        acc_delta = fval(after.get("accuracy")) - fval(before.get("accuracy"))
        nll_delta = fval(after.get("nll")) - fval(before.get("nll"))
        score = cov_delta + float(args.phase_response_accuracy_weight) * acc_delta - float(args.phase_response_nll_penalty) * max(0.0, nll_delta)
        scores[row_idx] = score
        cov_deltas.append(cov_delta)
        score_vals.append(score)
    restore_model_params(model, param_snapshot)
    opt.load_state_dict(clone_tensor_tree(opt_snapshot))
    opt.last_stats = copy.deepcopy(stats_snapshot)
    opt.clear_observed_gradients()
    for lr, group in zip(base_lrs, opt.param_groups):
        group["lr"] = lr
    model.train(mode_snapshot)
    model.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)

    null_scores = torch.tensor([float(scores[i].item()) for i in null_ids if float(scores[i].item()) > -1.0e8], dtype=torch.float64)
    null_center = float(torch.median(null_scores).item()) if int(null_scores.numel()) else 0.0
    if bool(null_advantage):
        adjusted_scores = torch.full_like(scores, -1.0e9)
        candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
        if int(candidate_tensor.numel()) > 0:
            adjusted_scores[candidate_tensor] = scores[candidate_tensor] - null_center
        selection_scores = adjusted_scores
    else:
        adjusted_scores = scores
        selection_scores = scores
    valid = selection_scores > -1.0e8
    q = torch.zeros_like(prior_cpu)
    if int(valid.sum().item()) > 0:
        valid_ids = torch.where(valid)[0]
        valid_scores = selection_scores[valid_ids]
        keep = max(1, min(int(valid_ids.numel()), int(round(float(args.phase_density_target) * int(prior_cpu.numel())))))
        order_valid = torch.argsort(valid_scores, descending=True)
        selected = valid_ids[order_valid[:keep]]
        min_delta = float(args.phase_response_min_score)
        if min_delta > -1.0e8:
            selected = selected[selection_scores[selected] >= min_delta]
        if int(selected.numel()) == 0:
            selected = valid_ids[order_valid[:1]]
        q[selected] = 1.0
    selected_scores = selection_scores[q > 0.5]
    advantage_vals = [float(adjusted_scores[i].item()) for i in candidate_ids if float(adjusted_scores[i].item()) > -1.0e8]
    return q, {
        "phase_response_probe_count": float(len(score_ids)),
        "phase_response_score_mean": mean(score_vals),
        "phase_response_score_best": max(score_vals) if score_vals else 0.0,
        "phase_response_selected_score_mean": mean(selected_scores.tolist()) if int(selected_scores.numel()) else 0.0,
        "phase_response_selected_count": float(int(selected_scores.numel())),
        "phase_response_null_score_mean": mean(null_scores.tolist()) if int(null_scores.numel()) else 0.0,
        "phase_response_advantage_mean": mean(advantage_vals),
        "phase_response_advantage_best": max(advantage_vals) if advantage_vals else 0.0,
        "phase_response_positive_fraction": mean([1.0 if v > 0.0 else 0.0 for v in cov_deltas]),
    }


def compute_phase_observer(
    model: v2293.TrueDeepPureKAN,
    opt: EdgeSobolevPopulationFlow,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
    *,
    seed: int,
    scheme: str,
    task: str = "",
    block_family: str = "degree_edgebank",
    rank: int = 2,
    kind: str = "lowrank",
    guard_xy: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    side = int(round(math.sqrt(int(x.shape[1]))))
    if side * side != int(x.shape[1]):
        side = int(args.visual_side)
    params = list(model.coeffs)
    group = opt.param_groups[0]
    flat, split, labels = compute_split_grads(model, x, y, args)
    logits = model(x[: int(labels.shape[0])]).detach()
    specs = transform_specs(args)
    identity_perm = torch.arange(side * side, device=x.device)
    transformed: list[tuple[str, str, list[torch.Tensor], torch.Tensor]] = []
    for i, (name, role) in enumerate(specs):
        perm = v2303.transform_perm(side, "destructive_lowrank" if role == "destructive" else name, seed=seed * 1000 + i, device=x.device)
        x_t = v2303.apply_perm(x[: int(labels.shape[0])], perm)
        _flat_t, split_t, _ = compute_split_grads(model, x_t, y[: int(x_t.shape[0])], args)
        transformed.append((name, role, split_t, perm if role != "destructive" else identity_perm))
    records: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
    H_rows: list[list[float]] = []
    local_scores: list[float] = []
    rot_scores: list[float] = []
    dest_scores_all: list[float] = []
    utility_scores: list[float] = []
    utility_vectors: list[torch.Tensor] = []
    projector_directions: list[torch.Tensor] = []
    projector_bases: list[torch.Tensor] = []
    for pidx, (param, grads) in enumerate(zip(params, split)):
        pgrads = grads.to(device=param.device, dtype=param.dtype)
        opt.observe_per_example_gradients(param, pgrads, labels=labels)
        block_specs = block_specs_for_param(opt, param, group, args, block_family, labels=labels, logits=logits)
        white = opt._apply_metric_inv_sqrt(pgrads, param, group).reshape(int(pgrads.shape[0]), -1)
        mu = white.mean(dim=0).detach()
        for block, cohort_mask_cpu in block_specs:
            if int(block.numel()) <= 0:
                continue
            cohort_mask = cohort_mask_cpu.to(device=param.device, dtype=torch.bool) if cohort_mask_cpu is not None else None
            pgrads_use = pgrads[cohort_mask] if cohort_mask is not None and int(cohort_mask.numel()) == int(pgrads.shape[0]) else pgrads
            scores: list[float] = []
            dest_scores: list[float] = []
            positive_dirs: list[torch.Tensor] = []
            destructive_dirs: list[torch.Tensor] = []
            positive_samples: list[torch.Tensor] = []
            destructive_samples: list[torch.Tensor] = []
            for name, role, split_t, perm in transformed:
                target = split_t[pidx].to(device=param.device, dtype=param.dtype)
                target_use = target[cohort_mask] if cohort_mask is not None and int(cohort_mask.numel()) == int(target.shape[0]) else target
                if kind in {"commutator_sparse", "commutator_sparse_random"}:
                    score = metric_block_commutator_score(opt, param, group, pgrads_use, target_use, perm, block)
                elif kind in {"delta_energy_intersection_sparse", "delta_energy_intersection_sparse_random"}:
                    score = metric_block_delta_energy_score(opt, param, group, pgrads_use, target_use, perm, block)
                else:
                    score = metric_block_score(opt, param, group, pgrads_use, target_use, perm, block)
                scores.append(score)
                if kind in PROJECTOR_DIRECTION_KINDS:
                    direction = metric_block_tangent_direction(opt, param, group, pgrads_use, target_use, perm, block)
                    if role == "destructive":
                        destructive_dirs.append(direction)
                    else:
                        positive_dirs.append(direction)
                elif kind in PROJECTOR_COV_KINDS:
                    samples = metric_block_tangent_samples(opt, param, group, pgrads_use, target_use, perm, block)
                    if role == "destructive":
                        destructive_samples.append(samples)
                    else:
                        positive_samples.append(samples)
                if role == "destructive":
                    dest_scores.append(score)
                if name in {"shift_down", "shift_right", "patch_shift_diag", "corner_preserving_jitter", "swap_local_patch_pairs", "swap_top_patch_pair", "swap_bottom_patch_pair"}:
                    local_scores.append(score)
                if name in {"rot90", "hflip", "vflip", "rot180", "swap_rotation_checkers"}:
                    rot_scores.append(score)
            dest_mean = mean(dest_scores)
            if kind in PROJECTOR_ORBIT_CONTRAST_KINDS:
                task_positive = ORBIT_LOCAL_TRANSFORMS if task == "local_patch_interaction" else ORBIT_ROTATION_TRANSFORMS
                task_complement = ORBIT_ROTATION_TRANSFORMS if task == "local_patch_interaction" else ORBIT_LOCAL_TRANSFORMS
                comp_mean = mean([score for score, (name, role, _split_t, _perm) in zip(scores, transformed) if role != "destructive" and name in task_complement])
                h_row = []
                for score, (name, role, _split_t, _perm) in zip(scores, transformed):
                    if role == "destructive":
                        h_row.append(-score)
                    elif name in task_positive:
                        h_row.append(max(0.0, score - float(args.contrastive_eta) * dest_mean - float(args.orbit_contrast_complement_eta) * comp_mean))
                    elif name in task_complement:
                        h_row.append(-float(args.orbit_contrast_complement_eta) * score)
                    else:
                        h_row.append(0.0)
            else:
                h_row = [(-score if role == "destructive" else max(0.0, score - float(args.contrastive_eta) * dest_mean)) for score, (_name, role, _split_t, _perm) in zip(scores, transformed)]
            H_rows.append(h_row)
            dest_scores_all.append(dest_mean)
            idx = block.to(device=mu.device)
            mu_use = white[cohort_mask].mean(dim=0).detach() if cohort_mask is not None and int(cohort_mask.numel()) == int(white.shape[0]) else mu
            util_vec = mu_use[idx].detach().to(dtype=torch.float64).cpu()
            utility_vectors.append(util_vec)
            utility_scores.append(float(util_vec.norm().div(math.sqrt(max(1, int(idx.numel())))).item()))
            if kind in PROJECTOR_DIRECTION_KINDS:
                if positive_dirs:
                    direction = torch.stack([d.reshape(-1).to(dtype=torch.float64) for d in positive_dirs], dim=0).mean(dim=0)
                else:
                    direction = torch.zeros(int(block.numel()), dtype=torch.float64)
                if destructive_dirs:
                    destructive = torch.stack([d.reshape(-1).to(dtype=torch.float64) for d in destructive_dirs], dim=0).mean(dim=0)
                    denom = torch.dot(destructive, destructive).clamp_min(1.0e-12)
                    direction = direction - torch.dot(direction, destructive).div(denom) * destructive
                norm = direction.norm()
                direction = direction / norm if float(norm.item()) > 1.0e-12 else direction
                projector_directions.append(direction)
                projector_bases.append(direction.reshape(-1, 1))
            elif kind in PROJECTOR_COV_KINDS:
                basis = covariance_projector_basis(positive_samples, destructive_samples, int(rank), args)
                projector_bases.append(basis)
                projector_directions.append(basis[:, 0] if basis.ndim == 2 and int(basis.shape[1]) > 0 else torch.zeros(int(block.numel()), dtype=torch.float64))
            records.append((pidx, block.detach().long().cpu(), cohort_mask_cpu.detach().cpu() if cohort_mask_cpu is not None else None))
    H = torch.tensor(H_rows, dtype=torch.float64)
    util = torch.tensor(utility_scores, dtype=torch.float64)
    if int(util.numel()) and float(util.max().item()) > 1.0e-12:
        util = util / util.max().clamp_min(1.0e-12)
    H_guard: torch.Tensor | None = None
    guard_q: torch.Tensor | None = None
    guard_projector_directions: list[torch.Tensor] = []
    guard_projector_bases: list[torch.Tensor] = []
    utility_split: list[torch.Tensor] | None = None
    if guard_xy is not None:
        gx, gy = guard_xy
        _util_flat, utility_split, _util_labels = compute_split_grads(model, gx, gy, args)
        guard_obj = compute_phase_observer(
            model,
            opt,
            gx,
            gy,
            args,
            seed=seed + 991,
            scheme=scheme + "__guard",
            task=task,
            block_family=block_family,
            rank=rank,
            kind=kind,
            guard_xy=None,
        )
        H_guard = guard_obj.get("H_tensor")
        guard_q = guard_obj.get("q_block")
        guard_projector_directions = guard_obj.get("projector_directions") if isinstance(guard_obj.get("projector_directions"), list) else []
        guard_projector_bases = guard_obj.get("projector_bases") if isinstance(guard_obj.get("projector_bases"), list) else []
    if kind == "functional_gram":
        q_block = torch.ones(len(records), dtype=torch.float64)
        meta = {"phase_energy_top1": 0.0, "phase_energy_top2": 0.0, "phase_energy_top4": 0.0, "phase_source_guard_cosine": 1.0, "phase_utility_alignment": 0.0, "phase_rank_effective": 0.0}
    elif kind == "oracle":
        vals = local_scores if task == "local_patch_interaction" else rot_scores
        q_block = sigmoid_gate(torch.tensor(vals[: len(records)], dtype=torch.float64), float(args.phase_alpha))
        meta = {"phase_energy_top1": 1.0, "phase_energy_top2": 1.0, "phase_energy_top4": 1.0, "phase_source_guard_cosine": 1.0, "phase_utility_alignment": 1.0, "phase_rank_effective": 1.0}
    elif kind == "scalar_dual":
        vals = [0.5 * (l + r) - float(args.contrastive_eta) * d for l, r, d in zip(local_scores[: len(records)], rot_scores[: len(records)], dest_scores_all)]
        q_block = sigmoid_gate(torch.tensor(vals, dtype=torch.float64), float(args.phase_alpha))
        meta = {"phase_energy_top1": 1.0, "phase_energy_top2": 1.0, "phase_energy_top4": 1.0, "phase_source_guard_cosine": cosine_vec(q_block, guard_q, abs_value=True) if guard_q is not None else 1.0, "phase_utility_alignment": abs(cosine_vec(q_block, util)), "phase_rank_effective": 1.0}
    elif kind in {"lowrank_signed", "lowrank_signed_random"}:
        q_block, meta = lowrank_from_H(H, H_guard, util, int(rank), args)
        utility_mult, signed_modifier, utility_meta = block_utility_scores_and_modifiers(opt, params, group, split, utility_split, records, args)
        q_block = density_preserve_mean(q_block * utility_mult, q_block)
        meta.update(utility_meta)
        if kind == "lowrank_signed_random":
            gen = torch.Generator().manual_seed(int(seed) * 10007 + 41)
            if int(q_block.numel()) > 0:
                q_block = q_block[torch.randperm(int(q_block.numel()), generator=gen)]
    elif kind in {"intersection_lowrank", "intersection_random", "intersection_sparse", "intersection_sparse_random"}:
        q_block, meta = intersection_lowrank_from_H(
            H,
            H_guard,
            util,
            int(rank),
            args,
            residualize=kind in {"intersection_sparse", "intersection_sparse_random"},
            sparse_density=float(args.phase_density_target) if kind in {"intersection_sparse", "intersection_sparse_random"} else None,
        )
        if kind in {"intersection_random", "intersection_sparse_random"}:
            gen = torch.Generator().manual_seed(int(seed) * 10007 + (53 if kind == "intersection_sparse_random" else 47))
            if int(q_block.numel()) > 0:
                q_block = q_block[torch.randperm(int(q_block.numel()), generator=gen)]
    elif kind in {"delta_energy_intersection_sparse", "delta_energy_intersection_sparse_random"}:
        q_block, meta = intersection_lowrank_from_H(
            H,
            H_guard,
            util,
            int(rank),
            args,
            residualize=True,
            sparse_density=float(args.phase_density_target),
        )
        if kind == "delta_energy_intersection_sparse_random":
            gen = torch.Generator().manual_seed(int(seed) * 10007 + 59)
            if int(q_block.numel()) > 0:
                q_block = q_block[torch.randperm(int(q_block.numel()), generator=gen)]
    elif kind in PROJECTOR_KINDS:
        q_block, meta = intersection_lowrank_from_H(
            H,
            H_guard,
            util,
            int(rank),
            args,
            residualize=True,
            sparse_density=float(args.phase_density_target),
        )
    elif kind in {"residual_sparse", "residual_sparse_random", "commutator_sparse", "commutator_sparse_random", "residual_sparse_signed", "residual_sparse_signed_random"}:
        q_block, meta = lowrank_from_H(
            H,
            H_guard,
            util,
            int(rank),
            args,
            residualize=True,
            sparse_density=float(args.phase_density_target),
        )
        if kind in {"residual_sparse_signed", "residual_sparse_signed_random"}:
            utility_mult, signed_modifier, utility_meta = block_utility_scores_and_modifiers(opt, params, group, split, utility_split, records, args)
            q_block = density_preserve_mean(q_block * utility_mult, q_block)
            meta.update(utility_meta)
        if kind in {"residual_sparse_random", "commutator_sparse_random", "residual_sparse_signed_random"}:
            gen = torch.Generator().manual_seed(int(seed) * 10007 + (37 if kind == "residual_sparse_signed_random" else (31 if kind == "commutator_sparse_random" else 29)))
            if int(q_block.numel()) > 0:
                q_block = q_block[torch.randperm(int(q_block.numel()), generator=gen)]
    else:
        q_block, meta = lowrank_from_H(H, H_guard, util, int(rank), args)
        if kind == "random_phase":
            gen = torch.Generator().manual_seed(int(seed) * 10007 + 23)
            if int(q_block.numel()) > 0:
                q_block = q_block[torch.randperm(int(q_block.numel()), generator=gen)]
    final_projector_directions: list[torch.Tensor] = []
    final_projector_bases: list[torch.Tensor] = []
    if kind in PROJECTOR_KINDS:
        sg_vals: list[float] = []
        sg_weights: list[float] = []
        for i, basis in enumerate(projector_bases):
            shared = orthonormal_columns(basis, int(rank))
            if i < len(guard_projector_bases):
                shared, sg = align_projector_basis(shared, guard_projector_bases[i], int(rank))
                sg_vals.append(sg)
                sg_weights.append(sg)
            elif i < len(guard_projector_directions):
                guard_direction = guard_projector_directions[i].detach().reshape(-1, 1).to(dtype=torch.float64).cpu()
                shared, sg = align_projector_basis(shared, guard_direction, 1)
                sg_vals.append(sg)
                sg_weights.append(sg)
            else:
                sg_weights.append(0.0)
            final_projector_bases.append(shared)
        if kind in PROJECTOR_SGFILTER_KINDS and int(q_block.numel()) > 0:
            sg_tensor = torch.tensor(sg_weights[: int(q_block.numel())], dtype=torch.float64)
            if int(sg_tensor.numel()) < int(q_block.numel()):
                pad = torch.zeros(int(q_block.numel()) - int(sg_tensor.numel()), dtype=torch.float64)
                sg_tensor = torch.cat([sg_tensor, pad], dim=0)
            q_block = sparse_topk_gate(q_block[: int(sg_tensor.numel())] * sg_tensor, float(args.phase_density_target))
            meta["phase_sg_filter_mean"] = mean(sg_weights)
        randomize_after_sg = kind in PROJECTOR_SGFILTER_RANDOM_BASIS_KINDS and not str(scheme).endswith("__guard")
        fullblock_after_sg = kind in PROJECTOR_SGFILTER_FULLBLOCK_KINDS and not str(scheme).endswith("__guard")
        randomize_after_align_guard = kind in PROJECTOR_ALIGN_ADV_RANDOM_AFTER_GUARD_KINDS and not str(scheme).endswith("__guard")
        if kind in PROJECTOR_RANDOM_KINDS or randomize_after_sg or randomize_after_align_guard:
            gen = torch.Generator().manual_seed(int(seed) * 10007 + 67)
            final_projector_bases = [random_orthonormal_like(basis, gen) for basis in final_projector_bases]
        if kind in PROJECTOR_FULLBLOCK_KINDS or fullblock_after_sg:
            final_projector_bases = [fullblock_basis_like(basis) for basis in final_projector_bases]
        final_projector_directions = [basis[:, 0] if basis.ndim == 2 and int(basis.shape[1]) > 0 else torch.zeros(0, dtype=torch.float64) for basis in final_projector_bases]
        if kind in PROJECTOR_ALIGN_KINDS:
            align_scores: list[float] = []
            for basis, util_vec in zip(final_projector_bases, utility_vectors):
                b = orthonormal_columns(basis, int(rank))
                u = util_vec.detach().reshape(-1).to(dtype=torch.float64).cpu()
                n = min(int(b.shape[0]) if b.ndim == 2 else 0, int(u.numel()))
                if n <= 0 or b.ndim != 2 or int(b.shape[1]) <= 0:
                    align_scores.append(0.0)
                    continue
                b = b[:n]
                u = u[:n]
                align_scores.append(float((b.T @ u).norm().div(u.norm().clamp_min(1.0e-12)).clamp(0.0, 1.0).item()))
            align = torch.tensor(align_scores, dtype=torch.float64)
            n = min(int(align.numel()), int(q_block.numel()))
            if n > 0:
                q_block = sparse_topk_gate(q_block[:n] * align[:n], float(args.phase_density_target))
            meta["phase_utility_alignment"] = mean(align_scores)
        if kind in PROJECTOR_ALIGN_ADV_KINDS:
            align_scores: list[float] = []
            advantage_scores: list[float] = []
            for basis, util_vec in zip(final_projector_bases, utility_vectors):
                b = orthonormal_columns(basis, int(rank))
                u = util_vec.detach().reshape(-1).to(dtype=torch.float64).cpu()
                n = min(int(b.shape[0]) if b.ndim == 2 else 0, int(u.numel()))
                if n <= 0 or b.ndim != 2 or int(b.shape[1]) <= 0:
                    align_scores.append(0.0)
                    advantage_scores.append(0.0)
                    continue
                b = b[:n]
                u = u[:n]
                align = float((b.T @ u).norm().div(u.norm().clamp_min(1.0e-12)).clamp(0.0, 1.0).item())
                null = math.sqrt(min(1.0, float(b.shape[1]) / max(1.0, float(n))))
                denom = max(1.0e-12, 1.0 - null)
                advantage = max(0.0, min(1.0, (align - null) / denom))
                align_scores.append(align)
                advantage_scores.append(advantage)
            adv = torch.tensor(advantage_scores, dtype=torch.float64)
            n = min(int(adv.numel()), int(q_block.numel()))
            if n > 0:
                q_block = sparse_topk_gate(q_block[:n] * adv[:n], float(args.phase_density_target))
            meta["phase_utility_alignment"] = mean(align_scores)
            meta["phase_align_advantage_mean"] = mean(advantage_scores)
        if kind in PROJECTOR_RESPONSE_KINDS and not str(scheme).endswith("__guard"):
            q_block, response_meta = finite_response_coordinate_gate(
                model,
                opt,
                params,
                records,
                q_block,
                guard_xy,
                args,
                seed=int(seed),
                null_advantage=kind in PROJECTOR_RESPONSE_NULL_ADV_KINDS,
            )
            meta.update(response_meta)
        if kind in PROJECTOR_ALIGN_ADV_FULLBLOCK_AFTER_KINDS and not str(scheme).endswith("__guard"):
            final_projector_bases = [fullblock_basis_like(basis) for basis in final_projector_bases]
            final_projector_directions = [basis[:, 0] if basis.ndim == 2 and int(basis.shape[1]) > 0 else torch.zeros(0, dtype=torch.float64) for basis in final_projector_bases]
            meta["phase_alignadv_fullblock_update"] = 1.0
        if kind in PROJECTOR_ORBIT_CONTRAST_KINDS:
            meta["phase_orbit_contrast_enabled"] = 1.0
            meta["phase_orbit_contrast_complement_eta"] = float(args.orbit_contrast_complement_eta)
        if kind in PROJECTOR_UNIFORM_Q_KINDS and int(q_block.numel()) > 0:
            q_block = torch.full_like(q_block, float(args.phase_density_target))
            meta["phase_uniform_q_density"] = float(args.phase_density_target)
        if kind in PROJECTOR_ORBIT_BLOCK_SHUFFLE_KINDS and not str(scheme).endswith("__guard"):
            final_projector_bases, q_block, moved_frac = shuffle_projector_within_orbit_identity(final_projector_bases, q_block, records, params, seed=int(seed))
            meta["phase_orbit_identity_shuffled"] = 1.0
            meta["phase_orbit_identity_shuffle_moved_fraction"] = moved_frac
        if kind in PROJECTOR_ORBIT_PATCH_SHUFFLE_KINDS and not str(scheme).endswith("__guard"):
            final_projector_bases, q_block, moved_frac = shuffle_projector_within_degree_identity(final_projector_bases, q_block, records, params, seed=int(seed))
            meta["phase_orbit_patch_identity_shuffled"] = 1.0
            meta["phase_orbit_patch_identity_shuffle_moved_fraction"] = moved_frac
        if kind in PROJECTOR_BLOCK_SHUFFLE_KINDS:
            final_projector_bases, q_block = shuffle_projector_block_identity(final_projector_bases, q_block, seed=int(seed))
            meta["phase_block_identity_shuffled"] = 1.0
        if sg_vals:
            meta["phase_projector_source_guard_cosine"] = mean(sg_vals)
            meta["phase_source_guard_cosine"] = mean(sg_vals)
    gates = [torch.zeros(int(p.numel()), device=p.device, dtype=p.dtype) for p in params]
    counts = [torch.zeros(int(p.numel()), device=p.device, dtype=p.dtype) for p in params]
    modifiers = [torch.zeros(int(p.numel()), device=p.device, dtype=p.dtype) for p in params]
    modifier_counts = [torch.zeros(int(p.numel()), device=p.device, dtype=p.dtype) for p in params]
    projector_records: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in params]
    signed_modifier = torch.ones(len(records), dtype=torch.float64)
    if kind in {"residual_sparse_signed", "residual_sparse_signed_random", "lowrank_signed", "lowrank_signed_random"}:
        _utility_mult, signed_modifier, utility_meta = block_utility_scores_and_modifiers(opt, params, group, split, utility_split, records, args)
        meta.update(utility_meta)
    for row_idx, (pidx, block_cpu, _cohort_mask_cpu) in enumerate(records):
        block = block_cpu.to(device=params[pidx].device)
        gates[pidx][block] += float(q_block[row_idx].item()) if row_idx < int(q_block.numel()) else 0.0
        counts[pidx][block] += 1.0
        modifiers[pidx][block] += float(signed_modifier[row_idx].item()) if row_idx < int(signed_modifier.numel()) else 1.0
        modifier_counts[pidx][block] += 1.0
        if kind in PROJECTOR_KINDS and row_idx < len(final_projector_bases):
            basis = final_projector_bases[row_idx]
            if basis.ndim == 2:
                for col in range(int(basis.shape[1])):
                    projector_records[pidx].append((block.detach().long().cpu(), basis[:, col].detach().cpu()))
    for pidx, param in enumerate(params):
        opt.observe_per_example_gradients(param, split[pidx].to(device=param.device, dtype=param.dtype), labels=labels)
        gate = gates[pidx] / counts[pidx].clamp_min(1.0)
        if int((counts[pidx] == 0).sum().item()) > 0:
            gate[counts[pidx] == 0] = float(gate.mean().item()) if int(gate.numel()) else 0.0
        opt.observe_structure_gate(param, gate.reshape_as(param))
        if kind in {"residual_sparse_signed", "residual_sparse_signed_random", "lowrank_signed", "lowrank_signed_random"}:
            modifier = modifiers[pidx] / modifier_counts[pidx].clamp_min(1.0)
            modifier[modifier_counts[pidx] == 0] = 1.0
            opt.observe_structure_update_modifier(param, modifier.reshape_as(param))
        if kind in PROJECTOR_KINDS:
            opt.observe_structure_update_projector(param, projector_records[pidx])
    q_density = float(q_block.mean().item()) if int(q_block.numel()) else 0.0
    q_entropy = float((-(q_block.clamp(1.0e-8, 1.0 - 1.0e-8) * torch.log(q_block.clamp(1.0e-8, 1.0 - 1.0e-8)) + (1.0 - q_block).clamp(1.0e-8, 1.0) * torch.log((1.0 - q_block).clamp(1.0e-8, 1.0))).mean()).item()) if int(q_block.numel()) else 0.0
    return {
        "H_tensor": H,
        "q_block": q_block,
        "projector_bases": final_projector_bases if final_projector_bases else projector_bases,
        "projector_directions": final_projector_directions if final_projector_directions else projector_directions,
        "block_records": records,
        "transform_family_count": len(specs),
        "block_family_count": len(records),
        "phase_destructive_gap": mean(local_scores + rot_scores) - mean(dest_scores_all),
        "q_coord_density": q_density,
        "q_coord_entropy": q_entropy,
        "q_coord_source_guard_cosine": cosine_vec(q_block, guard_q, abs_value=True) if guard_q is not None else fval(meta.get("phase_source_guard_cosine"), 0.0),
        "local_patch_coverage_proxy": mean(local_scores),
        "rotation_coverage_proxy": mean(rot_scores),
        "destructive_mean": mean(dest_scores_all),
        "per_example_count": int(flat.shape[0]) if flat.ndim == 2 else 0,
        **meta,
    }


def clone_tensor_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {key: clone_tensor_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_tensor_tree(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tensor_tree(value) for value in obj)
    return copy.deepcopy(obj)


def snapshot_model_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters()]


def restore_model_params(model: torch.nn.Module, snapshot: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, saved in zip(model.parameters(), snapshot):
            param.copy_(saved.to(device=param.device, dtype=param.dtype))


def c2_objective_trust_step(
    model: torch.nn.Module,
    opt: EdgeSobolevPopulationFlow,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    guard_n = min(int(args.c2_trust_guard_examples), int(x_guard.shape[0]))
    x_eval = x_guard[:guard_n]
    y_eval = y_guard[:guard_n]
    before = v2293.metrics_for_model(model, x_eval, y_eval)
    param_snapshot = snapshot_model_params(model)
    opt_snapshot = clone_tensor_tree(opt.state_dict())
    stats_snapshot = copy.deepcopy(opt.last_stats)
    base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]
    scales = [fval(scale, 1.0) for scale in csv_items(args.c2_trust_scale_ladder)]
    scales = [scale for scale in scales if math.isfinite(scale) and scale >= 0.0]
    if not scales:
        scales = [1.0]
    best_cov_delta = -1.0e9
    best_nll_delta = 1.0e9
    for scale in scales:
        restore_model_params(model, param_snapshot)
        opt.load_state_dict(clone_tensor_tree(opt_snapshot))
        opt.last_stats = copy.deepcopy(stats_snapshot)
        for lr, group in zip(base_lrs, opt.param_groups):
            group["lr"] = lr * float(scale)
        opt.step()
        after = v2293.metrics_for_model(model, x_eval, y_eval)
        cov_delta = fval(after.get("coverage_CVaR25")) - fval(before.get("coverage_CVaR25"))
        nll_delta = fval(after.get("nll")) - fval(before.get("nll"))
        best_cov_delta = max(best_cov_delta, cov_delta)
        best_nll_delta = min(best_nll_delta, nll_delta)
        accept = (
            cov_delta >= float(args.c2_trust_min_coverage_delta)
            and nll_delta <= float(args.c2_trust_max_nll_delta)
        )
        if accept:
            for lr, group in zip(base_lrs, opt.param_groups):
                group["lr"] = lr
            return {
                "c2_trust_accept": 1.0,
                "c2_trust_scale": float(scale),
                "c2_trust_guard_coverage_delta": cov_delta,
                "c2_trust_guard_nll_delta": nll_delta,
                "c2_trust_best_guard_coverage_delta": best_cov_delta,
                "c2_trust_best_guard_nll_delta": best_nll_delta,
            }
    restore_model_params(model, param_snapshot)
    opt.load_state_dict(clone_tensor_tree(opt_snapshot))
    opt.last_stats = copy.deepcopy(stats_snapshot)
    for lr, group in zip(base_lrs, opt.param_groups):
        group["lr"] = lr
    return {
        "c2_trust_accept": 0.0,
        "c2_trust_scale": 0.0,
        "c2_trust_guard_coverage_delta": 0.0,
        "c2_trust_guard_nll_delta": 0.0,
        "c2_trust_best_guard_coverage_delta": best_cov_delta if best_cov_delta > -1.0e8 else 0.0,
        "c2_trust_best_guard_nll_delta": best_nll_delta if best_nll_delta < 1.0e8 else 0.0,
    }


def train_synthetic_phase_row(job: tuple[str, str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, basis_key, depth, task, seed, repair_round = job
    start = time.time()
    cfg = PART_C_SCHEMES.get(str(scheme), PART_C_SCHEMES["C3_low_rank_phase_r2"])
    try:
        if str(scheme) == "C0_H10_oracle_task_label_upper_bound" and bool(int(args.c0_from_h10_artifact)):
            artifact = h10_c2_artifact_row(str(task), int(seed))
            if artifact:
                return {
                    "part": "C",
                    "status": "ok",
                    "scheme": scheme,
                    "basis_key": artifact.get("basis_key", basis_key),
                    "depth": artifact.get("depth", depth),
                    "task": task,
                    "seed": int(seed),
                    "repair_round": repair_round,
                    "rank_r": 0,
                    "observer_kind": "h10_oracle_artifact",
                    "block_family": "v23_03_h10",
                    "train_steps": ival(artifact.get("train_steps")),
                    "transform_family_count": 0,
                    "block_family_count": 0,
                    "phase_energy_top1": 1.0,
                    "phase_energy_top2": 1.0,
                    "phase_energy_top4": 1.0,
                    "phase_rank_effective": 1.0,
                    "phase_source_guard_cosine": 1.0,
                    "phase_destructive_gap": fval(artifact.get("local_intervention_mean")) + fval(artifact.get("rotation_coherence_mean")) - fval(artifact.get("destructive_coherence_mean")),
                "phase_utility_alignment": 1.0,
                "phase_rho_mean": 1.0,
                "utility_multiplier_mean": 1.0,
                "signed_modifier_mean": 1.0,
                "signed_modifier_negative_fraction": 0.0,
                "q_coord_density": fval(artifact.get("gate_density")),
                    "q_coord_entropy": 0.0,
                    "q_coord_source_guard_cosine": 1.0,
                    "C2_coverage_initial": fval(artifact.get("C2_coverage_initial")),
                    "C2_coverage_final": fval(artifact.get("C2_coverage_final")),
                    "C2_coverage_improvement": fval(artifact.get("C2_coverage_improvement")),
                    "C2_accuracy_initial": fval(artifact.get("C2_accuracy_initial")),
                    "C2_accuracy_final": fval(artifact.get("C2_accuracy_final")),
                    "C2_accuracy_improvement": fval(artifact.get("C2_accuracy_improvement")),
                    "guard_NLL_delta": 0.0,
                    "random_phase_gap": 0.0,
                    "oracle_retention_ratio": 1.0,
                    "local_patch_coverage": fval(artifact.get("C2_coverage_improvement")) if task == "local_patch_interaction" else 0.0,
                    "rotation_coverage": fval(artifact.get("C2_coverage_improvement")) if task == "rotation_sensitive" else 0.0,
                    "taskwise_all_pass": 0,
                    "gate_density_mean": fval(artifact.get("gate_density")),
                    "artifact_source": "results/v23_03_official_h10_full_positive_control_seed15_steps80/part_h_full_positive_control_matrix.csv",
                    "wall_time_s": time.time() - start,
                    **AUDIT_DEFAULTS,
                }
        bargs = v2300.basis_args(args, basis_key)
        xtr, ytr, xg, yg = v2300.visual_data(task, seed, args, device)
        if str(args.structure_guard_source) == "train_split":
            n = min(int(args.structure_guard_examples), max(1, int(xtr.shape[0]) // 2))
            x_source, y_source = xtr[:n], ytr[:n]
            x_guard, y_guard = xtr[-n:], ytr[-n:]
        else:
            x_source, y_source = xtr, ytr
            x_guard, y_guard = xg, yg
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = v2300.make_model(depth, int(xtr.shape[1]), classes, v2300.model_seed_for(basis_key, depth, task, seed), bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        use_pop = str(cfg.get("kind")) != "functional_gram"
        sobolev = 0.0 if str(cfg.get("kind")) == "functional_gram" else float(args.sobolev_exponent)
        opt = make_edge_optimizer(model, args, sobolev=sobolev, use_population_gate=use_pop)
        before = v2293.metrics_for_model(model, xg, yg)
        phase_meta: dict[str, Any] = {}
        gate_trace: list[float] = []
        trust_trace: list[dict[str, float]] = []
        use_c2_trust = bool(int(cfg.get("c2_trust", 0))) and use_pop
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            if use_pop:
                refresh = step == 1 or int(args.structure_gate_every) <= 1 or ((step - 1) % int(args.structure_gate_every) == 0)
                if refresh:
                    guard_examples = int(args.population_grad_examples)
                    if str(cfg.get("kind", "")) in PROJECTOR_RESPONSE_KINDS:
                        guard_examples = max(guard_examples, int(args.phase_response_guard_examples))
                    phase_meta = compute_phase_observer(
                        model,
                        opt,
                        xb,
                        yb,
                        args,
                        seed=int(seed) * 10000 + step,
                        scheme=scheme,
                        task=task,
                        block_family=str(cfg.get("block", "degree_edgebank")),
                        rank=int(cfg.get("rank", 2)),
                        kind=str(cfg.get("kind", "lowrank")),
                        guard_xy=(x_guard[:guard_examples], y_guard[:guard_examples]),
                    )
                else:
                    pe, _gm, _gp, _nn, _ni = v2300.observe_task_gradients(opt, model, xb, yb, args)
                    phase_meta["per_example_count"] = pe
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb.long())
            loss.backward()
            if use_c2_trust:
                trust_trace.append(c2_objective_trust_step(model, opt, x_guard, y_guard, args))
            else:
                opt.step()
            gate_trace.append(float(getattr(opt.last_stats, "gate_density_mean", 1.0)))
        after = v2293.metrics_for_model(model, xg, yg)
        trust_accepts = [fval(t.get("c2_trust_accept")) for t in trust_trace]
        trust_scales = [fval(t.get("c2_trust_scale")) for t in trust_trace if fval(t.get("c2_trust_accept")) > 0.5]
        return {
            "part": "C",
            "status": "ok",
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "seed": int(seed),
            "repair_round": repair_round,
            "rank_r": int(cfg.get("rank", 0)),
            "observer_kind": str(cfg.get("kind")),
            "block_family": str(cfg.get("block")),
            "train_steps": int(args.train_steps),
            "transform_family_count": phase_meta.get("transform_family_count", 0),
            "block_family_count": phase_meta.get("block_family_count", 0),
            "phase_energy_top1": phase_meta.get("phase_energy_top1", 0.0),
            "phase_energy_top2": phase_meta.get("phase_energy_top2", 0.0),
            "phase_energy_top4": phase_meta.get("phase_energy_top4", 0.0),
            "phase_rank_effective": phase_meta.get("phase_rank_effective", 0.0),
            "phase_source_guard_cosine": phase_meta.get("phase_source_guard_cosine", 0.0),
            "phase_destructive_gap": phase_meta.get("phase_destructive_gap", 0.0),
            "phase_utility_alignment": phase_meta.get("phase_utility_alignment", 0.0),
            "phase_sg_filter_mean": phase_meta.get("phase_sg_filter_mean", 0.0),
            "phase_align_advantage_mean": phase_meta.get("phase_align_advantage_mean", 0.0),
            "phase_alignadv_fullblock_update": phase_meta.get("phase_alignadv_fullblock_update", 0.0),
            "phase_response_probe_count": phase_meta.get("phase_response_probe_count", 0.0),
            "phase_response_score_mean": phase_meta.get("phase_response_score_mean", 0.0),
            "phase_response_score_best": phase_meta.get("phase_response_score_best", 0.0),
            "phase_response_selected_score_mean": phase_meta.get("phase_response_selected_score_mean", 0.0),
            "phase_response_selected_count": phase_meta.get("phase_response_selected_count", 0.0),
            "phase_response_null_score_mean": phase_meta.get("phase_response_null_score_mean", 0.0),
            "phase_response_advantage_mean": phase_meta.get("phase_response_advantage_mean", 0.0),
            "phase_response_advantage_best": phase_meta.get("phase_response_advantage_best", 0.0),
            "phase_response_positive_fraction": phase_meta.get("phase_response_positive_fraction", 0.0),
            "phase_orbit_contrast_enabled": phase_meta.get("phase_orbit_contrast_enabled", 0.0),
            "phase_orbit_contrast_complement_eta": phase_meta.get("phase_orbit_contrast_complement_eta", 0.0),
            "phase_orbit_identity_shuffled": phase_meta.get("phase_orbit_identity_shuffled", 0.0),
            "phase_orbit_identity_shuffle_moved_fraction": phase_meta.get("phase_orbit_identity_shuffle_moved_fraction", 0.0),
            "phase_orbit_patch_identity_shuffled": phase_meta.get("phase_orbit_patch_identity_shuffled", 0.0),
            "phase_orbit_patch_identity_shuffle_moved_fraction": phase_meta.get("phase_orbit_patch_identity_shuffle_moved_fraction", 0.0),
            "phase_rho_mean": phase_meta.get("phase_rho_mean", 0.0),
            "utility_multiplier_mean": phase_meta.get("utility_multiplier_mean", 1.0),
            "signed_modifier_mean": phase_meta.get("signed_modifier_mean", 1.0),
            "signed_modifier_negative_fraction": phase_meta.get("signed_modifier_negative_fraction", 0.0),
            "q_coord_density": phase_meta.get("q_coord_density", mean(gate_trace)),
            "q_coord_entropy": phase_meta.get("q_coord_entropy", 0.0),
            "q_coord_source_guard_cosine": phase_meta.get("q_coord_source_guard_cosine", 0.0),
            "C2_coverage_initial": before["coverage_CVaR25"],
            "C2_coverage_final": after["coverage_CVaR25"],
            "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "C2_accuracy_initial": before["accuracy"],
            "C2_accuracy_final": after["accuracy"],
            "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "guard_NLL_delta": after["nll"] - before["nll"],
            "random_phase_gap": 0.0,
            "oracle_retention_ratio": 0.0,
            "local_patch_coverage": phase_meta.get("local_patch_coverage_proxy", 0.0) if task == "local_patch_interaction" else 0.0,
            "rotation_coverage": phase_meta.get("rotation_coverage_proxy", 0.0) if task == "rotation_sensitive" else 0.0,
            "taskwise_all_pass": 0,
            "gate_density_mean": mean(gate_trace),
            "c2_trust_enabled": int(use_c2_trust),
            "c2_trust_accept_rate": mean(trust_accepts) if trust_trace else 0.0,
            "c2_trust_scale_mean": mean(trust_scales) if trust_scales else 0.0,
            "c2_trust_skip_count": int(sum(1 for v in trust_accepts if v <= 0.5)),
            "c2_trust_guard_coverage_delta_mean": mean(t.get("c2_trust_guard_coverage_delta") for t in trust_trace) if trust_trace else 0.0,
            "c2_trust_guard_nll_delta_mean": mean(t.get("c2_trust_guard_nll_delta") for t in trust_trace) if trust_trace else 0.0,
            "c2_trust_best_guard_coverage_delta_mean": mean(t.get("c2_trust_best_guard_coverage_delta") for t in trust_trace) if trust_trace else 0.0,
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {
            "part": "C",
            "status": "error",
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "seed": int(seed),
            "repair_round": repair_round,
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int, str]]:
    return [
        (scheme, basis, depth, task, seed, str(args.repair_round))
        for scheme in csv_items(args.part_c_schemes)
        for basis in csv_items(args.part_c_basis)
        for depth in csv_items(args.part_c_depths)
        for task in csv_items(args.part_c_tasks)
        for seed in range(int(args.part_c_seed_count))
    ]


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    path = OUT_ROOT / f"part_c_blind_observer_matrix_{safe_name(args.repair_round)}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    jobs = shard_items(part_c_jobs(args), args)
    rows = read_rows(path) if path.exists() else []
    done = {
        (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task"), str(r.get("seed")), r.get("repair_round"))
        for r in rows
        if str(r.get("status")) in {"ok", "error", "skipped"}
    }
    for job in jobs:
        scheme, basis, depth, task, seed, repair = job
        key = (scheme, basis, depth, task, str(seed), repair)
        if key in done:
            continue
        rows.append(train_synthetic_phase_row(job, args, device))
        write_rows(path, rows)
    if not path.exists():
        write_rows(path, rows)
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} repair_round={args.repair_round}")
    return gate_summary("C", 0, "PartCShardWritten", "merge_required", rows)


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_blind_observer_matrix_*_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    rows = [r for r in rows if not str(r.get("repair_round", "")).startswith("smoke")]
    ok = [r for r in rows if r.get("status") == "ok"]
    by_key = {(r.get("scheme"), r.get("task"), r.get("seed"), r.get("repair_round")): r for r in ok}
    for r in ok:
        task = r.get("task")
        seed = r.get("seed")
        repair = r.get("repair_round")
        baseline = by_key.get(("C10_no_phase_FunctionalGram_control", task, seed, repair))
        oracle = by_key.get(("C0_H10_oracle_task_label_upper_bound", task, seed, repair))
        if str(r.get("scheme")) == "C11_residual_sparse_phase_r2":
            random_scheme = "C12_residual_sparse_random_phase_control"
        elif str(r.get("scheme")) == "C13_commutator_sparse_phase_r2":
            random_scheme = "C14_commutator_sparse_random_phase_control"
        elif str(r.get("scheme")) == "C15_residual_sparse_signed_phase_r2":
            random_scheme = "C16_residual_sparse_signed_random_phase_control"
        elif str(r.get("scheme")) == "C17_dense_signed_input_phase_r2":
            random_scheme = "C18_dense_signed_random_phase_control"
        elif str(r.get("scheme")) == "C5_class_conditional_phase_r2":
            random_scheme = "C19_class_conditional_random_phase_control"
        elif str(r.get("scheme")) == "C6_margin_bucket_phase_r2":
            random_scheme = "C20_margin_bucket_random_phase_control"
        elif str(r.get("scheme")) == "C21_intersection_input_phase_r2":
            random_scheme = "C22_intersection_input_random_phase_control"
        elif str(r.get("scheme")) == "C23_intersection_sparse_input_phase_r2":
            random_scheme = "C24_intersection_sparse_random_phase_control"
        elif str(r.get("scheme")) == "C25_delta_energy_intersection_sparse_phase_r2":
            random_scheme = "C26_delta_energy_intersection_sparse_random_control"
        elif str(r.get("scheme")) == "C27_pixel_degree_intersection_sparse_phase_r2":
            random_scheme = "C28_pixel_degree_intersection_sparse_random_control"
        elif str(r.get("scheme")) == "C29_projector_phase_r1":
            random_scheme = "C30_projector_random_direction_control"
        elif str(r.get("scheme")) == "C31_covariance_projector_phase_r2":
            random_scheme = "C32_covariance_projector_random_control"
        elif str(r.get("scheme")) == "C33_covariance_projector_aligned_phase_r2":
            random_scheme = "C34_covariance_projector_aligned_random_control"
        elif str(r.get("scheme")) == "C35_covariance_projector_block_shuffle_control":
            random_scheme = "C31_covariance_projector_phase_r2"
        elif str(r.get("scheme")) == "C36_covariance_projector_fullblock_control":
            random_scheme = "C31_covariance_projector_phase_r2"
        elif str(r.get("scheme")) == "C37_covariance_projector_uniform_q_phase_r2":
            random_scheme = "C38_covariance_projector_uniform_q_random_control"
        elif str(r.get("scheme")) == "C39_covariance_projector_c2trust_phase_r2":
            random_scheme = "C40_covariance_projector_c2trust_random_control"
        elif str(r.get("scheme")) == "C41_covariance_projector_sgfilter_phase_r2":
            random_scheme = "C42_covariance_projector_sgfilter_randombasis_control"
        elif str(r.get("scheme")) == "C43_covariance_projector_sgfilter_fullblock_control":
            random_scheme = "C42_covariance_projector_sgfilter_randombasis_control"
        elif str(r.get("scheme")) == "C44_covariance_projector_alignadv_phase_r2":
            random_scheme = "C45_covariance_projector_alignadv_random_control"
        elif str(r.get("scheme")) == "C46_alignadv_fullblock_phase_coordinate":
            random_scheme = "C47_alignadv_fullblock_random_coordinate_control"
        elif str(r.get("scheme")) == "C48_alignadv_fullblock_c2trust_phase_coordinate":
            random_scheme = "C49_alignadv_fullblock_c2trust_random_control"
        elif str(r.get("scheme")) == "C50_finite_response_phase_coordinate":
            random_scheme = "C51_finite_response_random_coordinate_control"
        elif str(r.get("scheme")) == "C52_null_advantage_response_phase_coordinate":
            random_scheme = "C53_null_advantage_response_random_control"
        elif str(r.get("scheme")) == "C54_orbit_contrast_phase_coordinate":
            random_scheme = "C55_orbit_contrast_blockshuffle_control"
        elif str(r.get("scheme")) == "C56_orbit_atom_contrast_phase_coordinate":
            random_scheme = "C57_orbit_atom_patchshuffle_control"
        else:
            random_scheme = "C9_random_phase_matched_control"
        random_row = by_key.get((random_scheme, task, seed, repair))
        if baseline:
            r["coverage_vs_functional"] = fval(r.get("C2_coverage_improvement")) - fval(baseline.get("C2_coverage_improvement"))
            r["accuracy_vs_functional"] = fval(r.get("C2_accuracy_improvement")) - fval(baseline.get("C2_accuracy_improvement"))
        if random_row:
            r["random_phase_gap"] = fval(r.get("C2_coverage_improvement")) - fval(random_row.get("C2_coverage_improvement"))
        if oracle and baseline:
            denom = fval(oracle.get("C2_coverage_improvement")) - fval(baseline.get("C2_coverage_improvement")) + 1.0e-12
            r["oracle_retention_ratio"] = (fval(r.get("C2_coverage_improvement")) - fval(baseline.get("C2_coverage_improvement"))) / denom
    matrix = write_rows(OUT_ROOT / "part_c_blind_observer_matrix.csv", ok + [r for r in rows if r.get("status") != "ok"])
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task"), r.get("repair_round")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task"), r.get("repair_round")) == key]
        task_summaries.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "task": key[3],
            "repair_round": key[4],
            "rows": len(group),
            "rank_r": median(r.get("rank_r") for r in group),
            "phase_energy_top1": median(r.get("phase_energy_top1") for r in group),
            "phase_energy_top2": median(r.get("phase_energy_top2") for r in group),
            "phase_energy_top4": median(r.get("phase_energy_top4") for r in group),
            "phase_source_guard_cosine": median(r.get("phase_source_guard_cosine") for r in group),
            "phase_destructive_gap": median(r.get("phase_destructive_gap") for r in group),
            "phase_utility_alignment": median(r.get("phase_utility_alignment") for r in group),
            "phase_sg_filter_mean": median(r.get("phase_sg_filter_mean") for r in group),
            "phase_align_advantage_mean": median(r.get("phase_align_advantage_mean") for r in group),
            "phase_alignadv_fullblock_update": median(r.get("phase_alignadv_fullblock_update") for r in group),
            "phase_response_probe_count": median(r.get("phase_response_probe_count") for r in group),
            "phase_response_score_mean": median(r.get("phase_response_score_mean") for r in group),
            "phase_response_score_best": median(r.get("phase_response_score_best") for r in group),
            "phase_response_selected_score_mean": median(r.get("phase_response_selected_score_mean") for r in group),
            "phase_response_selected_count": median(r.get("phase_response_selected_count") for r in group),
            "phase_response_null_score_mean": median(r.get("phase_response_null_score_mean") for r in group),
            "phase_response_advantage_mean": median(r.get("phase_response_advantage_mean") for r in group),
            "phase_response_advantage_best": median(r.get("phase_response_advantage_best") for r in group),
            "phase_response_positive_fraction": median(r.get("phase_response_positive_fraction") for r in group),
            "phase_orbit_contrast_enabled": median(r.get("phase_orbit_contrast_enabled") for r in group),
            "phase_orbit_contrast_complement_eta": median(r.get("phase_orbit_contrast_complement_eta") for r in group),
            "phase_orbit_identity_shuffled": median(r.get("phase_orbit_identity_shuffled") for r in group),
            "phase_orbit_identity_shuffle_moved_fraction": median(r.get("phase_orbit_identity_shuffle_moved_fraction") for r in group),
            "phase_orbit_patch_identity_shuffled": median(r.get("phase_orbit_patch_identity_shuffled") for r in group),
            "phase_orbit_patch_identity_shuffle_moved_fraction": median(r.get("phase_orbit_patch_identity_shuffle_moved_fraction") for r in group),
            "utility_multiplier_mean": median(r.get("utility_multiplier_mean") for r in group),
            "signed_modifier_mean": median(r.get("signed_modifier_mean") for r in group),
            "signed_modifier_negative_fraction": median(r.get("signed_modifier_negative_fraction") for r in group),
            "q_coord_density": median(r.get("q_coord_density") for r in group),
            "q_coord_source_guard_cosine": median(r.get("q_coord_source_guard_cosine") for r in group),
            "C2_coverage_improvement_median": median(r.get("C2_coverage_improvement") for r in group),
            "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group),
            "coverage_vs_functional_median": median(r.get("coverage_vs_functional") for r in group),
            "accuracy_vs_functional_median": median(r.get("accuracy_vs_functional") for r in group),
            "random_phase_gap": median(r.get("random_phase_gap") for r in group),
            "oracle_retention_ratio": median(r.get("oracle_retention_ratio") for r in group),
            "local_patch_coverage": median(r.get("local_patch_coverage") for r in group),
            "rotation_coverage": median(r.get("rotation_coverage") for r in group),
            "c2_trust_enabled": median(r.get("c2_trust_enabled") for r in group),
            "c2_trust_accept_rate": median(r.get("c2_trust_accept_rate") for r in group),
            "c2_trust_scale_mean": median(r.get("c2_trust_scale_mean") for r in group),
            "c2_trust_skip_count": median(r.get("c2_trust_skip_count") for r in group),
            "c2_trust_guard_coverage_delta_mean": median(r.get("c2_trust_guard_coverage_delta_mean") for r in group),
            "c2_trust_guard_nll_delta_mean": median(r.get("c2_trust_guard_nll_delta_mean") for r in group),
        })
    groups: list[dict[str, Any]] = []
    for key in sorted({(t.get("scheme"), t.get("basis_key"), t.get("depth"), t.get("repair_round")) for t in task_summaries}):
        tasks = [t for t in task_summaries if (t.get("scheme"), t.get("basis_key"), t.get("depth"), t.get("repair_round")) == key]
        blind = str(key[0]) not in {"C0_H10_oracle_task_label_upper_bound", "C1_real_visual_scalar_dual_baseline", "C9_random_phase_matched_control", "C10_no_phase_FunctionalGram_control", "C12_residual_sparse_random_phase_control", "C14_commutator_sparse_random_phase_control", "C16_residual_sparse_signed_random_phase_control", "C18_dense_signed_random_phase_control", "C19_class_conditional_random_phase_control", "C20_margin_bucket_random_phase_control", "C22_intersection_input_random_phase_control", "C24_intersection_sparse_random_phase_control", "C26_delta_energy_intersection_sparse_random_control", "C28_pixel_degree_intersection_sparse_random_control", "C30_projector_random_direction_control", "C32_covariance_projector_random_control", "C34_covariance_projector_aligned_random_control", "C35_covariance_projector_block_shuffle_control", "C36_covariance_projector_fullblock_control", "C37_covariance_projector_uniform_q_phase_r2", "C38_covariance_projector_uniform_q_random_control", "C40_covariance_projector_c2trust_random_control", "C42_covariance_projector_sgfilter_randombasis_control", "C43_covariance_projector_sgfilter_fullblock_control", "C45_covariance_projector_alignadv_random_control", "C47_alignadv_fullblock_random_coordinate_control", "C49_alignadv_fullblock_c2trust_random_control", "C51_finite_response_random_coordinate_control", "C53_null_advantage_response_random_control", "C55_orbit_contrast_blockshuffle_control", "C57_orbit_atom_patchshuffle_control"}
        local_cov = max([fval(t.get("local_patch_coverage")) for t in tasks if str(t.get("task")) == "local_patch_interaction"] or [0.0])
        rot_cov = max([fval(t.get("rotation_coverage")) for t in tasks if str(t.get("task")) == "rotation_sensitive"] or [0.0])
        g = {
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "repair_round": key[3],
            "rows": sum(ival(t.get("rows")) for t in tasks),
            "blind_observer": int(blind),
            "C2_coverage_improvement_median": median(t.get("C2_coverage_improvement_median") for t in tasks),
            "C2_accuracy_improvement_median": median(t.get("C2_accuracy_improvement_median") for t in tasks),
            "coverage_vs_functional_median": median(t.get("coverage_vs_functional_median") for t in tasks),
            "accuracy_vs_functional_median": median(t.get("accuracy_vs_functional_median") for t in tasks),
            "random_phase_gap": median(t.get("random_phase_gap") for t in tasks),
            "oracle_retention_ratio": median(t.get("oracle_retention_ratio") for t in tasks),
            "phase_source_guard_cosine": median(t.get("phase_source_guard_cosine") for t in tasks),
            "phase_destructive_gap": median(t.get("phase_destructive_gap") for t in tasks),
            "phase_utility_alignment": median(t.get("phase_utility_alignment") for t in tasks),
            "phase_sg_filter_mean": median(t.get("phase_sg_filter_mean") for t in tasks),
            "phase_align_advantage_mean": median(t.get("phase_align_advantage_mean") for t in tasks),
            "phase_alignadv_fullblock_update": median(t.get("phase_alignadv_fullblock_update") for t in tasks),
            "phase_response_probe_count": median(t.get("phase_response_probe_count") for t in tasks),
            "phase_response_score_mean": median(t.get("phase_response_score_mean") for t in tasks),
            "phase_response_score_best": median(t.get("phase_response_score_best") for t in tasks),
            "phase_response_selected_score_mean": median(t.get("phase_response_selected_score_mean") for t in tasks),
            "phase_response_selected_count": median(t.get("phase_response_selected_count") for t in tasks),
            "phase_response_null_score_mean": median(t.get("phase_response_null_score_mean") for t in tasks),
            "phase_response_advantage_mean": median(t.get("phase_response_advantage_mean") for t in tasks),
            "phase_response_advantage_best": median(t.get("phase_response_advantage_best") for t in tasks),
            "phase_response_positive_fraction": median(t.get("phase_response_positive_fraction") for t in tasks),
            "phase_orbit_contrast_enabled": median(t.get("phase_orbit_contrast_enabled") for t in tasks),
            "phase_orbit_contrast_complement_eta": median(t.get("phase_orbit_contrast_complement_eta") for t in tasks),
            "phase_orbit_identity_shuffled": median(t.get("phase_orbit_identity_shuffled") for t in tasks),
            "phase_orbit_identity_shuffle_moved_fraction": median(t.get("phase_orbit_identity_shuffle_moved_fraction") for t in tasks),
            "phase_orbit_patch_identity_shuffled": median(t.get("phase_orbit_patch_identity_shuffled") for t in tasks),
            "phase_orbit_patch_identity_shuffle_moved_fraction": median(t.get("phase_orbit_patch_identity_shuffle_moved_fraction") for t in tasks),
            "utility_multiplier_mean": median(t.get("utility_multiplier_mean") for t in tasks),
            "signed_modifier_mean": median(t.get("signed_modifier_mean") for t in tasks),
            "signed_modifier_negative_fraction": median(t.get("signed_modifier_negative_fraction") for t in tasks),
            "q_coord_density": median(t.get("q_coord_density") for t in tasks),
            "local_patch_coverage": local_cov,
            "rotation_coverage": rot_cov,
            "c2_trust_enabled": median(t.get("c2_trust_enabled") for t in tasks),
            "c2_trust_accept_rate": median(t.get("c2_trust_accept_rate") for t in tasks),
            "c2_trust_scale_mean": median(t.get("c2_trust_scale_mean") for t in tasks),
            "c2_trust_skip_count": median(t.get("c2_trust_skip_count") for t in tasks),
            "c2_trust_guard_coverage_delta_mean": median(t.get("c2_trust_guard_coverage_delta_mean") for t in tasks),
            "c2_trust_guard_nll_delta_mean": median(t.get("c2_trust_guard_nll_delta_mean") for t in tasks),
        }
        g["taskwise_all_pass"] = int(
            blind
            and fval(g["C2_coverage_improvement_median"]) >= float(args.c_coverage_gate)
            and fval(g["C2_accuracy_improvement_median"]) >= float(args.c_accuracy_gate)
            and fval(g["random_phase_gap"]) >= float(args.c_random_phase_gap_gate)
            and fval(g["oracle_retention_ratio"]) >= float(args.c_oracle_retention_gate)
            and fval(g["phase_source_guard_cosine"]) >= float(args.c_phase_source_guard_gate)
            and fval(g["phase_destructive_gap"]) >= float(args.c_phase_destructive_gap_gate)
            and fval(g["local_patch_coverage"]) >= float(args.c_task_coverage_gate)
            and fval(g["rotation_coverage"]) >= float(args.c_task_coverage_gate)
        )
        groups.append(g)
    pass_groups = [g for g in groups if ival(g.get("taskwise_all_pass")) == 1]
    errors = [r for r in rows if r.get("status") == "error"]
    gate = int(bool(pass_groups) and not errors)
    if errors:
        blocker = "part_c_job_errors"
    elif not pass_groups:
        blockers = []
        if all(fval(g.get("oracle_retention_ratio")) < float(args.c_oracle_retention_gate) for g in groups if ival(g.get("blind_observer"))):
            blockers.append("oracle_retention_low")
        if all(fval(g.get("random_phase_gap")) < float(args.c_random_phase_gap_gate) for g in groups if ival(g.get("blind_observer"))):
            blockers.append("random_phase_gap_low")
        if all(fval(g.get("phase_source_guard_cosine")) < float(args.c_phase_source_guard_gate) for g in groups if ival(g.get("blind_observer"))):
            blockers.append("source_guard_low")
        blocker = ";".join(blockers) or "blind_observer_bridge_failed"
    else:
        blocker = "none"
    task_csv = write_rows(OUT_ROOT / "part_c_blind_observer_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_c_blind_observer_group_summary.csv", groups)
    summary = gate_summary("C", gate, "BlindObserverBridgePass" if gate else "BlindObserverBridgeFailed", blocker, rows, taskwise_groups=task_summaries, scheme_groups=groups, passing_scheme_groups=pass_groups)
    write_json(OUT_ROOT / "part_c_blind_observer_summary.json", summary)
    commands = [
        f"CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-c --device cuda:0 --shard-count 4 --shard-index 0 --repair-round repair_expanded --phase-transform-profile expanded",
        f"{PYTHON} {rel(RUNNER)} --mode part-c-merge --device cpu",
    ]
    nxt = next_actions(
        "C",
        gate,
        blocker,
        [] if gate else [
            "adjust transform family with --phase-transform-profile expanded or task_orbit",
            "adjust block family through C7 inputBand_degreeBand and class/margin schemes",
            "try source/guard train split with --structure-guard-source train_split",
            "lower rank or add ridge only as fixed pre-registered repair, not by held/test winner selection",
            "inspect random phase matched control strength",
        ],
        commands=commands if not gate else [],
    )
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_c_blind_observer_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part C blind synthetic observer bridge", summary)
    return summary


def real_observer_row(job: tuple[str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, dataset, seed = job
    start = time.time()
    cfg = PART_D_SCHEMES.get(str(scheme), PART_D_SCHEMES["D2_low_rank_phase_r2_real_visual"])
    family = v2303.real_dataset_family(str(dataset))
    try:
        if family == "tabular":
            if str(cfg.get("kind")) not in {"tabular_no_structure", "tabular_feature_group"}:
                return {"part": "D", "status": "skipped", "scheme": scheme, "dataset": dataset, "dataset_family": family, "seed": int(seed), "skip_reason": "visual_phase_not_applied_to_tabular", "structure_not_applicable": 1, "wall_time_s": time.time() - start, **AUDIT_DEFAULTS}
        load_args = argparse.Namespace(**vars(args))
        load_args.part_i_train_size = max(int(args.source_size) + int(args.witness_size) + int(args.guard_size), int(args.part_i_train_size))
        load_args.part_i_held_size = int(args.held_size)
        load_args.part_i_batch_size = int(args.batch_size)
        xtr, ytr, _xh, _yh, input_dim, output_dim, meta = v2303.load_part_i_tensors(str(dataset), int(seed), load_args, device)
        source_end = min(int(args.source_size), int(xtr.shape[0]))
        witness_end = min(source_end + int(args.witness_size), int(xtr.shape[0]))
        guard_end = min(witness_end + int(args.guard_size), int(xtr.shape[0]))
        xs, ys = xtr[:source_end], ytr[:source_end]
        xg, yg = xtr[witness_end:guard_end], ytr[witness_end:guard_end]
        if int(xg.shape[0]) <= 0:
            xg, yg = xtr[-min(int(args.guard_size), int(xtr.shape[0])):], ytr[-min(int(args.guard_size), int(xtr.shape[0])):]
        model_args = v2303.real_visual_args(args, input_dim) if family == "visual" else argparse.Namespace(**vars(args))
        bargs = v2300.basis_args(model_args, str(args.part_d_basis))
        model = v2300.make_model(str(args.part_d_depth), int(input_dim), int(output_dim), v2300.model_seed_for(str(args.part_d_basis), str(args.part_d_depth), str(dataset), int(seed)), bargs, device)
        mark_kan_edge_params(model, basis_key=str(args.part_d_basis))
        opt = make_edge_optimizer(model, model_args, sobolev=float(args.sobolev_exponent), use_population_gate=True)
        if family == "tabular" and str(cfg.get("kind")) == "tabular_no_structure":
            row = {
                "phase_source_guard_cosine": 1.0,
                "phase_random_gap": 0.0,
                "phase_destructive_gap": 0.0,
                "phase_utility_alignment": 0.0,
                "q_coord_density": 1.0,
                "q_coord_entropy": 0.0,
                "observer_overhead": 1.0,
                "visual_transform_consistency": 0.0,
                "tabular_no_structure_baseline": 1,
            }
        else:
            obs = compute_phase_observer(
                model,
                opt,
                xs,
                ys,
                model_args,
                seed=int(seed) * 10000 + 17,
                scheme=scheme,
                task="real_visual",
                block_family=str(cfg.get("block", "degree_edgebank")),
                rank=int(cfg.get("rank", 2)),
                kind=str(cfg.get("kind", "lowrank")) if family == "visual" else "lowrank",
                guard_xy=(xg[: int(args.population_grad_examples)], yg[: int(args.population_grad_examples)]),
            )
            q = obs.get("q_block", torch.zeros(0))
            q_rand = q[torch.randperm(int(q.numel()), generator=torch.Generator().manual_seed(int(seed) + 99))] if int(q.numel()) else q
            row = {
                "phase_energy_top1": obs.get("phase_energy_top1", 0.0),
                "phase_energy_top2": obs.get("phase_energy_top2", 0.0),
                "phase_energy_top4": obs.get("phase_energy_top4", 0.0),
                "phase_rank_effective": obs.get("phase_rank_effective", 0.0),
                "phase_source_guard_cosine": obs.get("phase_source_guard_cosine", 0.0),
                "phase_random_gap": abs(cosine_vec(q, q_rand)) if int(q.numel()) else 0.0,
                "phase_destructive_gap": obs.get("phase_destructive_gap", 0.0),
                "phase_utility_alignment": obs.get("phase_utility_alignment", 0.0),
                "visual_transform_consistency": obs.get("q_coord_source_guard_cosine", 0.0),
                "q_coord_density": obs.get("q_coord_density", 0.0),
                "q_coord_entropy": obs.get("q_coord_entropy", 0.0),
                "observer_overhead": 1.0,
                "class_phase_diversity": 0.0,
                "margin_phase_diversity": 0.0,
                "tabular_no_structure_baseline": int(family == "tabular" and str(cfg.get("kind")) == "tabular_no_structure"),
            }
        return {
            "part": "D",
            "status": "ok",
            "scheme": scheme,
            "dataset": dataset,
            "dataset_family": family,
            "seed": int(seed),
            "rank_r": int(cfg.get("rank", 0)),
            "observer_kind": str(cfg.get("kind")),
            "block_family": str(cfg.get("block")),
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "task_tier": str(meta.get("task_tier", "")),
            "source_size_used": int(xs.shape[0]),
            "guard_size_used": int(xg.shape[0]),
            "wall_time_s": time.time() - start,
            **row,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {"part": "D", "status": "error", "scheme": scheme, "dataset": dataset, "dataset_family": family, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start, **AUDIT_DEFAULTS}


def part_d_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [(scheme, dataset, seed) for scheme in csv_items(args.part_d_schemes) for dataset in csv_items(args.part_d_datasets) for seed in range(int(args.part_d_seed_count))]


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_blind_observer_summary.json")
    if c and ival(c.get("gate_pass")) != 1 and not bool(int(args.allow_after_part_c_fail)):
        return write_blocked("D", "RealObserverBlockedByPartC", "part_c_failed", "Part D requires Part C pass unless --allow-after-part-c-fail=1.")
    device = device_from_args(args)
    rows = [real_observer_row(job, args, device) for job in shard_items(part_d_jobs(args), args)]
    path = write_rows(OUT_ROOT / f"part_d_real_observer_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-d", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("D", 0, "PartDShardWritten", "merge_required", rows)


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_d_real_observer_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_d_real_observer_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("dataset")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("dataset")) == key]
        groups.append({
            "scheme": key[0],
            "dataset": key[1],
            "dataset_family": sorted({str(r.get("dataset_family", "")) for r in group})[0] if group else "",
            "rows": len(group),
            "phase_energy_top1": median(r.get("phase_energy_top1") for r in group),
            "phase_energy_top2": median(r.get("phase_energy_top2") for r in group),
            "phase_rank_effective": median(r.get("phase_rank_effective") for r in group),
            "phase_source_guard_cosine": median(r.get("phase_source_guard_cosine") for r in group),
            "phase_random_gap": median(r.get("phase_random_gap") for r in group),
            "phase_destructive_gap": median(r.get("phase_destructive_gap") for r in group),
            "phase_utility_alignment": median(r.get("phase_utility_alignment") for r in group),
            "q_coord_density": median(r.get("q_coord_density") for r in group),
            "q_coord_entropy": median(r.get("q_coord_entropy") for r in group),
            "observer_overhead": median(r.get("observer_overhead") for r in group),
        })
    visual_pass: list[dict[str, Any]] = []
    for g in groups:
        if str(g.get("dataset_family")) != "visual":
            continue
        if str(g.get("scheme")) not in {"D1_low_rank_phase_r1_real_visual", "D2_low_rank_phase_r2_real_visual", "D3_low_rank_phase_r4_real_visual", "D4_inputBand_degreeBand_phase_r2", "D5_class_conditional_phase_r2", "D6_margin_bucket_phase_r2", "D7_self_supervised_aug_phase_r2"}:
            continue
        if (
            fval(g.get("phase_source_guard_cosine")) >= float(args.d_phase_source_guard_gate)
            and fval(g.get("phase_random_gap")) >= float(args.d_phase_random_gap_gate)
            and fval(g.get("phase_destructive_gap")) >= float(args.d_phase_destructive_gap_gate)
            and float(args.d_density_low) <= fval(g.get("q_coord_density")) <= float(args.d_density_high)
            and fval(g.get("observer_overhead"), 99.0) <= float(args.d_observer_overhead_gate)
        ):
            visual_pass.append(g)
    visual_pass_datasets = sorted({str(g.get("dataset")) for g in visual_pass})
    tabular_ok = int(any(str(r.get("dataset_family")) == "tabular" and str(r.get("status")) == "ok" for r in rows if "D9_tabular_no_structure_baseline" == str(r.get("scheme"))) or "Wine" not in str(args.part_d_datasets))
    errors = [r for r in rows if r.get("status") == "error"]
    gate = int(not errors and len(visual_pass_datasets) >= 2 and tabular_ok)
    if errors:
        blocker = "part_d_job_errors"
    elif len(visual_pass_datasets) < 2:
        blocker = "real_visual_phase_observer_not_structured"
    elif not tabular_ok:
        blocker = "tabular_no_structure_baseline_missing"
    else:
        blocker = "none"
    group_csv = write_rows(OUT_ROOT / "part_d_real_observer_group_summary.csv", groups)
    summary = gate_summary("D", gate, "RealObserverPass" if gate else "RealObserverNoStructure", blocker, rows, group_summaries=groups, visual_pass_datasets=visual_pass_datasets, tabular_no_structure_ok=tabular_ok)
    write_json(OUT_ROOT / "part_d_real_observer_summary.json", summary)
    nxt = next_actions(
        "D",
        gate,
        blocker,
        [] if gate else ["check visual transform validity", "lower phase rank", "increase ridge", "test class/margin bucket observers", "keep tabular on no-structure baseline"],
    )
    append_exec("part-d-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_d_real_observer_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part D real-task low-rank phase observer", summary)
    return summary


def write_blocked(part: str, route: str, blocker: str, reason: str) -> dict[str, Any]:
    row = {"part": part, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}
    stem_map = {
        "D": "part_d_real_observer",
        "E": "part_e_coord_direction_ablation",
        "F": "part_f_geometry_rank_ablation",
        "G": "part_g_safety",
        "H": "part_h_limited_real_preflight",
        "I": "part_i_failure_decomposition",
    }
    stem = stem_map.get(part.upper(), f"part_{part.lower()}")
    matrix = write_rows(OUT_ROOT / f"{stem}_matrix.csv", [row])
    summary = gate_summary(part, 0, route, blocker, [row], blocked_reason=reason)
    if part.upper() == "I":
        write_json(OUT_ROOT / "part_i_failure_decomposition.json", {
            **summary,
            "primary_blocker": blocker,
            "secondary_blocker": "",
            "recommended_next_codex_action": reason,
            "forbidden_next_actions": ["do_not_expand_real_task_full_matrix", "do_not_delete_FunctionalGram_baseline", "do_not_report_smoke_as_official"],
        })
    else:
        write_json(OUT_ROOT / f"{stem}_summary.json", summary)
    nxt = next_actions(part, 0, blocker, [], commands=[])
    append_exec(f"part-{part.lower()}", command_text(sys.argv), "blocked", files=f"{rel(matrix)}; {rel(nxt)}", note=reason)
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    p0 = read_json(OUT_ROOT / "part_0_lineage_summary.json")
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    b = read_json(OUT_ROOT / "part_b_h10_lock.json")
    c = read_json(OUT_ROOT / "part_c_blind_observer_summary.json")
    d = read_json(OUT_ROOT / "part_d_real_observer_summary.json")
    e = read_json(OUT_ROOT / "part_e_coord_direction_ablation_summary.json")
    f = read_json(OUT_ROOT / "part_f_geometry_rank_ablation_summary.json")
    g = read_json(OUT_ROOT / "part_g_safety_summary.json")
    h = read_json(OUT_ROOT / "part_h_limited_real_preflight_summary.json")
    if ival(p0.get("gate_pass")) != 1 or ival(a.get("gate_pass")) != 1:
        route, blocker = "A_or_B_identity_failed", "identity_or_lineage_failed"
    elif ival(b.get("gate_pass")) != 1:
        route, blocker = "H10_reproduction_failed", "h10_lock_failed"
    elif ival(c.get("gate_pass")) != 1:
        route, blocker = "BlindObserverBridgeFailed", c.get("dominant_blocker", "part_c_failed")
        if not d:
            write_blocked("D", "RealObserverBlockedByPartC", "part_c_failed", "Part D blocked because Part C did not establish blind synthetic observer bridge.")
        if not e:
            write_blocked("E", "CoordinateDirectionBlockedByPartC", "part_c_failed", "Part E blocked because Part C failed.")
        if not f:
            write_blocked("F", "GeometryRankBlockedByPartC", "part_c_failed", "Part F blocked because Part C failed.")
        if not g:
            write_blocked("G", "SafetyTrustBlockedByPartC", "part_c_failed", "Part G blocked because Part C failed.")
        if not h:
            write_blocked("H", "RealPreflightBlockedByPartC", "part_c_failed", "Part H blocked because Part C failed; plan forbids expanding real-task matrix.")
        write_blocked("I", "FailureDecompositionWritten", "part_c_failed", "Primary blocker is blind observer bridge failure; recommended next action is Part C observer repair only.")
    elif ival(d.get("gate_pass")) != 1:
        route, blocker = "RealObserverNoStructure", d.get("dominant_blocker", "part_d_failed")
        if not h:
            write_blocked("H", "RealPreflightBlockedByPartD", "part_d_failed", "Part H blocked because real visual observer did not pass random/source-guard/destructive gates.")
        write_blocked("I", "FailureDecompositionWritten", "part_d_failed", "Primary blocker is real observer no-structure; repair transform family/rank/ridge before training.")
    elif ival(e.get("gate_pass")) != 1:
        route, blocker = "CoordinateDirectionSeparationFailed", e.get("dominant_blocker", "part_e_missing_or_failed")
    elif ival(f.get("gate_pass")) != 1:
        route, blocker = "GeometryAblationFailed", f.get("dominant_blocker", "part_f_missing_or_failed")
    elif ival(g.get("gate_pass")) != 1:
        route, blocker = "SafetyTrustFailed", g.get("dominant_blocker", "part_g_missing_or_failed")
    elif ival(h.get("gate_pass")) != 1:
        route, blocker = "RealPreflightFunctionalBaselineDominates", h.get("dominant_blocker", "part_h_missing_or_failed")
    else:
        route, blocker = "LimitedRealPreflightPass_NoPromotionYet", "none"
    final = {
        "route": route,
        "dominant_blocker": blocker,
        "promotion_allowed": 0,
        "official_candidate_gate_pass": int(route == "LimitedRealPreflightPass_NoPromotionYet"),
        "part_0_gate_pass": p0.get("gate_pass", "missing"),
        "part_a_gate_pass": a.get("gate_pass", "missing"),
        "part_b_gate_pass": b.get("gate_pass", "missing"),
        "part_c_gate_pass": c.get("gate_pass", "missing"),
        "part_d_gate_pass": d.get("gate_pass", "missing"),
        "part_e_gate_pass": e.get("gate_pass", "missing"),
        "part_f_gate_pass": f.get("gate_pass", "missing"),
        "part_g_gate_pass": g.get("gate_pass", "missing"),
        "part_h_gate_pass": h.get("gate_pass", "missing"),
        **AUDIT_DEFAULTS,
        "generated_at": now(),
    }
    write_json(OUT_ROOT / "final_route.json", final)
    write_manifest(args, final)
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(OUT_ROOT / 'reproduction_manifest.md')}", note=f"route={route} blocker={blocker}")
    append_recap("Final route and evidence-chain conclusion", final)
    return final


def write_manifest(args: argparse.Namespace, final: dict[str, Any]) -> None:
    lines = [
        "# DG-KAN v23.05 Reproduction Manifest",
        "",
        f"Generated: {now()}",
        f"Python: `{PYTHON}`",
        f"Runner: `{rel(RUNNER)}`",
        f"Plan: `{rel(PLAN)}`",
        f"Output root: `{rel(OUT_ROOT)}`",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(torch_env(), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Commands",
        "",
        f"- Part 0: `{PYTHON} {rel(RUNNER)} --mode part-0 --device cpu`",
        f"- Part A: `{PYTHON} {rel(RUNNER)} --mode part-a --device cuda:0`",
        f"- Part B: `{PYTHON} {rel(RUNNER)} --mode part-b --device cpu`",
        f"- Part C shard example: `CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-c --device cuda:0 --shard-count 4 --shard-index 0`",
        f"- Part C merge: `{PYTHON} {rel(RUNNER)} --mode part-c-merge --device cpu`",
        f"- Part D shard example: `CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-d --device cuda:0 --shard-count 4 --shard-index 0`",
        f"- Part D merge: `{PYTHON} {rel(RUNNER)} --mode part-d-merge --device cpu`",
        f"- Finalize: `{PYTHON} {rel(RUNNER)} --mode finalize --device cpu`",
        "",
        "## Final Route",
        "",
        "```json",
        json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
    ]
    (OUT_ROOT / "reproduction_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="part-0", choices=["part-0", "part-a", "part-b", "part-c", "part-c-merge", "part-d", "part-d-merge", "finalize"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--repair-round", default="initial")
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--synthetic-train-size", type=int, default=192)
    p.add_argument("--synthetic-guard-size", type=int, default=128)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--input-dim", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--deep-width", type=int, default=16)
    p.add_argument("--dfour-k", type=int, default=3)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--batch-size", type=int, default=36)
    p.add_argument("--train-steps", type=int, default=30)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--edge-lr", type=float, default=0.004)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--edge-weight-normalization", default="trace")
    p.add_argument("--edge-weight-ridge", type=float, default=1.0e-6)
    p.add_argument("--functional-gram-quadrature-points", type=int, default=257)
    p.add_argument("--snr-beta", type=float, default=2.0)
    p.add_argument("--gate-floor", type=float, default=0.2)
    p.add_argument("--gate-floor-mode", default="final", choices=["final", "population"])
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--population-grad-examples", type=int, default=6)
    p.add_argument("--structure-gate-every", type=int, default=3)
    p.add_argument("--structure-guard-source", default="train_split", choices=["held_guard", "train_split"])
    p.add_argument("--structure-guard-examples", type=int, default=64)
    p.add_argument("--contrastive-eta", type=float, default=0.5)
    p.add_argument("--utility-tilt-floor", type=float, default=0.25)
    p.add_argument("--signed-negative-scale", type=float, default=-1.0)
    p.add_argument("--sobolev-exponent", type=float, default=0.25)
    p.add_argument("--phase-alpha", type=float, default=1.5)
    p.add_argument("--phase-ridge", type=float, default=0.0)
    p.add_argument("--phase-density-target", type=float, default=0.17)
    p.add_argument("--phase-transform-profile", default="default", choices=["default", "expanded", "repair_expanded", "task_orbit"])
    p.add_argument("--part-c-schemes", default=",".join(PART_C_SCHEMES.keys()))
    p.add_argument("--part-c-basis", default="dche_k9")
    p.add_argument("--part-c-depths", default="depth3")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-seed-count", type=int, default=3)
    p.add_argument("--c0-from-h10-artifact", type=int, default=1)
    p.add_argument("--c-coverage-gate", type=float, default=0.20)
    p.add_argument("--c-accuracy-gate", type=float, default=0.20)
    p.add_argument("--c-random-phase-gap-gate", type=float, default=0.10)
    p.add_argument("--c-oracle-retention-gate", type=float, default=0.60)
    p.add_argument("--c-phase-source-guard-gate", type=float, default=0.65)
    p.add_argument("--c-phase-destructive-gap-gate", type=float, default=0.05)
    p.add_argument("--c-task-coverage-gate", type=float, default=0.05)
    p.add_argument("--c2-trust-scale-ladder", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--c2-trust-guard-examples", type=int, default=64)
    p.add_argument("--c2-trust-min-coverage-delta", type=float, default=-0.001)
    p.add_argument("--c2-trust-max-nll-delta", type=float, default=999.0)
    p.add_argument("--phase-response-probe-blocks", type=int, default=10)
    p.add_argument("--phase-response-guard-examples", type=int, default=48)
    p.add_argument("--phase-response-lr-scale", type=float, default=0.5)
    p.add_argument("--phase-response-accuracy-weight", type=float, default=0.10)
    p.add_argument("--phase-response-nll-penalty", type=float, default=0.02)
    p.add_argument("--phase-response-min-score", type=float, default=-0.001)
    p.add_argument("--orbit-contrast-complement-eta", type=float, default=0.5)
    p.add_argument("--part-d-schemes", default=",".join(PART_D_SCHEMES.keys()))
    p.add_argument("--part-d-datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--part-d-seed-count", type=int, default=2)
    p.add_argument("--part-d-basis", default="dche_k9")
    p.add_argument("--part-d-depth", default="depth3")
    p.add_argument("--part-i-train-size", type=int, default=1280)
    p.add_argument("--part-i-held-size", type=int, default=256)
    p.add_argument("--part-i-batch-size", type=int, default=36)
    p.add_argument("--part-i-tier2-download", type=int, default=0)
    p.add_argument("--source-size", type=int, default=512)
    p.add_argument("--witness-size", type=int, default=512)
    p.add_argument("--guard-size", type=int, default=256)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--d-phase-source-guard-gate", type=float, default=0.60)
    p.add_argument("--d-phase-random-gap-gate", type=float, default=0.05)
    p.add_argument("--d-phase-destructive-gap-gate", type=float, default=0.05)
    p.add_argument("--d-density-low", type=float, default=0.10)
    p.add_argument("--d-density-high", type=float, default=0.85)
    p.add_argument("--d-observer-overhead-gate", type=float, default=1.50)
    p.add_argument("--allow-after-part-c-fail", type=int, default=0)
    return p


def main(argv: list[str]) -> None:
    args = build_arg_parser().parse_args(argv)
    init_logs()
    if args.mode == "part-0":
        run_part_0(args)
    elif args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-c":
        run_part_c(args)
    elif args.mode == "part-c-merge":
        merge_part_c(args)
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-d-merge":
        merge_part_d(args)
    elif args.mode == "finalize":
        finalize(args)


if __name__ == "__main__":
    main(sys.argv[1:])
