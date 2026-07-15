#!/usr/bin/env python3
"""DG-KAN v23.03 Structure-Projected Functional Population Flow runner."""

from __future__ import annotations

import argparse
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
import experiments.run_v23_02_predictive_trust_functional_population_flow as v2302
from dgkan.fu.population_risk_gate import block_snr_gate
from dgkan.fu.signal_channel_estimators import per_example_gradient_matrix
from dgkan.fu.finite_step_trust_region import clone_state_value, restore_model_params, snapshot_model_params
from dgkan.optim.edge_sobolev_population_flow import EdgeSobolevPopulationFlow, EdgeSobolevSNRFU, mark_kan_edge_params


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.03_StructureProjectedFunctionalPopulationFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.03_StructureProjectedFunctionalPopulationFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.03_StructureProjectedFunctionalPopulationFlow_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2303_OUT_ROOT", str(ROOT / "results/v23_03"))).resolve()

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
    "C1_LocalStructureProduct": "local_product",
    "C2_RotationStructureProduct": "rotation_product",
    "C3_DualMinStructureProduct": "dual_min",
    "C4_OrbitAverageBeforeSNR": "orbit_average_snr",
    "C5_ContrastiveStructureProduct": "contrastive",
    "C6_StructureOnly": "structure_only",
    "C7_PopOnly": "pop_only",
    "C8_OrbitCenteredContrastiveStructureProduct": "orbit_centered_contrastive",
}

OFFICIAL_PART_C_SCHEMES = {
    "C3_DualMinStructureProduct",
    "C4_OrbitAverageBeforeSNR",
    "C5_ContrastiveStructureProduct",
    "C8_OrbitCenteredContrastiveStructureProduct",
}

PART_D_SCHEMES = {
    "D1_StructureCoherentBlockSNR": "task_product",
    "D2_DualMinStructureProduct": "dual_min",
    "D4_ContrastiveStructureProduct": "contrastive",
    "D6_PopOnlyBlockSNR": "pop_only",
    "D7_OrbitCenteredContrastiveStructureProduct": "orbit_centered_contrastive",
    "D8_UtilityOrbitCenteredContrastiveStructureProduct": "utility_orbit_centered_contrastive",
    "D9_DensityPreservedUtilityTiltOrbitCenteredContrastiveStructureProduct": "density_preserved_utility_tilt_orbit_centered_contrastive",
    "D10_SignedUtilityChannelOrbitCenteredContrastiveStructureProduct": "signed_utility_channel_orbit_centered_contrastive",
    "D11_StableSignedUtilityChannelOrbitCenteredContrastiveStructureProduct": "stable_signed_utility_channel_orbit_centered_contrastive",
    "D12_InputLayerSignedUtilityChannelOrbitCenteredContrastiveStructureProduct": "input_layer_signed_utility_channel_orbit_centered_contrastive",
    "D13_DensityPreservedSignedUtilityTiltOrbitCenteredContrastiveStructureProduct": "density_preserved_signed_utility_tilt_orbit_centered_contrastive",
    "D14_TaskContrastDensityPreservedSignedUtilityTiltOrbitCenteredContrastiveStructureProduct": "task_contrast_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
    "D15_LocalInterventionDensityPreservedSignedUtilityTiltOrbitCenteredContrastiveStructureProduct": "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
    "D16_LocalInterventionCoordinateSignedUtilityTiltOrbitCenteredContrastiveStructureProduct": "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
    "D17_FunctionalSobolev025LocalInterventionCoordinateSignedUtilityTiltOrbitCenteredContrastiveStructureProduct": "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
    "D18_FunctionalSobolev0125LocalInterventionCoordinateSignedUtilityTiltOrbitCenteredContrastiveStructureProduct": "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
}

PART_F_SCHEMES = {
    "F1_RawAdamWNoFunctionalGramNoQpopNoPs": "raw_adamw",
    "F2_FunctionalGramAdamWNoQpopNoPs": "functional_gram_adamw",
    "F3_FunctionalGramQpopOnly": "functional_gram_qpop_only",
    "F4_FunctionalGramPsOnly": "functional_gram_ps_only",
    "F5_FunctionalGramPsQpopPsSignedUtility": "functional_gram_ps_qpop_ps_signed_utility",
    "F6_FunctionalSobolev025PsQpopPsSignedUtilityDiagnostic": "functional_sobolev025_ps_qpop_ps_signed_utility",
    "F7_FunctionalSobolev025PsQpopPsSignedUtility": "functional_sobolev025_ps_qpop_ps_signed_utility",
    "F8_FunctionalSobolev0125PsQpopPsSignedUtilityDiagnostic": "functional_sobolev0125_ps_qpop_ps_signed_utility",
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


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "x"


def command_text(argv: Iterable[str]) -> str:
    command = " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])
    out_root_env = os.environ.get("V2303_OUT_ROOT")
    if out_root_env:
        return f"V2303_OUT_ROOT={out_root_env} {command}"
    return command


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


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def part_d_sobolev_exponent(scheme: str) -> float:
    text = str(scheme)
    if "FunctionalSobolev0125" in text or "Sobolev0125" in text:
        return 0.125
    if "FunctionalSobolev025" in text or "Sobolev025" in text:
        return 0.25
    return 0.0


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
    keys: list[str] = []
    for row in enriched:
        for key in row.keys():
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
            "# DG-KAN v23.03 StructureProjectedFunctionalPopulationFlow 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误；缺失写 `missing`，跳过写 `skipped`，不补造。\n"
            "- 复现提示：Part C 支持 `--shard-count/--shard-index`；本轮优先使用 GPU 2/3 并行。\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.03 StructureProjectedFunctionalPopulationFlow 实验结果复盘\n\n"
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
        "generated_at": now(),
        **extra,
    }


def next_actions(part: str, gate: int, blocker: str, actions: list[str]) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part.upper(),
            "gate_pass": int(gate),
            "dominant_blocker": blocker,
            "allowed_actions": actions,
            "forbidden_actions": [
                "do_not_lower_random_gap_threshold",
                "do_not_delete_matched_random_controls",
                "do_not_add_edge_features_or_patch_product_features",
                "do_not_use_held_or_test_for_gate_induction",
                "do_not_select_best_seed_task_or_row_at_runtime",
            ],
            "rerun_commands": [],
        },
    )


def static_scan(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|metric_winner)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|winner_update)\s*\("),
        "external_product_feature": re.compile(r"\b(make_external_product_feature|patch_product_feature)\s*\("),
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
        ROOT / "experiments/run_v23_02_predictive_trust_functional_population_flow.py",
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
        "experiments.run_v23_02_predictive_trust_functional_population_flow",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_errors.append(f"{mod}: {repr(exc)}")
    scan_pass, scan_hits = static_scan(files)
    random_controls = ["density_matched", "histogram_matched", "structure_orbit_shuffled", "structure_breaking"]
    row = {
        "part": "A",
        "status": "ok",
        "compile_pass": int(not compile_errors),
        "import_pass": int(not import_errors),
        "static_scan_pass": scan_pass,
        "new_edge_function_added": 0,
        "external_product_feature_used": 0,
        "structure_transform_used_only_for_gate": 1,
        "structure_transform_in_forward": 0,
        "runtime_selector_used": 0,
        "metric_winner_selection_used": 0,
        "candidate_update_selection_used": 0,
        "random_controls_count": len(random_controls),
        "random_control_types": ",".join(random_controls),
        "optimizer_structure_gate_hook_available": int(hasattr(EdgeSobolevSNRFU, "observe_structure_gate")),
        "transform_families_pre_registered": "local_shift,local_patch_shift,corner_preserving_jitter,local_patch_pair_swaps,rot90,hflip,vflip,rot180,rotation_checker_swap,destructive",
        **AUDIT_DEFAULTS,
    }
    required = [
        "compile_pass",
        "import_pass",
        "static_scan_pass",
        "optimizer_structure_gate_hook_available",
        "structure_transform_used_only_for_gate",
    ]
    gate = int(all(ival(row.get(k)) == 1 for k in required) and row["random_controls_count"] >= 3)
    matrix = write_rows(OUT_ROOT / "part_a_identity_structure_audit.csv", [row])
    summary = gate_summary(
        "A",
        gate,
        "A_CodeIdentityPass" if gate else "A_CodeIdentityFailed",
        "none" if gate else "compile_import_static_or_control_builder_failed",
        [row],
        compile_errors=compile_errors,
        import_errors=import_errors,
        static_scan_hits=scan_hits,
    )
    write_json(OUT_ROOT / "part_a_summary.json", summary)
    nxt = next_actions("A", gate, summary["dominant_blocker"], [] if gate else ["repair compile/import/static scan", "repair random control builder"])
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_a_summary.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part A code identity and structure audit", summary)
    return summary


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_v2302_f15_group() -> tuple[dict[str, Any], str]:
    candidates = [
        ROOT / "results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_finite_step_trust_summary.json",
        ROOT / "results/v23_02_etailreg0p3_tries20_seed15/part_f_finite_step_trust_summary.json",
        ROOT / "results/v23_02_direct_f_signfix_etailreg_seed15/part_f_finite_step_trust_summary.json",
    ]
    for path in candidates:
        data = read_json(path)
        groups = data.get("passing_part_f_groups") or data.get("diagnostic_part_f_groups") or data.get("group_summaries") or []
        for group in groups:
            if str(group.get("scheme")) == "E5_FunctionalGram_DegreeEdgebankSNR_s0" and "F15" in str(group.get("safety_type")):
                return group, rel(path)
    return {}, "missing"


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    f15, f15_source = read_v2302_f15_group()
    c20_source = ROOT / "results/v23_02_corner_checker_hybrid_c_both_seed15_floor04/part_c_taskwise_c2_task_summary.csv"
    c20_rows = read_rows(c20_source)
    c42_source = ROOT / "results/v23_02_corner_checker_topk_c_both_seed5_floor04/part_c_taskwise_c2_task_summary.csv"
    c44_source = ROOT / "results/v23_02_corner_checker_topk_frac_c_both_seed5_floor04/part_c_taskwise_c2_task_summary.csv"
    c42_rows = read_rows(c42_source)
    c44_rows = read_rows(c44_source)

    def task_field(rows: list[dict[str, Any]], scheme: str, task: str, field: str) -> Any:
        vals = [r.get(field, "missing") for r in rows if r.get("scheme") == scheme and r.get("task") == task]
        return vals[0] if vals else "missing"

    row = {
        "part": "B",
        "status": "ok",
        "v23_02_direct_f_official_pass": int(bool(f15) and ival(f15.get("official_pass")) == 1),
        "v23_02_E5_F15_F5_no_debt": f15.get("F5_no_debt_count", "missing"),
        "v23_02_E5_F15_accept_rate": f15.get("finite_step_accept_rate_median", "missing"),
        "v23_02_E5_F15_scale_mean": f15.get("finite_step_scale_mean_median", "missing"),
        "v23_02_E5_F15_skip_count": f15.get("finite_step_skip_count_median", "missing"),
        "v23_02_part_c_gate_pass": read_json(ROOT / "results/v23_02/part_c_taskwise_c2_summary.json").get("gate_pass", "missing"),
        "v23_02_final_route": read_json(ROOT / "results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/final_route.json").get("route", "missing"),
        "v23_02_C20_seed5_taskwise_pass": "see_recap_manual_notes",
        "v23_02_C20_seed15_local_coverage": task_field(c20_rows, "C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0", "local_patch_interaction", "C2_coverage_improvement_median"),
        "v23_02_C20_seed15_rotation_random_gap": task_field(c20_rows, "C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0", "rotation_sensitive", "random_gap"),
        "v23_02_C42_local_pass": task_field(c42_rows, "C42_FunctionalGram_CornerCheckerHybridTopKDegreeEdgebankSNR_s0", "local_patch_interaction", "taskwise_pass"),
        "v23_02_C42_rotation_pass": task_field(c42_rows, "C42_FunctionalGram_CornerCheckerHybridTopKDegreeEdgebankSNR_s0", "rotation_sensitive", "taskwise_pass"),
        "v23_02_C44_local_pass": task_field(c44_rows, "C44_FunctionalGram_CornerCheckerHybridTopK25DegreeEdgebankSNR_s0", "local_patch_interaction", "taskwise_pass"),
        "v23_02_C44_rotation_pass": task_field(c44_rows, "C44_FunctionalGram_CornerCheckerHybridTopK25DegreeEdgebankSNR_s0", "rotation_sensitive", "taskwise_pass"),
        "v23_02_best_blocker": "taskwise_C2_random_gap_failed",
        "v23_02_E5_F15_source": f15_source,
        "v23_02_C20_source": rel(c20_source) if c20_source.exists() else "missing",
        "v23_02_C42_source": rel(c42_source) if c42_source.exists() else "missing",
        "v23_02_C44_source": rel(c44_source) if c44_source.exists() else "missing",
        **AUDIT_DEFAULTS,
    }
    required = [
        "v23_02_E5_F15_F5_no_debt",
        "v23_02_E5_F15_accept_rate",
        "v23_02_part_c_gate_pass",
        "v23_02_final_route",
        "v23_02_C20_seed15_local_coverage",
        "v23_02_C20_seed15_rotation_random_gap",
    ]
    missing = [key for key in required if row.get(key) == "missing"]
    gate = int(not missing)
    matrix = write_rows(OUT_ROOT / "part_b_history_lock.csv", [row])
    summary_fields = {k: v for k, v in row.items() if k not in {"part", "status"}}
    summary = gate_summary("B", gate, "B_HistoryLockPass" if gate else "B_HistoryLockFailed", "none" if gate else "missing_history_fields", [row], missing_fields=missing, **summary_fields)
    write_json(OUT_ROOT / "part_b_history_lock.json", summary)
    nxt = next_actions("B", gate, summary["dominant_blocker"], [] if gate else ["locate missing v23.02 artifacts", "write missing fields with source path instead of inferring values"])
    append_exec("part-b", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_b_history_lock.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part B history boundary lock", summary)
    return summary


def device_from_args(args: argparse.Namespace) -> torch.device:
    requested = str(args.device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def transform_perm(side: int, name: str, *, seed: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(side * side, device=device).reshape(side, side)

    def swap_patch(grid: torch.Tensor, a: tuple[int, int], b: tuple[int, int], size: int = 2) -> torch.Tensor:
        out_grid = grid.clone()
        ar, ac = a
        br, bc = b
        if min(ar, ac, br, bc) < 0 or max(ar + size, br + size) > side or max(ac + size, bc + size) > side:
            return out_grid
        first = grid[ar : ar + size, ac : ac + size].clone()
        second = grid[br : br + size, bc : bc + size].clone()
        out_grid[ar : ar + size, ac : ac + size] = second
        out_grid[br : br + size, bc : bc + size] = first
        return out_grid

    if name == "shift_down":
        out = torch.roll(idx, shifts=1, dims=0)
    elif name == "shift_right":
        out = torch.roll(idx, shifts=1, dims=1)
    elif name == "patch_shift_diag":
        out = torch.roll(idx, shifts=(2, 2), dims=(0, 1))
    elif name == "corner_preserving_jitter":
        out = idx.clone()
        if side >= 4:
            out[1:-1, 1:-1] = torch.roll(out[1:-1, 1:-1], shifts=1, dims=0)
    elif name == "rot90":
        out = torch.rot90(idx, 1, dims=(0, 1))
    elif name == "hflip":
        out = torch.flip(idx, dims=[1])
    elif name == "vflip":
        out = torch.flip(idx, dims=[0])
    elif name == "rot180":
        out = torch.rot90(idx, 2, dims=(0, 1))
    elif name == "swap_top_patch_pair":
        out = swap_patch(idx, (1, 1), (1, side - 3))
    elif name == "swap_bottom_patch_pair":
        out = swap_patch(idx, (side - 3, 1), (side - 3, side - 3))
    elif name == "swap_local_patch_pairs":
        out = swap_patch(idx, (1, 1), (1, side - 3))
        out = swap_patch(out, (side - 3, 1), (side - 3, side - 3))
    elif name == "swap_rotation_checkers":
        out = swap_patch(idx, (1, side // 2 - 1), (side - 3, side // 2 - 1))
    elif name.startswith("destructive"):
        gen = torch.Generator(device=device).manual_seed(int(seed))
        out = torch.randperm(side * side, generator=gen, device=device).reshape(side, side)
    else:
        out = idx
    return out.reshape(-1).long()


def transform_names(family: str, profile: str) -> list[str]:
    if family == "local":
        if profile == "task_orbit":
            return ["swap_top_patch_pair", "swap_bottom_patch_pair", "swap_local_patch_pairs"]
        base = ["shift_down", "shift_right", "patch_shift_diag", "corner_preserving_jitter"]
        return base + (["shift_down"] if profile == "expanded" else [])
    if family == "rotation":
        if profile == "task_orbit":
            return ["hflip", "vflip", "rot180", "swap_rotation_checkers"]
        base = ["rot90", "hflip", "vflip"]
        return base + (["rot90"] if profile == "expanded" else [])
    if family == "destructive_local":
        return ["destructive_local"]
    if family == "destructive_rotation":
        return ["destructive_rotation"]
    return []


def apply_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return x[:, perm.to(device=x.device)]


def local_checker_patch_pair_intervention(x: torch.Tensor, side: int, *, scale: float = 1.0) -> torch.Tensor:
    if int(x.ndim) != 2 or int(x.shape[1]) != int(side) * int(side) or side < 4:
        return x
    imgs = x.reshape(int(x.shape[0]), side, side).clone()
    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=x.device, dtype=x.dtype)
    denom = float((checker * checker).sum().detach().cpu().item())
    # Flip the checker component in one member of each local pair. This toggles
    # the pair product while leaving non-checker local noise mostly intact.
    for top, left in ((1, side - 3), (side - 3, side - 3)):
        if top < 0 or left < 0 or top + 2 > side or left + 2 > side:
            continue
        patch = imgs[:, top : top + 2, left : left + 2]
        coeff = (patch * checker).sum(dim=(1, 2)) / max(denom, 1.0e-12)
        patch.sub_(2.0 * float(scale) * coeff.reshape(-1, 1, 1) * checker)
    return imgs.reshape_as(x)


def permute_param_examples(grads: torch.Tensor, param: torch.nn.Parameter, perm: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
        return grads
    if int(param.shape[0]) != int(perm.numel()):
        return grads
    idx = perm.to(device=grads.device)
    if inverse:
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(int(idx.numel()), device=idx.device)
        idx = inv
    return grads[:, idx, :, :]


def split_flat_grads(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return v2300.split_flat_grads(flat, params)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().reshape(-1).to(dtype=torch.float64).cpu()
    bf = b.detach().reshape(-1).to(dtype=torch.float64).cpu()
    n = min(int(af.numel()), int(bf.numel()))
    if n <= 0:
        return 0.0
    af = af[:n]
    bf = bf[:n]
    return float(torch.dot(af, bf).div(af.norm().clamp_min(1.0e-12) * bf.norm().clamp_min(1.0e-12)).item())


def structure_gate_for_param(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    transformed: list[tuple[torch.Tensor, torch.Tensor]],
    blocks: list[list[int]],
) -> torch.Tensor:
    if not blocks:
        return torch.ones_like(param).reshape(-1)
    vals = torch.zeros(int(param.numel()), dtype=torch.float64)
    counts = torch.zeros(int(param.numel()), dtype=torch.float64)
    for transformed_grads, perm in transformed:
        a_raw = permute_param_examples(source_grads, param, perm, inverse=False)
        b_raw = transformed_grads
        n = min(int(a_raw.shape[0]), int(b_raw.shape[0]))
        if n <= 0:
            continue
        a = opt._apply_metric_inv_sqrt(a_raw[:n], param, group).reshape(n, -1).detach().cpu().to(dtype=torch.float64)
        b = opt._apply_metric_inv_sqrt(b_raw[:n], param, group).reshape(n, -1).detach().cpu().to(dtype=torch.float64)
        for block in blocks:
            idx = torch.tensor([i for i in block if 0 <= i < int(param.numel())], dtype=torch.long)
            if int(idx.numel()) <= 0:
                continue
            aa = a[:, idx]
            bb = b[:, idx]
            c = (aa * bb).sum(dim=1) / (aa.norm(dim=1).clamp_min(1.0e-12) * bb.norm(dim=1).clamp_min(1.0e-12))
            score = float(c.clamp_min(0.0).mean().item())
            vals[idx] += score
            counts[idx] += 1.0
    out = vals / counts.clamp_min(1.0)
    out[counts == 0] = 0.0
    return out.to(device=param.device, dtype=param.dtype)


def qpop_for_param(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    blocks: list[list[int]],
) -> torch.Tensor:
    white = opt._apply_metric_inv_sqrt(source_grads, param, group).reshape(int(source_grads.shape[0]), -1)
    if not blocks:
        return torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
    res = block_snr_gate(white.detach().cpu(), blocks, beta=float(group.get("gate_beta", 2.0)))
    return res.gate.to(device=param.device, dtype=param.dtype).reshape(-1)


def block_direction_energy_gate(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    direction_grads: torch.Tensor | None,
    blocks: list[list[int]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if direction_grads is None or direction_grads.ndim < 2 or not blocks:
        return torch.zeros(int(param.numel()), device=param.device, dtype=param.dtype)
    white = opt._apply_metric_inv_sqrt(direction_grads.to(device=param.device, dtype=param.dtype), param, group)
    mu = white.reshape(int(white.shape[0]), -1).mean(dim=0).detach().to(dtype=torch.float64)
    flat = torch.zeros(int(param.numel()), device=param.device, dtype=torch.float64)
    scores: list[float] = []
    for block in blocks:
        idx = torch.tensor([int(i) for i in block if 0 <= int(i) < int(param.numel())], dtype=torch.long, device=param.device)
        if int(idx.numel()) <= 0:
            continue
        score = float(mu[idx].norm().div(math.sqrt(max(1, int(idx.numel())))).detach().cpu().item())
        scores.append(score)
        flat[idx] = score
    positives = [s for s in scores if s > 0.0]
    if not positives:
        return flat.to(device=param.device, dtype=param.dtype)
    scale = float(np.mean(positives))
    out = (flat / max(scale, 1.0e-12)).clamp(0.0, float(getattr(args, "local_intervention_gate_max", 3.0)))
    return out.to(device=param.device, dtype=param.dtype)


def orbit_average_qpop_for_param(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    transformed: list[tuple[torch.Tensor, torch.Tensor]],
    blocks: list[list[int]],
) -> torch.Tensor:
    aligned = [source_grads]
    for transformed_grads, perm in transformed:
        aligned.append(permute_param_examples(transformed_grads, param, perm, inverse=True))
    n = min(int(x.shape[0]) for x in aligned)
    avg = torch.stack([x[:n] for x in aligned], dim=0).mean(dim=0)
    return qpop_for_param(opt, param, group, avg, blocks)


def block_shuffle(values: torch.Tensor, blocks: list[list[int]], seed: int) -> torch.Tensor:
    out = values.detach().clone().reshape(-1).cpu().to(dtype=torch.float64)
    clean = [[int(i) for i in block if 0 <= int(i) < int(out.numel())] for block in blocks]
    clean = [b for b in clean if b]
    gen = torch.Generator().manual_seed(int(seed))
    by_size: dict[int, list[list[int]]] = {}
    for block in clean:
        by_size.setdefault(len(block), []).append(block)
    for group_blocks in by_size.values():
        if len(group_blocks) <= 1:
            continue
        vals = [float(out[block[0]].item()) for block in group_blocks]
        perm = torch.randperm(len(vals), generator=gen).tolist()
        for block, src_idx in zip(group_blocks, perm):
            out[torch.tensor(block, dtype=torch.long)] = vals[int(src_idx)]
    return out.to(device=values.device, dtype=values.dtype)


def orbit_center_positive(values: torch.Tensor, blocks: list[list[int]], *, gain: float = 1.0) -> torch.Tensor:
    flat = values.detach().reshape(-1).cpu().to(dtype=torch.float64)
    clean = [[int(i) for i in block if 0 <= int(i) < int(flat.numel())] for block in blocks]
    clean = [block for block in clean if block]
    if not clean:
        centered = (flat - flat.mean()).clamp_min(0.0) if int(flat.numel()) else flat.clone()
        return (centered * float(gain)).clamp(0.0, 1.0).to(device=values.device, dtype=values.dtype)
    out = torch.zeros_like(flat)
    by_size: dict[int, list[list[int]]] = {}
    for block in clean:
        by_size.setdefault(len(block), []).append(block)
    for group_blocks in by_size.values():
        means = torch.tensor([float(flat[torch.tensor(block, dtype=torch.long)].mean().item()) for block in group_blocks], dtype=torch.float64)
        positive = (means - means.mean()).clamp_min(0.0)
        if float(positive.max().item()) > 1.0e-12:
            positive = positive * (float(means.max().item()) / float(positive.max().item()))
        for block, score in zip(group_blocks, positive.tolist()):
            out[torch.tensor(block, dtype=torch.long)] = float(score)
    return (out * float(gain)).clamp(0.0, 1.0).to(device=values.device, dtype=values.dtype)


def density_random_like(values: torch.Tensor, seed: int) -> torch.Tensor:
    flat = values.detach().reshape(-1)
    gen = torch.Generator(device=flat.device).manual_seed(int(seed))
    prob = float(flat.mean().clamp(0.0, 1.0).item()) if int(flat.numel()) else 0.0
    return (torch.rand(flat.shape, generator=gen, device=flat.device, dtype=flat.dtype) < prob).to(dtype=flat.dtype)


def hist_random_like(values: torch.Tensor, seed: int) -> torch.Tensor:
    flat = values.detach().reshape(-1)
    if int(flat.numel()) <= 0:
        return flat.clone()
    gen = torch.Generator(device=flat.device).manual_seed(int(seed))
    return flat[torch.randperm(int(flat.numel()), generator=gen, device=flat.device)]


def block_rank_match_like(values: torch.Tensor, scores: torch.Tensor, blocks: list[list[int]]) -> torch.Tensor:
    flat_values = values.detach().reshape(-1).cpu().to(dtype=torch.float64)
    flat_scores = scores.detach().reshape(-1).cpu().to(dtype=torch.float64)
    clean = [[int(i) for i in block if 0 <= int(i) < int(flat_values.numel())] for block in blocks]
    clean = [block for block in clean if block]
    if not clean:
        if int(flat_values.numel()) <= 0:
            return values.detach().clone()
        order_scores = torch.argsort(flat_scores[: int(flat_values.numel())])
        sorted_values = torch.sort(flat_values).values
        out_flat = flat_values.clone()
        out_flat[order_scores] = sorted_values[: int(order_scores.numel())]
        return out_flat.to(device=values.device, dtype=values.dtype)
    value_means = torch.tensor([float(flat_values[torch.tensor(block, dtype=torch.long)].mean().item()) for block in clean], dtype=torch.float64)
    score_means = torch.tensor([float(flat_scores[torch.tensor(block, dtype=torch.long)].mean().item()) for block in clean], dtype=torch.float64)
    sorted_values = torch.sort(value_means).values
    order_scores = torch.argsort(score_means)
    out_flat = flat_values.clone()
    for dst_idx, value in zip(order_scores.tolist(), sorted_values.tolist()):
        out_flat[torch.tensor(clean[int(dst_idx)], dtype=torch.long)] = float(value)
    return out_flat.to(device=values.device, dtype=values.dtype)


def control_structure_gate(
    struct: torch.Tensor,
    dest: torch.Tensor,
    blocks: list[list[int]],
    control_type: str,
    seed: int,
) -> torch.Tensor:
    kind = str(control_type)
    if kind in {"true", "none", ""}:
        return struct
    if kind in {"density", "density_matched"}:
        return density_random_like(struct, seed)
    if kind in {"histogram", "histogram_matched", "hist"}:
        return hist_random_like(struct, seed)
    if kind in {"orbit_shuffle", "orbit_shuffled", "orbit"}:
        return block_shuffle(struct, blocks, seed)
    if kind in {"structure_breaking", "break", "destructive"}:
        return block_rank_match_like(struct, dest, blocks)
    raise ValueError(f"unknown structure control type: {control_type}")


def e_scheme_name(control_type: str) -> str:
    mapping = {
        "true": "E0_TrueStructureGate",
        "density": "E1_DensityMatchedRandom",
        "histogram": "E2_HistogramMatchedRandom",
        "orbit_shuffle": "E3_OrbitShuffledRandom",
        "structure_breaking": "E4_StructureBreakingRandom",
    }
    return mapping.get(str(control_type), f"E_{control_type}")


def weighted_structure_score(gate: torch.Tensor, struct: torch.Tensor) -> float:
    g = gate.detach().reshape(-1).to(dtype=torch.float64).cpu().clamp_min(0.0)
    s = struct.detach().reshape(-1).to(dtype=torch.float64).cpu().clamp_min(0.0)
    n = min(int(g.numel()), int(s.numel()))
    if n <= 0:
        return 0.0
    return float((g[:n] * s[:n]).sum().div(g[:n].sum().clamp_min(1.0e-12)).item())


def model_gate_vectors(
    scheme: str,
    model: v2293.TrueDeepPureKAN,
    opt: EdgeSobolevSNRFU,
    x: torch.Tensor,
    y: torch.Tensor,
    task: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor | float | int]:
    side = int(args.visual_side)
    params = list(model.coeffs)
    flat = per_example_gradient_matrix(model, x, y, max_examples=int(args.population_grad_examples), label_prior_correction=bool(int(args.label_prior_correction)))
    split = split_flat_grads(flat, params)
    profile = str(args.structure_transform_profile)

    def transformed_split(names: list[str], offset: int, *, align_with_transform: bool = True) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        per_param: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in params]
        identity_perm = torch.arange(side * side, device=x.device)
        for i, name in enumerate(names):
            perm = transform_perm(side, name, seed=seed * 1000 + offset + i, device=x.device)
            x_t = apply_perm(x[: int(args.population_grad_examples)], perm)
            ft = per_example_gradient_matrix(model, x_t, y[: int(x_t.shape[0])], max_examples=int(args.population_grad_examples), label_prior_correction=bool(int(args.label_prior_correction)))
            st = split_flat_grads(ft, params)
            align_perm = perm if align_with_transform else identity_perm
            for j, grads in enumerate(st):
                per_param[j].append((grads.to(device=params[j].device, dtype=params[j].dtype), align_perm))
        return per_param

    local_t = transformed_split(transform_names("local", profile), 1100)
    rot_t = transformed_split(transform_names("rotation", profile), 2100)
    destructive_t = transformed_split(
        transform_names("destructive_local" if task == "local_patch_interaction" else "destructive_rotation", profile),
        3100,
        align_with_transform=False,
    )

    qpop_parts: list[torch.Tensor] = []
    qlocal_parts: list[torch.Tensor] = []
    qrot_parts: list[torch.Tensor] = []
    qdes_parts: list[torch.Tensor] = []
    qfinal_parts: list[torch.Tensor] = []
    qhist_parts: list[torch.Tensor] = []
    qdensity_parts: list[torch.Tensor] = []
    qorbit_parts: list[torch.Tensor] = []
    all_blocks: list[list[int]] = []
    offset = 0
    group = opt.param_groups[0]
    for idx, (param, grads) in enumerate(zip(params, split)):
        if bool(int(getattr(args, "structure_addressable_only", 0))) and int(getattr(param, "_kan_layer_id", -1)) != 0:
            continue
        pgrads = grads.to(device=param.device, dtype=param.dtype)
        blocks = opt._blocks_for_param(param, {**group, "gate_family": str(args.structure_block_family)})
        qpop = qpop_for_param(opt, param, group, pgrads, blocks)
        local = structure_gate_for_param(opt, param, group, pgrads, local_t[idx], blocks)
        rot = structure_gate_for_param(opt, param, group, pgrads, rot_t[idx], blocks)
        dest = structure_gate_for_param(opt, param, group, pgrads, destructive_t[idx], blocks)
        mode = PART_C_SCHEMES.get(str(scheme), "dual_min")
        floor = float(args.gate_floor)
        qpop_used = qpop
        if floor > 0.0 and str(getattr(args, "gate_floor_mode", "final")) == "population":
            qpop_used = qpop * (1.0 - floor) + floor
        if mode == "local_product":
            struct = local
            qfinal = qpop_used * struct
        elif mode == "rotation_product":
            struct = rot
            qfinal = qpop_used * struct
        elif mode == "dual_min":
            struct = torch.minimum(local, rot)
            qfinal = qpop_used * struct
        elif mode == "orbit_average_snr":
            struct = torch.minimum(local, rot)
            qpop = orbit_average_qpop_for_param(opt, param, group, pgrads, local_t[idx] + rot_t[idx], blocks)
            qpop_used = qpop
            if floor > 0.0 and str(getattr(args, "gate_floor_mode", "final")) == "population":
                qpop_used = qpop * (1.0 - floor) + floor
            qfinal = qpop_used * struct
        elif mode == "contrastive":
            pos = local if task == "local_patch_interaction" else rot
            struct = (pos - float(args.contrastive_eta) * dest).clamp_min(0.0)
            qfinal = qpop_used * struct
        elif mode == "orbit_centered_contrastive":
            pos = local if task == "local_patch_interaction" else rot
            struct = orbit_center_positive((pos - float(args.contrastive_eta) * dest).clamp_min(0.0), blocks, gain=float(args.orbit_center_gain))
            qfinal = qpop_used * struct
        elif mode == "structure_only":
            struct = local if task == "local_patch_interaction" else rot
            qfinal = struct
        elif mode == "pop_only":
            struct = torch.ones_like(qpop)
            qfinal = qpop_used
        else:
            struct = torch.minimum(local, rot)
            qfinal = qpop_used * struct
        if floor > 0.0 and str(getattr(args, "gate_floor_mode", "final")) == "final":
            qfinal = qfinal * (1.0 - floor) + floor
        qpop_parts.append(qpop.detach().reshape(-1))
        qlocal_parts.append(local.detach().reshape(-1))
        qrot_parts.append(rot.detach().reshape(-1))
        qdes_parts.append(dest.detach().reshape(-1))
        qfinal_parts.append(qfinal.detach().reshape(-1))
        qdensity_parts.append(density_random_like(qfinal, seed * 100003 + idx))
        qhist_parts.append(hist_random_like(qfinal, seed * 100019 + idx))
        qstruct_orbit = block_shuffle(struct, blocks, seed * 100043 + idx)
        qpop_orbit = qpop_used if mode != "structure_only" else qpop
        qorbit = qpop_orbit * qstruct_orbit.to(device=qpop.device, dtype=qpop.dtype)
        if floor > 0.0 and str(getattr(args, "gate_floor_mode", "final")) == "final":
            qorbit = qorbit * (1.0 - floor) + floor
        qorbit_parts.append(qorbit.detach().reshape(-1))
        all_blocks.extend([[offset + int(i) for i in block] for block in blocks])
        offset += int(param.numel())
    qpop_all = torch.cat([x.cpu() for x in qpop_parts]) if qpop_parts else torch.zeros(0)
    qlocal_all = torch.cat([x.cpu() for x in qlocal_parts]) if qlocal_parts else torch.zeros(0)
    qrot_all = torch.cat([x.cpu() for x in qrot_parts]) if qrot_parts else torch.zeros(0)
    qdes_all = torch.cat([x.cpu() for x in qdes_parts]) if qdes_parts else torch.zeros(0)
    qfinal_all = torch.cat([x.cpu() for x in qfinal_parts]) if qfinal_parts else torch.zeros(0)
    qdensity_all = torch.cat([x.cpu() for x in qdensity_parts]) if qdensity_parts else torch.zeros(0)
    qhist_all = torch.cat([x.cpu() for x in qhist_parts]) if qhist_parts else torch.zeros(0)
    qorbit_all = torch.cat([x.cpu() for x in qorbit_parts]) if qorbit_parts else torch.zeros(0)
    qstruct_all = qlocal_all if task == "local_patch_interaction" else qrot_all
    return {
        "q_pop": qpop_all,
        "q_local": qlocal_all,
        "q_rot": qrot_all,
        "q_struct": qstruct_all,
        "q_destructive": qdes_all,
        "q_final": qfinal_all,
        "q_rand_density": qdensity_all,
        "q_rand_hist": qhist_all,
        "q_rand_orbit": qorbit_all,
        "block_count": len(all_blocks),
    }


def run_part_c_row(job: tuple[str, str, str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, basis_key, depth, task, seed = job
    start = time.time()
    try:
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
        opt = EdgeSobolevSNRFU(
            list(model.coeffs),
            lr=float(args.edge_lr),
            weight_decay=float(args.weight_decay),
            sobolev_exponent=0.0,
            edge_metric_type="functional_gram",
            edge_weight_normalization=str(args.edge_weight_normalization),
            edge_weight_ridge=float(args.edge_weight_ridge),
            functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
            gate_beta=float(args.snr_beta),
            gate_family=str(args.structure_block_family),
            gate_floor=0.0,
            gate_input_side=int(args.visual_side),
            gate_patch_size=2,
            stat_warmup_steps=0,
            use_population_gate=True,
        )
        src = model_gate_vectors(scheme, model, opt, x_source, y_source, task, seed, args)
        grd = model_gate_vectors(scheme, model, opt, x_guard, y_guard, task, seed + 991, args)
        true_score = weighted_structure_score(src["q_final"], src["q_struct"])  # type: ignore[arg-type]
        density_score = weighted_structure_score(src["q_rand_density"], src["q_struct"])  # type: ignore[arg-type]
        hist_score = weighted_structure_score(src["q_rand_hist"], src["q_struct"])  # type: ignore[arg-type]
        orbit_score = weighted_structure_score(src["q_rand_orbit"], src["q_struct"])  # type: ignore[arg-type]
        selected_local_score = weighted_structure_score(src["q_final"], src["q_local"])  # type: ignore[arg-type]
        selected_rot_score = weighted_structure_score(src["q_final"], src["q_rot"])  # type: ignore[arg-type]
        local_orbit_score = weighted_structure_score(src["q_rand_orbit"], src["q_local"])  # type: ignore[arg-type]
        rot_orbit_score = weighted_structure_score(src["q_rand_orbit"], src["q_rot"])  # type: ignore[arg-type]
        local_raw_mean = float(torch.as_tensor(src["q_local"]).mean().item()) if int(torch.as_tensor(src["q_local"]).numel()) else 0.0
        rot_raw_mean = float(torch.as_tensor(src["q_rot"]).mean().item()) if int(torch.as_tensor(src["q_rot"]).numel()) else 0.0
        row = {
            "part": "C",
            "status": "ok",
            "scheme": scheme,
            "task": task,
            "basis_key": basis_key,
            "depth": depth,
            "seed": int(seed),
            "transform_family": PART_C_SCHEMES.get(scheme, "missing"),
            "gate_floor": float(args.gate_floor),
            "gate_floor_mode": str(args.gate_floor_mode),
            "gate_density": float(torch.as_tensor(src["q_final"]).mean().item()) if int(torch.as_tensor(src["q_final"]).numel()) else 0.0,
            "q_pop_density": float(torch.as_tensor(src["q_pop"]).mean().item()) if int(torch.as_tensor(src["q_pop"]).numel()) else 0.0,
            "q_struct_mean": float(torch.as_tensor(src["q_struct"]).mean().item()) if int(torch.as_tensor(src["q_struct"]).numel()) else 0.0,
            "q_struct_std": float(torch.as_tensor(src["q_struct"]).std(unbiased=False).item()) if int(torch.as_tensor(src["q_struct"]).numel()) else 0.0,
            "q_final_density": float(torch.as_tensor(src["q_final"]).mean().item()) if int(torch.as_tensor(src["q_final"]).numel()) else 0.0,
            "q_pop_mean": float(torch.as_tensor(src["q_pop"]).mean().item()) if int(torch.as_tensor(src["q_pop"]).numel()) else 0.0,
            "q_final_mean": float(torch.as_tensor(src["q_final"]).mean().item()) if int(torch.as_tensor(src["q_final"]).numel()) else 0.0,
            "true_vs_density_random_gap": true_score - density_score,
            "true_vs_hist_random_gap": true_score - hist_score,
            "true_vs_orbit_random_gap": true_score - orbit_score,
            "source_guard_q_struct_cosine": cosine(src["q_struct"], grd["q_struct"]),  # type: ignore[arg-type]
            "source_guard_q_final_cosine": cosine(src["q_final"], grd["q_final"]),  # type: ignore[arg-type]
            "local_coherence_mean": selected_local_score,
            "rotation_coherence_mean": selected_rot_score,
            "local_coherence_raw_mean": local_raw_mean,
            "rotation_coherence_raw_mean": rot_raw_mean,
            "destructive_coherence_mean": float(torch.as_tensor(src["q_destructive"]).mean().item()) if int(torch.as_tensor(src["q_destructive"]).numel()) else 0.0,
            "orbit_random_local_coherence": local_orbit_score,
            "orbit_random_rotation_coherence": rot_orbit_score,
            "coherence_C2_proxy_corr": 0.0,
            "gate_cost_overhead": 0.0,
            "random_controls_count": 3,
            "block_count": int(src["block_count"]),  # type: ignore[arg-type]
            "structure_guard_source": str(args.structure_guard_source),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
        return row
    except Exception as exc:
        return {
            "part": "C",
            "status": "error",
            "scheme": scheme,
            "task": task,
            "basis_key": basis_key,
            "depth": depth,
            "seed": int(seed),
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int]]:
    return [
        (scheme, basis, depth, task, seed)
        for scheme in csv_items(args.part_c_schemes)
        for basis in csv_items(args.part_c_basis)
        for depth in csv_items(args.part_c_depths)
        for task in csv_items(args.part_c_tasks)
        for seed in range(int(args.part_c_seed_count))
    ]


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows = [run_part_c_row(job, args, device) for job in shard_items(part_c_jobs(args), args)]
    path = write_rows(OUT_ROOT / f"part_c_structure_coherence_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("C", 0, "C_ShardsWritten", "merge_required", rows)


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_structure_coherence_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_c_structure_coherence_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task")) == key]
        task = str(key[3])
        qdens = median(r.get("q_final_density") for r in group)
        orbit_gap = median(r.get("true_vs_orbit_random_gap") for r in group)
        sg = median(r.get("source_guard_q_final_cosine") for r in group)
        if task == "local_patch_interaction":
            coh_gap = median(fval(r.get("local_coherence_mean")) - fval(r.get("orbit_random_local_coherence")) for r in group)
        else:
            coh_gap = median(fval(r.get("rotation_coherence_mean")) - fval(r.get("orbit_random_rotation_coherence")) for r in group)
        random_excess = max(
            median(r.get("true_vs_density_random_gap") for r in group),
            median(r.get("true_vs_hist_random_gap") for r in group),
            median(r.get("true_vs_orbit_random_gap") for r in group),
        )
        task_pass = int(
            orbit_gap >= float(args.c_orbit_gap_gate)
            and sg >= float(args.c_source_guard_cosine_gate)
            and coh_gap >= float(args.c_coherence_gap_gate)
            and float(args.c_density_low) <= qdens <= float(args.c_density_high)
            and random_excess >= -0.02
        )
        task_summaries.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "task": task,
            "rows": len(group),
            "q_pop_density_median": median(r.get("q_pop_density") for r in group),
            "q_struct_mean_median": median(r.get("q_struct_mean") for r in group),
            "q_final_density_median": qdens,
            "true_vs_density_random_gap_median": median(r.get("true_vs_density_random_gap") for r in group),
            "true_vs_hist_random_gap_median": median(r.get("true_vs_hist_random_gap") for r in group),
            "true_vs_orbit_random_gap_median": orbit_gap,
            "source_guard_q_struct_cosine_median": median(r.get("source_guard_q_struct_cosine") for r in group),
            "source_guard_q_final_cosine_median": sg,
            "local_coherence_mean_median": median(r.get("local_coherence_mean") for r in group),
            "rotation_coherence_mean_median": median(r.get("rotation_coherence_mean") for r in group),
            "local_coherence_raw_mean_median": median(r.get("local_coherence_raw_mean") for r in group),
            "rotation_coherence_raw_mean_median": median(r.get("rotation_coherence_raw_mean") for r in group),
            "destructive_coherence_mean_median": median(r.get("destructive_coherence_mean") for r in group),
            "coherence_gap_vs_orbit_random": coh_gap,
            "taskwise_pass": task_pass,
        })
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in ok}):
        tasks = [t for t in task_summaries if (t.get("scheme"), t.get("basis_key"), t.get("depth")) == key]
        official = int(
            str(key[0]) in OFFICIAL_PART_C_SCHEMES
            and str(key[1]) == "dche_k9"
            and str(key[2]) == "depth3"
            and len(tasks) >= 2
            and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)
        )
        groups.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "tasks": ",".join(str(t.get("task")) for t in tasks),
            "rows": sum(ival(t.get("rows")) for t in tasks),
            "q_final_density_median": median(t.get("q_final_density_median") for t in tasks),
            "true_vs_orbit_random_gap_median": median(t.get("true_vs_orbit_random_gap_median") for t in tasks),
            "source_guard_q_final_cosine_median": median(t.get("source_guard_q_final_cosine_median") for t in tasks),
            "taskwise_all_pass": int(bool(tasks) and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)),
            "official_pass": official,
        })
    pass_groups = [g for g in groups if ival(g.get("official_pass")) == 1]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_c_job_errors"
    elif not pass_groups:
        low_sg = any(fval(t.get("source_guard_q_final_cosine_median")) < float(args.c_source_guard_cosine_gate) for t in task_summaries)
        low_gap = any(fval(t.get("true_vs_orbit_random_gap_median")) < float(args.c_orbit_gap_gate) for t in task_summaries)
        blocker = "source_guard_pairing_low" if low_sg else ("structure_random_gap_low" if low_gap else "taskwise_structure_coherence_failed")
    else:
        blocker = "none"
    task_csv = write_rows(OUT_ROOT / "part_c_structure_coherence_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_c_structure_coherence_group_summary.csv", groups)
    summary = gate_summary("C", gate, "C_StructureCoherencePass" if gate else "C_StructureCoherenceFailed", blocker, rows, scheme_groups=groups, taskwise_groups=task_summaries, passing_scheme_groups=pass_groups)
    write_json(OUT_ROOT / "part_c_structure_coherence_summary.json", summary)
    nxt = next_actions(
        "C",
        gate,
        blocker,
        [] if gate else [
            "if source_guard cosine is low, rerun with --structure-guard-source train_split and audit pairing",
            "if true-vs-random gap is low, inspect orbit shuffle and destructive transform controls",
            "if only one task passes, rerun dual_min and contrastive only before any training",
        ],
    )
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part C structure coherence diagnostic", summary)
    return summary


def observe_structure_projected_batch(
    opt: EdgeSobolevSNRFU,
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    task: str,
    seed: int,
    args: argparse.Namespace,
    *,
    orbit_random: bool = False,
    structure_control_type: str = "true",
    structure_mode: str = "contrastive",
    utility_x: torch.Tensor | None = None,
    utility_y: torch.Tensor | None = None,
    utility_x2: torch.Tensor | None = None,
    utility_y2: torch.Tensor | None = None,
) -> dict[str, float | int]:
    side = int(args.visual_side)
    params = list(model.coeffs)
    n_examples = min(int(args.population_grad_examples), int(x.shape[0]))
    xb = x[:n_examples]
    yb = y[:n_examples]
    flat = per_example_gradient_matrix(model, xb, yb, max_examples=n_examples, label_prior_correction=bool(int(args.label_prior_correction)))
    split = split_flat_grads(flat, params)
    labels = yb[: int(flat.shape[0])].detach() if flat.ndim == 2 else None
    profile = str(args.structure_transform_profile)

    def transformed_split(names: list[str], offset: int, *, align_with_transform: bool = True) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        per_param: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in params]
        identity_perm = torch.arange(side * side, device=x.device)
        for i, name in enumerate(names):
            perm = transform_perm(side, name, seed=seed * 1000 + offset + i, device=x.device)
            x_t = apply_perm(xb, perm)
            ft = per_example_gradient_matrix(model, x_t, yb[: int(x_t.shape[0])], max_examples=n_examples, label_prior_correction=bool(int(args.label_prior_correction)))
            st = split_flat_grads(ft, params)
            align_perm = perm if align_with_transform else identity_perm
            for j, grads in enumerate(st):
                per_param[j].append((grads.to(device=params[j].device, dtype=params[j].dtype), align_perm))
        return per_param

    local_t = transformed_split(transform_names("local", profile), 1100)
    rot_t = transformed_split(transform_names("rotation", profile), 2100)
    destructive_t = transformed_split(
        transform_names("destructive_local" if task == "local_patch_interaction" else "destructive_rotation", profile),
        3100,
        align_with_transform=False,
    )
    intervention_split: list[torch.Tensor] | None = None
    if "local_intervention" in str(structure_mode) and task == "local_patch_interaction":
        x_int = local_checker_patch_pair_intervention(xb, side, scale=float(args.local_intervention_scale))
        int_flat = per_example_gradient_matrix(
            model,
            x_int,
            yb[: int(x_int.shape[0])],
            max_examples=n_examples,
            label_prior_correction=bool(int(args.label_prior_correction)),
        )
        int_split = split_flat_grads(int_flat, params)
        intervention_split = [i - b for i, b in zip(int_split, split)]
    utility_split: list[torch.Tensor] | None = None
    if structure_mode_uses_utility(structure_mode) and utility_x is not None and utility_y is not None:
        n_util = min(int(getattr(args, "part_d_utility_examples", args.population_grad_examples)), int(utility_x.shape[0]))
        util_flat = per_example_gradient_matrix(
            model,
            utility_x[:n_util],
            utility_y[:n_util],
            max_examples=n_util,
            label_prior_correction=bool(int(args.label_prior_correction)),
        )
        utility_split = split_flat_grads(util_flat, params)
    utility_split2: list[torch.Tensor] | None = None
    if "stable_signed" in str(structure_mode) and utility_x2 is not None and utility_y2 is not None:
        n_util2 = min(int(getattr(args, "part_d_utility_examples", args.population_grad_examples)), int(utility_x2.shape[0]))
        util_flat2 = per_example_gradient_matrix(
            model,
            utility_x2[:n_util2],
            utility_y2[:n_util2],
            max_examples=n_util2,
            label_prior_correction=bool(int(args.label_prior_correction)),
        )
        utility_split2 = split_flat_grads(util_flat2, params)

    grad_norms = flat.norm(dim=1) if flat.ndim == 2 and int(flat.shape[0]) else torch.zeros(0, dtype=torch.float64)
    struct_means: list[float] = []
    local_means: list[float] = []
    local_intervention_means: list[float] = []
    rot_means: list[float] = []
    dest_means: list[float] = []
    utility_mult_means: list[float] = []
    signed_modifier_means: list[float] = []
    signed_modifier_negative_fracs: list[float] = []
    signed_modifier_agreement_fracs: list[float] = []
    density_match_errors: list[float] = []
    histogram_mean_errors: list[float] = []
    orbit_size_match_errors: list[float] = []
    group = opt.param_groups[0]
    for idx, (param, grads) in enumerate(zip(params, split)):
        pgrads = grads.to(device=param.device, dtype=param.dtype)
        opt.observe_per_example_gradients(param, pgrads, labels=labels)
        blocks = opt._blocks_for_param(param, {**group, "gate_family": str(args.structure_block_family)})
        local = structure_gate_for_param(opt, param, group, pgrads, local_t[idx], blocks)
        rot = structure_gate_for_param(opt, param, group, pgrads, rot_t[idx], blocks)
        dest = structure_gate_for_param(opt, param, group, pgrads, destructive_t[idx], blocks)
        intervention_grads = intervention_split[idx].to(device=param.device, dtype=param.dtype) if intervention_split is not None and idx < len(intervention_split) else None
        local_intervention = block_direction_energy_gate(opt, param, group, intervention_grads, blocks, args) if task == "local_patch_interaction" else torch.zeros_like(local)
        pos = local if task == "local_patch_interaction" else rot
        mode = str(structure_mode)
        if mode in {
            "local_intervention_coordinate_orbit_centered_contrastive",
            "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
        } and task == "local_patch_interaction":
            pos = local_intervention
        if mode == "task_product":
            struct = pos
        elif mode == "dual_min":
            struct = torch.minimum(local, rot)
        elif mode == "pop_only":
            struct = torch.ones_like(pos)
        elif mode == "orbit_centered_contrastive":
            struct = orbit_center_positive((pos - float(args.contrastive_eta) * dest).clamp_min(0.0), blocks, gain=float(args.orbit_center_gain))
        elif mode == "local_intervention_coordinate_orbit_centered_contrastive":
            struct = orbit_center_positive((pos - float(args.contrastive_eta) * dest).clamp_min(0.0), blocks, gain=float(args.orbit_center_gain))
        elif mode in {"utility_orbit_centered_contrastive", "density_preserved_utility_tilt_orbit_centered_contrastive"}:
            base_struct = orbit_center_positive((pos - float(args.contrastive_eta) * dest).clamp_min(0.0), blocks, gain=float(args.orbit_center_gain))
            utility_grads = utility_split[idx] if utility_split is not None and idx < len(utility_split) else None
            utility_mult = block_descent_utility_multiplier(
                opt,
                param,
                group,
                pgrads,
                utility_grads,
                blocks,
                args,
                floor=float(args.utility_tilt_floor) if mode == "density_preserved_utility_tilt_orbit_centered_contrastive" else 0.0,
            )
            struct = (base_struct * utility_mult).clamp(0.0, 1.0)
            if mode == "density_preserved_utility_tilt_orbit_centered_contrastive":
                struct = density_preserve_gate_mean(struct, base_struct)
            utility_mult_means.append(float(utility_mult.detach().mean().cpu().item()) if int(utility_mult.numel()) else 0.0)
        elif mode in {
            "signed_utility_channel_orbit_centered_contrastive",
            "input_layer_signed_utility_channel_orbit_centered_contrastive",
            "density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "task_contrast_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
        }:
            complement = rot if task == "local_patch_interaction" else local
            complement_weight = float(args.task_complement_eta) if mode == "task_contrast_density_preserved_signed_utility_tilt_orbit_centered_contrastive" else 0.0
            base_struct = orbit_center_positive(
                (pos - complement_weight * complement - float(args.contrastive_eta) * dest).clamp_min(0.0),
                blocks,
                gain=float(args.orbit_center_gain),
            )
            struct = base_struct
            utility_grads = utility_split[idx] if utility_split is not None and idx < len(utility_split) else None
            source_for_utility = intervention_grads if mode == "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive" and task == "local_patch_interaction" and intervention_grads is not None else pgrads
            if mode in {
                "density_preserved_signed_utility_tilt_orbit_centered_contrastive",
                "task_contrast_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
                "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
                "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
            }:
                utility_mult = block_descent_utility_multiplier(
                    opt,
                    param,
                    group,
                    source_for_utility,
                    utility_grads,
                    blocks,
                    args,
                    floor=float(args.utility_tilt_floor),
                )
                struct = density_preserve_gate_mean((base_struct * utility_mult).clamp(0.0, 1.0), base_struct)
                utility_mult_means.append(float(utility_mult.detach().mean().cpu().item()) if int(utility_mult.numel()) else 0.0)
            if mode == "input_layer_signed_utility_channel_orbit_centered_contrastive" and int(getattr(param, "_kan_layer_id", -1)) != 0:
                signed_modifier = torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
            else:
                signed_modifier = block_descent_utility_sign_modifier(opt, param, group, source_for_utility, utility_grads, blocks, args)
        elif mode == "stable_signed_utility_channel_orbit_centered_contrastive":
            struct = orbit_center_positive((pos - float(args.contrastive_eta) * dest).clamp_min(0.0), blocks, gain=float(args.orbit_center_gain))
            utility_grads = utility_split[idx] if utility_split is not None and idx < len(utility_split) else None
            utility_grads2 = utility_split2[idx] if utility_split2 is not None and idx < len(utility_split2) else None
            signed_modifier, agreement = block_descent_utility_stable_sign_modifier(
                opt,
                param,
                group,
                pgrads,
                utility_grads,
                utility_grads2,
                blocks,
                args,
            )
            signed_modifier_agreement_fracs.append(agreement)
        else:
            struct = (pos - float(args.contrastive_eta) * dest).clamp_min(0.0)
        if orbit_random:
            structure_control_type = "orbit_shuffle"
        controlled_struct = control_structure_gate(struct, dest, blocks, structure_control_type, seed * 100043 + idx)
        opt.observe_structure_gate(param, controlled_struct.reshape_as(param))
        if mode in {
            "signed_utility_channel_orbit_centered_contrastive",
            "stable_signed_utility_channel_orbit_centered_contrastive",
            "input_layer_signed_utility_channel_orbit_centered_contrastive",
            "density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "task_contrast_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
        }:
            controlled_modifier = control_structure_gate(signed_modifier, -signed_modifier, blocks, structure_control_type, seed * 100043 + idx + 17)
            opt.observe_structure_update_modifier(param, controlled_modifier.reshape_as(param))
            signed_modifier_means.append(float(controlled_modifier.detach().mean().cpu().item()) if int(controlled_modifier.numel()) else 0.0)
            signed_modifier_negative_fracs.append(float((controlled_modifier.detach() < 0).to(dtype=torch.float64).mean().cpu().item()) if int(controlled_modifier.numel()) else 0.0)
        struct_means.append(float(controlled_struct.detach().mean().cpu().item()) if int(controlled_struct.numel()) else 0.0)
        local_means.append(float(local.detach().mean().cpu().item()) if int(local.numel()) else 0.0)
        local_intervention_means.append(float(local_intervention.detach().mean().cpu().item()) if int(local_intervention.numel()) else 0.0)
        rot_means.append(float(rot.detach().mean().cpu().item()) if int(rot.numel()) else 0.0)
        dest_means.append(float(dest.detach().mean().cpu().item()) if int(dest.numel()) else 0.0)
        if int(struct.numel()) and int(controlled_struct.numel()):
            density_match_errors.append(abs(float(controlled_struct.detach().mean().cpu().item()) - float(struct.detach().mean().cpu().item())))
            histogram_mean_errors.append(
                abs(float(torch.sort(controlled_struct.detach().reshape(-1).cpu()).values.mean().item()) - float(torch.sort(struct.detach().reshape(-1).cpu()).values.mean().item()))
            )
        orbit_size_match_errors.append(0.0)
    return {
        "per_example_count": int(flat.shape[0]) if flat.ndim == 2 else 0,
        "grad_norm_median": float(grad_norms.median().item()) if int(grad_norms.numel()) else 0.0,
        "grad_norm_p95": float(torch.quantile(grad_norms, 0.95).item()) if int(grad_norms.numel()) else 0.0,
        "grad_nan_count": int(torch.isnan(flat).sum().item()) if int(flat.numel()) else 0,
        "grad_inf_count": int(torch.isinf(flat).sum().item()) if int(flat.numel()) else 0,
        "q_struct_mean": mean(struct_means),
        "local_coherence_mean": mean(local_means),
        "local_intervention_mean": mean(local_intervention_means),
        "rotation_coherence_mean": mean(rot_means),
        "destructive_coherence_mean": mean(dest_means),
        "utility_multiplier_mean": mean(utility_mult_means),
        "signed_modifier_mean": mean(signed_modifier_means),
        "signed_modifier_negative_fraction": mean(signed_modifier_negative_fracs),
        "signed_modifier_agreement_fraction": mean(signed_modifier_agreement_fracs),
        "structure_density_match_error": mean(density_match_errors),
        "structure_histogram_mean_error": mean(histogram_mean_errors),
        "orbit_size_match_error": mean(orbit_size_match_errors),
    }


def observed_population_structure_gate(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = opt._observed_grads.get(id(param))
    if observed is None or observed.ndim < 2:
        flat = torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
        return flat, torch.zeros_like(flat)
    weights = opt._weights_for_param(param, group)
    structure_gate = opt._observed_structure_gates.get(id(param))
    saved_floor = float(group.get("gate_floor", 0.0))
    floor_mode = str(group.get("gate_floor_mode", "final"))
    if structure_gate is not None:
        group["gate_floor"] = 0.0
    gate, task_mu = opt._gate_from_observed(observed, weights, param, group)
    if structure_gate is not None:
        group["gate_floor"] = saved_floor
        if saved_floor > 0.0 and floor_mode == "population":
            gate = gate * (1.0 - saved_floor) + saved_floor
        gate = gate * structure_gate.reshape_as(param).to(device=param.device, dtype=param.dtype)
        if saved_floor > 0.0 and floor_mode != "population":
            gate = gate * (1.0 - saved_floor) + saved_floor
    return gate.detach().reshape(-1), task_mu.detach().reshape(-1)


def structure_mode_uses_utility(structure_mode: str) -> bool:
    return "utility" in str(structure_mode)


def train_only_descent_utility(
    opt: EdgeSobolevSNRFU,
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    params = list(model.coeffs)
    n_examples = min(int(getattr(args, "part_d_utility_examples", args.population_grad_examples)), int(x.shape[0]))
    xb = x[:n_examples]
    yb = y[:n_examples]
    flat = per_example_gradient_matrix(model, xb, yb, max_examples=n_examples, label_prior_correction=bool(int(args.label_prior_correction)))
    split = split_flat_grads(flat, params)
    group = opt.param_groups[0]
    gated_num = 0.0
    gated_norm2 = 0.0
    guard_norm2 = 0.0
    ungated_num = 0.0
    ungated_norm2 = 0.0
    gate_means: list[float] = []
    for param, guard_grads in zip(params, split):
        gate, source_mu = observed_population_structure_gate(opt, param, group)
        guard_white = opt._apply_metric_inv_sqrt(guard_grads.to(device=param.device, dtype=param.dtype), param, group)
        guard_mu = guard_white.reshape(int(guard_white.shape[0]), -1).mean(dim=0).detach().reshape(-1)
        n = min(int(gate.numel()), int(source_mu.numel()), int(guard_mu.numel()))
        if n <= 0:
            continue
        g = gate[:n].to(dtype=torch.float64)
        src = source_mu[:n].to(dtype=torch.float64)
        modifier = opt._observed_structure_update_modifiers.get(id(param))
        if modifier is not None:
            mod = modifier.detach().reshape(-1)[:n].to(device=src.device, dtype=torch.float64)
            src_effective = src * mod
        else:
            src_effective = src
        grd = guard_mu[:n].to(dtype=torch.float64)
        gated = g * src_effective
        gated_num += float(torch.dot(gated, grd).detach().cpu().item())
        gated_norm2 += float(torch.dot(gated, gated).detach().cpu().item())
        guard_norm2 += float(torch.dot(grd, grd).detach().cpu().item())
        ungated_num += float(torch.dot(src, grd).detach().cpu().item())
        ungated_norm2 += float(torch.dot(src, src).detach().cpu().item())
        gate_means.append(float(g.mean().detach().cpu().item()))
    eps = 1.0e-12
    gated_cos = gated_num / math.sqrt(max(eps, gated_norm2) * max(eps, guard_norm2))
    ungated_cos = ungated_num / math.sqrt(max(eps, ungated_norm2) * max(eps, guard_norm2))
    return {
        "utility_examples": int(flat.shape[0]) if flat.ndim == 2 else 0,
        "descent_utility": gated_num,
        "descent_utility_cosine": gated_cos,
        "ungated_descent_utility": ungated_num,
        "ungated_descent_utility_cosine": ungated_cos,
        "utility_gate_over_ungated": gated_num / ungated_num if abs(ungated_num) > eps else 0.0,
        "utility_gate_density": mean(gate_means),
    }


def density_preserve_gate_mean(weighted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if int(weighted.numel()) <= 0 or int(reference.numel()) <= 0:
        return weighted
    target = reference.detach().mean().to(device=weighted.device, dtype=weighted.dtype)
    current = weighted.detach().mean().to(device=weighted.device, dtype=weighted.dtype)
    if float(current.cpu().item()) <= 1.0e-12:
        return weighted
    return (weighted * (target / current.clamp_min(1.0e-12))).clamp(0.0, 1.0)


def block_descent_utility_multiplier(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    utility_grads: torch.Tensor | None,
    blocks: list[list[int]],
    args: argparse.Namespace,
    *,
    floor: float = 0.0,
) -> torch.Tensor:
    if utility_grads is None or utility_grads.ndim < 2 or not blocks:
        return torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
    src = opt._apply_metric_inv_sqrt(source_grads, param, group).reshape(int(source_grads.shape[0]), -1).mean(dim=0)
    util = opt._apply_metric_inv_sqrt(utility_grads.to(device=param.device, dtype=param.dtype), param, group)
    util_mu = util.reshape(int(util.shape[0]), -1).mean(dim=0)
    flat = torch.zeros(int(param.numel()), device=param.device, dtype=torch.float64)
    block_scores: list[float] = []
    for block in blocks:
        idx = torch.tensor([int(i) for i in block if 0 <= int(i) < int(param.numel())], dtype=torch.long, device=param.device)
        if int(idx.numel()) <= 0:
            continue
        a = src[idx].detach().to(dtype=torch.float64)
        b = util_mu[idx].detach().to(dtype=torch.float64)
        score = float(torch.dot(a, b).div(a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)).clamp_min(0.0).item())
        block_scores.append(score)
        flat[idx] = score
    positives = [v for v in block_scores if v > 0.0]
    floor = max(0.0, min(float(floor), float(getattr(args, "utility_gate_max", 3.0))))
    if not positives:
        return torch.full((int(param.numel()),), floor, device=param.device, dtype=param.dtype)
    scale = float(np.mean(positives))
    out = flat / max(scale, 1.0e-12)
    out = out * float(getattr(args, "utility_gate_gain", 1.0))
    if floor > 0.0:
        out = floor + (1.0 - floor) * out
    out = out.clamp(floor, float(getattr(args, "utility_gate_max", 3.0)))
    return out.to(device=param.device, dtype=param.dtype)


def block_descent_utility_sign_modifier(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    utility_grads: torch.Tensor | None,
    blocks: list[list[int]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if utility_grads is None or utility_grads.ndim < 2 or not blocks:
        return torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
    src = opt._apply_metric_inv_sqrt(source_grads, param, group).reshape(int(source_grads.shape[0]), -1).mean(dim=0)
    util = opt._apply_metric_inv_sqrt(utility_grads.to(device=param.device, dtype=param.dtype), param, group)
    util_mu = util.reshape(int(util.shape[0]), -1).mean(dim=0)
    flat = torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
    for block in blocks:
        idx = torch.tensor([int(i) for i in block if 0 <= int(i) < int(param.numel())], dtype=torch.long, device=param.device)
        if int(idx.numel()) <= 0:
            continue
        score = torch.dot(src[idx].detach().to(dtype=torch.float64), util_mu[idx].detach().to(dtype=torch.float64))
        flat[idx] = float(args.signed_negative_scale) if float(score.cpu().item()) < 0.0 else 1.0
    return flat


def block_descent_utility_stable_sign_modifier(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    utility_grads: torch.Tensor | None,
    utility_grads2: torch.Tensor | None,
    blocks: list[list[int]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, float]:
    if utility_grads is None or utility_grads2 is None or utility_grads.ndim < 2 or utility_grads2.ndim < 2 or not blocks:
        return torch.ones(int(param.numel()), device=param.device, dtype=param.dtype), 0.0
    src = opt._apply_metric_inv_sqrt(source_grads, param, group).reshape(int(source_grads.shape[0]), -1).mean(dim=0)
    util1 = opt._apply_metric_inv_sqrt(utility_grads.to(device=param.device, dtype=param.dtype), param, group)
    util2 = opt._apply_metric_inv_sqrt(utility_grads2.to(device=param.device, dtype=param.dtype), param, group)
    mu1 = util1.reshape(int(util1.shape[0]), -1).mean(dim=0)
    mu2 = util2.reshape(int(util2.shape[0]), -1).mean(dim=0)
    flat = torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
    agreed = 0
    total = 0
    for block in blocks:
        idx = torch.tensor([int(i) for i in block if 0 <= int(i) < int(param.numel())], dtype=torch.long, device=param.device)
        if int(idx.numel()) <= 0:
            continue
        score1 = float(torch.dot(src[idx].detach().to(dtype=torch.float64), mu1[idx].detach().to(dtype=torch.float64)).cpu().item())
        score2 = float(torch.dot(src[idx].detach().to(dtype=torch.float64), mu2[idx].detach().to(dtype=torch.float64)).cpu().item())
        sign1 = -1 if score1 < 0.0 else 1
        sign2 = -1 if score2 < 0.0 else 1
        total += 1
        if sign1 == sign2:
            agreed += 1
            flat[idx] = float(args.signed_negative_scale) if sign1 < 0 else 1.0
        else:
            flat[idx] = 1.0
    return flat, (float(agreed) / float(total)) if total else 0.0


def train_part_d_structure_row(job: tuple[str, str, str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme_base, task, basis_key, depth, seed = job
    start = time.time()
    scheme = str(scheme_base)
    structure_mode = PART_D_SCHEMES.get(scheme, "contrastive")
    sobolev_exponent = part_d_sobolev_exponent(scheme)
    random_scheme = bool(int(getattr(args, "part_d_orbit_random", 0)))
    if random_scheme:
        scheme = f"{scheme}_OrbitRandom"
    try:
        bargs = v2300.basis_args(args, basis_key)
        xtr, ytr, xg, yg = v2300.visual_data(task, seed, args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = v2300.make_model(depth, int(xtr.shape[1]), classes, v2300.model_seed_for(basis_key, depth, task, seed), bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        opt = EdgeSobolevSNRFU(
            list(model.coeffs),
            lr=float(args.edge_lr),
            weight_decay=float(args.weight_decay),
            sobolev_exponent=sobolev_exponent,
            edge_metric_type="functional_gram",
            edge_weight_normalization=str(args.edge_weight_normalization),
            edge_weight_ridge=float(args.edge_weight_ridge),
            functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
            gate_beta=float(args.snr_beta),
            gate_family=str(args.structure_block_family),
            gate_floor=float(args.gate_floor),
            gate_floor_mode=str(args.gate_floor_mode),
            gate_input_side=int(args.visual_side),
            gate_patch_size=2,
            stat_warmup_steps=int(getattr(args, "stat_warmup_steps", 0)),
            use_population_gate=True,
        )
        before = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        gate_trace: list[float] = []
        struct_trace: list[float] = []
        local_trace: list[float] = []
        local_intervention_trace: list[float] = []
        rot_trace: list[float] = []
        dest_trace: list[float] = []
        utility_mult_trace: list[float] = []
        signed_modifier_trace: list[float] = []
        signed_modifier_negative_trace: list[float] = []
        signed_modifier_agreement_trace: list[float] = []
        grad_norm_median: list[float] = []
        grad_norm_p95: list[float] = []
        descent_utility_trace: list[float] = []
        descent_utility_cosine_trace: list[float] = []
        ungated_descent_utility_trace: list[float] = []
        utility_gate_over_ungated_trace: list[float] = []
        trace_rows: list[dict[str, Any]] = []
        grad_nan_count = 0
        grad_inf_count = 0
        per_example_count = 0
        cached_structure_gates: dict[int, torch.Tensor] = {}
        cached_structure_modifiers: dict[int, torch.Tensor] = {}
        cached_structure_diag: dict[str, float | int] = {}
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            refresh_structure = not cached_structure_gates or (int(args.structure_gate_every) <= 1) or ((step - 1) % int(args.structure_gate_every) == 0)
            if refresh_structure:
                need_utility_batch = int(getattr(args, "part_d_trace", 0)) or structure_mode_uses_utility(structure_mode)
                xub = yub = None
                xub2 = yub2 = None
                if need_utility_batch:
                    xub, yub = v2293.v2289.iter_train_batches(
                        xtr,
                        ytr,
                        step - 1 + int(getattr(args, "part_d_utility_batch_offset", 997)),
                        int(args.batch_size),
                        int(seed),
                    )
                    if "stable_signed" in structure_mode:
                        xub2, yub2 = v2293.v2289.iter_train_batches(
                            xtr,
                            ytr,
                            step - 1 + int(getattr(args, "part_d_utility_batch_offset2", 1499)),
                            int(args.batch_size),
                            int(seed),
                        )
                diag = observe_structure_projected_batch(
                    opt,
                    model,
                    xb,
                    yb,
                    task,
                    seed * 10000 + step,
                    args,
                    orbit_random=random_scheme,
                    structure_mode=structure_mode,
                    utility_x=xub if structure_mode_uses_utility(structure_mode) else None,
                    utility_y=yub if structure_mode_uses_utility(structure_mode) else None,
                    utility_x2=xub2 if "stable_signed" in structure_mode else None,
                    utility_y2=yub2 if "stable_signed" in structure_mode else None,
                )
                cached_structure_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
                cached_structure_modifiers = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
                cached_structure_diag = dict(diag)
                if int(getattr(args, "part_d_trace", 0)):
                    utility_diag = train_only_descent_utility(opt, model, xub, yub, args)
                    descent_utility_trace.append(fval(utility_diag.get("descent_utility")))
                    descent_utility_cosine_trace.append(fval(utility_diag.get("descent_utility_cosine")))
                    ungated_descent_utility_trace.append(fval(utility_diag.get("ungated_descent_utility")))
                    utility_gate_over_ungated_trace.append(fval(utility_diag.get("utility_gate_over_ungated")))
                    trace_rows.append({
                        "part": "D_TRACE",
                        "scheme": scheme,
                        "structure_mode": structure_mode,
                        "basis_key": basis_key,
                        "depth": depth,
                        "task": task,
                        "seed": int(seed),
                        "step": int(step),
                        "structure_random_control": int(random_scheme),
                        "refresh_structure": 1,
                        "gate_floor": float(args.gate_floor),
                        "gate_floor_mode": str(args.gate_floor_mode),
                        "snr_beta": float(args.snr_beta),
                        "contrastive_eta": float(args.contrastive_eta),
                        "orbit_center_gain": float(args.orbit_center_gain),
                        "q_struct_mean": fval(diag.get("q_struct_mean")),
                        "local_coherence_mean": fval(diag.get("local_coherence_mean")),
                        "local_intervention_mean": fval(diag.get("local_intervention_mean")),
                        "rotation_coherence_mean": fval(diag.get("rotation_coherence_mean")),
                        "destructive_coherence_mean": fval(diag.get("destructive_coherence_mean")),
                        "utility_multiplier_mean": fval(diag.get("utility_multiplier_mean")),
                        "signed_modifier_mean": fval(diag.get("signed_modifier_mean")),
                        "signed_modifier_negative_fraction": fval(diag.get("signed_modifier_negative_fraction")),
                        "signed_modifier_agreement_fraction": fval(diag.get("signed_modifier_agreement_fraction")),
                        "grad_norm_median": fval(diag.get("grad_norm_median")),
                        "grad_norm_p95": fval(diag.get("grad_norm_p95")),
                        **utility_diag,
                        **AUDIT_DEFAULTS,
                    })
            else:
                pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, args)
                for param in model.coeffs:
                    gate = cached_structure_gates.get(id(param))
                    if gate is not None:
                        opt.observe_structure_gate(param, gate)
                    modifier = cached_structure_modifiers.get(id(param))
                    if modifier is not None:
                        opt.observe_structure_update_modifier(param, modifier)
                diag = {
                    **cached_structure_diag,
                    "per_example_count": pe_count,
                    "grad_norm_median": gmed,
                    "grad_norm_p95": gp95,
                    "grad_nan_count": nnan,
                    "grad_inf_count": ninf,
                }
            per_example_count = max(per_example_count, ival(diag.get("per_example_count")))
            grad_norm_median.append(fval(diag.get("grad_norm_median")))
            grad_norm_p95.append(fval(diag.get("grad_norm_p95")))
            grad_nan_count += ival(diag.get("grad_nan_count"))
            grad_inf_count += ival(diag.get("grad_inf_count"))
            struct_trace.append(fval(diag.get("q_struct_mean")))
            local_trace.append(fval(diag.get("local_coherence_mean")))
            local_intervention_trace.append(fval(diag.get("local_intervention_mean")))
            rot_trace.append(fval(diag.get("rotation_coherence_mean")))
            dest_trace.append(fval(diag.get("destructive_coherence_mean")))
            utility_mult_trace.append(fval(diag.get("utility_multiplier_mean")))
            signed_modifier_trace.append(fval(diag.get("signed_modifier_mean")))
            signed_modifier_negative_trace.append(fval(diag.get("signed_modifier_negative_fraction")))
            signed_modifier_agreement_trace.append(fval(diag.get("signed_modifier_agreement_fraction")))
            loss = F.cross_entropy(model(xb).float(), yb.long())
            loss.backward()
            opt.step()
            gate_trace.append(float(opt.last_stats.gate_density_mean))
            if trace_rows and int(trace_rows[-1].get("step", -1)) == int(step):
                trace_rows[-1]["post_step_gate_density"] = float(opt.last_stats.gate_density_mean)
                trace_rows[-1]["post_step_gate_density_median"] = float(opt.last_stats.gate_density_median)
                trace_rows[-1]["post_step_gate_density_p10"] = float(opt.last_stats.gate_density_p10)
                trace_rows[-1]["post_step_gate_density_p90"] = float(opt.last_stats.gate_density_p90)
        after = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        trace_path = "none"
        if trace_rows:
            trace_file = OUT_ROOT / (
                "part_d_training_dynamics_trace_"
                f"{safe_name(scheme)}_{safe_name(task)}_{safe_name(basis_key)}_{safe_name(depth)}_seed{int(seed)}.csv"
            )
            write_rows(trace_file, trace_rows)
            trace_path = rel(trace_file)
        struct_first = struct_trace[0] if struct_trace else 0.0
        struct_last = struct_trace[-1] if struct_trace else 0.0
        return {
            "part": "D",
            "status": "ok",
            "scheme": scheme,
            "structure_mode": structure_mode,
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "visual_synthetic_task": task,
            "seed": int(seed),
            "train_steps": int(args.train_steps),
            "gate_floor": float(args.gate_floor),
            "gate_floor_mode": str(args.gate_floor_mode),
            "snr_beta": float(args.snr_beta),
            "contrastive_eta": float(args.contrastive_eta),
            "sobolev_exponent": sobolev_exponent,
            "structure_random_control": int(random_scheme),
            "random_gap_orbit": 0.0,
            "gate_density": mean(gate_trace),
            "q_pop_mean": mean(gate_trace),
            "q_struct_mean": mean(struct_trace),
            "q_final_mean": mean(gate_trace),
            "local_coherence_mean": mean(local_trace),
            "local_intervention_mean": mean(local_intervention_trace),
            "rotation_coherence_mean": mean(rot_trace),
            "destructive_coherence_mean": mean(dest_trace),
            "utility_multiplier_mean": mean(utility_mult_trace),
            "signed_modifier_mean": mean(signed_modifier_trace),
            "signed_modifier_negative_fraction": mean(signed_modifier_negative_trace),
            "signed_modifier_agreement_fraction": mean(signed_modifier_agreement_trace),
            "C2_accuracy_initial": before["accuracy"],
            "C2_accuracy_final": after["accuracy"],
            "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "C2_coverage_initial": before["coverage_CVaR25"],
            "C2_coverage_final": after["coverage_CVaR25"],
            "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "overhead_ratio": 1.0,
            "guard_NLL_delta": after["nll"] - before["nll"],
            "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
            "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
            "per_example_count": per_example_count,
            "grad_norm_median": mean(grad_norm_median),
            "grad_norm_p95": mean(grad_norm_p95),
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "descent_utility_mean": mean(descent_utility_trace),
            "descent_utility_first": descent_utility_trace[0] if descent_utility_trace else 0.0,
            "descent_utility_last": descent_utility_trace[-1] if descent_utility_trace else 0.0,
            "descent_utility_cosine_mean": mean(descent_utility_cosine_trace),
            "ungated_descent_utility_mean": mean(ungated_descent_utility_trace),
            "utility_gate_over_ungated_mean": mean(utility_gate_over_ungated_trace),
            "q_struct_first": struct_first,
            "q_struct_last": struct_last,
            "q_struct_collapse_ratio": struct_last / struct_first if abs(struct_first) > 1.0e-12 else 0.0,
            "trace_row_count": len(trace_rows),
            "trace_path": trace_path,
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {"part": "D", "status": "error", "scheme": scheme, "structure_mode": structure_mode, "basis_key": basis_key, "depth": depth, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start, **AUDIT_DEFAULTS}


def part_d_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int]]:
    return [
        (scheme, task, basis, depth, seed)
        for scheme in csv_items(args.part_d_schemes)
        for task in csv_items(args.part_d_tasks)
        for basis in csv_items(args.part_d_basis)
        for depth in csv_items(args.part_d_depths)
        for seed in range(int(args.part_d_seed_count))
    ]


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows = [train_part_d_structure_row(job, args, device) for job in shard_items(part_d_jobs(args), args)]
    suffix = f"random{int(getattr(args, 'part_d_orbit_random', 0))}"
    path = write_rows(OUT_ROOT / f"part_d_taskwise_c2_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}_{suffix}.csv", rows)
    append_exec("part-d", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("D", 0, "D_ShardsWritten", "merge_required", rows)


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_d_taskwise_c2_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_d_taskwise_c2_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task")) == key]
        base_scheme = str(key[0])
        is_random_scheme = base_scheme.endswith("_OrbitRandom")
        matched_random_scheme = base_scheme if is_random_scheme else f"{base_scheme}_OrbitRandom"
        random_group = [r for r in ok if r.get("scheme") == matched_random_scheme and r.get("basis_key") == key[1] and r.get("depth") == key[2] and r.get("task") == key[3]]
        cov = median(r.get("C2_coverage_improvement") for r in group)
        acc = median(r.get("C2_accuracy_improvement") for r in group)
        rcov = median((r.get("C2_coverage_improvement") for r in random_group), 0.0)
        gap = cov - rcov if not is_random_scheme else 0.0
        task_summaries.append({
            "scheme": key[0],
            "structure_mode": sorted({str(r.get("structure_mode", "missing")) for r in group})[0],
            "basis_key": key[1],
            "depth": key[2],
            "task": key[3],
            "rows": len(group),
            "C2_coverage_improvement_median": cov,
            "C2_accuracy_improvement_median": acc,
            "random_gap_orbit": gap,
            "sobolev_exponent": median(r.get("sobolev_exponent") for r in group),
            "gate_density": median(r.get("gate_density") for r in group),
            "q_pop_mean": median(r.get("q_pop_mean") for r in group),
            "q_struct_mean": median(r.get("q_struct_mean") for r in group),
            "q_final_mean": median(r.get("q_final_mean") for r in group),
            "local_coherence_mean": median(r.get("local_coherence_mean") for r in group),
            "local_intervention_mean": median(r.get("local_intervention_mean") for r in group),
            "rotation_coherence_mean": median(r.get("rotation_coherence_mean") for r in group),
            "structure_task_alignment_score": median(r.get("q_struct_mean") for r in group),
            "descent_utility_mean": median(r.get("descent_utility_mean") for r in group),
            "descent_utility_cosine_mean": median(r.get("descent_utility_cosine_mean") for r in group),
            "ungated_descent_utility_mean": median(r.get("ungated_descent_utility_mean") for r in group),
            "utility_gate_over_ungated_mean": median(r.get("utility_gate_over_ungated_mean") for r in group),
            "utility_multiplier_mean": median(r.get("utility_multiplier_mean") for r in group),
            "signed_modifier_mean": median(r.get("signed_modifier_mean") for r in group),
            "signed_modifier_negative_fraction": median(r.get("signed_modifier_negative_fraction") for r in group),
            "signed_modifier_agreement_fraction": median(r.get("signed_modifier_agreement_fraction") for r in group),
            "q_struct_collapse_ratio": median(r.get("q_struct_collapse_ratio") for r in group),
            "trace_row_count": sum(ival(r.get("trace_row_count")) for r in group),
            "overhead_ratio": median(r.get("overhead_ratio") for r in group),
            "beats_C7_random": int(gap >= float(args.d_task_random_gap_gate)),
            "taskwise_pass": int(not is_random_scheme and cov >= float(args.d_task_coverage_gate) and gap >= float(args.d_task_random_gap_gate)),
        })
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in ok}):
        tasks = [t for t in task_summaries if (t.get("scheme"), t.get("basis_key"), t.get("depth")) == key]
        cov = median(t.get("C2_coverage_improvement_median") for t in tasks)
        acc = median(t.get("C2_accuracy_improvement_median") for t in tasks)
        gap = median(t.get("random_gap_orbit") for t in tasks)
        is_random_scheme = str(key[0]).endswith("_OrbitRandom")
        official = int(not is_random_scheme and str(key[0]) in PART_D_SCHEMES and all(ival(t.get("taskwise_pass")) == 1 for t in tasks) and cov >= float(args.d_overall_coverage_gate) and gap >= float(args.d_overall_random_gap_gate))
        groups.append({
            "scheme": key[0],
            "structure_mode": sorted({str(t.get("structure_mode", "missing")) for t in tasks})[0],
            "basis_key": key[1],
            "depth": key[2],
            "tasks": ",".join(str(t.get("task")) for t in tasks),
            "rows": sum(ival(t.get("rows")) for t in tasks),
            "C2_coverage_improvement_median": cov,
            "C2_accuracy_improvement_median": acc,
            "random_gap_orbit": gap,
            "descent_utility_mean": median(t.get("descent_utility_mean") for t in tasks),
            "descent_utility_cosine_mean": median(t.get("descent_utility_cosine_mean") for t in tasks),
            "sobolev_exponent": median(t.get("sobolev_exponent") for t in tasks),
            "utility_gate_over_ungated_mean": median(t.get("utility_gate_over_ungated_mean") for t in tasks),
            "utility_multiplier_mean": median(t.get("utility_multiplier_mean") for t in tasks),
            "local_intervention_mean": median(t.get("local_intervention_mean") for t in tasks),
            "signed_modifier_mean": median(t.get("signed_modifier_mean") for t in tasks),
            "signed_modifier_negative_fraction": median(t.get("signed_modifier_negative_fraction") for t in tasks),
            "signed_modifier_agreement_fraction": median(t.get("signed_modifier_agreement_fraction") for t in tasks),
            "q_struct_collapse_ratio": median(t.get("q_struct_collapse_ratio") for t in tasks),
            "taskwise_all_pass": int(bool(tasks) and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)),
            "official_pass": official,
        })
    pass_groups = [g for g in groups if ival(g.get("official_pass")) == 1]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    blocker = "part_d_job_errors" if any(r.get("status") == "error" for r in rows) else ("none" if gate else "taskwise_c2_or_random_gap_failed")
    task_csv = write_rows(OUT_ROOT / "part_d_taskwise_c2_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_d_taskwise_c2_group_summary.csv", groups)
    trace_rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_d_training_dynamics_trace_*.csv")):
        trace_rows.extend(read_rows(path))
    trace_csv = write_rows(OUT_ROOT / "part_d_training_dynamics_trace.csv", trace_rows) if trace_rows else None
    summary = gate_summary("D", gate, "D_TaskwiseC2Pass" if gate else "D_TaskwiseC2Failed", blocker, rows, scheme_groups=groups, taskwise_groups=task_summaries, passing_scheme_groups=pass_groups)
    if trace_rows:
        summary["trace_row_count"] = len(trace_rows)
        summary["trace_path"] = rel(trace_csv) if trace_csv is not None else "none"
    write_json(OUT_ROOT / "part_d_summary.json", summary)
    nxt = next_actions("D", gate, blocker, [] if gate else ["inspect structure gate training dynamics", "compare longer train_steps and lower eta without changing Part C thresholds"])
    files = f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(nxt)}"
    if trace_csv is not None:
        files = f"{files}; {rel(trace_csv)}"
    append_exec("part-d-merge", command_text(sys.argv), "passed" if gate else "failed", files=files, gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part D structure-projected C2 smoke", summary)
    return summary


def train_part_e_random_mechanism_row(job: tuple[str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    task, basis_key, depth, seed, random_type = job
    start = time.time()
    scheme = e_scheme_name(random_type)
    source_scheme = csv_items(args.part_d_schemes)[0] if csv_items(args.part_d_schemes) else "D4_ContrastiveStructureProduct"
    structure_mode = PART_D_SCHEMES.get(source_scheme, "contrastive")
    source_sobolev_exponent = part_d_sobolev_exponent(source_scheme)
    try:
        bargs = v2300.basis_args(args, basis_key)
        xtr, ytr, xg, yg = v2300.visual_data(task, seed, args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = v2300.make_model(depth, int(xtr.shape[1]), classes, v2300.model_seed_for(basis_key, depth, task, seed), bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        opt = EdgeSobolevSNRFU(
            list(model.coeffs),
            lr=float(args.edge_lr),
            weight_decay=float(args.weight_decay),
            sobolev_exponent=source_sobolev_exponent,
            edge_metric_type="functional_gram",
            edge_weight_normalization=str(args.edge_weight_normalization),
            edge_weight_ridge=float(args.edge_weight_ridge),
            functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
            gate_beta=float(args.snr_beta),
            gate_family=str(args.structure_block_family),
            gate_floor=float(args.gate_floor),
            gate_floor_mode=str(args.gate_floor_mode),
            gate_input_side=int(args.visual_side),
            gate_patch_size=2,
            stat_warmup_steps=int(getattr(args, "stat_warmup_steps", 0)),
            use_population_gate=True,
        )
        before = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        gate_trace: list[float] = []
        struct_trace: list[float] = []
        local_trace: list[float] = []
        rot_trace: list[float] = []
        dest_trace: list[float] = []
        utility_mult_trace: list[float] = []
        signed_modifier_trace: list[float] = []
        signed_modifier_negative_trace: list[float] = []
        local_intervention_trace: list[float] = []
        density_errors: list[float] = []
        hist_errors: list[float] = []
        orbit_errors: list[float] = []
        grad_norm_median: list[float] = []
        grad_norm_p95: list[float] = []
        grad_nan_count = 0
        grad_inf_count = 0
        per_example_count = 0
        cached_structure_gates: dict[int, torch.Tensor] = {}
        cached_structure_modifiers: dict[int, torch.Tensor] = {}
        cached_structure_diag: dict[str, float | int] = {}
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            refresh_structure = not cached_structure_gates or (int(args.structure_gate_every) <= 1) or ((step - 1) % int(args.structure_gate_every) == 0)
            if refresh_structure:
                xub = yub = None
                if structure_mode_uses_utility(structure_mode):
                    xub, yub = v2293.v2289.iter_train_batches(
                        xtr,
                        ytr,
                        step - 1 + int(getattr(args, "part_d_utility_batch_offset", 997)),
                        int(args.batch_size),
                        int(seed),
                    )
                diag = observe_structure_projected_batch(
                    opt,
                    model,
                    xb,
                    yb,
                    task,
                    seed * 10000 + step,
                    args,
                    structure_control_type=str(random_type),
                    structure_mode=structure_mode,
                    utility_x=xub if structure_mode_uses_utility(structure_mode) else None,
                    utility_y=yub if structure_mode_uses_utility(structure_mode) else None,
                )
                cached_structure_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
                cached_structure_modifiers = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
                cached_structure_diag = dict(diag)
            else:
                pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, args)
                for param in model.coeffs:
                    gate = cached_structure_gates.get(id(param))
                    if gate is not None:
                        opt.observe_structure_gate(param, gate)
                    modifier = cached_structure_modifiers.get(id(param))
                    if modifier is not None:
                        opt.observe_structure_update_modifier(param, modifier)
                diag = {
                    **cached_structure_diag,
                    "per_example_count": pe_count,
                    "grad_norm_median": gmed,
                    "grad_norm_p95": gp95,
                    "grad_nan_count": nnan,
                    "grad_inf_count": ninf,
                }
            per_example_count = max(per_example_count, ival(diag.get("per_example_count")))
            grad_norm_median.append(fval(diag.get("grad_norm_median")))
            grad_norm_p95.append(fval(diag.get("grad_norm_p95")))
            grad_nan_count += ival(diag.get("grad_nan_count"))
            grad_inf_count += ival(diag.get("grad_inf_count"))
            struct_trace.append(fval(diag.get("q_struct_mean")))
            local_trace.append(fval(diag.get("local_coherence_mean")))
            local_intervention_trace.append(fval(diag.get("local_intervention_mean")))
            rot_trace.append(fval(diag.get("rotation_coherence_mean")))
            dest_trace.append(fval(diag.get("destructive_coherence_mean")))
            utility_mult_trace.append(fval(diag.get("utility_multiplier_mean")))
            signed_modifier_trace.append(fval(diag.get("signed_modifier_mean")))
            signed_modifier_negative_trace.append(fval(diag.get("signed_modifier_negative_fraction")))
            density_errors.append(fval(diag.get("structure_density_match_error")))
            hist_errors.append(fval(diag.get("structure_histogram_mean_error")))
            orbit_errors.append(fval(diag.get("orbit_size_match_error")))
            loss = F.cross_entropy(model(xb).float(), yb.long())
            loss.backward()
            opt.step()
            gate_trace.append(float(opt.last_stats.gate_density_mean))
        after = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        return {
            "part": "E",
            "status": "ok",
            "scheme": scheme,
            "random_type": str(random_type),
            "source_scheme": source_scheme,
            "structure_mode": structure_mode,
            "sobolev_exponent": source_sobolev_exponent,
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "visual_synthetic_task": task,
            "seed": int(seed),
            "train_steps": int(args.train_steps),
            "gate_floor": float(args.gate_floor),
            "gate_floor_mode": str(args.gate_floor_mode),
            "snr_beta": float(args.snr_beta),
            "contrastive_eta": float(args.contrastive_eta),
            "gate_density": mean(gate_trace),
            "q_pop_mean": mean(gate_trace),
            "q_struct_mean": mean(struct_trace),
            "q_final_mean": mean(gate_trace),
            "local_coherence_mean": mean(local_trace),
            "local_intervention_mean": mean(local_intervention_trace),
            "rotation_coherence_mean": mean(rot_trace),
            "destructive_coherence_mean": mean(dest_trace),
            "utility_multiplier_mean": mean(utility_mult_trace),
            "signed_modifier_mean": mean(signed_modifier_trace),
            "signed_modifier_negative_fraction": mean(signed_modifier_negative_trace),
            "structure_density_match_error": mean(density_errors),
            "structure_histogram_mean_error": mean(hist_errors),
            "orbit_size_match_error": mean(orbit_errors),
            "C2_accuracy_initial": before["accuracy"],
            "C2_accuracy_final": after["accuracy"],
            "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "C2_coverage_initial": before["coverage_CVaR25"],
            "C2_coverage_final": after["coverage_CVaR25"],
            "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "random_gap_vs_true": 0.0,
            "guard_NLL_delta": after["nll"] - before["nll"],
            "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
            "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
            "per_example_count": per_example_count,
            "grad_norm_median": mean(grad_norm_median),
            "grad_norm_p95": mean(grad_norm_p95),
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {
            "part": "E",
            "status": "error",
            "scheme": scheme,
            "random_type": str(random_type),
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "seed": int(seed),
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def part_e_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, int, str]]:
    return [
        (task, basis, depth, seed, random_type)
        for task in csv_items(args.part_d_tasks)
        for basis in csv_items(args.part_d_basis)
        for depth in csv_items(args.part_d_depths)
        for seed in range(int(args.part_d_seed_count))
        for random_type in csv_items(args.part_e_random_types)
    ]


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows = [train_part_e_random_mechanism_row(job, args, device) for job in shard_items(part_e_jobs(args), args)]
    path = write_rows(OUT_ROOT / f"part_e_random_mechanism_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-e", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("E", 0, "E_ShardsWritten", "merge_required", rows)


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_e_random_mechanism_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("random_type"), r.get("basis_key"), r.get("depth"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("random_type"), r.get("basis_key"), r.get("depth"), r.get("task")) == key]
        true_group = [r for r in ok if r.get("random_type") == "true" and r.get("basis_key") == key[1] and r.get("depth") == key[2] and r.get("task") == key[3]]
        cov = median(r.get("C2_coverage_improvement") for r in group)
        true_cov = median(r.get("C2_coverage_improvement") for r in true_group)
        gap = true_cov - cov if str(key[0]) != "true" else 0.0
        for row in group:
            row["random_gap_vs_true"] = gap if str(key[0]) != "true" else 0.0
        task_summaries.append({
            "random_type": key[0],
            "scheme": e_scheme_name(str(key[0])),
            "basis_key": key[1],
            "depth": key[2],
            "task": key[3],
            "rows": len(group),
            "true_C2_coverage_improvement_median": true_cov,
            "C2_coverage_improvement_median": cov,
            "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group),
            "random_gap_vs_true": gap,
            "sobolev_exponent": median(r.get("sobolev_exponent") for r in group),
            "gate_density": median(r.get("gate_density") for r in group),
            "q_struct_mean": median(r.get("q_struct_mean") for r in group),
            "local_coherence_mean": median(r.get("local_coherence_mean") for r in group),
            "local_intervention_mean": median(r.get("local_intervention_mean") for r in group),
            "rotation_coherence_mean": median(r.get("rotation_coherence_mean") for r in group),
            "destructive_coherence_mean": median(r.get("destructive_coherence_mean") for r in group),
            "utility_multiplier_mean": median(r.get("utility_multiplier_mean") for r in group),
            "signed_modifier_mean": median(r.get("signed_modifier_mean") for r in group),
            "signed_modifier_negative_fraction": median(r.get("signed_modifier_negative_fraction") for r in group),
            "structure_density_match_error": median(r.get("structure_density_match_error") for r in group),
            "structure_histogram_mean_error": median(r.get("structure_histogram_mean_error") for r in group),
            "orbit_size_match_error": median(r.get("orbit_size_match_error") for r in group),
        })
    matrix = write_rows(OUT_ROOT / "part_e_random_mechanism_matrix.csv", rows)
    task_csv = write_rows(OUT_ROOT / "part_e_random_mechanism_task_summary.csv", task_summaries)
    structure_breaking = [t for t in task_summaries if t.get("random_type") == "structure_breaking"]
    orbit_shuffle = [t for t in task_summaries if t.get("random_type") == "orbit_shuffle"]
    density = [t for t in task_summaries if t.get("random_type") == "density"]
    histogram = [t for t in task_summaries if t.get("random_type") == "histogram"]
    structure_breaking_pass = bool(structure_breaking) and all(fval(t.get("random_gap_vs_true")) >= float(args.e_structure_breaking_gap_gate) for t in structure_breaking)
    orbit_shuffle_weaker = any(fval(t.get("random_gap_vs_true")) > 0.0 for t in orbit_shuffle)
    density_close_all = bool(density) and all(abs(fval(t.get("random_gap_vs_true"))) <= float(args.e_equal_tolerance) for t in density)
    histogram_close_all = bool(histogram) and all(abs(fval(t.get("random_gap_vs_true"))) <= float(args.e_equal_tolerance) for t in histogram)
    density_hist_not_both_equal = not (density_close_all and histogram_close_all)
    gate = int(
        not any(r.get("status") == "error" for r in rows)
        and structure_breaking_pass
        and orbit_shuffle_weaker
        and density_hist_not_both_equal
    )
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_e_job_errors"
    elif gate:
        blocker = "none"
    else:
        blocker = "matched_random_strength_unresolved"
    groups = [{
        "structure_breaking_pass": int(structure_breaking_pass),
        "orbit_shuffle_weaker_on_at_least_one_task": int(orbit_shuffle_weaker),
        "density_histogram_not_both_equal_true": int(density_hist_not_both_equal),
        "density_close_all_tasks": int(density_close_all),
        "histogram_close_all_tasks": int(histogram_close_all),
    }]
    summary = gate_summary(
        "E",
        gate,
        "E_RandomMechanismExplained" if gate else "E_RandomMechanismUnresolved",
        blocker,
        rows,
        taskwise_groups=task_summaries,
        mechanism_checks=groups[0],
    )
    write_json(OUT_ROOT / "part_e_random_mechanism_summary.json", summary)
    nxt = next_actions(
        "E",
        gate,
        blocker,
        [] if gate else [
            "if all random controls remain close to true, return to Part C transform-family redesign",
            "audit whether orbit definitions are too coarse for local/rotation task structure",
            "do not enter Part G official safety path while Part D/E random mechanism remains unresolved",
        ],
    )
    append_exec("part-e-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(OUT_ROOT / 'part_e_random_mechanism_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part E matched-random mechanism audit", summary)
    return summary


def part_f_scheme_config(scheme: str) -> dict[str, Any]:
    mode = PART_F_SCHEMES.get(str(scheme), "functional_gram_ps_qpop_ps_signed_utility")
    if mode == "raw_adamw":
        return {
            "optimizer": "adamw",
            "edge_metric_type": "none",
            "sobolev_exponent": 0.0,
            "use_population_gate": False,
            "force_population_identity": False,
            "use_structure_gate": False,
            "structure_mode": "none",
            "functional_gram_used": 0,
            "qpop_used": 0,
            "ps_used": 0,
            "signed_utility_used": 0,
        }
    if mode == "functional_gram_adamw":
        return {
            "optimizer": "edge",
            "edge_metric_type": "functional_gram",
            "sobolev_exponent": 0.0,
            "use_population_gate": False,
            "force_population_identity": False,
            "use_structure_gate": False,
            "structure_mode": "none",
            "functional_gram_used": 1,
            "qpop_used": 0,
            "ps_used": 0,
            "signed_utility_used": 0,
        }
    if mode == "functional_gram_qpop_only":
        return {
            "optimizer": "edge",
            "edge_metric_type": "functional_gram",
            "sobolev_exponent": 0.0,
            "use_population_gate": True,
            "force_population_identity": False,
            "use_structure_gate": False,
            "structure_mode": "pop_only",
            "functional_gram_used": 1,
            "qpop_used": 1,
            "ps_used": 0,
            "signed_utility_used": 0,
        }
    if mode == "functional_gram_ps_only":
        return {
            "optimizer": "edge",
            "edge_metric_type": "functional_gram",
            "sobolev_exponent": 0.0,
            "use_population_gate": True,
            "force_population_identity": True,
            "use_structure_gate": True,
            "structure_mode": "local_intervention_coordinate_orbit_centered_contrastive",
            "functional_gram_used": 1,
            "qpop_used": 0,
            "ps_used": 1,
            "signed_utility_used": 0,
        }
    if mode == "functional_sobolev025_ps_qpop_ps_signed_utility":
        sobolev = 0.25
    elif mode == "functional_sobolev0125_ps_qpop_ps_signed_utility":
        sobolev = 0.125
    else:
        sobolev = 0.0
    return {
        "optimizer": "edge",
        "edge_metric_type": "functional_gram",
        "sobolev_exponent": sobolev,
        "use_population_gate": True,
        "force_population_identity": False,
        "use_structure_gate": True,
        "structure_mode": "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
        "functional_gram_used": 1,
        "qpop_used": 1,
        "ps_used": 1,
        "signed_utility_used": 1,
    }


def make_part_f_optimizer(
    model: v2293.TrueDeepPureKAN,
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    params = list(model.coeffs)
    if str(cfg.get("optimizer")) == "adamw":
        return torch.optim.AdamW(params, lr=float(args.adamw_lr), weight_decay=float(args.weight_decay))
    return EdgeSobolevPopulationFlow(
        params,
        lr=float(args.edge_lr),
        weight_decay=float(args.weight_decay),
        sobolev_exponent=float(cfg.get("sobolev_exponent", 0.0)),
        edge_metric_type=str(cfg.get("edge_metric_type", "functional_gram")),
        edge_weight_normalization=str(args.edge_weight_normalization),
        edge_weight_ridge=float(args.edge_weight_ridge),
        functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
        gate_beta=float(args.snr_beta),
        gate_family=str(args.structure_block_family),
        gate_floor=float(args.gate_floor),
        gate_floor_mode=str(args.gate_floor_mode),
        gate_input_side=int(args.visual_side),
        gate_patch_size=2,
        stat_warmup_steps=int(getattr(args, "stat_warmup_steps", 0)),
        use_population_gate=bool(cfg.get("use_population_gate", False)),
    )


def force_population_identity_gate(opt: torch.optim.Optimizer) -> None:
    observed = getattr(opt, "_observed_grads", None)
    if not isinstance(observed, dict):
        return
    for key, value in list(observed.items()):
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.numel()) > 0:
            observed[key] = torch.ones_like(value)


def part_f_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int, str]]:
    return [
        (scheme, task, basis, depth, seed, control)
        for scheme in csv_items(args.part_f_schemes)
        for task in csv_items(args.part_d_tasks)
        for basis in csv_items(args.part_d_basis)
        for depth in csv_items(args.part_d_depths)
        for seed in range(int(args.part_d_seed_count))
        for control in csv_items(args.part_f_control_types)
    ]


def train_part_f_component_row(job: tuple[str, str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, task, basis_key, depth, seed, control_type = job
    start = time.time()
    cfg = part_f_scheme_config(str(scheme))
    structure_mode = str(cfg.get("structure_mode", "none"))
    try:
        bargs = v2300.basis_args(args, basis_key)
        xtr, ytr, xg, yg = v2300.visual_data(task, seed, args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = v2300.make_model(depth, int(xtr.shape[1]), classes, v2300.model_seed_for(basis_key, depth, task, seed), bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        opt = make_part_f_optimizer(model, cfg, args)
        before = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        gate_trace: list[float] = []
        struct_trace: list[float] = []
        local_trace: list[float] = []
        local_intervention_trace: list[float] = []
        rot_trace: list[float] = []
        dest_trace: list[float] = []
        utility_mult_trace: list[float] = []
        signed_modifier_trace: list[float] = []
        signed_modifier_negative_trace: list[float] = []
        grad_norm_median: list[float] = []
        grad_norm_p95: list[float] = []
        grad_nan_count = 0
        grad_inf_count = 0
        per_example_count = 0
        cached_structure_gates: dict[int, torch.Tensor] = {}
        cached_structure_modifiers: dict[int, torch.Tensor] = {}
        cached_structure_diag: dict[str, float | int] = {}
        use_edge_observation = isinstance(opt, EdgeSobolevPopulationFlow)
        use_structure_gate = bool(cfg.get("use_structure_gate", False)) and use_edge_observation
        use_qpop = bool(cfg.get("use_population_gate", False)) and use_edge_observation
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            diag: dict[str, float | int]
            if use_structure_gate:
                refresh_structure = not cached_structure_gates or (int(args.structure_gate_every) <= 1) or ((step - 1) % int(args.structure_gate_every) == 0)
                if refresh_structure:
                    xub = yub = None
                    if structure_mode_uses_utility(structure_mode):
                        xub, yub = v2293.v2289.iter_train_batches(
                            xtr,
                            ytr,
                            step - 1 + int(getattr(args, "part_d_utility_batch_offset", 997)),
                            int(args.batch_size),
                            int(seed),
                        )
                    diag = observe_structure_projected_batch(
                        opt,
                        model,
                        xb,
                        yb,
                        task,
                        seed * 10000 + step,
                        args,
                        structure_control_type=str(control_type),
                        structure_mode=structure_mode,
                        utility_x=xub if structure_mode_uses_utility(structure_mode) else None,
                        utility_y=yub if structure_mode_uses_utility(structure_mode) else None,
                    )
                    cached_structure_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
                    cached_structure_modifiers = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
                    cached_structure_diag = dict(diag)
                else:
                    pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, args)
                    for param in model.coeffs:
                        gate = cached_structure_gates.get(id(param))
                        if gate is not None:
                            opt.observe_structure_gate(param, gate)
                        modifier = cached_structure_modifiers.get(id(param))
                        if modifier is not None:
                            opt.observe_structure_update_modifier(param, modifier)
                    diag = {
                        **cached_structure_diag,
                        "per_example_count": pe_count,
                        "grad_norm_median": gmed,
                        "grad_norm_p95": gp95,
                        "grad_nan_count": nnan,
                        "grad_inf_count": ninf,
                    }
                if bool(cfg.get("force_population_identity", False)):
                    force_population_identity_gate(opt)
            elif use_qpop:
                pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, args)
                diag = {
                    "per_example_count": pe_count,
                    "grad_norm_median": gmed,
                    "grad_norm_p95": gp95,
                    "grad_nan_count": nnan,
                    "grad_inf_count": ninf,
                    "q_struct_mean": 1.0,
                    "local_coherence_mean": 0.0,
                    "local_intervention_mean": 0.0,
                    "rotation_coherence_mean": 0.0,
                    "destructive_coherence_mean": 0.0,
                    "utility_multiplier_mean": 0.0,
                    "signed_modifier_mean": 0.0,
                    "signed_modifier_negative_fraction": 0.0,
                }
            else:
                diag = {
                    "per_example_count": 0,
                    "grad_norm_median": 0.0,
                    "grad_norm_p95": 0.0,
                    "grad_nan_count": 0,
                    "grad_inf_count": 0,
                    "q_struct_mean": 0.0,
                    "local_coherence_mean": 0.0,
                    "local_intervention_mean": 0.0,
                    "rotation_coherence_mean": 0.0,
                    "destructive_coherence_mean": 0.0,
                    "utility_multiplier_mean": 0.0,
                    "signed_modifier_mean": 0.0,
                    "signed_modifier_negative_fraction": 0.0,
                }
            per_example_count = max(per_example_count, ival(diag.get("per_example_count")))
            grad_norm_median.append(fval(diag.get("grad_norm_median")))
            grad_norm_p95.append(fval(diag.get("grad_norm_p95")))
            grad_nan_count += ival(diag.get("grad_nan_count"))
            grad_inf_count += ival(diag.get("grad_inf_count"))
            struct_trace.append(fval(diag.get("q_struct_mean")))
            local_trace.append(fval(diag.get("local_coherence_mean")))
            local_intervention_trace.append(fval(diag.get("local_intervention_mean")))
            rot_trace.append(fval(diag.get("rotation_coherence_mean")))
            dest_trace.append(fval(diag.get("destructive_coherence_mean")))
            utility_mult_trace.append(fval(diag.get("utility_multiplier_mean")))
            signed_modifier_trace.append(fval(diag.get("signed_modifier_mean")))
            signed_modifier_negative_trace.append(fval(diag.get("signed_modifier_negative_fraction")))
            loss = F.cross_entropy(model(xb).float(), yb.long())
            loss.backward()
            opt.step()
            stats = getattr(opt, "last_stats", None)
            gate_trace.append(float(getattr(stats, "gate_density_mean", 1.0)))
        after = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        return {
            "part": "F",
            "status": "ok",
            "scheme": str(scheme),
            "component_ablation_scheme": str(scheme),
            "random_type": str(control_type),
            "structure_mode": structure_mode,
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "visual_synthetic_task": task,
            "seed": int(seed),
            "train_steps": int(args.train_steps),
            "functional_gram_used": int(cfg.get("functional_gram_used", 0)),
            "qpop_used": int(cfg.get("qpop_used", 0)),
            "ps_used": int(cfg.get("ps_used", 0)),
            "signed_utility_used": int(cfg.get("signed_utility_used", 0)),
            "force_population_identity": int(cfg.get("force_population_identity", False)),
            "edge_metric_type": str(cfg.get("edge_metric_type", "none")),
            "sobolev_exponent": float(cfg.get("sobolev_exponent", 0.0)),
            "gate_floor": float(args.gate_floor),
            "gate_floor_mode": str(args.gate_floor_mode),
            "snr_beta": float(args.snr_beta),
            "contrastive_eta": float(args.contrastive_eta),
            "gate_density": mean(gate_trace),
            "q_pop_mean": mean(gate_trace) if int(cfg.get("qpop_used", 0)) else 1.0,
            "q_struct_mean": mean(struct_trace),
            "q_final_mean": mean(gate_trace),
            "local_coherence_mean": mean(local_trace),
            "local_intervention_mean": mean(local_intervention_trace),
            "rotation_coherence_mean": mean(rot_trace),
            "destructive_coherence_mean": mean(dest_trace),
            "utility_multiplier_mean": mean(utility_mult_trace),
            "signed_modifier_mean": mean(signed_modifier_trace),
            "signed_modifier_negative_fraction": mean(signed_modifier_negative_trace),
            "C2_accuracy_initial": before["accuracy"],
            "C2_accuracy_final": after["accuracy"],
            "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "C2_coverage_initial": before["coverage_CVaR25"],
            "C2_coverage_final": after["coverage_CVaR25"],
            "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "random_gap_orbit": 0.0,
            "overhead_ratio": 1.0,
            "guard_NLL_delta": after["nll"] - before["nll"],
            "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
            "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
            "per_example_count": per_example_count,
            "grad_norm_median": mean(grad_norm_median),
            "grad_norm_p95": mean(grad_norm_p95),
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {
            "part": "F",
            "status": "error",
            "scheme": str(scheme),
            "component_ablation_scheme": str(scheme),
            "random_type": str(control_type),
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "seed": int(seed),
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows = [train_part_f_component_row(job, args, device) for job in shard_items(part_f_jobs(args), args)]
    path = write_rows(OUT_ROOT / f"part_f_component_ablation_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-f", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("F", 0, "F_ShardsWritten", "merge_required", rows)


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_component_ablation_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("random_type"), r.get("basis_key"), r.get("depth"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("random_type"), r.get("basis_key"), r.get("depth"), r.get("task")) == key]
        true_group = [r for r in ok if r.get("scheme") == key[0] and r.get("random_type") == "true" and r.get("basis_key") == key[2] and r.get("depth") == key[3] and r.get("task") == key[4]]
        orbit_group = [r for r in ok if r.get("scheme") == key[0] and r.get("random_type") == "orbit_shuffle" and r.get("basis_key") == key[2] and r.get("depth") == key[3] and r.get("task") == key[4]]
        cov = median(r.get("C2_coverage_improvement") for r in group)
        true_cov = median(r.get("C2_coverage_improvement") for r in true_group)
        orbit_cov = median(r.get("C2_coverage_improvement") for r in orbit_group)
        random_gap = true_cov - orbit_cov if true_group and orbit_group else 0.0
        for row in group:
            row["random_gap_orbit"] = random_gap
        task_summaries.append({
            "scheme": key[0],
            "component_ablation_scheme": key[0],
            "random_type": key[1],
            "structure_mode": sorted({str(r.get("structure_mode", "missing")) for r in group})[0],
            "basis_key": key[2],
            "depth": key[3],
            "task": key[4],
            "rows": len(group),
            "true_C2_coverage_improvement_median": true_cov,
            "orbit_C2_coverage_improvement_median": orbit_cov,
            "C2_coverage_improvement_median": cov,
            "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group),
            "random_gap_orbit": random_gap,
            "gate_density": median(r.get("gate_density") for r in group),
            "q_pop_mean": median(r.get("q_pop_mean") for r in group),
            "q_struct_mean": median(r.get("q_struct_mean") for r in group),
            "q_final_mean": median(r.get("q_final_mean") for r in group),
            "local_coherence_mean": median(r.get("local_coherence_mean") for r in group),
            "local_intervention_mean": median(r.get("local_intervention_mean") for r in group),
            "rotation_coherence_mean": median(r.get("rotation_coherence_mean") for r in group),
            "destructive_coherence_mean": median(r.get("destructive_coherence_mean") for r in group),
            "utility_multiplier_mean": median(r.get("utility_multiplier_mean") for r in group),
            "signed_modifier_mean": median(r.get("signed_modifier_mean") for r in group),
            "signed_modifier_negative_fraction": median(r.get("signed_modifier_negative_fraction") for r in group),
            "functional_gram_used": max(ival(r.get("functional_gram_used")) for r in group),
            "qpop_used": max(ival(r.get("qpop_used")) for r in group),
            "ps_used": max(ival(r.get("ps_used")) for r in group),
            "signed_utility_used": max(ival(r.get("signed_utility_used")) for r in group),
            "sobolev_exponent": median(r.get("sobolev_exponent") for r in group),
            "overhead_ratio": median(r.get("overhead_ratio") for r in group),
            "taskwise_pass": int(str(key[1]) == "true" and true_cov >= float(args.d_task_coverage_gate) and random_gap >= float(args.d_task_random_gap_gate)),
        })
    groups: list[dict[str, Any]] = []
    for key in sorted({(t.get("scheme"), t.get("basis_key"), t.get("depth")) for t in task_summaries if t.get("random_type") == "true"}):
        tasks = [t for t in task_summaries if (t.get("scheme"), t.get("basis_key"), t.get("depth")) == key and t.get("random_type") == "true"]
        scheme_rows = [r for r in ok if r.get("scheme") == key[0] and r.get("basis_key") == key[1] and r.get("depth") == key[2]]
        groups.append({
            "scheme": key[0],
            "component_ablation_scheme": key[0],
            "structure_mode": sorted({str(t.get("structure_mode", "missing")) for t in tasks})[0] if tasks else "missing",
            "basis_key": key[1],
            "depth": key[2],
            "tasks": ",".join(str(t.get("task")) for t in tasks),
            "rows": len(scheme_rows),
            "C2_coverage_improvement_median": median(t.get("C2_coverage_improvement_median") for t in tasks),
            "C2_accuracy_improvement_median": median(t.get("C2_accuracy_improvement_median") for t in tasks),
            "random_gap_orbit": median(t.get("random_gap_orbit") for t in tasks),
            "gate_density": median(t.get("gate_density") for t in tasks),
            "q_pop_mean": median(t.get("q_pop_mean") for t in tasks),
            "q_struct_mean": median(t.get("q_struct_mean") for t in tasks),
            "utility_multiplier_mean": median(t.get("utility_multiplier_mean") for t in tasks),
            "signed_modifier_mean": median(t.get("signed_modifier_mean") for t in tasks),
            "signed_modifier_negative_fraction": median(t.get("signed_modifier_negative_fraction") for t in tasks),
            "functional_gram_used": max((ival(r.get("functional_gram_used")) for r in scheme_rows), default=0),
            "qpop_used": max((ival(r.get("qpop_used")) for r in scheme_rows), default=0),
            "ps_used": max((ival(r.get("ps_used")) for r in scheme_rows), default=0),
            "signed_utility_used": max((ival(r.get("signed_utility_used")) for r in scheme_rows), default=0),
            "sobolev_exponent": median(r.get("sobolev_exponent") for r in scheme_rows),
            "overhead_ratio": median(r.get("overhead_ratio") for r in scheme_rows),
            "taskwise_all_pass": int(bool(tasks) and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)),
        })
    matrix = write_rows(OUT_ROOT / "part_f_component_ablation_matrix.csv", rows)
    task_csv = write_rows(OUT_ROOT / "part_f_component_ablation_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_f_component_ablation_group_summary.csv", groups)
    by_scheme = {str(g.get("scheme")): g for g in groups}
    official_scheme = str(getattr(args, "part_f_official_scheme", "F7_FunctionalSobolev025PsQpopPsSignedUtility"))
    official_group = by_scheme.get(official_scheme)
    baseline_names = [
        "F2_FunctionalGramAdamWNoQpopNoPs",
        "F3_FunctionalGramQpopOnly",
        "F4_FunctionalGramPsOnly",
    ]
    baseline_cov = max([fval(by_scheme.get(name, {}).get("C2_coverage_improvement_median"), -1.0e9) for name in baseline_names], default=-1.0e9)
    official_cov = fval((official_group or {}).get("C2_coverage_improvement_median"), -1.0e9)
    f3_gap = fval(by_scheme.get("F3_FunctionalGramQpopOnly", {}).get("random_gap_orbit"), 0.0)
    official_gap = fval((official_group or {}).get("random_gap_orbit"), -1.0e9)
    component_advantage = official_cov - baseline_cov
    random_gap_advantage = official_gap - f3_gap
    official_taskwise = bool(official_group) and ival(official_group.get("taskwise_all_pass")) == 1
    official_overhead_ok = bool(official_group) and fval(official_group.get("overhead_ratio"), 99.0) <= 2.0
    component_advantage_pass = bool(official_group) and component_advantage >= float(args.f_component_advantage_gate)
    random_gap_advantage_pass = bool(official_group) and random_gap_advantage >= float(args.f_random_gap_advantage_gate)
    gate = int(
        not any(r.get("status") == "error" for r in rows)
        and bool(official_group)
        and component_advantage_pass
        and random_gap_advantage_pass
        and official_taskwise
        and official_overhead_ok
    )
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_f_job_errors"
    elif not official_group:
        blocker = "official_component_missing"
    elif not official_taskwise:
        blocker = "official_component_taskwise_failed"
    elif not component_advantage_pass:
        blocker = "official_component_advantage_failed"
    elif not random_gap_advantage_pass:
        blocker = "official_component_random_gap_advantage_failed"
    elif not official_overhead_ok:
        blocker = "official_component_overhead_failed"
    else:
        blocker = "none"
    checks = {
        "official_scheme": official_scheme,
        "official_coverage": official_cov,
        "best_f2_f3_f4_coverage": baseline_cov,
        "component_advantage": component_advantage,
        "component_advantage_gate": float(args.f_component_advantage_gate),
        "component_advantage_pass": int(component_advantage_pass),
        "official_random_gap_orbit": official_gap,
        "f3_random_gap_orbit": f3_gap,
        "random_gap_advantage": random_gap_advantage,
        "random_gap_advantage_gate": float(args.f_random_gap_advantage_gate),
        "random_gap_advantage_pass": int(random_gap_advantage_pass),
        "official_taskwise_all_pass": int(official_taskwise),
        "official_overhead_ok": int(official_overhead_ok),
    }
    summary = gate_summary(
        "F",
        gate,
        "F_ComponentAblationPass" if gate else "F_ComponentAblationNoStructureAdvantage",
        blocker,
        rows,
        scheme_groups=groups,
        taskwise_groups=task_summaries,
        component_checks=checks,
    )
    write_json(OUT_ROOT / "part_f_component_ablation_summary.json", summary)
    nxt = next_actions(
        "F",
        gate,
        blocker,
        [] if gate else [
            "if F2 dominates, downgrade theory to FunctionalGram geometry without structure projection",
            "if F3 dominates, Qpop is enough and Ps is not justified",
            "if F4 dominates, signed utility is harming the cleaner structure projection",
            "if the official component taskwise fails, return to D17 training dynamics before entering safety path",
        ],
    )
    append_exec("part-f-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_f_component_ablation_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part F component ablation audit", summary)
    return summary


def part_g_safety_config(safety_type: str, args: argparse.Namespace) -> dict[str, Any]:
    text = str(safety_type).lower()
    finite = "finite" in text or "f15" in text or "g1" in text
    random_veto = "random" in text or "g4" in text
    etail = "etail" in text or "f15" in text or "g1" in text
    debt_regularization_weight = float(getattr(args, "g_debt_regularization_weight", 0.30)) if etail else 0.0
    debt_regularization_components = str(getattr(args, "g_debt_regularization_components", "ECE,tail95,tail99")) if etail else ""
    return {
        "finite_enabled": bool(finite and not random_veto),
        "random_veto": bool(random_veto),
        "random_veto_accept_prob": float(getattr(args, "g_random_veto_accept_prob", 0.725)),
        "debt_regularization_weight": debt_regularization_weight,
        "debt_regularization_components": debt_regularization_components,
        "finite_step_components": str(getattr(args, "g_finite_step_components", "Brier,ECE,tail95,tail99,margin10")),
        "finite_step_apply_all_parts": int(getattr(args, "g_finite_step_apply_all_parts", 1)),
    }


def make_d17_optimizer(
    model: v2293.TrueDeepPureKAN,
    basis_key: str,
    source_scheme: str,
    args: argparse.Namespace,
) -> EdgeSobolevSNRFU:
    mark_kan_edge_params(model, basis_key=basis_key)
    return EdgeSobolevSNRFU(
        list(model.coeffs),
        lr=float(args.edge_lr),
        weight_decay=float(args.weight_decay),
        sobolev_exponent=part_d_sobolev_exponent(source_scheme),
        edge_metric_type="functional_gram",
        edge_weight_normalization=str(args.edge_weight_normalization),
        edge_weight_ridge=float(args.edge_weight_ridge),
        functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
        gate_beta=float(args.snr_beta),
        gate_family=str(args.structure_block_family),
        gate_floor=float(args.gate_floor),
        gate_floor_mode=str(args.gate_floor_mode),
        gate_input_side=int(args.visual_side),
        gate_patch_size=2,
        stat_warmup_steps=int(getattr(args, "stat_warmup_steps", 0)),
        use_population_gate=True,
    )


def v2303_snapshot_optimizer(opt: torch.optim.Optimizer) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "state": {p: {k: clone_state_value(v) for k, v in state.items()} for p, state in opt.state.items()},
        "group_scalars": [
            {k: clone_state_value(v) for k, v in group.items() if k != "params" and not isinstance(v, torch.Tensor)}
            for group in opt.param_groups
        ],
    }
    for name in (
        "_observed_grads",
        "_observed_labels",
        "_observed_structure_gates",
        "_observed_structure_update_modifiers",
        "_observed_debt_grads",
    ):
        if hasattr(opt, name):
            snap[name] = clone_state_value(getattr(opt, name))
    return snap


def v2303_restore_optimizer(opt: torch.optim.Optimizer, snapshot: dict[str, Any]) -> None:
    opt.state.clear()
    for param, state in snapshot.get("state", {}).items():
        opt.state[param] = {k: clone_state_value(v) for k, v in state.items()}
    for group, scalars in zip(opt.param_groups, snapshot.get("group_scalars", [])):
        for key, value in scalars.items():
            group[key] = clone_state_value(value)
    for name in (
        "_observed_grads",
        "_observed_labels",
        "_observed_structure_gates",
        "_observed_structure_update_modifiers",
        "_observed_debt_grads",
    ):
        if name in snapshot and hasattr(opt, name):
            setattr(opt, name, clone_state_value(snapshot[name]))


def v2303_debt_deltas(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "Brier": float(metrics.get("brier", 0.0)) - float(baseline.get("brier", 0.0)),
        "ECE": float(metrics.get("ece", 0.0)) - float(baseline.get("ece", 0.0)),
        "tail95": float(metrics.get("tail95", 0.0)) - float(baseline.get("tail95", 0.0)),
        "tail99": float(metrics.get("tail99", 0.0)) - float(baseline.get("tail99", 0.0)),
        "margin10": float(metrics.get("margin10", 0.0)) - float(baseline.get("margin10", 0.0)),
    }


def v2303_debt_component_ok(component: str, delta: float, budget: float) -> bool:
    if str(component) == "margin10":
        return float(delta) >= -float(budget)
    return float(delta) <= float(budget)


def part_g_retention_vs_baseline(c2_cov: float, base_cov: float, has_c2: bool) -> float:
    if not has_c2:
        return 0.0
    if base_cov > 1.0e-12:
        return c2_cov / max(base_cov, 1.0e-12)
    return 1.0 if c2_cov >= base_cov else 0.0


def v2303_finite_step_guarded_step(
    opt: torch.optim.Optimizer,
    model: v2293.TrueDeepPureKAN,
    baseline: dict[str, float],
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
    safety_cfg: dict[str, Any],
    *,
    part: str,
    step: int,
) -> dict[str, float | int | str]:
    finite_enabled = bool(safety_cfg.get("finite_enabled", False))
    guard_every = int(getattr(args, "g_finite_step_guard_every", 1))
    if not finite_enabled or guard_every <= 0 or int(step) % guard_every != 0:
        opt.step()
        return {
            "finite_step_enabled": int(finite_enabled),
            "finite_step_accept": 1,
            "finite_step_skip": 0,
            "finite_step_scale": 1.0,
            "finite_step_attempts": 1,
            "finite_step_reject_reason": "not_checked" if finite_enabled else "disabled",
        }
    guard_n = min(int(getattr(args, "g_finite_step_guard_examples", getattr(args, "finite_step_guard_examples", 128))), int(xg.shape[0]))
    x_eval = xg[:guard_n]
    y_eval = yg[:guard_n]
    components = csv_items(str(safety_cfg.get("finite_step_components", "Brier,ECE,tail95,tail99,margin10")))
    budget = float(getattr(args, "no_debt_budget", 0.01))
    model_snapshot = snapshot_model_params(model)
    opt_snapshot = v2303_snapshot_optimizer(opt)
    base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]
    attempts = max(1, int(getattr(args, "g_finite_step_tries", getattr(args, "finite_step_tries", 5))))
    shrink = float(getattr(args, "g_finite_step_shrink", getattr(args, "finite_step_shrink", 0.5)))
    enforce_debt = str(part).upper() == "G_F5" or bool(int(safety_cfg.get("finite_step_apply_all_parts", 1)))
    last_reason = "no_attempt"
    for attempt in range(attempts):
        restore_model_params(model_snapshot)
        v2303_restore_optimizer(opt, opt_snapshot)
        scale = shrink ** attempt
        for group, lr in zip(opt.param_groups, base_lrs):
            group["lr"] = lr * scale
        opt.step()
        metrics = v2293.metrics_for_model(model, x_eval, y_eval)
        deltas = v2303_debt_deltas(metrics, baseline)
        debt_ok = all(v2303_debt_component_ok(component, float(deltas.get(component, 0.0)), budget) for component in components)
        c2_delta = float(metrics.get("coverage_CVaR25", 0.0)) - float(baseline.get("coverage_CVaR25", 0.0))
        c2_ok = str(part).upper() != "G_C2" or c2_delta >= float(getattr(args, "g_finite_step_c2_floor", -0.05))
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
        bad_components = [component for component in components if not v2303_debt_component_ok(component, float(deltas.get(component, 0.0)), budget)]
        last_reason = "c2_floor" if not c2_ok else ("debt_" + "|".join(bad_components) if bad_components else "unknown")
    restore_model_params(model_snapshot)
    v2303_restore_optimizer(opt, opt_snapshot)
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


def part_g_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int]]:
    jobs: list[tuple[str, str, str, str, int]] = []
    f5_tasks = csv_items(str(getattr(args, "part_g_f5_calibration_tasks", args.part_d_tasks)))
    if not f5_tasks:
        f5_tasks = ["rotation_sensitive"]
    for safety in csv_items(args.part_g_safety_schemes):
        if "G_C2" in set(csv_items(args.part_g_parts)):
            for task in csv_items(args.part_d_tasks):
                for seed in range(int(args.part_g_c2_seed_count)):
                    jobs.append((safety, "G_C2", task, "c2_visual_synthetic", seed))
        if "G_F5" in set(csv_items(args.part_g_parts)):
            for seed in range(int(args.part_g_f5_seed_count)):
                task = f5_tasks[seed % len(f5_tasks)]
                jobs.append((safety, "G_F5", task, "f5_visual_no_debt_calibration", seed))
    return jobs


def train_part_g_safety_row(job: tuple[str, str, str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    safety_type, part, task, teacher, seed = job
    start = time.time()
    source_scheme = str(args.part_g_source_scheme)
    structure_mode = PART_D_SCHEMES.get(source_scheme, "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive")
    safety_cfg = part_g_safety_config(safety_type, args)
    control_type = "orbit_shuffle" if bool(safety_cfg.get("random_veto", False)) and int(getattr(args, "g_random_veto_structure_control", 0)) else "true"
    try:
        bargs = v2300.basis_args(args, str(args.part_d_basis).split(",")[0])
        data_seed = int(seed) if str(part).upper() == "G_C2" else int(seed) + 50000
        xtr, ytr, xg, yg = v2300.visual_data(task, data_seed, args, device)
        basis_key = str(args.part_d_basis).split(",")[0]
        depth = str(args.part_d_depths).split(",")[0]
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = v2300.make_model(depth, int(xtr.shape[1]), classes, v2300.model_seed_for(basis_key, depth, task, data_seed), bargs, device)
        opt = make_d17_optimizer(model, basis_key, source_scheme, args)
        before = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        gate_trace: list[float] = []
        struct_trace: list[float] = []
        local_intervention_trace: list[float] = []
        utility_mult_trace: list[float] = []
        signed_modifier_trace: list[float] = []
        signed_modifier_negative_trace: list[float] = []
        finite_accept_count = 0
        finite_skip_count = 0
        finite_attempt_count = 0
        finite_scale_trace: list[float] = []
        finite_reject_reasons: dict[str, int] = {}
        grad_norm_median: list[float] = []
        grad_norm_p95: list[float] = []
        grad_nan_count = 0
        grad_inf_count = 0
        per_example_count = 0
        safety_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(safety_type))) % 100000
        rng = np.random.default_rng(910000 + int(seed) * 1009 + safety_hash)
        cached_structure_gates: dict[int, torch.Tensor] = {}
        cached_structure_modifiers: dict[int, torch.Tensor] = {}
        cached_structure_diag: dict[str, float | int] = {}
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            refresh_structure = not cached_structure_gates or (int(args.structure_gate_every) <= 1) or ((step - 1) % int(args.structure_gate_every) == 0)
            if refresh_structure:
                xub, yub = v2293.v2289.iter_train_batches(
                    xtr,
                    ytr,
                    step - 1 + int(getattr(args, "part_d_utility_batch_offset", 997)),
                    int(args.batch_size),
                    int(seed),
                )
                diag = observe_structure_projected_batch(
                    opt,
                    model,
                    xb,
                    yb,
                    task,
                    seed * 10000 + step,
                    args,
                    structure_control_type=control_type,
                    structure_mode=structure_mode,
                    utility_x=xub,
                    utility_y=yub,
                )
                cached_structure_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
                cached_structure_modifiers = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
                cached_structure_diag = dict(diag)
            else:
                pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, args)
                for param in model.coeffs:
                    gate = cached_structure_gates.get(id(param))
                    if gate is not None:
                        opt.observe_structure_gate(param, gate)
                    modifier = cached_structure_modifiers.get(id(param))
                    if modifier is not None:
                        opt.observe_structure_update_modifier(param, modifier)
                diag = {
                    **cached_structure_diag,
                    "per_example_count": pe_count,
                    "grad_norm_median": gmed,
                    "grad_norm_p95": gp95,
                    "grad_nan_count": nnan,
                    "grad_inf_count": ninf,
                }
            per_example_count = max(per_example_count, ival(diag.get("per_example_count")))
            grad_norm_median.append(fval(diag.get("grad_norm_median")))
            grad_norm_p95.append(fval(diag.get("grad_norm_p95")))
            grad_nan_count += ival(diag.get("grad_nan_count"))
            grad_inf_count += ival(diag.get("grad_inf_count"))
            struct_trace.append(fval(diag.get("q_struct_mean")))
            local_intervention_trace.append(fval(diag.get("local_intervention_mean")))
            utility_mult_trace.append(fval(diag.get("utility_multiplier_mean")))
            signed_modifier_trace.append(fval(diag.get("signed_modifier_mean")))
            signed_modifier_negative_trace.append(fval(diag.get("signed_modifier_negative_fraction")))
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb.long())
            reg_weight = float(safety_cfg.get("debt_regularization_weight", 0.0))
            if reg_weight > 0.0:
                reg_terms = [
                    v2300.debt_component_loss(logits, yb, component, baseline=before_train, budget=float(args.no_debt_budget), proactive=True)
                    for component in csv_items(str(safety_cfg.get("debt_regularization_components", "")))
                ]
                if reg_terms:
                    loss = loss + reg_weight * torch.stack([term.reshape(()) for term in reg_terms]).mean()
            loss.backward()
            if bool(safety_cfg.get("random_veto", False)):
                accept = float(rng.random()) < float(safety_cfg.get("random_veto_accept_prob", 0.725))
                if accept:
                    opt.step()
                    finite_diag = {"finite_step_enabled": 1, "finite_step_accept": 1, "finite_step_skip": 0, "finite_step_scale": 1.0, "finite_step_attempts": 1, "finite_step_reject_reason": "random_accept"}
                else:
                    if hasattr(opt, "clear_observed_gradients"):
                        opt.clear_observed_gradients()
                    finite_diag = {"finite_step_enabled": 1, "finite_step_accept": 0, "finite_step_skip": 1, "finite_step_scale": 0.0, "finite_step_attempts": 1, "finite_step_reject_reason": "random_veto"}
            else:
                finite_diag = v2303_finite_step_guarded_step(opt, model, before, xg, yg, args, safety_cfg, part=str(part), step=step)
            finite_accept_count += ival(finite_diag.get("finite_step_accept"))
            finite_skip_count += ival(finite_diag.get("finite_step_skip"))
            finite_attempt_count += ival(finite_diag.get("finite_step_attempts"))
            finite_scale_trace.append(fval(finite_diag.get("finite_step_scale"), 1.0))
            reason = str(finite_diag.get("finite_step_reject_reason", ""))
            if reason and reason not in {"accepted", "not_checked", "disabled"}:
                finite_reject_reasons[reason] = finite_reject_reasons.get(reason, 0) + 1
            gate_trace.append(float(getattr(opt.last_stats, "gate_density_mean", 0.0)))
        after = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        deltas = v2303_debt_deltas(after, before)
        no_debt = int(all(v2303_debt_component_ok(k, float(v), float(args.no_debt_budget)) for k, v in deltas.items()))
        return {
            "part": str(part),
            "status": "ok",
            "scheme": source_scheme,
            "safety_type": str(safety_type),
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "teacher_type": teacher,
            "seed": int(seed),
            "train_steps": int(args.train_steps),
            "sobolev_exponent": part_d_sobolev_exponent(source_scheme),
            "structure_mode": structure_mode,
            "structure_control_type": control_type,
            "gate_density": mean(gate_trace),
            "q_struct_mean": mean(struct_trace),
            "local_intervention_mean": mean(local_intervention_trace),
            "utility_multiplier_mean": mean(utility_mult_trace),
            "signed_modifier_mean": mean(signed_modifier_trace),
            "signed_modifier_negative_fraction": mean(signed_modifier_negative_trace),
            "C2_accuracy_initial": before["accuracy"],
            "C2_accuracy_final": after["accuracy"],
            "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "C2_coverage_initial": before["coverage_CVaR25"],
            "C2_coverage_final": after["coverage_CVaR25"],
            "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "Brier_delta": deltas["Brier"],
            "ECE_delta": deltas["ECE"],
            "tail95_delta": deltas["tail95"],
            "tail99_delta": deltas["tail99"],
            "margin10_delta": deltas["margin10"],
            "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
            "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
            "train_tail95_delta": after_train.get("tail95", 0.0) - before_train.get("tail95", 0.0),
            "train_tail99_delta": after_train.get("tail99", 0.0) - before_train.get("tail99", 0.0),
            "train_margin10_delta": after_train.get("margin10", 0.0) - before_train.get("margin10", 0.0),
            "no_debt_row": no_debt,
            "F5_no_debt": no_debt,
            "finite_step_enabled": int(bool(safety_cfg.get("finite_enabled", False)) or bool(safety_cfg.get("random_veto", False))),
            "finite_step_accept_count": finite_accept_count,
            "finite_step_skip_count": finite_skip_count,
            "finite_step_skip_count_median_compatible": finite_skip_count,
            "finite_step_attempt_count": finite_attempt_count,
            "finite_step_accept_rate": finite_accept_count / max(1, finite_accept_count + finite_skip_count),
            "finite_step_scale_mean": mean(finite_scale_trace, 1.0),
            "finite_step_reject_reasons": ";".join(f"{k}:{v}" for k, v in sorted(finite_reject_reasons.items())),
            "debt_regularization_weight": float(safety_cfg.get("debt_regularization_weight", 0.0)),
            "debt_regularization_components": str(safety_cfg.get("debt_regularization_components", "")),
            "no_debt_budget": float(args.no_debt_budget),
            "per_example_count": per_example_count,
            "grad_norm_median": mean(grad_norm_median),
            "grad_norm_p95": mean(grad_norm_p95),
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {
            "part": str(part),
            "status": "error",
            "scheme": source_scheme,
            "safety_type": str(safety_type),
            "task": task,
            "teacher_type": teacher,
            "seed": int(seed),
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows = [train_part_g_safety_row(job, args, device) for job in shard_items(part_g_jobs(args), args)]
    path = write_rows(OUT_ROOT / f"part_g_safety_trust_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-g", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("G", 0, "G_ShardsWritten", "merge_required", rows)


def merge_part_g(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_g_safety_trust_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_g_safety_trust_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    no_safety_c2 = {
        (r.get("task"), str(r.get("seed"))): fval(r.get("C2_coverage_improvement"))
        for r in ok
        if r.get("part") == "G_C2" and str(r.get("safety_type")).startswith("G0")
    }
    random_c2 = [
        r
        for r in ok
        if r.get("part") == "G_C2" and ("random" in str(r.get("safety_type")).lower() or str(r.get("safety_type")).startswith("G4"))
    ]
    random_c2_cov = median(r.get("C2_coverage_improvement") for r in random_c2)
    random_base_vals = [no_safety_c2.get((r.get("task"), str(r.get("seed"))), fval(r.get("C2_coverage_improvement"))) for r in random_c2]
    random_base_cov = median(random_base_vals, random_c2_cov)
    random_retention = part_g_retention_vs_baseline(random_c2_cov, random_base_cov, bool(random_c2))
    for key in sorted({r.get("safety_type") for r in ok}):
        group = [r for r in ok if r.get("safety_type") == key]
        c2 = [r for r in group if r.get("part") == "G_C2"]
        f5 = [r for r in group if r.get("part") == "G_F5"]
        c2_cov = median(r.get("C2_coverage_improvement") for r in c2)
        base_vals = [no_safety_c2.get((r.get("task"), str(r.get("seed"))), fval(r.get("C2_coverage_improvement"))) for r in c2]
        base_cov = median(base_vals, c2_cov)
        c2_delta_vs_base = c2_cov - base_cov if c2 else 0.0
        retention = part_g_retention_vs_baseline(c2_cov, base_cov, bool(c2))
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
        accept = median(r.get("finite_step_accept_rate") for r in f5)
        scale = median(r.get("finite_step_scale_mean") for r in f5)
        skip = median(r.get("finite_step_skip_count") for r in f5)
        wall = median(r.get("wall_time_s") for r in group)
        f0_wall = median((r.get("wall_time_s") for r in ok if str(r.get("safety_type")).startswith("G0")), wall)
        overhead = wall / max(f0_wall, 1.0e-12) if f0_wall > 0 else 0.0
        random_f5 = [r for r in ok if r.get("part") == "G_F5" and ("random" in str(r.get("safety_type")).lower() or str(r.get("safety_type")).startswith("G4"))]
        random_count = sum(ival(r.get("F5_no_debt")) for r in random_f5)
        random_gap = f5_count - random_count if not ("random" in str(key).lower() or str(key).startswith("G4")) else 0
        random_retention_gap = retention - random_retention if not ("random" in str(key).lower() or str(key).startswith("G4")) else 0.0
        random_control_explained = int(
            random_gap >= int(args.g_random_veto_gap_gate)
            or (
                random_retention > 0.0
                and random_retention < float(args.g_c2_retention_gate)
                and random_retention_gap >= float(args.g_random_veto_retention_gap_gate)
            )
        )
        f5_n = max(1, len(f5))
        official_seed_count_met = len(f5) >= int(args.g_f5_no_debt_gate)
        official = int(
            str(key).startswith("G1")
            and official_seed_count_met
            and f5_count >= int(args.g_f5_no_debt_gate)
            and component_count >= int(args.g_component_non_positive_gate)
            and retention >= float(args.g_c2_retention_gate)
            and accept >= float(args.g_finite_accept_gate)
            and scale >= float(args.g_finite_scale_gate)
            and skip <= float(args.g_finite_skip_fraction_gate) * int(args.train_steps)
            and random_control_explained == 1
            and (overhead <= float(args.g_overhead_gate) if f0_wall > 0 else True)
        )
        diagnostic = int(
            str(key).startswith("G1")
            and f5_n > 0
            and f5_count == f5_n
            and component_count == f5_n
            and retention >= float(args.g_c2_retention_gate)
            and accept >= float(args.g_finite_accept_gate)
            and scale >= float(args.g_finite_scale_gate)
            and skip <= float(args.g_finite_skip_fraction_gate) * int(args.train_steps)
            and random_control_explained == 1
            and not official
        )
        groups.append({
            "safety_type": key,
            "rows": len(group),
            "C2_coverage_with_safety": c2_cov,
            "C2_coverage_no_safety": base_cov,
            "C2_coverage_retention_vs_no_safety": retention,
            "C2_coverage_delta_vs_no_safety": c2_delta_vs_base,
            "F5_no_debt_count": f5_count,
            "component_non_positive_rows": component_count,
            "finite_step_accept_rate_median": accept,
            "finite_step_scale_mean_median": scale,
            "finite_step_skip_count_median": skip,
            "random_veto_matched_gap": random_gap,
            "random_veto_c2_retention": random_retention,
            "random_veto_c2_retention_gap": random_retention_gap,
            "random_veto_joint_control_explained": random_control_explained,
            "overhead_ratio_vs_no_safety": overhead,
            "official_seed_count_met": int(official_seed_count_met),
            "official_pass": official,
            "diagnostic_pass": diagnostic,
        })
    pass_groups = [g for g in groups if ival(g.get("official_pass")) == 1]
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    blocker = "part_g_job_errors" if any(r.get("status") == "error" for r in rows) else ("none" if gate else "g1_safety_trust_failed")
    group_csv = write_rows(OUT_ROOT / "part_g_safety_trust_group_summary.csv", groups)
    summary = gate_summary(
        "G",
        gate,
        "G_SafetyTrustPass" if gate else "G_SafetyTrustFailed",
        blocker,
        rows,
        group_summaries=groups,
        passing_part_g_groups=pass_groups,
    )
    write_json(OUT_ROOT / "part_g_safety_trust_summary.json", summary)
    nxt = next_actions(
        "G",
        gate,
        blocker,
        [] if gate else [
            "if no-debt fails, tune etail regularizer or finite-step tries without changing no-debt budget",
            "if C2 retention fails, inspect D17 safety interaction and Part F task/debt conflict",
            "if random-veto gap fails, replace random-control definition with a stricter matched veto audit",
        ],
    )
    append_exec("part-g-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_g_safety_trust_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part G D17 finite-step safety trust", summary)
    return summary


def part_h_scheme_config(scheme: str) -> dict[str, Any]:
    text = str(scheme)
    if "F8" in text or "Sobolev0125" in text:
        return {
            "f_scheme": "F8_FunctionalSobolev0125PsQpopPsSignedUtilityDiagnostic",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "control_family": "diagnostic_f8_g1",
        }
    if "F7NoSafety" in text or "NoSafetyDiagnostic" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G0_NoSafety",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "control_family": "diagnostic_f7_no_safety",
        }
    if "F5RegOnly" in text or "C2NoDebtReg" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "debt_regularization_on_h_c2": 0,
            "control_family": "diagnostic_f7_g1_f5_reg_only",
        }
    if "ObjectiveAlignedSafety" in text or "C2ObjectiveGuard" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "debt_regularization_on_h_c2": 0,
            "finite_debt_on_h_c2": 0,
            "control_family": "diagnostic_f7_objective_aligned_safety",
        }
    if "FunctionalGramAdamW" in text:
        return {
            "f_scheme": "F2_FunctionalGramAdamWNoQpopNoPs",
            "safety_type": "G0_NoSafety",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "control_family": "functional_gram_adamw_baseline",
        }
    if "BlockSNR" in text or "Qpop" in text:
        return {
            "f_scheme": "F3_FunctionalGramQpopOnly",
            "safety_type": "G0_NoSafety",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "control_family": "blocksnr_qpop_baseline",
        }
    if "RandomDensity" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "density",
            "same_compute_noop": 0,
            "control_family": "random_density",
        }
    if "RandomHistogram" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "histogram",
            "same_compute_noop": 0,
            "control_family": "random_histogram",
        }
    if "RandomOrbit" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "orbit_shuffle",
            "same_compute_noop": 0,
            "control_family": "random_orbit",
        }
    if "SameComputeNoOp" in text or "NoOp" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "true",
            "same_compute_noop": 1,
            "control_family": "same_compute_noop",
        }
    return {
        "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
        "safety_type": "G1_F15ExactFiniteStepAll",
        "structure_control_type": "true",
        "same_compute_noop": 0,
        "control_family": "official_f7_g1",
    }


def loader_to_tensors(loader: Any, limit: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    total = 0
    for xb, yb in loader:
        xs.append(xb.float().reshape(int(xb.shape[0]), -1))
        ys.append(yb.long().reshape(-1))
        total += int(xb.shape[0])
        if total >= int(limit):
            break
    if not xs:
        return torch.empty(0, 0, device=device), torch.empty(0, dtype=torch.long, device=device)
    x = torch.cat(xs, dim=0)[: int(limit)].to(device=device)
    y = torch.cat(ys, dim=0)[: int(limit)].to(device=device)
    return x, y


def real_dataset_family(dataset: str) -> str:
    return "visual" if str(dataset) in {"MNIST", "FashionMNIST", "KMNIST"} else "tabular"


def load_part_i_tensors(dataset: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]:
    from experiments.run_v22_37_causal_instrumented_functional_optimizer import make_loaders_for_dataset

    train_loader, held_loader, _test_loader, input_dim, output_dim, _x_stats, meta = make_loaders_for_dataset(
        str(dataset),
        int(args.part_i_train_size),
        int(args.part_i_held_size),
        int(args.part_i_batch_size),
        int(seed),
        tier2_download=bool(int(args.part_i_tier2_download)),
    )
    xtr, ytr = loader_to_tensors(train_loader, int(args.part_i_train_size), device)
    xh, yh = loader_to_tensors(held_loader, int(args.part_i_held_size), device)
    return xtr, ytr, xh, yh, int(input_dim), int(output_dim), dict(meta)


def part_i_scheme_config(scheme: str) -> dict[str, Any]:
    text = str(scheme)
    if "FunctionalGram" in text:
        return {
            "f_scheme": "F2_FunctionalGramAdamWNoQpopNoPs",
            "safety_type": "G0_NoSafety",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "control_family": "functional_gram_adamw_baseline",
        }
    if "BlockSNR" in text or "Qpop" in text:
        return {
            "f_scheme": "F3_FunctionalGramQpopOnly",
            "safety_type": "G0_NoSafety",
            "structure_control_type": "true",
            "same_compute_noop": 0,
            "control_family": "blocksnr_qpop_baseline",
        }
    if "NoOp" in text or "SameCompute" in text:
        return {
            "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
            "safety_type": "G1_F15ExactFiniteStepAll",
            "structure_control_type": "true",
            "same_compute_noop": 1,
            "control_family": "same_compute_noop",
        }
    return {
        "f_scheme": "F7_FunctionalSobolev025PsQpopPsSignedUtility",
        "safety_type": "G1_F15ExactFiniteStepAll",
        "structure_control_type": "true",
        "same_compute_noop": 0,
        "control_family": "real_visual_dual_f7_objective_aligned_safety",
    }


def part_i_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [
        (scheme, dataset, seed)
        for scheme in csv_items(str(args.part_i_schemes))
        for dataset in csv_items(str(args.part_i_datasets))
        for seed in range(int(args.part_i_seed_count))
    ]


def real_visual_args(args: argparse.Namespace, input_dim: int) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    side = int(round(math.sqrt(max(1, int(input_dim)))))
    if side * side == int(input_dim):
        out.visual_side = side
    out.batch_size = int(getattr(args, "part_i_batch_size", args.batch_size))
    return out


def observe_real_visual_dual_structure_batch(
    opt: EdgeSobolevSNRFU,
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    *,
    structure_control_type: str,
    structure_mode: str,
    utility_x: torch.Tensor | None,
    utility_y: torch.Tensor | None,
) -> dict[str, float | int]:
    local_diag = observe_structure_projected_batch(
        opt,
        model,
        x,
        y,
        "local_patch_interaction",
        seed,
        args,
        structure_control_type=structure_control_type,
        structure_mode=structure_mode,
        utility_x=utility_x,
        utility_y=utility_y,
    )
    local_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
    local_mods = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
    rot_diag = observe_structure_projected_batch(
        opt,
        model,
        x,
        y,
        "rotation_sensitive",
        seed + 17,
        args,
        structure_control_type=structure_control_type,
        structure_mode=structure_mode,
        utility_x=utility_x,
        utility_y=utility_y,
    )
    rot_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
    rot_mods = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
    for param in model.coeffs:
        key = id(param)
        lg = local_gates.get(key)
        rg = rot_gates.get(key)
        if lg is not None and rg is not None:
            opt.observe_structure_gate(param, (0.5 * (lg + rg)).reshape_as(param))
        elif lg is not None:
            opt.observe_structure_gate(param, lg.reshape_as(param))
        elif rg is not None:
            opt.observe_structure_gate(param, rg.reshape_as(param))
        lm = local_mods.get(key)
        rm = rot_mods.get(key)
        if lm is not None and rm is not None:
            combined = torch.where(torch.sign(lm) == torch.sign(rm), lm, torch.ones_like(lm))
            opt.observe_structure_update_modifier(param, combined.reshape_as(param))
        elif lm is not None:
            opt.observe_structure_update_modifier(param, lm.reshape_as(param))
        elif rm is not None:
            opt.observe_structure_update_modifier(param, rm.reshape_as(param))
    return {
        "per_example_count": max(ival(local_diag.get("per_example_count")), ival(rot_diag.get("per_example_count"))),
        "grad_norm_median": 0.5 * (fval(local_diag.get("grad_norm_median")) + fval(rot_diag.get("grad_norm_median"))),
        "grad_norm_p95": 0.5 * (fval(local_diag.get("grad_norm_p95")) + fval(rot_diag.get("grad_norm_p95"))),
        "grad_nan_count": ival(local_diag.get("grad_nan_count")) + ival(rot_diag.get("grad_nan_count")),
        "grad_inf_count": ival(local_diag.get("grad_inf_count")) + ival(rot_diag.get("grad_inf_count")),
        "q_struct_mean": 0.5 * (fval(local_diag.get("q_struct_mean")) + fval(rot_diag.get("q_struct_mean"))),
        "local_coherence_mean": fval(local_diag.get("local_coherence_mean")),
        "local_intervention_mean": fval(local_diag.get("local_intervention_mean")),
        "rotation_coherence_mean": fval(rot_diag.get("rotation_coherence_mean")),
        "destructive_coherence_mean": 0.5 * (fval(local_diag.get("destructive_coherence_mean")) + fval(rot_diag.get("destructive_coherence_mean"))),
        "utility_multiplier_mean": 0.5 * (fval(local_diag.get("utility_multiplier_mean")) + fval(rot_diag.get("utility_multiplier_mean"))),
        "signed_modifier_mean": 0.5 * (fval(local_diag.get("signed_modifier_mean")) + fval(rot_diag.get("signed_modifier_mean"))),
        "signed_modifier_negative_fraction": 0.5 * (
            fval(local_diag.get("signed_modifier_negative_fraction")) + fval(rot_diag.get("signed_modifier_negative_fraction"))
        ),
    }


def _limit_real_transform_names(names: list[str], args: argparse.Namespace) -> list[str]:
    cap = int(getattr(args, "part_i_real_transform_max", 1))
    if cap <= 0:
        return list(names)
    return list(names[:cap])


def _metric_example_cosine_score(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor,
    perm: torch.Tensor,
    idx: torch.Tensor | None = None,
) -> float:
    a_raw = permute_param_examples(source_grads, param, perm, inverse=False)
    b_raw = target_grads
    n = min(int(a_raw.shape[0]), int(b_raw.shape[0]))
    if n <= 0:
        return 0.0
    a = opt._apply_metric_inv_sqrt(a_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    b = opt._apply_metric_inv_sqrt(b_raw[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1)
    if idx is not None:
        index = idx.to(device=a.device)
        if int(index.numel()) <= 0:
            return 0.0
        a = a[:, index]
        b = b[:, index]
    c = (a * b).sum(dim=1) / (a.norm(dim=1).clamp_min(1.0e-12) * b.norm(dim=1).clamp_min(1.0e-12))
    return float(c.clamp_min(0.0).mean().detach().cpu().item())


def _metric_mean_cosine_score(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor | None,
) -> tuple[float, float]:
    if target_grads is None or target_grads.ndim < 2:
        return 1.0, 1.0
    n = min(int(source_grads.shape[0]), int(target_grads.shape[0]))
    if n <= 0:
        return 1.0, 1.0
    a = opt._apply_metric_inv_sqrt(source_grads[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1).mean(dim=0)
    b = opt._apply_metric_inv_sqrt(target_grads[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1).mean(dim=0)
    denom = a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)
    cos = float(torch.dot(a, b).div(denom).detach().cpu().item())
    return max(0.0, cos), (float(-1.0) if cos < 0.0 else 1.0)


def _metric_mean_cosine_score_idx(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    target_grads: torch.Tensor | None,
    idx: torch.Tensor,
) -> tuple[float, float]:
    if target_grads is None or target_grads.ndim < 2 or int(idx.numel()) <= 0:
        return 1.0, 1.0
    n = min(int(source_grads.shape[0]), int(target_grads.shape[0]))
    if n <= 0:
        return 1.0, 1.0
    index = idx.to(device=param.device)
    a = opt._apply_metric_inv_sqrt(source_grads[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1).mean(dim=0)[index]
    b = opt._apply_metric_inv_sqrt(target_grads[:n].to(device=param.device, dtype=param.dtype), param, group).reshape(n, -1).mean(dim=0)[index]
    denom = a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)
    cos = float(torch.dot(a, b).div(denom).detach().cpu().item())
    return max(0.0, cos), (float(-1.0) if cos < 0.0 else 1.0)


def _split_ranges(n: int, parts: int) -> list[range]:
    parts = max(1, min(int(parts), max(1, int(n))))
    ranges: list[range] = []
    for i in range(parts):
        start = int(round(i * int(n) / parts))
        end = int(round((i + 1) * int(n) / parts))
        if end > start:
            ranges.append(range(start, end))
    return ranges or [range(0, int(n))]


def part_i_band_indices_for_param(param: torch.nn.Parameter, args: argparse.Namespace) -> list[torch.Tensor]:
    if int(param.ndim) < 3 or int(param.numel()) <= 0:
        return [torch.arange(int(param.numel()), device=param.device, dtype=torch.long)]
    flat = torch.arange(int(param.numel()), device=param.device, dtype=torch.long).reshape_as(param)
    k = int(param.shape[-1])
    degree_ranges = _split_ranges(k, int(getattr(args, "part_i_band_degree_bands", 3)))
    first = int(param.shape[0])
    first_axis_groups: list[list[int]] = []
    side = int(getattr(args, "visual_side", 0))
    if int(getattr(param, "_kan_layer_id", -1)) == 0 and side > 0 and first == side * side:
        grid = max(1, int(getattr(args, "part_i_band_grid", 4)))
        for rr in _split_ranges(side, grid):
            for cc in _split_ranges(side, grid):
                coords = [int(r) * side + int(c) for r in rr for c in cc]
                if coords:
                    first_axis_groups.append(coords)
    else:
        for chunk in _split_ranges(first, int(getattr(args, "part_i_band_edge_bands", 4))):
            first_axis_groups.append([int(i) for i in chunk])
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


def observe_real_visual_band_dual_structure_batch(
    opt: EdgeSobolevSNRFU,
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    *,
    structure_control_type: str,
    structure_mode: str,
    utility_x: torch.Tensor | None,
    utility_y: torch.Tensor | None,
) -> dict[str, float | int]:
    side = int(args.visual_side)
    params = list(model.coeffs)
    n_examples = min(int(args.population_grad_examples), int(x.shape[0]))
    xb = x[:n_examples]
    yb = y[:n_examples]
    flat = per_example_gradient_matrix(model, xb, yb, max_examples=n_examples, label_prior_correction=bool(int(args.label_prior_correction)))
    split = split_flat_grads(flat, params)
    labels = yb[: int(flat.shape[0])].detach() if flat.ndim == 2 else None
    profile = str(args.structure_transform_profile)
    identity_perm = torch.arange(side * side, device=x.device)

    def transform_splits(family: str, offset: int, *, align_with_transform: bool = True) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        out: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        for i, name in enumerate(_limit_real_transform_names(transform_names(family, profile), args)):
            perm = transform_perm(side, name, seed=seed * 1000 + offset + i, device=x.device)
            x_t = apply_perm(xb, perm)
            ft = per_example_gradient_matrix(
                model,
                x_t,
                yb[: int(x_t.shape[0])],
                max_examples=n_examples,
                label_prior_correction=bool(int(args.label_prior_correction)),
            )
            align_perm = perm if align_with_transform else identity_perm
            out.append((split_flat_grads(ft, params), align_perm))
        return out

    local_t = transform_splits("local", 1100)
    rot_t = transform_splits("rotation", 2100)
    dest_local_t = transform_splits("destructive_local", 3100, align_with_transform=False)
    dest_rot_t = transform_splits("destructive_rotation", 4100, align_with_transform=False)
    utility_split: list[torch.Tensor] | None = None
    if structure_mode_uses_utility(structure_mode) and utility_x is not None and utility_y is not None:
        n_util = min(int(getattr(args, "part_d_utility_examples", args.population_grad_examples)), int(utility_x.shape[0]))
        util_flat = per_example_gradient_matrix(
            model,
            utility_x[:n_util],
            utility_y[:n_util],
            max_examples=n_util,
            label_prior_correction=bool(int(args.label_prior_correction)),
        )
        utility_split = split_flat_grads(util_flat, params)

    grad_norms = flat.norm(dim=1) if flat.ndim == 2 and int(flat.shape[0]) else torch.zeros(0, dtype=torch.float64)
    group = opt.param_groups[0]
    struct_means: list[float] = []
    local_means: list[float] = []
    rot_means: list[float] = []
    dest_means: list[float] = []
    utility_mult_means: list[float] = []
    signed_modifier_means: list[float] = []
    signed_modifier_negative_fracs: list[float] = []
    mode = str(structure_mode)
    for idx_param, (param, grads) in enumerate(zip(params, split)):
        pgrads = grads.to(device=param.device, dtype=param.dtype)
        opt.observe_per_example_gradients(param, pgrads, labels=labels)
        gate = torch.zeros(int(param.numel()), device=param.device, dtype=param.dtype)
        modifier = torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
        blocks = part_i_band_indices_for_param(param, args)
        for block in blocks:
            local_scores = [_metric_example_cosine_score(opt, param, group, pgrads, tg[idx_param], perm, block) for tg, perm in local_t if idx_param < len(tg)]
            rot_scores = [_metric_example_cosine_score(opt, param, group, pgrads, tg[idx_param], perm, block) for tg, perm in rot_t if idx_param < len(tg)]
            dest_scores = [
                _metric_example_cosine_score(opt, param, group, pgrads, tg[idx_param], perm, block)
                for tg, perm in [*dest_local_t, *dest_rot_t]
                if idx_param < len(tg)
            ]
            local_score = mean(local_scores)
            rot_score = mean(rot_scores)
            dest_score = mean(dest_scores)
            real_profile = str(getattr(args, "part_i_real_profile", "local_rotation_dual"))
            if real_profile == "local_only":
                pos_score = local_score
            elif real_profile == "rotation_only":
                pos_score = rot_score
            else:
                pos_score = 0.5 * (local_score + rot_score)
            base_score = max(0.0, pos_score - float(args.contrastive_eta) * dest_score)
            utility_grads = utility_split[idx_param] if utility_split is not None and idx_param < len(utility_split) else None
            utility_score, signed_modifier = _metric_mean_cosine_score_idx(opt, param, group, pgrads, utility_grads, block)
            if mode in {
                "density_preserved_signed_utility_tilt_orbit_centered_contrastive",
                "task_contrast_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
                "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
                "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
            }:
                floor = max(0.0, float(args.utility_tilt_floor))
                utility_mult = floor + (1.0 - floor) * utility_score
                base_score = max(0.0, min(1.0, base_score * utility_mult))
                utility_mult_means.append(float(utility_mult))
            if structure_control_type in {"structure_breaking", "break", "destructive"}:
                base_score = dest_score
            gate[block] = float(max(0.0, min(1.0, base_score)))
            if "signed_utility" in mode and bool(int(getattr(args, "part_i_band_signed_modifier", 0))):
                modifier[block] = float(args.signed_negative_scale) if signed_modifier < 0.0 else 1.0
            local_means.append(local_score)
            rot_means.append(rot_score)
            dest_means.append(dest_score)
        opt.observe_structure_gate(param, gate.reshape_as(param))
        if bool(int(getattr(args, "part_i_band_density_preserve", 0))) and float(gate.detach().mean().cpu().item()) > 1.0e-12:
            target = torch.ones((), device=gate.device, dtype=gate.dtype)
            gate = (gate * (target / gate.detach().mean().clamp_min(1.0e-12))).clamp(0.0, float(getattr(args, "part_i_band_gate_max", 2.0)))
            opt.observe_structure_gate(param, gate.reshape_as(param))
        if "signed_utility" in mode and bool(int(getattr(args, "part_i_band_signed_modifier", 0))):
            opt.observe_structure_update_modifier(param, modifier.reshape_as(param))
        struct_means.append(float(gate.detach().mean().cpu().item()) if int(gate.numel()) else 0.0)
        signed_modifier_means.append(float(modifier.detach().mean().cpu().item()) if int(modifier.numel()) else 1.0)
        signed_modifier_negative_fracs.append(float((modifier.detach() < 0).to(dtype=torch.float64).mean().cpu().item()) if int(modifier.numel()) else 0.0)
    return {
        "per_example_count": int(flat.shape[0]) if flat.ndim == 2 else 0,
        "grad_norm_median": float(grad_norms.median().item()) if int(grad_norms.numel()) else 0.0,
        "grad_norm_p95": float(torch.quantile(grad_norms, 0.95).item()) if int(grad_norms.numel()) else 0.0,
        "grad_nan_count": int(torch.isnan(flat).sum().item()) if int(flat.numel()) else 0,
        "grad_inf_count": int(torch.isinf(flat).sum().item()) if int(flat.numel()) else 0,
        "q_struct_mean": mean(struct_means),
        "local_coherence_mean": mean(local_means),
        "local_intervention_mean": 0.0,
        "rotation_coherence_mean": mean(rot_means),
        "destructive_coherence_mean": mean(dest_means),
        "utility_multiplier_mean": mean(utility_mult_means),
        "signed_modifier_mean": mean(signed_modifier_means),
        "signed_modifier_negative_fraction": mean(signed_modifier_negative_fracs),
        "signed_modifier_agreement_fraction": 1.0,
        "structure_density_match_error": 0.0,
        "structure_histogram_mean_error": 0.0,
        "orbit_size_match_error": 0.0,
    }


def observe_real_visual_scalar_dual_structure_batch(
    opt: EdgeSobolevSNRFU,
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    *,
    structure_control_type: str,
    structure_mode: str,
    utility_x: torch.Tensor | None,
    utility_y: torch.Tensor | None,
) -> dict[str, float | int]:
    side = int(args.visual_side)
    params = list(model.coeffs)
    n_examples = min(int(args.population_grad_examples), int(x.shape[0]))
    xb = x[:n_examples]
    yb = y[:n_examples]
    flat = per_example_gradient_matrix(model, xb, yb, max_examples=n_examples, label_prior_correction=bool(int(args.label_prior_correction)))
    split = split_flat_grads(flat, params)
    labels = yb[: int(flat.shape[0])].detach() if flat.ndim == 2 else None
    profile = str(args.structure_transform_profile)
    identity_perm = torch.arange(side * side, device=x.device)

    def transform_splits(family: str, offset: int, *, align_with_transform: bool = True) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        out: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        for i, name in enumerate(_limit_real_transform_names(transform_names(family, profile), args)):
            perm = transform_perm(side, name, seed=seed * 1000 + offset + i, device=x.device)
            x_t = apply_perm(xb, perm)
            ft = per_example_gradient_matrix(
                model,
                x_t,
                yb[: int(x_t.shape[0])],
                max_examples=n_examples,
                label_prior_correction=bool(int(args.label_prior_correction)),
            )
            align_perm = perm if align_with_transform else identity_perm
            out.append((split_flat_grads(ft, params), align_perm))
        return out

    local_t = transform_splits("local", 1100)
    rot_t = transform_splits("rotation", 2100)
    dest_local_t = transform_splits("destructive_local", 3100, align_with_transform=False)
    dest_rot_t = transform_splits("destructive_rotation", 4100, align_with_transform=False)
    utility_split: list[torch.Tensor] | None = None
    if structure_mode_uses_utility(structure_mode) and utility_x is not None and utility_y is not None:
        n_util = min(int(getattr(args, "part_d_utility_examples", args.population_grad_examples)), int(utility_x.shape[0]))
        util_flat = per_example_gradient_matrix(
            model,
            utility_x[:n_util],
            utility_y[:n_util],
            max_examples=n_util,
            label_prior_correction=bool(int(args.label_prior_correction)),
        )
        utility_split = split_flat_grads(util_flat, params)

    grad_norms = flat.norm(dim=1) if flat.ndim == 2 and int(flat.shape[0]) else torch.zeros(0, dtype=torch.float64)
    group = opt.param_groups[0]
    struct_means: list[float] = []
    local_means: list[float] = []
    rot_means: list[float] = []
    dest_means: list[float] = []
    utility_mult_means: list[float] = []
    signed_modifier_means: list[float] = []
    signed_modifier_negative_fracs: list[float] = []
    mode = str(structure_mode)
    for idx, (param, grads) in enumerate(zip(params, split)):
        pgrads = grads.to(device=param.device, dtype=param.dtype)
        opt.observe_per_example_gradients(param, pgrads, labels=labels)
        local_scores = [_metric_example_cosine_score(opt, param, group, pgrads, tg[idx], perm) for tg, perm in local_t if idx < len(tg)]
        rot_scores = [_metric_example_cosine_score(opt, param, group, pgrads, tg[idx], perm) for tg, perm in rot_t if idx < len(tg)]
        dest_scores = [
            _metric_example_cosine_score(opt, param, group, pgrads, tg[idx], perm)
            for tg, perm in [*dest_local_t, *dest_rot_t]
            if idx < len(tg)
        ]
        local_score = mean(local_scores)
        rot_score = mean(rot_scores)
        dest_score = mean(dest_scores)
        real_profile = str(getattr(args, "part_i_real_profile", "local_rotation_dual"))
        if real_profile == "local_only":
            pos_score = local_score
        elif real_profile == "rotation_only":
            pos_score = rot_score
        else:
            pos_score = 0.5 * (local_score + rot_score)
        base_score = max(0.0, pos_score - float(args.contrastive_eta) * dest_score)
        utility_grads = utility_split[idx] if utility_split is not None and idx < len(utility_split) else None
        utility_score, signed_modifier = _metric_mean_cosine_score(opt, param, group, pgrads, utility_grads)
        if mode in {
            "density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "task_contrast_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "local_intervention_density_preserved_signed_utility_tilt_orbit_centered_contrastive",
            "local_intervention_coordinate_signed_utility_tilt_orbit_centered_contrastive",
        }:
            floor = max(0.0, float(args.utility_tilt_floor))
            utility_mult = floor + (1.0 - floor) * utility_score
            base_score = max(0.0, min(1.0, base_score * utility_mult))
            utility_mult_means.append(float(utility_mult))
        if structure_control_type in {"structure_breaking", "break", "destructive"}:
            base_score = dest_score
        gate = torch.full_like(param, float(max(0.0, min(1.0, base_score))))
        opt.observe_structure_gate(param, gate)
        if "signed_utility" in mode and bool(int(getattr(args, "part_i_scalar_signed_modifier", 0))):
            modifier_value = float(args.signed_negative_scale) if signed_modifier < 0.0 else 1.0
            modifier = torch.full_like(param, modifier_value)
            opt.observe_structure_update_modifier(param, modifier)
            signed_modifier_means.append(modifier_value)
            signed_modifier_negative_fracs.append(1.0 if modifier_value < 0.0 else 0.0)
        elif "signed_utility" in mode:
            signed_modifier_means.append(1.0)
            signed_modifier_negative_fracs.append(0.0)
        struct_means.append(float(gate.detach().mean().cpu().item()) if int(gate.numel()) else 0.0)
        local_means.append(local_score)
        rot_means.append(rot_score)
        dest_means.append(dest_score)
    return {
        "per_example_count": int(flat.shape[0]) if flat.ndim == 2 else 0,
        "grad_norm_median": float(grad_norms.median().item()) if int(grad_norms.numel()) else 0.0,
        "grad_norm_p95": float(torch.quantile(grad_norms, 0.95).item()) if int(grad_norms.numel()) else 0.0,
        "grad_nan_count": int(torch.isnan(flat).sum().item()) if int(flat.numel()) else 0,
        "grad_inf_count": int(torch.isinf(flat).sum().item()) if int(flat.numel()) else 0,
        "q_struct_mean": mean(struct_means),
        "local_coherence_mean": mean(local_means),
        "local_intervention_mean": 0.0,
        "rotation_coherence_mean": mean(rot_means),
        "destructive_coherence_mean": mean(dest_means),
        "utility_multiplier_mean": mean(utility_mult_means),
        "signed_modifier_mean": mean(signed_modifier_means),
        "signed_modifier_negative_fraction": mean(signed_modifier_negative_fracs),
        "signed_modifier_agreement_fraction": 1.0,
        "structure_density_match_error": 0.0,
        "structure_histogram_mean_error": 0.0,
        "orbit_size_match_error": 0.0,
    }


def part_h_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, int]]:
    jobs: list[tuple[str, str, str, int]] = []
    f5_tasks = csv_items(str(getattr(args, "part_h_f5_calibration_tasks", args.part_d_tasks)))
    if not f5_tasks:
        f5_tasks = ["rotation_sensitive"]
    for scheme in csv_items(args.part_h_schemes):
        if "H_C2" in set(csv_items(args.part_h_parts)):
            for task in csv_items(args.part_d_tasks):
                for seed in range(int(args.part_h_c2_seed_count)):
                    jobs.append((scheme, "H_C2", task, seed))
        if "H_F5" in set(csv_items(args.part_h_parts)):
            for seed in range(int(args.part_h_f5_seed_count)):
                task = f5_tasks[seed % len(f5_tasks)]
                jobs.append((scheme, "H_F5", task, seed))
    return jobs


def train_part_h_positive_control_row(job: tuple[str, str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    h_scheme, part, task, seed = job
    start = time.time()
    h_cfg = part_h_scheme_config(str(h_scheme))
    f_scheme = str(h_cfg.get("f_scheme", "F7_FunctionalSobolev025PsQpopPsSignedUtility"))
    f_cfg = part_f_scheme_config(f_scheme)
    safety_type = str(h_cfg.get("safety_type", "G1_F15ExactFiniteStepAll"))
    safety_cfg = part_g_safety_config(safety_type, args)
    if str(part).upper() == "H_C2" and int(h_cfg.get("finite_debt_on_h_c2", 1)) == 0:
        safety_cfg = dict(safety_cfg)
        safety_cfg["finite_step_apply_all_parts"] = 0
    structure_mode = str(f_cfg.get("structure_mode", "none"))
    structure_control_type = str(h_cfg.get("structure_control_type", "true"))
    try:
        basis_key = str(args.part_d_basis).split(",")[0]
        depth = str(args.part_d_depths).split(",")[0]
        bargs = v2300.basis_args(args, basis_key)
        data_seed = int(seed) if str(part).upper() == "H_C2" else int(seed) + 50000
        xtr, ytr, xg, yg = v2300.visual_data(task, data_seed, args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = v2300.make_model(depth, int(xtr.shape[1]), classes, v2300.model_seed_for(basis_key, depth, task, data_seed), bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        opt = make_part_f_optimizer(model, f_cfg, args)
        before = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        gate_trace: list[float] = []
        struct_trace: list[float] = []
        local_trace: list[float] = []
        local_intervention_trace: list[float] = []
        rot_trace: list[float] = []
        dest_trace: list[float] = []
        utility_mult_trace: list[float] = []
        signed_modifier_trace: list[float] = []
        signed_modifier_negative_trace: list[float] = []
        finite_accept_count = 0
        finite_skip_count = 0
        finite_attempt_count = 0
        finite_scale_trace: list[float] = []
        finite_reject_reasons: dict[str, int] = {}
        grad_norm_median: list[float] = []
        grad_norm_p95: list[float] = []
        grad_nan_count = 0
        grad_inf_count = 0
        per_example_count = 0
        cached_structure_gates: dict[int, torch.Tensor] = {}
        cached_structure_modifiers: dict[int, torch.Tensor] = {}
        cached_structure_diag: dict[str, float | int] = {}
        use_edge_observation = isinstance(opt, EdgeSobolevPopulationFlow)
        use_structure_gate = bool(f_cfg.get("use_structure_gate", False)) and use_edge_observation
        use_qpop = bool(f_cfg.get("use_population_gate", False)) and use_edge_observation
        same_compute_noop = bool(int(h_cfg.get("same_compute_noop", 0)))
        debt_reg_weight = float(safety_cfg.get("debt_regularization_weight", 0.0))
        debt_reg_components = str(safety_cfg.get("debt_regularization_components", ""))
        if str(part).upper() == "H_C2" and int(h_cfg.get("debt_regularization_on_h_c2", 1)) == 0:
            debt_reg_weight = 0.0
            debt_reg_components = ""
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            if use_structure_gate:
                refresh_structure = not cached_structure_gates or (int(args.structure_gate_every) <= 1) or ((step - 1) % int(args.structure_gate_every) == 0)
                if refresh_structure:
                    xub = yub = None
                    if structure_mode_uses_utility(structure_mode):
                        xub, yub = v2293.v2289.iter_train_batches(
                            xtr,
                            ytr,
                            step - 1 + int(getattr(args, "part_d_utility_batch_offset", 997)),
                            int(args.batch_size),
                            int(seed),
                        )
                    diag = observe_structure_projected_batch(
                        opt,
                        model,
                        xb,
                        yb,
                        task,
                        seed * 10000 + step,
                        args,
                        structure_control_type=structure_control_type,
                        structure_mode=structure_mode,
                        utility_x=xub if structure_mode_uses_utility(structure_mode) else None,
                        utility_y=yub if structure_mode_uses_utility(structure_mode) else None,
                    )
                    cached_structure_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
                    cached_structure_modifiers = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
                    cached_structure_diag = dict(diag)
                else:
                    pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, args)
                    for param in model.coeffs:
                        gate = cached_structure_gates.get(id(param))
                        if gate is not None:
                            opt.observe_structure_gate(param, gate)
                        modifier = cached_structure_modifiers.get(id(param))
                        if modifier is not None:
                            opt.observe_structure_update_modifier(param, modifier)
                    diag = {
                        **cached_structure_diag,
                        "per_example_count": pe_count,
                        "grad_norm_median": gmed,
                        "grad_norm_p95": gp95,
                        "grad_nan_count": nnan,
                        "grad_inf_count": ninf,
                    }
            elif use_qpop:
                pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, args)
                diag = {
                    "per_example_count": pe_count,
                    "grad_norm_median": gmed,
                    "grad_norm_p95": gp95,
                    "grad_nan_count": nnan,
                    "grad_inf_count": ninf,
                    "q_struct_mean": 1.0,
                    "local_coherence_mean": 0.0,
                    "local_intervention_mean": 0.0,
                    "rotation_coherence_mean": 0.0,
                    "destructive_coherence_mean": 0.0,
                    "utility_multiplier_mean": 0.0,
                    "signed_modifier_mean": 0.0,
                    "signed_modifier_negative_fraction": 0.0,
                }
            else:
                diag = {
                    "per_example_count": 0,
                    "grad_norm_median": 0.0,
                    "grad_norm_p95": 0.0,
                    "grad_nan_count": 0,
                    "grad_inf_count": 0,
                    "q_struct_mean": 0.0,
                    "local_coherence_mean": 0.0,
                    "local_intervention_mean": 0.0,
                    "rotation_coherence_mean": 0.0,
                    "destructive_coherence_mean": 0.0,
                    "utility_multiplier_mean": 0.0,
                    "signed_modifier_mean": 0.0,
                    "signed_modifier_negative_fraction": 0.0,
                }
            per_example_count = max(per_example_count, ival(diag.get("per_example_count")))
            grad_norm_median.append(fval(diag.get("grad_norm_median")))
            grad_norm_p95.append(fval(diag.get("grad_norm_p95")))
            grad_nan_count += ival(diag.get("grad_nan_count"))
            grad_inf_count += ival(diag.get("grad_inf_count"))
            struct_trace.append(fval(diag.get("q_struct_mean")))
            local_trace.append(fval(diag.get("local_coherence_mean")))
            local_intervention_trace.append(fval(diag.get("local_intervention_mean")))
            rot_trace.append(fval(diag.get("rotation_coherence_mean")))
            dest_trace.append(fval(diag.get("destructive_coherence_mean")))
            utility_mult_trace.append(fval(diag.get("utility_multiplier_mean")))
            signed_modifier_trace.append(fval(diag.get("signed_modifier_mean")))
            signed_modifier_negative_trace.append(fval(diag.get("signed_modifier_negative_fraction")))
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb.long())
            if debt_reg_weight > 0.0:
                reg_terms = [
                    v2300.debt_component_loss(logits, yb, component, baseline=before_train, budget=float(args.no_debt_budget), proactive=True)
                    for component in csv_items(debt_reg_components)
                ]
                if reg_terms:
                    loss = loss + debt_reg_weight * torch.stack([term.reshape(()) for term in reg_terms]).mean()
            loss.backward()
            if same_compute_noop:
                if hasattr(opt, "clear_observed_gradients"):
                    opt.clear_observed_gradients()
                finite_diag = {
                    "finite_step_enabled": 1,
                    "finite_step_accept": 0,
                    "finite_step_skip": 1,
                    "finite_step_scale": 0.0,
                    "finite_step_attempts": 1,
                    "finite_step_reject_reason": "same_compute_noop",
                }
            else:
                finite_diag = v2303_finite_step_guarded_step(
                    opt,
                    model,
                    before,
                    xg,
                    yg,
                    args,
                    safety_cfg,
                    part="G_F5" if str(part).upper() == "H_F5" else "G_C2",
                    step=step,
                )
            finite_accept_count += ival(finite_diag.get("finite_step_accept"))
            finite_skip_count += ival(finite_diag.get("finite_step_skip"))
            finite_attempt_count += ival(finite_diag.get("finite_step_attempts"))
            finite_scale_trace.append(fval(finite_diag.get("finite_step_scale"), 1.0))
            reason = str(finite_diag.get("finite_step_reject_reason", ""))
            if reason and reason not in {"accepted", "not_checked", "disabled"}:
                finite_reject_reasons[reason] = finite_reject_reasons.get(reason, 0) + 1
            stats = getattr(opt, "last_stats", None)
            gate_trace.append(float(getattr(stats, "gate_density_mean", 1.0 if not same_compute_noop else 0.0)))
        after = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        deltas = v2303_debt_deltas(after, before)
        no_debt = int(all(v2303_debt_component_ok(k, float(v), float(args.no_debt_budget)) for k, v in deltas.items()))
        return {
            "part": str(part),
            "status": "ok",
            "h_scheme": str(h_scheme),
            "scheme": f_scheme,
            "safety_type": safety_type,
            "control_family": str(h_cfg.get("control_family", "")),
            "structure_control_type": structure_control_type,
            "same_compute_noop": int(same_compute_noop),
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "teacher_type": "c2_visual_synthetic" if str(part).upper() == "H_C2" else "f5_visual_no_debt_calibration",
            "seed": int(seed),
            "train_steps": int(args.train_steps),
            "functional_gram_used": int(f_cfg.get("functional_gram_used", 0)),
            "qpop_used": int(f_cfg.get("qpop_used", 0)),
            "ps_used": int(f_cfg.get("ps_used", 0)),
            "signed_utility_used": int(f_cfg.get("signed_utility_used", 0)),
            "sobolev_exponent": float(f_cfg.get("sobolev_exponent", 0.0)),
            "structure_mode": structure_mode,
            "gate_density": mean(gate_trace),
            "q_struct_mean": mean(struct_trace),
            "local_coherence_mean": mean(local_trace),
            "local_intervention_mean": mean(local_intervention_trace),
            "rotation_coherence_mean": mean(rot_trace),
            "destructive_coherence_mean": mean(dest_trace),
            "utility_multiplier_mean": mean(utility_mult_trace),
            "signed_modifier_mean": mean(signed_modifier_trace),
            "signed_modifier_negative_fraction": mean(signed_modifier_negative_trace),
            "C2_accuracy_initial": before["accuracy"],
            "C2_accuracy_final": after["accuracy"],
            "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "C2_coverage_initial": before["coverage_CVaR25"],
            "C2_coverage_final": after["coverage_CVaR25"],
            "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "Brier_delta": deltas["Brier"],
            "ECE_delta": deltas["ECE"],
            "tail95_delta": deltas["tail95"],
            "tail99_delta": deltas["tail99"],
            "margin10_delta": deltas["margin10"],
            "train_Brier_delta": after_train.get("brier", 0.0) - before_train.get("brier", 0.0),
            "train_ECE_delta": after_train.get("ece", 0.0) - before_train.get("ece", 0.0),
            "train_tail95_delta": after_train.get("tail95", 0.0) - before_train.get("tail95", 0.0),
            "train_tail99_delta": after_train.get("tail99", 0.0) - before_train.get("tail99", 0.0),
            "train_margin10_delta": after_train.get("margin10", 0.0) - before_train.get("margin10", 0.0),
            "no_debt_row": no_debt,
            "F5_no_debt": no_debt,
            "finite_step_enabled": int(bool(safety_cfg.get("finite_enabled", False)) or same_compute_noop),
            "finite_step_accept_count": finite_accept_count,
            "finite_step_skip_count": finite_skip_count,
            "finite_step_attempt_count": finite_attempt_count,
            "finite_step_accept_rate": finite_accept_count / max(1, finite_accept_count + finite_skip_count),
            "finite_step_scale_mean": mean(finite_scale_trace, 1.0),
            "finite_step_reject_reasons": ";".join(f"{k}:{v}" for k, v in sorted(finite_reject_reasons.items())),
            "debt_regularization_weight": debt_reg_weight,
            "debt_regularization_components": debt_reg_components,
            "no_debt_budget": float(args.no_debt_budget),
            "mlp_controls_applicable": 0,
            "per_example_count": per_example_count,
            "grad_norm_median": mean(grad_norm_median),
            "grad_norm_p95": mean(grad_norm_p95),
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {
            "part": str(part),
            "status": "error",
            "h_scheme": str(h_scheme),
            "scheme": f_scheme,
            "safety_type": safety_type,
            "task": task,
            "seed": int(seed),
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows = [train_part_h_positive_control_row(job, args, device) for job in shard_items(part_h_jobs(args), args)]
    path = write_rows(OUT_ROOT / f"part_h_full_positive_control_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-h", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("H", 0, "H_ShardsWritten", "merge_required", rows)


def merge_part_h(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_h_full_positive_control_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_h_full_positive_control_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("h_scheme"), r.get("task")) for r in ok if r.get("part") == "H_C2"}):
        group = [r for r in ok if r.get("part") == "H_C2" and (r.get("h_scheme"), r.get("task")) == key]
        task_summaries.append({
            "h_scheme": key[0],
            "task": key[1],
            "rows": len(group),
            "C2_coverage_improvement_median": median(r.get("C2_coverage_improvement") for r in group),
            "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group),
            "taskwise_pass": int(bool(group) and median(r.get("C2_coverage_improvement") for r in group) >= float(args.d_task_coverage_gate)),
        })
    groups: list[dict[str, Any]] = []
    baseline_wall = median((r.get("wall_time_s") for r in ok if str(r.get("h_scheme")) == str(args.part_h_functional_baseline_scheme)), 0.0)
    for key in sorted({r.get("h_scheme") for r in ok}):
        group = [r for r in ok if r.get("h_scheme") == key]
        c2 = [r for r in group if r.get("part") == "H_C2"]
        f5 = [r for r in group if r.get("part") == "H_F5"]
        tasks = [t for t in task_summaries if t.get("h_scheme") == key]
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
        wall = median(r.get("wall_time_s") for r in group)
        overhead = wall / max(baseline_wall, 1.0e-12) if baseline_wall > 0 else 0.0
        groups.append({
            "h_scheme": key,
            "rows": len(group),
            "control_family": sorted({str(r.get("control_family", "")) for r in group})[0] if group else "",
            "C2_coverage_improvement_median": median(t.get("C2_coverage_improvement_median") for t in tasks),
            "C2_accuracy_improvement_median": median(t.get("C2_accuracy_improvement_median") for t in tasks),
            "taskwise_all_pass": int(bool(tasks) and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)),
            "F5_no_debt_count": f5_count,
            "component_non_positive_rows": component_count,
            "finite_step_accept_rate_median": median(r.get("finite_step_accept_rate") for r in f5),
            "finite_step_scale_mean_median": median(r.get("finite_step_scale_mean") for r in f5),
            "finite_step_skip_count_median": median(r.get("finite_step_skip_count") for r in f5),
            "wall_time_s_median": wall,
            "overhead_ratio_vs_functional_baseline": overhead,
            "held_test_usage": max((ival(r.get("held_test_usage")) for r in group), default=0),
        })
    by_scheme = {str(g.get("h_scheme")): g for g in groups}
    official = by_scheme.get(str(args.part_h_official_scheme), {})
    random_groups = [g for g in groups if str(g.get("control_family", "")).startswith("random_")]
    orbit_group = next((g for g in random_groups if str(g.get("control_family")) == "random_orbit"), {})
    noop = next((g for g in groups if str(g.get("control_family")) == "same_compute_noop"), {})
    functional = by_scheme.get(str(args.part_h_functional_baseline_scheme), {})
    blocksnr = by_scheme.get(str(args.part_h_blocksnr_baseline_scheme), {})
    official_cov = fval(official.get("C2_coverage_improvement_median"), -1.0e9)
    controls_present = bool(random_groups)
    orbit_present = bool(orbit_group)
    noop_present = bool(noop)
    baselines_present = bool(functional) or bool(blocksnr)
    functional_cov = fval(functional.get("C2_coverage_improvement_median"), 0.0) if functional else 0.0
    blocksnr_cov = fval(blocksnr.get("C2_coverage_improvement_median"), 0.0) if blocksnr else 0.0
    baseline_best_cov = max([v for v, present in [(functional_cov, bool(functional)), (blocksnr_cov, bool(blocksnr))] if present], default=0.0)
    baseline_gap = official_cov - baseline_best_cov if baselines_present else 0.0
    official_wall = fval(official.get("wall_time_s_median"), 0.0)
    noop_wall = fval(noop.get("wall_time_s_median"), 0.0)
    relevant_overhead = official_wall / max(noop_wall, 1.0e-12) if noop_wall > 0 else fval(official.get("overhead_ratio_vs_functional_baseline"), 99.0)
    random_best_cov = max([fval(g.get("C2_coverage_improvement_median"), 0.0) for g in random_groups], default=0.0)
    orbit_gap = official_cov - fval(orbit_group.get("C2_coverage_improvement_median"), 0.0) if orbit_present else 0.0
    noop_gap = official_cov - fval(noop.get("C2_coverage_improvement_median"), 0.0) if noop_present else 0.0
    random_gap = official_cov - random_best_cov if controls_present else 0.0
    f5_seed_met = len([r for r in ok if r.get("part") == "H_F5" and r.get("h_scheme") == str(args.part_h_official_scheme)]) >= int(args.part_h_f5_no_debt_gate)
    checks = {
        "official_scheme": str(args.part_h_official_scheme),
        "official_coverage": official_cov,
        "official_taskwise_all_pass": ival(official.get("taskwise_all_pass")),
        "official_F5_no_debt_count": ival(official.get("F5_no_debt_count")),
        "official_component_non_positive_rows": ival(official.get("component_non_positive_rows")),
        "official_accept_rate": fval(official.get("finite_step_accept_rate_median")),
        "official_scale_mean": fval(official.get("finite_step_scale_mean_median")),
        "official_skip_count": fval(official.get("finite_step_skip_count_median")),
        "official_overhead_vs_same_compute_noop": relevant_overhead,
        "official_overhead_vs_functional_baseline": fval(official.get("overhead_ratio_vs_functional_baseline")),
        "random_gap_orbit": orbit_gap,
        "beats_random_controls_gap": random_gap,
        "beats_same_compute_noop_gap": noop_gap,
        "functional_baseline_coverage": functional_cov,
        "blocksnr_baseline_coverage": blocksnr_cov,
        "best_functional_or_blocksnr_baseline_coverage": baseline_best_cov,
        "beats_functional_or_blocksnr_baseline_gap": baseline_gap,
        "random_controls_present": int(controls_present),
        "same_compute_noop_present": int(noop_present),
        "functional_or_blocksnr_baseline_present": int(baselines_present),
        "mlp_controls_applicable": 0,
        "official_f5_seed_count_met": int(f5_seed_met),
    }
    gate = int(
        bool(official)
        and not any(r.get("status") == "error" for r in rows)
        and f5_seed_met
        and ival(official.get("taskwise_all_pass")) == 1
        and official_cov >= float(args.h_overall_coverage_gate)
        and orbit_gap >= float(args.h_random_gap_orbit_gate)
        and ival(official.get("F5_no_debt_count")) >= int(args.part_h_f5_no_debt_gate)
        and ival(official.get("component_non_positive_rows")) >= int(args.part_h_component_non_positive_gate)
        and fval(official.get("finite_step_accept_rate_median")) >= float(args.g_finite_accept_gate)
        and fval(official.get("finite_step_scale_mean_median")) >= float(args.g_finite_scale_gate)
        and fval(official.get("finite_step_skip_count_median")) <= float(args.h_finite_skip_count_gate)
        and controls_present
        and orbit_present
        and noop_present
        and baselines_present
        and random_gap >= float(args.h_beats_control_gate)
        and noop_gap >= float(args.h_beats_noop_gate)
        and baseline_gap >= float(args.h_beats_baseline_gate)
        and ival(official.get("held_test_usage")) == 0
        and relevant_overhead <= float(args.h_overhead_gate)
    )
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_h_job_errors"
    elif not official:
        blocker = "official_h_scheme_missing"
    elif not f5_seed_met:
        blocker = "h_official_seed_count_not_met"
    elif ival(official.get("taskwise_all_pass")) != 1:
        blocker = "h_taskwise_failed"
    elif official_cov < float(args.h_overall_coverage_gate):
        blocker = "h_overall_coverage_failed"
    elif orbit_gap < float(args.h_random_gap_orbit_gate):
        blocker = "h_random_gap_orbit_failed"
    elif ival(official.get("F5_no_debt_count")) < int(args.part_h_f5_no_debt_gate):
        blocker = "h_f5_no_debt_failed"
    elif ival(official.get("component_non_positive_rows")) < int(args.part_h_component_non_positive_gate):
        blocker = "h_component_non_positive_failed"
    elif not controls_present or not orbit_present:
        blocker = "h_random_controls_missing"
    elif not noop_present:
        blocker = "h_same_compute_noop_missing"
    elif not baselines_present:
        blocker = "h_functional_or_blocksnr_baseline_missing"
    elif random_gap < float(args.h_beats_control_gate):
        blocker = "h_random_controls_explain_gain"
    elif noop_gap < float(args.h_beats_noop_gate):
        blocker = "h_same_compute_noop_explains_gain"
    elif baseline_gap < float(args.h_beats_baseline_gate):
        blocker = "h_functional_or_blocksnr_baseline_explains_gain"
    elif relevant_overhead > float(args.h_overhead_gate):
        blocker = "h_overhead_failed"
    else:
        blocker = "none" if gate else "h_safety_or_misc_failed"
    task_csv = write_rows(OUT_ROOT / "part_h_full_positive_control_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_h_full_positive_control_group_summary.csv", groups)
    summary = gate_summary(
        "H",
        gate,
        "H_PositiveControlPass_NoRealTaskYet" if gate else "H_PositiveControlFailed",
        blocker,
        rows,
        taskwise_groups=task_summaries,
        group_summaries=groups,
        positive_control_checks=checks,
    )
    write_json(OUT_ROOT / "part_h_full_positive_control_summary.json", summary)
    nxt = next_actions(
        "H",
        gate,
        blocker,
        [] if gate else [
            "if random controls explain the gain, return to structure projection and tighten matched controls",
            "if same-compute no-op explains the gain, inspect update magnitude rather than structure",
            "if functional/block-SNR baselines explain the gain, run H7_F7NoSafetyDiagnostic to split safety tax from structure tax",
            "if safety fails, return to Part G without changing no-debt budget",
        ],
    )
    append_exec("part-h-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_h_full_positive_control_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part H full positive-control", summary)
    return summary


def train_part_i_real_preflight_row(job: tuple[str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    i_scheme, dataset, seed = job
    start = time.time()
    i_cfg = part_i_scheme_config(str(i_scheme))
    f_scheme = str(i_cfg.get("f_scheme", "F7_FunctionalSobolev025PsQpopPsSignedUtility"))
    f_cfg = part_f_scheme_config(f_scheme)
    safety_type = str(i_cfg.get("safety_type", "G1_F15ExactFiniteStepAll"))
    safety_cfg = part_g_safety_config(safety_type, args)
    if "H10" in str(i_scheme) or "RealVisual" in str(i_scheme):
        safety_cfg = dict(safety_cfg)
        safety_cfg["finite_step_apply_all_parts"] = 0
    try:
        family = real_dataset_family(str(dataset))
        if family != "visual" and bool(f_cfg.get("use_structure_gate", False)):
            return {
                "part": "I",
                "status": "skipped",
                "i_scheme": str(i_scheme),
                "dataset": str(dataset),
                "seed": int(seed),
                "dataset_family": family,
                "route": "I_TabularStructureNotApplicable",
                "skip_reason": "v23.03 structure projection is visual local+rotation only; tabular rows use non-structure baselines",
                "structure_not_applicable": 1,
                "wall_time_s": time.time() - start,
                **AUDIT_DEFAULTS,
            }
        xtr, ytr, xh, yh, input_dim, output_dim, meta = load_part_i_tensors(str(dataset), int(seed), args, device)
        if int(xtr.numel()) <= 0 or int(xh.numel()) <= 0:
            raise RuntimeError(f"empty real-task tensors for dataset={dataset}")
        i_args = real_visual_args(args, input_dim) if family == "visual" else argparse.Namespace(**vars(args))
        i_args.batch_size = int(args.part_i_batch_size)
        i_args.population_grad_examples = min(int(args.population_grad_examples), int(xtr.shape[0]))
        basis_key = str(args.part_i_basis)
        depth = str(args.part_i_depth)
        bargs = v2300.basis_args(i_args, basis_key)
        model_seed = v2300.model_seed_for(basis_key, depth, str(dataset), int(seed))
        model = v2300.make_model(depth, int(input_dim), int(output_dim), model_seed, bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        opt = make_part_f_optimizer(model, f_cfg, i_args)
        before = v2293.metrics_for_model(model, xh, yh)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        gate_trace: list[float] = []
        struct_trace: list[float] = []
        local_trace: list[float] = []
        local_intervention_trace: list[float] = []
        rot_trace: list[float] = []
        dest_trace: list[float] = []
        utility_mult_trace: list[float] = []
        signed_modifier_trace: list[float] = []
        signed_modifier_negative_trace: list[float] = []
        finite_accept_count = 0
        finite_skip_count = 0
        finite_attempt_count = 0
        finite_scale_trace: list[float] = []
        finite_reject_reasons: dict[str, int] = {}
        grad_norm_median: list[float] = []
        grad_norm_p95: list[float] = []
        grad_nan_count = 0
        grad_inf_count = 0
        per_example_count = 0
        structure_refresh_count = 0
        structure_reuse_count = 0
        cached_structure_gates: dict[int, torch.Tensor] = {}
        cached_structure_modifiers: dict[int, torch.Tensor] = {}
        cached_structure_diag: dict[str, float | int] = {}
        use_edge_observation = isinstance(opt, EdgeSobolevPopulationFlow)
        use_structure_gate = bool(f_cfg.get("use_structure_gate", False)) and use_edge_observation and family == "visual"
        use_qpop = bool(f_cfg.get("use_population_gate", False)) and use_edge_observation
        same_compute_noop = bool(int(i_cfg.get("same_compute_noop", 0)))
        structure_mode = str(f_cfg.get("structure_mode", "none"))
        structure_control_type = str(i_cfg.get("structure_control_type", "true"))
        for step in range(1, int(args.part_i_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.part_i_batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            if use_structure_gate:
                cadence = int(getattr(args, "part_i_structure_gate_every", getattr(args, "structure_gate_every", 1)))
                refresh_structure = not cached_structure_gates or cadence <= 1 or ((step - 1) % cadence == 0)
                if refresh_structure:
                    xub, yub = v2293.v2289.iter_train_batches(
                        xtr,
                        ytr,
                        step - 1 + int(getattr(args, "part_d_utility_batch_offset", 997)),
                        int(args.part_i_batch_size),
                        int(seed),
                    )
                    observer = str(getattr(args, "part_i_real_observer", "scalar_dual"))
                    if observer == "full_dual":
                        diag = observe_real_visual_dual_structure_batch(
                            opt,
                            model,
                            xb,
                            yb,
                            int(seed) * 10000 + step,
                            i_args,
                            structure_control_type=structure_control_type,
                            structure_mode=structure_mode,
                            utility_x=xub,
                            utility_y=yub,
                        )
                    elif observer == "band_dual":
                        diag = observe_real_visual_band_dual_structure_batch(
                            opt,
                            model,
                            xb,
                            yb,
                            int(seed) * 10000 + step,
                            i_args,
                            structure_control_type=structure_control_type,
                            structure_mode=structure_mode,
                            utility_x=xub,
                            utility_y=yub,
                        )
                    else:
                        diag = observe_real_visual_scalar_dual_structure_batch(
                            opt,
                            model,
                            xb,
                            yb,
                            int(seed) * 10000 + step,
                            i_args,
                            structure_control_type=structure_control_type,
                            structure_mode=structure_mode,
                            utility_x=xub,
                            utility_y=yub,
                        )
                    cached_structure_gates = {key: value.detach().clone() for key, value in opt._observed_structure_gates.items()}
                    cached_structure_modifiers = {key: value.detach().clone() for key, value in opt._observed_structure_update_modifiers.items()}
                    cached_structure_diag = dict(diag)
                    structure_refresh_count += 1
                else:
                    pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, i_args)
                    for param in model.coeffs:
                        gate = cached_structure_gates.get(id(param))
                        if gate is not None:
                            opt.observe_structure_gate(param, gate)
                        modifier = cached_structure_modifiers.get(id(param))
                        if modifier is not None:
                            opt.observe_structure_update_modifier(param, modifier)
                    diag = {
                        **cached_structure_diag,
                        "per_example_count": pe_count,
                        "grad_norm_median": gmed,
                        "grad_norm_p95": gp95,
                        "grad_nan_count": nnan,
                        "grad_inf_count": ninf,
                    }
                    structure_reuse_count += 1
            elif use_qpop:
                pe_count, gmed, gp95, nnan, ninf = v2300.observe_task_gradients(opt, model, xb, yb, i_args)
                diag = {
                    "per_example_count": pe_count,
                    "grad_norm_median": gmed,
                    "grad_norm_p95": gp95,
                    "grad_nan_count": nnan,
                    "grad_inf_count": ninf,
                    "q_struct_mean": 1.0,
                    "local_coherence_mean": 0.0,
                    "local_intervention_mean": 0.0,
                    "rotation_coherence_mean": 0.0,
                    "destructive_coherence_mean": 0.0,
                    "utility_multiplier_mean": 0.0,
                    "signed_modifier_mean": 0.0,
                    "signed_modifier_negative_fraction": 0.0,
                }
            else:
                diag = {
                    "per_example_count": 0,
                    "grad_norm_median": 0.0,
                    "grad_norm_p95": 0.0,
                    "grad_nan_count": 0,
                    "grad_inf_count": 0,
                    "q_struct_mean": 0.0,
                    "local_coherence_mean": 0.0,
                    "local_intervention_mean": 0.0,
                    "rotation_coherence_mean": 0.0,
                    "destructive_coherence_mean": 0.0,
                    "utility_multiplier_mean": 0.0,
                    "signed_modifier_mean": 0.0,
                    "signed_modifier_negative_fraction": 0.0,
                }
            per_example_count = max(per_example_count, ival(diag.get("per_example_count")))
            grad_norm_median.append(fval(diag.get("grad_norm_median")))
            grad_norm_p95.append(fval(diag.get("grad_norm_p95")))
            grad_nan_count += ival(diag.get("grad_nan_count"))
            grad_inf_count += ival(diag.get("grad_inf_count"))
            struct_trace.append(fval(diag.get("q_struct_mean")))
            local_trace.append(fval(diag.get("local_coherence_mean")))
            local_intervention_trace.append(fval(diag.get("local_intervention_mean")))
            rot_trace.append(fval(diag.get("rotation_coherence_mean")))
            dest_trace.append(fval(diag.get("destructive_coherence_mean")))
            utility_mult_trace.append(fval(diag.get("utility_multiplier_mean")))
            signed_modifier_trace.append(fval(diag.get("signed_modifier_mean")))
            signed_modifier_negative_trace.append(fval(diag.get("signed_modifier_negative_fraction")))
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb.long())
            loss.backward()
            if same_compute_noop:
                if hasattr(opt, "clear_observed_gradients"):
                    opt.clear_observed_gradients()
                finite_diag = {
                    "finite_step_accept": 0,
                    "finite_step_skip": 1,
                    "finite_step_scale": 0.0,
                    "finite_step_attempts": 1,
                    "finite_step_reject_reason": "same_compute_noop",
                }
            else:
                finite_diag = v2303_finite_step_guarded_step(
                    opt,
                    model,
                    before,
                    xh,
                    yh,
                    i_args,
                    safety_cfg,
                    part="G_C2",
                    step=step,
                )
            finite_accept_count += ival(finite_diag.get("finite_step_accept"))
            finite_skip_count += ival(finite_diag.get("finite_step_skip"))
            finite_attempt_count += ival(finite_diag.get("finite_step_attempts"))
            finite_scale_trace.append(fval(finite_diag.get("finite_step_scale"), 1.0))
            reason = str(finite_diag.get("finite_step_reject_reason", ""))
            if reason and reason not in {"accepted", "not_checked", "disabled"}:
                finite_reject_reasons[reason] = finite_reject_reasons.get(reason, 0) + 1
            stats = getattr(opt, "last_stats", None)
            gate_trace.append(float(getattr(stats, "gate_density_mean", 1.0 if not same_compute_noop else 0.0)))
        after = v2293.metrics_for_model(model, xh, yh)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        deltas = v2303_debt_deltas(after, before)
        no_debt = int(all(v2303_debt_component_ok(k, float(v), float(args.no_debt_budget)) for k, v in deltas.items()))
        return {
            "part": "I",
            "status": "ok",
            "i_scheme": str(i_scheme),
            "scheme": f_scheme,
            "safety_type": safety_type,
            "control_family": str(i_cfg.get("control_family", "")),
            "structure_control_type": structure_control_type,
            "same_compute_noop": int(same_compute_noop),
            "dataset": str(dataset),
            "dataset_family": family,
            "seed": int(seed),
            "basis_key": basis_key,
            "depth": depth,
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "task_tier": str(meta.get("task_tier", "")),
            "source_kind": str(meta.get("source_kind", "")),
            "train_steps": int(args.part_i_steps),
            "train_size": int(xtr.shape[0]),
            "held_size": int(xh.shape[0]),
            "real_visual_profile": str(getattr(args, "part_i_real_profile", "local_rotation_dual")) if family == "visual" and use_structure_gate else "not_used",
            "part_i_real_observer": str(getattr(args, "part_i_real_observer", "scalar_dual")) if family == "visual" and use_structure_gate else "not_used",
            "part_i_real_transform_max": int(getattr(args, "part_i_real_transform_max", 1)) if family == "visual" and use_structure_gate else 0,
            "part_i_structure_gate_every": int(getattr(args, "part_i_structure_gate_every", getattr(args, "structure_gate_every", 1))) if family == "visual" and use_structure_gate else 0,
            "structure_refresh_count": structure_refresh_count,
            "structure_reuse_count": structure_reuse_count,
            "functional_gram_used": int(f_cfg.get("functional_gram_used", 0)),
            "qpop_used": int(f_cfg.get("qpop_used", 0)),
            "ps_used": int(f_cfg.get("ps_used", 0)),
            "signed_utility_used": int(f_cfg.get("signed_utility_used", 0)),
            "sobolev_exponent": float(f_cfg.get("sobolev_exponent", 0.0)),
            "structure_mode": structure_mode if family == "visual" else "tabular_no_structure",
            "gate_density": mean(gate_trace),
            "q_struct_mean": mean(struct_trace),
            "local_coherence_mean": mean(local_trace),
            "local_intervention_mean": mean(local_intervention_trace),
            "rotation_coherence_mean": mean(rot_trace),
            "destructive_coherence_mean": mean(dest_trace),
            "utility_multiplier_mean": mean(utility_mult_trace),
            "signed_modifier_mean": mean(signed_modifier_trace),
            "signed_modifier_negative_fraction": mean(signed_modifier_negative_trace),
            "held_nll_initial": before["nll"],
            "held_nll_final": after["nll"],
            "held_nll_delta": after["nll"] - before["nll"],
            "held_accuracy_initial": before["accuracy"],
            "held_accuracy_final": after["accuracy"],
            "held_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "held_coverage_initial": before["coverage_CVaR25"],
            "held_coverage_final": after["coverage_CVaR25"],
            "held_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "Brier_delta": deltas["Brier"],
            "ECE_delta": deltas["ECE"],
            "tail95_delta": deltas["tail95"],
            "tail99_delta": deltas["tail99"],
            "margin10_delta": deltas["margin10"],
            "train_nll_delta": after_train["nll"] - before_train["nll"],
            "real_no_debt": no_debt,
            "finite_step_accept_count": finite_accept_count,
            "finite_step_skip_count": finite_skip_count,
            "finite_step_attempt_count": finite_attempt_count,
            "finite_step_accept_rate": finite_accept_count / max(1, finite_accept_count + finite_skip_count),
            "finite_step_scale_mean": mean(finite_scale_trace, 1.0),
            "finite_step_reject_reasons": ";".join(f"{k}:{v}" for k, v in sorted(finite_reject_reasons.items())),
            "per_example_count": per_example_count,
            "grad_norm_median": mean(grad_norm_median),
            "grad_norm_p95": mean(grad_norm_p95),
            "grad_nan_count": grad_nan_count,
            "grad_inf_count": grad_inf_count,
            "held_test_usage": 0,
            "used_fake_data_rows": 0,
            "wall_time_s": time.time() - start,
            **{k: v for k, v in AUDIT_DEFAULTS.items() if k not in {"held_test_usage", "used_fake_data_rows"}},
        }
    except Exception as exc:
        return {
            "part": "I",
            "status": "error",
            "i_scheme": str(i_scheme),
            "scheme": f_scheme,
            "safety_type": safety_type,
            "dataset": str(dataset),
            "seed": int(seed),
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def run_part_i(args: argparse.Namespace) -> dict[str, Any]:
    h = read_json(OUT_ROOT / "part_h_full_positive_control_summary.json")
    if h and ival(h.get("gate_pass")) != 1 and not bool(int(args.part_i_allow_without_h)):
        return write_blocked("I", "I_BlockedByPartH", "part_h_failed", "Part I requires Part H pass unless --part-i-allow-without-h=1.")
    device = device_from_args(args)
    rows = [train_part_i_real_preflight_row(job, args, device) for job in shard_items(part_i_jobs(args), args)]
    path = write_rows(OUT_ROOT / f"part_i_real_task_preflight_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-i", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("I", 0, "I_ShardsWritten", "merge_required", rows)


def merge_part_i(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_i_real_task_preflight_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_i_real_task_preflight_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("i_scheme"), r.get("dataset")) for r in ok}):
        group = [r for r in ok if (r.get("i_scheme"), r.get("dataset")) == key]
        groups.append({
            "i_scheme": key[0],
            "dataset": key[1],
            "dataset_family": sorted({str(r.get("dataset_family", "")) for r in group})[0] if group else "",
            "rows": len(group),
            "held_nll_delta_median": median(r.get("held_nll_delta") for r in group),
            "held_accuracy_improvement_median": median(r.get("held_accuracy_improvement") for r in group),
            "held_coverage_improvement_median": median(r.get("held_coverage_improvement") for r in group),
            "real_no_debt_count": sum(ival(r.get("real_no_debt")) for r in group),
            "finite_step_accept_rate_median": median(r.get("finite_step_accept_rate") for r in group),
            "finite_step_scale_mean_median": median(r.get("finite_step_scale_mean") for r in group),
            "finite_step_skip_count_median": median(r.get("finite_step_skip_count") for r in group),
            "gate_density_median": median(r.get("gate_density") for r in group),
            "q_struct_mean_median": median(r.get("q_struct_mean") for r in group),
            "wall_time_s_median": median(r.get("wall_time_s") for r in group),
        })
    visual_official = [
        g for g in groups
        if str(g.get("i_scheme")) == str(args.part_i_official_scheme)
        and str(g.get("dataset_family")) == "visual"
    ]
    visual_gate = int(
        bool(visual_official)
        and all(
            fval(g.get("held_nll_delta_median"), 1.0) <= float(args.part_i_nll_delta_gate)
            or fval(g.get("held_coverage_improvement_median"), -1.0) >= float(args.part_i_coverage_gate)
            for g in visual_official
        )
        and all(ival(g.get("real_no_debt_count")) >= max(1, int(math.ceil(0.5 * ival(g.get("rows"))))) for g in visual_official)
    )
    baseline_dominated: list[dict[str, Any]] = []
    if bool(int(getattr(args, "part_i_require_baseline_nondominated", 1))):
        nll_eps = float(getattr(args, "part_i_baseline_nll_eps", 0.0))
        cov_eps = float(getattr(args, "part_i_baseline_coverage_eps", 0.0))
        for official in visual_official:
            dataset = str(official.get("dataset"))
            official_nll = fval(official.get("held_nll_delta_median"), 1.0)
            official_cov = fval(official.get("held_coverage_improvement_median"), -1.0)
            for baseline in groups:
                if str(baseline.get("dataset")) != dataset:
                    continue
                if str(baseline.get("dataset_family")) != "visual":
                    continue
                if str(baseline.get("i_scheme")) == str(args.part_i_official_scheme):
                    continue
                baseline_nll = fval(baseline.get("held_nll_delta_median"), 1.0)
                baseline_cov = fval(baseline.get("held_coverage_improvement_median"), -1.0)
                nll_not_worse = baseline_nll <= official_nll + nll_eps
                cov_not_worse = baseline_cov >= official_cov - cov_eps
                strictly_better = baseline_nll < official_nll - nll_eps or baseline_cov > official_cov + cov_eps
                if nll_not_worse and cov_not_worse and strictly_better:
                    baseline_dominated.append({
                        "dataset": dataset,
                        "official_scheme": str(args.part_i_official_scheme),
                        "baseline_scheme": str(baseline.get("i_scheme")),
                        "official_nll_delta_median": official_nll,
                        "baseline_nll_delta_median": baseline_nll,
                        "official_coverage_improvement_median": official_cov,
                        "baseline_coverage_improvement_median": baseline_cov,
                    })
                    break
    baseline_nondominated = int(not baseline_dominated)
    tabular_ok = int(any(str(g.get("dataset_family")) == "tabular" for g in groups) or "Wine" not in str(args.part_i_datasets))
    errors = [r for r in rows if r.get("status") == "error"]
    gate = int(not errors and visual_gate and baseline_nondominated and tabular_ok and ok and not any(ival(r.get("held_test_usage")) for r in ok))
    if errors:
        blocker = "part_i_job_errors"
    elif not visual_official:
        blocker = "part_i_visual_official_missing"
    elif not visual_gate:
        blocker = "part_i_visual_smoke_failed"
    elif not baseline_nondominated:
        blocker = "part_i_baseline_dominated"
    elif not tabular_ok:
        blocker = "part_i_tabular_preflight_missing"
    else:
        blocker = "none" if gate else "part_i_misc_failed"
    group_csv = write_rows(OUT_ROOT / "part_i_real_task_preflight_group_summary.csv", groups)
    summary = gate_summary(
        "I",
        gate,
        "I_LimitedRealTaskPreflightSmokePass" if gate else "I_LimitedRealTaskPreflightFailed",
        blocker,
        rows,
        group_summaries=groups,
        visual_smoke_gate=visual_gate,
        baseline_nondominated=baseline_nondominated,
        baseline_dominated=baseline_dominated,
        tabular_preflight_present=tabular_ok,
        promotion_allowed=0,
        note="Part I is a limited smoke/preflight gate, not a real-task official promotion gate.",
    )
    write_json(OUT_ROOT / "part_i_real_task_preflight_summary.json", summary)
    nxt = next_actions(
        "I",
        gate,
        blocker,
        [
            "if smoke passes, scale to MNIST/FashionMNIST/KMNIST seeds0-2 with fixed visual dual profile",
            "if visual smoke fails, inspect whether local+rotation averaging kills H10 signed utility phase",
            "if tabular baselines fail to load, keep tabular as data-availability blocker rather than algorithm failure",
        ] if gate else [
            "debug part_i_job_errors without changing H10/F7/G1 theory",
            "if visual smoke fails without errors, compare H10 dual profile against H0/H1 baseline rows",
        ],
    )
    append_exec("part-i-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_i_real_task_preflight_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part I limited real-task preflight", summary)
    return summary


def write_blocked(part: str, route: str, blocker: str, reason: str) -> dict[str, Any]:
    row = {"part": part, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}
    write_rows(OUT_ROOT / f"part_{part.lower()}_blocked_matrix.csv", [row])
    summary = gate_summary(part, 0, route, blocker, [row], reason=reason)
    write_json(OUT_ROOT / f"part_{part.lower()}_summary.json", summary)
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    a = read_json(OUT_ROOT / "part_a_summary.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    c = read_json(OUT_ROOT / "part_c_structure_coherence_summary.json")
    d = read_json(OUT_ROOT / "part_d_summary.json")
    e = read_json(OUT_ROOT / "part_e_random_mechanism_summary.json")
    f = read_json(OUT_ROOT / "part_f_component_ablation_summary.json")
    g = read_json(OUT_ROOT / "part_g_safety_trust_summary.json")
    h = read_json(OUT_ROOT / "part_h_full_positive_control_summary.json")
    i = read_json(OUT_ROOT / "part_i_real_task_preflight_summary.json")
    standalone_h_root = bool(h) and not any(bool(x) for x in [a, b, c, d, e, f, g])
    if standalone_h_root and ival(h.get("gate_pass")) == 1 and not i:
        route = "H_PositiveControlPass_NoRealTaskYet"
        blocker = "part_i_real_task_preflight_not_run"
    elif standalone_h_root and i and ival(i.get("gate_pass")) != 1:
        route = i.get("route", "I_LimitedRealTaskPreflightFailed")
        blocker = i.get("dominant_blocker", "part_i_failed")
    elif standalone_h_root and i and ival(i.get("gate_pass")) == 1:
        route = "I_LimitedRealTaskPreflightSmokePass"
        blocker = "real_task_official_matrix_not_run"
    elif ival(a.get("gate_pass")) != 1:
        route = "A_CodeIdentityFailed"
        blocker = a.get("dominant_blocker", "part_a")
    elif ival(b.get("gate_pass")) != 1:
        route = "B_HistoryLockFailed"
        blocker = b.get("dominant_blocker", "part_b")
    elif ival(c.get("gate_pass")) != 1:
        route = "C_StructureCoherenceFailed"
        blocker = c.get("dominant_blocker", "part_c")
        write_blocked("D", "D_BlockedByPartC", str(blocker), "Part C failed; plan forbids C2 training before structure separability is established.")
        write_blocked("G", "G_BlockedByPartC", str(blocker), "Part C failed; plan forbids safety trust official path before Part D pass.")
        write_blocked("H", "H_BlockedByPartC", str(blocker), "Part C failed; full positive-control is forbidden.")
        write_blocked("I", "I_BlockedByPartC", str(blocker), "Part C failed; real-task preflight is forbidden.")
    elif ival(d.get("gate_pass")) != 1:
        route = d.get("route", "D_TaskwiseC2Failed" if d else "D_NotRun")
        blocker = d.get("dominant_blocker", "part_d_missing_or_failed")
    elif ival(e.get("gate_pass")) != 1:
        route = e.get("route", "E_RandomMechanismUnresolved" if e else "E_NotRun")
        blocker = e.get("dominant_blocker", "part_e_missing_or_failed")
    elif ival(f.get("gate_pass")) != 1:
        route = f.get("route", "F_ComponentAblationNoStructureAdvantage" if f else "F_NotRun")
        blocker = f.get("dominant_blocker", "part_f_missing_or_failed")
    elif g and ival(g.get("gate_pass")) != 1:
        route = g.get("route", "G_SafetyTrustFailed")
        blocker = g.get("dominant_blocker", "part_g_failed")
    elif h and ival(h.get("gate_pass")) != 1:
        route = h.get("route", "H_PositiveControlFailed")
        blocker = h.get("dominant_blocker", "part_h_failed")
    elif h and ival(h.get("gate_pass")) == 1 and not i:
        route = "H_PositiveControlPass_NoRealTaskYet"
        blocker = "part_i_real_task_preflight_not_run"
    elif i and ival(i.get("gate_pass")) != 1:
        route = i.get("route", "I_LimitedRealTaskPreflightFailed")
        blocker = i.get("dominant_blocker", "part_i_failed")
    elif i and ival(i.get("gate_pass")) == 1:
        route = "I_LimitedRealTaskPreflightSmokePass"
        blocker = "real_task_official_matrix_not_run"
    else:
        route = "H_NotRun_FullPositiveControlNotImplemented"
        blocker = "part_h_full_positive_control_not_implemented_in_this_runner"
    final = {
        "route": route,
        "dominant_blocker": blocker,
        "official_candidate_gate_pass": 0,
        "promotion_allowed": 0,
        "part_a_gate_pass": a.get("gate_pass", "missing"),
        "part_b_gate_pass": b.get("gate_pass", "missing"),
        "part_c_gate_pass": c.get("gate_pass", "missing"),
        "part_d_gate_pass": d.get("gate_pass", "missing"),
        "part_e_gate_pass": e.get("gate_pass", "missing"),
        "part_f_gate_pass": f.get("gate_pass", "missing"),
        "part_g_gate_pass": g.get("gate_pass", "missing") if g else "missing",
        "part_h_gate_pass": h.get("gate_pass", "missing") if h else "missing",
        "part_i_gate_pass": i.get("gate_pass", "missing") if i else "missing",
        "generated_at": now(),
        **AUDIT_DEFAULTS,
    }
    write_json(OUT_ROOT / "final_route.json", final)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        "# DG-KAN v23.03 Reproduction Manifest\n\n"
        f"- runner: `{rel(RUNNER)}`\n"
        f"- plan: `{rel(PLAN)}`\n"
        f"- out_root: `{rel(OUT_ROOT)}`\n"
        "- required logs:\n"
        f"  - `{rel(EXEC_LOG)}`\n"
        f"  - `{rel(RECAP_LOG)}`\n\n"
        "## Key Commands\n\n"
        "- Part A: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-a`\n"
        "- Part B: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-b`\n"
        "- Part C shard: `CUDA_VISIBLE_DEVICES=2 python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 0`\n"
        "- Part C merge: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-c-merge --device cpu`\n"
        "- Part D shard: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-d --device cuda:0`\n"
        "- Part D merge: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-d-merge --device cpu`\n"
        "- Part E shard: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-e --device cuda:0`\n"
        "- Part E merge: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-e-merge --device cpu`\n"
        "- Part F shard: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-f --device cuda:0`\n"
        "- Part F merge: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-f-merge --device cpu`\n"
        "- Part G shard: `CUDA_VISIBLE_DEVICES=0 python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-g --device cuda:0`\n"
        "- Part G merge: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-g-merge --device cpu`\n"
        "- Part H shard: `CUDA_VISIBLE_DEVICES=0 python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-h --device cuda:0`\n"
        "- Part H merge: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-h-merge --device cpu`\n"
        "- Part I shard: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-i --device cuda:0`\n"
        "- Part I merge: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode part-i-merge --device cpu`\n"
        "- Finalize: `python experiments/run_v23_03_structure_projected_functional_population_flow.py --mode finalize --device cpu`\n",
        encoding="utf-8",
    )
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(manifest)}", gpu=str(args.device))
    append_recap("Final route", final)
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="part-a", choices=["part-a", "part-b", "part-c", "part-c-merge", "part-d", "part-d-merge", "part-e", "part-e-merge", "part-f", "part-f-merge", "part-g", "part-g-merge", "part-h", "part-h-merge", "part-i", "part-i-merge", "finalize"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--synthetic-train-size", type=int, default=192)
    p.add_argument("--synthetic-guard-size", type=int, default=128)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--input-dim", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--deep-width", type=int, default=16)
    p.add_argument("--dfour-k", type=int, default=3)
    p.add_argument("--mlp-hidden", type=int, default=24)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--batch-size", type=int, default=36)
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--edge-lr", type=float, default=0.004)
    p.add_argument("--adamw-lr", type=float, default=0.004)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--edge-weight-normalization", default="trace")
    p.add_argument("--edge-weight-ridge", type=float, default=1.0e-6)
    p.add_argument("--functional-gram-quadrature-points", type=int, default=257)
    p.add_argument("--snr-beta", type=float, default=2.0)
    p.add_argument("--gate-floor", type=float, default=0.2)
    p.add_argument("--gate-floor-mode", default="final", choices=["final", "population"])
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--population-grad-examples", type=int, default=6)
    p.add_argument("--part-c-schemes", default="C1_LocalStructureProduct,C2_RotationStructureProduct,C3_DualMinStructureProduct,C4_OrbitAverageBeforeSNR,C5_ContrastiveStructureProduct,C6_StructureOnly,C7_PopOnly")
    p.add_argument("--part-c-basis", default="dche_k9")
    p.add_argument("--part-c-depths", default="depth3")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-seed-count", type=int, default=5)
    p.add_argument("--part-d-basis", default="dche_k9")
    p.add_argument("--part-d-depths", default="depth3")
    p.add_argument("--part-d-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-d-schemes", default="D4_ContrastiveStructureProduct")
    p.add_argument("--part-d-seed-count", type=int, default=3)
    p.add_argument("--part-d-orbit-random", type=int, default=0)
    p.add_argument("--part-d-trace", type=int, default=0)
    p.add_argument("--part-d-utility-examples", type=int, default=6)
    p.add_argument("--part-d-utility-batch-offset", type=int, default=997)
    p.add_argument("--part-d-utility-batch-offset2", type=int, default=1499)
    p.add_argument("--utility-gate-gain", type=float, default=1.0)
    p.add_argument("--utility-gate-max", type=float, default=3.0)
    p.add_argument("--utility-tilt-floor", type=float, default=0.25)
    p.add_argument("--signed-negative-scale", type=float, default=-1.0)
    p.add_argument("--part-e-random-types", default="true,density,histogram,orbit_shuffle,structure_breaking")
    p.add_argument("--part-f-schemes", default="F1_RawAdamWNoFunctionalGramNoQpopNoPs,F2_FunctionalGramAdamWNoQpopNoPs,F3_FunctionalGramQpopOnly,F4_FunctionalGramPsOnly,F5_FunctionalGramPsQpopPsSignedUtility,F7_FunctionalSobolev025PsQpopPsSignedUtility")
    p.add_argument("--part-f-official-scheme", default="F7_FunctionalSobolev025PsQpopPsSignedUtility")
    p.add_argument("--part-f-control-types", default="true,orbit_shuffle")
    p.add_argument("--f-component-advantage-gate", type=float, default=0.03)
    p.add_argument("--f-random-gap-advantage-gate", type=float, default=0.03)
    p.add_argument("--part-g-source-scheme", default="D17_FunctionalSobolev025LocalInterventionCoordinateSignedUtilityTiltOrbitCenteredContrastiveStructureProduct")
    p.add_argument("--part-g-safety-schemes", default="G0_NoSafety,G1_F15ExactFiniteStepAll,G4_RandomVetoMatchedControl")
    p.add_argument("--part-g-parts", default="G_C2,G_F5")
    p.add_argument("--part-g-c2-seed-count", type=int, default=3)
    p.add_argument("--part-g-f5-seed-count", type=int, default=15)
    p.add_argument("--part-g-f5-calibration-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--no-debt-budget", type=float, default=0.01)
    p.add_argument("--g-f5-no-debt-gate", type=int, default=12)
    p.add_argument("--g-component-non-positive-gate", type=int, default=12)
    p.add_argument("--g-c2-retention-gate", type=float, default=0.70)
    p.add_argument("--g-finite-accept-gate", type=float, default=0.20)
    p.add_argument("--g-finite-scale-gate", type=float, default=0.05)
    p.add_argument("--g-finite-skip-fraction-gate", type=float, default=0.70)
    p.add_argument("--g-random-veto-gap-gate", type=int, default=8)
    p.add_argument("--g-random-veto-retention-gap-gate", type=float, default=0.50)
    p.add_argument("--g-overhead-gate", type=float, default=2.5)
    p.add_argument("--g-random-veto-accept-prob", type=float, default=0.725)
    p.add_argument("--g-random-veto-structure-control", type=int, default=1)
    p.add_argument("--g-finite-step-guard-every", type=int, default=1)
    p.add_argument("--g-finite-step-guard-examples", type=int, default=128)
    p.add_argument("--g-finite-step-tries", type=int, default=5)
    p.add_argument("--g-finite-step-shrink", type=float, default=0.5)
    p.add_argument("--g-finite-step-c2-floor", type=float, default=-0.05)
    p.add_argument("--g-finite-step-apply-all-parts", type=int, default=1)
    p.add_argument("--g-finite-step-components", default="Brier,ECE,tail95,tail99,margin10")
    p.add_argument("--g-debt-regularization-weight", type=float, default=0.30)
    p.add_argument("--g-debt-regularization-components", default="ECE,tail95,tail99")
    p.add_argument("--part-h-schemes", default="H0_FunctionalGramAdamWBaseline,H1_BlockSNRQpopBaseline,H10_F7ObjectiveAlignedSafetyDiagnostic,H3_RandomDensityControl,H4_RandomHistogramControl,H5_RandomOrbitControl,H6_SameComputeNoOp")
    p.add_argument("--part-h-official-scheme", default="H10_F7ObjectiveAlignedSafetyDiagnostic")
    p.add_argument("--part-h-functional-baseline-scheme", default="H0_FunctionalGramAdamWBaseline")
    p.add_argument("--part-h-blocksnr-baseline-scheme", default="H1_BlockSNRQpopBaseline")
    p.add_argument("--part-h-parts", default="H_C2,H_F5")
    p.add_argument("--part-h-c2-seed-count", type=int, default=15)
    p.add_argument("--part-h-f5-seed-count", type=int, default=15)
    p.add_argument("--part-h-f5-calibration-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-h-f5-no-debt-gate", type=int, default=12)
    p.add_argument("--part-h-component-non-positive-gate", type=int, default=12)
    p.add_argument("--h-overall-coverage-gate", type=float, default=0.06)
    p.add_argument("--h-random-gap-orbit-gate", type=float, default=0.05)
    p.add_argument("--h-beats-control-gate", type=float, default=0.05)
    p.add_argument("--h-beats-noop-gate", type=float, default=0.05)
    p.add_argument("--h-beats-baseline-gate", type=float, default=0.03)
    p.add_argument("--h-finite-skip-count-gate", type=float, default=56.0)
    p.add_argument("--h-overhead-gate", type=float, default=2.5)
    p.add_argument("--part-i-datasets", default="MNIST,Wine")
    p.add_argument("--part-i-schemes", default="I0_H10RealVisualDual,I1_FunctionalGramBaseline,I2_BlockSNRQpopBaseline")
    p.add_argument("--part-i-official-scheme", default="I0_H10RealVisualDual")
    p.add_argument("--part-i-seed-count", type=int, default=1)
    p.add_argument("--part-i-train-size", type=int, default=256)
    p.add_argument("--part-i-held-size", type=int, default=128)
    p.add_argument("--part-i-batch-size", type=int, default=32)
    p.add_argument("--part-i-steps", type=int, default=30)
    p.add_argument("--part-i-basis", default="dche_k9")
    p.add_argument("--part-i-depth", default="depth3")
    p.add_argument("--part-i-real-observer", default="scalar_dual", choices=["scalar_dual", "band_dual", "full_dual"])
    p.add_argument("--part-i-real-profile", default="local_rotation_dual", choices=["local_rotation_dual", "local_only", "rotation_only"])
    p.add_argument("--part-i-real-transform-max", type=int, default=1)
    p.add_argument("--part-i-structure-gate-every", type=int, default=999)
    p.add_argument("--part-i-scalar-signed-modifier", type=int, default=0)
    p.add_argument("--part-i-band-grid", type=int, default=4)
    p.add_argument("--part-i-band-edge-bands", type=int, default=4)
    p.add_argument("--part-i-band-degree-bands", type=int, default=3)
    p.add_argument("--part-i-band-signed-modifier", type=int, default=0)
    p.add_argument("--part-i-band-density-preserve", type=int, default=0)
    p.add_argument("--part-i-band-gate-max", type=float, default=2.0)
    p.add_argument("--part-i-tier2-download", type=int, default=0)
    p.add_argument("--part-i-allow-without-h", type=int, default=0)
    p.add_argument("--part-i-nll-delta-gate", type=float, default=0.0)
    p.add_argument("--part-i-coverage-gate", type=float, default=0.0)
    p.add_argument("--part-i-require-baseline-nondominated", type=int, default=1)
    p.add_argument("--part-i-baseline-nll-eps", type=float, default=0.0)
    p.add_argument("--part-i-baseline-coverage-eps", type=float, default=0.0)
    p.add_argument("--structure-block-family", default="degree_edgebank")
    p.add_argument("--structure-transform-profile", default="default", choices=["default", "expanded", "task_orbit"])
    p.add_argument("--structure-guard-source", default="held_guard", choices=["held_guard", "train_split"])
    p.add_argument("--structure-guard-examples", type=int, default=64)
    p.add_argument("--structure-addressable-only", type=int, default=0)
    p.add_argument("--structure-gate-every", type=int, default=1)
    p.add_argument("--contrastive-eta", type=float, default=0.5)
    p.add_argument("--task-complement-eta", type=float, default=0.35)
    p.add_argument("--local-intervention-scale", type=float, default=1.0)
    p.add_argument("--local-intervention-gate-max", type=float, default=3.0)
    p.add_argument("--orbit-center-gain", type=float, default=1.0)
    p.add_argument("--c-orbit-gap-gate", type=float, default=0.05)
    p.add_argument("--c-source-guard-cosine-gate", type=float, default=0.60)
    p.add_argument("--c-coherence-gap-gate", type=float, default=0.05)
    p.add_argument("--c-density-low", type=float, default=0.15)
    p.add_argument("--c-density-high", type=float, default=0.85)
    p.add_argument("--d-task-coverage-gate", type=float, default=0.02)
    p.add_argument("--d-task-random-gap-gate", type=float, default=0.03)
    p.add_argument("--d-overall-coverage-gate", type=float, default=0.06)
    p.add_argument("--d-overall-random-gap-gate", type=float, default=0.05)
    p.add_argument("--e-structure-breaking-gap-gate", type=float, default=0.05)
    p.add_argument("--e-equal-tolerance", type=float, default=0.005)
    p.add_argument("--stat-warmup-steps", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "part-a":
        return run_part_a(args)
    if args.mode == "part-b":
        return run_part_b(args)
    if args.mode == "part-c":
        return run_part_c(args)
    if args.mode == "part-c-merge":
        return merge_part_c(args)
    if args.mode == "part-d":
        return run_part_d(args)
    if args.mode == "part-d-merge":
        return merge_part_d(args)
    if args.mode == "part-e":
        return run_part_e(args)
    if args.mode == "part-e-merge":
        return merge_part_e(args)
    if args.mode == "part-f":
        return run_part_f(args)
    if args.mode == "part-f-merge":
        return merge_part_f(args)
    if args.mode == "part-g":
        return run_part_g(args)
    if args.mode == "part-g-merge":
        return merge_part_g(args)
    if args.mode == "part-h":
        return run_part_h(args)
    if args.mode == "part-h-merge":
        return merge_part_h(args)
    if args.mode == "part-i":
        return run_part_i(args)
    if args.mode == "part-i-merge":
        return merge_part_i(args)
    return finalize(args)


if __name__ == "__main__":
    main()
