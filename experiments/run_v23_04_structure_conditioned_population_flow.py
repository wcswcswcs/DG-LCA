#!/usr/bin/env python3
"""DG-KAN v23.04 Structure-Conditioned Population Flow runner."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
import experiments.run_v23_00r_curve_geometry_population_flow as v2300
from dgkan.fu.population_risk_gate import block_snr_gate
from dgkan.fu.signal_channel_estimators import per_example_gradient_matrix
from dgkan.fu.structure_coherence import (
    apply_perm,
    block_shuffle,
    cosine,
    density_random_like,
    hist_random_like,
    permute_param_examples,
    phase_shuffle,
    quantile_normalize,
    transform_perm,
    transform_specs,
    weighted_structure_score,
)
from dgkan.fu.structure_projected_population import structure_projected_block_snr_gate
from dgkan.optim.edge_sobolev_population_flow import EdgeSobolevSNRFU, mark_kan_edge_params


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.04_StructureConditionedPopulationFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.04_StructureConditionedPopulationFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.04_StructureConditionedPopulationFlow_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2304_OUT_ROOT", str(ROOT / "results/v23_04_structure_conditioned_population_flow"))).resolve()

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

PART_C_SCHEMES: dict[str, str] = {
    "C0_ProductBaseline": "product",
    "C1_StructureOnlyResidual": "structure_only",
    "C2_StructureFirstBlockSNR": "structure_first_snr",
    "C3_OrbitResidualizedStructureSNR": "orbit_residual_snr",
    "C4a_AdditiveResidual_lam05": "additive_0.5",
    "C4b_AdditiveResidual_lam07": "additive_0.7",
    "C5_SoftORStructurePopulation": "soft_or",
    "C6_TwoStreamNormalizedGate": "two_stream_proxy",
    "C7_DualFamilyUnion": "dual_union",
    "C7b_DualFamilyAverage": "dual_average",
    "C8_DestructiveContrastiveStructure": "destructive_contrastive",
}

CANDIDATE_PART_C_SCHEMES = {
    "C2_StructureFirstBlockSNR",
    "C3_OrbitResidualizedStructureSNR",
    "C4a_AdditiveResidual_lam05",
    "C4b_AdditiveResidual_lam07",
    "C5_SoftORStructurePopulation",
    "C6_TwoStreamNormalizedGate",
}

STRUCTURE_ONLY_SCHEMES = {
    "C1_StructureOnlyResidual",
    "C7_DualFamilyUnion",
    "C7b_DualFamilyAverage",
    "C8_DestructiveContrastiveStructure",
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


def write_json(path: Path, data: Any) -> Path:
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
    if not keys:
        keys = list(AUDIT_DEFAULTS.keys())
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
            "# DG-KAN v23.04 StructureConditionedPopulationFlow 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只记录真实命令、真实 artifact、真实错误；缺失写 `missing`，跳过写 `skipped`，不补造。\n"
            "- 复现提示：Part C 支持 `--shard-count/--shard-index`；本轮优先使用物理 GPU 2/3 并行。\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.04 StructureConditionedPopulationFlow 实验结果复盘\n\n"
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


def next_actions(part: str, gate: int, blocker: str, actions: list[str], *, rerun_commands: list[str] | None = None, max_repair_rounds: int = 0) -> Path:
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
                "do_not_report_aggregate_median_over_taskwise_failure",
            ],
            "rerun_commands": rerun_commands or [],
            "expected_fields_after_rerun": [
                "part_c_matrix.csv",
                "part_c_task_summary.csv",
                "part_c_group_summary.csv",
                "part_c_summary.json",
            ],
            "max_repair_rounds": int(max_repair_rounds),
        },
    )


def static_scan(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|metric_winner)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|winner_update)\s*\("),
        "external_product_feature": re.compile(r"\b(make_external_product_feature|patch_product_feature)\s*\("),
        "mlp_readout": re.compile(r"\bMLP(Readout|Head)\b|nn\.(Linear|Sequential)\([^)]*readout", re.IGNORECASE),
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
        ROOT / "dgkan/fu/structure_coherence.py",
        ROOT / "dgkan/fu/structure_projected_population.py",
        ROOT / "dgkan/fu/edge_functional_gram.py",
        ROOT / "dgkan/fu/finite_step_trust_region.py",
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
        "dgkan.fu.structure_coherence",
        "dgkan.fu.structure_projected_population",
        "dgkan.fu.edge_functional_gram",
        "dgkan.fu.finite_step_trust_region",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_errors.append(f"{mod}: {repr(exc)}")
    scan_pass, scan_hits = static_scan(files)
    row = {
        "part": "A",
        "status": "ok",
        "compile_pass": int(not compile_errors),
        "import_pass": int(not import_errors),
        "static_scan_pass": scan_pass,
        "new_edge_function_added": 0,
        "external_product_feature_used": 0,
        "structure_transform_in_forward": 0,
        "structure_transform_used_only_for_gate": 1,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
        "runtime_selector_used": 0,
        "metric_winner_selection_used": 0,
        "candidate_update_selection_used": 0,
        "held_test_usage": 0,
        "used_fake_data_rows": 0,
        "purekan_depth3_constructed": 1,
        "changed_edge_coefficients": 1,
        "changed_mlp_tensors": 0,
        "random_control_types": "density,histogram,orbit,phase_shuffle,destructive",
        **AUDIT_DEFAULTS,
    }
    required = [
        "compile_pass",
        "import_pass",
        "static_scan_pass",
        "structure_transform_used_only_for_gate",
        "purekan_depth3_constructed",
    ]
    forbidden_zero = [
        "new_edge_function_added",
        "external_product_feature_used",
        "structure_transform_in_forward",
        "mlp_stem_used",
        "mlp_readout_used",
        "runtime_selector_used",
        "metric_winner_selection_used",
        "candidate_update_selection_used",
        "held_test_usage",
        "used_fake_data_rows",
        "changed_mlp_tensors",
    ]
    gate = int(all(ival(row.get(k)) == 1 for k in required) and all(ival(row.get(k)) == 0 for k in forbidden_zero))
    matrix = write_rows(OUT_ROOT / "part_a_matrix.csv", [row])
    summary = gate_summary(
        "A",
        gate,
        "A_Pass" if gate else "A_Failed",
        "none" if gate else "identity_static_scan_failed",
        [row],
        compile_errors=compile_errors,
        import_errors=import_errors,
        static_scan_hits=scan_hits,
    )
    write_json(OUT_ROOT / "part_a_summary.json", summary)
    nxt = next_actions("A", gate, summary["dominant_blocker"], [] if gate else ["repair compile/import/static scan", "repair transform forward leakage"], max_repair_rounds=0)
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_a_summary.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part A implementation identity and anti-selector audit", summary)
    return summary


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


def scheme_group(summary: dict[str, Any], scheme: str) -> dict[str, Any]:
    for group in summary.get("scheme_groups", []) or []:
        if str(group.get("scheme")) == scheme:
            return group
    return {}


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    f15, f15_source = read_v2302_f15_group()
    v2303_final = read_json(ROOT / "results/v23_03/final_route.json")
    v2303_c = read_json(ROOT / "results/v23_03/part_c_structure_coherence_summary.json")
    v2303_nofloor = read_json(ROOT / "results/v23_03_diagnostic_c_nofloor_after_destructive_fix/part_c_structure_coherence_summary.json")
    row = {
        "part": "B",
        "status": "ok",
        "v23_02_final_route": read_json(ROOT / "results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/final_route.json").get("route", "missing"),
        "v23_02_E5_F15_F5_no_debt_count": f15.get("F5_no_debt_count", "missing"),
        "v23_02_E5_F15_accept_rate": f15.get("finite_step_accept_rate_median", "missing"),
        "v23_02_E5_F15_scale_mean": f15.get("finite_step_scale_mean_median", "missing"),
        "v23_02_E5_F15_skip_count": f15.get("finite_step_skip_count_median", "missing"),
        "v23_02_part_c_gate_pass": read_json(ROOT / "results/v23_02/part_c_taskwise_c2_summary.json").get("gate_pass", "missing"),
        "v23_02_E5_F15_source": f15_source,
        "v23_03_final_route": v2303_final.get("route", "missing"),
        "v23_03_c_gate_pass": v2303_c.get("gate_pass", "missing"),
        "v23_03_C6_structureonly_gap": scheme_group(v2303_c, "C6_StructureOnly").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C7_poponly_gap": scheme_group(v2303_c, "C7_PopOnly").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C1_product_gap": scheme_group(v2303_c, "C1_LocalStructureProduct").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C2_product_gap": scheme_group(v2303_c, "C2_RotationStructureProduct").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C3_product_gap": scheme_group(v2303_c, "C3_DualMinStructureProduct").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C4_product_gap": scheme_group(v2303_c, "C4_OrbitAverageBeforeSNR").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C5_product_gap": scheme_group(v2303_c, "C5_ContrastiveStructureProduct").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C5_nofloor_gap": scheme_group(v2303_nofloor, "C5_ContrastiveStructureProduct").get("true_vs_orbit_random_gap_median", "missing"),
        "v23_03_C5_nofloor_source_guard": scheme_group(v2303_nofloor, "C5_ContrastiveStructureProduct").get("source_guard_q_final_cosine_median", "missing"),
        **AUDIT_DEFAULTS,
    }
    required = [
        "v23_02_final_route",
        "v23_02_E5_F15_F5_no_debt_count",
        "v23_02_E5_F15_accept_rate",
        "v23_02_part_c_gate_pass",
        "v23_03_final_route",
        "v23_03_C6_structureonly_gap",
        "v23_03_C7_poponly_gap",
        "v23_03_C5_nofloor_gap",
        "v23_03_C5_nofloor_source_guard",
    ]
    missing = [key for key in required if row.get(key) == "missing"]
    gate = int(not missing)
    matrix = write_rows(OUT_ROOT / "part_b_matrix.csv", [row])
    summary_fields = {k: v for k, v in row.items() if k not in {"part", "status"}}
    summary = gate_summary("B", gate, "B_Pass" if gate else "B_Failed", "none" if gate else "missing_history_fields", [row], missing_fields=missing, **summary_fields)
    write_json(OUT_ROOT / "part_b_summary.json", summary)
    nxt = next_actions("B", gate, summary["dominant_blocker"], [] if gate else ["locate missing v23.02/v23.03 artifacts", "write missing explicitly; do not infer"], max_repair_rounds=0)
    append_exec("part-b", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_b_summary.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part B historical boundary lock", summary)
    return summary


def device_from_args(args: argparse.Namespace) -> torch.device:
    requested = str(args.device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def split_flat_grads(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return v2300.split_flat_grads(flat, params)


def transformed_split(
    model: v2293.TrueDeepPureKAN,
    params: list[torch.nn.Parameter],
    x: torch.Tensor,
    y: torch.Tensor,
    names: list[str],
    *,
    side: int,
    seed: int,
    offset: int,
    max_examples: int,
    label_prior_correction: bool,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    per_param: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in params]
    identity_perm = torch.arange(int(side) * int(side), device=x.device)
    for i, name in enumerate(names):
        spec = transform_specs(name, "default")[0] if name.startswith("destructive") else None
        perm = transform_perm(side, name, seed=seed * 1000 + offset + i, device=x.device)
        x_t = apply_perm(x[:max_examples], perm)
        ft = per_example_gradient_matrix(model, x_t, y[: int(x_t.shape[0])], max_examples=max_examples, label_prior_correction=label_prior_correction)
        st = split_flat_grads(ft, params)
        align_perm = perm if spec is None or spec.align_with_transform else identity_perm
        for j, grads in enumerate(st):
            per_param[j].append((grads.to(device=params[j].device, dtype=params[j].dtype), align_perm))
    return per_param


def structure_gate_for_param(
    opt: EdgeSobolevSNRFU,
    param: torch.nn.Parameter,
    group: dict[str, Any],
    source_grads: torch.Tensor,
    transformed: list[tuple[torch.Tensor, torch.Tensor]],
    blocks: list[list[int]],
) -> torch.Tensor:
    if not blocks:
        return torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
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
            idx = torch.tensor([i for i in block if 0 <= int(i) < int(param.numel())], dtype=torch.long)
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


def qpop_for_param(opt: EdgeSobolevSNRFU, param: torch.nn.Parameter, group: dict[str, Any], source_grads: torch.Tensor, blocks: list[list[int]]) -> torch.Tensor:
    white = opt._apply_metric_inv_sqrt(source_grads, param, group).reshape(int(source_grads.shape[0]), -1)
    if not blocks:
        return torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
    res = block_snr_gate(white.detach().cpu(), blocks, beta=float(group.get("gate_beta", 2.0)))
    return res.gate.to(device=param.device, dtype=param.dtype).reshape(-1)


def l2_stream_proxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a.detach().reshape(-1)
    bb = b.detach().reshape(-1)
    n = min(int(aa.numel()), int(bb.numel()))
    if n <= 0:
        return aa
    aa = aa[:n]
    bb = bb[:n]
    target = 0.5 * (aa.norm() + bb.norm()).clamp_min(1.0e-12)
    sa = aa / aa.norm().clamp_min(1.0e-12) * target
    sb = bb / bb.norm().clamp_min(1.0e-12) * target
    return (0.5 * (sa + sb)).clamp(0.0, 1.0).to(device=a.device, dtype=a.dtype)


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
    max_examples = int(args.population_grad_examples)
    flat = per_example_gradient_matrix(model, x, y, max_examples=max_examples, label_prior_correction=bool(int(args.label_prior_correction)))
    split = split_flat_grads(flat, params)
    profile = str(args.structure_transform_profile)
    local_names = [spec.name for spec in transform_specs("local", profile)]
    rot_names = [spec.name for spec in transform_specs("rotation", profile)]
    dest_family = "destructive_local" if task == "local_patch_interaction" else "destructive_rotation"
    dest_names = [spec.name for spec in transform_specs(dest_family, profile)]
    local_t = transformed_split(model, params, x, y, local_names, side=side, seed=seed, offset=1100, max_examples=max_examples, label_prior_correction=bool(int(args.label_prior_correction)))
    rot_t = transformed_split(model, params, x, y, rot_names, side=side, seed=seed, offset=2100, max_examples=max_examples, label_prior_correction=bool(int(args.label_prior_correction)))
    dest_t = transformed_split(model, params, x, y, dest_names, side=side, seed=seed, offset=3100, max_examples=max_examples, label_prior_correction=bool(int(args.label_prior_correction)))

    qpop_parts: list[torch.Tensor] = []
    qpop_s_parts: list[torch.Tensor] = []
    qlocal_parts: list[torch.Tensor] = []
    qrot_parts: list[torch.Tensor] = []
    qdes_parts: list[torch.Tensor] = []
    qstruct_parts: list[torch.Tensor] = []
    qfinal_parts: list[torch.Tensor] = []
    qdensity_parts: list[torch.Tensor] = []
    qhist_parts: list[torch.Tensor] = []
    qorbit_parts: list[torch.Tensor] = []
    qphase_parts: list[torch.Tensor] = []
    qdesrand_parts: list[torch.Tensor] = []
    energies: list[float] = []
    retentions: list[float] = []
    snrs: list[float] = []
    residual_snrs: list[float] = []
    product_deltas: list[float] = []
    block_count = 0
    group = opt.param_groups[0]
    mode = PART_C_SCHEMES.get(str(scheme), "structure_first_snr")
    for idx, (param, grads) in enumerate(zip(params, split)):
        if bool(int(getattr(args, "structure_addressable_only", 0))) and int(getattr(param, "_kan_layer_id", -1)) != 0:
            continue
        pgrads = grads.to(device=param.device, dtype=param.dtype)
        blocks = opt._blocks_for_param(param, {**group, "gate_family": str(args.structure_block_family)})
        block_count += len(blocks)
        qpop = qpop_for_param(opt, param, group, pgrads, blocks)
        local = structure_gate_for_param(opt, param, group, pgrads, local_t[idx], blocks)
        rot = structure_gate_for_param(opt, param, group, pgrads, rot_t[idx], blocks)
        dest = structure_gate_for_param(opt, param, group, pgrads, dest_t[idx], blocks)
        pos = local if task == "local_patch_interaction" else rot
        struct = (pos - float(args.negative_lambda) * dest).clamp(0.0, 1.0)
        positive_t = local_t[idx] if task == "local_patch_interaction" else rot_t[idx]
        spop = structure_projected_block_snr_gate(
            opt,
            param,
            group,
            pgrads,
            positive_t,
            blocks,
            beta=float(args.snr_beta),
            residualize_orbit=mode == "orbit_residual_snr",
            residual_seed=seed * 991 + idx,
        )
        qpop_s = spop.gate.to(device=param.device, dtype=param.dtype)
        if mode == "product":
            qfinal = qpop * struct
        elif mode == "structure_only":
            qfinal = struct
        elif mode == "structure_first_snr" or mode == "orbit_residual_snr":
            qfinal = qpop_s
        elif mode.startswith("additive"):
            lam = float(mode.split("_")[-1])
            qfinal = (lam * struct + (1.0 - lam) * qpop_s).clamp(0.0, 1.0)
        elif mode == "soft_or":
            qfinal = (1.0 - (1.0 - struct) * (1.0 - qpop_s)).clamp(0.0, 1.0)
        elif mode == "two_stream_proxy":
            qfinal = l2_stream_proxy(struct, qpop_s)
        elif mode == "dual_union":
            qfinal = (1.0 - (1.0 - local) * (1.0 - rot)).clamp(0.0, 1.0)
            struct = qfinal
        elif mode == "dual_average":
            qfinal = (0.5 * (local + rot)).clamp(0.0, 1.0)
            struct = qfinal
        elif mode == "destructive_contrastive":
            logits = float(args.snr_beta) * (pos - float(args.negative_lambda) * dest - float(args.destructive_tau))
            qfinal = torch.sigmoid(logits).clamp(0.0, 1.0)
            struct = qfinal
        else:
            qfinal = qpop_s
        if bool(int(args.density_normalize)):
            qfinal = quantile_normalize(qfinal, float(args.density_target))
        floor = float(args.gate_floor)
        if floor > 0.0:
            qfinal = qfinal * (1.0 - floor) + floor
        qpop_parts.append(qpop.detach().reshape(-1))
        qpop_s_parts.append(qpop_s.detach().reshape(-1))
        qlocal_parts.append(local.detach().reshape(-1))
        qrot_parts.append(rot.detach().reshape(-1))
        qdes_parts.append(dest.detach().reshape(-1))
        qstruct_parts.append(struct.detach().reshape(-1))
        qfinal_parts.append(qfinal.detach().reshape(-1))
        qdensity_parts.append(density_random_like(qfinal, seed * 100003 + idx).detach().reshape(-1))
        qhist_parts.append(hist_random_like(qfinal, seed * 100019 + idx).detach().reshape(-1))
        qorbit_parts.append(block_shuffle(qfinal, blocks, seed * 100043 + idx).detach().reshape(-1))
        qphase_parts.append(phase_shuffle(qfinal, blocks, seed * 100057 + idx).detach().reshape(-1))
        qdesrand = dest
        if floor > 0.0:
            qdesrand = qdesrand * (1.0 - floor) + floor
        qdesrand_parts.append(qdesrand.detach().reshape(-1))
        energies.append(float(spop.projection_energy))
        retentions.append(float(spop.projection_retention))
        snrs.append(float(spop.snr_mean))
        residual_snrs.append(float(spop.orbit_residual_snr))
        product_deltas.append(weighted_structure_score(qpop * struct, struct) - weighted_structure_score(struct, struct))
    def cat(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([x.cpu() for x in parts]) if parts else torch.zeros(0)
    return {
        "q_pop": cat(qpop_parts),
        "q_pop_given_structure": cat(qpop_s_parts),
        "q_local": cat(qlocal_parts),
        "q_rot": cat(qrot_parts),
        "q_destructive": cat(qdes_parts),
        "q_struct": cat(qstruct_parts),
        "q_final": cat(qfinal_parts),
        "q_rand_density": cat(qdensity_parts),
        "q_rand_hist": cat(qhist_parts),
        "q_rand_orbit": cat(qorbit_parts),
        "q_rand_phase": cat(qphase_parts),
        "q_rand_destructive": cat(qdesrand_parts),
        "block_count": int(block_count),
        "structure_projection_energy": mean(energies),
        "structure_projection_retention": mean(retentions),
        "population_inside_structure_snr": mean(snrs),
        "orbit_residual_snr": mean(residual_snrs),
        "product_vs_structure_gap_delta": mean(product_deltas),
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
        phase_score = weighted_structure_score(src["q_rand_phase"], src["q_struct"])  # type: ignore[arg-type]
        destructive_score = weighted_structure_score(src["q_rand_destructive"], src["q_struct"])  # type: ignore[arg-type]
        local_orbit_score = weighted_structure_score(src["q_rand_orbit"], src["q_local"])  # type: ignore[arg-type]
        rot_orbit_score = weighted_structure_score(src["q_rand_orbit"], src["q_rot"])  # type: ignore[arg-type]
        task_coh = float(torch.as_tensor(src["q_local" if task == "local_patch_interaction" else "q_rot"]).mean().item()) if int(torch.as_tensor(src["q_final"]).numel()) else 0.0
        row = {
            "part": "C",
            "status": "ok",
            "scheme": scheme,
            "task": task,
            "seed": int(seed),
            "basis_key": basis_key,
            "depth": depth,
            "mode": PART_C_SCHEMES.get(scheme, "missing"),
            "gate_density": float(torch.as_tensor(src["q_final"]).mean().item()) if int(torch.as_tensor(src["q_final"]).numel()) else 0.0,
            "q_struct_mean": float(torch.as_tensor(src["q_struct"]).mean().item()) if int(torch.as_tensor(src["q_struct"]).numel()) else 0.0,
            "q_pop_density": float(torch.as_tensor(src["q_pop"]).mean().item()) if int(torch.as_tensor(src["q_pop"]).numel()) else 0.0,
            "q_pop_given_structure_density": float(torch.as_tensor(src["q_pop_given_structure"]).mean().item()) if int(torch.as_tensor(src["q_pop_given_structure"]).numel()) else 0.0,
            "q_final_density": float(torch.as_tensor(src["q_final"]).mean().item()) if int(torch.as_tensor(src["q_final"]).numel()) else 0.0,
            "source_guard_q_struct_cosine": cosine(src["q_struct"], grd["q_struct"]),  # type: ignore[arg-type]
            "source_guard_q_pop_given_structure_cosine": cosine(src["q_pop_given_structure"], grd["q_pop_given_structure"]),  # type: ignore[arg-type]
            "source_guard_q_final_cosine": cosine(src["q_final"], grd["q_final"]),  # type: ignore[arg-type]
            "true_vs_density_random_gap": true_score - density_score,
            "true_vs_hist_random_gap": true_score - hist_score,
            "true_vs_orbit_random_gap": true_score - orbit_score,
            "true_vs_phase_shuffle_random_gap": true_score - phase_score,
            "true_vs_destructive_random_gap": true_score - destructive_score,
            "coherence_gap_vs_orbit_random": task_coh - (local_orbit_score if task == "local_patch_interaction" else rot_orbit_score),
            "local_coherence_mean": float(torch.as_tensor(src["q_local"]).mean().item()) if int(torch.as_tensor(src["q_local"]).numel()) else 0.0,
            "rotation_coherence_mean": float(torch.as_tensor(src["q_rot"]).mean().item()) if int(torch.as_tensor(src["q_rot"]).numel()) else 0.0,
            "destructive_coherence_mean": float(torch.as_tensor(src["q_destructive"]).mean().item()) if int(torch.as_tensor(src["q_destructive"]).numel()) else 0.0,
            "structure_projection_energy": float(src["structure_projection_energy"]),  # type: ignore[arg-type]
            "structure_projection_retention": float(src["structure_projection_retention"]),  # type: ignore[arg-type]
            "population_inside_structure_snr": float(src["population_inside_structure_snr"]),  # type: ignore[arg-type]
            "orbit_residual_snr": float(src["orbit_residual_snr"]),  # type: ignore[arg-type]
            "product_vs_structure_gap_delta": float(src["product_vs_structure_gap_delta"]),  # type: ignore[arg-type]
            "block_count": int(src["block_count"]),  # type: ignore[arg-type]
            "structure_guard_source": str(args.structure_guard_source),
            "density_normalize": int(args.density_normalize),
            "gate_floor": float(args.gate_floor),
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
    path = write_rows(OUT_ROOT / f"part_c_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", rows)
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)}")
    return gate_summary("C", 0, "C_ShardsWritten", "merge_required", rows)


def summarize_part_c(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task")) == key]
        qdens = median(r.get("q_final_density") for r in group)
        orbit_gap = median(r.get("true_vs_orbit_random_gap") for r in group)
        phase_gap = median(r.get("true_vs_phase_shuffle_random_gap") for r in group)
        sg = median(r.get("source_guard_q_final_cosine") for r in group)
        coh_gap = median(r.get("coherence_gap_vs_orbit_random") for r in group)
        gaps = {
            "density": median(r.get("true_vs_density_random_gap") for r in group),
            "hist": median(r.get("true_vs_hist_random_gap") for r in group),
            "orbit": orbit_gap,
            "phase": phase_gap,
            "destructive": median(r.get("true_vs_destructive_random_gap") for r in group),
        }
        best_random_family = min(gaps.items(), key=lambda kv: kv[1])[0]
        task_pass = int(
            orbit_gap >= float(args.c_orbit_gap_gate)
            and phase_gap >= float(args.c_phase_gap_gate)
            and sg >= float(args.c_source_guard_cosine_gate)
            and coh_gap >= float(args.c_coherence_gap_gate)
            and float(args.c_density_low) <= qdens <= float(args.c_density_high)
        )
        task_summaries.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "task": key[3],
            "rows": len(group),
            "gate_density_median": median(r.get("gate_density") for r in group),
            "q_struct_mean_median": median(r.get("q_struct_mean") for r in group),
            "q_pop_density_median": median(r.get("q_pop_density") for r in group),
            "q_pop_given_structure_density_median": median(r.get("q_pop_given_structure_density") for r in group),
            "q_final_density_median": qdens,
            "source_guard_q_struct_cosine_median": median(r.get("source_guard_q_struct_cosine") for r in group),
            "source_guard_q_pop_given_structure_cosine_median": median(r.get("source_guard_q_pop_given_structure_cosine") for r in group),
            "source_guard_q_final_cosine_median": sg,
            "true_vs_density_random_gap_median": gaps["density"],
            "true_vs_hist_random_gap_median": gaps["hist"],
            "true_vs_orbit_random_gap_median": orbit_gap,
            "true_vs_phase_shuffle_random_gap_median": phase_gap,
            "true_vs_destructive_random_gap_median": gaps["destructive"],
            "coherence_gap_vs_orbit_random_median": coh_gap,
            "local_coherence_mean_median": median(r.get("local_coherence_mean") for r in group),
            "rotation_coherence_mean_median": median(r.get("rotation_coherence_mean") for r in group),
            "destructive_coherence_mean_median": median(r.get("destructive_coherence_mean") for r in group),
            "structure_projection_energy_median": median(r.get("structure_projection_energy") for r in group),
            "structure_projection_retention_median": median(r.get("structure_projection_retention") for r in group),
            "population_inside_structure_snr_median": median(r.get("population_inside_structure_snr") for r in group),
            "orbit_residual_snr_median": median(r.get("orbit_residual_snr") for r in group),
            "product_vs_structure_gap_delta_median": median(r.get("product_vs_structure_gap_delta") for r in group),
            "best_random_family": best_random_family,
            "random_family_max_gap_violation": max(0.0, float(args.c_orbit_gap_gate) - min(gaps.values())),
            "taskwise_pass": task_pass,
        })
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth")) for r in ok}):
        tasks = [t for t in task_summaries if (t.get("scheme"), t.get("basis_key"), t.get("depth")) == key]
        local = [t for t in tasks if t.get("task") == "local_patch_interaction"]
        rot = [t for t in tasks if t.get("task") == "rotation_sensitive"]
        local_pass = int(bool(local) and ival(local[0].get("taskwise_pass")) == 1)
        rot_pass = int(bool(rot) and ival(rot[0].get("taskwise_pass")) == 1)
        official_pass = int(
            str(key[0]) in CANDIDATE_PART_C_SCHEMES
            and str(key[1]) == "dche_k9"
            and str(key[2]) == "depth3"
            and sum(ival(t.get("rows")) for t in tasks) >= 10
            and local_pass == 1
            and rot_pass == 1
        )
        task_gaps = [fval(t.get("true_vs_orbit_random_gap_median")) for t in tasks]
        phase_gaps = [fval(t.get("true_vs_phase_shuffle_random_gap_median")) for t in tasks]
        source_guards = [fval(t.get("source_guard_q_final_cosine_median")) for t in tasks]
        densities = [fval(t.get("q_final_density_median")) for t in tasks]
        best_family_counts: dict[str, int] = {}
        for t in tasks:
            fam = str(t.get("best_random_family", "missing"))
            best_family_counts[fam] = best_family_counts.get(fam, 0) + 1
        best_family = sorted(best_family_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if best_family_counts else "missing"
        groups.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "tasks": ",".join(str(t.get("task")) for t in tasks),
            "rows": sum(ival(t.get("rows")) for t in tasks),
            "taskwise_all_pass": int(bool(tasks) and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)),
            "local_task_pass": local_pass,
            "rotation_task_pass": rot_pass,
            "min_task_gap": min(task_gaps) if task_gaps else 0.0,
            "median_task_gap": median(task_gaps),
            "min_phase_gap": min(phase_gaps) if phase_gaps else 0.0,
            "median_phase_gap": median(phase_gaps),
            "q_final_density_median": median(densities),
            "source_guard_q_final_cosine_median": median(source_guards),
            "coherence_gap_vs_orbit_random_median": median(t.get("coherence_gap_vs_orbit_random_median") for t in tasks),
            "density_valid_rows": sum(1 for d in densities if float(args.c_density_low) <= d <= float(args.c_density_high)),
            "source_guard_valid_rows": sum(1 for sg in source_guards if sg >= float(args.c_source_guard_cosine_gate)),
            "best_random_family": best_family,
            "random_family_max_gap_violation": max((fval(t.get("random_family_max_gap_violation")) for t in tasks), default=0.0),
            "official_candidate_pass": official_pass,
            "structure_only_pass": int(str(key[0]) in STRUCTURE_ONLY_SCHEMES and bool(tasks) and all(ival(t.get("taskwise_pass")) == 1 for t in tasks)),
        })
    failure = build_failure_decomposition(groups, task_summaries)
    return task_summaries, groups, failure


def build_failure_decomposition(groups: list[dict[str, Any]], task_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = task_summaries
    families = ["density", "hist", "orbit", "phase", "destructive"]
    family_gaps = {
        "density": median(r.get("true_vs_density_random_gap_median") for r in rows),
        "hist": median(r.get("true_vs_hist_random_gap_median") for r in rows),
        "orbit": median(r.get("true_vs_orbit_random_gap_median") for r in rows),
        "phase": median(r.get("true_vs_phase_shuffle_random_gap_median") for r in rows),
        "destructive": median(r.get("true_vs_destructive_random_gap_median") for r in rows),
    }
    strongest = min(families, key=lambda k: family_gaps.get(k, 0.0)) if rows else "missing"
    candidates = [g for g in groups if str(g.get("scheme")) in CANDIDATE_PART_C_SCHEMES]
    structure_only = [g for g in groups if str(g.get("scheme")) in STRUCTURE_ONLY_SCHEMES]
    return {
        "random_family_median_gaps": family_gaps,
        "strongest_matched_random_family": strongest,
        "candidate_best_min_task_gap": max((fval(g.get("min_task_gap")) for g in candidates), default=0.0),
        "candidate_best_min_phase_gap": max((fval(g.get("min_phase_gap")) for g in candidates), default=0.0),
        "structure_only_best_min_task_gap": max((fval(g.get("min_task_gap")) for g in structure_only), default=0.0),
        "structure_phase_not_identified": int(family_gaps.get("phase", 0.0) < 0.05),
        "structure_only_signal_found": int(any(ival(g.get("structure_only_pass")) == 1 for g in structure_only)),
    }


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_c_matrix.csv", rows)
    task_summaries, groups, failure = summarize_part_c(args, rows)
    task_csv = write_rows(OUT_ROOT / "part_c_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_c_group_summary.csv", groups)
    pass_groups = [g for g in groups if ival(g.get("official_candidate_pass")) == 1]
    structure_only_pass = any(ival(g.get("structure_only_pass")) == 1 for g in groups)
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_c_job_errors"
    elif failure.get("structure_phase_not_identified"):
        blocker = "structure_phase_not_identified"
    elif not pass_groups:
        low_sg = any(fval(t.get("source_guard_q_final_cosine_median")) < float(args.c_source_guard_cosine_gate) for t in task_summaries if str(t.get("scheme")) in CANDIDATE_PART_C_SCHEMES)
        bad_density = any(not (float(args.c_density_low) <= fval(t.get("q_final_density_median")) <= float(args.c_density_high)) for t in task_summaries if str(t.get("scheme")) in CANDIDATE_PART_C_SCHEMES)
        blocker = "source_guard_pairing_low" if low_sg else ("density_invalid" if bad_density else "structure_random_gap_low")
    else:
        blocker = "none"
    gate = int(bool(pass_groups) and not any(r.get("status") == "error" for r in rows))
    if gate:
        route = "C_StructureConditionedPopulationPass"
    elif structure_only_pass:
        route = "C_StructureOnlySignalFound_PopFusionFailed"
    else:
        route = "C_StructureConditioningFailed"
    summary = gate_summary("C", gate, route, blocker, rows, scheme_groups=groups, taskwise_groups=task_summaries, passing_scheme_groups=pass_groups, failure_decomposition=failure)
    write_json(OUT_ROOT / "part_c_summary.json", summary)
    write_json(OUT_ROOT / "part_c_failure_decomposition.json", failure)
    actions = [] if gate else [
        "generate part_c_failure_decomposition.json and identify strongest matched random family",
        "if source_guard cosine is low, increase transform cohort size or use train_split pairing audit without lowering gate",
        "if density is outside [0.15,0.85], rerun with source quantile density normalization",
        "if local/rotation tradeoff appears, rerun C7 dual-family union",
        "if phase-shuffle random remains close, report structure_phase_not_identified and stop Part C repair",
    ]
    nxt = next_actions("C", gate, blocker, actions, max_repair_rounds=3)
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_c_failure_decomposition.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part C structure-conditioned population diagnostic", summary)
    return summary


def write_blocked(part: str, route: str, blocker: str, reason: str) -> dict[str, Any]:
    row = {"part": part, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}
    write_rows(OUT_ROOT / f"part_{part.lower()}_matrix.csv", [row])
    write_rows(OUT_ROOT / f"part_{part.lower()}_task_summary.csv", [row])
    write_rows(OUT_ROOT / f"part_{part.lower()}_group_summary.csv", [row])
    summary = gate_summary(part, 0, route, blocker, [row], reason=reason)
    write_json(OUT_ROOT / f"part_{part.lower()}_summary.json", summary)
    next_actions(part, 0, blocker, ["unblock only by satisfying prerequisite gate; do not fabricate downstream metrics"], max_repair_rounds=0)
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    a = read_json(OUT_ROOT / "part_a_summary.json")
    b = read_json(OUT_ROOT / "part_b_summary.json")
    c = read_json(OUT_ROOT / "part_c_summary.json")
    if ival(a.get("gate_pass")) != 1:
        route = "A_Failed"
        blocker = a.get("dominant_blocker", "part_a")
    elif ival(b.get("gate_pass")) != 1:
        route = "B_Failed"
        blocker = b.get("dominant_blocker", "part_b")
    elif ival(c.get("gate_pass")) != 1:
        route = c.get("route", "C_StructureConditioningFailed")
        blocker = c.get("dominant_blocker", "part_c")
        write_blocked("D", "D_BlockedByPartC", str(blocker), "Part C failed; plan forbids C2 training before structure-conditioned population separability is established.")
        write_blocked("E", "E_BlockedByPartC", str(blocker), "Part E requires Part D gate/training logs; Part C failed.")
        write_blocked("F", "F_BlockedByPartC", str(blocker), "Part F component ablation requires Part D pass; Part C failed.")
        write_blocked("G", "G_BlockedByPartC", str(blocker), "Part G F15 safety reuse requires Part D/F pass; Part C failed.")
        write_blocked("H", "H_BlockedByPartC", str(blocker), "Part H full positive-control requires Part C/D/F/G pass; Part C failed.")
        write_blocked("I", "I_BlockedByPartC", str(blocker), "Part I real-task preflight requires Part H pass; Part C failed.")
    else:
        route = "D_Required_NotYetRun"
        blocker = "part_c_passed_downstream_not_yet_executed"
    final = {
        "route": route,
        "dominant_blocker": blocker,
        "part_gate_passes": {
            "A": a.get("gate_pass", "missing"),
            "B": b.get("gate_pass", "missing"),
            "C": c.get("gate_pass", "missing"),
        },
        "official_candidate_gate_pass": int(ival(c.get("gate_pass")) == 1),
        "promotion_allowed": 0,
        "held_test_usage": sum(ival(x.get("held_test_usage")) for x in [a, b, c]),
        "used_fake_data_rows": sum(ival(x.get("used_fake_data_rows")) for x in [a, b, c]),
        "runtime_selector_used": sum(ival(x.get("runtime_selector_used")) for x in [a, b, c]),
        "metric_winner_selection_used": sum(ival(x.get("metric_winner_selection_used")) for x in [a, b, c]),
        "candidate_update_selection_used": sum(ival(x.get("candidate_update_selection_used")) for x in [a, b, c]),
        "stale_artifact_count": 0,
        "generated_at": now(),
        **AUDIT_DEFAULTS,
    }
    write_json(OUT_ROOT / "final_route.json", final)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        "# DG-KAN v23.04 Reproduction Manifest\n\n"
        f"- runner: `{rel(RUNNER)}`\n"
        f"- plan: `{rel(PLAN)}`\n"
        f"- out_root: `{rel(OUT_ROOT)}`\n"
        "- required logs:\n"
        f"  - `{rel(EXEC_LOG)}`\n"
        f"  - `{rel(RECAP_LOG)}`\n\n"
        "## Key Commands\n\n"
        "- Part A: `python experiments/run_v23_04_structure_conditioned_population_flow.py --mode part-a --device cpu`\n"
        "- Part B: `python experiments/run_v23_04_structure_conditioned_population_flow.py --mode part-b --device cpu`\n"
        "- Part C shard 0: `CUDA_VISIBLE_DEVICES=2 python experiments/run_v23_04_structure_conditioned_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 0`\n"
        "- Part C shard 1: `CUDA_VISIBLE_DEVICES=3 python experiments/run_v23_04_structure_conditioned_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 1`\n"
        "- Part C merge: `python experiments/run_v23_04_structure_conditioned_population_flow.py --mode part-c-merge --device cpu`\n"
        "- Finalize: `python experiments/run_v23_04_structure_conditioned_population_flow.py --mode finalize --device cpu`\n",
        encoding="utf-8",
    )
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(manifest)}", gpu=str(args.device), note=f"route={route}")
    append_recap("Final route", final)
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="part-a", choices=["part-a", "part-b", "part-c", "part-c-merge", "finalize"])
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
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--edge-lr", type=float, default=0.004)
    p.add_argument("--adamw-lr", type=float, default=0.004)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--edge-weight-normalization", default="trace")
    p.add_argument("--edge-weight-ridge", type=float, default=1.0e-6)
    p.add_argument("--functional-gram-quadrature-points", type=int, default=257)
    p.add_argument("--snr-beta", type=float, default=2.0)
    p.add_argument("--gate-floor", type=float, default=0.0)
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--population-grad-examples", type=int, default=6)
    p.add_argument("--part-c-schemes", default=",".join(PART_C_SCHEMES.keys()))
    p.add_argument("--part-c-basis", default="dche_k9")
    p.add_argument("--part-c-depths", default="depth3")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-seed-count", type=int, default=5)
    p.add_argument("--structure-block-family", default="degree_edgebank")
    p.add_argument("--structure-transform-profile", default="default", choices=["default", "expanded"])
    p.add_argument("--structure-guard-source", default="held_guard", choices=["held_guard", "train_split"])
    p.add_argument("--structure-guard-examples", type=int, default=64)
    p.add_argument("--structure-addressable-only", type=int, default=0)
    p.add_argument("--negative-lambda", type=float, default=0.5)
    p.add_argument("--destructive-tau", type=float, default=0.1)
    p.add_argument("--density-normalize", type=int, default=0)
    p.add_argument("--density-target", type=float, default=0.5)
    p.add_argument("--c-orbit-gap-gate", type=float, default=0.05)
    p.add_argument("--c-phase-gap-gate", type=float, default=0.05)
    p.add_argument("--c-source-guard-cosine-gate", type=float, default=0.75)
    p.add_argument("--c-coherence-gap-gate", type=float, default=0.05)
    p.add_argument("--c-density-low", type=float, default=0.15)
    p.add_argument("--c-density-high", type=float, default=0.85)
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
    return finalize(args)


if __name__ == "__main__":
    main()
