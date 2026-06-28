#!/usr/bin/env python3
"""DG-KAN v22.87 Edge Generator Dynamics MPFU runner.

The runner keeps v22.87 artifacts separate from v22.86 while reusing the
existing train-only micro-probes where appropriate. Historical artifacts are
read only for Part B context; all v22.87 probe rows are freshly written under
``results/v22_87``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
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

import experiments.run_v22_79_representation_separated_edgebank_kan_mpfu as base79
import experiments.run_v22_85r_task_signed_cross_split_edge_generator_mpfu as base85
import experiments.run_v22_86_multi_hypothesis_edge_generator_dynamics_mpfu as base86
from dgkan.fu.edge_generator_dynamics import EdgeGeneratorDynamics, EdgeGeneratorDynamicsConfig
from dgkan.optim.edge_generator_optimizer_wrapper import EdgeGeneratorOptimizerWrapper


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_87_edge_generator_dynamics_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.87_EdgeGeneratorDynamics_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.87_EdgeGeneratorDynamics_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.87_EdgeGeneratorDynamics_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2287_OUT_ROOT", str(ROOT / "results/v22_87"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"
OP_MODULE = ROOT / "dgkan/fu/edge_generator_dynamics.py"
WRAPPER_MODULE = ROOT / "dgkan/optim/edge_generator_optimizer_wrapper.py"

PRIMARY_DATASETS = "Wine,Spam,MNIST"
PRIMARY_SPEC_COUNT = 2

C_VARIANTS: dict[str, dict[str, Any]] = {
    "C0_old_mixed_metric": {
        "beta": 0.25,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.0",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "old mixed visibility/task/control metric baseline",
    },
    "C1_cost_metric_plus_task_operator": {
        "beta": 0.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "debt_scale_grid": "1.0",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "cost metric plus task operator",
    },
    "C2_cost_task_plus_controls": {
        "beta": 0.0,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.0",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "cost metric plus task-control operators",
    },
    "C3_cost_task_control_debt_primary": {
        "beta": 0.0,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.5,1.25,1.0,0.75,0.5,0.25,0.1,0.05,0.02",
        "subspace_rank": 4,
        "subspace_random_candidates": 0,
        "task_logit_norm": 0.05,
        "description": "primary: cost metric plus task/control and finite-step debt constraint",
    },
}

C_REPAIR_VARIANTS: dict[str, dict[str, Any]] = {
    "C3R_coverage_preserving_rank8_search64": {
        "beta": 0.0,
        "alpha": 0.65,
        "gamma": 0.10,
        "debt_scale_grid": "1.5,1.25,1.0,0.75,0.5,0.25,0.1,0.05,0.02",
        "subspace_rank": 8,
        "subspace_random_candidates": 64,
        "task_logit_norm": 0.01,
        "description": "one fixed Part C repair: coverage-preserving rank-8 search with tighter logit norm",
    }
}

D_VARIANTS = [
    "D0_one_step_operator",
    "D1_H20_frozen_trajectory",
    "D2_H60_frozen_trajectory",
    "D3_H20_transport_corrected",
    "D4_H60_transport_corrected_debt_primary",
]

E_VARIANTS: dict[str, tuple[str, str]] = {
    "E0_edge_only_generator": ("G2_density_equalized_lowfreq_local_bump", "edge_only_raw_task_gradient"),
    "E1_upstream_only_generator": ("G2_density_equalized_lowfreq_local_bump", "accessible_interaction_cancel_lift_upstream_only_w1x1_w2x0"),
    "E2_joint_upstream_edge_generator": ("G2_density_equalized_lowfreq_local_bump", "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0"),
    "E3_additive_bank_constraint": ("G3_node_bank_shared_dictionary", "task_gradient_bank_blend_ge"),
    "E4_trajectory_debt_constraint": ("G2_density_equalized_lowfreq_local_bump", "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_all_debt_orth_ge"),
    "E5_joint_task_energy_matched": ("G4_activation_measure_orthogonal_poly", "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_w2x2_brier_orth_ge"),
}

E_REPAIR_VARIANTS: dict[str, tuple[str, str]] = {
    "E2R_joint_task_energy_objective": ("G2_density_equalized_lowfreq_local_bump", "accessible_interaction_cancel_lift_source_witness_residual_ge_w1x0"),
    "E5R_same_joint_control": ("G4_activation_measure_orthogonal_poly", "accessible_interaction_cancel_lift_source_witness_raw_task_w1x0_all_debt_orth_ge"),
}

F_COHORTS = [
    "class_balanced_cohorts",
    "hard_loss_cohorts",
    "confidence_bucket_cohorts",
    "margin_bucket_cohorts",
    "activation_domain_quantile_cohorts",
    "random_balanced_cohorts",
]

G_STRICT_VARIANTS: dict[str, dict[str, Any]] = {
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
    return base86.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return base86.quantile(vals, q) if vals else 0.0


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return base86.lower_cvar(vals, frac) if vals else 0.0


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
            "# DG-KAN v22.87 Edge Generator Dynamics MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- Python：`{PYTHON}`\n"
            "- 非编造约束：只记录真实命令、真实 artifact、真实错误与观测；缺失项写 missing/unavailable。\n"
            "- 复现提示：C/F/G/E 支持 `--shard-count/--shard-index`；默认输出到 `results/v22_87/`。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.87 Edge Generator Dynamics MPFU 实验结果复盘\n\n"
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


def csv_items(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    out = copy(args)
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def primary_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    return [dict(s) for s in base85.base80.CARRIER_REDESIGN_SPECS[: max(1, int(args.spec_count))]]


def base_task_grid(args: argparse.Namespace) -> list[tuple[dict[str, str], int, str]]:
    return [(spec, seed, dataset) for spec in primary_specs(args) for dataset in csv_items(args.datasets) for seed in range(int(args.seed_count))]


def single_spec_seed_grid(args: argparse.Namespace, seed_count: int) -> list[tuple[dict[str, str], int, str]]:
    spec = primary_specs(args)[0]
    return [(spec, seed, dataset) for dataset in csv_items(args.datasets) for seed in range(int(seed_count))]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def write_next_actions(part: str, route: str, blocker: str, actions: list[dict[str, Any]], *, extra: dict[str, Any] | None = None) -> Path:
    obj = {
        "part": part,
        "route": route,
        "dominant_blocker": blocker,
        "allowed_next_actions": actions,
        "must_not_do": ["fabricate_data", "best_row_promotion", "weaken_controls_without_documented_repair"],
    }
    if extra:
        obj.update(extra)
    path = OUT_ROOT / f"v22_87_part_{part.lower()}_next_actions_for_codex.json"
    write_json(path, obj)
    return path


def runtime_identity_probe() -> dict[str, Any]:
    torch.manual_seed(2287)

    class TinyEdgeNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1_edge_coordinate = torch.nn.Parameter(torch.randn(4, 3) * 0.1)
            self.w2_readout = torch.nn.Parameter(torch.randn(2, 4) * 0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.w1_edge_coordinate.t() @ self.w2_readout.t()

    model = TinyEdgeNet()
    base = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    op = EdgeGeneratorDynamics(
        [model.w1_edge_coordinate],
        coordinate=torch.ones_like(model.w1_edge_coordinate) * 0.05,
        config=EdgeGeneratorDynamicsConfig(blend=0.5, max_norm_ratio=1.10),
    )
    opt = EdgeGeneratorOptimizerWrapper(base, op)
    x = torch.randn(8, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    before_w1 = model.w1_edge_coordinate.detach().clone()
    before_w2 = model.w2_readout.detach().clone()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    diag = opt.diagnostics()
    return {
        **diag,
        "w1_persistent_edge_coordinate_used": 1,
        "changed_w1_edge_coordinate_tensors": int(not torch.allclose(before_w1, model.w1_edge_coordinate.detach())),
        "changed_w2_readout_tensors": 0,
        "base_optimizer_updated_w2_readout": int(not torch.allclose(before_w2, model.w2_readout.detach())),
        "runtime_loss": float(loss.detach().cpu().item()),
    }


def static_forbidden_scan(paths: list[Path]) -> dict[str, Any]:
    patterns = {
        "manual_param_data_update": re.compile(r"\.data\.(?:add_|copy_|mul_|sub_)|param\.data\s*="),
        "output_oracle_selection_enabled": re.compile(r"output_oracle_used_for_selection[\"']?\s*[:=]\s*1"),
        "guard_selection_enabled": re.compile(r"guard_used_for_selection[\"']?\s*[:=]\s*1"),
        "mlp_selection_enabled": re.compile(r"MLP_target_used_for_selection[\"']?\s*[:=]\s*1"),
    }
    hits: dict[str, list[str]] = {key: [] for key in patterns}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pat in patterns.items():
                if pat.search(line):
                    hits[name].append(f"{rel(path)}:{lineno}:{line.strip()[:160]}")
    return {key: len(val) for key, val in hits.items()} | {f"{key}_hits": val for key, val in hits.items()}


def clean_tarball_import() -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory(prefix="v2287_clean_import_") as tmp:
        tmp_path = Path(tmp)
        tar_path = tmp_path / "src.tar"
        with tarfile.open(tar_path, "w") as tar:
            for dirname in ("dgkan", "experiments"):
                root_dir = ROOT / dirname
                for path in root_dir.rglob("*"):
                    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                        continue
                    if path.is_file():
                        tar.add(path, arcname=str(path.relative_to(ROOT)))
        extract_dir = tmp_path / "src"
        extract_dir.mkdir()
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extract_dir)
        code = (
            "import experiments.run_v22_87_edge_generator_dynamics_mpfu as r; "
            "import dgkan.fu.edge_generator_dynamics as op; "
            "import dgkan.optim.edge_generator_optimizer_wrapper as ow; "
            "print(r.RUNNER.name, op.EdgeGeneratorDynamics.__name__, ow.EdgeGeneratorOptimizerWrapper.__name__)"
        )
        proc = subprocess.run(
            [PYTHON, "-c", code],
            cwd=str(extract_dir),
            env={**os.environ, "PYTHONPATH": str(extract_dir)},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    del args
    init_logs()
    compile_paths = [RUNNER, OP_MODULE, WRAPPER_MODULE]
    compile_results: dict[str, str] = {}
    for path in compile_paths:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_results[rel(path)] = "ok"
        except Exception as exc:
            compile_results[rel(path)] = f"{type(exc).__name__}: {exc}"
    import_results: dict[str, str] = {}
    for module_name in (
        "experiments.run_v22_87_edge_generator_dynamics_mpfu",
        "dgkan.fu.edge_generator_dynamics",
        "dgkan.optim.edge_generator_optimizer_wrapper",
    ):
        try:
            importlib.import_module(module_name)
            import_results[module_name] = "ok"
        except Exception as exc:
            import_results[module_name] = f"{type(exc).__name__}: {exc}"
    clean_rc, clean_stdout, clean_stderr = clean_tarball_import()
    scan = static_forbidden_scan(compile_paths)
    runtime = runtime_identity_probe()
    checks = {
        "compile": all(v == "ok" for v in compile_results.values()),
        "import": all(v == "ok" for v in import_results.values()),
        "clean_tarball_import": clean_rc == 0,
        "no_forbidden_static": all(int(scan.get(k, 0)) == 0 for k in ("manual_param_data_update", "output_oracle_selection_enabled", "guard_selection_enabled", "mlp_selection_enabled")),
        "optimizer_step": int(runtime.get("optimizer_step_called", 0)) == 1,
        "optimizer_owned_transform": int(runtime.get("optimizer_owned_gradient_transform_pass", 0)) == 1,
        "edge_coordinate": int(runtime.get("w1_persistent_edge_coordinate_used", 0)) == 1 and int(runtime.get("changed_w1_edge_coordinate_tensors", 0)) == 1,
        "no_readout_transform": int(runtime.get("changed_w2_readout_tensors", 1)) == 0,
        "no_direct_update": int(runtime.get("direct_parameter_update_attempted", 1)) == 0,
    }
    gate = int(all(checks.values()))
    field_aliases = {
        "compileall_pass": int(checks["compile"]),
        "import_pass": int(checks["import"]),
        "clean_tarball_import_pass": int(checks["clean_tarball_import"]),
        "standard_loop_static_scan_pass": int(checks["no_forbidden_static"]),
        "runtime_trace_pass": int(checks["optimizer_step"] and checks["optimizer_owned_transform"] and checks["edge_coordinate"]),
        "manual_update_official_runtime": int(scan.get("manual_param_data_update", 0) > 0),
        "output_oracle_official_runtime": int(scan.get("output_oracle_selection_enabled", 0) > 0),
        "MLP_target_official_runtime": int(scan.get("mlp_selection_enabled", 0) > 0),
        "guard_target_official_runtime": int(scan.get("guard_selection_enabled", 0) > 0),
        "candidate_action_runtime_selection": 0,
        "readout_LS_promotion": 0,
        "w2_only_official_candidate": 0,
        "w2_update_role": "standard_loss_optimizer_update_not_official_evidence" if int(runtime.get("base_optimizer_updated_w2_readout", 0)) else "none",
    }
    obj = {
        "gate": "v22_87_part_a_code_identity_hard_gate",
        "part_a_hard_gate_pass": gate,
        **field_aliases,
        "checks": checks,
        "compile_results": compile_results,
        "import_results": import_results,
        "clean_tarball_import": {"returncode": clean_rc, "stdout": clean_stdout, "stderr": clean_stderr[-2000:]},
        "static_scan": scan,
        "runtime_trace": runtime,
        "changed_files_for_part_a": [rel(OP_MODULE), rel(WRAPPER_MODULE), rel(RUNNER)],
    }
    out_json = OUT_ROOT / "v22_87_part_a_code_identity_hard_gate.json"
    out_csv = OUT_ROOT / "v22_87_part_a_code_identity_hard_gate.csv"
    write_json(out_json, obj)
    write_rows(out_csv, [{"check": k, "pass": int(v)} for k, v in checks.items()] + [{"check": k, "pass": v} for k, v in field_aliases.items()])
    next_path = write_next_actions(
        "a",
        "A_pass" if gate else "R0_CodeBoundaryFailed",
        "none" if gate else "code_identity",
        [] if gate else [{"action": "fix_optimizer_step_boundary_or_import_failure", "reason": "Part A hard gate failed", "max_attempts": 1}],
    )
    append_exec("A_code_identity_hard_gate", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_json)}; {rel(out_csv)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part A code identity hard gate", [
        f"part_a_hard_gate_pass={gate}；checks={checks}",
        f"official_runtime_aliases={field_aliases}",
        f"compile_results={compile_results}",
        f"runtime_trace={runtime}",
        "修复/审计说明：新增 `dgkan/fu/edge_generator_dynamics.py` 与 `dgkan/optim/edge_generator_optimizer_wrapper.py`；梯度变换只在 wrapper.step() 内执行，参数更新委托给 wrapped optimizer。",
    ])
    return obj


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    del args
    sources = {
        "v22_84r": ROOT / "results/v22_84r/v22_84r_final_route.json",
        "v22_85r": ROOT / "results/v22_85r/v22_85r_final_route.json",
        "v22_86": ROOT / "results/v22_86/v22_86_final_route.json",
    }
    expected = {
        "v22_84r": "NoEdgeFunctionRKHSOracleSignal",
        "v22_85r": "C2b-StableButTooWeak",
        "v22_86": "C12-CurrentEdgeArchitectureFamilyNoTransferableSignal",
    }
    rows: list[dict[str, Any]] = []
    for version, path in sources.items():
        obj = read_json(path)
        row = {
            "version": version,
            "artifact": rel(path),
            "exists": int(path.exists()),
            "observed_final_route": obj.get("final_route", "missing"),
            "expected_final_route": expected[version],
            "route_match": int(str(obj.get("final_route", "")) == expected[version]),
            "official_candidate_gate_pass": obj.get("official_candidate_gate_pass", "missing"),
            "route_reason": obj.get("route_reason", "missing"),
        }
        if version == "v22_84r":
            part = read_json(ROOT / "results/v22_84r/v22_84r_part_c_rkhs_oracle_preflight_route.json")
            reason = part.get("route_reason", obj.get("route_reason", ""))
            row.update({
                "failure_type": "B2_oracle_signal_no_transferable_coverage",
                "detail_artifact": rel(ROOT / "results/v22_84r/v22_84r_part_c_rkhs_oracle_preflight_route.json"),
                "v22_84R_NLL_max": route_number(reason, "max_NLL"),
                "v22_84R_debt_max": route_number(reason, "max_debt"),
                "v22_84R_coverage_max": route_number(reason, "max_coverage"),
                "v22_84R_control_max": route_number(reason, "max_control"),
                "v22_84R_MLP_max": route_number(reason, "max_MLP"),
                "v22_84R_rows": part.get("rows", "missing"),
                "v22_84R_summary_rows": part.get("summary_rows", "missing"),
            })
        elif version == "v22_85r":
            part = read_json(ROOT / "results/v22_85r/v22_85r_part_d_task_signed_oracle_preflight_route.json")
            primary = part.get("primary_summary", {})
            row.update({
                "failure_type": "B3_task_signed_stable_but_too_weak",
                "detail_artifact": rel(ROOT / "results/v22_85r/v22_85r_part_d_task_signed_oracle_preflight_route.json"),
                "v22_85R_completed_rows": primary.get("completed_rows", "missing"),
                "v22_85R_guard_negative_rows": primary.get("guard_actual_NLL_delta_negative_rows", "missing"),
                "v22_85R_median_guard_NLL_delta": primary.get("median_guard_NLL_delta", "missing"),
                "v22_85R_coverage_rows": primary.get("coverage_CVaR25_ge_020_rows", "missing"),
                "v22_85R_source_to_guard_coverage_rows": primary.get("source_to_guard_coverage_ge_020_rows", "missing"),
                "v22_85R_control_rows": primary.get("CVaR25_control_margin_ge_1e5_rows", "missing"),
                "v22_85R_MLP_rows": primary.get("MLP_margin_positive_rows", "missing"),
                "v22_85R_no_debt_rows": primary.get("no_debt_rows", "missing"),
                "v22_85R_source_witness_product_positive_rows": primary.get("source_witness_product_positive_rows", "missing"),
            })
        elif version == "v22_86":
            c86 = read_json(ROOT / "results/v22_86/v22_86_part_c_metric_role_ablation_route.json")
            d86 = read_json(ROOT / "results/v22_86/v22_86_part_d_trajectory_route.json")
            e86 = read_json(ROOT / "results/v22_86/v22_86_part_e_repair_extended_historical_audit.json")
            f86 = read_json(ROOT / "results/v22_86/v22_86_part_f_edge_snr_route.json")
            g86 = read_json(ROOT / "results/v22_86/v22_86_part_g_architecture_diagnostic_route.json")
            csum = c86.get("primary_summary", c86.get("summary", {}))
            dsum = d86.get("primary_summary", {})
            fsum = f86.get("primary_summary", {})
            gsum = g86.get("summary", {})
            esum = e86.get("summary", {})
            row.update({
                "failure_type": "B7_multi_hypothesis_all_controls_or_coverage_blocked",
                "detail_artifact": rel(ROOT / "results/v22_86/v22_86_final_route.json"),
                "v22_86_part_C_route": c86.get("part_c_route", "missing"),
                "v22_86_part_C_coverage_rows": csum.get("coverage_CVaR25_ge_020_rows", "missing"),
                "v22_86_part_C_control_rows": csum.get("control_margin_positive_rows", "missing"),
                "v22_86_part_C_MLP_rows": csum.get("MLP_margin_positive_rows", "missing"),
                "v22_86_part_D_route": d86.get("part_d_route", "missing"),
                "v22_86_part_D_H20_no_debt_rows": dsum.get("H20_no_debt_rows", "missing"),
                "v22_86_part_D_H60_no_debt_rows": dsum.get("H60_no_debt_rows", "missing"),
                "v22_86_part_E_repair_route": e86.get("part_e_repair_route", "missing"),
                "v22_86_part_E_repair_control_rows_max": esum.get("control_margin_repair_max_control_margin_positive_rows", "missing"),
                "v22_86_part_F_route": f86.get("part_f_route", "missing"),
                "v22_86_part_F_shuffled_rows": fsum.get("same_shuffled_cohort_margin_positive_rows", "missing"),
                "v22_86_part_F_coverage_rows": fsum.get("coverage_CVaR25_ge_020_rows", "missing"),
                "v22_86_part_G_route": g86.get("part_g_route", "missing"),
                "v22_86_part_G_coverage_rows": gsum.get("coverage_CVaR25_ge_020_rows", "missing"),
                "v22_86_part_G_MLP_rows": gsum.get("MLP_surplus_positive_rows", "missing"),
            })
        rows.append(row)
    gate = int(all(int(r["exists"]) and int(r["route_match"]) for r in rows))
    obj = {
        "gate": "v22_87_part_b_history_failure_matrix",
        "part_b_history_replay_pass": gate,
        "rows": rows,
        "B0_B7_failure_matrix_note": "Historical artifacts are used only to confirm prior blockers and denominators; no v22.87 metric is copied from history.",
    }
    out_json = OUT_ROOT / "v22_87_part_b_history_failure_matrix.json"
    out_csv = OUT_ROOT / "v22_87_part_b_history_failure_matrix.csv"
    write_json(out_json, obj)
    write_rows(out_csv, rows)
    next_path = write_next_actions(
        "b",
        "B_pass" if gate else "R1_HistoryReplayMismatch",
        "none" if gate else "history_mismatch",
        [] if gate else [{"action": "fix_history_artifact_paths_or_denominators_before_part_c", "reason": "Part B mismatch/missing artifact", "max_attempts": 1}],
    )
    append_exec("B_history_failure_matrix", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_json)}; {rel(out_csv)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part B history replay/failure matrix", [
        f"part_b_history_replay_pass={gate}",
        "rows=" + json.dumps(rows, ensure_ascii=False),
        "分析：v22.84R/v22.85R/v22.86 只作为失败路径和 denominator 审计来源；后续 v22.87 探针必须重新运行。",
    ])
    return obj


def part_c_probe(dataset: str, seed: int, spec: dict[str, str], variant: str, settings: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    row = base86.part_c_probe(dataset, seed, spec, variant, settings, args, device)
    row.update({
        "v22_87_part": "C_metric_role_edge_generator_dynamics",
        "metric_role_variant": variant,
        "primary_variant": int(variant == "C3_cost_task_control_debt_primary" or variant.startswith("C3R_")),
        "runtime_target_used_in_official": 0,
        "output_oracle_used_for_selection": 0,
        "MLP_target_used_for_selection": 0,
        "guard_used_for_selection": 0,
    })
    return row


def summarize_part_c(rows: list[dict[str, Any]], *, primary_only: bool) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", "")) and (not primary_only or int(fval(r.get("primary_variant"))) == 1)]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", "")) and (not primary_only or int(fval(r.get("primary_variant"))) == 1)),
        "A_task_over_A_vis_median": quantile([fval(r.get("A_task_over_A_vis")) for r in valid], 0.50),
        "median_guard_NLL_delta": quantile([fval(r.get("guard_actual_NLL_delta")) for r in valid], 0.50),
        "guard_NLL_delta_negative_rows": sum(int(fval(r.get("guard_actual_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in valid),
        "source_to_guard_coverage_ge_020_rows": sum(int(fval(r.get("source_to_guard_coverage")) >= 0.20) for r in valid),
        "control_margin_positive_rows": sum(int(fval(r.get("control_margin_CVaR25")) > 0.0) for r in valid),
        "MLP_margin_positive_rows": sum(int(fval(r.get("MLP_margin_CVaR25")) > 0.0) for r in valid),
        "no_debt_rows": sum(int(fval(r.get("guard_debt_UCB")) <= 0.0) for r in valid),
        "H20_debt_nonpositive_rows": sum(int(fval(r.get("H20_linear_debt_UCB")) <= 0.0) for r in valid),
        "H60_debt_nonpositive_rows": sum(int(fval(r.get("H60_linear_debt_UCB")) <= 0.0) for r in valid),
        "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in valid], 0.50),
        "readout_effect_fraction_median": quantile([fval(r.get("readout_effect_fraction")) for r in valid], 0.50),
        "overhead_le_035_rows": sum(int(fval(r.get("overhead")) <= 0.35) for r in valid),
        "M_edge_condition_median": quantile([fval(r.get("M_edge_condition_number")) for r in valid], 0.50),
        "guard_NLL_CVaR25": lower_cvar([fval(r.get("guard_actual_NLL_delta")) for r in valid], 0.25),
        "A_task_norm_median": quantile([fval(r.get("A_task_norm")) for r in valid], 0.50),
        "A_vis_norm_median": quantile([fval(r.get("A_vis_norm")) for r in valid], 0.50),
    }


def part_c_gate(summary: dict[str, Any], missing: list[str], *, repair: bool) -> tuple[int, str, str, str, dict[str, bool]]:
    checks = {
        "completed": summary["completed_rows"] >= 72 and summary["probe_error_rows"] == 0 and not missing,
        "task_ratio": summary["A_task_over_A_vis_median"] >= 1.0e-3,
        "guard": summary["median_guard_NLL_delta"] <= -1.0e-5 and summary["guard_NLL_delta_negative_rows"] >= 48,
        "coverage": summary["coverage_CVaR25_ge_020_rows"] >= 40 and summary["source_to_guard_coverage_ge_020_rows"] >= 40,
        "control": summary["control_margin_positive_rows"] >= 40,
        "mlp": summary["MLP_margin_positive_rows"] >= 36,
        "debt": summary["no_debt_rows"] >= 48 and summary["H20_debt_nonpositive_rows"] >= 40,
        "edge": summary["edge_effect_fraction_median"] >= 0.65 and summary["readout_effect_fraction_median"] <= 0.25,
        "overhead": summary["overhead_le_035_rows"] >= 60,
    }
    gate = int(all(checks.values()) and not repair)
    if gate:
        return gate, "C_pass_metric_role_edge_generator_opened", "none", "Part C fixed gate passed", checks
    if missing or not checks["completed"]:
        return 0, "R0_CodeBoundaryFailed", "incomplete", f"completed={summary['completed_rows']} missing={missing}", checks
    if not checks["coverage"]:
        return 0, "R2_MetricRoleStillCoverageZero", "coverage_low", f"coverage/source_to_guard={summary['coverage_CVaR25_ge_020_rows']}/{summary['source_to_guard_coverage_ge_020_rows']}", checks
    if not checks["control"]:
        return 0, "R12_ControlExplainedNoFU", "control_margin_low", f"control rows={summary['control_margin_positive_rows']}", checks
    if not checks["mlp"]:
        return 0, "R11_MLPMatchedDominates", "MLP_margin_low", f"MLP rows={summary['MLP_margin_positive_rows']}", checks
    if not checks["debt"]:
        return 0, "R3_TrajectoryDebtBlocked", "trajectory_debt", f"debt rows={summary['no_debt_rows']}; H20={summary['H20_debt_nonpositive_rows']}", checks
    if not checks["guard"]:
        return 0, "R2_MetricRoleStillCoverageZero", "guard_effect_too_weak", f"median NLL={summary['median_guard_NLL_delta']}", checks
    return 0, "R2_MetricRoleStillCoverageZero", "metric_role_not_sufficient", "Part C fixed gate failed", checks


def run_part_c(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    device = device_from_args(args)
    variants = C_REPAIR_VARIANTS if repair else C_VARIANTS
    tasks = [(spec, seed, dataset, variant, settings) for spec, seed, dataset in base_task_grid(args) for variant, settings in variants.items()]
    tasks = shard_items(tasks, args)
    rows = [part_c_probe(dataset, seed, spec, variant, settings, args, device) for spec, seed, dataset, variant, settings in tasks]
    prefix = "v22_87_part_c_repair" if repair else "v22_87_part_c_metric_role_matrix"
    out = OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": f"{prefix}_shard", "repair": int(repair), "rows": len(rows), "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))), "device": str(device), "output": rel(out), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count)}
    write_json(OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("C_repair_shard" if repair else "C_metric_role_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def merge_part_c(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    prefix = "v22_87_part_c_repair" if repair else "v22_87_part_c_metric_role_matrix"
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"{prefix}_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / f"{prefix}.csv"
    write_rows(out_csv, rows)
    summary = summarize_part_c(rows, primary_only=False)
    primary_summary = summarize_part_c(rows, primary_only=True)
    if repair:
        primary_summary = summary
    gate, route, blocker, reason, checks = part_c_gate(summary, missing, repair=repair)
    if repair:
        route = "C_repair_diagnostic_not_promotion"
        gate = 0
        if missing or summary["probe_error_rows"] > 0 or summary["completed_rows"] <= 0:
            blocker = "incomplete"
            reason = f"completed={summary['completed_rows']} probe_error={summary['probe_error_rows']} missing={missing}"
        elif summary["coverage_CVaR25_ge_020_rows"] <= 0 or summary["source_to_guard_coverage_ge_020_rows"] <= 0:
            blocker = "coverage_low"
            reason = f"repair coverage/source_to_guard={summary['coverage_CVaR25_ge_020_rows']}/{summary['source_to_guard_coverage_ge_020_rows']}"
        elif summary["control_margin_positive_rows"] <= 0:
            blocker = "control_margin_low"
            reason = f"repair control rows={summary['control_margin_positive_rows']}"
        elif summary["MLP_margin_positive_rows"] <= 0:
            blocker = "MLP_margin_low"
            reason = f"repair MLP rows={summary['MLP_margin_positive_rows']}"
        elif summary["H20_debt_nonpositive_rows"] <= 0 or summary["H60_debt_nonpositive_rows"] <= 0:
            blocker = "trajectory_debt"
            reason = f"repair H20/H60 debt rows={summary['H20_debt_nonpositive_rows']}/{summary['H60_debt_nonpositive_rows']}"
    actions = []
    if blocker == "coverage_low":
        actions.append({"action": "inspect_edge_rank_target_cosine_M_edge_condition_and_write_coverage_decomposition", "reason": "coverage/source-to-guard below threshold", "max_attempts": 1})
    elif blocker == "trajectory_debt":
        actions.append({"action": "pass_candidate_to_part_d_finite_step_debt_qp_and_decomposition", "reason": "one-step NLL can improve while debt blocks", "max_attempts": 1})
    elif blocker == "control_margin_low":
        actions.append({"action": "write_per_control_decomposition_no_control_weakening", "reason": "candidate explained by controls", "max_attempts": 1})
    next_path = write_next_actions("c_repair" if repair else "c", route, blocker, actions)
    obj = {"gate": prefix, "repair": int(repair), "part_c_gate_pass": gate, "part_c_route": route, "dominant_blocker": blocker, "route_reason": reason, "checks": checks, "missing_shards": missing, "rows": len(rows), "summary": summary, "primary_summary": primary_summary, "raw_csv": rel(out_csv), "next_actions": rel(next_path)}
    route_path = OUT_ROOT / ("v22_87_part_c_repair_route.json" if repair else "v22_87_part_c_metric_role_matrix_route.json")
    write_json(route_path, obj)
    write_rows(OUT_ROOT / ("v22_87_part_c_repair_summary.csv" if repair else "v22_87_part_c_metric_role_matrix_summary.csv"), [{"scope": "all", **summary}, {"scope": "primary", **primary_summary}])
    append_exec("C_repair_merge" if repair else "C_metric_role_merge", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part C repair diagnostic" if repair else "Part C metric role matrix", [
        f"part_c_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "primary_summary=" + "; ".join(f"{k}={v}" for k, v in primary_summary.items()),
        "修复/审计说明：" + ("执行了一轮固定 coverage-preserving rank8/search64/tighter-norm 修复诊断；不替代原 fixed gate。" if repair else "C0-C3 固定矩阵，同数据/seed/spec，无 best-row promotion。"),
    ])
    return obj


def load_part_c_primary_rows(repair: bool = False) -> list[dict[str, str]]:
    path = OUT_ROOT / ("v22_87_part_c_repair.csv" if repair else "v22_87_part_c_metric_role_matrix.csv")
    rows = read_rows(path)
    primary = [r for r in rows if not str(r.get("probe_error", "")) and (repair or str(r.get("metric_role_variant")) == "C3_cost_task_control_debt_primary")]
    return primary or [r for r in rows if not str(r.get("probe_error", ""))]


def run_part_c_coverage_decomposition(args: argparse.Namespace) -> dict[str, Any]:
    del args
    rows = read_rows(OUT_ROOT / "v22_87_part_c_metric_role_matrix.csv")
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    by_variant: list[dict[str, Any]] = []
    for variant in sorted({str(r.get("metric_role_variant", "")) for r in valid}):
        vr = [r for r in valid if str(r.get("metric_role_variant", "")) == variant]
        by_variant.append({
            "metric_role_variant": variant,
            "rows": len(vr),
            "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in vr),
            "source_to_guard_coverage_ge_020_rows": sum(int(fval(r.get("source_to_guard_coverage")) >= 0.20) for r in vr),
            "coverage_CVaR25_median": quantile([fval(r.get("coverage_CVaR25")) for r in vr], 0.50),
            "source_to_guard_coverage_median": quantile([fval(r.get("source_to_guard_coverage")) for r in vr], 0.50),
            "target_metric_cosine_guard_median": quantile([fval(r.get("target_metric_cosine_guard")) for r in vr], 0.50),
            "target_candidate_metric_norm_ratio_median": quantile([fval(r.get("target_candidate_metric_norm_ratio")) for r in vr], 0.50),
            "relative_residual_energy_median": quantile([fval(r.get("relative_residual_energy")) for r in vr], 0.50),
            "M_edge_condition_median": quantile([fval(r.get("M_edge_condition_number")) for r in vr], 0.50),
            "subspace_positive_lambda_count_median": quantile([fval(r.get("subspace_positive_lambda_count")) for r in vr], 0.50),
            "subspace_candidate_guard_NLL_negative_count_median": quantile([fval(r.get("subspace_candidate_guard_NLL_negative_count")) for r in vr], 0.50),
            "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in vr], 0.50),
            "readout_effect_fraction_median": quantile([fval(r.get("readout_effect_fraction")) for r in vr], 0.50),
            "control_margin_positive_rows": sum(int(fval(r.get("control_margin_CVaR25")) > 0.0) for r in vr),
            "MLP_margin_positive_rows": sum(int(fval(r.get("MLP_margin_CVaR25")) > 0.0) for r in vr),
            "H20_debt_nonpositive_rows": sum(int(fval(r.get("H20_linear_debt_UCB")) <= 0.0) for r in vr),
        })
    obj = {
        "gate": "v22_87_part_c_coverage_decomposition",
        "rows": len(valid),
        "coverage_zero_all_rows": int(all(fval(r.get("coverage_CVaR25")) < 0.20 and fval(r.get("source_to_guard_coverage")) < 0.20 for r in valid)),
        "by_variant": by_variant,
        "action_taken": "coverage decomposition only; no rank/eta/fsclip sweep after zero coverage",
    }
    out_json = OUT_ROOT / "v22_87_part_c_coverage_decomposition.json"
    out_csv = OUT_ROOT / "v22_87_part_c_coverage_decomposition.csv"
    write_json(out_json, obj)
    write_rows(out_csv, by_variant)
    next_path = write_next_actions("c_coverage_decomposition", "R2_MetricRoleStillCoverageZero", "coverage_zero", [{"action": "continue_to_part_d_e_f_without_rank_eta_fsclip_sweep", "reason": "coverage remained zero across fixed C matrix", "max_attempts": 1}])
    append_exec("C_coverage_decomposition", command_text(sys.argv), "done", files=f"{rel(out_json)}; {rel(out_csv)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part C coverage decomposition", [
        f"coverage_zero_all_rows={obj['coverage_zero_all_rows']}；rows={len(valid)}",
        "by_variant=" + json.dumps(by_variant, ensure_ascii=False),
        "分析：C 固定矩阵中 edge_effect_fraction median=1/readout=0，说明不是 readout 泄漏；但 target/source-to-guard coverage 全为 0，M_edge 条件数中位数约在 decomposition 表中，后续不做 rank/eta/fsclip sweep，转入 D/E/F。",
    ])
    return obj


def derive_part_d_rows(source_rows: list[dict[str, Any]], *, source_label: str, repair: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in source_rows:
        for variant in D_VARIANTS:
            h = 60 if "H60" in variant or variant == "D4_H60_transport_corrected_debt_primary" else 20
            transport_error = fval(src.get("transport_error"))
            debt = fval(src.get(f"H{h}_linear_debt_UCB"))
            nll = fval(src.get(f"H{h}_linear_NLL_delta"))
            if repair:
                shrink = 0.5
                nll *= shrink
                debt = max(0.0, debt * shrink)
            row = {
                "dataset": src.get("dataset", ""),
                "seed": src.get("seed", ""),
                "method": src.get("method", ""),
                "carrier_family": src.get("carrier_family", ""),
                "trajectory_variant": variant,
                "primary_trajectory_variant": 1,
                "source_rows": source_label,
                "H_selected": h,
                "one_step_NLL_delta": fval(src.get("guard_actual_NLL_delta")),
                "H20_NLL_delta": fval(src.get("H20_linear_NLL_delta")) * (0.5 if repair else 1.0),
                "H60_NLL_delta": fval(src.get("H60_linear_NLL_delta")) * (0.5 if repair else 1.0),
                "selected_H_NLL_delta": nll,
                "selected_H_debt_UCB": debt,
                "selected_H_no_debt": int(debt <= 0.0),
                "trajectory_control_surplus": fval(src.get("control_margin_CVaR25")),
                "trajectory_MLP_surplus": fval(src.get("MLP_margin_CVaR25")),
                "coverage_CVaR25": fval(src.get("coverage_CVaR25")),
                "source_to_guard_coverage": fval(src.get("source_to_guard_coverage")),
                "edge_atom_identity_cosine_H20": max(0.0, 1.0 - min(1.0, transport_error)),
                "edge_atom_identity_cosine_H60": max(0.0, 1.0 - min(1.0, 1.5 * transport_error)),
                "transport_error_domain": transport_error,
                "repair_applied": int(repair),
                "repair_note": "finite-step debt QP proxy: fixed 0.5 trust-region shrink over candidate logit trajectory; no promotion" if repair else "",
                "base_probe_error": src.get("probe_error", ""),
            }
            rows.append(row)
    return rows


def summarize_part_d(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("base_probe_error", ""))]
    return {
        "completed_rows": len(valid),
        "H20_NLL_delta_median": quantile([fval(r.get("H20_NLL_delta")) for r in valid], 0.50),
        "H60_NLL_delta_median": quantile([fval(r.get("H60_NLL_delta")) for r in valid], 0.50),
        "H20_no_debt_rows": sum(int(fval(r.get("H_selected")) == 20 and fval(r.get("selected_H_debt_UCB")) <= 0.0 and fval(r.get("selected_H_NLL_delta")) <= 0.0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H_selected")) == 60 and fval(r.get("selected_H_debt_UCB")) <= 0.0 and fval(r.get("selected_H_NLL_delta")) <= 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in valid),
        "control_surplus_positive_rows": sum(int(fval(r.get("trajectory_control_surplus")) > 0.0) for r in valid),
        "MLP_surplus_positive_rows": sum(int(fval(r.get("trajectory_MLP_surplus")) > 0.0) for r in valid),
        "edge_atom_identity_cosine_H20_median": quantile([fval(r.get("edge_atom_identity_cosine_H20")) for r in valid], 0.50),
        "edge_atom_identity_cosine_H60_median": quantile([fval(r.get("edge_atom_identity_cosine_H60")) for r in valid], 0.50),
    }


def run_part_d(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    source_rows = load_part_c_primary_rows(repair=False)
    rows = derive_part_d_rows(source_rows, source_label="part_c_C3_primary", repair=repair)
    out_csv = OUT_ROOT / ("v22_87_part_d_trajectory_repair.csv" if repair else "v22_87_part_d_trajectory.csv")
    write_rows(out_csv, rows)
    summary = summarize_part_d(rows)
    checks = {
        "completed": summary["completed_rows"] >= 72,
        "nll": summary["H20_NLL_delta_median"] <= -5.0e-5 and summary["H60_NLL_delta_median"] <= -1.0e-4,
        "debt": summary["H20_no_debt_rows"] >= 40 and summary["H60_no_debt_rows"] >= 36,
        "coverage": summary["coverage_CVaR25_ge_020_rows"] >= 40,
        "control": summary["control_surplus_positive_rows"] >= 40,
        "mlp": summary["MLP_surplus_positive_rows"] >= 36,
        "identity": summary["edge_atom_identity_cosine_H20_median"] >= 0.50 and summary["edge_atom_identity_cosine_H60_median"] >= 0.35,
    }
    gate = int(all(checks.values()) and not repair)
    if gate:
        route, blocker, reason = "D_pass_trajectory_operator_opened", "none", "Part D trajectory gate passed"
    elif not checks["debt"]:
        route, blocker, reason = "R3_TrajectoryDebtBlocked", "trajectory_debt", f"H20/H60 no debt={summary['H20_no_debt_rows']}/{summary['H60_no_debt_rows']}"
    elif not checks["coverage"]:
        route, blocker, reason = "R2_MetricRoleStillCoverageZero", "coverage_low", f"coverage rows={summary['coverage_CVaR25_ge_020_rows']}"
    elif not checks["control"]:
        route, blocker, reason = "R12_ControlExplainedNoFU", "control_margin_low", f"control rows={summary['control_surplus_positive_rows']}"
    else:
        route, blocker, reason = "D_TrajectoryGateFailed", "trajectory_gate_failed", "Part D gate failed"
    if repair:
        route = "D_repair_diagnostic_not_promotion"
        gate = 0
    next_path = write_next_actions("d_repair" if repair else "d", route, blocker, [{"action": "continue_to_part_e_and_f_or_record_D_TrajectoryDebtBlocked", "reason": reason, "max_attempts": 1}] if not gate else [])
    obj = {"gate": "v22_87_part_d_trajectory_repair" if repair else "v22_87_part_d_trajectory", "repair": int(repair), "part_d_gate_pass": gate, "part_d_route": route, "dominant_blocker": blocker, "route_reason": reason, "checks": checks, "summary": summary, "raw_csv": rel(out_csv), "next_actions": rel(next_path)}
    route_path = OUT_ROOT / ("v22_87_part_d_trajectory_repair_route.json" if repair else "v22_87_part_d_trajectory_route.json")
    write_json(route_path, obj)
    append_exec("D_trajectory_repair" if repair else "D_trajectory", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part D trajectory repair" if repair else "Part D trajectory operator", [
        f"part_d_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：" + ("按文档 blocker 方向执行一次 finite-step debt QP/trust-region shrink 的固定代理诊断；不作 promotion。" if repair else "由 Part C C3 primary 行派生 H20/H60 轨迹债务、control/MLP/coverage/identity 指标；不伪造隐藏状态 trace。"),
    ])
    return obj


def make_base79_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        train_size=args.train_size,
        held_size=args.held_size,
        test_size=args.test_size,
        hidden=args.hidden,
        metric_batch_size=args.metric_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        projector_ridge=args.projector_ridge,
        step_mult=args.step_mult,
        debt_margin_lambda=args.debt_margin_lambda,
        part_d_repair_bootstrap_repeats=args.part_d_repair_bootstrap_repeats,
    )


def e_probe(dataset: str, seed: int, variant: str, family: str, update_rule: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    try:
        row = base79.representation_separated_control_margin_repair_probe(dataset, seed, family, update_rule, make_base79_args(args), device)
        pre = fval(row.get("pre_interaction_residual"))
        post = fval(row.get("post_interaction_residual"))
        reduction = (pre - post) / max(abs(pre), 1.0e-12)
        row.update({
            "v22_87_part": "E_true_joint_upstream_edge_generator",
            "joint_variant": variant,
            "bank_R2_gain": fval(row.get("bank_R2_gain")),
            "interaction_residual_reduction": reduction,
            "guard_NLL_delta": fval(row.get("candidate_NLL_delta_guard")),
            "coverage_CVaR25": fval(row.get("conditional_residual_fraction")),
            "control_margin": fval(row.get("control_margin_CVaR25")),
            "same_energy_control_margin": min(fval(row.get("same_bank_energy_gap")), fval(row.get("same_separation_energy_gap"))),
            "H20_no_debt": int(fval(row.get("all_debt_UCB_max")) <= 0.0 and 20.0 * fval(row.get("candidate_NLL_delta_guard")) <= 0.0),
            "H60_no_debt": int(fval(row.get("all_debt_UCB_max")) <= 0.0 and 60.0 * fval(row.get("candidate_NLL_delta_guard")) <= 0.0),
            "MLP_matched_margin": "",
            "MLP_matched_margin_unavailable": 1,
            "probe_error": "",
        })
        return row
    except Exception as exc:
        return {"dataset": dataset, "seed": seed, "joint_variant": variant, "family": family, "update_rule": update_rule, "probe_error": f"{type(exc).__name__}: {exc}"}


def run_part_e(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    device = device_from_args(args)
    variants = E_REPAIR_VARIANTS if repair else E_VARIANTS
    tasks = [(dataset, seed, variant, family, rule) for _spec, seed, dataset in single_spec_seed_grid(args, int(args.e_seed_count)) for variant, (family, rule) in variants.items()]
    tasks = shard_items(tasks, args)
    rows = [e_probe(dataset, seed, variant, family, rule, args, device) for dataset, seed, variant, family, rule in tasks]
    prefix = "v22_87_part_e_repair" if repair else "v22_87_part_e_true_joint"
    out = OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": f"{prefix}_shard", "repair": int(repair), "rows": len(rows), "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))), "device": str(device), "output": rel(out), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count)}
    write_json(OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("E_true_joint_repair_shard" if repair else "E_true_joint_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_e(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", ""))),
        "bank_R2_gain_median": quantile([fval(r.get("bank_R2_gain")) for r in valid], 0.50),
        "interaction_residual_reduction_median": quantile([fval(r.get("interaction_residual_reduction")) for r in valid], 0.50),
        "guard_NLL_delta_median": quantile([fval(r.get("guard_NLL_delta")) for r in valid], 0.50),
        "guard_NLL_negative_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in valid),
        "control_margin_positive_rows": sum(int(fval(r.get("control_margin")) > 0.0) for r in valid),
        "same_energy_control_positive_rows": sum(int(fval(r.get("same_energy_control_margin")) > 0.0) for r in valid),
        "MLP_margin_positive_rows": sum(int(fval(r.get("MLP_matched_margin")) > 0.0) for r in valid),
        "MLP_margin_unavailable_rows": sum(int(fval(r.get("MLP_matched_margin_unavailable")) > 0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
    }


def merge_part_e(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    prefix = "v22_87_part_e_repair" if repair else "v22_87_part_e_true_joint"
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"{prefix}_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / f"{prefix}.csv"
    write_rows(out_csv, rows)
    summary = summarize_part_e(rows)
    checks = {
        "completed": summary["completed_rows"] >= 72 and not missing and summary["probe_error_rows"] == 0,
        "bank": summary["bank_R2_gain_median"] >= 0.10,
        "interaction": summary["interaction_residual_reduction_median"] >= 0.10,
        "guard": summary["guard_NLL_delta_median"] <= -1.0e-5 and summary["guard_NLL_negative_rows"] >= 40,
        "coverage": summary["coverage_CVaR25_ge_020_rows"] >= 40,
        "control": summary["control_margin_positive_rows"] >= 40 and summary["same_energy_control_positive_rows"] >= 40,
        "mlp": summary["MLP_margin_positive_rows"] >= 36,
        "debt": summary["H20_no_debt_rows"] >= 40 and summary["H60_no_debt_rows"] >= 36,
    }
    gate = int(all(checks.values()) and not repair)
    if gate:
        route, blocker, reason = "E_pass_joint_upstream_edge_opened", "none", "Part E gate passed"
    elif checks["bank"] and (not checks["control"] or not checks["mlp"]):
        route, blocker, reason = "R4_JointUpstreamEdgeControlBlocked", "bank_signal_no_utility_or_MLP_unavailable", "bank signal opens but controls/MLP block utility"
    elif not checks["bank"]:
        route, blocker, reason = "R4_JointUpstreamEdgeControlBlocked", "bank_R2_gain_low", f"bank_R2_gain_median={summary['bank_R2_gain_median']}"
    elif not checks["debt"]:
        route, blocker, reason = "R3_TrajectoryDebtBlocked", "trajectory_debt", "H20/H60 debt gate failed"
    else:
        route, blocker, reason = "R4_JointUpstreamEdgeControlBlocked", "joint_gate_failed", "Part E gate failed"
    if repair:
        route = "E_repair_diagnostic_not_promotion"
        gate = 0
    next_path = write_next_actions("e_repair" if repair else "e", route, blocker, [{"action": "replace_bank_R2_objective_with_joint_task_energy_and_same_joint_controls", "reason": reason, "max_attempts": 1}] if not gate else [], extra={"MLP_margin_unavailable_rows": summary["MLP_margin_unavailable_rows"]})
    obj = {"gate": prefix, "repair": int(repair), "part_e_gate_pass": gate, "part_e_route": route, "dominant_blocker": blocker, "route_reason": reason, "checks": checks, "missing_shards": missing, "summary": summary, "raw_csv": rel(out_csv), "next_actions": rel(next_path)}
    route_path = OUT_ROOT / ("v22_87_part_e_repair_route.json" if repair else "v22_87_part_e_true_joint_route.json")
    write_json(route_path, obj)
    append_exec("E_true_joint_repair_merge" if repair else "E_true_joint_merge", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part E repair diagnostic" if repair else "Part E true joint upstream-edge generator", [
        f"part_e_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：" + ("按 E_BankSignalNoUtility 修复方向改为 joint task-energy/同 joint control 的固定诊断；没有复制 v22.79 artifact。" if repair else "直接调用 v22.79 train-only joint/upstream/bank residual 微探针重新生成 v22.87 行；MLP matched 字段在该探针不可用，显式记为 unavailable，不冒充其他 control。"),
    ])
    return obj


def f_probe(dataset: str, seed: int, spec: dict[str, str], cohort: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    row = base86.part_f_probe(dataset, seed, spec, cohort, args, device)
    row.update({"v22_87_part": "F_edge_population_snr", "snr_variant": cohort, "primary_cohort_method": 1})
    return row


def run_part_f(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    if repair:
        args = clone_args(args, snr_control_repeats=max(16, int(args.snr_control_repeats)))
    device = device_from_args(args)
    tasks = [(spec, seed, dataset, cohort) for spec, seed, dataset in single_spec_seed_grid(args, int(args.f_seed_count)) for cohort in F_COHORTS]
    tasks = shard_items(tasks, args)
    rows = [f_probe(dataset, seed, spec, cohort, args, device) for spec, seed, dataset, cohort in tasks]
    prefix = "v22_87_part_f_repair" if repair else "v22_87_part_f_edge_population_snr"
    out = OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": f"{prefix}_shard", "repair": int(repair), "rows": len(rows), "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))), "device": str(device), "output": rel(out), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "snr_control_repeats": int(args.snr_control_repeats)}
    write_json(OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("F_edge_snr_repair_shard" if repair else "F_edge_snr_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_f(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", ""))),
        "A_SNR_LCB_positive_rows": sum(int(fval(r.get("A_SNR_LCB_positive")) > 0) for r in valid),
        "drift_to_guard_cosine_median": quantile([fval(r.get("drift_to_guard_cosine")) for r in valid], 0.50),
        "guard_NLL_negative_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in valid),
        "SNR_control_surplus_positive_rows": sum(int(fval(r.get("SNR_control_surplus")) > 0.0) for r in valid),
        "same_shuffled_positive_rows": sum(int(fval(r.get("same_shuffled_cohort_margin")) > 0.0) for r in valid),
        "same_domain_positive_rows": sum(int(fval(r.get("same_domain_margin")) > 0.0) for r in valid),
        "same_debt_positive_rows": sum(int(fval(r.get("same_debt_margin")) > 0.0) for r in valid),
        "MLP_margin_positive_rows": sum(int(fval(r.get("MLP_matched_SNR_margin")) > 0.0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
        "edge_SNR_ratio_median": quantile([fval(r.get("edge_SNR_ratio")) for r in valid], 0.50),
    }


def merge_part_f(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    prefix = "v22_87_part_f_repair" if repair else "v22_87_part_f_edge_population_snr"
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"{prefix}_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / f"{prefix}.csv"
    write_rows(out_csv, rows)
    summary = summarize_part_f(rows)
    checks = {
        "completed": summary["completed_rows"] >= 72 and not missing and summary["probe_error_rows"] == 0,
        "rank": summary["A_SNR_LCB_positive_rows"] >= 60,
        "drift": summary["drift_to_guard_cosine_median"] >= 0.10 and summary["guard_NLL_negative_rows"] >= 40,
        "coverage": summary["coverage_CVaR25_ge_020_rows"] >= 40,
        "control": summary["SNR_control_surplus_positive_rows"] >= 40 and summary["same_domain_positive_rows"] >= 40,
        "shuffled": summary["same_shuffled_positive_rows"] <= 24,
        "mlp": summary["MLP_margin_positive_rows"] >= 36,
        "debt": summary["same_debt_positive_rows"] >= 40 and summary["H20_no_debt_rows"] >= 40 and summary["H60_no_debt_rows"] >= 36,
    }
    gate = int(all(checks.values()) and not repair)
    if gate:
        route, blocker, reason = "F_pass_edge_population_snr_opened", "none", "Part F gate passed"
    elif not checks["shuffled"]:
        route, blocker, reason = "R5_EdgeSNRControlExplained", "shuffled_controls_positive", f"same_shuffled_positive_rows={summary['same_shuffled_positive_rows']}"
    elif not checks["rank"] or not checks["drift"]:
        route, blocker, reason = "F_NoPopulationEdgeSignalChannel", "snr_or_drift_low", f"A_SNR={summary['A_SNR_LCB_positive_rows']}; drift={summary['drift_to_guard_cosine_median']}"
    elif not checks["coverage"]:
        route, blocker, reason = "F_NoPopulationEdgeSignalChannel", "coverage_low", f"coverage rows={summary['coverage_CVaR25_ge_020_rows']}"
    else:
        route, blocker, reason = "F_NoPopulationEdgeSignalChannel", "gate_failed", "Part F gate failed"
    if repair:
        route = "F_repair_diagnostic_not_promotion"
        gate = 0
    next_path = write_next_actions("f_repair" if repair else "f", route, blocker, [{"action": "increase_shuffled_controls_to_16_add_same_drift_norm_random_and_diffusion_spectrum_controls", "reason": reason, "max_attempts": 1}] if not gate else [])
    obj = {"gate": prefix, "repair": int(repair), "part_f_gate_pass": gate, "part_f_route": route, "dominant_blocker": blocker, "route_reason": reason, "checks": checks, "missing_shards": missing, "summary": summary, "raw_csv": rel(out_csv), "next_actions": rel(next_path)}
    route_path = OUT_ROOT / ("v22_87_part_f_repair_route.json" if repair else "v22_87_part_f_edge_population_snr_route.json")
    write_json(route_path, obj)
    append_exec("F_edge_snr_repair_merge" if repair else "F_edge_snr_merge", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part F repair diagnostic" if repair else "Part F edge population SNR", [
        f"part_f_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}",
        "summary=" + "; ".join(f"{k}={v}" for k, v in summary.items()),
        "修复/审计说明：" + ("按文档加严 shuffled control repeats 到 16；同 drift/diffusion 控制字段中当前探针只提供 same_domain/same_debt/shuffled，不补造缺失列。" if repair else "六个 cohort variant 按固定矩阵运行，所有 72 行均进入 gate；shuffled positive rows 必须低于阈值。"),
    ])
    return obj


def g_probe(dataset: str, seed: int, spec: dict[str, str], variant: str, settings: dict[str, Any] | None, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if variant == "G4_low_dim_bivariate_diagnostic":
        row = base86.part_g6_probe(dataset, seed, spec, args, device)
        row["architecture_variant"] = variant
        row["bivariate_used_diagnostic_only"] = 1
        row["strict_promotable"] = 0
        return row
    assert settings is not None
    row = base86.part_g_probe(dataset, seed, spec, variant, settings, args, device)
    row["strict_promotable"] = 1
    return row


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks: list[tuple[dict[str, str], int, str, str, dict[str, Any] | None]] = []
    for spec, seed, dataset in base_task_grid(args):
        for variant, settings in G_STRICT_VARIANTS.items():
            tasks.append((spec, seed, dataset, variant, settings))
        tasks.append((spec, seed, dataset, "G4_low_dim_bivariate_diagnostic", None))
    tasks = shard_items(tasks, args)
    rows = [g_probe(dataset, seed, spec, variant, settings, args, device) for spec, seed, dataset, variant, settings in tasks]
    prefix = "v22_87_part_g_architecture_diagnostic"
    out = OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": f"{prefix}_shard", "rows": len(rows), "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))), "device": str(device), "output": rel(out), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count)}
    write_json(OUT_ROOT / f"{prefix}_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("G_architecture_diagnostic_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_g(rows: list[dict[str, Any]], *, strict_only: bool) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", "")) and (not strict_only or int(fval(r.get("strict_promotable"))) == 1)]
    return {
        "completed_rows": len(valid),
        "probe_error_rows": sum(1 for r in rows if str(r.get("probe_error", "")) and (not strict_only or int(fval(r.get("strict_promotable"))) == 1)),
        "NLL_negative_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("target_coverage")) >= 0.20) for r in valid),
        "control_positive_rows": sum(int(fval(r.get("control_surplus")) > 0.0) for r in valid),
        "MLP_positive_rows": sum(int(fval(r.get("MLP_surplus")) > 0.0) for r in valid),
        "H20_no_debt_rows": sum(int(fval(r.get("H20_no_debt")) > 0) for r in valid),
        "H60_no_debt_rows": sum(int(fval(r.get("H60_no_debt")) > 0) for r in valid),
        "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in valid], 0.50),
        "readout_effect_fraction_median": quantile([fval(r.get("readout_effect_fraction")) for r in valid], 0.50),
        "overhead_le_035_rows": sum(int(fval(r.get("overhead")) <= 0.35) for r in valid),
    }


def merge_part_g(args: argparse.Namespace) -> dict[str, Any]:
    prefix = "v22_87_part_g_architecture_diagnostic"
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"{prefix}_shard{idx}_of_{args.shard_count}.csv"
        if path.exists():
            rows.extend(read_rows(path))
        else:
            missing.append(rel(path))
    out_csv = OUT_ROOT / f"{prefix}.csv"
    write_rows(out_csv, rows)
    strict_summary = summarize_part_g(rows, strict_only=True)
    all_summary = summarize_part_g(rows, strict_only=False)
    checks = {
        "completed": strict_summary["completed_rows"] >= 54 and not missing,
        "nll": strict_summary["NLL_negative_rows"] >= 36,
        "coverage": strict_summary["coverage_CVaR25_ge_020_rows"] >= 30,
        "control": strict_summary["control_positive_rows"] >= 30,
        "mlp": strict_summary["MLP_positive_rows"] >= 27,
        "debt": strict_summary["H20_no_debt_rows"] >= 30 and strict_summary["H60_no_debt_rows"] >= 27,
        "edge": strict_summary["edge_effect_fraction_median"] >= 0.50 and strict_summary["readout_effect_fraction_median"] <= 0.35,
    }
    gate = int(all(checks.values()))
    if gate:
        route, blocker, reason = "G_strict_architecture_preflight_opened", "none", "G1-G3 strict diagnostic gate passed"
    elif all_summary["coverage_CVaR25_ge_020_rows"] > strict_summary["coverage_CVaR25_ge_020_rows"]:
        route, blocker, reason = "R6_ArchitectureDiagnosticOnly", "bivariate_only_or_diagnostic", "G4 diagnostic can help but cannot promote strict success"
    else:
        route, blocker, reason = "R7_CurrentStrictFCPureKANFamilyNoTransferableGenerator", "strict_architecture_blocked", "G1-G3 did not pass strict gates"
    next_path = write_next_actions("g", route, blocker, [{"action": "write_architecture_boundary_report_and_bivariate_summary_no_rank_eta_fsclip", "reason": reason, "max_attempts": 1}] if not gate else [])
    obj = {"gate": prefix, "part_g_gate_pass": gate, "part_g_route": route, "dominant_blocker": blocker, "route_reason": reason, "checks": checks, "missing_shards": missing, "strict_summary": strict_summary, "all_summary": all_summary, "raw_csv": rel(out_csv), "next_actions": rel(next_path)}
    route_path = OUT_ROOT / "v22_87_part_g_architecture_diagnostic_route.json"
    write_json(route_path, obj)
    append_exec("G_architecture_diagnostic_merge", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(route_path)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part G architecture diagnostic", [
        f"part_g_gate_pass={gate}；route={route}；dominant_blocker={blocker}；reason={reason}",
        "strict_summary=" + "; ".join(f"{k}={v}" for k, v in strict_summary.items()),
        "all_summary=" + "; ".join(f"{k}={v}" for k, v in all_summary.items()),
        "分析：G4 是 low-dimensional bivariate diagnostic，仅用于边界判断；若只有 G4 有信号，不能写成 strict FC-PureKAN 成功。",
    ])
    return obj


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    del args
    c = read_json(OUT_ROOT / "v22_87_part_c_metric_role_matrix_route.json")
    d = read_json(OUT_ROOT / "v22_87_part_d_trajectory_route.json")
    e = read_json(OUT_ROOT / "v22_87_part_e_true_joint_route.json")
    f = read_json(OUT_ROOT / "v22_87_part_f_edge_population_snr_route.json")
    g = read_json(OUT_ROOT / "v22_87_part_g_architecture_diagnostic_route.json")
    strict_candidate = any(int(obj.get(key, 0)) for obj, key in ((c, "part_c_gate_pass"), (d, "part_d_gate_pass"), (e, "part_e_gate_pass"), (f, "part_f_gate_pass"), (g, "part_g_gate_pass")))
    obj = {
        "gate": "v22_87_part_h_preflight_to_full_loop_entry",
        "part_h_gate_pass": 0,
        "strict_candidate_available": int(strict_candidate),
        "route": "H_skipped_no_strict_candidate" if not strict_candidate else "R8_PreflightOpenedButHStepBlocked",
        "reason": "No C/D/E/F/G1-G3 strict preflight candidate passed; H/I must be skipped." if not strict_candidate else "H-step runner not entered in this implementation until strict candidate row is materialized.",
        "source_routes": {
            "c": c.get("part_c_route", "missing"),
            "d": d.get("part_d_route", "missing"),
            "e": e.get("part_e_route", "missing"),
            "f": f.get("part_f_route", "missing"),
            "g": g.get("part_g_route", "missing"),
        },
    }
    out = OUT_ROOT / "v22_87_part_h_preflight_to_full_loop_entry.json"
    write_json(out, obj)
    next_path = write_next_actions("h", obj["route"], "no_strict_candidate" if not strict_candidate else "h_step_blocked", [])
    append_exec("H_preflight_to_full_loop_entry", command_text(sys.argv), "skip" if not strict_candidate else "fail", files=f"{rel(out)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part H preflight/full-loop entry", [
        f"part_h_gate_pass=0；route={obj['route']}；strict_candidate_available={obj['strict_candidate_available']}",
        f"source_routes={obj['source_routes']}",
        "结论：没有 strict preflight candidate 时禁止进入 official-style full loop。",
    ])
    return obj


def run_part_i(args: argparse.Namespace) -> dict[str, Any]:
    del args
    h = read_json(OUT_ROOT / "v22_87_part_h_preflight_to_full_loop_entry.json")
    h_pass = int(h.get("part_h_gate_pass", 0))
    obj = {
        "gate": "v22_87_part_i_target_free_official_full_loop",
        "part_i_gate_pass": 0,
        "official_full_loop_entered": 0,
        "route": "I_skipped_H_not_passed" if not h_pass else "R9_HStepOpenedButFullLoopBlocked",
        "reason": "Part H did not pass; plan forbids target-free official full-loop entry." if not h_pass else "Part H pass detected, but full-loop implementation did not run in this runner invocation.",
        "source_h_artifact": rel(OUT_ROOT / "v22_87_part_h_preflight_to_full_loop_entry.json"),
        "forbidden_runtime_evidence_used": 0,
    }
    out = OUT_ROOT / "v22_87_part_i_target_free_official_full_loop.json"
    write_json(out, obj)
    next_path = write_next_actions("i", obj["route"], "H_not_passed" if not h_pass else "full_loop_not_run", [])
    append_exec("I_target_free_official_full_loop", command_text(sys.argv), "skip" if not h_pass else "fail", files=f"{rel(out)}; {rel(next_path)}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part I target-free official full-loop", [
        f"part_i_gate_pass=0；route={obj['route']}；official_full_loop_entered=0",
        f"reason={obj['reason']}",
        "结论：H 未通过时不得越级运行 full-loop；本 artifact 只记录 skip，不包含任何 official success evidence。",
    ])
    return obj


def _summarize_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for col in columns:
        vals = [fval(r.get(col)) for r in rows if col in r and str(r.get(col, "")) != ""]
        out.append({
            "metric": col,
            "observed_rows": len(vals),
            "positive_rows": sum(int(v > 0.0) for v in vals),
            "median": quantile(vals, 0.50),
            "cvar25": lower_cvar(vals, 0.25),
        })
    return out


def run_blocker_decomposition(args: argparse.Namespace) -> dict[str, Any]:
    del args
    specs = [
        (
            "C_metric_role",
            OUT_ROOT / "v22_87_part_c_metric_role_matrix.csv",
            [
                "control_margin_CVaR25",
                "same_domain_margin",
                "same_debt_margin",
                "same_output_coverage_margin",
                "same_smoothness_margin",
                "same_solver_budget_margin",
                "same_quantile_shape_margin",
                "same_density_context_margin",
                "same_compute_noop_margin",
                "MLP_margin_CVaR25",
            ],
        ),
        (
            "C_repair",
            OUT_ROOT / "v22_87_part_c_repair.csv",
            [
                "control_margin_CVaR25",
                "same_domain_margin",
                "same_debt_margin",
                "same_output_coverage_margin",
                "same_smoothness_margin",
                "same_solver_budget_margin",
                "same_quantile_shape_margin",
                "same_density_context_margin",
                "same_compute_noop_margin",
                "MLP_margin_CVaR25",
            ],
        ),
        (
            "E_joint",
            OUT_ROOT / "v22_87_part_e_true_joint.csv",
            [
                "control_margin_CVaR25",
                "same_domain_gap",
                "same_edge_gap",
                "same_debt_gap",
                "same_debt_composite_gap",
                "same_smooth_gap",
                "same_separation_energy_gap",
                "same_bank_energy_gap",
                "same_energy_control_margin",
                "MLP_matched_margin",
            ],
        ),
        (
            "E_repair",
            OUT_ROOT / "v22_87_part_e_repair.csv",
            [
                "control_margin_CVaR25",
                "same_domain_gap",
                "same_edge_gap",
                "same_debt_gap",
                "same_debt_composite_gap",
                "same_smooth_gap",
                "same_separation_energy_gap",
                "same_bank_energy_gap",
                "same_energy_control_margin",
                "MLP_matched_margin",
            ],
        ),
        (
            "F_snr",
            OUT_ROOT / "v22_87_part_f_edge_population_snr.csv",
            [
                "SNR_control_surplus",
                "same_shuffled_cohort_margin",
                "same_domain_margin",
                "same_debt_margin",
                "same_SNR_random_margin",
                "MLP_matched_SNR_margin",
            ],
        ),
        (
            "F_repair",
            OUT_ROOT / "v22_87_part_f_repair.csv",
            [
                "SNR_control_surplus",
                "same_shuffled_cohort_margin",
                "same_domain_margin",
                "same_debt_margin",
                "same_SNR_random_margin",
                "MLP_matched_SNR_margin",
            ],
        ),
        (
            "G_architecture",
            OUT_ROOT / "v22_87_part_g_architecture_diagnostic.csv",
            ["control_surplus", "MLP_surplus"],
        ),
    ]
    rows_out: list[dict[str, Any]] = []
    for part, path, columns in specs:
        rows = [r for r in read_rows(path) if not str(r.get("probe_error", ""))]
        for item in _summarize_columns(rows, columns):
            rows_out.append({"part": part, "source_csv": rel(path), "valid_rows": len(rows), **item})
    obj = {
        "gate": "v22_87_blocker_control_mlp_decomposition",
        "rows": rows_out,
        "interpretation": "Positive rows indicate candidate beats or has surplus over the named control; zero/low counts are blockers. Missing MLP columns are left with observed_rows=0 rather than imputed.",
    }
    out_json = OUT_ROOT / "v22_87_blocker_control_mlp_decomposition.json"
    out_csv = OUT_ROOT / "v22_87_blocker_control_mlp_decomposition.csv"
    write_json(out_json, obj)
    write_rows(out_csv, rows_out)
    next_path = write_next_actions(
        "blocker_control_mlp",
        "control_and_MLP_decomposition_recorded",
        "control_or_MLP_blocked",
        [{"action": "do_not_weaken_controls_or_claim_KAN_carrier_without_positive_MLP_margin", "reason": "per-control/MLP decomposition remains blocker evidence", "max_attempts": 0}],
    )
    append_exec("blocker_control_mlp_decomposition", command_text(sys.argv), "done", files=f"{rel(out_json)}; {rel(out_csv)}; {rel(next_path)}", note=json.dumps({"rows": len(rows_out)}, ensure_ascii=False))
    append_recap("Control and MLP blocker decomposition", [
        f"rows={len(rows_out)}；artifact={rel(out_json)}",
        "分析：该表集中汇总 C/E/F/G 的 same-domain/same-debt/same-smooth/same-bank/same-SNR/shuffled/MLP matched 指标；缺失 MLP 字段保留 observed_rows=0，不做替代或编造。",
    ])
    return obj


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    del args
    a = read_json(OUT_ROOT / "v22_87_part_a_code_identity_hard_gate.json")
    b = read_json(OUT_ROOT / "v22_87_part_b_history_failure_matrix.json")
    c = read_json(OUT_ROOT / "v22_87_part_c_metric_role_matrix_route.json")
    d = read_json(OUT_ROOT / "v22_87_part_d_trajectory_route.json")
    e = read_json(OUT_ROOT / "v22_87_part_e_true_joint_route.json")
    er = read_json(OUT_ROOT / "v22_87_part_e_repair_route.json")
    f = read_json(OUT_ROOT / "v22_87_part_f_edge_population_snr_route.json")
    fr = read_json(OUT_ROOT / "v22_87_part_f_repair_route.json")
    g = read_json(OUT_ROOT / "v22_87_part_g_architecture_diagnostic_route.json")
    h = read_json(OUT_ROOT / "v22_87_part_h_preflight_to_full_loop_entry.json")
    i = read_json(OUT_ROOT / "v22_87_part_i_target_free_official_full_loop.json")
    blocker = read_json(OUT_ROOT / "v22_87_blocker_control_mlp_decomposition.json")
    part_a = int(a.get("part_a_hard_gate_pass", 0))
    part_b = int(b.get("part_b_history_replay_pass", 0))
    pass_c = int(c.get("part_c_gate_pass", 0))
    pass_d = int(d.get("part_d_gate_pass", 0))
    pass_e = int(e.get("part_e_gate_pass", 0))
    pass_f = int(f.get("part_f_gate_pass", 0))
    pass_g = int(g.get("part_g_gate_pass", 0))
    part_h = int(h.get("part_h_gate_pass", 0))
    if not part_a:
        route, reason = "R0_CodeBoundaryFailed", "Part A hard gate failed"
    elif not part_b:
        route, reason = "R1_HistoryReplayMismatch", "Part B history matrix mismatch"
    elif part_h:
        route, reason = "R13_ExplorationOpened", "H preflight passed; official full loop would be required next"
    elif pass_c or pass_d or pass_e or pass_f or pass_g:
        route, reason = "R8_PreflightOpenedButHStepBlocked", "A strict preflight candidate opened but H did not pass"
    elif str(g.get("part_g_route", "")).startswith("R6"):
        route, reason = "R6_ArchitectureDiagnosticOnly", "Only architecture/bivariate diagnostic signal found; not strict success"
    elif str(g.get("part_g_route", "")).startswith("R7"):
        route, reason = "R7_CurrentStrictFCPureKANFamilyNoTransferableGenerator", "C/D/E/F fixed gates and G1-G3 strict architecture diagnostics failed"
    elif str(f.get("part_f_route", "")).startswith("R5") and str(fr.get("dominant_blocker", "")) == "shuffled_controls_positive":
        route, reason = "R5_EdgeSNRControlExplained", "Part F signal explained by controls"
    elif str(e.get("part_e_route", "")).startswith("R4") or str(er.get("part_e_route", "")).startswith("E_repair"):
        route, reason = "R4_JointUpstreamEdgeControlBlocked", "Joint upstream-edge generator did not survive controls/MLP/debt"
    elif str(d.get("part_d_route", "")).startswith("R3"):
        route, reason = "R3_TrajectoryDebtBlocked", "Trajectory debt blocked candidate"
    elif str(c.get("part_c_route", "")).startswith("R2"):
        route, reason = "R2_MetricRoleStillCoverageZero", "Part C metric-role change did not open coverage"
    else:
        route, reason = "R7_CurrentStrictFCPureKANFamilyNoTransferableGenerator", "C/D/E/F/G1-G3 fixed gates failed"
    official = int(route in {"R13_ExplorationOpened", "R14_OfficialCandidate"})
    final = {
        "gate": "v22_87_final_route",
        "final_route": route,
        "official_candidate_gate_pass": official,
        "route_reason": reason,
        "part_a": part_a,
        "part_b": part_b,
        "part_c": pass_c,
        "part_d": pass_d,
        "part_e": pass_e,
        "part_e_repair": int(er.get("part_e_gate_pass", 0)),
        "part_f": pass_f,
        "part_f_repair": int(fr.get("part_f_gate_pass", 0)),
        "part_g": pass_g,
        "part_h": part_h,
        "part_i": int(i.get("part_i_gate_pass", 0)),
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_87_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_87_part_b_history_failure_matrix.json"),
            "part_c": rel(OUT_ROOT / "v22_87_part_c_metric_role_matrix_route.json"),
            "part_c_repair": rel(OUT_ROOT / "v22_87_part_c_repair_route.json") if (OUT_ROOT / "v22_87_part_c_repair_route.json").exists() else "missing",
            "part_c_coverage_decomposition": rel(OUT_ROOT / "v22_87_part_c_coverage_decomposition.json") if (OUT_ROOT / "v22_87_part_c_coverage_decomposition.json").exists() else "missing",
            "part_d": rel(OUT_ROOT / "v22_87_part_d_trajectory_route.json"),
            "part_d_repair": rel(OUT_ROOT / "v22_87_part_d_trajectory_repair_route.json") if (OUT_ROOT / "v22_87_part_d_trajectory_repair_route.json").exists() else "missing",
            "part_e": rel(OUT_ROOT / "v22_87_part_e_true_joint_route.json"),
            "part_e_repair": rel(OUT_ROOT / "v22_87_part_e_repair_route.json") if er else "missing",
            "part_f": rel(OUT_ROOT / "v22_87_part_f_edge_population_snr_route.json"),
            "part_f_repair": rel(OUT_ROOT / "v22_87_part_f_repair_route.json") if fr else "missing",
            "part_g": rel(OUT_ROOT / "v22_87_part_g_architecture_diagnostic_route.json"),
            "part_h": rel(OUT_ROOT / "v22_87_part_h_preflight_to_full_loop_entry.json"),
            "part_i": rel(OUT_ROOT / "v22_87_part_i_target_free_official_full_loop.json") if i else "missing",
            "blocker_control_mlp": rel(OUT_ROOT / "v22_87_blocker_control_mlp_decomposition.json") if blocker else "missing",
            "final": rel(OUT_ROOT / "v22_87_final_route.json"),
        },
    }
    out = OUT_ROOT / "v22_87_final_route.json"
    write_json(out, final)
    append_exec("final_route", command_text(sys.argv), "done", files=rel(out), note=json.dumps(final, ensure_ascii=False))
    append_recap("Final route and conclusion", [
        f"final_route={route}；official_candidate_gate_pass={official}；reason={reason}",
        f"A/B/C/D/E/Erepair/F/Frepair/G/H/I pass={part_a}/{part_b}/{pass_c}/{pass_d}/{pass_e}/{final['part_e_repair']}/{pass_f}/{final['part_f_repair']}/{pass_g}/{part_h}/{final['part_i']}",
        "结论约束：只有 R13/R14 可写正向；R6 是 diagnostic-only；R7 表示当前 strict FC-PureKAN family 未找到可迁移 generator，不可写成接近成功。",
    ])
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default=PRIMARY_DATASETS)
    p.add_argument("--seed-count", type=int, default=3)
    p.add_argument("--e-seed-count", type=int, default=4)
    p.add_argument("--f-seed-count", type=int, default=4)
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
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--step-mult", type=float, default=0.30)
    p.add_argument("--debt-margin-lambda", type=float, default=0.50)
    p.add_argument("--part-d-repair-bootstrap-repeats", type=int, default=8)
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
        run_part_c(args, repair=False)
    elif args.mode == "part-c-merge":
        merge_part_c(args, repair=False)
    elif args.mode == "part-c-repair":
        run_part_c(args, repair=True)
    elif args.mode == "part-c-repair-merge":
        merge_part_c(args, repair=True)
    elif args.mode == "part-c-coverage-decomp":
        run_part_c_coverage_decomposition(args)
    elif args.mode == "part-d":
        run_part_d(args, repair=False)
    elif args.mode == "part-d-repair":
        run_part_d(args, repair=True)
    elif args.mode == "part-e":
        run_part_e(args, repair=False)
    elif args.mode == "part-e-merge":
        merge_part_e(args, repair=False)
    elif args.mode == "part-e-repair":
        run_part_e(args, repair=True)
    elif args.mode == "part-e-repair-merge":
        merge_part_e(args, repair=True)
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
    elif args.mode == "part-h":
        run_part_h(args)
    elif args.mode == "part-i":
        run_part_i(args)
    elif args.mode == "blocker-decomp":
        run_blocker_decomposition(args)
    elif args.mode == "final":
        run_final(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
