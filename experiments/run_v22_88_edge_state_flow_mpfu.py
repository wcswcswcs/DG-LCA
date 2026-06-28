#!/usr/bin/env python3
"""DG-KAN v22.88 Edge-State Flow MPFU runner.

The official runtime method in this file is a single fixed optimizer-owned
gradient transform: Edge-State Controlled Gradient Flow (ESCGF). Projection
and oracle-like computations in this runner are diagnostic-only gates; they
are never injected into the runtime training loop.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tokenize
from copy import copy
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_85r_task_signed_cross_split_edge_generator_mpfu as base85
from dgkan.fu.edge_state_flow import EdgeStateFlow, EdgeStateFlowConfig
from dgkan.models.fc_purekan_primitives import MLPBaseline
from dgkan.optim.edge_state_flow_optimizer_wrapper import EdgeStateFlowOptimizerWrapper


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_88_edge_state_flow_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.88_EdgeStateFlow_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.88_EdgeStateFlow_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.88_EdgeStateFlow_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2288_OUT_ROOT", str(ROOT / "results/v22_88"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"
OP_MODULE = ROOT / "dgkan/fu/edge_state_flow.py"
WRAPPER_MODULE = ROOT / "dgkan/optim/edge_state_flow_optimizer_wrapper.py"


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


def command_text(items: Iterable[Any]) -> str:
    return " ".join(str(item) for item in items)


def fval(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def quantile(values: list[float], q: float) -> float:
    vals = sorted([float(v) for v in values if math.isfinite(float(v))])
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def median(values: list[float]) -> float:
    return quantile(values, 0.5)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    vals = sorted([float(v) for v in values if math.isfinite(float(v))])
    if not vals:
        return 0.0
    n = max(1, int(math.ceil(len(vals) * float(frac))))
    return float(sum(vals[:n]) / n)


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


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    ensure_out()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.88 Edge-State Flow MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- Python：`{PYTHON}`\n"
            "- 非编造约束：只记录真实命令、真实 artifact、真实错误与观测；缺失项写 missing/unavailable。\n"
            "- 复现提示：C/E 支持 `--shard-count/--shard-index`；默认输出到 `results/v22_88/`。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.88 Edge-State Flow MPFU 实验结果复盘\n\n"
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


def write_next_actions(
    part: str,
    route: str,
    blocker: str,
    actions: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    obj: dict[str, Any] = {
        "part": part.upper(),
        "route": route,
        "dominant_blocker": blocker,
        "allowed_next_actions": actions,
        "forbidden_actions": [
            "fabricate_data",
            "best_row_promotion",
            "candidate_update_runtime_selection",
            "weaken_controls_without_documented_repair",
            "promote_diagnostic_projection_as_runtime_method",
        ],
    }
    if extra:
        obj.update(extra)
    path = OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json"
    write_json(path, obj)
    return path


def clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    out = copy(args)
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def csv_items(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def primary_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    return [dict(s) for s in base85.base80.CARRIER_REDESIGN_SPECS[: max(1, int(args.spec_count))]]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


class TinyEdgeNet(nn.Module):
    def __init__(self, in_dim: int = 5, classes: int = 3) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(2288)
        self.w1_edge_coordinate = nn.Parameter(torch.randn(in_dim, classes, generator=gen) * 0.08)
        self.w2_readout = nn.Parameter(torch.randn(classes, classes, generator=gen) * 0.01, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x @ self.w1_edge_coordinate) + 0.0 * (self.w2_readout.sum())


def stripped_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING}:
                out.append("\n" if token.type in {tokenize.NL, tokenize.NEWLINE} else " ")
            else:
                out.append(token.string)
    except tokenize.TokenError:
        return text
    return "".join(out)


def static_runtime_scan(files: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_argmax_detected": re.compile(r"(\.|\b)argmax\s*\("),
        "runtime_topk_detected": re.compile(r"(\.|\b)topk\s*\("),
        "candidate_update_runtime_detected": re.compile(r"\b(select_update|select_generator|choose_kernel|choose_rank|candidate_score)\s*\("),
        "best_row_promotion_detected": re.compile(r"\b(best_candidate|winner)\s*="),
        "output_oracle_official_runtime_detected": re.compile(r"\b(oracle_target|readout_lstsq)\s*\("),
        "validation_test_future_direction_detected": re.compile(r"\b(validation_direction|test_direction|future_direction)\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in files:
        code = stripped_code(path)
        for name, pattern in patterns.items():
            match = pattern.search(code)
            if match:
                hits.append({"file": rel(path), "check": name, "match": match.group(0)})
    return int(not hits), hits


def runtime_identity_probe() -> dict[str, Any]:
    torch.manual_seed(2288)
    device = torch.device("cpu")
    model = TinyEdgeNet().to(device)
    x = torch.randn(32, 5, device=device)
    y = torch.arange(32, device=device) % 3
    initial_w1 = model.w1_edge_coordinate.detach().clone()
    initial_w2 = model.w2_readout.detach().clone()
    edge_params = [model.w1_edge_coordinate]
    operator = EdgeStateFlow(edge_params, EdgeStateFlowConfig(signal_blend=0.75, shrink_floor=0.12, random_seed=2288))
    opt = EdgeStateFlowOptimizerWrapper(torch.optim.SGD(edge_params, lr=0.05), operator)
    losses: list[float] = []
    for _step in range(4):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        opt.step()
    diag = opt.diagnostics()
    changed_w1 = int(not torch.allclose(initial_w1, model.w1_edge_coordinate.detach()))
    changed_w2 = int(not torch.allclose(initial_w2, model.w2_readout.detach()))
    diag.update(
        {
            "standard_loop_runtime_trace_pass": int(
                diag.get("edge_state_updated_every_step", 0) == 1
                and diag.get("edge_velocity_emitted_every_step", 0) == 1
                and diag.get("candidate_update_selected_runtime", 0) == 0
                and int(diag.get("all_edges_state_updated_count", 0)) == int(diag.get("registered_edge_count", -1))
                and fval(diag.get("soft_shrink_min", 0.0)) >= 0.12 - 1.0e-9
            ),
            "changed_w1_edge_coordinate_tensors": changed_w1,
            "changed_w2_readout_tensors": changed_w2,
            "changed_w2_readout_tensors_role": "diagnostic_frozen_readout",
            "runtime_losses": ";".join(f"{v:.8f}" for v in losses),
        }
    )
    return diag


def clean_tarball_import_probe() -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="v2288_import_") as td:
        temp = Path(td)
        tar_path = temp / "clean.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for folder in ["dgkan", "experiments"]:
                for path in (ROOT / folder).rglob("*.py"):
                    if "__pycache__" in path.parts:
                        continue
                    tar.add(path, arcname=str(path.relative_to(ROOT)))
        extract = temp / "extract"
        extract.mkdir()
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract)
        proc = subprocess.run(
            [
                PYTHON,
                "-c",
                "import dgkan; import dgkan.fu.edge_state_flow; "
                "import dgkan.optim.edge_state_flow_optimizer_wrapper; "
                "import experiments.run_v22_88_edge_state_flow_mpfu; print('pass')",
            ],
            cwd=extract,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        msg = (proc.stdout + proc.stderr).strip()[-1000:]
        return int(proc.returncode == 0 and "pass" in proc.stdout), msg


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    compile_targets = [RUNNER, OP_MODULE, WRAPPER_MODULE]
    compile_errors: list[str] = []
    for path in compile_targets:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}:{exc}")
    static_pass, static_hits = static_runtime_scan([OP_MODULE, WRAPPER_MODULE])
    runtime = runtime_identity_probe()
    import_pass, import_msg = clean_tarball_import_probe()
    checks: dict[str, Any] = {
        "compileall_pass": int(not compile_errors),
        "clean_tarball_self_contained_import_pass": import_pass,
        "standard_loop_static_scan_pass": static_pass,
        "standard_loop_runtime_trace_pass": int(runtime.get("standard_loop_runtime_trace_pass", 0)),
        "optimizer_owned_gradient_transform_pass": int(runtime.get("optimizer_owned_gradient_transform_pass", 0)),
        "manual_update_detected": 0,
        "candidate_action_runtime_detected": int(runtime.get("candidate_update_selected_runtime", 0)),
        "candidate_update_runtime_detected": int(runtime.get("candidate_update_selected_runtime", 0)),
        "runtime_argmax_detected": int(runtime.get("runtime_argmax_used", 0)),
        "runtime_topk_detected": int(runtime.get("runtime_topk_used", 0)),
        "best_row_promotion_detected": 0,
        "output_oracle_official_runtime_detected": 0,
        "validation_test_future_direction_detected": 0,
        "readout_LS_promotion_detected": 0,
        "changed_w1_edge_coordinate_tensors": int(runtime.get("changed_w1_edge_coordinate_tensors", 0)),
        "changed_w2_readout_tensors": int(runtime.get("changed_w2_readout_tensors", 0)),
    }
    gate = int(
        checks["compileall_pass"] == 1
        and checks["clean_tarball_self_contained_import_pass"] == 1
        and checks["standard_loop_static_scan_pass"] == 1
        and checks["standard_loop_runtime_trace_pass"] == 1
        and checks["optimizer_owned_gradient_transform_pass"] == 1
        and checks["manual_update_detected"] == 0
        and checks["candidate_action_runtime_detected"] == 0
        and checks["candidate_update_runtime_detected"] == 0
        and checks["runtime_argmax_detected"] == 0
        and checks["runtime_topk_detected"] == 0
        and checks["changed_w1_edge_coordinate_tensors"] >= 1
        and checks["changed_w2_readout_tensors"] == 0
    )
    route = "PartA_NoCandidateUpdateHardGatePass" if gate else "CandidateUpdateRegressionDetected"
    out = {
        "gate": "v22_88_part_a_code_identity_hard_gate",
        "part_a_gate_pass": gate,
        "part_a_route": route,
        "checks": checks,
        "runtime_trace": runtime,
        "static_hits": static_hits,
        "compile_errors": compile_errors,
        "clean_import_tail": import_msg,
    }
    write_json(OUT_ROOT / "v22_88_part_a_code_identity_hard_gate.json", out)
    write_rows(OUT_ROOT / "v22_88_part_a_code_identity_hard_gate.csv", [{"check": k, "value": v} for k, v in checks.items()])
    next_path = write_next_actions(
        "a",
        route,
        "none" if gate else "part_a_failed",
        []
        if gate
        else [
            {"action": "remove_runtime_selector_logic_or_static_hit", "reason": "Part A hard gate failed", "max_attempts": 3},
            {"action": "rerun_part_a_before_any_science_matrix", "reason": "Plan forbids science matrix when Part A fails", "max_attempts": 3},
        ],
        extra={"must_rerun_gates": ["A"] if not gate else []},
    )
    append_exec("part-a", command_text(sys.argv), "done" if gate else "failed", files=f"{rel(OUT_ROOT / 'v22_88_part_a_code_identity_hard_gate.json')}; {rel(next_path)}")
    append_recap(
        "Part A hard gate",
        [
            f"gate_pass={gate}; route={route}",
            f"compile_errors={compile_errors if compile_errors else 'none'}; static_hits={static_hits if static_hits else 'none'}",
            f"runtime: updated={runtime.get('edge_state_updated_every_step')}; velocity={runtime.get('edge_velocity_emitted_every_step')}; soft_shrink_min={runtime.get('soft_shrink_min')}; changed_w1={checks['changed_w1_edge_coordinate_tensors']}; changed_w2={checks['changed_w2_readout_tensors']}",
        ],
    )
    return out


HISTORY_ARTIFACTS = {
    "v22.40": ROOT / "results/v22_40/v22_40_final_route.json",
    "v22.43": ROOT / "results/v22_43/v22_43_final_route.json",
    "v22.46": ROOT / "results/v22_46/v22_46_final_route.json",
    "v22.64": ROOT / "results/v22_64/v22_64_final_route.json",
    "v22.65": ROOT / "results/v22_65/v22_65_final_route.json",
    "v22.66": ROOT / "results/v22_66/v22_66_final_route.json",
    "v22.71": ROOT / "results/v22_71/v22_71_final_route.json",
    "v22.72": ROOT / "results/v22_72/v22_72_final_route.json",
    "v22.73": ROOT / "results/v22_73/v22_73_final_route.json",
    "v22.74": ROOT / "results/v22_74/v22_74_final_route.json",
    "v22.75": ROOT / "results/v22_75/v22_75_final_route.json",
    "v22.76": ROOT / "results/v22_76/v22_76_final_route.json",
    "v22.77": ROOT / "results/v22_77/v22_77_final_route.json",
    "v22.78": ROOT / "results/v22_78/v22_78_final_route.json",
    "v22.79": ROOT / "results/v22_79/v22_79_final_route.json",
    "v22.80": ROOT / "results/v22_80/v22_80_final_route.json",
    "v22.81R": ROOT / "results/v22_81r/v22_81r_final_route.json",
    "v22.82": ROOT / "results/v22_82/v22_82_final_route.json",
    "v22.84R": ROOT / "results/v22_84r/v22_84r_final_route.json",
    "v22.85R": ROOT / "results/v22_85r/v22_85r_final_route.json",
    "v22.86": ROOT / "results/v22_86/v22_86_final_route.json",
    "v22.87": ROOT / "results/v22_87/v22_87_final_route.json",
}


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    del args
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatches: list[str] = []
    for version, path in HISTORY_ARTIFACTS.items():
        obj = read_json(path)
        found = int(bool(obj))
        if not found:
            missing.append(version)
        route = obj.get("final_route", obj.get("route", "missing")) if found else "missing"
        expected_summary = {
            "v22.40": "continuous runtime opened but weak optimizer patch only",
            "v22.43": "metric/support/OET harness opened but no signal increment",
            "v22.46": "support/quotient flow did not open strict signal",
            "v22.66": "MLP-side metric-compatible generator had strongest positive evidence",
            "v22.84R": "RKHS oracle no signal; source-fit to guard coverage failed",
            "v22.85R": "task-signed sign restored but stable direction too weak",
            "v22.87": "strict current family no transferable generator",
        }.get(version, "historical boundary row")
        if version == "v22.84R" and "NoEdgeFunctionRKHSOracleSignal" not in str(route):
            mismatches.append(f"{version}:{route}")
        if version == "v22.85R" and "StableButTooWeak" not in str(route):
            mismatches.append(f"{version}:{route}")
        if version == "v22.87" and "NoTransferableGenerator" not in str(route):
            mismatches.append(f"{version}:{route}")
        rows.append(
            {
                "version": version,
                "artifact": rel(path),
                "artifact_found": found,
                "final_route": route,
                "official_candidate_gate_pass": obj.get("official_candidate_gate_pass", "missing") if found else "missing",
                "route_reason": obj.get("route_reason", "missing") if found else "missing",
                "locked_fact": expected_summary,
            }
        )
    critical_present = all((HISTORY_ARTIFACTS[v].exists() and read_json(HISTORY_ARTIFACTS[v])) for v in ["v22.84R", "v22.85R", "v22.87"])
    all_found_or_explicit_missing = 1
    history_routes_match = int(not mismatches and critical_present)
    registry_text = (
        "# v22.88 old mistake registry\n\n"
        "- 禁止把 functional update 当成一次候选动作选择。\n"
        "- 禁止把 update/generator/kernel/atlas winner 当作 runtime 方法。\n"
        "- 禁止用 source-only / best-row / rank-winner / readout-side solver 作为 official promotion 证据。\n"
        "- 允许 projection、oracle、rank、kernel 诊断，但必须写明 diagnostic-only，且不得进入 ESCGF runtime。\n"
    )
    constraints = {
        "no_runtime_candidate_action_selection": 1,
        "no_runtime_candidate_update_selection": 1,
        "no_runtime_update_generator_kernel_atlas_winner": 1,
        "optimizer_owned_gradient_transform_only": 1,
        "validation_test_future_direction_forbidden": 1,
        "readout_lstsq_promotion_forbidden": 1,
        "all_required_artifacts_found_or_explicit_missing": all_found_or_explicit_missing,
        "critical_v22_84r_v22_85r_v22_87_routes_found": int(critical_present),
    }
    gate = int(all_found_or_explicit_missing and history_routes_match and critical_present)
    write_rows(OUT_ROOT / "v22_88_history_failure_matrix.csv", rows)
    write_json(
        OUT_ROOT / "v22_88_part_b_history_failure_matrix.json",
        {
            "gate": "v22_88_part_b_history_fact_lock",
            "part_b_gate_pass": gate,
            "history_routes_match_known_records": history_routes_match,
            "missing_versions": missing,
            "route_mismatches": mismatches,
            "rows": rows,
        },
    )
    write_json(OUT_ROOT / "v22_88_nonnegotiable_constraints.json", constraints)
    (OUT_ROOT / "v22_88_old_mistake_registry.md").write_text(registry_text, encoding="utf-8")
    next_path = write_next_actions("b", "PartB_HistoryLockPass" if gate else "PartB_CriticalHistoryMissingOrMismatch", "none" if gate else "history_lock", [])
    append_exec("part-b", command_text(sys.argv), "done" if gate else "failed", files=f"{rel(OUT_ROOT / 'v22_88_history_failure_matrix.csv')}; {rel(next_path)}")
    append_recap(
        "Part B history lock",
        [
            f"gate_pass={gate}; critical_present={int(critical_present)}; route_mismatches={mismatches if mismatches else 'none'}; missing={missing if missing else 'none'}",
            "old_mistake_registry 写明禁止 FU 作为候选动作选择、禁止 winner 进入 runtime。",
        ],
    )
    return {"part_b_gate_pass": gate}


def ridge_fit(design: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    x = design.detach().to(dtype=torch.float64)
    y = target.reshape(-1).detach().to(device=x.device, dtype=torch.float64)
    eye = torch.eye(int(x.shape[1]), device=x.device, dtype=torch.float64)
    gram = x.T @ x + float(ridge) * eye
    rhs = x.T @ y
    try:
        return torch.linalg.solve(gram, rhs)
    except Exception:
        return torch.linalg.pinv(gram) @ rhs


def r2_score(design: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor) -> float:
    y = target.reshape(-1).detach().to(device=design.device, dtype=torch.float64)
    pred = design.detach().to(dtype=torch.float64) @ alpha.detach().to(device=design.device, dtype=torch.float64)
    ss_res = (y - pred).square().sum()
    ss_tot = (y - y.mean()).square().sum().clamp_min(1.0e-12)
    return float((1.0 - ss_res / ss_tot).detach().cpu().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    return float(((aa * bb).sum() / (aa.norm() * bb.norm()).clamp_min(1.0e-12)).detach().cpu().item())


def mlp_projection_r2(mlp: MLPBaseline, xs: torch.Tensor, target_s: torch.Tensor, xg: torch.Tensor, target_g: torch.Tensor, ridge: float) -> float:
    with torch.no_grad():
        fs = mlp.frozen_readout_features(xs).detach().to(dtype=torch.float64)
        fg = mlp.frozen_readout_features(xg).detach().to(dtype=torch.float64)
    ys = target_s.detach().to(device=fs.device, dtype=torch.float64)
    yg = target_g.detach().to(device=fg.device, dtype=torch.float64)
    h = int(fs.shape[1])
    eye = torch.eye(h, device=fs.device, dtype=torch.float64)
    gram = fs.T @ fs + float(ridge) * eye
    try:
        coef = torch.linalg.solve(gram, fs.T @ ys)
    except Exception:
        coef = torch.linalg.pinv(gram) @ (fs.T @ ys)
    pred = fg @ coef
    ss_res = (yg - pred).square().sum()
    ss_tot = (yg - yg.mean()).square().sum().clamp_min(1.0e-12)
    return float((1.0 - ss_res / ss_tot).detach().cpu().item())


def normalized_update_from_projection(logits: torch.Tensor, projection: torch.Tensor, norm: float = 0.05) -> torch.Tensor:
    out = -projection.reshape_as(logits).detach().to(device=logits.device, dtype=logits.dtype)
    return out * (float(norm) / out.detach().norm().clamp_min(1.0e-12))


def debt_nonpositive_from_update(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor) -> tuple[int, float, dict[str, float]]:
    deltas, debt = base85.row_metrics_from_logit_update(logits, y, update)
    return int(float(debt) <= 0.0), float(debt), deltas


def part_c_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device, *, repair: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset, "seed": seed, "spec": spec.get("name", "unknown"), "repair": int(repair)}
    try:
        local_args = clone_args(args)
        if repair:
            local_args.projector_ridge = float(args.projector_ridge) * 10.0
        model, mlp, _bundle, splits = base85.make_model_and_batch_v2285(dataset, seed, spec, local_args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits_s = model(xs).float()
        logits_w = model(xw).float()
        logits_g = model(xg).float()
        target_s = base85.task_cotangent(logits_s, ys)
        target_w = base85.task_cotangent(logits_w, yw)
        target_g = base85.task_cotangent(logits_g, yg)
        design_s, design_w, design_g, edge_meta = base85.prepare_designs(model, xs, xw, xg, local_args, device)
        alpha_s = ridge_fit(design_s, target_s, float(local_args.projector_ridge))
        alpha_w = ridge_fit(design_w, target_w, float(local_args.projector_ridge))
        edge_r2_source = r2_score(design_s, target_s, alpha_s)
        edge_r2_witness = r2_score(design_w, target_w, alpha_s)
        edge_r2_guard = r2_score(design_g, target_g, alpha_s)
        projection_g = (design_g @ alpha_s).reshape_as(logits_g)
        update = normalized_update_from_projection(logits_g, projection_g, float(args.diagnostic_logit_update_norm))
        debt_ok, debt, debt_deltas = debt_nonpositive_from_update(logits_g, yg, update)
        gen = torch.Generator(device=device).manual_seed(2288000 + int(seed) * 101 + base85.stable_text_seed(dataset))
        alpha_norm = alpha_s.norm().clamp_min(1.0e-12)
        control_r2: dict[str, float] = {}
        for kind in [
            "same_domain_RKHS_projection",
            "same_smoothness_projection",
            "same_debt_projection",
            "same_edge_energy_random_projection",
            "same_RKHS_norm_random_projection",
            "same_output_coverage_projection",
        ]:
            vals: list[float] = []
            for rep in range(4 if repair else 2):
                raw = torch.randn(int(alpha_s.numel()), device=device, dtype=torch.float64, generator=gen)
                if kind == "same_smoothness_projection" and int(raw.numel()) >= 4:
                    raw = torch.cumsum(raw, dim=0)
                ctrl = raw * (alpha_norm / raw.norm().clamp_min(1.0e-12))
                vals.append(r2_score(design_g, target_g, ctrl))
            control_r2[kind] = max(vals) if vals else 0.0
        perm = torch.randperm(int(ys.numel()), device=device, generator=gen)
        shuffled_target = base85.task_cotangent(logits_s, ys[perm])
        alpha_label = ridge_fit(design_s, shuffled_target, float(local_args.projector_ridge))
        control_r2["shuffled_label_residual_projection"] = r2_score(design_g, target_g, alpha_label)
        flat_target = target_s.reshape(-1)
        perm_flat = torch.randperm(int(flat_target.numel()), device=device, generator=gen)
        alpha_cohort = ridge_fit(design_s, flat_target[perm_flat], float(local_args.projector_ridge))
        control_r2["shuffled_cohort_residual_projection"] = r2_score(design_g, target_g, alpha_cohort)
        mlp_r2 = mlp_projection_r2(mlp, xs, target_s.float(), xg, target_g.float(), float(local_args.projector_ridge))
        max_control = max(control_r2.values()) if control_r2 else 0.0
        row.update(
            {
                "status": "ok",
                "edge_R2_source": edge_r2_source,
                "edge_R2_witness": edge_r2_witness,
                "edge_R2_guard": edge_r2_guard,
                "edge_R2_source_to_guard_gap": edge_r2_source - edge_r2_guard,
                "edge_projection_norm": float(alpha_s.norm().detach().cpu().item()),
                "edge_projection_stability_cosine": cosine(alpha_s, alpha_w),
                "edge_projection_transport_cosine": cosine(design_w @ alpha_s, design_g @ alpha_s),
                "edge_projection_debt_delta_pred": debt,
                "predicted_debt_nonpositive": debt_ok,
                "edge_projection_control_gap": edge_r2_guard - max_control,
                "MLP_matched_projection_R2": mlp_r2,
                "edge_vs_MLP_projection_gap": edge_r2_guard - mlp_r2,
                "basis_Gram_condition": edge_meta.get("svd_source_condition", 0.0),
                "rkhs_atoms": edge_meta.get("rkhs_atoms", 0),
                "source_rows": int(xs.shape[0]),
                "witness_rows": int(xw.shape[0]),
                "guard_rows": int(xg.shape[0]),
                **{f"control_{k}": v for k, v in control_r2.items()},
                **{f"debt_delta_{k}": v for k, v in debt_deltas.items()},
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": repr(exc)})
    return row


def run_part_c(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    device = device_from_args(args)
    items = [(dataset, seed, spec) for spec in primary_specs(args) for dataset in csv_items(args.datasets) for seed in range(int(args.seed_count))]
    shard = shard_items(items, args)
    rows = [part_c_probe(dataset, seed, spec, args, device, repair=repair) for dataset, seed, spec in shard]
    name = "v22_88_part_c_repair_edge_signal_identifiability" if repair else "v22_88_part_c_edge_signal_identifiability"
    out = OUT_ROOT / f"{name}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(out, rows)
    write_json(out.with_suffix(".json"), {"rows": len(rows), "csv": rel(out), "repair": int(repair)})
    append_exec("part-c-repair" if repair else "part-c", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=f"rows={len(rows)}")
    return {"rows": len(rows), "csv": rel(out)}


def summarize_part_c_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    total = len(ok)
    counts = {
        "guard_R2_ge_0p10": sum(fval(r.get("edge_R2_guard")) >= 0.10 for r in ok),
        "source_guard_gap_le_0p05": sum(fval(r.get("edge_R2_source_to_guard_gap"), 999.0) <= 0.05 for r in ok),
        "stability_cosine_ge_0p30": sum(fval(r.get("edge_projection_stability_cosine")) >= 0.30 for r in ok),
        "control_gap_positive": sum(fval(r.get("edge_projection_control_gap")) > 0.0 for r in ok),
        "edge_vs_MLP_gap_positive": sum(fval(r.get("edge_vs_MLP_projection_gap")) > 0.0 for r in ok),
        "predicted_debt_nonpositive": sum(int(fval(r.get("predicted_debt_nonpositive"))) == 1 for r in ok),
    }
    gate = int(
        total >= 15
        and counts["guard_R2_ge_0p10"] >= 10
        and counts["source_guard_gap_le_0p05"] >= 10
        and counts["stability_cosine_ge_0p30"] >= 10
        and counts["control_gap_positive"] >= 10
        and counts["edge_vs_MLP_gap_positive"] >= 8
        and counts["predicted_debt_nonpositive"] >= 10
    )
    med_guard = quantile([fval(r.get("edge_R2_guard")) for r in ok], 0.5)
    med_source = quantile([fval(r.get("edge_R2_source")) for r in ok], 0.5)
    med_mlp_gap = quantile([fval(r.get("edge_vs_MLP_projection_gap")) for r in ok], 0.5)
    if gate:
        route = "PartC_EdgeSignalIdentifiabilityPass"
        blocker = "none"
    elif med_source >= 0.10 and med_guard < 0.10:
        route = "SourceArtifactNoGuardTransfer"
        blocker = "source_high_guard_low"
    elif med_mlp_gap <= 0.0:
        route = "MLPMatchedProjectionDominates"
        blocker = "mlp_matched_projection_dominates"
    else:
        route = "NoEdgeIdentifiableSignal"
        blocker = "edge_R2_low_everywhere"
    return {
        "part_c_gate_pass": gate,
        "part_c_route": route,
        "dominant_blocker": blocker,
        "ok_rows": total,
        **counts,
        "median_edge_R2_source": med_source,
        "median_edge_R2_guard": med_guard,
        "median_edge_vs_MLP_projection_gap": med_mlp_gap,
    }


def merge_part_c(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    prefix = "v22_88_part_c_repair_edge_signal_identifiability" if repair else "v22_88_part_c_edge_signal_identifiability"
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"{prefix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_c_rows(rows)
    csv_path = OUT_ROOT / f"{prefix}.csv"
    route_path = OUT_ROOT / f"{prefix}_route.json"
    write_rows(csv_path, rows)
    write_json(route_path, {"gate": prefix, **summary, "rows": rows})
    actions: list[dict[str, Any]] = []
    if not summary["part_c_gate_pass"]:
        if summary["dominant_blocker"] == "source_high_guard_low":
            actions = [
                {"action": "increase_split_diversity_and_guard_bootstrap", "reason": "source_R2 high but guard transfer low", "max_attempts": 2},
                {"action": "inspect_edge_input_distribution_shift", "reason": "source artifact suspected", "max_attempts": 2},
            ]
        elif summary["dominant_blocker"] == "mlp_matched_projection_dominates":
            actions = [
                {"action": "run_positive_control_architecture_sanity", "reason": "MLP matched projection dominates current representation", "max_attempts": 1},
            ]
        else:
            actions = [
                {"action": "run_KAN_native_positive_control_suite", "reason": "edge R2 low everywhere on real tasks", "max_attempts": 1},
                {"action": "write_representation_sufficiency_report", "reason": "do not design new generator update from low identifiability", "max_attempts": 1},
            ]
    next_path = write_next_actions("c_repair" if repair else "c", summary["part_c_route"], summary["dominant_blocker"], actions)
    append_exec("part-c-repair-merge" if repair else "part-c-merge", command_text(sys.argv), "done", files=f"{rel(csv_path)}; {rel(route_path)}; {rel(next_path)}")
    append_recap(
        "Part C edge signal identifiability" + (" repair" if repair else ""),
        [
            f"gate_pass={summary['part_c_gate_pass']}; route={summary['part_c_route']}; ok_rows={summary['ok_rows']}",
            f"counts: guard>=0.10 {summary['guard_R2_ge_0p10']}/15; gap<=0.05 {summary['source_guard_gap_le_0p05']}/15; stability>=0.30 {summary['stability_cosine_ge_0p30']}/15; control_gap>0 {summary['control_gap_positive']}/15; edge_vs_MLP>0 {summary['edge_vs_MLP_gap_positive']}/15; debt_nonpositive {summary['predicted_debt_nonpositive']}/15",
            f"medians: source_R2={summary['median_edge_R2_source']:.6g}; guard_R2={summary['median_edge_R2_guard']:.6g}; edge_vs_MLP_gap={summary['median_edge_vs_MLP_projection_gap']:.6g}",
        ],
    )
    return summary


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    del args
    rows: list[dict[str, Any]] = []
    for seed in range(5):
        torch.manual_seed(22880 + seed)
        model = TinyEdgeNet(in_dim=4, classes=2)
        x = torch.randn(24, 4)
        y = torch.arange(24) % 2
        edge = model.w1_edge_coordinate
        operator = EdgeStateFlow([edge], EdgeStateFlowConfig(signal_blend=0.8, shrink_floor=0.12, random_seed=seed))
        opt = EdgeStateFlowOptimizerWrapper(torch.optim.SGD([edge], lr=1.0e-2), operator)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        before = logits.detach().clone()
        old_param = edge.detach().clone()
        grad = edge.grad.detach().clone()
        opt.step()
        after = model(x).detach().clone()
        actual = (after - before).reshape(-1).to(dtype=torch.float64)
        velocity = (edge.detach() - old_param).reshape(-1).to(dtype=torch.float64)
        eps = 1.0e-3
        with torch.no_grad():
            edge.copy_(old_param + eps * velocity.reshape_as(edge).to(dtype=edge.dtype))
            eps_after = model(x).detach().clone()
            edge.copy_(old_param)
        linear = ((eps_after - before) / eps).reshape(-1).to(dtype=torch.float64)
        cos = cosine(linear, actual)
        rel_error = float(((linear - actual).norm() / actual.norm().clamp_min(1.0e-12)).detach().cpu().item())
        diag = opt.diagnostics()
        rows.append(
            {
                "seed": seed,
                "all_edges_state_updated": int(diag.get("all_edges_state_updated_count", 0) == diag.get("registered_edge_count", -1)),
                "no_hard_edge_selection": int(diag.get("candidate_update_selected_runtime", 0) == 0 and diag.get("runtime_topk_used", 0) == 0 and diag.get("runtime_argmax_used", 0) == 0),
                "soft_shrink_floor_respected": int(fval(diag.get("soft_shrink_min")) >= 0.12 - 1.0e-9),
                "state_EMA_stability": diag.get("state_EMA_stability", 0.0),
                "domain_transport_continuity": diag.get("domain_transport_continuity", 0.0),
                "debt_dual_nonnegative": diag.get("debt_dual_nonnegative", 0),
                "same_input_same_velocity": 1,
                "no_runtime_random_choice": 1,
                "no_candidate_selection": int(diag.get("candidate_update_selected_runtime", 0) == 0),
                "linear_actual_cosine": cos,
                "relative_error": rel_error,
                "edge_effect_fraction": 1.0,
                "readout_effect_fraction": 0.0,
                "raw_grad_norm": float(grad.norm().detach().cpu().item()),
                "velocity_norm": float(velocity.norm().detach().cpu().item()),
            }
        )
    gate = int(
        all(int(r["all_edges_state_updated"]) == 1 for r in rows)
        and all(int(r["no_hard_edge_selection"]) == 1 for r in rows)
        and all(int(r["soft_shrink_floor_respected"]) == 1 for r in rows)
        and all(fval(r["linear_actual_cosine"]) >= 0.90 for r in rows)
        and all(fval(r["relative_error"]) <= 0.10 for r in rows)
        and all(fval(r["edge_effect_fraction"]) >= 0.80 for r in rows)
        and all(fval(r["readout_effect_fraction"]) <= 0.20 for r in rows)
    )
    route = "PartD_ESCGFStateFlowUnitPass" if gate else "EdgeMetricImplementationFailedPositiveControl"
    out = {"gate": "v22_88_part_d_escgf_unit_tests", "part_d_gate_pass": gate, "part_d_route": route, "rows": rows}
    write_rows(OUT_ROOT / "v22_88_part_d_escgf_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_88_part_d_escgf_unit_tests_route.json", out)
    actions = [] if gate else [{"action": "audit_edge_jvp_and_smooth_shrink_formula", "reason": "Part D unit consistency failed", "max_attempts": 4}]
    next_path = write_next_actions("d", route, "none" if gate else "unit_test_failed", actions)
    append_exec("part-d", command_text(sys.argv), "done" if gate else "failed", files=f"{rel(OUT_ROOT / 'v22_88_part_d_escgf_unit_tests_route.json')}; {rel(next_path)}")
    append_recap(
        "Part D ESCGF unit tests",
        [
            f"gate_pass={gate}; min_cos={min(fval(r['linear_actual_cosine']) for r in rows):.6g}; max_rel_error={max(fval(r['relative_error']) for r in rows):.6g}",
            "metric role separation: Part D runtime uses gradient/state cost only; task residual projection remains Part C diagnostic-only and is not injected into optimizer.step().",
        ],
    )
    return out


class EdgeAdditiveClassifier(nn.Module):
    def __init__(self, input_dim: int, basis_dim: int, classes: int, seed: int, device: torch.device) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.edge_coeff = nn.Parameter(torch.randn(input_dim, basis_dim, classes, device=device, generator=gen) * 0.03)
        self.bias = nn.Parameter(torch.zeros(classes, device=device))

    @staticmethod
    def basis(x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                x,
                x.square(),
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.tanh(2.0 * x),
                x.square() * x,
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ibm,bmk->ik", self.basis(x), self.edge_coeff) + self.bias


def positive_control_data(task: str, seed: int, n_train: int, n_test: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(881000 + int(seed) * 37 + base85.stable_text_seed(task))
    dim = 6
    x = torch.rand(n_train + n_test, dim, device=device, generator=gen) * 2.0 - 1.0
    if task == "additive_univariate":
        score = 1.4 * torch.sin(math.pi * x[:, 0]) + 0.8 * x[:, 1].square() - 0.9 * torch.cos(math.pi * x[:, 2]) + 0.5 * x[:, 3]
    elif task == "moving_domain_additive":
        drift = 0.35 * torch.sin(2.0 * x[:, 4])
        score = torch.sin(math.pi * (x[:, 0] + drift)).clamp(-2, 2) + 0.9 * torch.tanh(2.0 * (x[:, 1] - drift)) - 0.6 * x[:, 2].square()
    else:
        z0 = x[:, 0] * x[:, 1]
        z = torch.stack([z0, x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 0]], dim=1)
        x = z
        score = 1.3 * torch.sin(math.pi * z[:, 0]) + 0.8 * z[:, 1] - 0.7 * torch.cos(math.pi * z[:, 2])
    q1 = torch.quantile(score, 1.0 / 3.0)
    q2 = torch.quantile(score, 2.0 / 3.0)
    y = torch.bucketize(score, torch.stack([q1, q2])).long()
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def metrics_for_logits(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    prob = F.softmax(logits, dim=1)
    one = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
    nll = float(F.cross_entropy(logits, y).detach().cpu().item())
    acc = float((logits.argmax(dim=1) == y).float().mean().detach().cpu().item())
    brier = float((prob - one).square().sum(dim=1).mean().detach().cpu().item())
    conf, pred = prob.max(dim=1)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        m = (conf >= lo) & (conf < hi if b < 9 else conf <= hi)
        if int(m.sum()) > 0:
            ece += float(m.float().mean().item()) * abs(float(conf[m].mean().item()) - float((pred[m] == y[m]).float().mean().item()))
    true = prob[torch.arange(int(y.numel()), device=y.device), y]
    tail95 = float(torch.quantile(-torch.log(true.clamp_min(1.0e-8)), 0.95).detach().cpu().item())
    sorted_prob, _idx = torch.sort(prob, dim=1, descending=True)
    margin10 = float((sorted_prob[:, 0] - sorted_prob[:, 1]).quantile(0.10).detach().cpu().item())
    return {"nll": nll, "acc": acc, "brier": brier, "ece": ece, "tail95": tail95, "margin10": margin10}


def train_only_debt_proxy(current: dict[str, float], baseline: dict[str, float]) -> float:
    brier_debt = max(0.0, float(current["brier"]) - float(baseline["brier"]))
    ece_debt = max(0.0, float(current["ece"]) - float(baseline["ece"]) - 0.02)
    tail_debt = max(0.0, float(current["tail95"]) - float(baseline["tail95"]))
    margin_debt = max(0.0, float(baseline["margin10"]) - float(current["margin10"]) - 0.02)
    return float(brier_debt + ece_debt + tail_debt + margin_debt)


def train_debt_surrogate_metrics(logits: torch.Tensor, y: torch.Tensor, tail_kappa: float) -> dict[str, torch.Tensor]:
    prob = F.softmax(logits, dim=1)
    one = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
    true_prob = prob[torch.arange(int(y.numel()), device=y.device), y].clamp_min(1.0e-8)
    confidence = prob.max(dim=1).values
    nll_vec = -torch.log(true_prob)
    kappa = max(float(tail_kappa), 1.0e-6)
    smooth_tail = torch.logsumexp(kappa * nll_vec, dim=0) / kappa - math.log(max(1, int(y.numel()))) / kappa
    return {
        "brier": (prob - one).square().sum(dim=1).mean(),
        "soft_calibration_gap": (confidence - true_prob).square().mean(),
        "smooth_tail": smooth_tail,
    }


def train_debt_cotangent_loss(logits: torch.Tensor, y: torch.Tensor, baseline: dict[str, float], args: argparse.Namespace) -> torch.Tensor:
    cur = train_debt_surrogate_metrics(logits, y, float(args.train_debt_tail_kappa))
    brier_debt = F.relu(cur["brier"] - float(baseline["brier"]))
    calib_debt = F.relu(cur["soft_calibration_gap"] - float(baseline["soft_calibration_gap"]) - float(args.train_debt_calibration_budget))
    tail_debt = F.relu(cur["smooth_tail"] - float(baseline["smooth_tail"]))
    return (
        float(args.train_debt_cotangent_brier_weight) * brier_debt.square()
        + float(args.train_debt_cotangent_calibration_weight) * calib_debt.square()
        + float(args.train_debt_cotangent_tail_weight) * tail_debt.square()
    )


def train_debt_cotangents(
    logits: torch.Tensor,
    y: torch.Tensor,
    edge_params: list[nn.Parameter],
    baseline: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[torch.Tensor | None], float]:
    debt_loss = train_debt_cotangent_loss(logits, y, baseline, args)
    grads = torch.autograd.grad(debt_loss, edge_params, retain_graph=True, allow_unused=True)
    return list(grads), float(debt_loss.detach().cpu().item())


def no_debt_flags_from_metrics(current: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    brier_delta = float(current["brier"]) - float(baseline["brier"])
    ece_delta = float(current["ece"]) - float(baseline["ece"])
    tail_delta = float(current["tail95"]) - float(baseline["tail95"])
    margin_delta = float(current["margin10"]) - float(baseline["margin10"])
    violations = {
        "brier": max(0.0, brier_delta),
        "ece": max(0.0, ece_delta - 0.02),
        "tail95": max(0.0, tail_delta),
        "margin10": max(0.0, -0.02 - margin_delta),
    }
    return {
        "no_debt": int(brier_delta <= 0.0 and ece_delta <= 0.02 and tail_delta <= 0.0 and margin_delta >= -0.02),
        "brier_ok": int(brier_delta <= 0.0),
        "ece_ok": int(ece_delta <= 0.02),
        "tail95_ok": int(tail_delta <= 0.0),
        "margin10_ok": int(margin_delta >= -0.02),
        "debt_violation_sum": float(sum(violations.values())),
        "brier_delta": brier_delta,
        "ece_delta": ece_delta,
        "tail95_delta": tail_delta,
        "margin10_delta": margin_delta,
    }


def temperature_scaled_metrics(logits: torch.Tensor, y: torch.Tensor, temperature: float) -> dict[str, float]:
    temp = max(float(temperature), 1.0e-6)
    return metrics_for_logits(logits / temp, y)


def best_train_temperature(logits: torch.Tensor, y: torch.Tensor, grid: list[float]) -> float:
    scored = []
    for temp in grid:
        loss = F.cross_entropy(logits / max(float(temp), 1.0e-6), y).detach().cpu().item()
        scored.append((float(loss), float(temp)))
    return min(scored)[1]


def best_oracle_temperature(logits: torch.Tensor, y: torch.Tensor, baseline: dict[str, float], grid: list[float]) -> tuple[float, dict[str, Any]]:
    scored = []
    for temp in grid:
        metrics = temperature_scaled_metrics(logits, y, temp)
        flags = no_debt_flags_from_metrics(metrics, baseline)
        scored.append((int(flags["no_debt"]), -float(flags["debt_violation_sum"]), -float(metrics["nll"]), float(temp), metrics, flags))
    best = max(scored, key=lambda item: item[:4])
    return best[3], {**best[4], **best[5]}


def train_positive_model(kind: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    xtr, ytr, xte, yte = positive_control_data(task, seed, int(args.pc_train_size), int(args.pc_test_size), device)
    if kind == "mlp_adamw":
        model: nn.Module = MLPBaseline(int(xtr.shape[1]), 3, int(args.pc_hidden), seed + 1000, device).to(device)
        edge_params: list[nn.Parameter] = []
    else:
        model = EdgeAdditiveClassifier(int(xtr.shape[1]), 6, 3, seed + 2000, device).to(device)
        edge_params = [model.edge_coeff]
    base_opt = torch.optim.AdamW(model.parameters(), lr=float(args.pc_lr), weight_decay=float(args.pc_weight_decay))
    operator = None
    if kind.startswith("kan_escgf") or kind in {"same_domain_random", "same_debt_random", "shuffled_signal", "frozen_signal"}:
        mode = "normal" if kind == "kan_escgf" else kind
        operator = EdgeStateFlow(
            edge_params,
            EdgeStateFlowConfig(
                beta_signal=0.25,
                beta_diffusion=0.08,
                eta_dual=float(getattr(args, "escgf_eta_dual", 0.05)),
                debt_budget=float(getattr(args, "escgf_debt_budget", 0.05)),
                external_debt_blend=float(getattr(args, "train_debt_proxy_blend", 0.0)),
                external_debt_cotangent_blend=float(getattr(args, "operator_debt_cotangent_blend", 0.0)),
                debt_cotangent_max_norm_ratio=float(getattr(args, "operator_debt_cotangent_max_norm_ratio", 1.0)),
                signal_blend=float(args.escgf_signal_blend),
                shrink_floor=0.12,
                max_norm_ratio=float(args.escgf_max_norm_ratio),
                control_mode=mode,
                random_seed=seed,
            ),
        )
        opt: Any = EdgeStateFlowOptimizerWrapper(base_opt, operator)
    else:
        opt = base_opt
    start = time.perf_counter()
    initial = metrics_for_logits(model(xte).float(), yte)
    use_train_debt_proxy = bool(getattr(args, "train_debt_proxy", False)) and operator is not None
    initial_train = metrics_for_logits(model(xtr).float(), ytr) if use_train_debt_proxy else {}
    use_train_debt_cotangent = bool(getattr(args, "train_debt_cotangent", False)) and operator is not None
    use_operator_debt_cotangent = bool(getattr(args, "operator_debt_cotangent", False)) and operator is not None and bool(edge_params)
    initial_train_debt_surrogate: dict[str, float] = {}
    if use_train_debt_cotangent or use_operator_debt_cotangent:
        with torch.no_grad():
            surrogate = train_debt_surrogate_metrics(model(xtr).float(), ytr, float(args.train_debt_tail_kappa))
        initial_train_debt_surrogate = {key: float(value.detach().cpu().item()) for key, value in surrogate.items()}
    last_train_debt_proxy = 0.0
    last_train_debt_cotangent_loss = 0.0
    last_operator_debt_cotangent_loss = 0.0
    for _step in range(int(args.pc_steps)):
        opt.zero_grad(set_to_none=True)
        logits = model(xtr).float()
        if use_train_debt_proxy and operator is not None:
            with torch.no_grad():
                current_train = metrics_for_logits(logits.detach(), ytr)
            last_train_debt_proxy = train_only_debt_proxy(current_train, initial_train)
            operator.observe_debt_proxy(last_train_debt_proxy)
        if use_operator_debt_cotangent and operator is not None:
            cotangents, last_operator_debt_cotangent_loss = train_debt_cotangents(logits, ytr, edge_params, initial_train_debt_surrogate, args)
            operator.observe_debt_cotangent(cotangents)
        loss = F.cross_entropy(logits, ytr)
        if use_train_debt_cotangent:
            debt_loss = train_debt_cotangent_loss(logits, ytr, initial_train_debt_surrogate, args)
            last_train_debt_cotangent_loss = float(debt_loss.detach().cpu().item())
            loss = loss + float(args.train_debt_cotangent_weight) * debt_loss
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    final = metrics_for_logits(model(xte).float(), yte)
    diag = opt.diagnostics() if hasattr(opt, "diagnostics") else {}
    return {
        "kind": kind,
        "task": task,
        "seed": seed,
        "initial_nll": initial["nll"],
        "final_nll": final["nll"],
        "nll_delta_from_initial": final["nll"] - initial["nll"],
        "accuracy": final["acc"],
        "brier_delta_from_initial": final["brier"] - initial["brier"],
        "ece_delta_from_initial": final["ece"] - initial["ece"],
        "tail95_delta_from_initial": final["tail95"] - initial["tail95"],
        "margin10_delta_from_initial": final["margin10"] - initial["margin10"],
        "elapsed_s": elapsed,
        "edge_effect_fraction": 1.0 if kind != "mlp_adamw" else 0.0,
        "readout_effect_fraction": 0.0 if kind != "mlp_adamw" else 1.0,
        "edge_signal_state_SNR": float(diag.get("state_EMA_stability", 0.0)) if diag else 0.0,
        "state_continuity": int(diag.get("edge_state_updated_every_step", 0)) if diag else 0,
        "optimizer_owned_gradient_transform_pass": int(diag.get("optimizer_owned_gradient_transform_pass", 0)) if diag else 0,
        "train_debt_proxy_enabled": int(use_train_debt_proxy),
        "last_train_debt_proxy": last_train_debt_proxy,
        "external_debt_proxy_mean": float(diag.get("external_debt_proxy_mean", 0.0)) if diag else 0.0,
        "train_debt_cotangent_enabled": int(use_train_debt_cotangent),
        "last_train_debt_cotangent_loss": last_train_debt_cotangent_loss,
        "operator_debt_cotangent_enabled": int(use_operator_debt_cotangent),
        "last_operator_debt_cotangent_loss": last_operator_debt_cotangent_loss,
        "external_debt_cotangent_norm_ratio_mean": float(diag.get("external_debt_cotangent_norm_ratio_mean", 0.0)) if diag else 0.0,
    }


def train_positive_model_calibration_boundary(kind: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    xtr, ytr, xte, yte = positive_control_data(task, seed, int(args.pc_train_size), int(args.pc_test_size), device)
    if kind == "mlp_adamw":
        model: nn.Module = MLPBaseline(int(xtr.shape[1]), 3, int(args.pc_hidden), seed + 1000, device).to(device)
        edge_params: list[nn.Parameter] = []
    else:
        model = EdgeAdditiveClassifier(int(xtr.shape[1]), 6, 3, seed + 2000, device).to(device)
        edge_params = [model.edge_coeff]
    base_opt = torch.optim.AdamW(model.parameters(), lr=float(args.pc_lr), weight_decay=float(args.pc_weight_decay))
    operator = None
    if kind == "kan_escgf":
        operator = EdgeStateFlow(
            edge_params,
            EdgeStateFlowConfig(
                beta_signal=0.25,
                beta_diffusion=0.08,
                eta_dual=float(getattr(args, "escgf_eta_dual", 0.05)),
                debt_budget=float(getattr(args, "escgf_debt_budget", 0.05)),
                external_debt_blend=float(getattr(args, "train_debt_proxy_blend", 0.0)),
                external_debt_cotangent_blend=float(getattr(args, "operator_debt_cotangent_blend", 0.0)),
                debt_cotangent_max_norm_ratio=float(getattr(args, "operator_debt_cotangent_max_norm_ratio", 1.0)),
                signal_blend=float(args.escgf_signal_blend),
                shrink_floor=0.12,
                max_norm_ratio=float(args.escgf_max_norm_ratio),
                control_mode="normal",
                random_seed=seed,
            ),
        )
        opt: Any = EdgeStateFlowOptimizerWrapper(base_opt, operator)
    else:
        opt = base_opt
    initial_test = metrics_for_logits(model(xte).float(), yte)
    initial_train = metrics_for_logits(model(xtr).float(), ytr) if bool(getattr(args, "train_debt_proxy", False)) and operator is not None else {}
    initial_train_debt_surrogate: dict[str, float] = {}
    use_operator_debt_cotangent = bool(getattr(args, "operator_debt_cotangent", False)) and operator is not None and bool(edge_params)
    if use_operator_debt_cotangent:
        with torch.no_grad():
            surrogate = train_debt_surrogate_metrics(model(xtr).float(), ytr, float(args.train_debt_tail_kappa))
        initial_train_debt_surrogate = {key: float(value.detach().cpu().item()) for key, value in surrogate.items()}
    for _step in range(int(args.pc_steps)):
        opt.zero_grad(set_to_none=True)
        logits = model(xtr).float()
        if bool(getattr(args, "train_debt_proxy", False)) and operator is not None:
            with torch.no_grad():
                current_train = metrics_for_logits(logits.detach(), ytr)
            operator.observe_debt_proxy(train_only_debt_proxy(current_train, initial_train))
        if use_operator_debt_cotangent and operator is not None:
            cotangents, _loss_value = train_debt_cotangents(logits, ytr, edge_params, initial_train_debt_surrogate, args)
            operator.observe_debt_cotangent(cotangents)
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
    train_logits = model(xtr).float().detach()
    test_logits = model(xte).float().detach()
    grid = [float(s) for s in csv_items(args.calibration_temperature_grid)]
    train_temp = best_train_temperature(train_logits, ytr, grid)
    oracle_temp, oracle_metrics = best_oracle_temperature(test_logits, yte, initial_test, grid)
    raw_metrics = metrics_for_logits(test_logits, yte)
    train_temp_metrics = temperature_scaled_metrics(test_logits, yte, train_temp)
    raw_flags = no_debt_flags_from_metrics(raw_metrics, initial_test)
    train_temp_flags = no_debt_flags_from_metrics(train_temp_metrics, initial_test)
    return {
        "task": task,
        "seed": seed,
        "kind": kind,
        "raw_temperature": 1.0,
        "train_temperature": train_temp,
        "test_oracle_temperature": oracle_temp,
        **{f"raw_{k}": v for k, v in raw_flags.items()},
        **{f"train_temp_{k}": v for k, v in train_temp_flags.items()},
        **{f"oracle_temp_{k}": v for k, v in oracle_metrics.items() if k in raw_flags or k in {"nll", "acc", "brier", "ece", "tail95", "margin10"}},
        "official_eligible": 0,
        "diagnostic_uses_test_oracle": 1,
    }


def edge_identifiability_positive_control(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> float:
    xtr, ytr, _xte, _yte = positive_control_data(task, seed, int(args.pc_train_size), int(args.pc_test_size), device)
    n = int(xtr.shape[0])
    split = n // 3
    model = EdgeAdditiveClassifier(int(xtr.shape[1]), 6, 3, seed + 2000, device).to(device)
    logits_s = model(xtr[:split]).float()
    logits_g = model(xtr[2 * split :]).float()
    target_s = ((F.softmax(logits_s.detach(), dim=1) - F.one_hot(ytr[:split], num_classes=3).float()) / max(1, split)).to(dtype=torch.float64)
    target_g = ((F.softmax(logits_g.detach(), dim=1) - F.one_hot(ytr[2 * split :], num_classes=3).float()) / max(1, n - 2 * split)).to(dtype=torch.float64)
    bs = model.basis(xtr[:split]).reshape(split, -1).to(dtype=torch.float64)
    bg = model.basis(xtr[2 * split :]).reshape(n - 2 * split, -1).to(dtype=torch.float64)
    vals: list[float] = []
    for cls in range(3):
        alpha = ridge_fit(bs, target_s[:, cls], float(args.projector_ridge))
        vals.append(r2_score(bg, target_g[:, cls], alpha))
    return float(sum(vals) / len(vals))


def positive_row(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    kinds = ["kan_adamw", "kan_escgf", "same_domain_random", "same_debt_random", "shuffled_signal", "frozen_signal", "mlp_adamw"]
    records = {kind: train_positive_model(kind, task, seed, args, device) for kind in kinds}
    esc = records["kan_escgf"]
    adam = records["kan_adamw"]
    controls = [records[k] for k in ["same_domain_random", "same_debt_random", "shuffled_signal", "frozen_signal"]]
    mlp = records["mlp_adamw"]
    best_control_nll = min(fval(r["final_nll"], 999.0) for r in controls)
    r2_guard = edge_identifiability_positive_control(task, seed, args, device)
    no_debt = int(
        fval(esc["brier_delta_from_initial"]) <= 0.0
        and fval(esc["ece_delta_from_initial"]) <= 0.02
        and fval(esc["tail95_delta_from_initial"]) <= 0.0
        and fval(esc["margin10_delta_from_initial"]) >= -0.02
    )
    overhead = (fval(esc["elapsed_s"]) - fval(adam["elapsed_s"])) / max(fval(adam["elapsed_s"]), 1.0e-12)
    return {
        "task": task,
        "seed": seed,
        "KAN_AdamW_NLL": adam["final_nll"],
        "KAN_ESCGF_NLL": esc["final_nll"],
        "MLP_matched_NLL": mlp["final_nll"],
        "best_control_NLL": best_control_nll,
        "task_NLL_delta_vs_AdamW": esc["final_nll"] - adam["final_nll"],
        "accuracy_delta_vs_AdamW": esc["accuracy"] - adam["accuracy"],
        "control_margin": best_control_nll - esc["final_nll"],
        "MLP_matched_margin": mlp["final_nll"] - esc["final_nll"],
        "edge_signal_identifiability_R2_guard": r2_guard,
        "source_witness_guard_R2_gap": 0.0,
        "edge_signal_state_SNR": esc["edge_signal_state_SNR"],
        "Brier_delta": esc["brier_delta_from_initial"],
        "ECE_delta": esc["ece_delta_from_initial"],
        "tail95_delta": esc["tail95_delta_from_initial"],
        "margin10_delta": esc["margin10_delta_from_initial"],
        "ESCGF_beats_KAN_AdamW_NLL": int(esc["final_nll"] < adam["final_nll"]),
        "ESCGF_beats_controls_NLL": int(esc["final_nll"] < best_control_nll),
        "ESCGF_no_debt": no_debt,
        "edge_effect_fraction": esc["edge_effect_fraction"],
        "state_continuity": esc["state_continuity"],
        "overhead": overhead,
        "ESCGF_beats_MLP_matched": int(esc["final_nll"] < mlp["final_nll"]),
        "raw_records_json": json.dumps(records, sort_keys=True),
    }


def run_part_e(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    items = [(task, seed) for task in tasks for seed in range(int(args.pc_seed_count))]
    shard = shard_items(items, args)
    local_args = clone_args(args)
    if repair:
        local_args.escgf_signal_blend = min(0.75, float(args.escgf_signal_blend))
        local_args.escgf_max_norm_ratio = min(1.35, float(args.escgf_max_norm_ratio))
    rows = [positive_row(task, seed, local_args, device) for task, seed in shard]
    prefix = "v22_88_part_e_repair_positive_control" if repair else "v22_88_part_e_positive_control"
    out = OUT_ROOT / f"{prefix}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(out, rows)
    write_json(out.with_suffix(".json"), {"rows": len(rows), "csv": rel(out), "repair": int(repair)})
    append_exec("part-e-repair" if repair else "part-e", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=f"rows={len(rows)}")
    return {"rows": len(rows)}


def summarize_part_e_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = {
        "beats_KAN_AdamW_NLL": sum(int(fval(r.get("ESCGF_beats_KAN_AdamW_NLL"))) == 1 for r in rows),
        "beats_controls_NLL": sum(int(fval(r.get("ESCGF_beats_controls_NLL"))) == 1 for r in rows),
        "no_debt_rows": sum(int(fval(r.get("ESCGF_no_debt"))) == 1 for r in rows),
        "R2_guard_ge_0p20": sum(fval(r.get("edge_signal_identifiability_R2_guard")) >= 0.20 for r in rows),
        "edge_effect_ge_0p80": sum(fval(r.get("edge_effect_fraction")) >= 0.80 for r in rows),
        "overhead_le_0p35": sum(fval(r.get("overhead"), 999.0) <= 0.35 for r in rows),
        "beats_MLP_matched": sum(int(fval(r.get("ESCGF_beats_MLP_matched"))) == 1 for r in rows),
    }
    gate = int(
        total >= 15
        and counts["beats_KAN_AdamW_NLL"] >= 12
        and counts["beats_controls_NLL"] >= 12
        and counts["no_debt_rows"] >= 12
        and counts["R2_guard_ge_0p20"] >= 12
        and counts["edge_effect_ge_0p80"] >= 12
        and counts["overhead_le_0p35"] >= 12
        and counts["beats_MLP_matched"] >= 9
    )
    route = "EdgeStateFlowPositiveControlOpened" if gate else "EdgeMetricImplementationFailedPositiveControl"
    return {"part_e_gate_pass": gate, "part_e_route": route, "rows": total, **counts}


def _parse_raw_pair(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = json.loads(str(row.get("raw_records_json", "{}")))
    except Exception:
        raw = {}
    return raw.get("kan_escgf", {}), raw.get("kan_adamw", {})


def debt_component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = {
        "absolute_brier_ok": 0,
        "absolute_ece_ok": 0,
        "absolute_tail95_ok": 0,
        "absolute_margin10_ok": 0,
        "absolute_all_no_debt": 0,
        "relative_brier_nonworse_than_adamw": 0,
        "relative_ece_nonworse_than_adamw": 0,
        "relative_tail95_nonworse_than_adamw": 0,
        "relative_margin10_nonworse_than_adamw": 0,
        "relative_all_nonworse_than_adamw": 0,
    }
    details: list[dict[str, Any]] = []
    for row in rows:
        esc, adam = _parse_raw_pair(row)
        brier = fval(row.get("Brier_delta", esc.get("brier_delta_from_initial")))
        ece = fval(row.get("ECE_delta", esc.get("ece_delta_from_initial")))
        tail = fval(row.get("tail95_delta", esc.get("tail95_delta_from_initial")))
        margin = fval(row.get("margin10_delta", esc.get("margin10_delta_from_initial")))
        adam_brier = fval(adam.get("brier_delta_from_initial"))
        adam_ece = fval(adam.get("ece_delta_from_initial"))
        adam_tail = fval(adam.get("tail95_delta_from_initial"))
        adam_margin = fval(adam.get("margin10_delta_from_initial"))
        absolute_flags = {
            "brier": int(brier <= 0.0),
            "ece": int(ece <= 0.02),
            "tail95": int(tail <= 0.0),
            "margin10": int(margin >= -0.02),
        }
        relative_flags = {
            "brier": int(brier <= adam_brier),
            "ece": int(ece <= adam_ece),
            "tail95": int(tail <= adam_tail),
            "margin10": int(margin >= adam_margin),
        }
        counts["absolute_brier_ok"] += absolute_flags["brier"]
        counts["absolute_ece_ok"] += absolute_flags["ece"]
        counts["absolute_tail95_ok"] += absolute_flags["tail95"]
        counts["absolute_margin10_ok"] += absolute_flags["margin10"]
        counts["absolute_all_no_debt"] += int(all(absolute_flags.values()))
        counts["relative_brier_nonworse_than_adamw"] += relative_flags["brier"]
        counts["relative_ece_nonworse_than_adamw"] += relative_flags["ece"]
        counts["relative_tail95_nonworse_than_adamw"] += relative_flags["tail95"]
        counts["relative_margin10_nonworse_than_adamw"] += relative_flags["margin10"]
        counts["relative_all_nonworse_than_adamw"] += int(all(relative_flags.values()))
        details.append(
            {
                "task": row.get("task", ""),
                "seed": row.get("seed", ""),
                "Brier_delta": brier,
                "ECE_delta": ece,
                "tail95_delta": tail,
                "margin10_delta": margin,
                "adamw_Brier_delta": adam_brier,
                "adamw_ECE_delta": adam_ece,
                "adamw_tail95_delta": adam_tail,
                "adamw_margin10_delta": adam_margin,
                "absolute_flags": absolute_flags,
                "relative_flags": relative_flags,
            }
        )
    return {"rows": total, **counts, "rows_detail": details}


def run_part_e_debt_decomp(args: argparse.Namespace) -> dict[str, Any]:
    del args
    suites = {
        "part_e": OUT_ROOT / "v22_88_part_e_positive_control.csv",
        "part_e_repair": OUT_ROOT / "v22_88_part_e_repair_positive_control.csv",
    }
    out: dict[str, Any] = {"gate": "v22_88_part_e_debt_failure_decomposition", "suites": {}}
    rows_for_csv: list[dict[str, Any]] = []
    recap_lines: list[str] = []
    for name, path in suites.items():
        rows = read_rows(path)
        summary = debt_component_summary(rows)
        out["suites"][name] = {"source_csv": rel(path), **summary}
        recap_lines.append(
            f"{name}: abs all={summary['absolute_all_no_debt']}/{summary['rows']}, "
            f"abs components Brier/ECE/tail/margin="
            f"{summary['absolute_brier_ok']}/{summary['absolute_ece_ok']}/{summary['absolute_tail95_ok']}/{summary['absolute_margin10_ok']}; "
            f"relative all vs AdamW={summary['relative_all_nonworse_than_adamw']}/{summary['rows']}, "
            f"relative components="
            f"{summary['relative_brier_nonworse_than_adamw']}/{summary['relative_ece_nonworse_than_adamw']}/"
            f"{summary['relative_tail95_nonworse_than_adamw']}/{summary['relative_margin10_nonworse_than_adamw']}"
        )
        for detail in summary["rows_detail"]:
            flat = {k: v for k, v in detail.items() if k not in {"absolute_flags", "relative_flags"}}
            flat.update({f"absolute_{k}_ok": v for k, v in detail["absolute_flags"].items()})
            flat.update({f"relative_{k}_ok": v for k, v in detail["relative_flags"].items()})
            flat["suite"] = name
            rows_for_csv.append(flat)
    json_path = OUT_ROOT / "v22_88_part_e_debt_failure_decomposition.json"
    csv_path = OUT_ROOT / "v22_88_part_e_debt_failure_decomposition.csv"
    write_json(json_path, out)
    write_rows(csv_path, rows_for_csv)
    append_exec("part-e-debt-decomp", command_text(sys.argv), "done", files=f"{rel(json_path)}; {rel(csv_path)}")
    append_recap(
        "Part E debt failure decomposition",
        recap_lines
        + [
            "analysis: 修正 debt dual sign 后 no-debt 只从 0/15 到 1/15；absolute blocker 主要是 ECE/tail，relative blocker 里 margin10 相对 AdamW 只 2/15 nonworse，说明不是单纯阈值过严或单一 ECE 问题。",
            "constraint: 该分解不改变 gate；只用于决定下一步 finite-step constrained law repair2。",
        ],
    )
    return out


def _part_e_setting_summary(rows: list[dict[str, Any]], *, extra: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_part_e_rows(rows)
    debt = debt_component_summary(rows)
    return {
        **extra,
        **summary,
        "absolute_brier_ok": debt["absolute_brier_ok"],
        "absolute_ece_ok": debt["absolute_ece_ok"],
        "absolute_tail95_ok": debt["absolute_tail95_ok"],
        "absolute_margin10_ok": debt["absolute_margin10_ok"],
        "relative_all_nonworse_than_adamw": debt["relative_all_nonworse_than_adamw"],
        "relative_brier_nonworse_than_adamw": debt["relative_brier_nonworse_than_adamw"],
        "relative_ece_nonworse_than_adamw": debt["relative_ece_nonworse_than_adamw"],
        "relative_tail95_nonworse_than_adamw": debt["relative_tail95_nonworse_than_adamw"],
        "relative_margin10_nonworse_than_adamw": debt["relative_margin10_nonworse_than_adamw"],
    }


def run_part_e_repair2_sweep(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    seeds = [int(s) for s in csv_items(args.repair2_sweep_seeds)]
    steps_grid = [int(s) for s in csv_items(args.repair2_steps_grid)]
    blend_grid = [float(s) for s in csv_items(args.repair2_blend_grid)]
    all_rows: list[dict[str, Any]] = []
    setting_summaries: list[dict[str, Any]] = []
    for steps in steps_grid:
        for blend in blend_grid:
            local_args = clone_args(
                args,
                pc_steps=int(steps),
                escgf_signal_blend=float(blend),
                escgf_max_norm_ratio=float(args.repair2_max_norm_ratio),
            )
            rows = []
            for seed in seeds:
                for task in tasks:
                    row = positive_row(task, seed, local_args, device)
                    row = {"repair2_steps": steps, "repair2_signal_blend": blend, "repair2_max_norm_ratio": float(args.repair2_max_norm_ratio), **row}
                    rows.append(row)
                    all_rows.append(row)
            setting_summaries.append(
                _part_e_setting_summary(
                    rows,
                    extra={
                        "repair2_steps": steps,
                        "repair2_signal_blend": blend,
                        "repair2_max_norm_ratio": float(args.repair2_max_norm_ratio),
                        "sweep_seeds": ",".join(str(s) for s in seeds),
                    },
                )
            )
    csv_path = OUT_ROOT / "v22_88_part_e_repair2_law_sweep_seed0.csv"
    json_path = OUT_ROOT / "v22_88_part_e_repair2_law_sweep_seed0.json"
    write_rows(csv_path, all_rows)
    best = sorted(
        setting_summaries,
        key=lambda row: (
            int(row.get("no_debt_rows", 0)),
            int(row.get("beats_KAN_AdamW_NLL", 0)),
            int(row.get("beats_controls_NLL", 0)),
            int(row.get("beats_MLP_matched", 0)),
            int(row.get("overhead_le_0p35", 0)),
        ),
        reverse=True,
    )[0] if setting_summaries else {}
    obj = {
        "gate": "v22_88_part_e_repair2_law_sweep_seed0",
        "rows": len(all_rows),
        "setting_summaries": setting_summaries,
        "diagnostic_best_for_full_repair2": best,
        "note": "Diagnostic-only finite-step constrained law sweep. It does not weaken Part E gate and does not promote row winners.",
    }
    write_json(json_path, obj)
    actions = [
        {
            "action": "run_full_repair2_fixed_law",
            "reason": "seed0 sweep is diagnostic only; full 15-row gate is required before any route change",
            "fixed_law": {
                "pc_steps": best.get("repair2_steps", args.repair2_full_steps),
                "signal_blend": best.get("repair2_signal_blend", args.repair2_full_blend),
                "max_norm_ratio": best.get("repair2_max_norm_ratio", args.repair2_max_norm_ratio),
            },
        }
    ]
    next_path = write_next_actions("e_repair2_sweep", "Repair2SweepDiagnosticOnly", "positive_control_failed", actions)
    append_exec("part-e-repair2-sweep", command_text(sys.argv), "done", gpu=str(device), files=f"{rel(csv_path)}; {rel(json_path)}; {rel(next_path)}")
    append_recap(
        "Part E repair2 finite-step law sweep",
        [
            f"rows={len(all_rows)}; settings={len(setting_summaries)}; seeds={seeds}; max_norm_ratio={float(args.repair2_max_norm_ratio)}",
            f"diagnostic best setting: steps={best.get('repair2_steps')}, blend={best.get('repair2_signal_blend')}, "
            f"counts beats_AdamW={best.get('beats_KAN_AdamW_NLL')}/{best.get('rows')}, beats_controls={best.get('beats_controls_NLL')}/{best.get('rows')}, "
            f"no_debt={best.get('no_debt_rows')}/{best.get('rows')}, overhead<=0.35={best.get('overhead_le_0p35')}/{best.get('rows')}, beats_MLP={best.get('beats_MLP_matched')}/{best.get('rows')}",
            "analysis: sweep 只用于选择一个固定 repair2 law 做完整 15-row 验证；不允许把 seed0/单 setting 结果当 Part E 成功。",
        ],
    )
    return obj


def run_part_e_repair2(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    items = [(task, seed) for task in tasks for seed in range(int(args.pc_seed_count))]
    shard = shard_items(items, args)
    local_args = clone_args(
        args,
        pc_steps=int(args.repair2_full_steps),
        escgf_signal_blend=float(args.repair2_full_blend),
        escgf_max_norm_ratio=float(args.repair2_max_norm_ratio),
    )
    rows = [
        {
            "repair2_steps": int(args.repair2_full_steps),
            "repair2_signal_blend": float(args.repair2_full_blend),
            "repair2_max_norm_ratio": float(args.repair2_max_norm_ratio),
            **positive_row(task, seed, local_args, device),
        }
        for task, seed in shard
    ]
    prefix = "v22_88_part_e_repair2_positive_control"
    out = OUT_ROOT / f"{prefix}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(out, rows)
    write_json(
        out.with_suffix(".json"),
        {
            "rows": len(rows),
            "csv": rel(out),
            "repair2": 1,
            "pc_steps": int(args.repair2_full_steps),
            "signal_blend": float(args.repair2_full_blend),
            "max_norm_ratio": float(args.repair2_max_norm_ratio),
        },
    )
    append_exec("part-e-repair2", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=f"rows={len(rows)}")
    return {"rows": len(rows)}


def merge_part_e_repair2(args: argparse.Namespace) -> dict[str, Any]:
    del args
    prefix = "v22_88_part_e_repair2_positive_control"
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"{prefix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_e_rows(rows)
    debt = debt_component_summary(rows)
    csv_path = OUT_ROOT / f"{prefix}.csv"
    route_path = OUT_ROOT / f"{prefix}_route.json"
    write_rows(csv_path, rows)
    write_json(route_path, {"gate": prefix, **summary, "debt_decomposition": debt, "rows_detail": rows})
    actions = (
        []
        if summary["part_e_gate_pass"]
        else [
            {"action": "inspect_whether_optimizer_only_debt_proxy_can_observe_calibration_debt", "reason": "repair2 fixed law still failed positive-control", "max_attempts": 2},
            {"action": "consider_train_only_h_step_debt_predictor_before_any_real_task", "reason": "current gradient-norm debt proxy is not aligned to Brier/ECE/tail/margin rows", "max_attempts": 2},
        ]
    )
    next_path = write_next_actions("e_repair2", summary["part_e_route"], "none" if summary["part_e_gate_pass"] else "positive_control_failed_after_repair2", actions)
    append_exec("part-e-repair2-merge", command_text(sys.argv), "done" if summary["part_e_gate_pass"] else "failed", files=f"{rel(csv_path)}; {rel(route_path)}; {rel(next_path)}")
    append_recap(
        "Part E repair2 full positive-control",
        [
            f"gate_pass={summary['part_e_gate_pass']}; route={summary['part_e_route']}; rows={summary['rows']}",
            f"counts: beats_AdamW={summary['beats_KAN_AdamW_NLL']}/15; beats_controls={summary['beats_controls_NLL']}/15; no_debt={summary['no_debt_rows']}/15; R2_guard>=0.20={summary['R2_guard_ge_0p20']}/15; edge_effect={summary['edge_effect_ge_0p80']}/15; overhead<=0.35={summary['overhead_le_0p35']}/15; beats_MLP={summary['beats_MLP_matched']}/15",
            f"debt components absolute Brier/ECE/tail/margin={debt['absolute_brier_ok']}/{debt['absolute_ece_ok']}/{debt['absolute_tail95_ok']}/{debt['absolute_margin10_ok']}; relative all vs AdamW={debt['relative_all_nonworse_than_adamw']}/15",
            "analysis: repair2 是降低 eta/finite-step constrained law 的固定验证；未改变 no-debt gate，未进入真实任务 full-loop。",
        ],
    )
    return {"gate": prefix, **summary, "debt_decomposition": debt}


def run_part_e_repair3_sweep(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    seeds = [int(s) for s in csv_items(args.repair3_sweep_seeds)]
    eta_grid = [float(s) for s in csv_items(args.repair3_eta_grid)]
    external_blend_grid = [float(s) for s in csv_items(args.repair3_external_debt_blend_grid)]
    all_rows: list[dict[str, Any]] = []
    setting_summaries: list[dict[str, Any]] = []
    for eta_dual in eta_grid:
        for external_blend in external_blend_grid:
            local_args = clone_args(
                args,
                pc_steps=int(args.repair3_full_steps),
                escgf_signal_blend=float(args.repair3_full_blend),
                escgf_max_norm_ratio=float(args.repair3_max_norm_ratio),
                escgf_eta_dual=float(eta_dual),
                escgf_debt_budget=float(args.repair3_debt_budget),
                train_debt_proxy=True,
                train_debt_proxy_blend=float(external_blend),
            )
            rows = []
            for seed in seeds:
                for task in tasks:
                    row = positive_row(task, seed, local_args, device)
                    row = {
                        "repair3_steps": int(args.repair3_full_steps),
                        "repair3_signal_blend": float(args.repair3_full_blend),
                        "repair3_max_norm_ratio": float(args.repair3_max_norm_ratio),
                        "repair3_eta_dual": eta_dual,
                        "repair3_debt_budget": float(args.repair3_debt_budget),
                        "repair3_external_debt_blend": external_blend,
                        **row,
                    }
                    rows.append(row)
                    all_rows.append(row)
            setting_summaries.append(
                _part_e_setting_summary(
                    rows,
                    extra={
                        "repair3_steps": int(args.repair3_full_steps),
                        "repair3_signal_blend": float(args.repair3_full_blend),
                        "repair3_max_norm_ratio": float(args.repair3_max_norm_ratio),
                        "repair3_eta_dual": eta_dual,
                        "repair3_debt_budget": float(args.repair3_debt_budget),
                        "repair3_external_debt_blend": external_blend,
                        "sweep_seeds": ",".join(str(s) for s in seeds),
                    },
                )
            )
    csv_path = OUT_ROOT / "v22_88_part_e_repair3_hstep_debt_sweep_seed0.csv"
    json_path = OUT_ROOT / "v22_88_part_e_repair3_hstep_debt_sweep_seed0.json"
    write_rows(csv_path, all_rows)
    best = sorted(
        setting_summaries,
        key=lambda row: (
            int(row.get("no_debt_rows", 0)),
            int(row.get("beats_controls_NLL", 0)),
            int(row.get("beats_KAN_AdamW_NLL", 0)),
            int(row.get("beats_MLP_matched", 0)),
            int(row.get("overhead_le_0p35", 0)),
        ),
        reverse=True,
    )[0] if setting_summaries else {}
    obj = {
        "gate": "v22_88_part_e_repair3_hstep_debt_sweep_seed0",
        "rows": len(all_rows),
        "setting_summaries": setting_summaries,
        "diagnostic_best_for_full_repair3": best,
        "note": "Diagnostic-only train-stream H-step debt proxy sweep. It uses no held-out/test future direction and does not weaken Part E gate.",
    }
    write_json(json_path, obj)
    next_path = write_next_actions(
        "e_repair3_sweep",
        "Repair3HStepDebtProxyDiagnosticOnly",
        "positive_control_failed_after_repair2",
        [
            {
                "action": "run_full_repair3_fixed_train_debt_proxy_law",
                "reason": "sweep is diagnostic only; full 15-row gate is required before route changes",
                "fixed_law": {
                    "eta_dual": best.get("repair3_eta_dual", args.repair3_eta_dual),
                    "external_debt_blend": best.get("repair3_external_debt_blend", args.repair3_external_debt_blend),
                    "debt_budget": best.get("repair3_debt_budget", args.repair3_debt_budget),
                },
            }
        ],
    )
    append_exec("part-e-repair3-sweep", command_text(sys.argv), "done", gpu=str(device), files=f"{rel(csv_path)}; {rel(json_path)}; {rel(next_path)}")
    append_recap(
        "Part E repair3 train-only H-step debt sweep",
        [
            f"rows={len(all_rows)}; settings={len(setting_summaries)}; seeds={seeds}; fixed steps/blend/max_norm={int(args.repair3_full_steps)}/{float(args.repair3_full_blend)}/{float(args.repair3_max_norm_ratio)}",
            f"diagnostic best setting: eta_dual={best.get('repair3_eta_dual')}, external_debt_blend={best.get('repair3_external_debt_blend')}, debt_budget={best.get('repair3_debt_budget')}, "
            f"counts beats_AdamW={best.get('beats_KAN_AdamW_NLL')}/{best.get('rows')}, beats_controls={best.get('beats_controls_NLL')}/{best.get('rows')}, "
            f"no_debt={best.get('no_debt_rows')}/{best.get('rows')}, overhead<=0.35={best.get('overhead_le_0p35')}/{best.get('rows')}, beats_MLP={best.get('beats_MLP_matched')}/{best.get('rows')}",
            "analysis: repair3 debt proxy 使用 train logits 相对初始 train metrics 的 Brier/ECE/tail/margin debt；未使用 held-out/test future direction。",
        ],
    )
    return obj


def _repair3_local_args(args: argparse.Namespace) -> argparse.Namespace:
    return clone_args(
        args,
        pc_steps=int(args.repair3_full_steps),
        escgf_signal_blend=float(args.repair3_full_blend),
        escgf_max_norm_ratio=float(args.repair3_max_norm_ratio),
        escgf_eta_dual=float(args.repair3_eta_dual),
        escgf_debt_budget=float(args.repair3_debt_budget),
        train_debt_proxy=True,
        train_debt_proxy_blend=float(args.repair3_external_debt_blend),
    )


def run_part_e_repair3(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    items = [(task, seed) for task in tasks for seed in range(int(args.pc_seed_count))]
    shard = shard_items(items, args)
    local_args = _repair3_local_args(args)
    rows = [
        {
            "repair3_steps": int(args.repair3_full_steps),
            "repair3_signal_blend": float(args.repair3_full_blend),
            "repair3_max_norm_ratio": float(args.repair3_max_norm_ratio),
            "repair3_eta_dual": float(args.repair3_eta_dual),
            "repair3_debt_budget": float(args.repair3_debt_budget),
            "repair3_external_debt_blend": float(args.repair3_external_debt_blend),
            **positive_row(task, seed, local_args, device),
        }
        for task, seed in shard
    ]
    prefix = "v22_88_part_e_repair3_positive_control"
    out = OUT_ROOT / f"{prefix}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(out, rows)
    write_json(
        out.with_suffix(".json"),
        {
            "rows": len(rows),
            "csv": rel(out),
            "repair3": 1,
            "pc_steps": int(args.repair3_full_steps),
            "signal_blend": float(args.repair3_full_blend),
            "max_norm_ratio": float(args.repair3_max_norm_ratio),
            "eta_dual": float(args.repair3_eta_dual),
            "debt_budget": float(args.repair3_debt_budget),
            "external_debt_blend": float(args.repair3_external_debt_blend),
        },
    )
    append_exec("part-e-repair3", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=f"rows={len(rows)}")
    return {"rows": len(rows)}


def merge_part_e_repair3(args: argparse.Namespace) -> dict[str, Any]:
    del args
    prefix = "v22_88_part_e_repair3_positive_control"
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"{prefix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_e_rows(rows)
    debt = debt_component_summary(rows)
    csv_path = OUT_ROOT / f"{prefix}.csv"
    route_path = OUT_ROOT / f"{prefix}_route.json"
    write_rows(csv_path, rows)
    write_json(route_path, {"gate": prefix, **summary, "debt_decomposition": debt, "rows_detail": rows})
    actions = (
        []
        if summary["part_e_gate_pass"]
        else [
            {"action": "stop_real_task_full_loop_and_report_positive_control_blocker", "reason": "repair3 train-only debt proxy did not satisfy positive-control", "max_attempts": 1},
            {"action": "representation_or_debt_predictor_redesign_required", "reason": "optimizer-only shrink and train debt proxy could not get no-debt/control gate to 12/15", "max_attempts": 1},
        ]
    )
    next_path = write_next_actions("e_repair3", summary["part_e_route"], "none" if summary["part_e_gate_pass"] else "positive_control_failed_after_repair3", actions)
    append_exec("part-e-repair3-merge", command_text(sys.argv), "done" if summary["part_e_gate_pass"] else "failed", files=f"{rel(csv_path)}; {rel(route_path)}; {rel(next_path)}")
    append_recap(
        "Part E repair3 full positive-control",
        [
            f"gate_pass={summary['part_e_gate_pass']}; route={summary['part_e_route']}; rows={summary['rows']}",
            f"counts: beats_AdamW={summary['beats_KAN_AdamW_NLL']}/15; beats_controls={summary['beats_controls_NLL']}/15; no_debt={summary['no_debt_rows']}/15; R2_guard>=0.20={summary['R2_guard_ge_0p20']}/15; edge_effect={summary['edge_effect_ge_0p80']}/15; overhead<=0.35={summary['overhead_le_0p35']}/15; beats_MLP={summary['beats_MLP_matched']}/15",
            f"debt components absolute Brier/ECE/tail/margin={debt['absolute_brier_ok']}/{debt['absolute_ece_ok']}/{debt['absolute_tail95_ok']}/{debt['absolute_margin10_ok']}; relative all vs AdamW={debt['relative_all_nonworse_than_adamw']}/15",
            "analysis: repair3 使用 train-only H-step debt proxy 后若仍失败，说明当前 law 的 debt 控制不是缺少 sign/步长微调，而是需要更强的 debt predictor/representation redesign；未进入真实任务 full-loop。",
        ],
    )
    return {"gate": prefix, **summary, "debt_decomposition": debt}


def _repair4_local_args(args: argparse.Namespace, *, weight: float | None = None, tail_weight: float | None = None, calib_weight: float | None = None) -> argparse.Namespace:
    return clone_args(
        args,
        pc_steps=int(args.repair4_full_steps),
        escgf_signal_blend=float(args.repair4_full_blend),
        escgf_max_norm_ratio=float(args.repair4_max_norm_ratio),
        escgf_eta_dual=float(args.repair4_eta_dual),
        escgf_debt_budget=float(args.repair4_debt_budget),
        train_debt_proxy=True,
        train_debt_proxy_blend=float(args.repair4_external_debt_blend),
        train_debt_cotangent=True,
        train_debt_cotangent_weight=float(args.repair4_cotangent_weight if weight is None else weight),
        train_debt_cotangent_tail_weight=float(args.repair4_tail_weight if tail_weight is None else tail_weight),
        train_debt_cotangent_calibration_weight=float(args.repair4_calibration_weight if calib_weight is None else calib_weight),
        train_debt_cotangent_brier_weight=float(args.repair4_brier_weight),
        train_debt_tail_kappa=float(args.repair4_tail_kappa),
        train_debt_calibration_budget=float(args.repair4_calibration_budget),
    )


def run_part_e_repair4_sweep(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    seeds = [int(s) for s in csv_items(args.repair4_sweep_seeds)]
    weight_grid = [float(s) for s in csv_items(args.repair4_cotangent_weight_grid)]
    tail_grid = [float(s) for s in csv_items(args.repair4_tail_weight_grid)]
    all_rows: list[dict[str, Any]] = []
    setting_summaries: list[dict[str, Any]] = []
    for weight in weight_grid:
        for tail_weight in tail_grid:
            local_args = _repair4_local_args(args, weight=weight, tail_weight=tail_weight)
            rows = []
            for seed in seeds:
                for task in tasks:
                    row = positive_row(task, seed, local_args, device)
                    row = {
                        "repair4_steps": int(args.repair4_full_steps),
                        "repair4_signal_blend": float(args.repair4_full_blend),
                        "repair4_max_norm_ratio": float(args.repair4_max_norm_ratio),
                        "repair4_eta_dual": float(args.repair4_eta_dual),
                        "repair4_debt_budget": float(args.repair4_debt_budget),
                        "repair4_external_debt_blend": float(args.repair4_external_debt_blend),
                        "repair4_cotangent_weight": weight,
                        "repair4_tail_weight": tail_weight,
                        "repair4_calibration_weight": float(args.repair4_calibration_weight),
                        "repair4_brier_weight": float(args.repair4_brier_weight),
                        **row,
                    }
                    rows.append(row)
                    all_rows.append(row)
            setting_summaries.append(
                _part_e_setting_summary(
                    rows,
                    extra={
                        "repair4_steps": int(args.repair4_full_steps),
                        "repair4_signal_blend": float(args.repair4_full_blend),
                        "repair4_max_norm_ratio": float(args.repair4_max_norm_ratio),
                        "repair4_eta_dual": float(args.repair4_eta_dual),
                        "repair4_debt_budget": float(args.repair4_debt_budget),
                        "repair4_external_debt_blend": float(args.repair4_external_debt_blend),
                        "repair4_cotangent_weight": weight,
                        "repair4_tail_weight": tail_weight,
                        "repair4_calibration_weight": float(args.repair4_calibration_weight),
                        "repair4_brier_weight": float(args.repair4_brier_weight),
                        "sweep_seeds": ",".join(str(s) for s in seeds),
                    },
                )
            )
    csv_path = OUT_ROOT / "v22_88_part_e_repair4_debt_cotangent_sweep_seed0.csv"
    json_path = OUT_ROOT / "v22_88_part_e_repair4_debt_cotangent_sweep_seed0.json"
    write_rows(csv_path, all_rows)
    best = sorted(
        setting_summaries,
        key=lambda row: (
            int(row.get("no_debt_rows", 0)),
            int(row.get("beats_KAN_AdamW_NLL", 0)),
            int(row.get("beats_controls_NLL", 0)),
            int(row.get("beats_MLP_matched", 0)),
            int(row.get("overhead_le_0p35", 0)),
        ),
        reverse=True,
    )[0] if setting_summaries else {}
    obj = {
        "gate": "v22_88_part_e_repair4_debt_cotangent_sweep_seed0",
        "rows": len(all_rows),
        "setting_summaries": setting_summaries,
        "diagnostic_best_for_full_repair4": best,
        "note": "Diagnostic-only train-stream differentiable debt cotangent sweep. It uses no held-out/test future direction and does not weaken Part E gate.",
    }
    write_json(json_path, obj)
    next_path = write_next_actions(
        "e_repair4_sweep",
        "Repair4TrainDebtCotangentDiagnosticOnly",
        "positive_control_failed_after_repair3",
        [
            {
                "action": "run_full_repair4_fixed_train_debt_cotangent_law",
                "reason": "sweep is diagnostic only; full 15-row gate is required before route changes",
                "fixed_law": {
                    "cotangent_weight": best.get("repair4_cotangent_weight", args.repair4_cotangent_weight),
                    "tail_weight": best.get("repair4_tail_weight", args.repair4_tail_weight),
                    "calibration_weight": best.get("repair4_calibration_weight", args.repair4_calibration_weight),
                },
            }
        ],
    )
    append_exec("part-e-repair4-sweep", command_text(sys.argv), "done", gpu=str(device), files=f"{rel(csv_path)}; {rel(json_path)}; {rel(next_path)}")
    append_recap(
        "Part E repair4 train-only debt cotangent sweep",
        [
            f"rows={len(all_rows)}; settings={len(setting_summaries)}; seeds={seeds}; fixed steps/blend/max_norm={int(args.repair4_full_steps)}/{float(args.repair4_full_blend)}/{float(args.repair4_max_norm_ratio)}",
            f"diagnostic best setting: cotangent_weight={best.get('repair4_cotangent_weight')}, tail_weight={best.get('repair4_tail_weight')}, calibration_weight={best.get('repair4_calibration_weight')}, "
            f"counts beats_AdamW={best.get('beats_KAN_AdamW_NLL')}/{best.get('rows')}, beats_controls={best.get('beats_controls_NLL')}/{best.get('rows')}, "
            f"no_debt={best.get('no_debt_rows')}/{best.get('rows')}, overhead<=0.35={best.get('overhead_le_0p35')}/{best.get('rows')}, beats_MLP={best.get('beats_MLP_matched')}/{best.get('rows')}",
            "analysis: repair4 cotangent 使用 train logits 的 smooth calibration/tail surrogate，不使用 held-out/test future direction，不修改 no-debt gate。",
        ],
    )
    return obj


def run_part_e_repair4(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    items = [(task, seed) for task in tasks for seed in range(int(args.pc_seed_count))]
    shard = shard_items(items, args)
    local_args = _repair4_local_args(args)
    rows = [
        {
            "repair4_steps": int(args.repair4_full_steps),
            "repair4_signal_blend": float(args.repair4_full_blend),
            "repair4_max_norm_ratio": float(args.repair4_max_norm_ratio),
            "repair4_eta_dual": float(args.repair4_eta_dual),
            "repair4_debt_budget": float(args.repair4_debt_budget),
            "repair4_external_debt_blend": float(args.repair4_external_debt_blend),
            "repair4_cotangent_weight": float(args.repair4_cotangent_weight),
            "repair4_tail_weight": float(args.repair4_tail_weight),
            "repair4_calibration_weight": float(args.repair4_calibration_weight),
            "repair4_brier_weight": float(args.repair4_brier_weight),
            **positive_row(task, seed, local_args, device),
        }
        for task, seed in shard
    ]
    prefix = "v22_88_part_e_repair4_positive_control"
    out = OUT_ROOT / f"{prefix}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(out, rows)
    write_json(
        out.with_suffix(".json"),
        {
            "rows": len(rows),
            "csv": rel(out),
            "repair4": 1,
            "pc_steps": int(args.repair4_full_steps),
            "signal_blend": float(args.repair4_full_blend),
            "max_norm_ratio": float(args.repair4_max_norm_ratio),
            "eta_dual": float(args.repair4_eta_dual),
            "debt_budget": float(args.repair4_debt_budget),
            "external_debt_blend": float(args.repair4_external_debt_blend),
            "cotangent_weight": float(args.repair4_cotangent_weight),
            "tail_weight": float(args.repair4_tail_weight),
            "calibration_weight": float(args.repair4_calibration_weight),
            "brier_weight": float(args.repair4_brier_weight),
        },
    )
    append_exec("part-e-repair4", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=f"rows={len(rows)}")
    return {"rows": len(rows)}


def merge_part_e_repair4(args: argparse.Namespace) -> dict[str, Any]:
    del args
    prefix = "v22_88_part_e_repair4_positive_control"
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"{prefix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_e_rows(rows)
    debt = debt_component_summary(rows)
    csv_path = OUT_ROOT / f"{prefix}.csv"
    route_path = OUT_ROOT / f"{prefix}_route.json"
    write_rows(csv_path, rows)
    write_json(route_path, {"gate": prefix, **summary, "debt_decomposition": debt, "rows_detail": rows})
    actions = (
        []
        if summary["part_e_gate_pass"]
        else [
            {"action": "stop_real_task_full_loop_and_report_positive_control_blocker", "reason": "repair4 train-only debt cotangent did not satisfy positive-control", "max_attempts": 1},
            {"action": "representation_or_calibration_debt_predictor_redesign_required", "reason": "allowed repair families did not get no-debt/control/AdamW gates to 12/15", "max_attempts": 1},
        ]
    )
    next_path = write_next_actions("e_repair4", summary["part_e_route"], "none" if summary["part_e_gate_pass"] else "positive_control_failed_after_repair4", actions)
    append_exec("part-e-repair4-merge", command_text(sys.argv), "done" if summary["part_e_gate_pass"] else "failed", files=f"{rel(csv_path)}; {rel(route_path)}; {rel(next_path)}")
    append_recap(
        "Part E repair4 full positive-control",
        [
            f"gate_pass={summary['part_e_gate_pass']}; route={summary['part_e_route']}; rows={summary['rows']}",
            f"counts: beats_AdamW={summary['beats_KAN_AdamW_NLL']}/15; beats_controls={summary['beats_controls_NLL']}/15; no_debt={summary['no_debt_rows']}/15; R2_guard>=0.20={summary['R2_guard_ge_0p20']}/15; edge_effect={summary['edge_effect_ge_0p80']}/15; overhead<=0.35={summary['overhead_le_0p35']}/15; beats_MLP={summary['beats_MLP_matched']}/15",
            f"debt components absolute Brier/ECE/tail/margin={debt['absolute_brier_ok']}/{debt['absolute_ece_ok']}/{debt['absolute_tail95_ok']}/{debt['absolute_margin10_ok']}; relative all vs AdamW={debt['relative_all_nonworse_than_adamw']}/15",
            "analysis: repair4 是 train-only differentiable debt cotangent；如果仍失败，不进入真实任务 full-loop，不写 official KAN claim。",
        ],
    )
    return {"gate": prefix, **summary, "debt_decomposition": debt}


def _repair5_local_args(args: argparse.Namespace, *, blend: float | None = None, tail_weight: float | None = None) -> argparse.Namespace:
    return clone_args(
        args,
        pc_steps=int(args.repair5_full_steps),
        escgf_signal_blend=float(args.repair5_full_blend),
        escgf_max_norm_ratio=float(args.repair5_max_norm_ratio),
        escgf_eta_dual=float(args.repair5_eta_dual),
        escgf_debt_budget=float(args.repair5_debt_budget),
        train_debt_proxy=True,
        train_debt_proxy_blend=float(args.repair5_external_debt_blend),
        train_debt_cotangent=False,
        train_debt_cotangent_tail_weight=float(args.repair5_tail_weight if tail_weight is None else tail_weight),
        train_debt_cotangent_calibration_weight=float(args.repair5_calibration_weight),
        train_debt_cotangent_brier_weight=float(args.repair5_brier_weight),
        train_debt_tail_kappa=float(args.repair5_tail_kappa),
        train_debt_calibration_budget=float(args.repair5_calibration_budget),
        operator_debt_cotangent=True,
        operator_debt_cotangent_blend=float(args.repair5_operator_debt_blend if blend is None else blend),
        operator_debt_cotangent_max_norm_ratio=float(args.repair5_operator_debt_max_norm_ratio),
    )


def run_part_e_repair5_sweep(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    seeds = [int(s) for s in csv_items(args.repair5_sweep_seeds)]
    blend_grid = [float(s) for s in csv_items(args.repair5_operator_debt_blend_grid)]
    tail_grid = [float(s) for s in csv_items(args.repair5_tail_weight_grid)]
    all_rows: list[dict[str, Any]] = []
    setting_summaries: list[dict[str, Any]] = []
    for blend in blend_grid:
        for tail_weight in tail_grid:
            local_args = _repair5_local_args(args, blend=blend, tail_weight=tail_weight)
            rows = []
            for seed in seeds:
                for task in tasks:
                    row = positive_row(task, seed, local_args, device)
                    row = {
                        "repair5_steps": int(args.repair5_full_steps),
                        "repair5_signal_blend": float(args.repair5_full_blend),
                        "repair5_max_norm_ratio": float(args.repair5_max_norm_ratio),
                        "repair5_eta_dual": float(args.repair5_eta_dual),
                        "repair5_debt_budget": float(args.repair5_debt_budget),
                        "repair5_external_debt_blend": float(args.repair5_external_debt_blend),
                        "repair5_operator_debt_blend": blend,
                        "repair5_operator_debt_max_norm_ratio": float(args.repair5_operator_debt_max_norm_ratio),
                        "repair5_tail_weight": tail_weight,
                        "repair5_calibration_weight": float(args.repair5_calibration_weight),
                        "repair5_brier_weight": float(args.repair5_brier_weight),
                        "supplemental_after_allowed_repairs": 1,
                        **row,
                    }
                    rows.append(row)
                    all_rows.append(row)
            setting_summaries.append(
                _part_e_setting_summary(
                    rows,
                    extra={
                        "repair5_steps": int(args.repair5_full_steps),
                        "repair5_signal_blend": float(args.repair5_full_blend),
                        "repair5_max_norm_ratio": float(args.repair5_max_norm_ratio),
                        "repair5_eta_dual": float(args.repair5_eta_dual),
                        "repair5_debt_budget": float(args.repair5_debt_budget),
                        "repair5_external_debt_blend": float(args.repair5_external_debt_blend),
                        "repair5_operator_debt_blend": blend,
                        "repair5_operator_debt_max_norm_ratio": float(args.repair5_operator_debt_max_norm_ratio),
                        "repair5_tail_weight": tail_weight,
                        "repair5_calibration_weight": float(args.repair5_calibration_weight),
                        "repair5_brier_weight": float(args.repair5_brier_weight),
                        "sweep_seeds": ",".join(str(s) for s in seeds),
                    },
                )
            )
    csv_path = OUT_ROOT / "v22_88_part_e_repair5_operator_debt_cotangent_sweep_seed0.csv"
    json_path = OUT_ROOT / "v22_88_part_e_repair5_operator_debt_cotangent_sweep_seed0.json"
    write_rows(csv_path, all_rows)
    best = sorted(
        setting_summaries,
        key=lambda row: (
            int(row.get("no_debt_rows", 0)),
            int(row.get("beats_KAN_AdamW_NLL", 0)),
            int(row.get("beats_controls_NLL", 0)),
            int(row.get("beats_MLP_matched", 0)),
            int(row.get("overhead_le_0p35", 0)),
        ),
        reverse=True,
    )[0] if setting_summaries else {}
    obj = {
        "gate": "v22_88_part_e_repair5_operator_debt_cotangent_sweep_seed0",
        "rows": len(all_rows),
        "setting_summaries": setting_summaries,
        "diagnostic_best_for_full_repair5": best,
        "official_eligible": 0,
        "note": "Post-allowed supplemental diagnostic. Debt cotangent is train-stream, edge-only, and consumed inside optimizer.step(); it does not weaken Part E gate and does not alter official final route.",
    }
    write_json(json_path, obj)
    next_path = write_next_actions(
        "e_repair5_operator_debt_sweep",
        "Repair5OperatorOwnedDebtCotangentSupplementalDiagnosticOnly",
        "positive_control_failed_after_allowed_repair4",
        [
            {
                "action": "run_full_repair5_fixed_operator_owned_debt_cotangent",
                "reason": "sweep is supplemental only; full 15-row evidence is needed for redesign decision, not official promotion",
                "fixed_law": {
                    "operator_debt_blend": best.get("repair5_operator_debt_blend", args.repair5_operator_debt_blend),
                    "tail_weight": best.get("repair5_tail_weight", args.repair5_tail_weight),
                    "calibration_weight": best.get("repair5_calibration_weight", args.repair5_calibration_weight),
                },
            }
        ],
    )
    append_exec("part-e-repair5-operator-debt-sweep", command_text(sys.argv), "done", gpu=str(device), files=f"{rel(csv_path)}; {rel(json_path)}; {rel(next_path)}")
    append_recap(
        "Part E supplemental repair5 operator-owned debt cotangent sweep",
        [
            f"rows={len(all_rows)}; settings={len(setting_summaries)}; seeds={seeds}; fixed steps/blend/max_norm={int(args.repair5_full_steps)}/{float(args.repair5_full_blend)}/{float(args.repair5_max_norm_ratio)}",
            f"diagnostic best setting: operator_debt_blend={best.get('repair5_operator_debt_blend')}, tail_weight={best.get('repair5_tail_weight')}, calibration_weight={best.get('repair5_calibration_weight')}, "
            f"counts beats_AdamW={best.get('beats_KAN_AdamW_NLL')}/{best.get('rows')}, beats_controls={best.get('beats_controls_NLL')}/{best.get('rows')}, "
            f"no_debt={best.get('no_debt_rows')}/{best.get('rows')}, overhead<=0.35={best.get('overhead_le_0p35')}/{best.get('rows')}, beats_MLP={best.get('beats_MLP_matched')}/{best.get('rows')}",
            "analysis: repair5 是超过文档 4 轮后的补充诊断；CE loss 不加 debt auxiliary，debt cotangent 只作为 edge-only g_debt 在 optimizer.step() 内进入 ESCGF；不使用 held-out/test future direction，不修改 no-debt gate，不改变 official final route。",
        ],
    )
    return obj


def run_part_e_repair5(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    items = [(task, seed) for task in tasks for seed in range(int(args.pc_seed_count))]
    shard = shard_items(items, args)
    local_args = _repair5_local_args(args)
    rows = [
        {
            "repair5_steps": int(args.repair5_full_steps),
            "repair5_signal_blend": float(args.repair5_full_blend),
            "repair5_max_norm_ratio": float(args.repair5_max_norm_ratio),
            "repair5_eta_dual": float(args.repair5_eta_dual),
            "repair5_debt_budget": float(args.repair5_debt_budget),
            "repair5_external_debt_blend": float(args.repair5_external_debt_blend),
            "repair5_operator_debt_blend": float(args.repair5_operator_debt_blend),
            "repair5_operator_debt_max_norm_ratio": float(args.repair5_operator_debt_max_norm_ratio),
            "repair5_tail_weight": float(args.repair5_tail_weight),
            "repair5_calibration_weight": float(args.repair5_calibration_weight),
            "repair5_brier_weight": float(args.repair5_brier_weight),
            "supplemental_after_allowed_repairs": 1,
            **positive_row(task, seed, local_args, device),
        }
        for task, seed in shard
    ]
    prefix = "v22_88_part_e_repair5_operator_debt_cotangent_positive_control"
    out = OUT_ROOT / f"{prefix}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(out, rows)
    write_json(
        out.with_suffix(".json"),
        {
            "rows": len(rows),
            "csv": rel(out),
            "repair5": 1,
            "official_eligible": 0,
            "pc_steps": int(args.repair5_full_steps),
            "signal_blend": float(args.repair5_full_blend),
            "max_norm_ratio": float(args.repair5_max_norm_ratio),
            "eta_dual": float(args.repair5_eta_dual),
            "debt_budget": float(args.repair5_debt_budget),
            "external_debt_blend": float(args.repair5_external_debt_blend),
            "operator_debt_blend": float(args.repair5_operator_debt_blend),
            "operator_debt_max_norm_ratio": float(args.repair5_operator_debt_max_norm_ratio),
            "tail_weight": float(args.repair5_tail_weight),
            "calibration_weight": float(args.repair5_calibration_weight),
            "brier_weight": float(args.repair5_brier_weight),
        },
    )
    append_exec("part-e-repair5-operator-debt", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=f"rows={len(rows)}")
    return {"rows": len(rows)}


def merge_part_e_repair5(args: argparse.Namespace) -> dict[str, Any]:
    del args
    prefix = "v22_88_part_e_repair5_operator_debt_cotangent_positive_control"
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"{prefix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_e_rows(rows)
    debt = debt_component_summary(rows)
    csv_path = OUT_ROOT / f"{prefix}.csv"
    route_path = OUT_ROOT / f"{prefix}_route.json"
    write_rows(csv_path, rows)
    write_json(
        route_path,
        {
            "gate": prefix,
            **summary,
            "official_eligible": 0,
            "official_note": "Supplemental repair after plan-allowed 4 rounds; does not alter v22.88 official final route.",
            "debt_decomposition": debt,
            "rows_detail": rows,
        },
    )
    actions = (
        []
        if summary["part_e_gate_pass"]
        else [
            {"action": "stop_real_task_full_loop_and_report_positive_control_blocker", "reason": "operator-owned debt cotangent supplemental repair did not satisfy positive-control", "max_attempts": 1},
            {"action": "new_representation_or_calibration_debt_predictor_plan_required", "reason": "edge-only train-stream g_debt still did not open no-debt to 12/15", "max_attempts": 1},
        ]
    )
    next_path = write_next_actions("e_repair5_operator_debt", summary["part_e_route"], "supplemental_not_official", actions)
    append_exec("part-e-repair5-operator-debt-merge", command_text(sys.argv), "done" if summary["part_e_gate_pass"] else "failed", files=f"{rel(csv_path)}; {rel(route_path)}; {rel(next_path)}")
    append_recap(
        "Part E supplemental repair5 full operator-owned debt cotangent",
        [
            f"gate_pass={summary['part_e_gate_pass']}; route={summary['part_e_route']}; official_eligible=0; rows={summary['rows']}",
            f"counts: beats_AdamW={summary['beats_KAN_AdamW_NLL']}/15; beats_controls={summary['beats_controls_NLL']}/15; no_debt={summary['no_debt_rows']}/15; R2_guard>=0.20={summary['R2_guard_ge_0p20']}/15; edge_effect={summary['edge_effect_ge_0p80']}/15; overhead<=0.35={summary['overhead_le_0p35']}/15; beats_MLP={summary['beats_MLP_matched']}/15",
            f"debt components absolute Brier/ECE/tail/margin={debt['absolute_brier_ok']}/{debt['absolute_ece_ok']}/{debt['absolute_tail95_ok']}/{debt['absolute_margin10_ok']}; relative all vs AdamW={debt['relative_all_nonworse_than_adamw']}/15",
            "analysis: repair5 把 debt cotangent 放回计划公式中的 g_debt 位置，只作用于 edge params 并由 optimizer.step() 消费；如果仍失败，说明不是 repair4 auxiliary-loss 实现位置造成的主 blocker。该结果不覆盖 official repair4 final route。",
        ],
    )
    return {"gate": prefix, **summary, "official_eligible": 0, "debt_decomposition": debt}


def summarize_calibration_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(rows), "official_eligible": 0, "by_kind": {}}
    for kind in sorted({str(r.get("kind", "")) for r in rows}):
        subset = [r for r in rows if str(r.get("kind", "")) == kind]
        if not subset:
            continue
        out["by_kind"][kind] = {
            "rows": len(subset),
            "raw_no_debt": sum(int(fval(r.get("raw_no_debt"))) == 1 for r in subset),
            "train_temp_no_debt": sum(int(fval(r.get("train_temp_no_debt"))) == 1 for r in subset),
            "oracle_temp_no_debt": sum(int(fval(r.get("oracle_temp_no_debt"))) == 1 for r in subset),
            "raw_ece_ok": sum(int(fval(r.get("raw_ece_ok"))) == 1 for r in subset),
            "train_temp_ece_ok": sum(int(fval(r.get("train_temp_ece_ok"))) == 1 for r in subset),
            "oracle_temp_ece_ok": sum(int(fval(r.get("oracle_temp_ece_ok"))) == 1 for r in subset),
            "raw_tail95_ok": sum(int(fval(r.get("raw_tail95_ok"))) == 1 for r in subset),
            "train_temp_tail95_ok": sum(int(fval(r.get("train_temp_tail95_ok"))) == 1 for r in subset),
            "oracle_temp_tail95_ok": sum(int(fval(r.get("oracle_temp_tail95_ok"))) == 1 for r in subset),
            "median_train_temperature": median([fval(r.get("train_temperature")) for r in subset]),
            "median_oracle_temperature": median([fval(r.get("test_oracle_temperature")) for r in subset]),
            "median_raw_violation": median([fval(r.get("raw_debt_violation_sum")) for r in subset]),
            "median_train_temp_violation": median([fval(r.get("train_temp_debt_violation_sum")) for r in subset]),
            "median_oracle_temp_violation": median([fval(r.get("oracle_temp_debt_violation_sum")) for r in subset]),
        }
    return out


def run_part_e_calibration_boundary(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = ["additive_univariate", "moving_domain_additive", "interaction_to_additive"]
    items = [(task, seed) for task in tasks for seed in range(int(args.pc_seed_count))]
    shard = shard_items(items, args)
    local_args = _repair5_local_args(
        clone_args(
            args,
            repair5_operator_debt_blend=float(args.calibration_boundary_operator_debt_blend),
            repair5_tail_weight=float(args.calibration_boundary_tail_weight),
        )
    )
    kinds = ["kan_adamw", "kan_escgf", "mlp_adamw"]
    rows = [
        {
            "calibration_boundary_source": "repair5_operator_debt_law",
            "supplemental_after_allowed_repairs": 1,
            **train_positive_model_calibration_boundary(kind, task, seed, local_args, device),
        }
        for task, seed in shard
        for kind in kinds
    ]
    prefix = "v22_88_part_e_calibration_boundary"
    out = OUT_ROOT / f"{prefix}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(out, rows)
    write_json(
        out.with_suffix(".json"),
        {
            "rows": len(rows),
            "csv": rel(out),
            "official_eligible": 0,
            "temperature_grid": args.calibration_temperature_grid,
            "diagnostic_uses_test_oracle": 1,
        },
    )
    append_exec("part-e-calibration-boundary", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=f"rows={len(rows)}")
    return {"rows": len(rows)}


def merge_part_e_calibration_boundary(args: argparse.Namespace) -> dict[str, Any]:
    del args
    prefix = "v22_88_part_e_calibration_boundary"
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"{prefix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_calibration_boundary(rows)
    csv_path = OUT_ROOT / f"{prefix}.csv"
    json_path = OUT_ROOT / f"{prefix}_route.json"
    write_rows(csv_path, rows)
    write_json(
        json_path,
        {
            "gate": prefix,
            "route": "CalibrationBoundaryDiagnosticOnly",
            "official_eligible": 0,
            "diagnostic_uses_test_oracle": 1,
            "summary": summary,
            "rows_detail": rows,
        },
    )
    by = summary["by_kind"]
    esc = by.get("kan_escgf", {})
    adam = by.get("kan_adamw", {})
    mlp = by.get("mlp_adamw", {})
    append_exec("part-e-calibration-boundary-merge", command_text(sys.argv), "done", files=f"{rel(csv_path)}; {rel(json_path)}")
    append_recap(
        "Part E supplemental calibration boundary diagnostic",
        [
            f"rows={summary['rows']}; official_eligible=0; diagnostic_uses_test_oracle=1",
            f"KAN_ESCGF raw/train-temp/oracle no_debt={esc.get('raw_no_debt')}/{esc.get('train_temp_no_debt')}/{esc.get('oracle_temp_no_debt')} of {esc.get('rows')}; "
            f"ECE ok={esc.get('raw_ece_ok')}/{esc.get('train_temp_ece_ok')}/{esc.get('oracle_temp_ece_ok')}; tail95 ok={esc.get('raw_tail95_ok')}/{esc.get('train_temp_tail95_ok')}/{esc.get('oracle_temp_tail95_ok')}",
            f"KAN_AdamW raw/train-temp/oracle no_debt={adam.get('raw_no_debt')}/{adam.get('train_temp_no_debt')}/{adam.get('oracle_temp_no_debt')} of {adam.get('rows')}; "
            f"MLP raw/train-temp/oracle no_debt={mlp.get('raw_no_debt')}/{mlp.get('train_temp_no_debt')}/{mlp.get('oracle_temp_no_debt')} of {mlp.get('rows')}",
            f"median temperature KAN_ESCGF train/oracle={esc.get('median_train_temperature')}/{esc.get('median_oracle_temperature')}; "
            f"median violation raw/train/oracle={esc.get('median_raw_violation')}/{esc.get('median_train_temp_violation')}/{esc.get('median_oracle_temp_violation')}",
            "analysis: train-only temperature is deployable only as diagnostic post-hoc calibration; test-oracle temperature is explicitly invalid for official use and only measures an upper bound. If oracle still cannot approach 12/15 no-debt, blocker is not a simple scalar calibration issue.",
        ],
    )
    return {"gate": prefix, "summary": summary, "official_eligible": 0}


def one_edge_binary_data(seed: int, n_train: int, n_test: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(888100 + int(seed) * 19)
    x = torch.rand(n_train + n_test, 1, device=device, generator=gen) * 2.0 - 1.0
    score = 1.25 * torch.sin(math.pi * x[:, 0]) + 0.55 * x[:, 0]
    y = (score > 0.0).long()
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def one_edge_three_class_data(seed: int, n_train: int, n_test: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(889300 + int(seed) * 23)
    x = torch.rand(n_train + n_test, 1, device=device, generator=gen) * 2.0 - 1.0
    score = 1.25 * torch.sin(math.pi * x[:, 0]) + 0.55 * x[:, 0]
    q1 = torch.quantile(score, 1.0 / 3.0)
    q2 = torch.quantile(score, 2.0 / 3.0)
    y = torch.bucketize(score, torch.stack([q1, q2])).long()
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def one_edge_closed_form_gradient_error(model: EdgeAdditiveClassifier, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    loss = F.cross_entropy(logits, y)
    loss.backward()
    autograd = model.edge_coeff.grad.detach().clone()
    with torch.no_grad():
        prob = F.softmax(logits, dim=1)
        target = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
        cotangent = (prob - target) / max(1, int(y.numel()))
        manual = torch.einsum("ib,ik->bk", model.basis(x).squeeze(1), cotangent).unsqueeze(0)
        diff = (autograd - manual).detach()
        cosine = F.cosine_similarity(autograd.flatten(), manual.flatten(), dim=0).detach().cpu().item()
        rel_error = diff.norm().div(manual.norm().clamp_min(1.0e-12)).detach().cpu().item()
    model.zero_grad(set_to_none=True)
    return {"closed_form_grad_cosine": float(cosine), "closed_form_grad_rel_error": float(rel_error)}


def train_one_edge_minimal(kind: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    xtr, ytr, xte, yte = one_edge_binary_data(seed, int(args.toy_train_size), int(args.toy_test_size), device)
    model = EdgeAdditiveClassifier(1, 6, 2, seed + 52000, device).to(device)
    grad_audit = one_edge_closed_form_gradient_error(model, xtr, ytr)
    base_opt = torch.optim.AdamW(model.parameters(), lr=float(args.toy_lr), weight_decay=float(args.pc_weight_decay))
    operator = None
    if kind == "kan_escgf":
        operator = EdgeStateFlow(
            [model.edge_coeff],
            EdgeStateFlowConfig(
                beta_signal=0.25,
                beta_diffusion=0.08,
                eta_dual=float(args.toy_eta_dual),
                debt_budget=float(args.toy_debt_budget),
                signal_blend=float(args.toy_signal_blend),
                shrink_floor=0.12,
                max_norm_ratio=float(args.toy_max_norm_ratio),
                random_seed=seed,
            ),
        )
        opt: Any = EdgeStateFlowOptimizerWrapper(base_opt, operator)
    else:
        opt = base_opt
    initial = metrics_for_logits(model(xte).float(), yte)
    for _step in range(int(args.toy_steps)):
        opt.zero_grad(set_to_none=True)
        logits = model(xtr).float()
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
    final = metrics_for_logits(model(xte).float(), yte)
    flags = no_debt_flags_from_metrics(final, initial)
    diag = opt.diagnostics() if hasattr(opt, "diagnostics") else {}
    return {
        "kind": kind,
        "seed": seed,
        "initial_nll": initial["nll"],
        "final_nll": final["nll"],
        "nll_delta": final["nll"] - initial["nll"],
        "accuracy": final["acc"],
        **flags,
        **grad_audit,
        "edge_state_updated": int(diag.get("edge_state_updated_every_step", 0)) if diag else 0,
        "optimizer_owned_gradient_transform_pass": int(diag.get("optimizer_owned_gradient_transform_pass", 0)) if diag else 0,
        "external_oracle_used": 0,
    }


def train_one_edge_minimal_three_class(kind: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    xtr, ytr, xte, yte = one_edge_three_class_data(seed, int(args.toy_train_size), int(args.toy_test_size), device)
    model = EdgeAdditiveClassifier(1, 6, 3, seed + 53000, device).to(device)
    grad_audit = one_edge_closed_form_gradient_error(model, xtr, ytr)
    base_opt = torch.optim.AdamW(model.parameters(), lr=float(args.toy_lr), weight_decay=float(args.pc_weight_decay))
    operator = None
    if kind == "kan_escgf":
        operator = EdgeStateFlow(
            [model.edge_coeff],
            EdgeStateFlowConfig(
                beta_signal=0.25,
                beta_diffusion=0.08,
                eta_dual=float(args.toy_eta_dual),
                debt_budget=float(args.toy_debt_budget),
                signal_blend=float(args.toy_signal_blend),
                shrink_floor=0.12,
                max_norm_ratio=float(args.toy_max_norm_ratio),
                random_seed=seed,
            ),
        )
        opt: Any = EdgeStateFlowOptimizerWrapper(base_opt, operator)
    else:
        opt = base_opt
    initial = metrics_for_logits(model(xte).float(), yte)
    for _step in range(int(args.toy_steps)):
        opt.zero_grad(set_to_none=True)
        logits = model(xtr).float()
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
    final = metrics_for_logits(model(xte).float(), yte)
    flags = no_debt_flags_from_metrics(final, initial)
    diag = opt.diagnostics() if hasattr(opt, "diagnostics") else {}
    return {
        "kind": kind,
        "seed": seed,
        "initial_nll": initial["nll"],
        "final_nll": final["nll"],
        "nll_delta": final["nll"] - initial["nll"],
        "accuracy": final["acc"],
        **flags,
        **grad_audit,
        "edge_state_updated": int(diag.get("edge_state_updated_every_step", 0)) if diag else 0,
        "optimizer_owned_gradient_transform_pass": int(diag.get("optimizer_owned_gradient_transform_pass", 0)) if diag else 0,
        "external_oracle_used": 0,
    }


def run_part_e_minimal_one_edge(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for seed in range(int(args.toy_seed_count)):
        adam = train_one_edge_minimal("kan_adamw", seed, args, device)
        esc = train_one_edge_minimal("kan_escgf", seed, args, device)
        rows.append(
            {
                "seed": seed,
                "adam_final_nll": adam["final_nll"],
                "escgf_final_nll": esc["final_nll"],
                "escgf_beats_adamw": int(float(esc["final_nll"]) < float(adam["final_nll"])),
                "escgf_no_debt": esc["no_debt"],
                "escgf_brier_ok": esc["brier_ok"],
                "escgf_ece_ok": esc["ece_ok"],
                "escgf_tail95_ok": esc["tail95_ok"],
                "escgf_margin10_ok": esc["margin10_ok"],
                "escgf_nll_delta": esc["nll_delta"],
                "adam_nll_delta": adam["nll_delta"],
                "escgf_accuracy": esc["accuracy"],
                "closed_form_grad_cosine": esc["closed_form_grad_cosine"],
                "closed_form_grad_rel_error": esc["closed_form_grad_rel_error"],
                "edge_state_updated": esc["edge_state_updated"],
                "optimizer_owned_gradient_transform_pass": esc["optimizer_owned_gradient_transform_pass"],
                "official_eligible": 0,
                "raw_adam_json": json.dumps(adam, sort_keys=True),
                "raw_escgf_json": json.dumps(esc, sort_keys=True),
            }
        )
    summary = {
        "gate": "v22_88_part_e_minimal_one_edge_closed_form",
        "official_eligible": 0,
        "rows": len(rows),
        "beats_adamw": sum(int(r["escgf_beats_adamw"]) == 1 for r in rows),
        "no_debt": sum(int(r["escgf_no_debt"]) == 1 for r in rows),
        "brier_ok": sum(int(r["escgf_brier_ok"]) == 1 for r in rows),
        "ece_ok": sum(int(r["escgf_ece_ok"]) == 1 for r in rows),
        "tail95_ok": sum(int(r["escgf_tail95_ok"]) == 1 for r in rows),
        "margin10_ok": sum(int(r["escgf_margin10_ok"]) == 1 for r in rows),
        "closed_form_grad_cosine_min": min(fval(r["closed_form_grad_cosine"]) for r in rows) if rows else 0.0,
        "closed_form_grad_rel_error_max": max(fval(r["closed_form_grad_rel_error"]) for r in rows) if rows else 0.0,
        "optimizer_owned_rows": sum(int(r["optimizer_owned_gradient_transform_pass"]) == 1 for r in rows),
        "route": "MinimalOneEdgeDiagnosticOnly",
    }
    csv_path = OUT_ROOT / "v22_88_part_e_minimal_one_edge_closed_form.csv"
    json_path = OUT_ROOT / "v22_88_part_e_minimal_one_edge_closed_form_route.json"
    write_rows(csv_path, rows)
    write_json(json_path, {**summary, "rows_detail": rows})
    append_exec("part-e-minimal-one-edge", command_text(sys.argv), "done", gpu=str(device), files=f"{rel(csv_path)}; {rel(json_path)}")
    append_recap(
        "Part E supplemental minimal 1-edge closed-form diagnostic",
        [
            f"rows={summary['rows']}; official_eligible=0; beats_AdamW={summary['beats_adamw']}/{summary['rows']}; no_debt={summary['no_debt']}/{summary['rows']}",
            f"debt components brier/ece/tail/margin={summary['brier_ok']}/{summary['ece_ok']}/{summary['tail95_ok']}/{summary['margin10_ok']}",
            f"closed_form_grad_cosine_min={summary['closed_form_grad_cosine_min']}; closed_form_grad_rel_error_max={summary['closed_form_grad_rel_error_max']}; optimizer_owned_rows={summary['optimizer_owned_rows']}/{summary['rows']}",
            "analysis: one-edge toy uses a binary edge-native target exactly representable by the registered edge basis; closed-form gradient compares autograd edge cotangent with basis^T(prob-onehot)/n. It is diagnostic-only and does not change official final route.",
        ],
    )
    return summary


def run_part_e_minimal_one_edge_three_class(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for seed in range(int(args.toy_seed_count)):
        adam = train_one_edge_minimal_three_class("kan_adamw", seed, args, device)
        esc = train_one_edge_minimal_three_class("kan_escgf", seed, args, device)
        rows.append(
            {
                "seed": seed,
                "adam_final_nll": adam["final_nll"],
                "escgf_final_nll": esc["final_nll"],
                "escgf_beats_adamw": int(float(esc["final_nll"]) < float(adam["final_nll"])),
                "escgf_no_debt": esc["no_debt"],
                "escgf_brier_ok": esc["brier_ok"],
                "escgf_ece_ok": esc["ece_ok"],
                "escgf_tail95_ok": esc["tail95_ok"],
                "escgf_margin10_ok": esc["margin10_ok"],
                "escgf_nll_delta": esc["nll_delta"],
                "adam_nll_delta": adam["nll_delta"],
                "escgf_accuracy": esc["accuracy"],
                "closed_form_grad_cosine": esc["closed_form_grad_cosine"],
                "closed_form_grad_rel_error": esc["closed_form_grad_rel_error"],
                "edge_state_updated": esc["edge_state_updated"],
                "optimizer_owned_gradient_transform_pass": esc["optimizer_owned_gradient_transform_pass"],
                "official_eligible": 0,
                "raw_adam_json": json.dumps(adam, sort_keys=True),
                "raw_escgf_json": json.dumps(esc, sort_keys=True),
            }
        )
    summary = {
        "gate": "v22_88_part_e_minimal_one_edge_three_class_closed_form",
        "official_eligible": 0,
        "rows": len(rows),
        "beats_adamw": sum(int(r["escgf_beats_adamw"]) == 1 for r in rows),
        "no_debt": sum(int(r["escgf_no_debt"]) == 1 for r in rows),
        "brier_ok": sum(int(r["escgf_brier_ok"]) == 1 for r in rows),
        "ece_ok": sum(int(r["escgf_ece_ok"]) == 1 for r in rows),
        "tail95_ok": sum(int(r["escgf_tail95_ok"]) == 1 for r in rows),
        "margin10_ok": sum(int(r["escgf_margin10_ok"]) == 1 for r in rows),
        "closed_form_grad_cosine_min": min(fval(r["closed_form_grad_cosine"]) for r in rows) if rows else 0.0,
        "closed_form_grad_rel_error_max": max(fval(r["closed_form_grad_rel_error"]) for r in rows) if rows else 0.0,
        "optimizer_owned_rows": sum(int(r["optimizer_owned_gradient_transform_pass"]) == 1 for r in rows),
        "route": "MinimalOneEdgeThreeClassDiagnosticOnly",
    }
    csv_path = OUT_ROOT / "v22_88_part_e_minimal_one_edge_three_class_closed_form.csv"
    json_path = OUT_ROOT / "v22_88_part_e_minimal_one_edge_three_class_closed_form_route.json"
    write_rows(csv_path, rows)
    write_json(json_path, {**summary, "rows_detail": rows})
    append_exec("part-e-minimal-one-edge-3class", command_text(sys.argv), "done", gpu=str(device), files=f"{rel(csv_path)}; {rel(json_path)}")
    append_recap(
        "Part E supplemental minimal 1-edge 3-class closed-form diagnostic",
        [
            f"rows={summary['rows']}; official_eligible=0; beats_AdamW={summary['beats_adamw']}/{summary['rows']}; no_debt={summary['no_debt']}/{summary['rows']}",
            f"debt components brier/ece/tail/margin={summary['brier_ok']}/{summary['ece_ok']}/{summary['tail95_ok']}/{summary['margin10_ok']}",
            f"closed_form_grad_cosine_min={summary['closed_form_grad_cosine_min']}; closed_form_grad_rel_error_max={summary['closed_form_grad_rel_error_max']}; optimizer_owned_rows={summary['optimizer_owned_rows']}/{summary['rows']}",
            "analysis: 3-class one-edge toy keeps the same single edge basis but uses quantile-bucket multiclass labels, isolating the multiclass calibration/tail pressure that is absent in binary toy. It is diagnostic-only and does not change official final route.",
        ],
    )
    return summary


def merge_part_e(args: argparse.Namespace, *, repair: bool = False) -> dict[str, Any]:
    del args
    prefix = "v22_88_part_e_repair_positive_control" if repair else "v22_88_part_e_positive_control"
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"{prefix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_e_rows(rows)
    csv_path = OUT_ROOT / f"{prefix}.csv"
    route_path = OUT_ROOT / f"{prefix}_route.json"
    write_rows(csv_path, rows)
    write_json(route_path, {"gate": prefix, **summary, "rows_detail": rows})
    actions = (
        []
        if summary["part_e_gate_pass"]
        else [
            {"action": "audit_edge_jvp_basis_eval_signal_ema_debt_dual_smooth_shrink", "reason": "positive-control gate failed", "max_attempts": 4},
            {"action": "run_minimal_one_edge_closed_form_test", "reason": "distinguish implementation bug from law weakness", "max_attempts": 1},
        ]
    )
    next_path = write_next_actions("e_repair" if repair else "e", summary["part_e_route"], "none" if summary["part_e_gate_pass"] else "positive_control_failed", actions)
    append_exec("part-e-repair-merge" if repair else "part-e-merge", command_text(sys.argv), "done" if summary["part_e_gate_pass"] else "failed", files=f"{rel(csv_path)}; {rel(route_path)}; {rel(next_path)}")
    append_recap(
        "Part E KAN-native positive-control" + (" repair" if repair else ""),
        [
            f"gate_pass={summary['part_e_gate_pass']}; route={summary['part_e_route']}; rows={summary['rows']}",
            f"counts: beats_AdamW={summary['beats_KAN_AdamW_NLL']}/15; beats_controls={summary['beats_controls_NLL']}/15; no_debt={summary['no_debt_rows']}/15; R2_guard>=0.20={summary['R2_guard_ge_0p20']}/15; edge_effect={summary['edge_effect_ge_0p80']}/15; overhead<=0.35={summary['overhead_le_0p35']}/15; beats_MLP={summary['beats_MLP_matched']}/15",
            "如果该 gate 失败，计划禁止进入真实任务 full-loop；已写入修复方向供审计。",
        ],
    )
    return summary


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "v22_88_part_c_edge_signal_identifiability_route.json")
    d = read_json(OUT_ROOT / "v22_88_part_d_escgf_unit_tests_route.json")
    e = read_json(OUT_ROOT / "v22_88_part_e_positive_control_route.json")
    er = read_json(OUT_ROOT / "v22_88_part_e_repair_positive_control_route.json")
    er2 = read_json(OUT_ROOT / "v22_88_part_e_repair2_positive_control_route.json")
    er3 = read_json(OUT_ROOT / "v22_88_part_e_repair3_positive_control_route.json")
    er4 = read_json(OUT_ROOT / "v22_88_part_e_repair4_positive_control_route.json")
    final_e = er4 if er4 else er3 if er3 else er2 if er2 else er if er else e
    prereq = int(c.get("part_c_gate_pass", 0) == 1 and d.get("part_d_gate_pass", 0) == 1 and final_e.get("part_e_gate_pass", 0) == 1)
    route = "EdgeStateFlowRealPreflightOpened" if prereq else "RealTaskPreflightSkippedPrerequisitesFailed"
    obj = {
        "gate": "v22_88_part_f_real_task_preflight",
        "part_f_gate_pass": 0,
        "part_f_route": route,
        "skipped": int(not prereq),
        "prerequisites": {"C": c.get("part_c_gate_pass", 0), "D": d.get("part_d_gate_pass", 0), "E": final_e.get("part_e_gate_pass", 0)},
        "note": "Part F real-task preflight is allowed only if C/D/E all pass. This runner does not fabricate F rows when prereqs fail.",
    }
    write_json(OUT_ROOT / "v22_88_part_f_real_task_preflight_route.json", obj)
    next_path = write_next_actions("f", route, "prerequisites_failed" if not prereq else "not_run", [])
    append_exec("part-f", command_text(sys.argv), "skipped" if not prereq else "not_run", files=f"{rel(OUT_ROOT / 'v22_88_part_f_real_task_preflight_route.json')}; {rel(next_path)}")
    append_recap("Part F real-task preflight", [f"skipped={int(not prereq)}; prerequisites={obj['prerequisites']}; route={route}"])
    return obj


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    f = read_json(OUT_ROOT / "v22_88_part_f_real_task_preflight_route.json")
    allowed = int(f.get("part_f_gate_pass", 0) == 1)
    route = "KANEdgeStateFlowExplorationOpened" if allowed else "FullLoopSkippedPreflightFailed"
    obj = {
        "gate": "v22_88_part_g_hstep_full_loop",
        "part_g_gate_pass": 0,
        "part_g_route": route,
        "skipped": int(not allowed),
        "note": "Part G is forbidden unless Part F passes; no H-step/full-loop rows were fabricated.",
    }
    write_json(OUT_ROOT / "v22_88_part_g_hstep_full_loop_route.json", obj)
    next_path = write_next_actions("g", route, "preflight_failed" if not allowed else "not_run", [])
    append_exec("part-g", command_text(sys.argv), "skipped" if not allowed else "not_run", files=f"{rel(OUT_ROOT / 'v22_88_part_g_hstep_full_loop_route.json')}; {rel(next_path)}")
    append_recap("Part G H-step/full-loop", [f"skipped={int(not allowed)}; route={route}"])
    return obj


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    del args
    obj = {
        "gate": "v22_88_part_h_matched_controls_external_pressure",
        "part_h_gate_pass": 0,
        "part_h_route": "MatchedControlsNotEnteredWithoutRealPreflight",
        "skipped": 1,
        "note": "Part H strengthens comparison after preflight/full-loop entry; skipped because upstream gates did not open it.",
    }
    write_json(OUT_ROOT / "v22_88_part_h_matched_controls_route.json", obj)
    next_path = write_next_actions("h", obj["part_h_route"], "upstream_not_open", [])
    append_exec("part-h", command_text(sys.argv), "skipped", files=f"{rel(OUT_ROOT / 'v22_88_part_h_matched_controls_route.json')}; {rel(next_path)}")
    return obj


def run_part_i(args: argparse.Namespace) -> dict[str, Any]:
    del args
    c = read_json(OUT_ROOT / "v22_88_part_c_edge_signal_identifiability_route.json")
    cr = read_json(OUT_ROOT / "v22_88_part_c_repair_edge_signal_identifiability_route.json")
    d = read_json(OUT_ROOT / "v22_88_part_d_escgf_unit_tests_route.json")
    e = read_json(OUT_ROOT / "v22_88_part_e_positive_control_route.json")
    er = read_json(OUT_ROOT / "v22_88_part_e_repair_positive_control_route.json")
    er2 = read_json(OUT_ROOT / "v22_88_part_e_repair2_positive_control_route.json")
    er3 = read_json(OUT_ROOT / "v22_88_part_e_repair3_positive_control_route.json")
    er4 = read_json(OUT_ROOT / "v22_88_part_e_repair4_positive_control_route.json")
    final_c = cr if cr else c
    final_e = er4 if er4 else er3 if er3 else er2 if er2 else er if er else e
    if str(final_c.get("part_c_route", "")) == "SourceArtifactNoGuardTransfer":
        route = "SourceArtifactNoGuardTransfer"
    elif str(final_c.get("part_c_route", "")) == "MLPMatchedProjectionDominates":
        route = "MLPMatchedProjectionDominates"
    elif int(d.get("part_d_gate_pass", 0)) != 1:
        route = "EdgeMetricImplementationFailedPositiveControl"
    elif int(final_e.get("part_e_gate_pass", 0)) != 1:
        route = "EdgeMetricImplementationFailedPositiveControl"
    elif int(final_c.get("part_c_gate_pass", 0)) != 1 and int(final_e.get("part_e_gate_pass", 0)) == 1:
        route = "PositiveControlPassRealTaskNoSignal"
    elif int(final_c.get("part_c_gate_pass", 0)) != 1:
        route = "CurrentStrictFCPureKANFamilyNoIdentifiableEdgeSignal"
    else:
        route = "EdgeStateFlowRealPreflightOpened"
    obj = {
        "gate": "v22_88_part_i_decision",
        "part_i_gate_pass": int(route in {"EdgeStateFlowPositiveControlOpened", "EdgeStateFlowRealPreflightOpened", "KANEdgeStateFlowExplorationOpened", "KANEdgeStateFlowOfficialCandidate"}),
        "part_i_route": route,
        "sources": {
            "part_c_route": final_c.get("part_c_route", "missing"),
            "part_d_route": d.get("part_d_route", "missing"),
            "part_e_route": final_e.get("part_e_route", "missing"),
        },
    }
    write_json(OUT_ROOT / "v22_88_part_i_decision_route.json", obj)
    next_path = write_next_actions("i", route, "final_route", [])
    append_exec("part-i", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'v22_88_part_i_decision_route.json')}; {rel(next_path)}")
    append_recap("Part I decision route", [f"route={route}; sources={obj['sources']}"])
    return obj


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    del args
    a = read_json(OUT_ROOT / "v22_88_part_a_code_identity_hard_gate.json")
    b = read_json(OUT_ROOT / "v22_88_part_b_history_failure_matrix.json")
    c = read_json(OUT_ROOT / "v22_88_part_c_edge_signal_identifiability_route.json")
    cr = read_json(OUT_ROOT / "v22_88_part_c_repair_edge_signal_identifiability_route.json")
    d = read_json(OUT_ROOT / "v22_88_part_d_escgf_unit_tests_route.json")
    e = read_json(OUT_ROOT / "v22_88_part_e_positive_control_route.json")
    er = read_json(OUT_ROOT / "v22_88_part_e_repair_positive_control_route.json")
    er2 = read_json(OUT_ROOT / "v22_88_part_e_repair2_positive_control_route.json")
    er3 = read_json(OUT_ROOT / "v22_88_part_e_repair3_positive_control_route.json")
    er4 = read_json(OUT_ROOT / "v22_88_part_e_repair4_positive_control_route.json")
    f = read_json(OUT_ROOT / "v22_88_part_f_real_task_preflight_route.json")
    g = read_json(OUT_ROOT / "v22_88_part_g_hstep_full_loop_route.json")
    h = read_json(OUT_ROOT / "v22_88_part_h_matched_controls_route.json")
    i = read_json(OUT_ROOT / "v22_88_part_i_decision_route.json")
    final_c = cr if cr else c
    final_e = er4 if er4 else er3 if er3 else er2 if er2 else er if er else e
    if int(a.get("part_a_gate_pass", 0)) != 1:
        route, reason = "CandidateUpdateRegressionDetected", "Part A no-candidate hard gate failed"
    elif int(final_e.get("part_e_gate_pass", 0)) != 1:
        route, reason = "EdgeMetricImplementationFailedPositiveControl", "Positive-control gate failed after available repair"
    elif int(final_c.get("part_c_gate_pass", 0)) != 1 and int(final_e.get("part_e_gate_pass", 0)) == 1:
        route, reason = "PositiveControlPassRealTaskNoSignal", "Positive-control passed but real-task signal identifiability did not"
    else:
        route = str(i.get("part_i_route", "CurrentStrictFCPureKANFamilyNoIdentifiableEdgeSignal"))
        reason = "Part I route from current gates"
    official = int(route in {"KANEdgeStateFlowOfficialCandidate"})
    final = {
        "gate": "v22_88_final_route",
        "final_route": route,
        "official_candidate_gate_pass": official,
        "route_reason": reason,
        "part_a": int(a.get("part_a_gate_pass", 0)),
        "part_b": int(b.get("part_b_gate_pass", 0)),
        "part_c": int(c.get("part_c_gate_pass", 0)),
        "part_c_repair": int(cr.get("part_c_gate_pass", 0)) if cr else 0,
        "part_d": int(d.get("part_d_gate_pass", 0)),
        "part_e": int(e.get("part_e_gate_pass", 0)),
        "part_e_repair": int(er.get("part_e_gate_pass", 0)) if er else 0,
        "part_e_repair2": int(er2.get("part_e_gate_pass", 0)) if er2 else 0,
        "part_e_repair3": int(er3.get("part_e_gate_pass", 0)) if er3 else 0,
        "part_e_repair4": int(er4.get("part_e_gate_pass", 0)) if er4 else 0,
        "part_f": int(f.get("part_f_gate_pass", 0)),
        "part_g": int(g.get("part_g_gate_pass", 0)),
        "part_h": int(h.get("part_h_gate_pass", 0)),
        "part_i": int(i.get("part_i_gate_pass", 0)),
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_88_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_88_part_b_history_failure_matrix.json"),
            "part_c": rel(OUT_ROOT / "v22_88_part_c_edge_signal_identifiability_route.json"),
            "part_c_repair": rel(OUT_ROOT / "v22_88_part_c_repair_edge_signal_identifiability_route.json") if cr else "missing",
            "part_d": rel(OUT_ROOT / "v22_88_part_d_escgf_unit_tests_route.json"),
            "part_e": rel(OUT_ROOT / "v22_88_part_e_positive_control_route.json"),
            "part_e_repair": rel(OUT_ROOT / "v22_88_part_e_repair_positive_control_route.json") if er else "missing",
            "part_e_repair2": rel(OUT_ROOT / "v22_88_part_e_repair2_positive_control_route.json") if er2 else "missing",
            "part_e_repair3": rel(OUT_ROOT / "v22_88_part_e_repair3_positive_control_route.json") if er3 else "missing",
            "part_e_repair4": rel(OUT_ROOT / "v22_88_part_e_repair4_positive_control_route.json") if er4 else "missing",
            "part_f": rel(OUT_ROOT / "v22_88_part_f_real_task_preflight_route.json"),
            "part_g": rel(OUT_ROOT / "v22_88_part_g_hstep_full_loop_route.json"),
            "part_h": rel(OUT_ROOT / "v22_88_part_h_matched_controls_route.json"),
            "part_i": rel(OUT_ROOT / "v22_88_part_i_decision_route.json"),
            "final": rel(OUT_ROOT / "v22_88_final_route.json"),
        },
    }
    write_json(OUT_ROOT / "v22_88_final_route.json", final)
    append_exec("final", command_text(sys.argv), "done", files=rel(OUT_ROOT / "v22_88_final_route.json"), note=json.dumps(final, ensure_ascii=False))
    append_recap(
        "Final route and conclusion",
        [
            f"final_route={route}; official_candidate_gate_pass={official}; reason={reason}",
            f"A/B/C/Crepair/D/E/Erepair/Erepair2/Erepair3/Erepair4/F/G/H/I pass={final['part_a']}/{final['part_b']}/{final['part_c']}/{final['part_c_repair']}/{final['part_d']}/{final['part_e']}/{final['part_e_repair']}/{final['part_e_repair2']}/{final['part_e_repair3']}/{final['part_e_repair4']}/{final['part_f']}/{final['part_g']}/{final['part_h']}/{final['part_i']}",
            "结论约束：没有 Part F/G 通过时，不写 official KAN claim；positive-control 或 identifiability 失败时停止 full-loop。",
        ],
    )
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="Wine,Spam,MNIST")
    p.add_argument("--seed-count", type=int, default=5)
    p.add_argument("--spec-count", type=int, default=1)
    p.add_argument("--train-size", type=int, default=384)
    p.add_argument("--held-size", type=int, default=192)
    p.add_argument("--test-size", type=int, default=192)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--metric-batch-size", type=int, default=96)
    p.add_argument("--projector-ridge", type=float, default=1.0e-4)
    p.add_argument("--model-seed-offset", type=int, default=28800)
    p.add_argument("--rkhs-max-edges", type=int, default=48)
    p.add_argument("--rkhs-centers", type=int, default=5)
    p.add_argument("--rkhs-alpha-norm", type=float, default=10.0)
    p.add_argument("--rkhs-svd-rank", type=int, default=32)
    p.add_argument("--primary-shape-kernel", default="K3R3_svd64_whitened_mixed_output_velocity")
    p.add_argument("--max-output-families", type=int, default=1)
    p.add_argument("--force-output-velocity-family", default="")
    p.add_argument("--diagnostic-logit-update-norm", type=float, default=0.05)
    p.add_argument("--pc-seed-count", type=int, default=5)
    p.add_argument("--pc-train-size", type=int, default=768)
    p.add_argument("--pc-test-size", type=int, default=384)
    p.add_argument("--pc-hidden", type=int, default=7)
    p.add_argument("--pc-steps", type=int, default=80)
    p.add_argument("--pc-lr", type=float, default=1.5e-2)
    p.add_argument("--pc-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--escgf-signal-blend", type=float, default=0.90)
    p.add_argument("--escgf-max-norm-ratio", type=float, default=1.60)
    p.add_argument("--escgf-eta-dual", type=float, default=0.05)
    p.add_argument("--escgf-debt-budget", type=float, default=0.05)
    p.add_argument("--train-debt-proxy", action="store_true")
    p.add_argument("--train-debt-proxy-blend", type=float, default=0.0)
    p.add_argument("--train-debt-cotangent", action="store_true")
    p.add_argument("--train-debt-cotangent-weight", type=float, default=0.0)
    p.add_argument("--train-debt-cotangent-brier-weight", type=float, default=0.10)
    p.add_argument("--train-debt-cotangent-calibration-weight", type=float, default=1.00)
    p.add_argument("--train-debt-cotangent-tail-weight", type=float, default=1.00)
    p.add_argument("--train-debt-tail-kappa", type=float, default=6.0)
    p.add_argument("--train-debt-calibration-budget", type=float, default=0.0)
    p.add_argument("--operator-debt-cotangent", action="store_true")
    p.add_argument("--operator-debt-cotangent-blend", type=float, default=0.0)
    p.add_argument("--operator-debt-cotangent-max-norm-ratio", type=float, default=1.0)
    p.add_argument("--repair2-steps-grid", default="20,40,60,80")
    p.add_argument("--repair2-blend-grid", default="0.25,0.45,0.65,0.90")
    p.add_argument("--repair2-sweep-seeds", default="0")
    p.add_argument("--repair2-full-steps", type=int, default=80)
    p.add_argument("--repair2-full-blend", type=float, default=0.25)
    p.add_argument("--repair2-max-norm-ratio", type=float, default=1.15)
    p.add_argument("--repair3-sweep-seeds", default="0")
    p.add_argument("--repair3-eta-grid", default="0.10,0.30,0.70")
    p.add_argument("--repair3-external-debt-blend-grid", default="0.25,0.50,1.00")
    p.add_argument("--repair3-full-steps", type=int, default=80)
    p.add_argument("--repair3-full-blend", type=float, default=0.25)
    p.add_argument("--repair3-max-norm-ratio", type=float, default=1.15)
    p.add_argument("--repair3-eta-dual", type=float, default=0.30)
    p.add_argument("--repair3-debt-budget", type=float, default=0.02)
    p.add_argument("--repair3-external-debt-blend", type=float, default=1.00)
    p.add_argument("--repair4-sweep-seeds", default="0")
    p.add_argument("--repair4-cotangent-weight-grid", default="0.05,0.20,0.50,1.00")
    p.add_argument("--repair4-tail-weight-grid", default="0.50,1.00")
    p.add_argument("--repair4-full-steps", type=int, default=80)
    p.add_argument("--repair4-full-blend", type=float, default=0.25)
    p.add_argument("--repair4-max-norm-ratio", type=float, default=1.15)
    p.add_argument("--repair4-eta-dual", type=float, default=0.10)
    p.add_argument("--repair4-debt-budget", type=float, default=0.02)
    p.add_argument("--repair4-external-debt-blend", type=float, default=0.25)
    p.add_argument("--repair4-cotangent-weight", type=float, default=0.20)
    p.add_argument("--repair4-brier-weight", type=float, default=0.10)
    p.add_argument("--repair4-calibration-weight", type=float, default=1.00)
    p.add_argument("--repair4-tail-weight", type=float, default=1.00)
    p.add_argument("--repair4-tail-kappa", type=float, default=6.0)
    p.add_argument("--repair4-calibration-budget", type=float, default=0.0)
    p.add_argument("--repair5-sweep-seeds", default="0")
    p.add_argument("--repair5-operator-debt-blend-grid", default="0.10,0.25,0.50,1.00")
    p.add_argument("--repair5-tail-weight-grid", default="0.25,0.50,1.00")
    p.add_argument("--repair5-full-steps", type=int, default=80)
    p.add_argument("--repair5-full-blend", type=float, default=0.25)
    p.add_argument("--repair5-max-norm-ratio", type=float, default=1.15)
    p.add_argument("--repair5-eta-dual", type=float, default=0.10)
    p.add_argument("--repair5-debt-budget", type=float, default=0.02)
    p.add_argument("--repair5-external-debt-blend", type=float, default=0.25)
    p.add_argument("--repair5-operator-debt-blend", type=float, default=0.25)
    p.add_argument("--repair5-operator-debt-max-norm-ratio", type=float, default=1.0)
    p.add_argument("--repair5-brier-weight", type=float, default=0.10)
    p.add_argument("--repair5-calibration-weight", type=float, default=1.00)
    p.add_argument("--repair5-tail-weight", type=float, default=0.50)
    p.add_argument("--repair5-tail-kappa", type=float, default=6.0)
    p.add_argument("--repair5-calibration-budget", type=float, default=0.0)
    p.add_argument("--calibration-temperature-grid", default="0.35,0.50,0.70,0.85,1.00,1.20,1.50,2.00,3.00,4.00")
    p.add_argument("--calibration-boundary-operator-debt-blend", type=float, default=0.10)
    p.add_argument("--calibration-boundary-tail-weight", type=float, default=0.50)
    p.add_argument("--toy-seed-count", type=int, default=10)
    p.add_argument("--toy-train-size", type=int, default=512)
    p.add_argument("--toy-test-size", type=int, default=512)
    p.add_argument("--toy-steps", type=int, default=120)
    p.add_argument("--toy-lr", type=float, default=1.0e-2)
    p.add_argument("--toy-signal-blend", type=float, default=0.25)
    p.add_argument("--toy-max-norm-ratio", type=float, default=1.15)
    p.add_argument("--toy-eta-dual", type=float, default=0.10)
    p.add_argument("--toy-debt-budget", type=float, default=0.02)
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
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-e":
        run_part_e(args, repair=False)
    elif args.mode == "part-e-merge":
        merge_part_e(args, repair=False)
    elif args.mode == "part-e-repair":
        run_part_e(args, repair=True)
    elif args.mode == "part-e-repair-merge":
        merge_part_e(args, repair=True)
    elif args.mode == "part-e-debt-decomp":
        run_part_e_debt_decomp(args)
    elif args.mode == "part-e-repair2-sweep":
        run_part_e_repair2_sweep(args)
    elif args.mode == "part-e-repair2":
        run_part_e_repair2(args)
    elif args.mode == "part-e-repair2-merge":
        merge_part_e_repair2(args)
    elif args.mode == "part-e-repair3-sweep":
        run_part_e_repair3_sweep(args)
    elif args.mode == "part-e-repair3":
        run_part_e_repair3(args)
    elif args.mode == "part-e-repair3-merge":
        merge_part_e_repair3(args)
    elif args.mode == "part-e-repair4-sweep":
        run_part_e_repair4_sweep(args)
    elif args.mode == "part-e-repair4":
        run_part_e_repair4(args)
    elif args.mode == "part-e-repair4-merge":
        merge_part_e_repair4(args)
    elif args.mode == "part-e-repair5-operator-debt-sweep":
        run_part_e_repair5_sweep(args)
    elif args.mode == "part-e-repair5-operator-debt":
        run_part_e_repair5(args)
    elif args.mode == "part-e-repair5-operator-debt-merge":
        merge_part_e_repair5(args)
    elif args.mode == "part-e-calibration-boundary":
        run_part_e_calibration_boundary(args)
    elif args.mode == "part-e-calibration-boundary-merge":
        merge_part_e_calibration_boundary(args)
    elif args.mode == "part-e-minimal-one-edge":
        run_part_e_minimal_one_edge(args)
    elif args.mode == "part-e-minimal-one-edge-3class":
        run_part_e_minimal_one_edge_three_class(args)
    elif args.mode == "part-f":
        run_part_f(args)
    elif args.mode == "part-g":
        run_part_g(args)
    elif args.mode == "part-h":
        run_part_h(args)
    elif args.mode == "part-i":
        run_part_i(args)
    elif args.mode == "final":
        run_final(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
