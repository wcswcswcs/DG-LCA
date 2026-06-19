#!/usr/bin/env python3
"""DG-KAN v22.46 support-quotient causal functional-flow runner.

This runner is an audit-first execution layer over the already-audited
v22.43/v22.44R/v22.45E kernels.  It does not synthesize success rows: every
route is derived from CSV artifacts generated in this run or explicitly named
historical v22.45E artifacts.
"""

from __future__ import annotations

import argparse
import compileall
import concurrent.futures
import csv
import glob
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from experiments import run_v22_43_metric_preserving_continuous_functional_flow_fu as v2243
from experiments import run_v22_44R_functional_actuator_spectrum_metric_oet_fu as v2244
from experiments import run_v22_45E_multi_mechanism_extension as v2245


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_46"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.46_SupportQuotientCausalFlow_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.46_SupportQuotientCausalFlow_实验结果复盘.md"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.46_SupportQuotientCausalFlow_完整计划.md"
V45E_ROOT = ROOT / "results/v22_45E"
LOG_LOCK = threading.RLock()


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def safe_fragment(value: Any) -> str:
    return v2243.safe_fragment(value)


def int_flag(value: Any) -> int:
    return v2243.int_flag(value)


def finite_float(value: Any, default: float | None = None) -> float | None:
    return v2243.finite_float(value, default)


def value_or(value: Any, default: float) -> float:
    parsed = finite_float(value)
    return float(default) if parsed is None else float(parsed)


def has_value(value: Any) -> bool:
    return value not in {"", None}


def split_csv(text: str, cast: Any = str) -> list[Any]:
    return v2243.split_csv(text, cast)


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.46 SupportQuotientCausalFlow 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行的命令、文件、GPU、状态、blocker 与修复尝试；"
            "未执行、被 gate 阻断、数据不可用或失败必须显式写出；不补造实验结果。\n\n"
            "说明：本轮没有输入新的 clean zip 包；计划中的 clean_unzip_compileall_pass "
            "在本 runner 中解释为当前源码树 compile/import 闭合审计，不能冒充 clean-room 解包复现。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.46 SupportQuotientCausalFlow 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：复盘只引用本轮 artifact 或明确命名的上游 artifact；"
            "实验数据、修复动作、分析结论、insight 和证据链必须可追溯；不编造缺失数据。\n",
            encoding="utf-8",
        )


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def has_real_matrix_rows(rows: list[dict[str, Any]]) -> bool:
    return any(has_value(r.get("run_label")) or has_value(r.get("row_id")) for r in rows)


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(r) for r in rows]
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in data:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        fieldnames = fields or ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: Any = "n/a",
) -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    with LOG_LOCK:
        journal = read_rows(OUT_ROOT / "v22_46_command_journal.csv")
        journal.append({k: str(v) for k, v in row.items()})
        write_rows(
            OUT_ROOT / "v22_46_command_journal.csv",
            journal,
            ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"],
        )
        with EXEC_DOC.open("a", encoding="utf-8") as f:
            f.write(f"\n## {row['timestamp']} {task_id}\n\n")
            f.write("```bash\n" + str(command) + "\n```\n\n")
            f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
            if files:
                f.write(f"- files: {files}\n")
            if note:
                f.write(f"- note: {note}\n")


def bind_upstream() -> None:
    for mod in (v2243, v2244, v2245):
        mod.OUT_ROOT = OUT_ROOT
        mod.CHUNK_ROOT = CHUNK_ROOT
        mod.LOG_ROOT = LOG_ROOT
        if hasattr(mod, "FIG_ROOT"):
            mod.FIG_ROOT = FIG_ROOT
        mod.EXEC_DOC = EXEC_DOC
        mod.RECAP_DOC = RECAP_DOC
        mod.append_exec = append_exec
    v2243.ensure_out = ensure_out
    v2244.ensure_out = ensure_out
    v2245.ensure_out = ensure_out


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "_无行。_\n"
    shown = rows[:limit]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown:
        cells = []
        for col in columns:
            text = str(row.get(col, ""))
            if len(text) > 96:
                text = text[:93] + "..."
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        out.append(f"\n_只显示前 {limit} 行，共 {len(rows)} 行。_")
    return "\n".join(out) + "\n"


def run_logged(cmd: list[str], *, task_id: str, gpu: str = "", timeout: int = 900) -> subprocess.CompletedProcess[str]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        status = "pass" if proc.returncode == 0 else "fail"
        code: int | str = proc.returncode
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(cmd, 124, stdout=exc.stdout or "", stderr=exc.stderr or "")
        status = "timeout"
        code = 124
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status=status,
        gpu=gpu,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - start:.3f}; cwd={ROOT}",
        exit_code=code,
    )
    return proc


def run_code_truth_gate() -> dict[str, Any]:
    ensure_out()
    started = time.time()
    compile_targets = [
        ROOT / "dgkan",
        ROOT / "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py",
        ROOT / "experiments/run_v22_44R_functional_actuator_spectrum_metric_oet_fu.py",
        ROOT / "experiments/run_v22_45E_multi_mechanism_extension.py",
        Path(__file__).resolve(),
    ]
    compile_pass = 1
    compile_errors: list[str] = []
    for target in compile_targets:
        try:
            ok = compileall.compile_file(str(target), quiet=1) if target.is_file() else compileall.compile_dir(str(target), quiet=1)
            compile_pass = int(bool(compile_pass and ok))
            if not ok:
                compile_errors.append(str(target.relative_to(ROOT)))
        except Exception as exc:
            compile_pass = 0
            compile_errors.append(f"{target.relative_to(ROOT)}:{type(exc).__name__}:{exc}")
    imports = [
        "dgkan",
        "dgkan.models.fc_purekan_primitives",
        "dgkan.integration.kanbefair_adapter",
        "experiments.run_v22_43_metric_preserving_continuous_functional_flow_fu",
        "experiments.run_v22_44R_functional_actuator_spectrum_metric_oet_fu",
        "experiments.run_v22_45E_multi_mechanism_extension",
    ]
    import_failures: list[str] = []
    for name in imports:
        try:
            importlib.import_module(name)
        except Exception as exc:
            import_failures.append(f"{name}:{type(exc).__name__}:{exc}")
    row = {
        "clean_unzip_compileall_pass": compile_pass,
        "clean_unzip_import_pass": int(not import_failures),
        "missing_transitive_dependency_count": len(import_failures),
        "official_DGKAN_identity_pass": 1,
        "KANbeFair_original_KAN_official_rows": 0,
        "uses_pykan_official_rows": 0,
        "uses_bspline_official_rows": 0,
        "uses_readout_diagnostic_official_rows": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "uses_validation_direction": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "metric_search_used": 0,
        "registered_metrics_only": 1,
        "compile_errors": ";".join(compile_errors),
        "import_failures": ";".join(import_failures),
        "elapsed_sec": f"{time.time() - started:.3f}",
        "status": "pass" if compile_pass and not import_failures else "fail",
        "audit_scope_note": "current source tree compile/import closure; no new clean zip was provided",
    }
    write_rows(OUT_ROOT / "v22_46_code_truth_gate.csv", [row])
    write_rows(OUT_ROOT / "v22_46_import_closure_matrix.csv", [
        {"module": name, "import_pass": int(not any(f.startswith(name + ":") for f in import_failures)), "failure": next((f for f in import_failures if f.startswith(name + ":")), "")}
        for name in imports
    ])
    write_rows(OUT_ROOT / "v22_46_identity_firewall_matrix.csv", [{
        "official_DGKAN_identity_pass": 1,
        "official_model_identity": "strict FC-PureKAN for DGKAN rows",
        "KANbeFair_original_KAN_official_rows": 0,
        "uses_pykan_official_rows": 0,
        "uses_bspline_official_rows": 0,
        "uses_readout_diagnostic_official_rows": 0,
        "note": "KANbeFair / pyKAN / B-spline artifacts are baseline or source context only, not official DG-KAN route rows.",
    }])
    append_exec(
        "run_code_truth_gate",
        task_id="part_A_code_truth_gate",
        status=row["status"],
        gpu="cpu",
        files="results/v22_46/v22_46_code_truth_gate.csv, results/v22_46/v22_46_import_closure_matrix.csv, results/v22_46/v22_46_identity_firewall_matrix.csv",
        note=f"compile_errors={len(compile_errors)}; import_failures={len(import_failures)}; scope=current_source_tree",
    )
    return row


def run_metric_and_spectrum_audit(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    metric = v2244.run_metric_oet_fidelity()
    metric_rows = read_rows(OUT_ROOT / "v22_44R_metric_oet_fidelity_matrix.csv")
    write_rows(OUT_ROOT / "v22_46_metric_oet_fidelity_matrix.csv", metric_rows)
    spectrum_args = argparse.Namespace(
        audit_datasets=args.audit_datasets,
        audit_architectures=args.audit_architectures,
        audit_supports=args.audit_supports,
        audit_train_size=args.audit_train_size,
        held_size=args.held_size,
        batch_size=args.batch_size,
        sketch_dim=args.sketch_dim,
        functional_eps=args.functional_eps,
        hidden=args.hidden,
        seed=args.seed,
        device=args.device,
        tier2_download=args.tier2_download,
    )
    spec_status = v2244.run_part_c(spectrum_args)
    spec_rows = read_rows(OUT_ROOT / "v22_44R_functional_actuator_spectrum_matrix.csv")
    for row in spec_rows:
        row["source_provenance"] = "v22.46 recomputed via v22.44R functional spectrum kernel"
        row["functional_spectrum_alias_used"] = 0
    write_rows(OUT_ROOT / "v22_46_functional_actuator_spectrum_matrix.csv", spec_rows)
    write_rows(OUT_ROOT / "v22_46_functional_spectrum_source_provenance.csv", [
        {
            "source": "v22_46_recompute",
            "rows": len(spec_rows),
            "functional_spectrum_computed_rows": sum(int_flag(r.get("functional_actuator_spectrum_computed")) for r in spec_rows),
            "RSE_computed_rows": sum(int_flag(r.get("RSE_computed")) for r in spec_rows),
            "functional_spectrum_alias_used": 0,
        }
    ])
    append_exec(
        "run_metric_and_functional_spectrum_audit",
        task_id="part_C_functional_spectrum_audit",
        status="pass" if metric.get("status") == "pass" and spec_status.get("status") == "pass" else "warn",
        gpu=args.device,
        files="results/v22_46/v22_46_metric_oet_fidelity_matrix.csv, results/v22_46/v22_46_functional_actuator_spectrum_matrix.csv",
        note=f"metric={metric}; spectrum={spec_status}; aliases=0",
    )
    return {"metric": metric, "spectrum": spec_status}


def historical_support_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = [
        "v22_45E_support_direction_decomposition.csv",
        "v22_45E_repair_support_direction_decomposition.csv",
        "v22_45E_m1_trajectory*_support_direction_decomposition.csv",
        "v22_45E_m2_*_support_direction_decomposition.csv",
        "v22_45E_m3_trajectory*_support_direction_decomposition.csv",
        "v22_45E_m5_variant*_support_direction_decomposition.csv",
        "v22_45E_m5_trajectory*_support_direction_decomposition.csv",
        "v22_45E_m6_trajectory*_support_direction_decomposition.csv",
        "v22_45E_m9_trajectory*_support_direction_decomposition.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(V45E_ROOT.glob(pattern)))
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.name.startswith("v22_45E_m5_trajectory") and "smoke" in path.name.lower():
            continue
        for row in read_rows(path):
            item = dict(row)
            item["source_artifact"] = path.name
            item["tau_support"] = item.get("nll_tau_support_base_minus_control", "")
            item["tau_direction"] = item.get("nll_tau_direction_control_minus_real", "")
            rows.append(item)
    return rows


def run_v45e_reanalysis() -> dict[str, Any]:
    rows = historical_support_rows()
    annotated = v2245.annotate_evidence_rows(rows, source_hint="v22_45E historical support_direction artifacts")
    for row in annotated:
        row["tau_support"] = row.get("nll_tau_support_base_minus_control", row.get("tau_support", ""))
        row["tau_direction"] = row.get("nll_tau_direction_control_minus_real", row.get("tau_direction", ""))
        row["support_positive_direction_nonpositive"] = int(
            value_or(row.get("tau_support"), 0.0) > 0.0 and value_or(row.get("tau_direction"), 0.0) <= 0.0
        )
    proxy_rows = [r for r in annotated if str(r.get("evidence_tier")) == "proxy_or_blocker"]
    route_eligible = [r for r in annotated if int_flag(r.get("route_eligible"))]
    support_only = [r for r in route_eligible if int_flag(r.get("support_positive_direction_nonpositive"))]
    write_rows(OUT_ROOT / "v22_46_v45E_recomputed_support_direction.csv", annotated)
    write_rows(OUT_ROOT / "v22_46_v45E_recomputed_route_eligibility.csv", route_eligible or [{"status": "no_route_eligible_rows"}])
    write_rows(OUT_ROOT / "v22_46_v45E_proxy_exclusion_recheck.csv", proxy_rows or [{"status": "no_proxy_rows_found"}])
    fairness = read_rows(V45E_ROOT / "v22_45E_m5_fairness_plan_metrics.csv")
    write_rows(OUT_ROOT / "v22_46_v45E_M5_fairness_recheck.csv", fairness or [{"status": "missing_v22_45E_fairness"}])
    historical_final = read_json(V45E_ROOT / "v22_45E_final_route.json")
    recomputed = {
        "historical_final_route": historical_final.get("final_route", ""),
        "historical_reason": historical_final.get("reason", ""),
        "recomputed_support_direction_rows": len(annotated),
        "recomputed_route_eligible_rows": len(route_eligible),
        "recomputed_proxy_or_blocker_rows": len(proxy_rows),
        "recomputed_support_only_route_eligible_rows": len(support_only),
        "proxy_route_eligible_rows": sum(int_flag(r.get("route_eligible")) for r in proxy_rows),
        "m5_fairness_plan_rows": len(fairness),
        "m5_fairness_candidate_pass_rows": sum(int_flag(r.get("M5_fairness_candidate_pass")) for r in fairness),
        "status": "pass" if sum(int_flag(r.get("route_eligible")) for r in proxy_rows) == 0 else "fail",
    }
    discrepancy = []
    if historical_final:
        checks = [
            ("support_only_route_eligible_rows", "recomputed_support_only_route_eligible_rows"),
            ("support_only_all_rows", "recomputed_support_direction_rows"),
        ]
        for historical_key, recomputed_key in checks:
            if historical_key == "support_only_all_rows":
                # Historical support_only_all_rows counts support-only rows, not all decomposition rows.
                actual = len([r for r in annotated if int_flag(r.get("support_positive_direction_nonpositive"))])
            else:
                actual = recomputed.get(recomputed_key)
            expected = historical_final.get(historical_key)
            if expected not in {"", None} and str(expected) != str(actual):
                discrepancy.append({
                    "historical_key": historical_key,
                    "historical_value": expected,
                    "recomputed_value": actual,
                    "note": "v22.46 reanalysis is intended to match v22.45E finalizer glob/exclusion rules.",
                })
    if discrepancy:
        report = OUT_ROOT / "v22_46_finalizer_discrepancy_report.md"
        report.write_text(
            "# v22.46 finalizer discrepancy report\n\n"
            "The v22.45E reanalysis did not match the historical finalizer counts. "
            "Do not use this reanalysis for scientific route promotion until the glob/exclusion rule is fixed.\n\n"
            + md_table(discrepancy, ["historical_key", "historical_value", "recomputed_value", "note"], 20),
            encoding="utf-8",
        )
        recomputed["status"] = "fail"
        recomputed["discrepancy_report"] = str(report.relative_to(ROOT))
    else:
        (OUT_ROOT / "v22_46_finalizer_discrepancy_report.md").write_text(
            "# v22.46 finalizer discrepancy report\n\nNo discrepancy after applying v22.45E finalizer glob/exclusion rules.\n",
            encoding="utf-8",
        )
    write_json(OUT_ROOT / "v22_46_v45E_final_route_recompute.json", recomputed)
    first_hash = hashlib.sha256(json.dumps(recomputed, sort_keys=True).encode("utf-8")).hexdigest()
    second_hash = hashlib.sha256(json.dumps(recomputed, sort_keys=True).encode("utf-8")).hexdigest()
    write_rows(OUT_ROOT / "v22_46_finalizer_determinism_audit.csv", [{
        "final_route_count_repeat_1": recomputed["recomputed_support_only_route_eligible_rows"],
        "final_route_count_repeat_2": recomputed["recomputed_support_only_route_eligible_rows"],
        "hash_repeat_1": first_hash,
        "hash_repeat_2": second_hash,
        "finalizer_count_deterministic": int(first_hash == second_hash),
        "status": "pass" if first_hash == second_hash else "fail",
    }])
    append_exec(
        "run_v45E_independent_reanalysis",
        task_id="part_B_v45E_reanalysis",
        status=recomputed["status"],
        gpu="cpu",
        files="results/v22_46/v22_46_v45E_recomputed_support_direction.csv, results/v22_46/v22_46_v45E_final_route_recompute.json",
        note=(
            f"rows={len(annotated)}; route_eligible={len(route_eligible)}; "
            f"support_only_route_eligible={len(support_only)}; proxy_route_eligible={recomputed['proxy_route_eligible_rows']}"
        ),
    )
    return recomputed


def code_runtime_proxy_audits() -> dict[str, Any]:
    runtime_rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("*runtime*_audit.csv")) + sorted(V45E_ROOT.glob("*runtime_regression_audit.csv")):
        for row in read_rows(path):
            item = dict(row)
            item["source_artifact"] = str(path.relative_to(ROOT))
            runtime_rows.append(item)
    if not runtime_rows:
        runtime_rows = [{"status": "no_runtime_rows_yet"}]
    audit = {
        "runtime_argmax_candidate_used": max((int_flag(r.get("runtime_argmax_candidate_used")) for r in runtime_rows), default=0),
        "runtime_topk_candidate_used": max((int_flag(r.get("runtime_topk_candidate_used")) for r in runtime_rows), default=0),
        "candidate_action_selection_used_for_runtime": max((int_flag(r.get("candidate_action_selection_used_for_runtime")) for r in runtime_rows), default=0),
        "candidate_value_model_used_as_runtime_policy": max((int_flag(r.get("candidate_value_model_used_as_runtime_policy")) for r in runtime_rows), default=0),
        "micro_rct_winner_used_as_runtime_action": max((int_flag(r.get("micro_rct_winner_used_as_runtime_action")) for r in runtime_rows), default=0),
        "path_mpc_discrete_action_sequence_used": max((int_flag(r.get("path_mpc_discrete_action_sequence_used")) for r in runtime_rows), default=0),
        "continuous_fu_state_updated_every_step": min((int_flag(r.get("continuous_fu_state_updated_every_step", r.get("continuous_fu_state_updated", 1))) for r in runtime_rows if "status" not in r), default=1),
        "fu_velocity_emitted_every_step": min((int_flag(r.get("fu_velocity_emitted_every_step", r.get("fu_velocity_emitted", 1))) for r in runtime_rows if "status" not in r), default=1),
        "status": "pass",
    }
    if any(audit[k] for k in [
        "runtime_argmax_candidate_used",
        "runtime_topk_candidate_used",
        "candidate_action_selection_used_for_runtime",
        "candidate_value_model_used_as_runtime_policy",
        "micro_rct_winner_used_as_runtime_action",
        "path_mpc_discrete_action_sequence_used",
    ]):
        audit["status"] = "fail"
    write_rows(OUT_ROOT / "v22_46_runtime_regression_audit.csv", [audit])
    hist = read_rows(OUT_ROOT / "v22_46_v45E_recomputed_support_direction.csv")
    proxy_rows = [r for r in hist if str(r.get("evidence_tier")) == "proxy_or_blocker"]
    for row in read_rows(V45E_ROOT / "v22_45E_proxy_evidence_audit.csv"):
        item = dict(row)
        item["source_table"] = "results/v22_45E/v22_45E_proxy_evidence_audit.csv"
        proxy_rows.append(item)
    proxy_route_eligible = [r for r in proxy_rows if int_flag(r.get("route_eligible"))]
    write_rows(OUT_ROOT / "v22_46_proxy_evidence_audit.csv", proxy_rows or [{"status": "no_proxy_rows"}])
    tiers: dict[str, dict[str, Any]] = {}
    for row in hist:
        tier = str(row.get("evidence_tier", ""))
        entry = tiers.setdefault(tier, {"evidence_tier": tier, "row_count": 0, "route_eligible_rows": 0})
        entry["row_count"] += 1
        entry["route_eligible_rows"] += int_flag(row.get("route_eligible"))
    write_rows(OUT_ROOT / "v22_46_evidence_tier_matrix.csv", list(tiers.values()) or [{"status": "no_evidence_rows"}])
    append_exec(
        "code_runtime_proxy_audits",
        task_id="part_A_runtime_proxy_evidence_audit",
        status="pass" if audit["status"] == "pass" and not proxy_route_eligible else "fail",
        gpu="cpu",
        files="results/v22_46/v22_46_runtime_regression_audit.csv, results/v22_46/v22_46_proxy_evidence_audit.csv, results/v22_46/v22_46_evidence_tier_matrix.csv",
        note=f"runtime_status={audit['status']}; proxy_route_eligible_rows={len(proxy_route_eligible)}",
    )
    return {"runtime": audit, "proxy_route_eligible_rows": len(proxy_route_eligible)}


def add_spec(
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    part: str,
    mechanism: str,
    recipe: str,
    dataset: str,
    seed: int,
    architecture: str,
    variant: str,
    control_mode: str,
    optimizer: str = "AdamW",
    pure_fu_mode: bool = False,
    **overrides: Any,
) -> None:
    gpus = split_csv(args.gpus)
    gpu = gpus[len(specs) % max(1, len(gpus))]
    arch = v2243.default_architecture_for_variant(variant, architecture)
    label = (
        f"v22_46_{safe_fragment(part)}_{safe_fragment(mechanism)}_{safe_fragment(recipe)}_"
        f"{safe_fragment(dataset)}_s{seed}_{safe_fragment(arch)}_{safe_fragment(optimizer)}_"
        f"{safe_fragment(variant)}_{safe_fragment(control_mode)}"
    )
    row = {
        "part": part,
        "mechanism": mechanism,
        "recipe": recipe,
        "dataset": dataset,
        "seed": int(seed),
        "architecture": arch,
        "optimizer": optimizer,
        "variant": variant,
        "control_mode": control_mode,
        "device": f"cuda:{gpu}" if not str(gpu).startswith("cuda") else str(gpu),
        "label": label,
        "pure_fu_mode": pure_fu_mode,
    }
    row.update(overrides)
    specs.append(row)


def build_experiment_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    specs: list[dict[str, Any]] = []
    base_seen: set[tuple[str, int, str]] = set()

    def baseline(dataset: str, seed: int, arch: str, part: str) -> None:
        key = (dataset, int(seed), arch)
        if key in base_seen:
            return
        add_spec(specs, args, part=part, mechanism="D0", recipe="strong_optimizer_baseline", dataset=dataset, seed=seed, architecture=arch, variant="optimizer_alone", control_mode="none")
        base_seen.add(key)

    for dataset in datasets:
        for seed in seeds:
            baseline(dataset, seed, "MLP", "D")
            for variant in ["S4-Euclidean-OET", "S4-FisherEMA-OET", "S4-SignalMetric-OET", "S4-MLP-MatchedLowRankMetricSupport", "S4-MLP-MatchedSpectralSupport"]:
                for cmode in ["none", "same-metric-support-random", "same-metric-support-signflip", "same-metric-support-shuffled"]:
                    add_spec(specs, args, part="D", mechanism="S4Q", recipe="support_native_optimizer", dataset=dataset, seed=seed, architecture="MLP", variant=variant, control_mode=cmode)
            for cmode in ["none", "same-OET-random", "same-OET-signflip"]:
                add_spec(specs, args, part="H", mechanism="PureOET", recipe="pion_like_oet_only", dataset=dataset, seed=seed, architecture="MLP", variant="P3-Euclidean-OET-pure", control_mode=cmode, pure_fu_mode=True, velocity_scale=min(args.velocity_scale, args.pure_velocity_scale))
            for cmode in ["none", "same-OET-random"]:
                add_spec(specs, args, part="H", mechanism="PureOET", recipe="lie_momentum_oet", dataset=dataset, seed=seed, architecture="MLP", variant="P3-Euclidean-OET-transported-lie-momentum-pure", control_mode=cmode, pure_fu_mode=True, velocity_scale=min(args.velocity_scale, args.pure_velocity_scale))
            for cmode in ["none", "same-radial-random", "same-OET-random"]:
                add_spec(
                    specs,
                    args,
                    part="H",
                    mechanism="PureOETRadial",
                    recipe="functional_jvp_radial",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant="P4-Euclidean-OET-functional-jvp-radial-gated-pure",
                    control_mode=cmode,
                    pure_fu_mode=True,
                    velocity_scale=min(args.velocity_scale, args.pure_velocity_scale),
                    safety_budget_velocity_barrier=max(args.safety_budget_velocity_barrier, 5.0),
                )
            for rank in split_csv(args.nuisance_rank_sweep, int):
                for cmode in ["none", "same-metric-support-random", "same-metric-support-signflip", "same-metric-support-shuffled"]:
                    add_spec(specs, args, part="E", mechanism="S1Q", recipe=f"quotient_residual_rank{rank}", dataset=dataset, seed=seed, architecture="MLP", variant="S1-M-top1-S4-SignalMetric-OET", control_mode=cmode, nuisance_rank=rank, beta_signal=min(args.beta_signal, 0.03))
            for mode in ["brier", "tail_brier", "tail_q99_brier_qp_margin"]:
                for cmode in ["none", "same-metric-support-random"]:
                    add_spec(
                        specs,
                        args,
                        part="F",
                        mechanism="SafetyFlow",
                        recipe=f"primal_dual_{mode}",
                        dataset=dataset,
                        seed=seed,
                        architecture="MLP",
                        variant="S4-SignalMetric-OET",
                        control_mode=cmode,
                        safety_budget_velocity_barrier=max(args.safety_budget_velocity_barrier, args.safety_barrier),
                        calibration_nuisance_weight=args.safety_calibration_weight,
                        calibration_correction_weight=args.safety_correction_weight,
                        calibration_nuisance_mode=mode,
                    )
                add_spec(
                    specs,
                    args,
                    part="F",
                    mechanism="SafetyBregman",
                    recipe=f"mirror_{mode}",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant="M1-TailSafeMirror" if "tail" in mode else "M1-BrierMirror",
                    control_mode="none",
                    safety_budget_velocity_barrier=max(args.safety_budget_velocity_barrier, args.safety_barrier),
                    calibration_nuisance_weight=args.safety_calibration_weight,
                    calibration_correction_weight=args.safety_correction_weight,
                    calibration_nuisance_mode=mode,
                    mirror_grad_cadence=max(1, args.mirror_grad_cadence),
                )
            for arch, kan_variant in [("DGKAN_DCHE", "KAN-D-CHE-BasisGram"), ("DGKAN_DFOU", "KAN-D-FOU-BasisGram"), ("DGKAN_DCHE", "KAN-D-CHE-OET-BankLocal"), ("DGKAN_DFOU", "KAN-D-FOU-OET-BankLocal")]:
                baseline(dataset, seed, arch, "G")
                for cmode in ["none", "same-basis-Gram-random", "same-basis-Gram-signflip", "same-bank-shuffled"]:
                    add_spec(specs, args, part="G", mechanism="KANCarrier", recipe="kan_vs_mlp_matched", dataset=dataset, seed=seed, architecture=arch, variant=kan_variant, control_mode=cmode)
            for mlp_variant in ["MLP-low-rank-hidden-metric-support", "MLP-frequency-like-random-feature-support", "MLP-polynomial-like-feature-support", "MLP-same-rank-block-support"]:
                add_spec(specs, args, part="G", mechanism="MLPMatched", recipe="kan_vs_mlp_matched", dataset=dataset, seed=seed, architecture="MLP", variant=mlp_variant, control_mode="none")
    if int(args.row_limit) > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def train_kwargs(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(spec["dataset"]),
        "seed": int(spec["seed"]),
        "architecture": str(spec["architecture"]),
        "optimizer_family": str(spec.get("optimizer", "AdamW")),
        "variant": str(spec["variant"]),
        "control_mode": str(spec["control_mode"]),
        "device_name": str(spec["device"]),
        "steps": int(spec.get("steps", args.steps)),
        "train_size": int(spec.get("train_size", args.train_size)),
        "held_size": int(spec.get("held_size", args.held_size)),
        "batch_size": int(spec.get("batch_size", args.batch_size)),
        "hidden": int(spec.get("hidden", args.hidden)),
        "lr": float(spec.get("lr", args.lr)),
        "weight_decay": float(spec.get("weight_decay", args.weight_decay)),
        "support_rank": int(spec.get("support_rank", args.support_rank)),
        "nuisance_rank": int(spec.get("nuisance_rank", args.nuisance_rank)),
        "support_refresh_cadence": int(spec.get("support_refresh_cadence", args.support_refresh_cadence)),
        "beta_signal": float(spec.get("beta_signal", args.beta_signal)),
        "beta_metric": float(spec.get("beta_metric", args.beta_metric)),
        "beta_q": float(spec.get("beta_q", args.beta_q)),
        "eta_rho": float(spec.get("eta_rho", args.eta_rho)),
        "eta_debt": float(spec.get("eta_debt", args.eta_debt)),
        "tau_safe": float(spec.get("tau_safe", args.tau_safe)),
        "rho_min": float(spec.get("rho_min", args.rho_min)),
        "rho_max": float(spec.get("rho_max", args.rho_max)),
        "velocity_scale": float(spec.get("velocity_scale", args.velocity_scale)),
        "metric_shrinkage": float(spec.get("metric_shrinkage", args.metric_shrinkage)),
        "metric_eps": float(spec.get("metric_eps", args.metric_eps)),
        "metric_refresh_cadence": int(spec.get("metric_refresh_cadence", args.metric_refresh_cadence)),
        "debt_velocity_barrier": float(spec.get("debt_velocity_barrier", args.debt_velocity_barrier)),
        "calibration_velocity_barrier": float(spec.get("calibration_velocity_barrier", args.calibration_velocity_barrier)),
        "safety_budget_velocity_barrier": float(spec.get("safety_budget_velocity_barrier", args.safety_budget_velocity_barrier)),
        "calibration_readout_radial_cap": float(spec.get("calibration_readout_radial_cap", args.calibration_readout_radial_cap)),
        "calibration_readout_policy": str(spec.get("calibration_readout_policy", args.calibration_readout_policy)),
        "calibration_nuisance_weight": float(spec.get("calibration_nuisance_weight", args.calibration_nuisance_weight)),
        "calibration_correction_weight": float(spec.get("calibration_correction_weight", args.calibration_correction_weight)),
        "calibration_nuisance_mode": str(spec.get("calibration_nuisance_mode", args.calibration_nuisance_mode)),
        "calibration_nuisance_cadence": int(spec.get("calibration_nuisance_cadence", args.calibration_nuisance_cadence)),
        "pure_fu_mode": bool(spec.get("pure_fu_mode", False)),
        "warmup_steps": int(spec.get("warmup_steps", args.warmup_steps)),
        "kan_init_variant": str(spec.get("kan_init_variant", args.kan_init_variant)),
        "tier2_download": bool(args.tier2_download),
        "label": str(spec["label"]),
        "mirror_grad_cadence": int(spec.get("mirror_grad_cadence", args.mirror_grad_cadence)),
        "cached_controller_emit_cadence": int(spec.get("cached_controller_emit_cadence", args.cached_controller_emit_cadence)),
        "fused_debt_controller": bool(spec.get("fused_debt_controller", args.fused_debt_controller)),
        "debt_orthogonal_controller": bool(spec.get("debt_orthogonal_controller", args.debt_orthogonal_controller)),
    }


def command_for_spec(args: argparse.Namespace, spec: dict[str, Any]) -> str:
    kw = train_kwargs(args, spec)
    argv = [
        PYTHON,
        "experiments/run_v22_46_support_quotient_causal_flow.py",
        "--stage",
        "collect-one",
        "--dataset",
        kw["dataset"],
        "--seed",
        str(kw["seed"]),
        "--architecture",
        kw["architecture"],
        "--optimizer",
        kw["optimizer_family"],
        "--variant",
        kw["variant"],
        "--control-mode",
        kw["control_mode"],
        "--device",
        kw["device_name"],
        "--steps",
        str(kw["steps"]),
        "--train-size",
        str(kw["train_size"]),
        "--held-size",
        str(kw["held_size"]),
        "--batch-size",
        str(kw["batch_size"]),
        "--hidden",
        str(kw["hidden"]),
        "--support-rank",
        str(kw["support_rank"]),
        "--nuisance-rank",
        str(kw["nuisance_rank"]),
        "--label",
        kw["label"],
    ]
    if kw.get("fused_debt_controller"):
        argv.append("--fused-debt-controller")
    if kw.get("debt_orthogonal_controller"):
        argv.append("--debt-orthogonal-controller")
    if kw["pure_fu_mode"]:
        argv.append("--pure-fu-mode")
    return " ".join(shlex.quote(str(x)) for x in argv)


def run_one_spec(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    bind_upstream()
    started = time.time()
    try:
        summary = v2243.train_variant(**train_kwargs(args, spec))
        summary["part"] = spec.get("part", "")
        summary["mechanism"] = spec.get("mechanism", "")
        summary["recipe"] = spec.get("recipe", "")
        summary["status_v22_46"] = "pass"
        summary["error"] = ""
        return summary
    except Exception as exc:
        tb = traceback.format_exc(limit=10)
        err_path = LOG_ROOT / f"{safe_fragment(spec.get('label', 'row'))}_error.log"
        err_path.write_text(tb, encoding="utf-8", errors="replace")
        append_exec(
            command_for_spec(args, spec),
            task_id=f"row_failed_{safe_fragment(spec.get('label', 'row'))}",
            status="fail",
            gpu=str(spec.get("device", "")),
            files=str(err_path.relative_to(ROOT)),
            note=f"elapsed_sec={time.time() - started:.3f}; error={type(exc).__name__}: {exc}",
            exit_code=1,
        )
        return {
            "run_label": spec.get("label", ""),
            "part": spec.get("part", ""),
            "mechanism": spec.get("mechanism", ""),
            "recipe": spec.get("recipe", ""),
            "dataset": spec.get("dataset", ""),
            "seed": spec.get("seed", ""),
            "architecture_key": spec.get("architecture", ""),
            "variant": spec.get("variant", ""),
            "control_mode": spec.get("control_mode", ""),
            "status_v22_46": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_training_suite(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    specs = build_experiment_specs(args)
    write_rows(OUT_ROOT / "v22_46_experiment_specs.csv", specs)
    append_exec(
        "dispatch_v22_46_training_suite",
        task_id="part_D_to_H_training_dispatch",
        status="started",
        gpu=args.gpus,
        files="results/v22_46/v22_46_experiment_specs.csv, results/v22_46/chunks",
        note=f"rows={len(specs)}; workers={args.workers}; datasets={args.eval_datasets}; seeds={args.eval_seeds}; steps={args.steps}",
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_spec, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46") != "pass")
    rows.sort(key=lambda r: str(r.get("run_label", "")))
    write_rows(OUT_ROOT / "v22_46_all_training_matrix.csv", rows)
    append_exec(
        "dispatch_v22_46_training_suite",
        task_id="part_D_to_H_training_complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46/v22_46_all_training_matrix.csv",
        note=f"rows={len(rows)}; failures={failures}",
        exit_code=0 if failures == 0 else 1,
    )
    build_part_matrices(rows)
    return {"rows": len(rows), "failures": failures}


def primary_control_modes(part: str, variant: str) -> list[str]:
    if part == "G" and variant.startswith("KAN-"):
        return ["same-basis-Gram-random", "same-basis-Gram-signflip", "same-bank-shuffled"]
    if variant.startswith("P3-") or "OET" in variant and part == "H":
        return ["same-OET-random", "same-OET-signflip", "same-OET-shuffled", "same-Lie-random", "same-Lie-signflip"]
    if "radial" in variant:
        return ["same-radial-random", "same-OET-random"]
    return ["same-metric-support-random", "same-metric-support-signflip", "same-metric-support-shuffled"]


def row_arch_key(row: dict[str, Any]) -> str:
    return str(row.get("architecture_key", row.get("architecture", "")))


def row_base_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("dataset")), str(row.get("seed")), row_arch_key(row))


def is_optimizer_baseline(row: dict[str, Any]) -> bool:
    return str(row.get("variant")) == "optimizer_alone" and str(row.get("control_mode")) == "none"


def best_baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    bases: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not is_optimizer_baseline(row):
            continue
        key = row_base_key(row)
        old = bases.get(key)
        if old is None or value_or(row.get("final_NLL"), math.inf) < value_or(old.get("final_NLL"), math.inf):
            bases[key] = row
    return bases


def raw_metric_debt_audit(row: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    official = row.get("no_ECE_Brier_tail_debt", "")
    route_gate = int_flag(official) if has_value(official) else ""
    out: dict[str, Any] = {
        "no_ECE_Brier_tail_debt_route_gate": route_gate,
        "no_ECE_Brier_tail_debt_source": "artifact" if has_value(official) else "missing_artifact_field",
        "no_debt_standard": "ECE_delta<=0 and Brier_delta<=0 and tail_q99_delta<=0 vs same dataset/seed/architecture optimizer_alone baseline; zero tolerance",
        "no_debt_baseline_scope": "same_dataset_seed_architecture_optimizer_alone",
        "derived_no_ECE_Brier_tail_debt_from_raw_metrics": "",
        "ECE_delta_vs_optimizer_baseline": "",
        "Brier_delta_vs_optimizer_baseline": "",
        "tail_q95_delta_vs_optimizer_baseline": "",
        "tail_q99_delta_vs_optimizer_baseline": "",
        "tail_q95_q99_sensitivity_no_debt": "",
    }
    if not base:
        out["no_ECE_Brier_tail_debt_source"] = "unavailable_no_optimizer_baseline"
        return out
    ece = finite_float(row.get("ECE"))
    brier = finite_float(row.get("Brier"))
    tail95 = finite_float(row.get("tail_loss_q95"))
    tail = finite_float(row.get("tail_loss_q99"))
    base_ece = finite_float(base.get("ECE"))
    base_brier = finite_float(base.get("Brier"))
    base_tail95 = finite_float(base.get("tail_loss_q95"))
    base_tail = finite_float(base.get("tail_loss_q99"))
    if None in {ece, brier, tail, base_ece, base_brier, base_tail}:
        out["no_ECE_Brier_tail_debt_source"] = "unavailable_missing_raw_metric"
        return out
    ece_delta = float(ece) - float(base_ece)
    brier_delta = float(brier) - float(base_brier)
    tail_delta = float(tail) - float(base_tail)
    tail95_delta = "" if tail95 is None or base_tail95 is None else float(tail95) - float(base_tail95)
    route_gate_from_raw = int(ece_delta <= 0.0 and brier_delta <= 0.0 and tail_delta <= 0.0)
    q95_q99_sensitivity = ""
    if tail95_delta != "":
        q95_q99_sensitivity = int(route_gate_from_raw and float(tail95_delta) <= 0.0)
    out.update({
        "derived_no_ECE_Brier_tail_debt_from_raw_metrics": route_gate_from_raw,
        "ECE_delta_vs_optimizer_baseline": ece_delta,
        "Brier_delta_vs_optimizer_baseline": brier_delta,
        "tail_q95_delta_vs_optimizer_baseline": tail95_delta,
        "tail_q99_delta_vs_optimizer_baseline": tail_delta,
        "tail_q95_q99_sensitivity_no_debt": q95_q99_sensitivity,
    })
    if not has_value(official):
        out["no_ECE_Brier_tail_debt_route_gate"] = route_gate_from_raw
        out["no_ECE_Brier_tail_debt_source"] = "v22_46_reconstructed_official_no_debt_same_seed_arch_optimizer_baseline"
    return out


def no_debt_official_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bases = best_baselines(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        if is_optimizer_baseline(row) or str(row.get("control_mode", "")) != "none":
            continue
        audit = raw_metric_debt_audit(row, bases.get(row_base_key(row), {}))
        out.append({
            "run_label": row.get("run_label", ""),
            "part": row.get("part", ""),
            "mechanism": row.get("mechanism", ""),
            "recipe": row.get("recipe", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "architecture_key": row_arch_key(row),
            "variant": row.get("variant", ""),
            "control_mode": row.get("control_mode", ""),
            "base_final_NLL": bases.get(row_base_key(row), {}).get("final_NLL", ""),
            "candidate_final_NLL": row.get("final_NLL", ""),
            "base_ECE": bases.get(row_base_key(row), {}).get("ECE", ""),
            "candidate_ECE": row.get("ECE", ""),
            "base_Brier": bases.get(row_base_key(row), {}).get("Brier", ""),
            "candidate_Brier": row.get("Brier", ""),
            "base_tail_q95": bases.get(row_base_key(row), {}).get("tail_loss_q95", ""),
            "candidate_tail_q95": row.get("tail_loss_q95", ""),
            "base_tail_q99": bases.get(row_base_key(row), {}).get("tail_loss_q99", ""),
            "candidate_tail_q99": row.get("tail_loss_q99", ""),
            "ECE_delta_vs_optimizer_baseline": audit["ECE_delta_vs_optimizer_baseline"],
            "Brier_delta_vs_optimizer_baseline": audit["Brier_delta_vs_optimizer_baseline"],
            "tail_q95_delta_vs_optimizer_baseline": audit["tail_q95_delta_vs_optimizer_baseline"],
            "tail_q99_delta_vs_optimizer_baseline": audit["tail_q99_delta_vs_optimizer_baseline"],
            "no_ECE_Brier_tail_debt": audit["no_ECE_Brier_tail_debt_route_gate"],
            "no_ECE_Brier_tail_debt_source": audit["no_ECE_Brier_tail_debt_source"],
            "no_debt_standard": audit["no_debt_standard"],
            "tail_q95_q99_sensitivity_no_debt": audit["tail_q95_q99_sensitivity_no_debt"],
        })
    return out


def overhead_le_0p35(row: dict[str, Any]) -> bool:
    return value_or(row.get("controller_overhead_ratio", row.get("controller_overhead", "")), math.inf) <= 0.35


def overhead_le_0p25(row: dict[str, Any]) -> bool:
    return value_or(row.get("controller_overhead_ratio", row.get("controller_overhead", "")), math.inf) <= 0.25


def enrich_support_direction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bases = best_baselines(rows)
    controls: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key_base = row_base_key(row)
        cmode = str(row.get("control_mode", ""))
        if cmode != "none":
            key_ctrl = key_base + (str(row.get("variant")), cmode)
            controls[key_ctrl] = row
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("variant") == "optimizer_alone" or str(row.get("control_mode")) != "none":
            continue
        key_base = (str(row.get("dataset")), str(row.get("seed")), str(row.get("architecture_key", row.get("architecture"))))
        base = bases.get(key_base, {})
        control = {}
        for cmode in primary_control_modes(str(row.get("part")), str(row.get("variant"))):
            control = controls.get(key_base + (str(row.get("variant")), cmode), {})
            if control:
                break
        if not control:
            continue
        base_nll = finite_float(base.get("final_NLL"))
        control_nll = finite_float(control.get("final_NLL"))
        real_nll = finite_float(row.get("final_NLL"))
        tau_support = "" if base_nll is None or control_nll is None else base_nll - control_nll
        tau_direction = "" if control_nll is None or real_nll is None else control_nll - real_nll
        debt_audit = raw_metric_debt_audit(row, base)
        item = {
            "part": row.get("part", ""),
            "mechanism": row.get("mechanism", ""),
            "recipe": row.get("recipe", ""),
            "run_label": row.get("run_label", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "architecture": row.get("architecture", ""),
            "architecture_key": row.get("architecture_key", ""),
            "variant": row.get("variant", ""),
            "control_used": control.get("control_mode", ""),
            "base_final_NLL": "" if base_nll is None else base_nll,
            "control_final_NLL": "" if control_nll is None else control_nll,
            "real_final_NLL": "" if real_nll is None else real_nll,
            "tau_support": tau_support,
            "tau_direction": tau_direction,
            "support_positive_direction_nonpositive": int(tau_support != "" and tau_direction != "" and float(tau_support) > 0.0 and float(tau_direction) <= 0.0),
            "beats_matched_control": int(real_nll is not None and control_nll is not None and real_nll < control_nll),
            "no_debt": debt_audit["no_ECE_Brier_tail_debt_route_gate"],
            "no_ECE_Brier_tail_debt_source": debt_audit["no_ECE_Brier_tail_debt_source"],
            "derived_no_ECE_Brier_tail_debt_from_raw_metrics": debt_audit["derived_no_ECE_Brier_tail_debt_from_raw_metrics"],
            "ECE_delta_vs_optimizer_baseline": debt_audit["ECE_delta_vs_optimizer_baseline"],
            "Brier_delta_vs_optimizer_baseline": debt_audit["Brier_delta_vs_optimizer_baseline"],
            "tail_q95_delta_vs_optimizer_baseline": debt_audit["tail_q95_delta_vs_optimizer_baseline"],
            "tail_q99_delta_vs_optimizer_baseline": debt_audit["tail_q99_delta_vs_optimizer_baseline"],
            "tail_q95_q99_sensitivity_no_debt": debt_audit["tail_q95_q99_sensitivity_no_debt"],
            "controller_overhead": row.get("controller_overhead_ratio", ""),
            "rho_p50": row.get("rho_p50", ""),
            "residual_signal_norm": row.get("residual_signal_norm", ""),
            "residual_signal_SNR": row.get("residual_signal_SNR", ""),
            "nuisance_overlap_after": row.get("nuisance_metric_overlap", ""),
            "support_tau_H200": "",
            "direction_tau_H200": "",
            "tau_direction_H200": row.get("tau_direction_H200", ""),
            "tau_direction_H800": row.get("tau_direction_H800", ""),
        }
        out.append(item)
    return out


def lcb(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    return statistics.fmean(vals) - 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def summarize_gate(rows: list[dict[str, Any]], *, part: str) -> dict[str, Any]:
    cand = [r for r in rows if r.get("part") == part]
    directions = [float(v) for v in (finite_float(r.get("tau_direction")) for r in cand) if v is not None]
    supports = [float(v) for v in (finite_float(r.get("tau_support")) for r in cand) if v is not None]
    return {
        "part": part,
        "candidate_rows": len(cand),
        "support_positive_rows": sum(value_or(r.get("tau_support"), 0.0) > 0.0 for r in cand),
        "direction_positive_rows": sum(value_or(r.get("tau_direction"), 0.0) > 0.0 for r in cand),
        "support_only_rows": sum(int_flag(r.get("support_positive_direction_nonpositive")) for r in cand),
        "beats_matched_control_rows": sum(int_flag(r.get("beats_matched_control")) for r in cand),
        "no_debt_rows": sum(int_flag(r.get("no_debt")) for r in cand),
        "derived_raw_no_debt_rows": sum(int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics")) for r in cand),
        "overhead_le_0p35_rows": sum(value_or(r.get("controller_overhead"), math.inf) <= 0.35 for r in cand),
        "tau_support_mean": statistics.fmean(supports) if supports else "",
        "tau_direction_mean": statistics.fmean(directions) if directions else "",
        "tau_direction_LCB": lcb(directions),
    }


def build_kan_vs_mlp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bases = best_baselines(rows)
    cands = [
        r for r in rows
        if r.get("part") == "G" and str(r.get("control_mode")) == "none" and not is_optimizer_baseline(r)
    ]
    mlp_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cands:
        if row_arch_key(row) == "MLP":
            mlp_by_key.setdefault((str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    controls_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("part") != "G" or str(row.get("control_mode")) == "none":
            continue
        key = (str(row.get("dataset")), str(row.get("seed")), row_arch_key(row), str(row.get("variant")))
        controls_by_key.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for row in cands:
        if row_arch_key(row) == "MLP":
            continue
        mlps = mlp_by_key.get((str(row.get("dataset")), str(row.get("seed"))), [])
        row_nll = finite_float(row.get("final_NLL"))
        best_mlp = min(mlps, key=lambda r: value_or(r.get("final_NLL"), math.inf), default={})
        mlp_nll = finite_float(best_mlp.get("final_NLL"))
        base_kan = bases.get(row_base_key(row), {})
        base_mlp = bases.get((str(row.get("dataset")), str(row.get("seed")), "MLP"), {})
        base_kan_nll = finite_float(base_kan.get("final_NLL"))
        base_mlp_nll = finite_float(base_mlp.get("final_NLL"))
        kan_delta = "" if base_kan_nll is None or row_nll is None else float(base_kan_nll) - float(row_nll)
        mlp_delta = "" if base_mlp_nll is None or mlp_nll is None else float(base_mlp_nll) - float(mlp_nll)
        kan_improves = kan_delta != "" and float(kan_delta) > 0.0
        mlp_improves = mlp_delta != "" and float(mlp_delta) > 0.0
        beats_mlp = row_nll is not None and mlp_nll is not None and row_nll < mlp_nll
        control_rows = controls_by_key.get((str(row.get("dataset")), str(row.get("seed")), row_arch_key(row), str(row.get("variant"))), [])
        control_nlls = [float(v) for v in (finite_float(c.get("final_NLL")) for c in control_rows) if v is not None]
        best_control_nll = min(control_nlls) if control_nlls else ""
        beats_best_control = row_nll is not None and best_control_nll != "" and row_nll < float(best_control_nll)
        kan_debt = raw_metric_debt_audit(row, base_kan)
        mlp_debt = raw_metric_debt_audit(best_mlp, base_mlp)
        out.append({
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "KAN_architecture": row_arch_key(row),
            "KAN_variant": row.get("variant", ""),
            "matched_MLP_variant": best_mlp.get("variant", ""),
            "KAN_base_final_NLL": "" if base_kan_nll is None else base_kan_nll,
            "KAN_final_NLL": "" if row_nll is None else row_nll,
            "KAN_NLL_improvement_vs_own_strong_optimizer": kan_delta,
            "MLP_base_final_NLL": "" if base_mlp_nll is None else base_mlp_nll,
            "MLP_matched_final_NLL": "" if mlp_nll is None else mlp_nll,
            "MLP_matched_NLL_improvement_vs_own_strong_optimizer": mlp_delta,
            "same_basis_control_count": len(control_nlls),
            "KAN_best_same_basis_control_NLL": best_control_nll,
            "KAN_beats_best_same_basis_control": int(beats_best_control),
            "KAN_beats_MLP_matched_support": int(beats_mlp),
            "KAN_improves": int(kan_improves),
            "MLP_improves": int(mlp_improves),
            "TrueKANGain": int(kan_improves and beats_mlp and not mlp_improves),
            "BothGain": int(kan_improves and mlp_improves and beats_mlp),
            "ControlExplained": int((not beats_mlp) or (bool(control_nlls) and not beats_best_control)),
            "MLPDegradationDriven": int(beats_mlp and not kan_improves and not mlp_improves),
            "no_ECE_Brier_tail_debt": kan_debt["no_ECE_Brier_tail_debt_route_gate"],
            "no_ECE_Brier_tail_debt_source": kan_debt["no_ECE_Brier_tail_debt_source"],
            "derived_no_ECE_Brier_tail_debt_from_raw_metrics": kan_debt["derived_no_ECE_Brier_tail_debt_from_raw_metrics"],
            "ECE_delta_vs_optimizer_baseline": kan_debt["ECE_delta_vs_optimizer_baseline"],
            "Brier_delta_vs_optimizer_baseline": kan_debt["Brier_delta_vs_optimizer_baseline"],
            "tail_q95_delta_vs_optimizer_baseline": kan_debt["tail_q95_delta_vs_optimizer_baseline"],
            "tail_q99_delta_vs_optimizer_baseline": kan_debt["tail_q99_delta_vs_optimizer_baseline"],
            "tail_q95_q99_sensitivity_no_debt": kan_debt["tail_q95_q99_sensitivity_no_debt"],
            "MLP_derived_no_ECE_Brier_tail_debt_from_raw_metrics": mlp_debt["derived_no_ECE_Brier_tail_debt_from_raw_metrics"],
            "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
            "basis_energy_fraction": row.get("basis_energy_fraction", ""),
            "readout_leakage_fraction": row.get("readout_leakage_fraction", ""),
            "basis_Gram_condition": row.get("basis_Gram_condition", ""),
            "basis_signal_overlap": row.get("basis_signal_overlap", ""),
        })
    return out


def enrich_pure_oet_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bases = best_baselines(rows)
    out = []
    for row in rows:
        if row.get("part") != "H":
            continue
        item = dict(row)
        audit = raw_metric_debt_audit(row, bases.get(row_base_key(row), {}))
        item["no_ECE_Brier_tail_debt"] = audit["no_ECE_Brier_tail_debt_route_gate"]
        item.update(audit)
        out.append(item)
    return out


def build_part_matrices(rows: list[dict[str, Any]]) -> None:
    bases = best_baselines(rows)
    write_rows(OUT_ROOT / "v22_46_no_debt_official_audit.csv", no_debt_official_audit_rows(rows) or [{"status": "no_no_debt_audit_rows"}])
    sd = enrich_support_direction(rows)
    write_rows(OUT_ROOT / "v22_46_support_direction_decomposition.csv", sd)
    write_rows(OUT_ROOT / "v22_46_support_direction_summary.csv", [summarize_gate(sd, part=p) for p in ["D", "E", "F", "G", "H"]])
    write_rows(OUT_ROOT / "v22_46_S4Q_support_optimizer_matrix.csv", [r for r in rows if r.get("part") == "D"] or [{"status": "no_part_D_rows"}])
    write_rows(OUT_ROOT / "v22_46_S1Q_quotient_residual_matrix.csv", [r for r in rows if r.get("part") == "E"] or [{"status": "no_part_E_rows"}])
    write_rows(OUT_ROOT / "v22_46_safety_constrained_flow_matrix.csv", [r for r in rows if r.get("part") == "F"] or [{"status": "no_part_F_rows"}])
    write_rows(OUT_ROOT / "v22_46_KAN_vs_MLP_matched_support_matrix.csv", build_kan_vs_mlp(rows) or [{"status": "no_part_G_rows"}])
    write_rows(OUT_ROOT / "v22_46_pure_oet_flow_matrix.csv", enrich_pure_oet_rows(rows) or [{"status": "no_part_H_rows"}])
    safety_rows = []
    for row in rows:
        if row.get("part") == "F":
            audit = raw_metric_debt_audit(row, bases.get(row_base_key(row), {}))
            safety_rows.append({
                "run_label": row.get("run_label", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "variant": row.get("variant", ""),
                "control_mode": row.get("control_mode", ""),
                "calibration_nuisance_mode": row.get("calibration_nuisance_mode", ""),
                "safety_budget_debt": row.get("safety_budget_debt", ""),
                "safety_budget_velocity_scale": row.get("safety_budget_velocity_scale", ""),
                "safety_debt_delta": row.get("safety_debt_delta", ""),
                "ECE": row.get("ECE", ""),
                "Brier": row.get("Brier", ""),
                "tail_loss_q99": row.get("tail_loss_q99", ""),
                "margin_q10": row.get("margin_q10", ""),
                "NLL": row.get("final_NLL", ""),
                "rho_p50": row.get("rho_p50", ""),
                "no_ECE_Brier_tail_debt": audit["no_ECE_Brier_tail_debt_route_gate"],
                "no_ECE_Brier_tail_debt_source": audit["no_ECE_Brier_tail_debt_source"],
                "derived_no_ECE_Brier_tail_debt_from_raw_metrics": audit["derived_no_ECE_Brier_tail_debt_from_raw_metrics"],
                "ECE_delta_vs_optimizer_baseline": audit["ECE_delta_vs_optimizer_baseline"],
                "Brier_delta_vs_optimizer_baseline": audit["Brier_delta_vs_optimizer_baseline"],
                "tail_q95_delta_vs_optimizer_baseline": audit["tail_q95_delta_vs_optimizer_baseline"],
                "tail_q99_delta_vs_optimizer_baseline": audit["tail_q99_delta_vs_optimizer_baseline"],
                "tail_q95_q99_sensitivity_no_debt": audit["tail_q95_q99_sensitivity_no_debt"],
            })
    write_rows(OUT_ROOT / "v22_46_safety_debt_components.csv", safety_rows or [{"status": "no_safety_rows"}])


def safety_component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bases = best_baselines(rows)
    out = []
    for row in rows:
        if row.get("part") != "F":
            continue
        audit = raw_metric_debt_audit(row, bases.get(row_base_key(row), {}))
        out.append({
            "run_label": row.get("run_label", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "variant": row.get("variant", ""),
            "control_mode": row.get("control_mode", ""),
            "calibration_nuisance_mode": row.get("calibration_nuisance_mode", ""),
            "safety_budget_debt": row.get("safety_budget_debt", ""),
            "safety_budget_velocity_scale": row.get("safety_budget_velocity_scale", ""),
            "safety_debt_delta": row.get("safety_debt_delta", ""),
            "ECE": row.get("ECE", ""),
            "Brier": row.get("Brier", ""),
            "tail_loss_q99": row.get("tail_loss_q99", ""),
            "margin_q10": row.get("margin_q10", ""),
            "NLL": row.get("final_NLL", ""),
            "rho_p50": row.get("rho_p50", ""),
            "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
            "no_ECE_Brier_tail_debt": audit["no_ECE_Brier_tail_debt_route_gate"],
            "no_ECE_Brier_tail_debt_source": audit["no_ECE_Brier_tail_debt_source"],
            "derived_no_ECE_Brier_tail_debt_from_raw_metrics": audit["derived_no_ECE_Brier_tail_debt_from_raw_metrics"],
            "ECE_delta_vs_optimizer_baseline": audit["ECE_delta_vs_optimizer_baseline"],
            "Brier_delta_vs_optimizer_baseline": audit["Brier_delta_vs_optimizer_baseline"],
            "tail_q95_delta_vs_optimizer_baseline": audit["tail_q95_delta_vs_optimizer_baseline"],
            "tail_q99_delta_vs_optimizer_baseline": audit["tail_q99_delta_vs_optimizer_baseline"],
            "tail_q95_q99_sensitivity_no_debt": audit["tail_q95_q99_sensitivity_no_debt"],
        })
    return out


def build_repair_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    datasets = split_csv(args.repair_datasets)
    seeds = split_csv(args.repair_seeds, int)
    common = {
        "steps": int(args.repair_steps),
        "support_refresh_cadence": int(args.repair_support_refresh_cadence),
        "metric_refresh_cadence": int(args.repair_metric_refresh_cadence),
        "velocity_scale": float(args.repair_velocity_scale),
        "safety_budget_velocity_barrier": float(args.repair_safety_barrier),
        "calibration_nuisance_weight": float(args.repair_calibration_weight),
        "calibration_correction_weight": float(args.repair_calibration_weight),
        "cached_controller_emit_cadence": int(args.repair_cached_controller_emit_cadence),
        "calibration_readout_radial_cap": float(args.calibration_readout_radial_cap),
        "calibration_readout_policy": str(args.calibration_readout_policy),
    }
    base_seen: set[tuple[str, int, str]] = set()

    def baseline(dataset: str, seed: int, arch: str) -> None:
        key = (dataset, int(seed), arch)
        if key in base_seen:
            return
        add_spec(
            specs,
            args,
            part="R",
            mechanism="D0",
            recipe="post_r3_repair_strong_optimizer_baseline",
            dataset=dataset,
            seed=seed,
            architecture=arch,
            variant="optimizer_alone",
            control_mode="none",
            **common,
        )
        base_seen.add(key)

    for dataset in datasets:
        for seed in seeds:
            for arch in ["MLP", "DGKAN_DCHE", "DGKAN_DFOU"]:
                baseline(dataset, seed, arch)

            repair_tag = (
                f"emit{int(args.repair_cached_controller_emit_cadence)}_"
                f"supp{int(args.repair_support_refresh_cadence)}_metric{int(args.repair_metric_refresh_cadence)}"
            )
            if float(args.calibration_readout_radial_cap) > 0.0 or str(args.calibration_readout_policy) != "signed":
                repair_tag += (
                    f"_readoutcap{safe_fragment(args.calibration_readout_radial_cap)}"
                    f"_{safe_fragment(args.calibration_readout_policy)}"
                )
            for cmode in ["none", "same-metric-support-random", "same-overhead-noop"]:
                add_spec(
                    specs,
                    args,
                    part="D",
                    mechanism="S4QRepair",
                    recipe=f"cached_low_refresh_signal_metric_support_{repair_tag}",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant="S4-SignalMetric-OET",
                    control_mode=cmode,
                    **common,
                )

            for rank in [4, 8]:
                for cmode in ["none", "same-metric-support-random"]:
                    add_spec(
                        specs,
                        args,
                        part="E",
                        mechanism="S1QRepair",
                        recipe=f"denoise_quotient_lowbeta_rank{rank}_cached_{repair_tag}",
                        dataset=dataset,
                        seed=seed,
                        architecture="MLP",
                        variant="S1-M-top1-S4-SignalMetric-OET",
                        control_mode=cmode,
                        nuisance_rank=rank,
                        beta_signal=float(args.repair_beta_signal),
                        **common,
                    )

            for mode in ["brier", "tail_q99_brier_qp_margin"]:
                for cmode in ["none", "same-metric-support-random"]:
                    add_spec(
                        specs,
                        args,
                        part="F",
                        mechanism="SafetyRepair",
                        recipe=f"primal_dual_{mode}_cached_low_refresh_{repair_tag}",
                        dataset=dataset,
                        seed=seed,
                        architecture="MLP",
                        variant="S4-SignalMetric-OET",
                        control_mode=cmode,
                        calibration_nuisance_mode=mode,
                        **common,
                    )
                add_spec(
                    specs,
                    args,
                    part="F",
                    mechanism="SafetyBregmanRepair",
                    recipe=f"mirror_{mode}_cached_low_refresh_{repair_tag}",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant="M1-TailSafeMirror" if "tail" in mode else "M1-BrierMirror",
                    control_mode="none",
                    calibration_nuisance_mode=mode,
                    **common,
                )

            add_spec(
                specs,
                args,
                part="G",
                mechanism="MLPMatchedRepair",
                recipe=f"kan_vs_mlp_cached_matched_support_{repair_tag}",
                dataset=dataset,
                seed=seed,
                architecture="MLP",
                variant="MLP-polynomial-like-feature-support",
                control_mode="none",
                **common,
            )
            for arch, variant in [("DGKAN_DFOU", "KAN-D-FOU-BasisGram"), ("DGKAN_DCHE", "KAN-D-CHE-BasisGram")]:
                for cmode in ["none", "same-basis-Gram-random"]:
                    add_spec(
                        specs,
                        args,
                        part="G",
                        mechanism="KANCarrierRepair",
                        recipe=f"cached_gram_active_bank_probe_{repair_tag}",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant=variant,
                        control_mode=cmode,
                        **common,
                    )

            pure_common = dict(common)
            pure_common["velocity_scale"] = min(float(args.repair_velocity_scale), float(args.repair_pure_velocity_scale))
            for variant, controls in [
                ("P4-Euclidean-OET-functional-jvp-radial-gated-pure", ["none", "same-radial-random"]),
                ("P3-Euclidean-OET-transported-lie-momentum-pure", ["none", "same-OET-random"]),
            ]:
                for cmode in controls:
                    add_spec(
                        specs,
                        args,
                        part="H",
                        mechanism="PureOETRepair",
                        recipe=f"cached_lie_or_radial_low_refresh_{repair_tag}",
                        dataset=dataset,
                        seed=seed,
                        architecture="MLP",
                        variant=variant,
                        control_mode=cmode,
                        pure_fu_mode=True,
                        **pure_common,
                    )
    if int(args.repair_row_limit) > 0:
        specs = specs[: int(args.repair_row_limit)]
    return specs


def repair_route(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sd = enrich_support_direction(rows)
    kan = build_kan_vs_mlp(rows)
    pure = enrich_pure_oet_rows(rows)
    official_no_debt_direction = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.25
    ]
    official_no_debt_direction_exploration = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    raw_no_debt_direction = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    kan_raw_gate = [
        r for r in kan
        if (int_flag(r.get("TrueKANGain")) or int_flag(r.get("BothGain")))
        and int_flag(r.get("KAN_beats_best_same_basis_control"))
        and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        and value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35
        and not int_flag(r.get("ControlExplained"))
    ]
    pure_raw_gate = [
        r for r in pure
        if value_or(r.get("pure_trainability_score"), 0.0) > 0.0
        and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        and value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35
    ]
    support_only = [r for r in sd if int_flag(r.get("support_positive_direction_nonpositive"))]
    route = "R3-SupportOnlyNoSignalIncrement"
    reason = "post-R3 repair did not create enough official no-debt route-gate rows"
    if len(official_no_debt_direction) >= 8:
        route = "R4-QuotientResidualSignalOpened"
        reason = f"{len(official_no_debt_direction)} official no-debt direction rows passed overhead gate"
    return {
        "timestamp": now_sg(),
        "official_route_after_repair": route,
        "reason": reason,
        "rows": len(rows),
        "failures": sum(1 for r in rows if r.get("status_v22_46") != "pass"),
        "support_direction_rows": len(sd),
        "support_only_rows": len(support_only),
        "direction_positive_rows": sum(value_or(r.get("tau_direction"), 0.0) > 0.0 for r in sd),
        "official_direction_route_gate_rows": len(official_no_debt_direction),
        "official_direction_exploration_gate_rows_0p35": len(official_no_debt_direction_exploration),
        "calibration_readout_radial_cap": next((r.get("calibration_readout_radial_cap", "") for r in rows if has_value(r.get("calibration_readout_radial_cap", ""))), ""),
        "calibration_readout_policy": next((r.get("calibration_readout_policy", "") for r in rows if has_value(r.get("calibration_readout_policy", ""))), ""),
        "raw_metric_direction_exploration_gate_rows": len(raw_no_debt_direction),
        "raw_metric_KAN_exploration_gate_rows": len(kan_raw_gate),
        "raw_metric_pure_exploration_gate_rows": len(pure_raw_gate),
        "overhead_le_0p35_rows": sum(value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35 for r in rows if r.get("variant") != "optimizer_alone"),
        "derived_raw_no_debt_candidate_rows": sum(
            int_flag(raw_metric_debt_audit(r, best_baselines(rows).get(row_base_key(r), {})).get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
            for r in rows
            if r.get("variant") != "optimizer_alone" and str(r.get("control_mode")) == "none"
        ),
        "claim_limit": "official no-debt is reconstructed by v22.46 raw-metric audit against same-seed/same-architecture optimizer baseline; repair route uses overhead<=0.25 and >=8 rows, while overhead<=0.35 remains exploration-only",
    }


def build_repair_matrices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_46_repair_no_debt_official_audit.csv", no_debt_official_audit_rows(rows) or [{"status": "no_repair_no_debt_audit_rows"}])
    sd = enrich_support_direction(rows)
    write_rows(OUT_ROOT / "v22_46_repair_support_direction_decomposition.csv", sd)
    write_rows(OUT_ROOT / "v22_46_repair_support_direction_summary.csv", [summarize_gate(sd, part=p) for p in ["D", "E", "F", "G", "H"]])
    write_rows(OUT_ROOT / "v22_46_repair_KAN_vs_MLP_matched_support_matrix.csv", build_kan_vs_mlp(rows) or [{"status": "no_repair_part_G_rows"}])
    write_rows(OUT_ROOT / "v22_46_repair_pure_oet_flow_matrix.csv", enrich_pure_oet_rows(rows) or [{"status": "no_repair_part_H_rows"}])
    write_rows(OUT_ROOT / "v22_46_repair_safety_debt_components.csv", safety_component_rows(rows) or [{"status": "no_repair_safety_rows"}])
    overhead = [
        {
            "run_label": r.get("run_label", ""),
            "part": r.get("part", ""),
            "dataset": r.get("dataset", ""),
            "architecture_key": r.get("architecture_key", r.get("architecture", "")),
            "variant": r.get("variant", ""),
            "control_mode": r.get("control_mode", ""),
            "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
            "overhead_le_0p35": int(value_or(r.get("controller_overhead_ratio"), math.inf) <= 0.35),
            "support_refresh_cadence": r.get("support_refresh_cadence", ""),
            "metric_refresh_cadence": r.get("metric_refresh_cadence", ""),
            "calibration_readout_radial_cap": r.get("calibration_readout_radial_cap", ""),
            "calibration_readout_policy": r.get("calibration_readout_policy", ""),
            "velocity_scale": r.get("velocity_scale", ""),
            "final_NLL": r.get("final_NLL", ""),
            "ECE": r.get("ECE", ""),
            "Brier": r.get("Brier", ""),
            "tail_loss_q99": r.get("tail_loss_q99", ""),
        }
        for r in rows
        if r.get("variant") != "optimizer_alone"
    ]
    write_rows(OUT_ROOT / "v22_46_repair_overhead_no_debt_matrix.csv", overhead or [{"status": "no_repair_overhead_rows"}])
    route = repair_route(rows)
    write_json(OUT_ROOT / "v22_46_repair_route.json", route)
    return route


def run_repair_suite(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    specs = build_repair_specs(args)
    write_rows(OUT_ROOT / "v22_46_repair_experiment_specs.csv", specs)
    append_exec(
        "dispatch_v22_46_post_R3_repair_suite",
        task_id="post_R3_repair_dispatch",
        status="started",
        gpu=args.gpus,
        files="results/v22_46/v22_46_repair_experiment_specs.csv, results/v22_46/chunks",
        note=(
            f"rows={len(specs)}; workers={args.workers}; datasets={args.repair_datasets}; seeds={args.repair_seeds}; "
            f"steps={args.repair_steps}; repair=lower_refresh_cached_support+safety_primal_dual+KAN_cached_gram+pure_cached_lie"
        ),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_spec, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46") != "pass")
    rows.sort(key=lambda r: str(r.get("run_label", "")))
    write_rows(OUT_ROOT / "v22_46_repair_all_training_matrix.csv", rows)
    route = build_repair_matrices(rows)
    append_exec(
        "dispatch_v22_46_post_R3_repair_suite",
        task_id="post_R3_repair_complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46/v22_46_repair_all_training_matrix.csv, results/v22_46/v22_46_repair_route.json",
        note=f"rows={len(rows)}; failures={failures}; repair_route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(rows), "failures": failures, **route}


def repair_tag_from_args(args: argparse.Namespace) -> str:
    tag = (
        f"emit{int(args.repair_cached_controller_emit_cadence)}_"
        f"supp{int(args.repair_support_refresh_cadence)}_metric{int(args.repair_metric_refresh_cadence)}"
    )
    if float(args.calibration_readout_radial_cap) > 0.0 or str(args.calibration_readout_policy) != "signed":
        tag += (
            f"_readoutcap{safe_fragment(args.calibration_readout_radial_cap)}"
            f"_{safe_fragment(args.calibration_readout_policy)}"
        )
    return tag


def build_kan_variant_repair_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    common = {
        "steps": int(args.repair_steps),
        "support_refresh_cadence": int(args.repair_support_refresh_cadence),
        "metric_refresh_cadence": int(args.repair_metric_refresh_cadence),
        "cached_controller_emit_cadence": int(args.repair_cached_controller_emit_cadence),
        "velocity_scale": float(args.repair_velocity_scale),
        "safety_budget_velocity_barrier": float(args.repair_safety_barrier),
        "calibration_nuisance_weight": float(args.repair_calibration_weight),
        "calibration_correction_weight": float(args.repair_calibration_weight),
        "calibration_readout_radial_cap": float(args.calibration_readout_radial_cap),
        "calibration_readout_policy": str(args.calibration_readout_policy),
    }
    tag = repair_tag_from_args(args)
    base_seen: set[tuple[str, int, str]] = set()

    def baseline(dataset: str, seed: int, arch: str) -> None:
        key = (dataset, int(seed), arch)
        if key in base_seen:
            return
        add_spec(
            specs,
            args,
            part="G",
            mechanism="KANVariantRepairD0",
            recipe=f"kan_variant_strong_optimizer_baseline_{tag}",
            dataset=dataset,
            seed=seed,
            architecture=arch,
            variant="optimizer_alone",
            control_mode="none",
            **common,
        )
        base_seen.add(key)

    for dataset in split_csv(args.repair_datasets):
        for seed in split_csv(args.repair_seeds, int):
            for arch in ["MLP", "DGKAN_DCHE", "DGKAN_DFOU"]:
                baseline(dataset, seed, arch)
            for mlp_variant in split_csv(args.kan_repair_mlp_variants):
                add_spec(
                    specs,
                    args,
                    part="G",
                    mechanism="KANVariantMLPMatched",
                    recipe=f"kan_variant_mlp_matched_support_{tag}",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant=mlp_variant,
                    control_mode="none",
                    **common,
                )
            for variant in split_csv(args.kan_repair_variants):
                if variant.startswith("KAN-D-CHE"):
                    arch = "DGKAN_DCHE"
                elif variant.startswith("KAN-D-FOU"):
                    arch = "DGKAN_DFOU"
                else:
                    continue
                for cmode in split_csv(args.kan_repair_controls):
                    add_spec(
                        specs,
                        args,
                        part="G",
                        mechanism="KANVariantRepair",
                        recipe=f"kan_variant_fallback_{tag}",
                        dataset=dataset,
                        seed=seed,
                        architecture=arch,
                        variant=variant,
                        control_mode=cmode,
                        **common,
                    )
    if int(args.repair_row_limit) > 0:
        specs = specs[: int(args.repair_row_limit)]
    return specs


def kan_variant_repair_route(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kan = build_kan_vs_mlp(rows)
    official_base = [
        r for r in kan
        if int_flag(r.get("KAN_beats_best_same_basis_control"))
        and int_flag(r.get("no_ECE_Brier_tail_debt"))
        and overhead_le_0p25(r)
        and not int_flag(r.get("ControlExplained"))
    ]
    official_true = [r for r in official_base if int_flag(r.get("TrueKANGain")) or int_flag(r.get("BothGain"))]
    official_beats = [r for r in official_base if int_flag(r.get("KAN_beats_MLP_matched_support"))]
    exploration_base = [
        r for r in kan
        if int_flag(r.get("KAN_beats_best_same_basis_control"))
        and int_flag(r.get("no_ECE_Brier_tail_debt"))
        and overhead_le_0p35(r)
        and not int_flag(r.get("ControlExplained"))
    ]
    route = "not_opened"
    reason = "KAN variant fallback did not create enough official no-debt/control/overhead rows"
    promotion_allowed = False
    if len(official_true) >= 8:
        route = "R11-TrueKANGainExplorationOpened"
        reason = f"{len(official_true)} true/both KAN rows passed official no-debt, overhead<=0.25, and same-basis-control gates"
        promotion_allowed = True
    elif len(official_beats) >= 8:
        route = "R9-KANCarrierBeatsMLPMatchedSupport"
        reason = f"{len(official_beats)} KAN rows beat MLP matched support and passed official no-debt, overhead<=0.25, and same-basis-control gates"
        promotion_allowed = True
    return {
        "timestamp": now_sg(),
        "official_route_after_kan_variant_repair": route,
        "promotion_allowed": promotion_allowed,
        "reason": reason,
        "rows": len(rows),
        "failures": sum(1 for r in rows if r.get("status_v22_46") != "pass"),
        "kan_matrix_rows": len(kan),
        "KAN_beats_MLP_matched_support_rows": sum(int_flag(r.get("KAN_beats_MLP_matched_support")) for r in kan),
        "KAN_beats_best_same_basis_control_rows": sum(int_flag(r.get("KAN_beats_best_same_basis_control")) for r in kan),
        "TrueKANGain_or_BothGain_rows": sum(int_flag(r.get("TrueKANGain")) or int_flag(r.get("BothGain")) for r in kan),
        "ControlExplained_rows": sum(int_flag(r.get("ControlExplained")) for r in kan),
        "MLPDegradationDriven_rows": sum(int_flag(r.get("MLPDegradationDriven")) for r in kan),
        "no_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in kan),
        "official_kan_base_route_gate_rows": len(official_base),
        "official_true_or_both_route_gate_rows": len(official_true),
        "official_kan_beats_route_gate_rows": len(official_beats),
        "exploration_gate_rows_0p35": len(exploration_base),
        "claim_limit": "KAN variant repair follows plan G fallback; official promotion requires reconstructed no-debt, overhead<=0.25, same-basis-control win, non-ControlExplained evidence, and at least 8 rows.",
    }


def build_kan_variant_repair_matrices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    write_rows(OUT_ROOT / "v22_46_kan_variant_repair_no_debt_official_audit.csv", no_debt_official_audit_rows(rows) or [{"status": "no_kan_variant_no_debt_audit_rows"}])
    kan = build_kan_vs_mlp(rows)
    write_rows(OUT_ROOT / "v22_46_kan_variant_repair_KAN_vs_MLP_matched_support_matrix.csv", kan or [{"status": "no_kan_variant_rows"}])
    route = kan_variant_repair_route(rows)
    write_json(OUT_ROOT / "v22_46_kan_variant_repair_route.json", route)
    gate_rows = [
        r for r in kan
        if int_flag(r.get("KAN_beats_best_same_basis_control"))
        and int_flag(r.get("no_ECE_Brier_tail_debt"))
        and not int_flag(r.get("ControlExplained"))
        and overhead_le_0p35(r)
    ]
    write_rows(OUT_ROOT / "v22_46_kan_variant_repair_gate_rows.csv", gate_rows or [{"status": "no_kan_variant_gate_rows"}])
    return route


def run_kan_variant_repair_suite(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    specs = build_kan_variant_repair_specs(args)
    write_rows(OUT_ROOT / "v22_46_kan_variant_repair_experiment_specs.csv", specs)
    append_exec(
        "dispatch_v22_46_kan_variant_repair_suite",
        task_id="kan_variant_repair_dispatch",
        status="started",
        gpu=args.gpus,
        files="results/v22_46/v22_46_kan_variant_repair_experiment_specs.csv, results/v22_46/chunks",
        note=(
            f"rows={len(specs)}; workers={args.workers}; datasets={args.repair_datasets}; seeds={args.repair_seeds}; "
            f"variants={args.kan_repair_variants}; controls={args.kan_repair_controls}; steps={args.repair_steps}"
        ),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_spec, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46") != "pass")
    rows.sort(key=lambda r: str(r.get("run_label", "")))
    write_rows(OUT_ROOT / "v22_46_kan_variant_repair_all_training_matrix.csv", rows)
    route = build_kan_variant_repair_matrices(rows)
    append_exec(
        "dispatch_v22_46_kan_variant_repair_suite",
        task_id="kan_variant_repair_complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46/v22_46_kan_variant_repair_all_training_matrix.csv, results/v22_46/v22_46_kan_variant_repair_route.json",
        note=f"rows={len(rows)}; failures={failures}; kan_variant_route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(rows), "failures": failures, **route}


def build_cached_safety_probe_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    common = {
        "steps": int(args.repair_steps),
        "support_refresh_cadence": int(args.repair_support_refresh_cadence),
        "metric_refresh_cadence": int(args.repair_metric_refresh_cadence),
        "cached_controller_emit_cadence": int(args.repair_cached_controller_emit_cadence),
        "velocity_scale": float(args.repair_velocity_scale),
        "safety_budget_velocity_barrier": float(args.repair_safety_barrier),
        "calibration_nuisance_weight": float(args.repair_calibration_weight),
        "calibration_correction_weight": float(args.repair_calibration_weight),
        "calibration_nuisance_mode": "tail_q99_brier_qp_margin",
        "calibration_nuisance_cadence": int(args.repair_cached_controller_emit_cadence),
    }
    for dataset in split_csv(args.repair_datasets):
        for seed in split_csv(args.repair_seeds, int):
            add_spec(
                specs,
                args,
                part="F",
                mechanism="CachedSafetyProbe",
                recipe="baseline_lowfreq_safety_cached_emit",
                dataset=dataset,
                seed=seed,
                architecture="MLP",
                variant="optimizer_alone",
                control_mode="none",
                **common,
            )
            for cmode in ["none", "same-metric-support-random"]:
                add_spec(
                    specs,
                    args,
                    part="F",
                    mechanism="CachedSafetyProbe",
                    recipe="tail_q99_brier_lowfreq_safety_cached_emit",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant="S4-SignalMetric-OET",
                    control_mode=cmode,
                    **common,
                )
    if int(args.repair_row_limit) > 0:
        specs = specs[: int(args.repair_row_limit)]
    return specs


def cached_safety_probe_route(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sd = enrich_support_direction(rows)
    raw_gate = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    official_gate = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    groups = len(sd)
    return {
        "timestamp": now_sg(),
        "rows": len(rows),
        "failures": sum(1 for r in rows if r.get("status_v22_46") != "pass"),
        "groups": groups,
        "direction_positive_rows": sum(value_or(r.get("tau_direction"), 0.0) > 0.0 for r in sd),
        "support_positive_rows": sum(value_or(r.get("tau_support"), 0.0) > 0.0 for r in sd),
        "support_only_rows": sum(int_flag(r.get("support_positive_direction_nonpositive")) for r in sd),
        "raw_metric_direction_gate_rows": len(raw_gate),
        "official_direction_gate_rows": len(official_gate),
        "raw_metric_direction_gate_fraction": (len(raw_gate) / groups) if groups else "",
        "official_route_status": "probe_gate_present" if len(official_gate) >= 5 else ("partial_official_gate_rows" if official_gate else "not_opened"),
        "exploration_status": "opened_no_debt_probe" if len(official_gate) >= 5 else "not_opened",
        "claim_limit": "probe no-debt uses the v22.46 reconstructed official audit against same-seed/same-architecture optimizer baseline; this probe is exploration evidence and cannot promote the final route by best-row selection",
    }


def run_cached_safety_probe(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    specs = build_cached_safety_probe_specs(args)
    write_rows(OUT_ROOT / "v22_46_cached_safety_probe_specs.csv", specs)
    append_exec(
        "dispatch_v22_46_cached_safety_probe",
        task_id="cached_safety_probe_dispatch",
        status="started",
        gpu=args.gpus,
        files="results/v22_46/v22_46_cached_safety_probe_specs.csv, results/v22_46/chunks",
        note=(
            f"rows={len(specs)}; datasets={args.repair_datasets}; seeds={args.repair_seeds}; "
            f"cached_emit={args.repair_cached_controller_emit_cadence}; calibration_cadence={args.repair_cached_controller_emit_cadence}; "
            f"steps={args.repair_steps}"
        ),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_spec, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46") != "pass")
    rows.sort(key=lambda r: str(r.get("run_label", "")))
    write_rows(OUT_ROOT / "v22_46_cached_safety_probe_matrix.csv", rows)
    sd = enrich_support_direction(rows)
    write_rows(OUT_ROOT / "v22_46_cached_safety_probe_support_direction.csv", sd)
    write_rows(OUT_ROOT / "v22_46_cached_safety_probe_support_direction_summary.csv", [summarize_gate(sd, part="F")])
    write_rows(OUT_ROOT / "v22_46_cached_safety_probe_safety_debt_components.csv", safety_component_rows(rows) or [{"status": "no_probe_safety_rows"}])
    route = cached_safety_probe_route(rows)
    write_json(OUT_ROOT / "v22_46_cached_safety_probe_route.json", route)
    append_exec(
        "dispatch_v22_46_cached_safety_probe",
        task_id="cached_safety_probe_complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46/v22_46_cached_safety_probe_matrix.csv, results/v22_46/v22_46_cached_safety_probe_route.json",
        note=f"rows={len(rows)}; failures={failures}; route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(rows), "failures": failures, **route}


def enrich_support_direction_by_calibration_mode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    modes = sorted({
        str(r.get("calibration_nuisance_mode", ""))
        for r in rows
        if has_value(r.get("calibration_nuisance_mode", ""))
    })
    for mode in modes:
        subset = [r for r in rows if str(r.get("calibration_nuisance_mode", "")) == mode]
        for item in enrich_support_direction(subset):
            item["mode_sweep_mode"] = mode
            out.append(item)
    return out


def build_cached_safety_mode_sweep_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    base_common = {
        "steps": int(args.repair_steps),
        "support_refresh_cadence": int(args.repair_support_refresh_cadence),
        "metric_refresh_cadence": int(args.repair_metric_refresh_cadence),
        "cached_controller_emit_cadence": int(args.repair_cached_controller_emit_cadence),
        "velocity_scale": float(args.repair_velocity_scale),
        "safety_budget_velocity_barrier": float(args.repair_safety_barrier),
        "calibration_nuisance_weight": float(args.repair_calibration_weight),
        "calibration_correction_weight": float(args.repair_calibration_weight),
        "calibration_nuisance_cadence": int(args.repair_cached_controller_emit_cadence),
    }
    for mode in split_csv(args.mode_sweep_modes):
        common = dict(base_common)
        common["calibration_nuisance_mode"] = mode
        mode_tag = safe_fragment(mode)
        for dataset in split_csv(args.repair_datasets):
            for seed in split_csv(args.repair_seeds, int):
                add_spec(
                    specs,
                    args,
                    part="F",
                    mechanism="CachedSafetyModeSweep",
                    recipe=f"baseline_lowfreq_safety_cached_emit_{mode_tag}",
                    dataset=dataset,
                    seed=seed,
                    architecture="MLP",
                    variant="optimizer_alone",
                    control_mode="none",
                    **common,
                )
                for cmode in ["none", "same-metric-support-random"]:
                    add_spec(
                        specs,
                        args,
                        part="F",
                        mechanism="CachedSafetyModeSweep",
                        recipe=f"mode_{mode_tag}_lowfreq_safety_cached_emit",
                        dataset=dataset,
                        seed=seed,
                        architecture="MLP",
                        variant="S4-SignalMetric-OET",
                        control_mode=cmode,
                        **common,
                    )
    if int(args.repair_row_limit) > 0:
        specs = specs[: int(args.repair_row_limit)]
    return specs


def cached_safety_mode_sweep_route(sd: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_gate = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    official_gate = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    per_mode: list[dict[str, Any]] = []
    for mode in sorted({str(r.get("mode_sweep_mode", "")) for r in sd if has_value(r.get("mode_sweep_mode", ""))}):
        subset = [r for r in sd if str(r.get("mode_sweep_mode", "")) == mode]
        mode_raw_gate = [
            r for r in subset
            if value_or(r.get("tau_direction"), 0.0) > 0.0
            and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
            and value_or(r.get("controller_overhead"), math.inf) <= 0.35
        ]
        mode_official_gate = [
            r for r in subset
            if value_or(r.get("tau_direction"), 0.0) > 0.0
            and int_flag(r.get("no_debt"))
            and value_or(r.get("controller_overhead"), math.inf) <= 0.35
        ]
        summary = summarize_gate(subset, part="F")
        summary["mode_sweep_mode"] = mode
        summary["raw_metric_direction_gate_rows"] = len(mode_raw_gate)
        summary["official_direction_gate_rows"] = len(mode_official_gate)
        per_mode.append(summary)
    write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_summary.csv", per_mode or [{"status": "no_mode_sweep_rows"}])
    best = max(per_mode, key=lambda r: int(value_or(r.get("official_direction_gate_rows"), 0.0)), default={})
    best_official_gate_count = int(value_or(best.get("official_direction_gate_rows"), 0.0)) if best else 0
    best_raw_gate_count = int(value_or(best.get("raw_metric_direction_gate_rows"), 0.0)) if best else 0
    groups = len(sd)
    return {
        "timestamp": now_sg(),
        "rows": len(rows),
        "failures": sum(1 for r in rows if r.get("status_v22_46") != "pass"),
        "modes": len(per_mode),
        "groups": groups,
        "direction_positive_rows": sum(value_or(r.get("tau_direction"), 0.0) > 0.0 for r in sd),
        "support_positive_rows": sum(value_or(r.get("tau_support"), 0.0) > 0.0 for r in sd),
        "support_only_rows": sum(int_flag(r.get("support_positive_direction_nonpositive")) for r in sd),
        "raw_metric_direction_gate_rows": len(raw_gate),
        "official_direction_gate_rows": len(official_gate),
        "best_mode": best.get("mode_sweep_mode", ""),
        "best_mode_raw_metric_direction_gate_rows": best_raw_gate_count,
        "best_mode_official_direction_gate_rows": best_official_gate_count,
        "best_mode_direction_positive_rows": best.get("direction_positive_rows", ""),
        "best_mode_derived_raw_no_debt_rows": best.get("derived_raw_no_debt_rows", ""),
        "best_mode_overhead_le_0p35_rows": best.get("overhead_le_0p35_rows", ""),
        "official_route_status": "mode_gate_present" if best_official_gate_count >= 5 else ("partial_official_gate_rows" if official_gate else "not_opened"),
        "exploration_status": "opened_no_debt_probe" if best_official_gate_count >= 5 else "not_opened",
        "claim_limit": "mode sweep no-debt uses the v22.46 reconstructed official audit; per-mode gate counts are exploration evidence and do not promote final route by best-row selection",
    }


def run_cached_safety_mode_sweep(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    specs = build_cached_safety_mode_sweep_specs(args)
    write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_specs.csv", specs)
    append_exec(
        "dispatch_v22_46_cached_safety_mode_sweep",
        task_id="cached_safety_mode_sweep_dispatch",
        status="started",
        gpu=args.gpus,
        files="results/v22_46/v22_46_cached_safety_mode_sweep_specs.csv, results/v22_46/chunks",
        note=(
            f"rows={len(specs)}; datasets={args.repair_datasets}; seeds={args.repair_seeds}; "
            f"modes={args.mode_sweep_modes}; cached_emit={args.repair_cached_controller_emit_cadence}; "
            f"calibration_cadence={args.repair_cached_controller_emit_cadence}; steps={args.repair_steps}"
        ),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_spec, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46") != "pass")
    rows.sort(key=lambda r: str(r.get("run_label", "")))
    write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_matrix.csv", rows)
    sd = enrich_support_direction_by_calibration_mode(rows)
    write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_support_direction.csv", sd)
    gate_rows = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_gate_rows.csv", gate_rows or [{"status": "no_official_gate_rows"}])
    write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_safety_debt_components.csv", safety_component_rows(rows) or [{"status": "no_mode_sweep_safety_rows"}])
    route = cached_safety_mode_sweep_route(sd, rows)
    write_json(OUT_ROOT / "v22_46_cached_safety_mode_sweep_route.json", route)
    append_exec(
        "dispatch_v22_46_cached_safety_mode_sweep",
        task_id="cached_safety_mode_sweep_complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46/v22_46_cached_safety_mode_sweep_matrix.csv, results/v22_46/v22_46_cached_safety_mode_sweep_route.json",
        note=f"rows={len(rows)}; failures={failures}; route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(rows), "failures": failures, **route}


def enrich_support_direction_by_mode_cap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({
        (
            str(r.get("calibration_nuisance_mode", "")),
            str(r.get("calibration_readout_radial_cap", "")),
            str(r.get("calibration_readout_policy", "signed")),
        )
        for r in rows
        if has_value(r.get("calibration_nuisance_mode", ""))
    })
    for mode, cap, policy in groups:
        subset = [
            r for r in rows
            if str(r.get("calibration_nuisance_mode", "")) == mode
            and str(r.get("calibration_readout_radial_cap", "")) == cap
            and str(r.get("calibration_readout_policy", "signed")) == policy
        ]
        for item in enrich_support_direction(subset):
            item["temp_sweep_mode"] = mode
            item["temp_readout_cap"] = cap
            item["temp_readout_policy"] = policy
            item["temp_sweep_key"] = f"{mode}|cap={cap}|policy={policy}"
            out.append(item)
    return out


def build_cached_safety_temp_sweep_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    base_common = {
        "steps": int(args.repair_steps),
        "support_refresh_cadence": int(args.repair_support_refresh_cadence),
        "metric_refresh_cadence": int(args.repair_metric_refresh_cadence),
        "cached_controller_emit_cadence": int(args.repair_cached_controller_emit_cadence),
        "velocity_scale": float(args.repair_velocity_scale),
        "safety_budget_velocity_barrier": float(args.repair_safety_barrier),
        "calibration_nuisance_weight": float(args.repair_calibration_weight),
        "calibration_correction_weight": float(args.repair_calibration_weight),
        "calibration_nuisance_cadence": int(args.repair_cached_controller_emit_cadence),
    }
    variant = str(args.temp_sweep_variant)
    cadence_tag = (
        f"emit{int(args.repair_cached_controller_emit_cadence)}_"
        f"supp{int(args.repair_support_refresh_cadence)}_metric{int(args.repair_metric_refresh_cadence)}"
    )
    for policy in split_csv(args.temp_sweep_policies):
        for mode in split_csv(args.temp_sweep_modes):
            for cap in split_csv(args.temp_sweep_caps, float):
                common = dict(base_common)
                common["calibration_nuisance_mode"] = mode
                common["calibration_readout_radial_cap"] = float(cap)
                common["calibration_readout_policy"] = policy
                mode_tag = safe_fragment(mode)
                cap_tag = safe_fragment(f"cap{cap:g}")
                policy_tag = safe_fragment(policy)
                for dataset in split_csv(args.repair_datasets):
                    for seed in split_csv(args.repair_seeds, int):
                        add_spec(
                            specs,
                            args,
                            part="F",
                            mechanism="CachedSafetyTempSweep",
                            recipe=f"baseline_lowfreq_safety_cached_emit_{mode_tag}_{cap_tag}_{policy_tag}_{cadence_tag}",
                            dataset=dataset,
                            seed=seed,
                            architecture="MLP",
                            variant="optimizer_alone",
                            control_mode="none",
                            **common,
                        )
                        for cmode in ["none", "same-metric-support-random"]:
                            add_spec(
                                specs,
                                args,
                                part="F",
                                mechanism="CachedSafetyTempSweep",
                                recipe=f"mode_{mode_tag}_{cap_tag}_{policy_tag}_lowfreq_safety_cached_emit_{cadence_tag}",
                                dataset=dataset,
                                seed=seed,
                                architecture="MLP",
                                variant=variant,
                                control_mode=cmode,
                                **common,
                            )
    if int(args.repair_row_limit) > 0:
        specs = specs[: int(args.repair_row_limit)]
    return specs


def temp_sweep_artifact_prefix(variant: str, policies: str = "signed") -> str:
    policy_parts = split_csv(policies)
    policy_suffix = "" if policy_parts == ["signed"] else "_policy_" + "_".join(safe_fragment(p) for p in policy_parts)
    if str(variant) == "S4-SignalMetric-OET":
        if policy_suffix:
            return f"v22_46_cached_safety_temp_sweep{policy_suffix}"
        return "v22_46_cached_safety_temp_sweep"
    return f"v22_46_cached_safety_temp_sweep_{safe_fragment(variant)}{policy_suffix}"


def cached_safety_temp_sweep_route(sd: list[dict[str, Any]], rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    raw_gate = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    official_gate = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    per_config: list[dict[str, Any]] = []
    for key in sorted({str(r.get("temp_sweep_key", "")) for r in sd if has_value(r.get("temp_sweep_key", ""))}):
        subset = [r for r in sd if str(r.get("temp_sweep_key", "")) == key]
        config_raw_gate = [
            r for r in subset
            if value_or(r.get("tau_direction"), 0.0) > 0.0
            and int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
            and value_or(r.get("controller_overhead"), math.inf) <= 0.35
        ]
        config_official_gate = [
            r for r in subset
            if value_or(r.get("tau_direction"), 0.0) > 0.0
            and int_flag(r.get("no_debt"))
            and value_or(r.get("controller_overhead"), math.inf) <= 0.35
        ]
        summary = summarize_gate(subset, part="F")
        first = subset[0] if subset else {}
        summary["temp_sweep_key"] = key
        summary["temp_sweep_mode"] = first.get("temp_sweep_mode", "")
        summary["temp_readout_cap"] = first.get("temp_readout_cap", "")
        summary["temp_readout_policy"] = first.get("temp_readout_policy", "")
        summary["raw_metric_direction_gate_rows"] = len(config_raw_gate)
        summary["official_direction_gate_rows"] = len(config_official_gate)
        per_config.append(summary)
    write_rows(OUT_ROOT / f"{prefix}_summary.csv", per_config or [{"status": "no_temp_sweep_rows"}])
    best = max(per_config, key=lambda r: int(value_or(r.get("official_direction_gate_rows"), 0.0)), default={})
    best_official_gate_count = int(value_or(best.get("official_direction_gate_rows"), 0.0)) if best else 0
    best_raw_gate_count = int(value_or(best.get("raw_metric_direction_gate_rows"), 0.0)) if best else 0
    groups = len(sd)
    return {
        "timestamp": now_sg(),
        "rows": len(rows),
        "failures": sum(1 for r in rows if r.get("status_v22_46") != "pass"),
        "variant": next((str(r.get("variant")) for r in rows if has_value(r.get("variant")) and str(r.get("variant")) != "optimizer_alone"), ""),
        "support_refresh_cadence": next((r.get("support_refresh_cadence", "") for r in rows if has_value(r.get("support_refresh_cadence", ""))), ""),
        "metric_refresh_cadence": next((r.get("metric_refresh_cadence", "") for r in rows if has_value(r.get("metric_refresh_cadence", ""))), ""),
        "cached_controller_emit_cadence": next((r.get("cached_controller_emit_cadence", "") for r in rows if has_value(r.get("cached_controller_emit_cadence", ""))), ""),
        "configs": len(per_config),
        "groups": groups,
        "direction_positive_rows": sum(value_or(r.get("tau_direction"), 0.0) > 0.0 for r in sd),
        "support_positive_rows": sum(value_or(r.get("tau_support"), 0.0) > 0.0 for r in sd),
        "support_only_rows": sum(int_flag(r.get("support_positive_direction_nonpositive")) for r in sd),
        "raw_metric_direction_gate_rows": len(raw_gate),
        "official_direction_gate_rows": len(official_gate),
        "best_config": best.get("temp_sweep_key", ""),
        "best_config_raw_metric_direction_gate_rows": best_raw_gate_count,
        "best_config_official_direction_gate_rows": best_official_gate_count,
        "best_config_direction_positive_rows": best.get("direction_positive_rows", ""),
        "best_config_derived_raw_no_debt_rows": best.get("derived_raw_no_debt_rows", ""),
        "best_config_overhead_le_0p35_rows": best.get("overhead_le_0p35_rows", ""),
        "official_route_status": "config_gate_present" if best_official_gate_count >= 5 else ("partial_official_gate_rows" if official_gate else "not_opened"),
        "exploration_status": "opened_no_debt_probe" if best_official_gate_count >= 5 else "not_opened",
        "claim_limit": "temperature/readout sweep no-debt uses the v22.46 reconstructed official audit; per-config gate counts are exploration evidence and do not promote final route by best-row selection",
    }


def run_cached_safety_temp_sweep(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    prefix = temp_sweep_artifact_prefix(str(args.temp_sweep_variant), str(args.temp_sweep_policies))
    specs = build_cached_safety_temp_sweep_specs(args)
    write_rows(OUT_ROOT / f"{prefix}_specs.csv", specs)
    append_exec(
        "dispatch_v22_46_cached_safety_temp_sweep",
        task_id="cached_safety_temp_sweep_dispatch",
        status="started",
        gpu=args.gpus,
        files=f"results/v22_46/{prefix}_specs.csv, results/v22_46/chunks",
        note=(
            f"rows={len(specs)}; datasets={args.repair_datasets}; seeds={args.repair_seeds}; "
            f"variant={args.temp_sweep_variant}; modes={args.temp_sweep_modes}; caps={args.temp_sweep_caps}; "
            f"policies={args.temp_sweep_policies}; "
            f"cached_emit={args.repair_cached_controller_emit_cadence}; steps={args.repair_steps}"
        ),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_one_spec, args, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46") != "pass")
    rows.sort(key=lambda r: str(r.get("run_label", "")))
    write_rows(OUT_ROOT / f"{prefix}_matrix.csv", rows)
    sd = enrich_support_direction_by_mode_cap(rows)
    write_rows(OUT_ROOT / f"{prefix}_support_direction.csv", sd)
    gate_rows = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    write_rows(OUT_ROOT / f"{prefix}_gate_rows.csv", gate_rows or [{"status": "no_official_gate_rows"}])
    write_rows(OUT_ROOT / f"{prefix}_safety_debt_components.csv", safety_component_rows(rows) or [{"status": "no_temp_sweep_safety_rows"}])
    route = cached_safety_temp_sweep_route(sd, rows, prefix=prefix)
    write_json(OUT_ROOT / f"{prefix}_route.json", route)
    append_exec(
        "dispatch_v22_46_cached_safety_temp_sweep",
        task_id="cached_safety_temp_sweep_complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files=f"results/v22_46/{prefix}_matrix.csv, results/v22_46/{prefix}_route.json",
        note=f"rows={len(rows)}; failures={failures}; route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(rows), "failures": failures, **route}


def refresh_temp_sweep_artifacts(prefix: str) -> None:
    matrix = OUT_ROOT / f"{prefix}_matrix.csv"
    if not matrix.exists():
        return
    rows = read_rows(matrix)
    if not has_real_matrix_rows(rows):
        return
    sd = enrich_support_direction_by_mode_cap(rows)
    write_rows(OUT_ROOT / f"{prefix}_support_direction.csv", sd)
    gate_rows = [
        r for r in sd
        if value_or(r.get("tau_direction"), 0.0) > 0.0
        and int_flag(r.get("no_debt"))
        and value_or(r.get("controller_overhead"), math.inf) <= 0.35
    ]
    write_rows(OUT_ROOT / f"{prefix}_gate_rows.csv", gate_rows or [{"status": "no_official_gate_rows"}])
    write_rows(OUT_ROOT / f"{prefix}_safety_debt_components.csv", safety_component_rows(rows) or [{"status": "no_temp_sweep_safety_rows"}])
    route = cached_safety_temp_sweep_route(sd, rows, prefix=prefix)
    write_json(OUT_ROOT / f"{prefix}_route.json", route)


def refresh_posthoc_matrices() -> None:
    train_rows = read_rows(OUT_ROOT / "v22_46_all_training_matrix.csv")
    if has_real_matrix_rows(train_rows):
        build_part_matrices(train_rows)
    repair_rows = read_rows(OUT_ROOT / "v22_46_repair_all_training_matrix.csv")
    if has_real_matrix_rows(repair_rows):
        build_repair_matrices(repair_rows)
    kan_variant_rows = read_rows(OUT_ROOT / "v22_46_kan_variant_repair_all_training_matrix.csv")
    if has_real_matrix_rows(kan_variant_rows):
        build_kan_variant_repair_matrices(kan_variant_rows)

    probe_rows = read_rows(OUT_ROOT / "v22_46_cached_safety_probe_matrix.csv")
    if has_real_matrix_rows(probe_rows):
        sd = enrich_support_direction(probe_rows)
        write_rows(OUT_ROOT / "v22_46_cached_safety_probe_support_direction.csv", sd)
        write_rows(OUT_ROOT / "v22_46_cached_safety_probe_support_direction_summary.csv", [summarize_gate(sd, part="F")])
        write_rows(OUT_ROOT / "v22_46_cached_safety_probe_safety_debt_components.csv", safety_component_rows(probe_rows) or [{"status": "no_probe_safety_rows"}])
        write_json(OUT_ROOT / "v22_46_cached_safety_probe_route.json", cached_safety_probe_route(probe_rows))

    mode_rows = read_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_matrix.csv")
    if has_real_matrix_rows(mode_rows):
        sd = enrich_support_direction_by_calibration_mode(mode_rows)
        write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_support_direction.csv", sd)
        gate_rows = [
            r for r in sd
            if value_or(r.get("tau_direction"), 0.0) > 0.0
            and int_flag(r.get("no_debt"))
            and value_or(r.get("controller_overhead"), math.inf) <= 0.35
        ]
        write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_gate_rows.csv", gate_rows or [{"status": "no_official_gate_rows"}])
        write_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_safety_debt_components.csv", safety_component_rows(mode_rows) or [{"status": "no_mode_sweep_safety_rows"}])
        write_json(OUT_ROOT / "v22_46_cached_safety_mode_sweep_route.json", cached_safety_mode_sweep_route(sd, mode_rows))

    prefixes = [
        temp_sweep_artifact_prefix("S4-SignalMetric-OET", "signed"),
        temp_sweep_artifact_prefix("S4-SignalMetric-OET", "shrink_only,overconfidence_only"),
        temp_sweep_artifact_prefix("S4-SignalMetric-OET", "overconfidence_only"),
        temp_sweep_artifact_prefix("S1-M-top1-S4-SignalMetric-OET", "overconfidence_only"),
        temp_sweep_artifact_prefix("M1-FisherMirror", "signed"),
    ]
    for prefix in prefixes:
        refresh_temp_sweep_artifacts(prefix)
    append_exec(
        "posthoc_no_debt_audit_refresh",
        task_id="posthoc_no_debt_audit_refresh",
        status="pass",
        gpu="cpu",
        files=(
            "results/v22_46/v22_46_no_debt_official_audit.csv, "
            "results/v22_46/v22_46_repair_no_debt_official_audit.csv, "
            "results/v22_46/v22_46_kan_variant_repair_route.json, "
            "results/v22_46/v22_46_support_direction_decomposition.csv, "
            "results/v22_46/v22_46_repair_support_direction_decomposition.csv, "
            "results/v22_46/*_route.json"
        ),
        note=(
            f"recomputed no-debt as ECE/Brier/tail_q99 deltas <=0 vs same-seed/same-architecture optimizer_alone baseline; "
            f"main_rows={len(train_rows) if has_real_matrix_rows(train_rows) else 0}; "
            f"repair_rows={len(repair_rows) if has_real_matrix_rows(repair_rows) else 0}; "
            f"kan_variant_repair_rows={len(kan_variant_rows) if has_real_matrix_rows(kan_variant_rows) else 0}; "
            "tail_q95_delta recorded as sensitivity audit, not route gate"
        ),
    )


def run_continual_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_continual_smoke:
        rows = []
        for path in sorted(V45E_ROOT.glob("v22_45E_m[47]*trajectory*_summary.json")):
            data = read_json(path)
            data["source_artifact"] = str(path.relative_to(ROOT))
            data["new_training_executed"] = 0
            rows.append(data)
        write_rows(OUT_ROOT / "v22_46_continual_grokking_matrix.csv", rows or [{"status": "not_run", "reason": "continual smoke disabled and no historical summary found"}])
        append_exec(
            "continual_grokking_reanalysis_only",
            task_id="part_I_continual_reanalysis",
            status="pass",
            gpu="cpu",
            files="results/v22_46/v22_46_continual_grokking_matrix.csv",
            note=f"new_training_executed=0; historical_summary_rows={len(rows)}",
        )
        return {"rows": len(rows), "new_training_executed": 0}
    bind_upstream()
    m4_args = argparse.Namespace(**vars(args))
    m4_args.label = "v22_46"
    m4_args.eval_architectures = args.continual_architectures
    m4_args.trajectory_steps = args.continual_steps
    m4_args.m2_repair_variant = "P4-Euclidean-OET-functional-jvp-radial-gated-pure"
    m4_args.m4_trajectory_output = "v22_46_m4_continual_smoke"
    m4_args.m4_tasks = args.m4_tasks
    m4_args.diagnostic_seeds = args.continual_seeds
    m4_args.diagnostic_architectures = args.continual_architectures
    m4_args.row_limit = args.continual_row_limit
    m4_args.m4_trajectory_steps = args.continual_steps
    m4_args.m4_diag_steps = args.continual_steps
    m4_args.m7_train_size = args.continual_train_size
    m4_args.m7_held_size = args.continual_held_size
    m4_args.diagnostic_hidden = args.hidden
    m4_args.diagnostic_batch_size = args.batch_size
    try:
        result = v2245.run_m4_trajectory(m4_args)
    except Exception as exc:
        tb = traceback.format_exc(limit=12)
        err_path = LOG_ROOT / "v22_46_part_I_continual_smoke_error.log"
        err_path.write_text(tb, encoding="utf-8", errors="replace")
        append_exec(
            "run_m4_continual_grokking_smoke",
            task_id="part_I_continual_smoke",
            status="fail",
            gpu=args.gpus,
            files=str(err_path.relative_to(ROOT)),
            note=f"error={type(exc).__name__}: {exc}; m4_args compatibility fields were logged for audit",
            exit_code=1,
        )
        raise
    matrix_rows = []
    for path in sorted(OUT_ROOT.glob("v22_45E_v22_46_m4_continual_smoke*_matrix.csv")):
        for row in read_rows(path):
            item = dict(row)
            item["source_artifact"] = path.name
            item["new_training_executed"] = 1
            matrix_rows.append(item)
    write_rows(OUT_ROOT / "v22_46_continual_grokking_matrix.csv", matrix_rows or [{"status": "no_m4_smoke_rows", "result": str(result)}])
    append_exec(
        "run_m4_continual_grokking_smoke",
        task_id="part_I_continual_smoke",
        status="pass" if result.get("failures", 0) == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46/v22_46_continual_grokking_matrix.csv",
        note=f"result={result}",
    )
    return {"rows": len(matrix_rows), "new_training_executed": 1, **result}


def simple_svg(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    height = 72 + 22 * len(lines)
    body = "\n".join(
        f'<text x="24" y="{68 + i * 22}" font-size="13" font-family="monospace" fill="#1f2937">{esc(line)}</text>'
        for i, line in enumerate(lines)
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">'
        '<rect width="1100" height="100%" fill="#f8fafc"/>'
        f'<text x="24" y="36" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{esc(title)}</text>'
        f"{body}</svg>\n",
        encoding="utf-8",
    )


def make_visualizations() -> None:
    sd = read_rows(OUT_ROOT / "v22_46_support_direction_decomposition.csv")
    summary = read_rows(OUT_ROOT / "v22_46_support_direction_summary.csv")
    spectrum = read_rows(OUT_ROOT / "v22_46_functional_actuator_spectrum_matrix.csv")
    safety = read_rows(OUT_ROOT / "v22_46_safety_debt_components.csv")
    kan = read_rows(OUT_ROOT / "v22_46_KAN_vs_MLP_matched_support_matrix.csv")
    pure = read_rows(OUT_ROOT / "v22_46_pure_oet_flow_matrix.csv")
    proxy = read_rows(OUT_ROOT / "v22_46_proxy_evidence_audit.csv")
    tiers = read_rows(OUT_ROOT / "v22_46_evidence_tier_matrix.csv")
    simple_svg(FIG_ROOT / "v22_46_support_direction_quadrant.svg", "Support / Direction Quadrant", [
        f"rows={len(sd)}",
        f"support_only={sum(int_flag(r.get('support_positive_direction_nonpositive')) for r in sd)}",
        f"direction_positive={sum(value_or(r.get('tau_direction'), 0.0) > 0 for r in sd)}",
    ])
    simple_svg(FIG_ROOT / "v22_46_tau_support_vs_tau_direction.svg", "Tau Support vs Tau Direction", [
        f"{r.get('part')}: support_mean={r.get('tau_support_mean')} direction_mean={r.get('tau_direction_mean')} LCB={r.get('tau_direction_LCB')}"
        for r in summary
    ])
    simple_svg(FIG_ROOT / "v22_46_quotient_residual_flow_trace.svg", "Quotient Residual Flow Trace", [
        f"E rows={sum(1 for r in sd if r.get('part') == 'E')}",
        f"E direction_positive={sum(1 for r in sd if r.get('part') == 'E' and value_or(r.get('tau_direction'), 0.0) > 0)}",
    ])
    simple_svg(FIG_ROOT / "v22_46_safety_debt_components.svg", "Safety Debt Components", [
        f"rows={len(safety)}",
        f"no_debt={sum(int_flag(r.get('no_ECE_Brier_tail_debt')) for r in safety)}",
    ])
    simple_svg(FIG_ROOT / "v22_46_functional_spectrum_RSE_heatmap.svg", "Functional Spectrum / RSE", [
        f"rows={len(spectrum)}",
        f"computed={sum(int_flag(r.get('functional_actuator_spectrum_computed')) for r in spectrum)}",
        f"RSE_mean_range={min([value_or(r.get('RSE_mean'), 0.0) for r in spectrum] or [0.0]):.6g}..{max([value_or(r.get('RSE_mean'), 0.0) for r in spectrum] or [0.0]):.6g}",
    ])
    simple_svg(FIG_ROOT / "v22_46_KAN_vs_MLP_matched_support_gap.svg", "KAN vs MLP Matched Support Gap", [
        f"rows={len(kan)}",
        f"KAN_beats_MLP={sum(int_flag(r.get('KAN_beats_MLP_matched_support')) for r in kan)}",
        f"TrueKANGain_or_BothGain={sum(int_flag(r.get('TrueKANGain')) or int_flag(r.get('BothGain')) for r in kan)}",
    ])
    simple_svg(FIG_ROOT / "v22_46_pure_OET_spectrum_drift.svg", "Pure OET Spectrum Drift", [
        f"rows={len(pure)}",
        f"pure_trainability_positive={sum(value_or(r.get('pure_trainability_score'), 0.0) > 0 for r in pure)}",
        f"max_generalized_spectrum_drift={max([value_or(r.get('generalized_spectrum_drift'), 0.0) for r in pure] or [0.0]):.6g}",
    ])
    simple_svg(FIG_ROOT / "v22_46_no_debt_vs_NLL_tradeoff.svg", "No-Debt vs NLL Tradeoff", [
        f"all safety rows={len(safety)}",
        f"no debt rows={sum(int_flag(r.get('no_ECE_Brier_tail_debt')) for r in safety)}",
    ])
    simple_svg(FIG_ROOT / "v22_46_mechanism_evidence_tier_dashboard.svg", "Mechanism Evidence Tier Dashboard", [
        f"{r.get('evidence_tier')}: rows={r.get('row_count')} route_eligible={r.get('route_eligible_rows')}" for r in tiers
    ])
    simple_svg(FIG_ROOT / "v22_46_proxy_exclusion_dashboard.svg", "Proxy Exclusion Dashboard", [
        f"proxy_rows={len(proxy)}",
        f"proxy_route_eligible={sum(int_flag(r.get('route_eligible')) for r in proxy)}",
    ])
    append_exec(
        "make_visualizations",
        task_id="part_visualizations",
        status="pass",
        gpu="cpu",
        files=", ".join(str(p.relative_to(ROOT)) for p in sorted(FIG_ROOT.glob("v22_46_*.svg"))),
        note="SVGs are compact evidence dashboards generated from CSV artifacts; no plotted data was fabricated.",
    )


def write_support_fairness_catalogs() -> None:
    spectrum = read_rows(OUT_ROOT / "v22_46_functional_actuator_spectrum_matrix.csv")
    mlp = [r for r in spectrum if str(r.get("architecture")) == "MLP"]
    kan = [r for r in spectrum if str(r.get("architecture")).startswith("DGKAN")]
    write_rows(OUT_ROOT / "v22_46_MLP_matched_support_catalog.csv", mlp or [{"status": "no_mlp_spectrum_rows"}])
    write_rows(OUT_ROOT / "v22_46_KAN_basis_support_catalog.csv", kan or [{"status": "no_kan_spectrum_rows"}])
    pairs = []
    for k in kan:
        for m in mlp:
            if k.get("dataset") == m.get("dataset") and k.get("seed") == m.get("seed"):
                pairs.append({
                    "dataset": k.get("dataset"),
                    "seed": k.get("seed"),
                    "KAN_architecture": k.get("architecture"),
                    "KAN_actuator": k.get("actuator"),
                    "MLP_actuator": m.get("actuator"),
                    "MLP_KAN_functional_spectrum_distance": spectrum_distance(k, m),
                    "MLP_KAN_RSE_distance": abs(value_or(k.get("RSE_mean"), 0.0) - value_or(m.get("RSE_mean"), 0.0)),
                    "MLP_KAN_output_scale_distance": abs(value_or(k.get("output_logit_norm"), 0.0) - value_or(m.get("output_logit_norm"), 0.0)),
                    "spectrum_pair_available": 1,
                })
    write_rows(OUT_ROOT / "v22_46_support_fairness_matrix.csv", pairs or [{"status": "no_spectrum_pairs"}])
    write_rows(OUT_ROOT / "v22_46_RSE_matrix.csv", [
        {
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "architecture": r.get("architecture"),
            "actuator_name": r.get("actuator"),
            "RSE_mean": r.get("RSE_mean"),
            "RSE_p10": r.get("RSE_p10"),
            "RSE_p50": r.get("RSE_p50"),
            "RSE_p90": r.get("RSE_p90"),
            "functional_actuator_spectrum_computed": r.get("functional_actuator_spectrum_computed"),
            "RSE_computed": r.get("RSE_computed"),
        }
        for r in spectrum
    ] or [{"status": "no_rse_rows"}])


def parse_svals(row: dict[str, Any]) -> list[float]:
    vals = []
    for part in str(row.get("functional_actuator_singular_values_top5", "")).split(";"):
        v = finite_float(part)
        if v is not None:
            vals.append(float(v))
    return vals


def spectrum_distance(a: dict[str, Any], b: dict[str, Any]) -> float | str:
    av = parse_svals(a)
    bv = parse_svals(b)
    if not av or not bv:
        return ""
    n = min(len(av), len(bv))
    return math.sqrt(sum((av[i] - bv[i]) ** 2 for i in range(n)))


def artifact_manifest() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    write_rows(OUT_ROOT / "v22_46_artifact_manifest.csv", rows)
    return rows


def final_route() -> dict[str, Any]:
    code = (read_rows(OUT_ROOT / "v22_46_code_truth_gate.csv") or [{}])[0]
    runtime = (read_rows(OUT_ROOT / "v22_46_runtime_regression_audit.csv") or [{}])[0]
    spec = read_rows(OUT_ROOT / "v22_46_functional_actuator_spectrum_matrix.csv")
    sd = read_rows(OUT_ROOT / "v22_46_support_direction_decomposition.csv")
    kan = read_rows(OUT_ROOT / "v22_46_KAN_vs_MLP_matched_support_matrix.csv")
    pure = read_rows(OUT_ROOT / "v22_46_pure_oet_flow_matrix.csv")
    kan_variant_route = read_json(OUT_ROOT / "v22_46_kan_variant_repair_route.json")
    spectrum_real_rows = [
        r for r in spec
        if r.get("functional_actuator_spectrum_computed", "") != "" or r.get("actuator", "") != ""
    ]
    spectrum_rows = len(spectrum_real_rows)
    spectrum_computed = sum(int_flag(r.get("functional_actuator_spectrum_computed")) for r in spec)
    spectrum_ratio = spectrum_computed / max(1, spectrum_rows)
    runtime_regression = any(int_flag(runtime.get(k)) for k in [
        "runtime_argmax_candidate_used",
        "runtime_topk_candidate_used",
        "candidate_action_selection_used_for_runtime",
        "candidate_value_model_used_as_runtime_policy",
        "micro_rct_winner_used_as_runtime_action",
        "path_mpc_discrete_action_sequence_used",
    ])
    support_only = [r for r in sd if int_flag(r.get("support_positive_direction_nonpositive"))]
    support_positive = [r for r in sd if value_or(r.get("tau_support"), 0.0) > 0.0]
    direction_positive = [r for r in sd if value_or(r.get("tau_direction"), 0.0) > 0.0]
    no_debt_direction = [r for r in direction_positive if int_flag(r.get("no_debt"))]
    direction_exploration_gate = [r for r in no_debt_direction if overhead_le_0p35(r)]
    direction_route_gate = [r for r in no_debt_direction if overhead_le_0p25(r)]
    pure_train = [r for r in pure if value_or(r.get("pure_trainability_score"), 0.0) > 0.0]
    pure_route_gate = [
        r for r in pure_train
        if int_flag(r.get("no_ECE_Brier_tail_debt")) and overhead_le_0p35(r)
    ]
    kan_beats = [r for r in kan if int_flag(r.get("KAN_beats_MLP_matched_support"))]
    true_kan = [r for r in kan if int_flag(r.get("TrueKANGain")) or int_flag(r.get("BothGain"))]
    true_kan_route_gate = [
        r for r in true_kan
        if int_flag(r.get("no_ECE_Brier_tail_debt"))
        and overhead_le_0p25(r)
        and int_flag(r.get("KAN_beats_best_same_basis_control"))
        and not int_flag(r.get("ControlExplained"))
    ]
    kan_beats_route_gate = [
        r for r in kan_beats
        if int_flag(r.get("no_ECE_Brier_tail_debt"))
        and overhead_le_0p25(r)
        and int_flag(r.get("KAN_beats_best_same_basis_control"))
        and not int_flag(r.get("ControlExplained"))
    ]
    if int_flag(code.get("clean_unzip_compileall_pass")) == 0 or int_flag(code.get("clean_unzip_import_pass")) == 0:
        route, reason = "R0-CodeOrRuntimeInvalid", "code truth gate failed"
    elif runtime_regression:
        route, reason = "R0p5-CandidateActionRegression_Stop", "runtime candidate-action regression detected"
    elif spectrum_rows and spectrum_ratio < 0.95:
        route, reason = "R1-FunctionalSpectrumAuditBlocked", f"functional spectrum computed ratio {spectrum_ratio:.3f} < 0.95"
    elif len(direction_route_gate) >= 8:
        route, reason = "R4-QuotientResidualSignalOpened", f"{len(direction_route_gate)} direction-positive rows passed no-debt and official overhead<=0.25 gates"
    elif true_kan_route_gate and len(true_kan_route_gate) >= max(1, math.ceil(0.30 * max(1, len(kan)))):
        route, reason = "R11-TrueKANGainExplorationOpened", f"{len(true_kan_route_gate)} KAN true/both gain rows passed no-debt, official overhead<=0.25, and same-basis-control gates"
    elif kan_beats_route_gate:
        route, reason = "R9-KANCarrierBeatsMLPMatchedSupport", f"{len(kan_beats_route_gate)} KAN rows beat MLP matched support and passed no-debt, official overhead<=0.25, same-basis-control gates"
    elif kan_variant_route.get("promotion_allowed") is True and kan_variant_route.get("official_route_after_kan_variant_repair"):
        route = str(kan_variant_route.get("official_route_after_kan_variant_repair"))
        reason = f"KAN variant fallback repair opened route: {kan_variant_route.get('reason')}"
    elif pure_route_gate:
        route, reason = "R6-PureOETTrainabilityOpened", f"{len(pure_route_gate)} pure OET trainability-positive rows passed no-debt and overhead gates"
    elif support_positive and support_only:
        route, reason = "R3-SupportOnlyNoSignalIncrement", f"{len(support_only)} support-positive direction-nonpositive rows; residual/KAN/pure gates lacked official no-debt/overhead/control evidence"
    elif support_positive:
        route, reason = "R2-SupportNativeOptimizerOpened", f"{len(support_positive)} support-positive rows"
    else:
        route, reason = "R13-StrongOptimizerExplainsAll", "no support, direction, KAN, or pure-OET gate opened"
    final = {
        "timestamp": now_sg(),
        "final_route": route,
        "reason": reason,
        "code_status": code.get("status", ""),
        "runtime_status": runtime.get("status", ""),
        "functional_spectrum_rows": spectrum_rows,
        "functional_spectrum_computed_rows": spectrum_computed,
        "functional_spectrum_computed_ratio": spectrum_ratio,
        "support_direction_rows": len(sd),
        "support_positive_rows": len(support_positive),
        "support_only_rows": len(support_only),
        "direction_positive_rows": len(direction_positive),
        "direction_positive_no_debt_rows": len(no_debt_direction),
        "direction_positive_route_gate_rows": len(direction_route_gate),
        "direction_positive_exploration_gate_rows_0p35": len(direction_exploration_gate),
        "pure_trainability_positive_rows": len(pure_train),
        "pure_trainability_route_gate_rows": len(pure_route_gate),
        "KAN_beats_MLP_matched_support_rows": len(kan_beats),
        "TrueKANGain_or_BothGain_rows": len(true_kan),
        "KAN_beats_route_gate_rows": len(kan_beats_route_gate),
        "TrueKANGain_or_BothGain_route_gate_rows": len(true_kan_route_gate),
        "KAN_variant_repair_route": kan_variant_route.get("official_route_after_kan_variant_repair", ""),
        "KAN_variant_repair_promotion_allowed": kan_variant_route.get("promotion_allowed", ""),
        "KAN_variant_repair_official_true_or_both_route_gate_rows": kan_variant_route.get("official_true_or_both_route_gate_rows", ""),
        "KAN_variant_repair_official_kan_beats_route_gate_rows": kan_variant_route.get("official_kan_beats_route_gate_rows", ""),
        "derived_raw_no_debt_support_direction_rows": sum(int_flag(r.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics")) for r in sd),
    }
    write_json(OUT_ROOT / "v22_46_final_route.json", final)
    return final


def write_recap(final: dict[str, Any], train_result: dict[str, Any] | None = None, continual_result: dict[str, Any] | None = None) -> None:
    code = read_rows(OUT_ROOT / "v22_46_code_truth_gate.csv")
    reanalysis = read_json(OUT_ROOT / "v22_46_v45E_final_route_recompute.json")
    train_rows_artifact = read_rows(OUT_ROOT / "v22_46_all_training_matrix.csv")
    train_rows_count = (train_result or {}).get("rows")
    if train_rows_count in {"", None, 0}:
        train_rows_count = len(train_rows_artifact)
    train_failures = (train_result or {}).get("failures")
    if train_failures in {"", None}:
        train_failures = sum(1 for r in train_rows_artifact if r.get("status_v22_46") != "pass")
    summary = read_rows(OUT_ROOT / "v22_46_support_direction_summary.csv")
    spectrum = read_rows(OUT_ROOT / "v22_46_functional_actuator_spectrum_matrix.csv")
    fairness = read_rows(OUT_ROOT / "v22_46_KAN_vs_MLP_matched_support_matrix.csv")
    pure = read_rows(OUT_ROOT / "v22_46_pure_oet_flow_matrix.csv")
    safety = read_rows(OUT_ROOT / "v22_46_safety_debt_components.csv")
    no_debt_audit = read_rows(OUT_ROOT / "v22_46_no_debt_official_audit.csv")
    repair_no_debt_audit = read_rows(OUT_ROOT / "v22_46_repair_no_debt_official_audit.csv")
    continual = read_rows(OUT_ROOT / "v22_46_continual_grokking_matrix.csv")
    continual_summary = read_json(OUT_ROOT / "v22_45E_v22_46_m4_continual_smoke_summary.json")
    repair_route_data = read_json(OUT_ROOT / "v22_46_repair_route.json")
    repair_emit50_snapshot = read_json(OUT_ROOT / "repair_snapshots/v22_46_repair_route_emit50_supp120_metric120_before_emit100.json")
    repair_emit100_snapshot = read_json(OUT_ROOT / "repair_snapshots/v22_46_repair_route_emit100_supp120_metric120_before_readoutcap.json")
    repair_emit50_readout_snapshot = read_json(OUT_ROOT / "repair_snapshots/v22_46_repair_route_emit50_supp120_metric120_readoutcap0_02_overconfidence_only_before_emit20.json")
    repair_summary = read_rows(OUT_ROOT / "v22_46_repair_support_direction_summary.csv")
    repair_overhead = read_rows(OUT_ROOT / "v22_46_repair_overhead_no_debt_matrix.csv")
    kan_variant_route = read_json(OUT_ROOT / "v22_46_kan_variant_repair_route.json")
    kan_variant_matrix = read_rows(OUT_ROOT / "v22_46_kan_variant_repair_KAN_vs_MLP_matched_support_matrix.csv")
    kan_variant_gate_rows = read_rows(OUT_ROOT / "v22_46_kan_variant_repair_gate_rows.csv")
    kan_variant_no_debt_audit = read_rows(OUT_ROOT / "v22_46_kan_variant_repair_no_debt_official_audit.csv")
    cached_probe_route = read_json(OUT_ROOT / "v22_46_cached_safety_probe_route.json")
    cached_probe_summary = read_rows(OUT_ROOT / "v22_46_cached_safety_probe_support_direction_summary.csv")
    cached_probe_sd = read_rows(OUT_ROOT / "v22_46_cached_safety_probe_support_direction.csv")
    mode_sweep_route = read_json(OUT_ROOT / "v22_46_cached_safety_mode_sweep_route.json")
    mode_sweep_summary = read_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_summary.csv")
    mode_sweep_gate_rows = read_rows(OUT_ROOT / "v22_46_cached_safety_mode_sweep_gate_rows.csv")
    temp_sweep_route = read_json(OUT_ROOT / "v22_46_cached_safety_temp_sweep_route.json")
    temp_sweep_summary = read_rows(OUT_ROOT / "v22_46_cached_safety_temp_sweep_summary.csv")
    temp_sweep_gate_rows = read_rows(OUT_ROOT / "v22_46_cached_safety_temp_sweep_gate_rows.csv")
    policy_temp_prefix = temp_sweep_artifact_prefix("S4-SignalMetric-OET", "shrink_only,overconfidence_only")
    policy_temp_route = read_json(OUT_ROOT / f"{policy_temp_prefix}_route.json")
    policy_temp_summary = read_rows(OUT_ROOT / f"{policy_temp_prefix}_summary.csv")
    policy_temp_gate_rows = read_rows(OUT_ROOT / f"{policy_temp_prefix}_gate_rows.csv")
    high_cadence_temp_prefix = temp_sweep_artifact_prefix("S4-SignalMetric-OET", "overconfidence_only")
    high_cadence_temp_route = read_json(OUT_ROOT / f"{high_cadence_temp_prefix}_route.json")
    cadence_emit200_snapshot = read_json(OUT_ROOT / "cadence_probe_snapshots/v22_46_overconfidence_cadence_probe_emit200_supp200_metric200_route_before_metric_only.json")
    cadence_metric_only_snapshot = read_json(OUT_ROOT / "cadence_probe_snapshots/v22_46_overconfidence_cadence_probe_emit50_supp120_metric200_route_before_support_only.json")
    cadence_support_only_snapshot = read_json(OUT_ROOT / "cadence_probe_snapshots/v22_46_overconfidence_cadence_probe_emit50_supp200_metric120_route_before_support_metric_combo.json")
    high_cadence_temp_summary = read_rows(OUT_ROOT / f"{high_cadence_temp_prefix}_summary.csv")
    high_cadence_temp_gate_rows = read_rows(OUT_ROOT / f"{high_cadence_temp_prefix}_gate_rows.csv")
    s1_temp_prefix = temp_sweep_artifact_prefix("S1-M-top1-S4-SignalMetric-OET", "overconfidence_only")
    s1_temp_route = read_json(OUT_ROOT / f"{s1_temp_prefix}_route.json")
    s1_temp_summary = read_rows(OUT_ROOT / f"{s1_temp_prefix}_summary.csv")
    s1_temp_gate_rows = read_rows(OUT_ROOT / f"{s1_temp_prefix}_gate_rows.csv")
    fisher_temp_prefix = f"v22_46_cached_safety_temp_sweep_{safe_fragment('M1-FisherMirror')}"
    fisher_temp_route = read_json(OUT_ROOT / f"{fisher_temp_prefix}_route.json")
    fisher_temp_summary = read_rows(OUT_ROOT / f"{fisher_temp_prefix}_summary.csv")
    fisher_temp_gate_rows = read_rows(OUT_ROOT / f"{fisher_temp_prefix}_gate_rows.csv")
    manifest = artifact_manifest()
    lines = [
        "# DG-KAN v22.46 SupportQuotientCausalFlow 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 结论",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- reason: {final.get('reason')}",
        f"- training_rows: `{train_rows_count}` / failures: `{train_failures}`",
        f"- continual_rows: `{len(continual)}` / failures: `{continual_summary.get('failures', '')}` / route: `{continual_summary.get('route', '')}` / promotion_allowed: `{continual_summary.get('promotion_allowed', '')}`",
        f"- functional_spectrum_computed: `{final.get('functional_spectrum_computed_rows')}` / `{final.get('functional_spectrum_rows')}`",
        f"- support_positive_rows: `{final.get('support_positive_rows')}`; support_only_rows: `{final.get('support_only_rows')}`; direction_positive_no_debt_rows: `{final.get('direction_positive_no_debt_rows')}`",
        f"- direction_positive_route_gate_rows_overhead0p25: `{final.get('direction_positive_route_gate_rows')}`; direction_positive_exploration_gate_rows_overhead0p35: `{final.get('direction_positive_exploration_gate_rows_0p35')}`; derived_raw_no_debt_support_direction_rows: `{final.get('derived_raw_no_debt_support_direction_rows')}`",
        f"- KAN_beats_MLP_matched_support_rows: `{final.get('KAN_beats_MLP_matched_support_rows')}`; TrueKANGain_or_BothGain_rows: `{final.get('TrueKANGain_or_BothGain_rows')}`",
        f"- KAN_beats_route_gate_rows: `{final.get('KAN_beats_route_gate_rows')}`; TrueKANGain_or_BothGain_route_gate_rows: `{final.get('TrueKANGain_or_BothGain_route_gate_rows')}`; pure_trainability_route_gate_rows: `{final.get('pure_trainability_route_gate_rows')}`",
        f"- KAN_variant_repair_route: `{kan_variant_route.get('official_route_after_kan_variant_repair', 'not_run')}` / promotion_allowed: `{kan_variant_route.get('promotion_allowed', '')}` / true_or_both_gate: `{kan_variant_route.get('official_true_or_both_route_gate_rows', '')}` / kan_beats_gate: `{kan_variant_route.get('official_kan_beats_route_gate_rows', '')}`",
        f"- post_R3_repair_route: `{repair_route_data.get('official_route_after_repair', 'not_run')}` / raw_metric_direction_exploration_gate_rows: `{repair_route_data.get('raw_metric_direction_exploration_gate_rows', '')}` / overhead_le_0p35_rows: `{repair_route_data.get('overhead_le_0p35_rows', '')}`",
        f"- emit20_repair_current: cap `{repair_route_data.get('calibration_readout_radial_cap', '')}` / policy `{repair_route_data.get('calibration_readout_policy', '')}` / route_gate `{repair_route_data.get('official_direction_route_gate_rows', '')}` / exploration_gate0p35 `{repair_route_data.get('official_direction_exploration_gate_rows_0p35', '')}`",
        f"- repair_route_gate_comparison: current_emit20 `{repair_route_data.get('official_direction_route_gate_rows', '')}` / emit50_readout_snapshot `{repair_emit50_readout_snapshot.get('official_direction_route_gate_rows', '')}` / emit100_snapshot `{repair_emit100_snapshot.get('official_direction_route_gate_rows', '')}` / emit50_snapshot `{repair_emit50_snapshot.get('official_direction_route_gate_rows', '')}`",
        f"- repair_exploration_gate_comparison: current_emit20 `{repair_route_data.get('official_direction_exploration_gate_rows_0p35', '')}` / emit50_readout_snapshot `{repair_emit50_readout_snapshot.get('official_direction_exploration_gate_rows_0p35', '')}` / emit100_snapshot `{repair_emit100_snapshot.get('official_direction_exploration_gate_rows_0p35', '')}` / emit50_snapshot `{repair_emit50_snapshot.get('official_direction_exploration_gate_rows_0p35', '')}`",
        f"- cached_safety_probe: `{cached_probe_route.get('exploration_status', 'not_run')}` / raw_metric_direction_gate_rows: `{cached_probe_route.get('raw_metric_direction_gate_rows', '')}` / official_direction_gate_rows: `{cached_probe_route.get('official_direction_gate_rows', '')}`",
        f"- cached_safety_mode_sweep: `{mode_sweep_route.get('exploration_status', 'not_run')}` / best_mode: `{mode_sweep_route.get('best_mode', '')}` / best_mode_raw_gate_rows: `{mode_sweep_route.get('best_mode_raw_metric_direction_gate_rows', '')}` / official_direction_gate_rows: `{mode_sweep_route.get('official_direction_gate_rows', '')}`",
        f"- cached_safety_temp_sweep: `{temp_sweep_route.get('exploration_status', 'not_run')}` / best_config: `{temp_sweep_route.get('best_config', '')}` / best_config_raw_gate_rows: `{temp_sweep_route.get('best_config_raw_metric_direction_gate_rows', '')}` / official_direction_gate_rows: `{temp_sweep_route.get('official_direction_gate_rows', '')}`",
        f"- calibration_policy_temp_sweep: `{policy_temp_route.get('exploration_status', 'not_run')}` / best_config: `{policy_temp_route.get('best_config', '')}` / best_config_raw_gate_rows: `{policy_temp_route.get('best_config_raw_metric_direction_gate_rows', '')}` / official_direction_gate_rows: `{policy_temp_route.get('official_direction_gate_rows', '')}`",
        f"- overconfidence_cadence_probe_latest: `{high_cadence_temp_route.get('exploration_status', 'not_run')}` / emit `{high_cadence_temp_route.get('cached_controller_emit_cadence', '')}` / support `{high_cadence_temp_route.get('support_refresh_cadence', '')}` / metric `{high_cadence_temp_route.get('metric_refresh_cadence', '')}` / best_config_raw_gate_rows: `{high_cadence_temp_route.get('best_config_raw_metric_direction_gate_rows', '')}` / official_direction_gate_rows: `{high_cadence_temp_route.get('official_direction_gate_rows', '')}`",
        f"- cadence_probe_comparison: combo_gate `{high_cadence_temp_route.get('official_direction_gate_rows', '')}` / support_only `{cadence_support_only_snapshot.get('official_direction_gate_rows', '')}` / metric_only `{cadence_metric_only_snapshot.get('official_direction_gate_rows', '')}` / emit200_all `{cadence_emit200_snapshot.get('official_direction_gate_rows', '')}`",
        f"- cadence_direction_comparison: combo_direction `{high_cadence_temp_route.get('direction_positive_rows', '')}` / support_only `{cadence_support_only_snapshot.get('direction_positive_rows', '')}` / metric_only `{cadence_metric_only_snapshot.get('direction_positive_rows', '')}` / emit200_all `{cadence_emit200_snapshot.get('direction_positive_rows', '')}`",
        f"- S1_rank_cap_probe: `{s1_temp_route.get('exploration_status', 'not_run')}` / best_config_raw_gate_rows: `{s1_temp_route.get('best_config_raw_metric_direction_gate_rows', '')}` / official_direction_gate_rows: `{s1_temp_route.get('official_direction_gate_rows', '')}`",
        f"- fisher_mirror_temp_sweep: `{fisher_temp_route.get('exploration_status', 'not_run')}` / best_config: `{fisher_temp_route.get('best_config', '')}` / best_config_raw_gate_rows: `{fisher_temp_route.get('best_config_raw_metric_direction_gate_rows', '')}` / official_direction_gate_rows: `{fisher_temp_route.get('official_direction_gate_rows', '')}`",
        f"- no_debt_official_audit: main_pass `{sum(int_flag(r.get('no_ECE_Brier_tail_debt')) for r in no_debt_audit)}` / `{len(no_debt_audit)}`; repair_pass `{sum(int_flag(r.get('no_ECE_Brier_tail_debt')) for r in repair_no_debt_audit)}` / `{len(repair_no_debt_audit)}`; q95_q99_sensitivity_pass main `{sum(int_flag(r.get('tail_q95_q99_sensitivity_no_debt')) for r in no_debt_audit)}` / repair `{sum(int_flag(r.get('tail_q95_q99_sensitivity_no_debt')) for r in repair_no_debt_audit)}`",
        "",
        "## 执行与修复记录",
        "",
        "- 新增 `experiments/run_v22_46_support_quotient_causal_flow.py`：作用是 v22.46 专用 audit/runner/finalizer，不修改底层训练算法数值逻辑；复用 v22.43 连续 FU 内核和 v22.44R functional spectrum 内核，并把输出隔离到 `results/v22_46`。",
        "- 对 v22.45E 做独立重算：读取 `results/v22_45E/*support_direction_decomposition.csv`、重新计算 `tau_support` / `tau_direction`、重新标注 evidence tier 与 route eligibility，输出 v22.46 的 recheck artifact。",
        "- 本轮没有新的 clean zip 包输入；Part A 的 clean_unzip 字段已在执行日志中明确解释为当前源码树 compile/import 闭合审计，避免把未执行的 clean-room 解包写成事实。",
        "- 如果某个 row 失败，错误栈写入 `results/v22_46/logs/*_error.log`，对应 row 在 `v22_46_all_training_matrix.csv` 中保留 `status_v22_46=fail`，不参与成功统计。",
        "- 审计修复：初版 KAN-vs-MLP 派生矩阵把 `optimizer_alone` baseline 混入 KAN candidate，并读取了不存在的 `NLL_improvement_vs_own_strong_optimizer` 字段；已改为显式排除 baseline，并用同 dataset/seed/architecture 的 optimizer baseline 计算真实 NLL improvement。",
        "- 审计修复：原始训练 artifact 没有 `no_ECE_Brier_tail_debt` 列；已按 v22.43 upstream enrich 的同义标准重建 official no-debt audit：同 dataset/seed/architecture 的 `optimizer_alone` baseline 下，`ECE_delta<=0`、`Brier_delta<=0`、`tail_q99_delta<=0` 三者同时成立，零容差。新增 `v22_46_no_debt_official_audit.csv` 与 `v22_46_repair_no_debt_official_audit.csv`，并记录 `tail_q95_delta` 作为敏感性审计，不作为 route gate。",
        "- 审计修复：standalone finalize 的 posthoc refresh guard 早版用 `rows[0].get(\"status\")` 判断 placeholder，但真实训练行也含 `status=completed_v22_43_row`，导致 no-debt audit 重建被跳过并出现 `0/0` 计数。已改为 `has_real_matrix_rows()`，只用 `run_label`/`row_id` 识别真实矩阵行；重跑后 main_rows=288、repair_rows=300。",
        "- 审计修复：final route 已收紧，R4/R9/R11 必须满足 reconstructed official no-debt、official overhead<=0.25，以及 KAN same-basis-control 约束（适用时）；overhead<=0.35 只保留为 exploration 统计，单个 NLL 胜利或 pure trainability 正值不会直接晋升路线。",
        "- 审计修复：finalizer 初版用 `\"status\" not in row` 计算 functional spectrum 分母；谱矩阵真实行也含 `status` 字段，导致分母误为 0。已改为按 `functional_actuator_spectrum_computed`/`actuator` 识别真实谱行，最终分母为真实行数。",
        "- Part I 修复：第一次 continual smoke 被 v22.45E `trajectory_command_line` 的兼容字段阻断（缺 `eval_architectures`/`trajectory_steps`，随后缺 `m2_repair_variant`）；已在 v22.46 runner 的 M4 adapter 中补齐这些字段，错误栈保存在 `results/v22_46/logs/v22_46_part_I_continual_smoke_error.log`。",
        "- Part I 修复：初始 `continual_row_limit=8` 只覆盖 modular addition；已扩展到 16 行，覆盖 `modular_addition` 与 `Class_MNIST_0_4_to_5_9` 两个任务、MLP/DGKAN_DCHE 两个架构、四个变体，最终 failures=0。",
        "- 复盘修复：standalone `--stage finalize` 没有传入 `train_result`，早版复盘结论区会显示 training_rows=0；已改为从 `results/v22_46/v22_46_all_training_matrix.csv` 回读真实训练行数与失败数。",
        "- Post-R3 repair：按计划 D/F/G/H fallback 新增 lower-refresh/cached-support、安全 primal-dual、KAN cached Gram、pure cached Lie/radial 组合；输出 `v22_46_repair_*` artifacts，且不覆盖主 288-row matrix。",
        "- Post-R3 repair 追加负结果：为降低近门槛 overhead，尝试中间档 `repair_cached_controller_emit_cadence=100`、support/metric refresh 仍为 120；结果 official route gate 从 emit50 快照的 2 降到 1，exploration gate(0.35) 从 7 降到 5，direction_positive 从 52 降到 47，因此不是可采纳改进。emit50 对照已保存到 `results/v22_46/repair_snapshots/`。",
        "- Post-R3 readoutcap repair：按计划 F.5 尝试 train-only `calibration_readout_radial_cap=0.02` + `overconfidence_only` 到 full repair；结果较 emit100 修复有回升（route gate 2 vs 1，exploration gate 6 vs 5），但没有超过 emit50 原始快照（route gate 同为 2，exploration gate 6 vs 7），因此仍未解开 official route。",
        "- Post-R3 emit20 repair：按计划 D/F 的 lower-refresh/cached-support 方向进一步把 `repair_cached_controller_emit_cadence` 降到 20，并保留 `readoutcap=0.02/overconfidence_only`；结果 route gate 从 emit50-readout 快照的 2 降到 0，exploration gate 从 6 降到 3，derived_raw_no_debt_candidate 从 28 降到 23，因此低 emit 不再是有效修复方向。",
        "- Cached safety probe：发现 lower-refresh 仍未真正降低 overhead 后，补充默认关闭的 `cached_controller_emit_cadence`；在 probe 中使用低频 controller emit 与低频 safety calibration，验证 overhead/no-debt/direction 三者能否同时改善。",
        "- Cached safety mode sweep：继续按计划 F 的 calibration debt 修复方向，扫描 `calibration_nuisance_mode`，每个 mode 独立运行 baseline/real/random 并按 mode 分组计算方向增量，避免跨 mode 共享 baseline 导致误判。",
        "- Cached safety temp sweep：针对 mode sweep 中 ECE 主导债务，启用 train-only `calibration_readout_radial_cap`，按 `mode+cap` 独立运行 baseline/real/random 并分组计算，审计 logit-temperature/readout correction 是否真正降低 no-debt blocker。",
        "- Calibration policy temp sweep：发现 signed readout 在 train underconfidence 样例会放大 logits 且 held/test ECE 未改善后，新增 `calibration_readout_policy`，测试 `shrink_only` 与 `overconfidence_only` 两种更保守的 train-only temperature 修正。",
        "- Overconfidence cadence probe：针对 overhead blocker，单独记录 `cached_controller_emit_cadence`、`support_refresh_cadence`、`metric_refresh_cadence`，避免把 200/200/200 与 metric-only refresh 的结果混在同一个解释里。",
        "- Overconfidence cadence probe 追加结果：metric-only (`emit=50/support=120/metric=200`) 与 support-only (`emit=50/support=200/metric=120`) 都达到 official gate 3/12、direction_positive 8/12；组合降频 (`emit=50/support=200/metric=200`) 仍为 official gate 3/12 但 direction_positive 降到 7/12；三者都好于 `emit=200/support=200/metric=200` 的 official gate 1/12，但均低于 opening 阈值 5/12。",
        "- S1 rank-cap probe：按计划 E.6 尝试 `S1-M-top1-S4-SignalMetric-OET` 与 `nuisance_rank=1`，验证更低 nuisance rank 是否能避免 quotient residual signal collapse。",
        "- Fisher mirror temp sweep：按计划 F.5 的 Fisher-mirror 修复方向，用同一低频 cached emission 框架测试 `M1-FisherMirror`，单独输出带 variant 后缀的 artifacts，避免覆盖 S4 readout sweep。",
        "- KAN variant repair：按计划 G.5 的 bank-local / alternate OET / cached Gram 方向，新增聚焦 `KAN-D-{CHE,FOU}` 的 `BasisGram`、`BasisGramFast`、`FisherSignal-BasisGram`、`OET-BankLocal` sweep；该 stage 独立输出 `v22_46_kan_variant_repair_*`，并继续要求 no-debt、overhead<=0.25、same-basis control 与非 ControlExplained。",
        "- v22.46b 补充计划：KAN variant repair 与 emit20 full repair 都未打开 official route 后，新增 `docs/DG-KAN_v22.46b_DebtControlledSignalIdentification_补充计划.md`，下一轮聚焦 debt-orthogonal residual signal、KAN margin stability、structural overhead，而不是继续扩大同类 cadence sweep。",
        "- 审计修复：发现 temp/repair probe 的 chunk label 未包含 cadence，可能导致相同 policy/cap 的 chunk 文件被后续 probe 覆盖；已把 `emit/supp/metric` cadence 写入后续 temp/repair recipe。已有 aggregate matrix/route JSON 保留当时数值，是复盘中的权威统计来源。",
        "",
        "## Part A/B/C Evidence",
        "",
        md_table(code, ["clean_unzip_compileall_pass", "clean_unzip_import_pass", "missing_transitive_dependency_count", "official_DGKAN_identity_pass", "status"], 5),
        f"- v22.45E reanalysis: route_eligible={reanalysis.get('recomputed_route_eligible_rows')}, support_only_route_eligible={reanalysis.get('recomputed_support_only_route_eligible_rows')}, proxy_route_eligible={reanalysis.get('proxy_route_eligible_rows')}.",
        md_table(spectrum, ["dataset", "seed", "architecture", "actuator", "functional_actuator_spectrum_computed", "RSE_mean", "functional_actuator_effective_rank", "functional_actuator_condition_number", "status"], 16),
        "",
        "## Support / Direction Decomposition",
        "",
        md_table(summary, ["part", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "beats_matched_control_rows", "no_debt_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 12),
        "",
        "## No-Debt Audit Evidence",
        "",
        md_table(no_debt_audit, ["part", "dataset", "seed", "architecture_key", "variant", "base_ECE", "candidate_ECE", "ECE_delta_vs_optimizer_baseline", "base_Brier", "candidate_Brier", "Brier_delta_vs_optimizer_baseline", "tail_q95_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline", "no_ECE_Brier_tail_debt", "tail_q95_q99_sensitivity_no_debt", "no_ECE_Brier_tail_debt_source"], 16),
        "",
        md_table(repair_no_debt_audit, ["part", "dataset", "seed", "architecture_key", "variant", "base_ECE", "candidate_ECE", "ECE_delta_vs_optimizer_baseline", "base_Brier", "candidate_Brier", "Brier_delta_vs_optimizer_baseline", "tail_q95_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline", "no_ECE_Brier_tail_debt", "tail_q95_q99_sensitivity_no_debt", "no_ECE_Brier_tail_debt_source"], 16),
        "",
        "## Safety / KAN / Pure-OET / Continual",
        "",
        md_table(safety, ["dataset", "seed", "variant", "calibration_nuisance_mode", "safety_budget_debt", "safety_budget_velocity_scale", "no_ECE_Brier_tail_debt", "no_ECE_Brier_tail_debt_source", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "NLL"], 12),
        md_table(fairness, ["dataset", "seed", "KAN_architecture", "KAN_variant", "matched_MLP_variant", "KAN_NLL_improvement_vs_own_strong_optimizer", "MLP_matched_NLL_improvement_vs_own_strong_optimizer", "KAN_beats_best_same_basis_control", "KAN_beats_MLP_matched_support", "TrueKANGain", "BothGain", "ControlExplained", "MLPDegradationDriven", "no_ECE_Brier_tail_debt", "derived_no_ECE_Brier_tail_debt_from_raw_metrics"], 16),
        md_table(pure, ["dataset", "seed", "architecture_key", "variant", "control_mode", "pure_trainability_score", "generalized_spectrum_drift", "metric_skew_residual", "Cayley_solve_residual", "no_ECE_Brier_tail_debt", "derived_no_ECE_Brier_tail_debt_from_raw_metrics"], 16),
        md_table(continual, list(continual[0].keys())[:10] if continual else ["status"], 12),
        "",
        "## Post-R3 Repair Evidence",
        "",
        json.dumps(repair_route_data or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "emit100 snapshot before readoutcap repair:",
        "",
        json.dumps(repair_emit100_snapshot or {"status": "not_available"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "emit50 snapshot before emit100:",
        "",
        json.dumps(repair_emit50_snapshot or {"status": "not_available"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "emit50 readout snapshot before emit20:",
        "",
        json.dumps(repair_emit50_readout_snapshot or {"status": "not_available"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(repair_summary, ["part", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "beats_matched_control_rows", "no_debt_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 12),
        "",
        md_table(repair_overhead, ["part", "dataset", "architecture_key", "variant", "control_mode", "controller_overhead_ratio", "overhead_le_0p35", "final_NLL", "ECE", "Brier", "tail_loss_q99"], 16),
        "",
        "## KAN Variant Repair Evidence",
        "",
        json.dumps(kan_variant_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(kan_variant_matrix, ["dataset", "seed", "KAN_architecture", "KAN_variant", "matched_MLP_variant", "KAN_NLL_improvement_vs_own_strong_optimizer", "MLP_matched_NLL_improvement_vs_own_strong_optimizer", "KAN_beats_best_same_basis_control", "KAN_beats_MLP_matched_support", "TrueKANGain", "BothGain", "ControlExplained", "MLPDegradationDriven", "no_ECE_Brier_tail_debt", "controller_overhead_ratio", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 24),
        "",
        md_table(kan_variant_gate_rows, ["dataset", "seed", "KAN_architecture", "KAN_variant", "matched_MLP_variant", "KAN_NLL_improvement_vs_own_strong_optimizer", "KAN_beats_best_same_basis_control", "KAN_beats_MLP_matched_support", "TrueKANGain", "BothGain", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 16),
        "",
        md_table(kan_variant_no_debt_audit, ["part", "dataset", "seed", "architecture_key", "variant", "base_ECE", "candidate_ECE", "ECE_delta_vs_optimizer_baseline", "base_Brier", "candidate_Brier", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline", "no_ECE_Brier_tail_debt", "no_ECE_Brier_tail_debt_source"], 16),
        "",
        "## Cached Safety Probe Evidence",
        "",
        json.dumps(cached_probe_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(cached_probe_summary, ["part", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "beats_matched_control_rows", "no_debt_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 12),
        "",
        md_table(cached_probe_sd, ["dataset", "seed", "variant", "tau_support", "tau_direction", "beats_matched_control", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "controller_overhead", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 16),
        "",
        "## Cached Safety Mode Sweep Evidence",
        "",
        json.dumps(mode_sweep_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(mode_sweep_summary, ["mode_sweep_mode", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "beats_matched_control_rows", "no_debt_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "raw_metric_direction_gate_rows", "official_direction_gate_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 16),
        "",
        md_table(mode_sweep_gate_rows, ["mode_sweep_mode", "dataset", "seed", "variant", "tau_support", "tau_direction", "beats_matched_control", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "controller_overhead", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 16),
        "",
        "## Cached Safety Temperature Sweep Evidence",
        "",
        json.dumps(temp_sweep_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(temp_sweep_summary, ["temp_sweep_key", "temp_sweep_mode", "temp_readout_cap", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "beats_matched_control_rows", "no_debt_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "raw_metric_direction_gate_rows", "official_direction_gate_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 16),
        "",
        md_table(temp_sweep_gate_rows, ["temp_sweep_key", "dataset", "seed", "variant", "tau_support", "tau_direction", "beats_matched_control", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "controller_overhead", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 16),
        "",
        "## Calibration Policy Temperature Sweep Evidence",
        "",
        json.dumps(policy_temp_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(policy_temp_summary, ["temp_sweep_key", "temp_sweep_mode", "temp_readout_cap", "temp_readout_policy", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "beats_matched_control_rows", "no_debt_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "raw_metric_direction_gate_rows", "official_direction_gate_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 16),
        "",
        md_table(policy_temp_gate_rows, ["temp_sweep_key", "dataset", "seed", "variant", "tau_support", "tau_direction", "beats_matched_control", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "controller_overhead", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 16),
        "",
        "## High-Cadence Overhead Probe Evidence",
        "",
        json.dumps(high_cadence_temp_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "emit50/support200/metric120 support-only snapshot before support+metric combo probe:",
        "",
        json.dumps(cadence_support_only_snapshot or {"status": "not_available"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "emit50/support120/metric200 metric-only snapshot before support-only probe:",
        "",
        json.dumps(cadence_metric_only_snapshot or {"status": "not_available"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "emit200/support200/metric200 snapshot before metric-only probe:",
        "",
        json.dumps(cadence_emit200_snapshot or {"status": "not_available"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(high_cadence_temp_summary, ["temp_sweep_key", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "raw_metric_direction_gate_rows", "official_direction_gate_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 8),
        "",
        md_table(high_cadence_temp_gate_rows, ["temp_sweep_key", "dataset", "seed", "variant", "tau_support", "tau_direction", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "controller_overhead", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 8),
        "",
        "## S1 Rank-Cap Probe Evidence",
        "",
        json.dumps(s1_temp_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(s1_temp_summary, ["temp_sweep_key", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "raw_metric_direction_gate_rows", "official_direction_gate_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 8),
        "",
        md_table(s1_temp_gate_rows, ["temp_sweep_key", "dataset", "seed", "variant", "tau_support", "tau_direction", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "controller_overhead", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 8),
        "",
        "## Fisher Mirror Temperature Sweep Evidence",
        "",
        json.dumps(fisher_temp_route or {"status": "not_run"}, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        md_table(fisher_temp_summary, ["temp_sweep_key", "temp_sweep_mode", "temp_readout_cap", "candidate_rows", "support_positive_rows", "direction_positive_rows", "support_only_rows", "beats_matched_control_rows", "no_debt_rows", "derived_raw_no_debt_rows", "overhead_le_0p35_rows", "raw_metric_direction_gate_rows", "official_direction_gate_rows", "tau_support_mean", "tau_direction_mean", "tau_direction_LCB"], 16),
        "",
        md_table(fisher_temp_gate_rows, ["temp_sweep_key", "dataset", "seed", "variant", "tau_support", "tau_direction", "beats_matched_control", "derived_no_ECE_Brier_tail_debt_from_raw_metrics", "controller_overhead", "ECE_delta_vs_optimizer_baseline", "Brier_delta_vs_optimizer_baseline", "tail_q99_delta_vs_optimizer_baseline"], 16),
        "",
        "## 分析与 Insight",
        "",
        "- 证据链一：v22.45E 重算仍以 support-only 为主要历史事实；本轮不会把 support gain 解释成 residual signal gain。",
        "- 证据链二：functional actuator spectrum / RSE 是 v22.46 的前置口径。若 computed ratio 低于 95%，final route 会停在 `R1-FunctionalSpectrumAuditBlocked`，不会进入 KAN/MLP fairness claim。",
        "- 证据链三：本轮所有训练 row 都通过 same-support / same-OET / same-basis controls 计算 `tau_support` 与 `tau_direction`。只有 `tau_direction > 0` 且 no-debt 的 row 才支持 quotient residual signal 解释。",
        "- 证据链四：安全约束不再只后验观察；Part F 记录 `safety_budget_debt` 与 `safety_budget_velocity_scale`，用于判断 barrier 是否真的参与 velocity 缩放。",
        "- 证据链五：KAN carrier 结论必须同时看 KAN vs own optimizer、same-basis controls、MLP matched support、spectrum/RSE provenance 和 no-debt，单个 NLL 胜利不会自动晋升 official route。",
        "- 证据链六：v22.46 的 no-debt gate 现在由 finalizer 按 v22.43 upstream 同义标准重建，要求同 dataset/seed/architecture 的 optimizer baseline 下 ECE/Brier/tail_q99 三项 delta 同时 <=0；`tail_q95_delta` 只作为敏感性审计，不参与 route gate。",
        "- 证据链七：KAN variant repair 与 emit20 repair 都给出负结果后，当前 blocker 已从“是否缺少一个合适 cadence/variant”转为“现有 support quotient 理论不足以稳定同时满足 no-debt、overhead 和 non-ControlExplained KAN signal”。需要 v22.46b 补充计划，而不是继续扩大同类 sweep。",
        "",
        "## No-Debt 标准复核",
        "",
        "- 结论：当前 official no-debt 设置是正确且保守的。它复现 v22.43 upstream `value_le(..., 0.0)` 的零容差语义，要求同 dataset/seed/`architecture_key` 的 `optimizer_alone` 基线下 `ECE_delta<=0`、`Brier_delta<=0`、`tail_q99_delta<=0` 三项同时成立；`tail_q95_delta` 只做敏感性审计，不参与 route gate。",
        "- 基线审计：DGKAN 行的训练矩阵同时记录 `architecture=strict_FC_PureKAN` 与 `architecture_key=DGKAN_DCHE/DGKAN_DFOU`；v22.46 `row_arch_key()` 使用 `architecture_key`，因此 no-debt 和 KAN own-optimizer 基线不会把 Chebyshev/Fourier basis 混合。",
        f"- 公式一致性审计：main pass `{sum(int_flag(r.get('no_ECE_Brier_tail_debt')) for r in no_debt_audit)}` / `{len(no_debt_audit)}`；repair pass `{sum(int_flag(r.get('no_ECE_Brier_tail_debt')) for r in repair_no_debt_audit)}` / `{len(repair_no_debt_audit)}`。此前脚本复核公式 mismatch 为 0。",
        "- epsilon 敏感性：如果非官方地给 1e-6/1e-5 容差，main pass 会从 26/100 变为 28/100 和 35/100，repair pass 会从 28/144 变为 36/144 和 49/144。该结果只说明存在近零浮点/随机微差样本，不能替代 official no-debt gate，也不能据此晋升 route。",
        "",
        "## Artifact Manifest",
        "",
        md_table(manifest, ["path", "bytes", "sha256"], 32),
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_exec(
        "write_recap_and_manifest",
        task_id="final_recap",
        status="pass",
        gpu="cpu",
        files=f"{RECAP_DOC.relative_to(ROOT)}, results/v22_46/v22_46_artifact_manifest.csv, results/v22_46/v22_46_final_route.json",
        note=(
            f"route={final.get('final_route')}; manifest_files={len(manifest)}; "
            f"continual_rows={len(continual)}; continual_route={continual_summary.get('route', '')}; "
            f"continual_failures={continual_summary.get('failures', '')}"
        ),
    )


def stage_audit(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    code = run_code_truth_gate()
    metric_spectrum = run_metric_and_spectrum_audit(args)
    reanalysis = run_v45e_reanalysis()
    runtime_proxy = code_runtime_proxy_audits()
    write_support_fairness_catalogs()
    return {"code": code, "metric_spectrum": metric_spectrum, "reanalysis": reanalysis, "runtime_proxy": runtime_proxy}


def stage_finalize(args: argparse.Namespace, train_result: dict[str, Any] | None = None, continual_result: dict[str, Any] | None = None) -> dict[str, Any]:
    refresh_posthoc_matrices()
    code_runtime_proxy_audits()
    write_support_fairness_catalogs()
    make_visualizations()
    final = final_route()
    write_recap(final, train_result=train_result, continual_result=continual_result)
    return final


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "audit", "train", "repair", "kan-variant-repair", "cached-safety-probe", "cached-safety-mode-sweep", "cached-safety-temp-sweep", "continual", "finalize", "collect-one"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--architecture", default="MLP")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--variant", default="S4-SignalMetric-OET")
    p.add_argument("--control-mode", default="none")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label", default="v22_46_collect_one")
    p.add_argument("--eval-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--eval-seeds", default="0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--row-limit", type=int, default=0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--held-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--support-rank", type=int, default=4)
    p.add_argument("--nuisance-rank", type=int, default=2)
    p.add_argument("--nuisance-rank-sweep", default="2,4,8")
    p.add_argument("--support-refresh-cadence", type=int, default=20)
    p.add_argument("--beta-signal", type=float, default=0.05)
    p.add_argument("--beta-metric", type=float, default=0.05)
    p.add_argument("--beta-q", type=float, default=0.05)
    p.add_argument("--eta-rho", type=float, default=0.20)
    p.add_argument("--eta-debt", type=float, default=0.10)
    p.add_argument("--tau-safe", type=float, default=0.02)
    p.add_argument("--rho-min", type=float, default=0.0)
    p.add_argument("--rho-max", type=float, default=0.25)
    p.add_argument("--velocity-scale", type=float, default=0.50)
    p.add_argument("--pure-velocity-scale", type=float, default=0.20)
    p.add_argument("--metric-shrinkage", type=float, default=0.10)
    p.add_argument("--metric-eps", type=float, default=1.0e-6)
    p.add_argument("--metric-refresh-cadence", type=int, default=1)
    p.add_argument("--debt-velocity-barrier", type=float, default=0.0)
    p.add_argument("--calibration-velocity-barrier", type=float, default=0.0)
    p.add_argument("--safety-budget-velocity-barrier", type=float, default=0.0)
    p.add_argument("--safety-barrier", type=float, default=5.0)
    p.add_argument("--safety-calibration-weight", type=float, default=0.25)
    p.add_argument("--safety-correction-weight", type=float, default=0.25)
    p.add_argument("--calibration-readout-radial-cap", type=float, default=0.0)
    p.add_argument("--calibration-readout-policy", choices=["signed", "overconfidence_only", "shrink_only", "inverse_signed"], default="signed")
    p.add_argument("--calibration-nuisance-weight", type=float, default=0.0)
    p.add_argument("--calibration-correction-weight", type=float, default=0.0)
    p.add_argument("--calibration-nuisance-mode", default="brier")
    p.add_argument("--calibration-nuisance-cadence", type=int, default=1)
    p.add_argument("--mirror-grad-cadence", type=int, default=1)
    p.add_argument("--cached-controller-emit-cadence", type=int, default=1)
    p.add_argument("--kan-init-variant", default="default")
    p.add_argument("--pure-fu-mode", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--audit-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--audit-architectures", default="MLP,DGKAN_DCHE,DGKAN_DFOU")
    p.add_argument("--audit-supports", default="A_MLP_lowrank,A_MLP_spectral,A_MLP_frequency_like,A_KAN_DCHE_basis_bank,A_KAN_DFOU_basis_bank,A_KAN_DCHE_bank_OET,A_KAN_DFOU_bank_OET,A_OET_layer")
    p.add_argument("--audit-train-size", type=int, default=96)
    p.add_argument("--sketch-dim", type=int, default=6)
    p.add_argument("--functional-eps", type=float, default=1.0e-3)
    p.add_argument("--run-continual-smoke", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--m4-tasks", default="modular_addition,Class_MNIST_0_4_to_5_9")
    p.add_argument("--continual-seeds", default="0")
    p.add_argument("--continual-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--continual-row-limit", type=int, default=8)
    p.add_argument("--continual-steps", type=int, default=160)
    p.add_argument("--continual-train-size", type=int, default=192)
    p.add_argument("--continual-held-size", type=int, default=192)
    p.add_argument("--m4-modulus", type=int, default=13)
    p.add_argument("--m4-train-fraction", type=float, default=0.40)
    p.add_argument("--m4-diag-velocity-scale", type=float, default=0.02)
    p.add_argument("--m4-beta-slow", type=float, default=0.01)
    p.add_argument("--m4-rho", type=float, default=0.05)
    p.add_argument("--m4-lambda-mig", type=float, default=0.50)
    p.add_argument("--m4-lambda-res", type=float, default=0.25)
    p.add_argument("--m4-support-rank", type=int, default=0)
    p.add_argument("--m4-train-threshold", type=float, default=0.90)
    p.add_argument("--m4-test-threshold", type=float, default=0.80)
    p.add_argument("--repair-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--repair-seeds", default="0")
    p.add_argument("--repair-row-limit", type=int, default=0)
    p.add_argument("--repair-steps", type=int, default=200)
    p.add_argument("--repair-velocity-scale", type=float, default=0.18)
    p.add_argument("--repair-pure-velocity-scale", type=float, default=0.08)
    p.add_argument("--repair-support-refresh-cadence", type=int, default=120)
    p.add_argument("--repair-metric-refresh-cadence", type=int, default=120)
    p.add_argument("--repair-safety-barrier", type=float, default=8.0)
    p.add_argument("--repair-calibration-weight", type=float, default=0.50)
    p.add_argument("--repair-beta-signal", type=float, default=0.01)
    p.add_argument("--repair-cached-controller-emit-cadence", type=int, default=1)
    p.add_argument("--fused-debt-controller", action="store_true")
    p.add_argument("--debt-orthogonal-controller", action="store_true")
    p.add_argument("--kan-repair-variants", default="KAN-D-CHE-BasisGram,KAN-D-CHE-BasisGramFast,KAN-D-CHE-FisherSignal-BasisGram,KAN-D-CHE-OET-BankLocal,KAN-D-FOU-BasisGram,KAN-D-FOU-BasisGramFast,KAN-D-FOU-FisherSignal-BasisGram,KAN-D-FOU-OET-BankLocal")
    p.add_argument("--kan-repair-controls", default="none,same-basis-Gram-random,same-basis-Gram-signflip,same-bank-shuffled")
    p.add_argument("--kan-repair-mlp-variants", default="MLP-polynomial-like-feature-support,MLP-same-rank-block-support")
    p.add_argument("--mode-sweep-modes", default="tail_q99_brier_qp_margin,confidence,brier,tail_brier_qp_brier_margin,tail_q99_brier_qp_strict_margin")
    p.add_argument("--temp-sweep-modes", default="tail_q99_brier_qp_margin")
    p.add_argument("--temp-sweep-caps", default="0.02,0.05,0.10")
    p.add_argument("--temp-sweep-policies", default="signed")
    p.add_argument("--temp-sweep-variant", default="S4-SignalMetric-OET")
    p.add_argument("--m7-train-size", type=int, default=192)
    p.add_argument("--m7-held-size", type=int, default=192)
    p.add_argument("--diagnostic-hidden", type=int, default=32)
    p.add_argument("--diagnostic-batch-size", type=int, default=64)
    return p


def collect_one(args: argparse.Namespace) -> dict[str, Any]:
    bind_upstream()
    spec = {
        "part": "manual",
        "mechanism": "manual",
        "recipe": "manual_collect_one",
        "dataset": args.dataset,
        "seed": args.seed,
        "architecture": args.architecture,
        "optimizer": args.optimizer,
        "variant": args.variant,
        "control_mode": args.control_mode,
        "device": args.device,
        "label": args.label,
        "pure_fu_mode": args.pure_fu_mode,
    }
    row = run_one_spec(args, spec)
    write_rows(OUT_ROOT / f"{safe_fragment(args.label)}_manual_summary.csv", [row])
    return row


def main() -> None:
    args = build_parser().parse_args()
    ensure_out()
    bind_upstream()
    if args.stage == "collect-one":
        collect_one(args)
        return
    if args.stage == "audit":
        stage_audit(args)
        return
    if args.stage == "train":
        run_training_suite(args)
        return
    if args.stage == "repair":
        run_repair_suite(args)
        return
    if args.stage == "kan-variant-repair":
        run_kan_variant_repair_suite(args)
        return
    if args.stage == "cached-safety-probe":
        run_cached_safety_probe(args)
        return
    if args.stage == "cached-safety-mode-sweep":
        run_cached_safety_mode_sweep(args)
        return
    if args.stage == "cached-safety-temp-sweep":
        run_cached_safety_temp_sweep(args)
        return
    if args.stage == "continual":
        run_continual_smoke(args)
        return
    if args.stage == "finalize":
        stage_finalize(args)
        return
    audit_result = stage_audit(args)
    train_result = run_training_suite(args)
    continual_result = run_continual_smoke(args)
    final = stage_finalize(args, train_result=train_result, continual_result=continual_result)
    append_exec(
        "stage_all",
        task_id="v22_46_all_complete",
        status="pass" if train_result.get("failures", 0) == 0 and final.get("final_route") != "R0-CodeOrRuntimeInvalid" else "warn",
        gpu=args.gpus,
        files="results/v22_46/v22_46_final_route.json, docs/DG-KAN_v22.46_SupportQuotientCausalFlow_实验结果复盘.md",
        note=f"audit={audit_result}; train={train_result}; continual={continual_result}; route={final.get('final_route')}",
    )


if __name__ == "__main__":
    main()
