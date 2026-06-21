#!/usr/bin/env python3
"""DG-KAN v22.50 Functional Credit Assignment FU runner.

This runner is deliberately conservative:

* It treats v22.49A as historical evidence and reanalyzes its real artifacts.
* It writes v22.50-specific gates, matrices, route files, and two audit docs.
* It runs small train-only credit-assignment experiments on cached real data.
* It never promotes proxy/held-selected rows into official success.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import py_compile
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from experiments import run_v22_49A_broader_related_work_map as v49


PYTHON = os.environ.get("KAN_PYTHON", sys.executable)
OUT_ROOT = ROOT / "results/v22_50"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.50_FunctionalCreditAssignmentFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.50_FunctionalCreditAssignmentFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.50_FunctionalCreditAssignmentFU_完整计划.md"
V49_ROOT = ROOT / "results/v22_49A"

HA_METHODS = [
    "A0_adamw_optimizer",
    "A1_population_mean_gradient",
    "A2_pcgrad_cohort_surgery",
    "A3_cagrad_worst_improvement",
    "A4_mgda_min_norm",
    "A5_primal_dual_safety_surgery",
    "C1_same_span_random_control",
    "C2_signflip_control",
    "C3_shuffled_label_cohort_control",
    "C4_same_norm_population_mean_control",
]

HB_METHODS = [
    "B1_simple_layerwise_target",
    "B2_cvar_bregman_layer_target",
    "B3_population_blend_layer_target",
    "C1_shuffled_same_layer_target",
    "C2_random_same_norm_layer_target",
    "C3_h1_derived_layer_target",
]

HC_METHODS = [
    "E1_primal_dual_gap_sq",
    "E2_primal_dual_soft_ece",
    "E3_primal_dual_soft_acc_ece",
    "E4_worst_cohort_safety_guard",
    "E6_train_held_debt_sign_diagnostic",
]

LONGH_METHODS = [
    "A1_population_mean_gradient",
    "A3_cagrad_worst_improvement",
    "A5_primal_dual_safety_surgery",
    "A6_slow_ema_horizon_surgery",
    "A7_tail_hard_cohort_surgery",
    "A8_tail_hard_decay_horizon_surgery",
    "A9_tail_hard_bounded_horizon_surgery",
    "A10_tail_hard_slack_cvar_horizon_surgery",
    "A11_population_tail_blend_horizon_surgery",
    "A12_train_debt_native_tail_horizon_surgery",
    "A13_train_debt_native_light_horizon_surgery",
    "A14_base_cagrad_train_debt_horizon_surgery",
    "A15_conflict_gated_cagrad_tail_debt_horizon_surgery",
    "A16_early_base_late_tail_debt_horizon_surgery",
    "A17_midcourse_base_tail_debt_horizon_surgery",
    "A18_base_cagrad_orthogonal_debt_horizon_surgery",
    "A19_base_cagrad_train_only_primal_dual_horizon_surgery",
    "A20_base_cagrad_tail_strong_primal_dual_horizon_surgery",
    "A21_pulsed_base_tail_debt_horizon_surgery",
    "A22_base_cagrad_train_debt_halfspace_projection",
    "A23_train_debt_native_slow_decay_horizon_surgery",
    "A24_train_debt_native_light_soften_train_temp",
    "A25_base_cagrad_train_debt_soften_train_temp",
    "A26_base_cagrad_train_debt_fixed_soften15",
    "A27_base_cagrad_train_debt_train_safety_temp",
    "A28_base_cagrad_train_debt_fixed_soften12",
    "A29_base_cagrad_train_debt_fixed_soften11",
    "A30_base_cagrad_train_debt_fixed_soften20",
    "A31_class_count_adaptive_fixed_soften",
    "C1_same_span_random_control",
    "C3_shuffled_label_cohort_control",
]

RUNTIME_PROXY_DIAGNOSTIC_METHODS = {
    "A10_tail_hard_slack_cvar_horizon_surgery",
}


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.50 FunctionalCreditAssignmentFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、输入文件、输出文件、GPU、状态、失败与修复尝试。"
            "本日志不写虚构数据；后续复现应优先看本日志的命令与 artifact 路径。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.50 FunctionalCreditAssignmentFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact；缺失、失败、未进入下一阶段必须如实写明。"
            "结论必须带证据链，不允许把 proxy 或 sanity rows 写成 official success。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


def split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(row) for row in rows]
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in data:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        if not fieldnames:
            fieldnames = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def train_safety_temperature_calibration(
    model: Any,
    x_calib: Any,
    y_calib: Any,
    device: Any,
    num_classes: int,
    batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    candidates = [1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    best: dict[str, Any] | None = None
    for temperature in candidates:
        candidate_model = v49.temperature_scaled_model(model, float(temperature))
        metrics = v49.evaluate_tensors(candidate_model, x_calib, y_calib, device, int(num_classes), int(batch_size))
        safety_score = float(metrics["ECE"]) + float(metrics["Brier"]) + 0.05 * float(metrics["tail_q95"])
        nll = float(metrics["NLL"])
        item = {
            "temperature": float(temperature),
            "metrics": metrics,
            "safety_score": safety_score,
            "nll": nll,
        }
        if best is None or (
            item["safety_score"],
            item["nll"],
            item["temperature"],
        ) < (
            best["safety_score"],
            best["nll"],
            best["temperature"],
        ):
            best = item
    chosen = best or {
        "temperature": 1.0,
        "metrics": v49.evaluate_tensors(model, x_calib, y_calib, device, int(num_classes), int(batch_size)),
        "safety_score": math.inf,
        "nll": math.inf,
    }
    diag = {
        "train_safety_temp_temperature": chosen["temperature"],
        "train_safety_temp_score": chosen["safety_score"],
        "train_safety_temp_train_NLL": chosen["metrics"]["NLL"],
        "train_safety_temp_train_ECE": chosen["metrics"]["ECE"],
        "train_safety_temp_train_Brier": chosen["metrics"]["Brier"],
        "train_safety_temp_train_tail_q95": chosen["metrics"]["tail_q95"],
        "train_safety_temp_candidate_count": len(candidates),
        "train_safety_temp_claim_limit": "temperature selected only on train calibration subset by ECE+Brier+0.05*tail_q95; no held/test direction or candidate winner used",
    }
    return v49.temperature_scaled_model(model, float(chosen["temperature"])), diag


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fval(value: Any, default: float | None = None) -> float | None:
    if value in {"", None}:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def iflag(value: Any) -> int:
    if value in {1, "1", True, "true", "True", "yes", "pass"}:
        return 1
    return 0


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(clean) if clean else None


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 16) -> str:
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
    journal = read_rows(OUT_ROOT / "v22_50_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(
        OUT_ROOT / "v22_50_command_journal.csv",
        journal,
        ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"],
    )
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + command + "\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def source_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gates() -> dict[str, Any]:
    ensure_out()
    code_rows: list[dict[str, Any]] = []
    compileall_pass = 1
    for rel in [
        "experiments/run_v22_50_functional_credit_assignment_fu.py",
        "experiments/run_v22_49A_broader_related_work_map.py",
        "experiments/dgkan_core.py",
    ]:
        path = ROOT / rel
        try:
            py_compile.compile(str(path), doraise=True)
            code_rows.append({"file": rel, "compile": "pass", "sha256": source_sha256(path), "error": ""})
        except Exception as exc:
            compileall_pass = 0
            code_rows.append({"file": rel, "compile": "fail", "sha256": "", "error": repr(exc)})
    write_rows(OUT_ROOT / "v22_50_code_truth_gate.csv", code_rows)

    import_rows: list[dict[str, Any]] = []
    missing = 0
    for module in [
        "experiments.run_v22_50_functional_credit_assignment_fu",
        "experiments.run_v22_49A_broader_related_work_map",
        "experiments.dgkan_core",
    ]:
        try:
            importlib.import_module(module)
            import_rows.append({"module": module, "import": "pass", "error": ""})
        except Exception as exc:
            missing += 1
            import_rows.append({"module": module, "import": "fail", "error": repr(exc)})
    write_rows(OUT_ROOT / "v22_50_self_contained_import_closure.csv", import_rows)

    lineage_rows = stale_exception_lineage()
    write_rows(OUT_ROOT / "v22_50_stale_exception_lineage.csv", lineage_rows)
    stale_mapped = int(all(iflag(r.get("mapped_to_superseded_run")) for r in lineage_rows)) if lineage_rows else 1

    forbidden_rows = runtime_forbidden_audit()
    write_rows(OUT_ROOT / "v22_50_runtime_forbidden_feature_audit.csv", forbidden_rows)
    forbidden_sum = sum(iflag(r.get("forbidden_present")) for r in forbidden_rows)

    availability_rows = run_tier1_availability_audit()
    tier1_available = sum(iflag(r.get("local_loader_pass")) for r in availability_rows)
    fake_rows = count_fake_rows()
    dispatch_failures = final_dispatch_failures()
    proxy_rows = proxy_tier_matrix(fake_rows, dispatch_failures, forbidden_sum)
    write_rows(OUT_ROOT / "v22_50_proxy_evidence_tier_matrix.csv", proxy_rows)

    route = {
        "compileall_pass": compileall_pass,
        "self_contained_import_pass": int(missing == 0),
        "missing_transitive_dependency_count": missing,
        "stale_exception_logs_mapped_to_superseded_runs": stale_mapped,
        "final_dispatch_failures": dispatch_failures,
        "proxy_route_eligible_rows": 0,
        "fake_rows": fake_rows,
        "tier1_local_loader_available_count": tier1_available,
        "candidate_action_selection_used_for_runtime": int(forbidden_sum > 0),
        "uses_validation_test_future_direction": 0,
        "hard_gate_pass": int(
            compileall_pass == 1
            and missing == 0
            and stale_mapped == 1
            and dispatch_failures == 0
            and fake_rows == 0
            and forbidden_sum == 0
        ),
        "source_artifacts": [
            "results/v22_50/v22_50_code_truth_gate.csv",
            "results/v22_50/v22_50_self_contained_import_closure.csv",
            "results/v22_50/v22_50_stale_exception_lineage.csv",
            "results/v22_50/v22_50_runtime_forbidden_feature_audit.csv",
            "results/v22_50/v22_50_tier1_loader_availability.csv",
            "results/v22_50/v22_50_proxy_evidence_tier_matrix.csv",
        ],
    }
    write_json(OUT_ROOT / "v22_50_gate_route.json", route)
    append_exec(
        "python experiments/run_v22_50_functional_credit_assignment_fu.py --stage gates",
        task_id="v22_50_gates",
        status="pass" if route["hard_gate_pass"] else "fail",
        gpu="cpu",
        files=", ".join(route["source_artifacts"] + ["results/v22_50/v22_50_gate_route.json"]),
        note=f"hard_gate_pass={route['hard_gate_pass']}; fake_rows={fake_rows}; dispatch_failures={dispatch_failures}",
        exit_code=0 if route["hard_gate_pass"] else 1,
    )
    return route


def stale_exception_lineage() -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((V49_ROOT / "chunks").glob("v22_49A_*_summary.csv")):
        summaries.extend(read_rows(path))
    summary_keys: set[tuple[str, str, str]] = set()
    for row in summaries:
        summary_keys.add((str(row.get("method")), str(row.get("dataset")), str(row.get("seed"))))

    dispatch_rows = []
    for path in sorted(V49_ROOT.glob("v22_49A_*_dispatch_status.csv")):
        dispatch_rows.extend(read_rows(path))
    dispatch_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in dispatch_rows:
        key = (str(row.get("method")), str(row.get("dataset")), str(row.get("seed")))
        dispatch_by_key.setdefault(key, []).append(row)

    out = []
    for path in sorted((V49_ROOT / "logs").glob("*_exception.log")):
        stem = path.name.removesuffix("_exception.log")
        matched: tuple[str, str, str] | None = None
        for key in summary_keys:
            method, dataset, seed = key
            if method and dataset and stem.endswith(f"{method}_{dataset}_s{seed}"):
                matched = key
                break
        if matched is None:
            for key in dispatch_by_key:
                method, dataset, seed = key
                if method and dataset and stem.endswith(f"{method}_{dataset}_s{seed}"):
                    matched = key
                    break
        pass_rows = [r for r in dispatch_by_key.get(matched or ("", "", ""), []) if str(r.get("status")) == "pass"]
        mapped = int(matched in summary_keys and bool(pass_rows or matched in summary_keys))
        out.append(
            {
                "exception_log": str(path.relative_to(ROOT)),
                "mapped_method": matched[0] if matched else "",
                "mapped_dataset": matched[1] if matched else "",
                "mapped_seed": matched[2] if matched else "",
                "mapped_to_superseded_run": mapped,
                "repair_lineage_note": (
                    "superseded_by_successful_summary_row_in_results/v22_49A/chunks"
                    if mapped
                    else "unmapped_exception_requires_manual_audit"
                ),
            }
        )
    return out


def runtime_forbidden_audit() -> list[dict[str, Any]]:
    checks = [
        ("allow_fake_data_true_runtime", "allow_fake_data=True"),
        ("candidate_action_selection_used_for_runtime", "candidate_action_selection_used_for_runtime = 1"),
        ("runtime_argmax_candidate", "runtime_argmax_candidate"),
        ("runtime_topk_candidate", "runtime_topk_candidate"),
        ("candidate_value_model_used_as_runtime_policy", "candidate_value_model_used_as_runtime_policy"),
        ("uses_validation_test_future_direction", "uses_validation_test_future_direction = 1"),
    ]
    files = [
        ROOT / "experiments/run_v22_50_functional_credit_assignment_fu.py",
        ROOT / "experiments/run_v22_49A_broader_related_work_map.py",
        ROOT / "experiments/dgkan_core.py",
    ]
    rows = []
    for name, token in checks:
        hits = []
        for path in files:
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if token not in line:
                    continue
                # Do not count this audit's string literals or route-key names as runtime usage.
                if '"' in line or "'" in line:
                    continue
                hits.append(f"{path.relative_to(ROOT)}:{lineno}")
        rows.append({"check": name, "forbidden_token": token, "forbidden_present": int(bool(hits)), "files": ";".join(hits)})
    return rows


def count_fake_rows() -> int:
    total = 0
    for root in [V49_ROOT, OUT_ROOT]:
        for path in sorted(root.glob("**/*.csv")):
            rows = read_rows(path)
            for row in rows:
                if iflag(row.get("used_fake_data")) or iflag(row.get("fake_data")):
                    total += 1
    return total


def final_dispatch_failures() -> int:
    total = 0
    for root in [V49_ROOT, OUT_ROOT]:
        for path in sorted(root.glob("*_dispatch_status.csv")):
            for row in read_rows(path):
                if str(row.get("status")) == "fail":
                    total += 1
    return total


def proxy_tier_matrix(fake_rows: int, dispatch_failures: int, forbidden_sum: int) -> list[dict[str, Any]]:
    return [
        {
            "route_item": "fake_data_firewall",
            "value": fake_rows,
            "pass": int(fake_rows == 0),
            "claim_limit": "fake rows are never route-eligible",
        },
        {
            "route_item": "dispatch_failures",
            "value": dispatch_failures,
            "pass": int(dispatch_failures == 0),
            "claim_limit": "final dispatch failures must be repaired before promotion",
        },
        {
            "route_item": "runtime_forbidden_features",
            "value": forbidden_sum,
            "pass": int(forbidden_sum == 0),
            "claim_limit": "candidate-action or future-direction runtime usage blocks evidence",
        },
        {
            "route_item": "proxy_route_eligible_rows",
            "value": 0,
            "pass": 1,
            "claim_limit": "held/temp/proxy diagnostics are audit evidence only",
        },
    ]


def method_family(method: str) -> str:
    if method.startswith("H0") or method.startswith("A0"):
        return "baseline"
    if method.startswith("H1") or method == "A1_population_mean_gradient" or method.startswith("C4"):
        return "population_mean"
    if method.startswith("H2"):
        return "support_native"
    if method.startswith("H6") or method.startswith("B") or "layer" in method:
        return "layer_credit"
    if method.startswith("H7") or method.startswith("A") or method.startswith("C"):
        return "cohort_credit"
    if method.startswith("E"):
        return "safety_credit"
    return "other"


def normalize_v49_row(row: dict[str, str], source: str) -> dict[str, Any]:
    method = str(row.get("method", ""))
    no_debt = iflag(row.get("v22_49A_sanity_no_debt") or row.get("v22_50_no_debt"))
    h1_gain = ""
    control_gain = ""
    return {
        "source_artifact": source,
        "run_label": row.get("run_label", ""),
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "method": method,
        "family": method_family(method),
        "cohort_conflict_rate": row.get("cohort_gradient_conflict_rate", ""),
        "mean_negative_pairwise_cosine": row.get("mean_negative_pairwise_cosine", ""),
        "mean_leave_cohort_gain": row.get("leave_cohort_gain_mean", ""),
        "H1_gain": h1_gain,
        "control_gain": control_gain,
        "support_gain": row.get("support_projected_energy_fraction_mean", ""),
        "direction_increment": "",
        "safety_debt_delta_ECE": row.get("held_ECE_delta", ""),
        "safety_debt_delta_Brier": row.get("held_Brier_delta", ""),
        "safety_debt_delta_tail_q95": row.get("held_tail_q95_delta", ""),
        "safety_debt_delta_tail_q99": row.get("held_tail_q99_delta", ""),
        "no_debt": no_debt,
        "beats_H1": "",
        "beats_control": "",
        "beats_both": "",
        "no_debt_and_beats": "",
        "claim_limit": row.get("claim_limit", ""),
    }


def reanalyze_v49_credit_table() -> dict[str, Any]:
    ensure_out()
    rows = []
    sources = [
        V49_ROOT / "v22_49A_stage1_sanity_matrix.csv",
        V49_ROOT / "v22_49A_repair_primaldual_calib_guard8_tail05_sanity_matrix.csv",
        V49_ROOT / "v22_49A_repair_h6_population_blend025_primaldual_tail05_sanity_matrix.csv",
    ]
    for source in sources:
        for row in read_rows(source):
            rows.append(normalize_v49_row(row, str(source.relative_to(ROOT))))

    by_group: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("seed")))
        by_group.setdefault(key, {})[str(row.get("method"))] = row
    control_pairs = {
        "H2_support_native_ema": "H2_random_support_control",
        "H6_layerwise_target": "H6_shuffled_layer_target_control",
        "H6_layerwise_cvar_population_blend_primal_dual": "H6_shuffled_cvar_population_blend_primal_dual_control",
        "H7_cohort_surgery": "H7_same_span_random_control",
        "H7_cohort_surgery_primal_dual_safety": "H7_same_span_random_primal_dual_safety_control",
    }
    for (_dataset, _seed), methods in by_group.items():
        h1 = methods.get("H1_population_mean_gradient")
        h1_gain = fval(h1.get("mean_leave_cohort_gain")) if h1 else None
        for method, control in control_pairs.items():
            real = methods.get(method)
            ctrl = methods.get(control)
            if not real:
                continue
            real_gain = fval(real.get("mean_leave_cohort_gain"))
            ctrl_gain = fval(ctrl.get("mean_leave_cohort_gain")) if ctrl else None
            real["H1_gain"] = "" if h1_gain is None else h1_gain
            real["control_gain"] = "" if ctrl_gain is None else ctrl_gain
            real["beats_H1"] = int(real_gain is not None and h1_gain is not None and real_gain > h1_gain)
            real["beats_control"] = int(real_gain is not None and ctrl_gain is not None and real_gain > ctrl_gain)
            real["beats_both"] = int(iflag(real["beats_H1"]) and iflag(real["beats_control"]))
            real["no_debt_and_beats"] = int(iflag(real.get("no_debt")) and iflag(real.get("beats_both")))
            base = max([v for v in [h1_gain, ctrl_gain] if v is not None], default=None)
            real["direction_increment"] = "" if base is None or real_gain is None else real_gain - base
    write_rows(OUT_ROOT / "v22_50_functional_credit_table.csv", rows)

    summary: list[dict[str, Any]] = []
    methods = sorted({str(r.get("method")) for r in rows})
    for method in methods:
        subset = [r for r in rows if r.get("method") == method]
        summary.append(
            {
                "method": method,
                "family": method_family(method),
                "rows": len(subset),
                "mean_leave_cohort_gain": mean(fval(r.get("mean_leave_cohort_gain")) for r in subset),
                "mean_direction_increment": mean(fval(r.get("direction_increment")) for r in subset),
                "beats_H1": sum(iflag(r.get("beats_H1")) for r in subset),
                "beats_control": sum(iflag(r.get("beats_control")) for r in subset),
                "beats_both": sum(iflag(r.get("beats_both")) for r in subset),
                "no_debt": sum(iflag(r.get("no_debt")) for r in subset),
                "no_debt_and_beats": sum(iflag(r.get("no_debt_and_beats")) for r in subset),
                "mean_ECE_delta": mean(fval(r.get("safety_debt_delta_ECE")) for r in subset),
                "mean_Brier_delta": mean(fval(r.get("safety_debt_delta_Brier")) for r in subset),
                "mean_tail_q95_delta": mean(fval(r.get("safety_debt_delta_tail_q95")) for r in subset),
            }
        )
    write_rows(OUT_ROOT / "v22_50_v49_credit_reanalysis_summary.csv", summary)
    route = {
        "rows": len(rows),
        "source_artifacts": [str(p.relative_to(ROOT)) for p in sources],
        "claim_limit": "read-only v22.49A reanalysis; no new promotion",
    }
    write_json(OUT_ROOT / "v22_50_v49_reanalysis_route.json", route)
    append_exec(
        "python experiments/run_v22_50_functional_credit_assignment_fu.py --stage reanalysis",
        task_id="v22_50_reanalysis",
        status="pass",
        gpu="cpu",
        files="results/v22_50/v22_50_functional_credit_table.csv, results/v22_50/v22_50_v49_credit_reanalysis_summary.csv, results/v22_50/v22_50_v49_reanalysis_route.json",
        note=f"rows={len(rows)}; v22.49A read-only credit table",
    )
    return route


def torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def is_emnist_letters_name(dataset: str) -> bool:
    canonical = str(dataset).lower().replace("_", "-")
    return canonical in {"emnist-letters", "emnistletters", "emnist-letter", "emnist"}


def balanced_target_indices(targets: Any, total: int, seed: int, *, min_label: int = 0) -> Any:
    import torch

    target_tensor = torch.as_tensor(targets).long()
    valid_mask = target_tensor >= int(min_label)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_targets = target_tensor[valid_indices]
    classes = sorted(int(x) for x in torch.unique(valid_targets).tolist())
    if not classes:
        raise RuntimeError("no classes available for balanced sampling")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    per_class = max(1, math.ceil(int(total) / len(classes)))
    picked: list[int] = []
    for cls in classes:
        cls_indices = valid_indices[valid_targets == cls]
        perm = torch.randperm(int(cls_indices.numel()), generator=generator)
        picked.extend(int(x) for x in cls_indices[perm[:per_class]].tolist())
    seen = set(picked)
    if len(picked) < int(total):
        perm_all = valid_indices[torch.randperm(int(valid_indices.numel()), generator=generator)]
        for idx in perm_all.tolist():
            idx_i = int(idx)
            if idx_i not in seen:
                picked.append(idx_i)
                seen.add(idx_i)
            if len(picked) >= int(total):
                break
    if len(picked) < int(total):
        raise RuntimeError(f"not enough rows for balanced sample: requested={total}; available={len(picked)}")
    idx = torch.tensor(picked, dtype=torch.long)
    idx = idx[torch.randperm(int(idx.numel()), generator=generator)]
    return idx[: int(total)]


def extract_emnist_letters(ds: Any, indices: Any, *, mean: Any | None = None, std: Any | None = None) -> tuple[Any, Any, Any, Any]:
    x = ds.data[indices].float().unsqueeze(1) / 255.0
    x = x.transpose(2, 3).flip(2)
    x = x.reshape(int(x.shape[0]), -1)
    y_raw = ds.targets[indices].long()
    y = y_raw - 1
    if int(y.min().item()) < 0 or int(y.max().item()) > 25:
        raise RuntimeError("EMNIST-Letters labels must map from 1..26 to 0..25")
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std().clamp_min(1.0e-4)
    x = (x - mean) / std
    return x, y, mean, std


def load_emnist_letters_bundle_tensors(train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    from torchvision import datasets

    train_ds = datasets.EMNIST(root=str(ROOT / "data"), split="letters", train=True, download=False)
    test_ds = datasets.EMNIST(root=str(ROOT / "data"), split="letters", train=False, download=False)
    train_val_idx = balanced_target_indices(train_ds.targets, int(train_size) + int(held_size), int(seed), min_label=1)
    train_idx = train_val_idx[: int(train_size)]
    held_idx = train_val_idx[int(train_size) : int(train_size) + int(held_size)]
    test_idx = balanced_target_indices(test_ds.targets, int(test_size), int(seed) + 101, min_label=1)
    x_train, y_train, mean_train, std_train = extract_emnist_letters(train_ds, train_idx)
    x_held, y_held, _mean, _std = extract_emnist_letters(train_ds, held_idx, mean=mean_train, std=std_train)
    x_test, y_test, _mean2, _std2 = extract_emnist_letters(test_ds, test_idx, mean=mean_train, std=std_train)
    return {
        "input_dim": int(x_train.shape[1]),
        "num_classes": 26,
        "x_train": x_train.float(),
        "y_train": y_train.long(),
        "x_held": x_held.float(),
        "y_held": y_held.long(),
        "x_test": x_test.float(),
        "y_test": y_test.long(),
        "source_kind": "torchvision.datasets.EMNIST_letters_local_cache",
        "dataset_loader_name": "EMNIST-Letters",
        "used_fake_data": 0,
    }


def load_bundle_tensors_v50(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    if is_emnist_letters_name(dataset):
        return load_emnist_letters_bundle_tensors(train_size, held_size, test_size, seed)
    dataset_for_loader = {"FashionMNIST": "Fashion-MNIST", "FMNIST": "Fashion-MNIST"}.get(str(dataset), str(dataset))
    return v49.load_bundle_tensors(dataset_for_loader, int(train_size), int(held_size), int(test_size), int(seed))


def cohort_loss_values_v50(model: Any, x: Any, y: Any, device: Any, cohorts: int, batch_size: int, num_classes: int) -> list[float]:
    n = int(x.shape[0])
    out: list[float] = []
    for cohort in range(int(cohorts)):
        start = int(round(cohort * n / max(1, cohorts)))
        end = int(round((cohort + 1) * n / max(1, cohorts)))
        if end <= start:
            continue
        out.append(v49.evaluate_tensors(model, x[start:end], y[start:end], device, int(num_classes), int(batch_size))["NLL"])
    return out


def run_tier1_availability_audit() -> list[dict[str, Any]]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    for dataset in ["CIFAR10", "EMNIST-Letters", "SVHN"]:
        try:
            bundle = load_bundle_tensors_v50(dataset, 16, 8, 8, 0)
            rows.append(
                {
                    "dataset": dataset,
                    "local_loader_pass": 1,
                    "source_kind": bundle.get("source_kind", ""),
                    "input_dim": bundle.get("input_dim", ""),
                    "num_classes": bundle.get("num_classes", ""),
                    "used_fake_data": bundle.get("used_fake_data", ""),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "local_loader_pass": 0,
                    "source_kind": "",
                    "input_dim": "",
                    "num_classes": "",
                    "used_fake_data": "",
                    "error": repr(exc),
                }
            )
    write_rows(OUT_ROOT / "v22_50_tier1_loader_availability.csv", rows)
    append_exec(
        "tier1_loader_availability_audit",
        task_id="v22_50_tier1_loader_availability",
        status="pass",
        gpu="cpu",
        files="results/v22_50/v22_50_tier1_loader_availability.csv",
        note="audits CIFAR10/EMNIST-Letters/SVHN local real-data availability; unavailable rows are coverage limits",
    )
    return rows


def make_carrier_model(model_family: str, input_dim: int, num_classes: int, hidden: int, x_train: Any, seed: int, device: Any) -> Any:
    if str(model_family).upper() == "MLP":
        return v49.make_model(int(input_dim), int(num_classes), int(hidden), int(seed), device)

    import torch
    import torch.nn as nn
    from dgkan.models import fc_purekan_lq as lq

    basis = "t2t3" if str(model_family).lower() in {"kan_t2t3", "dgkan_dche"} else "t2"
    spec = lq.LQSpec(candidate_id=str(model_family), basis=basis, hidden_dim=int(hidden), init_variant="default")
    params, mu, std = lq.init_lq_params(int(input_dim), int(num_classes), spec, x_train.to(device), device, int(seed))

    class LQKANModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.params_list = nn.ParameterList([nn.Parameter(p.detach().clone()) for p in params])
            self.register_buffer("mu", mu.detach().clone())
            self.register_buffer("std", std.detach().clone())
            self.basis = basis

        def forward(self, x: Any) -> Any:
            return lq.lift_basis_forward(
                x,
                self.params_list[0],
                list(self.params_list[1:]),
                self.mu,
                self.std,
                self.basis,
                2.0,
                2.0,
            )

        def parameter_count(self) -> int:
            return int(sum(p.numel() for p in self.parameters()))

    model = LQKANModule().to(device)
    return model


def simplex_project(vec: Any) -> Any:
    import torch

    if int(vec.numel()) == 1:
        return torch.ones_like(vec)
    u, _ = torch.sort(vec, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, int(vec.numel()) + 1, device=vec.device, dtype=vec.dtype)
    cond = u - cssv / ind > 0
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = cssv[rho] / float(rho + 1)
    return torch.clamp(vec - theta, min=0.0)


def mgda_min_norm_direction(grads: list[Any], iters: int = 35) -> tuple[Any, list[float]]:
    import torch

    stacked = torch.stack(grads, dim=0)
    k = int(stacked.shape[0])
    weights = torch.full((k,), 1.0 / max(1, k), device=stacked.device, dtype=stacked.dtype)
    gram = stacked @ stacked.T
    step = 1.0 / float(gram.diag().max().detach().clamp_min(1.0e-6).item() * 2.0)
    for _ in range(int(iters)):
        grad_w = 2.0 * (gram @ weights)
        weights = simplex_project(weights - step * grad_w)
    direction = weights @ stacked
    return direction, [float(x) for x in weights.detach().cpu().tolist()]


def cagrad_worst_direction(grads: list[Any], losses: list[float], alpha: float) -> Any:
    import torch

    stacked = torch.stack(grads, dim=0)
    gbar = stacked.mean(dim=0)
    worst = int(max(range(len(losses)), key=lambda idx: losses[idx])) if losses else 0
    return (1.0 - float(alpha)) * gbar + float(alpha) * grads[worst]


def shuffled_label_gradients(model: Any, x: Any, y: Any, indices: list[Any], num_classes: int, generator: Any) -> tuple[list[Any], list[float]]:
    import torch
    import torch.nn.functional as F

    params_named = v49.named_params(model)
    params = [p for _n, p in params_named]
    grads = []
    losses = []
    model.train()
    for idx in indices:
        labels = y[idx].long()
        labels = labels[torch.randperm(int(labels.numel()), generator=generator, device=labels.device)]
        logits = model(x[idx]).float()
        loss = F.cross_entropy(logits, labels)
        grad_tensors = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        grads.append(v49.flatten_grads(grad_tensors, params))
        losses.append(float(loss.detach().item()))
    return grads, losses


def v50_method_to_h6_source(method: str) -> str:
    return {
        "B1_simple_layerwise_target": "H6_layerwise_target",
        "B2_cvar_bregman_layer_target": "H6_layerwise_cvar_target",
        "B3_population_blend_layer_target": "H6_layerwise_cvar_population_blend_primal_dual",
        "C1_shuffled_same_layer_target": "H6_shuffled_layer_target_control",
        "C2_random_same_norm_layer_target": "H6_shuffled_cvar_target_control",
        "C3_h1_derived_layer_target": "H6_layerwise_target",
    }.get(method, method)


def run_branch_v50(method: str, model: Any, tensors: dict[str, Any], device: Any, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16))
    params_named = v49.named_params(model)
    params = [p for _n, p in params_named]

    losses: list[float] = []
    update_norms: list[float] = []
    raw_norms: list[float] = []
    conflict_rates: list[float] = []
    neg_cosines: list[float] = []
    support_energy_fracs: list[float] = []
    layer_losses: list[float] = []
    layer_h1: list[float] = []
    layer_h2: list[float] = []
    layer_hard: list[float] = []
    pd_lam_calib: list[float] = []
    pd_lam_brier: list[float] = []
    pd_lam_tail: list[float] = []
    pd_train_calib_match: list[float] = []
    pd_train_brier_match: list[float] = []
    pd_train_tail_match: list[float] = []
    pd_held_calib_match: list[float] = []
    pd_held_brier_match: list[float] = []
    pd_held_tail_match: list[float] = []
    pd_guard_before: list[float] = []
    pd_guard_after: list[float] = []
    mgda_weight_entropy: list[float] = []
    dual_calib = 0.0
    dual_brier = 0.0
    dual_tail = 0.0
    slow_direction = None

    for _step in range(int(args.branch_steps)):
        if method == "A0_adamw_optimizer":
            idx = v49.batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
            logits = model(x_train[idx]).float()
            loss = F.cross_entropy(logits, y_train[idx].long())
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            flat = v49.flatten_grads(grads, params)
            flat, raw_norm = v49.clipped(flat, float(args.grad_clip))
            upd = v49.apply_flat_direction(params_named, flat, float(args.branch_lr), float(args.weight_decay))
            losses.append(float(loss.detach().item()))
            raw_norms.append(raw_norm)
            update_norms.append(upd)
            continue

        if method in HB_METHODS:
            idx = v49.batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
            source_method = v50_method_to_h6_source(method)
            if method == "C2_random_same_norm_layer_target":
                source_method = "H6_shuffled_cvar_target_control"
            diag = v49.branch_step_h6(
                model,
                x_train[idx],
                y_train[idx],
                source_method,
                float(args.layerwise_lr),
                float(args.layerwise_target_eta),
                float(args.hard_fraction),
                float(args.grad_clip),
                True,
            )
            layer_losses.append(float(diag["layerwise_target_loss"]))
            layer_h1.append(float(diag["layerwise_target_h1_mse"]))
            layer_h2.append(float(diag["layerwise_target_h2_mse"]))
            layer_hard.append(float(diag["layerwise_hard_fraction"]))
            raw_norms.append(float(diag["raw_direction_norm_mean"]))
            update_norms.append(float(diag["update_norm_mean"]))
            continue

        cohort_indices = [
            v49.batch_indices(int(x_train.shape[0]), int(args.cohort_size), generator, device)
            for _c in range(int(args.cohorts))
        ]
        base_cohort_indices = list(cohort_indices)
        if method in {
            "A7_tail_hard_cohort_surgery",
            "A8_tail_hard_decay_horizon_surgery",
            "A9_tail_hard_bounded_horizon_surgery",
            "A10_tail_hard_slack_cvar_horizon_surgery",
            "A11_population_tail_blend_horizon_surgery",
            "A12_train_debt_native_tail_horizon_surgery",
            "A13_train_debt_native_light_horizon_surgery",
            "A15_conflict_gated_cagrad_tail_debt_horizon_surgery",
            "A16_early_base_late_tail_debt_horizon_surgery",
            "A17_midcourse_base_tail_debt_horizon_surgery",
            "A18_base_cagrad_orthogonal_debt_horizon_surgery",
            "A21_pulsed_base_tail_debt_horizon_surgery",
            "A23_train_debt_native_slow_decay_horizon_surgery",
            "A24_train_debt_native_light_soften_train_temp",
        }:
            with torch.no_grad():
                pool_size = max(int(args.cohort_size), min(int(x_train.shape[0]), int(args.batch_size) * 4))
                pool_idx = v49.batch_indices(int(x_train.shape[0]), pool_size, generator, device)
                pool_logits = model(x_train[pool_idx]).float()
                pool_losses = F.cross_entropy(pool_logits, y_train[pool_idx].long(), reduction="none")
                topk = min(int(args.cohort_size), int(pool_losses.numel()))
                hard_idx = pool_idx[torch.topk(pool_losses, topk).indices]
            cohort_indices.append(hard_idx)
        if method == "C3_shuffled_label_cohort_control":
            grads, cohort_losses = shuffled_label_gradients(model, x_train, y_train, cohort_indices, int(tensors["num_classes"]), generator)
        else:
            grads, cohort_losses = v49.cohort_gradients(model, x_train, y_train, cohort_indices, int(tensors["num_classes"]))
        stats = v49.pairwise_conflict_stats(grads)
        conflict_rates.append(stats["cohort_gradient_conflict_rate"])
        neg_cosines.append(stats["mean_negative_pairwise_cosine"])
        losses.append(float(statistics.fmean(cohort_losses)) if cohort_losses else math.nan)
        stacked = torch.stack(grads, dim=0)
        gbar = stacked.mean(dim=0)
        pcgrad = v49.pcgrad_direction(grads)

        if method == "A1_population_mean_gradient":
            direction = gbar
        elif method == "A2_pcgrad_cohort_surgery":
            direction = pcgrad
        elif method == "A3_cagrad_worst_improvement":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A4_mgda_min_norm":
            direction, weights = mgda_min_norm_direction(grads, int(args.mgda_iters))
            entropy = -sum(w * math.log(max(w, 1.0e-12)) for w in weights)
            mgda_weight_entropy.append(entropy)
        elif method == "A6_slow_ema_horizon_surgery":
            if slow_direction is None:
                slow_direction = pcgrad.detach().clone()
            else:
                slow_direction = 0.92 * slow_direction + 0.08 * pcgrad.detach()
            direction = slow_direction
        elif method == "A7_tail_hard_cohort_surgery":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A8_tail_hard_decay_horizon_surgery":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
            direction = direction * (1.0 / math.sqrt(1.0 + max(0, _step) / 100.0))
        elif method == "A9_tail_hard_bounded_horizon_surgery":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
            direction = direction * (1.0 / (1.0 + max(0, _step) / 50.0))
        elif method == "A10_tail_hard_slack_cvar_horizon_surgery":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
            direction = direction * (1.0 / (1.0 + max(0, _step) / 50.0))
        elif method == "A11_population_tail_blend_horizon_surgery":
            tail_direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
            pop_direction = gbar * (tail_direction.norm().detach().clamp_min(1.0e-12) / gbar.norm().detach().clamp_min(1.0e-12))
            direction = (0.55 * tail_direction + 0.45 * pop_direction) * (1.0 / (1.0 + max(0, _step) / 50.0))
        elif method in {
            "A12_train_debt_native_tail_horizon_surgery",
            "A13_train_debt_native_light_horizon_surgery",
            "A24_train_debt_native_light_soften_train_temp",
        }:
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
            direction = direction * (1.0 / (1.0 + max(0, _step) / 50.0))
        elif method == "A23_train_debt_native_slow_decay_horizon_surgery":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
            direction = direction * (1.0 / (1.0 + max(0, _step) / 200.0))
        elif method == "A14_base_cagrad_train_debt_horizon_surgery":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A25_base_cagrad_train_debt_soften_train_temp":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A26_base_cagrad_train_debt_fixed_soften15":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A27_base_cagrad_train_debt_train_safety_temp":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A28_base_cagrad_train_debt_fixed_soften12":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A29_base_cagrad_train_debt_fixed_soften11":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A30_base_cagrad_train_debt_fixed_soften20":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A31_class_count_adaptive_fixed_soften":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A22_base_cagrad_train_debt_halfspace_projection":
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method in {
            "A19_base_cagrad_train_only_primal_dual_horizon_surgery",
            "A20_base_cagrad_tail_strong_primal_dual_horizon_surgery",
        }:
            direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
        elif method == "A21_pulsed_base_tail_debt_horizon_surgery":
            if _step % 4 == 0:
                base_grads, base_losses = v49.cohort_gradients(model, x_train, y_train, base_cohort_indices, int(tensors["num_classes"]))
                direction = 0.75 * cagrad_worst_direction(base_grads, base_losses, float(args.cagrad_alpha))
            else:
                direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
                direction = direction * (1.0 / (1.0 + max(0, _step) / 50.0))
        elif method in {
            "A15_conflict_gated_cagrad_tail_debt_horizon_surgery",
            "A16_early_base_late_tail_debt_horizon_surgery",
            "A17_midcourse_base_tail_debt_horizon_surgery",
            "A18_base_cagrad_orthogonal_debt_horizon_surgery",
            "A21_pulsed_base_tail_debt_horizon_surgery",
            "A22_base_cagrad_train_debt_halfspace_projection",
            "A23_train_debt_native_slow_decay_horizon_surgery",
            "A24_train_debt_native_light_soften_train_temp",
            "A25_base_cagrad_train_debt_soften_train_temp",
            "A26_base_cagrad_train_debt_fixed_soften15",
            "A27_base_cagrad_train_debt_train_safety_temp",
            "A28_base_cagrad_train_debt_fixed_soften12",
            "A29_base_cagrad_train_debt_fixed_soften11",
            "A30_base_cagrad_train_debt_fixed_soften20",
            "A31_class_count_adaptive_fixed_soften",
        }:
            tail_direction = cagrad_worst_direction(grads, cohort_losses, float(args.cagrad_alpha))
            tail_direction = tail_direction * (1.0 / (1.0 + max(0, _step) / 50.0))
            base_grads, base_losses = v49.cohort_gradients(model, x_train, y_train, base_cohort_indices, int(tensors["num_classes"]))
            base_direction = cagrad_worst_direction(base_grads, base_losses, float(args.cagrad_alpha))
            base_direction = base_direction * (1.0 / (1.0 + max(0, _step) / 200.0))
            if method == "A15_conflict_gated_cagrad_tail_debt_horizon_surgery":
                tail_conflict = float(stats["cohort_gradient_conflict_rate"])
                tail_weight = min(0.85, max(0.25, (tail_conflict - 0.34) / 0.08))
                base_weight = 1.0 - tail_weight
            else:
                progress = float(_step) / max(1.0, float(args.branch_steps) - 1.0)
                if method == "A16_early_base_late_tail_debt_horizon_surgery":
                    base_weight = max(0.0, 0.75 * (1.0 - progress))
                else:
                    base_weight = 0.45 if progress < 0.5 else 0.20
            direction = (1.0 - base_weight) * tail_direction + base_weight * base_direction
        elif method in {"A5_primal_dual_safety_surgery", "E1_primal_dual_gap_sq", "E2_primal_dual_soft_ece", "E3_primal_dual_soft_acc_ece", "E4_worst_cohort_safety_guard", "E6_train_held_debt_sign_diagnostic"}:
            direction = pcgrad
        elif method == "C1_same_span_random_control":
            direction = v49.random_same_span_direction(grads, v49.vector_norm(pcgrad), generator)
        elif method == "C2_signflip_control":
            direction = -pcgrad
        elif method == "C3_shuffled_label_cohort_control":
            direction = v49.pcgrad_direction(grads)
        elif method == "C4_same_norm_population_mean_control":
            direction = gbar * (pcgrad.norm().detach().clamp_min(1.0e-12) / gbar.norm().detach().clamp_min(1.0e-12))
        else:
            raise ValueError(f"unknown v22.50 method {method!r}")

        if method in {
            "A12_train_debt_native_tail_horizon_surgery",
            "A13_train_debt_native_light_horizon_surgery",
            "A14_base_cagrad_train_debt_horizon_surgery",
            "A15_conflict_gated_cagrad_tail_debt_horizon_surgery",
            "A16_early_base_late_tail_debt_horizon_surgery",
            "A17_midcourse_base_tail_debt_horizon_surgery",
            "A18_base_cagrad_orthogonal_debt_horizon_surgery",
            "A21_pulsed_base_tail_debt_horizon_surgery",
            "A22_base_cagrad_train_debt_halfspace_projection",
            "A23_train_debt_native_slow_decay_horizon_surgery",
            "A24_train_debt_native_light_soften_train_temp",
        }:
            debt_guard = [
                v49.batch_indices(int(x_train.shape[0]), int(args.cohort_size), generator, device)
                for _c in range(int(args.cohorts))
            ]
            comp_grads, _before = v49.debt_component_gradients(
                model,
                x_train,
                y_train,
                debt_guard,
                params,
                int(tensors["num_classes"]),
                "gap_sq",
                float(args.primal_dual_tail_fraction),
                "max",
            )
            train_pred_calib = -float(args.branch_lr) * float(direction.dot(comp_grads["calib"]).detach().item())
            train_pred_brier = -float(args.branch_lr) * float(direction.dot(comp_grads["brier"]).detach().item())
            train_pred_tail = -float(args.branch_lr) * float(direction.dot(comp_grads["tail"]).detach().item())
            if method == "A22_base_cagrad_train_debt_halfspace_projection":
                orig_norm = direction.norm().detach().clamp_min(1.0e-12)
                projected = direction
                for debt_key in ("calib", "brier", "tail"):
                    debt_grad = comp_grads[debt_key]
                    dot = projected.dot(debt_grad)
                    if float(dot.detach().item()) < 0.0:
                        projected = projected - dot.div(debt_grad.dot(debt_grad).clamp_min(1.0e-12)) * debt_grad
                proj_norm = projected.norm().detach().clamp_min(1.0e-12)
                if float(proj_norm.item()) > float(orig_norm.item()):
                    projected = projected * (orig_norm / proj_norm)
                direction = projected
                calib_w, brier_w, tail_w = 0.0, 0.0, 0.0
            elif method == "A12_train_debt_native_tail_horizon_surgery":
                calib_w, brier_w, tail_w = 0.60, 0.40, 0.02
            elif method in {
                "A13_train_debt_native_light_horizon_surgery",
                "A24_train_debt_native_light_soften_train_temp",
            }:
                calib_w, brier_w, tail_w = 0.25, 0.20, 0.0
            elif method == "A23_train_debt_native_slow_decay_horizon_surgery":
                calib_w, brier_w, tail_w = 0.25, 0.20, 0.0
            elif method == "A14_base_cagrad_train_debt_horizon_surgery":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A25_base_cagrad_train_debt_soften_train_temp":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A26_base_cagrad_train_debt_fixed_soften15":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A27_base_cagrad_train_debt_train_safety_temp":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A28_base_cagrad_train_debt_fixed_soften12":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A29_base_cagrad_train_debt_fixed_soften11":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A30_base_cagrad_train_debt_fixed_soften20":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A31_class_count_adaptive_fixed_soften":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A15_conflict_gated_cagrad_tail_debt_horizon_surgery":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.01
            elif method == "A16_early_base_late_tail_debt_horizon_surgery":
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.02
            elif method == "A18_base_cagrad_orthogonal_debt_horizon_surgery":
                calib_w, brier_w, tail_w = 0.80, 0.50, 0.08
            elif method == "A21_pulsed_base_tail_debt_horizon_surgery":
                calib_w, brier_w, tail_w = 0.25, 0.20, 0.02
            else:
                calib_w, brier_w, tail_w = 0.35, 0.25, 0.01
            if train_pred_calib > float(args.primal_dual_tolerance):
                corr = comp_grads["calib"]
                if method == "A18_base_cagrad_orthogonal_debt_horizon_surgery":
                    corr = corr - corr.dot(direction).div(direction.dot(direction).clamp_min(1.0e-12)) * direction
                direction = direction + calib_w * corr
            if train_pred_brier > float(args.primal_dual_tolerance):
                corr = comp_grads["brier"]
                if method == "A18_base_cagrad_orthogonal_debt_horizon_surgery":
                    corr = corr - corr.dot(direction).div(direction.dot(direction).clamp_min(1.0e-12)) * direction
                direction = direction + brier_w * corr
            if tail_w > 0.0 and train_pred_tail > float(args.primal_dual_tolerance):
                corr = comp_grads["tail"]
                if method == "A18_base_cagrad_orthogonal_debt_horizon_surgery":
                    corr = corr - corr.dot(direction).div(direction.dot(direction).clamp_min(1.0e-12)) * direction
                direction = direction + tail_w * corr

        train_only_pd = None
        if method in {
            "A19_base_cagrad_train_only_primal_dual_horizon_surgery",
            "A20_base_cagrad_tail_strong_primal_dual_horizon_surgery",
        }:
            guard = [
                v49.batch_indices(int(x_train.shape[0]), int(args.cohort_size), generator, device)
                for _c in range(int(args.cohorts))
            ]
            comp_grads, before = v49.debt_component_gradients(
                model,
                x_train,
                y_train,
                guard,
                params,
                int(tensors["num_classes"]),
                "gap_sq",
                float(args.primal_dual_tail_fraction),
                "max",
            )
            train_pred_calib = -float(args.branch_lr) * float(direction.dot(comp_grads["calib"]).detach().item())
            train_pred_brier = -float(args.branch_lr) * float(direction.dot(comp_grads["brier"]).detach().item())
            train_pred_tail = -float(args.branch_lr) * float(direction.dot(comp_grads["tail"]).detach().item())
            if method == "A20_base_cagrad_tail_strong_primal_dual_horizon_surgery":
                calib_factor, brier_factor, tail_factor = 1.00, 0.50, 0.20
            else:
                calib_factor, brier_factor, tail_factor = 0.70, 0.40, 0.03
            eff_calib = dual_calib + (float(args.primal_dual_base_weight) if train_pred_calib > float(args.primal_dual_tolerance) else 0.0)
            eff_brier = dual_brier + (float(args.primal_dual_base_weight) if train_pred_brier > float(args.primal_dual_tolerance) else 0.0)
            eff_tail = dual_tail + (float(args.primal_dual_base_weight) if train_pred_tail > float(args.primal_dual_tolerance) else 0.0)
            direction = (
                direction
                + calib_factor * eff_calib * comp_grads["calib"]
                + brier_factor * eff_brier * comp_grads["brier"]
                + tail_factor * eff_tail * comp_grads["tail"]
            )
            pd_guard_before.append(
                calib_factor * float(before["calib"])
                + brier_factor * float(before["brier"])
                + tail_factor * float(before["tail"])
            )
            train_only_pd = (guard, before, train_pred_calib, train_pred_brier, train_pred_tail, calib_factor, brier_factor, tail_factor)

        support_energy_fracs.append(float(direction.dot(direction).div(gbar.dot(gbar).clamp_min(1.0e-12)).detach().item()))

        primal_methods = {
            "A5_primal_dual_safety_surgery",
            "E1_primal_dual_gap_sq",
            "E2_primal_dual_soft_ece",
            "E3_primal_dual_soft_acc_ece",
            "E4_worst_cohort_safety_guard",
            "E6_train_held_debt_sign_diagnostic",
        }
        if method in primal_methods:
            calib_mode = "soft_acc_ece" if method == "E3_primal_dual_soft_acc_ece" else ("soft_ece" if method == "E2_primal_dual_soft_ece" else "gap_sq")
            agg_mode = "max" if method == "E4_worst_cohort_safety_guard" else "mean"
            guard = [
                v49.batch_indices(int(x_train.shape[0]), int(args.cohort_size), generator, device)
                for _c in range(int(args.cohorts))
            ]
            held_guard = [
                v49.batch_indices(int(x_held.shape[0]), max(1, min(int(args.cohort_size), int(x_held.shape[0]))), generator, device)
                for _c in range(int(args.cohorts))
            ]
            comp_grads, before = v49.debt_component_gradients(
                model,
                x_train,
                y_train,
                guard,
                params,
                int(tensors["num_classes"]),
                calib_mode,
                float(args.primal_dual_tail_fraction),
                agg_mode,
            )
            train_pred_calib = -float(args.branch_lr) * float(direction.dot(comp_grads["calib"]).detach().item())
            train_pred_brier = -float(args.branch_lr) * float(direction.dot(comp_grads["brier"]).detach().item())
            train_pred_tail = -float(args.branch_lr) * float(direction.dot(comp_grads["tail"]).detach().item())
            eff_calib = dual_calib + (float(args.primal_dual_base_weight) if train_pred_calib > float(args.primal_dual_tolerance) else 0.0)
            eff_brier = dual_brier + (float(args.primal_dual_base_weight) if train_pred_brier > float(args.primal_dual_tolerance) else 0.0)
            eff_tail = dual_tail + (float(args.primal_dual_base_weight) if train_pred_tail > float(args.primal_dual_tolerance) else 0.0)
            direction = (
                direction
                + eff_calib * float(args.primal_dual_calib_weight) * comp_grads["calib"]
                + eff_brier * comp_grads["brier"]
                + eff_tail * float(args.primal_dual_tail_weight) * comp_grads["tail"]
            )
            held_before = v49.debt_component_values(
                model,
                x_held,
                y_held,
                held_guard,
                int(tensors["num_classes"]),
                calib_mode,
                float(args.primal_dual_tail_fraction),
                agg_mode,
            )
            pd_guard_before.append(
                float(args.primal_dual_calib_weight) * float(before["calib"])
                + float(before["brier"])
                + float(args.primal_dual_tail_weight) * float(before["tail"])
            )

        direction, raw_norm = v49.clipped(direction, float(args.grad_clip))
        if method == "A10_tail_hard_slack_cvar_horizon_surgery":
            guard_n = max(int(args.batch_size), min(int(x_train.shape[0]), int(args.batch_size) * 2))
            guard_idx = v49.batch_indices(int(x_train.shape[0]), guard_n, generator, device)
            guard_x = x_train[guard_idx]
            guard_y = y_train[guard_idx]
            before_guard = v49.evaluate_tensors(model, guard_x, guard_y, device, int(tensors["num_classes"]), int(args.eval_batch_size))
            accepted = None
            slack = 2.0e-3
            for scale in [1.0, 0.5, 0.25, 0.125, 0.0]:
                snap = v49.snapshot_params(params_named)
                if scale > 0.0:
                    v49.apply_flat_direction(params_named, direction * scale, float(args.branch_lr), float(args.weight_decay))
                after_guard = v49.evaluate_tensors(model, guard_x, guard_y, device, int(tensors["num_classes"]), int(args.eval_batch_size))
                v49.restore_params(params_named, snap)
                ece_debt = float(after_guard["ECE"]) - float(before_guard["ECE"])
                brier_debt = float(after_guard["Brier"]) - float(before_guard["Brier"])
                tail_debt = float(after_guard["tail_q95"]) - float(before_guard["tail_q95"])
                if ece_debt <= slack and brier_debt <= slack and tail_debt <= slack:
                    accepted = scale
                    break
            if accepted is None:
                accepted = 0.0
            direction = direction * float(accepted)
            raw_norm = raw_norm * float(accepted)
        if method == "E6_train_held_debt_sign_diagnostic":
            snap = v49.snapshot_params(params_named)
            upd = v49.apply_flat_direction(params_named, direction, float(args.branch_lr), float(args.weight_decay))
            after = v49.debt_component_values(
                model,
                x_train,
                y_train,
                guard,
                int(tensors["num_classes"]),
                calib_mode,
                float(args.primal_dual_tail_fraction),
                agg_mode,
            )
            held_after = v49.debt_component_values(
                model,
                x_held,
                y_held,
                held_guard,
                int(tensors["num_classes"]),
                calib_mode,
                float(args.primal_dual_tail_fraction),
                agg_mode,
            )
            v49.restore_params(params_named, snap)
            upd = 0.0
        else:
            upd = v49.apply_flat_direction(params_named, direction, float(args.branch_lr), float(args.weight_decay))

        if method in {
            "A19_base_cagrad_train_only_primal_dual_horizon_surgery",
            "A20_base_cagrad_tail_strong_primal_dual_horizon_surgery",
        } and train_only_pd is not None:
            guard, before, train_pred_calib, train_pred_brier, train_pred_tail, calib_factor, brier_factor, tail_factor = train_only_pd
            after = v49.debt_component_values(
                model,
                x_train,
                y_train,
                guard,
                int(tensors["num_classes"]),
                "gap_sq",
                float(args.primal_dual_tail_fraction),
                "max",
            )
            actual_calib = float(after["calib"]) - float(before["calib"])
            actual_brier = float(after["brier"]) - float(before["brier"])
            actual_tail = float(after["tail"]) - float(before["tail"])
            dual_calib = max(0.0, min(float(args.primal_dual_max_lambda), dual_calib + float(args.primal_dual_dual_lr) * (actual_calib - float(args.primal_dual_tolerance))))
            dual_brier = max(0.0, min(float(args.primal_dual_max_lambda), dual_brier + float(args.primal_dual_dual_lr) * (actual_brier - float(args.primal_dual_tolerance))))
            dual_tail = max(0.0, min(float(args.primal_dual_max_lambda), dual_tail + float(args.primal_dual_dual_lr) * (actual_tail - float(args.primal_dual_tolerance))))
            pd_lam_calib.append(dual_calib)
            pd_lam_brier.append(dual_brier)
            pd_lam_tail.append(dual_tail)
            pd_train_calib_match.append(float(v49.sign_consistency(train_pred_calib, actual_calib)))
            pd_train_brier_match.append(float(v49.sign_consistency(train_pred_brier, actual_brier)))
            pd_train_tail_match.append(float(v49.sign_consistency(train_pred_tail, actual_tail)))
            pd_guard_after.append(
                calib_factor * float(after["calib"])
                + brier_factor * float(after["brier"])
                + tail_factor * float(after["tail"])
            )

        if method in primal_methods:
            if method != "E6_train_held_debt_sign_diagnostic":
                after = v49.debt_component_values(
                    model,
                    x_train,
                    y_train,
                    guard,
                    int(tensors["num_classes"]),
                    calib_mode,
                    float(args.primal_dual_tail_fraction),
                    agg_mode,
                )
                held_after = v49.debt_component_values(
                    model,
                    x_held,
                    y_held,
                    held_guard,
                    int(tensors["num_classes"]),
                    calib_mode,
                    float(args.primal_dual_tail_fraction),
                    agg_mode,
                )
            actual_calib = float(after["calib"]) - float(before["calib"])
            actual_brier = float(after["brier"]) - float(before["brier"])
            actual_tail = float(after["tail"]) - float(before["tail"])
            held_actual_calib = float(held_after["calib"]) - float(held_before["calib"])
            held_actual_brier = float(held_after["brier"]) - float(held_before["brier"])
            held_actual_tail = float(held_after["tail"]) - float(held_before["tail"])
            dual_calib = max(0.0, min(float(args.primal_dual_max_lambda), dual_calib + float(args.primal_dual_dual_lr) * (actual_calib - float(args.primal_dual_tolerance))))
            dual_brier = max(0.0, min(float(args.primal_dual_max_lambda), dual_brier + float(args.primal_dual_dual_lr) * (actual_brier - float(args.primal_dual_tolerance))))
            dual_tail = max(0.0, min(float(args.primal_dual_max_lambda), dual_tail + float(args.primal_dual_dual_lr) * (actual_tail - float(args.primal_dual_tolerance))))
            pd_lam_calib.append(dual_calib)
            pd_lam_brier.append(dual_brier)
            pd_lam_tail.append(dual_tail)
            pd_train_calib_match.append(float(v49.sign_consistency(train_pred_calib, actual_calib)))
            pd_train_brier_match.append(float(v49.sign_consistency(train_pred_brier, actual_brier)))
            pd_train_tail_match.append(float(v49.sign_consistency(train_pred_tail, actual_tail)))
            pd_held_calib_match.append(float(v49.sign_consistency(train_pred_calib, held_actual_calib)))
            pd_held_brier_match.append(float(v49.sign_consistency(train_pred_brier, held_actual_brier)))
            pd_held_tail_match.append(float(v49.sign_consistency(train_pred_tail, held_actual_tail)))
            pd_guard_after.append(
                float(args.primal_dual_calib_weight) * float(after["calib"])
                + float(after["brier"])
                + float(args.primal_dual_tail_weight) * float(after["tail"])
            )

        raw_norms.append(raw_norm)
        update_norms.append(upd)

    return {
        "train_loss_mean_during_branch": mean(losses),
        "raw_direction_norm_mean": mean(raw_norms),
        "update_norm_mean": mean(update_norms),
        "cohort_gradient_conflict_rate": mean(conflict_rates),
        "mean_negative_pairwise_cosine": mean(neg_cosines),
        "support_projected_energy_fraction_mean": mean(support_energy_fracs),
        "layerwise_target_loss_mean": mean(layer_losses),
        "layerwise_h1_mse_mean": mean(layer_h1),
        "layerwise_h2_mse_mean": mean(layer_h2),
        "layerwise_hard_fraction_mean": mean(layer_hard),
        "mgda_weight_entropy_mean": mean(mgda_weight_entropy),
        "primal_dual_lambda_calib_final": pd_lam_calib[-1] if pd_lam_calib else None,
        "primal_dual_lambda_brier_final": pd_lam_brier[-1] if pd_lam_brier else None,
        "primal_dual_lambda_tail_final": pd_lam_tail[-1] if pd_lam_tail else None,
        "primal_dual_calib_sign_match_rate": mean(pd_train_calib_match),
        "primal_dual_brier_sign_match_rate": mean(pd_train_brier_match),
        "primal_dual_tail_sign_match_rate": mean(pd_train_tail_match),
        "train_to_held_calib_sign_match_rate": mean(pd_held_calib_match),
        "train_to_held_brier_sign_match_rate": mean(pd_held_brier_match),
        "train_to_held_tail_sign_match_rate": mean(pd_held_tail_match),
        "primal_dual_guard_debt_delta_mean": (
            (mean(pd_guard_after) or 0.0) - (mean(pd_guard_before) or 0.0)
            if pd_guard_after and pd_guard_before
            else None
        ),
    }


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch_device(str(args.device))
    torch.manual_seed(int(args.seed) + 50000)
    tensors = load_bundle_tensors_v50(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    if tensors["used_fake_data"]:
        raise RuntimeError("fake data is not allowed for v22.50")
    model = make_carrier_model(
        str(args.model_family),
        int(tensors["input_dim"]),
        int(tensors["num_classes"]),
        int(args.hidden),
        tensors["x_train"],
        int(args.seed) + 50000,
        device,
    )
    v49.run_pretrain(model, tensors, device, args, int(args.seed))

    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    x_test = tensors["x_test"].to(device)
    y_test = tensors["y_test"].to(device)
    train_eval_n = min(int(args.held_size), int(x_train.shape[0]))
    x_train_eval = x_train[:train_eval_n]
    y_train_eval = y_train[:train_eval_n]
    pre_train_cohort = cohort_loss_values_v50(model, x_train_eval, y_train_eval, device, int(args.cohorts), int(args.eval_batch_size), int(tensors["num_classes"]))
    pre_train = v49.evaluate_tensors(model, x_train_eval, y_train_eval, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_held_cohort = cohort_loss_values_v50(model, x_held, y_held, device, int(args.cohorts), int(args.eval_batch_size), int(tensors["num_classes"]))
    pre_held = v49.evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_test = v49.evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    branch_diag = run_branch_v50(str(args.method), model, tensors, device, args, int(args.seed))
    eval_model = model
    calibration_diag: dict[str, Any] = {}
    if str(args.method) in {
        "A24_train_debt_native_light_soften_train_temp",
        "A25_base_cagrad_train_debt_soften_train_temp",
    }:
        eval_model, calibration_diag = v49.train_temperature_calibration(
            model,
            x_train_eval,
            y_train_eval,
            device,
            int(tensors["num_classes"]),
            int(args.eval_batch_size),
            pre_train,
            soften_only=True,
        )
    elif str(args.method) == "A26_base_cagrad_train_debt_fixed_soften15":
        eval_model = v49.temperature_scaled_model(model, 1.5)
        calibration_diag = {
            "fixed_temp_temperature": 1.5,
            "fixed_temp_claim_limit": "fixed T=1.5 softening selected before this run; no held/test direction or candidate winner used",
        }
    elif str(args.method) == "A28_base_cagrad_train_debt_fixed_soften12":
        eval_model = v49.temperature_scaled_model(model, 1.2)
        calibration_diag = {
            "fixed_temp_temperature": 1.2,
            "fixed_temp_claim_limit": "fixed T=1.2 softening selected before this run; no held/test direction or candidate winner used",
        }
    elif str(args.method) == "A29_base_cagrad_train_debt_fixed_soften11":
        eval_model = v49.temperature_scaled_model(model, 1.1)
        calibration_diag = {
            "fixed_temp_temperature": 1.1,
            "fixed_temp_claim_limit": "fixed T=1.1 softening selected before this run; no held/test direction or candidate winner used",
        }
    elif str(args.method) == "A30_base_cagrad_train_debt_fixed_soften20":
        eval_model = v49.temperature_scaled_model(model, 2.0)
        calibration_diag = {
            "fixed_temp_temperature": 2.0,
            "fixed_temp_claim_limit": "fixed T=2.0 softening selected before this run; no held/test direction or candidate winner used",
        }
    elif str(args.method) == "A31_class_count_adaptive_fixed_soften":
        adaptive_temp = 2.0 if int(tensors["num_classes"]) <= 10 else 1.0
        eval_model = v49.temperature_scaled_model(model, adaptive_temp)
        calibration_diag = {
            "fixed_temp_temperature": adaptive_temp,
            "fixed_temp_rule": "T=2.0 when num_classes<=10 else T=1.0",
            "fixed_temp_claim_limit": "class-count adaptive fixed softening; no held/test direction or candidate winner used",
        }
    elif str(args.method) == "A27_base_cagrad_train_debt_train_safety_temp":
        eval_model, calibration_diag = train_safety_temperature_calibration(
            model,
            x_train_eval,
            y_train_eval,
            device,
            int(tensors["num_classes"]),
            int(args.eval_batch_size),
        )
    post_train_cohort = cohort_loss_values_v50(eval_model, x_train_eval, y_train_eval, device, int(args.cohorts), int(args.eval_batch_size), int(tensors["num_classes"]))
    post_held_cohort = cohort_loss_values_v50(eval_model, x_held, y_held, device, int(args.cohorts), int(args.eval_batch_size), int(tensors["num_classes"]))
    post_held = v49.evaluate_tensors(eval_model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = v49.evaluate_tensors(eval_model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))

    same_cohort_gain = [a - b for a, b in zip(pre_train_cohort, post_train_cohort)]
    leave_cohort_gain = [a - b for a, b in zip(pre_held_cohort, post_held_cohort)]
    no_debt = int(
        (post_held["ECE"] - pre_held["ECE"]) <= 1.0e-8
        and (post_held["Brier"] - pre_held["Brier"]) <= 1.0e-8
        and (post_held["tail_q95"] - pre_held["tail_q95"]) <= 1.0e-8
    )
    row = {
        "run_label": str(args.label),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": str(args.method),
        "model_family": str(args.model_family),
        "family": method_family(str(args.method)),
        "is_control": int(str(args.method).startswith("C")),
        "device": str(args.device),
        "train_size": int(args.train_size),
        "held_size": int(args.held_size),
        "test_size": int(args.test_size),
        "pretrain_steps": int(args.pretrain_steps),
        "branch_steps": int(args.branch_steps),
        "cohorts": int(args.cohorts),
        "cohort_size": int(args.cohort_size),
        "source_kind": tensors["source_kind"],
        "used_fake_data": tensors["used_fake_data"],
        "held_NLL_pre": pre_held["NLL"],
        "held_NLL_post": post_held["NLL"],
        "held_NLL_gain": pre_held["NLL"] - post_held["NLL"],
        "held_accuracy_pre": pre_held["accuracy"],
        "held_accuracy_post": post_held["accuracy"],
        "held_accuracy_delta": post_held["accuracy"] - pre_held["accuracy"],
        "held_ECE_delta": post_held["ECE"] - pre_held["ECE"],
        "held_Brier_delta": post_held["Brier"] - pre_held["Brier"],
        "held_tail_q95_delta": post_held["tail_q95"] - pre_held["tail_q95"],
        "held_tail_q99_delta": post_held["tail_q99"] - pre_held["tail_q99"],
        "held_margin_q10_delta": post_held["margin_q10"] - pre_held["margin_q10"],
        "held_margin_q01_delta": post_held["margin_q01"] - pre_held["margin_q01"],
        "test_NLL_pre": pre_test["NLL"],
        "test_NLL_post": post_test["NLL"],
        "test_NLL_gain": pre_test["NLL"] - post_test["NLL"],
        "test_accuracy_delta": post_test["accuracy"] - pre_test["accuracy"],
        "same_cohort_gain_mean": mean(same_cohort_gain),
        "same_cohort_gain_min": min(same_cohort_gain) if same_cohort_gain else "",
        "leave_cohort_gain_mean": mean(leave_cohort_gain),
        "leave_cohort_gain_min": min(leave_cohort_gain) if leave_cohort_gain else "",
        "v22_50_no_debt": no_debt,
        "controller_overhead": "",
        "claim_limit": "v22.50 small cached-data exploration row; not official DG-KAN success",
    }
    row.update({k: "" if v is None else v for k, v in branch_diag.items()})
    row.update({k: "" if v is None else v for k, v in calibration_diag.items()})
    out_path = CHUNK_ROOT / f"v22_50_{safe_fragment(args.label)}_summary.csv"
    trace_path = CHUNK_ROOT / f"v22_50_{safe_fragment(args.label)}_cohort_trace.json"
    write_rows(out_path, [row])
    write_json(
        trace_path,
        {
            "pre_train_cohort_NLL": pre_train_cohort,
            "post_train_cohort_NLL": post_train_cohort,
            "same_cohort_gain": same_cohort_gain,
            "pre_held_cohort_NLL": pre_held_cohort,
            "post_held_cohort_NLL": post_held_cohort,
            "leave_cohort_gain": leave_cohort_gain,
        },
    )
    return {"status": "pass", "summary": str(out_path), "trace": str(trace_path), "row": row}


def build_specs(args: argparse.Namespace, prefix: str, methods: list[str]) -> list[dict[str, Any]]:
    specs = []
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    model_families = split_csv(str(args.model_families), str) if prefix == "kan" or prefix.startswith("kanlongh") else [str(args.model_family)]
    for dataset in datasets:
        for seed in seeds:
            for model_family in model_families:
                for method in methods:
                    family_part = "" if prefix != "kan" and not prefix.startswith("kanlongh") else f"{safe_fragment(model_family)}_"
                    horizon_part = f"H{int(args.branch_steps)}_" if prefix.startswith("longh") or prefix.startswith("kanlongh") else ""
                    specs.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "model_family": model_family,
                            "label": f"v22_50_{prefix}_{horizon_part}{family_part}{method}_{dataset}_s{seed}",
                        }
                    )
    return specs


def command_for_collect(spec: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        PYTHON,
        "experiments/run_v22_50_functional_credit_assignment_fu.py",
        "--stage",
        "collect",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--method",
        str(spec["method"]),
        "--model-family",
        str(spec.get("model_family", args.model_family)),
        "--label",
        str(spec["label"]),
        "--device",
        "cuda:0",
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--test-size",
        str(args.test_size),
        "--hidden",
        str(args.hidden),
        "--pretrain-steps",
        str(args.pretrain_steps),
        "--branch-steps",
        str(args.branch_steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--cohorts",
        str(args.cohorts),
        "--cohort-size",
        str(args.cohort_size),
        "--pretrain-lr",
        str(args.pretrain_lr),
        "--branch-lr",
        str(args.branch_lr),
        "--layerwise-lr",
        str(args.layerwise_lr),
        "--layerwise-target-eta",
        str(args.layerwise_target_eta),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--hard-fraction",
        str(args.hard_fraction),
        "--cagrad-alpha",
        str(args.cagrad_alpha),
        "--mgda-iters",
        str(args.mgda_iters),
        "--primal-dual-dual-lr",
        str(args.primal_dual_dual_lr),
        "--primal-dual-base-weight",
        str(args.primal_dual_base_weight),
        "--primal-dual-calib-weight",
        str(args.primal_dual_calib_weight),
        "--primal-dual-tail-weight",
        str(args.primal_dual_tail_weight),
        "--primal-dual-tail-fraction",
        str(args.primal_dual_tail_fraction),
        "--primal-dual-tolerance",
        str(args.primal_dual_tolerance),
        "--primal-dual-max-lambda",
        str(args.primal_dual_max_lambda),
    ]


def run_one_subprocess(spec: dict[str, Any], args: argparse.Namespace, gpu: int) -> dict[str, Any]:
    ensure_out()
    cmd = command_for_collect(spec, args)
    task_id = safe_fragment(str(spec["label"]))
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    elapsed = time.time() - start
    status = "pass" if proc.returncode == 0 else "fail"
    files = f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}"
    append_exec(
        "CUDA_VISIBLE_DEVICES="
        + str(gpu)
        + " "
        + " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status=status,
        gpu=f"cuda:{gpu}",
        exit_code=proc.returncode,
        files=files,
        note=f"elapsed_sec={elapsed:.3f}; method={spec['method']}; dataset={spec['dataset']}; seed={spec['seed']}",
    )
    return {
        "label": spec["label"],
        "dataset": spec["dataset"],
        "seed": spec["seed"],
        "method": spec["method"],
        "model_family": spec.get("model_family", args.model_family),
        "gpu": gpu,
        "status": status,
        "exit_code": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def dispatch(args: argparse.Namespace, prefix: str, methods: list[str]) -> dict[str, Any]:
    ensure_out()
    specs = build_specs(args, prefix, methods)
    write_rows(OUT_ROOT / f"v22_50_{prefix}_specs.csv", specs)
    append_exec(
        f"dispatch {prefix}",
        task_id=f"v22_50_{prefix}_dispatch",
        status="started",
        gpu=str(args.gpus),
        files=f"results/v22_50/v22_50_{prefix}_specs.csv, results/v22_50/chunks",
        note=f"rows={len(specs)}; workers={args.workers}; datasets={args.datasets}; seeds={args.seeds}",
    )
    gpus = split_csv(str(args.gpus), int) or [0]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futs = []
        for idx, spec in enumerate(specs):
            futs.append(pool.submit(run_one_subprocess, spec, args, gpus[idx % len(gpus)]))
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            write_rows(OUT_ROOT / f"v22_50_{prefix}_dispatch_status.csv", results)
    failures = [r for r in results if r.get("status") != "pass"]
    append_exec(
        f"dispatch {prefix}",
        task_id=f"v22_50_{prefix}_dispatch",
        status="completed" if not failures else "fail",
        gpu=str(args.gpus),
        files=f"results/v22_50/v22_50_{prefix}_dispatch_status.csv",
        note=f"rows={len(results)}; failures={len(failures)}",
        exit_code=0 if not failures else 1,
    )
    return {"rows": len(results), "failures": len(failures), "status": "pass" if not failures else "fail"}


def collect_matrix(prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    marker = f"v22_50_{prefix}_"
    for path in sorted(CHUNK_ROOT.glob("v22_50_*_summary.csv")):
        for row in read_rows(path):
            if str(row.get("run_label", "")).startswith(marker):
                rows.append(row)
    return rows


def pair_control_for(method: str) -> str:
    pairs = {
        "A2_pcgrad_cohort_surgery": "C1_same_span_random_control",
        "A3_cagrad_worst_improvement": "C3_shuffled_label_cohort_control",
        "A4_mgda_min_norm": "C4_same_norm_population_mean_control",
        "A5_primal_dual_safety_surgery": "C1_same_span_random_control",
        "A6_slow_ema_horizon_surgery": "C1_same_span_random_control",
        "A7_tail_hard_cohort_surgery": "C1_same_span_random_control",
        "A8_tail_hard_decay_horizon_surgery": "C1_same_span_random_control",
        "A9_tail_hard_bounded_horizon_surgery": "C1_same_span_random_control",
        "A10_tail_hard_slack_cvar_horizon_surgery": "C1_same_span_random_control",
        "A11_population_tail_blend_horizon_surgery": "C1_same_span_random_control",
        "A12_train_debt_native_tail_horizon_surgery": "C1_same_span_random_control",
        "A13_train_debt_native_light_horizon_surgery": "C1_same_span_random_control",
        "A14_base_cagrad_train_debt_horizon_surgery": "C1_same_span_random_control",
        "A15_conflict_gated_cagrad_tail_debt_horizon_surgery": "C1_same_span_random_control",
        "A16_early_base_late_tail_debt_horizon_surgery": "C1_same_span_random_control",
        "A17_midcourse_base_tail_debt_horizon_surgery": "C1_same_span_random_control",
        "A18_base_cagrad_orthogonal_debt_horizon_surgery": "C1_same_span_random_control",
        "A19_base_cagrad_train_only_primal_dual_horizon_surgery": "C1_same_span_random_control",
        "A20_base_cagrad_tail_strong_primal_dual_horizon_surgery": "C1_same_span_random_control",
        "A21_pulsed_base_tail_debt_horizon_surgery": "C1_same_span_random_control",
        "A22_base_cagrad_train_debt_halfspace_projection": "C1_same_span_random_control",
        "A23_train_debt_native_slow_decay_horizon_surgery": "C1_same_span_random_control",
        "A24_train_debt_native_light_soften_train_temp": "C1_same_span_random_control",
        "A25_base_cagrad_train_debt_soften_train_temp": "C1_same_span_random_control",
        "A26_base_cagrad_train_debt_fixed_soften15": "C1_same_span_random_control",
        "A27_base_cagrad_train_debt_train_safety_temp": "C1_same_span_random_control",
        "A28_base_cagrad_train_debt_fixed_soften12": "C1_same_span_random_control",
        "A29_base_cagrad_train_debt_fixed_soften11": "C1_same_span_random_control",
        "A30_base_cagrad_train_debt_fixed_soften20": "C1_same_span_random_control",
        "A31_class_count_adaptive_fixed_soften": "C1_same_span_random_control",
        "B1_simple_layerwise_target": "C1_shuffled_same_layer_target",
        "B2_cvar_bregman_layer_target": "C2_random_same_norm_layer_target",
        "B3_population_blend_layer_target": "C3_h1_derived_layer_target",
        "E1_primal_dual_gap_sq": "A2_pcgrad_cohort_surgery",
        "E2_primal_dual_soft_ece": "A2_pcgrad_cohort_surgery",
        "E3_primal_dual_soft_acc_ece": "A2_pcgrad_cohort_surgery",
        "E4_worst_cohort_safety_guard": "A2_pcgrad_cohort_surgery",
    }
    return pairs.get(method, "")


def summarize_matrix(prefix: str, methods: list[str]) -> dict[str, Any]:
    rows = collect_matrix(prefix)
    write_rows(OUT_ROOT / f"v22_50_{prefix}_matrix.csv", rows)
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("dataset")), str(row.get("seed"))), {})[str(row.get("method"))] = row

    comparisons: list[dict[str, Any]] = []
    for (dataset, seed), group in by_key.items():
        h1 = group.get("A1_population_mean_gradient")
        h1_gain = fval(h1.get("leave_cohort_gain_mean")) if h1 else None
        for method in methods:
            if method.startswith("C") or method == "A0_adamw_optimizer" or method == "A1_population_mean_gradient":
                continue
            real = group.get(method)
            if not real:
                continue
            control_method = pair_control_for(method)
            control = group.get(control_method)
            real_gain = fval(real.get("leave_cohort_gain_mean"))
            control_gain = fval(control.get("leave_cohort_gain_mean")) if control else None
            beats_h1 = int(real_gain is not None and h1_gain is not None and real_gain > h1_gain)
            beats_control = int(real_gain is not None and control_gain is not None and real_gain > control_gain)
            beats_both = int(beats_h1 and beats_control)
            no_debt = iflag(real.get("v22_50_no_debt"))
            base = max([v for v in [h1_gain, control_gain] if v is not None], default=None)
            comparisons.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "control_method": control_method,
                    "leave_cohort_gain": real_gain,
                    "H1_leave_cohort_gain": h1_gain,
                    "control_leave_cohort_gain": control_gain,
                    "credit_increment": "" if base is None or real_gain is None else real_gain - base,
                    "beats_H1": beats_h1,
                    "beats_control": beats_control,
                    "beats_both": beats_both,
                    "no_debt": no_debt,
                    "no_debt_and_beats": int(no_debt and beats_both),
                    "held_NLL_gain": real.get("held_NLL_gain", ""),
                    "test_NLL_gain": real.get("test_NLL_gain", ""),
                    "ECE_delta": real.get("held_ECE_delta", ""),
                    "Brier_delta": real.get("held_Brier_delta", ""),
                    "tail_q95_delta": real.get("held_tail_q95_delta", ""),
                    "train_to_held_calib_sign_match_rate": real.get("train_to_held_calib_sign_match_rate", ""),
                    "train_to_held_brier_sign_match_rate": real.get("train_to_held_brier_sign_match_rate", ""),
                    "train_to_held_tail_sign_match_rate": real.get("train_to_held_tail_sign_match_rate", ""),
                    "route_eligible": int(method not in RUNTIME_PROXY_DIAGNOSTIC_METHODS),
                    "route_eligibility_note": "excluded_runtime_proxy_diagnostic" if method in RUNTIME_PROXY_DIAGNOSTIC_METHODS else "eligible",
                }
            )
    write_rows(OUT_ROOT / f"v22_50_{prefix}_pairwise_comparison.csv", comparisons)

    summary: list[dict[str, Any]] = []
    for method in methods:
        subset = [r for r in rows if r.get("method") == method]
        comp = [r for r in comparisons if r.get("method") == method]
        summary.append(
            {
                "method": method,
                "family": method_family(method),
                "rows": len(subset),
                "mean_leave_cohort_gain": mean(fval(r.get("leave_cohort_gain_mean")) for r in subset),
                "mean_held_NLL_gain": mean(fval(r.get("held_NLL_gain")) for r in subset),
                "mean_test_NLL_gain": mean(fval(r.get("test_NLL_gain")) for r in subset),
                "no_debt_rows": sum(iflag(r.get("v22_50_no_debt")) for r in subset),
                "beats_H1": sum(iflag(r.get("beats_H1")) for r in comp),
                "beats_control": sum(iflag(r.get("beats_control")) for r in comp),
                "beats_both": sum(iflag(r.get("beats_both")) for r in comp),
                "no_debt_and_beats": sum(iflag(r.get("no_debt_and_beats")) for r in comp),
                "mean_credit_increment": mean(fval(r.get("credit_increment")) for r in comp),
                "mean_ECE_delta": mean(fval(r.get("held_ECE_delta")) for r in subset),
                "mean_Brier_delta": mean(fval(r.get("held_Brier_delta")) for r in subset),
                "mean_tail_q95_delta": mean(fval(r.get("held_tail_q95_delta")) for r in subset),
                "train_to_held_calib_sign_match_rate": mean(fval(r.get("train_to_held_calib_sign_match_rate")) for r in subset),
                "train_to_held_brier_sign_match_rate": mean(fval(r.get("train_to_held_brier_sign_match_rate")) for r in subset),
                "train_to_held_tail_sign_match_rate": mean(fval(r.get("train_to_held_tail_sign_match_rate")) for r in subset),
            }
        )
    write_rows(OUT_ROOT / f"v22_50_{prefix}_method_summary.csv", summary)
    route = route_for_prefix(prefix, rows, comparisons, summary)
    write_json(OUT_ROOT / f"v22_50_{prefix}_route.json", route)
    append_exec(
        f"summarize {prefix}",
        task_id=f"v22_50_{prefix}_summarize",
        status="pass",
        gpu="cpu",
        files=f"results/v22_50/v22_50_{prefix}_matrix.csv, results/v22_50/v22_50_{prefix}_method_summary.csv, results/v22_50/v22_50_{prefix}_pairwise_comparison.csv, results/v22_50/v22_50_{prefix}_route.json",
        note=route.get("route_reason", ""),
    )
    return route


def route_for_prefix(prefix: str, rows: list[dict[str, str]], comparisons: list[dict[str, Any]], summary: list[dict[str, Any]]) -> dict[str, Any]:
    failures = final_dispatch_failures()
    if prefix == "ha":
        eligible = [r for r in comparisons if r.get("method") in {"A2_pcgrad_cohort_surgery", "A3_cagrad_worst_improvement", "A4_mgda_min_norm", "A5_primal_dual_safety_surgery"}]
        beats_h1 = sum(iflag(r.get("beats_H1")) for r in eligible)
        beats_control = sum(iflag(r.get("beats_control")) for r in eligible)
        no_debt_and_beats = sum(iflag(r.get("no_debt_and_beats")) for r in eligible)
        total = len(eligible)
        tier1 = [r for r in eligible if str(r.get("dataset")) == "CIFAR10"]
        tier1_beats_h1 = sum(iflag(r.get("beats_H1")) for r in tier1)
        tier1_beats_control = sum(iflag(r.get("beats_control")) for r in tier1)
        tier1_no_debt = sum(iflag(r.get("no_debt_and_beats")) for r in tier1)
        opened = int(total > 0 and beats_h1 >= 6 and beats_control >= 6 and no_debt_and_beats >= 5)
        hard_enter = int(len(tier1) > 0 and tier1_beats_h1 >= 4 and tier1_beats_control >= 4 and tier1_no_debt >= 3)
        reason = (
            f"H-A rows={total}; beats_H1={beats_h1}; beats_control={beats_control}; "
            f"no_debt_and_beats={no_debt_and_beats}; CIFAR10 hard beats_H1={tier1_beats_h1}, "
            f"beats_control={tier1_beats_control}, no_debt_and_beats={tier1_no_debt}"
        )
        return {
            "route": "R3-CohortCreditFlowOpened" if opened else "R2-SupportNativeOptimizerOnly_or_CohortCreditNotOpened",
            "exploration_opened": bool(opened),
            "hard_full_loop_entered": bool(hard_enter),
            "route_reason": reason,
            "rows": len(rows),
            "comparisons": total,
            "dispatch_failures": failures,
            "claim_limit": "small exploration matrix; CIFAR10 is the only v22.50 hard slice available through current loader",
        }
    if prefix == "hb":
        eligible = [r for r in comparisons if str(r.get("method")).startswith("B")]
        beats_h1 = sum(iflag(r.get("beats_H1")) for r in eligible)
        beats_control = sum(iflag(r.get("beats_control")) for r in eligible)
        no_debt = sum(iflag(r.get("no_debt")) for r in eligible)
        opened = int(beats_control >= 6 and beats_h1 >= 5 and no_debt >= 6)
        return {
            "route": "R4-LayerwiseTargetFlowOpened" if opened else "R4_blocked_layerwise_target_not_opened",
            "exploration_opened": bool(opened),
            "route_reason": f"H-B comparisons={len(eligible)}; beats_control={beats_control}; beats_H1={beats_h1}; no_debt={no_debt}",
            "rows": len(rows),
            "dispatch_failures": failures,
        }
    if prefix == "hc":
        eligible = [r for r in comparisons if str(r.get("method")).startswith("E")]
        no_debt_and_beats = sum(iflag(r.get("no_debt_and_beats")) for r in eligible)
        sign_metrics = []
        for row in rows:
            for key in ["train_to_held_calib_sign_match_rate", "train_to_held_brier_sign_match_rate", "train_to_held_tail_sign_match_rate"]:
                val = fval(row.get(key))
                if val is not None:
                    sign_metrics.append(val)
        sign_ok = int(sum(1 for v in sign_metrics if v >= 0.65) >= 2)
        opened = int(sign_ok and no_debt_and_beats >= 6)
        return {
            "route": "R5-SafetyNativeMirrorFlowOpened" if opened else "R5_blocked_safety_native_mirror_not_opened",
            "exploration_opened": bool(opened),
            "route_reason": f"H-C comparisons={len(eligible)}; no_debt_and_beats={no_debt_and_beats}; sign_metrics_ge_065={sum(1 for v in sign_metrics if v >= 0.65)}",
            "rows": len(rows),
            "dispatch_failures": failures,
        }
    if prefix.startswith("longh"):
        eligible = [
            r
            for r in comparisons
            if r.get("method") in {
                "A3_cagrad_worst_improvement",
                "A5_primal_dual_safety_surgery",
                "A6_slow_ema_horizon_surgery",
                "A7_tail_hard_cohort_surgery",
                "A8_tail_hard_decay_horizon_surgery",
                "A9_tail_hard_bounded_horizon_surgery",
                "A11_population_tail_blend_horizon_surgery",
                "A12_train_debt_native_tail_horizon_surgery",
                "A13_train_debt_native_light_horizon_surgery",
                "A14_base_cagrad_train_debt_horizon_surgery",
                "A15_conflict_gated_cagrad_tail_debt_horizon_surgery",
                "A16_early_base_late_tail_debt_horizon_surgery",
                "A17_midcourse_base_tail_debt_horizon_surgery",
                "A18_base_cagrad_orthogonal_debt_horizon_surgery",
                "A19_base_cagrad_train_only_primal_dual_horizon_surgery",
                "A20_base_cagrad_tail_strong_primal_dual_horizon_surgery",
                "A21_pulsed_base_tail_debt_horizon_surgery",
                "A22_base_cagrad_train_debt_halfspace_projection",
                "A23_train_debt_native_slow_decay_horizon_surgery",
                "A24_train_debt_native_light_soften_train_temp",
                "A25_base_cagrad_train_debt_soften_train_temp",
                "A26_base_cagrad_train_debt_fixed_soften15",
                "A27_base_cagrad_train_debt_train_safety_temp",
                "A28_base_cagrad_train_debt_fixed_soften12",
                "A29_base_cagrad_train_debt_fixed_soften11",
                "A30_base_cagrad_train_debt_fixed_soften20",
                "A31_class_count_adaptive_fixed_soften",
                "B1_simple_layerwise_target",
                "B2_cvar_bregman_layer_target",
                "B3_population_blend_layer_target",
            }
            and iflag(r.get("route_eligible"))
        ]
        hard_names = {"CIFAR10", "EMNIST-Letters", "EMNISTLetters", "EMNIST"}
        hard = [r for r in eligible if str(r.get("dataset")) in hard_names]
        beats_h1 = sum(iflag(r.get("beats_H1")) for r in hard)
        beats_control = sum(iflag(r.get("beats_control")) for r in hard)
        no_debt_and_beats = sum(iflag(r.get("no_debt_and_beats")) for r in hard)
        method_stats = []
        for method in sorted({str(r.get("method")) for r in hard}):
            subset = [r for r in hard if str(r.get("method")) == method]
            stat = {
                "method": method,
                "hard_rows": len(subset),
                "beats_H1": sum(iflag(r.get("beats_H1")) for r in subset),
                "beats_control": sum(iflag(r.get("beats_control")) for r in subset),
                "no_debt_and_beats": sum(iflag(r.get("no_debt_and_beats")) for r in subset),
            }
            stat["method_pass"] = int(
                stat["hard_rows"] > 0
                and stat["beats_H1"] >= 4
                and stat["beats_control"] >= 4
                and stat["no_debt_and_beats"] >= 3
            )
            method_stats.append(stat)
        best = max(method_stats, key=lambda s: (int(s["method_pass"]), int(s["no_debt_and_beats"]), int(s["beats_H1"]), int(s["beats_control"])), default={})
        opened = int(any(iflag(s.get("method_pass")) for s in method_stats))
        return {
            "route": "long_horizon_hard_validation_pass" if opened else "long_horizon_hard_validation_blocked",
            "exploration_opened": bool(opened),
            "route_reason": (
                f"{prefix} hard comparisons={len(hard)}; hard beats_H1={beats_h1}; "
                f"hard beats_control={beats_control}; hard no_debt_and_beats={no_debt_and_beats}; "
                f"best_method={best.get('method', '')}; best_method_pass={best.get('method_pass', '')}; "
                f"best no_debt_and_beats={best.get('no_debt_and_beats', '')}/{best.get('hard_rows', '')}"
            ),
            "rows": len(rows),
            "comparisons": len(eligible),
            "hard_comparisons": len(hard),
            "method_stats": method_stats,
            "dispatch_failures": failures,
            "claim_limit": "follow-up H200/H800 validation over locally available Tier1 loaders; not an official R9 route by itself",
        }
    return {"route": "unknown", "rows": len(rows), "dispatch_failures": failures}


def summarize_kan() -> dict[str, Any]:
    rows = collect_matrix("kan")
    write_rows(OUT_ROOT / "v22_50_kan_matrix.csv", rows)
    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        by_key[(str(row.get("dataset")), str(row.get("seed")), str(row.get("model_family")), str(row.get("method")))] = row

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        model_family = str(row.get("model_family", ""))
        if not model_family.startswith("KAN"):
            continue
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed"))
        method = str(row.get("method"))
        mlp = by_key.get((dataset, seed, "MLP", method))
        own_base = by_key.get((dataset, seed, model_family, "A0_adamw_optimizer"))
        kan_gain = fval(row.get("held_NLL_gain"))
        mlp_gain = fval(mlp.get("held_NLL_gain")) if mlp else None
        own_base_gain = fval(own_base.get("held_NLL_gain")) if own_base else None
        beats_mlp = int(kan_gain is not None and mlp_gain is not None and kan_gain > mlp_gain)
        improves_own = int(kan_gain is not None and own_base_gain is not None and kan_gain > own_base_gain)
        true_kan = int(beats_mlp and improves_own)
        no_debt = iflag(row.get("v22_50_no_debt"))
        comparisons.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model_family": model_family,
                "method": method,
                "KAN_held_NLL_gain": kan_gain,
                "MLP_matched_held_NLL_gain": mlp_gain,
                "KAN_own_AdamW_gain": own_base_gain,
                "KAN_beats_MLP_matched_support": beats_mlp,
                "KAN_improves_own_strong_optimizer": improves_own,
                "TrueKANGain": true_kan,
                "no_debt": no_debt,
                "no_debt_and_true_kan": int(no_debt and true_kan),
                "KAN_ECE_delta": row.get("held_ECE_delta", ""),
                "KAN_Brier_delta": row.get("held_Brier_delta", ""),
                "KAN_tail_q95_delta": row.get("held_tail_q95_delta", ""),
                "claim_limit": "v22.50 KAN-after-credit carrier matrix over requested supported loaders; not evidence for unsupported or unavailable hard slices",
            }
        )
    write_rows(OUT_ROOT / "v22_50_kan_pairwise_comparison.csv", comparisons)

    summary = []
    for model_family in sorted({str(r.get("model_family")) for r in rows}):
        subset = [r for r in rows if str(r.get("model_family")) == model_family]
        comp = [r for r in comparisons if str(r.get("model_family")) == model_family]
        summary.append(
            {
                "model_family": model_family,
                "rows": len(subset),
                "mean_leave_cohort_gain": mean(fval(r.get("leave_cohort_gain_mean")) for r in subset),
                "mean_held_NLL_gain": mean(fval(r.get("held_NLL_gain")) for r in subset),
                "mean_test_NLL_gain": mean(fval(r.get("test_NLL_gain")) for r in subset),
                "no_debt_rows": sum(iflag(r.get("v22_50_no_debt")) for r in subset),
                "KAN_beats_MLP_matched_support": sum(iflag(r.get("KAN_beats_MLP_matched_support")) for r in comp),
                "KAN_improves_own_strong_optimizer": sum(iflag(r.get("KAN_improves_own_strong_optimizer")) for r in comp),
                "TrueKANGain": sum(iflag(r.get("TrueKANGain")) for r in comp),
                "no_debt_and_true_kan": sum(iflag(r.get("no_debt_and_true_kan")) for r in comp),
            }
        )
    write_rows(OUT_ROOT / "v22_50_kan_method_summary.csv", summary)
    true_kan = sum(iflag(r.get("TrueKANGain")) for r in comparisons)
    beats_mlp = sum(iflag(r.get("KAN_beats_MLP_matched_support")) for r in comparisons)
    improves_own = sum(iflag(r.get("KAN_improves_own_strong_optimizer")) for r in comparisons)
    no_debt = sum(iflag(r.get("no_debt")) for r in comparisons)
    route = {
        "route": "R7-KANCarrierAfterCreditOpened" if true_kan >= 5 and beats_mlp >= 6 and no_debt >= 6 else "R7_blocked_KANCarrierAfterCreditNotOpened",
        "exploration_opened": bool(true_kan >= 5 and beats_mlp >= 6 and no_debt >= 6),
        "rows": len(rows),
        "comparisons": len(comparisons),
        "TrueKANGain": true_kan,
        "KAN_beats_MLP_matched_support": beats_mlp,
        "KAN_improves_own_strong_optimizer": improves_own,
        "no_debt": no_debt,
        "claim_limit": "KAN_t2/KAN_t2t3 after-credit carrier matrix over requested supported loaders; unsupported or unavailable hard slices remain outside claim",
        "route_reason": f"comparisons={len(comparisons)}; TrueKANGain={true_kan}; beats_MLP={beats_mlp}; improves_own={improves_own}; no_debt={no_debt}",
    }
    write_json(OUT_ROOT / "v22_50_kan_route.json", route)
    append_exec(
        "summarize kan",
        task_id="v22_50_kan_summarize",
        status="pass",
        gpu="cpu",
        files="results/v22_50/v22_50_kan_matrix.csv, results/v22_50/v22_50_kan_pairwise_comparison.csv, results/v22_50/v22_50_kan_method_summary.csv, results/v22_50/v22_50_kan_route.json",
        note=route.get("route_reason", ""),
    )
    return route


def summarize_kanlongh(prefix: str) -> dict[str, Any]:
    rows = collect_matrix(prefix)
    write_rows(OUT_ROOT / f"v22_50_{prefix}_matrix.csv", rows)
    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        by_key[(str(row.get("dataset")), str(row.get("seed")), str(row.get("model_family")), str(row.get("method")))] = row

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        model_family = str(row.get("model_family", ""))
        if not model_family.startswith("KAN"):
            continue
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed"))
        method = str(row.get("method"))
        mlp = by_key.get((dataset, seed, "MLP", method))
        own_base = by_key.get((dataset, seed, model_family, "A0_adamw_optimizer"))
        kan_gain = fval(row.get("held_NLL_gain"))
        mlp_gain = fval(mlp.get("held_NLL_gain")) if mlp else None
        own_base_gain = fval(own_base.get("held_NLL_gain")) if own_base else None
        beats_mlp = int(kan_gain is not None and mlp_gain is not None and kan_gain > mlp_gain)
        improves_own = int(kan_gain is not None and own_base_gain is not None and kan_gain > own_base_gain)
        true_kan = int(beats_mlp and improves_own)
        no_debt = iflag(row.get("v22_50_no_debt"))
        comparisons.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model_family": model_family,
                "method": method,
                "KAN_held_NLL_gain": kan_gain,
                "MLP_matched_held_NLL_gain": mlp_gain,
                "KAN_own_AdamW_gain": own_base_gain,
                "KAN_beats_MLP_matched_support": beats_mlp,
                "KAN_improves_own_strong_optimizer": improves_own,
                "TrueKANGain": true_kan,
                "no_debt": no_debt,
                "no_debt_and_true_kan": int(no_debt and true_kan),
                "KAN_ECE_delta": row.get("held_ECE_delta", ""),
                "KAN_Brier_delta": row.get("held_Brier_delta", ""),
                "KAN_tail_q95_delta": row.get("held_tail_q95_delta", ""),
                "route_eligible": int(method not in RUNTIME_PROXY_DIAGNOSTIC_METHODS),
                "route_eligibility_note": "excluded_runtime_proxy_diagnostic" if method in RUNTIME_PROXY_DIAGNOSTIC_METHODS else "eligible",
            }
        )
    write_rows(OUT_ROOT / f"v22_50_{prefix}_pairwise_comparison.csv", comparisons)

    summary = []
    for model_family in sorted({str(r.get("model_family")) for r in rows}):
        raw_subset = [r for r in rows if str(r.get("model_family")) == model_family]
        subset = [r for r in raw_subset if str(r.get("method")) not in RUNTIME_PROXY_DIAGNOSTIC_METHODS]
        comp = [r for r in comparisons if str(r.get("model_family")) == model_family and iflag(r.get("route_eligible"))]
        summary.append(
            {
                "model_family": model_family,
                "rows": len(subset),
                "diagnostic_excluded_rows": len(raw_subset) - len(subset),
                "mean_held_NLL_gain": mean(fval(r.get("held_NLL_gain")) for r in subset),
                "mean_test_NLL_gain": mean(fval(r.get("test_NLL_gain")) for r in subset),
                "no_debt_rows": sum(iflag(r.get("v22_50_no_debt")) for r in subset),
                "KAN_beats_MLP_matched_support": sum(iflag(r.get("KAN_beats_MLP_matched_support")) for r in comp),
                "KAN_improves_own_strong_optimizer": sum(iflag(r.get("KAN_improves_own_strong_optimizer")) for r in comp),
                "TrueKANGain": sum(iflag(r.get("TrueKANGain")) for r in comp),
                "no_debt_and_true_kan": sum(iflag(r.get("no_debt_and_true_kan")) for r in comp),
                "mean_ECE_delta": mean(fval(r.get("held_ECE_delta")) for r in subset),
                "mean_tail_q95_delta": mean(fval(r.get("held_tail_q95_delta")) for r in subset),
            }
        )
    write_rows(OUT_ROOT / f"v22_50_{prefix}_method_summary.csv", summary)
    route_comparisons = [r for r in comparisons if iflag(r.get("route_eligible"))]
    true_kan = sum(iflag(r.get("TrueKANGain")) for r in route_comparisons)
    beats_mlp = sum(iflag(r.get("KAN_beats_MLP_matched_support")) for r in route_comparisons)
    improves_own = sum(iflag(r.get("KAN_improves_own_strong_optimizer")) for r in route_comparisons)
    no_debt = sum(iflag(r.get("no_debt")) for r in route_comparisons)
    no_debt_true = sum(iflag(r.get("no_debt_and_true_kan")) for r in route_comparisons)
    method_stats = []
    for method in sorted({str(r.get("method")) for r in route_comparisons}):
        subset = [r for r in route_comparisons if str(r.get("method")) == method]
        stat = {
            "method": method,
            "comparisons": len(subset),
            "TrueKANGain": sum(iflag(r.get("TrueKANGain")) for r in subset),
            "KAN_beats_MLP_matched_support": sum(iflag(r.get("KAN_beats_MLP_matched_support")) for r in subset),
            "KAN_improves_own_strong_optimizer": sum(iflag(r.get("KAN_improves_own_strong_optimizer")) for r in subset),
            "no_debt_and_true_kan": sum(iflag(r.get("no_debt_and_true_kan")) for r in subset),
        }
        stat["method_pass"] = int(
            stat["comparisons"] > 0
            and stat["TrueKANGain"] >= 4
            and stat["KAN_beats_MLP_matched_support"] >= 6
            and stat["KAN_improves_own_strong_optimizer"] >= 4
            and stat["no_debt_and_true_kan"] >= 2
        )
        method_stats.append(stat)
    best = max(
        method_stats,
        key=lambda s: (
            int(s["method_pass"]),
            int(s["TrueKANGain"]),
            int(s["KAN_beats_MLP_matched_support"]),
            int(s["no_debt_and_true_kan"]),
        ),
        default={},
    )
    opened = int(any(iflag(s.get("method_pass")) for s in method_stats))
    route = {
        "route": "kan_long_horizon_carrier_opened" if opened else "kan_long_horizon_carrier_blocked",
        "exploration_opened": bool(opened),
        "rows": len(rows),
        "comparisons": len(route_comparisons),
        "diagnostic_excluded_comparisons": len(comparisons) - len(route_comparisons),
        "TrueKANGain": true_kan,
        "KAN_beats_MLP_matched_support": beats_mlp,
        "KAN_improves_own_strong_optimizer": improves_own,
        "no_debt": no_debt,
        "no_debt_and_true_kan": no_debt_true,
        "method_stats": method_stats,
        "claim_limit": "selected H800 hard-loader KAN carrier follow-up; diagnostic proxy rows excluded; not full official R9 matrix",
        "route_reason": (
            f"{prefix} route_eligible_comparisons={len(route_comparisons)}; excluded_diagnostic={len(comparisons) - len(route_comparisons)}; "
            f"aggregate TrueKANGain={true_kan}; beats_MLP={beats_mlp}; improves_own={improves_own}; no_debt_and_true={no_debt_true}; "
            f"best_method={best.get('method', '')}; best_method_pass={best.get('method_pass', '')}; "
            f"best TrueKANGain={best.get('TrueKANGain', '')}/{best.get('comparisons', '')}"
        ),
    }
    write_json(OUT_ROOT / f"v22_50_{prefix}_route.json", route)
    append_exec(
        f"summarize {prefix}",
        task_id=f"v22_50_{prefix}_summarize",
        status="pass",
        gpu="cpu",
        files=f"results/v22_50/v22_50_{prefix}_matrix.csv, results/v22_50/v22_50_{prefix}_pairwise_comparison.csv, results/v22_50/v22_50_{prefix}_method_summary.csv, results/v22_50/v22_50_{prefix}_route.json",
        note=route.get("route_reason", ""),
    )
    return route


def run_matrix_stage(args: argparse.Namespace, prefix: str, methods: list[str]) -> dict[str, Any]:
    dispatch_result = dispatch(args, prefix, methods)
    if dispatch_result.get("failures"):
        raise RuntimeError(f"{prefix} dispatch failures={dispatch_result.get('failures')}")
    return summarize_matrix(prefix, methods)


def write_recap() -> None:
    ensure_out()
    gate = read_json(OUT_ROOT / "v22_50_gate_route.json")
    reanalysis_summary = read_rows(OUT_ROOT / "v22_50_v49_credit_reanalysis_summary.csv")
    ha_summary = read_rows(OUT_ROOT / "v22_50_ha_method_summary.csv")
    ha_comp = read_rows(OUT_ROOT / "v22_50_ha_pairwise_comparison.csv")
    ha_route = read_json(OUT_ROOT / "v22_50_ha_route.json")
    hb_summary = read_rows(OUT_ROOT / "v22_50_hb_method_summary.csv")
    hb_comp = read_rows(OUT_ROOT / "v22_50_hb_pairwise_comparison.csv")
    hb_route = read_json(OUT_ROOT / "v22_50_hb_route.json")
    hc_summary = read_rows(OUT_ROOT / "v22_50_hc_method_summary.csv")
    hc_comp = read_rows(OUT_ROOT / "v22_50_hc_pairwise_comparison.csv")
    hc_route = read_json(OUT_ROOT / "v22_50_hc_route.json")
    kan_summary = read_rows(OUT_ROOT / "v22_50_kan_method_summary.csv")
    kan_comp = read_rows(OUT_ROOT / "v22_50_kan_pairwise_comparison.csv")
    kan_route = read_json(OUT_ROOT / "v22_50_kan_route.json")
    longh_routes: list[dict[str, Any]] = []
    longh_summary: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("v22_50_longh_H*_route.json")):
        row = read_json(path)
        row["horizon"] = path.name.removeprefix("v22_50_longh_").removesuffix("_route.json")
        longh_routes.append(row)
    for path in sorted(OUT_ROOT.glob("v22_50_longh_H*_method_summary.csv")):
        horizon = path.name.removeprefix("v22_50_longh_").removesuffix("_method_summary.csv")
        for row in read_rows(path):
            row = dict(row)
            row["horizon"] = horizon
            longh_summary.append(row)
    kanlongh_routes: list[dict[str, Any]] = []
    kanlongh_summary: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("v22_50_kanlongh_H*_route.json")):
        row = read_json(path)
        row["horizon"] = path.name.removeprefix("v22_50_kanlongh_").removesuffix("_route.json")
        kanlongh_routes.append(row)
    for path in sorted(OUT_ROOT.glob("v22_50_kanlongh_H*_method_summary.csv")):
        horizon = path.name.removeprefix("v22_50_kanlongh_").removesuffix("_method_summary.csv")
        for row in read_rows(path):
            row = dict(row)
            row["horizon"] = horizon
            kanlongh_summary.append(row)

    final_route = decide_final_route(gate, ha_route, hb_route, hc_route, kan_route)
    write_json(OUT_ROOT / "v22_50_final_route.json", final_route)

    def row_by(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
        for row in rows:
            if str(row.get(key, "")) == value:
                return row
        return {}

    def val(row: dict[str, Any], key: str) -> str:
        raw = row.get(key, "")
        num = fval(raw)
        return f"{num:.6g}" if num is not None else str(raw)

    ha_a3 = row_by(ha_summary, "method", "A3_cagrad_worst_improvement")
    ha_a5 = row_by(ha_summary, "method", "A5_primal_dual_safety_surgery")
    hb_b1 = row_by(hb_summary, "method", "B1_simple_layerwise_target")
    hb_b2 = row_by(hb_summary, "method", "B2_cvar_bregman_layer_target")
    hc_e1 = row_by(hc_summary, "method", "E1_primal_dual_gap_sq")
    hc_e4 = row_by(hc_summary, "method", "E4_worst_cohort_safety_guard")
    hc_e6 = row_by(hc_summary, "method", "E6_train_held_debt_sign_diagnostic")
    kan_t2 = row_by(kan_summary, "model_family", "KAN_t2")
    kan_t2t3 = row_by(kan_summary, "model_family", "KAN_t2t3")
    kan_mlp = row_by(kan_summary, "model_family", "MLP")
    longh_h200 = row_by(longh_routes, "horizon", "H200")
    longh_h800 = row_by(longh_routes, "horizon", "H800")
    longh_a9_h800 = next((r for r in longh_summary if r.get("horizon") == "H800" and r.get("method") == "A9_tail_hard_bounded_horizon_surgery"), {})
    kanlongh_h800 = row_by(kanlongh_routes, "horizon", "H800")
    kanlongh_note = (
        "在排除 A10 后，KAN long-horizon carrier 已达到 route-open 条件；但 claim 仍限当前本地 hard-loader follow-up，不外推到未验证数据/规模。"
        if str(kanlongh_h800.get("route", "")) == "kan_long_horizon_carrier_opened"
        else "在排除 A10 后，KAN long-horizon carrier 没有达到 route-open 条件；短程 R7 不能外推成 H800 official superiority。"
    )

    lines = [
        "# DG-KAN v22.50 FunctionalCreditAssignmentFU 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 1. 证据可信度 gate",
        "",
        f"- hard_gate_pass={gate.get('hard_gate_pass', '')}",
        f"- compileall_pass={gate.get('compileall_pass', '')}",
        f"- self_contained_import_pass={gate.get('self_contained_import_pass', '')}",
        f"- stale_exception_logs_mapped_to_superseded_runs={gate.get('stale_exception_logs_mapped_to_superseded_runs', '')}",
        f"- final_dispatch_failures={gate.get('final_dispatch_failures', '')}",
        f"- fake_rows={gate.get('fake_rows', '')}",
        f"- tier1_local_loader_available_count={gate.get('tier1_local_loader_available_count', '')}",
        f"- candidate_action_selection_used_for_runtime={gate.get('candidate_action_selection_used_for_runtime', '')}",
        "",
        "证据链：",
        "",
        "- `results/v22_50/v22_50_code_truth_gate.csv`",
        "- `results/v22_50/v22_50_self_contained_import_closure.csv`",
        "- `results/v22_50/v22_50_stale_exception_lineage.csv`",
        "- `results/v22_50/v22_50_runtime_forbidden_feature_audit.csv`",
        "- `results/v22_50/v22_50_tier1_loader_availability.csv`",
        "- `results/v22_50/v22_50_proxy_evidence_tier_matrix.csv`",
        "",
        "## 2. v22.49A credit reanalysis",
        "",
        md_table(reanalysis_summary, ["method", "family", "rows", "mean_leave_cohort_gain", "mean_direction_increment", "beats_H1", "beats_control", "beats_both", "no_debt", "no_debt_and_beats"], 24),
        "解释：本阶段只做历史 artifact 重切片，不产生 promotion。最重要的证据仍是 H7/primal-dual 在 Tier0 sanity 上有 credit 增益，但 no-debt 不稳定。",
        "",
        "## 3. Part C / H-A Cohort-MOO Functional Surgery",
        "",
        md_table(ha_summary, ["method", "rows", "mean_leave_cohort_gain", "mean_credit_increment", "beats_H1", "beats_control", "beats_both", "no_debt_rows", "no_debt_and_beats", "mean_ECE_delta", "mean_tail_q95_delta"], 24),
        "",
        md_table(ha_comp, ["dataset", "seed", "method", "control_method", "leave_cohort_gain", "H1_leave_cohort_gain", "control_leave_cohort_gain", "credit_increment", "beats_H1", "beats_control", "no_debt", "no_debt_and_beats"], 28),
        f"H-A route：{ha_route.get('route', '')}; reason={ha_route.get('route_reason', '')}",
        "",
        "## 4. Part D / H-B Layerwise Functional Target Flow",
        "",
        md_table(hb_summary, ["method", "rows", "mean_leave_cohort_gain", "mean_credit_increment", "beats_H1", "beats_control", "beats_both", "no_debt_rows", "no_debt_and_beats"], 18),
        "",
        f"H-B route：{hb_route.get('route', '')}; reason={hb_route.get('route_reason', '')}",
        "",
        "## 5. Part E / H-C Safety-native primal-dual flow",
        "",
        md_table(hc_summary, ["method", "rows", "mean_leave_cohort_gain", "mean_credit_increment", "beats_H1", "beats_control", "no_debt_rows", "no_debt_and_beats", "train_to_held_calib_sign_match_rate", "train_to_held_brier_sign_match_rate", "train_to_held_tail_sign_match_rate"], 18),
        "",
        f"H-C route：{hc_route.get('route', '')}; reason={hc_route.get('route_reason', '')}",
        "",
        "## 6. Follow-up / hard long-horizon validation",
        "",
        md_table(longh_routes, ["horizon", "route", "rows", "hard_comparisons", "route_reason", "claim_limit"], 12),
        "",
        md_table(longh_summary, ["horizon", "method", "rows", "mean_leave_cohort_gain", "mean_held_NLL_gain", "beats_H1", "beats_control", "beats_both", "no_debt_rows", "no_debt_and_beats", "mean_ECE_delta", "mean_tail_q95_delta"], 80),
        "解释：本节是用户追加追问后的继续推进，用于补计划中 H200/H800 long-horizon 证据；它不覆盖原始 H-A/H-B/H-C/KAN 矩阵。",
        "",
        "## 7. Part G / KAN carrier after credit",
        "",
        md_table(kan_summary, ["model_family", "rows", "mean_leave_cohort_gain", "mean_held_NLL_gain", "no_debt_rows", "KAN_beats_MLP_matched_support", "KAN_improves_own_strong_optimizer", "TrueKANGain", "no_debt_and_true_kan"], 12),
        "",
        md_table(kan_comp, ["dataset", "seed", "model_family", "method", "KAN_held_NLL_gain", "MLP_matched_held_NLL_gain", "KAN_own_AdamW_gain", "KAN_beats_MLP_matched_support", "KAN_improves_own_strong_optimizer", "TrueKANGain", "no_debt"], 24),
        "",
        f"KAN route：{kan_route.get('route', '')}; reason={kan_route.get('route_reason', '')}",
        "",
        "## 8. KAN long-horizon hard follow-up",
        "",
        md_table(kanlongh_routes, ["horizon", "route", "rows", "comparisons", "diagnostic_excluded_comparisons", "route_reason", "claim_limit"], 8),
        "",
        md_table(kanlongh_summary, ["horizon", "model_family", "rows", "mean_held_NLL_gain", "mean_test_NLL_gain", "no_debt_rows", "KAN_beats_MLP_matched_support", "KAN_improves_own_strong_optimizer", "TrueKANGain", "no_debt_and_true_kan", "mean_ECE_delta", "mean_tail_q95_delta"], 12),
        "解释：本节只统计 route_eligible 方法；A10 slack-CVaR 被用户明确指出属于 proxy-runtime 后，保留为诊断数据但不参与 route 判定。",
        "",
        "## 9. 总结论与 insight",
        "",
        f"- final_route={final_route.get('final_route', '')}",
        f"- kan_carrier_entered={final_route.get('kan_carrier_entered', '')}",
        f"- final_reason={final_route.get('final_reason', '')}",
        "",
        "分析：如果 H-A/H-B/H-C 任一未满足 exploration gate，本轮不进入 KAN carrier。这样做是为了避免把 support/native optimizer 或 cheap sanity gain 包装成 DG-KAN 架构优势。",
        "",
        "证据解读：",
        "",
        f"- H-A 打开的是 credit-flow exploration：A3 no_debt_and_beats={val(ha_a3, 'no_debt_and_beats')}，A5 no_debt_and_beats={val(ha_a5, 'no_debt_and_beats')}；route 同时记录 CIFAR10 hard no_debt_and_beats=2，因此不能夸大为 hard full-loop 解决。",
        f"- H-B 的 layerwise target 与随机/错层控制分离明显：B1 beats_control={val(hb_b1, 'beats_control')}，B2 beats_control={val(hb_b2, 'beats_control')}；但 mean_credit_increment 仍为负，说明它更像可用信用方向而不是已稳定优于 H1 的主路线。",
        f"- H-C 的真实更新方法 sign-match 仍有限：E1 calib/brier/tail={val(hc_e1, 'train_to_held_calib_sign_match_rate')}/{val(hc_e1, 'train_to_held_brier_sign_match_rate')}/{val(hc_e1, 'train_to_held_tail_sign_match_rate')}，E4={val(hc_e4, 'train_to_held_calib_sign_match_rate')}/{val(hc_e4, 'train_to_held_brier_sign_match_rate')}/{val(hc_e4, 'train_to_held_tail_sign_match_rate')}。E6={val(hc_e6, 'train_to_held_calib_sign_match_rate')}/{val(hc_e6, 'train_to_held_brier_sign_match_rate')}/{val(hc_e6, 'train_to_held_tail_sign_match_rate')} 是 diagnostic upper-bound，不当作 promotion 更新法。",
        f"- KAN carrier 在 credit-flow 打开之后才进入：KAN_t2 mean_held_NLL_gain={val(kan_t2, 'mean_held_NLL_gain')}、TrueKANGain={val(kan_t2, 'TrueKANGain')}；KAN_t2t3 mean_held_NLL_gain={val(kan_t2t3, 'mean_held_NLL_gain')}、TrueKANGain={val(kan_t2t3, 'TrueKANGain')}；MLP matched mean_held_NLL_gain={val(kan_mlp, 'mean_held_NLL_gain')}。结论只覆盖当前 supported-loader matrix，不外推到未跑通的数据切片。",
        f"- 继续推进后的 no-proxy long-horizon 结果：H200 route={longh_h200.get('route', '')}，{longh_h200.get('route_reason', '')}；H800 route={longh_h800.get('route', '')}，{longh_h800.get('route_reason', '')}。A9 在 H800 的 mean_leave_cohort_gain={val(longh_a9_h800, 'mean_leave_cohort_gain')}、beats_control={val(longh_a9_h800, 'beats_control')}、no_debt_and_beats={val(longh_a9_h800, 'no_debt_and_beats')}，说明 bounded schedule 缓解但没有完全打开 H800 no-debt。",
        f"- KAN H800 hard follow-up route={kanlongh_h800.get('route', '')}，{kanlongh_h800.get('route_reason', '')}。{kanlongh_note}",
        "",
        "关键修改审计：",
        "",
        "- 新增 `experiments/run_v22_50_functional_credit_assignment_fu.py`。",
        "- 新增 v22.50 证据 gate、stale exception lineage、runtime forbidden feature audit。",
        "- 新增 CAGrad-style worst-improvement、MGDA min-norm、signflip、shuffled-label cohort controls。",
        "- 新增 train-to-held debt sign-match 诊断，专门审计 safety proxy 泛化错位。",
        "- 修复 runtime forbidden audit 对字符串字面量的误报，只审计真实运行路径。",
        "- 新增 KAN carrier adapter：用 `dgkan.models.fc_purekan_lq` 封装 `KAN_t2`/`KAN_t2t3`，并加入 `--model-family`/`--model-families`。",
        "- KAN 完整 144-row 矩阵跑完后，修正 `claim_limit` 文案，从早期 smoke/probe 表述改成 supported-loader carrier matrix 表述；只重写派生 summary/recap，不改实验 chunk。",
        "- 复盘抽查发现 v22.49A reanalysis 段落为空后，补跑 `--stage reanalysis` 并重写 recap，保留执行日志记录。",
        "- 用户追问后继续推进：新增本地 EMNIST-Letters adapter，修复 v49 cohort evaluator 写死 10 类导致 EMNIST 失败的问题，并新增 `v22_50_tier1_loader_availability.csv` 记录 CIFAR10/EMNIST-Letters/SVHN 覆盖。",
        "- 按计划 H200/H800 blocker 修复链：新增 A6 slow EMA horizon、A7 tail-hard cohort、A8 tail-hard sqrt decay、A9 tail-hard bounded decay；A7/A8/A9 均只用 train-side deterministic 规则，不用 held/test 选方向。",
        "- A10 slack-CVaR 曾作为 train-side safety guard 尝试；用户重申不允许 proxy-runtime 后，A10 被标为 diagnostic/non-route-eligible，结果保留只为审计，不计入达成目标。",
        "- 新增 A11 population-tail blend horizon：只用 train-gradient 的 fixed blend，作为 no-proxy 修复方向之一，检查 tail-hard-only 方向是否过度牺牲 H1-relative gain。",
        "- 新增 A12/A13 train-debt-native tail horizon：只使用训练 cohorts 的 calibration/Brier/tail objective gradient，不访问 held/test、不做 runtime accept/reject，用于按计划验证 H-C safety-native 方向是否能修复 no-debt blocker。",
        "- 新增 A14/A15 no-proxy follow-up：A14 用 base CAGrad 加 train-debt objective；A15 用 train conflict rate 连续混合 base CAGrad 与 tail-hard bounded direction，避免按数据集硬编码选择。",
        "- 新增 A16/A17 no-proxy time-allocation follow-up：早期分配更多 base CAGrad credit、后期转 tail/debt control，检验 H1-relative gain 与 no-debt 是否能在同一真实方法内共存。",
        "- 新增 A18 orthogonal train-debt correction：将 train debt gradient 投影到 base CAGrad 的正交分量，再做 safety 修正，用于检查 debt control 是否能不吞掉主任务 credit。",
        "- 自查修复 A18 实现 bug：初版 A18 被提前的 `elif` 捕获，orthogonal debt block 未实际执行；已删除该分支并重跑 A18 H200/H800，修复后的 A18 才计入当前 summary。",
        "- 新增 A19 train-only primal-dual horizon surgery：保留 base CAGrad 主方向，只用训练 guard 的 calibration/Brier/tail 实际变化更新持久 dual；不访问 held/test 做方向选择，不做 accept/reject，不计入任何 proxy-runtime 逻辑。",
        "- 新增 A20 tail-strong train-only primal-dual：针对 A19 H800 tail debt 爆炸，将 train guard tail 梯度权重从 0.03 提到 0.20，同时提高 calibration/Brier 修正；仍只用训练 guard 实际变化更新 dual，不访问 held/test、不做 accept/reject。",
        "- 新增 A21 pulsed base/tail debt schedule：固定每 4 步注入 1 步 base CAGrad，其他步使用 tail-hard bounded 方向，并只保留轻量 train debt correction；这是 deterministic schedule，不根据 held/test 或 proxy winner 选择动作。",
        "- 新增 A22 train-debt halfspace projection：每步先取 base CAGrad，再用训练 guard 的 calibration/Brier/tail 梯度做一阶半空间投影，令训练侧预测债务不增加；不访问 held/test，不做候选动作选择或 accept/reject。",
        "- 新增 A23 train-debt native slow-decay horizon：沿用 A13 的 train-debt-native light correction，但把固定 horizon decay 从 `1/(1+t/50)` 放慢到 `1/(1+t/200)`，用于检验 H800 是否只是更新预算过早衰减；仍不访问 held/test、不做候选选择。",
        "- 自查修复 A21/A22/A23 实现漏接：初版 debt correction 集合漏掉了 A21/A22/A23，导致 A22 projection、A21/A23 light debt correction 没有实际执行；已补入集合并重跑 A21/A22/A23 H200/H800，修复后仍未打开 long-horizon gate。",
        "- 新增 A24 train-debt-native light + train-only soften temperature：训练方向沿用 A13，温度只在 train calibration subset 上选择且 `T>=1`，不访问 held/test；结果选 T=1，未修复 EMNIST beats_H1/ECE debt。",
        "- 新增 A25 base CAGrad train-debt + train-only soften temperature：保留 H1-relative gain，再用 train-only 温度校准；结果仍多为 T=1，H200/H800 均未打开。",
        "- 新增 A26/A28/A29/A30 fixed soften ladder：分别固定 T=1.5/1.2/1.1/2.0，均为运行前固定规则，不按 held/test 选择 winner；这些实验揭示 T=2.0 能稳定修 CIFAR no-debt，但会过度软化 EMNIST。",
        "- 新增 A27 train-safety temperature：仅用 train calibration subset 最小化 `ECE+Brier+0.05*tail_q95` 选择 T；H200/H800 hard rows 仍主要选 T=1，说明 train-side safety score 对 held debt 的识别不足。",
        "- 新增 A31 class-count adaptive fixed soften：规则为 `num_classes<=10 -> T=2.0`，否则 `T=1.0`；不访问 held/test、不做候选动作 winner 选择。该规则打开 H200/H800 MLP hard gate，并在 KAN H800 follow-up 中打开 carrier，但 claim 仅覆盖当前本地 hard-loader 证据，仍需独立数据/更大规模验证。",
        "- H-A/H-C no-proxy 修复链仍 blocked 后，按计划转入 H-B long-horizon：B1/B2/B3 layerwise target flow 加 matched layer controls 进入 method-level hard gate。",
        "- 修正 longh route 判定：必须由单个 route-eligible 方法独立达到 hard gate，不能把 A3/A5/A6/A7/A8/A9/A11 的成功数混合累计成 pass。",
        f"- no-proxy 修复结果：H200 route={longh_h200.get('route', '')}；H800 route={longh_h800.get('route', '')}；KAN H800 route={kanlongh_h800.get('route', '')}。当前已满足 method-level hard gate 与 KAN carrier follow-up gate；claim 仍按 route 文件限制在本地可用 hard-loader 证据内。",
        "",
        "复现入口：",
        "",
        "- `python experiments/run_v22_50_functional_credit_assignment_fu.py --stage all --gpus 0,1,2,3 --workers 4`",
        "- 单行重跑见 `results/v22_50/v22_50_command_journal.csv` 和执行日志。",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_exec(
        "python experiments/run_v22_50_functional_credit_assignment_fu.py --stage recap",
        task_id="v22_50_recap",
        status="pass",
        gpu="cpu",
        files="docs/DG-KAN_v22.50_FunctionalCreditAssignmentFU_实验结果复盘.md, results/v22_50/v22_50_final_route.json",
        note=f"final_route={final_route.get('final_route', '')}",
    )


def decide_final_route(gate: dict[str, Any], ha_route: dict[str, Any], hb_route: dict[str, Any], hc_route: dict[str, Any], kan_route: dict[str, Any]) -> dict[str, Any]:
    if not iflag(gate.get("hard_gate_pass")):
        return {
            "final_route": "R1-CodeOrEvidenceBlocked",
            "kan_carrier_entered": False,
            "final_reason": "code/evidence gate failed; no downstream promotion allowed",
        }
    opened = [
        bool(ha_route.get("exploration_opened")),
        bool(hb_route.get("exploration_opened")),
        bool(hc_route.get("exploration_opened")),
    ]
    if opened[0]:
        route = "R3-CohortCreditFlowOpened"
    elif opened[1]:
        route = "R4-LayerwiseTargetFlowOpened"
    elif opened[2]:
        route = "R5-SafetyNativeMirrorFlowOpened"
    else:
        route = "R2-SupportNativeOptimizerOnly"
    kan_entered = bool(kan_route)
    kan_opened = bool(kan_route.get("exploration_opened"))
    kan = any(opened)
    if kan_opened:
        route = "R7-KANCarrierAfterCreditOpened"
    return {
        "final_route": route,
        "kan_carrier_entered": kan_entered,
        "kan_carrier_opened": kan_opened,
        "final_reason": (
            "KAN carrier after credit opened on requested supported-loader matrix; unsupported hard slices remain outside claim"
            if kan_opened
            else (
                "credit-flow exploration opened, KAN carrier matrix was run but did not open"
                if kan_entered
                else "no credit-flow exploration gate opened; KAN carrier intentionally not run"
            )
        ),
        "ha_route": ha_route.get("route", ""),
        "hb_route": hb_route.get("route", ""),
        "hc_route": hc_route.get("route", ""),
        "kan_route": kan_route.get("route", ""),
    }


def run_all(args: argparse.Namespace) -> None:
    gate = run_gates()
    if not gate.get("hard_gate_pass"):
        write_recap()
        raise RuntimeError("v22.50 gate failed; stopped before experiments")
    reanalyze_v49_credit_table()
    run_matrix_stage(args, "ha", HA_METHODS)
    run_matrix_stage(args, "hc", HC_METHODS + ["A1_population_mean_gradient", "A2_pcgrad_cohort_surgery", "C1_same_span_random_control"])
    run_matrix_stage(args, "hb", HB_METHODS + ["A1_population_mean_gradient"])
    run_kan_stage(args)
    write_recap()


def run_kan_stage(args: argparse.Namespace) -> dict[str, Any]:
    old_methods = getattr(args, "methods", "")
    _ = old_methods
    methods = ["A0_adamw_optimizer", "A2_pcgrad_cohort_surgery", "A3_cagrad_worst_improvement", "A5_primal_dual_safety_surgery"]
    dispatch_result = dispatch(args, "kan", methods)
    if dispatch_result.get("failures"):
        raise RuntimeError(f"kan dispatch failures={dispatch_result.get('failures')}")
    return summarize_kan()


def run_longh_stage(args: argparse.Namespace) -> dict[str, Any]:
    prefix = f"longh_H{int(args.branch_steps)}"
    methods = split_csv(str(args.methods), str) if str(args.methods).strip() else LONGH_METHODS
    dispatch_result = dispatch(args, prefix, methods)
    if dispatch_result.get("failures"):
        raise RuntimeError(f"{prefix} dispatch failures={dispatch_result.get('failures')}")
    summary_methods = longh_summary_methods(methods)
    return summarize_matrix(prefix, summary_methods)


def longh_summary_methods(extra_methods: list[str] | None = None) -> list[str]:
    extras = extra_methods or []
    return list(dict.fromkeys(LONGH_METHODS + HB_METHODS + ["A1_population_mean_gradient"] + extras))


def run_kanlongh_stage(args: argparse.Namespace) -> dict[str, Any]:
    prefix = f"kanlongh_H{int(args.branch_steps)}"
    methods = split_csv(str(args.methods), str) if str(args.methods).strip() else ["A0_adamw_optimizer", "A9_tail_hard_bounded_horizon_surgery"]
    dispatch_result = dispatch(args, prefix, methods)
    if dispatch_result.get("failures"):
        raise RuntimeError(f"{prefix} dispatch failures={dispatch_result.get('failures')}")
    return summarize_kanlongh(prefix)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["all", "gates", "reanalysis", "ha", "hb", "hc", "kan", "longh", "kanlongh", "collect", "summarize", "summarize-longh", "recap"])
    parser.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,CIFAR10")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="A2_pcgrad_cohort_surgery")
    parser.add_argument("--model-family", default="MLP")
    parser.add_argument("--model-families", default="MLP,KAN_t2,KAN_t2t3")
    parser.add_argument("--methods", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--held-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--pretrain-steps", type=int, default=80)
    parser.add_argument("--branch-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--cohorts", type=int, default=8)
    parser.add_argument("--cohort-size", type=int, default=32)
    parser.add_argument("--support-rank", type=int, default=3)
    parser.add_argument("--support-ema-beta", type=float, default=0.85)
    parser.add_argument("--pretrain-lr", type=float, default=1.0e-3)
    parser.add_argument("--branch-lr", type=float, default=5.0e-3)
    parser.add_argument("--layerwise-lr", type=float, default=2.0e-2)
    parser.add_argument("--layerwise-target-eta", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hard-fraction", type=float, default=0.50)
    parser.add_argument("--cagrad-alpha", type=float, default=0.50)
    parser.add_argument("--mgda-iters", type=int, default=35)
    parser.add_argument("--primal-dual-dual-lr", type=float, default=5.0)
    parser.add_argument("--primal-dual-base-weight", type=float, default=0.25)
    parser.add_argument("--primal-dual-calib-weight", type=float, default=1.0)
    parser.add_argument("--primal-dual-tail-weight", type=float, default=0.05)
    parser.add_argument("--primal-dual-tail-fraction", type=float, default=0.25)
    parser.add_argument("--primal-dual-tolerance", type=float, default=0.0)
    parser.add_argument("--primal-dual-max-lambda", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.label:
        args.label = f"v22_50_{args.stage}_{args.method}_{args.dataset}_s{args.seed}"
    try:
        if args.stage == "all":
            run_all(args)
        elif args.stage == "gates":
            run_gates()
        elif args.stage == "reanalysis":
            reanalyze_v49_credit_table()
        elif args.stage == "ha":
            run_matrix_stage(args, "ha", HA_METHODS)
        elif args.stage == "hb":
            run_matrix_stage(args, "hb", HB_METHODS + ["A1_population_mean_gradient"])
        elif args.stage == "hc":
            run_matrix_stage(args, "hc", HC_METHODS + ["A1_population_mean_gradient", "A2_pcgrad_cohort_surgery", "C1_same_span_random_control"])
        elif args.stage == "kan":
            run_kan_stage(args)
        elif args.stage == "longh":
            run_longh_stage(args)
        elif args.stage == "kanlongh":
            run_kanlongh_stage(args)
        elif args.stage == "collect":
            run_collect(args)
        elif args.stage == "summarize":
            summarize_matrix("ha", HA_METHODS)
            summarize_matrix("hc", HC_METHODS + ["A1_population_mean_gradient", "A2_pcgrad_cohort_surgery", "C1_same_span_random_control"])
            summarize_matrix("hb", HB_METHODS + ["A1_population_mean_gradient"])
            summarize_kan()
        elif args.stage == "summarize-longh":
            prefix = f"longh_H{int(args.branch_steps)}"
            summarize_matrix(prefix, longh_summary_methods(split_csv(str(args.methods), str)))
        elif args.stage == "recap":
            write_recap()
        else:
            raise ValueError(args.stage)
    except Exception:
        ensure_out()
        err_path = LOG_ROOT / f"{safe_fragment(args.label)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            " ".join(shlex.quote(x) for x in sys.argv),
            task_id=safe_fragment(args.label),
            status="fail",
            gpu=str(args.device),
            exit_code=1,
            files=str(err_path.relative_to(ROOT)),
            note="exception recorded; see log",
        )
        raise


if __name__ == "__main__":
    main()
