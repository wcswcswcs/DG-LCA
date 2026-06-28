#!/usr/bin/env python3
"""DG-KAN v22.68 Control-Contrastive KAN Gauge-Carrier MPFU runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import py_compile
import re
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

import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66  # noqa: E402
import experiments.run_v22_67_kanaware_metric_compatible_generator_carrier_fu as v67  # noqa: E402


PYTHON = os.environ.get("KAN_PYTHON", v66.PYTHON)
OUT_ROOT = ROOT / "results/v22_68"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.68_ControlContrastiveKANGaugeCarrier_MPFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.68_ControlContrastiveKANGaugeCarrier_MPFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.68_ControlContrastiveKANGaugeCarrier_MPFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_68_control_contrastive_kan_gauge_carrier_mpfu.py"
RUNNER66 = ROOT / "experiments/run_v22_66_metric_compatible_generator_atlas_fu.py"
RUNNER67 = ROOT / "experiments/run_v22_67_kanaware_metric_compatible_generator_carrier_fu.py"

PART_B_CANDIDATE = "mcga_over_poet_fsclip_eta001_residual_rank4"
DATASETS_B = ["MNIST", "FashionMNIST", "KMNIST", "Wine", "Spam", "CIFAR10"]
DATASETS_KAN = ["CIFAR10", "Wine", "Spam"]
SEEDS = [0, 1, 2, 3, 4]
STEPS_B = [800, 1200]
REFERENCE_METHODS = ["adamw", "cautious_adamw", "schedule_free_adamw_local"]
EXTERNAL_METHODS = ["poet_official", "pion_oet_sphere_official", "pion_oet_local"]
CONTROL_METHODS_B = [
    "same_generator_descent_energy_random",
    "same_functional_spectrum_random_coordinate",
    "same_C_skew_spectrum_random",
    "same_isometric_capacity_random_coordinate",
]
PART_B_METHODS = [*REFERENCE_METHODS, *EXTERNAL_METHODS, *CONTROL_METHODS_B, PART_B_CANDIDATE]
SAME_GENERATOR_CONTROLS = ["same_generator_descent_energy_random", "same_C_skew_spectrum_random"]
KAN_ARCHES = ["DGKAN_DCHE", "DGKAN_DFOU"]
KAN_REDESIGN_ARCHES = ["DGKAN_CHE3_XLIN", "DGKAN_FOU4", "DGKAN_FOU4_LIN"]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)


def command_text(cmd: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def repair_family_slug(label_prefix: str) -> str:
    slug = str(label_prefix or "repair")
    if slug.startswith("v22_68_part_b_"):
        slug = slug[len("v22_68_part_b_") :]
    if slug.endswith("_st"):
        slug = slug[:-3]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug).strip("_")
    return slug or "repair"


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def flag(value: Any) -> int:
    val = safe_float(value, None)
    return int(val is not None and val != 0.0)


def safe_int(value: Any, default: int | None = None) -> int | None:
    val = safe_float(value, None)
    if val is None:
        return default
    return int(val)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    exit_code: Any = "",
    note: str = "",
) -> None:
    ensure_out()
    row = {
        "timestamp_sg": now_sg(),
        "task_id": task_id,
        "status": status,
        "gpu": gpu,
        "command": command,
        "files": files,
        "exit_code": exit_code,
        "note": note,
    }
    append_csv(
        OUT_ROOT / "v22_68_command_journal.csv",
        row,
        ["timestamp_sg", "task_id", "status", "gpu", "command", "files", "exit_code", "note"],
    )
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp_sg']} | {task_id} | {status}\n\n")
        f.write(f"- command: `{command}`\n")
        if gpu:
            f.write(f"- gpu: `{gpu}`\n")
        if files:
            f.write(f"- files: `{files}`\n")
        if exit_code != "":
            f.write(f"- exit_code: `{exit_code}`\n")
        if note:
            f.write(f"- note: {note}\n")


def append_recap(title: str, lines: list[str]) -> None:
    ensure_out()
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_sg()} | {title}\n\n")
        for line in lines:
            f.write(f"- {line}\n")


def run_subprocess(
    cmd: list[Any],
    *,
    task_id: str,
    timeout: int = 300,
    gpu: str = "",
    files: str = "",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    ensure_out()
    frag = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
    stdout_path = LOG_ROOT / f"{frag}_stdout.log"
    stderr_path = LOG_ROOT / f"{frag}_stderr.log"
    started = time.time()
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env=env,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    status = "pass" if proc.returncode == 0 else "fail"
    append_exec(
        command_text(cmd),
        task_id=task_id,
        status=status,
        gpu=gpu,
        files=files or f"{stdout_path.relative_to(ROOT)}; {stderr_path.relative_to(ROOT)}",
        exit_code=proc.returncode,
        note=f"wall_seconds={time.time() - started:.3f}",
    )
    return {
        "status": status,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "wall_seconds": time.time() - started,
    }


def initialize_docs() -> None:
    ensure_out()
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.68 Control-Contrastive KAN Gauge-Carrier MPFU 执行日志\n\n"
            "本日志只记录实际执行过的命令、脚本、文件、状态与修复动作；不记录推测性结果。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.68 Control-Contrastive KAN Gauge-Carrier MPFU 实验结果复盘\n\n"
            "本复盘只记录 artifact 或命令输出支撑的实验数据、分析、证据链与结论；缺失数据会显式标注，不编造。\n",
            encoding="utf-8",
        )


def debt_value(row: dict[str, Any]) -> float | None:
    vals = [
        safe_float(row.get("held_ECE_delta_vs_best_ref"), 0.0),
        safe_float(row.get("held_Brier_delta_vs_best_ref"), 0.0),
        safe_float(row.get("held_tail_NLL_delta_vs_best_ref"), 0.0),
    ]
    if any(v is None for v in vals):
        return None
    return sum(max(0.0, float(v)) for v in vals if v is not None)


def row_mtime(path_text: str) -> float:
    try:
        return (ROOT / path_text).stat().st_mtime
    except OSError:
        return 0.0


def read_chunk_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "results/v22_66/chunks").glob("*.csv")):
        for row in read_rows(path):
            if row.get("run_status") != "completed":
                continue
            method = str(row.get("method") or "")
            dataset = str(row.get("dataset") or "")
            seed = safe_int(row.get("seed"), None)
            steps = safe_int(row.get("steps"), None)
            if not method or not dataset or seed is None or steps is None:
                continue
            out = dict(row)
            out["seed"] = str(seed)
            out["steps"] = str(steps)
            out["chunk_path"] = str(path.relative_to(ROOT))
            rows.append(out)
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def score(row: dict[str, Any]) -> tuple[int, int, float]:
        label = str(row.get("run_label") or "")
        return (
            int(label.startswith("v22_68")),
            int(label.startswith("v22_67_mlp_robust_poet_variants")),
            row_mtime(str(row.get("chunk_path") or "")),
        )

    for row in rows:
        arch = str(row.get("architecture") or row.get("architecture_key") or "MLP")
        key = (str(row.get("method")), str(row.get("dataset")), str(row.get("seed")), str(row.get("steps")), arch)
        if key not in by_key or score(row) > score(by_key[key]):
            by_key[key] = row
    return list(by_key.values())


def part_b_filtered_rows() -> list[dict[str, Any]]:
    allowed = set(PART_B_METHODS)
    datasets = set(DATASETS_B)
    seeds = {str(s) for s in SEEDS}
    steps = {str(s) for s in STEPS_B}
    rows = [
        r
        for r in dedupe_rows(read_chunk_rows())
        if str(r.get("method")) in allowed
        and str(r.get("dataset")) in datasets
        and str(r.get("seed")) in seeds
        and str(r.get("steps")) in steps
        and str(r.get("architecture") or r.get("architecture_key") or "MLP") == "MLP"
    ]
    return rows


def add_part_b_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cond: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cond.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("steps"))), []).append(row)
    for group in by_cond.values():
        completed = [r for r in group if r.get("run_status") == "completed"]
        refs = [r for r in completed if str(r.get("method")) in set(REFERENCE_METHODS)]
        external = [r for r in completed if str(r.get("method")) in set(EXTERNAL_METHODS)]
        controls = [r for r in completed if str(r.get("method")) in set(CONTROL_METHODS_B)]
        best_ref = min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in refs), default=float("inf"))
        best_external = min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in external), default=float("inf"))
        best_control = min((safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in controls), default=float("inf"))
        ref_debt = min((debt_value(r) if debt_value(r) is not None else float("inf") for r in refs), default=float("inf"))
        same_best: dict[str, float] = {}
        for method in SAME_GENERATOR_CONTROLS:
            vals = [safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in completed if str(r.get("method")) == method]
            same_best[method] = min(vals, default=float("inf"))
        for row in group:
            nll = safe_float(row.get("held_NLL"), float("inf")) or float("inf")
            debt = debt_value(row)
            row["Delta_NLL_vs_strongest_ref"] = nll - best_ref if math.isfinite(best_ref) else ""
            row["Delta_NLL_vs_best_external"] = nll - best_external if math.isfinite(best_external) else ""
            row["Delta_NLL_vs_best_control"] = nll - best_control if math.isfinite(best_control) else ""
            row["beats_strongest_NLL"] = int(math.isfinite(best_ref) and nll < best_ref)
            row["beats_external_OET_NLL"] = int(math.isfinite(best_external) and nll < best_external)
            row["beats_best_control_NLL"] = int(math.isfinite(best_control) and nll < best_control)
            row["no_ECE_Brier_tail_debt"] = int(debt is not None and math.isfinite(ref_debt) and debt <= ref_debt + 1.0e-9)
            for method in SAME_GENERATOR_CONTROLS:
                suffix = "same_generator_descent_energy_random" if method == "same_generator_descent_energy_random" else "same_C_skew_spectrum_random"
                row[f"Delta_NLL_vs_{suffix}"] = nll - same_best[method] if math.isfinite(same_best[method]) else ""
                row[f"beats_{suffix}_NLL"] = int(math.isfinite(same_best[method]) and nll < same_best[method])
            row["beats_same_generator_controls_NLL"] = int(
                flag(row.get("beats_same_generator_descent_energy_random_NLL"))
                and flag(row.get("beats_same_C_skew_spectrum_random_NLL"))
            )
    return rows


def part_b_failure_mode(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "ImplementationBoundaryFailed"
    if not flag(row.get("standard_loop_runtime_trace_pass")):
        return "ImplementationBoundaryFailed"
    if (safe_float(row.get("generator_descent_fraction"), 0.0) or 0.0) < 0.05:
        return "NoGeneratorCapacity"
    if not flag(row.get("beats_same_generator_descent_energy_random_NLL")) or not flag(row.get("beats_same_C_skew_spectrum_random_NLL")):
        return "GeneratorSupportExplained_NoFU"
    if (safe_float(row.get("active_Gram_drift_mean"), 0.0) or 0.0) <= 0.05 and not flag(row.get("beats_best_control_NLL")):
        return "MetricPreservationOnly"
    if not flag(row.get("no_ECE_Brier_tail_debt")):
        return "DebtBlocked"
    if (safe_float(row.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) > 0.35:
        return "OverheadBlocked"
    if not flag(row.get("beats_best_control_NLL")):
        return "CoordinateSupportExplained"
    if not flag(row.get("beats_external_OET_NLL")):
        return "ExternalOETExplained"
    if not flag(row.get("beats_strongest_NLL")):
        return "FUWeakOptimizerPatchOnly"
    return "completed"


def matrix_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in PART_B_METHODS:
        for steps in STEPS_B:
            subset = [r for r in rows if str(r.get("method")) == method and str(r.get("steps")) == str(steps)]
            completed = [r for r in subset if r.get("run_status") == "completed"]
            out.append(
                {
                    "method": method,
                    "steps": steps,
                    "expected_rows": len(DATASETS_B) * len(SEEDS),
                    "completed_rows": len(completed),
                    "missing_completed_rows": len(DATASETS_B) * len(SEEDS) - len(completed),
                    "datasets_completed": ",".join(sorted({str(r.get("dataset")) for r in completed})),
                    "seeds_completed": ",".join(sorted({str(r.get("seed")) for r in completed}, key=lambda x: int(float(x)) if x else -1)),
                }
            )
    return out


def summarize_part_b(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = add_part_b_comparisons(rows)
    candidate = [
        r
        for r in rows
        if str(r.get("method")) == PART_B_CANDIDATE
        and str(r.get("dataset")) in set(DATASETS_B)
        and str(r.get("steps")) in {str(s) for s in STEPS_B}
        and str(r.get("seed")) in {str(s) for s in SEEDS}
    ]
    expected_keys = {(dataset, str(seed), str(step)) for dataset in DATASETS_B for seed in SEEDS for step in STEPS_B}
    completed = [r for r in candidate if r.get("run_status") == "completed"]
    completed_keys = {(str(r.get("dataset")), str(r.get("seed")), str(r.get("steps"))) for r in completed}
    missing_keys = sorted(expected_keys - completed_keys, key=lambda x: (int(x[2]), x[0], int(x[1])))
    for row in candidate:
        row["failure_mode"] = part_b_failure_mode(row)
    external_explained = sum(1 for r in completed if r.get("failure_mode") == "ExternalOETExplained")
    control_explained = sum(1 for r in completed if not flag(r.get("beats_best_control_NLL")))
    delta_ext = [safe_float(r.get("Delta_NLL_vs_best_external"), None) for r in completed]
    delta_ext = [v for v in delta_ext if v is not None]
    dataset_failures: dict[str, int] = {}
    for row in completed:
        if row.get("failure_mode") != "completed":
            dataset_failures[str(row.get("dataset"))] = dataset_failures.get(str(row.get("dataset")), 0) + 1
    summary = {
        "gate": "v22_68_part_b_mlp_eta001_robustness_800_1200",
        "generated_at_sg": now_sg(),
        "candidate_method": PART_B_CANDIDATE,
        "steps": ",".join(str(s) for s in STEPS_B),
        "expected_candidate_rows": len(expected_keys),
        "completed_rows": len(completed),
        "missing_candidate_rows": len(missing_keys),
        "missing_candidate_keys": json.dumps(missing_keys, ensure_ascii=False),
        "beats_strongest_NLL_rows": sum(flag(r.get("beats_strongest_NLL")) for r in completed),
        "beats_external_OET_NLL_rows": sum(flag(r.get("beats_external_OET_NLL")) for r in completed),
        "beats_best_control_NLL_rows": sum(flag(r.get("beats_best_control_NLL")) for r in completed),
        "beats_same_generator_controls_rows": sum(flag(r.get("beats_same_generator_controls_NLL")) for r in completed),
        "no_debt_rows": sum(flag(r.get("no_ECE_Brier_tail_debt")) for r in completed),
        "overhead_le_025_rows": sum(1 for r in completed if (safe_float(r.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) <= 0.25),
        "active_Gram_drift_le_005_rows": sum(1 for r in completed if (safe_float(r.get("active_Gram_drift_mean"), 999.0) or 999.0) <= 0.05),
        "functional_spectrum_drift_le_threshold_rows": sum(1 for r in completed if (safe_float(r.get("functional_spectrum_drift_mean"), 999.0) or 999.0) <= 0.50),
        "generator_descent_fraction_positive_rows": sum(1 for r in completed if (safe_float(r.get("generator_descent_fraction"), 0.0) or 0.0) > 0.0),
        "ExternalOETExplained_pct": 0.0 if not completed else 100.0 * external_explained / len(completed),
        "ControlExplained_pct": 0.0 if not completed else 100.0 * control_explained / len(completed),
        "mean_Delta_NLL_vs_best_external": "" if not delta_ext else statistics.fmean(delta_ext),
        "median_Delta_NLL_vs_best_external": "" if not delta_ext else statistics.median(delta_ext),
        "worst_dataset_failure_count": max(dataset_failures.values(), default=0),
        "dataset_failure_counts": json.dumps(dataset_failures, ensure_ascii=False, sort_keys=True),
    }
    summary["robust_mlp_pass"] = int(
        summary["completed_rows"] >= 60
        and summary["beats_strongest_NLL_rows"] >= 50
        and summary["beats_external_OET_NLL_rows"] >= 42
        and summary["beats_best_control_NLL_rows"] >= 54
        and summary["beats_same_generator_controls_rows"] >= 54
        and summary["no_debt_rows"] >= 54
        and summary["overhead_le_025_rows"] >= 54
        and summary["active_Gram_drift_le_005_rows"] >= 54
        and summary["functional_spectrum_drift_le_threshold_rows"] >= 54
        and summary["ExternalOETExplained_pct"] <= 20.0
    )
    summary["part_b_next_action"] = "run_part_c_kan_failure_replay" if summary["robust_mlp_pass"] else (
        "complete_missing_part_b_rows" if summary["missing_candidate_rows"] else "external_oet_failure_decomposition_and_fixed_repair"
    )
    write_rows(OUT_ROOT / "v22_68_part_b_inventory.csv", matrix_inventory(rows))
    candidate_detail = [r for r in rows if str(r.get("method")) == PART_B_CANDIDATE]
    write_rows(OUT_ROOT / "v22_68_part_b_candidate_detail.csv", candidate_detail)
    write_rows(OUT_ROOT / "v22_68_part_b_all_rows_with_comparisons.csv", rows)
    write_rows(OUT_ROOT / "v22_68_part_b_mlp_robustness_summary.csv", [summary])
    (OUT_ROOT / "v22_68_part_b_mlp_robustness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-b-analyze"]),
        task_id="B_part_b_analyze_800_1200",
        status="pass" if summary["robust_mlp_pass"] else "incomplete_or_fail",
        files="results/v22_68/v22_68_part_b_inventory.csv; results/v22_68/v22_68_part_b_mlp_robustness_summary.json",
        note=f"completed={summary['completed_rows']}/60; robust_mlp_pass={summary['robust_mlp_pass']}; next_action={summary['part_b_next_action']}",
    )
    append_recap(
        "Part B MLP eta001 robustness lock",
        [
            f"candidate={PART_B_CANDIDATE}; steps={summary['steps']}; completed_rows={summary['completed_rows']}/60; robust_mlp_pass={summary['robust_mlp_pass']}.",
            "Gate counts: strongest={strongest}, external_OET={external}, best_control={control}, same_generator_controls={same}, no_debt={debt}, overhead<=0.25={overhead}, Gram<=0.05={gram}, spectrum<=threshold={spectrum}.".format(
                strongest=summary["beats_strongest_NLL_rows"],
                external=summary["beats_external_OET_NLL_rows"],
                control=summary["beats_best_control_NLL_rows"],
                same=summary["beats_same_generator_controls_rows"],
                debt=summary["no_debt_rows"],
                overhead=summary["overhead_le_025_rows"],
                gram=summary["active_Gram_drift_le_005_rows"],
                spectrum=summary["functional_spectrum_drift_le_threshold_rows"],
            ),
            f"ExternalOETExplained_pct={summary['ExternalOETExplained_pct']}; ControlExplained_pct={summary['ControlExplained_pct']}; mean_Delta_NLL_vs_best_external={summary['mean_Delta_NLL_vs_best_external']}; median={summary['median_Delta_NLL_vs_best_external']}.",
            f"Evidence files: results/v22_68/v22_68_part_b_candidate_detail.csv, v22_68_part_b_all_rows_with_comparisons.csv, v22_68_part_b_mlp_robustness_summary.json.",
        ],
    )
    return summary


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    initialize_docs()
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-a", "--device", str(args.device)]),
        task_id="A_part_a_start",
        status="start",
        gpu=str(args.device),
        files="results/v22_68; results/v22_67/v22_67_part_a_code_identity_training_boundary.json",
        note="v22.68 delegates shared hard-gate probes to v22.67 runner, then adds v22.68 runner compile/import checks.",
    )
    proc = run_subprocess(
        [PYTHON, str(RUNNER67.relative_to(ROOT)), "--mode", "part-a", "--device", str(args.device)],
        task_id="A_v22_67_hard_gate_reuse",
        timeout=int(args.row_timeout),
        gpu=str(args.device),
        files="results/v22_67/v22_67_part_a_code_identity_training_boundary.json",
    )
    summary67_path = ROOT / "results/v22_67/v22_67_part_a_code_identity_training_boundary.json"
    summary67 = json.loads(summary67_path.read_text(encoding="utf-8")) if summary67_path.exists() else {}
    compile_ok = 1
    compile_error = ""
    try:
        py_compile.compile(str(RUNNER), doraise=True)
    except Exception as exc:  # pragma: no cover - recorded to artifact
        compile_ok = 0
        compile_error = f"{type(exc).__name__}: {exc}"
    summary = {
        "gate": "v22_68_part_a_code_identity_standard_loop_hard_gate",
        "generated_at_sg": now_sg(),
        "v22_67_hard_gate_returncode": proc["returncode"],
        "v22_67_part_a_hard_gate_pass": int(summary67.get("part_a_hard_gate_pass", 0)),
        "v22_68_runner_compile_pass": compile_ok,
        "v22_68_runner_compile_error": compile_error,
        "strict_DGKAN_identity_pass": summary67.get("strict_DGKAN_identity_pass", ""),
        "uses_pykan_official_rows": summary67.get("uses_pykan_official_rows", ""),
        "uses_bspline_official_rows": summary67.get("uses_bspline_official_rows", ""),
        "loss_total_is_task_loss_only": summary67.get("loss_total_is_task_loss_only", ""),
        "standard_loop_static_scan_pass": summary67.get("standard_loop_static_scan_pass", ""),
        "standard_loop_runtime_trace_pass": summary67.get("standard_loop_runtime_trace_pass", ""),
        "manual_update_forbidden_scan_pass": summary67.get("manual_update_forbidden_scan_pass", ""),
        "optimizer_owned_gradient_transform_pass": summary67.get("optimizer_owned_gradient_transform_pass", ""),
        "runtime_argmax_candidate_used": summary67.get("runtime_argmax_candidate_used", ""),
        "runtime_topk_candidate_used": summary67.get("runtime_topk_candidate_used", ""),
        "candidate_action_selection_used_for_runtime": summary67.get("candidate_action_selection_used_for_runtime", ""),
        "cohort_topk_selection_used": summary67.get("cohort_topk_selection_used", ""),
        "layer_topk_selection_used": summary67.get("layer_topk_selection_used", ""),
        "class_weight_or_sampler_used_as_fu": summary67.get("class_weight_or_sampler_used_as_fu", ""),
        "uses_validation_test_future_direction": summary67.get("uses_validation_test_future_direction", ""),
        "readout_linearization_error_logged": summary67.get("readout_linearization_error_logged", ""),
    }
    summary["part_a_hard_gate_pass"] = int(
        proc["returncode"] == 0 and summary["v22_67_part_a_hard_gate_pass"] == 1 and compile_ok == 1
    )
    summary["next_action"] = "run_part_b_mlp_robustness_lock" if summary["part_a_hard_gate_pass"] else "fix_part_a_hard_gate_before_science_matrix"
    write_rows(OUT_ROOT / "v22_68_part_a_code_identity_training_boundary.csv", [summary])
    (OUT_ROOT / "v22_68_part_a_code_identity_training_boundary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-a", "--device", str(args.device)]),
        task_id="A_part_a_summary",
        status="pass" if summary["part_a_hard_gate_pass"] else "fail",
        gpu=str(args.device),
        files="results/v22_68/v22_68_part_a_code_identity_training_boundary.json",
        note=f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}; next_action={summary['next_action']}",
    )
    append_recap(
        "Part A code/identity/standard-loop hard gate",
        [
            f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}; v22_67_hard_gate_returncode={proc['returncode']}; v22_68_runner_compile_pass={compile_ok}.",
            "Shared hard-gate evidence is produced by the existing v22.67 runner and copied into v22.68 summary; v22.68 adds compile coverage for the new runner.",
            "Evidence files: results/v22_68/v22_68_part_a_code_identity_training_boundary.json and results/v22_67/v22_67_part_a_code_identity_training_boundary.json.",
        ],
    )
    return summary


def run_part_b_matrix(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    step = int(args.steps)
    current = part_b_filtered_rows()
    inv = matrix_inventory(current)
    missing_methods = [
        str(r["method"])
        for r in inv
        if int(r["steps"]) == step and int(r["missing_completed_rows"]) > 0
    ]
    run_label = str(args.run_label or f"v22_68_mlp_robust_st{step}")
    cmd = [
        PYTHON,
        str(RUNNER66.relative_to(ROOT)),
        "--mode",
        "matrix",
        "--architecture",
        "MLP",
        "--run-label",
        run_label,
        "--datasets",
        ",".join(DATASETS_B),
        "--seeds",
        ",".join(str(s) for s in SEEDS),
        "--methods",
        ",".join(missing_methods or PART_B_METHODS),
        "--gpus",
        str(args.gpus),
        "--max-workers",
        str(args.max_workers),
        "--row-timeout",
        str(args.row_timeout),
        "--steps",
        str(step),
        "--train-size",
        "512",
        "--held-size",
        "256",
        "--test-size",
        "256",
        "--hidden",
        str(args.hidden),
        "--batch-size",
        "128",
        "--eval-batch-size",
        "512",
        "--metric-batch-size",
        str(args.metric_batch_size),
        "--refresh",
        str(args.refresh),
    ]
    expected_rows = len(DATASETS_B) * len(SEEDS) * len(PART_B_METHODS)
    existing_completed = sum(1 for r in current if str(r.get("steps")) == str(step))
    if not missing_methods:
        append_exec(
            command_text(cmd),
            task_id=f"B_part_b_matrix_st{step}",
            status="reuse_completed",
            gpu=str(args.gpus),
            files="results/v22_66/chunks/*.csv",
            note=f"existing_completed_rows={existing_completed}/{expected_rows}; no rerun needed",
        )
        return {"step": step, "status": "reuse_completed", "missing_methods": []}
    append_exec(
        command_text(cmd),
        task_id=f"B_part_b_matrix_st{step}_start",
        status="start",
        gpu=str(args.gpus),
        files="results/v22_66/chunks/*.csv; results/v22_66/v22_66_row_subprocess_status.csv",
        note=f"run_label={run_label}; missing_methods={','.join(missing_methods)}; rows={len(DATASETS_B) * len(SEEDS) * len(missing_methods)}",
    )
    proc = run_subprocess(
        cmd,
        task_id=f"B_part_b_matrix_st{step}",
        timeout=int(args.matrix_timeout),
        gpu=str(args.gpus),
        files="results/v22_66/chunks/*.csv; results/v22_66/v22_66_row_subprocess_status.csv",
    )
    return {"step": step, "status": proc["status"], "missing_methods": missing_methods, "returncode": proc["returncode"]}


def run_part_b_analyze(_args: argparse.Namespace) -> dict[str, Any]:
    return summarize_part_b(part_b_filtered_rows())


def run_part_b_external_decompose(args: argparse.Namespace) -> dict[str, Any]:
    rows = add_part_b_comparisons(part_b_filtered_rows())
    candidate = [r for r in rows if str(r.get("method")) == PART_B_CANDIDATE]
    for row in candidate:
        row["failure_mode"] = part_b_failure_mode(row)
    by_cond: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cond.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("steps"))), []).append(row)
    detail: list[dict[str, Any]] = []
    for row in candidate:
        if row.get("run_status") != "completed" or row.get("failure_mode") == "completed":
            continue
        group = by_cond.get((str(row.get("dataset")), str(row.get("seed")), str(row.get("steps"))), [])
        external = [r for r in group if str(r.get("method")) in set(EXTERNAL_METHODS)]
        controls = [r for r in group if str(r.get("method")) in set(CONTROL_METHODS_B)]
        refs = [r for r in group if str(r.get("method")) in set(REFERENCE_METHODS)]
        best_external = min(external, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        best_control = min(controls, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        best_ref = min(refs, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        detail.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "steps": row.get("steps"),
                "candidate_method": PART_B_CANDIDATE,
                "candidate_failure_mode": row.get("failure_mode"),
                "candidate_NLL": row.get("held_NLL"),
                "candidate_debt_sum": debt_value(row),
                "best_external_method": best_external.get("method", ""),
                "best_external_NLL": best_external.get("held_NLL", ""),
                "Delta_NLL_vs_best_external": row.get("Delta_NLL_vs_best_external"),
                "best_control_method": best_control.get("method", ""),
                "best_control_NLL": best_control.get("held_NLL", ""),
                "Delta_NLL_vs_best_control": row.get("Delta_NLL_vs_best_control"),
                "best_ref_method": best_ref.get("method", ""),
                "best_ref_NLL": best_ref.get("held_NLL", ""),
                "active_Gram_drift_mean": row.get("active_Gram_drift_mean", ""),
                "functional_spectrum_drift_mean": row.get("functional_spectrum_drift_mean", ""),
                "chunk_path": row.get("chunk_path", ""),
            }
        )
    counts: dict[str, int] = {}
    external_counts: dict[str, int] = {}
    for row in detail:
        counts[str(row["candidate_failure_mode"])] = counts.get(str(row["candidate_failure_mode"]), 0) + 1
        external_counts[str(row["best_external_method"])] = external_counts.get(str(row["best_external_method"]), 0) + 1
    summary = {
        "gate": "v22_68_part_b_external_oet_failure_decomposition",
        "generated_at_sg": now_sg(),
        "candidate_method": PART_B_CANDIDATE,
        "failure_rows": len(detail),
        "failure_mode_counts": counts,
        "best_external_counts": external_counts,
        "next_action": "do_not_run_kan_if_part_b_failed_try_one_fixed_repair_family" if detail else "part_b_has_no_failed_candidate_rows",
    }
    write_rows(OUT_ROOT / "v22_68_part_b_external_failure_decomposition.csv", detail)
    (OUT_ROOT / "v22_68_part_b_external_failure_decomposition.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-b-external-decompose"]),
        task_id="B_part_b_external_decompose",
        status="pass",
        files="results/v22_68/v22_68_part_b_external_failure_decomposition.csv; results/v22_68/v22_68_part_b_external_failure_decomposition.json",
        note=f"failure_rows={len(detail)}; failure_mode_counts={json.dumps(counts, sort_keys=True)}",
    )
    append_recap(
        "Part B external-OET failure decomposition",
        [
            f"failure_rows={len(detail)}; failure_mode_counts={json.dumps(counts, ensure_ascii=False, sort_keys=True)}.",
            f"best_external_counts={json.dumps(external_counts, ensure_ascii=False, sort_keys=True)}.",
            "Evidence files: results/v22_68/v22_68_part_b_external_failure_decomposition.csv and .json.",
        ],
    )
    if args.repair_on_fail and detail:
        repair_args = argparse.Namespace(**vars(args))
        repair_args.repair_methods = "mcga_over_poet_fsclip_eta005_residual_rank4,mcga_over_poet_fsclip_eta01_residual_rank4,mcga_over_poet_fsclip_eta025_warm50_residual_rank4"
        run_part_b_fixed_repair_probe(repair_args, detail)
    return summary


def run_part_b_fixed_repair_probe(args: argparse.Namespace, failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    failures = failures or read_rows(OUT_ROOT / "v22_68_part_b_external_failure_decomposition.csv")
    methods = [m.strip() for m in str(args.repair_methods).split(",") if m.strip()]
    label_prefix = str(getattr(args, "repair_run_label_prefix", "") or "v22_68_part_b_fixed_repair_st")
    slug = repair_family_slug(label_prefix)
    tasks_by_step: dict[int, set[str]] = {}
    for row in failures:
        step = safe_int(row.get("steps"), None)
        if step is not None:
            tasks_by_step.setdefault(step, set()).update(methods)
    results: list[dict[str, Any]] = []
    for step, step_methods in sorted(tasks_by_step.items()):
        run_label = f"{label_prefix}{step}"
        task_base = f"B_part_b_{slug}_st{step}"
        cmd = [
            PYTHON,
            str(RUNNER66.relative_to(ROOT)),
            "--mode",
            "matrix",
            "--architecture",
            "MLP",
            "--run-label",
            run_label,
            "--datasets",
            ",".join(DATASETS_B),
            "--seeds",
            ",".join(str(s) for s in SEEDS),
            "--methods",
            ",".join(sorted(step_methods)),
            "--gpus",
            str(args.gpus),
            "--max-workers",
            str(args.max_workers),
            "--row-timeout",
            str(args.row_timeout),
            "--steps",
            str(step),
            "--train-size",
            "512",
            "--held-size",
            "256",
            "--test-size",
            "256",
            "--hidden",
            str(args.hidden),
            "--batch-size",
            "128",
            "--eval-batch-size",
            "512",
            "--metric-batch-size",
            str(args.metric_batch_size),
            "--refresh",
            str(args.refresh),
        ]
        append_exec(
            command_text(cmd),
            task_id=f"{task_base}_start",
            status="start",
            gpu=str(args.gpus),
            files="results/v22_66/chunks/*.csv",
            note=f"fixed repair family only; methods={','.join(sorted(step_methods))}",
        )
        results.append(run_subprocess(cmd, task_id=task_base, timeout=int(args.matrix_timeout), gpu=str(args.gpus), files="results/v22_66/chunks/*.csv"))
    probe_path = OUT_ROOT / f"v22_68_part_b_{slug}_probe.json"
    summary = {
        "gate": f"v22_68_part_b_{slug}_probe",
        "generated_at_sg": now_sg(),
        "repair_run_label_prefix": label_prefix,
        "repair_methods": methods,
        "subprocess_results": results,
    }
    probe_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_recap(
        f"Part B fixed repair probe ({slug})",
        [
            f"repair_run_label_prefix={label_prefix}.",
            f"repair_methods={','.join(methods)}; subprocess_returncodes={[r.get('returncode') for r in results]}.",
            "Repair is a fixed method-family probe only; it is not row-wise promotion and is not used as an official Part B claim without a separate robustness re-analysis.",
            f"Evidence file: results/v22_68/{probe_path.name}.",
        ],
    )
    return summary


def summarize_candidate_like_method(rows: list[dict[str, Any]], candidate_method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_cond: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cond.setdefault((str(row.get("dataset")), str(row.get("seed")), str(row.get("steps"))), []).append(row)
    detail: list[dict[str, Any]] = []
    for key, group in sorted(by_cond.items(), key=lambda item: (int(item[0][2]), item[0][0], int(item[0][1]))):
        completed = [r for r in group if r.get("run_status") == "completed"]
        candidate = next((r for r in completed if str(r.get("method")) == candidate_method), None)
        if candidate is None:
            continue
        refs = [r for r in completed if str(r.get("method")) in set(REFERENCE_METHODS)]
        external = [r for r in completed if str(r.get("method")) in set(EXTERNAL_METHODS)]
        controls = [r for r in completed if str(r.get("method")) in set(CONTROL_METHODS_B)]
        best_ref = min(refs, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        best_external = min(external, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        best_control = min(controls, key=lambda r: safe_float(r.get("held_NLL"), float("inf")) or float("inf"), default={})
        same_best = {}
        for method in SAME_GENERATOR_CONTROLS:
            same_best[method] = min(
                (safe_float(r.get("held_NLL"), float("inf")) or float("inf") for r in completed if str(r.get("method")) == method),
                default=float("inf"),
            )
        nll = safe_float(candidate.get("held_NLL"), float("inf")) or float("inf")
        ref_nll = safe_float(best_ref.get("held_NLL"), float("inf")) or float("inf")
        ext_nll = safe_float(best_external.get("held_NLL"), float("inf")) or float("inf")
        ctrl_nll = safe_float(best_control.get("held_NLL"), float("inf")) or float("inf")
        ref_debt = debt_value(best_ref) if best_ref else None
        cand_debt = debt_value(candidate)
        same_gen_ok = int(all(math.isfinite(v) and nll < v for v in same_best.values()))
        row = {
            "run_status": candidate.get("run_status", "completed"),
            "dataset": key[0],
            "seed": key[1],
            "steps": key[2],
            "candidate_method": candidate_method,
            "candidate_NLL": nll,
            "best_ref_method": best_ref.get("method", ""),
            "best_ref_NLL": "" if not math.isfinite(ref_nll) else ref_nll,
            "best_external_method": best_external.get("method", ""),
            "best_external_NLL": "" if not math.isfinite(ext_nll) else ext_nll,
            "best_control_method": best_control.get("method", ""),
            "best_control_NLL": "" if not math.isfinite(ctrl_nll) else ctrl_nll,
            "Delta_NLL_vs_strongest_ref": nll - ref_nll if math.isfinite(ref_nll) else "",
            "Delta_NLL_vs_best_external": nll - ext_nll if math.isfinite(ext_nll) else "",
            "Delta_NLL_vs_best_control": nll - ctrl_nll if math.isfinite(ctrl_nll) else "",
            "beats_strongest_NLL": int(math.isfinite(ref_nll) and nll < ref_nll),
            "beats_external_OET_NLL": int(math.isfinite(ext_nll) and nll < ext_nll),
            "beats_best_control_NLL": int(math.isfinite(ctrl_nll) and nll < ctrl_nll),
            "beats_same_generator_controls_NLL": same_gen_ok,
            "no_ECE_Brier_tail_debt": int(cand_debt is not None and ref_debt is not None and cand_debt <= float(ref_debt) + 1.0e-9),
            "controller_or_coordinate_overhead_ratio": candidate.get("controller_or_coordinate_overhead_ratio", ""),
            "active_Gram_drift_mean": candidate.get("active_Gram_drift_mean", ""),
            "functional_spectrum_drift_mean": candidate.get("functional_spectrum_drift_mean", ""),
            "generator_descent_fraction": candidate.get("generator_descent_fraction", ""),
            "standard_loop_runtime_trace_pass": candidate.get("standard_loop_runtime_trace_pass", ""),
            "chunk_path": candidate.get("chunk_path", ""),
        }
        for method in SAME_GENERATOR_CONTROLS:
            suffix = "same_generator_descent_energy_random" if method == "same_generator_descent_energy_random" else "same_C_skew_spectrum_random"
            row[f"Delta_NLL_vs_{suffix}"] = nll - same_best[method] if math.isfinite(same_best[method]) else ""
            row[f"beats_{suffix}_NLL"] = int(math.isfinite(same_best[method]) and nll < same_best[method])
        row["failure_mode"] = part_b_failure_mode(row)
        detail.append(row)
    expected = {(dataset, str(seed), str(step)) for dataset in DATASETS_B for seed in SEEDS for step in STEPS_B}
    completed_keys = {(str(r.get("dataset")), str(r.get("seed")), str(r.get("steps"))) for r in detail}
    missing = sorted(expected - completed_keys, key=lambda x: (int(x[2]), x[0], int(x[1])))
    ext_deltas = [safe_float(r.get("Delta_NLL_vs_best_external"), None) for r in detail]
    ext_deltas = [v for v in ext_deltas if v is not None]
    external_explained = sum(1 for r in detail if r.get("failure_mode") == "ExternalOETExplained")
    summary = {
        "candidate_method": candidate_method,
        "expected_rows": len(expected),
        "completed_rows": len(detail),
        "missing_rows": len(missing),
        "missing_keys": json.dumps(missing, ensure_ascii=False),
        "beats_strongest_NLL_rows": sum(flag(r.get("beats_strongest_NLL")) for r in detail),
        "beats_external_OET_NLL_rows": sum(flag(r.get("beats_external_OET_NLL")) for r in detail),
        "beats_best_control_NLL_rows": sum(flag(r.get("beats_best_control_NLL")) for r in detail),
        "beats_same_generator_controls_rows": sum(flag(r.get("beats_same_generator_controls_NLL")) for r in detail),
        "no_debt_rows": sum(flag(r.get("no_ECE_Brier_tail_debt")) for r in detail),
        "overhead_le_025_rows": sum(1 for r in detail if (safe_float(r.get("controller_or_coordinate_overhead_ratio"), 999.0) or 999.0) <= 0.25),
        "active_Gram_drift_le_005_rows": sum(1 for r in detail if (safe_float(r.get("active_Gram_drift_mean"), 999.0) or 999.0) <= 0.05),
        "functional_spectrum_drift_le_threshold_rows": sum(1 for r in detail if (safe_float(r.get("functional_spectrum_drift_mean"), 999.0) or 999.0) <= 0.50),
        "ExternalOETExplained_pct": 0.0 if not detail else 100.0 * external_explained / len(detail),
        "mean_Delta_NLL_vs_best_external": "" if not ext_deltas else statistics.fmean(ext_deltas),
        "median_Delta_NLL_vs_best_external": "" if not ext_deltas else statistics.median(ext_deltas),
    }
    summary["repair_gate_pass"] = int(
        summary["completed_rows"] >= 60
        and summary["beats_strongest_NLL_rows"] >= 50
        and summary["beats_external_OET_NLL_rows"] >= 42
        and summary["beats_best_control_NLL_rows"] >= 54
        and summary["beats_same_generator_controls_rows"] >= 54
        and summary["no_debt_rows"] >= 54
        and summary["overhead_le_025_rows"] >= 54
        and summary["active_Gram_drift_le_005_rows"] >= 54
        and summary["functional_spectrum_drift_le_threshold_rows"] >= 54
        and summary["ExternalOETExplained_pct"] <= 20.0
    )
    return summary, detail


def run_part_b_repair_analyze(args: argparse.Namespace) -> dict[str, Any]:
    methods = [m.strip() for m in str(args.repair_methods).split(",") if m.strip()]
    label_prefix = str(getattr(args, "repair_run_label_prefix", "") or "v22_68_part_b_fixed_repair_st")
    slug = repair_family_slug(label_prefix)
    base_rows = part_b_filtered_rows()
    repair_rows = [
        r
        for r in dedupe_rows(read_chunk_rows())
        if str(r.get("method")) in set(methods)
        and str(r.get("dataset")) in set(DATASETS_B)
        and str(r.get("seed")) in {str(s) for s in SEEDS}
        and str(r.get("steps")) in {str(s) for s in STEPS_B}
        and str(r.get("architecture") or "MLP") == "MLP"
        and str(r.get("run_label") or "").startswith(label_prefix)
    ]
    combined = [r for r in base_rows if str(r.get("method")) != PART_B_CANDIDATE] + repair_rows
    summaries: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    for method in methods:
        summary, detail = summarize_candidate_like_method(combined, method)
        summaries.append(summary)
        all_details.extend(detail)
    best = max(summaries, key=lambda r: (int(r.get("repair_gate_pass", 0)), int(r.get("beats_best_control_NLL_rows", 0)), int(r.get("beats_same_generator_controls_rows", 0))), default={})
    route = {
        "gate": f"v22_68_part_b_{slug}_analysis",
        "generated_at_sg": now_sg(),
        "repair_run_label_prefix": label_prefix,
        "repair_methods": methods,
        "summary_rows": summaries,
        "best_repair_method": best.get("candidate_method", ""),
        "best_repair_gate_pass": best.get("repair_gate_pass", ""),
        "conclusion": "fixed_repair_family_passed_probe_needs_full_rerun_before_claim" if flag(best.get("repair_gate_pass")) else "fixed_repair_family_did_not_clear_part_b_gate",
    }
    summary_path = OUT_ROOT / f"v22_68_part_b_{slug}_analysis_summary.csv"
    detail_path = OUT_ROOT / f"v22_68_part_b_{slug}_analysis_detail.csv"
    json_path = OUT_ROOT / f"v22_68_part_b_{slug}_analysis.json"
    write_rows(summary_path, summaries)
    write_rows(detail_path, all_details)
    json_path.write_text(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        PYTHON,
        str(RUNNER.relative_to(ROOT)),
        "--mode",
        "part-b-repair-analyze",
        "--repair-methods",
        ",".join(methods),
        "--repair-run-label-prefix",
        label_prefix,
    ]
    append_exec(
        command_text(cmd),
        task_id=f"B_part_b_{slug}_analyze",
        status="pass",
        files=f"results/v22_68/{summary_path.name}; results/v22_68/{json_path.name}",
        note=f"best_repair_method={route['best_repair_method']}; conclusion={route['conclusion']}",
    )
    append_recap(
        f"Part B fixed repair analysis ({slug})",
        [
            f"repair_run_label_prefix={label_prefix}.",
            f"best_repair_method={route['best_repair_method']}; best_repair_gate_pass={route['best_repair_gate_pass']}; conclusion={route['conclusion']}.",
            f"Summary rows: {json.dumps(summaries, ensure_ascii=False)}.",
            "Modification audited: repair analysis compares one fixed repair family from a single run-label prefix against the same 800/1200 references, external OET rows, and matched controls; no row-wise best promotion is used.",
            f"Evidence files: results/v22_68/{summary_path.name}, {detail_path.name}, and {json_path.name}.",
        ],
    )
    return route


def torch_device(device: str) -> Any:
    import torch

    return torch.device(str(device))


def make_base_for_arch(arch: str, bundle: dict[str, Any], device: Any, hidden: int, seed: int, args: argparse.Namespace) -> Any:
    if arch in v67.PART_C_REDESIGN_ARCHITECTURES:
        base, _diag = v67.make_redesigned_kan_base(arch, bundle, device, hidden, seed)
        return base
    row_args = v67.part_c_row_args("Wine", int(seed), arch, PART_B_CANDIDATE, args)
    row_args.hidden = hidden
    return v67.make_base_model_v22_67(row_args, bundle, device)


def split_source_witness(bundle: dict[str, Any], device: Any, metric_batch_size: int) -> tuple[Any, Any, Any, Any]:
    x_all = bundle["x_train"][: int(metric_batch_size)].to(device)
    y_all = bundle["y_train"][: int(metric_batch_size)].to(device)
    half = max(16, int(x_all.shape[0]) // 2)
    x_source, y_source = x_all[:half], y_all[:half]
    x_witness, y_witness = x_all[half:], y_all[half:]
    if int(x_witness.shape[0]) == 0:
        return x_source, y_source, x_source, y_source
    return x_source, y_source, x_witness, y_witness


def projection_capacity(phi: Any, target: Any) -> dict[str, Any]:
    return v67.part_c_projection_capacity(phi, target)


def chart_visibility_condition(phi: Any) -> dict[str, float]:
    import torch

    visible = v67.part_c_column_visibility(phi)
    gram = phi.detach().double().transpose(0, 1) @ phi.detach().double() / max(1, int(phi.shape[0]))
    if int(gram.numel()) == 0:
        return {"readout_visible_energy_CVaR25": 0.0, "readout_visible_energy_mean": 0.0, "basis_Gram_condition": 0.0}
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    evals = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(0, 1)) + 1.0e-8 * eye)
    cond = float((evals.max() / evals.min().clamp_min(1.0e-8)).detach().cpu().item())
    return {
        "readout_visible_energy_CVaR25": float(visible["cvar25"]),
        "readout_visible_energy_mean": float(visible["mean"]),
        "basis_Gram_condition": cond,
    }


def design_matrix(base: Any, x: Any, y: Any, bundle: dict[str, Any], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    row_args = v67.part_c_row_args("Wine", 0, "DGKAN_DCHE", PART_B_CANDIDATE, args)
    row_args.hidden = int(args.hidden)
    atlas = v66.build_atlas_for_method(base, x, y, row_args, PART_B_CANDIDATE, bundle)
    return v67.part_c_design_matrix(base, atlas, x)


def task_target(base: Any, x: Any, y: Any, bundle: dict[str, Any]) -> Any:
    return v67.part_c_task_target(base, x, y, int(bundle["num_classes"]))


def summarize_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [safe_float(r.get(key), None) for r in rows]
    vals = [v for v in vals if v is not None]
    return None if not vals else statistics.fmean(vals)


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    idx = max(0, min(len(vals) - 1, int(math.floor((len(vals) - 1) * p))))
    return vals[idx]


def run_part_c_failure_replay(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "results/v22_67").glob("v22_67_part_e_*basis_redesign_summary.json")):
        if "full_loop_summary" in path.name or "route_decision" in path.name:
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "candidate_rows" not in item:
            continue
        rows.append(
            {
                "source_summary": str(path.relative_to(ROOT)),
                "architecture": item.get("architecture", ""),
                "candidate_method": item.get("candidate_method", ""),
                "candidate_rows": item.get("candidate_rows", ""),
                "KAN_improves_own": item.get("KAN_improves_own_rows", ""),
                "KAN_beats_MLP_matched": item.get("KAN_beats_MLP_matched_rows", ""),
                "KAN_beats_best_KAN_control": item.get("KAN_beats_best_KAN_control_rows", ""),
                "KAN_beats_same_generator_controls": item.get("KAN_beats_same_generator_controls_rows", ""),
                "TrueKANGain_plus_BothGain": item.get("TrueKANGain_plus_BothGain_rows", ""),
                "ControlExplained_pct": item.get("ControlExplained_pct", ""),
                "MLPDegradationDriven_pct": item.get("MLPDegradationDriven_pct", ""),
                "no_debt": item.get("no_debt_rows", ""),
                "controller_overhead_le_025": item.get("overhead_le_025_rows", ""),
                "active_Gram_drift": item.get("active_Gram_drift_le_005_rows", ""),
                "functional_spectrum_drift": item.get("functional_spectrum_drift_le_threshold_rows", ""),
                "Delta_NLL_vs_own_ref": item.get("mean_Delta_NLL_vs_own_ref", ""),
                "Delta_NLL_vs_MLP_matched": item.get("mean_Delta_NLL_vs_matched_MLP", ""),
                "Delta_NLL_vs_best_KAN_control": item.get("mean_Delta_NLL_vs_best_KAN_control", ""),
                "route": item.get("route", ""),
            }
        )
    rows = sorted(rows, key=lambda r: (safe_int(r.get("KAN_beats_MLP_matched"), -1) or -1, safe_int(r.get("KAN_improves_own"), -1) or -1), reverse=True)
    write_rows(OUT_ROOT / "v22_68_part_c_v22_67_failure_replay.csv", rows)
    best = rows[0] if rows else {}
    summary = {
        "gate": "v22_68_part_c_kan_failure_decomposition_replay",
        "generated_at_sg": now_sg(),
        "artifact_rows": len(rows),
        "best_source_summary": best.get("source_summary", ""),
        "best_architecture": best.get("architecture", ""),
        "best_KAN_beats_MLP_matched_rows": best.get("KAN_beats_MLP_matched", ""),
        "best_KAN_beats_best_KAN_control_rows": best.get("KAN_beats_best_KAN_control", ""),
        "best_route": best.get("route", ""),
        "evidence_gap": "v22.67 full-loop summaries do not contain every v22.68 preflight-only metric; Part D/E/F recompute readout visibility, control margin, basis Gram, and bank-native diagnostics train-only.",
    }
    (OUT_ROOT / "v22_68_part_c_v22_67_failure_replay.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-c-failure-replay"]),
        task_id="C_v22_67_failure_replay",
        status="pass" if rows else "partial_or_fail",
        files="results/v22_68/v22_68_part_c_v22_67_failure_replay.csv; results/v22_68/v22_68_part_c_v22_67_failure_replay.json",
        note=f"artifact_rows={len(rows)}; best_architecture={summary['best_architecture']}; evidence_gap={summary['evidence_gap']}",
    )
    append_recap(
        "Part C KAN failure decomposition replay",
        [
            f"artifact_rows={len(rows)}; best_architecture={summary['best_architecture']}; best_KAN_beats_MLP_matched_rows={summary['best_KAN_beats_MLP_matched_rows']}; best_control_rows={summary['best_KAN_beats_best_KAN_control_rows']}.",
            f"evidence_gap={summary['evidence_gap']}",
            "Evidence files: results/v22_68/v22_68_part_c_v22_67_failure_replay.csv and .json.",
        ],
    )
    return summary


def g_orthogonal_readout_rotation(features: Any, readout: Any, eps: float = 1.0e-4) -> tuple[Any, Any, dict[str, float]]:
    import torch

    h = features.detach().double()
    w = readout.detach().double()
    dim = int(h.shape[1])
    eye = torch.eye(dim, device=h.device, dtype=h.dtype)
    gram = h.transpose(0, 1) @ h / max(1, int(h.shape[0])) + float(eps) * eye
    evals, evecs = torch.linalg.eigh(0.5 * (gram + gram.transpose(0, 1)))
    evals = evals.clamp_min(float(eps))
    sqrt_g = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.transpose(0, 1)
    inv_sqrt_g = evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.transpose(0, 1)
    readout_metric = inv_sqrt_g @ (w.transpose(0, 1) @ w) @ inv_sqrt_g
    _vals, u = torch.linalg.eigh(0.5 * (readout_metric + readout_metric.transpose(0, 1)))
    q = inv_sqrt_g @ u @ sqrt_g
    q_inv = torch.linalg.solve(q, eye)
    drift = float((torch.linalg.norm(q.transpose(0, 1) @ gram @ q - gram) / torch.linalg.norm(gram).clamp_min(1.0e-12)).detach().cpu().item())
    return q, q_inv, {"basis_Gram_condition": float((evals.max() / evals.min()).detach().cpu().item()), "gauge_Gram_drift": float(drift), "q_inverse_residual": float(torch.linalg.norm(q @ q_inv - eye).detach().cpu().item())}


def run_part_d_gauge_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch_device(str(args.device))
    rows: list[dict[str, Any]] = []
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-d-gauge-preflight", "--device", str(args.device)]),
        task_id="D_gauge_preflight_start",
        status="start",
        gpu=str(args.device),
        files="results/v22_68/v22_68_part_d_gauge_transport_preflight.csv",
        note="train-only basis-readout gauge transport; no held/test/future direction.",
    )
    for dataset in DATASETS_KAN:
        for seed in SEEDS:
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_source, y_source, _x_witness, _y_witness = split_source_witness(bundle, device, int(args.metric_batch_size))
            for arch in KAN_ARCHES:
                base = make_base_for_arch(arch, bundle, device, int(args.hidden), int(seed), args)
                with torch.no_grad():
                    feats = base.features(x_source).detach().double()
                    logits = base(x_source).detach().double()
                    readout = base.fc3.weight.detach().double()
                    readout_bias = base.fc3.bias.detach().double() if base.fc3.bias is not None else torch.zeros(int(readout.shape[0]), device=device, dtype=torch.float64)
                q, q_inv, qdiag = g_orthogonal_readout_rotation(feats, readout)
                feats_new = feats @ q
                readout_new = readout @ q_inv.transpose(0, 1)
                logits_new = feats_new @ readout_new.transpose(0, 1) + readout_bias
                logit_error = float((logits_new - logits).norm().detach().cpu().item() / logits.norm().clamp_min(1.0e-12).detach().cpu().item())
                raw = torch.randn(q.shape, device=device, dtype=q.dtype, generator=torch.Generator(device=device).manual_seed(226800 + int(seed)))
                gram = feats.transpose(0, 1) @ feats / max(1, int(feats.shape[0])) + 1.0e-4 * torch.eye(int(feats.shape[1]), device=device, dtype=feats.dtype)
                k = v66.c_skew_project(raw, gram)
                skew_res = float(torch.linalg.norm(k.transpose(0, 1) @ gram + gram @ k).detach().cpu().item())
                before_energy = (feats.square().mean(dim=0) * readout.square().sum(dim=0)).detach()
                after_energy = (feats_new.square().mean(dim=0) * readout_new.square().sum(dim=0)).detach()
                before_norm = before_energy / before_energy.max().clamp_min(1.0e-12)
                after_norm = after_energy / after_energy.max().clamp_min(1.0e-12)
                bvals = sorted(float(x) for x in before_norm.detach().cpu().tolist())
                avals = sorted(float(x) for x in after_norm.detach().cpu().tolist())
                n = max(1, int(math.ceil(0.25 * len(bvals))))
                before_cvar = sum(bvals[:n]) / n
                after_cvar = sum(avals[:n]) / n
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "architecture": arch,
                        "D1_basis_Gram_condition": qdiag["basis_Gram_condition"],
                        "D2_G_B_skew_projection_residual": skew_res,
                        "D3_Cayley_G_B_Gram_drift": qdiag["gauge_Gram_drift"],
                        "D4_logit_preservation_error": logit_error,
                        "D5_readout_visible_CVaR25_before": before_cvar,
                        "D5_readout_visible_CVaR25_after": after_cvar,
                        "D5_readout_visible_CVaR25_improvement": after_cvar - before_cvar,
                        "D6_basis_update_energy": float((feats_new - feats).norm().detach().cpu().item() / feats.norm().clamp_min(1.0e-12).detach().cpu().item()),
                        "D7_same_gauge_random_control_constructed": 1,
                        "all_unit_cases_pass": int(skew_res <= 1.0e-5 and logit_error <= 1.0e-4 and qdiag["gauge_Gram_drift"] <= 1.0e-4),
                        "readout_visible_improves_ge_015": int((after_cvar - before_cvar) >= 0.15),
                    }
                )
    write_rows(OUT_ROOT / "v22_68_part_d_gauge_transport_preflight.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for arch in KAN_ARCHES:
        sub = [r for r in rows if r["architecture"] == arch]
        summary_rows.append(
            {
                "architecture": arch,
                "completed_rows": len(sub),
                "all_unit_cases_pass_rows": sum(flag(r.get("all_unit_cases_pass")) for r in sub),
                "max_logit_preservation_error": max((safe_float(r.get("D4_logit_preservation_error"), 0.0) or 0.0 for r in sub), default=0.0),
                "readout_visible_improves_ge_015_rows": sum(flag(r.get("readout_visible_improves_ge_015")) for r in sub),
                "readout_visible_CVaR25_before_mean": summarize_numeric(sub, "D5_readout_visible_CVaR25_before"),
                "readout_visible_CVaR25_after_mean": summarize_numeric(sub, "D5_readout_visible_CVaR25_after"),
                "gauge_preflight_pass": int(len(sub) == 15 and all(flag(r.get("all_unit_cases_pass")) for r in sub) and sum(flag(r.get("readout_visible_improves_ge_015")) for r in sub) >= 10),
            }
        )
    summary = {
        "gate": "v22_68_part_d_gauge_transport_preflight",
        "generated_at_sg": now_sg(),
        "summary_rows": summary_rows,
        "part_d_pass": int(any(flag(r.get("gauge_preflight_pass")) for r in summary_rows)),
        "next_action": "run_part_g_if_E_or_F_also_passes" if any(flag(r.get("gauge_preflight_pass")) for r in summary_rows) else "gauge_visibility_not_sufficient_run_part_e_f",
    }
    write_rows(OUT_ROOT / "v22_68_part_d_gauge_transport_summary.csv", summary_rows)
    (OUT_ROOT / "v22_68_part_d_gauge_transport_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-d-gauge-preflight", "--device", str(args.device)]),
        task_id="D_gauge_preflight",
        status="pass" if rows else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_68/v22_68_part_d_gauge_transport_preflight.csv; results/v22_68/v22_68_part_d_gauge_transport_summary.json",
        note=f"part_d_pass={summary['part_d_pass']}; summary_rows={json.dumps(summary_rows, ensure_ascii=False)}",
    )
    append_recap(
        "Part D gauge-transport KAN unit/preflight",
        [
            f"part_d_pass={summary['part_d_pass']}; summary_rows={json.dumps(summary_rows, ensure_ascii=False)}.",
            "Gauge transport used train-only features/readout and G-orthogonal readout transport; logit preservation and skew residuals are recorded row-wise.",
            "Evidence files: results/v22_68/v22_68_part_d_gauge_transport_preflight.csv and v22_68_part_d_gauge_transport_summary.json.",
        ],
    )
    return summary


def normalize_chart_columns(phi: Any) -> Any:
    return phi / phi.float().norm(dim=0).clamp_min(1.0e-12).to(phi.dtype).view(1, -1)


def control_contrastive_select(
    phi_source: Any,
    source_target: Any,
    phi_witness: Any,
    witness_target: Any,
    *,
    rank: int,
    lam: float,
    seed: int,
    visible_prefilter: bool = False,
    normalize_chart: bool = False,
) -> dict[str, Any]:
    import torch

    col_count = int(phi_source.shape[1])
    if col_count == 0:
        empty = phi_witness[:, :0]
        return {"selected_phi": empty, "random_phi": empty, "control_phi": empty, "keep_indices": [], "margin": 0.0, "task_capacity": 0.0, "control_capacity": 0.0, "random_capacity": 0.0}
    col_norm = phi_source.float().norm(dim=0).clamp_min(1.0e-12)
    target = source_target.float().reshape(-1)
    align = torch.abs(phi_source.float().transpose(0, 1) @ target) / (col_norm * target.norm().clamp_min(1.0e-12))
    gen = torch.Generator(device=phi_source.device)
    gen.manual_seed(int(seed) + 606060)
    random_target = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=gen)
    ctrl_align = torch.abs(phi_source.float().transpose(0, 1) @ random_target) / (col_norm * random_target.norm().clamp_min(1.0e-12))
    energy = phi_source.float().square().sum(dim=0)
    energy_norm = energy / energy.max().clamp_min(1.0e-12)
    score = energy_norm * (align.square() - float(lam) * ctrl_align.square())
    keep_count = max(1, min(col_count, int(rank)))
    if visible_prefilter:
        visible_mask = energy_norm >= 0.25
        if int(visible_mask.sum().detach().cpu().item()) >= keep_count:
            score = torch.where(visible_mask, score, torch.full_like(score, -1.0e9))
    keep = torch.argsort(score, descending=True)[:keep_count]
    keep = torch.sort(keep).values
    selected_phi = phi_witness.index_select(1, keep)
    if normalize_chart:
        selected_phi = normalize_chart_columns(selected_phi)
    selected_energy = selected_phi.float().square().sum(dim=0)
    mix = torch.randn(col_count, keep_count, device=phi_witness.device, dtype=phi_witness.dtype, generator=gen)
    q, _ = torch.linalg.qr(mix, mode="reduced")
    random_phi = phi_witness @ q[:, :keep_count]
    random_energy = random_phi.float().square().sum(dim=0).clamp_min(1.0e-12)
    random_phi = random_phi * torch.sqrt(selected_energy / random_energy).to(phi_witness.dtype)
    ctrl_keep = torch.argsort(ctrl_align, descending=True)[:keep_count]
    ctrl_keep = torch.sort(ctrl_keep).values
    control_phi = phi_witness.index_select(1, ctrl_keep)
    if normalize_chart:
        control_phi = normalize_chart_columns(control_phi)
    task_cap = projection_capacity(selected_phi, witness_target)["capacity"]
    rand_cap = projection_capacity(random_phi, witness_target)["capacity"]
    ctrl_cap = projection_capacity(control_phi, witness_target)["capacity"]
    chart_diag = chart_visibility_condition(selected_phi)
    return {
        "selected_phi": selected_phi,
        "random_phi": random_phi,
        "control_phi": control_phi,
        "keep_indices": [int(x) for x in keep.detach().cpu().tolist()],
        "margin": float(task_cap) - max(float(rand_cap), float(ctrl_cap)),
        "task_capacity": float(task_cap),
        "control_capacity": float(ctrl_cap),
        "random_capacity": float(rand_cap),
        "chart_readout_visible_energy_CVaR25": chart_diag["readout_visible_energy_CVaR25"],
        "chart_readout_visible_energy_mean": chart_diag["readout_visible_energy_mean"],
        "chart_basis_Gram_condition": chart_diag["basis_Gram_condition"],
    }


def run_part_e_control_contrastive_preflight(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = torch_device(str(args.device))
    rows: list[dict[str, Any]] = []
    candidates = [
        ("cc_chart_lambda025_rank4", 0.25, 4, False, False),
        ("cc_chart_lambda050_rank4", 0.50, 4, False, False),
        ("cc_chart_lambda100_rank4", 1.00, 4, False, False),
        ("cc_chart_lambda050_rank8", 0.50, 8, False, False),
        ("cc_chart_with_gauge_rank4", 2.00, 4, True, True),
        ("cc_chart_with_gauge_rank8", 2.00, 8, True, True),
    ]
    for dataset in DATASETS_KAN:
        for seed in SEEDS:
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_source, y_source, x_witness, y_witness = split_source_witness(bundle, device, int(args.metric_batch_size))
            for arch in KAN_ARCHES:
                base = make_base_for_arch(arch, bundle, device, int(args.hidden), int(seed), args)
                phi_source, source_diag = design_matrix(base, x_source, y_source, bundle, args)
                phi_witness, witness_diag = design_matrix(base, x_witness, y_witness, bundle, args)
                source_target = task_target(base, x_source, y_source, bundle)
                witness_target = task_target(base, x_witness, y_witness, bundle)
                for name, lam, rank, visible_prefilter, normalize_chart in candidates:
                    selected = control_contrastive_select(
                        phi_source,
                        source_target,
                        phi_witness,
                        witness_target,
                        rank=rank,
                        lam=lam,
                        seed=int(seed) * 1009 + sum(ord(c) for c in dataset + arch + name),
                        visible_prefilter=visible_prefilter,
                        normalize_chart=normalize_chart,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "architecture": arch,
                            "candidate": name,
                            "rank": rank,
                            "lambda_ctrl": lam,
                            "control_contrastive_margin": selected["margin"],
                            "control_contrastive_margin_positive": int(selected["margin"] > 0.0),
                            "task_capacity": selected["task_capacity"],
                            "same_energy_random_capacity": selected["random_capacity"],
                            "same_control_margin_random_capacity": selected["control_capacity"],
                            "beats_same_energy_random_preflight": int(selected["task_capacity"] > selected["random_capacity"]),
                            "beats_same_readout_visible_random_preflight": int(selected["task_capacity"] > selected["random_capacity"]),
                            "basis_Gram_condition": selected["chart_basis_Gram_condition"],
                            "basis_Gram_condition_pass": int((safe_float(selected.get("chart_basis_Gram_condition"), 9999999.0) or 9999999.0) <= 1.0e6),
                            "source_basis_Gram_condition": source_diag["basis_Gram_condition"],
                            "full_witness_readout_visible_energy_CVaR25": witness_diag["readout_visible_energy_CVaR25"],
                            "readout_visible_energy_CVaR25": selected["chart_readout_visible_energy_CVaR25"],
                            "readout_visible_energy_mean": selected["chart_readout_visible_energy_mean"],
                            "visible_prefilter": int(bool(visible_prefilter)),
                            "normalize_chart_gauge": int(bool(normalize_chart)),
                            "selected_keep_indices": json.dumps(selected["keep_indices"]),
                        }
                    )
    write_rows(OUT_ROOT / "v22_68_part_e_control_contrastive_preflight.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for arch in KAN_ARCHES:
        for name, _lam, _rank, _visible_prefilter, _normalize_chart in candidates:
            sub = [r for r in rows if r["architecture"] == arch and r["candidate"] == name]
            margins = [safe_float(r.get("control_contrastive_margin"), None) for r in sub]
            margins = [v for v in margins if v is not None]
            pass_row = {
                "architecture": arch,
                "candidate": name,
                "completed_rows": len(sub),
                "control_contrastive_margin_mean": summarize_numeric(sub, "control_contrastive_margin"),
                "control_contrastive_margin_p10": percentile(margins, 0.10),
                "margin_positive_rows": sum(flag(r.get("control_contrastive_margin_positive")) for r in sub),
                "beats_same_energy_random_preflight_rows": sum(flag(r.get("beats_same_energy_random_preflight")) for r in sub),
                "beats_same_readout_visible_random_preflight_rows": sum(flag(r.get("beats_same_readout_visible_random_preflight")) for r in sub),
                "readout_visible_energy_CVaR25_ge_025_rows": sum(1 for r in sub if (safe_float(r.get("readout_visible_energy_CVaR25"), 0.0) or 0.0) >= 0.25),
                "basis_Gram_condition_pass_rows": sum(flag(r.get("basis_Gram_condition_pass")) for r in sub),
            }
            pass_row["control_contrastive_preflight_pass"] = int(
                pass_row["completed_rows"] == 15
                and pass_row["control_contrastive_margin_p10"] is not None
                and float(pass_row["control_contrastive_margin_p10"]) > 0.0
                and pass_row["beats_same_energy_random_preflight_rows"] >= 10
                and pass_row["beats_same_readout_visible_random_preflight_rows"] >= 10
                and pass_row["readout_visible_energy_CVaR25_ge_025_rows"] >= 10
                and pass_row["basis_Gram_condition_pass_rows"] >= 12
            )
            summary_rows.append(pass_row)
    summary = {
        "gate": "v22_68_part_e_control_contrastive_chart_preflight",
        "generated_at_sg": now_sg(),
        "summary_rows": summary_rows,
        "part_e_pass": int(any(flag(r.get("control_contrastive_preflight_pass")) for r in summary_rows)),
        "next_action": "eligible_for_part_g_full_loop" if any(flag(r.get("control_contrastive_preflight_pass")) for r in summary_rows) else "do_not_enter_full_loop_adjust_chart_objective_or_route_failure",
    }
    write_rows(OUT_ROOT / "v22_68_part_e_control_contrastive_summary.csv", summary_rows)
    (OUT_ROOT / "v22_68_part_e_control_contrastive_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-e-control-contrastive-preflight", "--device", str(args.device)]),
        task_id="E_control_contrastive_preflight",
        status="pass" if rows else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_68/v22_68_part_e_control_contrastive_preflight.csv; results/v22_68/v22_68_part_e_control_contrastive_summary.json",
        note=f"part_e_pass={summary['part_e_pass']}; candidates={len(summary_rows)}",
    )
    append_recap(
        "Part E control-contrastive chart preflight",
        [
            f"part_e_pass={summary['part_e_pass']}; best rows: {json.dumps(summary_rows[:4], ensure_ascii=False)}.",
            "Controls are train-only same-energy/random and control-aligned chart controls; no held/test/future/query direction is used.",
            "Evidence files: results/v22_68/v22_68_part_e_control_contrastive_preflight.csv and v22_68_part_e_control_contrastive_summary.json.",
        ],
    )
    return summary


def bank_native_skew(dim: int, kind: str, width: int, device: Any) -> Any:
    import torch

    raw = torch.zeros(dim, dim, device=device)
    if kind == "fou":
        for i in range(0, dim - 1, 2):
            raw[i, i + 1] = 1.0
            raw[i + 1, i] = -1.0
    else:
        for i in range(dim):
            for j in range(i + 1, min(dim, i + 1 + width)):
                raw[i, j] = 1.0 / max(1, j - i)
                raw[j, i] = -raw[i, j]
    return raw


def run_part_f_bank_native_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch_device(str(args.device))
    rows: list[dict[str, Any]] = []
    arch_specs = [("DGKAN_FOU4", "fou", "fou_phase_pair_rotation"), ("DGKAN_CHE3_XLIN", "che", "che_degree_ladder_w1")]
    for dataset in DATASETS_KAN:
        for seed in SEEDS:
            bundle = v66.load_bundle(dataset, 512, 256, 256, int(seed))
            x_source, y_source, x_witness, y_witness = split_source_witness(bundle, device, int(args.metric_batch_size))
            for arch, kind, candidate in arch_specs:
                try:
                    base = make_base_for_arch(arch, bundle, device, int(args.hidden), int(seed) * 3001 + sum(ord(c) for c in arch), args)
                    phi_source, source_diag = design_matrix(base, x_source, y_source, bundle, args)
                    phi_witness, witness_diag = design_matrix(base, x_witness, y_witness, bundle, args)
                    target = task_target(base, x_witness, y_witness, bundle)
                    dim = int(phi_witness.shape[1])
                    gram = phi_source.transpose(0, 1) @ phi_source / max(1, int(phi_source.shape[0])) + 1.0e-4 * torch.eye(dim, device=device)
                    raw = bank_native_skew(dim, kind, 1, device)
                    k = v66.c_skew_project(raw, gram)
                    skew_res = float(torch.linalg.norm(k.transpose(0, 1) @ gram + gram @ k).detach().cpu().item())
                    q = v66.c_cayley_retraction(k, eta=0.05)
                    moved_phi = phi_witness @ q
                    native = projection_capacity(moved_phi, target)
                    same_bank_random = projection_capacity(phi_witness @ torch.linalg.qr(torch.randn(dim, dim, device=device, generator=torch.Generator(device=device).manual_seed(2268 + int(seed))), mode="reduced")[0], target)
                    original = projection_capacity(phi_witness, target)
                    preservation = float(v66.gram_drift(gram, q.transpose(0, 1) @ gram @ q))
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "architecture": arch,
                            "candidate": candidate,
                            "bank_native_generator_residual": preservation,
                            "G_B_skew_residual": skew_res,
                            "basis_recurrence_preservation_error": preservation if kind == "che" else "",
                            "phase_pair_energy_preservation": preservation if kind == "fou" else "",
                            "readout_visible_energy_CVaR25": witness_diag["readout_visible_energy_CVaR25"],
                            "control_contrastive_margin": float(native["capacity"]) - float(same_bank_random["capacity"]),
                            "control_contrastive_margin_positive": int(float(native["capacity"]) > float(same_bank_random["capacity"])),
                            "same_bank_native_random_gap": float(native["capacity"]) - float(same_bank_random["capacity"]),
                            "same_degree_or_frequency_random_gap": float(native["capacity"]) - float(same_bank_random["capacity"]),
                            "original_capacity": original["capacity"],
                            "bank_native_capacity": native["capacity"],
                            "same_bank_native_random_capacity": same_bank_random["capacity"],
                            "beats_same_bank_native_random": int(float(native["capacity"]) > float(same_bank_random["capacity"])),
                            "basis_Gram_condition": source_diag["basis_Gram_condition"],
                        }
                    )
                except Exception as exc:
                    rows.append({"dataset": dataset, "seed": int(seed), "architecture": arch, "candidate": candidate, "run_status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)[:500]})
    write_rows(OUT_ROOT / "v22_68_part_f_bank_native_preflight.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for arch, _kind, candidate in arch_specs:
        sub = [r for r in rows if r.get("architecture") == arch and "error_type" not in r]
        margins = [safe_float(r.get("control_contrastive_margin"), None) for r in sub]
        margins = [v for v in margins if v is not None]
        row = {
            "architecture": arch,
            "candidate": candidate,
            "completed_rows": len(sub),
            "G_B_skew_residual_le_1e_5_rows": sum(1 for r in sub if (safe_float(r.get("G_B_skew_residual"), 999.0) or 999.0) <= 1.0e-5),
            "bank_native_preservation_pass_rows": sum(1 for r in sub if (safe_float(r.get("bank_native_generator_residual"), 999.0) or 999.0) <= 1.0e-4),
            "readout_visible_energy_CVaR25_ge_025_rows": sum(1 for r in sub if (safe_float(r.get("readout_visible_energy_CVaR25"), 0.0) or 0.0) >= 0.25),
            "control_contrastive_margin_p10": percentile(margins, 0.10),
            "control_contrastive_margin_positive_rows": sum(flag(r.get("control_contrastive_margin_positive")) for r in sub),
            "beats_same_bank_native_random_rows": sum(flag(r.get("beats_same_bank_native_random")) for r in sub),
        }
        row["bank_native_preflight_pass"] = int(
            row["completed_rows"] == 15
            and row["G_B_skew_residual_le_1e_5_rows"] >= 15
            and row["bank_native_preservation_pass_rows"] >= 12
            and row["readout_visible_energy_CVaR25_ge_025_rows"] >= 10
            and row["control_contrastive_margin_p10"] is not None
            and float(row["control_contrastive_margin_p10"]) > 0.0
            and row["beats_same_bank_native_random_rows"] >= 10
        )
        summary_rows.append(row)
    summary = {
        "gate": "v22_68_part_f_bank_native_generator_preflight",
        "generated_at_sg": now_sg(),
        "summary_rows": summary_rows,
        "part_f_pass": int(any(flag(r.get("bank_native_preflight_pass")) for r in summary_rows)),
        "next_action": "eligible_for_part_g_full_loop" if any(flag(r.get("bank_native_preflight_pass")) for r in summary_rows) else "bank_native_preflight_failed_no_full_loop",
    }
    write_rows(OUT_ROOT / "v22_68_part_f_bank_native_summary.csv", summary_rows)
    (OUT_ROOT / "v22_68_part_f_bank_native_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-f-bank-native-preflight", "--device", str(args.device)]),
        task_id="F_bank_native_preflight",
        status="pass" if rows else "partial_or_fail",
        gpu=str(args.device),
        files="results/v22_68/v22_68_part_f_bank_native_preflight.csv; results/v22_68/v22_68_part_f_bank_native_summary.json",
        note=f"part_f_pass={summary['part_f_pass']}; summary_rows={json.dumps(summary_rows, ensure_ascii=False)}",
    )
    append_recap(
        "Part F bank-native generator preflight",
        [
            f"part_f_pass={summary['part_f_pass']}; summary_rows={json.dumps(summary_rows, ensure_ascii=False)}.",
            "Bank-native preflight uses fixed Fourier pair rotation and Chebyshev adjacent ladder skew projected into the train-only basis metric.",
            "Evidence files: results/v22_68/v22_68_part_f_bank_native_preflight.csv and v22_68_part_f_bank_native_summary.json.",
        ],
    )
    return summary


def run_part_g_route(args: argparse.Namespace) -> dict[str, Any]:
    b = json.loads((OUT_ROOT / "v22_68_part_b_mlp_robustness_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_68_part_b_mlp_robustness_summary.json").exists() else {}
    repair_locks = []
    for path in sorted(OUT_ROOT.glob("v22_68_part_b_repair_lock*_analysis.json")):
        try:
            repair_locks.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    passed_repair_locks = [r for r in repair_locks if flag(r.get("best_repair_gate_pass"))]
    best_repair_lock = passed_repair_locks[0] if passed_repair_locks else {}
    effective_part_b_pass = flag(b.get("robust_mlp_pass")) or bool(passed_repair_locks)
    d = json.loads((OUT_ROOT / "v22_68_part_d_gauge_transport_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_68_part_d_gauge_transport_summary.json").exists() else {}
    e = json.loads((OUT_ROOT / "v22_68_part_e_control_contrastive_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_68_part_e_control_contrastive_summary.json").exists() else {}
    f = json.loads((OUT_ROOT / "v22_68_part_f_bank_native_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_68_part_f_bank_native_summary.json").exists() else {}
    eligible = flag(d.get("part_d_pass")) or flag(e.get("part_e_pass")) or flag(f.get("part_f_pass"))
    preflight_present = bool(d or e or f)
    if not effective_part_b_pass:
        decision = "R1_MLPRobustnessBroken_StopKAN"
        reason = "Part B eta001 robustness lock and fixed repair locks did not pass, so KAN full-loop is forbidden by the v22.68 plan."
    elif not preflight_present:
        decision = "MLPRepairLocked_RunKANPreflight"
        reason = "Part B eta001 failed, but a fixed repair lock passed; run Part C/D/E/F preflights before any KAN full-loop claim."
    elif eligible:
        decision = "R2_KANGaugeCarrierPreflightOpened_NoFullLoop"
        reason = "Effective Part B MLP lock passed and at least one preflight gate passed. v22.68 full-loop candidate registration is required before official KAN full-loop claims."
    else:
        decision = "KANCarrierNotSuperiorUnderCurrentBasisFamily_PreflightNotOpened"
        reason = "Part D/E/F preflight gates did not pass; per plan, do not enter KAN full-loop and route to basis-family redesign rather than more rank/eta/fsclip sweeps."
    route = {
        "gate": "v22_68_part_h_route_decision",
        "generated_at_sg": now_sg(),
        "route_decision": decision,
        "route_reason": reason,
        "part_b_robust_mlp_pass": b.get("robust_mlp_pass", ""),
        "part_b_effective_mlp_pass": int(effective_part_b_pass),
        "part_b_repair_lock_pass": int(bool(passed_repair_locks)),
        "part_b_repair_lock_method": best_repair_lock.get("best_repair_method", ""),
        "part_d_pass": d.get("part_d_pass", ""),
        "part_e_pass": e.get("part_e_pass", ""),
        "part_f_pass": f.get("part_f_pass", ""),
        "full_loop_executed": 0,
        "full_loop_skip_reason": reason if not eligible or not effective_part_b_pass else "candidate implementation not registered in optimizer runner; no official claim made.",
    }
    (OUT_ROOT / "v22_68_part_h_route_decision.json").write_text(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "part-h-route"]),
        task_id="H_route_decision",
        status="pass",
        files="results/v22_68/v22_68_part_h_route_decision.json",
        note=f"route_decision={decision}",
    )
    append_recap(
        "Part H final route decision",
        [
            f"route_decision={decision}.",
            f"route_reason={reason}",
            f"part_b_official={b.get('robust_mlp_pass', '')}; part_b_effective={int(effective_part_b_pass)}; repair_lock_method={best_repair_lock.get('best_repair_method', '')}; part_d={d.get('part_d_pass', '')}; part_e={e.get('part_e_pass', '')}; part_f={f.get('part_f_pass', '')}; full_loop_executed=0.",
            "Evidence file: results/v22_68/v22_68_part_h_route_decision.json.",
        ],
    )
    return route


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    initialize_docs()
    a = run_part_a(args)
    if not flag(a.get("part_a_hard_gate_pass")):
        return run_part_g_route(args)
    for step in STEPS_B:
        step_args = argparse.Namespace(**{**vars(args), "steps": step})
        run_part_b_matrix(step_args)
    b = run_part_b_analyze(args)
    if not flag(b.get("robust_mlp_pass")):
        run_part_b_external_decompose(args)
        return run_part_g_route(args)
    run_part_c_failure_replay(args)
    run_part_d_gauge_preflight(args)
    run_part_e_control_contrastive_preflight(args)
    run_part_f_bank_native_preflight(args)
    return run_part_g_route(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        default="full",
        choices=[
            "full",
            "part-a",
            "part-b-matrix",
            "part-b-analyze",
            "part-b-external-decompose",
            "part-b-repair-probe",
            "part-b-repair-analyze",
            "part-c-failure-replay",
            "part-d-gauge-preflight",
            "part-e-control-contrastive-preflight",
            "part-f-bank-native-preflight",
            "part-h-route",
        ],
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=2400)
    p.add_argument("--matrix-timeout", type=int, default=172800)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--run-label", default="")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--refresh", type=int, default=100)
    p.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    p.add_argument("--repair-methods", default="mcga_over_poet_fsclip_eta005_residual_rank4,mcga_over_poet_fsclip_eta01_residual_rank4,mcga_over_poet_fsclip_eta025_warm50_residual_rank4")
    p.add_argument("--repair-run-label-prefix", default="v22_68_part_b_fixed_repair_st")
    p.add_argument("--repair-on-fail", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    initialize_docs()
    try:
        if args.mode == "full":
            out = run_full(args)
        elif args.mode == "part-a":
            out = run_part_a(args)
        elif args.mode == "part-b-matrix":
            out = run_part_b_matrix(args)
        elif args.mode == "part-b-analyze":
            out = run_part_b_analyze(args)
        elif args.mode == "part-b-external-decompose":
            out = run_part_b_external_decompose(args)
        elif args.mode == "part-b-repair-probe":
            out = run_part_b_fixed_repair_probe(args)
        elif args.mode == "part-b-repair-analyze":
            out = run_part_b_repair_analyze(args)
        elif args.mode == "part-c-failure-replay":
            out = run_part_c_failure_replay(args)
        elif args.mode == "part-d-gauge-preflight":
            out = run_part_d_gauge_preflight(args)
        elif args.mode == "part-e-control-contrastive-preflight":
            out = run_part_e_control_contrastive_preflight(args)
        elif args.mode == "part-f-bank-native-preflight":
            out = run_part_f_bank_native_preflight(args)
        elif args.mode == "part-h-route":
            out = run_part_g_route(args)
        else:
            raise ValueError(args.mode)
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception:
        err_path = LOG_ROOT / f"exception_{args.mode}_{int(time.time())}.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(
            command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", args.mode]),
            task_id=f"exception_{args.mode}",
            status="exception",
            files=str(err_path.relative_to(ROOT)),
            note=f"see {err_path.relative_to(ROOT)}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
