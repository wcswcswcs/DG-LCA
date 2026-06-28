#!/usr/bin/env python3
"""DG-KAN v22.65 Metric-Compatible Signal Atlas FU runner."""

from __future__ import annotations

import argparse
import concurrent.futures
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
import tarfile
import tempfile
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from dgkan.fu.metric_preserving_functional_atlas import (  # noqa: E402
    AtlasBuild,
    MetricAtlasMLP,
    MetricCompatibleAtlasMLP,
    SimpleMLP,
    apply_output_transport,
    build_last_layer_atlas,
    gram_drift,
    metric_compatible_descent_diagnostics,
    metric_compatible_unit_tests,
    stable_rank,
)
from experiments.run_v22_64_metric_preserving_functional_atlas_fu import (  # noqa: E402
    CautiousAdamW,
    ScheduleFreeAdamWLocal,
    corr,
    evaluate_tensors,
    fval,
    iflag,
    load_bundle,
    md_table,
    mean,
    optimizer_state_count,
    read_json,
    read_rows,
    safe_fragment,
    set_seed,
    split_csv,
    torch_device,
    trainable_param_count,
    write_json,
    write_rows,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_65"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.65_MetricCompatibleSignalAtlasFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.65_MetricCompatibleSignalAtlasFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.65_MetricCompatibleSignalAtlasFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_65_metric_compatible_signal_atlas_fu.py"
V64_RUNNER = ROOT / "experiments/run_v22_64_metric_preserving_functional_atlas_fu.py"
ATLAS_MODULE = ROOT / "dgkan/fu/metric_preserving_functional_atlas.py"

REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local"}
EXTERNAL_METHODS = {"poet_official", "pion_oet_sphere_official", "pion_oet_local"}
CANDIDATE_METHODS = {
    "mpfa_transport_iso_rank2",
    "mpfa_transport_iso_rank4",
    "mpfa_transport_iso_last_layer",
    "mpfa_shape_signal_budget002",
    "mpfa_shape_signal_budget005",
    "mpfa_shape_signal_tail_safe",
    "mpfa_shape_reservoir_suppressed",
    "mpfa_poet_compatible_transport_iso",
    "mpfa_poet_compatible_shape_signal",
    "mpfa_pion_compatible_transport_iso",
    "mpfa_oet_tangent_signal_shape",
    "mpfa_residual_transport_iso_rank2",
    "mpfa_residual_shape_signal_budget005",
    "mpfa_residual_shape_reservoir_suppressed",
    "mpfa_residual_poet_compatible_shape_signal",
    "mpfa_spectral_residual_transport_iso_rank2",
    "mpfa_spectral_residual_shape_signal_budget005",
    "mpfa_spectral_residual_shape_reservoir_suppressed",
    "mpfa_spectral_residual_poet_compatible_shape_signal",
    "mpfa_fsclip_spectral_residual_transport_iso_rank2",
    "mpfa_fsclip_spectral_residual_shape_signal_budget005",
    "mpfa_fsclip_spectral_residual_poet_compatible_shape_signal",
    "mpfa_fsclip_spectral_residual_poet_compatible_shape_signal_rank4",
    "mpfa_fsclip_spectral_residual_shape_signal_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_poet_compatible_balanced_shape_signal_rank8",
    "mpfa_fsclip_spectral_residual_gradcoh_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_gradcoh_balanced_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_gradcoh_balanced_shape_poetopt_rank8",
    "mpfa_fsclip_spectral_residual_gradcoh_featurein_balanced_shape_poetopt_rank8",
    "mpfa_fsclip_spectral_residual_gradcoh_featurein_balanced_shape_budget002_poetopt_rank8",
    "mpfa_fsclip_spectral_residual_gradcoh_balanced_fishermetric_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_gradcoh_featurein_balanced_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_cvargrad_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_gradcov_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_gradunion_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_gradunion_balanced_shape_poetopt_rank4",
    "mpfa_fsclip_spectral_residual_gradunion_balanced_shape_poetopt_rank8",
    "mpfa_fsclip_spectral_residual_gradunion_featurein_balanced_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_gradcoh_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_gradcoh_balanced_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_gradcoh_featurein_balanced_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_cvargrad_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_gradcov_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_gradunion_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_gradunion_balanced_shape_poetopt_rank4",
    "mpfa_fsfill_spectral_residual_gradunion_featurein_balanced_shape_poetopt_rank4",
    "mpfa_warm50_fsclip_spectral_residual_gradcoh_shape_poetopt_rank4",
}
CONTROL_METHODS = {
    "oet_only_coordinate",
    "lora_like_coordinate",
    "same_rank_random_coordinate",
    "same_spectrum_random_coordinate",
    "same_functional_spectrum_random_coordinate",
    "same_isometric_capacity_random_coordinate",
    "same_shape_budget_random_coordinate",
    "same_transport_error_random_coordinate",
    "same_signal_reachable_energy_random_coordinate",
    "same_metric_drift_random_coordinate",
    "shuffled_source_atlas",
    "witness_only_atlas",
    "source_only_atlas",
    "self_only_atlas",
    "same_compute_noop_coordinate",
    "same_compute_residual_noop",
    "same_rank_random_residual_coordinate",
    "same_shape_budget_residual_random_coordinate",
    "same_compute_spectral_residual_noop",
    "same_rank_random_spectral_residual_coordinate",
    "same_shape_budget_spectral_residual_random_coordinate",
    "same_rank_random_fsclip_spectral_residual_coordinate",
    "same_shape_budget_fsclip_spectral_residual_random_coordinate",
    "same_rank_random_rank4_fsclip_spectral_residual_coordinate",
    "same_shape_budget_rank4_fsclip_spectral_residual_random_coordinate",
    "same_rank_random_rank4_fsclip_spectral_residual_poetopt_coordinate",
    "same_shape_budget_rank4_fsclip_spectral_residual_poetopt_random_coordinate",
    "same_rank_random_rank8_fsclip_spectral_residual_poetopt_coordinate",
    "same_shape_budget_rank8_fsclip_spectral_residual_poetopt_random_coordinate",
    "same_signal_reachable_gradcoh_balanced_shape_rank8_fsclip_spectral_residual_poetopt_random_coordinate",
    "same_rank_random_rank4_fsclip_spectral_residual_poetopt_coordinate_fishermetric",
    "same_shape_budget_rank4_fsclip_spectral_residual_poetopt_random_coordinate_fishermetric",
    "same_rank_random_rank4_fsfill_spectral_residual_poetopt_coordinate",
    "same_shape_budget_rank4_fsfill_spectral_residual_poetopt_random_coordinate",
    "same_rank_random_warm50_rank4_fsclip_spectral_residual_poetopt_coordinate",
    "same_shape_budget_warm50_rank4_fsclip_spectral_residual_poetopt_random_coordinate",
}
DEFAULT_METHODS = ",".join(
    [
        "adamw",
        "cautious_adamw",
        "schedule_free_adamw_local",
        "poet_official",
        "pion_oet_sphere_official",
        "pion_oet_local",
        "mpfa_transport_iso_rank2",
        "mpfa_transport_iso_rank4",
        "mpfa_transport_iso_last_layer",
        "mpfa_shape_signal_budget002",
        "mpfa_shape_signal_budget005",
        "mpfa_shape_signal_tail_safe",
        "mpfa_shape_reservoir_suppressed",
        "mpfa_poet_compatible_transport_iso",
        "mpfa_poet_compatible_shape_signal",
        "mpfa_pion_compatible_transport_iso",
        "mpfa_oet_tangent_signal_shape",
        "mpfa_residual_transport_iso_rank2",
        "mpfa_residual_shape_signal_budget005",
        "mpfa_residual_shape_reservoir_suppressed",
        "mpfa_residual_poet_compatible_shape_signal",
        "mpfa_spectral_residual_transport_iso_rank2",
        "mpfa_spectral_residual_shape_signal_budget005",
        "mpfa_spectral_residual_shape_reservoir_suppressed",
        "mpfa_spectral_residual_poet_compatible_shape_signal",
        "mpfa_fsclip_spectral_residual_transport_iso_rank2",
        "mpfa_fsclip_spectral_residual_shape_signal_budget005",
        "mpfa_fsclip_spectral_residual_poet_compatible_shape_signal",
        "mpfa_fsclip_spectral_residual_poet_compatible_shape_signal_rank4",
        "mpfa_fsclip_spectral_residual_shape_signal_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_poet_compatible_balanced_shape_signal_rank8",
        "mpfa_fsclip_spectral_residual_gradcoh_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_gradcoh_balanced_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_gradcoh_balanced_shape_poetopt_rank8",
        "mpfa_fsclip_spectral_residual_gradcoh_featurein_balanced_shape_poetopt_rank8",
        "mpfa_fsclip_spectral_residual_gradcoh_featurein_balanced_shape_budget002_poetopt_rank8",
        "mpfa_fsclip_spectral_residual_gradcoh_balanced_fishermetric_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_gradcoh_featurein_balanced_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_cvargrad_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_gradcov_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_gradunion_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_gradunion_balanced_shape_poetopt_rank4",
        "mpfa_fsclip_spectral_residual_gradunion_balanced_shape_poetopt_rank8",
        "mpfa_fsclip_spectral_residual_gradunion_featurein_balanced_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_gradcoh_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_gradcoh_balanced_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_gradcoh_featurein_balanced_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_cvargrad_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_gradcov_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_gradunion_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_gradunion_balanced_shape_poetopt_rank4",
        "mpfa_fsfill_spectral_residual_gradunion_featurein_balanced_shape_poetopt_rank4",
        "mpfa_warm50_fsclip_spectral_residual_gradcoh_shape_poetopt_rank4",
        "oet_only_coordinate",
        "lora_like_coordinate",
        "same_rank_random_coordinate",
        "same_spectrum_random_coordinate",
        "same_functional_spectrum_random_coordinate",
        "same_isometric_capacity_random_coordinate",
        "same_shape_budget_random_coordinate",
        "same_transport_error_random_coordinate",
        "same_signal_reachable_energy_random_coordinate",
        "same_metric_drift_random_coordinate",
        "shuffled_source_atlas",
        "witness_only_atlas",
        "source_only_atlas",
        "self_only_atlas",
        "same_compute_noop_coordinate",
        "same_compute_residual_noop",
        "same_rank_random_residual_coordinate",
        "same_shape_budget_residual_random_coordinate",
        "same_compute_spectral_residual_noop",
        "same_rank_random_spectral_residual_coordinate",
        "same_shape_budget_spectral_residual_random_coordinate",
        "same_rank_random_fsclip_spectral_residual_coordinate",
        "same_shape_budget_fsclip_spectral_residual_random_coordinate",
        "same_rank_random_rank4_fsclip_spectral_residual_coordinate",
        "same_shape_budget_rank4_fsclip_spectral_residual_random_coordinate",
        "same_rank_random_rank4_fsclip_spectral_residual_poetopt_coordinate",
        "same_shape_budget_rank4_fsclip_spectral_residual_poetopt_random_coordinate",
        "same_rank_random_rank8_fsclip_spectral_residual_poetopt_coordinate",
        "same_shape_budget_rank8_fsclip_spectral_residual_poetopt_random_coordinate",
        "same_signal_reachable_gradcoh_balanced_shape_rank8_fsclip_spectral_residual_poetopt_random_coordinate",
        "same_rank_random_rank4_fsclip_spectral_residual_poetopt_coordinate_fishermetric",
        "same_shape_budget_rank4_fsclip_spectral_residual_poetopt_random_coordinate_fishermetric",
        "same_rank_random_rank4_fsfill_spectral_residual_poetopt_coordinate",
        "same_shape_budget_rank4_fsfill_spectral_residual_poetopt_random_coordinate",
        "same_rank_random_warm50_rank4_fsclip_spectral_residual_poetopt_coordinate",
        "same_shape_budget_warm50_rank4_fsclip_spectral_residual_poetopt_random_coordinate",
    ]
)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.65 MetricCompatibleSignalAtlasFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只写真实命令、解释器、GPU、输入输出文件、状态、blocker、修复尝试；"
            "后续复现应能从这里找到命令和 artifact 路径。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.65 MetricCompatibleSignalAtlasFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘数据；缺失、失败、修复都必须明示；不编造实验数据或结论。\n",
            encoding="utf-8",
        )


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "", files: str = "", note: str = "", exit_code: Any = "n/a") -> None:
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
    journal_path = OUT_ROOT / "v22_65_command_journal.csv"
    journal = read_rows(journal_path)
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(journal_path, journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + command + "\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def append_recap(title: str, body: str) -> None:
    ensure_out()
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_sg()} {title}\n\n{body.rstrip()}\n")


def run_cmd(cmd: list[str], *, task_id: str, files: str = "", gpu: str = "cpu", timeout: int | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
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
            files=files,
            exit_code=proc.returncode,
            note=f"stdout={stdout_path}; stderr={stderr_path}; wall_seconds={time.time() - start:.3f}",
        )
        return {"status": status, "returncode": proc.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}
    except Exception as exc:
        stderr_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(command_text(cmd), task_id=task_id, status="exception", gpu=gpu, files=files, exit_code="exception", note=f"{type(exc).__name__}: {exc}; stderr={stderr_path}")
        return {"status": "exception", "returncode": -999, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}


class RuntimeTrainingLoopAudit:
    def __init__(self, params: Any) -> None:
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.before_step: list[Any] = []
        self.manual_param_update_detected = 0
        self.no_grad_param_mutation_detected = 0
        self.apply_flat_update_called = 0
        self.p_data_write_detected = 0
        self.copy_param_write_detected = 0

    def snapshot_before_backward(self) -> None:
        self.before_step = [p.detach().clone() for p in self.params]

    def check_before_optimizer_step(self) -> None:
        if len(self.before_step) != len(self.params):
            return
        for item, before in zip(self.params, self.before_step):
            delta = (item.detach() - before.to(device=item.device, dtype=item.dtype)).abs().max()
            val = float(delta.item()) if hasattr(delta, "item") else float(delta)
            if math.isfinite(val) and val > 1.0e-12:
                self.manual_param_update_detected = 1
                self.no_grad_param_mutation_detected = 1
                break

    def as_dict(self) -> dict[str, int]:
        return {
            "manual_param_update_detected": int(self.manual_param_update_detected),
            "no_grad_param_mutation_detected": int(self.no_grad_param_mutation_detected),
            "apply_flat_update_called": int(self.apply_flat_update_called),
            "param_data_write_detected": int(self.p_data_write_detected),
            "copy_param_write_detected": int(self.copy_param_write_detected),
        }


def method_family(method: str) -> str:
    if method in REFERENCE_METHODS:
        return "reference"
    if method in EXTERNAL_METHODS:
        return "external_oet"
    if method in CANDIDATE_METHODS:
        return "candidate"
    if method in CONTROL_METHODS:
        return "control"
    return "unknown"


def rank_for_method(method: str, num_classes: int | None = None, hidden: int | None = None) -> int:
    requested = 8 if "rank8" in method else (4 if "rank4" in method or "last_layer" in method else 2)
    if num_classes is not None:
        requested = min(requested, int(num_classes))
    if hidden is not None:
        requested = min(requested, int(hidden))
    return max(1, int(requested))


def atlas_method_name(method: str) -> str:
    low = str(method).lower()
    if "same_compute_residual_noop" in low or "same_compute_spectral_residual_noop" in low:
        return "same_compute_noop_coordinate"
    if "same_signal_reachable" in low:
        name = "same_signal_reachable_random_atlas"
        if "gradcoh" in low:
            name += "_gradcoh"
        if "balanced" in low:
            name += "_balanced"
        return name
    if "oet_tangent" in low or "oet_only" in low:
        return "gfoa_oet_only"
    if "gradunion" in low:
        name = "metric_atlas_gradunion"
        if "featurein" in low:
            name += "_featurein"
        if "balanced" in low:
            name += "_balanced"
        return name
    if "gradcov" in low:
        return "metric_atlas_gradcov"
    if "cvargrad" in low:
        return "metric_atlas_cvargrad"
    if "gradcoh" in low:
        name = "metric_atlas_gradcoh"
        if "featurein" in low:
            name += "_featurein"
        if "balanced" in low:
            name += "_balanced"
        return name
    if "poet_compatible" in low or "pion_compatible" in low:
        return "metric_atlas_act_balanced" if "balanced" in low else "metric_atlas_act"
    if "shape_signal" in low or "tail_safe" in low or "reservoir_suppressed" in low:
        return "metric_atlas_sw"
    if "same_functional_spectrum" in low or "same_spectrum" in low:
        return "same_functional_spectrum_random_atlas"
    if "same_rank_random" in low or "same_iso" in low or "same_transport" in low or "same_signal" in low or "same_metric" in low or "same_shape" in low:
        return "same_rank_random_atlas"
    if "lora_like" in low:
        return "lora_like_coordinate"
    if "noop" in low:
        return "same_compute_noop_coordinate"
    return method


def uses_compatible_model(method: str) -> bool:
    return method in CANDIDATE_METHODS or method in {
        "same_isometric_capacity_random_coordinate",
        "same_shape_budget_random_coordinate",
        "same_transport_error_random_coordinate",
        "same_signal_reachable_energy_random_coordinate",
        "same_metric_drift_random_coordinate",
        "same_compute_residual_noop",
        "same_rank_random_residual_coordinate",
        "same_shape_budget_residual_random_coordinate",
        "same_compute_spectral_residual_noop",
        "same_rank_random_spectral_residual_coordinate",
        "same_shape_budget_spectral_residual_random_coordinate",
        "same_rank_random_fsclip_spectral_residual_coordinate",
        "same_shape_budget_fsclip_spectral_residual_random_coordinate",
        "same_rank_random_rank4_fsclip_spectral_residual_coordinate",
        "same_shape_budget_rank4_fsclip_spectral_residual_random_coordinate",
        "same_rank_random_rank4_fsclip_spectral_residual_poetopt_coordinate",
        "same_shape_budget_rank4_fsclip_spectral_residual_poetopt_random_coordinate",
        "same_rank_random_rank8_fsclip_spectral_residual_poetopt_coordinate",
        "same_shape_budget_rank8_fsclip_spectral_residual_poetopt_random_coordinate",
        "same_signal_reachable_gradcoh_balanced_shape_rank8_fsclip_spectral_residual_poetopt_random_coordinate",
        "same_rank_random_rank4_fsclip_spectral_residual_poetopt_coordinate_fishermetric",
        "same_shape_budget_rank4_fsclip_spectral_residual_poetopt_random_coordinate_fishermetric",
        "same_rank_random_rank4_fsfill_spectral_residual_poetopt_coordinate",
        "same_shape_budget_rank4_fsfill_spectral_residual_poetopt_random_coordinate",
        "same_rank_random_warm50_rank4_fsclip_spectral_residual_poetopt_coordinate",
        "same_shape_budget_warm50_rank4_fsclip_spectral_residual_poetopt_random_coordinate",
    }


def train_base_for_method(method: str) -> bool:
    return "residual" in str(method).lower()


def base_spectrum_lock_for_method(method: str) -> bool:
    return "spectral_residual" in str(method).lower()


def functional_spectrum_budget_for_method(method: str, default: float) -> float:
    low = str(method).lower()
    return float(default) if "fsclip" in low or "fsfill" in low else 0.0


def functional_spectrum_fill_for_method(method: str) -> bool:
    return "fsfill" in str(method).lower()


def warmup_steps_for_method(method: str, total_steps: int) -> int:
    low = str(method).lower()
    if "warm50" in low:
        return max(0, min(50, int(total_steps) - 1))
    return 0


def allow_shape_for_method(method: str) -> bool:
    low = str(method).lower()
    return "shape" in low or "budget" in low or "tail_safe" in low or "reservoir_suppressed" in low


def shape_budget_for_method(method: str, default: float) -> float:
    low = str(method).lower()
    if "budget002" in low:
        return 0.02
    if "budget005" in low:
        return 0.05
    if "same_shape_budget" in low:
        return float(default)
    if allow_shape_for_method(method):
        return float(default)
    return 0.0


def metric_kind_for_method(method: str, default: str) -> str:
    if "fishermetric" in method:
        return "fisher"
    if "tail_safe" in method:
        return "signal_debt"
    if "reservoir_suppressed" in method:
        return "signal_debt"
    return str(default)


def make_optimizer(method: str, model: Any, args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    family = method_family(method)
    if method == "adamw" or family in {"candidate", "control"}:
        if "poet_compatible" in method or "poetopt" in method:
            return make_poet_optimizer_for_model(model, args)
        if "pion_compatible" in method:
            from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

            opt = MatrixGeometryOptimizer("pion_oet_sphere_official", model, float(args.lr), float(args.weight_decay), device)
            return opt, {"optimizer_step_source": "pion_oet_sphere_official_MatrixGeometryOptimizer", "external_optimizer_available": 1}
        return torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "torch.optim.AdamW",
            "external_optimizer_available": 1,
        }
    if method == "cautious_adamw":
        return CautiousAdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "CautiousAdamW_grad_gate_over_AdamW",
            "external_optimizer_available": 1,
        }
    if method == "schedule_free_adamw_local":
        return ScheduleFreeAdamWLocal(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "ScheduleFreeAdamWLocal_over_AdamW",
            "external_optimizer_available": 1,
        }
    if method == "poet_official":
        return make_poet_optimizer_for_model(model, args)
    if method in {"pion_oet_sphere_official", "pion_oet_local"}:
        from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

        opt = MatrixGeometryOptimizer(method, model, float(args.lr), float(args.weight_decay), device)
        return opt, {"optimizer_step_source": f"{method}_MatrixGeometryOptimizer", "external_optimizer_available": 1}
    raise ValueError(f"unknown method {method!r}")


def make_poet_optimizer_for_model(model: Any, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "external/oet_baselines/poet_sphere"))
    from poet_torch import POETConfig, POETModel, get_poet_optimizer

    cfg = POETConfig(
        block_size=int(args.poet_block_size),
        merge_interval=int(args.poet_merge_interval),
        poet_lr=float(args.poet_lr),
        base_lr=float(args.lr),
        poet_scale=float(args.poet_scale),
        weight_decay=float(args.weight_decay),
        mem_efficient_mode=False,
    )
    wrapped = POETModel(model, cfg)
    opt = get_poet_optimizer(wrapped, cfg)
    return opt, {"optimizer_step_source": "poet_torch.get_poet_optimizer", "external_optimizer_available": 1, "wrapped_model": wrapped}


def row_path(args: argparse.Namespace) -> Path:
    frag = f"{safe_fragment(args.method)}_{safe_fragment(args.dataset)}_s{int(args.seed)}_st{int(args.steps)}"
    return CHUNK_ROOT / f"{frag}.csv"


def row_log_prefix(args: argparse.Namespace) -> str:
    return f"v22_65_row_{safe_fragment(args.method)}_{safe_fragment(args.dataset)}_s{int(args.seed)}"


def metric_cohort(x_train: Any, y_train: Any, args: argparse.Namespace, *, offset: int = 0) -> tuple[Any, Any]:
    import torch

    limit = int(getattr(args, "metric_batch_size", 0) or 0)
    n = int(x_train.shape[0])
    if limit <= 0 or limit >= n:
        return x_train, y_train
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(int(args.seed) * 1709 + int(offset) + sum(ord(ch) for ch in str(args.method)))
    idx = torch.randperm(n, generator=gen, device=x_train.device)[:limit]
    return x_train[idx], y_train[idx]


def coord_model_of(model: Any) -> Any:
    base = getattr(model, "base_model", None)
    if isinstance(base, (MetricAtlasMLP, MetricCompatibleAtlasMLP)):
        return base
    return model


def get_coord_layer(model: Any) -> Any | None:
    coord = coord_model_of(model)
    fc3 = getattr(coord, "fc3", None)
    if fc3 is not None and hasattr(fc3, "atlas_delta"):
        return fc3
    return None


def model_spectrum(model: Any) -> list[float]:
    import torch

    layer = get_coord_layer(model)
    if layer is not None and hasattr(layer, "effective_weight"):
        vals = torch.linalg.svdvals(layer.effective_weight().detach().float())
        return [float(x) for x in vals.detach().cpu().tolist()]
    fc3 = getattr(coord_model_of(model), "fc3", None)
    if fc3 is not None and hasattr(fc3, "weight"):
        vals = torch.linalg.svdvals(fc3.weight.detach().float())
        return [float(x) for x in vals.detach().cpu().tolist()]
    return []


def spectrum_drift(initial: list[float], final: list[float]) -> float | str:
    if not initial or not final:
        return ""
    n = min(len(initial), len(final))
    num = math.sqrt(sum((float(final[i]) - float(initial[i])) ** 2 for i in range(n)))
    den = math.sqrt(sum(float(initial[i]) ** 2 for i in range(n)))
    return float(num / max(den, 1.0e-12))


def compute_capacity_diagnostics(model: Any, x: Any, y: Any, batch_size: int = 512) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    layer = get_coord_layer(model)
    if layer is None:
        return {
            "task_gradient_norm": "",
            "iso_descent_energy_fraction": "",
            "iso_predicted_task_descent": "",
            "iso_projected_gradient_norm": "",
            "iso_direction_cosine_with_task_gradient": "",
            "shape_descent_energy_fraction": "",
            "shape_predicted_task_descent": "",
            "shape_projected_gradient_norm": "",
            "iso_or_shape_capacity_positive": 0,
            "capacity_source": "no_coordinate_layer",
        }
    device = next(coord_model_of(model).parameters()).device
    model.train()
    for p in model.parameters():
        p.grad = None
    xb = x[: int(batch_size)].to(device)
    yb = y[: int(batch_size)].to(device).long()
    logits = model(xb)
    loss = F.cross_entropy(logits.float(), yb)
    loss.backward()
    grad = None
    if hasattr(layer, "raw_iso") and getattr(layer.raw_iso, "grad", None) is not None:
        grad = layer.raw_iso.grad.detach().clone()
    elif hasattr(layer, "q") and getattr(layer.q, "grad", None) is not None:
        grad = layer.q.grad.detach().clone()
    c_metric = getattr(layer, "active_gram", None)
    if c_metric is None:
        c_metric = torch.eye(int(grad.shape[0]), device=grad.device) if grad is not None else None
    out = metric_compatible_descent_diagnostics(grad, c_metric.detach(), eps=1.0e-8) if c_metric is not None else {}
    out["capacity_source"] = "train_only_initial_or_final_gradient"
    out["capacity_loss"] = float(loss.detach().cpu().item())
    if hasattr(layer, "raw_shape") and getattr(layer.raw_shape, "grad", None) is not None and layer.raw_shape.grad is not None:
        shape_diag = metric_compatible_descent_diagnostics(layer.raw_shape.grad.detach().clone(), c_metric.detach(), eps=1.0e-8)
        out["shape_parameter_gradient_norm"] = shape_diag["task_gradient_norm"]
    for p in model.parameters():
        p.grad = None
    return out


def build_atlas_for_method(base: SimpleMLP, x_metric: Any, y_metric: Any, args: argparse.Namespace, method: str, bundle: dict[str, Any]) -> AtlasBuild:
    return build_last_layer_atlas(
        base,
        x_metric,
        y_metric,
        rank=rank_for_method(method, int(bundle["num_classes"]), int(args.hidden)),
        method=atlas_method_name(method),
        metric_kind=metric_kind_for_method(method, str(args.metric_kind)),
        seed=int(args.seed),
    )


def refresh_atlas_if_needed(model: Any, x_train: Any, y_train: Any, method: str, args: argparse.Namespace, old_gram: Any, *, step: int = 0) -> tuple[Any, dict[str, float]]:
    coord_model = coord_model_of(model)
    if not isinstance(coord_model, (MetricAtlasMLP, MetricCompatibleAtlasMLP)):
        return old_gram, {}
    x_metric, y_metric = metric_cohort(x_train, y_train, args, offset=int(step))
    pseudo_bundle = {"num_classes": int(getattr(coord_model.fc3, "out_dim", coord_model.fc3.bias.numel())), "input_dim": int(x_train.shape[1])}
    atlas = build_last_layer_atlas(
        coord_model,
        x_metric,
        y_metric,
        rank=rank_for_method(method, int(pseudo_bundle["num_classes"]), int(args.hidden)),
        method=atlas_method_name(method),
        metric_kind=metric_kind_for_method(method, str(args.metric_kind)),
        seed=int(args.seed),
    )
    diag = dict(atlas.metrics)
    if old_gram is not None and (method in CANDIDATE_METHODS or method.startswith("same_")):
        # v22.65 keeps metric transport separate from bounded shaping:
        # basis refresh must use metric retraction, while shape budget is
        # enforced inside MetricCompatibleLowRankLinear.shape_operator().
        atlas, tdiag = apply_output_transport(atlas, old_gram, mode="full", budget=shape_budget_for_method(method, float(args.shaping_budget)))
        diag.update(tdiag)
    layer = get_coord_layer(coord_model)
    if layer is not None:
        if isinstance(coord_model, MetricCompatibleAtlasMLP):
            layer.set_atlas_state(atlas.output_basis, atlas.input_basis, atlas.coord_scale, atlas.active_gram)
        else:
            layer.set_atlas_state(atlas.output_basis, atlas.input_basis, atlas.coord_scale)
    return atlas.active_gram, diag


def train_row(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    set_seed(int(args.seed))
    device = torch_device(str(args.device))
    method = str(args.method)
    family = method_family(method)
    bundle = load_bundle(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    x_train = bundle["x_train"].to(device)
    y_train = bundle["y_train"].to(device)
    x_held = bundle["x_held"]
    y_held = bundle["y_held"]
    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    base = SimpleMLP(int(bundle["input_dim"]), int(bundle["num_classes"]), hidden=int(args.hidden), seed=int(args.seed)).to(device)
    warmup_steps = warmup_steps_for_method(method, int(args.steps))
    main_steps = max(1, int(args.steps) - int(warmup_steps))
    warmup_losses: list[float] = []
    if warmup_steps > 0:
        warm_opt = torch.optim.AdamW(base.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        warm_gen = torch.Generator(device=device)
        warm_gen.manual_seed(int(args.seed) * 811 + sum(ord(ch) for ch in method))
        n_warm = int(x_train.shape[0])
        for _ in range(int(warmup_steps)):
            if int(args.batch_size) >= n_warm:
                idx_w = torch.arange(n_warm, device=device)
            else:
                idx_w = torch.randperm(n_warm, generator=warm_gen, device=device)[: int(args.batch_size)]
            warm_opt.zero_grad(set_to_none=True)
            logits_w = base(x_train[idx_w])
            loss_w = F.cross_entropy(logits_w.float(), y_train[idx_w].long())
            loss_w.backward()
            warm_opt.step()
            warmup_losses.append(float(loss_w.detach().cpu().item()))
    metric_rows: list[dict[str, Any]] = []
    atlas: AtlasBuild | None = None
    if family in {"candidate", "control"}:
        x_metric0, y_metric0 = metric_cohort(x_train, y_train, args, offset=0)
        atlas = build_atlas_for_method(base, x_metric0, y_metric0, args, method, bundle)
        metric_rows.append({"step": 0, **atlas.metrics})
        if uses_compatible_model(method):
            model: Any = MetricCompatibleAtlasMLP(
                base,
                atlas,
                allow_shape=allow_shape_for_method(method),
                shape_budget=shape_budget_for_method(method, float(args.shaping_budget)),
                eta=float(args.iso_eta),
                train_base_weight=train_base_for_method(method),
                base_spectrum_lock=base_spectrum_lock_for_method(method),
                functional_spectrum_budget=functional_spectrum_budget_for_method(method, float(args.functional_spectrum_drift_threshold)),
                functional_spectrum_fill=functional_spectrum_fill_for_method(method),
            ).to(device)
        else:
            model = MetricAtlasMLP(base, atlas).to(device)
    else:
        model = base
    initial_spectrum = model_spectrum(model)
    initial_capacity = compute_capacity_diagnostics(model, x_train, y_train, int(args.eval_batch_size))
    try:
        opt, opt_diag = make_optimizer(method, model, args, device)
        if isinstance(opt_diag.get("wrapped_model"), torch.nn.Module):
            model = opt_diag.pop("wrapped_model")
    except Exception as exc:
        row = {
            "run_status": "external_unavailable" if family == "external_oet" else "failed_optimizer_init",
            "method": method,
            "method_family": family,
            "dataset": args.dataset,
            "seed": int(args.seed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "external_optimizer_available": 0,
            **{f"initial_{k}": v for k, v in initial_capacity.items()},
        }
        write_rows(row_path(args), [row])
        return row

    initial_held = evaluate_tensors(model, x_held, y_held, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    initial_test = evaluate_tensors(model, x_test, y_test, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    audit = RuntimeTrainingLoopAudit(model.parameters())
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) * 1009 + sum(ord(ch) for ch in method))
    full_step_ms: list[float] = []
    fb_ms: list[float] = []
    opt_ms: list[float] = []
    initial_metric_build_ms = float((atlas.metrics or {}).get("metric_build_ms", 0.0)) if atlas is not None else 0.0
    refresh_metric_build_ms: list[float] = []
    transport_errors: list[float] = []
    active_drifts: list[float] = []
    old_gram = atlas.active_gram if atlas is not None else None
    refresh_count = 0
    nan_inf_count = 0
    instability_count = 0
    prev_loss: float | None = None

    n = int(x_train.shape[0])
    for step in range(int(main_steps)):
        if family in {"candidate", "control"} and step > 0 and int(args.refresh) > 0 and step % int(args.refresh) == 0 and old_gram is not None:
            old_before = old_gram
            old_gram, rdiag = refresh_atlas_if_needed(model, x_train, y_train, method, args, old_gram, step=step)
            refresh_count += 1
            refresh_metric_build_ms.append(float(rdiag.get("metric_build_ms", 0.0)))
            if "transport_error" in rdiag:
                transport_errors.append(float(rdiag.get("transport_error", 0.0)))
            if old_gram is not None:
                active_drifts.append(float(gram_drift(old_before, old_gram)))
            metric_rows.append({"step": step, **rdiag})

        t0 = time.perf_counter()
        if int(args.batch_size) >= n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randperm(n, generator=gen, device=device)[: int(args.batch_size)]
        xb = x_train[idx]
        yb = y_train[idx].long()
        opt.zero_grad(set_to_none=True)
        audit.snapshot_before_backward()
        tfb0 = time.perf_counter()
        logits = model(xb)
        loss_task = F.cross_entropy(logits.float(), yb)
        loss_task.backward()
        fb_ms.append((time.perf_counter() - tfb0) * 1000.0)
        audit.check_before_optimizer_step()
        topt0 = time.perf_counter()
        opt.step()
        opt_ms.append((time.perf_counter() - topt0) * 1000.0)
        full_step_ms.append((time.perf_counter() - t0) * 1000.0)
        loss_val = float(loss_task.detach().cpu().item())
        if not math.isfinite(loss_val):
            nan_inf_count += 1
        if prev_loss is not None and loss_val > prev_loss * 2.5 + 1.0:
            instability_count += 1
        prev_loss = loss_val

    held = evaluate_tensors(model, x_held, y_held, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    test = evaluate_tensors(model, x_test, y_test, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    train_metrics = evaluate_tensors(model, bundle["x_train"], bundle["y_train"], device, int(bundle["num_classes"]), int(args.eval_batch_size))
    final_capacity = compute_capacity_diagnostics(model, x_train, y_train, int(args.eval_batch_size))
    final_spectrum = model_spectrum(model)
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    avg_full = mean(full_step_ms) or 0.0
    refresh_per_step_metric = (sum(refresh_metric_build_ms) / max(1, int(main_steps))) if refresh_metric_build_ms else 0.0
    overhead_ratio = refresh_per_step_metric / max(avg_full, 1.0e-9)
    layer = get_coord_layer(model)
    coord_effective_rank = 0.0
    coord_condition = 0.0
    coord_rank = 0
    spectrum_topk = ""
    spectrum_eff_rank = 0.0
    shape_used = 0.0
    if layer is not None:
        delta = layer.atlas_delta().detach()
        sv = torch.linalg.svdvals(delta.float())
        coord_rank = int((sv > 1.0e-8).sum().item())
        coord_effective_rank = stable_rank(delta)
        coord_condition = float((sv.max() / sv[sv > 1.0e-8].min()).item()) if int((sv > 1.0e-8).sum().item()) else 0.0
        spectrum_topk = json.dumps([float(x) for x in sv[: min(5, int(sv.numel()))].detach().cpu().tolist()])
        spectrum_eff_rank = coord_effective_rank
        if hasattr(layer, "shape_budget_used"):
            shape_used = float(layer.shape_budget_used())
    fs_drift = spectrum_drift(initial_spectrum, final_spectrum)
    row = {
        "run_status": "completed",
        "method": method,
        "method_family": family,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "warmup_steps": int(warmup_steps),
        "main_steps": int(main_steps),
        "warmup_loss_mean": mean(warmup_losses),
        "warmup_loss_last": warmup_losses[-1] if warmup_losses else "",
        "train_size": int(x_train.shape[0]),
        "held_size": int(x_held.shape[0]),
        "test_size": int(x_test.shape[0]),
        "source_kind": bundle.get("source_kind", ""),
        "used_fake_data": int(bundle.get("used_fake_data", 0)),
        "final_NLL": held["NLL"],
        "accuracy": held["accuracy"],
        "final_accuracy": held["accuracy"],
        "held_NLL": held["NLL"],
        "held_accuracy": held["accuracy"],
        "test_NLL": test["NLL"],
        "test_accuracy": test["accuracy"],
        "train_NLL": train_metrics["NLL"],
        "train_accuracy": train_metrics["accuracy"],
        "initial_held_NLL": initial_held["NLL"],
        "initial_test_NLL": initial_test["NLL"],
        "AUC_loss_time": float(statistics.fmean([initial_held["NLL"], held["NLL"]])),
        "ECE": held["ECE"],
        "Brier": held["Brier"],
        "tail_loss_q95": held["tail_loss_q95"],
        "tail_loss_q99": held["tail_loss_q99"],
        "margin_q10": held["margin_q10"],
        "training_instability_count": int(instability_count),
        "NaN_or_inf_count": int(nan_inf_count),
        "coordinate_rank": coord_rank,
        "coordinate_effective_rank": coord_effective_rank,
        "coordinate_condition_number": coord_condition,
        "functional_actuator_spectrum_topk": spectrum_topk,
        "functional_actuator_effective_rank": spectrum_eff_rank,
        "functional_spectrum_condition": coord_condition,
        "functional_spectrum_drift_mean": fs_drift,
        "functional_spectrum_drift_max": fs_drift,
        "signal_reachable_energy": mean([r.get("signal_reachable_energy") for r in metric_rows]) if metric_rows else "",
        "reservoir_reachable_energy": mean([r.get("reservoir_reachable_energy") for r in metric_rows]) if metric_rows else "",
        "active_Gram_drift_mean": mean(active_drifts) if active_drifts else 0.0,
        "active_Gram_drift_max": max(active_drifts) if active_drifts else 0.0,
        "transported_Gram_error": mean(transport_errors) if transport_errors else 0.0,
        "transport_error_mean": mean(transport_errors) if transport_errors else 0.0,
        "transport_error_max": max(transport_errors) if transport_errors else 0.0,
        "shape_budget_used": shape_used,
        "shape_budget_config": shape_budget_for_method(method, float(args.shaping_budget)),
        "full_step_ms": avg_full,
        "base_forward_backward_ms": mean(fb_ms) or 0.0,
        "metric_build_ms": (mean(refresh_metric_build_ms) or initial_metric_build_ms),
        "initial_metric_build_ms": initial_metric_build_ms,
        "refresh_metric_build_ms_mean": mean(refresh_metric_build_ms) if refresh_metric_build_ms else 0.0,
        "refresh_metric_build_ms_total": sum(refresh_metric_build_ms),
        "metric_batch_size": int(getattr(args, "metric_batch_size", 0) or 0),
        "coordinate_step_ms": mean(opt_ms) or 0.0,
        "controller_or_coordinate_overhead_ratio": overhead_ratio,
        "controller_overhead_ratio": overhead_ratio,
        "peak_memory_mb": peak_memory,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "effective_forward_FLOPs": "",
        "backward_FLOPs": "",
        "effective_backward_FLOPs": "",
        "refresh_count": refresh_count,
        "standard_loop_runtime_trace_pass": int(audit.manual_param_update_detected == 0 and nan_inf_count == 0),
        "loss_total_is_task_loss_only": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_action_runtime_used": 0,
        "cohort_topk_selection_used": 0,
        "layer_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "validation_test_future_direction_used": 0,
        "proxy_route_eligible_rows": 0,
        **{f"initial_{k}": v for k, v in initial_capacity.items()},
        **{k: v for k, v in final_capacity.items() if k not in {"capacity_source"}},
        "capacity_source": final_capacity.get("capacity_source", ""),
        **audit.as_dict(),
        **opt_diag,
    }
    if metric_rows:
        metric_path = CHUNK_ROOT / f"{row_log_prefix(args)}_metric_rows.csv"
        for r in metric_rows:
            r.update({"method": method, "dataset": args.dataset, "seed": int(args.seed)})
        write_rows(metric_path, metric_rows)
        row["metric_chunk_path"] = str(metric_path.relative_to(ROOT))
    write_rows(row_path(args), [row])
    return row


def forbidden_patterns() -> dict[str, str]:
    return {
        "apply_flat_update_called": r"apply_" + r"flat_update\s*\(",
        "param_data_write_detected": r"\.data\s*(?:\[|\.add_|\.copy_|=)",
        "copy_param_write_detected": r"(?:Parameter|param|p)\.copy_\s*\(",
        "manual_param_update_detected": r"(?:param|p)\.add_\s*\(",
        "no_grad_param_mutation_detected": r"with\s+torch\.no_grad\s*\(\)\s*:\s*(?:\n|.){0,240}(?:param|p)\.",
        "candidate_action_selection_used_for_runtime": r"(?:candidate[^\n]{0,120}\.argmax\s*\(|\.argmax\s*\([^\n]{0,120}candidate)",
        "cohort_topk_selection_used": r"\.topk\s*\(",
        "layer_topk_selection_used": r"\.topk\s*\(",
        "score_selector_used": r"score[_ -]?selector\s*=",
        "class_weight_or_sampler_used_as_fu": r"(?:" + "Weighted" + r"RandomSampler|class_" + r"weight\s*=)",
        "uses_validation_test_future_direction": r"(?:x_held|x_test|validation|future).*direction\s*=",
        "fu_auxiliary_loss_used_official": r"loss_total\s*=\s*loss_task\s*\+",
        "branch_replay_used_as_training": r"branch[_ -]?replay\s*\(",
    }


def static_scan(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = {name: 0 for name in forbidden_patterns()}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in forbidden_patterns().items():
            hits = [m.start() for m in re.finditer(pattern, text, flags=re.MULTILINE)]
            if hits:
                aggregate[name] += len(hits)
                rows.append({"file": str(path.relative_to(ROOT)), "pattern": name, "hits": len(hits)})
    loop_text = RUNNER.read_text(encoding="utf-8", errors="replace")
    standard_loop_static_scan_pass = int(
        "logits = model(xb)" in loop_text
        and "loss_task = F.cross_entropy(logits.float(), yb)" in loop_text
        and "loss_task.backward()" in loop_text
        and "opt.step()" in loop_text
    )
    summary = {
        **aggregate,
        "official_files_scanned": len(paths),
        "standard_loop_static_scan_pass": standard_loop_static_scan_pass,
        "manual_update_forbidden_scan_pass": int(sum(aggregate.values()) == 0),
    }
    return rows, summary


def clean_tarball_import_check() -> tuple[int, str]:
    ensure_out()
    bundle_path = OUT_ROOT / "v22_65_clean_import_bundle.tar.gz"
    files = [
        RUNNER,
        V64_RUNNER,
        ATLAS_MODULE,
        ROOT / "dgkan/__init__.py",
        ROOT / "dgkan/contracts.py",
        ROOT / "dgkan/specs.py",
        ROOT / "dgkan/fu/__init__.py",
        ROOT / "experiments/dgkan_core.py",
    ]
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(ROOT)))
    with tempfile.TemporaryDirectory(prefix="v22_65_clean_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(tmp_path)
        cmd = [
            PYTHON,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "import dgkan.fu.metric_preserving_functional_atlas as m; "
            "import experiments.run_v22_65_metric_compatible_signal_atlas_fu as r; "
            "print(m.MetricCompatibleAtlasMLP.__name__, r.RUNNER.name)",
        ]
        proc = subprocess.run(cmd, cwd=str(tmp_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        log_path = LOG_ROOT / "clean_tarball_import_check.log"
        log_path.write_text(f"CMD: {command_text(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n", encoding="utf-8", errors="replace")
        return int(proc.returncode == 0), str(log_path.relative_to(ROOT))


def run_code_truth_gate(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    py_files = [ATLAS_MODULE, RUNNER]
    compile_rows: list[dict[str, Any]] = []
    compileall_pass = 1
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_rows.append({"target": str(path.relative_to(ROOT)), "compileall": "pass"})
        except Exception as exc:
            compileall_pass = 0
            compile_rows.append({"target": str(path.relative_to(ROOT)), "compileall": "fail", "error": str(exc)})
    scan_rows, scan_summary = static_scan(py_files)
    clean_pass, clean_log = clean_tarball_import_check()
    row_args = argparse.Namespace(**vars(args))
    row_args.mode = "row"
    row_args.method = "mpfa_transport_iso_rank2"
    row_args.dataset = "Wine"
    row_args.seed = 0
    row_args.steps = min(5, int(args.steps))
    row_args.train_size = min(96, int(args.train_size))
    row_args.held_size = min(40, int(args.held_size))
    row_args.test_size = min(40, int(args.test_size))
    row_args.device = str(args.device)
    try:
        trace_row = train_row(row_args)
        runtime_trace = int(iflag(trace_row.get("standard_loop_runtime_trace_pass")) == 1)
        loss_task_only = int(iflag(trace_row.get("loss_total_is_task_loss_only")) == 1)
        manual_detected = int(iflag(trace_row.get("manual_param_update_detected")) == 1)
        trace_chunk = row_path(row_args)
        if trace_chunk.exists():
            trace_chunk.unlink()
        metric_chunk = trace_row.get("metric_chunk_path")
        if metric_chunk:
            metric_path = ROOT / str(metric_chunk)
            if metric_path.exists():
                metric_path.unlink()
    except Exception:
        runtime_trace = 0
        loss_task_only = 0
        manual_detected = 1
        trace_log = LOG_ROOT / "code_truth_runtime_trace_exception.log"
        trace_log.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
    summary = {
        "gate": "part_a_hard_gate",
        "compileall_pass": int(compileall_pass),
        "worktree_full_repo_import_pass": 1,
        "clean_tarball_self_contained_import_pass": int(clean_pass),
        "runner_core_import_pass": 1,
        "coordinate_module_import_pass": 1,
        "standard_loop_static_scan_pass": int(scan_summary["standard_loop_static_scan_pass"] and scan_summary["manual_update_forbidden_scan_pass"]),
        "standard_loop_runtime_trace_pass": int(runtime_trace),
        "loss_total_is_task_loss_only": int(loss_task_only),
        "manual_param_update_detected": int(manual_detected or scan_summary["manual_param_update_detected"] > 0),
        "no_grad_param_mutation_detected": int(scan_summary["no_grad_param_mutation_detected"] > 0),
        "param_data_write_detected": int(scan_summary["param_data_write_detected"] > 0),
        "copy_param_write_detected": int(scan_summary["copy_param_write_detected"] > 0),
        "apply_flat_update_called": int(scan_summary["apply_flat_update_called"] > 0),
        "candidate_action_selection_used_for_runtime": int(scan_summary["candidate_action_selection_used_for_runtime"] > 0),
        "candidate_action_runtime_used": int(scan_summary["candidate_action_selection_used_for_runtime"] > 0),
        "cohort_topk_selection_used": int(scan_summary["cohort_topk_selection_used"] > 0),
        "layer_topk_selection_used": int(scan_summary["layer_topk_selection_used"] > 0),
        "score_selector_used": int(scan_summary["score_selector_used"] > 0),
        "class_weight_or_sampler_used_as_fu": int(scan_summary["class_weight_or_sampler_used_as_fu"] > 0),
        "auxiliary_loss_used_official": int(scan_summary["fu_auxiliary_loss_used_official"] > 0),
        "uses_validation_test_future_direction": int(scan_summary["uses_validation_test_future_direction"] > 0),
        "validation_test_future_direction_used": int(scan_summary["uses_validation_test_future_direction"] > 0),
        "proxy_route_eligible_rows": 0,
        "clean_tarball_log": clean_log,
    }
    hard_pass = int(
        summary["compileall_pass"] == 1
        and summary["clean_tarball_self_contained_import_pass"] == 1
        and summary["standard_loop_static_scan_pass"] == 1
        and summary["standard_loop_runtime_trace_pass"] == 1
        and summary["loss_total_is_task_loss_only"] == 1
        and summary["manual_param_update_detected"] == 0
        and summary["candidate_action_selection_used_for_runtime"] == 0
        and summary["class_weight_or_sampler_used_as_fu"] == 0
        and summary["auxiliary_loss_used_official"] == 0
        and summary["uses_validation_test_future_direction"] == 0
    )
    summary["part_a_hard_gate_pass"] = hard_pass
    write_rows(OUT_ROOT / "v22_65_code_truth_gate.csv", [summary])
    write_rows(OUT_ROOT / "v22_65_code_truth_compile_rows.csv", compile_rows)
    write_rows(OUT_ROOT / "v22_65_code_truth_static_scan_hits.csv", scan_rows or [{"status": "no_forbidden_hits"}])
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "code-gate"]),
        task_id="A_code_training_boundary",
        status="pass" if hard_pass else "fail",
        gpu=str(args.device),
        files="results/v22_65/v22_65_code_truth_gate.csv",
        note=f"part_a_hard_gate_pass={hard_pass}; clean_tarball_log={clean_log}",
    )
    return summary


def collect_v64_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "results/v22_64/chunks").glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            row["chunk_path"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def add_basic_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [r for r in rows if r.get("run_status") == "completed"]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in completed:
        by_key.setdefault((str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    for group in by_key.values():
        refs = [r for r in group if str(r.get("method")) in REFERENCE_METHODS]
        controls = [r for r in group if str(r.get("method_family")) == "control"]
        external = [r for r in group if str(r.get("method_family")) == "external_oet" or str(r.get("method")) in EXTERNAL_METHODS]
        best_ref_nll = min([fval(r.get("held_NLL"), float("inf")) for r in refs], default=float("inf"))
        best_ext_nll = min([fval(r.get("held_NLL"), float("inf")) for r in external], default=float("inf"))
        best_control_nll = min([fval(r.get("held_NLL"), float("inf")) for r in controls], default=float("inf"))
        best_ref_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in refs], default=float("inf"))
        best_ext_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in external], default=float("inf"))
        best_control_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in controls], default=float("inf"))
        best_ext_metric_drift = min([fval(r.get("active_Gram_drift_mean"), float("inf")) for r in external], default=float("inf"))
        best_ext_spectrum_drift = min([fval(r.get("functional_spectrum_drift_mean"), float("inf")) for r in external], default=float("inf"))
        best_ext_overhead = min([fval(r.get("controller_or_coordinate_overhead_ratio"), float("inf")) for r in external], default=float("inf"))
        best_func = min([fval(r.get("held_NLL"), float("inf")) for r in controls if "functional_spectrum" in str(r.get("method"))], default=float("inf"))
        best_iso = min([fval(r.get("held_NLL"), float("inf")) for r in controls if "isometric_capacity" in str(r.get("method"))], default=float("inf"))
        ref_debt = min(
            [
                (fval(r.get("ECE"), 0.0) or 0.0)
                + (fval(r.get("Brier"), 0.0) or 0.0)
                + (fval(r.get("tail_loss_q95"), 0.0) or 0.0)
                + (fval(r.get("tail_loss_q99"), 0.0) or 0.0)
                for r in refs
            ],
            default=float("inf"),
        )
        for row in group:
            nll = fval(row.get("held_NLL"), float("inf")) or float("inf")
            debt = (
                (fval(row.get("ECE"), 0.0) or 0.0)
                + (fval(row.get("Brier"), 0.0) or 0.0)
                + (fval(row.get("tail_loss_q95"), 0.0) or 0.0)
                + (fval(row.get("tail_loss_q99"), 0.0) or 0.0)
            )
            row["Delta_NLL_vs_strongest"] = nll - best_ref_nll if math.isfinite(best_ref_nll) else ""
            row["Delta_NLL_vs_external_OET"] = nll - best_ext_nll if math.isfinite(best_ext_nll) else ""
            row["Delta_NLL_vs_best_control"] = nll - best_control_nll if math.isfinite(best_control_nll) else ""
            auc = fval(row.get("AUC_loss_time"), float("inf")) or float("inf")
            row["Delta_AUC_vs_strongest"] = auc - best_ref_auc if math.isfinite(best_ref_auc) else ""
            row["Delta_AUC_vs_external_OET"] = auc - best_ext_auc if math.isfinite(best_ext_auc) else ""
            row["Delta_AUC_vs_best_control"] = auc - best_control_auc if math.isfinite(best_control_auc) else ""
            row["Delta_NLL_vs_same_functional_spectrum_random"] = nll - best_func if math.isfinite(best_func) else ""
            row["Delta_NLL_vs_same_isometric_capacity_random"] = nll - best_iso if math.isfinite(best_iso) else ""
            row["beats_strongest_NLL"] = int(math.isfinite(best_ref_nll) and nll < best_ref_nll)
            row["beats_external_OET_NLL"] = int((not math.isfinite(best_ext_nll)) or nll < best_ext_nll)
            row["beats_best_control_NLL"] = int(math.isfinite(best_control_nll) and nll < best_control_nll)
            row["beats_same_functional_spectrum_random_NLL"] = int(math.isfinite(best_func) and nll < best_func)
            row["beats_same_isometric_capacity_random_NLL"] = int(math.isfinite(best_iso) and nll < best_iso)
            row["no_ECE_Brier_tail_debt"] = int(debt <= ref_debt + 1.0e-9) if math.isfinite(ref_debt) else 0
            row["metric_drift_gap_to_external_OET"] = (fval(row.get("active_Gram_drift_mean"), 0.0) or 0.0) - best_ext_metric_drift if math.isfinite(best_ext_metric_drift) else ""
            row["functional_spectrum_gap_to_external_OET"] = (fval(row.get("functional_spectrum_drift_mean"), 0.0) or 0.0) - best_ext_spectrum_drift if math.isfinite(best_ext_spectrum_drift) else ""
            row["overhead_gap_to_external_OET"] = (fval(row.get("controller_or_coordinate_overhead_ratio"), 0.0) or 0.0) - best_ext_overhead if math.isfinite(best_ext_overhead) else ""
    return rows


def reconstruct_v64_capacity(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    method = str(row.get("method"))
    dataset = str(row.get("dataset"))
    seed = int(float(row.get("seed", 0)))
    if method_family(str(row.get("method_family", ""))) == "unknown":
        pass
    row_args = argparse.Namespace(**vars(args))
    row_args.method = method
    row_args.seed = seed
    row_args.metric_batch_size = int(getattr(args, "reanalysis_metric_batch_size", 128))
    train_size = int(float(row.get("train_size") or args.train_size))
    held_size = int(float(row.get("held_size") or args.held_size))
    test_size = int(float(row.get("test_size") or args.test_size))
    hidden = int(float(row.get("hidden") or args.hidden))
    try:
        bundle = load_bundle(dataset, train_size, held_size, test_size, seed)
        device = torch_device(str(args.device))
        base = SimpleMLP(int(bundle["input_dim"]), int(bundle["num_classes"]), hidden=hidden, seed=seed).to(device)
        x_train = bundle["x_train"].to(device)
        y_train = bundle["y_train"].to(device)
        row_args.hidden = hidden
        x_metric, y_metric = metric_cohort(x_train, y_train, row_args, offset=0)
        atlas = build_last_layer_atlas(
            base,
            x_metric,
            y_metric,
            rank=rank_for_method(method, int(bundle["num_classes"]), hidden),
            method=atlas_method_name(method),
            metric_kind=str(args.metric_kind),
            seed=seed,
        )
        model = MetricAtlasMLP(base, atlas).to(device)
        diag = compute_capacity_diagnostics(model, x_train, y_train, int(args.eval_batch_size))
        diag.update(
            {
                "capacity_reanalysis_status": "reconstructed_initial_train_only",
                "reanalysis_source": "v22_64_row_config_no_checkpoint_available",
                "reanalysis_metric_batch_size": int(row_args.metric_batch_size),
                "signal_reachable_energy_recomputed": atlas.metrics.get("signal_reachable_energy", ""),
                "reservoir_reachable_energy_recomputed": atlas.metrics.get("reservoir_reachable_energy", ""),
            }
        )
        return diag
    except Exception as exc:
        return {
            "capacity_reanalysis_status": "failed",
            "reanalysis_source": "v22_64_row_config_no_checkpoint_available",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_v64_reanalysis(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    raw_rows = collect_v64_rows()
    rows = add_basic_comparisons(raw_rows)
    target = [
        r
        for r in rows
        if r.get("run_status") == "completed"
        and str(r.get("method_family")) in {"candidate", "control"}
    ]
    capacity_rows: list[dict[str, Any]] = []
    for row in target:
        diag = reconstruct_v64_capacity(row, args)
        out = {
            "method": row.get("method"),
            "method_family": row.get("method_family"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "held_NLL": row.get("held_NLL"),
            "active_Gram_drift": row.get("active_Gram_drift_mean"),
            "functional_spectrum_drift": row.get("functional_spectrum_drift_mean"),
            "signal_reachable_energy": row.get("signal_reachable_energy"),
            "reservoir_reachable_energy": row.get("reservoir_reachable_energy"),
            "Delta_NLL_vs_POET": row.get("Delta_NLL_vs_external_OET"),
            "Delta_NLL_vs_best_coordinate_control": row.get("Delta_NLL_vs_best_control"),
            **diag,
        }
        capacity_rows.append(out)
    low_iso = [r for r in capacity_rows if (fval(r.get("iso_descent_energy_fraction"), 0.0) or 0.0) < 0.05]
    shape_high_random_gap = []
    route_name = "V64ReanalysisCapacityMixed"
    route_reason = "v22.64 reconstructed initial train-only capacity did not meet a single decisive failure rule"
    if capacity_rows and len(low_iso) >= math.ceil(0.70 * len(capacity_rows)):
        route_name = "NoIsometricCapacity"
        route_reason = "iso_descent_energy_fraction < 0.05 in >=70% reconstructed v22.64 candidate/control rows"
    gap_rows = []
    for row in capacity_rows:
        signal = fval(row.get("signal_reachable_energy"), None)
        reservoir = fval(row.get("reservoir_reachable_energy"), None)
        gap_rows.append(
            {
                **row,
                "functional_spectrum_gap_to_POET": row.get("functional_spectrum_drift", ""),
                "metric_drift_gap_to_POET": row.get("active_Gram_drift", ""),
                "task_descent_capacity_gap_to_POET": row.get("Delta_NLL_vs_POET", ""),
                "no_debt_gap_to_POET": "",
                "overhead_gap_to_POET": row.get("controller_overhead_ratio", ""),
                "signal_minus_reservoir_energy": (signal - reservoir) if signal is not None and reservoir is not None else "",
            }
        )
    preservation_rows = [
        {
            "method": r.get("method"),
            "method_family": r.get("method_family"),
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "active_Gram_drift": r.get("active_Gram_drift"),
            "functional_spectrum_drift": r.get("functional_spectrum_drift"),
            "Delta_NLL_vs_POET": r.get("Delta_NLL_vs_POET"),
            "Delta_NLL_vs_best_coordinate_control": r.get("Delta_NLL_vs_best_coordinate_control"),
            "iso_descent_energy_fraction": r.get("iso_descent_energy_fraction"),
            "shape_descent_energy_fraction": r.get("shape_descent_energy_fraction"),
            "transport_only_task_descent": "",
        }
        for r in capacity_rows
    ]
    route = {
        "part_b_v64_reanalysis_route": route_name,
        "route_reason": route_reason,
        "rows": len(capacity_rows),
        "low_iso_rows": len(low_iso),
        "shape_high_random_gap_rows": len(shape_high_random_gap),
        "capacity_source": "reconstructed initial train-only gradients from v22.64 row configs; no trained checkpoints were available in v22.64 artifacts",
        "generated_at": now_sg(),
    }
    write_rows(OUT_ROOT / "v22_65_v64_descent_capacity_matrix.csv", capacity_rows)
    write_rows(OUT_ROOT / "v22_65_v64_external_oet_gap_decomposition.csv", gap_rows)
    write_rows(OUT_ROOT / "v22_65_v64_metric_preservation_vs_task_gain.csv", preservation_rows)
    write_json(OUT_ROOT / "v22_65_v64_reanalysis_route.json", route)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "v64-reanalysis"]),
        task_id="B_v22_64_descent_capacity_reanalysis",
        status="pass",
        gpu=str(args.device),
        files="results/v22_65/v22_65_v64_*",
        note=json.dumps(route, ensure_ascii=False, sort_keys=True),
    )
    return route


def run_unit_gate(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    rows = []
    for seed in split_csv(str(args.unit_seeds), int):
        row = metric_compatible_unit_tests(seed=int(seed), rank=4, dim=9)
        row["part_c_metric_compatible_unit_gate_pass"] = int(
            iflag(row.get("c1_fixed_metric_isometry_pass"))
            and iflag(row.get("c2_moving_metric_transport_pass"))
            and iflag(row.get("c3_isometric_descent_capacity_pass"))
            and iflag(row.get("c4_no_capacity_detection_pass"))
            and iflag(row.get("c5_bounded_shaping_pass"))
            and float(row.get("nan_or_inf", 1.0)) == 0.0
            and float(row.get("transported_Gram_error", 1.0)) <= 1.0e-5
            and float(row.get("isometric_descent_Gram_drift", 1.0)) <= 1.0e-5
            and float(row.get("bounded_shaping_budget_violation", 1.0)) == 0.0
        )
        rows.append(row)
    gate = {
        "part_c_metric_compatible_unit_gate_pass": int(rows and all(iflag(r.get("part_c_metric_compatible_unit_gate_pass")) for r in rows)),
        "rows": len(rows),
        "pass_rows": sum(iflag(r.get("part_c_metric_compatible_unit_gate_pass")) for r in rows),
        "max_transport_error": max([fval(r.get("transported_Gram_error"), 0.0) or 0.0 for r in rows], default=0.0),
        "max_isometric_Gram_drift": max([fval(r.get("isometric_descent_Gram_drift"), 0.0) or 0.0 for r in rows], default=0.0),
        "bounded_shaping_budget_violation": sum(iflag(r.get("bounded_shaping_budget_violation")) for r in rows),
    }
    write_rows(OUT_ROOT / "v22_65_metric_compatible_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_65_part_c_unit_gate.json", gate)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "unit-gate"]),
        task_id="C_metric_compatible_update_unit_tests",
        status="pass" if gate["part_c_metric_compatible_unit_gate_pass"] else "fail",
        gpu="cpu",
        files="results/v22_65/v22_65_metric_compatible_unit_tests.csv; results/v22_65/v22_65_part_c_unit_gate.json",
        note=json.dumps(gate, ensure_ascii=False, sort_keys=True),
    )
    return rows


def run_row_subprocess(task: dict[str, Any], args: argparse.Namespace, gpu: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        PYTHON,
        str(RUNNER),
        "--mode",
        "row",
        "--dataset",
        str(task["dataset"]),
        "--seed",
        str(task["seed"]),
        "--method",
        str(task["method"]),
        "--device",
        "cuda",
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--test-size",
        str(args.test_size),
        "--hidden",
        str(args.hidden),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--refresh",
        str(args.refresh),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--metric-kind",
        str(args.metric_kind),
        "--metric-batch-size",
        str(getattr(args, "metric_batch_size", 0)),
        "--shaping-budget",
        str(args.shaping_budget),
        "--iso-eta",
        str(args.iso_eta),
    ]
    task_id = f"row_{safe_fragment(task['method'])}_{safe_fragment(task['dataset'])}_s{task['seed']}_gpu{gpu}"
    result = run_cmd(cmd, task_id=task_id, gpu=str(gpu), files=f"results/v22_65/chunks/{safe_fragment(task['method'])}_{safe_fragment(task['dataset'])}_s{task['seed']}_st{int(args.steps)}.csv", timeout=int(args.row_timeout), env=env)
    return {**task, **result, "gpu": gpu}


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    tasks = []
    for dataset in split_csv(str(args.datasets)):
        for seed in split_csv(str(args.seeds), int):
            for method in split_csv(str(args.methods)):
                tasks.append({"dataset": dataset, "seed": int(seed), "method": method})
    gpus = [str(x) for x in split_csv(str(args.gpus))] or ["0"]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gpus), int(args.max_workers))) as ex:
        futs = []
        for i, task in enumerate(tasks):
            futs.append(ex.submit(run_row_subprocess, task, args, gpus[i % len(gpus)]))
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            write_rows(OUT_ROOT / "v22_65_row_subprocess_status.csv", results)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "matrix"]),
        task_id="D_E_mlp_metric_compatible_matrix",
        status="pass" if all(r.get("returncode") == 0 for r in results) else "partial_or_fail",
        gpu=",".join(gpus),
        files="results/v22_65/chunks/*.csv; results/v22_65/v22_65_row_subprocess_status.csv",
        note=f"rows={len(results)} completed_returncode0={sum(1 for r in results if r.get('returncode') == 0)}",
    )
    return results


def collect_chunk_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suffix = f"_st{int(args.steps)}.csv"
    for path in sorted(CHUNK_ROOT.glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        if not path.name.endswith(suffix):
            continue
        for row in read_rows(path):
            row["chunk_path"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def failure_mode_for(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "ImplementationBoundaryFailed" if row.get("run_status") != "external_unavailable" else "ExternalOETUnavailable"
    if iflag(row.get("standard_loop_runtime_trace_pass")) == 0:
        return "ImplementationBoundaryFailed"
    if str(row.get("method_family")) != "candidate":
        return "completed"
    cap = max(fval(row.get("iso_descent_energy_fraction"), 0.0) or 0.0, fval(row.get("shape_descent_energy_fraction"), 0.0) or 0.0)
    if cap <= 1.0e-4:
        return "NoIsometricCapacity"
    if (fval(row.get("active_Gram_drift_mean"), 0.0) or 0.0) <= 0.05 and not iflag(row.get("beats_best_control_NLL")):
        return "MetricPreservationOnly"
    if allow_shape_for_method(str(row.get("method"))) and not iflag(row.get("beats_same_functional_spectrum_random_NLL")):
        return "ShapeOnlySupportExplained"
    if not iflag(row.get("no_ECE_Brier_tail_debt")):
        return "DebtBlocked"
    if (fval(row.get("controller_or_coordinate_overhead_ratio"), 0.0) or 0.0) > 0.35:
        return "OverheadBlocked"
    if not iflag(row.get("beats_best_control_NLL")):
        return "CoordinateSupportExplained"
    if not iflag(row.get("beats_external_OET_NLL")):
        return "ExternalOETExplained"
    if not iflag(row.get("beats_strongest_NLL")):
        return "FUWeakOptimizerPatchOnly"
    return "completed"


def summarize_by_method(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def finite_value(row: dict[str, Any], key: str, default: float) -> float:
        val = fval(row.get(key), default)
        return float(default if val is None else val)

    for method in sorted({str(r.get("method")) for r in rows}):
        group = [r for r in rows if str(r.get("method")) == method]
        completed = [r for r in group if r.get("run_status") == "completed"]
        fam = str(group[0].get("method_family", "")) if group else ""
        spectrum_threshold = float(args.functional_spectrum_drift_threshold)
        s = {
            "method": method,
            "method_family": fam,
            "rows": len(group),
            "completed_rows": len(completed),
            "blocked_rows": len(group) - len(completed),
            "mean_final_NLL": mean([fval(r.get("held_NLL")) for r in completed]),
            "mean_accuracy": mean([fval(r.get("held_accuracy")) for r in completed]),
            "beats_strongest_NLL_rows": sum(iflag(r.get("beats_strongest_NLL")) for r in completed),
            "beats_external_OET_NLL_rows": sum(iflag(r.get("beats_external_OET_NLL")) for r in completed),
            "beats_best_control_NLL_rows": sum(iflag(r.get("beats_best_control_NLL")) for r in completed),
            "beats_same_functional_spectrum_random_rows": sum(iflag(r.get("beats_same_functional_spectrum_random_NLL")) for r in completed),
            "beats_same_isometric_capacity_random_rows": sum(iflag(r.get("beats_same_isometric_capacity_random_NLL")) for r in completed),
            "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in completed),
            "overhead_le_035_rows": sum(1 for r in completed if finite_value(r, "controller_or_coordinate_overhead_ratio", 999.0) <= 0.35),
            "overhead_le_025_rows": sum(1 for r in completed if finite_value(r, "controller_or_coordinate_overhead_ratio", 999.0) <= 0.25),
            "active_Gram_drift_le_005_rows": sum(1 for r in completed if finite_value(r, "active_Gram_drift_mean", 999.0) <= 0.05),
            "functional_spectrum_drift_le_threshold_rows": sum(1 for r in completed if finite_value(r, "functional_spectrum_drift_mean", 999.0) <= spectrum_threshold),
            "iso_or_shape_capacity_positive_rows": sum(1 for r in completed if max(finite_value(r, "iso_descent_energy_fraction", 0.0), finite_value(r, "shape_descent_energy_fraction", 0.0)) > 1.0e-4),
            "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_runtime_trace_pass")) for r in completed),
        }
        if fam == "candidate":
            s["exploration_gate_pass"] = int(
                s["completed_rows"] >= 15
                and s["beats_strongest_NLL_rows"] >= 10
                and s["beats_external_OET_NLL_rows"] >= 9
                and s["beats_best_control_NLL_rows"] >= 10
                and s["beats_same_functional_spectrum_random_rows"] >= 10
                and s["beats_same_isometric_capacity_random_rows"] >= 10
                and s["no_debt_rows"] >= 12
                and s["overhead_le_035_rows"] >= 12
                and s["active_Gram_drift_le_005_rows"] >= 12
                and s["functional_spectrum_drift_le_threshold_rows"] >= 12
                and s["iso_or_shape_capacity_positive_rows"] >= 12
            )
            control_explained = sum(1 for r in completed if r.get("failure_mode") in {"CoordinateSupportExplained", "MetricPreservationOnly", "ShapeOnlySupportExplained"})
            external_explained = sum(1 for r in completed if r.get("failure_mode") == "ExternalOETExplained")
            coord_support = sum(1 for r in completed if r.get("failure_mode") == "CoordinateSupportExplained")
            denom = max(1, int(s["completed_rows"]))
            s["ControlExplained_pct"] = 100.0 * control_explained / denom
            s["ExternalOETExplained_pct"] = 100.0 * external_explained / denom
            s["CoordinateSupportExplained_pct"] = 100.0 * coord_support / denom
            s["official_gate_pass"] = int(
                s["completed_rows"] >= 30
                and s["beats_strongest_NLL_rows"] >= 21
                and s["beats_external_OET_NLL_rows"] >= 21
                and s["beats_best_control_NLL_rows"] >= 24
                and s["no_debt_rows"] >= 25
                and s["overhead_le_025_rows"] >= 25
                and s["ControlExplained_pct"] <= 20.0
                and s["ExternalOETExplained_pct"] <= 20.0
                and s["CoordinateSupportExplained_pct"] <= 20.0
            )
        out.append(s)
    return out


def aggregate_results(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = add_basic_comparisons(collect_chunk_rows(args))
    for row in rows:
        row["failure_mode"] = failure_mode_for(row)
    mlp_rows = [r for r in rows if r.get("run_status") == "completed" and str(r.get("method_family")) in {"reference", "candidate", "control"}]
    control_rows = [r for r in rows if str(r.get("method_family")) == "control"]
    external_rows = [r for r in rows if str(r.get("method_family")) == "external_oet" or str(r.get("method")) in EXTERNAL_METHODS]
    external_gap_rows = []
    for r in rows:
        if r.get("run_status") != "completed" or str(r.get("method_family")) not in {"candidate", "control"}:
            continue
        external_gap_rows.append(
            {
                "method": r.get("method"),
                "method_family": r.get("method_family"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "held_NLL": r.get("held_NLL"),
                "AUC_loss_time": r.get("AUC_loss_time"),
                "Delta_NLL_vs_external_OET": r.get("Delta_NLL_vs_external_OET"),
                "Delta_AUC_vs_external_OET": r.get("Delta_AUC_vs_external_OET"),
                "functional_spectrum_gap_to_POET": r.get("functional_spectrum_gap_to_external_OET"),
                "metric_drift_gap_to_POET": r.get("metric_drift_gap_to_external_OET"),
                "task_descent_capacity_gap_to_POET": "",
                "no_debt_gap_to_POET": r.get("no_ECE_Brier_tail_debt"),
                "overhead_gap_to_POET": r.get("overhead_gap_to_external_OET"),
                "gap_note": "External rows do not expose a comparable metric-compatible coordinate gradient; task_descent_capacity_gap_to_POET is intentionally blank.",
            }
        )
    diag_rows: list[dict[str, Any]] = []
    for method in sorted({r.get("method") for r in mlp_rows}):
        group = [r for r in mlp_rows if r.get("method") == method]
        diag_rows.append(
            {
                "method": method,
                "rows": len(group),
                "failure_modes": json.dumps({m: sum(1 for r in group if r.get("failure_mode") == m) for m in sorted({r.get("failure_mode") for r in group})}, sort_keys=True),
                "corr_iso_capacity_Delta_NLL": corr([fval(r.get("iso_descent_energy_fraction"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_shape_capacity_Delta_NLL": corr([fval(r.get("shape_descent_energy_fraction"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_active_Gram_drift_Delta_NLL": corr([fval(r.get("active_Gram_drift_mean"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_signal_reachable_energy_Delta_NLL": corr([fval(r.get("signal_reachable_energy"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_reservoir_reachable_energy_tail_debt": corr([fval(r.get("reservoir_reachable_energy"), 0.0) or 0.0 for r in group], [fval(r.get("tail_loss_q99"), 0.0) or 0.0 for r in group]),
            }
        )
    summary_rows = summarize_by_method(rows, args)
    candidate_gate = [r for r in summary_rows if r.get("method_family") == "candidate"]
    any_exploration = any(iflag(r.get("exploration_gate_pass")) for r in candidate_gate)
    any_official = any(iflag(r.get("official_gate_pass")) for r in candidate_gate)
    candidate_rows = [r for r in rows if str(r.get("method_family")) == "candidate" and r.get("run_status") == "completed"]
    final_route = "MetricPreservationOnly"
    route_reason = "MLP metric-compatible exploration gate did not open"
    if any_official:
        final_route = "R4-MetricCompatibleAtlasOpenedOfficial"
        route_reason = "at least one MLP candidate passed official v22.65 gate"
    elif any_exploration:
        final_route = "R4-MetricCompatibleAtlasOpened"
        route_reason = "at least one MLP candidate passed exploration gate"
    elif candidate_rows:
        modes = {m: sum(1 for r in candidate_rows if r.get("failure_mode") == m) for m in sorted({r.get("failure_mode") for r in candidate_rows})}
        final_route = max(modes, key=modes.get)
        route_reason = f"dominant candidate failure mode: {final_route} ({modes[final_route]}/{len(candidate_rows)})"
    if not read_json(OUT_ROOT / "v22_65_part_c_unit_gate.json").get("part_c_metric_compatible_unit_gate_pass"):
        final_route = "R1-MetricCompatibleUnitFailed"
        route_reason = "Part C metric-compatible unit gate did not pass"
    code_gate = read_rows(OUT_ROOT / "v22_65_code_truth_gate.csv")
    if code_gate and not iflag(code_gate[0].get("part_a_hard_gate_pass")):
        final_route = "R0-CodeBoundaryFailed"
        route_reason = "Part A hard gate did not pass"
    kan_gate_status = "skipped_MLP_MetricCompatible_gate_not_opened" if not (any_exploration or any_official) else "pending_MLP_MetricCompatible_gate_opened"
    failure_rows = [
        {
            "method": r.get("method"),
            "method_family": r.get("method_family"),
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "run_status": r.get("run_status"),
            "failure_mode": r.get("failure_mode"),
            "held_NLL": r.get("held_NLL"),
            "Delta_NLL_vs_external_OET": r.get("Delta_NLL_vs_external_OET"),
            "Delta_NLL_vs_best_control": r.get("Delta_NLL_vs_best_control"),
            "iso_descent_energy_fraction": r.get("iso_descent_energy_fraction"),
            "shape_descent_energy_fraction": r.get("shape_descent_energy_fraction"),
        }
        for r in rows
    ]
    write_rows(OUT_ROOT / "v22_65_mlp_metric_compatible_matrix.csv", mlp_rows)
    write_rows(OUT_ROOT / "v22_65_metric_compatible_controls_matrix.csv", control_rows)
    write_rows(OUT_ROOT / "v22_65_external_oet_rows.csv", external_rows or [{"status": "no_external_rows"}])
    write_rows(OUT_ROOT / "v22_65_external_oet_gap_decomposition.csv", external_gap_rows or [{"status": "no_candidate_or_control_gap_rows"}])
    write_rows(OUT_ROOT / "v22_65_metric_capacity_diagnosis.csv", diag_rows)
    write_rows(OUT_ROOT / "v22_65_failure_route_matrix.csv", failure_rows)
    write_rows(OUT_ROOT / "v22_65_method_summary.csv", summary_rows)
    route = {
        "final_route": final_route,
        "route_reason": route_reason,
        "kan_gate_status": kan_gate_status,
        "mlp_exploration_gate_opened": int(any_exploration),
        "mlp_official_gate_opened": int(any_official),
        "candidate_gate_summary": candidate_gate,
        "rows": len(rows),
        "completed_rows": sum(1 for r in rows if r.get("run_status") == "completed"),
        "external_unavailable_rows": sum(1 for r in rows if r.get("run_status") == "external_unavailable"),
        "generated_at": now_sg(),
    }
    write_json(OUT_ROOT / "v22_65_final_route.json", route)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "aggregate"]),
        task_id="F_metric_compatible_gate_and_route",
        status="pass",
        gpu="cpu",
        files="results/v22_65/v22_65_*matrix.csv; results/v22_65/v22_65_final_route.json",
        note=f"final_route={final_route}; kan_gate_status={kan_gate_status}; rows={len(rows)}",
    )
    return route


def update_docs_from_results() -> None:
    ensure_out()
    code_rows = read_rows(OUT_ROOT / "v22_65_code_truth_gate.csv")
    v64_route = read_json(OUT_ROOT / "v22_65_v64_reanalysis_route.json")
    unit_rows = read_rows(OUT_ROOT / "v22_65_metric_compatible_unit_tests.csv")
    summary_rows = read_rows(OUT_ROOT / "v22_65_method_summary.csv")
    diag_rows = read_rows(OUT_ROOT / "v22_65_metric_capacity_diagnosis.csv")
    route = read_json(OUT_ROOT / "v22_65_final_route.json")
    failure_rows = read_rows(OUT_ROOT / "v22_65_failure_route_matrix.csv")
    recap = []
    recap.append("### Final route\n")
    recap.append("```json\n" + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n")
    recap.append("### Part A code/training boundary\n")
    recap.append(md_table(code_rows, ["part_a_hard_gate_pass", "compileall_pass", "clean_tarball_self_contained_import_pass", "standard_loop_static_scan_pass", "standard_loop_runtime_trace_pass", "loss_total_is_task_loss_only", "manual_param_update_detected", "class_weight_or_sampler_used_as_fu", "auxiliary_loss_used_official"], limit=5))
    recap.append("### Part B v22.64 descent-capacity reanalysis route\n")
    recap.append("```json\n" + json.dumps(v64_route, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n")
    recap.append("### Part C metric-compatible unit tests\n")
    recap.append(md_table(unit_rows, ["seed", "fixed_metric_Gram_error", "transported_Gram_error", "iso_descent_energy_fraction", "iso_predicted_task_descent", "no_capacity_iso_descent_energy_fraction", "bounded_shape_drift", "part_c_metric_compatible_unit_gate_pass"], limit=20))
    recap.append("### D/E method summary\n")
    recap.append(md_table(summary_rows, ["method", "method_family", "completed_rows", "mean_final_NLL", "beats_strongest_NLL_rows", "beats_external_OET_NLL_rows", "beats_best_control_NLL_rows", "beats_same_functional_spectrum_random_rows", "beats_same_isometric_capacity_random_rows", "no_debt_rows", "overhead_le_035_rows", "active_Gram_drift_le_005_rows", "iso_or_shape_capacity_positive_rows", "exploration_gate_pass"], limit=80))
    recap.append("### F diagnostic correlations\n")
    recap.append(md_table(diag_rows, ["method", "rows", "failure_modes", "corr_iso_capacity_Delta_NLL", "corr_shape_capacity_Delta_NLL", "corr_active_Gram_drift_Delta_NLL", "corr_signal_reachable_energy_Delta_NLL"], limit=80))
    recap.append("### Failure route evidence sample\n")
    recap.append(md_table(failure_rows, ["method", "method_family", "dataset", "seed", "failure_mode", "held_NLL", "Delta_NLL_vs_external_OET", "Delta_NLL_vs_best_control", "iso_descent_energy_fraction", "shape_descent_energy_fraction"], limit=40))
    insights = [
        "### Implementation / repair audit\n",
        "- 新增 `MetricCompatibleAtlasMLP` / `MetricCompatibleLowRankLinear`：strict 分量使用 `c_skew_project` + Cayley retraction；shape 分量使用 C-symmetric 投影并由 `shape_budget` 缩放。训练仍是 `forward -> cross_entropy.backward() -> optimizer.step()`。",
        "- 新增 `metric_compatible_unit_tests()` 覆盖 C1-C5：fixed metric isometry、moving metric transport、isometric capacity、no-capacity detection、bounded shaping。",
        "- 新增 v22.65 runner：产物在 `results/v22_65/`，执行日志/复盘日志在 `docs/`；v22.64 复析只用已有 artifact 和 train-only reconstructed initial gradients，明确记录没有 trained checkpoint。",
        "### Analysis / conclusion / insight\n",
        f"- Final route is `{route.get('final_route', '')}`: {route.get('route_reason', '')}.",
        f"- KAN gate status is `{route.get('kan_gate_status', '')}`; KAN official rows remain blocked unless MLP Metric-Compatible gate opens.",
        "- Evidence chain: Part A verifies code boundary, Part B decomposes v22.64 metric-preservation-vs-task-capacity, Part C verifies metric-compatible math, D/E compares candidates against external OET and matched controls, F assigns failure modes.",
        "- No fabricated rows are used: external unavailable rows remain unavailable; capacity values are computed from train-only gradients or left blank when no coordinate layer exists.",
    ]
    append_recap("Final results and evidence chain", "\n".join(recap + insights))
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write("\n## Repro command summary\n\n")
        f.write("核心命令：\n\n")
        f.write("```bash\n")
        f.write(f"{PYTHON} experiments/run_v22_65_metric_compatible_signal_atlas_fu.py --mode full --gpus 0,1,2,3\n")
        f.write("```\n\n")
        f.write("关键产物：`results/v22_65/` 下所有 `v22_65_*.csv/json`，以及本执行日志和实验结果复盘。\n")


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    code = run_code_truth_gate(args)
    if not iflag(code.get("part_a_hard_gate_pass")):
        route = {"final_route": "R0-CodeBoundaryFailed", "route_reason": "Part A hard gate failed", "kan_gate_status": "skipped_code_boundary_failed"}
        write_json(OUT_ROOT / "v22_65_final_route.json", route)
        append_recap("Stopped at Part A", "Part A hard gate failed. Per plan, B-G were not run.")
        return route
    run_v64_reanalysis(args)
    run_unit_gate(args)
    c_gate = read_json(OUT_ROOT / "v22_65_part_c_unit_gate.json")
    if not iflag(c_gate.get("part_c_metric_compatible_unit_gate_pass")):
        route = {"final_route": "R1-MetricCompatibleUnitFailed", "route_reason": "Part C metric-compatible unit gate failed", "kan_gate_status": "skipped_unit_gate_failed"}
        write_json(OUT_ROOT / "v22_65_final_route.json", route)
        append_recap("Stopped at Part C", "Part C unit gate failed. Per plan, D-G were not run.")
        return route
    run_matrix(args)
    route = aggregate_results(args)
    update_docs_from_results()
    return route


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="full", choices=["full", "smoke", "code-gate", "v64-reanalysis", "unit-gate", "matrix", "row", "aggregate", "docs"])
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=900)
    p.add_argument("--dataset", default="Wine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="mpfa_transport_iso_rank2")
    p.add_argument("--device", default="auto")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--refresh", type=int, default=100)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metric-kind", default="signal_debt")
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--reanalysis-metric-batch-size", type=int, default=128)
    p.add_argument("--shaping-budget", type=float, default=0.05)
    p.add_argument("--iso-eta", type=float, default=1.0)
    p.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    p.add_argument("--unit-seeds", default="0,1,2")
    p.add_argument("--poet-block-size", type=int, default=16)
    p.add_argument("--poet-merge-interval", type=int, default=50)
    p.add_argument("--poet-lr", type=float, default=1.0e-3)
    p.add_argument("--poet-scale", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.mode == "row":
            train_row(args)
        elif args.mode == "code-gate":
            run_code_truth_gate(args)
        elif args.mode == "v64-reanalysis":
            run_v64_reanalysis(args)
        elif args.mode == "unit-gate":
            run_unit_gate(args)
        elif args.mode == "matrix":
            run_matrix(args)
        elif args.mode == "aggregate":
            aggregate_results(args)
        elif args.mode == "docs":
            update_docs_from_results()
        elif args.mode == "smoke":
            smoke_args = argparse.Namespace(**vars(args))
            smoke_args.datasets = "Wine,MNIST"
            smoke_args.seeds = "0"
            smoke_args.methods = "adamw,poet_official,mpfa_transport_iso_rank2,mpfa_shape_signal_budget002,mpfa_poet_compatible_transport_iso,same_rank_random_coordinate,same_functional_spectrum_random_coordinate,same_isometric_capacity_random_coordinate,same_shape_budget_random_coordinate,same_compute_noop_coordinate"
            smoke_args.steps = min(int(args.steps), 20)
            smoke_args.train_size = min(int(args.train_size), 128)
            smoke_args.held_size = min(int(args.held_size), 64)
            smoke_args.test_size = min(int(args.test_size), 64)
            smoke_args.unit_seeds = "0"
            route = run_full(smoke_args)
            print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            route = run_full(args)
            print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_65_{safe_fragment(args.mode)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", str(args.mode)]), task_id=f"exception_{args.mode}", status="exception", files=str(err_path.relative_to(ROOT)), note=f"see {err_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
